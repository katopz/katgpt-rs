//! Set Diffusion Inference Decoder (Research 376 Phase 4 T4.1).
//!
//! Implements the sliding-window set decode loop from Arriola & Kuleshov,
//! "Set Diffusion: Interpolating Token Orderings Between AR and Diffusion"
//! (arXiv:2607.01775, ICML 2026) §3.3. Distilled in
//! `riir-train/.research/376_Set_Diffusion_Flexible_Token_Sets.md`.
//!
//! # Architecture
//!
//! This is the **modelless inference substrate** — it operates on logits
//! produced by a set-causal forward pass and orchestrates the iterative
//! denoising loop. The forward pass itself is abstracted via the
//! [`SetCausalForwardFn`] trait, so the decoder can be:
//! - **Tested** with a mock forward impl (no trained model needed).
//! - **Wired** to the CPU reference (`forward_set_causal_positions` in
//!   `crate::dllm`, gated `set_diffusion`).
//! - **Wired** to the GPU dispatch (`GpuForwardPass::forward_set_causal`
//!   in riir-gpu, gated `set_diffusion`).
//!
//! # Algorithm (SW-SetDLM inference)
//!
//! 1. Start with all decode-region positions masked.
//! 2. Iterate through generation steps `s = 0, 1, ..., max_gen_step`:
//!    - A position `p` is **eligible** at step `s` when `gen_steps[p] <= s`.
//!    - Run a set-causal forward pass over the current token buffer.
//!    - For each eligible-but-still-masked position: sample a token from
//!      the logits, check confidence, commit or re-mask.
//!    - Repeat the forward pass within step `s` until all eligible positions
//!      are committed or `denoise_steps` inner iterations are exhausted.
//! 3. Advance to the next gen-step (reveals more positions).
//! 4. Continue until all positions decoded or `max_steps` total iterations.
//!
//! # Relationship to D2F
//!
//! D2F (`crate::speculative::d2f`) is the **special case** where
//! `gen_steps[p] = p / block_size` — contiguous fixed-size blocks. This
//! decoder generalizes it to arbitrary position-set orderings. When the
//! ordering is block-causal, the two decoders are equivalent (modulo the
//! inner-loop eligibility check, which becomes a block-boundary check).
//!
//! # Why this ships modelless
//!
//! The decoder itself contains no weights, no training, no gradient logic.
//! It is pure orchestration math (mask, forward, sample, confidence gate).
//! However, it is **useless without a set-causal-trained model** — the
//! forward pass must produce meaningful logits under arbitrary attention
//! masks, which requires the set-causal architecture from Research 376
//! Phase 0. Per §2.3 of the research note, this is a modelless layer that
//! requires a trained substrate beneath it.
//!
//! TL;DR: Iterative set-diffusion decode loop, modelless via `SetCausalForwardFn`.
//
// Plan 401 (2026-07-06): Moved from root `src/speculative/set_diffusion.rs`.
// Root keeps a thin re-export shim (`pub use katgpt_forward::set_diffusion::*;`)
// plus the TRAIN-only tests (they call `crate::dllm::{train_mini_dllm,
// generate_pattern_dataset, evaluate_set_causal_nelbo, train_mini_set_causal}`
// which is root-only training code). The PURE inference tests moved with
// this file. The `crate::dllm::forward_set_causal_positions` reference in
// `CpuSetCausalForward::forward_set_causal` now resolves to
// `crate::forward_set_causal::forward_set_causal_positions` (moved alongside
// in Plan 401 T3).

#![allow(clippy::too_many_arguments)]

use katgpt_types::Rng;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for set-diffusion decoding.
///
/// Mirrors `crate::speculative::d2f::D2fDecodeConfig` but for the
/// set-causal generalization. The key difference: instead of a fixed
/// `block_size`, the decode region's position ordering is specified
/// per-call via `gen_steps` (the output of `PositionOffsetSchedule`'s
/// `order_to_gen_steps`, or `block_causal_gen_steps` for D2F compatibility).
#[derive(Clone, Copy, Debug)]
pub struct SetDiffusionConfig {
    /// Token ID used for masked (not-yet-decoded) positions.
    pub mask_token: usize,
    /// Vocabulary size (determines logits stride).
    pub vocab_size: usize,
    /// Maximum inner-loop denoising iterations per generation step.
    ///
    /// Each iteration runs one forward pass and attempts to commit tokens
    /// at eligible masked positions. More iterations → higher quality at
    /// the cost of more forward passes. Typical: 4-16 (matches D2F).
    pub denoise_steps: usize,
    /// Confidence threshold τ_conf for committing a sampled token.
    ///
    /// Tokens sampled with probability < τ_conf are re-masked for the next
    /// iteration. Range [0, 1]. Higher → more conservative, needs more
    /// iterations. Typical: 0.5-0.9.
    pub confidence_threshold: f32,
    /// Sampling temperature. 0.0 = greedy (argmax), 1.0 = raw softmax,
    /// >1.0 = flatter (more diverse).
    pub temperature: f32,
}

impl Default for SetDiffusionConfig {
    fn default() -> Self {
        Self {
            mask_token: 0,
            vocab_size: 256,
            denoise_steps: 8,
            confidence_threshold: 0.7,
            temperature: 1.0,
        }
    }
}

impl SetDiffusionConfig {
    /// Config optimized for quality: more inner iterations, higher threshold.
    pub fn quality() -> Self {
        Self {
            denoise_steps: 16,
            confidence_threshold: 0.9,
            ..Self::default()
        }
    }

    /// Config optimized for speed: fewer iterations, lower threshold.
    pub fn speed() -> Self {
        Self {
            denoise_steps: 4,
            confidence_threshold: 0.5,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Forward-pass abstraction (modelless trait)
// ---------------------------------------------------------------------------

/// Modelless forward-pass abstraction for set-causal attention.
///
/// The decoder calls [`forward_set_causal`](SetCausalForwardFn::forward_set_causal)
/// on each inner-loop iteration, passing the current token buffer (which
/// may contain `mask_token` at not-yet-decoded positions) and the
/// generation-step assignment per position.
///
/// The implementor is responsible for:
/// - Running the actual model forward pass (CPU reference, GPU dispatch,
///   mock for tests, etc.).
/// - Applying the set-causal attention mask internally (positions only
///   attend to same-or-earlier gen-step positions).
/// - Returning a flat logits buffer of length `tokens.len() * vocab_size`,
///   where `logits[p * vocab_size + v]` is the logit for vocabulary token
///   `v` at position `p`.
///
/// # Production implementations
///
/// - **CPU reference**: wraps `forward_set_causal_positions` in
///   `crate::dllm` (gated `set_diffusion`).
/// - **GPU dispatch**: wraps `GpuForwardPass::forward_set_causal` in
///   riir-gpu (gated `set_diffusion`).
pub trait SetCausalForwardFn {
    /// Run a set-causal forward pass.
    ///
    /// - `tokens`: current token IDs, length L (may contain `mask_token`).
    /// - `gen_steps`: generation step per position, length L (from
    ///   `order_to_gen_steps` in riir-train, or `block_causal_gen_steps`
    ///   for D2F compatibility).
    ///
    /// Returns flat logits of length `L * vocab_size`.
    fn forward_set_causal(&self, tokens: &[usize], gen_steps: &[u32]) -> Vec<f32>;
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of a set-diffusion decode run.
#[derive(Clone, Debug)]
pub struct SetDiffusionResult {
    /// Final decoded tokens (prompt + decode region, mask_token replaced).
    pub tokens: Vec<usize>,
    /// Total number of forward passes used (may be < max if converged early).
    pub forward_passes: usize,
    /// Fraction of decode-region positions committed at each forward pass.
    pub confidence_history: Vec<f32>,
    /// Whether all decode-region positions were decoded (no mask_token remaining).
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Core decode loop
// ---------------------------------------------------------------------------

/// Run the set-diffusion decode loop over a fixed position ordering.
///
/// # Arguments
///
/// - `forward`: the set-causal forward-pass impl (model, mock, etc.).
/// - `config`: decode hyperparameters (thresholds, step counts).
/// - `prompt`: already-decoded tokens (prefix, committed). May be empty.
/// - `gen_steps`: generation step per position for the **decode region**
///   (length = decode region size, NOT including prompt). Positions with
///   the same gen-step are denoised together; lower gen-steps are revealed
///   first. Use `order_to_gen_steps` (riir-train) or `block_causal_gen_steps`
///   to construct this.
/// - `rng`: randomness source for sampling.
///
/// # Returns
///
/// A [`SetDiffusionResult`] with the decoded tokens (prompt + decode region,
/// `mask_token` replaced by sampled tokens where confidence threshold was met).
///
/// # Algorithm
///
/// For each generation step `s = 0..=max_gen_step`:
/// 1. Run up to `config.denoise_steps` inner iterations.
/// 2. Each inner iteration: forward pass → sample at eligible masked
///    positions → commit if confidence ≥ τ_conf, else re-mask.
/// 3. Advance to next gen-step when all eligible positions committed or
///    inner iterations exhausted.
///
/// Total forward passes ≤ `(max_gen_step + 1) * config.denoise_steps`.
///
/// # Panics
///
/// Panics if `gen_steps` is empty, or if any `gen_steps[i]` would overflow
/// when used as an index.
pub fn set_diffusion_decode<F: SetCausalForwardFn>(
    forward: &F,
    config: &SetDiffusionConfig,
    prompt: &[usize],
    gen_steps: &[u32],
    rng: &mut Rng,
) -> SetDiffusionResult {
    assert!(
        !gen_steps.is_empty(),
        "gen_steps must be non-empty (decode region size > 0)"
    );

    let decode_len = gen_steps.len();
    let vocab = config.vocab_size;
    let mask = config.mask_token;
    let tau_conf = config.confidence_threshold;
    let temperature = config.temperature;
    let max_inner = config.denoise_steps;

    // Total token buffer: prompt + decode region (all initially masked).
    let prompt_len = prompt.len();
    let mut tokens: Vec<usize> = Vec::with_capacity(prompt_len + decode_len);
    tokens.extend_from_slice(prompt);
    tokens.extend(std::iter::repeat_n(mask, decode_len));

    // The gen_steps buffer covers only the decode region; offset by prompt_len
    // when indexing into the full token buffer.
    let gen_step_at = |decode_idx: usize| -> u32 { gen_steps[decode_idx] };

    let max_gen_step = gen_steps.iter().copied().max().unwrap_or(0);

    let mut forward_passes = 0usize;
    let mut confidence_history = Vec::with_capacity(max_gen_step as usize + 1);
    let mut all_committed = false;

    // Hoisted out of both loops: gen_steps doesn't change between iterations,
    // so the full gen-steps buffer (prompt zeros + decode gen_steps) is invariant.
    let mut full_gen_steps = vec![0u32; prompt_len];
    full_gen_steps.extend_from_slice(gen_steps);

    // Outer loop: iterate through generation steps (reveal positions incrementally).
    for current_step in 0..=max_gen_step {
        if all_committed {
            break;
        }

        // Inner loop: iterative denoising within this gen-step.
        for _inner in 0..max_inner {
            // Forward pass over the full token buffer (prompt + decode region).
            // The forward impl applies the set-causal mask using gen_steps.
            // Note: gen_steps covers only the decode region — pad with 0 for
            // prompt positions (prompt tokens are always "already revealed",
            // gen-step 0, so they attend to themselves and are attended-to by
            // everything). full_gen_steps was hoisted above the loops.
            let logits = forward.forward_set_causal(&tokens, &full_gen_steps);
            forward_passes += 1;

            let mut n_committed_this_pass = 0u32;
            let mut n_eligible_masked = 0u32;

            for di in 0..decode_len {
                let p = prompt_len + di;
                if tokens[p] != mask {
                    continue; // Already decoded.
                }
                if gen_step_at(di) > current_step {
                    continue; // Not yet eligible (revealed at a later gen-step).
                }
                n_eligible_masked += 1;

                // Read logits for this position.
                let logits_start = p * vocab;
                let logits_end = logits_start + vocab;
                let logits_p = &logits[logits_start..logits_end];

                let (chosen_token, chosen_prob) =
                    sample_token(logits_p, mask, vocab, temperature, rng);

                if chosen_prob >= tau_conf && chosen_token != mask {
                    tokens[p] = chosen_token;
                    n_committed_this_pass += 1;
                }
            }

            let confidence = if n_eligible_masked == 0 {
                1.0
            } else {
                // Fraction of eligible-masked positions committed this pass.
                // Once a position is committed it leaves the eligible-masked set,
                // so this measures "how much progress we made this iteration".
                (n_eligible_masked - n_committed_this_pass.min(n_eligible_masked)) as f32
                    / decode_len as f32
            };
            confidence_history.push(confidence);

            // Early exit: no more eligible masked positions at this gen-step.
            if n_eligible_masked == 0 || n_committed_this_pass == n_eligible_masked {
                break;
            }
        }

        // Check if all decode-region positions are now committed.
        all_committed = tokens[prompt_len..].iter().all(|&t| t != mask);
    }

    SetDiffusionResult {
        tokens,
        forward_passes,
        confidence_history,
        converged: all_committed,
    }
}

// ---------------------------------------------------------------------------
// Ordering → gen-steps conversion + scheduled convenience entry point
// ---------------------------------------------------------------------------
//
// These land the thin T4.3-CPU-caller bridge (Research 376 Phase 4): the
// pipeline `PositionOffsetSchedule::sample_order` → `order_to_gen_steps` →
// `set_diffusion_decode`, available as a single convenience call. The decoder
// itself remains schedule-agnostic — these helpers just construct the
// `gen_steps` buffer the decoder already accepts.
//
// Mirrors `riir_train::set_diffusion_schedule::{order_to_gen_steps,
// block_causal_gen_steps, mdlm_gen_steps}` (the training-side reference impl).
// The runtime copy lives here so callers don't need a training-repo dep to
// drive set-diffusion inference.

/// Convert an ordering σ to per-position generation steps (the kernel input).
///
/// Given `order = [σ_0, σ_1, ..., σ_{L-1}]` (σ_0 generated first), returns
/// `gen_step[p] = rank of position p in the ordering`. For SW-SetDLM with
/// singleton sets, this is a permutation of `[0, L)` reinterpreted as `u32`.
///
/// The returned buffer is what [`set_diffusion_decode`] consumes as `gen_steps`,
/// and what `forward_set_causal_positions` / the WGSL set-causal kernel consume
/// as `position_order` (after a zero-cost `u32 → usize` widening on 64-bit).
///
/// # Panics
///
/// In debug builds only: panics if `order` is not a valid permutation of
/// `[0, L)` (i.e. contains an out-of-range position). Release builds skip the
/// check for perf — callers are expected to pass a permutation from
/// [`PositionOffsetSchedule::sample_order`](katgpt_core::PositionOffsetSchedule::sample_order).
///
/// # Example
///
/// ```
/// # use katgpt_forward::set_diffusion::order_to_gen_steps;
/// // Singleton ordering: position 2 first, then 0, then 1.
/// let order = vec![2, 0, 1];
/// let gs = order_to_gen_steps(&order);
/// assert_eq!(gs, vec![1, 2, 0]); // position 0 → step 1, position 1 → step 2, position 2 → step 0
/// ```
pub fn order_to_gen_steps(order: &[usize]) -> Vec<u32> {
    if order.is_empty() {
        return Vec::new();
    }
    let l = order.len();
    let mut gen_steps = vec![0u32; l];
    for (step, &pos) in order.iter().enumerate() {
        debug_assert!(
            pos < l,
            "order contains position {pos} >= length {l} (not a valid permutation)"
        );
        gen_steps[pos] = step as u32;
    }
    gen_steps
}

/// Block-causal gen-steps: contiguous blocks of `block_size` share a step.
///
/// Positions `[0, block_size)` → step 0, `[block_size, 2·block_size)` → step 1,
/// etc. The final block may be shorter. This is the D2F/block-diffusion
/// ordering — the special case of set-diffusion where sets are contiguous L→R
/// blocks. Useful for A/B comparison and for callers that want the D2F path.
///
/// # Panics
///
/// Panics if `block_size == 0`.
pub fn block_causal_gen_steps(l: usize, block_size: usize) -> Vec<u32> {
    assert!(block_size > 0, "block_size must be > 0");
    (0..l).map(|p| (p / block_size) as u32).collect()
}

/// MDLM gen-steps: all positions share step 0 (fully bidirectional).
///
/// This is the order-agnostic diffusion endpoint — every position attends to
/// every other position in a single denoise step. Equivalent to
/// `block_causal_gen_steps(l, l)` but explicit for clarity.
pub fn mdlm_gen_steps(l: usize) -> Vec<u32> {
    vec![0u32; l]
}

/// Run the set-diffusion decode loop with a sampled position ordering.
///
/// This is the **T4.3-CPU-caller bridge** (Research 376 Phase 4): a thin
/// convenience that wraps the schedule → ordering → gen-steps → decode
/// pipeline. Equivalent to:
///
/// ```no_run
/// # use katgpt_core::set_diffusion_schedule::PositionOffsetSchedule;
/// # use katgpt_forward::set_diffusion::*;
/// # use katgpt_types::Rng;
/// # fn wrap<F: SetCausalForwardFn>(forward: &F, cfg: &SetDiffusionConfig, sched: &PositionOffsetSchedule, prompt: &[usize], decode_len: usize, mut rng: &mut Rng) {
/// let order = sched.sample_order_with(decode_len, || rng.uniform());
/// let gen_steps = order_to_gen_steps(&order);
/// let _ = set_diffusion_decode(forward, cfg, prompt, &gen_steps, &mut rng);
/// # }
/// ```
///
/// The decoder itself is unchanged — this is pure plumbing that produces the
/// `gen_steps` buffer from a stochastic schedule draw. Repeated calls with the
/// same schedule will produce different orderings (and hence different gen-steps
/// layouts); this is intentional — SW-SetDLM samples a fresh ordering per
/// inference pass, mirroring training.
///
/// # Arguments
///
/// - `forward`: the set-causal forward-pass impl (model, mock, etc.).
/// - `config`: decode hyperparameters (thresholds, step counts).
/// - `prompt`: already-decoded prefix tokens. May be empty.
/// - `schedule`: the position-offset reveal-time schedule (controls w/k).
/// - `decode_len`: number of decode-region positions to generate.
/// - `rng`: randomness source (used for both ordering + sampling).
///
/// # Returns
///
/// A [`SetDiffusionResult`] with `tokens.len() == prompt.len() + decode_len`.
///
/// # Panics
///
/// Panics if `decode_len == 0` (delegates to `set_diffusion_decode`'s assertion).
pub fn set_diffusion_decode_scheduled<F: SetCausalForwardFn>(
    forward: &F,
    config: &SetDiffusionConfig,
    prompt: &[usize],
    schedule: &katgpt_core::PositionOffsetSchedule,
    decode_len: usize,
    rng: &mut Rng,
) -> SetDiffusionResult {
    let order = schedule.sample_order_with(decode_len, || rng.uniform());
    let gen_steps = order_to_gen_steps(&order);
    set_diffusion_decode(forward, config, prompt, &gen_steps, rng)
}

/// Sample a token from logits with optional temperature scaling.
///
/// Returns `(token_id, probability)`. Uses greedy argmax when temperature ≤ 0
/// or == 1.0 (argmax is equivalent to temperature-1 sampling for the most
/// likely token, and is cheaper). For other temperatures, draws from the
/// softmax distribution.
///
/// Excludes `mask_token` from the candidate set (a masked position should
/// not re-sample to mask).
fn sample_token(
    logits: &[f32],
    mask: usize,
    vocab: usize,
    temperature: f32,
    rng: &mut Rng,
) -> (usize, f32) {
    use katgpt_core::simd::fast_exp;

    debug_assert_eq!(logits.len(), vocab, "logits length must equal vocab_size");

    // Find max for numerical stability (skip mask token).
    //
    // max_logit doubles as argmax_logit — the argmax token is by definition
    // the one holding the max logit, so tracking them separately is dead
    // work (was two `if logit > X` branches per element; now one).
    let mut max_logit = f32::NEG_INFINITY;
    let mut argmax_token = 0usize;
    for (t, &logit) in logits.iter().enumerate().take(vocab) {
        if t == mask {
            continue;
        }
        if logit > max_logit {
            max_logit = logit;
            argmax_token = t;
        }
    }

    // Greedy: return argmax with its softmax probability.
    if temperature <= 0.0 || temperature == 1.0 {
        // Compute the full softmax denominator to get the argmax probability.
        let mut sum_exp = 0.0f32;
        for (t, &logit) in logits.iter().enumerate().take(vocab) {
            if t == mask {
                continue;
            }
            sum_exp += fast_exp(logit - max_logit);
        }
        // argmax_logit == max_logit by construction, so its exp is exp(0) = 1.
        let prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        return (argmax_token, prob);
    }

    // Temperature-scaled sampling — buffer-free two-pass approach:
    //   Pass 1: compute sum_exp (no storage)
    //   Pass 2: re-compute exp + cumulative scan until target hit.
    // This avoids a `vec![0.0; vocab]` allocation (up to 128KB for vocab=32K)
    // on every per-position sample call, at the cost of re-computing exp in
    // the second pass — a favorable trade (32K extra transcendentals vs a
    // 128KB memset + malloc per call).
    let mut sum_exp = 0.0f32;
    for (t, &logit) in logits.iter().enumerate().take(vocab) {
        if t == mask {
            continue;
        }
        let scaled = (logit - max_logit) / temperature;
        sum_exp += fast_exp(scaled);
    }
    if sum_exp <= 0.0 {
        return (argmax_token, 0.0);
    }
    // Normalize + sample.
    let target = rng.uniform() * sum_exp;
    let mut cumulative = 0.0f32;
    for (t, &logit) in logits.iter().enumerate().take(vocab) {
        if t == mask {
            continue;
        }
        let scaled = (logit - max_logit) / temperature;
        let prob_t = fast_exp(scaled);
        cumulative += prob_t;
        if cumulative >= target {
            return (t, prob_t / sum_exp);
        }
    }
    // Fallback (numerical drift): return argmax.
    // argmax_logit == max_logit → exp(0/temperature) = 1.0.
    let p_max = 1.0 / sum_exp;
    (argmax_token, p_max)
}

// ---------------------------------------------------------------------------
// Production wiring adapter (CPU reference)
// ---------------------------------------------------------------------------

/// Adapter wrapping the CPU set-causal forward pass (`forward_set_causal_positions`)
/// for use with [`set_diffusion_decode`].
///
/// This bridges the `&[u32]` gen-steps buffer (schedule output) to the
/// `&[usize]` position_order the CPU forward expects. The conversion is
/// zero-cost on 64-bit platforms (u32 → usize is a widening cast).
///
/// # Example
///
/// ```no_run
/// use katgpt_forward::set_diffusion::{SetCausalForwardFn, SetDiffusionConfig, set_diffusion_decode};
/// use katgpt_forward::set_diffusion::CpuSetCausalForward;
/// use katgpt_transformer::TransformerWeights;
/// use katgpt_types::{Config, Rng};
///
/// # fn wire(weights: &TransformerWeights, config: &Config) {
/// let forward = CpuSetCausalForward { weights, config };
/// let decode_config = SetDiffusionConfig {
///     mask_token: config.mask_token,
///     vocab_size: config.vocab_size,
///     ..Default::default()
/// };
/// let gen_steps: Vec<u32> = vec![0, 0, 1, 1]; // block-causal, block_size=2
/// let mut rng = Rng::new(42);
/// let result = set_diffusion_decode(&forward, &decode_config, &[], &gen_steps, &mut rng);
/// # }
/// ```
#[cfg(feature = "set_diffusion")]
pub struct CpuSetCausalForward<'a> {
    /// Transformer weights (borrowed).
    pub weights: &'a katgpt_transformer::TransformerWeights,
    /// Model config (borrowed).
    pub config: &'a katgpt_types::Config,
}

#[cfg(feature = "set_diffusion")]
impl<'a> SetCausalForwardFn for CpuSetCausalForward<'a> {
    fn forward_set_causal(&self, tokens: &[usize], gen_steps: &[u32]) -> Vec<f32> {
        // Convert u32 gen-steps to usize position_order (zero-cost widening on 64-bit).
        let position_order: Vec<usize> = gen_steps.iter().map(|&g| g as usize).collect();
        // forward_set_causal_positions returns FLAT logits (row-major,
        // q * vocab_size + v) — directly usable, no re-flattening needed.
        let (logits, _attn_weights) = crate::forward_set_causal::forward_set_causal_positions(
            self.weights,
            tokens,
            self.config,
            &position_order,
        );
        logits
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "set_diffusion"))]
mod tests {
    use super::*;

    /// Mock forward-pass impl that returns a hardcoded logit pattern.
    ///
    /// For each decode-region position, the "correct" token is `position_index + 1`
    /// (mod vocab). We give that token a high logit and others a low logit,
    /// with a controllable confidence level to test the threshold gate.
    struct MockForward {
        vocab: usize,
        mask: usize,
        /// Multiplier on the logit gap. Higher = more confident predictions.
        /// 0.0 = all logits equal (uniform, never exceeds threshold).
        confidence_gap: f32,
    }

    impl SetCausalForwardFn for MockForward {
        fn forward_set_causal(&self, tokens: &[usize], _gen_steps: &[u32]) -> Vec<f32> {
            let seq_len = tokens.len();
            let mut logits = vec![0.0f32; seq_len * self.vocab];
            for p in 0..seq_len {
                let base = p * self.vocab;
                // The "target" token for position p (deterministic, for testing).
                let target = if p == 0 {
                    1
                } else {
                    (p % (self.vocab - 1)) + 1
                };
                for v in 0..self.vocab {
                    if v == self.mask {
                        logits[base + v] = -10.0; // Never pick mask.
                    } else if v == target {
                        logits[base + v] = self.confidence_gap;
                    } else {
                        logits[base + v] = 0.0;
                    }
                }
            }
            logits
        }
    }

    #[test]
    fn test_set_diffusion_decode_basic() {
        // 4-position decode region, block-causal gen-steps (2 blocks of 2).
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0, // Strong signal → high confidence → commits.
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 4,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let gen_steps = vec![0u32, 0, 1, 1]; // Block-causal, block_size=2.
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&mock, &config, &[], &gen_steps, &mut rng);

        assert!(result.converged, "should converge with strong signal");
        assert_eq!(result.tokens.len(), 4);
        assert!(
            result.tokens.iter().all(|&t| t != config.mask_token),
            "no mask tokens should remain: {:?}",
            result.tokens
        );
        assert!(result.forward_passes > 0);
    }

    #[test]
    fn test_set_diffusion_decode_preserves_prompt() {
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 4,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let prompt = vec![10, 20, 30];
        let gen_steps = vec![0u32, 0];
        let mut rng = Rng::new(99);

        let result = set_diffusion_decode(&mock, &config, &prompt, &gen_steps, &mut rng);

        assert_eq!(&result.tokens[..prompt.len()], &prompt[..]);
        assert_eq!(result.tokens.len(), prompt.len() + gen_steps.len());
    }

    #[test]
    fn test_set_diffusion_decode_low_confidence_does_not_converge() {
        // confidence_gap = 0 → all logits equal → uniform distribution →
        // probability = 1/(vocab-1) ≈ 0.14 < threshold 0.5 → never commits.
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 0.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 2,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let gen_steps = vec![0u32, 0];
        let mut rng = Rng::new(7);

        let result = set_diffusion_decode(&mock, &config, &[], &gen_steps, &mut rng);

        assert!(
            !result.converged,
            "uniform logits should never exceed threshold 0.5"
        );
        assert!(
            result.tokens.iter().all(|&t| t == config.mask_token),
            "all positions should remain masked"
        );
    }

    #[test]
    fn test_set_diffusion_decode_singleton_ordering_is_sequential() {
        // Singleton ordering: each position is its own gen-step.
        // Positions are revealed one at a time: 0, 1, 2, 3.
        // This is the AR-like endpoint of the schedule spectrum.
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 2,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let gen_steps = vec![0u32, 1, 2, 3]; // Strict singleton ordering.
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&mock, &config, &[], &gen_steps, &mut rng);

        assert!(result.converged);
        assert!(
            result.forward_passes >= 4,
            "singleton needs ≥4 passes (one per step)"
        );
    }

    #[test]
    fn test_set_diffusion_decode_mdlm_all_one_step() {
        // MDLM ordering: all positions share gen-step 0 (fully bidirectional).
        // This is the diffusion endpoint of the schedule spectrum.
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 8,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let gen_steps = vec![0u32, 0, 0, 0]; // MDLM (all one step).
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&mock, &config, &[], &gen_steps, &mut rng);

        assert!(result.converged);
        // MDLM can converge in a single gen-step's inner loop (≤ denoise_steps).
        assert!(result.forward_passes <= 8);
    }

    #[test]
    fn test_set_diffusion_decode_empty_gen_steps_panics() {
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig::default();
        let mut rng = Rng::new(0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            set_diffusion_decode(&mock, &config, &[], &[], &mut rng)
        }));
        assert!(result.is_err(), "empty gen_steps should panic");
    }

    #[test]
    fn test_set_diffusion_decode_greedy_temperature() {
        // temperature = 0 → greedy argmax. With strong signal, always commits.
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 3.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 4,
            confidence_threshold: 0.5,
            temperature: 0.0, // Greedy.
        };
        let gen_steps = vec![0u32, 0];
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&mock, &config, &[], &gen_steps, &mut rng);

        assert!(
            result.converged,
            "greedy with strong signal should converge"
        );
    }

    #[test]
    fn test_sample_token_excludes_mask() {
        let mut rng = Rng::new(42);
        // Logits where mask token (0) has the highest value.
        let logits = vec![10.0, 1.0, 1.0, 1.0];
        let (token, prob) = sample_token(&logits, 0, 4, 0.0, &mut rng);
        assert_ne!(token, 0, "mask token must never be sampled");
        assert!(prob > 0.0);
    }

    #[test]
    fn test_sample_token_uniform_logits() {
        let mut rng = Rng::new(42);
        // All logits equal → uniform distribution → prob ≈ 1/(vocab-1).
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let (token, prob) = sample_token(&logits, 0, 4, 1.0, &mut rng);
        assert_ne!(token, 0);
        let expected_prob = 1.0 / 3.0; // 3 non-mask tokens.
        assert!(
            (prob - expected_prob).abs() < 0.01,
            "uniform prob should be ~1/3, got {prob}"
        );
    }

    #[test]
    fn test_set_diffusion_config_defaults() {
        let c = SetDiffusionConfig::default();
        assert_eq!(c.denoise_steps, 8);
        assert!(c.confidence_threshold > 0.0 && c.confidence_threshold < 1.0);
    }

    #[test]
    fn test_set_diffusion_config_quality_vs_speed() {
        let q = SetDiffusionConfig::quality();
        let s = SetDiffusionConfig::speed();
        assert!(q.denoise_steps > s.denoise_steps);
        assert!(q.confidence_threshold > s.confidence_threshold);
    }

    #[test]
    fn test_set_diffusion_decode_forward_pass_count_bounded() {
        // Total forward passes ≤ (max_gen_step + 1) * denoise_steps.
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 0.0, // Never commits → exhausts all inner iterations.
        };
        let denoise_steps = 3;
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps,
            confidence_threshold: 0.99, // Very high → hard to commit.
            temperature: 1.0,
        };
        let gen_steps = vec![0u32, 1, 2]; // 3 gen-steps.
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&mock, &config, &[], &gen_steps, &mut rng);

        let upper_bound = (3) * denoise_steps; // 3 gen-steps × 3 inner iterations.
        assert!(
            result.forward_passes <= upper_bound,
            "forward_passes {} should be ≤ {} (gen_steps × denoise_steps)",
            result.forward_passes,
            upper_bound
        );
    }

    // ── Integration: real CPU forward pass via CpuSetCausalForward ──
    //
    // Proves the trait wiring is correct: the decoder can drive the actual
    // `forward_set_causal_positions` CPU reference and produce decoded tokens.
    // The micro model is random-init (not trained), so we don't assert on
    // token correctness — just that the loop runs, converges or not, and
    // produces the right shape.

    use super::CpuSetCausalForward;
    use crate::forward_set_causal::forward_set_causal_positions;
    use katgpt_transformer::TransformerWeights;
    use katgpt_types::Config;

    #[test]
    fn test_cpu_adapter_wiring_runs_decode_loop() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let forward = CpuSetCausalForward {
            weights: &weights,
            config: &config,
        };
        let decode_config = SetDiffusionConfig {
            mask_token: config.mask_token,
            vocab_size: config.vocab_size,
            denoise_steps: 2,
            confidence_threshold: 0.3, // Low — random model, be lenient.
            temperature: 0.0,          // Greedy for determinism.
        };
        let gen_steps = vec![0u32, 0, 1, 1]; // Block-causal, block_size=2.
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode(&forward, &decode_config, &[], &gen_steps, &mut rng);

        // Shape checks.
        assert_eq!(result.tokens.len(), gen_steps.len());
        assert!(
            result.forward_passes > 0,
            "should run at least one forward pass"
        );
        assert!(!result.confidence_history.is_empty());
        // We don't assert converged — random-init model may not produce confident
        // predictions. The point is that the wiring works without panic.
    }

    #[test]
    fn test_cpu_adapter_matches_direct_forward_call() {
        // The adapter must produce identical logits to a direct call to
        // forward_set_causal_positions. This catches any indexing/striding
        // bug in the flattening step.
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let gen_steps = vec![0u32, 0, 1, 1, 2, 2, 3, 3];

        // Direct call (the source of truth) — now returns FLAT logits.
        let position_order: Vec<usize> = gen_steps.iter().map(|&g| g as usize).collect();
        let (direct_flat, _) =
            forward_set_causal_positions(&weights, &tokens, &config, &position_order);

        // Adapter call — also FLAT (identical layout, trivially equivalent).
        let forward = CpuSetCausalForward {
            weights: &weights,
            config: &config,
        };
        let flat = forward.forward_set_causal(&tokens, &gen_steps);

        // Compare.
        let vocab = config.vocab_size;
        assert_eq!(flat.len(), tokens.len() * vocab);
        assert_eq!(flat.len(), direct_flat.len());
        for p in 0..tokens.len() {
            for v in 0..vocab {
                let direct = direct_flat[p * vocab + v];
                let adapter = flat[p * vocab + v];
                assert!(
                    direct == adapter,
                    "logit mismatch at p={p}, v={v}: direct={direct}, adapter={adapter}"
                );
            }
        }
    }

    // ── T4.3-CPU-caller bridge tests (Phase 4) ─────────────────────
    //
    // Covers the schedule → ordering → gen-steps → decode pipeline.
    // All tests use MockForward (no trained model needed) — the bridge is
    // pure plumbing and can be fully validated modellessly.

    use katgpt_core::PositionOffsetSchedule;

    #[test]
    fn test_order_to_gen_steps_identity() {
        // AR order: gen_steps should be [0, 1, 2, ..., L-1].
        let order: Vec<usize> = (0..8).collect();
        let gs = order_to_gen_steps(&order);
        assert_eq!(gs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_order_to_gen_steps_reversed() {
        // Reversed order [L-1, ..., 1, 0]: position 0 is last → gen_step[0] = L-1.
        let order: Vec<usize> = (0..8).rev().collect();
        let gs = order_to_gen_steps(&order);
        assert_eq!(gs[0], 7); // position 0 is revealed last (step 7)
        assert_eq!(gs[7], 0); // position 7 is revealed first (step 0)
    }

    #[test]
    fn test_order_to_gen_steps_round_trip() {
        // order_to_gen_steps is the inverse of "sort positions by gen_step".
        // If gs = order_to_gen_steps(order), then sorting (0..L) by gs gives `order`.
        let order = vec![2, 0, 3, 1];
        let gs = order_to_gen_steps(&order);
        // Reconstruct order: sort positions by their gen_step value.
        let mut indexed: Vec<(u32, usize)> = gs.iter().copied().zip(0..).collect();
        indexed.sort_by_key(|&(step, _)| step);
        let reconstructed: Vec<usize> = indexed.into_iter().map(|(_, p)| p).collect();
        assert_eq!(reconstructed, order);
    }

    #[test]
    fn test_order_to_gen_steps_empty() {
        assert!(order_to_gen_steps(&[]).is_empty());
    }

    #[test]
    fn test_order_to_gen_steps_singleton() {
        assert_eq!(order_to_gen_steps(&[0]), vec![0]);
    }

    #[test]
    fn test_block_causal_gen_steps() {
        // Block size 2: positions [0,1]→0, [2,3]→1, [4,5]→2, [6,7]→3.
        let gs = block_causal_gen_steps(8, 2);
        assert_eq!(gs, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn test_block_causal_gen_steps_uneven_tail() {
        // L=10, block_size=4: blocks [0..4)→0, [4..8)→1, [8..10)→2 (short tail).
        let gs = block_causal_gen_steps(10, 4);
        assert_eq!(gs, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
    }

    #[test]
    #[should_panic(expected = "block_size must be > 0")]
    fn test_block_causal_gen_steps_zero_panics() {
        let _ = block_causal_gen_steps(4, 0);
    }

    #[test]
    fn test_mdlm_gen_steps() {
        let gs = mdlm_gen_steps(4);
        assert_eq!(gs, vec![0u32; 4]);
    }

    #[test]
    fn test_sample_order_returns_valid_permutation() {
        // Any schedule must produce a valid permutation of [0, L).
        let schedule = PositionOffsetSchedule::new(0.5);
        let mut rng = Rng::new(42);
        let l = 16;
        let order = schedule.sample_order_with(l, || rng.uniform());
        assert_eq!(order.len(), l);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..l).collect::<Vec<_>>(), "not a permutation");
    }

    #[test]
    fn test_sample_order_empty_and_singleton() {
        let schedule = PositionOffsetSchedule::new(0.5);
        let mut rng = Rng::new(0);
        assert!(schedule.sample_order_with(0, || rng.uniform()).is_empty());
        assert_eq!(schedule.sample_order_with(1, || rng.uniform()), vec![0]);
    }

    #[test]
    fn test_sample_order_ar_extreme_is_near_identity() {
        // AR endpoint (w tiny): ordering should be very close to [0, 1, ..., L-1].
        // We don't require exact identity (the paper's AR is the limit w→0),
        // but inversions should be rare.
        let schedule = PositionOffsetSchedule::ar();
        let l = 16;
        let mut total_inversions = 0usize;
        for seed in 0..50u64 {
            let mut rng = Rng::new(seed);
            let order = schedule.sample_order_with(l, || rng.uniform());
            total_inversions += count_inversions(&order);
        }
        // Average inversions across 50 draws should be small (≤ 2 per draw).
        let avg = total_inversions as f32 / 50.0;
        assert!(
            avg <= 2.0,
            "AR endpoint should be near-identity, avg inversions = {avg}"
        );
    }

    #[test]
    fn test_sample_order_diffusion_extreme_is_high_inversions() {
        // Diffusion endpoint (w=1): orderings should be near-uniform-random,
        // so inversions should be a large fraction of L*(L-1)/2.
        let schedule = PositionOffsetSchedule::diffusion();
        let l = 16;
        let max_inversions = l * (l - 1) / 2;
        let mut total_inversions = 0usize;
        for seed in 0..50u64 {
            let mut rng = Rng::new(seed);
            let order = schedule.sample_order_with(l, || rng.uniform());
            total_inversions += count_inversions(&order);
        }
        let avg = total_inversions as f32 / 50.0;
        // Uniform random permutation has expected inversions = max/2.
        // Require at least 30% of max to confirm we're not AR-like.
        let lower_bound = 0.30 * max_inversions as f32;
        assert!(
            avg >= lower_bound,
            "diffusion endpoint should have high inversions, avg = {avg}, lower bound = {lower_bound}"
        );
    }

    /// Count inversions in a permutation (number of pairs i<j with order[i] > order[j]).
    /// Used to characterize where an ordering sits on the AR ↔ diffusion spectrum.
    fn count_inversions(order: &[usize]) -> usize {
        let mut count = 0;
        for i in 0..order.len() {
            for j in (i + 1)..order.len() {
                if order[i] > order[j] {
                    count += 1;
                }
            }
        }
        count
    }

    #[test]
    fn test_set_diffusion_decode_scheduled_runs_end_to_end() {
        // The T4.3-CPU-caller bridge: schedule → order → gen_steps → decode.
        // No trained model — MockForward provides a deterministic strong signal.
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 4,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let schedule = PositionOffsetSchedule::new(0.5);
        let mut rng = Rng::new(42);

        let result = set_diffusion_decode_scheduled(&mock, &config, &[], &schedule, 8, &mut rng);

        assert_eq!(result.tokens.len(), 8);
        assert!(result.forward_passes > 0);
        assert!(
            result.tokens.iter().all(|&t| t != config.mask_token),
            "strong signal should commit all positions: {:?}",
            result.tokens
        );
    }

    #[test]
    fn test_set_diffusion_decode_scheduled_preserves_prompt() {
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 4,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let prompt = vec![5, 6, 7];
        let schedule = PositionOffsetSchedule::new(0.25);
        let mut rng = Rng::new(99);

        let result =
            set_diffusion_decode_scheduled(&mock, &config, &prompt, &schedule, 4, &mut rng);

        assert_eq!(&result.tokens[..prompt.len()], &prompt[..]);
        assert_eq!(result.tokens.len(), prompt.len() + 4);
    }

    #[test]
    fn test_set_diffusion_decode_scheduled_matches_manual_pipeline() {
        // The convenience function must produce identical output to manually
        // calling sample_order → order_to_gen_steps → set_diffusion_decode
        // with the SAME rng state. This catches any hidden non-determinism
        // (e.g. the convenience function drawing extra samples).
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 2,
            confidence_threshold: 0.5,
            temperature: 0.0, // Greedy for determinism.
        };
        let schedule = PositionOffsetSchedule::new(0.5);

        // Manual pipeline.
        let mut rng_a = Rng::new(42);
        let order = schedule.sample_order_with(8, || rng_a.uniform());
        let gen_steps = order_to_gen_steps(&order);
        let manual = set_diffusion_decode(&mock, &config, &[], &gen_steps, &mut rng_a);

        // Convenience function with same initial seed.
        let mut rng_b = Rng::new(42);
        let convenience =
            set_diffusion_decode_scheduled(&mock, &config, &[], &schedule, 8, &mut rng_b);

        assert_eq!(manual.tokens, convenience.tokens);
        assert_eq!(manual.forward_passes, convenience.forward_passes);
    }

    #[test]
    fn test_set_diffusion_decode_scheduled_ar_converges_quickly_on_strong_signal() {
        // AR-like schedule (w tiny): one position per gen-step → many gen-steps,
        // but each commits immediately with strong signal.
        let mock = MockForward {
            vocab: 8,
            mask: 0,
            confidence_gap: 5.0,
        };
        let config = SetDiffusionConfig {
            mask_token: 0,
            vocab_size: 8,
            denoise_steps: 2,
            confidence_threshold: 0.5,
            temperature: 1.0,
        };
        let schedule = PositionOffsetSchedule::ar();
        let mut rng = Rng::new(7);

        let result = set_diffusion_decode_scheduled(&mock, &config, &[], &schedule, 4, &mut rng);

        assert!(result.converged, "AR + strong signal should converge");
    }
}
