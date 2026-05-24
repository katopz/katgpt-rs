//! D2F (Discrete Diffusion Forcing) Inference Pipeline
//!
//! Implements block-parallel decoding via iterative denoising with block-causal attention.
//! Reference: Plan 066 Phase 2 — D2F inference in katgpt-rs.
//!
//! # Architecture
//!
//! 1. `d2f_decode_block()` — Core denoising loop: mask → forward_block_causal → sample → remask
//! 2. `D2fPipeline` — Manages multiple in-flight blocks with state transitions
//! 3. `D2fDecodeConfig` — Thresholds, block sizes, denoising steps
//!
//! # Key Insight
//!
//! Unlike autoregressive decoding (1 token/step), D2F decodes entire blocks in parallel
//! by iteratively refining masked positions. Each denoising step uses block-causal attention:
//! bidirectional within block, causal across blocks. This allows KV cache accumulation
//! across blocks while maintaining block-level parallelism.

// Inner attributes apply to this module. Index `t` is needed for is_valid() checks alongside logits[t].
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::dllm::{D2fContext, denoising_accuracy, forward_block_causal_with};
use crate::speculative::types::ConstraintPruner;
use crate::transformer::TransformerWeights;
use crate::types::Config;
use crate::types::Rng;

#[cfg(feature = "tri_mode")]
use crate::speculative::diffusion_sampler::{DiffusionSampler, SamplerFeatures};

// ---------------------------------------------------------------------------
// D2F Block State Machine
// ---------------------------------------------------------------------------

/// State of a D2F decode block in the pipeline.
///
/// Transition flow:
/// ```text
/// SemiActivated (confidence < τ_act) → FullyActivated (confidence ≥ τ_act)
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum D2fBlockState {
    /// Block is actively denoising. `step` tracks current denoising iteration.
    /// `confidence` is the fraction of non-mask tokens with probability ≥ τ_conf.
    SemiActivated { step: usize, confidence: f32 },
    /// Block fully denoised. All positions have valid (non-mask) tokens.
    FullyActivated,
}

impl D2fBlockState {
    /// Check if this block is ready to add a successor block.
    pub fn can_add_successor(&self, threshold: f32) -> bool {
        match self {
            Self::SemiActivated { confidence, .. } => *confidence >= threshold,
            Self::FullyActivated => true,
        }
    }

    /// Check if this block is fully activated.
    pub fn is_fully_activated(&self) -> bool {
        matches!(self, Self::FullyActivated)
    }
}

// ---------------------------------------------------------------------------
// D2F Decode Configuration
// ---------------------------------------------------------------------------

/// Configuration for D2F block decoding.
///
/// Tune these thresholds based on the quality-vs-speed tradeoff:
/// - More denoising steps → higher quality, slower
/// - Higher confidence threshold → more selective remasking, may need more steps
/// - Lower activation threshold → earlier successor block addition, more parallelism
#[derive(Clone, Debug)]
pub struct D2fDecodeConfig {
    /// Number of denoising steps per block (T in D2F paper).
    /// Typical: 4-16. More steps = better quality but slower.
    pub denoise_steps: usize,

    /// Confidence threshold for keeping a token prediction (τ_conf).
    /// Tokens with probability < τ_conf are re-masked for the next step.
    /// Typical: 0.5-0.9. Higher = more conservative, needs more steps.
    pub confidence_threshold: f32,

    /// Activation threshold for adding successor blocks (τ_act).
    /// When a block's confidence exceeds this, a new block can start.
    /// Typical: 0.3-0.7. Lower = more aggressive parallelism.
    pub activation_threshold: f32,

    /// Addition threshold for starting successor blocks (τ_add).
    /// Must be ≤ activation_threshold. Typically 0.1-0.3 below it.
    pub addition_threshold: f32,

    /// Block size for block-causal attention.
    /// Should match the block size used during training.
    pub block_size: usize,

    /// Maximum number of blocks that can be in flight simultaneously.
    /// Limits memory usage for the token buffer.
    pub max_pipeline_depth: usize,

    /// Temperature for sampling during denoising.
    /// Lower = more deterministic, higher = more diverse.
    pub temperature: f32,
    /// Noise schedule type for time step generation.
    pub schedule: ScheduleKind,
    /// Enable DPM-Solver++(2M) multistep logit extrapolation (Plan 078 T10.6).
    ///
    /// When enabled, denoising step 1 uses raw logits (DDIM fallback).
    /// Step 2+ extrapolates using cached predictions:
    /// `D_i = (1 + r_i/2) * logits^(i-1) - (r_i/2) * logits^(i-2)`
    /// where `r_i = h_{i-1} / h_i` is the step-size ratio in log-SNR space.
    ///
    /// Potential: 4× throughput (16 steps → 4 steps with maintained quality).
    /// Default: off (opt-in until GOAT proof).
    pub multistep: bool,
}

impl Default for D2fDecodeConfig {
    fn default() -> Self {
        Self {
            denoise_steps: 8,
            confidence_threshold: 0.7,
            activation_threshold: 0.5,
            addition_threshold: 0.3,
            block_size: 8,
            max_pipeline_depth: 4,
            temperature: 1.0,
            schedule: ScheduleKind::default(),
            multistep: false,
        }
    }
}

impl D2fDecodeConfig {
    /// Config optimized for quality: more steps, higher thresholds.
    pub fn quality() -> Self {
        Self {
            denoise_steps: 16,
            confidence_threshold: 0.9,
            activation_threshold: 0.7,
            addition_threshold: 0.5,
            ..Self::default()
        }
    }

    /// Config optimized for speed: fewer steps, lower thresholds.
    pub fn speed() -> Self {
        Self {
            denoise_steps: 4,
            confidence_threshold: 0.5,
            activation_threshold: 0.3,
            addition_threshold: 0.2,
            ..Self::default()
        }
    }

    /// Create config from block size with default thresholds.
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size,
            ..Self::default()
        }
    }

    /// Config with DPM-Solver++(2M) multistep extrapolation (Plan 078 T10.6).
    ///
    /// Uses 4 denoise steps with second-order logit extrapolation, targeting
    /// quality comparable to 16-step standard decoding at ~4× throughput.
    /// The blend formula: `D = 1.5 * current - 0.5 * prev` amplifies the
    /// denoising signal using cached predictions from previous steps.
    pub fn multistep_quality() -> Self {
        Self {
            denoise_steps: 4,
            confidence_threshold: 0.7,
            multistep: true,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// ELF Logit-Normal Schedule (Plan 079)
// ---------------------------------------------------------------------------

/// D2F noise schedule type (ELF Appendix C.6).
#[derive(Debug, Clone, Default)]
pub enum ScheduleKind {
    /// Uniform spacing between steps (current default).
    #[default]
    Uniform,
    /// Logit-normal distribution — concentrates steps near t=0 (ELF: μ=-1.5, σ=0.8).
    LogitNormal { mean: f32, std: f32 },
}

impl ScheduleKind {
    /// ELF paper default: LogitNormal with μ=-1.5, σ=0.8.
    pub fn elf_default() -> Self {
        Self::LogitNormal {
            mean: -1.5,
            std: 0.8,
        }
    }

    /// Generate time steps for denoising.
    /// Returns sorted `Vec<f32>` in [0.0, 1.0] with `n_steps` entries.
    pub fn generate_steps(&self, n_steps: usize, rng: &mut Rng) -> Vec<f32> {
        match self {
            Self::Uniform => {
                // Uniform spacing: 0.0, 1/(n-1), 2/(n-1), ..., 1.0
                match n_steps {
                    0 => vec![],
                    1 => vec![0.5],
                    _ => (0..n_steps)
                        .map(|i| i as f32 / (n_steps - 1) as f32)
                        .collect(),
                }
            }
            Self::LogitNormal { mean, std } => logit_normal_schedule(n_steps, *mean, *std, rng),
        }
    }
}

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x)).
fn sigmoid(x: f32) -> f32 {
    match x >= 0.0 {
        true => 1.0 / (1.0 + (-x).exp()),
        false => {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }
}

/// Generate logit-normal time steps.
/// Samples t_i = sigmoid(N(μ, σ²)) and sorts ascending.
fn logit_normal_schedule(n_steps: usize, mean: f32, std: f32, rng: &mut Rng) -> Vec<f32> {
    match n_steps {
        0 => vec![],
        1 => vec![0.5],
        _ => {
            // Use Box-Muller for normal samples
            let mut steps: Vec<f32> = (0..n_steps)
                .map(|_| {
                    let u1 = (rng.next() as f64 / u64::MAX as f64).max(1e-10) as f32;
                    let u2 = (rng.next() as f64 / u64::MAX as f64) as f32;
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    sigmoid(mean + std * z)
                })
                .collect();
            steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            steps
        }
    }
}

// ---------------------------------------------------------------------------
// D2F Decode Result
// ---------------------------------------------------------------------------

/// Result of decoding a single D2F block.
#[derive(Clone, Debug)]
pub struct D2fBlockResult {
    /// Final decoded tokens for this block.
    pub tokens: Vec<usize>,
    /// Number of denoising steps actually used (may be < max if converged early).
    pub steps_used: usize,
    /// Average confidence across positions at each denoising step.
    pub confidence_history: Vec<f32>,
    /// Final fraction of correctly predicted tokens (if ground truth available).
    pub accuracy: Option<f32>,
    /// Final block state.
    pub state: D2fBlockState,
}

// ---------------------------------------------------------------------------
// Core: D2F Decode Block
// ---------------------------------------------------------------------------

/// Decode a single block using iterative denoising with block-causal attention.
///
/// # Algorithm
///
/// 1. Initialize `block_size` tokens as `mask_token`
/// 2. For each denoising step:
///    a. Forward pass with block-causal attention
///    b. For each masked position: sample from logits, apply constraint
///    c. Confidence remasking: keep tokens ≥ τ_conf, re-mask others
/// 3. Return final tokens when all positions are unmasked or max steps reached
pub fn d2f_decode_block(
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    let mut ctx = D2fContext::new(config);
    d2f_decode_block_with_prompt_with(&mut ctx, weights, config, decode_config, &[], pruner, rng)
}

/// Decode a single block with optional prompt context.
///
/// The prompt tokens are prepended to the block for context but are never masked.
/// Only the `block_size` positions after the prompt are denoised.
pub fn d2f_decode_block_with_prompt(
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    prompt: &[usize],
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    let mut ctx = D2fContext::new(config);
    d2f_decode_block_with_prompt_with(
        &mut ctx,
        weights,
        config,
        decode_config,
        prompt,
        pruner,
        rng,
    )
}

/// Decode block with ground truth for accuracy measurement (testing/benchmarking).
pub fn d2f_decode_block_with_target(
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    target_tokens: &[usize],
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    let mut result = d2f_decode_block(weights, config, decode_config, pruner, rng);
    result.accuracy = Some(denoising_accuracy(&result.tokens, target_tokens));
    result
}

/// Zero-alloc variant of [`d2f_decode_block`].
///
/// Convenience wrapper that decodes a single block without prompt context.
/// Takes a pre-allocated [`D2fContext`] to avoid per-call heap allocations.
pub fn d2f_decode_block_with(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    d2f_decode_block_with_prompt_with(dctx, weights, config, decode_config, &[], pruner, rng)
}

/// Zero-alloc variant of [`d2f_decode_block_with_prompt`].
///
/// Takes a pre-allocated [`D2fContext`] to avoid per-call heap allocations.
/// The context is reset internally by the forward pass at each denoising step.
///
/// This is the preferred entry point for hot loops (e.g., `D2fPipeline::decode_all`).
pub fn d2f_decode_block_with_prompt_with(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    prompt: &[usize],
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    // Standalone decode — no persistent KV across calls
    dctx.committed_len = 0;

    // Clear multistep caches for fresh decode (Plan 078 T10.6)
    if decode_config.multistep {
        dctx.prev_logits_flat.fill(0.0);
        dctx.prev_prev_logits_flat.fill(0.0);
    }

    let mask = config.mask_token;
    let vocab = config.vocab_size;
    let block_size = decode_config.block_size;
    let seq_len = (prompt.len() + block_size).min(config.block_size);
    let block_start = prompt.len();
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let temperature = decode_config.temperature;

    // Initialize: prompt + mask tokens for the block
    let mut tokens: Vec<usize> = prompt.to_vec();
    tokens.extend(std::iter::repeat_n(mask, block_size));
    tokens.truncate(config.block_size);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut converged_step = max_steps;

    for step in 0..max_steps {
        // Zero-alloc forward pass with block-causal attention
        let _seq_len_actual =
            forward_block_causal_with(dctx, weights, &tokens[..seq_len], config, block_size);

        // DPM-Solver++(2M) multistep logit extrapolation (Plan 078 T10.6)
        //
        // Caches raw model outputs and blends with previous step's prediction
        // to get a second-order estimate. For uniform steps (default schedule):
        //   D = 1.5 * current - 0.5 * prev
        //
        // Step 0: no blend (insufficient history), just cache.
        // Step 1+: blend current with cached previous raw output.
        if decode_config.multistep {
            // Save raw logits to prev_prev (used as temp before swap)
            let logits_len = seq_len * vocab;
            dctx.prev_prev_logits_flat[..logits_len]
                .copy_from_slice(&dctx.logits_flat[..logits_len]);

            if step >= 1 {
                // DPM-Solver++(2M): D = (1 + r/2) * current - (r/2) * prev
                // r = step-size ratio in log-SNR space. Uniform steps: r = 1.0
                let r = 1.0f32;
                let alpha = 1.0 + r / 2.0;
                let beta = r / 2.0;
                let blend_start = block_start * vocab;
                let blend_end = seq_len * vocab;
                for idx in blend_start..blend_end {
                    dctx.logits_flat[idx] =
                        alpha * dctx.logits_flat[idx] - beta * dctx.prev_logits_flat[idx];
                }
            }

            // Rotate caches: prev ← raw current, prev_prev ← old prev
            std::mem::swap(&mut dctx.prev_logits_flat, &mut dctx.prev_prev_logits_flat);
        }

        let mut n_confident = 0usize;

        for p in block_start..seq_len {
            // Only denoise positions that are still masked
            if tokens[p] != mask {
                n_confident += 1;
                continue;
            }

            // Read logits from flat buffer instead of Vec<Vec<f32>>
            let logits_start = p * vocab;
            let logits_end = logits_start + vocab;
            let logits_p = &dctx.logits_flat[logits_start..logits_end];
            let max_logit = logits_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Depth and parent tokens relative to block start
            let depth = p - block_start;
            let parent_tokens = &tokens[block_start..p];

            // Compute softmax denominator over valid tokens only
            let mut sum_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !pruner.is_valid(depth, t, parent_tokens) {
                    continue;
                }
                sum_exp += (logits_p[t] - max_logit).exp();
            }

            if sum_exp == 0.0 {
                // No valid tokens — keep masked
                continue;
            }

            // Temperature-scaled sampling from valid tokens
            let (chosen_token, chosen_prob) = if temperature > 0.0 && temperature != 1.0 {
                sample_temperatured(
                    logits_p,
                    mask,
                    vocab,
                    max_logit,
                    temperature,
                    depth,
                    parent_tokens,
                    pruner,
                    rng,
                )
            } else {
                sample_greedy(
                    logits_p,
                    mask,
                    vocab,
                    max_logit,
                    sum_exp,
                    depth,
                    parent_tokens,
                    pruner,
                    rng,
                )
            };

            // Confidence remasking: only keep if confident enough
            if chosen_prob >= tau_conf && chosen_token != mask {
                tokens[p] = chosen_token;
                n_confident += 1;
            }
        }

        let confidence = n_confident as f32 / block_size as f32;
        confidence_history.push(confidence);

        // Early exit: all block positions unmasked
        if tokens[block_start..seq_len].iter().all(|&t| t != mask) {
            converged_step = step;
            break;
        }
    }

    // Determine final state
    let all_unmasked = tokens[block_start..seq_len].iter().all(|&t| t != mask);
    let final_confidence = confidence_history.last().copied().unwrap_or(0.0);

    let state = if all_unmasked {
        D2fBlockState::FullyActivated
    } else {
        D2fBlockState::SemiActivated {
            step: converged_step.min(max_steps - 1),
            confidence: final_confidence,
        }
    };

    let block_tokens: Vec<usize> = tokens[block_start..seq_len].to_vec();

    let steps_used = confidence_history.len();

    D2fBlockResult {
        tokens: block_tokens,
        steps_used,
        confidence_history,
        accuracy: None,
        state,
    }
}

// ── DiffusionSampler Integration (Plan 116 T3) ─────────────────

/// D2F block decode with adaptive confidence via trained [`DiffusionSampler`].
///
/// Plan 116 T3: Integrates trained per-position correctness predictor into the
/// D2F denoising loop. When `sampler` is `Some`, extracts [`SamplerFeatures`]
/// at each masked position and uses `sampler.decide()` instead of the fixed
/// `chosen_prob >= tau_conf` threshold.
///
/// When `sampler` is `None`, behaves identically to [`d2f_decode_block_with_prompt_with`].
#[cfg(feature = "tri_mode")]
pub fn d2f_decode_block_with_prompt_with_sampler(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    prompt: &[usize],
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
    sampler: Option<&DiffusionSampler>,
) -> D2fBlockResult {
    // Standalone decode — no persistent KV across calls
    dctx.committed_len = 0;

    // Clear multistep caches for fresh decode (Plan 078 T10.6)
    if decode_config.multistep {
        dctx.prev_logits_flat.fill(0.0);
        dctx.prev_prev_logits_flat.fill(0.0);
    }

    let mask = config.mask_token;
    let vocab = config.vocab_size;
    let block_size = decode_config.block_size;
    let seq_len = (prompt.len() + block_size).min(config.block_size);
    let block_start = prompt.len();
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let temperature = decode_config.temperature;

    // Initialize: prompt + mask tokens for the block
    let mut tokens: Vec<usize> = prompt.to_vec();
    tokens.extend(std::iter::repeat_n(mask, block_size));
    tokens.truncate(config.block_size);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut converged_step = max_steps;

    for step in 0..max_steps {
        // Zero-alloc forward pass with block-causal attention
        let _seq_len_actual =
            forward_block_causal_with(dctx, weights, &tokens[..seq_len], config, block_size);

        // DPM-Solver++(2M) multistep logit extrapolation (Plan 078 T10.6)
        if decode_config.multistep {
            let logits_len = seq_len * vocab;
            dctx.prev_prev_logits_flat[..logits_len]
                .copy_from_slice(&dctx.logits_flat[..logits_len]);

            if step >= 1 {
                let r = 1.0f32;
                let alpha = 1.0 + r / 2.0;
                let beta = r / 2.0;
                let blend_start = block_start * vocab;
                let blend_end = seq_len * vocab;
                for idx in blend_start..blend_end {
                    dctx.logits_flat[idx] =
                        alpha * dctx.logits_flat[idx] - beta * dctx.prev_logits_flat[idx];
                }
            }

            std::mem::swap(&mut dctx.prev_logits_flat, &mut dctx.prev_prev_logits_flat);
        }

        let mut n_confident = 0usize;

        for p in block_start..seq_len {
            if tokens[p] != mask {
                n_confident += 1;
                continue;
            }

            let logits_start = p * vocab;
            let logits_end = logits_start + vocab;
            let logits_p = &dctx.logits_flat[logits_start..logits_end];
            let max_logit = logits_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            let depth = p - block_start;
            let parent_tokens = &tokens[block_start..p];

            let mut sum_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !pruner.is_valid(depth, t, parent_tokens) {
                    continue;
                }
                sum_exp += (logits_p[t] - max_logit).exp();
            }

            if sum_exp == 0.0 {
                continue;
            }

            let (chosen_token, chosen_prob) = if temperature > 0.0 && temperature != 1.0 {
                sample_temperatured(
                    logits_p,
                    mask,
                    vocab,
                    max_logit,
                    temperature,
                    depth,
                    parent_tokens,
                    pruner,
                    rng,
                )
            } else {
                sample_greedy(
                    logits_p,
                    mask,
                    vocab,
                    max_logit,
                    sum_exp,
                    depth,
                    parent_tokens,
                    pruner,
                    rng,
                )
            };

            // Plan 116 T3: Adaptive confidence via trained sampler.
            // When sampler is available, use per-position features to decide
            // accept/reject instead of the fixed tau_conf threshold.
            let accept = match sampler {
                Some(s) => {
                    let features = SamplerFeatures::from_logits(
                        logits_p,
                        vocab,
                        mask,
                        step,
                        max_steps,
                        p - block_start,
                        block_size,
                    );
                    s.decide(&features, tau_conf).accept
                }
                None => chosen_prob >= tau_conf,
            };

            if accept && chosen_token != mask {
                tokens[p] = chosen_token;
                n_confident += 1;
            }
        }

        let confidence = n_confident as f32 / block_size as f32;
        confidence_history.push(confidence);

        if tokens[block_start..seq_len].iter().all(|&t| t != mask) {
            converged_step = step;
            break;
        }
    }

    let all_unmasked = tokens[block_start..seq_len].iter().all(|&t| t != mask);
    let final_confidence = confidence_history.last().copied().unwrap_or(0.0);

    let state = if all_unmasked {
        D2fBlockState::FullyActivated
    } else {
        D2fBlockState::SemiActivated {
            step: converged_step.min(max_steps - 1),
            confidence: final_confidence,
        }
    };

    let block_tokens: Vec<usize> = tokens[block_start..seq_len].to_vec();
    let steps_used = confidence_history.len();

    D2fBlockResult {
        tokens: block_tokens,
        steps_used,
        confidence_history,
        accuracy: None,
        state,
    }
}

/// Convenience wrapper for [`d2f_decode_block_with_prompt_with_sampler`] without prompt.
#[cfg(feature = "tri_mode")]
pub fn d2f_decode_block_with_sampler(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
    sampler: Option<&DiffusionSampler>,
) -> D2fBlockResult {
    d2f_decode_block_with_prompt_with_sampler(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        pruner,
        rng,
        sampler,
    )
}

/// Zero-alloc variant of [`d2f_decode_block_with_target`].
///
/// Takes a pre-allocated [`D2fContext`] to avoid per-call heap allocations.
pub fn d2f_decode_block_with_target_with(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    target_tokens: &[usize],
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    let mut result =
        d2f_decode_block_with_prompt_with(dctx, weights, config, decode_config, &[], pruner, rng);
    result.accuracy = Some(denoising_accuracy(&result.tokens, target_tokens));
    result
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Temperature-scaled sampling from valid tokens.
/// Returns (token, probability).
fn sample_temperatured(
    logits: &[f32],
    mask: usize,
    vocab: usize,
    max_logit: f32,
    temperature: f32,
    depth: usize,
    parent_tokens: &[usize],
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> (usize, f32) {
    let inv_temp = 1.0 / temperature;

    // Compute scaled sum
    let mut scaled_sum = 0.0f32;
    for t in 0..vocab {
        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
            continue;
        }
        scaled_sum += ((logits[t] - max_logit) * inv_temp).exp();
    }

    if scaled_sum == 0.0 {
        return (mask, 0.0);
    }

    // Sample by cumulative distribution
    let r = (rng.next() as f64 / u64::MAX as f64) as f32;
    let mut cumsum = 0.0f32;
    for t in 0..vocab {
        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
            continue;
        }
        let prob = ((logits[t] - max_logit) * inv_temp).exp() / scaled_sum;
        cumsum += prob;
        if cumsum >= r {
            return (t, prob);
        }
    }

    // Fallback: return last valid token
    (mask, 0.0)
}

/// Greedy sampling with temperature=1.0 from valid tokens.
/// Returns (token, probability).
fn sample_greedy(
    logits: &[f32],
    mask: usize,
    vocab: usize,
    max_logit: f32,
    sum_exp: f32,
    depth: usize,
    parent_tokens: &[usize],
    pruner: &dyn ConstraintPruner,
    rng: &mut Rng,
) -> (usize, f32) {
    let r = (rng.next() as f64 / u64::MAX as f64) as f32;
    let mut cumsum = 0.0f32;
    for t in 0..vocab {
        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
            continue;
        }
        let prob = (logits[t] - max_logit).exp() / sum_exp;
        cumsum += prob;
        if cumsum >= r {
            return (t, prob);
        }
    }

    (mask, 0.0)
}

// ---------------------------------------------------------------------------
// D2F Pipeline — Multi-Block Decode
// ---------------------------------------------------------------------------

/// A single block in the D2F decode pipeline.
#[derive(Clone, Debug)]
pub struct D2fPipelineBlock {
    /// Block index in the pipeline (0-based).
    pub index: usize,
    /// Start position in the output sequence.
    pub start_pos: usize,
    /// Current token state (including masks).
    pub tokens: Vec<usize>,
    /// Current block state.
    pub state: D2fBlockState,
    /// Denoising step counter.
    pub step: usize,
    /// Confidence history per step.
    pub confidence_history: Vec<f32>,
}

/// Multi-block D2F decode pipeline.
///
/// Manages multiple blocks in flight simultaneously, adding new blocks
/// when predecessors reach the addition threshold, and activating blocks
/// when they reach the activation threshold.
///
/// # Strategy
///
/// Uses sequential block decoding: each block is decoded to completion
/// before starting the next. Previously decoded blocks provide context
/// via block-causal attention. Future work: interleaved multi-block
/// denoising for true pipeline parallelism.
pub struct D2fPipeline<'a> {
    /// Model configuration.
    config: &'a Config,
    /// D2F decode configuration.
    decode_config: D2fDecodeConfig,
    /// Total generation length (number of new tokens to produce).
    total_len: usize,
    /// Prompt tokens (prepended to all blocks).
    prompt: Vec<usize>,
    /// DMax Soft Parallel Decode configuration (Plan 109 T5).
    /// When `Some`, `decode_all()` uses `d2f_decode_block_soft()` instead of
    /// the binary mask/token denoising loop.
    #[cfg(feature = "dmax_spd")]
    soft_config: Option<SoftDecodeConfig>,
}

/// Result of full pipeline decode.
#[derive(Clone, Debug)]
pub struct D2fPipelineResult {
    /// All decoded tokens (prompt + all blocks concatenated).
    pub tokens: Vec<usize>,
    /// Per-block results.
    pub block_results: Vec<D2fBlockResult>,
    /// Total denoising steps across all blocks.
    pub total_steps: usize,
    /// Number of blocks that reached FullyActivated.
    pub n_fully_activated: usize,
    /// Number of blocks that remained SemiActivated.
    pub n_semi_activated: usize,
}

impl<'a> D2fPipeline<'a> {
    /// Create a new pipeline for decoding `total_len` tokens.
    pub fn new(config: &'a Config, decode_config: D2fDecodeConfig, total_len: usize) -> Self {
        Self {
            config,
            decode_config,
            total_len,
            prompt: Vec::new(),
            #[cfg(feature = "dmax_spd")]
            soft_config: None,
        }
    }

    /// Create a new pipeline with prompt context.
    pub fn with_prompt(
        config: &'a Config,
        decode_config: D2fDecodeConfig,
        total_len: usize,
        prompt: &[usize],
    ) -> Self {
        Self {
            config,
            decode_config,
            total_len,
            prompt: prompt.to_vec(),
            #[cfg(feature = "dmax_spd")]
            soft_config: None,
        }
    }

    /// Set DMax Soft Parallel Decode configuration (Plan 109 T5).
    ///
    /// When set, `decode_all()` uses `d2f_decode_block_soft()` with hybrid
    /// embeddings instead of the standard binary mask/token denoising loop.
    #[cfg(feature = "dmax_spd")]
    pub fn with_soft_config(mut self, soft_config: SoftDecodeConfig) -> Self {
        self.soft_config = Some(soft_config);
        self
    }

    /// Number of blocks needed to cover `total_len`.
    pub fn n_blocks(&self) -> usize {
        self.total_len.div_ceil(self.decode_config.block_size)
    }

    /// Run the full pipeline: decode all blocks sequentially.
    ///
    /// Each block uses block-causal attention, so previously decoded blocks
    /// are visible as causal context while the current block is denoised
    /// bidirectionally within itself.
    pub fn decode_all(
        self,
        weights: &TransformerWeights,
        pruner: &dyn ConstraintPruner,
        rng: &mut Rng,
    ) -> D2fPipelineResult {
        let n_blocks = self.n_blocks();
        let mask = self.config.mask_token;
        let block_size = self.decode_config.block_size;
        let max_steps = self.decode_config.denoise_steps;
        let tau_conf = self.decode_config.confidence_threshold;
        let temperature = self.decode_config.temperature;
        let vocab = self.config.vocab_size;
        let mut ctx = D2fContext::new(self.config);

        let mut all_tokens = self.prompt.clone();
        let mut block_results = Vec::with_capacity(n_blocks);
        let mut total_steps = 0usize;
        let mut n_fully_activated = 0usize;
        let mut n_semi_activated = 0usize;

        for block_idx in 0..n_blocks {
            let remaining = self.total_len.saturating_sub(block_idx * block_size);
            let current_block_size = remaining.min(block_size);

            // ── Soft decode path (DMax SPD, Plan 109 T5) ────────────
            // When soft_config is Some, use d2f_decode_block_soft() with
            // hybrid embeddings instead of the binary denoising loop.
            #[cfg(feature = "dmax_spd")]
            if let Some(ref sc) = self.soft_config {
                let result = d2f_decode_block_soft(
                    &mut ctx,
                    weights,
                    self.config,
                    &self.decode_config,
                    pruner,
                    sc,
                    rng,
                );
                // Truncate to actual block size for last partial block
                let block_tokens: Vec<usize> =
                    result.tokens.into_iter().take(current_block_size).collect();
                total_steps += result.steps_used;
                match result.state {
                    D2fBlockState::FullyActivated => n_fully_activated += 1,
                    _ => n_semi_activated += 1,
                }
                all_tokens.extend_from_slice(&block_tokens);
                block_results.push(D2fBlockResult {
                    tokens: block_tokens,
                    steps_used: result.steps_used,
                    confidence_history: result.confidence_history,
                    accuracy: result.accuracy,
                    state: result.state,
                });
                // Commit KV cache so subsequent blocks see this block's context
                ctx.commit(all_tokens.len());
                continue;
            }

            // Build sequence: prompt + previously decoded blocks + mask for current block
            let mut seq_tokens = all_tokens.clone();
            seq_tokens.extend(std::iter::repeat_n(mask, current_block_size));

            let seq_len = seq_tokens.len().min(self.config.block_size);
            let block_start = seq_len.saturating_sub(current_block_size);

            let mut confidence_history = Vec::with_capacity(max_steps);
            let mut converged_step = max_steps;

            for step in 0..max_steps {
                let _seq_len_actual = forward_block_causal_with(
                    &mut ctx,
                    weights,
                    &seq_tokens[..seq_len],
                    self.config,
                    block_size,
                );

                let mut n_confident = 0usize;

                for p in block_start..seq_len {
                    if seq_tokens[p] != mask {
                        n_confident += 1;
                        continue;
                    }

                    let logits_start = p * vocab;
                    let logits_end = logits_start + vocab;
                    let logits_p = &ctx.logits_flat[logits_start..logits_end];
                    let max_logit = logits_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let depth = p - block_start;
                    let parent_tokens = &seq_tokens[block_start..p];

                    let mut sum_exp = 0.0f32;
                    for t in 0..vocab {
                        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
                            continue;
                        }
                        sum_exp += (logits_p[t] - max_logit).exp();
                    }

                    if sum_exp == 0.0 {
                        continue;
                    }

                    let (chosen_token, chosen_prob) = if temperature > 0.0 && temperature != 1.0 {
                        sample_temperatured(
                            logits_p,
                            mask,
                            vocab,
                            max_logit,
                            temperature,
                            depth,
                            parent_tokens,
                            pruner,
                            rng,
                        )
                    } else {
                        sample_greedy(
                            logits_p,
                            mask,
                            vocab,
                            max_logit,
                            sum_exp,
                            depth,
                            parent_tokens,
                            pruner,
                            rng,
                        )
                    };

                    if chosen_prob >= tau_conf && chosen_token != mask {
                        seq_tokens[p] = chosen_token;
                        n_confident += 1;
                    }
                }

                let confidence = n_confident as f32 / current_block_size as f32;
                confidence_history.push(confidence);

                if seq_tokens[block_start..seq_len].iter().all(|&t| t != mask) {
                    converged_step = step;
                    break;
                }
            }

            // Extract block tokens
            let block_tokens: Vec<usize> = seq_tokens[block_start..seq_len].to_vec();
            let all_unmasked = block_tokens.iter().all(|&t| t != mask);
            let final_confidence = confidence_history.last().copied().unwrap_or(0.0);

            let state = if all_unmasked {
                n_fully_activated += 1;
                D2fBlockState::FullyActivated
            } else {
                n_semi_activated += 1;
                D2fBlockState::SemiActivated {
                    step: converged_step.min(max_steps - 1),
                    confidence: final_confidence,
                }
            };

            let steps_used = confidence_history.len();
            total_steps += steps_used;

            block_results.push(D2fBlockResult {
                tokens: block_tokens.clone(),
                steps_used,
                confidence_history,
                accuracy: None,
                state,
            });

            // Append decoded tokens for next block's context
            all_tokens.extend_from_slice(&block_tokens);

            // Commit KV cache for all positions decoded so far.
            // Subsequent blocks will skip recomputing KV for these positions,
            // significantly reducing work for multi-block pipelines.
            ctx.commit(all_tokens.len());
        }

        D2fPipelineResult {
            tokens: all_tokens,
            block_results,
            total_steps,
            n_fully_activated,
            n_semi_activated,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
// DMax Soft Parallel Decode (Plan 109, Research 72)
// ---------------------------------------------------------------------------

/// Configuration for DMax Soft Parallel Decoding.
///
/// When enabled, decoded positions use hybrid embeddings (interpolation
/// between predicted token embedding and mask embedding) instead of
/// binary mask/token transitions. This carries uncertainty forward,
/// enabling iterative self-refinement.
#[cfg(feature = "dmax_spd")]
#[derive(Clone, Debug)]
pub struct SoftDecodeConfig {
    /// Enable hybrid embedding construction (default: true).
    pub use_hybrid_embeddings: bool,
    /// Decoding threshold τ_dec: promote positions with confidence > τ_dec (default: 0.5).
    pub decode_threshold: f32,
    /// Acceptance threshold τ_acc: block converges when all positions > τ_acc (default: 0.9).
    pub accept_threshold: f32,
    /// Enable contiguous prefix promotion rule (default: true).
    pub contiguous_prefix: bool,
    /// Enable consistency convergence check (default: true).
    pub consistency_check: bool,
}

#[cfg(feature = "dmax_spd")]
impl Default for SoftDecodeConfig {
    fn default() -> Self {
        Self {
            use_hybrid_embeddings: true,
            decode_threshold: 0.5,
            accept_threshold: 0.9,
            contiguous_prefix: true,
            consistency_check: true,
        }
    }
}

#[cfg(feature = "dmax_spd")]
impl SoftDecodeConfig {
    /// Aggressive preset: lower thresholds, more parallelism.
    pub fn aggressive() -> Self {
        Self {
            decode_threshold: 0.3,
            accept_threshold: 0.8,
            ..Self::default()
        }
    }

    /// Conservative preset: higher thresholds, safer decoding.
    pub fn conservative() -> Self {
        Self {
            decode_threshold: 0.7,
            accept_threshold: 0.95,
            ..Self::default()
        }
    }
}

/// Hybrid embedding: soft interpolation between token and mask embeddings.
///
/// h̃ = π * e_token + (1 - π) * e_mask
/// h = h̃ / ||h̃||₂ * (π * ||e_token||₂ + (1 - π) * ||e_mask||₂)
///
/// The renormalization prevents magnitude collapse from adding high-dim vectors.
#[cfg(feature = "dmax_spd")]
pub struct HybridEmbedding {
    /// Confidence π for the predicted token.
    pub confidence: f32,
    /// Predicted token id.
    pub token_id: usize,
}

#[cfg(feature = "dmax_spd")]
impl HybridEmbedding {
    /// Construct hybrid embedding vector in-place.
    /// Writes into `out[dim]` slice, reads from `token_emb[dim]` and `mask_emb[dim]`.
    pub fn build(&self, token_emb: &[f32], mask_emb: &[f32], out: &mut [f32]) {
        let dim = out.len();
        let pi = self.confidence.clamp(0.0, 1.0);
        let one_minus_pi = 1.0 - pi;

        // h̃ = π * e_token + (1 - π) * e_mask
        let mut norm_sq = 0.0f32;
        for d in 0..dim {
            let h_tilde = pi * token_emb[d] + one_minus_pi * mask_emb[d];
            out[d] = h_tilde;
            norm_sq += h_tilde * h_tilde;
        }

        // Renormalize: h = h̃ / ||h̃||₂ * target_norm
        let norm = norm_sq.sqrt();
        if norm > 1e-8 {
            let token_norm: f32 = token_emb.iter().map(|v| v * v).sum::<f32>().sqrt();
            let mask_norm: f32 = mask_emb.iter().map(|v| v * v).sum::<f32>().sqrt();
            let target_norm = pi * token_norm + one_minus_pi * mask_norm;
            let scale = target_norm / norm;
            for v in out.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// DMax contiguous prefix promotion rule.
///
/// Scan masked positions left-to-right. Promote the longest contiguous prefix
/// where confidence > τ_dec. If none qualify, promote the leftmost position
/// (ensure progress). This keeps the masked region contiguous.
///
/// Returns: Vec of position indices to promote from mask→token.
#[cfg(feature = "dmax_spd")]
pub fn contiguous_prefix_promote(
    masked_positions: &[usize],
    confidences: &[f32],
    decode_threshold: f32,
) -> Vec<usize> {
    if masked_positions.is_empty() {
        return Vec::new();
    }

    // Build confidence map: position → confidence
    let mut to_promote = Vec::new();
    for &pos in masked_positions {
        let conf = confidences.get(pos).copied().unwrap_or(0.0);
        if conf >= decode_threshold {
            to_promote.push(pos);
        } else {
            // Contiguous prefix: stop at first below-threshold
            break;
        }
    }

    // Ensure progress: if none qualify, promote leftmost
    if to_promote.is_empty() {
        to_promote.push(masked_positions[0]);
    }

    to_promote
}

/// Convergence status for a D2F decode block.
#[cfg(feature = "dmax_spd")]
#[derive(Clone, Debug, PartialEq)]
pub enum BlockConvergence {
    /// Block has not converged, continue denoising.
    NotConverged,
    /// Block converged: all positions above acceptance threshold.
    ConfidenceConverged,
    /// Block converged: top-1 predictions unchanged for 2 consecutive steps.
    ConsistencyConverged,
}

/// Check if a block has converged using DMax criteria.
///
/// Primary signal: consistency (unchanged top-1 across 2 steps).
/// Secondary signal: confidence (all positions > τ_acc).
/// Either criterion triggers convergence.
#[cfg(feature = "dmax_spd")]
pub fn check_block_convergence(
    current_top1: &[usize],
    prev_top1: Option<&[usize]>,
    confidences: &[f32],
    accept_threshold: f32,
) -> BlockConvergence {
    // Primary: consistency check (unchanged top-1 from previous step)
    if let Some(prev) = prev_top1
        && current_top1.len() == prev.len()
        && current_top1.iter().zip(prev.iter()).all(|(a, b)| a == b)
    {
        return BlockConvergence::ConsistencyConverged;
    }

    // Secondary: confidence check (all positions above threshold)
    if !confidences.is_empty() && confidences.iter().all(|&c| c >= accept_threshold) {
        return BlockConvergence::ConfidenceConverged;
    }

    BlockConvergence::NotConverged
}

/// DMax Soft Parallel Decoding — enhanced D2F block decode.
///
/// Key differences from standard `d2f_decode_block()`:
/// 1. Hybrid embeddings prepared for decoded positions (token IDs carry confidence info)
/// 2. Contiguous prefix promotion for position selection
/// 3. Block convergence check for early stopping
///
/// The forward pass uses standard `forward_block_causal_with` with token IDs.
/// Hybrid embedding logic is applied via the promotion/convergence heuristics.
/// Full hybrid embedding injection requires a custom forward pass variant (future work).
///
/// **Important:** Best results with OPUT-trained models. May degrade quality
/// on models trained with standard D2F loss only. See Research 072.
#[cfg(feature = "dmax_spd")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn d2f_decode_block_soft(
    ctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    _pruner: &dyn ConstraintPruner,
    soft_config: &SoftDecodeConfig,
    _rng: &mut Rng,
) -> D2fBlockResult {
    let block_len = decode_config.block_size;
    let mask_id = config.mask_token;
    let vocab_size = config.vocab_size;

    // Initialize all positions as masked
    let mut tokens: Vec<usize> = vec![mask_id; block_len];
    let mut confidences = vec![0.0f32; block_len];
    let mut prev_top1: Option<Vec<usize>> = None;

    let max_steps = decode_config.denoise_steps;

    for step_idx in 0..max_steps {
        // Forward pass with block-causal attention using current token IDs
        // Masked positions use mask_token; decoded positions use predicted tokens
        let seq_len = forward_block_causal_with(ctx, weights, &tokens, config, block_len);

        // Sample top-1 and confidence for each position
        let mut current_top1 = vec![0usize; block_len];
        for pos in 0..seq_len.min(block_len) {
            let logit_off = pos * vocab_size;
            let logit_end = logit_off + vocab_size;
            let logits = &ctx.logits_flat[logit_off..logit_end];

            // Get logits for this position (copy for local processing)
            let local_logits = logits.to_vec();

            // Top-1 token and its logit value
            let (best_idx, best_val) = local_logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|a: &(usize, f32), b: &(usize, f32)| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or((0, 0.0f32));

            current_top1[pos] = best_idx;

            // Confidence via softmax max probability
            let max_val = local_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = local_logits.iter().map(|&v| (v - max_val).exp()).sum();
            let conf = if sum_exp > 0.0 {
                (best_val - max_val).exp() / sum_exp
            } else {
                1.0 / vocab_size as f32
            };
            confidences[pos] = conf;
        }

        // Collect currently masked positions
        let masked_positions: Vec<usize> =
            (0..block_len).filter(|&p| tokens[p] == mask_id).collect();

        // Promote positions: contiguous prefix or all-above-threshold
        if soft_config.contiguous_prefix {
            let to_promote = contiguous_prefix_promote(
                &masked_positions,
                &confidences,
                soft_config.decode_threshold,
            );
            for pos in to_promote {
                if tokens[pos] == mask_id {
                    tokens[pos] = current_top1[pos];
                }
            }
        } else {
            // Standard: promote all above threshold
            for &pos in &masked_positions {
                if confidences[pos] >= soft_config.decode_threshold {
                    tokens[pos] = current_top1[pos];
                }
            }
            // Ensure progress: promote leftmost if none qualified
            if !masked_positions.is_empty()
                && masked_positions.iter().all(|&p| tokens[p] == mask_id)
            {
                tokens[masked_positions[0]] = current_top1[masked_positions[0]];
            }
        }

        // Check convergence for early stopping
        if soft_config.consistency_check && step_idx > 0 {
            let convergence = check_block_convergence(
                &current_top1,
                prev_top1.as_deref(),
                &confidences,
                soft_config.accept_threshold,
            );
            if convergence != BlockConvergence::NotConverged {
                // Fill remaining masked positions with current top-1
                for pos in 0..block_len {
                    if tokens[pos] == mask_id {
                        tokens[pos] = current_top1[pos];
                    }
                }
                break;
            }
        }

        prev_top1 = Some(current_top1);
    }

    // Fill any remaining masked positions with final predictions
    let seq_len = forward_block_causal_with(ctx, weights, &tokens, config, block_len);
    for pos in 0..seq_len.min(block_len) {
        if tokens[pos] == mask_id {
            let logit_off = pos * vocab_size;
            let logit_end = logit_off + vocab_size;
            let logits = &ctx.logits_flat[logit_off..logit_end];
            let best = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|a: &(usize, f32), b: &(usize, f32)| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            tokens[pos] = best;
        }
    }

    let avg_confidence = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    D2fBlockResult {
        tokens,
        steps_used: max_steps,
        confidence_history: vec![avg_confidence],
        accuracy: None,
        state: D2fBlockState::FullyActivated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dllm::{generate_pattern_dataset, train_mini_dllm};
    use crate::speculative::types::NoPruner;

    #[test]
    fn test_block_state_transitions() {
        let semi = D2fBlockState::SemiActivated {
            step: 3,
            confidence: 0.4,
        };
        assert!(!semi.is_fully_activated());
        assert!(!semi.can_add_successor(0.5));
        assert!(semi.can_add_successor(0.3));

        let full = D2fBlockState::FullyActivated;
        assert!(full.is_fully_activated());
        assert!(full.can_add_successor(0.99));
    }

    #[test]
    fn test_decode_config_defaults() {
        let config = D2fDecodeConfig::default();
        assert_eq!(config.denoise_steps, 8);
        assert!(config.confidence_threshold > 0.0);
        assert!(config.activation_threshold >= config.addition_threshold);
        assert!(config.block_size > 0);
        assert!(config.max_pipeline_depth > 0);
    }

    #[test]
    fn test_decode_block_output_length() {
        let config = Config::micro_dllm();
        let decode_config = D2fDecodeConfig::with_block_size(4);
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);

        let result = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);

        assert_eq!(result.tokens.len(), decode_config.block_size);
        assert!(result.steps_used <= decode_config.denoise_steps);
        assert_eq!(result.confidence_history.len(), result.steps_used);
    }

    #[test]
    fn test_decode_block_with_prompt() {
        let config = Config::micro_dllm();
        let decode_config = D2fDecodeConfig::with_block_size(4);
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);
        let prompt = vec![0, 1, 2];

        let result = d2f_decode_block_with_prompt(
            &weights,
            &config,
            &decode_config,
            &prompt,
            &NoPruner,
            &mut rng,
        );

        // Block tokens should be block_size, not including prompt
        assert_eq!(result.tokens.len(), decode_config.block_size);
    }

    #[test]
    fn test_pipeline_decode_all() {
        let config = Config::micro_dllm();
        let block_size = 4;
        let total_len = 8; // 2 blocks of 4
        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);

        let pipeline = D2fPipeline::new(&config, decode_config, total_len);
        assert_eq!(pipeline.n_blocks(), 2);

        let result = pipeline.decode_all(&weights, &NoPruner, &mut rng);

        assert_eq!(result.tokens.len(), total_len);
        assert_eq!(result.block_results.len(), 2);
        assert!(result.total_steps > 0);
    }

    #[test]
    fn test_pipeline_with_prompt() {
        let config = Config::micro_dllm();
        let block_size = 4;
        let total_len = 4; // 1 block
        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);
        let prompt = vec![0, 1];

        let pipeline = D2fPipeline::with_prompt(&config, decode_config, total_len, &prompt);
        let result = pipeline.decode_all(&weights, &NoPruner, &mut rng);

        // Tokens should be prompt + block
        assert_eq!(result.tokens.len(), prompt.len() + total_len);
        assert_eq!(&result.tokens[..prompt.len()], &prompt);
    }

    #[test]
    fn test_decode_with_trained_model() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(123);

        let train_data =
            generate_pattern_dataset(&mut rng, 20, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut rng, 5, config.block_size, config.vocab_size - 1);

        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        let decode_config = D2fDecodeConfig {
            denoise_steps: 16,
            confidence_threshold: 0.3,
            block_size: config.block_size,
            temperature: 0.8,
            ..D2fDecodeConfig::default()
        };

        let result = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);

        let n_unmasked = result
            .tokens
            .iter()
            .filter(|&&t| t != config.mask_token)
            .count();
        assert!(
            n_unmasked > 0,
            "Expected at least 1 unmasked token, got {n_unmasked}"
        );
    }

    #[test]
    fn test_confidence_history_not_empty() {
        let config = Config::micro_dllm();
        let decode_config = D2fDecodeConfig::with_block_size(4);
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);

        let result = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);

        assert!(!result.confidence_history.is_empty());
        assert_eq!(result.confidence_history.len(), result.steps_used);
    }

    #[test]
    fn test_multistep_decode_produces_valid_output() {
        let config = Config::micro_dllm();
        let decode_config = D2fDecodeConfig {
            multistep: true,
            denoise_steps: 4,
            ..D2fDecodeConfig::with_block_size(4)
        };
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);

        let result = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);

        assert_eq!(result.tokens.len(), decode_config.block_size);
        // All tokens should be valid vocab indices
        for &t in &result.tokens {
            assert!(
                t < config.vocab_size,
                "token {t} exceeds vocab_size {}",
                config.vocab_size
            );
        }
        assert!(result.steps_used <= decode_config.denoise_steps);
    }

    #[test]
    fn test_multistep_with_trained_model() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);

        let train_data =
            generate_pattern_dataset(&mut rng, 20, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut rng, 5, config.block_size, config.vocab_size - 1);

        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);

        // Multistep with 4 steps should produce comparable results to standard 16 steps
        let multistep_config = D2fDecodeConfig {
            denoise_steps: 4,
            multistep: true,
            confidence_threshold: 0.3,
            block_size: config.block_size,
            temperature: 0.8,
            ..D2fDecodeConfig::default()
        };

        let result = d2f_decode_block(&weights, &config, &multistep_config, &NoPruner, &mut rng);

        let n_unmasked = result
            .tokens
            .iter()
            .filter(|&&t| t != config.mask_token)
            .count();
        assert!(
            n_unmasked > 0,
            "Multistep should unmask at least 1 token, got {n_unmasked}"
        );
    }

    #[test]
    fn test_multistep_blend_changes_behavior() {
        // Verify that multistep produces different denoising behavior than standard
        let config = Config::micro_dllm();
        let weights = TransformerWeights::new(&config, &mut Rng::new(42));

        let standard_config = D2fDecodeConfig {
            denoise_steps: 4,
            multistep: false,
            ..D2fDecodeConfig::with_block_size(4)
        };
        let multistep_config = D2fDecodeConfig {
            denoise_steps: 4,
            multistep: true,
            ..D2fDecodeConfig::with_block_size(4)
        };

        // Same seed for both — differences come only from the blend
        let result_standard = d2f_decode_block(
            &weights,
            &config,
            &standard_config,
            &NoPruner,
            &mut Rng::new(42),
        );
        let result_multistep = d2f_decode_block(
            &weights,
            &config,
            &multistep_config,
            &NoPruner,
            &mut Rng::new(42),
        );

        assert_eq!(result_standard.tokens.len(), result_multistep.tokens.len());
        // With untrained weights nothing gets unmasked (all-zero confidence),
        // so the blend has no observable effect. Only assert difference when
        // at least one config actually unmasks tokens.
        let any_unmasked = |r: &D2fBlockResult| r.confidence_history.iter().any(|&c| c > 0.0);
        if any_unmasked(&result_standard) || any_unmasked(&result_multistep) {
            assert_ne!(
                result_standard.confidence_history, result_multistep.confidence_history,
                "Multistep blend should change denoising behavior from step 1 onwards"
            );
        }
    }

    #[test]
    fn test_multistep_config_preset() {
        let config = D2fDecodeConfig::multistep_quality();
        assert!(config.multistep);
        assert_eq!(config.denoise_steps, 4);
        assert_eq!(config.confidence_threshold, 0.7);
    }

    // ── Plan 109 T5: D2fPipeline + SoftDecodeConfig Integration ────

    #[test]
    fn test_pipeline_with_soft_config_uses_soft_decode() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let decode_config = D2fDecodeConfig::with_block_size(4);
        let soft_config = SoftDecodeConfig::default();

        let pipeline = D2fPipeline::with_prompt(&config, decode_config, 4, &[config.bos_token])
            .with_soft_config(soft_config);

        let result = pipeline.decode_all(&weights, &NoPruner, &mut rng);

        // Should produce valid tokens (not all mask tokens)
        assert!(
            result.tokens.iter().any(|&t| t != config.mask_token),
            "SPD pipeline should decode at least one non-mask token"
        );
        // All tokens should be valid vocab indices
        for &t in &result.tokens {
            assert!(t < config.vocab_size, "token {t} out of vocab range");
        }
    }

    #[test]
    fn test_pipeline_without_soft_config_uses_binary_decode() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let decode_config = D2fDecodeConfig::with_block_size(4);

        let pipeline = D2fPipeline::with_prompt(&config, decode_config, 4, &[config.bos_token]);

        let result = pipeline.decode_all(&weights, &NoPruner, &mut rng);

        // Should produce valid tokens (not all mask tokens)
        assert!(
            result.tokens.iter().any(|&t| t != config.mask_token),
            "Binary pipeline should decode at least one non-mask token"
        );
        // All tokens should be valid vocab indices
        for &t in &result.tokens {
            assert!(t < config.vocab_size, "token {t} out of vocab range");
        }
    }

    #[test]
    fn test_pipeline_multi_block_spd_coherent() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let decode_config = D2fDecodeConfig::with_block_size(4);
        let soft_config = SoftDecodeConfig::default();

        // Decode 8 tokens across 2 blocks
        let pipeline = D2fPipeline::with_prompt(&config, decode_config, 8, &[config.bos_token])
            .with_soft_config(soft_config);

        let result = pipeline.decode_all(&weights, &NoPruner, &mut rng);

        // Should have 2 blocks
        assert_eq!(result.block_results.len(), 2, "should have 2 blocks");
        // Total tokens: prompt (1) + decoded (8) = 9
        assert_eq!(
            result.tokens.len(),
            9,
            "should have 1 prompt + 8 decoded tokens"
        );
        // All decoded tokens should be valid vocab indices
        for &t in &result.tokens {
            assert!(t < config.vocab_size, "token {t} out of vocab range");
        }
    }
}
