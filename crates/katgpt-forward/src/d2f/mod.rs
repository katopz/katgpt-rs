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

// Plan 399 (2026-07-05): moved from root `src/speculative/d2f.rs`.
// Import rewrites applied: `crate::dllm::*` → `crate::d2f_context::*`,
// `crate::speculative::types::*` → `katgpt_core::traits::*`,
// `crate::transformer::TransformerWeights` → `katgpt_transformer::*`,
// `crate::types::*` → `katgpt_types::*`.

use crate::d2f_context::{D2fContext, denoising_accuracy, forward_block_causal_with};
use katgpt_core::simd::{simd_add_scalar_inplace, simd_exp_inplace, simd_max_f32};
use katgpt_core::traits::{ConstraintPruner, ScreeningPruner};
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;
use katgpt_types::Rng;

#[cfg(feature = "tri_mode")]
use crate::diffusion_sampler::{DiffusionSampler, SamplerFeatures};

// ---------------------------------------------------------------------------
// D2F Block State Machine
// ---------------------------------------------------------------------------

/// State of a D2F decode block in the pipeline.
///
/// Transition flow:
/// ```text
/// SemiActivated (confidence < τ_act) → FullyActivated (confidence ≥ τ_act)
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug)]
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
    /// Argmax drafting (Issue 587 / FLARE Eq 21): pick the top-scoring valid
    /// token deterministically instead of sampling. Makes the drafter's
    /// proposal law a point mass, which is what the distribution-preserving
    /// `SoftmaxArgmax` acceptance policy requires for exactness. Consumes no
    /// RNG for token choice.
    pub greedy_draft: bool,
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
            greedy_draft: false,
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
// Schedule-Aware Multistep Ratios (Plan 079 T16)
// ---------------------------------------------------------------------------

/// Compute DPM-Solver++(2M) step-size ratios from schedule timesteps.
///
/// Given denoising timesteps `t_0 < t_1 < ... < t_{N-1}` in `[0,1]`, computes
/// `r_i = h_{i-1} / h_i` where `h_i = λ_i - λ_{i+1}` and `λ = logit(t)`
/// is the log-SNR (logit of mask probability).
///
/// For uniform steps, all `r_i = 1.0` (the Plan 078 default).
/// Non-uniform schedules (LogitNormal, EquiProbability) produce varying `r_i`,
/// giving the solver more accurate second-order extrapolation.
///
/// Returns `Vec<f32>` of length `max(0, n_steps - 1)`.
/// `ratios[i]` is used at denoising step `i+1` (step 0 has no history).
pub fn compute_multistep_ratios(steps: &[f32]) -> Vec<f32> {
    if steps.len() < 3 {
        // Need at least 3 points for one ratio: h_0 / h_1.
        // With < 3 steps, fall back to r=1.0 for all available steps.
        return vec![1.0f32; steps.len().saturating_sub(1)];
    }

    // Compute log-SNR (logit) for each timestep.
    // λ(t) = log(t / (1-t)) = logit(t)
    let lambdas: Vec<f32> = steps
        .iter()
        .map(|&t| {
            let clamped = t.clamp(1e-6, 1.0 - 1e-6);
            (clamped / (1.0 - clamped)).ln()
        })
        .collect();

    // h_i = λ_i - λ_{i+1} (step sizes in log-SNR space)
    // r_i = h_{i-1} / h_i for i >= 1
    let h: Vec<f32> = lambdas.windows(2).map(|w| w[0] - w[1]).collect();

    h.windows(2)
        .map(|w| {
            if w[1].abs() < 1e-8 {
                1.0 // Avoid division by zero; fall back to uniform
            } else {
                (w[0] / w[1]).abs().min(10.0) // Clamp for numerical stability
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// ELF Logit-Normal Schedule (Plan 079)
// ---------------------------------------------------------------------------

/// D2F noise schedule type (ELF Appendix C.6).
#[derive(Debug, Clone, Copy, Default)]
pub enum ScheduleKind {
    /// Uniform spacing between steps (current default).
    #[default]
    Uniform,
    /// Logit-normal distribution — concentrates steps near t=0 (ELF: μ=-1.5, σ=0.8).
    LogitNormal { mean: f32, std: f32 },
    /// Equi-probability partitioning from DiffusionBlocks (arXiv:2506.14202).
    ///
    /// Partitions noise levels by equal cumulative probability mass under
    /// log-normal, allocating more blocks to intermediate noise levels where
    /// denoising is hardest. The boundary σ values are:
    ///   σ_b = exp(P_mean + P_std · Φ⁻¹(q_b))  where q_b = q_min + (b/B)(q_max - q_min)
    ///
    /// This produces deterministic, evenly-spaced-in-CDF timesteps (no RNG needed).
    EquiProbability { mean: f32, std: f32 },
}

impl ScheduleKind {
    /// ELF paper default: LogitNormal with μ=-1.5, σ=0.8.
    pub fn elf_default() -> Self {
        Self::LogitNormal {
            mean: -1.5,
            std: 0.8,
        }
    }

    /// DiffusionBlocks (arXiv:2506.14202) equi-probability default.
    /// Uses EDM-style P_mean=-1.2, P_std=1.2 for the log-normal prior.
    pub fn diffusion_blocks_default() -> Self {
        Self::EquiProbability {
            mean: -1.2,
            std: 1.2,
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
            Self::EquiProbability { mean, std } => equi_probability_schedule(n_steps, *mean, *std),
        }
    }
}

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x)).
///
/// Delegates to `katgpt_core::simd::fast_sigmoid` (Cephes polynomial).
fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
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
            steps.sort_by(|a, b| a.total_cmp(b));
            steps
        }
    }
}

/// Deterministic equi-probability schedule from DiffusionBlocks (arXiv:2506.14202).
///
/// Places timesteps at equal intervals along the cumulative distribution function
/// of a log-normal distribution, so each interval carries equal probability mass.
/// This concentrates timesteps where the model spends the most denoising effort.
///
/// σ_b = exp(mean + std · Φ⁻¹(q_b))  where q_b = b / (B+1)  (interior quantiles)
///
/// Returns values mapped to [0.0, 1.0] via sigmoid for compatibility with the
/// denoising pipeline.
fn equi_probability_schedule(n_steps: usize, mean: f32, std: f32) -> Vec<f32> {
    match n_steps {
        0 => vec![],
        1 => vec![0.5],
        _ => {
            // Place quantiles at q_b = (b + 1) / (B + 1) for b = 0..B-1
            // (interior points that avoid the extreme tails)
            (0..n_steps)
                .map(|b| {
                    let q = (b + 1) as f32 / (n_steps + 1) as f32;
                    let inv_cdf = mean + std * approx_inverse_normal_cdf(q);
                    sigmoid(inv_cdf)
                })
                .collect()
        }
    }
}

/// Rational approximation of the inverse normal CDF (Φ⁻¹) using Peter Acklam's algorithm.
/// Accurate to ~1.15e-9 in absolute value across the full range.
#[allow(clippy::excessive_precision)]
fn approx_inverse_normal_cdf(p: f32) -> f32 {
    // Coefficients for the rational approximation
    const A1: f32 = -3.9696830e+01;
    const A2: f32 = 2.20946098e+02;
    const A3: f32 = -2.75928510e+02;
    const A4: f32 = 1.38357752e+02;
    const A5: f32 = -3.06647980e+01;
    const A6: f32 = 2.50662828e+00;

    const B1: f32 = -5.44760988e+01;
    const B2: f32 = 1.61585837e+02;
    const B3: f32 = -1.55698980e+02;
    const B4: f32 = 6.68013119e+01;
    const B5: f32 = -1.32806816e+01;

    const C1: f32 = -7.78489400e-03;
    const C2: f32 = -3.22396458e-01;
    const C3: f32 = -2.40075828e+00;
    const C4: f32 = -2.54973254e+00;
    const C5: f32 = 4.37466414e+00;
    const C6: f32 = 2.93816398e+00;

    const D1: f32 = 7.78469571e-03;
    const D2: f32 = 3.22467129e-01;
    const D3: f32 = 2.44513414e+00;
    const D4: f32 = 3.75440866e+00;

    const P_LOW: f32 = 0.02425;
    const P_HIGH: f32 = 1.0 - P_LOW;

    if p < P_LOW {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
            / ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)
    } else if p <= P_HIGH {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * q
            / (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1.0)
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6)
            / ((((D1 * q + D2) * q + D3) * q + D4) * q + 1.0)
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
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    let mut ctx = D2fContext::new(config);
    d2f_decode_block_with_prompt_with(
        &mut ctx,
        weights,
        config,
        decode_config,
        &[],
        pruner,
        screener,
        rng,
    )
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
    screener: &dyn ScreeningPruner,
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
        screener,
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
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    let mut result = d2f_decode_block(weights, config, decode_config, pruner, screener, rng);
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
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    d2f_decode_block_with_prompt_with(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        pruner,
        screener,
        rng,
    )
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
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    d2f_decode_block_prompt_q_core(
        dctx,
        weights,
        config,
        decode_config,
        prompt,
        pruner,
        screener,
        rng,
        None,
    )
}

/// [`d2f_decode_block_with_prompt_with`] + draft-time proposal-law capture
/// (Issue 587, FLARE Eq 8 analog).
///
/// `q_out` receives one vocab-sized row per denoised position holding the
/// **law that actually generated the committed token** at that position:
/// the normalized (pruned, relevance-weighted, temperature-scaled)
/// distribution the sampler drew from, or a point mass at the committed token
/// under `greedy_draft`. Positions still masked when decoding ends get a point
/// mass at `mask_token`. `q_out.len()` must be `>= block_positions * vocab_size`
/// (shorter buffers are written up to capacity, never panic).
///
/// The out-buffer shape (vs a `D2fBlockResult` field) keeps the hot path
/// zero-alloc: the caller pre-allocates once and reuses across calls (G4).
pub fn d2f_decode_block_with_prompt_with_q(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    prompt: &[usize],
    pruner: &dyn ConstraintPruner,
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
    q_out: &mut [f32],
) -> D2fBlockResult {
    d2f_decode_block_prompt_q_core(
        dctx,
        weights,
        config,
        decode_config,
        prompt,
        pruner,
        screener,
        rng,
        Some(q_out),
    )
}

/// Shared core of the D2F block decode, optionally capturing per-position
/// proposal laws (Issue 587).
#[allow(clippy::too_many_arguments, clippy::needless_option_as_deref)]
fn d2f_decode_block_prompt_q_core(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    prompt: &[usize],
    pruner: &dyn ConstraintPruner,
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
    mut q_out: Option<&mut [f32]>,
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
    let denoising_positions = seq_len - block_start;
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let temperature = decode_config.temperature;

    // Pre-compute schedule-aware multistep ratios from log-SNR (Plan 079 T16).
    // For uniform schedules, all r_i = 1.0 (identical to Plan 078 behavior).
    // Non-uniform schedules get schedule-adaptive extrapolation coefficients.
    let multistep_ratios = if decode_config.multistep {
        let schedule_steps = decode_config.schedule.generate_steps(max_steps, rng);
        compute_multistep_ratios(&schedule_steps)
    } else {
        Vec::new()
    };

    // Initialize: prompt + mask tokens for the block
    let mut tokens: Vec<usize> = prompt.to_vec();
    tokens.extend(std::iter::repeat_n(mask, block_size));
    tokens.truncate(config.block_size);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut converged_step = max_steps;
    // Reusable exp scratch buffer for sample_temperatured (avoids per-position
    // allocation in the denoising loop).
    let mut sample_exp_buf: Vec<f32> = Vec::with_capacity(vocab);

    for step in 0..max_steps {
        // Zero-alloc forward pass with block-causal attention
        let _seq_len_actual =
            forward_block_causal_with(dctx, weights, &tokens[..seq_len], config, block_size);

        // DPM-Solver++(2M) multistep logit extrapolation (Plan 078 T10.6, Plan 079 T16)
        //
        // Caches raw model outputs and blends with previous step's prediction
        // to get a second-order estimate:
        //   D_i = (1 + r_i/2) * current - (r_i/2) * prev  (RePlaid Eq 16)
        //
        // r_i = h_{i-1}/h_i is the step-size ratio in log-SNR space.
        // For uniform steps: r = 1.0, so D = 1.5 * current - 0.5 * prev.
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
                // r = step-size ratio in log-SNR space (schedule-aware, Plan 079 T16)
                let r = multistep_ratios.get(step - 1).copied().unwrap_or(1.0);
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
            let max_logit = simd_max_f32(logits_p);

            // Depth and parent tokens relative to block start
            let depth = p - block_start;
            let parent_tokens = &tokens[block_start..p];

            // Compute relevance-weighted softmax denominator over valid tokens only.
            // ScreeningPruner relevance ∈ [0.0, 1.0] multiplies the softmax exponent,
            // boosting semantically relevant tokens and dampening irrelevant ones.
            //
            // Perf: bulk SIMD exp into sample_exp_buf, then scalar masked accumulate.
            // The exp is the dominant cost (~10-20 cycles/elem scalar); SIMD cuts it
            // to ~2-4 cycles/elem. The mask + pruner branches are not SIMD-amenable
            // (dyn dispatch via trait objects) so they stay scalar.
            sample_exp_buf.resize(vocab, 0.0);
            bulk_exp_shift_into(logits_p, max_logit, &mut sample_exp_buf[..]);
            let mut sum_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !pruner.is_valid(depth, t, parent_tokens) {
                    continue;
                }
                let relevance = screener.relevance(depth, t, parent_tokens);
                sum_exp += sample_exp_buf[t] * relevance;
            }

            if sum_exp == 0.0 {
                // No valid tokens — keep masked
                continue;
            }

            // Token choice (Issue 587): argmax drafting under `greedy_draft`
            // (deterministic point-mass proposal law — the exactness
            // precondition of the SoftmaxArgmax accept policy), else the
            // historical temperature-scaled / greedy samplers.
            let (chosen_token, chosen_prob) = if decode_config.greedy_draft {
                let (tok, weight) = greedy_argmax_weighted(
                    &sample_exp_buf,
                    mask,
                    vocab,
                    depth,
                    parent_tokens,
                    pruner,
                    screener,
                );
                (tok, if sum_exp > 0.0 { weight / sum_exp } else { 0.0 })
            } else if temperature > 0.0 && temperature != 1.0 {
                sample_temperatured(
                    logits_p,
                    mask,
                    vocab,
                    max_logit,
                    temperature,
                    depth,
                    parent_tokens,
                    pruner,
                    screener,
                    rng,
                    &mut sample_exp_buf,
                )
            } else {
                sample_greedy(
                    &sample_exp_buf,
                    mask,
                    vocab,
                    sum_exp,
                    depth,
                    parent_tokens,
                    pruner,
                    screener,
                    rng,
                )
            };

            // Confidence remasking: only keep if confident enough
            if chosen_prob >= tau_conf && chosen_token != mask {
                tokens[p] = chosen_token;
                n_confident += 1;
                // Capture the draft-time proposal law q for this position
                // (Issue 587, ExactQ policy). Must happen at commit time:
                // once committed the position is never revisited, so this
                // is the last moment the generating distribution is at hand.
                if let Some(q) = q_out.as_deref_mut() {
                    capture_q_row(
                        q,
                        p - block_start,
                        vocab,
                        mask,
                        &sample_exp_buf,
                        chosen_token,
                        chosen_prob,
                        decode_config.greedy_draft,
                    );
                }
            }
        }

        let confidence = n_confident as f32 / block_size as f32;
        confidence_history.push(confidence);

        // Early exit: all block positions unmasked.
        // O(1) check via the counter we already maintain, avoiding a per-step
        // O(denoising_positions) linear scan of the token slice.
        if n_confident == denoising_positions {
            converged_step = step;
            break;
        }
    }

    // Fill q rows for positions that never committed: the final content is
    // the still-masked token (deterministic), so the honest proposal law is a
    // point mass at `mask`. Every row is now rewritten each call — no stale
    // state leaks across calls (the verifier reuses one buffer).
    if let Some(q) = q_out.as_deref_mut() {
        let q_rows = q.len() / vocab;
        for row in 0..q_rows.min(denoising_positions) {
            if tokens[block_start + row] == mask {
                write_q_point_mass(q, row, vocab, mask);
            }
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
    screener: &dyn ScreeningPruner,
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
    let denoising_positions = seq_len - block_start;
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let temperature = decode_config.temperature;

    // Pre-compute schedule-aware multistep ratios from log-SNR (Plan 079 T16).
    let multistep_ratios = if decode_config.multistep {
        let schedule_steps = decode_config.schedule.generate_steps(max_steps, rng);
        compute_multistep_ratios(&schedule_steps)
    } else {
        Vec::new()
    };

    // Initialize: prompt + mask tokens for the block
    let mut tokens: Vec<usize> = prompt.to_vec();
    tokens.extend(std::iter::repeat_n(mask, block_size));
    tokens.truncate(config.block_size);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut converged_step = max_steps;
    // Reusable exp scratch buffer for sample_temperatured (avoids per-position
    // allocation in the denoising loop).
    let mut sample_exp_buf: Vec<f32> = Vec::with_capacity(vocab);

    for step in 0..max_steps {
        // Zero-alloc forward pass with block-causal attention
        let _seq_len_actual =
            forward_block_causal_with(dctx, weights, &tokens[..seq_len], config, block_size);

        // DPM-Solver++(2M) multistep logit extrapolation (Plan 078 T10.6, Plan 079 T16)
        if decode_config.multistep {
            let logits_len = seq_len * vocab;
            dctx.prev_prev_logits_flat[..logits_len]
                .copy_from_slice(&dctx.logits_flat[..logits_len]);

            if step >= 1 {
                let r = multistep_ratios.get(step - 1).copied().unwrap_or(1.0);
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
            let max_logit = simd_max_f32(logits_p);

            let depth = p - block_start;
            let parent_tokens = &tokens[block_start..p];

            let mut sum_exp = 0.0f32;
            // Perf: bulk SIMD exp into sample_exp_buf, then scalar masked accumulate
            // (see site 1 for rationale).
            sample_exp_buf.resize(vocab, 0.0);
            bulk_exp_shift_into(logits_p, max_logit, &mut sample_exp_buf[..]);
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !pruner.is_valid(depth, t, parent_tokens) {
                    continue;
                }
                let relevance = screener.relevance(depth, t, parent_tokens);
                sum_exp += sample_exp_buf[t] * relevance;
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
                    screener,
                    rng,
                    &mut sample_exp_buf,
                )
            } else {
                sample_greedy(
                    &sample_exp_buf,
                    mask,
                    vocab,
                    sum_exp,
                    depth,
                    parent_tokens,
                    pruner,
                    screener,
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

        // Early exit: all block positions unmasked.
        // O(1) check via the counter we already maintain, avoiding a per-step
        // O(denoising_positions) linear scan of the token slice.
        if n_confident == denoising_positions {
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
    screener: &dyn ScreeningPruner,
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
        screener,
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
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    let mut result = d2f_decode_block_with_prompt_with(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        pruner,
        screener,
        rng,
    );
    result.accuracy = Some(denoising_accuracy(&result.tokens, target_tokens));
    result
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Bulk exp(logits - max_logit) into `exp_buf` using SIMD.
///
/// `exp_buf` must be at least `logits.len()` long. The result is the standard
/// softmax numerator: `exp(logit_t - max_logit)`, used by both the sum_exp
/// precheck loop and (with an extra `* inv_temp` factor) the temperature-scaled
/// sampler. Doing the exp once in bulk via SIMD replaces vocab scalar `.exp()`
/// calls (10-20 cycles each) with a single NEON/AVX2 pass (2-4 cycles/elem).
#[inline]
fn bulk_exp_shift_into(logits: &[f32], max_logit: f32, exp_buf: &mut [f32]) {
    let vocab = logits.len();
    exp_buf[..vocab].copy_from_slice(logits);
    // exp_buf -= max_logit (SIMD)
    simd_add_scalar_inplace(&mut exp_buf[..vocab], -max_logit);
    // exp_buf = exp(exp_buf) (SIMD)
    simd_exp_inplace(&mut exp_buf[..vocab]);
}

/// Temperature-scaled sampling from valid tokens.
/// Returns (token, probability).
///
/// `exp_buf` is a caller-owned scratch buffer (vocab-sized) reused across
/// calls to avoid per-position allocation in the denoising loop.
fn sample_temperatured(
    logits: &[f32],
    mask: usize,
    vocab: usize,
    max_logit: f32,
    temperature: f32,
    depth: usize,
    parent_tokens: &[usize],
    pruner: &dyn ConstraintPruner,
    screener: &dyn ScreeningPruner,
    rng: &mut Rng,
    exp_buf: &mut Vec<f32>,
) -> (usize, f32) {
    let inv_temp = 1.0 / temperature;

    // Bulk SIMD exp((logits - max_logit) * inv_temp) into exp_buf.
    // Replaces per-token scalar .exp() (~10-20 cycles each) with a single
    // NEON/AVX2 pass (~2-4 cycles/elem). Then a scalar masked pass applies
    // the pruner/screener gating and accumulates scaled_sum.
    exp_buf.resize(vocab, 0.0);
    let buf = &mut exp_buf[..vocab];
    // Slice `logits` to `vocab` up front so its bounds check hoists out of the
    // vocab-length loop (vocab can be 256K on production models).
    let lg = &logits[..vocab];
    for t in 0..vocab {
        buf[t] = (lg[t] - max_logit) * inv_temp;
    }
    simd_exp_inplace(buf);

    let mut scaled_sum = 0.0f32;
    for t in 0..vocab {
        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
            buf[t] = 0.0; // invalid token — zero weight, index-aligned
            continue;
        }
        let relevance = screener.relevance(depth, t, parent_tokens);
        buf[t] *= relevance;
        scaled_sum += buf[t];
    }

    if scaled_sum == 0.0 {
        return (mask, 0.0);
    }

    // Sample by cumulative distribution (no exp recompute).
    // Re-slice to `vocab` so the Vec index bounds check hoists out of the scan.
    let r = (rng.next() as f64 / u64::MAX as f64) as f32;
    let inv_scaled = 1.0 / scaled_sum;
    let mut cumsum = 0.0f32;
    let buf = &exp_buf[..vocab];
    for t in 0..vocab {
        let p = buf[t] * inv_scaled;
        cumsum += p;
        if cumsum >= r {
            return (t, p);
        }
    }

    // Fallback: return last valid token
    (mask, 0.0)
}

/// Argmax over valid (pruner-respecting, mask-skipping, relevance-weighted)
/// tokens. Deterministic — consumes no RNG. Returns `(token, unnormalized
/// weight)`; the caller divides by `sum_exp` for the distribution probability.
/// Returns `(mask, 0.0)` when no valid token exists.
///
/// Issue 587 (FLARE Eq 21): argmax drafting makes the drafter's proposal law
/// a point mass, which is the precondition the SoftmaxArgmax acceptance
/// policy needs for distribution-preserving verification.
fn greedy_argmax_weighted(
    exp_buf: &[f32],
    mask: usize,
    vocab: usize,
    depth: usize,
    parent_tokens: &[usize],
    pruner: &dyn ConstraintPruner,
    screener: &dyn ScreeningPruner,
) -> (usize, f32) {
    let mut best_tok = mask;
    let mut best_w = -1.0f32;
    // Slice once so the per-token bounds check hoists out of the scan.
    let exp_buf = &exp_buf[..vocab];
    for t in 0..vocab {
        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
            continue;
        }
        let w = exp_buf[t] * screener.relevance(depth, t, parent_tokens);
        if w > best_w {
            best_w = w;
            best_tok = t;
        }
    }
    (best_tok, best_w.max(0.0))
}

/// Write one captured proposal-law row (Issue 587, ExactQ policy).
///
/// `greedy` → point mass at `chosen` (the argmax law is deterministic).
/// Sampled   → `q[t] = exp_buf[t] / S`, recovering `1/S` from
///             `chosen_prob = exp_buf[chosen] / S` (avoids re-deriving S).
///
/// The captured row is the law the sampler *actually* drew from — for the
/// temperature path `exp_buf` holds the relevance-weighted exps, for the
/// temp==1 path the unweighted exps normalized by the relevance-weighted
/// `sum_exp` (matching `sample_greedy`'s historical behavior exactly).
#[allow(clippy::too_many_arguments)]
fn capture_q_row(
    q_flat: &mut [f32],
    row: usize,
    vocab: usize,
    mask: usize,
    exp_buf: &[f32],
    chosen: usize,
    chosen_prob: f32,
    greedy: bool,
) {
    let start = row * vocab;
    let end = start + vocab;
    if end > q_flat.len() {
        return; // short buffer — write what fits, never panic
    }
    let q_row = &mut q_flat[start..end];
    if greedy {
        write_q_point_mass(q_row, 0, vocab, chosen);
        return;
    }
    let denom = exp_buf.get(chosen).copied().unwrap_or(0.0);
    let inv = if denom > 0.0 && chosen_prob > 0.0 {
        chosen_prob / denom
    } else {
        0.0
    };
    let exp_buf = &exp_buf[..vocab.min(exp_buf.len())];
    for t in 0..vocab {
        q_row[t] = exp_buf.get(t).copied().unwrap_or(0.0) * inv;
    }
    // The temp==1 arm's `exp_buf` holds the UNWEIGHTED exps, but its sum_exp
    // denominator excludes the mask slot (mask can never be sampled) — so the
    // reconstructed row carries a stray mask mass of exp_buf[mask]/sum_exp,
    // which deflates every sampleable entry and biases exact_q_step's
    // acceptance min(1, p(d)/q(d)). Zero the mask slot: the proposal law is a
    // distribution over SAMPLEABLE tokens (the module doc's contract), and a
    // point-mass-at-mask row is exactly what write_q_point_mass emits for
    // never-committed positions.
    if mask < vocab {
        q_row[mask] = 0.0;
    }
}

/// Write a point-mass row (deterministic law) at `token` into `q_flat[row]`.
fn write_q_point_mass(q_flat: &mut [f32], row: usize, vocab: usize, token: usize) {
    let start = row * vocab;
    let end = start + vocab;
    if end > q_flat.len() {
        return;
    }
    q_flat[start..end].fill(0.0);
    if token < vocab {
        q_flat[start + token] = 1.0;
    }
}

/// Greedy sampling with temperature=1.0 from valid tokens.
/// Returns (token, probability).
///
/// `exp_buf` is the precomputed `exp(logits - max_logit)` vector (the same
/// one used to compute `sum_exp`). Reusing it avoids recomputing exp per
/// sampled token during the cumulative-distribution scan.
fn sample_greedy(
    exp_buf: &[f32],
    mask: usize,
    vocab: usize,
    sum_exp: f32,
    depth: usize,
    parent_tokens: &[usize],
    pruner: &dyn ConstraintPruner,
    _screener: &dyn ScreeningPruner,
    rng: &mut Rng,
) -> (usize, f32) {
    let r = (rng.next() as f64 / u64::MAX as f64) as f32;
    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 }; // multiply per token instead of divide
    let mut cumsum = 0.0f32;
    // Callers always `resize(vocab)` the buffer before calling; slice once so the
    // per-token bounds check hoists out of the vocab-length scan.
    let exp_buf = &exp_buf[..vocab];
    for t in 0..vocab {
        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
            continue;
        }
        let prob = exp_buf[t] * inv_sum;
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
        screener: &dyn ScreeningPruner,
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
        // Reused across blocks: rebuilt in place each iteration instead of
        // re-cloning the (growing) prefix per block.
        let mut seq_tokens = Vec::with_capacity(self.config.block_size);
        // Hoisted outside the block loop: both buffers are reused across blocks
        // (vocab and max_steps are loop-invariant). sample_exp_buf is purely
        // internal scratch; confidence_history is moved into each block's result
        // via mem::take, leaving an empty Vec for the next iteration.
        let mut sample_exp_buf: Vec<f32> = Vec::with_capacity(vocab);
        let mut confidence_history: Vec<f32> = Vec::with_capacity(max_steps);

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
            seq_tokens.clear();
            seq_tokens.extend_from_slice(&all_tokens);
            seq_tokens.extend(std::iter::repeat_n(mask, current_block_size));

            let seq_len = seq_tokens.len().min(self.config.block_size);
            let block_start = seq_len.saturating_sub(current_block_size);

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
                    let max_logit = simd_max_f32(logits_p);
                    let depth = p - block_start;
                    let parent_tokens = &seq_tokens[block_start..p];

                    let mut sum_exp = 0.0f32;
                    // Perf: bulk SIMD exp into sample_exp_buf, then scalar masked accumulate
                    // (see site 1 for rationale).
                    sample_exp_buf.resize(vocab, 0.0);
                    bulk_exp_shift_into(logits_p, max_logit, &mut sample_exp_buf[..]);
                    for t in 0..vocab {
                        if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
                            continue;
                        }
                        let relevance = screener.relevance(depth, t, parent_tokens);
                        sum_exp += sample_exp_buf[t] * relevance;
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
                            screener,
                            rng,
                            &mut sample_exp_buf,
                        )
                    } else {
                        sample_greedy(
                            &sample_exp_buf,
                            mask,
                            vocab,
                            sum_exp,
                            depth,
                            parent_tokens,
                            pruner,
                            screener,
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

                // Early exit: all block positions unmasked.
                // O(1) check via the counter we already maintain, avoiding a per-step
                // O(current_block_size) linear scan of the token slice.
                if n_confident == current_block_size {
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
                confidence_history: std::mem::take(&mut confidence_history),
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
#[derive(Clone, Copy, Debug)]
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
        // Slice the two inputs to `dim` up front so their bounds checks hoist out
        // of the loop (the `token_norm`/`mask_norm` reductions below still read
        // the full input slices, so those keep the original bindings).
        let te = &token_emb[..dim];
        let me = &mask_emb[..dim];
        let mut norm_sq = 0.0f32;
        for d in 0..dim {
            let h_tilde = pi * te[d] + one_minus_pi * me[d];
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
#[repr(u8)]
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

    // Reusable per-step scratch buffers — hoisted outside the denoising loop
    // to avoid allocating `current_top1` and `masked_positions` every step.
    let mut current_top1: Vec<usize> = vec![0usize; block_len];
    let mut masked_positions: Vec<usize> = Vec::with_capacity(block_len);
    // Vocab-sized softmax scratch — reused across all positions and steps.
    // Avoids per-position iterator-chain allocation for the sum_exp computation.
    let mut softmax_scratch: Vec<f32> = vec![0.0f32; vocab_size];

    let max_steps = decode_config.denoise_steps;

    for step_idx in 0..max_steps {
        // Forward pass with block-causal attention using current token IDs
        // Masked positions use mask_token; decoded positions use predicted tokens
        let seq_len = forward_block_causal_with(ctx, weights, &tokens, config, block_len);

        // Sample top-1 and confidence for each position
        // (current_top1 is reused from the hoisted buffer; cleared+refilled)
        current_top1.fill(0);
        for pos in 0..seq_len.min(block_len) {
            let logit_off = pos * vocab_size;
            let logit_end = logit_off + vocab_size;
            let logits = &ctx.logits_flat[logit_off..logit_end];

            // Single pass: argmax gives both best_idx and max_val (they coincide).
            // Working directly on the logits slice avoids a per-position Vec alloc.
            // total_cmp: branch-free, NaN-deterministic (NaN-broken weights get a
            // defined answer instead of whichever happens to be first).
            let (best_idx, max_val) = logits
                .iter()
                .copied()
                .enumerate()
                // float_order: a NaN logit must never win greedy argmax.
                .max_by(|a: &(usize, f32), b: &(usize, f32)| {
                    katgpt_core::float_order::cmp_for_max(a.1, b.1)
                })
                .unwrap_or((0, f32::NEG_INFINITY));

            current_top1[pos] = best_idx;

            // Confidence via softmax max probability.
            // best_val == max_val → exp(0) == 1, so conf = 1 / sum_exp.
            // SIMD-accelerated: copy logits into scratch, subtract max (in-place),
            // then fused exp+sum via simd_exp_sum_inplace.
            softmax_scratch[..vocab_size].copy_from_slice(logits);
            katgpt_core::simd::simd_add_scalar_inplace(
                &mut softmax_scratch[..vocab_size],
                -max_val,
            );
            let sum_exp: f32 =
                katgpt_core::simd::simd_exp_sum_inplace(&mut softmax_scratch[..vocab_size]);
            let conf = if sum_exp > 0.0 {
                1.0 / sum_exp
            } else {
                1.0 / vocab_size as f32
            };
            confidences[pos] = conf;
        }

        // Collect currently masked positions into the hoisted buffer
        masked_positions.clear();
        for p in 0..block_len {
            if tokens[p] == mask_id {
                masked_positions.push(p);
            }
        }

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

        // Swap: current_top1 becomes prev_top1 for next iteration.
        // We need a fresh buffer for the next iteration's current_top1, so
        // move the old prev_top1 (if any) back into current_top1 and reuse it.
        let prev = prev_top1.replace(core::mem::take(&mut current_top1));
        current_top1 = prev.unwrap_or_else(|| vec![0usize; block_len]);
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
                // float_order: a NaN logit must never win greedy argmax.
                .max_by(|a: &(usize, f32), b: &(usize, f32)| {
                    katgpt_core::float_order::cmp_for_max(a.1, b.1)
                })
                .map_or(0, |(i, _)| i);
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
mod tests;
