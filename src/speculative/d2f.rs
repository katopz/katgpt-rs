//! D2F (Discrete Diffusion Forcing) Inference Pipeline
//!
//! Implements block-parallel decoding via iterative denoising with block-causal attention.
//! Reference: Plan 066 Phase 2 — D2F inference in microgpt-rs.
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
    /// Confidence at each denoising step.
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
        }
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
        let config = Config::dllm_micro();
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
        let config = Config::dllm_micro();
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
        let config = Config::dllm_micro();
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
        let config = Config::dllm_micro();
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
        let config = Config::dllm_micro();
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
        let config = Config::dllm_micro();
        let decode_config = D2fDecodeConfig::with_block_size(4);
        let mut rng = Rng::new(42);

        let weights = TransformerWeights::new(&config, &mut rng);

        let result = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);

        assert!(!result.confidence_history.is_empty());
        assert_eq!(result.confidence_history.len(), result.steps_used);
    }
}
