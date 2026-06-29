//! FlashAR Strided Anchor-Then-Fill D2F Decoding
//!
//! Plan 166 T11 (stretch goal): Two-round decoding inspired by FlashAR's
//! diagonal-step parallel decoding pattern.
//!
//! # Architecture
//!
//! Round 1 (Anchor — diagonal analog):
//!   AR predicts every S-th position. Stride S controls anchor density.
//!   Few AR forward passes (block_size / stride) produce high-quality anchor tokens.
//!
//! Round 2 (Fill — parallel denoising):
//!   D2F decodes the remaining positions with anchor tokens pre-filled.
//!   Anchor positions start unmasked, reducing the denoising search space.
//!   Expected: fewer denoising iterations → faster convergence.
//!
//! # Feature Gate
//!
//! `flashar_anchor` (requires `dllm`)

#![allow(clippy::too_many_arguments)]

use crate::dllm::D2fContext;
use crate::speculative::d2f::{D2fBlockResult, D2fDecodeConfig};
use crate::speculative::types::{NoPruner, NoScreeningPruner};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, Rng, softmax_scaled};

use katgpt_core::speculative::sampling::sample_from_distribution;

// ---------------------------------------------------------------------------
// Anchor-Then-Fill Configuration
// ---------------------------------------------------------------------------

/// Configuration for strided anchor-then-fill decoding.
#[derive(Clone, Debug)]
pub struct AnchorConfig {
    /// Stride S: predict every S-th position via AR in Round 1.
    /// S=1 → pure AR (every position anchored).
    /// S=block_size → pure D2F (no anchors).
    /// Recommended: 2–4 for balanced anchor density.
    pub stride: usize,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self { stride: 2 }
    }
}

impl AnchorConfig {
    pub fn with_stride(stride: usize) -> Self {
        Self {
            stride: stride.max(1),
        }
    }
}

/// Result of the two-round anchor-then-fill decode.
#[derive(Clone, Debug)]
pub struct AnchorFillResult {
    /// Final decoded tokens for the block.
    pub tokens: Vec<usize>,
    /// Number of anchor positions predicted in Round 1.
    pub n_anchors: usize,
    /// Number of denoising steps used in Round 2.
    pub fill_steps_used: usize,
    /// Denoising steps used by baseline D2F (no anchors) for comparison.
    pub baseline_steps_used: usize,
    /// Reduction in denoising steps vs baseline.
    pub step_reduction: usize,
}

// ---------------------------------------------------------------------------
// Round 1: Strided AR Anchor Prediction
// ---------------------------------------------------------------------------

/// Predict anchor tokens at every S-th position using AR forward passes.
///
/// Returns the number of anchor tokens written into `token_buf`.
/// Anchor positions are `stride-1, 2*stride-1, 3*stride-1, ...` (0-indexed
/// within the block). Positions between anchors remain `mask_token`.
fn predict_anchors(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    weights: &TransformerWeights,
    config: &Config,
    seed_token: usize,
    start_pos: usize,
    block_size: usize,
    stride: usize,
    mask_token: usize,
    token_buf: &mut [usize],
    rng: &mut Rng,
) -> usize {
    let vocab = config.vocab_size;
    let temperature = config.temperature;
    let mut n_anchors = 0usize;

    // Initialize all block positions to mask
    for t in token_buf.iter_mut().take(block_size) {
        *t = mask_token;
    }

    // AR walk: predict tokens sequentially, but only "anchor" at stride positions
    let mut cur_token = seed_token;
    let mut ar_logits_buf = vec![0.0f32; vocab];

    for (pos_in_block, slot) in token_buf.iter_mut().enumerate().take(block_size) {
        let global_pos = start_pos + pos_in_block;

        // Forward pass at this position
        let logits = forward(ctx, weights, cache, cur_token, global_pos, config);
        ar_logits_buf.copy_from_slice(logits);

        // Sample from logits
        softmax_scaled(&mut ar_logits_buf, 1.0 / temperature);
        let next_token = sample_from_distribution_weighted(&ar_logits_buf, rng);

        // Anchor: store at stride positions (stride-1, 2*stride-1, ...)
        if (pos_in_block + 1) % stride == 0 {
            *slot = next_token;
            n_anchors += 1;
        }

        cur_token = next_token;
    }

    n_anchors
}

// ---------------------------------------------------------------------------
// Round 2: D2F Fill with Anchors Pre-filled
// ---------------------------------------------------------------------------

/// Run D2F denoising with anchor positions pre-filled in the token buffer.
///
/// This is a modified version of `d2f_decode_block_with_prompt_with` that
/// accepts pre-filled anchor tokens instead of starting from all-mask.
fn fill_with_anchors(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    prompt: &[usize],
    anchor_tokens: &[usize],
    pruner: &dyn crate::speculative::types::ConstraintPruner,
    screener: &dyn crate::speculative::types::ScreeningPruner,
    rng: &mut Rng,
) -> D2fBlockResult {
    // Use the prompt + anchor-initialized block
    let mask = config.mask_token;
    let vocab = config.vocab_size;
    let block_size = decode_config.block_size;
    let seq_len = (prompt.len() + block_size).min(config.block_size);
    let block_start = prompt.len();
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let _temperature = decode_config.temperature;

    // Initialize: prompt + anchor-prefilled tokens
    let mut tokens: Vec<usize> = prompt.to_vec();
    // Copy anchor tokens (non-mask positions already filled)
    tokens.extend_from_slice(anchor_tokens);
    tokens.truncate(config.block_size);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut converged_step = max_steps;

    // Scratch buffer reused across all positions and steps — fused
    // single-pass sampling (see below). `exp_scratch[t]` caches
    // `exp(logit_t − max)` so the second sampling pass can read it without
    // re-doing the transcendentals. Per-position we also stash
    // `exp(logit-max) · relevance` into `sum_exp` directly (it's a scalar,
    // not a slice). Allocating once here and overwriting per-position
    // halves the `.exp()` calls per token in the hot path.
    let mut exp_scratch = vec![0.0f32; vocab];

    for step in 0..max_steps {
        let _seq_len_actual = crate::dllm::forward_block_causal_with(
            dctx,
            weights,
            &tokens[..seq_len],
            config,
            block_size,
        );

        let mut n_confident = 0usize;

        for p in block_start..seq_len {
            // Skip positions that are already filled (anchors or previously denoised)
            if tokens[p] != mask {
                n_confident += 1;
                continue;
            }

            let logits_start = p * vocab;
            let logits_end = logits_start + vocab;
            let logits_p = &dctx.logits_flat[logits_start..logits_end];
            let max_logit = logits_p.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let depth = p - block_start;
            let parent_tokens = &tokens[block_start..p];

            // ── Fused single pass: compute weights + sum_exp together ──
            // The original two-pass code had an asymmetry: `sum_exp` included
            // `* relevance` but the second-pass `cum` did NOT. We preserve that
            // exact semantics by caching `exp(logit-max)` in `exp_scratch` and
            // `exp(logit-max) * relevance` in `weights_scratch`. The second
            // pass reads from `exp_scratch` instead of recomputing `.exp()`.
            let mut sum_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    exp_scratch[t] = 0.0;
                    continue;
                }
                if !pruner.is_valid(depth, t, parent_tokens) {
                    exp_scratch[t] = 0.0;
                    continue;
                }
                let relevance = screener.relevance(depth, t, parent_tokens);
                let e = (logits_p[t] - max_logit).exp();
                exp_scratch[t] = e;
                // The original code's sum_exp multiplied by relevance (even for
                // negative relevance, which is unusual but kept for parity).
                sum_exp += e * relevance;
            }

            if sum_exp == 0.0 {
                continue;
            }

            // ── Sample via cumulative sum over the cached `exp_scratch` ──
            // The original sampling loop used UNWEIGHTED `exp(logit-max)`
            // (no relevance). We read from `exp_scratch[t]` to avoid the
            // second `.exp()` call per token.
            let threshold = rng.uniform() * sum_exp;
            let mut best_token = mask;
            let mut cum = 0.0f32;
            for (t, &exp_val) in exp_scratch.iter().enumerate().take(vocab) {
                if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
                    continue;
                }
                cum += exp_val;
                if cum >= threshold && best_token == mask {
                    best_token = t;
                    // Keep iterating to preserve the original loop semantics
                    // (no early break — matches d2f.rs reference).
                }
            }

            // Compute probability of chosen token from the cached `exp_scratch`.
            // Original semantics: UNWEIGHTED exp divided by WEIGHTED sum_exp.
            let best_prob = if best_token != mask {
                exp_scratch[best_token] / sum_exp
            } else {
                0.0
            };

            if best_prob >= tau_conf && best_token != mask {
                tokens[p] = best_token;
                n_confident += 1;
            }
        }

        let confidence = n_confident as f32 / block_size as f32;
        confidence_history.push(confidence);

        // Early exit: all block positions unmasked.
        if tokens[block_start..seq_len].iter().all(|&t| t != mask) {
            converged_step = step;
            break;
        }
    }

    let all_unmasked = tokens[block_start..seq_len].iter().all(|&t| t != mask);
    let final_confidence = confidence_history.last().copied().unwrap_or(0.0);

    let state = if all_unmasked {
        crate::speculative::d2f::D2fBlockState::FullyActivated
    } else {
        crate::speculative::d2f::D2fBlockState::SemiActivated {
            step: converged_step.min(max_steps - 1),
            confidence: final_confidence,
        }
    };

    let block_tokens: Vec<usize> = tokens[block_start..seq_len].to_vec();

    crate::speculative::d2f::D2fBlockResult {
        tokens: block_tokens,
        steps_used: confidence_history.len(),
        confidence_history,
        accuracy: None,
        state,
    }
}

// ---------------------------------------------------------------------------
// Public API: Anchor-Then-Fill Decode
// ---------------------------------------------------------------------------

/// Run two-round anchor-then-fill D2F decoding.
///
/// 1. **Round 1 (Anchor):** AR predicts every S-th token.
/// 2. **Round 2 (Fill):** D2F denoises remaining positions with anchors pre-filled.
///
/// Also runs a baseline D2F decode (no anchors) to measure step reduction.
pub fn anchor_then_fill(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    anchor_config: &AnchorConfig,
    seed_token: usize,
    start_pos: usize,
    rng: &mut Rng,
) -> AnchorFillResult {
    let block_size = decode_config.block_size;
    let mask = config.mask_token;

    // ── Round 1: Strided AR anchor prediction ──
    let mut anchor_buf = vec![mask; block_size];
    let n_anchors = predict_anchors(
        ctx,
        cache,
        weights,
        config,
        seed_token,
        start_pos,
        block_size,
        anchor_config.stride,
        mask,
        &mut anchor_buf,
        rng,
    );

    // ── Round 2: D2F fill with anchors ──
    let fill_result = fill_with_anchors(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        &anchor_buf,
        &NoPruner,
        &NoScreeningPruner,
        rng,
    );

    // ── Baseline: D2F without anchors (for comparison) ──
    let baseline_result = crate::speculative::d2f::d2f_decode_block_with_prompt_with(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        &NoPruner,
        &NoScreeningPruner,
        rng,
    );

    AnchorFillResult {
        tokens: fill_result.tokens,
        n_anchors,
        fill_steps_used: fill_result.steps_used,
        baseline_steps_used: baseline_result.steps_used,
        step_reduction: baseline_result
            .steps_used
            .saturating_sub(fill_result.steps_used),
    }
}

// ---------------------------------------------------------------------------
// Helper: Weighted sampling from probability distribution
// ---------------------------------------------------------------------------

/// Sample from a probability distribution (sums to ~1.0).
/// Reuses the pattern from `sample_from_distribution` but works on raw probs.
fn sample_from_distribution_weighted(probs: &[f32], rng: &mut Rng) -> usize {
    sample_from_distribution(probs, rng)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dllm::D2fContext;
    use crate::speculative::d2f::D2fDecodeConfig;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    fn make_config() -> Config {
        Config::micro_dllm()
    }

    /// Train a mini D2F model on pattern data (same recipe as
    /// `speculative::d2f::tests::test_decode_with_trained_model`).
    ///
    /// The anchor-then-fill tests assert properties that only hold for a
    /// trained model — a random `TransformerWeights::new` produces a
    /// degenerate all-mask baseline that trivially converges in 1 step by
    /// emitting the same token at every position, which inverts the
    /// step-reduction comparison the tests are trying to verify.
    fn make_trained_weights() -> (Config, TransformerWeights) {
        use crate::dllm::{generate_pattern_dataset, train_mini_dllm};
        let config = make_config();
        let mut train_rng = Rng::new(123);
        let train_data =
            generate_pattern_dataset(&mut train_rng, 20, config.block_size, config.vocab_size - 1);
        let test_data =
            generate_pattern_dataset(&mut train_rng, 5, config.block_size, config.vocab_size - 1);
        let (weights, _) = train_mini_dllm(&config, &train_data, &test_data, 200, 0.01, 0.3, 42);
        (config, weights)
    }

    #[test]
    fn test_anchor_config_default_stride() {
        let cfg = AnchorConfig::default();
        assert_eq!(cfg.stride, 2);
    }

    #[test]
    fn test_anchor_config_with_stride_clamped() {
        let cfg = AnchorConfig::with_stride(0);
        assert_eq!(cfg.stride, 1);
    }

    #[test]
    fn test_predict_anchors_fills_stride_positions() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let block_size = 8;
        let stride = 2;
        let mask = config.mask_token;

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut token_buf = vec![mask; block_size];

        let n_anchors = predict_anchors(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            0, // seed_token
            0, // start_pos
            block_size,
            stride,
            mask,
            &mut token_buf,
            &mut rng,
        );

        // With stride=2 and block_size=8, anchors at positions 1, 3, 5, 7 → 4 anchors
        assert_eq!(n_anchors, 4);
        // Anchor positions should be non-mask
        for pos in [1, 3, 5, 7] {
            assert_ne!(token_buf[pos], mask, "position {pos} should be anchored");
        }
        // Non-anchor positions should still be mask
        for pos in [0, 2, 4, 6] {
            assert_eq!(token_buf[pos], mask, "position {pos} should still be mask");
        }
    }

    #[test]
    fn test_predict_anchors_stride_4() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let block_size = 8;
        let stride = 4;
        let mask = config.mask_token;

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut token_buf = vec![mask; block_size];

        let n_anchors = predict_anchors(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            0,
            0,
            block_size,
            stride,
            mask,
            &mut token_buf,
            &mut rng,
        );

        // stride=4: anchors at positions 3, 7 → 2 anchors
        assert_eq!(n_anchors, 2);
        assert_ne!(token_buf[3], mask);
        assert_ne!(token_buf[7], mask);
        for pos in [0, 1, 2, 4, 5, 6] {
            assert_eq!(token_buf[pos], mask);
        }
    }

    #[test]
    fn test_anchor_then_fill_produces_valid_output() {
        // Uses a trained mini D2F model — random weights produce a degenerate
        // all-same-token output that doesn't exercise the fill path.
        let (config, weights) = make_trained_weights();
        let mut rng = Rng::new(42);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0, // seed_token
            0, // start_pos
            &mut rng,
        );

        // Should produce block_size tokens
        assert_eq!(result.tokens.len(), block_size);
        // All tokens should be valid (non-mask)
        let mask = config.mask_token;
        for (i, &t) in result.tokens.iter().enumerate() {
            assert_ne!(t, mask, "token at position {i} should not be mask");
        }
        // Should have anchors
        assert!(result.n_anchors > 0, "should have at least 1 anchor");
    }

    #[test]
    fn test_anchor_then_fill_stride_1_all_anchored() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(1); // Every position anchored

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng,
        );

        // stride=1: all positions are anchors
        assert_eq!(result.n_anchors, block_size);
        // Fill steps should be minimal (nothing to denoise)
        assert!(
            result.fill_steps_used <= 1,
            "with all anchors, fill should converge immediately, got {}",
            result.fill_steps_used
        );
    }

    #[test]
    fn test_anchor_then_fill_reduces_steps() {
        // The "anchors reduce denoising steps" property only holds for a
        // trained model — with random weights, the all-mask baseline
        // degenerately converges in 1 step (same token at every position),
        // inverting the comparison. See `make_trained_weights` doc.
        let (config, weights) = make_trained_weights();
        let mut rng = Rng::new(42);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng,
        );

        // With anchors pre-filled, fill steps should be ≤ baseline steps
        // (anchors reduce the denoising search space)
        assert!(
            result.fill_steps_used <= result.baseline_steps_used,
            "anchor fill ({}) should use ≤ baseline ({}) steps",
            result.fill_steps_used,
            result.baseline_steps_used,
        );

        println!(
            "  anchors={}, fill_steps={}, baseline_steps={}, reduction={}",
            result.n_anchors,
            result.fill_steps_used,
            result.baseline_steps_used,
            result.step_reduction,
        );
    }

    #[test]
    fn test_anchor_then_fill_deterministic() {
        let config = make_config();
        let block_size = 8;
        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut rng1 = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng1);

        let mut rng2 = Rng::new(42);
        let weights2 = TransformerWeights::new(&config, &mut rng2);

        let mut rng1 = Rng::new(99);
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let mut dctx1 = D2fContext::new(&config);

        let result1 = anchor_then_fill(
            &mut ctx1,
            &mut cache1,
            &mut dctx1,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng1,
        );

        let mut rng2 = Rng::new(99);
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let mut dctx2 = D2fContext::new(&config);

        let result2 = anchor_then_fill(
            &mut ctx2,
            &mut cache2,
            &mut dctx2,
            &weights2,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng2,
        );

        assert_eq!(
            result1.tokens, result2.tokens,
            "results should be deterministic"
        );
        assert_eq!(result1.n_anchors, result2.n_anchors);
        assert_eq!(result1.fill_steps_used, result2.fill_steps_used);
    }

    #[test]
    fn test_anchor_density_vs_stride() {
        // Verify anchor count = block_size / stride for various strides
        let config = make_config();
        let block_size = 8;

        for stride in [1, 2, 4, 8] {
            let mut rng = Rng::new(42);
            let weights = TransformerWeights::new(&config, &mut rng);
            let mask = config.mask_token;

            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let mut token_buf = vec![mask; block_size];

            let n_anchors = predict_anchors(
                &mut ctx,
                &mut cache,
                &weights,
                &config,
                0,
                0,
                block_size,
                stride,
                mask,
                &mut token_buf,
                &mut rng,
            );

            let expected = block_size / stride;
            assert_eq!(
                n_anchors, expected,
                "stride={stride}: expected {expected} anchors, got {n_anchors}"
            );
        }
    }
}
