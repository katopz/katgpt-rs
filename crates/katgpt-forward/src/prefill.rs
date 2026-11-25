//! Speculative prefill scorers (Plan 394, 2026-07-05).
//!
//! Moved from root `src/speculative/prefill.rs`. This file hosts the
//! forward-coupled scorers (`AttentionScorer`, `BlockAttentionScorer`) that
//! compose `forward()` (moved here in Plan 385). The pure substrate half
//! (`PrefillScorer` trait + `RandomScorer` + `UniformScorer` + compression /
//! selection functions) lives in `katgpt_speculative::prefill`; this file
//! re-exports those names so historical `katgpt_rs::speculative::prefill::*`
//! paths keep resolving.
//!
//! The `block_select_entmax` function (Plan 106 T20) **stays root** in
//! `src/speculative/prefill.rs` because it consumes `crate::dash_attn::*`
//! (a re-export from katgpt-attn's heavy `dash_attn` feature chain). Adding
//! katgpt-attn as a dep of katgpt-forward would create a cycle (katgpt-attn
//! already depends on katgpt-forward for forward glue).

// Re-export the pure substrate from the leaf crate so historical
// `katgpt_rs::speculative::prefill::*` paths resolve via the root's
// `pub use katgpt_forward::prefill::*` chain.
pub use katgpt_speculative::prefill::{
    PrefillScorer, RandomScorer, UniformScorer, block_compression_ratio, block_select,
    block_select_grid, compress_prompt, compress_prompt_blocks, should_compress,
    speculative_prefill, speculative_prefill_adaptive, speculative_prefill_block,
};
// `block_score_maxsim` is gated `maxsim` in the leaf crate; preserve the same
// gate on the re-export.
#[cfg(feature = "maxsim")]
pub use katgpt_speculative::prefill::block_score_maxsim;

use crate::{SpeculativeContext, forward};
use katgpt_core::speculative::types::FlashPrefillConfig;
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;

// ── Attention Scorer (forward-coupled) ────────────────────────

/// Attention-based importance scorer (PFlash-inspired).
/// Uses softmax'd self-attention weights from draft model forward pass.
///
/// After each `forward()` call, `ctx.scores[0..=pos]` contains the last
/// attention head's normalized attention weights. The weight at index `pos`
/// (the self-attention weight) serves as a proxy for per-token importance.
pub struct AttentionScorer;

impl AttentionScorer {
    /// Zero-alloc scoring using pre-allocated context.
    /// `scores` must be `>= prompt_tokens.len()`.
    pub fn score_with(
        &self,
        sctx: &mut SpeculativeContext,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        if prompt_tokens.is_empty() {
            return;
        }

        sctx.cache.reset();

        let len = prompt_tokens.len().min(scores.len());
        scores[..len].fill(0.0f32);

        for (pos, &token) in prompt_tokens.iter().enumerate() {
            if pos >= draft_config.block_size {
                break;
            }
            let _logits = forward(
                &mut sctx.ctx,
                draft_weights,
                &mut sctx.cache,
                token,
                pos,
                draft_config,
            );
            scores[pos] = sctx.ctx.scores[pos];
        }

        // Normalize scores to [0, 1] range
        let max_score = scores[..len].iter().cloned().fold(0.0f32, f32::max);
        if max_score > 0.0 {
            for s in scores[..len].iter_mut() {
                *s /= max_score;
            }
        }
    }
}

impl PrefillScorer for AttentionScorer {
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        if prompt_tokens.is_empty() {
            return Vec::new();
        }

        let mut sctx = SpeculativeContext::new(draft_config);
        let mut scores = vec![0.0f32; prompt_tokens.len()];
        self.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            prompt_tokens,
            &mut scores,
        );
        scores
    }
}

// ── Block Attention Scorer (forward-coupled, block-aware) ─────

/// Block-sparse attention scorer (CPU fallback for PFlash).
///
/// Aggregates token-level attention scores into block-level importance,
/// then upsamples back to per-token scores for compression.
pub struct BlockAttentionScorer {
    pub config: FlashPrefillConfig,
}

impl BlockAttentionScorer {
    /// Zero-alloc scoring using pre-allocated context.
    pub fn score_with(
        &self,
        sctx: &mut SpeculativeContext,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
        scores: &mut [f32],
    ) {
        let block_size = self.config.block_size;
        let seq_len = prompt_tokens.len();
        let num_blocks = seq_len.div_ceil(block_size);

        if seq_len == 0 {
            return;
        }

        sctx.cache.reset();

        let filled = seq_len.min(draft_config.block_size);
        for (pos, &token) in prompt_tokens.iter().enumerate().take(filled) {
            let _logits = forward(
                &mut sctx.ctx,
                draft_weights,
                &mut sctx.cache,
                token,
                pos,
                draft_config,
            );
        }

        let mut block_scores = vec![0.0f32; num_blocks];
        let mut block_counts = vec![0usize; num_blocks];

        let tail_start = seq_len.saturating_sub(self.config.tail_window * block_size);
        for (pos, &score) in sctx.ctx.scores[..filled]
            .iter()
            .enumerate()
            .skip(tail_start)
        {
            let block_idx = pos / block_size;
            if block_idx < num_blocks {
                block_scores[block_idx] += score;
                block_counts[block_idx] += 1;
            }
        }

        for (block_score, &count) in block_scores.iter_mut().zip(&block_counts) {
            if count > 0 {
                *block_score /= count as f32;
            }
        }

        let max_block = block_scores.iter().cloned().fold(0.0f32, f32::max);
        if max_block > 0.0 {
            for s in &mut block_scores {
                *s /= max_block;
            }
        }

        for (pos, slot) in scores.iter_mut().enumerate().take(seq_len) {
            *slot = block_scores[pos / block_size];
        }
    }
}

impl PrefillScorer for BlockAttentionScorer {
    fn score(
        &self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        prompt_tokens: &[usize],
    ) -> Vec<f32> {
        let mut sctx = SpeculativeContext::new(draft_config);
        let mut scores = vec![0.0f32; prompt_tokens.len()];
        self.score_with(
            &mut sctx,
            draft_weights,
            draft_config,
            prompt_tokens,
            &mut scores,
        );
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_transformer::TransformerWeights;
    use katgpt_types::Rng;

    fn make_draft() -> (TransformerWeights, Config) {
        let config = Config::draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        (weights, config)
    }

    #[test]
    fn test_attention_scorer_produces_scores() {
        let (weights, config) = make_draft();
        let tokens: Vec<usize> = (0..config.vocab_size.min(8)).collect();
        let scorer = AttentionScorer;
        let scores = scorer.score(&weights, &config, &tokens);
        assert_eq!(scores.len(), tokens.len());
        // All scores should be finite
        for &s in &scores {
            assert!(s.is_finite(), "score should be finite, got {s}");
        }
    }
}
