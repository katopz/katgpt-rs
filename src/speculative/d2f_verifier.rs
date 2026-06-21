//! D2F Drafter Verifier — D2F diffusion drafts, AR verifies.
//!
//! Plan 089: Tri-Mode Inference — "self-speculation" mode.
//! Uses existing D2F block decode as drafter + existing AR as verifier.
//! Behind `tri_mode` feature gate.

use crate::dllm::D2fContext;
use crate::speculative::d2f::{D2fDecodeConfig, d2f_decode_block_with_prompt_with};
use crate::speculative::sampling::sample_from_distribution;
use crate::speculative::types::{NoPruner, NoScreeningPruner};
use crate::speculative::verifier::SpeculativeVerifier;
use crate::transformer::TransformerWeights;
use crate::transformer::{ForwardContext, MultiLayerKVCache, forward};
use crate::types::{Config, Rng, softmax_scaled};

/// Speculative verifier that uses D2F diffusion as drafter, AR as verifier.
///
/// This is the Nemotron "self-speculation" mode — same draft→verify→accept
/// pattern as LeviathanVerifier, but D2F drafts in parallel instead of
/// DFlash drafting sequentially.
///
/// Key difference from LeviathanVerifier:
/// - Draft: `d2f_decode_block()` (parallel, bidirectional within block)
/// - Verify: `forward()` with causal attention (same as Leviathan)
/// - KV caches are separate (block-causal for draft, causal for verify)
pub struct D2fDrafterVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
    pub d2f_config: D2fDecodeConfig,
    pub draft_width: usize,
    target_ctx: ForwardContext,
    target_cache: MultiLayerKVCache,
    d2f_ctx: D2fContext,
    // Buffer for target probability distribution
    probs_buf: Vec<f32>,
    // Flat p-distributions buffer: `[(draft_width + 1) * vocab_size]`.
    // Pre-allocated once, reused across speculate() calls (zero-alloc hot path).
    p_distributions_flat: Vec<f32>,
    // Pre-allocated accepted tokens buffer: `[draft_width + 1]`.
    // Cleared + reused across speculate() calls.
    accepted_buf: Vec<usize>,
}

impl<'a> D2fDrafterVerifier<'a> {
    /// Create a new D2F drafter verifier.
    ///
    /// `draft_width` must match `d2f_config.block_size` — the number of tokens
    /// the D2F drafter produces in parallel per block decode.
    pub fn new(
        target_weights: &'a TransformerWeights,
        target_config: &'a Config,
        d2f_config: D2fDecodeConfig,
        draft_width: usize,
    ) -> Self {
        // Ensure block_size is at least draft_width
        let block_size = d2f_config.block_size.max(draft_width);
        let config = D2fDecodeConfig {
            block_size,
            ..d2f_config
        };
        let vocab_size = target_config.vocab_size;
        Self {
            target_weights,
            target_config,
            d2f_config: config,
            draft_width,
            target_ctx: ForwardContext::new(target_config),
            target_cache: MultiLayerKVCache::new(target_config),
            d2f_ctx: D2fContext::new(target_config),
            probs_buf: vec![0.0f32; vocab_size],
            p_distributions_flat: vec![0.0f32; (draft_width + 1) * vocab_size],
            accepted_buf: Vec::with_capacity(draft_width + 1),
        }
    }
}

impl SpeculativeVerifier for D2fDrafterVerifier<'_> {
    #[allow(clippy::needless_range_loop)]
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        let target_temp = self.target_config.temperature;
        let vocab_size = self.target_config.vocab_size;
        let draft_width = self.draft_width;

        // ── Phase 0: Score initial token with target model ──────────
        self.target_cache.reset();
        {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                token,
                pos,
                self.target_config,
            );
            self.probs_buf.copy_from_slice(logits);
            softmax_scaled(&mut self.probs_buf, 1.0 / target_temp);
        }

        // ── Phase 1: D2F block decode — parallel draft ──────────────
        // Use the prompt (initial token) as context for D2F block decode.
        let prompt = &[token];
        let d2f_result = d2f_decode_block_with_prompt_with(
            &mut self.d2f_ctx,
            draft_weights,
            draft_config,
            &self.d2f_config,
            prompt,
            &NoPruner,
            &NoScreeningPruner,
            rng,
        );

        let draft_tokens = &d2f_result.tokens;
        let k = draft_tokens.len().min(draft_width);

        if k == 0 {
            // Fallback: no draft tokens produced, sample from target distribution
            return vec![sample_from_distribution(&self.probs_buf, rng)];
        }

        // Copy draft tokens to stack to avoid borrow conflict with &mut self
        let mut token_stack = [0usize; 64];
        let k_bounded = k.min(token_stack.len());
        token_stack[..k_bounded].copy_from_slice(&draft_tokens[..k_bounded]);

        // Store p_dist[0] from Phase 0 into flat buffer (no allocation)
        self.p_distributions_flat[..vocab_size].copy_from_slice(&self.probs_buf);

        // ── Phase 2: Target scoring of draft tokens ─────────────────
        // Score each drafted token through the target AR model to get p_dist[i+1].
        for (i, &draft_tok) in token_stack[..k_bounded].iter().enumerate() {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                draft_tok,
                pos + 1 + i,
                self.target_config,
            );
            self.probs_buf.copy_from_slice(logits);
            softmax_scaled(&mut self.probs_buf, 1.0 / target_temp);
            let start = (i + 1) * vocab_size;
            self.p_distributions_flat[start..start + vocab_size].copy_from_slice(&self.probs_buf);
        }

        // ── Phase 3: Rejection sampling with argmax comparison ──────
        // For D2F drafter, we use simple prefix matching:
        // Compare draft[i] with argmax of target p_dist[i+1].
        // Accept longest matching prefix + bonus token at first mismatch.
        self.accepted_buf.clear();
        let mut all_accepted = true;

        for i in 0..k_bounded {
            let draft_tok = token_stack[i];
            let offset = (i + 1) * vocab_size;
            let p_dist = &self.p_distributions_flat[offset..offset + vocab_size];

            // Argmax of target distribution = target's preferred token
            let target_tok = p_dist
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if draft_tok == target_tok {
                self.accepted_buf.push(draft_tok);
            } else {
                // Mismatch: take target's preferred token (bonus at rejection point)
                self.accepted_buf.push(target_tok);
                all_accepted = false;
                break;
            }
        }

        // ── Phase 4: Bonus token if all accepted ────────────────────
        if all_accepted {
            let bonus_start = k_bounded * vocab_size;
            let bonus_end = bonus_start + vocab_size;
            let bonus = if bonus_end <= self.p_distributions_flat.len() {
                let bonus_dist = &self.p_distributions_flat[bonus_start..bonus_end];
                sample_from_distribution(bonus_dist, rng)
            } else {
                // Fallback if buffer too small (shouldn't happen with correct sizing)
                sample_from_distribution(&self.probs_buf, rng)
            };
            self.accepted_buf.push(bonus);
        }

        // Safety: always return at least one token
        if self.accepted_buf.is_empty() {
            let p_dist = &self.p_distributions_flat[..vocab_size];
            self.accepted_buf
                .push(sample_from_distribution(p_dist, rng));
        }

        // OPT: avoid clone — move buffer contents out, leaving empty Vec for next call
        std::mem::take(&mut self.accepted_buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Rng;

    fn make_config() -> Config {
        let mut c = Config::micro();
        c.vocab_size = 64;
        c
    }

    #[test]
    fn test_d2f_verifier_returns_at_least_one() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let d2f_config = D2fDecodeConfig::with_block_size(4);
        let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);

        let accepted = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0,
            &mut Rng::new(100),
        );
        assert!(
            !accepted.is_empty(),
            "speculate must always return at least one token"
        );
    }

    #[test]
    fn test_d2f_verifier_deterministic() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let d2f_config = D2fDecodeConfig::with_block_size(4);

        let r1 = {
            let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);
            verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };

        let r2 = {
            let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);
            verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(r1, r2, "same seed must produce identical output");
    }

    #[test]
    fn test_d2f_verifier_max_tokens_bounded() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let draft_width = 4;
        let d2f_config = D2fDecodeConfig::with_block_size(draft_width);
        let mut verifier =
            D2fDrafterVerifier::new(&target_weights, &config, d2f_config, draft_width);

        // Run many seeds — accepted.len() should never exceed draft_width + 1 (bonus)
        for seed in 0..50u64 {
            let accepted = verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0,
                &mut Rng::new(seed),
            );
            assert!(
                accepted.len() <= draft_width + 1,
                "accepted {} tokens but max is {}",
                accepted.len(),
                draft_width + 1,
            );
        }
    }
}
