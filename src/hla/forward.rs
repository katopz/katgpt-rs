//! Forward pass implementations for Higher-order Linear Attention.
//!
//! Two variants:
//! - `forward_hla`: Symmetric second-order A·Aᵀ·V with O(d²) constant cache
//! - `forward_ahla`: Asymmetric second-order A·A·V with O(d·dv) constant cache
//!
//! Both mirror `forward_base()` structure but replace the KV cache store + attention
//! loop with streaming HLA/AHLA update + readout. No context window limit — streaming
//! is O(1) per token regardless of sequence length.
//!
//! Reference: Zhang, Qin, Wang, Gu (2026). "Higher-order Linear Attention."
//! See `.research/28_Higher_order_Linear_Attention.md` for full derivation.

use crate::hla::kernel::{ahla_layer_step, hla_layer_readout, hla_layer_update};
use crate::hla::types::{MultiLayerAhlaCache, MultiLayerHlaCache};
use crate::simd::{simd_add_inplace, simd_add_into};
use crate::transformer::{ForwardContext, TransformerWeights};
use crate::types::{self, Config};

// ── Symmetric Second-Order HLA Forward ─────────────────────────

/// Forward pass using symmetric second-order HLA cache.
///
/// Replaces the growing KV cache with constant-size prefix sufficient statistics.
/// The attention output is computed as:
///
/// ```text
/// o_t = q_tᵀ (SK_t · CQV_t − G_t) / (q_tᵀ (SK_t · mQ_t − h_t) + ε)
/// ```
///
/// # vs `forward_base()`
///
/// | Step | `forward_base` | `forward_hla` |
/// |------|----------------|---------------|
/// | QKV projection | Same | Same |
/// | Cache store | `layer_cache.key[pos] = k` | **Skip** (state update instead) |
/// | Attention | Loop over past positions O(N·d) | **Readout** O(d²) constant |
/// | MLP | Same | Same |
/// | Memory per layer | O(block_size × kv_dim) growing | O(d² + d·dv) constant |
///
/// # Arguments
/// * `ctx` — Pre-allocated forward context with Q/K/V buffers
/// * `weights` — Transformer weights (same as `forward_base`)
/// * `cache` — Symmetric HLA cache (constant-size, no `block_size` dependency)
/// * `token` — Input token index
/// * `pos` — Position in sequence (used for position embedding only, not cache sizing)
/// * `config` — Model configuration
pub fn forward_hla<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerHlaCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);

    // Pre-allocate temp buffers once (reused across layers)
    // Size: head_dim floats each — negligible for hd=4..16
    let mut tmp_k_cqv = vec![0.0f32; hd];
    let mut tmp_u = vec![0.0f32; hd];

    // 1. Embedding: x = wte[token] + wpe[pos]
    // SIMD-accelerated elementwise add (was an unchecked manual loop).
    let tok_off = token * n;
    let pos_off = pos * n;
    simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off..pos_off + n],
    );

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_state = &mut cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        types::rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);

        // QKV projections (GQA: K/V produce kv_dim = n_kv_head × head_dim)
        types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // ── HLA: update + readout (replaces KV store + attention loop) ──

        // Update streaming state with new (q, k, v)
        // Cross-terms G, h use OLD CQV, mQ (correct causal ordering)
        hla_layer_update(
            layer_state,
            &ctx.q,
            &ctx.k,
            &ctx.v,
            config,
            cache.gamma,
            &mut tmp_k_cqv,
        );

        // Readout: o_t = qᵀ(SK·CQV − G) with optional normalization
        ctx.attn_out[..n].fill(0.0);
        hla_layer_readout(
            layer_state,
            &ctx.q,
            config,
            true, // normalize: divide by qᵀ(SK·mQ − h) + ε
            cache.eps,
            &mut ctx.attn_out,
            &mut tmp_u,
        );

        // ── End HLA ──

        // Output projection + residual (SIMD-accelerated)
        types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                types::matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        types::matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        // SIMD-accelerated residual add (was an unchecked manual loop).
        simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    types::matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

// ── Asymmetric AHLA Forward ────────────────────────────────────

/// Forward pass using asymmetric second-order AHLA cache.
///
/// Lower state cost than symmetric HLA: O(d·dv) instead of O(d²).
/// The attention output is computed as:
///
/// ```text
/// o_t = q_tᵀ · E_t / (q_tᵀ · n_t + ε)
/// ```
///
/// AHLA routes value through key index i (left-cascaded A·A·V),
/// capturing second-order interactions at linear attention cost.
///
/// # When to Use AHLA vs Symmetric HLA
///
/// | Criterion | Symmetric HLA | AHLA |
/// |-----------|---------------|------|
/// | State per head (hd=4) | 80 floats | 16 floats |
/// | State per head (hd=8) | 320 floats | 32 floats |
/// | Per-token cost | O(d² + d·dv) | O(d·dv) |
/// | Expressivity | Higher (data-dependent metric) | Moderate |
/// | Best for | Small hd, quality-critical | Large hd, perf-critical |
pub fn forward_ahla<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerAhlaCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let kvd = types::kv_dim(config);

    // Pre-allocate temp buffer once (reused across layers)
    let mut tmp_r = vec![0.0f32; config.head_dim];

    // 1. Embedding: x = wte[token] + wpe[pos]
    // SIMD-accelerated elementwise add (was an unchecked manual loop).
    let tok_off = token * n;
    let pos_off = pos * n;
    simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off..pos_off + n],
    );

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_state = &mut cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        types::rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);

        // QKV projections (GQA: K/V produce kv_dim)
        types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // ── AHLA: combined update + readout (replaces KV store + attention loop) ──

        ctx.attn_out[..n].fill(0.0);
        ahla_layer_step(
            layer_state,
            &ctx.q,
            &ctx.k,
            &ctx.v,
            config,
            cache.gamma,
            true, // normalize: divide by qᵀ·n + ε
            cache.eps,
            &mut ctx.attn_out,
            &mut tmp_r,
        );

        // ── End AHLA ──

        // Output projection + residual (SIMD-accelerated)
        types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                types::matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        types::matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        // SIMD-accelerated residual add (was an unchecked manual loop).
        simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    types::matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Generate tokens using symmetric HLA cache (streaming, no context window limit).
///
/// Convenience wrapper matching `generate_into()` API.
pub fn generate_hla_into(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerHlaCache,
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut crate::types::Rng,
    n_tokens: usize,
    tokens: &mut Vec<usize>,
) {
    tokens.clear();
    let mut token = config.bos_token;

    for pos in 0..n_tokens {
        let logits = forward_hla(ctx, weights, cache, token, pos, config);
        types::softmax_scaled(logits, 1.0 / config.temperature);
        let next_token = types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf);
        tokens.push(next_token);
        token = next_token;
    }
}

/// Generate tokens using asymmetric AHLA cache (streaming, no context window limit).
///
/// Convenience wrapper matching `generate_into()` API.
pub fn generate_ahla_into(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerAhlaCache,
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut crate::types::Rng,
    n_tokens: usize,
    tokens: &mut Vec<usize>,
) {
    tokens.clear();
    let mut token = config.bos_token;

    for pos in 0..n_tokens {
        let logits = forward_ahla(ctx, weights, cache, token, pos, config);
        types::softmax_scaled(logits, 1.0 / config.temperature);
        let next_token = types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf);
        tokens.push(next_token);
        token = next_token;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::Config;

    /// Helper: create random weights for testing.
    fn random_weights(config: &Config) -> TransformerWeights {
        let mut rng = crate::types::Rng::new(42);
        TransformerWeights::new(config, &mut rng)
    }

    /// Verify forward_hla produces finite logits on random weights.
    #[test]
    fn forward_hla_produces_finite_logits() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerHlaCache::new(&config);

        let logits = forward_hla(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

        assert_eq!(logits.len(), config.vocab_size);
        for &l in logits.iter() {
            assert!(l.is_finite(), "Logit should be finite: {l}");
        }
    }

    /// Verify forward_ahla produces finite logits on random weights.
    #[test]
    fn forward_ahla_produces_finite_logits() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerAhlaCache::new(&config);

        let logits = forward_ahla(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

        assert_eq!(logits.len(), config.vocab_size);
        for &l in logits.iter() {
            assert!(l.is_finite(), "Logit should be finite: {l}");
        }
    }

    /// Verify multi-token generation doesn't diverge.
    #[test]
    fn forward_hla_multi_token_stable() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerHlaCache::new(&config);
        let mut rng = crate::types::Rng::new(42);
        let mut tokens = Vec::new();

        generate_hla_into(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            &mut rng,
            16,
            &mut tokens,
        );

        assert_eq!(tokens.len(), 16);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of vocab range");
        }
    }

    /// Verify AHLA multi-token generation doesn't diverge.
    #[test]
    fn forward_ahla_multi_token_stable() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerAhlaCache::new(&config);
        let mut rng = crate::types::Rng::new(42);
        let mut tokens = Vec::new();

        generate_ahla_into(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            &mut rng,
            16,
            &mut tokens,
        );

        assert_eq!(tokens.len(), 16);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of vocab range");
        }
    }

    /// Verify cache reset allows re-generation without state leakage.
    #[test]
    fn forward_hla_reset_clean() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerHlaCache::new(&config);

        // First run
        let logits1 =
            forward_hla(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config).to_vec();

        // Reset and re-run with same input
        cache.reset();
        let logits2 =
            forward_hla(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config).to_vec();

        // Should produce identical logits after reset
        for (a, b) in logits1.iter().zip(logits2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Logits should match after reset: {a} vs {b}"
            );
        }
    }

    /// Verify AHLA state is smaller than symmetric HLA for non-trivial configs.
    #[test]
    fn ahla_memory_smaller_than_symmetric() {
        let config = Config::bpe(); // hd=8, n_head=4, n_kv_head=2
        let hla_cache = MultiLayerHlaCache::new(&config);
        let ahla_cache = MultiLayerAhlaCache::new(&config);

        let hla_bytes = hla_cache.memory_bytes();
        let ahla_bytes = ahla_cache.memory_bytes();

        assert!(
            ahla_bytes < hla_bytes,
            "AHLA ({ahla_bytes}B) should be smaller than HLA ({hla_bytes}B)"
        );
    }

    /// Verify forward_hla works with all standard configs.
    #[test]
    fn forward_hla_all_configs() {
        for config in [
            Config::micro(),
            Config::game(),
            Config::bpe(),
            Config::gqa_draft(),
        ] {
            let weights = random_weights(&config);
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerHlaCache::new(&config);

            let logits = forward_hla(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

            assert_eq!(
                logits.len(),
                config.vocab_size,
                "Config vocab={vocab_size}: wrong logits len",
                vocab_size = config.vocab_size
            );
            for (i, &l) in logits.iter().enumerate() {
                assert!(l.is_finite(), "Config logits[{i}] not finite: {l}");
            }
        }
    }

    /// Verify forward_ahla works with GQA config (n_head=8, n_kv_head=2).
    #[test]
    fn forward_ahla_gqa_draft() {
        let config = Config::gqa_draft(); // n_head=8, n_kv_head=2
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerAhlaCache::new(&config);

        // Single token
        let logits = forward_ahla(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "AHLA GQA logits[{i}] not finite: {l}");
        }

        // Multi-token streaming
        let mut rng = crate::types::Rng::new(42);
        let mut tokens = Vec::new();
        generate_ahla_into(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            &mut rng,
            16,
            &mut tokens,
        );
        assert_eq!(tokens.len(), 16);
        for &t in &tokens {
            assert!(
                t < config.vocab_size,
                "AHLA GQA token {t} out of vocab range"
            );
        }
    }
}
