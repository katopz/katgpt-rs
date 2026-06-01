//! Forward pass implementations for Gated DeltaNet-2 (GDN2) recurrent attention.
//!
//! Implements O(1) per-token decode with decoupled erase/write gates, replacing
//! the growing KV cache with a fixed-size recurrent state matrix S ∈ R^{d_k × d_v}
//! per KV head. The core recurrence is:
//!
//! ```text
//! S ← Diag(α) · S + k ⊗ (w⊙v − Sᵀ(b⊙k))
//! o = Sᵀ q
//! ```
//!
//! # Gate Configurations
//!
//! | Config | Gates | State | Notes |
//! |--------|-------|-------|-------|
//! | `EraseOnly` | b (channel), w (scalar) | dk×dv | ~90% of full gain |
//! | `Full` | b (channel), w (channel) | dk×dv | Full GDN2 |
//! | `Kda` | β (scalar, tied) | dk×dv | KDA baseline |
//!
//! Reference: "Gated DeltaNet" (2024). See `.research/` for derivation.
//! Plan 105: GDN2 module.

use crate::gdn2::kernel::{gdn2_state_readout, gdn2_state_update, l2_normalize};
use crate::gdn2::types::MultiLayerGdn2Cache;
use crate::transformer::{ForwardContext, TransformerWeights};
use crate::types::{self, Config};

// ── GDN2 Forward Pass ──────────────────────────────────────────

/// Forward pass using GDN2 recurrent attention.
///
/// Replaces the growing KV cache with constant-size O(d_k × d_v) recurrent state.
/// The attention output is computed via a single recurrent step per head:
///
/// ```text
/// S ← Diag(α) · S + k ⊗ (w⊙v − Sᵀ(b⊙k))   [update]
/// o = Sᵀ q                                    [readout]
/// ```
///
/// # vs `forward_base()`
///
/// | Step | `forward_base` | `forward_gdn2` |
/// |------|----------------|----------------|
/// | QKV projection | Same | Same + L2 normalize q, k |
/// | Cache store | `layer_cache.key[pos] = k` | **Skip** (state update instead) |
/// | Attention | Loop over past positions O(N·d) | **Recurrent step** O(d²) constant |
/// | Gates | None | Erase b, write w, decay α |
/// | MLP | Same | Same |
/// | Memory per layer | O(block_size × kv_dim) growing | O(d_k × d_v) constant |
///
/// # Arguments
/// * `ctx` — Pre-allocated forward context with Q/K/V buffers
/// * `weights` — Transformer weights (same as `forward_base`)
/// * `cache` — GDN2 cache (constant-size, no `block_size` dependency)
/// * `token` — Input token index
/// * `pos` — Position in sequence (used for position embedding only, not cache sizing)
/// * `config` — Model configuration
pub fn forward_gdn2<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerGdn2Cache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let gate_config = cache
        .layers
        .first()
        .map(|l| l.gate_config)
        .unwrap_or_default();

    // Default gate values for MVP (no learned gate projections yet)
    let write_w_scalar = 1.0f32; // scalar w for EraseOnly/Kda

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) =
                *weights.wte.get_unchecked(tok_off + i) + *weights.wpe.get_unchecked(pos_off + i);
        }
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        types::rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        types::rmsnorm(&mut ctx.x);

        // QKV projections (GQA: K/V produce kv_dim = n_kv_head × head_dim)
        types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // L2 normalize q and k (stability requirement for recurrent attention)
        for h in 0..config.n_head {
            l2_normalize(&mut ctx.q[h * hd..(h + 1) * hd]);
        }
        for h in 0..config.n_kv_head {
            l2_normalize(&mut ctx.k[h * hd..(h + 1) * hd]);
        }

        // ── GDN2: recurrent step (replaces KV store + attention loop) ──
        // With GQA, state update is identical for heads sharing a KV group.
        // Split into: (1) update state once per KV group, (2) readout per Q head.

        // Update state: once per KV group (decay + read + outer-product)
        for g in 0..config.n_kv_head {
            let s = &mut layer_cache.heads[g].s;
            let k_h = &ctx.k[g * hd..(g + 1) * hd];
            let v_h = &ctx.v[g * hd..(g + 1) * hd];
            gdn2_state_update(
                s,
                k_h,
                v_h,
                &layer_cache.decay_alpha,
                &layer_cache.erase_b,
                write_w_scalar,
                &layer_cache.write_w_channel,
                &mut layer_cache.temp_buf,
                &mut layer_cache.delta,
                hd,
                hd,
                gate_config,
            );
        }

        // Readout: once per Q head (state is shared within KV group)
        for h in 0..config.n_head {
            let kv_group = h * config.n_kv_head / config.n_head;
            let s = &layer_cache.heads[kv_group].s;
            let q_h = &ctx.q[h * hd..(h + 1) * hd];
            gdn2_state_readout(s, q_h, &mut layer_cache.out_buf, hd, hd);
            ctx.attn_out[h * hd..(h + 1) * hd].copy_from_slice(&layer_cache.out_buf);
        }

        // ── End GDN2 ──

        // Output projection + residual
        types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

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
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
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

// ── Streaming Generation ───────────────────────────────────────

/// Generate tokens using GDN2 cache (streaming, no context window limit).
///
/// Convenience wrapper matching `generate_hla_into()` API.
pub fn generate_gdn2_into(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerGdn2Cache,
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut crate::types::Rng,
    n_tokens: usize,
    tokens: &mut Vec<usize>,
) {
    tokens.clear();
    tokens.reserve(n_tokens);
    let mut token = config.bos_token;

    for pos in 0..n_tokens {
        let logits = forward_gdn2(ctx, weights, cache, token, pos, config);
        types::softmax_scaled(logits, 1.0 / config.temperature);
        let next_token = types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf);
        tokens.push(next_token);
        token = next_token;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn2::types::Gdn2GateConfig;
    use crate::types::Config;

    /// Helper: create random weights for testing.
    fn random_weights(config: &Config) -> TransformerWeights {
        let mut rng = crate::types::Rng::new(42);
        TransformerWeights::new(config, &mut rng)
    }

    /// Verify forward_gdn2 produces finite logits on random weights.
    #[test]
    fn forward_gdn2_produces_finite_logits() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);

        let logits = forward_gdn2(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "Logit[{i}] should be finite: {l}");
        }
    }

    /// Verify multi-token generation doesn't diverge.
    #[test]
    fn forward_gdn2_multi_token_stable() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);
        let mut rng = crate::types::Rng::new(42);
        let mut tokens = Vec::new();

        generate_gdn2_into(
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
    fn forward_gdn2_reset_clean() {
        let config = Config::micro();
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);

        // First run
        let logits1 =
            forward_gdn2(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config).to_vec();

        // Reset and re-run with same input
        cache.reset();
        let logits2 =
            forward_gdn2(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config).to_vec();

        // Should produce identical logits after reset
        for (i, (a, b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Logits[{i}] should match after reset: {a} vs {b}"
            );
        }
    }

    /// Verify forward_gdn2 works with all standard configs.
    #[test]
    fn forward_gdn2_all_configs() {
        for config in [
            Config::micro(),
            Config::game(),
            Config::bpe(),
            Config::gqa_draft(),
        ] {
            let weights = random_weights(&config);
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerGdn2Cache::new(&config);

            let logits = forward_gdn2(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

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

    /// Verify forward_gdn2 works with GQA config (n_head=8, n_kv_head=2).
    #[test]
    fn forward_gdn2_gqa_draft() {
        let config = Config::gqa_draft(); // n_head=8, n_kv_head=2
        let weights = random_weights(&config);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerGdn2Cache::new(&config);

        // Single token
        let logits = forward_gdn2(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "GDN2 GQA logits[{i}] not finite: {l}");
        }

        // Multi-token streaming
        let mut rng = crate::types::Rng::new(42);
        let mut tokens = Vec::new();
        generate_gdn2_into(
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
                "GDN2 GQA token {t} out of vocab range"
            );
        }
    }

    /// Verify different gate configs produce finite output.
    #[test]
    fn forward_gdn2_all_gate_configs() {
        for gate_config in [
            Gdn2GateConfig::EraseOnly,
            Gdn2GateConfig::Full,
            Gdn2GateConfig::Kda,
        ] {
            let config = Config::micro();
            let weights = random_weights(&config);
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerGdn2Cache::new(&config);
            // Override gate config
            for layer in &mut cache.layers {
                layer.gate_config = gate_config;
            }

            let logits = forward_gdn2(&mut ctx, &weights, &mut cache, config.bos_token, 0, &config);

            assert_eq!(logits.len(), config.vocab_size);
            for (i, &l) in logits.iter().enumerate() {
                assert!(
                    l.is_finite(),
                    "Gate {gate_config:?}: logits[{i}] not finite: {l}"
                );
            }
        }
    }
}
