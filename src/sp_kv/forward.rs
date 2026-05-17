//! SP-KV forward pass: gated attention with utility-based bias.
//!
//! `attention_head_gated()` is a drop-in replacement for `attention_head()` that adds
//! an optional per-position gate bias to Q·K scores before softmax.
//!
//! ## Gate Bias Semantics
//!
//! | Mode | bias[s] | Effect |
//! |------|---------|--------|
//! | `None` | — | Identical to `attention_head()` |
//! | Soft (training) | `log(u_s + ε)` | Smooth, differentiable masking |
//! | Hard (inference) | `0` or `-∞` | Binary retain/prune |
//! | TAHG (annealing) | Blended | Gradual hardening |
//!
//! Positions within the sliding window always get bias = 0 (unconditionally attended).

use crate::simd::simd_dot_f32;

/// Pre-allocated context for SP-KV forward passes.
///
/// Holds gate bias buffer and utility predictor scratch buffer.
/// Create once, reuse across calls. Zero alloc in hot path.
pub struct SpKvForwardContext {
    /// Gate bias buffer: [block_size]. Rebuilt every forward call.
    gate_bias: crate::sp_kv::types::GateBiasBuffer,
    /// Utility predictor scratch: [predictor_hidden]. Reused across layers.
    predictor_buf: Vec<f32>,
    /// Per-head utilities: [n_kv_heads]. Reused across layers.
    head_utilities: Vec<f32>,
}

impl SpKvForwardContext {
    /// Create SP-KV forward context for the given model and SP-KV config.
    pub fn new(
        config: &crate::types::Config,
        sp_kv_config: &crate::sp_kv::types::SpKvConfig,
    ) -> Self {
        let hidden = sp_kv_config.predictor_hidden.max(16);
        let n_kv_heads = config.n_kv_head;
        Self {
            gate_bias: crate::sp_kv::types::GateBiasBuffer::new(config.block_size),
            predictor_buf: vec![0.0; hidden],
            head_utilities: vec![0.0; n_kv_heads],
        }
    }
}

/// Fused attention head with GQA support and optional SP-KV gate bias.
///
/// Identical to `attention_head()` in `transformer.rs` when `gate_bias` is `None`.
/// When `gate_bias` is `Some(&[bias])`, adds `bias[t]` to each Q·K score:
///
/// ```text
/// score = dot(q, k[t]) * scale + gate_bias[t]
/// ```
///
/// This is the **single insertion point** for SP-KV in the attention kernel.
/// Everything else (softmax, weighted accumulation) remains unchanged.
///
/// ## SAFETY
///
/// Caller must ensure all indices are in bounds:
/// - `q[q_head_offset..q_head_offset + hd]` valid
/// - `key_cache[t * kv_dim + kv_group_offset..t * kv_dim + kv_group_offset + hd]` valid for t in 0..t_n
/// - `value_cache` same layout as `key_cache`
/// - `attn_out[q_head_offset..q_head_offset + hd]` valid
/// - `scores_buf[..t_n]` valid
/// - `gate_bias` if `Some`: `gate_bias[..t_n]` valid
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn attention_head_gated(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    gate_bias: Option<&[f32]>,
) {
    // Pass 1: compute Q·K scores with optional gate bias, find max for numerical stability
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        // SAFETY: caller guarantees bounds (see doc comment)
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), hd);
            let k_slice = std::slice::from_raw_parts(key_cache.as_ptr().add(k_off), hd);
            simd_dot_f32(q_slice, k_slice, hd)
        };
        let raw_score = dot * scale;

        // SP-KV gate bias: additive bias per key position
        let score = match gate_bias {
            Some(bias) => {
                // SAFETY: caller guarantees bias[..t_n] valid
                unsafe { raw_score + *bias.get_unchecked(t) }
            }
            None => raw_score,
        };

        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        if score > max_score {
            max_score = score;
        }
    }

    // Pass 2: exp(scores - max) and accumulate sum (unchanged from attention_head)
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation (unchanged from attention_head)
    let inv_sum = 1.0 / sum;
    for d in 0..hd {
        let mut val = 0.0f32;
        for t in 0..t_n {
            unsafe {
                val += *scores_buf.get_unchecked(t)
                    * inv_sum
                    * *value_cache.get_unchecked(t * kv_dim + kv_group_offset + d);
            }
        }
        unsafe {
            *attn_out.get_unchecked_mut(q_head_offset + d) = val;
        }
    }
}

/// Build per-position gate biases for all positions up to `pos`.
///
/// Convenience function that dispatches to the appropriate mode:
/// - Soft: `log(u_s + ε)` outside window, `0` inside
/// - Hard: `0` if retained or in window, `-∞` otherwise
/// - TAHG: blended soft/hard with annealing
///
/// Writes into `buf.bias[..=pos]`. Positions after `pos` are left unchanged.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn build_gate_biases(
    buf: &mut crate::sp_kv::types::GateBiasBuffer,
    utilities: &[f32],
    retained: &[bool],
    pos: usize,
    window: usize,
    threshold: f32,
    mode: crate::sp_kv::types::SpKvGateMode,
) {
    use crate::sp_kv::types::SpKvGateMode;

    match mode {
        SpKvGateMode::Soft => buf.build_soft(utilities, pos, window),
        SpKvGateMode::Hard => buf.build_hard(utilities, retained, pos, window, threshold),
        SpKvGateMode::Tahg { step, total_steps } => {
            buf.build_tahg(utilities, pos, window, threshold, step, total_steps);
        }
    }
}

/// Check if a position is within the sliding window from the current decode position.
#[inline(always)]
pub fn is_in_window(current_pos: usize, source_pos: usize, window: usize) -> bool {
    current_pos.saturating_sub(source_pos) < window
}

/// SP-KV forward pass: utility-gated attention with conditional KV write.
///
/// Mirrors `forward_base()` from `transformer.rs` with three SP-KV modifications:
///
/// 1. **Utility prediction** (after RMSNorm, before QKV): 2-layer MLP predicts per-KV-head utility
/// 2. **Conditional KV write**: only store if `utility ≥ τ` or position is in sliding window
/// 3. **Gate-biased attention**: `attention_head_gated()` adds `log(u)` bias to Q·K scores
///
/// Pipeline composability: PFlash (prefill) → SP-KV (decode) → TurboQuant (storage).
///
/// # Arguments
///
/// * `ctx` — Pre-allocated forward context (from `transformer.rs`)
/// * `weights` — Transformer weights (embeddings, per-layer, LM head)
/// * `sp_cache` — SP-KV sparse KV cache (replaces `MultiLayerKVCache`)
/// * `predictors` — Per-layer utility predictor weights
/// * `sp_ctx` — SP-KV forward context (gate bias buffer, predictor scratch)
/// * `token` — Current input token index
/// * `pos` — Current sequence position
/// * `config` — Model configuration
/// * `lora` — Optional LoRA adapter
/// * `gate_mode` — Soft (training), Hard (inference), or TAHG (annealing)
/// * `domain_latent` — Optional domain latent for mid-layer injection (feature-gated)
///
/// # Returns
///
/// Mutable reference to logits buffer `ctx.logits`.
///
/// # When to Use
///
/// During **decode** (autoregressive generation) with SP-KV enabled.
/// Prefill should use standard `forward_base()` or PFlash.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[inline(always)]
pub fn forward_sp_kv<'a>(
    ctx: &'a mut crate::transformer::ForwardContext,
    weights: &crate::transformer::TransformerWeights,
    sp_cache: &mut crate::sp_kv::types::SpKvCache,
    predictors: &crate::sp_kv::types::SpKvPredictors,
    sp_ctx: &mut SpKvForwardContext,
    token: usize,
    pos: usize,
    config: &crate::types::Config,
    lora: Option<&crate::types::LoraAdapter>,
    gate_mode: crate::sp_kv::types::SpKvGateMode,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    use crate::types::{kv_dim, matmul, matmul_relu, rmsnorm};

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let n_kv = config.n_kv_head;
    let sp_config = &sp_cache.config;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    for i in 0..n {
        unsafe {
            *ctx.x.get_unchecked_mut(i) = *weights.wte.get_unchecked(tok_off + i)
                + *weights.wpe.get_unchecked(pos_off_emb + i);
        }
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut sp_cache.layers[layer_idx];

        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // ── SP-KV: Predict utility from hidden state (after RMSNorm, before QKV) ──
        let predictor = &predictors.layers[layer_idx];
        sp_ctx.head_utilities = crate::sp_kv::utility_predictor::predict(
            predictor,
            &ctx.x,
            n,
            sp_config.predictor_hidden,
            n_kv,
            &mut sp_ctx.predictor_buf,
        );

        // Aggregate per-head utilities to single scalar for cache write decision
        let pos_utility = crate::sp_kv::utility_predictor::aggregate_utilities(
            &sp_ctx.head_utilities,
            crate::sp_kv::utility_predictor::UtilityAggregation::Max,
        );

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            for i in 0..kvd {
                unsafe {
                    *ctx.k.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                    *ctx.v.get_unchecked_mut(i) += *dl.embedding.get_unchecked(i);
                }
            }
        }

        // ── SP-KV: Conditional KV write (skip positions with low utility) ──
        // Current position is always in window (distance 0 < window).
        layer_cache.write_gated(
            &ctx.k,
            &ctx.v,
            pos_utility,
            pos,
            true, // current pos always in window
            sp_config.threshold,
            kvd,
        );

        // ── SP-KV: Build gate biases for all past positions ──
        build_gate_biases(
            &mut sp_ctx.gate_bias,
            &layer_cache.utilities,
            &layer_cache.retained,
            pos,
            sp_config.window,
            sp_config.threshold,
            gate_mode,
        );

        // Multi-head attention with GQA + SP-KV gate bias
        let scale = 1.0 / (hd as f32).sqrt();
        ctx.attn_out[..n].fill(0.0);
        let t_n = pos + 1;

        for h in 0..config.n_head {
            let kv_group = h * n_kv / config.n_head;
            unsafe {
                attention_head_gated(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                    Some(&sp_ctx.gate_bias.bias),
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // MLP w2: sparse when feature enabled and sparsity is high enough
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = crate::types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut ctx.lora_buf);
        }
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }
    }

    // Snapshot hidden state (Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: standard matmul (clustered LM head not yet wired — TODO: expose as pub(crate))
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

// ---------------------------------------------------------------------------
// SP-KV + TurboQuant Fusion (T12) — stub for Plan 070 Phase 3
// ---------------------------------------------------------------------------

/// SP-KV + TurboQuant fused forward: selective write + lossy quantize.
///
/// Two-stage KV compression pipeline:
/// 1. **SP-KV selective write**: utility predictor decides which positions to retain
/// 2. **TurboQuant quantize**: retained positions are compressed (f32 → 2-4 bits)
///
/// This is the maximal-compression decode path:
/// ```text
/// Prefill: PFlash (block-sparse token selection)
/// Decode:  SP-KV (selective write) → TurboQuant (lossy quantize retained)
/// Result:  only useful KV pairs kept, those compressed to 2-4 bits/coord
/// ```
///
/// ## Implementation Strategy
///
/// `forward_sp_kv_tq()` mirrors `forward_sp_kv()` but replaces `SpKvCache` with
/// a hybrid cache that combines:
/// - `SpKvLayerCache.retained[]` bitfield (which positions to keep)
/// - `TurboQuantKVCache` (compressed storage for retained positions)
///
/// For retained positions: `TurboQuantKVCache::store_key/value()` quantizes in-place.
/// For pruned positions: no write (cache stays at default).
/// During attention: dequantize only retained positions into flat buffer, then
/// `attention_head_gated()` with hard gate bias.
///
/// ## TODO (Phase 3)
///
/// - [ ] Create `SpKvTqCache` hybrid type wrapping `SpKvLayerCache` + `TurboQuantKVCache`
/// - [ ] Implement `forward_sp_kv_tq()` with dual cache
/// - [ ] Benchmark: density × compression ratio vs standalone SP-KV and TQ
/// - [ ] Wire into `forward()` dispatch as `AttentionMode::SpKvTq`
///
/// ## Estimated Compression
///
/// At τ=0.5 (~30% density) + 3-bit TQ: ~10.7 bits/position vs 32-bit baseline = **3× compression**.
/// At τ=0.7 (~11% density) + 3-bit TQ: ~33 bits/position vs 32-bit baseline = **~29× compression**.
#[allow(dead_code)]
fn forward_sp_kv_tq() {
    // Stub — will be implemented in Phase 3 when TurboQuant integration is wired.
    // See Plan 070, Task T12.
    todo!("forward_sp_kv_tq: SP-KV + TurboQuant fusion (Plan 070 Phase 3)")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline: attention_head_gated with gate_bias=None matches standard attention.
    /// Compares against a manual reference implementation.
    #[test]
    fn test_gated_none_matches_baseline() {
        let hd = 8;
        let kv_dim = 16; // 2 KV heads × 8 head_dim
        let t_n = 4;

        let q = vec![0.1; kv_dim];
        let key_cache = vec![0.2; t_n * kv_dim];
        let value_cache = vec![0.3; t_n * kv_dim];
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out_gated = vec![0.0; kv_dim];
        let mut scores_gated = vec![0.0; t_n];

        let mut attn_out_baseline = vec![0.0; kv_dim];
        let mut scores_baseline = vec![0.0; t_n];

        unsafe {
            // Gated with None
            attention_head_gated(
                &q,
                &key_cache,
                &value_cache,
                &mut attn_out_gated,
                &mut scores_gated,
                0,
                0,
                kv_dim,
                hd,
                t_n,
                scale,
                None,
            );

            // Manual baseline: same as gated but we compute manually
            for h_off in [0] {
                let mut max_s = f32::NEG_INFINITY;
                for t in 0..t_n {
                    let k_off = t * kv_dim + h_off;
                    let dot: f32 = q[h_off..h_off + hd]
                        .iter()
                        .zip(&key_cache[k_off..k_off + hd])
                        .map(|(&a, &b)| a * b)
                        .sum();
                    let score = dot * scale;
                    scores_baseline[t] = score;
                    if score > max_s {
                        max_s = score;
                    }
                }
                let sum: f32 = scores_baseline.iter().map(|s| (s - max_s).exp()).sum();
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let v: f32 = (0..t_n)
                        .map(|t| {
                            let exp = (scores_baseline[t] - max_s).exp();
                            exp * inv * value_cache[t * kv_dim + h_off + d]
                        })
                        .sum();
                    attn_out_baseline[h_off + d] = v;
                }
            }
        }

        for d in 0..hd {
            assert!(
                (attn_out_gated[d] - attn_out_baseline[d]).abs() < 1e-4,
                "Mismatch at d={d}: gated={gated}, baseline={baseline}",
                gated = attn_out_gated[d],
                baseline = attn_out_baseline[d],
            );
        }
    }

    /// Gate bias = -inf should zero out attention weight for that position.
    #[test]
    fn test_hard_gate_prunes_position() {
        let hd = 4;
        let kv_dim = 4;
        let t_n = 4;
        let scale = 1.0 / (hd as f32).sqrt();

        let q = vec![1.0; kv_dim];
        let key_cache = vec![1.0; t_n * kv_dim];
        let value_cache = vec![1.0; t_n * kv_dim];

        // Gate: keep positions 0,1,3; prune position 2
        let gate_bias = vec![0.0, 0.0, f32::NEG_INFINITY, 0.0];

        let mut attn_out = vec![0.0; kv_dim];
        let mut scores = vec![0.0; t_n];

        unsafe {
            attention_head_gated(
                &q,
                &key_cache,
                &value_cache,
                &mut attn_out,
                &mut scores,
                0,
                0,
                kv_dim,
                hd,
                t_n,
                scale,
                Some(&gate_bias),
            );
        }

        // All value_cache entries are 1.0, so attn_out[d] should be 1.0
        // (weighted average of 1.0s across non-pruned positions)
        for d in 0..hd {
            assert!(
                (attn_out[d] - 1.0).abs() < 1e-4,
                "Expected ~1.0 at d={d}, got {v}",
                v = attn_out[d],
            );
        }

        // Verify pruned position has zero attention weight
        // (positions 0,1,3 each get ~1/3 weight)
        let total_weight: f32 = scores[0] + scores[1] + scores[3];
        assert!(total_weight > 0.0, "Non-pruned weights should be positive");
        assert!(
            scores[2] < 1e-20,
            "Pruned position should have ~0 weight, got {w}",
            w = scores[2],
        );
    }

    /// Window positions should have zero bias regardless of utility.
    #[test]
    fn test_is_in_window() {
        assert!(is_in_window(10, 5, 8)); // 10-5=5 < 8
        assert!(is_in_window(10, 9, 8)); // 10-9=1 < 8
        assert!(!is_in_window(10, 2, 8)); // 10-2=8, NOT < 8
        assert!(is_in_window(5, 0, 128)); // 5-0=5 < 128
    }
}
