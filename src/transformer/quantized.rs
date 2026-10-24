use super::*;
use crate::types::{self};

/// Forward pass using quantized KV cache (Plan 043, generalized Plan 063).
///
/// Mirrors [`forward_base`] but stores K/V into a compressed cache and
/// dequantizes on-the-fly during attention scoring. The rest of the
/// transformer (embedding, QKV projection, MLP, LM head) is unchanged.
///
/// Generic over any [`types::QuantizedKVCache`] backend (SpectralQuant, TurboQuant, etc.).
///
/// **Trade-off**: ~8× KV cache memory savings at the cost of dequantization
/// overhead during attention. Best for long sequences where cache memory
/// dominates.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_quantized<'a, C: types::QuantizedKVCache>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut C,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    katgpt_core::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // Adaptive Depth Tier: cap layer count at inference time (Plan 284 T10).
    let max_layer = ctx
        .depth_tier
        .map_or(config.n_layer, |t| t.max_layers(config.n_layer));

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate().take(max_layer) {
        // Pre-attention: RMSNorm → save residual
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Issue 960 T0b: raw-QKV calibration capture (`kv_capture`) — the raw
        // projections before the quantizing store, exactly what attention would
        // consume on the lossless path.
        #[cfg(feature = "kv_capture")]
        if let Some(cap) = ctx.kv_capture.as_mut() {
            cap.push(layer_idx, pos, &ctx.q[..n], &ctx.k[..kvd], &ctx.v[..kvd]);
        }

        // Store compressed K,V
        cache.store_key(layer_idx, pos, &ctx.k[..kvd]);
        cache.store_value(layer_idx, pos, &ctx.v[..kvd]);

        // Incremental dequant (Plan 068): only dequant the new position when possible.
        // Tracks per-layer progress: if tq_dequant_pos[layer] == pos - 1, the flat buffer
        // already contains positions 0..pos-1 from the previous decode step for this layer.
        // On mismatch (first call, layer switch, reset, pos jump), rebuild all positions.
        // t_n is hoisted outside the layer loop (loop-invariant).
        let last_pos = ctx.dequant_pos[layer_idx];
        if last_pos + 1 == pos && pos > 0 {
            // Incremental: only dequant the new position
            cache.dequantize_key_into(
                layer_idx,
                pos,
                &mut ctx.paged_flat_key[pos * kvd..(pos + 1) * kvd],
            );
            cache.dequantize_value_into(
                layer_idx,
                pos,
                &mut ctx.paged_flat_value[pos * kvd..(pos + 1) * kvd],
            );
        } else {
            // Full rebuild: dequantize all positions (first call, reset, or pos jump)
            for t in 0..t_n {
                cache.dequantize_key_into(
                    layer_idx,
                    t,
                    &mut ctx.paged_flat_key[t * kvd..(t + 1) * kvd],
                );
                cache.dequantize_value_into(
                    layer_idx,
                    t,
                    &mut ctx.paged_flat_value[t * kvd..(t + 1) * kvd],
                );
            }
        }
        ctx.dequant_pos[layer_idx] = pos;

        // Multi-head attention with GQA using dequantized flat cache
        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h] as usize;
            unsafe {
                attention_head(
                    &ctx.q,
                    &ctx.paged_flat_key,
                    &ctx.paged_flat_value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        #[cfg(feature = "gated_mlp")]
        {
            // SwiGLU: SiLU(W_gate·h) ⊙ W_up·h → W_down·hidden
            types::matmul(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx.hidden2,
                &layer_weights.mlp_w_up,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            types::swiglu_inplace(&mut ctx.hidden, &ctx.hidden2);
        }
        #[cfg(not(feature = "gated_mlp"))]
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2 (W_down): sparse when feature enabled and sparsity is high enough (Plan 022)
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
        katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
            if block_idx < ctx.block_deltas.len() {
                katgpt_core::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head (uses matmul_parallel for large vocab)
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Backward-compat alias: forward using TurboQuant-specific cache.
///
/// Prefer [`forward_quantized`] for new code — it's generic over any
/// [`types::QuantizedKVCache`] backend.
#[cfg(feature = "turboquant")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_turboquant<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut katgpt_quant::turboquant::TurboQuantKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    forward_quantized(ctx, weights, cache, token, pos, config)
}
