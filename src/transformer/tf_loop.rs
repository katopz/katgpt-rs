#[cfg(feature = "tf_loop")]
use super::*;
#[cfg(feature = "tf_loop")]
use crate::types::{self};

// ---------------------------------------------------------------------------
// Training-Free Loop Wrapper (Plan 136, Research 94)
// ---------------------------------------------------------------------------

/// Training-free loop forward pass — ODE-refined sub-stepping over a window.
///
/// Pure inference-time retrofit: re-applies a contiguous mid-stack block of
/// layers K times with damped sub-stepping and anchor blending. No training needed.
///
/// # Algorithm (block-mode)
///
/// ```text
/// 1. Embedding: x = wte[token] + wpe[pos]
/// 2. Pre-loop:  for layer 0..window_start:  standard forward, write KV
/// 3. Anchor:    forward window once → x_anchor
/// 4. Loop K times:
///      a. Forward window layers
///      b. Sub-step: x += (1/K)·(y − x)  [damped Euler]
/// 5. Blend with anchor: x = β·x_anchor + (1−β)·x
/// 6. Stash:     single forward through window writes canonical KV
///               (`CacheStrategy::Mean` skips this pass entirely — the loop
///               iterations' running-mean K/V is written back instead,
///               Issue 698 T5)
/// 7. Post-loop: for layer window_end+1..n_layer: standard forward, write KV
/// 8. LM head
/// ```
#[cfg(feature = "tf_loop")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn forward_training_free_loop<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    tf_config: &TrainingFreeLoopConfig,
    // Plan 617 T1.3 — LoopCD guidance on the TrainingFree lane. The window
    // loop's iteration-`ref_loop` refined state (post sub-step) is the h_k
    // analog. `None` = bit-identical to pre-Plan-617 behavior.
    #[cfg(feature = "loop_guidance")] guidance: Option<&mut super::loop_guidance::LoopGuidance>,
) -> &'a mut [f32] {
    use crate::tf_loop::{anchor_blend, sub_step_damped_euler};
    use katgpt_core::types::{CacheStrategy, IterationMode, SubStepStrategy};

    cache.advance_pos(pos);

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;
    // Adaptive Depth Tier: cap effective layer count (Plan 284 T10).
    let max_layer = ctx
        .depth_tier
        .map_or(weights.layers.len(), |t| t.max_layers(config.n_layer));
    let n_layer = max_layer;
    let window_start = tf_config.window_start.min(n_layer);
    let window_end = tf_config.window_end.min(n_layer.saturating_sub(1));
    let k = tf_config.loop_count;
    let beta = match tf_config.strategy {
        SubStepStrategy::DampedEuler => 0.0, // no anchor blend for pure Euler
        SubStepStrategy::KStageRK { beta } => beta,
    };

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    katgpt_core::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // 2. Pre-loop layers: standard forward with KV writes
    for (layer_idx, layer_weights) in weights.layers[..window_start].iter().enumerate() {
        forward_single_layer(
            ctx,
            layer_weights,
            &mut cache.layers[layer_idx],
            pos,
            config,
            n,
            hd,
            kvd,
            n_kv,
        );
    }

    // Save state before window for anchor computation
    ctx.tf_x_pre_window[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Anchor: forward window once to get x_anchor
    if beta > 0.0 {
        for layer_idx in window_start..=window_end {
            forward_single_layer(
                ctx,
                &weights.layers[layer_idx],
                &mut cache.layers[layer_idx],
                pos,
                config,
                n,
                hd,
                kvd,
                n_kv,
            );
        }
        ctx.tf_x_anchor[..n].copy_from_slice(&ctx.x[..n]);
        // Restore x to pre-window state for loop iterations
        ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
    }

    // Temp buffer for window output (pre-allocated on ForwardContext).
    // No pre-zero needed: `tf_y_buf` is the only user of this buffer in the whole
    // crate (grep: written at 2 sites, read at 2 sites), and every read is
    // immediately preceded by a full-width
    // `tf_y_buf[..n].copy_from_slice(&ctx.x[..n])` in the same loop iteration —
    // in both `IterationMode` arms. Nothing after the loop reads it, so when
    // `k == 0` the buffer is never observed at all.

    // 4. Loop K times over the window with sub-stepping
    //
    // Issue 698 T5: when `cache_strategy == Mean`, each iteration's freshly
    // written K/V rows at `pos` are folded into a per-layer incremental
    // running mean (fixed order ⇒ deterministic f32 sum). The guard is
    // read-once — `First`/`Last` paths are structurally unchanged.
    let mean_kv = tf_config.cache_strategy == CacheStrategy::Mean;
    // Plan 617 T1.1 — guidance capture plumbing (see forward_looped for the
    // full contract): hoisted actives + per-call reset. The tf-loop capture
    // point is the iteration-`ref_loop` refined state (post sub-step), in
    // BOTH IterationMode arms — `Layer` mode's per-layer sub-step means the
    // capture lands at the LAST layer of the `ref_loop`-th outer iteration.
    #[cfg(feature = "loop_guidance")]
    let mut guidance = guidance;
    #[cfg(feature = "loop_guidance")]
    let guidance_armed = guidance.as_ref().is_some_and(|g| g.config.is_armed());
    #[cfg(feature = "loop_guidance")]
    let guidance_ref_loop = guidance.as_ref().map_or(0, |g| g.config.ref_loop);
    #[cfg(feature = "loop_guidance")]
    if let Some(g) = guidance.as_deref_mut() {
        g.begin_step();
    }
    #[cfg(feature = "loop_guidance")]
    macro_rules! lg_capture {
        () => {
            if guidance_armed
                && let Some(g) = guidance.as_deref_mut()
                && !g.guidance_captured()
            {
                g.capture_reference(&ctx.x[..n], &weights.lm_head, config.vocab_size, n);
            }
        };
    }
    match tf_config.iteration_mode {
        IterationMode::Block => {
            for it in 0..k {
                // Forward through window layers
                for layer_idx in window_start..=window_end {
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                    if mean_kv {
                        fold_kv_mean(
                            &cache.layers[layer_idx],
                            &mut ctx.tf_kv_mean_k,
                            &mut ctx.tf_kv_mean_v,
                            layer_idx,
                            pos,
                            kvd,
                            it + 1,
                        );
                    }
                }
                // Save window output
                ctx.tf_y_buf[..n].copy_from_slice(&ctx.x[..n]);
                // Restore x to pre-window for sub-step computation
                ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
                // Apply sub-step: x += (1/K)·(y − x)
                sub_step_damped_euler(&mut ctx.x[..n], &ctx.tf_y_buf[..n], k);
                // Plan 617 T1.1 — capture the iteration-`ref_loop` refined
                // state (post sub-step) as h_k.
                #[cfg(feature = "loop_guidance")]
                if it + 1 == guidance_ref_loop {
                    lg_capture!();
                }
            }
        }
        IterationMode::Layer => {
            for it in 0..k {
                for layer_idx in window_start..=window_end {
                    // Forward single layer
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                    if mean_kv {
                        fold_kv_mean(
                            &cache.layers[layer_idx],
                            &mut ctx.tf_kv_mean_k,
                            &mut ctx.tf_kv_mean_v,
                            layer_idx,
                            pos,
                            kvd,
                            it + 1,
                        );
                    }
                    // Sub-step per layer
                    ctx.tf_y_buf[..n].copy_from_slice(&ctx.x[..n]);
                    ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
                    sub_step_damped_euler(&mut ctx.x[..n], &ctx.tf_y_buf[..n], k);
                    // Plan 617 T1.1 — `Layer` mode captures at the LAST
                    // layer of the `ref_loop`-th outer iteration (the guard
                    // inside the macro also blocks double capture).
                    #[cfg(feature = "loop_guidance")]
                    if layer_idx == window_end && it + 1 == guidance_ref_loop {
                        lg_capture!();
                    }
                }
            }
        }
    }

    // 5. Blend with anchor
    if beta > 0.0 {
        anchor_blend(&mut ctx.x[..n], &ctx.tf_x_anchor[..n], beta);
    }

    // 6. Stash: single forward through window writes canonical KV entries
    //
    // Issue 698 T5: `Mean` skips this window-forward entirely — the loop's
    // own running-mean K/V rows are written back instead (one whole window
    // pass per token deleted). Post-loop consumption follows the `First`
    // shape: ctx.x stays the blended state.
    {
        // Mean over zero iterations is the pre-window state — First semantics.
        let cache_strategy = if k == 0 {
            CacheStrategy::First
        } else {
            tf_config.cache_strategy
        };
        match cache_strategy {
            CacheStrategy::Last => {
                // Forward with final state → writes KV
                for layer_idx in window_start..=window_end {
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                }
            }
            CacheStrategy::First => {
                // Stash the blended state, forward with pre-window state →
                // writes KV, then restore the blended state.
                ctx.tf_stash_x[..n].copy_from_slice(&ctx.x[..n]);
                ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
                for layer_idx in window_start..=window_end {
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                }
                // Restore the blended state
                ctx.x[..n].copy_from_slice(&ctx.tf_stash_x[..n]);
            }
            CacheStrategy::Mean => {
                // Write the accumulated running means (the loop already
                // produced the canonical KV) — NO window-forward here.
                for layer_idx in window_start..=window_end {
                    let mean_off = layer_idx * kvd;
                    let pos_off = pos * kvd;
                    cache.layers[layer_idx].key[pos_off..pos_off + kvd]
                        .copy_from_slice(&ctx.tf_kv_mean_k[mean_off..mean_off + kvd]);
                    cache.layers[layer_idx].value[pos_off..pos_off + kvd]
                        .copy_from_slice(&ctx.tf_kv_mean_v[mean_off..mean_off + kvd]);
                }
            }
        }
    }

    // 7. Post-loop layers: standard forward with KV writes
    for layer_idx in (window_end + 1)..n_layer {
        forward_single_layer(
            ctx,
            &weights.layers[layer_idx],
            &mut cache.layers[layer_idx],
            pos,
            config,
            n,
            hd,
            kvd,
            n_kv,
        );
    }

    // Snapshot hidden state
    // Plan 617 T1.2 — same split as forward_looped: hidden mode blends into
    // ctx.x BEFORE the snapshot + head (one head pass); logit mode applies
    // on ctx.logits after the head.
    #[cfg(feature = "loop_guidance")]
    {
        if let Some(g) = guidance.as_deref_mut() {
            let hidden = if g.config.mode == super::loop_guidance::GuidanceMode::Hidden {
                Some(&mut ctx.x[..n])
            } else {
                None
            };
            g.apply_at_readout(hidden, &mut []);
        }
    }
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 8. LM Head
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    // Plan 617 T1.2 — logit-mode contrast on the final readout / hidden-
    // mode output-margin record (see forward_looped).
    #[cfg(feature = "loop_guidance")]
    {
        // Tail use: moving the Option is fine here (no later access).
        if let Some(g) = guidance {
            if g.config.mode == super::loop_guidance::GuidanceMode::Logits {
                g.apply_at_readout(None, &mut ctx.logits);
            } else {
                g.record_output_margin(&ctx.logits);
            }
        }
    }

    &mut ctx.logits
}

/// Single transformer layer forward: attention + MLP with KV cache write.
///
/// Extracted from `forward_base` to be reusable by both standard and looped paths.
#[cfg(feature = "tf_loop")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn forward_single_layer(
    ctx: &mut ForwardContext,
    layer_weights: &LayerWeights,
    layer_cache: &mut KVCache,
    pos: usize,
    config: &Config,
    n: usize,
    hd: usize,
    kvd: usize,
    _n_kv: usize,
) {
    // Pre-attention: RMSNorm → save residual
    types::rmsnorm(&mut ctx.x);
    ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

    // QKV projections
    types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
    types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
    types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

    // Store K,V in cache
    let pos_off = pos * kvd;
    unsafe {
        std::ptr::copy_nonoverlapping(
            ctx.k.as_ptr(),
            layer_cache.key.as_mut_ptr().add(pos_off),
            kvd,
        );
        std::ptr::copy_nonoverlapping(
            ctx.v.as_ptr(),
            layer_cache.value.as_mut_ptr().add(pos_off),
            kvd,
        );
    }

    // Multi-head attention with GQA
    let scale = ctx.attn_scale;
    let t_n = pos + 1;
    for h in 0..config.n_head {
        let kv_group = ctx.kv_group_lut[h] as usize;
        unsafe {
            attention_head(
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
            );
        }
    }

    // Output projection + residual
    types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
    katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

    // MLP: save residual → RMSNorm → MLP → residual
    ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
    types::rmsnorm(&mut ctx.x);
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
    types::matmul(
        &mut ctx.x,
        &layer_weights.mlp_w2,
        &ctx.hidden,
        n,
        config.mlp_hidden,
    );
    katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
}

/// Fold the freshly written K/V rows at `pos` into the per-layer running
/// mean (Issue 698 T5).
///
/// Incremental streaming mean over the k loop iterations in fixed order —
/// a deterministic f32 sum:
///
/// ```text
/// mean ← mean + (fresh − mean) / count   (count 1-based)
/// ```
///
/// `count == 1` is the plain copy (also the staleness reset: every Mean
/// forward rewrites its window rows before the write-back, so no state can
/// leak across forwards). Zero alloc; the row is kvd-wide, negligible next
/// to the matmuls it amortises (one deleted window-forward per token).
#[cfg(feature = "tf_loop")]
#[inline]
fn fold_kv_mean(
    layer_cache: &KVCache,
    mean_k: &mut [f32],
    mean_v: &mut [f32],
    layer_idx: usize,
    pos: usize,
    kvd: usize,
    count: usize,
) {
    let pos_off = pos * kvd;
    let mean_off = layer_idx * kvd;
    let fresh_k = &layer_cache.key[pos_off..pos_off + kvd];
    let fresh_v = &layer_cache.value[pos_off..pos_off + kvd];
    let mk = &mut mean_k[mean_off..mean_off + kvd];
    let mv = &mut mean_v[mean_off..mean_off + kvd];
    if count <= 1 {
        mk.copy_from_slice(fresh_k);
        mv.copy_from_slice(fresh_v);
        return;
    }
    let inv = 1.0f32 / count as f32;
    for i in 0..kvd {
        mk[i] += (fresh_k[i] - mk[i]) * inv;
    }
    for i in 0..kvd {
        mv[i] += (fresh_v[i] - mv[i]) * inv;
    }
}

/// Delta routing: softmax over delta sources, additive to residual (Plan 097).
///
/// depth_route(sources, residual, proj, norm):
///   V = stack(sources)          // [N, D]
///   K = norm(V)                  // RMSNorm
///   logits = dot(proj_weight, K) // per-source score
///   weights = softmax(logits)    // routing weights
///   return residual + weighted_sum(weights, V)  // additive
///
/// ## Stability analysis (Plan 134, MGR paper §3.2 — arXiv:2605.23259)
///
/// The MGR paper proves that convex-combination residual updates (lerp gates)
/// guarantee bounded activation norms: `x_{l+1} = (1-α)·x_l + α·f(x_l)`.
///
/// **Our routing is NOT a convex combination.** It is additive:
/// `residual += Σ_i w_i · V_i`, where `w_i = softmax(...)` and `Σ w_i = 1`.
/// Since softmax weights sum to 1 but are applied to arbitrary source vectors (not
/// the residual itself), the MGR convex-combination stability guarantee does not
/// formally apply.
///
/// Practical stability comes from two normalization mechanisms:
/// - **RMSNorm** bounds the input scale to the routing logits, preventing
///   exploding score magnitudes.
/// - **Softmax normalization** ensures routing weights are non-negative and sum
///   to 1, so the weighted sum cannot exceed the convex hull of source vectors.
///
/// Unlike MGR's convex lerp, norms *can* still grow layer-to-layer (each additive
/// step contributes additional magnitude). However, empirical testing across 36+
/// layers shows bounded growth: `‖x_L‖ ≤ 10 × ‖x_0‖` (see
/// `proof_depth_route_norm_stability` test).
///
/// ## MGR Eq. 14 — lerp gate bias initialization
///
/// If a convex-combination lerp gate were ever added (e.g. for training), the
/// MGR paper recommends initializing the gate bias as:
///
///   b_l = log(1 - 1/L)
///
/// where L is the total number of layers. For L=36, b_l ≈ -0.0285.
/// This encourages near-identity routing at initialization.
#[cfg(feature = "delta_routing")]
#[allow(dead_code, clippy::needless_range_loop)]
#[inline(always)]
pub(crate) fn depth_route(
    residual: &mut [f32],
    sources: &[&[f32]],     // N delta vectors, each [n_embd]
    query_weight: &[f32],   // [n_embd] per-layer query
    norm_weight: &[f32],    // [n_embd] RMSNorm gamma
    logits_buf: &mut [f32], // [N] temp buffer
    scaled_buf: &mut [f32], // [n_embd] scratch for SIMD dot
    n_embd: usize,
) {
    let n_sources = sources.len();
    if n_sources == 0 {
        return;
    }

    // 1. RMSNorm each source and compute dot product with query
    let eps = 1e-5f32;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &src) in sources.iter().enumerate() {
        // SIMD sum-of-squares for RMSNorm
        let sum_sq = katgpt_core::simd::simd_sum_sq(&src[..n_embd], n_embd);
        let rms = (sum_sq / n_embd as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale src * inv_rms * norm_weight into scratch via fused SIMD, then dot with query
        scaled_buf[..n_embd].copy_from_slice(&src[..n_embd]);
        katgpt_core::simd::simd_scale_mul_inplace(
            &mut scaled_buf[..n_embd],
            &norm_weight[..n_embd],
            inv_rms,
        );
        let logit = katgpt_core::simd::simd_dot_f32(&scaled_buf[..n_embd], query_weight, n_embd);

        logits_buf[i] = logit;
        // Branch-free max reduction: f32::max compiles to a single instruction
        // (vmaxss on x86-64 SSE, fmax on AArch64 NEON). Avoids predicted-branch
        // mispredicts when logits are similar (typical for well-normalized sources).
        max_logit = max_logit.max(logit);
    }

    // 2. Softmax (numerically stable, SIMD batch)
    katgpt_core::simd::simd_add_scalar_inplace(&mut logits_buf[..n_sources], -max_logit);
    katgpt_core::simd::simd_exp_inplace(&mut logits_buf[..n_sources]);
    let sum_exp = katgpt_core::simd::simd_sum_f32(&logits_buf[..n_sources]);
    let inv_sum = 1.0 / sum_exp;

    // 3. Weighted sum of sources, added to residual (additive routing).
    //    Fused into a single SIMD pass: residual[i] += src[i] * weight.
    //    Eliminates the scaled_buf copy + separate scale + add passes.
    for (i, &src) in sources.iter().enumerate() {
        let weight = logits_buf[i] * inv_sum;
        katgpt_core::simd::simd_fused_scale_acc(
            &mut residual[..n_embd],
            &src[..n_embd],
            weight,
            n_embd,
        );
    }
}

// re-exported at the top of this file. See the Phase F block above.)

/// Compute delta routing softmax weights without modifying residual (Plan 097 T8).
///
/// Returns the routing weight distribution over sources for inspection.
/// Used by GOAT sharpness tests to verify max_weight ≥ 0.4 in deep layers.
#[cfg(feature = "delta_routing")]
#[allow(clippy::needless_range_loop)]
pub fn depth_route_weights(
    sources: &[&[f32]],   // N delta vectors, each [n_embd]
    query_weight: &[f32], // [n_embd] per-layer query
    norm_weight: &[f32],  // [n_embd] RMSNorm gamma
    n_embd: usize,
) -> Vec<f32> {
    let n_sources = sources.len();
    if n_sources == 0 {
        return Vec::new();
    }

    let eps = 1e-5f32;
    let mut logits = vec![0.0f32; n_sources];
    let mut scaled = vec![0.0f32; n_embd];
    let mut max_logit = f32::NEG_INFINITY;

    // 1. RMSNorm each source and compute dot product with query
    for (i, &src) in sources.iter().enumerate() {
        // SIMD sum-of-squares for RMSNorm
        let sum_sq = katgpt_core::simd::simd_sum_sq(&src[..n_embd], n_embd);
        let rms = (sum_sq / n_embd as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale src * inv_rms * norm_weight into scratch via fused SIMD, then dot with query
        scaled[..n_embd].copy_from_slice(&src[..n_embd]);
        katgpt_core::simd::simd_scale_mul_inplace(
            &mut scaled[..n_embd],
            &norm_weight[..n_embd],
            inv_rms,
        );
        let logit = katgpt_core::simd::simd_dot_f32(&scaled[..n_embd], query_weight, n_embd);

        logits[i] = logit;
        // Branch-free max reduction (single SIMD instruction, no predicted branch).
        max_logit = max_logit.max(logit);
    }

    // 2. Softmax (SIMD batch)
    katgpt_core::simd::simd_add_scalar_inplace(&mut logits, -max_logit);
    katgpt_core::simd::simd_exp_inplace(&mut logits);
    let sum_exp = katgpt_core::simd::simd_sum_f32(&logits);
    let inv_sum = 1.0 / sum_exp;
    katgpt_core::simd::simd_scale_inplace(&mut logits, inv_sum);

    logits
}
