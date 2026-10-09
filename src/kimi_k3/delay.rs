//! Delay-architecture forward + analytic backward (riir-train Issue 482 /
//! Plan 452; paper arXiv:2608.23841 Ch.4 via Research 508).
//!
//! Screening-scale CPU reference for the dependency-rewrite arms:
//! - **δd (`dense_delay`)** — the FFN site reads `Norm(x_in^ℓ)` (the
//!   layer-ENTRY anchor) instead of the post-attention mixed hidden. The
//!   always-on (shared/dense) stream is the paper's dense stream; attention is
//!   NEVER delayed (the paper's floor).
//! - **δe (`expert_delay`)** — the ROUTED experts route AND compute off the
//!   pre-dense anchor `Norm(x_in^ℓ)`, injecting their output at layer
//!   ℓ+δe's step-7 slot (δe layers later); sources with ℓ+δe ≥ L flush into
//!   the final residual BEFORE the output attn-res snapshot (never dropped).
//!   The shared expert stays immediate in every arm.
//!
//! # Semantics lock (Plan 452 §1 — pinned before implementation)
//!
//! - `x_in^ℓ` = `prefix_sum` at layer-ℓ entry = `LayerSavedActivations::prefix_sum_in`.
//! - `Norm` = the consuming layer's FFN-site norm (`post_attention_layernorm`
//!   gamma + rms_eps) — the pre-norm reading.
//! - δd=1 ⇒ the FFN input and the routing anchor coincide (`Norm(x_in^ℓ)`).
//! - Under δd=1 the MLP attn-res mixing is SKIPPED (the input switch is the
//!   arm's observable effect at our layer granularity); `mixed_mlp` /
//!   `block_state_mlp` stay EMPTY and the backward derives the path from the
//!   config, never from a field being set.
//! - The routed stream reads the anchor iff `δd > 0 || δe > 0` (A0 = no
//!   anchor anywhere); the shared/dense stream reads it iff `δd > 0`.
//!
//! # G0 bit-identity law
//!
//! With `DelayArchConfig::STANDARD` (δd=0, δe=0) the delay forward is
//! BIT-IDENTICAL to [`crate::kimi_k3::backward::kimi_k3_forward_token_saved`]
//! (logits + intermediates): the split MoE forward summed shared-first equals
//! the fused output bit-identically on the latent path (the T1a pin,
//! katgpt-transformer `1766701ae`), and the δe=0 residual add sums the two
//! stream outputs into ONE buffer before the single `prefix_sum` add — the
//! same op order as the fused assembly. The backward's MoE-internal grads at
//! the TOP MoE layer are bit-identical; every gradient downstream of the
//! FFN-site input gradient matches within the f32 reassociation band (the
//! split folds the two stream input-gradients into two buffers summed
//! shared-first vs the fused's one — the same 1-ULP class T1a pinned).
//! Anything beyond that band is a transcription defect, not a semantics
//! choice.
//!
//! Training-time reference only (modelless-by-mandate exception, same class
//! as `kimi_k3_backward`); never on the production inference path.

use katgpt_attn::gdn2::kda_backward::{
    KdaSavedActivations, kda_backward_sequence, kda_forward_token_with_saved,
};
use katgpt_attn::mla_backward::{
    MlaSavedActivations, mla_backward_token, mla_forward_token_with_saved,
    rmsnorm_backward as mla_rmsnorm_backward,
};
use katgpt_core::simd::{
    simd_add_inplace, simd_matmul_rows, simd_outer_product_acc, simd_sum_sq,
    simd_transpose_matvec_into,
};
use katgpt_core::types::math::rmsnorm_with_gamma_eps;
use katgpt_kv::shard_kv::rope::RopeFreqs;
use katgpt_transformer::attn_res::{
    AttnResBlockState, AttnResScratch, apply_attn_res,
};
use katgpt_transformer::moe_backward::{
    moe_backward_routed_stream, moe_backward_shared_stream, moe_forward_token_split,
};

use super::backward::{
    KimiK3ModelGradients, LayerSavedActivations, TokenSavedActivations, attn_res_backward,
    dense_situ_ffn_backward, dense_situ_ffn_forward_saved,
};
use super::decoder_layer::{
    KimiAttentionConfig, KimiAttentionScratch, KimiAttentionState, KimiAttentionWeights,
    KimiDecoderLayerConfig, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnScratch,
    KimiFfnWeights,
};
use super::loader::KimiK3ModelWeights;
use super::model::{KimiK3ModelConfig, KimiK3Runtime};

// ─── Config ─────────────────────────────────────────────────────────────────

/// Delay-architecture arm selector (Plan 452 §0's six arms are three
/// `(dense_delay, expert_delay)` pairs plus seed ablations).
///
/// The paper's pre-dense routing anchor is implied: the ROUTED stream reads
/// `Norm(x_in^ℓ)` iff `δd > 0 || δe > 0`, so the A0 baseline (0, 0) is the
/// standard architecture with no anchor anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DelayArchConfig {
    /// δd — the FFN-site input delay. `0` = standard (post-attention mixed
    /// hidden); `1` = the FFN block reads the layer-entry anchor. `>1` panics
    /// (a cross-layer dense delay is outside screening scope — Plan 452 §1).
    pub dense_delay: u8,
    /// δe — the routed-expert output injection delay. `0` = inject at the
    /// source layer's own step-7 slot; `k > 0` = inject at layer ℓ+k's step-7
    /// slot; a slot at or past the last layer flushes into the final residual.
    pub expert_delay: u8,
}

impl DelayArchConfig {
    /// The A0 baseline arm — bit-identical to the standard architecture.
    pub const STANDARD: Self = Self {
        dense_delay: 0,
        expert_delay: 0,
    };

    /// Construct + validate. Panics on `dense_delay > 1`.
    pub fn new(dense_delay: u8, expert_delay: u8) -> Self {
        assert!(
            dense_delay <= 1,
            "dense_delay > 1 crosses layers — outside Plan 452 screening scope"
        );
        Self {
            dense_delay,
            expert_delay,
        }
    }

    /// The shared/dense stream reads the layer-entry anchor (δd = 1).
    #[inline]
    fn anchor_shared(&self) -> bool {
        self.dense_delay == 1
    }

    /// The routed stream reads the layer-entry anchor (δd = 1 or δe > 0).
    #[inline]
    fn anchor_routed(&self) -> bool {
        self.dense_delay == 1 || self.expert_delay > 0
    }
}

// ─── Saved activations ───────────────────────────────────────────────────────

/// Saved activations for one token under a delay arm.
///
/// Wraps the standard [`TokenSavedActivations`] unchanged (its
/// `layers[ℓ].prefix_sum_in` IS the anchor source) plus the delay-side save:
/// the anchor norm's inv_rms per layer. Only read at layers where a stream
/// reads the anchor — under δd=1 it equals `base.layers[ℓ].mlp_inv_rms` (the
/// FFN-site norm IS the anchor norm there); 0.0 elsewhere and never read.
pub struct DelayTokenSavedActivations {
    pub base: TokenSavedActivations,
    pub anchor_inv_rms: Vec<f32>,
}

impl DelayTokenSavedActivations {
    pub fn new() -> Self {
        Self {
            base: TokenSavedActivations::new(),
            anchor_inv_rms: Vec::new(),
        }
    }
}

impl Default for DelayTokenSavedActivations {
    fn default() -> Self {
        Self::new()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Forward with saved activations
// ═════════════════════════════════════════════════════════════════════════════

/// Delay-architecture forward for one token, saving all activations.
///
/// Mirrors [`crate::kimi_k3::backward::kimi_k3_forward_token_saved`] exactly
/// (steps 1–4 verbatim: self attn-res, boundary push, input_layernorm +
/// attention, residual add); the FFN site is the delay-aware replacement.
/// With [`DelayArchConfig::STANDARD`] the result is bit-identical to the
/// standard forward (the G0 law in this module's docs).
#[allow(clippy::too_many_arguments)]
pub fn kimi_k3_delay_forward_token_saved(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    token_id: u32,
    pos: usize,
    delay: &DelayArchConfig,
    saved: &mut DelayTokenSavedActivations,
) {
    let d = config.hidden_size;
    let num_layers = config.num_layers;

    runtime.block_state.clear();
    saved.base.layers.clear();
    saved.anchor_inv_rms.clear();
    saved.base.pos = pos;
    saved.base.token_id = token_id;
    saved.base.is_latent = false;

    // Embedding lookup (verbatim).
    let embed_start = (token_id as usize) * d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_start + d]);

    // Pending routed injections: `pending[j]` = the routed output scheduled to
    // inject at layer j's step-7 slot (source j − δe; at most one source per
    // slot). `flush` = tail sources whose slot lands at/past the last layer.
    // Ascending source order is preserved by construction (layer loop order).
    let mut pending: Vec<Option<Vec<f32>>> = vec![None; num_layers];
    let mut flush: Vec<Vec<f32>> = Vec::new();

    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_rt = &mut runtime.layers[layer_idx];

        let mut layer_saved = LayerSavedActivations {
            prefix_sum_in: runtime.hidden.clone(),
            has_self_attn_res: false,
            mixed_self: Vec::new(),
            self_inv_rms: 0.0,
            is_boundary: false,
            block_state_self: Vec::new(),
            attn_out: Vec::new(),
            prefix_sum_after_attn: Vec::new(),
            mixed_mlp: Vec::new(),
            mlp_inv_rms: 0.0,
            block_state_mlp: Vec::new(),
            ffn_out: Vec::new(),
            mla_saved: None,
            kda_saved: None,
            moe_saved: None,
            dense_saved: None,
        };

        forward_layer_delay(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
            delay,
            &mut pending,
            &mut flush,
            &mut saved.anchor_inv_rms,
            &mut layer_saved,
        );

        saved.base.layers.push(layer_saved);
    }

    // δe tail flush — joins the final residual BEFORE the output attn-res
    // snapshot (ascending source order; deterministic).
    for r in flush {
        simd_add_inplace(&mut runtime.hidden[..d], &r[..d]);
    }

    // Output attn-res + final norm + LM head — VERBATIM mirror of
    // `kimi_k3_forward_token_saved`'s tail.
    saved.base.prefix_sum_final = runtime.hidden.clone();
    saved.base.block_state_final = runtime.block_state.residuals.clone();
    saved.base.has_output_attn_res = !runtime.block_state.is_empty();

    if !runtime.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);
    }

    saved.base.pre_final_norm = runtime.hidden.clone();

    let sum_sq = simd_sum_sq(&runtime.hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + config.rms_eps).sqrt());
    saved.base.final_norm_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );
    saved.base.final_hidden = runtime.hidden.clone();

    simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );
    saved.base.logits = runtime.logits.clone();
}

/// One decoder layer under a delay arm. Steps 1–4 are VERBATIM from
/// `forward_layer_saved` (the mirror law — attention is never delayed); the
/// FFN site (steps 5–7) is the delay-aware replacement.
#[allow(clippy::too_many_arguments)]
fn forward_layer_delay(
    layer_idx: usize,
    config: &KimiDecoderLayerConfig,
    weights: &KimiDecoderLayerWeights,
    attn_state: &mut KimiAttentionState,
    attn_scratch: &mut KimiAttentionScratch,
    ffn_scratch: &mut KimiFfnScratch,
    attn_res_self_scratch: &mut AttnResScratch,
    attn_res_mlp_scratch: &mut AttnResScratch,
    block_state: &mut AttnResBlockState,
    rope_freqs: Option<&mut RopeFreqs>,
    prefix_sum: &mut [f32],
    scratch_hidden: &mut [f32],
    delay: &DelayArchConfig,
    pending: &mut [Option<Vec<f32>>],
    flush: &mut Vec<Vec<f32>>,
    anchor_inv_rms: &mut Vec<f32>,
    saved: &mut LayerSavedActivations,
) {
    let d = config.attn_res.d();
    let eps = config.rms_eps;
    let block_size = config.attn_res.block_size;
    let is_boundary = layer_idx.is_multiple_of(block_size);
    saved.is_boundary = is_boundary;
    let mut anchor_inv: f32 = 0.0;

    // Step 1: self-attn-res mixing — VERBATIM.
    if !block_state.is_empty() {
        saved.has_self_attn_res = true;
        saved.block_state_self = block_state.residuals.clone();
        let mixed = apply_attn_res(
            &config.attn_res,
            &weights.self_attn_res,
            block_state,
            attn_res_self_scratch,
            prefix_sum,
        );
        scratch_hidden.copy_from_slice(mixed);
    } else {
        scratch_hidden.copy_from_slice(prefix_sum);
    }
    saved.mixed_self = scratch_hidden.to_vec();

    // Step 2: boundary push — VERBATIM.
    if is_boundary {
        block_state.push(prefix_sum);
        prefix_sum.fill(0.0);
    }

    // Step 3: input_layernorm + attention — VERBATIM.
    let sum_sq = simd_sum_sq(scratch_hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
    saved.self_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(scratch_hidden, &weights.input_layernorm_weight, eps as f64);

    let attn_out: Vec<f32> = match (&config.attention, &weights.attention) {
        (KimiAttentionConfig::Mla(cfg), KimiAttentionWeights::Mla(w)) => {
            let KimiAttentionState::Mla(cache) = attn_state else {
                panic!("MLA state mismatch")
            };
            let KimiAttentionScratch::Mla(scratch) = attn_scratch else {
                panic!("MLA scratch mismatch")
            };
            let Some(rf) = rope_freqs else {
                panic!("MLA needs rope")
            };
            let (out, s) = mla_forward_token_with_saved(cfg, w, cache, scratch, rf, scratch_hidden);
            saved.mla_saved = Some(s);
            out
        }
        (KimiAttentionConfig::Kda(cfg), KimiAttentionWeights::Kda(w)) => {
            let KimiAttentionState::Kda(cache) = attn_state else {
                panic!("KDA state mismatch")
            };
            let KimiAttentionScratch::Kda(scratch) = attn_scratch else {
                panic!("KDA scratch mismatch")
            };
            let (out, s) = kda_forward_token_with_saved(cfg, w, cache, scratch, scratch_hidden);
            saved.kda_saved = Some(s);
            out
        }
        _ => panic!("attention mismatch"),
    };
    saved.attn_out = attn_out;

    // Step 4: prefix_sum += attn_out — VERBATIM.
    simd_add_inplace(&mut prefix_sum[..d], &saved.attn_out[..d]);
    saved.prefix_sum_after_attn = prefix_sum.to_vec();

    // Steps 5–6: the FFN-site input under the delay arm.
    if !delay.anchor_shared() {
        // δd=0: standard MLP attn-res mixing + FFN-site norm — VERBATIM.
        saved.block_state_mlp = block_state.residuals.clone();
        let mixed = apply_attn_res(
            &config.attn_res,
            &weights.mlp_attn_res,
            block_state,
            attn_res_mlp_scratch,
            prefix_sum,
        );
        scratch_hidden.copy_from_slice(mixed);
        saved.mixed_mlp = scratch_hidden.to_vec();
        let sum_sq = simd_sum_sq(scratch_hidden, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        saved.mlp_inv_rms = inv_rms;
        rmsnorm_with_gamma_eps(
            scratch_hidden,
            &weights.post_attention_layernorm_weight,
            eps as f64,
        );
    } else {
        // δd=1: the whole FFN block reads Norm(x_in^ℓ) — the layer-entry
        // anchor normed with the FFN-site norm. The MLP attn-res mixing is
        // SKIPPED (mixed_mlp / block_state_mlp stay empty; the backward
        // derives the path from the config). scratch_hidden is free here —
        // attention already consumed its step-3 contents.
        scratch_hidden.copy_from_slice(&saved.prefix_sum_in);
        let sum_sq = simd_sum_sq(scratch_hidden, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        saved.mlp_inv_rms = inv_rms;
        anchor_inv = inv_rms;
        rmsnorm_with_gamma_eps(
            scratch_hidden,
            &weights.post_attention_layernorm_weight,
            eps as f64,
        );
    }

    // Step 6b+7: the FFN call(s) + residual injection.
    match (&config.ffn, &weights.ffn) {
        (
            KimiFfnConfig::Dense {
                situ_beta,
                situ_linear_beta,
                ..
            },
            KimiFfnWeights::Dense(expert),
        ) => {
            // Input = the δd-governed buffer (anchor under δd=1, normed mixed
            // under δd=0). No routed stream; δe never applies.
            let (out, s) = dense_situ_ffn_forward_saved(
                expert,
                scratch_hidden,
                ffn_scratch,
                *situ_beta,
                *situ_linear_beta,
            );
            saved.dense_saved = Some(s);
            saved.ffn_out = out;
            simd_add_inplace(&mut prefix_sum[..d], &saved.ffn_out[..d]);
        }
        (KimiFfnConfig::Moe(moe_cfg), KimiFfnWeights::Moe(moe_w)) => {
            let (shared_out, routed_out, moe_saved) = if delay.anchor_routed()
                && !delay.anchor_shared()
            {
                // δd=0, δe>0: the routed stream reads the ANCHOR — the
                // layer-entry residual under the same site norm (a fresh
                // buffer; scratch_hidden stays the shared stream's input).
                let mut anchor = saved.prefix_sum_in.clone();
                let sum_sq = simd_sum_sq(&anchor, d);
                anchor_inv = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
                rmsnorm_with_gamma_eps(
                    &mut anchor,
                    &weights.post_attention_layernorm_weight,
                    eps as f64,
                );
                moe_forward_token_split(moe_w, moe_cfg, scratch_hidden, &anchor, &mut ffn_scratch.moe)
            } else {
                // G0 (both streams read the standard normed input) or δd=1
                // (both read the anchor) — one input for both streams.
                moe_forward_token_split(
                    moe_w,
                    moe_cfg,
                    scratch_hidden,
                    scratch_hidden,
                    &mut ffn_scratch.moe,
                )
            };
            saved.moe_saved = Some(moe_saved);

            if delay.expert_delay == 0 {
                // δe=0: sum shared-first into ONE buffer, ONE residual add —
                // the fused path assembles `hidden_out = shared; += routed`
                // elementwise, so this op order keeps the G0 arm
                // BIT-IDENTICAL to the standard forward (T1a pin).
                let mut ffn_buf = shared_out;
                for (s, r) in ffn_buf.iter_mut().zip(routed_out.iter()) {
                    *s += *r;
                }
                simd_add_inplace(&mut prefix_sum[..d], &ffn_buf[..d]);
                saved.ffn_out = ffn_buf;
            } else {
                // δe>0: the shared stream injects NOW; the routed output is
                // scheduled for its δe-later slot (or the tail flush).
                simd_add_inplace(&mut prefix_sum[..d], &shared_out[..d]);
                saved.ffn_out = shared_out;
                let slot = layer_idx + delay.expert_delay as usize;
                if slot < pending.len() {
                    pending[slot] = Some(routed_out);
                } else {
                    flush.push(routed_out);
                }
            }
        }
        _ => panic!("FFN mismatch"),
    }

    // Pending routed injection due at THIS layer's step-7 slot (source =
    // layer_idx − δe): lands AFTER this layer's own step-7 add. Layer 0 never
    // receives one — a slot of 0 implies a source at −δe < 1, and only MoE
    // layers ≥ 1 ever schedule.
    if let Some(r) = pending[layer_idx].take() {
        simd_add_inplace(&mut prefix_sum[..d], &r[..d]);
    }

    anchor_inv_rms.push(anchor_inv);
}

// ═════════════════════════════════════════════════════════════════════════════
// Backward
// ═════════════════════════════════════════════════════════════════════════════

/// Delay-architecture full-model backward for a sequence of tokens.
///
/// Mirrors `kimi_k3_backward_sequence_with_input_grad`'s walk (steps 1, 2b,
/// 2c verbatim — attention is never delayed) with the FFN-site split:
/// - The SHARED stream's backward runs at its own layer (walk step ℓ) with
///   `d_shared = d_prefix[t]` (the step-7 slot gradient).
/// - The ROUTED stream's backward runs at its INJECTION layer (walk step
///   ℓ+δe, or step 1 for tail-flushed sources) with the same slot gradient,
///   then flows through the ANCHOR norm backward into a deposit at the SOURCE
///   layer's entry slot — bypassing the mixed_mlp chain.
/// - δd=1 routes the shared/dense input gradient the same way (its FFN-site
///   norm IS the anchor norm).
/// - δe=0 folds both streams into one call pair at the same walk step, summed
///   shared-first — the G0 band (see module docs).
pub fn kimi_k3_delay_backward_sequence(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &KimiK3Runtime,
    saved_tokens: &[DelayTokenSavedActivations],
    d_logits: &[Vec<f32>],
    delay: &DelayArchConfig,
    grads: &mut KimiK3ModelGradients,
) {
    let d = config.hidden_size;
    let v = config.vocab_size;
    let l = saved_tokens.len();
    debug_assert_eq!(d_logits.len(), l);
    if l == 0 {
        return;
    }
    let num_layers = config.num_layers;
    let de = delay.expert_delay as usize;
    let anchor_shared = delay.anchor_shared();

    let mut d_prefix: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
    let mut block_grads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l);
    // Anchor-norm deposits landing at layer ℓ's ENTRY slot: produced whenever
    // a routed/δd stream's backward runs at a walk step other than ℓ (or at
    // the flush), added to d_prefix[t] at the END of walk step ℓ.
    let mut entry_deposits: Vec<Vec<Vec<f32>>> = (0..l)
        .map(|_| (0..num_layers).map(|_| vec![0.0f32; d]).collect())
        .collect();

    // ── Step 1: per-token output backward — VERBATIM (LM head + final norm +
    // output attn-res) → d_prefix[t]. ──
    for t in 0..l {
        let saved = &saved_tokens[t].base;

        let mut d_fh = vec![0.0f32; d];
        simd_transpose_matvec_into(
            &mut d_fh,
            &weights.lm_head_weight,
            &d_logits[t],
            v,
            d,
        );
        simd_outer_product_acc(
            &mut grads.lm_head_weight,
            &d_logits[t],
            &saved.final_hidden,
            v,
            d,
        );

        let d_pre_fn = mla_rmsnorm_backward(
            &d_fh,
            &saved.pre_final_norm,
            &weights.final_norm_weight,
            saved.final_norm_inv_rms,
            &mut grads.final_norm_weight,
            config.rms_eps,
        );

        let num_blocks = saved.block_state_final.len();
        let mut bg: Vec<Vec<f32>> = (0..num_blocks).map(|_| vec![0.0f32; d]).collect();

        if saved.has_output_attn_res {
            attn_res_backward(
                &config.attn_res_config,
                &weights.output_attn_res,
                &saved.block_state_final,
                &saved.prefix_sum_final,
                &d_pre_fn,
                &mut bg,
                &mut d_prefix[t],
                &mut grads.output_attn_res_norm,
                &mut grads.output_attn_res_proj,
            );
        } else {
            d_prefix[t].copy_from_slice(&d_pre_fn);
        }
        block_grads.push(bg);
    }

    // ── Step 1b: tail-flushed routed-stream backwards (δe > 0 only). ──
    // The flush joined the final residual BEFORE the output attn-res
    // snapshot, so each flushed source receives the full d_prefix[t] (the
    // additive-site gradient at the flush point). Ascending source order —
    // deterministic, and disjoint from the walk's layer grads.
    if de > 0 {
        // The flush-source scan indexes per-token deposit rows by source
        // layer — the range form mirrors the walk's layer-keyed structure.
        #[allow(clippy::needless_range_loop)]
        for src in 1..num_layers {
            let src_is_moe =
                matches!(weights.layers[src].ffn, KimiFfnWeights::Moe(_)) && src + de >= num_layers;
            if !src_is_moe {
                continue;
            }
            for t in 0..l {
                routed_source_backward_and_deposit(
                    config,
                    weights,
                    &saved_tokens[t],
                    &d_prefix[t],
                    src,
                    grads,
                    &mut entry_deposits[t][src],
                );
            }
        }
    }

    // Hoisted scratch (the standard walk's pattern).
    let mut d_attn_out: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
    let mut d_ps_after_attn_all: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

    // ── Step 2: layer-by-layer backward (reverse). ──
    for layer_idx in (0..num_layers).rev() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_w = &weights.layers[layer_idx];
        let is_mla = config.is_mla_layer(layer_idx);
        let is_boundary = layer_idx.is_multiple_of(config.attn_res_config.block_size);

        // ── Step 2a-pre: the routed stream whose injection slot is THIS layer
        // (source = layer_idx − δe, a MoE layer ≥ 1). Runs BEFORE the layer's
        // own shared backward so the two never borrow conflicting grads rows;
        // G0 has no foreign-layer source and stays single-path. The slot
        // gradient is the SAME d_prefix[t] the shared backward consumes —
        // both streams were added at this step-7 slot in the forward. ──
        if de > 0 && layer_idx > de {
            let src = layer_idx - de;
            if matches!(weights.layers[src].ffn, KimiFfnWeights::Moe(_)) {
                for t in 0..l {
                    routed_source_backward_and_deposit(
                        config,
                        weights,
                        &saved_tokens[t],
                        &d_prefix[t],
                        src,
                        grads,
                        &mut entry_deposits[t][src],
                    );
                }
            }
        }

        let layer_grads = &mut grads.layers[layer_idx];

        // ── Step 2a: FFN-site backward (per token). ──
        for t in 0..l {
            let saved = &saved_tokens[t].base.layers[layer_idx];

            let d_normed: Vec<f32> = match (&layer_cfg.ffn, &layer_w.ffn) {
                (
                    KimiFfnConfig::Dense {
                        situ_beta,
                        situ_linear_beta,
                        ..
                    },
                    KimiFfnWeights::Dense(expert_w),
                ) => {
                    let dense_saved = saved.dense_saved.as_ref().unwrap();
                    dense_situ_ffn_backward(
                        expert_w,
                        dense_saved,
                        &d_prefix[t],
                        layer_grads.dense_grads.as_mut().unwrap(),
                        *situ_beta,
                        *situ_linear_beta,
                    )
                }
                (KimiFfnConfig::Moe(moe_cfg), KimiFfnWeights::Moe(moe_w)) => {
                    let moe_saved = saved.moe_saved.as_ref().unwrap();
                    let mut dh_shared = vec![0.0f32; d];
                    moe_backward_shared_stream(
                        moe_cfg,
                        moe_w,
                        moe_saved,
                        &d_prefix[t],
                        &mut dh_shared,
                        layer_grads.moe_grads.as_mut().unwrap(),
                    );
                    if de == 0 {
                        // δe=0: the routed stream injected at this layer's own
                        // step-7 — its backward runs here with the same slot
                        // gradient. Shared FIRST (the fused accumulation
                        // order; keeps G0's MoE grads bit-identical), then the
                        // shared-first dh sum (the T1a pin's order).
                        let mut dh_routed = vec![0.0f32; d];
                        moe_backward_routed_stream(
                            moe_cfg,
                            moe_w,
                            moe_saved,
                            &d_prefix[t],
                            &mut dh_routed,
                            layer_grads.moe_grads.as_mut().unwrap(),
                        );
                        for (s, r) in dh_shared.iter_mut().zip(dh_routed.iter()) {
                            *s += *r;
                        }
                    }
                    dh_shared
                }
                _ => panic!("FFN config/weights mismatch"),
            };

            // Norm + mixing backward for this layer's FFN-site input.
            if anchor_shared {
                // δd=1: the FFN-site norm ran on the LAYER-ENTRY anchor —
                // deposit at the entry slot, bypassing the mixed_mlp chain
                // (there was no MLP attn-res).
                let dep = mla_rmsnorm_backward(
                    &d_normed,
                    &saved.prefix_sum_in,
                    &layer_w.post_attention_layernorm_weight,
                    saved.mlp_inv_rms,
                    &mut layer_grads.post_attention_layernorm_weight,
                    config.rms_eps,
                );
                simd_add_inplace(&mut entry_deposits[t][layer_idx], &dep);
                // Residual gradient only (no attn-res contribution).
                d_attn_out[t].copy_from_slice(&d_prefix[t]);
            } else {
                // δd=0: the standard path — norm on mixed_mlp, then the MLP
                // attn-res backward (VERBATIM from the standard walk).
                let d_mixed_mlp = mla_rmsnorm_backward(
                    &d_normed,
                    &saved.mixed_mlp,
                    &layer_w.post_attention_layernorm_weight,
                    saved.mlp_inv_rms,
                    &mut layer_grads.post_attention_layernorm_weight,
                    config.rms_eps,
                );

                let num_mlp_blocks = saved.block_state_mlp.len();
                while block_grads[t].len() < num_mlp_blocks {
                    block_grads[t].push(vec![0.0f32; d]);
                }
                let mut d_mlp_blocks: Vec<Vec<f32>> =
                    (0..num_mlp_blocks).map(|_| vec![0.0f32; d]).collect();
                d_attn_out[t].copy_from_slice(&d_prefix[t]);

                attn_res_backward(
                    &config.attn_res_config,
                    &layer_w.mlp_attn_res,
                    &saved.block_state_mlp,
                    &saved.prefix_sum_after_attn,
                    &d_mixed_mlp,
                    &mut d_mlp_blocks,
                    &mut d_attn_out[t],
                    &mut layer_grads.mlp_attn_res_norm,
                    &mut layer_grads.mlp_attn_res_proj,
                );

                for (bg_row, d_row) in block_grads[t].iter_mut().zip(d_mlp_blocks.iter()) {
                    let bg = &mut bg_row[..d];
                    let dr = &d_row[..d];
                    for j in 0..d {
                        bg[j] += dr[j];
                    }
                }
            }
            d_ps_after_attn_all[t].copy_from_slice(&d_attn_out[t]);
        }

        // ── Step 2b: attention backward — VERBATIM. ──
        let mut d_normed_self: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

        if is_mla {
            let KimiAttentionWeights::Mla(mla_w) = &layer_w.attention else {
                panic!("MLA layer but non-MLA weights");
            };
            let KimiAttentionState::Mla(cache) = &runtime.layers[layer_idx].attn_state else {
                panic!("MLA layer but non-MLA cache");
            };

            let all_saved: Vec<MlaSavedActivations> = (0..l)
                .map(|t| {
                    saved_tokens[t]
                        .base
                        .layers[layer_idx]
                        .mla_saved
                        .clone()
                        .unwrap()
                })
                .collect();

            let mut all_dh: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
            let mla_grads = layer_grads.mla_grads.as_mut().unwrap();
            let mut rf = RopeFreqs::new_with_theta(
                config.mla_config.qk_rope_head_dim,
                config.mla_config.rope_theta,
            );

            for t in 0..l {
                mla_backward_token(
                    &config.mla_config,
                    mla_w,
                    cache,
                    &all_saved[t],
                    &all_saved,
                    &mut rf,
                    &d_attn_out[t],
                    &mut all_dh,
                    mla_grads,
                );
            }
            d_normed_self = all_dh;
        } else {
            let KimiAttentionWeights::Kda(kda_w) = &layer_w.attention else {
                panic!("KDA layer but non-KDA weights");
            };
            let all_saved_kda: Vec<KdaSavedActivations> = (0..l)
                .map(|t| {
                    saved_tokens[t]
                        .base
                        .layers[layer_idx]
                        .kda_saved
                        .clone()
                        .unwrap()
                })
                .collect();
            let kda_grads = layer_grads.kda_grads.as_mut().unwrap();

            kda_backward_sequence(
                &config.kda_config,
                kda_w,
                &all_saved_kda,
                &d_attn_out,
                &mut d_normed_self,
                kda_grads,
            );
        }

        // ── Step 2c: self-attn block backward — VERBATIM, plus the delay
        // entry-deposit add at the end. ──
        for t in 0..l {
            let saved = &saved_tokens[t].base.layers[layer_idx];

            let d_mixed_self = mla_rmsnorm_backward(
                &d_normed_self[t],
                &saved.mixed_self,
                &layer_w.input_layernorm_weight,
                saved.self_inv_rms,
                &mut layer_grads.input_layernorm_weight,
                config.rms_eps,
            );

            let mut d_ps_in = if is_boundary {
                block_grads[t].pop().unwrap_or_else(|| vec![0.0f32; d])
            } else {
                vec![0.0f32; d]
            };

            if !is_boundary {
                simd_add_inplace(&mut d_ps_in[..d], &d_ps_after_attn_all[t][..d]);
            }

            if saved.has_self_attn_res {
                let num_self_blocks = saved.block_state_self.len();
                let mut d_self_blocks: Vec<Vec<f32>> =
                    (0..num_self_blocks).map(|_| vec![0.0f32; d]).collect();
                let mut d_ps_from_attnres = vec![0.0f32; d];

                attn_res_backward(
                    &config.attn_res_config,
                    &layer_w.self_attn_res,
                    &saved.block_state_self,
                    &saved.prefix_sum_in,
                    &d_mixed_self,
                    &mut d_self_blocks,
                    &mut d_ps_from_attnres,
                    &mut layer_grads.self_attn_res_norm,
                    &mut layer_grads.self_attn_res_proj,
                );

                for (bg_row, d_row) in block_grads[t].iter_mut().zip(d_self_blocks.iter()) {
                    simd_add_inplace(&mut bg_row[..d], &d_row[..d]);
                }
                simd_add_inplace(&mut d_ps_in[..d], &d_ps_from_attnres[..d]);
            } else {
                simd_add_inplace(&mut d_ps_in[..d], &d_mixed_self[..d]);
            }

            // d_ps_in is the standard-flow entry gradient; add the anchor
            // deposits that target THIS layer's entry slot (routed streams
            // whose backward ran at an earlier walk step / the flush, and —
            // already folded via d_normed — none for the shared stream).
            simd_add_inplace(&mut d_ps_in[..d], &entry_deposits[t][layer_idx][..d]);

            d_prefix[t] = d_ps_in;
        }
    }

    // ── Step 3: embedding backward — VERBATIM (the token forward sets
    // is_latent = false; the latent-input variant is out of screening scope).
    for t in 0..l {
        if saved_tokens[t].base.is_latent {
            continue;
        }
        let token_id = saved_tokens[t].base.token_id as usize;
        let base = token_id * d;
        simd_add_inplace(
            &mut grads.embed_weight[base..base + d],
            &d_prefix[t][..d],
        );
    }
}

/// One routed-source backward: run the routed-stream backward of layer `src`
/// against the injection-site gradient `d_slot`, flow its input gradient
/// through the SOURCE layer's anchor norm, and deposit the result at the
/// source layer's entry slot. Also accumulates the anchor norm's gamma grad
/// at the source layer.
///
/// Used by both the in-walk consumption (source injects at the current walk
/// layer) and the step-1 tail-flush sources.
fn routed_source_backward_and_deposit(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    saved_token: &DelayTokenSavedActivations,
    d_slot: &[f32],
    src: usize,
    grads: &mut KimiK3ModelGradients,
    entry_deposit: &mut [f32],
) {
    let d = config.hidden_size;
    let src_layer_cfg = config.layer_config(src);
    let (src_moe_cfg, src_moe_w) = match (&src_layer_cfg.ffn, &weights.layers[src].ffn) {
        (KimiFfnConfig::Moe(c), KimiFfnWeights::Moe(w)) => (c, w),
        _ => panic!("routed source {src} is not a MoE layer"),
    };
    let moe_saved = saved_token.base.layers[src].moe_saved.as_ref().unwrap();

    let mut dh_routed = vec![0.0f32; d];
    moe_backward_routed_stream(
        src_moe_cfg,
        src_moe_w,
        moe_saved,
        d_slot,
        &mut dh_routed,
        grads.layers[src].moe_grads.as_mut().unwrap(),
    );

    // The routed stream read the anchor Norm(x_in^src) under the SOURCE
    // layer's FFN-site norm — backward against (prefix_sum_in, anchor_inv_rms)
    // deposits at the source layer's ENTRY slot.
    let dep = mla_rmsnorm_backward(
        &dh_routed,
        &saved_token.base.layers[src].prefix_sum_in,
        &weights.layers[src].post_attention_layernorm_weight,
        saved_token.anchor_inv_rms[src],
        &mut grads.layers[src].post_attention_layernorm_weight,
        config.rms_eps,
    );
    simd_add_inplace(&mut entry_deposit[..d], &dep[..d]);
}

// ═════════════════════════════════════════════════════════════════════════════
// Gates (Plan 452 T1b): G0 bit-identity + per-arm finite-difference checks.
// ═════════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kimi_k3::backward::{
        kimi_k3_backward_sequence, kimi_k3_forward_token_saved,
    };
    use katgpt_attn::gdn2::kda_forward::KdaConfig;
    use katgpt_attn::mla::MlaConfig;
    use katgpt_transformer::attn_res::AttnResConfig;
    use katgpt_transformer::moe::MoeConfig;

    /// Toy config mirroring `tests/kimi_k3_backward_grad_check.rs`'s builder:
    /// layer 0 (KDA + Dense), layers 1.. (KDA + MoE), last layer MLA + MoE
    /// when `use_mla`. LATENT MoE (the Kimi family path).
    fn toy_config(use_mla: bool, num_layers: usize) -> KimiK3ModelConfig {
        let d = 16;
        KimiK3ModelConfig {
            hidden_size: d,
            vocab_size: 32,
            num_layers,
            rms_eps: 1e-5,
            mla_layer_indices: if use_mla {
                vec![num_layers - 1]
            } else {
                vec![]
            },
            mla_config: MlaConfig {
                kv_lora_rank: 8,
                q_lora_rank: 8,
                qk_nope_head_dim: 4,
                qk_rope_head_dim: 4,
                v_head_dim: 4,
                n_heads: 2,
                hidden_size: d,
                use_output_gate: true,
                use_nope: true,
                rope_theta: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            kda_config: KdaConfig {
                hidden_size: d,
                n_heads: 2,
                head_dim: 8,
                conv_kernel_size: 4,
                ..KdaConfig::kimi_k3_0_40b()
            },
            dense_ffn_config: KimiFfnConfig::Dense {
                intermediate_size: 32,
                hidden_size: d,
                situ_beta: 4.0,
                situ_linear_beta: Some(25.0),
            },
            moe_config: MoeConfig {
                num_experts: 4,
                num_experts_per_token: 2,
                num_shared_experts: 1,
                moe_intermediate_size: 16,
                hidden_size: d,
                routed_expert_hidden_size: Some(8),
                ..MoeConfig::kimi_k3_0_40b()
            },
            attn_res_config: AttnResConfig {
                hidden_size: d,
                block_size: 4,
                rms_eps: 1e-5,
            },
        }
    }

    fn assert_bits_eq(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}: element {i}: {x} vs {y}"
            );
        }
    }

    /// Tolerance-band comparator (the T1a pin's shape: rel band immune to
    /// codegen, tight enough to catch real defects).
    fn assert_within_band(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        let mut max_drift = 0.0f32;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let drift = (x - y).abs();
            let bound = tol * (1.0 + y.abs());
            assert!(
                drift <= bound,
                "{what}: element {i}: {x} vs {y} (drift {drift}, bound {bound})"
            );
            max_drift = max_drift.max(drift / (1.0 + y.abs()));
        }
        eprintln!("{what}: max rel drift {max_drift:.3e}");
    }

    /// G0 forward pin: STANDARD delay arm == the standard forward,
    /// BIT-IDENTICALLY (logits + final residual, per token, including the
    /// MLA top layer + KDA lower layers + boundary push at layer 0).
    #[test]
    fn g0_forward_bit_identical_to_standard() {
        let config = toy_config(true, 3);
        let weights = KimiK3ModelWeights::random(&config, 42);
        let tokens = [3u32, 1, 4];

        let mut rt_std = KimiK3Runtime::new(&config, tokens.len());
        let mut sv_std = TokenSavedActivations::new();
        let mut rt_dly = KimiK3Runtime::new(&config, tokens.len());
        let mut sv_dly = DelayTokenSavedActivations::new();

        for (pos, &tok) in tokens.iter().enumerate() {
            kimi_k3_forward_token_saved(&config, &weights, &mut rt_std, tok, pos, &mut sv_std);
            kimi_k3_delay_forward_token_saved(
                &config,
                &weights,
                &mut rt_dly,
                tok,
                pos,
                &DelayArchConfig::STANDARD,
                &mut sv_dly,
            );
            assert_bits_eq(&sv_std.logits, &sv_dly.base.logits, "G0 logits");
            assert_bits_eq(
                &sv_std.prefix_sum_final,
                &sv_dly.base.prefix_sum_final,
                "G0 prefix_sum_final",
            );
        }
        // Anchor side-save unused at G0 — all zeros by construction.
        assert!(sv_dly.anchor_inv_rms.iter().all(|&v| v == 0.0));
    }

    /// G0 backward pin: MoE-internal grads at the TOP MoE layer + all step-1
    /// grads BIT-IDENTICAL; everything downstream of the FFN-site input
    /// gradient within the split-reassociation band (T1a's 1-ULP class,
    /// widened to a few ULPs for the multi-hop propagation through lower
    /// layers' norm/attn-res/attention backwards).
    #[test]
    fn g0_backward_grads_match_standard() {
        let config = toy_config(true, 3);
        let weights = KimiK3ModelWeights::random(&config, 42);
        let tokens = [3u32, 1, 4];
        let l = tokens.len();

        let mut rt_std = KimiK3Runtime::new(&config, l);
        let mut sv_std: Vec<TokenSavedActivations> =
            (0..l).map(|_| TokenSavedActivations::new()).collect();
        for (pos, &tok) in tokens.iter().enumerate() {
            kimi_k3_forward_token_saved(&config, &weights, &mut rt_std, tok, pos, &mut sv_std[pos]);
        }
        let mut rt_dly = KimiK3Runtime::new(&config, l);
        let mut sv_dly: Vec<DelayTokenSavedActivations> =
            (0..l).map(|_| DelayTokenSavedActivations::new()).collect();
        for (pos, &tok) in tokens.iter().enumerate() {
            kimi_k3_delay_forward_token_saved(
                &config,
                &weights,
                &mut rt_dly,
                tok,
                pos,
                &DelayArchConfig::STANDARD,
                &mut sv_dly[pos],
            );
        }

        // G0 forward is bit-identical, so both backwards consume identical
        // logits — one d_logits set serves both.
        let d_logits: Vec<Vec<f32>> = sv_std
            .iter()
            .map(|s| s.logits.iter().map(|&v| 2.0 * v).collect())
            .collect();

        let mut g_std = KimiK3ModelGradients::zeros_like(&config, &weights);
        kimi_k3_backward_sequence(&config, &weights, &rt_std, &sv_std, &d_logits, &mut g_std);
        let mut g_dly = KimiK3ModelGradients::zeros_like(&config, &weights);
        kimi_k3_delay_backward_sequence(
            &config,
            &weights,
            &rt_dly,
            &sv_dly,
            &d_logits,
            &DelayArchConfig::STANDARD,
            &mut g_dly,
        );

        // Bit-identical class: step-1 grads (identical inputs, verbatim code)
        // + the TOP layer's MoE-internal grads (its slot gradient arrives
        // bit-identical; the split halves preserve the fused accumulation).
        assert_bits_eq(&g_std.lm_head_weight, &g_dly.lm_head_weight, "G0 lm_head");
        assert_bits_eq(
            &g_std.final_norm_weight,
            &g_dly.final_norm_weight,
            "G0 final_norm",
        );
        assert_bits_eq(
            &g_std.output_attn_res_norm,
            &g_dly.output_attn_res_norm,
            "G0 output_attn_res_norm",
        );
        assert_bits_eq(
            &g_std.output_attn_res_proj,
            &g_dly.output_attn_res_proj,
            "G0 output_attn_res_proj",
        );
        let top = config.num_layers - 1;
        let m_std = g_std.layers[top].moe_grads.as_ref().unwrap();
        let m_dly = g_dly.layers[top].moe_grads.as_ref().unwrap();
        assert_bits_eq(&m_std.router_weight, &m_dly.router_weight, "G0 top router");
        for (e_s, e_d) in m_std.experts.iter().zip(m_dly.experts.iter()) {
            assert_bits_eq(&e_s.gate_proj, &e_d.gate_proj, "G0 top expert gate");
            assert_bits_eq(&e_s.up_proj, &e_d.up_proj, "G0 top expert up");
            assert_bits_eq(&e_s.down_proj, &e_d.down_proj, "G0 top expert down");
        }
        for (e_s, e_d) in m_std
            .shared_experts
            .iter()
            .zip(m_dly.shared_experts.iter())
        {
            assert_bits_eq(&e_s.gate_proj, &e_d.gate_proj, "G0 top shared gate");
            assert_bits_eq(&e_s.up_proj, &e_d.up_proj, "G0 top shared up");
            assert_bits_eq(&e_s.down_proj, &e_d.down_proj, "G0 top shared down");
        }
        assert_bits_eq(
            m_std.routed_expert_down_proj.as_ref().unwrap(),
            m_dly.routed_expert_down_proj.as_ref().unwrap(),
            "G0 top latent down_proj",
        );
        assert_bits_eq(
            m_std.routed_expert_up_proj.as_ref().unwrap(),
            m_dly.routed_expert_up_proj.as_ref().unwrap(),
            "G0 top latent up_proj",
        );

        // Band class: everything downstream of the FFN-site input gradient
        // (embed, per-layer norms/attn-res/attention grads, lower MoE layers,
        // the dense layer). Multi-hop reassociation band — measured, printed.
        const BAND: f32 = 1e-5;
        assert_within_band(&g_std.embed_weight, &g_dly.embed_weight, BAND, "G0 embed");
        for li in 0..config.num_layers {
            let s = &g_std.layers[li];
            let dl = &g_dly.layers[li];
            assert_within_band(
                &s.input_layernorm_weight,
                &dl.input_layernorm_weight,
                BAND,
                "G0 input_ln",
            );
            assert_within_band(
                &s.post_attention_layernorm_weight,
                &dl.post_attention_layernorm_weight,
                BAND,
                "G0 post_ln",
            );
            assert_within_band(
                &s.self_attn_res_norm,
                &dl.self_attn_res_norm,
                BAND,
                "G0 self_ar_norm",
            );
            assert_within_band(
                &s.self_attn_res_proj,
                &dl.self_attn_res_proj,
                BAND,
                "G0 self_ar_proj",
            );
            assert_within_band(
                &s.mlp_attn_res_norm,
                &dl.mlp_attn_res_norm,
                BAND,
                "G0 mlp_ar_norm",
            );
            assert_within_band(
                &s.mlp_attn_res_proj,
                &dl.mlp_attn_res_proj,
                BAND,
                "G0 mlp_ar_proj",
            );
            if li != top {
                if let (Some(s_m), Some(d_m)) = (s.moe_grads.as_ref(), dl.moe_grads.as_ref()) {
                    assert_within_band(
                        &s_m.router_weight,
                        &d_m.router_weight,
                        BAND,
                        "G0 lower router",
                    );
                    for (e_s, e_d) in s_m.experts.iter().zip(d_m.experts.iter()) {
                        assert_within_band(&e_s.gate_proj, &e_d.gate_proj, BAND, "G0 lower gate");
                        assert_within_band(&e_s.up_proj, &e_d.up_proj, BAND, "G0 lower up");
                        assert_within_band(
                            &e_s.down_proj,
                            &e_d.down_proj,
                            BAND,
                            "G0 lower down",
                        );
                    }
                }
                if let (Some(s_d), Some(d_d)) = (s.dense_grads.as_ref(), dl.dense_grads.as_ref()) {
                    assert_within_band(&s_d.gate_proj, &d_d.gate_proj, BAND, "G0 dense gate");
                    assert_within_band(&s_d.up_proj, &d_d.up_proj, BAND, "G0 dense up");
                    assert_within_band(&s_d.down_proj, &d_d.down_proj, BAND, "G0 dense down");
                }
            }
            // Attention grads (MLA at top, KDA below) — verbatim-mirrored
            // code; the band catches a transcription defect in 2b.
            if let (Some(s_m), Some(d_m)) = (s.mla_grads.as_ref(), dl.mla_grads.as_ref()) {
                macro_rules! mla_band {
                    ($f:ident, $what:literal) => {
                        assert_within_band(&s_m.$f, &d_m.$f, BAND, $what);
                    };
                }
                mla_band!(w_dkv, "G0 mla w_dkv");
                mla_band!(w_dq, "G0 mla w_dq");
                mla_band!(w_uq, "G0 mla w_uq");
                mla_band!(w_qr, "G0 mla w_qr");
                mla_band!(w_uk, "G0 mla w_uk");
                mla_band!(w_uv, "G0 mla w_uv");
                mla_band!(w_kr, "G0 mla w_kr");
                mla_band!(w_o, "G0 mla w_o");
                mla_band!(q_a_norm_weight, "G0 mla q_a_norm");
                mla_band!(kv_a_norm_weight, "G0 mla kv_a_norm");
                if let (Some(s_g), Some(d_g)) = (s_m.w_g.as_ref(), d_m.w_g.as_ref()) {
                    assert_within_band(s_g, d_g, BAND, "G0 mla w_g");
                }
            }
            if let (Some(s_k), Some(d_k)) = (s.kda_grads.as_ref(), dl.kda_grads.as_ref()) {
                macro_rules! kda_band {
                    ($f:ident, $what:literal) => {
                        assert_within_band(&s_k.$f, &d_k.$f, BAND, $what);
                    };
                }
                kda_band!(q_proj, "G0 kda q_proj");
                kda_band!(k_proj, "G0 kda k_proj");
                kda_band!(v_proj, "G0 kda v_proj");
                kda_band!(q_conv_weight, "G0 kda q_conv");
                kda_band!(k_conv_weight, "G0 kda k_conv");
                kda_band!(v_conv_weight, "G0 kda v_conv");
                kda_band!(a_log, "G0 kda a_log");
                kda_band!(f_a_proj, "G0 kda f_a_proj");
                kda_band!(f_b_proj, "G0 kda f_b_proj");
                kda_band!(dt_bias, "G0 kda dt_bias");
                kda_band!(beta_proj, "G0 kda beta_proj");
                kda_band!(g_proj, "G0 kda g_proj");
                kda_band!(o_norm_weight, "G0 kda o_norm");
                kda_band!(o_proj, "G0 kda o_proj");
            }
        }
    }

    /// Delay-arm forward loss for the FD checks (fresh runtime per eval —
    /// deterministic, the existing FD convention).
    fn run_delay_forward_loss(
        config: &KimiK3ModelConfig,
        weights: &KimiK3ModelWeights,
        tokens: &[u32],
        delay: &DelayArchConfig,
    ) -> f32 {
        let mut runtime = KimiK3Runtime::new(config, tokens.len());
        let mut saved = DelayTokenSavedActivations::new();
        let mut loss = 0.0f32;
        for (pos, &tok) in tokens.iter().enumerate() {
            kimi_k3_delay_forward_token_saved(
                config,
                weights,
                &mut runtime,
                tok,
                pos,
                delay,
                &mut saved,
            );
            loss += saved.base.logits.iter().map(|&v| v * v).sum::<f32>();
        }
        loss
    }

    fn rel_err(analytic: f32, numeric: f32) -> f32 {
        let denom = analytic.abs().max(numeric.abs()).max(1e-2);
        (analytic - numeric).abs() / denom
    }

    /// Reborrow helper for the FD macro (see use-site comment).
    fn moe_w_mut(layer_ffn: &mut KimiFfnWeights) -> &mut katgpt_transformer::moe::MoeWeights {
        match layer_ffn {
            KimiFfnWeights::Moe(w) => w,
            _ => panic!("FD helper: layer is not MoE"),
        }
    }

    /// Per-arm finite-difference gradient check (the
    /// `kimi_k3_backward_grad_check` pattern) over a param spread that
    /// covers every NEW gradient path: the anchor norm (post_ln at MoE
    /// layers), the split streams (router/expert/shared), the entry-deposit
    /// end-to-end path (embed), and the untouched attention path (input_ln).
    ///
    /// L=4 so δe=2 exercises BOTH an in-loop injection (source 1 → slot 3)
    /// and tail flushes (sources 2, 3 → past the end) — the plan's 2-layer
    /// toy could not reach the pending-injection mechanism.
    fn fd_check_arm(dense_delay: u8, expert_delay: u8, seed: u64) {
        let config = toy_config(true, 4);
        let delay = DelayArchConfig::new(dense_delay, expert_delay);
        let weights = KimiK3ModelWeights::random(&config, seed);
        let tokens: Vec<u32> = vec![2, 5, 1];
        let l = tokens.len();

        let mut runtime = KimiK3Runtime::new(&config, l);
        let mut saved_tokens: Vec<DelayTokenSavedActivations> =
            (0..l).map(|_| DelayTokenSavedActivations::new()).collect();
        for (pos, &tok) in tokens.iter().enumerate() {
            kimi_k3_delay_forward_token_saved(
                &config,
                &weights,
                &mut runtime,
                tok,
                pos,
                &delay,
                &mut saved_tokens[pos],
            );
        }
        let d_logits: Vec<Vec<f32>> = saved_tokens
            .iter()
            .map(|s| s.base.logits.iter().map(|&v| 2.0 * v).collect())
            .collect();
        let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
        kimi_k3_delay_backward_sequence(
            &config,
            &weights,
            &runtime,
            &saved_tokens,
            &d_logits,
            &delay,
            &mut grads,
        );

        let mut max_rel_err = 0.0f32;
        let mut max_rel_err_label = String::new();
        let mut weights_mut = weights.clone();
        let epsilon = 5e-3f32;

        macro_rules! fd_check {
            ($get_mut:expr, $analytic:expr, $label:expr) => {{
                let orig = *$get_mut;
                *$get_mut = orig + epsilon;
                let lp = run_delay_forward_loss(&config, &weights_mut, &tokens, &delay);
                *$get_mut = orig - epsilon;
                let lm = run_delay_forward_loss(&config, &weights_mut, &tokens, &delay);
                *$get_mut = orig;
                let numeric = (lp - lm) / (2.0 * epsilon);
                let err = rel_err($analytic, numeric);
                if err > max_rel_err {
                    max_rel_err = err;
                    max_rel_err_label = $label.to_string();
                }
            }};
        }

        // End-to-end through the entry deposits.
        for j in 0..4 {
            fd_check!(&mut weights_mut.embed_weight[j], grads.embed_weight[j], format!("embed[{j}]"));
        }
        fd_check!(
            &mut weights_mut.final_norm_weight[1],
            grads.final_norm_weight[1],
            "final_norm[1]"
        );
        // Anchor norm at MoE layers (γ feeds both streams' anchor) + the
        // dense layer's δd=1 anchor path.
        for li in 0..config.num_layers {
            fd_check!(
                &mut weights_mut.layers[li].post_attention_layernorm_weight[2],
                grads.layers[li].post_attention_layernorm_weight[2],
                format!("L{li}_post_ln[2]")
            );
        }
        // Untouched attention path (sanity).
        fd_check!(
            &mut weights_mut.layers[2].input_layernorm_weight[1],
            grads.layers[2].input_layernorm_weight[1],
            "L2_input_ln[1]"
        );
        // Split streams at a mid MoE layer: router + selected experts +
        // shared + the latent projections. The reborrow helper keeps each
        // macro statement's mutable borrow disjoint from the immutable
        // `&weights_mut` the loss evals take (NLL, the existing FD pattern).
        {
            let li = 2;
            let moe_g = grads.layers[li].moe_grads.as_ref().unwrap();
            let mut sel: Vec<usize> = Vec::new();
            for sv in &saved_tokens {
                for &idx in &sv.base.layers[li].moe_saved.as_ref().unwrap().topk_indices {
                    if !sel.contains(&idx) {
                        sel.push(idx);
                    }
                }
            }
            for j in 0..4 {
                fd_check!(
                    &mut moe_w_mut(&mut weights_mut.layers[li].ffn).router_weight[j],
                    moe_g.router_weight[j],
                    format!("L{li}_router[{j}]")
                );
            }
            for &e in &sel {
                for j in 0..2 {
                    fd_check!(
                        &mut moe_w_mut(&mut weights_mut.layers[li].ffn).experts[e].gate_proj[j],
                        moe_g.experts[e].gate_proj[j],
                        format!("L{li}_expert{e}_gate[{j}]")
                    );
                    fd_check!(
                        &mut moe_w_mut(&mut weights_mut.layers[li].ffn).experts[e].down_proj[j],
                        moe_g.experts[e].down_proj[j],
                        format!("L{li}_expert{e}_down[{j}]")
                    );
                }
            }
            fd_check!(
                &mut moe_w_mut(&mut weights_mut.layers[li].ffn).shared_experts[0].down_proj[1],
                moe_g.shared_experts[0].down_proj[1],
                format!("L{li}_shared_down[1]")
            );
            fd_check!(
                &mut moe_w_mut(&mut weights_mut.layers[li].ffn)
                    .routed_expert_down_proj
                    .as_mut()
                    .unwrap()[3],
                moe_g.routed_expert_down_proj.as_ref().unwrap()[3],
                format!("L{li}_latent_down[3]")
            );
            fd_check!(
                &mut moe_w_mut(&mut weights_mut.layers[li].ffn)
                    .routed_expert_up_proj
                    .as_mut()
                    .unwrap()[5],
                moe_g.routed_expert_up_proj.as_ref().unwrap()[5],
                format!("L{li}_latent_up[5]")
            );
        }

        eprintln!(
            "FD arm δd={dense_delay} δe={expert_delay}: max rel_err = {:.4}% at {}",
            max_rel_err * 100.0,
            max_rel_err_label
        );
        assert!(
            max_rel_err < 8e-2,
            "FD arm δd={dense_delay} δe={expert_delay} FAILED: {:.4}% at {}",
            max_rel_err * 100.0,
            max_rel_err_label
        );
    }

    #[test]
    fn fd_check_d1_e0() {
        fd_check_arm(1, 0, 52);
    }

    #[test]
    fn fd_check_d0_e2() {
        fd_check_arm(0, 2, 53);
    }

    #[test]
    fn fd_check_d1_e2() {
        fd_check_arm(1, 2, 54);
    }

    /// δd=1 pins: the MLP attn-res weights are UNUSED in the forward, so
    /// their grads must be exactly zero — a silent nonzero would mean the
    /// backward ran a path the forward never took.
    #[test]
    fn d1_unused_mlp_attn_res_grads_stay_zero() {
        let config = toy_config(true, 4);
        let delay = DelayArchConfig::new(1, 2);
        let weights = KimiK3ModelWeights::random(&config, 55);
        let tokens: Vec<u32> = vec![2, 5, 1];
        let l = tokens.len();

        let mut runtime = KimiK3Runtime::new(&config, l);
        let mut saved_tokens: Vec<DelayTokenSavedActivations> =
            (0..l).map(|_| DelayTokenSavedActivations::new()).collect();
        for (pos, &tok) in tokens.iter().enumerate() {
            kimi_k3_delay_forward_token_saved(
                &config,
                &weights,
                &mut runtime,
                tok,
                pos,
                &delay,
                &mut saved_tokens[pos],
            );
        }
        let d_logits: Vec<Vec<f32>> = saved_tokens
            .iter()
            .map(|s| s.base.logits.iter().map(|&v| 2.0 * v).collect())
            .collect();
        let mut grads = KimiK3ModelGradients::zeros_like(&config, &weights);
        kimi_k3_delay_backward_sequence(
            &config,
            &weights,
            &runtime,
            &saved_tokens,
            &d_logits,
            &delay,
            &mut grads,
        );
        for (li, lg) in grads.layers.iter().enumerate() {
            assert!(
                lg.mlp_attn_res_norm.iter().all(|&v| v == 0.0),
                "L{li} mlp_attn_res_norm nonzero under δd=1"
            );
            assert!(
                lg.mlp_attn_res_proj.iter().all(|&v| v == 0.0),
                "L{li} mlp_attn_res_proj nonzero under δd=1"
            );
        }
        // And the forward side of the invariant: no mixed_mlp saved under δd=1.
        for sv in &saved_tokens {
            for (li, layer) in sv.base.layers.iter().enumerate() {
                assert!(
                    layer.mixed_mlp.is_empty() && layer.block_state_mlp.is_empty(),
                    "L{li} mixed_mlp/block_state_mlp populated under δd=1"
                );
            }
        }
    }
}
