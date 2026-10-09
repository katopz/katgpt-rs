//! Kimi-K3 full-model analytic backward (Plan 318 Phase C C6).
//!
//! Composes the three per-primitive backward modules (MLA + MoE + KDA from
//! C4/C5) with the model-level composition backward (attn-res + RMSNorm +
//! dense SiTU FFN + LM head + embedding) into a full-model gradient pass.
//!
//! # Architecture
//!
//! The Kimi-K3 model uses a non-standard residual structure:
//! - A running `prefix_sum` accumulates attention + FFN outputs across layers
//! - At block boundaries (`layer_idx % block_size == 0`), the current
//!   `prefix_sum` is pushed to `block_state` and reset to zero
//! - Before each sub-layer (attention + FFN), `prefix_sum` is mixed with past
//!   block residuals via `apply_attn_res` (softmax-weighted averaging)
//!
//! The backward distributes gradients through this prefix_sum + block_state +
//! attn-res pattern.
//!
//! # Cross-token gradient flow
//!
//! MLA handles cross-token KV-cache gradients internally (each token attends to
//! ALL cached tokens via shared W_UK/W_UV). KDA uses BPTT across the conv ring +
//! SSM state. MoE/Dense FFN are stateless. The model-level backward processes
//! layers in reverse; for each layer, it collects dL/d(attention_output) for
//! all tokens, then runs the primitive backward (MLA cross-token or KDA BPTT).
//!
//! # Scope
//!
//! CPU reference for the GPU training loop (C10). Gated behind `kimi_k3_backward`.

use katgpt_attn::gdn2::kda_backward::{
    KdaGradients, KdaSavedActivations, kda_backward_sequence, kda_forward_token_with_saved,
};
use katgpt_attn::mla_backward::{
    MlaGradients, MlaSavedActivations, mla_backward_token, mla_forward_token_with_saved,
    rmsnorm_backward as mla_rmsnorm_backward,
};
use katgpt_core::simd::{
    simd_dot_f32, simd_matmul_rows, simd_outer_product_acc, simd_sum_sq, simd_transpose_matvec_into,
};
use katgpt_core::types::math::{rmsnorm_with_gamma_eps, situ};
use katgpt_kv::shard_kv::rope::RopeFreqs;
use katgpt_transformer::attn_res::{
    AttnResBlockState, AttnResConfig, AttnResScratch, AttnResWeights, apply_attn_res,
};
use katgpt_transformer::moe::SwiGluExpertWeights;
use katgpt_transformer::moe_backward::{
    MoeGradients, MoeSavedActivations, moe_backward_token, moe_forward_token_with_saved,
};

use super::decoder_layer::{
    KimiAttentionConfig, KimiAttentionScratch, KimiAttentionState, KimiAttentionWeights,
    KimiDecoderLayerConfig, KimiDecoderLayerWeights, KimiFfnConfig, KimiFfnScratch, KimiFfnWeights,
};
use super::loader::KimiK3ModelWeights;
use super::model::{KimiK3ModelConfig, KimiK3Runtime};

// ─── Dense SiTU FFN saved activations (layer 0 only) ────────────────────────

#[derive(Clone)]
pub struct DenseFfnSavedActivations {
    pub h: Vec<f32>,
    pub gate_inter: Vec<f32>,
    pub up_inter: Vec<f32>,
    pub act_out: Vec<f32>,
}

/// Dense FFN gradients (gate_proj, up_proj, down_proj).
pub struct DenseFfnGradients {
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

// ─── Per-layer saved activations ───────────────────────────────────────────

/// Saved activations for one decoder layer at one token position.
#[derive(Clone)]
pub struct LayerSavedActivations {
    /// `prefix_sum` at layer entry (input from previous layer / embedding). `[d]`.
    pub prefix_sum_in: Vec<f32>,
    /// Whether `apply_attn_res` was called for the self block.
    pub has_self_attn_res: bool,
    /// The mixed hidden (input to input_layernorm). `[d]`.
    pub mixed_self: Vec<f32>,
    /// RMSNorm inv_rms for input_layernorm.
    pub self_inv_rms: f32,
    /// Whether this layer is a block boundary.
    pub is_boundary: bool,
    /// Block-state entries snapshot BEFORE the self-attn-res mixing. `[num_entries][d]`.
    pub block_state_self: Vec<Vec<f32>>,
    /// Attention output. `[d]`.
    pub attn_out: Vec<f32>,
    /// `prefix_sum` after attention accumulation. `[d]`.
    pub prefix_sum_after_attn: Vec<f32>,
    /// The mixed hidden (input to post_attention_layernorm). `[d]`.
    pub mixed_mlp: Vec<f32>,
    /// RMSNorm inv_rms for post_attention_layernorm.
    pub mlp_inv_rms: f32,
    /// Block-state entries for the MLP attn-res. `[num_entries][d]`.
    pub block_state_mlp: Vec<Vec<f32>>,
    /// FFN output. `[d]`.
    pub ffn_out: Vec<f32>,
    /// MLA saved activations (if MLA layer).
    pub mla_saved: Option<MlaSavedActivations>,
    /// KDA saved activations (if KDA layer).
    pub kda_saved: Option<KdaSavedActivations>,
    /// MoE saved activations (if MoE layer).
    pub moe_saved: Option<MoeSavedActivations>,
    /// Dense FFN intermediates (if dense layer).
    pub dense_saved: Option<DenseFfnSavedActivations>,
}

/// Model-level saved activations for one token.
pub struct TokenSavedActivations {
    pub token_id: u32,
    pub layers: Vec<LayerSavedActivations>,
    /// `prefix_sum` after last layer (before output attn-res). `[d]`.
    pub prefix_sum_final: Vec<f32>,
    /// Hidden after output attn-res, before final norm (= final norm input). `[d]`.
    pub pre_final_norm: Vec<f32>,
    /// Block-state snapshot before output attn-res. `[num_entries][d]`.
    pub block_state_final: Vec<Vec<f32>>,
    pub has_output_attn_res: bool,
    /// Final normed hidden (input to LM head). `[d]`.
    pub final_hidden: Vec<f32>,
    pub final_norm_inv_rms: f32,
    /// Logits. `[vocab_size]`.
    pub logits: Vec<f32>,
    pub pos: usize,
    /// True when this position's input hidden state was supplied by the caller
    /// (`kimi_k3_forward_token_hidden_saved`) rather than looked up from the
    /// embedding table.
    ///
    /// Plan 340: `token_id` is repurposed as an iteration index for such
    /// positions, so the embedding backward MUST NOT use it as a row index —
    /// doing so scatters the position's input gradient onto an unrelated
    /// embedding row. See `kimi_k3_backward_sequence_with_input_grad`.
    pub is_latent: bool,
}

impl TokenSavedActivations {
    /// Allocate an empty struct (layers vector empty; will be filled by forward).
    pub fn new() -> Self {
        Self {
            token_id: 0,
            layers: Vec::new(),
            prefix_sum_final: Vec::new(),
            pre_final_norm: Vec::new(),
            block_state_final: Vec::new(),
            has_output_attn_res: false,
            final_hidden: Vec::new(),
            final_norm_inv_rms: 0.0,
            logits: Vec::new(),
            pos: 0,
            is_latent: false,
        }
    }
}

impl Default for TokenSavedActivations {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Per-layer gradients ───────────────────────────────────────────────────

pub struct LayerGradients {
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub self_attn_res_norm: Vec<f32>,
    pub self_attn_res_proj: Vec<f32>,
    pub mlp_attn_res_norm: Vec<f32>,
    pub mlp_attn_res_proj: Vec<f32>,
    pub mla_grads: Option<MlaGradients>,
    pub kda_grads: Option<KdaGradients>,
    pub moe_grads: Option<MoeGradients>,
    pub dense_grads: Option<DenseFfnGradients>,
}

impl LayerGradients {
    pub fn zeros_like(
        config: &KimiK3ModelConfig,
        layer_idx: usize,
        weights: &KimiDecoderLayerWeights,
    ) -> Self {
        let d = config.hidden_size;
        let is_mla = config.is_mla_layer(layer_idx);
        let is_dense = layer_idx == 0;

        let (mla_grads, kda_grads) = if is_mla {
            let KimiAttentionWeights::Mla(w) = &weights.attention else {
                panic!("MLA layer but non-MLA weights");
            };
            (Some(MlaGradients::zeros_like(w)), None)
        } else {
            let KimiAttentionWeights::Kda(w) = &weights.attention else {
                panic!("KDA layer but non-KDA weights");
            };
            (None, Some(KdaGradients::zeros_like(w)))
        };

        let (moe_grads, dense_grads) = if is_dense {
            let KimiFfnWeights::Dense(e) = &weights.ffn else {
                panic!("Dense layer but non-dense FFN");
            };
            (
                None,
                Some(DenseFfnGradients {
                    gate_proj: vec![0.0; e.gate_proj.len()],
                    up_proj: vec![0.0; e.up_proj.len()],
                    down_proj: vec![0.0; e.down_proj.len()],
                }),
            )
        } else {
            let KimiFfnWeights::Moe(w) = &weights.ffn else {
                panic!("MoE layer but non-MoE FFN");
            };
            (Some(MoeGradients::zeros_like(w)), None)
        };

        Self {
            input_layernorm_weight: vec![0.0; d],
            post_attention_layernorm_weight: vec![0.0; d],
            self_attn_res_norm: vec![0.0; d],
            self_attn_res_proj: vec![0.0; d],
            mlp_attn_res_norm: vec![0.0; d],
            mlp_attn_res_proj: vec![0.0; d],
            mla_grads,
            kda_grads,
            moe_grads,
            dense_grads,
        }
    }
}

/// Full model gradient accumulator.
pub struct KimiK3ModelGradients {
    pub embed_weight: Vec<f32>,
    pub layers: Vec<LayerGradients>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
    pub output_attn_res_norm: Vec<f32>,
    pub output_attn_res_proj: Vec<f32>,
}

impl KimiK3ModelGradients {
    pub fn zeros_like(config: &KimiK3ModelConfig, weights: &KimiK3ModelWeights) -> Self {
        let layers = (0..config.num_layers)
            .map(|i| LayerGradients::zeros_like(config, i, &weights.layers[i]))
            .collect();
        Self {
            embed_weight: vec![0.0; weights.embed_weight.len()],
            layers,
            final_norm_weight: vec![0.0; weights.final_norm_weight.len()],
            lm_head_weight: vec![0.0; weights.lm_head_weight.len()],
            output_attn_res_norm: vec![0.0; config.hidden_size],
            output_attn_res_proj: vec![0.0; config.hidden_size],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward with saved activations
// ═══════════════════════════════════════════════════════════════════════════

/// Run the full Kimi-K3 forward for one token, saving all activations.
///
/// Mirrors `kimi_k3_forward_token` exactly but captures all intermediates.
/// Returns a reference to `runtime.logits`.
#[allow(clippy::too_many_arguments)]
pub fn kimi_k3_forward_token_saved(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    token_id: u32,
    pos: usize,
    saved: &mut TokenSavedActivations,
) {
    let d = config.hidden_size;

    runtime.block_state.clear();
    saved.layers.clear();
    saved.pos = pos;
    saved.token_id = token_id;
    saved.is_latent = false;

    // Embedding lookup
    let embed_start = (token_id as usize) * d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_start + d]);

    // Decoder layers
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

        forward_layer_saved(
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
            &mut layer_saved,
        );

        saved.layers.push(layer_saved);
    }

    // Output attn-res
    saved.prefix_sum_final = runtime.hidden.clone();
    saved.block_state_final = runtime.block_state.residuals.clone();
    saved.has_output_attn_res = !runtime.block_state.is_empty();

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

    // Save post-output-attn-res hidden (= final norm input)
    saved.pre_final_norm = runtime.hidden.clone();

    // Final RMSNorm
    let sum_sq = simd_sum_sq(&runtime.hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + config.rms_eps).sqrt());
    saved.final_norm_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );
    saved.final_hidden = runtime.hidden.clone();

    // LM head
    simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );
    saved.logits = runtime.logits.clone();
}

/// Run the full Kimi-K3 forward starting from an arbitrary hidden state,
/// saving all activations (Plan 324 Phase A4 — Ouro looped training).
///
/// Mirrors [`kimi_k3_forward_token_hidden`] (the inference variant) but
/// captures all intermediates for the analytic backward pass. This is the
/// saved-activations counterpart of `kimi_k3_forward_token_hidden`: same
/// decoder path (clear block_state → skip embedding → layers → output
/// attn-res → final norm → LM head), but the caller must set `runtime.hidden`
/// to the desired input hidden state before calling.
///
/// Used by the Ouro/LoopLM looped training recipe: each loop iteration after
/// the first feeds the previous iteration's final hidden state back through
/// the full decoder stack. The KDA recurrent state + MLA KV cache accumulate
/// across iterations (handled by the per-layer forward primitives); only the
/// `block_state` (residual stream) is reset per iteration.
///
/// **Block state contract:** clears `runtime.block_state` at entry (step 0),
/// matching `kimi_k3_forward_token_saved`. The caller does NOT need to clear
/// it separately.
///
/// **Saved activations contract:** `saved.layers` is cleared at entry. The
/// `saved.token_id` and `saved.pos` fields are set to the caller-provided
/// values (the loop iteration uses the same `pos` as the first iteration —
/// RoPE is position-dependent, not iteration-dependent).
#[allow(clippy::too_many_arguments)]
pub fn kimi_k3_forward_token_hidden_saved(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    pos: usize,
    iteration: usize,
    saved: &mut TokenSavedActivations,
) {
    let d = config.hidden_size;

    // Step 0: reset per-iteration block state (matches forward_token_hidden).
    runtime.block_state.clear();
    saved.layers.clear();
    saved.pos = pos;
    // token_id is meaningless for hidden-input iterations; store the iteration
    // index so downstream consumers can distinguish saved-activation sets.
    saved.token_id = iteration as u32;
    // The input hidden state came from the caller, so `token_id` above is an
    // iteration index and is NOT a valid embedding row. The embedding backward
    // keys off this flag rather than the id.
    saved.is_latent = true;

    // Step 1 (embedding) is SKIPPED — runtime.hidden is whatever the caller
    // set (the previous iteration's final hidden state, post-LM-head-pre-norm).
    // The caller is responsible for ensuring runtime.hidden has length d.
    debug_assert_eq!(
        runtime.hidden.len(),
        d,
        "runtime.hidden must be set to [hidden_size] before calling forward_token_hidden_saved"
    );

    // Steps 2-5: same decoder path as kimi_k3_forward_token_saved.
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

        forward_layer_saved(
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
            &mut layer_saved,
        );

        saved.layers.push(layer_saved);
    }

    // Output attn-res (same as forward_token_saved).
    saved.prefix_sum_final = runtime.hidden.clone();
    saved.block_state_final = runtime.block_state.residuals.clone();
    saved.has_output_attn_res = !runtime.block_state.is_empty();

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

    // Save post-output-attn-res hidden (= final norm input).
    saved.pre_final_norm = runtime.hidden.clone();

    // Final RMSNorm.
    let sum_sq = simd_sum_sq(&runtime.hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + config.rms_eps).sqrt());
    saved.final_norm_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );
    saved.final_hidden = runtime.hidden.clone();

    // LM head.
    simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );
    saved.logits = runtime.logits.clone();
}

#[allow(clippy::too_many_arguments)]
fn forward_layer_saved(
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
    saved: &mut LayerSavedActivations,
) {
    let d = config.attn_res.d();
    let eps = config.rms_eps;
    let block_size = config.attn_res.block_size;
    let is_boundary = layer_idx.is_multiple_of(block_size);
    saved.is_boundary = is_boundary;

    // Step 1: self-attn-res mixing
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

    // Step 2: boundary push
    if is_boundary {
        block_state.push(prefix_sum);
        // `fill` lowers to a memset instead of a per-element store loop.
        prefix_sum.fill(0.0);
    }

    // Step 3: input_layernorm + attention
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
    // Move (not clone) the freshly-allocated attn_out into `saved` and read it
    // back through `saved.attn_out` — saves one `[d]` Vec allocation + copy per
    // layer per token. Values and accumulation order are untouched.
    saved.attn_out = attn_out;

    // Step 4: prefix_sum += attn_out
    // Elementwise SIMD add: each lane is a single independent `a + b`, so no
    // reassociation and the gradients that later consume `prefix_sum` are
    // bit-identical. `[..d]` keeps the old panic-on-short-operand behaviour.
    katgpt_core::simd::simd_add_inplace(&mut prefix_sum[..d], &saved.attn_out[..d]);
    saved.prefix_sum_after_attn = prefix_sum.to_vec();

    // Step 5: MLP attn-res
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

    // Step 6: post_attention_layernorm + FFN
    let sum_sq = simd_sum_sq(scratch_hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
    saved.mlp_inv_rms = inv_rms;
    rmsnorm_with_gamma_eps(
        scratch_hidden,
        &weights.post_attention_layernorm_weight,
        eps as f64,
    );

    let ffn_out: Vec<f32> = match (&config.ffn, &weights.ffn) {
        (
            KimiFfnConfig::Dense {
                situ_beta,
                situ_linear_beta,
                ..
            },
            KimiFfnWeights::Dense(expert),
        ) => {
            let (out, s) = dense_situ_ffn_forward_saved(
                expert,
                scratch_hidden,
                ffn_scratch,
                *situ_beta,
                *situ_linear_beta,
            );
            saved.dense_saved = Some(s);
            out
        }
        (KimiFfnConfig::Moe(cfg), KimiFfnWeights::Moe(w)) => {
            let (out, s) =
                moe_forward_token_with_saved(w, cfg, scratch_hidden, &mut ffn_scratch.moe);
            ffn_scratch.dense_out[..d].copy_from_slice(&out[..d]);
            saved.moe_saved = Some(s);
            out
        }
        _ => panic!("FFN mismatch"),
    };
    // Move (not clone) — same rationale as `saved.attn_out` above.
    saved.ffn_out = ffn_out;

    // Step 7: prefix_sum += ffn_out — elementwise SIMD add, see Step 4.
    katgpt_core::simd::simd_add_inplace(&mut prefix_sum[..d], &saved.ffn_out[..d]);
}

pub(crate) fn dense_situ_ffn_forward_saved(
    expert: &SwiGluExpertWeights,
    hidden: &[f32],
    scratch: &mut KimiFfnScratch,
    beta: f32,
    linear_beta: Option<f32>,
) -> (Vec<f32>, DenseFfnSavedActivations) {
    let d_in = scratch.dense_out.len();
    let d_ffn = expert.gate_proj.len() / d_in;

    simd_matmul_rows(
        &mut scratch.dense_gate,
        &expert.gate_proj,
        hidden,
        d_ffn,
        d_in,
    );
    simd_matmul_rows(&mut scratch.dense_up, &expert.up_proj, hidden, d_ffn, d_in);
    situ(
        &mut scratch.dense_act,
        &scratch.dense_gate,
        &scratch.dense_up,
        beta,
        linear_beta,
    );
    simd_matmul_rows(
        &mut scratch.dense_out,
        &expert.down_proj,
        &scratch.dense_act,
        d_in,
        d_ffn,
    );

    let saved = DenseFfnSavedActivations {
        h: hidden.to_vec(),
        gate_inter: scratch.dense_gate.clone(),
        up_inter: scratch.dense_up.clone(),
        act_out: scratch.dense_act.clone(),
    };
    (scratch.dense_out.clone(), saved)
}

// ═══════════════════════════════════════════════════════════════════════════
// Backward
// ═══════════════════════════════════════════════════════════════════════════

/// Full-model backward for a sequence of tokens.
///
/// Computes gradients w.r.t. all model parameters by composing the three
/// per-primitive backward modules (MLA + MoE + KDA) with the model-level
/// composition backward (attn-res + RMSNorm + dense FFN + LM head + embedding).
///
/// # Gradient flow
///
/// Forward per token: `embed → [layers] → output_attn_res → final_norm → lm_head → logits`
///
/// Each layer forward (prefix_sum pattern):
/// ```text
/// mixed_self = attn_res(ps_in, blocks_self) or ps_in
/// if boundary: blocks.push(ps_in); ps_in = 0
/// attn_out = attention(rmsnorm(mixed_self))
/// ps_after_attn = ps_in + attn_out
/// mixed_mlp = attn_res(ps_after_attn, blocks_mlp)
/// ffn_out = ffn(rmsnorm(mixed_mlp))
/// ps_out = ps_after_attn + ffn_out
/// ```
///
/// # Arguments
/// - `config` — model config
/// - `weights` — model weights
/// - `runtime` — runtime state (for MLA KV cache access)
/// - `saved_tokens` — saved activations from forward (L tokens)
/// - `d_logits` — per-token gradient w.r.t. logits `[L][vocab_size]`
/// - `grads` — gradient accumulator (must be zeroed)
pub fn kimi_k3_backward_sequence(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &KimiK3Runtime,
    saved_tokens: &[TokenSavedActivations],
    d_logits: &[Vec<f32>],
    grads: &mut KimiK3ModelGradients,
) {
    kimi_k3_backward_sequence_with_input_grad(
        config,
        weights,
        runtime,
        saved_tokens,
        d_logits,
        grads,
        None,
    );
}

/// [`kimi_k3_backward_sequence`], additionally returning dL/d(input hidden).
///
/// `d_input_hidden`, when supplied, must have one `[d]`-wide slot per saved token
/// and receives `d_prefix[t]` — the gradient at the decoder stack's *input*. Two
/// consumers need it:
///
/// - **Prefix/latent tuning (Plan 340 LOPD):** the composer's parameters live
///   upstream of the model, so its only gradient path is through the injected
///   hidden state. Without this out-parameter a composer receives nothing.
/// - **Looped training (Plan 324 Ouro):** iteration `k`'s input gradient is
///   iteration `k+1`'s output gradient.
///
/// **Embedding scatter contract.** For a position with `is_latent == false` the
/// gradient is accumulated into `grads.embed_weight[token_id * d ..]` as before.
/// For `is_latent == true` that scatter is **skipped**: such a position's
/// `token_id` is an iteration index, not an embedding row, so scattering there
/// would corrupt an unrelated row of a table the caller believes is frozen —
/// silently, since nothing about it is ill-formed. The gradient is delivered via
/// `d_input_hidden` instead, which is the only correct destination for it.
pub fn kimi_k3_backward_sequence_with_input_grad(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &KimiK3Runtime,
    saved_tokens: &[TokenSavedActivations],
    d_logits: &[Vec<f32>],
    grads: &mut KimiK3ModelGradients,
    mut d_input_hidden: Option<&mut [Vec<f32>]>,
) {
    let d = config.hidden_size;
    let v = config.vocab_size;
    let l = saved_tokens.len();
    debug_assert_eq!(d_logits.len(), l);
    if l == 0 {
        return;
    }

    // ── Step 1: Per-token output backward (LM head + final norm + output attn-res) ──
    // Produces d_prefix_final[t] = dL/d(prefix_sum at model exit) for each token.
    let mut d_prefix: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
    // Block-state gradient accumulators (per token). These track gradient w.r.t.
    // block entries pushed at boundary layers. Entries get popped in reverse
    // layer processing when the originating boundary layer is reached.
    let mut block_grads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l);

    for t in 0..l {
        let saved = &saved_tokens[t];

        // LM head backward: dL/d(final_hidden) = lm_head^T · dL/d(logits)
        let mut d_fh = vec![0.0f32; d];
        simd_transpose_matvec_into(&mut d_fh, &weights.lm_head_weight, &d_logits[t], v, d);
        simd_outer_product_acc(
            &mut grads.lm_head_weight,
            &d_logits[t],
            &saved.final_hidden,
            v,
            d,
        );

        // Final RMSNorm backward → dL/d(pre_final_norm = post-output-attn-res hidden)
        let d_pre_fn = mla_rmsnorm_backward(
            &d_fh,
            &saved.pre_final_norm,
            &weights.final_norm_weight,
            saved.final_norm_inv_rms,
            &mut grads.final_norm_weight,
            config.rms_eps,
        );

        // Output attn-res backward
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

    // Hoisted out of the layer loop: `d_attn_out[t]` and `d_ps_after_attn_all[t]`
    // are both established by a full-width `copy_from_slice` for EVERY `t` in
    // `0..l` inside the per-token loop below (which has no `break`/`continue`),
    // so no slot can be read before it is written in the current layer's pass.
    // Reusing them turns 2 × l Vec allocations *per layer* into 2 × l total.
    // ── Step 2a scratch: Produces d_attn_out[t] = dL/d(attention_output) per token.
    let mut d_attn_out: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
    // Also save d_ps_after_attn for the self-attn residual backward.
    let mut d_ps_after_attn_all: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

    // ── Step 2: Layer-by-layer backward (reverse) ──
    for layer_idx in (0..config.num_layers).rev() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_w = &weights.layers[layer_idx];
        let layer_grads = &mut grads.layers[layer_idx];
        let is_mla = config.is_mla_layer(layer_idx);
        let is_boundary = layer_idx.is_multiple_of(config.attn_res_config.block_size);

        for t in 0..l {
            let saved = &saved_tokens[t].layers[layer_idx];

            // FFN residual: ps_out = ps_after_attn + ffn_out
            // dL/d(ffn_out) = d_prefix[t]; dL/d(ps_after_attn)_from_ffn = d_prefix[t]
            // `ffn_backward` takes `d_output: &[f32]` and never mutates it, so
            // borrow `d_prefix[t]` directly instead of cloning a `[d]` Vec per
            // token per layer.

            // FFN backward → dL/d(normed_mlp) + FFN weight grads
            let d_normed_mlp = ffn_backward(
                &layer_cfg.ffn,
                &layer_w.ffn,
                saved,
                &d_prefix[t],
                layer_grads,
            );

            // Post-attn RMSNorm backward → dL/d(mixed_mlp)
            let d_mixed_mlp = mla_rmsnorm_backward(
                &d_normed_mlp,
                &saved.mixed_mlp,
                &layer_w.post_attention_layernorm_weight,
                saved.mlp_inv_rms,
                &mut layer_grads.post_attention_layernorm_weight,
                config.rms_eps,
            );

            // MLP attn-res backward → distributes to ps_after_attn + block grads
            let num_mlp_blocks = saved.block_state_mlp.len();
            while block_grads[t].len() < num_mlp_blocks {
                block_grads[t].push(vec![0.0f32; d]);
            }
            let mut d_mlp_blocks: Vec<Vec<f32>> =
                (0..num_mlp_blocks).map(|_| vec![0.0f32; d]).collect();
            // d_prefix[t] is the FFN residual gradient; now add attn-res contribution.
            // Accumulate straight into the pre-allocated `d_attn_out[t]` row instead
            // of cloning `d_prefix[t]` into a fresh Vec and then cloning that again
            // below: same starting contents, same in/out accumulation by
            // `attn_res_backward`, and `d_attn_out[t]` was going to receive exactly
            // this value anyway. Drops 2 of the 3 `[d]` allocations per token per layer.
            d_attn_out[t].copy_from_slice(&d_prefix[t]); // dL/d(ps_after_attn)_from_ffn

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

            // Accumulate block grads.
            // `block_grads[t]` may be LONGER than `d_mlp_blocks` (the `while` above
            // only grows it to at least `num_mlp_blocks`) and `d_mlp_blocks` has
            // exactly `num_mlp_blocks` rows, so `zip` visits exactly the same
            // `0..num_mlp_blocks` rows the indexed loop did. Pre-slicing the rows
            // drops the `Vec<Vec<f32>>` re-indexing + bounds checks per element;
            // elementwise `+=` means no reassociation.
            for (bg_row, d_row) in block_grads[t].iter_mut().zip(d_mlp_blocks.iter()) {
                let bg = &mut bg_row[..d];
                let dr = &d_row[..d];
                for j in 0..d {
                    bg[j] += dr[j];
                }
            }

            // d_attn_out[t] is now dL/d(ps_after_attn) = FFN residual + attn-res
            // contribution; mirror it into d_ps_after_attn_all[t].
            d_ps_after_attn_all[t].copy_from_slice(&d_attn_out[t]);
        }

        // ── Step 2b: Attention backward (MLA cross-token or KDA BPTT) ──
        // Produces d_normed_self[t] = dL/d(normed_self) for each token.
        let mut d_normed_self: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

        if is_mla {
            let KimiAttentionWeights::Mla(mla_w) = &layer_w.attention else {
                panic!("MLA layer but non-MLA weights");
            };
            let KimiAttentionState::Mla(cache) = &runtime.layers[layer_idx].attn_state else {
                panic!("MLA layer but non-MLA cache");
            };

            // Collect all MLA saved activations for this layer.
            // mla_backward_token needs &[MlaSavedActivations] (owned refs).
            let all_saved: Vec<MlaSavedActivations> = (0..l)
                .map(|t| saved_tokens[t].layers[layer_idx].mla_saved.clone().unwrap())
                .collect();

            // all_dh[t] accumulates dL/d(input hidden to MLA) = dL/d(normed_self).
            let mut all_dh: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
            let mla_grads = layer_grads.mla_grads.as_mut().unwrap();
            // Create a fresh RopeFreqs for the backward (MLA backward mutates it
            // for RoPE inversion via apply(negate=true)).
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
            // KDA BPTT across the sequence
            let KimiAttentionWeights::Kda(kda_w) = &layer_w.attention else {
                panic!("KDA layer but non-KDA weights");
            };
            let all_saved_kda: Vec<KdaSavedActivations> = (0..l)
                .map(|t| saved_tokens[t].layers[layer_idx].kda_saved.clone().unwrap())
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

        // ── Step 2c: Self-attn block backward (per token) → d_prefix[t] for previous layer ──
        for t in 0..l {
            let saved = &saved_tokens[t].layers[layer_idx];

            // Input RMSNorm backward → dL/d(mixed_self)
            let d_mixed_self = mla_rmsnorm_backward(
                &d_normed_self[t],
                &saved.mixed_self,
                &layer_w.input_layernorm_weight,
                saved.self_inv_rms,
                &mut layer_grads.input_layernorm_weight,
                config.rms_eps,
            );

            // Pop block entry if boundary (gradient for the pushed ps_in).
            // The push happens AFTER self-attn-res but BEFORE attention.
            // So we pop AFTER computing d_mixed_self but BEFORE self-attn-res backward.
            let mut d_ps_in = if is_boundary {
                block_grads[t].pop().unwrap_or_else(|| vec![0.0f32; d])
            } else {
                vec![0.0f32; d]
            };

            // Self-attn residual: for non-boundary, ps_after_attn = ps_in + attn_out
            // → dL/d(ps_in)_from_residual = d_ps_after_attn
            // For boundary, ps_in was reset to 0 → no residual gradient.
            // Elementwise SIMD adds throughout this block: each lane is a single
            // independent `a + b`, so no reassociation — the accumulated gradients
            // are bit-identical to the scalar indexed loops. The `[..d]` slices keep
            // the old panic-on-short-operand behaviour.
            if !is_boundary {
                katgpt_core::simd::simd_add_inplace(
                    &mut d_ps_in[..d],
                    &d_ps_after_attn_all[t][..d],
                );
            }

            // Self attn-res backward (if applied) or copy gradient.
            if saved.has_self_attn_res {
                let num_self_blocks = saved.block_state_self.len();
                // block_grads[t] should have exactly num_self_blocks entries now
                // (after the boundary pop above).
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

                // Accumulate block grads. `d_self_blocks` has exactly
                // `num_self_blocks` rows and `block_grads[t]` has at least that
                // many, so `zip` covers exactly the rows the indexed loop did.
                for (bg_row, d_row) in block_grads[t].iter_mut().zip(d_self_blocks.iter()) {
                    katgpt_core::simd::simd_add_inplace(&mut bg_row[..d], &d_row[..d]);
                }
                // Add attn-res gradient for ps_in
                katgpt_core::simd::simd_add_inplace(&mut d_ps_in[..d], &d_ps_from_attnres[..d]);
            } else {
                // mixed_self = ps_in (copy)
                katgpt_core::simd::simd_add_inplace(&mut d_ps_in[..d], &d_mixed_self[..d]);
            }

            // d_ps_in becomes d_prefix for the previous layer
            d_prefix[t] = d_ps_in;
        }
    }

    // ── Step 3: Embedding backward ──
    // d_prefix[t] = dL/d(input hidden at the decoder stack's entry). For a token
    // position that IS dL/d(embedding row token_id); for a latent position it is
    // the gradient the caller must receive, and there is no embedding row to
    // credit.
    for t in 0..l {
        if let Some(out) = d_input_hidden.as_mut()
            && let Some(slot) = out.get_mut(t)
        {
            slot.clear();
            slot.extend_from_slice(&d_prefix[t][..d]);
        }
        if saved_tokens[t].is_latent {
            // Skip the scatter: `token_id` is an iteration index here. Writing to
            // `embed_weight[iteration * d ..]` would corrupt a row of a table the
            // caller believes is frozen, and nothing would report it.
            continue;
        }
        let token_id = saved_tokens[t].token_id as usize;
        let base = token_id * d;
        // Elementwise SIMD add — same per-lane `a + b`, bit-identical.
        katgpt_core::simd::simd_add_inplace(
            &mut grads.embed_weight[base..base + d],
            &d_prefix[t][..d],
        );
    }
}

/// FFN backward (MoE or Dense). Returns dL/d(normed_mlp) and accumulates weight grads.
pub(crate) fn ffn_backward(
    ffn_config: &KimiFfnConfig,
    ffn_weights: &KimiFfnWeights,
    saved: &LayerSavedActivations,
    d_output: &[f32],
    layer_grads: &mut LayerGradients,
) -> Vec<f32> {
    match (ffn_config, ffn_weights) {
        (
            KimiFfnConfig::Dense {
                situ_beta,
                situ_linear_beta,
                ..
            },
            KimiFfnWeights::Dense(expert),
        ) => {
            let dense_saved = saved.dense_saved.as_ref().unwrap();
            let dense_grads = layer_grads.dense_grads.as_mut().unwrap();
            dense_situ_ffn_backward(
                expert,
                dense_saved,
                d_output,
                dense_grads,
                *situ_beta,
                *situ_linear_beta,
            )
        }
        (KimiFfnConfig::Moe(moe_cfg), KimiFfnWeights::Moe(moe_w)) => {
            let moe_saved = saved.moe_saved.as_ref().unwrap();
            let moe_grads = layer_grads.moe_grads.as_mut().unwrap();
            let mut dh = vec![0.0f32; d_output.len()];
            moe_backward_token(moe_cfg, moe_w, moe_saved, d_output, &mut dh, moe_grads);
            dh
        }
        _ => panic!("FFN config/weights mismatch"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Attn-res backward
// ═══════════════════════════════════════════════════════════════════════════

/// Backward through `apply_attn_res`.
///
/// Forward:
/// ```text
/// score_weight = norm_weight ⊙ proj_weight
/// for each entry v[i] (block residuals + prefix_sum):
///     score[i] = inv_rms(v[i]) * dot(v[i], score_weight)
/// probs = softmax(scores)
/// out = Σ_i probs[i] * v[i]
/// ```
///
/// # Arguments
/// - `d_out` — gradient w.r.t. the attn-res output `[d]`
/// - `block_values` — the block residual entries `[num_blocks][d]`
/// - `prefix_sum` — the prefix_sum entry `[d]`
/// - `d_block_values` — accumulator for block residual gradients (mutated)
/// - `d_prefix_sum` — accumulator for prefix_sum gradient (mutated, += )
/// - `d_norm_weight`, `d_proj_weight` — weight gradient accumulators
#[allow(clippy::too_many_arguments)]
pub fn attn_res_backward(
    config: &AttnResConfig,
    weights: &AttnResWeights,
    block_values: &[Vec<f32>],
    prefix_sum: &[f32],
    d_out: &[f32],
    d_block_values: &mut [Vec<f32>],
    d_prefix_sum: &mut [f32],
    d_norm_weight: &mut [f32],
    d_proj_weight: &mut [f32],
) {
    let d = config.d();
    let eps = config.rms_eps;
    let num_entries = block_values.len() + 1; // blocks + prefix_sum

    // Reconstruct score_weight = norm_weight ⊙ proj_weight
    let score_weight: Vec<f32> = (0..d)
        .map(|i| weights.norm_weight[i] * weights.proj_weight[i])
        .collect();

    // Reconstruct scores + inv_rms for each entry
    let mut scores = vec![0.0f32; num_entries];
    let mut inv_rms_values = vec![0.0f32; num_entries];

    // Block entries
    for (i, residual) in block_values.iter().enumerate() {
        let sum_sq = simd_sum_sq(residual, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        inv_rms_values[i] = inv_rms;
        let raw_dot = simd_dot_f32(residual, &score_weight, d);
        scores[i] = inv_rms * raw_dot;
    }
    // Prefix sum entry
    {
        let sum_sq = simd_sum_sq(prefix_sum, d);
        let inv_rms = 1.0 / ((sum_sq / d as f32 + eps).sqrt());
        inv_rms_values[num_entries - 1] = inv_rms;
        let raw_dot = simd_dot_f32(prefix_sum, &score_weight, d);
        scores[num_entries - 1] = inv_rms * raw_dot;
    }

    // Softmax reconstruction (from scores)
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_scores = vec![0.0f32; num_entries];
    let mut sum_exp = 0.0f32;
    for i in 0..num_entries {
        exp_scores[i] = (scores[i] - max_score).exp();
        sum_exp += exp_scores[i];
    }
    let probs: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();

    // ── Backward through weighted sum: out = Σ_i probs[i] * v[i] ──
    // dL/dv[i] += probs[i] * dL/dout
    // dL/dprob[i] = dot(v[i], dL/dout)
    let mut d_probs = vec![0.0f32; num_entries];
    let d_out_d = &d_out[..d];
    for (i, residual) in block_values.iter().enumerate() {
        // `probs[i]` is invariant across `j` — load it once and pre-slice both
        // rows so the inner loop carries no bounds checks and no re-indexing of
        // the `Vec<Vec<f32>>`. Per-element expression `dst += p * g` is unchanged.
        let p = probs[i];
        let dbv = &mut d_block_values[i][..d];
        for j in 0..d {
            dbv[j] += p * d_out_d[j];
        }
        d_probs[i] = simd_dot_f32(residual, d_out, d);
    }
    // Prefix sum entry
    {
        let p = probs[num_entries - 1];
        let dps = &mut d_prefix_sum[..d];
        for j in 0..d {
            dps[j] += p * d_out_d[j];
        }
        d_probs[num_entries - 1] = simd_dot_f32(prefix_sum, d_out, d);
    }

    // ── Softmax backward: dL/dscore[i] = prob[i] * (d_prob[i] - Σ_j prob[j]*d_prob[j]) ──
    let dot_prob_dprob: f32 = probs
        .iter()
        .zip(d_probs.iter())
        .map(|(&p, &dp)| p * dp)
        .sum();
    let d_scores: Vec<f32> = probs
        .iter()
        .zip(d_probs.iter())
        .map(|(&p, &dp)| p * (dp - dot_prob_dprob))
        .collect();

    // ── Score backward: score[i] = inv_rms(v[i]) * dot(v[i], score_weight) ──
    // Let r_i = inv_rms(v[i]), s_i = dot(v[i], score_weight)
    // score[i] = r_i * s_i
    // dL/d(r_i) = dL/dscore[i] * s_i
    // dL/d(s_i) = dL/dscore[i] * r_i
    // dL/d(score_weight) += Σ_i dL/d(s_i) * r_i * v[i]
    // dL/d(v[i]) += dL/d(s_i) * r_i * score_weight   (from dot product)
    //            += dL/d(r_i) * d(r_i)/d(v[i])       (from inv_rms)

    let mut d_score_weight = vec![0.0f32; d];
    // `d as f32` was re-converted on every element of the inner loop below.
    let d_f = d as f32;
    let score_weight_d = &score_weight[..d];
    for i in 0..num_entries {
        let r_i = inv_rms_values[i];
        let s_i = scores[i] / r_i; // = dot(v[i], score_weight)
        let d_score_i = d_scores[i];
        let d_r_i = d_score_i * s_i;
        let d_s_i = d_score_i * r_i;

        // dL/d(score_weight) += d_s_i * v[i] (from dot(v[i], score_weight))
        // The v[i] is block_values[i] or prefix_sum
        let v_i: &[f32] = if i < block_values.len() {
            &block_values[i][..d]
        } else {
            &prefix_sum[..d]
        };
        // The `i < block_values.len()` discrimination is INVARIANT across `j`, so
        // pick the destination row once instead of branching (and re-walking the
        // `Vec<Vec<f32>>` indirection) on every element. `d_block_values` /
        // `d_prefix_sum` are distinct parameters from `block_values` / `prefix_sum`,
        // so this `&mut` row cannot alias `v_i`.
        let d_v_i: &mut [f32] = if i < block_values.len() {
            &mut d_block_values[i][..d]
        } else {
            &mut d_prefix_sum[..d]
        };
        let d_sw = &mut d_score_weight[..d];

        // GRADIENT SAFETY: the two former `j` loops are fused into one. Every
        // element's expression is byte-for-byte identical — same operand order,
        // same grouping `d_r_i * (-v_i[j] * r_i * r_i * r_i / d_f)`, no
        // reassociation and no FMA contraction introduced. `d_score_weight[j]`,
        // `d_v_i[j]` and `v_i[j]` are independent per `j`, so the two loops were
        // already order-independent w.r.t. each other; fusing them only halves
        // the loads of `v_i[j]` and the loop overhead.
        for j in 0..d {
            d_sw[j] += d_s_i * v_i[j];

            // dL/d(v[i]) from the dot product: d_s_i * score_weight[j]
            // (s_i = dot(v[i], score_weight), so d(s_i)/d(v[i][j]) = score_weight[j])
            let dot_grad = d_s_i * score_weight_d[j];

            // dL/d(v[i]) from inv_rms: d_r_i * d(r_i)/d(v[i][j])
            // r_i = 1/sqrt(mean(v²) + eps) = (mean(v²) + eps)^(-1/2)
            // dr/dv[j] = -1/2 * (mean(v²)+eps)^(-3/2) * (2*v[j]/n)
            //          = -v[j] * r_i³ / n
            let inv_rms_grad = d_r_i * (-v_i[j] * r_i * r_i * r_i / d_f);

            d_v_i[j] += dot_grad + inv_rms_grad;
        }
    }

    // ── score_weight = norm_weight ⊙ proj_weight ──
    // dL/d(norm_weight[j]) = dL/d(score_weight[j]) * proj_weight[j]
    // dL/d(proj_weight[j]) = dL/d(score_weight[j]) * norm_weight[j]
    // Pre-sliced to `d` so the four accesses per element carry no bounds checks.
    {
        let d_sw = &d_score_weight[..d];
        let pw = &weights.proj_weight[..d];
        let nw = &weights.norm_weight[..d];
        let d_nw = &mut d_norm_weight[..d];
        let d_pw = &mut d_proj_weight[..d];
        for j in 0..d {
            d_nw[j] += d_sw[j] * pw[j];
            d_pw[j] += d_sw[j] * nw[j];
        }
    }
}

// ─── Dense SiTU FFN backward ────────────────────────────────────────────────

/// Backward through dense SiTU FFN: `down_proj(SiTU(gate_proj(h), up_proj(h)))`.
///
/// Returns dL/d(h) and accumulates weight gradients.
///
/// `pub(crate)`: consumed by the delay-arch backward (Issue 482 / Plan 452,
/// `kimi_k3::delay`) — same crate, no public-API widening.
pub(crate) fn dense_situ_ffn_backward(
    expert: &SwiGluExpertWeights,
    saved: &DenseFfnSavedActivations,
    d_output: &[f32],
    grads: &mut DenseFfnGradients,
    beta: f32,
    linear_beta: Option<f32>,
) -> Vec<f32> {
    let d_in = d_output.len();
    let d_ffn = expert.gate_proj.len() / d_in;

    // out = down_proj · act  →  dL/d(act) = down_proj^T · dL/d(out)
    let mut d_act = vec![0.0f32; d_ffn];
    simd_transpose_matvec_into(&mut d_act, &expert.down_proj, d_output, d_in, d_ffn);
    // dL/d(down_proj) += outer(d_output, act)
    simd_outer_product_acc(&mut grads.down_proj, d_output, &saved.act_out, d_in, d_ffn);

    // SiTU backward: act = SiTU(gate, up)
    let mut d_gate = vec![0.0f32; d_ffn];
    let mut d_up = vec![0.0f32; d_ffn];
    situ_backward(
        &d_act,
        &saved.gate_inter,
        &saved.up_inter,
        &mut d_gate,
        &mut d_up,
        beta,
        linear_beta,
    );

    // gate = gate_proj · h  →  dL/d(h) += gate_proj^T · dL/d(gate)
    let mut d_h = vec![0.0f32; d_in];
    simd_transpose_matvec_into(&mut d_h, &expert.gate_proj, &d_gate, d_ffn, d_in);
    simd_outer_product_acc(&mut grads.gate_proj, &d_gate, &saved.h, d_ffn, d_in);

    // up = up_proj · h  →  dL/d(h) += up_proj^T · dL/d(up)
    let mut d_h_up = vec![0.0f32; d_in];
    simd_transpose_matvec_into(&mut d_h_up, &expert.up_proj, &d_up, d_ffn, d_in);
    simd_outer_product_acc(&mut grads.up_proj, &d_up, &saved.h, d_ffn, d_in);

    // Elementwise SIMD add — per-lane `a + b`, bit-identical to the scalar loop.
    katgpt_core::simd::simd_add_inplace(&mut d_h[..d_in], &d_h_up[..d_in]);

    d_h
}

/// SiTU activation backward.
fn situ_backward(
    d_act: &[f32],
    gate_inter: &[f32],
    up_inter: &[f32],
    d_gate: &mut [f32],
    d_up: &mut [f32],
    beta: f32,
    linear_beta: Option<f32>,
) {
    let inv_beta = 1.0 / beta;
    let n = d_act.len();

    // Pre-slice all five operands to `n` so the inner loops carry no bounds
    // checks. `[..n]` panics on a short operand exactly like the old indexed
    // loops did, so panic semantics are unchanged.
    let d_act = &d_act[..n];
    let gate_inter = &gate_inter[..n];
    let up_inter = &up_inter[..n];
    let d_gate = &mut d_gate[..n];
    let d_up = &mut d_up[..n];

    if let Some(lb) = linear_beta {
        let inv_lb = 1.0 / lb;
        for i in 0..n {
            let g = gate_inter[i];
            let u = up_inter[i];
            let da = d_act[i];
            let gs = 1.0 / (1.0 + (-g).exp()); // sigmoid(g)
            let gt = (g * inv_beta).tanh(); // tanh(g/beta)
            // `tanh(u/lb)` was evaluated TWICE per element (once inside `ut`, once
            // as `tanh_u`). Compute it once and derive `ut = lb * tanh_u` — the
            // exact same expression tree, so bit-identical, but it removes one
            // `tanh` (a genuine transcendental) per element on the gradient path.
            let tanh_u = (u * inv_lb).tanh();
            let ut = lb * tanh_u; // lb * tanh(u/lb)

            let d_act_dg = (1.0 - gt * gt) * gs * ut + beta * gt * gs * (1.0 - gs) * ut;
            let d_act_du = beta * gt * gs * (1.0 - tanh_u * tanh_u);

            d_gate[i] = da * d_act_dg;
            d_up[i] = da * d_act_du;
        }
    } else {
        for i in 0..n {
            let g = gate_inter[i];
            let u = up_inter[i];
            let da = d_act[i];
            let gs = 1.0 / (1.0 + (-g).exp());
            let gt = (g * inv_beta).tanh();

            let d_act_dg = (1.0 - gt * gt) * gs * u + beta * gt * gs * (1.0 - gs) * u;
            let d_act_du = beta * gt * gs;

            d_gate[i] = da * d_act_dg;
            d_up[i] = da * d_act_du;
        }
    }
}
