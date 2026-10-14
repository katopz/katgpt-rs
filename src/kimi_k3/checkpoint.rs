//! Gradient checkpointing for Kimi-K3 full-model backward (Plan 318 Phase C C7).
//!
//! Cuts activation memory from ~24 GB to ~3 GB by recomputing per-layer
//! activations during backward instead of storing them all from forward.
//!
//! # Strategy
//!
//! **Forward** saves only:
//! - Per token per layer: `prefix_sum_in` (the layer input hidden state, `[d]`)
//! - Per token per layer: block-state snapshots (`block_state_self`, `block_state_mlp`)
//!   taken before each attn-res mixing — small (`num_blocks × d` where
//!   `num_blocks ≈ layer_idx / block_size`)
//! - Per token: final-stage data (`prefix_sum_final`, `block_state_final`,
//!   `pre_final_norm`, `final_hidden`, `final_norm_inv_rms`, `logits`)
//! - MLA KV cache + KDA SSM state remain in `runtime` (populated by forward,
//!   read by backward — relatively small vs full saved activations)
//!
//! **Backward** processes layers in reverse. For each layer:
//! 1. **Recompute phase**: reset the layer's MLA/KDA state, then replay the
//!    layer forward for all tokens using saved `prefix_sum_in` + block-state
//!    snapshots → rebuilds per-token `LayerSavedActivations` (mla_saved /
//!    kda_saved / moe_saved / dense_saved + intermediates).
//! 2. **Backward phase**: run the layer backward using the recomputed
//!    activations (same code path as the non-checkpointed backward).
//! 3. **Discard**: the per-layer activations are dropped before the next layer.
//!
//! # Memory math (4B model, seq=1024, d=2560, 32 layers, block_size=3)
//!
//! - `prefix_sum_in`: 32 × 1024 × 2560 × 4 B = 336 MB
//! - block-state snapshots: ~5 entries × 2560 × 4 B × 2 (self+mlp) × 32 × 1024 ≈ 3.2 GB
//! - final-stage per token: ~1 GB (dominated by logits: 1024 × vocab × 4 B)
//! - MLA KV cache: in runtime, ~0.5 GB
//! - **Total: ~5 GB** (vs ~24 GB without checkpointing)
//!
//! The block-state snapshots dominate. A future optimization could save only
//! every-K-layer block states and replay the intermediate layers, but for the
//! 24 GB VRAM target the current design has sufficient headroom.
//!
//! # Compute overhead
//!
//! One extra forward pass per layer during backward = 1× additional forward
//! FLOPs total (33% overhead vs non-checkpointed forward+backward). This is
//! the standard gradient-checkpointing trade-off.
//!
//! # Scope
//!
//! CPU reference for the GPU training loop (C10). Gated behind `kimi_k3_backward`
//! alongside the non-checkpointed backward in `backward.rs`.

use katgpt_attn::gdn2::kda_backward::kda_forward_token_with_saved;
use katgpt_attn::gdn2::kda_backward::{KdaSavedActivations, kda_backward_sequence};
use katgpt_attn::mla_backward::{
    MlaSavedActivations, mla_backward_token, mla_forward_token_with_saved,
    rmsnorm_backward as mla_rmsnorm_backward,
};
use katgpt_core::simd::simd_sum_sq;
use katgpt_core::types::math::rmsnorm_with_gamma_eps;
use katgpt_kv::shard_kv::rope::RopeFreqs;
use katgpt_transformer::attn_res::{AttnResBlockState, apply_attn_res};
use katgpt_transformer::moe_backward::moe_forward_token_with_saved;

use super::backward::{
    KimiK3ModelGradients, LayerSavedActivations, attn_res_backward, dense_situ_ffn_forward_saved,
    ffn_backward,
};
use super::decoder_layer::{KimiAttentionState, KimiDecoderLayerConfig, KimiDecoderLayerWeights};
use super::loader::KimiK3ModelWeights;
use super::model::{KimiK3ModelConfig, KimiK3Runtime};

// ─── Checkpoint data structures ────────────────────────────────────────────

/// Per-layer checkpoint data (minimal — just enough to rebuild full
/// `LayerSavedActivations` during backward via layer-forward replay).
#[derive(Clone)]
pub struct LayerCheckpoint {
    /// Layer input hidden state (= `prefix_sum` at layer entry). `[d]`.
    pub prefix_sum_in: Vec<f32>,
    /// Block-state residuals snapshot BEFORE the self-attn-res mixing.
    /// `[num_entries][d]` (empty if no self-attn-res was applied).
    pub block_state_self: Vec<Vec<f32>>,
    /// Block-state residuals snapshot BEFORE the MLP attn-res mixing.
    /// `[num_entries][d]`.
    pub block_state_mlp: Vec<Vec<f32>>,
}

/// Final-stage (post-layers) saved data for one token.
#[derive(Clone)]
pub struct FinalStageSaved {
    /// `prefix_sum` after last layer (before output attn-res). `[d]`.
    pub prefix_sum_final: Vec<f32>,
    /// Block-state snapshot before output attn-res. `[num_blocks][d]`.
    pub block_state_final: Vec<Vec<f32>>,
    /// Whether output attn-res was applied.
    pub has_output_attn_res: bool,
    /// Hidden after output attn-res, before final norm (= final norm input). `[d]`.
    pub pre_final_norm: Vec<f32>,
    /// Final normed hidden (input to LM head). `[d]`.
    pub final_hidden: Vec<f32>,
    /// Final RMSNorm inv_rms.
    pub final_norm_inv_rms: f32,
    /// Logits. `[vocab_size]`.
    pub logits: Vec<f32>,
}

/// Checkpoint data for one token: per-layer inputs + final-stage data.
pub struct TokenCheckpoint {
    pub token_id: u32,
    pub pos: usize,
    pub layers: Vec<LayerCheckpoint>,
    pub final_stage: FinalStageSaved,
}

/// Checkpoint data for a sequence of tokens.
pub struct SequenceCheckpoint {
    pub tokens: Vec<TokenCheckpoint>,
}

// ─── Forward (checkpointed) ────────────────────────────────────────────────

/// Run the full Kimi-K3 forward for one token, saving only checkpoint data
/// (layer inputs + block-state snapshots + final-stage data).
///
/// Equivalent to `kimi_k3_forward_token_saved` but stores ~10× less data per
/// layer (no mla_saved / kda_saved / moe_saved / dense_saved intermediates).
/// The MLA KV cache + KDA SSM state are still populated in `runtime` (the
/// backward reads them during the recompute phase).
#[allow(clippy::too_many_arguments)]
pub fn kimi_k3_forward_token_ckpt(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    token_id: u32,
    pos: usize,
    ckpt: &mut TokenCheckpoint,
) {
    let d = config.hidden_size;

    runtime.block_state.clear();
    ckpt.layers.clear();
    ckpt.pos = pos;
    ckpt.token_id = token_id;

    // Embedding lookup
    let embed_start = (token_id as usize) * d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_start + d]);

    // Decoder layers — capture only layer input + block-state snapshots
    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        let block_state_self_snapshot: Vec<Vec<f32>> = if !runtime.block_state.is_empty() {
            runtime.block_state.residuals().to_vec()
        } else {
            Vec::new()
        };

        let prefix_sum_in = runtime.hidden.clone();

        // Run the non-saving layer forward (mutates runtime state).
        // We re-derive all intermediates during backward.
        let layer_cfg = config.layer_config(layer_idx);
        let layer_rt = &mut runtime.layers[layer_idx];
        super::decoder_layer::kimi_decoder_layer_forward(
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
        );

        let block_state_mlp_snapshot: Vec<Vec<f32>> = runtime.block_state.residuals().to_vec();

        ckpt.layers.push(LayerCheckpoint {
            prefix_sum_in,
            block_state_self: block_state_self_snapshot,
            block_state_mlp: block_state_mlp_snapshot,
        });
    }

    // Output attn-res
    let prefix_sum_final = runtime.hidden.clone();
    let block_state_final = runtime.block_state.residuals().to_vec();
    let has_output_attn_res = !runtime.block_state.is_empty();

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

    let pre_final_norm = runtime.hidden.clone();

    // Final RMSNorm
    let sum_sq = simd_sum_sq(&runtime.hidden, d);
    let inv_rms = 1.0 / ((sum_sq / d as f32 + config.rms_eps).sqrt());
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );

    let final_hidden = runtime.hidden.clone();

    // LM head
    katgpt_core::simd::simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );

    ckpt.final_stage = FinalStageSaved {
        prefix_sum_final,
        block_state_final,
        has_output_attn_res,
        pre_final_norm,
        final_hidden,
        final_norm_inv_rms: inv_rms,
        logits: runtime.logits.clone(),
    };
}

// ─── Backward (checkpointed) ───────────────────────────────────────────────

/// Full-model backward using checkpoint data.
///
/// Produces the same gradients as `kimi_k3_backward_sequence` but with ~10×
/// less activation memory by recomputing per-layer activations on-the-fly
/// during backward.
///
/// # Arguments
/// - `config` — model config
/// - `weights` — model weights
/// - `runtime` — runtime state (MLA caches + KDA states are RESET and
///   REPOPULATED during the recompute phase for each layer)
/// - `ckpt` — checkpoint data from `kimi_k3_forward_token_ckpt`
/// - `d_logits` — per-token gradient w.r.t. logits `[L][vocab_size]`
/// - `grads` — gradient accumulator (must be zeroed)
#[allow(clippy::too_many_arguments)]
pub fn kimi_k3_backward_sequence_ckpt(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    ckpt: &SequenceCheckpoint,
    d_logits: &[Vec<f32>],
    grads: &mut KimiK3ModelGradients,
) {
    let d = config.hidden_size;
    let v = config.vocab_size;
    let l = ckpt.tokens.len();
    debug_assert_eq!(d_logits.len(), l);
    if l == 0 {
        return;
    }

    // ── Step 1: Per-token output backward (LM head + final norm + output attn-res) ──
    // Produces d_prefix[t] = dL/d(prefix_sum at model exit) for each token.
    let mut d_prefix: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
    let mut block_grads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(l);

    for t in 0..l {
        let fs = &ckpt.tokens[t].final_stage;

        // LM head backward
        let mut d_fh = vec![0.0f32; d];
        katgpt_core::simd::simd_transpose_matvec_into(
            &mut d_fh,
            &weights.lm_head_weight,
            &d_logits[t],
            v,
            d,
        );
        katgpt_core::simd::simd_outer_product_acc(
            &mut grads.lm_head_weight,
            &d_logits[t],
            &fs.final_hidden,
            v,
            d,
        );

        // Final RMSNorm backward → dL/d(pre_final_norm)
        let d_pre_fn = mla_rmsnorm_backward(
            &d_fh,
            &fs.pre_final_norm,
            &weights.final_norm_weight,
            fs.final_norm_inv_rms,
            &mut grads.final_norm_weight,
            config.rms_eps,
        );

        // Output attn-res backward
        let num_blocks = fs.block_state_final.len();
        let mut bg: Vec<Vec<f32>> = (0..num_blocks).map(|_| vec![0.0f32; d]).collect();

        if fs.has_output_attn_res {
            attn_res_backward(
                &config.attn_res_config,
                &weights.output_attn_res,
                &fs.block_state_final,
                &fs.prefix_sum_final,
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
    // `0..l` inside the Step-2a per-token loop below (no `break`/`continue`), so
    // no slot can be read before it is written in the current layer's pass.
    // Reusing them turns 2 × l Vec allocations *per layer* into 2 × l total.
    let mut d_attn_out: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
    let mut d_ps_after_attn_all: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

    // ── Step 2: Layer-by-layer backward (reverse) ──
    for layer_idx in (0..config.num_layers).rev() {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_w = &weights.layers[layer_idx];
        let layer_grads = &mut grads.layers[layer_idx];
        let is_mla = config.is_mla_layer(layer_idx);
        let is_boundary = layer_idx.is_multiple_of(config.attn_res_config.block_size);

        // ── Step 2.0: Recompute per-layer saved activations ──
        // Reset this layer's attention state (MLA cache or KDA SSM state),
        // then replay the layer forward for all tokens using the saved
        // checkpoint data. This rebuilds the per-token LayerSavedActivations
        // that the backward needs (mla_saved / kda_saved / moe_saved +
        // intermediates).
        {
            let layer_rt = &mut runtime.layers[layer_idx];
            match &mut layer_rt.attn_state {
                KimiAttentionState::Mla(cache) => cache.reset(),
                KimiAttentionState::Kda(cache) => cache.reset(),
            }
        }

        let mut layer_saved_per_token: Vec<LayerSavedActivations> = Vec::with_capacity(l);
        for t in 0..l {
            let lc = &ckpt.tokens[t].layers[layer_idx];
            let mut saved = LayerSavedActivations {
                prefix_sum_in: lc.prefix_sum_in.clone(),
                has_self_attn_res: !lc.block_state_self.is_empty(),
                mixed_self: Vec::new(),
                self_inv_rms: 0.0,
                is_boundary,
                block_state_self: lc.block_state_self.clone(),
                attn_out: Vec::new(),
                prefix_sum_after_attn: Vec::new(),
                mixed_mlp: Vec::new(),
                mlp_inv_rms: 0.0,
                block_state_mlp: lc.block_state_mlp.clone(),
                ffn_out: Vec::new(),
                mla_saved: None,
                kda_saved: None,
                moe_saved: None,
                dense_saved: None,
            };

            // Reconstruct a temporary block_state from the saved snapshot.
            // The snapshot was taken BEFORE the self-attn-res mixing, so it
            // represents the block_state at layer entry.
            let mut block_state = AttnResBlockState::new(d);
            for entry in &lc.block_state_self {
                block_state.push(entry);
            }

            let mut prefix_sum = lc.prefix_sum_in.clone();
            let mut scratch_hidden = vec![0.0f32; d];

            let layer_rt = &mut runtime.layers[layer_idx];
            recompute_layer_forward_saved(
                layer_idx,
                &layer_cfg,
                layer_w,
                &mut layer_rt.attn_state,
                &mut layer_rt.attn_scratch,
                &mut layer_rt.ffn_scratch,
                &mut layer_rt.attn_res_self_scratch,
                &mut layer_rt.attn_res_mlp_scratch,
                &mut block_state,
                Some(&mut runtime.rope_freqs),
                &mut prefix_sum,
                &mut scratch_hidden,
                &mut saved,
            );

            layer_saved_per_token.push(saved);
        }

        // ── Step 2a: MLP/FFN block backward (per token) ──
        for t in 0..l {
            let saved = &layer_saved_per_token[t];

            // `ffn_backward` takes `d_output: &[f32]` and never mutates it, so
            // borrow `d_prefix[t]` directly instead of cloning a `[d]` Vec per
            // token per layer.
            let d_normed_mlp = ffn_backward(
                &layer_cfg.ffn,
                &layer_w.ffn,
                saved,
                &d_prefix[t],
                layer_grads,
            );

            let d_mixed_mlp = mla_rmsnorm_backward(
                &d_normed_mlp,
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
            // Accumulate straight into the pre-allocated `d_attn_out[t]` row
            // instead of cloning `d_prefix[t]` into a fresh Vec and then cloning
            // that again at the end of the iteration. Same starting contents
            // (`d_prefix[t]`), same in/out accumulation by `attn_res_backward`,
            // and `d_attn_out[t]` was going to receive exactly this value anyway
            // — so this drops 2 of the 3 `[d]` allocations per token per layer.
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

            // Pre-sliced row walk: `block_grads[t]` may be LONGER than
            // `d_mlp_blocks` (the `while` above only grows it to at least
            // `num_mlp_blocks`), and `d_mlp_blocks` has exactly `num_mlp_blocks`
            // rows — so `zip` visits exactly the same `0..num_mlp_blocks` rows the
            // indexed loop did. Elementwise `+=`, no reassociation.
            for (bg_row, d_row) in block_grads[t].iter_mut().zip(d_mlp_blocks.iter()) {
                let bg = &mut bg_row[..d];
                let dr = &d_row[..d];
                for j in 0..d {
                    bg[j] += dr[j];
                }
            }

            d_ps_after_attn_all[t].copy_from_slice(&d_attn_out[t]);
        }

        // ── Step 2b: Attention backward (MLA cross-token or KDA BPTT) ──
        let mut d_normed_self: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();

        if is_mla {
            let KimiDecoderLayerWeights {
                attention: super::decoder_layer::KimiAttentionWeights::Mla(mla_w),
                ..
            } = layer_w
            else {
                panic!("MLA layer but non-MLA weights");
            };

            let all_saved: Vec<MlaSavedActivations> = (0..l)
                .map(|t| layer_saved_per_token[t].mla_saved.clone().unwrap())
                .collect();

            let mut all_dh: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; d]).collect();
            let mla_grads = layer_grads.mla_grads.as_mut().unwrap();
            let mut rf = RopeFreqs::new_with_theta(
                config.mla_config.qk_rope_head_dim,
                config.mla_config.rope_theta,
            );

            // The MLA cache was repopulated during the recompute phase above.
            // Borrow it immutably for the backward.
            let layer_rt = &runtime.layers[layer_idx];
            let KimiAttentionState::Mla(cache) = &layer_rt.attn_state else {
                panic!("MLA layer but non-MLA cache");
            };

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
            let KimiDecoderLayerWeights {
                attention: super::decoder_layer::KimiAttentionWeights::Kda(kda_w),
                ..
            } = layer_w
            else {
                panic!("KDA layer but non-KDA weights");
            };
            let all_saved_kda: Vec<KdaSavedActivations> = (0..l)
                .map(|t| layer_saved_per_token[t].kda_saved.clone().unwrap())
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

        // ── Step 2c: Self-attn block backward (per token) ──
        for t in 0..l {
            let saved = &layer_saved_per_token[t];

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

            // Elementwise SIMD adds throughout this block: each lane is a single
            // independent `a + b`, so no reassociation — accumulated gradients are
            // bit-identical to the scalar indexed loops. `[..d]` keeps the old
            // panic-on-short-operand behaviour.
            if !is_boundary {
                katgpt_core::simd::simd_add_inplace(
                    &mut d_ps_in[..d],
                    &d_ps_after_attn_all[t][..d],
                );
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

                // `d_self_blocks` has exactly `num_self_blocks` rows and
                // `block_grads[t]` has at least that many, so `zip` covers exactly
                // the rows the indexed loop did.
                for (bg_row, d_row) in block_grads[t].iter_mut().zip(d_self_blocks.iter()) {
                    katgpt_core::simd::simd_add_inplace(&mut bg_row[..d], &d_row[..d]);
                }
                katgpt_core::simd::simd_add_inplace(&mut d_ps_in[..d], &d_ps_from_attnres[..d]);
            } else {
                katgpt_core::simd::simd_add_inplace(&mut d_ps_in[..d], &d_mixed_self[..d]);
            }

            d_prefix[t] = d_ps_in;
        }

        // Drop per-layer recomputed activations before processing the next layer.
        // (Implicit — `layer_saved_per_token` goes out of scope at the loop end.)
        drop(layer_saved_per_token);
    }

    // ── Step 3: Embedding backward ──
    for (t, ckpt_tok) in ckpt.tokens.iter().enumerate() {
        let token_id = ckpt_tok.token_id as usize;
        let base = token_id * d;
        // Elementwise SIMD add — same per-lane `a + b`, bit-identical.
        katgpt_core::simd::simd_add_inplace(
            &mut grads.embed_weight[base..base + d],
            &d_prefix[t][..d],
        );
    }
}

/// Recompute one layer's forward, capturing all saved activations.
///
/// This is a clone of `forward_layer_saved` from `backward.rs`, kept here to
/// avoid making that private function public. The two must stay in sync.
#[allow(clippy::too_many_arguments)]
fn recompute_layer_forward_saved(
    layer_idx: usize,
    config: &KimiDecoderLayerConfig,
    weights: &KimiDecoderLayerWeights,
    attn_state: &mut KimiAttentionState,
    attn_scratch: &mut super::decoder_layer::KimiAttentionScratch,
    ffn_scratch: &mut super::decoder_layer::KimiFfnScratch,
    attn_res_self_scratch: &mut katgpt_transformer::attn_res::AttnResScratch,
    attn_res_mlp_scratch: &mut katgpt_transformer::attn_res::AttnResScratch,
    block_state: &mut AttnResBlockState,
    rope_freqs: Option<&mut RopeFreqs>,
    prefix_sum: &mut [f32],
    scratch_hidden: &mut [f32],
    saved: &mut LayerSavedActivations,
) {
    use super::decoder_layer::{
        KimiAttentionConfig, KimiAttentionWeights, KimiFfnConfig, KimiFfnWeights,
    };

    let d = config.attn_res.d();
    let eps = config.rms_eps;
    let block_size = config.attn_res.block_size;
    let is_boundary = layer_idx.is_multiple_of(block_size);
    saved.is_boundary = is_boundary;

    // Step 1: self-attn-res mixing
    if !block_state.is_empty() {
        saved.has_self_attn_res = true;
        // block_state_self already set from checkpoint snapshot — don't overwrite
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
        for x in prefix_sum.iter_mut() {
            *x = 0.0;
        }
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
            let super::decoder_layer::KimiAttentionScratch::Mla(scratch) = attn_scratch else {
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
            let super::decoder_layer::KimiAttentionScratch::Kda(scratch) = attn_scratch else {
                panic!("KDA scratch mismatch")
            };
            let (out, s) = kda_forward_token_with_saved(cfg, w, cache, scratch, scratch_hidden);
            saved.kda_saved = Some(s);
            out
        }
        _ => panic!("attention mismatch"),
    };
    saved.attn_out = attn_out.clone();

    // Step 4: prefix_sum += attn_out
    for i in 0..d {
        prefix_sum[i] += attn_out[i];
    }
    saved.prefix_sum_after_attn = prefix_sum.to_vec();

    // Step 5: MLP attn-res
    // block_state_mlp already set from checkpoint snapshot
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
    saved.ffn_out = ffn_out.clone();

    // Step 7: prefix_sum += ffn_out
    for i in 0..d {
        prefix_sum[i] += ffn_out[i];
    }
}
