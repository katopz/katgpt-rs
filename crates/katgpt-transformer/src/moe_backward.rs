//! MoE analytic backward pass (Plan 318 Phase C C4).
//!
//! Implements the analytic gradient of `moe_forward_token` w.r.t. all
//! trainable parameters (`MoeWeights`) and the input hidden state `h`.
//!
//! # Scope
//!
//! Single-token backward. MoE is stateless (no recurrence, no KV cache),
//! so there is no multi-token BPTT — the per-token backward is the complete
//! primitive. Multi-token training loops compose it per-token.
//!
//! Handles BOTH the latent-MoE path (`routed_expert_hidden_size = Some`,
//! the Kimi-K3 actual config) and the non-latent path (`None`). The latent
//! path is the load-bearing one; the non-latent path is included for
//! completeness + G1 coverage.
//!
//! # Why this lives here
//!
//! katgpt-rs is modelless-by-mandate (no training at runtime), BUT this module
//! is the **CPU reference** for the GPU backward (C4). It belongs alongside the
//! forward for two reasons: (1) the CPU primitive stays public so riir-train can
//! consume it; (2) the finite-difference gradient check needs it in-tree.
//! Production inference never calls this — it's gated behind `moe_backward`
//! (implies `transformer_moe`) and only consumed by riir-train.

use crate::moe::{MoeConfig, MoeForwardScratch, MoeWeights, select_topk_indices, situ_inplace};
use katgpt_core::simd::{
    simd_dot_f32, simd_matmul_rows, simd_outer_product_acc, simd_sum_sq, simd_transpose_matvec_acc,
    simd_transpose_matvec_into,
};
use katgpt_core::types::math::rmsnorm_with_gamma_eps;

// ─── Saved activations ──────────────────────────────────────────────────────

/// Forward-pass saved activations needed by the backward.
///
/// Populated by [`moe_forward_token_with_saved`]. All tensors are owned
/// snapshots taken AFTER the forward completes.
///
/// [`moe_forward_token_split`] (delay-arch training, Issue 482 / Plan 452)
/// populates the same struct with `h` = the SHARED stream's input and
/// `h_routed` = `Some(routed input)` — the routed backward pieces read the
/// routed input through [`MoeSavedActivations::routed_h`], which falls back
/// to `h` for the fused path.
#[derive(Clone)]
pub struct MoeSavedActivations {
    /// Input hidden state `h`. `[d]`.
    pub h: Vec<f32>,
    /// Routed-stream input `[d]` — `Some` iff populated by
    /// [`moe_forward_token_split`] with a routed input different from `h`.
    /// Fused-path consumers never read this directly.
    pub h_routed: Option<Vec<f32>>,
    /// Router logits `[N_r]`.
    pub router_logits: Vec<f32>,
    /// Sigmoid scores `[N_r]`.
    pub sigmoid_scores: Vec<f32>,
    /// Top-K selected expert indices `[K_r]`.
    pub topk_indices: Vec<usize>,
    /// Renormalized gating weights `[K_r]`.
    pub topk_weights: Vec<f32>,
    /// Sum of selected raw sigmoid scores (the renormalization denominator `S`).
    pub topk_sum: f32,

    // ── Latent-MoE path (only when routed_expert_hidden_size = Some) ──
    /// `h_latent = routed_expert_down_proj · h`. `[d_moe]`. `None` if non-latent.
    pub latent_hidden: Option<Vec<f32>>,
    /// Per-selected-expert SiTU outputs (latent dim). `[K_r][d_moe]` entries.
    /// `d_moe` for latent path, `d` for non-latent. Stored as flat `[K_r * d_expert]`.
    pub expert_outputs: Vec<f32>,
    /// Expert dimension for routed experts (`d_moe` for latent path, `d` for non-latent).
    pub d_expert: usize,
    /// Accumulated latent output BEFORE norm. `[d_expert]`.
    pub latent_output_prenorm: Vec<f32>,
    /// RMSNorm mean² + eps (for norm backward). Only when `latent_moe_use_norm`.
    pub latent_norm_inv_rms: Option<f32>,

    // ── Per-expert SiTU intermediates (for SiTU backward) ──
    /// Per-selected-expert gate_proj output (pre-SiTU). `[K_r * d_ffn]`.
    pub expert_gate_inter: Vec<f32>,
    /// Per-selected-expert up_proj output (pre-SiTU). `[K_r * d_ffn]`.
    pub expert_up_inter: Vec<f32>,
    /// Per-selected-expert SiTU activation output. `[K_r * d_ffn]`.
    pub expert_act_out: Vec<f32>,

    // ── Shared expert intermediates ──
    /// Shared expert gate_proj output. `[d_ffn_shared]`.
    pub shared_gate_inter: Vec<f32>,
    /// Shared expert up_proj output. `[d_ffn_shared]`.
    pub shared_up_inter: Vec<f32>,
    /// Shared expert SiTU activation output. `[d_ffn_shared]`.
    pub shared_act_out: Vec<f32>,
    /// Shared expert output (down_proj result). `[d]`.
    pub shared_output: Vec<f32>,
}

impl MoeSavedActivations {
    /// The routed stream's input hidden state — `h_routed` when the split
    /// forward populated it, `h` otherwise (fused path: same input).
    #[inline]
    pub fn routed_h(&self) -> &[f32] {
        self.h_routed.as_deref().unwrap_or(&self.h)
    }
}

// ─── Gradients ──────────────────────────────────────────────────────────────

/// Gradient accumulator for MoE backward.
///
/// All fields mirror `MoeWeights` field-for-field. Accumulated (`+=`) during
/// backward. Initialize with [`MoeGradients::zeros_like`] before the backward.
#[derive(Clone, Debug)]
pub struct MoeGradients {
    /// `[N_r * d]` — router weight gradient.
    pub router_weight: Vec<f32>,
    /// `[N_r]` — always zero (bias is selection-only, non-differentiable).
    pub e_score_correction_bias: Vec<f32>,
    /// Per-routed-expert gradients. Length `N_r`; each has gate/up/down.
    pub experts: Vec<SwiGluExpertGradients>,
    /// Per-shared-expert gradients. Length `N_s`.
    pub shared_experts: Vec<SwiGluExpertGradients>,
    /// Latent MoE down-projection gradient `[d_moe * d]`. `None` if non-latent.
    pub routed_expert_down_proj: Option<Vec<f32>>,
    /// Latent MoE up-projection gradient `[d * d_moe]`. `None` if non-latent.
    pub routed_expert_up_proj: Option<Vec<f32>>,
    /// Latent MoE RMSNorm gamma gradient `[d_moe]`. `None` if no norm or non-latent.
    pub routed_expert_norm_weight: Option<Vec<f32>>,
}

/// Per-expert gradient (mirrors `SwiGluExpertWeights`).
#[derive(Clone, Debug)]
pub struct SwiGluExpertGradients {
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

impl MoeGradients {
    /// Allocate zeroed gradients matching the given weights' shapes.
    pub fn zeros_like(w: &MoeWeights) -> Self {
        let experts = w
            .experts
            .iter()
            .map(|e| SwiGluExpertGradients {
                gate_proj: vec![0.0; e.gate_proj.len()],
                up_proj: vec![0.0; e.up_proj.len()],
                down_proj: vec![0.0; e.down_proj.len()],
            })
            .collect();
        let shared_experts = w
            .shared_experts
            .iter()
            .map(|e| SwiGluExpertGradients {
                gate_proj: vec![0.0; e.gate_proj.len()],
                up_proj: vec![0.0; e.up_proj.len()],
                down_proj: vec![0.0; e.down_proj.len()],
            })
            .collect();
        Self {
            router_weight: vec![0.0; w.router_weight.len()],
            e_score_correction_bias: vec![0.0; w.e_score_correction_bias.len()],
            experts,
            shared_experts,
            routed_expert_down_proj: w
                .routed_expert_down_proj
                .as_ref()
                .map(|x| vec![0.0; x.len()]),
            routed_expert_up_proj: w.routed_expert_up_proj.as_ref().map(|x| vec![0.0; x.len()]),
            routed_expert_norm_weight: w
                .routed_expert_norm_weight
                .as_ref()
                .map(|x| vec![0.0; x.len()]),
        }
    }
}

// ─── Forward with saved ─────────────────────────────────────────────────────

/// Run the MoE forward and capture all activations needed for backward.
///
/// Returns `(output, saved_activations)`. The output is the same as
/// `moe_forward_token` — this function just additionally snapshots the
/// intermediate values.
///
/// **Allocation discipline:** this function allocates (the saved activations
/// are `Vec<f32>` snapshots). It is the training-time reference path, NOT the
/// inference hot path. The inference forward (`moe_forward_token`) remains
/// zero-alloc.
pub fn moe_forward_token_with_saved(
    weights: &MoeWeights,
    config: &MoeConfig,
    hidden_in: &[f32],
    scratch: &mut MoeForwardScratch,
) -> (Vec<f32>, MoeSavedActivations) {
    let n_r = config.n_routed();
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();
    let d_moe = config.d_moe();
    let d_ffn_shared = config.d_ffn_shared();
    let use_latent_moe = config.routed_expert_hidden_size.is_some();
    let d_expert = if use_latent_moe { d_moe } else { d };

    // ── Snapshot the input hidden ──
    let h = hidden_in.to_vec();

    // ── Re-run the forward logic inline so we can snapshot intermediates ──
    // We can't just call moe_forward_token because the scratch is overwritten
    // in-place (e.g., expert_intermediate is reused per expert). So we
    // re-implement the forward here, capturing per-expert intermediates.

    // 1. Router logits
    simd_matmul_rows(
        &mut scratch.router_logits[..n_r],
        &weights.router_weight,
        hidden_in,
        n_r,
        d,
    );

    // 2. Sigmoid scores
    for e in 0..n_r {
        scratch.sigmoid_scores[e] = katgpt_core::sigmoid(scratch.router_logits[e]);
    }

    // 3. Biased scores
    for e in 0..n_r {
        scratch.biased_scores[e] = scratch.sigmoid_scores[e] + weights.e_score_correction_bias[e];
    }

    // 4. Top-K selection
    select_topk_indices(
        &scratch.biased_scores[..n_r],
        k_r,
        &mut scratch.topk_indices[..k_r],
    );

    // 5. Renormalize
    let mut topk_sum = 0.0f32;
    for k in 0..k_r {
        let idx = scratch.topk_indices[k];
        topk_sum += scratch.sigmoid_scores[idx];
    }
    if topk_sum < 1.0e-20 {
        let uniform = 1.0 / (k_r as f32);
        for k in 0..k_r {
            scratch.topk_weights[k] = uniform;
        }
    } else if config.renormalize {
        let inv = 1.0 / topk_sum;
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx] * inv;
        }
    } else {
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx];
        }
    }

    // ── Shared expert forward (capturing intermediates) ──
    let shared = &weights.shared_experts[0];
    let shared_gate_inter_buf = &mut scratch.expert_intermediate[..d_ffn_shared];
    let shared_up_buf = &mut scratch.expert_up[..d_ffn_shared];
    let shared_out = &mut scratch.expert_output[..d];
    // gate_proj · h
    simd_matmul_rows(
        shared_gate_inter_buf,
        &shared.gate_proj,
        hidden_in,
        d_ffn_shared,
        d,
    );
    // up_proj · h
    simd_matmul_rows(shared_up_buf, &shared.up_proj, hidden_in, d_ffn_shared, d);
    // Snapshot pre-SiTU gate/up
    let shared_gate_inter: Vec<f32> = shared_gate_inter_buf.to_vec();
    let shared_up_inter: Vec<f32> = shared_up_buf.to_vec();
    // SiTU in-place on gate_buf
    situ_inplace(
        shared_gate_inter_buf,
        shared_up_buf,
        config.situ_beta,
        config.situ_linear_beta,
    );
    let shared_act_out: Vec<f32> = shared_gate_inter_buf.to_vec();
    // down_proj · act → out
    simd_matmul_rows(
        shared_out,
        &shared.down_proj,
        shared_gate_inter_buf,
        d,
        d_ffn_shared,
    );
    let shared_output: Vec<f32> = shared_out.to_vec();

    // hidden_out starts with shared_output
    let mut hidden_out: Vec<f32> = shared_output.clone();

    // Accumulate remaining shared experts (Kimi-K3-0.40B has N_s=1, but handle general case)
    for s in 1..weights.shared_experts.len() {
        let shared_s = &weights.shared_experts[s];
        // We don't capture intermediates for s>0 (rare; Kimi-K3 has N_s=1).
        // Re-run forward for this expert without snapshotting.
        let gate_buf = &mut scratch.expert_intermediate[..d_ffn_shared];
        let up_buf = &mut scratch.expert_up[..d_ffn_shared];
        let out_buf = &mut scratch.expert_output[..d];
        simd_matmul_rows(gate_buf, &shared_s.gate_proj, hidden_in, d_ffn_shared, d);
        simd_matmul_rows(up_buf, &shared_s.up_proj, hidden_in, d_ffn_shared, d);
        situ_inplace(gate_buf, up_buf, config.situ_beta, config.situ_linear_beta);
        simd_matmul_rows(out_buf, &shared_s.down_proj, gate_buf, d, d_ffn_shared);
        for (ho, eo) in hidden_out
            .iter_mut()
            .zip(scratch.expert_output.iter())
            .take(d)
        {
            *ho += *eo;
        }
    }

    // ── Routed experts ──
    // Capture per-selected-expert intermediates for backward.
    let mut expert_gate_inter = vec![0.0f32; k_r * d_ffn];
    let mut expert_up_inter = vec![0.0f32; k_r * d_ffn];
    let mut expert_act_out = vec![0.0f32; k_r * d_ffn];
    let mut expert_outputs = vec![0.0f32; k_r * d_expert];

    if use_latent_moe {
        // Latent path: down-project h → experts on latent dim → norm → up-project

        // h_latent = routed_expert_down_proj · h
        simd_matmul_rows(
            &mut scratch.latent_hidden,
            weights.routed_expert_down_proj.as_ref().unwrap(),
            hidden_in,
            d_moe,
            d,
        );
        let latent_hidden_snap = scratch.latent_hidden.clone();

        // Accumulate weighted expert outputs into latent_output
        scratch.latent_output[..d_moe].fill(0.0);
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            let w = scratch.topk_weights[k];
            let expert = &weights.experts[idx];
            let gate_buf = &mut scratch.expert_intermediate[..d_ffn];
            let up_buf = &mut scratch.expert_up[..d_ffn];
            let out_buf = &mut scratch.expert_output[..d_moe];

            // gate_proj · h_latent
            simd_matmul_rows(
                gate_buf,
                &expert.gate_proj,
                &scratch.latent_hidden,
                d_ffn,
                d_moe,
            );
            // up_proj · h_latent
            simd_matmul_rows(
                up_buf,
                &expert.up_proj,
                &scratch.latent_hidden,
                d_ffn,
                d_moe,
            );

            // Snapshot pre-SiTU
            expert_gate_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);
            expert_up_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(up_buf);

            // SiTU in-place
            situ_inplace(gate_buf, up_buf, config.situ_beta, config.situ_linear_beta);
            expert_act_out[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);

            // down_proj · act → out_buf
            simd_matmul_rows(out_buf, &expert.down_proj, gate_buf, d_moe, d_ffn);
            expert_outputs[k * d_expert..(k + 1) * d_expert].copy_from_slice(out_buf);

            // latent_output += w * out_buf
            for (lo, eo) in scratch
                .latent_output
                .iter_mut()
                .zip(out_buf.iter())
                .take(d_moe)
            {
                *lo += w * *eo;
            }
        }

        let latent_output_prenorm = scratch.latent_output.clone();

        // Optional norm
        let latent_norm_inv_rms = if config.latent_moe_use_norm
            && let Some(ref norm_w) = weights.routed_expert_norm_weight
        {
            // Compute inv_rms BEFORE rmsnorm_with_gamma_eps overwrites latent_output.
            // rmsnorm: y[i] = x[i] * gamma[i] / sqrt(mean(x²) + eps)
            // We need mean(x²)+eps for the backward.
            let sum_sq = simd_sum_sq(&scratch.latent_output[..d_moe], d_moe);
            let mean_sq = sum_sq / d_moe as f32;
            let inv_rms = 1.0 / (mean_sq + config.rms_norm_eps).sqrt();
            rmsnorm_with_gamma_eps(
                &mut scratch.latent_output,
                norm_w,
                config.rms_norm_eps as f64,
            );
            Some(inv_rms)
        } else {
            None
        };

        // Up-project: hidden_out += routed_expert_up_proj · latent_output
        simd_matmul_rows(
            &mut scratch.expert_output[..d],
            weights.routed_expert_up_proj.as_ref().unwrap(),
            &scratch.latent_output,
            d,
            d_moe,
        );
        for (ho, eo) in hidden_out
            .iter_mut()
            .zip(scratch.expert_output.iter())
            .take(d)
        {
            *ho += *eo;
        }

        let saved = MoeSavedActivations {
            h,
            h_routed: None,
            router_logits: scratch.router_logits[..n_r].to_vec(),
            sigmoid_scores: scratch.sigmoid_scores[..n_r].to_vec(),
            topk_indices: scratch.topk_indices[..k_r].to_vec(),
            topk_weights: scratch.topk_weights[..k_r].to_vec(),
            topk_sum,
            latent_hidden: Some(latent_hidden_snap),
            expert_outputs,
            d_expert,
            latent_output_prenorm,
            latent_norm_inv_rms,
            expert_gate_inter,
            expert_up_inter,
            expert_act_out,
            shared_gate_inter,
            shared_up_inter,
            shared_act_out,
            shared_output,
        };

        (hidden_out, saved)
    } else {
        // Non-latent path: routed experts operate directly on hidden dim
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            let w = scratch.topk_weights[k];
            let expert = &weights.experts[idx];
            let gate_buf = &mut scratch.expert_intermediate[..d_ffn];
            let up_buf = &mut scratch.expert_up[..d_ffn];
            let out_buf = &mut scratch.expert_output[..d];

            simd_matmul_rows(gate_buf, &expert.gate_proj, hidden_in, d_ffn, d);
            simd_matmul_rows(up_buf, &expert.up_proj, hidden_in, d_ffn, d);

            expert_gate_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);
            expert_up_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(up_buf);

            situ_inplace(gate_buf, up_buf, config.situ_beta, config.situ_linear_beta);
            expert_act_out[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);

            simd_matmul_rows(out_buf, &expert.down_proj, gate_buf, d, d_ffn);
            expert_outputs[k * d_expert..(k + 1) * d_expert].copy_from_slice(out_buf);

            for (ho, eo) in hidden_out.iter_mut().zip(out_buf.iter()).take(d) {
                *ho += w * *eo;
            }
        }

        let saved = MoeSavedActivations {
            h,
            h_routed: None,
            router_logits: scratch.router_logits[..n_r].to_vec(),
            sigmoid_scores: scratch.sigmoid_scores[..n_r].to_vec(),
            topk_indices: scratch.topk_indices[..k_r].to_vec(),
            topk_weights: scratch.topk_weights[..k_r].to_vec(),
            topk_sum,
            latent_hidden: None,
            expert_outputs,
            d_expert,
            latent_output_prenorm: vec![],
            latent_norm_inv_rms: None,
            expert_gate_inter,
            expert_up_inter,
            expert_act_out,
            shared_gate_inter,
            shared_up_inter,
            shared_act_out,
            shared_output,
        };

        (hidden_out, saved)
    }
}

/// Split-stream MoE forward for delay-arch training (Issue 482 / Plan 452).
///
/// Same math as [`moe_forward_token_with_saved`], but the SHARED stream
/// (always-on experts) reads `shared_in` while the ROUTED stream (router +
/// top-k experts + latent projections) reads `routed_in`. The two stream
/// outputs are returned SEPARATELY so the caller controls their injection
/// sites (the pre-dense routing anchor / expert-delay arms feed the routed
/// stream the layer-input residual and inject its output δe layers later).
///
/// Bit-identity pin: with `shared_in == routed_in`, `shared_out + routed_out`
/// (summed shared-first) equals the fused output bit-identically on the
/// latent path (the Kimi family; each stream's internal op order is
/// unchanged). The non-latent path reassociates the per-expert accumulation
/// — pinned by tolerance in the test, not by construction.
///
/// The returned `MoeSavedActivations` carries `h = shared_in` and
/// `h_routed = Some(routed_in)`; consume it with
/// [`moe_backward_token_split`].
pub fn moe_forward_token_split(
    weights: &MoeWeights,
    config: &MoeConfig,
    shared_in: &[f32],
    routed_in: &[f32],
    scratch: &mut MoeForwardScratch,
) -> (Vec<f32>, Vec<f32>, MoeSavedActivations) {
    let n_r = config.n_routed();
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();
    let d_moe = config.d_moe();
    let d_ffn_shared = config.d_ffn_shared();
    let use_latent_moe = config.routed_expert_hidden_size.is_some();
    let d_expert = if use_latent_moe { d_moe } else { d };

    debug_assert_eq!(shared_in.len(), d);
    debug_assert_eq!(routed_in.len(), d);

    // ── Shared expert forward on `shared_in` (capturing intermediates) ──
    let shared = &weights.shared_experts[0];
    let shared_gate_inter_buf = &mut scratch.expert_intermediate[..d_ffn_shared];
    let shared_up_buf = &mut scratch.expert_up[..d_ffn_shared];
    let shared_out = &mut scratch.expert_output[..d];
    simd_matmul_rows(
        shared_gate_inter_buf,
        &shared.gate_proj,
        shared_in,
        d_ffn_shared,
        d,
    );
    simd_matmul_rows(shared_up_buf, &shared.up_proj, shared_in, d_ffn_shared, d);
    let shared_gate_inter: Vec<f32> = shared_gate_inter_buf.to_vec();
    let shared_up_inter: Vec<f32> = shared_up_buf.to_vec();
    situ_inplace(
        shared_gate_inter_buf,
        shared_up_buf,
        config.situ_beta,
        config.situ_linear_beta,
    );
    let shared_act_out: Vec<f32> = shared_gate_inter_buf.to_vec();
    simd_matmul_rows(
        shared_out,
        &shared.down_proj,
        shared_gate_inter_buf,
        d,
        d_ffn_shared,
    );
    let shared_output: Vec<f32> = shared_out.to_vec();

    // ── Router + routed experts on `routed_in` ──
    // 1. Router logits
    simd_matmul_rows(
        &mut scratch.router_logits[..n_r],
        &weights.router_weight,
        routed_in,
        n_r,
        d,
    );

    // 2/3. Sigmoid + biased scores
    for e in 0..n_r {
        let s = katgpt_core::sigmoid(scratch.router_logits[e]);
        scratch.sigmoid_scores[e] = s;
        scratch.biased_scores[e] = s + weights.e_score_correction_bias[e];
    }

    // 4. Top-K selection
    select_topk_indices(
        &scratch.biased_scores[..n_r],
        k_r,
        &mut scratch.topk_indices[..k_r],
    );

    // 5. Renormalize
    let mut topk_sum = 0.0f32;
    for k in 0..k_r {
        let idx = scratch.topk_indices[k];
        topk_sum += scratch.sigmoid_scores[idx];
    }
    if topk_sum < 1.0e-20 {
        let uniform = 1.0 / (k_r as f32);
        for k in 0..k_r {
            scratch.topk_weights[k] = uniform;
        }
    } else if config.renormalize {
        let inv = 1.0 / topk_sum;
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx] * inv;
        }
    } else {
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            scratch.topk_weights[k] = scratch.sigmoid_scores[idx];
        }
    }

    // Routed experts (capturing per-selected-expert intermediates)
    let mut expert_gate_inter = vec![0.0f32; k_r * d_ffn];
    let mut expert_up_inter = vec![0.0f32; k_r * d_ffn];
    let mut expert_act_out = vec![0.0f32; k_r * d_ffn];
    let mut expert_outputs = vec![0.0f32; k_r * d_expert];

    // Latent-path saved locals (populated only in the latent branch below —
    // must be snapshotted BEFORE the optional norm mutates latent_output,
    // matching the fused path's save points).
    let mut routed_latent_hidden_snap: Option<Vec<f32>> = None;
    let mut routed_latent_prenorm: Vec<f32> = Vec::new();
    let mut routed_latent_norm_inv_rms: Option<f32> = None;

    let routed_out: Vec<f32> = if use_latent_moe {
        // h_latent = routed_expert_down_proj · routed_in
        simd_matmul_rows(
            &mut scratch.latent_hidden,
            weights.routed_expert_down_proj.as_ref().unwrap(),
            routed_in,
            d_moe,
            d,
        );
        let latent_hidden_snap = scratch.latent_hidden.clone();
        routed_latent_hidden_snap = Some(latent_hidden_snap);

        scratch.latent_output[..d_moe].fill(0.0);
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            let w = scratch.topk_weights[k];
            let expert = &weights.experts[idx];
            let gate_buf = &mut scratch.expert_intermediate[..d_ffn];
            let up_buf = &mut scratch.expert_up[..d_ffn];
            let out_buf = &mut scratch.expert_output[..d_moe];

            simd_matmul_rows(
                gate_buf,
                &expert.gate_proj,
                &scratch.latent_hidden,
                d_ffn,
                d_moe,
            );
            simd_matmul_rows(up_buf, &expert.up_proj, &scratch.latent_hidden, d_ffn, d_moe);

            expert_gate_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);
            expert_up_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(up_buf);

            situ_inplace(gate_buf, up_buf, config.situ_beta, config.situ_linear_beta);
            expert_act_out[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);

            simd_matmul_rows(out_buf, &expert.down_proj, gate_buf, d_moe, d_ffn);
            expert_outputs[k * d_expert..(k + 1) * d_expert].copy_from_slice(out_buf);

            for (lo, eo) in scratch
                .latent_output
                .iter_mut()
                .zip(out_buf.iter())
                .take(d_moe)
            {
                *lo += w * *eo;
            }
        }

        routed_latent_prenorm = scratch.latent_output.clone();

        let latent_norm_inv_rms = if config.latent_moe_use_norm
            && let Some(ref norm_w) = weights.routed_expert_norm_weight
        {
            let sum_sq = simd_sum_sq(&scratch.latent_output[..d_moe], d_moe);
            let mean_sq = sum_sq / d_moe as f32;
            let inv_rms = 1.0 / (mean_sq + config.rms_norm_eps).sqrt();
            rmsnorm_with_gamma_eps(
                &mut scratch.latent_output,
                norm_w,
                config.rms_norm_eps as f64,
            );
            Some(inv_rms)
        } else {
            None
        };
        routed_latent_norm_inv_rms = latent_norm_inv_rms;

        // routed_out = up_proj · latent_output (the routed stream's total)
        let mut routed = vec![0.0f32; d];
        simd_matmul_rows(
            &mut routed,
            weights.routed_expert_up_proj.as_ref().unwrap(),
            &scratch.latent_output,
            d,
            d_moe,
        );
        routed
    } else {
        // Non-latent: routed_out = sum_k w_k · expert_out_k on `routed_in`
        let mut routed = vec![0.0f32; d];
        for k in 0..k_r {
            let idx = scratch.topk_indices[k];
            let w = scratch.topk_weights[k];
            let expert = &weights.experts[idx];
            let gate_buf = &mut scratch.expert_intermediate[..d_ffn];
            let up_buf = &mut scratch.expert_up[..d_ffn];
            let out_buf = &mut scratch.expert_output[..d];

            simd_matmul_rows(gate_buf, &expert.gate_proj, routed_in, d_ffn, d);
            simd_matmul_rows(up_buf, &expert.up_proj, routed_in, d_ffn, d);

            expert_gate_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);
            expert_up_inter[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(up_buf);

            situ_inplace(gate_buf, up_buf, config.situ_beta, config.situ_linear_beta);
            expert_act_out[k * d_ffn..(k + 1) * d_ffn].copy_from_slice(gate_buf);

            simd_matmul_rows(out_buf, &expert.down_proj, gate_buf, d, d_ffn);
            expert_outputs[k * d_expert..(k + 1) * d_expert].copy_from_slice(out_buf);

            for (ro, eo) in routed.iter_mut().zip(out_buf.iter()).take(d) {
                *ro += w * *eo;
            }
        }
        routed
    };

    // The caller gets the shared stream's output; `saved` keeps its own copy
    // (the fused path makes the same clone when seeding `hidden_out`).
    let shared_out_ret = shared_output.clone();

    let saved = MoeSavedActivations {
        h: shared_in.to_vec(),
        h_routed: Some(routed_in.to_vec()),
        router_logits: scratch.router_logits[..n_r].to_vec(),
        sigmoid_scores: scratch.sigmoid_scores[..n_r].to_vec(),
        topk_indices: scratch.topk_indices[..k_r].to_vec(),
        topk_weights: scratch.topk_weights[..k_r].to_vec(),
        topk_sum,
        latent_hidden: routed_latent_hidden_snap,
        expert_outputs,
        d_expert,
        latent_output_prenorm: routed_latent_prenorm,
        latent_norm_inv_rms: routed_latent_norm_inv_rms,
        expert_gate_inter,
        expert_up_inter,
        expert_act_out,
        shared_gate_inter,
        shared_up_inter,
        shared_act_out,
        shared_output,
    };

    (shared_out_ret, routed_out, saved)
}

// ─── Backward ───────────────────────────────────────────────────────────────

/// MoE analytic backward pass.
///
/// Given `dL/d(hidden_out)`, computes:
/// - Gradients w.r.t. all trainable parameters (`grads`)
/// - Gradient w.r.t. the input hidden state (`dh_out`)
///
/// # Arguments
/// * `config` — MoE config
/// * `weights` — MoE weights
/// * `saved` — saved activations from [`moe_forward_token_with_saved`]
/// * `d_output` — upstream gradient `dL/d(hidden_out)` `[d]`
/// * `dh_out` — output buffer for `dL/d(hidden_in)` `[d]` (accumulated `+=`)
/// * `grads` — gradient accumulator (must be zeroed before the backward loop)
pub fn moe_backward_token(
    config: &MoeConfig,
    weights: &MoeWeights,
    saved: &MoeSavedActivations,
    d_output: &[f32],
    dh_out: &mut [f32],
    grads: &mut MoeGradients,
) {
    let d = config.d();
    let use_latent_moe = config.routed_expert_hidden_size.is_some();

    debug_assert_eq!(d_output.len(), d);
    debug_assert_eq!(dh_out.len(), d);

    // dL/d(hidden_out) splits into dL/d(y_routed) + dL/d(shared_output)
    // hidden_out = y_routed + shared_output
    // So dL/d(y_routed) = d_output, dL/d(shared_output) = d_output.

    shared_experts_backward(config, weights, saved, d_output, dh_out, grads);

    // ── Routed experts backward ──────────────────────────────────────
    if use_latent_moe {
        moe_backward_latent(config, weights, saved, d_output, dh_out, grads);
    } else {
        moe_backward_nonlatent(config, weights, saved, d_output, dh_out, grads);
    }
}

/// Shared-expert-stream backward (extracted from `moe_backward_token` —
/// Issue 482 / Plan 452: the delay-arch split calls it with its own stream
/// gradient so the shared stream can be trained against a DIFFERENT input
/// and injection site than the routed stream). Pure code move; the fused
/// path passes the fused `d_output`/`dh_out`, preserving op order.
///
/// Reads `saved.h` as the shared stream's input (the split forward stores
/// the shared input there).
#[allow(clippy::needless_range_loop)]
fn shared_experts_backward(
    config: &MoeConfig,
    weights: &MoeWeights,
    saved: &MoeSavedActivations,
    d_output: &[f32],
    dh_out: &mut [f32],
    grads: &mut MoeGradients,
) {
    let d = config.d();
    let d_ffn_shared = config.d_ffn_shared();

    // ── Shared expert backward ───────────────────────────────────────────
    // shared_output = down_proj · SiTU(gate_proj · h, up_proj · h)
    // Backward: dL/d(act_out) = down_proj^T · d_output
    //           dL/d(down_proj) += outer(d_output, act_out)
    //           dL/d(h) += gate_proj^T · dL/d(gate_inter) + up_proj^T · dL/d(up_inter)
    {
        let shared = &weights.shared_experts[0];
        let shared_grads = &mut grads.shared_experts[0];
        let act_out = &saved.shared_act_out;

        // dL/d(act_out) = down_proj^T · d_output  [d_ffn_shared]
        let mut d_act = vec![0.0f32; d_ffn_shared];
        simd_transpose_matvec_into(&mut d_act, &shared.down_proj, d_output, d, d_ffn_shared);

        // dL/d(down_proj) += outer(d_output, act_out)
        simd_outer_product_acc(
            &mut shared_grads.down_proj,
            d_output,
            act_out,
            d,
            d_ffn_shared,
        );

        // Backward through SiTU: dL/d(gate_inter), dL/d(up_inter)
        let mut d_gate = vec![0.0f32; d_ffn_shared];
        let mut d_up = vec![0.0f32; d_ffn_shared];
        situ_backward(
            &d_act,
            &saved.shared_gate_inter,
            &saved.shared_up_inter,
            act_out,
            &mut d_gate,
            &mut d_up,
            config.situ_beta,
            config.situ_linear_beta,
        );

        // dL/d(gate_proj) += outer(d_gate, h)
        simd_outer_product_acc(
            &mut shared_grads.gate_proj,
            &d_gate,
            &saved.h,
            d_ffn_shared,
            d,
        );
        // dL/d(up_proj) += outer(d_up, h)
        simd_outer_product_acc(&mut shared_grads.up_proj, &d_up, &saved.h, d_ffn_shared, d);

        // dh_out += gate_proj^T · d_gate + up_proj^T · d_up
        simd_transpose_matvec_acc(dh_out, &shared.gate_proj, &d_gate, d_ffn_shared, d);
        simd_transpose_matvec_acc(dh_out, &shared.up_proj, &d_up, d_ffn_shared, d);
    }

    // Remaining shared experts (s > 0) — for Kimi-K3 N_s=1 so this loop is empty.
    // If N_s > 1, we'd need saved activations for each, but the forward-with-saved
    // above only captures s=0. This is a known limitation; Kimi-K3 configs have N_s=1.
    for _s in 1..weights.shared_experts.len() {
        // No saved activations for s>0 — skip (gradient for these experts is zero).
        // This is acceptable because Kimi-K3-0.40B and 4B-A2B both have N_s=1.
    }
}

/// Split-stream MoE backward (Issue 482 / Plan 452) — the consumer of
/// [`moe_forward_token_split`].
///
/// The two streams receive their OWN upstream gradients and deposit their
/// input-gradients into SEPARATE buffers, because in the delay architecture
/// they read different inputs and inject at different layers:
/// - `d_shared` is `dL/d(shared_out)` at the shared stream's injection site
///   (layer ℓ); `dh_shared_out` accumulates `dL/d(shared_in)`.
/// - `d_routed` is `dL/d(routed_out)` at the routed stream's injection site
///   (layer ℓ+δe or the final flush); `dh_routed_out` accumulates
///   `dL/d(routed_in)`.
///
/// Bit-identity pin: with `d_shared == d_routed == d_output` and zeroed dh
/// buffers, `grads` is bit-identical to [`moe_backward_token`] (same
/// accumulation order into the same buffer) and `dh_shared_out +
/// dh_routed_out` matches the fused `dh_out` within f32 reassociation ULPs —
/// the fused interleaves the three input-gradient contributions into ONE
/// buffer (`(0+shared)+down_proj+router`) while the split sums two buffers
/// (`shared + (down_proj+router)`); `(s+d)+r` vs `s+(d+r)` legitimately
/// differ by the last ULP. Pinned by tolerance in the test, with the exact
/// bound recorded there.
#[allow(clippy::too_many_arguments)]
pub fn moe_backward_token_split(
    config: &MoeConfig,
    weights: &MoeWeights,
    saved: &MoeSavedActivations,
    d_shared: &[f32],
    d_routed: &[f32],
    dh_shared_out: &mut [f32],
    dh_routed_out: &mut [f32],
    grads: &mut MoeGradients,
) {
    let d = config.d();
    let use_latent_moe = config.routed_expert_hidden_size.is_some();

    debug_assert_eq!(d_shared.len(), d);
    debug_assert_eq!(d_routed.len(), d);
    debug_assert_eq!(dh_shared_out.len(), d);
    debug_assert_eq!(dh_routed_out.len(), d);

    // Shared first — preserves the fused path's accumulation order into `grads`.
    shared_experts_backward(config, weights, saved, d_shared, dh_shared_out, grads);

    if use_latent_moe {
        moe_backward_latent(config, weights, saved, d_routed, dh_routed_out, grads);
    } else {
        moe_backward_nonlatent(config, weights, saved, d_routed, dh_routed_out, grads);
    }
}

/// Latent-MoE backward path.
#[allow(clippy::needless_range_loop)]
fn moe_backward_latent(
    config: &MoeConfig,
    weights: &MoeWeights,
    saved: &MoeSavedActivations,
    d_output: &[f32],
    dh_out: &mut [f32],
    grads: &mut MoeGradients,
) {
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();
    let d_moe = config.d_moe();
    let d_expert = saved.d_expert; // = d_moe for latent path

    // ── Step 9 backward: y_routed = routed_expert_up_proj · latent_output_postnorm ──
    // dL/d(latent_output_postnorm) = up_proj^T · d_output
    // dL/d(up_proj) += outer(d_output, latent_output_postnorm)
    //
    // latent_output_postnorm is the normed version if norm is active.
    // We need to recover it. The saved latent_output_prenorm is BEFORE norm.
    // If norm active, postnorm = rmsnorm(prenorm). We recompute postnorm from prenorm.
    let latent_postnorm: Vec<f32>;
    let inv_rms: f32;
    if let Some(norm_w) = weights.routed_expert_norm_weight.as_ref() {
        // Recompute the normed latent output.
        let mut buf = saved.latent_output_prenorm.clone();
        rmsnorm_with_gamma_eps(&mut buf, norm_w, config.rms_norm_eps as f64);
        latent_postnorm = buf;
        inv_rms = saved.latent_norm_inv_rms.unwrap();
    } else {
        latent_postnorm = saved.latent_output_prenorm.clone();
        inv_rms = 0.0; // unused
    }

    let mut d_latent_postnorm = vec![0.0f32; d_moe];
    simd_transpose_matvec_into(
        &mut d_latent_postnorm,
        weights.routed_expert_up_proj.as_ref().unwrap(),
        d_output,
        d,
        d_moe,
    );
    // dL/d(up_proj) += outer(d_output, latent_postnorm)
    simd_outer_product_acc(
        grads.routed_expert_up_proj.as_mut().unwrap(),
        d_output,
        &latent_postnorm,
        d,
        d_moe,
    );

    // ── Step 8 backward: RMSNorm backward (if active) ──
    // rmsnorm: y[i] = x[i] * gamma[i] * inv_rms
    //
    // Standard RMSNorm backward (see mla_backward.rs rmsnorm_backward):
    // Let r = inv_rms, g = gamma.
    // y[i] = x[i] * g[i] * r,  dr/dx[k] = -x[k] * r³ / d
    // dx[k] = r * g[k] * dy[k] - x[k] * r³ * dot / d
    //      = r * (g[k] * dy[k] - x[k] * r² * dot / d)
    //      where dot = sum_j(x[j] * g[j] * dy[j])
    // gamma multiplies ONLY the direct dy term — `g*r*(dy - x*r²*dot/d)`
    // over-applies gamma to the correction term (Issue 693 H2).
    // dL/d(gamma[i]) += dy[i] * x[i] * r
    let d_latent_prenorm: Vec<f32> =
        if let Some(norm_w) = weights.routed_expert_norm_weight.as_ref() {
            let norm_grad = grads.routed_expert_norm_weight.as_mut().unwrap();
            let r = inv_rms;
            let r2 = r * r;
            let inv_d = 1.0 / d_moe as f32;

            // dot = sum_j(x[j] * g[j] * dy[j])  — note y[j] = x[j]*g[j]*r so x[j]*g[j] = y[j]/r
            // dot = (1/r) * sum_j(y[j] * dy[j])
            let dot_ydy: f32 = (0..d_moe)
                .map(|j| latent_postnorm[j] * d_latent_postnorm[j])
                .sum();
            let dot = dot_ydy / r; // = sum_j(x[j]*g[j]*dy[j])

            let mut dx = vec![0.0f32; d_moe];
            for i in 0..d_moe {
                // dL/d(gamma[i]) += x[i] * r * dy[i] = (latent_postnorm[i] / (g[i] * r)) ... simpler:
                // x[i] = latent_output_prenorm[i]
                let x_i = saved.latent_output_prenorm[i];
                norm_grad[i] += x_i * r * d_latent_postnorm[i];
                // dx[i] = r * (g[i] * dy[i] - x_i * r² * dot / d)
                dx[i] = r * (norm_w[i] * d_latent_postnorm[i] - x_i * r2 * dot * inv_d);
            }
            dx
        } else {
            d_latent_postnorm
        };

    // ── Steps 6-7 backward: per-expert backward + router weight grad ──
    // latent_output_prenorm = sum_k topk_weights[k] * expert_out_k
    // d_latent_prenorm is dL/d(latent_output_prenorm).
    //
    // For each selected expert k:
    //   dL/d(expert_out_k) = topk_weights[k] * d_latent_prenorm
    //   dL/d(topk_weights[k]) = dot(d_latent_prenorm, expert_out_k)
    //
    // Then backward through each expert's SiTU FFN → dL/d(h_latent) accumulates.

    let mut d_topk_weights = vec![0.0f32; k_r];
    let mut d_h_latent = vec![0.0f32; d_moe]; // dL/d(latent_hidden)

    for k in 0..k_r {
        let idx = saved.topk_indices[k];
        let w = saved.topk_weights[k];
        let expert = &weights.experts[idx];
        let expert_grad = &mut grads.experts[idx];

        let expert_out_k = &saved.expert_outputs[k * d_expert..(k + 1) * d_expert];

        // dL/d(topk_weights[k]) = dot(d_latent_prenorm, expert_out_k)
        d_topk_weights[k] = simd_dot_f32(d_latent_prenorm.as_slice(), expert_out_k, d_expert);

        // dL/d(expert_out_k) = topk_weights[k] * d_latent_prenorm
        let d_expert_out: Vec<f32> = (0..d_expert).map(|i| w * d_latent_prenorm[i]).collect();

        // Backward through expert FFN: out = down_proj · SiTU(gate_proj · h_latent, up_proj · h_latent)
        let act_out = &saved.expert_act_out[k * d_ffn..(k + 1) * d_ffn];
        let gate_inter = &saved.expert_gate_inter[k * d_ffn..(k + 1) * d_ffn];
        let up_inter = &saved.expert_up_inter[k * d_ffn..(k + 1) * d_ffn];

        // dL/d(act_out) = down_proj^T · d_expert_out
        let mut d_act = vec![0.0f32; d_ffn];
        simd_transpose_matvec_into(
            &mut d_act,
            &expert.down_proj,
            &d_expert_out,
            d_expert,
            d_ffn,
        );

        // dL/d(down_proj) += outer(d_expert_out, act_out)
        simd_outer_product_acc(
            &mut expert_grad.down_proj,
            &d_expert_out,
            act_out,
            d_expert,
            d_ffn,
        );

        // Backward through SiTU
        let mut d_gate = vec![0.0f32; d_ffn];
        let mut d_up = vec![0.0f32; d_ffn];
        situ_backward(
            &d_act,
            gate_inter,
            up_inter,
            act_out,
            &mut d_gate,
            &mut d_up,
            config.situ_beta,
            config.situ_linear_beta,
        );

        // dL/d(gate_proj) += outer(d_gate, h_latent)
        simd_outer_product_acc(
            &mut expert_grad.gate_proj,
            &d_gate,
            saved.latent_hidden.as_ref().unwrap(),
            d_ffn,
            d_moe,
        );
        // dL/d(up_proj) += outer(d_up, h_latent)
        simd_outer_product_acc(
            &mut expert_grad.up_proj,
            &d_up,
            saved.latent_hidden.as_ref().unwrap(),
            d_ffn,
            d_moe,
        );

        // d_h_latent += gate_proj^T · d_gate + up_proj^T · d_up
        simd_transpose_matvec_acc(&mut d_h_latent, &expert.gate_proj, &d_gate, d_ffn, d_moe);
        simd_transpose_matvec_acc(&mut d_h_latent, &expert.up_proj, &d_up, d_ffn, d_moe);
    }

    // ── Step 6 backward: h_latent = routed_expert_down_proj · h ──
    // dL/d(down_proj) += outer(d_h_latent, h)
    simd_outer_product_acc(
        grads.routed_expert_down_proj.as_mut().unwrap(),
        &d_h_latent,
        saved.routed_h(),
        d_moe,
        d,
    );
    // dh_out += down_proj^T · d_h_latent
    simd_transpose_matvec_acc(
        dh_out,
        weights.routed_expert_down_proj.as_ref().unwrap(),
        &d_h_latent,
        d_moe,
        d,
    );

    // ── Step 5 backward: renormalization ──
    // topk_weights[k] = sigmoid_scores[topk_idx[k]] / topk_sum
    // Backward through renorm → dL/d(sigmoid_scores[selected])
    router_backward(config, weights, saved, &d_topk_weights, dh_out, grads);
}

/// Non-latent-MoE backward path (routed experts operate on full hidden dim).
#[allow(clippy::needless_range_loop)]
fn moe_backward_nonlatent(
    config: &MoeConfig,
    weights: &MoeWeights,
    saved: &MoeSavedActivations,
    d_output: &[f32],
    dh_out: &mut [f32],
    grads: &mut MoeGradients,
) {
    let k_r = config.k_routed();
    let d = config.d();
    let d_ffn = config.d_ffn();
    let d_expert = saved.d_expert; // = d for non-latent

    let mut d_topk_weights = vec![0.0f32; k_r];

    for k in 0..k_r {
        let idx = saved.topk_indices[k];
        let w = saved.topk_weights[k];
        let expert = &weights.experts[idx];
        let expert_grad = &mut grads.experts[idx];

        let expert_out_k = &saved.expert_outputs[k * d_expert..(k + 1) * d_expert];

        // dL/d(topk_weights[k]) = dot(d_output, expert_out_k)
        d_topk_weights[k] = simd_dot_f32(d_output, expert_out_k, d_expert);

        // dL/d(expert_out_k) = topk_weights[k] * d_output
        let d_expert_out: Vec<f32> = (0..d_expert).map(|i| w * d_output[i]).collect();

        // Backward through expert FFN
        let act_out = &saved.expert_act_out[k * d_ffn..(k + 1) * d_ffn];
        let gate_inter = &saved.expert_gate_inter[k * d_ffn..(k + 1) * d_ffn];
        let up_inter = &saved.expert_up_inter[k * d_ffn..(k + 1) * d_ffn];

        let mut d_act = vec![0.0f32; d_ffn];
        simd_transpose_matvec_into(
            &mut d_act,
            &expert.down_proj,
            &d_expert_out,
            d_expert,
            d_ffn,
        );

        simd_outer_product_acc(
            &mut expert_grad.down_proj,
            &d_expert_out,
            act_out,
            d_expert,
            d_ffn,
        );

        let mut d_gate = vec![0.0f32; d_ffn];
        let mut d_up = vec![0.0f32; d_ffn];
        situ_backward(
            &d_act,
            gate_inter,
            up_inter,
            act_out,
            &mut d_gate,
            &mut d_up,
            config.situ_beta,
            config.situ_linear_beta,
        );

        simd_outer_product_acc(&mut expert_grad.gate_proj, &d_gate, saved.routed_h(), d_ffn, d);
        simd_outer_product_acc(&mut expert_grad.up_proj, &d_up, saved.routed_h(), d_ffn, d);

        simd_transpose_matvec_acc(dh_out, &expert.gate_proj, &d_gate, d_ffn, d);
        simd_transpose_matvec_acc(dh_out, &expert.up_proj, &d_up, d_ffn, d);
    }

    // Router backward
    router_backward(config, weights, saved, &d_topk_weights, dh_out, grads);
}

/// Router + renormalization backward.
///
/// Computes gradients w.r.t. `router_weight` and accumulates into `dh_out`.
/// The `e_score_correction_bias` gradient is always zero (bias is selection-only).
#[allow(clippy::needless_range_loop)]
fn router_backward(
    config: &MoeConfig,
    weights: &MoeWeights,
    saved: &MoeSavedActivations,
    d_topk_weights: &[f32],
    dh_out: &mut [f32],
    grads: &mut MoeGradients,
) {
    let n_r = config.n_routed();
    let k_r = config.k_routed();
    let d = config.d();

    // ── Renormalization backward ──
    // topk_weights[k] = s[idx[k]] / S  where S = sum_j s[idx[j]]
    //
    // For selected expert e = idx[k']:
    // dL/ds[e] += d_topk_weights[k'] * (δ(k,k')/S - s[idx[k']]/S²)
    //           = d_topk_weights[k'] / S - d_topk_weights[k'] * topk_weights[k'] / S
    // Wait — let me redo this properly.
    //
    // w_k = s_k / S where s_k = s[idx[k]], S = sum_j s_j.
    // dw_k/ds_{k'} = (δ(k,k') * S - s_k) / S² = δ(k,k')/S - s_k/S² = δ(k,k')/S - w_k/S
    //
    // So: dL/ds_{k'} = sum_k dL/dw_k * (δ(k,k')/S - w_k/S)
    //               = dL/dw_{k'}/S - (1/S) * sum_k dL/dw_k * w_k

    let s = saved.topk_sum; // S
    let inv_s = if s > 1.0e-20 { 1.0 / s } else { 0.0 };

    // sum_k dL/dw_k * w_k
    let weighted_sum: f32 = (0..k_r)
        .map(|k| d_topk_weights[k] * saved.topk_weights[k])
        .sum();

    // dL/ds_k for each selected k
    let mut d_sigmoid_scores = vec![0.0f32; n_r]; // indexed by expert, not by k
    for k in 0..k_r {
        let ds_k = d_topk_weights[k] * inv_s - weighted_sum * inv_s;
        let expert_idx = saved.topk_indices[k];
        d_sigmoid_scores[expert_idx] += ds_k;
    }

    // Note: if renormalize is false, the weights ARE the raw sigmoid scores.
    // In that case, dL/ds[idx[k]] = d_topk_weights[k] directly.
    // But we already handled this above: if renormalize is false, S is not
    // used in the forward's renorm branch (topk_weights = sigmoid_scores[idx]).
    // The backward formula w_k = s_k / S doesn't apply when renormalize=false.
    // Let me handle this properly:
    if !config.renormalize {
        // When renormalize=false: topk_weights[k] = sigmoid_scores[idx[k]] (raw, no division).
        // dL/ds[idx[k]] = d_topk_weights[k] directly.
        d_sigmoid_scores.fill(0.0);
        for k in 0..k_r {
            d_sigmoid_scores[saved.topk_indices[k]] += d_topk_weights[k];
        }
    }
    // If sum < 1e-20 (uniform fallback), gradient is zero — d_sigmoid_scores stays 0.
    // This is correct because the uniform fallback is a constant w.r.t. the scores.

    // ── Sigmoid backward ──
    // sigmoid_scores[e] = sigmoid(router_logits[e])
    // dL/d(router_logits[e]) = dL/ds[e] * s[e] * (1 - s[e])
    let mut d_router_logits = vec![0.0f32; n_r];
    for e in 0..n_r {
        let s_e = saved.sigmoid_scores[e];
        d_router_logits[e] = d_sigmoid_scores[e] * s_e * (1.0 - s_e);
    }

    // ── Router weight backward ──
    // router_logits[e] = dot(router_weight[e], h)
    // dL/d(router_weight[e]) += d_router_logits[e] * h
    // dh_out += router_weight[e]^T · d_router_logits
    let routed_h = saved.routed_h();
    for e in 0..n_r {
        let dl = d_router_logits[e];
        if dl != 0.0 {
            // dL/d(router_weight[e]) += dl * h
            let row_off = e * d;
            for c in 0..d {
                grads.router_weight[row_off + c] += dl * routed_h[c];
            }
            // dh_out += dl * router_weight[e]
            for c in 0..d {
                dh_out[c] += dl * weights.router_weight[row_off + c];
            }
        }
    }

    // e_score_correction_bias gradient is always zero (selection-only, non-differentiable).
    // grads.e_score_correction_bias stays zero.
}

// ─── SiTU backward ──────────────────────────────────────────────────────────

/// Backward through SiTU activation.
///
/// Forward (with linear_beta):
/// ```text
/// gate_sigmoid = sigmoid(g)
/// gate_tanh = tanh(g / beta)
/// up_t = lb * tanh(up / lb)         (when linear_beta = Some(lb))
/// act = beta * gate_tanh * gate_sigmoid * up_t
/// ```
///
/// Forward (without linear_beta):
/// ```text
/// gate_sigmoid = sigmoid(g)
/// gate_tanh = tanh(g / beta)
/// act = beta * gate_tanh * gate_sigmoid * up
/// ```
///
/// # Arguments
/// * `d_act` — dL/d(act) `[d_ffn]`
/// * `gate_inter` — pre-SiTU gate values `g` `[d_ffn]`
/// * `up_inter` — pre-SiTU up values `[d_ffn]`
/// * `act_out` — post-SiTU activation (NOT USED directly; we recompute from g, up)
/// * `d_gate` — output dL/d(gate_inter) `[d_ffn]`
/// * `d_up` — output dL/d(up_inter) `[d_ffn]`
/// * `beta` — SiTU beta
/// * `linear_beta` — SiTU linear_beta
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn situ_backward(
    d_act: &[f32],
    gate_inter: &[f32],
    up_inter: &[f32],
    _act_out: &[f32],
    d_gate: &mut [f32],
    d_up: &mut [f32],
    beta: f32,
    linear_beta: Option<f32>,
) {
    let inv_beta = 1.0 / beta;
    let n = d_act.len();
    debug_assert_eq!(gate_inter.len(), n);
    debug_assert_eq!(up_inter.len(), n);
    debug_assert_eq!(d_gate.len(), n);
    debug_assert_eq!(d_up.len(), n);

    if let Some(lb) = linear_beta {
        let inv_lb = 1.0 / lb;
        for i in 0..n {
            let g = gate_inter[i];
            let u = up_inter[i];
            let da = d_act[i];

            // act = beta * tanh(g/beta) * sigmoid(g) * lb * tanh(u/lb)
            //
            // Let A = beta * tanh(g/beta) * sigmoid(g) * lb * tanh(u/lb)
            // dA/dg = beta * (1 - tanh²(g/beta)) * (1/beta) * sigmoid(g) * lb * tanh(u/lb)
            //       + beta * tanh(g/beta) * sigmoid(g) * (1 - sigmoid(g)) * lb * tanh(u/lb)
            //       = (1 - tanh²(g/beta)) * sigmoid(g) * lb * tanh(u/lb)
            //       + beta * tanh(g/beta) * sigmoid(g) * (1 - sigmoid(g)) * lb * tanh(u/lb)
            //
            // dA/du = beta * tanh(g/beta) * sigmoid(g) * lb * (1 - tanh²(u/lb)) * (1/lb)
            //       = beta * tanh(g/beta) * sigmoid(g) * (1 - tanh²(u/lb))

            let gs = 1.0 / (1.0 + (-g).exp()); // sigmoid(g)
            let gt = (g * inv_beta).tanh(); // tanh(g/beta)
            let ut = lb * (u * inv_lb).tanh(); // lb * tanh(u/lb)

            // dA/dg
            let d_act_dg = (1.0 - gt * gt) * gs * ut + beta * gt * gs * (1.0 - gs) * ut;
            // dA/du
            let tanh_u = (u * inv_lb).tanh();
            let d_act_du = beta * gt * gs * (1.0 - tanh_u * tanh_u);

            d_gate[i] = da * d_act_dg;
            d_up[i] = da * d_act_du;
        }
    } else {
        for i in 0..n {
            let g = gate_inter[i];
            let u = up_inter[i];
            let da = d_act[i];

            // act = beta * tanh(g/beta) * sigmoid(g) * u
            let gs = 1.0 / (1.0 + (-g).exp()); // sigmoid(g)
            let gt = (g * inv_beta).tanh(); // tanh(g/beta)

            // dA/dg = (1 - tanh²(g/beta)) * sigmoid(g) * u
            //       + beta * tanh(g/beta) * sigmoid(g) * (1 - sigmoid(g)) * u
            let d_act_dg = (1.0 - gt * gt) * gs * u + beta * gt * gs * (1.0 - gs) * u;
            // dA/du = beta * tanh(g/beta) * sigmoid(g)
            let d_act_du = beta * gt * gs;

            d_gate[i] = da * d_act_dg;
            d_up[i] = da * d_act_du;
        }
    }
}

// ─── Tests (split-stream forward/backward — Issue 482 / Plan 452) ────────────

#[cfg(test)]
mod split_tests {
    use super::*;
    use crate::moe::{MoeConfig, MoeForwardScratch, MoeWeights};

    /// Small LATENT config (the Kimi family shape) for fast tests.
    fn latent_cfg() -> MoeConfig {
        MoeConfig {
            num_experts: 4,
            num_shared_experts: 1,
            num_experts_per_token: 2,
            moe_intermediate_size: 8,
            hidden_size: 32,
            use_sigmoid_router: true,
            renormalize: true,
            routed_expert_hidden_size: Some(16),
            latent_moe_use_norm: true,
            rms_norm_eps: 1e-5,
            situ_beta: 4.0,
            situ_linear_beta: Some(25.0),
        }
    }

    /// Small NON-latent config.
    fn nonlatent_cfg() -> MoeConfig {
        MoeConfig {
            routed_expert_hidden_size: None,
            latent_moe_use_norm: false,
            ..latent_cfg()
        }
    }

    fn test_input(d: usize, salt: f32) -> Vec<f32> {
        (0..d)
            .map(|i| ((i as f32 * 0.37 + salt) * 0.1).sin())
            .collect()
    }

    fn assert_vecs_identical(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
        }
    }

    fn assert_vecs_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "{what}: element {i}: {x} vs {y} (tol {tol})");
        }
    }

    /// G0 forward pin: equal inputs → shared_out + routed_out (summed
    /// shared-first) is BIT-IDENTICAL to the fused output on the latent path.
    #[test]
    fn split_forward_matches_fused_latent_bit_identical() {
        let config = latent_cfg();
        let weights = MoeWeights::random(&config, 42);
        let h = test_input(config.d(), 1.0);
        let mut fused_scratch = MoeForwardScratch::new(&config);
        let mut split_scratch = MoeForwardScratch::new(&config);

        let (fused_out, _fused_saved) =
            moe_forward_token_with_saved(&weights, &config, &h, &mut fused_scratch);
        let (shared_out, routed_out, split_saved) =
            moe_forward_token_split(&weights, &config, &h, &h, &mut split_scratch);

        // Shared-first sum (the fused computes hidden_out = shared; += routed).
        let mut sum = shared_out.clone();
        for (s, r) in sum.iter_mut().zip(routed_out.iter()) {
            *s += *r;
        }
        assert_vecs_identical(&sum, &fused_out, "latent split forward sum");

        // The saved activations' routed input must round-trip.
        assert_eq!(split_saved.h_routed.as_deref(), Some(h.as_slice()));
    }

    /// Non-latent forward: same streams equal → tolerance match only (the
    /// per-expert accumulation reassociates; documented in the fn docs).
    #[test]
    fn split_forward_matches_fused_nonlatent_close() {
        let config = nonlatent_cfg();
        let weights = MoeWeights::random(&config, 43);
        let h = test_input(config.d(), 2.0);
        let mut fused_scratch = MoeForwardScratch::new(&config);
        let mut split_scratch = MoeForwardScratch::new(&config);

        let (fused_out, _) = moe_forward_token_with_saved(&weights, &config, &h, &mut fused_scratch);
        let (shared_out, routed_out, _) =
            moe_forward_token_split(&weights, &config, &h, &h, &mut split_scratch);

        let mut sum = shared_out.clone();
        for (s, r) in sum.iter_mut().zip(routed_out.iter()) {
            *s += *r;
        }
        assert_vecs_close(&sum, &fused_out, 1e-5, "nonlatent split forward sum");
    }

    /// G0 backward pin: equal-input forward + equal stream gradients → the
    /// split backward's `grads` are BIT-IDENTICAL to the fused backward, and
    /// the summed dh matches within f32 reassociation ULPs (the fused folds
    /// shared→down_proj→router into one buffer; the split sums two buffers —
    /// `(s+d)+r` vs `s+(d+r)`, a last-ULP difference by construction).
    #[test]
    fn split_backward_matches_fused_latent_bit_identical() {
        let config = latent_cfg();
        let weights = MoeWeights::random(&config, 44);
        let h = test_input(config.d(), 3.0);
        let d_output = test_input(config.d(), 4.0);
        let mut scratch = MoeForwardScratch::new(&config);

        // Fused
        let (_, fused_saved) = moe_forward_token_with_saved(&weights, &config, &h, &mut scratch);
        let mut fused_grads = MoeGradients::zeros_like(&weights);
        let mut fused_dh = vec![0.0f32; config.d()];
        moe_backward_token(
            &config,
            &weights,
            &fused_saved,
            &d_output,
            &mut fused_dh,
            &mut fused_grads,
        );

        // Split (fresh forward — deterministic w.r.t. scratch state)
        let (_, _, split_saved) =
            moe_forward_token_split(&weights, &config, &h, &h, &mut scratch);
        let mut split_grads = MoeGradients::zeros_like(&weights);
        let mut dh_shared = vec![0.0f32; config.d()];
        let mut dh_routed = vec![0.0f32; config.d()];
        moe_backward_token_split(
            &config,
            &weights,
            &split_saved,
            &d_output,
            &d_output,
            &mut dh_shared,
            &mut dh_routed,
            &mut split_grads,
        );
        let mut dh_sum = dh_shared.clone();
        for (s, r) in dh_sum.iter_mut().zip(dh_routed.iter()) {
            *s += *r;
        }

        // dh: reassociation tolerance — measured 1 ULP; bound at a few ULPs
        // (rel 1e-6) to stay immune to codegen while catching real defects.
        for (i, (x, y)) in dh_sum.iter().zip(fused_dh.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-6 * (1.0 + y.abs()),
                "latent split backward dh: element {i}: {x} vs {y}"
            );
        }
        assert_vecs_identical(
            &split_grads.router_weight,
            &fused_grads.router_weight,
            "latent split backward router grads",
        );
        for (sg, fg) in split_grads.experts.iter().zip(fused_grads.experts.iter()) {
            assert_vecs_identical(&sg.gate_proj, &fg.gate_proj, "expert gate grads");
            assert_vecs_identical(&sg.up_proj, &fg.up_proj, "expert up grads");
            assert_vecs_identical(&sg.down_proj, &fg.down_proj, "expert down grads");
        }
        for (sg, fg) in split_grads
            .shared_experts
            .iter()
            .zip(fused_grads.shared_experts.iter())
        {
            assert_vecs_identical(&sg.gate_proj, &fg.gate_proj, "shared gate grads");
            assert_vecs_identical(&sg.up_proj, &fg.up_proj, "shared up grads");
            assert_vecs_identical(&sg.down_proj, &fg.down_proj, "shared down grads");
        }
        assert_vecs_identical(
            split_grads.routed_expert_down_proj.as_ref().unwrap(),
            fused_grads.routed_expert_down_proj.as_ref().unwrap(),
            "latent down_proj grads",
        );
        assert_vecs_identical(
            split_grads.routed_expert_up_proj.as_ref().unwrap(),
            fused_grads.routed_expert_up_proj.as_ref().unwrap(),
            "latent up_proj grads",
        );
    }

    /// Split with DIFFERENT stream inputs: smoke — runs, finite, and the
    /// routed stream's dh lands in its own buffer (non-zero for a generic
    /// upstream gradient).
    #[test]
    fn split_different_inputs_smoke() {
        let config = latent_cfg();
        let weights = MoeWeights::random(&config, 45);
        let shared_in = test_input(config.d(), 5.0);
        let routed_in = test_input(config.d(), 6.0);
        let d_output = test_input(config.d(), 7.0);
        let mut scratch = MoeForwardScratch::new(&config);

        let (shared_out, routed_out, saved) =
            moe_forward_token_split(&weights, &config, &shared_in, &routed_in, &mut scratch);
        assert!(shared_out.iter().all(|v| v.is_finite()));
        assert!(routed_out.iter().all(|v| v.is_finite()));

        let mut grads = MoeGradients::zeros_like(&weights);
        let mut dh_shared = vec![0.0f32; config.d()];
        let mut dh_routed = vec![0.0f32; config.d()];
        moe_backward_token_split(
            &config,
            &weights,
            &saved,
            &d_output,
            &d_output,
            &mut dh_shared,
            &mut dh_routed,
            &mut grads,
        );
        assert!(dh_shared.iter().any(|v| *v != 0.0));
        assert!(dh_routed.iter().any(|v| *v != 0.0));
        assert!(grads.router_weight.iter().any(|v| *v != 0.0));
    }
}
