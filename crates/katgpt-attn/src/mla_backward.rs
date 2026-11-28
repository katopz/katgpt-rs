//! MLA analytic backward pass (Plan 318 Phase C C4).
//!
//! Implements the analytic gradient of `mla_forward_token` w.r.t. all
//! trainable parameters (`MlaWeights`) and the input hidden state `h`.
//!
//! # Scope
//!
//! Single-token backward. MLA caches `c_kv` (the compressed KV latent) per
//! token; the backward computes weight gradients accumulating from ALL cached
//! tokens (since W_UK/W_UV are shared). The outgoing gradient `dL/d(c_kv_t)`
//! for the current token is returned for multi-token BPTT threading.
//!
//! # Why this lives here
//!
//! katgpt-rs is modelless-by-mandate (no training at runtime), BUT this module
//! is the **CPU reference** for the GPU backward (C4). It belongs alongside the
//! forward. Gated behind `mla_backward` (implies `mla_attention`).

use crate::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token};
use katgpt_core::simd::{
    simd_dot_f32, simd_outer_product_acc, simd_sum_sq, simd_transpose_matvec_acc,
    simd_transpose_matvec_into,
};
use katgpt_kv::shard_kv::rope::RopeFreqs;

// ─── Saved activations ──────────────────────────────────────────────────────

/// Forward-pass saved activations needed by the backward.
///
/// Populated by [`mla_forward_token_with_saved`].
#[derive(Clone)]
pub struct MlaSavedActivations {
    /// Input hidden state `h`. `[d]`.
    pub h: Vec<f32>,
    /// `c_kv_raw = W_DKV · h` BEFORE norm. `[d_c]`.
    pub c_kv_raw: Vec<f32>,
    /// `c_kv` AFTER norm (the cached value). `[d_c]`.
    pub c_kv_normed: Vec<f32>,
    /// `c_q_raw = W_DQ · h` BEFORE norm. `[d_qc]`.
    pub c_q_raw: Vec<f32>,
    /// `c_q` AFTER norm. `[d_qc]`.
    pub c_q_normed: Vec<f32>,
    /// RMSNorm inv_rms for c_kv.
    pub c_kv_inv_rms: f32,
    /// RMSNorm inv_rms for c_q.
    pub c_q_inv_rms: f32,

    /// `q_c = W_UQ · c_q_normed` (post-RoPE if use_nope=false — but q_c has no RoPE). `[d_h*n_h]`.
    pub q_c: Vec<f32>,
    /// `q_r_raw = W_QR · c_q_normed` BEFORE RoPE. `[d_r*n_h]`.
    pub q_r_raw: Vec<f32>,
    /// `q_r` AFTER RoPE (same as q_r_raw if use_nope). `[d_r*n_h]`.
    pub q_r: Vec<f32>,
    /// `k_r_raw = W_KR · h` BEFORE RoPE. `[d_r]`.
    pub k_r_raw: Vec<f32>,
    /// `k_r` AFTER RoPE. `[d_r]`.
    pub k_r: Vec<f32>,

    /// Per-head attention weights (post-softmax). `[n_h][seq]` — flattened `[n_h * seq]`.
    pub attn_weights: Vec<f32>,
    /// Per-head attention output (pre-gate). `[v_h * n_h]`.
    pub attn_out: Vec<f32>,
    /// Per-head attention output (post-gate = attn_out * gate). `[v_h * n_h]`.
    /// Same as attn_out when use_output_gate is false.
    pub attn_out_gated: Vec<f32>,
    /// Output gate values `sigmoid(W_g · h)`. `[v_h * n_h]`. Empty if no gate.
    pub gate_values: Vec<f32>,
    /// Position of the current token.
    pub pos: usize,
    /// Sequence length (number of cached tokens including current).
    pub seq: usize,
}

// ─── Gradients ──────────────────────────────────────────────────────────────

/// Gradient accumulator for MLA backward.
///
/// All fields mirror `MlaWeights`. Accumulated (`+=`) during backward.
#[derive(Clone, Debug)]
pub struct MlaGradients {
    pub w_dkv: Vec<f32>,
    pub w_dq: Vec<f32>,
    pub w_uq: Vec<f32>,
    pub w_qr: Vec<f32>,
    pub w_uk: Vec<f32>,
    pub w_uv: Vec<f32>,
    pub w_kr: Vec<f32>,
    pub w_o: Vec<f32>,
    pub q_a_norm_weight: Vec<f32>,
    pub kv_a_norm_weight: Vec<f32>,
    /// `None` if `use_output_gate` is false.
    pub w_g: Option<Vec<f32>>,
}

impl MlaGradients {
    /// Allocate zeroed gradients matching the given weights' shapes.
    pub fn zeros_like(w: &MlaWeights) -> Self {
        Self {
            w_dkv: vec![0.0; w.w_dkv.len()],
            w_dq: vec![0.0; w.w_dq.len()],
            w_uq: vec![0.0; w.w_uq.len()],
            w_qr: vec![0.0; w.w_qr.len()],
            w_uk: vec![0.0; w.w_uk.len()],
            w_uv: vec![0.0; w.w_uv.len()],
            w_kr: vec![0.0; w.w_kr.len()],
            w_o: vec![0.0; w.w_o.len()],
            q_a_norm_weight: vec![0.0; w.q_a_norm_weight.len()],
            kv_a_norm_weight: vec![0.0; w.kv_a_norm_weight.len()],
            w_g: w.w_g.as_ref().map(|x| vec![0.0; x.len()]),
        }
    }
}

// ─── Forward with saved ─────────────────────────────────────────────────────

/// Run the MLA forward and capture all activations needed for backward.
///
/// Returns `(output, saved_activations)`.
///
/// **Allocation discipline:** allocates the saved activations. This is the
/// training-time reference path, NOT the inference hot path.
pub fn mla_forward_token_with_saved(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
) -> (Vec<f32>, MlaSavedActivations) {
    let d = config.hidden_size;
    let d_c = config.kv_lora_rank;
    let d_qc = config.q_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;
    let pos = cache.seq_len;
    let seq_before = cache.seq_len;

    // Snapshot input hidden
    let h_snap = h.to_vec();

    // ── Run the stock forward ──
    let output_ref = mla_forward_token(config, weights, cache, scratch, rope_freqs, h);
    let output = output_ref.to_vec();

    // The forward has mutated scratch in-place. Now snapshot the intermediates.
    // We need to reconstruct pre-norm values since rmsnorm_inplace overwrites
    // c_kv and c_q in-place.

    // c_kv_raw: recompute from h (W_DKV · h)
    let mut c_kv_raw = vec![0.0f32; d_c];
    katgpt_core::simd::simd_matmul_rows(&mut c_kv_raw, &weights.w_dkv, h, d_c, d);

    // c_q_raw: recompute from h (W_DQ · h)
    let mut c_q_raw = vec![0.0f32; d_qc];
    katgpt_core::simd::simd_matmul_rows(&mut c_q_raw, &weights.w_dq, h, d_qc, d);

    // Compute inv_rms for both norms (matching rmsnorm_inplace's computation)
    let c_kv_inv_rms = {
        let sum_sq = simd_sum_sq(&c_kv_raw, d_c);
        1.0 / (sum_sq / d_c as f32 + config.rms_norm_eps).sqrt()
    };
    let c_q_inv_rms = {
        let sum_sq = simd_sum_sq(&c_q_raw, d_qc);
        1.0 / (sum_sq / d_qc as f32 + config.rms_norm_eps).sqrt()
    };

    // c_kv_normed = c_kv_raw * inv_rms * gamma (what the forward computed)
    // The forward applied rmsnorm_inplace to scratch.c_kv, so scratch.c_kv
    // is the normed value. But the cache also has it (cache was appended).
    // The current token's c_kv_normed is at cache position `pos` = seq_before.
    let c_kv_normed = cache.latent_kv_at(seq_before).to_vec();

    // Debug: verify c_kv_normed = c_kv_raw * gamma * inv_rms
    debug_assert_eq!(c_kv_normed.len(), d_c);
    for i in 0..d_c {
        let expected = c_kv_raw[i] * weights.kv_a_norm_weight[i] * c_kv_inv_rms;
        let re = (c_kv_normed[i] - expected).abs() / expected.abs().max(1e-6);
        debug_assert!(
            re < 1e-4,
            "c_kv_normed mismatch at {}: cached={:.6e} expected={:.6e} re={:.6}",
            i,
            c_kv_normed[i],
            expected,
            re
        );
    }
    let c_q_normed = scratch.c_q.clone(); // post-norm (forward overwrote it)

    // q_c, q_r are in scratch (post up-projection; q_r has RoPE applied if !use_nope)
    let q_c = scratch.q_c.clone();
    let q_r = scratch.q_r.clone(); // post-RoPE

    // q_r_raw: recompute (pre-RoPE) — only needed if !use_nope
    let q_r_raw = if !config.use_nope {
        // q_r_raw = W_QR · c_q_normed (pre-RoPE)
        let mut buf = vec![0.0f32; d_r * n_h];
        katgpt_core::simd::simd_matmul_rows(&mut buf, &weights.w_qr, &c_q_normed, d_r * n_h, d_qc);
        buf
    } else {
        q_r.clone() // identity when use_nope
    };

    // k_r is in scratch (post-RoPE)
    let k_r = scratch.k_r.clone();

    // k_r_raw: recompute (pre-RoPE)
    let k_r_raw = if !config.use_nope {
        // k_r_raw = W_KR · h
        let mut buf = vec![0.0f32; d_r];
        katgpt_core::simd::simd_matmul_rows(&mut buf, &weights.w_kr, h, d_r, d);
        buf
    } else {
        k_r.clone()
    };

    // Attention weights: recompute from scratch.scores (which were softmaxed
    // in-place during the forward). But scores were overwritten per-head in a
    // loop — only the LAST head's scores remain. We need to recompute all
    // heads' attention weights.
    //
    // The forward computes softmax(scores) per head, then immediately uses it
    // for the value weighted sum. The softmax weights are NOT saved per-head.
    // We recompute them here.
    let scale = config.attn_scale();
    let seq = cache.seq_len; // includes the current token
    let mut attn_weights = vec![0.0f32; n_h * seq];

    for head in 0..n_h {
        let q_c_h = &scratch.q_c[head * d_h..(head + 1) * d_h];
        let q_r_h = &scratch.q_r[head * d_r..(head + 1) * d_r];
        let scores_h = &mut attn_weights[head * seq..head * seq + seq];

        let mut max_score = f32::NEG_INFINITY;
        #[allow(clippy::needless_range_loop)]
        for j in 0..seq {
            let c_kv_j = cache.latent_kv_at(j);
            let k_r_j = cache.rope_key_at(j);

            // k_c_j = W_UK[head slice] · c_kv_j
            let mut k_c_j = vec![0.0f32; d_h];
            katgpt_core::simd::simd_matmul_rows(
                &mut k_c_j,
                &weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                c_kv_j,
                d_h,
                d_c,
            );

            let content_dot = simd_dot_f32(q_c_h, &k_c_j, d_h);
            let rope_dot = simd_dot_f32(q_r_h, k_r_j, d_r);
            let score = (content_dot + rope_dot) * scale;
            scores_h[j] = score;
            if score > max_score {
                max_score = score;
            }
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for s in scores_h.iter_mut().take(seq) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        for s in scores_h.iter_mut().take(seq) {
            *s *= inv_sum;
        }
    }

    // Output gate values (must compute BEFORE attn_out_pre recovery)
    let proj_size = v_h * n_h;
    let gate_values = if config.use_output_gate
        && let Some(ref w_g) = weights.w_g
    {
        // gate = sigmoid(W_g · h) — recompute
        let mut buf = vec![0.0f32; proj_size];
        katgpt_core::simd::simd_matmul_rows(&mut buf, w_g, h, proj_size, d);
        for val in buf.iter_mut() {
            *val = 1.0 / (1.0 + (-*val).exp());
        }
        buf
    } else {
        Vec::new()
    };

    // attn_out (PRE-gate) — scratch.attn_out is POST-gate (the gate is applied
    // in-place). Recover pre-gate by dividing by gate values.
    let attn_out_gated = scratch.attn_out.clone();
    let attn_out_pre = if config.use_output_gate && !gate_values.is_empty() {
        (0..proj_size)
            .map(|i| {
                let g = gate_values[i];
                // Issue 460: `attn_out_pre` is genuinely UNRECOVERABLE where the
                // gate underflowed to zero — `gated` is then 0 too and 0/0
                // carries no information. The old unguarded divide produced NaN
                // here and poisoned every consumer downstream.
                //
                // The gradient path no longer reads this field (the `g` cancels
                // analytically — see `d_gate_pre` below), so nothing depends on
                // the reconstruction any more. It is kept because the field is
                // public, and reported as 0.0 on the singular elements: a
                // finite, obviously-neutral value beats a NaN that silently
                // propagates. Callers needing the exact pre-gate activation
                // should use the checkpoint path, which SAVES it instead of
                // dividing it back out (`kimi_k3/checkpoint.rs`).
                if g > 0.0 { attn_out_gated[i] / g } else { 0.0 }
            })
            .collect::<Vec<_>>()
    } else {
        attn_out_gated.clone()
    };

    let saved = MlaSavedActivations {
        h: h_snap,
        c_kv_raw,
        c_kv_normed,
        c_q_raw,
        c_q_normed,
        c_kv_inv_rms,
        c_q_inv_rms,
        q_c,
        q_r_raw,
        q_r,
        k_r_raw,
        k_r,
        attn_weights,
        attn_out: attn_out_pre,
        attn_out_gated,
        gate_values,
        pos,
        seq,
    };

    let _ = (d_h, d_r, v_h);
    (output, saved)
}

// ─── Backward ───────────────────────────────────────────────────────────────

/// MLA analytic backward pass.
///
/// Delegate over [`mla_backward_token_with_cache_grads`] with the export
/// slots absent (`None`/`None`) — bit-identical incumbent.
///
/// Given `dL_t/d(output_t)`, computes:
/// - Gradients w.r.t. all trainable parameters (`grads`) — accumulated across
///   all attention positions (shared weights).
/// - Gradient w.r.t. input hidden states for ALL tokens `0..=pos` (`all_dh`).
///   The current token's `all_dh[pos]` gets the full path (W_DKV + W_DQ + W_KR
///   + W_g). Past tokens' `all_dh[j]` get the KV-cache path (W_DKV + W_KR
///     through their cached c_kv and k_r).
///
/// # Cross-token gradient propagation
///
/// In MLA decode, the current token `t` attends to ALL cached tokens `j ≤ t`.
/// The gradient `dL_t/d(c_kv_j)` flows back to `dL_t/d(h_j)` through the
/// `j`-th token's `W_DKV` projection and RMSNorm. Similarly for `k_r_j`.
/// This function propagates these cross-token gradients to `all_dh[j]`.
///
/// # Arguments
/// * `config` — MLA config
/// * `weights` — MLA weights
/// * `cache` — KV cache (read-only — past tokens' c_kv and k_r)
/// * `saved` — saved activations for the CURRENT token
/// * `all_saved` — saved activations for ALL tokens (for cross-token rmsnorm)
/// * `rope_freqs` — RoPE frequency table (for inverse RoPE in backward)
/// * `d_output` — upstream gradient `dL_t/d(output_t)` `[d]`
/// * `all_dh` — output buffers for `dL/d(h)` per token (accumulated `+=`)
/// * `grads` — gradient accumulator
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn mla_backward_token(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &MlaKVCache,
    saved: &MlaSavedActivations,
    all_saved: &[MlaSavedActivations],
    rope_freqs: &mut RopeFreqs,
    d_output: &[f32],
    all_dh: &mut [Vec<f32>],
    grads: &mut MlaGradients,
) {
    mla_backward_token_with_cache_grads(
        config, weights, cache, saved, all_saved, rope_freqs, d_output, all_dh, grads, None, None,
    );
}

/// [`mla_backward_token`] with the per-position KV-cache-entry gradient
/// export (Issue 1253 — riir-train Issue 618 T0.b looped-training consumer).
/// Identical body; when the export slots are armed, the PRE-consumption
/// accumulator values are additionally folded into the caller's buffers
/// (`+=`):
///
/// - `d_cache_latent[j*d_c..(j+1)*d_c] += dL/d(c_kv_normed_j)` for all
///   `j ∈ 0..seq`. The cache stores the POST-rmsnorm latent, so this is
///   exactly `dL/d(cache.latent_kv[j])`, taken BEFORE the `rmsnorm_backward`
///   pass below folds it into each position's W_DKV/`all_dh` chain.
/// - `d_cache_rope[j*d_r..(j+1)*d_r] += dL/d(k_r_j)`. The cache stores the
///   POST-RoPE key, so this is exactly `dL/d(cache.rope_key[j])`, taken
///   BEFORE the inverse-RoPE pass below folds it into W_KR/`all_dh`.
///
/// Accumulation (`+=`) makes repeated calls over the same cache rows
/// compose: a cache entry is read by many backward calls, and each call adds
/// its own reads. Slot buffers are caller-owned, `[max_seq][kv_lora_rank]` /
/// `[max_seq][qk_rope_head_dim]`-shaped (only rows `0..seq` are touched —
/// the asserts name the sizing). The `None` posture is bit-identical to the
/// incumbent delegate.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn mla_backward_token_with_cache_grads(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &MlaKVCache,
    saved: &MlaSavedActivations,
    all_saved: &[MlaSavedActivations],
    rope_freqs: &mut RopeFreqs,
    d_output: &[f32],
    all_dh: &mut [Vec<f32>],
    grads: &mut MlaGradients,
    d_cache_latent: Option<&mut [f32]>,
    d_cache_rope: Option<&mut [f32]>,
) {
    let d = config.hidden_size;
    let d_c = config.kv_lora_rank;
    let d_qc = config.q_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;
    let scale = config.attn_scale();
    let seq = saved.seq;
    let proj_size = v_h * n_h;
    let pos = saved.pos;

    debug_assert_eq!(d_output.len(), d);
    debug_assert!(pos < all_dh.len());
    debug_assert!(pos < all_saved.len());

    // dh_out for the current token — used for the query/gate paths (which only
    // affect the current token's hidden state). We use `all_dh[pos]` directly
    // at each call site to avoid holding a mutable borrow across the
    // multi-token propagation loops below.
    debug_assert!(pos < all_dh.len());
    debug_assert!(pos < all_saved.len());

    // ── Step 12 backward: output = W_O · gated_attn_out ──
    // dL/d(gated_attn_out) = W_O^T · d_output
    // dL/d(W_O) += outer(d_output, gated_attn_out)
    let mut d_gated_attn = vec![0.0f32; proj_size];
    simd_transpose_matvec_into(&mut d_gated_attn, &weights.w_o, d_output, d, proj_size);
    simd_outer_product_acc(
        &mut grads.w_o,
        d_output,
        &saved.attn_out_gated,
        d,
        proj_size,
    );

    // ── Step 11 backward: output gate ──
    // gated_attn = attn_out * sigmoid(g_proj · h)
    // dL/d(attn_out) = dL/d(gated_attn) * gate
    // dL/d(g_proj_out) = dL/d(gated_attn) * attn_out * gate * (1 - gate)
    let d_attn_out: Vec<f32> = if config.use_output_gate && !saved.gate_values.is_empty() {
        let mut d_attn = vec![0.0f32; proj_size];
        let mut d_gate_pre = vec![0.0f32; proj_size]; // dL/d(W_g · h) before sigmoid
        for i in 0..proj_size {
            let g = saved.gate_values[i];
            d_attn[i] = d_gated_attn[i] * g;
            // Issue 460: this reads
            //     d_gate_pre = d_gated_attn * attn_out_PRE * g * (1 - g)
            // and `attn_out_pre` used to be reconstructed as `attn_out_gated / g`
            // (see the recompute path above). That division is singular exactly
            // when the gate saturates: `sigmoid(W_g · h)` underflows to a HARD
            // ZERO in f32 once the pre-activation drops below ~-88, and then the
            // reconstruction is 0/0 = NaN.
            //
            // Since `attn_out_gated == attn_out_pre * g` by construction, the
            // `g` cancels:
            //     attn_out_pre * g * (1 - g) == attn_out_gated * (1 - g)
            // so the identity below is bit-equivalent wherever the old form was
            // finite, removes the singularity entirely (no epsilon, no clamp,
            // no bias), and drops one divide per element.
            //
            // Measured before the fix (`random_train_init` seed 42, 0.40B
            // shape): 4 of 10 real sequences produced an all-zero gate at one
            // token, which turned every downstream gradient non-finite — while
            // the forward and the loss stayed perfectly normal, so nothing
            // upstream flagged it.
            d_gate_pre[i] = d_gated_attn[i] * saved.attn_out_gated[i] * (1.0 - g);
        }
        // dL/d(W_g) += outer(d_gate_pre, h)
        if let Some(ref mut w_g_grad) = grads.w_g {
            simd_outer_product_acc(w_g_grad, &d_gate_pre, &saved.h, proj_size, d);
        }
        // dh_out += W_g^T · d_gate_pre
        if let Some(ref w_g) = weights.w_g {
            simd_transpose_matvec_acc(&mut all_dh[pos], w_g, &d_gate_pre, proj_size, d);
        }
        d_attn
    } else {
        d_gated_attn
    };

    // ── Steps 6-10 backward: attention ──
    // For each head h:
    //   o_h = Σ_j attn_weight_j * v_c_j_h
    //   dL/d(v_c_j_h) += attn_weight_j * dL/d(o_h)
    //   dL/d(attn_weight_j) = dot(dL/d(o_h), v_c_j_h)
    //   softmax backward → dL/d(score_j)
    //   score_j = (q_c_h · k_c_j_h + q_r_h · k_r_j) * scale
    //   dL/d(q_c_h) += Σ_j dL/d(score_j) * scale * k_c_j_h
    //   dL/d(k_c_j_h) += dL/d(score_j) * scale * q_c_h
    //   dL/d(q_r_h) += Σ_j dL/d(score_j) * scale * k_r_j
    //   dL/d(k_r_j) += Σ_j dL/d(score_j) * scale * q_r_h

    let mut d_q_c = vec![0.0f32; d_h * n_h]; // dL/d(q_c)
    let mut d_q_r = vec![0.0f32; d_r * n_h]; // dL/d(q_r post-RoPE)
    let mut d_k_r_all = vec![0.0f32; d_r * seq]; // dL/d(k_r_j) for all j
    // Per-position dL_t/d(c_kv_normed) — each position j has its OWN c_kv that
    // flows through its OWN rmsnorm + W_DKV, so we track gradients per position.
    let mut d_c_kv_normed_all: Vec<Vec<f32>> = (0..seq).map(|_| vec![0.0f32; d_c]).collect();

    for head in 0..n_h {
        let d_o_h = &d_attn_out[head * v_h..(head + 1) * v_h];
        let attn_w_h = &saved.attn_weights[head * seq..head * seq + seq];
        let q_c_h = &saved.q_c[head * d_h..(head + 1) * d_h];
        let q_r_h = &saved.q_r[head * d_r..(head + 1) * d_r];

        // ── Phase 1: compute d_attn_w[j] for all j (needed for softmax backward) ──
        // d_attn_w[j] = dot(d_o_h, v_c_j_h)
        // Also accumulate W_UV gradient + d_c_kv_normed for j==pos.
        let mut d_attn_w = vec![0.0f32; seq];
        for j in 0..seq {
            let c_kv_j = if j == saved.pos {
                &saved.c_kv_normed[..]
            } else {
                cache.latent_kv_at(j)
            };

            // v_c_j_h = W_UV[head slice] · c_kv_j
            let mut v_c_j_h = vec![0.0f32; v_h];
            katgpt_core::simd::simd_matmul_rows(
                &mut v_c_j_h,
                &weights.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
                c_kv_j,
                v_h,
                d_c,
            );

            d_attn_w[j] = simd_dot_f32(d_o_h, &v_c_j_h, v_h);

            // dL/d(v_c_j_h) = attn_weight_j * d_o_h
            let d_v_c_j_h: Vec<f32> = (0..v_h).map(|i| attn_w_h[j] * d_o_h[i]).collect();

            // dL/d(W_UV[head slice]) += outer(d_v_c_j_h, c_kv_j)
            simd_outer_product_acc(
                &mut grads.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
                &d_v_c_j_h,
                c_kv_j,
                v_h,
                d_c,
            );

            // dL/d(c_kv_j) += W_UV[head slice]^T · d_v_c_j_h
            // Accumulate for ALL positions — each position's c_kv flows back
            // through its own rmsnorm + W_DKV.
            simd_transpose_matvec_acc(
                &mut d_c_kv_normed_all[j],
                &weights.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
                &d_v_c_j_h,
                v_h,
                d_c,
            );
        }

        // ── Softmax backward ──
        let dot_w_dw: f32 = (0..seq).map(|k| attn_w_h[k] * d_attn_w[k]).sum();
        let mut d_scores = vec![0.0f32; seq];
        for j in 0..seq {
            d_scores[j] = attn_w_h[j] * (d_attn_w[j] - dot_w_dw);
        }

        // ── Phase 2: score backward → q/k gradients + W_UK + d_c_kv_normed ──
        let mut d_q_c_h = vec![0.0f32; d_h];
        let mut d_q_r_h = vec![0.0f32; d_r];

        for j in 0..seq {
            let c_kv_j = if j == saved.pos {
                &saved.c_kv_normed[..]
            } else {
                cache.latent_kv_at(j)
            };
            let k_r_j = cache.rope_key_at(j);

            // k_c_j_h = W_UK[head slice] · c_kv_j
            let mut k_c_j_h = vec![0.0f32; d_h];
            katgpt_core::simd::simd_matmul_rows(
                &mut k_c_j_h,
                &weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                c_kv_j,
                d_h,
                d_c,
            );

            let ds = d_scores[j] * scale;

            // dL/d(q_c_h) += ds * k_c_j_h
            for i in 0..d_h {
                d_q_c_h[i] += ds * k_c_j_h[i];
            }
            // dL/d(k_c_j_h) = ds * q_c_h
            let d_k_c_j_h: Vec<f32> = (0..d_h).map(|i| ds * q_c_h[i]).collect();

            // dL/d(W_UK[head slice]) += outer(d_k_c_j_h, c_kv_j)
            simd_outer_product_acc(
                &mut grads.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                &d_k_c_j_h,
                c_kv_j,
                d_h,
                d_c,
            );

            // dL/d(c_kv_j) += W_UK[head slice]^T · d_k_c_j_h
            // Accumulate for ALL positions.
            simd_transpose_matvec_acc(
                &mut d_c_kv_normed_all[j],
                &weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                &d_k_c_j_h,
                d_h,
                d_c,
            );

            // dL/d(q_r_h) += ds * k_r_j
            for i in 0..d_r {
                d_q_r_h[i] += ds * k_r_j[i];
            }
            // dL/d(k_r_j) += ds * q_r_h
            for i in 0..d_r {
                d_k_r_all[j * d_r + i] += ds * q_r_h[i];
            }
        }

        d_q_c[head * d_h..(head + 1) * d_h].copy_from_slice(&d_q_c_h);
        d_q_r[head * d_r..(head + 1) * d_r].copy_from_slice(&d_q_r_h);
        let _ = q_r_h;
    }

    // ── Issue 1253 export: per-position cache-entry gradients ──
    // Taken AFTER the head loop (both accumulators are complete) and BEFORE
    // the consumption passes below (rmsnorm → W_DKV/all_dh for the latent,
    // inverse-RoPE → W_KR/all_dh for the rope key). The cache stores exactly
    // these values — post-rmsnorm latent, post-RoPE key — so the exported
    // slot IS dL/d(cache entry). Includes j == pos: for the current token,
    // the saved copy equals the cache entry (the debug-asserted identity),
    // so its slot value is the self-read gradient of the entry this very
    // call wrote.
    if let Some(d_cache_latent) = d_cache_latent {
        assert!(
            d_cache_latent.len() >= seq * d_c,
            "d_cache_latent must be [max_seq][kv_lora_rank]-shaped (len {}, need seq*kv_lora_rank = {})",
            d_cache_latent.len(),
            seq * d_c
        );
        for j in 0..seq {
            let src = &d_c_kv_normed_all[j];
            let dst = &mut d_cache_latent[j * d_c..(j + 1) * d_c];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }
    }
    if let Some(d_cache_rope) = d_cache_rope {
        assert!(
            d_cache_rope.len() >= seq * d_r,
            "d_cache_rope must be [max_seq][qk_rope_head_dim]-shaped (len {}, need seq*qk_rope_head_dim = {})",
            d_cache_rope.len(),
            seq * d_r
        );
        for j in 0..seq {
            let src = &d_k_r_all[j * d_r..(j + 1) * d_r];
            let dst = &mut d_cache_rope[j * d_r..(j + 1) * d_r];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }
    }

    // ── Step 5 backward: q_r post-RoPE → q_r_raw ──
    // If !use_nope: q_r = RoPE(q_r_raw). Backward: dL/d(q_r_raw) = RoPE^T · dL/d(q_r)
    // RoPE is a rotation, so RoPE^T = RoPE with negated angle = apply(negate=true).
    let d_q_r_raw = if !config.use_nope {
        let mut buf = d_q_r.clone();
        for head in 0..n_h {
            let start = head * d_r;
            rope_freqs.apply(&mut buf[start..start + d_r], saved.pos, true);
        }
        buf
    } else {
        d_q_r
    };

    // ── Step 4 backward: q_c = W_UQ · c_q_normed ──
    // dL/d(c_q_normed) = W_UQ^T · dL/d(q_c)
    // dL/d(W_UQ) += outer(d_q_c, c_q_normed)
    let mut d_c_q_normed = vec![0.0f32; d_qc];
    simd_transpose_matvec_into(&mut d_c_q_normed, &weights.w_uq, &d_q_c, d_h * n_h, d_qc);
    simd_outer_product_acc(&mut grads.w_uq, &d_q_c, &saved.c_q_normed, d_h * n_h, d_qc);

    // ── Step 5b backward: q_r_raw = W_QR · c_q_normed ──
    // dL/d(c_q_normed) += W_QR^T · dL/d(q_r_raw)
    // dL/d(W_QR) += outer(d_q_r_raw, c_q_normed)
    simd_transpose_matvec_acc(
        &mut d_c_q_normed,
        &weights.w_qr,
        &d_q_r_raw,
        d_r * n_h,
        d_qc,
    );
    simd_outer_product_acc(
        &mut grads.w_qr,
        &d_q_r_raw,
        &saved.c_q_normed,
        d_r * n_h,
        d_qc,
    );

    // ── Step 3 backward: c_q_normed = rmsnorm(c_q_raw, q_a_norm_weight) ──
    // dL/d(c_q_raw), dL/d(q_a_norm_weight)
    let d_c_q_raw = rmsnorm_backward(
        &d_c_q_normed,
        &saved.c_q_raw,
        &weights.q_a_norm_weight,
        saved.c_q_inv_rms,
        &mut grads.q_a_norm_weight,
        config.rms_norm_eps,
    );

    // ── Step 1b backward: c_q_raw = W_DQ · h ──
    // dL/d(W_DQ) += outer(d_c_q_raw, h)
    // dh_out += W_DQ^T · d_c_q_raw
    simd_outer_product_acc(&mut grads.w_dq, &d_c_q_raw, &saved.h, d_qc, d);
    simd_transpose_matvec_acc(&mut all_dh[pos], &weights.w_dq, &d_c_q_raw, d_qc, d);

    // ── k_r gradient: propagate d_k_r_all[j] for ALL positions ──
    // For each j in 0..seq: k_r_j = RoPE(W_KR · h_j) at position j.
    //   dL/d(k_r_j_raw) = RoPE^T(position j) · dL/d(k_r_j)
    //   dL/d(W_KR) += outer(d_k_r_j_raw, h_j)
    //   all_dh[j] += W_KR^T · d_k_r_j_raw
    for j in 0..seq {
        let d_k_r_j = &d_k_r_all[j * d_r..(j + 1) * d_r];
        let d_k_r_j_raw = if !config.use_nope {
            let mut buf = d_k_r_j.to_vec();
            rope_freqs.apply(&mut buf, j, true);
            buf
        } else {
            d_k_r_j.to_vec()
        };
        simd_outer_product_acc(&mut grads.w_kr, &d_k_r_j_raw, &all_saved[j].h, d_r, d);
        simd_transpose_matvec_acc(&mut all_dh[j], &weights.w_kr, &d_k_r_j_raw, d_r, d);
    }

    // ── Step 1 backward: c_kv_normed = rmsnorm(c_kv_raw, kv_a_norm_weight) ──
    // Propagate d_c_kv_normed_all[j] for ALL positions through each position's
    // OWN rmsnorm + W_DKV to all_dh[j] and grads.w_dkv.
    for j in 0..seq {
        let d_c_kv_j_raw = rmsnorm_backward(
            &d_c_kv_normed_all[j],
            &all_saved[j].c_kv_raw,
            &weights.kv_a_norm_weight,
            all_saved[j].c_kv_inv_rms,
            &mut grads.kv_a_norm_weight,
            config.rms_norm_eps,
        );

        // dL/d(W_DKV) += outer(d_c_kv_j_raw, h_j)
        // all_dh[j] += W_DKV^T · d_c_kv_j_raw
        simd_outer_product_acc(&mut grads.w_dkv, &d_c_kv_j_raw, &all_saved[j].h, d_c, d);
        simd_transpose_matvec_acc(&mut all_dh[j], &weights.w_dkv, &d_c_kv_j_raw, d_c, d);
    }

    let _ = (d_h, d_r, v_h, proj_size);
}

// ─── RMSNorm backward ───────────────────────────────────────────────────────

/// Backward through `rmsnorm_inplace`: `y[i] = x[i] * gamma[i] * inv_rms`.
///
/// Returns `dL/d(x_raw)`. Accumulates into `grad_gamma`.
///
/// The `inv_rms` is the value computed during the forward (before norm).
pub fn rmsnorm_backward(
    d_y: &[f32],
    x_raw: &[f32],
    gamma: &[f32],
    inv_rms: f32,
    grad_gamma: &mut [f32],
    eps: f32,
) -> Vec<f32> {
    let n = d_y.len();
    let r = inv_rms;
    let r2 = r * r;
    let inv_n = 1.0 / n as f32;

    // dot = sum_j(x_raw[j] * gamma[j] * d_y[j])
    // y[j] = x_raw[j] * gamma[j] * r
    // x_raw[j] * gamma[j] = y[j] / r (if r ≠ 0)
    // But we have x_raw directly, so:
    let dot: f32 = (0..n).map(|j| x_raw[j] * gamma[j] * d_y[j]).sum();

    let mut dx = vec![0.0f32; n];
    for i in 0..n {
        // dL/d(gamma[i]) += x_raw[i] * r * d_y[i]
        grad_gamma[i] += x_raw[i] * r * d_y[i];
        // dx[i] = r * (gamma[i] * d_y[i] - x_raw[i] * r2 * dot * inv_n)
        //
        // Derivation: y[i] = x[i] * gamma[i] * r where r = (mean(x^2)+eps)^{-1/2}
        //   dL/dx[k] = sum_i d_y[i] * dy[i]/dx[k]
        //   dy[i]/dx[k] = gamma[i] * (delta_ik * r + x[i] * dr/dx[k])
        //   dr/dx[k] = -x[k] * r^3 / n
        //   => dL/dx[k] = r * gamma[k] * d_y[k] - x[k] * r^3 * dot / n
        //   where dot = sum_i(x[i] * gamma[i] * d_y[i])
        //   => dL/dx[k] = r * (gamma[k] * d_y[k] - x[k] * r^2 * dot / n)
        //
        // NOTE: gamma multiplies ONLY d_y[k], NOT the dot term. The common
        // incorrect form `gamma * r * (d_y - x*r2*dot*inv_n)` over-applies
        // gamma to the normalization correction.
        dx[i] = r * (gamma[i] * d_y[i] - x_raw[i] * r2 * dot * inv_n);
    }
    let _ = eps;
    dx
}
