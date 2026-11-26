//! KDA analytic backward pass (Issue 389 T4 + Plan 318 Phase C C5 implementation).
//!
//! Implements the analytic gradient of `kda_forward_token` w.r.t. all
//! trainable parameters (`KdaWeights`) and the input hidden state `h`.
//! Derived in `riir-train/.issues/389_kda_gpu_backward_ssm_research.md` T2
//! by manual differentiation of the forward.
//!
//! # Scope
//!
//! Two entry points:
//! - [`kda_backward_token`] — single-token backward (the BPTT building block).
//!   Propagates the k=0 ShortConv tap only (current-token contribution). Correct
//!   for L=1 or as the composition primitive where the caller handles multi-token
//!   conv threading externally.
//! - [`kda_backward_sequence`] — multi-token BPTT with conv-ring cross-token
//!   gradient propagation (Plan 318 Phase C C5). Distributes all K ShortConv
//!   taps across past tokens. This is the training-loop entry point.
//!
//! Both share [`kda_core_backward`] (steps 5→2: output proj + RMSNorm gate +
//! recurrence + gates/decay/L2Norm), which is DRY-extracted so the two entry
//! points cannot diverge on the recurrence/norm math.
//!
//! # Why this lives here
//!
//! katgpt-rs is modelless-by-mandate (no training at runtime), BUT this module
//! is the **CPU reference** for the GPU backward (C5). It belongs alongside the
//! forward for two reasons: (1) the CPU primitive stays public so riir-train can
//! consume it; (2) the finite-difference gradient check (T4 + C5) needs it
//! in-tree. Production inference never calls this — it's gated behind
//! `kda_backward` (implies `kda_linear`) and only consumed by riir-train-gpu.

/// Clamp an f64 value to the f32-safe range before casting. Prevents
/// overflow-to-inf when the KDA backward's intermediate gradient values
/// exceed f32::MAX (3.4e38). The clamp threshold 1e35 leaves room for
/// downstream multiplications without overflow. NaN → 0.
///
/// This is the numerical stabilization for the exploding-gradient problem
/// in the KDA state-space backward: the dk×dk state matrix operations can
/// amplify gradients by ~1e6 per layer, and after 6 KDA layers the products
/// exceed f32 range. The clamp preserves gradient direction; only magnitude
/// is bounded.
#[inline]
fn clamp_f64_to_f32(val: f64) -> f32 {
    const F32_SAFE_MAX: f64 = 1e35;
    if !val.is_finite() {
        0.0
    } else if val.abs() > F32_SAFE_MAX {
        (val.signum() * F32_SAFE_MAX) as f32
    } else {
        val as f32
    }
}

use crate::gdn2::kda_forward::kda_forward_token;
use crate::gdn2::kda_forward::{KdaConfig, KdaForwardScratch, KdaLayerCache, KdaWeights};
use katgpt_core::simd::{
    simd_outer_product_acc, simd_sum_sq, simd_transpose_matvec_acc, simd_transpose_matvec_into,
};

// ─── Saved activations ──────────────────────────────────────────────────────

/// Forward-pass saved activations needed by the backward.
///
/// Populated by [`kda_forward_token_with_saved`]. All tensors are owned
/// snapshots taken AFTER the forward completes (the forward mutates its
/// scratch in-place; we clone the values it leaves behind + reconstruct
/// per-head intermediates from the post-forward state).
#[derive(Clone)]
pub struct KdaSavedActivations {
    /// Input hidden state `h`. `[hidden_size]`.
    pub h: Vec<f32>,
    /// `z_q = W^q · h` (pre-conv). `[proj]`.
    pub z_q: Vec<f32>,
    /// `z_k = W^k · h` (pre-conv). `[proj]`.
    pub z_k: Vec<f32>,
    /// `z_v = W^v · h` (pre-conv). `[proj]`.
    pub z_v: Vec<f32>,
    /// `z_q_conv` post-SiLU (= the conv output after SiLU). `[proj]`.
    pub z_q_conv: Vec<f32>,
    /// `z_k_conv` post-SiLU. `[proj]`.
    pub z_k_conv: Vec<f32>,
    /// `z_v_conv` post-SiLU. `[proj]`.
    pub z_v_conv: Vec<f32>,
    /// `f_a_hidden = W^{f_a} · h`. `[dk]`.
    pub f_a_hidden: Vec<f32>,
    /// `g_raw = W^{f_b} · f_a_hidden`. `[proj]`.
    pub g_raw: Vec<f32>,
    /// `beta_pre = W^β · h`. `[n_heads]`.
    pub beta_pre: Vec<f32>,
    /// `g_out = W^g · h`. `[proj]`.
    pub g_out: Vec<f32>,
    /// Post-norm+gate output (= o_concat). `[proj]`.
    pub o_concat: Vec<f32>,
    /// Per-head activations (indexed `[head]`).
    pub heads: Vec<KdaHeadActivations>,
    /// ShortConv ring buffer snapshot for q (BEFORE the forward step). `[proj*ks]`.
    pub conv_buf_q: Vec<f32>,
    /// ShortConv ring buffer snapshot for k (BEFORE the forward step). `[proj*ks]`.
    pub conv_buf_k: Vec<f32>,
    /// ShortConv ring buffer snapshot for v (BEFORE the forward step). `[proj*ks]`.
    pub conv_buf_v: Vec<f32>,
    /// Ring buffer write index BEFORE the forward step.
    pub conv_buf_idx: usize,
}

/// Per-head forward activations.
#[derive(Clone)]
pub struct KdaHeadActivations {
    /// `S_{t-1}` — incoming state (before decay+update). `[dk*dk]`.
    pub s_prev: Vec<f32>,
    /// `S' = S_{t-1} · diag(a)` — post-decay. `[dk*dk]`.
    pub s_prime: Vec<f32>,
    /// `S''` — post-update (= outgoing state S_t). `[dk*dk]`.
    pub s_post: Vec<f32>,
    /// `q_h` — L2-normed + scaled query. `[dk]`.
    pub q_normed: Vec<f32>,
    /// `k_h` — L2-normed key. `[dk]`.
    pub k_normed: Vec<f32>,
    /// `v_h` — value (= z_v_conv head slice). `[dk]`.
    pub v: Vec<f32>,
    /// `decay[i] = max(exp(gk[i]), alpha_eps)`. `[dk]`.
    pub a_decay: Vec<f32>,
    /// `gk[i] = -alpha_head · softplus(g_plus[i])`. `[dk]`.
    pub gk: Vec<f32>,
    /// `g_plus[i] = g_raw[off+i] + dt_bias[off+i]`. `[dk]`.
    pub g_plus: Vec<f32>,
    /// `delta[j] = beta_h · v_h[j] − r[j]`. `[dk]`.
    pub delta: Vec<f32>,
    /// `out_raw[j]` — pre-RMSNorm readout output. `[dk]`.
    pub out_raw: Vec<f32>,
    /// `alpha_head = exp(A_log[h])`.
    pub alpha_head: f32,
    /// `beta_h = sigmoid(beta_pre[h])`.
    pub beta_h: f32,
    /// `1 / sqrt(mean(out_raw²) + rms_eps)`.
    pub inv_rms: f32,
}

// ─── Gradient accumulator ───────────────────────────────────────────────────

/// Gradient accumulator mirroring `KdaWeights`.
///
/// All fields are accumulated (`+=`) by `kda_backward_token`. Initialize with
/// [`KdaGradients::zeros_like`] before the backward loop.
#[derive(Clone)]
pub struct KdaGradients {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub q_conv_weight: Vec<f32>,
    pub k_conv_weight: Vec<f32>,
    pub v_conv_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub f_a_proj: Vec<f32>,
    pub f_b_proj: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub beta_proj: Vec<f32>,
    pub g_proj: Vec<f32>,
    pub o_norm_weight: Vec<f32>,
    pub o_proj: Vec<f32>,
}

impl KdaGradients {
    /// Allocate zeroed gradients matching the given weights' shapes.
    pub fn zeros_like(w: &KdaWeights) -> Self {
        Self {
            q_proj: vec![0.0; w.q_proj.len()],
            k_proj: vec![0.0; w.k_proj.len()],
            v_proj: vec![0.0; w.v_proj.len()],
            q_conv_weight: vec![0.0; w.q_conv_weight.len()],
            k_conv_weight: vec![0.0; w.k_conv_weight.len()],
            v_conv_weight: vec![0.0; w.v_conv_weight.len()],
            a_log: vec![0.0; w.a_log.len()],
            f_a_proj: vec![0.0; w.f_a_proj.len()],
            f_b_proj: vec![0.0; w.f_b_proj.len()],
            dt_bias: vec![0.0; w.dt_bias.len()],
            beta_proj: vec![0.0; w.beta_proj.len()],
            g_proj: vec![0.0; w.g_proj.len()],
            o_norm_weight: vec![0.0; w.o_norm_weight.len()],
            o_proj: vec![0.0; w.o_proj.len()],
        }
    }
}

// ─── Forward with saved activations ─────────────────────────────────────────

/// Run the KDA forward + capture all activations needed for the backward.
///
/// Produces the same output as [`kda_forward_token`]; additionally snapshots
/// all intermediates the backward needs. Returns `(output, saved)` where
/// `output` is an owned `Vec<f32>` (length `hidden_size`).
///
/// The output is owned (not a borrow) because building `saved` requires
/// immutable reads of the forward's scratch after it has been mutably
/// borrowed — a lifetime conflict an owned return resolves.
pub fn kda_forward_token_with_saved(
    config: &KdaConfig,
    weights: &KdaWeights,
    cache: &mut KdaLayerCache,
    scratch: &mut KdaForwardScratch,
    h: &[f32],
) -> (Vec<f32>, KdaSavedActivations) {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;
    let scale = config.q_scale();

    // Snapshot conv ring buffers + state BEFORE the forward mutates them.
    let conv_buf_q = cache.q_conv.buf.clone();
    let conv_buf_k = cache.k_conv.buf.clone();
    let conv_buf_v = cache.v_conv.buf.clone();
    let conv_buf_idx = cache.q_conv.buf_idx;
    let s_prev_per_head: Vec<Vec<f32>> = cache.heads.iter().map(|hd| hd.s.clone()).collect();

    // Run the stock forward (fills scratch + mutates cache in-place).
    let output_ref = kda_forward_token(config, weights, cache, scratch, h);
    let output: Vec<f32> = output_ref.to_vec();

    // Reconstruct per-head activations from the now-filled scratch + saved S_{t-1}.
    let mut head_acts = Vec::with_capacity(n_h);
    #[allow(clippy::needless_range_loop)]
    // head indexes s_prev_per_head + weights.a_log + cache.heads + scratch.beta + stride offset
    for head in 0..n_h {
        let off = head * dk;
        let s_prev = &s_prev_per_head[head];

        // q_normed: L2Norm(z_q_conv slice) then scale. Recompute from saved z_q_conv.
        let mut q_normed = vec![0.0f32; dk];
        q_normed.copy_from_slice(&scratch.z_q_conv[off..off + dk]);
        l2_normalize_eps_kda(&mut q_normed);
        for q in q_normed.iter_mut() {
            *q *= scale;
        }

        // k_normed: L2Norm(z_k_conv slice), no scale.
        let mut k_normed = vec![0.0f32; dk];
        k_normed.copy_from_slice(&scratch.z_k_conv[off..off + dk]);
        l2_normalize_eps_kda(&mut k_normed);

        // v: identity (slice of z_v_conv).
        let v = scratch.z_v_conv[off..off + dk].to_vec();

        // Decay derivation.
        let alpha_head = weights.a_log[head].exp();
        let mut gk = vec![0.0f32; dk];
        let mut g_plus = vec![0.0f32; dk];
        let mut a_decay = vec![0.0f32; dk];
        for i in 0..dk {
            let gp = scratch.g_raw[off + i] + weights.dt_bias[off + i];
            g_plus[i] = gp;
            let g = -alpha_head * softplus(gp);
            gk[i] = g;
            let a = g.exp();
            a_decay[i] = if a < config.alpha_eps {
                config.alpha_eps
            } else {
                a
            };
        }

        // S' = decay(S_{t-1}).
        let mut s_prime = vec![0.0f32; dk * dk];
        for i in 0..dk {
            for j in 0..dk {
                s_prime[i * dk + j] = s_prev[i * dk + j] * a_decay[i];
            }
        }

        // S'' = post-forward state (= cache.heads[head].s after the forward).
        let s_post = cache.heads[head].s.clone();

        // beta_h.
        let beta_h = scratch.beta[head];

        // r, delta.
        let mut r = vec![0.0f32; dk];
        for j in 0..dk {
            for i in 0..dk {
                r[j] += s_prime[i * dk + j] * beta_h * k_normed[i];
            }
        }
        let delta: Vec<f32> = (0..dk).map(|j| beta_h * v[j] - r[j]).collect();

        // out_raw: readout from S'' (before norm). = Σ_i S''[i,j] · q_normed[i].
        let out_raw: Vec<f32> = (0..dk)
            .map(|j| {
                let mut acc = 0.0f32;
                for i in 0..dk {
                    acc += s_post[i * dk + j] * q_normed[i];
                }
                acc
            })
            .collect();

        // inv_rms.
        let sum_sq = simd_sum_sq(&out_raw, dk);
        let mean_sq = sum_sq / dk as f32;
        let inv_rms = 1.0 / (mean_sq + config.rms_eps).sqrt();

        head_acts.push(KdaHeadActivations {
            s_prev: s_prev.clone(),
            s_prime,
            s_post,
            q_normed,
            k_normed,
            v,
            a_decay,
            gk,
            g_plus,
            delta,
            out_raw,
            alpha_head,
            beta_h,
            inv_rms,
        });
    }

    let saved = KdaSavedActivations {
        h: h.to_vec(),
        z_q: scratch.z_q.clone(),
        z_k: scratch.z_k.clone(),
        z_v: scratch.z_v.clone(),
        z_q_conv: scratch.z_q_conv.clone(),
        z_k_conv: scratch.z_k_conv.clone(),
        z_v_conv: scratch.z_v_conv.clone(),
        f_a_hidden: scratch.f_a_hidden.clone(),
        g_raw: scratch.g_raw.clone(),
        beta_pre: scratch.beta_pre.clone(),
        g_out: scratch.g_out.clone(),
        o_concat: scratch.o_concat.clone(),
        heads: head_acts,
        conv_buf_q,
        conv_buf_k,
        conv_buf_v,
        conv_buf_idx,
    };

    let _ = (proj, d);
    (output, saved)
}

// ─── Core backward (steps 5→2) ──────────────────────────────────────────────
//
// Extracted so both the single-token [`kda_backward_token`] and the multi-token
// [`kda_backward_sequence`] share the same recurrence/norm/gate backward math
// (DRY: the only divergence is in the conv backward — single-token propagates
// the k=0 tap only; multi-token distributes all K taps across past tokens).

/// Output of the core backward (steps 5→2).
///
/// These are the per-token intermediate gradients the caller needs to drive the
/// conv backward (step 1) + projection backward (step 0). The core itself
/// accumulates into `grads`: `o_proj`, `o_norm_weight`, `a_log`, `dt_bias`.
///
/// Core backward: steps 5 (output proj) → 4 (RMSNorm gate) → 3 (recurrence) →
/// 2 (gates + decay + L2Norm).
///
/// Accumulates into `grads`: `o_proj`, `o_norm_weight`, `a_log`, `dt_bias`.
/// Writes `dL/dS_{t-1}` into `ds_prev_out` (caller-allocated, `[n_heads][dk*dk]`).
/// Returns the intermediate gradients the conv + projection backward need.
#[derive(Clone)]
pub struct KdaCoreBackwardOutput {
    /// dL/dz_q_conv `[proj]` — grad w.r.t. post-conv post-SiLU q.
    pub dz_q_conv: Vec<f32>,
    /// dL/dz_k_conv `[proj]`.
    pub dz_k_conv: Vec<f32>,
    /// dL/dz_v_conv `[proj]`.
    pub dz_v_conv: Vec<f32>,
    /// dL/dg_out `[proj]` — grad w.r.t. the output gate g_out = W^g · h.
    pub dg_out_full: Vec<f32>,
    /// dL/dbeta_pre `[n_heads]` — grad w.r.t. the pre-sigmoid beta projection.
    pub dbeta_pre: Vec<f32>,
    /// dL/dg_raw `[proj]` — grad w.r.t. the raw gate g_raw = W^{f_b} · f_a_hidden.
    pub dg_raw: Vec<f32>,
}

/// Core backward: steps 5 (output proj) → 4 (RMSNorm gate) → 3 (recurrence) →
/// 2 (gates + decay + L2Norm).
///
/// Accumulates into `grads`: `o_proj`, `o_norm_weight`, `a_log`, `dt_bias`.
/// Writes `dL/dS_{t-1}` into `ds_prev_out` (caller-allocated, `[n_heads][dk*dk]`).
/// Returns the intermediate gradients the conv + projection backward need.
#[allow(clippy::too_many_arguments)]
pub fn kda_core_backward(
    config: &KdaConfig,
    weights: &KdaWeights,
    saved: &KdaSavedActivations,
    d_output: &[f32],
    ds_next: &[Vec<f32>],
    grads: &mut KdaGradients,
    ds_prev_out: &mut [Vec<f32>],
) -> KdaCoreBackwardOutput {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;
    let scale = config.q_scale();

    debug_assert_eq!(d_output.len(), d);
    debug_assert_eq!(ds_next.len(), n_h);
    debug_assert_eq!(ds_prev_out.len(), n_h);

    // ── Step 5 backward: output projection ─────────────────────────────────
    // output = W^o · o_concat.
    // grads.o_proj += outer(d_output, o_concat)
    // do_concat = W^oᵀ · d_output
    simd_outer_product_acc(&mut grads.o_proj, d_output, &saved.o_concat, d, proj);
    let mut do_concat = vec![0.0f32; proj];
    simd_transpose_matvec_into(&mut do_concat, &weights.o_proj, d_output, d, proj);

    // Per-head work buffers (reused across heads).
    let mut dout_raw = vec![0.0f32; dk];
    let mut ds_post = vec![0.0f32; dk * dk];
    let mut ds_prime = vec![0.0f32; dk * dk];
    let mut dq_normed = vec![0.0f32; dk];
    let mut dk_normed = vec![0.0f32; dk];
    let mut dv_h = vec![0.0f32; dk];
    let mut ddelta = vec![0.0f32; dk];
    let mut dr = vec![0.0f32; dk];
    let mut derase_b = vec![0.0f32; dk];
    let mut da_decay = vec![0.0f32; dk];
    let mut dgk = vec![0.0f32; dk];
    let mut dg_plus = vec![0.0f32; dk];
    let mut dz_q_conv = vec![0.0f32; proj];
    let mut dz_k_conv = vec![0.0f32; proj];
    let mut dz_v_conv = vec![0.0f32; proj];
    let mut dg_raw = vec![0.0f32; proj];
    let mut dbeta_pre = vec![0.0f32; n_h];
    let mut dg_out_full = vec![0.0f32; proj];

    for head in 0..n_h {
        let off = head * dk;
        let ha = &saved.heads[head];

        // ── Step 4 backward: FusedRMSNormGated ────────────────────────────
        let do_concat_h = &do_concat[off..off + dk];
        let inv_rms = ha.inv_rms;
        let gamma = &weights.o_norm_weight;
        let mut dy = vec![0.0f32; dk];
        for i in 0..dk {
            let sig_gout = sigmoid(saved.g_out[off + i]);
            let d_oc_i = do_concat_h[i];
            let y_i = ha.out_raw[i] * inv_rms;
            dy[i] = d_oc_i * gamma[i] * sig_gout;
            grads.o_norm_weight[i] += d_oc_i * y_i * sig_gout;
            let dg_out_i = d_oc_i * y_i * gamma[i] * sig_gout * (1.0 - sig_gout);
            dg_out_full[off + i] = dg_out_i;
        }
        // RMSNorm backward: dL/dout_raw_j = dy_j·inv_rms − out_raw_j · Σ_i(dy_i·out_raw_i) / (dk · rms³)
        //
        // f64 guard: the product dy[i] · out_raw[i] can overflow f32 when the
        // upstream gradient has been amplified across multiple KDA layers AND
        // the forward state values (→ out_raw) are large. The individual
        // products fit in f64, and the subsequent correction term partially
        // cancels the direct term — f64 preserves that cancellation. Computing
        // in f32 overflows the sum to inf, making correction = inf, and
        // dout_raw = finite − inf = ∓inf, which cascades to NaN through
        // +inf + (−inf) in the state-matrix readout backward (line ~443).
        // See: katgpt-rs/.issues/ NaN-investigation (seq 31/34 backward pass).
        let inv_rms_f64 = inv_rms as f64;
        let rms_cubed_f64 = inv_rms_f64.powi(-3); // = (1/inv_rms)^3 = 1/inv_rms^3
        let sum_dy_out_f64: f64 = (0..dk)
            .map(|i| (dy[i] as f64) * (ha.out_raw[i] as f64))
            .sum();
        let correction_f64 = sum_dy_out_f64 / (dk as f64 * rms_cubed_f64);
        // Clamp threshold for f32-safe cast: f32::MAX / 10 to leave room for
        // downstream multiplications without overflow. The clamp preserves
        // gradient direction; only the magnitude is bounded.
        const DOUT_RAW_CLAMP: f64 = 1e35;
        for j in 0..dk {
            let direct = (dy[j] as f64) * inv_rms_f64;
            let corr = (ha.out_raw[j] as f64) * correction_f64;
            let val = direct - corr;
            let clamped = if !val.is_finite() {
                0.0
            } else if val.abs() > DOUT_RAW_CLAMP {
                val.signum() * DOUT_RAW_CLAMP
            } else {
                val
            };
            dout_raw[j] = clamped as f32;
        }

        // ── Step 3 backward: gdn2_recurrent_step ──────────────────────────
        // ds_post = ds_next (incoming from future) + readout contribution.
        ds_post[..dk * dk].copy_from_slice(&ds_next[head][..dk * dk]);
        // Readout: out[j] = Σ_i S''[i,j] · q_h[i]
        // f64 accumulation for dq_i: dout_raw × s_post products are the primary
        // cross-layer amplification path; f64 prevents overflow when both are
        // large (deep backward + large forward state).
        for i in 0..dk {
            let qi = ha.q_normed[i];
            let mut dq_i_f64 = 0.0f64;
            for j in 0..dk {
                // f64 for the accumulation, then clamp to f32-safe range.
                let ds_contrib = (dout_raw[j] as f64) * (qi as f64);
                let ds_prev_val = ds_post[i * dk + j] as f64;
                let ds_new = ds_prev_val + ds_contrib;
                ds_post[i * dk + j] = if !ds_new.is_finite() {
                    0.0
                } else if ds_new.abs() > 1e35 {
                    ds_new.signum() as f32 * 1e35
                } else {
                    ds_new as f32
                };
                dq_i_f64 += (dout_raw[j] as f64) * (ha.s_post[i * dk + j] as f64);
            }
            let dq = if !dq_i_f64.is_finite() {
                0.0
            } else if dq_i_f64.abs() > 1e35 {
                dq_i_f64.signum() * 1e35
            } else {
                dq_i_f64
            };
            dq_normed[i] = dq as f32;
        }
        // Update: S''[i,j] = S'[i,j] + k_h[i] · delta[j]
        for i in 0..dk {
            let mut dk_i_f64 = 0.0f64;
            for j in 0..dk {
                ds_prime[i * dk + j] = ds_post[i * dk + j]; // passthrough
                dk_i_f64 += (ds_post[i * dk + j] as f64) * (ha.delta[j] as f64);
            }
            dk_normed[i] = clamp_f64_to_f32(dk_i_f64);
        }
        for j in 0..dk {
            let mut dd_j_f64 = 0.0f64;
            for i in 0..dk {
                dd_j_f64 += (ds_post[i * dk + j] as f64) * (ha.k_normed[i] as f64);
            }
            ddelta[j] = clamp_f64_to_f32(dd_j_f64);
        }
        // delta: delta[j] = beta_h · v_h[j] − r[j]
        let mut dbeta_from_delta_f64 = 0.0f64;
        for j in 0..dk {
            dv_h[j] = clamp_f64_to_f32(ddelta[j] as f64 * ha.beta_h as f64);
            dr[j] = clamp_f64_to_f32(-(ddelta[j] as f64));
            dbeta_from_delta_f64 += (ddelta[j] as f64) * (ha.v[j] as f64);
        }
        // Read: r[j] = Σ_i S'[i,j] · beta_h · k_h[i]  (erase_b = beta_h broadcast)
        let mut dbeta_from_read_f64 = 0.0f64;
        for i in 0..dk {
            let mut dk_from_read_f64 = 0.0f64;
            for j in 0..dk {
                let dr_j = dr[j] as f64;
                let beta = ha.beta_h as f64;
                let kn = ha.k_normed[i] as f64;
                let sp = ha.s_prime[i * dk + j] as f64;
                // ds_prime update
                let contrib = dr_j * beta * kn;
                let curr = ds_prime[i * dk + j] as f64;
                ds_prime[i * dk + j] = clamp_f64_to_f32(curr + contrib);
                // derase_b accumulation
                let de = dr_j * sp * kn;
                let curr_er = derase_b[i] as f64;
                derase_b[i] = clamp_f64_to_f32(curr_er + de);
                // dk_from_read
                dk_from_read_f64 += dr_j * sp * beta;
                // dbeta_from_read
                dbeta_from_read_f64 += dr_j * sp * kn;
            }
            dk_normed[i] = clamp_f64_to_f32(dk_normed[i] as f64 + dk_from_read_f64);
        }
        // Decay: S'[i,j] = S_{t-1}[i,j] · a[i]
        // f64 accumulation: ds_prime (BPTT-accumulated) × s_prev (forward state)
        // products can overflow f32 in deep multi-layer backprop. See comment
        // at the RMSNorm backward above for the overflow mechanism.
        for i in 0..dk {
            let mut da_i_f64 = 0.0f64;
            for j in 0..dk {
                ds_prev_out[head][i * dk + j] =
                    clamp_f64_to_f32((ds_prime[i * dk + j] as f64) * (ha.a_decay[i] as f64));
                da_i_f64 += (ds_prime[i * dk + j] as f64) * (ha.s_prev[i * dk + j] as f64);
            }
            da_decay[i] = clamp_f64_to_f32(da_i_f64);
        }

        // ── Step 2 backward: gates + decay derivation ─────────────────────
        // dbeta_h total = dbeta_from_delta + dbeta_from_read
        //   (= dbeta_from_delta + Σ_i derase_b[i], since erase_b[i] = beta_h).
        let dbeta_h_total = clamp_f64_to_f32(dbeta_from_delta_f64 + dbeta_from_read_f64);
        dbeta_pre[head] =
            clamp_f64_to_f32(dbeta_h_total as f64 * ha.beta_h as f64 * (1.0 - ha.beta_h as f64));

        // a[i] = max(exp(gk[i]), alpha_eps) — relu-clamp
        let mut dalpha_head_f64 = 0.0f64;
        for i in 0..dk {
            let exp_gk = ha.gk[i].exp();
            let mask = if ha.a_decay[i] > config.alpha_eps {
                1.0
            } else {
                0.0
            };
            dgk[i] = clamp_f64_to_f32((da_decay[i] as f64) * (exp_gk as f64) * mask);
            dalpha_head_f64 += (dgk[i] as f64) * (-softplus(ha.g_plus[i]) as f64);
            dg_plus[i] = clamp_f64_to_f32(
                (dgk[i] as f64) * (-(ha.alpha_head as f64) * sigmoid(ha.g_plus[i]) as f64),
            );
        }
        grads.a_log[head] += clamp_f64_to_f32(dalpha_head_f64 * (ha.alpha_head as f64));

        // g_plus → g_raw + dt_bias
        // NOTE: NOT a pure copy — this loop does copy (dg_raw) AND accumulate
        // (grads.dt_bias +=). clippy::manual_memcpy suggests copy_from_slice,
        // which would silently drop the dt_bias gradient. Do NOT "fix".
        #[allow(clippy::manual_memcpy)]
        for i in 0..dk {
            dg_raw[off + i] = dg_plus[i];
            grads.dt_bias[off + i] += dg_plus[i];
        }

        // ── L2Norm backward for q and k ───────────────────────────────────
        // Forward (l2_normalize_eps_kda): inv = 1/(||x|| + 1e-6), x_norm = x · inv.
        // q also scales by `scale` after norm.
        // dL/dx_j = inv · (dL/dy_j − x_j · inv² · Σ_i dL/dy_i · x_i)
        // For q: dL/dy_j (unscaled) = dq_normed[j] · scale (backward through scale).

        // q path:
        let z_q_slice = &saved.z_q_conv[off..off + dk];
        let norm_q = l2_norm_value(z_q_slice);
        let inv_q = 1.0 / (norm_q + 1e-6);
        let inv_q_sq = inv_q * inv_q;
        let mut sum_dq_z = 0.0f32;
        for i in 0..dk {
            sum_dq_z += dq_normed[i] * scale * z_q_slice[i];
        }
        for i in 0..dk {
            let d_unscaled = dq_normed[i] * scale;
            dz_q_conv[off + i] = inv_q * (d_unscaled - z_q_slice[i] * sum_dq_z * inv_q_sq);
        }

        // k path (no scale):
        let z_k_slice = &saved.z_k_conv[off..off + dk];
        let norm_k = l2_norm_value(z_k_slice);
        let inv_k = 1.0 / (norm_k + 1e-6);
        let inv_k_sq = inv_k * inv_k;
        let mut sum_dk_z = 0.0f32;
        for i in 0..dk {
            sum_dk_z += dk_normed[i] * z_k_slice[i];
        }
        for i in 0..dk {
            dz_k_conv[off + i] = inv_k * (dk_normed[i] - z_k_slice[i] * sum_dk_z * inv_k_sq);
        }

        // v path (identity):
        dz_v_conv[off..off + dk].copy_from_slice(&dv_h[..dk]);
    }

    KdaCoreBackwardOutput {
        dz_q_conv,
        dz_k_conv,
        dz_v_conv,
        dg_out_full,
        dbeta_pre,
        dg_raw,
    }
}

// ─── Backward ───────────────────────────────────────────────────────────────

/// Analytic backward of `kda_forward_token` for a single token.
///
/// Consumes the [`KdaSavedActivations`] (from [`kda_forward_token_with_saved`])
/// + upstream `d_output` (= dL/doutput, `[hidden_size]`) + incoming state grad
///   `ds_next` (= dL/dS_t from the future BPTT step, `[n_heads][dk*dk]`).
///
/// Produces:
/// - `grads` — weight gradients (accumulated `+=`).
/// - `dh_out` — dL/dh (`[hidden_size]`, **overwritten**).
/// - `ds_prev_out` — dL/dS_{t-1} (`[n_heads][dk*dk]`, **overwritten**).
///
/// The caller threads `ds_prev_out` from step t into `ds_next` of step t-1 for
/// BPTT. The conv backward propagates ONLY the k=0 tap (current-token
/// contribution) — for multi-token sequences with conv cross-token propagation,
/// use [`kda_backward_sequence`] instead.
#[allow(clippy::too_many_arguments)]
pub fn kda_backward_token(
    config: &KdaConfig,
    weights: &KdaWeights,
    saved: &KdaSavedActivations,
    d_output: &[f32],
    dh_out: &mut [f32],
    grads: &mut KdaGradients,
    ds_next: &[Vec<f32>],
    ds_prev_out: &mut [Vec<f32>],
) {
    let d = config.hidden_size;
    let dk = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;

    debug_assert_eq!(d_output.len(), d);
    debug_assert_eq!(dh_out.len(), d);

    dh_out.fill(0.0);

    // ── Steps 5→2: core backward (recurrence + norm + gates). ─────────────
    let core = kda_core_backward(
        config,
        weights,
        saved,
        d_output,
        ds_next,
        grads,
        ds_prev_out,
    );

    // ── Step 1 backward: ShortConv + SiLU (k=0 tap only — single-token scope).
    // Forward: z_q_conv[c] = silu( Σ_k weight[c,k] · buf_q[c, slot(k)] )
    // We recover pre-SiLU via Newton inversion, then propagate the k=0 tap.
    //
    // IMPORTANT: the saved conv_buf_* snapshots were taken BEFORE the forward
    // wrote the current input. The conv forward reads the buffer AFTER writing
    // the current input at buf_idx. We must patch the buffer: write the current
    // input (saved.z_q for q) at conv_buf_idx before using it for weight grads.
    let mut conv_buf_q_patched = saved.conv_buf_q.clone();
    let mut conv_buf_k_patched = saved.conv_buf_k.clone();
    let mut conv_buf_v_patched = saved.conv_buf_v.clone();
    let ks = config.conv_kernel_size;
    for c in 0..proj {
        conv_buf_q_patched[c * ks + saved.conv_buf_idx] = saved.z_q[c];
        conv_buf_k_patched[c * ks + saved.conv_buf_idx] = saved.z_k[c];
        conv_buf_v_patched[c * ks + saved.conv_buf_idx] = saved.z_v[c];
    }

    let mut dz_q_conv = core.dz_q_conv;
    let mut dz_k_conv = core.dz_k_conv;
    let mut dz_v_conv = core.dz_v_conv;
    backward_conv_silu(
        &saved.z_q_conv,
        &mut dz_q_conv,
        &conv_buf_q_patched,
        weights.q_conv_weight.as_slice(),
        saved.conv_buf_idx,
        config.conv_kernel_size,
        proj,
        &mut grads.q_conv_weight,
    );
    backward_conv_silu(
        &saved.z_k_conv,
        &mut dz_k_conv,
        &conv_buf_k_patched,
        weights.k_conv_weight.as_slice(),
        saved.conv_buf_idx,
        config.conv_kernel_size,
        proj,
        &mut grads.k_conv_weight,
    );
    backward_conv_silu(
        &saved.z_v_conv,
        &mut dz_v_conv,
        &conv_buf_v_patched,
        weights.v_conv_weight.as_slice(),
        saved.conv_buf_idx,
        config.conv_kernel_size,
        proj,
        &mut grads.v_conv_weight,
    );

    // ── Step 0 backward: projections ──────────────────────────────────────
    // Each: y = W · h.  dL/dW += outer(dL/dy, h).  dL/dh += Wᵀ · dL/dy.
    proj_backward(
        &mut grads.q_proj,
        dh_out,
        &dz_q_conv,
        &saved.h,
        &weights.q_proj,
        proj,
        d,
    );
    proj_backward(
        &mut grads.k_proj,
        dh_out,
        &dz_k_conv,
        &saved.h,
        &weights.k_proj,
        proj,
        d,
    );
    proj_backward(
        &mut grads.v_proj,
        dh_out,
        &dz_v_conv,
        &saved.h,
        &weights.v_proj,
        proj,
        d,
    );
    proj_backward(
        &mut grads.g_proj,
        dh_out,
        &core.dg_out_full,
        &saved.h,
        &weights.g_proj,
        proj,
        d,
    );
    proj_backward(
        &mut grads.beta_proj,
        dh_out,
        &core.dbeta_pre,
        &saved.h,
        &weights.beta_proj,
        n_h,
        d,
    );

    // Two-stage gate: f_a_hid = W^{f_a} · h, then g_raw = W^{f_b} · f_a_hid.
    let mut df_a_hidden = vec![0.0f32; dk];
    proj_backward(
        &mut grads.f_b_proj,
        &mut df_a_hidden,
        &core.dg_raw,
        &saved.f_a_hidden,
        &weights.f_b_proj,
        proj,
        dk,
    );
    proj_backward(
        &mut grads.f_a_proj,
        dh_out,
        &df_a_hidden,
        &saved.h,
        &weights.f_a_proj,
        dk,
        d,
    );
}

// ─── Multi-token backward (Plan 318 Phase C C5) ─────────────────────────────

/// Analytic backward over a sequence of `L` tokens with conv-ring cross-token
/// gradient propagation.
///
/// This is the full BPTT loop that composes the per-token core backward
/// ([`kda_core_backward`]) with cross-token conv distribution. Unlike calling
/// [`kda_backward_token`] in a manual reverse loop, this propagates the k>0
/// ShortConv taps — past-token contributions through the conv ring buffer —
/// which the single-token primitive intentionally omits (Issue 389 T4 scope).
///
/// # Algorithm
///
/// Forward (per token t): `conv_out_t[c] = Σ_{k=0}^{K-1} weight[c,k] · z_{t-k}[c]`.
/// So `dL/dz_{t'}[c] = Σ_{t : t'∈[t-K+1,t]} dL/d(conv_out_t[c]) · weight[c, t-t']`.
///
/// Process tokens in reverse (t = L-1 .. 0). When processing token t:
/// 1. Core backward → `dz_q_conv_t`, etc. + `ds_prev_t` + non-conv/non-proj grads.
/// 2. SiLU backward → `dconv_out_t`.
/// 3. Distribute conv: for each tap k, add `dconv_out_t[c] · weight[c,k]` to
///    `dz_accum[t-k][c]` (skip if `t-k < 0` — the ring buffer started at zeros),
///    and accumulate `grads.conv_weight[c,k] += dconv_out_t[c] · saved[t-k].z[c]`.
/// 4. `dz_accum[t]` is now complete (all contributions from tokens t, t+1, ...,
///    min(t+K-1, L-1) have arrived) → run the projection backward for token t.
/// 5. Thread `ds_prev_t` → `ds_next` for the next (t-1) iteration.
///
/// # Arguments
/// * `saved_tokens` — per-token saved activations (from a forward pass over the
///   sequence with a freshly-reset cache).
/// * `d_outputs` — per-token upstream gradients `dL/doutput[t]`, `[hidden_size]` each.
/// * `dh_outs` — per-token output gradients `dL/dh[t]`, `[hidden_size]` each
///   (**overwritten**).
/// * `grads` — weight gradients (accumulated `+=` across all tokens).
pub fn kda_backward_sequence(
    config: &KdaConfig,
    weights: &KdaWeights,
    saved_tokens: &[KdaSavedActivations],
    d_outputs: &[Vec<f32>],
    dh_outs: &mut [Vec<f32>],
    grads: &mut KdaGradients,
) {
    let l = saved_tokens.len();
    debug_assert_eq!(d_outputs.len(), l);
    debug_assert_eq!(dh_outs.len(), l);
    if l == 0 {
        return;
    }

    let dk = config.head_dim;
    let n_h = config.n_heads;
    let proj = dk * n_h;
    let d = config.hidden_size;
    let ks = config.conv_kernel_size;
    let state_sz = dk * dk;

    // Cross-token conv accumulators for dz_q, dz_k, dz_v (pre-conv projection outputs).
    let mut dz_q_accum: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; proj]).collect();
    let mut dz_k_accum: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; proj]).collect();
    let mut dz_v_accum: Vec<Vec<f32>> = (0..l).map(|_| vec![0.0f32; proj]).collect();

    // BPTT state gradient. Init to zeros at t=L (no future contribution).
    let mut ds_next: Vec<Vec<f32>> = (0..n_h).map(|_| vec![0.0f32; state_sz]).collect();
    let mut ds_prev: Vec<Vec<f32>> = (0..n_h).map(|_| vec![0.0f32; state_sz]).collect();

    for t in (0..l).rev() {
        // ── Steps 5→2: core backward for token t. ─────────────────────────
        let core = kda_core_backward(
            config,
            weights,
            &saved_tokens[t],
            &d_outputs[t],
            &ds_next,
            grads,
            &mut ds_prev,
        );

        // ── Step 1: SiLU backward → dconv_out, then conv distribution. ─────
        // SiLU backward: dL/d(conv_out) = dL/dz_conv · silu'(pre_silu)
        // where pre_silu is recovered via Newton inversion from the saved
        // post-SiLU value (same as backward_conv_silu phase 1).
        let mut dconv_out_q = vec![0.0f32; proj];
        let mut dconv_out_k = vec![0.0f32; proj];
        let mut dconv_out_v = vec![0.0f32; proj];
        for c in 0..proj {
            let y_q = saved_tokens[t].z_q_conv[c];
            let x_q = silu_inverse(y_q);
            let sig_q = sigmoid(x_q);
            dconv_out_q[c] = core.dz_q_conv[c] * sig_q * (1.0 + x_q * (1.0 - sig_q));

            let y_k = saved_tokens[t].z_k_conv[c];
            let x_k = silu_inverse(y_k);
            let sig_k = sigmoid(x_k);
            dconv_out_k[c] = core.dz_k_conv[c] * sig_k * (1.0 + x_k * (1.0 - sig_k));

            let y_v = saved_tokens[t].z_v_conv[c];
            let x_v = silu_inverse(y_v);
            let sig_v = sigmoid(x_v);
            dconv_out_v[c] = core.dz_v_conv[c] * sig_v * (1.0 + x_v * (1.0 - sig_v));
        }

        // Conv distribution: for each tap k, contribute to token t-k.
        // Forward: conv_out_t[c] = Σ_k weight[c,k] · z_{t-k}[c].
        // Backward: dz_{t-k}[c] += dconv_out_t[c] · weight[c,k]
        //           dweight[c,k] += dconv_out_t[c] · z_{t-k}[c]
        for k in 0..ks {
            let t_past = t as isize - k as isize;
            if t_past < 0 {
                break; // Ring buffer was zero before the sequence start; those
                // contributions are zero (z=0 → dweight term = 0; and
                // there's no dz accumulator for t < 0).
            }
            let tp = t_past as usize;
            let z_q_tp = &saved_tokens[tp].z_q;
            let z_k_tp = &saved_tokens[tp].z_k;
            let z_v_tp = &saved_tokens[tp].z_v;
            let dz_q_tp = &mut dz_q_accum[tp];
            let dz_k_tp = &mut dz_k_accum[tp];
            let dz_v_tp = &mut dz_v_accum[tp];
            for c in 0..proj {
                let w_off = c * ks + k;
                let wq = weights.q_conv_weight[w_off];
                let wk = weights.k_conv_weight[w_off];
                let wv = weights.v_conv_weight[w_off];
                let dq = dconv_out_q[c];
                let dk_val = dconv_out_k[c];
                let dv = dconv_out_v[c];
                dz_q_tp[c] += dq * wq;
                dz_k_tp[c] += dk_val * wk;
                dz_v_tp[c] += dv * wv;
                grads.q_conv_weight[w_off] += dq * z_q_tp[c];
                grads.k_conv_weight[w_off] += dk_val * z_k_tp[c];
                grads.v_conv_weight[w_off] += dv * z_v_tp[c];
            }
        }

        // ── Step 0: projection backward for token t (dz_*_accum[t] is complete).
        // Each: y = W · h.  dL/dW += outer(dL/dy, h).  dL/dh += Wᵀ · dL/dy.
        let dh_t = &mut dh_outs[t];
        dh_t.fill(0.0);
        proj_backward(
            &mut grads.q_proj,
            dh_t,
            &dz_q_accum[t],
            &saved_tokens[t].h,
            &weights.q_proj,
            proj,
            d,
        );
        proj_backward(
            &mut grads.k_proj,
            dh_t,
            &dz_k_accum[t],
            &saved_tokens[t].h,
            &weights.k_proj,
            proj,
            d,
        );
        proj_backward(
            &mut grads.v_proj,
            dh_t,
            &dz_v_accum[t],
            &saved_tokens[t].h,
            &weights.v_proj,
            proj,
            d,
        );
        proj_backward(
            &mut grads.g_proj,
            dh_t,
            &core.dg_out_full,
            &saved_tokens[t].h,
            &weights.g_proj,
            proj,
            d,
        );
        proj_backward(
            &mut grads.beta_proj,
            dh_t,
            &core.dbeta_pre,
            &saved_tokens[t].h,
            &weights.beta_proj,
            n_h,
            d,
        );

        // Two-stage gate: f_a_hid = W^{f_a} · h, then g_raw = W^{f_b} · f_a_hid.
        let mut df_a_hidden = vec![0.0f32; dk];
        proj_backward(
            &mut grads.f_b_proj,
            &mut df_a_hidden,
            &core.dg_raw,
            &saved_tokens[t].f_a_hidden,
            &weights.f_b_proj,
            proj,
            dk,
        );
        proj_backward(
            &mut grads.f_a_proj,
            dh_t,
            &df_a_hidden,
            &saved_tokens[t].h,
            &weights.f_a_proj,
            dk,
            d,
        );

        // ── Thread ds_prev → ds_next for the next (t-1) iteration. ─────────
        core::mem::swap(&mut ds_next, &mut ds_prev);
    }
}

// ─── Backward helpers ───────────────────────────────────────────────────────

/** Backward through SiLU + ShortConv1D for one of q/k/v.
 *
 * Given `dz_conv` (grad w.r.t. post-SiLU output) + the saved post-SiLU values
 * `z_conv_saved`, recovers the pre-SiLU conv output (via Newton inversion of
 * silu), then:
 * - accumulates `dL/dconv_weight` into `grad_conv_weight`.
 * - writes `dL/dz_preconv` (current-token contribution only) into `dz_conv`
 *   (overwriting the input — the caller reuses it as the projection-backward
 *   input).
 */
#[allow(clippy::too_many_arguments)]
pub fn backward_conv_silu(
    z_conv_saved: &[f32], // post-SiLU values [proj]
    dz_conv: &mut [f32],  // IN: grad w.r.t. post-SiLU; OUT: dL/dz_preconv [proj]
    conv_buf: &[f32],     // ring buffer snapshot (pre-forward) [proj*ks]
    conv_weight: &[f32],  // [proj*ks]
    conv_buf_idx: usize,  // ring index (pre-forward)
    kernel_size: usize,
    n_channels: usize,
    grad_conv_weight: &mut [f32], // accumulated
) {
    let ks = kernel_size;
    let newest_slot = conv_buf_idx; // the forward wrote the current input here

    // Phase 1: read dz_conv + compute dconv_out (SiLU backward).
    // We must finish ALL reads before any writes since dz_conv is &mut.
    let mut dconv_out = vec![0.0f32; n_channels];
    for c in 0..n_channels {
        let y = z_conv_saved[c]; // post-SiLU
        let x = silu_inverse(y); // pre-SiLU (= conv output)
        let sig = sigmoid(x);
        let silu_deriv = sig * (1.0 + x * (1.0 - sig));
        dconv_out[c] = dz_conv[c] * silu_deriv;
    }

    // Phase 2: ShortConv backward + write dL/dz_preconv into dz_conv.
    // Forward: conv_out[c] = Σ_k weight[c,k] · buf[c, slot(k)]
    //   where slot(k) = (newest_slot + ks − k) % ks.
    // dL/dweight[c,k] += dconv_out[c] · buf[c, slot(k)]
    // dL/d(current input)[c] = dconv_out[c] · weight[c, 0]   (k=0 = current token)
    let mut dz_pre = vec![0.0f32; n_channels];
    for c in 0..n_channels {
        let w_off = c * ks;
        let b_off = c * ks;
        let dc = dconv_out[c];
        for k in 0..ks {
            let slot = (newest_slot + ks - k) % ks;
            grad_conv_weight[w_off + k] += dc * conv_buf[b_off + slot];
        }
        dz_pre[c] = dc * conv_weight[w_off]; // k=0 tap (current token)
    }

    // Overwrite dz_conv with the pre-conv grad.
    dz_conv.copy_from_slice(&dz_pre);
}

/// Backward through a linear projection `y = W · h` (simd_matmul_rows convention).
/// `grad_W[r,c] += grad_y[r] · h[c]`. `grad_h[c] += Σ_r W[r,c] · grad_y[r]`.
fn proj_backward(
    grad_w: &mut [f32],
    grad_h: &mut [f32],
    grad_y: &[f32],
    h: &[f32],
    w: &[f32],
    rows: usize,
    cols: usize,
) {
    simd_outer_product_acc(grad_w, grad_y, h, rows, cols);
    simd_transpose_matvec_acc(grad_h, w, grad_y, rows, cols);
}

/// L2-normalize with eps outside the sqrt (mirrors kda_forward.rs).
fn l2_normalize_eps_kda(x: &mut [f32]) {
    let norm_sq = simd_sum_sq(x, x.len());
    let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-6);
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

/// Compute `sqrt(Σ x_i²)` (no eps).
fn l2_norm_value(x: &[f32]) -> f32 {
    simd_sum_sq(x, x.len()).sqrt()
}

// ─── Math helpers (mirror kda_forward.rs) ───────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    crate::gdn2::kernel::sigmoid(x)
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x >= 20.0 {
        x
    } else if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Inverse SiLU via Newton's method: solve `silu(x) = x · sigmoid(x) = y` for x.
/// Converges in ~5-8 iterations for |y| < ~10.
fn silu_inverse(y: f32) -> f32 {
    let mut x = y; // good initial guess for small |y|; silu(x) ≈ x for large x.
    for _ in 0..10 {
        let sig = sigmoid(x);
        let f = x * sig - y;
        let fp = sig * (1.0 + x * (1.0 - sig));
        if fp.abs() < 1e-12 {
            break;
        }
        let delta = f / fp;
        x -= delta;
        if delta.abs() < 1e-8 {
            break;
        }
        if !x.is_finite() {
            return y;
        }
    }
    x
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn2::kda_forward::{KdaForwardScratch, KdaLayerCache};

    fn small_config() -> KdaConfig {
        KdaConfig {
            head_dim: 8,
            n_heads: 2,
            hidden_size: 16,
            conv_kernel_size: 4,
            alpha_eps: 1e-5,
            rms_eps: 1e-5,
        }
    }

    /// Smoke test: forward-with-saved produces the same output as the stock forward.
    #[test]
    fn forward_with_saved_matches_stock() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 42);
        let h: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();

        let mut cache1 = KdaLayerCache::new(&config);
        let mut scratch1 = KdaForwardScratch::new(&config);
        let out1 = kda_forward_token(&config, &weights, &mut cache1, &mut scratch1, &h);

        let mut cache2 = KdaLayerCache::new(&config);
        let mut scratch2 = KdaForwardScratch::new(&config);
        let (out2, _saved) =
            kda_forward_token_with_saved(&config, &weights, &mut cache2, &mut scratch2, &h);

        for i in 0..config.hidden_size {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-5,
                "output mismatch at {}: {} vs {}",
                i,
                out1[i],
                out2[i]
            );
        }
    }

    /// Backward smoke test: runs without panicking, produces finite gradients.
    #[test]
    fn backward_smoke_finite() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 42);
        let h: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();

        let mut cache = KdaLayerCache::new(&config);
        let mut fwd_scratch = KdaForwardScratch::new(&config);
        let (_output, saved) =
            kda_forward_token_with_saved(&config, &weights, &mut cache, &mut fwd_scratch, &h);

        let d_output = vec![0.1f32; config.hidden_size];
        let mut dh = vec![0.0f32; config.hidden_size];
        let mut grads = KdaGradients::zeros_like(&weights);
        let ds_next: Vec<Vec<f32>> = (0..config.n_heads)
            .map(|_| vec![0.0; config.head_dim * config.head_dim])
            .collect();
        let mut ds_prev: Vec<Vec<f32>> = (0..config.n_heads)
            .map(|_| vec![0.0; config.head_dim * config.head_dim])
            .collect();

        kda_backward_token(
            &config,
            &weights,
            &saved,
            &d_output,
            &mut dh,
            &mut grads,
            &ds_next,
            &mut ds_prev,
        );

        for &g in &dh {
            assert!(g.is_finite(), "non-finite dh: {g}");
        }
        for &g in &grads.a_log {
            assert!(g.is_finite(), "non-finite a_log grad: {g}");
        }
        for &g in grads.q_proj.iter().take(20) {
            assert!(g.is_finite(), "non-finite q_proj grad: {g}");
        }
    }

    /// Verify S'' reconstruction: s_prime + k⊗delta should equal s_post.
    #[test]
    fn saved_activations_s_post_consistent() {
        let config = small_config();
        let weights = KdaWeights::random(&config, 42);
        let h: Vec<f32> = (0..config.hidden_size)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();

        let mut cache = KdaLayerCache::new(&config);
        let mut fwd_scratch = KdaForwardScratch::new(&config);
        let (_output, saved) =
            kda_forward_token_with_saved(&config, &weights, &mut cache, &mut fwd_scratch, &h);

        let dk = config.head_dim;
        for (head, ha) in saved.heads[..config.n_heads].iter().enumerate() {
            for i in 0..dk {
                for j in 0..dk {
                    let reconstructed = ha.s_prime[i * dk + j] + ha.k_normed[i] * ha.delta[j];
                    let actual = ha.s_post[i * dk + j];
                    assert!(
                        (reconstructed - actual).abs() < 1e-4,
                        "head {head} S'' mismatch at [{i},{j}]: reconstructed {reconstructed} vs actual {actual}"
                    );
                }
            }
        }
    }
}
