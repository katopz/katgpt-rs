//! GDN2 (Gated DeltaNet-2) recurrent step kernel.
//!
//! Core O(d_k × d_v) recurrent step implementing decoupled erase/write gates.
//! The four-step recurrence per token:
//!
//! ```text
//! 1. Decay:  S *= Diag(α)        — row-wise scale by per-channel decay
//! 2. Read:   r = Sᵀ(b ⊙ k)      — gated matvec (erase gate modulates key)
//! 3. Update: S += k ⊗ (w⊙v − r) — outer product delta (write gate modulates value)
//! 4. Readout: o = Sᵀ q           — query the updated state
//! ```
//!
//! Reference: Research 70 — Gated DeltaNet-2: O(1) Decode with Decoupled Erase/Write Gates.

use super::types::Gdn2GateConfig;
use crate::simd::{simd_outer_product_acc, simd_scale_inplace};

/// Core GDN2 recurrent step: O(d_k × d_v) per token per head.
///
/// Updates state S in-place and writes output to `out`.
///
/// # Arguments
/// * `k` — Key vector `[dk]`, should be L2-normalized for stability
/// * `v` — Value vector `[dv]`
/// * `q` — Query vector `[dk]`, should be L2-normalized for stability
/// * `s` — State matrix `[dk × dv]`, updated in-place (row-major)
/// * `alpha` — Per-channel decay `[dk]`, values typically in (0, 1]
/// * `b` — Erase gate `[dk]`, values in [0, 1]
/// * `w_val` — Scalar write weight for `EraseOnly`/`Kda` modes
/// * `w` — Channel-wise write gate `[dv]` for `Full` mode, values in [0, 1]
/// * `out` — Output buffer `[dv]`, written to
/// * `temp` — Temporary buffer `[dv]`, used internally for the read step
/// * `dk` — Key/query dimension (head_dim)
/// * `dv` — Value dimension (head_dim)
/// * `gate_config` — Which gate variant to use
#[allow(clippy::too_many_arguments)]
pub fn gdn2_recurrent_step(
    k: &[f32],
    v: &[f32],
    q: &[f32],
    s: &mut [f32],
    alpha: &[f32],
    b: &[f32],
    w_val: f32,
    w: &[f32],
    out: &mut [f32],
    temp: &mut [f32],
    dk: usize,
    dv: usize,
    gate_config: Gdn2GateConfig,
) {
    debug_assert_eq!(k.len(), dk);
    debug_assert_eq!(v.len(), dv);
    debug_assert_eq!(q.len(), dk);
    debug_assert_eq!(s.len(), dk * dv);
    debug_assert_eq!(alpha.len(), dk);
    debug_assert_eq!(b.len(), dk);
    debug_assert!(w.len() >= dv || gate_config != Gdn2GateConfig::Full);
    debug_assert_eq!(out.len(), dv);
    debug_assert_eq!(temp.len(), dv);

    // Step 1: Decay S *= Diag(alpha) — row-wise scale
    for (row, &alpha_row) in alpha.iter().enumerate() {
        let row_start = row * dv;
        simd_scale_inplace(&mut s[row_start..row_start + dv], alpha_row);
    }

    // Step 2: Read r = Sᵀ(b ⊙ k) — gated matvec
    // Compute b_k[i] = b[i] * k[i], then r[j] = Σ_i S[i*dv+j] * b_k[i]
    temp.fill(0.0);
    for i in 0..dk {
        let bk_i = b[i] * k[i];
        if bk_i == 0.0 {
            continue;
        }
        let row_start = i * dv;
        for j in 0..dv {
            unsafe {
                *temp.get_unchecked_mut(j) += *s.get_unchecked(row_start + j) * bk_i;
            }
        }
    }

    // Step 3: Update S += k ⊗ (w⊙v − r)
    // Compute delta = w⊙v − r, then outer product accumulate
    let mut delta = vec![0.0f32; dv];
    match gate_config {
        Gdn2GateConfig::EraseOnly | Gdn2GateConfig::Kda => {
            for j in 0..dv {
                delta[j] = w_val * v[j] - temp[j];
            }
        }
        Gdn2GateConfig::Full => {
            for j in 0..dv {
                delta[j] = w[j] * v[j] - temp[j];
            }
        }
    }
    // S += k ⊗ delta using SIMD-accelerated outer product
    simd_outer_product_acc(s, k, &delta, dk, dv);

    // Step 4: Readout o = Sᵀ q
    for j in 0..dv {
        let mut sum = 0.0f32;
        for i in 0..dk {
            unsafe {
                sum += *s.get_unchecked(i * dv + j) * *q.get_unchecked(i);
            }
        }
        unsafe {
            *out.get_unchecked_mut(j) = sum;
        }
    }
}

/// Sigmoid function for gate projections.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// L2 normalize a vector in-place: x /= ‖x‖₂ + ε.
#[inline]
pub fn l2_normalize(x: &mut [f32]) {
    let norm_sq: f32 = x.iter().map(|&v| v * v).sum();
    let inv_norm = 1.0 / (norm_sq.sqrt() + 1e-8);
    simd_scale_inplace(x, inv_norm);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify sigmoid output is in [0, 1].
    #[test]
    fn sigmoid_range() {
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let s = sigmoid(x);
            assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s} out of [0,1]");
        }
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6, "sigmoid(0) ≈ 0.5");
    }

    /// Verify L2 normalize produces unit vector.
    #[test]
    fn l2_normalize_unit() {
        let mut x = vec![3.0, 4.0, 0.0];
        l2_normalize(&mut x);
        let norm: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm = {norm}, expected 1.0");
    }

    /// Verify L2 normalize handles zero vector gracefully.
    #[test]
    fn l2_normalize_zero() {
        let mut x = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut x);
        for &v in &x {
            assert!(v.is_finite(), "zero-normalize should produce finite: {v}");
        }
    }

    /// Verify basic recurrent step produces finite output with zero state.
    #[test]
    fn recurrent_step_zero_state_finite() {
        let dk = 4;
        let dv = 4;
        let mut s = vec![0.0; dk * dv];
        let k = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let q = vec![0.5, 0.5, 0.5, 0.5];
        let alpha = vec![0.99; dk];
        let b = vec![1.0; dk];
        let w_channel = vec![1.0; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];

        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        for &o in &out {
            assert!(o.is_finite(), "output should be finite: {o}");
        }
    }

    /// Verify recurrent step with all gate configs produces finite output.
    #[test]
    fn recurrent_step_all_gate_configs() {
        let dk = 4;
        let dv = 4;
        let k = vec![0.5; dk];
        let v = vec![1.0; dv];
        let q = vec![0.5; dk];
        let alpha = vec![0.99; dk];
        let b = vec![0.8; dk];
        let w_channel = vec![0.9; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];

        for gate_config in [
            Gdn2GateConfig::EraseOnly,
            Gdn2GateConfig::Full,
            Gdn2GateConfig::Kda,
        ] {
            let mut s = vec![0.0; dk * dv];
            gdn2_recurrent_step(
                &k,
                &v,
                &q,
                &mut s,
                &alpha,
                &b,
                1.0,
                &w_channel,
                &mut out,
                &mut temp,
                dk,
                dv,
                gate_config,
            );
            for &o in &out {
                assert!(
                    o.is_finite(),
                    "output should be finite for {gate_config:?}: {o}"
                );
            }
        }
    }

    /// Verify decay shrinks state: after step with v=0, state should be smaller.
    #[test]
    fn recurrent_step_decay_shrinks_state() {
        let dk = 4;
        let dv = 4;
        // Pre-fill state
        let mut s = vec![1.0; dk * dv];
        let k = vec![0.5; dk];
        let v = vec![0.0; dv]; // zero value: no new write
        let q = vec![0.5; dk];
        let alpha = vec![0.5; dk]; // strong decay
        let b = vec![1.0; dk];
        let w_channel = vec![1.0; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];

        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // State should decay to ~0.5 (plus the read-correction which is 0 for zero value)
        let s_max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            s_max < 0.9,
            "state should decay with alpha=0.5, max={s_max}"
        );
    }

    /// Verify outer product update: with zero initial state, S should become k⊗(w⊙v).
    #[test]
    fn recurrent_step_writes_outer_product() {
        let dk = 4;
        let dv = 4;
        let mut s = vec![0.0; dk * dv];
        let k = vec![1.0, 0.0, 0.0, 0.0]; // unit basis
        let v = vec![0.0, 0.0, 0.0, 1.0]; // unit basis
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let alpha = vec![1.0; dk]; // no decay
        let b = vec![1.0; dk]; // open erase gate
        let w_channel = vec![1.0; dv]; // open write gate
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];

        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // S should be k ⊗ v (row 0 = [0,0,0,1], rest zero)
        assert!((s[0] - 0.0).abs() < 1e-6, "s[0] should be 0");
        assert!((s[3] - 1.0).abs() < 1e-6, "s[3] should be 1.0 (k[0]*v[3])");
        for (offset, &val) in s[dk..dk * dv].iter().enumerate() {
            let i = dk + offset;
            assert!(val.abs() < 1e-6, "s[{i}] should be 0, got {val}");
        }
    }

    /// Verify multi-step accumulation: two tokens should accumulate correctly.
    #[test]
    fn recurrent_step_multi_step_accumulates() {
        let dk = 4;
        let dv = 4;
        let mut s = vec![0.0; dk * dv];
        let alpha = vec![1.0; dk]; // no decay
        let b = vec![1.0; dk];
        let w_channel = vec![1.0; dv];
        let mut out = vec![0.0; dv];
        let mut temp = vec![0.0; dv];

        // Token 1: k = [1,0,0,0], v = [1,0,0,0]
        let k1 = vec![1.0, 0.0, 0.0, 0.0];
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let q1 = vec![1.0, 0.0, 0.0, 0.0];
        gdn2_recurrent_step(
            &k1,
            &v1,
            &q1,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // Token 2: k = [0,1,0,0], v = [0,1,0,0]
        let k2 = vec![0.0, 1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let q2 = vec![1.0, 0.0, 0.0, 0.0];
        gdn2_recurrent_step(
            &k2,
            &v2,
            &q2,
            &mut s,
            &alpha,
            &b,
            1.0,
            &w_channel,
            &mut out,
            &mut temp,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // S should have k1⊗v1 + k2⊗v2 (no decay, no erase correction for zero state)
        // s[0] = k1[0]*v1[0] = 1.0
        // s[1*4+1] = k2[1]*v2[1] = 1.0
        assert!(
            (s[0] - 1.0).abs() < 1e-5,
            "s[0] should be ~1.0 from first token, got {}",
            s[0]
        );
        assert!(
            (s[5] - 1.0).abs() < 1e-5,
            "s[5] should be ~1.0 from second token, got {}",
            s[5]
        );
    }
}
