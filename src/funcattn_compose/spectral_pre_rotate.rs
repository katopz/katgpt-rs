//! T5.1 — Compose FUNCATTN with SpectralQuant's calibrated eigenbasis.
//!
//! Hypothesis (Plan 286 T5.1): a FUNCATTN basis whose rows are aligned with the
//! data's principal eigen-directions is *more expressive per parameter* than a
//! randomly/orthogonally-init'd basis, because signal energy concentrates in
//! the top eigen-directions (SpectralQuant's core finding, Plan 077).
//!
//! The rotation primitive itself ships in `katgpt-core::funcattn`
//! ([`katgpt_core::funcattn::pre_rotate_basis_weights_into`]) — it computes
//! `W_Φ' = W_Φ · Vᵀ` in place, where `V` is the eigenbasis (columns =
//! eigenvectors, sorted by eigenvalue descending). This module is the
//! **composition glue**: it calls SpectralQuant's
//! [`calibrate_eigenbasis`](crate::spectralquant::calibrate_eigenbasis) to
//! obtain `V` from calibration samples, then feeds it to the pre-rotator, and
//! exposes the spectral-concentration diagnostics that connect FUNCATTN's `k`
//! to SpectralQuant's cumulative-variance thresholds.
//!
//! # Zero-alloc hot path preserved
//!
//! Pre-rotation is a **one-time calibration-time transform**. After it runs,
//! the rotated `w_basis` is handed to [`katgpt_core::funcattn::funcattn_forward`]
//! unchanged — the forward path is byte-identical and remains G5 zero-alloc.
//! No function in this module runs per decode step.
//!
//! # Not a GOAT gate
//!
//! This ships the *mechanism* + *diagnostics*. Proving the "more expressive per
//! parameter" hypothesis requires a composition-specific benchmark (FUNCATTN
//! with random basis vs eigen-aligned basis at matched param budget) which is
//! deferred per the plan's Gain-tier opt-in policy. The mechanics verified here:
//! (1) the glue produces a non-trivial rotation for real data, (2) the forward
//! pass remains finite + partition-of-unity after rotation (delegated to the
//! primitive's own test suite in `katgpt-core`).

use katgpt_core::funcattn::pre_rotate_basis_weights_into;

use crate::spectralquant::{CalibrationResult, calibrate_eigenbasis};

/// Calibrate a SpectralQuant eigenbasis from `samples`, then pre-rotate the
/// FUNCATTN basis weights `w_basis` into that eigen-frame, in place.
///
/// One-call composition of SpectralQuant × FUNCATTN (Plan 286 T5.1). After this
/// returns, `w_basis` is eigen-aligned and ready to pass to
/// [`katgpt_core::funcattn::funcattn_forward`].
///
/// # Arguments
/// * `w_basis` — `(k, d)` row-major basis projection weights, mutated in place.
///   Must be orthogonally initialized by the caller before the first forward
///   pass (reference L20-21); this rotation preserves orthogonality (verified
///   in `katgpt-core`'s primitive test suite).
/// * `samples` — calibration vectors, each of length `d`. Typically a sample of
///   the input stream the head will see (pre-projection latents). Reuses
///   SpectralQuant's standard calibration path so the eigenbasis is identical
///   to what KV-cache compression would produce.
/// * `k, d` — FUNCATTN basis dim `k` and head/feature dim `d`.
///
/// # Returns
/// The [`CalibrationResult`] (eigenvalues, eigenvectors, `var_95`, `var_99`,
/// `d_eff`) so the caller can introspect spectral concentration — e.g. decide
/// whether to truncate `k` to [`effective_basis_rank`] for a compression gain.
///
/// # Panics
/// Debug-asserts shape compatibility. Delegates to
/// [`calibrate_eigenbasis`] (panics on empty samples / dim mismatch) and
/// [`pre_rotate_basis_weights_into`] (debug-asserts `w_basis.len() == k*d`,
/// `eigenvectors.len() == d*d`).
pub fn calibrate_and_pre_rotate_basis(
    w_basis: &mut [f32],
    samples: &[Vec<f32>],
    k: usize,
    d: usize,
) -> CalibrationResult {
    let cal = calibrate_eigenbasis(samples, d);
    pre_rotate_basis_weights_into(w_basis, &cal.eigenvectors, k, d);
    cal
}

/// Pre-rotate `w_basis` using an already-computed [`CalibrationResult`].
///
/// Use this when the eigenbasis is shared across heads/layers and calibrated
/// once — avoids recomputing the eigendecomposition per head. Validates that
/// the calibration's `head_dim` matches `d`.
///
/// # Panics
/// Debug-asserts `cal.head_dim == d`.
pub fn pre_rotate_from_calibration(w_basis: &mut [f32], cal: &CalibrationResult, k: usize, d: usize) {
    debug_assert_eq!(
        cal.head_dim, d,
        "calibration head_dim ({}) must match basis dim d ({})",
        cal.head_dim, d
    );
    pre_rotate_basis_weights_into(w_basis, &cal.eigenvectors, k, d);
}

/// Recommend a basis rank `r ≤ k` that captures `variance_threshold` (e.g. 0.95)
/// of the total spectral energy, per the calibrated eigenvalue spectrum.
///
/// This is the spectral-concentration payoff of composing with SpectralQuant:
/// it tells you how many FUNCATTN basis partitions are "spent" on directions
/// that carry negligible energy. If `effective_basis_rank(cal, 0.95)` returns
/// `r ≪ k`, the basis is over-provisioned — a caller could shrink `k` to `r`
/// for a parameter/latency win without losing expressive capacity.
///
/// Returns the smallest `r` such that `Σ_{i<r} λ_i / Σ λ ≥ variance_threshold`,
/// clamped to `[1, min(k, d)]`. Uses the precomputed `var_95`/`var_99` when the
/// threshold matches exactly (no rescan); otherwise scans the sorted-descending
/// eigenvalues (SpectralQuant guarantees descending order).
pub fn effective_basis_rank(cal: &CalibrationResult, variance_threshold: f32, k: usize) -> usize {
    let d = cal.head_dim;
    if d == 0 || cal.eigenvalues.is_empty() {
        return 1;
    }
    // Fast paths for the two standard thresholds SpectralQuant precomputes.
    let fast = if (variance_threshold - 0.95).abs() < 1e-6 {
        Some(cal.var_95)
    } else if (variance_threshold - 0.99).abs() < 1e-6 {
        Some(cal.var_99)
    } else {
        None
    };
    if let Some(r) = fast {
        return r.min(k.max(1)).max(1);
    }

    let total: f64 = cal.eigenvalues.iter().map(|&x| x as f64).sum();
    if total <= 0.0 {
        return 1;
    }
    let target = total * (variance_threshold as f64);
    let mut acc = 0.0f64;
    let mut r = 1usize;
    for (i, &x) in cal.eigenvalues.iter().enumerate() {
        acc += x as f64;
        r = i + 1;
        if acc >= target {
            break;
        }
    }
    r.min(k.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::funcattn::{FuncAttnBasis, FuncAttnConfig, FuncAttnScratch, funcattn_forward};

    /// Simple deterministic RNG (LCG) so tests don't depend on fastrand state.
    fn lcg_fill(out: &mut [f32], seed: u32) {
        let mut s = seed;
        for x in out.iter_mut() {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *x = ((s >> 8) as f32 / 16777216.0) - 0.5;
        }
    }

    /// Build an orthogonal-ish `d×d` eigenbasis by Gram-Schmidt on random rows.
    fn random_orthogonal_matrix(d: usize, seed: u32) -> Vec<f32> {
        let mut m = vec![0.0f32; d * d];
        lcg_fill(&mut m, seed);
        // Gram-Schmidt over rows.
        for i in 0..d {
            for j in 0..i {
                let dot = (0..d).map(|c| m[i * d + c] * m[j * d + c]).sum::<f32>();
                for c in 0..d {
                    m[i * d + c] -= dot * m[j * d + c];
                }
            }
            let nrm = (0..d).map(|c| m[i * d + c] * m[i * d + c]).sum::<f32>().sqrt();
            let nrm = if nrm < 1e-12 { 1.0 } else { nrm };
            for c in 0..d {
                m[i * d + c] /= nrm;
            }
        }
        m
    }

    #[test]
    fn glue_produces_nontrivial_rotation_for_nonidentity_data() {
        // A non-identity eigenbasis must change w_basis (not be a no-op).
        let k = 4;
        let d = 8;
        let mut w_basis = vec![0.0f32; k * d];
        lcg_fill(&mut w_basis, 1);
        let original = w_basis.clone();

        // Calibration samples: random vectors in R^d. The resulting eigenbasis
        // is essentially a random rotation (not identity) for generic data.
        let samples: Vec<Vec<f32>> = (0..32).map(|i| {
            let mut v = vec![0.0f32; d];
            lcg_fill(&mut v, 100 + i);
            v
        }).collect();

        let cal = calibrate_and_pre_rotate_basis(&mut w_basis, &samples, k, d);
        assert_eq!(cal.head_dim, d);
        assert_eq!(cal.eigenvectors.len(), d * d);

        let diff: f64 = w_basis
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (*a - *b) as f64 * (*a - *b) as f64)
            .sum::<f64>()
            .sqrt();
        assert!(
            diff > 1e-3,
            "non-identity eigenbasis must rotate w_basis (diff={diff:.6})"
        );
    }

    #[test]
    fn forward_remains_finite_after_glue_rotation() {
        // Mechanics: after calibrate+pre-rotate, the forward pass must still
        // produce finite, bounded output (Plan 286 G1 spirit).
        let n = 16;
        let d = 8;
        let k = 4;
        let mut w_basis = random_orthogonal_matrix_rows(k, d, 7);
        let w_q = random_orthogonal_matrix(d, 11);
        let w_k = random_orthogonal_matrix(d, 12);
        let w_v = random_orthogonal_matrix(d, 13);

        let samples: Vec<Vec<f32>> = (0..24).map(|i| {
            let mut v = vec![0.0f32; d];
            lcg_fill(&mut v, 200 + i);
            v
        }).collect();
        let _cal = calibrate_and_pre_rotate_basis(&mut w_basis, &samples, k, d);

        let mut x = vec![0.0f32; n * d];
        lcg_fill(&mut x, 300);
        // Sigmoid basis needs a sharp temperature for small-magnitude inputs
        // (see funcattn.rs module doc "Temperature requirement for sigmoid").
        let cfg = FuncAttnConfig { d, k, basis: FuncAttnBasis::Sigmoid, alpha: 0.5, temperature: 0.1, cholesky_jitter: 1e-6 };
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        funcattn_forward(&x, &x, &w_basis, &w_q, &w_k, &w_v, &cfg, &mut scratch, &mut out)
            .expect("forward must succeed after rotation");

        let all_finite = out.iter().all(|v| v.is_finite());
        assert!(all_finite, "output must be finite after eigen-rotation");
        let max_abs = out.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_abs < 1e4, "output must be bounded (max_abs={max_abs})");
    }

    #[test]
    fn pre_rotate_from_calibration_matches_one_call() {
        // pre_rotate_from_calibration must reproduce calibrate_and_pre_rotate_basis
        // when given the same calibration result.
        let k = 4;
        let d = 8;
        let samples: Vec<Vec<f32>> = (0..16).map(|i| {
            let mut v = vec![0.0f32; d];
            lcg_fill(&mut v, 400 + i);
            v
        }).collect();

        let mut w1 = random_orthogonal_matrix_rows(k, d, 5);
        let mut w2 = w1.clone();
        let cal = calibrate_eigenbasis(&samples, d);

        calibrate_and_pre_rotate_basis(&mut w1, &samples, k, d);
        pre_rotate_from_calibration(&mut w2, &cal, k, d);

        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-4, "two-call and one-call must match");
        }
    }

    #[test]
    fn effective_basis_rank_respects_threshold_and_k() {
        // Construct a calibration with a known spectrum: 2 large + rest tiny.
        // Top-1 captures 10/18.06 ≈ 0.554; top-2 captures ≈ 0.997.
        let d = 8;
        let eigenvalues = vec![10.0f32, 8.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];
        let eigenvectors = random_orthogonal_matrix(d, 99);
        // Set CORRECT precomputed var thresholds (the fast path trusts these):
        // 0.554 < 0.95 ≤ 0.997 → var_95 = 2; var_99 = 2.
        let cal = CalibrationResult {
            eigenvectors,
            eigenvalues: eigenvalues.clone(),
            d_eff: 0.0,
            spectral_gap: None,
            var_95: 2,
            var_99: 2,
            n_samples: 16,
            head_dim: d,
        };

        // Fast path: 0.95 threshold uses precomputed var_95 = 2.
        let r_95 = effective_basis_rank(&cal, 0.95, 4);
        assert_eq!(r_95, 2, "95% variance (precomputed) needs rank 2");

        // Scan path: non-standard threshold 0.50 → top-1 captures 0.554 ≥ 0.50.
        let r_50 = effective_basis_rank(&cal, 0.50, 4);
        assert_eq!(r_50, 1, "50% variance (scan) captured by rank 1");

        // Clamped to k when the spectrum is flat-ish but k is small.
        let cal_flat = CalibrationResult {
            eigenvectors: random_orthogonal_matrix(d, 98),
            eigenvalues: vec![1.0f32; d], // uniform spectrum
            d_eff: 0.0,
            spectral_gap: None,
            var_95: d,
            var_99: d,
            n_samples: 16,
            head_dim: d,
        };
        // Uniform spectrum, threshold 0.5: need d/2 = 4 dims, but clamp to k=3.
        let r_flat = effective_basis_rank(&cal_flat, 0.5, 3);
        assert_eq!(r_flat, 3, "uniform spectrum rank clamped to k");
    }

    /// Helper: k orthogonal rows of length d (for w_basis, k may be < d).
    fn random_orthogonal_matrix_rows(k: usize, d: usize, seed: u32) -> Vec<f32> {
        let full = random_orthogonal_matrix(d, seed);
        full.into_iter().take(k * d).collect()
    }
}
