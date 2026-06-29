//! Shared spectral retraction helpers — power iteration + L2 retraction.
//!
//! Extracted as the DRY primitive common to:
//! - [`crate::gauge_invariant::gauge_rebalance`] (Plan 270, Research 238) —
//!   power iteration for σ_max estimation on LoRA factor pairs.
//! - [`crate::manifold_power_iter_router`] (Plan 279, Research 246) — power
//!   iteration + L2 retraction of MoE router rows against expert Grams.
//!
//! Both are instances of "power-iteration step + norm retraction on a vector
//! against a PSD operator." This module provides the explicit-PSD-operator
//! form (the MPI router has a pre-materialized `D × D` Gram); the LoRA path
//! in `gauge_invariant.rs` uses an *implicit* two-matrix fused form
//! (`M^T·(M·v)` without ever materializing `M^T·M`) to avoid allocation, and
//! therefore composes its own loop — see the doc comment on
//! [`power_iter_retract`] for the migration note.
//!
//! # Always-On
//!
//! This module is NOT feature-gated. `gauge_invariant` is a default-ON feature
//! and may consume the helpers here in the future; gating would break that.
//! The MPI router itself is feature-gated to `manifold_power_iter_router`.

#![allow(clippy::too_many_arguments)]

use katgpt_core::simd::{simd_dot_f32, simd_fused_decay_write, simd_scale_inplace};

/// Caller-owned scratch for one power-iteration + retraction.
///
/// Reused across rows / experts / calls — zero allocation in the hot path.
/// Sized for vectors of length up to `dim` (see [`PowerRetractScratch::new`]).
///
/// Mirrors the `PowerIterationScratch` pattern from `src/distill/peira.rs` and
/// the `GaugeRebalanceScratch` pattern from `src/gauge_invariant.rs`.
#[derive(Debug, Clone)]
pub struct PowerRetractScratch {
    /// Mat-vec output buffer: `psd_op · v`. Length `dim`.
    pub mv_out: Vec<f32>,
    /// Cached L2 norm of the most recent `mv_out` (post-matvec, pre-normalize).
    /// Exposed for diagnostics (e.g. σ_max estimation reads this).
    pub norm: f32,
}

impl PowerRetractScratch {
    /// Create scratch sized for vectors of length `dim`.
    #[inline]
    pub fn new(dim: usize) -> Self {
        Self {
            mv_out: vec![0.0; dim],
            norm: 0.0,
        }
    }

    /// Resize buffers if the dimension changed. No-op if already correct.
    /// Useful when a single scratch is reused across differently-sized calls.
    #[inline]
    pub fn ensure_dim(&mut self, dim: usize) {
        if self.mv_out.len() != dim {
            self.mv_out.clear();
            self.mv_out.resize(dim, 0.0);
        }
        self.norm = 0.0;
    }
}

impl Default for PowerRetractScratch {
    /// Default-sized scratch (dim = 0). Callers MUST `ensure_dim` before use.
    fn default() -> Self {
        Self {
            mv_out: Vec::new(),
            norm: 0.0,
        }
    }
}

/// ONE step of power iteration on `v` against an arbitrary PSD operator,
/// expressed via a caller-supplied matvec closure.
///
/// Computes:
/// 1. `mv_out <- matvec(v)` (caller-defined — explicit Gram for MPI, implicit
///    `M^T M` for gauge).
/// 2. `norm <- ‖mv_out‖₂` (stored in scratch for diagnostics).
/// 3. If `norm >= 1e-20`: `v <- mv_out / norm` (unit normalization).
///    If `norm < 1e-20`: leave `v` unchanged (zero-row / null-space safety —
///    matches `test_gauge_rebalance_zero_matrix_safe` semantics).
///
/// Returns the pre-normalization norm. The caller decides what to do with it:
/// - `gauge_invariant::power_iterate_sigma_max`: uses the final-step norm as
///   σ_max estimate (via a follow-up `‖M·v‖²` pass).
/// - `power_iter_retract`: applies `target_norm` scaling after the loop.
///
/// # Zero-alloc
///
/// `mv_out` is caller-owned. No heap allocation in this call.
///
/// # Determinism
///
/// Pure function of `(v, matvec-output)`. Same inputs -> byte-identical `v'`.
#[inline]
pub fn power_iter_step<F>(v: &mut [f32], mv_out: &mut [f32], matvec: F) -> f32
where
    F: Fn(&[f32], &mut [f32]),
{
    let n = v.len();
    debug_assert_eq!(mv_out.len(), n, "power_iter_step: mv_out length mismatch");

    // Step 1: mv_out <- matvec(v).
    matvec(v, mv_out);

    // Step 2: norm <- ‖mv_out‖₂.
    let norm = simd_dot_f32(mv_out, mv_out, n).sqrt();

    // Step 3: normalize or leave unchanged on degeneracy.
    if norm < 1e-20 {
        // Degenerate — operator annihilated v (zero row, null space).
        // Leave v untouched so the caller's zero-row safety holds.
        return 0.0;
    }
    let inv = 1.0 / norm;
    simd_scale_inplace(mv_out, inv);
    v.copy_from_slice(mv_out);
    norm
}

/// One matrix-vector product `out = psd_op · v` for a row-major `dim × dim`
/// PSD operator. Zero-allocation — writes into `scratch.mv_out`.
///
/// `psd_op` is the explicit Gram matrix `M = W·W^T` (symmetric, PSD). For the
/// LoRA gauge path the PSD operator is `A^T·A` (or `B^T·B`) — but formed
/// implicitly, so that path composes its own matvec.
#[inline]
pub fn matvec_psd_into(
    v: &[f32],
    psd_op: &[f32],
    dim: usize,
    scratch: &mut PowerRetractScratch,
) {
    debug_assert_eq!(v.len(), dim, "v length mismatch");
    debug_assert_eq!(psd_op.len(), dim * dim, "psd_op size mismatch");
    debug_assert!(scratch.mv_out.len() >= dim, "scratch.mv_out too small");

    // out[i] = Σ_j psd_op[i*dim + j] * v[j]
    // Zero the output region first.
    for i in 0..dim {
        scratch.mv_out[i] = 0.0;
    }
    for i in 0..dim {
        let row = &psd_op[i * dim..(i + 1) * dim];
        scratch.mv_out[i] = simd_dot_f32(row, v, dim);
    }
}

/// One step of power iteration against an explicit PSD operator, with L2
/// retraction to `target_norm`. Updates `v` in place.
///
/// One step: `v ← psd_op · v`, then `v ← target_norm · v / ‖v‖₂`.
/// Repeats `iters` times. Zero allocation — caller-owned `scratch`.
///
/// # Degenerate Input
///
/// If `‖psd_op · v‖₂ < 1e-20` (zero matrix, or `v` in the null space), `v` is
/// left **unchanged** and the function returns early. This mirrors
/// `gauge_rebalance`'s zero-matrix safety (Plan 270).
///
/// # Determinism
///
/// Bit-identical output for the same `(v, psd_op, dim, target_norm, iters)`
/// across runs — safe for `SyncBlock → ChainConsensus` quorum commit.
///
/// # Migration Note (Plan 279 DRY)
///
/// `gauge_invariant::power_iterate_sigma_max` (Plan 270) does power iteration
/// for σ_max on an *implicit* `M^T·M` operator (the input is a tall `outer×rank`
/// matrix, and the matvec is the fused two-pass `M^T·(M·v)` that avoids
/// materializing the `rank × rank` Gram). Migrating it to call this helper
/// would require forming the Gram (`rank²` floats) — a one-time scratch cost
/// but a regression vs the current zero-alloc fused form. The migration is
/// therefore deferred: this helper serves the MPI router (which has a
/// pre-materialized Gram) and any future caller that already has an explicit
/// PSD operator. The LoRA path keeps its fused form for allocation discipline.
pub fn power_iter_retract(
    v: &mut [f32],
    psd_op: &[f32],
    dim: usize,
    target_norm: f32,
    iters: u8,
    scratch: &mut PowerRetractScratch,
) {
    debug_assert_eq!(v.len(), dim, "v length mismatch");
    debug_assert_eq!(psd_op.len(), dim * dim, "psd_op size mismatch");
    debug_assert!(target_norm >= 0.0, "target_norm must be non-negative");
    debug_assert!(scratch.mv_out.len() >= dim, "scratch too small");

    if dim == 0 || iters == 0 {
        return;
    }

    for _ in 0..iters {
        // matvec: mv_out = psd_op · v
        matvec_psd_into(v, psd_op, dim, scratch);

        // L2 norm of mv_out.
        let norm_sq = simd_dot_f32(&scratch.mv_out[..dim], &scratch.mv_out[..dim], dim);
        let norm = norm_sq.sqrt();
        scratch.norm = norm;

        if norm < 1e-20 {
            // Degenerate — leave v unchanged (mirror gauge_rebalance zero safety).
            return;
        }

        // Retract: v ← target_norm · mv_out / ‖mv_out‖
        let scale = target_norm / norm;
        // v = scale * mv_out. Fused scale-copy via decay_write with decay=0
        // (single NEON/AVX2 pass; was a scalar loop). v and mv_out do not alias.
        simd_fused_decay_write(&mut v[..dim], 0.0, &scratch.mv_out[..dim], scale);
    }
}

/// Compute the L2 norm of `v[..dim]`. Convenience wrapper around `simd_dot_f32`.
#[inline]
pub fn l2_norm(v: &[f32], dim: usize) -> f32 {
    simd_dot_f32(&v[..dim], &v[..dim], dim).sqrt()
}

/// Scale `v` in place by `s` using the SIMD-accelerated scaler. Convenience
/// wrapper around `simd_scale_inplace`.
#[inline]
pub fn scale_inplace(v: &mut [f32], s: f32) {
    simd_scale_inplace(v, s);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vector (xorshift64).
    fn seeded_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut s = seed;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            v.push(((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0);
        }
        v
    }

    /// Build a known PSD matrix `M = w·w^T + ε·I` (rank-1 + diagonal regularization).
    /// The principal eigenvector is `w/‖w‖` with eigenvalue `‖w‖² + ε`.
    fn rank1_psd(w: &[f32], dim: usize, eps: f32) -> Vec<f32> {
        let mut m = vec![0.0_f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                m[i * dim + j] = w[i] * w[j];
            }
            m[i * dim + i] += eps;
        }
        m
    }

    #[test]
    fn test_power_iter_retract_converges_to_principal_direction() {
        // For a rank-1 PSD M = w·w^T + ε·I, the principal eigenvector is w/‖w‖.
        // After 5 iterations of power_iter_retract, v should be ≈ ±target_norm * w/‖w‖.
        let dim = 8;
        let w = seeded_vec(42, dim);
        let psd = rank1_psd(&w, dim, 1e-3);
        let target_norm = 2.5;
        let mut v = seeded_vec(7, dim);
        let mut scratch = PowerRetractScratch::new(dim);
        power_iter_retract(&mut v, &psd, dim, target_norm, 5, &mut scratch);

        // Normalize both for cosine comparison.
        let w_norm = l2_norm(&w, dim);
        let v_norm = l2_norm(&v, dim);
        let dot: f32 = (0..dim).map(|i| v[i] * w[i] / (v_norm * w_norm)).sum();
        let cos = dot.abs(); // sign-agnostic
        assert!(cos > 0.99, "cosine {cos} < 0.99 after 5 iters");

        // Norm invariant.
        assert!(
            (v_norm - target_norm).abs() < 1e-4,
            "norm {v_norm} != target {target_norm}"
        );
    }

    #[test]
    fn test_power_iter_retract_norm_invariant_after_one_step() {
        let dim = 4;
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let psd = rank1_psd(&w, dim, 0.1);
        let target = 1.7;
        let mut v = vec![0.1, 0.2, 0.3, 0.4];
        let mut scratch = PowerRetractScratch::new(dim);
        power_iter_retract(&mut v, &psd, dim, target, 1, &mut scratch);
        let n = l2_norm(&v, dim);
        assert!((n - target).abs() < 1e-5, "norm {n} != target {target}");
    }

    #[test]
    fn test_power_iter_retract_zero_matrix_is_noop() {
        // Zero PSD operator → degenerate → v unchanged.
        let dim = 4;
        let psd = vec![0.0_f32; dim * dim];
        let original = vec![0.5, -0.25, 0.125, 0.8];
        let mut v = original.clone();
        let mut scratch = PowerRetractScratch::new(dim);
        power_iter_retract(&mut v, &psd, dim, 1.0, 3, &mut scratch);
        for i in 0..dim {
            assert!((v[i] - original[i]).abs() < 1e-20, "v[{i}] changed on zero PSD");
        }
    }

    #[test]
    fn test_power_iter_retract_deterministic() {
        let dim = 6;
        let w = seeded_vec(123, dim);
        let psd = rank1_psd(&w, dim, 1e-2);
        let target = 1.0;
        let mut scratch = PowerRetractScratch::new(dim);

        let mut v1 = seeded_vec(7, dim);
        let mut v2 = seeded_vec(7, dim);
        power_iter_retract(&mut v1, &psd, dim, target, 3, &mut scratch);
        power_iter_retract(&mut v2, &psd, dim, target, 3, &mut scratch);
        for i in 0..dim {
            assert!(v1[i].to_bits() == v2[i].to_bits(), "non-deterministic at [{i}]");
        }
    }

    #[test]
    fn test_matvec_psd_matches_naive() {
        let dim = 3;
        let m: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let v: Vec<f32> = vec![1.0, 0.5, -0.5];
        let mut scratch = PowerRetractScratch::new(dim);
        matvec_psd_into(&v, &m, dim, &mut scratch);
        // Naive: out[i] = Σ_j m[i*dim+j] * v[j]
        let expected = [
            1.0 * 1.0 + 2.0 * 0.5 + 3.0 * (-0.5),
            4.0 * 1.0 + 5.0 * 0.5 + 6.0 * (-0.5),
            7.0 * 1.0 + 8.0 * 0.5 + 9.0 * (-0.5),
        ];
        for (i, expected_val) in expected.iter().enumerate().take(dim) {
            assert!((scratch.mv_out[i] - expected_val).abs() < 1e-6, "matvec mismatch at [{i}]");
        }
    }

    #[test]
    fn test_ensure_dim_grows_and_resets() {
        let mut s = PowerRetractScratch::default();
        assert!(s.mv_out.is_empty());
        s.ensure_dim(8);
        assert_eq!(s.mv_out.len(), 8);
        s.ensure_dim(8); // no-op
        assert_eq!(s.mv_out.len(), 8);
        s.ensure_dim(16);
        assert_eq!(s.mv_out.len(), 16);
        // Norm is reset on resize.
        s.norm = 99.0;
        s.ensure_dim(16);
        assert_eq!(s.norm, 0.0);
    }

    #[test]
    fn test_power_iter_step_normalizes_to_unit() {
        // Matvec scales v by 3 → pre-norm = 3·‖v_in‖; v should be unit after.
        let mut v = vec![1.0_f32, 2.0, 2.0]; // ‖v‖ = 3
        let mut mv_out = vec![0.0_f32; 3];
        let norm = power_iter_step(&mut v, &mut mv_out, |vin, vout| {
            for i in 0..vin.len() {
                vout[i] = vin[i] * 3.0;
            }
        });
        // pre-norm = 3 * ‖v_in‖ = 3 * 3 = 9.
        assert!((norm - 9.0).abs() < 1e-5, "norm={norm}, expected 9");
        let n = l2_norm(&v, 3);
        assert!((n - 1.0).abs() < 1e-5, "v should be unit, got {n}");
    }

    #[test]
    fn test_power_iter_step_degenerate_returns_zero() {
        // Matvec produces zero → step returns 0, v left unchanged.
        let mut v = vec![0.5_f32, 0.5, 0.5];
        let v_orig = v.clone();
        let mut mv_out = vec![0.0_f32; 3];
        let norm = power_iter_step(&mut v, &mut mv_out, |_vin, vout| {
            vout.fill(0.0);
        });
        assert_eq!(norm, 0.0);
        assert_eq!(v, v_orig, "v should be unchanged on degeneracy");
    }
}
