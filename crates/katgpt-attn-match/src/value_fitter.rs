//! Cv (compact value) fitting via Ordinary Least Squares.
//!
//! Solves:
//! ```text
//! min_Cv || X · Cv − Y ||²_F
//! ```
//! where `X_i = softmax((q_i Ck^T + β) / √d)` is the compact attention row
//! and `Y_i = softmax(q_i K^T / √d) V` is the original attention output.
//!
//! Closed-form solution: `Cv = (X^T X)^{-1} X^T Y`.
//!
//! Default solver: the shared leaf Cholesky kernels
//! (`katgpt_core::linalg::ridge_solve::{try_cholesky_strict_f32, chol_solve_f32}`)
//! with diagonal jitter fallback for rank-deficient systems.
//! No external LAPACK dependency.
//!
//! # Numerics note (the `f978a20b` lesson, carried from the retired local copy)
//!
//! The retired local factorisations (unblocked + blocked, scalar inner loops)
//! deliberately avoided `dot_8wide` because Cholesky on near-singular
//! matrices is a numerical cliff edge where ULP-level summation-order changes
//! flipped a pivot and degraded the G5 gate 0.7% -> 7.5% relative error
//! (`f978a20b`). The shared kernels carry SIMD-interior dots with a
//! relative-tolerance pivot clamp instead; the swap was graded on the G5
//! RELATIVE-ERROR floors (never bit identity) per Issue 995.

#![allow(clippy::needless_range_loop)]

use katgpt_core::linalg::ridge_solve::{chol_solve_f32, try_cholesky_strict_f32};

use crate::{STABILITY_EPS, score_matrix_simd::dot_8wide};

/// Configuration for Cv fitting.
#[derive(Debug, Clone, Copy)]
pub struct ValueFitConfig {
    /// Ridge regularization λ for `X^T X + λ I` (default 0 — paper found it hurts).
    pub ridge_lambda: f32,
    /// Diagonal jitter added to `X^T X` if Cholesky fails (rank-deficient).
    pub cholesky_jitter: f32,
}

impl Default for ValueFitConfig {
    fn default() -> Self {
        Self {
            ridge_lambda: 0.0,
            cholesky_jitter: 1e-6,
        }
    }
}

/// Result of Cv fitting.
#[derive(Debug, Clone)]
pub struct ValueFitResult {
    /// Compacted values `Cv` — flat `t * d` f32, row-major.
    pub compact_values: Vec<f32>,
    /// Relative Frobenius error `||X·Cv − Y||_F / ||Y||_F`.
    pub relative_error: f32,
    /// Whether the Cholesky solver succeeded (false = used jitter or fallback).
    pub solver_succeeded: bool,
    /// Final jitter used (0.0 if no jitter needed).
    pub jitter_used: f32,
}

/// Fit Cv via ordinary least squares.
///
/// # Arguments
/// * `x` - The `(n, t)` compact attention matrix (row-major). `X_i = softmax((q_i Ck^T + β) / √d)`.
/// * `y` - The `(n, d)` target attention-output matrix (row-major). `Y_i = softmax(q_i K^T / √d) V`.
/// * `n` - Number of reference queries.
/// * `t` - Number of compact keys.
/// * `d` - Head dimension.
/// * `config` - Solver configuration.
pub fn fit_cv_least_squares(
    x: &[f32],
    y: &[f32],
    n: usize,
    t: usize,
    d: usize,
    config: &ValueFitConfig,
) -> ValueFitResult {
    assert_eq!(x.len(), n * t, "compact attention matrix size mismatch");
    assert_eq!(y.len(), n * d, "target output matrix size mismatch");

    // Build X^T X (t × t) and X^T Y (t × d).
    // Only the lower triangle (k <= j) of X^T X is accumulated: the matrix is
    // symmetric, and Cholesky only reads the lower triangle. The blocked
    // Cholesky path extracts full diagonal blocks, so we mirror the lower
    // triangle to the upper once after the accumulation (cheaper than t²
    // redundant FMAs inside the n-loop for n ≥ 2). Halves the X^T X FMA count.
    let mut xtx = vec![0.0f32; t * t];
    let mut xty = vec![0.0f32; t * d];
    for i in 0..n {
        let x_row = &x[i * t..(i + 1) * t];
        let y_row = &y[i * d..(i + 1) * d];
        for j in 0..t {
            let x_ij = x_row[j];
            // Pre-slice the accumulator rows so the per-element bounds checks
            // hoist out of the inner loops. The k-walk order is unchanged for
            // both accumulators, so every partial sum is formed in the same
            // order → bit-identical (strict ordered adds — scalar on x86_64,
            // Issue 871).
            // X^T Y: full column (all d dims needed downstream).
            let xty_row = &mut xty[j * d..(j + 1) * d];
            for (acc, &y_k) in xty_row.iter_mut().zip(y_row.iter()) {
                *acc += x_ij * y_k;
            }
            // X^T X: lower triangle only (k <= j).
            let xtx_row = &mut xtx[j * t..=j * t + j];
            for (acc, &x_k) in xtx_row.iter_mut().zip(x_row.iter()) {
                *acc += x_ij * x_k;
            }
        }
    }
    // Mirror lower triangle → upper so the blocked Cholesky's full-block
    // extraction reads correct symmetric values.
    for j in 1..t {
        for k in 0..j {
            xtx[k * t + j] = xtx[j * t + k];
        }
    }

    // Apply ridge regularization if requested.
    if config.ridge_lambda > 0.0 {
        for j in 0..t {
            xtx[j * t + j] += config.ridge_lambda;
        }
    }

    // Try Cholesky; escalate jitter on failure.
    let mut jitter = 0.0f32;
    let mut solver_succeeded = true;
    loop {
        // Add jitter if needed.
        if jitter > 0.0 {
            for j in 0..t {
                xtx[j * t + j] += jitter;
            }
        }
        let mut l = vec![0.0f32; t * t];
        if try_cholesky_strict_f32(&mut l, &xtx, t).is_ok() {
            // Shared leaf solve: L L^T Cv = X^T Y per RHS column.
            // Issue 995: the factorisation + solve are linalg's kernels
            // (SIMD-interior dots + relative-tolerance pivot clamp); the
            // solution is graded by the G5 relative-error floors (no bit-pin).
            let mut cv = vec![0.0f32; t * d];
            let mut z_scratch = vec![0.0f32; t * d];
            chol_solve_f32(&mut cv, &mut z_scratch, &l, &xty, t, d);
            let rel_err = compute_relative_error(&cv, x, y, n, t, d);
            return ValueFitResult {
                compact_values: cv,
                relative_error: rel_err,
                solver_succeeded,
                jitter_used: jitter,
            };
        } else {
            solver_succeeded = false;
            let next_jitter = if jitter == 0.0 {
                config.cholesky_jitter
            } else {
                jitter * 10.0
            };
            if next_jitter > 1.0 {
                // Final fallback: pseudoinverse via normal equations diagonal-loading.
                // Add large diagonal and bail.
                for j in 0..t {
                    xtx[j * t + j] += 1.0;
                }
                // Try one more time with the heavy loading.
                let mut l = vec![0.0f32; t * t];
                if try_cholesky_strict_f32(&mut l, &xtx, t).is_ok() {
                    let mut cv = vec![0.0f32; t * d];
                    let mut z_scratch = vec![0.0f32; t * d];
                    chol_solve_f32(&mut cv, &mut z_scratch, &l, &xty, t, d);
                    let rel_err = compute_relative_error(&cv, x, y, n, t, d);
                    return ValueFitResult {
                        compact_values: cv,
                        relative_error: rel_err,
                        solver_succeeded: false,
                        jitter_used: 1.0,
                    };
                }
                // Give up — return zeros. This indicates a severely degenerate input.
                return ValueFitResult {
                    compact_values: vec![0.0; t * d],
                    relative_error: 1.0,
                    solver_succeeded: false,
                    jitter_used: 1.0,
                };
            }
            jitter = next_jitter;
        }
    }
}

#[inline]
fn compute_relative_error(cv: &[f32], x: &[f32], y: &[f32], n: usize, t: usize, d: usize) -> f32 {
    let mut residual_sq = 0.0f32;
    let mut y_norm_sq = 0.0f32;

    // Transpose cv from (t, d) row-major to (d, t) row-major so each output
    // dimension's coefficients are contiguous. The prior form did strided
    // `cv[j*d + k]` reads — one cache miss per j per k. For typical
    // (t=64, d=64, n=8) this is 32K strided reads; the transpose costs
    // 4K elements once and makes every subsequent `dot_8wide` sequential.
    let mut cv_t = vec![0.0f32; d * t];
    for j in 0..t {
        let cv_row = &cv[j * d..(j + 1) * d];
        for k in 0..d {
            cv_t[k * t + j] = cv_row[k];
        }
    }

    // Fused single-pass: compute X·Cv row entry inline and fold directly into
    // the residual / y-norm accumulators. Uses the shared 8-wide SIMD dot
    // kernel — DRY with the other score-matrix paths.
    for i in 0..n {
        let x_row = &x[i * t..(i + 1) * t];
        let y_row = &y[i * d..(i + 1) * d];
        for k in 0..d {
            let cv_row = &cv_t[k * t..(k + 1) * t];
            let s = dot_8wide(x_row, cv_row, t);
            let r = s - y_row[k];
            residual_sq += r * r;
            y_norm_sq += y_row[k] * y_row[k];
        }
    }
    if y_norm_sq < STABILITY_EPS {
        return 0.0;
    }
    (residual_sq / y_norm_sq).sqrt()
}

/// Block size for the blocked Cholesky algorithm. 32 is a common choice that
/// fits a 32×32 f32 block (4 KB) in L1 alongside two scratch buffers of the
/// same size (12 KB total) — well within typical 32 KB L1 caches.
pub const CHOLESKY_BLOCK_SIZE: usize = 32;

/// Helper: compute the X attention matrix from compact keys and β.
///
/// Given reference queries `Q`, compact keys `Ck`, and β, returns the `(n, t)`
/// attention matrix `X = softmax((q Ck^T + β) / √d)`. The caller typically uses
/// this directly as input to [`fit_cv_least_squares`].
pub fn compute_compact_attention(
    queries: &[f32],      // (n, d)
    compact_keys: &[f32], // (t, d)
    beta: &[f32],         // (t,)
    n: usize,
    t: usize,
    d: usize,
    x_out: &mut [f32], // (n, t)
) {
    assert_eq!(queries.len(), n * d);
    assert_eq!(compact_keys.len(), t * d);
    assert_eq!(beta.len(), t);
    assert_eq!(x_out.len(), n * t);

    // Compute raw scores S = Q · Ck^T · inv_sqrt_d + β (broadcast per row).
    let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
    for i in 0..n {
        let q_row = &queries[i * d..(i + 1) * d];
        let out_row = &mut x_out[i * t..(i + 1) * t];
        for j in 0..t {
            let k_row = &compact_keys[j * d..(j + 1) * d];
            // Strict ordered dot — scalar on x86_64 (Issue 871).
            let dot = dot_8wide(q_row, k_row, d);
            out_row[j] = dot * inv_sqrt_d + beta[j];
        }
    }

    // Apply max-shift + exp + normalize per row. Fused per-row (no scratch
    // allocation): the row is hot in L1 between the max scan and the exp
    // pass. Replaces `t` divisions per row with 1 reciprocal + `t` multiplies.
    for i in 0..n {
        let row = &mut x_out[i * t..(i + 1) * t];
        // Pass 1: per-row max (scalar accumulator, no buffer).
        let mut m = row[0];
        for &v in &row[1..] {
            m = m.max(v);
        }
        // Pass 2: shifted exp + accumulate sum_exp.
        let mut sum_exp = 0.0f32;
        for j in 0..t {
            let e = (row[j] - m).exp();
            row[j] = e;
            sum_exp += e;
        }
        // Pass 3: normalize via reciprocal-multiply.
        let denom = sum_exp.max(STABILITY_EPS);
        let inv_denom = 1.0 / denom;
        for v in row.iter_mut() {
            *v *= inv_denom;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_fit_recovers_known_solution() {
        // Construct a system where the answer is known.
        // n=4 queries, t=2 compact keys, d=3 dim.
        // If X Cv = Y, and X has rank t, we recover Cv exactly.
        let n = 4;
        let t = 2;
        let d = 3;
        // X: 4×2 — full column rank
        let x = vec![
            1.0f32, 0.0, //
            0.0, 1.0, //
            1.0, 1.0, //
            2.0, 1.0,
        ];
        // True Cv: 2×3
        let cv_true = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Compute Y = X Cv
        let mut y = vec![0.0f32; n * d];
        for i in 0..n {
            for k in 0..d {
                let mut s = 0.0f32;
                for j in 0..t {
                    s += x[i * t + j] * cv_true[j * d + k];
                }
                y[i * d + k] = s;
            }
        }
        let cfg = ValueFitConfig::default();
        let result = fit_cv_least_squares(&x, &y, n, t, d, &cfg);
        assert!(
            result.solver_succeeded,
            "solver should succeed on full-rank system (jitter={})",
            result.jitter_used
        );
        for j in 0..t {
            for k in 0..d {
                let got = result.compact_values[j * d + k];
                let exp = cv_true[j * d + k];
                assert!(
                    (got - exp).abs() < 1e-3,
                    "cv[{j},{k}] should be {exp}, got {got}"
                );
            }
        }
        assert!(
            result.relative_error < 1e-4,
            "relative error {} too large for exact-recovery system",
            result.relative_error
        );
    }

    #[test]
    fn test_cv_fit_handles_rank_deficient() {
        // n=1 query, t=3 keys — underdetermined. Should fall back to jitter.
        let n = 1;
        let t = 3;
        let d = 2;
        let x = vec![1.0f32, 1.0, 1.0]; // (1,3)
        let y = vec![2.0f32, 4.0]; // (1,2)
        let cfg = ValueFitConfig::default();
        let result = fit_cv_least_squares(&x, &y, n, t, d, &cfg);
        // Solver should still return something (with jitter).
        assert_eq!(result.compact_values.len(), t * d);
        // Reconstruction error should be small (we have many degrees of freedom).
        assert!(
            result.relative_error < 0.1,
            "rank-deficient system should still reconstruct Y well"
        );
    }

    #[test]
    fn test_cholesky_spd() {
        // [[2,1],[1,2]] is SPD. Issue 995: pins the shared leaf kernel.
        let a = vec![2.0f32, 1.0, 1.0, 2.0];
        let mut l = vec![0.0f32; 4];
        try_cholesky_strict_f32(&mut l, &a, 2).expect("SPD matrix");
        // Verify L L^T == A
        let r00 = l[0] * l[0];
        let r01 = l[2] * l[0];
        let r11 = l[2] * l[2] + l[3] * l[3];
        assert!((r00 - 2.0).abs() < 1e-5);
        assert!((r01 - 1.0).abs() < 1e-5);
        assert!((r11 - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_compact_attention_normalizes() {
        let n = 1;
        let t = 4;
        let d = 2;
        let queries = vec![1.0f32, 0.0];
        let compact_keys = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let beta = vec![0.0f32; t];
        let mut x = vec![0.0f32; n * t];
        compute_compact_attention(&queries, &compact_keys, &beta, n, t, d, &mut x);
        // All keys identical → uniform attention 0.25.
        let sum: f32 = x.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1, got {sum}"
        );
        for &v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    /// T2.4 lineage — large-`t` fit must hold reconstruction through the
    /// shared kernel (the retired local blocked/unblocked split is gone;
    /// `CHOLESKY_BLOCK_SIZE` survives as the trace's eligibility metric).
    /// Synthetic SPD matrix `A = M·M^T + n·I` (with random M) is PD by
    /// construction; the factorised `L` must satisfy `L·L^T == A` to within
    /// numerical tolerance.
    #[test]
    fn test_cholesky_blocked_path_activated() {
        let t = CHOLESKY_BLOCK_SIZE + 8; // the historical blocked threshold
        let mut seed = 31415u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        // M is (t, t) random.
        let m: Vec<f32> = (0..t * t).map(|_| rng()).collect();
        // A = M·M^T + t·I (diagonal loading makes it strictly PD).
        let mut a = vec![0.0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                let mut s = 0.0f32;
                for k in 0..t {
                    s += m[i * t + k] * m[j * t + k];
                }
                a[i * t + j] = s;
            }
            a[i * t + i] += t as f32;
        }
        let mut l = vec![0.0f32; t * t];
        try_cholesky_strict_f32(&mut l, &a, t).expect("A is PD by construction");
        // Verify L·L^T == A.
        for i in 0..t {
            for j in 0..t {
                let mut s = 0.0f32;
                let k_max = (i + 1).max(j + 1);
                for k in 0..k_max {
                    let li = if k <= i { l[i * t + k] } else { 0.0 };
                    let lj = if k <= j { l[j * t + k] } else { 0.0 };
                    s += li * lj;
                }
                let a_ij = a[i * t + j];
                assert!(
                    (s - a_ij).abs() < 1e-3 * a_ij.abs().max(1.0),
                    "L·L^T mismatch at ({i},{j}): got {s} want {a_ij}"
                );
            }
        }
    }

    /// Regression test for the Cholesky numerical cliff-edge (commit `f978a20b`).
    ///
    /// When `dot_8wide` (8-accumulator SIMD) was used for the Cholesky inner
    /// dot products, the different summation order caused ULP-level differences
    /// that flipped near-zero pivots negative on rank-deficient `X^T X`, triggering
    /// the jitter fallback and degrading solution quality 10×. This test
    /// constructs a near-singular system (duplicate columns → low-rank `X^T X`)
    /// and asserts the solver produces a high-quality solution. If someone
    /// re-introduces `dot_8wide` into the Cholesky inner loops, the jitter
    /// fallback will kick in unnecessarily and the solution quality will degrade
    /// well above the 1e-3 threshold.
    #[test]
    fn test_cholesky_near_singular_quality_guard() {
        // Build a rank-4 X (n=8, t=8) where columns come in duplicate pairs.
        // X^T X is 8×8 with rank 4 — near-singular, exactly the cliff-edge
        // scenario. The system is consistent (Y is in the column space), so a
        // correct solver should reconstruct Y to high accuracy.
        let n = 8usize;
        let t = 8usize;
        let d = 4usize;
        let mut x = vec![0.0f32; n * t];
        let mut seed = 42u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32)
        };
        for i in 0..n {
            // 4 unique values, each duplicated into 2 adjacent columns.
            let v0 = rng();
            let v1 = rng();
            let v2 = rng();
            let v3 = rng();
            x[i * t] = v0;
            x[i * t + 1] = v0;
            x[i * t + 2] = v1;
            x[i * t + 3] = v1;
            x[i * t + 4] = v2;
            x[i * t + 5] = v2;
            x[i * t + 6] = v3;
            x[i * t + 7] = v3;
        }
        // Build a consistent Y = X · Cv_true so the least-squares residual is
        // exactly zero in the column space.
        let cv_true: Vec<f32> = (0..t * d).map(|k| (k as f32) * 0.1).collect();
        let mut y = vec![0.0f32; n * d];
        for i in 0..n {
            for k in 0..d {
                let mut s = 0.0f32;
                for j in 0..t {
                    s += x[i * t + j] * cv_true[j * d + k];
                }
                y[i * d + k] = s;
            }
        }
        let cfg = ValueFitConfig::default();
        let result = fit_cv_least_squares(&x, &y, n, t, d, &cfg);
        // The solution should reconstruct Y to high accuracy (<1e-3). This
        // guards the Issue 995 solver swap: the shared leaf kernel must hold
        // the same near-singular quality the retired scalar copy delivered
        // (the `f978a20b` cliff-edge class). If the relative-error floors are
        // breached here, the G5 gate posture (relative-error, never bit-pin)
        // is the adjudicator — see the module numerics note.
        assert!(
            result.relative_error < 1e-3,
            "near-singular consistent system should reconstruct with <1e-3 error, got {}",
            result.relative_error
        );
    }
}
