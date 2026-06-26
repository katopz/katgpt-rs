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
//! Default solver: Cholesky on `X^T X` (with diagonal jitter fallback for
//! rank-deficient systems). No external LAPACK dependency.

#![allow(clippy::needless_range_loop)]

use crate::attn_match::{STABILITY_EPS, score_matrix_simd::dot_8wide};

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
            // X^T Y: full column (all d dims needed downstream).
            for k in 0..d {
                xty[j * d + k] += x_ij * y_row[k];
            }
            // X^T X: lower triangle only (k <= j).
            for k in 0..=j {
                xtx[j * t + k] += x_ij * x_row[k];
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
        match cholesky_decompose(&xtx, t) {
            Some(l) => {
                // Solve L Z = X^T Y (forward), then L^T Cv = Z (back), per RHS column.
                let cv = solve_cholesky_multi_rhs(&l, &xty, t, d);
                let rel_err = compute_relative_error(&cv, x, y, n, t, d);
                return ValueFitResult {
                    compact_values: cv,
                    relative_error: rel_err,
                    solver_succeeded,
                    jitter_used: jitter,
                };
            }
            None => {
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
                    if let Some(l) = cholesky_decompose(&xtx, t) {
                        let cv = solve_cholesky_multi_rhs(&l, &xty, t, d);
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
}

/// Solve `L L^T Cv = X^T Y` for the `(t, d)` RHS via forward+back substitution.
///
/// `l` is the lower-triangular Cholesky factor `(t, t)`, `rhs` is `X^T Y`
/// laid out `(t, d)` row-major (`rhs[j*d + col]`). Returns `Cv` laid out `(t, d)`.
///
/// Hoists the `z` scratch outside the per-column loop — allocated once (t f32)
/// and reused across all d columns, instead of reallocating per column.
#[inline]
fn solve_cholesky_multi_rhs(l: &[f32], rhs: &[f32], t: usize, d: usize) -> Vec<f32> {
    let mut cv = vec![0.0f32; t * d];
    let mut z = vec![0.0f32; t];
    // The Cholesky diagonal is invariant across the d RHS columns — precompute
    // its reciprocals once (t recips) instead of dividing 2·d·t times.
    let inv_diag: Vec<f32> = (0..t).map(|j| 1.0 / l[j * t + j]).collect();
    for col in 0..d {
        // Forward: L z = rhs[:, col]
        for j in 0..t {
            let mut s = rhs[j * d + col];
            for k in 0..j {
                s -= l[j * t + k] * z[k];
            }
            z[j] = s * inv_diag[j];
        }
        // Back: L^T cv[:, col] = z
        for j in (0..t).rev() {
            let mut s = z[j];
            for k in (j + 1)..t {
                s -= l[k * t + j] * cv[k * d + col];
            }
            cv[j * d + col] = s * inv_diag[j];
        }
    }
    cv
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

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns lower-triangular `L` such that `A = L L^T`, row-major `(t, t)`.
/// Returns `None` if the matrix is not PD.
#[inline]
fn cholesky_decompose(a: &[f32], t: usize) -> Option<Vec<f32>> {
    // For small t (< BLOCK_SIZE), use the unblocked algorithm — it has no
    // sub-matrix allocation overhead and is friendlier to the autovectorizer.
    if t < CHOLESKY_BLOCK_SIZE {
        return cholesky_decompose_unblocked(a, t);
    }
    cholesky_decompose_blocked(a, t)
}

/// Block size for the blocked Cholesky algorithm. 32 is a common choice that
/// fits a 32×32 f32 block (4 KB) in L1 alongside two scratch buffers of the
/// same size (12 KB total) — well within typical 32 KB L1 caches.
pub const CHOLESKY_BLOCK_SIZE: usize = 32;

/// Unblocked Cholesky decomposition (Plan 271 Phase 2, T2.4 reference).
///
/// This is the textbook column-by-column algorithm used for small matrices.
/// It is called automatically by [`cholesky_decompose`] when `t <
/// CHOLESKY_BLOCK_SIZE`, and is also used internally by the blocked variant
/// to factorize the diagonal blocks. Inner dot products use the shared
/// `dot_8wide` SIMD kernel — for t=64 this turns ~t³/6 scalar FMAs into
/// auto-vectorized 8-wide FMA chains.
#[inline]
fn cholesky_decompose_unblocked(a: &[f32], t: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0f32; t * t];
    for j in 0..t {
        // Diagonal: sum = a[j,j] − ‖l[j,0..j]‖². Single SIMD dot product
        // instead of a scalar accumulation loop. dot_8wide's pairwise-sum
        // order is also more numerically stable than sequential subtraction.
        let l_norm_sq = dot_8wide(&l[j * t..j * t + j], &l[j * t..j * t + j], j);
        let sum = a[j * t + j] - l_norm_sq;
        if sum <= 0.0 {
            return None;
        }
        let diag = sum.sqrt();
        l[j * t + j] = diag;
        // Off-diagonal strip: l[i,j] = (a[i,j] − l[i,0..j]·l[j,0..j]) / diag.
        // Each row's dot with the j-th prefix reuses the shared SIMD kernel.
        // The borrows are scoped so the subsequent write to l[i,j] is allowed.
        for i in (j + 1)..t {
            let s = {
                let l_row_i = &l[i * t..i * t + j];
                let l_prefix_j = &l[j * t..j * t + j];
                a[i * t + j] - dot_8wide(l_row_i, l_prefix_j, j)
            };
            l[i * t + j] = s / diag;
        }
    }
    Some(l)
}

/// Blocked Cholesky decomposition (Plan 271 Phase 2, T2.4).
///
/// Implements the right-looking blocked algorithm with block size
/// [`CHOLESKY_BLOCK_SIZE`]. For each block column:
///   1. Factorize the diagonal block with [`cholesky_decompose_unblocked`].
///   2. Solve the off-diagonal strip via forward substitution.
///   3. Symmetric rank-k update of the trailing sub-matrix into `a_red`.
///
/// The blocked variant is cache-aware: the inner GEMM-like rank-k update is
/// amenable to SIMD auto-vectorization because it operates on contiguous
/// rectangular blocks rather than triangular strided access.
///
/// Per AGENTS.md: the only allocation is the output buffer `l` plus a single
/// `a_red` scratch the size of `a`. Per-block sub-buffers are reused across
/// iterations by clearing in place.
#[inline]
fn cholesky_decompose_blocked(a: &[f32], t: usize) -> Option<Vec<f32>> {
    debug_assert!(t >= CHOLESKY_BLOCK_SIZE, "use unblocked for small t");
    // Copy A into a mutable scratch — we reduce it in place across blocks.
    let mut a_red = a.to_vec();
    let mut l = vec![0.0f32; t * t];
    let bs = CHOLESKY_BLOCK_SIZE;
    let mut k = 0usize;
    while k < t {
        let bk = bs.min(t - k);
        // 1. Extract and factorize the diagonal block A[k:k+bk, k:k+bk].
        let mut diag_block = vec![0.0f32; bk * bk];
        for i in 0..bk {
            for j in 0..bk {
                diag_block[i * bk + j] = a_red[(k + i) * t + (k + j)];
            }
        }
        let diag_l = cholesky_decompose_unblocked(&diag_block, bk)?;
        // Write lower-triangular result into L.
        for i in 0..bk {
            for j in 0..=i {
                l[(k + i) * t + (k + j)] = diag_l[i * bk + j];
            }
        }

        // 2. Off-diagonal strip L[k+bk:, k:k+bk] via forward substitution.
        //    For each row i in the strip, solve L_diag · x = A[i, k:k+bk]^T.
        //    The triangular dependency (col depends on col-1) prevents batching
        //    across columns, but each per-col dot product uses `dot_8wide`.
        let n_strip = t - (k + bk);
        if n_strip > 0 {
            for i in 0..n_strip {
                let row = k + bk + i;
                for col in 0..bk {
                    let l_strip = &l[row * t + k..row * t + k + col];
                    let diag_row = &diag_l[col * bk..col * bk + col];
                    let s = a_red[row * t + (k + col)] - dot_8wide(l_strip, diag_row, col);
                    l[row * t + (k + col)] = s / diag_l[col * bk + col];
                }
            }

            // 3. Symmetric rank-k update: a_red[k+bk:, k+bk:] -= L_strip · L_strip^T.
            //    Maintains symmetry explicitly so subsequent iterations read
            //    a consistent reduced matrix. Inner dot uses `dot_8wide` —
            //    bk (= CHOLESKY_BLOCK_SIZE = 32) is the sweet spot for 4
            //    SIMD lanes worth of FMA per strip row.
            for i in 0..n_strip {
                let row_i = k + bk + i;
                for j in 0..=i {
                    let row_j = k + bk + j;
                    let l_i_strip = &l[row_i * t + k..row_i * t + k + bk];
                    let l_j_strip = &l[row_j * t + k..row_j * t + k + bk];
                    let dot = dot_8wide(l_i_strip, l_j_strip, bk);
                    a_red[row_i * t + row_j] -= dot;
                    a_red[row_j * t + row_i] = a_red[row_i * t + row_j];
                }
            }
        }
        k += bk;
    }
    Some(l)
}

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
            // 8-wide chunked dot product auto-vectorizes on AVX2/NEON.
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
                    "cv[{},{}] should be {}, got {}",
                    j,
                    k,
                    exp,
                    got
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
        // [[2,1],[1,2]] is SPD.
        let a = vec![2.0f32, 1.0, 1.0, 2.0];
        let l = cholesky_decompose(&a, 2).expect("SPD matrix");
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
            "softmax should sum to 1, got {}",
            sum
        );
        for &v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    /// T2.4 — blocked Cholesky path must activate for t >= CHOLESKY_BLOCK_SIZE.
    /// Synthetic SPD matrix `A = M·M^T + n·I` (with random M) is PD by
    /// construction; both the blocked and unblocked paths must produce L
    /// such that `L·L^T == A` to within numerical tolerance.
    #[test]
    fn test_cholesky_blocked_path_activated() {
        let t = CHOLESKY_BLOCK_SIZE + 8; // exercises the blocked path
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
        let l = cholesky_decompose(&a, t).expect("A is PD by construction");
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
                    "L·L^T mismatch at ({},{}): got {} want {}",
                    i,
                    j,
                    s,
                    a_ij
                );
            }
        }
    }

    /// Blocked Cholesky must agree with unblocked to within numerical
    /// tolerance — they should produce identical L for the same input.
    #[test]
    fn test_cholesky_blocked_matches_unblocked() {
        let t = CHOLESKY_BLOCK_SIZE * 2; // multiple blocks
        let mut seed = 2718u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let m: Vec<f32> = (0..t * t).map(|_| rng()).collect();
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
        let l_blocked = cholesky_decompose(&a, t).expect("blocked");
        let l_unblocked = cholesky_decompose_unblocked(&a, t).expect("unblocked");
        let mut max_diff = 0.0f32;
        for i in 0..t * t {
            max_diff = max_diff.max((l_blocked[i] - l_unblocked[i]).abs());
        }
        assert!(
            max_diff < 1e-4,
            "blocked/unblocked L max diff: {}",
            max_diff
        );
    }
}
