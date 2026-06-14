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

use crate::attn_match::{score_matrix::row_max, STABILITY_EPS};

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
    let mut xtx = vec![0.0f32; t * t];
    let mut xty = vec![0.0f32; t * d];
    for i in 0..n {
        let x_row = &x[i * t..(i + 1) * t];
        let y_row = &y[i * d..(i + 1) * d];
        // X^T X outer-product accumulation.
        for j in 0..t {
            let x_ij = x_row[j];
            // X^T Y
            for k in 0..d {
                xty[j * d + k] += x_ij * y_row[k];
            }
            // X^T X (symmetric)
            for k in 0..t {
                xtx[j * t + k] += x_ij * x_row[k];
            }
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
                // Solve L Z = X^T Y (forward substitution), then L^T Cv = Z (back substitution).
                let mut cv = vec![0.0f32; t * d];
                // For each column of X^T Y (= each dim of d), solve the triangular system.
                for col in 0..d {
                    // Forward: L z = xty[:, col]
                    let mut z = vec![0.0f32; t];
                    for j in 0..t {
                        let mut s = xty[j * d + col];
                        for k in 0..j {
                            s -= l[j * t + k] * z[k];
                        }
                        z[j] = s / l[j * t + j];
                    }
                    // Back: L^T cv[:, col] = z
                    for j in (0..t).rev() {
                        let mut s = z[j];
                        for k in (j + 1)..t {
                            s -= l[k * t + j] * cv[k * d + col];
                        }
                        cv[j * d + col] = s / l[j * t + j];
                    }
                }
                // Compute reconstruction error.
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
                        let mut cv = vec![0.0f32; t * d];
                        for col in 0..d {
                            let mut z = vec![0.0f32; t];
                            for j in 0..t {
                                let mut s = xty[j * d + col];
                                for k in 0..j {
                                    s -= l[j * t + k] * z[k];
                                }
                                z[j] = s / l[j * t + j];
                            }
                            for j in (0..t).rev() {
                                let mut s = z[j];
                                for k in (j + 1)..t {
                                    s -= l[k * t + j] * cv[k * d + col];
                                }
                                cv[j * d + col] = s / l[j * t + j];
                            }
                        }
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

#[inline]
fn compute_relative_error(
    cv: &[f32],
    x: &[f32],
    y: &[f32],
    n: usize,
    t: usize,
    d: usize,
) -> f32 {
    let mut residual_sq = 0.0f32;
    let mut y_norm_sq = 0.0f32;
    let mut xcv = vec![0.0f32; d];
    for i in 0..n {
        let x_row = &x[i * t..(i + 1) * t];
        let y_row = &y[i * d..(i + 1) * d];
        // X Cv row = sum_j x_row[j] * cv[j, :]
        for k in 0..d {
            let mut s = 0.0f32;
            for j in 0..t {
                s += x_row[j] * cv[j * d + k];
            }
            xcv[k] = s;
        }
        for k in 0..d {
            let r = xcv[k] - y_row[k];
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
    let mut l = vec![0.0f32; t * t];
    for j in 0..t {
        let mut sum = a[j * t + j];
        for k in 0..j {
            sum -= l[j * t + k] * l[j * t + k];
        }
        if sum <= 0.0 {
            return None;
        }
        let diag = sum.sqrt();
        l[j * t + j] = diag;
        for i in (j + 1)..t {
            let mut s = a[i * t + j];
            for k in 0..j {
                s -= l[i * t + k] * l[j * t + k];
            }
            l[i * t + j] = s / diag;
        }
    }
    Some(l)
}

/// Helper: compute the X attention matrix from compact keys and β.
///
/// Given reference queries `Q`, compact keys `Ck`, and β, returns the `(n, t)`
/// attention matrix `X = softmax((q Ck^T + β) / √d)`. The caller typically uses
/// this directly as input to [`fit_cv_least_squares`].
pub fn compute_compact_attention(
    queries: &[f32], // (n, d)
    compact_keys: &[f32], // (t, d)
    beta: &[f32], // (t,)
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
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += q_row[k] * k_row[k];
            }
            out_row[j] = dot * inv_sqrt_d + beta[j];
        }
    }

    // Apply max-shift + exp + normalize per row.
    let mut maxes = vec![f32::NEG_INFINITY; n];
    row_max(x_out, n, t, &mut maxes);
    for i in 0..n {
        let m = maxes[i];
        let row = &mut x_out[i * t..(i + 1) * t];
        let mut sum_exp = 0.0f32;
        for j in 0..t {
            let e = (row[j] - m).exp();
            row[j] = e;
            sum_exp += e;
        }
        let denom = if sum_exp < STABILITY_EPS {
            STABILITY_EPS
        } else {
            sum_exp
        };
        for j in 0..t {
            row[j] /= denom;
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
        let cv_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
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
        assert!(result.relative_error < 0.1, "rank-deficient system should still reconstruct Y well");
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
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {}", sum);
        for &v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }
}
