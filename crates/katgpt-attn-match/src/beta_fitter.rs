//! β (beta) fitting via Nonnegative Least Squares (NNLS).
//!
//! Solves:
//! ```text
//! min_{w ≥ 0} || A w − m ||²
//! ```
//! where `A_ij = exp(q_i (Ck)_j^T / √d)` (the mass feature matrix),
//! `m_i = Σ_k exp(q_i K_k^T / √d)` (the target mass per query),
//! and `β_j = log(w_j)` is the per-token additive attention bias.
//!
//! Per the paper (Appendix C.2, Algorithm 3): projected gradient descent with
//! fixed step size `η = 1/L`, `L ≈ ||M||²` estimated via a few power-iteration
//! steps. Warm-started from a clamped closed-form least-squares solution.
//!
//! # Solver note (Issue 995)
//!
//! This module KEEPS its local scalar Cholesky + substitution solve — a
//! reviewed exception to the Issue 995 dedup, not a forgotten duplicate.
//! The β solve feeds a nested near-degenerate chain: β's ULP-level
//! summation-order differences perturb the softmax rows, and the
//! downstream `value_fitter` Cv system (itself rank-deficient at
//! G5's shape) amplifies the perturbation ~10x. Measured on the G5
//! fixture (`bench_271_attn_match_goat`): local scalar β solve +
//! shared-kernel Cv solve = 0.0102 relative error; shared-kernel β solve
//! = 0.0708 (G5 FAIL vs the 0.05 floor; the all-local baseline is
//! 0.0071). The scalar discipline here is load-bearing — the `f978a20b`
//! lesson applies to the WHOLE chain, not just one factorisation. If
//! this copy is ever retired, the shared family needs a scalar-discipline
//! twin and the G5 gate must re-grade the nested chain end to end.
//!
//! Box constraints (`w_lower ≤ w_j ≤ w_upper`) match the paper's stability
//! bounds to prevent degenerate solutions where one key absorbs all mass.

// Index-based loops are intentional for numerical clarity in this NNLS kernel.
#![allow(clippy::needless_range_loop)]

use crate::{STABILITY_EPS, score_matrix_simd::dot_8wide};

/// Configuration for β fitting.
#[derive(Debug, Clone, Copy)]
pub struct BetaFitConfig {
    /// Projected gradient descent iterations (0 = closed-form clamped LS only).
    pub iters: usize,
    /// Lower bound on `w = exp(β)`.
    pub w_lower: f32,
    /// Upper bound on `w = exp(β)`.
    pub w_upper: f32,
    /// Power-iteration steps for estimating `L ≈ ||M||²`.
    pub power_iter_steps: usize,
}

impl Default for BetaFitConfig {
    fn default() -> Self {
        Self {
            iters: 2,
            w_lower: 1e-3,
            w_upper: 20.0855, // e^3
            power_iter_steps: 3,
        }
    }
}

/// Result of β fitting: the recovered weights `w = exp(β)` and the β values.
#[derive(Debug, Clone)]
pub struct BetaFitResult {
    /// β values (length `t`). `β_j = log(w_j)`.
    pub beta: Vec<f32>,
    /// Weights `w = exp(β)` (length `t`).
    pub weights: Vec<f32>,
    /// Relative mass error `||A w − m||_2 / ||m||_2`.
    pub relative_error: f32,
}

/// Fit β via NNLS on the mass feature matrix.
///
/// # Arguments
/// * `a` - The `(n, t)` mass feature matrix, row-major. `A_ij = exp(q_i Ck_j^T / √d)`.
/// * `m` - The target mass vector, length `n`. `m_i = Σ_k exp(q_i K_k^T / √d)`.
/// * `n` - Number of reference queries.
/// * `t` - Number of compact keys.
/// * `config` - Solver configuration.
pub fn fit_beta_nnls(
    a: &[f32],
    m: &[f32],
    n: usize,
    t: usize,
    config: &BetaFitConfig,
) -> BetaFitResult {
    assert_eq!(a.len(), n * t, "mass feature matrix size mismatch");
    assert_eq!(m.len(), n, "target mass vector size mismatch");
    assert!(config.w_lower > 0.0, "w_lower must be > 0");

    let mut w = vec![0.0f32; t];

    // Step 1: Closed-form clamped least squares (warm start).
    // Solve normal equations: (A^T A) w = A^T m, then clamp to [w_lower, w_upper].
    // This is a small (t×t) system; we compute A^T A and A^T m in one pass.
    // Build A^T A (t × t, symmetric) and A^T m (t). Only the lower triangle
    // (k <= j) of A^T A is accumulated — Cholesky only reads the lower
    // triangle, so this halves the inner FMA count. We mirror to the upper
    // once at the end for robustness (keeps the matrix consistent if a future
    // blocked Cholesky reads full diagonal blocks).
    let mut ata = vec![0.0f32; t * t];
    let mut atm = vec![0.0f32; t];
    for i in 0..n {
        let row = &a[i * t..(i + 1) * t];
        let mi = m[i];
        // Split the fused j-loop into (a) the A^T m axpy and (b) the A^T A
        // triangular update. Each `atm[j]` / `ata[j][k]` cell is still touched
        // exactly once per `i` and the outer `i` order is unchanged, so every
        // accumulator sees the same addend sequence → bit-identical. Both
        // loops now walk pre-sliced rows, so the bounds checks hoist out.
        for (acc, &r_j) in atm.iter_mut().zip(row.iter()) {
            *acc += r_j * mi;
        }
        for j in 0..t {
            let row_j = row[j];
            let ata_row = &mut ata[j * t..=j * t + j];
            for (acc, &r_k) in ata_row.iter_mut().zip(row.iter()) {
                *acc += row_j * r_k;
            }
        }
    }
    // Mirror lower triangle → upper.
    for j in 1..t {
        for k in 0..j {
            ata[k * t + j] = ata[j * t + k];
        }
    }

    // Solve via Cholesky; if rank-deficient, add diagonal jitter.
    // We mutate `ata` in place (adding/removing jitter each attempt) to avoid
    // a per-iteration `clone()` of the (t*t) buffer in the hot retry loop.
    let mut jitter = 0.0f32;
    let l_mat;
    loop {
        if let Some(l) = cholesky_decompose(&ata, t) {
            l_mat = l;
            break;
        } else {
            // Remove previous jitter before adding the new one.
            if jitter > 0.0 {
                for j in 0..t {
                    ata[j * t + j] -= jitter;
                }
            }
            jitter = if jitter == 0.0 { 1e-6 } else { jitter * 10.0 };
            for j in 0..t {
                ata[j * t + j] += jitter;
            }
            if jitter > 1e3 {
                // Fall back to a diagonal-only "solution": w = atm[j] / ata[j,j]
                for j in 0..t {
                    let diag = ata[j * t + j];
                    w[j] = if diag > STABILITY_EPS {
                        atm[j] / diag
                    } else {
                        config.w_lower
                    };
                }
                return finalize(w, a, m, n, t, config);
            }
        }
    }
    // Forward substitution: L z = A^T m.
    let mut z = vec![0.0f32; t];
    for j in 0..t {
        let mut s = atm[j];
        // Pre-sliced row + zip: same k order (0..j) → bit-identical.
        let l_row = &l_mat[j * t..j * t + j];
        for (&l_jk, &z_k) in l_row.iter().zip(z[..j].iter()) {
            s -= l_jk * z_k;
        }
        z[j] = s / l_mat[j * t + j];
    }
    // Back substitution: L^T w = z.
    for j in (0..t).rev() {
        let mut s = z[j];
        for k in (j + 1)..t {
            s -= l_mat[k * t + j] * w[k];
        }
        w[j] = s / l_mat[j * t + j];
    }

    // Clamp to box constraints (initial closed-form solution). Branchless.
    for j in 0..t {
        w[j] = w[j].clamp(config.w_lower, config.w_upper);
    }

    // Step 2: Projected gradient descent (only if iters > 0).
    if config.iters > 0 {
        // Estimate L ≈ ||A||² (spectral norm squared) via power iteration on A^T A.
        let l_estimate = estimate_spectral_norm_squared(a, n, t, config.power_iter_steps);

        if l_estimate > STABILITY_EPS {
            let eta = 1.0f32 / l_estimate;
            // Reusable scratch buffers.
            let mut grad = vec![0.0f32; t];
            let mut aw = vec![0.0f32; n];
            for _ in 0..config.iters {
                // aw = A w  (matvec; reuse the shared SIMD dot kernel)
                let w_slice = &w[..t];
                for i in 0..n {
                    let row = &a[i * t..(i + 1) * t];
                    aw[i] = dot_8wide(row, w_slice, t);
                }
                // grad = A^T (A w - m). Iterate i in the outer loop so reads
                // from `a` are sequential (row-major). The prior j-outer form
                // did strided `a[i*t + j]` reads — cache-hostile for large n.
                grad.fill(0.0);
                for i in 0..n {
                    let row = &a[i * t..(i + 1) * t];
                    let r_i = aw[i] - m[i];
                    // zip over the pre-sliced row: identical j order, no
                    // per-element bounds check on `grad`/`row`.
                    for (g, &r_j) in grad.iter_mut().zip(row.iter()) {
                        *g += r_j * r_i;
                    }
                }
                // Step: w ← w - η * grad; clamp to box. Branchless clamp
                // (emits minss/maxss) — keeps the inner loop branch-free
                // for SIMD friendliness.
                for j in 0..t {
                    let new_w = w[j] - eta * grad[j];
                    w[j] = new_w.clamp(config.w_lower, config.w_upper);
                }
            }
        }
    }

    finalize(w, a, m, n, t, config)
}

#[inline]
fn finalize(
    mut w: Vec<f32>,
    a: &[f32],
    m: &[f32],
    n: usize,
    t: usize,
    config: &BetaFitConfig,
) -> BetaFitResult {
    // Final clamp (defensive). Branchless — emits minss/maxss.
    for j in 0..t {
        w[j] = w[j].clamp(config.w_lower, config.w_upper);
    }
    // β = log(w)
    let beta: Vec<f32> = w.iter().map(|&wi| wi.ln()).collect();
    // Relative mass error ||A w - m|| / ||m||
    let m_norm = vector_norm(m);
    let w_slice = &w[..t];
    let mut residual_norm_sq = 0.0f32;
    for i in 0..n {
        let row = &a[i * t..(i + 1) * t];
        let aw_i = dot_8wide(row, w_slice, t);
        let r = aw_i - m[i];
        residual_norm_sq += r * r;
    }
    let relative_error = if m_norm > STABILITY_EPS {
        (residual_norm_sq.sqrt()) / m_norm
    } else {
        0.0
    };
    BetaFitResult {
        beta,
        weights: w,
        relative_error,
    }
}

#[inline]
fn vector_norm(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns lower-triangular `L` such that `A = L L^T`, row-major `(t, t)`.
/// Returns `None` if the matrix is not PD (negative pivot encountered).
///
/// Inner dot products use scalar sequential accumulation, NOT the shared
/// `dot_8wide` SIMD kernel. See `value_fitter::cholesky_decompose_unblocked`
/// doc for the full rationale: Cholesky on near-singular matrices is a
/// numerical cliff edge where ULP-level summation-order differences can flip
/// a pivot from positive to negative, triggering the jitter fallback and
/// degrading solution quality 10×. The canonical failure was commit
/// `f978a20b` (G5 gate: 0.7% → 7.5% relative error).
#[inline]
fn cholesky_decompose(a: &[f32], t: usize) -> Option<Vec<f32>> {
    let mut l = vec![0.0f32; t * t];
    for j in 0..t {
        // Diagonal: a[j,j] − Σ_k l[j,k]². Sequential scalar subtraction.
        let mut sum = a[j * t + j];
        for &l_jk in &l[j * t..j * t + j] {
            sum -= l_jk * l_jk;
        }
        if sum <= 0.0 {
            return None;
        }
        let diag = sum.sqrt();
        l[j * t + j] = diag;
        // Off-diagonal: (a[i,j] − Σ_k l[i,k]·l[j,k]) / diag.
        // `split_at_mut((j+1)*t)` separates the read-only j-th row from the
        // mutable i>j rows so both can be pre-sliced. Same scalar k-order
        // (0..j) and the same subtraction sequence → bit-identical, which is
        // load-bearing here (see the doc comment above).
        let (l_head, l_tail) = l.split_at_mut((j + 1) * t);
        let l_j = &l_head[j * t..j * t + j];
        for (i_off, l_i) in l_tail.chunks_exact_mut(t).enumerate() {
            let i = j + 1 + i_off;
            let mut s = a[i * t + j];
            for (&l_ik, &l_jk) in l_i[..j].iter().zip(l_j.iter()) {
                s -= l_ik * l_jk;
            }
            l_i[j] = s / diag;
        }
    }
    Some(l)
}

/// Estimate `||A||²` (largest eigenvalue of A^T A) via power iteration on `A^T A`.
#[inline]
fn estimate_spectral_norm_squared(a: &[f32], n: usize, t: usize, steps: usize) -> f32 {
    // Start with all-ones vector v ∈ R^t, normalized.
    let mut v_norm = (t as f32).sqrt();
    if v_norm < STABILITY_EPS {
        return 1.0;
    }
    let mut v: Vec<f32> = vec![1.0f32 / v_norm; t];

    // Pre-allocate both scratch buffers outside the power-iteration loop to
    // avoid a per-step `Vec` allocation (`new_v` was previously reallocated
    // every iteration).
    let mut aw = vec![0.0f32; n];
    let mut new_v = vec![0.0f32; t];
    for _ in 0..steps.max(1) {
        // aw = A v  (matvec)
        let v_slice = &v[..t];
        for i in 0..n {
            let row = &a[i * t..(i + 1) * t];
            aw[i] = dot_8wide(row, v_slice, t);
        }
        // v = A^T aw  (reuse new_v instead of reallocating)
        new_v.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..n {
            let row = &a[i * t..(i + 1) * t];
            let aw_i = aw[i];
            // zip over the pre-sliced row: identical j order, no bounds check.
            for (acc, &r_j) in new_v.iter_mut().zip(row.iter()) {
                *acc += r_j * aw_i;
            }
        }
        v_norm = vector_norm(&new_v);
        if v_norm < STABILITY_EPS {
            return 1.0;
        }
        // Kept as a division (not a hoisted reciprocal-multiply) so the
        // quotient stays bit-identical.
        for (vj, &nvj) in v.iter_mut().zip(new_v.iter()) {
            *vj = nvj / v_norm;
        }
    }
    // Rayleigh quotient: ||A v||^2 ≈ lambda where lambda ≈ ||A||^2.
    // Recompute Av with the final v.
    let v_slice = &v[..t];
    let mut av_norm_sq = 0.0f32;
    for i in 0..n {
        let row = &a[i * t..(i + 1) * t];
        let s = dot_8wide(row, v_slice, t);
        av_norm_sq += s * s;
    }
    // av_norm_sq ≈ v^T A^T A v ≈ ||A||² (since ||v||=1).
    av_norm_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_fit_recovers_uniform() {
        // Construct A and m such that the optimal w is uniform.
        // 3 queries, 2 keys. m_i = sum of A_ij * w_j with w = [2, 2].
        let t = 2;
        let n = 3;
        let a = vec![1.0f32, 1.0, 0.5, 1.5, 2.0, 0.5]; // (3, 2)
        // m_i = A[i,0]*2 + A[i,1]*2
        let m: Vec<f32> = (0..n).map(|i| (a[i * t] + a[i * t + 1]) * 2.0).collect();
        let cfg = BetaFitConfig {
            iters: 5,
            w_lower: 1e-3,
            w_upper: 100.0,
            power_iter_steps: 5,
        };
        let result = fit_beta_nnls(&a, &m, n, t, &cfg);
        // Optimal w should be [2, 2], β = log(2) ≈ 0.693
        for j in 0..t {
            assert!(
                (result.weights[j] - 2.0).abs() < 0.2,
                "weight {} should be ~2.0, got {}",
                j,
                result.weights[j]
            );
            assert!(
                (result.beta[j] - 2.0f32.ln()).abs() < 0.1,
                "beta {} should be ~{}, got {}",
                j,
                2.0f32.ln(),
                result.beta[j]
            );
        }
        assert!(result.relative_error < 0.1);
    }

    #[test]
    fn test_beta_fit_box_constraints() {
        let t = 2;
        let n = 2;
        let a = vec![10.0f32, 0.0, 0.0, 10.0]; // diagonal — easy system
        let m = vec![100.0f32, 100.0]; // implies w = [10, 10]
        let cfg = BetaFitConfig {
            iters: 5,
            w_lower: 1e-3,
            w_upper: 5.0, // Force upper bound
            power_iter_steps: 3,
        };
        let result = fit_beta_nnls(&a, &m, n, t, &cfg);
        // w should be clamped to 5.0.
        for j in 0..t {
            assert!(
                (result.weights[j] - 5.0).abs() < 1e-3,
                "weight should be clamped to 5.0, got {}",
                result.weights[j]
            );
        }
    }

    #[test]
    fn test_beta_fit_zero_iters_clamped_ls() {
        let t = 3;
        let n = 4;
        // Identity-ish: A = diag-heavy
        let mut a = vec![0.0f32; n * t];
        for i in 0..n.min(t) {
            a[i * t + i] = 2.0;
        }
        // m_i = 2 * w_i, target w = [3, 1, 2]
        let w_true = [3.0f32, 1.0, 2.0];
        let mut m = vec![0.0f32; n];
        for i in 0..n.min(t) {
            m[i] = 2.0 * w_true[i];
        }
        let cfg = BetaFitConfig {
            iters: 0, // Pure clamped LS
            w_lower: 1e-3,
            w_upper: 100.0,
            power_iter_steps: 0,
        };
        let result = fit_beta_nnls(&a, &m, n, t, &cfg);
        for j in 0..t.min(n) {
            assert!(
                (result.weights[j] - w_true[j]).abs() < 0.1,
                "weight {} should be ~{}, got {}",
                j,
                w_true[j],
                result.weights[j]
            );
        }
    }

    #[test]
    fn test_cholesky_basic() {
        // A = [[4, 2], [2, 3]] is SPD.
        let a = vec![4.0f32, 2.0, 2.0, 3.0];
        let l = cholesky_decompose(&a, 2).expect("SPD matrix");
        // L L^T should equal A.
        let reconstructed_l_lt = [
            l[0] * l[0],
            l[2] * l[0],
            l[2] * l[0],
            l[2] * l[2] + l[3] * l[3],
        ];
        for i in 0..4 {
            assert!(
                (reconstructed_l_lt[i] - a[i]).abs() < 1e-5,
                "Cholesky reconstruction failed at {}: got {} expected {}",
                i,
                reconstructed_l_lt[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_cholesky_not_pd() {
        // Indefinite matrix.
        let a = vec![1.0f32, 2.0, 2.0, 1.0];
        let result = cholesky_decompose(&a, 2);
        assert!(result.is_none(), "should fail on non-PD matrix");
    }
}
