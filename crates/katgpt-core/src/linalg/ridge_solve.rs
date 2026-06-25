//! f32 ridge-regression kernels via Cholesky factorisation.
//!
//! Extracted from the same mathematical pattern as
//! `peira::predictor_with_scratch` (the `(N + λI)^{-1}` step in Plan 153), but
//! kept as a standalone f32 path rather than refactoring PEIRA's f64 numerics.
//! See the module-level note in [`super`] for the rationale.
//!
//! // TODO: unify with peira's f64 path once a generic-over-`T: Float` Cholesky
//! // is verified bit-identical to PEIRA's f64 specialisation (Plan 153 G4).
//!
//! # Numerical contract
//!
//! All routines assume the input Gram/covariance matrices are
//! symmetric positive (semi-)definite and write into caller-provided buffers.
//! `cholesky_f32` panics on a non-positive-definite leading minor — the ridge
//! diagonal `+λI` added by the caller is what guarantees positive-definiteness.
//! `λ > 0` is a hard precondition for every public entry point here.
//!
//! # Determinism
//!
//! These kernels are pure float arithmetic with no SIMD backend selection that
//! changes results across runs on the same CPU (the [`crate::simd`] dispatchers
//! pick a backend once and stick with it). Two identical inputs produce
//! bit-identical outputs on the same host — this is what backs KARC G4
//! (Plan 308 §GOAT gate).

use crate::simd::simd_dot_f32;

// ── f64 path (numerically robust, used by KARC's fit for small-λ regimes) ────
//
// f32 Cholesky can fail on near-singular Grams when λ is below f32 epsilon
// relative to the matrix scale (see `cholesky_f32` tolerance note). KARC's
// fit (a cold path) accumulates the Gram/covariance in f64 and solves via the
// f64 kernels below, casting the resulting Wout to f32 for the forecast matvec.
// This mirrors PEIRA's f64 numerics without touching PEIRA's code.

/// f64 dot product (scalar accumulation — cold path, no SIMD needed).
#[inline]
fn dot_f64(a: &[f64], b: &[f64], len: usize) -> f64 {
    let mut s = 0.0f64;
    let mut i = 0;
    while i + 4 <= len {
        s = a[i].mul_add(b[i], s);
        s = a[i + 1].mul_add(b[i + 1], s);
        s = a[i + 2].mul_add(b[i + 2], s);
        s = a[i + 3].mul_add(b[i + 3], s);
        i += 4;
    }
    while i < len {
        s = a[i].mul_add(b[i], s);
        i += 1;
    }
    s
}

/// f64 Cholesky factorisation `A = L·Lᵀ` of an SPD matrix `A` (`k×k`).
/// Relative-tolerance pivot clamp for near-singular inputs (see `cholesky_f32`).
#[inline]
pub fn cholesky_f64(l: &mut [f64], a: &[f64], k: usize) {
    let mut a_max = 1.0f64;
    for &v in a.iter().take(k * k) {
        let av = v.abs();
        if av > a_max {
            a_max = av;
        }
    }
    let tol = a_max * (k as f64) * 2.0f64.powi(-50);
    let floor = a_max * 2.0f64.powi(-52);
    for v in l.iter_mut().take(k * k) {
        *v = 0.0;
    }
    for j in 0..k {
        let j_row = j * k;
        let sum = if j > 0 {
            dot_f64(&l[j_row..j_row + j], &l[j_row..j_row + j], j)
        } else {
            0.0
        };
        let mut diag = a[j_row + j] - sum;
        if diag <= 0.0 {
            assert!(
                diag > -tol,
                "matrix not positive definite in cholesky_f64 (pivot {} < -{})",
                diag,
                tol
            );
            diag = floor;
        }
        let diag_sqrt = diag.sqrt();
        l[j_row + j] = diag_sqrt;
        let mut i = j + 1;
        while i < k {
            let i_row = i * k;
            let s = if j > 0 {
                dot_f64(&l[i_row..i_row + j], &l[j_row..j_row + j], j)
            } else {
                0.0
            };
            l[i_row + j] = (a[i_row + j] - s) / diag_sqrt;
            i += 1;
        }
    }
}

/// f64 forward substitution `L·Z = B` (L lower-triangular, row-major).
fn solve_lower_f64(z: &mut [f64], l: &[f64], b: &[f64], k: usize, n_rhs: usize) {
    for col in 0..n_rhs {
        for i in 0..k {
            let i_row = i * k;
            let mut s = b[i * n_rhs + col];
            let mut j = 0;
            while j < i {
                s -= l[i_row + j] * z[j * n_rhs + col];
                j += 1;
            }
            z[i * n_rhs + col] = s / l[i_row + i];
        }
    }
}

/// f64 back substitution `Lᵀ·X = Z` (reads previously-written `x[j]` for j>i).
fn solve_upper_f64(x: &mut [f64], l: &[f64], z: &[f64], k: usize, n_rhs: usize) {
    for col in 0..n_rhs {
        for ii in 0..k {
            let i = k - 1 - ii;
            let i_row = i * k;
            let mut s = z[i * n_rhs + col];
            let mut j = i + 1;
            while j < k {
                s -= l[j * k + i] * x[j * n_rhs + col];
                j += 1;
            }
            x[i * n_rhs + col] = s / l[i_row + i];
        }
    }
}

/// Direct f64 ridge solve: `Wᵀ = (XᵀX + λI)⁻¹ XᵀY`. Robust for small λ.
/// Writes the `d_h × n_out` solution into `w_t` (f64). Scratch buffers sized
/// `d_h*d_h` (l) and `d_h*n_out` (z).
#[inline]
pub fn ridge_solve_direct_f64(
    w_t: &mut [f64],
    l_scratch: &mut [f64],
    z_scratch: &mut [f64],
    gram_reg: &[f64],
    cov: &[f64],
    d_h: usize,
    n_out: usize,
) {
    cholesky_f64(l_scratch, gram_reg, d_h);
    solve_lower_f64(z_scratch, l_scratch, cov, d_h, n_out);
    solve_upper_f64(w_t, l_scratch, z_scratch, d_h, n_out);
}

/// Solve `L·Lᵀ·X = B` given a **pre-computed** f64 Cholesky factor `L`
/// (lower-triangular, `k×k`), writing the `k×n_rhs` solution into `x` (f64,
/// row-major). `z_scratch` must hold `k*n_rhs` f64 and is overwritten.
///
/// Used by Plan 308 Phase 2 ALS (`low_rank_fit`): factor `G+λI` once via
/// [`cholesky_f64`], then call this each ALS iteration for the B-step
/// back-substitution with `r` right-hand sides.
#[inline]
pub fn chol_solve_f64(
    x: &mut [f64],
    z_scratch: &mut [f64],
    l: &[f64],
    b: &[f64],
    k: usize,
    n_rhs: usize,
) {
    solve_lower_f64(z_scratch, l, b, k, n_rhs);
    solve_upper_f64(x, l, z_scratch, k, n_rhs);
}

/// Cholesky factorisation `A = L·Lᵀ` of an SPD matrix `A` (row-major, `k×k`).
///
/// Writes the lower-triangular factor (including the diagonal) into `l`;
/// the strict upper triangle of `l` is left untouched (callers should zero it
/// first or only read the lower triangle).
///
/// Panics if `A` is not positive definite beyond a small relative tolerance
/// (f32-precision safety margin: a pivot within `-k·ε·‖A‖_max` of zero is
/// clamped to a tiny positive floor so near-singular ridge Grams with small
/// `λ` do not spuriously fail). Inner dot products use [`simd_dot_f32`].
#[inline]
pub fn cholesky_f32(l: &mut [f32], a: &[f32], k: usize) {
    // Estimate the matrix scale for the relative tolerance.
    let mut a_max = 1.0f32;
    for &v in a.iter().take(k * k) {
        let av = v.abs();
        if av > a_max {
            a_max = av;
        }
    }
    // f32 epsilon ≈ 1.19e-7; allow a few ULPs of slack for accumulation noise.
    let tol = a_max * (k as f32) * 2.0f32.powi(-23);
    let floor = a_max * 2.0f32.powi(-24); // minimum pivot magnitude
    // Zero the lower-triangle scratch we are about to fill so unused entries
    // do not accidentally contribute to subsequent dot products.
    for v in l.iter_mut().take(k * k) {
        *v = 0.0;
    }
    for j in 0..k {
        let j_row = j * k;
        // Diagonal pivot.
        let sum = if j > 0 {
            simd_dot_f32(&l[j_row..j_row + j], &l[j_row..j_row + j], j)
        } else {
            0.0
        };
        let mut diag = a[j_row + j] - sum;
        if diag <= 0.0 {
            assert!(
                diag > -tol,
                "matrix not positive definite in cholesky_f32 (pivot {} < -{})",
                diag,
                tol
            );
            diag = floor; // clamp near-singular pivot
        }
        let diag_sqrt = diag.sqrt();
        l[j_row + j] = diag_sqrt;
        // Lower-triangle column below the diagonal.
        let mut i = j + 1;
        while i < k {
            let i_row = i * k;
            let s = if j > 0 {
                simd_dot_f32(&l[i_row..i_row + j], &l[j_row..j_row + j], j)
            } else {
                0.0
            };
            l[i_row + j] = (a[i_row + j] - s) / diag_sqrt;
            i += 1;
        }
    }
}

/// Solve `L·Lᵀ·X = B` given the Cholesky factor `L` (lower-triangular, `k×k`),
/// writing the solution `X` (`k×n_rhs`, row-major) into `x`.
///
/// Two triangular solves fused: forward-substitute `L·Z = B` (into
/// `z_scratch`), then back-substitute `Lᵀ·X = Z` (into `x`).
///
/// `z_scratch` must hold at least `k * n_rhs` floats and is overwritten.
#[inline]
pub fn chol_solve_f32(
    x: &mut [f32],
    z_scratch: &mut [f32],
    l: &[f32],
    b: &[f32],
    k: usize,
    n_rhs: usize,
) {
    solve_lower_triangular_strided(z_scratch, l, b, k, n_rhs);
    solve_upper_triangular_transposed_strided(x, l, z_scratch, k, n_rhs);
}

/// `L·Z = B` for lower-triangular `L` (row-major, only lower triangle read).
fn solve_lower_triangular_strided(z: &mut [f32], l: &[f32], b: &[f32], k: usize, n_rhs: usize) {
    for col in 0..n_rhs {
        for i in 0..k {
            let i_row = i * k;
            let mut s = b[i * n_rhs + col];
            let mut j = 0;
            while j + 4 <= i {
                s -= l[i_row + j] * z[j * n_rhs + col];
                s -= l[i_row + j + 1] * z[(j + 1) * n_rhs + col];
                s -= l[i_row + j + 2] * z[(j + 2) * n_rhs + col];
                s -= l[i_row + j + 3] * z[(j + 3) * n_rhs + col];
                j += 4;
            }
            while j < i {
                s -= l[i_row + j] * z[j * n_rhs + col];
                j += 1;
            }
            z[i * n_rhs + col] = s / l[i_row + i];
        }
    }
}

/// `Lᵀ·X = Z` for the transpose of lower-triangular `L`.
///
/// This is back-substitution: `x[i]` depends on `x[j]` for `j > i` (already
/// computed), NOT on `z[j]`. The output `x` is written top-down from the last
/// row; we read previously-written `x[j]` entries for the off-diagonal terms.
fn solve_upper_triangular_transposed_strided(
    x: &mut [f32],
    l: &[f32],
    z: &[f32],
    k: usize,
    n_rhs: usize,
) {
    for col in 0..n_rhs {
        // i runs from k-1 down to 0.
        for ii in 0..k {
            let i = k - 1 - ii;
            let i_row = i * k;
            let mut s = z[i * n_rhs + col];
            let mut j = i + 1;
            // Lᵀ[i,j] = L[j,i] for j > i; subtract L[j,i] * x[j] (already solved).
            while j + 4 <= k {
                s -= l[j * k + i] * x[j * n_rhs + col];
                s -= l[(j + 1) * k + i] * x[(j + 1) * n_rhs + col];
                s -= l[(j + 2) * k + i] * x[(j + 2) * n_rhs + col];
                s -= l[(j + 3) * k + i] * x[(j + 3) * n_rhs + col];
                j += 4;
            }
            while j < k {
                s -= l[j * k + i] * x[j * n_rhs + col];
                j += 1;
            }
            x[i * n_rhs + col] = s / l[i_row + i];
        }
    }
}

/// Compute the inverse of an SPD matrix `A` (`k×k`) via Cholesky.
///
/// Writes the `k×k` inverse into `inv`. Reuses `l_scratch` (`k×k`) and
/// `inv_l_scratch` (`k×k`) as workspace. The result is symmetric; both triangles
/// are filled.
#[inline]
pub fn spd_inverse_f32(
    inv: &mut [f32],
    l_scratch: &mut [f32],
    inv_l_scratch: &mut [f32],
    a: &[f32],
    k: usize,
) {
    cholesky_f32(l_scratch, a, k);
    // Invert lower-triangular L into inv_l_scratch.
    invert_lower_triangular_f32(inv_l_scratch, l_scratch, k);
    // inv = Lᵀ⁻¹ · L⁻¹ = (L⁻¹)ᵀ · (L⁻¹). With M = L⁻¹ (lower-tri), inv = Mᵀ · M.
    for i in 0..k {
        for j in 0..k {
            // inv[i,j] = sum_l M[l,i] * M[l,j]  (M is lower-tri, so l >= max(i,j))
            let lo = if i > j { i } else { j };
            let mut s = 0.0f32;
            let mut l = lo;
            while l + 4 <= k {
                s = inv_l_scratch[l * k + i].mul_add(inv_l_scratch[l * k + j], s);
                s = inv_l_scratch[(l + 1) * k + i].mul_add(
                    inv_l_scratch[(l + 1) * k + j],
                    s,
                );
                s = inv_l_scratch[(l + 2) * k + i].mul_add(
                    inv_l_scratch[(l + 2) * k + j],
                    s,
                );
                s = inv_l_scratch[(l + 3) * k + i].mul_add(
                    inv_l_scratch[(l + 3) * k + j],
                    s,
                );
                l += 4;
            }
            while l < k {
                s = inv_l_scratch[l * k + i].mul_add(inv_l_scratch[l * k + j], s);
                l += 1;
            }
            inv[i * k + j] = s;
        }
    }
}

/// Invert a lower-triangular matrix `L` (`k×k`) into `inv_l`.
/// Reads only the lower triangle of `l`; writes a full lower triangle (upper
/// stays zero).
fn invert_lower_triangular_f32(inv_l: &mut [f32], l: &[f32], k: usize) {
    for v in inv_l.iter_mut().take(k * k) {
        *v = 0.0;
    }
    for j in 0..k {
        inv_l[j * k + j] = 1.0 / l[j * k + j];
        for i in (j + 1)..k {
            let i_row = i * k;
            // inv_l[i,j] = -sum_{p=j}^{i-1} L[i,p] * inv_l[p,j] / L[i,i]
            let mut s = 0.0f32;
            let mut p = j;
            while p + 4 <= i {
                s = l[i_row + p].mul_add(inv_l[p * k + j], s);
                s = l[i_row + p + 1].mul_add(inv_l[(p + 1) * k + j], s);
                s = l[i_row + p + 2].mul_add(inv_l[(p + 2) * k + j], s);
                s = l[i_row + p + 3].mul_add(inv_l[(p + 3) * k + j], s);
                p += 4;
            }
            while p < i {
                s = l[i_row + p].mul_add(inv_l[p * k + j], s);
                p += 1;
            }
            inv_l[i_row + j] = -s / l[i_row + i];
        }
    }
}

/// Direct (feature-space) ridge solve.
///
/// Given a pre-accumulated Gram matrix `gram = XᵀX + λI` (`d_h × d_h`, SPD) and
/// cross-covariance `cov = XᵀY` (`d_h × n_out`, row-major), solves
/// `Wᵀ = (XᵀX + λI)⁻¹ · XᵀY` and writes the `d_h × n_out` solution into `w_t`.
/// The caller typically transposes this to row-major `n_out × d_h` for the
/// forecast matvec.
///
/// This is the form to use when `d_h ≤ N` (feature count ≤ sample count),
/// matching paper Eq. (14) directly. For the `d_h > N` regime, use
/// [`ridge_solve_woodbury_f32`] which inverts the smaller sample-space matrix.
///
/// Scratch buffers (`l_scratch`, `z_scratch`) must each hold `d_h * d_h` and
/// `d_h * n_out` floats respectively.
#[inline]
pub fn ridge_solve_direct_f32(
    w_t: &mut [f32],
    l_scratch: &mut [f32],
    z_scratch: &mut [f32],
    gram_reg: &[f32], // XᵀX + λI, d_h × d_h
    cov: &[f32],      // XᵀY,     d_h × n_out
    d_h: usize,
    n_out: usize,
) {
    cholesky_f32(l_scratch, gram_reg, d_h);
    // Solve L Lᵀ Wᵀ = cov  →  Wᵀ written into w_t.
    solve_lower_triangular_strided(z_scratch, l_scratch, cov, d_h, n_out);
    solve_upper_triangular_transposed_strided(w_t, l_scratch, z_scratch, d_h, n_out);
}

/// Woodbury (sample-space) ridge solve for the `d_h > N` regime.
///
/// Solves the same ridge problem as [`ridge_solve_direct_f32`] but via the
/// Woodbury identity (paper Eq. 40–41): when the feature dimension `d_h`
/// exceeds the sample count `N`, inverting the `N × N` sample-space Gram is
/// cheaper than the `d_h × d_h` feature-space Gram. Produces the same `Wᵀ`
/// (`d_h × n_out`) up to floating-point ordering differences between the two
/// factorisations.
///
/// Inputs:
/// - `sample_gram_reg = X Xᵀ + λI` (`N × N`, SPD)
/// - `y` = targets `Y` (`N × n_out`, row-major; one row per sample)
/// - `x` = features `X` (`N × d_h`, row-major; one row per sample)
///
/// Output: `w_t = Xᵀ · (X Xᵀ + λI)⁻¹ · Y`, shape `d_h × n_out`.
///
/// Scratch: `l_scratch` (`N*N`), `z_scratch` (`N*n_out`),
/// `xt_z_scratch` accumulator is folded into `w_t` directly.
#[inline]
#[allow(clippy::too_many_arguments)] // Woodbury dual-form ridge solve API: (w_t, 2× scratch, gram, y, x, n, d_h, n_out) is intrinsic
pub fn ridge_solve_woodbury_f32(
    w_t: &mut [f32],
    l_scratch: &mut [f32],
    z_scratch: &mut [f32],
    sample_gram_reg: &[f32], // X Xᵀ + λI, N × N
    y: &[f32],               // N × n_out
    x: &[f32],               // N × d_h
    n: usize,
    d_h: usize,
    n_out: usize,
) {
    cholesky_f32(l_scratch, sample_gram_reg, n);
    // Z = (X Xᵀ + λI)⁻¹ Y, shape N × n_out.
    solve_lower_triangular_strided(z_scratch, l_scratch, y, n, n_out);
    let z_owned: Vec<f32> = z_scratch[..n * n_out].to_vec();
    solve_upper_triangular_transposed_strided(z_scratch, l_scratch, &z_owned, n, n_out);
    // Wᵀ = Xᵀ Z, shape d_h × n_out.  Xᵀ row i = column i of X.
    for i in 0..d_h {
        for col in 0..n_out {
            let mut s = 0.0f32;
            let mut r = 0;
            while r + 4 <= n {
                s = x[r * d_h + i].mul_add(z_scratch[r * n_out + col], s);
                s = x[(r + 1) * d_h + i].mul_add(z_scratch[(r + 1) * n_out + col], s);
                s = x[(r + 2) * d_h + i].mul_add(z_scratch[(r + 2) * n_out + col], s);
                s = x[(r + 3) * d_h + i].mul_add(z_scratch[(r + 3) * n_out + col], s);
                r += 4;
            }
            while r < n {
                s = x[r * d_h + i].mul_add(z_scratch[r * n_out + col], s);
                r += 1;
            }
            w_t[i * n_out + col] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn cholesky_identity() {
        let k = 3;
        let a = vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let mut l = vec![0.0; k * k];
        cholesky_f32(&mut l, &a, k);
        // L for 2I is sqrt(2) I.
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 2.0f32.sqrt() } else { 0.0 };
                assert!(approx_eq(l[i * k + j], expected, 1e-5));
            }
        }
    }

    #[test]
    fn spd_inverse_recovers_original() {
        let k = 3;
        // A well-conditioned SPD matrix.
        let a = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let mut inv = vec![0.0; k * k];
        let mut l = vec![0.0; k * k];
        let mut inv_l = vec![0.0; k * k];
        spd_inverse_f32(&mut inv, &mut l, &mut inv_l, &a, k);
        // A · A⁻¹ = I.
        let mut prod = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * inv[kk * k + j];
                }
                prod[i * k + j] = s;
            }
        }
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(prod[i * k + j], expected, 1e-4), "({},{}): {} vs {}", i, j, prod[i * k + j], expected);
            }
        }
    }

    #[test]
    fn ridge_direct_solves_small_problem() {
        // X = [[1,0],[0,1],[1,1]] (3 samples, 2 features), Y = [[1],[1],[2]] (3×1).
        // XᵀX = [[2,1],[1,2]], λ=1e-6 ≈ 0 → gram ≈ [[2,1],[1,2]].
        // XᵀY = [3, 3]. Solve [[2,1],[1,2]] w = [3,3] → w=[1,1].
        let d_h = 2;
        let n_out = 1;
        let gram = vec![2.0, 1.0, 1.0, 2.0];
        let cov = vec![3.0, 3.0];
        let mut w_t = vec![0.0; d_h * n_out];
        let mut l = vec![0.0; d_h * d_h];
        let mut z = vec![0.0; d_h * n_out];
        ridge_solve_direct_f32(&mut w_t, &mut l, &mut z, &gram, &cov, d_h, n_out);
        assert!(approx_eq(w_t[0], 1.0, 1e-4), "w[0]={}", w_t[0]);
        assert!(approx_eq(w_t[1], 1.0, 1e-4), "w[1]={}", w_t[1]);
    }

    #[test]
    fn ridge_direct_matches_hand_computed_linear_map() {
        // From forecaster_fits_and_forecasts_linear_map: X=[[1,x_i]] for
        // x in {-1,-0.9,...,0.9}, Y=[2x_i]. XᵀX=[[20,-1],[-1,6.7]], XᵀY=[-2,13.4].
        // Hand-solved: W=[0, 2].
        let d_h = 2;
        let n_out = 1;
        let gram = vec![20.0 + 1e-6, -1.0, -1.0, 6.7 + 1e-6];
        let cov = vec![-2.0, 13.4];
        let mut w_t = vec![0.0; d_h * n_out];
        let mut l = vec![0.0; d_h * d_h];
        let mut z = vec![0.0; d_h * n_out];
        ridge_solve_direct_f32(&mut w_t, &mut l, &mut z, &gram, &cov, d_h, n_out);
        assert!(approx_eq(w_t[0], 0.0, 1e-3), "w[0]={}", w_t[0]);
        assert!(approx_eq(w_t[1], 2.0, 1e-3), "w[1]={}", w_t[1]);
    }

    #[test]
    fn woodbury_matches_direct() {
        // Same problem as ridge_direct_solves_small_problem but via Woodbury.
        let n = 3;
        let d_h = 2;
        let n_out = 1;
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3×2 row-major
        let y = vec![1.0, 1.0, 2.0]; // 3×1
        // X Xᵀ = [[1,0,1],[0,1,1],[1,1,2]], λ=1e-6 ≈ 0.
        let mut sample_gram = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..d_h {
                    s += x[i * d_h + kk] * x[j * d_h + kk];
                }
                sample_gram[i * n + j] = s;
            }
        }
        let lambda = 1e-6f32;
        for i in 0..n {
            sample_gram[i * n + i] += lambda;
        }
        let mut w_t = vec![0.0; d_h * n_out];
        let mut l = vec![0.0; n * n];
        let mut z = vec![0.0; n * n_out];
        ridge_solve_woodbury_f32(
            &mut w_t, &mut l, &mut z, &sample_gram, &y, &x, n, d_h, n_out,
        );
        // Expect w ≈ [1, 1].
        assert!(approx_eq(w_t[0], 1.0, 1e-3), "woodbury w[0]={}", w_t[0]);
        assert!(approx_eq(w_t[1], 1.0, 1e-3), "woodbury w[1]={}", w_t[1]);
    }
}
