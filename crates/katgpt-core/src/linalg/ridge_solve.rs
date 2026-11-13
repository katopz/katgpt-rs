//! f32 ridge-regression kernels via Cholesky factorisation.
//!
//! Extracted from the same mathematical pattern as
//! `peira::predictor_with_scratch` (the `(N + λI)^{-1}` step in Plan 153), but
//! kept as a standalone f32 path rather than refactoring PEIRA's f64 numerics.
//! See the module-level note in [`super`] for the rationale.
//!
//! // TODO: unify with peira's f64 path once a generic-over-`T: Float` Cholesky
//! // is verified bit-identical to PEIRA's f64 specialisation. (Plan 153 is
//! // complete; this unification was NOT a Plan 153 task — it's standalone
//! // future work. File a new plan before implementing.)
//!
//! # Numerical contract
//!
//! All routines assume the input Gram/covariance matrices are
//! symmetric positive (semi-)definite and write into caller-provided buffers.
//! `cholesky_f32` panics on a non-positive-definite leading minor — the ridge
//! diagonal `+λI` added by the caller is what guarantees positive-definiteness.
//! `λ > 0` is a hard precondition for every public entry point here. Fit paths
//! that must degrade instead of panic use the non-panicking twins
//! [`try_cholesky_f32`] / [`try_ridge_solve_woodbury_f32`] (KARC's
//! `fit_woodbury` returns `FitError::Singular`; Issue 904).
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
                "matrix not positive definite in cholesky_f64 (pivot {diag} < -{tol})"
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
/// A Cholesky pivot fell to `pivot ≤ -tol` — the input's leading minor is not
/// positive definite beyond f32 accumulation noise, and the near-singular
/// clamp (which absorbs pivots in `(-tol, 0]`) refused it.
///
/// `pivot` and `tol` are carried so a caller's log line can print exactly what
/// the panicking [`cholesky_f32`] assert prints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotPositiveDefinite {
    /// The offending pivot (at or below `-tol`).
    pub pivot: f32,
    /// The relative tolerance the pivot was tested against (`a_max · k · 2⁻²³`).
    pub tol: f32,
}

impl core::fmt::Display for NotPositiveDefinite {
    #[cold]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "matrix not positive definite (pivot {} < -{})",
            self.pivot, self.tol
        )
    }
}

impl std::error::Error for NotPositiveDefinite {}

/// Strict failure-mode twin of [`try_cholesky_f32`]: rejects ANY
/// non-positive pivot (`diag <= 0`) instead of clamping near-singular ones
/// to the tolerance floor. Same factorisation loop, same SIMD-interior dots
/// — only the failure semantics differ, mirroring the retired local copies
/// (attn-match `beta_fitter`/`value_fitter`, Issue 995) whose `sum <= 0 ->
/// None` contract is what drives those callers' caller-side jitter
/// escalation. Without the strict twin, the relative-tolerance clamp
/// succeeds on rank-deficient systems, the jitter policy never fires, and
/// the solve rides clamped pseudo-pivots (measured on the G5 fixture:
/// 0.033 -> 0.095 relative error). Use when the caller owns a
/// degrade-by-retry policy; use [`try_cholesky_f32`] when a clamped
/// pseudo-inverse solve is acceptable.
#[inline]
pub fn try_cholesky_strict_f32(
    l: &mut [f32],
    a: &[f32],
    k: usize,
) -> Result<(), NotPositiveDefinite> {
    // Zero the lower-triangle scratch we are about to fill.
    for v in l.iter_mut().take(k * k) {
        *v = 0.0;
    }
    for j in 0..k {
        let j_row = j * k;
        let sum = if j > 0 {
            simd_dot_f32(&l[j_row..j_row + j], &l[j_row..j_row + j], j)
        } else {
            0.0
        };
        let diag = a[j_row + j] - sum;
        if diag <= 0.0 {
            return Err(NotPositiveDefinite {
                pivot: diag,
                tol: 0.0,
            });
        }
        let d_sqrt = diag.sqrt();
        l[j_row + j] = d_sqrt;
        let inv = 1.0 / d_sqrt;
        for i in (j + 1)..k {
            let i_row = i * k;
            let s = a[i_row + j] - simd_dot_f32(&l[i_row..i_row + j], &l[j_row..j_row + j], j);
            l[i_row + j] = s * inv;
        }
    }
    Ok(())
}

/// Non-panicking [`cholesky_f32`]: returns [`NotPositiveDefinite`] instead of
/// asserting when a pivot falls below `-tol`. The success path is the same
/// loop (bit-identical factorisation); only the failure mode differs. Fit
/// paths with a warn-and-degrade contract call this — KARC's `fit_woodbury`
/// maps the error to [`crate::karc::FitError::Singular`] (Issue 904, where the
/// panic bypassed every `tick_karc` caller's warn-and-keep arm); the panicking
/// form stays for callers that want the assert.
#[inline]
pub fn try_cholesky_f32(l: &mut [f32], a: &[f32], k: usize) -> Result<(), NotPositiveDefinite> {
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
            if diag <= -tol {
                return Err(NotPositiveDefinite { pivot: diag, tol });
            }
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
    Ok(())
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
/// Prefer [`try_cholesky_f32`] when the caller has a degrade path.
#[inline]
pub fn cholesky_f32(l: &mut [f32], a: &[f32], k: usize) {
    match try_cholesky_f32(l, a, k) {
        Ok(()) => {}
        Err(e) => panic!(
            "matrix not positive definite in cholesky_f32 (pivot {} < -{})",
            e.pivot, e.tol
        ),
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

/// In-place variant of [`solve_upper_triangular_transposed_strided`] where
/// `x` and `z` share the same buffer. This is valid because the back-substitution
/// processes rows in strictly decreasing order of `i`: when computing row `i`,
/// it reads `z[i]` (which was written by the prior forward-substitution and is
/// below the current `i`, so still untouched) and reads `x[j]` for `j > i`
/// (which were already overwritten with the final solution by earlier iterations
/// of this back-substitution loop). Writing `x[i]` only clobbers the `z[i]`
/// slot, which is never read again. This is the standard textbook in-place
/// Cholesky solve and eliminates the `.to_vec()` allocation the previous call
/// site performed.
fn solve_upper_triangular_transposed_strided_inplace(
    xz: &mut [f32],
    l: &[f32],
    k: usize,
    n_rhs: usize,
) {
    for col in 0..n_rhs {
        for ii in 0..k {
            let i = k - 1 - ii;
            let i_row = i * k;
            let mut s = xz[i * n_rhs + col];
            let mut j = i + 1;
            while j + 4 <= k {
                s -= l[j * k + i] * xz[j * n_rhs + col];
                s -= l[(j + 1) * k + i] * xz[(j + 1) * n_rhs + col];
                s -= l[(j + 2) * k + i] * xz[(j + 2) * n_rhs + col];
                s -= l[(j + 3) * k + i] * xz[(j + 3) * n_rhs + col];
                j += 4;
            }
            while j < k {
                s -= l[j * k + i] * xz[j * n_rhs + col];
                j += 1;
            }
            xz[i * n_rhs + col] = s / l[i_row + i];
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
                s = inv_l_scratch[(l + 1) * k + i].mul_add(inv_l_scratch[(l + 1) * k + j], s);
                s = inv_l_scratch[(l + 2) * k + i].mul_add(inv_l_scratch[(l + 2) * k + j], s);
                s = inv_l_scratch[(l + 3) * k + i].mul_add(inv_l_scratch[(l + 3) * k + j], s);
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

/// Non-panicking [`ridge_solve_woodbury_f32"]: returns [`NotPositiveDefinite`] when the
/// sample Gram's Cholesky pivot falls below the relative tolerance, instead of
/// asserting. The success path is the same loop (bit-identical `w_t`); only
/// the failure mode differs. KARC's `fit_woodbury` consumes this so an
/// indefinite f32 sample Gram degrades to `FitError::Singular` through the
/// caller's warn-and-keep arm instead of panicking the tick loop (Issue 904).
#[inline]
#[allow(clippy::too_many_arguments)] // Woodbury dual-form ridge solve API: (w_t, 2× scratch, gram, y, x, n, d_h, n_out) is intrinsic
pub fn try_ridge_solve_woodbury_f32(
    w_t: &mut [f32],
    l_scratch: &mut [f32],
    z_scratch: &mut [f32],
    sample_gram_reg: &[f32], // X Xᵀ + λI, N × N
    y: &[f32],               // N × n_out
    x: &[f32],               // N × d_h
    n: usize,
    d_h: usize,
    n_out: usize,
) -> Result<(), NotPositiveDefinite> {
    try_cholesky_f32(l_scratch, sample_gram_reg, n)?;
    // Z = (X Xᵀ + λI)⁻¹ Y, shape N × n_out.
    // Two triangular solves in-place: lower-substitute into z_scratch, then
    // back-substitute in-place (the back-sub reads x[j] for j>i which the
    // forward pass already wrote into z_scratch itself, so we can reuse the
    // same buffer and avoid the previous `z_scratch.to_vec()` allocation).
    solve_lower_triangular_strided(z_scratch, l_scratch, y, n, n_out);
    solve_upper_triangular_transposed_strided_inplace(z_scratch, l_scratch, n, n_out);
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
    Ok(())
}

/// Woodbury (sample-space) ridge solve for the `d_h > N` regime.
///
/// Panicking form of [`try_ridge_solve_woodbury_f32`] (same loop, same
/// bit-identical `w_t` on success; the failure mode is the historical
/// `cholesky_f32` assert, message preserved).
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
///
/// Panics if the sample Gram is not positive definite beyond the relative
/// tolerance (see [`try_ridge_solve_woodbury_f32`] for the Result form).
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
    match try_ridge_solve_woodbury_f32(
        w_t,
        l_scratch,
        z_scratch,
        sample_gram_reg,
        y,
        x,
        n,
        d_h,
        n_out,
    ) {
        Ok(()) => {}
        Err(e) => panic!(
            "matrix not positive definite in cholesky_f32 (pivot {} < -{})",
            e.pivot, e.tol
        ),
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
                assert!(
                    approx_eq(prod[i * k + j], expected, 1e-4),
                    "({},{}): {} vs {}",
                    i,
                    j,
                    prod[i * k + j],
                    expected
                );
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
            &mut w_t,
            &mut l,
            &mut z,
            &sample_gram,
            &y,
            &x,
            n,
            d_h,
            n_out,
        );
        // Expect w ≈ [1, 1].
        assert!(approx_eq(w_t[0], 1.0, 1e-3), "woodbury w[0]={}", w_t[0]);
        assert!(approx_eq(w_t[1], 1.0, 1e-3), "woodbury w[1]={}", w_t[1]);
    }

    #[test]
    fn try_cholesky_ok_on_pd_and_errs_on_indefinite() {
        // PD control: [[4,2],[2,3]] → l00=2, l10=1, pivot₁ = 3-1 = 2 > 0.
        let a_pd = vec![4.0, 2.0, 2.0, 3.0];
        let mut l = vec![0.0; 4];
        assert!(try_cholesky_f32(&mut l, &a_pd, 2).is_ok());
        assert!(approx_eq(l[0], 2.0, 1e-6));
        assert!(approx_eq(l[3], 2.0f32.sqrt(), 1e-6));

        // Indefinite: [[2,3],[3,2]] (eigenvalues 5, -1). Pivot₁ = 2 - 3²/2 =
        // -2.5, far below the relative tolerance → Err with both fields set.
        let a_indef = vec![2.0, 3.0, 3.0, 2.0];
        let mut l2 = vec![0.0; 4];
        let err = try_cholesky_f32(&mut l2, &a_indef, 2).unwrap_err();
        assert!(approx_eq(err.pivot, -2.5, 1e-5), "pivot={}", err.pivot);
        assert!(
            err.pivot <= -err.tol,
            "pivot {} vs tol {}",
            err.pivot,
            err.tol
        );
        assert!(
            err.to_string()
                .starts_with("matrix not positive definite (pivot"),
            "display={}",
            err
        );
    }

    #[test]
    #[should_panic(expected = "matrix not positive definite in cholesky_f32")]
    fn cholesky_f32_still_panics_on_indefinite_input() {
        // The panicking form keeps its historical contract (Issue 904 kept it
        // for callers that want the assert); only the fit path degrades.
        let a_indef = vec![2.0, 3.0, 3.0, 2.0];
        let mut l = vec![0.0; 4];
        cholesky_f32(&mut l, &a_indef, 2);
    }

    #[test]
    fn try_woodbury_errs_on_indefinite_sample_gram() {
        // The Issue-904 failure mechanism, deterministic and in exact powers
        // of two: rows 0/1 colinear → pivot₁ lands on exactly 0.0, clamped to
        // the a_max·2⁻²⁴ floor → l₁₁ = 2⁻¹¹ → the next column's entry l₂₁ =
        // (2 - 1) / 2⁻¹¹ = 2¹¹ → pivot₂ = 1 - 0.25 - 2²² ≈ -4.2e6 < -tol.
        // The f32 Woodbury fit path returned Err (FitError::Singular) here;
        // before Issue 904 it panicked the tick loop.
        let n = 3;
        let sample_gram: Vec<f32> = vec![
            4.0, 4.0, 1.0, //
            4.0, 4.0, 2.0, //
            1.0, 2.0, 1.0,
        ];
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3×2 row-major
        let y = vec![1.0, 2.0, 3.0]; // 3×1
        let mut w_t = vec![0.0; 2];
        let mut l = vec![0.0; n * n];
        let mut z = vec![0.0; n];
        let err =
            try_ridge_solve_woodbury_f32(&mut w_t, &mut l, &mut z, &sample_gram, &y, &x, n, 2, 1)
                .unwrap_err();
        assert!(
            err.pivot <= -err.tol,
            "pivot {} vs tol {}",
            err.pivot,
            err.tol
        );
        assert!(
            err.pivot < -1e6,
            "expected the floor-clamp cascade, pivot={}",
            err.pivot
        );
    }

    #[test]
    fn try_woodbury_matches_panicking_form_on_pd_input() {
        // Same loop, same bits on the success path — the two APIs must agree
        // exactly so the degrade path cannot change any healthy fit.
        let n = 3;
        let d_h = 2;
        let sample_gram: Vec<f32> = vec![
            4.0, 2.0, 0.0, //
            2.0, 3.0, 0.0, //
            0.0, 0.0, 2.0,
        ];
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3×2 row-major
        let y = vec![1.0, 1.0, 2.0]; // 3×1
        let mut w_try = vec![0.0; d_h];
        let mut l1 = vec![0.0; n * n];
        let mut z1 = vec![0.0; n];
        try_ridge_solve_woodbury_f32(
            &mut w_try,
            &mut l1,
            &mut z1,
            &sample_gram,
            &y,
            &x,
            n,
            d_h,
            1,
        )
        .unwrap();
        let mut w_panic = vec![0.0; d_h];
        let mut l2 = vec![0.0; n * n];
        let mut z2 = vec![0.0; n];
        ridge_solve_woodbury_f32(
            &mut w_panic,
            &mut l2,
            &mut z2,
            &sample_gram,
            &y,
            &x,
            n,
            d_h,
            1,
        );
        assert_eq!(w_try, w_panic, "try/panicking forms must be bit-identical");
    }
}
