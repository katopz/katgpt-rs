//! Tests for [`super::symmetric_eig`].
//!
//! Three test families:
//! 1. **Known-answer** — analytic eigenpairs for diagonal / identity / 2×2 /
//!    3×3 Toeplitz matrices.
//! 2. **Parity vs `jacobi_eigen`** — eigenvalues agree to `1e-12` and
//!    eigenvectors agree up to sign (`|v_h·v_j| > 1 - 1e-10`) on random SPD
//!    matrices at `n ≤ 16`. This is the Issue 186 T3 GOAT G1 gate.
//! 3. **Reconstruction + orthonormality** — for random SPD, verify
//!    `A ≈ V · diag(d) · Vᵀ` and `V · Vᵀ = I`.

use super::{SymmetricEigScratch, symmetric_eig};

#[cfg(feature = "karc_forecaster")]
use crate::karc::jacobi_eigen;

/// A simple deterministic PRNG (xorshift64) so the tests are bit-reproducible
/// across runs (no dependency on `rand`).
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Map a u64 to f64 in [-1, 1].
fn u64_to_unit_f64(x: u64) -> f64 {
    // Use the upper 53 bits for the mantissa, sign from bit 63.
    let u = x >> 11;
    let sign = (x >> 63) & 1;
    let m = (u as f64) / ((1u64 << 53) as f64);
    if sign == 1 { -m } else { m }
}

/// Generate a random SPD matrix with distinct eigenvalues.
///
/// Recipe: `A = MᵀM + I + ε·diag(0..n)` where `M` is `n×n` with i.i.d. uniform
/// entries in `[-1, 1]`, `I` is identity (ensures SPD), and the small diagonal
/// perturbation `ε = 0.01` breaks degeneracy (so eigenvectors are uniquely
/// determined up to sign, enabling parity comparison).
fn random_spd(state: &mut u64, a_out: &mut [f64], n: usize) {
    // A = MᵀM where M[i,j] ~ U[-1,1]
    // Accumulate directly: A[i,j] = sum_k M[k,i] * M[k,j]
    for v in a_out.iter_mut().take(n * n) {
        *v = 0.0;
    }
    let mut m_col = vec![0.0_f64; n];
    for _k in 0..n {
        for m_col_j in m_col.iter_mut().take(n) {
            *m_col_j = u64_to_unit_f64(xorshift64(state));
        }
        for i in 0..n {
            for j in 0..n {
                a_out[i * n + j] += m_col[i] * m_col[j];
            }
        }
    }
    // A += I + ε·diag(0..n)
    for i in 0..n {
        a_out[i * n + i] += 1.0 + 0.01 * (i as f64);
    }
}

/// Verify A · v = λ · v for each eigenpair. Returns the max error.
fn check_eigenpairs(a: &[f64], eigvals: &[f64], eigvecs: &[f64], n: usize, tol: f64) -> f64 {
    let mut max_err = 0.0_f64;
    for k in 0..n {
        for i in 0..n {
            // (A · v_k)[i] = sum_j A[i,j] · v_k[j]
            let mut av = 0.0_f64;
            for j in 0..n {
                av += a[i * n + j] * eigvecs[j * n + k];
            }
            let lv = eigvals[k] * eigvecs[i * n + k];
            let err = (av - lv).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    assert!(
        max_err < tol,
        "A·v ≠ λ·v: max_err = {max_err:e} (n={n}, tol={tol:e})"
    );
    max_err
}

/// Verify eigenvectors are orthonormal: VᵀV = I. Returns the max error.
fn check_orthonormal(eigvecs: &[f64], n: usize, tol: f64) -> f64 {
    let mut max_err = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += eigvecs[k * n + i] * eigvecs[k * n + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            let err = (s - expected).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    assert!(
        max_err < tol,
        "VᵀV ≠ I: max_err = {max_err:e} (n={n}, tol={tol:e})"
    );
    max_err
}

// ─── Known-answer tests ─────────────────────────────────────────────────────

#[test]
fn n_equals_1() {
    let a = [3.0_f64];
    let mut eigvals = vec![0.0];
    let mut eigvecs = vec![0.0];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, 1, 30);
    assert!((eigvals[0] - 3.0).abs() < 1e-15);
    assert!((eigvecs[0] - 1.0).abs() < 1e-15);
}

#[test]
fn n_equals_2_diagonal() {
    let a = [3.0_f64, 0.0, 0.0, 5.0];
    let mut eigvals = vec![0.0; 2];
    let mut eigvecs = vec![0.0; 4];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, 2, 30);
    let mut eigs: Vec<f64> = eigvals.to_vec();
    eigs.sort_by(|a, b| a.total_cmp(b));
    assert!((eigs[0] - 3.0).abs() < 1e-12, "eigvals = {eigs:?}");
    assert!((eigs[1] - 5.0).abs() < 1e-12);
    check_eigenpairs(&a, &eigvals, &eigvecs, 2, 1e-12);
}

#[test]
fn n_equals_2_analytic() {
    // A = [[2, 1], [1, 2]] → eigenvalues 1, 3; eigenvectors (1,-1)/√2 and (1,1)/√2.
    let a = [2.0_f64, 1.0, 1.0, 2.0];
    let mut eigvals = vec![0.0; 2];
    let mut eigvecs = vec![0.0; 4];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, 2, 30);
    let mut eigs: Vec<f64> = eigvals.to_vec();
    eigs.sort_by(|a, b| a.total_cmp(b));
    assert!((eigs[0] - 1.0).abs() < 1e-12, "eigvals = {eigs:?}");
    assert!((eigs[1] - 3.0).abs() < 1e-12);
    check_eigenpairs(&a, &eigvals, &eigvecs, 2, 1e-12);
    check_orthonormal(&eigvecs, 2, 1e-12);
}

#[test]
fn n_equals_3_diagonal() {
    // diag(3, 1, 2) → eigenvalues {1, 2, 3}; eigenvectors = standard basis permuted.
    let a = [3.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let mut eigvals = vec![0.0; 3];
    let mut eigvecs = vec![0.0; 9];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, 3, 30);
    let mut eigs: Vec<f64> = eigvals.to_vec();
    eigs.sort_by(|a, b| a.total_cmp(b));
    for (got, expected) in eigs.iter().zip(&[1.0_f64, 2.0, 3.0]) {
        assert!((got - expected).abs() < 1e-12, "eigs = {eigs:?}");
    }
    check_eigenpairs(&a, &eigvals, &eigvecs, 3, 1e-12);
}

#[test]
fn n_equals_3_toeplitz() {
    // A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]] → eigenvalues {2, 2±√2}.
    let sqrt2 = 2.0_f64.sqrt();
    let a = [2.0_f64, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let mut eigvals = vec![0.0; 3];
    let mut eigvecs = vec![0.0; 9];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, 3, 30);
    let mut eigs: Vec<f64> = eigvals.to_vec();
    eigs.sort_by(|a, b| a.total_cmp(b));
    let expected = vec![2.0 - sqrt2, 2.0, 2.0 + sqrt2];
    for (got, exp) in eigs.iter().zip(&expected) {
        assert!(
            (got - exp).abs() < 1e-12,
            "eigs = {eigs:?} (expected {expected:?})"
        );
    }
    check_eigenpairs(&a, &eigvals, &eigvecs, 3, 1e-12);
    check_orthonormal(&eigvecs, 3, 1e-12);
}

#[test]
fn identity_matrix() {
    let n = 8;
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    let mut eigvals = vec![0.0; n];
    let mut eigvecs = vec![0.0; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);
    for v in &eigvals {
        assert!((v - 1.0).abs() < 1e-12, "eigvals = {eigvals:?}");
    }
    check_orthonormal(&eigvecs, n, 1e-12);
}

// ─── Parity vs jacobi_eigen (Issue 186 T3 GOAT G1 gate) ─────────────────────
//
// Only compiled when `karc_forecaster` is on (the feature gating `jacobi_eigen`).
// The rest of the test suite is independent of `karc`.

#[cfg(feature = "karc_forecaster")]
#[test]
fn parity_vs_jacobi_random_spd() {
    // 10 random SPD matrices at n = 4, 8, 16. Householder+QL and Jacobi must
    // agree on eigenvalues to 1e-12 and on eigenvectors up to sign.
    let mut state: u64 = 0x1234567890abcdef;
    let configs = [4_usize, 8, 16];
    for &n in &configs {
        for _trial in 0..10 {
            // Generate a random SPD matrix.
            let mut a = vec![0.0_f64; n * n];
            random_spd(&mut state, &mut a, n);

            // Run both eigensolvers on identical input.
            let mut hh_eigvals = vec![0.0; n];
            let mut hh_eigvecs = vec![0.0; n * n];
            let mut hh_scratch = SymmetricEigScratch::new();
            symmetric_eig(&mut hh_eigvals, &mut hh_eigvecs, &a, &mut hh_scratch, n, 30);

            let mut jac_eigvals = vec![0.0; n];
            let mut jac_eigvecs = vec![0.0; n * n];
            let mut jac_scratch = vec![0.0; n * n];
            jacobi_eigen(
                &mut jac_eigvals,
                &mut jac_eigvecs,
                &a,
                &mut jac_scratch,
                n,
                1e-15,
                100,
            );

            // Eigenvalue comparison (sort first — algorithms don't return
            // eigenvalues in the same order).
            let mut hh_sorted: Vec<f64> = hh_eigvals.to_vec();
            let mut jac_sorted: Vec<f64> = jac_eigvals.to_vec();
            hh_sorted.sort_by(|a, b| a.total_cmp(b));
            jac_sorted.sort_by(|a, b| a.total_cmp(b));
            let mut max_eigval_err = 0.0_f64;
            for (&h, &j) in hh_sorted.iter().zip(&jac_sorted) {
                let err = (h - j).abs();
                if err > max_eigval_err {
                    max_eigval_err = err;
                }
            }
            assert!(
                max_eigval_err < 1e-12,
                "n={n}: eigenvalue parity failed, max_err = {max_eigval_err:e}"
            );

            // Eigenvector comparison: for each Householder eigenvector v_h
            // (paired with eigenvalue λ_h), find the matching Jacobi
            // eigenvector v_j (paired with the closest eigenvalue) and verify
            // |v_h · v_j| > 1 - 1e-10 (allow sign flip).
            for k_h in 0..n {
                let lambda_h = hh_eigvals[k_h];
                // Find the Jacobi index with the closest eigenvalue.
                let mut best_j = 0;
                let mut best_err = f64::INFINITY;
                for (k_j, &jac_lam) in jac_eigvals.iter().enumerate().take(n) {
                    let err = (jac_lam - lambda_h).abs();
                    if err < best_err {
                        best_err = err;
                        best_j = k_j;
                    }
                }
                // Compute |v_h · v_j| (both are unit vectors).
                let mut dot = 0.0_f64;
                for row in 0..n {
                    dot += hh_eigvecs[row * n + k_h] * jac_eigvecs[row * n + best_j];
                }
                let alignment = dot.abs();
                assert!(
                    alignment > 1.0 - 1e-10,
                    "n={}: eigenvector parity failed at k_h={} (lambda={}): \
                     alignment={:e} (best_j={}, lambda_j={})",
                    n,
                    k_h,
                    lambda_h,
                    alignment,
                    best_j,
                    jac_eigvals[best_j]
                );
            }

            // Sanity: both must satisfy A·v = λ·v. Tolerance is loose
            // (1e-7) because the small-eigenvalue-gap regime (0.01·diag
            // perturbation) amplifies eigenvector errors via 1/gap; the
            // actual contract is the parity check above (1e-12 eigenvalues,
            // 1-1e-10 alignment), which is the strict correctness gate.
            check_eigenpairs(&a, &hh_eigvals, &hh_eigvecs, n, 1e-7);
            check_eigenpairs(&a, &jac_eigvals, &jac_eigvecs, n, 1e-7);
        }
    }
}

// ─── Reconstruction (A = V · diag(d) · Vᵀ) + orthonormality ─────────────────

#[test]
fn reconstruction_matches_input() {
    let mut state: u64 = 0xdeadbeefcafef00d;
    for &n in &[8_usize, 16, 32] {
        let mut a = vec![0.0_f64; n * n];
        random_spd(&mut state, &mut a, n);

        let mut eigvals = vec![0.0; n];
        let mut eigvecs = vec![0.0; n * n];
        let mut scratch = SymmetricEigScratch::new();
        symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);

        // Check V·Vᵀ = I.
        check_orthonormal(&eigvecs, n, 1e-10);

        // Check A = V·diag(d)·Vᵀ: A[i,j] = sum_k V[i,k] · d[k] · V[j,k].
        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += eigvecs[i * n + k] * eigvals[k] * eigvecs[j * n + k];
                }
                let err = (s - a[i * n + j]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // Tolerance scales with the matrix norm (n²·eps roughly).
        let a_max = a.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let tol = (n as f64) * (n as f64) * 1e-13 * a_max;
        assert!(
            max_err < tol,
            "n={n}: A ≠ V·diag(d)·Vᵀ, max_err = {max_err:e} (tol={tol:e})"
        );
    }
}

// ─── Bit-reproducibility ─────────────────────────────────────────────────────

#[test]
fn bit_reproducible_across_calls() {
    let mut state: u64 = 0xfeedface12345678;
    let n = 16;
    let mut a = vec![0.0_f64; n * n];
    random_spd(&mut state, &mut a, n);

    // Run twice with independent scratch instances.
    let mut eigvals_1 = vec![0.0; n];
    let mut eigvecs_1 = vec![0.0; n * n];
    let mut scratch_1 = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals_1, &mut eigvecs_1, &a, &mut scratch_1, n, 30);

    let mut eigvals_2 = vec![0.0; n];
    let mut eigvecs_2 = vec![0.0; n * n];
    let mut scratch_2 = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals_2, &mut eigvecs_2, &a, &mut scratch_2, n, 30);

    // Bit-identical (not just close): same float-operation order.
    assert_eq!(
        eigvals_1, eigvals_2,
        "eigvals differ across calls — non-deterministic"
    );
    assert_eq!(
        eigvecs_1, eigvecs_2,
        "eigvecs differ across calls — non-deterministic"
    );
}

// ─── Larger-n sanity (Issue 186 target regime) ───────────────────────────────

#[test]
fn n_equals_64_reconstruction() {
    // Exercises the Householder loop (62 reflections) and ~200 QL iterations.
    let mut state: u64 = 0xa1b2c3d4e5f60718;
    let n = 64;
    let mut a = vec![0.0_f64; n * n];
    random_spd(&mut state, &mut a, n);

    let mut eigvals = vec![0.0; n];
    let mut eigvecs = vec![0.0; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);

    check_orthonormal(&eigvecs, n, 1e-9);
    check_eigenpairs(&a, &eigvals, &eigvecs, n, 1e-9);
}

// ─── Issue 931: graded symmetric matrices ───────────────────────────────────
//
// The riir-ai style-shard covariance (Bench 980) — large diagonal spread with
// mixed-magnitude off-diagonals, cond ≈ 7.3e5, min eig 7.08e-8 > 0 — made the
// QL phase diverge: loud NaN panic at the raw scale, and (worse) eigenvalues
// at ±1e78 after a uniform rescale to O(1) diagonals. numpy solves the same
// input to full precision, so the input is well-posed and the solver was at
// fault. These tests pin the repaired behavior: finite, accurate eigenvalues
// on the graded family at every scale, no panic, no garbage.

/// Deterministic orthonormal basis via two-pass Gram-Schmidt over xorshift
/// columns. Column k of the result is orthonormal to all others; the whole
/// matrix is the rotation V in `A = V Λ Vᵀ`.
fn random_orthonormal(state: &mut u64, v_out: &mut [f64], n: usize) {
    for k in 0..n {
        for i in 0..n {
            v_out[i * n + k] = u64_to_unit_f64(xorshift64(state));
        }
    }
    // Two modified Gram-Schmidt passes (re-orthogonalization keeps the basis
    // orthonormal to machine precision even for near-parallel seeds).
    for _pass in 0..2 {
        for k in 0..n {
            for j in 0..k {
                let mut dot = 0.0_f64;
                for i in 0..n {
                    dot += v_out[i * n + k] * v_out[i * n + j];
                }
                for i in 0..n {
                    v_out[i * n + k] -= dot * v_out[i * n + j];
                }
            }
            let mut norm = 0.0_f64;
            for i in 0..n {
                norm += v_out[i * n + k] * v_out[i * n + k];
            }
            let norm = norm.sqrt();
            for i in 0..n {
                v_out[i * n + k] /= norm;
            }
        }
    }
}

/// Build `A = V · diag(λ) · Vᵀ` from a rotation and a target spectrum. The
/// eigenvalues of the result are exactly the λ vector (up to f64 rounding in
/// the triple product, ~eps·‖A‖).
fn spectral_matrix(v: &[f64], lambda: &[f64], a_out: &mut [f64], n: usize) {
    for val in a_out.iter_mut().take(n * n) {
        *val = 0.0;
    }
    // A = Σ_k λ_k · v_k v_kᵀ — one rank-1 update per eigenvalue.
    for k in 0..n {
        let col = &v[k..n * n].iter().step_by(n).cloned().collect::<Vec<_>>();
        let lam = lambda[k];
        for i in 0..n {
            let ci = lam * col[i];
            for j in 0..n {
                a_out[i * n + j] += ci * col[j];
            }
        }
    }
}

/// Sorted-eigenvalue comparison against the exactly-known spectrum.
fn check_known_spectrum(eigvals: &[f64], lambda_true: &mut [f64], n: usize, tol: f64) {
    let mut got: Vec<f64> = eigvals.iter().take(n).cloned().collect();
    got.sort_by(|a, b| a.partial_cmp(b).expect("solver returned NaN eigenvalue"));
    lambda_true.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for k in 0..n {
        let err = (got[k] - lambda_true[k]).abs();
        assert!(
            err < tol,
            "eigenvalue {k} off: got {:.17e} want {:.17e} err {err:e} (tol {tol:e})",
            got[k],
            lambda_true[k]
        );
    }
}

/// The Issue 931 graded family: cond ≈ 7.3e5 (the Bench 980 style-shard
/// covariance shape) at three uniform scales — the raw scale (loud NaN panic
/// class) and the O(1)-normalized scale (silent ±1e78 garbage class), plus a
/// large-scale arm. Every arm must solve to known eigenvalues, orthonormal
/// vectors, and small residuals.
#[test]
fn graded_matrix_issue931_scales() {
    let n = 64;
    // Log-spaced spectrum over [7.08e-8, 5.14e-2]: the measured triggering
    // shape (min eig 7.08e-8, max 5.14e-2, cond ≈ 7.3e5).
    let lo = 7.08e-8_f64;
    let hi = 5.14e-2_f64;
    let lambda: Vec<f64> = (0..n)
        .map(|k| lo * (hi / lo).powf(k as f64 / (n - 1) as f64))
        .collect();

    let mut state: u64 = 0x9A31_0000_C0DE_D00D;
    let mut v = vec![0.0_f64; n * n];
    random_orthonormal(&mut state, &mut v, n);

    let mut a = vec![0.0_f64; n * n];
    spectral_matrix(&v, &lambda, &mut a, n);

    for &scale in &[1.0_f64, 1.0 / hi, 1e6] {
        let mut a_scaled = a.clone();
        for val in a_scaled.iter_mut() {
            *val *= scale;
        }
        let mut eigvals = vec![0.0_f64; n];
        let mut eigvecs = vec![0.0_f64; n * n];
        let mut scratch = SymmetricEigScratch::new();
        symmetric_eig(&mut eigvals, &mut eigvecs, &a_scaled, &mut scratch, n, 200);
        // Accuracy bar: ±1e-9 vs the known spectrum — but the spectrum scales
        // too, so compare against the scaled λ (tolerance scales with ‖A‖).
        let mut lambda_scaled = lambda.clone();
        for val in lambda_scaled.iter_mut() {
            *val *= scale;
        }
        let tol = 1e-9 * scale.max(1.0);
        check_known_spectrum(&eigvals, &mut lambda_scaled, n, tol);
        check_orthonormal(&eigvecs, n, 1e-9);
        check_eigenpairs(&a_scaled, &eigvals, &eigvecs, n, tol.max(1e-9));
    }
}

/// Higher dynamic range: cond ≈ 7e12 (λ over [1e-11, 1]) — beyond the reported
/// trigger but the same class. The solver must stay finite and resolve the
/// dominant spectrum to tolerance (the subspace holding λ ≤ 1e-11 is allowed
/// relative error ~eps·‖A‖/λ_min = O(1), so only λ ≥ 1e-6 are pinned; the
/// residual bound uses ‖A‖-scaled tolerance).
#[test]
fn graded_matrix_issue931_extreme_range() {
    let n = 48;
    let lo = 1e-11_f64;
    let hi = 1.0_f64;
    let lambda: Vec<f64> = (0..n)
        .map(|k| lo * (hi / lo).powf(k as f64 / (n - 1) as f64))
        .collect();
    let mut state: u64 = 0x0DEFACED_931D0001;
    let mut v = vec![0.0_f64; n * n];
    random_orthonormal(&mut state, &mut v, n);
    let mut a = vec![0.0_f64; n * n];
    spectral_matrix(&v, &lambda, &mut a, n);

    let mut eigvals = vec![0.0_f64; n];
    let mut eigvecs = vec![0.0_f64; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 200);

    // Every eigenvalue must at least be finite.
    for (k, &lv) in eigvals.iter().enumerate().take(n) {
        assert!(lv.is_finite(), "eigenvalue {k} non-finite: {lv:e}");
    }
    // The well-resolved head of the spectrum (λ ≥ 1e-6) is pinned to
    // eps·‖A‖-scaled tolerance; the deep tail is only required finite.
    let tol = 1e-9;
    let mut got: Vec<f64> = eigvals.iter().take(n).cloned().collect();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut want = lambda.clone();
    want.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for k in 0..n {
        if want[k] >= 1e-6 {
            let err = (got[k] - want[k]).abs();
            assert!(
                err < tol,
                "eigenvalue {k} off: got {:+.17e} want {:+.17e} err {err:e}",
                got[k],
                want[k]
            );
        }
    }
    check_orthonormal(&eigvecs, n, 1e-9);
    check_eigenpairs(&a, &eigvals, &eigvecs, n, 1e-9);
}

/// Diagonally-graded family: diag entries graded over many orders while
/// off-diagonals stay comparable to the geometric mean of their endpoints —
/// locally breaking diagonal dominance (big off-diagonal beside tiny
/// diagonals). This is the EISPACK-era "graded matrix" shape that broke NR
/// `tqli` and the shape the Bench 980 style covariance reports ("large
/// diagonal spread with mixed-magnitude off-diagonals"). Oracles: residual
/// (A·v = λ·v) + orthonormality — both must hold at eps·‖A‖ level.
#[test]
fn graded_matrix_issue931_diagonally_graded() {
    let n = 64;
    // diag spans 1 → 1e-12; A[i][j] = g_ij · √(D_i·D_j) with g_ij ∈ [-1, 1],
    // drawn once per {i,j} pair and mirrored (the matrix must be symmetric —
    // an asymmetric input is out of contract and the solver's output is
    // undefined for it).
    let mut state: u64 = 0x9A31_D1A0_0AFF_0002;
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        let di = 10.0_f64.powf(-(i as f64) * 12.0 / (n - 1) as f64);
        for j in 0..=i {
            let dj = 10.0_f64.powf(-(j as f64) * 12.0 / (n - 1) as f64);
            let g = u64_to_unit_f64(xorshift64(&mut state));
            let v = g * (di * dj).sqrt();
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }

    let mut eigvals = vec![0.0_f64; n];
    let mut eigvecs = vec![0.0_f64; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 200);

    for (k, &lv) in eigvals.iter().enumerate().take(n) {
        assert!(lv.is_finite(), "eigenvalue {k} non-finite: {lv:e}");
    }
    check_orthonormal(&eigvecs, n, 1e-9);
    check_eigenpairs(&a, &eigvals, &eigvecs, n, 1e-9);
}

/// The real-pipeline shape: a covariance accumulated from N snapshots over
/// features whose scales span orders of magnitude (the Bench 980 style-shard
/// sigma is exactly this — 1024 snapshots, d=64, moment columns of mixed
/// magnitude). Built as X̃ᵀX̃/N with graded feature scales + shared factors.
#[test]
fn graded_matrix_issue931_snapshot_covariance() {
    let n = 64;
    let snapshots = 1024_usize;
    let mut state: u64 = 0x9A31_5A1A_0003;
    // Feature scales span 1 → 1e-4 (diag of the covariance spans 1 → 1e-8).
    let scale = |j: usize| 10.0_f64.powf(-(j as f64) * 4.0 / (n - 1) as f64);
    // Latent factors drive correlation: f ∈ R^8 per snapshot.
    let mut a = vec![0.0_f64; n * n];
    let mut x = vec![0.0_f64; n];
    for _t in 0..snapshots {
        let mut f = [0.0_f64; 8];
        for fv in f.iter_mut() {
            *fv = u64_to_unit_f64(xorshift64(&mut state));
        }
        for j in 0..n {
            let mut v = f[j % 8] + 0.5 * u64_to_unit_f64(xorshift64(&mut state));
            v *= scale(j);
            x[j] = v;
        }
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] += x[i] * x[j];
            }
        }
    }
    for val in a.iter_mut() {
        *val /= snapshots as f64;
    }

    let mut eigvals = vec![0.0_f64; n];
    let mut eigvecs = vec![0.0_f64; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 200);

    for (k, &lv) in eigvals.iter().enumerate().take(n) {
        assert!(lv.is_finite(), "eigenvalue {k} non-finite: {lv:e}");
    }
    check_orthonormal(&eigvecs, n, 1e-9);
    check_eigenpairs(&a, &eigvals, &eigvecs, n, 1e-9);
}

/// A NaN entry must refuse loudly at the input seam (Issue 931): the QL
/// deflation tests read false against NaN, so the pre-guard panic at entry
/// replaces the old spin-to-cap failure ("QL failed to converge … [NaN, NaN]").
#[test]
#[should_panic(expected = "non-finite input at (i, j) = (1, 2)")]
fn non_finite_input_refuses_loudly() {
    let n = 4;
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    a[n + 2] = f64::NAN;
    a[2 * n + 1] = f64::NAN; // keep it symmetric — position (1,2) is named
    let mut eigvals = vec![0.0; n];
    let mut eigvecs = vec![0.0; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);
}

/// An inf entry must refuse the same way.
#[test]
#[should_panic(expected = "non-finite input")]
fn infinite_input_refuses_loudly() {
    let n = 3;
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
    }
    a[0] = f64::INFINITY;
    let mut eigvals = vec![0.0; n];
    let mut eigvecs = vec![0.0; n * n];
    let mut scratch = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);
}

// ─── Issue 186 T4 GOAT G2 perf gate (ad-hoc timing) ─────────────────────────
//
// Run with: cargo test --release --features karc_forecaster --lib \
//           linalg::symmetric_eig::tests::timing_householder_vs_jacobi \
//           -- --nocapture --ignored
//
// Reports the Householder+QL vs Jacobi wall-time at n = 64, 128, 256, 512,
// along with the speedup ratio. The Issue 186 T4 gate requires ≥5× at
// n = 256+; this test exists to confirm that gate empirically.

#[cfg(feature = "karc_forecaster")]
#[test]
#[ignore]
fn timing_householder_vs_jacobi() {
    use std::time::Instant;
    let mut state: u64 = 0xabcdef1234567890;
    for &n in &[64_usize, 128, 256, 512] {
        let mut a = vec![0.0_f64; n * n];
        random_spd(&mut state, &mut a, n);

        // Time Householder+QL (3 trials, take min).
        let mut hh_min = u128::MAX;
        for _ in 0..3 {
            let mut eigvals = vec![0.0; n];
            let mut eigvecs = vec![0.0; n * n];
            let mut scratch = SymmetricEigScratch::new();
            let t0 = Instant::now();
            symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);
            let dt = t0.elapsed().as_nanos();
            if dt < hh_min {
                hh_min = dt;
            }
        }

        // Time Jacobi (3 trials, take min).
        let mut jac_min = u128::MAX;
        for _ in 0..3 {
            let mut eigvals = vec![0.0; n];
            let mut eigvecs = vec![0.0; n * n];
            let mut scratch = vec![0.0; n * n];
            let t0 = Instant::now();
            jacobi_eigen(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 1e-15, 100);
            let dt = t0.elapsed().as_nanos();
            if dt < jac_min {
                jac_min = dt;
            }
        }

        let speedup = jac_min as f64 / hh_min as f64;
        eprintln!(
            "n={n:>4}: householder={hh_min:>10} ns  jacobi={jac_min:>10} ns  speedup={speedup:>5.2}×"
        );
    }
}

// ─── Parallel-path parity tests (Issue 187 T4 bit-identity contract) ────────
//
// These run only under the `karc_householder_eig_par` feature. They verify
// that `symmetric_eig_par` produces bit-identical (eigvals, eigvecs) output
// to the serial `symmetric_eig` on the same input. The parallel path uses
// row-parallel rayon with no cross-row reductions, so the per-row numerical
// work is identical; these tests pin that contract.

#[cfg(feature = "karc_householder_eig_par")]
#[test]
fn par_vs_serial_bit_identity_small() {
    use super::par::symmetric_eig_par;

    let mut state: u64 = 0xfeed_beef_dead_1234;
    for &n in &[1_usize, 2, 3, 4, 8, 16, 17, 31, 32, 33, 64] {
        let mut a = vec![0.0_f64; n * n];
        random_spd(&mut state, &mut a, n);

        // Serial
        let mut eigvals_s = vec![0.0; n];
        let mut eigvecs_s = vec![0.0; n * n];
        let mut scratch_s = SymmetricEigScratch::new();
        symmetric_eig(&mut eigvals_s, &mut eigvecs_s, &a, &mut scratch_s, n, 30);

        // Parallel
        let mut eigvals_p = vec![0.0; n];
        let mut eigvecs_p = vec![0.0; n * n];
        let mut scratch_p = SymmetricEigScratch::new();
        symmetric_eig_par(&mut eigvals_p, &mut eigvecs_p, &a, &mut scratch_p, n, 30);

        // Bit-identity check: bytes must match exactly.
        let eigvals_match = eigvals_s
            .iter()
            .zip(&eigvals_p)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let eigvecs_match = eigvecs_s
            .iter()
            .zip(&eigvecs_p)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            eigvals_match,
            "n={n}: eigvals differ between serial and parallel"
        );
        assert!(
            eigvecs_match,
            "n={}: eigvecs differ between serial and parallel (first diff at index {:?})",
            n,
            eigvecs_s
                .iter()
                .zip(&eigvecs_p)
                .position(|(a, b)| a.to_bits() != b.to_bits())
        );
    }
}

#[cfg(feature = "karc_householder_eig_par")]
#[test]
fn par_vs_serial_bit_identity_medium() {
    use super::par::symmetric_eig_par;

    // Larger sizes that exercise more Householder reflections + more QL
    // rotations — where any scheduling-dependent drift would compound.
    let mut state: u64 = 0xface_feed_1234_5678;
    for &n in &[128_usize, 256] {
        let mut a = vec![0.0_f64; n * n];
        random_spd(&mut state, &mut a, n);

        let mut eigvals_s = vec![0.0; n];
        let mut eigvecs_s = vec![0.0; n * n];
        let mut scratch_s = SymmetricEigScratch::new();
        symmetric_eig(&mut eigvals_s, &mut eigvecs_s, &a, &mut scratch_s, n, 30);

        let mut eigvals_p = vec![0.0; n];
        let mut eigvecs_p = vec![0.0; n * n];
        let mut scratch_p = SymmetricEigScratch::new();
        symmetric_eig_par(&mut eigvals_p, &mut eigvecs_p, &a, &mut scratch_p, n, 30);

        // Bit-identity on eigenvalues.
        for (i, (vs, vp)) in eigvals_s.iter().zip(&eigvals_p).enumerate() {
            assert_eq!(
                vs.to_bits(),
                vp.to_bits(),
                "n={n}, i={i}: eigval bits differ ({vs} vs {vp})"
            );
        }
        // Bit-identity on eigenvectors.
        let first_diff = eigvecs_s
            .iter()
            .zip(&eigvecs_p)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            first_diff.is_none(),
            "n={n}: first eigvec diff at byte-index {first_diff:?}"
        );
    }
}

#[cfg(feature = "karc_householder_eig_par")]
#[test]
fn par_vs_serial_diagonal_identity() {
    use super::par::symmetric_eig_par;

    // Diagonal matrix — eigenvalues are the diagonal entries; eigenvectors
    // are the standard basis. Both serial and parallel must reconstruct
    // this exactly.
    let n = 64;
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        a[i * n + i] = (i as f64) + 1.0;
    }

    let mut eigvals_s = vec![0.0; n];
    let mut eigvecs_s = vec![0.0; n * n];
    let mut scratch_s = SymmetricEigScratch::new();
    symmetric_eig(&mut eigvals_s, &mut eigvecs_s, &a, &mut scratch_s, n, 30);

    let mut eigvals_p = vec![0.0; n];
    let mut eigvecs_p = vec![0.0; n * n];
    let mut scratch_p = SymmetricEigScratch::new();
    symmetric_eig_par(&mut eigvals_p, &mut eigvecs_p, &a, &mut scratch_p, n, 30);

    assert!(
        eigvals_s
            .iter()
            .zip(&eigvals_p)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert!(
        eigvecs_s
            .iter()
            .zip(&eigvecs_p)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[cfg(feature = "karc_householder_eig_par")]
#[test]
#[ignore]
fn timing_par_vs_serial() {
    // Issue 187 T5 benchmark gate: ≥4× speedup at n ≥ 256. The bar is lower
    // than the Issue 186 T4 serial-vs-Jacobi gate (≥5×) because rayon's
    // speedup is bounded by core count; this test exists to confirm the
    // parallel path is actually using the cores.
    use super::par::symmetric_eig_par;
    use std::time::Instant;

    let mut state: u64 = 0xabc_def_123_456;
    for &n in &[256_usize, 512, 1024, 2048] {
        let mut a = vec![0.0_f64; n * n];
        random_spd(&mut state, &mut a, n);

        // Time serial (min of 3).
        let mut s_min = u128::MAX;
        for _ in 0..3 {
            let mut eigvals = vec![0.0; n];
            let mut eigvecs = vec![0.0; n * n];
            let mut scratch = SymmetricEigScratch::new();
            let t0 = Instant::now();
            symmetric_eig(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);
            let dt = t0.elapsed().as_nanos();
            if dt < s_min {
                s_min = dt;
            }
        }

        // Time parallel (min of 3).
        let mut p_min = u128::MAX;
        for _ in 0..3 {
            let mut eigvals = vec![0.0; n];
            let mut eigvecs = vec![0.0; n * n];
            let mut scratch = SymmetricEigScratch::new();
            let t0 = Instant::now();
            symmetric_eig_par(&mut eigvals, &mut eigvecs, &a, &mut scratch, n, 30);
            let dt = t0.elapsed().as_nanos();
            if dt < p_min {
                p_min = dt;
            }
        }

        let speedup = s_min as f64 / p_min as f64;
        eprintln!(
            "n={n:>5}: serial={s_min:>10} ns  parallel={p_min:>10} ns  speedup={speedup:>5.2}×"
        );
    }
}
