//! `dense` — pinned cyclic-Jacobi eigenvalue kernel for small symmetric
//! matrices, `d ≤ 32` (Issue 676 T2).
//!
//! ## Why pinned (the determinism policy)
//!
//! Any **committed** readout (chain predicates, canonical-gauge bytes,
//! committed floats) must be reproducible bit-for-bit **per binary**.
//! Library eigensolvers (LAPACK/ ndarray / `nalgebra`) vary rotation
//! order, blocking, and vectorization across versions and targets — fine
//! for science, wrong for commitment. This kernel is a self-contained
//! classical cyclic Jacobi sweep with a **pinned rotation schedule**
//! (strict `p < q` row-major order), a **pinned convergence rule**
//! (off² ≤ τ², τ = `off_tol`·‖A‖_F, max `MAX_SWEEPS` sweeps), and a
//! **pinned sort** (selection sort on the diagonal, eigenvector columns
//! permuted alongside). Same binary + same input → identical bytes.
//!
//! ## Cost
//!
//! One sweep = `d(d−1)/2` rotations, each `O(d)` → `O(d³)` per sweep,
//! ~6–10 sweeps typical for `d ≤ 32` random symmetric input
//! (`≈ 4/3·d³ + slack` total, the paper's §7.3 inference figure).

/// Maximum sweeps before the pinned loop stops (converged or not).
/// 30 is ~3× the empirical worst case at d=32; the residual is reported.
pub const MAX_SWEEPS: u8 = 30;

/// Relative off-diagonal tolerance: stop when `‖off(A)‖₂ ≤ off_tol ·
/// ‖A‖_F` (0 ⇒ run to the sweep cap — fully pinned iteration count).
pub const OFF_TOL: f32 = 1e-7;

/// Caller-owned scratch for the dense kernel. Sized entirely by `D`;
/// zero allocation, reusable across every call.
pub struct DenseScratch<const D: usize> {
    /// Working copy of the matrix under diagonalization.
    pub a: [[f32; D]; D],
    /// Eigenvector accumulator (columns; `V·e_i` = eigenvector of the
    /// i-th sorted eigenvalue after [`jacobi_eigen`]). Starts as
    /// identity each call when eigenvectors are requested.
    pub v: [[f32; D]; D],
    /// Sorted eigenvalues (ascending) after [`jacobi_eigen`].
    pub values: [f32; D],
    /// Off-diagonal Frobenius residual at stop.
    pub off_residual: f32,
    /// Sweeps actually run.
    pub sweeps: u8,
}

impl<const D: usize> DenseScratch<D> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            a: [[0.0; D]; D],
            v: [[0.0; D]; D],
            values: [0.0; D],
            off_residual: 0.0,
            sweeps: 0,
        }
    }
}

impl<const D: usize> Default for DenseScratch<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Result summary for one diagonalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JacobiReport {
    pub sweeps: u8,
    /// `sqrt(Σ_{i<j} a_ij²)` at stop — the not-yet-diagonalized residue.
    pub off_residual: f32,
    /// True if the off-tolerance was met before the sweep cap.
    pub converged: bool,
}

/// Diagonalize a full symmetric `D×D` matrix in place on `scratch`.
///
/// On return: `scratch.values` holds the eigenvalues **ascending**;
/// `scratch.v` holds the eigenvectors as **columns** aligned with those
/// sorted values (`want_vectors == false` skips the accumulator — the
/// values alone are cheaper).
///
/// Classical Jacobi rotation for element `(p, q)` uses the
/// smaller-magnitude tangent root
/// `t = sign(θ)/(|θ| + √(θ²+1))`, `θ = (a_qq − a_pp)/(2·a_pq)` — the
/// standard numerically-stable form (Golub & Van Loan Alg. 8.4.1).
pub fn jacobi_eigen<const D: usize>(
    a_full: &[[f32; D]; D],
    want_vectors: bool,
    scratch: &mut DenseScratch<D>,
) -> JacobiReport {
    scratch.a = *a_full;
    if want_vectors {
        for (i, v_row) in scratch.v.iter_mut().enumerate() {
            for (j, v_ij) in v_row.iter_mut().enumerate() {
                *v_ij = if i == j { 1.0 } else { 0.0 };
            }
        }
    }

    // Scale reference: Frobenius norm of the input (fixed at entry — the
    // tolerance anchor does not drift as the matrix diagonalizes).
    let mut fro_sq = 0.0_f64;
    for row in scratch.a.iter() {
        for &x in row.iter() {
            fro_sq += f64::from(x) * f64::from(x);
        }
    }
    let fro = (fro_sq as f32).sqrt().max(f32::MIN_POSITIVE);
    let tol = OFF_TOL * fro;

    let mut report = JacobiReport {
        sweeps: 0,
        off_residual: 0.0,
        converged: false,
    };
    for sweep in 0..MAX_SWEEPS {
        // off-diagonal residual
        let mut off_sq = 0.0_f64;
        for (p, a_row) in scratch.a.iter().enumerate() {
            for &apq in a_row[(p + 1)..].iter() {
                off_sq += f64::from(apq) * f64::from(apq);
            }
        }
        let off = (off_sq as f32).sqrt();
        report.off_residual = off;
        report.sweeps = sweep + 1;
        if off <= tol {
            report.converged = true;
            break;
        }

        // One cyclic sweep: every (p, q) pair, pinned row-major order.
        for p in 0..D {
            for q in (p + 1)..D {
                let apq = scratch.a[p][q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (scratch.a[q][q] - scratch.a[p][p]) / (2.0 * apq);
                // smaller-magnitude root of t² + 2tθ − 1 = 0
                let t = sign_f32(theta) / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Apply the rotation J(p,q; θ) as A ← JᵀAJ on rows/cols
                // p and q. Unrolled three-index form (G&VL 8.4).
                let app = scratch.a[p][p];
                let aqq = scratch.a[q][q];
                scratch.a[p][p] = app - t * apq;
                scratch.a[q][q] = aqq + t * apq;
                scratch.a[p][q] = 0.0;
                scratch.a[q][p] = 0.0;
                for i in 0..D {
                    if i != p && i != q {
                        let aip = scratch.a[i][p];
                        let aiq = scratch.a[i][q];
                        let new_ip = c * aip - s * aiq;
                        let new_iq = s * aip + c * aiq;
                        scratch.a[i][p] = new_ip;
                        scratch.a[i][q] = new_iq;
                        scratch.a[p][i] = new_ip;
                        scratch.a[q][i] = new_iq;
                    }
                }
                if want_vectors {
                    for v_row in scratch.v.iter_mut() {
                        let vip = v_row[p];
                        let viq = v_row[q];
                        v_row[p] = c * vip - s * viq;
                        v_row[q] = s * vip + c * viq;
                    }
                }
            }
        }
    }

    // Sort eigenvalues ascending; permute eigenvector columns alongside.
    // Selection sort = pinned order (stable for equal keys by index).
    for (i, val) in scratch.values.iter_mut().enumerate() {
        *val = scratch.a[i][i];
    }
    if want_vectors {
        for i in 0..D {
            // find min in [i, D)
            let mut m = i;
            for j in (i + 1)..D {
                if scratch.values[j] < scratch.values[m] {
                    m = j;
                }
            }
            if m != i {
                scratch.values.swap(i, m);
                for v_row in scratch.v.iter_mut() {
                    v_row.swap(i, m);
                }
            }
        }
    } else {
        for i in 0..D {
            let mut m = i;
            for j in (i + 1)..D {
                if scratch.values[j] < scratch.values[m] {
                    m = j;
                }
            }
            if m != i {
                scratch.values.swap(i, m);
            }
        }
    }
    report
}

#[inline]
fn sign_f32(x: f32) -> f32 {
    // Pinned convention: sign(0) = 1 (the rotation degenerates to
    // identity at apq→0 anyway; this only fixes the root's branch).
    if x < 0.0 { -1.0 } else { 1.0 }
}

/// The k-th smallest eigenvalue (0-indexed `k < D`) of a full symmetric
/// matrix — the convenience wrapper over [`jacobi_eigen`].
#[must_use]
pub fn kth_eigenvalue<const D: usize>(
    a_full: &[[f32; D]; D],
    k: usize,
    scratch: &mut DenseScratch<D>,
) -> f32 {
    debug_assert!(k < D, "k={k} out of range for D={D}");
    jacobi_eigen(a_full, false, scratch);
    scratch.values[k.min(D - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0).max(a.abs().max(1.0))
    }

    #[test]
    fn diagonal_matrix_is_one_sweep_exact() {
        const D: usize = 4;
        let a = [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ];
        let mut s = DenseScratch::<D>::new();
        let r = jacobi_eigen(&a, true, &mut s);
        assert!(r.converged);
        assert_eq!(r.sweeps, 1); // first residual check passes immediately
        let expect = [-1.0_f32, 0.5, 2.0, 3.0];
        for (got, want) in s.values.iter().zip(expect.iter()) {
            assert!((*got - *want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn known_2x2_spectrum() {
        const D: usize = 2;
        // [[2, 1], [1, 2]] → eigenvalues 1 and 3
        let a = [[2.0_f32, 1.0], [1.0, 2.0]];
        let mut s = DenseScratch::<D>::new();
        jacobi_eigen(&a, true, &mut s);
        assert!(approx(s.values[0], 1.0, 1e-6));
        assert!(approx(s.values[1], 3.0, 1e-6));
        // eigenvector check: v0 ∝ (1, −1)/√2 for λ=1
        assert!((s.v[0][0] - -s.v[1][0]).abs() < 1e-5 || (s.v[0][0] + s.v[1][0]).abs() < 1e-5);
    }

    #[test]
    fn eigenvectors_reconstruct_the_matrix() {
        const D: usize = 5;
        let mut rng = 123_u64;
        let mut a = [[0.0_f32; D]; D];
        for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((rng >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0;
            a[i][j] = v;
            a[j][i] = v;
        }
        let mut s = DenseScratch::<D>::new();
        jacobi_eigen(&a, true, &mut s);
        // A·V == V·diag(λ)
        for (i, a_row) in a.iter().enumerate() {
            for k in 0..D {
                let av: f32 = (0..D).map(|j| a_row[j] * s.v[j][k]).sum();
                let vl = s.values[k] * s.v[i][k];
                assert!(
                    (av - vl).abs() < 1e-4,
                    "A·v != λv at ({i},{k}): {av} vs {vl}"
                );
            }
        }
    }

    #[test]
    fn deterministic_repeat_calls_bit_identical() {
        const D: usize = 6;
        let mut rng = 999_u64;
        let mut a = [[0.0_f32; D]; D];
        for (i, j) in (0..D).flat_map(|i| (i..D).map(move |j| (i, j))) {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((rng >> 33) as f32 / 2.0_f32.powi(31)) * 2.0 - 1.0;
            a[i][j] = v;
            a[j][i] = v;
        }
        let mut s1 = DenseScratch::<D>::new();
        let mut s2 = DenseScratch::<D>::new();
        jacobi_eigen(&a, true, &mut s1);
        jacobi_eigen(&a, true, &mut s2);
        assert_eq!(s1.values, s2.values);
        assert_eq!(s1.v, s2.v);
    }
}
