//! `spectral_audit` — chain composition verifier (Plan 330 Phase 4, G6 gate).
//!
//! Implements a small DCT-II projection of the tangent operator of a
//! [`TransportOperator`] at identity, then reports per-mode gain and a
//! spurious-coupling score. The G6 GOAT gate checks:
//!
//! - Known-good composite (clean FuncAttn-style operators) → ≤ 5% spurious.
//! - Known-bad (random operator) → > 5% spurious.
//!
//! # Why this exists
//!
//! ASOC's stale-draft fallback path **accepts a possibly-wrong action to stay
//! non-blocking** (by design — see Plan 330 § "Risks"). The spectral audit
//! runs against the *completed* join path's composite operator in the warm-tier
//! reflection cycle (`ReestimationScheduler`), NOT in the ASOC hot path. It is
//! the late-binding verifier that catches a bad composite after the fact and
//! emits a correction.
//!
//! # Math
//!
//! ## Tangent operator (numerical Jacobian at identity)
//!
//! For a transport operator `C: R^k → R^k`, the tangent at the identity vector
//! `e = (1, 1, ..., 1) / sqrt(k)` is the matrix of directional derivatives:
//!
//! ```text
//! T[i,j] = (C(e + ε·e_j)[i] - C(e - ε·e_j)[i]) / (2ε)
//! ```
//!
//! where `e_j` is the j-th standard basis vector and `ε` is a small step
//! (default `1e-3`). This is the standard central-difference Jacobian.
//!
//! ## DCT-II projection
//!
//! For `M = min(fourier_modes, k)` DCT-II basis vectors `φ_m`, we project
//! each column of `T` onto `φ_m` and report the per-mode gain
//! `g_m = Σ_j |⟨T[:,j], φ_m⟩|`. The DCT-II basis is:
//!
//! ```text
//! φ_m[j] = cos(π·m·(2j+1) / (2k))     for m = 0..M, j = 0..k
//! ```
//!
//! (Unnormalized — the normalization cancels in the spurious-coupling ratio.)
//!
//! ## Spurious coupling
//!
//! The "diagonal" of the mode-projection matrix `P[m, n] = ⟨T·φ_n, φ_m⟩`
//! captures how each mode maps to itself. The off-diagonal mass captures
//! spurious cross-mode coupling. We report `spurious_ratio = off_diag_mass / total_mass`.
//!
//! A clean composite (e.g. identity, or well-conditioned FuncAttn output) has
//! most of its energy on the diagonal → low spurious ratio. A random operator
//! spreads energy across all `(m, n)` → high spurious ratio.

use crate::analytic_lattice::TransportOperator;

/// Default number of DCT-II modes to project onto.
pub const DEFAULT_FOURIER_MODES: usize = 8;

/// Default central-difference step for the numerical Jacobian.
pub const DEFAULT_EPSILON: f32 = 1e-3;

/// G6 threshold: composites with `spurious_ratio` above this are flagged as
/// "known-bad" (random / corrupted / numerically divergent).
pub const SPURIOUS_THRESHOLD: f32 = 0.05;

/// Audit report for a transport operator.
#[derive(Debug, Clone)]
pub struct AuditReport {
    /// Per-mode gain `g_m = Σ_j |⟨T[:,j], φ_m⟩|` for each of the `M` DCT-II modes.
    pub mode_gains: Vec<f32>,
    /// Full mode-projection matrix `P[m, n] = ⟨T·φ_n, φ_m⟩` (M × M).
    /// Diagonal = self-coupling, off-diagonal = spurious cross-mode coupling.
    pub mode_matrix: Vec<f32>,
    /// Off-diagonal mass / total mass. Low (< 5%) = clean composite.
    /// High (> 5%) = spurious coupling (random or corrupted operator).
    pub spurious_ratio: f32,
    /// Number of DCT-II modes used.
    pub modes: usize,
    /// The epsilon used for the numerical Jacobian.
    pub epsilon: f32,
    /// The operator dimension `k`.
    pub k: usize,
}

impl AuditReport {
    /// Returns `true` if this operator passes the G6 gate (spurious ≤ threshold).
    pub fn passes_g6(&self) -> bool {
        self.spurious_ratio <= SPURIOUS_THRESHOLD
    }

    /// Read entry `(m, n)` of the mode matrix (row-major M × M).
    pub fn mode_matrix_get(&self, m: usize, n: usize) -> f32 {
        self.mode_matrix[m * self.modes + n]
    }
}

/// Audit a transport operator for spurious spectral coupling.
///
/// Computes the tangent (numerical Jacobian at identity), projects onto
/// `min(fourier_modes, k)` DCT-II modes, and reports the spurious-coupling
/// ratio. See the module docs for the math.
///
/// # Allocation
///
/// This is an **audit-cadence** primitive, not a hot-path primitive. It
/// allocates working buffers (the Jacobian, the mode matrix). Per AGENTS.md,
/// audit-cadence primitives are allowed to allocate; only the hot path must be
/// zero-alloc. The audit runs in the warm-tier `ReestimationScheduler`, not in
/// the ASOC tick.
pub fn spectral_audit(operator: &TransportOperator) -> AuditReport {
    spectral_audit_with_modes(operator, DEFAULT_FOURIER_MODES, DEFAULT_EPSILON)
}

/// Audit with explicit mode count and epsilon (for tuning / testing).
pub fn spectral_audit_with_modes(
    operator: &TransportOperator,
    fourier_modes: usize,
    epsilon: f32,
) -> AuditReport {
    let k = operator.k;
    let m = fourier_modes.min(k);

    // 1. Tangent operator T (k × k numerical Jacobian at the identity vector e).
    let mut tangent = vec![0.0f32; k * k];
    compute_tangent(operator, epsilon, &mut tangent);

    // 2. DCT-II basis vectors φ_0..φ_{M-1}, each of length k.
    let mut basis = vec![0.0f32; m * k];
    compute_dct2_basis(&mut basis, m, k);

    // 3. Mode matrix P[m, n] = ⟨T·φ_n, φ_m⟩.
    //    T·φ_n is the n-th mode propagated through the operator.
    //    ⟨·, φ_m⟩ projects the result onto the m-th mode.
    let mut mode_matrix = vec![0.0f32; m * m];
    let mut t_phi_n = vec![0.0f32; k]; // scratch for T·φ_n
    for n in 0..m {
        let phi_n = &basis[n * k..(n + 1) * k];
        // t_phi_n = T · phi_n  (matvec)
        matvec(&tangent, phi_n, &mut t_phi_n, k);
        for mm in 0..m {
            let phi_m = &basis[mm * k..(mm + 1) * k];
            mode_matrix[mm * m + n] = dot(&t_phi_n, phi_m, k);
        }
    }

    // 4. Per-mode gain g_m = Σ_j |P[m, n]| over n (row mass of |P|).
    let mut mode_gains = vec![0.0f32; m];
    for mm in 0..m {
        let mut s = 0.0f32;
        for n in 0..m {
            s += mode_matrix[mm * m + n].abs();
        }
        mode_gains[mm] = s;
    }

    // 5. Spurious ratio: off-diagonal |P| mass / total |P| mass.
    let mut diag_mass = 0.0f32;
    let mut total_mass = 0.0f32;
    for mm in 0..m {
        for n in 0..m {
            let v = mode_matrix[mm * m + n].abs();
            total_mass += v;
            if mm == n {
                diag_mass += v;
            }
        }
    }
    let off_diag_mass = (total_mass - diag_mass).max(0.0);
    let spurious_ratio = if total_mass > 0.0 {
        off_diag_mass / total_mass
    } else {
        0.0
    };

    AuditReport {
        mode_gains,
        mode_matrix,
        spurious_ratio,
        modes: m,
        epsilon,
        k,
    }
}

/// Compute the tangent operator T (numerical Jacobian of `C` at the identity
/// vector `e = (1,...,1)/sqrt(k)`) via central differences.
///
/// `T[i, j] = (C(e + ε·e_j)[i] - C(e - ε·e_j)[i]) / (2ε)`
fn compute_tangent(operator: &TransportOperator, epsilon: f32, out: &mut [f32]) {
    let k = operator.k;
    debug_assert_eq!(out.len(), k * k);

    let inv_sqrt_k = 1.0 / (k as f32).sqrt();
    let mut e_plus = vec![inv_sqrt_k; k];
    let mut e_minus = vec![inv_sqrt_k; k];
    let mut c_plus = vec![0.0f32; k];
    let mut c_minus = vec![0.0f32; k];

    for j in 0..k {
        // Reset to identity vector.
        for v in e_plus.iter_mut() {
            *v = inv_sqrt_k;
        }
        for v in e_minus.iter_mut() {
            *v = inv_sqrt_k;
        }
        // Perturb along e_j.
        e_plus[j] += epsilon;
        e_minus[j] -= epsilon;

        // Apply C to both.
        matvec(&operator.data, &e_plus, &mut c_plus, k);
        matvec(&operator.data, &e_minus, &mut c_minus, k);

        // Central difference fills column j of T.
        let inv_2eps = 1.0 / (2.0 * epsilon);
        for i in 0..k {
            out[i * k + j] = (c_plus[i] - c_minus[i]) * inv_2eps;
        }
    }
}

/// Compute the (unnormalized) DCT-II basis matrix: `basis[m*k + j] = cos(π·m·(2j+1)/(2k))`.
fn compute_dct2_basis(basis: &mut [f32], modes: usize, k: usize) {
    debug_assert_eq!(basis.len(), modes * k);
    let denom = 2.0 * k as f32;
    for m in 0..modes {
        for j in 0..k {
            let phase = std::f32::consts::PI * (m as f32) * (2 * j + 1) as f32 / denom;
            basis[m * k + j] = phase.cos();
        }
    }
}

/// Row-major matvec: `out = mat · v` where `mat` is `rows × cols` row-major.
#[inline]
fn matvec(mat: &[f32], v: &[f32], out: &mut [f32], k: usize) {
    debug_assert_eq!(mat.len(), k * k);
    debug_assert_eq!(v.len(), k);
    debug_assert_eq!(out.len(), k);
    for i in 0..k {
        let row = &mat[i * k..(i + 1) * k];
        out[i] = dot(row, v, k);
    }
}

/// Plain dot product (no SIMD — audit is not hot-path).
#[inline]
fn dot(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..len {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic_lattice::compose_chain;

    fn make_kxk(k: usize, entries: &[f32]) -> TransportOperator {
        TransportOperator::from_row_major(k, entries.to_vec()).unwrap()
    }

    #[test]
    fn identity_passes_g6() {
        // Identity operator: tangent = I (the identity maps e ± ε·e_j linearly,
        // so the Jacobian is the identity). Diagonal mode matrix → 0% spurious.
        let id = TransportOperator::identity(4);
        let report = spectral_audit(&id);
        assert!(
            report.passes_g6(),
            "identity should pass G6, got spurious = {}",
            report.spurious_ratio
        );
        assert!(
            report.spurious_ratio < 1e-3,
            "identity spurious {} should be ~0",
            report.spurious_ratio
        );
    }

    #[test]
    fn scaled_identity_passes_g6() {
        // 2*I has the same mode structure as I (just scaled) → still diagonal
        // in EVERY basis, including DCT-II.
        let k = 4;
        let mut data = vec![0.0f32; k * k];
        for i in 0..k {
            data[i * k + i] = 2.0;
        }
        let op = TransportOperator::from_row_major(k, data).unwrap();
        let report = spectral_audit(&op);
        assert!(
            report.passes_g6(),
            "scaled identity should pass G6, got spurious = {}",
            report.spurious_ratio
        );
    }

    /// Build an operator that is diagonal in the DCT-II basis.
    ///
    /// `C = Σ_m λ_m · φ_m φ_m^T` where `φ_m` are normalized DCT-II basis vectors.
    /// By construction, `C·φ_n = λ_n·φ_n`, so the mode matrix is purely
    /// diagonal → 0% spurious coupling. This is what a "clean" spectral
    /// transport operator looks like (e.g. a well-conditioned FuncAttn output).
    fn make_dct2_diagonal_operator(k: usize, eigenvalues: &[f32]) -> TransportOperator {
        // Normalized DCT-II basis: φ_m[j] = α_m · cos(π·m·(2j+1)/(2k))
        // where α_0 = sqrt(1/k), α_m = sqrt(2/k) for m > 0.
        let m = eigenvalues.len().min(k);
        let mut basis = vec![0.0f32; m * k];
        let denom = 2.0 * k as f32;
        for mode in 0..m {
            let alpha = if mode == 0 {
                (1.0 / k as f32).sqrt()
            } else {
                (2.0 / k as f32).sqrt()
            };
            for j in 0..k {
                let phase = std::f32::consts::PI * (mode as f32) * (2 * j + 1) as f32 / denom;
                basis[mode * k + j] = alpha * phase.cos();
            }
        }

        // C[i,j] = Σ_m λ_m · φ_m[i] · φ_m[j]
        let mut data = vec![0.0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0f32;
                for mode in 0..m {
                    s += eigenvalues[mode] * basis[mode * k + i] * basis[mode * k + j];
                }
                data[i * k + j] = s;
            }
        }
        TransportOperator::from_row_major(k, data).unwrap()
    }

    #[test]
    fn dct2_diagonal_operator_passes_g6() {
        // An operator that is diagonal in the DCT-II basis with moderate
        // eigenvalues (near 1 — what a well-conditioned composite looks like).
        // By construction, it has ZERO cross-mode coupling.
        let op = make_dct2_diagonal_operator(4, &[1.0, 0.9, 0.8, 0.7]);
        let report = spectral_audit(&op);
        assert!(
            report.passes_g6(),
            "DCT-II-diagonal operator should pass G6, got spurious = {}",
            report.spurious_ratio
        );
    }

    #[test]
    fn standard_basis_diagonal_couples_dct2_modes() {
        // A non-uniform diagonal operator in the STANDARD basis has DIFFERENT
        // entries on the diagonal. In the DCT-II basis, this is NOT diagonal —
        // the non-uniform scaling mixes modes. This is CORRECT behavior: the
        // audit flags it as having spurious coupling, because a spectral
        // transport operator should act per-mode, not per-coordinate.
        let op = make_kxk(4, &[2.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.3]);
        let report = spectral_audit(&op);
        // This SHOULD fail G6 — non-uniform diagonal mixes DCT-II modes.
        assert!(
            !report.passes_g6(),
            "non-uniform standard-basis diagonal should FAIL G6 (it couples DCT-II modes), got spurious = {}",
            report.spurious_ratio
        );
    }

    #[test]
    fn random_operator_fails_g6() {
        // A dense random-ish operator: lots of cross-mode coupling.
        let op = make_kxk(4, &[
            0.7, -0.4, 0.2, 0.9,
            -0.3, 0.6, -0.8, 0.1,
            0.5, -0.2, 0.4, -0.7,
            -0.6, 0.8, 0.3, -0.5,
        ]);
        let report = spectral_audit(&op);
        assert!(
            !report.passes_g6(),
            "random operator should FAIL G6, got spurious = {}",
            report.spurious_ratio
        );
        assert!(
            report.spurious_ratio > SPURIOUS_THRESHOLD,
            "random spurious {} should be > {}",
            report.spurious_ratio,
            SPURIOUS_THRESHOLD
        );
    }

    #[test]
    fn clean_composite_passes_g6() {
        // Composite of two DCT-II-diagonal operators with moderate eigenvalues.
        // Each is clean in DCT-II space; their composite (product of two
        // DCT-II-diagonal matrices with the SAME eigenvectors) is also
        // DCT-II-diagonal → still clean.
        let a = make_dct2_diagonal_operator(4, &[0.95, 0.90, 0.85, 0.80]);
        let b = make_dct2_diagonal_operator(4, &[1.0, 0.95, 0.90, 0.85]);
        let composite = compose_chain(&[a, b]).unwrap();
        let report = spectral_audit(&composite);
        assert!(
            report.passes_g6(),
            "clean DCT-II composite should pass G6, got spurious = {}",
            report.spurious_ratio
        );
    }

    #[test]
    fn dct2_basis_orthogonal_structure() {
        // DCT-II basis vectors should be mutually orthogonal (up to the
        // unnormalized scaling). Verify ⟨φ_0, φ_1⟩ ≈ 0.
        let k = 8;
        let modes = 4;
        let mut basis = vec![0.0f32; modes * k];
        compute_dct2_basis(&mut basis, modes, k);

        let phi_0 = &basis[0..k];
        let phi_1 = &basis[k..2 * k];
        let cross = dot(phi_0, phi_1, k);
        assert!(
            cross.abs() < 1e-5,
            "DCT-II basis not orthogonal: ⟨φ_0, φ_1⟩ = {}",
            cross
        );
    }

    #[test]
    fn audit_report_mode_matrix_shape() {
        let id = TransportOperator::identity(8);
        let report = spectral_audit(&id);
        assert_eq!(report.modes, DEFAULT_FOURIER_MODES);
        assert_eq!(report.mode_matrix.len(), DEFAULT_FOURIER_MODES * DEFAULT_FOURIER_MODES);
        assert_eq!(report.mode_gains.len(), DEFAULT_FOURIER_MODES);
        assert_eq!(report.k, 8);
    }
}
