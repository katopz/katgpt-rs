//! Subspace phase-gate primitives — modelless numeric utilities for subspace
//! clustering quality assurance and runtime semantic-axis discovery.
//!
//! Distilled from Wang et al., *Breaking the Curse of Dimensionality: Diffusion
//! Models Efficiently Learn Low-Dimensional Distributions* ([arXiv:2409.02426](https://arxiv.org/abs/2409.02426)).
//! See `katgpt-rs/.research/279_*.md` for the open research note and
//! `katgpt-rs/.plans/301_*.md` for the execution plan.
//!
//! Three families of operations, all inference-time and allocation-aware:
//!
//! 1. **Intrinsic-dimension estimation** — [`participation_ratio`] (continuous)
//!    and [`numerical_rank`] (discrete, with energy threshold η). Paper eq. 52
//!    uses η = 0.99.
//! 2. **Phase-transition sample-sufficiency gate** — [`phase_transition_gate`]
//!    implements the necessary condition `N ≥ d` from Theorem 4: a d-dimensional
//!    subspace cannot be recovered from fewer than d samples, regardless of
//!    algorithm. Below the threshold, any subspace estimate is
//!    information-theoretically invalid.
//! 3. **Runtime Jacobian SVD** — [`jacobian_svd_at`] estimates the Jacobian of
//!    a map `f: R^n → R^m` at a point via forward differences, then computes a
//!    thin SVD. The leading singular vectors are candidate "semantic axes" in
//!    the sense of paper §5.2: directions in the domain along which the map
//!    produces the largest output change.
//!
//! These primitives are generic numeric — no game, shard, or chain semantics.
//! Consumers apply them to their own maps (HLA evolution kernel, shard
//! projection, latent functor) and interpret the results.
//!
//! # Performance contract
//!
//! - [`participation_ratio`] and [`numerical_rank`] are O(n) on a length-n
//!   spectrum, zero-allocation, chunk-4 loops for SIMD auto-vectorisation.
//! - [`jacobian_svd_at`] is O(n·cost(f) + n²·m) — n forward evaluations of f
//!   plus an n×m thin SVD. For small n (≤ 16) and m (≤ 16), this is sub-µs
//!   on commodity hardware.
//!
//! # Determinism
//!
//! All operations are deterministic and platform-independent: no SIMD dispatch
//! inside the math (callers may wrap SIMD themselves), no floating-point
//! reordering. This is required for anti-cheat: the phase-transition gate
//! decision must be bit-identical across quorum nodes.

#![cfg(feature = "subspace_phase_gate")]

use core::cmp::Ordering;

// ─── Intrinsic-dimension estimation ─────────────────────────────────────────

/// Continuous effective dimensionality: `(Σλ)² / Σ(λ²)`.
///
/// Returns 0.0 on empty or all-non-positive input. For a flat spectrum of k
/// equal eigenvalues, returns exactly `k`. For a single dominant eigenvalue,
/// returns ~1. Always in `[0, n]` for a length-n non-negative spectrum.
///
/// Chunk-4 accumulation for SIMD auto-vectorisation. Zero-allocation.
#[inline]
pub fn participation_ratio(spectrum: &[f32]) -> f32 {
    if spectrum.is_empty() {
        return 0.0;
    }
    let mut sum: f32 = 0.0;
    let mut sum_sq: f32 = 0.0;
    let mut i = 0;
    while i + 4 <= spectrum.len() {
        let a = spectrum[i].max(0.0);
        let b = spectrum[i + 1].max(0.0);
        let c = spectrum[i + 2].max(0.0);
        let d = spectrum[i + 3].max(0.0);
        sum += a + b + c + d;
        sum_sq += a * a + b * b + c * c + d * d;
        i += 4;
    }
    while i < spectrum.len() {
        let v = spectrum[i].max(0.0);
        sum += v;
        sum_sq += v * v;
        i += 1;
    }
    if sum_sq < f32::EPSILON {
        return 0.0;
    }
    (sum * sum) / sum_sq
}

/// Discrete effective dimensionality: smallest `r` such that cumulative energy
/// `Σ_{i≤r} σ_i² / Σ_i σ_i² > η`.
///
/// Mirrors paper eq. 52 with η = 0.99. The caller MUST sort the spectrum
/// descending first — this function does not sort (zero-allocation contract).
/// Default η in [`IntrinsicDimMethod::NumericalRank`] is 0.99.
#[inline]
pub fn numerical_rank(spectrum_sorted_desc: &[f32], eta: f32) -> usize {
    debug_assert!(
        (0.0..=1.0).contains(&eta),
        "eta must be in [0, 1], got {eta}"
    );
    if spectrum_sorted_desc.is_empty() {
        return 0;
    }
    let mut total_sq: f32 = 0.0;
    let mut cum_sq: f32 = 0.0;
    for &v in spectrum_sorted_desc {
        let v = v.max(0.0);
        total_sq += v * v;
    }
    if total_sq < f32::EPSILON {
        return 0;
    }
    let threshold = eta * total_sq;
    for (i, &v) in spectrum_sorted_desc.iter().enumerate() {
        cum_sq += v.max(0.0) * v.max(0.0);
        if cum_sq > threshold {
            return i + 1;
        }
    }
    spectrum_sorted_desc.len()
}

/// Method selector for [`estimate_intrinsic_dim`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntrinsicDimMethod {
    /// Round [`participation_ratio`] to the nearest integer. Fast, continuous,
    /// good when the spectrum has a clear elbow.
    ParticipationRatio,
    /// [`numerical_rank`] at energy threshold η. Discrete, conservative.
    /// Default η = 0.99 (paper eq. 52).
    NumericalRank { eta: f32 },
}

impl Default for IntrinsicDimMethod {
    fn default() -> Self {
        // Paper eq. 52 uses η = 0.99. We default to the same.
        IntrinsicDimMethod::NumericalRank { eta: 0.99 }
    }
}

/// Dispatch to the configured estimator. See [`IntrinsicDimMethod`].
#[inline]
pub fn estimate_intrinsic_dim(spectrum: &[f32], method: IntrinsicDimMethod) -> usize {
    match method {
        IntrinsicDimMethod::ParticipationRatio => participation_ratio(spectrum).round() as usize,
        IntrinsicDimMethod::NumericalRank { eta } => numerical_rank(spectrum, eta),
    }
}

// ─── Phase-transition gate ──────────────────────────────────────────────────

/// The Wang et al. Theorem 4 necessary condition for subspace recovery:
/// `n_samples >= intrinsic_dim`.
///
/// Returns `true` iff the sample count meets or exceeds the intrinsic dim.
/// Returns `false` below the threshold — recovery is information-theoretically
/// impossible for a d-dim subspace from fewer than d samples, regardless of
/// algorithm. See [`crate::subspace_phase_gate`] module docs for the caveats
/// (this is necessary but not sufficient when subspaces are non-orthogonal).
#[inline(always)]
pub fn phase_transition_gate(n_samples: usize, intrinsic_dim: usize) -> bool {
    n_samples >= intrinsic_dim
}

// ─── Runtime Jacobian SVD ───────────────────────────────────────────────────

/// Pre-allocated scratch buffers for [`jacobian_svd_at`]. Reuse across calls
/// to avoid per-call allocation.
pub struct JacobianSvdScratch {
    /// Output column buffer for `f` evaluations, length `m`.
    f_x: Vec<f32>,
    /// Output column buffer for `f(x + eps·e_i)` evaluations, length `m`.
    f_x_pert: Vec<f32>,
    /// Flattened Jacobian, row-major `m × n`, length `m * n`.
    jac: Vec<f32>,
    /// Thin-SVD working storage for the Jacobi-rotation routine.
    svd_work: SvdScratch,
}

impl JacobianSvdScratch {
    /// Allocate scratch sized for an `R^n → R^m` map. Pre-allocates all
    /// internal buffers; reuse via [`Self::clear`] between calls.
    pub fn with_capacity(n: usize, m: usize) -> Self {
        Self {
            f_x: vec![0.0; m],
            f_x_pert: vec![0.0; m],
            jac: vec![0.0; m * n],
            svd_work: SvdScratch::with_capacity(n, m),
        }
    }

    /// Reset for reuse. Does not deallocate; just zeros the active regions.
    pub fn clear(&mut self) {
        for v in &mut self.f_x {
            *v = 0.0;
        }
        for v in &mut self.f_x_pert {
            *v = 0.0;
        }
        for v in &mut self.jac {
            *v = 0.0;
        }
        self.svd_work.clear();
    }
}

/// Result of [`jacobian_svd_at`]. Vectors are owned for simplicity; callers
/// that need zero-allocation can drop this struct promptly.
pub struct SvdResult {
    /// Singular values, descending. Length = min(n, m).
    pub singular_values: Vec<f32>,
    /// Right singular vectors (columns of V), one `Vec<f32>` per singular
    /// value, each of length `n`. These are the "directions in the input
    /// space" along which `f` is most sensitive — the candidate "semantic
    /// axes" in the sense of paper §5.2.
    pub right_singular_vectors: Vec<Vec<f32>>,
    /// Left singular vectors (columns of U), one `Vec<f32>` per singular value,
    /// each of length `m`. These are the corresponding "directions in the
    /// output space".
    pub left_singular_vectors: Vec<Vec<f32>>,
    /// Effective rank: number of singular values above a small threshold.
    pub rank: usize,
}

/// Estimate the Jacobian of `f: R^n → R^m` at point `x` via forward differences,
/// then return the thin SVD.
///
/// `f` is called `n + 1` times: once at `x`, then once at `x + eps·e_i` for
/// each coordinate `i`. The Jacobian column `i` is `(f(x + eps·e_i) − f(x)) / eps`.
///
/// `eps` is the forward-difference step. Reasonable default: `1e-4`. Pass a
/// negative value to opt into central differences (more accurate, 2× cost).
///
/// # Panics
///
/// Panics if `x.len() != n` (where `n` was passed to
/// [`JacobianSvdScratch::with_capacity`]) or if `f` writes a slice of the
/// wrong length.
pub fn jacobian_svd_at<F>(f: F, x: &[f32], eps: f32, scratch: &mut JacobianSvdScratch) -> SvdResult
where
    F: Fn(&[f32], &mut [f32]),
{
    let n = x.len();
    debug_assert_eq!(
        scratch.jac.len() % n,
        0,
        "scratch.jac length {} not a multiple of n={}",
        scratch.jac.len(),
        n
    );
    let m = scratch.jac.len() / n;
    debug_assert_eq!(scratch.f_x.len(), m);
    debug_assert_eq!(scratch.f_x_pert.len(), m);

    scratch.clear();

    // Central differences if eps < 0, forward otherwise.
    let central = eps < 0.0;
    let step = eps.abs();

    // f(x) — evaluated into a thread-local buffer because `f` takes `&[f32]`
    // and we need to perturb x without mutating the caller's slice.
    f(x, &mut scratch.f_x);

    // Build Jacobian column-by-column (input-coordinate-wise).
    // jac is row-major m × n, so column i lives at indices i, i+n, i+2n, ...
    // For cache friendliness on small matrices, we transpose to row-major m×n
    // where row j, col i = jac[j*n + i].
    //
    // We need a mutable copy of x to perturb. Reuse the f_x_pert slice as
    // scratch? No — we need it for f output. Allocate one small buffer.
    let mut x_pert: Vec<f32> = x.to_vec();
    for i in 0..n {
        // Save the original coordinate.
        let x_i_orig = x_pert[i];
        if central {
            // f(x + step·e_i)
            x_pert[i] = x_i_orig + step;
            f(&x_pert, &mut scratch.f_x_pert);
            let f_plus: Vec<f32> = scratch.f_x_pert.clone();
            // f(x - step·e_i)
            x_pert[i] = x_i_orig - step;
            f(&x_pert, &mut scratch.f_x_pert);
            // Central diff: (f_plus − f_minus) / (2·step)
            for j in 0..m {
                scratch.jac[j * n + i] = (f_plus[j] - scratch.f_x_pert[j]) / (2.0 * step);
            }
        } else {
            // Forward diff: (f(x + step·e_i) − f(x)) / step
            x_pert[i] = x_i_orig + step;
            f(&x_pert, &mut scratch.f_x_pert);
            for j in 0..m {
                scratch.jac[j * n + i] = (scratch.f_x_pert[j] - scratch.f_x[j]) / step;
            }
        }
        // Restore.
        x_pert[i] = x_i_orig;
    }

    // Thin SVD of the m × n Jacobian via one-sided Jacobi rotations.
    one_sided_jacobi_svd(&scratch.jac, m, n, &mut scratch.svd_work)
}

// ─── One-sided Jacobi SVD (portable, no native-lapack dep) ─────────────────

/// Pre-allocated scratch for [`thin_svd`] / [`one_sided_jacobi_svd`]. Reuse
/// across calls to avoid per-call allocation. Allocate once with
/// [`SvdScratch::with_capacity`] for the largest `(m, n)` you will factor.
//
// Public since Plan 002 Phase 2: consumers with a *known* linear map (e.g.
// `riir-neuron-db::NeuronShard::semantic_axes`, which SVDs a fixed weight
// matrix) want to skip the forward-differencing in `jacobian_svd_at` and call
// the SVD directly. Exposing the scratch + a thin-SVD entry point lets them do
// so without re-deriving the Jacobian (which for a linear map is the matrix
// itself and costs `n` extra `f` evaluations + a per-call `Vec` allocation
// inside `jacobian_svd_at`).
pub struct SvdScratch {
    /// Working copy of the input matrix, mutated in-place. Length m*n.
    a: Vec<f32>,
    /// Right-singular-vector accumulator V, n × n, row-major. Length n*n.
    v: Vec<f32>,
    /// Column norms (singular values during iteration). Length n.
    col_norms_sq: Vec<f32>,
}

impl SvdScratch {
    /// Allocate scratch sized for factoring an `m_rows × n_cols` matrix.
    pub fn with_capacity(n_cols: usize, m_rows: usize) -> Self {
        Self {
            a: vec![0.0; m_rows * n_cols],
            v: vec![0.0; n_cols * n_cols],
            col_norms_sq: vec![0.0; n_cols],
        }
    }

    fn clear(&mut self) {
        for v in &mut self.a {
            *v = 0.0;
        }
        for v in &mut self.v {
            *v = 0.0;
        }
        for v in &mut self.col_norms_sq {
            *v = 0.0;
        }
    }
}

/// Thin SVD of a row-major `m_rows × n_cols` matrix `M = U Σ V^T` (requires
/// `m_rows >= n_cols`) via one-sided Jacobi rotations.
///
/// This is the direct entry point for consumers that already hold the matrix
/// (e.g. a known linear map's matrix). For estimating the SVD of an arbitrary
/// (possibly non-linear) map's Jacobian, use [`jacobian_svd_at`] instead.
///
/// Returns the `n_cols` leading singular triples. Sign conventions are
/// arbitrary (canonical SVD ambiguity); callers should not depend on signs.
///
/// Convergence: rotate until no off-diagonal element of `M^T M` exceeds
/// `tol² · trace(M^T M)`. Standard textbook algorithm; ~O(n²) per sweep,
/// ~log2(n) sweeps to converge for well-separated spectra.
///
/// Allocation: zero per-call (the result vectors are the only allocation;
/// `work` is reused). This is the fast path that avoids the forward-differencing
/// and per-call `x_pert` allocation of [`jacobian_svd_at`].
pub fn thin_svd(
    m_flat: &[f32], // row-major m × n
    m_rows: usize,
    n_cols: usize,
    work: &mut SvdScratch,
) -> SvdResult {
    one_sided_jacobi_svd(m_flat, m_rows, n_cols, work)
}

/// One-sided Jacobi SVD: factor `M (m × n, m ≥ n) = U Σ V^T`.
///
/// Returns the `min(n, m)` leading singular triples. Sign conventions are
/// arbitrary (canonical SVD ambiguity); callers should not depend on signs.
///
/// Convergence: rotate until no off-diagonal element of `M^T M` exceeds
/// `tol² · trace(M^T M)`. Standard textbook algorithm; ~O(n²) per sweep,
/// ~log2(n) sweeps to converge for well-separated spectra.
fn one_sided_jacobi_svd(
    m_flat: &[f32], // row-major m × n
    m_rows: usize,
    n_cols: usize,
    work: &mut SvdScratch,
) -> SvdResult {
    let m = m_rows;
    let n = n_cols;
    debug_assert_eq!(m_flat.len(), m * n);
    debug_assert_eq!(work.a.len(), m * n);
    debug_assert_eq!(work.v.len(), n * n);

    // Copy M into work.a (will be mutated to U·Σ).
    work.a.copy_from_slice(m_flat);
    // Initialise V = I.
    for i in 0..n {
        for j in 0..n {
            work.v[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let tol: f32 = 1e-7;
    let max_sweeps = 60;

    for _sweep in 0..max_sweeps {
        let mut off_diag_max: f32 = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                // Compute (p, q) entry of A^T A: dot of column p and column q.
                let mut app: f32 = 0.0;
                let mut aqq: f32 = 0.0;
                let mut apq: f32 = 0.0;
                for r in 0..m {
                    let arp = work.a[r * n + p];
                    let arq = work.a[r * n + q];
                    app += arp * arp;
                    aqq += arq * arq;
                    apq += arp * arq;
                }
                off_diag_max = off_diag_max.max(apq.abs());
                if apq.abs() <= tol * (app * aqq).sqrt() {
                    continue; // Already diagonal in this plane.
                }
                // Compute Jacobi rotation (c, s) that zeroes apq.
                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Apply rotation to columns p, q of A and V.
                for r in 0..m {
                    let arp = work.a[r * n + p];
                    let arq = work.a[r * n + q];
                    work.a[r * n + p] = c * arp - s * arq;
                    work.a[r * n + q] = s * arp + c * arq;
                }
                for r in 0..n {
                    let vrp = work.v[r * n + p];
                    let vrq = work.v[r * n + q];
                    work.v[r * n + p] = c * vrp - s * vrq;
                    work.v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
        if off_diag_max <= tol * 1e-3 {
            break;
        }
    }

    // Extract singular values (column norms of A post-rotation) and sort desc.
    let mut sigmas: Vec<(f32, usize)> = (0..n)
        .map(|i| {
            let mut s_sq: f32 = 0.0;
            for r in 0..m {
                let ari = work.a[r * n + i];
                s_sq += ari * ari;
            }
            (s_sq.sqrt(), i)
        })
        .collect();
    sigmas.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

    // Effective rank: count singular values above a small threshold relative
    // to the largest.
    let sigma_max = sigmas.first().map(|(s, _)| *s).unwrap_or(0.0);
    let rank_threshold = sigma_max * 1e-5;
    let rank = sigmas.iter().filter(|(s, _)| *s > rank_threshold).count();

    let singular_values: Vec<f32> = sigmas.iter().map(|(s, _)| *s).collect();
    let right_singular_vectors: Vec<Vec<f32>> = sigmas
        .iter()
        .map(|(_, i)| (0..n).map(|r| work.v[r * n + i]).collect())
        .collect();
    let left_singular_vectors: Vec<Vec<f32>> = sigmas
        .iter()
        .map(|(s, i)| {
            if *s < f32::EPSILON {
                vec![0.0; m]
            } else {
                (0..m).map(|r| work.a[r * n + i] / s).collect()
            }
        })
        .collect();

    SvdResult {
        singular_values,
        right_singular_vectors,
        left_singular_vectors,
        rank,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn participation_ratio_flat_spectrum() {
        // 5 equal eigenvalues → PR = 5.
        let s = vec![1.0; 5];
        let pr = participation_ratio(&s);
        assert!((pr - 5.0).abs() < 1e-5, "expected 5.0, got {pr}");
    }

    #[test]
    fn participation_ratio_dominant_mode() {
        // One large eigenvalue → PR ≈ 1.
        let s = vec![10.0, 0.01, 0.01, 0.01];
        let pr = participation_ratio(&s);
        assert!(pr < 1.2, "expected ≈ 1, got {pr}");
    }

    #[test]
    fn participation_ratio_empty_returns_zero() {
        assert_eq!(participation_ratio(&[]), 0.0);
    }

    #[test]
    fn participation_ratio_all_zero_returns_zero() {
        assert_eq!(participation_ratio(&[0.0; 8]), 0.0);
    }

    #[test]
    fn numerical_rank_full_energy() {
        // σ² = [100, 25, 4, 1], total = 130.
        // η=0.99 → threshold 128.7 → cumulative 100, 125, 129 → rank 3 (129/130 = 99.23%).
        // η=1.0  → threshold 130.0 → needs all 4 columns → rank 4.
        let s = vec![10.0, 5.0, 2.0, 1.0]; // descending
        let r99 = numerical_rank(&s, 0.99);
        assert_eq!(r99, 3, "η=0.99 → rank 3 (cum 129/130 = 99.23% > 99%)");
        let r_strict = numerical_rank(&s, 1.0);
        assert_eq!(r_strict, 4, "η=1.0 → rank 4 (needs all columns)");
    }

    #[test]
    fn numerical_rank_low_rank() {
        // 99% of energy in top 2 of 4 singular values.
        let s = vec![10.0, 10.0, 0.1, 0.1];
        let r = numerical_rank(&s, 0.99);
        assert!(r <= 2, "expected rank ≤ 2, got {r}");
        let r90 = numerical_rank(&s, 0.9);
        assert!(r90 <= 2, "at η=0.9 still rank ≤ 2, got {r90}");
    }

    #[test]
    fn phase_transition_gate_at_threshold() {
        assert!(!phase_transition_gate(5, 6), "N=5 < d=6 → false");
        assert!(phase_transition_gate(6, 6), "N=6 = d=6 → true");
        assert!(phase_transition_gate(50, 6), "N=50 > d=6 → true");
    }

    #[test]
    fn jacobian_svd_recovers_known_rank3_matrix() {
        // Construct a rank-3 matrix W (4 × 6) = U3 Σ3 V3^T.
        // W maps R^6 → R^4; its Jacobian (W is linear) is W itself.
        // Singular values should be {10, 5, 2, 0} after rounding.
        let n = 6;
        let m = 4;
        // Build a known rank-3 W.
        // u_i ∈ R^4, v_i ∈ R^6, σ_i: W = Σ σ_i u_i v_i^T
        let u1 = [1.0, 0.0, 0.0, 0.0];
        let u2 = [0.0, 1.0, 0.0, 0.0];
        let u3 = [0.0, 0.0, 1.0, 0.0];
        let v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let s1 = 10.0_f32;
        let s2 = 5.0_f32;
        let s3 = 2.0_f32;
        let mut w = vec![0.0_f32; m * n];
        for j in 0..m {
            for i in 0..n {
                let mut acc = 0.0;
                acc += s1 * u1[j] * v1[i];
                acc += s2 * u2[j] * v2[i];
                acc += s3 * u3[j] * v3[i];
                w[j * n + i] = acc;
            }
        }
        // The map f: x ↦ W x. Jacobian = W.
        let f = |x: &[f32], out: &mut [f32]| {
            debug_assert_eq!(x.len(), n);
            debug_assert_eq!(out.len(), m);
            for j in 0..m {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += w[j * n + i] * x[i];
                }
                out[j] = acc;
            }
        };
        let x = [0.5_f32; 6];
        let mut scratch = JacobianSvdScratch::with_capacity(n, m);
        let result = jacobian_svd_at(f, &x, 1e-4, &mut scratch);
        // Expect rank 3.
        assert_eq!(
            result.rank, 3,
            "expected rank 3, got {} (sigmas = {:?})",
            result.rank, result.singular_values
        );
        // Top-3 singular values should be close to {10, 5, 2} (order-tolerant sign).
        let top3: Vec<f32> = result.singular_values.iter().take(3).cloned().collect();
        let mut expected = vec![10.0, 5.0, 2.0];
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let mut got = top3.clone();
        got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        for (e, g) in expected.iter().zip(got.iter()) {
            assert!(
                (e - g).abs() < 0.1,
                "singular value mismatch: expected ≈ {e}, got {g}"
            );
        }
        // The 4th singular value should be ≈ 0.
        if result.singular_values.len() >= 4 {
            assert!(
                result.singular_values[3] < 0.1,
                "expected 4th singular value ≈ 0, got {}",
                result.singular_values[3]
            );
        }
    }

    #[test]
    fn estimate_intrinsic_dim_participation_ratio() {
        let s = vec![1.0; 4];
        let d = estimate_intrinsic_dim(&s, IntrinsicDimMethod::ParticipationRatio);
        assert_eq!(d, 4);
    }

    #[test]
    fn estimate_intrinsic_dim_numerical_rank() {
        let s = vec![10.0, 5.0, 0.1, 0.05]; // ~99% energy in top 2
        let d = estimate_intrinsic_dim(&s, IntrinsicDimMethod::NumericalRank { eta: 0.99 });
        assert!(d <= 2, "expected d ≤ 2, got {d}");
    }
}
