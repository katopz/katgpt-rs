//! GD-free transport chain for the leakage probe (Issue 736, Research 540).
//!
//! The chain maps one embedding sample into another embedding space using
//! ONLY closed-form / deterministic-iterative steps — no gradient descent,
//! no RNG (the subspace-iteration seed is a fixed LCG, deterministic across
//! runs and platforms):
//!
//! 1. **Subspace projection** — deterministic subspace (orthogonal)
//!    iteration on the covariance pulls the top-`k` principal subspace out
//!    of a `d`-dim space without a full `d×d` eigendecomposition.
//! 2. **PCA whitening** — the small `k×k` projected covariance is
//!    eigendecomposed (Householder + implicit-shift QL via the always-on
//!    [`crate::linalg::symmetric_eig`]) and whitened to unit variance, then
//!    rows are re-projected onto the unit sphere.
//! 3. **Entropic Sinkhorn self-labeling** — optimal transport between the
//!    two whitened point clouds supplies UNPAIRED pseudo-correspondences.
//! 4. **Orthogonal Procrustes** — the polar factor of the pseudo-paired
//!    cross-covariance (again via [`crate::linalg::symmetric_eig`]) is the
//!    closed-form best orthogonal map; steps 3–4 alternate for a few rounds.
//!
//! This is a *probe*-grade transport: the paper (arXiv:2505.12540) shows
//! modelless OT baselines sit at ≈random cross-backbone while a trained
//! nonlinear translator reaches rank ≈1 — so this module's quality ceiling
//! is the honest OT tier. The leak score it feeds
//! ([`crate::leakage_probe`]) is calibrated to be conservative accordingly:
//! a `Low` verdict under a weak transport is exactly the point (the probe
//! measures *transferable* attribute structure, and a transport that cannot
//! align geometry cannot transfer attributes either).
//!
//! Audit-cadence allocation is allowed (the `analytic_lattice::audit`
//! precedent): only per-tick hot paths must be alloc-free.

use crate::linalg::{SymmetricEigScratch, symmetric_eig};

/// Cap on raw input dimensionality per space. The covariance build is
/// O(n·d²) and O(d²) memory; audit-cadence budgets stop long before this.
pub const MAX_DIM: usize = 1024;

/// Cap on the common transport dimensionality (the whitened `k`).
pub const MAX_LATENT_DIM: usize = 64;

/// Minimum samples per side for a stable covariance + non-degenerate
/// Sinkhorn plan.
pub const MIN_SAMPLES: usize = 16;

/// Fixed LCG seed for the subspace-iteration start basis. Constant — the
/// whole chain must be reproducible byte-for-byte.
const SUBSPACE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Maximum iterations per eigenvalue handed to `symmetric_eig` (NR uses 30;
/// doubled for margin — the matrices here are small and well-conditioned
/// after the ridge floor).
const EIG_MAX_ITERS: usize = 60;

/// Ridge floor applied to eigenvalues before inverse-sqrt (whitening and
/// polar factor). Prevents division by ~0 on degenerate subspaces.
const EIG_FLOOR: f64 = 1e-6;

/// A deterministic linear-congruential stream over `u64` — enough for a
/// reproducible start basis; NOT a statistical RNG.
struct Lcg(u64);

impl Lcg {
    #[inline]
    fn next_f32_unit(&mut self) -> f32 {
        // Numerical Recipes constants (64-bit Kronecker).
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Top 24 bits → [-1, 1).
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / 8_388_608.0_f32) - 1.0
    }
}

/// Errors from the transport chain. Wrapped by the probe's error enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransportError {
    DimMismatch,
    TooFewSamples,
    DimTooLarge,
    LatentDimTooLarge,
    LatentDimExceedsSamples,
}

/// Reused working buffers for the transport chain (audit-cadence: allocate
/// once, reuse across probes).
#[derive(Default)]
pub(crate) struct TransportScratch {
    pub(crate) cov: Vec<f32>,
    pub(crate) basis: Vec<f32>,
    pub(crate) basis_tmp: Vec<f32>,
    pub(crate) row_buf: Vec<f32>,
    pub(crate) eig: SymmetricEigScratch,
    pub(crate) eigvals: Vec<f64>,
    pub(crate) eigvecs: Vec<f64>,
    pub(crate) sym: Vec<f64>,
    pub(crate) w: Vec<f32>,
}

impl TransportScratch {
    pub(crate) fn new() -> Self {
        Self::default()
    }
}

/// Projects row-major `x` (`n × d`) onto its top-`k` principal subspace,
/// writing an `n × k` row-major result into `out`. Deterministic: the start
/// basis is a fixed-seed LCG draw, orthonormalized before the first power
/// step.
///
/// Cost: O(n·d²) for the covariance + O(s·(d²·k)) for `s` subspace
/// iterations — no `d×d` eigendecomposition.
pub(crate) fn project_topk(
    x: &[f32],
    n: usize,
    d: usize,
    k: usize,
    iters: usize,
    out: &mut Vec<f32>,
    sc: &mut TransportScratch,
) {
    debug_assert_eq!(x.len(), n * d);
    debug_assert!(k <= d && k <= MAX_LATENT_DIM);

    // Mean-center into `out` first (it becomes the working copy).
    out.clear();
    out.extend_from_slice(x);
    for j in 0..d {
        let mut mean = 0.0f32;
        for i in 0..n {
            mean += x[i * d + j];
        }
        mean /= n as f32;
        for i in 0..n {
            out[i * d + j] -= mean;
        }
    }

    // Covariance C = XᵀX / n (d×d, symmetric).
    let cov = &mut sc.cov;
    cov.clear();
    cov.resize(d * d, 0.0);
    for i in 0..n {
        let row = &out[i * d..(i + 1) * d];
        for a in 0..d {
            let ra = row[a];
            for b in a..d {
                cov[a * d + b] += ra * row[b];
            }
        }
    }
    let inv_n = 1.0 / n as f32;
    for a in 0..d {
        for b in a..d {
            let v = cov[a * d + b] * inv_n;
            cov[a * d + b] = v;
            cov[b * d + a] = v;
        }
    }

    // Deterministic start basis (d×k), then orthonormalize.
    let basis = &mut sc.basis;
    let basis_tmp = &mut sc.basis_tmp;
    basis.clear();
    basis.resize(d * k, 0.0);
    let mut lcg = Lcg(SUBSPACE_SEED);
    for v in basis.iter_mut() {
        *v = lcg.next_f32_unit();
    }
    orthonormalize_columns(basis, d, k);

    // Subspace iteration: B ← QR(C·B), `iters` times.
    basis_tmp.clear();
    basis_tmp.resize(d * k, 0.0);
    for _ in 0..iters {
        for c in 0..k {
            for r in 0..d {
                let mut acc = 0.0f32;
                for t in 0..d {
                    acc += cov[r * d + t] * basis[t * k + c];
                }
                basis_tmp[r * k + c] = acc;
            }
        }
        orthonormalize_columns(basis_tmp, d, k);
        basis.copy_from_slice(basis_tmp);
    }

    // Project: out = X_centered · basis  (n×d · d×k → n×k). Row-by-row
    // through a scratch copy — `out` is overwritten in place and the write
    // region [i·k, i·k+k) would otherwise alias the read region [i·d, i·d+d)
    // for early rows.
    let row = &mut sc.row_buf;
    row.clear();
    row.resize(d, 0.0);
    for i in 0..n {
        row.copy_from_slice(&out[i * d..(i + 1) * d]);
        for c in 0..k {
            let mut acc = 0.0f32;
            for t in 0..d {
                acc += row[t] * basis[t * k + c];
            }
            out[i * k + c] = acc;
        }
    }
    out.truncate(n * k);
}

/// In-place modified Gram-Schmidt on the `k` columns of a row-major `d×k`
/// buffer.
fn orthonormalize_columns(b: &mut [f32], d: usize, k: usize) {
    for c in 0..k {
        for p in 0..c {
            let mut dot = 0.0f32;
            for r in 0..d {
                dot += b[r * k + c] * b[r * k + p];
            }
            for r in 0..d {
                b[r * k + c] -= dot * b[r * k + p];
            }
        }
        let mut norm = 0.0f32;
        for r in 0..d {
            norm += b[r * k + c] * b[r * k + c];
        }
        let norm = norm.sqrt().max(1e-12);
        for r in 0..d {
            b[r * k + c] /= norm;
        }
    }
}

/// PCA-whitens row-major `x` (`n × k`, already projected) to unit per-dim
/// variance and re-projects rows onto the unit sphere, in place. Uses the
/// shared Householder+QL eigensolver on the small `k×k` covariance.
pub(crate) fn whiten_inplace(x: &mut [f32], n: usize, k: usize, sc: &mut TransportScratch) {
    debug_assert_eq!(x.len(), n * k);

    // Re-center defensively (the projected basis need not kill the mean).
    for c in 0..k {
        let mut mean = 0.0f32;
        for i in 0..n {
            mean += x[i * k + c];
        }
        mean /= n as f32;
        for i in 0..n {
            x[i * k + c] -= mean;
        }
    }

    // k×k covariance.
    let sym = &mut sc.sym;
    sym.clear();
    sym.resize(k * k, 0.0);
    for i in 0..n {
        let row = &x[i * k..(i + 1) * k];
        for a in 0..k {
            let ra = row[a] as f64;
            for b in a..k {
                sym[a * k + b] += ra * row[b] as f64;
            }
        }
    }
    let inv_n = 1.0 / n as f64;
    for a in 0..k {
        for b in a..k {
            let v = sym[a * k + b] * inv_n;
            sym[a * k + b] = v;
            sym[b * k + a] = v;
        }
    }

    // Symmetric eigendecomposition (columns of `eigvecs` are eigenvectors).
    let eigvals = &mut sc.eigvals;
    let eigvecs = &mut sc.eigvecs;
    eigvals.clear();
    eigvals.resize(k, 0.0);
    eigvecs.clear();
    eigvecs.resize(k * k, 0.0);
    for i in 0..k {
        eigvecs[i * k + i] = 1.0;
    }
    symmetric_eig(eigvals, eigvecs, sym, &mut sc.eig, k, EIG_MAX_ITERS);

    // Whitening matrix W = Q · diag(λ^-1/2) · Qᵀ (k×k, f32).
    let w = &mut sc.w;
    w.clear();
    w.resize(k * k, 0.0);
    for a in 0..k {
        for p in 0..k {
            let lam = eigvals[p].max(EIG_FLOOR);
            let inv_sqrt = 1.0 / lam.sqrt();
            for b in 0..k {
                w[a * k + b] += ((eigvecs[a * k + p] * eigvecs[b * k + p]) * inv_sqrt) as f32;
            }
        }
    }

    // Apply W row-wise, then renormalize rows to unit norm (project onto the
    // sphere so cosine geometry governs Sinkhorn + kNN).
    let mut src = vec![0.0f32; k];
    for i in 0..n {
        let row_start = i * k;
        src.copy_from_slice(&x[row_start..row_start + k]);
        for b in 0..k {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += src[t] * w[t * k + b];
            }
            x[row_start + b] = acc;
        }
        let mut norm = 0.0f32;
        for v in &x[row_start..row_start + k] {
            norm += v * v;
        }
        let norm = norm.sqrt().max(1e-12);
        for v in &mut x[row_start..row_start + k] {
            *v /= norm;
        }
    }
}

/// Entropic Sinkhorn on a cost matrix (`n1 × n2`, row-major), uniform
/// masses, writing the balanced transport plan into `plan`. Deterministic.
pub(crate) fn sinkhorn_plan(cost: &[f32], n1: usize, n2: usize, eps: f32, iters: usize, plan: &mut Vec<f32>) {
    debug_assert_eq!(cost.len(), n1 * n2);
    plan.clear();
    plan.resize(n1 * n2, 0.0);
    let inv_eps = 1.0 / eps.max(1e-6);
    for (p, c) in cost.iter().enumerate() {
        plan[p] = (-c * inv_eps).exp();
    }
    let mut u = vec![1.0f32; n1];
    let mut v = vec![1.0f32; n2];
    for _ in 0..iters {
        // u_i ∝ 1 / (K·v)_i
        for i in 0..n1 {
            let row = &plan[i * n2..(i + 1) * n2];
            let mut acc = 0.0f32;
            for (idx, kval) in row.iter().enumerate() {
                acc += v[idx] * kval;
            }
            u[i] = 1.0 / acc.max(1e-20);
        }
        // v_j ∝ 1 / (Kᵀ·u)_j
        for j in 0..n2 {
            let mut acc = 0.0f32;
            for i in 0..n1 {
                acc += plan[i * n2 + j] * u[i];
            }
            v[j] = 1.0 / acc.max(1e-20);
        }
    }
    // plan = diag(u)·K·diag(v), uniform source mass folded in.
    for i in 0..n1 {
        for j in 0..n2 {
            plan[i * n2 + j] *= u[i] * v[j] / (n1 as f32);
        }
    }
}

/// Greedy pseudo-pairing from a transport plan: `pairs[i]` = argmax target
/// for source row `i`. Ties break toward the lower index (deterministic).
pub(crate) fn greedy_pairs(plan: &[f32], n1: usize, n2: usize) -> Vec<usize> {
    let mut pairs = Vec::with_capacity(n1);
    for i in 0..n1 {
        let row = &plan[i * n2..(i + 1) * n2];
        let mut best_j = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (j, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_j = j;
            }
        }
        pairs.push(best_j);
    }
    pairs
}

/// Orthogonal-Procrustes polar factor: given pseudo-pairs (`a` rows ↔
/// `b` rows via `pairs`, both `n × k` unit-sphere), writes the row-vector
/// map `R` (`k × k`) minimizing ‖a·R − b‖_F among orthogonal `R`:
/// R = M·(MᵀM)^{-1/2} with M = AᵀB, via the shared symmetric eigensolver.
pub(crate) fn procrustes_polar_into(
    a: &[f32],
    b: &[f32],
    pairs: &[usize],
    n: usize,
    k: usize,
    sc: &mut TransportScratch,
    r_out: &mut Vec<f32>,
) {
    debug_assert_eq!(a.len(), n * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(pairs.len(), n);

    // M = Σ_i a_iᵀ · b_{pairs[i]}  (k×k, row-major).
    let mut m = vec![0.0f32; k * k];
    for (i, &j) in pairs.iter().enumerate() {
        let ar = &a[i * k..(i + 1) * k];
        let br = &b[j * k..(j + 1) * k];
        for x in 0..k {
            let ax = ar[x];
            for y in 0..k {
                m[x * k + y] += ax * br[y];
            }
        }
    }

    // S = MᵀM (k×k symmetric).
    let sym = &mut sc.sym;
    sym.clear();
    sym.resize(k * k, 0.0);
    for x in 0..k {
        for y in x..k {
            let mut acc = 0.0f64;
            for t in 0..k {
                acc += m[t * k + x] as f64 * m[t * k + y] as f64;
            }
            sym[x * k + y] = acc;
            sym[y * k + x] = acc;
        }
    }

    let eigvals = &mut sc.eigvals;
    let eigvecs = &mut sc.eigvecs;
    eigvals.clear();
    eigvals.resize(k, 0.0);
    eigvecs.clear();
    eigvecs.resize(k * k, 0.0);
    for i in 0..k {
        eigvecs[i * k + i] = 1.0;
    }
    symmetric_eig(eigvals, eigvecs, sym, &mut sc.eig, k, EIG_MAX_ITERS);

    // T = Q·diag(λ^{-1/2})·Qᵀ, then R = M·T (row-vector convention).
    let mut t = vec![0.0f32; k * k];
    for a in 0..k {
        for p in 0..k {
            let lam = eigvals[p].max(EIG_FLOOR);
            let inv_sqrt = 1.0 / lam.sqrt();
            for b in 0..k {
                t[a * k + b] += ((eigvecs[a * k + p] * eigvecs[b * k + p]) * inv_sqrt) as f32;
            }
        }
    }
    r_out.clear();
    r_out.resize(k * k, 0.0);
    for r in 0..k {
        for b in 0..k {
            let mut acc = 0.0f32;
            for t_i in 0..k {
                acc += m[r * k + t_i] * t[t_i * k + b];
            }
            r_out[r * k + b] = acc;
        }
    }
}

/// Validates transport-input invariants shared by the probe entry point.
/// Returns the effective common dimension `k`.
pub(crate) fn validate_dims(
    n1: usize,
    d1: usize,
    n2: usize,
    d2: usize,
    latent_dim: usize,
) -> Result<usize, TransportError> {
    if d1 == 0 || d2 == 0 {
        return Err(TransportError::DimMismatch);
    }
    if n1 < MIN_SAMPLES || n2 < MIN_SAMPLES {
        return Err(TransportError::TooFewSamples);
    }
    if d1 > MAX_DIM || d2 > MAX_DIM {
        return Err(TransportError::DimTooLarge);
    }
    if latent_dim > MAX_LATENT_DIM {
        return Err(TransportError::LatentDimTooLarge);
    }
    let k = latent_dim.min(d1).min(d2);
    if n1 < 4 * k || n2 < 4 * k {
        return Err(TransportError::LatentDimExceedsSamples);
    }
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An orthonormal d×d basis built from a fixed LCG + Gram-Schmidt —
    /// deterministic test fixture with no RNG dependency.
    fn rotation(d: usize, seed: u64) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        let mut m: Vec<f32> = (0..d * d).map(|_| lcg.next_f32_unit()).collect();
        orthonormalize_columns(&mut m, d, d);
        m
    }

    #[test]
    fn project_topk_recovers_dominant_subspace() {
        // 6 observed dims, only 2 carry variance; the 2-dim projection must
        // keep the high-variance plane.
        let n = 64;
        let d = 6;
        // Independent LCG streams per column — consecutive draws from ONE
        // stream are correlated in the high bits (LCG lattice artifact),
        // which tilts the sample covariance off the coordinate axes.
        let mut lcg_a = Lcg(7);
        let mut lcg_b = Lcg(8);
        let mut lcg_n = Lcg(9);
        let mut x = vec![0.0f32; n * d];
        for r in x.chunks_exact_mut(d) {
            let a = lcg_a.next_f32_unit() * 10.0;
            let b = lcg_b.next_f32_unit() * 8.0;
            r[0] = a;
            r[1] = b;
            r[2] = lcg_n.next_f32_unit() * 0.01;
            r[3] = lcg_n.next_f32_unit() * 0.01;
            r[4] = lcg_n.next_f32_unit() * 0.01;
            r[5] = lcg_n.next_f32_unit() * 0.01;
        }
        let mut out = Vec::new();
        let mut sc = TransportScratch::new();
        project_topk(&x, n, d, 2, 30, &mut out, &mut sc);
        assert_eq!(out.len(), n * 2);
        // Each projected column must align (sign-free) with its own data
        // column: col 0 ↔ the a-column, col 1 ↔ the b-column.
        for c in 0..2 {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..n {
                dot += out[i * 2 + c] * x[i * d + c];
                na += out[i * 2 + c] * out[i * 2 + c];
                nb += x[i * d + c] * x[i * d + c];
            }
            let cos = (dot / (na.sqrt() * nb.sqrt())).abs();
            // 0.98, not 1.0: at n=64 the covariance ESTIMATE of the
            // smaller-eigenvalue column carries ~1° sampling tilt even with
            // a converged eigensolver.
            assert!(cos > 0.98, "col {c} cos {cos}");
        }
    }

    #[test]
    fn sinkhorn_plan_rows_sum_to_uniform_mass() {
        let n1 = 8;
        let n2 = 8;
        let mut lcg = Lcg(11);
        let cost: Vec<f32> = (0..n1 * n2).map(|_| (lcg.next_f32_unit() + 1.0).abs()).collect();
        let mut plan = Vec::new();
        sinkhorn_plan(&cost, n1, n2, 0.1, 50, &mut plan);
        for i in 0..n1 {
            let s: f32 = plan[i * n2..(i + 1) * n2].iter().sum();
            assert!((s - 1.0 / n1 as f32).abs() < 1e-3, "row {i} sum {s}");
        }
    }

    #[test]
    fn procrustes_recovers_rotation() {
        // b = a·R_true (unit-sphere rows) → the polar factor must return
        // ≈R_true.
        let n = 32;
        let k = 4;
        let mut lcg = Lcg(13);
        let mut a: Vec<f32> = (0..n * k).map(|_| lcg.next_f32_unit()).collect();
        for row in a.chunks_exact_mut(k) {
            let s = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            for v in row.iter_mut() {
                *v /= s.max(1e-12);
            }
        }
        let r_true = rotation(k, 99);
        let mut b = vec![0.0f32; n * k];
        for i in 0..n {
            for c in 0..k {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a[i * k + t] * r_true[t * k + c];
                }
                b[i * k + c] = acc;
            }
        }
        let pairs: Vec<usize> = (0..n).collect();
        let mut sc = TransportScratch::new();
        let mut r = Vec::new();
        procrustes_polar_into(&a, &b, &pairs, n, k, &mut sc, &mut r);
        let diff: f32 = r.iter().zip(&r_true).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!(diff.sqrt() < 1e-2, "‖R−R_true‖ = {}", diff.sqrt());
    }

    #[test]
    fn validate_dims_enforces_floors() {
        assert_eq!(validate_dims(16, 8, 16, 8, 4), Ok(4));
        assert_eq!(validate_dims(4, 8, 16, 8, 4), Err(TransportError::TooFewSamples));
        assert_eq!(validate_dims(16, 0, 16, 8, 4), Err(TransportError::DimMismatch));
        assert_eq!(validate_dims(64, 2048, 64, 8, 4), Err(TransportError::DimTooLarge));
        assert_eq!(validate_dims(64, 8, 64, 8, 65), Err(TransportError::LatentDimTooLarge));
        assert_eq!(validate_dims(16, 8, 16, 8, 8), Err(TransportError::LatentDimExceedsSamples));
    }
}
