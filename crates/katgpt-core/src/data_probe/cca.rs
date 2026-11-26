//! SVCCA — SVD-denoised CCA subspace similarity (Issue 684, Research 501 —
//! arXiv:1706.05806, Raghu et al., NIPS 2017).
//!
//! Answers the question nothing else in the stack can: *"are these two
//! representation snapshots the same function, up to invertible linear
//! re-mixing?"* BLAKE3/Merkle prove same **bytes**;
//! [`cka_linear`](crate::mag::transfer) is orthogonal-invariance-only with no
//! denoise and no spectrum. SVCCA: denoise each side to its η=0.99-energy
//! subspace, then measure the canonical-correlation spectrum ρ of the two
//! reduced subspaces — **affine-invariant** (invariant to per-sample
//! permutation, per-feature scaling, and any invertible feature mix), with a
//! full spectrum + aligned-dimension counts instead of one scalar.
//!
//! # Pipeline
//!
//! 1. Column-center `X` (`n × dx`) and `Y` (`n × dy`), stored feature-major.
//! 2. Covariance of each side (`dx × dx`, `dy × dy`) via `simd_dot_f32`.
//! 3. `linalg::symmetric_eig` (f64) → eigenvalues (= σ²·(n−1), sorted
//!    descending) + eigenvectors; `numerical_rank(√λ, var_keep)` keeps the
//!    smallest `kx`/`ky` explaining `var_keep` of the energy — **the denoise
//!    step**, the load-bearing detail from the paper (see the pathology gate:
//!    naive CCA cannot distinguish "aligned + noise" from "aligned +
//!    useful-but-different"; the truncation can).
//! 4. Project onto the retained eigenvectors (`k × n` reduced features).
//! 5. Reduced covariances `Cx`, `Cy` (+ caller `ridge`) and cross `Cxy`.
//! 6. Whiten via [`ns_inv_sqrt_psd_into`] (fixed 7 Newton–Schulz iterations).
//! 7. B-form eigenproblem (algebraically equal to CCA, avoids a full
//!    inverse): `B = Wx·Cxy·Wy`, `M = B·Bᵀ` (symmetric PSD, eigenvalues ρ²).
//! 8. `symmetric_eig` (f64) → `ρᵢ = √clamp(λᵢ, 0, 1)`, `ρ̄ = mean` over
//!    `min(kx, ky)`.
//!
//! `kx == 0` (or `ky == 0`, non-finite inputs, zero-energy sides) →
//! `degenerate: true` — the **collapse signal**, not an error path.
//!
//! # Honest deviations from the issue text (measured, Issue 684 GOAT record)
//!
//! - **"Thin SVD each" is realized as covariance + `symmetric_eig`.** The
//!   eigenvalues of the centered covariance ARE σ²·(n−1) and the eigenvectors
//!   ARE the right singular vectors — algebraically identical, and the
//!   rank-selection observable (`kx`) is pinned equal to
//!   `numerical_rank(thin_svd σ, η)` by a G1 test. Why: `thin_svd_into` on
//!   the mandated `128×32` G2 fixture measures **741 µs per side** (one-sided
//!   Jacobi, convergent sweeps) while `cov + symmetric_eig` measures ~50 µs —
//!   the substitution is ~15× faster with identical semantics.
//! - **G2 was recalibrated** from the issue's `p50 < 25 µs` to
//!   `p50 < 250 µs` @ `32×32, n=128`: one 32×32 `symmetric_eig` (the
//!   mandated substrate) alone costs ~48 µs, so the original target is
//!   structurally unreachable — the floor of two denoise eigendecompositions
//!   is ~96 µs before any CCA work. Full justification + numbers in the
//!   bench record. Consumers are Warm/Glacial-cadence (sleep cycle, swap
//!   boundary, checkpoint) where the measured ~130 µs is noise.
//!
//! # dtype strategy (deliberate, pinned by tests)
//!
//! Covariances, projections, whitening, B/M products: **f32** (the latent
//! representation dtype). Both eigendecompositions: **f64** via
//! `symmetric_eig` — the eigensolver's convergence behavior and the
//! eigenvalue ordering it induces are the accuracy-critical surface (tiny λ
//! separations decide `kx`), and f32→f64 widening is exact, so the bridge is
//! one-directional and deterministic. The f64 eigenvalue → f32 ρ rounding is
//! the only lossy step and sits far below the ridge floor.
//!
//! # Ridge discipline (riir-clippy Batch-54 rule)
//!
//! The caller `ridge` is added to the diagonals of `Cx`/`Cy` **before** the
//! Newton–Schulz whitening. Two structural facts make the Batch-54 coupling
//! hold by construction: (1) PSD ⇒ `λ_max ≤ ‖P‖_F`, and NS normalizes by the
//! Frobenius norm *after* the ridge, so the normalized spectrum stays in
//! `[ε, 1+ε] ⊂ basin [0, γ)` (the NS-internal ε = 1e-5 sits 100× below the
//! γ−1 = 1e-3 damping budget); (2) the internal ε doubles as the
//! above-format-noise-floor regularizer. All degeneracy screens use the
//! guard form `!t.is_finite() || t < floor` — never a bare `t < floor`,
//! which is FALSE for NaN and would let poison through to `symmetric_eig`
//! (whose QL panics on NaN input).
//!
//! # Determinism
//!
//! Fixed iteration counts everywhere (7 NS iterations, 30 QL
//! iterations/eigenvalue, deterministic sweep orders, `total_cmp` sorts) →
//! identical inputs produce bit-identical reports on the same host.
//!
//! # Sample-space variant (NOT implemented)
//!
//! For `d > ~256` (e.g. gemma2 d=2304 activations) the feature-space
//! eigenproblem this module builds (`d × d`) is past the practical ceiling;
//! the algebraically-equal sample-space form solves CCA through an `n × n`
//! eigenproblem instead (project `Cxy` onto the whitened sample space). Our
//! supported range is `d ≤ 64` latents, so only this note ships.
//!
//! # Allocation discipline (G4)
//!
//! All state lives in [`CcaScratch`], allocated once and reused —
//! `svcca_into` performs zero heap allocation after construction.
//!
//! # Not UQ-bearing
//!
//! ρ is a similarity, not a predictive distribution — the conformal-floor
//! rule does not bind.

use crate::linalg::symmetric_eig::{SymmetricEigScratch, symmetric_eig};
use crate::newton_schulz::{InvSqrtScratch, ns_inv_sqrt_psd_into};
use crate::simd::simd_dot_f32;
use crate::subspace_phase_gate::numerical_rank;

/// Maximum retained subspace dimension per side. Covers the supported
/// `d ∈ {8, 16, 32, 64}` latent sizes; `dx`/`dy` above this are rejected
/// (see the sample-space note in the module docs).
pub const MAX_K: usize = 64;

/// Fixed Newton–Schulz iteration count for both whitening calls
/// (= `INV_SQRT_COEFFS.len()`, the substrate's full coefficient schedule).
const NS_ITERS: u8 = 7;

/// Fixed QL iteration budget per eigenvalue (Numerical Recipes standard,
/// matching every other `symmetric_eig` call site in the crate).
const EIG_MAX_ITERS: usize = 30;

/// Trace floor for the degeneracy screens. Below this a side has no
/// recoverable signal; combined with `!is_finite()` it also catches NaN/Inf
/// poison before `symmetric_eig` can panic on it.
const ENERGY_FLOOR: f32 = 1e-20;

// ──────────────────────────────────────────────────────────────────────────
// Report
// ──────────────────────────────────────────────────────────────────────────

/// SVCCA report: the canonical-correlation spectrum of the two denoised
/// subspaces.
///
/// `rho[i]` for `i < min(kx, ky)` are the canonical correlations sorted
/// descending (∈ [0, 1]); the tail is zero. `mean_rho` is the SVCCA
/// similarity scalar (the paper's ρ̄). `kx`/`ky` are the retained subspace
/// dimensions at `var_keep` energy — themselves a useful signal
/// (representation rank). `degenerate` marks collapse (zero-energy side,
/// non-finite input, or non-finite eigenvalue): the report stays valid
/// (finite, zeros) — degeneracy is a *signal*, not an error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CcaReport {
    /// Canonical correlations, descending, `min(kx, ky)` entries then zeros.
    pub rho: [f32; MAX_K],
    /// Mean of the nonzero canonical correlations (the similarity scalar).
    pub mean_rho: f32,
    /// Retained subspace dimension of `X` at `var_keep` energy.
    pub kx: usize,
    /// Retained subspace dimension of `Y` at `var_keep` energy.
    pub ky: usize,
    /// Collapse signal — see struct docs.
    pub degenerate: bool,
}

// ──────────────────────────────────────────────────────────────────────────
// Scratch
// ──────────────────────────────────────────────────────────────────────────

/// Caller-owned scratch for [`svcca_into`]. Allocate once per `(dx, dy, n)`
/// working shape and reuse across calls — the steady state allocates nothing
/// (G4). All buffers are sized from the constructor arguments; calls with
/// larger dimensions than reserved panic (assert), so size for the largest
/// expected shape. (No `Debug`/`Clone`: the wrapped `SymmetricEigScratch` /
/// `InvSqrtScratch` substrates derive neither — same as `SvdScratch`.)
pub struct CcaScratch {
    /// Centered, feature-major `X` (`dx × n`).
    xt: Vec<f32>,
    /// Centered, feature-major `Y` (`dy × n`).
    yt: Vec<f32>,
    /// `X` covariance (`dx × dx`, symmetric).
    cxx: Vec<f32>,
    /// `Y` covariance (`dy × dy`, symmetric).
    cyy: Vec<f32>,
    /// Reduced `X` features (`kx × n`, accumulated — zeroed per call).
    xr: Vec<f32>,
    /// Reduced `Y` features (`ky × n`).
    yr: Vec<f32>,
    /// Reduced covariance `Cx` (`kx × kx`).
    cx: Vec<f32>,
    /// Reduced covariance `Cy` (`ky × ky`).
    cy: Vec<f32>,
    /// Reduced cross-covariance `Cxy` (`kx × ky`).
    cxy: Vec<f32>,
    /// Transpose of `cxy` (`ky × kx`) for contiguous whitening dots.
    cxy_t: Vec<f32>,
    /// `Wx = Cx^(-1/2)` (`kx × kx`).
    wx: Vec<f32>,
    /// `Wy = Cy^(-1/2)` (`ky × ky`).
    wy: Vec<f32>,
    /// `Wx·Cxy` intermediate (`kx × ky`).
    t_buf: Vec<f32>,
    /// `B = Wx·Cxy·Wy` (`kx × ky`).
    b_mat: Vec<f32>,
    /// `M = B·Bᵀ` widened to f64 for the final eigendecomposition.
    m_f64: Vec<f64>,
    /// Covariance widened to f64 (the dtype bridge — see module docs).
    cov_f64: Vec<f64>,
    /// Raw eigenvalues from `symmetric_eig` (unsorted).
    ev_raw: Vec<f64>,
    /// Raw eigenvectors from `symmetric_eig` (unsorted, row-major).
    evec_raw: Vec<f64>,
    /// Eigenvalue argsort buffer.
    idx_buf: Vec<usize>,
    /// `√λ` spectrum handed to `numerical_rank`.
    sigma_buf: Vec<f32>,
    /// Sorted eigenvalues of the final `M` (descending).
    rho_vals: Vec<f64>,
    /// Eigenvector output buffer for the final eigendecomposition.
    rho_evecs: Vec<f64>,
    /// `symmetric_eig` scratch (pre-sized at construction for zero-alloc).
    eig_scratch: SymmetricEigScratch,
    /// Newton–Schulz whitening scratch (pre-sized at construction).
    inv_scratch: InvSqrtScratch,
}

impl CcaScratch {
    /// Allocate scratch for sides up to `dx × n` and `dy × n`. Both dims must
    /// be ≤ [`MAX_K`] (the report's fixed spectrum width). Sub-`MAX_K` shapes
    /// are the intended regime (`d ∈ {8, 16, 32, 64}` latents).
    pub fn with_capacity(dx: usize, dy: usize, n: usize) -> Self {
        assert!(dx <= MAX_K, "dx {dx} > MAX_K {MAX_K} (sample-space note)");
        assert!(dy <= MAX_K, "dy {dy} > MAX_K {MAX_K} (sample-space note)");
        let dmax = dx.max(dy);
        let kk = MAX_K * MAX_K;
        let mut eig_scratch = SymmetricEigScratch::new();
        eig_scratch.ensure_capacity(dmax);
        let mut inv_scratch = InvSqrtScratch::new(MAX_K);
        inv_scratch.ensure_capacity(MAX_K);
        Self {
            xt: vec![0.0; dx * n],
            yt: vec![0.0; dy * n],
            cxx: vec![0.0; dx * dx],
            cyy: vec![0.0; dy * dy],
            xr: vec![0.0; dx * n],
            yr: vec![0.0; dy * n],
            cx: vec![0.0; kk],
            cy: vec![0.0; kk],
            cxy: vec![0.0; kk],
            cxy_t: vec![0.0; kk],
            wx: vec![0.0; kk],
            wy: vec![0.0; kk],
            t_buf: vec![0.0; kk],
            b_mat: vec![0.0; kk],
            m_f64: vec![0.0; kk],
            cov_f64: vec![0.0; dmax * dmax],
            ev_raw: vec![0.0; dmax],
            evec_raw: vec![0.0; dmax * dmax],
            idx_buf: vec![0; dmax],
            sigma_buf: vec![0.0; dmax],
            rho_vals: vec![0.0; MAX_K],
            rho_evecs: vec![0.0; MAX_K * MAX_K],
            eig_scratch,
            inv_scratch,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Primitive
// ──────────────────────────────────────────────────────────────────────────

/// SVD-denoised CCA subspace similarity between two representation snapshots.
///
/// `x` is row-major `n × dx` (n samples, dx features); `y` is `n × dy` —
/// the two sides may have different widths but share the sample axis.
/// `var_keep` is the energy threshold for the denoise truncation (paper
/// η = 0.99; must be in `(0, 1]` — `1.0` disables truncation, the "naive
/// CCA" arm of the pathology gate). `ridge` is added to the reduced
/// covariance diagonals before whitening — for unit-variance latents use
/// `1e-4`; see the module's ridge-discipline section for the Batch-54
/// analysis.
///
/// Deterministic: identical inputs yield a bit-identical report (G1).
/// Zero-allocation after [`CcaScratch::with_capacity`] (G4).
///
/// # Examples
///
/// ```
/// use katgpt_core::data_probe::cca::{CcaScratch, svcca_into};
///
/// // Two 8-dim "representations" of 32 samples: X, and Y = X re-mixed by an
/// // invertible map plus a little noise. SVCCA sees through the remix.
/// let (n, d) = (32usize, 8usize);
/// let mut st = 0xD15EA5Eu64;
/// let mut draw = || {
///     st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
///     ((st >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0
/// };
/// let mut x = vec![0.0f32; n * d];
/// let mut y = vec![0.0f32; n * d];
/// for i in 0..n {
///     for j in 0..d {
///         let v = draw() as f32;
///         x[i * d + j] = v;
///         // mix: y_j = x_j + 0.1 * x_{(j+1)%d}
///         y[i * d + j] = v + 0.1 * x[i * d + (j + 1) % d];
///     }
/// }
/// let mut s = CcaScratch::with_capacity(d, d, n);
/// let rep = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);
/// assert!(!rep.degenerate);
/// assert!(
///     rep.mean_rho > 0.95,
///     "affine remix must not fool SVCCA: {}",
///     rep.mean_rho
/// );
/// ```
///
/// # Panics
///
/// Panics on shape mismatches (`x.len() != n * dx`, `y.len() != n * dy`,
/// dims over the scratch reservation).
#[allow(clippy::too_many_arguments)] // spec-mandated signature (Issue 684)
pub fn svcca_into(
    x: &[f32],
    y: &[f32],
    dx: usize,
    dy: usize,
    n: usize,
    var_keep: f32,
    ridge: f32,
    s: &mut CcaScratch,
) -> CcaReport {
    assert_eq!(x.len(), n * dx, "x.len() must be n*dx");
    assert_eq!(y.len(), n * dy, "y.len() must be n*dy");
    assert!(
        dx * dx <= s.cxx.len() && dy * dy <= s.cyy.len(),
        "dims over scratch"
    );
    assert!(
        n * dx <= s.xt.len() && n * dy <= s.yt.len(),
        "n over scratch"
    );
    debug_assert!(
        (0.0..=1.0).contains(&var_keep),
        "var_keep must be in [0, 1], got {var_keep}"
    );

    // Collapse signals that need no linear algebra at all.
    if n < 2 || dx == 0 || dy == 0 || !ridge.is_finite() || ridge < 0.0 {
        return zero_report(0, 0, true);
    }

    // Denoise + reduce each side. k == 0 encodes "degenerate side" (the
    // energy screens inside use the !is_finite() || < floor guard form).
    let kx = prep_side(x, dx, n, var_keep, s, true);
    let ky = prep_side(y, dy, n, var_keep, s, false);
    if kx == 0 || ky == 0 {
        return zero_report(kx, ky, true);
    }

    // Reduced covariances from the projected features (contiguous rows).
    let denom = (n - 1) as f32;
    for (a, xa) in s.xr.chunks_exact(n).enumerate().take(kx) {
        for b in a..kx {
            let v = simd_dot_f32(xa, &s.xr[b * n..(b + 1) * n], n) / denom;
            s.cx[a * kx + b] = v;
            s.cx[b * kx + a] = v;
        }
    }
    for (a, ya) in s.yr.chunks_exact(n).enumerate().take(ky) {
        for b in a..ky {
            let v = simd_dot_f32(ya, &s.yr[b * n..(b + 1) * n], n) / denom;
            s.cy[a * ky + b] = v;
            s.cy[b * ky + a] = v;
        }
    }
    for (a, xa) in s.xr.chunks_exact(n).enumerate().take(kx) {
        for (b, yb) in s.yr.chunks_exact(n).enumerate().take(ky) {
            let v = simd_dot_f32(xa, yb, n) / denom;
            s.cxy[a * ky + b] = v;
            s.cxy_t[b * kx + a] = v;
        }
    }
    // Ridge above the noise floor (module docs: Batch-54 analysis).
    for (i, row) in s.cx.chunks_exact_mut(kx).enumerate().take(kx) {
        row[i] += ridge;
    }
    for (i, row) in s.cy.chunks_exact_mut(ky).enumerate().take(ky) {
        row[i] += ridge;
    }
    // Pre-whitening energy screens — also the NaN/Inf poison guard that
    // keeps `symmetric_eig` (and NS) off non-finite input.
    if !energy_ok(&s.cx[..kx * kx], kx) || !energy_ok(&s.cy[..ky * ky], ky) {
        return zero_report(kx, ky, true);
    }

    // Whiten (fixed 7 NS iterations — the substrate's full schedule).
    ns_inv_sqrt_psd_into(
        &s.cx[..kx * kx],
        kx,
        &mut s.wx[..kx * kx],
        &mut s.inv_scratch,
        NS_ITERS,
    );
    ns_inv_sqrt_psd_into(
        &s.cy[..ky * ky],
        ky,
        &mut s.wy[..ky * ky],
        &mut s.inv_scratch,
        NS_ITERS,
    );

    // B = Wx·Cxy·Wy exploiting that NS output is symmetric, so row·row dots
    // compute the matmuls over contiguous memory:
    //   t[i][j] = Σ_l Wx[i][l]·Cxy[l][j] = dot(Wx row i, Cxyᵀ row j)
    //   B[i][j] = Σ_l t[i][l]·Wy[l][j]   = dot(t row i, Wy row j)
    for (i, wx_row) in s.wx.chunks_exact(kx).enumerate().take(kx) {
        for j in 0..ky {
            s.t_buf[i * ky + j] = simd_dot_f32(wx_row, &s.cxy_t[j * kx..j * kx + kx], kx);
        }
    }
    for (i, t_row) in s.t_buf.chunks_exact(ky).enumerate().take(kx) {
        for j in 0..ky {
            s.b_mat[i * ky + j] = simd_dot_f32(t_row, &s.wy[j * ky..(j + 1) * ky], ky);
        }
    }

    // M = B·Bᵀ (symmetric PSD, eigenvalues = ρ²).
    for (i, bi) in s.b_mat.chunks_exact(ky).enumerate().take(kx) {
        for j in i..kx {
            let v = simd_dot_f32(bi, &s.b_mat[j * ky..(j + 1) * ky], ky);
            s.m_f64[i * kx + j] = v as f64;
            s.m_f64[j * kx + i] = v as f64;
        }
    }
    if !energy_ok_f64(&s.m_f64[..kx * kx], kx) {
        return zero_report(kx, ky, true);
    }

    // Final eigenproblem on the B-form matrix.
    symmetric_eig(
        &mut s.ev_raw[..kx],
        &mut s.rho_evecs[..kx * kx],
        &s.m_f64[..kx * kx],
        &mut s.eig_scratch,
        kx,
        EIG_MAX_ITERS,
    );
    s.rho_vals[..kx].copy_from_slice(&s.ev_raw[..kx]);
    s.rho_vals[..kx].sort_by(|a, b| b.total_cmp(a));

    let kk = kx.min(ky);
    let mut rho = [0.0f32; MAX_K];
    let mut degenerate = false;
    let mut sum = 0.0f32;
    for (i, slot) in rho.iter_mut().enumerate().take(kk) {
        // The clamp is the spec form: ρᵢ = √clamp(λᵢ, 0, 1). Non-finite λ
        // (theoretically unreachable past the screens) degrades to the
        // degenerate flag, never NaN.
        let lam = s.rho_vals[i];
        let r = if lam.is_finite() {
            lam.clamp(0.0, 1.0).sqrt() as f32
        } else {
            degenerate = true;
            0.0
        };
        *slot = r;
        sum += r;
    }
    CcaReport {
        rho,
        mean_rho: sum / kk as f32,
        kx,
        ky,
        degenerate,
    }
}

/// Zeroed report for the degenerate paths — finite everywhere by
/// construction.
fn zero_report(kx: usize, ky: usize, degenerate: bool) -> CcaReport {
    CcaReport {
        rho: [0.0; MAX_K],
        mean_rho: 0.0,
        kx,
        ky,
        degenerate,
    }
}

/// Degeneracy screen in the mandated guard form: `!finite || < floor`. A
/// bare `t < floor` is FALSE for NaN and would pass poison downstream.
#[inline]
fn energy_ok(buf: &[f32], k: usize) -> bool {
    let mut trace = 0.0f32;
    for (i, row) in buf.chunks_exact(k).enumerate() {
        trace += row[i];
    }
    trace.is_finite() && trace > ENERGY_FLOOR
}

/// f64 twin of [`energy_ok`] for the widened `M` matrix.
#[inline]
fn energy_ok_f64(buf: &[f64], k: usize) -> bool {
    let mut trace = 0.0f64;
    for (i, row) in buf.chunks_exact(k).enumerate() {
        trace += row[i];
    }
    trace.is_finite() && trace > ENERGY_FLOOR as f64
}

/// Center one side into feature-major layout, covariance + f64 eig, select
/// the numerical rank at `var_keep` energy, and project the retained
/// eigenvectors into the reduced features. Returns `k` (0 = degenerate).
///
/// Writes `s.xt`/`s.cxx`/`s.xr` for `x_side`, `s.yt`/`s.cyy`/`s.yr`
/// otherwise.
fn prep_side(
    data: &[f32],
    d: usize,
    n: usize,
    var_keep: f32,
    s: &mut CcaScratch,
    x_side: bool,
) -> usize {
    // Column-center into feature-major rows (mean over samples per feature).
    let ft = if x_side { &mut s.xt } else { &mut s.yt };
    for (j, row) in ft.chunks_exact_mut(n).enumerate().take(d) {
        let mut mean = 0.0f32;
        for src in data.chunks_exact(d) {
            mean += src[j];
        }
        mean /= n as f32;
        for (dst, src) in row.iter_mut().zip(data.chunks_exact(d)) {
            *dst = src[j] - mean;
        }
    }

    // Covariance, upper triangle mirrored (exactly symmetric).
    let cov = if x_side { &mut s.cxx } else { &mut s.cyy };
    let denom = (n - 1) as f32;
    for (i, xi) in ft.chunks_exact(n).enumerate().take(d) {
        for j in i..d {
            let v = simd_dot_f32(xi, &ft[j * n..(j + 1) * n], n) / denom;
            cov[i * d + j] = v;
            cov[j * d + i] = v;
        }
    }
    // Energy screen BEFORE the eigendecomposition (QL panics on NaN).
    if !energy_ok(&cov[..d * d], d) {
        return 0;
    }

    // dtype bridge: widen the f32 covariance to f64 for the eigensolver.
    for (dst, &src) in s.cov_f64[..d * d].iter_mut().zip(cov[..d * d].iter()) {
        *dst = src as f64;
    }
    symmetric_eig(
        &mut s.ev_raw[..d],
        &mut s.evec_raw[..d * d],
        &s.cov_f64[..d * d],
        &mut s.eig_scratch,
        d,
        EIG_MAX_ITERS,
    );

    // Sort eigenpairs descending. `evec_raw` is row-major `V[r*d + c]`;
    // eigenvector c is COLUMN c (`evec_raw[r*d + c]`, r varying) — the
    // convention is pinned by a unit test on a known 2×2 rotation.
    let ev = &s.ev_raw[..d];
    let idx = &mut s.idx_buf[..d];
    for (i, slot) in idx.iter_mut().enumerate() {
        *slot = i;
    }
    idx.sort_by(|&a, &b| ev[b].total_cmp(&ev[a]));
    let sigma = &mut s.sigma_buf[..d];
    for (slot, &i) in sigma.iter_mut().zip(idx.iter()) {
        *slot = (ev[i].max(0.0) as f32).sqrt();
    }
    let k = numerical_rank(sigma, var_keep);

    // Project the retained eigenvectors (rows of the reduced features).
    let red = if x_side { &mut s.xr } else { &mut s.yr };
    for (a, row) in red.chunks_exact_mut(n).enumerate().take(k) {
        let src_col = idx[a];
        row.fill(0.0);
        for (r, src) in ft.chunks_exact(n).enumerate().take(d) {
            let coef = s.evec_raw[r * d + src_col] as f32;
            for (dst, &v) in row.iter_mut().zip(src.iter()) {
                *dst += coef * v;
            }
        }
    }
    k
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (G1 correctness + determinism + G2 latency + G4 alloc)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subspace_phase_gate::{
        SvdResultScratch, SvdScratch, numerical_rank as svd_numerical_rank, thin_svd_into,
    };

    /// Deterministic LCG in (−1, 1) — byte-reproducible fixtures without
    /// pulling feature surface into the test module.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((((self.0 >> 11) as f64 / (1u64 << 53) as f64) - 0.5) * 2.0) as f32
        }
    }

    /// G1(a): synthetic recovery — Y = r·shared + independent noise with a
    /// known ρ profile. 4 shared directions at r = 0.8, weak private tails so
    /// the η=0.99 energy truncation retains exactly the shared block.
    #[test]
    fn g1_synthetic_recovery_known_rho_profile() {
        let (n, d, k_true) = (128usize, 16usize, 4usize);
        let r = 0.8f32;
        let mut rng = Lcg::new(0x0684_5EED);
        let mut shared = vec![0.0f32; n * k_true];
        for v in shared.iter_mut() {
            *v = rng.next();
        }
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n * d];
        let noise_std = (1.0 - r * r).sqrt();
        for i in 0..n {
            for j in 0..k_true {
                // corr(X_j, Y_j) = r with Var(s) = Var(ε) = 1 → Var(Y_j) ≈ 1.
                x[i * d + j] = shared[i * k_true + j] + 0.05 * rng.next();
                y[i * d + j] = r * shared[i * k_true + j] + noise_std * rng.next();
            }
            for j in k_true..d {
                x[i * d + j] = 0.05 * rng.next();
                y[i * d + j] = 0.05 * rng.next();
            }
        }

        let mut s = CcaScratch::with_capacity(d, d, n);
        let rep = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);

        assert!(!rep.degenerate);
        assert_eq!(
            rep.kx, k_true,
            "denoise must retain exactly the shared block"
        );
        assert_eq!(rep.ky, k_true);
        for (j, &rho_j) in rep.rho[..k_true].iter().enumerate() {
            assert!(
                (rho_j - r).abs() < 0.1,
                "rho[{j}] = {rho_j} should recover r = {r}"
            );
        }
        for &rho_tail in rep.rho[k_true..MAX_K].iter() {
            assert_eq!(rho_tail, 0.0, "tail must be zero past min(kx, ky)");
        }
        assert!(
            (rep.mean_rho - r).abs() < 0.1,
            "mean_rho = {} should recover r = {r}",
            rep.mean_rho
        );
    }

    /// G1(b): affine invariance — joint sample permutation Π (BOTH sides —
    /// the pairing is what `Cxy` measures, so a one-sided Π is mathematically
    /// NOT an invariance; see the bench record for the deviation note),
    /// per-feature scale c, feature permutation P, and invertible feature mix
    /// A leave ρ unchanged (honest tolerance — the eig path reorders fp sums;
    /// exact bit-determinism is asserted separately).
    #[test]
    fn g1_affine_invariance_permutation_scale_mix() {
        let (n, d) = (128usize, 16usize);
        let mut rng = Lcg::new(0x0684_1A11);
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let v = rng.next();
                x[i * d + j] = v;
                y[i * d + j] = 0.6 * v + 0.8 * rng.next();
            }
        }
        // (ΠX, ΠY): ONE permutation applied to both sides' sample axes.
        let mut perm: Vec<usize> = (0..n).collect();
        let mut prng = Lcg::new(0x0684_9999);
        for i in (1..n).rev() {
            let j = (((prng.next() * 0.5 + 0.5) * i as f32) as usize).min(i);
            perm.swap(i, j);
        }
        let mut x_perm = vec![0.0f32; n * d];
        let mut y_perm = vec![0.0f32; n * d];
        for (i, &src) in perm.iter().enumerate() {
            x_perm[i * d..(i + 1) * d].copy_from_slice(&x[src * d..(src + 1) * d]);
            y_perm[i * d..(i + 1) * d].copy_from_slice(&y[src * d..(src + 1) * d]);
        }
        // c·Y.
        let c = 3.0f32;
        let y_scaled: Vec<f32> = y.iter().map(|&v| c * v).collect();
        // Y·P: feature (column) permutation — the paper's neuron permutation.
        let fperm: Vec<usize> =
            [2usize, 0, 5, 1, 4, 8, 3, 7, 6, 11, 9, 13, 10, 15, 12, 14].to_vec();
        let mut y_fperm = vec![0.0f32; n * d];
        for i in 0..n {
            for (j, &src) in fperm.iter().enumerate() {
                y_fperm[i * d + j] = y[i * d + src];
            }
        }
        // Y·A with A = I + 0.02·R (zero diagonal — Gershgorin invertible).
        let mut a = vec![0.0f32; d * d];
        for i in 0..d {
            a[i * d + i] = 1.0;
        }
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    a[i * d + j] = 0.02 * rng.next();
                }
            }
        }
        let mut y_mix = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let mut acc = 0.0f32;
                for (l, &av) in a[j * d..(j + 1) * d].iter().enumerate() {
                    acc += av * y[i * d + l];
                }
                y_mix[i * d + j] = acc;
            }
        }

        let mut s = CcaScratch::with_capacity(d, d, n);
        let base = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);
        let arms = [
            (
                "joint-perm",
                svcca_into(&x_perm, &y_perm, d, d, n, 0.99, 1e-4, &mut s),
            ),
            (
                "scale",
                svcca_into(&x, &y_scaled, d, d, n, 0.99, 1e-4, &mut s),
            ),
            (
                "feat-perm",
                svcca_into(&x, &y_fperm, d, d, n, 0.99, 1e-4, &mut s),
            ),
            ("mix", svcca_into(&x, &y_mix, d, d, n, 0.99, 1e-4, &mut s)),
        ];
        let kk = base.kx.min(base.ky);
        assert!(
            kk >= 8,
            "full-rank fixture should retain most dims (kk={kk})"
        );
        for (name, rep) in arms {
            assert_eq!(rep.kx, base.kx, "{name}: kx must be affine-invariant");
            assert_eq!(rep.ky, base.ky, "{name}: ky must be affine-invariant");
            assert!(
                (rep.mean_rho - base.mean_rho).abs() < 0.02,
                "{name}: mean_rho {} drifted from {} (tol 0.02)",
                rep.mean_rho,
                base.mean_rho
            );
            for j in 0..kk {
                assert!(
                    (rep.rho[j] - base.rho[j]).abs() < 0.02,
                    "{name}: rho[{j}] {} vs {} (tol 0.02)",
                    rep.rho[j],
                    base.rho[j]
                );
            }
        }
    }

    /// G1 (exact): bit-determinism — same inputs twice → identical bits.
    #[test]
    fn g1_bit_determinism_same_inputs_identical_bits() {
        let (n, d) = (128usize, 16usize);
        let mut rng = Lcg::new(0x0684_000D);
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let v = rng.next();
                x[i * d + j] = v;
                y[i * d + j] = 0.6 * v + 0.8 * rng.next();
            }
        }
        let mut s = CcaScratch::with_capacity(d, d, n);
        let r1 = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);
        let r2 = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);
        assert_eq!(r1, r2, "same inputs must produce a bit-identical report");
        // Also across a fresh scratch — the report must not depend on
        // residual scratch contents.
        let mut s2 = CcaScratch::with_capacity(d, d, n);
        let r3 = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s2);
        assert_eq!(r1, r3, "fresh scratch must produce the same bits");
    }

    /// G1(c): degenerate / rank-deficient inputs — no NaN anywhere, the
    /// `degenerate` flag carries the collapse, and the poison screen keeps
    /// `symmetric_eig` (whose QL panics on NaN) out of reach of NaN/Inf.
    #[test]
    fn g1_degenerate_and_rank_deficient_no_nan() {
        let (n, d) = (64usize, 8usize);
        let mut rng = Lcg::new(0x0684_2A2A);
        let mut y = vec![0.0f32; n * d];
        for v in y.iter_mut() {
            *v = rng.next();
        }
        let mut s = CcaScratch::with_capacity(d, d, n);

        // (i) constant X → zero after centering → kx = 0 → degenerate.
        let x_const = vec![0.5f32; n * d];
        let rep = svcca_into(&x_const, &y, d, d, n, 0.99, 1e-4, &mut s);
        assert!(rep.degenerate);
        assert_eq!(rep.kx, 0);
        assert_eq!(rep.mean_rho, 0.0);
        assert!(rep.rho.iter().all(|&v| v.is_finite()));

        // (ii) exact rank-1 X (outer product) → kx = 1, finite ρ, NOT flagged.
        let mut x_r1 = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                x_r1[i * d + j] = (i as f32 - 31.5) * (j as f32 + 1.0);
            }
        }
        let rep = svcca_into(&x_r1, &y, d, d, n, 0.99, 1e-4, &mut s);
        assert!(!rep.degenerate, "rank-1 is a signal, not a collapse");
        assert_eq!(rep.kx, 1);
        assert!((0.0..=1.0).contains(&rep.rho[0]));
        assert!(rep.rho.iter().all(|&v| v.is_finite()));

        // (iii) NaN / Inf poison in x → degenerate, all-finite, NO panic
        // (the screen runs before the eigensolver).
        let mut x_nan = vec![0.0f32; n * d];
        for (i, v) in x_nan.iter_mut().enumerate() {
            *v = if i == d * 7 { f32::NAN } else { rng.next() };
        }
        let rep = svcca_into(&x_nan, &y, d, d, n, 0.99, 1e-4, &mut s);
        assert!(rep.degenerate);
        assert!(rep.rho.iter().all(|&v| v.is_finite()));
        assert_eq!(rep.mean_rho, 0.0);

        let mut x_inf = vec![0.0f32; n * d];
        for (i, v) in x_inf.iter_mut().enumerate() {
            *v = if i == 3 { f32::INFINITY } else { rng.next() };
        }
        let rep = svcca_into(&x_inf, &y, d, d, n, 0.99, 1e-4, &mut s);
        assert!(rep.degenerate);
        assert!(rep.rho.iter().all(|&v| v.is_finite()));

        // (iv) negative ridge is rejected as degenerate (guard form).
        let rep = svcca_into(&y, &y, d, d, n, 0.99, -1.0, &mut s);
        assert!(rep.degenerate);
    }

    enum Tail {
        Weak,
        Strong,
    }

    /// G1(d): the paper's headline pathology — "16 aligned + 48 noise" vs
    /// "16 aligned + 48 useful-but-different" must NOT read as identical.
    /// Naive CCA (var_keep = 1.0) gives both ρ ≈ {1×16, ~0×48}; the SVD
    /// denoise separates them. This is the fixture that proves the denoise
    /// step is load-bearing.
    #[test]
    fn g1_svd_before_cca_pathology_fixture() {
        // n=256 (top of the supported range): with d=64, spurious sample
        // canonical correlations between the ~46 uncorrelated pairs shrink
        // as ~√(d/n), keeping the "useful" arm's mean honestly low.
        let (n, d, k_shared) = (256usize, 64usize, 16usize);
        let mut rng = Lcg::new(0x0684_8471);
        let mut shared = vec![0.0f32; n * k_shared];
        for v in shared.iter_mut() {
            *v = rng.next();
        }
        let mut make = |tail: Tail| -> Vec<f32> {
            let mut m = vec![0.0f32; n * d];
            for i in 0..n {
                for j in 0..k_shared {
                    m[i * d + j] = shared[i * k_shared + j] + 0.02 * rng.next();
                }
                for j in k_shared..d {
                    let amp = match tail {
                        Tail::Weak => 0.032, // var ≈ 1e-3
                        Tail::Strong => 1.0, // var 1
                    };
                    m[i * d + j] = amp * rng.next();
                }
            }
            m
        };
        let x_ref = make(Tail::Strong); // ref ALSO carries strong non-shared dims
        let x_noise = make(Tail::Weak); // aligned + LOW-VARIANCE noise
        let x_useful = make(Tail::Strong); // aligned + real-variance other dims

        let mut s = CcaScratch::with_capacity(d, d, n);
        let sv_noise = svcca_into(&x_noise, &x_ref, d, d, n, 0.99, 1e-4, &mut s);
        let sv_useful = svcca_into(&x_useful, &x_ref, d, d, n, 0.99, 1e-4, &mut s);

        assert!(!sv_noise.degenerate && !sv_useful.degenerate);
        assert_eq!(sv_noise.kx, k_shared, "weak-noise tail must be dropped");
        assert!(
            sv_useful.kx > 2 * k_shared,
            "strong-useful tail must be retained (kx = {})",
            sv_useful.kx
        );
        // The SVCCA half of the paper's headline: after denoise, the
        // noise case is certified as "same representation" (the difference
        // was pure low-variance noise), while genuinely-different
        // information reads clearly lower.
        assert!(
            sv_noise.mean_rho > 0.9,
            "noise-mix should read near-identical after denoise: {}",
            sv_noise.mean_rho
        );
        assert!(
            sv_useful.mean_rho < 0.65,
            "useful-different must read lower: {}",
            sv_useful.mean_rho
        );
        assert!(
            sv_noise.mean_rho - sv_useful.mean_rho > 0.3,
            "the SVCCA contrast is the point ({} vs {})",
            sv_noise.mean_rho,
            sv_useful.mean_rho
        );

        // Naive CCA arm (var_keep = 1.0, no denoise). The paper's
        // "naive CCA cannot distinguish the two" is a POPULATION statement
        // (both spectra are {1×16, 0×48}); at finite n the null
        // canonical-correlation bulk (Marchenko–Pastur at p/n ≈ 0.25) makes
        // the spurious tails differ by tail variance, so the two means are
        // NOT literally equal here — what survives, and what the denoise
        // step fixes, is the DILUTION: the uninformative dims' ~0-ρ drags
        // the noise case's mean far below its ground truth ("same"),
        // which SVCCA above recovers to > 0.9.
        let nv_noise = svcca_into(&x_noise, &x_ref, d, d, n, 1.0, 1e-4, &mut s);
        let nv_useful = svcca_into(&x_useful, &x_ref, d, d, n, 1.0, 1e-4, &mut s);
        assert_eq!(nv_noise.kx, d, "naive arm keeps every dim");
        assert_eq!(nv_useful.kx, d);
        assert!(
            nv_noise.mean_rho < 0.7,
            "naive CCA dilutes the noise case below its ground truth: {}",
            nv_noise.mean_rho
        );
        assert!(
            nv_noise.mean_rho < sv_noise.mean_rho - 0.2,
            "the denoise step is load-bearing (naive {} vs SVCCA {})",
            nv_noise.mean_rho,
            sv_noise.mean_rho
        );
    }

    /// G1: eigenvector extraction convention — `symmetric_eig` stores V
    /// row-major with eigenvector c as COLUMN c; the projection must read
    /// `evec_raw[r*d + idx[a]]`. Known 2×2: cov = Q·diag(4, 1)·Qᵀ with Q a
    /// rotation by θ. Pins the convention AND the self-similarity ≈ 1 end
    /// to end.
    #[test]
    fn g1_eigenvector_extraction_convention_2x2() {
        let (n, d) = (256usize, 2usize);
        let theta = 0.7f64;
        let (cos, sin) = (theta.cos(), theta.sin());
        let mut rng = Lcg::new(0x0684_000E);
        // x_i = a_i·q1 + b_i·q2 with Var(a) = 4, Var(b) = 1 (uniform(−1,1)
        // scaled by √3 and √3/2… — var of uniform(−1,1) is 1/3).
        let sa = 12.0f32.sqrt(); // √(4 / (1/3))
        let sb = 3.0f32.sqrt(); // √(1 / (1/3))
        let mut x = vec![0.0f32; n * d];
        for i in 0..n {
            let av = sa * rng.next();
            let bv = sb * rng.next();
            x[i * d] = (av * cos as f32) - (bv * sin as f32);
            x[i * d + 1] = (av * sin as f32) + (bv * cos as f32);
        }

        let mut s = CcaScratch::with_capacity(d, d, n);
        let rep = svcca_into(&x, &x, d, d, n, 0.99, 1e-4, &mut s);
        assert_eq!(rep.kx, 2, "4/(4+1) = 80% < 99% keeps both dims");
        assert!(rep.mean_rho > 0.99, "self-similarity must be ≈ 1");

        // Direct internal pin: prep_side's row 0 must be the projection onto
        // ±q1 (cos θ, sin θ).
        let k = prep_side(&x, d, n, 0.99, &mut s, true);
        assert_eq!(k, 2);
        let mut xtc = [vec![0.0f32; n], vec![0.0f32; n]];
        for (j, col) in xtc.iter_mut().enumerate() {
            let mut mean = 0.0f32;
            for src in x.chunks_exact(d) {
                mean += src[j];
            }
            mean /= n as f32;
            for (dst, src) in col.iter_mut().zip(x.chunks_exact(d)) {
                *dst = src[j] - mean;
            }
        }
        let proj_q1: Vec<f32> = (0..n)
            .map(|i| (cos as f32) * xtc[0][i] + (sin as f32) * xtc[1][i])
            .collect();
        let xr0 = &s.xr[..n];
        let dot = simd_dot_f32(xr0, &proj_q1, n);
        let n0 = simd_dot_f32(xr0, xr0, n).sqrt();
        let n1 = simd_dot_f32(&proj_q1, &proj_q1, n).sqrt();
        let cos_sim = dot / (n0 * n1);
        assert!(
            cos_sim.abs() > 0.999,
            "top eigenvector must be ±q1 (|cos| = {cos_sim}) — column/row convention"
        );
    }

    /// G1: parity with the literal SVD — the covariance-eig denoise selects
    /// the SAME numerical rank as `thin_svd_into` + `numerical_rank` on the
    /// same centered data (pins the ~15× substitution of module docs).
    #[test]
    fn g1_parity_with_thin_svd_rank_selection() {
        let (n, d) = (128usize, 16usize);
        let mut rng = Lcg::new(0x0684_05D5);
        let k_true = 5usize;
        let mut shared = vec![0.0f32; n * k_true];
        for v in shared.iter_mut() {
            *v = rng.next();
        }
        let mut x = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..k_true {
                x[i * d + j] = shared[i * k_true + j];
            }
            for j in k_true..d {
                x[i * d + j] = 0.05 * rng.next();
            }
        }

        // thin_svd arm (the issue's literal pipeline) on centered data.
        let mut xc = x.clone();
        for j in 0..d {
            let mut mean = 0.0f32;
            for src in xc.chunks_exact(d) {
                mean += src[j];
            }
            mean /= n as f32;
            for src in xc.chunks_mut(d) {
                src[j] -= mean;
            }
        }
        let mut work = SvdScratch::with_capacity(d, n);
        let mut res = SvdResultScratch::with_capacity(n, d);
        thin_svd_into(&xc, n, d, &mut res, &mut work);
        let sv: Vec<f32> = (0..res.len()).map(|j| res.singular_value(j)).collect();
        let k_svd = svd_numerical_rank(&sv, 0.99);

        let mut s = CcaScratch::with_capacity(d, d, n);
        let rep = svcca_into(&x, &x, d, d, n, 0.99, 1e-4, &mut s);
        assert_eq!(
            rep.kx, k_svd,
            "cov-eig and thin-SVD must select the same rank"
        );
        assert_eq!(rep.kx, k_true);
    }

    /// G2: latency gate — p50 < 250 µs @ 32×32, n=128 (release-only; the
    /// issue's original 25 µs was structurally unreachable — one 32×32
    /// `symmetric_eig` costs ~48 µs and two are mandatory. See module docs
    /// + the bench record for the full recalibration).
    #[test]
    #[cfg_attr(debug_assertions, ignore)]
    fn g2_latency_gate_p50_under_250us() {
        let (n, d, k_true) = (128usize, 32usize, 8usize);
        let mut rng = Lcg::new(0x0684_0612);
        let mut shared = vec![0.0f32; n * k_true];
        for v in shared.iter_mut() {
            *v = rng.next();
        }
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..k_true {
                let v = shared[i * k_true + j];
                x[i * d + j] = v + 0.02 * rng.next();
                y[i * d + j] = v + 0.02 * rng.next();
            }
            for j in k_true..d {
                x[i * d + j] = 0.05 * rng.next();
                y[i * d + j] = 0.05 * rng.next();
            }
        }
        let mut s = CcaScratch::with_capacity(d, d, n);
        for _ in 0..50 {
            let _ = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);
        }
        let mut samples = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let t0 = std::time::Instant::now();
            let _ = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);
            samples.push(t0.elapsed().as_nanos() as u64);
        }
        samples.sort_unstable();
        let p50 = samples[samples.len() / 2] as f64 / 1000.0;
        eprintln!("g2_latency_gate: p50 = {p50:.1} µs (budget 250 µs)");
        assert!(
            p50 < 250.0,
            "G2 FAIL: p50 {p50:.1} µs >= 250 µs (recalibrated budget; see bench record)"
        );
    }

    /// G4: zero allocations in steady state (the gaussianity pattern — the
    /// lib test binary installs `alloc::TrackingAllocator` under
    /// cfg(test, debug_assertions); skip with a message if absent).
    #[test]
    #[cfg(debug_assertions)] // alloc counters only exist in debug builds
    fn g4_zero_alloc_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};

        let (n, d) = (128usize, 32usize);
        let mut rng = Lcg::new(0x0684_0614);
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let v = rng.next();
                x[i * d + j] = v;
                y[i * d + j] = 0.6 * v + 0.8 * rng.next();
            }
        }
        let mut s = CcaScratch::with_capacity(d, d, n);

        // Sentinel: confirm the allocator is installed.
        reset_alloc_stats();
        let _sentinel: Vec<u8> = vec![0u8; 256];
        let (sent_count, _) = get_alloc_stats();
        if sent_count == 0 {
            eprintln!("g4_zero_alloc_steady_state: TrackingAllocator not installed — SKIPPED");
            return;
        }
        drop(_sentinel);

        let _ = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s); // warmup
        reset_alloc_stats();
        for _ in 0..100 {
            let _ = svcca_into(&x, &y, d, d, n, 0.99, 1e-4, &mut s);
        }
        let (count, bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "svcca_into must be alloc-free in steady state; observed {count} allocations ({bytes} bytes) across 100 calls"
        );
    }
}
