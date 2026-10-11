//! basis_split — matched-pair block identification over a frozen embedding
//! (Plan 625; Research 618, SplitJEPA arXiv:2610.12349).
//!
//! # What it certifies — and what it never certifies
//!
//! Given (frozen embedding accessor, matched pairs) this module identifies the
//! **variant subspace** (top eigenspace of the whitened pair-difference second
//! moment) and its **invariant complement**, under a refuse-loud
//! identifiability verdict. It certifies **subspaces and insensitivity,
//! never individual latent axes** (SplitJEPA Thm 2 — identification is up to
//! within-block isometries). Every published split carries its held-out
//! [`SplitReport::pair_leakage`] residual beside it.
//!
//! **Naming law:** [`SplitReport::pair_leakage`] is a *generalization residual
//! of the fitted null space*, NOT purity — "purity" is reserved for
//! ground-truth settings (planted-block tests; deterministic-sim evaluations).
//! Never compare it to the paper's `‖A₁₂‖` figures (those use latent ground
//! truth for evaluation; we do not have it at serve time).
//!
//! # Whitening is the default (T1.1)
//!
//! Without whitening, low-total-variance directions masquerade as invariant.
//! The caller supplies the total covariance `Σ` (the module builds the ZCA
//! whitener `W = Σ^{-1/2}` in f64) or a precomputed whitener; with neither,
//! [`split`] returns [`Verdict::Unwhitened`] — never a silent unwhitened
//! split. Equivalence: bottom-eigenspace-of-whitened-`M` solves the
//! generalized eigenproblem `M v = λ Σ v` (whiten-then-eigendecompose is the
//! symmetric form of the same problem).
//!
//! **Contract for [`Whitening::Whitener`]:** the moment must have been
//! accumulated through [`PairDifferenceMoment::accumulate_whitened`] with the
//! SAME `W` — the module cannot verify that pairing, and a raw moment under a
//! `Whitener` claim is silently wrong. `TotalCovariance` + raw accumulation
//! has the same requirement (whiten at accumulate time with the built `W`);
//! see [`build_zca_whitener`].
//!
//! # Pair-budget law (Research 618 measured facts)
//!
//! The invariant nullspace identifies with ≈1k pairs; the complementary block
//! needs ≳ k + slack samples (head-fit is sample-hungry). Budget held-out
//! pairs for [`held_out_leakage`] accordingly.
//!
//! # Lineage
//!
//! The estimator is classical — SFA (Wiskott & Sejnowski 2002) when pairs are
//! temporal, within-class scatter / Fisher LDA when pairs are matched groups.
//! The contribution here is the identifiability gates + the certification
//! instruments, not the operator. In-workspace cousins: `tilr`
//! (Plan 425, the DEFAULT-ON τ-rank form this module's T0.3 arm measured),
//! `subspace_phase_gate` (the SVD primitive family), `ica_lens`
//! (katgpt-spectral, the whitening form precedent).
//!
//! # Envelope
//!
//! f64 accumulation throughout, d ≤ 1024 documented (`M` is d×d f64 — 8 MiB
//! at d=1024, caller-owned). Zero-alloc on every armed path: all scratch is a
//! caller-owned parameter, including the eigenvalue sort (in-place heapsort
//! with lockstep column permutation — O(d²) column moves worst case, a
//! calibration-time cost). Deterministic under fixed input order.
//!
//! # Solver routing (T1.2)
//!
//! d ≤ 16 → `karc::jacobi_eigen` (cyclic Jacobi, small-d envelope); otherwise
//! `linalg::symmetric_eig` (Householder+QL). The 16 < d < 256 crossover band
//! routes to `symmetric_eig` as well — measure a dedicated solver only if a
//! profile ever shows the band matters (per the plan: no third solver).

use crate::karc::jacobi_eigen;
use crate::linalg::symmetric_eig::{symmetric_eig, SymmetricEigScratch};

/// Maximum documented dimension (`M` is d×d f64, caller-owned).
pub const MAX_DIM: usize = 1024;
/// Dimensions at or below this route to the small-d Jacobi solver.
pub const JACOBI_MAX_DIM: usize = 16;

// ── errors ─────────────────────────────────────────────────────────────────────

/// Refuse-loud error set. Every variant names the caller-fixable fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasisSplitError {
    /// A buffer length does not match the expected size (named).
    DimensionMismatch { expected: usize },
    /// `d == 0` or `d > [`MAX_DIM`].
    DimOutOfRange,
    /// A pair difference carried a non-finite value (index disclosed).
    NonFiniteDifference { index: usize },
    /// The moment accumulated zero total variance — no split is definable.
    ZeroVariance,
    /// Whitening was requested from `Σ` but `Σ` is not positive definite
    /// (an eigenvalue ≤ 0 — whitening undefined).
    CovarianceNotPositiveDefinite,
    /// A scratch slice was shorter than the required length (names it).
    ScratchTooShort { required: usize },
}

impl core::fmt::Display for BasisSplitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DimensionMismatch { expected } => {
                write!(f, "buffer length != expected {expected}")
            }
            Self::DimOutOfRange => write!(f, "d out of range (1..={MAX_DIM})"),
            Self::NonFiniteDifference { index } => {
                write!(f, "pair difference[{index}] carries a non-finite value")
            }
            Self::ZeroVariance => write!(f, "difference moment has zero total variance"),
            Self::CovarianceNotPositiveDefinite => {
                write!(f, "total covariance has a non-positive eigenvalue; whitening undefined")
            }
            Self::ScratchTooShort { required } => {
                write!(f, "scratch shorter than required {required}")
            }
        }
    }
}

impl std::error::Error for BasisSplitError {}

// ── whitening (T1.1) ───────────────────────────────────────────────────────────

/// The whitening geometry. Supply [`Whitening::TotalCovariance`] (the module
/// builds the ZCA whitener in f64) or [`Whitening::Whitener`] (a precomputed
/// `W` with `W Σ Wᵀ = I`, row-major d×d — see the module contract on pairing
/// it with whitened accumulation). Absent both, [`split`] refuses with
/// [`Verdict::Unwhitened`].
#[derive(Debug, Clone, Copy)]
pub enum Whitening<'a> {
    /// d×d row-major total covariance `Σ` (the whitener is built internally).
    TotalCovariance(&'a [f64]),
    /// d×d row-major precomputed whitener `W` (`W Σ Wᵀ = I`); the moment must
    /// already be whitened with this same `W` at accumulate time.
    Whitener(&'a [f64]),
}

// ── the moment accumulator (T1.2) ──────────────────────────────────────────────

/// Zero-alloc accumulator for the pair-difference second moment
/// `M = (1/N) Σᵢ Δᵢ Δᵢᵀ` over supplied (whitened) differences.
///
/// `m` is the caller-owned d×d row-major buffer ([`Self::new`] zeroes it).
/// Deterministic accumulation order = call order. [`Self::finish`] divides by
/// N in place and returns the finalized moment.
pub struct PairDifferenceMoment<'m> {
    m: &'m mut [f64],
    d: usize,
    n: u64,
}

impl<'m> PairDifferenceMoment<'m> {
    /// Zeroes `m` and returns the accumulator. `m.len()` must be `d*d`.
    pub fn new(m: &'m mut [f64], d: usize) -> Result<Self, BasisSplitError> {
        if d == 0 || d > MAX_DIM {
            return Err(BasisSplitError::DimOutOfRange);
        }
        if m.len() != d * d {
            return Err(BasisSplitError::DimensionMismatch { expected: d * d });
        }
        m.fill(0.0);
        Ok(Self { m, d, n: 0 })
    }

    /// Accumulates one (already whitened) difference: `M += Δ Δᵀ`.
    /// Non-finite entries refuse BEFORE any mutation (the moment stays exact).
    pub fn accumulate(&mut self, delta: &[f64]) -> Result<(), BasisSplitError> {
        if delta.len() != self.d {
            return Err(BasisSplitError::DimensionMismatch { expected: self.d });
        }
        for (i, v) in delta.iter().enumerate() {
            if !v.is_finite() {
                return Err(BasisSplitError::NonFiniteDifference { index: i });
            }
        }
        // Symmetric rank-1 update: upper triangle + mirror. `di == 0.0` rows
        // contribute nothing anywhere — the skip keeps the O(d²) worst case
        // at O(d·nnz-rows) for sparse differences (deterministic either way).
        let d = self.d;
        for (i, &di) in delta.iter().enumerate() {
            if di == 0.0 {
                continue;
            }
            let row = i * d;
            for (j, dj) in delta[i..].iter().enumerate() {
                let v = di * dj;
                self.m[row + i + j] += v;
                if i + j != i {
                    self.m[(i + j) * d + i] += v;
                }
            }
        }
        self.n += 1;
        Ok(())
    }

    /// Accumulates one RAW difference under whitener `W` (row-major d×d):
    /// scratch receives `Δ Wᵀ` (length d), then the rank-1 update runs on it.
    /// Zero-alloc.
    pub fn accumulate_whitened(
        &mut self,
        whitener: &[f64],
        delta: &[f64],
        scratch: &mut [f64],
    ) -> Result<(), BasisSplitError> {
        let d = self.d;
        if whitener.len() != d * d {
            return Err(BasisSplitError::DimensionMismatch { expected: d * d });
        }
        if scratch.len() < d {
            return Err(BasisSplitError::ScratchTooShort { required: d });
        }
        // w_delta[j] = Σ_i Δ[i]·W[j][i]  (the j-th row of W dotted with Δ).
        for (j, wd) in scratch[..d].iter_mut().enumerate() {
            let row = j * d;
            let mut acc = 0.0;
            for (i, di) in delta.iter().enumerate() {
                acc += di * whitener[row + i];
            }
            *wd = acc;
        }
        self.accumulate(&scratch[..d])
    }

    /// Number of accumulated pairs.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Divides `M` by N in place (finalization) and returns the moment slice.
    pub fn finish(&mut self) -> &[f64] {
        if self.n > 0 {
            let inv = 1.0 / self.n as f64;
            for v in self.m.iter_mut() {
                *v *= inv;
            }
        }
        self.m
    }
}

// ── verdicts + report (T1.3 / T1.5) ────────────────────────────────────────────

/// Why a split was refused. Every variant is a *finding about the pairs*, not
/// an internal failure.
#[derive(Debug, Clone, PartialEq)]
pub enum UnidentifiedReason {
    /// The boundary fails the eigengap test `λ_k / λ_{k+1} ≥ τ_gap` — the
    /// claimed block boundary is not actually a spectral gap. NaN gap (a
    /// zero-eigenvalue division) lands here too.
    GapBelowTau { k: usize, gap_ratio: f64, tau_gap: f64 },
    /// `k` was supplied but out of range (`0` or `≥ d`).
    RankOutOfRange { k: usize },
}

/// The identifiability verdict (T1.3). `Unidentified` refuses loud — never a
/// silent split (the T0.3 adversarial gaps: shuffled pairs → `M ≈ 2I`, no
/// gap; degenerate pairs → rank-1 moment — both land here).
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The gap at the (supplied or selected) `k` clears `τ_gap`.
    SufficientVariation { k: usize, gap_ratio: f64 },
    /// Refuse-loud: the pairs do not certify a block boundary at `k`.
    Unidentified(UnidentifiedReason),
    /// Refuse-loud: no whitening geometry was supplied. The bottom-eigenspace
    /// reading is invalid without it (quiet-axis defense) — rerun with
    /// [`Whitening::TotalCovariance`] or [`Whitening::Whitener`].
    Unwhitened,
}

/// The split report (T1.5). `pair_leakage`/`agreement_residual` fill only via
/// [`held_out_leakage`] (they need held-out pairs); the rest are split-time
/// quantities.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitReport {
    /// Selected (or supplied) block dimension (0 on [`Verdict::Unwhitened`]).
    pub k: usize,
    /// The eigengap ratio `λ_k / λ_{k+1}` at the verdict boundary.
    pub gap_ratio: f64,
    /// `τ_gap` the verdict was held against.
    pub tau_gap: f64,
    /// Participation ratio `(Σλ)² / Σλ²` — REPORT-ONLY this pass. Never gate
    /// on it without a pre-registered derived threshold.
    pub participation_ratio: f64,
    /// `true` when `k` was chosen at the largest gap (caller supplied none).
    pub k_from_gap: bool,
    /// Held-out generalization residual `mean ‖P Δᵢ‖² / ‖Δᵢ‖²` — `None` until
    /// [`held_out_leakage`] runs on held-out pairs.
    pub pair_leakage: Option<f64>,
    /// Held-out agreement residual `mean ‖P Δᵢ‖ / ‖Δᵢ‖` (the non-squared
    /// sibling) — `None` until held-out pairs run.
    pub agreement_residual: Option<f64>,
}

// ── the eigengap gate (T1.3) ───────────────────────────────────────────────────

/// The eigengap rank gate. `eigvals_desc` must be sorted DESCENDING.
///
/// - `k = Some(k)`: the boundary is claimed — check `λ_k / λ_{k+1} ≥ τ_gap`
///   (1-based: `λ_k` is the k-th eigenvalue, `λ_{k+1}` the first of the
///   invariant block; `k` must be in `1..d`).
/// - `k = None`: select `k` at the LARGEST gap `λ_j / λ_{j+1}` over
///   `j ∈ 1..d`, then apply the same test.
///
/// A missing gap means the invariant block is not actually null — THE
/// identifiability-breaking failure — and the verdict refuses.
pub fn check_rank_gate(eigvals_desc: &[f64], k: Option<usize>, tau_gap: f64) -> (Verdict, bool) {
    let d = eigvals_desc.len();
    let (k, gap_ratio, k_from_gap) = match k {
        Some(k) => {
            if k == 0 || k >= d {
                return (Verdict::Unidentified(UnidentifiedReason::RankOutOfRange { k }), false);
            }
            (k, eigvals_desc[k - 1] / eigvals_desc[k], false)
        }
        None => {
            let mut best_j = 1usize;
            let mut best_gap = f64::NEG_INFINITY;
            for j in 1..d {
                let gap = eigvals_desc[j - 1] / eigvals_desc[j];
                if gap > best_gap {
                    best_gap = gap;
                    best_j = j;
                }
            }
            (best_j, best_gap, true)
        }
    };
    // NaN (a 0/0 division — the rank-exhausted moment) refuses; +inf is the
    // STRONGEST gap evidence (an exact-zero eigenvalue below the boundary —
    // the planted/degenerate-null case) and must pass.
    let verdict = if gap_ratio.is_nan() || gap_ratio < tau_gap {
        Verdict::Unidentified(UnidentifiedReason::GapBelowTau { k, gap_ratio, tau_gap })
    } else {
        Verdict::SufficientVariation { k, gap_ratio }
    };
    (verdict, k_from_gap)
}

/// Participation ratio `(Σλ)² / Σλ²` over the positive eigenvalues —
/// report-only. Delegates to the existing `subspace_phase_gate` primitive
/// (f32 — a report-only shape statistic; no second copy per the DRY law).
/// Zero-alloc: fixed stack buffer (d ≤ `MAX_DIM`).
pub fn participation_ratio(eigvals_desc: &[f64]) -> f64 {
    let d = eigvals_desc.len().min(MAX_DIM);
    let mut f32_spectrum = [0.0f32; MAX_DIM];
    for (dst, &v) in f32_spectrum[..d].iter_mut().zip(eigvals_desc) {
        *dst = v.max(0.0) as f32;
    }
    crate::subspace_phase_gate::participation_ratio(&f32_spectrum[..d]) as f64
}

// ── the split (T1.4) ───────────────────────────────────────────────────────────

/// In-place descending heapsort of `eigvals` with lockstep permutation of the
/// eigenvector COLUMNS of `eigvecs` (d×d row-major, column j at `j*d..`).
/// Zero-alloc; O(d log d) column swaps (each O(d) — a calibration-time cost).
/// MIN-heap + extract-to-end (the max-heap form yields ASCENDING order).
fn sort_spectrum_desc(eigvals: &mut [f64], eigvecs: &mut [f64], d: usize) {
    let n = eigvals.len();
    // Sift-down heap over (value, column) pairs.
    fn sift(v: &mut [f64], e: &mut [f64], d: usize, root: usize, end: usize) {
        let mut root = root;
        loop {
            let child = 2 * root + 1;
            if child >= end {
                break;
            }
            let mut swap = root;
            if v[swap] > v[child] {
                swap = child;
            }
            if child + 1 < end && v[swap] > v[child + 1] {
                swap = child + 1;
            }
            if swap == root {
                break;
            }
            v.swap(root, swap);
            let (a, b) = (root * d, swap * d);
            for i in 0..d {
                e.swap(a + i, b + i);
            }
            root = swap;
        }
    }
    // Heapify.
    let start = n / 2;
    for r in (0..=start).rev() {
        sift(eigvals, eigvecs, d, r, n);
    }
    // Extract min repeatedly (min lands at the end → array descending).
    for end in (1..n).rev() {
        eigvals.swap(0, end);
        let (a, b) = (0, end * d);
        for i in 0..d {
            eigvecs.swap(a + i, b + i);
        }
        sift(eigvals, eigvecs, d, 0, end);
    }
}

/// Whitened-difference split over a finalized moment (T1.4).
///
/// Whitening geometry (two forms):
/// - [`Whitening::TotalCovariance`]: caller supplies the RAW moment + the
///   total covariance Σ; this fn builds `W = Σ^{-1/2}` (into `whitener_out`)
///   and applies the similarity transform `M' = W M Wᵀ` itself (d³
///   calibration-time work; `M'` lands in `moment_scratch`). Use for
///   one-shot splits.
/// - [`Whitening::Whitener`]: the moment was ALREADY whitened at accumulate
///   time through [`PairDifferenceMoment::accumulate_whitened`] with the
///   same `W` (the streaming-cheap path — no d³ transform). The module
///   cannot verify that pairing; a raw moment under a `Whitener` claim is
///   silently wrong.
///
/// Scratch contract (all caller-owned, zero-alloc on this path):
/// - `eigvals` ≥ `d`, `eigvecs` ≥ `d*d` — outputs, sorted descending with
///   column-aligned vectors on success (including the `Unidentified` arms —
///   the spectrum is still the caller's diagnostic),
/// - `sym_scratch` — the `symmetric_eig` scratch (used at every d),
/// - `jacobi_scratch` ≥ `d*d` f64 — the Jacobi work matrix (d ≤ 16 arm),
/// - `moment_scratch` ≥ `d*d` f64 — receives `M' = W M Wᵀ` under
///   `TotalCovariance`; untouched under `Whitener`,
/// - `whitener_out` ≥ `d*d` — receives the built ZCA whitener under
///   `TotalCovariance`; untouched under `Whitener`,
///
/// On [`Verdict::SufficientVariation`], `basis_out` (`k*d` row-major)
/// receives the variant basis rows (top-k eigenvector columns).
#[allow(clippy::too_many_arguments)]
pub fn split(
    moment: &[f64],
    d: usize,
    whitening: Option<Whitening<'_>>,
    k: Option<usize>,
    tau_gap: f64,
    eigvals: &mut [f64],
    eigvecs: &mut [f64],
    basis_out: &mut [f64],
    sym_scratch: &mut SymmetricEigScratch,
    jacobi_scratch: &mut [f64],
    moment_scratch: &mut [f64],
    whitener_out: &mut [f64],
) -> Result<(Verdict, SplitReport), BasisSplitError> {
    if d == 0 || d > MAX_DIM {
        return Err(BasisSplitError::DimOutOfRange);
    }
    if moment.len() != d * d || eigvals.len() < d || eigvecs.len() < d * d {
        return Err(BasisSplitError::DimensionMismatch { expected: d * d });
    }
    let empty_report = || SplitReport {
        k: 0,
        gap_ratio: f64::NAN,
        tau_gap,
        participation_ratio: f64::NAN,
        k_from_gap: false,
        pair_leakage: None,
        agreement_residual: None,
    };
    let Some(geo) = whitening else {
        return Ok((Verdict::Unwhitened, empty_report()));
    };

    // Total-variance screen BEFORE any solve (the TILR ZeroVariance law).
    let total: f64 = (0..d).map(|i| moment[i * d + i]).sum();
    if !(total.is_finite() && total > 0.0) {
        return Err(BasisSplitError::ZeroVariance);
    }

    // The moment the eigensolve reads: raw under `Whitener` (caller
    // pre-whitened at accumulate time), `W M Wᵀ` under `TotalCovariance`.
    let solve_moment: &[f64] = match geo {
        Whitening::Whitener(_) => moment,
        Whitening::TotalCovariance(sigma) => {
            build_zca_whitener(sigma, d, whitener_out, eigvals, eigvecs, sym_scratch)?;
            if moment_scratch.len() < d * d {
                return Err(BasisSplitError::ScratchTooShort { required: d * d });
            }
            // M' = W M Wᵀ via T = M Wᵀ, then W T (d³ calibration-time work).
            for i in 0..d {
                for j in 0..d {
                    let mut acc = 0.0;
                    for l in 0..d {
                        acc += moment[i * d + l] * whitener_out[j * d + l];
                    }
                    moment_scratch[i * d + j] = acc;
                }
            }
            for i in 0..d {
                for j in 0..d {
                    let mut acc = 0.0;
                    for l in 0..d {
                        acc += whitener_out[i * d + l] * moment_scratch[l * d + j];
                    }
                    moment_scratch[i * d + j] = acc;
                }
            }
            &moment_scratch[..d * d]
        }
    };

    // Eigensolve the (whitened) moment. Deterministic; both solvers emit
    // (near-)orthonormal vectors — normalized exact below.
    if d <= JACOBI_MAX_DIM {
        if jacobi_scratch.len() < d * d {
            return Err(BasisSplitError::ScratchTooShort { required: d * d });
        }
        eigvals[..d].fill(0.0);
        jacobi_eigen(eigvals, eigvecs, solve_moment, jacobi_scratch, d, 1e-14, 100);
    } else {
        sym_scratch.ensure_capacity(d);
        symmetric_eig(eigvals, eigvecs, solve_moment, sym_scratch, d, 200);
    }

    sort_spectrum_desc(eigvals, eigvecs, d);

    // Pin orthonormality exact (the QL path is near-orthonormal).
    for j in 0..d {
        let col = j * d;
        let norm: f64 = (0..d).map(|r| eigvecs[col + r] * eigvecs[col + r]).sum::<f64>().sqrt();
        if norm > 0.0 {
            for r in 0..d {
                eigvecs[col + r] /= norm;
            }
        }
    }

    let (verdict, k_from_gap) = check_rank_gate(&eigvals[..d], k, tau_gap);
    let pr = participation_ratio(&eigvals[..d]);
    let (k_out, gap_out) = match &verdict {
        Verdict::SufficientVariation { k, gap_ratio } => (*k, *gap_ratio),
        Verdict::Unidentified(UnidentifiedReason::GapBelowTau { k, gap_ratio, .. }) => {
            (*k, *gap_ratio)
        }
        Verdict::Unidentified(UnidentifiedReason::RankOutOfRange { k }) => (*k, f64::NAN),
        Verdict::Unwhitened => unreachable!("whitening supplied above"),
    };

    if let Verdict::SufficientVariation { k: kk, .. } = &verdict {
        if basis_out.len() < kk * d {
            return Err(BasisSplitError::ScratchTooShort { required: kk * d });
        }
        for row in 0..*kk {
            basis_out[row * d..(row + 1) * d].copy_from_slice(&eigvecs[row * d..(row + 1) * d]);
        }
    }

    Ok((
        verdict,
        SplitReport {
            k: k_out,
            gap_ratio: gap_out,
            tau_gap,
            participation_ratio: pr,
            k_from_gap,
            pair_leakage: None,
            agreement_residual: None,
        },
    ))
}

/// Builds the ZCA whitener `W = Σ^{-1/2}` (row-major d×d) from the total
/// covariance via the f64 symmetric eigendecomposition:
/// `Σ = V Λ Vᵀ ⇒ W = V Λ^{-1/2} Vᵀ`. Refuses on a non-positive eigenvalue
/// (whitening undefined). `eigvals`/`eigvecs` are scratch outputs (the
/// caller's diagnostics).
pub fn build_zca_whitener(
    sigma: &[f64],
    d: usize,
    whitener_out: &mut [f64],
    eigvals: &mut [f64],
    eigvecs: &mut [f64],
    sym_scratch: &mut SymmetricEigScratch,
) -> Result<(), BasisSplitError> {
    if d == 0 || d > MAX_DIM {
        return Err(BasisSplitError::DimOutOfRange);
    }
    if sigma.len() != d * d || whitener_out.len() < d * d || eigvals.len() < d || eigvecs.len() < d * d
    {
        return Err(BasisSplitError::DimensionMismatch { expected: d * d });
    }
    sym_scratch.ensure_capacity(d);
    symmetric_eig(eigvals, eigvecs, sigma, sym_scratch, d, 200);
    let min_ev = eigvals[..d].iter().cloned().fold(f64::INFINITY, f64::min);
    if !(min_ev.is_finite() && min_ev > 0.0) {
        return Err(BasisSplitError::CovarianceNotPositiveDefinite);
    }
    // W = V Λ^{-1/2} Vᵀ via t = Λ^{-1/2} Vᵀ (row j = column j scaled), then
    // W = t·Vᵀ — symmetric, computed upper-triangle + mirror.
    for j in 0..d {
        let s = 1.0 / eigvals[j].sqrt();
        for r in 0..d {
            eigvecs[j * d + r] *= s;
        }
    }
    for i in 0..d {
        for l in i..d {
            let mut acc = 0.0;
            for j in 0..d {
                acc += eigvecs[j * d + i] * eigvecs[j * d + l];
            }
            whitener_out[i * d + l] = acc;
            if i != l {
                whitener_out[l * d + i] = acc;
            }
        }
    }
    Ok(())
}

// ── the invariant projector (T1.4) + residuals (T1.5) ─────────────────────────

/// Applies the invariant projector `P = I − VᵀV` (variant basis rows `basis`,
/// `k × d` row-major) to `x`: `out = x − Bᵀ(B x)`. `bx_scratch` holds the
/// `k` dot products. Zero-alloc, one d×d-matvec class of work.
pub fn project_invariant_into(
    basis: &[f64],
    k: usize,
    d: usize,
    x: &[f64],
    bx_scratch: &mut [f64],
    out: &mut [f64],
) -> Result<(), BasisSplitError> {
    if k == 0 || k > MAX_DIM || basis.len() < k * d || x.len() < d || out.len() < d {
        return Err(BasisSplitError::DimensionMismatch { expected: k * d });
    }
    if bx_scratch.len() < k {
        return Err(BasisSplitError::ScratchTooShort { required: k });
    }
    // B x  (k dots).
    for row in 0..k {
        let b_row = &basis[row * d..(row + 1) * d];
        let mut acc = 0.0;
        for i in 0..d {
            acc += b_row[i] * x[i];
        }
        bx_scratch[row] = acc;
    }
    // out = x − Bᵀ (B x).
    for i in 0..d {
        let mut acc = 0.0;
        for row in 0..k {
            acc += basis[row * d + i] * bx_scratch[row];
        }
        out[i] = x[i] - acc;
    }
    Ok(())
}

/// Held-out generalization residual (T1.5): fills `report.pair_leakage`
/// (`mean ‖P Δᵢ‖² / ‖Δᵢ‖²`) and `report.agreement_residual` (the non-squared
/// sibling) over the supplied pairs. Zero-alloc.
///
/// This is a *generalization residual of the fitted null space* — it
/// approaches 0 by construction once N ≫ d on pairs from the same generator.
/// It is NOT purity.
pub fn held_out_leakage(
    basis: &[f64],
    k: usize,
    d: usize,
    pairs: &[&[f64]],
    proj_scratch: &mut [f64],
    bx_scratch: &mut [f64],
    report: &mut SplitReport,
) -> Result<(), BasisSplitError> {
    if pairs.is_empty() {
        return Ok(());
    }
    if proj_scratch.len() < d {
        return Err(BasisSplitError::ScratchTooShort { required: d });
    }
    let mut sq_sum = 0.0;
    let mut lin_sum = 0.0;
    for (idx, delta) in pairs.iter().enumerate() {
        if delta.len() != d {
            return Err(BasisSplitError::DimensionMismatch { expected: d });
        }
        for v in delta.iter() {
            if !v.is_finite() {
                return Err(BasisSplitError::NonFiniteDifference { index: idx });
            }
        }
        project_invariant_into(basis, k, d, delta, bx_scratch, proj_scratch)?;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..d {
            num += proj_scratch[i] * proj_scratch[i];
            den += delta[i] * delta[i];
        }
        if den > 0.0 {
            sq_sum += num / den;
            lin_sum += (num / den).sqrt();
        }
    }
    report.pair_leakage = Some(sq_sum / pairs.len() as f64);
    report.agreement_residual = Some(lin_sum / pairs.len() as f64);
    Ok(())
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG + Box–Muller (the unseeded-global-RNG gate).
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn next_gauss(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
        }
    }

    const D: usize = 32; // routes to symmetric_eig (the d > 16 path)
    const K: usize = 8; // planted invariant dims
    /// The VARIANT dimension (the gate k = the variant rank, D - K).
    const VK: usize = D - K;

    /// Planted world: pairs share the invariant block exactly, variant block
    /// full-rank standard normal, coordinates axis-aligned.
    fn planted_differences(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                let mut delta = vec![0.0; D];
                for (i, dv) in delta.iter_mut().enumerate() {
                    if i >= K {
                        *dv = rng.next_gauss();
                    }
                }
                delta
            })
            .collect()
    }

    fn identity_sigma() -> Vec<f64> {
        let mut s = vec![0.0; D * D];
        for i in 0..D {
            s[i * D + i] = 1.0;
        }
        s
    }

    struct SplitRig {
        eigvals: Vec<f64>,
        eigvecs: Vec<f64>,
        basis: Vec<f64>,
        sym: SymmetricEigScratch,
        jacobi: Vec<f64>,
        moment_scratch: Vec<f64>,
        whitener: Vec<f64>,
    }

    impl SplitRig {
        fn new(basis_rows: usize) -> Self {
            Self {
                eigvals: vec![0.0; D],
                eigvecs: vec![0.0; D * D],
                basis: vec![0.0; basis_rows * D],
                sym: SymmetricEigScratch::new(),
                jacobi: vec![0.0; D * D],
                moment_scratch: vec![0.0; D * D],
                whitener: vec![0.0; D * D],
            }
        }
        fn run(
            &mut self,
            m: &[f64],
            k: Option<usize>,
            tau_gap: f64,
        ) -> Result<(Verdict, SplitReport), BasisSplitError> {
            split(
                m,
                D,
                Some(Whitening::TotalCovariance(&identity_sigma())),
                k,
                tau_gap,
                &mut self.eigvals,
                &mut self.eigvecs,
                &mut self.basis,
                &mut self.sym,
                &mut self.jacobi,
                &mut self.moment_scratch,
                &mut self.whitener,
            )
        }
    }

    fn accumulate(diffs: &[Vec<f64>]) -> Vec<f64> {
        let mut m = vec![0.0; D * D];
        let mut acc = PairDifferenceMoment::new(&mut m, D).unwrap();
        for delta in diffs {
            acc.accumulate(delta).unwrap();
        }
        acc.finish();
        m
    }

    /// G1 core: planted recovery — variant basis rows live only on the variant
    /// coordinates and the complement preserves each invariant axis exactly.
    #[test]
    fn split_recovers_planted_blocks_axis_aligned() {
        let diffs = planted_differences(2048, 0x6255_0001);
        let m = accumulate(&diffs);
        let mut rig = SplitRig::new(VK);
        let (verdict, report) = rig.run(&m, Some(VK), 10.0).unwrap();
        assert!(matches!(verdict, Verdict::SufficientVariation { .. }));
        // The gap is enormous (isotropic variant vs exact-zero invariant).
        assert!(report.gap_ratio > 1e6, "gap_ratio = {}", report.gap_ratio);
        assert!(!report.k_from_gap);

        for (row_idx, row) in rig.basis.chunks(D).enumerate() {
            debug_assert_eq!(row.len(), D);
            for (t, v) in row.iter().take(K).enumerate() {
                assert!(
                    v.abs() < 1e-6,
                    "basis row {row_idx} leaks invariant axis {t}: {v}"
                );
            }
        }
        // P e_t = e_t for every invariant axis.
        let mut bx = vec![0.0; VK];
        let mut out = vec![0.0; D];
        for t in 0..K {
            let mut e = vec![0.0; D];
            e[t] = 1.0;
            project_invariant_into(&rig.basis, VK, D, &e, &mut bx, &mut out).unwrap();
            for (i, o) in out.iter().enumerate() {
                let want = if i == t { 1.0 } else { 0.0 };
                assert!((o - want).abs() < 1e-9, "P e_{t}[{i}] = {o}, want {want}");
            }
        }
    }

    /// The whitened path (T1.1 quiet-axis defense): a NOISY invariant block
    /// (matched pairs never share z_H exactly in measurement) plus a
    /// shared-content-dominated Σ. Unwhitened, the boundary gap is
    /// min σ²_L / ε² = 4 (< τ_gap = 10) → refuse; whitened by Σ (the H axes'
    /// total variance is dominated by the shared amplitude a_H = 10), the
    /// whitened H variation drops to ε²/a_H² = 1e-4 → boundary gap 400 →
    /// SufficientVariation. THAT is what whitening buys: it ranks axes by
    /// relative variation, not absolute.
    #[test]
    fn whitened_accumulation_defends_the_quiet_axis() {
        // Model: every axis carries shared content (Σ is NOT identity); the H
        // axes carry the most (a_H = 10); pair differences: H = ε noise
        // (ε = 0.1), L = anisotropic variant (σ² from 0.04 up).
        const A_H: f64 = 10.0;
        const EPS: f64 = 0.1;
        let mut sigma_l = [0.0f64; VK];
        for (t, s) in sigma_l.iter_mut().enumerate() {
            *s = 0.2 + 0.05 * t as f64;
        }
        let mut rng = Lcg(0x6255_0002);
        let n = 8192;
        let mut raw: Vec<Vec<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut delta = vec![0.0; D];
            for (i, dv) in delta.iter_mut().enumerate() {
                if i < K {
                    *dv = rng.next_gauss() * EPS;
                } else {
                    *dv = rng.next_gauss() * sigma_l[i - K];
                }
            }
            raw.push(delta);
        }
        // Σ of the raw coordinates (diagonal by construction): H axes
        // a_H² + ε², L axes 1 + σ² (shared amplitude 1.0 + variant).
        let mut sigma = vec![0.0; D * D];
        for i in 0..D {
            sigma[i * D + i] = if i < K {
                A_H * A_H + EPS * EPS
            } else {
                1.0 + sigma_l[i - K] * sigma_l[i - K]
            };
        }
        let mut w = vec![0.0; D * D];
        let mut ev = vec![0.0; D];
        let mut evec = vec![0.0; D * D];
        let mut sym = SymmetricEigScratch::new();
        build_zca_whitener(&sigma, D, &mut w, &mut ev, &mut evec, &mut sym).unwrap();

        // Unwhitened (identity whitening = raw moment): boundary gap ≈ 4 < 10.
        let m_unwhitened = accumulate(&raw);
        let ident = {
            let mut s = vec![0.0; D * D];
            for i in 0..D {
                s[i * D + i] = 1.0;
            }
            s
        };
        let mut rig_u = SplitRig::new(VK);
        let (v_u, rep_u) = split(
            &m_unwhitened,
            D,
            Some(Whitening::Whitener(&ident)),
            Some(VK),
            10.0,
            &mut rig_u.eigvals,
            &mut rig_u.eigvecs,
            &mut rig_u.basis,
            &mut rig_u.sym,
            &mut rig_u.jacobi,
            &mut rig_u.moment_scratch,
            &mut rig_u.whitener,
        )
        .unwrap();
        assert!(
            matches!(v_u, Verdict::Unidentified(_)),
            "unwhitened noisy-invariant world must refuse at τ=10, got {v_u:?} (gap {})",
            rep_u.gap_ratio
        );

        // Whitened at accumulate time (the streaming path): boundary gap ≈ 400.
        let mut m_w = vec![0.0; D * D];
        let mut scratch = vec![0.0; D];
        {
            let mut acc = PairDifferenceMoment::new(&mut m_w, D).unwrap();
            for delta in &raw {
                acc.accumulate_whitened(&w, delta, &mut scratch).unwrap();
            }
            acc.finish();
        }
        let mut rig_w = SplitRig::new(VK);
        let (v_w, rep_w) = split(
            &m_w,
            D,
            Some(Whitening::Whitener(&w)),
            Some(VK),
            10.0,
            &mut rig_w.eigvals,
            &mut rig_w.eigvecs,
            &mut rig_w.basis,
            &mut rig_w.sym,
            &mut rig_w.jacobi,
            &mut rig_w.moment_scratch,
            &mut rig_w.whitener,
        )
        .unwrap();
        assert!(matches!(v_w, Verdict::SufficientVariation { .. }));
        assert!(
            rep_w.gap_ratio > rep_u.gap_ratio * 10.0,
            "whitened gap {} should dwarf unwhitened gap {}",
            rep_w.gap_ratio,
            rep_u.gap_ratio
        );
        // And the whitened basis concentrates on the variant coordinates: the
        // H-axis loadings are small (whitened H variation is ε²/a_H² relative).
        for row in rig_w.basis.chunks(D) {
            debug_assert_eq!(row.len(), D);
            let h_mass: f64 = row.iter().take(K).map(|v| v * v).sum();
            assert!(
                h_mass < 0.05,
                "whitened basis row carries {h_mass} of its mass on invariant axes"
            );
        }
    }

    /// Adversarial (T0.3's gaps, now gated): value-shuffled pairs destroy the
    /// gap — the gate must refuse, never split.
    #[test]
    fn shuffled_pairs_refuse_loud() {
        let mut diffs = planted_differences(2048, 0x6255_0003);
        let mut rng = Lcg(0x5100_0000_6255_0004);
        for delta in diffs.iter_mut() {
            for _ in 0..delta.len() {
                let i = (rng.next_f64() * delta.len() as f64) as usize % delta.len();
                let j = (rng.next_f64() * delta.len() as f64) as usize % delta.len();
                delta.swap(i, j);
            }
        }
        let m = accumulate(&diffs);
        let mut rig = SplitRig::new(VK);
        let (verdict, report) = rig.run(&m, Some(VK), 10.0).unwrap();
        assert!(
            matches!(verdict, Verdict::Unidentified(_)),
            "shuffled pairs must refuse, got {verdict:?} (gap_ratio {})",
            report.gap_ratio
        );
    }

    /// `k = None` selects at the largest gap and reports it.
    #[test]
    fn k_selection_picks_the_largest_gap() {
        let diffs = planted_differences(2048, 0x6255_0005);
        let m = accumulate(&diffs);
        let mut rig = SplitRig::new(D); // k unknown yet — generous rows
        let (verdict, report) = rig.run(&m, None, 10.0).unwrap();
        assert!(matches!(verdict, Verdict::SufficientVariation { .. }));
        assert!(report.k_from_gap);
        assert_eq!(report.k, VK, "the largest gap in the planted world IS the block boundary");
    }

    /// Unwhitened refuses; degenerate (identical) pairs refuse; zero variance
    /// errors; the small-d Jacobi path (d ≤ 16) agrees with the QL path.
    #[test]
    fn refusal_paths_and_jacobi_parity() {
        let diffs = planted_differences(256, 0x6255_0006);
        let m = accumulate(&diffs);

        // Unwhitened.
        let mut ev = vec![0.0; D];
        let mut evec = vec![0.0; D * D];
        let mut basis = vec![0.0; K * D];
        let mut sym = SymmetricEigScratch::new();
        let mut jac = vec![0.0; D * D];
        let mut m_scr = vec![0.0; D * D];
        let mut wh = vec![0.0; D * D];
        let (verdict, _) = split(
            &m, D, None, Some(K), 10.0, &mut ev, &mut evec, &mut basis, &mut sym, &mut jac,
            &mut m_scr, &mut wh,
        )
        .unwrap();
        assert_eq!(verdict, Verdict::Unwhitened);

        // Degenerate: all-identical differences → rank-1 moment → λ_k/λ_{k+1}
        // at k ≥ 2 is 0/0 = NaN → refuse.
        let mut m_deg = vec![0.0; D * D];
        {
            let mut acc = PairDifferenceMoment::new(&mut m_deg, D).unwrap();
            let delta = vec![1.0; D];
            for _ in 0..64 {
                acc.accumulate(&delta).unwrap();
            }
            acc.finish();
        }
        let (verdict_deg, _) = split(
            &m_deg, D, Some(Whitening::TotalCovariance(&identity_sigma())), Some(K), 10.0,
            &mut ev, &mut evec, &mut basis, &mut sym, &mut jac, &mut m_scr, &mut wh,
        )
        .unwrap();
        assert!(matches!(verdict_deg, Verdict::Unidentified(_)));

        // Zero variance.
        let m_zero = vec![0.0; D * D];
        let err = split(
            &m_zero, D, Some(Whitening::TotalCovariance(&identity_sigma())), Some(K), 10.0,
            &mut ev, &mut evec, &mut basis, &mut sym, &mut jac, &mut m_scr, &mut wh,
        );
        assert!(matches!(err, Err(BasisSplitError::ZeroVariance)));

        // Jacobi parity at d ≤ 16: eigenvalues must match a fresh QL solve.
        const DS: usize = 12;
        const KS: usize = 4;
        let mut rng = Lcg(0x6255_0007);
        let mut ms = vec![0.0; DS * DS];
        {
            let mut acc = PairDifferenceMoment::new(&mut ms, DS).unwrap();
            for _ in 0..1024 {
                let mut delta = vec![0.0; DS];
                for (i, dv) in delta.iter_mut().enumerate() {
                    if i >= KS {
                        *dv = rng.next_gauss();
                    }
                }
                acc.accumulate(&delta).unwrap();
            }
            acc.finish();
        }
        let mut ev_j = vec![0.0; DS];
        let mut evec_j = vec![0.0; DS * DS];
        let mut basis_j = vec![0.0; (DS - KS) * DS];
        let mut ws_small = vec![0.0; DS * DS];
        let mut m_scr2 = vec![0.0; DS * DS];
        let mut ident_s = vec![0.0; DS * DS];
        for i in 0..DS {
            ident_s[i * DS + i] = 1.0;
        }
        let (v_j, _) = split(
            &ms,
            DS,
            Some(Whitening::TotalCovariance(&ident_s)),
            Some(DS - KS),
            10.0,
            &mut ev_j,
            &mut evec_j,
            &mut basis_j,
            &mut sym,
            &mut ws_small,
            &mut m_scr2,
            &mut wh,
        )
        .unwrap();
        assert!(matches!(v_j, Verdict::SufficientVariation { .. }));
        let mut ev_q = vec![0.0; DS];
        let mut evec_q = vec![0.0; DS * DS];
        sym.ensure_capacity(DS);
        symmetric_eig(&mut ev_q, &mut evec_q, &ms, &mut sym, DS, 200);
        ev_q.sort_by(|a, b| b.total_cmp(a));
        for i in 0..DS {
            assert!(
                (ev_j[i] - ev_q[i]).abs() < 1e-9 + 1e-7 * ev_q[i].abs(),
                "jacobi vs QL eigenvalue {i}: {} vs {}",
                ev_j[i],
                ev_q[i]
            );
        }
    }

    /// Held-out residual (T1.5): ~0 on same-generator held-out pairs — the
    /// generalization residual of the fitted null space, never a purity claim.
    #[test]
    fn held_out_residual_near_zero_on_same_generator() {
        let mut diffs = planted_differences(2048, 0x6255_0008);
        let held = diffs.split_off(1536);
        let m = accumulate(&diffs);
        let mut rig = SplitRig::new(VK);
        let (verdict, mut report) = rig.run(&m, Some(VK), 10.0).unwrap();
        assert!(matches!(verdict, Verdict::SufficientVariation { .. }));
        let held_refs: Vec<&[f64]> = held.iter().map(|d| d.as_slice()).collect();
        let mut proj = vec![0.0; D];
        let mut bx = vec![0.0; VK];
        held_out_leakage(&rig.basis, VK, D, &held_refs, &mut proj, &mut bx, &mut report).unwrap();
        let leakage = report.pair_leakage.expect("filled");
        let agreement = report.agreement_residual.expect("filled");
        assert!(
            leakage < 1e-6,
            "same-generator held-out leakage {leakage} — should be ~0 (generalization residual, not purity)"
        );
        assert!(agreement < 1e-3);
    }
}
