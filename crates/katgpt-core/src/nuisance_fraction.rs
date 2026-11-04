//! ρ alignability measurement — the state-explained variance fraction over
//! grouped multi-view embeddings, plus the rotation-alignment floor it
//! implies (Issue 978, Research 646 — [arXiv:2610.39302](https://arxiv.org/abs/2610.39302)
//! Thm 3 + Prop 2, Balestriero & LeCun, *Proving and Improving the Platonic
//! Representation Hypothesis with JEPAs*).
//!
//! # The measurement
//!
//! Given `G` groups of views sharing one world state per group (a rule's
//! files under formatting nuisances, an object under camera poses),
//! [`rho_fraction`] measures
//!
//! ```text
//! ρ(f) = tr(Cov(E[f(x)|z])^{1/2}) / k ∈ [0, 1]
//! ```
//!
//! the fraction of embedding variance the *state* explains — a **one-space**
//! statistic: no pairs, no labels, no training. Thm 3 lower-bounds the
//! best-rotation cross-space alignment error by
//! [`alignment_floor`]` = 2k(1 − min ρ)` in the whitened RMS² metric, so ρ is
//! a go/no-go gate: when the floor already proves misalignment, Procrustes /
//! transport spend is refused before it runs.
//!
//! # The estimator (pinned conventions)
//!
//! 1. Population covariance `1/N` over all rows (f64, sequential folds);
//!    f64 ZCA whitening `W = V Λ̃^{-1/2} Vᵀ` from a
//!    [`crate::linalg::symmetric_eig`], with the pseudo-inverse floor
//!    `λ̃ = max(λ, 1e-10·λ_max)` — near-singular populations get the floor;
//!    a non-positive `λ_max` refuses.
//! 2. Per-group whitened mean `m̂_g` and within covariance
//!    `C_W,g = scatter_g/(n_g−1)` (unbiased). The corrected between-group
//!    covariance subtracts the finite-group spread:
//!    `C_B = Σ_g (n_g/N)·[(m̂_g − m̄)(m̂_g − m̄)ᵀ − C_W,g/n_g]` — at equal
//!    `n_g` this is `sample-between − C̄_W/n` (the paper's Table-6 form; the
//!    uncorrected form overstates `Cov(E[f|z])` by ≈ within-variance/n_g).
//!    The correction is carried in [`RHO_CONVENTION`], which every report
//!    quotes.
//! 3. `ρ = Σᵢ √max(λᵢ, 0)/k` over the eigenvalues of `C_B` (one more
//!    `symmetric_eig`). Tiny negatives from numeric noise clamp to 0; the
//!    RAW spectrum (sorted descending) stays readable on the scratch via
//!    [`RhoScratch::last_between_eigenvalues`] — negative eigenvalues
//!    disclose how far the correction over-subtracted.
//!
//! # The G ≪ k regime (read before quoting a number)
//!
//! `Σ√λ/k` divides by `k`, but the between-scatter of `G` group means has
//! rank ≤ `G−1`. When `G < k` most directions carry clamped-to-zero
//! eigenvalues and ρ reads LOW (concavity + clamping over an unequal
//! spectrum): the pre-registered `k=16, 8×4` fixture reads planted 0.95 as
//! ≈0.42 — monotone but compressed. The `planted_rho_recovers` arm therefore
//! pins recovery at `G ≫ k` (the paper's regime), and the brief's literal
//! 8×4 shape ships as `regime_arm_g_below_k`, pinning monotonicity + the
//! compression so the regime boundary can never silently move. Within one
//! (G, k) shape ρ stays monotone-informative (encoder ranking survives);
//! absolute values do not transfer across shapes.
//!
//! # Loud refusals, never silent zeros
//!
//! ρ is **undefined on single-view data** (Prop 2(i) forbids the unpaired
//! fallback): fewer than two groups, any group with fewer than two members,
//! a ragged row, non-finite input, `N < k` (structurally rank-deficient
//! population — the floor would whiten garbage), or a non-positive `λ_max`
//! all return [`RhoError`]. ρ never silently reads 0.
//!
//! # The planting trap (convention sign, pinned by test)
//!
//! Fixtures MUST plant `f = ρ·s + √(1−ρ²)·n` with `Cov(s) = Cov(n) = I`:
//! then `Cov(E[f|z]) = ρ²·I` and ρ reads the planted value. The naive
//! `f = √ρ·s + √(1−ρ)·n` makes `C_f = ρ·I` and reads **√ρ** (planted 0.5
//! reads 0.707) — `naive_planting_tripwire` asserts exactly that, so the
//! convention can never silently flip.
//!
//! # Cost
//!
//! Two O(k³) f64 eigendecompositions + O(N·k²) streaming per report —
//! offline audit cadence (`knn_entropy` precedent): the report is amortized
//! the moment it refuses one doomed Procrustes/transport run. Measured
//! figures live in the `g2_report_latency` arm (the issue's ~10 µs bar was
//! mis-estimated for dense f64 eigensolves — Ω(N·k² + k³) work at the named
//! posture; corrected there with box state).
//!
//! # Neighbor metric, not a duplicate (substrate-first note)
//!
//! [`crate::data_probe::geometry::within_class_effective_rank`] ranks the
//! WITHIN-class spread; ρ is the corrected BETWEEN-group variance fraction
//! plus the Thm 3 floor — different question, different statistic.
//! Vocabulary aliases for grep: `nuisance_fraction`, `rho_fraction`,
//! `alignability`, `between_group`, `scene_fraction`.
//!
//! # Feature gating
//!
//! Opt-in `nuisance_fraction` per the no-default-consumer rule; promotion is
//! consumer-gated (Issue 978 T4: a consumer must show refusing a doomed
//! rotation/transport saves measurable compute — candidates: `leakage_probe`
//! front-end, katgpt-canon GL(k) pre-check per Research 646 Law 2, MAG
//! TransferMetric headline). T5 consumers live in OTHER repos
//! (riir-refine / riir-reflex issue discipline). Concurrent art to cite in
//! consumer docs: Schnaus et al. arXiv:2610.09411 (cross-set predictability)
//! — the gate here is adoption engineering, not a novelty claim.

// (Module gating is handled by `#[cfg(feature = "nuisance_fraction")]` on the
// `mod` declaration in `lib.rs`; this file must NOT duplicate it.)

use crate::linalg::{SymmetricEigScratch, symmetric_eig};

/// The whitening + estimator conventions every [`RhoReport`] carries. ρ is
/// invariant across *choices* of whitening (they differ by an orthogonal
/// transform) but incomparable across differently-prepared spaces — always
/// print this string beside a ρ number.
pub const RHO_CONVENTION: &str = "population-whitened f64 zca (cov 1/N, pseudo-inv floor eps=1e-10*lam_max); between = sum_g(n_g/N)[(m_g-m_bar)(m_g-m_bar)^T - C_Wg/n_g], within cov 1/(n_g-1); rho = sum sqrt(max(lam,0))/k";

/// Relative pseudo-inverse floor under the population spectrum
/// (`λ̃ = max(λ, eps·λ_max)`). Near-singular populations get the floor, not a
/// refusal — the refusal is reserved for a non-positive `λ_max` and for the
/// structural `N < k` case. At the 1e-10 floor the noise amplification
/// (`√(λ_max/λ̃)`) can reach 1e5 in floored directions: treat ρ from a
/// posture with `N ≈ k` as advisory, not calibrated.
const EPS_REL: f64 = 1e-10;

/// Per-eigenvalue QL iteration cap — the [`crate::linalg::symmetric_eig`]
/// posture `leakage_probe::transport` ships with.
const EIG_MAX_ITERS: usize = 60;

/// Why a ρ measurement refused to run. Every variant is a loud refusal — ρ is
/// undefined on single-view data and never silently reads 0 (Research 646 §8).
#[derive(Debug, Clone, PartialEq)]
pub enum RhoError {
    /// Fewer than two groups — the between-group covariance does not exist.
    TooFewGroups {
        /// The group count that was refused.
        got: usize,
    },
    /// A group carried fewer than two members — its mean carries no
    /// within-variance to correct and ρ is undefined on single-view groups.
    GroupTooSmall {
        /// Index of the offending group.
        group: usize,
        /// Its member count (always < 2).
        len: usize,
    },
    /// A group's buffer length is not a whole number of `k`-wide rows
    /// (row-major flat slices: a ragged row shows up as a length mismatch).
    DimMismatch {
        /// Index of the offending group.
        group: usize,
        /// The first row's width.
        expected: usize,
        /// This row's width.
        got: usize,
    },
    /// A non-finite (NaN / ±inf) value in a group's rows.
    NonFinite {
        /// Index of the offending group.
        group: usize,
    },
    /// The first row was empty — `k = 0` has no eigensolve.
    EmptyDimension,
    /// `N < k`: the `k×k` population covariance from `N` rows is
    /// structurally rank-deficient; the pseudo-inverse floor would whiten
    /// garbage and the `Σ√λ/k` normalization has no meaning. Reduce `k` or
    /// gather more rows (the floor handles near-singular `N > k` postures).
    RankDeficient {
        /// Total row count across groups.
        rows: usize,
        /// The embedding dimension `k`.
        dim: usize,
    },
    /// The population spectrum collapsed (`λ_max ≤ 0` or non-finite) — every
    /// row is (numerically) identical, so no whitening exists.
    DegeneratePopulation {
        /// The largest population eigenvalue as computed.
        lambda_max: f64,
    },
}

impl core::fmt::Display for RhoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewGroups { got } => write!(
                f,
                "rho needs >= 2 groups (state-explained variance is undefined on single-view data); got {got}"
            ),
            Self::GroupTooSmall { group, len } => {
                write!(
                    f,
                    "group {group} has {len} member(s); rho needs >= 2 views per group"
                )
            }
            Self::DimMismatch {
                group,
                expected,
                got,
            } => write!(f, "group {group} has row width {got}, expected {expected}"),
            Self::NonFinite { group } => {
                write!(f, "group {group} carries a non-finite (NaN/inf) value")
            }
            Self::EmptyDimension => write!(f, "first row is empty: k = 0 has no eigensolve"),
            Self::RankDeficient { rows, dim } => write!(
                f,
                "{rows} rows < dim {dim}: population covariance is structurally rank-deficient; reduce k or gather more rows"
            ),
            Self::DegeneratePopulation { lambda_max } => write!(
                f,
                "population spectrum collapsed (lam_max = {lambda_max:e}); every row is numerically identical"
            ),
        }
    }
}

impl std::error::Error for RhoError {}

/// One ρ measurement — `Copy`, zero heap: the between-group eigenvalue
/// spectrum lives on the [`RhoScratch`] that produced the report
/// ([`RhoScratch::last_between_eigenvalues`]), not in the report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RhoReport {
    /// `ρ = tr(C_B^{1/2})/k` — the state-explained variance fraction, in
    /// `[0, 1]` up to finite-sample noise. Compression in the `G < k` regime
    /// (see the module docs) — compare ρ only within one (G, k) shape.
    pub rho: f64,
    /// The space dimension `k` the measurement ran at.
    pub k: usize,
    /// `Σᵢ max(λᵢ, 0)` — the post-clamp trace of the corrected between-group
    /// covariance. Raw (unclamped) eigenvalues, including the negatives that
    /// disclose correction over-subtraction, sit on the scratch.
    pub between_trace_corrected: f64,
    /// The pinned estimator + whitening conventions ([`RHO_CONVENTION`]) —
    /// print it beside every ρ number; ρ across differently-prepared spaces
    /// is incomparable.
    pub convention: &'static str,
}

/// Pre-allocated scratch for [`rho_fraction`], sized by dimension alone —
/// group count and group sizes need no capacity (the row pass streams groups
/// into fixed `k`/`k²` accumulators). Construct once via
/// [`RhoScratch::new`], [`RhoScratch::resize`] for a new dimension, reuse
/// across reports: steady-state calls allocate NOTHING (G4, CountingAllocator
/// gate). Holds the internal [`SymmetricEigScratch`] both eigensolves share.
pub struct RhoScratch {
    eig: SymmetricEigScratch,
    /// `k×k` staging: population covariance, then the corrected between
    /// covariance fed to the second decomposition.
    cov: Vec<f64>,
    /// `k×k` eigenvectors of the population covariance (whitening basis).
    vec_a: Vec<f64>,
    /// `k×k` eigenvector output of the between decomposition (unused beyond
    /// the solver contract; eigenvalues are the disclosure).
    vec_b: Vec<f64>,
    /// `k` eigenvalues — population pass first, then the between spectrum,
    /// sorted descending as computed (UNclamped; negatives disclose
    /// correction over-subtraction). Read via
    /// [`RhoScratch::last_between_eigenvalues`].
    vals: Vec<f64>,
    /// `k×k` ZCA whitening map `W = V Λ̃^{-1/2} Vᵀ`.
    whiten: Vec<f64>,
    /// `k×k` per-group whitened second-moment accumulator `Σ y yᵀ`.
    m2: Vec<f64>,
    /// `k×k` between accumulator `Σ_g (n_g/N) m̂_g m̂_gᵀ`.
    mgt_w: Vec<f64>,
    /// `k×k` correction accumulator `Σ_g scatter_g/(n_g(n_g−1))` (÷N applied
    /// when `C_B` is assembled).
    corr: Vec<f64>,
    /// `k` population mean `μ`.
    pop_mean: Vec<f64>,
    /// `k` grand-mean sum of whitened rows (÷N gives `m̄`).
    grand: Vec<f64>,
    /// `k` current group's whitened-row sum.
    group_sum: Vec<f64>,
    /// `k` centered-row staging (population pass).
    cen: Vec<f64>,
    /// `k` whitened-row staging (streaming pass).
    row: Vec<f64>,
    /// `k` current group's mean staging.
    mhat: Vec<f64>,
}

impl RhoScratch {
    /// Allocate every buffer for dimension `k`. All subsequent same-`k`
    /// [`rho_fraction`] calls reuse these buffers in place.
    pub fn new(k: usize) -> Self {
        let mut s = Self {
            eig: SymmetricEigScratch::new(),
            cov: Vec::new(),
            vec_a: Vec::new(),
            vec_b: Vec::new(),
            vals: Vec::new(),
            whiten: Vec::new(),
            m2: Vec::new(),
            mgt_w: Vec::new(),
            corr: Vec::new(),
            pop_mean: Vec::new(),
            grand: Vec::new(),
            group_sum: Vec::new(),
            cen: Vec::new(),
            row: Vec::new(),
            mhat: Vec::new(),
        };
        s.ensure(k);
        s
    }

    /// Grow the scratch to a new dimension. Grow-only — shrinking keeps the
    /// current capacity (allocation happens only when `k` increases).
    pub fn resize(&mut self, k: usize) {
        self.ensure(k);
    }

    /// The dimension this scratch is currently sized for.
    pub fn dim(&self) -> usize {
        self.pop_mean.len()
    }

    /// The between-group eigenvalue spectrum of the most recent successful
    /// call — **descending, unclamped** (negatives disclose how far the
    /// finite-group correction over-subtracted; ρ clamps them). Valid after
    /// a successful [`rho_fraction`]; all zeros before one.
    pub fn last_between_eigenvalues(&self) -> &[f64] {
        &self.vals
    }

    fn ensure(&mut self, k: usize) {
        let grow = |buf: &mut Vec<f64>, n: usize| {
            if buf.len() < n {
                buf.resize(n, 0.0);
            }
        };
        self.eig.ensure_capacity(k);
        grow(&mut self.cov, k * k);
        grow(&mut self.vec_a, k * k);
        grow(&mut self.vec_b, k * k);
        grow(&mut self.vals, k);
        grow(&mut self.whiten, k * k);
        grow(&mut self.m2, k * k);
        grow(&mut self.mgt_w, k * k);
        grow(&mut self.corr, k * k);
        grow(&mut self.pop_mean, k);
        grow(&mut self.grand, k);
        grow(&mut self.group_sum, k);
        grow(&mut self.cen, k);
        grow(&mut self.row, k);
        grow(&mut self.mhat, k);
    }
}

/// Measure ρ over grouped multi-view embeddings.
///
/// `groups[g]` is group `g`'s views as a flat row-major buffer; the row width
/// `k` is DECLARED BY THE SCRATCH (`RhoScratch::new(k)` — the
/// `AuditScratch::new(latent_dim)` contract: the scratch is the dimension
/// declaration, and each group's buffer length must be a whole number of
/// `k`-wide rows, loud [`RhoError::DimMismatch`] otherwise). Every group
/// needs ≥ 2 members and at least two groups are required, `N ≥ k` must
/// hold — every failure is a loud [`RhoError`], never a zero (see the
/// module docs). Zero-alloc in steady state: all work buffers are
/// scratch-owned.
pub fn rho_fraction(groups: &[&[f32]], scratch: &mut RhoScratch) -> Result<RhoReport, RhoError> {
    // ── validation (loud, cheapest first) ─────────────────────────────────
    let n_groups = groups.len();
    if n_groups < 2 {
        return Err(RhoError::TooFewGroups { got: n_groups });
    }
    let k = scratch.dim();
    if k == 0 {
        return Err(RhoError::EmptyDimension);
    }
    for (gi, g) in groups.iter().enumerate() {
        if !g.len().is_multiple_of(k) {
            return Err(RhoError::DimMismatch {
                group: gi,
                expected: k,
                got: g.len(),
            });
        }
        let members = g.len() / k;
        if members < 2 {
            return Err(RhoError::GroupTooSmall {
                group: gi,
                len: members,
            });
        }
        if g.iter().any(|v| !v.is_finite()) {
            return Err(RhoError::NonFinite { group: gi });
        }
    }
    let n_total: usize = groups.iter().map(|g| g.len() / k).sum();
    if n_total < k {
        return Err(RhoError::RankDeficient {
            rows: n_total,
            dim: k,
        });
    }
    scratch.ensure(k);

    // ── 1. population covariance (f64, 1/N) ───────────────────────────────
    let pop_mean = &mut scratch.pop_mean[..k];
    pop_mean.fill(0.0);
    {
        let cov = &mut scratch.cov[..k * k];
        cov.fill(0.0);
        let cen = &mut scratch.cen[..k];
        for g in groups {
            for row in g.chunks_exact(k) {
                for (m, x) in pop_mean.iter_mut().zip(row) {
                    *m += f64::from(*x);
                }
                for (c, x) in cen.iter_mut().zip(row) {
                    *c = f64::from(*x);
                }
                for i in 0..k {
                    let ci = cen[i];
                    for (cv, cj) in cov[i * k..i * k + k].iter_mut().zip(cen[..k].iter()) {
                        *cv += ci * cj;
                    }
                }
            }
        }
    }
    for m in pop_mean.iter_mut() {
        *m /= n_total as f64;
    }
    {
        let cov = &mut scratch.cov[..k * k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = cov[i * k + j] / n_total as f64 - pop_mean[i] * pop_mean[j];
            }
        }
    }

    // ── 2. whitening basis (population eigendecomposition) ────────────────
    {
        let vals = &mut scratch.vals[..k];
        let vec_a = &mut scratch.vec_a[..k * k];
        let cov = &scratch.cov[..k * k];
        let eig = &mut scratch.eig;
        symmetric_eig(vals, vec_a, cov, eig, k, EIG_MAX_ITERS);
    }
    let lam_max = scratch.vals[..k]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !lam_max.is_finite() || lam_max <= 0.0 {
        return Err(RhoError::DegeneratePopulation {
            lambda_max: lam_max,
        });
    }
    // ZCA map W = V Λ̃^{-1/2} Vᵀ with the pseudo-inverse floor. Fixed p-order
    // sequential fold — deterministic (Issue 940 posture).
    {
        let whiten = &mut scratch.whiten[..k * k];
        whiten.fill(0.0);
        let vec_a = &scratch.vec_a[..k * k];
        let vals = &scratch.vals[..k];
        let floor = EPS_REL * lam_max;
        for i in 0..k {
            for j in i..k {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += vec_a[i * k + p] * vec_a[j * k + p] / vals[p].max(floor).sqrt();
                }
                whiten[i * k + j] = acc;
                whiten[j * k + i] = acc;
            }
        }
    }

    // ── 3. streaming group pass: whitened rows → accumulators ─────────────
    let nf = n_total as f64;
    {
        let grand = &mut scratch.grand[..k];
        grand.fill(0.0);
    }
    {
        let m2 = &mut scratch.m2[..k * k];
        let mgt_w = &mut scratch.mgt_w[..k * k];
        let corr = &mut scratch.corr[..k * k];
        m2.fill(0.0);
        mgt_w.fill(0.0);
        corr.fill(0.0);
    }
    let whiten = &scratch.whiten[..k * k];
    let pop_mean = &scratch.pop_mean[..k];
    for g in groups {
        let ng = g.len() / k;
        {
            let group_sum = &mut scratch.group_sum[..k];
            group_sum.fill(0.0);
        }
        {
            let m2 = &mut scratch.m2[..k * k];
            m2.fill(0.0);
        }
        for row_in in g.chunks_exact(k) {
            {
                let cen = &mut scratch.cen[..k];
                for (c, (x, m)) in cen.iter_mut().zip(row_in.iter().zip(pop_mean.iter())) {
                    *c = f64::from(*x) - *m;
                }
            }
            {
                let cen = &scratch.cen[..k];
                let row = &mut scratch.row[..k];
                for (r, w) in row.iter_mut().zip(whiten.chunks_exact(k)) {
                    let mut acc = 0.0;
                    for (wv, cv) in w.iter().zip(cen.iter()) {
                        acc += *wv * *cv;
                    }
                    *r = acc;
                }
            }
            let row = &scratch.row[..k];
            {
                let group_sum = &mut scratch.group_sum[..k];
                let grand = &mut scratch.grand[..k];
                for (gs, (r, gv)) in group_sum.iter_mut().zip(row.iter().zip(grand.iter_mut())) {
                    *gs += *r;
                    *gv += *r;
                }
            }
            let m2 = &mut scratch.m2[..k * k];
            for i in 0..k {
                let ri = row[i];
                for (mv, rj) in m2[i * k..i * k + k].iter_mut().zip(row.iter()) {
                    *mv += ri * *rj;
                }
            }
        }
        // Per-group close: m̂_g, the (n_g/N)·m̂m̂ᵀ between term, and the
        // correction accumulator C_W,g = scatter/(n_g−1) via the identity
        // scatter = m2 − n_g·m̂m̂ᵀ (no second pass over the group's rows).
        // The brief's weight Σ_g(n_g/N)·C_W,g/n_g = Σ_g C_W,g/N — the n_g
        // cancels — so the 1/N lands ONCE when C_B is assembled.
        let inv = 1.0 / ((ng - 1) as f64);
        {
            let group_sum = &scratch.group_sum[..k];
            let mhat = &mut scratch.mhat[..k];
            for (mh, gs) in mhat.iter_mut().zip(group_sum.iter()) {
                *mh = gs / ng as f64;
            }
        }
        let mhat = &scratch.mhat[..k];
        let wgt = ng as f64 / nf;
        {
            let mgt_w = &mut scratch.mgt_w[..k * k];
            let corr = &mut scratch.corr[..k * k];
            let m2 = &scratch.m2[..k * k];
            for i in 0..k {
                for j in 0..k {
                    let outer = mhat[i] * mhat[j];
                    mgt_w[i * k + j] += wgt * outer;
                    corr[i * k + j] += (m2[i * k + j] - ng as f64 * outer) * inv;
                }
            }
        }
    }

    // ── 4. corrected between-group covariance ─────────────────────────────
    // C_B = Σ_g (n_g/N) m̂_g m̂_gᵀ − m̄ m̄ᵀ − (Σ_g scatter_g/(n_g(n_g−1)))/N
    // (the Σ(n_g/N)(m̂−m̄)(·)ᵀ identity turns the brief's per-group form into
    // these three accumulators; Σ_g(n_g/N)·C_W,g/n_g = corr/N exactly).
    let grand_mean = &mut scratch.pop_mean[..k];
    for (gm, gv) in grand_mean.iter_mut().zip(scratch.grand[..k].iter()) {
        *gm = gv / nf;
    }
    {
        let cov = &mut scratch.cov[..k * k];
        let mgt_w = &scratch.mgt_w[..k * k];
        let corr = &scratch.corr[..k * k];
        let mbar = &scratch.pop_mean[..k];
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] = mgt_w[i * k + j] - mbar[i] * mbar[j] - corr[i * k + j] / nf;
            }
        }
    }

    // ── 5. between-group spectrum + ρ ─────────────────────────────────────
    {
        let vals = &mut scratch.vals[..k];
        let vec_b = &mut scratch.vec_b[..k * k];
        let cov = &scratch.cov[..k * k];
        let eig = &mut scratch.eig;
        symmetric_eig(vals, vec_b, cov, eig, k, EIG_MAX_ITERS);
    }
    let mut trace_sqrt = 0.0;
    let mut trace_clamped = 0.0;
    for lam in scratch.vals[..k].iter() {
        let l = lam.max(0.0);
        trace_sqrt += l.sqrt();
        trace_clamped += l;
    }
    let rho = trace_sqrt / k as f64;
    scratch.vals[..k].sort_by(|a, b| b.total_cmp(a));

    Ok(RhoReport {
        rho,
        k,
        between_trace_corrected: trace_clamped,
        convention: RHO_CONVENTION,
    })
}

/// The Thm 3 alignment floor: the best-rotation error between two spaces of
/// the same state distribution is **at least** `2k(1 − min ρ)` — measured as
/// the mean squared error over the `k` dimensions per paired row, in BOTH
/// spaces' whitened coordinates (`E‖f_A − R·f_B‖²` over rotations `R`).
/// Consumers compare a post-Procrustes RMS² against this, never raw cosine
/// (Prop 2(i): no unpaired second-moment statistic CERTIFIES alignment).
///
/// ρ is invariant across whitening *choices* but ρ values across
/// differently-whitened spaces are incomparable — both spaces must report
/// [`RHO_CONVENTION`] before the floor means anything.
pub fn alignment_floor(rho_a: f64, rho_b: f64, k: usize) -> f64 {
    2.0 * k as f64 * (1.0 - rho_a.min(rho_b))
}

// Plain `cfg(test)`: the enclosing module already carries the
// `nuisance_fraction` feature gate — re-stating it here is what would let
// the two drift apart (the alloc-module convention).
#[cfg(test)]
mod tests {
    use super::*;

    // ── fixture helpers ────────────────────────────────────────────────────

    /// Gaussian via the 12-uniform CLT (`data_probe::geometry` precedent):
    /// mean 0, variance 1, range ±6.
    fn gauss(rng: &mut fastrand::Rng) -> f32 {
        let sum: f32 = (0..12).map(|_| rng.f32()).sum();
        sum - 6.0
    }

    /// Planted fixture, CORRECT form: groups share the signal `s`, per-view
    /// noise `n` is independent — `f = ρ·s + √(1−ρ²)·n` with
    /// `Cov(s) = Cov(n) = I` gives `Cov(E[f|z]) = ρ²·I`, so ρ reads the
    /// planted value (in the G ≫ k regime). Each group is a flat
    /// `views × k` row-major buffer.
    fn planted(
        rho: f64,
        k: usize,
        n_groups: usize,
        views: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<Vec<f32>> {
        let hi = (1.0 - rho * rho).sqrt();
        let mut out = Vec::with_capacity(n_groups);
        for _ in 0..n_groups {
            let s: Vec<f64> = (0..k).map(|_| f64::from(gauss(rng))).collect();
            let mut flat = Vec::with_capacity(views * k);
            for _ in 0..views {
                for sc in &s {
                    let n = f64::from(gauss(rng));
                    flat.push((rho * *sc + hi * n) as f32);
                }
            }
            out.push(flat);
        }
        out
    }

    /// The WRONG planting `f = √ρ·s + √(1−ρ)·n`: `C_f = ρ·I`, so
    /// `tr(C_f^{1/2})/k = √ρ` — planted 0.5 reads 0.707. The tripwire test
    /// pins this so the convention sign can never silently flip.
    fn planted_naive(
        rho: f64,
        k: usize,
        n_groups: usize,
        views: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<Vec<f32>> {
        let sa = rho.sqrt();
        let sb = (1.0 - rho).sqrt();
        let mut out = Vec::with_capacity(n_groups);
        for _ in 0..n_groups {
            let s: Vec<f64> = (0..k).map(|_| f64::from(gauss(rng))).collect();
            let mut flat = Vec::with_capacity(views * k);
            for _ in 0..views {
                for sc in &s {
                    let n = f64::from(gauss(rng));
                    flat.push((sa * *sc + sb * n) as f32);
                }
            }
            out.push(flat);
        }
        out
    }

    fn refs(v: &[Vec<f32>]) -> Vec<&[f32]> {
        v.iter().map(|g| g.as_slice()).collect()
    }

    /// Fisher-Yates the ROWS across all groups, then re-split by the
    /// original group sizes — destroys the group/nuisance relation while
    /// keeping the population covariance identical.
    fn shuffle_rows(groups: &[Vec<f32>], k: usize, rng: &mut fastrand::Rng) -> Vec<Vec<f32>> {
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for g in groups {
            for r in g.chunks_exact(k) {
                rows.push(r.to_vec());
            }
        }
        for i in (1..rows.len()).rev() {
            let j = rng.usize(..=i);
            rows.swap(i, j);
        }
        let mut out = Vec::with_capacity(groups.len());
        let mut off = 0usize;
        for g in groups {
            let ng = g.len() / k;
            let mut flat = Vec::with_capacity(ng * k);
            for r in &rows[off..off + ng] {
                flat.extend_from_slice(r);
            }
            out.push(flat);
            off += ng;
        }
        out
    }

    /// Random orthogonal k×k matrix (row-major) via Gram–Schmidt over a
    /// seeded Gaussian batch.
    fn orthogonal(k: usize, rng: &mut fastrand::Rng) -> Vec<f32> {
        let mut q: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..k).map(|_| gauss(rng)).collect())
            .collect();
        for i in 0..k {
            for j in 0..i {
                let dot: f32 = (0..k).map(|c| q[i][c] * q[j][c]).sum();
                let (left, right) = q.split_at_mut(i);
                let qj = &left[j];
                for (qc, qjc) in right[0].iter_mut().zip(qj.iter()) {
                    *qc -= dot * *qjc;
                }
            }
            let norm = (0..k).map(|c| q[i][c] * q[i][c]).sum::<f32>().sqrt();
            for qc in q[i].iter_mut() {
                *qc /= norm;
            }
        }
        q.iter().flatten().copied().collect()
    }

    /// `y = x·Q` for every row (k×k row-major Q).
    fn rotate(groups: &[Vec<f32>], q: &[f32], k: usize) -> Vec<Vec<f32>> {
        groups
            .iter()
            .map(|g| {
                g.chunks_exact(k)
                    .flat_map(|row| {
                        (0..k).map(move |p| (0..k).map(|c| q[p * k + c] * row[c]).sum::<f32>())
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }

    /// Test-local whitening (the same 1/N population covariance + ZCA map the
    /// primitive uses, built on [`crate::linalg::symmetric_eig`]) — the bound
    /// arm needs the whitened ROWS, which the zero-alloc API deliberately
    /// does not expose.
    fn whiten_rows(rows: &[f64], n: usize, k: usize) -> Vec<f64> {
        let mut mean = vec![0.0f64; k];
        for row in rows.chunks_exact(k) {
            for (m, x) in mean.iter_mut().zip(row) {
                *m += *x;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        let mut cov = vec![0.0f64; k * k];
        let mut cen = vec![0.0f64; k];
        for row in rows.chunks_exact(k) {
            for (c, (x, m)) in cen.iter_mut().zip(row.iter().zip(mean.iter())) {
                *c = *x - *m;
            }
            for i in 0..k {
                let ci = cen[i];
                for (cv, cj) in cov[i * k..i * k + k].iter_mut().zip(cen.iter()) {
                    *cv += ci * cj;
                }
            }
        }
        for v in cov.iter_mut() {
            *v /= n as f64;
        }
        let mut vals = vec![0.0f64; k];
        let mut vecs = vec![0.0f64; k * k];
        let mut eig = crate::linalg::SymmetricEigScratch::new();
        crate::linalg::symmetric_eig(&mut vals, &mut vecs, &cov, &mut eig, k, 60);
        let lam_max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let floor = 1e-10 * lam_max;
        let mut w = vec![0.0f64; k * k];
        for i in 0..k {
            for j in i..k {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += vecs[i * k + p] * vecs[j * k + p] / vals[p].max(floor).sqrt();
                }
                w[i * k + j] = acc;
                w[j * k + i] = acc;
            }
        }
        let mut out = vec![0.0f64; n * k];
        for (r, row) in rows.chunks_exact(k).enumerate() {
            for (c, (x, m)) in cen.iter_mut().zip(row.iter().zip(mean.iter())) {
                *c = *x - *m;
            }
            for p in 0..k {
                let mut acc = 0.0;
                for (wv, cv) in w[p * k..p * k + k].iter().zip(cen.iter()) {
                    acc += *wv * *cv;
                }
                out[r * k + p] = acc;
            }
        }
        out
    }

    /// Test-local orthogonal Procrustes WITHOUT new deps: the polar factor of
    /// `M = BᵀA` via a [`symmetric_eig`]-based SVD (`MᵀM = VΣ²Vᵀ`,
    /// `R = M·(MᵀM)^{-1/2} = polar(M)`) — leaf law, no katgpt-spectral. Built
    /// in the convention-INVARIANT element-product form of the shipped
    /// `leakage_probe::procrustes_polar_into` (`T = V·diag(λ̃^-1/2)·Vᵀ` from
    /// `eigvecs[i*k+p]·eigvecs[j*k+p]` products, then `R = M·T`): the U·Vᵀ
    /// column assembly reads columns vs rows of the solver's stored basis,
    /// and both readings of that convention produced an uncorrelated R
    /// (per-row resid ≈ ‖a‖²+‖b‖² at every planted ρ) before this form was
    /// adopted. Objective pinned: Σᵢ‖R·aᵢ − bᵢ‖² = const − 2⟨R, BᵀA⟩_F, so
    /// the minimizer is exactly R = polar(BᵀA) (von Neumann trace bound).
    fn polar_procrustes(a: &[f64], b: &[f64], n: usize, k: usize) -> Vec<f64> {
        // M = BᵀA: M[p][q] = Σ_i b_i[p]·a_i[q] (b indexed by the ROW p).
        let mut m = vec![0.0f64; k * k];
        for i in 0..n {
            let ar = &a[i * k..(i + 1) * k];
            let br = &b[i * k..(i + 1) * k];
            for p in 0..k {
                let bpv = br[p];
                for (mq, aqv) in m[p * k..p * k + k].iter_mut().zip(ar.iter()) {
                    *mq += bpv * *aqv;
                }
            }
        }
        // S = MᵀM (k×k symmetric).
        let mut mtm = vec![0.0f64; k * k];
        for p in 0..k {
            for q in 0..k {
                mtm[p * k + q] = (0..k).map(|t| m[t * k + p] * m[t * k + q]).sum::<f64>();
            }
        }
        let mut lam = vec![0.0f64; k];
        let mut vecs = vec![0.0f64; k * k];
        let mut eig = crate::linalg::SymmetricEigScratch::new();
        crate::linalg::symmetric_eig(&mut lam, &mut vecs, &mtm, &mut eig, k, 60);
        // T = V·diag(1/√λ̃)·Vᵀ — the element-product form. Floor λ̃ at
        // 1e-8·λ_max: under-estimating only shrinks the corresponding
        // direction, never amplifies.
        let lam_max = lam.iter().copied().fold(0.0f64, f64::max);
        let lam_floor = lam_max * 1e-8;
        let mut t = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += vecs[i * k + p] * vecs[j * k + p] / lam[p].max(lam_floor).sqrt();
                }
                t[i * k + j] = acc;
            }
        }
        // R = M·T (the left-acting minimizer of Σ_i ‖R·a_i − b_i‖²).
        let mut r = vec![0.0f64; k * k];
        for p in 0..k {
            for q in 0..k {
                r[p * k + q] = (0..k).map(|t2| m[p * k + t2] * t[t2 * k + q]).sum::<f64>();
            }
        }
        r
    }

    /// The UNCORRECTED form — whitened between scatter with NO finite-group
    /// correction — computed locally for the shuffled-control pin. Never a
    /// second public API.
    fn uncorrected_rho(groups: &[Vec<f32>], k: usize) -> f64 {
        let n_total: usize = groups.iter().map(|g| g.len() / k).sum();
        let mut rows = Vec::with_capacity(n_total * k);
        for g in groups {
            rows.extend(g.iter().map(|x| f64::from(*x)));
        }
        let white = whiten_rows(&rows, n_total, k);
        let mut mbar = vec![0.0f64; k];
        for row in white.chunks_exact(k) {
            for (m, x) in mbar.iter_mut().zip(row) {
                *m += *x;
            }
        }
        for m in mbar.iter_mut() {
            *m /= n_total as f64;
        }
        let mut sb = vec![0.0f64; k * k];
        let mut mg = vec![0.0f64; k];
        let mut off = 0usize;
        for g in groups {
            let ng = g.len() / k;
            mg.iter_mut().for_each(|m| *m = 0.0);
            for row in white[off..off + ng * k].chunks_exact(k) {
                for (m, x) in mg.iter_mut().zip(row) {
                    *m += *x;
                }
            }
            off += ng * k;
            for m in mg.iter_mut() {
                *m /= ng as f64;
            }
            let wgt = ng as f64 / n_total as f64;
            for i in 0..k {
                for j in 0..k {
                    sb[i * k + j] += wgt * (mg[i] - mbar[i]) * (mg[j] - mbar[j]);
                }
            }
        }
        let mut vals = vec![0.0f64; k];
        let mut vecs = vec![0.0f64; k * k];
        let mut eig = crate::linalg::SymmetricEigScratch::new();
        crate::linalg::symmetric_eig(&mut vals, &mut vecs, &sb, &mut eig, k, 60);
        vals.iter().map(|l| l.max(0.0).sqrt()).sum::<f64>() / k as f64
    }

    // ── the arms ───────────────────────────────────────────────────────────

    const K: usize = 16;
    /// G ≫ k: the paper's regime — the recovery/tripwire/shuffled arms ride
    /// this shape because at G < k the Σ√λ/k normalization compresses ρ
    /// (see the module docs + the regime arm below).
    const G: usize = 64;
    const VIEWS: usize = 16;
    const SEEDS: u64 = 5;
    const BOUND_G: usize = 64;
    const BOUND_VIEWS: usize = 4;

    /// G1 — planted-ρ recovery at G ≫ k. Measured on the numpy oracle
    /// (7 seeds): |bias| ≤ 0.037, seed std ≤ 0.008 → tol 0.07 ≥ 3× the
    /// observed seed spread at every level. Monotone on the means.
    #[test]
    fn planted_rho_recovers_across_levels() {
        const TOL: f64 = 0.07;
        const LEVELS: [f64; 4] = [0.2, 0.5, 0.8, 0.95];
        let mut means: Vec<f64> = Vec::with_capacity(LEVELS.len());
        for (li, &rho) in LEVELS.iter().enumerate() {
            let mut acc = 0.0;
            for seed in 0..SEEDS {
                let mut rng = fastrand::Rng::with_seed(0x9780 + (li as u64) * 100 + seed);
                let groups = planted(rho, K, G, VIEWS, &mut rng);
                let rep = rho_fraction(&refs(&groups), &mut RhoScratch::new(K)).unwrap();
                assert_eq!(rep.k, K);
                assert_eq!(rep.convention, RHO_CONVENTION);
                acc += rep.rho;
            }
            let mean = acc / SEEDS as f64;
            assert!(
                (mean - rho).abs() <= TOL,
                "planted rho={rho} read mean {mean:.4} (tol {TOL}; G={G} >> k={K}, {SEEDS} seeds)"
            );
            means.push(mean);
        }
        for w in means.windows(2) {
            assert!(
                w[0] < w[1],
                "measured rho must be strictly monotone across planted levels: {means:?}"
            );
        }
    }

    /// The brief's literal fixture shape (k=16, 8 groups × 4 views): G−1 = 7
    /// < k, so the between scatter is rank-deficient in k and ρ reads LOW
    /// (oracle: planted 0.95 reads ≈0.42 — monotone but compressed). Pin
    /// monotonicity + the compression band so the regime boundary can never
    /// move silently.
    #[test]
    fn regime_arm_g_below_k_compresses_but_stays_monotone() {
        const LEVELS: [f64; 4] = [0.2, 0.5, 0.8, 0.95];
        let mut means: Vec<f64> = Vec::with_capacity(LEVELS.len());
        for (li, &rho) in LEVELS.iter().enumerate() {
            let mut acc = 0.0;
            for seed in 0..3u64 {
                let mut rng = fastrand::Rng::with_seed(0x9781 + (li as u64) * 100 + seed);
                let groups = planted(rho, K, 8, 4, &mut rng);
                acc += rho_fraction(&refs(&groups), &mut RhoScratch::new(K))
                    .unwrap()
                    .rho;
            }
            let mean = acc / 3.0;
            assert!(
                (0.0..=1.0).contains(&mean),
                "rho out of [0,1] in the G<k regime: {mean}"
            );
            means.push(mean);
        }
        for w in means.windows(2) {
            assert!(
                w[0] < w[1],
                "G<k regime must stay strictly monotone: {means:?}"
            );
        }
        assert!(
            means[3] < 0.6,
            "planted 0.95 must read compressed below 0.6 at G=8 < k=16 (read {:.4})",
            means[3]
        );
        assert!(
            means[3] > 0.25,
            "compression must not collapse to noise (read {:.4})",
            means[3]
        );
    }

    /// The mandatory tripwire: the WRONG planting `√ρ·s + √(1−ρ)·n` must read
    /// ≈ √ρ (oracle: 0.6707 ± 0.006 vs 0.7071, finite-sample concavity at
    /// n_g = 16 — tol 0.06 ≥ 3× spread), AND stay separated from the correct
    /// form's reading (0.465) by ≥ 0.15 — the convention sign is pinned.
    #[test]
    fn naive_planting_tripwire_reads_sqrt_rho() {
        const TOL: f64 = 0.06;
        let sqrt_half = 0.5f64.sqrt();
        let mut naive_sum = 0.0;
        let mut correct_sum = 0.0;
        for seed in 0..SEEDS {
            let mut rng = fastrand::Rng::with_seed(0x9782 + seed);
            let naive = planted_naive(0.5, K, G, VIEWS, &mut rng);
            naive_sum += rho_fraction(&refs(&naive), &mut RhoScratch::new(K))
                .unwrap()
                .rho;
            let mut rng = fastrand::Rng::with_seed(0x9783 + seed);
            let correct = planted(0.5, K, G, VIEWS, &mut rng);
            correct_sum += rho_fraction(&refs(&correct), &mut RhoScratch::new(K))
                .unwrap()
                .rho;
        }
        let naive = naive_sum / SEEDS as f64;
        let correct = correct_sum / SEEDS as f64;
        assert!(
            (naive - sqrt_half).abs() <= TOL,
            "the naive planting must read ~sqrt(rho)={sqrt_half:.4}, read {naive:.4} — the convention sign flipped"
        );
        assert!(
            naive - correct >= 0.15,
            "tripwire separation lost: naive {naive:.4} vs correct-form {correct:.4}"
        );
    }

    /// Shuffled-group control: permuting rows across group labels collapses
    /// the corrected ρ toward the noise floor, while the UNCORRECTED form
    /// (computed locally, never a second public API) stays high — pinning
    /// exactly why the finite-group correction exists. Oracle: corrected
    /// 0.074 mean / 0.083 max (brief bar ≤ 0.1), uncorrected 0.242 mean.
    #[test]
    fn shuffled_groups_collapse_corrected_not_uncorrected() {
        const SHUF_SEEDS: u64 = 5;
        let mut corr_sum = 0.0;
        let mut unc_sum = 0.0;
        for seed in 0..SHUF_SEEDS {
            let mut rng = fastrand::Rng::with_seed(0x9784 + seed);
            let groups = planted(0.5, K, G, VIEWS, &mut rng);
            let sh = shuffle_rows(&groups, K, &mut rng);
            let corrected = rho_fraction(&refs(&sh), &mut RhoScratch::new(K))
                .unwrap()
                .rho;
            let uncorrected = uncorrected_rho(&sh, K);
            corr_sum += corrected;
            unc_sum += uncorrected;
            assert!(
                uncorrected > 2.0 * corrected,
                "seed {seed}: uncorrected {uncorrected:.4} must exceed 2x corrected {corrected:.4}"
            );
        }
        let corr = corr_sum / SHUF_SEEDS as f64;
        let unc = unc_sum / SHUF_SEEDS as f64;
        assert!(
            corr <= 0.1,
            "shuffled-group corrected rho must collapse to <= 0.1 (brief bar), read {corr:.4}"
        );
        assert!(
            unc >= 0.2,
            "the UNCORRECTED form must NOT collapse on shuffled labels (read {unc:.4}) — pinning why the correction exists"
        );
    }

    /// One space = a rotation of the other ⇒ identical ρ (the whole pipeline
    /// is similarity-invariant; eigenvalues are preserved exactly, the
    /// residual is solver noise). Oracle: max |Δρ| = 2.6e-9.
    #[test]
    fn rotated_copy_is_invariant() {
        for seed in 0..3u64 {
            let mut rng = fastrand::Rng::with_seed(0x9785 + seed);
            let groups = planted(0.6, K, 8, 4, &mut rng);
            let q = orthogonal(K, &mut rng);
            let rot = rotate(&groups, &q, K);
            let r1 = rho_fraction(&refs(&groups), &mut RhoScratch::new(K))
                .unwrap()
                .rho;
            let r2 = rho_fraction(&refs(&rot), &mut RhoScratch::new(K))
                .unwrap()
                .rho;
            assert!(
                (r1 - r2).abs() <= 1e-6,
                "seed {seed}: rho(X)={r1:.9} vs rho(X·Q)={r2:.9}"
            );
        }
    }

    /// Bound arm: two INDEPENDENT views sharing an ANISOTROPIC signal — the
    /// post-Procrustes whitened RMS² must respect 2k(1 − min ρ̂) at every
    /// planted level (slack 0.2), the floor must stay positive, and the
    /// residual must sit below 2k so the bound is not vacuous. Procrustes =
    /// the polar factor of AᵀB via symmetric_eig — no new deps, no
    /// katgpt-spectral (leaf law).
    ///
    /// ⛔ The signal MUST be anisotropic (fixed per-dim spread D, same both
    /// spaces): with s ~ N(0, I) the whitened cross-covariance collapses to
    /// ρ²n·I — degenerate singular vectors, the computed rotation captures
    /// none of the shared signal (resid ≈ 2k, was 1186-per-row in the first
    /// draft) and the theorem's bound is about the OPTIMAL rotation, which
    /// needs a well-posed Procrustes. With M ≈ nρ²D²(ρ²D²+(1−ρ²)I)⁻² —
    /// diagonal, distinct values — R* is unique; analytic margins at planted
    /// {0.2, 0.5, 0.8}: resid/floor·(1−slack) ≈ 1.53 / 1.84 / 1.38.
    ///
    /// ⚠ APPLICABILITY (consumer feedback, Issue 978 T5 / riir-reflex Issue
    /// 106, measured 2026-11-04): this fixture's noises are drawn
    /// INDEPENDENTLY per view — the regime where the floor is a true bound.
    /// Real SAME-INPUT cross-embedder pairs (both spaces functions of the
    /// same x) are the CORRELATED-NUISANCE regime: the x-level shared signal
    /// is invisible to class-ρ yet alignable by the rotation, and the
    /// measured post-RMS² landed 46–74% BELOW the class-grouped floor on all
    /// three reflex suites. The grouping that would bound such a transport
    /// (state = the input itself) is single-view per space, where the
    /// primitive refuses loud — the correct behavior, per the single-view
    /// law. A class-grouped floor must therefore NOT be sold as a gate on
    /// paired cross-embedder transports; the gate use is consumer-shape-bound
    /// (the refine lane's companion negative: 8-D transports are too cheap
    /// to gate at all).
    #[test]
    fn bound_arm_respects_alignment_floor() {
        const SLACK: f64 = 0.2;
        const LEVELS: [f64; 3] = [0.2, 0.5, 0.8];
        // Fixed anisotropic signal spectrum, identical for both spaces.
        let spread: Vec<f64> = (0..K)
            .map(|d| 0.5 + 1.5 * d as f64 / (K - 1) as f64)
            .collect();
        for (li, &rho) in LEVELS.iter().enumerate() {
            for seed in 0..3u64 {
                let mut rng = fastrand::Rng::with_seed(0x978A + (li as u64) * 10 + seed);
                let hi = (1.0 - rho * rho).sqrt();
                let mut ga: Vec<Vec<f32>> = Vec::with_capacity(BOUND_G);
                let mut gb: Vec<Vec<f32>> = Vec::with_capacity(BOUND_G);
                for _ in 0..BOUND_G {
                    let s: Vec<f64> = (0..K)
                        .map(|d| spread[d] * f64::from(gauss(&mut rng)))
                        .collect();
                    let mut fa = Vec::with_capacity(BOUND_VIEWS * K);
                    let mut fb = Vec::with_capacity(BOUND_VIEWS * K);
                    for _ in 0..BOUND_VIEWS {
                        for sc in &s {
                            fa.push((rho * *sc + hi * f64::from(gauss(&mut rng))) as f32);
                            fb.push((rho * *sc + hi * f64::from(gauss(&mut rng))) as f32);
                        }
                    }
                    ga.push(fa);
                    gb.push(fb);
                }
                let ra = rho_fraction(&refs(&ga), &mut RhoScratch::new(K))
                    .unwrap()
                    .rho;
                let rb = rho_fraction(&refs(&gb), &mut RhoScratch::new(K))
                    .unwrap()
                    .rho;
                let n = BOUND_G * BOUND_VIEWS;
                let mut flat_a = Vec::with_capacity(n * K);
                let mut flat_b = Vec::with_capacity(n * K);
                for (g_a, g_b) in ga.iter().zip(gb.iter()) {
                    flat_a.extend(g_a.iter().map(|x| f64::from(*x)));
                    flat_b.extend(g_b.iter().map(|x| f64::from(*x)));
                }
                let wa = whiten_rows(&flat_a, n, K);
                let wb = whiten_rows(&flat_b, n, K);
                let r = polar_procrustes(&wa, &wb, n, K);
                let mut resid = 0.0f64;
                for i in 0..n {
                    let a = &wa[i * K..(i + 1) * K];
                    let b = &wb[i * K..(i + 1) * K];
                    for p in 0..K {
                        let mut rap = 0.0;
                        for (rq, av) in r[p * K..p * K + K].iter().zip(a.iter()) {
                            rap += *rq * *av;
                        }
                        resid += (rap - b[p]) * (rap - b[p]);
                    }
                }
                let resid = resid / n as f64;
                let floor = alignment_floor(ra, rb, K);
                assert!(
                    floor > 0.0,
                    "floor must be positive at planted rho < 1 (rho={rho})"
                );
                // The brief's non-vacuity pin is "resid < 2k ... at high ρ":
                // at ρ=0.2 the population residual 2k(1−ρ²)=30.7 sits within
                // sampling noise of 2k=32, so the strict bound there is a
                // seed coin-flip — apply it where vacuity can actually bite.
                if rho >= 0.5 {
                    assert!(
                        resid < 2.0 * K as f64,
                        "bound vacuous at rho={rho}: resid {resid:.2} >= 2k"
                    );
                }
                assert!(
                    resid >= floor * (1.0 - SLACK),
                    "Thm3 floor violated at rho={rho} seed={seed}: resid {resid:.2} < floor {floor:.2}*(1-{SLACK})"
                );
            }
        }
    }

    /// Loud refusals on every degenerate shape — ρ is undefined, never 0.
    #[test]
    fn degenerate_inputs_refuse_loud() {
        // no groups / single group
        match rho_fraction(&[], &mut RhoScratch::new(K)) {
            Err(RhoError::TooFewGroups { got: 0 }) => {}
            other => panic!("expected TooFewGroups{{got:0}}, got {other:?}"),
        }
        let single = vec![vec![0.1f32; 4 * K]];
        match rho_fraction(&refs(&single), &mut RhoScratch::new(K)) {
            Err(RhoError::TooFewGroups { got: 1 }) => {}
            other => panic!("expected TooFewGroups{{got:1}}, got {other:?}"),
        }
        // one-member group (buffer length k = 1 row)
        let g0 = vec![0.5f32; K];
        let g1 = vec![0.1f32; 2 * K];
        match rho_fraction(&[g0.as_slice(), g1.as_slice()], &mut RhoScratch::new(K)) {
            Err(RhoError::GroupTooSmall { group: 0, len: 1 }) => {}
            other => panic!("expected GroupTooSmall{{group:0,len:1}}, got {other:?}"),
        }
        // ragged group: buffer length not a whole number of rows
        let g0 = vec![0.5f32; 2 * K];
        let g1 = vec![0.1f32; 2 * K + 1];
        match rho_fraction(&[g0.as_slice(), g1.as_slice()], &mut RhoScratch::new(K)) {
            Err(RhoError::DimMismatch {
                group: 1,
                expected: 16,
                got: 33,
            }) => {}
            other => panic!("expected DimMismatch{{group:1}}, got {other:?}"),
        }
        // non-finite input
        let mut g1 = vec![0.1f32; 2 * K];
        g1[5] = f32::NAN;
        let g0 = vec![0.5f32; 2 * K];
        match rho_fraction(&[g0.as_slice(), g1.as_slice()], &mut RhoScratch::new(K)) {
            Err(RhoError::NonFinite { group: 1 }) => {}
            other => panic!("expected NonFinite{{group:1}}, got {other:?}"),
        }
        // N < k: structurally rank-deficient population
        let g0 = vec![0.3f32; 2 * K];
        let g1 = vec![0.7f32; 2 * K];
        match rho_fraction(&[g0.as_slice(), g1.as_slice()], &mut RhoScratch::new(K)) {
            Err(RhoError::RankDeficient { rows: 4, dim: K }) => {}
            other => panic!("expected RankDeficient{{rows:4,dim:{K}}}, got {other:?}"),
        }
        // numerically identical population: no whitening exists (needs
        // N >= k to clear the rank guard first)
        let g0 = vec![1.0f32; 8 * K];
        let g1 = vec![1.0f32; 8 * K];
        match rho_fraction(&[g0.as_slice(), g1.as_slice()], &mut RhoScratch::new(K)) {
            Err(RhoError::DegeneratePopulation { lambda_max }) => {
                assert_eq!(lambda_max, 0.0);
            }
            other => panic!("expected DegeneratePopulation, got {other:?}"),
        }
        // empty first row: k = 0
        let g0: Vec<f32> = Vec::new();
        let g1: Vec<f32> = Vec::new();
        match rho_fraction(&[g0.as_slice(), g1.as_slice()], &mut RhoScratch::new(0)) {
            Err(RhoError::EmptyDimension) => {}
            other => panic!("expected EmptyDimension, got {other:?}"),
        }
    }

    /// Same input through fresh scratches → bit-identical report + spectrum
    /// (Issue 940 sequential-f64 posture; deterministic by construction).
    #[test]
    fn determinism_bit_identical_replay() {
        let mut rng = fastrand::Rng::with_seed(0x978D);
        let groups = planted(0.6, K, 8, 4, &mut rng);
        let r = refs(&groups);
        let mut s1 = RhoScratch::new(K);
        let rep1 = rho_fraction(&r, &mut s1).unwrap();
        let eig1: Vec<u64> = s1
            .last_between_eigenvalues()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut s2 = RhoScratch::new(K);
        let rep2 = rho_fraction(&r, &mut s2).unwrap();
        let eig2: Vec<u64> = s2
            .last_between_eigenvalues()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(rep1.rho.to_bits(), rep2.rho.to_bits());
        assert_eq!(
            rep1.between_trace_corrected.to_bits(),
            rep2.between_trace_corrected.to_bits()
        );
        assert_eq!(eig1, eig2);
        assert_eq!(rep1.convention, RHO_CONVENTION);
    }

    /// G2 — report latency at d=256. The issue's ~10 µs bar was mis-estimated
    /// for the dense-f64 posture (the report is Ω(N·k² + k³): two f64
    /// eigendecompositions + population/within streaming; at the named
    /// posture that is tens of ms, not µs — the abandoned first draft of this
    /// issue measured the same conclusion). Also N = k is rank-deficient
    /// (rank ≤ N−1), so the primary cell rides N = 512 and the brief's
    /// literal N = 256 cell is timed floored. Asserts a 300 ms regression
    /// tripwire (~2.6× the 2026-11-03 measured medians: 117.5 ms at N=512,
    /// 134.5 ms at N=256, M3 release, AC); measured medians are printed for
    /// the issue record.
    #[ignore]
    #[test]
    #[cfg(not(debug_assertions))] // release-only posture; never a dev-profile assert
    fn g2_report_latency_at_d256() {
        let cells = [
            (
                256usize,
                16usize,
                32usize,
                0x978Eu64,
                "d=256 N=512 (G=16x32)",
            ),
            (256, 8, 32, 0x978F, "d=256 N=256 (G=8x32, N=k floored)"),
        ];
        for (d, g, views, seed, label) in cells {
            let mut rng = fastrand::Rng::with_seed(seed);
            let groups = planted(0.5, d, g, views, &mut rng);
            let r = refs(&groups);
            let mut scratch = RhoScratch::new(d);
            let _ = rho_fraction(&r, &mut scratch).unwrap(); // warmup
            let reps = 10usize;
            let mut times: Vec<u128> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = std::time::Instant::now();
                let rep = rho_fraction(&r, &mut scratch).unwrap();
                times.push(t.elapsed().as_nanos());
                assert!(rep.rho.is_finite());
            }
            times.sort();
            let median = times[reps / 2];
            println!(
                "g2 {label}: median {median} ns ({:.1} ms)",
                median as f64 / 1e6
            );
            assert!(
                median <= 300_000_000,
                "g2 {label}: {median} ns exceeds the 300 ms regression tripwire"
            );
        }
    }

    /// G4 — zero steady-state allocation (CountingAllocator sentinel; the
    /// tracking machinery exists exactly when
    /// `any(debug_assertions, feature = "alloc_tracking")` per alloc.rs).
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    #[test]
    fn g4_zero_alloc_steady_state() {
        use crate::alloc::{get_alloc_stats, reset_alloc_stats};
        let mut rng = fastrand::Rng::with_seed(0x9790);
        let groups = planted(0.5, K, 8, 4, &mut rng);
        let r = refs(&groups);
        let mut scratch = RhoScratch::new(K);
        let _ = rho_fraction(&r, &mut scratch).unwrap(); // warm (pre-sized; defensive)
        // Sentinel: prove the TrackingAllocator is installed before measuring.
        reset_alloc_stats();
        let _sentinel: Vec<u8> = vec![0u8; 256];
        let (sent_count, _) = get_alloc_stats();
        if sent_count == 0 {
            eprintln!("g4_zero_alloc_steady_state: TrackingAllocator not installed — SKIPPED");
            return;
        }
        drop(_sentinel);
        reset_alloc_stats();
        for _ in 0..100 {
            let rep = rho_fraction(&r, &mut scratch).unwrap();
            assert!(rep.rho.is_finite());
        }
        let (count, bytes) = get_alloc_stats();
        assert_eq!(
            count, 0,
            "steady-state rho_fraction allocated {count} time(s) ({bytes} bytes)"
        );
    }
}
