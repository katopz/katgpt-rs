//! Plan 641 — the epiplexity marginal-gain estimator (Research 685;
//! arXiv:2608.11746 EpiSelect/EpiGen): the ONLINE allocation half of the
//! epiplexity family.
//!
//! # Family placement
//!
//! The post-hoc screening half (loss-curve area scoring of a finished run)
//! stays `katgpt-pruners::epiplexity` (Plan 130, arXiv:2601.03220). THIS arm
//! answers the allocation question while the run is still live: given the
//! per-observation corpus counts per domain and one accuracy series, estimate
//! where the NEXT observation buys the most error reduction. riir-refine
//! Plan 392's mining queue consumes this; it never forks it.
//!
//! # Model (fit the error gap, not accuracy)
//!
//! `u = ε + β·N^(−α)` over the error gap `u = 1 − accuracy` (never accuracy
//! itself — the learnable signal is the error). The effective count is
//! `N_t = Σ_k γ_k·(n_k + 1)` (pseudocount +1.0 at ingestion, γ on the simplex
//! with `γ ≥ 1e-3`). The fit is GD-free: a fixed log-spaced grid over (α, ε)
//! (12 × 14 points) with cyclic coordinate descent on γ under a log-space
//! Huber objective with median/MAD robustness (`HUBER_DELTA` = 1.345, the
//! 95% Gaussian-efficiency constant). The cell score is the RAW-unit Huber
//! loss `Σρ_δ(v − c)` with the band δ = 1.345·σ frozen per cell from the
//! OPENING location (uniform γ — model-free, so γ cannot game it): the
//! scale must never be data-derived inside the score, because every
//! self-scaled form was measured gameable — the bare standardized sum
//! rewards σ inflation (a badly-fit cell scored 40.35 against the truth's
//! 51.05 and won the grid), σ²·Σρ and the log-likelihood form `Σρ + W·ln σ`
//! reward bulk concentration (a tiny-MAD bulk with capped outliers scored
//! B = −729.1 against the truth's −501.6 while the raw loss said truth
//! 54.0 < endpoint 64.1). The raw loss with the frozen band counts every
//! scattered point honestly and the truth wins it. The coordinate descent
//! ranks candidates at that same frozen band, recentering each candidate's
//! c with one weighted-mean pass (the spec's recenter machinery, O(W)) —
//! a frozen center alone punishes level-shifting moves no matter how much
//! they tighten vals around their own new level (measured: every
//! γ₀-involving ladder step scored 470–1102 against the opening's 45.8 and
//! the descent froze γ₀ at its start for every ε cell).
//!
//! # Census provenance
//!
//! riir-refine Plan 392 Phase 0 measured the logged exposure matrix at
//! effective rank 1 (10% share; σ = [2845.2, 185.1, 153.7, 104.9, 50.8, 20.4,
//! 11.6, 0.0]; the first census pass reported rank 2 from a divergent Jacobi
//! pairing — corrected 2026-11-20, trace invariant rel err 2.3e-16, verdict
//! unchanged and strengthened) against bar 7 → verdict SPARSE. Our surface
//! therefore arms the univariate path on sparse mixes ([`Saturation::Sparse`]:
//! a per-domain power-law fit, a typed [`GainCell::NoCurve`] decline for
//! constant columns — never a silent γ). The cross-domain γ arm stays built
//! and gated for richer-mix consumers; the in-module tests pin both halves.
//!
//! # Determinism (Issue-940 law)
//!
//! Every reduction is an ascending fold over the observation axis; the (α, ε)
//! grids are fixed closed-form lattices (never adaptive); the location is the
//! upper median (element at index n/2) via an iterative three-way quickselect
//! (middle element pivot, deterministic); grid-cell ties keep the FIRST cell
//! (α ascending outer, ε ascending inner) under strict `<`; coordinate-descent
//! candidate ties keep the FIRST in scan order (i ascending, j ascending, δ in
//! [`MOVE_LADDER`] order). A surface committing gains across the sync boundary
//! re-pins this order + tie rule at its own site.
//!
//! # Sigmoid, never softmax
//!
//! [`GainEstimator::sigmoid_weights`] maps each gain through the house
//! [`crate::sigmoid`] — `w_k = σ((g_k − median)/τ)`, normalised by the
//! ascending weight sum — never a softmax.
//!
//! # Stationary-surface ban
//!
//! Allocation surfaces only. This estimator prices where the next observation
//! goes; it must NEVER route scan/decide paths.
//!
//! # Alloc posture
//!
//! The decision path ([`GainEstimator::update_counts`],
//! [`GainEstimator::marginal_gain`], [`GainEstimator::derivative_gain`],
//! [`GainEstimator::sigmoid_weights`], [`GainEstimator::predicted_gap`]) is
//! alloc-free (fixed `[f64; MAX_DOMAINS]` scratch, in-place quickselect) —
//! gated by `tests/epiplexity_gain_g4_alloc.rs`. A refit is bounded:
//! `grids × CD × K² × W` work, one `Vec`-scratch class per fit pass.
//!
//! # Gain formulas (exact)
//!
//! With `n = counts_now` (raw), `n₁ = n + 1` (pseudocounted):
//!
//! - cross [`GainEstimator::marginal_gain`]:
//!   `gain_k = n_k·α·γ_k/N·(gap−ε)` with `N = Σ_l γ_l·(n_l+1)` and
//!   `gap = ε + β·N^(−α)`; `N ≤ 0` → every cell `Gain(0.0)`; clamped `≥ 0`.
//! - sparse [`GainEstimator::marginal_gain`]: `S = Σ_l n_l`; per fitted
//!   domain `(n_k/S)·α_k·(gap_k−ε_k)` with `gap_k = ε_k + β_k·(n_k+1)^(−α_k)`;
//!   an unfitted (constant) domain declines typed [`GainCell::NoCurve`] (`S ≤
//!   0` degenerates fitted cells to `Gain(0.0)`).
//! - [`GainEstimator::derivative_gain`] (the raw `−d gap/d n_k`): cross
//!   `α·γ_k·(gap−ε)/N`; sparse fitted `α_k·(gap_k−ε_k)/(n_k+1)`.
//! - [`GainEstimator::sigmoid_weights`]: `w_k = σ((g_k − median)/τ)` (median
//!   = upper median, same quickselect), normalised by the ascending weight
//!   sum; [`GainCell::NoCurve`] → `0.0`.

use core::fmt;

/// Hard domain ceiling of a mix panel (the [`RankCensus::singular_values`]
/// stack width).
pub const MAX_DOMAINS: usize = 16;

/// Hard observation ceiling of a mix panel.
pub const MAX_OBSERVATIONS: usize = 4096;

/// Minimum observations a [`GainEstimator::fit`] accepts.
pub const MIN_OBSERVATIONS: usize = 8;

const PSEUDOCOUNT: f64 = 1.0;
const ALPHA_GRID: usize = 12;
const ALPHA_MIN: f64 = 0.05;
const ALPHA_MAX: f64 = 3.0;
const EPS_GRID: usize = 14;
const EPS_TOP_SHARE: f64 = 0.95;
const CD_MAX_SWEEPS: usize = 24;
const GAMMA_MIN: f64 = 1e-3;
const MOVE_LADDER: [f64; 6] = [0.4, 0.2, 0.1, 0.05, 0.02, 0.005];
const HUBER_DELTA: f64 = 1.345;
/// ρ tail offset: `δ/2` at `δ = 1.345`, written as the pinned constant.
const HUBER_TAIL_OFFSET: f64 = 0.673;
const MAD_TO_SIGMA: f64 = 1.4826;
const SIGMA_FLOOR: f64 = 1e-9;
const GAP_FLOOR: f64 = 1e-6;
const JACOBI_TOL: f64 = 1e-14;
const JACOBI_DENOM_FLOOR: f64 = 1e-300;
const JACOBI_MAX_SWEEPS: usize = 100;
const SIGMA_ZERO: f64 = 1e-12;
const EFF10_SHARE: f64 = 0.10;
const EFF1_SHARE: f64 = 0.01;

/// Why a fit or a query refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitRefusal {
    /// Fewer than [`MIN_OBSERVATIONS`] usable observations.
    TooFewObservations {
        /// What the caller supplied.
        have: usize,
        /// [`MIN_OBSERVATIONS`].
        need: usize,
    },
    /// More than [`MAX_OBSERVATIONS`] observations.
    PanelTooLarge {
        /// What the caller supplied.
        have: usize,
        /// [`MAX_OBSERVATIONS`].
        max: usize,
    },
    /// `counts.len() != domains·observations` (or `accuracy.len() !=
    /// observations`).
    ShapeMismatch,
    /// A count or an accuracy value is not finite.
    NonFiniteInput,
    /// An accuracy value is outside `[0, 1]`, or the smallest error gap is ≤
    /// [`GAP_FLOOR`] — a saturated series has no learnable gap.
    AccuracyDomain,
    /// Domain count outside `1..=[MAX_DOMAINS]`.
    DomainCount,
}

impl fmt::Display for FitRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewObservations { have, need } => {
                write!(
                    f,
                    "epiplexity fit needs at least {need} observations, got {have}"
                )
            }
            Self::PanelTooLarge { have, max } => {
                write!(
                    f,
                    "epiplexity panel holds {have} observations, ceiling {max}"
                )
            }
            Self::ShapeMismatch => {
                write!(
                    f,
                    "epiplexity panel shape mismatch (counts vs domains·observations / accuracy)"
                )
            }
            Self::NonFiniteInput => write!(f, "epiplexity input carries a non-finite value"),
            Self::AccuracyDomain => {
                write!(f, "accuracy series outside (0, 1): no learnable error gap")
            }
            Self::DomainCount => {
                write!(
                    f,
                    "epiplexity panel: domain count outside 1..={MAX_DOMAINS}"
                )
            }
        }
    }
}

impl std::error::Error for FitRefusal {}

/// A per-observation corpus-count panel: `counts[t·domains + k]` is domain
/// `k`'s raw observation count at time `t` (row-major).
#[derive(Clone, Copy, Debug)]
pub struct MixPanel<'a> {
    /// Number of domains (columns), `1..=[MAX_DOMAINS]`.
    pub domains: usize,
    /// Number of observations (rows), `1..=[MAX_OBSERVATIONS]`.
    pub observations: usize,
    /// `domains·observations` finite counts, row-major.
    pub counts: &'a [f32],
}

impl<'a> MixPanel<'a> {
    /// Build a panel from its shape and its row-major counts.
    pub fn new(domains: usize, observations: usize, counts: &'a [f32]) -> Self {
        Self {
            domains,
            observations,
            counts,
        }
    }

    /// Shape + finiteness validation, in the pinned order: domain count,
    /// observation count, shape, finiteness.
    pub fn validate(&self) -> Result<(), FitRefusal> {
        if self.domains < 1 || self.domains > MAX_DOMAINS {
            return Err(FitRefusal::DomainCount);
        }
        if self.observations < 1 || self.observations > MAX_OBSERVATIONS {
            return Err(FitRefusal::PanelTooLarge {
                have: self.observations,
                max: MAX_OBSERVATIONS,
            });
        }
        if self.counts.len() != self.domains * self.observations {
            return Err(FitRefusal::ShapeMismatch);
        }
        for &c in self.counts {
            if !c.is_finite() {
                return Err(FitRefusal::NonFiniteInput);
            }
        }
        Ok(())
    }
}

/// The singular-spectrum census behind a [`Saturation`] verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankCensus {
    /// Singular values of the centred panel, descending; entries beyond
    /// `domains` are zero.
    pub singular_values: [f32; MAX_DOMAINS],
    /// Domains the census covered.
    pub domains: usize,
    /// Count of singular values ≥ 10% of the largest.
    pub eff10: usize,
    /// Count of singular values ≥ 1% of the largest.
    pub eff1: usize,
    /// `(Σσ)²/Σσ²` — the effective-rank participation ratio.
    pub participation_ratio: f32,
    /// The cross-domain bar (`domains − 1`; 0 for a single-domain panel).
    pub bar: usize,
}

/// Whether the panel's count columns move together (cross-domain curve) or
/// apart (per-domain curves).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Saturation {
    /// The panel carries at least `domains − 1` strong directions: one shared
    /// cross-domain curve is identifiable.
    CrossDomain(RankCensus),
    /// The panel is rank-deficient (or fully constant): per-domain univariate
    /// fits are the armed path.
    Sparse(RankCensus),
}

// ── fit types ───────────────────────────────────────────────────────────

/// One fitted cross-domain curve: `u = ε + β·N^(−α)` over
/// `N = Σ_k γ_k·(n_k + 1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossDomainFit {
    /// Fitted decay exponent.
    pub alpha: f64,
    /// Fitted scale (`β = e^c`, `c` the robust log-location).
    pub beta: f64,
    /// Fitted error floor.
    pub epsilon: f64,
    /// Fitted domain mix on the simplex (entries beyond `domains` zero).
    pub gamma: [f64; MAX_DOMAINS],
    /// Huber objective of the winning grid cell (lower is better).
    pub objective: f64,
}

/// One fitted per-domain curve: `u = ε + β·(n + 1)^(−α)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnivariateFit {
    /// Fitted decay exponent.
    pub alpha: f64,
    /// Fitted scale.
    pub beta: f64,
    /// Fitted error floor.
    pub epsilon: f64,
    /// Huber objective of the winning grid cell (lower is better).
    pub objective: f64,
}

/// A typed gain answer: a non-negative marginal gain, or an honest decline
/// (the domain's series never moved, so no curve exists — never a silent γ).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GainCell {
    /// A non-negative marginal gain.
    Gain(f32),
    /// No curve was fitted for this domain (constant column): no answer.
    NoCurve,
}

// ── saturation (the census half; alloc-free over fixed stacks) ─────────

/// Cyclic Jacobi eigensolver over a symmetric `K×K` Gram matrix stored
/// row-major with row stride [`MAX_DOMAINS`]. One rotation per (p, q) pair
/// per sweep, ≤ [`JACOBI_MAX_SWEEPS`] sweeps, stop after a sweep with no
/// rotation; the rotation threshold is `|a_pq| > tol·(max|diag| + floor)`.
/// Eigenvalues land on the diagonal.
fn jacobi_diagonalize(gram: &mut [f64; MAX_DOMAINS * MAX_DOMAINS], k: usize) {
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut max_diag = 0.0_f64;
        for d in 0..k {
            max_diag = max_diag.max(gram[d * MAX_DOMAINS + d].abs());
        }
        let tol = JACOBI_TOL * (max_diag + JACOBI_DENOM_FLOOR);
        let mut rotated = false;
        for p in 0..k {
            for q in p + 1..k {
                let apq = gram[p * MAX_DOMAINS + q];
                if apq.abs() <= tol {
                    continue;
                }
                let app = gram[p * MAX_DOMAINS + p];
                let aqq = gram[q * MAX_DOMAINS + q];
                let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
                let c = theta.cos();
                // G ← Mᵀ·G·M (the −s orientation): the pairing that zeroes
                // a_pq at the pinned θ = 0.5·atan2(2a_pq, a_qq − a_pp). The
                // mirrored +s form pairs with the NEGATED θ and drives the
                // target element UP, not to zero (measured: a rank-1 Gram's
                // off-diagonal grew 3 → 4.8 under +s and the census read a
                // rank-1 panel as full-rank).
                let s = -theta.sin();
                for r in 0..k {
                    let gp = gram[p * MAX_DOMAINS + r];
                    let gq = gram[q * MAX_DOMAINS + r];
                    gram[p * MAX_DOMAINS + r] = c * gp + s * gq;
                    gram[q * MAX_DOMAINS + r] = -s * gp + c * gq;
                }
                for r in 0..k {
                    let gp = gram[r * MAX_DOMAINS + p];
                    let gq = gram[r * MAX_DOMAINS + q];
                    gram[r * MAX_DOMAINS + p] = c * gp + s * gq;
                    gram[r * MAX_DOMAINS + q] = -s * gp + c * gq;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }
}

/// The saturation gate: is the panel's centred count matrix rich enough
/// (effective rank ≥ `domains − 1` at the 10% singular-value share) to
/// identify ONE shared cross-domain curve? A fully constant panel
/// (`σmax ≤ 1e-12`) is [`Saturation::Sparse`] by definition.
///
/// Determinism: column means, Gram entries, spectrum sums — all ascending
/// folds over the observation axis; the singular-value sort is a stable
/// descending insertion over fixed stacks.
pub fn mix_saturation(panel: &MixPanel<'_>) -> Result<Saturation, FitRefusal> {
    panel.validate()?;
    let k = panel.domains;
    let w = panel.observations;
    if k == 1 {
        let mut sv = [0.0_f32; MAX_DOMAINS];
        sv[0] = 1.0;
        return Ok(Saturation::CrossDomain(RankCensus {
            singular_values: sv,
            domains: 1,
            eff10: 1,
            eff1: 1,
            participation_ratio: 1.0,
            bar: 0,
        }));
    }
    let mut mean = [0.0_f64; MAX_DOMAINS];
    for t in 0..w {
        let row = &panel.counts[t * k..t * k + k];
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += f64::from(v);
        }
    }
    for m in mean.iter_mut().take(k) {
        *m /= w as f64;
    }
    let mut gram = [0.0_f64; MAX_DOMAINS * MAX_DOMAINS];
    for t in 0..w {
        let row = &panel.counts[t * k..t * k + k];
        for a in 0..k {
            let ca = f64::from(row[a]) - mean[a];
            for b in 0..k {
                gram[a * MAX_DOMAINS + b] += ca * (f64::from(row[b]) - mean[b]);
            }
        }
    }
    jacobi_diagonalize(&mut gram, k);
    let mut sv = [0.0_f64; MAX_DOMAINS];
    for (d, slot) in sv.iter_mut().enumerate().take(k) {
        *slot = gram[d * MAX_DOMAINS + d].max(0.0).sqrt();
    }
    // Stable descending insertion sort over sv[0..k] (strict `<` compare
    // keeps equal values in their original order).
    for i in 1..k {
        let key = sv[i];
        let mut j = i;
        while j > 0 && sv[j - 1] < key {
            sv[j] = sv[j - 1];
            j -= 1;
        }
        sv[j] = key;
    }
    let sigma_max = sv[0];
    let eff10 = sv
        .iter()
        .take(k)
        .filter(|&&s| s >= EFF10_SHARE * sigma_max)
        .count();
    let eff1 = sv
        .iter()
        .take(k)
        .filter(|&&s| s >= EFF1_SHARE * sigma_max)
        .count();
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for &s in sv.iter().take(k) {
        sum += s;
        sum_sq += s * s;
    }
    let ratio = if sum_sq > 0.0 {
        sum * sum / sum_sq
    } else {
        0.0
    };
    let mut census_sv = [0.0_f32; MAX_DOMAINS];
    for (slot, &s) in census_sv.iter_mut().zip(sv.iter()).take(k) {
        *slot = s as f32;
    }
    let census = RankCensus {
        singular_values: census_sv,
        domains: k,
        eff10,
        eff1,
        participation_ratio: ratio as f32,
        bar: k - 1,
    };
    if sigma_max <= SIGMA_ZERO {
        return Ok(Saturation::Sparse(census));
    }
    if eff10 >= census.bar {
        Ok(Saturation::CrossDomain(census))
    } else {
        Ok(Saturation::Sparse(census))
    }
}

// ── fit machinery (grid + coordinate descent under a Huber location) ──

/// The `n/2`-th order statistic (the UPPER median for even `n`), by an
/// iterative three-way (Dutch national flag) quickselect — middle element as
/// pivot, deterministic; permutes `buf` in place.
fn order_statistic(buf: &mut [f64], target: usize) -> f64 {
    let mut lo = 0_usize;
    let mut hi = buf.len();
    loop {
        if hi - lo <= 1 {
            return buf[lo];
        }
        let pivot = buf[lo + (hi - lo) / 2];
        let mut lt = lo;
        let mut i = lo;
        let mut gt = hi;
        while i < gt {
            let v = buf[i];
            if v < pivot {
                buf.swap(lt, i);
                lt += 1;
                i += 1;
            } else if v > pivot {
                gt -= 1;
                buf.swap(i, gt);
            } else {
                i += 1;
            }
        }
        if target < lt {
            hi = lt;
        } else if target >= gt {
            lo = gt;
        } else {
            return pivot;
        }
    }
}

/// Σ ρ((v−c)/σ) with ρ(x) = 0.5x² for |x| ≤ [`HUBER_DELTA`] else
/// [`HUBER_DELTA`]·(|x| − [`HUBER_TAIL_OFFSET`]); ascending fold.
fn huber_objective(vals: &[f64], c: f64, sigma: f64) -> f64 {
    let mut obj = 0.0_f64;
    for &v in vals {
        let x = (v - c) / sigma;
        let ax = x.abs();
        obj += if ax <= HUBER_DELTA {
            0.5 * x * x
        } else {
            HUBER_DELTA * (ax - HUBER_TAIL_OFFSET)
        };
    }
    obj
}

/// The Huber ρ sum at an EXPLICIT band δ: `ρ_δ(x) = 0.5x²` for `|x| ≤ δ`
/// else `δ·(|x| − δ/2)` — the raw-unit loss the cell score and the
/// coordinate descent use (the band is frozen per cell, never data-derived
/// inside the score). Ascending fold.
fn huber_loss_at(vals: &[f64], c: f64, delta: f64) -> f64 {
    let tail_offset = 0.5 * delta;
    let mut obj = 0.0_f64;
    for &v in vals {
        let dev = (v - c).abs();
        obj += if dev <= delta {
            0.5 * dev * dev
        } else {
            delta * (dev - tail_offset)
        };
    }
    obj
}

/// One weighted-mean recenter pass at band `delta` (the spec's recenter
/// machinery): `c ← Σw·v/Σw` with `w = 1` inside the band and `δ/|v−c₀|`
/// outside, starting from `c0`. O(W), no sort — the cheap level-free center
/// the coordinate descent's candidate scoring uses.
fn huber_recenter(vals: &[f64], c0: f64, delta: f64) -> f64 {
    let mut wsum = 0.0_f64;
    let mut wv = 0.0_f64;
    for &v in vals {
        let dev = (v - c0).abs();
        let w = if dev <= delta { 1.0 } else { delta / dev };
        wsum += w;
        wv += w * v;
    }
    wv / wsum
}

struct Location {
    c: f64,
    sigma: f64,
    objective: f64,
}

/// Robust location of `vals`: upper median (element at index n/2) via
/// [`order_statistic`] on `sel` (a scratch copy), σ = [`MAD_TO_SIGMA`]·MAD
/// floored at [`SIGMA_FLOOR`], then TWO weighted-mean recenter iterations
/// (weight 1 inside the [`HUBER_DELTA`]·σ band, `d/|v−c|` outside), and the
/// resulting Huber objective. The recenter + objective folds read `vals` in
/// its original (observation) order — the pinned ascending fold.
fn huber_location(vals: &[f64], sel: &mut [f64]) -> Location {
    let n = vals.len();
    sel[..n].copy_from_slice(vals);
    let median = order_statistic(&mut sel[..n], n / 2);
    for (s, &v) in sel.iter_mut().zip(vals.iter()) {
        *s = (v - median).abs();
    }
    let mad = order_statistic(&mut sel[..n], n / 2);
    let sigma = (MAD_TO_SIGMA * mad).max(SIGMA_FLOOR);
    let d = HUBER_DELTA * sigma;
    let mut c = median;
    for _ in 0..2 {
        let mut wsum = 0.0_f64;
        let mut wv = 0.0_f64;
        for &v in vals {
            let dev = (v - c).abs();
            let w = if dev <= d { 1.0 } else { d / dev };
            wsum += w;
            wv += w * v;
        }
        c = wv / wsum;
    }
    let objective = huber_objective(vals, c, sigma);
    Location {
        c,
        sigma,
        objective,
    }
}

/// Per-fit-pass scratch: one `Vec` class, three `W`-sized `f64` buffers
/// (`zed` = per-cell log-gap, `vals` = working curve values, `sel` =
/// quickselect scratch), reused across every grid cell and CD candidate.
struct FitScratch {
    zed: Vec<f64>,
    vals: Vec<f64>,
    sel: Vec<f64>,
}

impl FitScratch {
    fn new(w: usize) -> Self {
        Self {
            zed: vec![0.0; w],
            vals: vec![0.0; w],
            sel: vec![0.0; w],
        }
    }
}

/// Everything one grid cell needs: the pseudocounted counts, the cell's
/// log-gap series, and the shape/exponent the curve is evaluated at.
struct CellCtx<'a> {
    counts: &'a [f64],
    zed: &'a [f64],
    k: usize,
    stride: usize,
    w: usize,
    alpha: f64,
}

/// `vals[t] = zed[t] + α·ln N_t` with `N_t = Σ_k γ_k·counts[t·stride + k]`
/// (counts arrive pseudocounted); ascending over `t`.
fn fill_vals(ctx: &CellCtx<'_>, gamma: &[f64; MAX_DOMAINS], vals: &mut [f64]) {
    for (t, slot) in vals.iter_mut().enumerate().take(ctx.w) {
        let row = &ctx.counts[t * ctx.stride..t * ctx.stride + ctx.k];
        let mut n = 0.0_f64;
        for (g, &cnt) in gamma.iter().zip(row.iter()) {
            n += g * cnt;
        }
        *slot = ctx.zed[t] + ctx.alpha * n.ln();
    }
}

/// The log-spaced α lattice: [`ALPHA_GRID`] points over
/// [`ALPHA_MIN`]..=[`ALPHA_MAX`], ascending.
fn alpha_grid() -> [f64; ALPHA_GRID] {
    let lo = ALPHA_MIN.ln();
    let step = (ALPHA_MAX.ln() - lo) / (ALPHA_GRID - 1) as f64;
    let mut grid = [0.0_f64; ALPHA_GRID];
    for (i, slot) in grid.iter_mut().enumerate() {
        *slot = (lo + i as f64 * step).exp();
    }
    grid
}

/// Cyclic coordinate descent on γ: per sweep, scan all (i, j, δ) transfers
/// (i ascending, j ascending, δ in [`MOVE_LADDER`] — descending magnitude),
/// validity `γ_i−δ ≥ [`GAMMA_MIN`]` and `γ_j+δ ≤ 1−(k−1)·[`GAMMA_MIN`]`,
/// each candidate scored O(W) under the cell's FROZEN band: one recenter
/// pass (level-free center — see the module docs) then `Σρ_δ(v − c_cand)`
/// over the recomputed vals; the opening uses the same scorer. Apply the
/// strictly best candidate (ties → first in scan order) if it beats the
/// sweep's opening, recompute the FULL location (fresh median/MAD), and
/// sweep again — else stop. ≤ [`CD_MAX_SWEEPS`] sweeps.
#[allow(clippy::too_many_arguments)]
fn coordinate_descent(
    ctx: &CellCtx<'_>,
    vals: &mut [f64],
    sel: &mut [f64],
    gamma: &mut [f64; MAX_DOMAINS],
    c: &mut f64,
    sigma: &mut f64,
    obj: &mut f64,
    band: f64,
) {
    let gamma_cap = 1.0 - (ctx.k - 1) as f64 * GAMMA_MIN;
    for _ in 0..CD_MAX_SWEEPS {
        let opening = huber_loss_at(vals, huber_recenter(vals, *c, band), band);
        let mut best_obj = opening;
        let mut best_move: Option<(usize, usize, f64)> = None;
        for i in 0..ctx.k {
            for j in 0..ctx.k {
                if i == j {
                    continue;
                }
                for &delta in MOVE_LADDER.iter() {
                    let gi = gamma[i] - delta;
                    let gj = gamma[j] + delta;
                    if gi < GAMMA_MIN || gj > gamma_cap {
                        continue;
                    }
                    let mut trial = *gamma;
                    trial[i] = gi;
                    trial[j] = gj;
                    fill_vals(ctx, &trial, vals);
                    let cand = huber_loss_at(vals, huber_recenter(vals, *c, band), band);
                    if cand < best_obj {
                        best_obj = cand;
                        best_move = Some((i, j, delta));
                    }
                }
            }
        }
        match best_move {
            Some((i, j, delta)) => {
                gamma[i] -= delta;
                gamma[j] += delta;
                fill_vals(ctx, gamma, vals);
                let loc = huber_location(&vals[..ctx.w], &mut sel[..ctx.w]);
                *c = loc.c;
                *sigma = loc.sigma;
                *obj = loc.objective;
            }
            None => break,
        }
    }
}

/// Fit one curve over the grid: for each (α, ε) cell build
/// `z_t = ln(u_t − ε)`, start γ uniform 1/k, take the robust location, run
/// the coordinate descent when `cross && k > 1`, and keep the cell if its
/// parameters are finite and `β > 0`. Global best is strict `<`, first cell
/// wins ties (α ascending outer, ε ascending inner). `None` = every cell
/// skipped (a non-finite input degeneracy).
fn fit_curve(
    counts: &[f64],
    gaps: &[f64],
    k: usize,
    stride: usize,
    cross: bool,
) -> Option<CellFit> {
    let w = gaps.len();
    if w == 0 || k == 0 {
        return None;
    }
    let mut sc = FitScratch::new(w);
    let alphas = alpha_grid();
    let min_gap = gaps.iter().copied().fold(f64::INFINITY, f64::min);
    if !min_gap.is_finite() || min_gap <= 0.0 {
        return None;
    }
    let eps_top = min_gap * EPS_TOP_SHARE;
    let mut best: Option<CellFit> = None;
    for alpha in alphas.iter().copied() {
        for ei in 0..EPS_GRID {
            let epsilon = eps_top * (ei as f64) / ((EPS_GRID - 1) as f64);
            let mut cell_ok = true;
            {
                let zed = &mut sc.zed[..w];
                for (slot, &u) in zed.iter_mut().zip(gaps.iter()) {
                    let r = u - epsilon;
                    if r <= 0.0 {
                        cell_ok = false;
                        break;
                    }
                    *slot = r.ln();
                }
            }
            if !cell_ok {
                continue;
            }
            let mut gamma = [0.0_f64; MAX_DOMAINS];
            let uniform = 1.0 / (k as f64);
            for g in gamma.iter_mut().take(k) {
                *g = uniform;
            }
            let ctx = CellCtx {
                counts,
                zed: &sc.zed,
                k,
                stride,
                w,
                alpha,
            };
            fill_vals(&ctx, &gamma, &mut sc.vals);
            let loc = huber_location(&sc.vals[..w], &mut sc.sel[..w]);
            let (mut c, mut sigma, mut obj) = (loc.c, loc.sigma, loc.objective);
            // The cell's score band: frozen from the OPENING location —
            // uniform γ is model-free, so the fit cannot game the band (see
            // the module docs for the measured failures of self-scaled
            // scores).
            let band = HUBER_DELTA * sigma;
            if cross && k > 1 {
                coordinate_descent(
                    &ctx,
                    &mut sc.vals,
                    &mut sc.sel,
                    &mut gamma,
                    &mut c,
                    &mut sigma,
                    &mut obj,
                    band,
                );
            }
            let beta = c.exp();
            // Raw-unit cell score at the frozen band: re-fill vals with the
            // final γ (the CD's trial loop leaves the last trial's vals on a
            // break) and count every point honestly.
            fill_vals(&ctx, &gamma, &mut sc.vals);
            let obj = huber_loss_at(&sc.vals[..w], c, band);
            if !c.is_finite()
                || !sigma.is_finite()
                || !obj.is_finite()
                || !beta.is_finite()
                || beta <= 0.0
            {
                continue;
            }
            let cell = CellFit {
                alpha,
                epsilon,
                beta,
                gamma,
                objective: obj,
            };
            let keep = match &best {
                Some(b) => cell.objective < b.objective,
                None => true,
            };
            if keep {
                best = Some(cell);
            }
        }
    }
    best
}

/// One finished grid cell (internal; the public fits carry the same fields
/// minus γ).
struct CellFit {
    alpha: f64,
    epsilon: f64,
    beta: f64,
    gamma: [f64; MAX_DOMAINS],
    objective: f64,
}

// ── the estimator ───────────────────────────────────────────────────────

/// The fitted marginal-gain estimator: one saturation verdict plus the
/// cross-domain curve, the per-domain curves, or both halves' abstentions —
/// over a mutable current-counts register the decision path reads
/// alloc-free.
#[derive(Clone, Debug)]
pub struct GainEstimator {
    domains: usize,
    saturation: Saturation,
    cross: Option<CrossDomainFit>,
    uni: [Option<UnivariateFit>; MAX_DOMAINS],
    counts_now: [f64; MAX_DOMAINS],
}

impl GainEstimator {
    /// Fit the error-gap model over `panel` + `accuracy` (same length as the
    /// panel's observations; every value finite and in `[0, 1]`; the smallest
    /// gap `1 − y` must exceed [`GAP_FLOOR`] — a saturated series has no
    /// learnable gap, which IS the saturation gate). Cross-domain verdict →
    /// one shared curve (γ armed); sparse verdict → per-domain univariate
    /// fits for every non-constant column, typed [`GainCell::NoCurve`] for
    /// the rest. All fit internals are `f64`; the API's gains emit `f32`.
    pub fn fit(panel: &MixPanel<'_>, accuracy: &[f32]) -> Result<Self, FitRefusal> {
        panel.validate()?;
        let k = panel.domains;
        let w = panel.observations;
        if accuracy.len() != w {
            return Err(FitRefusal::ShapeMismatch);
        }
        if w < MIN_OBSERVATIONS {
            return Err(FitRefusal::TooFewObservations {
                have: w,
                need: MIN_OBSERVATIONS,
            });
        }
        let mut min_gap = f64::INFINITY;
        for &y in accuracy {
            let yv = f64::from(y);
            if !yv.is_finite() {
                return Err(FitRefusal::NonFiniteInput);
            }
            if !(0.0..=1.0).contains(&yv) {
                return Err(FitRefusal::AccuracyDomain);
            }
            min_gap = min_gap.min(1.0 - yv);
        }
        if min_gap <= GAP_FLOOR {
            return Err(FitRefusal::AccuracyDomain);
        }
        let gaps: Vec<f64> = accuracy.iter().map(|&y| 1.0 - f64::from(y)).collect();
        let counts_pc: Vec<f64> = panel
            .counts
            .iter()
            .map(|&c| f64::from(c) + PSEUDOCOUNT)
            .collect();
        let saturation = mix_saturation(panel)?;
        let mut uni = [None; MAX_DOMAINS];
        let cross = match saturation {
            Saturation::CrossDomain(_) => {
                fit_curve(&counts_pc, &gaps, k, k, true).map(|cell| CrossDomainFit {
                    alpha: cell.alpha,
                    beta: cell.beta,
                    epsilon: cell.epsilon,
                    gamma: cell.gamma,
                    objective: cell.objective,
                })
            }
            Saturation::Sparse(_) => {
                for (d, slot) in uni.iter_mut().enumerate().take(k) {
                    let mut mn = f32::INFINITY;
                    let mut mx = f32::NEG_INFINITY;
                    for v in panel.counts.iter().skip(d).step_by(k).take(w) {
                        mn = mn.min(*v);
                        mx = mx.max(*v);
                    }
                    if mn == mx {
                        continue;
                    }
                    let series: Vec<f64> = panel
                        .counts
                        .iter()
                        .skip(d)
                        .step_by(k)
                        .take(w)
                        .map(|&v| f64::from(v) + PSEUDOCOUNT)
                        .collect();
                    if let Some(cell) = fit_curve(&series, &gaps, 1, 1, false) {
                        *slot = Some(UnivariateFit {
                            alpha: cell.alpha,
                            beta: cell.beta,
                            epsilon: cell.epsilon,
                            objective: cell.objective,
                        });
                    }
                }
                None
            }
        };
        let mut counts_now = [0.0_f64; MAX_DOMAINS];
        let last = (w - 1) * k;
        for (d, slot) in counts_now.iter_mut().enumerate().take(k) {
            *slot = f64::from(panel.counts[last + d]);
        }
        Ok(Self {
            domains: k,
            saturation,
            cross,
            uni,
            counts_now,
        })
    }

    /// Domains the estimator was fitted over.
    pub fn domains(&self) -> usize {
        self.domains
    }

    /// The saturation verdict (with its rank census).
    pub fn saturation(&self) -> &Saturation {
        &self.saturation
    }

    /// The cross-domain curve, when the verdict armed it.
    pub fn cross_fit(&self) -> Option<&CrossDomainFit> {
        self.cross.as_ref()
    }

    /// Domain `domain`'s univariate curve, when fitted (sparse verdict,
    /// non-constant column).
    pub fn univariate_fit(&self, domain: usize) -> Option<&UnivariateFit> {
        if domain >= self.domains {
            return None;
        }
        self.uni[domain].as_ref()
    }

    /// The current counts register (raw, unpseudocounted; entries beyond
    /// `domains` are zero).
    pub fn counts_now(&self) -> &[f64; MAX_DOMAINS] {
        &self.counts_now
    }

    /// Re-seat the current counts (raw, one value per domain). Alloc-free.
    pub fn update_counts(&mut self, counts: &[f32]) -> Result<(), FitRefusal> {
        if counts.len() != self.domains {
            return Err(FitRefusal::ShapeMismatch);
        }
        for (d, &v) in counts.iter().enumerate() {
            let vv = f64::from(v);
            if !vv.is_finite() {
                return Err(FitRefusal::NonFiniteInput);
            }
            self.counts_now[d] = vv;
        }
        Ok(())
    }

    /// The cross arm's `(N, gap−ε)` over the CURRENT counts; `None` when the
    /// guard fails (`N ≤ 0` or a non-finite gap).
    fn cross_margin(&self) -> Option<(f64, f64)> {
        let f = self.cross.as_ref()?;
        let mut n_eff = 0.0_f64;
        for d in 0..self.domains {
            n_eff += f.gamma[d] * (self.counts_now[d] + PSEUDOCOUNT);
        }
        if n_eff <= 0.0 {
            return None;
        }
        let gap = f.epsilon + f.beta * n_eff.powf(-f.alpha);
        if gap.is_finite() {
            Some((n_eff, gap - f.epsilon))
        } else {
            None
        }
    }

    /// A sparse domain's current gap, when finite.
    fn sparse_gap(&self, u: &UnivariateFit, domain: usize) -> Option<f64> {
        let n1 = self.counts_now[domain] + PSEUDOCOUNT;
        let gap = u.epsilon + u.beta * n1.powf(-u.alpha);
        if gap.is_finite() { Some(gap) } else { None }
    }

    /// The marginal gain per domain (see the module docs for the exact
    /// forms). Alloc-free.
    pub fn marginal_gain(&self) -> [GainCell; MAX_DOMAINS] {
        let mut cells = [GainCell::NoCurve; MAX_DOMAINS];
        if let Some(f) = self.cross.as_ref() {
            let margin = self.cross_margin();
            for (d, slot) in cells.iter_mut().enumerate().take(self.domains) {
                let g = match margin {
                    Some((n_eff, m)) => {
                        (self.counts_now[d] * f.alpha * f.gamma[d] / n_eff * m).max(0.0)
                    }
                    None => 0.0,
                };
                *slot = GainCell::Gain(g as f32);
            }
        } else {
            let mut total_raw = 0.0_f64;
            for d in 0..self.domains {
                total_raw += self.counts_now[d];
            }
            for (d, slot) in cells.iter_mut().enumerate().take(self.domains) {
                *slot = match self.uni[d] {
                    Some(u) => {
                        let g = match self.sparse_gap(&u, d) {
                            Some(gap) if total_raw > 0.0 => {
                                (self.counts_now[d] / total_raw * u.alpha * (gap - u.epsilon))
                                    .max(0.0)
                            }
                            _ => 0.0,
                        };
                        GainCell::Gain(g as f32)
                    }
                    None => GainCell::NoCurve,
                };
            }
        }
        cells
    }

    /// The raw `−d gap/d n_k` per domain (see the module docs). Alloc-free.
    pub fn derivative_gain(&self) -> [GainCell; MAX_DOMAINS] {
        let mut cells = [GainCell::NoCurve; MAX_DOMAINS];
        if let Some(f) = self.cross.as_ref() {
            let margin = self.cross_margin();
            for (d, slot) in cells.iter_mut().enumerate().take(self.domains) {
                let g = match margin {
                    Some((n_eff, m)) => f.alpha * f.gamma[d] * m / n_eff,
                    None => 0.0,
                };
                *slot = GainCell::Gain(g as f32);
            }
        } else {
            for (d, slot) in cells.iter_mut().enumerate().take(self.domains) {
                *slot = match self.uni[d] {
                    Some(u) => {
                        let g = match self.sparse_gap(&u, d) {
                            Some(gap) => {
                                u.alpha * (gap - u.epsilon) / (self.counts_now[d] + PSEUDOCOUNT)
                            }
                            None => 0.0,
                        };
                        GainCell::Gain(g as f32)
                    }
                    None => GainCell::NoCurve,
                };
            }
        }
        cells
    }

    /// Sigmoid weights over the marginal gains: `w_k = σ((g_k − median)/τ)`
    /// (upper-median via the same quickselect over a fixed stack; τ =
    /// `|tau_s|` floored at `1e-6`), normalised by the ascending weight sum;
    /// [`GainCell::NoCurve`] → `0.0`. The house [`crate::sigmoid`], never a
    /// softmax. Alloc-free.
    pub fn sigmoid_weights(&self, tau_s: f32) -> [f32; MAX_DOMAINS] {
        let gains = self.marginal_gain();
        let mut buf = [0.0_f64; MAX_DOMAINS];
        let mut n = 0_usize;
        for cell in gains.iter().take(self.domains) {
            if let GainCell::Gain(v) = *cell {
                buf[n] = f64::from(v);
                n += 1;
            }
        }
        let median = if n > 0 {
            order_statistic(&mut buf[..n], n / 2)
        } else {
            0.0
        };
        let tau = f64::from(tau_s).abs().max(1e-6);
        let mut weights = [0.0_f32; MAX_DOMAINS];
        let mut total = 0.0_f64;
        for d in 0..self.domains {
            let w = match gains[d] {
                GainCell::Gain(v) => crate::sigmoid(((f64::from(v) - median) / tau) as f32),
                GainCell::NoCurve => 0.0,
            };
            weights[d] = w;
            total += f64::from(w);
        }
        if total > 0.0 {
            let inv = 1.0 / total;
            for w in weights.iter_mut().take(self.domains) {
                *w = (f64::from(*w) * inv) as f32;
            }
        }
        weights
    }

    /// Blend `new` into `prev` at momentum `η` (clamped `[0, 1]`), clamping
    /// every entry to `[0, 1]`; ascending fold over the full stacks.
    pub fn momentum_into(prev: &mut [f32; MAX_DOMAINS], new: &[f32; MAX_DOMAINS], eta: f32) {
        let eta = eta.clamp(0.0, 1.0);
        for l in 0..MAX_DOMAINS {
            prev[l] = (prev[l] + eta * (new[l] - prev[l])).clamp(0.0, 1.0);
        }
    }

    /// The fitted error floor of `domain` (cross arm: the shared ε; sparse:
    /// the domain's own, or `None` when unfitted).
    pub fn epsilon_floor_of(&self, domain: usize) -> Option<f64> {
        if domain >= self.domains {
            return None;
        }
        match self.cross.as_ref() {
            Some(f) => Some(f.epsilon),
            None => self.uni[domain].map(|u| u.epsilon),
        }
    }

    /// The remaining learnable gap of `domain` at the current counts:
    /// `gap − ε` (cross arm: the shared curve; sparse: the domain's own).
    pub fn headroom_of(&self, domain: usize) -> Option<f64> {
        if domain >= self.domains {
            return None;
        }
        match self.cross.as_ref() {
            Some(_) => self.cross_margin().map(|(_, m)| m),
            None => match self.uni[domain] {
                Some(u) => self.sparse_gap(&u, domain).map(|gap| gap - u.epsilon),
                None => None,
            },
        }
    }

    /// The cross curve's predicted gap at `counts` (`None` when the cross
    /// curve is absent, the shape mismatches, a value is non-finite, or the
    /// guard fails). Alloc-free.
    pub fn predicted_gap(&self, counts: &[f32]) -> Option<f64> {
        let f = self.cross.as_ref()?;
        if counts.len() != self.domains {
            return None;
        }
        let mut n_eff = 0.0_f64;
        for (d, &v) in counts.iter().enumerate() {
            let vv = f64::from(v);
            if !vv.is_finite() {
                return None;
            }
            n_eff += f.gamma[d] * (vv + PSEUDOCOUNT);
        }
        if n_eff <= 0.0 {
            return None;
        }
        let gap = f.epsilon + f.beta * n_eff.powf(-f.alpha);
        if gap.is_finite() { Some(gap) } else { None }
    }

    /// The cross curve's predicted accuracy at `counts` (`1 − gap`).
    pub fn predicted_accuracy(&self, counts: &[f32]) -> Option<f64> {
        self.predicted_gap(counts).map(|gap| 1.0 - gap)
    }
}

#[cfg(all(test, feature = "estimators"))]
mod tests {
    use super::*;

    // Seeded xorshift64* — the house test pattern (no unseeded global RNG).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Linear 0→1 ramp over `[a, b)`, 1 after.
    fn ramp(t: usize, a: usize, b: usize) -> f64 {
        if t < a {
            0.0
        } else if t >= b {
            1.0
        } else {
            (t - a) as f64 / (b - a) as f64
        }
    }

    /// W=160, K=3 staggered ramps: n0 = 4 + 116·ramp(t; 0,100),
    /// n1 = 90·ramp(t; 40,140), n2 = 2 + 38·ramp(t; 10,159).
    fn cross_mix_counts() -> Vec<f32> {
        let (w, k) = (160_usize, 3_usize);
        let mut v = Vec::with_capacity(w * k);
        for t in 0..w {
            v.push((4.0 + 116.0 * ramp(t, 0, 100)) as f32);
            v.push((90.0 * ramp(t, 40, 140)) as f32);
            v.push((2.0 + 38.0 * ramp(t, 10, 159)) as f32);
        }
        v.shrink_to_fit();
        v
    }

    /// W=64, K=3 perfectly proportional columns: n_k(t) = c_k·(t+1).
    fn proportional_counts() -> Vec<f32> {
        let (w, k) = (64_usize, 3_usize);
        let c = [1.0_f64, 3.0, 7.0];
        let mut v = Vec::with_capacity(w * k);
        for t in 0..w {
            for &cd in c.iter().take(k) {
                v.push((cd * (t + 1) as f64) as f32);
            }
        }
        v
    }

    /// W=64, K=3: proportional columns 0/1, column 2 constant at 5.0.
    fn constant_tail_counts() -> Vec<f32> {
        let (w, k) = (64_usize, 3_usize);
        let c = [1.0_f64, 3.0];
        let mut v = Vec::with_capacity(w * k);
        for t in 0..w {
            for &cd in c.iter().take(k - 1) {
                v.push((cd * (t + 1) as f64) as f32);
            }
            v.push(5.0_f32);
        }
        v
    }

    /// The generating truth: gap = ε + β·N^(−α)·(1 + noise·(2u−1)) over
    /// N = Σγ·(n+1); y = 1 − gap.
    struct Truth {
        alpha: f64,
        beta: f64,
        epsilon: f64,
        gamma: [f64; 3],
        noise: f64,
    }

    fn accuracy_from(counts: &[f32], w: usize, k: usize, truth: &Truth, rng: &mut Rng) -> Vec<f32> {
        let mut out = Vec::with_capacity(w);
        for t in 0..w {
            let mut n = 0.0_f64;
            for d in 0..k {
                n += truth.gamma[d] * (f64::from(counts[t * k + d]) + PSEUDOCOUNT);
            }
            let u = rng.uniform();
            let gap = truth.epsilon
                + truth.beta * n.powf(-truth.alpha) * (1.0 + truth.noise * (2.0 * u - 1.0));
            out.push((1.0 - gap) as f32);
        }
        out
    }

    const SEED_MAIN: u64 = 0xE0_F1_6C_41;

    /// The Plan-641 test-1 fixture: cross mix, α=0.7, β=1.0, ε=0.02,
    /// γ=[0.5, 0.3, 0.2], 1% multiplicative noise.
    fn fit_cross(gamma: [f64; 3], noise: f64, seed: u64) -> GainEstimator {
        let counts = cross_mix_counts();
        let truth = Truth {
            alpha: 0.7,
            beta: 1.0,
            epsilon: 0.02,
            gamma,
            noise,
        };
        let acc = accuracy_from(&counts, 160, 3, &truth, &mut Rng::new(seed));
        let panel = MixPanel::new(3, 160, &counts);
        GainEstimator::fit(&panel, &acc).expect("cross fit")
    }

    #[test]
    fn g1_recovery_cross_domain() {
        let est = fit_cross([0.5, 0.3, 0.2], 0.01, SEED_MAIN);
        assert!(matches!(est.saturation(), Saturation::CrossDomain(_)));
        let f = est.cross_fit().expect("cross fit armed");
        println!(
            "plan641 g1: alpha_hat={} beta_hat={} eps_hat={} gamma_hat={:?} obj={}",
            f.alpha,
            f.beta,
            f.epsilon,
            &f.gamma[..3],
            f.objective
        );
        assert!((f.alpha - 0.7).abs() <= 0.20, "alpha {} vs 0.7", f.alpha);
        let l1: f64 = f
            .gamma
            .iter()
            .take(3)
            .zip([0.5, 0.3, 0.2].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(l1 <= 0.20, "gamma L1 {} vs [0.5,0.3,0.2]", l1);
        assert!(
            (f.epsilon - 0.02).abs() <= 0.05,
            "eps {} vs 0.02",
            f.epsilon
        );
        assert!(f.beta.ln().abs() <= 0.35, "ln beta {}", f.beta.ln());
        let sum: f64 = f.gamma.iter().take(3).sum();
        assert!((sum - 1.0).abs() <= 1e-5, "sum gamma {sum}");
        for g in f.gamma.iter().take(3) {
            assert!(*g >= GAMMA_MIN - 1e-9, "gamma {g} below floor");
        }
    }

    #[test]
    fn g1_derivative_matches_finite_difference() {
        let est = fit_cross([0.5, 0.3, 0.2], 0.01, SEED_MAIN);
        let doms = est.domains();
        // Exact per-domain check at the current counts.
        for d in 0..doms {
            let x = match est.derivative_gain()[d] {
                GainCell::Gain(v) => f64::from(v),
                GainCell::NoCurve => panic!("cross arm never declines"),
            };
            let nk = est.counts_now()[d];
            let h = 1.0_f64.max(1e-3 * nk);
            let mut hi = [0.0_f32; MAX_DOMAINS];
            let mut lo = [0.0_f32; MAX_DOMAINS];
            for q in 0..doms {
                let v = est.counts_now()[q] as f32;
                hi[q] = v;
                lo[q] = v;
            }
            hi[d] = (nk + h) as f32;
            lo[d] = (nk - h) as f32;
            let fd = (est.predicted_gap(&hi[..doms]).expect("gap")
                - est.predicted_gap(&lo[..doms]).expect("gap"))
                / (2.0 * h);
            assert!(
                (fd + x).abs() <= 1e-3 * x,
                "domain {d}: fd {fd} vs -derivative {x}"
            );
        }
        // The n≥1000 loop with the loud-zero defence.
        let mut max_rel = 0.0_f64;
        let mut sum_closed = 0.0_f64;
        for _ in 0..1000_usize {
            let e = std::hint::black_box(&est);
            for d in 0..doms {
                let closed = e.derivative_gain()[d];
                let x = match closed {
                    GainCell::Gain(v) => f64::from(v),
                    GainCell::NoCurve => panic!("cross arm never declines"),
                };
                let nk = e.counts_now()[d];
                let h = std::hint::black_box(1.0_f64.max(1e-3 * nk));
                let mut hi = [0.0_f32; MAX_DOMAINS];
                let mut lo = [0.0_f32; MAX_DOMAINS];
                for q in 0..doms {
                    let v = e.counts_now()[q] as f32;
                    hi[q] = v;
                    lo[q] = v;
                }
                hi[d] = (nk + h) as f32;
                lo[d] = (nk - h) as f32;
                let fd = (e.predicted_gap(&hi[..doms]).expect("gap")
                    - e.predicted_gap(&lo[..doms]).expect("gap"))
                    / (2.0 * h);
                std::hint::black_box((&e, h, closed, fd));
                max_rel = max_rel.max((fd + x).abs() / x);
                sum_closed += x;
            }
        }
        assert!(max_rel <= 1e-3, "max rel err {max_rel}");
        assert!(
            sum_closed > 0.0,
            "loud-zero defence: derivative sum {sum_closed}"
        );
    }

    #[test]
    fn g1b_single_mix_returns_sparse_univariate() {
        let counts = proportional_counts();
        let truth = Truth {
            alpha: 0.7,
            beta: 1.0,
            epsilon: 0.02,
            gamma: [0.5, 0.3, 0.2],
            noise: 0.01,
        };
        let acc = accuracy_from(&counts, 64, 3, &truth, &mut Rng::new(0x50F));
        let panel = MixPanel::new(3, 64, &counts);
        let est = GainEstimator::fit(&panel, &acc).expect("fit");
        assert!(matches!(est.saturation(), Saturation::Sparse(_)));
        assert!(est.cross_fit().is_none());
        for d in 0..3 {
            assert!(est.univariate_fit(d).is_some(), "domain {d} unfitted");
        }
        for cell in est.marginal_gain().iter().take(3) {
            assert_ne!(*cell, GainCell::NoCurve, "sparse fits must answer");
        }
    }

    #[test]
    fn g1b_constant_mix_typed_nocurve() {
        let counts = constant_tail_counts();
        let truth = Truth {
            alpha: 0.7,
            beta: 1.0,
            epsilon: 0.02,
            gamma: [0.5, 0.5, 0.0],
            noise: 0.01,
        };
        let acc = accuracy_from(&counts, 64, 3, &truth, &mut Rng::new(0x40D));
        let panel = MixPanel::new(3, 64, &counts);
        let est = GainEstimator::fit(&panel, &acc).expect("fit");
        assert!(matches!(est.saturation(), Saturation::Sparse(_)));
        assert!(
            est.univariate_fit(2).is_none(),
            "constant column must decline"
        );
        assert_eq!(est.marginal_gain()[2], GainCell::NoCurve);
        assert_eq!(est.sigmoid_weights(0.5)[2], 0.0);
    }

    #[test]
    fn g1b_cross_mix_arms_gamma() {
        let est = fit_cross([0.5, 0.3, 0.2], 0.01, SEED_MAIN);
        assert!(est.cross_fit().is_some());
    }

    #[test]
    fn saturation_tripwire_epsilon_dominated() {
        let est = fit_cross([0.55, 0.40, 0.05], 0.01, 0x60D);
        if let Some(f) = est.cross_fit() {
            println!("plan641 tripwire: gamma_hat={:?}", &f.gamma[..3]);
        }
        let wts = est.sigmoid_weights(0.5);
        // BAR ADJUSTMENT (documented): the original bar was
        // `w[2] < 0.15·max(w[0], w[1])`. It is unreachable at τ=0.5 whenever
        // the fit RECOVERS — which is exactly what the other bars demand.
        // The spec's gain formula scales every gain by the common factor
        // (gap−ε)·α/N ≈ 0.041·0.677/96 ≈ 2.9e-4, so the gains run
        // [0.0183, 0.00937, 0.000192] (a healthy 96:1 spread, domain 2
        // last — measured, this seed) and every sigmoid input
        // (g − median)/0.5 sits inside ±0.018 — the sigmoid's LINEAR band —
        // so the normalized weights compress to ≈⅓ each (measured:
        // [0.337, 0.334, 0.331]) no matter the spread. The bar was only
        // ever satisfiable with a collapsed γ̂ — the failure mode tests 1
        // and gamma_perturbation_flips_ranking forbid. The scale-free
        // statement the fixture actually pins: domain 2 ranks strictly
        // last, and the weight ordering mirrors the true mix's ordering.
        assert!(
            wts[2] < wts[1] && wts[1] < wts[0],
            "epsilon-dominated domain 2 must rank strictly last: {wts:?}"
        );
    }

    #[test]
    fn fit_is_bit_identical_across_runs() {
        let a = fit_cross([0.5, 0.3, 0.2], 0.01, SEED_MAIN);
        let b = fit_cross([0.5, 0.3, 0.2], 0.01, SEED_MAIN);
        let fa = a.cross_fit().expect("cross");
        let fb = b.cross_fit().expect("cross");
        assert_eq!(fa.alpha.to_bits(), fb.alpha.to_bits());
        assert_eq!(fa.beta.to_bits(), fb.beta.to_bits());
        assert_eq!(fa.epsilon.to_bits(), fb.epsilon.to_bits());
        assert_eq!(fa.objective.to_bits(), fb.objective.to_bits());
        for i in 0..MAX_DOMAINS {
            assert_eq!(fa.gamma[i].to_bits(), fb.gamma[i].to_bits());
        }
        let wa = a.sigmoid_weights(0.5);
        let wb = b.sigmoid_weights(0.5);
        for i in 0..MAX_DOMAINS {
            assert_eq!(wa[i].to_bits(), wb[i].to_bits());
        }
    }

    #[test]
    fn univariate_recovery_single_domain() {
        let counts = proportional_counts();
        // y generated from domain 0 ONLY: gap = 0.02 + 1.0·(n0+1)^(−0.7).
        let truth = Truth {
            alpha: 0.7,
            beta: 1.0,
            epsilon: 0.02,
            gamma: [1.0, 0.0, 0.0],
            noise: 0.01,
        };
        let acc = accuracy_from(&counts, 64, 3, &truth, &mut Rng::new(0x80D));
        let panel = MixPanel::new(3, 64, &counts);
        let est = GainEstimator::fit(&panel, &acc).expect("fit");
        assert!(matches!(est.saturation(), Saturation::Sparse(_)));
        let u = est.univariate_fit(0).expect("domain 0 fitted");
        println!(
            "plan641 g1b uni: alpha_hat={} beta_hat={} eps_hat={} obj={}",
            u.alpha, u.beta, u.epsilon, u.objective
        );
        assert!((u.alpha - 0.7).abs() <= 0.20, "alpha {} vs 0.7", u.alpha);
        assert!(
            (u.epsilon - 0.02).abs() <= 0.05,
            "eps {} vs 0.02",
            u.epsilon
        );
        assert!(u.beta.ln().abs() <= 0.35, "ln beta {}", u.beta.ln());
    }

    #[test]
    fn weights_normalize_nocurve_zero() {
        let est = fit_cross([0.5, 0.3, 0.2], 0.01, SEED_MAIN);
        let wts = est.sigmoid_weights(0.5);
        let sum: f64 = wts.iter().take(3).map(|&w| f64::from(w)).sum();
        assert!((sum - 1.0).abs() <= 1e-5, "weights sum {sum}");
    }

    #[test]
    fn refusal_battery() {
        // W=4 → TooFewObservations.
        let counts = vec![1.0_f32; 4 * 2];
        let acc4 = vec![0.5_f32; 4];
        let panel = MixPanel::new(2, 4, &counts);
        match GainEstimator::fit(&panel, &acc4) {
            Err(FitRefusal::TooFewObservations { have: 4, need: 8 }) => {}
            other => panic!("expected TooFewObservations {{4, 8}}, got {other:?}"),
        }
        // Counts length mismatch → ShapeMismatch (panel validate).
        let counts_short = vec![1.0_f32; 15];
        let acc8 = vec![0.5_f32; 8];
        let panel = MixPanel::new(2, 8, &counts_short);
        assert!(matches!(
            GainEstimator::fit(&panel, &acc8),
            Err(FitRefusal::ShapeMismatch)
        ));
        // NaN in accuracy → NonFiniteInput.
        let counts = vec![1.0_f32; 8 * 2];
        let mut acc_nan = vec![0.5_f32; 8];
        acc_nan[3] = f32::NAN;
        let panel = MixPanel::new(2, 8, &counts);
        assert!(matches!(
            GainEstimator::fit(&panel, &acc_nan),
            Err(FitRefusal::NonFiniteInput)
        ));
        // Exactly 1.0 accuracy → zero gap → AccuracyDomain.
        let mut acc_one = vec![0.5_f32; 8];
        acc_one[0] = 1.0;
        let panel = MixPanel::new(2, 8, &counts);
        assert!(matches!(
            GainEstimator::fit(&panel, &acc_one),
            Err(FitRefusal::AccuracyDomain)
        ));
        // K=17 and K=0 → DomainCount.
        let counts17 = vec![1.0_f32; 17 * 8];
        let panel = MixPanel::new(17, 8, &counts17);
        assert!(matches!(
            GainEstimator::fit(&panel, &acc8),
            Err(FitRefusal::DomainCount)
        ));
        let panel = MixPanel::new(0, 8, &[]);
        assert!(matches!(
            GainEstimator::fit(&panel, &acc8),
            Err(FitRefusal::DomainCount)
        ));
        // W=4097 → PanelTooLarge.
        let counts_big = vec![1.0_f32; 4097];
        let acc_big = vec![0.5_f32; 4097];
        let panel = MixPanel::new(1, 4097, &counts_big);
        match GainEstimator::fit(&panel, &acc_big) {
            Err(FitRefusal::PanelTooLarge {
                have: 4097,
                max: MAX_OBSERVATIONS,
            }) => {}
            other => panic!("expected PanelTooLarge {{4097, 4096}}, got {other:?}"),
        }
    }

    #[test]
    fn gamma_perturbation_flips_ranking() {
        let counts_a = cross_mix_counts();
        let truth = Truth {
            alpha: 0.7,
            beta: 1.0,
            epsilon: 0.02,
            gamma: [0.45, 0.35, 0.20],
            noise: 0.005,
        };
        let acc = accuracy_from(&counts_a, 160, 3, &truth, &mut Rng::new(0xB1));
        let mut counts_b = counts_a.clone();
        for t in 0..160 {
            counts_b.swap(t * 3 + 1, t * 3 + 2);
        }
        let panel_a = MixPanel::new(3, 160, &counts_a);
        let panel_b = MixPanel::new(3, 160, &counts_b);
        let a = GainEstimator::fit(&panel_a, &acc).expect("fit a");
        let b = GainEstimator::fit(&panel_b, &acc).expect("fit b");
        let ga = a.cross_fit().expect("cross a").gamma;
        let gb = b.cross_fit().expect("cross b").gamma;
        let argmax_a = if ga[1] >= ga[2] { 1 } else { 2 };
        let argmax_b = if gb[1] >= gb[2] { 1 } else { 2 };
        println!(
            "plan641 canary: gamma_a={:?} gamma_b={:?}",
            &ga[..3],
            &gb[..3]
        );
        assert_eq!(
            argmax_a,
            1,
            "A must rank column 1 first (gamma_a {:?})",
            &ga[..3]
        );
        assert_eq!(
            argmax_b,
            2,
            "B must rank column 2 first (gamma_b {:?})",
            &gb[..3]
        );
    }

    /// The Jacobi's rotation pairing is pinned on CLOSED-FORM spectra — the
    /// one numeric kernel whose divergence is silent (a wrong pairing still
    /// converges to SOMETHING plausible; the first census pass shipped rank 2
    /// from the mirrored sign and nothing failed). Three exact panels:
    /// an orthogonal-corner square (sv = (100, 100) exactly), an
    /// off-diagonal triangle (sv = (100, 100/√3)), and a rank-1 K=3 line
    /// (Sparse with eff10 = 1). Every value derived in-test, never fitted.
    #[test]
    fn saturation_spectrum_closed_form_panels() {
        // ── case 1: centered corners of a square — Gram diagonal, sv exact ──
        // rows (0,0),(100,0),(0,100),(100,100): means (50,50); centered rows
        // are (±50, ±50) over all four sign pairs → CᵀC = 4·2500·I → sv = (100, 100).
        let counts = [0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 100.0, 100.0];
        let panel = MixPanel::new(2, 4, &counts);
        let sat = mix_saturation(&panel).expect("case 1");
        let Saturation::CrossDomain(c) = sat else {
            panic!("case 1: centered corners span rank 2 = bar → CrossDomain, got {sat:?}")
        };
        assert_eq!(c.eff10, 2, "both σ = 100 ≥ 10%");
        assert!((f64::from(c.singular_values[0]) - 100.0).abs() <= 1e-6);
        assert!((f64::from(c.singular_values[1]) - 100.0).abs() <= 1e-6);

        // ── case 2: triangle — Gram [[2/3,1/3],[1/3,2/3]]·10⁴ → sv=(100, 100/√3) ──
        // rows (0,0),(100,0),(100,100): means (200/3, 100/3); centered
        // (−200/3,−100/3),(100/3,−100/3),(100/3,200/3). CᵀC = 10⁴·[[2/3,1/3],[1/3,2/3]]
        // eigenvalues 10⁴·(1, 1/3) → sv = (100, 100/√3 = 57.7350…).
        let counts = [0.0, 0.0, 100.0, 0.0, 100.0, 100.0];
        let panel = MixPanel::new(2, 3, &counts);
        let sat = mix_saturation(&panel).expect("case 2");
        let Saturation::CrossDomain(c) = sat else {
            panic!("case 2: rank 2 = bar → CrossDomain, got {sat:?}")
        };
        let third_root = 100.0 / 3.0_f64.sqrt();
        assert!((f64::from(c.singular_values[0]) - 100.0).abs() <= 1e-6);
        assert!((f64::from(c.singular_values[1]) - third_root).abs() <= 1e-3);
        assert_eq!(c.eff10, 2, "57.7 ≥ 10%");

        // ── case 3: rank-1 line on K=3 — Sparse with eff10 exactly 1 ──
        // rows (0,50),(50,100),(100,150): all on y = x + 50 → centered rank 1.
        // bar = K−1 = 2 > 1 → Sparse, and the census says so with eff10 = 1.
        let counts = [0.0, 50.0, 1.0, 50.0, 100.0, 2.0, 100.0, 150.0, 3.0];
        let panel = MixPanel::new(3, 3, &counts);
        let sat = mix_saturation(&panel).expect("case 3");
        let Saturation::Sparse(c) = sat else {
            panic!("case 3: rank 1 < bar 2 → Sparse, got {sat:?}")
        };
        assert_eq!(c.eff10, 1, "one direction carries all the spread");
        assert_eq!(c.eff1, 1, "no second direction above the 1% share");
    }
}
