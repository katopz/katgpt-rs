//! Hint-Regret VoI primitive — paired-rollout value-of-information estimation.
//!
//! **Plan 576** · Research 496 (SPADE, arXiv:2608.19197) · Research 500
//! (EnvHarness wrapper composition) · game-side guide: riir-ai
//! `.research/340` (frontier curriculum).
//!
//! The distilled modelless question this module answers: *how much would one
//! hint (a demonstration, a revealed best arm, a demo path) improve the
//! agent's return on this content?* That quantity — the hint regret `r̂` — is
//! the discriminator a curriculum needs to separate content that is
//! **learnable-hard** (hint unlocks it → offer it) from content that is
//! **mastered** (hint adds nothing, agent already succeeds) or
//! **intractable** (hint adds nothing, agent fails anyway → retire it).
//! Static difficulty heuristics cannot make this distinction; the paired arm
//! can, at ~µs scale, with zero model weights on the hot path.
//!
//! # Composition
//!
//! - [`HintRegretEstimator`] (this file) — Phase 1: the paired estimator.
//! - [`gate`] — Phase 2: sigmoid band-pass difficulty gate + three-regime
//!   triage + Wilson CI on the learnable-share statistic.
//! - [`memory`] — Phase 3: regret-scored content memory (salience decay,
//!   oldest-first + absorbing-intractable eviction) and Beta-LCB frontier
//!   ordering over [`crate::best_belief`] (DRY consume, not re-implementation).
//!
//! # The paired contract (common random numbers)
//!
//! Each pair is one content instance rolled out twice under the **same
//! random seed** (CRN — common random numbers): arm A with the hint, arm B
//! without. The estimator is pure arithmetic over the recorded pairs — the
//! CALLER owns the rollouts and the seed pairing. CRN is load-bearing: the
//! shared noise source cancels in the paired difference, so the empirical
//! variance of `r̂` collapses relative to independent-seed estimation
//! (pinned by the G2 gate: variance ratio ≥ 2×).
//!
//! Sign convention (Guide 340 §"distilled modelless primitive", resolved
//! against the landed consumer — riir-mmorpg-examples `frontier_regime_of`):
//! `r̂ = mean(hinted) − mean(unhinted)` = **the regret the unhinted agent
//! pays for missing the hint** = the value of the hint. High `r̂` → the hint
//! unlocks a lot → frontier content. (The guide's prose lists the arms
//! "with-hint / without-hint" and writes `mean(B) − mean(A)`, which under
//! that listing yields the hint's *negative*; the triage direction
//! `r̂ ≥ τ_r → Frontier` only makes sense with the consumer's gain sign, so
//! the consumer wins — the primitive must supersede the inline collapse
//! *semantically*, not just generically.)
//!
//! # Confidence machinery (honesty notes)
//!
//! - The **Hoeffding** half-width uses only the known per-arm return range
//!   `[lo, hi]` (difference range `2·(hi−lo)`). It is distribution-free and
//!   CRN-invariant — the guarantee cannot see the variance reduction. It is
//!   the named schedule: `K(ε,δ) = ⌈(b−a)²/(2ε²)·ln(2/δ)⌉` pairs suffice.
//! - The **empirical-Bernstein** half-width ([`RegretEstimate::eb_half_width`])
//!   adapts to the sample variance and therefore DOES capture the CRN win,
//!   at the cost of an additive `O((b−a)/n)` term (Maurer & Pontil 2009).
//!   Callers whose pairs are CRN-shared may stop on it once `n` is large
//!   enough that the additive term is small.
//! - The **CLT** half-width (`1.96·s/√n`) is exposed for diagnostics only —
//!   no finite-sample guarantee.
//!
//! Zero-allocation steady state: the estimator is three Welford
//! accumulators (O(1) memory, no heap); all outputs are by-value structs.
//!
//! hint_regret — paired-rollout value-of-information estimation (Plan 576,
//! Research 496 SPADE arXiv:2608.19197 / Research 500 EnvHarness, game-side
//! guide riir-ai .research/340). The frontier-curriculum discriminator: how
//! much would ONE hint (demo / revealed arm) improve the return on this
//! content? Paired CRN estimator (Welford, zero-alloc) + Hoeffding /
//! empirical-Bernstein machinery + sigmoid band gate + three-regime triage
//! + Beta-LCB frontier ordering (DRY over best_belief) + regret-scored
//! memory with absorbing-intractable eviction. Opt-in pending the GOAT
//! verdict; consumers: riir-ai Guide 340 P0 (supersedes the mmorpg inline
//! collapse), riir-train Plan 346 arena opponent selection.

pub mod gate;
pub mod memory;

#[cfg(test)]
mod tests;

pub use gate::{Regime, learnable_band_gate, triage, wilson_score_ci};
pub use memory::{
    FRONTIER_EPSILON, ObserveOutcome, RegretMemory, RegretMemoryEntry, beta_lcb, beta_lcb_order,
    beta_lcb_order_into, salience,
};

/// Per-arm return bounds used by the range-based (Hoeffding) machinery.
///
/// Both arms must return values in `[lo, hi]`; the paired difference then
/// lies in `[lo − hi, hi − lo]` (range `2·(hi − lo)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReturnBounds {
    /// Inclusive lower bound on a single arm's return.
    pub lo: f32,
    /// Inclusive upper bound on a single arm's return.
    pub hi: f32,
}

impl ReturnBounds {
    /// Constructs bounds, asserting the only invariant the estimator needs:
    /// `lo < hi` and both finite. Degenerate (`lo == hi`) bounds make every
    /// difference exactly zero — pointless but not unsound; rejected to catch
    /// caller wiring mistakes early.
    #[inline]
    pub fn new(lo: f32, hi: f32) -> Self {
        assert!(
            lo < hi && lo.is_finite() && hi.is_finite(),
            "ReturnBounds requires lo < hi, both finite (got lo={lo}, hi={hi})"
        );
        Self { lo, hi }
    }

    /// Range of the paired difference `B − A`: `2·(hi − lo)`.
    #[inline]
    pub fn diff_range(&self) -> f32 {
        2.0 * (self.hi - self.lo)
    }
}

/// Hoeffding sample schedule: the number of pairs after which the guaranteed
/// CI half-width on `r̂` is ≤ `eps` at confidence `1 − delta`.
///
/// `K(ε,δ) = ⌈(b−a)²/(2ε²)·ln(2/δ)⌉` where `(b−a)` is the paired-difference
/// range `2·(hi − lo)`. Returns at least 1.
#[inline]
pub fn hoeffding_k(eps: f32, delta: f32, bounds: ReturnBounds) -> u32 {
    debug_assert!(eps > 0.0, "eps must be positive");
    debug_assert!(delta > 0.0 && delta < 1.0, "delta must be in (0,1)");
    let range = bounds.diff_range() as f64;
    let k = (range * range / (2.0 * eps as f64 * eps as f64)) * (2.0f64 / delta as f64).ln();
    k.ceil().max(1.0) as u32
}

/// Guaranteed (Hoeffding) half-width on `|r̂ − E[r]|` after `n` pairs at
/// confidence `1 − delta`: `(b−a)·√(ln(2/δ)/(2n))`.
///
/// Distribution-free and CRN-invariant — see the module honesty notes. `n`
/// is clamped to ≥ 1.
#[inline]
pub fn hoeffding_half_width(n: u32, delta: f32, bounds: ReturnBounds) -> f32 {
    debug_assert!(delta > 0.0 && delta < 1.0, "delta must be in (0,1)");
    let n = n.max(1) as f64;
    let range = bounds.diff_range() as f64;
    (range * ((2.0f64 / delta as f64).ln() / (2.0 * n)).sqrt()) as f32
}

/// Empirical-Bernstein half-width (Maurer & Pontil 2009, Theorem 4 with
/// `B = (b−a)/2`): variance-adaptive AND finite-sample valid, at the cost of
/// an additive `7(b−a)ln(2/δ)/(3(n−1))` term.
///
/// This is the bound that SEES the CRN variance reduction: under
/// common-random-number pairing the sample variance collapses and this
/// half-width shrinks with it, while the pure Hoeffding bound stays pinned
/// to the range. Requires `n ≥ 2`; returns `f32::MAX` for smaller `n`
/// (uninformative, never a false stop).
#[inline]
pub fn empirical_bernstein_half_width(
    n: u32,
    sample_variance: f32,
    delta: f32,
    bounds: ReturnBounds,
) -> f32 {
    debug_assert!(delta > 0.0 && delta < 1.0, "delta must be in (0,1)");
    if n < 2 {
        return f32::MAX;
    }
    let n = n as f64;
    let range = bounds.diff_range() as f64;
    let ln = (2.0f64 / delta as f64).ln();
    let sqrt_term = (2.0 * sample_variance as f64 * ln / n).sqrt();
    let add_term = 7.0 * range * ln / (3.0 * (n - 1.0));
    (sqrt_term + add_term) as f32
}

/// The paired-regret estimate: one pass of pure arithmetic over the
/// accumulated pairs.
///
/// `r_hat` follows the module sign convention (mean(hinted) −
/// mean(unhinted) = the regret of missing the hint = the hint's value).
/// `arm_means` is `(mean_hinted, mean_unhinted)` so consumers can recover
/// both sides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegretEstimate {
    /// Point estimate of the hint regret (the hint's value / the unhinted
    /// agent's regret from missing it): `mean(hinted) − mean(unhinted)`.
    pub r_hat: f32,
    /// Guaranteed Hoeffding half-width at the `delta` passed to
    /// [`HintRegretEstimator::estimate`]. Range-based, CRN-invariant.
    pub ci_half_width: f32,
    /// Variance-adaptive empirical-Bernstein half-width (finite-sample
    /// valid; captures the CRN win). `f32::MAX` below 2 pairs.
    pub eb_half_width: f32,
    /// CLT half-width `1.96·s/√n` — diagnostic only, no finite-sample
    /// guarantee. `f32::MAX` below 2 pairs.
    pub empirical_half_width: f32,
    /// Pairs recorded so far.
    pub n_pairs: u32,
    /// `(mean_hinted, mean_unhinted)`.
    pub arm_means: (f32, f32),
}

impl RegretEstimate {
    /// Sequential-stopping check: has the GUARANTEED (Hoeffding) half-width
    /// tightened to `eps`? Equivalent to `n_pairs ≥ hoeffding_k(eps, δ)`.
    ///
    /// This is the plan's named stopping rule — the conservative one. A
    /// caller whose pairs are CRN-shared may instead stop on
    /// [`eb_half_width`](Self::eb_half_width) ≤ eps once the additive term
    /// is small; that is a documented, deliberate loosening.
    #[inline]
    pub fn should_stop(&self, eps: f32) -> bool {
        self.ci_half_width <= eps
    }
}

/// Paired hint-regret estimator — three Welford accumulators (arm A, arm B,
/// difference), O(1) memory, zero heap.
///
/// The caller runs the rollouts (same solver, same seed, two arms) and
/// records each pair via [`record_pair`](Self::record_pair). All estimates
/// are single-pass arithmetic over the accumulators — allocation-free by
/// construction.
///
/// ```ignore
/// use katgpt_core::hint_regret::{HintRegretEstimator, ReturnBounds};
/// let mut est = HintRegretEstimator::new(ReturnBounds::new(0.0, 1.0));
/// for seed in 0..64 {
///     let (a, b) = run_pair(seed); // caller: arm A hinted, arm B unhinted, SAME seed
///     est.record_pair(a, b);
/// }
/// let e = est.estimate(0.05);
/// if e.should_stop(0.1) { /* CI tight enough — stop sampling */ }
/// ```
#[derive(Debug, Clone)]
pub struct HintRegretEstimator {
    bounds: ReturnBounds,
    n: u32,
    // Welford accumulators, f64 for accumulation stability (f32 outputs).
    mean_a: f64,
    m2_a: f64,
    mean_b: f64,
    m2_b: f64,
    mean_d: f64,
    m2_d: f64,
}

impl HintRegretEstimator {
    /// New estimator over returns bounded by `bounds`.
    #[inline]
    pub fn new(bounds: ReturnBounds) -> Self {
        Self {
            bounds,
            n: 0,
            mean_a: 0.0,
            m2_a: 0.0,
            mean_b: 0.0,
            m2_b: 0.0,
            mean_d: 0.0,
            m2_d: 0.0,
        }
    }

    /// The bounds this estimator was constructed with.
    #[inline]
    pub fn bounds(&self) -> ReturnBounds {
        self.bounds
    }

    /// Pairs recorded so far.
    #[inline]
    pub fn n_pairs(&self) -> u32 {
        self.n
    }

    /// Records one paired rollout: `hinted` = arm A's return (with the
    /// hint), `unhinted` = arm B's return (without), SAME underlying seed.
    ///
    /// Values are clamped to the declared bounds before accumulation — an
    /// out-of-range return is a caller wiring bug, and clamping keeps the
    /// range-based guarantees honest rather than silently violated.
    #[inline]
    pub fn record_pair(&mut self, hinted: f32, unhinted: f32) {
        let a = hinted.clamp(self.bounds.lo, self.bounds.hi) as f64;
        let b = unhinted.clamp(self.bounds.lo, self.bounds.hi) as f64;
        // Module sign convention: d = hinted − unhinted (the hint's gain =
        // the unhinted agent's regret from missing the hint).
        let d = a - b;
        self.n = self.n.saturating_add(1);
        let n = self.n as f64;

        // Welford update, three parallel streams (A, B, difference).
        let da = a - self.mean_a;
        self.mean_a += da / n;
        self.m2_a += da * (a - self.mean_a);

        let db = b - self.mean_b;
        self.mean_b += db / n;
        self.m2_b += db * (b - self.mean_b);

        let dd = d - self.mean_d;
        self.mean_d += dd / n;
        self.m2_d += dd * (d - self.mean_d);
    }

    /// Sample variance of the paired differences (unbiased, `n−1`
    /// denominator). Zero for `n < 2`.
    #[inline]
    pub fn diff_sample_variance(&self) -> f32 {
        if self.n < 2 {
            return 0.0;
        }
        (self.m2_d / (self.n as f64 - 1.0)) as f32
    }

    /// One-pass estimate at confidence `1 − delta` (delta ∈ (0,1)).
    ///
    /// With zero pairs returns the uninformative estimate
    /// (`r_hat = 0`, all half-widths `f32::MAX`).
    pub fn estimate(&self, delta: f32) -> RegretEstimate {
        debug_assert!(delta > 0.0 && delta < 1.0, "delta must be in (0,1)");
        if self.n == 0 {
            return RegretEstimate {
                r_hat: 0.0,
                ci_half_width: f32::MAX,
                eb_half_width: f32::MAX,
                empirical_half_width: f32::MAX,
                n_pairs: 0,
                arm_means: (0.0, 0.0),
            };
        }
        let var = self.diff_sample_variance();
        RegretEstimate {
            r_hat: self.mean_d as f32,
            ci_half_width: hoeffding_half_width(self.n, delta, self.bounds),
            eb_half_width: empirical_bernstein_half_width(self.n, var, delta, self.bounds),
            empirical_half_width: if self.n < 2 {
                f32::MAX
            } else {
                (1.959_963_984_540_054 * (var as f64 / self.n as f64).sqrt()) as f32
            },
            n_pairs: self.n,
            arm_means: (self.mean_a as f32, self.mean_b as f32),
        }
    }
}
