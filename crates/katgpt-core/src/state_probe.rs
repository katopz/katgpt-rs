//! state_probe — the FlyBy decision-layer probe kernel (Plan 621 Phase 1).
//!
//! Given a plain ensemble outcome over one decision state — `n` sampled
//! continuations, `pass_count` of which reached a verified answer, and the
//! histogram of the answers they produced — this computes the statistics the
//! Phase-2 bottleneck classifier consumes:
//!
//! * `v_hat = pass_count / n` — the pass-fraction point estimate `V(s)`
//!   (FlyBy: "productive reasoning moves toward `(V, H) = (1, 0)`");
//! * `h_mm` — the answer entropy `H(s)` in nats with the Miller–Madow bias
//!   correction `(K′ − 1) / (2N)`, where `K′` is the number of NONZERO bins
//!   (the observed-support reading of Miller 1955; on a fully populated
//!   histogram `K′` equals the alphabet size, which is Research 609's
//!   `(K − 1)/(2N)` form and the fixture's case);
//! * the Wilson score interval on `v_hat` — **CONSUMED from
//!   [`crate::hint_regret::wilson_score_ci`]**, never a third copy
//!   (Research 609's DRY pin; the byte-identical `speculative::qmc`
//!   bootstrap twin is recorded there as a pre-existing observation).
//!
//! NOT calibrated UQ: the interval is a binomial CI on the ensemble
//! pass-rate, nothing more. If a consumer treats it as calibrated
//! uncertainty, the Report-the-Floor extension binds at that consumer's
//! re-gate (Plan 621 GOAT summary). Reflex-free by construction: the kernel
//! takes plain counts and emits plain floats — no prompt, no model, no
//! sampling.
//!
//! Zero-alloc: borrows the caller's histogram slice; no collections.

use crate::hint_regret::wilson_score_ci;

/// The ensemble outcome over one decision state (plain counts — the caller
/// owns the sampling; the kernel never generates continuations).
#[derive(Debug, Clone, Copy)]
pub struct ProbeInput<'a> {
    /// Ensemble members whose continuation reached a verified answer.
    pub pass_count: u32,
    /// Total ensemble members (`n` in `V(s) = k/n`).
    pub n: u32,
    /// Answer histogram over the ensemble (index = answer class). Should
    /// summarize the SAME ensemble as `n` when answers are tracked; an empty
    /// slice means answers-not-tracked and yields `h_mm = 0.0`.
    pub histogram: &'a [u16],
}

/// The probe's read over one state — plain floats, no policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeEstimate {
    /// `V̂ = pass_count / n` (0.0 at `n = 0`).
    pub v_hat: f64,
    /// Miller–Madow answer entropy `Ĥ = H_emp + (K′−1)/(2N)` in nats
    /// (0.0 at `N = 0`).
    pub h_mm: f64,
    /// Wilson score interval on `v_hat` (substrate-computed; `(0.0, 1.0)`
    /// at `n = 0`).
    pub wilson_lo: f64,
    /// Wilson score interval upper bound.
    pub wilson_hi: f64,
}

/// Probe one state. `z` is the two-sided Wilson critical value
/// (`1.959963984540054` for 95%) — passed straight through to the substrate.
#[inline]
pub fn probe(input: &ProbeInput<'_>, z: f64) -> ProbeEstimate {
    let v_hat = if input.n == 0 {
        0.0
    } else {
        f64::from(input.pass_count) / f64::from(input.n)
    };
    let (wilson_lo, wilson_hi) = wilson_score_ci(v_hat, u64::from(input.n), z);
    ProbeEstimate {
        v_hat,
        h_mm: miller_madow_entropy(input.histogram),
        wilson_lo,
        wilson_hi,
    }
}

/// Miller–Madow entropy over a count histogram, nats.
///
/// `H_emp = ln N − (Σ cᵢ ln cᵢ)/N` over nonzero bins (algebraically the
/// plug-in `−Σ pᵢ ln pᵢ`, one division fewer), plus the Miller–Madow
/// correction `(K′ − 1)/(2N)` with `K′` = number of NONZERO bins. `N = 0`
/// (empty histogram) → 0.0.
///
/// Why nonzero bins and not the raw alphabet length: the correction
/// compensates for classes the sample did NOT observe; counting empty bins
/// overstates it (Grassberger 2003 §3.2). On a fully populated histogram the
/// two readings coincide, and Research 609's uniform fixture is that case.
#[inline]
pub fn miller_madow_entropy(histogram: &[u16]) -> f64 {
    let mut total: u64 = 0;
    let mut k_nonzero: u64 = 0;
    let mut sum_c_ln_c: f64 = 0.0;
    for &c in histogram {
        if c == 0 {
            continue;
        }
        let c = u64::from(c);
        total += c;
        k_nonzero += 1;
        let cf = c as f64;
        sum_c_ln_c += cf * cf.ln();
    }
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let h_emp = n.ln() - sum_c_ln_c / n;
    h_emp + (k_nonzero as f64 - 1.0) / (2.0 * n)
}

// ── Phase 2: the bottleneck classifier (T2.1–T2.3) ───────────────────────

/// The bottleneck class of one decision state (Plan 621 Phase 2).
///
/// FlyBy's split: *execution* bottlenecks are recoverable by more local
/// sampling/refinement (the ensemble's pass rate is measurably nonzero);
/// *knowledge* bottlenecks require external information — more local
/// continuations consolidate mass onto already-reachable solutions and
/// cannot help. The classifier reads ONLY the Wilson interval on the
/// measured pass rate, never a verbalized confidence (Research 609's
/// design-law table, row 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeClass {
    /// `wilson_hi < ε` — even the OPTIMISTIC bound on the pass rate is below
    /// the bar: the state is knowledge-like. Escalation is the only rescue.
    KnowledgeLike,
    /// The interval straddles ε — undetermined at this N. Adaptive stopping
    /// says: keep sampling (up to the caller's `n_max` cap).
    Undetermined,
    /// `wilson_lo ≥ ε` — decisively execution-reachable: the local path is
    /// predicted to succeed; escalation would be waste.
    Productive,
}

impl ProbeClass {
    /// `true` iff this is the knowledge-like class (the escalate signal).
    #[inline]
    #[must_use]
    pub const fn is_knowledge_like(self) -> bool {
        matches!(self, Self::KnowledgeLike)
    }
}

/// The per-N classification rule (T2.1): knowledge-like ⟺ `wilson_hi < ε`;
/// productive ⟺ `wilson_lo ≥ ε`; otherwise the interval straddles the bar.
/// NaN bounds compare false into [`ProbeClass::Undetermined`] — a corrupt
/// interval never produces a decisive class.
#[inline]
#[must_use]
pub fn classify(e: &ProbeEstimate, epsilon: f64) -> ProbeClass {
    if e.wilson_hi < epsilon {
        ProbeClass::KnowledgeLike
    } else if e.wilson_lo >= epsilon {
        ProbeClass::Productive
    } else {
        ProbeClass::Undetermined
    }
}

/// One disclosed classification flip (T2.1/T2.3): every flip carries the N
/// it happened at and the interval width that decided it — a flip without
/// its width is the uncalibrated-confidence smell the design law bans.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassFlip {
    /// The ensemble size at which the class flipped.
    pub at_n: u32,
    /// The class held before this observation.
    pub from: ProbeClass,
    /// The class held after it.
    pub to: ProbeClass,
    /// `wilson_hi − wilson_lo` at the flip — the width that decided it.
    pub interval_width: f64,
}

/// The adaptive-N stopping decision for one observation (T2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveDecision {
    /// A decisive class was reached — stop sampling.
    Settled(ProbeClass),
    /// Undetermined and `n < n_max` — the caller should keep sampling.
    Continue,
    /// Undetermined at `n ≥ n_max` — the cap. The caller falls back to the
    /// CONSERVATIVE non-knowledge-like reading (never escalates on a guess:
    /// an undetermined pass rate might still be productive).
    CappedUndetermined,
}

/// The adaptive classifier driver (T2.1): the caller re-probes at growing
/// N (it owns the ensemble), feeding each [`ProbeEstimate`] to
/// [`observe`](Self::observe); the tracker stops at the first decisive
/// class, records every flip WITH its interval width into the caller-owned
/// log slice (zero-alloc), and refuses to flip between decisive classes
/// (KnowledgeLike ↔ Productive without passing through Undetermined is
/// impossible by the rule's geometry — `wilson_hi < ε` and
/// `wilson_lo ≥ ε` are disjoint — and an interval that WIDENS back over ε
/// across N is treated as a new Undetermined, never a reverse flip).
#[derive(Debug)]
pub struct AdaptiveClassifier<'a> {
    epsilon: f64,
    n_max: u32,
    flips: &'a mut [ClassFlip],
    flips_len: usize,
    last: Option<ProbeClass>,
}

impl<'a> AdaptiveClassifier<'a> {
    /// `flips` is the caller-owned disclosure log (a full log drops further
    /// flips on the floor — the plan's disclosure duty is the caller's to
    /// size; 4 entries covers n_max/n_min doubling from 8 to 128).
    #[must_use]
    pub fn new(epsilon: f64, n_max: u32, flips: &'a mut [ClassFlip]) -> Self {
        Self {
            epsilon,
            n_max,
            flips,
            flips_len: 0,
            last: None,
        }
    }

    /// Flips recorded so far (in observation order).
    #[inline]
    #[must_use]
    pub fn flips(&self) -> &[ClassFlip] {
        &self.flips[..self.flips_len]
    }

    /// The last decisive class seen, if any.
    #[inline]
    #[must_use]
    pub const fn settled_class(&self) -> Option<ProbeClass> {
        self.last
    }

    /// Feed one observation at ensemble size `n`.
    #[must_use]
    pub fn observe(&mut self, e: &ProbeEstimate, n: u32) -> AdaptiveDecision {
        let class = classify(e, self.epsilon);
        if let Some(prev) = self.last
            && prev != class
        {
            // Disclosure duty: every flip carries its width — including
            // Undetermined→decisive (Undetermined IS the flip origin per
            // T2.3's monotone law). A full log drops the record
            // (caller-sized; documented in `new`).
            if self.flips_len < self.flips.len() {
                self.flips[self.flips_len] = ClassFlip {
                    at_n: n,
                    from: prev,
                    to: class,
                    interval_width: e.wilson_hi - e.wilson_lo,
                };
                self.flips_len += 1;
            }
        }
        self.last = Some(class);
        match class {
            ProbeClass::KnowledgeLike | ProbeClass::Productive => AdaptiveDecision::Settled(class),
            ProbeClass::Undetermined => {
                if n >= self.n_max {
                    AdaptiveDecision::CappedUndetermined
                } else {
                    AdaptiveDecision::Continue
                }
            }
        }
    }
}
