//! The gate kernel — `ClosedUnitCompactionGate<R>`.
//!
//! This is the heart of CUCG (Plan 320, Research 300). It composes a rubric,
//! a fire rule, a token-pct backstop, and an optional skip-if-reliable fuse
//! into a single `evaluate` entry point that produces a
//! [`CompactionDecision`] + [`CompactionAuditRecord`].
//!
//! # Decision order
//!
//! 1. **Backstop check** — if the prompt has exceeded the token-pct budget,
//!    return `Forced` immediately. The rubric is still evaluated so the audit
//!    records whether it *would have* fired, but the decision is forced.
//! 2. **Rubric evaluation** — run the rubric against the trajectory prefix.
//! 3. **Fire-rule check** — apply the Boolean fire rule to the verdict mask.
//! 4. **Skip-if-reliable** — if the fire rule says COMPRESS but the CLR
//!    reliability vote exceeds the suppression threshold, suppress to
//!    `Continue` (paper §4.1 skip-if-correct oracle, modelless).
//! 5. **Default** — `Continue`.
//!
//! # Zero-allocation contract
//!
//! `evaluate` performs no heap allocation for `N ≤ 8` (the bitmask width).
//! The audit record is a stack `#[repr(C)]` POD; the scratch is caller-owned
//! and reused. The only `Box` is the fire-rule tree, built once at
//! construction and never touched on the hot path beyond a tree walk.

use super::audit::{CompactionAuditRecord, DecisionKind, FireRuleEval, PredicateAudit};
use super::backstop::Backstop;
use super::decision::CompactionDecision;
use super::fire_rule::FireRule;
use super::rubric::{PredicateReason, PredicateResult, Rubric, RubricScratch, RubricVerdict};

/// Closed-Unit Compaction Gate — generic, rubric-gated, training-free context
/// compaction primitive.
///
/// Generic over the rubric `R` (which fixes the arity `N`) and the arity
/// `N` itself (const generic, `N ≤ 8`). The same gate hosts:
/// - trajectory compaction (paper's C1/C2/C3/N1 search rubric, `N = 4`),
/// - math-derivation compaction (paper's Q1/Q2/Q3 math rubric, `N = 3`),
/// - shard consolidation freeze (riir-neuron-db `can_freeze`, `N = 2`, G7).
///
/// # Construction
///
/// Use [`ClosedUnitCompactionGate::builder`] (or the explicit
/// [`ClosedUnitCompactionGate::new`]) to wire the rubric + fire rule +
/// backstop + optional skip-if-reliable fuse. The gate is then immutable;
/// all hot-path state lives in the caller-supplied [`RubricScratch`].
///
/// # Performance
///
/// Target (Plan 320 T6.1): `evaluate()` ≥ 50M decisions/sec for `N = 4`
/// (parity with Salience Tri-Gate's 120M/sec — CUCG has the same
/// two-sigmoid + fire-rule cost shape) and ≤ 50 ns latency.
pub struct ClosedUnitCompactionGate<R, const N: usize>
where
    R: Rubric<N>,
{
    rubric: R,
    fire_rule: FireRule,
    backstop: Backstop,
    /// Suppression fuse: if `Some(τ)`, suppress `Compress` when
    /// `clr_vote > τ`. Implements the paper's §4.1 skip-if-correct oracle,
    /// modellessly (CLR is the reliability signal; Plan 284).
    skip_if_reliable: Option<f32>,
    /// Probe interval in tokens (paper's `N`). Probes at
    /// `t ∈ {N, 2N, 3N, ...}`. The gate does not enforce this — the caller's
    /// probe loop calls `evaluate` at these intervals.
    probe_interval_tokens: usize,
}

impl<R, const N: usize> ClosedUnitCompactionGate<R, N>
where
    R: Rubric<N>,
{
    /// Construct a gate from its components.
    ///
    /// Prefer [`Self::builder`] for readability; this is the low-level
    /// constructor.
    ///
    /// # Panics
    ///
    /// Debug-asserts `N <= 8` (the bitmask width). A larger arity cannot be
    /// packed into the `u8` fire-rule mask.
    #[inline]
    #[allow(clippy::too_many_arguments)] // all args are the gate's knobs
    #[must_use]
    pub fn new(
        rubric: R,
        fire_rule: FireRule,
        backstop: Backstop,
        skip_if_reliable: Option<f32>,
        probe_interval_tokens: usize,
    ) -> Self {
        debug_assert!(
            N <= 8,
            "ClosedUnitCompactionGate: arity N must be <= 8 (fire-rule bitmask width), got {N}"
        );
        if let Some(threshold) = skip_if_reliable {
            debug_assert!(
                (0.0..=1.0).contains(&threshold),
                "skip_if_reliable threshold must be in [0,1], got {threshold}"
            );
        }
        debug_assert!(
            probe_interval_tokens > 0,
            "probe_interval_tokens must be > 0, got {probe_interval_tokens}"
        );
        Self {
            rubric,
            fire_rule,
            backstop,
            skip_if_reliable,
            probe_interval_tokens,
        }
    }

    /// Construct a builder for ergonomic configuration.
    #[must_use]
    pub fn builder(rubric: R) -> ClosedUnitCompactionGateBuilder<R, N> {
        ClosedUnitCompactionGateBuilder::new(rubric)
    }

    /// Returns the configured probe interval (tokens).
    #[inline]
    #[must_use]
    pub const fn probe_interval_tokens(&self) -> usize {
        self.probe_interval_tokens
    }

    /// Evaluate the gate at a probe point.
    ///
    /// # Arguments
    /// * `trajectory_prefix` — `y_{1:t}` (borrowed bytes; encoding is
    ///   rubric-defined).
    /// * `prompt_len` — current prompt length, for the backstop check.
    /// * `ctx_window` — context window size, for the backstop check.
    /// * `clr_vote` — optional CLR reliability vote on recent completions
    ///   (`None` disables skip-suppression for this call regardless of the
    ///   configured fuse).
    /// * `scratch` — caller-owned reusable scratch.
    ///
    /// # Returns
    ///
    /// A [`CompactionDecision`] carrying a full [`CompactionAuditRecord`].
    ///
    /// # Zero-allocation
    ///
    /// No heap allocation for `N ≤ 8`. The audit record is stack-allocated.
    #[inline]
    pub fn evaluate(
        &self,
        trajectory_prefix: &[u8],
        prompt_len: usize,
        ctx_window: usize,
        clr_vote: Option<f32>,
        scratch: &mut RubricScratch,
    ) -> CompactionDecision<N> {
        // 1. Backstop check — highest priority.
        let backstop_triggered = self.backstop.should_force(prompt_len, ctx_window);

        // 2. Rubric evaluation (always run — the audit records the verdict
        //    even when the backstop forces, so callers can detect "I was
        //    forced but the rubric said no").
        let verdict = self.rubric.evaluate(trajectory_prefix, scratch);

        // 3. Fire-rule check.
        let yes_mask = verdict.yes_mask();
        let fired = self.fire_rule.evaluate_mask(yes_mask);

        // 4. Skip-if-reliable suppression (only meaningful when the fire
        //    rule says COMPRESS and the backstop did NOT force — a forced
        //    decision cannot be suppressed).
        let skip_if_reliable_triggered = !backstop_triggered
            && fired
            && self
                .skip_if_reliable
                .zip(clr_vote)
                .is_some_and(|(threshold, vote)| vote > threshold);

        // 5. Build the audit record (the bridge: latent verdict → raw POD).
        let audit = self.build_audit(
            trajectory_prefix.len(),
            &verdict,
            yes_mask,
            fired,
            backstop_triggered,
            skip_if_reliable_triggered,
        );

        // 6. Decide.
        if backstop_triggered {
            let mut a = audit;
            a.decision = DecisionKind::Forced.to_byte();
            CompactionDecision::Forced { audit: a }
        } else if fired && !skip_if_reliable_triggered {
            let mut a = audit;
            a.decision = DecisionKind::Compress.to_byte();
            CompactionDecision::Compress { audit: a }
        } else {
            // Either the fire rule declined, or skip-suppression fired.
            // Both land on Continue.
            let mut a = audit;
            a.decision = DecisionKind::Continue.to_byte();
            CompactionDecision::Continue { audit: a }
        }
    }

    /// Bridge function: latent `RubricVerdict` → raw `CompactionAuditRecord`
    /// POD. Fixed-threshold projection (Yes/No) + trajectory-span recording.
    /// Zero-allocation.
    #[inline]
    fn build_audit(
        &self,
        trajectory_len: usize,
        verdict: &RubricVerdict<N>,
        yes_mask: u8,
        fired: bool,
        backstop_triggered: bool,
        skip_if_reliable_triggered: bool,
    ) -> CompactionAuditRecord<N> {
        // Project each PredicateResult → PredicateAudit POD.
        let predicates = {
            // Build a stack array by projecting element-by-element. The
            // array-init pattern uses `Default::default()` then per-slot
            // writes to avoid the const-generic array-init limitation.
            let mut arr: [PredicateAudit; N] = [PredicateAudit::default(); N];
            for (i, p) in verdict.predicates.iter().enumerate() {
                arr[i] = project_predicate(p);
            }
            arr
        };

        CompactionAuditRecord {
            trajectory_len: u32::try_from(trajectory_len).unwrap_or(u32::MAX),
            predicates,
            fire_rule_eval: FireRuleEval {
                yes_mask,
                fired: u8::from(fired),
                _pad: 0,
            },
            backstop_triggered: u8::from(backstop_triggered),
            skip_if_reliable_triggered: u8::from(skip_if_reliable_triggered),
            decision: DecisionKind::Continue.to_byte(), // finalized by caller
            _pad: 0,
        }
    }
}

/// Project a rich [`PredicateResult`] to a [`PredicateAudit`] POD. This is
/// the latent → raw bridge per AGENTS.md.
#[inline]
fn project_predicate(p: &PredicateResult) -> PredicateAudit {
    match p {
        PredicateResult::Yes {
            quote_start,
            quote_len,
        } => PredicateAudit::yes(*quote_start, *quote_len),
        PredicateResult::No { reason } => PredicateAudit::no(*reason),
    }
}

/// Builder for [`ClosedUnitCompactionGate`].
pub struct ClosedUnitCompactionGateBuilder<R, const N: usize>
where
    R: Rubric<N>,
{
    rubric: R,
    fire_rule: Option<FireRule>,
    backstop: Backstop,
    skip_if_reliable: Option<f32>,
    probe_interval_tokens: usize,
}

impl<R, const N: usize> ClosedUnitCompactionGateBuilder<R, N>
where
    R: Rubric<N>,
{
    /// Start a builder with the given rubric. Defaults: paper's search fire
    /// rule if `N == 4` (else a permissive `Or(0xFF)`), `Backstop::default()`
    /// (30%), no skip fuse, probe interval 1024 tokens.
    #[must_use]
    pub fn new(rubric: R) -> Self {
        let fire_rule = match N {
            4 => FireRule::search_rule_4(),
            3 => FireRule::math_rule_3(),
            2 => FireRule::shard_freeze_rule_2(),
            _ => FireRule::And(0xFF), // permissive default for unknown arity
        };
        Self {
            rubric,
            fire_rule: Some(fire_rule),
            backstop: Backstop::default(),
            skip_if_reliable: None,
            probe_interval_tokens: 1024,
        }
    }

    /// Set the fire rule.
    #[must_use]
    pub fn fire_rule(mut self, rule: FireRule) -> Self {
        self.fire_rule = Some(rule);
        self
    }

    /// Set the backstop.
    #[must_use]
    pub fn backstop(mut self, backstop: Backstop) -> Self {
        self.backstop = backstop;
        self
    }

    /// Set the skip-if-reliable suppression threshold.
    #[must_use]
    pub fn skip_if_reliable(mut self, threshold: f32) -> Self {
        self.skip_if_reliable = Some(threshold);
        self
    }

    /// Set the probe interval in tokens.
    #[must_use]
    pub fn probe_interval_tokens(mut self, tokens: usize) -> Self {
        self.probe_interval_tokens = tokens;
        self
    }

    /// Build the gate. Panics (debug) if the fire rule was not set.
    #[must_use]
    pub fn build(self) -> ClosedUnitCompactionGate<R, N> {
        ClosedUnitCompactionGate::new(
            self.rubric,
            self.fire_rule.expect("fire_rule must be set"),
            self.backstop,
            self.skip_if_reliable,
            self.probe_interval_tokens,
        )
    }
}

/// Re-export the reason constant for callers that want the paper's "still
/// novel" No-reason without importing the full enum.
impl<R, const N: usize> ClosedUnitCompactionGate<R, N>
where
    R: Rubric<N>,
{
    /// Convenience: the `PredicateReason::StillNovel` discriminant, for
    /// rubric implementations that want to mirror the paper's N1 wording.
    pub const REASON_STILL_NOVEL: PredicateReason = PredicateReason::StillNovel;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::rubric::{PredicateReason, PredicateResult, RubricVerdict};

    /// Test rubric: arity 4. Predicates fire iff the trajectory contains
    /// `b'C'` (C1), `b'R'` (C2 rank), `b'P'` (C3 progress); N1 (stuck) fires
    /// iff trajectory contains `b'S'` (stuck marker).
    struct SearchRubric;
    impl Rubric<4> for SearchRubric {
        fn evaluate(&self, traj: &[u8], _scratch: &mut RubricScratch) -> RubricVerdict<4> {
            let mk = |marker: u8, no_reason: PredicateReason| -> PredicateResult {
                if let Some(i) = traj.iter().position(|&b| b == marker) {
                    PredicateResult::Yes {
                        quote_start: i as u32,
                        quote_len: 1,
                    }
                } else {
                    PredicateResult::No { reason: no_reason }
                }
            };
            RubricVerdict::new([
                mk(b'C', PredicateReason::NotClosedUnit),
                mk(b'R', PredicateReason::TooHighRank),
                mk(b'P', PredicateReason::NoProgress),
                mk(b'S', PredicateReason::StillNovel),
            ])
        }
    }

    fn gate() -> ClosedUnitCompactionGate<SearchRubric, 4> {
        ClosedUnitCompactionGate::builder(SearchRubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None) // disable backstop for rubric-focused tests
            .build()
    }

    #[test]
    fn search_rule_fires_on_complete_safe_trajectory() {
        let g = gate();
        let mut s = RubricScratch::new();
        // C, R, P present (C1,C2,C3 Yes); S absent (N1 No → ¬N1 holds).
        let d = g.evaluate(b"step: C R P done", 0, 10_000, None, &mut s);
        assert!(d.is_compress(), "C1∧C2∧C3∧¬N1 must Compress");
        assert!(d.audit().fire_rule_eval.fired == 1);
        assert_eq!(d.audit().fire_rule_eval.yes_mask, 0b0111); // C1,C2,C3
        assert!(!d.audit().is_forced());
    }

    #[test]
    fn search_rule_does_not_fire_when_stuck() {
        let g = gate();
        let mut s = RubricScratch::new();
        // S present → N1 Yes → ¬N1 fails.
        let d = g.evaluate(b"C R P S", 0, 10_000, None, &mut s);
        assert!(d.is_continue(), "N1 Yes blocks Compress");
        assert_eq!(d.audit().fire_rule_eval.yes_mask, 0b1111); // all four incl N1
        assert_eq!(d.audit().fire_rule_eval.fired, 0); // rule declined
    }

    #[test]
    fn search_rule_does_not_fire_when_missing_progress() {
        let g = gate();
        let mut s = RubricScratch::new();
        // P absent → C3 No.
        let d = g.evaluate(b"C R no_progress", 0, 10_000, None, &mut s);
        assert!(d.is_continue());
        assert_eq!(d.audit().fire_rule_eval.yes_mask, 0b0011); // C1,C2 only
    }

    #[test]
    fn backstop_forces_even_when_rubric_declines() {
        let g = ClosedUnitCompactionGate::builder(SearchRubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::token_pct(0.30))
            .build();
        let mut s = RubricScratch::new();
        // Empty trajectory → rubric all-No → would Continue. But prompt_len
        // (500) >= 30% of ctx_window (1000) → backstop forces.
        let d = g.evaluate(b"", 500, 1000, None, &mut s);
        assert!(d.is_forced(), "backstop must override rubric decline");
        assert_eq!(d.audit().backstop_triggered, 1);
        // The audit still records the rubric's verdict.
        assert_eq!(d.audit().fire_rule_eval.fired, 0); // rubric declined
    }

    #[test]
    fn skip_if_reliable_suppresses_compress() {
        let g = ClosedUnitCompactionGate::builder(SearchRubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .skip_if_reliable(0.8) // suppress if CLR vote > 0.8
            .build();
        let mut s = RubricScratch::new();

        // Rubric would fire (C,R,P, no S), but CLR vote 0.9 > 0.8 → suppress.
        let d = g.evaluate(b"C R P", 0, 10_000, Some(0.9), &mut s);
        assert!(d.is_continue(), "skip-if-reliable suppresses Compress");
        assert_eq!(d.audit().fire_rule_eval.fired, 1); // rubric DID fire
        assert_eq!(d.audit().skip_if_reliable_triggered, 1);

        // CLR vote 0.7 ≤ 0.8 → no suppression → Compress proceeds.
        let d2 = g.evaluate(b"C R P", 0, 10_000, Some(0.7), &mut s);
        assert!(d2.is_compress());
        assert_eq!(d2.audit().skip_if_reliable_triggered, 0);
    }

    #[test]
    fn skip_if_reliable_ignored_when_clr_vote_none() {
        let g = ClosedUnitCompactionGate::builder(SearchRubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::None)
            .skip_if_reliable(0.5)
            .build();
        let mut s = RubricScratch::new();
        // clr_vote = None → no suppression even though fuse is configured.
        let d = g.evaluate(b"C R P", 0, 10_000, None, &mut s);
        assert!(d.is_compress());
        assert_eq!(d.audit().skip_if_reliable_triggered, 0);
    }

    #[test]
    fn backstop_cannot_be_suppressed_by_skip_if_reliable() {
        // A forced decision (backstop) takes precedence over skip-suppression.
        let g = ClosedUnitCompactionGate::builder(SearchRubric)
            .fire_rule(FireRule::search_rule_4())
            .backstop(Backstop::token_pct(0.10))
            .skip_if_reliable(0.1) // very aggressive suppression
            .build();
        let mut s = RubricScratch::new();
        // prompt_len 500 >= 10% of 1000 → backstop forces. CLR vote 0.9 > 0.1
        // would suppress a Compress, but the decision is Forced — skip is
        // ignored.
        let d = g.evaluate(b"C R P", 500, 1000, Some(0.9), &mut s);
        assert!(d.is_forced());
        assert_eq!(d.audit().backstop_triggered, 1);
        assert_eq!(
            d.audit().skip_if_reliable_triggered,
            0,
            "skip-suppression does not apply to Forced"
        );
    }

    #[test]
    fn math_rule_q1_or_q2_q3() {
        struct MathRubric;
        impl Rubric<3> for MathRubric {
            fn evaluate(&self, traj: &[u8], _s: &mut RubricScratch) -> RubricVerdict<3> {
                let mk = |b: u8| -> PredicateResult {
                    if traj.contains(&(b)) {
                        PredicateResult::Yes {
                            quote_start: 0,
                            quote_len: 1,
                        }
                    } else {
                        PredicateResult::No {
                            reason: PredicateReason::Unset,
                        }
                    }
                };
                RubricVerdict::new([mk(b'1'), mk(b'2'), mk(b'3')])
            }
        }
        let g = ClosedUnitCompactionGate::builder(MathRubric).build();
        let mut s = RubricScratch::new();

        // Q1 only → fires.
        assert!(g.evaluate(b"1", 0, 10_000, None, &mut s).is_compress());
        // Q2 ∧ Q3 → fires.
        assert!(g.evaluate(b"23", 0, 10_000, None, &mut s).is_compress());
        // Q2 only → no fire.
        assert!(g.evaluate(b"2", 0, 10_000, None, &mut s).is_continue());
    }

    #[test]
    fn probe_interval_tokens_round_trips() {
        let g = ClosedUnitCompactionGate::builder(SearchRubric)
            .probe_interval_tokens(2048)
            .build();
        assert_eq!(g.probe_interval_tokens(), 2048);
    }

    #[test]
    fn audit_record_is_deterministic_same_input_same_output() {
        let g = gate();
        let mut s = RubricScratch::new();
        let d1 = g.evaluate(b"C R P done", 100, 10_000, None, &mut s);
        let d2 = g.evaluate(b"C R P done", 100, 10_000, None, &mut s);
        // Two calls on identical input MUST produce bit-identical audit.
        assert_eq!(d1.audit(), d2.audit());
    }
}
