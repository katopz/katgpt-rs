//! escalation_probe_gate — the cost-aware escalation selection utility
//! (Plan 621 Phase 3; Research 609 §2.4, FlyBy arXiv:2609.34327).
//!
//! [`gate`] is `gain_cost_halt`'s scissors law generalized to escalation
//! decisions, with FlyBy's reward shape preserved exactly:
//!
//! ```text
//! U(action) = I[predicted success(action)] · (1 − λ·Ĉ(action))
//! ```
//!
//! The cost multiplier applies ONLY on predicted success (FlyBy: "cost
//! penalized only on success, so failing cheaply is never rewarded") — the
//! subtractive form is the recorded λ=0.2 failure mode (cost-centering
//! assigns negative advantage to correct-but-expensive trajectories,
//! Research 609 §1). There is NO reward scale and NO cost subtraction: the
//! utility lives in [0, 1] and the selection is argmax with stay winning
//! ties.
//!
//! The two arms:
//!
//! * **stay-local** — predicted success iff the state is NOT knowledge-like
//!   (the Phase-2 rule: `wilson_hi ≥ ε`); Ĉ = 0 (the local path has no
//!   corpus dependence). `U_stay = I[¬KL]`.
//! * **escalate(d)** — predicted success iff the depth's success prior
//!   clears `r_min`; Ĉ_d = the consumer-supplied corpus distance
//!   (far corpus → less rescue value). `U_esc(d) = I[r_d ≥ r_min]·(1 − λ·Ĉ_d)`.
//!
//! FlyBy's asymmetry falls out STRUCTURALLY, not as a special case: a
//! failing escalation (`r_d < r_min`) scores 0 and can never out-score a
//! cheaper success (a productive state scores `U_stay = 1`); a productive
//! state never escalates (`U_stay = 1 ≥ U_esc ≤ 1`); and λ is monotone
//! (higher λ only shrinks escalation utilities ⇒ weakly fewer escalations).
//! At `λ = 0` the gate reduces to the pred-success-only incumbent rule —
//! the G3 byte-identity anchor (T3.3; the consumer-side seam adapters that
//! make it byte-identical to `modelless_cap`/`EscalateSpec` are
//! composition, T3.2, and live consumer-side).
//!
//! Reflex-free: the kernel takes plain floats (probe estimate + per-depth
//! priors + distances). The corpus-distance signal is CONSUMER-SUPPLIED
//! (`CorpusDistanceGate` on the reflex side; rescue-mining on the refine
//! side) — this module never computes one.

use crate::state_probe::{ProbeEstimate, classify};

/// One escalation depth option (T3.1): `{stay, escalate(d)}`'s right side.
#[derive(Debug, Clone, Copy)]
pub struct EscalationOption {
    /// The depth `d` (opaque to the kernel — the consumer's cost-tier id).
    pub depth: u32,
    /// `r_d` — the depth's measured/declared success prior ∈ [0, 1]
    /// (e.g. a tool/teacher's offline-measured solve rate at this budget).
    pub success_prior: f64,
    /// `Ĉ_d` — the corpus distance ∈ [0, 1] (1 = far/novel: the escalation
    /// corpus covers this state poorly, so its rescue value is discounted).
    pub corpus_distance: f64,
}

/// The gate's input: the probe read + the escalation ladder + λ.
#[derive(Debug, Clone, Copy)]
pub struct GateInput<'a> {
    /// The Phase-1 probe read over the state.
    pub probe: ProbeEstimate,
    /// The escalation ladder (the consumer's cost tiers, ascending or not —
    /// selection is by utility, ties by lowest index).
    pub options: &'a [EscalationOption],
    /// λ ≥ 0 — the corpus-distance penalty strength (FlyBy used 0.1; their
    /// recorded λ=0.2 failure is the argument for staying at or below it).
    /// NaN λ fails closed: every `1 − NaN·Ĉ` is NaN, NaN comparisons are
    /// false, and the gate stays local.
    pub lambda: f64,
}

/// The gate's selection (T3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Answer locally with the modelless path — the incumbent default, and
    /// the fallback whenever the probe is off (the module compiles away) or
    /// the utility comparison does not favor escalation.
    StayLocal,
    /// Escalate to depth `d` (the option's index is carried so the consumer
    /// can reach its cost/tier metadata without re-searching).
    Escalate { index: usize, depth: u32 },
}

/// The stay arm's utility: `I[¬KL] · (1 − λ·0) = I[¬KL]`.
#[inline]
#[must_use]
pub fn stay_utility(probe: &ProbeEstimate, epsilon: f64) -> f64 {
    let class = classify(probe, epsilon);
    if class.is_knowledge_like() {
        0.0
    } else {
        1.0
    }
}

/// The escalate arm's utility for one option:
/// `I[r_d ≥ r_min] · (1 − λ·Ĉ_d)`, floored at 0 (`λ·Ĉ > 1` must never read
/// as a NEGATIVE utility — the multiplier is a discount, not a penalty;
/// NaN anywhere fails closed to 0).
#[inline]
#[must_use]
pub fn escalate_utility(option: &EscalationOption, lambda: f64, r_min: f64) -> f64 {
    if option.success_prior >= r_min {
        let u = 1.0 - lambda * option.corpus_distance;
        if u.is_finite() {
            u.clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else {
        // Below the success bar (or NaN — the comparison fails): a failing
        // escalation scores 0, the asymmetry the truth table pins.
        0.0
    }
}

/// The selection gate (T3.1): escalate only when the state is
/// knowledge-like AND some depth's utility strictly beats the stay arm;
/// ties go to stay (the conservative default, and the demote-the-loser
/// fallback's shape). NaN λ / NaN priors / empty ladders all stay local —
/// the gate is fail-closed in every corrupt direction.
#[must_use]
pub fn gate(input: &GateInput<'_>, epsilon: f64, r_min: f64) -> GateDecision {
    if stay_utility(&input.probe, epsilon) > 0.0 {
        // A productive (or capped-undetermined) state: the local path is
        // predicted to succeed at utility 1 ≥ any U_esc ≤ 1 — FlyBy's
        // asymmetry, structurally.
        return GateDecision::StayLocal;
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, opt) in input.options.iter().enumerate() {
        let u = escalate_utility(opt, input.lambda, r_min);
        if u <= 0.0 {
            continue;
        }
        // Strict `>` keeps the LOWEST index on ties (deterministic).
        if best.is_none_or(|(_, bu)| u > bu) {
            best = Some((i, u));
        }
    }
    match best {
        Some((i, _)) => GateDecision::Escalate {
            index: i,
            depth: input.options[i].depth,
        },
        None => GateDecision::StayLocal,
    }
}

#[cfg(test)]
mod tests {
    use super::{EscalationOption, GateDecision, GateInput, escalate_utility, gate, stay_utility};
    use crate::state_probe::{ProbeEstimate, classify};

    /// wilson interval helper: the tests write (lo, hi) directly — the
    /// interval's provenance is Phase 1's fixture suite; these tests pin
    /// the SELECTION geometry, not the CI math.
    const fn est(lo: f64, hi: f64) -> ProbeEstimate {
        ProbeEstimate {
            v_hat: (lo + hi) / 2.0,
            h_mm: 0.0,
            wilson_lo: lo,
            wilson_hi: hi,
        }
    }

    const EPS: f64 = 0.1;
    const R_MIN: f64 = 0.5;

    fn input<'a>(probe: ProbeEstimate, opts: &'a [EscalationOption], lambda: f64) -> GateInput<'a> {
        GateInput {
            probe,
            options: opts,
            lambda,
        }
    }

    // ── T3.4: the truth table ───────────────────────────────────────────

    #[test]
    fn failing_escalation_never_outscores_a_cheaper_success() {
        // r_d < r_min ⇒ U_esc = 0: a failing escalation stays at 0 forever.
        let ladder = [
            EscalationOption { depth: 1, success_prior: 0.49, corpus_distance: 0.0 },
            EscalationOption { depth: 2, success_prior: 0.9, corpus_distance: 0.0 },
        ];
        // Knowledge-like state: escalate to the PASSING depth (index 1).
        let g = gate(&input(est(0.0, 0.05), &ladder, 0.1), EPS, R_MIN);
        assert_eq!(g, GateDecision::Escalate { index: 1, depth: 2 });
        // The failing depth alone: stay (never escalate on a failing arm).
        let only_failing = [EscalationOption { depth: 1, success_prior: 0.2, corpus_distance: 0.0 }];
        let g = gate(&input(est(0.0, 0.05), &only_failing, 0.1), EPS, R_MIN);
        assert_eq!(g, GateDecision::StayLocal);
        // A PRODUCTIVE state with a passing escalation: stay wins — the
        // cheaper success (utility 1) out-scores the escalation (≤ 1).
        let productive = [EscalationOption { depth: 1, success_prior: 0.99, corpus_distance: 0.0 }];
        let g = gate(&input(est(0.3, 0.6), &productive, 0.0), EPS, R_MIN);
        assert_eq!(g, GateDecision::StayLocal);
        assert_eq!(stay_utility(&est(0.3, 0.6), EPS), 1.0);
        assert_eq!(stay_utility(&est(0.0, 0.05), EPS), 0.0);
    }

    #[test]
    fn utility_monotone_in_lambda_weakly_fewer_escalations() {
        // A knowledge-like state over a ladder whose rescue value decays
        // with corpus distance: as λ rises, the far-depth utility crosses
        // below the near-depth and eventually to 0 — the escalation choice
        // moves DOWN the ladder and then to stay, never the reverse.
        let ladder = [
            EscalationOption { depth: 3, success_prior: 0.9, corpus_distance: 0.9 },
            EscalationOption { depth: 2, success_prior: 0.8, corpus_distance: 0.4 },
            EscalationOption { depth: 1, success_prior: 0.7, corpus_distance: 0.1 },
        ];
        let kl = est(0.0, 0.02);
        let lambdas = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];
        let mut prev_depth = u32::MAX;
        let mut seq: Vec<u32> = Vec::new();
        for &lam in &lambdas {
            let g = gate(&input(kl, &ladder, lam), EPS, R_MIN);
            match g {
                GateDecision::Escalate { depth, .. } => {
                    seq.push(depth);
                    assert!(
                        depth <= prev_depth,
                        "λ={lam}: depth {depth} after {prev_depth} — λ must move DOWN the ladder, never up"
                    );
                    prev_depth = depth;
                }
                GateDecision::StayLocal => {
                    seq.push(0);
                    prev_depth = 0;
                }
            }
        }
        // The pinned shape: λ=0's all-1.0 tie takes the LOWEST INDEX (depth
        // 3); every λ>0 re-ranks to the near corpus (depth 1); depth 3's
        // Ĉ=0.9 leaves the running at λ=2 (1−1.8 → 0) but depth 1's
        // Ĉ=0.1 still clears it — STAY would need the whole ladder
        // discounted out. Monotone: [3, then 1 forever].
        assert_eq!(seq, vec![3, 1, 1, 1, 1, 1, 1], "utility-λ geometry moved");
    }

    // ── T3.3: λ=0 byte-identity with the pred-success-only rule ─────────

    #[test]
    fn lambda_zero_is_the_incumbent_pred_success_rule() {
        // λ=0 strips the cost multipliers: escalate iff knowledge-like AND
        // some r_d ≥ r_min — the incumbent rule the consumer cap-gate falls
        // back to (the byte-identity anchor; consumer-side seam adapters
        // are T3.2 composition).
        let ladder = [
            EscalationOption { depth: 1, success_prior: 0.6, corpus_distance: 0.9 },
            EscalationOption { depth: 2, success_prior: 0.55, corpus_distance: 0.99 },
        ];
        let cases = [
            (est(0.0, 0.05), GateDecision::Escalate { index: 0, depth: 1 }),
            (est(0.08, 0.4), GateDecision::StayLocal), // undetermined: conservative stay
            (est(0.2, 0.6), GateDecision::StayLocal),  // productive
        ];
        for (probe, want) in cases {
            let g = gate(&input(probe, &ladder, 0.0), EPS, R_MIN);
            assert_eq!(g, want, "λ=0 must equal the pred-success rule at {probe:?}");
        }
        // And the utility itself at λ=0 is the bare indicator.
        assert_eq!(escalate_utility(&ladder[0], 0.0, R_MIN), 1.0);
        assert_eq!(escalate_utility(&ladder[1], 0.0, R_MIN), 1.0);
    }

    // ── fail-closed directions ──────────────────────────────────────────

    #[test]
    fn corrupt_inputs_stay_local_never_panic() {
        let ladder = [EscalationOption { depth: 1, success_prior: 0.9, corpus_distance: 0.5 }];
        // NaN λ → every 1−λ·Ĉ is NaN → 0 → stay.
        assert_eq!(
            gate(&input(est(0.0, 0.05), &ladder, f64::NAN), EPS, R_MIN),
            GateDecision::StayLocal
        );
        // NaN prior → the r_min comparison fails → U_esc 0 → stay.
        let nan_prior = [EscalationOption { depth: 1, success_prior: f64::NAN, corpus_distance: 0.0 }];
        assert_eq!(
            gate(&input(est(0.0, 0.05), &nan_prior, 0.1), EPS, R_MIN),
            GateDecision::StayLocal
        );
        // Empty ladder → stay.
        assert_eq!(gate(&input(est(0.0, 0.05), &[], 0.1), EPS, R_MIN), GateDecision::StayLocal);
        // NaN probe bounds → undetermined class → stay (never a decisive KL).
        assert_eq!(
            classify(&est(f64::NAN, f64::NAN), EPS),
            crate::state_probe::ProbeClass::Undetermined
        );
        // λ·Ĉ > 1 floors at 0 (a discount, never a negative utility).
        assert_eq!(
            escalate_utility(&EscalationOption { depth: 1, success_prior: 0.9, corpus_distance: 1.0 }, 2.0, R_MIN),
            0.0
        );
    }
}
