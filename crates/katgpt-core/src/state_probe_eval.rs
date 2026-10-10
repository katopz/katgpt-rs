//! Plan 621 Phase 4 — the healer-lane escalation evaluation instrument
//! (T4.1–T4.3): a planted two-class corpus of fix-miss states, the three
//! comparison arms (never-escalate floor / incumbent cap-only / probe-gated),
//! and the G2 probe-cost budget guard.
//!
//! # What this module is (and is not)
//!
//! This is the EVAL instrument behind `.benchmarks/621_escalation_eval_goat.md`
//! — a deterministic, seed-pinned simulator over a PLANTED corpus (T4.1's
//! design: fix-miss states with both classes present, so a gate that never
//! fires FAILS G1). It is NOT production telemetry: the rescue and cost models
//! below are DECLARED (documented constants), the corpus is synthetic by
//! design, and the Super-GOAT label this instrument decides is a claim about
//! the decision layer's sample-efficiency ON THIS MODEL — the production
//! proof is the consumer wiring (Plan 621 T5.2, riir-refine Issue 156), which
//! re-gates on real misses.
//!
//! # The declared world model
//!
//! Every corpus state is a fix-miss: the modelless healer ran its local path
//! and failed. Each state carries a planted TRUE local pass rate `p` — the
//! probability that one additional local attempt (ensemble member) lands the
//! fix:
//!
//! | planted class | true `p` | what rescues it |
//! |---|---|---|
//! | `KnowledgeLike` | 0.0 | escalation ONLY (the fix needs external knowledge; more local sampling consolidates mass onto already-failing answers) |
//! | `ExecutionLike` | 0.20 | local retries (`1−(1−p)^k` over `k` attempts) — and NOT escalation: the expert has no local context, so on execution-bottleneck states it does no better than the exhausted local path (FlyBy's scissors law; their 28/76 rescue mass sits on the knowledge class) |
//! | `Productive` | 0.60 | local retries, almost surely; a miss was a draw artifact — escalation is pure waste |
//!
//! Rescue accounting per state: `rescued = local_rescue ∨ (escalated ∧
//! state is knowledge-like ∧ escalation_rescue)` — one rescue per state
//! even if both paths fire. The escalation solve rate [`R_ESC`] applies to
//! the knowledge class only (the declared, FlyBy-faithful scissors law:
//! a class-agnostic expert rate would make triage worthless by
//! construction, and the measured 28/76 mass contradicts it).
//!
//! Spend: one escalation costs [`C_ESC`] units. All arms share the corpus's
//! per-state draws (common random numbers — an arm never gets a luckier
//! stream) and the same cap [`cap_units`] (the incumbent's `modelless_cap`
//! is exactly this: a total-spend ceiling and nothing else — a miss state is
//! escalated iff budget remains, in corpus order).
//!
//! The probe arm classifies each state adaptively (the Phase-2 N ladder) and
//! escalates only classified-`KnowledgeLike` states ([`CappedUndetermined`]
//! falls back to the conservative non-knowledge-like reading — never
//! escalates on a guess). Its probe evaluations are counted; [`G2`] holds the
//! ≤ 10 % budget line.
//!
//! # Determinism
//!
//! Seed-pinned splitmix64 (the T1.2 fixture convention). Same seed + same
//! corpus shape ⇒ bit-identical reports; the benchmark record freezes one
//! read and cites the seed.

use crate::escalation_probe_gate::{EscalationOption, GateDecision, GateInput, gate};
use crate::state_probe::{
    AdaptiveClassifier, AdaptiveDecision, ClassFlip, ProbeClass, ProbeInput, probe,
};

/// The escalation solve rate on the knowledge class — the hosted expert's
/// declared offline-measured solve probability where escalation is the only
/// channel (the scissors law: on execution-like states the expert adds
/// nothing, so its rate there is 0 by declaration).
pub const R_ESC: f64 = 0.6;

/// Spend units per escalation (the unit of account; the cap is expressed in
/// the same units).
pub const C_ESC: f64 = 1.0;

/// Local retry attempts per miss state (the healer's per-state attempt
/// budget before the state is written off).
pub const LOCAL_RETRIES: u32 = 4;

/// The G2 budget line (plan T4.3): the probe's total evaluation spend must
/// stay ≤ 10 % of the gated decisions' escalation spend; on breach the arm
/// FALLS BACK to the incumbent decision for the remaining states (never
/// silently eats the budget).
pub const PROBE_BUDGET_FRACTION: f64 = 0.1;

/// The probe's declared per-evaluation cost, in the same units as `C_ESC`.
/// The bench measures the real ns/call and re-derives this ratio; the unit
/// here prices one evaluation at a hosted-call's 1/10_000 (an escalation is
/// a network round trip; a probe is a ns-class scan — 4+ orders apart, so
/// the 10 % line is about DISCIPLINE, not a live risk; the test pins the
/// breach→fallback path with a deliberately absurd per-eval cost).
pub const PROBE_EVAL_COST_UNITS: f64 = 0.000_1;

/// The planted ground-truth class of a corpus state (distinct vocabulary
/// from the probe's [`ProbeClass`] — the planted truth is never observed
/// directly by the arms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlantedClass {
    /// True rate 0.0 — escalation is the only rescue.
    KnowledgeLike,
    /// True rate in the execution band — local retries rescue.
    ExecutionLike,
    /// True rate high — a miss was a draw artifact; escalation is waste.
    Productive,
}

impl PlantedClass {
    /// The planted true per-attempt pass rate.
    #[must_use]
    pub const fn true_rate(self) -> f64 {
        match self {
            Self::KnowledgeLike => 0.0,
            Self::ExecutionLike => 0.2,
            Self::Productive => 0.6,
        }
    }
}

/// One planted fix-miss state: the ground truth + the common-random-number
/// draw streams every arm shares.
#[derive(Debug, Clone)]
pub struct EvalState {
    /// The planted truth.
    pub planted: PlantedClass,
    /// The state id (corpus order).
    pub id: u32,
    /// Realized local-attempt outcomes (1 = that retry landed the fix), one
    /// per [`LOCAL_RETRIES`] attempt. Shared by every arm.
    pub local_draws: [bool; LOCAL_RETRIES as usize],
    /// Realized escalation outcome (would the hosted expert solve it).
    /// Shared by every arm.
    pub escalation_draw: bool,
    /// The realized ensemble the probe reads: pass counts at the N ladder's
    /// rungs (the probe never sees `true_rate` — only these draws).
    pub ensemble: EnsembleDraws,
}

/// The probe's realized ensemble: pass counts at fixed rungs. The adaptive
/// ladder walks these in order (8 → 16 → 32 → 64 → 128 members), and the
/// per-rung answer histogram is realized too (one dominant bin + the pass
/// spread — the entropy term's food).
#[derive(Debug, Clone, Copy)]
pub struct EnsembleDraws {
    /// Cumulative passes at each rung (`rungs[i]` members realized).
    pub passes_at_rung: [u32; N_RUNGS],
    /// The rung sizes (the ladder).
    pub rungs: [u32; N_RUNGS],
    /// The final rung's answer histogram (u16 counts; sums to the rung).
    pub histogram: [u16; HIST_BINS],
}

/// Ladder rung count (8/16/32/64/128).
pub const N_RUNGS: usize = 5;

/// Answer-histogram bins (the answer alphabet the corpus realizes into).
pub const HIST_BINS: usize = 16;

/// Seed-pinned splitmix64 — deterministic draws, no dep, no global RNG
/// (the T1.2 fixture convention).
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A generator pinned at `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    /// The next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// The next `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A Bernoulli(`p`) draw.
    pub fn next_bool(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

/// The planted corpus, CLASS-INTERLEAVED (round-robin across the three
/// blocks — a real miss stream is not class-aligned, and a knowledge-first
/// layout would hand the cap-only incumbent its best case by construction).
#[must_use]
pub fn planted_corpus(n_knowledge: usize, n_execution: usize, n_productive: usize, seed: u64) -> Vec<EvalState> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(n_knowledge + n_execution + n_productive);
    let mut id: u32 = 0;
    let push_block = |rng: &mut SplitMix64, out: &mut Vec<EvalState>, class: PlantedClass, count: usize, id: &mut u32| {
        for _ in 0..count {
            let p = class.true_rate();
            let mut local_draws = [false; LOCAL_RETRIES as usize];
            for slot in &mut local_draws {
                *slot = rng.next_bool(p);
            }
            let escalation_draw = rng.next_bool(R_ESC);
            let ensemble = realize_ensemble(rng, p);
            out.push(EvalState { planted: class, id: *id, local_draws, escalation_draw, ensemble });
            *id += 1;
        }
    };
    let mut knowledge = Vec::with_capacity(n_knowledge);
    let mut execution = Vec::with_capacity(n_execution);
    let mut productive = Vec::with_capacity(n_productive);
    push_block(&mut rng, &mut knowledge, PlantedClass::KnowledgeLike, n_knowledge, &mut id);
    push_block(&mut rng, &mut execution, PlantedClass::ExecutionLike, n_execution, &mut id);
    push_block(&mut rng, &mut productive, PlantedClass::Productive, n_productive, &mut id);
    // Deterministic round-robin interleave (no RNG — layout is contract).
    let mut ki = 0;
    let mut ei = 0;
    let mut pi = 0;
    while ki < knowledge.len() || ei < execution.len() || pi < productive.len() {
        if ki < knowledge.len() {
            out.push(knowledge[ki].clone());
            ki += 1;
        }
        if ei < execution.len() {
            out.push(execution[ei].clone());
            ei += 1;
        }
        if pi < productive.len() {
            out.push(productive[pi].clone());
            pi += 1;
        }
    }
    // The ids ride the INTERLEAVED order (the cap spends in corpus order).
    for (i, state) in out.iter_mut().enumerate() {
        state.id = u32::try_from(i).unwrap_or(u32::MAX);
    }
    out
}

/// Realize the probe-readable ensemble at the ladder rungs under true rate
/// `p`, plus the final rung's answer histogram (dominant bin + spread —
/// under `p = 0` the histogram collapses onto one failing answer class,
/// which is exactly the low-entropy shape the entropy term rewards).
fn realize_ensemble(rng: &mut SplitMix64, p: f64) -> EnsembleDraws {
    let rungs = [8u32, 16, 32, 64, 128];
    let mut passes_at_rung = [0u32; N_RUNGS];
    let mut cumulative = 0u32;
    let mut histogram = [0u16; HIST_BINS];
    for (i, &rung) in rungs.iter().enumerate() {
        while cumulative < rung {
            cumulative += 1;
            if rng.next_bool(p) {
                passes_at_rung[i] += 1;
            }
        }
    }
    // The histogram summarizes the FINAL rung's answers: passing members
    // land in the correct-answer bin 0; failing members spread over the
    // wrong-answer bins (a memoryless draw onto bins 1..). Under p = 0 the
    // mass is all-wrong and spread — the corpus does not hand the entropy
    // term a free low-entropy signal; the pass rate carries the class.
    let final_passes = passes_at_rung[N_RUNGS - 1];
    histogram[0] = u16::try_from(final_passes).unwrap_or(u16::MAX);
    let failures = rungs[N_RUNGS - 1] - final_passes;
    for _ in 0..failures {
        let bin = 1 + (rng.next_u64() as usize) % (HIST_BINS - 1);
        histogram[bin] = histogram[bin].saturating_add(1);
    }
    EnsembleDraws { passes_at_rung, rungs, histogram }
}

/// One arm's per-corpus outcome.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArmReport {
    /// States the arm rescued at all (local ∨ escalated).
    pub rescues: u32,
    /// Rescues attributable to escalation alone (the state's local draws all
    /// failed and the escalation saved it — under the scissors law these are
    /// exactly the knowledge-class escalations that landed).
    pub escalation_net_rescues: u32,
    /// Escalations performed.
    pub escalations: u32,
    /// Escalations that landed on a planted-knowledge-like state (the
    /// truth-aware precision numerator — the arms know the corpus truth;
    /// the probe's vote is the classification, this is its grade).
    pub knowledge_escalations: u32,
    /// Total escalation spend (units) — `escalations × C_ESC`, capped.
    pub spend_units: f64,
    /// Probe evaluations spent (probe arm only; 0 elsewhere).
    pub probe_evals: u32,
    /// Whether the G2 budget guard forced the incumbent fallback.
    pub budget_fallback: bool,
}

impl ArmReport {
    /// Share of escalations that landed on states only escalation can
    /// rescue — the triage quality metric (NOT the solve-rate draw: the
    /// net-rescue count folds `R_ESC` in, this does not).
    #[must_use]
    pub fn escalation_precision(&self) -> f64 {
        if self.escalations == 0 {
            0.0
        } else {
            f64::from(self.knowledge_escalations) / f64::from(self.escalations)
        }
    }
}

/// Arm A — the never-escalate floor (G1's "a gate that never fires" control:
/// its rescue count is the corpus's local-only rescue mass).
#[must_use]
pub fn arm_never_escalate(corpus: &[EvalState]) -> ArmReport {
    let mut r = ArmReport::default();
    for state in corpus {
        if state.local_draws.iter().any(|&d| d) {
            r.rescues += 1;
        }
    }
    r
}

/// Arm B — the incumbent (`modelless_cap` alone): escalate every miss while
/// the cap holds, in corpus order. No classification.
#[must_use]
pub fn arm_incumbent(corpus: &[EvalState], cap_units: f64) -> ArmReport {
    let mut r = ArmReport::default();
    let mut budget = cap_units;
    for state in corpus {
        if budget < C_ESC {
            break; // the cap is exhausted — later states get nothing
        }
        budget -= C_ESC;
        r.escalations += 1;
        r.spend_units += C_ESC;
        if state.planted == PlantedClass::KnowledgeLike {
            r.knowledge_escalations += 1;
        }
        let local = state.local_draws.iter().any(|&d| d);
        if local {
            r.rescues += 1;
        } else if state.planted == PlantedClass::KnowledgeLike && state.escalation_draw {
            // The scissors law: only a knowledge-class state is rescuable by
            // escalation; on the other classes the expert does no better
            // than the exhausted local path.
            r.rescues += 1;
            r.escalation_net_rescues += 1;
        }
    }
    r
}

/// The probe arm's tunables (the Phase-2 ladder + the Phase-3 gate inputs).
#[derive(Debug, Clone, Copy)]
pub struct ProbeArmConfig {
    /// The Wilson bar ε (a state is knowledge-like when the optimistic bound
    /// sits below it).
    pub epsilon: f64,
    /// The adaptive ladder's cap.
    pub n_max: u32,
    /// The escalation ladder fed to the gate: one depth, the declared solve
    /// prior, zero corpus distance (the plan's λ economics live at the
    /// consumer; the eval holds λ = 0.1, FlyBy's operating point).
    pub option: EscalationOption,
    /// λ — the corpus-distance penalty (FlyBy's 0.1; Research 609's λ=0.2
    /// failure is the ceiling argument).
    pub lambda: f64,
    /// The gate's `r_min` — the depth's prior must clear it to be usable.
    pub r_min: f64,
    /// The G2 budget: probe spend above `PROBE_BUDGET_FRACTION × escalation
    /// spend so far` flips the arm into incumbent fallback for the REST of
    /// the corpus (loud, reported on [`ArmReport::budget_fallback`]).
    pub probe_eval_cost_units: f64,
}

impl Default for ProbeArmConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            n_max: 128,
            option: EscalationOption { depth: 1, success_prior: R_ESC, corpus_distance: 0.0 },
            lambda: 0.1,
            r_min: 0.3,
            probe_eval_cost_units: PROBE_EVAL_COST_UNITS,
        }
    }
}

/// The probe's full read over one state: the adaptive ladder walk (evals
/// counted), the settled class (a capped walk reads conservative
/// `Undetermined` — never an escalation), and the FINAL-rung estimate the
/// gate scores. Intermediate rungs pass an empty histogram (answers-not-
/// tracked is a documented legal posture; the class rule reads only the
/// Wilson interval — the entropy term is informational at the gate).
struct StateProbeRead {
    class: ProbeClass,
    estimate: crate::state_probe::ProbeEstimate,
    evals: u32,
}

fn probe_state(state: &EvalState, cfg: &ProbeArmConfig, z95: f64) -> StateProbeRead {
    let mut flips = [ClassFlip {
        at_n: 0,
        from: ProbeClass::Undetermined,
        to: ProbeClass::Undetermined,
        interval_width: 0.0,
    }; 4];
    let mut classifier = AdaptiveClassifier::new(cfg.epsilon, cfg.n_max, &mut flips);
    let mut evals = 0u32;
    let mut last_estimate = None;
    for (i, &rung) in state.ensemble.rungs.iter().enumerate() {
        evals += 1;
        let input = ProbeInput {
            pass_count: state.ensemble.passes_at_rung[i],
            n: rung,
            histogram: if i == N_RUNGS - 1 { &state.ensemble.histogram } else { &[] },
        };
        let est = probe(&input, z95);
        last_estimate = Some(est);
        match classifier.observe(&est, rung) {
            AdaptiveDecision::Settled(class) => {
                return StateProbeRead { class, estimate: est, evals };
            }
            AdaptiveDecision::Continue => {}
            AdaptiveDecision::CappedUndetermined => break,
        }
    }
    StateProbeRead {
        class: ProbeClass::Undetermined,
        estimate: last_estimate.expect("the ladder is non-empty"),
        evals,
    }
}

/// Arm C — probe-gated: classify adaptively, escalate only `KnowledgeLike`,
/// same cap as the incumbent. Every probe evaluation is counted against the
/// G2 budget; a breach falls back to the incumbent rule for the remaining
/// states (T4.3's "never silently eat the budget").
#[must_use]
pub fn arm_probe_gated(corpus: &[EvalState], cap_units: f64, cfg: &ProbeArmConfig) -> ArmReport {
    let mut r = ArmReport::default();
    let mut budget = cap_units;
    let mut probe_spend;
    let mut fallback = false;
    let z95 = 1.959_963_984_540_054;
    for state in corpus {
        let local = state.local_draws.iter().any(|&d| d);
        let mut escalated = false;
        if !fallback {
            let read = probe_state(state, cfg, z95);
            r.probe_evals += read.evals;
            probe_spend = f64::from(r.probe_evals) * cfg.probe_eval_cost_units;
            if read.class == ProbeClass::KnowledgeLike && budget >= C_ESC {
                let decision = gate(
                    &GateInput {
                        probe: read.estimate,
                        options: &[cfg.option],
                        lambda: cfg.lambda,
                    },
                    cfg.epsilon,
                    cfg.r_min,
                );
                escalated = matches!(decision, GateDecision::Escalate { .. });
            }
            // The budget guard: probe spend vs the escalation spend so far.
            if r.spend_units > 0.0 && probe_spend > PROBE_BUDGET_FRACTION * r.spend_units {
                fallback = true; // T4.3: fall back to the incumbent rule, loudly.
                r.budget_fallback = true;
            }
        } else {
            // Incumbent fallback posture: the cap-only rule decides.
            escalated = budget >= C_ESC;
        }
        if escalated {
            budget -= C_ESC;
            r.escalations += 1;
            r.spend_units += C_ESC;
            if state.planted == PlantedClass::KnowledgeLike {
                r.knowledge_escalations += 1;
            }
        }
        if local {
            r.rescues += 1;
        } else if escalated && state.planted == PlantedClass::KnowledgeLike && state.escalation_draw {
            r.rescues += 1;
            r.escalation_net_rescues += 1;
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The corpus builder is deterministic: same seed, same bytes.
    #[test]
    fn corpus_is_seed_pinned() {
        let a = planted_corpus(20, 12, 8, 0xABCD);
        let b = planted_corpus(20, 12, 8, 0xABCD);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.planted, y.planted);
            assert_eq!(x.local_draws, y.local_draws);
            assert_eq!(x.escalation_draw, y.escalation_draw);
            assert_eq!(x.ensemble.passes_at_rung, y.ensemble.passes_at_rung);
        }
    }

    /// The planted blocks are INTERLEAVED round-robin (the layout contract:
    /// corpus order is not class-aligned, so the cap-only incumbent's blind
    /// spend hits all classes in stream order).
    #[test]
    fn corpus_blocks_are_interleaved() {
        let c = planted_corpus(5, 3, 2, 7);
        assert_eq!(c[0].planted, PlantedClass::KnowledgeLike);
        assert_eq!(c[1].planted, PlantedClass::ExecutionLike);
        assert_eq!(c[2].planted, PlantedClass::Productive);
        assert_eq!(c[3].planted, PlantedClass::KnowledgeLike);
        assert_eq!(c.iter().filter(|s| s.planted == PlantedClass::KnowledgeLike).count(), 5);
        assert_eq!(c.iter().filter(|s| s.planted == PlantedClass::ExecutionLike).count(), 3);
        assert_eq!(c.iter().filter(|s| s.planted == PlantedClass::Productive).count(), 2);
    }

    /// The floor arm counts exactly the locally-rescuable states.
    #[test]
    fn floor_counts_local_only() {
        let c = planted_corpus(10, 10, 10, 42);
        let r = arm_never_escalate(&c);
        let want = c.iter().filter(|s| s.local_draws.iter().any(|&d| d)).count() as u32;
        assert_eq!(r.rescues, want);
        assert_eq!(r.escalations, 0);
    }

    /// The incumbent respects the cap exactly (never overspends).
    #[test]
    fn incumbent_respects_the_cap() {
        let c = planted_corpus(4, 4, 4, 9);
        let r = arm_incumbent(&c, 5.0);
        assert_eq!(r.escalations, 5);
        assert!((r.spend_units - 5.0).abs() < 1e-9);
    }
}
