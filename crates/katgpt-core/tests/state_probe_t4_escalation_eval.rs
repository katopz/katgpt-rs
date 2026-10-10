//! Plan 621 Phase 4 — the healer-lane escalation evaluation (T4.1–T4.3).
//!
//! G1 law: a gate that never fires FAILS this suite (the never-escalate
//! floor arm + the fires-assertion). The comparison table prints on every
//! run (`--nocapture`); `.benchmarks/621_escalation_eval_goat.md` freezes
//! the recorded read.
//!
//! Run: `cargo test -p katgpt-core --features escalation_probe_gate --test
//! state_probe_t4_escalation_eval -- --nocapture`.

#![cfg(feature = "escalation_probe_gate")]

use katgpt_core::state_probe_eval::{
    ArmReport, C_ESC, PlantedClass, ProbeArmConfig, arm_incumbent, arm_never_escalate,
    arm_probe_gated, planted_corpus,
};

/// The frozen corpus shape + seed (the .benchmarks record cites both; a
/// re-pin is a deliberate edit to this file AND the record in one commit).
const SEED: u64 = 0x621_621_621_621;
const N_KNOWLEDGE: usize = 60;
const N_EXECUTION: usize = 60;
const N_PRODUCTIVE: usize = 30;

/// The shared cap: generous enough that the uncapped incumbent escalates
/// EVERY state (its rescues = the corpus's max), so the G1 rescue comparison
/// is probe-vs-ceiling, and TIGHT_ENOUGH below for the matched-spend cell.
const CAP_GENEROUS: f64 = 150.0;
/// The tight cap: ~1/3 of the corpus — the incumbent burns it in corpus
/// order (knowledge block first, but it also pays for execution/productive
/// states as they come); the probe spends only on classified knowledge-like
/// states and reaches deeper into the corpus.
const CAP_TIGHT: f64 = 50.0;

fn corpus() -> Vec<katgpt_core::state_probe_eval::EvalState> {
    planted_corpus(N_KNOWLEDGE, N_EXECUTION, N_PRODUCTIVE, SEED)
}

fn print_table(title: &str, floor: &ArmReport, incumbent: &ArmReport, probe: &ArmReport) {
    println!("\n{title}");
    println!(
        "{:<22} {:>8} {:>12} {:>10} {:>14} {:>12} {:>10}",
        "arm", "rescues", "esc-net", "escs", "spend (units)", "probe evals", "esc-prec"
    );
    for (name, r) in [("never-escalate", floor), ("incumbent (cap)", incumbent), ("probe-gated", probe)] {
        println!(
            "{:<22} {:>8} {:>12} {:>10} {:>14.1} {:>12} {:>9.2}",
            name, r.rescues, r.escalation_net_rescues, r.escalations, r.spend_units, r.probe_evals,
            r.escalation_precision()
        );
    }
}

/// T4.1 — the corpus carries BOTH classes in force: a gate that never fires
/// fails G1, so the knowledge block must be large enough to be unmissable
/// and the floor arm must leave real headroom for escalation.
#[test]
fn corpus_carries_both_classes_and_the_floor_leaves_headroom() {
    let c = corpus();
    assert_eq!(c.len(), N_KNOWLEDGE + N_EXECUTION + N_PRODUCTIVE);
    let knowledge = c.iter().filter(|s| s.planted == PlantedClass::KnowledgeLike).count();
    let execution = c.iter().filter(|s| s.planted == PlantedClass::ExecutionLike).count();
    assert_eq!(knowledge, N_KNOWLEDGE, "the escalate-worthy class must be present");
    assert_eq!(execution, N_EXECUTION, "the waste-prone class must be present");

    // Headroom: locally-unrescuable states exist at scale (escalation is
    // the only channel for them), so a gate that fires can beat the floor.
    let floor = arm_never_escalate(&c);
    let headroom = c.len() as u32 - floor.rescues;
    assert!(
        headroom > N_KNOWLEDGE as u32 / 2,
        "floor rescues {floor:?} leave only {headroom} headroom — the corpus cannot discriminate"
    );
}

/// T4.2 G1 (generous cap) — the probe gate FIRES, never under-rescues the
/// incumbent, and never overspends it. The floor arm sits strictly below
/// both (escalation adds net rescues on this corpus).
#[test]
fn generous_cap_probe_fires_and_dominates_the_floor() {
    let c = corpus();
    let cfg = ProbeArmConfig::default();
    let floor = arm_never_escalate(&c);
    let incumbent = arm_incumbent(&c, CAP_GENEROUS);
    let probe_arm = arm_probe_gated(&c, CAP_GENEROUS, &cfg);
    print_table("G1 generous cap (uncapped incumbent = every state)", &floor, &incumbent, &probe_arm);

    // THE GATE FIRES (the never-fires failure).
    assert!(probe_arm.escalations > 0, "probe-gated never escalated — G1 fails by the floor law");
    assert!(!probe_arm.budget_fallback, "the G2 budget breached at the declared cost model — the arm silently degraded");

    // The probe is a strict improvement on the floor (escalation nets value)
    assert!(probe_arm.rescues > floor.rescues, "probe-gated must beat the never-escalate floor");
    // …and never under-rescues the incumbent while spending far less — under
    // the scissors law the incumbent's extra escalations rescue nothing the
    // probe misses, so rescues TIE at generous cap and spend does not.
    assert!(
        probe_arm.rescues >= incumbent.rescues,
        "probe {probe_arm:?} under-rescued the incumbent {incumbent:?}"
    );
    assert!(
        probe_arm.spend_units < incumbent.spend_units,
        "probe spend {} must be strictly below the incumbent's {}",
        probe_arm.spend_units,
        incumbent.spend_units
    );
    // The probe's escalations are ~all TRUE knowledge-like states —
    // precision is the point of the gate (the incumbent's is ~1/3 on the
    // interleaved stream).
    assert!(
        probe_arm.escalation_precision() > 0.9,
        "probe escalation precision {:.2} — the gate is escalating non-knowledge states",
        probe_arm.escalation_precision()
    );
    assert!(
        probe_arm.escalation_precision() > incumbent.escalation_precision(),
        "probe precision {:.2} vs incumbent {:.2}",
        probe_arm.escalation_precision(),
        incumbent.escalation_precision()
    );
}

/// T4.2 G1 (tight cap, matched spend) — under the SAME cap the probe
/// directs the budget to the escalate-worthy block and strictly beats the
/// incumbent's rescues: the FlyBy triage value, measured.
#[test]
fn tight_cap_probe_strictly_beats_the_incumbent_at_matched_spend() {
    let c = corpus();
    let cfg = ProbeArmConfig::default();
    let incumbent = arm_incumbent(&c, CAP_TIGHT);
    let probe_arm = arm_probe_gated(&c, CAP_TIGHT, &cfg);
    let floor = arm_never_escalate(&c);
    print_table("G1 tight cap (matched spend, corpus order)", &floor, &incumbent, &probe_arm);

    // Same cap ⇒ the probe's spend ≤ the incumbent's (it escalates less).
    assert!(probe_arm.spend_units <= incumbent.spend_units + 1e-9);
    // The probe's escalation PRECISION is the mechanism: a strictly higher
    // share of its escalations lands on states only escalation can rescue.
    assert!(
        probe_arm.escalation_precision() > incumbent.escalation_precision(),
        "probe precision {:.2} vs incumbent {:.2} — the gate is not triaging",
        probe_arm.escalation_precision(),
        incumbent.escalation_precision()
    );
    // And at matched spend the probe rescues strictly more.
    assert!(
        probe_arm.rescues > incumbent.rescues,
        "matched-spend rescues: probe {} vs incumbent {} — the Super-GOAT claim fails",
        probe_arm.rescues,
        incumbent.rescues
    );
}

/// T4.3 — the budget guard's happy path at the declared cost model: the
/// probe's evaluation spend is orders below the 10% line (an escalation is
/// a hosted call; a probe is a ns-class scan).
#[test]
fn budget_holds_at_the_declared_cost_model() {
    let c = corpus();
    let cfg = ProbeArmConfig::default();
    let probe_arm = arm_probe_gated(&c, CAP_GENEROUS, &cfg);
    assert!(!probe_arm.budget_fallback, "the guard breached at the declared model — re-price PROBE_EVAL_COST_UNITS or fix the ladder");
    let probe_spend = f64::from(probe_arm.probe_evals) * katgpt_core::state_probe_eval::PROBE_EVAL_COST_UNITS;
    assert!(
        probe_spend <= 0.1 * probe_arm.spend_units,
        "probe spend {probe_spend:.4} vs 10% of escalation spend {}",
        0.1 * probe_arm.spend_units
    );
}

/// T4.3 — the breach→fallback path: a deliberately absurd per-eval cost
/// forces the guard, and the arm finishes the corpus under the incumbent
/// rule LOUDLY (the flag is set, never a silent degradation).
#[test]
fn budget_breach_falls_back_to_the_incumbent_loudly() {
    let c = corpus();
    // One probe evaluation costs more than an escalation: the first state
    // with any escalation spend breaches the 10% line immediately.
    let cfg = katgpt_core::state_probe_eval::ProbeArmConfig {
        probe_eval_cost_units: 10.0 * C_ESC,
        ..ProbeArmConfig::default()
    };
    let probe_arm = arm_probe_gated(&c, CAP_GENEROUS, &cfg);
    assert!(probe_arm.budget_fallback, "an absurd probe cost must trip the guard");
    // The fallback posture escalates like the incumbent (cap-only), so its
    // escalation count approaches the incumbent's on the remaining states.
    let incumbent = arm_incumbent(&c, CAP_GENEROUS);
    assert!(
        probe_arm.escalations >= incumbent.escalations - 1,
        "fallback arm escalated {} vs incumbent {} — the fallback is not the incumbent rule",
        probe_arm.escalations,
        incumbent.escalations
    );
}

/// Determinism: the same seed realizes the same corpus and the same reports
/// (the .benchmarks record freezes one read; this pins its reproducibility).
#[test]
fn eval_is_bit_reproducible() {
    let c1 = corpus();
    let c2 = planted_corpus(N_KNOWLEDGE, N_EXECUTION, N_PRODUCTIVE, SEED);
    let cfg = ProbeArmConfig::default();
    let a1 = arm_probe_gated(&c1, CAP_TIGHT, &cfg);
    let a2 = arm_probe_gated(&c2, CAP_TIGHT, &cfg);
    assert_eq!(a1, a2);
}
