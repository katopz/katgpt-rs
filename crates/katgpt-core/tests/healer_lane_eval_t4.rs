//! Plan 621 Phase 4 — the healer-lane eval suite (T4.1–T4.4).
//!
//! The Super-GOAT decider: over ONE planted two-class fix-miss corpus and
//! ONE shared outcome stream (paired arms), the probe-gated fusion must beat
//! the `modelless_cap` incumbent at matched spend (T4.4's bar), the gate must
//! FIRE where escalation is the only rescue (T4.1 — the never-escalate floor
//! is the control that must lose), and the probe's cost line must fall back
//! loudly when breached (T4.3). The table's blake3 pins cross-run (and via
//! `bench_621b`, cross-profile) determinism.
//!
//! The engine lives in `common/plan621_healer_sim.rs` (`#[path]`-included by
//! this suite and the bench — eval-only, never a shipped API).

#![cfg(feature = "escalation_probe_gate")]

#[path = "common/plan621_healer_sim.rs"]
mod sim;

use sim::{ArmName, EPSILON, N_EXEC, N_KNOWLEDGE, N_MAX, N_MIDDLE, PROBE_COST_FRAC};

/// The default record posture: a hosted-expert escalation costs 500 local
/// draft+verify evaluations. The mid posture of the sweep {100, 250, 500,
/// 1000}; the T4.3 line at ε=0.1 settles knowledge-like pools at n=40, so
/// the line passes here (40 ≤ 50) and breaches below c_depth 400.
const C_DEPTH_DEFAULT: f64 = 500.0;

fn corpus() -> sim::Corpus {
    sim::gen_corpus()
}

fn draws(c: &sim::Corpus) -> sim::OutcomeStream {
    sim::OutcomeStream::draw(c)
}

// ── T4.1 — the corpus carries BOTH classes and the gate FIRES ───────────

#[test]
fn t41_corpus_plants_both_classes_and_class_counts_hold() {
    let c = corpus();
    assert_eq!(c.states.len(), N_EXEC + N_KNOWLEDGE + N_MIDDLE);
    let exec = c
        .states
        .iter()
        .filter(|s| s.kind == sim::Kind::Exec)
        .count();
    let kl = c
        .states
        .iter()
        .filter(|s| s.kind == sim::Kind::Knowledge)
        .count();
    assert_eq!(exec, N_EXEC, "the execution-like class must be planted");
    assert_eq!(kl, N_KNOWLEDGE, "the knowledge-like class must be planted");
    // The classes are separated where it matters: every knowledge-like true
    // rate sits BELOW ε (escalation is the only reliable rescue), every
    // execution-like rate sits ABOVE it.
    for s in &c.states {
        match s.kind {
            sim::Kind::Knowledge => assert!(
                s.p_local < EPSILON,
                "knowledge-like state planted above ε ({})",
                s.p_local
            ),
            sim::Kind::Exec => assert!(
                s.p_local > EPSILON,
                "execution-like state planted below ε ({})",
                s.p_local
            ),
            sim::Kind::Middle => {}
        }
    }
}

#[test]
fn t41_gate_fires_on_knowledge_states_and_floor_arm_never_escalates() {
    let c = corpus();
    let d = draws(&c);
    let floor = sim::run_floor(&c, &d);
    assert_eq!(
        floor.outcome.escalations, 0,
        "the floor arm is the never-escalate control — it must never escalate"
    );

    let probe = sim::run_probe_gated(&c, &d, C_DEPTH_DEFAULT, N_MAX);
    // The gate must FIRE: knowledge-like states whose local channel ran dry
    // (no pass through the settle point) must escalate. Count them from the
    // per-state results: knowledge states whose action is Escalate.
    let kl_escalated = c
        .states
        .iter()
        .zip(&probe.results)
        .filter(|(s, r)| {
            s.kind == sim::Kind::Knowledge && matches!(r.action, sim::Action::Escalate { .. })
        })
        .count();
    // Upper bound: knowledge states that could NOT escalate because a lucky
    // local pass rescued them first (the local channel legitimately closes
    // the state before any decision). Everything else must have escalated.
    let kl_local_rescued = c
        .states
        .iter()
        .zip(&probe.results)
        .filter(|(s, r)| s.kind == sim::Kind::Knowledge && r.action == sim::Action::LocalRescue)
        .count();
    assert!(
        kl_escalated > 0,
        "G1 FAILURE — the gate never fired: a never-escalate policy is indistinguishable \
         from the probe-gated arm (the corpus or the classifier is degenerate)"
    );
    assert_eq!(
        kl_escalated + kl_local_rescued,
        N_KNOWLEDGE,
        "every knowledge-like state must end in exactly one of {{escalate, local-rescue}}"
    );
    // Knowledge-like states must never take the conservative stay at
    // n_max=64: the all-fail pool settles KnowledgeLike at n=40, before
    // the cap — a stay here would mean the classifier stalled.
    let kl_stayed = c
        .states
        .iter()
        .zip(&probe.results)
        .filter(|(s, r)| s.kind == sim::Kind::Knowledge && r.action == sim::Action::Stay)
        .count();
    assert_eq!(
        kl_stayed, 0,
        "knowledge-like states must never take the conservative stay at n_max=64 — \
         the all-fail pool settles KnowledgeLike at n=40 before the cap"
    );
    // …and the escalation set must be ALMOST exclusively knowledge-like:
    // execution states carry 15–45% verify rates; a settled-KL on one is a
    // rare unlucky draw, not a pattern.
    let exec_escalated = c
        .states
        .iter()
        .zip(&probe.results)
        .filter(|(s, r)| {
            s.kind == sim::Kind::Exec && matches!(r.action, sim::Action::Escalate { .. })
        })
        .count();
    assert!(
        exec_escalated <= 3,
        "the gate escalated {exec_escalated} execution-like states — the classifier is \
         misrouting locally-reachable fixes (expected ≤ 3 unlucky draws of 100)"
    );
}

// ── T4.2 — the comparison table and the Super-GOAT ordering ─────────────

#[test]
fn t42_table_ordering_floor_loses_both_arms_win_spend() {
    let c = corpus();
    let d = draws(&c);
    let floor = sim::run_floor(&c, &d);
    let incumbent = sim::run_incumbent_cap(&c, &d, C_DEPTH_DEFAULT, None);
    let probe = sim::run_probe_gated(&c, &d, C_DEPTH_DEFAULT, N_MAX);

    sim::print_table(
        &[
            sim::table_row(ArmName::NeverEscalateFloor, &floor),
            sim::table_row(ArmName::IncumbentCap, &incumbent),
            sim::table_row(ArmName::ProbeGated, &probe),
        ],
        c.states.len(),
        C_DEPTH_DEFAULT,
    );

    // The floor must lose rescues to BOTH arms — the corpus makes escalation
    // necessary (T4.1's gate-must-fire property, seen in outcomes).
    assert!(
        floor.outcome.rescued < incumbent.outcome.rescued,
        "the never-escalate floor out-rescued the incumbent — the corpus has no \
         escalation-only rescues (degenerate corpus)"
    );
    assert!(
        floor.outcome.rescued < probe.outcome.rescued,
        "the never-escalate floor out-rescued the probe-gated fusion — degenerate corpus"
    );

    // The fusion's raw dominance at the default posture: MORE rescues on
    // LESS spend than the incumbent (the escalated-state triage: the
    // incumbent escalates execution-like pool-misses the fusion rescues
    // locally for ~1% of an escalation's cost).
    assert!(
        probe.outcome.rescued > incumbent.outcome.rescued,
        "probe-gated rescued {} vs incumbent {} — the fusion must win rescues",
        probe.outcome.rescued,
        incumbent.outcome.rescued
    );
    assert!(
        probe.outcome.spend < incumbent.outcome.spend,
        "probe-gated spend {:.0} vs incumbent {:.0} — the fusion must win spend",
        probe.outcome.spend,
        incumbent.outcome.spend
    );

    // And the fusion must NEVER rescue fewer than the floor (its stay paths
    // are strictly informed versions of the floor's draws).
    assert!(probe.outcome.rescued >= floor.outcome.rescued);
}

#[test]
fn t42_super_goat_beats_the_incumbent_at_matched_spend() {
    let c = corpus();
    let d = draws(&c);
    let probe = sim::run_probe_gated(&c, &d, C_DEPTH_DEFAULT, N_MAX);

    // Hand the incumbent EXACTLY the fusion's budget (binary-searched cap).
    let (cap, incumbent_capped) =
        sim::incumbent_cap_for_spend(&c, &d, C_DEPTH_DEFAULT, probe.outcome.spend);
    println!(
        "── T4.4 Super-GOAT at matched spend (incumbent cap {cap:.0} ≈ probe spend {:.0}) ──",
        probe.outcome.spend
    );
    println!(
        "probe-gated: rescued {} / spend {:.0}  |  incumbent@cap: rescued {} / spend {:.0}",
        probe.outcome.rescued,
        probe.outcome.spend,
        incumbent_capped.outcome.rescued,
        incumbent_capped.outcome.spend
    );
    assert!(
        incumbent_capped.outcome.spend <= probe.outcome.spend + C_DEPTH_DEFAULT,
        "the matched-spend search must land the incumbent AT the fusion's budget \
         ({} vs {}) — the search is broken",
        incumbent_capped.outcome.spend,
        probe.outcome.spend
    );
    assert!(
        probe.outcome.rescued > incumbent_capped.outcome.rescued,
        "SUPER-GOAT FAILURE — at matched spend the incumbent ({}) matched the fusion ({})",
        incumbent_capped.outcome.rescued,
        probe.outcome.rescued
    );
    // The margin is real, not a one-fix artifact: the fusion must rescue at
    // least 5% more fixes at equal spend.
    let margin = probe.outcome.rescued as f64 / incumbent_capped.outcome.rescued as f64;
    println!("matched-spend rescue ratio: {margin:.4}");
    assert!(
        margin >= 1.05,
        "the matched-spend margin collapsed to {margin:.4} — the triage advantage is gone"
    );
}

// ── T4.3 — the probe cost line and the loud breach fallback ─────────────

#[test]
fn t43_cost_line_passes_at_default_and_every_escalation_is_armed() {
    let c = corpus();
    let d = draws(&c);
    let probe = sim::run_probe_gated(&c, &d, C_DEPTH_DEFAULT, N_MAX);
    // At c_depth 500 the line is probe_cost ≤ 50; knowledge pools settle at
    // n=40 — inside budget, zero breaches expected.
    assert_eq!(
        probe.outcome.breach_fallbacks, 0,
        "breaches at the default posture — the cost constants moved without re-deriving the line"
    );
    // Every escalation carried probe cost inside the line (re-derived from
    // the per-state draws, not trusted from the aggregate).
    for (s, r) in c.states.iter().zip(&probe.results) {
        if let sim::Action::Escalate { breach_fallback } = r.action {
            assert!(
                !breach_fallback,
                "state p={:.3}: an escalation rode a breach fallback at the default posture",
                s.p_local
            );
            assert!(
                f64::from(r.draws_used) <= PROBE_COST_FRAC * C_DEPTH_DEFAULT,
                "escalation at draws_used={} exceeds the 10% line ({})",
                r.draws_used,
                PROBE_COST_FRAC * C_DEPTH_DEFAULT
            );
        }
    }
}

#[test]
fn t43_breach_posture_falls_back_loudly_and_never_underscores_the_incumbent() {
    let c = corpus();
    let d = draws(&c);
    // c_depth 250: the line is 25 draws; knowledge pools settle at 40 —
    // EVERY gated decision breaches, and every breach falls back to the
    // incumbent decision (escalate) with the fallback COUNTED.
    let c_depth = 250.0;
    let probe = sim::run_probe_gated(&c, &d, c_depth, N_MAX);
    let incumbent = sim::run_incumbent_cap(&c, &d, c_depth, None);
    assert!(
        probe.outcome.escalations > 0,
        "the breach posture must still escalate (the fallback IS the incumbent decision)"
    );
    assert_eq!(
        probe.outcome.breach_fallbacks, probe.outcome.escalations,
        "every escalation at c_depth 250 must be a disclosed breach fallback"
    );
    sim::print_table(
        &[
            sim::table_row(ArmName::IncumbentCap, &incumbent),
            sim::table_row(ArmName::ProbeGated, &probe),
        ],
        c.states.len(),
        c_depth,
    );
    // The fallback never underscores the incumbent: the probe arm's extended
    // local channel only ADDS rescues (states the incumbent escalated, the
    // probe often rescues locally first); allow a small paired-draw
    // tolerance for states where the incumbent's escalation got lucky while
    // the probe's late local draws missed.
    assert!(
        probe.outcome.rescued + 5 >= incumbent.outcome.rescued,
        "the breach fallback scored WORSE than the incumbent it falls back to ({} vs {}) — \
         the fallback is not the incumbent decision",
        probe.outcome.rescued,
        incumbent.outcome.rescued
    );
}

// ── T4.4 — the ratio sweep + the determinism pins ───────────────────────

#[test]
fn t44_ratio_sweep_verdict_holds_across_the_affordable_range() {
    let c = corpus();
    let d = draws(&c);
    println!("── T4.4 cost-ratio sweep (the verdict is never one constant) ──");
    for c_depth in [100.0, 250.0, 500.0, 1000.0] {
        let incumbent = sim::run_incumbent_cap(&c, &d, c_depth, None);
        let probe = sim::run_probe_gated(&c, &d, c_depth, N_MAX);
        let (_cap, matched) = sim::incumbent_cap_for_spend(&c, &d, c_depth, probe.outcome.spend);
        println!(
            "c_depth {c_depth:>5.0}: incumbent rescued {:3} spend {:7.0} | probe rescued {:3} \
             spend {:7.0} (breaches {}) | matched-spend incumbent rescued {}",
            incumbent.outcome.rescued,
            incumbent.outcome.spend,
            probe.outcome.rescued,
            probe.outcome.spend,
            probe.outcome.breach_fallbacks,
            matched.outcome.rescued,
        );
        // The fusion never rescues fewer than the incumbent at ANY posture
        // in the sweep (its local channel is a superset of the incumbent's).
        assert!(
            probe.outcome.rescued + 5 >= incumbent.outcome.rescued,
            "c_depth {c_depth}: fusion ({}) fell below the incumbent ({})",
            probe.outcome.rescued,
            incumbent.outcome.rescued
        );
        // At the affordable postures (no breaches) the raw dominance holds.
        if probe.outcome.breach_fallbacks == 0 {
            assert!(
                probe.outcome.rescued > incumbent.outcome.rescued
                    && probe.outcome.spend < incumbent.outcome.spend,
                "c_depth {c_depth}: the affordable posture lost raw dominance"
            );
            assert!(
                probe.outcome.rescued > matched.outcome.rescued,
                "c_depth {c_depth}: matched-spend win evaporated ({} vs {})",
                probe.outcome.rescued,
                matched.outcome.rescued
            );
        }
    }
}

#[test]
fn t44_probe_ci_coverage_measured_on_the_run() {
    let c = corpus();
    let d = draws(&c);
    let probe = sim::run_probe_gated(&c, &d, C_DEPTH_DEFAULT, N_MAX);
    assert!(
        probe.outcome.probe_observations > 100,
        "the run took too few observations to measure coverage"
    );
    let coverage = probe.outcome.coverage_hits as f64 / probe.outcome.probe_observations as f64;
    println!(
        "── G1 UQ note — Wilson CI empirical coverage on the run: {coverage:.4} over {} observations ──",
        probe.outcome.probe_observations
    );
    // All-fail pools make the upper bound the live side (lo = 0); the
    // straddling middle band's late observations are the known
    // under-coverage cases. The floor is honest, not flattering: 85%.
    assert!(
        coverage >= 0.85,
        "Wilson CI coverage {coverage:.4} collapsed below the honest floor — the interval \
         is not usable even as a screening read"
    );
}

#[test]
fn t44_table_blake3_replayable_across_runs() {
    let c = corpus();
    let d = draws(&c);
    let a = sim::default_table(&c, &d, C_DEPTH_DEFAULT);
    let b = sim::default_table(&c, &d, C_DEPTH_DEFAULT);
    let ha = sim::table_blake3(&a);
    let hb = sim::table_blake3(&b);
    assert_eq!(ha, hb, "the table bytes moved between identical runs");
    println!("table blake3: {}", hex(&ha));
}

/// Hex helper (the pin's display form; `bench_621b` asserts the same bytes).
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
