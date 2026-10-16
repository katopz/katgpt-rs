//! Plan 621 Phase 4 T4.4 — the healer-lane Super-GOAT bench.
//!
//! Runs the three-arm eval (never-escalate floor | `modelless_cap`
//! incumbent | probe-gated fusion) at the RELEASE posture and asserts the
//! verdict the plan gates the `escalation_probe_gate` label on:
//!
//! * G1 — the gate FIRES on knowledge-like fix-misses (the never-escalate
//!   floor loses real rescues); the Wilson CI's empirical coverage prints
//!   beside the table (the GOAT summary's UQ note).
//! * Super-GOAT — the fusion beats the incumbent at matched spend, and at
//!   the default posture wins BOTH axes raw (more rescues, less spend).
//! * T4.3 — every escalation's probe cost sits inside the 10% line at the
//!   default posture; the breach posture (c_depth 250) is run and its
//!   fallback COUNT is disclosed (never a silent overrun).
//! * T4.4 — the table's blake3 must equal the test-suite pin
//!   (`healer_lane_eval_t4::t44_table_blake3_replayable_across_runs`
//!   prints it): f64 IEEE sequential ops are profile-independent, so the
//!   release bench and the debug suite must produce BYTE-identical tables.
//!
//! Run: `cargo bench -p katgpt-core --features escalation_probe_gate --bench
//! bench_621b_healer_lane_eval` (cargo bench is release by construction).

#![cfg(feature = "escalation_probe_gate")]

#[path = "../tests/common/plan621_healer_sim.rs"]
mod sim;

/// The default record posture — MUST equal the test suite's
/// `C_DEPTH_DEFAULT` (the blake3 pin crosses the two binaries).
const C_DEPTH_DEFAULT: f64 = 500.0;

/// The test-suite-pinned table digest (T4.4's cross-profile determinism
/// pin). Re-derive deliberately when the corpus, the constants, or the
/// kernel changes — a moved pin is a loud formula-change detector, not a
/// correctness oracle.
const TABLE_BLAKE3_PIN: &str = "c4a00f955e49783e9a8365e51b486930fdb271a066e1b481641577620d60ffc1";

fn main() {
    let corpus = sim::gen_corpus();
    let draws = sim::OutcomeStream::draw(&corpus);
    let n = corpus.states.len();

    println!("=== Plan 621 Phase 4 — healer-lane Super-GOAT bench (T4.4) ===");
    println!(
        "corpus: {n} planted fix-miss states ({} exec / {} knowledge / {} middle), \
         seed {:#x}, paired outcome stream",
        sim::N_EXEC,
        sim::N_KNOWLEDGE,
        sim::N_MIDDLE,
        sim::CORPUS_SEED
    );

    let floor = sim::run_floor(&corpus, &draws);
    let incumbent = sim::run_incumbent_cap(&corpus, &draws, C_DEPTH_DEFAULT, None);
    let probe = sim::run_probe_gated(&corpus, &draws, C_DEPTH_DEFAULT, sim::N_MAX);

    let rows = [
        sim::table_row(sim::ArmName::NeverEscalateFloor, &floor),
        sim::table_row(sim::ArmName::IncumbentCap, &incumbent),
        sim::table_row(sim::ArmName::ProbeGated, &probe),
    ];
    sim::print_table(&rows, n, C_DEPTH_DEFAULT);

    // The matched-spend cell (the Super-GOAT bar).
    let (cap, matched) =
        sim::incumbent_cap_for_spend(&corpus, &draws, C_DEPTH_DEFAULT, probe.outcome.spend);
    println!(
        "matched spend: incumbent cap {cap:.0} → rescued {} (probe: {})",
        matched.outcome.rescued, probe.outcome.rescued
    );

    // The breach posture (T4.3's loud-fallback half).
    let probe_breach = sim::run_probe_gated(&corpus, &draws, 250.0, sim::N_MAX);
    println!(
        "breach posture (c_depth 250): escalations {} / disclosed fallbacks {}",
        probe_breach.outcome.escalations, probe_breach.outcome.breach_fallbacks
    );

    // ── the asserted verdict ────────────────────────────────────────────
    let mut failures: Vec<&'static str> = Vec::new();

    if floor.outcome.escalations != 0 {
        failures.push("G1: the floor arm escalated (control broken)");
    }
    if probe.outcome.escalations == 0 {
        failures.push("G1: the gate never fired (never-escalate indistinguishable)");
    }
    if probe.outcome.rescued <= floor.outcome.rescued {
        failures.push("G1: the corpus does not make escalation necessary");
    }
    if probe.outcome.rescued <= incumbent.outcome.rescued
        || probe.outcome.spend >= incumbent.outcome.spend
    {
        failures.push("Super-GOAT: the fusion lost raw dominance at the default posture");
    }
    if probe.outcome.rescued <= matched.outcome.rescued {
        failures.push("Super-GOAT: the fusion failed at matched spend");
    }
    if probe.outcome.breach_fallbacks != 0 {
        failures.push("T4.3: breaches at the default posture (constants moved)");
    }
    if probe_breach.outcome.breach_fallbacks != probe_breach.outcome.escalations {
        failures.push("T4.3: the breach posture did not fall back on every gated decision");
    }

    let digest = sim::table_blake3(&rows);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    println!("table blake3: {hex}");
    if TABLE_BLAKE3_PIN != "PENDING_FIRST_RUN" && hex != TABLE_BLAKE3_PIN {
        failures.push("T4.4: the table digest moved off the cross-profile pin");
    }

    if failures.is_empty() {
        println!("   Phase 4 verdict ✓ — Super-GOAT PASS (see .benchmarks/ for the record)");
    } else {
        for f in &failures {
            println!("   ✗ {f}");
        }
        println!("   Phase 4 verdict ✗ — the fusion is NOT promoted on this evidence");
        std::process::exit(1);
    }
}
