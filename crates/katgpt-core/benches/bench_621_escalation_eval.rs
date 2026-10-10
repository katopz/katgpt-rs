//! Plan 621 Phase 4 — the escalation eval's G2 cost cell (T4.3).
//!
//! Two numbers, both asserted:
//!
//! 1. **Probe wall cost per corpus state** across the full adaptive ladder
//!    (8 → 128 members, worst case = the capped walk). The T1.4 micro-bench
//!    pinned the kernel at ~167 ns/call; this cell prices the EVAL's real
//!    per-state cost (≤ 5 ladder probes + the gate) against the escalation
//!    action the gate controls.
//! 2. **The declared-units budget ratio** — the same accounting the G1 run
//!    uses ([`PROBE_EVAL_COST_UNITS`] vs [`C_ESC`]) over the recorded
//!    corpus, asserted ≤ the 10 % line.
//!
//! Every timed loop accumulates into a black-boxed sink (the absent-work
//! defence); the bar is a release-profile claim (`cargo bench` is release
//! by construction).
//!
//! Run: `cargo bench -p katgpt-core --features escalation_probe_gate --bench
//! bench_621_escalation_eval`.

#![cfg(feature = "escalation_probe_gate")]

use katgpt_core::state_probe_eval::{
    C_ESC, PROBE_BUDGET_FRACTION, PROBE_EVAL_COST_UNITS, ProbeArmConfig, arm_probe_gated,
    planted_corpus,
};
use std::hint::black_box;

/// The recorded corpus shape (the .benchmarks record's shape; the G1 test
/// pins the same one at the same seed).
const SEED: u64 = 0x621_621_621_621;
const N_STATES: usize = 150;

fn main() {
    let corpus = planted_corpus(60, 60, 30, SEED);
    assert_eq!(corpus.len(), N_STATES);
    let cfg = ProbeArmConfig::default();

    // ── 1. per-state probe wall cost (worst case: full ladder + gate) ──
    let iterations = 200_usize;
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let mut sink = 0.0f64;
        for state in black_box(&corpus) {
            let report = arm_probe_gated(black_box(std::slice::from_ref(state)), 1.0, &cfg);
            sink += report.probe_evals as f64;
        }
        black_box(sink);
        samples.push(start.elapsed().as_secs_f64() * 1e9 / corpus.len() as f64);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let median_ns_per_state = samples[samples.len() / 2];
    println!(
        "probe arm per-state cost (full ladder + gate, corpus of {N_STATES}): {median_ns_per_state:.1} ns/state (median of {iterations})"
    );
    assert!(
        median_ns_per_state < 5_000.0,
        "probe arm {median_ns_per_state:.1} ns/state exceeds the 5 µs ceiling — the ladder or the gate regressed"
    );

    // ── 2. the declared-units budget ratio over the recorded corpus ──
    let report = arm_probe_gated(&corpus, 150.0, &cfg);
    let probe_spend = f64::from(report.probe_evals) * PROBE_EVAL_COST_UNITS;
    let line = PROBE_BUDGET_FRACTION * report.spend_units;
    println!(
        "probe evals {} → declared spend {probe_spend:.4} units vs escalation spend {:.1} units (10% line {line:.3}) — breach: {}",
        report.probe_evals,
        report.spend_units,
        report.budget_fallback
    );
    assert!(!report.budget_fallback, "the budget guard breached at the declared cost model");
    assert!(
        probe_spend <= line,
        "probe declared spend {probe_spend:.4} exceeds the 10% line {line:.3} (C_ESC = {C_ESC})"
    );
    println!("G2 cost cell PASS");
}
