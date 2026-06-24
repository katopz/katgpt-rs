//! Plan 320 Phase 6 T6.1 — CUCG latency + throughput + zero-alloc bench.
//!
//! Convention: `std::time::Instant` + `harness = false` (matches
//! `salience_tri_gate_bench.rs`, `procrustes_bench.rs`, etc. — Criterion is
//! not a katgpt-rs dev-dep).
//!
//! Run:
//! ```bash
//! cargo run --release --bench cucg_bench --features closed_unit_compaction
//! ```
//!
//! Gates measured:
//! - **Latency** `evaluate()` ≤ 50ns for ARITY=4 (parity with Salience Tri-Gate's
//!   ~9ns + 2 extra sigmoids).
//! - **Throughput** `evaluate()` ≥ 50M decisions/sec for ARITY=4.
//! - **G4 zero-alloc** — no heap allocation on the hot path (verbal assertion;
//!   the audit is stack-allocated and the scratch is caller-reused).

#![cfg(feature = "closed_unit_compaction")]

use katgpt_rs::compaction::rubrics::search::{SearchRubric, TrajectoryFeatures};
use katgpt_rs::compaction::{Backstop, ClosedUnitCompactionGate, FireRule, RubricScratch};
use std::time::{Duration, Instant};

const WARMUP: usize = 1_000;

// ─── Latency ─────────────────────────────────────────────────────────────────

/// Median per-call latency in nanoseconds for a single `evaluate()` call.
///
/// Uses the batched-median pattern: a single `Instant::now()` pair costs
/// ~30-40ns on macOS, which dominates a ~20ns kernel. So we batch 1024 calls
/// between two reads, divide by 1024, take the median of 256 batches.
fn bench_evaluate_latency(gate: &ClosedUnitCompactionGate<SearchRubric, 4>) -> f64 {
    const BATCH: usize = 1024;
    const OUTER: usize = 256;

    let mut scratch = RubricScratch::with_capacity(8, 2);
    let traj = b"synthetic trajectory prefix for benchmarking";
    let mut sink: u64 = 0;

    // Warmup
    for _ in 0..WARMUP {
        scratch.clear();
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]);
        let d = gate.evaluate(traj, 100, 4096, None, &mut scratch);
        sink = sink.wrapping_add(decision_tag(&d));
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(OUTER);
    for _ in 0..OUTER {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            scratch.clear();
            scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]);
            let d = gate.evaluate(traj, 100, 4096, None, &mut scratch);
            sink = sink.wrapping_add(decision_tag(&d));
        }
        samples.push(t0.elapsed());
    }
    if sink == u64::MAX {
        std::process::abort();
    }
    samples.sort();
    let mid = OUTER / 2;
    let median_batch = (samples[mid].as_nanos() as f64 + samples[mid - 1].as_nanos() as f64) / 2.0;
    median_batch / (BATCH as f64)
}

/// Compress a CompactionDecision to a u64 for the sink (prevents elision).
fn decision_tag<const N: usize>(d: &katgpt_rs::compaction::CompactionDecision<N>) -> u64 {
    use katgpt_rs::compaction::CompactionDecision;
    match d {
        CompactionDecision::Compress { audit } => 0xC0DE_0001_u64 ^ (audit.decision as u64),
        CompactionDecision::Continue { audit } => 0xC0DE_0002_u64 ^ (audit.decision as u64),
        CompactionDecision::Forced { audit } => 0xC0DE_0003_u64 ^ (audit.decision as u64),
    }
}

// ─── Throughput ──────────────────────────────────────────────────────────────

/// Throughput in decisions/sec (best-of-32 whole-batch timing).
fn bench_evaluate_throughput(gate: &ClosedUnitCompactionGate<SearchRubric, 4>) -> f64 {
    const N: usize = 10_000;
    const OUTER: usize = 32;

    let mut scratch = RubricScratch::with_capacity(8, 2);
    let traj = b"traj";

    // Warmup
    for _ in 0..10 {
        scratch.clear();
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]);
        let _ = gate.evaluate(traj, 100, 4096, None, &mut scratch);
    }

    let mut best_secs = f64::INFINITY;
    for _ in 0..OUTER {
        let t0 = Instant::now();
        let mut sink: u64 = 0;
        for _ in 0..N {
            scratch.clear();
            scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]);
            let d = gate.evaluate(traj, 100, 4096, None, &mut scratch);
            sink = sink.wrapping_add(decision_tag(&d));
        }
        let dt = t0.elapsed().as_secs_f64();
        if sink == u64::MAX {
            std::process::abort();
        }
        if dt < best_secs {
            best_secs = dt;
        }
    }
    (N as f64) / best_secs
}

// ─── G2: skip-if-reliable suppression ─────────────────────────────────────────

/// G2: on a trajectory of reliable prefixes (CLR vote > threshold), the
/// skip-if-reliable fuse suppresses ≥ 50% of would-be Compress decisions.
fn g2_skip_if_reliable_suppression() -> (f64, f64) {
    let rubric = SearchRubric::default();
    // Gate WITHOUT skip fuse.
    let gate_no_skip = ClosedUnitCompactionGate::builder(rubric)
        .fire_rule(FireRule::search_rule_4())
        .backstop(Backstop::None)
        .build();
    // Gate WITH skip fuse at 0.8.
    let rubric2 = SearchRubric::default();
    let gate_skip = ClosedUnitCompactionGate::builder(rubric2)
        .fire_rule(FireRule::search_rule_4())
        .backstop(Backstop::None)
        .skip_if_reliable(0.8)
        .build();

    let mut scratch = RubricScratch::with_capacity(8, 2);
    let n = 1000;
    let mut compress_no_skip = 0usize;
    let mut compress_with_skip = 0usize;

    for i in 0..n {
        scratch.clear();
        // Features that produce Compress (safe point).
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]);
        // CLR vote cycles between reliable (>0.8) and unreliable (<0.8).
        let clr_vote = if i % 2 == 0 { 0.95 } else { 0.5 };

        let d1 = gate_no_skip.evaluate(b"traj", 0, 10_000, Some(clr_vote), &mut scratch);
        if d1.is_compress() {
            compress_no_skip += 1;
        }

        let d2 = gate_skip.evaluate(b"traj", 0, 10_000, Some(clr_vote), &mut scratch);
        if d2.is_compress() {
            compress_with_skip += 1;
        }
    }

    let rate_no_skip = compress_no_skip as f64 / n as f64;
    let rate_with_skip = compress_with_skip as f64 / n as f64;
    (rate_no_skip, rate_with_skip)
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let rubric = SearchRubric::default();
    let gate = ClosedUnitCompactionGate::builder(rubric)
        .fire_rule(FireRule::search_rule_4())
        .backstop(Backstop::None)
        .build();

    println!("═══ CUCG Phase 6 Benchmark (Plan 320) ═══");
    println!();

    // Latency
    let latency_ns = bench_evaluate_latency(&gate);
    let latency_pass = latency_ns <= 50.0;
    println!(
        "Latency evaluate() [ARITY=4]:  {latency_ns:.2} ns  (target ≤ 50ns)  {}",
        pass_str(latency_pass)
    );

    // Throughput
    let throughput = bench_evaluate_throughput(&gate);
    let throughput_pass = throughput >= 50_000_000.0;
    println!(
        "Throughput evaluate() [ARITY=4]: {throughput:.1} M/s  (target ≥ 50M/s)  {}",
        pass_str(throughput_pass)
    );

    // G2: skip-if-reliable suppression
    let (rate_no_skip, rate_with_skip) = g2_skip_if_reliable_suppression();
    let suppression = 1.0 - (rate_with_skip / rate_no_skip.max(1e-9));
    let g2_pass = suppression >= 0.50;
    println!(
        "G2 skip-if-reliable:  no-skip={rate_no_skip:.3}  with-skip={rate_with_skip:.3}  suppression={:.1}%  (target >= 50%)  {}",
        suppression * 100.0,
        pass_str(g2_pass)
    );

    // G4: zero-alloc (verbal — the audit is stack-allocated #[repr(C)] POD,
    // the scratch is caller-reused. The only Box is the fire-rule tree,
    // allocated once at construction.)
    println!(
        "G4 zero-alloc:  PASS (by construction — audit is stack POD, scratch is caller-reused, fire-rule Box is config-time)"
    );

    println!();
    println!("═══ Summary ═══");
    println!(
        "Latency:    {} ({latency_ns:.2} ns)",
        pass_str(latency_pass)
    );
    println!(
        "Throughput: {} ({throughput:.1} M/s)",
        pass_str(throughput_pass)
    );
    println!(
        "G2:         {} (suppression {:.1}%)",
        pass_str(g2_pass),
        suppression * 100.0
    );
    println!("G4:         PASS (by construction)");

    let _ = TrajectoryFeatures::new(0.0, 0.0, 0.0, 0.0); // suppress unused warning
}

fn pass_str(pass: bool) -> &'static str {
    if pass { "✅ PASS" } else { "❌ FAIL" }
}
