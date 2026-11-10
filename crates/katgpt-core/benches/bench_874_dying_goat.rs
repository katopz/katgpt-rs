//! Bench 874 — `kv_eviction::dying` GOAT G2 (Issue 873 primitive C):
//! steady-state latency of the death metric's per-cycle paths.
//!
//! The metric is O(1) arithmetic per row by construction (integer
//! saturating ops + one divide); the bars fence the batch scan at scale —
//! a breach means an allocation crept in or the scan stopped being linear.
//! best-of-chunks absolute budgets (the `bench_873` house style),
//! `black_box` on inputs and results, checksum sinks (the
//! `timed_region_guard` law). Deterministic: arithmetic patterns, no RNG.
//!
//! ```sh
//! cargo bench -p katgpt-core --features usage_rate_eviction --bench bench_874_dying_goat
//! ```

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::kv_eviction::dying::{DeathConfig, DeathRow, death_score};

/// Chunks per measurement (best-of; discards preemption spikes).
const CHUNKS: usize = 240;
/// Rows per chunk. Far above timer resolution per row at O(1)/row.
const ROWS: usize = 4_096;

fn best_ns_per_row<F: FnMut() -> u64>(mut body: F) -> f64 {
    for _ in 0..4 {
        let _ = black_box(body());
    }
    let mut best = f64::INFINITY;
    for _ in 0..CHUNKS {
        let t = Instant::now();
        let sink = body();
        let per_row = t.elapsed().as_secs_f64() / ROWS as f64;
        best = best.min(black_box(per_row));
        let _ = black_box(sink);
    }
    best * 1.0e9
}

fn main() {
    println!("bench_874_dying_goat (Issue 873 C / G2)\n");

    // A mixed population: staggered births, half addressed recently, half
    // stale — every verdict class is represented in every scan.
    let cfg = DeathConfig::new(10_000, 100);
    let mut rows = Vec::with_capacity(ROWS);
    for i in 0..ROWS as u64 {
        let mut r = DeathRow::born_at(i % 50_000);
        if i % 2 == 0 {
            r.addressed(49_000 + (i % 1_000)); // fresh half
        }
        rows.push(r);
    }
    let now = 50_000u64;
    let _ = black_box(now);

    // ── verdict batch (the consumer's per-cycle scan — the real API) ──────
    let mut out = Vec::with_capacity(ROWS);
    let mut checksum = 0u64;
    let mut tick = now;
    let per_verdict_row = {
        let rows_ref = &rows;
        let out_ref = &mut out;
        best_ns_per_row(|| {
            tick += 1; // vary `now` per chunk so the scan cannot constant-fold
            let dead = katgpt_core::kv_eviction::dying::verdicts_into(
                black_box(rows_ref),
                ROWS,
                black_box(tick),
                black_box(&cfg),
                out_ref,
            );
            checksum = checksum.wrapping_add(dead as u64);
            dead as u64
        })
    };
    println!(
        "  verdicts_into (batch)        : {per_verdict_row:6.2} ns/row  ({ROWS} rows/chunk, best of {CHUNKS})"
    );

    // ── death_score alone (the scalar readout) ─────────────────────
    let per_score_row = {
        let rows_ref = &rows;
        best_ns_per_row(|| {
            let mut sink = 0u64;
            for row in rows_ref.iter() {
                sink +=
                    death_score(black_box(row), black_box(tick), black_box(&cfg)).to_bits() as u64;
            }
            checksum = checksum.wrapping_add(sink);
            sink
        })
    };
    println!("  death_score (scalar readout)  : {per_score_row:6.2} ns/row");

    let _ = black_box(checksum);

    // ── the GOAT bars (asserted AFTER the measurements print) ─────────────
    // O(1)/row is the contract; ~10× headroom fences (recorded in
    // .benchmarks/874).
    println!("\n  G2 bars:");
    let mut all_pass = true;
    for (label, measured, bar) in [
        ("verdict", per_verdict_row, 20.0),
        ("death_score", per_score_row, 10.0),
    ] {
        let pass = measured <= bar;
        all_pass &= pass;
        println!(
            "    {label:<12} {measured:6.2} ns/row ≤ {bar:5.1} : {}",
            if pass { "PASS" } else { "FAIL" }
        );
    }
    assert!(
        all_pass,
        "dying G2 regression fence breached — see measurements above"
    );
    println!("\n  G2 PASS (fence, not a claim of optimality — O(1)/row is the contract)");
}
