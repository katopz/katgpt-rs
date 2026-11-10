//! Plan 571 Phase 1 — Phase Separation Probe GOAT Gate (G2 perf + G4 alloc).
//!
//! Exercises the GOAT gate for the `phase_separation` primitive:
//! - **G2** — perf: `phase_separation_sorted` wall time at N ∈ {10, 100,
//!   1000, 10000}. Target: < 10µs at N=1000 (sub-µs expected). Reports
//!   O(N log N) scaling (N=10000 should be ~13× N=1000, not 100×).
//! - **G4** — alloc-free: 1000 steady-state calls at N=1000 with a
//!   pre-allocated scratch buffer → 0 allocations after warmup
//!   (CountingAllocator).
//!
//! G1 (determinism on integer phases) + G1d (LRC bound confirmation) +
//! G1e (sorted-matches-naive property) live in the lib unit tests
//! (`src/phase_separation.rs::tests`). G3 (no-regression) is verified
//! externally via `cargo test -p katgpt-core --lib` with the feature off
//! (default) AND on (`--features phase_separation`).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core \
//!     --bench bench_571_phase_separation_goat -- --nocapture
//! ```
//!
//! Or directly (working around the macOS dyld/trustd stall):
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/katgpt-plan-571 cargo build --release -p katgpt-core \
//!     --features phase_separation --bench bench_571_phase_separation_goat
//! /tmp/katgpt-plan-571/release/bench_571_phase_separation_goat-* --nocapture
//! ```

use katgpt_core::phase_separation::{from_speeds_and_tick, phase_separation_sorted};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── GateResult ─────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

// ─── G2: perf ───────────────────────────────────────────────────────────────

/// Generate N deterministic phases in [0, 1) via an xorshift PRNG.
fn make_phases(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.push((s >> 40) as f32 / (1u64 << 40) as f32);
    }
    out
}

/// Measure `phase_separation_sorted` wall time at the given N.
/// Returns mean nanoseconds per call over `n_iters` iterations (after a
/// warmup pass).
fn bench_sorted(n: usize, n_warmup: usize, n_iters: usize) -> f64 {
    let phases = make_phases(n, 0xDEAD_BEEF_CAFE_BABE);
    let mut scratch_perm = vec![0_usize; n];
    let mut out = vec![0.0_f32; n];

    // Warmup.
    for _ in 0..n_warmup {
        phase_separation_sorted(&phases, &mut scratch_perm, &mut out);
    }

    // Measure.
    let start = Instant::now();
    for _ in 0..n_iters {
        phase_separation_sorted(&phases, &mut scratch_perm, &mut out);
        black_box(&mut out);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / n_iters as f64
}

fn gate_g2_perf() -> Vec<GateResult> {
    let mut results = Vec::new();

    // N=10, 100, 1000, 10000. The headline gate is N=1000 < 10µs.
    // We also report the N=1000 → N=10000 scaling factor (should be ~13×
    // for O(N log N), not 100× for O(N²)).
    let n10 = bench_sorted(10, 100, 100_000);
    let n100 = bench_sorted(100, 100, 100_000);
    let n1000 = bench_sorted(1000, 50, 10_000);
    let n10000 = bench_sorted(10_000, 10, 1_000);

    println!("  G2 perf breakdown:");
    println!("    N=10    : {n10:>10.1} ns/call");
    println!("    N=100   : {n100:>10.1} ns/call");
    println!("    N=1000  : {n1000:>10.1} ns/call");
    println!("    N=10000 : {n10000:>10.1} ns/call");

    // Headline gate: N=1000 < 10µs (= 10_000 ns).
    let target_ns = 10_000.0_f64;
    results.push(if n1000 < target_ns {
        GateResult::pass(
            "G2.n1000_under_10us",
            format!("{n1000:.1} ns/call < {target_ns:.0} ns target"),
        )
    } else {
        GateResult::fail(
            "G2.n1000_under_10us",
            format!("{n1000:.1} ns/call >= {target_ns:.0} ns target"),
        )
    });

    // Scaling gate: N=10000 / N=1000 should be < 20× (O(N log N) predicts
    // (10000·log10000)/(1000·log1000) = 10 · 13.3 = 13.3×). We use 20× as a
    // generous upper bound — anything > 20× would suggest O(N²) creep.
    let scaling = n10000 / n1000;
    let scaling_target = 20.0_f64;
    results.push(if scaling < scaling_target {
        GateResult::pass(
            "G2.scaling_o_n_log_n",
            format!(
                "N=10000/N=1000 = {scaling:.2}× < {scaling_target:.0}× target (O(N log N) predicts ~13×)"
            ),
        )
    } else {
        GateResult::fail(
            "G2.scaling_o_n_log_n",
            format!(
                "N=10000/N=1000 = {scaling:.2}× >= {scaling_target:.0}× target — suggests O(N²) creep"
            ),
        )
    });

    results
}

// ─── G4: alloc-free steady-state ────────────────────────────────────────────

fn gate_g4_alloc_free() -> GateResult {
    // 1000 steady-state calls at N=1000 with a pre-allocated scratch buffer.
    // Assert 0 allocations after warmup.
    const N: usize = 1000;
    const N_WARMUP: usize = 10;
    const N_CALLS: usize = 1000;

    let phases = make_phases(N, 0xBEEF_1234_5678_DEAD);
    let mut scratch_perm = vec![0_usize; N];
    let mut out = vec![0.0_f32; N];

    // Warmup — sort + scan; the scratch/out vectors are already sized, so no
    // allocation is expected here either, but warmup rules out any first-call
    // lazy init.
    for _ in 0..N_WARMUP {
        phase_separation_sorted(&phases, &mut scratch_perm, &mut out);
    }

    // Measure: N_CALLS calls, sum the alloc delta.
    let ((), alloc_count) = alloc_delta(|| {
        for _ in 0..N_CALLS {
            phase_separation_sorted(&phases, &mut scratch_perm, &mut out);
            black_box(&mut out);
        }
    });

    if alloc_count == 0 {
        GateResult::pass(
            "G4.alloc_free_steady_state",
            format!("0 allocs / {N_CALLS} calls at N={N}"),
        )
    } else {
        GateResult::fail(
            "G4.alloc_free_steady_state",
            format!("{alloc_count} allocs / {N_CALLS} calls at N={N} (expected 0)"),
        )
    }
}

// ─── Bonus: integer-speeds end-to-end smoke (exercises the raw bridge) ──────

/// A small end-to-end check that the raw time-phase bridge + sorted scan
/// produce sensible output on the LRC's canonical 7-runner setup. Not a
/// formal GOAT gate — just a smoke test that the wiring works.
fn smoke_lrc_n7() -> GateResult {
    let speeds: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
    let mut phases = [0.0_f32; 7];
    let mut scratch_perm = [0_usize; 7];
    let mut sep = [0.0_f32; 7];

    // Pick a tick where the runners are spread out (not all at 0).
    from_speeds_and_tick(&speeds, 17, 420, &mut phases);
    phase_separation_sorted(&phases, &mut scratch_perm, &mut sep);

    // Sanity: all separations in [0, 0.5].
    for (i, &s) in sep.iter().enumerate() {
        if !(0.0..=0.5).contains(&s) {
            return GateResult::fail(
                "smoke.lrc_n7_range",
                format!("entity {i} separation {s} out of [0, 0.5]"),
            );
        }
    }

    // Sanity: at least one entity has nonzero separation (not all co-located).
    let max_sep = sep.iter().copied().fold(0.0_f32, f32::max);
    if max_sep > 0.0 {
        GateResult::pass(
            "smoke.lrc_n7_range",
            format!("all separations in [0, 0.5]; max = {max_sep:.4}"),
        )
    } else {
        GateResult::fail(
            "smoke.lrc_n7_range",
            "all separations are 0 — runners all co-located (unexpected at tick 17)".to_string(),
        )
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("  Plan 571 — Phase Separation Probe GOAT Gate (G2 perf + G4 alloc)");
    println!("  (G1 determinism + LRC bound live in lib unit tests)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    let mut all_pass = true;
    let mut all_results = Vec::new();
    all_results.extend(gate_g2_perf());
    all_results.push(gate_g4_alloc_free());
    all_results.push(smoke_lrc_n7());

    for r in &all_results {
        let status = if r.passed { "✓ PASS" } else { "✗ FAIL" };
        println!("  [{status}] {:<32}  {}", r.name, r.detail);
        if !r.passed {
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!("  ── G2/G4 ALL PASS ──");
        println!("  G1 (determinism + LRC bound) verified in lib tests:");
        println!("    cargo test -p katgpt-core --lib phase_separation");
        println!("  G3 (no-regression) verified externally:");
        println!("    cargo test -p katgpt-core --lib                          (feature off)");
        println!("    cargo test -p katgpt-core --lib (feature on)");
    } else {
        println!("  ── SOME GATES FAILED ──");
        std::process::exit(1);
    }
}
