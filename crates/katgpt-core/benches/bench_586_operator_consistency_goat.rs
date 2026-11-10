//! Issue 586 — Rule-Application Consistency Metric GOAT gate.
//!
//! Measures:
//! - **G1** regime separation: three synthetic operator-application fixtures
//!   (consistent / noisy-flaky / complexity-clustered) must classify into
//!   their respective [`ConsistencyRegime`] variants, and the promotion gate
//!   must return Promote / Hold / SeekExemplar respectively.
//! - **G2** latency: `rule_consistency` at N ∈ {16, 64, 256} applications.
//!   Target: sub-µs at N ≤ 64 (issue acceptance).
//! - **G4** alloc-free steady state: 0 allocations across 100 calls
//!   (CountingAllocator, bench_013/bench_022 convention).
//!
//! Convention: `std::time::Instant` + `harness = false` (mirrors
//! `bench_324_bisimulation_goat.rs`, no Criterion dev-dep).
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_586_operator_consistency_goat \
//!     --features operator_consistency
//! ```
//!
//! Documented in `.benchmarks/633_operator_consistency_goat.md`.

#![cfg(feature = "bisimulation_operator_inference")]

use katgpt_core::bisimulation::consistency::{
    ApplicationOutcome, ConsistencyGateConfig, ConsistencyRegime, PromotionVerdict,
    promotion_verdict, rule_consistency,
};
use std::time::{Duration, Instant};

// ─── Config ────────────────────────────────────────────────────────────────

/// Application counts to sweep for the G2 latency gate.
const SIZES: &[usize] = &[16, 64, 256];

/// Warmup iterations.
const WARMUP: usize = 10;

/// Number of timed runs to take the median over.
const TIMED_RUNS: usize = 50;

/// Calls for the G4 alloc count.
const G4_CALLS: usize = 100;

// ─── G1 fixtures ───────────────────────────────────────────────────────────

/// (a) CONSISTENT: ≤2% tolerated noise. Every application correct except 1
/// in 100.
fn fixture_consistent(n: usize) -> Vec<ApplicationOutcome> {
    let mut apps = Vec::with_capacity(n);
    let mut task = 0u32;
    for i in 0..n {
        // One tolerated failure at the midpoint.
        let correct = i != n / 2;
        let level = (i % 6) as u8;
        if i % 4 == 0 && i > 0 {
            task += 1;
        }
        apps.push(ApplicationOutcome::new(correct, correct, level, task));
    }
    apps
}

/// (b) Noisy-flaky: ~14% failure at EVERY level, decorrelated (fail bit
/// cycles mod 7, level cycles mod 6 — coprime). Accuracy stays above the
/// gate floor but application is inconsistent → Hold.
fn fixture_flaky(n: usize) -> Vec<ApplicationOutcome> {
    let mut apps = Vec::with_capacity(n);
    let mut task = 0u32;
    for i in 0..n {
        let correct = i % 7 != 3;
        let level = ((i * 5) % 6) as u8;
        if i % 3 == 0 && i > 0 {
            task += 1;
        }
        apps.push(ApplicationOutcome::new(correct, correct, level, task));
    }
    apps
}

/// (c) Complexity-clustered: levels 0..=3 clean, 4..=5 broken (paper's
/// nesting-depth signature). Requires n ≥ 16 so each level has samples and
/// the suffix has ≥ MIN_SUFFIX_FAILURES failures.
fn fixture_clustered(n: usize) -> Vec<ApplicationOutcome> {
    let mut apps = Vec::with_capacity(n);
    for i in 0..n {
        let level = (i % 6) as u8;
        let correct = level <= 3;
        apps.push(ApplicationOutcome::new(correct, correct, level, i as u32));
    }
    apps
}

fn run_g1() -> bool {
    let cfg = ConsistencyGateConfig::default();

    let consistent = rule_consistency(&fixture_consistent(100));
    let v_a = promotion_verdict(&consistent, &cfg);
    let ok_a =
        consistent.regime == ConsistencyRegime::Consistent && v_a == PromotionVerdict::Promote;

    let flaky = rule_consistency(&fixture_flaky(96));
    let v_b = promotion_verdict(&flaky, &cfg);
    let ok_b = flaky.regime == ConsistencyRegime::NoisyFlaky && v_b == PromotionVerdict::Hold;

    let clustered = rule_consistency(&fixture_clustered(48));
    let v_c = promotion_verdict(&clustered, &cfg);
    let ok_c = clustered.regime == ConsistencyRegime::ComplexityClustered { level: 4 }
        && v_c == PromotionVerdict::SeekExemplar { level: 4 };

    println!("G1 regime separation:");
    println!(
        "  (a) consistent:    regime={:?} verdict={v_a:?} acc={:.3}",
        consistent.regime, consistent.application_accuracy
    );
    println!(
        "  (b) noisy-flaky:   regime={:?} verdict={v_b:?} gap={:.3}",
        flaky.regime, flaky.gap
    );
    println!(
        "  (c) clustered:     regime={:?} verdict={v_c:?} acc={:.3}",
        clustered.regime, clustered.application_accuracy
    );

    if ok_a && ok_b && ok_c {
        println!("G1 PASS — three regimes separate + gate verdicts correct");
        true
    } else {
        println!("G1 FAIL — ok_a={ok_a} ok_b={ok_b} ok_c={ok_c}");
        false
    }
}

// ─── G2 latency ────────────────────────────────────────────────────────────

fn bench_rule_consistency(apps: &[ApplicationOutcome]) -> Duration {
    for _ in 0..WARMUP {
        let _ = rule_consistency(apps);
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let r = rule_consistency(apps);
        samples.push(t0.elapsed());
        // Prevent the compiler from eliding the call.
        if r.n_applications == u32::MAX {
            std::process::abort();
        }
    }
    samples.sort();
    samples[TIMED_RUNS / 2]
}

fn run_g2() -> bool {
    println!("G2 latency (median of {TIMED_RUNS}):");
    let mut pass = true;
    for &n in SIZES {
        let apps = fixture_clustered(n);
        let d = bench_rule_consistency(&apps);
        // Target: sub-µs at N ≤ 64 (issue acceptance). Report all sizes;
        // gate only N ≤ 64.
        let gate = n <= 64 && d < Duration::from_micros(1);
        if n <= 64 {
            pass &= gate;
        }
        println!(
            "  N={n:>4}: {d:?}{}",
            if n <= 64 { "" } else { " (report-only)" }
        );
    }
    if pass {
        println!("G2 PASS — sub-µs at N ≤ 64");
    } else {
        println!("G2 FAIL — exceeded 1µs at N ≤ 64");
    }
    pass
}

// ─── G4 alloc-free ─────────────────────────────────────────────────────────

fn run_g4() -> bool {
    use std::alloc::{GlobalAlloc, Layout};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingAllocator;
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            unsafe { std::alloc::System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static A: CountingAllocator = CountingAllocator;

    let apps = fixture_clustered(64);
    // Warmup (fixture Vecs are already built; rule_consistency itself must
    // not allocate).
    for _ in 0..10 {
        let _ = rule_consistency(&apps);
    }
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0u32;
    for _ in 0..G4_CALLS {
        let r = rule_consistency(&apps);
        sink += r.n_applications;
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    println!("G4 alloc-free: {allocs} allocations across {G4_CALLS} calls (sink={sink})");
    if allocs == 0 {
        println!("G4 PASS — zero steady-state allocation");
        true
    } else {
        println!("G4 FAIL — steady-state allocation detected");
        false
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let g1 = run_g1();
    let g2 = run_g2();
    let g4 = run_g4();
    if g1 && g2 && g4 {
        println!("\nIssue 586 GOAT gate: ALL PASS (G1 + G2 + G4)");
    } else {
        println!("\nIssue 586 GOAT gate: FAIL");
        std::process::exit(1);
    }
}
