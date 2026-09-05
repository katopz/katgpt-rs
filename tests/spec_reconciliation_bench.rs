//! Micro-benchmark: reconciliation latency vs offline duration.
//!
//! Measures reconciliation latency at various offline durations.
//!
//! **The budget is per similarity COMPARISON, not per call.** `reconcile` is
//! O(client x k x steps) by construction — stage 3 sweeps every client point
//! against every manifold point, and this bench passes `steps = n` alongside a
//! client trajectory of `n`, so the work is quadratic in `n`. The pre-723 bar
//! ("p50 < 1 ms" at every size) therefore could not be right at five different
//! sizes, and had never been executed in release at the two larger ones: the
//! debug arm caps `n` at 60 while the release arm runs 600, so nothing had ever
//! measured the configuration the bar was asserted on. Best-of-25 at n=600 is
//! 9.8 ms — 10x the old bar and *not* a regression, just the first look. The
//! per-comparison rate is the scale-invariant quantity a regression actually
//! moves; the absolute wall-clock is a function of the caller's `n`.
//!
//! **Issue 723 Class A / T7 — the asserted quantity is the BEST of N, not the
//! p50.** This is an *absolute* budget with no second arm to ratio against, so
//! the load-invariant statistic is the minimum: contention can only ever add
//! time, so the smallest of N samples is the closest observation of the
//! machine's true cost and the only one a busy box cannot inflate. The
//! enumerating run that filed 723 took p50 over **five** samples on a box at
//! load 8–16 — that is a measurement of the scheduler, not of `reconcile`.
//! p50/p99 stay as telemetry, now via the workspace percentile-of-record
//! (`katgpt_core::stats::nearest_rank`, which also reports **tail support** —
//! `sorted[(0.99 * (n-1)).round()]` on five samples was the maximum printed
//! under a percentile's name).
//!
//! Run: `cargo test --features spec_reconciliation --test spec_reconciliation_bench -- --nocapture`

use std::time::Instant;

use katgpt_core::stats::nearest_rank;
use katgpt_rs::types::Rng;
use katgpt_speculative::spec_reconciliation::{
    ReconciliationConfig, SpecReconciler, TrajectoryPoint,
};

fn bench_config(k: usize) -> ReconciliationConfig {
    ReconciliationConfig {
        k,
        max_speed: 600.0,
        map_bounds: [0.0, 0.0, 4096.0, 4096.0],
        accept_threshold: 0.5,
        quarantine_threshold: 0.2,
        kill_rate_sigma: 5.0,
        noise_sigma: 0.1,
        dt: 1.0 / 60.0,
    }
}

fn h_last() -> TrajectoryPoint {
    TrajectoryPoint::from_fields(2048.0, 2048.0, 10.0, 5.0, 2.0, 0.0, 1.0, 0.0)
}

/// Generate a legitimate client trajectory: small movements from h_last.
fn make_client_trajectory(h: &TrajectoryPoint, n: usize) -> Vec<TrajectoryPoint> {
    (0..n)
        .map(|i| {
            let t = i as f32;
            TrajectoryPoint::from_fields(
                h.pos_x() + t * 0.1,
                h.pos_y() + t * 0.05,
                10.0,
                5.0,
                2.0,
                0.0,
                1.0,
                0.0,
            )
        })
        .collect()
}

// ── Duration sweep ──────────────────────────────────────────────────────────

#[test]
fn bench_reconciliation_latency_vs_duration() {
    // In debug mode, use small point counts to keep test time reasonable.
    // The GOAT proof test (G5) already verifies correctness at small scale.
    // This benchmark focuses on the scaling behavior.
    let durations: &[(&str, usize)] = if cfg!(debug_assertions) {
        // Debug: small point counts to keep test fast
        &[
            ("1s", 10),
            ("10s", 20),
            ("60s", 30),
            ("300s", 50),
            ("600s", 60),
        ]
    } else {
        &[
            ("1s", 60),
            ("10s", 600),
            ("60s", 600),
            ("300s", 600),
            ("600s", 600),
        ]
    };
    // Issue 723 T7: 5 samples cannot support a p50 on a contended box. 25 is
    // enough for the minimum to converge on the uncontended cost while keeping
    // the whole sweep well under a second in release.
    let iters = if cfg!(debug_assertions) { 5 } else { 25 };
    let config = bench_config(16);
    // Per-comparison budget (Issue 723 T7). Measured best-of-25 on the M3 Max,
    // release, 2026-09-05: 3.24 ns/cmp at n=60 (fixed setup amortised over only
    // 57.6K comparisons) down to 1.70 ns/cmp at n=600 (5.76M comparisons). The
    // 8.0 bar is ~2.5x the worst measured point — loose enough that the box
    // cannot decide the verdict, tight enough that losing the fused single-pass
    // sweep in stage 3 (or the SIMD in `score_against_manifold`) reds it.
    let ns_budget: f64 = if cfg!(debug_assertions) { 400.0 } else { 8.0 };

    println!();
    println!(
        "┌────────────┬─────────┬────────────┬────────────┬────────────┬──────────┬───────────┐"
    );
    println!(
        "│ Duration   │ Points  │ BEST (µs)  │ P50 (µs)   │ P99 (µs/s) │ ns/cmp   │ Pass/Fail │"
    );
    println!(
        "├────────────┼─────────┼────────────┼────────────┼────────────┼──────────┼───────────┤"
    );

    for &(label, n) in durations {
        let h = h_last();
        let client = make_client_trajectory(&h, n);

        let mut latencies = Vec::with_capacity(iters);
        for seed in 0..iters {
            let mut reconciler = SpecReconciler::new(config);
            let mut rng = Rng::new(seed as u64);
            let start = Instant::now();
            let accepted = reconciler.reconcile(&h, &client, &[], n, &mut rng);
            let elapsed = start.elapsed().as_nanos() as f64 / 1000.0;
            // Consume the verdict: an unused result is what lets fat LTO delete
            // the callee's work outright (Issue 723 Class A2).
            std::hint::black_box(&accepted);
            latencies.push(elapsed);
        }
        latencies.sort_by(|a, b| a.total_cmp(b));

        // Load-invariant: the minimum. Contention only ever adds.
        let best = latencies[0];
        let (p50, _) = nearest_rank(&latencies, 0.50);
        let (p99, p99_support) = nearest_rank(&latencies, 0.99);

        assert!(
            best > 0.0,
            "instrument failure: reconcile({n} points) measured 0 µs over {iters} \
             samples — work eliminated or below timer resolution, not a pass",
        );

        // The bar is per COMPARISON, not per call — see the module header.
        let comparisons = (n * config.k * n) as f64;
        let ns_per_comparison = best * 1000.0 / comparisons;
        let pass = ns_per_comparison < ns_budget;
        let status = if pass { "PASS" } else { "FAIL" };

        println!(
            "│ {label:<10} │ {n:>7} │ {best:>10.1} │ {p50:>10.1} │ {p99:>7.1}/{p99_support:<2} │ {ns_per_comparison:>8.2} │ {status:>9} │",
        );

        assert!(
            pass,
            "best-of-{iters} reconcile({n} points) = {best:.1} µs over {comparisons:.0} \
             similarity comparisons = {ns_per_comparison:.2} ns/comparison, above the \
             {ns_budget:.1} ns bar — the minimum is load-invariant, so this is the \
             primitive, not the box",
        );
    }

    println!(
        "└────────────┴─────────┴────────────┴────────────┴────────────┴──────────┴───────────┘"
    );
    println!("  (P99 column prints value/tail-support; support 1 means the max)");
}

// ── K-sweep ─────────────────────────────────────────────────────────────────

#[test]
fn bench_reconciliation_k_sweep() {
    let k_values: &[usize] = &[4, 8, 16];
    let iters = 3;
    let n = 20; // Small for debug build performance

    let h = h_last();
    let client = make_client_trajectory(&h, n);

    println!();
    println!("┌────────┬───────────┬────────────┬────────────┬────────────┐");
    println!("│ K      │ Manifolds │ BEST (µs)  │ P50 (µs)   │ P99 (µs)   │");
    println!("├────────┼───────────┼────────────┼────────────┼────────────┤");

    for &k in k_values {
        let config = bench_config(k);
        let mut latencies = Vec::with_capacity(iters);
        for seed in 0..iters {
            let mut reconciler = SpecReconciler::new(config);
            let mut rng = Rng::new(seed as u64);
            let start = Instant::now();
            let _ = reconciler.reconcile(&h, &client, &[], n, &mut rng);
            let elapsed = start.elapsed().as_nanos() as f64 / 1000.0;
            latencies.push(elapsed);
        }
        latencies.sort_by(|a, b| a.total_cmp(b));

        let best = latencies[0];
        let (p50, _) = nearest_rank(&latencies, 0.50);
        let (p99, _) = nearest_rank(&latencies, 0.99);

        println!("│ {k:<6} │ {k:>9} │ {best:>10.1} │ {p50:>10.1} │ {p99:>10.1} │");
    }

    println!("└────────┴───────────┴────────────┴────────────┴────────────┘");
}
