//! Micro-benchmark: reconciliation latency vs offline duration.
//!
//! Measures P50/P99 latency for reconciliation at various offline durations.
//! In release builds, P50 should be <1ms. Debug builds are expected to be
//! much slower, so thresholds are relaxed accordingly.
//!
//! Run: `cargo test --features spec_reconciliation --test spec_reconciliation_bench -- --nocapture`



use std::time::Instant;

use katgpt_rs::spec_reconciliation::{ReconciliationConfig, SpecReconciler, TrajectoryPoint};
use katgpt_rs::types::Rng;

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

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
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
    let iters = 5;
    let config = bench_config(16);

    println!();
    println!("┌────────────┬─────────┬────────────┬────────────┬───────────┐");
    println!("│ Duration   │ Points  │ P50 (µs)   │ P99 (µs)   │ Pass/Fail │");
    println!("├────────────┼─────────┼────────────┼────────────┼───────────┤");

    for &(label, n) in durations {
        let h = h_last();
        let client = make_client_trajectory(&h, n);

        let mut latencies = Vec::with_capacity(iters);
        for seed in 0..iters {
            let mut reconciler = SpecReconciler::new(config);
            let mut rng = Rng::new(seed as u64);
            let start = Instant::now();
            let _ = reconciler.reconcile(&h, &client, &[], n, &mut rng);
            let elapsed = start.elapsed().as_nanos() as f64 / 1000.0;
            latencies.push(elapsed);
        }
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&latencies, 50.0);
        let p99 = percentile(&latencies, 99.0);

        // In release mode, assert <1ms. In debug, just report (no assertion).
        let pass = if cfg!(debug_assertions) {
            p50 < 500_000.0 // 500ms in debug
        } else {
            p50 < 1_000.0 // 1ms in release
        };
        let status = if pass { "PASS" } else { "FAIL" };

        println!(
            "│ {:<10} │ {:>7} │ {:>10.1} │ {:>10.1} │ {:>9} │",
            label, n, p50, p99, status,
        );

        assert!(
            pass,
            "P50 latency {:.1} µs exceeds threshold for duration {} ({} points)",
            p50, label, n,
        );
    }

    println!("└────────────┴─────────┴────────────┴────────────┴───────────┘");
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
    println!("┌────────┬───────────┬────────────┬────────────┐");
    println!("│ K      │ Manifolds │ P50 (µs)   │ P99 (µs)   │");
    println!("├────────┼───────────┼────────────┼────────────┼");

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
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50 = percentile(&latencies, 50.0);
        let p99 = percentile(&latencies, 99.0);

        println!("│ {:<6} │ {:>9} │ {:>10.1} │ {:>10.1} │", k, k, p50, p99);
    }

    println!("└────────┴───────────┴────────────┴────────────┘");
}
