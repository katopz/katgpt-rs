//! Plan 294 Phase 5 T5.1 — GOAT Gate G4: hot-path cost.
//!
//! Measures `BranchingDetector::observe_and_detect_into` at K=8,
//! action_dim=32 over 10K iterations. Target: **≤ 50µs per call** (the
//! plasma budget — what makes the ICT selector viable for 20Hz ×
//! thousands-of-NPCs crowd-scale cognitive allocation).
//!
//! ## Why not criterion
//!
//! `criterion` is NOT a `katgpt-rs` dev-dependency (root Cargo.toml
//! `[dev-dependencies]` lists only `ratatui`, `crossterm`, `tempfile`).
//! This file follows the repo bench convention set by
//! `benches/fpcg_probe_forecast_bench.rs`: `std::time::Instant` +
//! `std::hint::black_box` + `harness = false` + `fn main()`.
//!
//! ## Run
//!
//! ```text
//! cargo bench --bench bench_294_ict_perf --features ict_branching
//! # or:
//! cargo run --release --bench bench_294_ict_perf --features ict_branching
//! ```
//!
//! Release build is required — debug builds do not engage SIMD autovectorization.

#![cfg(feature = "ict_branching")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::ict::{BranchingDetector, BranchingReport};

const K_TRAJECTORIES: usize = 8;
const ACTION_DIM: usize = 32;
const WARMUP_ITERS: usize = 1_000;
const BENCH_ITERS: usize = 10_000;

fn main() {
    println!("=== Plan 294 Phase 5 G4 — BranchingDetector hot-path cost ===");
    println!(
        "K={}, action_dim={}, warmup={}, timed={}",
        K_TRAJECTORIES, ACTION_DIM, WARMUP_ITERS, BENCH_ITERS
    );
    println!("Target: ≤ 50µs per observe_and_detect_into call.\n");

    // ── Build deterministic K=8 × action_dim=32 trajectory set. ──
    // Concentrated distributions (one dominant action + tail) — realistic
    // for a policy that has committed to a primary action with exploration
    // noise on the rest.
    let mut trajectories: Vec<Vec<f32>> = Vec::with_capacity(K_TRAJECTORIES);
    let mut seed = 0x1234_5678u64;
    for _ in 0..K_TRAJECTORIES {
        let mut p = vec![0.0_f32; ACTION_DIM];
        // LCG for reproducibility.
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let dom = (seed >> 32) as usize % ACTION_DIM;
        let dom_mass = 0.5 + 0.2 * ((seed & 0xFFFF) as f32 / 65535.0);
        p[dom] = dom_mass;
        let rest = (1.0 - dom_mass) / (ACTION_DIM - 1) as f32;
        for j in 0..ACTION_DIM {
            if j != dom {
                p[j] = rest;
            }
        }
        trajectories.push(p);
    }
    let traj_refs_owned: Vec<Vec<f32>> = trajectories.clone();
    let traj_refs: Vec<&[f32]> = traj_refs_owned.iter().map(|v| v.as_slice()).collect();

    // ── Build the detector + reusable report (zero-alloc hot path). ──
    let mut det = BranchingDetector::new(K_TRAJECTORIES, ACTION_DIM, 0.10, 0.05);
    let mut report = BranchingReport {
        mask: vec![false; K_TRAJECTORIES],
        beta_per_step: vec![0.0; K_TRAJECTORIES],
        uniqueness_scores: vec![0.0; K_TRAJECTORIES],
    };

    // ── Warmup. ──
    let mut sink = 0.0_f32;
    for _ in 0..WARMUP_ITERS {
        det.observe_and_detect_into(&traj_refs, &mut report);
        sink = sink + report.uniqueness_scores[0];
    }
    if black_box(sink.is_nan()) {
        eprintln!("warmup sink nan (impossible for finite inputs)");
    }

    // ── Timed loop. ──
    let mut samples_us: Vec<u64> = Vec::with_capacity(BENCH_ITERS);
    let mut sink2 = 0.0_f32;
    for _ in 0..BENCH_ITERS {
        let t0 = Instant::now();
        det.observe_and_detect_into(black_box(&traj_refs), black_box(&mut report));
        let dt = t0.elapsed();
        sink2 = sink2 + report.uniqueness_scores[0];
        samples_us.push(dt.as_micros() as u64);
    }
    if black_box(sink2.is_nan()) {
        eprintln!("timed sink nan (impossible for finite inputs)");
    }

    // ── Stats: mean, p50, p99. ──
    samples_us.sort();
    let n = samples_us.len();
    let sum: u64 = samples_us.iter().sum();
    let mean = sum as f64 / n as f64;
    let p50 = samples_us[n / 2] as f64;
    let p99 = samples_us[(99 * n) / 100] as f64;
    let max = samples_us[n - 1] as f64;

    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12}",
        "mean µs", "p50 µs", "p99 µs", "max µs", "verdict"
    );
    let verdict = if mean <= 50.0 { "PASS" } else { "FAIL" };
    println!(
        "{:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>12}",
        mean, p50, p99, max, verdict
    );
    println!(
        "\nG4 {}: mean {:.2}µs, p50 {:.2}µs, p99 {:.2}µs (target ≤ 50µs).",
        verdict, mean, p50, p99
    );

    // Exit code: 0 on PASS, non-zero on FAIL (so CI can pick it up).
    std::process::exit(if mean <= 50.0 { 0 } else { 1 });
}
