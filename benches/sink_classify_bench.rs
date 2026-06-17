//! Sink-Aware Attention classifier benchmark (Plan 287 Phase 2, T2.4).
//!
//! Sweeps the stable-rank kernel and full `classify_sink_at` over
//! `n ∈ {32, 128, 512}` × `d_h ∈ {64, 128}`. Plan target: < 1µs for
//! `n=32, d_h=64` (plasma tier).
//!
//! Uses `std::time::Instant` (NOT criterion — matches `manifold_power_iter_
//! router_bench.rs`, `triggered_injection_bench.rs`). `harness = false`.
//!
//! Run:
//! ```bash
//! cargo run --release --bench sink_classify_bench --features sink_aware_attn
//! ```

#![cfg(feature = "sink_aware_attn")]

use katgpt_rs::data_probe::sink_classify::{
    SinkClassifierConfig, StableRankScratch, classify_sink_at, stable_rank_update_into,
};
use std::time::{Duration, Instant};

/// Deterministic xorshift64* PRNG.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        // xorshift64
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Map to [-1, 1].
        ((self.0 & 0xFFFF) as f32 / 0x8000 as f32) - 1.0
    }
}

/// Build an `(n, d)` matrix of pseudo-random values.
fn rand_matrix(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.next_f32()).collect())
        .collect()
}

/// Build a rank-1 `(n, d)` matrix (all rows are the same vector).
fn rank1_matrix(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    let v: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
    (0..n).map(|_| v.clone()).collect()
}

/// Build a length-n attention column with all entries = `strength`.
fn uniform_attn_column(n: usize, strength: f32) -> Vec<f32> {
    vec![strength; n]
}

/// Best-of-N wall-clock microseconds for a closure.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::from_secs(60);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

fn main() {
    println!("=== Sink-Aware Attention Classifier Benchmark (Plan 287 T2.4) ===\n");

    let n_values: &[usize] = &[32, 128, 512];
    let d_values: &[usize] = &[64, 128];

    // ── 1. stable_rank_update_into on random matrices ──────────
    println!("## stable_rank_update_into on random (n, d) matrices\n");
    println!(
        "{:>5} {:>5} {:>14} {:>14} {:>10}",
        "n", "d_h", "sr_random_us", "sr_rank1_us", "sr_value"
    );
    println!("{}", "-".repeat(60));

    let mut scratch = StableRankScratch::new(128);
    for &n in n_values {
        for &d in d_values {
            let o_rand = rand_matrix(n, d, 42 + (n * d) as u64);
            let o_rank1 = rank1_matrix(n, d, 100 + (n * d) as u64);
            scratch.ensure_capacity(d);

            // Random matrix — typically full rank, slow convergence.
            let us_rand = bench_us(3, 30, || {
                let sr = stable_rank_update_into(&o_rand, &mut scratch, 5);
                std::hint::black_box(sr);
            });

            // Rank-1 matrix — fast early-exit path.
            let us_rank1 = bench_us(3, 30, || {
                let sr = stable_rank_update_into(&o_rank1, &mut scratch, 5);
                std::hint::black_box(sr);
            });

            let sr_value = stable_rank_update_into(&o_rand, &mut scratch, 5);

            println!(
                "{:>5} {:>5} {:>14.3} {:>14.3} {:>10.3}",
                n, d, us_rand, us_rank1, sr_value
            );
        }
    }
    println!();

    // ── 2. classify_sink_at full path (stable-rank included) ────
    println!("## classify_sink_at full path (with update_O)\n");
    println!(
        "{:>5} {:>5} {:>16} {:>16} {:>10}",
        "n", "d_h", "classify_rand_us", "classify_rank1_us", "kind"
    );
    println!("{}", "-".repeat(70));

    let cfg = SinkClassifierConfig::default();
    for &n in n_values {
        for &d in d_values {
            let values = rand_matrix(n, d, 7 + (n * d) as u64);
            let o_rand = rand_matrix(n, d, 42 + (n * d) as u64);
            let o_rank1 = rank1_matrix(n, d, 100 + (n * d) as u64);
            let attn_column = uniform_attn_column(n, 0.7); // above τ_sink=0.5
            scratch.ensure_capacity(d);

            let us_rand = bench_us(3, 30, || {
                let d = classify_sink_at(
                    0,
                    &attn_column,
                    &values,
                    Some(&o_rand),
                    &cfg,
                    &mut scratch,
                );
                std::hint::black_box(d);
            });

            let us_rank1 = bench_us(3, 30, || {
                let d = classify_sink_at(
                    0,
                    &attn_column,
                    &values,
                    Some(&o_rank1),
                    &cfg,
                    &mut scratch,
                );
                std::hint::black_box(d);
            });

            // Capture kind for the rank-1 case for sanity.
            let diag = classify_sink_at(
                0,
                &attn_column,
                &values,
                Some(&o_rank1),
                &cfg,
                &mut scratch,
            );

            println!(
                "{:>5} {:>5} {:>16.3} {:>16.3} {:>10?}",
                n, d, us_rand, us_rank1, diag.kind
            );
        }
    }
    println!();

    // ── 3. G2.4 target check ────────────────────────────────────
    println!("## G2.4 Target Check\n");
    let n = 32usize;
    let d = 64usize;
    let values = rand_matrix(n, d, 7);
    let o_rand = rand_matrix(n, d, 42);
    let attn_column = uniform_attn_column(n, 0.7);
    scratch.ensure_capacity(d);
    let us = bench_us(5, 100, || {
        let d = classify_sink_at(
            0,
            &attn_column,
            &values,
            Some(&o_rand),
            &cfg,
            &mut scratch,
        );
        std::hint::black_box(d);
    });
    println!("  n=32, d_h=64 classify_sink_at: {:.3} µs", us);
    println!(
        "  G2.4 target <1µs: {}",
        if us < 1.0 { "PASS" } else { "FAIL (documented)" }
    );
}
