#![cfg(feature = "dash_attn")]
//! Benchmark — DashAttention α-entmax Overhead (Plan 106, T25)
//!
//! Measures entmax_1p5() threshold-finding time for varying input sizes.
//! Target: < 50µs for 256 chunks (trivial vs attention cost).
//!
//! Run: `cargo test --features dash_attn --test bench_106_dash_attn_entmax -- --nocapture`

use std::hint::black_box;
use std::time::Instant;

use microgpt_rs::dash_attn::entmax_1p5;

const WARMUP: usize = 100;
const ITERS: usize = 1000;

/// Generate scores with a mix of peaked and spread distributions.
fn make_scores(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            // Deterministic pseudo-random-ish pattern
            let x = ((i.wrapping_mul(2654435761)).wrapping_add(seed)) as f32;
            let normalized = (x / (n as f32)).sin();
            // Add a dominant peak so entmax has meaningful sparsity
            if i == seed % n {
                10.0 + normalized
            } else {
                normalized
            }
        })
        .collect()
}

/// Run entmax benchmark for a given chunk count, return median µs.
fn bench_entmax(n_chunks: usize) -> (f64, f64) {
    let scores = make_scores(n_chunks, 42);

    // Warmup
    for _ in 0..WARMUP {
        black_box(entmax_1p5(black_box(&scores)));
    }

    // Measure
    let mut times_us: Vec<f64> = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let scores_i = make_scores(n_chunks, i);
        let start = Instant::now();
        let (probs, tau) = entmax_1p5(&scores_i);
        let elapsed = start.elapsed();
        black_box((&probs, tau));
        times_us.push(elapsed.as_nanos() as f64 / 1000.0);
    }

    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = times_us[times_us.len() / 2];
    let mean = times_us.iter().sum::<f64>() / times_us.len() as f64;

    (median, mean)
}

#[test]
fn bench_entmax_64_chunks() {
    let (median, mean) = bench_entmax(64);
    println!("[entmax] n_chunks=64: median={median:.1}µs, mean={mean:.1}µs");
}

#[test]
fn bench_entmax_128_chunks() {
    let (median, mean) = bench_entmax(128);
    println!("[entmax] n_chunks=128: median={median:.1}µs, mean={mean:.1}µs");
}

#[test]
fn bench_entmax_256_chunks_under_50us() {
    let (median, mean) = bench_entmax(256);
    println!("[entmax] n_chunks=256: median={median:.1}µs, mean={mean:.1}µs");

    // Debug builds are ~56µs (no optimization), release is ~2µs.
    // The 50µs target is for release; use relaxed bound for debug.
    let threshold_us = if cfg!(debug_assertions) { 100.0 } else { 50.0 };
    assert!(
        median < threshold_us,
        "entmax_1p5() for 256 chunks must be < {threshold_us}µs ({build}), got median={median:.1}µs",
        build = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
}

#[test]
fn bench_entmax_512_chunks() {
    let (median, mean) = bench_entmax(512);
    println!("[entmax] n_chunks=512: median={median:.1}µs, mean={mean:.1}µs");
}

/// Verify entmax produces valid sparse distributions across all sizes.
#[test]
fn bench_entmax_correctness_across_sizes() {
    for &n_chunks in &[64, 128, 256, 512] {
        let scores = make_scores(n_chunks, 0);
        let (probs, _tau) = entmax_1p5(&scores);

        // Sum to 1.0
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "n_chunks={n_chunks}: probs sum={sum}, expected 1.0"
        );

        // Non-negative
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                p >= 0.0,
                "n_chunks={n_chunks}: negative prob at index {i}: {p}"
            );
        }

        // Sparse: at least one zero (for peaked distributions)
        let zeros = probs.iter().filter(|&&p| p < 1e-8).count();
        println!("[entmax] n_chunks={n_chunks}: {zeros} zero entries (sparse)");
        assert!(
            zeros > 0,
            "n_chunks={n_chunks}: entmax should produce sparse output"
        );
    }
}
