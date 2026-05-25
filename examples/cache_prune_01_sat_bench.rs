//! CachePrune GOAT Proof — SAT build + query vs naive scan benchmark (Plan 140).
//!
//! Validates G1 (SAT region_sum correctness) and G2 (SAT throughput).
//!
//! Run: `cargo run --example cache_prune_01_sat_bench --features cache_prune`

use katgpt_rs::cache_prune::SummedAreaTable;

fn main() {
    println!("=== CachePrune SAT Benchmark (Plan 140 GOAT) ===\n");

    // G1: Correctness — verify SAT region_sum matches naive scan on random matrices.
    correctness_check();

    // G2: Throughput — benchmark SAT build + query vs naive O(n²) scan.
    throughput_benchmark(64);
    throughput_benchmark(256);
    throughput_benchmark(512);
}

/// Naive region sum by iterating over all elements.
fn naive_region_sum(matrix: &[Vec<f32>], x1: usize, x2: usize, y1: usize, y2: usize) -> f32 {
    let mut sum = 0.0;
    for i in x1..=x2 {
        for j in y1..=y2 {
            sum += matrix[i][j];
        }
    }
    sum
}

/// Generate a random-ish attention matrix using simple PRNG.
fn make_attention_matrix(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    let mut next_f32 = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f32 / (1u64 << 31) as f32
    };

    (0..n)
        .map(|_| (0..n).map(|_| next_f32()).collect())
        .collect()
}

fn correctness_check() {
    println!("── G1: SAT Correctness ──");

    for &n in &[8, 16, 32, 64] {
        let matrix = make_attention_matrix(n, 42);
        let mut sat_data = matrix.clone();
        let sat = SummedAreaTable::build(&mut sat_data);

        let mut max_err = 0.0f32;
        let mut tests = 0;

        // Test all possible rectangular regions (sample for large n).
        let step = if n > 16 { 3 } else { 1 };
        for x1 in (0..n).step_by(step) {
            for x2 in (x1..n).step_by(step) {
                for y1 in (0..n).step_by(step) {
                    for y2 in (y1..n).step_by(step) {
                        let sat_sum = sat.region_sum(x1, x2, y1, y2);
                        let naive = naive_region_sum(&matrix, x1, x2, y1, y2);
                        let err = (sat_sum - naive).abs();
                        max_err = max_err.max(err);
                        tests += 1;
                    }
                }
            }
        }

        let status = if max_err < 1e-4 { "✓" } else { "✗" };
        println!("  n={n:3}: {tests:8} queries, max_err={max_err:.2e} {status}");
    }

    println!();
}

fn throughput_benchmark(n: usize) {
    println!("── G2: SAT Throughput (n={n}) ──");

    let matrix = make_attention_matrix(n, 123);

    // SAT build
    let mut sat_data = matrix.clone();
    let build_start = std::time::Instant::now();
    let sat = SummedAreaTable::build(&mut sat_data);
    let build_us = build_start.elapsed().as_micros();

    // SAT queries
    let num_queries = 1_000_000.min(n * n);
    let query_start = std::time::Instant::now();
    for _ in 0..num_queries {
        let x1 = 0;
        let x2 = n / 2;
        let y1 = 0;
        let y2 = n / 2;
        std::hint::black_box(sat.region_sum(x1, x2, y1, y2));
    }
    let query_us = query_start.elapsed().as_micros();
    let qps = num_queries as f64 / (query_us as f64 / 1_000_000.0);

    // Naive scan comparison
    let naive_queries = 10_000.min(n * n);
    let naive_start = std::time::Instant::now();
    for _ in 0..naive_queries {
        std::hint::black_box(naive_region_sum(&matrix, 0, n / 2, 0, n / 2));
    }
    let naive_us = naive_start.elapsed().as_micros();
    let naive_qps = naive_queries as f64 / (naive_us as f64 / 1_000_000.0);

    let speedup = naive_qps / qps.max(1.0);

    println!("  Build:  {build_us:>8} µs");
    println!("  SAT:    {qps:>12.0} queries/sec ({num_queries} queries in {query_us} µs)");
    println!("  Naive:  {naive_qps:>12.0} queries/sec ({naive_queries} queries in {naive_us} µs)");
    println!("  Ratio:  naive is {speedup:.1}× slower per query");
    println!();
}
