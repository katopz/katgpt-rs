//! Minimal standalone SIMD bisect benchmark.
//!
//! Compile: cargo build --release --example bench_simd_bisect --features sparse_mlp,plasma_path
//! Run:     ./target/release/examples/bench_simd_bisect
//!
//! Tests only the two regressed ops:
//!   - Dense matmul 64×16 (calls simd_dot_f32)
//!   - Ternary matvec 128×128 (calls neon_ternary_matvec)

use std::time::Instant;

fn main() {
    // 30s thermal soak
    println!("🌡️  Thermal soak: sleeping 30s...");
    std::thread::sleep(std::time::Duration::from_secs(30));
    println!("🌡️  Done. Starting benchmarks.\n");

    // ── 1. Dense matmul 64×16 ──
    {
        let rows = 64usize;
        let cols = 16usize;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let mut input = vec![0.0f32; cols];
        let alive_count = (cols as f32 * 0.10) as usize;
        for v in input.iter_mut().take(alive_count) {
            *v = 1.0;
        }
        let mut output = vec![0.0f32; rows];

        let warmup = 1000;
        let iters = 500_000;

        for _ in 0..warmup {
            katgpt_rs::types::matmul(&mut output, &weight, &input, rows, cols);
        }
        let start = Instant::now();
        for _ in 0..iters {
            katgpt_rs::types::matmul(&mut output, &weight, &input, rows, cols);
        }
        let elapsed = start.elapsed();
        let throughput = iters as f64 / elapsed.as_secs_f64();
        let latency_us = elapsed.as_micros() as f64 / iters as f64;
        // Prevent elision
        std::hint::black_box(&output);
        println!("Dense matmul {rows}×{cols}: {latency_us:.4} μs/iter  {throughput:.0} ops/s");
    }

    // 10s cooldown
    println!("\n❄️  Cooldown 10s...");
    std::thread::sleep(std::time::Duration::from_secs(10));

    // ── 2. Ternary matvec 128×128 ──
    #[cfg(feature = "plasma_path")]
    {
        use katgpt_rs::simd::simd_ternary_matvec;
        use katgpt_rs::types::TernaryWeights;

        let rows = 128usize;
        let cols = 128usize;
        let f32_weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) as f32) * 0.001)
            .collect();
        let tw = TernaryWeights::quantize_from_f32(&f32_weights, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut y = vec![0.0f32; rows];

        let warmup = 200;
        let iters = 10_000;

        for _ in 0..warmup {
            simd_ternary_matvec(&tw, &x, &mut y);
        }
        let start = Instant::now();
        for _ in 0..iters {
            simd_ternary_matvec(&tw, &x, &mut y);
        }
        let elapsed = start.elapsed();
        let throughput = iters as f64 / elapsed.as_secs_f64();
        let latency_us = elapsed.as_micros() as f64 / iters as f64;
        std::hint::black_box(&y);
        println!("Ternary matvec {rows}×{cols}: {latency_us:.4} μs/iter  {throughput:.0} ops/s");
    }
    #[cfg(not(feature = "plasma_path"))]
    {
        println!("Ternary matvec: SKIPPED (plasma_path feature not enabled)");
    }

    println!("\n✅ Done.");
}
