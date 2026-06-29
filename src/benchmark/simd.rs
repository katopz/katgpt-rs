//! SIMD / Perf benchmark — Paper Feature Comparison Matrix (SIMD dimension).
//!
//! Benchmarks four key performance primitives:
//! 1. NEON/SIMD dense vs sparse matmul throughput
//! 2. PlasmaPath bit-plane ternary SIMD matvec (feature-gated)
//! 3. Zero-alloc forward pass throughput
//! 4. Minkowski lattice embedding lookup

use super::{BenchCategory, BenchResult};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights, forward};
use crate::types::{Config, Rng};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn bench_simd_perf() -> Vec<BenchResult> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║          SIMD / Perf Benchmarks (Feature: SIMD)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // 4 sub-benchmarks × ~2 results each.
    let mut results = Vec::with_capacity(16);

    bench_matmul(&mut results);
    bench_plasma_path(&mut results);
    bench_forward_pass(&mut results);
    bench_lattice_lookup(&mut results);

    print_summary_table(&results);

    results
}

// ---------------------------------------------------------------------------
// 1. NEON SIMD matmul — dense vs sparse
// ---------------------------------------------------------------------------

#[cfg(feature = "sparse_mlp")]
fn bench_matmul(results: &mut Vec<BenchResult>) {
    use crate::types::{matmul, sparse_matmul};

    println!("── Dense vs Sparse Matmul ──────────────────────────────────");

    let sizes: [(usize, usize); 2] = [(64, 16), (128, 32)];
    let warmup = 100;
    let iters = 5000;

    for &(rows, cols) in &sizes {
        // Weight matrix [rows × cols]
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();

        // Input at 90% sparsity: only 10% alive
        let mut input = vec![0.0f32; cols];
        let alive_count = (cols as f32 * 0.10) as usize;
        for v in input.iter_mut().take(alive_count) {
            *v = 1.0;
        }

        let mut output_dense = vec![0.0f32; rows];
        let mut output_sparse = vec![0.0f32; rows];
        let mut active_indices = vec![0usize; cols];
        let mut active_values = vec![0.0f32; cols];

        // ── Dense matmul benchmark ──
        for _ in 0..warmup {
            matmul(&mut output_dense, &weight, &input, rows, cols);
        }
        let start = Instant::now();
        for _ in 0..iters {
            matmul(&mut output_dense, &weight, &input, rows, cols);
        }
        let elapsed_dense = start.elapsed();
        let dense_throughput = iters as f64 / elapsed_dense.as_secs_f64();
        let dense_latency_us = elapsed_dense.as_micros() as f64 / iters as f64;

        results.push(BenchResult {
            label: format!("Dense matmul {rows}×{cols}"),
            throughput: dense_throughput,
            time_per_step_us: dense_latency_us,
            avg_acceptance_len: 0.0,
            color: (70, 130, 180), // steel blue
            category: BenchCategory::SimdPerf,
            feature_dim: "SIMD".into(),
        });

        // ── Sparse matmul benchmark (90% sparsity) ──
        for _ in 0..warmup {
            sparse_matmul(
                &mut output_sparse,
                &weight,
                &input,
                rows,
                cols,
                &mut active_indices,
                &mut active_values,
            );
        }
        let start = Instant::now();
        for _ in 0..iters {
            sparse_matmul(
                &mut output_sparse,
                &weight,
                &input,
                rows,
                cols,
                &mut active_indices,
                &mut active_values,
            );
        }
        let elapsed_sparse = start.elapsed();
        let sparse_throughput = iters as f64 / elapsed_sparse.as_secs_f64();
        let sparse_latency_us = elapsed_sparse.as_micros() as f64 / iters as f64;

        results.push(BenchResult {
            label: format!("Sparse matmul {rows}×{cols} (90% sparse)"),
            throughput: sparse_throughput,
            time_per_step_us: sparse_latency_us,
            avg_acceptance_len: 0.0,
            color: (60, 179, 113), // medium sea green
            category: BenchCategory::SimdPerf,
            feature_dim: "SIMD".into(),
        });

        // Correctness check
        for i in 0..rows {
            let diff = (output_dense[i] - output_sparse[i]).abs();
            assert!(
                diff < 1e-2,
                "Mismatch at {i}: dense={} sparse={}",
                output_dense[i],
                output_sparse[i]
            );
        }

        let speedup = elapsed_dense.as_secs_f64() / elapsed_sparse.as_secs_f64();
        println!(
            "  {rows}×{cols}: dense={dense_latency_us:.2}μs ({dense_throughput:.0}/s)  \
             sparse={sparse_latency_us:.2}μs ({sparse_throughput:.0}/s)  speedup={speedup:.2}x"
        );
    }
}

#[cfg(not(feature = "sparse_mlp"))]
fn bench_matmul(_results: &mut Vec<BenchResult>) {
    println!("── Dense vs Sparse Matmul: SKIPPED (feature \"sparse_mlp\" not enabled) ──");
}

// ---------------------------------------------------------------------------
// 2. PlasmaPath bit-plane ternary SIMD matvec
// ---------------------------------------------------------------------------

#[cfg(feature = "plasma_path")]
fn bench_plasma_path(results: &mut Vec<BenchResult>) {
    use katgpt_core::simd::simd_ternary_matvec;
    use crate::types::TernaryWeights;

    println!("── PlasmaPath Ternary SIMD Matvec ─────────────────────────");

    let warmup = 100;
    let iters = 5000;
    let sizes: [(usize, usize); 3] = [(64, 64), (128, 128), (256, 256)];

    for &(rows, cols) in &sizes {
        // Generate random f32 weights and quantize to ternary
        let f32_weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i.wrapping_mul(1103515245).wrapping_add(12345)) as f32) * 0.001)
            .collect();
        let tw = TernaryWeights::quantize_from_f32(&f32_weights, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut y = vec![0.0f32; rows];

        // Warmup
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

        results.push(BenchResult {
            label: format!("Ternary matvec {rows}×{cols}"),
            throughput,
            time_per_step_us: latency_us,
            avg_acceptance_len: 0.0,
            color: (60, 179, 113), // medium sea green (SIMD variant)
            category: BenchCategory::SimdPerf,
            feature_dim: "SIMD".into(),
        });

        println!("  {rows}×{cols}: {latency_us:.2}μs/matvec ({throughput:.0}/s)");
    }
}

#[cfg(not(feature = "plasma_path"))]
fn bench_plasma_path(_results: &mut Vec<BenchResult>) {
    println!("── PlasmaPath Ternary SIMD Matvec: SKIPPED (feature \"plasma_path\" not enabled) ──");
}

// ---------------------------------------------------------------------------
// 3. Zero-alloc forward pass throughput
// ---------------------------------------------------------------------------

fn bench_forward_pass(results: &mut Vec<BenchResult>) {
    println!("── Zero-alloc Forward Pass ────────────────────────────────");

    let config = Config::micro();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);
    let mut ctx = ForwardContext::new(&config);
    let mut cache = MultiLayerKVCache::new(&config);

    let warmup = 50;
    let iters = 500;
    let token = 0usize;
    let pos = 0usize;

    // Warmup
    for _ in 0..warmup {
        let _ = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
        cache.reset();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iters {
        let logits = forward(&mut ctx, &weights, &mut cache, token, pos, &config);
        std::hint::black_box(logits);
        cache.reset();
    }
    let elapsed = start.elapsed();

    let throughput = iters as f64 / elapsed.as_secs_f64();
    let latency_us = elapsed.as_micros() as f64 / iters as f64;

    results.push(BenchResult {
        label: "forward() micro config".into(),
        throughput,
        time_per_step_us: latency_us,
        avg_acceptance_len: 0.0,
        color: (255, 69, 0), // orange red
        category: BenchCategory::SimdPerf,
        feature_dim: "SIMD".into(),
    });

    println!("  forward() micro: {latency_us:.2}μs/pass ({throughput:.0} passes/s)");
}

// ---------------------------------------------------------------------------
// 4. Minkowski lattice embedding lookup
// ---------------------------------------------------------------------------

fn bench_lattice_lookup(results: &mut Vec<BenchResult>) {
    println!("── Minkowski Lattice Embedding Lookup ─────────────────────");

    let warmup = 100;
    let iters = 5000;
    let dims = [8, 16, 32];
    let scale = 10.0f32;

    // Pre-generate query coordinates for each dimension
    for &dim in &dims {
        let coords: Vec<f32> = (0..dim).map(|i| (i as f32 * 1.7).sin()).collect();
        let mut lattice_idx = vec![0usize; dim];

        // Warmup
        for _ in 0..warmup {
            for (j, &c) in coords.iter().enumerate() {
                lattice_idx[j] = (c * scale).floor() as usize;
            }
        }

        let start = Instant::now();
        let mut checksum = 0u64;
        for _ in 0..iters {
            for (j, &c) in coords.iter().enumerate() {
                lattice_idx[j] = (c * scale).floor() as usize;
            }
            checksum = checksum.wrapping_add(lattice_idx.iter().map(|&x| x as u64).sum::<u64>());
        }
        let elapsed = start.elapsed();

        // Prevent optimizer from removing the computation
        std::hint::black_box(checksum);

        let throughput = iters as f64 / elapsed.as_secs_f64();
        let latency_us = elapsed.as_micros() as f64 / iters as f64;

        results.push(BenchResult {
            label: format!("Lattice lookup dim={dim}"),
            throughput,
            time_per_step_us: latency_us,
            avg_acceptance_len: 0.0,
            color: (255, 215, 0), // gold
            category: BenchCategory::SimdPerf,
            feature_dim: "SIMD".into(),
        });

        println!("  dim={dim}: {latency_us:.3}μs/lookup ({throughput:.0} lookups/s)");
    }
}

// ---------------------------------------------------------------------------
// Summary table
// ---------------------------------------------------------------------------

fn print_summary_table(results: &[BenchResult]) {
    println!("\n┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  SIMD / Perf Summary                                              │");
    println!("├──────────────────────────────────┬──────────────┬──────────────────┤");
    println!("│ Benchmark                        │   μs/step    │    steps/sec     │");
    println!("├──────────────────────────────────┼──────────────┼──────────────────┤");

    for r in results {
        println!(
            "│ {:<32} │ {:>10.2}   │ {:>14.0}   │",
            r.label, r.time_per_step_us, r.throughput
        );
    }

    println!("└──────────────────────────────────┴──────────────┴──────────────────┘");
    println!("  Total benchmarks: {}\n", results.len());
}
