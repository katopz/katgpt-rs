//! SSD Block + Cumprodsum Benchmark (Plan 263 Phases 1, 2, 3, 4).
//!
//! GOAT Gate:
//!   - SIMD cumprodsum vs scalar (Phase 1)
//!   - SSD block vs naive quadratic attention at T=256..4096 (Phase 2)
//!   - SemiseparablePruner branching reduction simulation (Phase 3)
//!   - Adaptive thinking budget on fresh vs stale context (Phase 4)

use std::hint::black_box;

use katgpt_core::ConstraintPruner;
use katgpt_rs::cumprodsum::{context_freshness, cumprodsum_batched, cumprodsum_batched_simd};
use katgpt_rs::pruners::SemiseparablePruner;
use katgpt_rs::speculative::{ThinkingConfig, ThinkingController};
use katgpt_rs::ssd_block::{
    SsdBlockConfig, SsdScratch, auto_block_len, ssd_block_forward, ssd_naive,
};

const WARMUP: usize = 100;
const ITERS: usize = 1_000;

fn bench_simd_cumprodsum() {
    println!("\n=== Phase 1: SIMD Cumprodsum ===\n");

    for &(n_channels, seq_len) in &[(8, 64), (16, 128), (32, 256), (8, 1024)] {
        let total = n_channels * seq_len;
        let a: Vec<f32> = (0..total)
            .map(|i| (i as f32 * 0.001).sin() * 0.4 + 0.5)
            .collect();
        let x: Vec<f32> = (0..total).map(|i| (i as f32 * 0.002).cos()).collect();
        let h_init: Vec<f32> = vec![0.1; n_channels];
        let mut out_scalar = vec![0.0; total];
        let mut out_simd = vec![0.0; total];

        // Warmup
        for _ in 0..WARMUP {
            cumprodsum_batched(
                black_box(&a),
                black_box(&x),
                black_box(&h_init),
                &mut out_scalar,
                n_channels,
                seq_len,
            );
            cumprodsum_batched_simd(
                black_box(&a),
                black_box(&x),
                black_box(&h_init),
                &mut out_simd,
                n_channels,
                seq_len,
            );
        }

        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            cumprodsum_batched(
                black_box(&a),
                black_box(&x),
                black_box(&h_init),
                black_box(&mut out_scalar),
                n_channels,
                seq_len,
            );
        }
        let scalar_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        let start = std::time::Instant::now();
        for _ in 0..ITERS {
            cumprodsum_batched_simd(
                black_box(&a),
                black_box(&x),
                black_box(&h_init),
                black_box(&mut out_simd),
                n_channels,
                seq_len,
            );
        }
        let simd_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Verify correctness
        let max_diff = out_scalar
            .iter()
            .zip(&out_simd)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "SIMD mismatch: max_diff={max_diff}");

        let speedup = scalar_ns / simd_ns;
        println!(
            "  N={n_channels:>3}, T={seq_len:>4}: scalar={scalar_ns:>8.0}ns  simd={simd_ns:>8.0}ns  speedup={speedup:.2}x"
        );
    }
}

fn bench_ssd_block() {
    println!("\n=== Phase 2: SSD Block vs Naive Attention ===\n");

    let state_dim = 4;
    let head_dim = 8;

    for &t in &[64, 128, 256, 512, 1024] {
        let x: Vec<f32> = (0..t * head_dim)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let a: Vec<f32> = (0..t).map(|i| 0.8 + (i as f32 * 0.0001)).collect();
        let b: Vec<f32> = (0..t * state_dim)
            .map(|i| (i as f32 * 0.01).cos() * 0.3)
            .collect();
        let c: Vec<f32> = (0..t * state_dim)
            .map(|i| (i as f32 * 0.02).sin() * 0.3)
            .collect();

        let block_len = auto_block_len(t);
        let config = SsdBlockConfig {
            block_len,
            state_dim,
            head_dim,
        };
        let mut out_block = vec![0.0; t * head_dim];
        let mut out_naive = vec![0.0; t * head_dim];
        let mut scratch = SsdScratch::new(&config, t);

        // Warmup
        for _ in 0..WARMUP.min(50) {
            ssd_block_forward(&x, &a, &b, &c, &config, &mut out_block, &mut scratch);
            ssd_naive(&x, &a, &b, &c, head_dim, state_dim, &mut out_naive);
        }

        // Verify correctness
        let max_diff = out_block
            .iter()
            .zip(&out_naive)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let correct = max_diff < 0.1;

        // Benchmark block decomposition
        let iters = if t <= 256 { ITERS } else { ITERS / 4 };
        let start = std::time::Instant::now();
        for _ in 0..iters {
            ssd_block_forward(
                black_box(&x),
                black_box(&a),
                black_box(&b),
                black_box(&c),
                black_box(&config),
                black_box(&mut out_block),
                black_box(&mut scratch),
            );
        }
        let block_ns = start.elapsed().as_nanos() as f64 / iters as f64;

        // Benchmark naive (fewer iterations for large T)
        let naive_iters = if t <= 128 {
            ITERS
        } else if t <= 512 {
            50
        } else {
            5
        };
        let start = std::time::Instant::now();
        for _ in 0..naive_iters {
            ssd_naive(
                black_box(&x),
                black_box(&a),
                black_box(&b),
                black_box(&c),
                black_box(head_dim),
                black_box(state_dim),
                black_box(&mut out_naive),
            );
        }
        let naive_ns = start.elapsed().as_nanos() as f64 / naive_iters as f64;

        let speedup = naive_ns / block_ns;
        println!(
            "  T={t:>5} (Q={block_len:>3}): naive={naive_ns:>10.0}ns  block={block_ns:>10.0}ns  speedup={speedup:.2}x  match={}",
            if correct { "OK" } else { "MISMATCH" }
        );
    }
}

fn bench_ss_pruner() {
    println!("\n=== Phase 3: SemiseparablePruner ===\n");

    // Simulate DDTree branching: count valid nodes at each depth
    // with and without the SS pruner.
    let vocab_size = 100;

    for &(decay, threshold, label) in &[
        (1.0, 0.0, "no-prune"),
        (0.95, 0.3, "medium"),
        (0.8, 0.05, "aggressive"),
    ] {
        let pruner = SemiseparablePruner::from_uniform(decay, 64, threshold);
        let no_pruner = katgpt_core::NoPruner;

        let mut total_with = 0;
        let mut total_without = 0;

        for depth in 0..32 {
            for tok in 0..vocab_size {
                if pruner.is_valid(depth, tok, &[]) {
                    total_with += 1;
                }
                if no_pruner.is_valid(depth, tok, &[]) {
                    total_without += 1;
                }
            }
        }

        let reduction = (1.0 - (total_with as f64 / total_without as f64)) * 100.0;
        println!(
            "  decay={decay:.2}, thresh={threshold:.2} ({label:>10}): nodes={total_with:>5}/{total_without}  reduction={reduction:.1}%"
        );
    }

    // Benchmark pruner overhead
    let pruner = SemiseparablePruner::from_uniform(0.9, 64, 0.1);
    let start = std::time::Instant::now();
    for _ in 0..100_000 {
        let depth = black_box(10);
        let tok = black_box(5);
        let valid = pruner.is_valid(depth, tok, &[]);
        black_box(valid);
    }
    let ns = start.elapsed().as_nanos() as f64 / 100_000.0;
    println!("  is_valid latency: {ns:.1} ns (target: <100ns)");
}

fn bench_adaptive_thinking() {
    println!("\n=== Phase 4: Adaptive Thinking Budget ===\n");

    let ctrl = ThinkingController::new(ThinkingConfig {
        min_blocks: 0,
        max_blocks: 8,
        ..Default::default()
    });

    for &(decay, label) in &[
        (1.0, "no-decay (stale)"),
        (0.99, "slow-decay"),
        (0.9, "medium-decay"),
        (0.5, "fast-decay (fresh)"),
        (0.1, "very-fresh"),
    ] {
        let factors = vec![decay; 128];
        let freshness = context_freshness(&factors);
        let budget = ctrl.adaptive_budget_default(&factors);
        println!(
            "  decay={decay:.2} ({label:>18}): freshness={freshness:.4}  budget={budget} blocks"
        );
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Plan 263: Cumprodsum + SSD Block Benchmark  ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("Warmup: {WARMUP}, Iters: {ITERS}");

    bench_simd_cumprodsum();
    bench_ssd_block();
    bench_ss_pruner();
    bench_adaptive_thinking();

    println!("\n=== GOAT Gate Summary ===");
    println!("  - SIMD cumprodsum: see Phase 1 results above");
    println!("  - SSD block: compare speedup at T≥256 in Phase 2");
    println!("  - SS pruner: verify branching reduction in Phase 3");
    println!("  - Adaptive budget: verify fresh < stale in Phase 4");
}
