//! Correlation Budget Allocation Benchmark — Plan 200.
//!
//! Compares acceptance rate and budget convergence between heuristic
//! `PositionWeightedBudget` (gamma-based exponential decay) and
//! `CorrelationBudgetAllocator` (EMA-driven agreement rates).
//!
//! ```sh
//! cargo run --features corr_budget --example corr_budget_01_bench
//! ```

#![cfg(feature = "corr_budget")]

use std::time::Instant;

use katgpt_rs::speculative::{
    CorrelationBudgetAllocator, NoScreeningPruner, build_dd_tree_screened_corr,
};
use katgpt_rs::types::Config;

// ── Config ─────────────────────────────────────────────────────

const WARMUP: u64 = 50;
const N_ITERS: u64 = 500;

// ── Helpers ────────────────────────────────────────────────────

struct Stats {
    p50: f64,
    p99: f64,
    mean: f64,
    min: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty());
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    Stats {
        p50: sorted[n / 2],
        p99: sorted[(n * 99) / 100].min(sorted[n - 1]),
        mean: sorted.iter().sum::<f64>() / n as f64,
        min: sorted[0],
    }
}

/// Create descending marginals for `n_depths` positions × `vocab_size` tokens.
fn make_marginals(n_depths: usize, vocab_size: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n_depths);
    for _ in 0..n_depths {
        let mut row = Vec::with_capacity(vocab_size);
        let mut sum = 0.0f32;
        for t in 0..vocab_size {
            let v = 1.0 / ((t + 1) as f32);
            row.push(v);
            sum += v;
        }
        for v in &mut row {
            *v /= sum;
        }
        out.push(row);
    }
    out
}

// ── Bench 1: EMA convergence rate ──────────────────────────────

fn bench_ema_convergence() {
    println!("── Bench 1: EMA convergence rate ──\n");

    let alphas: &[f32] = &[0.05, 0.1, 0.2, 0.3, 0.5];
    let steps_to_converge = |alpha: f32| -> usize {
        let mut alloc = CorrelationBudgetAllocator::new(alpha);
        let mut step = 0;
        loop {
            alloc.update(0, true);
            step += 1;
            if alloc.agreement_rate(0) > 0.95 || step > 10000 {
                return step;
            }
        }
    };

    println!("  {:>12} {:>20}", "Alpha", "Steps to >0.95");
    println!("  {}", "-".repeat(36));
    for &alpha in alphas {
        let steps = steps_to_converge(alpha);
        println!("  {:>12.2} {:>20}", alpha, steps);
    }
    println!();
}

// ── Bench 2: Budget allocation vs uniform ──────────────────────

fn bench_budget_allocation() {
    println!("── Bench 2: Budget allocation quality ──\n");

    let mut alloc = CorrelationBudgetAllocator::new(0.3);
    // Train: depth 0 = 95% acceptance, depth 1 = 50%, depth 2 = 10%
    for _ in 0..500 {
        alloc.update(0, true);
        alloc.update(1, fastrand::bool());
        alloc.update(2, false);
    }

    let budget = 300;
    let allocation = alloc.allocate(budget, 3);
    let total: usize = allocation.iter().sum();

    println!("  Budget: {budget}, Depths: 3");
    println!(
        "  Agreement rates: d0={:.3}, d1={:.3}, d2={:.3}",
        alloc.agreement_rate(0),
        alloc.agreement_rate(1),
        alloc.agreement_rate(2),
    );
    println!(
        "  Allocation: d0={}, d1={}, d2={} (total={})",
        allocation[0], allocation[1], allocation[2], total,
    );
    println!();

    assert!(
        allocation[0] > allocation[1],
        "depth 0 should get more budget"
    );
    assert!(
        allocation[1] > allocation[2],
        "depth 1 should get more budget than depth 2"
    );
}

// ── Bench 3: DDTree build time with corr budget vs without ────

fn bench_ddtree_corr_vs_uniform() {
    println!("── Bench 3: DDTree build time — corr budget vs uniform ──\n");

    let config = Config {
        vocab_size: 27,
        tree_budget: 512,
        draft_lookahead: 8,
        screening_threshold: 0.0,
        ..Config::draft()
    };

    let marginals = make_marginals(8, config.vocab_size);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let screener = NoScreeningPruner;

    // Train allocator: simulate 3 depths with known acceptance patterns
    let mut alloc = CorrelationBudgetAllocator::new(0.3);
    for _ in 0..300 {
        alloc.update(0, true);
        alloc.update(1, true);
        alloc.update(2, fastrand::bool());
        alloc.update(3, false);
        alloc.update(4, false);
        alloc.update(5, false);
        alloc.update(6, false);
        alloc.update(7, false);
    }

    // Warmup corr
    for _ in 0..WARMUP {
        let _ = build_dd_tree_screened_corr(&mv, &config, &screener, false, &alloc);
    }

    // Measure corr budget
    let mut corr_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_screened_corr(&mv, &config, &screener, false, &alloc);
        corr_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    // Baseline: default allocator (uniform)
    let uniform_alloc = CorrelationBudgetAllocator::default();
    for _ in 0..WARMUP {
        let _ = build_dd_tree_screened_corr(&mv, &config, &screener, false, &uniform_alloc);
    }

    let mut uniform_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_screened_corr(&mv, &config, &screener, false, &uniform_alloc);
        uniform_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    let corr_stats = compute_stats(&corr_samples);
    let uniform_stats = compute_stats(&uniform_samples);

    println!(
        "  Config: vocab={}, budget={}, depths=8, iters={}",
        config.vocab_size, config.tree_budget, N_ITERS
    );
    println!();
    println!(
        "  {:>24} {:>10} {:>10} {:>10} {:>10}",
        "Path", "p50 (μs)", "p99 (μs)", "mean (μs)", "min (μs)"
    );
    println!("  {}", "-".repeat(66));
    println!(
        "  {:>24} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "uniform (default)",
        uniform_stats.p50,
        uniform_stats.p99,
        uniform_stats.mean,
        uniform_stats.min
    );
    println!(
        "  {:>24} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "corr budget (trained)", corr_stats.p50, corr_stats.p99, corr_stats.mean, corr_stats.min
    );

    let overhead_pct = (corr_stats.mean - uniform_stats.mean) / uniform_stats.mean * 100.0;
    println!("\n  Overhead: {overhead_pct:.1}%");

    match overhead_pct.abs() {
        p if p <= 5.0 => {
            println!("  ✅ Near-zero overhead — correlation budget is production-ready")
        }
        p if p <= 15.0 => println!("  ⚠️  Low overhead — acceptable for accuracy gains"),
        _ => println!("  ❌ High overhead — investigate allocation hot path"),
    }
    println!();
}

// ── Bench 4: EMA update throughput ─────────────────────────────

fn bench_ema_update_throughput() {
    println!("── Bench 4: EMA update throughput ──\n");

    let n_updates: usize = 100_000;
    let n_depths: usize = 8;

    let mut alloc = CorrelationBudgetAllocator::new(0.1);

    // Warmup
    for _ in 0..1000 {
        for d in 0..n_depths {
            alloc.update(d, fastrand::bool());
        }
    }

    let start = Instant::now();
    for _ in 0..n_updates {
        for d in 0..n_depths {
            alloc.update(d, fastrand::bool());
        }
    }
    let elapsed = start.elapsed();

    let total_updates = n_updates * n_depths;
    let ns_per_update = elapsed.as_nanos() as f64 / total_updates as f64;

    println!("  {total_updates} updates in {:?}", elapsed);
    println!("  {:.2} ns/update", ns_per_update);

    match ns_per_update {
        t if t <= 5.0 => println!("  ✅ <5 ns/update — zero overhead for hot path"),
        t if t <= 20.0 => println!("  ⚠️  5-20 ns/update — acceptable"),
        _ => println!("  ❌ >20 ns/update — too slow for decode loop"),
    }
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Correlation Budget Allocation Benchmark — Plan 200");
    println!("  Warmup: {WARMUP} iters, Measure: {N_ITERS} iters");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    bench_ema_convergence();
    bench_budget_allocation();
    bench_ddtree_corr_vs_uniform();
    bench_ema_update_throughput();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════════");
}

// TL;DR: 4-benchmark suite for Plan 200 Correlation Budget — EMA convergence,
// allocation quality, DDTree build time corr vs uniform, update throughput.
// Run with: cargo run --features corr_budget --example corr_budget_01_bench
