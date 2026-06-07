//! Rosetta Pruners Benchmark — Plan 201.
//!
//! Measures DDTree build time and node reduction with vs without RosettaPruner.
//! Demonstrates O(1) fast-path concept map lookup vs O(N) full pruner evaluation.
//!
//! ```sh
//! cargo run --features rosetta_pruner --example rosetta_01_bench
//! ```

#![cfg(feature = "rosetta_pruner")]

use std::sync::Arc;
use std::time::Instant;

use katgpt_rs::pruners::RosettaPruner;
use katgpt_rs::speculative::{
    ConstraintPruner, NoScreeningPruner, build_dd_tree_pruned, build_dd_tree_screened,
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

// ── Pruners ────────────────────────────────────────────────────

/// Pruner that accepts tokens where token_idx % modulus == 0.
struct ModPruner {
    modulus: usize,
}

impl ConstraintPruner for ModPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx.is_multiple_of(self.modulus)
    }
}

/// Pruner that accepts tokens below a threshold.
struct ThresholdPruner {
    max_token: usize,
}

impl ConstraintPruner for ThresholdPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx <= self.max_token
    }
}

// ── Bench 1: Concept mining throughput ─────────────────────────

fn bench_concept_mining() {
    println!("── Bench 1: Concept mining throughput ──\n");

    let pruners: Vec<Arc<dyn ConstraintPruner>> = vec![
        Arc::new(ModPruner { modulus: 2 }),
        Arc::new(ModPruner { modulus: 3 }),
        Arc::new(ThresholdPruner { max_token: 20 }),
    ];

    let depths = [5, 10, 20];
    let tokens: Vec<usize> = (0..27).collect();

    println!(
        "  {:>10} {:>12} {:>18} {:>12}",
        "Depths", "Tokens", "Discovered", "Time (μs)"
    );
    println!("  {}", "-".repeat(56));

    for &max_depth in &depths {
        let mut rosetta = RosettaPruner::new(pruners.clone());

        let start = Instant::now();
        let discovered = rosetta.mine_concepts(max_depth, &tokens, &[]);
        let elapsed = start.elapsed();

        println!(
            "  {:>10} {:>12} {:>18} {:>12.2}",
            max_depth,
            tokens.len(),
            discovered,
            elapsed.as_nanos() as f64 / 1000.0,
        );
    }
    println!();
}

// ── Bench 2: DDTree build time with vs without RosettaPruner ───

fn bench_ddtree_with_without_rosetta() {
    println!("── Bench 2: DDTree build time — with vs without RosettaPruner ──\n");

    let config = Config {
        vocab_size: 27,
        tree_budget: 512,
        draft_lookahead: 8,
        ..Config::draft()
    };

    let marginals = make_marginals(8, config.vocab_size);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let mod2: Arc<dyn ConstraintPruner> = Arc::new(ModPruner { modulus: 2 });
    let mod3: Arc<dyn ConstraintPruner> = Arc::new(ModPruner { modulus: 3 });
    let thresh: Arc<dyn ConstraintPruner> = Arc::new(ThresholdPruner { max_token: 20 });

    // Baseline: build with individual Mod2 pruner
    let baseline_pruner = ModPruner { modulus: 2 };

    // Rosetta: combine all three
    let mut rosetta = RosettaPruner::new(vec![mod2.clone(), mod3.clone(), thresh.clone()]);
    let tokens: Vec<usize> = (0..config.vocab_size).collect();
    rosetta.mine_concepts(8, &tokens, &[]);

    // Warmup
    for _ in 0..WARMUP {
        let _ = build_dd_tree_pruned(&mv, &config, &baseline_pruner, false);
    }
    for _ in 0..WARMUP {
        let _ = build_dd_tree_pruned(&mv, &config, &rosetta, false);
    }

    // Measure baseline
    let mut baseline_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_pruned(&mv, &config, &baseline_pruner, false);
        baseline_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    // Measure Rosetta
    let mut rosetta_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_pruned(&mv, &config, &rosetta, false);
        rosetta_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    let baseline_stats = compute_stats(&baseline_samples);
    let rosetta_stats = compute_stats(&rosetta_samples);

    // Count tree nodes
    let baseline_tree = build_dd_tree_pruned(&mv, &config, &baseline_pruner, false);
    let rosetta_tree = build_dd_tree_pruned(&mv, &config, &rosetta, false);

    println!(
        "  Config: vocab={}, budget={}, depths=8, iters={}",
        config.vocab_size, config.tree_budget, N_ITERS
    );
    println!();
    println!(
        "  {:>20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Path", "Nodes", "p50 (μs)", "p99 (μs)", "mean (μs)", "min (μs)"
    );
    println!("  {}", "-".repeat(76));
    println!(
        "  {:>20} {:>10} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "Mod2 baseline",
        baseline_tree.len(),
        baseline_stats.p50,
        baseline_stats.p99,
        baseline_stats.mean,
        baseline_stats.min,
    );
    println!(
        "  {:>20} {:>10} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
        "Rosetta (3 pruners)",
        rosetta_tree.len(),
        rosetta_stats.p50,
        rosetta_stats.p99,
        rosetta_stats.mean,
        rosetta_stats.min,
    );

    let node_reduction = if baseline_tree.is_empty() {
        0.0
    } else {
        (1.0 - rosetta_tree.len() as f64 / baseline_tree.len() as f64) * 100.0
    };
    let time_overhead = (rosetta_stats.mean - baseline_stats.mean) / baseline_stats.mean * 100.0;

    println!("\n  Node reduction: {node_reduction:.1}%");
    println!("  Time overhead: {time_overhead:.1}%");

    if node_reduction >= 20.0 {
        println!("  ✅ ≥20% node reduction — RosettaPruner is effective");
    } else if node_reduction > 0.0 {
        println!("  ⚠️  <20% node reduction — may need more diverse pruners");
    }
    println!();
}

// ── Bench 3: ScreeningPruner path — DDTree with screened build ──

fn bench_ddtree_screened_rosetta() {
    println!("── Bench 3: DDTree screened build with RosettaPruner ──\n");

    let config = Config {
        vocab_size: 27,
        tree_budget: 512,
        draft_lookahead: 8,
        screening_threshold: 0.1,
        ..Config::draft()
    };

    let marginals = make_marginals(8, config.vocab_size);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let mod2: Arc<dyn ConstraintPruner> = Arc::new(ModPruner { modulus: 2 });
    let mod3: Arc<dyn ConstraintPruner> = Arc::new(ModPruner { modulus: 3 });

    let mut rosetta = RosettaPruner::new(vec![mod2, mod3]);
    let tokens: Vec<usize> = (0..config.vocab_size).collect();
    rosetta.mine_concepts(8, &tokens, &[]);

    // Baseline: NoScreeningPruner
    let baseline_screener = NoScreeningPruner;

    // Warmup
    for _ in 0..WARMUP {
        let _ = build_dd_tree_screened(&mv, &config, &baseline_screener, false);
    }
    for _ in 0..WARMUP {
        let _ = build_dd_tree_screened(&mv, &config, &rosetta, false);
    }

    // Measure baseline
    let mut baseline_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_screened(&mv, &config, &baseline_screener, false);
        baseline_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    // Measure Rosetta as ScreeningPruner
    let mut rosetta_samples = Vec::with_capacity(N_ITERS as usize);
    for _ in 0..N_ITERS {
        let start = Instant::now();
        let tree = build_dd_tree_screened(&mv, &config, &rosetta, false);
        rosetta_samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
        std::hint::black_box(tree);
    }

    let baseline_stats = compute_stats(&baseline_samples);
    let rosetta_stats = compute_stats(&rosetta_samples);

    let baseline_tree = build_dd_tree_screened(&mv, &config, &baseline_screener, false);
    let rosetta_tree = build_dd_tree_screened(&mv, &config, &rosetta, false);

    println!(
        "  {:>20} {:>10} {:>10} {:>10}",
        "Path", "Nodes", "p50 (μs)", "mean (μs)"
    );
    println!("  {}", "-".repeat(56));
    println!(
        "  {:>20} {:>10} {:>10.2} {:>10.2}",
        "NoScreeningPruner",
        baseline_tree.len(),
        baseline_stats.p50,
        baseline_stats.mean,
    );
    println!(
        "  {:>20} {:>10} {:>10.2} {:>10.2}",
        "RosettaPruner",
        rosetta_tree.len(),
        rosetta_stats.p50,
        rosetta_stats.mean,
    );
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Rosetta Pruners Benchmark — Plan 201");
    println!("  Warmup: {WARMUP} iters, Measure: {N_ITERS} iters");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    bench_concept_mining();
    bench_ddtree_with_without_rosetta();
    bench_ddtree_screened_rosetta();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════════");
}

// TL;DR: 3-benchmark suite for Plan 201 Rosetta Pruners — concept mining throughput,
// DDTree build with vs without RosettaPruner, screened build with RosettaPruner.
// Run with: cargo run --features rosetta_pruner --example rosetta_01_bench
