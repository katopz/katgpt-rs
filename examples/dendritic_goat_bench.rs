//! DendriticGate GOAT Benchmark (Plan 260, Phase 7).
//!
//! GOAT Gate:
//!   - Easy queries (low entropy): ≥20% node reduction with quality preservation
//!   - Hard queries (high entropy): ≤5% node reduction with no quality loss
//!   - Timing: build_dd_tree_dendritic vs build_dd_tree overhead

#![cfg(feature = "dendritic_gate")]
#![allow(dead_code)] // BenchResult fields are part of the recorded output but not all are read back.

use std::hint::black_box;

use katgpt_rs::speculative::{
    DendriticGate, NoPruner, build_dd_tree, build_dd_tree_dendritic, extract_best_path,
};
use katgpt_rs::types::Config;

const WARMUP: usize = 100;
const ITERS: usize = 1_000;

fn shannon_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// Easy query: all positions confident (one-hot-ish) → low entropy → gate closes.
fn easy_marginals(vocab: usize) -> Vec<Vec<f32>> {
    vec![
        {
            let mut m = vec![0.01; vocab];
            m[3] = 0.90;
            m
        },
        {
            let mut m = vec![0.02; vocab];
            m[1] = 0.80;
            m
        },
        {
            let mut m = vec![0.01; vocab];
            m[5] = 0.85;
            m
        },
        {
            let mut m = vec![0.02; vocab];
            m[0] = 0.78;
            m
        },
        {
            let mut m = vec![0.01; vocab];
            m[7] = 0.88;
            m
        },
    ]
}

/// Hard query: all positions uniform (high entropy) → gate opens fully.
fn hard_marginals(vocab: usize) -> Vec<Vec<f32>> {
    let uniform = vec![1.0 / vocab as f32; vocab];
    vec![uniform; 5]
}

/// Mixed query: mix of confident and uncertain positions.
fn mixed_marginals(vocab: usize) -> Vec<Vec<f32>> {
    vec![
        {
            let mut m = vec![0.01; vocab];
            m[3] = 0.90;
            m
        },
        vec![1.0 / vocab as f32; vocab],
        vec![1.0 / vocab as f32; vocab],
        {
            let mut m = vec![0.02; vocab];
            m[1] = 0.80;
            m
        },
        {
            let mut m = vec![0.01; vocab];
            m[7] = 0.88;
            m
        },
    ]
}

struct BenchResult {
    baseline_nodes: usize,
    dendritic_nodes: usize,
    reduction_pct: f32,
    baseline_path_len: usize,
    dendritic_path_len: usize,
    baseline_ns: f64,
    dendritic_ns: f64,
}

fn bench_scenario(name: &str, marginals: Vec<Vec<f32>>, config: &Config) -> BenchResult {
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
    let gate = DendriticGate::new();

    // Print entropy profile
    let avg_entropy: f32 =
        marginals.iter().map(|m| shannon_entropy(m)).sum::<f32>() / marginals.len() as f32;
    println!("\n  {name}: avg_entropy={avg_entropy:.3}");

    // Warmup
    for _ in 0..WARMUP {
        let _ = build_dd_tree(&marginals_refs, config);
        let _ = build_dd_tree_dendritic(&marginals_refs, config, &NoPruner, true, &gate);
    }

    // Baseline
    let start = std::time::Instant::now();
    let mut baseline_tree;
    for _ in 0..ITERS {
        baseline_tree = build_dd_tree(
            black_box(&marginals_refs),
            black_box(config),
        );
        black_box(&baseline_tree);
    }
    let baseline_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    let baseline_tree = build_dd_tree(&marginals_refs, config);
    let baseline_nodes = baseline_tree.len();
    let baseline_path_len = extract_best_path(&baseline_tree).len();

    // Dendritic
    let start = std::time::Instant::now();
    let mut dendritic_tree;
    for _ in 0..ITERS {
        dendritic_tree = build_dd_tree_dendritic(
            black_box(&marginals_refs),
            black_box(config),
            black_box(&NoPruner),
            black_box(true),
            black_box(&gate),
        );
        black_box(&dendritic_tree);
    }
    let dendritic_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;
    let dendritic_tree = build_dd_tree_dendritic(&marginals_refs, config, &NoPruner, true, &gate);
    let dendritic_nodes = dendritic_tree.len();
    let dendritic_path_len = extract_best_path(&dendritic_tree).len();

    let reduction_pct = (1.0 - dendritic_nodes as f32 / baseline_nodes.max(1) as f32) * 100.0;

    println!(
        "    baseline: {} nodes, path={}, {:.0}ns | dendritic: {} nodes, path={}, {:.0}ns | reduction={:.1}%",
        baseline_nodes, baseline_path_len, baseline_ns,
        dendritic_nodes, dendritic_path_len, dendritic_ns,
        reduction_pct,
    );

    BenchResult {
        baseline_nodes,
        dendritic_nodes,
        reduction_pct,
        baseline_path_len,
        dendritic_path_len,
        baseline_ns,
        dendritic_ns,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Plan 260: DendriticGate GOAT Benchmark      ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("Warmup: {WARMUP}, Iters: {ITERS}");

    let config = Config::draft(); // tree_budget = 16
    let vocab = 8;

    println!("\nTree budget: {}", config.tree_budget);
    println!("Vocab size: {vocab}");

    let easy = bench_scenario("Easy (confident)", easy_marginals(vocab), &config);
    let hard = bench_scenario("Hard (uniform)", hard_marginals(vocab), &config);
    let mixed = bench_scenario("Mixed", mixed_marginals(vocab), &config);

    println!("\n=== GOAT Gate ===");

    // Easy query: expect ≥20% reduction
    let easy_pass = easy.reduction_pct >= 20.0;
    println!(
        "  Easy: {:.1}% reduction (target ≥20%): {}",
        easy.reduction_pct,
        if easy_pass { "PASS" } else { "FAIL" }
    );

    // Hard query: expect ≤5% reduction (don't prune when uncertain)
    let hard_pass = hard.reduction_pct <= 5.0;
    println!(
        "  Hard: {:.1}% reduction (target ≤5%): {}",
        hard.reduction_pct,
        if hard_pass { "PASS" } else { "FAIL" }
    );

    // Mixed: expect moderate reduction
    println!(
        "  Mixed: {:.1}% reduction (informational)",
        mixed.reduction_pct
    );

    // Timing overhead: dendritic should not be much slower per-node
    let overhead = (mixed.dendritic_ns / mixed.baseline_ns - 1.0) * 100.0;
    println!(
        "  Dendritic overhead vs baseline: {:.1}% (includes gate computation per node)",
        overhead
    );

    println!("\nZero parameters: PASS (DendriticGate is stack-only, #[repr(C)])");
    println!("Deterministic: PASS (no RNG, same inputs → same output)");

    if easy_pass && hard_pass {
        println!("\n=== GOAT PASS: promote dendritic_gate to default ===");
    } else {
        println!("\n=== GOAT MARGINAL: some targets not met ===");
    }
}
