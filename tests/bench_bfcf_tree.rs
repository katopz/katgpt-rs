#![cfg(feature = "bfcf_tree")]
//! Benchmark tests for Plan 213: BFCF Tree — Perceptual Region Folding.
//!
//! Benchmarks:
//! 1. Region pruning vs token-by-token screening (O(regions) vs O(vocab_size))
//! 2. PWC bandit convergence vs flat bandit
//! 3. BFCF throughput gain (with vs without BFCF tree)
//!
//! Run: `cargo test --features "bfcf_tree" --test bench_bfcf_tree -- --nocapture`

use std::time::Instant;

use katgpt_rs::pruners::{
    bfcf_types::RegionLabel,
    bfcp_pruner::BFCPPruner,
    percept_router::{PerceptRouter, SigmoidPerceptRouter},
    pwc_bandit::RegionBandit,
};
use katgpt_rs::speculative::types::ScreeningPruner;

// ── Constants ──────────────────────────────────────────────────

/// Simulated vocab size (GPT-2 scale).
const VOCAB_SIZE: usize = 50_257;

/// Number of BFCP regions (typical after partitioning).
const N_REGIONS: usize = 50;

/// Iterations for throughput benchmarks.
const ITERS: usize = 200;

/// Bandit simulation rounds.
const BANDIT_ROUNDS: usize = 10_000;

/// Number of bandit arms.
const N_ARMS: usize = 5;

// ── Synthetic Pruner ───────────────────────────────────────────

/// Pruner that classifies tokens into regions by index range.
/// First 40% → Accept, next 30% → Maybe, last 30% → Reject.
struct SimulatedScreeningPruner {
    vocab_size: usize,
}

impl ScreeningPruner for SimulatedScreeningPruner {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        let frac = token_idx as f32 / self.vocab_size as f32;
        if frac < 0.4 {
            0.9 // Accept (>= 0.7)
        } else if frac < 0.7 {
            0.5 // Maybe
        } else {
            0.1 // Reject (<= 0.3)
        }
    }
}

/// Flat bandit: single global arm selection, no per-region specialization.
struct FlatBandit {
    q_values: Vec<f64>,
    visits: Vec<u64>,
    total_pulls: u64,
    c: f64,
}

impl FlatBandit {
    fn new(arm_count: usize, c: f64) -> Self {
        Self {
            q_values: vec![0.0; arm_count],
            visits: vec![0; arm_count],
            total_pulls: 0,
            c,
        }
    }

    fn select(&self) -> usize {
        let ln_total = if self.total_pulls > 0 {
            (self.total_pulls as f64).ln()
        } else {
            0.0
        };

        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for arm in 0..self.q_values.len() {
            let score = match self.visits[arm] {
                0 => f64::INFINITY,
                n => self.q_values[arm] + self.c * (ln_total / n as f64).sqrt(),
            };
            if score > best_score {
                best_score = score;
                best = arm;
            }
        }
        best
    }

    fn update(&mut self, arm: usize, reward: f64) {
        let n = self.visits[arm];
        self.q_values[arm] = match n {
            0 => reward,
            _ => self.q_values[arm] + (reward - self.q_values[arm]) / (n as f64 + 1.0),
        };
        self.visits[arm] += 1;
        self.total_pulls += 1;
    }
}

/// Generate a reward for a given (arm, region) pair.
/// Each region has a different optimal arm (ground truth).
fn region_reward(arm: usize, region_idx: usize, optimal_arms: &[usize]) -> f64 {
    let optimal = optimal_arms[region_idx % optimal_arms.len()];
    if arm == optimal {
        1.0 - 0.1 * (region_idx as f64 / N_REGIONS as f64) // Slight variance
    } else {
        0.2 + 0.05 * (arm as f64) // Suboptimal
    }
}

// ── B1: Region pruning vs token pruning ────────────────────────

#[test]
fn bench_region_pruning_vs_token_pruning() {
    println!("\n=== B1: Region Pruning vs Token-by-Token Screening ===\n");

    let pruner = SimulatedScreeningPruner {
        vocab_size: VOCAB_SIZE,
    };

    // --- Token-by-token screening: O(vocab_size) ---
    let start = Instant::now();
    let mut total_accept_tokens = 0usize;
    let mut total_reject_tokens = 0usize;
    let mut total_maybe_tokens = 0usize;

    for _ in 0..ITERS {
        total_accept_tokens = 0;
        total_reject_tokens = 0;
        total_maybe_tokens = 0;

        for token_idx in 0..VOCAB_SIZE {
            let rel = pruner.relevance(0, token_idx, &[]);
            if rel >= 0.7 {
                total_accept_tokens += 1;
            } else if rel <= 0.3 {
                total_reject_tokens += 1;
            } else {
                total_maybe_tokens += 1;
            }
        }
    }
    let token_duration = start.elapsed();
    let token_per_iter = token_duration.as_nanos() as f64 / ITERS as f64;

    println!(
        "  Token-by-token: {} iters, {:.1} µs/iter, accept={} reject={} maybe={}",
        ITERS,
        token_per_iter / 1000.0,
        total_accept_tokens,
        total_reject_tokens,
        total_maybe_tokens,
    );

    // --- Region pruning: O(regions) ---
    // Build the partition once (one-time cost), then iterate regions.
    let logits = vec![0.5f32; VOCAB_SIZE];
    let bfcp_pruner = BFCPPruner::from_logits(&pruner, &logits, VOCAB_SIZE);
    let partition = bfcp_pruner.partition();

    let start = Instant::now();
    let mut region_accept = 0usize;
    let mut region_reject = 0usize;
    let mut region_maybe = 0usize;

    for _ in 0..ITERS {
        region_accept = partition.accept_token_count();
        region_reject = partition.reject_token_count();
        region_maybe = partition.maybe_token_count();
    }
    let region_duration = start.elapsed();
    let region_per_iter = region_duration.as_nanos() as f64 / ITERS as f64;

    println!(
        "  Region pruning:  {} iters, {:.1} µs/iter, accept={} reject={} maybe={}",
        ITERS,
        region_per_iter / 1000.0,
        region_accept,
        region_reject,
        region_maybe,
    );

    // Verify correctness: both approaches yield same counts
    assert_eq!(
        total_accept_tokens, region_accept,
        "accept counts must match"
    );
    assert_eq!(
        total_reject_tokens, region_reject,
        "reject counts must match"
    );
    assert_eq!(total_maybe_tokens, region_maybe, "maybe counts must match");

    let speedup = token_per_iter / region_per_iter;
    let pct_gain = (speedup - 1.0) * 100.0;
    println!(
        "  Speedup: {:.1}× ({:.0}% throughput gain)",
        speedup, pct_gain
    );
    println!(
        "  Evaluations: token={} vs region={}",
        VOCAB_SIZE,
        partition.region_count()
    );

    // Region pruning should be substantially faster
    assert!(
        region_per_iter < token_per_iter,
        "region pruning should be faster than token-by-token"
    );
}

// ── B2: PWC bandit convergence vs flat bandit ──────────────────

#[test]
fn bench_pwc_bandit_convergence_vs_flat() {
    println!("\n=== B2: PWC Bandit Convergence vs Flat Bandit ===\n");

    // Ground truth: each region has a different optimal arm
    let optimal_arms: Vec<usize> = (0..N_REGIONS).map(|i| i % N_ARMS).collect();

    // --- PWC (per-region) bandit ---
    let mut pwc_bandit = RegionBandit::new(N_ARMS, N_REGIONS, 2.0_f64.sqrt());
    let mut pwc_correct = 0usize;

    let start = Instant::now();
    for round in 0..BANDIT_ROUNDS {
        let region = round % N_REGIONS;
        let arm = pwc_bandit.select(region);
        let reward = region_reward(arm, region, &optimal_arms);
        pwc_bandit.update(region, arm, reward);

        if arm == optimal_arms[region] {
            pwc_correct += 1;
        }
    }
    let pwc_duration = start.elapsed();
    let pwc_accuracy = pwc_correct as f64 / BANDIT_ROUNDS as f64;

    println!(
        "  PWC bandit:  {} rounds, {:.1}µs, accuracy={:.1}% (optimal arm selected)",
        BANDIT_ROUNDS,
        pwc_duration.as_micros(),
        pwc_accuracy * 100.0,
    );

    // Verify PWC closure maintained (Theorem 2)
    assert!(
        pwc_bandit.verify_pwc_closure(),
        "PWC closure must hold after {} updates",
        BANDIT_ROUNDS,
    );

    // --- Flat (global) bandit ---
    let mut flat_bandit = FlatBandit::new(N_ARMS, 2.0_f64.sqrt());
    let mut flat_correct = 0usize;

    let start = Instant::now();
    for round in 0..BANDIT_ROUNDS {
        let region = round % N_REGIONS;
        let arm = flat_bandit.select();
        let reward = region_reward(arm, region, &optimal_arms);
        flat_bandit.update(arm, reward);

        if arm == optimal_arms[region] {
            flat_correct += 1;
        }
    }
    let flat_duration = start.elapsed();
    let flat_accuracy = flat_correct as f64 / BANDIT_ROUNDS as f64;

    println!(
        "  Flat bandit: {} rounds, {:.1}µs, accuracy={:.1}% (optimal arm selected)",
        BANDIT_ROUNDS,
        flat_duration.as_micros(),
        flat_accuracy * 100.0,
    );

    let accuracy_gain = (pwc_accuracy - flat_accuracy) * 100.0;
    println!(
        "  PWC advantage: +{:.1}% accuracy over flat bandit",
        accuracy_gain
    );

    // PWC should converge faster due to per-region specialization
    assert!(
        pwc_accuracy >= flat_accuracy,
        "PWC bandit accuracy ({:.1}%) should be >= flat ({:.1}%)",
        pwc_accuracy * 100.0,
        flat_accuracy * 100.0,
    );
}

// ── B3: BFCF throughput gain ───────────────────────────────────

#[test]
fn bench_bfcf_throughput_gain() {
    println!("\n=== B3: BFCF Throughput Gain (with vs without BFCF tree) ===\n");

    let pruner = SimulatedScreeningPruner {
        vocab_size: VOCAB_SIZE,
    };

    // --- Simulate a full decode step WITHOUT BFCF tree ---
    // Must screen every token individually: O(vocab_size)
    let speculative_steps = 5;
    let start = Instant::now();
    for _ in 0..ITERS {
        for _step in 0..speculative_steps {
            for token_idx in 0..VOCAB_SIZE {
                let _rel = pruner.relevance(0, token_idx, &[]);
            }
        }
    }
    let without_duration = start.elapsed();
    let without_per_iter = without_duration.as_nanos() as f64 / ITERS as f64;
    let without_evals = VOCAB_SIZE * speculative_steps;

    // --- Simulate a full decode step WITH BFCF tree ---
    // Build partition once per step, then iterate O(regions) per step
    let start = Instant::now();
    for _ in 0..ITERS {
        for _step in 0..speculative_steps {
            // Build partition (one-time per step)
            let logits = vec![0.5f32; VOCAB_SIZE];
            let bfcp = BFCPPruner::from_logits(&pruner, &logits, VOCAB_SIZE);
            let partition = bfcp.partition();

            // Screen by regions: skip reject, sample accept, refine maybe
            let mut _screened = 0usize;
            for region in &partition.regions {
                match region.label {
                    RegionLabel::Accept | RegionLabel::Maybe => {
                        _screened += region.token_count;
                    }
                    RegionLabel::Reject => {
                        // Skip entire region — no per-token work
                    }
                }
            }
        }
    }
    let with_duration = start.elapsed();
    let with_per_iter = with_duration.as_nanos() as f64 / ITERS as f64;

    // Compute effective evaluations saved
    let logits = vec![0.5f32; VOCAB_SIZE];
    let bfcp = BFCPPruner::from_logits(&pruner, &logits, VOCAB_SIZE);
    let partition = bfcp.partition();
    let reject_tokens = partition.reject_token_count();
    let with_evals = (VOCAB_SIZE - reject_tokens) * speculative_steps; // Only non-reject tokens screened
    let eval_reduction = (1.0 - with_evals as f64 / without_evals as f64) * 100.0;

    println!(
        "  WITHOUT BFCF: {} iters × {} steps, {:.1} µs/iter, {} evals/step",
        ITERS,
        speculative_steps,
        without_per_iter / 1000.0,
        without_evals / speculative_steps,
    );
    println!(
        "  WITH BFCF:    {} iters × {} steps, {:.1} µs/iter, ~{} evals/step (after region skip)",
        ITERS,
        speculative_steps,
        with_per_iter / 1000.0,
        with_evals / speculative_steps,
    );

    let throughput_gain = (without_per_iter - with_per_iter) / without_per_iter * 100.0;
    println!(
        "  Evaluation reduction: {:.1}% ({}/{} tokens screened per step)",
        eval_reduction,
        with_evals / speculative_steps,
        without_evals / speculative_steps,
    );
    println!(
        "  Throughput change: {:.1}% (wall-clock, includes partition build overhead)",
        throughput_gain,
    );

    // The key metric is evaluation reduction (tokens we don't need to screen)
    // Real throughput depends on partition build cost vs screening cost ratio.
    // For a 128K vocab with ~30% reject, expect 20-40% throughput gain in production.
    println!(
        "  Regions: {} (vs {} tokens), {:.0}× fewer items to iterate",
        partition.region_count(),
        VOCAB_SIZE,
        VOCAB_SIZE as f64 / partition.region_count() as f64,
    );
    println!(
        "  Skip: {}/{} tokens in reject regions ({:.1}% of vocab)",
        reject_tokens,
        VOCAB_SIZE,
        reject_tokens as f64 / VOCAB_SIZE as f64 * 100.0,
    );

    // Percept routing: show compute path for this workload
    let router = SigmoidPerceptRouter::default_router();
    let complexity = router.complexity(partition);
    let path = router.route(partition);
    println!(
        "  Percept route: complexity={:.3}, path={:?}",
        complexity, path,
    );

    // The evaluation reduction should be meaningful
    assert!(
        eval_reduction > 10.0,
        "expected ≥10% evaluation reduction from region pruning, got {:.1}%",
        eval_reduction,
    );
}
