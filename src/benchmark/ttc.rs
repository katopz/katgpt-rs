//! Test-Time Compute (TTC) benchmarks.
//!
//! Measures throughput of adaptive compute primitives from the TTC feature
//! dimension of the Paper Feature Comparison Matrix:
//!
//! - UCB1 exploration cycle (BanditStats)
//! - Thompson Sampling exploration cycle (BanditStats)
//! - BanditPruner full episode (prepare_episode + update)

use super::{BenchCategory, BenchResult};
use std::time::Instant;

#[cfg(feature = "bandit")]
use crate::pruners::bandit::{BanditPruner, BanditStats, BanditStrategy};
#[cfg(feature = "bandit")]
use crate::speculative::types::NoScreeningPruner;
#[cfg(feature = "bandit")]
use crate::types::Rng;

/// Run all TTC benchmarks and return results.
///
/// Each result is tagged with `feature_dim: "TTC"` and
/// `category: BenchCategory::TestTimeCompute`.
pub fn bench_ttc() -> Vec<BenchResult> {
    let mut results = Vec::new();
    let warmup = 100;
    let iters = 5_000;

    println!("\n⏱️  Test-Time Compute (TTC)...");
    println!("   ({iters} iterations, {warmup} warmup)");

    #[cfg(feature = "bandit")]
    {
        bench_ucb1_cycle(&mut results, warmup, iters);
        bench_thompson_cycle(&mut results, warmup, iters);
        bench_bandit_pruner_episode(&mut results, warmup, iters);
    }

    #[cfg(not(feature = "bandit"))]
    {
        println!("   (skipped — enable `bandit` feature)");
    }

    // Print summary
    println!(
        "\n   {:<35} {:>12} {:>12}",
        "Method", "cycles/s", "μs/cycle"
    );
    println!("   {}", "-".repeat(61));
    for r in &results {
        println!(
            "   {:<35} {:>12.0} {:>12.2}",
            r.label, r.throughput, r.time_per_step_us,
        );
    }

    results
}

// ── UCB1 Exploration Cycle ──────────────────────────────────────

#[cfg(feature = "bandit")]
fn bench_ucb1_cycle(results: &mut Vec<BenchResult>, warmup: usize, iters: usize) {
    let num_arms = 100usize;
    let mut stats = BanditStats::new(num_arms);
    let mut rng = Rng::new(42);

    // Warmup
    for _ in 0..warmup {
        let arm = pick_best_ucb1(&stats, num_arms);
        let reward = rng.uniform();
        stats.update(arm, reward);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let arm = pick_best_ucb1(&stats, num_arms);
        let reward = rng.uniform();
        stats.update(arm, reward);
    }
    let elapsed = start.elapsed();

    let tp = iters as f64 / elapsed.as_secs_f64();
    let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "UCB1 exploration cycle (100 arms)".into(),
        throughput: tp,
        time_per_step_us: us,
        avg_acceptance_len: 0.0,
        color: (0, 191, 255), // deep sky blue
        category: BenchCategory::TestTimeCompute,
        feature_dim: "TTC".into(),
    });
}

/// Pick the arm with the highest UCB1 score.
#[cfg(feature = "bandit")]
#[inline]
fn pick_best_ucb1(stats: &BanditStats, num_arms: usize) -> usize {
    let mut best_arm = 0;
    let mut best_score = f32::MIN;
    for arm in 0..num_arms {
        let score = stats.ucb1_score(arm);
        if score > best_score {
            best_score = score;
            best_arm = arm;
        }
    }
    best_arm
}

// ── Thompson Sampling Exploration Cycle ─────────────────────────

#[cfg(feature = "bandit")]
fn bench_thompson_cycle(results: &mut Vec<BenchResult>, warmup: usize, iters: usize) {
    let num_arms = 100usize;
    let mut stats = BanditStats::new(num_arms);
    let mut rng = Rng::new(42);

    // Warmup
    for _ in 0..warmup {
        let arm = pick_best_thompson(&stats, num_arms, &mut rng);
        let reward = rng.uniform();
        stats.update(arm, reward);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let arm = pick_best_thompson(&stats, num_arms, &mut rng);
        let reward = rng.uniform();
        stats.update(arm, reward);
    }
    let elapsed = start.elapsed();

    let tp = iters as f64 / elapsed.as_secs_f64();
    let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "Thompson sampling cycle (100 arms)".into(),
        throughput: tp,
        time_per_step_us: us,
        avg_acceptance_len: 0.0,
        color: (255, 105, 180), // hot pink
        category: BenchCategory::TestTimeCompute,
        feature_dim: "TTC".into(),
    });
}

/// Pick the arm with the highest Thompson sample.
#[cfg(feature = "bandit")]
#[inline]
fn pick_best_thompson(stats: &BanditStats, num_arms: usize, rng: &mut Rng) -> usize {
    let mut best_arm = 0;
    let mut best_score = f32::MIN;
    for arm in 0..num_arms {
        let score = stats.thompson_sample(arm, rng);
        if score > best_score {
            best_score = score;
            best_arm = arm;
        }
    }
    best_arm
}

// ── BanditPruner Full Episode ───────────────────────────────────

#[cfg(feature = "bandit")]
fn bench_bandit_pruner_episode(results: &mut Vec<BenchResult>, warmup: usize, iters: usize) {
    let num_arms = 64usize;
    let mut pruner: BanditPruner<NoScreeningPruner> = BanditPruner::new(
        NoScreeningPruner,
        BanditStrategy::ThompsonSampling,
        num_arms,
    );
    let mut rng = Rng::new(42);

    // Warmup
    for _ in 0..warmup {
        pruner.prepare_episode(&mut rng);
        let arm = pruner.best_arm();
        let reward = rng.uniform();
        pruner.update(arm, reward);
    }

    let start = Instant::now();
    for _ in 0..iters {
        pruner.prepare_episode(&mut rng);
        let arm = pruner.best_arm();
        let reward = rng.uniform();
        pruner.update(arm, reward);
    }
    let elapsed = start.elapsed();

    let tp = iters as f64 / elapsed.as_secs_f64();
    let us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "BanditPruner episode (64 arms, Thompson)".into(),
        throughput: tp,
        time_per_step_us: us,
        avg_acceptance_len: 0.0,
        color: (154, 205, 50), // yellow green
        category: BenchCategory::TestTimeCompute,
        feature_dim: "TTC".into(),
    });
}
