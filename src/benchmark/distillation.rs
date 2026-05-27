use super::{BenchCategory, BenchResult};
use std::time::Instant;

/// Distillation / Compression benchmarks for the Paper Feature Comparison Matrix.
///
/// Benchmarks three feature-gated sub-systems:
/// - **BT Pairwise Ranking** (`bt_rank`): `bt_pair_random` + `bt_fit` throughput
/// - **BanditPruner** (`bandit`): `update` + `prepare_episode` throughput
/// - **AbsorbCompress** (`g_zero`): `absorb` + `compress` throughput
///
/// All results use `feature_dim: "Distill"` and `category: BenchCategory::Distillation`.
pub fn bench_distillation() -> Vec<BenchResult> {
    let mut results = Vec::new();

    #[cfg(feature = "bt_rank")]
    results.extend(bench_bt_rank());

    #[cfg(feature = "bandit")]
    results.extend(bench_bandit());

    #[cfg(feature = "g_zero")]
    results.extend(bench_absorb_compress());

    print_summary(&results);
    results
}

// ── BT Pairwise Ranking ────────────────────────────────────────

#[cfg(feature = "bt_rank")]
fn bench_bt_rank() -> Vec<BenchResult> {
    use crate::pruners::bt_rank::{BtComparison, BtConfig, bt_fit, bt_pair_random};
    use fastrand::Rng;
    use std::hint::black_box;

    let warmup = 100;
    let iters = 5_000;
    let n_candidates = 20;
    let k_per_candidate = 5;

    println!("   BT Ranking ({iters} iters, {warmup} warmup, {n_candidates} candidates)...");

    let mut rng = Rng::new();
    let config = BtConfig::default();

    // Generate comparison pairs with bt_pair_random
    // Warmup
    for _ in 0..warmup {
        let _ = black_box(bt_pair_random(n_candidates, k_per_candidate, &mut rng));
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = black_box(bt_pair_random(n_candidates, k_per_candidate, &mut rng));
    }
    let elapsed = start.elapsed();
    let pair_throughput = iters as f64 / elapsed.as_secs_f64();
    let pair_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    // Prepare comparisons for bt_fit benchmark
    let pairs = bt_pair_random(n_candidates, k_per_candidate, &mut rng);
    let comparisons: Vec<BtComparison> = pairs
        .into_iter()
        .map(|(a, b)| BtComparison {
            winner: a,
            loser: b,
        })
        .collect();

    // Warmup bt_fit
    for _ in 0..warmup {
        let _ = black_box(bt_fit(&comparisons, n_candidates, &config));
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = black_box(bt_fit(&comparisons, n_candidates, &config));
    }
    let elapsed = start.elapsed();
    let fit_throughput = iters as f64 / elapsed.as_secs_f64();
    let fit_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    vec![
        BenchResult {
            label: "BT bt_pair_random".into(),
            throughput: pair_throughput,
            time_per_step_us: pair_us,
            avg_acceptance_len: 0.0,
            color: (255, 99, 71), // tomato
            category: BenchCategory::Distillation,
            feature_dim: "Distill".into(),
        },
        BenchResult {
            label: "BT bt_fit".into(),
            throughput: fit_throughput,
            time_per_step_us: fit_us,
            avg_acceptance_len: 0.0,
            color: (255, 99, 71), // tomato
            category: BenchCategory::Distillation,
            feature_dim: "Distill".into(),
        },
    ]
}

// ── BanditPruner ───────────────────────────────────────────────

#[cfg(feature = "bandit")]
#[allow(clippy::unit_arg)]
fn bench_bandit() -> Vec<BenchResult> {
    use crate::pruners::bandit::{BanditPruner, BanditStrategy};
    use crate::speculative::types::NoScreeningPruner;
    use crate::types::Rng;
    use std::hint::black_box;

    let warmup = 100;
    let iters = 5_000;
    let num_arms = 100;

    println!("   BanditPruner ({iters} iters, {warmup} warmup, {num_arms} arms)...");

    let mut rng = Rng::new(42);

    // ── update() throughput ──
    let mut bandit: BanditPruner<NoScreeningPruner> =
        BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);

    for _ in 0..warmup {
        let arm = (rng.next() as usize) % num_arms;
        black_box(bandit.update(arm, 0.5));
    }

    let start = Instant::now();
    for _ in 0..iters {
        let arm = (rng.next() as usize) % num_arms;
        black_box(bandit.update(arm, 0.5));
    }
    let elapsed = start.elapsed();
    let update_throughput = iters as f64 / elapsed.as_secs_f64();
    let update_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    // ── prepare_episode() throughput (Thompson Sampling) ──
    let mut bandit_ts: BanditPruner<NoScreeningPruner> = BanditPruner::new(
        NoScreeningPruner,
        BanditStrategy::ThompsonSampling,
        num_arms,
    );

    // Seed some visits so Thompson has data to sample from
    for arm in 0..num_arms {
        bandit_ts.update(arm, 0.5);
    }

    for _ in 0..warmup {
        black_box(bandit_ts.prepare_episode(&mut rng));
    }

    let start = Instant::now();
    for _ in 0..iters {
        black_box(bandit_ts.prepare_episode(&mut rng));
    }
    let elapsed = start.elapsed();
    let prep_throughput = iters as f64 / elapsed.as_secs_f64();
    let prep_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    vec![
        BenchResult {
            label: "Bandit update()".into(),
            throughput: update_throughput,
            time_per_step_us: update_us,
            avg_acceptance_len: 0.0,
            color: (50, 205, 50), // lime green
            category: BenchCategory::Distillation,
            feature_dim: "Distill".into(),
        },
        BenchResult {
            label: "Bandit prepare_episode()".into(),
            throughput: prep_throughput,
            time_per_step_us: prep_us,
            avg_acceptance_len: 0.0,
            color: (0, 128, 128), // teal
            category: BenchCategory::Distillation,
            feature_dim: "Distill".into(),
        },
    ]
}

// ── AbsorbCompress ─────────────────────────────────────────────

#[cfg(feature = "g_zero")]
#[allow(clippy::unit_arg)]
fn bench_absorb_compress() -> Vec<BenchResult> {
    use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer, CompressConfig};
    use crate::speculative::types::NoScreeningPruner;
    use crate::types::Rng;
    use std::hint::black_box;

    let warmup = 100;
    let iters = 10_000;
    let num_arms = 6;

    println!("   AbsorbCompress ({iters} iters, {warmup} warmup, {num_arms} arms)...");

    let mut rng = Rng::new(42);

    let compress_config = CompressConfig::new(20, 0.1, 2, 100);

    // ── absorb() throughput ──
    let mut ac: AbsorbCompressLayer<NoScreeningPruner> =
        AbsorbCompressLayer::new(NoScreeningPruner, num_arms, compress_config.clone());

    for _ in 0..warmup {
        let arm = (rng.next() as usize) % num_arms;
        let reward = rng.uniform();
        black_box(ac.absorb(arm, reward));
    }

    let start = Instant::now();
    for _ in 0..iters {
        let arm = (rng.next() as usize) % num_arms;
        let reward = rng.uniform();
        black_box(ac.absorb(arm, reward));
    }
    let elapsed = start.elapsed();
    let absorb_throughput = iters as f64 / elapsed.as_secs_f64();
    let absorb_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    // ── compress() throughput ──
    // Seed enough visits so compress() has candidates to evaluate
    let mut ac2: AbsorbCompressLayer<NoScreeningPruner> =
        AbsorbCompressLayer::new(NoScreeningPruner, num_arms, compress_config);
    for arm in 0..num_arms {
        for _ in 0..25 {
            ac2.absorb(arm, 0.05); // low rewards → compress candidates
        }
    }

    for _ in 0..warmup {
        let _ = black_box(ac2.compress());
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = black_box(ac2.compress());
    }
    let elapsed = start.elapsed();
    let compress_throughput = iters as f64 / elapsed.as_secs_f64();
    let compress_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    vec![
        BenchResult {
            label: "AbsorbCompress absorb()".into(),
            throughput: absorb_throughput,
            time_per_step_us: absorb_us,
            avg_acceptance_len: 0.0,
            color: (255, 215, 0), // gold
            category: BenchCategory::Distillation,
            feature_dim: "Distill".into(),
        },
        BenchResult {
            label: "AbsorbCompress compress()".into(),
            throughput: compress_throughput,
            time_per_step_us: compress_us,
            avg_acceptance_len: 0.0,
            color: (218, 112, 214), // orchid
            category: BenchCategory::Distillation,
            feature_dim: "Distill".into(),
        },
    ]
}

// ── Summary ────────────────────────────────────────────────────

fn print_summary(results: &[BenchResult]) {
    if results.is_empty() {
        println!(
            "   [Distill] No benchmarks enabled (requires bt_rank, bandit, or g_zero features)"
        );
        return;
    }

    println!("\n   ┌─────────────────────────────────────────────────────────────────────┐");
    println!("   │  Distillation / Compression Benchmarks                             │");
    println!("   ├──────────────────────────────┬──────────────┬──────────────┬────────┤");
    println!("   │ Benchmark                    │   ops/sec    │    μs/op     │ Color  │");
    println!("   ├──────────────────────────────┼──────────────┼──────────────┼────────┤");
    for r in results {
        println!(
            "   │ {:<28} │ {:>12.0} │ {:>12.2} │ {:>6} │",
            r.label,
            r.throughput,
            r.time_per_step_us,
            format!("({},{},{})", r.color.0, r.color.1, r.color.2)
        );
    }
    println!("   └──────────────────────────────┴──────────────┴──────────────┴────────┘");
}
