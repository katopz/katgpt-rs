//! Benchmarks for Plan 062: FFO Distillation — Dual Cutoff Analysis.
//!
//! T2: Q-value distribution baseline — does BanditPruner already mask low-Q arms?
//! T5: A/B benchmark — dual_cutoff=0.0 vs 0.2 vs 0.5
//!
//! ```sh
//! RUST_LOG=info cargo test -p katgpt-rs --test bench_ffo_distillation --features bandit -- --nocapture
//! ```

#[cfg(feature = "bandit")]
use katgpt_rs::pruners::{BanditPruner, BanditStrategy};
#[cfg(feature = "bandit")]
use katgpt_rs::speculative::ScreeningPruner;
#[cfg(feature = "bandit")]
use katgpt_rs::speculative::types::NoScreeningPruner;
#[cfg(feature = "bandit")]
use katgpt_rs::types::Rng;

// ── Helpers ──────────────────────────────────────────────────────

/// Reward profile for simulated arm: defines the true expected reward.
#[cfg(feature = "bandit")]
#[derive(Clone, Copy)]
enum ArmProfile {
    /// High reward: expected ~0.85
    High,
    /// Medium reward: expected ~0.4
    Medium,
    /// Low reward: expected ~0.1
    Low,
}

/// Generate reward profiles: arms 0..5 = HIGH, 5..10 = MED, rest = LOW.
#[cfg(feature = "bandit")]
fn make_profiles(num_arms: usize) -> Vec<ArmProfile> {
    (0..num_arms)
        .map(|arm| match arm {
            0..=4 => ArmProfile::High,
            5..=9 => ArmProfile::Medium,
            _ => ArmProfile::Low,
        })
        .collect()
}

/// Sample a noisy reward for an arm given its profile.
#[cfg(feature = "bandit")]
fn sample_reward(profile: ArmProfile, rng: &mut Rng) -> f32 {
    let base = match profile {
        ArmProfile::High => 0.8 + rng.uniform() * 0.2,
        ArmProfile::Medium => 0.3 + rng.uniform() * 0.2,
        ArmProfile::Low => rng.uniform() * 0.2,
    };
    (base + (rng.uniform() - 0.5) * 0.1).clamp(0.0, 1.0)
}

// ── T2: Baseline — BanditPruner Q-Value Distribution ─────────────

/// Analyze whether BanditPruner already provides "active-set masking".
/// Run N episodes with simulated rewards, record Q-value distribution,
/// check if low-Q arms are effectively suppressed by the domain × bandit product.
///
/// **Gate T2:** If ≥80% of arms already have relevance < 0.01 after training,
/// the "masking" is already happening via soft blending. Skip to "NO GAIN".
#[cfg(feature = "bandit")]
#[test]
fn test_baseline_bandit_q_distribution() {
    let num_arms = 27;
    let episodes = 1000;
    let seed: u64 = 42;
    let mut rng = Rng::new(seed);

    let profiles = make_profiles(num_arms);
    let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);

    // Train: random arm selection, noisy rewards from profiles
    for _ in 0..episodes {
        let arm = (rng.next() as usize) % num_arms;
        let reward = sample_reward(profiles[arm], &mut rng);
        pruner.update(arm, reward);
    }

    // Prepare episode (for Thompson cache, no-op for UCB1)
    pruner.prepare_episode(&mut Rng::new(seed + 1));

    let q_values = pruner.q_values().to_vec();
    let visits = pruner.visits().to_vec();
    let relevances: Vec<f32> = (0..num_arms)
        .map(|arm| pruner.relevance(0, arm, &[]))
        .collect();

    // Bucket boundaries for histogram
    let boundaries = [0.01f32, 0.1, 0.3, 0.5, 0.7, 1.0];
    let labels = [
        "<0.01",
        "[0.01,0.1)",
        "[0.1,0.3)",
        "[0.3,0.5)",
        "[0.5,0.7)",
        "[0.7,1.0)",
    ];

    let near_zero_q = q_values.iter().filter(|&&q| q < 0.01).count();
    let near_zero_rel = relevances.iter().filter(|&&r| r < 0.01).count();
    let low_rel_pct = (near_zero_rel as f64 / num_arms as f64) * 100.0;

    println!("\n=== T2: Baseline Q-Value Distribution Analysis ===");
    println!("  Arms: {num_arms}, Episodes: {episodes}, Strategy: UCB1");
    println!();

    // Q-value histogram
    println!("  Q-Value Distribution:");
    for (i, label) in labels.iter().enumerate() {
        let count = if i == 0 {
            q_values.iter().filter(|&&q| q < boundaries[0]).count()
        } else {
            q_values
                .iter()
                .filter(|&&q| q >= boundaries[i - 1] && q < boundaries[i])
                .count()
        };
        let bar: String = "█".repeat(count);
        println!("    {label:14}: {count:3} {bar}");
    }

    println!();

    // Relevance histogram
    println!("  Relevance Distribution (domain × bandit):");
    for (i, label) in labels.iter().enumerate() {
        let count = if i == 0 {
            relevances.iter().filter(|&&r| r < boundaries[0]).count()
        } else {
            relevances
                .iter()
                .filter(|&&r| r >= boundaries[i - 1] && r < boundaries[i])
                .count()
        };
        let bar: String = "█".repeat(count);
        println!("    {label:14}: {count:3} {bar}");
    }

    println!();
    println!(
        "  Arms with Q < 0.01:        {near_zero_q}/{num_arms} ({:.1}%)",
        (near_zero_q as f64 / num_arms as f64) * 100.0
    );
    println!("  Arms with rel < 0.01:       {near_zero_rel}/{num_arms} ({low_rel_pct:.1}%)");

    // Per-arm detail
    println!();
    println!("  Per-arm detail:");
    println!(
        "    {:>4} {:>4} {:>8} {:>8} {:>8}",
        "Arm", "Type", "Visits", "Q-Value", "Relevance"
    );
    for arm in 0..num_arms {
        let profile_label = match profiles[arm] {
            ArmProfile::High => "HIGH",
            ArmProfile::Medium => "MED ",
            ArmProfile::Low => "LOW ",
        };
        println!(
            "    {arm:>4} {profile_label:>4} {:>8} {:>8.4} {:>8.4}",
            visits[arm], q_values[arm], relevances[arm]
        );
    }

    println!();
    if low_rel_pct >= 80.0 {
        println!(
            "  ✅ GATE T2 PASSED: ≥80% arms already near-zero relevance — soft blending already masks"
        );
    } else {
        println!(
            "  ⚠️  GATE T2: Only {low_rel_pct:.1}% arms near-zero — hard cutoff MAY add signal"
        );
    }
}

// ── T5: A/B Benchmark — Dual Cutoff Comparison ───────────────────

/// Compare dual_cutoff=0.0 (baseline) vs 0.2 vs 0.5.
/// Trains with identical reward landscape, then measures how cutoff
/// changes the number of active arms and total relevance mass.
#[cfg(feature = "bandit")]
#[test]
fn test_bench_dual_cutoff_vs_baseline() {
    let num_arms = 27;
    let episodes = 1000;
    let seed: u64 = 42;
    let profiles = make_profiles(num_arms);

    // Pre-generate training sequence (arm, reward) for reproducibility
    let training_data: Vec<(usize, f32)> = {
        let mut rng = Rng::new(seed);
        (0..episodes)
            .map(|_| {
                let arm = (rng.next() as usize) % num_arms;
                let reward = sample_reward(profiles[arm], &mut rng);
                (arm, reward)
            })
            .collect()
    };

    let cutoff_configs: Vec<(&str, f32)> = vec![
        ("baseline (0.0)", 0.0),
        ("cutoff=0.2", 0.2),
        ("cutoff=0.5", 0.5),
    ];

    println!("\n=== T5: A/B Benchmark — Dual Cutoff Comparison ===");
    println!("  Arms: {num_arms}, Episodes: {episodes}, Strategy: UCB1");
    println!();

    println!(
        "  {:20} | {:>7} | {:>7} | {:>8} | {:>9} | {:>8}",
        "Config", "Active", "Masked", "Active%", "Rel Mass", "Avg Rel"
    );
    println!(
        "  {}-+-{}-+-{}-+-{}-+-{}-+-{}",
        "-".repeat(20),
        "-".repeat(7),
        "-".repeat(7),
        "-".repeat(8),
        "-".repeat(9),
        "-".repeat(8)
    );

    let mut results: Vec<(f32, usize, usize, f64, f32, f32)> = Vec::new();

    for (name, cutoff) in &cutoff_configs {
        let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
        pruner.set_dual_cutoff(*cutoff);

        // Train with identical data
        for &(arm, reward) in &training_data {
            pruner.update(arm, reward);
        }

        pruner.prepare_episode(&mut Rng::new(seed + 1));

        let relevances: Vec<f32> = (0..num_arms)
            .map(|arm| pruner.relevance(0, arm, &[]))
            .collect();

        let active = relevances.iter().filter(|&&r| r > 0.0).count();
        let masked = num_arms - active;
        let active_pct = (active as f64 / num_arms as f64) * 100.0;
        let rel_mass: f32 = relevances.iter().sum();
        let avg_rel = rel_mass / num_arms as f32;

        results.push((*cutoff, active, masked, active_pct, rel_mass, avg_rel));

        println!(
            "  {:20} | {:>7} | {:>7} | {:>7.1}% | {:>9.4} | {:>8.4}",
            name, active, masked, active_pct, rel_mass, avg_rel
        );
    }

    // Delta analysis
    if results.len() >= 2 {
        let baseline = &results[0];
        println!();
        println!("  === Delta from Baseline ===");
        for r in &results[1..] {
            let active_delta = r.1 as isize - baseline.1 as isize;
            let mass_delta = r.4 - baseline.4;
            let mass_pct = if baseline.4 > 0.0 {
                (mass_delta / baseline.4) * 100.0
            } else {
                0.0
            };
            println!(
                "  cutoff={:.1}: active arms {active_delta:+}, relevance mass {mass_delta:+.4} ({mass_pct:+.1}%)",
                r.0
            );
        }
    }

    // Strategy comparison with cutoff=0.2
    println!();
    println!("  === Strategy Comparison (cutoff=0.2) ===");
    println!(
        "  {:20} | {:>7} | {:>7} | {:>9} | {:>8}",
        "Strategy", "Active", "Masked", "Rel Mass", "Avg Rel"
    );
    println!(
        "  {}-+-{}-+-{}-+-{}-+-{}",
        "-".repeat(20),
        "-".repeat(7),
        "-".repeat(7),
        "-".repeat(9),
        "-".repeat(8)
    );

    let strategies: Vec<(&str, BanditStrategy)> = vec![
        ("UCB1", BanditStrategy::Ucb1),
        ("Thompson", BanditStrategy::ThompsonSampling),
        (
            "ε-greedy(0.3)",
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.3,
                decay: 0.995,
            },
        ),
    ];

    for (name, strategy) in strategies {
        let mut pruner = BanditPruner::new(NoScreeningPruner, strategy, num_arms);
        pruner.set_dual_cutoff(0.2);

        for &(arm, reward) in &training_data {
            pruner.update(arm, reward);
        }

        pruner.prepare_episode(&mut Rng::new(seed + 1));

        let relevances: Vec<f32> = (0..num_arms)
            .map(|arm| pruner.relevance(0, arm, &[]))
            .collect();

        let active = relevances.iter().filter(|&&r| r > 0.0).count();
        let masked = num_arms - active;
        let rel_mass: f32 = relevances.iter().sum();
        let avg_rel = rel_mass / num_arms as f32;

        println!(
            "  {:20} | {:>7} | {:>7} | {:>9.4} | {:>8.4}",
            name, active, masked, rel_mass, avg_rel
        );
    }
}
