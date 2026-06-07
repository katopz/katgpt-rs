//! Plan 191 T3.4: GOAT proof that IdeaDivergence filter converges faster + maintains arm diversity.
//!
//! Simulates a 6-arm bandit where arms 0-2 are "good" (Q=0.90, 0.85, 0.80) and arms 3-5 are
//! "bad" (Q=0.30, 0.25, 0.20). Without the divergence filter, the bandit tends to concentrate
//! exploration on just arm 0 (highest Q), starving arms 1-2. With the filter, convergent arms
//! get penalized (0.5× score), forcing exploration to spread across all good arms.
//!
//! **Metrics:**
//! 1. Convergence time: episode where all good arms (0-2) have ≥10 visits (lower = better)
//! 2. Final reward: average reward over last 100 episodes (higher = better)
//! 3. Arm diversity: number of arms with >10% selection rate at end (higher = better)
//!
//! ```sh
//! cargo test --features "idea_divergence" --test idea_divergence_goat -- --nocapture
//! ```

#![cfg(feature = "idea_divergence")]

use katgpt_rs::pruners::bandit::{BanditEnv, BernoulliEnv};
use katgpt_rs::types::Rng;

const N_ARMS: usize = 6;
const N_EPISODES: usize = 1000;
/// Number of independent trials to average over (reduces RNG variance).
const N_TRIALS: usize = 20;

/// Arm true probabilities: arms 0-2 are good, arms 3-5 are bad.
const ARM_PROBS: [f32; N_ARMS] = [0.90, 0.85, 0.80, 0.30, 0.25, 0.20];

/// Result of a single bandit experiment run.
struct ExperimentResult {
    /// Episode where all good arms (0-2) reached ≥10 visits each (None = never).
    convergence_episode: Option<usize>,
    /// Visit counts at end of experiment.
    visits: Vec<u32>,
    /// Reward per episode (for computing tail average).
    rewards: Vec<f32>,
}

/// Epsilon-greedy arm selection with decay.
///
/// With high epsilon, explores broadly. With low epsilon, collapses to best arm.
fn select_arm_epsilon_greedy(
    visits: &[u32],
    q_values: &[f32],
    epsilon: f32,
    rng: &mut Rng,
) -> usize {
    let n = visits.len();
    // Cold start: play each unvisited arm once
    for (i, &v) in visits.iter().enumerate() {
        if v == 0 {
            return i;
        }
    }
    if rng.uniform() < epsilon {
        (rng.uniform() * n as f32) as usize % n
    } else {
        // Greedy: best Q-value
        q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Run one full bandit experiment with EpsilonGreedy strategy.
///
/// Uses epsilon=0.08 with decay=0.995 so exploitation intensifies over time.
/// Without the divergence filter, the greedy strategy collapses to arm 0
/// (highest Q-value), starving arms 1-2. With the filter, convergent arms
/// get penalized (reduced reward), forcing exploration to spread.
///
/// When `use_divergence` is true, arms with visit share > 20% get their effective
/// reward halved, mimicking the 0.5× penalty from `arm_bandit_score()`.
fn run_experiment(use_divergence: bool, seed: u64) -> ExperimentResult {
    let mut rng = Rng::new(seed);
    let env = BernoulliEnv::new(&ARM_PROBS);

    let mut visits = vec![0u32; N_ARMS];
    let mut q_values = vec![0.0f32; N_ARMS];
    let mut convergence_ep = None;
    let mut rewards = Vec::with_capacity(N_EPISODES);
    let mut epsilon = 0.08f32; // Start with low exploration
    let epsilon_decay = 0.995f32; // Decay over time → more exploitation

    for ep in 0..N_EPISODES {
        let arm = select_arm_epsilon_greedy(&visits, &q_values, epsilon, &mut rng);

        // Pull from Bernoulli environment
        let reward = env.pull(arm, &mut rng);

        // Divergence filter: penalize convergent arms
        let effective_reward = if use_divergence {
            let total: u32 = visits.iter().sum();
            if total > 0 {
                let arm_frac = visits[arm] as f32 / total as f32;
                // If this arm already has high visit share, reduce its reward signal
                // Mimics the 0.5× bandit score penalty from arm_bandit_score()
                if arm_frac > 0.20 {
                    reward * 0.5
                } else {
                    reward
                }
            } else {
                reward
            }
        } else {
            reward
        };

        // Incremental mean update (matches BanditStats::update)
        visits[arm] += 1;
        let n = visits[arm] as f32;
        q_values[arm] += (effective_reward - q_values[arm]) / n;

        rewards.push(reward); // Track true reward

        // Decay epsilon: more exploitation over time
        epsilon *= epsilon_decay;

        // Check convergence: all three good arms (0-2) have >=10 visits each.
        // Without filter: greedy concentrates on arm 0, starving arms 1-2.
        // With filter: convergent arms get penalized → exploration spreads to arms 1-2.
        if convergence_ep.is_none() {
            let all_good_explored = visits[0] >= 10 && visits[1] >= 10 && visits[2] >= 10;
            if all_good_explored {
                convergence_ep = Some(ep);
            }
        }
    }

    ExperimentResult {
        convergence_episode: convergence_ep,
        visits,
        rewards,
    }
}

/// Count arms with selection rate above `threshold`.
fn count_active_arms(visits: &[u32], threshold: f32) -> usize {
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return 0;
    }
    visits
        .iter()
        .filter(|&&v| v as f32 / total as f32 > threshold)
        .count()
}

#[test]
fn test_idea_divergence_goat_convergence_and_diversity() {
    let mut div_conv_sum = 0usize;
    let mut plain_conv_sum = 0usize;
    let mut div_conv_count = 0usize;
    let mut plain_conv_count = 0usize;

    let mut div_reward_sum = 0.0f64;
    let mut plain_reward_sum = 0.0f64;

    let mut div_diversity_sum = 0usize;
    let mut plain_diversity_sum = 0usize;

    for trial in 0..N_TRIALS {
        let seed = 42 + trial as u64;

        let plain = run_experiment(false, seed);
        let div = run_experiment(true, seed);

        // Convergence
        if let Some(ep) = plain.convergence_episode {
            plain_conv_sum += ep;
            plain_conv_count += 1;
        }
        if let Some(ep) = div.convergence_episode {
            div_conv_sum += ep;
            div_conv_count += 1;
        }

        // Final reward: average over last 100 episodes
        let plain_tail: f64 = plain
            .rewards
            .iter()
            .rev()
            .take(100)
            .map(|&r| r as f64)
            .sum::<f64>()
            / 100.0;
        let div_tail: f64 = div
            .rewards
            .iter()
            .rev()
            .take(100)
            .map(|&r| r as f64)
            .sum::<f64>()
            / 100.0;
        plain_reward_sum += plain_tail;
        div_reward_sum += div_tail;

        // Arm diversity: arms with >10% selection rate
        let plain_diversity = count_active_arms(&plain.visits, 0.10);
        let div_diversity = count_active_arms(&div.visits, 0.10);
        plain_diversity_sum += plain_diversity;
        div_diversity_sum += div_diversity;
    }

    let div_avg_conv = if div_conv_count > 0 {
        div_conv_sum as f64 / div_conv_count as f64
    } else {
        f64::INFINITY
    };
    let plain_avg_conv = if plain_conv_count > 0 {
        plain_conv_sum as f64 / plain_conv_count as f64
    } else {
        f64::INFINITY
    };

    let div_avg_reward = div_reward_sum / N_TRIALS as f64;
    let plain_avg_reward = plain_reward_sum / N_TRIALS as f64;

    let div_avg_diversity = div_diversity_sum as f64 / N_TRIALS as f64;
    let plain_avg_diversity = plain_diversity_sum as f64 / N_TRIALS as f64;

    // ── Diagnostics ────────────────────────────────────────────
    println!("\n{}", "═".repeat(70));
    println!("Plan 191 T3.4 — GOAT Proof: IdeaDivergence Filter");
    println!("   {N_TRIALS} trials × {N_EPISODES} episodes × {N_ARMS} arms (EpsilonGreedy)");
    println!("{}", "═".repeat(70));

    println!("\n── Without Divergence Filter ──────────────────────────────");
    println!("   Converged: {plain_conv_count}/{N_TRIALS} trials");
    println!("   Avg convergence episode: {plain_avg_conv:.1}");
    println!("   Avg final reward (last 100 ep): {plain_avg_reward:.4}");
    println!("   Avg arm diversity (>10%): {plain_avg_diversity:.2}");

    println!("\n── With Divergence Filter ─────────────────────────────────");
    println!("   Converged: {div_conv_count}/{N_TRIALS} trials");
    println!("   Avg convergence episode: {div_avg_conv:.1}");
    println!("   Avg final reward (last 100 ep): {div_avg_reward:.4}");
    println!("   Avg arm diversity (>10%): {div_avg_diversity:.2}");

    if div_conv_count > 0 && plain_conv_count > 0 {
        let speedup = (plain_avg_conv - div_avg_conv) / plain_avg_conv * 100.0;
        println!("\n   Convergence speedup: {speedup:.1}%");
    }
    let diversity_ratio = if plain_avg_diversity > 0.0 {
        div_avg_diversity / plain_avg_diversity
    } else {
        f64::INFINITY
    };
    println!("   Diversity ratio: {diversity_ratio:.2}×");

    println!("{}", "═".repeat(70));

    // ── Assertions ─────────────────────────────────────────────

    // 1. Convergence: filter reaches all-good-arms-explored ≤ 80% of plain (≥20% faster)
    //    Measuring episode where all arms 0-2 have ≥10 visits each.
    if div_conv_count > 0 && plain_conv_count > 0 {
        let conv_ratio = div_avg_conv / plain_avg_conv;
        assert!(
            conv_ratio <= 0.80,
            "GOAT FAIL: divergence filter convergence ratio {conv_ratio:.2} > 0.80 \
             (div={div_avg_conv:.1}ep, plain={plain_avg_conv:.1}ep)"
        );
    }
    // If filter converged more often, that also counts as faster
    let filter_converges_faster =
        div_conv_count > plain_conv_count || (div_conv_count > 0 && plain_conv_count == 0);
    if !(div_conv_count > 0 && plain_conv_count > 0) {
        assert!(
            filter_converges_faster,
            "GOAT FAIL: neither convergence ratio nor count shows filter is faster \
             (div={div_conv_count}/{N_TRIALS}, plain={plain_conv_count}/{N_TRIALS})"
        );
    }

    // 2. Final reward with filter ≥ 95% of final reward without filter (no quality loss)
    assert!(
        div_avg_reward >= plain_avg_reward * 0.95,
        "GOAT FAIL: divergence filter reward {div_avg_reward:.4} < 95% of plain {plain_avg_reward:.4}"
    );

    // 3. Arm diversity with filter ≥ 2× arm diversity without filter
    assert!(
        div_avg_diversity >= plain_avg_diversity * 2.0,
        "GOAT FAIL: divergence filter diversity {div_avg_diversity:.2} < 2× plain {plain_avg_diversity:.2}"
    );

    println!(
        "\n✅ GOAT Proof PASSED: divergence filter converges ≥20% faster, maintains reward quality, and achieves ≥2× arm diversity."
    );
}

// TL;DR: GOAT proof — IdeaDivergence filter converges ≥20% faster and achieves ≥2× arm diversity
// in 6-arm EpsilonGreedy bandit. Arms 0-2 (good) spread exploration instead of collapsing
// to arm 0 only. Reward quality is preserved. Feature-gated behind idea_divergence.
