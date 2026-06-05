//! Plan 191 T1.5: GOAT proof that partial scoring converges faster than binary scoring.
//!
//! Simulates a 4-arm bandit where each arm represents a game strategy with different
//! underlying win rates and partial score distributions. The key insight:
//!
//! With binary scoring, arms 0 (win_rate=0.80) and 1 (win_rate=0.60) look similar
//! (both often win). With partial scoring, arm 0 is clearly better because it has
//! higher partial scores even in "loss" episodes (survives longer, more kills).
//!
//! We use Thompson Sampling (stochastic arm selection) so both paths explore
//! differently. Partial scoring's richer signal helps Thompson's posterior
//! concentrate faster on the optimal arm.
//!
//! ```sh
//! cargo test --features "partial_scoring" --test partial_scoring_goat -- --nocapture
//! ```

#![cfg(feature = "partial_scoring")]

use katgpt_core::{GameTrace, PartialScorer};
use katgpt_rs::pruners::bandit::{BanditPruner, BanditStrategy};
use katgpt_rs::pruners::partial_scorer::BomberPartialScorer;
use katgpt_rs::speculative::types::NoScreeningPruner;
use katgpt_rs::types::Rng;

const N_ARMS: usize = 4;
const N_EPISODES: usize = 500;
const MAX_TICKS: u32 = 200;
/// Number of independent trials to average over (reduces RNG variance).
const N_TRIALS: usize = 20;

/// Arm profiles: (survival_ticks_mean, kills_mean, win_rate)
///
/// Arm 0 (best): survives 180/200 ticks, ~3 kills, wins 80%
/// Arm 1 (good): survives 120/200 ticks, ~1.5 kills, wins 60%
/// Arm 2 (ok):   survives 60/200 ticks,  ~0.5 kills, wins 40%
/// Arm 3 (worst): survives 30/200 ticks, ~0.1 kills, wins 20%
const ARM_PROFILES: [(f32, f32, f32); N_ARMS] = [
    (180.0, 3.0, 0.80),
    (120.0, 1.5, 0.60),
    (60.0, 0.5, 0.40),
    (30.0, 0.1, 0.20),
];

/// Simulate a game episode for the given arm.
///
/// Generates a `GameTrace` with arm-correlated survival, kills, and win outcome.
/// Noise is added so the bandit must learn from repeated sampling.
fn simulate_episode(arm: usize, rng: &mut Rng) -> GameTrace {
    let (survival_mean, kills_mean, win_rate) = ARM_PROFILES[arm];

    // Noisy survival: base + uniform noise [-10, +10]
    let noise = (rng.uniform() - 0.5) * 20.0;
    let survival_ticks = ((survival_mean + noise).clamp(1.0, MAX_TICKS as f32)) as u32;

    // Noisy kills: sometimes spike, mostly proportional
    let kills = if rng.uniform() < kills_mean / 4.0 {
        (kills_mean + rng.uniform() * 2.0) as u32
    } else {
        (kills_mean * rng.uniform()) as u32
    };

    // Win/loss outcome based on arm's win rate
    let final_reward = if (rng.uniform() as f64) < win_rate as f64 {
        1.0
    } else {
        0.0
    };

    let actions_taken = (survival_ticks / 4).max(1);

    GameTrace {
        survival_ticks,
        kills,
        actions_taken,
        max_ticks: MAX_TICKS,
        final_reward,
    }
}

/// Thompson Sampling arm selection using Beta posterior.
///
/// α = Q·n + 1, β = (1-Q)·n + 1 (Laplace smoothing).
/// Unvisited arms get uniform sample.
fn select_arm_thompson(visits: &[u32], q_values: &[f32], rng: &mut Rng) -> usize {
    let mut best_arm = 0;
    let mut best_sample = f32::NEG_INFINITY;

    for i in 0..visits.len() {
        let sample = if visits[i] == 0 {
            rng.uniform()
        } else {
            let n = visits[i] as f32;
            let q = q_values[i].clamp(0.0, 1.0);
            let alpha = q * n + 1.0;
            let beta = (1.0 - q) * n + 1.0;
            sample_beta(alpha, beta, rng)
        };

        if sample > best_sample {
            best_sample = sample;
            best_arm = i;
        }
    }

    best_arm
}

/// Beta distribution sampling via Johnk's algorithm (matches bandit.rs).
fn sample_beta(alpha: f32, beta: f32, rng: &mut Rng) -> f32 {
    if alpha <= 1.0 && beta <= 1.0 {
        // Johnk's algorithm for α,β ∈ (0,1]
        for _ in 0..100 {
            let u = rng.uniform().powf(1.0 / alpha);
            let v = rng.uniform().powf(1.0 / beta);
            if u + v <= 1.0 {
                return if u + v > 0.0 { u / (u + v) } else { 0.5 };
            }
        }
        return 0.5;
    }

    // Normal approximation for large α,β
    let mean = alpha / (alpha + beta);
    let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
    let std_dev = variance.sqrt().max(1e-6);
    // Box-Muller via two uniform samples
    let u1 = rng.uniform().max(1e-10);
    let u2 = rng.uniform();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    (mean + std_dev * z).clamp(0.0, 1.0)
}

/// Result of a single bandit experiment run.
struct ExperimentResult {
    /// Episode where convergence was first achieved (None = never converged).
    convergence_episode: Option<usize>,
    /// Q-values at end of experiment.
    q_values: Vec<f32>,
    /// Visit counts at end of experiment.
    #[allow(dead_code)]
    visits: Vec<u32>,
    /// Total cumulative reward across all episodes.
    cumulative_reward: f32,
}

/// Run one full bandit experiment (binary or partial scoring path).
///
/// Uses Thompson Sampling for stochastic arm selection, enabling meaningful
/// convergence comparison between binary and partial reward signals.
fn run_experiment(partial: bool, scorer: &BomberPartialScorer, seed: u64) -> ExperimentResult {
    let mut rng = Rng::new(seed);
    let mut visits = vec![0u32; N_ARMS];
    let mut q_values = vec![0.0f32; N_ARMS];
    let mut cumulative_reward = 0.0f32;
    let mut convergence_ep = None;

    for ep in 0..N_EPISODES {
        let arm = select_arm_thompson(&visits, &q_values, &mut rng);
        let trace = simulate_episode(arm, &mut rng);

        let reward = if partial {
            scorer.partial_score(&trace)
        } else if trace.final_reward > 0.0 {
            1.0
        } else {
            0.0
        };

        // Incremental mean update (matches BanditStats::update)
        visits[arm] += 1;
        let n = visits[arm] as f32;
        q_values[arm] += (reward - q_values[arm]) / n;

        cumulative_reward += reward;

        // Convergence: arm 0 has >50% of total visits, highest Q-value, and
        // Q[0] is at least 10% above Q[1] (clear separation, not marginal).
        // Only check after episode 40 (enough data for meaningful comparison).
        if convergence_ep.is_none() && ep >= 40 {
            let total_v: u32 = visits.iter().sum();
            if total_v > 20 {
                let arm0_frac = visits[0] as f32 / total_v as f32;
                let best_q_arm = q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if arm0_frac > 0.50 && best_q_arm == 0 {
                    // Check clear separation from second-best arm
                    let second_best_q = q_values
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != 0)
                        .map(|(_, q)| *q)
                        .fold(f32::NEG_INFINITY, f32::max);
                    if q_values[0] > second_best_q * 1.1 {
                        convergence_ep = Some(ep);
                    }
                }
            }
        }
    }

    ExperimentResult {
        convergence_episode: convergence_ep,
        q_values,
        visits,
        cumulative_reward,
    }
}

#[test]
fn test_partial_scoring_goat_convergence() {
    let scorer = BomberPartialScorer {
        max_ticks: MAX_TICKS,
    };

    // Run N_TRIALS with different seeds for statistical robustness
    let mut binary_convergence_sum = 0usize;
    let mut partial_convergence_sum = 0usize;
    let mut binary_converged_count = 0usize;
    let mut partial_converged_count = 0usize;
    let mut binary_reward_sum = 0.0f32;
    let mut partial_reward_sum = 0.0f32;
    let mut binary_best_correct = 0usize;
    let mut partial_best_correct = 0usize;

    for trial in 0..N_TRIALS {
        let seed = 42 + trial as u64;

        let binary = run_experiment(false, &scorer, seed);
        let partial = run_experiment(true, &scorer, seed);

        if let Some(ep) = binary.convergence_episode {
            binary_convergence_sum += ep;
            binary_converged_count += 1;
        }
        if let Some(ep) = partial.convergence_episode {
            partial_convergence_sum += ep;
            partial_converged_count += 1;
        }

        binary_reward_sum += binary.cumulative_reward;
        partial_reward_sum += partial.cumulative_reward;

        // Check if arm 0 is best
        let binary_best = binary
            .q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let partial_best = partial
            .q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if binary_best == 0 {
            binary_best_correct += 1;
        }
        if partial_best == 0 {
            partial_best_correct += 1;
        }
    }

    let binary_avg_conv = if binary_converged_count > 0 {
        binary_convergence_sum as f64 / binary_converged_count as f64
    } else {
        f64::INFINITY
    };
    let partial_avg_conv = if partial_converged_count > 0 {
        partial_convergence_sum as f64 / partial_converged_count as f64
    } else {
        f64::INFINITY
    };

    // ── Diagnostics ────────────────────────────────────────────
    println!("\n{}", "═".repeat(70));
    println!("Plan 191 T1.5 — GOAT Proof: Partial Scoring Convergence");
    println!("   {N_TRIALS} trials × {N_EPISODES} episodes × {N_ARMS} arms (Thompson Sampling)");
    println!("{}", "═".repeat(70));

    println!("\n── Binary Scoring Path ──────────────────────────────────────");
    println!("   Converged: {binary_converged_count}/{N_TRIALS} trials");
    println!("   Avg convergence episode: {binary_avg_conv:.1}");
    println!("   Best arm correct: {binary_best_correct}/{N_TRIALS}");
    println!(
        "   Avg cumulative reward:   {binary_reward_sum:.1} ({:.2}/trial)",
        binary_reward_sum / N_TRIALS as f32
    );

    println!("\n── Partial Scoring Path ─────────────────────────────────────");
    println!("   Converged: {partial_converged_count}/{N_TRIALS} trials");
    println!("   Avg convergence episode: {partial_avg_conv:.1}");
    println!("   Best arm correct: {partial_best_correct}/{N_TRIALS}");
    println!(
        "   Avg cumulative reward:   {partial_reward_sum:.1} ({:.2}/trial)",
        partial_reward_sum / N_TRIALS as f32
    );

    if partial_converged_count > 0 && binary_converged_count > 0 {
        let speedup = (binary_avg_conv - partial_avg_conv) / binary_avg_conv * 100.0;
        println!("\n   Speedup: partial converges {speedup:.1}% faster than binary");
    } else if partial_converged_count > 0 && binary_converged_count == 0 {
        println!("\n   Speedup: partial converged, binary did not (strong GOAT)");
    }

    println!("{}", "═".repeat(70));

    // ── Assertions ─────────────────────────────────────────────

    // 1. Partial converges in fewer episodes than binary (on average)
    //    OR partial converges more often than binary
    let partial_converges_faster = partial_converged_count > binary_converged_count
        || (partial_converged_count > 0
            && binary_converged_count > 0
            && partial_avg_conv <= binary_avg_conv);
    assert!(
        partial_converges_faster,
        "GOAT FAIL: partial does not converge faster. \
         partial={partial_avg_conv:.1}ep ({partial_converged_count}/{N_TRIALS}), \
         binary={binary_avg_conv:.1}ep ({binary_converged_count}/{N_TRIALS})"
    );

    // 2. Both paths should identify arm 0 as best arm in ≥80% of trials
    let min_correct = (N_TRIALS * 8) / 10;
    assert!(
        binary_best_correct >= min_correct,
        "Binary path should identify arm 0 as best in ≥{min_correct}/{N_TRIALS} trials, got {binary_best_correct}"
    );
    assert!(
        partial_best_correct >= min_correct,
        "Partial path should identify arm 0 as best in ≥{min_correct}/{N_TRIALS} trials, got {partial_best_correct}"
    );

    // 3. Partial path has higher average cumulative reward
    let avg_binary_reward = binary_reward_sum / N_TRIALS as f32;
    let avg_partial_reward = partial_reward_sum / N_TRIALS as f32;
    assert!(
        avg_partial_reward > avg_binary_reward,
        "Partial avg reward ({avg_partial_reward:.2}) should exceed binary ({avg_binary_reward:.2})"
    );

    println!(
        "\n✅ GOAT Proof PASSED: partial scoring converges faster, identifies optimal arm, and accumulates more reward."
    );
}

/// Validate that BanditPruner.update_with_trace() integration works end-to-end.
#[test]
fn test_bandit_pruner_partial_scoring_integration() {
    let scorer = BomberPartialScorer {
        max_ticks: MAX_TICKS,
    };
    let mut rng = Rng::new(123);

    // Create bandit with partial scorer
    let mut bandit = BanditPruner::with_partial_scorer(
        NoScreeningPruner,
        BanditStrategy::ThompsonSampling,
        N_ARMS,
        Box::new(scorer),
    );

    // Run episodes using update_with_trace + prepare_episode (Thompson Sampling)
    for _ in 0..N_EPISODES {
        bandit.prepare_episode(&mut rng);

        // Use internal stats for Thompson selection
        let arm = {
            let visits = bandit.visits().to_vec();
            let q_values = bandit.q_values().to_vec();
            select_arm_thompson(&visits, &q_values, &mut rng)
        };

        let trace = simulate_episode(arm, &mut rng);
        bandit.update_with_trace(arm, &trace);
    }

    // Verify arm 0 is best
    let best = bandit.best_arm();
    assert_eq!(
        best, 0,
        "BanditPruner with partial scorer should identify arm 0 as best, got {best}"
    );

    // Verify Q-value ordering: Q(0) > Q(1) > Q(2) > Q(3)
    let q = bandit.q_values();
    println!("\n── BanditPruner Integration Q-values ──");
    for i in 0..N_ARMS {
        println!("   Arm {i}: Q={:.4}, visits={}", q[i], bandit.visits()[i]);
    }

    assert!(q[0] > q[1], "Q[0] ({}) should > Q[1] ({})", q[0], q[1]);
    assert!(q[1] > q[2], "Q[1] ({}) should > Q[2] ({})", q[1], q[2]);
    assert!(q[2] > q[3], "Q[2] ({}) should > Q[3] ({})", q[2], q[3]);

    println!("\n✅ BanditPruner integration test PASSED.");
}

// TL;DR: GOAT proof — partial scoring converges faster than binary scoring in 4-arm bandit simulation.
// Uses Thompson Sampling (stochastic selection) across 20 independent trials for statistical
// robustness. Partial scoring provides richer gradient signal (survival + kills + efficiency),
// enabling faster discrimination between arms with similar win rates but different gameplay quality.
