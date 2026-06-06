//! Idea Divergence Demo — Collapse Prevention (Plan 191 T4.3)
//!
//! Demonstrates how the `IdeaDivergence` filter prevents a bandit from collapsing
//! onto a single dominant arm, maintaining strategic diversity.
//!
//! Scenario: 10 arms with distinct strategies (different Bernoulli probs).
//! Two bandit sessions compared:
//! - **Without filter**: Greedy exploitation — top arm dominates
//! - **With filter**: Epsilon-greedy + `IdeaDivergence` — more even arm distribution
//!
//! The divergence filter uses a score vector `[q_value, visit_density]` per arm.
//! Visit density = visits / total_pulls captures how "over-explored" an arm is.
//! Arms too similar to others get reduced selection probability.
//!
//! Run: `cargo run --features "idea_divergence" --example idea_divergence_demo`

#[cfg(feature = "idea_divergence")]
use katgpt_rs::pruners::{BanditPruner, BanditStats, BanditStrategy, IdeaDivergence};
#[cfg(feature = "idea_divergence")]
use katgpt_rs::speculative::{NoScreeningPruner, ScreeningPruner};
#[cfg(feature = "idea_divergence")]
use katgpt_rs::types::Rng;

#[cfg(feature = "idea_divergence")]
const ARMS: usize = 10;
#[cfg(feature = "idea_divergence")]
const EPISODES: usize = 300;
#[cfg(feature = "idea_divergence")]
const SEED: u64 = 42;
/// Novelty threshold — arms within this L2 distance in score-space are "too similar".
#[cfg(feature = "idea_divergence")]
const DIVERGENCE_THRESHOLD: f32 = 0.3;
/// Exploration rate for the *without* session (low → collapse-prone).
#[cfg(feature = "idea_divergence")]
const EPSILON_GREEDY: f32 = 0.05;
/// Exploration rate for the *with* session (same base + divergence).
#[cfg(feature = "idea_divergence")]
const EPSILON_FILTERED: f32 = 0.05;

// ── Helpers ─────────────────────────────────────────────────────

/// Generate true Q-values: [0.90, 0.88, 0.86, ..., 0.72]
#[cfg(feature = "idea_divergence")]
fn arm_probs() -> Vec<f32> {
    (0..ARMS).map(|i| 0.90 - i as f32 * 0.02).collect()
}

/// Simulate a Bernoulli pull: reward 1.0 with probability `p`, else 0.0.
#[cfg(feature = "idea_divergence")]
fn bernoulli(p: f32, rng: &mut Rng) -> f32 {
    if rng.uniform() < p { 1.0 } else { 0.0 }
}

/// Epsilon-greedy arm selection from BanditStats (baseline, no filter).
#[cfg(feature = "idea_divergence")]
fn select_epsilon_greedy(stats: &BanditStats, rng: &mut Rng) -> usize {
    if rng.uniform() < EPSILON_GREEDY {
        (rng.uniform() * ARMS as f32) as usize % ARMS
    } else {
        stats.best_arm()
    }
}

/// Epsilon-greedy with divergence-based re-weighting.
///
/// After the cold-start round, arms whose score vector `[q, visit_density]`
/// is too close to other arms get a probability penalty, redistributing
/// exploration toward under-explored regions.
#[cfg(feature = "idea_divergence")]
fn select_epsilon_greedy_with_divergence(
    stats: &BanditStats,
    arm_scores: &[Vec<f32>],
    rng: &mut Rng,
) -> usize {
    if rng.uniform() < EPSILON_FILTERED {
        return (rng.uniform() * ARMS as f32) as usize % ARMS;
    }

    // Exploit with divergence re-weighting
    let mut best_arm = 0;
    let mut best_score = f32::NEG_INFINITY;
    for arm in 0..ARMS {
        let q = stats.q_value(arm);
        let visits = stats.visit_count(arm) as f32;
        let total = stats.total_pulls() as f32;
        let visit_density = if total > 0.0 { visits / total } else { 0.0 };
        let score_vec = [q, visit_density];

        // Compute min distance to other arms
        let mut min_dist = f32::MAX;
        for (other_i, other) in arm_scores.iter().enumerate() {
            if other_i != arm && !other.is_empty() {
                let d = IdeaDivergence::divergence(&score_vec, other);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }

        // Proportional bonus: diverse arms get boosted, similar arms get penalized
        let diversity_factor = if min_dist == f32::MAX {
            1.0
        } else if min_dist < DIVERGENCE_THRESHOLD {
            // Too similar: penalize by scaling factor
            0.3 + 0.7 * (min_dist / DIVERGENCE_THRESHOLD)
        } else {
            1.0
        };

        // Boost arms with fewer visits (anti-collapse)
        let exploration_bonus = if total > 0.0 && visits > 0.0 {
            1.0 + 2.0 * (1.0 - visit_density)
        } else {
            1.0
        };

        let adjusted = q * diversity_factor * exploration_bonus;

        if adjusted > best_score {
            best_score = adjusted;
            best_arm = arm;
        }
    }
    best_arm
}

/// Run a plain epsilon-greedy session (no divergence filter).
#[cfg(feature = "idea_divergence")]
fn run_without_filter(probs: &[f32], seed: u64) -> (Vec<u32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut stats = BanditStats::new(ARMS);

    for _ in 0..EPISODES {
        let arm = select_epsilon_greedy(&stats, &mut rng);
        let reward = bernoulli(probs[arm], &mut rng);
        stats.update(arm, reward);
    }

    let visits: Vec<u32> = (0..ARMS).map(|i| stats.visit_count(i)).collect();
    let q_values = stats.q_values().to_vec();
    (visits, q_values)
}

/// Run epsilon-greedy *with* IdeaDivergence filter.
#[cfg(feature = "idea_divergence")]
fn run_with_filter(probs: &[f32], seed: u64) -> (Vec<u32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut stats = BanditStats::new(ARMS);
    let mut arm_scores: Vec<Vec<f32>> = vec![vec![]; ARMS];

    for _ in 0..EPISODES {
        let arm = select_epsilon_greedy_with_divergence(&stats, &arm_scores, &mut rng);
        let reward = bernoulli(probs[arm], &mut rng);
        stats.update(arm, reward);

        // Update arm score vector
        let q = stats.q_value(arm);
        let visits = stats.visit_count(arm) as f32;
        let total = stats.total_pulls() as f32;
        let visit_density = if total > 0.0 { visits / total } else { 0.0 };
        arm_scores[arm] = vec![q, visit_density];
    }

    let visits: Vec<u32> = (0..ARMS).map(|i| stats.visit_count(i)).collect();
    let q_values = stats.q_values().to_vec();
    (visits, q_values)
}

/// Also run using BanditPruner's integrated divergence.
#[cfg(feature = "idea_divergence")]
fn run_with_bandit_pruner(probs: &[f32], seed: u64) -> (Vec<u32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let mut pruner = BanditPruner::with_idea_divergence(
        NoScreeningPruner,
        BanditStrategy::EpsilonGreedy {
            epsilon: EPSILON_FILTERED,
            decay: 1.0,
        },
        ARMS,
        DIVERGENCE_THRESHOLD,
    );

    for _ in 0..EPISODES {
        pruner.prepare_episode(&mut rng);
        let arm = (0..ARMS)
            .max_by(|&a, &b| {
                let sa = pruner.relevance(0, a, &[]);
                let sb = pruner.relevance(0, b, &[]);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let reward = bernoulli(probs[arm], &mut rng);
        pruner.update(arm, reward);
        pruner.update_divergence(arm);
    }

    let visits = pruner.visits().to_vec();
    let q_values = pruner.q_values().to_vec();
    (visits, q_values)
}

/// Count arms with >10% of total visits.
#[cfg(feature = "idea_divergence")]
fn active_arms(visits: &[u32]) -> usize {
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return 0;
    }
    visits
        .iter()
        .filter(|&&v| v as f32 / total as f32 > 0.10)
        .count()
}

/// Top arm visit count and percentage.
#[cfg(feature = "idea_divergence")]
fn top_arm_stats(visits: &[u32]) -> (u32, f32) {
    let total: u32 = visits.iter().sum();
    let &max = visits.iter().max().unwrap_or(&0);
    let pct = if total > 0 {
        max as f32 / total as f32 * 100.0
    } else {
        0.0
    };
    (max, pct)
}

/// Print a divergence matrix for a subset of arms.
#[cfg(feature = "idea_divergence")]
fn print_divergence_matrix(q_values: &[f32], visits: &[u32], subset: usize) {
    let n = subset.min(q_values.len());
    let total: u32 = visits.iter().sum();
    let score_vecs: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            let vd = if total > 0 {
                visits[i] as f32 / total as f32
            } else {
                0.0
            };
            [q_values[i], vd]
        })
        .collect();

    println!("Divergence Matrix (L2 distance between arm score vectors):");
    println!("  Score vector per arm: [Q-value, visit_density]");

    print!("     ");
    for j in 0..n {
        print!("{:>6}", j);
    }
    println!();

    for i in 0..n {
        print!("{:>4}", i);
        for j in 0..n {
            let d = IdeaDivergence::divergence(&score_vecs[i], &score_vecs[j]);
            print!("{:>6.2}", d);
        }
        println!();
    }
}

/// Print visit distribution bars.
#[cfg(feature = "idea_divergence")]
fn print_visit_distribution(visits: &[u32]) {
    let total: u32 = visits.iter().sum();
    for (i, &v) in visits.iter().enumerate() {
        let pct = if total > 0 {
            v as f32 / total as f32 * 100.0
        } else {
            0.0
        };
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    Arm {i:>2}: {v:>4} ({pct:>5.1}%) {bar}");
    }
}

// ── Main ────────────────────────────────────────────────────────

#[cfg(feature = "idea_divergence")]
fn main() {
    let probs = arm_probs();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Idea Divergence Demo — Collapse Prevention          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Print arm configuration
    println!("Arm configuration (Bernoulli probabilities):");
    for (i, &p) in probs.iter().enumerate() {
        let bar_len = (p * 50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  Arm {i:>2}: {p:.2} {bar}");
    }
    println!();

    // Run sessions
    println!("Running {EPISODES} episodes per session...");
    let (visits_without, _q_without) = run_without_filter(&probs, SEED);
    let (visits_with, q_with) = run_with_filter(&probs, SEED);
    let (visits_pruner, _q_pruner) = run_with_bandit_pruner(&probs, SEED);
    println!("Done.");
    println!();

    // Print divergence matrix (from the filtered run, first 5 arms)
    println!("─────────────────────────────────────────────────────────────");
    print_divergence_matrix(&q_with, &visits_with, 5);
    println!("─────────────────────────────────────────────────────────────");
    println!();

    // Comparison
    let active_without = active_arms(&visits_without);
    let active_with = active_arms(&visits_with);
    let (top_visits_no, top_pct_no) = top_arm_stats(&visits_without);
    let (top_visits_yes, top_pct_yes) = top_arm_stats(&visits_with);

    println!("=== Without Divergence Filter ===");
    println!("  Active arms (>10% visits): {active_without}");
    println!("  Top arm visits: {top_visits_no} ({top_pct_no:.1}%)");
    println!();
    println!("  Visit distribution:");
    print_visit_distribution(&visits_without);
    println!();

    println!("=== With Divergence Filter (threshold={DIVERGENCE_THRESHOLD}) ===");
    println!("  Active arms (>10% visits): {active_with}");
    println!("  Top arm visits: {top_visits_yes} ({top_pct_yes:.1}%)");
    println!();
    println!("  Visit distribution:");
    print_visit_distribution(&visits_with);
    println!();

    // BanditPruner integration verification
    let active_pruner = active_arms(&visits_pruner);
    let (top_pruner, pct_pruner) = top_arm_stats(&visits_pruner);
    println!("=== BanditPruner Integration (soft-route + divergence) ===");
    println!("  Active arms (>10% visits): {active_pruner}");
    println!("  Top arm visits: {top_pruner} ({pct_pruner:.1}%)");
    println!();

    // Summary
    let ratio = if active_without > 0 {
        active_with as f32 / active_without as f32
    } else {
        0.0
    };
    println!("Divergence filter maintains {ratio:.0}× more active arms.");
    println!();

    // Q-value comparison
    println!("Final Q-values (filtered session):");
    for (i, &q) in q_with.iter().enumerate() {
        let true_p = probs[i];
        let err = (q - true_p).abs();
        println!("  Arm {i:>2}: Q={q:.4} (true={true_p:.2}, err={err:.4})");
    }
}

// TL;DR: Demonstrates IdeaDivergence preventing bandit collapse — 10-arm epsilon-greedy with filter maintains more diverse arm selection vs plain.

#[cfg(not(feature = "idea_divergence"))]
fn main() {
    eprintln!(
        "Enable idea_divergence feature: cargo run --features idea_divergence --example idea_divergence_demo"
    );
}
