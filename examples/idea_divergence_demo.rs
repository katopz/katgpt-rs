//! Idea Divergence Demo — Collapse Prevention (Plan 191 T4.3)
//!
//! Demonstrates how the `IdeaDivergence` filter prevents a bandit from collapsing
//! onto a single dominant arm, maintaining strategic diversity.
//!
//! Two 10-arm Bernoulli bandits with UCB1 are compared:
//! - **Without filter**: Plain bandit — top arm hogs most visits
//! - **With filter**: `IdeaDivergence(threshold=0.3)` — more even arm distribution
//!
//! Run: `cargo run --features "idea_divergence" --example idea_divergence_demo`

#![cfg(feature = "idea_divergence")]

use katgpt_rs::pruners::{BanditPruner, BanditStrategy, IdeaDivergence};
use katgpt_rs::speculative::NoScreeningPruner;
use katgpt_rs::types::Rng;

const ARMS: usize = 10;
const EPISODES: usize = 200;
const SEED: u64 = 42;
const DIVERGENCE_THRESHOLD: f32 = 0.3;

// ── Helpers ─────────────────────────────────────────────────────

/// Generate true Q-values: [0.90, 0.88, 0.86, ..., 0.72]
fn arm_probs() -> Vec<f32> {
    (0..ARMS).map(|i| 0.90 - i as f32 * 0.02).collect()
}

/// Simulate a Bernoulli pull: reward 1.0 with probability `p`, else 0.0.
fn bernoulli(p: f32, rng: &mut Rng) -> f32 {
    if rng.uniform() < p { 1.0 } else { 0.0 }
}

/// Run a plain UCB1 bandit session (no divergence filter).
fn run_without_filter(probs: &[f32], seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut pruner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, ARMS);

    for _ in 0..EPISODES {
        pruner.prepare_episode(&mut rng);
        // Select arm via pruner's relevance scores
        let arm = pruner.best_arm();
        let reward = bernoulli(probs[arm], &mut rng);
        pruner.update(arm, reward);
    }

    pruner.visits().to_vec()
}

/// Run a UCB1 bandit session *with* IdeaDivergence filter.
fn run_with_filter(probs: &[f32], seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut pruner = BanditPruner::with_idea_divergence(
        NoScreeningPruner,
        BanditStrategy::Ucb1,
        ARMS,
        DIVERGENCE_THRESHOLD,
    );

    for _ in 0..EPISODES {
        pruner.prepare_episode(&mut rng);
        let arm = pruner.best_arm();
        let reward = bernoulli(probs[arm], &mut rng);
        pruner.update(arm, reward);
        pruner.update_divergence(arm);
    }

    pruner.visits().to_vec()
}

/// Count arms with >10% of total visits.
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
fn print_divergence_matrix(q_values: &[f32], visits: &[u32], subset: usize) {
    let n = subset.min(q_values.len());
    let max_visits = visits.iter().copied().max().unwrap_or(1).max(1) as f32;
    let score_vecs: Vec<[f32; 2]> = (0..n)
        .map(|i| [q_values[i], visits[i] as f32 / max_visits])
        .collect();

    println!("Divergence Matrix (L2 distance between arm score vectors):");

    // Header row
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

// ── Main ────────────────────────────────────────────────────────

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

    // Run both sessions
    println!("Running {EPISODES} episodes per session...");
    let visits_without = run_without_filter(&probs, SEED);
    let visits_with = run_with_filter(&probs, SEED);
    println!("Done.");
    println!();

    // Build Q-values for the divergence matrix from the filtered run
    // Re-run briefly to capture final Q-values
    let mut rng = Rng::new(SEED);
    let mut pruner_for_matrix = BanditPruner::with_idea_divergence(
        NoScreeningPruner,
        BanditStrategy::Ucb1,
        ARMS,
        DIVERGENCE_THRESHOLD,
    );
    for _ in 0..EPISODES {
        pruner_for_matrix.prepare_episode(&mut rng);
        let arm = pruner_for_matrix.best_arm();
        let reward = bernoulli(probs[arm], &mut rng);
        pruner_for_matrix.update(arm, reward);
        pruner_for_matrix.update_divergence(arm);
    }
    let q_values = pruner_for_matrix.q_values().to_vec();
    let final_visits = pruner_for_matrix.visits().to_vec();

    // Print divergence matrix (first 5 arms for readability)
    println!("─────────────────────────────────────────────────────────────");
    print_divergence_matrix(&q_values, &final_visits, 5);
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
    let total_no: u32 = visits_without.iter().sum();
    for (i, &v) in visits_without.iter().enumerate() {
        let pct = if total_no > 0 {
            v as f32 / total_no as f32 * 100.0
        } else {
            0.0
        };
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    Arm {i:>2}: {v:>4} ({pct:>5.1}%) {bar}");
    }
    println!();

    println!("=== With Divergence Filter (threshold={DIVERGENCE_THRESHOLD}) ===");
    println!("  Active arms (>10% visits): {active_with}");
    println!("  Top arm visits: {top_visits_yes} ({top_pct_yes:.1}%)");
    println!();
    println!("  Visit distribution:");
    let total_yes: u32 = visits_with.iter().sum();
    for (i, &v) in visits_with.iter().enumerate() {
        let pct = if total_yes > 0 {
            v as f32 / total_yes as f32 * 100.0
        } else {
            0.0
        };
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("    Arm {i:>2}: {v:>4} ({pct:>5.1}%) {bar}");
    }
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
    for (i, &q) in q_values.iter().enumerate() {
        let true_p = probs[i];
        let err = (q - true_p).abs();
        println!("  Arm {i:>2}: Q={q:.4} (true={true_p:.2}, err={err:.4})");
    }
}

// TL;DR: Demonstrates IdeaDivergence preventing bandit collapse — 10-arm UCB1 with filter maintains more diverse arm selection vs plain UCB1.
