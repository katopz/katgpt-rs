//! Partial Scoring Demo — Binary vs Graduated Reward Learning Curves (Plan 191 T4.1)
//!
//! Demonstrates that BomberPartialScorer provides richer learning signal than binary
//! win/loss, enabling the bandit to distinguish arm quality faster.
//!
//! Run: `cargo run --features "partial_scoring" --example partial_scoring_demo`

#[cfg(feature = "partial_scoring")]
use katgpt_core::GameTrace;
#[cfg(feature = "partial_scoring")]
use katgpt_core::PartialScorer;
#[cfg(feature = "partial_scoring")]
use katgpt_rs::pruners::{BanditStats, BomberPartialScorer, WinLossScorer};
#[cfg(feature = "partial_scoring")]
use katgpt_rs::types::Rng;

#[cfg(feature = "partial_scoring")]
const MAX_TICKS: u32 = 200;
#[cfg(feature = "partial_scoring")]
const NUM_ARMS: usize = 4;
#[cfg(feature = "partial_scoring")]
const EPISODES: usize = 200;
#[cfg(feature = "partial_scoring")]
const SNAPSHOT_EPISODES: [usize; 4] = [10, 50, 100, 200];

/// Arm profiles: (mean_survival_ticks, mean_kills, win_rate).
///
/// Arm 0 is the GOAT (high survival + kills even in loss),
/// descending to Arm 3 which is objectively terrible.
#[cfg(feature = "partial_scoring")]
const ARM_PROFILES: [(f32, f32, f32); NUM_ARMS] = [
    (180.0, 3.0, 0.80), // Arm 0: dominant
    (120.0, 1.5, 0.60), // Arm 1: decent
    (60.0, 0.5, 0.40),  // Arm 2: weak
    (30.0, 0.1, 0.20),  // Arm 3: terrible
];

/// Simulate a game episode for a given arm, producing a noisy `GameTrace`.
#[cfg(feature = "partial_scoring")]
fn simulate_episode(arm: usize, rng: &mut Rng) -> GameTrace {
    let (mean_surv, mean_kills, win_rate) = ARM_PROFILES[arm];

    // Win/loss coin flip
    let win = rng.uniform() < win_rate;
    let final_reward = if win { 1.0 } else { 0.0 };

    // Survival: add noise, clamp. Loses die faster.
    let survival_base = if win { mean_surv } else { mean_surv * 0.5 };
    let survival = ((survival_base + rng.normal() * 20.0).clamp(1.0, MAX_TICKS as f32)) as u32;

    // Kills: add noise, clamp to 0+
    let kills = ((mean_kills + rng.normal() * 0.5).clamp(0.0, 10.0)) as u32;

    // Actions taken ≈ survival * some activity factor
    let actions_taken = ((survival as f32 * 0.25).clamp(1.0, 200.0)) as u32;

    GameTrace {
        survival_ticks: survival,
        kills,
        actions_taken,
        max_ticks: MAX_TICKS,
        final_reward,
    }
}

/// UCB1 arm selection: unvisited arms first, then highest UCB1 score.
#[cfg(feature = "partial_scoring")]
fn select_ucb1(stats: &BanditStats) -> usize {
    for i in 0..stats.num_arms() {
        if stats.visit_count(i) == 0 {
            return i;
        }
    }
    (0..stats.num_arms())
        .max_by(|&a, &b| {
            stats
                .ucb1_score(a)
                .partial_cmp(&stats.ucb1_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

#[cfg(feature = "partial_scoring")]
fn main() {
    let mut rng = Rng::new(42);

    let binary_scorer = WinLossScorer;
    let partial_scorer = BomberPartialScorer {
        max_ticks: MAX_TICKS,
    };

    // Two independent bandit trackers for fair comparison
    let mut binary_stats = BanditStats::new(NUM_ARMS);
    let mut partial_stats = BanditStats::new(NUM_ARMS);

    // Track snapshots at specific episodes
    let mut snapshots: Vec<(usize, Vec<f32>, Vec<f32>)> =
        Vec::with_capacity(SNAPSHOT_EPISODES.len());

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  Partial Scoring Demo — Plan 191 T4.1                                ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {} arms, {} episodes, UCB1 strategy", NUM_ARMS, EPISODES);
    println!("  Arm profiles (mean_survival, mean_kills, win_rate):");
    for (i, (surv, kills, wr)) in ARM_PROFILES.iter().enumerate() {
        println!("    Arm {i}: ({surv:.0}, {kills:.1}, {wr:.2})");
    }
    println!();

    // ── Run episodes ────────────────────────────────────────────────
    let mut snapshot_idx = 0;

    for ep in 1..=EPISODES {
        // Select arm (use partial stats for selection — both see same arm sequence)
        let arm = select_ucb1(&partial_stats);

        // Simulate episode
        let trace = simulate_episode(arm, &mut rng);

        // Binary reward
        let binary_reward = binary_scorer.partial_score(&trace);
        binary_stats.update(arm, binary_reward);

        // Partial reward
        let partial_reward = partial_scorer.partial_score(&trace);
        partial_stats.update(arm, partial_reward);

        // Snapshot at checkpoints
        if snapshot_idx < SNAPSHOT_EPISODES.len() && ep == SNAPSHOT_EPISODES[snapshot_idx] {
            let bq: Vec<f32> = binary_stats.q_values().to_vec();
            let pq: Vec<f32> = partial_stats.q_values().to_vec();
            snapshots.push((ep, bq, pq));
            snapshot_idx += 1;
        }
    }

    // ── Print learning curve table ──────────────────────────────────
    println!("━━━ Learning Curve: Binary Q-values vs Partial Q-values ━━━━━━━━━━━━━");
    println!();
    println!("  Episode | Binary Q-values           | Partial Q-values          | Best Arm");
    println!(
        "  {}─┼─{}─┼─{}─┼──────────",
        "─".repeat(8),
        "─".repeat(26),
        "─".repeat(26)
    );

    for (ep, bq, pq) in &snapshots {
        let bq_str = format!("[{:.2}, {:.2}, {:.2}, {:.2}]", bq[0], bq[1], bq[2], bq[3]);
        let pq_str = format!("[{:.2}, {:.2}, {:.2}, {:.2}]", pq[0], pq[1], pq[2], pq[3]);
        let binary_best = bq
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let partial_best = pq
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        println!(
            "  {:>8} | {:<26} | {:<26} | {} / {}",
            ep, bq_str, pq_str, binary_best, partial_best
        );
    }
    println!();

    // ── Final analysis ──────────────────────────────────────────────
    println!("━━━ Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let final_bq = binary_stats.q_values();
    let final_pq = partial_stats.q_values();

    // Spread: std dev of Q-values — wider = more discriminative
    let binary_mean: f32 = final_bq.iter().copied().sum::<f32>() / NUM_ARMS as f32;
    let partial_mean: f32 = final_pq.iter().copied().sum::<f32>() / NUM_ARMS as f32;
    let binary_spread: f32 = final_bq
        .iter()
        .map(|&q| (q - binary_mean).powi(2))
        .sum::<f32>()
        .sqrt();
    let partial_spread: f32 = final_pq
        .iter()
        .map(|&q| (q - partial_mean).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("  Binary Q-value spread (σ):  {binary_spread:.4}");
    println!("  Partial Q-value spread (σ): {partial_spread:.4}");
    println!(
        "  Spread ratio:               {:.2}× (partial is {}discriminative)",
        partial_spread / binary_spread.max(1e-6),
        if partial_spread > binary_spread {
            "more "
        } else {
            "less "
        }
    );
    println!();

    println!("  Binary visits:  {:?}", binary_stats.visits());
    println!("  Partial visits: {:?}", partial_stats.visits());
    println!();

    println!("━━━ Conclusion ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  Partial scoring provides richer signal than binary win/loss.");
    println!("  Arms that lose but survive long (e.g. Arm 0) get partial credit,");
    println!("  letting UCB1 distinguish quality even before the win rate converges.");
    println!("  Binary scoring collapses loss episodes to 0.0, discarding this signal.");
    println!();
    println!("{}", "═".repeat(72));
}

#[cfg(not(feature = "partial_scoring"))]
fn main() {
    eprintln!(
        "Enable partial_scoring feature: cargo run --features partial_scoring --example partial_scoring_demo"
    );
}
