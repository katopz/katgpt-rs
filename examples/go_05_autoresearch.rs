//! Plan 065 Phase 5 T42: AutoResearch Loop — automated hyperparameter search.
//!
//! Runs an AutoResearch loop: generates random hyperparameter configs ("arms"),
//! evaluates each against a baseline player via internal self-play, and uses
//! UCB1 bandit selection to focus on promising configurations.
//!
//! ```sh
//! # Quick: 5 arms × 10 evals × 5 games (greedy only)
//! GO_SET=quick cargo run --features go --example go_05_autoresearch
//!
//! # Default: 10 arms × 50 evals × 10 games (greedy only)
//! cargo run --features go --example go_05_autoresearch
//!
//! # Custom: 20 arms, 100 evals, 20 games per eval
//! GO_ARMS=20 GO_EVALS=100 GO_GAMES=20 cargo run --features go --example go_05_autoresearch
//!
//! # Against Greedy baseline (harder)
//! GO_BASELINE=greedy cargo run --features go --example go_05_autoresearch
//!
//! # Enable MCTS arms (slow!)
//! GO_MCTS=1 cargo run --features go --example go_05_autoresearch
//!
//! # Disable early stopping
//! GO_NO_PRUNE=1 cargo run --features go --example go_05_autoresearch
//! ```

use std::env;

use katgpt_rs::pruners::go::autoresearch::{
    AutoResearchConfig, BaselinePlayer, run_autoresearch,
};

// ── Constants ──────────────────────────────────────────────────

/// Default number of arms.
const DEFAULT_ARMS: usize = 10;

/// Default evaluations budget.
const DEFAULT_EVALS: usize = 50;

/// Default games per evaluation.
const DEFAULT_GAMES: usize = 10;

/// Default board size.
const DEFAULT_BOARD_SIZE: usize = 9;

/// Default progress interval (print every N evaluations).
const DEFAULT_PROGRESS: usize = 10;

/// Quick demo: tiny search.
const QUICK_ARMS: usize = 5;
const QUICK_EVALS: usize = 15;
const QUICK_GAMES: usize = 3;

// ── Output Formatting ─────────────────────────────────────────

/// Print the experiment header.
fn print_header(config: &AutoResearchConfig, max_mcts: Option<usize>) {
    let baseline_label = config.baseline.label();
    let mcts_label = match max_mcts {
        Some(0) => "Greedy Only",
        Some(_) => "MCTS Capped",
        None => "Full Range",
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║       AUTORESEARCH — {}×{}                            ║",
        config.board_size, config.board_size
    );
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Arms              : {:<6}                              ║",
        config.num_arms
    );
    println!(
        "║  Total Evals       : {:<6}                              ║",
        config.total_evaluations
    );
    println!(
        "║  Games/Eval        : {:<6}                              ║",
        config.games_per_eval
    );
    println!(
        "║  Baseline          : {:<10}                          ║",
        baseline_label
    );
    println!(
        "║  Player Mode       : {:<10}                          ║",
        mcts_label
    );
    println!(
        "║  Early Stopping    : {:<6}                              ║",
        if config.enable_pruning { "YES" } else { "NO" }
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

/// Print a section header.
fn print_section(title: &str) {
    println!();
    println!("━━━ {title} ━━━");
}

/// Print evaluation progress table header.
fn print_eval_table_header() {
    println!();
    println!(
        "  {:>4}  {:>5}  {:>30}  {:>6}  {:>6}  {:>7}",
        "#", "Arm", "Config", "Wins", "Games", "WR%"
    );
    println!(
        "  {}  {}  {}  {}  {}  {}",
        "─".repeat(4),
        "─".repeat(5),
        "─".repeat(30),
        "─".repeat(6),
        "─".repeat(6),
        "─".repeat(7)
    );
}

/// Print a single evaluation row.
fn print_eval_row(idx: usize, trial: &katgpt_rs::pruners::go::autoresearch::TrialLog) {
    println!(
        "  {:>4}  {:>5}  {:>30}  {:>6}  {:>6}  {:>6.1}%",
        idx + 1,
        trial.arm_index,
        trial.config_label,
        trial.wins,
        trial.games_played,
        trial.win_rate * 100.0,
    );
}

/// Print per-arm leaderboard.
fn print_leaderboard(result: &katgpt_rs::pruners::go::autoresearch::AutoResearchResult) {
    let mut arms: Vec<_> = result.arms.iter().filter(|a| a.pulls > 0).collect();

    arms.sort_by(|a, b| {
        b.mean_reward()
            .partial_cmp(&a.mean_reward())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!();
    println!("  Arm Leaderboard (by cumulative win rate):");
    println!(
        "  {:>4}  {:>4}  {:>30}  {:>7}  {:>6}  {:>6}  {:>7}  {:>8}",
        "Rank", "Arm", "Config", "Mean%", "Pulls", "Games", "Best%", "Status"
    );
    println!(
        "  {}  {}  {}  {}  {}  {}  {}  {}",
        "─".repeat(4),
        "─".repeat(4),
        "─".repeat(30),
        "─".repeat(7),
        "─".repeat(6),
        "─".repeat(6),
        "─".repeat(7),
        "─".repeat(8)
    );

    for (rank, arm) in arms.iter().enumerate() {
        let status = match arm.status {
            katgpt_rs::pruners::go::autoresearch::ArmStatus::Active => "Active",
            katgpt_rs::pruners::go::autoresearch::ArmStatus::Dropped => "Dropped",
        };
        println!(
            "  {:>4}  {:>4}  {:>30}  {:>6.1}%  {:>6}  {:>6}  {:>6.1}%  {:>8}",
            rank + 1,
            arm.index,
            arm.config.label(),
            arm.mean_reward() * 100.0,
            arm.pulls,
            arm.total_games,
            arm.best_win_rate * 100.0,
            status,
        );
    }
}

/// Print win rate evolution over evaluations.
fn print_win_rate_evolution(result: &katgpt_rs::pruners::go::autoresearch::AutoResearchResult) {
    println!();
    println!("  Win Rate Evolution (cumulative mean per evaluation):");

    let trials = &result.trials;
    let step = match trials.len() / 20 {
        0 => 1,
        s => s,
    };

    println!("  {:>6}  {:>8}  {:>8}", "Eval", "Inst%", "Cum%");
    println!("  {}  {}  {}", "─".repeat(6), "─".repeat(8), "─".repeat(8));

    for (i, trial) in trials.iter().enumerate() {
        if i % step == 0 || i == trials.len() - 1 {
            println!(
                "  {:>6}  {:>7.1}%  {:>7.1}%",
                i + 1,
                trial.win_rate * 100.0,
                trial.cumulative_win_rate * 100.0,
            );
        }
    }
}

/// Print convergence analysis.
fn print_convergence(result: &katgpt_rs::pruners::go::autoresearch::AutoResearchResult) {
    let trials = &result.trials;
    if trials.len() < 4 {
        return;
    }

    // Split into quarters
    let q = trials.len() / 4;
    let q1_mean: f32 = trials[..q].iter().map(|t| t.win_rate).sum::<f32>() / q as f32;
    let q2_mean: f32 = trials[q..q * 2].iter().map(|t| t.win_rate).sum::<f32>() / q as f32;
    let q3_mean: f32 = trials[q * 2..q * 3].iter().map(|t| t.win_rate).sum::<f32>() / q as f32;
    let q4_mean: f32 =
        trials[q * 3..].iter().map(|t| t.win_rate).sum::<f32>() / (trials.len() - q * 3) as f32;

    println!();
    println!("  Convergence (quarterly average win rates):");
    println!(
        "  Q1: {:.1}%  Q2: {:.1}%  Q3: {:.1}%  Q4: {:.1}%",
        q1_mean * 100.0,
        q2_mean * 100.0,
        q3_mean * 100.0,
        q4_mean * 100.0
    );

    let improvement = (q4_mean - q1_mean) * 100.0;
    if improvement > 5.0 {
        println!("  Trend: IMPROVING (+{improvement:.1}pp from Q1→Q4)");
    } else if improvement < -5.0 {
        println!("  Trend: DECLINING ({improvement:.1}pp from Q1→Q4)");
    } else {
        println!("  Trend: STABLE ({improvement:+.1}pp from Q1→Q4)");
    }
}

// ── Config Parsing ─────────────────────────────────────────────

/// Parse env var as usize with default.
fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Parse env var as bool (any non-empty value = true).
fn env_bool(key: &str) -> bool {
    env::var(key).is_ok()
}

/// Parse baseline player from env.
fn env_baseline() -> BaselinePlayer {
    match env::var("GO_BASELINE").as_deref() {
        Ok("greedy") | Ok("Greedy") => BaselinePlayer::Greedy,
        _ => BaselinePlayer::Random,
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    // Check for quick mode
    let quick = env_bool("GO_SET_QUICK") || env::var("GO_SET").as_deref() == Ok("quick");

    let (num_arms, total_evals, games_per_eval) = match quick {
        true => (QUICK_ARMS, QUICK_EVALS, QUICK_GAMES),
        false => (
            env_usize("GO_ARMS", DEFAULT_ARMS),
            env_usize("GO_EVALS", DEFAULT_EVALS),
            env_usize("GO_GAMES", DEFAULT_GAMES),
        ),
    };

    let board_size = env_usize("GO_BOARD", DEFAULT_BOARD_SIZE);
    let enable_pruning = !env_bool("GO_NO_PRUNE");
    let enable_mcts = env_bool("GO_MCTS");
    let baseline = env_baseline();

    // Cap MCTS budget: greedy only unless GO_MCTS=1
    let max_mcts_budget = match enable_mcts {
        true => None,          // Full range
        false => Some(0usize), // Greedy only
    };

    let config = AutoResearchConfig {
        num_arms,
        games_per_eval,
        total_evaluations: total_evals,
        board_size,
        baseline: baseline.clone(),
        enable_pruning,
        progress_interval: DEFAULT_PROGRESS,
        max_mcts_budget,
    };

    // ── Section 0: Header ──────────────────────────────────────
    print_header(&config, max_mcts_budget);

    // ── Section 1: Quick Config Scan ───────────────────────────
    print_section("Section 1: AutoResearch Hyperparameter Scan");

    let mut rng = fastrand::Rng::new();
    let result = run_autoresearch(&config, &mut rng);

    // ── Section 2: Evaluation Log ──────────────────────────────
    print_section("Section 2: Evaluation Log");
    print_eval_table_header();

    for (i, trial) in result.trials.iter().enumerate() {
        print_eval_row(i, trial);
    }

    // ── Section 3: Leaderboard ─────────────────────────────────
    print_section("Section 3: Arm Leaderboard");
    print_leaderboard(&result);

    // ── Section 4: Win Rate Evolution ──────────────────────────
    print_section("Section 4: Win Rate Evolution");
    print_win_rate_evolution(&result);

    // ── Section 5: Convergence ─────────────────────────────────
    print_section("Section 5: Convergence Analysis");
    print_convergence(&result);

    // ── Section 6: Final Summary ───────────────────────────────
    print_section("Section 6: Final Summary");
    result.print_summary();

    // Print best config details
    println!("  Best Config Details:");
    println!("    MCTS Budget    : {}", result.best_config.mcts_budget);
    println!("    Rollout Depth  : {}", result.best_config.rollout_depth);
    println!(
        "    Exploration C  : {:.3}",
        result.best_config.exploration_constant
    );
    println!(
        "    Bandit ε       : {:.3}",
        result.best_config.bandit_epsilon
    );
    println!("    Templates      : {}", result.best_config.template_count);
    println!(
        "    Weights        : [{:.2}, {:.2}, {:.2}, {:.2}]",
        result.best_config.heuristic_weights[0],
        result.best_config.heuristic_weights[1],
        result.best_config.heuristic_weights[2],
        result.best_config.heuristic_weights[3],
    );
    println!("    [liberty, capture, influence, center]");
    println!();
}
