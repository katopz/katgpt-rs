//! AutoResearch Loop — automated hyperparameter search for Go AI.
//!
//! Plan 065 Phase 5 (T38–T41):
//! - T38: Module creation
//! - T39: `GoResearchConfig` — hyperparameters to optimize
//! - T40: `AutoResearchLoop` — UCB1 bandit over config arms with early stopping
//! - T41: `run_autoresearch()` — N arms × K games, find best config
//!
//! ## How It Works
//!
//! 1. Generate N random hyperparameter configurations ("arms")
//! 2. Select next arm via UCB1 (balances exploration vs exploitation)
//! 3. Evaluate arm: play K games against a baseline player (internal, no API)
//! 4. Record win rate as reward, update arm's bandit stats
//! 5. Early stopping: drop arms below 25th percentile after 10 evaluations
//! 6. Return best config found
//!
//! ## Research Player
//!
//! Each arm creates a [`ResearchPlayer`] that applies the arm's heuristic weights:
//! - `mcts_budget > 0` → delegates to [`GoMctsPlayer`] with budget/depth
//! - `mcts_budget == 0` → weighted greedy scoring with `heuristic_weights`
//!
//! The weights `[liberty, capture, influence, center]` modify the base greedy
//! scoring, enabling the search to discover good weight combinations.

use std::io::Write;
use std::time::Instant;

use fastrand::Rng;

use super::players::{GoGreedyPlayer, GoMctsPlayer, GoPlayer, GoRandomPlayer};
use super::state::GoState;
use super::types::{GoAction, GoCell};
use crate::pruners::game_state::GameState;

// ── Constants ──────────────────────────────────────────────────

/// UCB1 exploration constant for arm selection.
const ARM_UCB1_C: f32 = 2.0;

/// Default number of arms to explore.
const DEFAULT_NUM_ARMS: usize = 30;

/// Default games per arm evaluation.
const DEFAULT_GAMES_PER_EVAL: usize = 50;

/// Minimum evaluations per arm before early stopping can prune it.
const MIN_EVALS_FOR_PRUNING: usize = 10;

/// Percentile threshold for early stopping (drop below this).
const PRUNING_PERCENTILE: f32 = 0.25;

// ── Weight indices ─────────────────────────────────────────────

/// Index into `heuristic_weights` for liberty scoring.
const W_LIBERTY: usize = 0;
/// Index into `heuristic_weights` for capture scoring.
const W_CAPTURE: usize = 1;
/// Index into `heuristic_weights` for influence scoring.
const W_INFLUENCE: usize = 2;
/// Index into `heuristic_weights` for center scoring.
const W_CENTER: usize = 3;

// ── T39: GoResearchConfig ──────────────────────────────────────

/// Hyperparameter configuration for Go AI research.
///
/// Each field is a dimension in the search space. The AutoResearch loop
/// explores combinations to find the config that maximizes win rate
/// against a baseline player.
#[derive(Clone, Debug)]
pub struct GoResearchConfig {
    /// Board dimension (typically 9 for speed).
    pub board_size: usize,
    /// MCTS iterations per move (0 = pure greedy, 100–10000 for MCTS).
    pub mcts_budget: usize,
    /// MCTS rollout depth (5–50).
    pub rollout_depth: usize,
    /// UCB1 exploration constant (0.5–2.0).
    pub exploration_constant: f32,
    /// ε-greedy exploration rate (0.05–0.5).
    pub bandit_epsilon: f32,
    /// Number of active templates (2–4, G7 scope reduction).
    pub template_count: usize,
    /// Heuristic weights: `[liberty, capture, influence, center]`.
    pub heuristic_weights: [f32; 4],
}

impl GoResearchConfig {
    /// Create a config with sensible defaults.
    pub fn new() -> Self {
        Self {
            board_size: 9,
            mcts_budget: 200,
            rollout_depth: 30,
            exploration_constant: 1.414,
            bandit_epsilon: 0.15,
            template_count: 4,
            heuristic_weights: [0.5, 10.0, 1.0, 2.0],
        }
    }

    /// Generate a random config within reasonable ranges.
    pub fn random(rng: &mut Rng) -> Self {
        let mcts_budget = [0, 50, 100, 200, 500, 1000, 2000][rng.usize(..7)];
        let rollout_depth = [10, 20, 30, 40, 50][rng.usize(..5)];
        let exploration_constant = 0.5 + rng.f32() * 1.5;
        let bandit_epsilon = 0.05 + rng.f32() * 0.45;
        let template_count = 2 + rng.usize(..3);
        let heuristic_weights = [
            0.1 + rng.f32() * 2.0,
            1.0 + rng.f32() * 20.0,
            rng.f32() * 3.0,
            0.5 + rng.f32() * 4.0,
        ];

        Self {
            board_size: 9,
            mcts_budget,
            rollout_depth,
            exploration_constant,
            bandit_epsilon,
            template_count,
            heuristic_weights,
        }
    }

    /// Human-readable label for this config.
    pub fn label(&self) -> String {
        format!(
            "M{}:D{}:C{:.1}:E{:.2}:T{}",
            self.mcts_budget,
            self.rollout_depth,
            self.exploration_constant,
            self.bandit_epsilon,
            self.template_count,
        )
    }
}

impl Default for GoResearchConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Arm Status ─────────────────────────────────────────────────

/// Status of a research arm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArmStatus {
    /// Active and being evaluated.
    Active,
    /// Dropped by early stopping.
    Dropped,
}

// ── Research Arm ───────────────────────────────────────────────

/// A research arm: a config with bandit evaluation stats.
#[derive(Clone, Debug)]
pub struct ResearchArm {
    /// The hyperparameter configuration.
    pub config: GoResearchConfig,
    /// Arm index (0-based).
    pub index: usize,
    /// Current status.
    pub status: ArmStatus,
    /// Number of evaluations (pulls).
    pub pulls: usize,
    /// Total reward (sum of win rates).
    pub total_reward: f32,
    /// Total games played.
    pub total_games: usize,
    /// Total wins.
    pub total_wins: usize,
    /// Best single-evaluation win rate observed.
    pub best_win_rate: f32,
}

impl ResearchArm {
    /// Create a new arm with the given config and index.
    fn new(config: GoResearchConfig, index: usize) -> Self {
        Self {
            config,
            index,
            status: ArmStatus::Active,
            pulls: 0,
            total_reward: 0.0,
            total_games: 0,
            total_wins: 0,
            best_win_rate: 0.0,
        }
    }

    /// Mean win rate across all evaluations.
    pub fn mean_reward(&self) -> f32 {
        match self.pulls {
            0 => 0.0,
            _ => self.total_reward / self.pulls as f32,
        }
    }

    /// UCB1 score for arm selection.
    fn ucb1(&self, total_pulls: usize) -> f32 {
        if self.pulls == 0 || total_pulls == 0 {
            return f32::MAX;
        }
        let q = self.mean_reward();
        let n = self.pulls as f32;
        let total = total_pulls as f32;
        q + (ARM_UCB1_C * total.ln() / n).sqrt()
    }

    /// Record an evaluation result.
    fn record(&mut self, wins: usize, games: usize) {
        let win_rate = match games {
            0 => 0.0,
            _ => wins as f32 / games as f32,
        };
        self.pulls += 1;
        self.total_reward += win_rate;
        self.total_games += games;
        self.total_wins += wins;
        if win_rate > self.best_win_rate {
            self.best_win_rate = win_rate;
        }
    }
}

// ── Trial Log ──────────────────────────────────────────────────

/// Record of a single arm evaluation trial.
#[derive(Clone, Debug)]
pub struct TrialLog {
    /// Arm index.
    pub arm_index: usize,
    /// Config label.
    pub config_label: String,
    /// Games played in this evaluation.
    pub games_played: usize,
    /// Wins in this evaluation.
    pub wins: usize,
    /// Win rate for this evaluation (0.0–1.0).
    pub win_rate: f32,
    /// Cumulative mean win rate for this arm across all evaluations.
    pub cumulative_win_rate: f32,
    /// Evaluation duration.
    pub duration: std::time::Duration,
}

// ── Internal Game Result ───────────────────────────────────────

/// Result of a single internal game.
#[allow(dead_code)]
#[derive(Clone, Debug)]
struct InternalGameResult {
    /// Did player1 (Black) win?
    won: bool,
    /// Score from Black's perspective (positive = Black wins).
    score_delta: f32,
    /// Total moves played.
    total_moves: usize,
    /// Game duration.
    duration: std::time::Duration,
}

// ── AutoResearch Result ────────────────────────────────────────

/// Final result of an AutoResearch run.
#[derive(Clone, Debug)]
pub struct AutoResearchResult {
    /// Best config found.
    pub best_config: GoResearchConfig,
    /// Best mean win rate across all evaluations.
    pub best_win_rate: f32,
    /// Total arms generated (including dropped).
    pub total_arms: usize,
    /// Arms still active at end.
    pub active_arms: usize,
    /// Total evaluations performed.
    pub total_evaluations: usize,
    /// Total games played across all evaluations.
    pub total_games: usize,
    /// All trial logs in order.
    pub trials: Vec<TrialLog>,
    /// Per-arm final stats.
    pub arms: Vec<ResearchArm>,
    /// Total duration.
    pub duration: std::time::Duration,
}

impl AutoResearchResult {
    /// Print a formatted summary to stdout.
    pub fn print_summary(&self) {
        println!();
        println!("══════════════════════════════════════════════════════════════");
        println!("  AUTORESEARCH RESULTS");
        println!("══════════════════════════════════════════════════════════════");
        println!("  Best Config     : {}", self.best_config.label());
        println!("  Best Win Rate   : {:.1}%", self.best_win_rate * 100.0);
        println!(
            "  Total Arms      : {} ({} active, {} dropped)",
            self.total_arms,
            self.active_arms,
            self.total_arms - self.active_arms
        );
        println!("  Total Evals     : {}", self.total_evaluations);
        println!("  Total Games     : {}", self.total_games);
        println!(
            "  Duration        : {:.1}s ({:.0} games/s)",
            self.duration.as_secs_f64(),
            self.total_games as f64 / self.duration.as_secs_f64().max(0.001)
        );
        println!();

        // Top 5 arms
        println!("  Top 5 Arms:");
        println!(
            "  {:>4}  {:>30}  {:>8}  {:>6}  {:>6}",
            "#", "Config", "WinRate", "Pulls", "Games"
        );
        println!(
            "  {}  {}  {}  {}  {}",
            "─".repeat(4),
            "─".repeat(30),
            "─".repeat(8),
            "─".repeat(6),
            "─".repeat(6)
        );

        let mut sorted_arms: Vec<_> = self
            .arms
            .iter()
            .filter(|a| a.status == ArmStatus::Active && a.pulls > 0)
            .collect();
        sorted_arms.sort_by(|a, b| {
            b.mean_reward()
                .partial_cmp(&a.mean_reward())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (rank, arm) in sorted_arms.iter().take(5).enumerate() {
            println!(
                "  {:>4}  {:>30}  {:>7.1}%  {:>6}  {:>6}",
                rank + 1,
                arm.config.label(),
                arm.mean_reward() * 100.0,
                arm.pulls,
                arm.total_games,
            );
        }
    }
}

// ── Board Helpers ──────────────────────────────────────────────
// Issue 001 H-20: `board_neighbors` and `flood_group` were copy-pasted across
// players.rs, g_zero_player.rs, and autoresearch.rs. Now imported from
// `go::utils` so all three call sites share one implementation.
use super::utils::{board_neighbors, flood_group};

/// Stones captured by `me` between two states (before → after).
fn captures_for(me: GoCell, before: &GoState, after: &GoState) -> u32 {
    match me {
        GoCell::Black => after.captured_black.saturating_sub(before.captured_black),
        GoCell::White => after.captured_white.saturating_sub(before.captured_white),
        GoCell::Empty => 0,
    }
}

/// Center proximity bonus: 1.0 at center, 0.0 at corners.
fn center_bonus(row: usize, col: usize, size: usize) -> f32 {
    let center = (size - 1) as f32 / 2.0;
    let max_dist = center;
    if max_dist == 0.0 {
        return 1.0;
    }
    let dist = ((row as f32 - center).abs() + (col as f32 - center).abs()) / 2.0;
    1.0 - dist / max_dist
}

/// True if (row, col) is on the first board line (edge).
fn is_first_line(row: usize, col: usize, size: usize) -> bool {
    row == 0 || row == size - 1 || col == 0 || col == size - 1
}

// ── Weighted Greedy Scoring ────────────────────────────────────

/// Weighted greedy score using research config's `heuristic_weights`.
///
/// The 4 weights control:
/// - `W_LIBERTY` (0): liberty count multiplier
/// - `W_CAPTURE` (1): capture stone value
/// - `W_INFLUENCE` (2): atari threat multiplier
/// - `W_CENTER` (3): center bonus multiplier
fn weighted_greedy_score(state: &GoState, row: usize, col: usize, weights: &[f32; 4]) -> f32 {
    let me = state.to_play;
    let opp = me.opponent();
    let size = state.size;
    let idx = state.flat_index(row, col);

    let action = GoAction::Place(row, col);
    let new_state = state.advance(&action, me.player_id());

    // Capture priority
    let captures = captures_for(me, state, &new_state);
    let mut score = captures as f32 * weights[W_CAPTURE];

    // Liberty gain of resulting group
    let (_, libs) = flood_group(&new_state.board, idx, size);
    score += libs.len() as f32 * weights[W_LIBERTY];

    // Atari threat: opponent groups with 1 liberty after placement
    for n in board_neighbors(idx, size) {
        if new_state.board[n] == opp {
            let (_, opp_libs) = flood_group(&new_state.board, n, size);
            if opp_libs.len() == 1 {
                score += 5.0 * weights[W_INFLUENCE];
            }
        }
    }

    // Center bonus
    score += center_bonus(row, col, size) * weights[W_CENTER];

    // Edge penalty (unless capturing)
    if captures == 0 && is_first_line(row, col, size) {
        score -= 3.0;
    }

    // Self-atari penalty
    if libs.len() == 1 && captures == 0 {
        score -= 20.0;
    }

    score
}

// ── Research Player ────────────────────────────────────────────

/// A player that uses `GoResearchConfig` weights for move evaluation.
///
/// - If `mcts_budget > 0`: delegates to [`GoMctsPlayer`] with budget/depth.
/// - If `mcts_budget == 0`: uses weighted greedy scoring.
struct ResearchPlayer {
    /// MCTS player (if budget > 0).
    mcts: Option<GoMctsPlayer>,
    /// Heuristic weights for greedy scoring.
    weights: [f32; 4],
    /// Whether to use MCTS (true) or greedy (false).
    use_mcts: bool,
}

impl ResearchPlayer {
    /// Create a research player from a config.
    fn from_config(config: &GoResearchConfig) -> Self {
        let mcts = if config.mcts_budget > 0 {
            Some(GoMctsPlayer::new(config.mcts_budget, config.rollout_depth))
        } else {
            None
        };
        Self {
            mcts,
            weights: config.heuristic_weights,
            use_mcts: config.mcts_budget > 0,
        }
    }
}

impl GoPlayer for ResearchPlayer {
    fn select_move(
        &mut self,
        state: &GoState,
        legal_moves: &[(usize, usize)],
        rng: &mut Rng,
    ) -> GoAction {
        if legal_moves.is_empty() {
            return GoAction::Pass;
        }

        // MCTS path
        if self.use_mcts
            && let Some(ref mut mcts) = self.mcts
        {
            return mcts.select_move(state, legal_moves, rng);
        }

        // Weighted greedy fallback
        let best = legal_moves
            .iter()
            .max_by(|&&a, &&b| {
                let sa = weighted_greedy_score(state, a.0, a.1, &self.weights);
                let sb = weighted_greedy_score(state, b.0, b.1, &self.weights);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("legal_moves is non-empty");

        GoAction::Place(best.0, best.1)
    }

    fn name(&self) -> &'static str {
        match self.use_mcts {
            true => "Research-MCTS",
            false => "Research-Greedy",
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ── Baseline Player Selector ──────────────────────────────────

/// Baseline player type for evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BaselinePlayer {
    /// Random legal moves with occasional pass.
    Random,
    /// Greedy scorer (captures + liberties + position).
    Greedy,
}

impl BaselinePlayer {
    /// Create a boxed baseline player.
    pub fn create_player(&self) -> Box<dyn GoPlayer> {
        match self {
            Self::Random => Box::new(GoRandomPlayer),
            Self::Greedy => Box::new(GoGreedyPlayer),
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::Greedy => "Greedy",
        }
    }
}

// ── Internal Game Evaluation ───────────────────────────────────

/// Play a single internal game between two players.
///
/// Player 1 plays Black, player 2 plays White.
/// Returns result from player 1's (Black's) perspective.
fn play_internal_game(
    player1: &mut dyn GoPlayer,
    player2: &mut dyn GoPlayer,
    board_size: usize,
    rng: &mut Rng,
) -> InternalGameResult {
    let start = Instant::now();
    let max_moves = board_size * board_size * 3;
    let mut state = GoState::new(board_size);
    let mut moves_played = 0usize;
    let mut legal_buf = Vec::with_capacity(board_size * board_size);

    while !state.is_terminal() && moves_played < max_moves {
        state.legal_moves_into(&mut legal_buf);
        let legal_moves = &legal_buf;

        if legal_moves.is_empty() {
            state.play_pass();
            moves_played += 1;
            continue;
        }

        let action = match state.to_play {
            GoCell::Black => player1.select_move(&state, &legal_moves, rng),
            GoCell::White => player2.select_move(&state, &legal_moves, rng),
            GoCell::Empty => GoAction::Pass,
        };

        match &action {
            GoAction::Place(row, col) => {
                state.play_move(*row, *col);
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        moves_played += 1;
    }

    // Force game end if max moves reached
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
    }

    let score = state.score();
    let won = score > 0.0; // Positive = Black wins = player1 wins

    InternalGameResult {
        won,
        score_delta: score,
        total_moves: moves_played,
        duration: start.elapsed(),
    }
}

/// Evaluate a config by playing N games against a baseline.
///
/// Our player plays Black, baseline plays White.
/// Returns `(wins, total_games)`.
fn evaluate_arm(
    config: &GoResearchConfig,
    baseline: &BaselinePlayer,
    num_games: usize,
    rng: &mut Rng,
) -> (usize, usize) {
    let mut research_player = ResearchPlayer::from_config(config);
    let mut baseline_player = baseline.create_player();

    let mut wins = 0usize;

    for _ in 0..num_games {
        let result = play_internal_game(
            &mut research_player,
            &mut *baseline_player,
            config.board_size,
            rng,
        );
        if result.won {
            wins += 1;
        }
        research_player.reset();
        baseline_player.reset();
    }

    (wins, num_games)
}

// ── Early Stopping ─────────────────────────────────────────────

/// Prune arms below the 25th percentile of mean reward.
///
/// Only prunes arms that have been evaluated at least `MIN_EVALS_FOR_PRUNING` times.
fn prune_weak_arms(arms: &mut [ResearchArm]) {
    let mut rewards: Vec<f32> = arms
        .iter()
        .filter(|a| a.status == ArmStatus::Active && a.pulls >= MIN_EVALS_FOR_PRUNING)
        .map(|a| a.mean_reward())
        .collect();

    if rewards.len() < 4 {
        return; // Not enough arms to prune meaningfully
    }

    rewards.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold_idx = (rewards.len() as f32 * PRUNING_PERCENTILE) as usize;
    let threshold = rewards[threshold_idx];

    for arm in arms.iter_mut() {
        if arm.status == ArmStatus::Active
            && arm.pulls >= MIN_EVALS_FOR_PRUNING
            && arm.mean_reward() < threshold
        {
            arm.status = ArmStatus::Dropped;
        }
    }
}

// ── T40: AutoResearchConfig ────────────────────────────────────

/// Configuration for the AutoResearch loop.
#[derive(Clone, Debug)]
pub struct AutoResearchConfig {
    /// Number of arms (hyperparameter configurations) to generate.
    pub num_arms: usize,
    /// Number of games per arm evaluation.
    pub games_per_eval: usize,
    /// Total evaluation budget (total evaluations across all arms).
    pub total_evaluations: usize,
    /// Board size for evaluation games.
    pub board_size: usize,
    /// Baseline player to evaluate against.
    pub baseline: BaselinePlayer,
    /// Enable early stopping (drop weak arms).
    pub enable_pruning: bool,
    /// Print progress every N evaluations.
    pub progress_interval: usize,
    /// Cap MCTS budget per arm (None = unlimited, Some(0) = greedy only).
    pub max_mcts_budget: Option<usize>,
}

impl Default for AutoResearchConfig {
    fn default() -> Self {
        Self {
            num_arms: DEFAULT_NUM_ARMS,
            games_per_eval: DEFAULT_GAMES_PER_EVAL,
            total_evaluations: 150,
            board_size: 9,
            baseline: BaselinePlayer::Random,
            enable_pruning: true,
            progress_interval: 10,
            max_mcts_budget: None,
        }
    }
}

// ── T41: Run AutoResearch ──────────────────────────────────────

/// Run the AutoResearch loop.
///
/// ## Algorithm
///
/// 1. Generate `num_arms` random hyperparameter configs
/// 2. Loop `total_evaluations` times:
///    a. Select active arm via UCB1
///    b. Evaluate arm: play `games_per_eval` games against baseline
///    c. Record win rate, update arm stats
///    d. Periodically prune weak arms (early stopping)
/// 3. Return best config by cumulative mean win rate
///
/// ## Example
///
/// ```ignore
/// use katgpt_rs::pruners::go::autoresearch::{
///     AutoResearchConfig, BaselinePlayer, run_autoresearch,
/// };
///
/// let mut rng = fastrand::Rng::new();
/// let config = AutoResearchConfig {
///     num_arms: 10,
///     games_per_eval: 20,
///     total_evaluations: 50,
///     baseline: BaselinePlayer::Random,
///     ..Default::default()
/// };
///
/// let result = run_autoresearch(&config, &mut rng);
/// println!("Best config: {} ({:.1}% win rate)", result.best_config.label(), result.best_win_rate * 100.0);
/// ```
pub fn run_autoresearch(config: &AutoResearchConfig, rng: &mut Rng) -> AutoResearchResult {
    let start = Instant::now();

    // Generate random arms
    let mut arms: Vec<ResearchArm> = (0..config.num_arms)
        .map(|i| {
            let mut cfg = GoResearchConfig::random(rng);
            cfg.board_size = config.board_size;
            // Clamp MCTS budget if configured (Some(0) = greedy only for fast tests)
            if let Some(max) = config.max_mcts_budget {
                cfg.mcts_budget = cfg.mcts_budget.min(max);
            }
            ResearchArm::new(cfg, i)
        })
        .collect();

    let mut trials: Vec<TrialLog> = Vec::new();
    let mut total_pulls = 0usize;
    let mut total_games = 0usize;

    for eval_idx in 0..config.total_evaluations {
        // Select arm via UCB1 (active arms only)
        let active_indices: Vec<usize> = arms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.status == ArmStatus::Active)
            .map(|(i, _)| i)
            .collect();

        if active_indices.is_empty() {
            log::warn!("All arms dropped at evaluation {eval_idx}");
            break;
        }

        let arm_idx = active_indices
            .iter()
            .max_by(|&&a, &&b| {
                arms[a]
                    .ucb1(total_pulls)
                    .partial_cmp(&arms[b].ucb1(total_pulls))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(active_indices[0]);

        // Evaluate arm
        let eval_start = Instant::now();
        let (wins, games) = evaluate_arm(
            &arms[arm_idx].config,
            &config.baseline,
            config.games_per_eval,
            rng,
        );
        let eval_duration = eval_start.elapsed();

        // Update arm stats
        let win_rate = match games {
            0 => 0.0_f32,
            _ => wins as f32 / games as f32,
        };
        arms[arm_idx].record(wins, games);
        total_pulls += 1;
        total_games += games;

        // Record trial
        trials.push(TrialLog {
            arm_index: arm_idx,
            config_label: arms[arm_idx].config.label(),
            games_played: games,
            wins,
            win_rate,
            cumulative_win_rate: arms[arm_idx].mean_reward(),
            duration: eval_duration,
        });

        // Progress print
        if (eval_idx + 1) % config.progress_interval == 0 {
            let active = arms
                .iter()
                .filter(|a| a.status == ArmStatus::Active)
                .count();
            eprintln!(
                "  [{:>3}/{}] arm={arm_idx} WR={win_rate:.0}% cum={:.1}% active={active}",
                eval_idx + 1,
                config.total_evaluations,
                arms[arm_idx].mean_reward() * 100.0,
            );
            let _ = std::io::stderr().flush();
        }

        // Early stopping: prune weak arms periodically
        if config.enable_pruning && (eval_idx + 1) % (MIN_EVALS_FOR_PRUNING * 2) == 0 {
            prune_weak_arms(&mut arms);
        }
    }

    // Find best arm by cumulative mean reward
    let best_arm = arms.iter().filter(|a| a.pulls > 0).max_by(|a, b| {
        a.mean_reward()
            .partial_cmp(&b.mean_reward())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (best_config, best_win_rate) = match best_arm {
        Some(arm) => (arm.config.clone(), arm.mean_reward()),
        None => (GoResearchConfig::default(), 0.0),
    };

    let active_arms = arms
        .iter()
        .filter(|a| a.status == ArmStatus::Active)
        .count();

    AutoResearchResult {
        best_config,
        best_win_rate,
        total_arms: config.num_arms,
        active_arms,
        total_evaluations: total_pulls,
        total_games,
        trials,
        arms,
        duration: start.elapsed(),
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_research_config_random_generates_valid_configs() {
        let mut rng = Rng::with_seed(42);

        for _ in 0..100 {
            let config = GoResearchConfig::random(&mut rng);
            assert_eq!(config.board_size, 9);
            assert!(config.mcts_budget <= 2000);
            assert!(config.rollout_depth >= 10 && config.rollout_depth <= 50);
            assert!(config.exploration_constant >= 0.5 && config.exploration_constant <= 2.0);
            assert!(config.bandit_epsilon >= 0.05 && config.bandit_epsilon <= 0.5);
            assert!(config.template_count >= 2 && config.template_count <= 4);
        }
    }

    #[test]
    fn research_arm_ucb1_unpulled_is_max() {
        let config = GoResearchConfig::new();
        let arm = ResearchArm::new(config, 0);
        assert_eq!(arm.ucb1(0), f32::MAX);
    }

    #[test]
    fn research_arm_records_evaluation() {
        let config = GoResearchConfig::new();
        let mut arm = ResearchArm::new(config, 0);

        arm.record(7, 10);
        assert_eq!(arm.pulls, 1);
        assert_eq!(arm.total_games, 10);
        assert_eq!(arm.total_wins, 7);
        assert!((arm.mean_reward() - 0.7).abs() < 0.001);
        assert!((arm.best_win_rate - 0.7).abs() < 0.001);

        arm.record(5, 10);
        assert_eq!(arm.pulls, 2);
        assert_eq!(arm.total_games, 20);
        assert_eq!(arm.total_wins, 12);
        assert!((arm.mean_reward() - 0.6).abs() < 0.001);
    }

    #[test]
    fn evaluate_arm_completes_games() {
        let mut rng = Rng::with_seed(42);
        let config = GoResearchConfig {
            mcts_budget: 0,
            ..GoResearchConfig::new()
        };

        let (wins, games) = evaluate_arm(&config, &BaselinePlayer::Random, 10, &mut rng);

        assert_eq!(games, 10);
        assert!(wins <= games);
    }

    #[test]
    fn autoresearch_finds_best_arm() {
        let mut rng = Rng::with_seed(42);
        let config = AutoResearchConfig {
            num_arms: 5,
            games_per_eval: 5,
            total_evaluations: 20,
            board_size: 9,
            baseline: BaselinePlayer::Random,
            enable_pruning: false,
            progress_interval: 100,
            max_mcts_budget: Some(0), // Greedy only for speed
        };

        let result = run_autoresearch(&config, &mut rng);

        assert!(result.total_arms > 0);
        assert!(result.total_evaluations > 0);
        assert!(result.total_games > 0);
        assert!(!result.best_config.label().is_empty());
        assert!(result.best_win_rate >= 0.0 && result.best_win_rate <= 1.0);
    }

    #[test]
    fn autoresearch_with_pruning_produces_valid_result() {
        let mut rng = Rng::with_seed(123);
        let config = AutoResearchConfig {
            num_arms: 10,
            games_per_eval: 5,
            total_evaluations: 200,
            board_size: 9,
            baseline: BaselinePlayer::Random,
            enable_pruning: true,
            progress_interval: 100,
            max_mcts_budget: Some(0), // Greedy only for speed
        };

        let result = run_autoresearch(&config, &mut rng);

        assert_eq!(result.total_arms, 10);
        assert!(result.active_arms > 0);
        assert!(result.total_games > 0);
    }

    #[test]
    fn play_internal_game_completes() {
        let mut rng = Rng::with_seed(42);
        let mut p1 = GoRandomPlayer;
        let mut p2 = GoRandomPlayer;

        let result = play_internal_game(&mut p1, &mut p2, 9, &mut rng);

        assert!(result.total_moves > 0);
        assert!(result.duration.as_nanos() > 0);
    }

    #[test]
    fn research_player_uses_mcts_when_budget_positive() {
        let config = GoResearchConfig {
            mcts_budget: 100,
            ..GoResearchConfig::new()
        };
        let player = ResearchPlayer::from_config(&config);
        assert!(player.use_mcts);
        assert_eq!(player.name(), "Research-MCTS");
    }

    #[test]
    fn research_player_uses_greedy_when_budget_zero() {
        let config = GoResearchConfig {
            mcts_budget: 0,
            ..GoResearchConfig::new()
        };
        let player = ResearchPlayer::from_config(&config);
        assert!(!player.use_mcts);
        assert_eq!(player.name(), "Research-Greedy");
    }

    #[test]
    fn weighted_greedy_score_produces_finite_results() {
        let state = GoState::new(9);
        let weights = [0.5, 10.0, 1.0, 2.0];

        let score = weighted_greedy_score(&state, 4, 4, &weights);
        assert!(score.is_finite(), "Score should be finite, got {score}");
    }

    #[test]
    fn autoresearch_config_labels_are_mostly_unique() {
        let mut rng = Rng::with_seed(42);
        let mut labels = std::collections::HashSet::new();

        for _ in 0..30 {
            let config = GoResearchConfig::random(&mut rng);
            labels.insert(config.label());
        }

        assert!(
            labels.len() > 10,
            "Expected mostly unique labels, got {} unique out of 30",
            labels.len()
        );
    }

    #[test]
    fn baseline_player_creates_correctly() {
        let random = BaselinePlayer::Random.create_player();
        assert_eq!(random.name(), "Random");

        let greedy = BaselinePlayer::Greedy.create_player();
        assert_eq!(greedy.name(), "Greedy");
    }

    #[test]
    fn prune_weak_arms_drops_bottom_quarter() {
        let mut arms: Vec<ResearchArm> = (0..10)
            .map(|i| ResearchArm::new(GoResearchConfig::new(), i))
            .collect();

        // Simulate evaluations with varying win rates
        for (i, arm) in arms.iter_mut().enumerate() {
            let win_rate = (i as f32 + 1.0) / 10.0; // 10% to 100%
            let wins = (win_rate * 20.0) as usize;
            for _ in 0..MIN_EVALS_FOR_PRUNING {
                arm.record(wins, 20);
            }
        }

        let before_active = arms
            .iter()
            .filter(|a| a.status == ArmStatus::Active)
            .count();
        assert_eq!(before_active, 10);

        prune_weak_arms(&mut arms);

        let after_active = arms
            .iter()
            .filter(|a| a.status == ArmStatus::Active)
            .count();
        // Some arms should have been dropped
        assert!(
            after_active < 10,
            "Expected some arms to be pruned, but all {after_active} remain active"
        );
        assert!(
            after_active > 0,
            "Expected at least some arms to survive pruning"
        );
    }

    #[test]
    fn trial_log_records_cumulative_stats() {
        let mut rng = Rng::with_seed(42);
        let config = AutoResearchConfig {
            num_arms: 3,
            games_per_eval: 5,
            total_evaluations: 9,
            board_size: 9,
            baseline: BaselinePlayer::Random,
            enable_pruning: false,
            progress_interval: 100,
            max_mcts_budget: Some(0), // Greedy only for speed
        };

        let result = run_autoresearch(&config, &mut rng);

        // Should have one trial per evaluation
        assert_eq!(result.trials.len(), 9);

        // Each trial should have valid stats
        for trial in &result.trials {
            assert!(trial.win_rate >= 0.0 && trial.win_rate <= 1.0);
            assert!(trial.cumulative_win_rate >= 0.0 && trial.cumulative_win_rate <= 1.0);
            assert_eq!(trial.games_played, 5);
        }
    }
}
