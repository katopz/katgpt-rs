//! Plan 065 Phase 1 T16: MCTS player vs Random on 9×9.
//!
//! Demonstrates [`GoState`] with [`mcts_search`] against a random baseline.
//! Configurable via environment variables:
//!
//! ```sh
//! # Default: 20 games, MCTS budget 200
//! cargo run --features go --example go_01_mcts
//!
//! # Custom: 100 games, budget 500
//! GO_GAMES=100 GO_BUDGET=500 cargo run --features go --example go_01_mcts
//! ```

use std::env;
use std::time::Instant;

use katgpt_rs::pruners::game_state::{GameState, StateHeuristic, mcts_search};
use katgpt_rs::pruners::go::{DEFAULT_KOMI, GoAction, GoCell, GoHeuristic, GoReplay, GoState};

/// Number of games to play.
const DEFAULT_NUM_GAMES: usize = 20;

/// MCTS budget (number of `advance()` calls per search).
const DEFAULT_BUDGET: usize = 200;

/// Max moves before forcing game end (safety limit).
const MAX_MOVES: usize = 300;

// ── Players ────────────────────────────────────────────────────

/// Select a move using MCTS search.
fn mcts_select_move(state: &GoState, budget: usize, rng: &mut fastrand::Rng) -> GoAction {
    let player_id = state.to_play.player_id();
    let actions = state.available_actions(player_id);

    // If only pass available, just pass
    if actions.len() == 1 {
        return actions[0].clone();
    }

    // Filter out pass for MCTS — only search stone placements.
    // Pass is used as fallback when no good placement found.
    let stone_actions: Vec<GoAction> = actions
        .iter()
        .filter(|a| matches!(a, GoAction::Place(_, _)))
        .cloned()
        .collect();

    if stone_actions.is_empty() {
        return GoAction::Pass;
    }

    let heuristic = GoHeuristic;
    mcts_search(
        state,
        player_id,
        budget,
        50, // rollout_depth
        &|s: &GoState, pid: u8| heuristic.evaluate(s, pid),
        rng,
    )
}

/// Select a random legal move.
fn random_select_move(state: &GoState, rng: &mut fastrand::Rng) -> GoAction {
    let legal = state.legal_moves();
    if legal.is_empty() || rng.f32() < 0.02 {
        // Occasional pass to avoid infinite games
        return GoAction::Pass;
    }
    let idx = rng.usize(..legal.len());
    GoAction::Place(legal[idx].0, legal[idx].1)
}

// ── Game Runner ────────────────────────────────────────────────

/// Who plays which color for a given game index.
#[derive(Clone, Copy, Debug)]
enum PlayerAssignment {
    /// MCTS plays Black, Random plays White.
    McstBlack,
    /// MCTS plays White, Random plays Black.
    McstWhite,
}

impl PlayerAssignment {
    fn from_index(i: usize) -> Self {
        if i.is_multiple_of(2) {
            Self::McstBlack
        } else {
            Self::McstWhite
        }
    }

    fn mcts_color(self) -> GoCell {
        match self {
            Self::McstBlack => GoCell::Black,
            Self::McstWhite => GoCell::White,
        }
    }
}

/// Result of a single game.
struct GameResult {
    /// Which assignment was used.
    assignment: PlayerAssignment,
    /// Did MCTS win?
    mcts_won: bool,
    /// Final score (Black perspective).
    score: f32,
    /// Total moves played.
    moves: usize,
    /// Time spent on this game.
    duration: std::time::Duration,
}

/// Play one game between MCTS and Random.
fn play_game(assignment: PlayerAssignment, budget: usize, rng: &mut fastrand::Rng) -> GameResult {
    let start = Instant::now();
    let mut state = GoState::new(9);
    let mut replay = GoReplay::new(9, DEFAULT_KOMI);
    let mut moves = 0usize;

    for _ in 0..MAX_MOVES {
        if state.is_terminal() {
            break;
        }

        let legal_count = state.legal_move_count();
        let is_mcts_turn = state.to_play == assignment.mcts_color();

        let action = if is_mcts_turn {
            mcts_select_move(&state, budget, rng)
        } else {
            random_select_move(&state, rng)
        };

        // Apply action
        match &action {
            GoAction::Place(row, col) => {
                let ok = state.play_move(*row, *col);
                debug_assert!(ok, "MCTS selected illegal move ({row},{col})");
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        replay.record(&action, state.to_play.opponent(), legal_count);
        moves += 1;

        // Progress logging every 50 moves
        if moves.is_multiple_of(50) {
            log::debug!(
                "  Move {moves}: {} to play, {} legal",
                state.to_play,
                state.legal_move_count()
            );
        }
    }

    // Force game end if not terminal
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
        moves += 2;
    }

    let score = state.score();
    let winner = state.get_winner();
    let mcts_won = winner == Some(assignment.mcts_color());

    replay.finalize(winner, score);

    let duration = start.elapsed();

    GameResult {
        assignment,
        mcts_won,
        score,
        moves,
        duration,
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let num_games: usize = env::var("GO_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_GAMES);

    let budget: usize = env::var("GO_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BUDGET);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           Go MCTS vs Random — 9×9 Benchmark            ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Games:  {num_games:<6}                                       ║");
    println!("║  Budget: {budget:<6} (MCTS advances/search)                ║");
    println!("║  Board:  9×9, komi={DEFAULT_KOMI}                             ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    let mut rng = fastrand::Rng::with_seed(42);
    let mut results: Vec<GameResult> = Vec::with_capacity(num_games);

    let mut mcts_wins = 0usize;
    let mut total_moves = 0usize;
    let mut total_duration = std::time::Duration::ZERO;

    for i in 0..num_games {
        let assignment = PlayerAssignment::from_index(i);
        let color_label = match assignment {
            PlayerAssignment::McstBlack => "MCTS=Black",
            PlayerAssignment::McstWhite => "MCTS=White",
        };

        print!("  [{:>3}/{}] {color_label:>12} ", i + 1, num_games);
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let result = play_game(assignment, budget, &mut rng);

        let outcome = if result.mcts_won { "WIN" } else { "LOSS" };
        let score_display = if result.score > 0.0 {
            format!("B+{:.1}", result.score)
        } else {
            format!("W+{:.1}", result.score.abs())
        };
        println!(
            "{outcome:>4} {score_display:>8} {:>3} moves ({:.1}s)",
            result.moves,
            result.duration.as_secs_f64()
        );

        if result.mcts_won {
            mcts_wins += 1;
        }
        total_moves += result.moves;
        total_duration += result.duration;
        results.push(result);
    }

    // ── Summary ────────────────────────────────────────────────

    let losses = num_games - mcts_wins;
    let win_rate = mcts_wins as f64 / num_games as f64 * 100.0;
    let avg_moves = total_moves as f64 / num_games as f64;
    let avg_time = total_duration.as_secs_f64() / num_games as f64;
    let moves_per_sec = total_moves as f64 / total_duration.as_secs_f64();

    // Per-color breakdown
    let (mcts_black_wins, mcts_black_total) = results
        .iter()
        .filter(|r| matches!(r.assignment, PlayerAssignment::McstBlack))
        .fold((0usize, 0usize), |(w, t), r| {
            (w + r.mcts_won as usize, t + 1)
        });
    let (mcts_white_wins, mcts_white_total) = results
        .iter()
        .filter(|r| matches!(r.assignment, PlayerAssignment::McstWhite))
        .fold((0usize, 0usize), |(w, t), r| {
            (w + r.mcts_won as usize, t + 1)
        });

    println!();
    println!("════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("════════════════════════════════════════════════════════════");
    println!();
    println!("  MCTS Win Rate: {win_rate:.1}% ({mcts_wins}W / {losses}L)");
    println!("  Avg Moves/Game: {avg_moves:.1}");
    println!("  Avg Time/Game:  {avg_time:.2}s");
    println!("  Moves/sec:      {moves_per_sec:.0}");
    println!();
    println!(
        "  As Black: {}W / {}G ({:.0}%)",
        mcts_black_wins,
        mcts_black_total,
        mcts_black_wins as f64 / mcts_black_total.max(1) as f64 * 100.0
    );
    println!(
        "  As White: {}W / {}G ({:.0}%)",
        mcts_white_wins,
        mcts_white_total,
        mcts_white_wins as f64 / mcts_white_total.max(1) as f64 * 100.0
    );
    println!();
    println!("  MCTS Budget: {budget} advances/search");
    println!("════════════════════════════════════════════════════════════");
}
