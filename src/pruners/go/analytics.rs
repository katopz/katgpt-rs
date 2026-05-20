//! Go PGD Game Analytics — modelless feature extraction from `GoReplay` data.
//!
//! Plan 081 (Modelless Path): Extracts analytics features without requiring
//! a neural network model. Uses `GoHeuristic` for per-move evaluation,
//! greedy scoring for coincidence rate computation, and `categorize_move()`
//! for player style vectors.

use serde::{Deserialize, Serialize};

use super::players::{categorize_move, greedy_score};
use super::replay::{GoActionSer, GoCellSer, GoReplay, MoveRecord};
use super::state::{GoHeuristic, GoState};

use crate::pruners::game_state::StateHeuristic;

// ── GoGameAnalytics ────────────────────────────────────────────

/// Analytics extracted from a completed Go game replay.
///
/// All heuristic-based features are modelless — no neural network required.
/// Useful for player profiling, game quality assessment, and PGD training
/// signal augmentation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoGameAnalytics {
    /// Win-rate trace evaluated at each move (Black perspective, player_id=0).
    ///
    /// Each entry is `GoHeuristic.evaluate(&state_before_move, 0)`.
    pub win_rate_trace: Vec<f32>,
    /// Territory score trace at each move (Black perspective).
    ///
    /// Each entry is `state.score()` before the move was applied.
    pub score_trace: Vec<f32>,
    /// Percentage of moves played after the game was effectively decided.
    ///
    /// Computed as `(total_moves - garbage_start_move) / total_moves`.
    pub garbage_move_ratio: f32,
    /// 0-based move index where the game effectively ended, if detected.
    ///
    /// Determined by a moving-average threshold on the win-rate trace.
    pub garbage_start_move: Option<usize>,
    /// Number of lead changes (zero-crossings in `win_rate_trace`).
    pub unstable_round_count: usize,
    /// Average heuristic delta per move for the losing player.
    ///
    /// Measures how much evaluation ground the loser conceded each turn.
    pub mean_loss_win_rate: f32,
    /// Percentage of Place moves that agree with the greedy player's choice.
    ///
    /// Only `Place` moves are counted; `Pass` moves are excluded.
    pub coincidence_rate: f32,
    /// Normalized histogram of move categories (style vector, sums to 1.0).
    ///
    /// Index maps to [`GoMoveCategory`] discriminant (0..8).
    pub category_distribution: [f32; 8],
    /// Total number of moves in the replay.
    pub total_moves: usize,
    /// Winner of the game (copied from replay).
    pub winner: Option<GoCellSer>,
}

// ── compute_analytics ──────────────────────────────────────────

/// Compute full analytics from a game replay.
///
/// Replays all moves on a fresh board, collecting heuristic evaluations,
/// territory scores, greedy agreement, and move category distributions.
///
/// # Edge Cases
///
/// - Empty replay (0 moves) returns zeroed analytics with empty traces.
/// - Games with only `Pass` moves produce `coincidence_rate = 0.0`.
pub fn compute_analytics(replay: &GoReplay) -> GoGameAnalytics {
    if replay.moves.is_empty() {
        return GoGameAnalytics {
            win_rate_trace: Vec::new(),
            score_trace: Vec::new(),
            garbage_move_ratio: 0.0,
            garbage_start_move: None,
            unstable_round_count: 0,
            mean_loss_win_rate: 0.0,
            coincidence_rate: 0.0,
            category_distribution: [0.0; 8],
            total_moves: 0,
            winner: replay.winner,
        };
    }

    let mut state = GoState::with_komi(replay.size, replay.komi);
    let heuristic = GoHeuristic;

    let mut win_rate_trace: Vec<f32> = Vec::with_capacity(replay.moves.len());
    let mut score_trace: Vec<f32> = Vec::with_capacity(replay.moves.len());

    let mut category_counts: [f32; 8] = [0.0; 8];
    let mut coincidence_count: usize = 0;
    let mut place_move_count: usize = 0;

    for record in &replay.moves {
        // ── Evaluate BEFORE applying the move ──
        let win_rate = heuristic.evaluate(&state, 0); // Black perspective
        let score = state.score();

        win_rate_trace.push(win_rate);
        score_trace.push(score);

        // ── Analyze the move (before applying) ──
        if let GoActionSer::Place { row, col } = &record.action {
            place_move_count += 1;

            // Find greedy best move among all legal moves
            let legal = state.legal_moves();
            let mut best_score: f32 = f32::NEG_INFINITY;
            let mut best_move: Option<(usize, usize)> = None;

            for &(r, c) in &legal {
                let gs = greedy_score(&state, r, c);
                if gs > best_score {
                    best_score = gs;
                    best_move = Some((r, c));
                }
            }

            // Check coincidence with greedy choice
            if best_move == Some((*row, *col)) {
                coincidence_count += 1;
            }

            // Categorize move into style histogram
            let cat = categorize_move(&state, *row, *col);
            category_counts[cat as usize] += 1.0;
        }

        // ── Apply the move to state ──
        match &record.action {
            GoActionSer::Place { row, col } => {
                state.play_move(*row, *col);
            }
            GoActionSer::Pass => {
                state.play_pass();
            }
        }
    }

    // ── Post-processing ──
    let garbage_start_move = detect_garbage_moves(&win_rate_trace, 0.85, 4);
    let unstable_round_count = detect_unstable_rounds(&win_rate_trace);
    let mean_loss_win_rate = compute_mlwr(&win_rate_trace, &replay.moves, replay.winner);

    let garbage_move_ratio = match garbage_start_move {
        Some(start) if replay.moves.len() > start => {
            (replay.moves.len() - start) as f32 / replay.moves.len() as f32
        }
        _ => 0.0,
    };

    // Normalize category distribution to sum to 1.0
    let category_distribution = {
        let total: f32 = category_counts.iter().sum();
        if total > 0.0 {
            std::array::from_fn(|i| category_counts[i] / total)
        } else {
            [0.0; 8]
        }
    };

    let coincidence_rate = if place_move_count > 0 {
        coincidence_count as f32 / place_move_count as f32
    } else {
        0.0
    };

    GoGameAnalytics {
        win_rate_trace,
        score_trace,
        garbage_move_ratio,
        garbage_start_move,
        unstable_round_count,
        mean_loss_win_rate,
        coincidence_rate,
        category_distribution,
        total_moves: replay.moves.len(),
        winner: replay.winner,
    }
}

// ── detect_garbage_moves ───────────────────────────────────────

/// Detect when the game enters a "stable zone" — one player has effectively won.
///
/// Finds the first move index where the moving average of the heuristic
/// stays above `threshold` (in absolute value) for the remainder of the game.
///
/// # Algorithm
///
/// For each position `i` in `0..=trace.len()-window`:
/// 1. Compute average of `trace[i..i+window]`.
/// 2. If `|avg| >= threshold`, verify all subsequent windows also satisfy this.
/// 3. Return the first such `i`.
///
/// Returns `None` if the game never stabilizes or the trace is shorter than
/// the window.
pub fn detect_garbage_moves(trace: &[f32], threshold: f32, window: usize) -> Option<usize> {
    if trace.len() < window || window == 0 {
        return None;
    }

    let max_start = trace.len() - window;

    for i in 0..=max_start {
        let avg: f32 = trace[i..i + window].iter().sum::<f32>() / window as f32;

        if avg.abs() >= threshold {
            // Verify all subsequent windows also satisfy the threshold
            let all_stable = ((i + 1)..=max_start).all(|j| {
                let sub_avg: f32 = trace[j..j + window].iter().sum::<f32>() / window as f32;
                sub_avg.abs() >= threshold
            });

            if all_stable {
                return Some(i);
            }
        }
    }

    None
}

// ── detect_unstable_rounds ─────────────────────────────────────

/// Count zero-crossings in the win-rate trace.
///
/// A zero-crossing occurs when consecutive values have different signs,
/// indicating a lead change between Black and White. Zero-to-nonzero
/// transitions are counted as crossings; zero-to-zero is not.
pub fn detect_unstable_rounds(trace: &[f32]) -> usize {
    if trace.len() < 2 {
        return 0;
    }

    let mut count: usize = 0;

    for i in 0..(trace.len() - 1) {
        let sa = sign_f32(trace[i]);
        let sb = sign_f32(trace[i + 1]);

        if sa != sb {
            count += 1;
        }
    }

    count
}

/// Returns -1 for negative, +1 for positive, 0 for zero.
#[inline]
fn sign_f32(x: f32) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

// ── compute_mlwr ───────────────────────────────────────────────

/// Compute Mean Loss Win Rate (MLWR) for the losing player.
///
/// For each move made by the losing player, measures the absolute change
/// in the heuristic evaluation. A high MLWR indicates the losing player
/// was consistently losing ground on their turns.
///
/// # Returns
///
/// - `0.0` if there is no winner (draw or incomplete game).
/// - `0.0` if the losing player has no moves with a predecessor trace value.
/// - Otherwise, the average `|trace[i] - trace[i-1]|` over the loser's moves.
pub fn compute_mlwr(trace: &[f32], moves: &[MoveRecord], winner: Option<GoCellSer>) -> f32 {
    let winner_cell = match winner {
        Some(w) => w,
        None => return 0.0,
    };

    // Determine the loser
    let loser = match winner_cell {
        GoCellSer::Black => GoCellSer::White,
        GoCellSer::White => GoCellSer::Black,
    };

    let mut total_delta: f32 = 0.0;
    let mut count: usize = 0;

    for i in 0..moves.len() {
        if moves[i].player == loser && i > 0 {
            let delta = (trace[i] - trace[i - 1]).abs();
            total_delta += delta;
            count += 1;
        }
    }

    if count > 0 {
        total_delta / count as f32
    } else {
        0.0
    }
}
