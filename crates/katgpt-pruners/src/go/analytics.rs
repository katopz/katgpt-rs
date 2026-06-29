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

use crate::game_state::StateHeuristic;

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
    let mut legal_buf: Vec<(usize, usize)> = Vec::with_capacity(replay.size * replay.size);

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
            state.legal_moves_into(&mut legal_buf);
            let mut best_score: f32 = f32::NEG_INFINITY;
            let mut best_move: Option<(usize, usize)> = None;

            for &(r, c) in &legal_buf {
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

// ── Data Bridge: Natsukaze → Analytics (T0) ────────────────────

/// Action type for raw Go samples, mirroring riir-gpu's `GoActionType`.
///
/// This decoupled type allows `samples_to_replay()` to work without
/// a dependency on the riir-gpu crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawGoAction {
    /// Place a stone at (row, col).
    Place { row: usize, col: usize },
    /// Pass turn.
    Pass,
}

/// Raw Go training sample, mirroring riir-gpu's `GoGameSample`.
///
/// Decoupled from the riir-gpu crate so that `samples_to_replay()`
/// can live in katgpt-rs without a cross-project dependency.
/// Convert from `GoGameSample` trivially:
///
/// ```ignore
/// let raw = RawGoSample {
///     board: sample.board.clone(),
///     size: sample.size,
///     action: match sample.action_type {
///         GoActionType::Place { row, col } => RawGoAction::Place { row, col },
///         GoActionType::Pass => RawGoAction::Pass,
///     },
///     quality: sample.quality,
///     move_number: sample.move_number,
///     legal_moves: sample.legal_moves,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct RawGoSample {
    /// Board state: `size * size` cells (0=Empty, 1=Black, 2=White).
    pub board: Vec<u8>,
    /// Board dimension (typically 9).
    pub size: usize,
    /// Action taken in this state.
    pub action: RawGoAction,
    /// Outcome quality: 0.0 (loss) to 1.0 (win) for the mover.
    pub quality: f32,
    /// Move number within the game (1-based).
    pub move_number: usize,
    /// Legal move count at this position.
    pub legal_moves: usize,
}

/// Split a flat sample stream into per-game groups.
///
/// Detects game boundaries where `move_number` drops back to 1.
/// Returns a Vec of games, where each game is a Vec of samples in order.
///
/// # Edge Cases
///
/// - Empty input returns empty Vec.
/// - Samples with `move_number == 0` are skipped (invalid).
pub fn split_samples_into_games(samples: &[RawGoSample]) -> Vec<Vec<&RawGoSample>> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut games: Vec<Vec<&RawGoSample>> = Vec::new();
    let mut current: Vec<&RawGoSample> = Vec::new();

    for sample in samples {
        if sample.move_number == 0 {
            continue; // Skip invalid
        }

        // New game starts when move_number drops to 1 and we already have samples
        if sample.move_number == 1 && !current.is_empty() {
            games.push(current);
            current = Vec::new();
        }

        current.push(sample);
    }

    if !current.is_empty() {
        games.push(current);
    }

    games
}

/// Convert raw Go samples (from Natsukaze `.flat.zip` pipeline) to a `GoReplay`.
///
/// Reconstructs a `MoveRecord` sequence from per-move sample data.
/// Assumes samples belong to a single game (use `split_samples_into_games` first).
///
/// # Algorithm
///
/// 1. Player is inferred from `move_number` parity: odd = Black, even = White.
/// 2. Winner is inferred from the last sample's `quality`:
///    - `quality >= 0.5` → last mover won
///    - `quality < 0.5` → last mover lost
/// 3. Final score is computed by replaying all moves on a fresh `GoState`.
///
/// # Errors
///
/// Returns `Err` if:
/// - Samples are empty
/// - Move numbers are non-sequential (gap detected)
/// - Board size is inconsistent across samples
///
/// # Example
///
/// ```ignore
/// use katgpt_rs::pruners::go::analytics::{RawGoSample, RawGoAction, samples_to_replay};
///
/// let samples = vec![
///     RawGoSample {
///         board: vec![0; 81],
///         size: 9,
///         action: RawGoAction::Place { row: 4, col: 4 },
///         quality: 1.0,
///         move_number: 1,
///         legal_moves: 80,
///     },
/// ];
///
/// let replay = samples_to_replay(&samples, 7.5).unwrap();
/// assert_eq!(replay.moves.len(), 1);
/// ```
pub fn samples_to_replay(samples: &[RawGoSample], komi: f32) -> Result<GoReplay, String> {
    if samples.is_empty() {
        return Err("Cannot convert empty samples to replay".to_string());
    }

    let board_size = samples[0].size;
    let mut moves: Vec<MoveRecord> = Vec::with_capacity(samples.len());

    for (i, sample) in samples.iter().enumerate() {
        // Validate board size consistency
        if sample.size != board_size {
            return Err(format!(
                "Inconsistent board size at sample {i}: expected {board_size}, got {}",
                sample.size
            ));
        }

        // Validate sequential move numbers (allow gaps from filtered samples)
        let expected = if moves.is_empty() {
            sample.move_number
        } else {
            moves.last().unwrap().move_number as usize + 1
        };

        if sample.move_number != expected {
            return Err(format!(
                "Non-sequential move_number at sample {i}: expected {expected}, got {}",
                sample.move_number
            ));
        }

        // Infer player from move_number parity: odd = Black, even = White
        let player = if sample.move_number % 2 == 1 {
            GoCellSer::Black
        } else {
            GoCellSer::White
        };

        // Convert action
        let action = match &sample.action {
            RawGoAction::Place { row, col } => GoActionSer::Place {
                row: *row,
                col: *col,
            },
            RawGoAction::Pass => GoActionSer::Pass,
        };

        moves.push(MoveRecord {
            action,
            player,
            move_number: sample.move_number as u32,
            legal_move_count: sample.legal_moves,
        });
    }

    // Infer winner from last sample's quality
    let last = samples.last().unwrap();
    let last_player = if last.move_number % 2 == 1 {
        GoCellSer::Black
    } else {
        GoCellSer::White
    };

    let winner = if last.quality >= 0.5 {
        Some(last_player)
    } else {
        // Last mover lost → opponent won
        Some(match last_player {
            GoCellSer::Black => GoCellSer::White,
            GoCellSer::White => GoCellSer::Black,
        })
    };

    // Compute final score by replaying all moves
    let final_score = {
        let mut state = GoState::with_komi(board_size, komi);
        for record in &moves {
            match &record.action {
                GoActionSer::Place { row, col } => {
                    state.play_move(*row, *col);
                }
                GoActionSer::Pass => {
                    state.play_pass();
                }
            }
        }
        state.score()
    };

    Ok(GoReplay {
        size: board_size,
        komi,
        moves,
        winner,
        final_score,
    })
}
