//! Go self-play replay generation — outputs per-move JSONL samples for riir-ai LoRA training.
//!
//! Plan 271 T2.2: Run N self-play games and write one `JsonlGoSample` per move as JSONL.
//!
//! ## Usage
//!
//! ```sh
//! # Default: 100 games, 9x9 board, quality threshold 0.5 (winner only)
//! cargo run --features go --example go_replay_gen
//!
//! # Custom: 500 games, 13x13 board, all moves
//! cargo run --features go --example go_replay_gen -- --games 500 --board-size 13 --quality-threshold 0.0
//!
//! # With G-Zero self-play (template-based) instead of random
//! cargo run --features go --example go_replay_gen -- --player gzero --games 200
//! ```
//!
//! ## Output
//!
//! `output/replays/go_replay_{timestamp}.jsonl` — one JSON object per line.
//! Each line is a `JsonlGoSample` with board state, action, player, quality, legal moves, and BLAKE3 checksum.

use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fastrand::Rng;

use katgpt_rs::pruners::game_state::GameState;
use katgpt_rs::pruners::go::replay_writer::{GameSampleCollector, GoReplayWriter};
use katgpt_rs::pruners::go::state::GoState;
use katgpt_rs::pruners::go::types::{GoAction, GoCell};

// ── CLI Config ─────────────────────────────────────────────────

/// Player strategy for self-play.
#[derive(Clone, Copy, Debug)]
enum PlayerType {
    /// Random legal move selection.
    Random,
    /// G-Zero template-based self-play with greedy move selection.
    GZero,
}

impl std::str::FromStr for PlayerType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "random" => Ok(Self::Random),
            "gzero" => Ok(Self::GZero),
            _ => Err(format!(
                "Unknown player type: {s} (expected: random, gzero)"
            )),
        }
    }
}

/// Parsed CLI arguments.
struct Config {
    board_size: usize,
    game_count: usize,
    quality_threshold: f32,
    player: PlayerType,
    output_dir: PathBuf,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config = Config {
        board_size: 9,
        game_count: 100,
        quality_threshold: 0.5,
        player: PlayerType::Random,
        output_dir: PathBuf::from("output/replays"),
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--board-size" | "-b" => {
                i += 1;
                config.board_size = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid board size: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--games" | "-n" => {
                i += 1;
                config.game_count = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid game count: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--quality-threshold" | "-q" => {
                i += 1;
                config.quality_threshold = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid quality threshold: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--player" | "-p" => {
                i += 1;
                config.player = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid player type: {}", args[i]);
                    std::process::exit(1);
                });
            }
            "--output" | "-o" => {
                i += 1;
                config.output_dir = PathBuf::from(&args[i]);
            }
            "--help" | "-h" => {
                eprintln!("Go Self-Play Replay Generator (Plan 271 T2.2)");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --board-size, -b   Board size (default: 9)");
                eprintln!("  --games, -n        Number of games (default: 100)");
                eprintln!("  --quality-threshold, -q  Min quality to write (default: 0.5)");
                eprintln!("  --player, -p       Player type: random, gzero (default: random)");
                eprintln!("  --output, -o       Output directory (default: output/replays)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown option: {other}");
                eprintln!("Use --help for usage");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    config
}

// ── Player Strategies ──────────────────────────────────────────

/// Select a random legal move.
fn select_random_move(legal_moves: &[(usize, usize)], rng: &mut Rng) -> GoAction {
    if legal_moves.is_empty() {
        return GoAction::Pass;
    }
    let (r, c) = legal_moves[rng.usize(..legal_moves.len())];
    GoAction::Place(r, c)
}

/// Select a greedy move (highest stone count delta + center proximity).
///
/// Simple heuristic: prefer captures, then center proximity, then liberties.
/// Uses sigmoid for score blending (NOT softmax).
fn select_greedy_move(state: &GoState, legal_moves: &[(usize, usize)]) -> GoAction {
    if legal_moves.is_empty() {
        return GoAction::Pass;
    }

    let center = (state.size - 1) as f32 / 2.0;
    let me = state.to_play;

    let best = legal_moves
        .iter()
        .map(|&(r, c)| {
            // Simulate move
            let action = GoAction::Place(r, c);
            let next = state.advance(&action, me.player_id());

            // Capture score
            let caps = match me {
                GoCell::Black => next.captured_black.saturating_sub(state.captured_black),
                GoCell::White => next.captured_white.saturating_sub(state.captured_white),
                GoCell::Empty => 0,
            };

            // Center proximity via sigmoid
            let dist = ((r as f32 - center).abs() + (c as f32 - center).abs()) / 2.0;
            let max_dist = center;
            let center_score = if max_dist > 0.0 {
                1.0 / (1.0 + (-2.0 * (1.0 - dist / max_dist)).exp()) // sigmoid
            } else {
                1.0
            };

            // Liberty count: count empty neighbors of placed stone
            let mut liberty_count = 0usize;
            if next.at(r, c).is_stone() {
                if r > 0 && next.at(r - 1, c) == GoCell::Empty {
                    liberty_count += 1;
                }
                if r + 1 < state.size && next.at(r + 1, c) == GoCell::Empty {
                    liberty_count += 1;
                }
                if c > 0 && next.at(r, c - 1) == GoCell::Empty {
                    liberty_count += 1;
                }
                if c + 1 < state.size && next.at(r, c + 1) == GoCell::Empty {
                    liberty_count += 1;
                }
            }

            let score = caps as f32 * 10.0 + center_score + liberty_count as f32 * 0.5;
            (r, c, score)
        })
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    match best {
        Some((r, c, _)) => GoAction::Place(r, c),
        None => GoAction::Pass,
    }
}

// ── Game Loop ──────────────────────────────────────────────────

/// Result of a single game.
struct GameResult {
    winner: Option<GoCell>,
    total_moves: usize,
    samples_written: usize,
}

/// Play a single self-play game and write per-move samples.
///
/// Uses the deprecated JSON `finalize_and_write` until the riir-ai training
/// pipeline migrates to `finalize_and_write_binary` (see method deprecation
/// note). Tracked as part of the riir-ai migration, not a katgpt-rs fix.
#[allow(deprecated)]
fn play_game(
    config: &Config,
    rng: &mut Rng,
    writer: &mut GoReplayWriter,
    game_num: usize,
) -> GameResult {
    let max_moves = config.board_size * config.board_size * 3;
    let mut state = GoState::new(config.board_size);
    let mut collector = GameSampleCollector::new(config.board_size);
    let mut moves_played = 0usize;

    while !state.is_terminal() && moves_played < max_moves {
        let legal_moves = state.legal_moves();

        if legal_moves.is_empty() {
            collector.record_move(&state, &GoAction::Pass, moves_played as u32 + 1);
            state.play_pass();
            moves_played += 1;
            continue;
        }

        let action = match config.player {
            PlayerType::Random => select_random_move(&legal_moves, rng),
            PlayerType::GZero => select_greedy_move(&state, &legal_moves),
        };

        collector.record_move(&state, &action, moves_played as u32 + 1);

        match &action {
            GoAction::Place(r, c) => {
                state.play_move(*r, *c);
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        moves_played += 1;
    }

    // Force end if not terminal
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
    }

    let winner = state.get_winner();

    let samples_written = collector
        .finalize_and_write(winner, writer, config.quality_threshold)
        .unwrap_or_else(|e| {
            eprintln!("[ERROR] Game {game_num}: write failed: {e}");
            0
        });

    GameResult {
        winner,
        total_moves: moves_played,
        samples_written,
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let config = parse_args();

    eprintln!("=== Go Self-Play Replay Generator (Plan 271 T2.2) ===");
    eprintln!(
        "Board: {}×{}, Games: {}, Player: {:?}",
        config.board_size, config.board_size, config.game_count, config.player,
    );
    eprintln!("Quality threshold: {}", config.quality_threshold);

    let (vocab_size, block_size) =
        katgpt_rs::pruners::go::replay_writer::JsonlGoSample::token_dims(config.board_size);
    eprintln!("Token dims: vocab={vocab_size}, block={block_size}");

    // Create output file with timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("go_replay_{timestamp}.jsonl");
    let output_path = config.output_dir.join(&filename);

    let mut writer = GoReplayWriter::create(&output_path, config.board_size).unwrap_or_else(|e| {
        eprintln!("Failed to create output file: {e}");
        std::process::exit(1);
    });

    let mut rng = Rng::with_seed(42);
    let start = Instant::now();

    let mut black_wins = 0usize;
    let mut white_wins = 0usize;
    let mut draws = 0usize;
    let mut total_moves = 0usize;
    let mut total_samples = 0usize;

    for game_num in 1..=config.game_count {
        let result = play_game(&config, &mut rng, &mut writer, game_num);

        match result.winner {
            Some(GoCell::Black) => black_wins += 1,
            Some(GoCell::White) => white_wins += 1,
            _ => draws += 1,
        }

        total_moves += result.total_moves;
        total_samples += result.samples_written;

        if game_num % 10 == 0 || game_num == config.game_count {
            let elapsed = start.elapsed().as_secs_f32();
            let gps = game_num as f32 / elapsed.max(0.001);
            eprintln!(
                "  [{game_num:4}/{game_count}] B:{black_wins} W:{white_wins} D:{draws} | \
                 samples:{total_samples} moves:{total_moves} | {gps:.1} games/s",
                game_count = config.game_count,
            );
        }
    }

    writer.flush().unwrap();

    let elapsed = start.elapsed();
    let file_size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);

    eprintln!();
    eprintln!("=== Results ===");
    eprintln!("Output: {}", output_path.display());
    eprintln!("File size: {:.2} MB", file_size as f64 / 1e6);
    eprintln!(
        "Games: {} ({:.1} games/s)",
        config.game_count,
        config.game_count as f64 / elapsed.as_secs_f64().max(0.001)
    );
    eprintln!(
        "Black wins: {black_wins} ({:.1}%)",
        black_wins as f64 / config.game_count as f64 * 100.0
    );
    eprintln!(
        "White wins: {white_wins} ({:.1}%)",
        white_wins as f64 / config.game_count as f64 * 100.0
    );
    eprintln!("Draws: {draws}");
    eprintln!(
        "Total samples: {total_samples} (quality >= {})",
        config.quality_threshold
    );
    eprintln!("Total moves: {total_moves}");
    eprintln!("Elapsed: {:.2}s", elapsed.as_secs_f64());
}
