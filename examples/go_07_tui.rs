//! Go TUI — AI vs AI Auto-Play Replay (9×9)
//!
//! Records move snapshots during a Go game between two AI players,
//! then replays them with ratatui + crossterm. Two-panel layout:
//! board grid (unicode stones) + scoreboard.
//!
//! Controls:
//!   ← / Backspace  — Previous move
//!   → / Enter      — Next move
//!   Space           — Toggle auto-play
//!   Home/End        — Jump to start/end
//!   R               — New game (re-play)
//!   Q / Esc         — Quit
//!
//! Run: `cargo run --features go --example go_07_tui --black hl --white greedy`
//!
//! Options:
//!   --black <player>  Black player: random, greedy, validator, hl, gzero (default: greedy)
//!   --white <player>  White player: random, greedy, validator, hl, gzero (default: validator)
//!   --size <n>        Board size: 9, 13, 19 (default: 9)
//!   --seed <n>        RNG seed (default: 42)

use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use fastrand::Rng;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{Frame, Terminal};

use katgpt_rs::pruners::go::{
    GoAction, GoCell, GoGZeroPlayer, GoGreedyPlayer, GoHLPlayer, GoMctsPlayer, GoPlayer,
    GoRandomPlayer, GoState, GoValidatorPlayer,
};

// ── Constants ──────────────────────────────────────────────────

const BLACK_STONE: &str = "●";
const WHITE_STONE: &str = "○";

const AUTO_STEP_MS: u64 = 300;
const MAX_MOVES: usize = 300;

// ── Snapshot ───────────────────────────────────────────────────

/// A recorded game state for replay.
#[derive(Clone)]
struct MoveSnapshot {
    board: Vec<GoCell>,
    size: usize,
    to_play: GoCell,
    move_count: u32,
    consecutive_passes: u8,
    captured_black: u32,
    captured_white: u32,
    ko_point: Option<usize>,
    last_move: Option<(usize, usize)>,
    action_taken: Option<GoAction>,
    score_estimate: f32,
}

// ── Recorded Game ──────────────────────────────────────────────

struct RecordedGame {
    snapshots: Vec<MoveSnapshot>,
    final_score: f32,
    winner: Option<GoCell>,
    total_moves: u32,
    black_name: &'static str,
    white_name: &'static str,
    board_size: usize,
}

// ── CLI ────────────────────────────────────────────────────────

fn parse_args() -> (&'static str, &'static str, usize, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut black = "greedy";
    let mut white = "validator";
    let mut size = 9usize;
    let mut seed = 42u64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--black" if i + 1 < args.len() => {
                i += 1;
                black = validate_player(&args[i]);
            }
            "--white" if i + 1 < args.len() => {
                i += 1;
                white = validate_player(&args[i]);
            }
            "--size" if i + 1 < args.len() => {
                i += 1;
                size = args[i].parse().unwrap_or_else(|e| {
                    eprintln!("Bad size: {e}");
                    std::process::exit(1);
                });
                if !matches!(size, 9 | 13 | 19) {
                    eprintln!("Size must be 9, 13, or 19");
                    std::process::exit(1);
                }
            }
            "--seed" if i + 1 < args.len() => {
                i += 1;
                seed = args[i].parse().unwrap_or_else(|e| {
                    eprintln!("Bad seed: {e}");
                    std::process::exit(1);
                });
            }
            _ => {}
        }
        i += 1;
    }
    (black, white, size, seed)
}

fn validate_player(name: &str) -> &'static str {
    match name {
        "random" => "random",
        "greedy" => "greedy",
        "validator" => "validator",
        "hl" => "hl",
        "gzero" => "gzero",
        "mcts" => "mcts",
        _ => {
            eprintln!("Unknown player: {name}. Use: random, greedy, validator, hl, gzero, mcts");
            std::process::exit(1);
        }
    }
}

// ── Player Factory ─────────────────────────────────────────────

fn make_player(name: &str) -> Box<dyn GoPlayer> {
    match name {
        "random" => Box::new(GoRandomPlayer),
        "greedy" => Box::new(GoGreedyPlayer),
        "validator" => Box::new(GoValidatorPlayer),
        "hl" => Box::new(GoHLPlayer::new()),
        "gzero" => Box::new(GoGZeroPlayer::new()),
        "mcts" => Box::new(GoMctsPlayer::new(200, 50)),
        _ => Box::new(GoGreedyPlayer),
    }
}

// ── Snapshot Capture ───────────────────────────────────────────

fn capture_snapshot(state: &GoState, action_taken: Option<GoAction>) -> MoveSnapshot {
    MoveSnapshot {
        board: state.board.clone(),
        size: state.size,
        to_play: state.to_play,
        move_count: state.move_count,
        consecutive_passes: state.consecutive_passes,
        captured_black: state.captured_black,
        captured_white: state.captured_white,
        ko_point: state.ko_point,
        last_move: match &action_taken {
            Some(GoAction::Place(r, c)) => Some((*r, *c)),
            _ => None,
        },
        action_taken,
        score_estimate: state.score(),
    }
}

// ── Game Recording ─────────────────────────────────────────────

fn record_game(
    seed: u64,
    board_size: usize,
    black_name: &'static str,
    white_name: &'static str,
) -> RecordedGame {
    let mut rng = Rng::with_seed(seed);
    let mut state = GoState::new(board_size);
    let mut black_player = make_player(black_name);
    let mut white_player = make_player(white_name);

    black_player.reset();
    white_player.reset();

    let mut snapshots = Vec::new();

    // Initial empty board snapshot
    snapshots.push(capture_snapshot(&state, None));

    for _ in 0..MAX_MOVES {
        let legal_moves = state.legal_moves();
        let current_name = match state.to_play {
            GoCell::Black => black_name,
            GoCell::White => white_name,
            GoCell::Empty => unreachable!(),
        };

        // Skip MCTS for speed in TUI — only use fast players
        let action = match state.to_play {
            GoCell::Black => black_player.select_move(&state, &legal_moves, &mut rng),
            GoCell::White => white_player.select_move(&state, &legal_moves, &mut rng),
            GoCell::Empty => unreachable!(),
        };

        // Apply action
        match &action {
            GoAction::Place(row, col) => {
                let ok = state.play_move(*row, *col);
                debug_assert!(ok, "Player {current_name} made illegal move {action}");
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        // Record snapshot after move
        snapshots.push(capture_snapshot(&state, Some(action)));

        // Check game end
        if state.is_terminal() {
            break;
        }
    }

    // Force game end if max moves reached
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
        snapshots.push(capture_snapshot(&state, Some(GoAction::Pass)));
    }

    let final_score = state.score();
    let winner = match final_score {
        s if s > 0.0 => Some(GoCell::Black),
        s if s < 0.0 => Some(GoCell::White),
        _ => None,
    };
    let total_moves = state.move_count;

    RecordedGame {
        snapshots,
        final_score,
        winner,
        total_moves,
        black_name,
        white_name,
        board_size,
    }
}

// ── Star Points ────────────────────────────────────────────────

fn is_star_point(row: usize, col: usize, size: usize) -> bool {
    match size {
        9 => matches!((row, col), (2, 2) | (2, 6) | (4, 4) | (6, 2) | (6, 6)),
        13 => matches!(
            (row, col),
            (3, 3) | (3, 6) | (3, 9) | (6, 3) | (6, 6) | (6, 9) | (9, 3) | (9, 6) | (9, 9)
        ),
        19 => matches!(
            (row, col),
            (3, 3) | (3, 9) | (3, 15) | (9, 3) | (9, 9) | (9, 15) | (15, 3) | (15, 9) | (15, 15)
        ),
        _ => false,
    }
}

// ── Rendering ──────────────────────────────────────────────────

fn render_board(f: &mut Frame, snap: &MoveSnapshot, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    // Column headers — each col label + space = 2 terminal cols
    let mut header_spans: Vec<Span> = vec![Span::styled("   ", Style::default())];
    for c in 0..snap.size {
        let col_label = match c {
            0..=25 => ((b'A' + c as u8) as char).to_string(),
            _ => format!("{c}"),
        };
        header_spans.push(Span::styled(
            format!("{col_label} "),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ));
    }
    lines.push(Line::from(header_spans));

    // Board rows — 1-char-wide symbols + space, per-cell coloring
    for r in 0..snap.size {
        let mut spans: Vec<Span> = vec![Span::styled(
            format!("{r:>2} "),
            Style::default().fg(Color::DarkGray),
        )];

        for c in 0..snap.size {
            let idx = r * snap.size + c;
            let is_last = snap.last_move == Some((r, c));
            let is_ko = snap.ko_point == Some(idx) && snap.board[idx] == GoCell::Empty;

            let (sym, style) = if is_ko {
                ("x", Style::default().fg(Color::Red))
            } else {
                match snap.board[idx] {
                    GoCell::Black => {
                        if is_last {
                            (
                                "●",
                                Style::default()
                                    .fg(Color::Green)
                                    .add_modifier(Modifier::BOLD),
                            )
                        } else {
                            ("●", Style::default().fg(Color::White))
                        }
                    }
                    GoCell::White => {
                        if is_last {
                            (
                                "○",
                                Style::default()
                                    .fg(Color::Green)
                                    .add_modifier(Modifier::BOLD),
                            )
                        } else {
                            ("○", Style::default().fg(Color::Gray))
                        }
                    }
                    GoCell::Empty => {
                        if is_star_point(r, c, snap.size) {
                            ("+", Style::default().fg(Color::DarkGray))
                        } else {
                            ("·", Style::default().fg(Color::DarkGray))
                        }
                    }
                }
            };

            spans.push(Span::styled(format!("{sym} "), style));
        }

        lines.push(Line::from(spans));
    }

    let title = format!(" Go {}×{} ", snap.size, snap.size);
    let block = Block::default().borders(Borders::ALL).title(Span::styled(
        title,
        Style::default().add_modifier(Modifier::BOLD),
    ));
    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

fn render_scoreboard(
    f: &mut Frame,
    snap: &MoveSnapshot,
    game: &RecordedGame,
    cursor: usize,
    total_snaps: usize,
    auto_play: bool,
    area: Rect,
) {
    let mut lines = Vec::new();

    // Black player
    lines.push(Line::from(Span::styled(
        format!(" {} Black ({})", BLACK_STONE, game.black_name),
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::styled(
        format!("   Captures: {}", snap.captured_black),
        Style::default().fg(Color::White),
    )));

    lines.push(Line::from(""));

    // White player
    lines.push(Line::from(Span::styled(
        format!(" {} White ({})", WHITE_STONE, game.white_name),
        Style::default()
            .fg(Color::Gray)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::styled(
        format!("   Captures: {}", snap.captured_white),
        Style::default().fg(Color::Gray),
    )));

    lines.push(Line::from(""));

    // Current turn
    let turn_emoji = match snap.to_play {
        GoCell::Black => BLACK_STONE,
        GoCell::White => WHITE_STONE,
        GoCell::Empty => "?",
    };
    let turn_name = match snap.to_play {
        GoCell::Black => game.black_name,
        GoCell::White => game.white_name,
        GoCell::Empty => "?",
    };
    let turn_style = if snap.to_play == GoCell::Black {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(Color::Gray)
    };
    lines.push(Line::from(Span::styled(
        format!(" Turn: {turn_emoji} {turn_name}"),
        turn_style.add_modifier(Modifier::BOLD),
    )));

    lines.push(Line::from(""));

    // Move info
    lines.push(Line::from(Span::styled(
        format!(" Move: {}/{}", snap.move_count, game.total_moves),
        Style::default(),
    )));
    lines.push(Line::from(Span::styled(
        format!(" Passes: {}/2", snap.consecutive_passes),
        Style::default(),
    )));

    // Last action
    let action_str = match &snap.action_taken {
        Some(GoAction::Place(r, c)) => {
            let col_label = match c {
                0..=25 => ((b'A' + *c as u8) as char).to_string(),
                _ => format!("{c}"),
            };
            format!(" Last: {col_label}{}", r)
        }
        Some(GoAction::Pass) => " Last: Pass".to_string(),
        None => " Last: —".to_string(),
    };
    lines.push(Line::from(Span::styled(action_str, Style::default())));

    lines.push(Line::from(""));

    // Score estimate
    let score_label = if snap.consecutive_passes >= 2 {
        "Final"
    } else {
        "Est."
    };
    let score_str = format!(" Score {score_label}: {:+.1}", snap.score_estimate);
    let score_style = match snap.score_estimate {
        s if s > 0.5 => Style::default().fg(Color::White),
        s if s < -0.5 => Style::default().fg(Color::Gray),
        _ => Style::default().fg(Color::Yellow),
    };
    lines.push(Line::from(Span::styled(score_str, score_style)));

    // Winner if game over
    if snap.consecutive_passes >= 2 {
        let winner_str = match snap.score_estimate {
            s if s > 0.0 => format!(" Winner: {BLACK_STONE} Black (+{s:.1})"),
            s if s < 0.0 => format!(" Winner: {WHITE_STONE} White (+{:.1})", -s),
            _ => " Result: Jigo (Draw)".to_string(),
        };
        let winner_style = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);
        lines.push(Line::from(Span::styled(winner_str, winner_style)));
    }

    lines.push(Line::from(""));

    // Auto-play
    lines.push(Line::from(Span::styled(
        format!(" Auto: {}", if auto_play { "ON ▶" } else { "OFF ⏸" }),
        Style::default(),
    )));

    // Playback position
    lines.push(Line::from(Span::styled(
        format!(" Frame: {}/{}", cursor, total_snaps.saturating_sub(1)),
        Style::default(),
    )));

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " Controls:",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(" ←/→    Step"));
    lines.push(Line::from(" Space   Auto-play"));
    lines.push(Line::from(" R       New game"));
    lines.push(Line::from(" Q       Quit"));

    let block = Block::default().borders(Borders::ALL).title(Span::styled(
        " Scoreboard ",
        Style::default().add_modifier(Modifier::BOLD),
    ));
    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

// ── Main Loop ──────────────────────────────────────────────────

fn main() -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Parse CLI args
    let (black_name, white_name, board_size, default_seed) = parse_args();

    // Record initial game
    let mut game_idx = 0usize;
    let mut recorded = record_game(default_seed, board_size, black_name, white_name);
    let mut cursor = 0usize;
    let mut auto_play = false;

    // Event loop
    loop {
        let total_snaps = recorded.snapshots.len();
        let snap = recorded
            .snapshots
            .get(cursor)
            .unwrap_or_else(|| recorded.snapshots.last().unwrap());

        terminal.draw(|f| {
            let sidebar_width = match board_size {
                9 => 30,
                13 => 30,
                19 => 30,
                _ => 30,
            };
            let board_min = match board_size {
                9 => 30,
                13 => 35,
                19 => 45,
                _ => 30,
            };

            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Min(board_min),
                    Constraint::Length(sidebar_width),
                ])
                .split(f.area());

            render_board(f, snap, chunks[0]);
            render_scoreboard(
                f,
                snap,
                &recorded,
                cursor,
                total_snaps,
                auto_play,
                chunks[1],
            );
        })?;

        // Auto-play
        if auto_play && cursor < total_snaps.saturating_sub(1) {
            let dur = Duration::from_millis(AUTO_STEP_MS);
            if event::poll(dur)? {
                if let Event::Key(key) = event::read()?
                    && key.kind == KeyEventKind::Press
                {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char(' ') => auto_play = !auto_play,
                        KeyCode::Char('r') => {
                            game_idx += 1;
                            recorded = record_game(
                                default_seed + game_idx as u64,
                                board_size,
                                black_name,
                                white_name,
                            );
                            cursor = 0;
                        }
                        _ => {}
                    }
                }
            } else {
                cursor = (cursor + 1).min(total_snaps.saturating_sub(1));
                if cursor >= total_snaps.saturating_sub(1) {
                    auto_play = false;
                }
            }
            continue;
        }

        // Wait for key
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Char(' ') => auto_play = !auto_play,
                KeyCode::Right | KeyCode::Enter => {
                    cursor = (cursor + 1).min(total_snaps.saturating_sub(1));
                }
                KeyCode::Left | KeyCode::Backspace => {
                    cursor = cursor.saturating_sub(1);
                }
                KeyCode::Home => cursor = 0,
                KeyCode::End => cursor = total_snaps.saturating_sub(1),
                KeyCode::Char('r') => {
                    game_idx += 1;
                    recorded = record_game(
                        default_seed + game_idx as u64,
                        board_size,
                        black_name,
                        white_name,
                    );
                    cursor = 0;
                }
                _ => {}
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Print final result
    println!();
    println!("═══ Game {} Result ═══", game_idx + 1);
    println!(
        "  {} Black ({}) vs {} White ({})",
        BLACK_STONE, black_name, WHITE_STONE, white_name
    );
    println!("  Board: {}×{}", recorded.board_size, recorded.board_size);
    println!("  Moves: {}", recorded.total_moves);
    println!(
        "  Captures: {} by Black, {} by White",
        recorded
            .snapshots
            .last()
            .map(|s| s.captured_black)
            .unwrap_or(0),
        recorded
            .snapshots
            .last()
            .map(|s| s.captured_white)
            .unwrap_or(0),
    );
    println!("  Score: {:+.1}", recorded.final_score);
    match recorded.winner {
        Some(GoCell::Black) => println!("  Winner: {} Black", BLACK_STONE),
        Some(GoCell::White) => println!("  Winner: {} White", WHITE_STONE),
        None => println!("  Result: Jigo (Draw)"),
        Some(GoCell::Empty) => unreachable!(),
    }

    Ok(())
}
