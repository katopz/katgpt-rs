//! Bomberman HL Arena — Animated TUI Replay (Plan 033, Task 7)
//!
//! Records tick snapshots during a game round, then replays them with
//! ratatui + crossterm. Two-panel layout: arena grid (emoji) + scoreboard.
//!
//! Controls:
//!   ← / Backspace  — Previous tick
//!   → / Enter      — Next tick
//!   Space           — Toggle auto-play
//!   Home/End        — Jump to start/end
//!   R               — New round (re-generate arena)
//!   Q / Esc         — Quit
//!
//! Run: `cargo run --example bomber_02_tui --features bomber`

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

use katgpt_rs::pruners::bomber::arena::{EMPTY_ARENA, PILLAR_HEAVY_ARENA, STANDARD_ARENA};
use katgpt_rs::pruners::bomber::{
    Alive, ArenaGrid, BombFuse, BomberPlayer, Cell, GameEvent, GreedyPlayer, GridPos, HLPlayer,
    PlayerEntities, PowerUpKind, RandomPlayer, ValidatorPlayer, init_world, init_world_with_arena,
    run_tick, spawn_players,
};

// ── Emoji Constants ────────────────────────────────────────────

const P_EMOJI: [&str; 4] = ["🐰", "🐱", "🐶", "🐵"];
const P_DEAD: &str = "💀";
const WALL_FIXED: &str = "🧱";
const WALL_DESTRUCT: &str = "📦";
const FLOOR: &str = "··";
const BLAST_EMOJI: &str = "💥";
const BOMB_FRESH: &str = "💣";
const BOMB_LOW: &str = "🧨";
const PU_BOMB: &str = "🌠";
const PU_FIRE: &str = "🎇";
const PU_SPEED: &str = "👟";

// ── Timing ─────────────────────────────────────────────────────

const AUTO_STEP_MS: u64 = 300;

// ── CLI ────────────────────────────────────────────────────────

fn parse_map_arg() -> (Option<&'static str>, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut map_preset = None;
    let mut seed = 42u64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--map" if i + 1 < args.len() => {
                i += 1;
                map_preset = match args[i].as_str() {
                    "empty" => Some(EMPTY_ARENA),
                    "standard" => Some(STANDARD_ARENA),
                    "pillar_heavy" => Some(PILLAR_HEAVY_ARENA),
                    other => {
                        eprintln!("Unknown map: {other}. Use: empty, standard, pillar_heavy");
                        std::process::exit(1);
                    }
                };
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
    (map_preset, seed)
}

// ── Snapshot ───────────────────────────────────────────────────

/// A recorded game state for replay.
#[derive(Clone)]
struct TickSnapshot {
    grid: ArenaGrid,
    player_pos: [(i32, i32); 4],
    player_alive: [bool; 4],
    bombs: Vec<((i32, i32), u32)>, // (pos, fuse_remaining)
    blasts: Vec<(i32, i32)>,
    powerups: Vec<((i32, i32), PowerUpKind)>,
    scores: [i32; 4],
    tick: u32,
    _events: Vec<GameEvent>,
}

// ── Round Recording ────────────────────────────────────────────

struct RecordedRound {
    snapshots: Vec<TickSnapshot>,
    final_scores: [i32; 4],
    winner: Option<u8>,
    _total_ticks: u32,
}

fn record_round(
    seed: u64,
    map_preset: Option<&'static str>,
    players: &mut [Box<dyn BomberPlayer>],
    rng: &mut Rng,
) -> RecordedRound {
    let mut world = match map_preset {
        Some(template) => {
            let arena = ArenaGrid::fixed(template).unwrap_or_else(|e| {
                eprintln!("Invalid map preset: {e}");
                std::process::exit(1);
            });
            init_world_with_arena(arena)
        }
        None => init_world(seed),
    };
    let entities = spawn_players(&mut world);

    for p in players.iter_mut() {
        p.reset();
    }

    let mut snapshots = Vec::new();
    let mut all_events: Vec<GameEvent> = Vec::new();

    for _ in 0..katgpt_rs::pruners::bomber::TICK_LIMIT {
        // Snapshot current state
        let snap = capture_snapshot(&mut world, &all_events);
        snapshots.push(snap);

        // Drain previous events
        all_events.clear();
        drain_events(&mut world, &mut all_events);

        // Select actions
        let mut actions = [None; 4];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world
                .get::<katgpt_rs::pruners::bomber::Alive>(entities[i])
                .is_some();
            if alive {
                let grid = world.resource::<ArenaGrid>().clone();
                actions[i] = Some(player.select_action(&grid, pos, &all_events, rng));
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Final snapshot
    drain_events(&mut world, &mut all_events);
    let final_snap = capture_snapshot(&mut world, &all_events);
    snapshots.push(final_snap);

    // Compute final scores
    let mut scores = [0i32; 4];
    let mut survivors = Vec::new();
    for event in &all_events {
        match event {
            GameEvent::PlayerKilled { victim, killer } => {
                scores[*victim as usize] -= 3;
                match killer {
                    Some(k) if *k != *victim => {
                        scores[*k as usize] += 3;
                    }
                    _ => {
                        // Suicide (killer == victim or killer unknown)
                        scores[*victim as usize] -= 2;
                    }
                }
            }
            GameEvent::PowerUpCollected { player, .. } => {
                scores[*player as usize] += 1;
            }
            GameEvent::RoundEnd { survivors: s } => {
                survivors = s.clone();
            }
            _ => {}
        }
    }

    let winner = if survivors.len() == 1 {
        scores[survivors[0] as usize] += 5;
        Some(survivors[0])
    } else {
        None
    };

    let total_ticks = snapshots.last().map(|s| s.tick).unwrap_or(0);

    RecordedRound {
        snapshots,
        final_scores: scores,
        winner,
        _total_ticks: total_ticks,
    }
}

fn capture_snapshot(world: &mut bevy_ecs::world::World, events: &[GameEvent]) -> TickSnapshot {
    use bevy_ecs::prelude::*;

    // Clone/copy all resource data upfront to release immutable borrows before queries.
    // Entity is Copy so `.entities` copies [Entity; 4] and releases the borrow immediately.
    let grid = world.resource::<ArenaGrid>().clone();
    let entity_list: [bevy_ecs::entity::Entity; 4] = world.resource::<PlayerEntities>().entities;
    let scores = world
        .resource::<katgpt_rs::pruners::bomber::ScoreBoard>()
        .scores;
    let tick = world
        .resource::<katgpt_rs::pruners::bomber::TickCounter>()
        .tick;

    // Entity lookups — world.get takes &self, all resource borrows already released
    let mut player_pos = [(0i32, 0i32); 4];
    let mut player_alive = [false; 4];
    for (i, &entity) in entity_list.iter().enumerate() {
        player_pos[i] = world
            .get::<GridPos>(entity)
            .map(|p| (p.x, p.y))
            .unwrap_or((-1, -1));
        player_alive[i] = world.get::<Alive>(entity).is_some();
    }

    // Query filtered — needs &mut World (all immutable borrows released above)
    let mut bombs = Vec::new();
    let mut blasts = Vec::new();
    let mut powerups = Vec::new();

    {
        let mut q = world
            .query_filtered::<(&GridPos, &BombFuse), With<katgpt_rs::pruners::bomber::Bomb>>();
        for (pos, fuse) in q.iter(world) {
            bombs.push(((pos.x, pos.y), fuse.ticks_remaining));
        }
    }
    {
        let mut q = world.query_filtered::<&GridPos, With<katgpt_rs::pruners::bomber::Blast>>();
        for pos in q.iter(world) {
            blasts.push((pos.x, pos.y));
        }
    }
    {
        let mut q =
            world.query_filtered::<(&GridPos, &katgpt_rs::pruners::bomber::PowerUp), ()>();
        for (pos, pu) in q.iter(world) {
            powerups.push(((pos.x, pos.y), pu.kind));
        }
    }

    TickSnapshot {
        grid,
        player_pos,
        player_alive,
        bombs,
        blasts,
        powerups,
        scores,
        tick,
        _events: events.to_vec(),
    }
}

fn drain_events(world: &mut bevy_ecs::world::World, out: &mut Vec<GameEvent>) {
    use bevy_ecs::event::Events;
    let mut ev = world.resource_mut::<Events<GameEvent>>();
    out.extend(ev.drain().collect::<Vec<GameEvent>>());
}

// ── TUI Rendering ──────────────────────────────────────────────

fn render_arena(f: &mut Frame, snap: &TickSnapshot, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    for y in 0..snap.grid.height {
        let mut row = String::new();
        for x in 0..snap.grid.width {
            let pos = (x as i32, y as i32);

            // Check player (alive players take priority over blast visuals)
            // Movement happens after explosions — a player may walk into a
            // blast cell post-explosion and correctly survive (blast is instantaneous)
            let mut player_found = None;
            for (i, &(px, py)) in snap.player_pos.iter().enumerate() {
                if (px, py) == pos {
                    player_found = Some(i);
                    break;
                }
            }
            if let Some(pi) = player_found {
                if snap.player_alive[pi] {
                    row.push_str(P_EMOJI[pi]);
                } else {
                    row.push_str(P_DEAD);
                }
                continue;
            }

            // Check blast (only render if no player occupies this cell)
            if snap.blasts.contains(&pos) {
                row.push_str(BLAST_EMOJI);
                continue;
            }

            // Check bomb
            if let Some((_, fuse)) = snap.bombs.iter().find(|(bp, _)| *bp == pos) {
                row.push_str(if *fuse <= 2 { BOMB_LOW } else { BOMB_FRESH });
                continue;
            }

            // Check powerup
            if let Some((_, kind)) = snap.powerups.iter().find(|(pp, _)| *pp == pos) {
                row.push_str(match kind {
                    PowerUpKind::BombUp => PU_BOMB,
                    PowerUpKind::FireUp => PU_FIRE,
                    PowerUpKind::SpeedUp => PU_SPEED,
                });
                continue;
            }

            // Grid cell
            row.push_str(match snap.grid.get(x as i32, y as i32) {
                Cell::Floor => FLOOR,
                Cell::FixedWall => WALL_FIXED,
                Cell::DestructibleWall => WALL_DESTRUCT,
                Cell::PowerUpHidden(_) => WALL_DESTRUCT,
            });
        }
        lines.push(Line::from(Span::styled(row, Style::default())));
    }

    let block = Block::default().borders(Borders::ALL).title(Span::styled(
        " Arena (13×13) ",
        Style::default().add_modifier(Modifier::BOLD),
    ));
    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

#[expect(
    clippy::too_many_arguments,
    reason = "TUI render helper: each parameter is a distinct widget input; bundling into a struct would add indirection without clarity"
)]
fn render_scoreboard(
    f: &mut Frame,
    snap: &TickSnapshot,
    round: usize,
    total_rounds: usize,
    cursor: usize,
    total_ticks: usize,
    auto_play: bool,
    area: Rect,
) {
    let names = ["Random", "Greedy", "Validator", "HL"];
    let mut lines = Vec::new();

    for i in 0..4 {
        let alive = if snap.player_alive[i] { "✓" } else { "✗" };
        let style = if snap.player_alive[i] {
            Style::default().fg(Color::Green)
        } else {
            Style::default().fg(Color::Red)
        };
        lines.push(Line::from(Span::styled(
            format!(
                " {} P{} {:<10} {:>+4} pts  {}",
                P_EMOJI[i],
                i + 1,
                names[i],
                snap.scores[i],
                alive,
            ),
            style,
        )));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(" Round: {}/{}", round + 1, total_rounds),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::styled(
        format!(" Tick:  {}/{}", cursor, total_ticks.saturating_sub(1)),
        Style::default(),
    )));

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(" Auto: {}", if auto_play { "ON ▶" } else { "OFF ⏸" }),
        Style::default(),
    )));

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " Controls:",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(" ←/→  Step"));
    lines.push(Line::from(" Space Auto-play"));
    lines.push(Line::from(" R     New round"));
    lines.push(Line::from(" Q     Quit"));

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

    // Initial round
    let (map_preset, default_seed) = parse_map_arg();
    let mut rng = Rng::with_seed(default_seed);
    let mut round_idx = 0usize;
    let total_rounds = 5usize;

    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    let mut recorded = record_round(
        default_seed + round_idx as u64,
        map_preset,
        &mut players,
        &mut rng,
    );
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
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(30), Constraint::Length(32)])
                .split(f.area());

            render_arena(f, snap, chunks[0]);
            render_scoreboard(
                f,
                snap,
                round_idx,
                total_rounds,
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
                            round_idx = (round_idx + 1) % total_rounds;
                            recorded = record_round(
                                default_seed + round_idx as u64,
                                map_preset,
                                &mut players,
                                &mut rng,
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
                    round_idx = (round_idx + 1) % total_rounds;
                    recorded = record_round(
                        default_seed + round_idx as u64,
                        map_preset,
                        &mut players,
                        &mut rng,
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

    // Print final standings
    println!();
    println!("═══ Round {} Result ═══", round_idx + 1);
    let names = ["Random", "Greedy", "Validator", "HL"];
    for i in 0..4 {
        println!(
            "  {} P{} {:<10} Score={:+4}",
            P_EMOJI[i],
            i + 1,
            names[i],
            recorded.final_scores[i],
        );
    }
    if let Some(w) = recorded.winner {
        println!("  Winner: {} P{}", P_EMOJI[w as usize], w + 1);
    } else {
        println!("  Winner: Draw");
    }

    Ok(())
}
