//! Balanced replay generator for Bomberman LoRA training (v2).
//! Captures ALL players (winners AND losers) with enriched features.
//! Output: output/replays_v2/bomber_replay_v2_{timestamp}.jsonl
//!
//! Run: `cargo run --example bomber_06_replay_gen_v2 --features bomber`

use std::path::PathBuf;

use fastrand::Rng;

use katgpt_rs::pruners::bomber::arena::{EMPTY_ARENA, PILLAR_HEAVY_ARENA, STANDARD_ARENA};
use katgpt_rs::pruners::bomber::replay::{
    ReplaySample, ReplayWriter, serialize_board, serialize_bombs, serialize_powerups,
};
use katgpt_rs::pruners::bomber::{
    ArenaGrid, BomberPlayer, Cell, GameEvent, GreedyPlayer, HLPlayer, RandomPlayer,
    ValidatorPlayer, init_world, init_world_with_arena, run_tick, spawn_players,
};

// ── Config ─────────────────────────────────────────────────────

const ROUNDS: usize = 2000;
const TICK_LIMIT: u32 = 200;
const ACTION_NAMES: [&str; 7] = ["Up", "Down", "Left", "Right", "Bomb", "Wait", "Detonate"];
const PLAYER_NAMES: [&str; 4] = ["Random", "Greedy", "Validator", "HL"];

// ── CLI ────────────────────────────────────────────────────────

fn parse_args() -> (Option<&'static str>, u64, PathBuf) {
    let args: Vec<String> = std::env::args().collect();
    let mut map_preset = None;
    let mut seed = 42u64;
    let mut output_dir = PathBuf::from("output/replays_v2");
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
            other if !other.starts_with('-') => {
                output_dir = PathBuf::from(other);
            }
            _ => {}
        }
        i += 1;
    }
    (map_preset, seed, output_dir)
}

// ── Pending Sample ─────────────────────────────────────────────

struct PendingSample {
    board: Vec<u8>,
    player_pos: [u8; 2],
    player_id: u8,
    bombs: Vec<[u8; 4]>,
    powerups: Vec<[u8; 2]>,
    action: u8,
    tick: u32,
    player_type: String,
    opponent_positions: Vec<(i32, i32)>,
}

// ── Round Result ───────────────────────────────────────────────

struct RoundResult {
    _scores: [i32; 4],
    survivors: Vec<u8>,
    _deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
    _ticks: u32,
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let (map_preset, default_seed, output_dir) = parse_args();
    std::fs::create_dir_all(&output_dir).ok();

    let mut rng = Rng::with_seed(default_seed);
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    // Create a single ReplayWriter with timestamp filename
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("bomber_replay_v2_{timestamp}.jsonl");
    let path = output_dir.join(&filename);
    let mut writer = ReplayWriter::create(&path, 0).expect("failed to create replay writer");

    println!("╔═══ Bomberman Replay Generator v2 (Balanced) ═════════════╗");
    println!("║  P0 Random | P1 Greedy | P2 Validator | P3 HL           ║");
    println!("║  ALL players, winners AND losers, enriched features     ║");
    println!("║  {ROUNDS} rounds, quality spread tracking                 ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let mut action_counts = [0usize; 7];
    let mut total_quality = 0.0f64;
    let mut total_samples = 0u64;
    let mut quality_low = 0u64; // < 0.3
    let mut quality_mid = 0u64; // 0.3 - 0.7
    let mut quality_high = 0u64; // > 0.7
    let mut player_type_counts = [0u64; 4];
    let mut player_type_quality = [0.0f64; 4];

    for round in 0..ROUNDS {
        let seed = default_seed + round as u64;
        let (result, pending) = run_round(seed, map_preset, &mut players, &mut rng);

        // Update HL player with outcome
        let p3_survived = result.survivors.contains(&3);
        let p3_killed = result.kills.iter().any(|(k, _)| *k == 3);
        let p3_pu_count = result.powerups.iter().filter(|(p, _)| *p == 3).count();
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(p3_survived, p3_killed, p3_pu_count as u32);
        }

        // Backfill quality and write ALL samples (no quality filter)
        for ps in pending {
            let survived = result.survivors.contains(&ps.player_id);
            let winner = result.survivors.len() == 1 && result.survivors[0] == ps.player_id;
            let pu_count = result
                .powerups
                .iter()
                .filter(|(p, _)| *p == ps.player_id)
                .count() as u32;
            let kill_count = result
                .kills
                .iter()
                .filter(|(k, _)| *k == ps.player_id)
                .count() as u32;
            let quality = ReplaySample::quality(survived, winner, pu_count, kill_count);

            // Reconstruct grid and bombs for enrichment
            let grid = ArenaGrid::generate(seed);
            let sample = ReplaySample {
                board: ps.board,
                player_pos: ps.player_pos,
                player_id: ps.player_id,
                bombs: ps.bombs.clone(),
                bomb_types: vec![],
                powerups: ps.powerups,
                action: ps.action,
                quality,
                tick: ps.tick,
                round: round as u32,
                player_type: ps.player_type.clone(),
                danger_level: compute_danger_level(ps.player_pos, &grid, &ps.bombs),
                nearest_opponent_dist: compute_nearest_opponent_dist(
                    ps.player_pos,
                    &ps.opponent_positions,
                ),
                escape_routes: compute_escape_routes(ps.player_pos, &grid),
                template_id: 255, // not set
            };

            writer.write_sample(&sample).ok();
            action_counts[ps.action as usize] += 1;
            total_quality += quality as f64;
            total_samples += 1;

            // Quality bucket tracking
            if quality < 0.3 {
                quality_low += 1;
            } else if quality <= 0.7 {
                quality_mid += 1;
            } else {
                quality_high += 1;
            }

            // Per-player-type tracking
            if (ps.player_id as usize) < 4 {
                player_type_counts[ps.player_id as usize] += 1;
                player_type_quality[ps.player_id as usize] += quality as f64;
            }
        }

        // Progress every 500 rounds
        if (round + 1) % 500 == 0 {
            println!("  [Round {}/{}] samples={total_samples}", round + 1, ROUNDS,);
        }
    }

    writer.flush().ok();

    // ── Stats ──────────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  REPLAY GENERATION v2 COMPLETE ({ROUNDS} rounds)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Total samples written: {total_samples}");

    println!("  Quality distribution:");
    if total_samples > 0 {
        let low_pct = (quality_low as f64 / total_samples as f64) * 100.0;
        let mid_pct = (quality_mid as f64 / total_samples as f64) * 100.0;
        let high_pct = (quality_high as f64 / total_samples as f64) * 100.0;
        println!("    low    (<0.3): {quality_low:>7}  ({low_pct:.1}%)");
        println!("    mid  (0.3-0.7): {quality_mid:>7}  ({mid_pct:.1}%)");
        println!("    high   (>0.7): {quality_high:>7}  ({high_pct:.1}%)");
    }

    println!("  Action distribution:");
    for (i, name) in ACTION_NAMES.iter().enumerate() {
        let count = action_counts[i];
        if count > 0 {
            let pct = (count as f64 / total_samples as f64) * 100.0;
            println!("    {name:<8} {count:>6}  ({pct:.1}%)");
        }
    }

    println!("  Player type distribution:");
    for (i, name) in PLAYER_NAMES.iter().enumerate() {
        let count = player_type_counts[i];
        if count > 0 {
            let pct = (count as f64 / total_samples as f64) * 100.0;
            let avg_q = player_type_quality[i] / count as f64;
            println!("    {name:<10} {count:>6}  ({pct:.1}%)  avg_quality={avg_q:.3}");
        }
    }

    let avg_quality = if total_samples > 0 {
        total_quality / total_samples as f64
    } else {
        0.0
    };
    println!("  Average quality: {avg_quality:.3}");
    println!("  Output: {}", path.display());
    println!();
}

// ── Enrichment Helpers ─────────────────────────────────────────

/// Compute danger level from player position, grid, and bomb list.
/// 0=safe, 1=adjacent to blast zone, 2=in blast zone
fn compute_danger_level(pos: [u8; 2], grid: &ArenaGrid, bombs: &[[u8; 4]]) -> u8 {
    let px = pos[0] as i32;
    let py = pos[1] as i32;

    // Check if player is directly in a blast zone
    for &[bx, by, range, _fuse] in bombs {
        if in_blast_path(px, py, bx as i32, by as i32, range as u32, grid) {
            return 2;
        }
    }

    // Check if any adjacent cell is in blast zone
    for &(dx, dy) in &[(0i32, -1), (0, 1), (-1, 0), (1, 0)] {
        let nx = px + dx;
        let ny = py + dy;
        for &[bx, by, range, _fuse] in bombs {
            if in_blast_path(nx, ny, bx as i32, by as i32, range as u32, grid) {
                return 1;
            }
        }
    }

    0
}

/// Check if (x,y) is in the blast path of a bomb at (bx,by) with given range.
/// Blast propagates in 4 cardinal directions, stops at walls.
fn in_blast_path(x: i32, y: i32, bx: i32, by: i32, range: u32, grid: &ArenaGrid) -> bool {
    let directions: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for &(dx, dy) in &directions {
        for d in 1..=range as i32 {
            let cx = bx + dx * d;
            let cy = by + dy * d;
            let cell = grid.get(cx, cy);
            match cell {
                Cell::FixedWall => break,
                Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                    if cx == x && cy == y {
                        return true;
                    }
                    break;
                }
                Cell::Floor => {
                    if cx == x && cy == y {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Manhattan distance to nearest opponent. Returns 255 if none.
fn compute_nearest_opponent_dist(pos: [u8; 2], opponents: &[(i32, i32)]) -> u8 {
    let px = pos[0] as i32;
    let py = pos[1] as i32;
    opponents
        .iter()
        .map(|&(ox, oy)| ((px - ox).abs() + (py - oy).abs()) as u8)
        .min()
        .unwrap_or(255)
}

/// Count walkable adjacent cells.
fn compute_escape_routes(pos: [u8; 2], grid: &ArenaGrid) -> u8 {
    let px = pos[0] as i32;
    let py = pos[1] as i32;
    [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
        .iter()
        .filter(|&&(dx, dy)| grid.is_walkable(px + dx, py + dy))
        .count() as u8
}

// ── Round Runner ───────────────────────────────────────────────

fn run_round(
    seed: u64,
    map_preset: Option<&'static str>,
    players: &mut [Box<dyn BomberPlayer>],
    rng: &mut Rng,
) -> (RoundResult, Vec<PendingSample>) {
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

    let mut all_events: Vec<GameEvent> = Vec::new();
    let mut pending_samples: Vec<PendingSample> = Vec::new();

    for _tick in 0..TICK_LIMIT {
        // Drain events from previous tick
        let tick_events: Vec<GameEvent> = {
            use bevy_ecs::event::Events;
            let mut ev = world.resource_mut::<Events<GameEvent>>();
            ev.drain().collect()
        };
        all_events.extend(tick_events.iter().cloned());

        // Collect opponent positions for this tick
        let all_positions: [(i32, i32); 4] = {
            let mut positions = [(0i32, 0i32); 4];
            for (i, &entity) in entities.iter().enumerate() {
                if let Some(pos) = world.get::<katgpt_rs::pruners::bomber::GridPos>(entity) {
                    positions[i] = (pos.x, pos.y);
                }
            }
            positions
        };

        // Each player selects an action
        let mut actions = [None; 4];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<katgpt_rs::pruners::bomber::GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world
                .get::<katgpt_rs::pruners::bomber::Alive>(entities[i])
                .is_some();
            if alive {
                let grid = world
                    .resource::<katgpt_rs::pruners::bomber::ArenaGrid>()
                    .clone();
                let action = player.select_action(&grid, pos, &tick_events, rng);
                actions[i] = Some(action);

                // Capture ALL players (indices 0-3)
                let grid_ref = world.resource::<katgpt_rs::pruners::bomber::ArenaGrid>();
                let board = serialize_board(grid_ref);
                let bombs = serialize_bombs(&mut world);
                let powerups = serialize_powerups(&mut world);
                let tick = world
                    .resource::<katgpt_rs::pruners::bomber::TickCounter>()
                    .tick;

                // Opponent positions = all positions except self
                let opponent_positions: Vec<(i32, i32)> = all_positions
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &p)| p)
                    .collect();

                pending_samples.push(PendingSample {
                    board,
                    player_pos: [pos.x as u8, pos.y as u8],
                    player_id: i as u8,
                    bombs,
                    powerups,
                    action: action.as_usize() as u8,
                    tick,
                    player_type: PLAYER_NAMES[i].to_string(),
                    opponent_positions,
                });
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Drain remaining events
    {
        use bevy_ecs::event::Events;
        let mut ev = world.resource_mut::<Events<GameEvent>>();
        all_events.extend(ev.drain().collect::<Vec<GameEvent>>());
    }

    // Compute scores from events
    let mut scores = [0i32; 4];
    let mut deaths = Vec::new();
    let mut kills = Vec::new();
    let mut powerups = Vec::new();
    let mut survivors = Vec::new();

    for event in &all_events {
        match event {
            GameEvent::PlayerKilled { victim, killer } => {
                deaths.push(*victim);
                scores[*victim as usize] -= 3;
                match killer {
                    Some(k) if *k != *victim => {
                        kills.push((*k, *victim));
                        scores[*k as usize] += 3;
                    }
                    _ => {
                        scores[*victim as usize] -= 2;
                    }
                }
            }
            GameEvent::PowerUpCollected { player, .. } => {
                scores[*player as usize] += 1;
                powerups.push((*player, 1));
            }
            GameEvent::RoundEnd { survivors: s } => {
                survivors = s.clone();
            }
            _ => {}
        }
    }

    // Winner / timeout bonus
    if survivors.len() == 1 {
        scores[survivors[0] as usize] += 5;
    } else if survivors.len() > 1 {
        for &s in &survivors {
            scores[s as usize] += 3;
        }
    }

    let ticks = world
        .resource::<katgpt_rs::pruners::bomber::TickCounter>()
        .tick;

    let result = RoundResult {
        _scores: scores,
        survivors,
        _deaths: deaths,
        kills,
        powerups,
        _ticks: ticks,
    };

    (result, pending_samples)
}
