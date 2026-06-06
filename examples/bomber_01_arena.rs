//! Bomberman HL Arena — Headless Tournament Runner (Plan 033, Task 6)
//!
//! Runs N rounds of 4-player Bomberman with progressively more HL technology.
//! Output: per-round results and cumulative standings.
//!
//! With `--replay-dir <path>`, dumps per-round JSONL replay files for all players.
//!
//! With `--device gate`, P4 uses InferenceRouter + TriggerGate for routed inference
//! instead of the default HLPlayer (bandit-based learning).
//!
//! Run: `cargo run --example bomber_01_arena --features bomber`
//! Gate: `cargo run --example bomber_01_arena --features bomber -- --device gate`
//! Replay: `cargo run --example bomber_01_arena --features bomber -- --replay-dir output/replays`

use std::path::PathBuf;

use fastrand::Rng;

use katgpt_rs::pruners::bomber::arena::{EMPTY_ARENA, PILLAR_HEAVY_ARENA, STANDARD_ARENA};
use katgpt_rs::pruners::bomber::replay::{
    ReplaySample, ReplayWriter, serialize_board, serialize_bombs, serialize_powerups,
};
use katgpt_rs::pruners::bomber::{
    ArenaGrid, BomberPlayer, GameEvent, GreedyPlayer, HLPlayer, RandomPlayer, ValidatorPlayer,
    init_world, init_world_with_arena, run_tick, spawn_players,
};

// ── Config ─────────────────────────────────────────────────────

const ROUNDS: usize = 100;
const TICK_LIMIT: u32 = 200;

// ── CLI ────────────────────────────────────────────────────────

/// Parsed CLI arguments.
struct CliArgs {
    map_preset: Option<&'static str>,
    seed: u64,
    device_gate: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut map_preset = None;
    let mut seed = 42u64;
    let mut device_gate = false;
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
            "--device" if i + 1 < args.len() => {
                i += 1;
                match args[i].as_str() {
                    "cpu" => device_gate = false,
                    "gate" => device_gate = true,
                    other => {
                        eprintln!("Unknown device: {other}. Use: cpu, gate");
                        std::process::exit(1);
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    CliArgs {
        map_preset,
        seed,
        device_gate,
    }
}

// ── Pending Capture ────────────────────────────────────────────

struct PendingCapture {
    sample: ReplaySample,
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    // Parse --replay-dir <path>
    let replay_dir: Option<PathBuf> = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--replay-dir")
        .map(|w| PathBuf::from(&w[1]));
    if let Some(ref dir) = replay_dir {
        std::fs::create_dir_all(dir).ok();
    }

    let cli = parse_args();
    let default_seed = cli.seed;
    let mut rng = Rng::with_seed(default_seed);

    // Build player roster. When --device gate, P4 becomes GatePlayer.
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
    ];
    if cli.device_gate {
        players.push(Box::new(katgpt_rs::pruners::bomber::GatePlayer::new(3)));
    } else {
        players.push(Box::new(HLPlayer::new(3)));
    }

    let p4_name = if cli.device_gate { "Gate" } else { "HL" };
    let p4_emoji = if cli.device_gate { "🚀" } else { "🐵" };

    println!("╔═══ Bomberman HL Arena ═══════════════════════════════════╗");
    println!(
        "║  P1 🐰 Random  |  P2 🐱 Greedy  |  P3 🐶 Validator  |  P4 {} {}  ║",
        p4_emoji, p4_name
    );
    match cli.map_preset {
        Some(_) => println!("║  Map: fixed preset                                      ║"),
        None => println!("║  Map: procedural (seed={default_seed:<35})  ║"),
    }
    if cli.device_gate {
        println!("║  Device: gate (InferenceRouter + TriggerGate)           ║");
    }
    if let Some(ref dir) = replay_dir {
        println!("║  Replay dir: {:<42}║", dir.display());
    }
    println!("╚═════════════════════════════════════════════════════════╝");
    println!();

    let mut scores = [0i32; 4];
    let mut wins = [0u32; 4];
    let mut deaths = [0u32; 4];
    let mut suicides = [0u32; 4]; // killed by own bomb or unknown
    let mut kill_matrix: [[u32; 4]; 4] = [[0; 4]; 4]; // kill_matrix[killer][victim]
    let mut total_replay_samples: u64 = 0;

    for round in 0..ROUNDS {
        // Create per-round replay writer
        let mut replay_writer: Option<ReplayWriter> = None;
        if let Some(ref dir) = replay_dir {
            let path = dir.join(format!("bomber_replay_{round:04}.jsonl"));
            replay_writer = ReplayWriter::create(&path, round as u32).ok();
        }

        let seed = default_seed + round as u64;
        let result = run_round(
            seed,
            cli.map_preset,
            &mut players,
            &mut rng,
            round as u32,
            &mut replay_writer,
        );

        // Accumulate replay samples
        if let Some(writer) = replay_writer {
            total_replay_samples += writer.sample_count();
        }

        // Update stats
        for (i, s) in result.scores.iter().enumerate() {
            scores[i] += s;
        }
        if let Some(winner) = result.winner {
            wins[winner as usize] += 1;
        }
        for &victim in &result.deaths {
            deaths[victim as usize] += 1;
        }

        // Track kill matrix and suicides
        for &(killer, victim) in &result.kills {
            kill_matrix[killer as usize][victim as usize] += 1;
        }
        // Deaths without a killer entry = suicide or unknown
        let killed_by_someone: std::collections::HashSet<usize> =
            result.kills.iter().map(|(_, v)| *v as usize).collect();
        for &victim in &result.deaths {
            if !killed_by_someone.contains(&(victim as usize)) {
                suicides[victim as usize] += 1;
            }
        }

        // Update HL player with outcome
        let survived = !result.deaths.contains(&3);
        let killed = result.kills.iter().any(|(killer, _)| *killer == 3);
        let powerups = result.powerups.iter().filter(|(p, _)| *p == 3).count();
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(survived, killed, powerups as u32);
        }

        // Print round result
        let emoji = ["🐰", "🐱", "🐶", p4_emoji];
        let winner_str = match result.winner {
            Some(w) => format!("{} P{}", emoji[w as usize], w),
            None => "Draw".to_string(),
        };
        println!(
            "Round {:>3}: Winner={:<12} Scores=[{}] Ticks={}",
            round + 1,
            winner_str,
            result
                .scores
                .iter()
                .enumerate()
                .map(|(i, s)| format!("{}:{:+}", emoji[i], s))
                .collect::<Vec<_>>()
                .join(" "),
            result.ticks,
        );
    }

    // Final standings
    println!();
    println!("═══ Final Standings ({ROUNDS} rounds) ═══");
    let emoji = ["🐰", "🐱", "🐶", p4_emoji];
    let names = ["Random", "Greedy", "Validator", p4_name];
    let mut ranking: Vec<(usize, i32)> = scores.iter().copied().enumerate().collect();
    ranking.sort_by(|a, b| b.1.cmp(&a.1));

    for (rank, (idx, score)) in ranking.iter().enumerate() {
        println!(
            "  #{} {} {:<10} Score={:+5}  Wins={}  Deaths={}  Suicides={}",
            rank + 1,
            emoji[*idx],
            names[*idx],
            score,
            wins[*idx],
            deaths[*idx],
            suicides[*idx],
        );
    }

    // Kill matrix
    println!();
    println!("═══ Kill Matrix (killer → victim) ═══");
    print!("           ");
    for j in 0..4 {
        print!("  {} {:<6}", emoji[j], names[j]);
    }
    println!("  Total");
    for i in 0..4 {
        let total_kills: u32 = kill_matrix[i].iter().sum();
        print!("  {} {:<6}", emoji[i], names[i]);
        for (j, &kills) in kill_matrix[i].iter().enumerate() {
            if i == j {
                print!("    ---   ");
            } else {
                print!("  {kills:>8} ");
            }
        }
        println!("  {}", total_kills);
    }

    // Replay stats
    if replay_dir.is_some() {
        println!();
        println!("  Replay: {total_replay_samples} total samples written");
    }

    // Router stats (gate mode only)
    if cli.device_gate
        && let Some(gate_player) = players[3]
            .as_any()
            .downcast_ref::<katgpt_rs::pruners::bomber::GatePlayer>()
    {
            let stats = gate_player.router_stats();
            println!();
            println!("═══ InferenceRouter Stats ═══");
            println!("  Tier: {}", stats.current_tier);
            println!("  Total inferences: {}", stats.total_inferences);
            println!("  Estimated QPS: {:.1}", stats.estimated_qps);
            println!("  Tier transitions: {}", stats.tier_transitions);
            println!("  Last backend: {}", stats.last_backend);
            println!("  Total routed calls: {}", gate_player.total_routed());
    }
}

// ── Round Runner ────────────────────────────────────────────────

struct RoundResult {
    scores: [i32; 4],
    winner: Option<u8>,
    deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
    ticks: u32,
}

fn run_round(
    seed: u64,
    map_preset: Option<&'static str>,
    players: &mut [Box<dyn BomberPlayer>],
    rng: &mut Rng,
    round_num: u32,
    replay_writer: &mut Option<ReplayWriter>,
) -> RoundResult {
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

    // Reset players
    for p in players.iter_mut() {
        p.reset();
    }

    let mut round_events: Vec<GameEvent> = Vec::new();
    let mut pending: Vec<PendingCapture> = Vec::new();
    let player_names = ["Random", "Greedy", "Validator", "HL"];
    let capture_replay = replay_writer.is_some();

    // Run tick loop
    for _tick in 0..TICK_LIMIT {
        // Drain events from previous tick (tick-scoped for AI, accumulated for scoring)
        let tick_events: Vec<GameEvent> = {
            let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
            event_reader.drain().collect()
        };
        round_events.extend(tick_events.iter().cloned());

        // Each player selects an action (only sees THIS tick's events)
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
                actions[i] = Some(
                    player.select_action(
                        &world
                            .resource::<katgpt_rs::pruners::bomber::ArenaGrid>()
                            .clone(),
                        pos,
                        &tick_events,
                        rng,
                    ),
                );
            }
        }

        // Capture replay data for all alive players
        if capture_replay {
            let board = serialize_board(world.resource::<katgpt_rs::pruners::bomber::ArenaGrid>());
            let bombs = serialize_bombs(&mut world);
            let powerups = serialize_powerups(&mut world);
            let tick = world
                .resource::<katgpt_rs::pruners::bomber::TickCounter>()
                .tick;

            for i in 0..4 {
                let pos = world
                    .get::<katgpt_rs::pruners::bomber::GridPos>(entities[i])
                    .copied()
                    .unwrap_or_default();
                let alive = world
                    .get::<katgpt_rs::pruners::bomber::Alive>(entities[i])
                    .is_some();
                if alive && actions[i].is_some() {
                    pending.push(PendingCapture {
                        sample: ReplaySample {
                            board: board.clone(),
                            player_pos: [pos.x as u8, pos.y as u8],
                            player_id: i as u8,
                            bombs: bombs.clone(),
                            bomb_types: vec![],
                            powerups: powerups.clone(),
                            action: actions[i].map(|a| a.as_usize() as u8).unwrap_or(0),
                            quality: 0.0, // backfilled later
                            tick,
                            round: 0, // backfilled later
                            player_type: player_names[i].to_string(),
                            danger_level: 0,            // backfilled later
                            nearest_opponent_dist: 255, // backfilled later
                            escape_routes: 0,           // backfilled later
                            template_id: 255,           // not set (non-template player)
                        },
                    });
                }
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Drain remaining events
    {
        let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
        round_events.extend(event_reader.drain().collect::<Vec<GameEvent>>());
    }

    // Compute scores from events
    let mut scores = [0i32; 4];
    let mut deaths = Vec::new();
    let mut kills = Vec::new();
    let mut powerups = Vec::new();
    let mut survivors = Vec::new();

    for event in &round_events {
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
                        // Suicide (killer == victim or killer unknown)
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

    // Winner bonus
    let winner = if survivors.len() == 1 {
        scores[survivors[0] as usize] += 5;
        Some(survivors[0])
    } else if survivors.len() > 1 {
        // Timeout: survivors get +3 each
        for &s in &survivors {
            scores[s as usize] += 3;
        }
        None
    } else {
        None
    };

    let ticks = world
        .resource::<katgpt_rs::pruners::bomber::TickCounter>()
        .tick;

    // Compute quality and write replay samples
    if let Some(writer) = replay_writer {
        for mut cap in pending {
            let survived = survivors.contains(&cap.sample.player_id);
            let is_winner = survivors.len() == 1 && survivors[0] == cap.sample.player_id;
            let pu_count = powerups
                .iter()
                .filter(|(p, _)| *p == cap.sample.player_id)
                .count() as u32;
            let kill_count = kills
                .iter()
                .filter(|(k, _)| *k == cap.sample.player_id)
                .count() as u32;
            cap.sample.quality = ReplaySample::quality(survived, is_winner, pu_count, kill_count);
            cap.sample.round = round_num;
            writer.write_sample(&cap.sample).ok();
        }
        writer.flush().ok();
    }

    RoundResult {
        scores,
        winner,
        deaths,
        kills,
        powerups,
        ticks,
    }
}
