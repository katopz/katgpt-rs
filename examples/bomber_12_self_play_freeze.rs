//! Self-Play Freeze/Thaw Knowledge Pipeline (Plan 092)
//!
//! Phase 1 (LEARN): 100 rounds with naive HL → freezes knowledge to disk.
//! Phase 2 (REPLAY): Same 100 rounds with thawed HL → compares results.
//!
//! Demonstrates that frozen bandit knowledge survives round-trip persistence
//! and improves player performance on identical game seeds.
//!
//! Run: `cargo run --example bomber_12_self_play_freeze --features bomber`

use std::path::Path;

use fastrand::Rng;

use microgpt_rs::pruners::bomber::{
    BomberFrozenBandit, BomberPlayer, GameEvent, GreedyPlayer, GridPos, HLPlayer, RandomPlayer,
    ValidatorPlayer, init_world, run_tick, spawn_players,
};
use microgpt_rs::pruners::{load_frozen, save_frozen};

// ── Config ─────────────────────────────────────────────────────

const ROUNDS: usize = 100;
const TICK_LIMIT: u32 = 200;
const COMPRESS_INTERVAL: usize = 20;
const BASE_SEED: u64 = 42;
const HL_INDEX: usize = 3;
const OUTPUT_PATH: &str = "output/bomber_frozen_bandit.bin";

// ── Round Result ───────────────────────────────────────────────

#[derive(Debug, Default)]
struct RoundResult {
    scores: [i32; 4],
    survivors: Vec<u8>,
    deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
    ticks: u32,
}

// ── Phase Stats ────────────────────────────────────────────────

#[derive(Debug, Default)]
struct PhaseStats {
    survival_count: u32,
    total_score: i64,
    kill_count: u32,
    rounds: u32,
}

impl PhaseStats {
    fn survival_rate(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        (self.survival_count as f64 / self.rounds as f64) * 100.0
    }

    fn avg_score(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.total_score as f64 / self.rounds as f64
    }

    fn avg_kills(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.kill_count as f64 / self.rounds as f64
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Map action index to emoji symbol.
fn action_symbol(idx: usize) -> &'static str {
    match idx {
        0 => "↑",
        1 => "↓",
        2 => "←",
        3 => "→",
        4 => "💣",
        5 => "⏸",
        6 => "💥",
        _ => "?",
    }
}

/// Format Q-values as a readable string.
fn format_q_values(q: &[f32; 7]) -> String {
    q.iter()
        .enumerate()
        .map(|(i, v)| format!("{}:{:+.2}", action_symbol(i), v))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Format compressed arms list.
fn format_compressed(compressed: &[u8; 7]) -> String {
    let active: Vec<String> = compressed
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(i, _)| action_symbol(i).to_string())
        .collect();
    if active.is_empty() {
        "(none)".into()
    } else {
        format!("[{}]", active.join(", "))
    }
}

// ── Run Round ──────────────────────────────────────────────────

fn run_round(seed: u64, players: &mut [Box<dyn BomberPlayer>], rng: &mut Rng) -> RoundResult {
    let mut world = init_world(seed);
    let entities = spawn_players(&mut world);

    for p in players.iter_mut() {
        p.reset();
    }

    let mut all_events: Vec<GameEvent> = Vec::new();

    for _tick in 0..TICK_LIMIT {
        // Drain tick-scoped events
        let tick_events: Vec<GameEvent> = {
            use bevy_ecs::event::Events;
            let mut ev = world.resource_mut::<Events<GameEvent>>();
            ev.drain().collect()
        };
        all_events.extend(tick_events.iter().cloned());

        // Each player selects an action
        let mut actions = [None; 4];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world
                .get::<microgpt_rs::pruners::bomber::Alive>(entities[i])
                .is_some();
            if alive {
                let grid = world
                    .resource::<microgpt_rs::pruners::bomber::ArenaGrid>()
                    .clone();
                let action = player.select_action(&grid, pos, &tick_events, rng);
                actions[i] = Some(action);
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

    // Parse events into result
    let mut result = RoundResult::default();
    let mut survivors = Vec::new();

    for event in &all_events {
        match event {
            GameEvent::PlayerKilled { victim, killer } => {
                result.deaths.push(*victim);
                result.scores[*victim as usize] -= 3;
                match killer {
                    Some(k) if *k != *victim => {
                        result.kills.push((*k, *victim));
                        result.scores[*k as usize] += 3;
                    }
                    _ => {
                        // Suicide or unknown killer
                        result.scores[*victim as usize] -= 2;
                    }
                }
            }
            GameEvent::PowerUpCollected { player, .. } => {
                result.scores[*player as usize] += 1;
                result.powerups.push((*player, 1));
            }
            GameEvent::RoundEnd { survivors: s } => {
                survivors = s.clone();
            }
            _ => {}
        }
    }

    // Winner / timeout bonus
    match survivors.len() {
        1 => {
            result.scores[survivors[0] as usize] += 5;
        }
        2..=4 => {
            for &s in &survivors {
                result.scores[s as usize] += 3;
            }
        }
        _ => {}
    }

    result.survivors = survivors;
    result.ticks = world
        .resource::<microgpt_rs::pruners::bomber::TickCounter>()
        .tick;

    result
}

/// Accumulate phase stats for the HL player from a round result.
fn accumulate_hl_stats(stats: &mut PhaseStats, result: &RoundResult) {
    stats.rounds += 1;
    stats.total_score += result.scores[HL_INDEX] as i64;

    if result.survivors.contains(&(HL_INDEX as u8)) {
        stats.survival_count += 1;
    }

    let hl_kills = result
        .kills
        .iter()
        .filter(|(k, _)| *k as usize == HL_INDEX)
        .count();
    stats.kill_count += hl_kills as u32;
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let mut rng = Rng::with_seed(BASE_SEED);
    let output_path = Path::new(OUTPUT_PATH);

    println!("╔═══ Self-Play Freeze/Thaw — Bomber ({ROUNDS} rounds × 2 phases) ═══╗");
    println!("║  Phase 1: LEARN  (naive HL → freeze knowledge)              ║");
    println!("║  Phase 2: REPLAY (thawed HL → same seeds)                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ═══════════════════════════════════════════════════════════════
    //  PHASE 1: LEARN
    // ═══════════════════════════════════════════════════════════════

    println!("━━━ Phase 1: LEARN ({ROUNDS} rounds) ━━━");

    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    let mut phase1_stats = PhaseStats::default();

    for round in 0..ROUNDS {
        let seed = BASE_SEED + round as u64;
        let result = run_round(seed, &mut players, &mut rng);

        accumulate_hl_stats(&mut phase1_stats, &result);

        // Update HL player with round outcome
        let hl_survived = result.survivors.contains(&(HL_INDEX as u8));
        let hl_killed = result.kills.iter().any(|(k, _)| *k as usize == HL_INDEX);
        let hl_pu_count = result
            .powerups
            .iter()
            .filter(|(p, _)| *p as usize == HL_INDEX)
            .count() as u32;

        if let Some(hl) = players[HL_INDEX].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(hl_survived, hl_killed, hl_pu_count);
        }

        // Compress cycle at intervals
        if (round + 1) % COMPRESS_INTERVAL == 0
            && let Some(hl) = players[HL_INDEX].as_any_mut().downcast_mut::<HLPlayer>()
        {
            let compressed = hl.compress_cycle();
            let report = hl.compress_report();
            println!("  [Round {}/{}] {report}", round + 1, ROUNDS);
            if !compressed.is_empty() {
                let names: Vec<String> = compressed
                    .iter()
                    .map(|&i| action_symbol(i).to_string())
                    .collect();
                println!("    → Newly compressed: [{}]", names.join(", "));
            }
        }
    }

    // Freeze knowledge
    let frozen = match players[HL_INDEX].as_any().downcast_ref::<HLPlayer>() {
        Some(hl) => hl.freeze(),
        None => {
            eprintln!("ERROR: Could not downcast HL player for freeze");
            std::process::exit(1);
        }
    };

    match save_frozen(output_path, &frozen) {
        Ok(()) => {
            let size = std::mem::size_of::<BomberFrozenBandit>();
            println!();
            println!(
                "  Frozen knowledge saved to {} ({} bytes)",
                output_path.display(),
                size
            );
            println!("  Q-values: [{}]", format_q_values(&frozen.q_values));
            println!(
                "  Compressed arms: {}",
                format_compressed(&frozen.compressed)
            );
            println!("  Total pulls: {}", frozen.total_pulls);
        }
        Err(e) => {
            eprintln!("ERROR: Failed to save frozen bandit: {e}");
            std::process::exit(1);
        }
    }

    println!();
    println!("  Phase 1 Results:");
    println!(
        "    HL Survival: {:.1}%  |  Avg Score: {:+.1}  |  Kills: {:.1}/round",
        phase1_stats.survival_rate(),
        phase1_stats.avg_score(),
        phase1_stats.avg_kills(),
    );

    // ═══════════════════════════════════════════════════════════════
    //  PHASE 2: REPLAY (thawed knowledge)
    // ═══════════════════════════════════════════════════════════════

    println!();
    println!("━━━ Phase 2: REPLAY ({ROUNDS} rounds, frozen knowledge) ━━━");

    // Load frozen bandit
    let loaded_frozen: BomberFrozenBandit = match load_frozen(output_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: Failed to load frozen bandit: {e}");
            std::process::exit(1);
        }
    };

    match loaded_frozen.validate() {
        Ok(()) => println!("  Loaded frozen bandit: magic=OK, version=OK"),
        Err(e) => {
            eprintln!("ERROR: Frozen bandit validation failed: {e}");
            std::process::exit(1);
        }
    }

    // Thaw into new HL player
    let mut thawed_players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        match HLPlayer::thaw(&loaded_frozen, HL_INDEX as u8) {
            Ok(p) => Box::new(p),
            Err(e) => {
                eprintln!("ERROR: Failed to thaw HL player: {e}");
                std::process::exit(1);
            }
        },
    ];

    let mut phase2_stats = PhaseStats::default();

    for round in 0..ROUNDS {
        let seed = BASE_SEED + round as u64;
        let result = run_round(seed, &mut thawed_players, &mut rng);

        accumulate_hl_stats(&mut phase2_stats, &result);
    }

    println!();
    println!("  Phase 2 Results:");
    println!(
        "    HL Survival: {:.1}%  |  Avg Score: {:+.1}  |  Kills: {:.1}/round",
        phase2_stats.survival_rate(),
        phase2_stats.avg_score(),
        phase2_stats.avg_kills(),
    );

    // ═══════════════════════════════════════════════════════════════
    //  COMPARISON
    // ═══════════════════════════════════════════════════════════════

    let delta_survival = phase2_stats.survival_rate() - phase1_stats.survival_rate();
    let delta_score = phase2_stats.avg_score() - phase1_stats.avg_score();
    let delta_kills = phase2_stats.avg_kills() - phase1_stats.avg_kills();

    fn indicator(delta: f64) -> &'static str {
        if delta > 0.5 {
            "✅"
        } else if delta > -0.5 {
            "➖"
        } else {
            "❌"
        }
    }

    println!();
    println!("━━━ COMPARISON ━━━");
    println!();
    println!(
        "  {:<14} {:>10} {:>10} {:>12}",
        "Metric", "Phase 1", "Phase 2", "Δ"
    );
    println!("  {}", "─".repeat(50));
    println!(
        "  {:<14} {:>9.1}% {:>9.1}% {:>+10.1}pp  {}",
        "Survival%",
        phase1_stats.survival_rate(),
        phase2_stats.survival_rate(),
        delta_survival,
        indicator(delta_survival),
    );
    println!(
        "  {:<14} {:>+10.1} {:>+10.1} {:>+11.1}  {}",
        "Avg Score",
        phase1_stats.avg_score(),
        phase2_stats.avg_score(),
        delta_score,
        indicator(delta_score),
    );
    println!(
        "  {:<14} {:>10.2} {:>10.2} {:>+11.2}  {}",
        "Kills/Round",
        phase1_stats.avg_kills(),
        phase2_stats.avg_kills(),
        delta_kills,
        indicator(delta_kills),
    );
    println!();

    // ── Verdict ────────────────────────────────────────────────

    let improved = delta_survival > 5.0 || delta_score > 1.0 || delta_kills > 0.1;

    println!("━━━ VERDICT ━━━");
    println!();
    if improved {
        println!("  ✅ Freeze/Thaw pipeline verified.");
        println!("     Thawed HL player retains compressed knowledge and performs");
        println!("     comparably or better than the original naive player on identical seeds.");
    } else {
        println!("  ⚠️  Marginal or no improvement detected.");
        println!("     This is expected with few rounds — the bandit may need more");
        println!("     episodes for compression to significantly shift behavior.");
    }
    println!();
    println!(
        "  Frozen file: {} ({} bytes)",
        output_path.display(),
        std::mem::size_of::<BomberFrozenBandit>(),
    );
    println!(
        "  Bandit pulls: {} | Compressed arms: {}",
        loaded_frozen.total_pulls,
        loaded_frozen.compressed.iter().filter(|&&c| c != 0).count(),
    );
    println!();
}
