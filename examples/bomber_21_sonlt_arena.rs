//! Plan 296 Phase 7 T7.1 — SON-LT Bomber Arena Evaluation.
//!
//! Loads a SON-LT (Sparse OPD-Native LOra Training) trained LoRA adapter
//! and evaluates it in a 1000-round Bomberman tournament against the
//! HLPlayer baseline.
//!
//! # Setup
//!
//! - P1 🐰 Random       — baseline (no strategy)
//! - P2 🐱 Greedy      — heuristic scoring
//! - P3 🧠 SON-LT      — LoRA-trained Transformer (Plan 296 adapter)
//! - P4 🐵 HL          — bandit-adaptive baseline (Plan 033 GOAT)
//!
//! # GOAT Gate
//!
//! Target: SON-LT (P3) avg_score ≥ +500 (HL baseline ~+475).
//! The SON-LT adapter was trained on Bomberman replay data using
//! multi-adapter LoRA (q/k/v/o/mlp1/mlp2) on a `Config::game()` Transformer.
//!
//! # Run
//!
//! ```sh
//! cargo run --example bomber_21_sonlt_arena --features bomber -- \
//!     --lora-path /path/to/game_lora_sonlt_t71.bin
//! ```

#![cfg(feature = "bomber")]
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use fastrand::Rng;

use katgpt_rs::pruners::bomber::arena::{EMPTY_ARENA, PILLAR_HEAVY_ARENA, STANDARD_ARENA};
use katgpt_rs::pruners::bomber::{
    ArenaGrid, BomberPlayer, GameEvent, GreedyPlayer, GridPos, HLPlayer, RandomPlayer,
    SonltPlayer, init_world, init_world_with_arena, run_tick, spawn_players,
};

// ── Config ─────────────────────────────────────────────────────

/// Paper target for arena evaluation (Plan 296 T7.1).
const ROUNDS: usize = 1000;

/// Per-round tick budget.
const TICK_LIMIT: u32 = 200;

/// SON-LT GOAT gate: P3 avg_score must reach this.
///
/// # Issue 306 correction (2026-06-28)
///
/// The original target was 500.0, which is physically impossible — the max
/// theoretical per-round score is ~+17 (3 kills × +3, last-survivor +5,
/// ~3 powerups × +1). The best heuristic players (Greedy, HL) score ~+2.5/round.
/// The 500.0 value appears to have confused cumulative score (over 1000 rounds)
/// with per-round average. The realistic gate is: SON-LT beats HL on per-round
/// avg_score. We keep an absolute floor of +1.5 (mid-range for a competent
/// heuristic) as a sanity check, but the primary gate is the relative comparison.
const SONLT_TARGET_SCORE: f32 = 1.5;

/// Expected HL baseline avg_score (Plan 033 reference).
///
/// Originally 475.0 (same cumulative/per-round confusion as above). HL
/// actually scores ~+1.1 to +2.1 per round depending on map/RNG. We use +1.5
/// as the midpoint reference.
const HL_BASELINE_SCORE: f32 = 1.5;

/// Default LoRA path relative to CARGO_MANIFEST_DIR.
const DEFAULT_LORA_REL: &str = "../../../output/game_lora_sonlt_t71.bin";

// ── CLI ────────────────────────────────────────────────────────

fn parse_args() -> (Option<&'static str>, u64, PathBuf) {
    let args: Vec<String> = std::env::args().collect();
    let mut map_preset = None;
    let mut seed = 42u64;
    let default_lora = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_LORA_REL);
    let mut lora_path = default_lora;

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
            "--lora-path" if i + 1 < args.len() => {
                i += 1;
                lora_path = PathBuf::from(&args[i]);
            }
            "--rounds" if i + 1 < args.len() => {
                i += 1;
                // ROUNDS is a compile-time const; --rounds is advisory only.
                if args[i].parse::<usize>().is_err() {
                    eprintln!("Note: --rounds value invalid (compiled ROUNDS={ROUNDS})");
                }
            }
            _ => {}
        }
        i += 1;
    }
    (map_preset, seed, lora_path)
}

// ── Stats ──────────────────────────────────────────────────────

#[derive(Clone, Default)]
struct PlayerStats {
    survival_count: u32,
    kill_count: u32,
    death_count: u32,
    powerup_count: u32,
    total_score: i64,
    rounds_played: u32,
}

impl PlayerStats {
    fn survival_rate(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.survival_count as f32 / self.rounds_played as f32
    }

    fn avg_score(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.total_score as f32 / self.rounds_played as f32
    }

    fn avg_kills(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.kill_count as f32 / self.rounds_played as f32
    }

    fn powerup_efficiency(&self) -> f32 {
        if self.rounds_played == 0 {
            return 0.0;
        }
        self.powerup_count as f32 / self.rounds_played as f32
    }
}

// ── Round Result ───────────────────────────────────────────────

struct RoundResult {
    scores: [i32; 4],
    survivors: Vec<u8>,
    deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
}

fn run_round(
    seed: u64,
    map_preset: Option<&'static str>,
    players: &mut [Box<dyn BomberPlayer>],
    rng: &mut Rng,
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

    for p in players.iter_mut() {
        p.reset();
    }

    let mut all_events: Vec<GameEvent> = Vec::new();

    for _tick in 0..TICK_LIMIT {
        let tick_events: Vec<GameEvent> = {
            use bevy_ecs::event::Events;
            let mut ev = world.resource_mut::<Events<GameEvent>>();
            ev.drain().collect()
        };
        all_events.extend(tick_events.iter().cloned());

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
                let grid = world
                    .resource::<katgpt_rs::pruners::bomber::ArenaGrid>()
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

    // Drain remaining events.
    {
        use bevy_ecs::event::Events;
        let mut ev = world.resource_mut::<Events<GameEvent>>();
        all_events.extend(ev.drain().collect::<Vec<GameEvent>>());
    }

    // Score from events.
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

    if survivors.len() == 1 {
        scores[survivors[0] as usize] += 5;
    } else if survivors.len() > 1 {
        for &s in &survivors {
            scores[s as usize] += 3;
        }
    }

    RoundResult {
        scores,
        survivors,
        deaths,
        kills,
        powerups,
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let (map_preset, default_seed, lora_path) = parse_args();
    let lora_exists = lora_path.exists();

    println!("╔═══ Plan 296 Phase 7 T7.1 — SON-LT Bomber Arena ═══════════════╗");
    println!("║  1000-round tournament: SON-LT LoRA vs HL baseline            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  LoRA path: {}", lora_path.display());
    println!("  LoRA exists: {lora_exists}");
    println!("  Map: {}", map_preset.unwrap_or("procedural"));
    println!("  Seed: {default_seed}");
    println!();

    // Print adapter info if the file exists.
    if lora_exists {
        match katgpt_rs::types::LoraAdapter::load(&lora_path) {
            Ok(adapters) => {
                println!("  LoRA adapters loaded: {}", adapters.len());
                for (i, a) in adapters.iter().enumerate() {
                    println!(
                        "    [{}] rank={} alpha={:.1} in_dim={} out_dim={}",
                        i, a.rank, a.alpha, a.in_dim, a.out_dim
                    );
                }
            }
            Err(e) => {
                println!("  ⚠ LoRA load error: {e}");
            }
        }
    } else {
        println!("  ⚠ LoRA file not found — P3 will run in heuristic fallback mode");
    }
    println!();

    let mut rng = Rng::with_seed(default_seed);
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(SonltPlayer::new_with_lora(2, lora_path.to_str().unwrap_or(""))),
        Box::new(HLPlayer::new(3)),
    ];

    println!("╔═══ Players ═══════════════════════════════════════════════════╗");
    println!("║  P1 🐰 Random    |  P2 🐱 Greedy    |  P3 🧠 SON-LT  |  P4 🐵 HL  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let mut stats: Vec<PlayerStats> = vec![PlayerStats::default(); 4];

    for round in 0..ROUNDS {
        let seed = default_seed + round as u64;
        let result = run_round(seed, map_preset, &mut players, &mut rng);

        for (i, s) in result.scores.iter().enumerate() {
            stats[i].total_score += *s as i64;
            stats[i].rounds_played += 1;
        }
        for &victim in &result.deaths {
            stats[victim as usize].death_count += 1;
        }
        for (killer, _victim) in &result.kills {
            stats[*killer as usize].kill_count += 1;
        }
        for (player, count) in &result.powerups {
            stats[*player as usize].powerup_count += count;
        }
        for &s in &result.survivors {
            stats[s as usize].survival_count += 1;
        }

        // Update HL player (P4) with outcome for bandit adaptation.
        let p4_survived = result.survivors.contains(&3);
        let p4_killed = result.kills.iter().any(|(k, _)| *k == 3);
        let p4_pu_count = result.powerups.iter().filter(|(p, _)| *p == 3).count();
        if let Some(hl) = players[3].as_any_mut().downcast_mut::<HLPlayer>() {
            hl.update_outcome(p4_survived, p4_killed, p4_pu_count as u32);
        }

        // Progress every 200 rounds.
        if (round + 1) % 200 == 0 {
            let emoji = ["🐰", "🐱", "🧠", "🐵"];
            let names = ["Random", "Greedy", "SON-LT", "HL"];
            println!("  [Round {}/{}]", round + 1, ROUNDS);
            for i in 0..4 {
                println!(
                    "    {} {:<10} survival={:.1}%  avg_score={:+.1}  kills={:.1}/round",
                    emoji[i],
                    names[i],
                    stats[i].survival_rate() * 100.0,
                    stats[i].avg_score(),
                    stats[i].avg_kills(),
                );
            }
            println!();
        }
    }

    // ── Final Results ──────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FINAL RESULTS ({ROUNDS} rounds)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let emoji = ["🐰", "🐱", "🧠", "🐵"];
    let names = ["Random", "Greedy", "SON-LT", "HL"];
    let tech = ["(baseline)", "(heuristic)", "(+LoRA)", "(+bandit)"];

    println!(
        "  {:<4} {:<10} {:<12} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "", "Player", "Tech", "Survival%", "AvgScore", "Kills/Round", "Deaths", "PU/Round"
    );
    println!("  {}", "─".repeat(80));

    let mut ranking: Vec<usize> = (0..4).collect();
    ranking.sort_by(|&a, &b| {
        stats[b]
            .avg_score()
            .partial_cmp(&stats[a].avg_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, &idx) in ranking.iter().enumerate() {
        println!(
            "  #{:<3} {} {:<10} {:<12} {:>9.1}% {:>+9.1} {:>11.2} {:>10} {:>9.2}",
            rank + 1,
            emoji[idx],
            names[idx],
            tech[idx],
            stats[idx].survival_rate() * 100.0,
            stats[idx].avg_score(),
            stats[idx].avg_kills(),
            stats[idx].death_count,
            stats[idx].powerup_efficiency(),
        );
    }

    // ── GOAT Gate: P3 (SON-LT) vs P4 (HL) ─────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GOAT GATE: P3 (🧠 SON-LT) vs P4 (🐵 HL baseline)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  P3 SON-LT  avg_score: {:+.1}  (target ≥ {SONLT_TARGET_SCORE:.0})",
        stats[2].avg_score()
    );
    println!(
        "  P4 HL      avg_score: {:+.1}  (baseline ref ~{HL_BASELINE_SCORE:.0})",
        stats[3].avg_score()
    );
    println!(
        "  P3 SON-LT  survival:  {:.1}%",
        stats[2].survival_rate() * 100.0
    );
    println!(
        "  P4 HL      survival:  {:.1}%",
        stats[3].survival_rate() * 100.0
    );

    let sonlt_passes = stats[2].avg_score() >= SONLT_TARGET_SCORE;
    let beats_hl = stats[2].avg_score() > stats[3].avg_score();
    println!();
    if sonlt_passes && beats_hl {
        println!("  ✅ GOAT PASSED: SON-LT avg_score ≥ {SONLT_TARGET_SCORE:.0} AND beats HL");
        println!("     The SON-LT LoRA adapter generalizes to live arena play.");
    } else if beats_hl {
        println!("  ⚠ PARTIAL: SON-LT beats HL but below {SONLT_TARGET_SCORE:.0} target");
        println!("     avg_score={:.1} vs target {SONLT_TARGET_SCORE:.0}",
            stats[2].avg_score());
    } else {
        println!("  ❌ GOAT NOT PASSED: SON-LT did not meet the target");
        println!("     avg_score={:.1}, HL={:.1}, target={SONLT_TARGET_SCORE:.0}",
            stats[2].avg_score(), stats[3].avg_score());
    }

    println!();
}
