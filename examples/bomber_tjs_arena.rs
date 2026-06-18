//! Plan 300 T1.11 — TJS-LoRA vs Dense-LoRA Bomber Arena (head-to-head ELO).
//!
//! Loads two LoRA adapters trained by `riir-train-gpu/examples/train_bomber_tjs.rs`
//! (one dense baseline arm via `--no-tjs`, one TJS-LoRA arm) and runs a
//! 4-player round-robin where the two LoRA variants compete head-to-head
//! alongside Random and Greedy baselines.
//!
//! # GOAT gate (Plan 300 T1.11)
//!
//! TJS-LoRA at rank 16 must achieve ≥ 95% of dense-LoRA ELO at 50% parameter
//! density. Paper finding (Zheng et al. 2026, §4.4): the task-conditioned
//! Jacobian support mask suffices — sparse training recovers nearly all of
//! dense-LoRA quality at much lower parameter count.
//!
//! # Setup
//!
//! - P1 🐰 Random  — baseline (no strategy)
//! - P2 🐱 Greedy  — heuristic scoring
//! - P3 🧠 Dense   — LoRA-trained Transformer (`--no-tjs` arm)
//! - P4 ✨ TJS     — TJS-LoRA-trained Transformer (rank-16 sparse arm)
//!
//! Both LoRA arms are produced by the same trainer on the same data/seed.
//! The only difference is whether the TJS hooks (compose_sparse_grad /
//! observe_jvp_ema / finalize_support_masks / enforce_sparsity_bound) fired.
//!
//! # ELO methodology
//!
//! 4-player Bomber matches use **pairwise survival-based ELO**: after each
//! round, for every (i, j) pair, the survivor wins. If both/neither survived,
//! the higher-scoring player wins; if scores tie, no update. Each player
//! starts at 1000; k=32 (standard `EloCalculator`).
//!
//! # Run
//!
//! ```sh
//! # From katgpt-rs workspace root:
//! cargo run --release --example bomber_tjs_arena --features bomber -- \
//!     --dense-path /path/to/game_lora_dense_t111.bin \
//!     --tjs-path   /path/to/game_lora_tjs_t111.bin \
//!     --rounds 1000
//! ```

#![cfg(feature = "bomber")]
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use fastrand::Rng;

use katgpt_rs::pruners::arena::types::EloCalculator;
use katgpt_rs::pruners::bomber::arena::{EMPTY_ARENA, PILLAR_HEAVY_ARENA, STANDARD_ARENA};
use katgpt_rs::pruners::bomber::{
    ArenaGrid, BomberPlayer, GameEvent, GreedyPlayer, GridPos, RandomPlayer, SonltPlayer,
    init_world, init_world_with_arena, run_tick, spawn_players,
};

// ── Config ─────────────────────────────────────────────────────

/// Default round count (paper-finding scale; matches bomber_21_sonlt_arena).
const ROUNDS: usize = 1000;

/// Per-round tick budget (matches bomber_21_sonlt_arena).
const TICK_LIMIT: u32 = 200;

/// T1.11 gate threshold: TJS ELO must be ≥ this fraction of Dense ELO.
const TJS_ELO_RATIO_TARGET: f64 = 0.95;

/// Standard ELO parameters (matches EloCalculator defaults + go_09_lora_arena).
const ELO_K: f64 = 32.0;
const ELO_BASE: f64 = 1000.0;

/// Default LoRA paths relative to CARGO_MANIFEST_DIR.
const DEFAULT_DENSE_REL: &str = "../../../output/game_lora_dense_t111.bin";
const DEFAULT_TJS_REL: &str = "../../../output/game_lora_tjs_t111.bin";

// ── CLI ────────────────────────────────────────────────────────

struct CliArgs {
    map_preset: Option<&'static str>,
    seed: u64,
    dense_path: PathBuf,
    tjs_path: PathBuf,
    rounds: usize,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut map_preset = None;
    let mut seed = 42u64;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dense_path = manifest.join(DEFAULT_DENSE_REL);
    let mut tjs_path = manifest.join(DEFAULT_TJS_REL);
    let mut rounds = ROUNDS;

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
            "--dense-path" if i + 1 < args.len() => {
                i += 1;
                dense_path = PathBuf::from(&args[i]);
            }
            "--tjs-path" if i + 1 < args.len() => {
                i += 1;
                tjs_path = PathBuf::from(&args[i]);
            }
            "--rounds" if i + 1 < args.len() => {
                i += 1;
                match args[i].parse::<usize>() {
                    Ok(r) if r > 0 => rounds = r,
                    _ => eprintln!("Note: invalid --rounds, using default {ROUNDS}"),
                }
            }
            _ => {}
        }
        i += 1;
    }

    CliArgs {
        map_preset,
        seed,
        dense_path,
        tjs_path,
        rounds,
    }
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
    /// Running ELO rating (starts at ELO_BASE, updated pairwise per round).
    elo: f64,
}

impl PlayerStats {
    fn new() -> Self {
        Self {
            elo: ELO_BASE,
            ..Self::default()
        }
    }

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

// ── Round ──────────────────────────────────────────────────────

struct RoundResult {
    scores: [i32; 4],
    survivors: Vec<u8>,
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

    if survivors.len() == 1 {
        scores[survivors[0] as usize] += 5;
    } else if survivors.len() > 1 {
        for &s in &survivors {
            scores[s as usize] += 3;
        }
    }

    RoundResult { scores, survivors }
}

/// Pairwise multi-player ELO update.
///
/// For every (i, j) pair in the 4-player match, update ELO based on:
/// 1. If exactly one of {i, j} survived → survivor wins.
/// 2. If both or neither survived → higher score wins; tie = no update.
///
/// This is the standard generalization of ELO to N-player games (used by
/// e.g. chess.com for multi-table tournaments). Each player's ELO is updated
/// against all 3 opponents per round.
fn update_elo_pairwise(stats: &mut [PlayerStats], result: &RoundResult) {
    let calc = EloCalculator {
        k: ELO_K,
        base: ELO_BASE,
    };
    let n = stats.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let i_surv = result.survivors.contains(&(i as u8));
            let j_surv = result.survivors.contains(&(j as u8));
            let i_wins = match (i_surv, j_surv) {
                (true, false) => true,
                (false, true) => false,
                _ => {
                    // Both survived or both died → compare scores.
                    let si = result.scores[i];
                    let sj = result.scores[j];
                    if si == sj {
                        continue; // tie → no ELO update
                    }
                    si > sj
                }
            };
            let (new_i, new_j) = calc.update(stats[i].elo, stats[j].elo, i_wins);
            stats[i].elo = new_i;
            stats[j].elo = new_j;
        }
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let cli = parse_args();
    let dense_exists = cli.dense_path.exists();
    let tjs_exists = cli.tjs_path.exists();

    println!("╔═══ Plan 300 T1.11 — TJS-LoRA vs Dense-LoRA Bomber Arena ═════╗");
    println!("║  {}-round head-to-head: TJS-LoRA vs Dense-LoRA              ║", cli.rounds);
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Dense LoRA: {} {}", cli.dense_path.display(), if dense_exists { "✓" } else { "⚠ missing" });
    println!("  TJS LoRA:   {} {}", cli.tjs_path.display(), if tjs_exists { "✓" } else { "⚠ missing" });
    println!("  Map:    {}", cli.map_preset.unwrap_or("procedural"));
    println!("  Seed:   {}", cli.seed);
    println!("  ELO:    k={ELO_K}, base={ELO_BASE}, pairwise survival-based");
    println!();

    // Print adapter info if files exist.
    for (label, path, exists) in [
        ("Dense", &cli.dense_path, dense_exists),
        ("TJS", &cli.tjs_path, tjs_exists),
    ] {
        if exists {
            match katgpt_rs::types::LoraAdapter::load(path) {
                Ok(adapters) => {
                    println!("  {label} LoRA adapters loaded: {}", adapters.len());
                    for (i, a) in adapters.iter().enumerate() {
                        println!(
                            "    [{}] rank={} alpha={:.1} in_dim={} out_dim={}",
                            i, a.rank, a.alpha, a.in_dim, a.out_dim
                        );
                    }
                }
                Err(e) => {
                    println!("  ⚠ {label} LoRA load error: {e}");
                }
            }
        } else {
            println!("  ⚠ {label} LoRA file not found — player will run in heuristic fallback mode");
        }
    }
    println!();

    let mut rng = Rng::with_seed(cli.seed);
    // P3 = Dense, P4 = TJS — both LoRA-backed, head-to-head.
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(SonltPlayer::new_with_lora(2, cli.dense_path.to_str().unwrap_or(""))),
        Box::new(SonltPlayer::new_with_lora(3, cli.tjs_path.to_str().unwrap_or(""))),
    ];

    println!("╔═══ Players ═══════════════════════════════════════════════════╗");
    println!("║  P1 🐰 Random | P2 🐱 Greedy | P3 🧠 Dense | P4 ✨ TJS   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let mut stats: Vec<PlayerStats> = (0..4).map(|_| PlayerStats::new()).collect();

    for round in 0..cli.rounds {
        let seed = cli.seed + round as u64;
        let result = run_round(seed, cli.map_preset, &mut players, &mut rng);

        for (i, s) in result.scores.iter().enumerate() {
            stats[i].total_score += *s as i64;
            stats[i].rounds_played += 1;
        }

        // Update ELO pairwise (mutates stats[].elo).
        update_elo_pairwise(&mut stats, &result);

        // Progress every 200 rounds.
        if (round + 1) % 200 == 0 || round + 1 == cli.rounds {
            let emoji = ["🐰", "🐱", "🧠", "✨"];
            let names = ["Random", "Greedy", "Dense", "TJS"];
            println!("  [Round {}/{}]", round + 1, cli.rounds);
            for i in 0..4 {
                println!(
                    "    {} {:<10} ELO={:7.1}  survival={:.1}%  avg_score={:+.1}",
                    emoji[i],
                    names[i],
                    stats[i].elo,
                    stats[i].survival_rate() * 100.0,
                    stats[i].avg_score(),
                );
            }
            println!();
        }
    }

    // ── Final Results ──────────────────────────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  FINAL RESULTS ({} rounds)", cli.rounds);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let emoji = ["🐰", "🐱", "🧠", "✨"];
    let names = ["Random", "Greedy", "Dense", "TJS"];
    let tech = ["(baseline)", "(heuristic)", "(+dense LoRA)", "(+TJS LoRA)"];

    println!(
        "  {:<4} {:<4} {:<10} {:<14} {:>8} {:>8} {:>10} {:>10}",
        "", "", "Player", "Tech", "ELO", "Surv%", "AvgScore", "Survival%"
    );
    println!("  {}", "─".repeat(80));

    let mut ranking: Vec<usize> = (0..4).collect();
    ranking.sort_by(|&a, &b| {
        stats[b]
            .elo
            .partial_cmp(&stats[a].elo)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, &idx) in ranking.iter().enumerate() {
        println!(
            "  #{:<3} {} {:<10} {:<14} {:>8.1} {:>7.1}% {:>+9.1} {:>9.1}%",
            rank + 1,
            emoji[idx],
            names[idx],
            tech[idx],
            stats[idx].elo,
            stats[idx].survival_rate() * 100.0,
            stats[idx].avg_score(),
            stats[idx].survival_rate() * 100.0,
        );
    }

    // ── GOAT Gate: T1.11 — TJS vs Dense ELO ────────────────────────

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GOAT GATE: T1.11 — TJS (P4 ✨) vs Dense (P3 🧠) ELO ratio");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  P3 🧠 Dense ELO: {:>9.1}", stats[2].elo);
    println!("  P4 ✨ TJS   ELO: {:>9.1}", stats[3].elo);
    println!();

    let elo_ratio = if stats[2].elo > 0.0 {
        stats[3].elo / stats[2].elo
    } else {
        0.0
    };
    let delta_elo = stats[3].elo - stats[2].elo;

    println!("  ELO ratio (TJS / Dense): {elo_ratio:.4}", );
    println!("  ELO delta  (TJS - Dense): {delta_elo:+.1}");
    println!("  Target ratio:            ≥ {:.2}", TJS_ELO_RATIO_TARGET);
    println!();

    let tjs_passes = elo_ratio >= TJS_ELO_RATIO_TARGET;
    if tjs_passes {
        println!("  ✅ T1.11 PASSED: TJS-LoRA achieves ≥ {:.0}% of Dense-LoRA ELO", TJS_ELO_RATIO_TARGET * 100.0);
        println!("     Paper finding (Zheng et al. 2026 §4.4) confirmed: the task-conditioned");
        println!("     Jacobian support mask recovers ≥95% of dense-LoRA quality at lower density.");
    } else if elo_ratio >= 0.90 {
        println!("  ⚠ T1.11 PARTIAL: TJS-LoRA achieves {:.1}% of Dense ELO (target {:.0}%)", elo_ratio * 100.0, TJS_ELO_RATIO_TARGET * 100.0);
        println!("     Close to gate but below threshold. Consider longer training, higher λ_sparse,");
        println!("     or longer warmup before finalizing the support mask.");
    } else {
        println!("  ❌ T1.11 NOT PASSED: TJS-LoRA achieves only {:.1}% of Dense ELO", elo_ratio * 100.0);
        println!("     The sparse mask is too aggressive. Inspect the TJS summary from training");
        println!("     (mask density, total support size) and tune hyperparameters.");
    }

    println!();
    println!("  Secondary metric — avg_score:");
    println!("    Dense: {:+.1}", stats[2].avg_score());
    println!("    TJS:   {:+.1}", stats[3].avg_score());
    println!();

    println!("═╡ Done ╞═");
}
