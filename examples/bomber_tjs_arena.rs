//! Plan 300 T1.11 (+ optional T1.10) — TJS-LoRA vs Dense-LoRA Bomber Arena.
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
//! # Optional GOAT gate (Plan 300 T1.10)
//!
//! When `--manifold-path` is supplied, a second 4-player arena (Arena B) runs
//! after Arena A using the TJS+ManifoldE adapter as P4. Both arenas share the
//! same seed/map sequence, so P1/P2/P3 face identical conditions — the only
//! difference between Arena A and Arena B is P4's adapter.
//!
//! T1.10 gate: TJS+ManifoldE ELO must be ≥ 99% of TJS-alone ELO (Arena B P4
//! vs Arena A P4). Confirms ManifoldE composition does not regress arena play.
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
//! # From katgpt-rs workspace root.
//!
//! # Arena A only (T1.11 gate):
//! cargo run --release --example bomber_tjs_arena --features bomber -- \
//!     --dense-path /path/to/game_lora_dense_t111.bin \
//!     --tjs-path   /path/to/game_lora_tjs_t111.bin \
//!     --rounds 1000
//!
//! # Dual-arena (T1.11 + T1.10 gates):
//! cargo run --release --example bomber_tjs_arena --features bomber -- \
//!     --dense-path    /path/to/game_lora_dense_t111.bin \
//!     --tjs-path      /path/to/game_lora_tjs_t111.bin \
//!     --manifold-path /path/to/game_lora_tjs_manifold_t110.bin \
//!     --rounds 1000
//! ```

#![cfg(feature = "bomber")]
#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};

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

/// T1.10 gate threshold: TJS+ManifoldE ELO must be ≥ this fraction of
/// TJS-alone ELO (Arena B P4 vs Arena A P4).
const MANIFOLD_ELO_RATIO_TARGET: f64 = 0.99;

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
    /// Optional TJS+ManifoldE adapter. When set, runs Arena B + T1.10 gate.
    manifold_path: Option<PathBuf>,
    rounds: usize,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut map_preset = None;
    let mut seed = 42u64;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut dense_path = manifest.join(DEFAULT_DENSE_REL);
    let mut tjs_path = manifest.join(DEFAULT_TJS_REL);
    let mut manifold_path: Option<PathBuf> = None;
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
            "--manifold-path" if i + 1 < args.len() => {
                i += 1;
                manifold_path = Some(PathBuf::from(&args[i]));
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
        manifold_path,
        rounds,
    }
}

// ── Stats ──────────────────────────────────────────────────────

#[derive(Clone, Default)]
#[allow(dead_code)] // demo stats: full surface retained for future scoreboards
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

#[allow(dead_code)] // demo stats: full surface retained for future scoreboards
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

// ── Arena runner ──────────────────────────────────────────────

/// Run one 4-player arena: {Random, Greedy, Dense-LoRA, P4-LoRA}.
///
/// Used for both Arena A (T1.11, P4=TJS) and Arena B (T1.10, P4=TJS+Manifold).
/// Both arenas use the same `seed` and `map_preset`, so P1/P2/P3 face identical
/// conditions — the only difference between Arena A and Arena B is P4's adapter.
///
/// Returns the per-player stats and P4's (index 3) final ELO.
fn run_arena(
    label: &str,
    dense_path: &Path,
    p4_path: &Path,
    p4_label: &str,
    map_preset: Option<&'static str>,
    seed: u64,
    rounds: usize,
) -> (Vec<PlayerStats>, f64) {
    // Resolve P4 display metadata from its label.
    let (p4_emoji, p4_name, p4_tech): (&str, &str, &str) = match p4_label {
        "TJS" => ("✨", "TJS", "(+TJS LoRA)"),
        "TJS+Manifold" => ("🌀", "TJS+Manifold", "(+TJS+Manifold LoRA)"),
        other => ("❓", other, "(+LoRA)"),
    };

    let dense_exists = dense_path.exists();
    let p4_exists = p4_path.exists();

    println!("── Arena {label} ── P4 {p4_emoji} {p4_name}-LoRA vs P3 🧠 Dense-LoRA ──");
    println!();
    println!("  Dense LoRA:   {} {}", dense_path.display(), if dense_exists { "✓" } else { "⚠ missing" });
    println!("  {p4_name} LoRA: {} {}", p4_path.display(), if p4_exists { "✓" } else { "⚠ missing" });
    println!("  Map:    {}", map_preset.unwrap_or("procedural"));
    println!("  Seed:   {seed}");
    println!("  ELO:    k={ELO_K}, base={ELO_BASE}, pairwise survival-based");
    println!();

    // Print adapter info if files exist.
    for (adapter_label, path, exists) in [
        ("Dense", dense_path, dense_exists),
        (p4_name, p4_path, p4_exists),
    ] {
        if exists {
            match katgpt_rs::types::LoraAdapter::load(path) {
                Ok(adapters) => {
                    println!("  {adapter_label} LoRA adapters loaded: {}", adapters.len());
                    for (i, a) in adapters.iter().enumerate() {
                        println!(
                            "    [{}] rank={} alpha={:.1} in_dim={} out_dim={}",
                            i, a.rank, a.alpha, a.in_dim, a.out_dim
                        );
                    }
                }
                Err(e) => {
                    println!("  ⚠ {adapter_label} LoRA load error: {e}");
                }
            }
        } else {
            println!("  ⚠ {adapter_label} LoRA file not found — player will run in heuristic fallback mode");
        }
    }
    println!();

    let mut rng = Rng::with_seed(seed);
    // P3 = Dense, P4 = {p4_label} — both LoRA-backed, head-to-head.
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(SonltPlayer::new_with_lora(2, dense_path.to_str().unwrap_or(""))),
        Box::new(SonltPlayer::new_with_lora(3, p4_path.to_str().unwrap_or(""))),
    ];

    println!("╔═══ Players (Arena {label}) ══════════════════════════════════════╗");
    println!("║  P1 🐰 Random | P2 🐱 Greedy | P3 🧠 Dense | P4 {p4_emoji} {p4_name:<10}║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    let mut stats: Vec<PlayerStats> = (0..4).map(|_| PlayerStats::new()).collect();

    for round in 0..rounds {
        let round_seed = seed + round as u64;
        let result = run_round(round_seed, map_preset, &mut players, &mut rng);

        for (i, s) in result.scores.iter().enumerate() {
            stats[i].total_score += *s as i64;
            stats[i].rounds_played += 1;
        }

        // Update ELO pairwise (mutates stats[].elo).
        update_elo_pairwise(&mut stats, &result);

        // Progress every 200 rounds.
        if (round + 1) % 200 == 0 || round + 1 == rounds {
            let emoji = ["🐰", "🐱", "🧠", p4_emoji];
            let names = ["Random", "Greedy", "Dense", p4_name];
            println!("  [Arena {label} · Round {}/{}]", round + 1, rounds);
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
    println!("  ARENA {label} FINAL RESULTS ({rounds} rounds)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let emoji = ["🐰", "🐱", "🧠", p4_emoji];
    let names = ["Random", "Greedy", "Dense", p4_name];
    let tech = ["(baseline)", "(heuristic)", "(+dense LoRA)", p4_tech];

    println!(
        "  {:<4} {:<4} {:<10} {:<14} {:>8} {:>8} {:>10} {:<10}",
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

    let p4_elo = stats[3].elo;
    (stats, p4_elo)
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let cli = parse_args();
    let dual_arena = cli.manifold_path.is_some();

    // Top-level banner.
    if dual_arena {
        println!("╔═══ Plan 300 T1.11 + T1.10 — TJS / TJS+ManifoldE vs Dense-LoRA ════╗");
        println!("║  Dual-arena mode: Arena A (T1.11) → Arena B (T1.10)              ║");
        println!("║  {}-round per arena, shared seeds/maps for fair P4 comparison    ║", cli.rounds);
        println!("╚══════════════════════════════════════════════════════════════════╝");
    } else {
        println!("╔═══ Plan 300 T1.11 — TJS-LoRA vs Dense-LoRA Bomber Arena ═════╗");
        println!("║  {}-round head-to-head: TJS-LoRA vs Dense-LoRA              ║", cli.rounds);
        println!("╚════════════════════════════════════════════════════════════════╝");
    }
    println!();

    // ── Arena A: T1.11 (TJS vs Dense) ─────────────────────────────
    let (stats_a, tjs_elo) = run_arena(
        "A",
        &cli.dense_path,
        &cli.tjs_path,
        "TJS",
        cli.map_preset,
        cli.seed,
        cli.rounds,
    );
    let dense_elo = stats_a[2].elo;

    // ── GOAT Gate: T1.11 — TJS vs Dense ELO ────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GOAT GATE: T1.11 — TJS (P4 ✨) vs Dense (P3 🧠) ELO ratio");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  P3 🧠 Dense ELO: {:>9.1}", dense_elo);
    println!("  P4 ✨ TJS   ELO: {:>9.1}", tjs_elo);
    println!();

    let tjs_ratio = if dense_elo > 0.0 {
        tjs_elo / dense_elo
    } else {
        0.0
    };
    let tjs_delta = tjs_elo - dense_elo;

    println!("  ELO ratio (TJS / Dense): {tjs_ratio:.4}");
    println!("  ELO delta  (TJS - Dense): {tjs_delta:+.1}");
    println!("  Target ratio:            ≥ {:.2}", TJS_ELO_RATIO_TARGET);
    println!();

    if tjs_ratio >= TJS_ELO_RATIO_TARGET {
        println!("  ✅ T1.11 PASSED: TJS-LoRA achieves ≥ {:.0}% of Dense-LoRA ELO", TJS_ELO_RATIO_TARGET * 100.0);
        println!("     Paper finding (Zheng et al. 2026 §4.4) confirmed: the task-conditioned");
        println!("     Jacobian support mask recovers ≥95% of dense-LoRA quality at lower density.");
    } else if tjs_ratio >= 0.90 {
        println!("  ⚠ T1.11 PARTIAL: TJS-LoRA achieves {:.1}% of Dense ELO (target {:.0}%)", tjs_ratio * 100.0, TJS_ELO_RATIO_TARGET * 100.0);
        println!("     Close to gate but below threshold. Consider longer training, higher λ_sparse,");
        println!("     or longer warmup before finalizing the support mask.");
    } else {
        println!("  ❌ T1.11 NOT PASSED: TJS-LoRA achieves only {:.1}% of Dense ELO", tjs_ratio * 100.0);
        println!("     The sparse mask is too aggressive. Inspect the TJS summary from training");
        println!("     (mask density, total support size) and tune hyperparameters.");
    }

    println!();
    println!("  Secondary metric — avg_score:");
    println!("    Dense: {:+.1}", stats_a[2].avg_score());
    println!("    TJS:   {:+.1}", stats_a[3].avg_score());
    println!();

    // ── Arena B: T1.10 (TJS+Manifold vs Dense, same seeds) ────────
    if let Some(manifold_path) = &cli.manifold_path {
        println!();
        println!("═══════════════════════════════════════════════════════════════");
        println!("  ARENA B — T1.10: TJS+ManifoldE vs Dense-LoRA (same seeds as A)");
        println!("═══════════════════════════════════════════════════════════════");

        let (stats_b, manifold_elo) = run_arena(
            "B",
            &cli.dense_path,
            manifold_path,
            "TJS+Manifold",
            cli.map_preset,
            cli.seed,
            cli.rounds,
        );

        // ── GOAT Gate: T1.10 — TJS+Manifold vs TJS-alone ELO ────────
        println!();
        println!("═══════════════════════════════════════════════════════════════");
        println!("  GOAT GATE: T1.10 — TJS+ManifoldE (Arena B P4 🌀) vs");
        println!("                     TJS-alone      (Arena A P4 ✨) ELO ratio");
        println!("═══════════════════════════════════════════════════════════════");
        println!();
        println!("  Arena A P4 ✨ TJS            ELO: {:>9.1}", tjs_elo);
        println!("  Arena B P4 🌀 TJS+Manifold   ELO: {:>9.1}", manifold_elo);
        println!();

        let manifold_ratio = if tjs_elo > 0.0 {
            manifold_elo / tjs_elo
        } else {
            0.0
        };
        let manifold_delta = manifold_elo - tjs_elo;

        println!("  ELO ratio (Manifold / TJS):  {manifold_ratio:.4}");
        println!("  ELO delta  (Manifold - TJS): {manifold_delta:+.1}");
        println!("  Target ratio:                ≥ {:.2}", MANIFOLD_ELO_RATIO_TARGET);
        println!();

        if manifold_ratio >= MANIFOLD_ELO_RATIO_TARGET {
            println!("  ✅ T1.10 PASSED: TJS+ManifoldE achieves ≥ {:.0}% of TJS-alone ELO", MANIFOLD_ELO_RATIO_TARGET * 100.0);
            println!("     Plan 300 T1.10 confirmed: at equal rank, the ManifoldE composition");
            println!("     does not regress arena ELO — the entropy-regularized manifold estimate");
            println!("     is compatible with the TJS sparse support mask.");
        } else if manifold_ratio >= 0.95 {
            println!("  ⚠ T1.10 PARTIAL: TJS+ManifoldE achieves {:.1}% of TJS-alone ELO (target {:.0}%)", manifold_ratio * 100.0, MANIFOLD_ELO_RATIO_TARGET * 100.0);
            println!("     Close to gate but below threshold. The ManifoldE head is slightly");
            println!("     destabilizing the TJS support mask. Consider lowering the manifold");
            println!("     mixing coefficient or warmup-stepping the entropy regularizer.");
        } else {
            println!("  ❌ T1.10 NOT PASSED: TJS+ManifoldE achieves only {:.1}% of TJS-alone ELO", manifold_ratio * 100.0);
            println!("     ManifoldE composition is regressing arena play. Inspect the");
            println!("     training curve for divergence between the TJS and Manifold heads,");
            println!("     and verify the manifold estimator is not amplifying noise.");
        }

        println!();
        println!("  Secondary metric — avg_score:");
        println!("    TJS           (Arena A): {:+.1}", stats_a[3].avg_score());
        println!("    TJS+Manifold  (Arena B): {:+.1}", stats_b[3].avg_score());
        println!();
    }

    println!("═╡ Done ╞═");
}
