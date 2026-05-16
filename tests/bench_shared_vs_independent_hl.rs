//! Tournament benchmark: 4× shared HL vs 4× independent HL in bomber arena.
//!
//! Compares convergence speed, win rate, survival rate, and average score
//! between agents sharing a Q-table vs agents with independent Q-tables.
//!
//! Run: `cargo test -p microgpt-rs --test bench_shared_vs_independent_hl
//!       --features "bomber,bandit" -- --nocapture`

#[cfg(feature = "bomber")]
use std::sync::Arc;

#[cfg(feature = "bomber")]
use fastrand::Rng;

#[cfg(feature = "bomber")]
use microgpt_rs::pruners::bomber::{
    ArenaGrid, BomberPlayer, GameEvent, GridPos, HLPlayer, init_world_with_arena, run_tick,
    spawn_players,
};

#[cfg(feature = "bomber")]
use microgpt_rs::pruners::bomber::arena::STANDARD_ARENA;

#[cfg(all(feature = "bomber", feature = "bandit"))]
use microgpt_rs::pruners::bomber::SharedBanditStats;

// ── Constants ──────────────────────────────────────────────────

#[cfg(feature = "bomber")]
const NUM_PLAYERS: usize = 4;

#[cfg(feature = "bomber")]
const TOTAL_GAMES: usize = 1000;

#[cfg(feature = "bomber")]
const BLOCK_SIZE: usize = 100;

#[cfg(feature = "bomber")]
const TICK_LIMIT: u32 = 200;

#[cfg(feature = "bomber")]
const ACTION_COUNT: usize = 7;

// ── Game Result ────────────────────────────────────────────────

#[cfg(feature = "bomber")]
struct GameResult {
    scores: [i32; NUM_PLAYERS],
    survivors: Vec<u8>,
    _deaths: Vec<u8>,
    kills: Vec<(u8, u8)>,
    powerups: Vec<(u8, u32)>,
}

// ── Run a single game ──────────────────────────────────────────

#[cfg(feature = "bomber")]
fn run_game(players: &mut [Box<dyn BomberPlayer>], rng: &mut Rng) -> GameResult {
    let arena = ArenaGrid::fixed(STANDARD_ARENA).expect("STANDARD_ARENA must be valid");
    let mut world = init_world_with_arena(arena);
    let entities = spawn_players(&mut world);

    // Reset per-round state (Q-values persist across games — that's the learning)
    for p in players.iter_mut() {
        p.reset();
    }

    let mut all_events: Vec<GameEvent> = Vec::new();

    for _tick in 0..TICK_LIMIT {
        // Drain events from previous tick
        let tick_events: Vec<GameEvent> = {
            use bevy_ecs::event::Events;
            let mut ev = world.resource_mut::<Events<GameEvent>>();
            ev.drain().collect()
        };
        all_events.extend(tick_events.iter().cloned());

        // Each alive player selects an action
        let mut actions = [None; NUM_PLAYERS];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world
                .get::<microgpt_rs::pruners::bomber::Alive>(entities[i])
                .is_some();
            if alive {
                let grid = world.resource::<ArenaGrid>().clone();
                actions[i] = Some(player.select_action(&grid, pos, &tick_events, rng));
            }
        }

        if !run_tick(&mut world, actions) {
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
    let mut scores = [0i32; NUM_PLAYERS];
    let mut _deaths = Vec::new();
    let mut kills = Vec::new();
    let mut powerups = Vec::new();
    let mut survivors = Vec::new();

    for event in &all_events {
        match event {
            GameEvent::PlayerKilled { victim, killer } => {
                _deaths.push(*victim);
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

    GameResult {
        scores,
        survivors,
        _deaths,
        kills,
        powerups,
    }
}

// ── Q-value spread extraction ──────────────────────────────────

/// Parse Q-values from HLPlayer's compress_report string.
/// Format: "Pulls=X Compressed=Y/6 [...] Q=[↑:0.50 ↓:1.20 ...]"
#[cfg(feature = "bomber")]
fn parse_q_values(report: &str) -> Vec<f32> {
    let q_start = match report.find("Q=[") {
        Some(i) => i + 3,
        None => return vec![0.0; ACTION_COUNT],
    };
    let q_end = match report[q_start..].find(']') {
        Some(i) => q_start + i,
        None => return vec![0.0; ACTION_COUNT],
    };
    let q_section = &report[q_start..q_end];

    q_section
        .split_whitespace()
        .filter_map(|pair| {
            let parts: Vec<&str> = pair.split(':').collect();
            if parts.len() == 2 {
                parts[1].parse::<f32>().ok()
            } else {
                None
            }
        })
        .collect()
}

/// Compute Q-value spread (max - min) from a vector of Q-values.
#[cfg(feature = "bomber")]
fn q_value_spread(q_values: &[f32]) -> f32 {
    if q_values.is_empty() {
        return 0.0;
    }
    let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_q = q_values.iter().cloned().fold(f32::INFINITY, f32::min);
    max_q - min_q
}

/// Compute Q-value spread from SharedBanditStats.
#[cfg(all(feature = "bomber", feature = "bandit"))]
fn shared_q_spread(stats: &SharedBanditStats) -> f32 {
    let q_values: Vec<f32> = (0..ACTION_COUNT).map(|i| stats.q_value(i)).collect();
    q_value_spread(&q_values)
}

// ── Tournament Result ──────────────────────────────────────────

#[cfg(feature = "bomber")]
struct TournamentResult {
    total_scores: [i32; NUM_PLAYERS],
    wins: [u32; NUM_PLAYERS],
    survival_count: [u32; NUM_PLAYERS],
    convergence_blocks: Vec<f32>, // Q-value spread at each 100-game block boundary
}

// ── Independent HL Tournament ──────────────────────────────────

#[cfg(feature = "bomber")]
fn run_independent_tournament() -> TournamentResult {
    let mut rng = Rng::new();
    let mut players: Vec<Box<dyn BomberPlayer>> = (0..NUM_PLAYERS)
        .map(|i| Box::new(HLPlayer::new(i as u8)) as Box<dyn BomberPlayer>)
        .collect();

    let mut total_scores = [0i32; NUM_PLAYERS];
    let mut wins = [0u32; NUM_PLAYERS];
    let mut survival_count = [0u32; NUM_PLAYERS];
    let mut convergence_blocks: Vec<f32> = Vec::new();

    for game in 0..TOTAL_GAMES {
        let result = run_game(&mut players, &mut rng);

        // Accumulate stats
        for (i, s) in result.scores.iter().enumerate() {
            total_scores[i] += s;
        }
        for &s in &result.survivors {
            survival_count[s as usize] += 1;
        }
        if result.survivors.len() == 1 {
            wins[result.survivors[0] as usize] += 1;
        }

        // Update outcomes — distribute reward across actions taken this round
        for (i, player) in players.iter_mut().enumerate() {
            let survived = result.survivors.contains(&(i as u8));
            let killed = result.kills.iter().any(|(k, _)| *k == i as u8);
            let powerup_count = result
                .powerups
                .iter()
                .filter(|(p, _)| *p == i as u8)
                .count() as u32;
            if let Some(hl) = player.as_any_mut().downcast_mut::<HLPlayer>() {
                hl.update_outcome(survived, killed, powerup_count);
            }
        }

        // Convergence checkpoint at block boundaries
        if (game + 1) % BLOCK_SIZE == 0 {
            let spreads: Vec<f32> = players
                .iter()
                .filter_map(|p| {
                    p.as_any()
                        .downcast_ref::<HLPlayer>()
                        .map(|hl| q_value_spread(&parse_q_values(&hl.compress_report())))
                })
                .collect();

            let mean_spread = if spreads.is_empty() {
                0.0
            } else {
                spreads.iter().sum::<f32>() / spreads.len() as f32
            };
            convergence_blocks.push(mean_spread);
        }
    }

    TournamentResult {
        total_scores,
        wins,
        survival_count,
        convergence_blocks,
    }
}

// ── Shared HL Tournament ───────────────────────────────────────

#[cfg(all(feature = "bomber", feature = "bandit"))]
fn run_shared_tournament() -> TournamentResult {
    let mut rng = Rng::new();
    let shared_stats = Arc::new(SharedBanditStats::new(ACTION_COUNT));

    let mut players: Vec<Box<dyn BomberPlayer>> = (0..NUM_PLAYERS)
        .map(|i| {
            Box::new(HLPlayer::with_shared_stats(
                i as u8,
                shared_stats.clone(),
                None,
            )) as Box<dyn BomberPlayer>
        })
        .collect();

    let mut total_scores = [0i32; NUM_PLAYERS];
    let mut wins = [0u32; NUM_PLAYERS];
    let mut survival_count = [0u32; NUM_PLAYERS];
    let mut convergence_blocks: Vec<f32> = Vec::new();

    for game in 0..TOTAL_GAMES {
        let result = run_game(&mut players, &mut rng);

        // Accumulate stats
        for (i, s) in result.scores.iter().enumerate() {
            total_scores[i] += s;
        }
        for &s in &result.survivors {
            survival_count[s as usize] += 1;
        }
        if result.survivors.len() == 1 {
            wins[result.survivors[0] as usize] += 1;
        }

        // Update outcomes — all agents write to the shared Q-table
        for (i, player) in players.iter_mut().enumerate() {
            let survived = result.survivors.contains(&(i as u8));
            let killed = result.kills.iter().any(|(k, _)| *k == i as u8);
            let powerup_count = result
                .powerups
                .iter()
                .filter(|(p, _)| *p == i as u8)
                .count() as u32;
            if let Some(hl) = player.as_any_mut().downcast_mut::<HLPlayer>() {
                hl.update_outcome(survived, killed, powerup_count);
            }
        }

        // Convergence checkpoint at block boundaries — single shared Q-table
        if (game + 1) % BLOCK_SIZE == 0 {
            let spread = shared_q_spread(&shared_stats);
            convergence_blocks.push(spread);
        }
    }

    TournamentResult {
        total_scores,
        wins,
        survival_count,
        convergence_blocks,
    }
}

// ── Print helpers ──────────────────────────────────────────────

#[cfg(feature = "bomber")]
fn print_tournament_result(label: &str, result: &TournamentResult) {
    let avg_score = result.total_scores.iter().sum::<i32>() as f64 / NUM_PLAYERS as f64;
    let total_wins: u32 = result.wins.iter().sum();
    let win_rate = (total_wins as f64 / TOTAL_GAMES as f64) * 100.0;
    let total_survivals: u32 = result.survival_count.iter().sum();
    let survival_rate =
        (total_survivals as f64 / (TOTAL_GAMES as f64 * NUM_PLAYERS as f64)) * 100.0;

    println!("{label} ({NUM_PLAYERS} players, {TOTAL_GAMES} games):");
    println!(
        "  Avg Score: {avg_score:+.1}  Win Rate: {win_rate:.0}%  Survival Rate: {survival_rate:.0}%"
    );

    // Print convergence at key blocks: 1, 5, 10
    let block_indices: Vec<usize> = vec![0, 4, 9];
    let convergence_parts: Vec<String> = block_indices
        .iter()
        .filter_map(|&idx| {
            result.convergence_blocks.get(idx).map(|&spread| {
                let block_num = idx + 1;
                format!("block {block_num}={spread:.2}")
            })
        })
        .collect();

    if !convergence_parts.is_empty() {
        println!("  Q-value convergence: {}", convergence_parts.join(", "));
    }
    println!();
}

// ── Main Benchmark Test ────────────────────────────────────────

#[cfg(feature = "bomber")]
#[test]
fn bench_shared_vs_independent_hl() {
    println!("\n{}", "═".repeat(70));
    println!("  Tournament Benchmark: Shared HL vs Independent HL");
    println!("{}", "═".repeat(70));
    println!();

    // Phase 1: Independent team — each HLPlayer has its own Q-table
    let independent = run_independent_tournament();
    print_tournament_result("Independent HL", &independent);

    // Phase 2: Shared team — all 4 HLPlayers share one Q-table (requires bandit)
    #[cfg(feature = "bandit")]
    {
        let shared = run_shared_tournament();
        print_tournament_result("Shared HL", &shared);
    }

    #[cfg(not(feature = "bandit"))]
    {
        println!("  (Shared HL tournament requires 'bandit' feature)");
        println!("  Run with: cargo test --features \"bomber,bandit\" -- --nocapture");
    }

    // Sanity: independent players should have accumulated non-zero scores
    let total: i32 = independent.total_scores.iter().sum();
    assert!(
        total != 0,
        "Expected non-zero total scores after {TOTAL_GAMES} games"
    );
}
