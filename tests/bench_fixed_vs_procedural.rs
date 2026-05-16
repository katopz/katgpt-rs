//! Benchmark: Fixed vs Procedural arena score variance (Issue 052, Task B5)
//!
//! Compares score variance between fixed STANDARD_ARENA maps and
//! procedurally generated maps across 100 games each.
//!
//! Expected: fixed map has lower coefficient of variation (CV = σ/μ)
//! than procedural maps, since the terrain layout is consistent.
//!
//! Run: `cargo test --features bomber bench_fixed_vs_procedural -- --nocapture`

#[cfg(feature = "bomber")]
use fastrand::Rng;

#[cfg(feature = "bomber")]
use microgpt_rs::pruners::bomber::arena::STANDARD_ARENA;

#[cfg(feature = "bomber")]
use microgpt_rs::pruners::bomber::{
    Alive, ArenaGrid, BomberPlayer, GameEvent, GreedyPlayer, GridPos, HLPlayer, RandomPlayer,
    ValidatorPlayer, init_world, init_world_with_arena, run_tick, spawn_players,
};

// ── Arena Source ────────────────────────────────────────────────

/// How to create the arena for a benchmark game.
#[cfg(feature = "bomber")]
enum ArenaSource {
    Fixed(ArenaGrid),
    Procedural(u64),
}

// ── Score ───────────────────────────────────────────────────────

/// Per-game scores computed from events.
#[cfg(feature = "bomber")]
#[derive(Clone, Copy, Debug, Default)]
struct GameScores {
    scores: [i32; 4],
}

// ── Stats ───────────────────────────────────────────────────────

/// Descriptive statistics for a sample of values.
#[cfg(feature = "bomber")]
struct Stats {
    mean: f64,
    std_dev: f64,
    cv: f64,
}

#[cfg(feature = "bomber")]
fn compute_stats(values: &[f64]) -> Stats {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let cv = match mean.abs() < f64::EPSILON {
        true => f64::INFINITY,
        false => std_dev / mean.abs(),
    };
    Stats { mean, std_dev, cv }
}

// ── Game Runner ─────────────────────────────────────────────────

/// Run a single game with actual players and compute scores from events.
#[cfg(feature = "bomber")]
fn run_game(
    source: ArenaSource,
    players: &mut [Box<dyn BomberPlayer>],
    rng: &mut Rng,
    tick_limit: u32,
) -> GameScores {
    let mut world = match source {
        ArenaSource::Fixed(arena) => init_world_with_arena(arena),
        ArenaSource::Procedural(seed) => init_world(seed),
    };
    let entities = spawn_players(&mut world);

    for p in players.iter_mut() {
        p.reset();
    }

    let mut all_events: Vec<GameEvent> = Vec::new();

    for _ in 0..tick_limit {
        // Drain tick events
        let tick_events: Vec<GameEvent> = {
            use bevy_ecs::event::Events;
            let mut ev = world.resource_mut::<Events<GameEvent>>();
            ev.drain().collect()
        };
        all_events.extend(tick_events.iter().cloned());

        // Select actions for alive players
        let mut actions = [None; 4];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world.get::<Alive>(entities[i]).is_some();
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

    match survivors.len() {
        0 => {}
        1 => {
            scores[survivors[0] as usize] += 5;
        }
        _ => {
            for &s in &survivors {
                scores[s as usize] += 3;
            }
        }
    }

    GameScores { scores }
}

/// Create fresh player instances for a game.
#[cfg(feature = "bomber")]
fn create_players() -> Vec<Box<dyn BomberPlayer>> {
    vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ]
}

// ── Test ────────────────────────────────────────────────────────

#[cfg(feature = "bomber")]
#[test]
fn bench_fixed_vs_procedural() {
    let n: usize = 100;
    let tick_limit: u32 = 200;
    let mut rng = Rng::with_seed(12345);

    // ── Fixed map: same STANDARD_ARENA every game ──
    let fixed_arena = ArenaGrid::fixed(STANDARD_ARENA).expect("standard arena should parse");

    let mut fixed_p4_scores: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut players = create_players();
        let result = run_game(
            ArenaSource::Fixed(fixed_arena.clone()),
            &mut players,
            &mut rng,
            tick_limit,
        );
        fixed_p4_scores.push(result.scores[3] as f64);
        std::hint::black_box(result);
    }

    // ── Procedural: different seed per game ──
    let mut proc_p4_scores: Vec<f64> = Vec::with_capacity(n);
    for seed in 0..n as u64 {
        let mut players = create_players();
        let result = run_game(
            ArenaSource::Procedural(seed),
            &mut players,
            &mut rng,
            tick_limit,
        );
        proc_p4_scores.push(result.scores[3] as f64);
        std::hint::black_box(result);
    }

    // ── Statistics ──
    let fixed_stats = compute_stats(&fixed_p4_scores);
    let proc_stats = compute_stats(&proc_p4_scores);

    println!("\n🧪 Fixed vs Procedural Arena — Score Variance ({n} games, P4 🐵 HL)");
    println!("{}", "═".repeat(65));
    println!();
    println!("  Fixed (STANDARD_ARENA, same map each game):");
    println!("    Mean:    {:+.2}", fixed_stats.mean);
    println!("    StdDev:  {:.2}", fixed_stats.std_dev);
    println!(
        "    CV:      {:.4} ({:.1}%)",
        fixed_stats.cv,
        fixed_stats.cv * 100.0
    );
    println!();
    println!("  Procedural (seeds 0..{n}, different map each game):");
    println!("    Mean:    {:+.2}", proc_stats.mean);
    println!("    StdDev:  {:.2}", proc_stats.std_dev);
    println!(
        "    CV:      {:.4} ({:.1}%)",
        proc_stats.cv,
        proc_stats.cv * 100.0
    );
    println!();
    println!(
        "  Fixed CV ({:.4}) < Procedural CV ({:.4}): {}",
        fixed_stats.cv,
        proc_stats.cv,
        match fixed_stats.cv < proc_stats.cv {
            true => "✅ YES",
            false => "❌ NO",
        }
    );

    assert!(
        fixed_stats.cv < proc_stats.cv,
        "Fixed map CV ({:.4}) should be < procedural CV ({:.4}) — \
         fixed maps produce less score variance",
        fixed_stats.cv,
        proc_stats.cv,
    );
}
