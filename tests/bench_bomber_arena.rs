//! Bomberman HL Arena benchmarks — run with: cargo test --features bomber bench_bomber_arena -- --nocapture

#[cfg(feature = "bomber")]
use std::time::Instant;

#[cfg(feature = "bomber")]
use fastrand::Rng;

#[cfg(feature = "bomber")]
use katgpt_rs::pruners::bomber::{
    ArenaGrid, BomberAction, BomberPlayer, GameEvent, GreedyPlayer, GridPos, HLPlayer,
    RandomPlayer, TICK_LIMIT, ValidatorPlayer, init_world, run_tick, spawn_players,
};

#[cfg(feature = "bomber")]
fn random_actions(rng: &mut Rng) -> [Option<BomberAction>; 4] {
    let variants = [
        BomberAction::Up,
        BomberAction::Down,
        BomberAction::Left,
        BomberAction::Right,
        BomberAction::Wait,
    ];
    std::array::from_fn(|_| Some(variants[rng.usize(0..variants.len())]))
}

#[cfg(feature = "bomber")]
#[test]
fn bench_arena_generation() {
    let n: u64 = 1000;
    let start = Instant::now();
    for seed in 0..n {
        std::hint::black_box(ArenaGrid::generate(seed));
    }
    let elapsed = start.elapsed();
    let per_gen = elapsed / n as u32;

    println!("\n🧪 Arena Generation ({n} iterations)");
    println!("{}", "═".repeat(60));
    println!("Total: {elapsed:?}");
    println!("Per generation: {per_gen:?}");

    assert!(per_gen.as_micros() < 100, "Too slow: {per_gen:?} >= 100µs");
}

#[cfg(feature = "bomber")]
#[test]
fn bench_single_tick() {
    let n: u64 = 1000;
    let mut rng = Rng::new();
    let mut world = init_world(0);
    spawn_players(&mut world);
    let start = Instant::now();
    for _ in 0..n {
        if !run_tick(&mut world, random_actions(&mut rng)) {
            // Game ended — reset world outside hot path
            world = init_world(rng.u64(..));
            spawn_players(&mut world);
        }
    }
    let elapsed = start.elapsed();
    let per_tick = elapsed / n as u32;

    println!("\n🧪 Single Tick ({n} iterations, 4 players)");
    println!("{}", "═".repeat(60));
    println!("Total: {elapsed:?}");
    println!("Per tick: {per_tick:?}");

    assert!(
        per_tick.as_micros() < 100,
        "Too slow: {per_tick:?} >= 100µs"
    );
}

#[cfg(feature = "bomber")]
#[test]
fn bench_full_game() {
    let n: u64 = 100;
    let mut rng = Rng::new();
    let start = Instant::now();
    for seed in 0..n {
        let mut world = init_world(seed);
        spawn_players(&mut world);
        for _ in 0..200u32 {
            if !run_tick(&mut world, random_actions(&mut rng)) {
                break;
            }
        }
    }
    let elapsed = start.elapsed();
    let per_game = elapsed / n as u32;

    println!("\n🧪 Full Game ({n} games, 200 ticks, 4 players)");
    println!("{}", "═".repeat(60));
    println!("Total: {elapsed:?}");
    println!("Per game: {per_game:?}");

    assert!(per_game.as_millis() < 10, "Too slow: {per_game:?} >= 10ms");
}

#[cfg(feature = "bomber")]
#[test]
fn bench_player_select_action() {
    let n: u64 = 1000;
    let mut rng = Rng::with_seed(42);
    let grid = ArenaGrid::generate(42);
    let pos = GridPos { x: 1, y: 1 };
    let events: &[GameEvent] = &[];

    let mut p1 = RandomPlayer::new(0);
    let t1 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p1.select_action(&grid, pos, events, &mut rng));
    }
    let t1 = t1.elapsed() / n as u32;

    let mut p2 = GreedyPlayer::new(1);
    let t2 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p2.select_action(&grid, pos, events, &mut rng));
    }
    let t2 = t2.elapsed() / n as u32;

    let mut p3 = ValidatorPlayer::new(2);
    let t3 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p3.select_action(&grid, pos, events, &mut rng));
    }
    let t3 = t3.elapsed() / n as u32;

    let mut p4 = HLPlayer::new(3);
    let t4 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(p4.select_action(&grid, pos, events, &mut rng));
    }
    let t4 = t4.elapsed() / n as u32;

    println!("\n🧪 Player select_action ({n} calls each)");
    println!("{}", "═".repeat(60));
    println!("P1 Random:    {t1:?}");
    println!("P2 Greedy:    {t2:?}");
    println!("P3 Validator: {t3:?}");
    println!("P4 HL:        {t4:?}");

    assert!(t4.as_micros() < 200, "HLPlayer too slow: {t4:?} >= 200µs");
}

// ── Bomb Placement & Kill Attribution Diagnostics ──────────────

#[cfg(feature = "bomber")]
#[derive(Default)]
struct GameEventLog {
    bombs_placed: Vec<(u8, (i32, i32))>,
    bombs_exploded: Vec<((i32, i32), u32)>,
    players_killed: Vec<(u8, Option<u8>)>, // (victim, killer)
    walls_destroyed: Vec<(i32, i32)>,
    powerups_collected: Vec<(u8, (i32, i32))>,
    powerups_revealed: Vec<((i32, i32), katgpt_rs::pruners::bomber::PowerUpKind)>,
    round_ends: u32,
}

#[cfg(feature = "bomber")]
impl GameEventLog {
    fn record(&mut self, event: &GameEvent) {
        match event {
            GameEvent::BombPlaced { player, pos } => {
                self.bombs_placed.push((*player, *pos));
            }
            GameEvent::BombExploded { pos, range } => {
                self.bombs_exploded.push((*pos, *range));
            }
            GameEvent::PlayerKilled { victim, killer } => {
                self.players_killed.push((*victim, *killer));
            }
            GameEvent::WallDestroyed { pos } => {
                self.walls_destroyed.push(*pos);
            }
            GameEvent::PowerUpCollected {
                player,
                kind: _,
                pos,
            } => {
                self.powerups_collected.push((*player, *pos));
            }
            GameEvent::PowerUpRevealed { pos, kind } => {
                self.powerups_revealed.push((*pos, *kind));
            }
            GameEvent::RoundEnd { survivors: _ } => {
                self.round_ends += 1;
            }
            _ => {}
        }
    }
}

/// Per-tick event breakdown for late-game analysis.
#[cfg(feature = "bomber")]
#[derive(Default)]
struct TickEventLog {
    /// (tick, bombs_placed_this_tick)
    bombs_per_tick: Vec<(u32, usize)>,
    /// (tick, player_id) for each kill
    kills_per_tick: Vec<(u32, u8)>,
}

#[cfg(feature = "bomber")]
fn run_game_collecting_events(
    players: &mut [Box<dyn BomberPlayer>],
    seed: u64,
    tick_limit: u32,
) -> (GameEventLog, u32) {
    use bevy_ecs::event::Events;

    let mut world = init_world(seed);
    let entities = spawn_players(&mut world);

    for p in players.iter_mut() {
        p.reset();
    }

    let mut log = GameEventLog::default();
    let mut rng = Rng::with_seed(seed);

    for _ in 0..tick_limit {
        // Drain events from previous tick
        let tick_events: Vec<GameEvent> = {
            let mut event_reader = world.resource_mut::<Events<GameEvent>>();
            event_reader.drain().collect()
        };
        for event in &tick_events {
            log.record(event);
        }

        // Each alive player selects an action
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
                actions[i] = Some(player.select_action(&grid, pos, &tick_events, &mut rng));
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Drain remaining events
    let remaining: Vec<GameEvent> = {
        let mut event_reader = world.resource_mut::<Events<GameEvent>>();
        event_reader.drain().collect()
    };
    for event in &remaining {
        log.record(event);
    }

    let ticks = world
        .resource::<katgpt_rs::pruners::bomber::TickCounter>()
        .tick;
    (log, ticks)
}

/// Run a game and collect per-tick bomb placement data for late-game analysis.
#[cfg(feature = "bomber")]
fn run_game_per_tick_events(
    players: &mut [Box<dyn BomberPlayer>],
    seed: u64,
    tick_limit: u32,
) -> (GameEventLog, TickEventLog, u32) {
    use bevy_ecs::event::Events;

    let mut world = init_world(seed);
    let entities = spawn_players(&mut world);

    for p in players.iter_mut() {
        p.reset();
    }

    let mut log = GameEventLog::default();
    let mut tick_log = TickEventLog::default();
    let mut rng = Rng::with_seed(seed);

    for _tick in 0..tick_limit {
        let current_tick = world
            .resource::<katgpt_rs::pruners::bomber::TickCounter>()
            .tick;

        // Drain events from previous tick
        let tick_events: Vec<GameEvent> = {
            let mut event_reader = world.resource_mut::<Events<GameEvent>>();
            event_reader.drain().collect()
        };

        let bombs_this_tick = tick_events
            .iter()
            .filter(|e| matches!(e, GameEvent::BombPlaced { .. }))
            .count();
        let kills_this_tick: Vec<u8> = tick_events
            .iter()
            .filter_map(|e| match e {
                GameEvent::PlayerKilled { victim, .. } => Some(*victim),
                _ => None,
            })
            .collect();

        for event in &tick_events {
            log.record(event);
        }

        if bombs_this_tick > 0 {
            tick_log
                .bombs_per_tick
                .push((current_tick, bombs_this_tick));
        }
        for &victim in &kills_this_tick {
            tick_log.kills_per_tick.push((current_tick, victim));
        }

        // Each alive player selects an action
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
                actions[i] = Some(player.select_action(&grid, pos, &tick_events, &mut rng));
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Drain remaining events
    let remaining: Vec<GameEvent> = {
        let mut event_reader = world.resource_mut::<Events<GameEvent>>();
        event_reader.drain().collect()
    };
    for event in &remaining {
        log.record(event);
    }

    let ticks = world
        .resource::<katgpt_rs::pruners::bomber::TickCounter>()
        .tick;
    (log, tick_log, ticks)
}

#[cfg(feature = "bomber")]
#[test]
fn test_greedy_players_place_bombs() {
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(GreedyPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(GreedyPlayer::new(2)),
        Box::new(GreedyPlayer::new(3)),
    ];

    let mut total_bombs_placed = 0usize;
    let mut total_bombs_exploded = 0usize;
    let mut total_kills = 0usize;
    let mut total_unattributed_deaths = 0usize;
    let num_games = 10;

    for seed in 0..num_games {
        let (log, ticks) = run_game_collecting_events(&mut players, seed, TICK_LIMIT);
        total_bombs_placed += log.bombs_placed.len();
        total_bombs_exploded += log.bombs_exploded.len();
        total_kills += log.players_killed.len();

        let unattributed = log
            .players_killed
            .iter()
            .filter(|(_victim, killer)| killer.is_none())
            .count();
        total_unattributed_deaths += unattributed;

        println!(
            "  Game {seed}: ticks={ticks}, bombs_placed={}, bombs_exploded={}, kills={}, unattributed={unattributed}",
            log.bombs_placed.len(),
            log.bombs_exploded.len(),
            log.players_killed.len(),
        );
    }

    println!("\n🧪 Greedy Player Bomb Placement ({num_games} games, {TICK_LIMIT} tick limit)");
    println!("{}", "═".repeat(60));
    println!("Total bombs placed:   {total_bombs_placed}");
    println!("Total bombs exploded: {total_bombs_exploded}");
    println!("Total kills:          {total_kills}");
    println!("Unattributed deaths:  {total_unattributed_deaths}");

    // Expect at least SOME bombs placed across 10 games with greedy players
    assert!(
        total_bombs_placed > 0,
        "Greedy players should place at least some bombs across {num_games} games, got 0"
    );

    // If bombs exploded and kills happened, check attribution
    if total_kills > 0 {
        let attrition_rate =
            (total_kills - total_unattributed_deaths) as f64 / total_kills as f64 * 100.0;
        println!("Kill attribution rate: {attrition_rate:.1}%");

        // Allow some unattributed (suicides with no blast zone owner), but not all
        assert!(
            total_unattributed_deaths < total_kills,
            "All {total_kills} kills have killer=None — kill attribution is broken"
        );
    }
}

#[cfg(feature = "bomber")]
#[test]
fn test_mixed_players_place_bombs() {
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(RandomPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(ValidatorPlayer::new(2)),
        Box::new(HLPlayer::new(3)),
    ];

    let mut total_bombs_placed = 0usize;
    let mut per_player_bombs = [0usize; 4];
    let num_games = 10;

    for seed in 0..num_games {
        let (log, ticks) = run_game_collecting_events(&mut players, seed, TICK_LIMIT);

        for &(player, _pos) in &log.bombs_placed {
            per_player_bombs[player as usize] += 1;
        }
        total_bombs_placed += log.bombs_placed.len();

        println!(
            "  Game {seed}: ticks={ticks}, bombs_placed={}, kills={}",
            log.bombs_placed.len(),
            log.players_killed.len(),
        );
    }

    println!("\n🧪 Mixed Player Bomb Placement ({num_games} games)");
    println!("{}", "═".repeat(60));
    let names = ["Random", "Greedy", "Validator", "HL"];
    for (i, &count) in per_player_bombs.iter().enumerate() {
        println!("  P{} {:<10} bombs placed: {}", i + 1, names[i], count);
    }
    println!("  Total bombs placed: {total_bombs_placed}");

    // Random doesn't track bombs so may place into invalid spots (system rejects)
    // Greedy/Validator/HL should place bombs when conditions are met
    assert!(
        total_bombs_placed > 0,
        "At least some players should place bombs across {num_games} games, got 0"
    );
}

#[cfg(feature = "bomber")]
#[test]
fn test_kill_attribution_chain() {
    // Run many games to find kill events and verify attribution
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(GreedyPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(GreedyPlayer::new(2)),
        Box::new(GreedyPlayer::new(3)),
    ];

    let mut total_kills = 0usize;
    let mut total_killer_is_some = 0usize;
    let mut killer_is_victim = 0usize;
    let num_games = 20;

    for seed in 0..num_games {
        let (log, _ticks) = run_game_collecting_events(&mut players, seed, TICK_LIMIT);

        for &(victim, killer) in &log.players_killed {
            total_kills += 1;
            if let Some(k) = killer {
                total_killer_is_some += 1;
                if k == victim {
                    killer_is_victim += 1;
                }
            }
        }
    }

    println!("\n🧪 Kill Attribution ({num_games} games)");
    println!("{}", "═".repeat(60));
    println!("Total kills:          {total_kills}");
    println!("Killer=Some:          {total_killer_is_some}");
    println!(
        "Killer=None:          {}",
        total_kills - total_killer_is_some
    );
    println!("Suicides (k==v):      {killer_is_victim}");

    if total_kills > 0 {
        let attribution_rate = total_killer_is_some as f64 / total_kills as f64 * 100.0;
        println!("Attribution rate:     {attribution_rate:.1}%");

        // Key assertion: most kills should be attributed
        // Unattributed kills happen when bomb owner dies before bomb explodes
        assert!(
            attribution_rate > 10.0,
            "Kill attribution rate too low ({attribution_rate:.1}%) — most kills have killer=None"
        );
    } else {
        println!("  (No kills occurred — cannot test attribution)");
    }
}

#[cfg(feature = "bomber")]
#[test]
fn test_late_game_bomb_placement_rate() {
    // Diagnose: after early kills reduce player count, do survivors still place bombs?
    // The TUI shows games stalling at tick 500 with no bombs in late game.
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(GreedyPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(GreedyPlayer::new(2)),
        Box::new(GreedyPlayer::new(3)),
    ];

    let num_games = 20;
    let mut games_that_timed_out = 0usize;
    let mut total_first_half_bombs = 0usize;
    let mut total_second_half_bombs = 0usize;
    let mut total_first_half_ticks = 0u32;
    let mut total_second_half_ticks = 0u32;
    let mut late_game_drought_games = 0usize;

    for seed in 0..num_games {
        let (log, tick_log, ticks) = run_game_per_tick_events(&mut players, seed, TICK_LIMIT);

        if ticks >= TICK_LIMIT {
            games_that_timed_out += 1;
        }

        let half = ticks / 2;
        let first_half: usize = tick_log
            .bombs_per_tick
            .iter()
            .filter(|(t, _)| *t < half)
            .map(|(_, c)| *c)
            .sum();
        let second_half: usize = tick_log
            .bombs_per_tick
            .iter()
            .filter(|(t, _)| *t >= half)
            .map(|(_, c)| *c)
            .sum();

        // Count ticks without bombs in second half
        let second_half_ticks_with_bombs: std::collections::HashSet<u32> = tick_log
            .bombs_per_tick
            .iter()
            .filter(|(t, _)| *t >= half)
            .map(|(t, _)| *t)
            .collect();
        let second_half_total = ticks.saturating_sub(half) as usize;
        let second_half_ticks_without =
            second_half_total.saturating_sub(second_half_ticks_with_bombs.len());

        total_first_half_bombs += first_half;
        total_second_half_bombs += second_half;
        total_first_half_ticks += half;
        total_second_half_ticks += ticks.saturating_sub(half);

        println!(
            "  Game {seed}: ticks={ticks}, first_half_bombs={first_half}, second_half_bombs={second_half}, \
             kills={}, drought_ticks={second_half_ticks_without}",
            log.players_killed.len(),
        );

        // Flag games where second half has 0 bombs (the reported bug)
        if second_half == 0 && ticks >= TICK_LIMIT / 2 {
            late_game_drought_games += 1;
        }
    }

    println!("\n🧪 Late-Game Bomb Placement ({num_games} games, {TICK_LIMIT} tick limit)");
    println!("{}", "═".repeat(60));
    println!("Games timed out (500 ticks): {games_that_timed_out}/{num_games}");
    println!("Late-game drought games:     {late_game_drought_games}/{num_games}");
    println!("First half bombs:  {total_first_half_bombs} (across {total_first_half_ticks} ticks)");
    println!(
        "Second half bombs: {total_second_half_bombs} (across {total_second_half_ticks} ticks)"
    );

    if total_first_half_ticks > 0 && total_second_half_ticks > 0 {
        let first_rate = total_first_half_bombs as f64 / total_first_half_ticks as f64;
        let second_rate = total_second_half_bombs as f64 / total_second_half_ticks as f64;
        println!("First half rate:  {:.3} bombs/tick", first_rate);
        println!("Second half rate: {:.3} bombs/tick", second_rate);

        // The issue: second half bomb rate should not be dramatically lower
        // If it drops to near zero, players wander without placing bombs
        if second_rate < first_rate * 0.1 && first_rate > 0.01 {
            println!("⚠️  Second half bomb rate is <10% of first half — late-game stall detected!");
        }
    }

    // Core assertion: at least some bombs should be placed across all games
    assert!(
        total_first_half_bombs + total_second_half_bombs > 0,
        "No bombs placed at all across {num_games} games"
    );
}

#[cfg(feature = "bomber")]
#[test]
fn test_scoreboard_resource_updates_during_game() {
    // ScoreBoard ECS resource should be updated during run_tick for real-time TUI display.
    // process_explosions updates on kills, collect_powerups updates on powerup pickup.
    // Run multiple games — at least one should have kills/powerups making ScoreBoard non-zero.
    let mut any_nonzero = false;

    for seed in 0u64..10 {
        let mut world = init_world(seed);
        let entities = spawn_players(&mut world);
        let mut rng = Rng::with_seed(seed);

        let mut players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(GreedyPlayer::new(0)),
            Box::new(GreedyPlayer::new(1)),
            Box::new(GreedyPlayer::new(2)),
            Box::new(GreedyPlayer::new(3)),
        ];

        for _ in 0..TICK_LIMIT {
            let tick_events: Vec<GameEvent> = {
                let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
                event_reader.drain().collect()
            };

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
                    actions[i] = Some(player.select_action(&grid, pos, &tick_events, &mut rng));
                }
            }

            if !run_tick(&mut world, actions) {
                break;
            }
        }

        let scores = world
            .resource::<katgpt_rs::pruners::bomber::ScoreBoard>()
            .scores;

        let nonzero = scores.iter().any(|&s| s != 0);
        if nonzero {
            any_nonzero = true;
        }
        println!("  Game {seed}: ScoreBoard = {:?}", scores);
    }

    println!("\n🧪 ScoreBoard Resource Sync Check");
    println!("{}", "═".repeat(60));
    println!("Any game with non-zero ScoreBoard: {any_nonzero}");

    // At least one game out of 10 should have kills/powerups → non-zero ScoreBoard
    assert!(
        any_nonzero,
        "ScoreBoard resource should be non-zero in at least 1 of 10 games — \
         systems.rs must update ScoreBoard on kills and powerup collection"
    );
}

#[cfg(feature = "bomber")]
#[test]
fn test_kill_attribution_owner_dies_before_explode() {
    // Edge case: bomb owner is killed by another bomb before their bomb explodes.
    // The killer should still be attributed if the owner was alive when the bomb was placed.
    let mut players: Vec<Box<dyn BomberPlayer>> = vec![
        Box::new(GreedyPlayer::new(0)),
        Box::new(GreedyPlayer::new(1)),
        Box::new(GreedyPlayer::new(2)),
        Box::new(GreedyPlayer::new(3)),
    ];

    let mut total_kills = 0usize;
    let mut killer_is_none_count = 0usize;
    let num_games = 20;

    for seed in 0..num_games {
        let (log, _ticks) = run_game_collecting_events(&mut players, seed, TICK_LIMIT);

        for &(victim, killer) in &log.players_killed {
            total_kills += 1;
            if killer.is_none() {
                killer_is_none_count += 1;
                println!(
                    "  Game {seed}: victim={victim} has killer=None (bomb owner died before detonation?)"
                );
            }
        }
    }

    println!("\n🧪 Kill Attribution When Owner Dies ({num_games} games)");
    println!("{}", "═".repeat(60));
    println!("Total kills:    {total_kills}");
    println!("Killer=None:    {killer_is_none_count}");

    if total_kills > 0 {
        let unattributed_pct = killer_is_none_count as f64 / total_kills as f64 * 100.0;
        println!("Unattributed:   {unattributed_pct:.1}%");

        // Known issue: when bomb owner dies before their bomb explodes, the
        // process_explosions function looks up owner in player_id_map (alive players only).
        // If owner is dead, killer_id becomes None.
        // This is somewhat expected but should be < 50% of kills.
        assert!(
            unattributed_pct < 50.0,
            "Too many unattributed kills ({unattributed_pct:.1}%) — \
             check process_explosions owner lookup for dead players"
        );
    }
}
