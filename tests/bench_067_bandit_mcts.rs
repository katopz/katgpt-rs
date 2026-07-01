//! Plan 067: NFSP/MCTS Duality — Bandit-guided MCTS Benchmark
//!
//! Run with: cargo test --features bandit_mcts --test bench_067_bandit_mcts -- --nocapture
//!
//! Tournament: BanditMCTS (P0) vs MCTS (P1) vs Random (P2) vs Random (P3)
//! Hypothesis: BanditMCTS > MCTS ≈ Random (bandit Q-values inform forward search)

#[cfg(feature = "bandit_mcts")]
use std::time::Instant;

#[cfg(feature = "bandit_mcts")]
use fastrand::Rng;

#[cfg(feature = "bandit_mcts")]
use katgpt_rs::pruners::{
    ArenaGrid, BanditBomberHeuristic, BanditStats, BomberAction, BomberHeuristic, BomberState,
    StateHeuristic,
    game_state::{
        BanditRolloutPolicy, GameState, RandomRolloutPolicy, mcts_search, mcts_search_informed,
    },
};

// ── Player Functions ───────────────────────────────────────────

#[cfg(feature = "bandit_mcts")]
fn bandit_mcts_player(
    state: &BomberState,
    player_id: u8,
    heuristic: &BanditBomberHeuristic,
    rng: &mut Rng,
) -> BomberAction {
    let actions = state.available_actions(player_id);
    if actions.is_empty() {
        return BomberAction::Wait;
    }
    if actions.len() == 1 {
        return actions[0];
    }

    let mut policy = BanditRolloutPolicy::new(
        heuristic.stats(),
        0.2, // ε = 20% explore
        |a: &BomberAction| a.as_usize(),
    );

    mcts_search_informed(state, player_id, 200, 10, heuristic, &mut policy, rng)
}

#[cfg(feature = "bandit_mcts")]
fn mcts_player(
    state: &BomberState,
    player_id: u8,
    heuristic: &BomberHeuristic,
    rng: &mut Rng,
) -> BomberAction {
    let actions = state.available_actions(player_id);
    if actions.is_empty() {
        return BomberAction::Wait;
    }
    if actions.len() == 1 {
        return actions[0];
    }

    mcts_search(
        state,
        player_id,
        200, // budget: same as BanditMCTS
        10,
        &|s: &BomberState, pid: u8| heuristic.evaluate(s, pid),
        rng,
    )
}

#[cfg(feature = "bandit_mcts")]
fn random_player(state: &BomberState, player_id: u8, rng: &mut Rng) -> BomberAction {
    let actions = state.available_actions(player_id);
    match actions.is_empty() {
        true => BomberAction::Wait,
        false => actions[rng.usize(0..actions.len())],
    }
}

// ── Game Loop ──────────────────────────────────────────────────

#[cfg(feature = "bandit_mcts")]
fn play_round(seed: u64, heuristic: &mut BanditBomberHeuristic) -> Option<u8> {
    let grid = ArenaGrid::generate(seed);
    let mut state = BomberState::from_grid(&grid);
    let domain_heuristic = BomberHeuristic;
    let mut rng = Rng::with_seed(seed);

    while !state.is_terminal() {
        let mut actions = [BomberAction::Wait; 4];
        for pid in 0..4u8 {
            if !state.players[pid as usize].alive {
                continue;
            }
            actions[pid as usize] = match pid {
                0 => bandit_mcts_player(&state, pid, heuristic, &mut rng),
                1 => mcts_player(&state, pid, &domain_heuristic, &mut rng),
                _ => random_player(&state, pid, &mut rng),
            };
        }

        for pid in 0..4u8 {
            if state.players[pid as usize].alive {
                state = state.advance(&actions[pid as usize], pid);
            }
            if state.is_terminal() {
                break;
            }
        }
    }

    // Determine winner
    let winner = state
        .players
        .iter()
        .enumerate()
        .find(|(_, p)| p.alive)
        .map(|(i, _)| i as u8);

    // Reward the bandit for P0's actions (update via heuristic)
    if winner == Some(0) {
        for arm in 0..7 {
            heuristic.update(arm, 1.0);
        }
    } else {
        for arm in 0..7 {
            heuristic.update(arm, 0.0);
        }
    }

    winner
}

#[cfg(feature = "bandit_mcts")]
fn play_round_no_bandit(seed: u64) -> Option<u8> {
    let grid = ArenaGrid::generate(seed);
    let mut state = BomberState::from_grid(&grid);
    let heuristic = BomberHeuristic;
    let mut rng = Rng::with_seed(seed);

    while !state.is_terminal() {
        let mut actions = [BomberAction::Wait; 4];
        for pid in 0..4u8 {
            if !state.players[pid as usize].alive {
                continue;
            }
            actions[pid as usize] = match pid {
                1 => mcts_player(&state, pid, &heuristic, &mut rng),
                _ => random_player(&state, pid, &mut rng),
            };
        }

        for pid in 0..4u8 {
            if state.players[pid as usize].alive {
                state = state.advance(&actions[pid as usize], pid);
            }
            if state.is_terminal() {
                break;
            }
        }
    }

    state
        .players
        .iter()
        .enumerate()
        .find(|(_, p)| p.alive)
        .map(|(i, _)| i as u8)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(feature = "bandit_mcts")]
#[test]
fn bench_067_bandit_mcts_tournament() {
    let rounds = 100usize;
    let mut wins = [0usize; 4];
    let mut draws = 0usize;
    let bandit_stats = BanditStats::new(7); // 7 BomberAction variants
    let mut bandit_heuristic = BanditBomberHeuristic::new(bandit_stats, 1.0);

    let start = Instant::now();

    println!("\n🧪 Plan 067: NFSP/MCTS Duality — Tournament ({rounds} rounds)");
    println!("{}", "═".repeat(70));
    println!("P0: BanditMCTS (budget=200, depth=10, ε=0.2, λ=1.0)");
    println!("P1: MCTS (budget=200, depth=10, random rollouts)");
    println!("P2: Random");
    println!("P3: Random");
    println!("{}", "─".repeat(70));

    for round in 0..rounds {
        let seed = 42 + round as u64;
        let winner = play_round(seed, &mut bandit_heuristic);

        match winner {
            Some(pid) => wins[pid as usize] += 1,
            None => draws += 1,
        }

        if (round + 1) % 25 == 0 {
            let pct = (round + 1) * 100 / rounds;
            println!(
                "[{pct:3}%] Round {:3}/{rounds} — BanditMCTS: {}, MCTS: {}, Random: {}/{}, Draws: {}",
                round + 1,
                wins[0],
                wins[1],
                wins[2],
                wins[3],
                draws,
            );
        }
    }

    let elapsed = start.elapsed();

    println!("{}", "─".repeat(70));
    println!("Time: {elapsed:.2?}");
    println!();
    println!("┌──────────────────┬──────┬──────────┐");
    println!("│ Player           │ Wins │ Win Rate │");
    println!("├──────────────────┼──────┼──────────┤");
    for (pid, name) in ["BanditMCTS", "MCTS", "Random", "Random"]
        .iter()
        .enumerate()
    {
        let rate = wins[pid] as f64 / rounds as f64 * 100.0;
        println!("│ {name:<16} │ {:>4} │ {:>6.1}%  │", wins[pid], rate);
    }
    println!(
        "│ {:<16} │ {:>4} │ {:>6.1}%  │",
        "Draws",
        draws,
        draws as f64 / rounds as f64 * 100.0
    );
    println!("└──────────────────┴──────┴──────────┘");

    // Bandit Q-values after tournament
    println!();
    println!("Bandit Q-values after {rounds} episodes:");
    let action_names = ["Up", "Down", "Left", "Right", "Bomb", "Wait", "Detonate"];
    let stats = bandit_heuristic.stats();
    for (i, name) in action_names.iter().enumerate() {
        let q = stats.q_value(i);
        let v = stats.visit_count(i);
        println!("  {name:<10}: Q={q:.3}, visits={v}");
    }

    let bandit_rate = wins[0] as f64 / rounds as f64;
    let mcts_rate = wins[1] as f64 / rounds as f64;
    let avg_random = (wins[2] + wins[3]) as f64 / 2.0 / rounds as f64;

    println!();
    println!("BanditMCTS win rate:    {:.1}%", bandit_rate * 100.0);
    println!("MCTS win rate:          {:.1}%", mcts_rate * 100.0);
    println!("Avg Random win rate:    {:.1}%", avg_random * 100.0);
    println!(
        "Δ BanditMCTS vs MCTS:   {:+.1}pp",
        (bandit_rate - mcts_rate) * 100.0
    );
    println!();

    // Quality gate: BanditMCTS should beat MCTS (any positive improvement)
    if bandit_rate > mcts_rate {
        println!(
            "✅ BanditMCTS beats MCTS ({:.1}% > {:.1}%)",
            bandit_rate * 100.0,
            mcts_rate * 100.0
        );
    } else {
        println!(
            "⚠️  BanditMCTS does NOT beat MCTS ({:.1}% <= {:.1}%)",
            bandit_rate * 100.0,
            mcts_rate * 100.0
        );
    }

    // Baseline: both should beat random
    if bandit_rate > avg_random {
        println!(
            "✅ BanditMCTS beats Random ({:.1}% > {:.1}%)",
            bandit_rate * 100.0,
            avg_random * 100.0
        );
    }
    if mcts_rate > avg_random {
        println!(
            "✅ MCTS beats Random ({:.1}% > {:.1}%)",
            mcts_rate * 100.0,
            avg_random * 100.0
        );
    }
}

#[cfg(feature = "bandit_mcts")]
#[test]
fn bench_067_mcts_baseline() {
    let rounds = 100usize;
    let mut wins = [0usize; 4];
    let mut draws = 0usize;

    let start = Instant::now();

    println!("\n🧪 Plan 067: MCTS Baseline — MCTS (P1) vs 3× Random ({rounds} rounds)");
    println!("{}", "═".repeat(70));

    for round in 0..rounds {
        let seed = round as u64;
        let winner = play_round_no_bandit(seed);

        match winner {
            Some(pid) => wins[pid as usize] += 1,
            None => draws += 1,
        }
    }

    let elapsed = start.elapsed();

    println!("Time: {elapsed:.2?}");
    println!();
    println!("┌──────────────────┬──────┬──────────┐");
    println!("│ Player           │ Wins │ Win Rate │");
    println!("├──────────────────┼──────┼──────────┤");
    let names = ["Random (P0)", "MCTS (P1)", "Random (P2)", "Random (P3)"];
    for (pid, name) in names.iter().enumerate() {
        let rate = wins[pid] as f64 / rounds as f64 * 100.0;
        println!("│ {name:<16} │ {:>4} │ {:>6.1}%  │", wins[pid], rate);
    }
    println!(
        "│ {:<16} │ {:>4} │ {:>6.1}%  │",
        "Draws",
        draws,
        draws as f64 / rounds as f64 * 100.0
    );
    println!("└──────────────────┴──────┴──────────┘");

    let mcts_rate = wins[1] as f64 / rounds as f64;
    let avg_random = (wins[0] + wins[2] + wins[3]) as f64 / 3.0 / rounds as f64;
    println!();
    println!("MCTS win rate:       {:.1}%", mcts_rate * 100.0);
    println!("Avg Random win rate: {:.1}%", avg_random * 100.0);
    println!();

    // Plan 056 finding: MCTS ≈ Random in Bomberman
    if mcts_rate > avg_random + 0.05 {
        println!("⚠️  MCTS significantly beats Random (contradicts Plan 056 finding)");
    } else {
        println!("📊 MCTS ≈ Random (confirms Plan 056: generic MCTS has no backward signal)");
    }
}

#[cfg(feature = "bandit_mcts")]
#[test]
fn bench_067_rollout_policy_micro() {
    use katgpt_rs::pruners::game_state::RolloutPolicy;

    let state = BomberState::from_grid(&ArenaGrid::generate(42));
    let actions = state.available_actions(0);

    if actions.is_empty() {
        println!("⚠️  No actions available, skipping micro benchmark");
        return;
    }

    let iters = 10_000usize;
    let mut rng = Rng::with_seed(42);

    // Random rollout
    let start = Instant::now();
    let mut random_policy = RandomRolloutPolicy;
    for _ in 0..iters {
        std::hint::black_box(random_policy.select(&state, &actions, 0, &mut rng));
    }
    let random_time = start.elapsed();

    // Bandit rollout (cold stats)
    let cold_stats = BanditStats::new(7);
    let mut cold_policy =
        BanditRolloutPolicy::new(&cold_stats, 0.2, |a: &BomberAction| a.as_usize());
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(cold_policy.select(&state, &actions, 0, &mut rng));
    }
    let cold_bandit_time = start.elapsed();

    // Bandit rollout (warm stats)
    let mut warm_stats = BanditStats::new(7);
    for arm in 0..7 {
        warm_stats.update(arm, 0.5);
        warm_stats.update(arm, 1.0);
    }
    let mut warm_policy =
        BanditRolloutPolicy::new(&warm_stats, 0.2, |a: &BomberAction| a.as_usize());
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(warm_policy.select(&state, &actions, 0, &mut rng));
    }
    let warm_bandit_time = start.elapsed();

    println!("\n🧪 Plan 067: Rollout Policy Micro Benchmark ({iters} iterations)");
    println!("{}", "═".repeat(60));
    println!("┌────────────────────────┬───────────┐");
    println!("│ Policy                 │ μs/call   │");
    println!("├────────────────────────┼───────────┤");
    println!(
        "│ RandomRolloutPolicy    │ {:>7.2}   │",
        random_time.as_secs_f64() / iters as f64 * 1e6
    );
    println!(
        "│ BanditRollout (cold)   │ {:>7.2}   │",
        cold_bandit_time.as_secs_f64() / iters as f64 * 1e6
    );
    println!(
        "│ BanditRollout (warm)   │ {:>7.2}   │",
        warm_bandit_time.as_secs_f64() / iters as f64 * 1e6
    );
    println!("└────────────────────────┴───────────┘");

    // Bandit should not be more than 10× slower than random
    let cold_overhead = cold_bandit_time.as_secs_f64() / random_time.as_secs_f64();
    println!();
    println!("Bandit overhead (cold): {cold_overhead:.1}×");
    assert!(
        cold_overhead < 10.0,
        "Bandit rollout overhead too high: {cold_overhead:.1}× (limit: 10×)"
    );
}

// ── No-op when feature is disabled ─────────────────────────────

#[cfg(not(feature = "bandit_mcts"))]
#[test]
fn bench_067_requires_bandit_mcts_feature() {
    println!(
        "⚠️  Run with: cargo test --features bandit_mcts --test bench_067_bandit_mcts -- --nocapture"
    );
}
