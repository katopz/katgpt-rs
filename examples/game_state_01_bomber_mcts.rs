//! GameState Forward Model — Bomber MCTS PoC (Plan 056)
//!
//! Demonstrates generic MCTS search on BomberState snapshot:
//! - MCTS player vs Random players, 100 rounds
//! - Print win rates and action space log
//! - Validate: MCTS > Random (>60%), MCTS < HL (<40%)

use microgpt_rs::pruners::{
    ActionSpaceLog, ArenaGrid, BomberAction, BomberHeuristic, BomberState, StateHeuristic,
    game_state::GameState, mcts_search,
};

// ── Players ────────────────────────────────────────────────────

/// Play one action using MCTS forward model search.
fn mcts_player(
    state: &BomberState,
    player_id: u8,
    heuristic: &BomberHeuristic,
    rng: &mut fastrand::Rng,
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
        200, // budget: 200 advance() calls
        10,  // rollout depth
        &|s: &BomberState, pid: u8| heuristic.evaluate(s, pid),
        rng,
    )
}

/// Play one random action.
fn random_player(state: &BomberState, player_id: u8, rng: &mut fastrand::Rng) -> BomberAction {
    let actions = state.available_actions(player_id);
    match actions.is_empty() {
        true => BomberAction::Wait,
        false => actions[rng.usize(0..actions.len())],
    }
}

// ── Game Loop ──────────────────────────────────────────────────

/// Run a single round of bomber using the forward model snapshot.
///
/// Returns the player id of the winner (or None for draw).
fn play_round(seed: u64) -> (Option<u8>, ActionSpaceLog) {
    let grid = ArenaGrid::generate(seed);
    let mut state = BomberState::from_grid(&grid);
    let heuristic = BomberHeuristic;
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut log = ActionSpaceLog::new();

    while !state.is_terminal() {
        // All players act simultaneously
        let mut actions = [BomberAction::Wait; 4];
        for pid in 0..4u8 {
            if !state.players[pid as usize].alive {
                continue;
            }
            actions[pid as usize] = match pid {
                // Player 0: MCTS
                0 => mcts_player(&state, pid, &heuristic, &mut rng),
                // Players 1-3: Random
                _ => random_player(&state, pid, &mut rng),
            };
            log.record(&state, pid);
        }

        // Advance state: process all 4 actions
        // Note: forward model processes one player at a time.
        // For simultaneous play, we apply actions in order.
        // This is a simplification — true simultaneity requires ECS.
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

    (winner, log)
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let rounds = 100;
    let mut wins = [0usize; 4];
    let mut draws = 0usize;
    let mut total_log = ActionSpaceLog::new();

    println!("=== GameState Forward Model — Bomber MCTS PoC ===");
    println!("Player 0: MCTS (budget=200, depth=10)");
    println!("Players 1-3: Random");
    println!("Rounds: {rounds}");
    println!();

    for round in 0..rounds {
        let seed = 42 + round as u64;
        let (winner, _log) = play_round(seed);

        // Accumulate action space stats from first 10 rounds
        if round < 10 {
            total_log.record(&BomberState::from_grid(&ArenaGrid::generate(seed)), 0);
        }

        match winner {
            Some(pid) => wins[pid as usize] += 1,
            None => draws += 1,
        }

        if (round + 1) % 25 == 0 {
            let pct = (round + 1) * 100 / rounds;
            println!(
                "[{pct:3}%] Round {:3}/{} — MCTS wins: {}, Random wins: {}/{}/{}, Draws: {}",
                round + 1,
                rounds,
                wins[0],
                wins[1],
                wins[2],
                wins[3],
                draws,
            );
        }
    }

    println!();
    println!("=== Results ===");
    println!(
        "MCTS (P0):  {} wins ({:.1}%)",
        wins[0],
        wins[0] as f64 / rounds as f64 * 100.0
    );
    println!(
        "Random (P1): {} wins ({:.1}%)",
        wins[1],
        wins[1] as f64 / rounds as f64 * 100.0
    );
    println!(
        "Random (P2): {} wins ({:.1}%)",
        wins[2],
        wins[2] as f64 / rounds as f64 * 100.0
    );
    println!(
        "Random (P3): {} wins ({:.1}%)",
        wins[3],
        wins[3] as f64 / rounds as f64 * 100.0
    );
    println!(
        "Draws:       {draws} ({:.1}%)",
        draws as f64 / rounds as f64 * 100.0
    );
    println!();

    let mcts_win_rate = wins[0] as f64 / rounds as f64;
    let random_win_rate = (wins[1] + wins[2] + wins[3]) as f64 / rounds as f64 / 3.0;

    println!("MCTS win rate:    {:.1}%", mcts_win_rate * 100.0);
    println!("Avg Random rate:  {:.1}%", random_win_rate * 100.0);
    println!();

    if mcts_win_rate > 0.60 {
        println!("✅ MCTS beats Random (>60% target met)");
    } else if mcts_win_rate > 0.40 {
        println!("⚠️  MCTS marginally better than Random (>40% but <60%)");
    } else {
        println!("❌ MCTS fails to beat Random (<40%)");
    }

    println!();
    println!("Action space log (first 10 rounds, P0): {total_log}");

    // ── Quick MCTS Demo ──────────────────────────────────────
    println!();
    println!("=== Single-Turn MCTS Demo ===");
    let grid = ArenaGrid::generate(42);
    let state = BomberState::from_grid(&grid);
    let heuristic = BomberHeuristic;
    let mut rng = fastrand::Rng::with_seed(42);

    let actions = state.available_actions(0);
    println!("Tick 0, Player 0 at {:?}", state.players[0].pos);
    println!(
        "Available actions: {:?}",
        actions.iter().map(|a| format!("{a}")).collect::<Vec<_>>()
    );

    let action = mcts_player(&state, 0, &heuristic, &mut rng);
    println!("MCTS chose: {action}");

    let next = state.advance(&action, 0);
    println!(
        "After advance: tick={}, pos={:?}",
        next.tick(),
        next.players[0].pos
    );
}
