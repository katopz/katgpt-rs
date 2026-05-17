//! Go Benchmark Suite — Plan 065 Phase 6
//!
//! Measures GoState performance, MCTS throughput, and player scaling laws.
//!
//! ```sh
//! # Full benchmark
//! cargo run --features go --example go_06_bench --release
//!
//! # With API benchmark (requires AutoGo server)
//! GO_API_URL=http://localhost:5000 cargo run --features go --example go_06_bench --release
//! ```

use std::env;
use std::io::Write;
use std::time::Instant;

use fastrand::Rng;
use microgpt_rs::pruners::game_state::{GameState, StateHeuristic, mcts_search};
use microgpt_rs::pruners::go::{
    GoAction, GoCell, GoGreedyPlayer, GoHLPlayer, GoHeuristic, GoMctsPlayer, GoPlayer,
    GoRandomPlayer, GoState, GoValidatorPlayer,
};

// ── Constants ──────────────────────────────────────────────────

const SEED: u64 = 42;
const MAX_MOVES: usize = 300;

const WARMUP_ADVANCE: usize = 100;
const BENCH_ADVANCE: usize = 10_000;

const BENCH_MCTS: usize = 5;

const TOURNAMENT_GAMES: usize = 20;

// ════════════════════════════════════════════════════════════════
//  T43: GoState::advance() Benchmark
// ════════════════════════════════════════════════════════════════

/// Play N random moves to reach a mid-game position.
fn random_position(size: usize, target_moves: usize, rng: &mut Rng) -> GoState {
    let mut state = GoState::new(size);
    for _ in 0..target_moves {
        if state.is_terminal() {
            break;
        }
        let legal = state.legal_moves();
        if legal.is_empty() || rng.f32() < 0.02 {
            state.play_pass();
        } else {
            let (r, c) = legal[rng.usize(..legal.len())];
            state.play_move(r, c);
        }
    }
    state
}

fn bench_go_state() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  T43: GoState::advance() Performance");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut rng = Rng::with_seed(SEED);

    let configs: &[(usize, usize, &str)] = &[
        (9, 0, "9×9 opening"),
        (9, 30, "9×9 midgame (~30 moves)"),
        (9, 80, "9×9 endgame (~80 moves)"),
        (19, 0, "19×19 opening"),
        (19, 50, "19×19 midgame (~50 moves)"),
        (19, 200, "19×19 endgame (~200 moves)"),
    ];

    println!(
        "  {:<28} {:>10} {:>12} {:>10} {:>10}",
        "Config", "Legal", "ops/sec", "µs/adv", "µs/clone"
    );
    println!("  ──────────────────────────  ──────────  ────────────  ──────────  ──────────");

    for &(size, target_moves, label) in configs {
        let state = random_position(size, target_moves, &mut rng);
        let player_id = state.to_play.player_id();
        let actions = state.available_actions(player_id);

        if actions.is_empty() {
            println!(
                "  {:<28} {:>10} {:>12} {:>10} {:>10}",
                label, 0, "—", "—", "—"
            );
            continue;
        }

        // Warmup
        for _ in 0..WARMUP_ADVANCE {
            for action in &actions {
                let _ = state.advance(action, player_id);
            }
        }

        // Bench advance
        let start = Instant::now();
        let mut total_advances = 0usize;
        for _ in 0..BENCH_ADVANCE {
            for action in &actions {
                let _ = state.advance(action, player_id);
                total_advances += 1;
            }
        }
        let elapsed = start.elapsed();
        let ops_sec = total_advances as f64 / elapsed.as_secs_f64();
        let us_per_advance = elapsed.as_micros() as f64 / total_advances as f64;

        // Bench clone
        let clone_start = Instant::now();
        for _ in 0..BENCH_ADVANCE {
            let _ = state.clone();
        }
        let clone_elapsed = clone_start.elapsed();
        let us_per_clone = clone_elapsed.as_micros() as f64 / BENCH_ADVANCE as f64;

        println!(
            "  {:<28} {:>10} {:>12.0} {:>10.2} {:>10.2}",
            label,
            actions.len(),
            ops_sec,
            us_per_advance,
            us_per_clone
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  T44: MCTS Search Benchmark
// ════════════════════════════════════════════════════════════════

fn bench_go_mcts() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  T44: MCTS Search Throughput (9×9, ~10 moves played)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut rng = Rng::with_seed(SEED);
    let heuristic = GoHeuristic;
    let heuristic_fn = |s: &GoState, pid: u8| heuristic.evaluate(s, pid);

    let budgets: &[usize] = &[50, 200, 500, 1000];

    println!(
        "  {:>8} {:>12} {:>12} {:>14}",
        "Budget", "µs/search", "actions/sec", "nodes/sec"
    );
    println!("  ────────  ────────────  ────────────  ──────────────");

    for &budget in budgets {
        // Create a position with ~10 random moves
        let state = random_position(9, 10, &mut rng);
        let player_id = state.to_play.player_id();

        // Warmup
        let _ = mcts_search(&state, player_id, budget, 50, &heuristic_fn, &mut rng);

        // Bench
        let start = Instant::now();
        for _ in 0..BENCH_MCTS {
            let _ = mcts_search(&state, player_id, budget, 50, &heuristic_fn, &mut rng);
        }
        let elapsed = start.elapsed();

        let us_per_search = elapsed.as_micros() as f64 / BENCH_MCTS as f64;
        let searches_per_sec = BENCH_MCTS as f64 / elapsed.as_secs_f64();
        // Each search returns 1 action
        let actions_per_sec = searches_per_sec;
        // Each budget unit ≈ 1 node expansion (advance + rollout)
        let nodes_per_sec = budget as f64 * searches_per_sec;

        println!(
            "  {:>8} {:>12.1} {:>12.0} {:>14.0}",
            budget, us_per_search, actions_per_sec, nodes_per_sec
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  T46: Player Scaling Law Data (Tournament)
// ════════════════════════════════════════════════════════════════

/// Play a game between two players, return (winner, total_moves).
fn play_game(
    black: &mut dyn GoPlayer,
    white: &mut dyn GoPlayer,
    state: &mut GoState,
    rng: &mut Rng,
) -> (Option<GoCell>, usize) {
    let mut moves = 0;
    while !state.is_terminal() && moves < MAX_MOVES {
        let legal = state.legal_moves();
        let action = if state.to_play == GoCell::Black {
            black.select_move(state, &legal, rng)
        } else {
            white.select_move(state, &legal, rng)
        };
        match &action {
            GoAction::Place(r, c) => {
                state.play_move(*r, *c);
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }
        moves += 1;
    }

    // Force end if not terminal
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
        moves += 2;
    }

    (state.get_winner(), moves)
}

/// Update bandit-based players (HL) after a game.
fn update_player_outcome(player: &mut dyn GoPlayer, won: bool) {
    if let Some(hl) = player.as_any_mut().downcast_mut::<GoHLPlayer>() {
        hl.update_outcome(won);
    }
}

/// Create a player by name for tournament.
fn make_player(name: &str) -> Box<dyn GoPlayer> {
    match name {
        "random" => Box::new(GoRandomPlayer),
        "greedy" => Box::new(GoGreedyPlayer),
        "validator" => Box::new(GoValidatorPlayer),
        "hl" => Box::new(GoHLPlayer::new()),
        "mcts200" => Box::new(GoMctsPlayer::new(200, 50)),
        _ => panic!("Unknown player: {name}"),
    }
}

fn bench_player_scaling() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  T46: Player Scaling Laws (9×9, {TOURNAMENT_GAMES} games each)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut rng = Rng::with_seed(SEED);
    let board_size = 9;

    let matchups: &[(&str, &str)] = &[
        ("random", "random"),
        ("greedy", "random"),
        ("validator", "random"),
        ("hl", "random"),
        ("mcts200", "random"),
    ];

    // CSV header
    println!("  Player,Opponent,Wins,Losses,Draws,WinRate%");
    println!("  ──────────────────────────────────────────────");

    let mut all_results: Vec<(String, String, usize, usize, usize, f64)> = Vec::new();

    for &(player_name, opponent_name) in matchups {
        let mut player = make_player(player_name);
        let mut opponent = make_player(opponent_name);

        let mut wins = 0usize;
        let mut losses = 0usize;
        let mut draws = 0usize;

        for game_idx in 0..TOURNAMENT_GAMES {
            // Alternate colors
            let (_black_name, _white_name) = if game_idx % 2 == 0 {
                (player_name, opponent_name)
            } else {
                (opponent_name, player_name)
            };

            let mut state = GoState::new(board_size);

            let winner = if game_idx % 2 == 0 {
                let (w, _moves) =
                    play_game(player.as_mut(), opponent.as_mut(), &mut state, &mut rng);
                w
            } else {
                let (w, _moves) =
                    play_game(opponent.as_mut(), player.as_mut(), &mut state, &mut rng);
                w
            };

            let player_color = if game_idx % 2 == 0 {
                GoCell::Black
            } else {
                GoCell::White
            };

            match winner {
                Some(c) if c == player_color => {
                    wins += 1;
                    update_player_outcome(player.as_mut(), true);
                    update_player_outcome(opponent.as_mut(), false);
                }
                Some(_) => {
                    losses += 1;
                    update_player_outcome(player.as_mut(), false);
                    update_player_outcome(opponent.as_mut(), true);
                }
                None => {
                    draws += 1;
                }
            }

            // Progress
            print!(".");
            let _ = std::io::stdout().flush();
        }

        let total = wins + losses + draws;
        let win_rate = wins as f64 / total as f64 * 100.0;

        println!(
            "  {},{},{},{},{},{:.1}",
            player_name, opponent_name, wins, losses, draws, win_rate
        );

        all_results.push((
            player_name.to_string(),
            opponent_name.to_string(),
            wins,
            losses,
            draws,
            win_rate,
        ));

        player.reset();
        opponent.reset();
    }

    // Summary table
    println!();
    println!("─── Summary ───────────────────────────────────────────────────");
    println!(
        "  {:<14} {:>5} {:>5} {:>5} {:>8}",
        "Player", "Wins", "Loss", "Draw", "WinRate%"
    );
    println!("  ────────────── ───── ───── ───── ────────");
    for (player, _opp, wins, losses, draws, wr) in &all_results {
        println!(
            "  {:<14} {:>5} {:>5} {:>5} {:>7.1}%",
            player, wins, losses, draws, wr
        );
    }
    println!();
}

// ════════════════════════════════════════════════════════════════
//  T45 (Optional): API Benchmark
// ════════════════════════════════════════════════════════════════

fn bench_go_api() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  T45: API Benchmark (AutoGo Server)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let api_url = match env::var("GO_API_URL") {
        Ok(url) => url,
        Err(_) => {
            println!("  ⏭  Skipped (set GO_API_URL to enable)");
            println!("     e.g. GO_API_URL=http://localhost:5000");
            println!();
            return;
        }
    };

    use microgpt_rs::pruners::go::AutoGoClient;

    let client = AutoGoClient::new(&api_url);

    let mut rng = Rng::with_seed(SEED);
    let num_games = 5;
    let mut total_moves = 0usize;

    println!("  Server: {api_url}");
    println!("  Games:  {num_games}");
    println!();

    let start = Instant::now();

    for i in 0..num_games {
        let color = if i % 2 == 0 { "black" } else { "white" };
        print!("  [{}/{}] {} ", i + 1, num_games, color);
        let _ = std::io::stdout().flush();

        // Start a new game vs server random agent
        let game_state = match client.new_game(9, color, "random") {
            Ok(gs) => gs,
            Err(e) => {
                println!("ERROR: {e}");
                println!("  Ensure AutoGo server is running at {api_url}");
                println!();
                return;
            }
        };

        let game_id = game_state.game_id.clone();
        let mut current = game_state;
        let mut moves = 0usize;

        for _ in 0..MAX_MOVES {
            if current.is_over {
                break;
            }

            // Use legal_moves from API response directly
            if current.legal_moves.is_empty() {
                match client.pass_move(&game_id) {
                    Ok(next) => current = next,
                    Err(_) => break,
                }
            } else {
                let (r, c) = current.legal_moves[rng.usize(..current.legal_moves.len())];
                match client.make_move(&game_id, r, c) {
                    Ok(next) => current = next,
                    Err(_) => break,
                }
            }

            moves += 1;
        }

        let result_str = current.result.as_deref().unwrap_or("?");
        println!("{moves:>3} moves  {result_str}");
        total_moves += moves;
    }

    let elapsed = start.elapsed();
    let games_sec = num_games as f64 / elapsed.as_secs_f64();

    println!();
    println!("─── API Results ───────────────────────────────────────────────");
    println!("  Total time:    {:.2}s", elapsed.as_secs_f64());
    println!("  Games/sec:     {games_sec:.2}");
    println!(
        "  Avg moves:     {:.1}",
        total_moves as f64 / num_games as f64
    );
    println!();
}

// ════════════════════════════════════════════════════════════════
//  Main
// ════════════════════════════════════════════════════════════════

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Go Benchmark Suite — Plan 065 Phase 6");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    bench_go_state();
    bench_go_mcts();
    bench_player_scaling();
    bench_go_api();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmark Complete");
    println!("═══════════════════════════════════════════════════════════════");
}
