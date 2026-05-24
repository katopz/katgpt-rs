//! Plan 065 Phase 0 T7: Play random games against AutoGo via API bridge.
//!
//! Demonstrates the [`AutoGoClient`] API and validates the head-to-head
//! benchmarking path. Requires a running AutoGo server (see `scripts/autogo_server.sh`).
//!
//! ```sh
//! # Start AutoGo server first
//! ./scripts/autogo_server.sh
//!
//! # Run this example
//! # Default: connects to localhost:8765 (override with GO_URL)
//! cargo run --features go --example go_00_api_bridge
//!
//! # Custom server
//! GO_URL=http://192.168.1.100:8765 cargo run --features go --example go_00_api_bridge
//! ```

use katgpt_rs::pruners::go::AutoGoClient;

/// Number of games to play.
const NUM_GAMES: usize = 5;

/// Default AutoGo server URL.
const DEFAULT_AUTOGO_URL: &str = "http://localhost:8765";

/// Board fill percentage at which we start passing every turn to force game end.
/// Random play without passes leads to infinite capture/recapture cycles.
/// At 70% fill on 9×9 (~51 stones), the remaining territory is clear enough.
const PASS_FILL_PCT: f64 = 0.70;

/// Game result tracked for summary.
#[derive(Debug)]
struct GameResult {
    game_id: String,
    our_color: String,
    result: Option<String>,
    moves_played: usize,
    #[allow(dead_code)]
    message: String,
}

fn main() {
    let autogo_url = std::env::var("GO_URL").unwrap_or_else(|_| DEFAULT_AUTOGO_URL.to_string());
    let client = AutoGoClient::new(&autogo_url);

    // Discover available agents
    let agents = match client.list_agents() {
        Ok(agents) => agents,
        Err(e) => {
            eprintln!("ERROR: Cannot reach AutoGo server at {autogo_url}");
            eprintln!("  {e}");
            eprintln!();
            eprintln!("Start the server first:");
            eprintln!("  cd .raw/autogo && uv run -m alpha_go.play --host 127.0.0.1 --port 8765");
            eprintln!();
            eprintln!("Or set GO_URL:");
            eprintln!(
                "  GO_URL=http://localhost:8765 cargo run --features go --example go_00_api_bridge"
            );
            std::process::exit(1);
        }
    };
    println!("Available agents: {agents:?}");

    if !agents.iter().any(|a| a == "random") {
        eprintln!("ERROR: 'random' agent not available on server");
        std::process::exit(1);
    }

    let mut results: Vec<GameResult> = Vec::with_capacity(NUM_GAMES);
    let colors = ["black", "white"];

    for i in 0..NUM_GAMES {
        let color = colors[i % 2];
        println!("\n═══ Game {}/{} as {color} ═══", i + 1, NUM_GAMES);

        let game = play_random_game(&client, color);
        results.push(game);
    }

    // Summary
    println!("\n{}", "═".repeat(50));
    println!("  SUMMARY: {NUM_GAMES} games against random agent");
    println!("{}", "═".repeat(50));

    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut draws = 0usize;

    for r in &results {
        let outcome = match &r.result {
            Some(score) => {
                let we_play_black = r.our_color == "black";
                let we_won = (we_play_black && score.starts_with('B'))
                    || (!we_play_black && score.starts_with('W'));
                if we_won {
                    wins += 1;
                    "WIN "
                } else if score.contains('+') {
                    losses += 1;
                    "LOSS"
                } else {
                    draws += 1;
                    "DRAW"
                }
            }
            None => {
                draws += 1;
                "????"
            }
        };
        println!(
            "  [{outcome}] {our_color:>5} vs random — {moves:>3} moves — {id}",
            outcome = outcome,
            our_color = r.our_color,
            moves = r.moves_played,
            id = r.game_id,
        );
        if let Some(ref result) = r.result {
            println!("         result: {result}");
        }
    }

    println!(
        "\n  W:{wins} L:{losses} D:{draws} (win rate: {:.0}%)",
        wins as f64 / NUM_GAMES as f64 * 100.0
    );
}

/// Play a full game using random legal moves, returning the result.
///
/// When the board is >70% full, we pass every turn to force game end via two
/// consecutive passes. Random play without passes leads to infinite capture/recapture cycles.
fn play_random_game(client: &AutoGoClient, color: &str) -> GameResult {
    let state = client
        .new_game(9, color, "random")
        .expect("Failed to start new game");
    let game_id = state.game_id.clone();
    let mut moves_played = 0usize;
    let total_cells = state.size * state.size;

    println!(
        "  Game {game_id} started, {} to play, {} legal moves",
        format_player(state.to_play),
        state.legal_moves.len()
    );

    let mut current = state;

    // Safety limit: 9×9 max ~500 moves (should never reach with fill-based pass logic)
    for _ in 0..500 {
        if current.is_over {
            break;
        }

        // Count stones on board — pass when mostly full to force game end.
        let stone_count: usize = current
            .board
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c != 0)
            .count();
        let fill_pct = stone_count as f64 / total_cells as f64;
        if fill_pct >= PASS_FILL_PCT {
            log::debug!(
                "  Late game pass at {moves_played} moves ({:.0}% full)",
                fill_pct * 100.0
            );
            current = client.pass_move(&game_id).unwrap_or_else(|e| {
                panic!("Pass failed for game {game_id}: {e}");
            });
            moves_played += 1;
            continue;
        }

        // Pick a random legal move
        match current.legal_moves.as_slice() {
            [] => {
                // No legal moves — must pass
                log::debug!("  No legal moves, passing");
                current = client.pass_move(&game_id).unwrap_or_else(|e| {
                    panic!("Pass failed for game {game_id}: {e}");
                });
            }
            moves => {
                let idx = fastrand::usize(..moves.len());
                let (row, col) = moves[idx];
                log::debug!("  Playing ({row},{col})");
                current = client.make_move(&game_id, row, col).unwrap_or_else(|e| {
                    panic!("Move failed for game {game_id} at ({row},{col}): {e}");
                });
            }
        }

        moves_played += 1;

        // Log every 20 moves
        if moves_played.is_multiple_of(20) {
            println!(
                "  ... {moves_played} moves, {} to play, {} legal",
                format_player(current.to_play),
                current.legal_moves.len()
            );
        }
    }

    println!(
        "  Game over! Result: {} ({})",
        current.result.as_deref().unwrap_or("?"),
        current.message
    );

    GameResult {
        game_id,
        our_color: color.to_string(),
        result: current.result,
        moves_played,
        message: current.message,
    }
}

/// Format player number as readable name.
fn format_player(player: i8) -> &'static str {
    match player {
        1 => "BLACK",
        2 => "WHITE",
        _ => "???",
    }
}
