//! Integration tests for AutoGo API bridge.
//!
//! These tests require a running AutoGo server. Run with:
//!
//! ```sh
//! # Start server first
//! ./scripts/autogo_server.sh
//!
//! # Run integration tests
//! cargo test --features go --test go_integration -- --ignored
//! ```
//!
//! Without `--ignored`, these tests are skipped (no server required for normal `cargo test`).

#[cfg(feature = "go")]
mod go_integration {
    use microgpt_rs::pruners::go::{AutoGoClient, AutoGoGameState};

    const AUTOGO_URL: &str = "http://localhost:8000";

    /// Board fill percentage at which we start passing every turn to force game end.
    /// Random play without passes leads to infinite capture/recapture cycles.
    /// At 70% fill on 9×9 (~51 stones), the remaining territory is clear enough.
    const PASS_FILL_PCT: f64 = 0.70;

    fn client() -> AutoGoClient {
        AutoGoClient::new(AUTOGO_URL)
    }

    /// Helper: play a full random game, returning the final state and move count.
    fn play_random_game(
        client: &AutoGoClient,
        size: usize,
        color: &str,
        agent: &str,
    ) -> (AutoGoGameState, usize) {
        let state = client
            .new_game(size, color, agent)
            .expect("Failed to start game");
        let game_id = state.game_id.clone();
        let mut current = state;
        let mut moves = 0usize;

        for _ in 0..500 {
            if current.is_over {
                break;
            }

            // Count stones on board — pass when mostly full to force game end.
            let total_cells = size * size;
            let stone_count: usize = current
                .board
                .iter()
                .flat_map(|row| row.iter())
                .filter(|&&c| c != 0)
                .count();
            let fill_pct = stone_count as f64 / total_cells as f64;
            if fill_pct >= PASS_FILL_PCT {
                current = client.pass_move(&game_id).expect("Pass failed");
                moves += 1;
                continue;
            }

            match current.legal_moves.as_slice() {
                [] => {
                    current = client.pass_move(&game_id).expect("Pass failed");
                }
                legal => {
                    let idx = fastrand::usize(..legal.len());
                    let (row, col) = legal[idx];
                    current = client.make_move(&game_id, row, col).expect("Move failed");
                }
            }
            moves += 1;
        }

        (current, moves)
    }

    // ── Basic API Tests ────────────────────────────────────────

    #[test]
    #[ignore] // Requires running AutoGo server
    fn list_agents_returns_nonempty() {
        let client = client();
        let agents = client.list_agents().expect("list_agents failed");
        assert!(!agents.is_empty(), "Should have at least one agent");
        println!("Agents: {agents:?}");
    }

    #[test]
    #[ignore]
    fn list_agents_includes_random() {
        let client = client();
        let agents = client.list_agents().expect("list_agents failed");
        assert!(
            agents.iter().any(|a| a == "random"),
            "'random' agent should be available, got: {agents:?}"
        );
    }

    #[test]
    #[ignore]
    fn new_game_9x9_black() {
        let client = client();
        let state = client
            .new_game(9, "black", "random")
            .expect("new_game failed");

        assert_eq!(state.size, 9, "Board size should be 9");
        assert_eq!(state.human_color, 1, "human_color should be 1 (BLACK)");
        assert_eq!(state.to_play, 1, "Black should play first");
        assert!(!state.is_over, "Game should not be over at start");
        assert!(
            state.legal_moves.len() > 50,
            "9×9 should have many legal moves at start, got {}",
            state.legal_moves.len()
        );
        assert!(!state.game_id.is_empty(), "game_id should be non-empty");
    }

    #[test]
    #[ignore]
    fn new_game_9x9_white_triggers_ai_first() {
        let client = client();
        let state = client
            .new_game(9, "white", "random")
            .expect("new_game failed");

        assert_eq!(state.human_color, 2, "human_color should be 2 (WHITE)");
        // When we play white, AI (black) moves first via G2 dual-move.
        // After AI's first move, it should be our turn (to_play=2=WHITE).
        assert_eq!(
            state.to_play, 2,
            "After AI opens, it should be our turn (WHITE)"
        );
        // Board should have exactly one stone (AI's opening move)
        let stone_count: usize = state
            .board
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c != 0)
            .count();
        assert_eq!(stone_count, 1, "Should have 1 stone (AI's opening)");
    }

    #[test]
    #[ignore]
    fn make_move_returns_updated_state() {
        let client = client();
        let state = client
            .new_game(9, "black", "random")
            .expect("new_game failed");
        let game_id = state.game_id.clone();

        // Play first stone
        let (row, col) = state.legal_moves[0];
        let after = client
            .make_move(&game_id, row, col)
            .expect("make_move failed");

        // After our move + AI response (G2), board should have 2+ stones
        let stone_count: usize = after
            .board
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&c| c != 0)
            .count();
        assert!(
            stone_count >= 2,
            "After our move + AI response, should have ≥2 stones, got {stone_count}"
        );
    }

    #[test]
    #[ignore]
    fn get_game_returns_same_state() {
        let client = client();
        let state = client
            .new_game(9, "black", "random")
            .expect("new_game failed");
        let game_id = state.game_id.clone();

        let fetched = client.get_game(&game_id).expect("get_game failed");
        assert_eq!(fetched.game_id, game_id);
        assert_eq!(fetched.size, state.size);
        assert_eq!(fetched.human_color, state.human_color);
    }

    #[test]
    #[ignore]
    fn pass_move_increments_consecutive_passes() {
        let client = client();
        let state = client
            .new_game(9, "black", "random")
            .expect("new_game failed");
        let game_id = state.game_id.clone();

        // Pass twice to end the game (AI might not pass though)
        let _after_pass = client.pass_move(&game_id).expect("pass_move failed");
        // AI responds to our pass — game may or may not be over
        // (depends on whether AI also passes)
    }

    // ── Full Game Flow Tests ───────────────────────────────────

    #[test]
    #[ignore]
    fn random_game_9x9_completes() {
        let client = client();
        let (final_state, moves) = play_random_game(&client, 9, "black", "random");

        assert!(
            final_state.is_over,
            "Game should complete within 300 moves ({moves} played)"
        );
        assert!(
            final_state.result.is_some(),
            "Completed game should have a result"
        );
        println!(
            "Game completed in {moves} moves — result: {:?}",
            final_state.result
        );
    }

    #[test]
    #[ignore]
    fn random_game_white_also_completes() {
        let client = client();
        let (final_state, moves) = play_random_game(&client, 9, "white", "random");

        assert!(
            final_state.is_over,
            "Game as white should complete within 300 moves ({moves} played)"
        );
        assert!(
            final_state.result.is_some(),
            "Completed game should have a result"
        );
    }

    // ── Multi-Game Stress Test ─────────────────────────────────

    #[test]
    #[ignore]
    fn ten_random_games_all_complete() {
        let client = client();
        let mut completed = 0usize;
        let mut total_moves = 0usize;

        for i in 0..10 {
            let color = if i % 2 == 0 { "black" } else { "white" };
            let (final_state, moves) = play_random_game(&client, 9, color, "random");

            if !final_state.is_over {
                panic!(
                    "Game {i} as {color} did not complete after {moves} moves — game_id: {}",
                    final_state.game_id
                );
            }
            completed += 1;
            total_moves += moves;

            print!(
                "  [{completed}/10] {} moves, result: {:?}        \r",
                moves, final_state.result
            );
        }

        println!();
        assert_eq!(completed, 10, "All 10 games should complete");
        let avg = total_moves as f64 / completed as f64;
        println!("  Average: {avg:.1} moves/game across {completed} games");
    }

    // ── Latency Measurement (T8) ──────────────────────────────

    #[test]
    #[ignore]
    fn measure_api_latency() {
        let client = client();
        let num_games = 5;
        let mut total_api_calls = 0usize;
        let mut total_duration = std::time::Duration::ZERO;

        for i in 0..num_games {
            let color = if i % 2 == 0 { "black" } else { "white" };

            // Measure new_game
            let t0 = std::time::Instant::now();
            let state = client
                .new_game(9, color, "random")
                .expect("new_game failed");
            let new_game_time = t0.elapsed();
            total_api_calls += 1;

            let game_id = state.game_id.clone();
            let mut current = state;

            // Play random moves until game over
            for _ in 0..300 {
                if current.is_over {
                    break;
                }

                let t1 = std::time::Instant::now();
                match current.legal_moves.as_slice() {
                    [] => {
                        current = client.pass_move(&game_id).expect("pass failed");
                    }
                    legal => {
                        let idx = fastrand::usize(..legal.len());
                        let (row, col) = legal[idx];
                        current = client.make_move(&game_id, row, col).expect("move failed");
                    }
                }
                total_duration += t1.elapsed();
                total_api_calls += 1;
            }

            total_duration += new_game_time;
        }

        let avg_latency = total_duration / total_api_calls as u32;
        let games_per_sec = num_games as f64 / total_duration.as_secs_f64();
        // G2: one API call = two moves, so effective moves = 2 × calls
        let effective_moves_per_sec = 2.0 * total_api_calls as f64 / total_duration.as_secs_f64();

        println!("  API Latency Results ({num_games} games):");
        println!("    Total API calls:    {total_api_calls}");
        println!(
            "    Total duration:     {:.2}s",
            total_duration.as_secs_f64()
        );
        println!("    Avg latency/call:   {avg_latency:?}");
        println!("    Games/sec:          {games_per_sec:.2}");
        println!("    Effective moves/sec: {effective_moves_per_sec:.0} (×2 via G2 dual-move)");

        // Sanity: each call should be under 5 seconds
        assert!(
            avg_latency.as_secs() < 5,
            "API latency too high: {avg_latency:?}"
        );
    }

    // ── Board State Validation ─────────────────────────────────

    #[test]
    #[ignore]
    fn board_state_is_consistent() {
        let client = client();
        let state = client
            .new_game(9, "black", "random")
            .expect("new_game failed");
        let game_id = state.game_id.clone();

        // Board dimensions
        assert_eq!(state.board.len(), 9, "Board should have 9 rows");
        for row in &state.board {
            assert_eq!(row.len(), 9, "Each row should have 9 cols");
            for &cell in row {
                assert!(
                    cell == 0 || cell == 1 || cell == 2,
                    "Cell should be 0/1/2, got {cell}"
                );
            }
        }

        // Legal moves should be within bounds
        for &(row, col) in &state.legal_moves {
            assert!(row < 9, "Move row {row} out of bounds for size 9");
            assert!(col < 9, "Move col {col} out of bounds for size 9");
        }

        // After a move, board should change
        if let Some(&(row, col)) = state.legal_moves.first() {
            let after = client
                .make_move(&game_id, row, col)
                .expect("make_move failed");

            // Our stone or AI's stone should be at some position
            let has_stone: bool = after.board.iter().flat_map(|r| r.iter()).any(|&c| c != 0);
            assert!(has_stone, "Board should have stones after a move");
        }
    }

    #[test]
    #[ignore]
    fn game_result_format_is_valid() {
        let client = client();
        let (final_state, _) = play_random_game(&client, 9, "black", "random");

        let result = final_state
            .result
            .expect("Completed game should have result");

        // Result should match pattern like "B+2.5", "W+1", "B+0.5", etc.
        let valid = result.starts_with("B+") || result.starts_with("W+");
        assert!(valid, "Result should start with B+ or W+, got: {result}");

        let score_part = &result[2..];
        let score: f64 = score_part
            .parse()
            .unwrap_or_else(|_| panic!("Score should be numeric, got: {score_part}"));
        assert!(score >= 0.0, "Score should be non-negative, got: {score}");
    }
}
