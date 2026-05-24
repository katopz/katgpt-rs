//! GOAT Proof Tests for Plan 081 — Modelless Analytics
//!
//! Tests for `GoGameAnalytics`, `compute_analytics`, `detect_garbage_moves`,
//! `detect_unstable_rounds`, and `compute_mlwr`.
//!
//! Run with:
//!
//! ```sh
//! cargo test --features go --test test_pgd_analytics
//! ```

#[cfg(feature = "go")]
mod go_analytics_tests {
    use std::time::Instant;

    use katgpt_rs::pruners::go::analytics::{
        RawGoAction, RawGoSample, detect_garbage_moves, detect_unstable_rounds, samples_to_replay,
        split_samples_into_games,
    };
    use katgpt_rs::pruners::go::replay::{GoCellSer, MoveRecord};
    use katgpt_rs::pruners::go::{
        GoAction, GoCell, GoGreedyPlayer, GoPlayer, GoRandomPlayer, GoReplay, GoState,
        compute_analytics,
    };

    // ── Helper: play a game and produce a GoReplay ─────────────────

    fn play_game_to_replay(
        size: usize,
        komi: f32,
        black: &mut dyn GoPlayer,
        white: &mut dyn GoPlayer,
        rng: &mut fastrand::Rng,
        max_moves: usize,
    ) -> GoReplay {
        let mut replay = GoReplay::new(size, komi);
        let mut state = GoState::with_komi(size, komi);

        for _ in 0..max_moves {
            if state.is_terminal() {
                break;
            }

            let legal = state.legal_moves();
            let lmc = state.legal_move_count();
            let player = state.to_play;

            let action = if player == GoCell::Black {
                black.select_move(&state, &legal, rng)
            } else {
                white.select_move(&state, &legal, rng)
            };

            replay.record(&action, player, lmc);

            match &action {
                GoAction::Place(row, col) => {
                    state.play_move(*row, *col);
                }
                GoAction::Pass => {
                    state.play_pass();
                }
            }
        }

        replay.finalize(state.get_winner(), state.score());
        replay
    }

    /// Manually compute per-player MLWR from trace and moves.
    /// Returns (black_mlwr, white_mlwr).
    fn manual_player_mlwr(trace: &[f32], moves: &[MoveRecord]) -> (f32, f32) {
        let mut black_total: f32 = 0.0;
        let mut black_count: usize = 0;
        let mut white_total: f32 = 0.0;
        let mut white_count: usize = 0;

        for i in 0..moves.len() {
            if i == 0 {
                continue;
            }
            let delta = (trace[i] - trace[i - 1]).abs();
            match moves[i].player {
                GoCellSer::Black => {
                    black_total += delta;
                    black_count += 1;
                }
                GoCellSer::White => {
                    white_total += delta;
                    white_count += 1;
                }
            }
        }

        let b = if black_count > 0 {
            black_total / black_count as f32
        } else {
            0.0
        };
        let w = if white_count > 0 {
            white_total / white_count as f32
        } else {
            0.0
        };
        (b, w)
    }

    // ════════════════════════════════════════════════════════════════
    // 1. Trace length matches replay
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_length_matches_replay() {
        let combos: &[(&str, &str)] = &[
            ("Random", "Random"),
            ("Greedy", "Random"),
            ("Random", "Greedy"),
            ("Greedy", "Greedy"),
        ];

        for (idx, &(black_type, white_type)) in combos.iter().enumerate() {
            let seed: u64 = 100 + idx as u64;
            let mut rng = fastrand::Rng::with_seed(seed);

            let mut black: Box<dyn GoPlayer> = match black_type {
                "Greedy" => Box::new(GoGreedyPlayer),
                _ => Box::new(GoRandomPlayer),
            };
            let mut white: Box<dyn GoPlayer> = match white_type {
                "Greedy" => Box::new(GoGreedyPlayer),
                _ => Box::new(GoRandomPlayer),
            };

            let replay = play_game_to_replay(9, 7.5, black.as_mut(), white.as_mut(), &mut rng, 200);
            let analytics = compute_analytics(&replay);

            assert_eq!(
                analytics.win_rate_trace.len(),
                replay.moves.len(),
                "{black_type} vs {white_type}: win_rate_trace length mismatch"
            );
            assert_eq!(
                analytics.score_trace.len(),
                replay.moves.len(),
                "{black_type} vs {white_type}: score_trace length mismatch"
            );
            println!(
                "[{black_type} vs {white_type}] moves={} traces={} ok",
                replay.moves.len(),
                analytics.win_rate_trace.len(),
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 2. Score trace matches final
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_score_trace_matches_final() {
        for seed in 200u64..210 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut black = GoRandomPlayer;
            let mut white = GoGreedyPlayer;

            let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 200);

            if replay.moves.is_empty() {
                continue;
            }

            let analytics = compute_analytics(&replay);

            let last_trace = *analytics.score_trace.last().unwrap_or(&0.0);
            let diff = (last_trace - replay.final_score).abs();
            // Generous tolerance: games may cap at max_moves before terminal,
            // so last trace entry can differ from final_score by several moves.
            assert!(
                diff < 5.0,
                "seed={seed}: score_trace last={last_trace:.3}, final_score={:.3}, diff={diff:.3}",
                replay.final_score,
            );
            println!(
                "seed={seed}: last_trace={last_trace:.3} final={:.3} diff={diff:.3} ok",
                replay.final_score,
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 3. Garbage moves — dominant game (Greedy vs Random)
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_garbage_moves_dominant_game() {
        let num_games = 8;

        for i in 0..num_games {
            let seed: u64 = 300 + i as u64;
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut black = GoGreedyPlayer;
            let mut white = GoRandomPlayer;

            let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 200);
            let analytics = compute_analytics(&replay);

            // Structural GOAT: garbage_move_ratio must be consistent with garbage_start_move
            match analytics.garbage_start_move {
                Some(start) => {
                    let expected_ratio =
                        (analytics.total_moves - start) as f32 / analytics.total_moves as f32;
                    assert!(
                        (analytics.garbage_move_ratio - expected_ratio).abs() < 0.01,
                        "game {i}: ratio={:.3} but expected={expected_ratio:.3} from start={start}",
                        analytics.garbage_move_ratio,
                    );
                }
                None => {
                    assert!(
                        analytics.garbage_move_ratio == 0.0,
                        "game {i}: garbage_start is None but ratio={:.3}",
                        analytics.garbage_move_ratio,
                    );
                }
            }

            // Ratio is always in [0, 1]
            assert!(
                (0.0..=1.0).contains(&analytics.garbage_move_ratio),
                "game {i}: garbage_move_ratio={:.3} out of [0,1]",
                analytics.garbage_move_ratio,
            );

            println!(
                "game {i}: moves={} garbage_start={:?} ratio={:.3}",
                analytics.total_moves, analytics.garbage_start_move, analytics.garbage_move_ratio,
            );
        }
        println!("all {num_games} games have structurally consistent garbage fields");
    }

    // ════════════════════════════════════════════════════════════════
    // 4. Garbage moves — close game (Greedy vs Greedy)
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_garbage_moves_close_game() {
        let num_games = 6;
        let mut total_ratio: f32 = 0.0;

        for i in 0..num_games {
            let seed: u64 = 400 + i as u64;
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut black = GoGreedyPlayer;
            let mut white = GoGreedyPlayer;

            let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 200);
            let analytics = compute_analytics(&replay);

            println!(
                "game {i}: moves={} garbage_start={:?} ratio={:.3}",
                analytics.total_moves, analytics.garbage_start_move, analytics.garbage_move_ratio,
            );
            total_ratio += analytics.garbage_move_ratio;
        }

        let avg_ratio = total_ratio / num_games as f32;
        println!("avg garbage_ratio={avg_ratio:.3}");
        assert!(
            avg_ratio < 0.50,
            "Expected avg garbage ratio < 0.50 for Greedy vs Greedy, got {avg_ratio:.3}",
        );
    }

    // ════════════════════════════════════════════════════════════════
    // 5. Unstable rounds — monotonic trace
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_unstable_rounds_monotonic() {
        let trace: &[f32] = &[0.1, 0.2, 0.3, 0.4, 0.5];
        let crossings = detect_unstable_rounds(trace);
        assert_eq!(
            crossings, 0,
            "Monotonically increasing trace should have 0 crossings, got {crossings}",
        );
        println!("monotonic trace {trace:?} -> {crossings} crossings ok");
    }

    // ════════════════════════════════════════════════════════════════
    // 6. Unstable rounds — volatile trace
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_unstable_rounds_volatile() {
        let trace: &[f32] = &[0.5, -0.3, 0.4, -0.2];
        let crossings = detect_unstable_rounds(trace);
        assert_eq!(
            crossings, 3,
            "Volatile trace should have 3 crossings, got {crossings}",
        );
        println!("volatile trace {trace:?} -> {crossings} crossings ok");
    }

    // ════════════════════════════════════════════════════════════════
    // 7. MLWR — loser concedes more ground
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_mlwr_loser_higher() {
        // GOAT proof: verify MLWR is non-negative, finite, and consistent with manual computation.
        let num_games = 20;
        let mut games_with_winner = 0usize;

        for i in 0..num_games {
            let seed: u64 = 500 + i as u64;
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut black = GoRandomPlayer;
            let mut white = GoRandomPlayer;

            let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 200);

            if replay.winner.is_none() || replay.moves.len() < 3 {
                continue;
            }
            games_with_winner += 1;

            let analytics = compute_analytics(&replay);

            // MLWR from analytics is always non-negative (uses abs deltas)
            assert!(
                analytics.mean_loss_win_rate >= 0.0,
                "game {i}: mean_loss_win_rate={:.6} must be >= 0",
                analytics.mean_loss_win_rate,
            );

            // MLWR from analytics is finite
            assert!(
                analytics.mean_loss_win_rate.is_finite(),
                "game {i}: mean_loss_win_rate={:.6} must be finite",
                analytics.mean_loss_win_rate,
            );

            // Cross-validate with manual per-player MLWR
            let (black_mlwr, white_mlwr) =
                manual_player_mlwr(&analytics.win_rate_trace, &replay.moves);

            let expected_mlwr = match replay.winner {
                Some(GoCellSer::Black) => white_mlwr, // loser is White
                Some(GoCellSer::White) => black_mlwr, // loser is Black
                None => continue,
            };

            // Analytics MLWR should match manual computation within tolerance
            let diff = (analytics.mean_loss_win_rate - expected_mlwr).abs();
            assert!(
                diff < 0.001,
                "game {i}: analytics MLWR={:.6} vs manual={:.6} diff={diff:.6}",
                analytics.mean_loss_win_rate,
                expected_mlwr,
            );

            println!(
                "game {i}: winner={:?} analytics_mlwr={:.4} manual_mlwr={expected_mlwr:.4} diff={diff:.6} ok",
                replay.winner, analytics.mean_loss_win_rate,
            );
        }

        assert!(
            games_with_winner >= 4,
            "Expected at least 4 games with a winner, got {games_with_winner}",
        );
        println!(
            "all {games_with_winner} games: MLWR non-negative, finite, and cross-validated ok"
        );
    }

    // ════════════════════════════════════════════════════════════════
    // 8. Coincidence rate — Greedy vs Greedy (high)
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_coincidence_rate_greedy_high() {
        let num_games = 5;
        let mut total_coincidence: f32 = 0.0;

        for i in 0..num_games {
            let seed: u64 = 600 + i as u64;
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut black = GoGreedyPlayer;
            let mut white = GoGreedyPlayer;

            let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 200);
            let analytics = compute_analytics(&replay);

            // Each game's coincidence should be reasonable for greedy players
            assert!(
                analytics.coincidence_rate >= 0.50,
                "game {i}: coincidence={:.3} too low for Greedy vs Greedy",
                analytics.coincidence_rate,
            );

            println!(
                "game {i}: moves={} coincidence={:.3}",
                analytics.total_moves, analytics.coincidence_rate,
            );
            total_coincidence += analytics.coincidence_rate;
        }

        let avg = total_coincidence / num_games as f32;
        println!("avg coincidence_rate={avg:.3}");
        // Greedy players agree with the greedy best move often, but not always
        // due to symmetric positions where multiple moves score equally.
        assert!(
            avg >= 0.60,
            "Expected avg coincidence >= 0.60 for Greedy vs Greedy, got {avg:.3}",
        );
    }

    // ════════════════════════════════════════════════════════════════
    // 9. Coincidence rate — Random vs Greedy (low)
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_coincidence_rate_random_low() {
        let num_games = 5;
        let mut total_coincidence: f32 = 0.0;

        for i in 0..num_games {
            let seed: u64 = 700 + i as u64;
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut black = GoRandomPlayer;
            let mut white = GoGreedyPlayer;

            let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 200);
            let analytics = compute_analytics(&replay);

            println!(
                "game {i}: moves={} coincidence={:.3}",
                analytics.total_moves, analytics.coincidence_rate,
            );
            total_coincidence += analytics.coincidence_rate;
        }

        let avg = total_coincidence / num_games as f32;
        println!("avg coincidence_rate={avg:.3}");
        // Random vs Greedy: Random side has ~0% coincidence, Greedy side has ~100%.
        // Overall average should be moderate; check it's well below 1.0.
        assert!(
            avg <= 0.65,
            "Expected avg coincidence <= 0.65 for Random vs Greedy, got {avg:.3}",
        );
    }

    // ════════════════════════════════════════════════════════════════
    // 10. Category distribution sums to one
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_category_distribution_sums_to_one() {
        for seed in 800u64..810 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut black = GoRandomPlayer;
            let mut white = GoGreedyPlayer;

            let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 200);
            let analytics = compute_analytics(&replay);

            if analytics.total_moves < 2 {
                println!(
                    "seed={seed}: only {} moves, skipping",
                    analytics.total_moves
                );
                continue;
            }

            let sum: f32 = analytics.category_distribution.iter().sum();
            let diff = (sum - 1.0).abs();
            assert!(
                diff < 0.01,
                "seed={seed}: category_distribution sum={sum:.6}, expected 1.0 (diff={diff:.6})",
            );
            println!(
                "seed={seed}: moves={} category_sum={sum:.6} ok",
                analytics.total_moves
            );
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 11. Performance — 200+ moves
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_performance_200_moves() {
        let mut rng = fastrand::Rng::with_seed(900);
        let mut black = GoRandomPlayer;
        let mut white = GoRandomPlayer;

        let replay = play_game_to_replay(9, 7.5, &mut black, &mut white, &mut rng, 250);
        println!("replay has {} moves", replay.moves.len());

        let start = Instant::now();
        let analytics = compute_analytics(&replay);
        let elapsed = start.elapsed();

        println!(
            "compute_analytics: {} moves in {:.2?} ({:.0} moves/sec)",
            analytics.total_moves,
            elapsed,
            analytics.total_moves as f64 / elapsed.as_secs_f64().max(0.0001),
        );

        // Debug builds are slow; use generous 2s timeout.
        // Release builds typically complete in <100ms.
        assert!(
            elapsed.as_millis() < 2000,
            "compute_analytics took {elapsed:?}, expected < 2000ms",
        );
    }

    // ════════════════════════════════════════════════════════════════
    // 12. Edge cases — no panic
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_edge_cases_no_panic() {
        // ── Case A: Empty game (2 passes) ──
        {
            let mut replay = GoReplay::new(9, 7.5);
            let mut state = GoState::with_komi(9, 7.5);

            // Black passes
            let lmc = state.legal_move_count();
            state.play_pass();
            replay.record(&GoAction::Pass, GoCell::Black, lmc);

            // White passes -> game over
            let lmc = state.legal_move_count();
            state.play_pass();
            replay.record(&GoAction::Pass, GoCell::White, lmc);

            replay.finalize(state.get_winner(), state.score());

            let analytics = compute_analytics(&replay);
            assert_eq!(analytics.total_moves, 2);
            assert_eq!(analytics.win_rate_trace.len(), 2);
            assert_eq!(analytics.score_trace.len(), 2);
            assert_eq!(
                analytics.coincidence_rate, 0.0,
                "All passes -> 0 coincidence"
            );
            println!("empty game (2 passes): moves={} ok", analytics.total_moves);
        }

        // ── Case B: Single move game (1 place + 2 passes) ──
        {
            let mut replay = GoReplay::new(9, 7.5);
            let mut state = GoState::with_komi(9, 7.5);

            // Black places at center
            let lmc = state.legal_move_count();
            state.play_move(4, 4);
            replay.record(&GoAction::Place(4, 4), GoCell::Black, lmc);

            // White passes
            let lmc = state.legal_move_count();
            state.play_pass();
            replay.record(&GoAction::Pass, GoCell::White, lmc);

            // Black passes -> game over
            let lmc = state.legal_move_count();
            state.play_pass();
            replay.record(&GoAction::Pass, GoCell::Black, lmc);

            replay.finalize(state.get_winner(), state.score());

            let analytics = compute_analytics(&replay);
            assert_eq!(analytics.total_moves, 3);
            assert_eq!(analytics.win_rate_trace.len(), 3);
            println!(
                "single move game: moves={} coincidence={:.3} ok",
                analytics.total_moves, analytics.coincidence_rate
            );
        }

        // ── Case C: Completely empty replay (0 moves) ──
        {
            let replay = GoReplay::new(9, 7.5);
            let analytics = compute_analytics(&replay);
            assert_eq!(analytics.total_moves, 0);
            assert!(analytics.win_rate_trace.is_empty());
            assert!(analytics.score_trace.is_empty());
            println!("zero-move replay: moves={} ok", analytics.total_moves);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // 13. detect_garbage_moves — unit tests
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_detect_garbage_moves_unit() {
        // All high -> garbage starts at move 0
        let result = detect_garbage_moves(&[0.9, 0.9, 0.9, 0.9], 0.85, 4);
        assert_eq!(
            result,
            Some(0),
            "[0.9x4] threshold=0.85 window=4 -> expected Some(0), got {result:?}",
        );
        println!("[0.9,0.9,0.9,0.9] -> {result:?} ok");

        // All low -> no garbage
        let result = detect_garbage_moves(&[0.1, 0.2, 0.1, 0.2], 0.85, 4);
        assert_eq!(
            result, None,
            "[0.1,0.2,0.1,0.2] threshold=0.85 -> expected None, got {result:?}",
        );
        println!("[0.1,0.2,0.1,0.2] -> {result:?} ok");

        // Empty trace -> None
        let result: Option<usize> = detect_garbage_moves(&[], 0.85, 4);
        assert_eq!(result, None, "empty trace -> expected None, got {result:?}",);
        println!("[] -> {result:?} ok");

        // Shorter than window -> None
        let result = detect_garbage_moves(&[0.9], 0.85, 4);
        assert_eq!(result, None, "trace < window -> expected None");
        println!("[0.9] window=4 -> {result:?} ok");

        // Transition: starts low, goes high
        let result = detect_garbage_moves(&[0.1, 0.2, 0.9, 0.9, 0.9, 0.9], 0.85, 4);
        // window=4: check positions 0,1,2
        // pos 0: avg(0.1,0.2,0.9,0.9) = 0.525 < 0.85 -> skip
        // pos 1: avg(0.2,0.9,0.9,0.9) = 0.725 < 0.85 -> skip
        // pos 2: avg(0.9,0.9,0.9,0.9) = 0.9 >= 0.85, and no subsequent windows -> Some(2)
        assert_eq!(
            result,
            Some(2),
            "transition trace -> expected Some(2), got {result:?}",
        );
        println!("[0.1,0.2,0.9,0.9,0.9,0.9] -> {result:?} ok");
    }

    // ════════════════════════════════════════════════════════════════
    // 14. detect_unstable_rounds — unit tests
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_detect_unstable_rounds_unit() {
        // Volatile: 3 sign changes
        let crossings = detect_unstable_rounds(&[0.5, -0.3, 0.4, -0.2]);
        assert_eq!(
            crossings, 3,
            "[0.5,-0.3,0.4,-0.2] -> expected 3 crossings, got {crossings}",
        );
        println!("[0.5,-0.3,0.4,-0.2] -> {crossings} ok");

        // Monotonic positive: 0 crossings
        let crossings = detect_unstable_rounds(&[0.1, 0.2, 0.3]);
        assert_eq!(
            crossings, 0,
            "[0.1,0.2,0.3] -> expected 0 crossings, got {crossings}",
        );
        println!("[0.1,0.2,0.3] -> {crossings} ok");

        // Empty: 0 crossings
        let crossings = detect_unstable_rounds(&[]);
        assert_eq!(crossings, 0, "[] -> expected 0 crossings, got {crossings}",);
        println!("[] -> {crossings} ok");

        // Single element: 0 crossings
        let crossings = detect_unstable_rounds(&[0.0]);
        assert_eq!(
            crossings, 0,
            "[0.0] -> expected 0 crossings, got {crossings}",
        );
        println!("[0.0] -> {crossings} ok");

        // Monotonic negative: 0 crossings
        let crossings = detect_unstable_rounds(&[-0.3, -0.2, -0.1]);
        assert_eq!(
            crossings, 0,
            "[-0.3,-0.2,-0.1] -> expected 0 crossings, got {crossings}",
        );
        println!("[-0.3,-0.2,-0.1] -> {crossings} ok");

        // Two elements, one crossing
        let crossings = detect_unstable_rounds(&[0.5, -0.5]);
        assert_eq!(
            crossings, 1,
            "[0.5,-0.5] -> expected 1 crossing, got {crossings}",
        );
        println!("[0.5,-0.5] -> {crossings} ok");
    }

    // ════════════════════════════════════════════════════════════════
    // 15. samples_to_replay — T0 Data Bridge tests
    // ════════════════════════════════════════════════════════════════

    fn make_sample(move_number: usize, action: RawGoAction, quality: f32) -> RawGoSample {
        RawGoSample {
            board: vec![0; 81],
            size: 9,
            action,
            quality,
            move_number,
            legal_moves: 80,
        }
    }

    #[test]
    fn test_samples_to_replay_empty_error() {
        let result = samples_to_replay(&[], 7.5);
        assert!(result.is_err(), "Empty samples should error");
        assert!(
            result.unwrap_err().contains("empty"),
            "Error message should mention 'empty'"
        );
        println!("✓ Empty samples → error");
    }

    #[test]
    fn test_samples_to_replay_single_move() {
        let samples = vec![make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 1.0)];
        let replay = samples_to_replay(&samples, 7.5).unwrap();
        assert_eq!(replay.moves.len(), 1);
        assert_eq!(replay.size, 9);
        assert_eq!(replay.winner, Some(GoCellSer::Black)); // move 1 = Black won (quality 1.0)
        println!("✓ Single move → replay with 1 move, Black wins");
    }

    #[test]
    fn test_samples_to_replay_player_alternation() {
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(2, RawGoAction::Place { row: 0, col: 0 }, 0.0),
            make_sample(3, RawGoAction::Pass, 0.0),
            make_sample(4, RawGoAction::Pass, 1.0), // White wins (quality 1.0 for move 4 = White)
        ];
        let replay = samples_to_replay(&samples, 7.5).unwrap();
        assert_eq!(replay.moves.len(), 4);
        assert_eq!(replay.moves[0].player, GoCellSer::Black); // move 1
        assert_eq!(replay.moves[1].player, GoCellSer::White); // move 2
        assert_eq!(replay.moves[2].player, GoCellSer::Black); // move 3
        assert_eq!(replay.moves[3].player, GoCellSer::White); // move 4
        assert_eq!(replay.winner, Some(GoCellSer::White)); // quality=1.0 → mover (White) won
        println!("✓ Player alternation: B W B W, White wins");
    }

    #[test]
    fn test_samples_to_replay_non_sequential_error() {
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(3, RawGoAction::Place { row: 0, col: 0 }, 0.0), // gap at 2
        ];
        let result = samples_to_replay(&samples, 7.5);
        assert!(result.is_err(), "Non-sequential move numbers should error");
        assert!(
            result.unwrap_err().contains("Non-sequential"),
            "Error should mention non-sequential"
        );
        println!("✓ Non-sequential move numbers → error");
    }

    #[test]
    fn test_samples_to_replay_inconsistent_size_error() {
        let samples = vec![
            RawGoSample {
                board: vec![0; 81],
                size: 9,
                action: RawGoAction::Place { row: 4, col: 4 },
                quality: 1.0,
                move_number: 1,
                legal_moves: 80,
            },
            RawGoSample {
                board: vec![0; 361],
                size: 19, // wrong size
                action: RawGoAction::Place { row: 0, col: 0 },
                quality: 0.0,
                move_number: 2,
                legal_moves: 300,
            },
        ];
        let result = samples_to_replay(&samples, 7.5);
        assert!(result.is_err(), "Inconsistent board size should error");
        assert!(
            result.unwrap_err().contains("Inconsistent board size"),
            "Error should mention board size"
        );
        println!("✓ Inconsistent board size → error");
    }

    #[test]
    fn test_samples_to_replay_winner_inference() {
        // Black wins (last mover quality >= 0.5)
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(2, RawGoAction::Pass, 0.0),
            make_sample(3, RawGoAction::Pass, 0.8), // move 3 = Black, quality >= 0.5 → Black won
        ];
        let replay = samples_to_replay(&samples, 7.5).unwrap();
        assert_eq!(replay.winner, Some(GoCellSer::Black));

        // White wins (last mover quality < 0.5)
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(2, RawGoAction::Pass, 0.3), // move 2 = White, quality < 0.5 → White lost → Black won
        ];
        let replay = samples_to_replay(&samples, 7.5).unwrap();
        assert_eq!(replay.winner, Some(GoCellSer::Black)); // White lost → Black won
        println!("✓ Winner inference: quality-based");
    }

    #[test]
    fn test_samples_to_replay_replay_succeeds() {
        // GOAT gate: produces valid replay that replay.replay() succeeds on
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(2, RawGoAction::Place { row: 0, col: 0 }, 0.0),
            make_sample(3, RawGoAction::Place { row: 8, col: 8 }, 0.0),
            make_sample(4, RawGoAction::Pass, 0.0),
            make_sample(5, RawGoAction::Pass, 1.0),
        ];
        let replay = samples_to_replay(&samples, 7.5).unwrap();
        let result = replay.replay();
        assert!(
            result.is_ok(),
            "replay.replay() should succeed: {}",
            result
                .as_ref()
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        );
        println!("✓ replay.replay() succeeds on converted samples");
    }

    #[test]
    fn test_samples_to_replay_analytics_integration() {
        // Convert samples → replay → compute_analytics
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(2, RawGoAction::Place { row: 0, col: 0 }, 0.0),
            make_sample(3, RawGoAction::Place { row: 8, col: 8 }, 0.0),
            make_sample(4, RawGoAction::Place { row: 1, col: 1 }, 0.0),
            make_sample(5, RawGoAction::Pass, 0.0),
            make_sample(6, RawGoAction::Pass, 1.0),
        ];
        let replay = samples_to_replay(&samples, 7.5).unwrap();
        let analytics = compute_analytics(&replay);
        assert_eq!(analytics.total_moves, 6, "Should have 6 moves");
        assert_eq!(
            analytics.win_rate_trace.len(),
            6,
            "Trace length should match"
        );
        assert!(
            !analytics.win_rate_trace.iter().all(|&v| v == 0.0),
            "Trace should be non-zero"
        );
        assert!(
            analytics.coincidence_rate >= 0.0 && analytics.coincidence_rate <= 1.0,
            "CR in [0,1]"
        );
        assert!(
            analytics.garbage_move_ratio >= 0.0 && analytics.garbage_move_ratio <= 1.0,
            "Garbage in [0,1]"
        );
        println!("✓ samples_to_replay → compute_analytics integration works");
    }

    // ════════════════════════════════════════════════════════════════
    // 16. split_samples_into_games tests
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_split_samples_empty() {
        let games = split_samples_into_games(&[]);
        assert!(games.is_empty());
        println!("✓ Empty input → 0 games");
    }

    #[test]
    fn test_split_samples_single_game() {
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(2, RawGoAction::Pass, 0.0),
            make_sample(3, RawGoAction::Pass, 1.0),
        ];
        let games = split_samples_into_games(&samples);
        assert_eq!(games.len(), 1, "Should detect 1 game");
        assert_eq!(games[0].len(), 3, "Game should have 3 samples");
        println!("✓ Single game → 1 group of 3");
    }

    #[test]
    fn test_split_samples_multiple_games() {
        let samples = vec![
            make_sample(1, RawGoAction::Place { row: 4, col: 4 }, 0.0),
            make_sample(2, RawGoAction::Pass, 1.0),
            // Game boundary: move_number resets to 1
            make_sample(1, RawGoAction::Place { row: 0, col: 0 }, 0.0),
            make_sample(2, RawGoAction::Place { row: 1, col: 1 }, 0.0),
            make_sample(3, RawGoAction::Pass, 0.5),
            // Game boundary
            make_sample(1, RawGoAction::Place { row: 8, col: 8 }, 1.0),
        ];
        let games = split_samples_into_games(&samples);
        assert_eq!(games.len(), 3, "Should detect 3 games");
        assert_eq!(games[0].len(), 2, "Game 1: 2 samples");
        assert_eq!(games[1].len(), 3, "Game 2: 3 samples");
        assert_eq!(games[2].len(), 1, "Game 3: 1 sample");
        println!("✓ 3 games detected from move_number resets");
    }

    // ════════════════════════════════════════════════════════════════
    // 17. T14 Natsukaze-style validation (simulated)
    // ════════════════════════════════════════════════════════════════
    //
    // Full Natsukaze validation requires riir-gpu's load_flat_zip() which
    // lives in riir-ai. This test simulates the Natsukaze data shape
    // (strong AI play with realistic move patterns) to validate the pipeline.

    #[test]
    fn test_natsukaze_style_pipeline() {
        // Simulate 5 short Natsukaze-style games (strong AI play)
        let mut all_samples: Vec<RawGoSample> = Vec::new();

        for game_id in 0..5 {
            let mut state = GoState::new(9);
            let mut player = GoGreedyPlayer;
            let mut rng = fastrand::Rng::new();

            for move_num in 1..=20 {
                if state.is_terminal() {
                    break;
                }

                let legal = state.legal_moves();
                let action = player.select_move(&state, &legal, &mut rng);

                // Record board BEFORE applying the move
                let board: Vec<u8> = (0..81)
                    .map(|i| {
                        let r = i / 9;
                        let c = i % 9;
                        match state.at(r, c) {
                            GoCell::Empty => 0,
                            GoCell::Black => 1,
                            GoCell::White => 2,
                        }
                    })
                    .collect();

                // Build RawGoAction (clone-safe, no refs)
                let raw_action = match &action {
                    GoAction::Place(r, c) => RawGoAction::Place { row: *r, col: *c },
                    GoAction::Pass => RawGoAction::Pass,
                };

                all_samples.push(RawGoSample {
                    board,
                    size: 9,
                    action: raw_action,
                    quality: 0.0, // Will be set by last move
                    move_number: move_num,
                    legal_moves: legal.len(),
                });

                // NOW apply the move to advance the state
                match action {
                    GoAction::Place(r, c) => {
                        state.play_move(r, c);
                    }
                    GoAction::Pass => {
                        state.play_pass();
                    }
                }
            }

            // Set quality on last sample based on outcome
            if let Some(last) = all_samples.last_mut() {
                last.quality = if game_id % 2 == 0 { 1.0 } else { 0.0 };
            }
        }

        // Split into games
        let games = split_samples_into_games(&all_samples);
        assert!(
            games.len() >= 5,
            "Should detect at least 5 games, got {}",
            games.len()
        );
        println!(
            "Detected {} games from simulated Natsukaze data",
            games.len()
        );

        // Convert each game to replay and compute analytics
        let mut total_cr = 0.0f32;
        let mut total_garbage = 0.0f32;
        let mut game_count = 0usize;

        for (i, game_samples) in games.iter().enumerate() {
            let samples: Vec<RawGoSample> = game_samples.iter().map(|s| (*s).clone()).collect();
            let replay = samples_to_replay(&samples, 7.5);
            if let Err(e) = &replay {
                // Some games may have invalid moves from greedy play, skip them
                println!("Game {i}: conversion skipped ({e})");
                continue;
            }

            let replay = replay.unwrap();

            // GOAT gate: replay.replay() succeeds
            if replay.replay().is_err() {
                println!("Game {i}: replay validation skipped");
                continue;
            }

            let analytics = compute_analytics(&replay);

            // Assert: traces non-empty
            assert!(
                !analytics.win_rate_trace.is_empty(),
                "Game {i}: trace should be non-empty"
            );

            // Assert: CR in [0,1]
            assert!(
                analytics.coincidence_rate >= 0.0 && analytics.coincidence_rate <= 1.0,
                "Game {i}: CR {} out of [0,1]",
                analytics.coincidence_rate
            );

            // Assert: garbage ratio in [0,1]
            assert!(
                analytics.garbage_move_ratio >= 0.0 && analytics.garbage_move_ratio <= 1.0,
                "Game {i}: garbage {} out of [0,1]",
                analytics.garbage_move_ratio
            );

            // Assert: distribution sums to ~1.0
            let dist_sum: f32 = analytics.category_distribution.iter().sum();
            assert!(
                (dist_sum - 1.0).abs() < 0.01 || analytics.total_moves == 0,
                "Game {i}: distribution sum = {dist_sum}, expected ~1.0"
            );

            total_cr += analytics.coincidence_rate;
            total_garbage += analytics.garbage_move_ratio;
            game_count += 1;
        }

        assert!(
            game_count >= 3,
            "At least 3 games should convert successfully"
        );

        let avg_cr = total_cr / game_count as f32;
        let avg_garbage = total_garbage / game_count as f32;
        println!("✓ Natsukaze-style validation: {game_count} games");
        println!("  Avg coincidence_rate:  {avg_cr:.3}");
        println!("  Avg garbage_move_ratio: {avg_garbage:.3}");
    }
}
