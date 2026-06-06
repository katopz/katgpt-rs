//! Go SDPG Tournament — Burn-in Oracle + GOAT Gate (Plan 194).
//!
//! Three-phase tournament:
//!   Phase 1: **Burn-in** — GoHLPlayer vs GoGreedyPlayer for N games
//!            → extract category Q-values as teacher oracle
//!   Phase 2: **Tournament** — SDPG(oracle) vs HL vs Greedy vs Random
//!   Phase 3: **GOAT Gate** — Verify SDPG win rate > HL win rate
//!
//! Run: `cargo run --features go,sdpg_bandit --example go_10_sdpg_tournament`
//!
//! Config:
//!   GO_BURNIN=100     Burn-in games for teacher oracle (default: 50)
//!   GO_GAMES=100      Tournament games per matchup (default: 50)
//!   GO_BOARD=9        Board size (default: 9)
//!   GO_GOAT_GAMES=200 GOAT gate verification games (default: 100)

use std::env;
use std::time::Instant;

use fastrand::Rng;
use katgpt_rs::pruners::go::{
    GoAction, GoCell, GoGreedyPlayer, GoHLPlayer, GoMoveCategory, GoPlayer, GoRandomPlayer,
    GoSdpgPlayer, GoState,
};

// ── Constants ──────────────────────────────────────────────────

const DEFAULT_BURNIN: usize = 50;
const DEFAULT_GAMES: usize = 50;
const DEFAULT_BOARD: usize = 9;
const DEFAULT_GOAT_GAMES: usize = 100;
const MAX_MOVES: usize = 300;

// ── Helpers ────────────────────────────────────────────────────

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Play a single game between two players, returning (first_player_won, moves).
fn play_game(
    player_a: &mut dyn GoPlayer,
    player_b: &mut dyn GoPlayer,
    player_a_color: GoCell,
    board_size: usize,
    rng: &mut Rng,
) -> (bool, usize) {
    let mut state = GoState::new(board_size);
    let mut moves = 0usize;

    for _ in 0..MAX_MOVES {
        if state.is_terminal() {
            break;
        }

        let legal = state.legal_moves();
        let action = if state.to_play == player_a_color {
            player_a.select_move(&state, &legal, rng)
        } else {
            player_b.select_move(&state, &legal, rng)
        };

        match &action {
            GoAction::Place(row, col) => {
                let ok = state.play_move(*row, *col);
                debug_assert!(ok, "Player selected illegal move ({row},{col})");
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        moves += 1;
    }

    // Force game end if not terminal
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
        moves += 2;
    }

    let winner = state.get_winner();
    let first_player_won = winner == Some(player_a_color);
    (first_player_won, moves)
}

/// Update `update_outcome()` on learning players.
fn update_learning_players(player_a: &mut dyn GoPlayer, player_b: &mut dyn GoPlayer, a_won: bool) {
    if let Some(hl) = player_a.as_any_mut().downcast_mut::<GoHLPlayer>() {
        hl.update_outcome(a_won);
    }
    if let Some(hl) = player_b.as_any_mut().downcast_mut::<GoHLPlayer>() {
        hl.update_outcome(!a_won);
    }
    if let Some(sdpg) = player_a.as_any_mut().downcast_mut::<GoSdpgPlayer>() {
        sdpg.update_outcome(a_won);
    }
    if let Some(sdpg) = player_b.as_any_mut().downcast_mut::<GoSdpgPlayer>() {
        sdpg.update_outcome(!a_won);
    }
}

// ── Phase 1: Burn-in ──────────────────────────────────────────

fn run_burnin(num_games: usize, board_size: usize, rng: &mut Rng) -> Vec<f32> {
    println!("\nPhase 1: Burn-in ({num_games} games, HL vs Greedy)");
    println!("─────────────────────────────────────────────────");

    let mut hl = GoHLPlayer::new();
    let mut greedy = GoGreedyPlayer;
    let mut hl_wins = 0usize;

    let start = Instant::now();

    for i in 0..num_games {
        let hl_color = if i % 2 == 0 {
            GoCell::Black
        } else {
            GoCell::White
        };
        let (hl_won, _) = play_game(&mut hl, &mut greedy, hl_color, board_size, rng);
        update_learning_players(&mut hl, &mut greedy, hl_won);
        if hl_won {
            hl_wins += 1;
        }
    }

    let duration = start.elapsed();
    let wr = hl_wins as f64 / num_games as f64 * 100.0;
    println!("  HL win rate: {wr:.1}% ({hl_wins}/{num_games})");
    println!("  Duration:    {d:.1}s", d = duration.as_secs_f64());

    // Extract category Q-values as teacher oracle
    let teacher_q: Vec<f32> = hl.q_values().to_vec();
    let variance = compute_variance(&teacher_q);

    println!("\n  Teacher Q (category oracle):");
    for (i, &q) in teacher_q.iter().enumerate() {
        let cat = GoMoveCategory::all()[i];
        let bar = "█".repeat(((q * 40.0).clamp(0.0, 40.0)) as usize);
        println!("    {:<12} {:.4} {}", cat.name(), q, bar);
    }
    println!("  Q variance: {variance:.4}");

    if variance < 0.01 {
        println!("  ⚠️  Low variance — categories may not differentiate strongly.");
        println!("     Consider more burn-in games (GO_BURNIN=200).");
    }

    teacher_q
}

// ── Phase 2: Tournament ───────────────────────────────────────

struct PlayerEntry {
    name: &'static str,
    wins: usize,
    games: usize,
}

fn run_tournament(
    teacher_q: &[f32],
    num_games: usize,
    board_size: usize,
    rng: &mut Rng,
) -> Vec<PlayerEntry> {
    println!("\nPhase 2: Tournament ({num_games} games per matchup, {board_size}×{board_size})");
    println!("─────────────────────────────────────────────────");

    let mut entries = vec![
        PlayerEntry {
            name: "Random",
            wins: 0,
            games: 0,
        },
        PlayerEntry {
            name: "Greedy",
            wins: 0,
            games: 0,
        },
        PlayerEntry {
            name: "HL",
            wins: 0,
            games: 0,
        },
        PlayerEntry {
            name: "SDPG",
            wins: 0,
            games: 0,
        },
    ];

    let start = Instant::now();

    // Round-robin: each pair plays num_games games
    let pairs: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

    for (a_idx, b_idx) in pairs {
        let a_name = entries[a_idx].name;
        let b_name = entries[b_idx].name;
        print!("  {a_name:>8} vs {b_name:<8} ");
        let _ = std::io::Write::flush(&mut std::io::stdout());

        let mut a_wins = 0usize;
        let pair_start = Instant::now();

        for i in 0..num_games {
            // Create fresh players for each game (no cross-game memory)
            let mut player_a: Box<dyn GoPlayer> = make_player(a_name, teacher_q);
            let mut player_b: Box<dyn GoPlayer> = make_player(b_name, teacher_q);

            let a_color = if i % 2 == 0 {
                GoCell::Black
            } else {
                GoCell::White
            };
            let (a_won, _) = play_game(
                player_a.as_mut(),
                player_b.as_mut(),
                a_color,
                board_size,
                rng,
            );

            // Feed outcome to learning players
            update_learning_players(player_a.as_mut(), player_b.as_mut(), a_won);

            if a_won {
                a_wins += 1;
            }
        }

        let b_wins = num_games - a_wins;
        let a_wr = a_wins as f64 / num_games as f64 * 100.0;
        let pair_dur = pair_start.elapsed();
        println!(
            "{:>3}–{:>3} ({:>5.1}%) {:.1}s",
            a_wins,
            b_wins,
            a_wr,
            pair_dur.as_secs_f64()
        );

        entries[a_idx].wins += a_wins;
        entries[a_idx].games += num_games;
        entries[b_idx].wins += b_wins;
        entries[b_idx].games += num_games;
    }

    let duration = start.elapsed();
    println!("\n  Duration: {d:.1}s", d = duration.as_secs_f64());

    // Sort by win rate descending
    entries.sort_by(|a, b| {
        let wr_a = a.wins as f64 / a.games as f64;
        let wr_b = b.wins as f64 / b.games as f64;
        wr_b.partial_cmp(&wr_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Print leaderboard
    println!("\n  Leaderboard:");
    println!("  ┌──────┬──────────┬──────┬──────┬───────┬───────┐");
    println!("  │ Rank │ Player   │  W   │  L   │ Games │ Win%  │");
    println!("  ├──────┼──────────┼──────┼──────┼───────┼───────┤");
    for (i, e) in entries.iter().enumerate() {
        let losses = e.games - e.wins;
        let wr = e.wins as f64 / e.games as f64 * 100.0;
        println!(
            "  │  {rank:>2} │ {name:<8} │ {wins:>4} │ {losses:>4} │ {games:>5} │ {wr:>5.1}% │",
            rank = i + 1,
            name = e.name,
            wins = e.wins,
            losses = losses,
            games = e.games
        );
    }
    println!("  └──────┴──────────┴──────┴──────┴───────┴───────┘");

    entries
}

// ── Phase 3: GOAT Gate ────────────────────────────────────────

fn run_goat_gate(teacher_q: &[f32], num_games: usize, board_size: usize, rng: &mut Rng) -> bool {
    println!("\nPhase 3: GOAT Gate ({num_games} games, SDPG(oracle) vs HL)");
    println!("─────────────────────────────────────────────────");

    let mut sdpg_wins = 0usize;
    let mut hl_wins_count = 0usize;
    let start = Instant::now();

    for i in 0..num_games {
        let mut sdpg: Box<dyn GoPlayer> =
            Box::new(GoSdpgPlayer::with_teacher_q(teacher_q.to_vec()));
        let mut hl: Box<dyn GoPlayer> = Box::new(GoHLPlayer::new());

        let sdpg_color = if i % 2 == 0 {
            GoCell::Black
        } else {
            GoCell::White
        };
        let (sdpg_won, _) = play_game(sdpg.as_mut(), hl.as_mut(), sdpg_color, board_size, rng);

        update_learning_players(sdpg.as_mut(), hl.as_mut(), sdpg_won);

        if sdpg_won {
            sdpg_wins += 1;
        } else {
            hl_wins_count += 1;
        }
    }

    let duration = start.elapsed();
    let sdpg_wr = sdpg_wins as f64 / num_games as f64 * 100.0;
    let hl_wr = hl_wins_count as f64 / num_games as f64 * 100.0;

    println!("  SDPG(oracle)  {sdpg_wins:>3}W  {hl_wins_count:>3}L  ({sdpg_wr:.1}%)");
    println!("  HL            {hl_wins_count:>3}W  {sdpg_wins:>3}L  ({hl_wr:.1}%)");
    println!("  Duration:     {:.1}s", duration.as_secs_f64());

    let gate_pass = sdpg_wins > hl_wins_count;

    println!();
    if gate_pass {
        println!("  ✅ GOAT PASSED — SDPG(oracle) > HL on {board_size}×{board_size}");
    } else {
        println!(
            "  ❌ GOAT FAILED — SDPG(oracle) {} ≤ HL {} on {board_size}×{board_size}",
            sdpg_wins, hl_wins_count
        );
    }

    gate_pass
}

// ── Player Factory ─────────────────────────────────────────────

fn make_player(name: &str, teacher_q: &[f32]) -> Box<dyn GoPlayer> {
    match name {
        "Random" => Box::new(GoRandomPlayer),
        "Greedy" => Box::new(GoGreedyPlayer),
        "HL" => Box::new(GoHLPlayer::new()),
        "SDPG" => Box::new(GoSdpgPlayer::with_teacher_q(teacher_q.to_vec())),
        _ => panic!("Unknown player: {name}"),
    }
}

// ── Utility ────────────────────────────────────────────────────

fn compute_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let var = values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    var
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let burnin_games = env_usize("GO_BURNIN", DEFAULT_BURNIN);
    let tournament_games = env_usize("GO_GAMES", DEFAULT_GAMES);
    let board_size = env_usize("GO_BOARD", DEFAULT_BOARD);
    let goat_games = env_usize("GO_GOAT_GAMES", DEFAULT_GOAT_GAMES);

    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  Go SDPG Tournament (Plan 194)");
    println!(
        "  Board: {board_size}×{board_size} | Burn-in: {burnin_games} | Games: {tournament_games} | GOAT: {goat_games}"
    );
    println!("══════════════════════════════════════════════════════════════");

    let mut rng = Rng::new();

    // Phase 1: Burn-in
    let teacher_q = run_burnin(burnin_games, board_size, &mut rng);

    // Phase 2: Tournament
    let results = run_tournament(&teacher_q, tournament_games, board_size, &mut rng);

    // Phase 3: GOAT Gate
    let goat_pass = run_goat_gate(&teacher_q, goat_games, board_size, &mut rng);

    // Summary
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("══════════════════════════════════════════════════════════════");
    println!("  Teacher Q variance: {:.4}", compute_variance(&teacher_q));

    if let Some(sdpg_entry) = results.iter().find(|e| e.name == "SDPG") {
        let sdpg_wr = sdpg_entry.wins as f64 / sdpg_entry.games as f64 * 100.0;
        println!(
            "  SDPG tournament:  {}/{} ({sdpg_wr:.1}%)",
            sdpg_entry.wins, sdpg_entry.games
        );
    }

    if let Some(hl_entry) = results.iter().find(|e| e.name == "HL") {
        let hl_wr = hl_entry.wins as f64 / hl_entry.games as f64 * 100.0;
        println!(
            "  HL tournament:    {}/{} ({hl_wr:.1}%)",
            hl_entry.wins, hl_entry.games
        );
    }

    println!(
        "  GOAT gate:        {}",
        if goat_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    // Markdown summary
    println!("\n```markdown");
    println!("## Go SDPG Tournament Results (Plan 194)");
    println!();
    println!("| Rank | Player | W | L | Games | Win% |");
    println!("|------|--------|---|---|-------|------|");
    for (i, e) in results.iter().enumerate() {
        let losses = e.games - e.wins;
        let wr = e.wins as f64 / e.games as f64 * 100.0;
        println!(
            "| {rank} | {name} | {wins} | {losses} | {games} | {wr:.1}% |",
            rank = i + 1,
            name = e.name,
            wins = e.wins,
            losses = losses,
            games = e.games
        );
    }
    println!("```");
    println!();
}
