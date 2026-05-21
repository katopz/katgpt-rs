//! Plan 092: Self-Play Freeze/Thaw Knowledge Pipeline.
//!
//! Demonstrates GoHLPlayer freeze/thaw:
//! 1. Phase 1 (LEARN): 100 games, naive GoHLPlayer (Black) vs GoRandomPlayer (White)
//! 2. Freeze bandit knowledge to disk
//! 3. Phase 2 (REPLAY): 100 games, thawed GoHLPlayer (Black) vs GoRandomPlayer (White)
//! 4. Compare Phase 1 vs Phase 2 results
//!
//! ```sh
//! cargo run --features go --example go_08_self_play_freeze
//! ```

use std::path::Path;

use fastrand::Rng;

use microgpt_rs::pruners::go::{
    GoAction, GoCell, GoFrozenBandit, GoHLPlayer, GoPlayer, GoRandomPlayer, GoState,
};
use microgpt_rs::pruners::{load_frozen, save_frozen};

// ── Constants ──────────────────────────────────────────────────

const ROUNDS: usize = 100;
const BOARD_SIZE: usize = 9;
const BASE_SEED: u64 = 42;
const FREEZE_PATH: &str = "output/go_frozen_bandit.bin";

/// Short category names for Q-value display (matching GoMoveCategory order).
const CATEGORY_NAMES: [&str; 8] = [
    "Corner", "Side", "Center", "Cap", "Def", "Ext", "Inf", "Pass",
];

// ── Game Result ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct GameResult {
    black_won: bool,
    score_delta: f32,
    total_moves: usize,
}

// ── Game Loop ──────────────────────────────────────────────────

/// Play a single game between two players with deterministic RNG.
fn play_game(
    player1: &mut dyn GoPlayer,
    player2: &mut dyn GoPlayer,
    board_size: usize,
    seed: u64,
) -> GameResult {
    let mut rng = Rng::with_seed(seed);
    let max_moves = board_size * board_size * 3;
    let mut state = GoState::new(board_size);
    let mut moves_played = 0usize;

    while !state.is_terminal() && moves_played < max_moves {
        let legal_moves = state.legal_moves();

        if legal_moves.is_empty() {
            state.play_pass();
            moves_played += 1;
            continue;
        }

        let action = match state.to_play {
            GoCell::Black => player1.select_move(&state, &legal_moves, &mut rng),
            GoCell::White => player2.select_move(&state, &legal_moves, &mut rng),
            GoCell::Empty => GoAction::Pass,
        };

        match &action {
            GoAction::Place(row, col) => {
                state.play_move(*row, *col);
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }

        moves_played += 1;
    }

    // Force game end if max moves reached
    if !state.is_terminal() {
        state.play_pass();
        state.play_pass();
    }

    let score = state.score(); // Positive = Black wins

    GameResult {
        black_won: score > 0.0,
        score_delta: score,
        total_moves: moves_played,
    }
}

// ── Phase Runner ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct PhaseStats {
    wins: usize,
    total_score_delta: f32,
    total_moves: usize,
}

/// Run ROUNDS games, returning aggregate stats.
fn run_phase(
    hl_player: &mut dyn GoPlayer,
    random_player: &mut dyn GoPlayer,
    update_hl: bool,
) -> PhaseStats {
    let mut wins = 0usize;
    let mut total_score_delta = 0.0f32;
    let mut total_moves = 0usize;

    for round in 0..ROUNDS {
        // Deterministic seed per round
        let seed = BASE_SEED.wrapping_add(round as u64);

        // Reset trace between games
        hl_player.reset();
        random_player.reset();

        let result = play_game(hl_player, random_player, BOARD_SIZE, seed);

        if result.black_won {
            wins += 1;
        }
        total_score_delta += result.score_delta;
        total_moves += result.total_moves;

        // Update HL bandit after each game (Phase 1 only)
        if update_hl && let Some(hl) = hl_player.as_any_mut().downcast_mut::<GoHLPlayer>() {
            hl.update_outcome(result.black_won);
        }
    }

    PhaseStats {
        wins,
        total_score_delta,
        total_moves,
    }
}

// ── Output Formatting ─────────────────────────────────────────

fn print_header() {
    println!(
        "╔═══ Self-Play Freeze/Thaw — Go {board_size}×{board_size} ({rounds} rounds × 2 phases) ═════╗",
        board_size = BOARD_SIZE,
        rounds = ROUNDS,
    );
    println!("║  Phase 1: LEARN (naive GoHL) vs Random                         ║");
    println!("║  Phase 2: REPLAY (frozen GoHL) vs Random                       ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_phase_results(label: &str, stats: &PhaseStats) {
    let win_pct = stats.wins as f32 / ROUNDS as f32 * 100.0;
    let avg_delta = stats.total_score_delta / ROUNDS as f32;
    let avg_moves = stats.total_moves as f32 / ROUNDS as f32;
    println!("  {label} Results:");
    println!(
        "    HL Wins: {wins}/{rounds} ({pct:.1}%)  |  Avg Score Delta: {delta:+.1}  |  Avg Moves: {moves:.0}",
        wins = stats.wins,
        rounds = ROUNDS,
        pct = win_pct,
        delta = avg_delta,
        moves = avg_moves,
    );
}

fn print_q_values(frozen: &GoFrozenBandit) {
    let q_strs: Vec<String> = CATEGORY_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| format!("{name}:{:.2}", frozen.q_values[i]))
        .collect();
    println!("  Q-values: [{}]", q_strs.join(" "));
    println!("  Epsilon: {:.3}", frozen.epsilon);
    println!("  Total pulls: {}", frozen.total_pulls);
}

fn print_comparison(p1: &PhaseStats, p2: &PhaseStats) {
    let p1_win_pct = p1.wins as f32 / ROUNDS as f32 * 100.0;
    let p2_win_pct = p2.wins as f32 / ROUNDS as f32 * 100.0;
    let p1_avg_delta = p1.total_score_delta / ROUNDS as f32;
    let p2_avg_delta = p2.total_score_delta / ROUNDS as f32;

    let win_delta = p2_win_pct - p1_win_pct;
    let score_delta = p2_avg_delta - p1_avg_delta;

    let win_icon = match win_delta > 0.0 {
        true => "✅",
        false => "⚠️",
    };
    let score_icon = match score_delta > 0.0 {
        true => "✅",
        false => "⚠️",
    };

    println!("━━━ COMPARISON ━━━");
    println!("  Metric          Phase 1    Phase 2    Δ");
    println!(
        "  Win Rate        {p1:>5.1}%    {p2:>5.1}%    {delta:+.1}pp  {icon}",
        p1 = p1_win_pct,
        p2 = p2_win_pct,
        delta = win_delta,
        icon = win_icon,
    );
    println!(
        "  Avg Δ Score     {p1:>+6.1}    {p2:>+6.1}    {delta:+.1}     {icon}",
        p1 = p1_avg_delta,
        p2 = p2_avg_delta,
        delta = score_delta,
        icon = score_icon,
    );
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    print_header();

    // ── Phase 1: LEARN ────────────────────────────────────────
    println!("━━━ Phase 1: LEARN ({rounds} rounds) ━━━", rounds = ROUNDS);

    let mut hl_player = GoHLPlayer::new();
    let mut random_player = GoRandomPlayer;

    let p1_stats = run_phase(&mut hl_player, &mut random_player, true);
    print_phase_results("Phase 1", &p1_stats);
    println!();

    // Freeze knowledge
    let frozen = hl_player.freeze();
    let path = Path::new(FREEZE_PATH);
    match save_frozen(path, &frozen) {
        Ok(()) => println!("  Frozen knowledge saved to {FREEZE_PATH}"),
        Err(e) => {
            eprintln!("  ERROR: Failed to save frozen knowledge: {e}");
            std::process::exit(1);
        }
    }
    print_q_values(&frozen);
    println!();

    // ── Phase 2: REPLAY ───────────────────────────────────────
    println!(
        "━━━ Phase 2: REPLAY ({rounds} rounds, frozen knowledge) ━━━",
        rounds = ROUNDS,
    );

    // Load frozen knowledge
    let loaded: GoFrozenBandit = match load_frozen(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  ERROR: Failed to load frozen knowledge: {e}");
            std::process::exit(1);
        }
    };

    // Validate
    if let Err(e) = loaded.validate() {
        eprintln!("  ERROR: Frozen knowledge validation failed: {e}");
        std::process::exit(1);
    }

    // Thaw into new player
    let mut thawed_player = match GoHLPlayer::thaw(&loaded) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  ERROR: Failed to thaw player: {e}");
            std::process::exit(1);
        }
    };
    let mut random_player2 = GoRandomPlayer;

    let p2_stats = run_phase(&mut thawed_player, &mut random_player2, false);
    print_phase_results("Phase 2", &p2_stats);
    println!();

    // ── Comparison ────────────────────────────────────────────
    print_comparison(&p1_stats, &p2_stats);
    println!();
}
