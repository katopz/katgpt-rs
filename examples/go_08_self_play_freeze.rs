//! Plan 092: Self-Play Freeze/Thaw Knowledge Pipeline.
//!
//! Demonstrates GoHLPlayer freeze/thaw with meaningful opponent:
//! 1. Phase 1 (LEARN):    200 games, naive GoHL vs Validator — bandit learns from strong opponent
//! 2. Phase 2 (FROZEN):   200 games, frozen GoHL vs Validator — test knowledge
//! 3. Phase 3 (BASELINE): 200 games, naive GoHL vs Validator — no knowledge
//!
//! Phase 2 vs Phase 3 delta shows whether frozen knowledge transfers.
//!
//! ```sh
//! cargo run --features go --example go_08_self_play_freeze
//! ```

use std::path::Path;

use fastrand::Rng;

use katgpt_rs::pruners::go::{
    GoAction, GoCell, GoFrozenBandit, GoHLPlayer, GoPlayer, GoState, GoValidatorPlayer,
};
use katgpt_rs::pruners::{load_frozen, save_frozen};

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
    black: &mut dyn GoPlayer,
    white: &mut dyn GoPlayer,
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
            GoCell::Black => black.select_move(&state, &legal_moves, &mut rng),
            GoCell::White => white.select_move(&state, &legal_moves, &mut rng),
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

#[derive(Debug, Clone, Copy, Default)]
struct PhaseStats {
    /// Wins for the player we're tracking (first arg to run_phase).
    wins: usize,
    total_score_delta: f32,
    total_moves: usize,
}

/// Run ROUNDS games, alternating colors, returning aggregate stats.
/// `track_player` is 0=black, 1=white — which arg to track wins for.
/// `update_hl` optionally updates HL bandit after each game (learning phase).
fn run_phase(
    player_a: &mut dyn GoPlayer,
    player_b: &mut dyn GoPlayer,
    update_hl: bool,
) -> PhaseStats {
    let mut stats = PhaseStats::default();

    for round in 0..ROUNDS {
        let seed = BASE_SEED.wrapping_add(round as u64);

        player_a.reset();
        player_b.reset();

        // Alternate colors for fairness
        let result = if round % 2 == 0 {
            play_game(player_a, player_b, BOARD_SIZE, seed)
        } else {
            let r = play_game(player_b, player_a, BOARD_SIZE, seed);
            // Flip perspective — player_a was White
            // When White wins: score < 0, so -score > 0 → a_won = true
            // Draw: score == 0 → a_won = false
            GameResult {
                black_won: r.score_delta < 0.0,
                score_delta: -r.score_delta,
                total_moves: r.total_moves,
            }
        };

        if result.black_won {
            stats.wins += 1;
        }
        stats.total_score_delta += result.score_delta;
        stats.total_moves += result.total_moves;

        // Update HL bandit after each game (Phase 1 learning only)
        if update_hl {
            if let Some(hl) = player_a.as_any_mut().downcast_mut::<GoHLPlayer>() {
                hl.update_outcome(result.black_won);
            }
        }
    }

    stats
}

// ── Output Formatting ─────────────────────────────────────────

fn print_header() {
    println!(
        "╔═══ Freeze/Thaw — Go {board_size}×{board_size} ({rounds} rounds × 3 phases) ═══════╗",
        board_size = BOARD_SIZE,
        rounds = ROUNDS,
    );
    println!("║  Phase 1 LEARN:    naive GoHL vs Validator  (bandit learns)  ║");
    println!("║  Phase 2 FROZEN:   frozen GoHL vs Validator  (test it)      ║");
    println!("║  Phase 3 BASELINE: naive GoHL vs Validator   (no knowledge)  ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
}

fn print_phase_results(label: &str, opponent: &str, stats: &PhaseStats) {
    let win_pct = stats.wins as f32 / ROUNDS as f32 * 100.0;
    let avg_delta = stats.total_score_delta / ROUNDS as f32;
    let avg_moves = stats.total_moves as f32 / ROUNDS as f32;
    println!("  {label} Results (vs {opponent}):");
    println!(
        "    Wins: {wins}/{rounds} ({pct:.1}%)  |  Avg Score: {delta:+.1}  |  Avg Moves: {moves:.0}",
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

fn print_verdict(frozen_stats: &PhaseStats, baseline_stats: &PhaseStats) {
    let frozen_pct = frozen_stats.wins as f32 / ROUNDS as f32 * 100.0;
    let baseline_pct = baseline_stats.wins as f32 / ROUNDS as f32 * 100.0;
    let frozen_delta = frozen_stats.total_score_delta / ROUNDS as f32;
    let baseline_delta = baseline_stats.total_score_delta / ROUNDS as f32;

    let win_diff = frozen_pct - baseline_pct;
    let score_diff = frozen_delta - baseline_delta;

    let win_icon = if win_diff > 0.0 {
        "✅"
    } else if win_diff == 0.0 {
        "➖"
    } else {
        "❌"
    };
    let score_icon = if score_diff > 0.0 {
        "✅"
    } else if score_diff == 0.0 {
        "➖"
    } else {
        "❌"
    };

    println!("━━━ COMPARISON: Frozen vs Baseline (both vs Validator) ━━━");
    println!();
    println!("  Metric          Frozen    Baseline      Δ");
    println!("  ──────────────────────────────────────────────────");
    println!(
        "  Win Rate        {f:>5.1}%    {b:>5.1}%    {d:+.1}pp  {icon}",
        f = frozen_pct,
        b = baseline_pct,
        d = win_diff,
        icon = win_icon,
    );
    println!(
        "  Avg Score       {f:>+6.1}    {b:>+6.1}    {d:+.1}     {icon}",
        f = frozen_delta,
        b = baseline_delta,
        d = score_diff,
        icon = score_icon,
    );
    println!();

    if win_diff > 5.0 {
        println!(
            "  ✅ Frozen knowledge helps! +{diff:.0}pp win rate vs Validator.",
            diff = win_diff,
        );
    } else if win_diff > 0.0 {
        println!(
            "  🟡 Marginal improvement: +{diff:.0}pp win rate. May need more learning rounds.",
            diff = win_diff,
        );
    } else if win_diff == 0.0 {
        println!("  ➖ No difference — frozen knowledge has no measurable effect.");
        println!("     Both GoHL players use the same category priors, so the bandit");
        println!("     needs more rounds or finer granularity to matter vs Validator.");
    } else {
        println!(
            "  ❌ Frozen knowledge hurts: {diff:.0}pp worse. Possible overfitting.",
            diff = win_diff,
        );
    }

    println!();
    println!("  Frozen file: {path} (92 bytes)", path = FREEZE_PATH);
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    print_header();

    // ── Phase 1: LEARN (HL vs Validator, bandit learns) ───────
    println!("━━━ Phase 1: LEARN ({rounds} rounds) ━━━", rounds = ROUNDS);

    let mut hl_player = GoHLPlayer::new();
    let mut validator_learn = GoValidatorPlayer;

    let p1_stats = run_phase(&mut hl_player, &mut validator_learn, true);
    print_phase_results("Phase 1", "Validator", &p1_stats);

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

    // ── Phase 2: FROZEN (thawed HL vs Validator) ──────────────
    println!(
        "━━━ Phase 2: FROZEN ({rounds} rounds, thawed HL vs Validator) ━━━",
        rounds = ROUNDS,
    );

    let loaded: GoFrozenBandit = match load_frozen(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  ERROR: Failed to load frozen knowledge: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = loaded.validate() {
        eprintln!("  ERROR: Frozen knowledge validation failed: {e}");
        std::process::exit(1);
    }

    let mut thawed_hl = match GoHLPlayer::thaw(&loaded) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  ERROR: Failed to thaw player: {e}");
            std::process::exit(1);
        }
    };
    let mut validator = GoValidatorPlayer;

    let p2_stats = run_phase(&mut thawed_hl, &mut validator, false);
    print_phase_results("Phase 2", "Validator", &p2_stats);
    println!();

    // ── Phase 3: BASELINE (naive HL vs Validator) ─────────────
    println!(
        "━━━ Phase 3: BASELINE ({rounds} rounds, naive HL vs Validator) ━━━",
        rounds = ROUNDS,
    );

    let mut naive_hl = GoHLPlayer::new();
    let mut validator2 = GoValidatorPlayer;

    let p3_stats = run_phase(&mut naive_hl, &mut validator2, false);
    print_phase_results("Phase 3", "Validator", &p3_stats);
    println!();

    // ── Verdict ───────────────────────────────────────────────
    println!();
    print_verdict(&p2_stats, &p3_stats);
}
