//! Plan 065 Phase 3 T31: Go Head-to-Head Tournament against AutoGo Agents.
//!
//! Runs our player strategies against AutoGo's AI agents via REST API.
//! Requires a running AutoGo server (`play.py`).
//!
//! ```sh
//! # Start AutoGo server first
//! cd autogo && python play.py
//!
//! # Default: 10 games per matchup, 9×9 board, localhost:8765
//! cargo run --features go --example go_03_head_to_head
//!
//! # Custom: 20 games, different server
//! GO_GAMES=20 GO_URL=http://192.168.1.100:8765 cargo run --features go --example go_03_head_to_head
//!
//! # Quick test with 2 games
//! GO_GAMES=2 cargo run --features go --example go_03_head_to_head
//! ```

use std::env;

use katgpt_rs::pruners::go::{
    GoPlayerType, GoTournamentConfig, TournamentDef, print_batch_table, run_tournament_batch,
};

// ── Constants ──────────────────────────────────────────────────

/// Default number of games per matchup.
const DEFAULT_NUM_GAMES: usize = 10;

/// Default board size.
const DEFAULT_BOARD_SIZE: usize = 9;

/// Default AutoGo server URL.
const DEFAULT_AUTOGO_URL: &str = "http://localhost:8765";

// ── Tournament Definitions ─────────────────────────────────────

/// Baseline tournaments (T28):
/// - Random vs Random (sanity: should be ~50/50)
/// - Random vs gnugo1 (expect: we lose badly)
/// - Greedy vs gnugo1 (expect: competitive)
const BASELINE_TOURNAMENTS: &[TournamentDef] = &[
    TournamentDef::new(GoPlayerType::Random, "random"),
    TournamentDef::new(GoPlayerType::Greedy, "random"),
    TournamentDef::new(GoPlayerType::Greedy, "gnugo1"),
];

/// HL tournaments (T29):
/// - HL vs Random (expect: >70% win rate)
/// - HL vs gnugo1 (target: >55% win rate)
const HL_TOURNAMENTS: &[TournamentDef] = &[
    TournamentDef::new(GoPlayerType::HL, "random"),
    TournamentDef::new(GoPlayerType::HL, "gnugo1"),
];

/// G-Zero tournaments (T30):
/// - GZero vs random (expect: >70% win rate)
/// - GZero vs gnugo1 (stretch: >55% win rate)
const GZERO_TOURNAMENTS: &[TournamentDef] = &[
    TournamentDef::new(GoPlayerType::GZero, "random"),
    TournamentDef::new(GoPlayerType::GZero, "gnugo1"),
];

/// MCTS tournaments (bonus):
/// - MCTS vs random
/// - MCTS vs gnugo1
const MCTS_TOURNAMENTS: &[TournamentDef] = &[
    TournamentDef::new(GoPlayerType::MCTS, "random"),
    TournamentDef::new(GoPlayerType::MCTS, "gnugo1"),
];

// ── Output Formatting ─────────────────────────────────────────

/// Print the tournament header.
fn print_header(num_games: usize, board_size: usize, url: &str) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       Go Head-to-Head Tournament — {board_size}×{board_size}                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Games per matchup : {num_games:<6}                              ║");
    println!("║  AutoGo server     : {url:<38}  ║");
    println!("║  Both colors       : YES                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

/// Print a section header.
fn print_section(title: &str) {
    println!();
    println!("━━━ {title} ━━━");
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    // Parse env configuration
    let num_games: usize = env::var("GO_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_GAMES);

    let board_size: usize = env::var("GO_BOARD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BOARD_SIZE);

    let autogo_url = env::var("GO_URL")
        .ok()
        .unwrap_or_else(|| DEFAULT_AUTOGO_URL.to_string());

    // Select which tournament set to run
    let tournament_set = env::var("GO_SET").ok().unwrap_or_else(|| "all".to_string());

    // Collect all tournaments to run
    let mut all_tournaments: Vec<&TournamentDef> = Vec::new();

    match tournament_set.as_str() {
        "baseline" => all_tournaments.extend(BASELINE_TOURNAMENTS),
        "hl" => all_tournaments.extend(HL_TOURNAMENTS),
        "gzero" => all_tournaments.extend(GZERO_TOURNAMENTS),
        "mcts" => all_tournaments.extend(MCTS_TOURNAMENTS),
        _ => {
            // "all" or unspecified — run everything
            all_tournaments.extend(BASELINE_TOURNAMENTS);
            all_tournaments.extend(HL_TOURNAMENTS);
            all_tournaments.extend(GZERO_TOURNAMENTS);
            all_tournaments.extend(MCTS_TOURNAMENTS);
        }
    }

    print_header(num_games, board_size, &autogo_url);

    let config_template = GoTournamentConfig {
        board_size,
        num_games,
        our_player: GoPlayerType::Random, // Overridden per-tournament
        their_agent: "random".to_string(), // Overridden per-tournament
        autogo_url: autogo_url.clone(),
        play_both_colors: true,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let mut all_results: Vec<(TournamentDef, _)> = Vec::new();

    // Run each section
    let sections: &[(&str, &[TournamentDef])] = &[
        ("T28: Baseline", BASELINE_TOURNAMENTS),
        ("T29: Heuristic Learning", HL_TOURNAMENTS),
        ("T30: G-Zero", GZERO_TOURNAMENTS),
        ("Bonus: MCTS", MCTS_TOURNAMENTS),
    ];

    for (section_title, tournaments) in sections {
        // Skip sections not in the selected set
        if tournament_set != "all" {
            let section_matches = tournaments.iter().any(|t| {
                all_tournaments.iter().any(|at| {
                    at.our_player.label() == t.our_player.label() && at.their_agent == t.their_agent
                })
            });
            if !section_matches {
                continue;
            }
        }

        print_section(section_title);

        let section_results = run_tournament_batch(tournaments, &config_template, &mut rng);

        // Print section summary
        for (def, result) in &section_results {
            let wr = result.win_rate() * 100.0;
            let total = result.total_games();
            println!(
                "  {}: {}W/{loss}D/{loss2}L ({wr:.1}%) in {total} games",
                def.label(),
                result.our_wins,
                loss = result.draws,
                loss2 = result.their_wins,
            );
        }

        all_results.extend(
            section_results
                .into_iter()
                .map(|(d, r)| (d.clone() as TournamentDef, r)),
        );
    }

    // Final summary table
    if !all_results.is_empty() {
        let owned: Vec<(TournamentDef, _)> = all_results
            .into_iter()
            .map(|(d, r)| (d.clone(), r))
            .collect();
        print_batch_table(&owned);
    } else {
        println!();
        println!("  No tournaments completed. Is the AutoGo server running?");
        println!("  Start it with: cd autogo && python play.py");
        println!(
            "  Then try:      GO_GAMES=2 cargo run --features go --example go_03_head_to_head"
        );
    }

    println!();
}
