//! Bomberman SDPG Tournament — SdpgPlayer vs all baselines (Plan 180).
//!
//! Round-robin tournament pitting 8 player types in 4-player matches:
//! Random, Greedy, Validator, HL, GZero, Rubric, SDAR, SDPG
//!
//! Run: `cargo run --example bomber_18_sdpg_tournament --features sdpg_bandit,sdar_gate,ropd_rubric,g_zero,bomber`
//!
//! Output: per-matchup win rates, ELO ratings, markdown leaderboard.

use std::collections::HashMap;

use katgpt_rs::pruners::arena::types::EloCalculator;
use katgpt_rs::pruners::bomber::arena_runner::{BomberArenaConfig, run_bomber_matchup};
use katgpt_rs::pruners::bomber::{
    BomberPlayer, GreedyPlayer, HLPlayer, RandomPlayer, ValidatorPlayer,
};

#[cfg(feature = "g_zero")]
use katgpt_rs::pruners::bomber::GZeroPlayer;
#[cfg(feature = "ropd_rubric")]
use katgpt_rs::pruners::bomber::RubricPlayer;
#[cfg(feature = "sdar_gate")]
use katgpt_rs::pruners::bomber::SdarPlayer;
#[cfg(feature = "sdpg_bandit")]
use katgpt_rs::pruners::bomber::SdpgPlayer;

// ── Constants ──────────────────────────────────────────────────

/// Games per matchup.
const GAMES_PER_MATCHUP: usize = 50;

/// ELO K-factor (volatility).
const ELO_K: f64 = 24.0;

/// ELO base rating.
const ELO_BASE: f64 = 1000.0;

/// All player type names in display order.
const ALL_PLAYERS: &[&str] = &[
    "Random",
    "Greedy",
    "Validator",
    "HL",
    "GZero",
    "Rubric",
    "SDAR",
    "SDPG",
];

// ── Matchup Definition ─────────────────────────────────────────

/// A tournament matchup specification: 4 players in one heat.
struct MatchupSpec {
    /// Display label.
    label: &'static str,
    /// Player type names in slot order.
    players: [&'static str; 4],
}

/// Matchups to run in sequence.
const MATCHUPS: &[MatchupSpec] = &[
    MatchupSpec {
        label: "Baseline Hierarchy",
        players: ["Random", "Greedy", "Validator", "HL"],
    },
    MatchupSpec {
        label: "GZero Challenge",
        players: ["Random", "HL", "GZero", "Validator"],
    },
    MatchupSpec {
        label: "SDPG Challenge",
        players: ["Random", "HL", "SDPG", "Validator"],
    },
    MatchupSpec {
        label: "Championship",
        players: ["GZero", "SDAR", "SDPG", "HL"],
    },
    MatchupSpec {
        label: "SDPG vs SDAR",
        players: ["SDPG", "SDAR", "HL", "Validator"],
    },
];

// ── Player Factory ─────────────────────────────────────────────

/// Create a player by type name with the given slot ID.
fn make_player(name: &str, id: u8) -> Box<dyn BomberPlayer> {
    match name {
        "Random" => Box::new(RandomPlayer::new(id)),
        "Greedy" => Box::new(GreedyPlayer::new(id)),
        "Validator" => Box::new(ValidatorPlayer::new(id)),
        "HL" => Box::new(HLPlayer::new(id)),
        #[cfg(feature = "g_zero")]
        "GZero" => Box::new(GZeroPlayer::new(id)),
        #[cfg(feature = "ropd_rubric")]
        "Rubric" => Box::new(RubricPlayer::new(id)),
        #[cfg(feature = "sdar_gate")]
        "SDAR" => Box::new(SdarPlayer::new(id)),
        #[cfg(feature = "sdpg_bandit")]
        "SDPG" => Box::new(SdpgPlayer::new(id)),
        _ => panic!("Unknown player: {name}"),
    }
}

/// Get the emoji for a player type.
fn emoji_for(name: &str) -> &'static str {
    match name {
        "Random" => "🐰",
        "Greedy" => "🐱",
        "Validator" => "🐶",
        "HL" => "🐵",
        "GZero" => "🧠",
        "Rubric" => "📋",
        "SDAR" => "🔐",
        "SDPG" => "🎓",
        _ => "❓",
    }
}

// ── ELO Helpers ────────────────────────────────────────────────

/// Update ELO ratings after a single 4-player game.
///
/// Winner gets pairwise ELO gain vs each loser.
/// Draws result in pairwise 0.5/0.5 adjustments.
fn update_elo_after_game(
    elos: &mut HashMap<String, f64>,
    player_names: &[String],
    winner: Option<usize>,
    calc: &EloCalculator,
) {
    match winner {
        Some(w) => {
            let w_name = player_names[w].clone();
            let w_rating = elos[&w_name];
            let mut loser_deltas: Vec<(String, f64)> = Vec::with_capacity(3);
            let mut w_delta = 0.0;

            for (i, name) in player_names.iter().enumerate() {
                if i == w {
                    continue;
                }
                let l_rating = elos[name];
                let expected_w = calc.expected(w_rating, l_rating);
                w_delta += calc.k * (1.0 - expected_w);
                loser_deltas.push((name.clone(), -calc.k * expected_w));
            }

            *elos.get_mut(&w_name).unwrap() += w_delta;
            for (name, delta) in loser_deltas {
                *elos.get_mut(&name).unwrap() += delta;
            }
        }
        None => {
            // Draw: pairwise 0.5 score adjustments
            let n = player_names.len();
            let mut all_deltas = vec![0.0f64; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let ri = elos[&player_names[i]];
                    let rj = elos[&player_names[j]];
                    let expected_i = calc.expected(ri, rj);
                    let delta = calc.k * (0.5 - expected_i);
                    all_deltas[i] += delta;
                    all_deltas[j] -= delta;
                }
            }
            for (i, name) in player_names.iter().enumerate() {
                *elos.get_mut(name).unwrap() += all_deltas[i];
            }
        }
    }
}

// ── Output Formatting ─────────────────────────────────────────

/// Print the tournament header.
fn print_header() {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  Bomber SDPG Tournament (Plan 180)");
    let matchups = MATCHUPS.len();
    let games = GAMES_PER_MATCHUP;
    println!("  Players: 8 | Matchups: {matchups} | Games: {games} each");
    println!("══════════════════════════════════════════════════════════════");
}

/// Print a matchup lineup with emojis.
fn print_lineup(players: &[&str; 4]) {
    let p0 = players[0];
    let p1 = players[1];
    let p2 = players[2];
    let p3 = players[3];
    let e0 = emoji_for(p0);
    let e1 = emoji_for(p1);
    let e2 = emoji_for(p2);
    let e3 = emoji_for(p3);
    println!("  {e0} {p0}  ·  {e1} {p1}  ·  {e2} {p2}  ·  {e3} {p3}");
    let sep = "─".repeat(50);
    println!("  {sep}");
}

/// Print matchup results sorted by wins descending.
fn print_matchup_results(
    player_names: &[&str; 4],
    wins: &[usize; 4],
    total_games: usize,
    duration: std::time::Duration,
) {
    let secs = duration.as_secs_f64();
    println!("\n  Results ({total_games} games, {secs:.1}s):");

    // Sort by wins descending
    let mut indexed: Vec<(usize, usize)> = wins.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));

    let total_wins: usize = wins.iter().sum();
    let draws = total_games.saturating_sub(total_wins);

    for (idx, win_count) in indexed {
        let name = player_names[idx];
        let emoji = emoji_for(name);
        let losses = total_games.saturating_sub(win_count);
        let win_pct = win_count as f64 / total_games as f64 * 100.0;
        println!("  {emoji} {name:<10} {win_count:>3}W  {losses:>3}L  ({win_pct:>5.1}%)");
    }

    if draws > 0 {
        println!("  💀 Draws: {draws}");
    }
    println!();
}

/// Print the final leaderboard with ELO ratings and markdown output.
fn print_leaderboard(
    elos: &HashMap<String, f64>,
    win_counts: &HashMap<String, usize>,
    game_counts: &HashMap<String, usize>,
) {
    println!("══════════════════════════════════════════════════════════════");
    println!("  FINAL LEADERBOARD");
    println!("══════════════════════════════════════════════════════════════");
    println!(
        "  | {:<4} | {:<13} | {:>3} | {:>3} | {:>5} | {:>6} | {:>5} |",
        "Rank", "Player", "W", "L", "Games", "Win%", "ELO"
    );
    println!("  |------|---------------|-----|-----|-------|--------|-------|");

    // Sort by ELO descending
    let mut entries: Vec<(&String, &f64)> = elos.iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (i, (name, elo)) in entries.iter().enumerate() {
        let rank = i + 1;
        let emoji = emoji_for(name);
        let wins = win_counts[*name];
        let games = game_counts[*name];
        let losses = games.saturating_sub(wins);
        let win_pct = match games {
            0 => 0.0,
            _ => wins as f64 / games as f64 * 100.0,
        };
        let display_name = format!("{emoji} {name}");
        println!(
            "  | {rank:>4} | {display_name:<13} | {wins:>3} | {losses:>3} | {games:>5} | {win_pct:>5.1}% | {elo:>5.0} |",
        );
    }

    println!("══════════════════════════════════════════════════════════════");
    println!();

    // Markdown table for copy-paste
    println!("```markdown");
    println!("## Bomber SDPG Tournament Results");
    println!();
    println!("| Rank | Player | W | L | Games | Win% | ELO |");
    println!("|------|--------|---|---|-------|------|-----|");
    for (i, (name, elo)) in entries.iter().enumerate() {
        let rank = i + 1;
        let wins = win_counts[*name];
        let games = game_counts[*name];
        let losses = games.saturating_sub(wins);
        let win_pct = match games {
            0 => 0.0,
            _ => wins as f64 / games as f64 * 100.0,
        };
        println!("| {rank} | {name} | {wins} | {losses} | {games} | {win_pct:.1}% | {elo:.0} |",);
    }
    println!("```");
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    print_header();

    let config = BomberArenaConfig {
        games: GAMES_PER_MATCHUP,
        tick_limit: 300,
        procedural: true,
        ..Default::default()
    };

    let calc = EloCalculator {
        k: ELO_K,
        base: ELO_BASE,
    };

    let mut elos: HashMap<String, f64> = HashMap::new();
    let mut win_counts: HashMap<String, usize> = HashMap::new();
    let mut game_counts: HashMap<String, usize> = HashMap::new();

    // Initialize all player types at base ELO
    for name in ALL_PLAYERS {
        elos.insert((*name).to_string(), calc.base);
        win_counts.insert((*name).to_string(), 0);
        game_counts.insert((*name).to_string(), 0);
    }

    let total_start = std::time::Instant::now();

    for (idx, matchup) in MATCHUPS.iter().enumerate() {
        let matchup_num = idx + 1;
        let total_matchups = MATCHUPS.len();
        println!(
            "\nMatchup {matchup_num}/{total_matchups}: {}",
            matchup.label
        );
        print_lineup(&matchup.players);
        let games = config.games;
        println!("  Games: {games} | Map: procedural");

        // Create fresh players for this matchup
        let mut players: Vec<Box<dyn BomberPlayer>> = matchup
            .players
            .iter()
            .enumerate()
            .map(|(i, name)| make_player(name, i as u8))
            .collect();

        // Run matchup
        let matchup_start = std::time::Instant::now();
        let result = run_bomber_matchup(&mut players, &config);
        let matchup_duration = matchup_start.elapsed();

        // Process each game: update ELO and count wins
        let total_games = result.games.len();
        let mut matchup_wins = [0usize; 4];

        for game in &result.games {
            update_elo_after_game(&mut elos, &result.player_names, game.winner, &calc);
            if let Some(w) = game.winner {
                matchup_wins[w] += 1;
            }
        }

        // Accumulate global stats
        for (i, name) in result.player_names.iter().enumerate() {
            *game_counts.get_mut(name).unwrap() += total_games;
            *win_counts.get_mut(name).unwrap() += matchup_wins[i];
        }

        print_matchup_results(
            &matchup.players,
            &matchup_wins,
            total_games,
            matchup_duration,
        );
    }

    let total_duration = total_start.elapsed();
    let total_secs = total_duration.as_secs_f64();
    println!("Total tournament time: {total_secs:.1}s");

    print_leaderboard(&elos, &win_counts, &game_counts);
}
