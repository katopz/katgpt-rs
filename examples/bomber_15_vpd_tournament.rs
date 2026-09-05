//! Bomber VPD Tournament — VpdPlayer vs SDAR vs GZero vs Random (Plan 120).
//!
//! Round-robin tournament pitting 4 player types in 4-player matches:
//! Random, GZero, SDAR, VPD
//!
//! Demonstrates VPD EM-style co-evolutionary distillation vs baselines.
//! VPD actively trains the feedback-conditioned teacher (E-step) then distills
//! back to the student (M-step), outperforming passive SDAR gating.
//!
//! Run: `cargo run --example bomber_15_vpd_tournament --features vpd_em_distill,g_zero,bomber`
//!
//! Output: per-matchup win rates, ELO ratings, markdown leaderboard, VPD EM cycle stats.

use std::collections::HashMap;

use katgpt_rs::pruners::arena::types::EloCalculator;
use katgpt_rs::pruners::bomber::arena_runner::{BomberArenaConfig, run_bomber_matchup};
use katgpt_rs::pruners::bomber::{BomberPlayer, RandomPlayer};

#[cfg(feature = "g_zero")]
use katgpt_rs::pruners::bomber::GZeroPlayer;
#[cfg(feature = "sdar_gate")]
use katgpt_rs::pruners::bomber::SdarPlayer;
#[cfg(feature = "vpd_em_distill")]
use katgpt_rs::pruners::bomber::VpdPlayer;

// ── Constants ──────────────────────────────────────────────────

/// Games per matchup (100 for example speed; GOAT proofs use 1000).
const GAMES_PER_MATCHUP: usize = 100;

/// ELO K-factor (volatility).
const ELO_K: f64 = 24.0;

/// ELO base rating.
const ELO_BASE: f64 = 1000.0;

/// All player type names in display order.
const ALL_PLAYERS: &[&str] = &["Random", "GZero", "SDAR", "VPD"];

// ── Matchup Definition ─────────────────────────────────────────

/// A tournament matchup specification: 4 players in one heat.
struct MatchupSpec {
    label: &'static str,
    players: [&'static str; 4],
}

const MATCHUPS: &[MatchupSpec] = &[
    MatchupSpec {
        label: "Baseline Hierarchy",
        players: ["Random", "GZero", "SDAR", "VPD"],
    },
    MatchupSpec {
        label: "VPD vs All",
        players: ["Random", "SDAR", "GZero", "VPD"],
    },
    MatchupSpec {
        label: "Championship",
        players: ["GZero", "SDAR", "VPD", "Random"],
    },
];

// ── Player Factory ─────────────────────────────────────────────

fn make_player(name: &str, id: u8) -> Box<dyn BomberPlayer> {
    match name {
        "Random" => Box::new(RandomPlayer::new(id)),
        #[cfg(feature = "g_zero")]
        "GZero" => Box::new(GZeroPlayer::new(id)),
        #[cfg(feature = "sdar_gate")]
        "SDAR" => Box::new(SdarPlayer::new(id)),
        #[cfg(feature = "vpd_em_distill")]
        "VPD" => Box::new(VpdPlayer::new(id)),
        _ => panic!("Unknown player: {name}"),
    }
}

fn emoji_for(name: &str) -> &'static str {
    match name {
        "Random" => "🎲",
        "GZero" => "🧠",
        "SDAR" => "🔀",
        "VPD" => "🧬",
        _ => "❓",
    }
}

// ── ELO Helpers ────────────────────────────────────────────────

fn update_elo_after_game(
    elos: &mut HashMap<String, f64>,
    player_names: &[String],
    winner: Option<usize>,
    calc: &EloCalculator,
) {
    if let Some(w) = winner {
        let w_name = player_names[w].clone();
        let w_rating = elos[&w_name];
        let mut w_delta = 0.0;
        let mut loser_deltas: Vec<(String, f64)> = Vec::with_capacity(3);

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
    } else {
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

// ── Output Formatting ─────────────────────────────────────────

fn print_header() {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  Bomber VPD EM-Style Tournament (Plan 120)");
    let matchups = MATCHUPS.len();
    let games = GAMES_PER_MATCHUP;
    println!("  Players: 4 | Matchups: {matchups} | Games: {games} each");
    println!("  VPD: E-step every F=5, BCO β=0.1, KL λ=0.1, dynamic prior");
    println!("══════════════════════════════════════════════════════════════");
}

fn print_lineup(players: &[&str; 4]) {
    let e = [
        emoji_for(players[0]),
        emoji_for(players[1]),
        emoji_for(players[2]),
        emoji_for(players[3]),
    ];
    println!(
        "  {} {}  ·  {} {}  ·  {} {}  ·  {} {}",
        e[0], players[0], e[1], players[1], e[2], players[2], e[3], players[3]
    );
    println!("  {}", "─".repeat(50));
}

fn print_matchup_results(
    player_names: &[&str; 4],
    wins: &[usize; 4],
    total_games: usize,
    duration: std::time::Duration,
) {
    println!(
        "\n  Results ({total_games} games, {:.1}s):",
        duration.as_secs_f64()
    );

    let mut indexed: Vec<(usize, usize)> = wins.iter().copied().enumerate().collect();
    indexed.sort_by_key(|a| std::cmp::Reverse(a.1));

    for (idx, win_count) in indexed {
        let name = player_names[idx];
        let emoji = emoji_for(name);
        let losses = total_games.saturating_sub(win_count);
        let win_pct = win_count as f64 / total_games as f64 * 100.0;
        println!("  {emoji} {name:<10} {win_count:>3}W  {losses:>3}L  ({win_pct:>5.1}%)");
    }

    let draws = total_games.saturating_sub(wins.iter().sum::<usize>());
    if draws > 0 {
        println!("  💀 Draws: {draws}");
    }
    println!();
}

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

    let mut entries: Vec<(&String, &f64)> = elos.iter().collect();
    entries.sort_by(|a, b| b.1.total_cmp(a.1));

    for (i, (name, elo)) in entries.iter().enumerate() {
        let rank = i + 1;
        let emoji = emoji_for(name);
        let wins = win_counts[*name];
        let games = game_counts[*name];
        let losses = games.saturating_sub(wins);
        let win_pct = if games == 0 {
            0.0
        } else {
            wins as f64 / games as f64 * 100.0
        };
        let display_name = format!("{emoji} {name}");
        println!(
            "  | {rank:>4} | {display_name:<13} | {wins:>3} | {losses:>3} | {games:>5} | {win_pct:>5.1}% | {elo:>5.0} |"
        );
    }

    println!("══════════════════════════════════════════════════════════════\n");

    // Markdown table
    println!("```markdown");
    println!("## Bomber VPD Tournament Results\n");
    println!("| Rank | Player | W | L | Games | Win% | ELO |");
    println!("|------|--------|---|---|-------|------|-----|");
    for (i, (name, elo)) in entries.iter().enumerate() {
        let rank = i + 1;
        let wins = win_counts[*name];
        let games = game_counts[*name];
        let losses = games.saturating_sub(wins);
        let win_pct = if games == 0 {
            0.0
        } else {
            wins as f64 / games as f64 * 100.0
        };
        println!("| {rank} | {name} | {wins} | {losses} | {games} | {win_pct:.1}% | {elo:.0} |");
    }
    println!("```");
}

#[cfg(feature = "vpd_em_distill")]
fn print_vpd_stats(player: &dyn BomberPlayer, matchup_label: &str) {
    if let Some(vpd) = player.as_any().downcast_ref::<VpdPlayer>() {
        let (m_steps, e_steps, reward_shift, best_template) = vpd.vpd_summary();

        println!("══════════════════════════════════════════════════════════════");
        println!("  🧬 VPD EM CYCLE STATS ({matchup_label})");
        println!("══════════════════════════════════════════════════════════════");
        println!("  M-steps:       {m_steps:>6} (KL-gated distillation)");
        println!("  E-steps:       {e_steps:>6} (BCO teacher refinement)");
        println!("  Reward shift:  {reward_shift:>9.4} (δ centering)");
        println!("  Best template: {best_template:?}");
        println!("══════════════════════════════════════════════════════════════\n");
    }
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
    let mut elos: HashMap<String, f64> = HashMap::with_capacity(ALL_PLAYERS.len());
    let mut win_counts: HashMap<String, usize> = HashMap::with_capacity(ALL_PLAYERS.len());
    let mut game_counts: HashMap<String, usize> = HashMap::with_capacity(ALL_PLAYERS.len());

    for name in ALL_PLAYERS {
        elos.insert((*name).to_string(), calc.base);
        win_counts.insert((*name).to_string(), 0);
        game_counts.insert((*name).to_string(), 0);
    }

    let total_start = std::time::Instant::now();
    let mut last_vpd_idx: Option<usize> = None;

    for (idx, matchup) in MATCHUPS.iter().enumerate() {
        let matchup_num = idx + 1;
        let total_matchups = MATCHUPS.len();
        println!(
            "\nMatchup {matchup_num}/{total_matchups}: {}",
            matchup.label
        );
        print_lineup(&matchup.players);
        println!("  Games: {} | Map: procedural", config.games);

        let mut players: Vec<Box<dyn BomberPlayer>> = matchup
            .players
            .iter()
            .enumerate()
            .map(|(i, name)| make_player(name, i as u8))
            .collect();

        let matchup_start = std::time::Instant::now();
        let result = run_bomber_matchup(&mut players, &config);
        let matchup_duration = matchup_start.elapsed();

        let total_games = result.games.len();
        let mut matchup_wins = [0usize; 4];

        for game in &result.games {
            update_elo_after_game(&mut elos, &result.player_names, game.winner, &calc);
            if let Some(w) = game.winner {
                matchup_wins[w] += 1;
            }
        }

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

        // Print VPD stats right after the matchup where it played
        #[cfg(feature = "vpd_em_distill")]
        if let Some(slot) = matchup.players.iter().position(|p| *p == "VPD") {
            print_vpd_stats(players[slot].as_ref(), matchup.label);
            last_vpd_idx = Some(idx);
        }
    }

    let total_secs = total_start.elapsed().as_secs_f64();
    println!("Total tournament time: {total_secs:.1}s");

    print_leaderboard(&elos, &win_counts, &game_counts);

    // VPD overall summary
    #[cfg(feature = "vpd_em_distill")]
    {
        let vpd_wins = win_counts["VPD"];
        let vpd_games = game_counts["VPD"];
        let vpd_losses = vpd_games.saturating_sub(vpd_wins);
        let vpd_pct = if vpd_games == 0 {
            0.0
        } else {
            vpd_wins as f64 / vpd_games as f64 * 100.0
        };

        // Compare with SDAR if available
        #[cfg(feature = "sdar_gate")]
        {
            let sdar_wins = win_counts["SDAR"];
            let sdar_games = game_counts["SDAR"];
            let sdar_pct = if sdar_games == 0 {
                0.0
            } else {
                sdar_wins as f64 / sdar_games as f64 * 100.0
            };
            let delta = vpd_pct - sdar_pct;
            let comparison = if delta >= 0.0 { "≥" } else { "<" };
            println!(
                "🧬 VPD vs 🔀 SDAR: {vpd_pct:.1}% {comparison} {sdar_pct:.1}% (Δ = {delta:+.1}%)"
            );
        }

        println!(
            "🧬 VPD Overall: {vpd_wins}W {vpd_losses}L ({vpd_pct:.1}%) across {vpd_games} games"
        );
        match last_vpd_idx {
            Some(idx) => println!("   Last appeared in: {}", MATCHUPS[idx].label),
            None => println!("   (VPD did not participate in any matchup)"),
        }
    }
}
