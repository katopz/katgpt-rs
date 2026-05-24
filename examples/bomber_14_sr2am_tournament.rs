//! Bomber SR²AM Tournament — Sr2amPlayer vs all baselines.
//!
//! Round-robin tournament pitting 6 player types in 4-player matches:
//! Random, Greedy, Validator, HL, GZero, SR²AM
//!
//! Run: `cargo run --example bomber_14_sr2am_tournament --features sr2am_configurator,g_zero,bomber`
//!
//! Output: per-matchup win rates, ELO ratings, markdown leaderboard, SR²AM decision stats.

use std::collections::HashMap;

use katgpt_rs::pruners::arena::types::EloCalculator;
use katgpt_rs::pruners::bomber::arena_runner::{BomberArenaConfig, run_bomber_matchup};
use katgpt_rs::pruners::bomber::{
    BomberPlayer, GreedyPlayer, HLPlayer, RandomPlayer, ValidatorPlayer,
};

#[cfg(feature = "g_zero")]
use katgpt_rs::pruners::bomber::GZeroPlayer;
#[cfg(feature = "sr2am_configurator")]
use katgpt_rs::pruners::bomber::Sr2amPlayer;

// ── Constants ──────────────────────────────────────────────────

const GAMES_PER_MATCHUP: usize = 50;
const ELO_K: f64 = 24.0;
const ELO_BASE: f64 = 1000.0;

const ALL_PLAYERS: &[&str] = &["Random", "Greedy", "Validator", "HL", "GZero", "SR²AM"];

// ── Matchup Definition ─────────────────────────────────────────

struct MatchupSpec {
    label: &'static str,
    players: [&'static str; 4],
}

const MATCHUPS: &[MatchupSpec] = &[
    MatchupSpec {
        label: "Baseline Hierarchy",
        players: ["Random", "HL", "GZero", "Validator"],
    },
    MatchupSpec {
        label: "SR²AM Challenge",
        players: ["Random", "HL", "GZero", "SR²AM"],
    },
    MatchupSpec {
        label: "Championship",
        players: ["Random", "HL", "SR²AM", "GZero"],
    },
];

// ── Player Factory ─────────────────────────────────────────────

fn make_player(name: &str, id: u8) -> Box<dyn BomberPlayer> {
    match name {
        "Random" => Box::new(RandomPlayer::new(id)),
        "Greedy" => Box::new(GreedyPlayer::new(id)),
        "Validator" => Box::new(ValidatorPlayer::new(id)),
        "HL" => Box::new(HLPlayer::new(id)),
        #[cfg(feature = "g_zero")]
        "GZero" => Box::new(GZeroPlayer::new(id)),
        #[cfg(feature = "sr2am_configurator")]
        "SR²AM" => Box::new(Sr2amPlayer::new(id)),
        _ => panic!("Unknown player: {name}"),
    }
}

fn emoji_for(name: &str) -> &'static str {
    match name {
        "Random" => "🐰",
        "Greedy" => "🐱",
        "Validator" => "🐶",
        "HL" => "🐵",
        "GZero" => "🧠",
        "SR²AM" => "🎯",
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
    match winner {
        Some(w) => {
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
        }
        None => {
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

fn print_header() {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  Bomber SR²AM Tournament");
    let matchups = MATCHUPS.len();
    let games = GAMES_PER_MATCHUP;
    println!("  Players: 6 | Matchups: {matchups} | Games: {games} each");
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
    indexed.sort_by(|a, b| b.1.cmp(&a.1));

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
    entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

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
    println!("## Bomber SR²AM Tournament Results\n");
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

#[cfg(feature = "sr2am_configurator")]
fn print_sr2am_stats(player: &dyn BomberPlayer, matchup_label: &str) {
    if let Some(sr2am) = player.as_any().downcast_ref::<Sr2amPlayer>() {
        let (plan_new, plan_extend, plan_skip) = sr2am.decision_stats();
        let total = plan_new + plan_extend + plan_skip;
        let pct = |v: usize| {
            if total == 0 {
                0.0
            } else {
                v as f64 / total as f64 * 100.0
            }
        };

        println!("══════════════════════════════════════════════════════════════");
        println!("  🎯 SR²AM CONFIGURATOR STATS ({matchup_label})");
        println!("══════════════════════════════════════════════════════════════");
        println!("  PlanNew:    {plan_new:>6} ({:>5.1}%)", pct(plan_new));
        println!(
            "  PlanExtend: {plan_extend:>6} ({:>5.1}%)",
            pct(plan_extend)
        );
        println!("  PlanSkip:   {plan_skip:>6} ({:>5.1}%)", pct(plan_skip));
        println!("  Total:      {total:>6}");
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
    let mut elos: HashMap<String, f64> = HashMap::new();
    let mut win_counts: HashMap<String, usize> = HashMap::new();
    let mut game_counts: HashMap<String, usize> = HashMap::new();

    for name in ALL_PLAYERS {
        elos.insert((*name).to_string(), calc.base);
        win_counts.insert((*name).to_string(), 0);
        game_counts.insert((*name).to_string(), 0);
    }

    let total_start = std::time::Instant::now();
    let mut last_sr2am_idx: Option<usize> = None;

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

        // Print SR²AM stats right after the matchup where it played
        #[cfg(feature = "sr2am_configurator")]
        if let Some(slot) = matchup.players.iter().position(|p| *p == "SR²AM") {
            print_sr2am_stats(players[slot].as_ref(), matchup.label);
            last_sr2am_idx = Some(idx);
        }
    }

    let total_secs = total_start.elapsed().as_secs_f64();
    println!("Total tournament time: {total_secs:.1}s");

    print_leaderboard(&elos, &win_counts, &game_counts);

    // SR²AM overall summary
    #[cfg(feature = "sr2am_configurator")]
    {
        let sr2am_wins = win_counts["SR²AM"];
        let sr2am_games = game_counts["SR²AM"];
        let sr2am_losses = sr2am_games.saturating_sub(sr2am_wins);
        let sr2am_pct = if sr2am_games == 0 {
            0.0
        } else {
            sr2am_wins as f64 / sr2am_games as f64 * 100.0
        };
        println!(
            "🎯 SR²AM Overall: {sr2am_wins}W {sr2am_losses}L ({sr2am_pct:.1}%) across {sr2am_games} games"
        );
        match last_sr2am_idx {
            Some(idx) => println!("   Last appeared in: {}", MATCHUPS[idx].label),
            None => println!("   (SR²AM did not participate in any matchup)"),
        }
    }
}
