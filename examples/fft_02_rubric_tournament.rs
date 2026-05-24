//! FFT Tactics Rubric Tournament — RubricFFTPlayer vs all baselines (Plan 077).
//!
//! 4v4 battles comparing party strategies:
//! Random, Greedy, Validator, HL, GZeroFFT, RubricFFT
//!
//! Each matchup: one party strategy vs one enemy strategy, N battles.
//! Round-robin: every strategy faces every other strategy.
//!
//! Run: `cargo run --example fft_02_rubric_tournament --features ropd_rubric,g_zero,fft`

use std::any::Any;
use std::collections::HashMap;

use fastrand::Rng;

use katgpt_rs::pruners::EloCalculator;
use katgpt_rs::pruners::fft::{
    Action, ActionType, BattleState, FftArenaConfig, FftPlayer, GZeroFFTPlayer, GreedyFFTPlayer,
    HLFFTPlayer, RubricFFTPlayer, ValidatorFFTPlayer, run_fft_matchup,
};

// ── Constants ──────────────────────────────────────────────────

/// Number of battles per matchup.
const GAMES_PER_MATCHUP: usize = 20;

// ── Strategy Enum ──────────────────────────────────────────────

/// All tournament strategies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Strategy {
    Random,
    Greedy,
    Validator,
    HL,
    GZero,
    Rubric,
}

impl Strategy {
    const fn label(&self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::Greedy => "Greedy",
            Self::Validator => "Validator",
            Self::HL => "HL",
            Self::GZero => "GZero",
            Self::Rubric => "Rubric",
        }
    }

    const fn emoji(&self) -> &'static str {
        match self {
            Self::Random => "🎲",
            Self::Greedy => "⚔️",
            Self::Validator => "🛡️",
            Self::HL => "📊",
            Self::GZero => "🧠",
            Self::Rubric => "📋",
        }
    }

    fn all() -> &'static [Strategy] {
        &[
            Strategy::Random,
            Strategy::Greedy,
            Strategy::Validator,
            Strategy::HL,
            Strategy::GZero,
            Strategy::Rubric,
        ]
    }
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.emoji(), self.label())
    }
}

// ── Random FFT Player (local baseline) ─────────────────────────

/// Naive random action selector — picks any valid action uniformly.
struct RandomFFTPlayer;

impl FftPlayer for RandomFFTPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        let unit = &state.units[unit_id as usize];
        let reachable = state.reachable_positions(unit_id);
        let move_to = reachable.get(rng.usize(..reachable.len().max(1))).copied();

        let enemy_team = BattleState::enemy_team(unit.team);
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);

        let mut options = vec![ActionType::Wait, ActionType::Defend];
        if !enemies.is_empty() {
            options.push(ActionType::Attack);
        }
        if !enemies.is_empty() && unit.can_afford(ActionType::BlackMagic) {
            options.push(ActionType::BlackMagic);
        }
        if !allies.is_empty() && unit.can_afford(ActionType::WhiteMagic) {
            options.push(ActionType::WhiteMagic);
        }
        if unit.can_afford(ActionType::Potion) {
            options.push(ActionType::Potion);
        }

        let action_type = options[rng.usize(..options.len())];
        let target_id = match action_type {
            ActionType::Attack | ActionType::BlackMagic => {
                enemies.get(rng.usize(..enemies.len().max(1))).copied()
            }
            ActionType::WhiteMagic => allies.get(rng.usize(..allies.len().max(1))).copied(),
            ActionType::Potion => Some(unit_id),
            _ => None,
        };

        Action {
            action_type,
            target_id,
            move_to,
        }
    }

    fn name(&self) -> &'static str {
        "Random"
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Player Factory ─────────────────────────────────────────────

/// Create 4 players (one per unit) for the given strategy.
fn make_party(strategy: Strategy) -> Vec<Box<dyn FftPlayer>> {
    match strategy {
        Strategy::Random => vec![
            Box::new(RandomFFTPlayer),
            Box::new(RandomFFTPlayer),
            Box::new(RandomFFTPlayer),
            Box::new(RandomFFTPlayer),
        ],
        Strategy::Greedy => vec![
            Box::new(GreedyFFTPlayer),
            Box::new(GreedyFFTPlayer),
            Box::new(GreedyFFTPlayer),
            Box::new(GreedyFFTPlayer),
        ],
        Strategy::Validator => vec![
            Box::new(ValidatorFFTPlayer),
            Box::new(ValidatorFFTPlayer),
            Box::new(ValidatorFFTPlayer),
            Box::new(ValidatorFFTPlayer),
        ],
        Strategy::HL => vec![
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
            Box::new(HLFFTPlayer::new()),
        ],
        Strategy::GZero => vec![
            Box::new(GZeroFFTPlayer::new(0)),
            Box::new(GZeroFFTPlayer::new(1)),
            Box::new(GZeroFFTPlayer::new(2)),
            Box::new(GZeroFFTPlayer::new(3)),
        ],
        Strategy::Rubric => vec![
            Box::new(RubricFFTPlayer::new(0)),
            Box::new(RubricFFTPlayer::new(1)),
            Box::new(RubricFFTPlayer::new(2)),
            Box::new(RubricFFTPlayer::new(3)),
        ],
    }
}

// ── Per-Strategy Stats ─────────────────────────────────────────

#[derive(Clone, Debug)]
struct StrategyStats {
    wins: usize,
    losses: usize,
    draws: usize,
    elo: f64,
}

impl StrategyStats {
    fn new() -> Self {
        Self {
            wins: 0,
            losses: 0,
            draws: 0,
            elo: 1000.0,
        }
    }

    fn total(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    fn win_pct(&self) -> f64 {
        match self.total() {
            0 => 0.0,
            t => self.wins as f64 / t as f64 * 100.0,
        }
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("═══ FFT Tactics Rubric Tournament (Plan 077) ═══\n");

    let config = FftArenaConfig {
        games: GAMES_PER_MATCHUP,
        turn_limit: 200,
    };

    let strategies = Strategy::all();
    let n = strategies.len();

    // Win rate matrix: matrix[party_idx][enemy_idx] = party wins
    let mut win_matrix: [[usize; 6]; 6] = [[0; 6]; 6];
    let mut total_matrix: [[usize; 6]; 6] = [[0; 6]; 6];

    // Per-strategy stats
    let mut stats: HashMap<Strategy, StrategyStats> = HashMap::new();
    for &s in strategies {
        stats.insert(s, StrategyStats::new());
    }

    // ELO calculator
    let elo_calc = EloCalculator::default();

    // GZero vs Rubric head-to-head tracking
    let mut gzero_party_wins_vs_rubric: usize = 0;
    let mut gzero_party_losses_vs_rubric: usize = 0;
    let mut rubric_party_wins_vs_gzero: usize = 0;
    let mut rubric_party_losses_vs_gzero: usize = 0;

    let total_matchups = n * (n - 1);
    let mut matchup_idx = 0;

    // ── Round-Robin ────────────────────────────────────────────
    for (i, &party_strat) in strategies.iter().enumerate() {
        for (j, &enemy_strat) in strategies.iter().enumerate() {
            if i == j {
                continue;
            }

            matchup_idx += 1;
            let party_label = party_strat.label();
            let enemy_label = enemy_strat.label();
            let party_emoji = party_strat.emoji();
            let enemy_emoji = enemy_strat.emoji();
            println!(
                "Matchup {matchup_idx}/{total_matchups}: {party_emoji}{party_label}(Party) vs {enemy_emoji}{enemy_label}(Enemy)",
            );

            let mut party = make_party(party_strat);
            let mut enemy = make_party(enemy_strat);

            let result = run_fft_matchup(&mut party, &mut enemy, &config);

            let party_wins = result.wins_for(0);
            let enemy_wins = result.wins_for(1);
            let draws = config.games - party_wins - enemy_wins;
            let win_rate = result.win_rate(0) * 100.0;

            // Print per-game results (10 per line)
            print!("  ");
            for (g, game) in result.games.iter().enumerate() {
                let symbol = match game.winner {
                    Some(0) => "W",
                    Some(1) => "L",
                    None => "D",
                    Some(_) => "D",
                };
                let game_num = g + 1;
                let total = config.games;
                print!("[{game_num:>2}/{total:>2}]{symbol} ");
                if game_num % 10 == 0 && game_num < config.games {
                    print!("\n  ");
                }
            }
            println!();

            println!(
                "  Result: Party {party_wins}W / Enemy {enemy_wins}L / {draws}D ({win_rate:.1}% win rate)\n",
            );

            // Update matrix
            win_matrix[i][j] = party_wins;
            total_matrix[i][j] = config.games;

            // Update per-strategy stats
            let ps = stats.get_mut(&party_strat).unwrap();
            ps.wins += party_wins;
            ps.losses += enemy_wins;
            ps.draws += draws;

            let es = stats.get_mut(&enemy_strat).unwrap();
            es.wins += enemy_wins;
            es.losses += party_wins;
            es.draws += draws;

            // Track GZero vs Rubric head-to-head
            match (party_strat, enemy_strat) {
                (Strategy::GZero, Strategy::Rubric) => {
                    gzero_party_wins_vs_rubric += party_wins;
                    gzero_party_losses_vs_rubric += enemy_wins;
                }
                (Strategy::Rubric, Strategy::GZero) => {
                    rubric_party_wins_vs_gzero += party_wins;
                    rubric_party_losses_vs_gzero += enemy_wins;
                }
                _ => {}
            }

            // Update ELO (per game for accuracy)
            for game in &result.games {
                let (elo_party, elo_enemy) = {
                    let ps = stats.get(&party_strat).unwrap();
                    let es = stats.get(&enemy_strat).unwrap();
                    (ps.elo, es.elo)
                };

                let party_won = game.winner == Some(0);
                let (new_party_elo, new_enemy_elo) =
                    elo_calc.update(elo_party, elo_enemy, party_won);

                stats.get_mut(&party_strat).unwrap().elo = new_party_elo;
                stats.get_mut(&enemy_strat).unwrap().elo = new_enemy_elo;
            }
        }
    }

    // ── Win Rate Matrix ────────────────────────────────────────
    println!("═══ Win Rate Matrix ═══\n");

    let col_width: usize = 10;

    // Header row
    print!("| {:<14}|", "Party \\ Enemy ");
    for &s in strategies {
        let label = s.label();
        print!(" {label:>width$} |", width = col_width);
    }
    println!();

    // Separator
    print!("|{0:-<15}|", "");
    for _ in strategies {
        print!("-{0:-<width$}-|", "", width = col_width);
    }
    println!();

    // Data rows
    for (i, &party_strat) in strategies.iter().enumerate() {
        let label = party_strat.label();
        print!("| {label:<14}|");
        for j in 0..n {
            let cell = match i == j {
                true => "—".to_string(),
                false => {
                    let wins = win_matrix[i][j];
                    let total = total_matrix[i][j];
                    match total {
                        0 => "—".to_string(),
                        _ => {
                            let pct = wins as f64 / total as f64 * 100.0;
                            format!("{pct:.0}%")
                        }
                    }
                }
            };
            print!(" {cell:>width$} |", width = col_width);
        }
        println!();
    }

    // ── ELO Rankings ───────────────────────────────────────────
    println!("\n═══ ELO Rankings ═══\n");

    let mut rankings: Vec<(&Strategy, &StrategyStats)> = stats.iter().collect();
    rankings.sort_by(|a, b| {
        b.1.elo
            .partial_cmp(&a.1.elo)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("| Rank | Strategy  | ELO    | W    | L    | D    | Win%   |");
    println!("|------|-----------|--------|------|------|------|--------|");
    for (rank, (strat, s)) in rankings.iter().enumerate() {
        let rank = rank + 1;
        let label = strat.label();
        let elo = s.elo;
        let wins = s.wins;
        let losses = s.losses;
        let draws = s.draws;
        let win_pct = s.win_pct();
        println!(
            "| {rank:<4} | {label:<9} | {elo:<6.0} | {wins:<4} | {losses:<4} | {draws:<4} | {win_pct:>6.1}% |",
        );
    }

    // ── GZero vs Rubric Head-to-Head ───────────────────────────
    println!("\n═══ GZero vs Rubric Head-to-Head (Plan 071 Hypothesis) ═══\n");

    let gzero_total_h2h_wins = gzero_party_wins_vs_rubric + rubric_party_losses_vs_gzero;
    let rubric_total_h2h_wins = rubric_party_wins_vs_gzero + gzero_party_losses_vs_rubric;
    let h2h_total_games = GAMES_PER_MATCHUP * 2;

    let gzero_dir_pct = match GAMES_PER_MATCHUP {
        0 => 0.0,
        _ => gzero_party_wins_vs_rubric as f64 / GAMES_PER_MATCHUP as f64 * 100.0,
    };
    let rubric_dir_pct = match GAMES_PER_MATCHUP {
        0 => 0.0,
        _ => rubric_party_wins_vs_gzero as f64 / GAMES_PER_MATCHUP as f64 * 100.0,
    };
    let gzero_h2h_pct = match h2h_total_games {
        0 => 0.0,
        _ => gzero_total_h2h_wins as f64 / h2h_total_games as f64 * 100.0,
    };
    let rubric_h2h_pct = match h2h_total_games {
        0 => 0.0,
        _ => rubric_total_h2h_wins as f64 / h2h_total_games as f64 * 100.0,
    };

    println!(
        "  🧠 GZero(Party)  vs 📋 Rubric(Enemy): {gzero_party_wins_vs_rubric}W / {gzero_party_losses_vs_rubric}L ({gzero_dir_pct:.1}%)",
    );
    println!(
        "  📋 Rubric(Party) vs 🧠 GZero(Enemy):  {rubric_party_wins_vs_gzero}W / {rubric_party_losses_vs_gzero}L ({rubric_dir_pct:.1}%)",
    );
    println!("  Combined: {h2h_total_games} games");
    println!("    🧠 GZero wins:  {gzero_total_h2h_wins} ({gzero_h2h_pct:.1}%)");
    println!("    📋 Rubric wins: {rubric_total_h2h_wins} ({rubric_h2h_pct:.1}%)");

    let verdict = match gzero_total_h2h_wins.cmp(&rubric_total_h2h_wins) {
        std::cmp::Ordering::Less => "📋 Rubric > 🧠 GZero ✓ (Plan 071 hypothesis confirmed!)",
        std::cmp::Ordering::Greater => "🧠 GZero > 📋 Rubric ✗ (Plan 071 hypothesis not confirmed)",
        std::cmp::Ordering::Equal => "Tie — inconclusive",
    };
    println!("\n  Result: {verdict}");

    // ── Summary ────────────────────────────────────────────────
    println!("\n═══ Tournament Complete ═══");
    let total_battles = total_matchups * GAMES_PER_MATCHUP;
    println!(
        "  {total_matchups} matchups × {GAMES_PER_MATCHUP} games = {total_battles} total battles"
    );

    if let Some((best_strat, best_stats)) = rankings.first() {
        let elo = best_stats.elo;
        let win_pct = best_stats.win_pct();
        println!("  Champion: {best_strat} (ELO {elo:.0}, Win% {win_pct:.1}%)");
    }
}
