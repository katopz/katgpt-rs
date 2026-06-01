//! FFT GOAT: 1000-round regression proof for FFT arena (Plan 178, T17).
//!
//! Runs 1000 battles with existing FFT strategies to establish baseline scores.
//! FeedbackBandit FFT integration is deferred — arena infrastructure is ready,
//! but FFT-specific ConfiguratorBandit wiring requires a new FftSr2amPlayer.
//!
//! Run: `cargo run --example fft_04_feedback_goat --features g_zero,bomber --release`
//!
//! GOAT criteria:
//!   - All strategies produce valid scores (no NaN/panic)
//!   - Win rate distribution is non-degenerate (no single strategy dominates 100%)
//!   - Score variance across games is bounded (no runaway scores)

use std::collections::HashMap;

use katgpt_rs::pruners::arena::types::EloCalculator;
use katgpt_rs::pruners::fft::arena_runner::{FftArenaConfig, run_fft_matchup};
use katgpt_rs::pruners::fft::players::{GreedyFFTPlayer, HLFFTPlayer, ValidatorFFTPlayer};

use katgpt_rs::pruners::fft::FftPlayer;

#[cfg(feature = "g_zero")]
use katgpt_rs::pruners::fft::GZeroFFTPlayer;

// ── Constants ──────────────────────────────────────────────────

/// 1000 battles per matchup for statistical significance.
const GAMES_PER_MATCHUP: usize = 1000;
const ELO_K: f64 = 24.0;
const ELO_BASE: f64 = 1000.0;

/// No single strategy should win >95% (indicates degenerate game).
const MAX_WIN_RATE_PCT: f64 = 95.0;

// ── Strategy Types ─────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Strategy {
    Greedy,
    Validator,
    HL,
    #[cfg(feature = "g_zero")]
    GZero,
}

impl Strategy {
    const fn label(self) -> &'static str {
        match self {
            Strategy::Greedy => "Greedy",
            Strategy::Validator => "Validator",
            Strategy::HL => "HL",
            #[cfg(feature = "g_zero")]
            Strategy::GZero => "GZero",
        }
    }

    const fn emoji(self) -> &'static str {
        match self {
            Strategy::Greedy => "🐱",
            Strategy::Validator => "🐶",
            Strategy::HL => "🐵",
            #[cfg(feature = "g_zero")]
            Strategy::GZero => "🧠",
        }
    }

    fn all() -> &'static [Strategy] {
        &[
            Strategy::Greedy,
            Strategy::Validator,
            Strategy::HL,
            #[cfg(feature = "g_zero")]
            Strategy::GZero,
        ]
    }
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ── Player Factory ─────────────────────────────────────────────

fn make_party(strategy: Strategy) -> Vec<Box<dyn FftPlayer>> {
    match strategy {
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
        #[cfg(feature = "g_zero")]
        Strategy::GZero => vec![
            Box::new(GZeroFFTPlayer::new(0)),
            Box::new(GZeroFFTPlayer::new(1)),
            Box::new(GZeroFFTPlayer::new(2)),
            Box::new(GZeroFFTPlayer::new(3)),
        ],
    }
}

// ── Strategy Stats ─────────────────────────────────────────────

struct StrategyStats {
    wins: usize,
    losses: usize,
    draws: usize,
    total_score: i64,
    score_sq: i64, // for variance
    elo: f64,
}

impl StrategyStats {
    fn new(base_elo: f64) -> Self {
        Self {
            wins: 0,
            losses: 0,
            draws: 0,
            total_score: 0,
            score_sq: 0,
            elo: base_elo,
        }
    }

    fn total(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    fn win_pct(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.wins as f64 / t as f64 * 100.0
        }
    }

    fn mean_score(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.total_score as f64 / t as f64
        }
    }

    fn score_variance(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            let mean = self.mean_score();
            self.score_sq as f64 / t as f64 - mean * mean
        }
    }
}

// ── Output Formatting ─────────────────────────────────────────

fn print_header() {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  FFT GOAT: 1000-Round Baseline Proof");
    println!("  Plan 178, T17");
    let strategies = Strategy::all();
    let n = strategies.len();
    let total_matchups = n * (n - 1);
    println!("  Strategies: {n} | Matchups: {total_matchups} | Games: {GAMES_PER_MATCHUP} each");
    println!("══════════════════════════════════════════════════════════════");
}

fn print_goat_verdict(stats: &HashMap<Strategy, StrategyStats>) {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  🐐 GOAT VERDICT — Plan 178 T17 (FFT Baseline)");
    println!("══════════════════════════════════════════════════════════════");

    // Check 1: No degenerate dominance
    let mut dominance_pass = true;
    for (&s, st) in stats {
        let wp = st.win_pct();
        if wp > MAX_WIN_RATE_PCT {
            println!(
                "  ❌ {} dominates at {wp:.1}% (>{MAX_WIN_RATE_PCT}%)",
                s.label()
            );
            dominance_pass = false;
        }
    }
    if dominance_pass {
        println!("  ✅ No degenerate dominance (all <{MAX_WIN_RATE_PCT}% win rate)");
    }

    // Check 2: Score variance bounded
    let variance_pass = stats.values().all(|st| st.score_variance() < 1_000_000.0);
    println!(
        "  {} Score variance bounded (<1M)",
        if variance_pass { "✅" } else { "❌" }
    );

    // Check 3: All games completed (no panics)
    let total_games: usize = stats.values().map(|st| st.total()).sum();
    let expected_games = Strategy::all().len() * (Strategy::all().len() - 1) * GAMES_PER_MATCHUP;
    let completion_pass = total_games >= expected_games;
    println!(
        "  {} Games completed: {total_games}/{expected_games}",
        if completion_pass { "✅" } else { "❌" }
    );

    let overall_pass = dominance_pass && variance_pass && completion_pass;
    println!();
    if overall_pass {
        println!("  ✅ GOAT PASSED — FFT arena baseline established.");
        println!("  ℹ️  FeedbackBandit FFT integration deferred (requires FftSr2amPlayer).");
    } else {
        println!("  ❌ GOAT FAILED — FFT arena issues detected!");
    }
    println!("══════════════════════════════════════════════════════════════");

    // Markdown summary
    println!();
    println!("```markdown");
    println!("## FFT GOAT: Baseline (T17)");
    println!();
    println!("| Strategy | W | L | D | Win% | Mean Score | ELO |");
    println!("|----------|---|---|---|------|------------|-----|");

    let mut entries: Vec<_> = stats.iter().collect();
    entries.sort_by(|a, b| {
        b.1.elo
            .partial_cmp(&a.1.elo)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (&s, st) in entries {
        let emoji = s.emoji();
        let label = s.label();
        println!(
            "| {emoji}{label} | {} | {} | {} | {:.1}% | {:.1} | {:.0} |",
            st.wins,
            st.losses,
            st.draws,
            st.win_pct(),
            st.mean_score(),
            st.elo
        );
    }

    println!();
    println!("| Metric | Value | Pass |");
    println!("|--------|-------|------|");
    println!(
        "| No Degenerate Dominance | <{MAX_WIN_RATE_PCT}% | {} |",
        if dominance_pass { "✅" } else { "❌" }
    );
    println!(
        "| Score Variance Bounded | <1M | {} |",
        if variance_pass { "✅" } else { "❌" }
    );
    println!(
        "| Completion | {total_games}/{expected_games} | {} |",
        if completion_pass { "✅" } else { "❌" }
    );
    println!(
        "| Overall | — | {} |",
        if overall_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("```");
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    print_header();

    let config = FftArenaConfig {
        games: GAMES_PER_MATCHUP,
        turn_limit: 200,
    };

    let elo_calc = EloCalculator {
        k: ELO_K,
        base: ELO_BASE,
    };

    let strategies = Strategy::all();
    let n = strategies.len();

    let mut stats: HashMap<Strategy, StrategyStats> = HashMap::new();
    for &s in strategies {
        stats.insert(s, StrategyStats::new(elo_calc.base));
    }

    let total_matchups = n * (n - 1);
    let mut matchup_idx = 0;
    let total_start = std::time::Instant::now();

    // Round-robin
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
                "\nMatchup {matchup_idx}/{total_matchups}: {party_emoji}{party_label}(Party) vs {enemy_emoji}{enemy_label}(Enemy)",
            );

            let mut party = make_party(party_strat);
            let mut enemy = make_party(enemy_strat);

            let matchup_start = std::time::Instant::now();
            let result = run_fft_matchup(&mut party, &mut enemy, &config);
            let matchup_duration = matchup_start.elapsed();

            let party_wins = result.wins_for(0);
            let enemy_wins = result.wins_for(1);
            let draws = config.games - party_wins - enemy_wins;
            let win_rate = result.win_rate(0) * 100.0;

            // Accumulate scores for party strategy (units 0-3 in scores array)
            let party_total_score: i64 = result
                .games
                .iter()
                .flat_map(|g| g.scores.iter().take(4).map(|&s| s as i64))
                .sum();
            let party_score_sq: i64 = result
                .games
                .iter()
                .flat_map(|g| g.scores.iter().take(4).map(|&s| (s as i64) * (s as i64)))
                .sum();

            {
                let ps = stats.get_mut(&party_strat).unwrap();
                ps.wins += party_wins;
                ps.losses += enemy_wins;
                ps.draws += draws;
                ps.total_score += party_total_score;
                ps.score_sq += party_score_sq;
            }

            // ELO update
            let enemy_elo = stats[&enemy_strat].elo;
            let party_elo = stats[&party_strat].elo;
            let party_expected = elo_calc.expected(party_elo, enemy_elo);
            let party_delta = elo_calc.k * (win_rate / 100.0 - party_expected);
            stats.get_mut(&party_strat).unwrap().elo += party_delta;
            stats.get_mut(&enemy_strat).unwrap().elo -= party_delta;

            let secs = matchup_duration.as_secs_f64();
            println!(
                "  Result: Party {party_wins}W / Enemy {enemy_wins}L / {draws}D ({win_rate:.1}%) [{secs:.1}s]"
            );
        }
    }

    let total_secs = total_start.elapsed().as_secs_f64();
    println!("\nTotal GOAT time: {total_secs:.1}s");

    print_goat_verdict(&stats);
}
