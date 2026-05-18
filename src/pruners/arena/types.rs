//! Shared tournament types for cross-arena competitions.

use std::cmp::Ordering;
use std::fmt;
use std::time::Duration;

/// Which game domain the tournament runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArenaKind {
    Bomber,
    Fft,
}

impl fmt::Display for ArenaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bomber => write!(f, "Bomber"),
            Self::Fft => write!(f, "FFT"),
        }
    }
}

/// Result of a single game/round.
#[derive(Clone, Debug)]
pub struct GameResult {
    /// Player index of the winner (None for draw).
    pub winner: Option<usize>,
    /// Per-player scores for this game.
    pub scores: Vec<i32>,
    /// Number of ticks/turns played.
    pub ticks: u32,
    /// Duration of the game.
    pub duration: Duration,
}

/// Result of a matchup (N games between same player set).
#[derive(Clone, Debug)]
pub struct MatchupResult {
    /// Arena domain.
    pub arena: ArenaKind,
    /// Player names in lineup order.
    pub player_names: Vec<String>,
    /// Individual game results.
    pub games: Vec<GameResult>,
}

impl MatchupResult {
    /// Wins for player at given index across all games.
    pub fn wins_for(&self, idx: usize) -> usize {
        self.games.iter().filter(|g| g.winner == Some(idx)).count()
    }

    /// Win rate for player at given index (0.0–1.0).
    pub fn win_rate(&self, idx: usize) -> f64 {
        match self.games.is_empty() {
            true => 0.0,
            false => self.wins_for(idx) as f64 / self.games.len() as f64,
        }
    }

    /// Average game duration.
    pub fn avg_duration(&self) -> Duration {
        match self.games.is_empty() {
            true => Duration::ZERO,
            false => {
                let total: Duration = self.games.iter().map(|g| g.duration).sum();
                total / self.games.len() as u32
            }
        }
    }
}

/// Player ranking entry for leaderboard.
#[derive(Clone, Debug)]
pub struct Ranking {
    /// Player name.
    pub name: String,
    /// Arena domain.
    pub arena: ArenaKind,
    /// Total wins across all matchups.
    pub wins: usize,
    /// Total losses across all matchups.
    pub losses: usize,
    /// Total draws.
    pub draws: usize,
    /// ELO rating.
    pub elo: f64,
}

impl Ranking {
    /// Total games played.
    pub fn total(&self) -> usize {
        self.wins + self.losses + self.draws
    }

    /// Win percentage (0.0–100.0).
    pub fn win_pct(&self) -> f64 {
        match self.total() {
            0 => 0.0,
            t => self.wins as f64 / t as f64 * 100.0,
        }
    }
}

/// Aggregated leaderboard across all matchups.
#[derive(Clone, Debug, Default)]
pub struct Leaderboard {
    pub rankings: Vec<Ranking>,
}

impl Leaderboard {
    /// Sort rankings by ELO descending.
    pub fn sort(&mut self) {
        self.rankings
            .sort_by(|a, b| b.elo.partial_cmp(&a.elo).unwrap_or(Ordering::Equal));
    }

    /// Format as markdown table.
    pub fn to_markdown(&self, arena: ArenaKind) -> String {
        let mut md = format!("## {arena} Arena Leaderboard\n\n");
        md.push_str("| Rank | Player | W | L | D | Win% | ELO |\n");
        md.push_str("|------|--------|---|---|---|------|-----|\n");
        for (i, r) in self.rankings.iter().enumerate() {
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.1}% | {:.0} |\n",
                i + 1,
                r.name,
                r.wins,
                r.losses,
                r.draws,
                r.win_pct(),
                r.elo
            ));
        }
        md
    }
}

/// ELO rating calculator.
pub struct EloCalculator {
    pub k: f64,
    pub base: f64,
}

impl Default for EloCalculator {
    fn default() -> Self {
        Self {
            k: 32.0,
            base: 1000.0,
        }
    }
}

impl EloCalculator {
    /// Expected score for player A vs player B.
    pub fn expected(&self, rating_a: f64, rating_b: f64) -> f64 {
        1.0 / (1.0 + 10.0_f64.powf((rating_b - rating_a) / 400.0))
    }

    /// Update ratings after a game. Returns (new_a, new_b).
    pub fn update(&self, rating_a: f64, rating_b: f64, a_won: bool) -> (f64, f64) {
        let expected_a = self.expected(rating_a, rating_b);
        let actual_a = match a_won {
            true => 1.0,
            false => 0.0,
        };
        let actual_b = 1.0 - actual_a;

        let new_a = rating_a + self.k * (actual_a - expected_a);
        let new_b = rating_b + self.k * (actual_b - (1.0 - expected_a));
        (new_a, new_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_kind_display() {
        assert_eq!(format!("{}", ArenaKind::Bomber), "Bomber");
        assert_eq!(format!("{}", ArenaKind::Fft), "FFT");
    }

    #[test]
    fn matchup_result_wins_and_rate() {
        let mr = MatchupResult {
            arena: ArenaKind::Bomber,
            player_names: vec!["Alice".into(), "Bob".into()],
            games: vec![
                GameResult {
                    winner: Some(0),
                    scores: vec![10, 5],
                    ticks: 100,
                    duration: Duration::from_millis(50),
                },
                GameResult {
                    winner: Some(1),
                    scores: vec![3, 8],
                    ticks: 80,
                    duration: Duration::from_millis(40),
                },
                GameResult {
                    winner: Some(0),
                    scores: vec![12, 2],
                    ticks: 120,
                    duration: Duration::from_millis(60),
                },
            ],
        };
        assert_eq!(mr.wins_for(0), 2);
        assert_eq!(mr.wins_for(1), 1);
        assert!((mr.win_rate(0) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn matchup_result_empty_games() {
        let mr = MatchupResult {
            arena: ArenaKind::Fft,
            player_names: vec![],
            games: vec![],
        };
        assert_eq!(mr.wins_for(0), 0);
        assert_eq!(mr.win_rate(0), 0.0);
        assert_eq!(mr.avg_duration(), Duration::ZERO);
    }

    #[test]
    fn matchup_result_avg_duration() {
        let mr = MatchupResult {
            arena: ArenaKind::Bomber,
            player_names: vec!["A".into()],
            games: vec![
                GameResult {
                    winner: Some(0),
                    scores: vec![1],
                    ticks: 10,
                    duration: Duration::from_millis(100),
                },
                GameResult {
                    winner: None,
                    scores: vec![0],
                    ticks: 10,
                    duration: Duration::from_millis(200),
                },
            ],
        };
        assert_eq!(mr.avg_duration(), Duration::from_millis(150));
    }

    #[test]
    fn ranking_total_and_win_pct() {
        let r = Ranking {
            name: "Test".into(),
            arena: ArenaKind::Bomber,
            wins: 3,
            losses: 1,
            draws: 1,
            elo: 1050.0,
        };
        assert_eq!(r.total(), 5);
        assert!((r.win_pct() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn ranking_zero_total() {
        let r = Ranking {
            name: "Newbie".into(),
            arena: ArenaKind::Fft,
            wins: 0,
            losses: 0,
            draws: 0,
            elo: 1000.0,
        };
        assert_eq!(r.total(), 0);
        assert_eq!(r.win_pct(), 0.0);
    }

    #[test]
    fn leaderboard_sort_by_elo() {
        let mut lb = Leaderboard {
            rankings: vec![
                Ranking {
                    name: "Low".into(),
                    arena: ArenaKind::Bomber,
                    wins: 1,
                    losses: 2,
                    draws: 0,
                    elo: 900.0,
                },
                Ranking {
                    name: "High".into(),
                    arena: ArenaKind::Bomber,
                    wins: 5,
                    losses: 0,
                    draws: 0,
                    elo: 1200.0,
                },
                Ranking {
                    name: "Mid".into(),
                    arena: ArenaKind::Bomber,
                    wins: 3,
                    losses: 1,
                    draws: 0,
                    elo: 1050.0,
                },
            ],
        };
        lb.sort();
        assert_eq!(lb.rankings[0].name, "High");
        assert_eq!(lb.rankings[1].name, "Mid");
        assert_eq!(lb.rankings[2].name, "Low");
    }

    #[test]
    fn leaderboard_to_markdown() {
        let lb = Leaderboard {
            rankings: vec![Ranking {
                name: "Alpha".into(),
                arena: ArenaKind::Fft,
                wins: 2,
                losses: 0,
                draws: 0,
                elo: 1100.0,
            }],
        };
        let md = lb.to_markdown(ArenaKind::Fft);
        assert!(md.contains("FFT Arena Leaderboard"));
        assert!(md.contains("Alpha"));
        assert!(md.contains("1100"));
    }

    #[test]
    fn elo_expected_symmetry() {
        let calc = EloCalculator::default();
        let ea = calc.expected(1000.0, 1000.0);
        assert!((ea - 0.5).abs() < 1e-9);

        let ea_high = calc.expected(1200.0, 800.0);
        assert!(ea_high > 0.5);
    }

    #[test]
    fn elo_update_a_wins() {
        let calc = EloCalculator::default();
        let (new_a, new_b) = calc.update(1000.0, 1000.0, true);
        assert!(new_a > 1000.0);
        assert!(new_b < 1000.0);
        // Conservation: total change sums to zero
        assert!(((new_a - 1000.0) + (new_b - 1000.0)).abs() < 1e-9);
    }

    #[test]
    fn elo_update_b_wins() {
        let calc = EloCalculator::default();
        let (new_a, new_b) = calc.update(1000.0, 1000.0, false);
        assert!(new_a < 1000.0);
        assert!(new_b > 1000.0);
    }

    #[test]
    fn elo_upset_has_larger_change() {
        let calc = EloCalculator::default();
        let (fav_gain, _) = calc.update(1200.0, 800.0, true);
        let (_, upset_gain) = calc.update(800.0, 1200.0, true);
        // Upset winner gains more than expected winner
        assert!(upset_gain - 800.0 > fav_gain - 1200.0);
    }
}
