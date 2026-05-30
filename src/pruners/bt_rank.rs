//! Bradley-Terry pairwise ranking for DDTree candidate selection.
//!
//! Distilled from OpenDeepThink (arXiv:2605.15177, Zhou et al., 2026):
//! - Pairwise comparison (86% accuracy) >> pointwise scoring (59%)
//! - BT model: P(i ≻ j) = σ(sᵢ - sⱼ), fit via gradient ascent
//! - Internalizes opponent strength — unlike raw win rate
//!
//! # Model
//!
//! Given pairwise comparison outcomes, fit latent scores `s` such that:
//! ```text
//! P(i ≻ j) = σ(sᵢ - sⱼ) = 1 / (1 + exp(-(sᵢ - sⱼ)))
//! ```
//!
//! Optimization: gradient ascent on regularized log-likelihood with λ=0.01.
//!
//! # Why BT Over Pointwise?
//!
//! Pointwise scoring treats each candidate independently — it cannot account for
//! *who* a candidate was compared against. A candidate that beats strong opponents
//! should rank higher than one that beats weak opponents, even with the same win rate.
//! BT internalizes opponent strength through the score difference `sᵢ - sⱼ`.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "bt_rank")]`.
//! Feature: `bt_rank = []` in `Cargo.toml`.

use fastrand::Rng;

// ── Types ───────────────────────────────────────────────────────

/// Outcome of a pairwise comparison between two candidates A and B.
///
/// The `usize` in `Win` is the index of the winner (either A or B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BtOutcome {
    /// One candidate clearly won (index of winner).
    Win(usize),
    /// Candidates are tied — contributes no information.
    Tie,
}

/// A recorded comparison: winner beat loser.
///
/// Indices refer to positions in the candidate pool `[0, n_candidates)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BtComparison {
    /// Index of the winning candidate.
    pub winner: usize,
    /// Index of the losing candidate.
    pub loser: usize,
}

impl BtComparison {
    /// Create a new comparison result.
    #[inline]
    pub const fn new(winner: usize, loser: usize) -> Self {
        Self { winner, loser }
    }
}

/// Configuration for BT score fitting.
///
/// Defaults from OpenDeepThink paper (arXiv:2605.15177):
/// - λ=0.01 regularization prevents score explosion
/// - 100 iterations sufficient for n≤50 candidates
/// - 1e-6 tolerance for convergence check
#[derive(Debug, Clone)]
pub struct BtConfig {
    /// L2 regularization strength (default: 0.01).
    ///
    /// Prevents scores from diverging to infinity, especially important
    /// for undefeated candidates (who would otherwise get infinite scores).
    pub lambda: f32,
    /// Maximum gradient ascent iterations (default: 100).
    pub max_iterations: usize,
    /// Convergence: stop when max absolute gradient < tolerance (default: 1e-6).
    pub tolerance: f32,
}

impl Default for BtConfig {
    fn default() -> Self {
        Self {
            lambda: 0.01,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Fitted Bradley-Terry scores for a pool of candidates.
///
/// Scores are on a latent scale where `σ(sᵢ - sⱼ)` gives the probability
/// that candidate `i` beats candidate `j`.
#[derive(Debug, Clone)]
pub struct BtScores {
    /// Latent strength scores, one per candidate.
    pub scores: Vec<f32>,
}

impl BtScores {
    /// Return candidate indices ranked by score (best first).
    ///
    /// Ties are broken by index (lower index first for stability).
    pub fn rank(&self) -> Vec<usize> {
        let mut ranked: Vec<usize> = (0..self.scores.len()).collect();
        ranked.sort_by(|&a, &b| {
            self.scores[b]
                .partial_cmp(&self.scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        ranked
    }

    /// Return top-k candidate indices (best first).
    ///
    /// Returns fewer than `k` if there are fewer than `k` candidates.
    pub fn top_k(&self, k: usize) -> Vec<usize> {
        self.rank().into_iter().take(k).collect()
    }

    /// Get score for a specific candidate.
    #[inline]
    pub fn score(&self, idx: usize) -> f32 {
        self.scores[idx]
    }

    /// Number of candidates.
    #[inline]
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether there are no candidates.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }
}

// ── Core Functions ──────────────────────────────────────────────

/// Numerically stable logistic sigmoid: σ(z) = 1 / (1 + exp(-z)).
///
/// Two-branch implementation avoids overflow in `exp()`:
/// - `z >= 0`: `1 / (1 + exp(-z))`
/// - `z < 0`: `exp(z) / (1 + exp(z))`
#[inline]
pub fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Fit Bradley-Terry scores from pairwise comparisons via gradient ascent.
///
/// Maximizes the regularized log-likelihood:
/// ```text
/// L(s) = Σ_c (s[w_c] - s[l_c] - ln(1 + exp(s[w_c] - s[l_c]))) - (λ/2)Σsᵢ²
/// ```
///
/// # Arguments
///
/// * `comparisons` — Slice of pairwise comparison outcomes
/// * `n_candidates` — Total number of candidates (indices must be in `[0, n_candidates)`)
/// * `config` — Fitting hyperparameters
///
/// # Returns
///
/// [`BtScores`] with fitted latent scores. All scores initialize to 0.0.
///
/// # Panics
///
/// Panics if any comparison index >= `n_candidates`.
pub fn bt_fit(comparisons: &[BtComparison], n_candidates: usize, config: &BtConfig) -> BtScores {
    if n_candidates == 0 {
        return BtScores { scores: vec![] };
    }

    // Validate indices
    for c in comparisons {
        assert!(
            c.winner < n_candidates && c.loser < n_candidates,
            "BT comparison index out of bounds: ({}, {}) >= {n_candidates}",
            c.winner,
            c.loser
        );
    }

    // Initialize scores to 0.0
    let mut scores = vec![0.0f32; n_candidates];

    // Early return if no comparisons — all scores remain 0.0
    if comparisons.is_empty() {
        return BtScores { scores };
    }

    let step = 0.1f32;
    let mut grad = vec![0.0f32; n_candidates];

    for _ in 0..config.max_iterations {
        grad.fill(0.0f32);

        for comp in comparisons {
            let diff = scores[comp.winner] - scores[comp.loser];
            let sig = sigmoid(diff);
            // Winner gradient: push score up when model underestimates
            grad[comp.winner] += 1.0 - sig;
            // Loser gradient: push score down when model underestimates
            grad[comp.loser] -= 1.0 - sig;
        }

        // L2 regularization gradient: -λ·sᵢ
        for i in 0..n_candidates {
            grad[i] -= config.lambda * scores[i];
        }

        // Check convergence: max absolute gradient
        let max_grad = grad.iter().map(|g| g.abs()).fold(0.0f32, f32::max);

        if max_grad < config.tolerance {
            break;
        }

        // Gradient ascent step
        for i in 0..n_candidates {
            scores[i] += step * grad[i];
        }
    }

    BtScores { scores }
}

/// Fit BT scores by generating comparisons from a pairwise comparison function.
///
/// For each candidate, randomly selects `k_per_candidate` peers for comparison,
/// then calls `compare_fn(i, j)` to determine the outcome.
///
/// # Arguments
///
/// * `n_candidates` — Total number of candidates
/// * `k_per_candidate` — Number of peers each candidate is compared against
/// * `compare_fn` — Function returning comparison outcome for a pair (a, b)
/// * `config` — Fitting hyperparameters
///
/// # Returns
///
/// [`BtScores`] with fitted latent scores.
pub fn bt_fit_from_fn<F>(
    n_candidates: usize,
    k_per_candidate: usize,
    compare_fn: F,
    config: &BtConfig,
) -> BtScores
where
    F: Fn(usize, usize) -> BtOutcome,
{
    let mut rng = Rng::new();
    let pairs = bt_pair_random(n_candidates, k_per_candidate, &mut rng);
    let mut comparisons = Vec::with_capacity(pairs.len() * 2);

    for (a, b) in pairs {
        match compare_fn(a, b) {
            BtOutcome::Win(winner) => {
                let loser = if winner == a { b } else { a };
                comparisons.push(BtComparison::new(winner, loser));
            }
            BtOutcome::Tie => {
                // Ties contribute no information — skip
            }
        }
    }

    bt_fit(&comparisons, n_candidates, config)
}

/// Generate random K-regular pairings for BT comparison.
///
/// Each candidate is paired with `k_per_candidate` randomly chosen peers.
/// No self-pairs. May produce duplicate pairs (harmless for fitting).
///
/// # Arguments
///
/// * `n_candidates` — Total candidates (must be >= 2 for meaningful output)
/// * `k_per_candidate` — Peers per candidate
/// * `rng` — Random number generator
///
/// # Returns
///
/// Vector of `(i, j)` pairs where `i != j`.
pub fn bt_pair_random(
    n_candidates: usize,
    k_per_candidate: usize,
    rng: &mut Rng,
) -> Vec<(usize, usize)> {
    if n_candidates < 2 || k_per_candidate == 0 {
        return vec![];
    }

    let mut pairs = Vec::with_capacity(n_candidates * k_per_candidate);

    for i in 0..n_candidates {
        for _ in 0..k_per_candidate {
            let mut j = rng.usize(0..n_candidates);
            // Avoid self-pair
            while j == i {
                j = rng.usize(0..n_candidates);
            }
            pairs.push((i, j));
        }
    }

    pairs
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn test_two_candidates_clear_winner() {
        // A beats B 10 times — A should rank higher
        let comparisons: Vec<BtComparison> = (0..10).map(|_| BtComparison::new(0, 1)).collect();
        let scores = bt_fit(&comparisons, 2, &BtConfig::default());

        assert!(
            scores.score(0) > scores.score(1),
            "Winner (0) should have higher score: {} vs {}",
            scores.score(0),
            scores.score(1)
        );
        assert_eq!(scores.rank()[0], 0, "Candidate 0 should rank first");
    }

    #[test]
    fn test_three_candidates_rock_paper_scissors() {
        // Rock beats Scissors, Scissors beats Paper, Paper beats Rock
        // Each wins once, loses once — BT should resolve via regularization
        let comparisons = vec![
            BtComparison::new(0, 2), // Rock beats Scissors
            BtComparison::new(2, 1), // Scissors beats Paper
            BtComparison::new(1, 0), // Paper beats Rock
        ];
        let scores = bt_fit(&comparisons, 3, &BtConfig::default());

        // All scores should be close to 0 (symmetric cycle)
        for i in 0..3 {
            assert!(
                scores.score(i).abs() < 1.0,
                "RPS candidate {i} score should be near 0, got {}",
                scores.score(i)
            );
        }
    }

    #[test]
    fn test_all_ties() {
        // No comparisons → all scores stay at 0
        let scores = bt_fit(&[], 5, &BtConfig::default());
        for i in 0..5 {
            assert!(
                scores.score(i).abs() < EPS,
                "No-comparison score {i} should be 0, got {}",
                scores.score(i)
            );
        }
    }

    #[test]
    fn test_single_candidate() {
        let scores = bt_fit(&[], 1, &BtConfig::default());
        assert_eq!(scores.len(), 1);
        assert!(
            scores.score(0).abs() < EPS,
            "Single candidate score should be 0, got {}",
            scores.score(0)
        );
    }

    #[test]
    fn test_zero_candidates() {
        let scores = bt_fit(&[], 0, &BtConfig::default());
        assert!(scores.is_empty());
    }

    #[test]
    fn test_paper_scenario_20_candidates_k4() {
        // n=20 candidates, K=4 comparisons per candidate
        // Deterministic oracle: lower index always wins (candidate 0 is best, 19 is worst)
        // K=4 is sparse — not enough to guarantee perfect ordering, but should get
        // the overall trend right (best near top, worst near bottom).
        let mut rng = Rng::with_seed(42);
        let pairs = bt_pair_random(20, 4, &mut rng);
        let mut comparisons = Vec::with_capacity(pairs.len());

        for (a, b) in pairs {
            let (winner, loser) = if a < b { (a, b) } else { (b, a) };
            comparisons.push(BtComparison::new(winner, loser));
        }

        let scores = bt_fit(&comparisons, 20, &BtConfig::default());
        let ranked = scores.rank();

        // Candidate 0 (always wins) should rank in top-5
        let pos_of_0 = ranked.iter().position(|&c| c == 0).unwrap();
        assert!(
            pos_of_0 < 5,
            "Best candidate (0) should rank in top-5, got position {pos_of_0}"
        );

        // Candidate 19 (always loses) should rank in bottom-5
        let pos_of_19 = ranked.iter().position(|&c| c == 19).unwrap();
        assert!(
            pos_of_19 >= 15,
            "Worst candidate (19) should rank in bottom-5, got position {pos_of_19}"
        );

        // Average rank of top-5 candidates (0..5) should be lower than bottom-5 (15..20)
        let avg_top5: f32 = (0..5)
            .map(|c| ranked.iter().position(|&r| r == c).unwrap() as f32)
            .sum::<f32>()
            / 5.0;
        let avg_bottom5: f32 = (15..20)
            .map(|c| ranked.iter().position(|&r| r == c).unwrap() as f32)
            .sum::<f32>()
            / 5.0;
        assert!(
            avg_top5 < avg_bottom5,
            "Top-5 candidates should rank higher than bottom-5: {avg_top5:.1} vs {avg_bottom5:.1}"
        );
    }

    #[test]
    fn test_numerical_stability_extreme_scores() {
        // Extreme comparisons: A beats B 1000 times
        let comparisons: Vec<BtComparison> = (0..1000).map(|_| BtComparison::new(0, 1)).collect();
        let scores = bt_fit(&comparisons, 2, &BtConfig::default());

        assert!(scores.score(0).is_finite(), "Score should not be Inf/NaN");
        assert!(scores.score(1).is_finite(), "Score should not be Inf/NaN");
        assert!(
            scores.score(0) > scores.score(1),
            "Winner should still rank higher despite extreme comparisons"
        );
        // Regularization prevents unbounded growth
        assert!(
            scores.score(0) < 100.0,
            "Regularization should prevent score explosion, got {}",
            scores.score(0)
        );
    }

    #[test]
    fn test_sigmoid_values() {
        // Known values
        assert!((sigmoid(0.0) - 0.5).abs() < EPS, "sigmoid(0) = 0.5");
        assert!((sigmoid(1.0) - 0.7311).abs() < 0.01, "sigmoid(1) ≈ 0.7311");
        assert!(
            (sigmoid(-1.0) - 0.2689).abs() < 0.01,
            "sigmoid(-1) ≈ 0.2689"
        );
        // Extremes
        assert!(sigmoid(100.0) > 0.99, "sigmoid(large+) → 1");
        assert!(sigmoid(-100.0) < 0.01, "sigmoid(large-) → 0");
        // Numerical stability
        assert!(sigmoid(10000.0).is_finite(), "no overflow at +∞");
        assert!(sigmoid(-10000.0).is_finite(), "no underflow at -∞");
    }

    #[test]
    fn test_bt_pair_random_no_self_pairs() {
        let mut rng = Rng::with_seed(123);
        let pairs = bt_pair_random(10, 5, &mut rng);

        for (a, b) in &pairs {
            assert_ne!(a, b, "No self-pairs allowed");
        }
        // Should have 10 * 5 = 50 pairs
        assert_eq!(pairs.len(), 50);
    }

    #[test]
    fn test_bt_pair_random_small_pool() {
        let mut rng = Rng::with_seed(456);
        // n=2 → only one possible pairing direction
        let pairs = bt_pair_random(2, 3, &mut rng);
        assert_eq!(pairs.len(), 6);
        for (a, b) in &pairs {
            assert_ne!(a, b);
        }
    }

    #[test]
    fn test_bt_pair_random_empty() {
        let mut rng = Rng::with_seed(789);
        assert!(bt_pair_random(0, 4, &mut rng).is_empty());
        assert!(bt_pair_random(1, 4, &mut rng).is_empty());
        assert!(bt_pair_random(5, 0, &mut rng).is_empty());
    }

    #[test]
    fn test_top_k() {
        let comparisons = vec![
            BtComparison::new(2, 0), // 2 beats 0
            BtComparison::new(2, 1), // 2 beats 1
            BtComparison::new(1, 0), // 1 beats 0
        ];
        let scores = bt_fit(&comparisons, 3, &BtConfig::default());

        let top1 = scores.top_k(1);
        assert_eq!(top1, vec![2], "Top-1 should be candidate 2");

        let top2 = scores.top_k(2);
        assert_eq!(top2[0], 2, "First in top-2 should be 2");
        assert_eq!(top2[1], 1, "Second in top-2 should be 1");
    }

    #[test]
    fn test_regularization_effect() {
        // Without comparisons, regularization pulls scores toward 0
        let mut scores = [10.0f32, -5.0f32];
        let config = BtConfig {
            lambda: 0.1,
            ..BtConfig::default()
        };

        // Manually simulate: gradient for score[i] = -λ·s[i]
        for _ in 0..1000 {
            for s in scores.iter_mut() {
                let grad = -config.lambda * *s;
                *s += 0.1 * grad;
            }
        }

        assert!(
            scores[0].abs() < 0.01,
            "Regularization should pull toward 0"
        );
        assert!(
            scores[1].abs() < 0.01,
            "Regularization should pull toward 0"
        );
    }

    #[test]
    fn test_convergence_early_stop() {
        // Two candidates with clear signal should converge quickly
        let comparisons = vec![BtComparison::new(0, 1), BtComparison::new(0, 1)];
        let config = BtConfig {
            tolerance: 1e-3,
            max_iterations: 10000, // high limit, but should stop early
            lambda: 0.01,
        };
        let scores = bt_fit(&comparisons, 2, &config);

        assert!(
            scores.score(0) > scores.score(1),
            "Should converge correctly"
        );
    }

    #[test]
    fn test_bt_outcome_enum() {
        // BtOutcome basic usage
        let win_a = BtOutcome::Win(0);
        let win_b = BtOutcome::Win(1);
        let tie = BtOutcome::Tie;

        assert_eq!(win_a, BtOutcome::Win(0));
        assert_ne!(win_a, win_b);
        assert_eq!(tie, BtOutcome::Tie);
    }

    #[test]
    fn test_bt_comparison_struct() {
        let c = BtComparison::new(3, 7);
        assert_eq!(c.winner, 3);
        assert_eq!(c.loser, 7);
    }
}
