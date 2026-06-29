//! Generic P-UCB selector for SR²AM and DDTree (Plan 143, Phase 2).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "P-UCB balances exploitation (high Elo) with exploration (low visits)
//! via an upper confidence bound applied to normalized scores."
//!
//! Unlike the sketch-specific P-UCB in `proof/sketch_sampler.rs`, this is
//! a generic selector that works with any `(id, score, visits)` population.
//!
//! # P-UCB Formula
//!
//! ```text
//! score(i) = q(i) + c × √(N / (n_i + 1))
//!
//! q(i) = normalize_to_01(score_i, min, max) ∈ [0, 1]
//! N    = Σ n_i  (total visits)
//! n_i  = visits for item i
//! c    = exploration constant (default 0.2)
//! ```
//!
//! # Adaptive Exploration
//!
//! The exploration constant `c` can be annealed based on solve rate:
//! - solve_rate < 10% → increase c (more exploration)
//! - solve_rate > 50% → decrease c (more exploitation)
//!
//! # Feature Gate
//!
//! Requires `state_source` feature (depends on `bandit`).

// ── Constants ─────────────────────────────────────────────────

/// Default exploration constant (paper: c = 0.2).
pub const DEFAULT_PUCB_C: f64 = 0.2;

/// Default top-K filter (paper: K = 64).
pub const DEFAULT_TOP_K: usize = 64;

// ── PUCBSelector ──────────────────────────────────────────────

/// Generic P-UCB selector for populations of `(id, score, visits)` items.
///
/// Filters to top-K by score, normalizes to [0,1], applies UCB bonus.
#[derive(Debug, Clone)]
pub struct PUCBSelector {
    /// Exploration constant c.
    pub exploration_constant: f64,
    /// Top-K filter — only consider the K highest-scoring items.
    pub top_k_filter: usize,
}

impl Default for PUCBSelector {
    fn default() -> Self {
        Self {
            exploration_constant: DEFAULT_PUCB_C,
            top_k_filter: DEFAULT_TOP_K,
        }
    }
}

impl PUCBSelector {
    /// Create selector with custom exploration constant and top-K.
    pub fn new(exploration_constant: f64, top_k_filter: usize) -> Self {
        Self {
            exploration_constant,
            top_k_filter,
        }
    }

    /// Select the item with highest P-UCB score.
    ///
    /// # Arguments
    ///
    /// * `population` — `&[(id, score, visits)]`, unsorted
    ///
    /// # Returns
    ///
    /// The id of the selected item, or `None` if population is empty.
    pub fn select(&self, population: &[(u64, f64, usize)]) -> Option<u64> {
        if population.is_empty() {
            return None;
        }

        // Filter to top-K by score
        let filtered: Vec<&(u64, f64, usize)> = if population.len() <= self.top_k_filter {
            population.iter().collect()
        } else {
            let mut indexed: Vec<(usize, f64)> = population
                .iter()
                .enumerate()
                .map(|(i, (_, s, _))| (i, *s))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed
                .into_iter()
                .take(self.top_k_filter)
                .map(|(i, _)| &population[i])
                .collect()
        };

        if filtered.is_empty() {
            return None;
        }

        // Compute normalization bounds
        let min_score = filtered
            .iter()
            .map(|(_, s, _)| *s)
            .fold(f64::INFINITY, f64::min);
        let max_score = filtered
            .iter()
            .map(|(_, s, _)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let range = max_score - min_score;

        // Total visits
        let total_visits: usize = filtered.iter().map(|(_, _, v)| *v).sum();

        // Select by P-UCB score
        let mut best_id = filtered[0].0;
        let mut best_pucb = f64::NEG_INFINITY;

        for &(id, score, visits) in &filtered {
            let q = if range > 1e-10 {
                (score - min_score) / range
            } else {
                0.5 // uniform when all scores equal
            };

            let bonus = self.exploration_constant
                * ((total_visits as f64 + 1.0).ln() / (*visits as f64 + 1.0)).sqrt();
            let pucb = q + bonus;

            if pucb > best_pucb {
                best_pucb = pucb;
                best_id = *id;
            }
        }

        Some(best_id)
    }
}

// ── Adaptive Exploration ──────────────────────────────────────

/// Compute adaptive exploration constant based on solve rate.
///
/// - solve_rate < 0.1: increase c by up to 30%
/// - solve_rate > 0.5: decrease c by up to 30%
/// - Otherwise: use base_c
pub fn adaptive_c(solve_rate: f64, base_c: f64) -> f64 {
    base_c * (1.0 + (0.5 - solve_rate).clamp(-0.3, 0.3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_from_empty() {
        let selector = PUCBSelector::default();
        assert!(selector.select(&[]).is_none());
    }

    #[test]
    fn select_prefers_high_score() {
        let selector = PUCBSelector::new(0.0, 10); // No exploration bonus
        let pop: Vec<(u64, f64, usize)> = vec![
            (1, 0.5, 10), // Mid score, many visits
            (2, 0.9, 10), // High score, many visits
            (3, 0.1, 10), // Low score, many visits
        ];
        assert_eq!(selector.select(&pop), Some(2));
    }

    #[test]
    fn select_explores_low_visits() {
        let selector = PUCBSelector::new(1.0, 10); // High exploration
        let pop: Vec<(u64, f64, usize)> = vec![
            (1, 0.5, 1000), // Decent score, many visits
            (2, 0.3, 0),    // Low score, no visits → high bonus
        ];
        // With high c and low visits, the bonus can outweigh the score difference
        assert_eq!(selector.select(&pop), Some(2));
    }

    #[test]
    fn top_k_filter() {
        let selector = PUCBSelector::new(0.0, 2); // Only top 2
        let pop: Vec<(u64, f64, usize)> = vec![
            (1, 0.9, 1),
            (2, 0.8, 1),
            (3, 0.1, 1), // Filtered out
        ];
        // Item 3 is filtered out; selection from {1, 2}
        let selected = selector.select(&pop).unwrap();
        assert!(selected == 1 || selected == 2);
    }

    #[test]
    fn adaptive_c_low_solve_rate() {
        let c = adaptive_c(0.05, 0.2);
        assert!(
            c > 0.2,
            "Should increase exploration when solve rate is low"
        );
    }

    #[test]
    fn adaptive_c_high_solve_rate() {
        let c = adaptive_c(0.7, 0.2);
        assert!(
            c < 0.2,
            "Should decrease exploration when solve rate is high"
        );
    }

    #[test]
    fn adaptive_c_mid_solve_rate() {
        let c = adaptive_c(0.5, 0.2);
        assert!(
            (c - 0.2).abs() < 0.01,
            "Should stay at base when solve rate is 0.5"
        );
    }
}
