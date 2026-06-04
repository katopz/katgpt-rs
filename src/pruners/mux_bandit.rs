//! MuxBanditWidth — Bandit-based adaptive superposition width selection (Research 158, MUX).
//!
//! Uses a multi-armed bandit to adaptively select the superposition width K for
//! multiplexed latent reasoning. The arms correspond to different K values from
//! the Fibonacci-inspired set `{1, 2, 3, 5, 8}`.
//!
//! # Algorithm
//!
//! UCB1-style exploration: select the arm with the highest upper confidence bound,
//! which balances exploitation (high average reward) with exploration (less-visited arms).
//!
//! Q(a) + sqrt(2 * ln(N) / n(a))
//!
//! where Q(a) is the average reward for arm a, N is total pulls, n(a) is pulls for arm a.

use fastrand::Rng;

/// Candidate superposition widths — Fibonacci-inspired K values.
pub const MUX_WIDTH_ARMS: [usize; 5] = [1, 2, 3, 5, 8];

/// Bandit-based adaptive superposition width selector.
///
/// Maintains per-arm statistics (pull count, cumulative reward) and uses UCB1
/// to balance exploration vs exploitation when selecting the superposition width K.
///
/// # Usage
///
/// ```rust,ignore
/// let mut bandit = MuxBanditWidth::new();
/// let k = bandit.select(&mut rng);
/// // ... use k for multiplexed decoding ...
/// bandit.update(k_arm_index, reward);
/// let best = bandit.best_width(); // converges to optimal K
/// ```
///
/// (Research 158, MUX)
pub struct MuxBanditWidth {
    /// Candidate superposition widths.
    pub arms: [usize; 5],
    /// Pull count per arm.
    pub counts: [u32; 5],
    /// Cumulative reward per arm.
    pub rewards: [f64; 5],
}

impl MuxBanditWidth {
    /// Create a new bandit with zero pulls and zero rewards.
    pub fn new() -> Self {
        Self {
            arms: MUX_WIDTH_ARMS,
            counts: [0; 5],
            rewards: [0.0; 5],
        }
    }

    /// Select a superposition width using UCB1.
    ///
    /// If any arm has not been pulled yet, selects it first (initialization phase).
    /// Otherwise, selects the arm maximizing the UCB1 score.
    ///
    /// Returns the selected width value (not the arm index).
    pub fn select(&mut self, rng: &mut Rng) -> usize {
        let total: u32 = self.counts.iter().sum();

        // Initialization: pull each arm at least once
        for (i, &count) in self.counts.iter().enumerate() {
            if count == 0 {
                return self.arms[i];
            }
        }

        // UCB1 selection
        let ln_total = (total as f64).ln();
        let mut best_idx = 0;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..self.arms.len() {
            let q = self.rewards[i] / self.counts[i] as f64;
            let exploration = (2.0 * ln_total / self.counts[i] as f64).sqrt();
            let score = q + exploration;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        // Break ties randomly for diversity
        let mut tied: Vec<usize> = Vec::new();
        for i in 0..self.arms.len() {
            let q = self.rewards[i] / self.counts[i] as f64;
            let exploration = (2.0 * ln_total / self.counts[i] as f64).sqrt();
            let score = q + exploration;
            if (score - best_score).abs() < 1e-10 {
                tied.push(i);
            }
        }

        let idx = if tied.len() > 1 {
            tied[rng.usize(..tied.len())]
        } else {
            best_idx
        };

        self.arms[idx]
    }

    /// Update the bandit with a reward for the given arm index.
    ///
    /// `arm` should be the width value (e.g., 3), not the array index.
    /// If `arm` is not in the arms list, the update is ignored.
    pub fn update(&mut self, arm: usize, reward: f64) {
        if let Some(i) = self.arms.iter().position(|&a| a == arm) {
            self.counts[i] += 1;
            self.rewards[i] += reward;
        }
    }

    /// Return the best-performing width (highest average reward).
    ///
    /// Returns the first arm if no pulls have been made yet.
    pub fn best_width(&self) -> usize {
        let mut best_idx = 0;
        let mut best_avg = f64::NEG_INFINITY;

        for (i, &count) in self.counts.iter().enumerate() {
            if count > 0 {
                let avg = self.rewards[i] / count as f64;
                if avg > best_avg {
                    best_avg = avg;
                    best_idx = i;
                }
            }
        }

        // If no pulls yet, return default (first arm)
        if best_avg == f64::NEG_INFINITY {
            return self.arms[0];
        }

        self.arms[best_idx]
    }

    /// Total number of pulls across all arms.
    pub fn total_pulls(&self) -> u32 {
        self.counts.iter().sum()
    }

    /// Average reward for a specific arm.
    ///
    /// Returns `None` if the arm hasn't been pulled or doesn't exist.
    pub fn avg_reward(&self, arm: usize) -> Option<f64> {
        let i = self.arms.iter().position(|&a| a == arm)?;
        if self.counts[i] == 0 {
            return None;
        }
        Some(self.rewards[i] / self.counts[i] as f64)
    }
}

impl Default for MuxBanditWidth {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mux_bandit_new() {
        let bandit = MuxBanditWidth::new();
        assert_eq!(bandit.arms, [1, 2, 3, 5, 8]);
        assert_eq!(bandit.counts, [0, 0, 0, 0, 0]);
        assert_eq!(bandit.rewards, [0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mux_bandit_initialization_phase() {
        let mut bandit = MuxBanditWidth::new();
        let mut rng = Rng::with_seed(42);

        // First 5 selects should cover all arms (initialization)
        let mut seen = vec![false; 5];
        for _ in 0..5 {
            let width = bandit.select(&mut rng);
            let idx = bandit.arms.iter().position(|&a| a == width).unwrap();
            seen[idx] = true;
            bandit.update(width, 0.5);
        }
        assert!(
            seen.iter().all(|&s| s),
            "all arms should be pulled during initialization"
        );
    }

    #[test]
    fn test_mux_bandit_width_convergence() {
        let mut bandit = MuxBanditWidth::new();
        let mut rng = Rng::with_seed(42);

        // Simulate: arm index 2 (width=3) is the best with reward ~0.9
        for _ in 0..200 {
            let width = bandit.select(&mut rng);
            let reward = if width == 3 { 0.9 } else { 0.3 };
            bandit.update(width, reward);
        }

        let best = bandit.best_width();
        assert_eq!(
            best, 3,
            "bandit should converge to width=3 (highest reward arm)"
        );
    }

    #[test]
    fn test_mux_bandit_update_ignores_unknown_arm() {
        let mut bandit = MuxBanditWidth::new();
        bandit.update(999, 1.0);
        assert_eq!(bandit.total_pulls(), 0);
    }

    #[test]
    fn test_mux_bandit_avg_reward() {
        let mut bandit = MuxBanditWidth::new();
        bandit.update(3, 0.8);
        bandit.update(3, 1.0);
        assert!((bandit.avg_reward(3).unwrap() - 0.9).abs() < 1e-10);
        assert!(bandit.avg_reward(999).is_none());
        assert!(bandit.avg_reward(1).is_none()); // not pulled yet
    }

    #[test]
    fn test_mux_bandit_best_width_no_pulls() {
        let bandit = MuxBanditWidth::new();
        assert_eq!(bandit.best_width(), 1); // first arm as default
    }

    #[test]
    fn test_mux_bandit_total_pulls() {
        let mut bandit = MuxBanditWidth::new();
        bandit.update(1, 0.5);
        bandit.update(3, 0.5);
        bandit.update(8, 0.5);
        assert_eq!(bandit.total_pulls(), 3);
    }

    #[test]
    fn test_mux_bandit_default() {
        let bandit = MuxBanditWidth::default();
        assert_eq!(bandit.arms, MUX_WIDTH_ARMS);
    }
}
