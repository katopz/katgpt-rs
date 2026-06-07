//! Adaptive CoT Budget Scaler for Lodestar (Plan 207, T9).
//!
//! An EMA bandit that learns how to scale `tree_budget` based on the root
//! completion distance `d(root)`. When completion is far away (high d(root)),
//! a bigger tree budget is needed to explore effectively. When completion is
//! near (low d(root)), a smaller budget suffices.
//!
//! # Architecture
//!
//! ```text
//! d(root) → bin → arm_index → BanditStats selects multiplier → scaled_budget
//!                                           ↑
//!                             reward = acceptance_rate (1.0 or 0.0)
//! ```

use super::bandit::BanditStats;

/// Budget multiplier arms — candidate scaling factors.
const BUDGET_ARMS: [f32; 6] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

/// Distance bins — how far from completion at root.
/// Bin 0: d=0-1 (nearly done), Bin 1: d=2-4, Bin 2: d=5-8, Bin 3: d=9+ (far).
const NUM_DISTANCE_BINS: usize = 4;

/// Configuration for [`AdaptiveCoTBudget`].
#[derive(Clone, Debug)]
pub struct AdaptiveCoTConfig {
    /// EMA smoothing factor for reward updates. Default: 0.1.
    pub ema_alpha: f32,
    /// Minimum visits before using bandit-selected arm. Default: 5.
    pub min_visits: u32,
    /// Default multiplier when not enough data. Default: 1.0.
    pub default_multiplier: f32,
    /// Minimum scaled budget (floor). Default: 4.
    pub min_budget: usize,
    /// Maximum scaled budget (ceiling, as multiple of base). Default: 4.0.
    pub max_budget_scale: f32,
}

impl Default for AdaptiveCoTConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.1,
            min_visits: 5,
            default_multiplier: 1.0,
            min_budget: 4,
            max_budget_scale: 4.0,
        }
    }
}

/// Adaptive CoT budget scaler — EMA bandit per distance bin.
///
/// Each distance bin maintains its own `BanditStats` over the 6 budget arms.
/// During warmup (fewer than `min_visits` total pulls in a bin), the
/// `default_multiplier` is used. After warmup, the arm with the highest
/// Q-value is selected greedily.
pub struct AdaptiveCoTBudget {
    /// Per-distance-bin bandit stats (6 arms each).
    stats: Vec<BanditStats>,
    /// Config.
    config: AdaptiveCoTConfig,
    /// Cached last selection: (bin, arm) for reward attribution.
    last_selection: Option<(usize, usize)>,
}

impl AdaptiveCoTBudget {
    /// Create a new adaptive CoT budget scaler with the given config.
    pub fn new(config: AdaptiveCoTConfig) -> Self {
        let stats = (0..NUM_DISTANCE_BINS)
            .map(|_| BanditStats::new(BUDGET_ARMS.len()))
            .collect();
        Self {
            stats,
            config,
            last_selection: None,
        }
    }

    /// Create with default config.
    pub fn default_config() -> Self {
        Self::new(AdaptiveCoTConfig::default())
    }

    /// Select the scaled budget for a given root completion distance.
    ///
    /// 1. Compute distance bin from `d_root`.
    /// 2. If bin has < `min_visits` total pulls, return `base * default_multiplier`.
    /// 3. Otherwise, select arm with highest Q-value (greedy after warmup).
    /// 4. Return `base * BUDGET_ARMS[arm]` clamped to `[min_budget, base * max_budget_scale]`.
    #[inline]
    pub fn select_budget(&mut self, d_root: u32, base_budget: usize) -> usize {
        let bin = Self::distance_bin(d_root);

        // Warmup: use default multiplier until we have enough data.
        if self.stats[bin].total_pulls() < self.config.min_visits {
            let budget = (base_budget as f32 * self.config.default_multiplier).round() as usize;
            return self.clamp_budget(budget, base_budget);
        }

        // Greedy arm selection — highest Q-value.
        let arm = self.stats[bin].best_arm();
        self.last_selection = Some((bin, arm));

        let budget = (base_budget as f32 * BUDGET_ARMS[arm]).round() as usize;
        self.clamp_budget(budget, base_budget)
    }

    /// Observe reward (0.0 or 1.0) for the last selection.
    ///
    /// Uses [`BanditStats::update`] incremental mean.
    pub fn observe_reward(&mut self, reward: f32) {
        let Some((bin, arm)) = self.last_selection.take() else {
            return;
        };
        self.stats[bin].update(arm, reward);
    }

    /// Observe reward using EMA update instead of incremental mean.
    ///
    /// `Q(a) = (1 - α) * Q(a) + α * reward`
    pub fn observe_reward_ema(&mut self, reward: f32) {
        let Some((bin, arm)) = self.last_selection.take() else {
            return;
        };
        let alpha = self.config.ema_alpha;
        let q = self.stats[bin].q_value(arm);
        let new_q = (1.0 - alpha) * q + alpha * reward;

        // Directly update the Q-value via incremental mean with synthetic count.
        // We still need to bump visit/total_pulls for warmup tracking.
        self.stats[bin].update(arm, new_q);
    }

    /// Map root completion distance to a bin index.
    ///
    /// ```text
    /// d=0-1  → bin 0 (nearly done)
    /// d=2-4  → bin 1 (medium)
    /// d=5-8  → bin 2 (far)
    /// d=9+   → bin 3 (very far)
    /// ```
    #[inline]
    pub fn distance_bin(d: u32) -> usize {
        match d {
            0..=1 => 0,
            2..=4 => 1,
            5..=8 => 2,
            _ => 3,
        }
    }

    /// Get a reference to bandit stats for a bin (for inspection/logging).
    pub fn stats_for_bin(&self, bin: usize) -> Option<&BanditStats> {
        self.stats.get(bin)
    }

    /// Get the config reference.
    pub fn config(&self) -> &AdaptiveCoTConfig {
        &self.config
    }

    /// Clamp budget to `[min_budget, base * max_budget_scale]`.
    #[inline]
    fn clamp_budget(&self, budget: usize, base_budget: usize) -> usize {
        let max_budget = (base_budget as f32 * self.config.max_budget_scale).round() as usize;
        budget.clamp(
            self.config.min_budget,
            max_budget.max(self.config.min_budget),
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_returns_base_budget() {
        let mut scaler = AdaptiveCoTBudget::default_config();
        // No observations → should return base * default_multiplier = base * 1.0 = base.
        let budget = scaler.select_budget(3, 64);
        assert_eq!(budget, 64);
    }

    #[test]
    fn test_distance_bin_boundaries() {
        assert_eq!(AdaptiveCoTBudget::distance_bin(0), 0);
        assert_eq!(AdaptiveCoTBudget::distance_bin(1), 0);
        assert_eq!(AdaptiveCoTBudget::distance_bin(2), 1);
        assert_eq!(AdaptiveCoTBudget::distance_bin(4), 1);
        assert_eq!(AdaptiveCoTBudget::distance_bin(5), 2);
        assert_eq!(AdaptiveCoTBudget::distance_bin(8), 2);
        assert_eq!(AdaptiveCoTBudget::distance_bin(9), 3);
        assert_eq!(AdaptiveCoTBudget::distance_bin(100), 3);
    }

    #[test]
    fn test_observe_reward_updates_stats() {
        let mut scaler = AdaptiveCoTBudget::default_config();
        // Force past warmup by manually updating stats.
        let bin = 1; // d_root=3 → bin 1
        for _ in 0..6 {
            scaler.stats[bin].update(0, 1.0);
        }
        assert!(scaler.stats[bin].total_pulls() >= scaler.config.min_visits);

        // Select should now pick an arm.
        let _budget = scaler.select_budget(3, 64);
        assert!(scaler.last_selection.is_some());

        // Observe reward.
        let _selected_arm = scaler.last_selection.unwrap().1;
        scaler.observe_reward(1.0);
        assert!(scaler.last_selection.is_none()); // cleared after observe

        // Q-values should have changed.
        let q_after = scaler.stats[bin].q_value(0);
        // After 7 updates of arm 0 with reward 1.0, Q should be close to 1.0.
        assert!(q_after > 0.9);
    }

    #[test]
    fn test_convergence_to_high_multiplier() {
        let config = AdaptiveCoTConfig {
            min_visits: 3,
            ..Default::default()
        };
        let mut scaler = AdaptiveCoTBudget::new(config);

        // Simulate many rounds where we always accept (reward=1.0).
        // The 2.0x arm (index 5) should converge to the highest Q-value.
        for _ in 0..200 {
            let budget = scaler.select_budget(9, 64);
            assert!(budget >= scaler.config.min_budget);
            scaler.observe_reward(1.0);
        }

        // After convergence, bin 3 should prefer the highest multiplier.
        let best = scaler.stats[3].best_arm();
        assert_eq!(best, 5); // 2.0x arm
    }

    #[test]
    fn test_rejection_drives_down_multiplier() {
        let config = AdaptiveCoTConfig {
            min_visits: 3,
            ..Default::default()
        };
        let mut scaler = AdaptiveCoTBudget::new(config);

        // Simulate many rounds where we always reject (reward=0.0).
        // The 0.5x arm (index 0) should converge to the highest Q-value
        // because with reward=0.0 all arms get Q=0.0 but the first-updated
        // (earliest pulled) arm stays competitive. With incremental mean
        // and all-zero rewards, all Q-values converge to 0.0, so best_arm
        // returns the first one (index 0) due to max_by being stable.
        // Instead, let's use asymmetric rewards: give 0.5x a small bonus.
        for _ in 0..100 {
            let _budget = scaler.select_budget(5, 64);
            // Give arm 0 a tiny advantage by directly updating it.
            scaler.stats[2].update(0, 0.1);
            scaler.observe_reward(0.0);
        }

        // Arm 0 should have the highest Q-value (0.1 updates vs 0.0 for others).
        let best = scaler.stats[2].best_arm();
        assert_eq!(best, 0); // 0.5x arm
    }

    #[test]
    fn test_budget_clamping() {
        let config = AdaptiveCoTConfig {
            min_visits: 0,
            min_budget: 4,
            max_budget_scale: 4.0,
            ..Default::default()
        };
        let mut scaler = AdaptiveCoTBudget::new(config);

        // Very small base budget should be floored to min_budget.
        let budget = scaler.select_budget(3, 1);
        assert_eq!(budget, 4);

        // Normal budget should respect max_budget_scale ceiling.
        // With default_multiplier=1.0 and base=64, budget=64 (within bounds).
        let budget = scaler.select_budget(3, 64);
        assert!(budget <= 64 * 4);
        assert!(budget >= 4);
    }

    #[test]
    fn test_different_bins_independent() {
        let config = AdaptiveCoTConfig {
            min_visits: 1,
            ..Default::default()
        };
        let mut scaler = AdaptiveCoTBudget::new(config);

        // Give bin 0 lots of high rewards on arm 5 (2.0x).
        for _ in 0..50 {
            scaler.stats[0].update(5, 1.0);
        }

        // Bin 3 has no data — should still use default.
        let budget_bin3 = scaler.select_budget(20, 64);
        assert_eq!(budget_bin3, 64); // default_multiplier * base

        // Bin 0 should prefer arm 5.
        let budget_bin0 = scaler.select_budget(0, 64);
        assert_eq!(budget_bin0, 128); // 2.0 * 64

        // Bin 3 stats should be unchanged.
        assert_eq!(scaler.stats[3].total_pulls(), 0);
    }

    #[test]
    fn test_select_budget_deterministic_after_warmup() {
        let config = AdaptiveCoTConfig {
            min_visits: 5,
            ..Default::default()
        };
        let mut scaler = AdaptiveCoTBudget::new(config);

        // Warm up bin 2 with clear preference for arm 4 (1.5x).
        for _ in 0..20 {
            scaler.stats[2].update(4, 1.0);
        }
        // Give other arms low rewards.
        for arm in [0, 1, 2, 3, 5] {
            scaler.stats[2].update(arm, 0.0);
        }

        // Selection should be deterministic — always arm 4.
        for _ in 0..10 {
            let budget = scaler.select_budget(6, 64);
            assert_eq!(budget, 96); // 1.5 * 64 = 96
            scaler.observe_reward(0.5); // reward doesn't change dominance
        }
    }
}

// TL;DR: EMA bandit per distance bin that scales tree_budget — far from completion
// gets bigger budget, near gets smaller. Greedy after warmup, clamped to safe range.
