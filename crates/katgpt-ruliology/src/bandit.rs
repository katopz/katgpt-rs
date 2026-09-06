//! Ruliology Bandit — FSM strategies as UCB1 bandit arms.
//!
//! Pipeline: enumerate FSMs → tournament → Pareto filter → UCB1 selection.
//! Includes AbsorbCompress-style promotion for stable high-payoff FSMs.
//! Plan 188 Phase 2.

use std::collections::HashSet;

use crate::fsm::{FsmEnumerator, FsmStrategy};
use crate::types::SimpleProgram;

// ── RuliologyArm ──────────────────────────────────────────────

/// Bandit arm backed by an enumerated FSM strategy.
pub struct RuliologyArm {
    /// The FSM strategy for this arm.
    pub strategy: FsmStrategy,
    /// Running payoff estimate (updated via incremental mean).
    payoff_estimate: f64,
    /// Number of times this arm has been pulled.
    pull_count: u32,
}

impl RuliologyArm {
    pub fn new(strategy: FsmStrategy) -> Self {
        Self {
            strategy,
            payoff_estimate: 0.0,
            pull_count: 0,
        }
    }

    /// Update payoff with incremental mean.
    pub fn update(&mut self, reward: f64) {
        self.pull_count += 1;
        let n = self.pull_count as f64;
        self.payoff_estimate += (reward - self.payoff_estimate) / n;
    }

    /// Get the current payoff estimate.
    #[inline]
    pub fn payoff(&self) -> f64 {
        self.payoff_estimate
    }

    /// Number of pulls.
    #[inline]
    pub fn pulls(&self) -> u32 {
        self.pull_count
    }

    /// UCB1 score for this arm.
    pub fn ucb1_score(&self, total_pulls: u32) -> f64 {
        if self.pull_count == 0 {
            return f64::INFINITY;
        }
        let exploration = (2.0 * (total_pulls as f64).ln() / self.pull_count as f64).sqrt();
        self.payoff_estimate + exploration
    }
}

// ── RuliologyBandit ───────────────────────────────────────────

/// Bandit that selects from Pareto-filtered FSM strategies.
///
/// Workflow:
/// 1. Enumerate all N-state FSMs via FsmEnumerator
/// 2. Run round-robin tournament to rank them
/// 3. Filter to Pareto-optimal arms via WinMatrix::pareto_front
/// 4. Use UCB1 to select best arm per round
/// 5. Update payoffs from game results
pub struct RuliologyBandit {
    /// Pre-filtered arms from Pareto front.
    arms: Vec<RuliologyArm>,
    /// Total pulls across all arms.
    total_pulls: u32,
    /// Minimum payoff threshold for filtering.
    #[allow(dead_code)]
    payoff_threshold: f64,
    /// Maximum complexity for filtering.
    #[allow(dead_code)]
    complexity_threshold: f32,
}

impl RuliologyBandit {
    /// Create from pre-enumerated FSM strategies.
    ///
    /// Runs tournament, extracts Pareto front, creates arms from optimal strategies.
    pub fn from_strategies(
        strategies: &[FsmStrategy],
        rounds: u32,
        payoff_fn: &dyn Fn(u8, u8) -> f64,
        payoff_threshold: f64,
        complexity_threshold: f32,
    ) -> Self {
        // Run tournament
        let win_matrix = FsmEnumerator::tournament(strategies, rounds, payoff_fn);

        // Compute complexities for Pareto front
        let complexities: Vec<f32> = strategies.iter().map(|s| s.complexity()).collect();

        // Extract Pareto-optimal strategies
        let pareto = win_matrix.pareto_front(&complexities);

        // Create arms from Pareto-optimal strategies
        // Index strategies by id once to avoid O(n²) find() inside the
        // pareto loop below (was strategies.iter().find() per pareto entry).
        use std::collections::HashMap;
        let by_id: HashMap<u64, &FsmStrategy> = strategies.iter().map(|s| (s.id(), s)).collect();
        let arms: Vec<RuliologyArm> = pareto
            .iter()
            .filter(|(_, payoff, _)| *payoff >= payoff_threshold)
            .filter_map(|(id, _, _)| by_id.get(id).copied())
            .map(|s| RuliologyArm::new(s.clone()))
            .collect();

        Self {
            arms,
            total_pulls: 0,
            payoff_threshold,
            complexity_threshold,
        }
    }

    /// Select best arm using UCB1.
    pub fn select_arm(&self) -> usize {
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            let score = arm.ucb1_score(self.total_pulls);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// Get the FSM strategy for an arm.
    pub fn strategy(&self, arm: usize) -> &FsmStrategy {
        &self.arms[arm].strategy
    }

    /// Update an arm's payoff after observing a reward.
    pub fn update(&mut self, arm: usize, reward: f64) {
        self.arms[arm].update(reward);
        self.total_pulls += 1;
    }

    /// Number of arms (Pareto-optimal strategies).
    #[inline]
    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    /// Best arm by payoff estimate.
    pub fn best_arm(&self) -> usize {
        let mut best = 0usize;
        let mut best_payoff = f64::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            let p = arm.payoff();
            // A NaN payoff must never become the best arm: `p > best_payoff`
            // is false for NaN (float_order semantics — NaN loses), whereas
            // total_cmp would rank NaN above +inf and select it.
            if p > best_payoff {
                best_payoff = p;
                best = i;
            }
        }
        best
    }

    /// Best arm by payoff, excluding already-promoted arms.
    pub fn best_unpromoted_arm(&self, promoted: &HashSet<usize>) -> usize {
        let mut best = 0usize;
        let mut best_payoff = f64::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            if promoted.contains(&i) {
                continue;
            }
            let p = arm.payoff();
            if p.total_cmp(&best_payoff).is_gt() {
                best_payoff = p;
                best = i;
            }
        }
        best
    }

    /// Get all arms.
    #[inline]
    pub fn arms(&self) -> &[RuliologyArm] {
        &self.arms
    }

    /// Total pulls across all arms.
    #[inline]
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payoff::matching_pennies;

    #[test]
    fn test_ruliology_arm_ucb1_unvisited_max() {
        let transitions = [[0u8; 2]; crate::fsm::MAX_STATES];
        let outputs = [0u8; crate::fsm::MAX_STATES];
        let strategy = FsmStrategy::new(transitions, outputs, 1, 0);
        let arm = RuliologyArm::new(strategy);
        // Unvisited arm should have infinite UCB1 score.
        assert!(arm.ucb1_score(100).is_infinite());
        assert!(arm.ucb1_score(0).is_infinite());
    }

    #[test]
    fn test_ruliology_arm_update_incremental_mean() {
        let transitions = [[0u8; 2]; crate::fsm::MAX_STATES];
        let outputs = [0u8; crate::fsm::MAX_STATES];
        let strategy = FsmStrategy::new(transitions, outputs, 1, 0);
        let mut arm = RuliologyArm::new(strategy);

        // Feed rewards 1.0, 0.0, 1.0, 0.0 → should converge to 0.5
        let rewards = [1.0, 0.0, 1.0, 0.0];
        for r in rewards {
            arm.update(r);
        }
        assert!(
            (arm.payoff() - 0.5).abs() < 1e-9,
            "expected 0.5, got {}",
            arm.payoff()
        );
        assert_eq!(arm.pulls(), 4);
    }

    #[test]
    fn test_ruliology_bandit_from_strategies_mp() {
        let strategies = FsmEnumerator::enumerate(2);
        // Expect ~22 distinct 2-state FSMs (Wolfram result, may vary with dedup).
        assert!(
            strategies.len() >= 18 && strategies.len() <= 30,
            "expected ~22 distinct 2-state FSMs, got {}",
            strategies.len()
        );

        let bandit = RuliologyBandit::from_strategies(
            &strategies,
            100,
            &matching_pennies,
            f64::NEG_INFINITY, // accept all payoffs
            f32::MAX,          // accept all complexities
        );

        // Should have at least some Pareto-optimal arms.
        assert!(
            bandit.num_arms() >= 1,
            "expected >= 1 arm, got {}",
            bandit.num_arms()
        );
    }

    #[test]
    fn test_ruliology_bandit_selects_best_arm() {
        let strategies = FsmEnumerator::enumerate(2);

        let mut bandit = RuliologyBandit::from_strategies(
            &strategies,
            100,
            &matching_pennies,
            f64::NEG_INFINITY,
            f32::MAX,
        );

        assert!(bandit.num_arms() > 0);

        // Pull arms many times, giving arm 0 a consistently higher reward.
        for _ in 0..200 {
            let arm = bandit.select_arm();
            // Give arm 0 a higher reward than others to make it converge.
            let reward = if arm == 0 { 1.0 } else { -1.0 };
            bandit.update(arm, reward);
        }

        // After many updates with biased rewards, arm 0 should be best.
        let best = bandit.best_arm();
        assert_eq!(best, 0, "arm 0 should be best after biased updates");
    }

    #[test]
    fn test_ruliology_bandit_num_arms_positive() {
        let strategies = FsmEnumerator::enumerate(2);

        let bandit = RuliologyBandit::from_strategies(
            &strategies,
            100,
            &matching_pennies,
            f64::NEG_INFINITY,
            f32::MAX,
        );

        // Pareto front from 22 strategies should yield at least 1 arm.
        assert!(bandit.num_arms() >= 1);
    }
}

// ── RuliologyAbsorbCompress ─────────────────────────────────────

/// Promotion config for ruliology absorb-compress.
#[derive(Clone, Copy, Debug)]
pub struct RuliologyPromoteConfig {
    /// Minimum visits before an arm can be promoted.
    pub min_visits: u32,
    /// Minimum average payoff to be promoted (stable positive payoff).
    pub payoff_threshold: f64,
    /// Maximum number of arms to promote per compress() call.
    pub max_promotions: usize,
    /// How often to check for promotions (every N absorbs).
    pub check_interval: u32,
}

impl Default for RuliologyPromoteConfig {
    fn default() -> Self {
        Self {
            min_visits: 50,
            payoff_threshold: 0.05,
            max_promotions: 3,
            check_interval: 100,
        }
    }
}

/// Absorb-compress adapter for RuliologyBandit.
///
/// Promotes FSM arms with stable positive payoff to "winner" status.
/// The inverse of the standard AbsorbCompress (which blocks bad arms) —
/// this identifies and locks in *good* arms.
///
/// Usage:
/// ```ignore
/// let mut ac = RuliologyAbsorbCompress::new(bandit, RuliologyPromoteConfig::default());
/// // Each game episode:
/// ac.absorb(arm_idx, reward);
/// if ac.should_compress() {
///     let promoted = ac.compress();
///     // promoted arms have stable positive payoff
/// }
/// // Get the current best strategy:
/// if let Some(idx) = ac.promoted_winner() {
///     let strategy = ac.bandit().strategy(idx);
/// }
/// ```
pub struct RuliologyAbsorbCompress {
    /// Inner bandit.
    bandit: RuliologyBandit,
    /// Promotion config.
    config: RuliologyPromoteConfig,
    /// Total observations absorbed.
    total_absorbed: u32,
    /// Promoted arm indices (stable winners).
    promoted: HashSet<usize>,
}

impl RuliologyAbsorbCompress {
    /// Create new absorb-compress adapter wrapping a RuliologyBandit.
    pub fn new(bandit: RuliologyBandit, config: RuliologyPromoteConfig) -> Self {
        Self {
            bandit,
            config,
            total_absorbed: 0,
            promoted: HashSet::new(),
        }
    }

    /// Absorb an observation: update arm payoff.
    pub fn absorb(&mut self, arm: usize, reward: f64) {
        self.bandit.update(arm, reward);
        self.total_absorbed += 1;
    }

    /// Whether enough observations have been absorbed to trigger compression.
    pub fn should_compress(&self) -> bool {
        self.total_absorbed > 0
            && self
                .total_absorbed
                .is_multiple_of(self.config.check_interval)
    }

    /// Promote arms with stable positive payoff.
    ///
    /// Returns indices of newly promoted arms.
    pub fn compress(&mut self) -> Vec<usize> {
        let mut newly_promoted = Vec::new();

        for (i, arm) in self.bandit.arms().iter().enumerate() {
            if self.promoted.contains(&i) {
                continue;
            }
            if arm.pulls() < self.config.min_visits {
                continue;
            }
            if arm.payoff() >= self.config.payoff_threshold {
                self.promoted.insert(i);
                newly_promoted.push(i);
                if newly_promoted.len() >= self.config.max_promotions {
                    break;
                }
            }
        }

        newly_promoted.shrink_to_fit();
        newly_promoted
    }

    /// Get the best promoted arm (winner), if any.
    pub fn promoted_winner(&self) -> Option<usize> {
        if self.promoted.is_empty() {
            return None;
        }
        // Among promoted arms, find the one with highest payoff.
        self.promoted
            .iter()
            .filter(|&&i| i < self.bandit.arms().len())
            .max_by(|&&a, &&b| {
                let pa = self.bandit.arms()[a].payoff();
                let pb = self.bandit.arms()[b].payoff();
                pa.total_cmp(&pb)
            })
            .copied()
    }

    /// Access the inner bandit.
    pub fn bandit(&self) -> &RuliologyBandit {
        &self.bandit
    }

    /// Mutable access to the inner bandit.
    pub fn bandit_mut(&mut self) -> &mut RuliologyBandit {
        &mut self.bandit
    }

    /// Promoted arm indices.
    pub fn promoted_arms(&self) -> &HashSet<usize> {
        &self.promoted
    }

    /// Total observations absorbed.
    #[inline]
    pub fn total_absorbed(&self) -> u32 {
        self.total_absorbed
    }
}

#[cfg(test)]
mod absorb_compress_tests {
    use super::*;
    use crate::payoff::matching_pennies;

    fn make_test_bandit() -> RuliologyBandit {
        let strategies = FsmEnumerator::enumerate(2);
        RuliologyBandit::from_strategies(
            &strategies,
            100,
            &matching_pennies,
            f64::NEG_INFINITY,
            f32::MAX,
        )
    }

    #[test]
    fn test_absorb_compress_no_initial_promotions() {
        let bandit = make_test_bandit();
        let ac = RuliologyAbsorbCompress::new(bandit, RuliologyPromoteConfig::default());
        assert!(ac.promoted_arms().is_empty());
        assert_eq!(ac.total_absorbed(), 0);
        assert!(ac.promoted_winner().is_none());
    }

    #[test]
    fn test_absorb_compress_promotes_after_min_visits() {
        let bandit = make_test_bandit();
        let config = RuliologyPromoteConfig {
            min_visits: 10,
            payoff_threshold: 0.5,
            max_promotions: 5,
            check_interval: 20,
        };
        let mut ac = RuliologyAbsorbCompress::new(bandit, config);

        // Give arm 0 consistently positive rewards.
        for _ in 0..15 {
            ac.absorb(0, 1.0);
        }

        // Give other arms negative rewards.
        for arm in 1..ac.bandit().num_arms() {
            for _ in 0..15 {
                ac.absorb(arm, -1.0);
            }
        }

        // total_absorbed = 15 * num_arms. We need it to be a multiple of check_interval=20.
        // Absorb a few more to reach a multiple of 20.
        while !ac.should_compress() {
            ac.absorb(0, 1.0);
        }

        let promoted = ac.compress();

        assert!(!promoted.is_empty(), "arm 0 should be promoted");
        assert!(promoted.contains(&0), "arm 0 should be in promoted list");
        assert!(ac.promoted_winner().is_some());
    }

    #[test]
    fn test_absorb_compress_no_promotion_below_visits() {
        let bandit = make_test_bandit();
        let config = RuliologyPromoteConfig {
            min_visits: 100,
            payoff_threshold: 0.5,
            max_promotions: 5,
            check_interval: 10,
        };
        let mut ac = RuliologyAbsorbCompress::new(bandit, config);

        // Only 5 visits — below min_visits.
        for _ in 0..5 {
            ac.absorb(0, 1.0);
        }

        let promoted = ac.compress();
        assert!(
            promoted.is_empty(),
            "no arm should be promoted with <100 visits"
        );
    }

    #[test]
    fn test_absorb_compress_max_promotions_limit() {
        let bandit = make_test_bandit();
        let config = RuliologyPromoteConfig {
            min_visits: 5,
            payoff_threshold: 0.3,
            max_promotions: 2,
            check_interval: 10,
        };
        let mut ac = RuliologyAbsorbCompress::new(bandit, config);

        // Give all arms high rewards.
        for arm in 0..ac.bandit().num_arms() {
            for _ in 0..10 {
                ac.absorb(arm, 1.0);
            }
        }

        let promoted = ac.compress();
        assert!(
            promoted.len() <= 2,
            "should promote at most 2 arms, got {}",
            promoted.len()
        );
    }
}

// TL;DR: RuliologyArm (UCB1 arm) + RuliologyBandit (Pareto-filtered selection) + RuliologyAbsorbCompress (promote stable winners).
