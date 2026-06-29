//! WealthBanditPruner — Economic Bandit Arms via Hayek Market Selection (Modelless)
//!
//! A BanditPruner variant that replaces UCB1's statistical optimism with wealth-based
//! economic selection. Arms accumulate wealth from rewards, spend wealth on bids, and
//! bankrupt arms get "rebirthed" from successful arm mutations.
//!
//! # Architecture
//!
//! - [`WealthArm`] — per-arm state: wealth, Q-value, pulls
//! - [`WealthBanditPruner`] — wraps any `ScreeningPruner`, adds wealth-based relevance
//! - [`WealthPrunerConfig`] — configuration with defaults
//!
//! # Insight
//!
//! Economic selection > statistical optimism for exploration: arms that earn more
//! get more opportunity, bankrupt arms get replaced with mutations of successful arms.
//! This is the Economy of Minds (EoM) insight distilled to modelless Rust.
//!
//! **Feature gate:** `wealth_pruner` (depends on `bandit`)

use katgpt_speculative::ScreeningPruner;
use katgpt_types::Rng;

// ── WealthArm ─────────────────────────────────────────────────────

/// Per-arm state for wealth-based bandit selection.
///
/// Each arm tracks its accumulated wealth, Q-value estimate, and pull count.
/// Wealth acts as an economic resource: positive wealth enables continued play,
/// negative wealth triggers bankruptcy and rebirth.
#[derive(Debug, Clone)]
pub struct WealthArm {
    /// Accumulated economic wealth. Replenished by rewards, drained by bids/rent.
    pub wealth: f64,
    /// Running average reward (Q-value estimate).
    pub q_value: f64,
    /// Total reward accumulated.
    pub total_reward: f64,
    /// Number of times this arm has been pulled.
    pub pulls: u32,
}

impl WealthArm {
    /// Create a new arm with the given initial wealth.
    pub fn new(initial_wealth: f64) -> Self {
        Self {
            wealth: initial_wealth,
            q_value: 0.0,
            total_reward: 0.0,
            pulls: 0,
        }
    }

    /// Check if this arm is bankrupt (wealth < 0).
    #[inline]
    pub fn is_bankrupt(&self) -> bool {
        self.wealth < 0.0
    }

    /// Create a reborn arm from a parent arm.
    ///
    /// The new arm inherits the parent's Q-value with Gaussian noise σ,
    /// and resets wealth to the initial value. Pull count and total reward reset.
    pub fn rebirth_from(parent: &Self, sigma: f64, initial_wealth: f64, rng: &mut Rng) -> Self {
        let noise = if sigma > 0.0 {
            rng.normal() as f64 * sigma
        } else {
            0.0
        };
        Self {
            wealth: initial_wealth,
            q_value: (parent.q_value + noise).clamp(0.0, 1.0),
            total_reward: 0.0,
            pulls: 0,
        }
    }
}

// ── WealthPrunerConfig ────────────────────────────────────────────

/// Configuration for `WealthBanditPruner`.
#[derive(Debug, Clone)]
pub struct WealthPrunerConfig {
    /// Initial wealth for each arm. Default: 0.5.
    pub initial_wealth: f64,
    /// Bid scaling factor: relevance = q_value + wealth * bid_alpha. Default: 0.1.
    pub bid_alpha: f64,
    /// Rent charged per interval. Default: 0.0 (disabled).
    pub rent: f64,
    /// Charge rent every N episodes. 0 = disabled. Default: 0.
    pub rent_interval: u32,
    /// Standard deviation for rebirth noise. Default: 0.1.
    pub rebirth_sigma: f64,
    /// Enable chain credit assignment. Default: false.
    pub use_chain_credit: bool,
    /// Chain credit window size. Default: 3.
    pub chain_window_size: usize,
}

impl Default for WealthPrunerConfig {
    fn default() -> Self {
        Self {
            initial_wealth: 0.5,
            bid_alpha: 0.1,
            rent: 0.0,
            rent_interval: 0,
            rebirth_sigma: 0.1,
            use_chain_credit: false,
            chain_window_size: 3,
        }
    }
}

// ── WealthBanditPruner ────────────────────────────────────────────

/// Wealth-based bandit pruner wrapping any `ScreeningPruner`.
///
/// Replaces UCB1's optimism bonus with economic selection:
/// arms with more wealth get higher relevance, bankrupt arms are rebirthed
/// from the richest arm's Q-value (with Gaussian perturbation).
pub struct WealthBanditPruner<P: ScreeningPruner> {
    /// Per-arm wealth state.
    arms: Vec<WealthArm>,
    /// Inner domain pruner providing base relevance scores.
    inner: P,
    /// Configuration.
    config: WealthPrunerConfig,
    /// Episode counter (for rent interval).
    episode_count: u32,
    /// Total rebirth events for statistics.
    rebirth_count: u32,
    /// Optional chain credit assigner state (arm indices).
    chain_window: Vec<usize>,
}

impl<P: ScreeningPruner> WealthBanditPruner<P> {
    /// Create a new wealth bandit pruner.
    pub fn new(inner: P, num_arms: usize, config: WealthPrunerConfig) -> Self {
        let arms = vec![WealthArm::new(config.initial_wealth); num_arms];
        Self {
            arms,
            inner,
            chain_window: if config.use_chain_credit {
                Vec::with_capacity(config.chain_window_size)
            } else {
                Vec::new()
            },
            config,
            episode_count: 0,
            rebirth_count: 0,
        }
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    /// Get a reference to the arm at the given index.
    pub fn arm(&self, idx: usize) -> Option<&WealthArm> {
        self.arms.get(idx)
    }

    /// Get a mutable reference to the arm at the given index.
    pub fn arm_mut(&mut self, idx: usize) -> Option<&mut WealthArm> {
        self.arms.get_mut(idx)
    }

    /// Index of the arm with the highest Q-value.
    pub fn best_arm(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.q_value
                    .partial_cmp(&b.q_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Index of the arm with the highest wealth.
    pub fn richest_arm(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.wealth
                    .partial_cmp(&b.wealth)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Total rebirth events since creation.
    pub fn rebirth_count(&self) -> u32 {
        self.rebirth_count
    }

    /// Total episodes run.
    pub fn episode_count(&self) -> u32 {
        self.episode_count
    }

    /// Wealth-based relevance for a single arm.
    ///
    /// Returns `q_value + wealth * bid_alpha`. This replaces UCB1's optimism bonus
    /// with economic selection: richer arms get more opportunity.
    #[inline]
    pub fn wealth_score(&self, arm: usize) -> f64 {
        match self.arms.get(arm) {
            Some(a) => a.q_value + a.wealth * self.config.bid_alpha,
            None => 0.0,
        }
    }

    /// Update an arm with a reward observation.
    ///
    /// Updates Q-value incrementally, adds reward to arm wealth, and increments episode count.
    pub fn update(&mut self, arm: usize, reward: f64) {
        if arm >= self.arms.len() {
            return;
        }
        let a = &mut self.arms[arm];
        a.pulls += 1;
        let n = a.pulls as f64;
        // Incremental mean update
        a.q_value += (reward - a.q_value) / n;
        a.total_reward += reward;
        // Wealth accumulation
        a.wealth += reward;
    }

    /// Update with chain credit assignment: split reward across recent arms in window.
    pub fn update_chain(&mut self, reward: f64) {
        if self.chain_window.is_empty() {
            return;
        }
        // Find unique arms in window
        let mut unique_arms: Vec<usize> = self.chain_window.clone();
        unique_arms.sort_unstable();
        unique_arms.dedup();

        let credit = reward / unique_arms.len() as f64;
        for &arm in &unique_arms {
            if arm < self.arms.len() {
                let a = &mut self.arms[arm];
                a.pulls += 1;
                let n = a.pulls as f64;
                a.q_value += (credit - a.q_value) / n;
                a.total_reward += credit;
                a.wealth += credit;
            }
        }
    }

    /// Record an arm selection for chain credit tracking.
    pub fn record_arm(&mut self, arm: usize) {
        if !self.config.use_chain_credit {
            return;
        }
        self.chain_window.push(arm);
        if self.chain_window.len() > self.config.chain_window_size {
            self.chain_window.remove(0);
        }
    }

    /// Charge rent from all arms (periodic wealth drain).
    ///
    /// Returns the number of arms that went bankrupt from this charge.
    pub fn charge_rent(&mut self) -> u32 {
        if self.config.rent <= 0.0 {
            return 0;
        }
        let mut bankrupt_count = 0u32;
        for arm in &mut self.arms {
            arm.wealth -= self.config.rent;
            if arm.is_bankrupt() {
                bankrupt_count += 1;
            }
        }
        bankrupt_count
    }

    /// Rebirth all bankrupt arms from the richest arm's Q-value.
    ///
    /// Returns the number of arms rebirthed.
    pub fn rebirth_bankrupt_arms(&mut self, rng: &mut Rng) -> u32 {
        let richest = self.richest_arm();
        let parent = self.arms[richest].clone();

        let sigma = self.config.rebirth_sigma;
        let initial = self.config.initial_wealth;
        let mut count = 0u32;

        for arm in &mut self.arms {
            if arm.is_bankrupt() {
                *arm = WealthArm::rebirth_from(&parent, sigma, initial, rng);
                count += 1;
            }
        }

        self.rebirth_count += count;
        count
    }

    /// End-of-episode maintenance: increment episode, charge rent if due, rebirth.
    ///
    /// Call after each episode's reward update.
    pub fn end_episode(&mut self, rng: &mut Rng) {
        self.episode_count += 1;

        // Charge rent if interval is configured and due
        if self.config.rent_interval > 0 && self.episode_count.is_multiple_of(self.config.rent_interval) {
            self.charge_rent();
        }

        // Rebirth bankrupt arms
        self.rebirth_bankrupt_arms(rng);
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable access to the inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
}

impl<P: ScreeningPruner> ScreeningPruner for WealthBanditPruner<P> {
    /// Wealth-based relevance: `domain_score * wealth_score`, clamped to [0, 1].
    ///
    /// The domain score comes from the inner pruner. The wealth score is
    /// `q_value + wealth * bid_alpha`, providing economic selection pressure.
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if token_idx >= self.arms.len() {
            return 0.0;
        }

        let domain = self.inner.relevance(depth, token_idx, parent_tokens);
        if domain <= 0.0 {
            return 0.0;
        }

        // Cold start: no pulls yet, use domain only
        if self.arms[token_idx].pulls == 0 {
            return domain;
        }

        let ws = self.wealth_score(token_idx);
        // Harmonic blend: domain × wealth_score, clamped
        (domain * ws as f32).clamp(0.0, 1.0)
    }
}

// ── ChainCreditAssigner ───────────────────────────────────────────

/// Rolling window credit assigner for trajectory reward splitting.
///
/// Records recent arm selections in a fixed-size window and distributes
/// reward equally among unique arms in the window. O(W) per reward
/// where W = window_size.
#[derive(Debug, Clone)]
pub struct ChainCreditAssigner {
    /// Rolling window of arm indices.
    window: Vec<usize>,
    /// Maximum window size.
    window_size: usize,
}

impl ChainCreditAssigner {
    /// Create a new chain credit assigner with the given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size: window_size.max(1),
        }
    }

    /// Record an arm selection. Trims window to max size.
    pub fn record_arm(&mut self, arm: usize) {
        self.window.push(arm);
        if self.window.len() > self.window_size {
            self.window.remove(0);
        }
    }

    /// Distribute reward across unique arms in the window.
    ///
    /// Returns true if any arm was credited, false if window is empty.
    pub fn distribute_reward(&self, reward: f64, arms: &mut [WealthArm]) -> bool {
        if self.window.is_empty() {
            return false;
        }

        // Find unique arms
        let mut unique: Vec<usize> = self.window.clone();
        unique.sort_unstable();
        unique.dedup();

        let credit = reward / unique.len() as f64;
        for &arm in &unique {
            if arm < arms.len() {
                let a = &mut arms[arm];
                a.pulls += 1;
                let n = a.pulls as f64;
                a.q_value += (credit - a.q_value) / n;
                a.total_reward += credit;
                a.wealth += credit;
            }
        }
        true
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.window.clear();
    }

    /// Current window contents.
    pub fn window(&self) -> &[usize] {
        &self.window
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_speculative::NoScreeningPruner;

    #[test]
    fn test_wealth_arm_creation() {
        let arm = WealthArm::new(0.5);
        assert!(!arm.is_bankrupt());
        assert_eq!(arm.wealth, 0.5);
        assert_eq!(arm.q_value, 0.0);
        assert_eq!(arm.pulls, 0);
    }

    #[test]
    fn test_wealth_arm_bankruptcy() {
        let mut arm = WealthArm::new(0.5);
        assert!(!arm.is_bankrupt());
        arm.wealth = -0.1;
        assert!(arm.is_bankrupt());
    }

    #[test]
    fn test_wealth_arm_rebirth() {
        let parent = WealthArm {
            wealth: 10.0,
            q_value: 0.8,
            total_reward: 50.0,
            pulls: 100,
        };
        let mut rng = Rng::new(42);
        let child = WealthArm::rebirth_from(&parent, 0.0, 0.5, &mut rng);
        // With sigma=0, child inherits parent's Q-value exactly
        assert_eq!(child.q_value, 0.8);
        assert_eq!(child.wealth, 0.5);
        assert_eq!(child.pulls, 0);
        assert_eq!(child.total_reward, 0.0);
    }

    #[test]
    fn test_wealth_arm_rebirth_with_noise() {
        let parent = WealthArm {
            wealth: 10.0,
            q_value: 0.5,
            total_reward: 50.0,
            pulls: 100,
        };
        let mut rng = Rng::new(42);
        let child = WealthArm::rebirth_from(&parent, 0.1, 0.5, &mut rng);
        // With sigma=0.1, Q-value should be perturbed but close to parent
        assert!(child.q_value >= 0.0 && child.q_value <= 1.0);
        assert_eq!(child.wealth, 0.5);
    }

    #[test]
    fn test_wealth_score_formula() {
        let config = WealthPrunerConfig {
            initial_wealth: 1.0,
            bid_alpha: 0.2,
            ..Default::default()
        };
        let pruner = WealthBanditPruner::new(NoScreeningPruner, 2, config);
        // After creation, q_value=0, wealth=1.0, so score = 0.0 + 1.0 * 0.2 = 0.2
        assert!((pruner.wealth_score(0) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_update_increments_pulls_and_updates_q() {
        let mut pruner =
            WealthBanditPruner::new(NoScreeningPruner, 3, WealthPrunerConfig::default());
        pruner.update(0, 0.8);
        pruner.update(0, 0.6);
        assert_eq!(pruner.arm(0).unwrap().pulls, 2);
        // Q-value should be average: (0.8 + 0.6) / 2 = 0.7
        assert!((pruner.arm(0).unwrap().q_value - 0.7).abs() < 1e-10);
        // Wealth = initial_wealth(0.5) + 0.8 + 0.6 = 1.9
        assert!((pruner.arm(0).unwrap().wealth - 1.9).abs() < 1e-10);
    }

    #[test]
    fn test_rebirth_bankrupt_arms() {
        let config = WealthPrunerConfig {
            initial_wealth: 0.5,
            rebirth_sigma: 0.0, // no noise for deterministic test
            ..Default::default()
        };
        let mut pruner = WealthBanditPruner::new(NoScreeningPruner, 3, config);

        // Make arm 0 rich with high Q
        pruner.update(0, 0.9);
        pruner.update(0, 0.9);
        pruner.arms[0].q_value = 0.9;

        // Make arms 1 and 2 bankrupt
        pruner.arms[1].wealth = -1.0;
        pruner.arms[2].wealth = -0.5;

        let mut rng = Rng::new(42);
        let count = pruner.rebirth_bankrupt_arms(&mut rng);
        assert_eq!(count, 2);
        // Both should inherit Q from arm 0 (richest)
        assert!((pruner.arms[1].q_value - 0.9).abs() < 1e-10);
        assert!((pruner.arms[2].q_value - 0.9).abs() < 1e-10);
        // Wealth reset to initial
        assert!((pruner.arms[1].wealth - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rent_charge_triggers_bankruptcy() {
        let config = WealthPrunerConfig {
            initial_wealth: 0.1,
            rent: 1.0,
            rent_interval: 1,
            ..Default::default()
        };
        let mut pruner = WealthBanditPruner::new(NoScreeningPruner, 3, config);

        // Charge rent once: all arms go from 0.1 to -0.9
        let bankrupt = pruner.charge_rent();
        assert_eq!(bankrupt, 3);
        assert!(pruner.arms[0].is_bankrupt());
    }

    #[test]
    fn test_richest_arm() {
        let mut pruner =
            WealthBanditPruner::new(NoScreeningPruner, 3, WealthPrunerConfig::default());
        pruner.update(0, 0.5);
        pruner.update(1, 0.9);
        pruner.update(2, 0.3);
        assert_eq!(pruner.richest_arm(), 1);
    }

    #[test]
    fn test_best_arm() {
        let mut pruner =
            WealthBanditPruner::new(NoScreeningPruner, 3, WealthPrunerConfig::default());
        pruner.update(0, 0.3);
        pruner.update(1, 0.9);
        pruner.update(2, 0.5);
        assert_eq!(pruner.best_arm(), 1);
    }

    #[test]
    fn test_screening_pruner_cold_start() {
        let pruner = WealthBanditPruner::new(NoScreeningPruner, 3, WealthPrunerConfig::default());
        // No pulls yet — should return domain score (1.0 from NoScreeningPruner)
        assert_eq!(pruner.relevance(0, 0, &[]), 1.0);
    }

    #[test]
    fn test_screening_pruner_with_pulls() {
        let mut pruner = WealthBanditPruner::new(
            NoScreeningPruner,
            3,
            WealthPrunerConfig {
                bid_alpha: 0.0, // zero alpha for deterministic test
                ..Default::default()
            },
        );
        pruner.update(0, 0.5);
        // With bid_alpha=0, wealth_score = q_value = 0.5
        // relevance = domain(1.0) * 0.5 = 0.5
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_end_episode_with_rent() {
        let config = WealthPrunerConfig {
            initial_wealth: 0.1,
            rent: 1.0,
            rent_interval: 2,
            rebirth_sigma: 0.0,
            ..Default::default()
        };
        let mut pruner = WealthBanditPruner::new(NoScreeningPruner, 2, config);

        let mut rng = Rng::new(42);
        pruner.end_episode(&mut rng); // ep 1 — no rent
        assert!(!pruner.arms[0].is_bankrupt());

        pruner.end_episode(&mut rng); // ep 2 — rent charged, then rebirthed
        // Arms should have been rebirthed (bankrupt after rent, then rebirthed)
        assert!(!pruner.arms[0].is_bankrupt());
        assert_eq!(pruner.rebirth_count(), 2);
    }

    // ── ChainCreditAssigner Tests ──────────────────────────────

    #[test]
    fn test_chain_credit_basic() {
        let mut cca = ChainCreditAssigner::new(3);
        cca.record_arm(0);
        cca.record_arm(1);
        cca.record_arm(2);

        let mut arms = vec![
            WealthArm::new(0.5),
            WealthArm::new(0.5),
            WealthArm::new(0.5),
        ];
        let distributed = cca.distribute_reward(0.9, &mut arms);
        assert!(distributed);
        // Each arm gets 0.9/3 = 0.3 credit
        assert!((arms[0].total_reward - 0.3).abs() < 1e-10);
        assert!((arms[1].total_reward - 0.3).abs() < 1e-10);
        assert!((arms[2].total_reward - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_chain_credit_dedup() {
        let mut cca = ChainCreditAssigner::new(4);
        cca.record_arm(0);
        cca.record_arm(0);
        cca.record_arm(1);

        let mut arms = vec![WealthArm::new(0.5), WealthArm::new(0.5)];
        cca.distribute_reward(1.0, &mut arms);
        // Unique arms: {0, 1} → each gets 0.5
        assert!((arms[0].total_reward - 0.5).abs() < 1e-10);
        assert!((arms[1].total_reward - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_chain_credit_sum_equals_total() {
        let mut cca = ChainCreditAssigner::new(5);
        for arm in [2, 0, 3, 0, 4] {
            cca.record_arm(arm);
        }

        let mut arms = vec![WealthArm::new(0.5); 5];
        let reward = 1.0;
        cca.distribute_reward(reward, &mut arms);

        // Sum of rewards should equal total reward
        let sum: f64 = arms.iter().map(|a| a.total_reward).sum();
        assert!((sum - reward).abs() < 1e-10);
    }

    #[test]
    fn test_chain_credit_empty_window() {
        let cca = ChainCreditAssigner::new(3);
        let mut arms = vec![WealthArm::new(0.5)];
        assert!(!cca.distribute_reward(1.0, &mut arms));
    }

    #[test]
    fn test_chain_credit_window_trim() {
        let mut cca = ChainCreditAssigner::new(2);
        cca.record_arm(0);
        cca.record_arm(1);
        cca.record_arm(2);
        assert_eq!(cca.window(), &[1, 2]);
    }

    // ── Convergence Test ───────────────────────────────────────

    #[test]
    fn test_wealth_pruner_convergence() {
        // K=10 arms, one optimal arm at index 7 with mean 0.9
        let arm_means: [f64; 10] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.4, 0.9, 0.3, 0.2];
        let optimal = 7;
        let config = WealthPrunerConfig {
            initial_wealth: 0.5,
            bid_alpha: 0.1,
            rent: 0.0,
            rent_interval: 0,
            rebirth_sigma: 0.1,
            use_chain_credit: false,
            chain_window_size: 3,
        };

        let mut pruner = WealthBanditPruner::new(NoScreeningPruner, 10, config);
        let mut rng = Rng::new(42);

        for _ in 0..1000 {
            // Select arm: unvisited first, then highest wealth_score
            let mut best = 0;
            let mut best_score = f64::NEG_INFINITY;
            for i in 0..10 {
                let score = if pruner.arm(i).unwrap().pulls == 0 {
                    f64::INFINITY // explore unvisited first
                } else {
                    pruner.wealth_score(i)
                };
                if score > best_score {
                    best_score = score;
                    best = i;
                }
            }

            // Simulate reward (Bernoulli)
            let reward = if rng.uniform() < arm_means[best] as f32 {
                1.0
            } else {
                0.0
            };

            pruner.update(best, reward);
            pruner.end_episode(&mut rng);
        }

        let best = pruner.best_arm();
        // WealthPruner should find the optimal arm in 1000 episodes
        assert!(
            best == optimal,
            "WealthPruner found arm {best} as best, expected {optimal}. Q-values: {:?}",
            pruner.arms.iter().map(|a| a.q_value).collect::<Vec<_>>()
        );
    }
}
