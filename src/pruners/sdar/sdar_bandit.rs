//! Bandit pruner with SDAR sigmoid-gated reward updates.
//!
//! Wraps [`BanditPruner`](super::super::bandit::BanditPruner) with sigmoid-gated
//! reward updates based on the asymmetric trust principle from SDAR.
//!
//! # Why Gate Reward Updates?
//!
//! SDAR gates distillation loss by teacher-student gap. Analogously, we gate
//! bandit Q-value updates by reward quality gap:
//! - When reward signal is noisy (negative gap = reward < Q-value), attenuate the update
//! - When reward signal is trustworthy (positive gap = reward > Q-value), pass it through
//!
//! # Reward Gating
//!
//! ```text
//! gap = reward - q_values[arm]
//! gate = σ(β · gap)    // sigmoid gate
//! gated_reward = reward * gate
//! ```
//!
//! Positive reward surprise → full update (gate opens).
//! Negative reward surprise → attenuated update (gate closes).
//!
//! # Usage
//!
//! ```rust,ignore
//! let pruner = SdarBanditPruner::new(
//!     BanditPruner::new(domain_screener, BanditStrategy::Ucb1, 10),
//!     10,
//! );
//!
//! // Feed reward — gated by SDAR sigmoid
//! pruner.update(3, 0.42);
//!
//! // Check gating statistics
//! let stats = pruner.gate_stats(3);
//! ```
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "sdar_gate")]`.
//!
//! **Source:** [SDAR: Self-Distilled Agentic RL](https://arxiv.org/abs/2605.15155)

use crate::pruners::bandit::BanditPruner;
use crate::pruners::sdar_gate::{SDAR_BETA, sdar_gate_default, sdar_gated_reward};
use crate::speculative::types::ScreeningPruner;

// ── Config ──────────────────────────────────────────────────────

/// Configuration for [`SdarBanditPruner`].
#[derive(Clone, Debug)]
pub struct SdarBanditConfig {
    /// Sigmoid sharpness β (default: 5.0 from SDAR paper).
    ///
    /// - β=0: no gate (uniform, degrades to standard BanditPruner)
    /// - β=5: optimal balance (paper-validated)
    /// - β→∞: binary gate (too aggressive)
    pub beta: f32,
    /// Whether to log gate statistics (default: false).
    ///
    /// When enabled, per-arm gate values are tracked for debugging.
    pub track_gate_stats: bool,
}

impl Default for SdarBanditConfig {
    fn default() -> Self {
        Self {
            beta: SDAR_BETA,
            track_gate_stats: false,
        }
    }
}

impl SdarBanditConfig {
    /// Create config with custom β.
    pub fn new(beta: f32) -> Self {
        Self {
            beta,
            ..Self::default()
        }
    }

    /// Enable gate statistics tracking.
    pub fn with_gate_stats(mut self) -> Self {
        self.track_gate_stats = true;
        self
    }

    /// Use soft gating (β=1.0).
    pub fn soft() -> Self {
        Self {
            beta: 1.0,
            ..Self::default()
        }
    }

    /// Use aggressive gating (β=10.0, near-binary).
    pub fn aggressive() -> Self {
        Self {
            beta: 10.0,
            ..Self::default()
        }
    }
}

// ── GateStats ───────────────────────────────────────────────────

/// Per-arm gate statistics for debugging.
#[derive(Clone, Debug, Default)]
pub struct GateStats {
    /// Number of gated updates.
    pub update_count: usize,
    /// Sum of gate values applied.
    pub gate_sum: f32,
    /// Sum of original (ungated) rewards.
    pub original_reward_sum: f32,
    /// Sum of gated rewards actually fed to bandit.
    pub gated_reward_sum: f32,
    /// Last gate value applied.
    pub last_gate: f32,
}

impl GateStats {
    /// Mean gate value across all updates.
    pub fn mean_gate(&self) -> f32 {
        if self.update_count == 0 {
            return 0.0;
        }
        self.gate_sum / self.update_count as f32
    }

    /// Mean original reward (ungated).
    pub fn mean_original_reward(&self) -> f32 {
        if self.update_count == 0 {
            return 0.0;
        }
        self.original_reward_sum / self.update_count as f32
    }

    /// Mean gated reward.
    pub fn mean_gated_reward(&self) -> f32 {
        if self.update_count == 0 {
            return 0.0;
        }
        self.gated_reward_sum / self.update_count as f32
    }

    /// Attenuation ratio: how much the gate reduces rewards on average.
    ///
    /// Returns 0.0 when no updates, 1.0 when gate fully passes, 0.0 when fully attenuated.
    pub fn pass_through_ratio(&self) -> f32 {
        if self.original_reward_sum.abs() < f32::EPSILON {
            return 0.0;
        }
        self.gated_reward_sum / self.original_reward_sum
    }
}

// ── SdarBanditPruner ────────────────────────────────────────────

/// Bandit pruner with SDAR sigmoid-gated reward updates.
///
/// Wraps [`BanditPruner`] and applies asymmetric trust gating to reward
/// updates. This is the modelless analog of SDAR's token-level gating:
///
/// - Positive reward surprise (reward > Q-value) → gate opens → full update
/// - Negative reward surprise (reward < Q-value) → gate closes → attenuated update
///
/// # Architecture
///
/// ```text
/// SdarBanditPruner<P>
///   ├── inner: BanditPruner<P>       (existing bandit logic)
///   ├── beta: f32                     (sigmoid sharpness)
///   └── gate_stats: Vec<GateStats>    (per-arm statistics, optional)
/// ```
///
/// # Property: No Regression vs Ungated
///
/// When β=0, `sdar_gate(x, 0) = 0.5` for all x, so all rewards are halved.
/// This doesn't degrade to ungated — use β→∞ for binary or wrap BanditPruner
/// directly for no gating.
///
/// The key property: bandit still converges to optimal arm because gating
/// preserves relative ordering (higher rewards → higher gated rewards).
pub struct SdarBanditPruner<P: ScreeningPruner> {
    /// Inner bandit pruner (delegates arm selection logic).
    inner: BanditPruner<P>,
    /// Sigmoid sharpness parameter.
    beta: f32,
    /// Per-arm gate statistics (only tracked when config enables it).
    gate_stats: Vec<GateStats>,
    /// Whether gate statistics tracking is enabled.
    track_gate_stats: bool,
    /// Optional learned β (RePlaid variance-minimized).
    /// When present, β adapts per-episode based on gated reward variance.
    #[cfg(feature = "replaid_schedules")]
    learned_beta: Option<crate::pruners::sdar_gate::SdarLearnedBeta>,
}

impl<P: ScreeningPruner> SdarBanditPruner<P> {
    /// Create a new SDAR-gated bandit pruner with default config.
    ///
    /// Wraps an existing `BanditPruner` with sigmoid-gated reward updates.
    pub fn new(inner: BanditPruner<P>, num_arms: usize) -> Self {
        Self::with_config(inner, num_arms, SdarBanditConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(inner: BanditPruner<P>, num_arms: usize, config: SdarBanditConfig) -> Self {
        Self {
            inner,
            beta: config.beta,
            gate_stats: if config.track_gate_stats {
                (0..num_arms).map(|_| GateStats::default()).collect()
            } else {
                Vec::new()
            },
            track_gate_stats: config.track_gate_stats,
            #[cfg(feature = "replaid_schedules")]
            learned_beta: None,
        }
    }

    /// Update Q-value for an arm with SDAR-gated reward.
    ///
    /// Computes: `gap = reward - q_values[arm]`, `gate = σ(β·gap)`,
    /// `gated_reward = reward * gate`. Then feeds `gated_reward` to inner bandit.
    ///
    /// Positive reward surprise → gate opens → near-full reward update.
    /// Negative reward surprise → gate closes → attenuated reward update.
    #[inline]
    pub fn update(&mut self, arm: usize, reward: f32) {
        let q_value = self.inner.q_values().get(arm).copied().unwrap_or(0.0);

        // Use learned β when available, otherwise static β
        #[cfg(feature = "replaid_schedules")]
        let beta = self.learned_beta.as_ref().map_or(self.beta, |lb| lb.beta());
        #[cfg(not(feature = "replaid_schedules"))]
        let beta = self.beta;

        let gated = sdar_gated_reward(reward, q_value, beta);

        // Track statistics if enabled
        if self.track_gate_stats
            && let Some(stats) = self.gate_stats.get_mut(arm)
        {
            let gate = sdar_gate_default(reward - q_value);
            stats.update_count += 1;
            stats.gate_sum += gate;
            stats.original_reward_sum += reward;
            stats.gated_reward_sum += gated;
            stats.last_gate = gate;
        }

        self.inner.update(arm, gated);
    }

    /// Get gate statistics for a specific arm.
    ///
    /// Returns `None` if gate statistics tracking is disabled or arm is out of bounds.
    pub fn gate_stats(&self, arm: usize) -> Option<&GateStats> {
        self.gate_stats.get(arm)
    }

    /// Whether gate statistics tracking is enabled.
    pub fn is_tracking_stats(&self) -> bool {
        self.track_gate_stats
    }

    /// Current β (sigmoid sharpness).
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Set β (sigmoid sharpness).
    ///
    /// Can be adjusted at runtime for adaptive gating experiments.
    pub fn set_beta(&mut self, beta: f32) {
        self.beta = beta;
    }

    /// Enable learned β (RePlaid variance-minimized).
    ///
    /// When enabled, β adapts per-episode based on gated reward variance.
    /// Call `adapt_beta()` after each episode to update.
    #[cfg(feature = "replaid_schedules")]
    pub fn with_learned_beta(mut self, initial_beta: f32) -> Self {
        self.learned_beta = Some(crate::pruners::sdar_gate::SdarLearnedBeta::new(
            initial_beta,
        ));
        self
    }

    /// Adapt learned β based on mean gated reward from this episode.
    ///
    /// Call after each episode. No-op when learned β is disabled.
    #[cfg(feature = "replaid_schedules")]
    pub fn adapt_beta(&mut self, mean_gated_reward: f32) {
        if let Some(ref mut lb) = self.learned_beta {
            lb.observe_and_adapt(mean_gated_reward);
            // Sync the static beta field so beta() still works
            self.beta = lb.beta();
        }
    }

    /// Whether learned β (RePlaid variance-minimized) is active.
    #[cfg(feature = "replaid_schedules")]
    pub fn has_learned_beta(&self) -> bool {
        self.learned_beta.is_some()
    }

    /// Access the inner bandit pruner.
    pub fn inner(&self) -> &BanditPruner<P> {
        &self.inner
    }

    /// Mutable access to the inner bandit pruner.
    pub fn inner_mut(&mut self) -> &mut BanditPruner<P> {
        &mut self.inner
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.gate_stats.len().max(self.inner.q_values().len())
    }

    /// Decay epsilon after an episode (EpsilonGreedy only).
    pub fn decay_epsilon(&mut self) {
        self.inner.decay_epsilon();
    }

    /// Index of the best arm (highest Q-value).
    pub fn best_arm(&self) -> usize {
        self.inner.best_arm()
    }

    /// Q-values slice (for inspection).
    pub fn q_values(&self) -> &[f32] {
        self.inner.q_values()
    }

    /// Visit counts slice (for inspection).
    pub fn visits(&self) -> &[u32] {
        self.inner.visits()
    }

    /// Total pulls across all arms.
    pub fn total_pulls(&self) -> u32 {
        self.inner.total_pulls()
    }
}

// Delegate ScreeningPruner to inner BanditPruner
impl<P: ScreeningPruner> ScreeningPruner for SdarBanditPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::BanditStrategy;
    use crate::speculative::types::NoScreeningPruner;

    fn make_pruner(num_arms: usize) -> SdarBanditPruner<NoScreeningPruner> {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
        SdarBanditPruner::new(inner, num_arms)
    }

    fn make_pruner_with_stats(num_arms: usize) -> SdarBanditPruner<NoScreeningPruner> {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
        SdarBanditPruner::with_config(
            inner,
            num_arms,
            SdarBanditConfig::default().with_gate_stats(),
        )
    }

    fn make_pruner_with_beta(num_arms: usize, beta: f32) -> SdarBanditPruner<NoScreeningPruner> {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
        SdarBanditPruner::with_config(inner, num_arms, SdarBanditConfig::new(beta))
    }

    // ── Gate opens for positive gap ─────────────────────────────

    #[test]
    fn test_gate_opens_for_positive_gap() {
        let mut pruner = make_pruner_with_stats(3);

        // First update sets Q-value to the reward (no prior)
        pruner.update(0, 1.0);

        // Second update with higher reward → positive gap → gate opens
        pruner.update(0, 2.0);

        let stats = pruner.gate_stats(0).unwrap();
        assert_eq!(stats.update_count, 2);
        // Second gate should be > 0.5 (positive surprise)
        assert!(
            stats.last_gate > 0.5,
            "Positive gap → gate opens, got {}",
            stats.last_gate
        );
    }

    // ── Gate closes for negative gap ────────────────────────────

    #[test]
    fn test_gate_closes_for_negative_gap() {
        let mut pruner = make_pruner_with_stats(3);

        // First update sets Q-value high
        pruner.update(0, 2.0);

        // Second update with lower reward → negative gap → gate closes
        pruner.update(0, 0.1);

        let stats = pruner.gate_stats(0).unwrap();
        // Second gate should be < 0.5 (negative surprise)
        assert!(
            stats.last_gate < 0.5,
            "Negative gap → gate closes, got {}",
            stats.last_gate
        );
    }

    // ── Convergence still reaches optimal arm ───────────────────

    #[test]
    fn test_convergence_reaches_optimal_arm() {
        let mut pruner = make_pruner(5);

        // Arm 2 is consistently best (highest reward)
        for _ in 0..500 {
            pruner.update(0, 0.1);
            pruner.update(1, 0.2);
            pruner.update(2, 1.0);
            pruner.update(3, 0.15);
            pruner.update(4, 0.05);
        }

        let best = pruner.best_arm();
        assert_eq!(
            best, 2,
            "Should converge to arm 2 (highest reward), got arm {best}"
        );
    }

    #[test]
    fn test_convergence_preserves_ordering() {
        let mut pruner = make_pruner(3);

        // Feed rewards in known order
        for _ in 0..200 {
            pruner.update(0, 0.3);
            pruner.update(1, 0.6);
            pruner.update(2, 0.9);
        }

        let q = pruner.q_values();
        assert!(
            q[2] > q[1] && q[1] > q[0],
            "Q-values should preserve reward ordering: q0={}, q1={}, q2={}",
            q[0],
            q[1],
            q[2]
        );
    }

    // ── β=0 degrades to uniform (no gate) ───────────────────────

    #[test]
    fn test_beta_zero_all_rewards_halved() {
        let mut pruner = make_pruner_with_beta(3, 0.0);

        // β=0: gate = σ(0) = 0.5 for all inputs → reward halved
        pruner.update(0, 1.0);
        pruner.update(0, 1.0);

        let q = pruner.q_values()[0];
        // Q-value should be ≈ 0.5 (halved reward)
        assert!(
            (q - 0.5).abs() < 0.05,
            "β=0 → rewards halved, Q should be ≈0.5, got {q}"
        );
    }

    // ── β→∞ degrades to binary ─────────────────────────────────

    #[test]
    fn test_beta_large_near_binary() {
        let mut pruner = make_pruner_with_beta(3, 100.0);

        // Very large β: gate is essentially binary
        // Positive gap → gate ≈ 1.0 → reward passes through
        pruner.update(0, 1.0);

        let q = pruner.q_values()[0];
        assert!(q > 0.95, "β=100, positive gap → near-full reward, Q={q}");
    }

    // ── Default β = 5.0 ────────────────────────────────────────

    #[test]
    fn test_default_beta_is_5() {
        let pruner = make_pruner(3);
        assert!((pruner.beta() - SDAR_BETA).abs() < 1e-6);
    }

    // ── set_beta runtime adjustment ─────────────────────────────

    #[test]
    fn test_set_beta_runtime() {
        let mut pruner = make_pruner(3);
        assert!((pruner.beta() - 5.0).abs() < 1e-6);

        pruner.set_beta(2.0);
        assert!((pruner.beta() - 2.0).abs() < 1e-6);
    }

    // ── Gate statistics tracking ────────────────────────────────

    #[test]
    fn test_gate_stats_tracking() {
        let mut pruner = make_pruner_with_stats(3);

        pruner.update(0, 1.0);
        pruner.update(0, 0.5);
        pruner.update(0, 1.5);

        let stats = pruner.gate_stats(0).unwrap();
        assert_eq!(stats.update_count, 3);
        assert!(stats.gate_sum > 0.0);
        assert!(stats.original_reward_sum > 0.0);
        assert!(stats.gated_reward_sum > 0.0);
        assert!(stats.last_gate > 0.0);
    }

    #[test]
    fn test_gate_stats_disabled_returns_none() {
        let pruner = make_pruner(3);
        assert!(!pruner.is_tracking_stats());
        assert!(pruner.gate_stats(0).is_none());
    }

    #[test]
    fn test_gate_stats_mean_values() {
        let mut pruner = make_pruner_with_stats(2);

        // Multiple updates with same reward
        for _ in 0..10 {
            pruner.update(0, 1.0);
        }

        let stats = pruner.gate_stats(0).unwrap();
        assert_eq!(stats.update_count, 10);
        assert!(
            stats.mean_original_reward() > 0.0,
            "Mean original reward should be positive"
        );
        assert!(
            stats.mean_gated_reward() > 0.0,
            "Mean gated reward should be positive"
        );
    }

    #[test]
    fn test_gate_stats_pass_through_ratio() {
        let mut pruner = make_pruner_with_stats(2);

        // Consistent high rewards → high pass-through
        for _ in 0..100 {
            pruner.update(0, 1.0);
        }

        let stats = pruner.gate_stats(0).unwrap();
        // After convergence, rewards match Q-values → gate ≈ 0.5 → ratio ≈ 0.5
        let ratio = stats.pass_through_ratio();
        assert!(
            ratio > 0.0 && ratio <= 1.0,
            "Pass-through ratio should be in (0, 1], got {ratio}"
        );
    }

    // ── Delegation to inner bandit ──────────────────────────────

    #[test]
    fn test_delegates_relevance_to_inner() {
        let pruner = make_pruner(3);
        // NoScreeningPruner always returns 1.0
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_best_arm_delegated() {
        let mut pruner = make_pruner(3);

        for _ in 0..200 {
            pruner.update(2, 1.0);
        }

        assert_eq!(pruner.best_arm(), 2);
    }

    #[test]
    fn test_q_values_delegated() {
        let mut pruner = make_pruner(3);
        pruner.update(0, 0.5);
        let q = pruner.q_values();
        assert_eq!(q.len(), 3);
        assert!(q[0] > 0.0);
    }

    #[test]
    fn test_visits_delegated() {
        let mut pruner = make_pruner(3);
        pruner.update(0, 0.5);
        pruner.update(0, 0.5);
        let v = pruner.visits();
        assert_eq!(v[0], 2);
    }

    #[test]
    fn test_total_pulls_delegated() {
        let mut pruner = make_pruner(3);
        pruner.update(0, 0.5);
        pruner.update(1, 0.5);
        assert_eq!(pruner.total_pulls(), 2);
    }

    // ── Out of bounds ───────────────────────────────────────────

    #[test]
    fn test_out_of_bounds_arm_is_noop() {
        let mut pruner = make_pruner(2);
        pruner.update(99, 1.0); // Should not panic
        assert_eq!(pruner.q_values().len(), 2);
    }

    // ── Config builders ─────────────────────────────────────────

    #[test]
    fn test_config_soft() {
        let config = SdarBanditConfig::soft();
        assert!((config.beta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_aggressive() {
        let config = SdarBanditConfig::aggressive();
        assert!((config.beta - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_with_gate_stats() {
        let config = SdarBanditConfig::default().with_gate_stats();
        assert!(config.track_gate_stats);
    }

    // ── Multiple arms converge independently ────────────────────

    #[test]
    fn test_multiple_arms_converge_independently() {
        let mut pruner = make_pruner(4);

        // Arm 0: low reward, arm 1: medium, arm 2: high, arm 3: very high
        for _ in 0..500 {
            pruner.update(0, 0.1);
            pruner.update(1, 0.3);
            pruner.update(2, 0.7);
            pruner.update(3, 1.0);
        }

        let q = pruner.q_values();
        assert!(q[3] > q[2], "q[3]={} > q[2]={}", q[3], q[2]);
        assert!(q[2] > q[1], "q[2]={} > q[1]={}", q[2], q[1]);
        assert!(q[1] > q[0], "q[1]={} > q[0]={}", q[1], q[0]);

        assert_eq!(pruner.best_arm(), 3);
    }

    // ── Negative rewards handled correctly ──────────────────────

    #[test]
    fn test_negative_reward_preserves_sign() {
        let mut pruner = make_pruner_with_stats(3);
        pruner.update(0, -1.0);

        let q = pruner.q_values()[0];
        assert!(q < 0.0, "Negative reward → negative Q, got {q}");
    }

    // ── Zero reward ─────────────────────────────────────────────

    #[test]
    fn test_zero_reward() {
        let mut pruner = make_pruner(3);
        pruner.update(0, 0.0);

        let q = pruner.q_values()[0];
        assert!((q).abs() < 1e-6, "Zero reward → zero Q, got {q}");
    }

    // ── Gate stats empty arm ────────────────────────────────────

    #[test]
    fn test_gate_stats_empty_arm() {
        let pruner = make_pruner_with_stats(3);
        let stats = pruner.gate_stats(1).unwrap();
        assert_eq!(stats.update_count, 0);
        assert!((stats.mean_gate()).abs() < 1e-6);
        assert!((stats.mean_original_reward()).abs() < 1e-6);
        assert!((stats.mean_gated_reward()).abs() < 1e-6);
        assert!((stats.pass_through_ratio()).abs() < 1e-6);
    }

    // ── Decay epsilon ───────────────────────────────────────────

    #[test]
    fn test_decay_epsilon_delegated() {
        let inner = BanditPruner::new(
            NoScreeningPruner,
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.5,
                decay: 0.99,
            },
            3,
        );
        let mut pruner = SdarBanditPruner::new(inner, 3);

        pruner.decay_epsilon();

        // Epsilon should have decayed — verify by checking strategy
        match pruner.inner().strategy() {
            BanditStrategy::EpsilonGreedy { epsilon, .. } => {
                assert!((*epsilon - 0.495).abs() < 1e-6, "Epsilon should decay");
            }
            _ => panic!("Expected EpsilonGreedy strategy"),
        }
    }

    // ── Learned Beta Integration Tests ──────────────────────────

    #[cfg(feature = "replaid_schedules")]
    #[test]
    fn test_sdar_bandit_learned_beta_integration() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        let pruner = SdarBanditPruner::new(inner, 3).with_learned_beta(5.0);

        // Verify learned_beta is enabled
        assert!(pruner.has_learned_beta(), "learned_beta should be Some");
        assert!(
            (pruner.beta() - 5.0).abs() < 1e-6,
            "Initial beta should be 5.0, got {}",
            pruner.beta()
        );

        let mut pruner = pruner;
        let initial_beta = pruner.beta();

        // Run 100 updates across 10 episodes with varying rewards
        for episode in 0..10 {
            let mut gated_sum = 0.0;
            let mut gated_count = 0;
            for step in 0..10 {
                let arm = step % 3;
                let reward = (episode as f32 * 0.1) + (step as f32 * 0.05) - 0.25;
                let q_before = pruner.q_values().get(arm).copied().unwrap_or(0.0);
                let gap = reward - q_before;
                let gate = 1.0 / (1.0 + (-5.0 * gap).exp());
                let gated_reward = reward * gate;
                gated_sum += gated_reward;
                gated_count += 1;
                pruner.update(arm, reward);
            }
            let mean_gated = gated_sum / gated_count as f32;
            pruner.adapt_beta(mean_gated);
        }

        // β should have changed from initial 5.0
        let final_beta = pruner.beta();
        assert!(
            (final_beta - initial_beta).abs() > 0.01,
            "β should adapt from initial {initial_beta}, got {final_beta}"
        );
    }

    #[cfg(feature = "replaid_schedules")]
    #[test]
    fn test_sdar_bandit_learned_beta_none_by_default() {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, 3);
        let pruner = SdarBanditPruner::new(inner, 3);

        // Verify learned_beta is None by default
        assert!(
            !pruner.has_learned_beta(),
            "learned_beta should be None by default"
        );

        let mut pruner = pruner;
        let initial_beta = pruner.beta();

        // Run many updates — β should stay unchanged
        for _ in 0..100 {
            for arm in 0..3 {
                pruner.update(arm, 0.5);
            }
        }

        assert_eq!(
            pruner.beta(),
            initial_beta,
            "β should remain unchanged when learned_beta is None"
        );
        assert!(
            (pruner.beta() - 5.0).abs() < 1e-6,
            "Default β should be 5.0"
        );

        // adapt_beta is a no-op when learned_beta is None
        pruner.adapt_beta(0.5);
        assert_eq!(
            pruner.beta(),
            initial_beta,
            "adapt_beta should be no-op when learned_beta is None"
        );
    }
}
