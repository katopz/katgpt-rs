//! FeedbackBandit — Harness + Weight Co-Evolution (Plan 163, Research 033).
//!
//! Extends the existing [`ConfiguratorBandit`] (Plan 112) with two new arms
//! that close the model-based/modelless loop:
//! - `HarnessUpdate`: AbsorbCompress promote + HotSwapPruner reload
//! - `WeightUpdate`: trigger riir-gpu training step on accumulated TrialLog
//!
//! The bandit learns when to switch levers based on trajectory dynamics
//! (stall detection), not a fixed schedule. UCB1 naturally explores the
//! new arms when existing SR²AM arms plateau.
//!
//! Reference: [arXiv:2605.27276](https://arxiv.org/pdf/2605.27276) — SIA: Self Improving AI

use crate::pruners::configurator_bandit::ConfiguratorBandit;
use katgpt_core::{ConfiguratorContext, PlanningDecision};

// ── Configuration ─────────────────────────────────────────────

/// Configuration for FeedbackBandit stall detection and reward shaping.
#[derive(Debug, Clone)]
pub struct FeedbackBanditConfig {
    /// Number of consecutive episodes with low reward delta before stall triggers.
    /// Default: 10.
    pub stall_patience: usize,
    /// Reward delta threshold below which an episode is considered "stalled".
    /// Default: 0.01 (1% improvement).
    pub stall_epsilon: f32,
    /// Cost multiplier for WeightUpdate arm in reward shaping.
    /// Training is expensive — default 2.0 means WeightUpdate costs 20× PlanSkip.
    pub weight_update_cost: f32,
    /// Cost multiplier for HarnessUpdate arm.
    /// Harness reload is cheaper — default 0.5.
    pub harness_update_cost: f32,
}

impl Default for FeedbackBanditConfig {
    fn default() -> Self {
        Self {
            stall_patience: 10,
            stall_epsilon: 0.01,
            weight_update_cost: 2.0,
            harness_update_cost: 0.5,
        }
    }
}

// ── Weight Update Request ─────────────────────────────────────

/// Request emitted when the bandit selects the `WeightUpdate` arm.
///
/// Contains the accumulated TrialLog data needed for riir-gpu training.
/// The actual training is handled by `FeedbackTrainingBridge` in riir-ai,
/// not by the bandit itself — keeping katgpt-rs free of GPU dependencies.
#[derive(Debug, Clone)]
pub struct WeightUpdateRequest {
    /// Domain index for the training run.
    pub domain: usize,
    /// Episode range (start..end) to include in training data.
    pub episode_range: (usize, usize),
    /// Suggested RL algorithm based on reward signal density.
    pub suggested_algorithm: RlAlgorithmHint,
}

/// Hint for the training bridge about which RL algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RlAlgorithmHint {
    /// Dense reward signal → GRPO (group relative policy optimization).
    Grpo,
    /// Sparse or reward-skewed signal → entropic advantage weighting.
    EntropicAdvantage,
    /// Very sparse signal → Best-of-N SFT cold-start, then GRPO.
    BestOfNSft,
}

// ── Trajectory Summary ────────────────────────────────────────

/// Compressed view of recent trajectory dynamics for stall detection.
///
/// Fixed-size — no per-episode allocation. Updated incrementally.
#[derive(Debug, Clone, Default)]
pub struct TrajectorySummary {
    /// Running mean of reward deltas (incremental).
    pub mean_reward_delta: f32,
    /// Number of consecutive episodes below stall epsilon.
    pub stall_count: usize,
    /// Total episodes tracked.
    pub total_episodes: usize,
    /// Distribution of arm pulls: [PlanNew, PlanExtend, PlanSkip, SpecHop, HarnessUpdate, WeightUpdate].
    pub arm_pulls: [usize; 6],
}

impl TrajectorySummary {
    /// Update the trajectory summary with a new reward delta observation.
    pub fn observe(&mut self, reward_delta: f32, config: &FeedbackBanditConfig) {
        self.total_episodes += 1;

        // Incremental mean update
        let n = self.total_episodes as f32;
        self.mean_reward_delta += (reward_delta - self.mean_reward_delta) / n;

        // Stall detection
        if reward_delta.abs() < config.stall_epsilon {
            self.stall_count += 1;
        } else {
            self.stall_count = 0;
        }
    }

    /// Record an arm pull in the distribution tracker.
    pub fn record_arm(&mut self, decision: PlanningDecision) {
        let idx = match decision {
            PlanningDecision::PlanNew => 0,
            PlanningDecision::PlanExtend => 1,
            PlanningDecision::PlanSkip => 2,
            PlanningDecision::SpecHop { .. } => 3,
            PlanningDecision::HarnessUpdate => 4,
            PlanningDecision::WeightUpdate => 5,
        };
        if idx < self.arm_pulls.len() {
            self.arm_pulls[idx] += 1;
        }
    }

    /// Whether the trajectory is stalled (N consecutive episodes with low reward delta).
    pub fn is_stalled(&self, config: &FeedbackBanditConfig) -> bool {
        self.stall_count >= config.stall_patience
    }
}

// ── FeedbackBandit ────────────────────────────────────────────

/// Extended configurator bandit with harness and weight update arms.
///
/// Wraps a [`ConfiguratorBandit`] and adds:
/// - Stall detection: tracks reward delta trajectory, flags when plateaued
/// - `HarnessUpdate` arm: selected when stall detected and compress may help
/// - `WeightUpdate` arm: selected when stall persists despite harness update
///
/// The underlying UCB1 selection naturally explores new arms when existing
/// ones plateau (Q-values converge, exploration bonus increases for unvisited).
pub struct FeedbackBandit {
    /// Inner SR²AM configurator bandit (6 arms with sia_feedback).
    inner: ConfiguratorBandit,
    /// FeedbackBandit configuration.
    config: FeedbackBanditConfig,
    /// Trajectory summary for stall detection.
    trajectory: TrajectorySummary,
    /// Pending WeightUpdate request (set when WeightUpdate arm is selected).
    pending_weight_request: Option<WeightUpdateRequest>,
    /// Episode counter for request range tracking.
    episode_count: usize,
}

impl FeedbackBandit {
    /// Create a new FeedbackBandit with default configuration.
    pub fn new() -> Self {
        Self::with_config(FeedbackBanditConfig::default())
    }

    /// Create a new FeedbackBandit with custom configuration.
    pub fn with_config(config: FeedbackBanditConfig) -> Self {
        Self {
            inner: ConfiguratorBandit::new(),
            config,
            trajectory: TrajectorySummary::default(),
            pending_weight_request: None,
            episode_count: 0,
        }
    }

    /// Select a planning decision using UCB1, considering stall state.
    ///
    /// When stalled, the bandit context is augmented to encourage exploration
    /// of HarnessUpdate and WeightUpdate arms. UCB1 handles this naturally —
    /// stalled contexts get independent Q-values.
    pub fn select(&mut self, context: ConfiguratorContext) -> PlanningDecision {
        self.episode_count += 1;

        // Augment context with stalled flag for Q-value isolation
        let augmented = if self.trajectory.is_stalled(&self.config) {
            // Use desperation_bin to encode stall state
            context.with_desperation(1.0)
        } else {
            context
        };

        let decision = self.inner.select(augmented);

        // Track arm pull
        self.trajectory.record_arm(decision);

        // Generate WeightUpdateRequest if that arm was selected
        if matches!(decision, PlanningDecision::WeightUpdate) {
            let start = self
                .episode_count
                .saturating_sub(self.config.stall_patience);
            self.pending_weight_request = Some(WeightUpdateRequest {
                domain: context.domain,
                episode_range: (start, self.episode_count),
                suggested_algorithm: self.suggest_algorithm(),
            });
        }

        decision
    }

    /// Update Q-values after observing reward for a decision.
    pub fn update(
        &mut self,
        context: ConfiguratorContext,
        decision: PlanningDecision,
        quality_gain: f32,
    ) {
        let token_cost = match decision {
            PlanningDecision::PlanNew => 1.0,
            PlanningDecision::PlanExtend => 0.3,
            PlanningDecision::PlanSkip => 0.0,
            PlanningDecision::SpecHop { k } => 0.1 * (k.min(8) as f32),
            PlanningDecision::HarnessUpdate => self.config.harness_update_cost,
            PlanningDecision::WeightUpdate => self.config.weight_update_cost,
        };

        let reward = ConfiguratorBandit::reward_signal(quality_gain, token_cost, 0.1);

        // Update trajectory summary for stall detection
        self.trajectory.observe(quality_gain, &self.config);

        // Use augmented context for Q-value update (matches select)
        let augmented = if self.trajectory.is_stalled(&self.config) {
            context.with_desperation(1.0)
        } else {
            context
        };

        self.inner.update(augmented, decision, reward);
    }

    /// Take the pending WeightUpdate request (if any).
    ///
    /// Returns `Some(WeightUpdateRequest)` the first time after a
    /// `WeightUpdate` arm was selected, `None` thereafter.
    pub fn take_weight_request(&mut self) -> Option<WeightUpdateRequest> {
        self.pending_weight_request.take()
    }

    /// Whether the trajectory is currently stalled.
    pub fn is_stalled(&self) -> bool {
        self.trajectory.is_stalled(&self.config)
    }

    /// Get trajectory summary snapshot.
    pub fn trajectory_summary(&self) -> &TrajectorySummary {
        &self.trajectory
    }

    /// Get reference to inner configurator bandit.
    pub fn inner(&self) -> &ConfiguratorBandit {
        &self.inner
    }

    /// Suggest RL algorithm based on reward signal density.
    ///
    /// Uses the trajectory's mean reward delta to classify:
    /// - High variance (|delta| > 0.1) → dense → GRPO
    /// - Medium variance → EntropicAdvantage
    /// - Near-zero for many episodes → BestOfNSft
    fn suggest_algorithm(&self) -> RlAlgorithmHint {
        let mean_abs_delta = self.trajectory.mean_reward_delta.abs();
        if mean_abs_delta > 0.1 {
            RlAlgorithmHint::Grpo
        } else if mean_abs_delta > 0.01 {
            RlAlgorithmHint::EntropicAdvantage
        } else {
            RlAlgorithmHint::BestOfNSft
        }
    }
}

impl Default for FeedbackBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context() -> ConfiguratorContext {
        ConfiguratorContext::new(0, 5) // domain=0, entropy_bin=5
    }

    #[test]
    fn test_feedback_bandit_default_config() {
        let config = FeedbackBanditConfig::default();
        assert_eq!(config.stall_patience, 10);
        assert!((config.stall_epsilon - 0.01).abs() < f32::EPSILON);
        assert!((config.weight_update_cost - 2.0).abs() < f32::EPSILON);
        assert!((config.harness_update_cost - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_trajectory_summary_observe_updates_mean() {
        let config = FeedbackBanditConfig::default();
        let mut traj = TrajectorySummary::default();

        traj.observe(0.5, &config);
        assert!((traj.mean_reward_delta - 0.5).abs() < 1e-6);
        assert_eq!(traj.total_episodes, 1);
        assert_eq!(traj.stall_count, 0);

        traj.observe(0.005, &config);
        assert_eq!(traj.stall_count, 1); // below epsilon

        traj.observe(0.5, &config);
        assert_eq!(traj.stall_count, 0); // reset
    }

    #[test]
    fn test_trajectory_stall_detection() {
        let config = FeedbackBanditConfig {
            stall_patience: 3,
            stall_epsilon: 0.01,
            ..Default::default()
        };
        let mut traj = TrajectorySummary::default();

        // 3 episodes below epsilon → stalled
        for _ in 0..3 {
            traj.observe(0.005, &config);
        }
        assert!(traj.is_stalled(&config));

        // One good episode resets
        traj.observe(0.5, &config);
        assert!(!traj.is_stalled(&config));
    }

    #[test]
    fn test_feedback_bandit_select_explores_all_arms() {
        let mut bandit = FeedbackBandit::new();
        let ctx = make_context();

        // Pull many times — UCB1 should explore all 6 arms
        let mut seen = [false; 6];
        for _ in 0..200 {
            let decision = bandit.select(ctx);
            let idx = match decision {
                PlanningDecision::PlanNew => 0,
                PlanningDecision::PlanExtend => 1,
                PlanningDecision::PlanSkip => 2,
                PlanningDecision::SpecHop { .. } => 3,
                PlanningDecision::HarnessUpdate => 4,
                PlanningDecision::WeightUpdate => 5,
            };
            seen[idx] = true;
            bandit.update(ctx, decision, 0.5);
        }

        // All arms should have been tried at least once
        for (i, s) in seen.iter().enumerate() {
            assert!(*s, "arm {i} was never selected");
        }
    }

    #[test]
    fn test_weight_update_request_emitted() {
        let mut bandit = FeedbackBandit::new();
        let ctx = make_context();

        // Force WeightUpdate selection by exhausting other arms' exploration bonus
        // First, ensure we can get a WeightUpdate by selecting many times
        let mut got_weight_update = false;
        for _ in 0..200 {
            let decision = bandit.select(ctx);
            bandit.update(ctx, decision, 0.1);
            if matches!(decision, PlanningDecision::WeightUpdate) {
                got_weight_update = true;
                let req = bandit.take_weight_request();
                assert!(req.is_some(), "WeightUpdate request should be emitted");
                let req = req.unwrap();
                assert_eq!(req.domain, 0);
                break;
            }
        }
        assert!(
            got_weight_update,
            "WeightUpdate arm should have been selected"
        );

        // Request should be consumed
        assert!(bandit.take_weight_request().is_none());
    }

    #[test]
    fn test_suggest_algorithm() {
        let mut bandit = FeedbackBandit::new();

        // High delta → GRPO
        for _ in 0..10 {
            bandit.trajectory.observe(0.5, &bandit.config);
        }
        assert_eq!(bandit.suggest_algorithm(), RlAlgorithmHint::Grpo);

        // Reset with low delta → BestOfNSft
        bandit.trajectory = TrajectorySummary::default();
        for _ in 0..10 {
            bandit.trajectory.observe(0.005, &bandit.config);
        }
        assert_eq!(bandit.suggest_algorithm(), RlAlgorithmHint::BestOfNSft);
    }

    #[test]
    fn test_stall_triggers_desperation_context() {
        let config = FeedbackBanditConfig {
            stall_patience: 2,
            stall_epsilon: 0.01,
            ..Default::default()
        };
        let mut bandit = FeedbackBandit::with_config(config);
        let ctx = make_context();

        // Create stall condition
        for _ in 0..3 {
            let decision = bandit.select(ctx);
            bandit.update(ctx, decision, 0.005);
        }

        assert!(bandit.is_stalled());
    }

    #[test]
    fn test_record_arm_distribution() {
        let mut traj = TrajectorySummary::default();

        traj.record_arm(PlanningDecision::PlanNew);
        traj.record_arm(PlanningDecision::PlanNew);
        traj.record_arm(PlanningDecision::WeightUpdate);

        assert_eq!(traj.arm_pulls[0], 2); // PlanNew
        assert_eq!(traj.arm_pulls[5], 1); // WeightUpdate
    }
}
