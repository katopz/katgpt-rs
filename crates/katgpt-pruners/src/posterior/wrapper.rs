//! PosteriorGuidedPruner — decorator that adds posterior tracking to any ScreeningPruner.
//!
//! This is the Phase 3 integration layer. It wraps an inner pruner and:
//! - Delegates `relevance()` to the inner pruner (domain signal)
//! - Records evidence on each arm evaluation, updating per-arm precision vectors
//! - Uses surprise-gated precision to modulate the relevance signal
//! - Exposes `lifecycle_action()` for posterior-guided lifecycle decisions
//!
//! Architecture: layered decorator pattern — inner pruner → posterior tracking → policy.
//! Same pattern as `BanditPruner<P>` but with Bayesian posterior instead of bandit Q-values.
//!
//! Zero-allocation on hot path: all precision vectors are pre-allocated fixed-size arrays.

use katgpt_core::ScreeningPruner;

use super::policy::{LifecycleAction, PrecisionPolicy, PrecisionPolicyConfig};
use super::precision::{PRECISION_DIM, PrecisionVector};
use super::types::{EvidenceContext, EvidenceOutcome, FailureMode, PosteriorEvidence};

/// A ScreeningPruner decorator that adds posterior-guided precision tracking.
///
/// Wraps any `ScreeningPruner` and tracks per-arm precision vectors using
/// BAKE-style sequential Bayesian updates. The relevance signal is modulated
/// by the posterior state: well-explored arms with high precision get their
/// domain signal through; uncertain arms get exploration bonuses.
///
/// Usage:
/// ```ignore
/// let inner = SudokuPruner::new(grid);
/// let mut pgp = PosteriorGuidedPruner::new(inner, 9, EvidenceContext::Sudoku);
/// // Use pgp.relevance() as normal — precision tracking is transparent
/// // After episode: pgp.record_evidence(...)
/// // Check lifecycle: pgp.lifecycle_action()
/// ```
pub struct PosteriorGuidedPruner<P: ScreeningPruner> {
    /// Inner domain pruner providing the base relevance signal.
    inner: P,
    /// Per-arm precision vectors. Indexed by arm (token_idx).
    precision: Vec<PrecisionVector>,
    /// Domain context for evidence conditioning.
    context: EvidenceContext,
    /// Policy for lifecycle decisions.
    policy: PrecisionPolicy,
    /// Last observed KL surprise per arm (for lifecycle decisions).
    last_surprise: Vec<f32>,
    /// Cumulative reward per arm (for observation vector construction).
    arm_reward_sum: Vec<f32>,
    /// Cumulative visit count per arm.
    arm_visits: Vec<u32>,
}

impl<P: ScreeningPruner> PosteriorGuidedPruner<P> {
    /// Create a new posterior-guided pruner wrapping an inner pruner.
    ///
    /// `num_arms`: vocabulary size (number of discrete tokens/actions).
    /// `context`: domain context for evidence conditioning (e.g., Sudoku, Bomber).
    pub fn new(inner: P, num_arms: usize, context: EvidenceContext) -> Self {
        Self {
            inner,
            precision: (0..num_arms).map(|_| PrecisionVector::new()).collect(),
            context,
            policy: PrecisionPolicy::default(),
            last_surprise: vec![0.0; num_arms],
            arm_reward_sum: vec![0.0; num_arms],
            arm_visits: vec![0; num_arms],
        }
    }

    /// Create with custom policy config.
    pub fn with_policy_config(
        inner: P,
        num_arms: usize,
        context: EvidenceContext,
        config: PrecisionPolicyConfig,
    ) -> Self {
        Self {
            inner,
            precision: (0..num_arms).map(|_| PrecisionVector::new()).collect(),
            context,
            policy: PrecisionPolicy::new(config),
            last_surprise: vec![0.0; num_arms],
            arm_reward_sum: vec![0.0; num_arms],
            arm_visits: vec![0; num_arms],
        }
    }

    /// Record evidence for an arm and update its precision vector.
    ///
    /// This is the main entry point for posterior updates. Call after
    /// an episode completes or after an arm evaluation with a verified outcome.
    ///
    /// Returns the KL surprise from the update (useful for monitoring).
    pub fn record_evidence(
        &mut self,
        arm: usize,
        outcome: EvidenceOutcome,
        failure_mode: Option<FailureMode>,
        reward: f32,
    ) -> f32 {
        if arm >= self.precision.len() {
            return 0.0;
        }

        self.arm_visits[arm] += 1;
        self.arm_reward_sum[arm] += reward;

        // Construct observation vector from current state
        let observation = self.build_observation(arm, reward);

        // Update precision vector — returns KL surprise
        let kl = self.precision[arm].update(outcome, &observation, failure_mode);
        self.last_surprise[arm] = kl;

        kl
    }

    /// Record full structured evidence (with all feature buckets).
    ///
    /// Use this when you have complete evidence including token/latency buckets.
    /// Falls back to `record_evidence()` with outcome and failure mode extraction.
    pub fn record_structured_evidence(&mut self, evidence: &PosteriorEvidence, arm: usize) -> f32 {
        if arm >= self.precision.len() {
            return 0.0;
        }

        // Compute reward from outcome
        let reward = match evidence.outcome {
            EvidenceOutcome::Success => 1.0,
            EvidenceOutcome::Failure => 0.0,
        };

        self.record_evidence(arm, evidence.outcome, evidence.failure_mode, reward)
    }

    /// Get the current lifecycle action for a specific arm.
    ///
    /// Uses the arm's precision vector, last surprise, and optional peer
    /// comparison to decide the next action.
    pub fn lifecycle_action(&self, arm: usize) -> LifecycleAction {
        self.lifecycle_action_with_peer(arm, None)
    }

    /// Get lifecycle action with peer comparison (for SPLIT detection).
    ///
    /// `peer_arm`: index of a peer arm to compare precision divergence with.
    /// If `None`, SPLIT cannot trigger (no peer to compare against).
    pub fn lifecycle_action_with_peer(
        &self,
        arm: usize,
        peer_arm: Option<usize>,
    ) -> LifecycleAction {
        if arm >= self.precision.len() {
            return LifecycleAction::Explore;
        }

        let peer = peer_arm.and_then(|p| self.precision.get(p));
        self.policy
            .decide(&self.precision[arm], self.last_surprise[arm], peer)
    }

    /// Get the lifecycle action for the best-performing arm.
    ///
    /// Useful for deciding what to do with the current top arm.
    pub fn best_arm_lifecycle_action(&self) -> (usize, LifecycleAction) {
        let best = self.best_arm();
        (best, self.lifecycle_action(best))
    }

    /// Get the precision vector for a specific arm.
    pub fn precision(&self, arm: usize) -> Option<&PrecisionVector> {
        self.precision.get(arm)
    }

    /// Get the last KL surprise for a specific arm.
    pub fn last_surprise(&self, arm: usize) -> f32 {
        self.last_surprise.get(arm).copied().unwrap_or(0.0)
    }

    /// Get the arm with highest success probability.
    pub fn best_arm(&self) -> usize {
        let mut best = 0;
        let mut best_prob = 0.0f32;
        for (i, pv) in self.precision.iter().enumerate() {
            if pv.observations() == 0 {
                continue;
            }
            let prob = pv.success_probability();
            if prob > best_prob {
                best_prob = prob;
                best = i;
            }
        }
        best
    }

    /// Number of arms being tracked.
    pub fn num_arms(&self) -> usize {
        self.precision.len()
    }

    /// Total observations across all arms.
    pub fn total_observations(&self) -> u32 {
        self.precision.iter().map(|pv| pv.observations()).sum()
    }

    /// Get a reference to the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Get a mutable reference to the inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Get a reference to the policy.
    pub fn policy(&self) -> &PrecisionPolicy {
        &self.policy
    }

    /// Get the domain context.
    pub fn context(&self) -> EvidenceContext {
        self.context
    }

    /// Visit count for a specific arm.
    pub fn arm_visits(&self, arm: usize) -> u32 {
        self.arm_visits.get(arm).copied().unwrap_or(0)
    }

    /// Average reward for a specific arm.
    pub fn arm_avg_reward(&self, arm: usize) -> f32 {
        let visits = self.arm_visits.get(arm).copied().unwrap_or(0);
        if visits == 0 {
            return 0.0;
        }
        self.arm_reward_sum.get(arm).copied().unwrap_or(0.0) / visits as f32
    }

    /// Build an observation vector for the precision update.
    ///
    /// The observation vector encodes the current evaluation context:
    /// - [0] context_similarity: reward signal (1.0 for success, 0.0 for failure)
    /// - [1] failure_density: cumulative failure rate for this arm
    /// - [2] eval_rate: normalized visit frequency
    /// - [3] latency_signal: placeholder (0.5 uniform, actual latency injected externally)
    /// - [4] reward_mean: running average reward for this arm
    /// - [5] reward_variance: placeholder (0.5 uniform, updated from trace)
    /// - [6] exploration_bonus: inverse visit count (higher = less explored)
    /// - [7] domain_coverage: uniform prior (0.5)
    fn build_observation(&self, arm: usize, reward: f32) -> [f32; PRECISION_DIM] {
        let visits = self.arm_visits[arm].max(1) as f32;
        let avg_reward = self.arm_reward_sum[arm] / visits;
        let total_obs: f32 = self
            .precision
            .iter()
            .map(|pv| pv.observations() as f32)
            .sum();
        let eval_rate = if total_obs > 0.0 {
            visits / total_obs
        } else {
            1.0 / self.precision.len() as f32
        };

        let pv = &self.precision[arm];
        let failure_density = if pv.observations() > 0 {
            (pv.beta() - 1.0) / (pv.alpha() + pv.beta() - 2.0) // Posterior failure rate
        } else {
            0.5
        };

        let exploration_bonus = 1.0 / (1.0 + visits);

        [
            reward,            // context_similarity
            failure_density,   // failure_density
            eval_rate,         // eval_rate
            0.5,               // latency_signal (placeholder)
            avg_reward,        // reward_mean
            0.5,               // reward_variance (placeholder)
            exploration_bonus, // exploration_bonus
            0.5,               // domain_coverage (uniform)
        ]
    }
}

impl<P: ScreeningPruner> ScreeningPruner for PosteriorGuidedPruner<P> {
    /// Relevance with precision-gated modulation.
    ///
    /// Delegates to the inner pruner for the domain signal, then modulates
    /// based on the arm's posterior state:
    /// - Cold start (no observations): pass through domain signal unchanged
    /// - High-precision arms (well-explored): pass through domain signal
    /// - Low-precision arms (uncertain): exploration bonus
    /// - Retired arms (failure-dominant): relevance reduced to 0.0
    ///
    /// This is zero-allocation on the hot path — no Vecs, no HashMaps.
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if token_idx >= self.precision.len() {
            return 0.0;
        }

        // Domain signal from inner pruner
        let domain = self.inner.relevance(depth, token_idx, parent_tokens);
        if domain <= 0.0 {
            return 0.0;
        }

        let pv = &self.precision[token_idx];

        // Cold start: no observations — pass domain through unchanged
        if pv.observations() == 0 {
            return domain;
        }

        // Retired arms: relevance = 0.0 (failure-dominant)
        let action = self.lifecycle_action(token_idx);
        if matches!(action, LifecycleAction::Retire) {
            return 0.0;
        }

        // Precision modulation:
        // - High precision (well-explored, successful): trust domain signal
        // - Low precision (uncertain): small exploration bonus
        let success_prob = pv.success_probability();
        let avg_prec = pv.avg_precision();

        // Exploration bonus: inversely proportional to precision
        // sigmoid(-λ × (precision - threshold)) gives bonus for low-precision arms
        let exploration_bonus = {
            let x = -0.5 * (avg_prec - 3.0); // λ=0.5, threshold=3.0
            let s = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let ex = x.exp();
                ex / (1.0 + ex)
            };
            0.1 * s // Max 10% bonus, decays with precision
        };

        // Precision confidence: how much we trust this arm's signal
        // High precision + high success = strong trust
        let confidence = (0.9 + 0.1 * success_prob).min(1.0);

        (domain * confidence + exploration_bonus).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple pruner that returns fixed relevance values per arm.
    struct FixedPruner {
        values: Vec<f32>,
    }

    impl FixedPruner {
        fn new(values: Vec<f32>) -> Self {
            Self { values }
        }
    }

    impl ScreeningPruner for FixedPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.values.get(token_idx).copied().unwrap_or(0.0)
        }
    }

    #[test]
    fn cold_start_passes_domain_through() {
        let inner = FixedPruner::new(vec![0.8, 0.6, 0.4]);
        let pgp = PosteriorGuidedPruner::new(inner, 3, EvidenceContext::Generic);

        // No observations yet — should return domain signal unchanged
        assert!((pgp.relevance(0, 0, &[]) - 0.8).abs() < 1e-6);
        assert!((pgp.relevance(0, 1, &[]) - 0.6).abs() < 1e-6);
        assert!((pgp.relevance(0, 2, &[]) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn retired_arm_gets_zero_relevance() {
        let inner = FixedPruner::new(vec![0.8, 0.8, 0.8]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 3, EvidenceContext::Generic);

        // Arm 1: many failures → should be retired
        for _ in 0..10 {
            pgp.record_evidence(1, EvidenceOutcome::Failure, None, 0.0);
        }

        // Domain is 0.8 but posterior says retire → should be 0.0
        assert_eq!(pgp.relevance(0, 1, &[]), 0.0);
    }

    #[test]
    fn record_evidence_updates_precision() {
        let inner = FixedPruner::new(vec![1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Sudoku);

        assert_eq!(pgp.precision(0).unwrap().observations(), 0);

        let kl = pgp.record_evidence(0, EvidenceOutcome::Success, None, 1.0);
        assert!(kl >= 0.0);
        assert_eq!(pgp.precision(0).unwrap().observations(), 1);
        assert_eq!(pgp.arm_visits(0), 1);
    }

    #[test]
    fn best_arm_converges_to_successful() {
        let inner = FixedPruner::new(vec![1.0, 1.0, 1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 3, EvidenceContext::Generic);

        // Arm 0: 90% success rate
        for _ in 0..9 {
            pgp.record_evidence(0, EvidenceOutcome::Success, None, 1.0);
        }
        pgp.record_evidence(0, EvidenceOutcome::Failure, None, 0.0);

        // Arm 1: 50% success rate
        for _ in 0..5 {
            pgp.record_evidence(1, EvidenceOutcome::Success, None, 1.0);
        }
        for _ in 0..5 {
            pgp.record_evidence(1, EvidenceOutcome::Failure, None, 0.0);
        }

        // Arm 2: 10% success rate
        pgp.record_evidence(2, EvidenceOutcome::Success, None, 1.0);
        for _ in 0..9 {
            pgp.record_evidence(2, EvidenceOutcome::Failure, None, 0.0);
        }

        assert_eq!(pgp.best_arm(), 0);
    }

    #[test]
    fn lifecycle_explore_when_no_data() {
        let inner = FixedPruner::new(vec![1.0]);
        let pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Generic);
        assert_eq!(pgp.lifecycle_action(0), LifecycleAction::Explore);
    }

    #[test]
    fn lifecycle_retire_on_dominant_failure() {
        let inner = FixedPruner::new(vec![1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Generic);

        // Overwhelming failures
        for _ in 0..10 {
            pgp.record_evidence(
                0,
                EvidenceOutcome::Failure,
                Some(FailureMode::FalseAccept),
                0.0,
            );
        }

        assert_eq!(pgp.lifecycle_action(0), LifecycleAction::Retire);
    }

    #[test]
    fn lifecycle_patch_on_repeated_failure_with_surprise() {
        let inner = FixedPruner::new(vec![1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Generic);

        // Mix of successes and failures, with repeated FalseAccept
        for _ in 0..5 {
            pgp.record_evidence(0, EvidenceOutcome::Success, None, 1.0);
        }
        // Create high surprise + repeated failure mode
        for _ in 0..3 {
            pgp.record_evidence(
                0,
                EvidenceOutcome::Failure,
                Some(FailureMode::FalseAccept),
                0.0,
            );
        }

        let action = pgp.lifecycle_action(0);
        assert!(matches!(action, LifecycleAction::Patch { .. }));
    }

    #[test]
    fn lifecycle_compress_on_high_precision() {
        let inner = FixedPruner::new(vec![1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Generic);

        // Many successful observations → high precision
        for _ in 0..100 {
            pgp.record_evidence(0, EvidenceOutcome::Success, None, 1.0);
        }

        let action = pgp.lifecycle_action(0);
        assert_eq!(action, LifecycleAction::Compress);
    }

    #[test]
    fn split_on_divergence() {
        let inner = FixedPruner::new(vec![1.0, 1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 2, EvidenceContext::Generic);

        // Arm 0: many observations
        for _ in 0..100 {
            pgp.record_evidence(0, EvidenceOutcome::Success, None, 1.0);
        }
        // Arm 1: few observations
        pgp.record_evidence(1, EvidenceOutcome::Success, None, 1.0);

        let action = pgp.lifecycle_action_with_peer(0, Some(1));
        assert_eq!(action, LifecycleAction::Split);
    }

    #[test]
    fn out_of_bounds_arm_returns_zero() {
        let inner = FixedPruner::new(vec![1.0]);
        let pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Generic);

        assert_eq!(pgp.relevance(0, 99, &[]), 0.0);
    }

    #[test]
    fn record_evidence_out_of_bounds_is_noop() {
        let inner = FixedPruner::new(vec![1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Generic);

        // Should not panic
        let kl = pgp.record_evidence(99, EvidenceOutcome::Success, None, 1.0);
        assert_eq!(kl, 0.0);
    }

    #[test]
    fn inner_pruner_accessible() {
        let inner = FixedPruner::new(vec![0.7]);
        let pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Sudoku);

        assert!((pgp.inner().relevance(0, 0, &[]) - 0.7).abs() < 1e-6);
        assert_eq!(pgp.context(), EvidenceContext::Sudoku);
    }

    #[test]
    fn total_observations_aggregates_across_arms() {
        let inner = FixedPruner::new(vec![1.0, 1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 2, EvidenceContext::Generic);

        pgp.record_evidence(0, EvidenceOutcome::Success, None, 1.0);
        pgp.record_evidence(0, EvidenceOutcome::Success, None, 1.0);
        pgp.record_evidence(1, EvidenceOutcome::Failure, None, 0.0);

        assert_eq!(pgp.total_observations(), 3);
    }

    #[test]
    fn structured_evidence_delegates_correctly() {
        let inner = FixedPruner::new(vec![1.0]);
        let mut pgp = PosteriorGuidedPruner::new(inner, 1, EvidenceContext::Sudoku);

        let evidence = PosteriorEvidence {
            task_id: 42,
            outcome: EvidenceOutcome::Success,
            context: EvidenceContext::Sudoku,
            failure_mode: None,
            eval_bucket: super::super::types::EvalBucket::Medium,
            latency_bucket: super::super::types::LatencyBucket::Fast,
        };

        let kl = pgp.record_structured_evidence(&evidence, 0);
        assert!(kl >= 0.0);
        assert_eq!(pgp.precision(0).unwrap().observations(), 1);
    }

    #[test]
    fn with_custom_policy_config() {
        let config = PrecisionPolicyConfig {
            retire_min_beta: 100.0,
            ..Default::default()
        }; // Very high threshold

        let inner = FixedPruner::new(vec![1.0]);
        let mut pgp =
            PosteriorGuidedPruner::with_policy_config(inner, 1, EvidenceContext::Generic, config);

        // 10 failures would normally trigger retire, but threshold is 100
        for _ in 0..10 {
            pgp.record_evidence(0, EvidenceOutcome::Failure, None, 0.0);
        }

        // Should NOT be retired (beta = 11 < 100)
        assert!(!matches!(pgp.lifecycle_action(0), LifecycleAction::Retire));
    }
}
