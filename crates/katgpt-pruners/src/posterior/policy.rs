//! Precision policy for posterior-guided lifecycle actions.
//!
//! Maps posterior state (precision vector + surprise) to one of five
//! lifecycle actions: Explore, Patch, Split, Compress, Retire.
//!
//! This goes beyond the paper's fixed thresholds by using:
//! - Sigmoid-gated surprise triggers (not count-based)
//! - Precision-gated actions (not Q-threshold)
//! - Per-dimension analysis (not aggregate)

use crate::posterior::precision::PrecisionVector;
use crate::posterior::surprise::SurpriseComputer;
use crate::posterior::types::FailureMode;

/// The five posterior-guided lifecycle actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleAction {
    /// No observations available. Collect evidence before deciding.
    Explore,
    /// Same failure mode repeated. Inject guardrail/patch into pruner context.
    Patch { failure_mode: FailureMode },
    /// Arm's precision diverges from peers. Split into separate experts.
    Split,
    /// High precision, stable performance. Compress for efficiency.
    Compress,
    /// Failure evidence dominates. Retire this arm/skill.
    Retire,
}

/// Configuration for precision policy thresholds.
#[derive(Debug, Clone)]
pub struct PrecisionPolicyConfig {
    /// Surprise threshold for PATCH trigger (sigmoid gate must exceed this).
    pub patch_surprise_threshold: f32,
    /// Minimum failure mode count for PATCH trigger.
    pub patch_min_failure_count: u32,
    /// Minimum precision divergence for SPLIT trigger.
    pub split_divergence_threshold: f32,
    /// Minimum observations before SPLIT can trigger.
    pub split_min_observations: u32,
    /// Minimum precision (avg) for COMPRESS trigger.
    pub compress_precision_threshold: f32,
    /// Minimum observations before COMPRESS can trigger.
    pub compress_min_observations: u32,
    /// Maximum success probability for RETIRE trigger.
    pub retire_max_success_prob: f32,
    /// Minimum failure count (beta) for RETIRE trigger.
    pub retire_min_beta: f32,
    /// Sensitivity parameter for surprise computation.
    pub surprise_beta: f32,
    /// Surprise floor (minimum KL to count as surprising).
    pub surprise_floor: f32,
}

impl Default for PrecisionPolicyConfig {
    fn default() -> Self {
        // Calibrated from paper's Table 1 thresholds, adapted for precision
        Self {
            patch_surprise_threshold: 0.7,
            patch_min_failure_count: 2,
            split_divergence_threshold: 10.0,
            split_min_observations: 4,
            compress_precision_threshold: 5.0,
            compress_min_observations: 3,
            retire_max_success_prob: 0.45,
            retire_min_beta: 4.0,
            surprise_beta: 2.0,
            surprise_floor: 0.1,
        }
    }
}

/// Posterior-guided precision policy.
///
/// Consumes precision state and surprise signal, emits lifecycle actions.
/// The decision follows an ordered priority (same as paper's Eq. 12),
/// but uses precision-weighted triggers instead of fixed count thresholds.
#[derive(Debug, Clone)]
pub struct PrecisionPolicy {
    config: PrecisionPolicyConfig,
    surprise_computer: SurpriseComputer,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::new(PrecisionPolicyConfig::default())
    }
}

impl PrecisionPolicy {
    /// Create with custom config.
    pub fn new(config: PrecisionPolicyConfig) -> Self {
        let surprise_computer = SurpriseComputer::new(config.surprise_beta, config.surprise_floor);
        Self {
            config,
            surprise_computer,
        }
    }

    /// Decide the lifecycle action for a pruner arm.
    ///
    /// Priority order (matching paper's Eq. 12):
    /// 1. EXPLORE: no observations yet
    /// 2. RETIRE: failure-dominant, low success probability
    /// 3. PATCH: repeated failure mode detected
    /// 4. SPLIT: precision diverges from peers (if peer provided)
    /// 5. COMPRESS: high precision, stable performance
    /// 6. EXPLORE: default (continue collecting evidence)
    pub fn decide(
        &self,
        precision: &PrecisionVector,
        surprise_kl: f32,
        peer_precision: Option<&PrecisionVector>,
    ) -> LifecycleAction {
        // 1. EXPLORE: no observations
        if precision.observations() == 0 {
            return LifecycleAction::Explore;
        }

        // 2. RETIRE: failure evidence dominates
        if precision.beta() >= self.config.retire_min_beta
            && precision.success_probability() < self.config.retire_max_success_prob
        {
            return LifecycleAction::Retire;
        }

        // 3. PATCH: repeated failure mode with surprise
        let (max_fm, max_fm_count) = precision.max_failure_mode_count();
        if max_fm_count >= self.config.patch_min_failure_count {
            let (_, gate) = self
                .surprise_computer
                .compute_surprise(precision, surprise_kl);
            if gate > self.config.patch_surprise_threshold {
                return LifecycleAction::Patch {
                    failure_mode: max_fm,
                };
            }
        }

        // 4. SPLIT: precision diverges from peers
        if precision.observations() >= self.config.split_min_observations
            && let Some(peer) = peer_precision
        {
            let divergence = precision.precision_divergence(peer);
            if divergence > self.config.split_divergence_threshold {
                return LifecycleAction::Split;
            }
        }

        // 5. COMPRESS: high precision, stable
        if precision.observations() >= self.config.compress_min_observations
            && precision.avg_precision() >= self.config.compress_precision_threshold
        {
            return LifecycleAction::Compress;
        }

        // 6. Default: continue exploring
        LifecycleAction::Explore
    }

    /// Get a reference to the surprise computer.
    pub fn surprise(&self) -> &SurpriseComputer {
        &self.surprise_computer
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &PrecisionPolicyConfig {
        &self.config
    }
}

#[cfg(test)]
#[allow(clippy::op_ref)]
mod tests {
    use super::*;
    use crate::posterior::types::EvidenceOutcome;

    fn make_precision(successes: u32, failures: u32) -> PrecisionVector {
        let mut pv = PrecisionVector::new();
        let obs = [0.5; 8];
        for _ in 0..successes {
            pv.update(EvidenceOutcome::Success, &obs, None);
        }
        for _ in 0..failures {
            pv.update(EvidenceOutcome::Failure, &obs, None);
        }
        pv
    }

    fn make_precision_with_failure(
        successes: u32,
        failures: u32,
        fm: FailureMode,
    ) -> PrecisionVector {
        let mut pv = PrecisionVector::new();
        let obs = [0.5; 8];
        for _ in 0..successes {
            pv.update(EvidenceOutcome::Success, &obs, None);
        }
        for _ in 0..failures {
            pv.update(EvidenceOutcome::Failure, &obs, Some(fm));
        }
        pv
    }

    #[test]
    fn explore_when_no_observations() {
        let policy = PrecisionPolicy::default();
        let pv = PrecisionVector::new();
        let action = policy.decide(&pv, 0.0, None);
        assert_eq!(action, LifecycleAction::Explore);
    }

    #[test]
    fn retire_when_failure_dominates() {
        let policy = PrecisionPolicy::default();
        // 1 success, 10 failures → p ≈ 0.15
        let pv = make_precision(1, 10);
        assert!(pv.success_probability() < 0.45);
        assert!(pv.beta() >= 4.0);
        let action = policy.decide(&pv, 0.0, None);
        assert_eq!(action, LifecycleAction::Retire);
    }

    #[test]
    fn patch_on_repeated_failure() {
        let policy = PrecisionPolicy::default();
        // 5 successes, 3 failures all FalseAccept → p = 6/10 = 0.6 (not retire)
        let pv = make_precision_with_failure(5, 3, FailureMode::FalseAccept);
        let action = policy.decide(&pv, 5.0, None); // High KL = surprising
        assert!(matches!(
            action,
            LifecycleAction::Patch {
                failure_mode: FailureMode::FalseAccept
            }
        ));
    }

    #[test]
    fn no_patch_without_surprise() {
        let policy = PrecisionPolicy::default();
        let pv = make_precision_with_failure(5, 3, FailureMode::FalseAccept);
        let action = policy.decide(&pv, 0.01, None); // Low KL = not surprising
        // Should not trigger PATCH (surprise gate too low)
        assert!(!matches!(action, LifecycleAction::Patch { .. }));
    }

    #[test]
    fn compress_when_high_precision() {
        let policy = PrecisionPolicy::default();
        // Many observations → high precision
        let pv = make_precision(100, 5);
        assert!(pv.avg_precision() >= 5.0);
        let action = policy.decide(&pv, 0.0, None);
        assert_eq!(action, LifecycleAction::Compress);
    }

    #[test]
    fn split_on_divergence() {
        let config = PrecisionPolicyConfig {
            compress_precision_threshold: 1000.0, // Prevent compress from firing first
            ..PrecisionPolicyConfig::default()
        };
        let policy = PrecisionPolicy::new(config);

        let mut pv1 = PrecisionVector::new();
        let mut pv2 = PrecisionVector::new();
        let obs = [0.5; 8];

        // pv1 gets many observations
        for _ in 0..100 {
            pv1.update(EvidenceOutcome::Success, &obs, None);
        }
        // pv2 gets few
        pv2.update(EvidenceOutcome::Success, &obs, None);

        // pv1 needs >= 4 observations (it has 100)
        // divergence between pv1 (precision ~101) and pv2 (precision ~2) is ~99
        let action = policy.decide(&pv1, 0.0, Some(&pv2));
        assert_eq!(action, LifecycleAction::Split);
    }

    #[test]
    fn default_explore_when_insufficient_evidence() {
        let policy = PrecisionPolicy::default();
        // 2 observations, not enough for any trigger
        let pv = make_precision(1, 1);
        let action = policy.decide(&pv, 0.0, None);
        assert_eq!(action, LifecycleAction::Explore);
    }

    #[test]
    fn priority_order_retire_over_patch() {
        let policy = PrecisionPolicy::default();
        // 0 successes, 5 failures all FalseAccept → p ≈ 0.17 → RETIRE wins
        let pv = make_precision_with_failure(0, 5, FailureMode::FalseAccept);
        let action = policy.decide(&pv, 5.0, None);
        assert_eq!(action, LifecycleAction::Retire);
    }
}
