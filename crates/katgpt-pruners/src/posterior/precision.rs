//! BAKE-style precision vector for posterior tracking.
//!
//! Each pruner arm carries a fixed-size precision vector `[f32; D]`.
//! Precision grows with observations; posterior mean is weighted by
//! precision relative to total. This is the continuous-space analog
//! of the paper's CategoricalBayesState, using BAKE-style
//! sequential Bayesian updates instead of discrete bucket counting.
//!
//! Zero-allocation, SIMD-friendly, fixed-size.

use crate::posterior::types::{EvidenceOutcome, FailureMode};

/// Default dimensionality for precision vectors.
/// 8 dimensions: context_similarity, failure_density, eval_rate,
/// latency_signal, reward_mean, reward_variance, exploration_bonus, domain_coverage.
pub const PRECISION_DIM: usize = 8;

/// A BAKE-style precision vector that tracks per-dimension certainty.
///
/// Each dimension has:
/// - `precision`: how certain we are (grows with observations)
/// - `mean`: the posterior mean (reward/signal in that dimension)
///
/// Update rule (sequential Bayesian):
/// ```text
/// precision_new = precision_old + observation_weight
/// mean_new = mean_old + (observation - mean_old) * (obs_weight / precision_new)
/// ```
#[derive(Debug, Clone)]
pub struct PrecisionVector {
    /// Per-dimension precision (certainty). Grows with observations.
    precision: [f32; PRECISION_DIM],
    /// Per-dimension posterior mean.
    mean: [f32; PRECISION_DIM],
    /// Total observations recorded.
    total_observations: u32,
    /// Beta distribution alpha (success pseudo-count).
    alpha: f32,
    /// Beta distribution beta (failure pseudo-count).
    beta: f32,
    /// Per-failure-mode observation counts.
    failure_mode_counts: [u32; FailureMode::COUNT],
}

#[allow(clippy::derivable_impls)]
impl Default for PrecisionVector {
    fn default() -> Self {
        Self::new()
    }
}

impl PrecisionVector {
    /// Create a new precision vector with uniform prior (precision = 1.0, mean = 0.5).
    pub fn new() -> Self {
        Self {
            precision: [1.0; PRECISION_DIM],
            mean: [0.5; PRECISION_DIM],
            total_observations: 0,
            alpha: 1.0,
            beta: 1.0,
            failure_mode_counts: [0; FailureMode::COUNT],
        }
    }

    /// Create with custom prior.
    pub fn with_prior(precision: f32, mean: f32) -> Self {
        Self {
            precision: [precision; PRECISION_DIM],
            mean: [mean; PRECISION_DIM],
            total_observations: 0,
            alpha: 1.0,
            beta: 1.0,
            failure_mode_counts: [0; FailureMode::COUNT],
        }
    }

    /// Update the precision vector with a new observation.
    ///
    /// Returns the KL divergence between posterior and prior (surprise signal).
    /// Zero-allocation: all updates in-place on fixed-size arrays.
    pub fn update(
        &mut self,
        outcome: EvidenceOutcome,
        observation: &[f32; PRECISION_DIM],
        failure_mode: Option<FailureMode>,
    ) -> f32 {
        self.total_observations += 1;

        // Beta-Bernoulli update (compatible with paper)
        match outcome {
            EvidenceOutcome::Success => self.alpha += 1.0,
            EvidenceOutcome::Failure => self.beta += 1.0,
        }

        // Track failure mode counts
        if let Some(fm) = failure_mode {
            self.failure_mode_counts[fm.as_index()] += 1;
        }

        // Sequential Bayesian update per dimension
        let mut total_kl = 0.0f32;
        let obs_weight = 1.0; // Uniform observation weight

        for ((prec, mean), obs) in self
            .precision
            .iter_mut()
            .zip(self.mean.iter_mut())
            .zip(observation.iter())
        {
            let prior_precision = *prec;
            let prior_mean = *mean;

            // Update precision
            *prec += obs_weight;
            let posterior_precision = *prec;

            // Update mean (precision-weighted)
            let innovation = obs - prior_mean;
            *mean = prior_mean + innovation * (obs_weight / posterior_precision);

            // KL divergence contribution (Gaussian approximation):
            // KL = 0.5 * (precision_ratio - 1 - ln(precision_ratio))
            //      + 0.5 * precision_old * (mean_new - mean_old)^2
            let precision_ratio = posterior_precision / prior_precision;
            let kl_dim = 0.5 * (precision_ratio - 1.0 - precision_ratio.ln())
                + 0.5 * prior_precision * innovation * innovation;
            total_kl += kl_dim;
        }

        // Clamp to prevent NaN from numerical issues
        total_kl.clamp(0.0, f32::MAX)
    }

    /// Overall success probability (Beta-Bernoulli posterior mean).
    pub fn success_probability(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get the precision for a specific dimension.
    pub fn precision_at(&self, dim: usize) -> f32 {
        self.precision[dim]
    }

    /// Get the posterior mean for a specific dimension.
    pub fn mean_at(&self, dim: usize) -> f32 {
        self.mean[dim]
    }

    /// Get the total precision (sum across dimensions).
    pub fn total_precision(&self) -> f32 {
        self.precision.iter().sum()
    }

    /// Get the average precision across dimensions.
    pub fn avg_precision(&self) -> f32 {
        self.total_precision() / PRECISION_DIM as f32
    }

    /// Total number of observations.
    pub fn observations(&self) -> u32 {
        self.total_observations
    }

    /// Alpha (success count + prior).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Beta (failure count + prior).
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Count of a specific failure mode.
    pub fn failure_mode_count(&self, fm: FailureMode) -> u32 {
        self.failure_mode_counts[fm.as_index()]
    }

    /// Max failure mode count (for PATCH trigger: paper uses failure_count >= 2).
    pub fn max_failure_mode_count(&self) -> (FailureMode, u32) {
        let mut best = FailureMode::FalseAccept;
        let mut best_count = 0;
        for (i, &count) in self.failure_mode_counts.iter().enumerate() {
            if count > best_count {
                best_count = count;
                // SAFETY: index is within enum range
                best = unsafe { std::mem::transmute::<u8, FailureMode>(i as u8) };
            }
        }
        (best, best_count)
    }

    /// Check if any failure mode exceeds the given threshold.
    pub fn has_repeated_failure(&self, threshold: u32) -> bool {
        self.failure_mode_counts.iter().any(|&c| c >= threshold)
    }

    /// Compute precision divergence between two vectors.
    /// Returns the max absolute difference in precision across dimensions.
    /// Used for SPLIT trigger: if two arms diverge, they should be split.
    pub fn precision_divergence(&self, other: &Self) -> f32 {
        let mut max_diff = 0.0f32;
        for i in 0..PRECISION_DIM {
            let diff = (self.precision[i] - other.precision[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        max_diff
    }

    /// Check if precision has converged to uninformative (near-zero effective precision).
    /// Used for RETIRE trigger.
    pub fn is_precision_depleted(&self, threshold: f32) -> bool {
        // Effective precision = precision - prior (1.0)
        let effective = self.avg_precision() - 1.0;
        effective < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_uniform_prior() {
        let pv = PrecisionVector::new();
        assert_eq!(pv.observations(), 0);
        assert!((pv.success_probability() - 0.5).abs() < 1e-6);
        assert!((pv.avg_precision() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn update_increments_observations() {
        let mut pv = PrecisionVector::new();
        let obs = [0.8; PRECISION_DIM];
        pv.update(EvidenceOutcome::Success, &obs, None);
        assert_eq!(pv.observations(), 1);
        pv.update(EvidenceOutcome::Failure, &obs, None);
        assert_eq!(pv.observations(), 2);
    }

    #[test]
    fn success_probability_converges() {
        let mut pv = PrecisionVector::new();
        let obs = [0.5; PRECISION_DIM];

        // 9 successes, 1 failure → should converge to ~0.9
        for _ in 0..9 {
            pv.update(EvidenceOutcome::Success, &obs, None);
        }
        pv.update(EvidenceOutcome::Failure, &obs, None);

        // α = 1 (prior) + 9 = 10, β = 1 (prior) + 1 = 2
        // p = 10 / 12 = 0.833...
        assert!((pv.success_probability() - 10.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn precision_grows_with_observations() {
        let mut pv = PrecisionVector::new();
        let obs = [0.5; PRECISION_DIM];
        let initial = pv.precision_at(0);

        pv.update(EvidenceOutcome::Success, &obs, None);
        assert!(pv.precision_at(0) > initial);
    }

    #[test]
    fn mean_updates_toward_observation() {
        let mut pv = PrecisionVector::new();
        // Prior mean = 0.5, observation = 0.9
        let obs = [0.9; PRECISION_DIM];
        pv.update(EvidenceOutcome::Success, &obs, None);

        // Mean should move toward 0.9 from 0.5
        assert!(pv.mean_at(0) > 0.5);
        assert!(pv.mean_at(0) < 0.9); // Not all the way on first observation
    }

    #[test]
    fn failure_mode_tracking() {
        let mut pv = PrecisionVector::new();
        let obs = [0.5; PRECISION_DIM];

        pv.update(
            EvidenceOutcome::Failure,
            &obs,
            Some(FailureMode::FalseAccept),
        );
        pv.update(
            EvidenceOutcome::Failure,
            &obs,
            Some(FailureMode::FalseAccept),
        );
        pv.update(EvidenceOutcome::Failure, &obs, Some(FailureMode::Timeout));

        assert_eq!(pv.failure_mode_count(FailureMode::FalseAccept), 2);
        assert_eq!(pv.failure_mode_count(FailureMode::Timeout), 1);
        assert_eq!(pv.failure_mode_count(FailureMode::BlankOutput), 0);
    }

    #[test]
    fn repeated_failure_detection() {
        let mut pv = PrecisionVector::new();
        let obs = [0.5; PRECISION_DIM];

        assert!(!pv.has_repeated_failure(2));

        pv.update(
            EvidenceOutcome::Failure,
            &obs,
            Some(FailureMode::FalseAccept),
        );
        assert!(!pv.has_repeated_failure(2));

        pv.update(
            EvidenceOutcome::Failure,
            &obs,
            Some(FailureMode::FalseAccept),
        );
        assert!(pv.has_repeated_failure(2));
    }

    #[test]
    fn precision_divergence_between_arms() {
        let mut pv1 = PrecisionVector::new();
        let mut pv2 = PrecisionVector::new();
        let obs = [0.5; PRECISION_DIM];

        // pv1 gets many observations
        for _ in 0..100 {
            pv1.update(EvidenceOutcome::Success, &obs, None);
        }
        // pv2 gets few observations
        pv2.update(EvidenceOutcome::Success, &obs, None);

        let div = pv1.precision_divergence(&pv2);
        assert!(div > 50.0); // Should be large
    }

    #[test]
    fn kl_surprise_is_nonnegative() {
        let mut pv = PrecisionVector::new();
        let obs = [0.8; PRECISION_DIM];
        let kl = pv.update(EvidenceOutcome::Success, &obs, None);
        assert!(kl >= 0.0);
    }

    #[test]
    fn max_failure_mode() {
        let mut pv = PrecisionVector::new();
        let obs = [0.5; PRECISION_DIM];

        pv.update(
            EvidenceOutcome::Failure,
            &obs,
            Some(FailureMode::FalseAccept),
        );
        pv.update(
            EvidenceOutcome::Failure,
            &obs,
            Some(FailureMode::FalseAccept),
        );
        pv.update(EvidenceOutcome::Failure, &obs, Some(FailureMode::Timeout));

        let (fm, count) = pv.max_failure_mode_count();
        assert_eq!(fm, FailureMode::FalseAccept);
        assert_eq!(count, 2);
    }
}
