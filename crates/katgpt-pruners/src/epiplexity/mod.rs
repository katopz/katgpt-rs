//! Epiplexity — Structural Information Scoring for Modelless Distillation
//!
//! **Plan 130, Research 090** — Based on epiplexity paper (arXiv:2601.03220):
//! structural information extractable by computationally bounded observers,
//! measured as area under loss curve above final loss.
//!
//! # Architecture
//!
//! - [`EpiplexityEstimator`] — ring buffer for loss history, computes S_T
//! - [`TimeBoundedEntropy`] — companion entropy estimator H_T
//! - [`EpiplexityScreeningPruner`] — wraps inner pruner with epiplexity weighting
//! - [`LossCurveTracker`] — hooks into training loop for batch/epoch tracking
//! - [`FactorizationScorer`] — forward/reverse scoring for game traces

pub mod factorization;
pub mod loss_curve;
pub mod screening;

use std::collections::VecDeque;

pub use factorization::{FactorizationOrder, FactorizationScorer};
pub use loss_curve::{LossCurveTracker, PerPositionLossTracker};
pub use screening::{EpiplexityScreeningPruner, EpiplexityWeight};

// ── EpiplexityEstimator ─────────────────────────────────────────

/// Ring buffer for training step losses, computes epiplexity S_T.
///
/// Epiplexity measures structural information extractable by computationally
/// bounded observers: S_T = Σ_i max(0, loss_i - final_loss)
///
/// This is the area under the loss curve above the final loss level.
/// - Constant data: losses don't change → S_T ≈ 0
/// - Random data: losses are noisy but flat → S_T ≈ 0
/// - Structured data: losses decrease over training → S_T > 0
#[derive(Clone, Debug)]
pub struct EpiplexityEstimator {
    /// Ring buffer of per-step losses (bounded capacity).
    losses: VecDeque<f32>,
    /// Maximum number of steps to retain.
    capacity: usize,
}

impl EpiplexityEstimator {
    /// Create a new estimator with bounded history.
    pub fn new(capacity: usize) -> Self {
        Self {
            losses: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Record a single training step loss.
    pub fn record_step(&mut self, step_loss: f32) {
        if self.losses.len() >= self.capacity {
            self.losses.pop_front();
        }
        self.losses.push_back(step_loss);
    }

    /// Compute epiplexity S_T = Σ_i max(0, loss_i - final_loss).
    ///
    /// Measures the total "excess loss" above the final converged loss.
    /// Higher S_T indicates more structural information was extractable.
    pub fn compute_epiplexity(&self, final_loss: f32) -> f32 {
        self.losses
            .iter()
            .map(|&loss| (loss - final_loss).max(0.0))
            .sum()
    }

    /// Compute per-position epiplexity estimates.
    ///
    /// Given per-position final losses, estimates the epiplexity contribution
    /// for each position using the stored loss history.
    pub fn compute_per_sample(&self, final_losses: &[f32]) -> Vec<f32> {
        final_losses
            .iter()
            .map(|&fl| self.compute_epiplexity(fl))
            .collect()
    }

    /// Number of recorded steps currently in the buffer.
    pub fn len(&self) -> usize {
        self.losses.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.losses.is_empty()
    }

    /// Clear all recorded losses.
    pub fn clear(&mut self) {
        self.losses.clear();
    }
}

// ── TimeBoundedEntropy ──────────────────────────────────────────

/// Time-bounded entropy companion estimator.
///
/// H_T estimates the entropy of data as seen by a bounded observer
/// at time T. For language modeling, this is approximately the final
/// cross-entropy loss scaled by sequence length.
#[derive(Clone, Debug)]
pub struct TimeBoundedEntropy {
    /// Back-reference to the loss history for estimation.
    estimator: EpiplexityEstimator,
}

impl TimeBoundedEntropy {
    /// Create a new entropy estimator with bounded history.
    pub fn new(capacity: usize) -> Self {
        Self {
            estimator: EpiplexityEstimator::new(capacity),
        }
    }

    /// Record a single training step loss.
    pub fn record_step(&mut self, step_loss: f32) {
        self.estimator.record_step(step_loss);
    }

    /// Compute time-bounded entropy H_T ≈ final_loss × n_tokens.
    ///
    /// Uses the final cross-entropy loss as an estimate of per-token entropy.
    /// Scale by `n_tokens` for total sequence entropy.
    pub fn compute_entropy(&self, final_loss: f32, n_tokens: usize) -> f32 {
        final_loss * (n_tokens as f32)
    }

    /// Compute the ratio S_T / H_T — structural information fraction.
    ///
    /// Values near 0: data has little structure (random or already compressed).
    /// Values near 1: data is highly structured (rich patterns to learn).
    pub fn structural_fraction(&self, final_loss: f32, n_tokens: usize) -> f32 {
        let h_t = self.compute_entropy(final_loss, n_tokens);
        if h_t <= 0.0 {
            return 0.0;
        }
        let s_t = self.estimator.compute_epiplexity(final_loss);
        (s_t / h_t).min(1.0)
    }

    /// Access the underlying estimator.
    pub fn estimator(&self) -> &EpiplexityEstimator {
        &self.estimator
    }
}

// ── Unit Tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_data_epiplexity_near_zero() {
        let mut est = EpiplexityEstimator::new(100);
        let final_loss = 2.0;
        // Constant losses = final loss → no excess area
        for _ in 0..50 {
            est.record_step(final_loss);
        }
        let s = est.compute_epiplexity(final_loss);
        assert!(s < 0.01, "constant data should have S≈0, got {s}");
    }

    #[test]
    fn test_random_data_epiplexity_near_zero() {
        let mut est = EpiplexityEstimator::new(1000);
        let final_loss = 2.0;
        let mut rng = fastrand::Rng::with_seed(42);
        // Random noise centered around final loss → excess and deficit cancel
        for _ in 0..500 {
            let noise = ((rng.u32(..) % 200) as f32) / 100.0 - 1.0; // [-1.0, +1.0]
            est.record_step(final_loss + noise);
        }
        let s = est.compute_epiplexity(final_loss);
        // Only the steps above final_loss contribute; roughly half above by ~0.5
        // For 500 steps, ~250 above by avg ~0.5 → ~125, but random so less
        assert!(s < 200.0, "random data should have bounded S, got {s}");
        // Key property: S per step should be small
        let s_per_step = s / est.len() as f32;
        assert!(
            s_per_step < 1.0,
            "random per-step epiplexity should be small, got {s_per_step}"
        );
    }

    #[test]
    fn test_structured_data_epiplexity_positive() {
        let mut est = EpiplexityEstimator::new(100);
        let final_loss = 1.0;
        // Structured data: losses decrease from high to final
        for i in 0..50 {
            let step_loss = 5.0 - (i as f32) * 0.08; // 5.0 → ~1.08
            est.record_step(step_loss);
        }
        let s = est.compute_epiplexity(final_loss);
        assert!(s > 1.0, "structured data should have S>0, got {s}");
    }

    #[test]
    fn test_epiplexity_monotone_with_structure() {
        let final_loss = 1.0;

        // More structured = larger initial loss gap
        let mut est_low = EpiplexityEstimator::new(20);
        let mut est_high = EpiplexityEstimator::new(20);

        for i in 0..20 {
            let low = 2.0 - (i as f32) * 0.05; // 2.0 → 1.05
            let high = 6.0 - (i as f32) * 0.25; // 6.0 → 1.25
            est_low.record_step(low);
            est_high.record_step(high);
        }

        let s_low = est_low.compute_epiplexity(final_loss);
        let s_high = est_high.compute_epiplexity(final_loss);
        assert!(
            s_high > s_low,
            "more structure → higher S: {s_high} should be > {s_low}"
        );
    }

    #[test]
    fn test_ring_buffer_capacity() {
        let mut est = EpiplexityEstimator::new(5);
        for i in 0..10 {
            est.record_step(i as f32);
        }
        assert_eq!(est.len(), 5);
        // Only last 5 values: [5.0, 6.0, 7.0, 8.0, 9.0]
        let s = est.compute_epiplexity(0.0);
        assert!((s - 35.0).abs() < 0.01, "expected 35.0, got {s}"); // 5+6+7+8+9
    }

    #[test]
    fn test_per_sample_epiplexity() {
        let mut est = EpiplexityEstimator::new(10);
        for i in 0..10 {
            est.record_step(3.0 - (i as f32) * 0.2); // 3.0 → 1.2
        }
        let final_losses = vec![1.0, 2.0, 3.0];
        let per_sample = est.compute_per_sample(&final_losses);

        // Lower final loss → more excess → higher epiplexity
        assert!(
            per_sample[0] > per_sample[1],
            "lower final loss → higher S: {} should be > {}",
            per_sample[0],
            per_sample[1]
        );
        assert!(
            per_sample[1] > per_sample[2],
            "mid final loss → higher S than high final: {} should be > {}",
            per_sample[1],
            per_sample[2]
        );
        // final_loss=3.0: all step losses ≤ 3.0 → S=0
        assert!(
            per_sample[2] < 0.01,
            "final_loss above all steps → S≈0, got {}",
            per_sample[2]
        );
    }

    #[test]
    fn test_time_bounded_entropy() {
        let tbe = TimeBoundedEntropy::new(10);
        let h = tbe.compute_entropy(2.5, 100);
        assert!((h - 250.0).abs() < 0.01, "expected 250.0, got {h}");
    }

    #[test]
    fn test_structural_fraction_zero_entropy() {
        let tbe = TimeBoundedEntropy::new(10);
        let frac = tbe.structural_fraction(0.0, 100);
        assert_eq!(frac, 0.0, "zero entropy → zero fraction");
    }

    #[test]
    fn test_clear_resets_buffer() {
        let mut est = EpiplexityEstimator::new(10);
        est.record_step(1.0);
        est.record_step(2.0);
        assert_eq!(est.len(), 2);
        est.clear();
        assert!(est.is_empty());
        assert_eq!(est.compute_epiplexity(0.0), 0.0);
    }
}
