//! EpiplexityScreeningPruner — wraps inner ScreeningPruner with epiplexity weighting.
//!
//! Blend formula: `inner.relevance() * (1.0 - α) + epiplexity_weight * α`
//!
//! - α = 0: pure inner pruner (baseline)
//! - α = 1: pure epiplexity scoring
//! - α ∈ (0, 1): blended

use crate::epiplexity::EpiplexityEstimator;
use katgpt_speculative::ScreeningPruner;

// ── EpiplexityWeight ────────────────────────────────────────────

/// How to derive the epiplexity weight for blending with inner pruner.
#[derive(Clone, Debug)]
#[derive(Default)]
pub enum EpiplexityWeight {
    /// Uniform weight — returns 1.0 (baseline, no epiplexity modulation).
    #[default]
    Uniform,
    /// Weight by |loss_before - loss_after| at the position.
    ///
    /// Maps absolute loss drop through sigmoid to [0, 1].
    LossDrop,
    /// Weight by cumulative epiplexity area contribution.
    ///
    /// Normalized by total epiplexity sum to produce a relative weight.
    CumulativeArea,
}


// ── EpiplexityScreeningPruner ───────────────────────────────────

/// Wraps any `ScreeningPruner` and blends its relevance with epiplexity signal.
///
/// ```text
/// blended_relevance = inner.relevance() * (1 - α) + epiplexity_signal * α
/// ```
///
/// When α = 0, this is equivalent to the inner pruner.
/// When α = 1, only the epiplexity signal is used.
pub struct EpiplexityScreeningPruner<P: ScreeningPruner> {
    /// Inner domain-specific pruner.
    inner: P,
    /// Blend factor: 0 = pure inner, 1 = pure epiplexity.
    alpha: f32,
    /// How to compute the epiplexity weight.
    weight_mode: EpiplexityWeight,
    /// Loss history estimator.
    estimator: EpiplexityEstimator,
    /// Final converged loss (set externally or estimated).
    final_loss: f32,
    /// Per-position loss drops for LossDrop mode.
    position_drops: Vec<f32>,
}

impl<P: ScreeningPruner> EpiplexityScreeningPruner<P> {
    /// Create a new epiplexity-screening pruner wrapping `inner`.
    pub fn new(inner: P, alpha: f32, weight_mode: EpiplexityWeight, capacity: usize) -> Self {
        Self {
            inner,
            alpha: alpha.clamp(0.0, 1.0),
            weight_mode,
            estimator: EpiplexityEstimator::new(capacity),
            final_loss: 0.0,
            position_drops: Vec::new(),
        }
    }

    /// Set the final converged loss for epiplexity computation.
    pub fn set_final_loss(&mut self, final_loss: f32) {
        self.final_loss = final_loss;
    }

    /// Record a training step loss into the estimator.
    pub fn record_step(&mut self, step_loss: f32) {
        self.estimator.record_step(step_loss);
    }

    /// Record per-position loss drops for LossDrop weighting mode.
    pub fn set_position_drops(&mut self, drops: Vec<f32>) {
        self.position_drops = drops;
    }

    /// Compute the epiplexity weight based on the current mode.
    fn epiplexity_signal(&self, depth: usize) -> f32 {
        match &self.weight_mode {
            EpiplexityWeight::Uniform => 1.0,
            EpiplexityWeight::LossDrop => {
                let drop = self.position_drops.get(depth).copied().unwrap_or(0.0);
                // Sigmoid-like mapping: large drop → weight near 1
                sigmoid(drop)
            }
            EpiplexityWeight::CumulativeArea => {
                let s_total = self.estimator.compute_epiplexity(self.final_loss);
                if s_total <= 0.0 {
                    return 0.0;
                }
                // Per-depth contribution scales linearly with total
                let per_depth = s_total / (self.estimator.len().max(1) as f32);
                sigmoid(per_depth)
            }
        }
    }

    /// Access the underlying estimator.
    pub fn estimator(&self) -> &EpiplexityEstimator {
        &self.estimator
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Get the current alpha blend factor.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set the alpha blend factor.
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }
}

impl<P: ScreeningPruner> ScreeningPruner for EpiplexityScreeningPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let inner_rel = self.inner.relevance(depth, token_idx, parent_tokens);
        let epi_signal = self.epiplexity_signal(depth);
        inner_rel * (1.0 - self.alpha) + epi_signal * self.alpha
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Sigmoid function mapping ℝ → (0, 1).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial pruner: always returns 1.0.
    struct UnitPruner;

    impl ScreeningPruner for UnitPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    /// Trivial pruner: always returns 0.5.
    struct HalfPruner;

    impl ScreeningPruner for HalfPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            0.5
        }
    }

    #[test]
    fn test_alpha_zero_preserves_inner() {
        let pruner = EpiplexityScreeningPruner::new(
            UnitPruner,
            0.0, // α = 0
            EpiplexityWeight::Uniform,
            100,
        );
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (rel - 1.0).abs() < 1e-6,
            "α=0 should preserve inner, got {rel}"
        );
    }

    #[test]
    fn test_alpha_one_uses_epiplexity() {
        let mut pruner = EpiplexityScreeningPruner::new(
            UnitPruner,
            1.0, // α = 1
            EpiplexityWeight::Uniform,
            100,
        );
        pruner.set_final_loss(1.0);
        let rel = pruner.relevance(0, 0, &[]);
        // Uniform weight returns 1.0
        assert!(
            (rel - 1.0).abs() < 1e-6,
            "α=1 Uniform should be 1.0, got {rel}"
        );
    }

    #[test]
    fn test_blending_half_alpha() {
        let mut pruner = EpiplexityScreeningPruner::new(
            HalfPruner, // inner = 0.5
            0.5,        // α = 0.5
            EpiplexityWeight::Uniform,
            100,
        );
        pruner.set_final_loss(1.0);
        let rel = pruner.relevance(0, 0, &[]);
        // 0.5 * (1 - 0.5) + 1.0 * 0.5 = 0.25 + 0.5 = 0.75
        assert!(
            (rel - 0.75).abs() < 1e-6,
            "α=0.5 blend should be 0.75, got {rel}"
        );
    }

    #[test]
    fn test_loss_drop_weight_mode() {
        let mut pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 1.0, EpiplexityWeight::LossDrop, 100);
        pruner.set_final_loss(1.0);
        pruner.set_position_drops(vec![0.0, 2.0, 10.0]);

        // depth 0: drop=0.0 → sigmoid(0) = 0.5
        let rel0 = pruner.relevance(0, 0, &[]);
        assert!(
            (rel0 - 0.5).abs() < 1e-4,
            "drop=0 → sigmoid=0.5, got {rel0}"
        );

        // depth 1: drop=2.0 → sigmoid(2) ≈ 0.88
        let rel1 = pruner.relevance(1, 0, &[]);
        assert!(rel1 > 0.8, "drop=2 → sigmoid>0.8, got {rel1}");

        // depth 2: drop=10.0 → sigmoid(10) ≈ 0.99995
        let rel2 = pruner.relevance(2, 0, &[]);
        assert!(rel2 > 0.99, "drop=10 → sigmoid>0.99, got {rel2}");
    }

    #[test]
    fn test_cumulative_area_weight_mode() {
        let mut pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 1.0, EpiplexityWeight::CumulativeArea, 100);
        // Record structured losses
        for i in 0..10 {
            pruner.record_step(5.0 - (i as f32) * 0.4);
        }
        pruner.set_final_loss(1.0);

        let rel = pruner.relevance(0, 0, &[]);
        // Non-zero epiplexity → sigmoid of positive per_depth → > 0.5
        assert!(
            rel > 0.5,
            "cumulative area with structure should be > 0.5, got {rel}"
        );
    }

    #[test]
    fn test_cumulative_area_empty_history() {
        let pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 1.0, EpiplexityWeight::CumulativeArea, 100);
        let rel = pruner.relevance(0, 0, &[]);
        // Empty history → 0.0 signal
        assert!((rel - 0.0).abs() < 1e-6, "empty history → 0.0, got {rel}");
    }

    #[test]
    fn test_set_alpha_clamps() {
        let mut pruner =
            EpiplexityScreeningPruner::new(UnitPruner, 0.5, EpiplexityWeight::Uniform, 10);
        pruner.set_alpha(-1.0);
        assert!((pruner.alpha() - 0.0).abs() < 1e-6);
        pruner.set_alpha(2.0);
        assert!((pruner.alpha() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(-100.0) < 0.01);
        assert!(sigmoid(100.0) > 0.99);
    }
}
