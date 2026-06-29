//! Surprise computation for posterior-guided pruner evolution.
//!
//! Computes KL divergence between posterior and prior to detect
//! surprising observations. Uses sigmoid gating (not softmax, per
//! project rules) to trigger lifecycle actions.
//!
//! This goes beyond the paper's fixed thresholds by using
//! surprise magnitude to modulate action sensitivity.

use crate::posterior::precision::PrecisionVector;

/// Computes surprise from precision updates and gates it with sigmoid.
#[derive(Debug, Clone)]
pub struct SurpriseComputer {
    /// Sensitivity parameter β for sigmoid gating.
    /// Higher β = more sensitive to surprise.
    pub beta: f32,
    /// Minimum KL divergence to consider "surprising."
    pub surprise_floor: f32,
}

impl Default for SurpriseComputer {
    fn default() -> Self {
        Self {
            beta: 2.0,
            surprise_floor: 0.1,
        }
    }
}

impl SurpriseComputer {
    /// Create with custom sensitivity.
    pub fn new(beta: f32, surprise_floor: f32) -> Self {
        Self {
            beta,
            surprise_floor,
        }
    }

    /// Sigmoid function. Uses the numerically stable form:
    /// sigmoid(x) = 1 / (1 + exp(-x)) for x >= 0
    /// sigmoid(x) = exp(x) / (1 + exp(x)) for x < 0
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }

    /// Compute surprise gate from a KL divergence value.
    ///
    /// Returns a value in (0, 1) indicating how surprising the observation was.
    /// Uses sigmoid(β × max(0, KL - floor)) to gate the surprise.
    #[inline]
    pub fn surprise_gate(&self, kl_divergence: f32) -> f32 {
        let excess = (kl_divergence - self.surprise_floor).max(0.0);
        Self::sigmoid(self.beta * excess)
    }

    /// Check if the surprise exceeds the action trigger threshold.
    ///
    /// Paper uses fixed `failure_count >= 2`. We use:
    /// `sigmoid(β × max(0, KL - floor)) > threshold`
    #[inline]
    pub fn should_trigger(&self, kl_divergence: f32, threshold: f32) -> bool {
        self.surprise_gate(kl_divergence) > threshold
    }

    /// Compute precision-weighted surprise for an arm.
    ///
    /// Combines the precision state with a novel observation's KL divergence.
    /// Low-precision arms (uncertain) produce less surprise because we
    /// expect them to shift. High-precision arms (certain) produce more
    /// surprise because shifts are unexpected.
    ///
    /// Returns (raw_kl, gate_value).
    pub fn compute_surprise(&self, precision: &PrecisionVector, kl_divergence: f32) -> (f32, f32) {
        // Weight surprise by inverse precision: uncertain arms → less surprise
        let avg_precision = precision.avg_precision();
        let precision_weight = 1.0 / (1.0 + avg_precision); // Diminishes as precision grows
        let weighted_kl = kl_divergence * (1.0 - precision_weight);
        let gate = self.surprise_gate(weighted_kl);
        (weighted_kl, gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        // Sigmoid should be in (0, 1] for finite inputs
        // Note: f32 precision means sigmoid(100) rounds to exactly 1.0
        for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let s = SurpriseComputer::sigmoid(x);
            assert!(s > 0.0 && s <= 1.0, "sigmoid({x}) = {s}, expected (0, 1]");
        }
    }

    #[test]
    fn sigmoid_midpoint() {
        let s = SurpriseComputer::sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigmoid_monotone() {
        let s1 = SurpriseComputer::sigmoid(-1.0);
        let s2 = SurpriseComputer::sigmoid(0.0);
        let s3 = SurpriseComputer::sigmoid(1.0);
        assert!(s1 < s2);
        assert!(s2 < s3);
    }

    #[test]
    fn surprise_gate_no_surprise_below_floor() {
        let sc = SurpriseComputer::new(2.0, 0.1);
        let gate = sc.surprise_gate(0.05);
        // Below floor → excess = 0 → sigmoid(0) = 0.5
        assert!((gate - 0.5).abs() < 1e-6);
    }

    #[test]
    fn surprise_gate_increases_with_kl() {
        let sc = SurpriseComputer::new(2.0, 0.1);
        let g1 = sc.surprise_gate(0.5);
        let g2 = sc.surprise_gate(2.0);
        let g3 = sc.surprise_gate(5.0);
        assert!(g1 < g2);
        assert!(g2 < g3);
    }

    #[test]
    fn should_trigger_at_threshold() {
        let sc = SurpriseComputer::new(2.0, 0.1);
        // Low KL → should not trigger
        assert!(!sc.should_trigger(0.05, 0.7));
        // High KL → should trigger
        assert!(sc.should_trigger(5.0, 0.7));
    }

    #[test]
    fn precision_weighted_surprise_high_precision_more_surprising() {
        let sc = SurpriseComputer::new(2.0, 0.1);

        // Create two precision vectors: one with many observations, one with few
        let mut high_prec = PrecisionVector::new();
        let mut low_prec = PrecisionVector::new();
        let obs = [0.5; 8];

        for _ in 0..100 {
            high_prec.update(
                crate::posterior::types::EvidenceOutcome::Success,
                &obs,
                None,
            );
        }
        low_prec.update(
            crate::posterior::types::EvidenceOutcome::Success,
            &obs,
            None,
        );

        let (kl_high, gate_high) = sc.compute_surprise(&high_prec, 1.0);
        let (kl_low, gate_low) = sc.compute_surprise(&low_prec, 1.0);

        // Same raw KL, but high-precision arm should have higher weighted surprise
        assert!(
            kl_high > kl_low,
            "high_prec kl={kl_high}, low_prec kl={kl_low}"
        );
        assert!(
            gate_high > gate_low,
            "high_prec gate={gate_high}, low_prec gate={gate_low}"
        );
    }
}
