//! Precision-Aware Speculative Drafting (PASD).
//! Detects when draft tokens are near quantization boundaries
//! and applies a penalty to improve draft acceptance rate.
//! Feature-gated behind `precision_aware_draft`.

/// Boundary penalty scorer for speculative drafting.
/// Penalizes draft tokens whose logits are close to quantization grid boundaries.
#[derive(Debug, Clone)]
pub struct BoundaryPenalty {
    /// Weight of the boundary penalty in draft scoring.
    pub penalty_weight: f32,
    /// Quantization scale factor (e.g., max_abs / quant_levels).
    pub quant_scale: f32,
    /// Number of quantization levels (e.g., 256 for INT8).
    pub quant_levels: u32,
    /// Epsilon: how close to a boundary counts as "near boundary".
    pub boundary_epsilon: f32,
}

impl Default for BoundaryPenalty {
    fn default() -> Self {
        Self {
            penalty_weight: 0.1,
            quant_scale: 1.0 / 127.0,
            quant_levels: 256,
            boundary_epsilon: 0.05,
        }
    }
}

impl BoundaryPenalty {
    pub fn new(quant_levels: u32, quant_scale: f32) -> Self {
        Self {
            penalty_weight: 0.1,
            quant_scale,
            quant_levels,
            boundary_epsilon: 0.05,
        }
    }

    /// Compute how close a logit value is to the nearest quantization boundary.
    /// Returns 0.0 if far from boundary, 1.0 if exactly on boundary.
    pub fn boundary_proximity(&self, logit: f32) -> f32 {
        // Quantize to grid
        let quantized = (logit / self.quant_scale).round() * self.quant_scale;
        // Distance to nearest grid point
        let dist = (logit - quantized).abs();
        // How close to the midpoint between grid points (the boundary)
        let half_scale = self.quant_scale * 0.5;
        let boundary_dist = (dist - half_scale).abs();

        // Sigmoid-based proximity: near boundary → high proximity
        if boundary_dist < self.boundary_epsilon * self.quant_scale {
            1.0 / (1.0 + (boundary_dist / self.quant_scale * 20.0 - 5.0).exp())
        } else {
            0.0
        }
    }

    /// Compute boundary score for a single token's logits.
    /// Higher score = more penalty (closer to boundaries).
    pub fn compute_boundary_score(&self, token_logits: &[f32]) -> f32 {
        let total_proximity: f32 = token_logits
            .iter()
            .map(|&l| self.boundary_proximity(l))
            .sum();
        total_proximity / token_logits.len().max(1) as f32
    }

    /// Apply boundary penalty to draft scores.
    /// Returns modified scores (higher is better, penalty reduces score).
    pub fn apply_penalty(&self, draft_scores: &mut [f32], logits_per_token: &[Vec<f32>]) {
        for (score, token_logits) in draft_scores.iter_mut().zip(logits_per_token.iter()) {
            let boundary_score = self.compute_boundary_score(token_logits);
            // Penalty reduces the score proportionally
            *score -= self.penalty_weight * boundary_score * score.abs();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_proximity_on_grid() {
        let bp = BoundaryPenalty::new(256, 1.0 / 127.0);
        // Value exactly on quantization grid
        let on_grid = bp.boundary_proximity(0.0);
        // Should be low (not near boundary)
        assert!(on_grid < 0.5);
    }

    #[test]
    fn test_boundary_proximity_at_boundary() {
        let bp = BoundaryPenalty::new(256, 1.0 / 127.0);
        // Value at midpoint between grid points
        let half = bp.quant_scale * 0.5;
        let at_boundary = bp.boundary_proximity(half);
        // Should be high (near boundary)
        assert!(at_boundary > 0.3);
    }

    #[test]
    fn test_compute_boundary_score() {
        let bp = BoundaryPenalty::default();
        let logits = vec![0.5, 0.3, 0.7, 0.1];
        let score = bp.compute_boundary_score(&logits);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_apply_penalty_reduces_scores() {
        let bp = BoundaryPenalty {
            penalty_weight: 0.5,
            ..Default::default()
        };
        let mut scores = vec![1.0, 0.8, 0.6];
        let logits = vec![vec![0.5, 0.3], vec![0.5, 0.3], vec![0.5, 0.3]];
        let original = scores.clone();
        bp.apply_penalty(&mut scores, &logits);

        // Scores should be reduced or unchanged
        for (i, &s) in scores.iter().enumerate() {
            assert!(s <= original[i] + 1e-6);
        }
    }

    #[test]
    fn test_default_config() {
        let bp = BoundaryPenalty::default();
        assert_eq!(bp.quant_levels, 256);
        assert!(bp.penalty_weight > 0.0);
    }
}
