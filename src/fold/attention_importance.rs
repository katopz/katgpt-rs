//! Attention importance scorer — Plan 195 T2.
//!
//! Ranks reasoning steps by mean attention weight. The idea: steps with
//! higher mean attention from later tokens are more "essential" to the
//! reasoning chain and should be kept during folding.

use super::types::StepBoundary;

/// Attention-based step importance scorer.
///
/// Computes per-step importance as the mean of attention scores within
/// each step's token range. O(n) scan over scores, grouped by boundaries.
#[derive(Debug, Clone)]
pub struct AttentionImportance;

impl AttentionImportance {
    /// Create a new attention importance scorer.
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Compute per-step importance scores from raw attention weights.
    ///
    /// `scores[i]` = attention weight at token position `i`.
    /// `boundaries` = step boundaries (sorted by token_pos).
    ///
    /// Returns `Vec<f32>` where `result[step_index]` = mean attention for that step.
    /// Uses sigmoid normalization to keep scores in [0, 1].
    pub fn score_steps(&self, scores: &[f32], boundaries: &[StepBoundary]) -> Vec<f32> {
        if boundaries.is_empty() || scores.is_empty() {
            return Vec::new();
        }

        let mut step_scores = Vec::with_capacity(boundaries.len());

        for (i, boundary) in boundaries.iter().enumerate() {
            let start = boundary.token_pos;
            let end = match boundaries.get(i + 1) {
                Some(next) => next.token_pos.min(scores.len()),
                None => scores.len(),
            };

            if start >= scores.len() || start >= end {
                step_scores.push(0.0);
                continue;
            }

            let slice = &scores[start..end];
            let mean = slice.iter().sum::<f32>() / slice.len() as f32;
            step_scores.push(sigmoid(mean));
        }

        step_scores
    }
}

/// Sigmoid function for normalization (not softmax per project rules).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Default for AttentionImportance {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_boundaries(positions: &[usize]) -> Vec<StepBoundary> {
        positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| StepBoundary::new(pos, i, false))
            .collect()
    }

    #[test]
    fn test_empty_inputs() {
        let ai = AttentionImportance::new();

        let result = ai.score_steps(&[], &make_boundaries(&[0]));
        assert!(result.is_empty());

        let result = ai.score_steps(&[0.5, 0.3], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_step() {
        let ai = AttentionImportance::new();
        let scores = &[0.5_f32, 0.7, 0.3];
        let boundaries = make_boundaries(&[0]);

        let result = ai.score_steps(scores, &boundaries);
        assert_eq!(result.len(), 1);

        let expected_mean = (0.5 + 0.7 + 0.3) / 3.0;
        let expected = sigmoid(expected_mean);
        assert!((result[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_multiple_steps() {
        let ai = AttentionImportance::new();
        // 10 tokens, 2 steps: [0..5) and [5..10)
        let scores: Vec<f32> = vec![0.9; 5].into_iter().chain(vec![0.1; 5]).collect();
        let boundaries = make_boundaries(&[0, 5]);

        let result = ai.score_steps(&scores, &boundaries);
        assert_eq!(result.len(), 2);

        // Step 0 (high attention) should be much more important than step 1.
        assert!(result[0] > result[1]);
    }

    #[test]
    fn test_sigmoid_range() {
        // Sigmoid maps any real to (0, 1).
        assert!(sigmoid(0.0) > 0.49 && sigmoid(0.0) < 0.51);
        assert!(sigmoid(-100.0) < 0.01);
        assert!(sigmoid(100.0) > 0.99);
    }

    #[test]
    fn test_boundary_beyond_scores() {
        let ai = AttentionImportance::new();
        let scores = &[0.5_f32, 0.3];
        // Boundary starts beyond scores length.
        let boundaries = make_boundaries(&[10]);

        let result = ai.score_steps(scores, &boundaries);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0); // Start beyond scores → 0.0
    }

    #[test]
    fn test_default_trait() {
        let _ai = AttentionImportance;
    }

    #[test]
    fn test_score_ordering_preserved() {
        let ai = AttentionImportance::new();
        // 6 tokens, 3 steps of 2 tokens each.
        let scores: Vec<f32> = vec![0.1, 0.1, 0.5, 0.5, 0.9, 0.9];
        let boundaries = make_boundaries(&[0, 2, 4]);

        let result = ai.score_steps(&scores, &boundaries);
        assert_eq!(result.len(), 3);

        // Importance should be monotonically increasing.
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }
}
