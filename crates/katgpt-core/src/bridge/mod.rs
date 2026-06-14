//! ActionBridge — generic latent-to-raw action bridge (Plan 262).
//!
//! Bridges latent Q-values to raw game actions via sigmoid-gated projection.
//! Any game with NPC AI that selects discrete actions from a continuous latent
//! space uses this.
//!
//! Zero allocation, fixed-size. Uses `sigmoid(dot())` — never softmax.

// Sigmoid delegates to shared crate::simd::fast_sigmoid (bounded (0,1), libm-exp).

/// Dot product of an `f32` Q-value vector with an `i8` ternary direction vector.
///
/// `sum(q_values[i] * direction[i] as f32)`
#[inline(always)]
fn dot_f32_i8<const D: usize>(q_values: &[f32; D], direction: &[i8; D]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..D {
        // FMA: acc = q[i] * dir[i] + acc (single rounding when target has FMA).
        acc = (q_values[i]).mul_add(direction[i] as f32, acc);
    }
    acc
}

/// Bridges latent Q-values to raw game actions via sigmoid-gated projection.
///
/// Generic over action space size (`A`) and latent dimension (`D`).
///
/// For each action `a`: `score[a] = sigmoid(dot(q_values, action_directions[a]))`.
/// Actions with confidence below `threshold` are suppressed.
///
/// Zero allocation, fixed-size. Uses `sigmoid(dot())` — never softmax.
///
/// ## Type Parameters
///
/// - `A` — Number of actions in the action space (e.g., 6 for abilities).
/// - `D` — Dimension of the Q-value/latent vector (e.g., 8 for HLA state).
pub struct ActionBridge<const A: usize, const D: usize> {
    /// Direction vectors per action (ternary `{-1, 0, +1}`).
    action_directions: [[i8; D]; A],
    /// Confidence threshold — actions below this are suppressed.
    threshold: f32,
}

impl<const A: usize, const D: usize> ActionBridge<A, D> {
    /// Creates a new `ActionBridge` from direction vectors and a confidence threshold.
    ///
    /// The threshold gates which actions are considered valid. Actions whose
    /// projected sigmoid score falls below the threshold are suppressed.
    #[inline]
    pub fn new(directions: [[i8; D]; A], threshold: f32) -> Self {
        Self {
            action_directions: directions,
            threshold,
        }
    }

    /// Selects the best action from Q-values.
    ///
    /// For each action: `score[a] = sigmoid(dot(q_values, action_directions[a]))`.
    /// Returns `(best_action_index, confidence_score)`.
    ///
    /// If the best confidence is below `threshold`, the returned index points
    /// to the best action anyway (caller can inspect confidence to decide
    /// whether to suppress).
    #[inline]
    pub fn select_action(&self, q_values: &[f32; D]) -> (usize, f32) {
        if A == 0 {
            return (0, 0.0);
        }

        let mut best_idx: usize = 0;
        let mut best_score = f32::MIN;

        for a in 0..A {
            let dot = dot_f32_i8(q_values, &self.action_directions[a]);
            let score = crate::simd::fast_sigmoid(dot);
            if score > best_score {
                best_score = score;
                best_idx = a;
            }
        }

        (best_idx, best_score)
    }

    /// Fills `out` with the top-K actions sorted by confidence (descending).
    ///
    /// Returns the number of items written (capped at `min(k, A)` and `out.len()`).
    /// Uses insertion sort — K is typically small (≤ 8), so this is optimal
    /// and avoids heap allocation.
    ///
    /// The caller provides the buffer; no allocation occurs inside.
    #[inline]
    pub fn select_top_k(&self, q_values: &[f32; D], k: usize, out: &mut [(usize, f32)]) -> usize {
        if A == 0 {
            return 0;
        }

        let write_count = k.min(A).min(out.len());
        if write_count == 0 {
            return 0;
        }

        // Compute all scores into a stack buffer.
        let mut scores = [0.0f32; A];
        for (a, score) in scores.iter_mut().enumerate() {
            let dot = dot_f32_i8(q_values, &self.action_directions[a]);
            *score = crate::simd::fast_sigmoid(dot);
        }

        // Track which action indices have already been selected.
        let mut used = [false; A];

        // Select top-K via linear scan + insertion into output buffer.
        // This is O(K * A) — efficient for small K.
        for entry in out.iter_mut().take(write_count) {
            let mut best_idx = 0usize;
            let mut best_score = f32::MIN;

            for a in 0..A {
                if used[a] {
                    continue;
                }
                if scores[a] > best_score {
                    best_score = scores[a];
                    best_idx = a;
                }
            }

            used[best_idx] = true;
            *entry = (best_idx, best_score);
        }

        write_count
    }

    /// Returns the confidence threshold used for suppression.
    #[inline]
    pub const fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl<const A: usize, const D: usize> Default for ActionBridge<A, D> {
    #[inline]
    fn default() -> Self {
        Self::new([[0; D]; A], 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a 3-action, 2-dim bridge with known directions.
    fn make_bridge() -> ActionBridge<3, 2> {
        // Action 0: prefers positive q0
        // Action 1: prefers negative q0 (avoid)
        // Action 2: prefers positive q1
        let directions: [[i8; 2]; 3] = [[1, 0], [-1, 0], [0, 1]];
        ActionBridge::new(directions, 0.5)
    }

    #[test]
    fn test_select_action_deterministic() {
        let bridge = make_bridge();

        let q: [f32; 2] = [5.0, 1.0];
        let (best_idx, confidence) = bridge.select_action(&q);

        // dot for action 0 = 5*1 + 1*0 = 5 → sigmoid(5) ≈ 0.993
        // dot for action 1 = -5 → sigmoid ≈ 0.007
        // dot for action 2 = 1 → sigmoid(1) ≈ 0.731
        assert_eq!(best_idx, 0, "action 0 should win with large positive q0");
        assert!(
            confidence > 0.99,
            "confidence should be near 1 for dot=5, got {confidence}"
        );

        // Repeated calls must be identical (deterministic).
        let (idx2, conf2) = bridge.select_action(&q);
        assert_eq!(best_idx, idx2);
        assert!((confidence - conf2).abs() < 1e-7);
    }

    #[test]
    fn test_select_action_all_zero_q() {
        let bridge = make_bridge();
        let q: [f32; 2] = [0.0, 0.0];
        let (best_idx, confidence) = bridge.select_action(&q);

        // All dots = 0 → all sigmoids = 0.5. First action wins on tie.
        assert_eq!(best_idx, 0);
        assert!(
            (confidence - 0.5).abs() < 1e-5,
            "sigmoid(0) = 0.5, got {confidence}"
        );
    }

    #[test]
    fn test_threshold_suppression() {
        let bridge = make_bridge();

        // With q = [0, 0], best confidence = 0.5 which equals threshold.
        let q: [f32; 2] = [0.0, 0.0];
        let (_, confidence) = bridge.select_action(&q);

        // The bridge returns the best action regardless; caller checks threshold.
        // Verify threshold accessor.
        assert!(
            confidence <= bridge.threshold() + 1e-5,
            "confidence {confidence} should be at or below threshold {}",
            bridge.threshold()
        );
        assert_eq!(bridge.threshold(), 0.5);
    }

    #[test]
    fn test_threshold_suppression_low_confidence() {
        let directions: [[i8; 2]; 2] = [[-1, -1], [-1, -1]];
        let bridge = ActionBridge::new(directions, 0.9);

        let q: [f32; 2] = [5.0, 5.0];
        let (best_idx, confidence) = bridge.select_action(&q);

        // dot = -10 for both → sigmoid ≈ 0.00005, well below threshold.
        assert_eq!(best_idx, 0); // first wins on tie
        assert!(
            confidence < 0.01,
            "confidence {confidence} should be near 0 for dot=-10"
        );
        assert!(confidence < bridge.threshold());
    }

    #[test]
    fn test_select_top_k_sorted_descending() {
        let bridge = make_bridge();
        let q: [f32; 2] = [5.0, 1.0];

        let mut out = [(0usize, 0.0f32); 3];
        let count = bridge.select_top_k(&q, 3, &mut out);

        assert_eq!(count, 3);
        // Descending by confidence
        assert!(
            out[0].1 >= out[1].1,
            "not sorted desc: {} vs {}",
            out[0].1,
            out[1].1
        );
        assert!(
            out[1].1 >= out[2].1,
            "not sorted desc: {} vs {}",
            out[1].1,
            out[2].1
        );

        // Action 0 (dot=5) should be first.
        assert_eq!(out[0].0, 0);
    }

    #[test]
    fn test_select_top_k_k_exceeds_a() {
        let bridge = make_bridge();
        let q: [f32; 2] = [1.0, 1.0];

        // Request k=10 but only A=3 actions exist.
        let mut out = [(0usize, 0.0f32); 10];
        let count = bridge.select_top_k(&q, 10, &mut out);

        assert_eq!(count, 3, "should only return A items, not k");

        // Verify descending order.
        for i in 1..count {
            assert!(out[i - 1].1 >= out[i].1);
        }
    }

    #[test]
    fn test_select_top_k_k_zero() {
        let bridge = make_bridge();
        let q: [f32; 2] = [1.0, 1.0];

        let mut out = [(0usize, 0.0f32); 3];
        let count = bridge.select_top_k(&q, 0, &mut out);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_select_top_k_small_buffer() {
        let bridge = make_bridge();
        let q: [f32; 2] = [5.0, 1.0];

        // Buffer smaller than k — should cap to buffer size.
        let mut out = [(0usize, 0.0f32); 1];
        let count = bridge.select_top_k(&q, 3, &mut out);

        assert_eq!(count, 1, "should cap to out.len()");
        assert_eq!(out[0].0, 0, "top-1 should be action 0");
    }

    #[test]
    fn test_select_top_k_all_distinct_indices() {
        let bridge = make_bridge();
        let q: [f32; 2] = [3.0, 2.0];

        let mut out = [(0usize, 0.0f32); 3];
        let count = bridge.select_top_k(&q, 3, &mut out);
        assert_eq!(count, 3);

        // All indices should be distinct.
        let mut indices = [out[0].0, out[1].0, out[2].0];
        indices.sort();
        assert_eq!(indices, [0, 1, 2], "each action should appear exactly once");
    }

    #[test]
    fn test_default_threshold_is_zero() {
        let bridge: ActionBridge<2, 2> = ActionBridge::default();
        assert_eq!(bridge.threshold(), 0.0);
    }
}
