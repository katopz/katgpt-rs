//! GRM gate computation for segment retrieval (Plan 223b).
//!
//! Uses sigmoid (NOT softmax) per project convention.

/// Compute GRM gates for all cached segments.
/// γ(i) = sigmoid(dot(query, summary(S(i))))
///
/// Returns one gate value per segment, all in [0, 1].
pub fn compute_gates(query: &[f32], summaries: &[&[f32]]) -> Vec<f32> {
    summaries
        .iter()
        .map(|s| {
            let dot = dot_product(query, s);
            sigmoid(dot)
        })
        .collect()
}

/// Top-k gates: only keep the k highest gate values.
/// Returns (segment_index, gate_value) pairs sorted by gate descending.
pub fn top_k_gates(query: &[f32], summaries: &[&[f32]], k: usize) -> Vec<(usize, f32)> {
    let gates = compute_gates(query, summaries);
    let mut indexed: Vec<(usize, f32)> = gates.into_iter().enumerate().collect();
    // Partial sort: keep top k
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed
}

/// Sigmoid function: σ(x) = 1 / (1 + exp(-x))
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dot product of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let min_len = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..min_len {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_range() {
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let s = sigmoid(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({}) = {} not in (0,1)", x, s);
        }
    }

    #[test]
    fn test_sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_gates() {
        let query = vec![1.0, 0.0];
        let summaries: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];
        let gates = compute_gates(&query, &summaries);
        assert_eq!(gates.len(), 2);
        // First summary is aligned with query → higher gate
        assert!(gates[0] > gates[1]);
    }

    #[test]
    fn test_top_k_gates() {
        let query = vec![1.0, 0.0];
        let summaries: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0], &[0.5, 0.5]];
        let top = top_k_gates(&query, &summaries, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // first summary has highest gate
    }

    /// Sigmoid gates are INDEPENDENT (not softmax — they don't sum to 1.0).
    /// If this were softmax, aligned vectors would suppress unaligned ones.
    /// With sigmoid, each gate is independently in (0, 1).
    #[test]
    fn test_gates_are_sigmoid_not_softmax() {
        let query = vec![1.0, 0.0];
        let summaries: Vec<&[f32]> = vec![&[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]];
        let gates = compute_gates(&query, &summaries);

        // All three summaries are identical and aligned → all gates should be equal and high.
        // With softmax, they'd each be 1/3 ≈ 0.333. With sigmoid, they're all ~0.731.
        assert!(
            gates.iter().all(|&g| g > 0.5),
            "sigmoid gates for aligned summaries should all be > 0.5, got {:?}",
            gates
        );

        // Gates do NOT sum to 1.0 (definitively NOT softmax)
        let sum: f32 = gates.iter().sum();
        assert!(
            sum > 1.5,
            "sigmoid gates should sum to > 1.5 for 3 aligned summaries (not 1.0 like softmax), got {}",
            sum
        );

        // Each gate independently in (0, 1)
        for &g in &gates {
            assert!(g > 0.0 && g < 1.0, "gate {} not in (0,1)", g);
        }
    }
}
