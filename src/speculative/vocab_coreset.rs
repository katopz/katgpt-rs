//! DDTree Vocab Coreset — dMoE block-level expert aggregation (Research 161, Plan 181).
//!
//! Aggregates K+1 draft marginals → vocab coreset → restricts DDTree expansion.
//! Reduces branching factor from |V| to |coreset|.

/// Compute vocab coreset from K+1 draft marginals.
///
/// Aggregates marginals by taking max probability per vocab token across
/// all draft positions, then applies top-p to select the coreset.
///
/// # Arguments
/// * `marginals` - K+1 marginals, each of length vocab_size
/// * `p` - Top-p threshold (e.g., 0.95)
/// * `coreset` - Output mask, pre-allocated to vocab_size
///
/// # Returns
/// Number of tokens in the coreset.
pub fn vocab_coreset(marginals: &[&[f32]], p: f32, coreset: &mut [bool]) -> usize {
    if marginals.is_empty() {
        for m in coreset.iter_mut() {
            *m = false;
        }
        return 0;
    }

    let vocab_size = marginals[0].len();
    debug_assert_eq!(coreset.len(), vocab_size);

    let mut max_scores = vec![0.0f32; vocab_size];

    // Aggregate: max probability per token across positions
    for marginal in marginals {
        for (v, &score) in marginal.iter().enumerate() {
            max_scores[v] = max_scores[v].max(score);
        }
    }

    // Sort by score descending
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_by(|&a, &b| {
        max_scores[b]
            .partial_cmp(&max_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total: f32 = max_scores.iter().map(|s| s.max(0.0)).sum();
    if total <= 0.0 {
        for m in coreset.iter_mut() {
            *m = true;
        }
        return vocab_size;
    }

    // Clear coreset
    for m in coreset.iter_mut() {
        *m = false;
    }

    let mut cumsum = 0.0f32;
    let mut selected = 0usize;

    for &idx in &indices {
        cumsum += max_scores[idx].max(0.0) / total;
        coreset[idx] = true;
        selected += 1;
        if cumsum >= p {
            break;
        }
    }

    selected
}

/// Gate for delta sparse matmul based on routing overlap.
///
/// Only enable delta sparse when there's significant overlap between
/// consecutive tokens' active neurons.
///
/// # Arguments
/// * `step_overlap` - Overlap ratios between consecutive token routing
///
/// # Returns
/// Whether delta sparse should be used.
pub fn should_use_delta_sparse(step_overlap: &[f64]) -> bool {
    if step_overlap.is_empty() {
        return false;
    }
    let avg_overlap: f64 = step_overlap.iter().sum::<f64>() / step_overlap.len() as f64;
    avg_overlap > 0.30
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_coreset_basic() {
        // 3 tokens, vocab_size = 5
        let m0: Vec<f32> = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        let m1: Vec<f32> = vec![0.1, 0.6, 0.2, 0.05, 0.05];
        let marginals: Vec<&[f32]> = vec![&m0, &m1];

        let mut coreset = vec![false; 5];
        let count = vocab_coreset(&marginals, 0.9, &mut coreset);

        assert!(count > 0, "should select at least 1 token");
        assert!(count < 5, "should not select all tokens at p=0.9");
        // Token 0 and 1 have highest max probs
        assert!(coreset[0], "token 0 should be in coreset (max=0.5)");
        assert!(coreset[1], "token 1 should be in coreset (max=0.6)");
    }

    #[test]
    fn test_vocab_coreset_empty_marginals() {
        let mut coreset = vec![false; 10];
        let count = vocab_coreset(&[], 0.9, &mut coreset);
        assert_eq!(count, 0);
        for m in &coreset {
            assert!(!m, "all should be false for empty marginals");
        }
    }

    #[test]
    fn test_vocab_coreset_p1_selects_all() {
        let m0: Vec<f32> = vec![0.3, 0.3, 0.2, 0.1, 0.1];
        let marginals: Vec<&[f32]> = vec![&m0];

        let mut coreset = vec![false; 5];
        let count = vocab_coreset(&marginals, 1.0, &mut coreset);

        assert_eq!(count, 5, "p=1.0 should select all tokens");
    }

    #[test]
    fn test_should_use_delta_sparse_high_overlap() {
        let overlaps = vec![0.8, 0.75, 0.85];
        assert!(
            should_use_delta_sparse(&overlaps),
            "high overlap should enable delta sparse"
        );
    }

    #[test]
    fn test_should_use_delta_sparse_low_overlap() {
        let overlaps = vec![0.1, 0.15, 0.2];
        assert!(
            !should_use_delta_sparse(&overlaps),
            "low overlap should not enable delta sparse"
        );
    }

    #[test]
    fn test_should_use_delta_sparse_empty() {
        assert!(
            !should_use_delta_sparse(&[]),
            "empty should not enable delta sparse"
        );
    }
}
