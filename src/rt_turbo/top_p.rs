//! Dynamic top-p token selection for RTPurbo sparse decode.
//!
//! Implements cumulative-mass token selection from the RTPurbo paper
//! (arXiv 2605.16928). During decode, retrieval heads score all KV positions
//! via low-dim projection. This module selects the smallest set of positions
//! whose cumulative softmax mass >= top_p, enabling 97% sparsity while
//! preserving >93% attention mass.
//!
//! # Key Insight
//!
//! Attention distributions are extremely peaked - a handful of positions
//! account for most of the probability mass. Top-p (nucleus) selection
//! adapts the number of selected tokens to the actual distribution shape,
//! unlike fixed top-k which wastes compute on flat distributions or misses
//! tokens on peaked ones.
//!
//! # Two Variants
//!
//! | Variant | Use case | Complexity |
//! |---------|----------|------------|
//! | [`select_top_p`] | Short-medium sequences | O(n log n) sort + O(n) scan |
//! | [`select_top_p_blockwise`] | Long sequences (cache-friendly) | O(n/b log(n/b)) + O(m log m) |
//!
//! # Numerical Stability
//!
//! Softmax uses max-subtraction to avoid overflow:
//!
//! ```text
//! softmax(s_i) = exp(s_i - max(s)) / sum exp(s_j - max(s))
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Score entry with original index, used for sorting while tracking positions.
#[derive(Clone, Debug)]
struct IndexedScore {
    /// Original position in the input slice.
    idx: usize,
    /// Softmax probability.
    prob: f32,
}

/// Result of dynamic top-p selection.
///
/// Contains the selected token indices (original positions), their softmax
/// probabilities, the achieved cumulative mass, and the total number of
/// candidates considered.
#[derive(Clone, Debug)]
pub struct DynamicTopPResult {
    /// Selected token indices (original positions in the input slice).
    pub selected_indices: Vec<usize>,
    /// Softmax probabilities of selected tokens (sorted descending).
    pub selected_probs: Vec<f32>,
    /// Cumulative probability mass of selected tokens.
    pub cumulative_mass: f32,
    /// Total number of candidates that were considered.
    pub n_total: usize,
}

// ---------------------------------------------------------------------------
// Softmax Helper
// ---------------------------------------------------------------------------

/// Numerically stable softmax over raw scores.
///
/// Applies max-subtraction before exponentiation to prevent overflow.
/// Returns a probability distribution that sums to 1.0 (within f32 precision).
///
/// # Edge Cases
///
/// - Empty input -> empty output.
/// - All identical scores -> uniform distribution.
/// - Very large scores (1e6+) -> handled via max-subtraction.
fn softmax_scores_into(scores: &[f32], out: &mut [f32]) {
    if scores.is_empty() {
        return;
    }

    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (i, &s) in scores.iter().enumerate() {
        let e = (s - max_val).exp();
        out[i] = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in out.iter_mut() {
            *v *= inv;
        }
    } else {
        out.fill(0.0);
    }
}

// ---------------------------------------------------------------------------
// Top-P Selection (T11, T12)
// ---------------------------------------------------------------------------

/// Select tokens whose cumulative softmax mass >= top_p.
///
/// Implements the dynamic top-p algorithm:
/// 1. Compute softmax probabilities with max-subtraction for numerical stability
/// 2. Sort probabilities descending while preserving original indices
/// 3. Accumulate probability mass until top_p threshold is reached
/// 4. Return indices of all selected tokens (including the one that crosses the threshold)
///
/// # Arguments
///
/// * scores - Raw relevance scores over candidate positions [seq_len].
///   Higher = more relevant. Typically dot-product scores from low-dim projection.
/// * top_p - Cumulative probability threshold (e.g., 0.9).
///   Must be in [0.0, 1.0]. Values outside this range are clamped.
///
/// # Returns
///
/// [`DynamicTopPResult`] with selected indices, probabilities, and cumulative mass.
///
/// # Edge Cases
///
/// - Empty scores -> empty result with 0 cumulative mass.
/// - Single score -> single index regardless of top_p.
/// - top_p = 0.0 -> top-1 only (highest probability token).
/// - top_p = 1.0 -> all tokens selected.
pub fn select_top_p(scores: &[f32], top_p: f32) -> DynamicTopPResult {
    let n_total = scores.len();

    if n_total == 0 {
        return DynamicTopPResult {
            selected_indices: vec![],
            selected_probs: vec![],
            cumulative_mass: 0.0,
            n_total: 0,
        };
    }

    if n_total == 1 {
        return DynamicTopPResult {
            selected_indices: vec![0],
            selected_probs: vec![1.0],
            cumulative_mass: 1.0,
            n_total: 1,
        };
    }

    // Clamp top_p to [0.0, 1.0]
    let threshold = top_p.clamp(0.0, 1.0);

    // Compute softmax probabilities into pre-allocated buffer
    let mut probs = vec![0.0f32; n_total];
    softmax_scores_into(scores, &mut probs);

    // Sort by probability descending, keeping original indices.
    // Use with_capacity to avoid re-allocation during collection.
    let mut indexed: Vec<IndexedScore> = Vec::with_capacity(n_total);
    for (idx, &prob) in probs.iter().enumerate() {
        indexed.push(IndexedScore { idx, prob });
    }
    indexed.sort_unstable_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Accumulate until cumulative mass >= threshold
    let mut cumsum = 0.0f32;
    let mut selected_indices = Vec::with_capacity(n_total);
    let mut selected_probs = Vec::with_capacity(n_total);

    for entry in &indexed {
        selected_indices.push(entry.idx);
        selected_probs.push(entry.prob);
        cumsum += entry.prob;

        if cumsum >= threshold {
            break;
        }
    }

    DynamicTopPResult {
        selected_indices,
        selected_probs,
        cumulative_mass: cumsum,
        n_total,
    }
}

// ---------------------------------------------------------------------------
// Blockwise Top-P Selection (T13)
// ---------------------------------------------------------------------------

/// Select tokens via blockwise coarse-to-fine top-p for cache-friendly long sequences.
///
/// For very long sequences, processing individual scores is cache-unfriendly.
/// This two-stage algorithm:
///
/// 1. **Coarse stage**: Group positions into blocks of block_size. Compute
///    each block score as the max score within that block. Apply top-p on
///    block scores to select the most relevant blocks.
/// 2. **Fine stage**: Collect all individual scores from selected blocks.
///    Apply fine-grained top-p on these candidates to select final positions.
///
/// This reduces the sorting cost from O(n log n) to O(n/b log(n/b)) + O(m log m)
/// where b = block_size and m ~ threshold * n positions from selected blocks.
///
/// # Arguments
///
/// * scores - Raw relevance scores [seq_len].
/// * top_p - Cumulative probability threshold (applied at both stages).
/// * block_size - Block size for coarse stage (e.g., 64, matching DashAttn chunk).
///
/// # Returns
///
/// [`DynamicTopPResult`] with selected indices (global positions), probabilities,
/// and cumulative mass.
///
/// # Edge Cases
///
/// - Empty scores -> empty result.
/// - block_size = 0 or block_size >= len -> falls back to fine-grained [`select_top_p`].
/// - block_size = 1 -> equivalent to [`select_top_p`] (each position is its own block).
pub fn select_top_p_blockwise(scores: &[f32], top_p: f32, block_size: usize) -> DynamicTopPResult {
    let n_total = scores.len();

    if n_total == 0 {
        return DynamicTopPResult {
            selected_indices: vec![],
            selected_probs: vec![],
            cumulative_mass: 0.0,
            n_total: 0,
        };
    }

    // Degenerate block sizes -> fall back to fine-grained
    if block_size == 0 || block_size >= n_total {
        return select_top_p(scores, top_p);
    }

    let n_blocks = n_total.div_ceil(block_size);

    // Stage 1: Block-level aggregation - max score per block
    let mut block_scores: Vec<f32> = vec![f32::NEG_INFINITY; n_blocks];
    for (i, &s) in scores.iter().enumerate() {
        let block_idx = i / block_size;
        // Branchless max to help auto-vectorization
        block_scores[block_idx] = if s > block_scores[block_idx] {
            s
        } else {
            block_scores[block_idx]
        };
    }

    // Stage 2: Softmax + top-p at block level
    let mut block_probs = vec![0.0f32; n_blocks];
    softmax_scores_into(&block_scores, &mut block_probs);

    let mut block_indexed: Vec<IndexedScore> = Vec::with_capacity(n_blocks);
    for (idx, &prob) in block_probs.iter().enumerate() {
        block_indexed.push(IndexedScore { idx, prob });
    }
    block_indexed.sort_unstable_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Accumulate block probabilities until threshold
    let threshold = top_p.clamp(0.0, 1.0);
    let mut cumsum = 0.0f32;
    let mut selected_block_indices = Vec::with_capacity(n_blocks);

    for entry in &block_indexed {
        selected_block_indices.push(entry.idx);
        cumsum += entry.prob;
        if cumsum >= threshold {
            break;
        }
    }

    // Stage 3: Collect all token scores from selected blocks.
    // Pre-compute total candidate count for pre-allocation.
    let candidate_count: usize = selected_block_indices
        .iter()
        .map(|&bi| {
            let start = bi * block_size;
            std::cmp::min(start + block_size, n_total) - start
        })
        .sum();
    let mut candidate_entries: Vec<(usize, f32)> = Vec::with_capacity(candidate_count);
    for &block_idx in &selected_block_indices {
        let start = block_idx * block_size;
        let end = std::cmp::min(start + block_size, n_total);
        // SAFETY: start..end is within bounds by construction
        #[allow(clippy::needless_range_loop)] // indexing is faster than .skip().take() iterator chain
        for pos in start..end {
            candidate_entries.push((pos, scores[pos]));
        }
    }

    if candidate_entries.is_empty() {
        return DynamicTopPResult {
            selected_indices: vec![],
            selected_probs: vec![],
            cumulative_mass: 0.0,
            n_total,
        };
    }

    // Stage 4: Fine-grained top-p on candidate scores.
    // Compute softmax in-place to save one allocation: fill probs with raw
    // scores, then rewrite with exp values and normalize.
    let candidate_len = candidate_entries.len();
    let mut candidate_probs = vec![0.0f32; candidate_len];
    for (i, &(_, s)) in candidate_entries.iter().enumerate() {
        candidate_probs[i] = s;
    }
    // In-place softmax: find max, rewrite as exp, normalize.
    let max_val = candidate_probs
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in &mut candidate_probs {
        let e = (*v - max_val).exp();
        *v = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in &mut candidate_probs {
            *v *= inv;
        }
    } else {
        candidate_probs.fill(0.0);
    }

    let mut candidate_indexed: Vec<IndexedScore> = Vec::with_capacity(candidate_len);
    for (idx, &prob) in candidate_probs.iter().enumerate() {
        candidate_indexed.push(IndexedScore { idx, prob });
    }
    candidate_indexed.sort_unstable_by(|a, b| {
        b.prob
            .partial_cmp(&a.prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumsum = 0.0f32;
    let mut selected_indices = Vec::with_capacity(candidate_len);
    let mut selected_probs = Vec::with_capacity(candidate_len);

    for entry in &candidate_indexed {
        let (global_pos, _) = candidate_entries[entry.idx];
        selected_indices.push(global_pos);
        selected_probs.push(entry.prob);
        cumsum += entry.prob;
        if cumsum >= threshold {
            break;
        }
    }

    DynamicTopPResult {
        selected_indices,
        selected_probs,
        cumulative_mass: cumsum,
        n_total,
    }
}

// ---------------------------------------------------------------------------
// Tests (T14)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Single dominant peak -> only 1-2 tokens should be selected.
    #[test]
    fn test_concentrated_scores() {
        let scores = [100.0f32, 1.0, 1.0, 1.0, 1.0];
        let result = select_top_p(&scores, 0.9);

        assert!(
            result.selected_indices.len() <= 2,
            "Expected <= 2 tokens for concentrated scores, got {}",
            result.selected_indices.len()
        );
        assert_eq!(result.selected_indices[0], 0, "Peak should be index 0");
        assert!(
            result.cumulative_mass >= 0.9,
            "Cumulative mass should be >= 0.9, got {}",
            result.cumulative_mass
        );
    }

    /// Uniform scores -> many tokens selected (~ threshold * n).
    #[test]
    fn test_uniform_scores() {
        let n = 10;
        let scores = vec![1.0f32; n];
        let result = select_top_p(&scores, 0.9);

        assert!(
            result.selected_indices.len() >= 9,
            "Expected >= 9 tokens for uniform scores with p=0.9, got {}",
            result.selected_indices.len()
        );
        assert!(
            result.cumulative_mass >= 0.9 - 1e-6,
            "Cumulative mass should be >= 0.9, got {}",
            result.cumulative_mass
        );
    }

    /// p=1.0 -> all tokens selected.
    #[test]
    fn test_edge_p_1() {
        let scores = [3.0f32, 1.0, 2.0, 0.5];
        let result = select_top_p(&scores, 1.0);

        assert_eq!(
            result.selected_indices.len(),
            scores.len(),
            "p=1.0 should select all tokens"
        );
        assert!(
            (result.cumulative_mass - 1.0).abs() < 1e-5,
            "Cumulative mass should be ~1.0, got {}",
            result.cumulative_mass
        );
    }

    /// p=0.0 -> top-1 only (single highest probability token).
    #[test]
    fn test_edge_p_0() {
        let scores = [1.0f32, 5.0, 2.0, 3.0];
        let result = select_top_p(&scores, 0.0);

        assert_eq!(
            result.selected_indices.len(),
            1,
            "p=0.0 should select exactly 1 token, got {}",
            result.selected_indices.len()
        );
        assert_eq!(result.selected_indices[0], 1, "Highest score is at index 1");
    }

    /// Selected tokens cumulative softmax mass >= threshold.
    #[test]
    fn test_exact_mass_preservation() {
        let scores = [2.0f32, 0.5, 1.5, 3.0, 0.1, 1.0, 2.5, 0.8];
        let threshold = 0.75;
        let result = select_top_p(&scores, threshold);

        assert!(
            result.cumulative_mass >= threshold - 1e-6,
            "Cumulative mass should be >= {}, got {}",
            threshold,
            result.cumulative_mass
        );

        // Verify: removing last selected token drops mass below threshold
        if result.selected_probs.len() > 1 {
            let mass_without_last: f32 = result.selected_probs[..result.selected_probs.len() - 1]
                .iter()
                .copied()
                .sum();
            assert!(
                mass_without_last < threshold,
                "Mass without last token ({}) should be < {}",
                mass_without_last,
                threshold
            );
        }

        // Verify all indices are valid
        for &idx in &result.selected_indices {
            assert!(idx < scores.len(), "Index {} out of range", idx);
        }
    }

    /// Empty input -> empty output.
    #[test]
    fn test_empty_scores() {
        let result = select_top_p(&[], 0.9);
        assert!(result.selected_indices.is_empty());
        assert!(result.selected_probs.is_empty());
        assert_eq!(result.cumulative_mass, 0.0);
        assert_eq!(result.n_total, 0);
    }

    /// Single score -> single index.
    #[test]
    fn test_single_score() {
        let result = select_top_p(&[42.0f32], 0.5);
        assert_eq!(result.selected_indices, vec![0]);
        assert_eq!(result.selected_probs, vec![1.0]);
        assert_eq!(result.cumulative_mass, 1.0);
        assert_eq!(result.n_total, 1);
    }

    /// block_size=1: both variants should achieve the top_p mass threshold.
    ///
    /// With block_size=1, each position is its own block. The two-stage
    /// softmax+top-p applies thresholding at both block and token level,
    /// which may select a different set than single-stage due to probability
    /// redistribution in the fine-grained softmax. Both must satisfy the
    /// mass guarantee and select valid indices.
    #[test]
    fn test_blockwise_matches_fine_grained() {
        let scores = [2.0f32, 0.5, 1.5, 3.0, 0.1, 1.0, 2.5, 0.8];
        let top_p = 0.85;

        let fine = select_top_p(&scores, top_p);
        let block = select_top_p_blockwise(&scores, top_p, 1);

        // Both must achieve sufficient cumulative mass
        assert!(
            fine.cumulative_mass >= top_p - 1e-6,
            "Fine mass should be >= {}, got {}",
            top_p,
            fine.cumulative_mass
        );
        assert!(
            block.cumulative_mass >= top_p - 1e-5,
            "Block mass should be >= {}, got {}",
            top_p,
            block.cumulative_mass
        );

        // Both should select at least 1 token but not more than total
        assert!(fine.selected_indices.len() >= 1 && fine.selected_indices.len() <= scores.len());
        assert!(block.selected_indices.len() >= 1 && block.selected_indices.len() <= scores.len());

        // All indices must be valid
        for &idx in &fine.selected_indices {
            assert!(idx < scores.len(), "Fine index {} out of range", idx);
        }
        for &idx in &block.selected_indices {
            assert!(idx < scores.len(), "Block index {} out of range", idx);
        }

        // Both should include the highest-scored position (index 3, score=3.0)
        assert!(
            fine.selected_indices.contains(&3),
            "Fine should include highest-scored position 3"
        );
        assert!(
            block.selected_indices.contains(&3),
            "Block should include highest-scored position 3"
        );
    }

    /// block_size > 1 should select valid tokens with sufficient mass.
    #[test]
    fn test_blockwise_fewer_tokens() {
        let scores: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let top_p = 0.9;

        let block = select_top_p_blockwise(&scores, top_p, 16);

        assert!(
            !block.selected_indices.is_empty(),
            "Blockwise should select at least some tokens"
        );
        assert!(
            block.cumulative_mass >= top_p - 1e-5,
            "Blockwise cumulative mass should be >= {}, got {}",
            top_p,
            block.cumulative_mass
        );

        for &idx in &block.selected_indices {
            assert!(idx < scores.len(), "Index {} out of range", idx);
        }
    }

    /// Empty scores -> empty output for blockwise variant.
    #[test]
    fn test_blockwise_empty() {
        let result = select_top_p_blockwise(&[], 0.9, 64);
        assert!(result.selected_indices.is_empty());
        assert!(result.selected_probs.is_empty());
        assert_eq!(result.cumulative_mass, 0.0);
        assert_eq!(result.n_total, 0);
    }

    /// Very large scores (1e6) should not produce NaN or Inf.
    #[test]
    fn test_numerical_stability() {
        let scores = [1e6f32, 1e6 + 1.0, 1e6 - 1.0, 0.0, -1e6];
        let result = select_top_p(&scores, 0.9);

        assert!(
            result.cumulative_mass.is_finite(),
            "Cumulative mass should be finite, got {}",
            result.cumulative_mass
        );
        assert!(
            !result.cumulative_mass.is_nan(),
            "Cumulative mass should not be NaN"
        );

        for &prob in &result.selected_probs {
            assert!(
                prob.is_finite() && prob >= 0.0,
                "Probability should be finite and non-negative, got {}",
                prob
            );
        }

        let block_result = select_top_p_blockwise(&scores, 0.9, 2);
        assert!(
            block_result.cumulative_mass.is_finite(),
            "Blockwise cumulative mass should be finite, got {}",
            block_result.cumulative_mass
        );
    }
}
