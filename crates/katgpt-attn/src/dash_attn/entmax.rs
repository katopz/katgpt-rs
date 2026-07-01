//! α-entmax with α=1.5 special case (Plan 106, Research 68).
//!
//! Implements the closed-form quadratic entmax for α=1.5, which produces
//! sparse probability distributions with adaptive support size.
//!
//! # Algorithm
//!
//! 1. Sort scores descending
//! 2. Find threshold τ via cumulative sum: `τ = (Σ s_i - 1) / |support|`
//!    for the largest k where `s_k > τ`
//! 3. `p_i = max(0, 0.5 * s_i - τ)²` (quadratic for α=1.5)
//!
//! # References
//!
//! - Peters et al. (2019), "Sparse Sequence-to-Sequence Models"
//! - Correia et al. (2019), "Adaptively Sparse Transformers"

/// Compute entmax-1.5: returns (probabilities, threshold τ).
///
/// `p_i = max(0, 0.5 * s_i - τ)²`. Two-pass: sort → find threshold.
/// The resulting probabilities are non-negative, sum to 1.0, and are
/// exactly zero for scores below the threshold (sparse).
pub fn entmax_1p5(scores: &[f32]) -> (Vec<f32>, f32) {
    let n = scores.len();
    if n == 0 {
        return (vec![], 0.0);
    }
    let mut sorted_buf = Vec::with_capacity(n);
    let mut probs = vec![0.0f32; n];
    let tau = entmax_1p5_into(scores, &mut sorted_buf, &mut probs);
    (probs, tau)
}

/// Zero-alloc variant of [`entmax_1p5`].
///
/// Reuses `sorted_buf` (cleared and refilled) and `probs_buf` (overwritten).
/// Returns the threshold τ.
pub fn entmax_1p5_into(
    scores: &[f32],
    sorted_buf: &mut Vec<(usize, f32)>,
    probs_buf: &mut [f32],
) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let n = scores.len();
    debug_assert!(probs_buf.len() >= n);

    sorted_buf.clear();
    sorted_buf.extend(scores.iter().copied().enumerate());
    sorted_buf.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0f32;
    let mut tau = 0.0f32;
    let mut support_size = 0usize;

    for (k, &(_, score)) in sorted_buf.iter().enumerate() {
        cumsum += score;
        let t = (cumsum - 1.0) / ((k + 1) as f32);
        if score > t {
            tau = t;
            support_size = k + 1;
        }
    }

    probs_buf[..n].fill(0.0);
    for &(orig_idx, score) in sorted_buf.iter().take(support_size) {
        let v = 0.5 * score - 0.5 * tau;
        probs_buf[orig_idx] = v * v;
    }

    // Normalize to sum to 1
    let sum: f32 = probs_buf[..n].iter().sum();
    if sum > 0.0 {
        for p in probs_buf[..n].iter_mut() {
            *p /= sum;
        }
    }

    tau
}

/// Extract active indices from entmax weights (positions where weight > ε).
///
/// Returns indices of non-zero probability entries, representing the
/// adaptively selected support set.
pub fn entmax_support(probs: &[f32]) -> Vec<usize> {
    let mut result = Vec::with_capacity(probs.len());
    entmax_support_into(probs, &mut result);
    result
}

/// Zero-alloc variant of [`entmax_support`].
///
/// Appends active indices to `buf` (cleared first).
pub fn entmax_support_into(probs: &[f32], buf: &mut Vec<usize>) {
    buf.clear();
    for (i, &p) in probs.iter().enumerate() {
        if p > 1e-8 {
            buf.push(i);
        }
    }
}

/// Average entmax probabilities across query heads in the same GQA group.
///
/// Maps `n_query_heads` routing distributions down to `n_kv_heads` by
/// averaging probabilities within each KV group.
///
/// # Arguments
///
/// * `head_probs` - `[n_query_heads][n_chunks]` per-head routing probabilities
/// * `n_query_heads` - Number of query heads
/// * `n_kv_heads` - Number of KV heads (groups)
/// * `n_chunks` - Number of chunks being routed over
///
/// # Returns
///
/// `[n_kv_heads][n_chunks]` aggregated routing probabilities
pub fn entmax_gqa_aggregate<T: AsRef<[f32]>>(
    head_probs: &[T],
    n_query_heads: usize,
    n_kv_heads: usize,
    n_chunks: usize,
) -> Vec<Vec<f32>> {
    let mut result = vec![vec![0.0f32; n_chunks]; n_kv_heads];
    let mut counts = vec![0usize; n_kv_heads];

    for (h, head_prob) in head_probs.iter().enumerate() {
        let kv_group = h * n_kv_heads / n_query_heads;
        counts[kv_group] += 1;
        let group = &mut result[kv_group];
        let hp = head_prob.as_ref();
        for (&prob, dest) in hp.iter().zip(group.iter_mut()) {
            *dest += prob;
        }
    }

    for g in 0..n_kv_heads {
        if counts[g] > 0 {
            let inv = 1.0 / counts[g] as f32;
            for val in &mut result[g] {
                *val *= inv;
            }
        }
    }

    result
}

/// Zero-alloc variant of [`entmax_gqa_aggregate`].
///
/// Writes aggregated probabilities into `result` and uses `counts` as scratch.
/// Both must be pre-sized to at least `n_kv_heads` length; each `result[i]`
/// must have at least `n_chunks` elements.
pub fn entmax_gqa_aggregate_into<T: AsRef<[f32]>>(
    head_probs: &[T],
    n_query_heads: usize,
    n_kv_heads: usize,
    n_chunks: usize,
    result: &mut [Vec<f32>],
    counts: &mut [usize],
) {
    debug_assert!(result.len() >= n_kv_heads);
    debug_assert!(counts.len() >= n_kv_heads);

    counts[..n_kv_heads].fill(0);
    for row in result.iter_mut().take(n_kv_heads) {
        row[..n_chunks].fill(0.0);
    }

    for (h, head_prob) in head_probs.iter().enumerate() {
        let kv_group = h * n_kv_heads / n_query_heads;
        counts[kv_group] += 1;
        let group = &mut result[kv_group];
        let hp = head_prob.as_ref();
        for (&prob, dest) in hp.iter().zip(group.iter_mut()) {
            *dest += prob;
        }
    }

    for g in 0..n_kv_heads {
        if counts[g] > 0 {
            let inv = 1.0 / counts[g] as f32;
            for val in &mut result[g][..n_chunks] {
                *val *= inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entmax_empty_input() {
        let (probs, tau) = entmax_1p5(&[]);
        assert!(probs.is_empty());
        assert_eq!(tau, 0.0);
    }

    #[test]
    fn test_entmax_single_input() {
        let (probs, _tau) = entmax_1p5(&[3.0]);
        assert_eq!(probs.len(), 1);
        assert!(
            (probs[0] - 1.0).abs() < 1e-6,
            "single input should have prob 1.0"
        );
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_entmax_non_negative() {
        let scores = [1.0, 0.5, 0.2, -1.0, 2.0, 0.0];
        let (probs, _) = entmax_1p5(&scores);
        for (i, &p) in probs.iter().enumerate() {
            assert!(p >= 0.0, "prob at index {i} is negative: {p}");
        }
    }

    #[test]
    fn test_entmax_sums_to_one() {
        let scores = [2.0, 1.0, 0.5, 0.1, -0.5, 1.5];
        let (probs, _) = entmax_1p5(&scores);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "probs should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_entmax_sparse_zeros() {
        // With widely spread scores, low scores should get exactly zero.
        // For [10, 9, 0.01, -5, -10]: tau=9.0, only score > tau is 10.0.
        // So only index 0 is in support (9.0 is NOT > 9.0).
        let scores = [10.0, 9.0, 0.01, -5.0, -10.0];
        let (probs, _) = entmax_1p5(&scores);

        // The negative scores should be exactly zero
        assert!(
            probs[3] < 1e-8,
            "negative score should have ~0 prob, got {}",
            probs[3]
        );
        assert!(
            probs[4] < 1e-8,
            "very negative score should have ~0 prob, got {}",
            probs[4]
        );

        // Highest score should be active and dominate
        assert!(probs[0] > 0.0, "highest score should be active");
        assert!(
            probs[0] > 0.99,
            "highest score should dominate, got {}",
            probs[0]
        );
        // Second highest (9.0) is exactly at threshold, so it gets ~0
        assert!(
            probs[1] < 1e-6,
            "second highest at threshold boundary should be ~0, got {}",
            probs[1]
        );
    }

    #[test]
    fn test_entmax_uniform_input() {
        // All equal scores → all should have equal probability
        let scores = [1.0, 1.0, 1.0, 1.0];
        let (probs, _) = entmax_1p5(&scores);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {sum}");

        // All should be roughly equal (0.25 each)
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - 0.25).abs() < 1e-5,
                "uniform input should give 0.25 each, got {p} at {i}"
            );
        }
    }

    #[test]
    fn test_entmax_threshold_positive() {
        // Threshold should be less than the maximum score
        let scores = [5.0, 3.0, 1.0];
        let (_, tau) = entmax_1p5(&scores);
        assert!(tau < 5.0, "tau ({tau}) should be < max score (5.0)");
    }

    #[test]
    fn test_entmax_support_extraction() {
        // Use closer scores so both top entries are above threshold.
        // [3.0, 2.9, 2.8, -5.0]: k=0 t=2.0 (<3.0 ✓), k=1 t=2.45 (<2.9 ✓) → both active.
        let scores = [3.0, 2.9, 2.8, -5.0];
        let (probs, _) = entmax_1p5(&scores);
        let support = entmax_support(&probs);

        // Should include indices 0 and 1 (both above threshold)
        assert!(
            support.contains(&0),
            "support should contain highest score index"
        );
        assert!(
            support.contains(&1),
            "support should contain second highest score index"
        );
        // Should NOT contain the very negative score
        assert!(
            !support.contains(&3),
            "support should not contain very negative score index"
        );
    }

    #[test]
    fn test_entmax_support_all_active() {
        // Uniform scores → all active
        let scores = [2.0, 2.0, 2.0];
        let (probs, _) = entmax_1p5(&scores);
        let support = entmax_support(&probs);
        assert_eq!(support.len(), 3, "uniform should have all 3 active");
    }

    #[test]
    fn test_entmax_gqa_aggregate_basic() {
        // 4 query heads, 2 KV heads, 3 chunks.
        // GQA mapping: kv_group = h * n_kv_heads / n_query_heads
        //   h=0 → group 0, h=1 → group 0, h=2 → group 1, h=3 → group 1
        let n_query_heads = 4;
        let n_kv_heads = 2;
        let n_chunks = 3;

        let head_probs = vec![
            vec![0.5, 0.3, 0.2], // Q head 0 → KV group 0
            vec![0.6, 0.2, 0.2], // Q head 1 → KV group 0
            vec![0.4, 0.4, 0.2], // Q head 2 → KV group 1
            vec![0.3, 0.5, 0.2], // Q head 3 → KV group 1
        ];

        let result = entmax_gqa_aggregate(&head_probs, n_query_heads, n_kv_heads, n_chunks);

        assert_eq!(result.len(), n_kv_heads);
        assert_eq!(result[0].len(), n_chunks);

        // KV group 0: average of heads 0 and 1 → [(0.5+0.6)/2, (0.3+0.2)/2, (0.2+0.2)/2]
        assert!(
            (result[0][0] - 0.55).abs() < 1e-6,
            "KV group 0 chunk 0: expected 0.55, got {}",
            result[0][0]
        );
        assert!(
            (result[0][1] - 0.25).abs() < 1e-6,
            "KV group 0 chunk 1: expected 0.25, got {}",
            result[0][1]
        );
        assert!(
            (result[0][2] - 0.20).abs() < 1e-6,
            "KV group 0 chunk 2: expected 0.20, got {}",
            result[0][2]
        );

        // KV group 1: average of heads 2 and 3 → [(0.4+0.3)/2, (0.4+0.5)/2, (0.2+0.2)/2]
        assert!(
            (result[1][0] - 0.35).abs() < 1e-6,
            "KV group 1 chunk 0: expected 0.35, got {}",
            result[1][0]
        );
        assert!(
            (result[1][1] - 0.45).abs() < 1e-6,
            "KV group 1 chunk 1: expected 0.45, got {}",
            result[1][1]
        );
        assert!(
            (result[1][2] - 0.20).abs() < 1e-6,
            "KV group 1 chunk 2: expected 0.20, got {}",
            result[1][2]
        );
    }

    #[test]
    fn test_entmax_gqa_aggregate_single_kv_head() {
        // All query heads map to 1 KV head
        let n_query_heads = 3;
        let n_kv_heads = 1;
        let n_chunks = 2;

        let head_probs = vec![vec![0.6, 0.4], vec![0.8, 0.2], vec![0.4, 0.6]];

        let result = entmax_gqa_aggregate(&head_probs, n_query_heads, n_kv_heads, n_chunks);

        assert_eq!(result.len(), 1);
        // Average: (0.6+0.8+0.4)/3 = 0.6, (0.4+0.2+0.6)/3 = 0.4
        assert!((result[0][0] - 0.6).abs() < 1e-6);
        assert!((result[0][1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_entmax_known_values() {
        // Pre-computed reference: scores [4.0, 2.0, 1.0]
        // Sorted: [4.0, 2.0, 1.0]
        // k=0: cumsum=4.0, t=(4-1)/1=3.0, check t<sorted[0]=4.0 ✓, t>=sorted[1]=2.0 → skip
        // k=1: cumsum=6.0, t=(6-1)/2=2.5, check t<sorted[1]=2.0? No → continue
        // k=2: cumsum=7.0, t=(7-1)/3=2.0, check t<sorted[2]=1.0? No → done
        // So tau=3.0, support_size=1
        // p_0 = (0.5*4.0 - 0.5*3.0)² = (2.0-1.5)² = 0.25
        // After normalization: p_0 = 1.0 (only one active)
        let scores = [4.0, 2.0, 1.0];
        let (probs, tau) = entmax_1p5(&scores);

        assert!((tau - 3.0).abs() < 1e-5, "expected tau=3.0, got {tau}");
        assert!(
            (probs[0] - 1.0).abs() < 1e-5,
            "expected prob[0]=1.0, got {}",
            probs[0]
        );
        assert!(probs[1] < 1e-8, "expected prob[1]≈0, got {}", probs[1]);
        assert!(probs[2] < 1e-8, "expected prob[2]≈0, got {}", probs[2]);
    }
}
