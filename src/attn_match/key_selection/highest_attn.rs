//! Highest Attention Keys selector.
//!
//! Ranks keys by aggregated attention score across reference queries, then
//! selects the top-t. Aggregation methods:
//! - Mean: `mean_i a_ij`
//! - RMS (default): `sqrt(mean_i a_ij^2)` — most robust per paper Appendix F.1
//! - Max: `max_i a_ij`
//!
//! Returns the selected indices; the caller is responsible for fitting β
//! (via [`crate::attn_match::fit_beta_nnls`]) on the resulting subset.
//!
//! Per the paper (Section 3.3, "Highest attention keys"), this is the fastest
//! selector and a strong baseline.

#![allow(clippy::too_many_arguments)]

use crate::attn_match::{
    key_selection::KeySelection,
    score_matrix::{compute_score_matrix, compute_softmax_attention},
    types::ScoreMethod,
};

/// Select top-t keys by aggregated attention score.
///
/// # Arguments
/// * `keys` - Original `(T, d)` key matrix, flat row-major.
/// * `queries` - Reference queries `(n, d)`, flat row-major.
/// * `t` - Number of keys to select.
/// * `score_method` - Aggregation method (Mean/Rms/Max).
/// * `t_len` - Original sequence length `T`.
/// * `d` - Head dimension.
/// * `n` - Number of reference queries.
/// * `scratch_scores` - Caller-allocated scratch `(n, T)` for the score matrix
///   (pass `&mut Vec::new()` to have it sized on first call; reuse across calls).
/// * `scratch_attn` - Caller-allocated scratch `(n, T)` for the softmax matrix.
pub fn select_highest_attn_keys(
    keys: &[f32],
    queries: &[f32],
    t: usize,
    score_method: ScoreMethod,
    t_len: usize,
    d: usize,
    n: usize,
    scratch_scores: &mut Vec<f32>,
    scratch_attn: &mut Vec<f32>,
) -> KeySelection {
    assert_eq!(keys.len(), t_len * d);
    assert_eq!(queries.len(), n * d);
    assert!(t <= t_len, "t ({}) must be ≤ T ({})", t, t_len);

    // Compute score matrix S = Q K^T / √d into scratch.
    scratch_scores.clear();
    scratch_scores.resize(n * t_len, 0.0);
    compute_score_matrix(queries, keys, n, t_len, d, scratch_scores);

    // Compute softmax attention matrix A = softmax(S) into scratch_attn.
    // We also need the per-query mass to compute coverage later, but the selector
    // only uses the normalized attention weights.
    scratch_attn.clear();
    scratch_attn.resize(n * t_len, 0.0);
    let mut mass = vec![0.0f32; n];
    compute_softmax_attention(scratch_scores, n, t_len, scratch_attn, &mut mass);

    // Aggregate per-key attention scores.
    // Iterate queries in the outer loop so reads from `scratch_attn` are
    // sequential (row-major) and writes to `per_key_score` are sequential.
    // The previous j-outer/i-inner form did strided `scratch_attn[i*t_len + j]`
    // reads — cache-hostile for large `t_len`. FLOPs are identical; only the
    // memory access order changes.
    let mut per_key_score = vec![0.0f32; t_len];
    let inv_n = 1.0f32 / (n as f32);
    match score_method {
        ScoreMethod::Mean => {
            for i in 0..n {
                let row = &scratch_attn[i * t_len..(i + 1) * t_len];
                for j in 0..t_len {
                    per_key_score[j] += row[j];
                }
            }
            for v in &mut per_key_score {
                *v *= inv_n;
            }
        }
        ScoreMethod::Rms => {
            for i in 0..n {
                let row = &scratch_attn[i * t_len..(i + 1) * t_len];
                for j in 0..t_len {
                    per_key_score[j] += row[j] * row[j];
                }
            }
            for v in &mut per_key_score {
                *v = (*v * inv_n).sqrt();
            }
        }
        ScoreMethod::Max => {
            per_key_score.fill(f32::NEG_INFINITY);
            for i in 0..n {
                let row = &scratch_attn[i * t_len..(i + 1) * t_len];
                for j in 0..t_len {
                    per_key_score[j] = per_key_score[j].max(row[j]);
                }
            }
        }
    }

    // Partial selection: O(T) partition to find the top-t boundary, then
    // O(t log t) sort of just the survivors. For t << T this beats the prior
    // full O(T log T) argsort. total_cmp replaces partial_cmp().unwrap_or()
    // (no NaN branch — scores are finite dot products).
    let mut indexed: Vec<(usize, f32)> = per_key_score.iter().copied().enumerate().collect();
    if t > 0 && t < indexed.len() {
        indexed.select_nth_unstable_by(t - 1, |a, b| b.1.total_cmp(&a.1));
    }
    indexed[..t].sort_by(|a, b| b.1.total_cmp(&a.1));

    let indices: Vec<usize> = indexed[..t].iter().map(|(i, _)| *i).collect();
    // No NNLS weights here — caller will fit β separately.
    let weights = vec![1.0f32; t];

    KeySelection { indices, weights }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_returns_correct_count() {
        let t_len = 10;
        let d = 4;
        let n = 3;
        let keys: Vec<f32> = (0..t_len * d).map(|i| (i as f32) * 0.1).collect();
        let queries: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.2).collect();
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        let sel = select_highest_attn_keys(
            &keys,
            &queries,
            3,
            ScoreMethod::Rms,
            t_len,
            d,
            n,
            &mut s1,
            &mut s2,
        );
        assert_eq!(sel.indices.len(), 3);
        assert_eq!(sel.weights.len(), 3);
        assert_eq!(sel.weights, vec![1.0, 1.0, 1.0]);
        // Indices should be unique.
        let mut sorted = sel.indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "indices must be unique");
    }

    #[test]
    fn test_select_picks_high_attention_keys() {
        // Construct keys where only one is aligned with the query.
        let t_len = 5;
        let d = 4;
        let n = 1;
        let mut keys = vec![0.0f32; t_len * d];
        // Make key 2 strongly aligned with query 0.
        keys[2 * d] = 10.0;
        keys[2 * d + 1] = 10.0;
        let mut queries = vec![0.0f32; n * d];
        queries[0] = 10.0;
        queries[1] = 10.0;
        let mut s1 = Vec::new();
        let mut s2 = Vec::new();
        let sel = select_highest_attn_keys(
            &keys,
            &queries,
            1,
            ScoreMethod::Max,
            t_len,
            d,
            n,
            &mut s1,
            &mut s2,
        );
        // Top-1 should be index 2.
        assert_eq!(sel.indices.len(), 1);
        assert_eq!(sel.indices[0], 2);
    }

    #[test]
    fn test_select_mean_vs_rms_vs_max_consistent() {
        let t_len = 8;
        let d = 4;
        let n = 4;
        let keys: Vec<f32> = (0..t_len * d).map(|i| (i as f32).sin() * 0.5).collect();
        let queries: Vec<f32> = (0..n * d).map(|i| (i as f32).cos() * 0.3).collect();
        for &method in &[
            ScoreMethod::Mean,
            ScoreMethod::Rms,
            ScoreMethod::Max,
        ] {
            let mut s1 = Vec::new();
            let mut s2 = Vec::new();
            let sel = select_highest_attn_keys(
                &keys,
                &queries,
                4,
                method,
                t_len,
                d,
                n,
                &mut s1,
                &mut s2,
            );
            assert_eq!(sel.indices.len(), 4);
            // No panic, returns valid result.
        }
    }
}
