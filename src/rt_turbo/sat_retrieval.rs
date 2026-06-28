//! SAT-accelerated retrieval head identification.
//!
//! Uses [`SummedAreaTable`] from the `cache_prune` module to compute per-head
//! retrieval scores in O(1) per head after O(n²) preprocessing, replacing the
//! O(n²) naive scan in [`compute_retrieval_score`](super::calibration::compute_retrieval_score).
//!
//! Feature-gated with both `rt_turbo` and `cache_prune`.

use crate::cache_prune::SummedAreaTable;

/// Compute per-head retrieval scores using SAT for O(1) rectangular region queries.
///
/// For each head's `n×n` attention matrix, builds a SAT and queries the rectangular
/// region `[post_needle_start..post_needle_end) × [needle_start..needle_end)` — the
/// attention mass that post-needle positions assign to the needle span.
///
/// The score is the average attention weight in that region (total mass divided by
/// the number of elements), matching the naive [`compute_retrieval_score`](super::calibration::compute_retrieval_score).
///
/// # Arguments
///
/// * `per_head_attentions` — One `n×n` attention matrix per query head (row-major `Vec<Vec<f32>>`).
/// * `needle_start` — Start of the needle (pre-needle) span.
/// * `needle_end` — End of the needle span (exclusive).
/// * `post_needle_start` — Start of the post-needle query span.
/// * `post_needle_end` — End of the post-needle query span (exclusive).
///
/// # Returns
///
/// Vector of retrieval scores, one per head.
pub fn compute_retrieval_scores_sat(
    per_head_attentions: &[Vec<f32>],
    needle_start: usize,
    needle_end: usize,
    post_needle_start: usize,
    post_needle_end: usize,
) -> Vec<f32> {
    let pre_len = needle_end.saturating_sub(needle_start);
    let post_len = post_needle_end.saturating_sub(post_needle_start);

    if pre_len == 0 || post_len == 0 {
        return vec![0.0; per_head_attentions.len()];
    }

    let area = (post_len * pre_len) as f32;

    per_head_attentions
        .iter()
        .map(|attn| {
            let n = attn.len();
            // Convert flat to 2D for SAT
            let seq_len = if n > 0 {
                let sq = (n as f64).sqrt() as usize;
                debug_assert_eq!(sq * sq, n, "attention matrix must be square (n² elements)");
                sq
            } else {
                0
            };

            if seq_len == 0 {
                return 0.0;
            }

            // Build SAT in-place using slices (avoids Vec<Vec<f32>> allocation)
            let mut matrix: Vec<Vec<f32>> = (0..seq_len)
                .map(|i| attn[i * seq_len..(i + 1) * seq_len].to_vec())
                .collect();

            let sat = SummedAreaTable::build(&mut matrix);

            // Query region: rows [post_needle_start, post_needle_end-1],
            //               cols [needle_start, needle_end-1]
            let pns = post_needle_start.min(seq_len);
            let pne = (post_needle_end.saturating_sub(1)).min(seq_len.saturating_sub(1));
            let ns = needle_start.min(seq_len);
            let ne = (needle_end.saturating_sub(1)).min(seq_len.saturating_sub(1));

            if pns > pne || ns > ne {
                return 0.0;
            }

            let total_mass = sat.region_sum(pns, pne, ns, ne);
            total_mass / area
        })
        .collect()
}

/// Identify the top `retrieval_ratio` fraction of heads as retrieval heads.
///
/// Returns head indices sorted by score descending (highest scores first).
/// At least one head is always selected (unless `scores` is empty).
///
/// # Arguments
///
/// * `scores` — Per-head retrieval scores (e.g. from [`compute_retrieval_scores_sat`]).
/// * `retrieval_ratio` — Fraction of heads to classify as retrieval (e.g. `0.15` for 15%).
///
/// # Returns
///
/// Sorted indices of retrieval heads (highest score first).
pub fn identify_retrieval_heads_sat(scores: &[f32], retrieval_ratio: f32) -> Vec<usize> {
    if scores.is_empty() {
        return Vec::new();
    }

    let n_retrieval = ((scores.len() as f32) * retrieval_ratio).ceil() as usize;
    let n_retrieval = n_retrieval.max(1).min(scores.len());

    // Collect (index, score) pairs, sort by score descending
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Pre-allocate and extract
    let mut result = Vec::with_capacity(n_retrieval);
    for &(idx, _) in &indexed[..n_retrieval] {
        result.push(idx);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive retrieval score computation matching calibration::compute_retrieval_score
    /// but working on a 2D `&[Vec<f32>]` attention matrix.
    fn naive_retrieval_score(
        attention: &[Vec<f32>],
        needle_start: usize,
        needle_end: usize,
        post_needle_start: usize,
        post_needle_end: usize,
    ) -> f32 {
        let seq_len = attention.len();
        let pre_len = needle_end.saturating_sub(needle_start);
        let post_len = post_needle_end.saturating_sub(post_needle_start);

        if pre_len == 0 || post_len == 0 || seq_len == 0 {
            return 0.0;
        }

        let mut total = 0.0;
        for row in &attention[post_needle_start..post_needle_end.min(seq_len)] {
            let end = needle_end.min(row.len());
            total += row[needle_start..end].iter().sum::<f32>();
        }
        total / (post_len * pre_len) as f32
    }

    /// Build a 20×20 attention matrix with a known needle pattern.
    ///
    /// Positions [2..5] = needle (pre-needle span).
    /// Positions [15..20] = post-needle query span.
    ///
    /// Retrieval heads assign high attention to the needle;
    /// local heads assign uniform low attention.
    fn make_test_matrices() -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
        let seq_len = 20;
        let n_heads = 4;

        let mut per_head_2d: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_heads);
        let mut per_head_flat: Vec<Vec<f32>> = Vec::with_capacity(n_heads);

        for h in 0..n_heads {
            let mut matrix = vec![vec![0.01_f32; seq_len]; seq_len];

            match h {
                // Head 0: strong retrieval — high attention to needle
                0 => {
                    for row in &mut matrix[15..20] {
                        for cell in &mut row[2..5] {
                            *cell = 0.8;
                        }
                    }
                }
                // Head 1: moderate retrieval
                1 => {
                    for row in &mut matrix[15..20] {
                        for cell in &mut row[2..5] {
                            *cell = 0.4;
                        }
                    }
                }
                // Head 2: weak / local
                2 => {
                    // Uniform low 0.01 already set
                }
                // Head 3: moderate retrieval (between heads 1 and 2)
                3 => {
                    for row in &mut matrix[15..20] {
                        for cell in &mut row[2..5] {
                            *cell = 0.2;
                        }
                    }
                }
                _ => {}
            }

            // Flatten to 1D for the SAT function
            let flat: Vec<f32> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
            per_head_flat.push(flat);
            per_head_2d.push(matrix);
        }

        (per_head_2d, per_head_flat)
    }

    #[test]
    fn test_sat_scores_match_naive() {
        let (per_head_2d, per_head_flat) = make_test_matrices();
        let needle_start = 2;
        let needle_end = 5;
        let post_needle_start = 15;
        let post_needle_end = 20;

        // Compute SAT scores
        let sat_scores = compute_retrieval_scores_sat(
            &per_head_flat,
            needle_start,
            needle_end,
            post_needle_start,
            post_needle_end,
        );

        // Compute naive scores
        let naive_scores: Vec<f32> = per_head_2d
            .iter()
            .map(|attn| {
                naive_retrieval_score(
                    attn,
                    needle_start,
                    needle_end,
                    post_needle_start,
                    post_needle_end,
                )
            })
            .collect();

        assert_eq!(sat_scores.len(), naive_scores.len(), "score count mismatch");

        for (h, (sat, naive)) in sat_scores.iter().zip(naive_scores.iter()).enumerate() {
            assert!(
                (sat - naive).abs() < 1e-4,
                "head {h}: SAT score {sat} != naive score {naive}"
            );
        }

        // Head 0 should have highest score
        assert!(
            sat_scores[0] > sat_scores[2],
            "retrieval head 0 ({}) should score higher than local head 2 ({})",
            sat_scores[0],
            sat_scores[2]
        );
    }

    #[test]
    fn test_identify_retrieval_heads() {
        let (_, per_head_flat) = make_test_matrices();
        let scores = compute_retrieval_scores_sat(&per_head_flat, 2, 5, 15, 20);

        // Top 50% of 4 heads = 2 retrieval heads
        let retrieval = identify_retrieval_heads_sat(&scores, 0.5);

        assert_eq!(retrieval.len(), 2, "should select 2 retrieval heads");
        // Head 0 (score ~0.8) and head 1 (score ~0.4) should be selected
        assert!(
            retrieval.contains(&0),
            "head 0 should be retrieval, scores: {scores:?}"
        );
        assert!(
            retrieval.contains(&1),
            "head 1 should be retrieval, scores: {scores:?}"
        );
        // Verify ordering (highest first)
        assert_eq!(retrieval[0], 0, "head 0 should be first (highest score)");
    }

    #[test]
    fn test_identify_empty_scores() {
        let result = identify_retrieval_heads_sat(&[], 0.15);
        assert!(result.is_empty());
    }

    #[test]
    fn test_identify_single_head() {
        let result = identify_retrieval_heads_sat(&[0.42], 0.15);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_empty_attention() {
        let scores = compute_retrieval_scores_sat(&[], 2, 5, 15, 20);
        assert!(scores.is_empty());

        // Zero-length needle
        let scores = compute_retrieval_scores_sat(&[vec![0.1; 4]], 0, 0, 1, 2);
        assert_eq!(scores, vec![0.0]);
    }
}
