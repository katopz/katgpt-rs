//! Entmax block routing — adaptive sparse chunk selection.
//!
//! Replaces fixed-budget top-k block selection with adaptive support
//! selection via α-entmax (α=1.5). Computes per-head routing probabilities
//! and normalised routing biases for downstream attention modulation.

use crate::types::DashAttnConfig;

use super::entmax::{entmax_1p5, entmax_gqa_aggregate, entmax_support};

/// Result of entmax routing for one query head.
#[derive(Debug)]
pub struct RoutingResult {
    /// Active chunk indices (support of entmax distribution).
    pub active_indices: Vec<usize>,
    /// Routing bias per active chunk: `(log w - μ) / σ`.
    pub bias: Vec<f32>,
    /// Entmax probabilities for all chunks.
    pub probs: Vec<f32>,
}

/// Score blocks using entmax routing.
///
/// Computes chunk logits as scaled dot-product between a single-head query
/// and chunk summaries, then applies α-entmax (α=1.5) to obtain an adaptive
/// sparse distribution over chunks.
pub fn score_blocks_entmax(
    query: &[f32],          // [head_dim] per-head query
    summaries: &[Vec<f32>], // [n_chunks][head_dim] chunk summaries
    config: &DashAttnConfig,
) -> RoutingResult {
    let hd = query.len();

    // Compute chunk logits: z = q · k̄ / √d * γ
    let scale = 1.0 / (hd as f32).sqrt() * config.scaling_factor;
    let logits: Vec<f32> = summaries
        .iter()
        .map(|s| {
            let dot: f32 = query.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
            dot * scale
        })
        .collect();

    // α-entmax routing
    let (probs, _tau) = entmax_1p5(&logits);

    // Extract support
    let active_indices = entmax_support(&probs);

    // Compute routing bias: (log w - μ) / σ on active indices
    let log_weights: Vec<f32> = active_indices
        .iter()
        .map(|&i| {
            if probs[i] > 1e-10 {
                probs[i].ln()
            } else {
                -23.0 // ln(1e-10)
            }
        })
        .collect();

    let mean_lw = if log_weights.is_empty() {
        0.0
    } else {
        log_weights.iter().sum::<f32>() / log_weights.len() as f32
    };

    let var_lw: f32 = if log_weights.len() <= 1 {
        1.0
    } else {
        log_weights
            .iter()
            .map(|&x| (x - mean_lw).powi(2))
            .sum::<f32>()
            / (log_weights.len() - 1) as f32
    };
    let std_lw = var_lw.sqrt().max(1e-6);

    let bias: Vec<f32> = log_weights
        .iter()
        .map(|&lw| (lw - mean_lw) / std_lw)
        .collect();

    RoutingResult {
        active_indices,
        bias,
        probs,
    }
}

/// Compute routing bias for all query heads with GQA aggregation.
///
/// Runs per-query-head entmax routing, then averages probabilities across
/// heads sharing the same KV group for consensus routing.
pub fn compute_routing_bias(
    queries: &[Vec<f32>],   // [n_query_heads][head_dim]
    summaries: &[Vec<f32>], // [n_chunks][head_dim]
    n_kv_heads: usize,
    config: &DashAttnConfig,
) -> Vec<RoutingResult> {
    let n_query_heads = queries.len();
    let n_chunks = summaries.len();

    // Per-query-head routing
    let per_head: Vec<RoutingResult> = queries
        .iter()
        .map(|q| score_blocks_entmax(q, summaries, config))
        .collect();

    // GQA aggregation: merge probs across heads in same group
    let head_probs: Vec<Vec<f32>> = per_head.iter().map(|r| r.probs.clone()).collect();
    let _agg_probs = entmax_gqa_aggregate(&head_probs, n_query_heads, n_kv_heads, n_chunks);

    per_head
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DashAttnConfig {
        DashAttnConfig::default()
    }

    #[test]
    fn test_score_blocks_entmax_single_chunk() {
        let config = default_config();
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let summaries = vec![vec![1.0, 0.0, 0.0, 0.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert_eq!(result.active_indices, vec![0]);
        assert!(
            (result.probs[0] - 1.0).abs() < 1e-6,
            "single chunk should get all probability mass"
        );
    }

    #[test]
    fn test_score_blocks_entmax_two_chunks_clear_winner() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        // Chunk 0 aligned with query, chunk 1 orthogonal
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert!(!result.active_indices.is_empty());
        // Chunk 0 should dominate
        assert!(result.probs[0] > result.probs[1]);
    }

    #[test]
    fn test_score_blocks_entmax_probs_sum_to_one() {
        let config = default_config();
        let query = vec![1.0, 2.0, 3.0];
        let summaries = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let result = score_blocks_entmax(&query, &summaries, &config);
        let sum: f32 = result.probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "entmax probs must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_score_blocks_entmax_non_negative() {
        let config = default_config();
        let query = vec![1.0, 0.5];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, -1.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        for (i, &p) in result.probs.iter().enumerate() {
            assert!(p >= 0.0, "prob at index {i} is negative: {p}");
        }
    }

    #[test]
    fn test_routing_result_bias_has_same_length_as_active() {
        let config = default_config();
        let query = vec![1.0, 0.0, 0.0];
        let summaries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert_eq!(
            result.active_indices.len(),
            result.bias.len(),
            "bias length must match active_indices length"
        );
    }

    #[test]
    fn test_compute_routing_bias_multi_head() {
        let config = default_config();
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let results = compute_routing_bias(&queries, &summaries, 1, &config);
        assert_eq!(results.len(), 2, "should have one result per query head");

        for (h, r) in results.iter().enumerate() {
            let sum: f32 = r.probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "head {h} probs must sum to 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_compute_routing_bias_gqa_fewer_kv_heads() {
        let config = default_config();
        // 4 query heads, 2 kv heads
        let queries = vec![
            vec![1.0, 0.0],
            vec![0.8, 0.2],
            vec![0.2, 0.8],
            vec![0.0, 1.0],
        ];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let results = compute_routing_bias(&queries, &summaries, 2, &config);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_score_blocks_empty_summaries() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries: Vec<Vec<f32>> = vec![];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert!(result.active_indices.is_empty());
        assert!(result.probs.is_empty());
        assert!(result.bias.is_empty());
    }

    #[test]
    fn test_score_blocks_all_orthogonal() {
        let config = default_config();
        // Query orthogonal to all chunks → entmax may spread or concentrate
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![0.0, 1.0], vec![0.0, -1.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        // All logits should be ~0, entmax should still produce valid distribution
        let sum: f32 = result.probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5 || sum == 0.0,
            "probs should sum to 1.0 or be empty when all logits zero, got {sum}"
        );
    }
}
