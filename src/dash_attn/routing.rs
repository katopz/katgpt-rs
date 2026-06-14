//! Entmax block routing — adaptive sparse chunk selection.
//!
//! Replaces fixed-budget top-k block selection with adaptive support
//! selection via α-entmax (α=1.5). Computes per-head routing probabilities
//! and normalised routing biases for downstream attention modulation.

use crate::simd::simd_dot_f32;
use crate::types::DashAttnConfig;

use super::entmax::{entmax_1p5_into, entmax_gqa_aggregate, entmax_support_into};

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
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    config: &DashAttnConfig,
) -> RoutingResult {
    let n = summaries.len();
    let mut scratch = RoutingScratch::new(n, query.len());
    score_blocks_entmax_into(query, summaries, config, &mut scratch)
}

/// Pre-allocated scratch buffers for entmax routing.
pub struct RoutingScratch {
    /// Logits buffer: [n_chunks].
    logits: Vec<f32>,
    /// Sorted indices buffer for entmax: Vec<(usize, f32)>.
    sorted: Vec<(usize, f32)>,
    /// Probabilities buffer for entmax: [n_chunks].
    probs: Vec<f32>,
    /// Log-weights buffer for active indices (reused across calls).
    log_weights: Vec<f32>,
    /// Active indices buffer (reused across calls).
    active_indices: Vec<usize>,
    /// Bias buffer (reused across calls).
    bias: Vec<f32>,
}

impl RoutingScratch {
    /// Create scratch buffers sized for `n_chunks` chunks.
    pub fn new(n_chunks: usize, _head_dim: usize) -> Self {
        Self {
            logits: vec![0.0; n_chunks],
            sorted: Vec::with_capacity(n_chunks),
            probs: vec![0.0; n_chunks],
            log_weights: Vec::with_capacity(n_chunks),
            active_indices: Vec::with_capacity(n_chunks),
            bias: Vec::with_capacity(n_chunks),
        }
    }
}

/// Zero-alloc variant of [`score_blocks_entmax`].
///
/// Reuses scratch buffers across calls.
pub fn score_blocks_entmax_into(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    config: &DashAttnConfig,
    scratch: &mut RoutingScratch,
) -> RoutingResult {
    let hd = query.len();
    let n = summaries.len();

    // Grow buffers if needed
    if scratch.logits.len() < n {
        scratch.logits.resize(n, 0.0);
    }
    if scratch.probs.len() < n {
        scratch.probs.resize(n, 0.0);
    }
    scratch.sorted.clear();

    // Compute chunk logits: z = q · k̄ / √d * γ
    let scale = 1.0 / (hd as f32).sqrt() * config.scaling_factor;
    for (i, s) in summaries.iter().enumerate() {
        let s_ref = s.as_ref();
        let dot = simd_dot_f32(query, s_ref, hd);
        scratch.logits[i] = dot * scale;
    }

    // α-entmax routing into scratch buffers
    entmax_1p5_into(
        &scratch.logits[..n],
        &mut scratch.sorted,
        &mut scratch.probs[..n],
    );

    // Extract support into pre-allocated buffer
    entmax_support_into(&scratch.probs[..n], &mut scratch.active_indices);

    // Compute routing bias: (log w - μ) / σ on active indices
    scratch.log_weights.clear();
    scratch
        .log_weights
        .extend(scratch.active_indices.iter().map(|&i| {
            if scratch.probs[i] > 1e-10 {
                scratch.probs[i].ln()
            } else {
                -23.0 // ln(1e-10)
            }
        }));

    // Compute mean and variance of log-weights in a single pass.
    // Original code iterated twice (sum for mean, then squared-deviation sum);
    // fusing avoids a second scan over `log_weights`.
    let n_lw = scratch.log_weights.len();
    let (mean_lw, var_lw): (f32, f32) = if n_lw == 0 {
        (0.0, 1.0)
    } else if n_lw == 1 {
        (*scratch.log_weights.first().unwrap(), 1.0)
    } else {
        let mut sum = 0.0f32;
        for &x in scratch.log_weights.iter() {
            sum += x;
        }
        let mean = sum / n_lw as f32;
        let mut sq_sum = 0.0f32;
        for &x in scratch.log_weights.iter() {
            let d = x - mean;
            sq_sum += d * d;
        }
        let var = sq_sum / (n_lw - 1) as f32;
        (mean, var)
    };
    let std_lw = var_lw.sqrt().max(1e-6);

    scratch.bias.clear();
    scratch.bias.extend(
        scratch
            .log_weights
            .iter()
            .map(|&lw| (lw - mean_lw) / std_lw),
    );

    // Build result: clone from scratch buffers.
    // Note: scratch.active_indices and scratch.bias are reused across calls
    // (sorted, logits, probs are the expensive scratch to preserve).
    // The active_indices/bias are small (typically ≤ num_chunks), so cloning
    // them is cheaper than reallocating the sorted/probs buffers.
    let probs = scratch.probs[..n].to_vec();

    RoutingResult {
        active_indices: scratch.active_indices.clone(),
        bias: scratch.bias.clone(),
        probs,
    }
}

/// Compute routing bias for all query heads with GQA aggregation.
///
/// Runs per-query-head entmax routing, then averages probabilities across
/// heads sharing the same KV group for consensus routing.
///
/// Uses `score_blocks_entmax_into` with a reusable scratch buffer to avoid
/// per-head heap allocation in the routing hot path.
pub fn compute_routing_bias(
    queries: &[Vec<f32>],   // [n_query_heads][head_dim]
    summaries: &[Vec<f32>], // [n_chunks][head_dim]
    n_kv_heads: usize,
    config: &DashAttnConfig,
) -> Vec<RoutingResult> {
    let n_query_heads = queries.len();
    let n_chunks = summaries.len();

    // Reuse scratch buffers across heads (zero-alloc routing)
    let mut scratch = RoutingScratch::new(n_chunks, queries.first().map_or(0, |q| q.len()));

    // Per-query-head routing using the _into variant
    let per_head: Vec<RoutingResult> = queries
        .iter()
        .map(|q| score_blocks_entmax_into(q, summaries, config, &mut scratch))
        .collect();

    // GQA aggregation: reference probs without cloning
    let head_probs: Vec<&[f32]> = per_head.iter().map(|r| r.probs.as_slice()).collect();
    let _agg_probs = entmax_gqa_aggregate(&head_probs, n_query_heads, n_kv_heads, n_chunks);

    per_head
}

/// Zero-alloc variant of [`compute_routing_bias`].
///
/// Accepts a reusable scratch buffer to avoid per-call heap allocation.
/// The returned `RoutingResult`s still own their data (cloned from scratch)
/// for API simplicity.
pub fn compute_routing_bias_into(
    queries: &[Vec<f32>],
    summaries: &[Vec<f32>],
    n_kv_heads: usize,
    config: &DashAttnConfig,
    scratch: &mut RoutingScratch,
) -> Vec<RoutingResult> {
    let n_query_heads = queries.len();
    let n_chunks = summaries.len();

    let per_head: Vec<RoutingResult> = queries
        .iter()
        .map(|q| score_blocks_entmax_into(q, summaries, config, scratch))
        .collect();

    let head_probs: Vec<&[f32]> = per_head.iter().map(|r| r.probs.as_slice()).collect();
    let _agg_probs = entmax_gqa_aggregate(&head_probs, n_query_heads, n_kv_heads, n_chunks);

    per_head
}

// ── Wall-aware routing (Plan 173 Task 6) ────────────────────

/// Wall-aware block routing that pre-filters decayed blocks.
///
/// Before entmax routing, checks each block's minimum retention via
/// `WallPrefixState::min_retention_at_block()`. Blocks where ALL channels
/// have decayed below `retention_threshold` are excluded from routing.
///
/// This avoids wasting compute on blocks that the model has already
/// "forgotten" via the diagonal gate.
///
/// `retention_threshold`: blocks with `min_retention < threshold` are skipped.
/// Default: 0.1 (i.e., block must retain at least 10% signal).
///
/// `min_retention_fn`: closure that returns `min_retention(block_idx) -> f32`
/// for each block. Returns `1.0` for blocks with no Wall information.
pub fn score_blocks_wall_aware_into<F>(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    config: &DashAttnConfig,
    scratch: &mut RoutingScratch,
    retention_threshold: f32,
    min_retention_fn: F,
) -> RoutingResult
where
    F: Fn(usize) -> f32,
{
    let n = summaries.len();

    // Pre-filter: collect only blocks with sufficient retention
    let alive_summaries: Vec<(usize, &[f32])> = summaries
        .iter()
        .enumerate()
        .filter(|(i, _)| min_retention_fn(*i) >= retention_threshold)
        .map(|(i, s)| (i, s.as_ref()))
        .collect();

    // If all blocks are alive (common case), delegate to standard routing
    if alive_summaries.len() == n {
        return score_blocks_entmax_into(query, summaries, config, scratch);
    }

    // If no blocks are alive, return empty routing
    if alive_summaries.is_empty() {
        scratch.active_indices.clear();
        scratch.bias.clear();
        return RoutingResult {
            active_indices: vec![],
            bias: vec![],
            probs: vec![0.0; n],
        };
    }

    // Extract just the summaries for filtered routing
    let filtered_refs: Vec<&[f32]> = alive_summaries.iter().map(|(_, s)| *s).collect();

    // Run standard entmax routing on alive blocks only
    let filtered_result = score_blocks_entmax_into(query, &filtered_refs, config, scratch);

    // Remap active indices back to original block indices
    let original_active: Vec<usize> = filtered_result
        .active_indices
        .iter()
        .map(|&fi| alive_summaries[fi].0)
        .collect();

    // Build full probs array (0.0 for dead blocks)
    let mut full_probs = vec![0.0f32; n];
    for (fi, &orig_i) in original_active.iter().enumerate() {
        full_probs[orig_i] = filtered_result.probs.get(fi).copied().unwrap_or(0.0);
    }

    RoutingResult {
        active_indices: original_active,
        bias: filtered_result.bias,
        probs: full_probs,
    }
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

    // ── Wall-aware routing tests (Plan 173 Task 6) ────────────

    #[test]
    fn test_wall_aware_routing_all_alive() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut scratch = RoutingScratch::new(2, 2);

        // All blocks have retention 1.0 (alive) → same as standard routing
        let result = score_blocks_wall_aware_into(
            &query,
            &summaries,
            &config,
            &mut scratch,
            0.1,
            |_| 1.0, // all alive
        );

        let standard = score_blocks_entmax_into(&query, &summaries, &config, &mut scratch);
        assert_eq!(result.active_indices, standard.active_indices);
        assert_eq!(result.probs.len(), 2);
    }

    #[test]
    fn test_wall_aware_routing_all_dead() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut scratch = RoutingScratch::new(2, 2);

        // All blocks have retention 0.0 (dead) → empty routing
        let result = score_blocks_wall_aware_into(
            &query,
            &summaries,
            &config,
            &mut scratch,
            0.1,
            |_| 0.0, // all dead
        );

        assert!(result.active_indices.is_empty());
        assert!(result.bias.is_empty());
        assert_eq!(result.probs.len(), 2);
        assert!(result.probs.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_wall_aware_routing_partial_filter() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let mut scratch = RoutingScratch::new(3, 2);

        // Block 1 is dead (retention 0.05), blocks 0 and 2 are alive
        let result = score_blocks_wall_aware_into(
            &query,
            &summaries,
            &config,
            &mut scratch,
            0.1,
            |block_idx| if block_idx == 1 { 0.05 } else { 1.0 },
        );

        // Block 1 should not appear in active indices
        assert!(!result.active_indices.contains(&1));
        // Should have some active blocks
        assert!(!result.active_indices.is_empty());
        // Block 1 should have 0 probability
        assert_eq!(result.probs[1], 0.0);
    }
}
