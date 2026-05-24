//! Head-wise sparse decode/prefill integration for RTPurbo.
//!
//! Implements the decode-phase routing logic that combines:
//! - Head calibration (retrieval vs local classification)
//! - Low-dimensional projection (pre-RoPE scoring)
//! - Dynamic top-p token selection
//!
//! # Decode Path
//!
//! For each decode step, the routing logic:
//! 1. Computes local window + sink tokens for local heads
//! 2. For each retrieval head:
//!    a. Projects pre-RoPE query/keys into low-dim space
//!    b. Selects top-p tokens via blockwise selection
//!    c. Returns selected indices for full-dim SDPA
//!
//! # Prefill Path
//!
//! During prefill, all heads attend densely — the optimization only applies
//! to the decode phase. This module validates calibration state and returns
//! "dense for all heads".
//!
//! # GOAT Proofs
//!
//! The routing functions are pure (no side effects, no GPU) and operate on
//! synthetic data, enabling formal verification of correctness properties.

use crate::rt_turbo::top_p::select_top_p_blockwise;
use crate::rt_turbo::{HeadCalibration, RetrievalProjection};
use crate::types::RtTurboConfig;

// ---------------------------------------------------------------------------
// Result Types
// ---------------------------------------------------------------------------

/// Result of RTPurbo decode-phase routing.
///
/// Contains per-retrieval-head selected token indices, the local window
/// range for local heads, and sink token positions.
///
/// # Field Semantics
///
/// - `selected_indices[i]` — token indices selected by the i-th retrieval head
///   (ordered by `calibration.retrieval_set`)
/// - `local_window` — `(start, end)` range for sliding window attention on local heads
/// - `sink_indices` — always-included sink token positions
#[derive(Clone, Debug)]
pub struct RtTurboDecodeResult {
    /// Per retrieval head, the selected token indices for sparse attention.
    pub selected_indices: Vec<Vec<usize>>,
    /// `(start, end)` of sliding window for local heads.
    pub local_window: (usize, usize),
    /// Sink token positions (first `config.sink_tokens` positions).
    pub sink_indices: Vec<usize>,
    /// Number of retrieval heads.
    pub n_retrieval_heads: usize,
    /// Number of local heads.
    pub n_local_heads: usize,
}

/// Result of RTPurbo prefill-phase routing.
///
/// During prefill, all heads attend densely — no sparsity optimization.
/// This result confirms calibration state is loaded and reports total head count.
#[derive(Clone, Debug)]
pub struct RtTurboPrefillResult {
    /// All heads attend densely during prefill.
    pub dense_heads: Vec<usize>,
    /// Total number of heads.
    pub n_total_heads: usize,
}

// ---------------------------------------------------------------------------
// Forward Functions
// ---------------------------------------------------------------------------

/// Decode-phase RTPurbo routing: determines which tokens each head attends to.
///
/// Combines head calibration, low-dim projection, and dynamic top-p selection
/// to produce per-head token index sets for sparse decode.
///
/// # Arguments
///
/// * `calibration` — Head classification (retrieval vs local).
/// * `projection` — Low-dim projection weights for retrieval heads.
/// * `config` — RTPurbo hyperparameters.
/// * `kv_cache` — Simulated KV cache: one `Vec<f32>` per position, concatenated
///   K/V for ALL heads (shape: `[seq_len][2 * n_heads * head_dim]`).
///   Used for seq_len determination; actual scoring uses `key_pre_rope`.
/// * `query` — Pre-RoPE query vectors per head (shape: `[n_heads][head_dim]`).
/// * `key_pre_rope` — Pre-RoPE key vectors per position (shape:
///   `[seq_len][n_heads * head_dim]`).
///
/// # Returns
///
/// [`RtTurboDecodeResult`] with per-retrieval-head selected indices,
/// local window range, and sink positions.
///
/// # Panics
///
/// Panics if dimensions are inconsistent (e.g., query length != n_heads,
/// or projection head count != calibration retrieval count).
pub fn forward_rt_turbo_decode(
    calibration: &HeadCalibration,
    projection: &RetrievalProjection,
    config: &RtTurboConfig,
    kv_cache: &[Vec<f32>],
    query: &[Vec<f32>],
    key_pre_rope: &[Vec<f32>],
) -> RtTurboDecodeResult {
    let seq_len = kv_cache.len();
    let n_heads = calibration.n_heads();
    let head_dim = projection.head_dim();

    // Validate dimensions
    assert_eq!(
        query.len(),
        n_heads,
        "query length ({}) must match n_heads ({n_heads})",
        query.len(),
    );
    assert_eq!(
        key_pre_rope.len(),
        seq_len,
        "key_pre_rope length ({}) must match kv_cache length ({seq_len})",
        key_pre_rope.len(),
    );
    assert_eq!(
        projection.n_retrieval_heads(),
        calibration.n_retrieval(),
        "projection n_retrieval_heads ({}) must match calibration ({})",
        projection.n_retrieval_heads(),
        calibration.n_retrieval(),
    );
    if !key_pre_rope.is_empty() {
        assert_eq!(
            key_pre_rope[0].len(),
            n_heads * head_dim,
            "key_pre_rope[0] length ({}) must match n_heads ({n_heads}) * head_dim ({head_dim})",
            key_pre_rope[0].len(),
        );
    }

    // Step 1: Local window [max(0, seq_len - sliding_window), seq_len)
    let local_window_start = seq_len.saturating_sub(config.sliding_window);
    let local_window = (local_window_start, seq_len);

    // Step 2: Sink tokens — first `sink_tokens` positions
    let sink_count = config.sink_tokens.min(seq_len);
    let sink_indices: Vec<usize> = (0..sink_count).collect();

    // Step 3: For each retrieval head, compute low-dim scores and select top-p
    let mut selected_indices: Vec<Vec<usize>> = Vec::with_capacity(calibration.n_retrieval());

    for (retrieval_idx, &global_head) in calibration.retrieval_set.iter().enumerate() {
        // Get pre-RoPE query for this head
        let q_pre = &query[global_head];
        assert_eq!(
            q_pre.len(),
            head_dim,
            "query[{global_head}] dimension ({}) must match head_dim ({head_dim})",
            q_pre.len(),
        );

        // Extract pre-RoPE keys for this head from all positions.
        // key_pre_rope[pos] has shape [n_heads * head_dim];
        // keys for head h start at offset h * head_dim.
        let k_cache: Vec<f32> = key_pre_rope
            .iter()
            .flat_map(|pos_keys| {
                let offset = global_head * head_dim;
                pos_keys[offset..offset + head_dim].to_vec()
            })
            .collect();

        // Compute low-dim scores via projection
        let scores = projection.batch_project_scores(retrieval_idx, q_pre, &k_cache);

        // Select top-p tokens via blockwise selection
        let top_p_result = select_top_p_blockwise(&scores, config.top_p, config.block_size);

        // Merge with sink indices (union of top-p selected + sinks)
        let mut merged: Vec<usize> = top_p_result.selected_indices;
        for &sink in &sink_indices {
            if !merged.contains(&sink) {
                merged.push(sink);
            }
        }
        merged.sort_unstable();

        selected_indices.push(merged);
    }

    RtTurboDecodeResult {
        selected_indices,
        local_window,
        sink_indices,
        n_retrieval_heads: calibration.n_retrieval(),
        n_local_heads: calibration.n_local(),
    }
}

/// Prefill-phase RTPurbo routing: all heads attend densely.
///
/// During prefill, the sparsity optimization does not apply — all heads
/// attend to all positions. This function validates calibration state
/// and returns a result indicating dense attention for every head.
///
/// # Arguments
///
/// * `calibration` — Head classification (used for head count validation).
/// * `projection` — Projection weights (validated as loaded, not used for routing).
///
/// # Returns
///
/// [`RtTurboPrefillResult`] with all head indices as dense.
///
/// # Panics
///
/// Panics if projection head count doesn't match calibration retrieval count.
pub fn forward_rt_turbo_prefill(
    calibration: &HeadCalibration,
    projection: &RetrievalProjection,
) -> RtTurboPrefillResult {
    let n_total_heads = calibration.n_heads();

    assert_eq!(
        projection.n_retrieval_heads(),
        calibration.n_retrieval(),
        "projection n_retrieval_heads ({}) must match calibration ({})",
        projection.n_retrieval_heads(),
        calibration.n_retrieval(),
    );

    let dense_heads: Vec<usize> = (0..n_total_heads).collect();

    RtTurboPrefillResult {
        dense_heads,
        n_total_heads,
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

/// Per-layer state for RTPurbo decode.
///
/// Stores calibration, projection weights, and reusable token indices.
/// Created once per layer at model load time, updated each decode step.
///
/// # Lifecycle
///
/// 1. Create with [`RtTurboCache::new`] at model load
/// 2. Call [`RtTurboCache::prefill`] during prompt processing
/// 3. Call [`RtTurboCache::decode`] each decode step
/// 4. Call [`RtTurboCache::update_selected_indices`] to cache results
#[derive(Clone, Debug)]
pub struct RtTurboCache {
    /// Calibration result (head classification).
    pub calibration: HeadCalibration,
    /// Low-dim projection weights.
    pub projection: RetrievalProjection,
    /// Configuration.
    pub config: RtTurboConfig,
    /// Last selected indices per retrieval head (reusable until KV shifts).
    pub last_selected_indices: Vec<Vec<usize>>,
    /// Layer index this cache is for.
    pub layer_idx: usize,
}

impl RtTurboCache {
    /// Create a new per-layer RTPurbo cache.
    ///
    /// # Arguments
    ///
    /// * `calibration` — Head classification from offline calibration.
    /// * `projection` — Trained low-dim projection weights.
    /// * `config` — RTPurbo hyperparameters.
    /// * `layer_idx` — Which transformer layer this cache serves.
    pub fn new(
        calibration: HeadCalibration,
        projection: RetrievalProjection,
        config: RtTurboConfig,
        layer_idx: usize,
    ) -> Self {
        let n_retrieval = calibration.n_retrieval();
        Self {
            calibration,
            projection,
            config,
            last_selected_indices: vec![vec![]; n_retrieval],
            layer_idx,
        }
    }

    /// Execute decode-phase routing for this layer.
    ///
    /// Delegates to [`forward_rt_turbo_decode`] with this cache's
    /// calibration, projection, and config.
    pub fn decode(
        &self,
        kv_cache: &[Vec<f32>],
        query: &[Vec<f32>],
        key_pre_rope: &[Vec<f32>],
    ) -> RtTurboDecodeResult {
        forward_rt_turbo_decode(
            &self.calibration,
            &self.projection,
            &self.config,
            kv_cache,
            query,
            key_pre_rope,
        )
    }

    /// Execute prefill-phase routing for this layer.
    ///
    /// Delegates to [`forward_rt_turbo_prefill`] with this cache's
    /// calibration and projection.
    pub fn prefill(&self) -> RtTurboPrefillResult {
        forward_rt_turbo_prefill(&self.calibration, &self.projection)
    }

    /// Update cached selected indices after a decode step.
    ///
    /// Stores the selected indices for potential reuse in subsequent
    /// decode steps (until KV cache shifts invalidate them).
    pub fn update_selected_indices(&mut self, indices: Vec<Vec<usize>>) {
        self.last_selected_indices = indices;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rt_turbo::calibration::calibrate_from_scores;
    use crate::types::RtTurboConfig;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn default_config() -> RtTurboConfig {
        RtTurboConfig::default()
    }

    /// Micro config for fast tests: small dims, small window.
    fn micro_config() -> RtTurboConfig {
        RtTurboConfig {
            retrieval_head_ratio: 0.5,
            low_dim: 4,
            top_p: 0.9,
            sliding_window: 8,
            sink_tokens: 2,
            block_size: 4,
        }
    }

    /// Create a calibration with the first `n_retrieval` heads as retrieval.
    ///
    /// Assigns decreasing scores to the first `n_retrieval` heads so they
    /// are always classified as retrieval under the given ratio.
    fn make_calibration(
        n_heads: usize,
        n_retrieval: usize,
        config: &RtTurboConfig,
    ) -> HeadCalibration {
        let mut scores = vec![0.0f32; n_heads];
        for i in 0..n_retrieval.min(n_heads) {
            scores[i] = 1.0 - i as f32 * 0.01;
        }
        calibrate_from_scores(&scores, config)
    }

    /// Create identity projection for testing.
    fn make_projection(
        n_retrieval_heads: usize,
        head_dim: usize,
        low_dim: usize,
    ) -> RetrievalProjection {
        RetrievalProjection::identity(n_retrieval_heads, head_dim, low_dim)
    }

    /// Create synthetic KV cache: `[seq_len][2 * n_heads * head_dim]`.
    fn make_kv_cache(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<Vec<f32>> {
        let kv_dim = 2 * n_heads * head_dim;
        (0..seq_len)
            .map(|pos| vec![pos as f32 * 0.1; kv_dim])
            .collect()
    }

    /// Create synthetic query vectors: `[n_heads][head_dim]`.
    fn make_query(n_heads: usize, head_dim: usize, value: f32) -> Vec<Vec<f32>> {
        (0..n_heads).map(|_| vec![value; head_dim]).collect()
    }

    /// Create synthetic pre-RoPE keys: `[seq_len][n_heads * head_dim]`.
    fn make_key_pre_rope(seq_len: usize, n_heads: usize, head_dim: usize) -> Vec<Vec<f32>> {
        let total_dim = n_heads * head_dim;
        (0..seq_len)
            .map(|pos| {
                (0..total_dim)
                    .map(|d| (pos * total_dim + d) as f32 * 0.01)
                    .collect()
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // T20: Integration Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_local_heads_get_window() {
        let config = micro_config();
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 20;
        let n_retrieval = 2;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        assert_eq!(calibration.n_retrieval(), n_retrieval);
        assert_eq!(calibration.n_local(), n_heads - n_retrieval);

        let projection = make_projection(n_retrieval, head_dim, config.low_dim);
        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        // Local window should be [seq_len - sliding_window, seq_len)
        let expected_start = seq_len.saturating_sub(config.sliding_window);
        assert_eq!(result.local_window, (expected_start, seq_len));
        assert_eq!(result.n_local_heads, n_heads - n_retrieval);
    }

    #[test]
    fn test_decode_retrieval_heads_get_top_p() {
        let config = micro_config();
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 16;
        let n_retrieval = 2;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(n_retrieval, head_dim, config.low_dim);
        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        // Should have selected indices for each retrieval head
        assert_eq!(result.selected_indices.len(), n_retrieval);

        // Each retrieval head should have at least 1 selected token
        for indices in &result.selected_indices {
            assert!(
                !indices.is_empty(),
                "retrieval head should select at least 1 token"
            );
            // All indices must be valid positions
            for &idx in indices {
                assert!(idx < seq_len, "index {idx} must be < seq_len {seq_len}");
            }
        }
    }

    #[test]
    fn test_decode_sink_tokens_always_included() {
        let config = micro_config(); // sink_tokens = 2
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 16;
        let n_retrieval = 2;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(n_retrieval, head_dim, config.low_dim);
        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        // Sink tokens should be [0, 1] for sink_tokens=2
        assert_eq!(result.sink_indices, vec![0, 1]);

        // Every retrieval head's selection must include all sink tokens
        for (i, indices) in result.selected_indices.iter().enumerate() {
            for &sink in &result.sink_indices {
                assert!(
                    indices.contains(&sink),
                    "retrieval head {i} must include sink token {sink}, got {indices:?}",
                );
            }
        }
    }

    #[test]
    fn test_prefill_all_heads_dense() {
        let config = micro_config();
        let n_heads = 4;
        let n_retrieval = 2;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(n_retrieval, 8, config.low_dim);

        let result = forward_rt_turbo_prefill(&calibration, &projection);

        // All heads should be dense
        assert_eq!(result.n_total_heads, n_heads);
        assert_eq!(result.dense_heads, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_cache_round_trip() {
        let config = micro_config();
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 16;
        let n_retrieval = 2;
        let layer_idx = 3;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(n_retrieval, head_dim, config.low_dim);

        let mut cache = RtTurboCache::new(calibration, projection, config, layer_idx);
        assert_eq!(cache.layer_idx, layer_idx);
        assert_eq!(cache.last_selected_indices.len(), n_retrieval);

        // Prefill
        let prefill_result = cache.prefill();
        assert_eq!(prefill_result.n_total_heads, n_heads);

        // Decode
        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let decode_result = cache.decode(&kv_cache, &query, &key_pre_rope);
        assert_eq!(decode_result.n_retrieval_heads, n_retrieval);
        assert!(!decode_result.selected_indices.is_empty());

        // Update indices
        let indices = decode_result.selected_indices.clone();
        cache.update_selected_indices(indices.clone());
        assert_eq!(cache.last_selected_indices, indices);

        // Decode again — should produce the same result
        let decode_result2 = cache.decode(&kv_cache, &query, &key_pre_rope);
        assert_eq!(decode_result2.n_retrieval_heads, n_retrieval);
        assert_eq!(
            decode_result2.selected_indices,
            decode_result.selected_indices
        );
    }

    #[test]
    fn test_empty_kv_cache() {
        let config = micro_config();
        let n_heads = 4;
        let head_dim = 8;
        let n_retrieval = 2;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(n_retrieval, head_dim, config.low_dim);

        let kv_cache: Vec<Vec<f32>> = vec![];
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope: Vec<Vec<f32>> = vec![];

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        // Should handle gracefully
        assert_eq!(result.local_window, (0, 0));
        assert!(result.sink_indices.is_empty());
        // Retrieval heads select nothing from empty cache
        for indices in &result.selected_indices {
            assert!(indices.is_empty());
        }
    }

    #[test]
    fn test_single_position_kv() {
        let config = micro_config();
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 1;
        let n_retrieval = 2;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(n_retrieval, head_dim, config.low_dim);

        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        // Local window should be [0, 1)
        assert_eq!(result.local_window, (0, 1));

        // Sink tokens: min(sink_tokens=2, seq_len=1) = 1 → [0]
        assert_eq!(result.sink_indices, vec![0]);

        // Each retrieval head should select position 0
        for indices in &result.selected_indices {
            assert!(
                indices.contains(&0),
                "must contain position 0, got {indices:?}"
            );
        }
    }

    #[test]
    fn test_decode_output_shapes() {
        let config = micro_config();
        let n_heads = 6;
        let head_dim = 8;
        let seq_len = 32;
        let n_retrieval = 3; // 50% of 6

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        assert_eq!(calibration.n_retrieval(), n_retrieval);
        assert_eq!(calibration.n_local(), n_heads - n_retrieval);

        let projection = make_projection(n_retrieval, head_dim, config.low_dim);
        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        // Shape checks
        assert_eq!(result.selected_indices.len(), n_retrieval);
        assert_eq!(result.n_retrieval_heads, n_retrieval);
        assert_eq!(result.n_local_heads, n_heads - n_retrieval);
        assert_eq!(result.sink_indices.len(), config.sink_tokens);

        // Local window end == seq_len
        assert_eq!(result.local_window.1, seq_len);
        assert!(result.local_window.0 <= seq_len);

        // All selected indices are within [0, seq_len)
        for indices in &result.selected_indices {
            for &idx in indices {
                assert!(idx < seq_len, "index {idx} must be < seq_len {seq_len}");
            }
        }
    }

    #[test]
    fn test_decode_short_seq_uses_full_window() {
        // When seq_len < sliding_window, local window starts at 0
        let config = default_config(); // sliding_window = 8192
        let n_heads = 4;
        let head_dim = 16;
        let seq_len = 100; // much less than 8192
        let n_retrieval = 1; // 15% of 4 → ceil(0.6) = 1

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(calibration.n_retrieval(), head_dim, config.low_dim);
        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 0.5);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        // Full context is within the window
        assert_eq!(result.local_window, (0, seq_len));
    }

    #[test]
    fn test_decode_zero_sink_tokens() {
        let mut config = micro_config();
        config.sink_tokens = 0;
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 16;
        let n_retrieval = 2;

        let calibration = make_calibration(n_heads, n_retrieval, &config);
        let projection = make_projection(n_retrieval, head_dim, config.low_dim);
        let kv_cache = make_kv_cache(seq_len, n_heads, head_dim);
        let query = make_query(n_heads, head_dim, 1.0);
        let key_pre_rope = make_key_pre_rope(seq_len, n_heads, head_dim);

        let result = forward_rt_turbo_decode(
            &calibration,
            &projection,
            &config,
            &kv_cache,
            &query,
            &key_pre_rope,
        );

        assert!(result.sink_indices.is_empty());
        // Retrieval heads still get top-p selection (no sinks added)
        for indices in &result.selected_indices {
            assert!(!indices.is_empty());
        }
    }

    #[test]
    fn test_prefill_zero_heads() {
        // Edge: all heads are local (0 retrieval heads)
        let config = micro_config();
        let n_heads = 4;

        let calibration = HeadCalibration::all_local(n_heads, &config);
        assert_eq!(calibration.n_retrieval(), 0);

        let projection = make_projection(0, 8, config.low_dim);
        let result = forward_rt_turbo_prefill(&calibration, &projection);

        assert_eq!(result.n_total_heads, n_heads);
        assert_eq!(result.dense_heads.len(), n_heads);
    }
}
