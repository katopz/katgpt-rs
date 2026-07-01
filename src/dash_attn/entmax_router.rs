//! EntmaxRouter — thin VortexFlow wrapper over existing DashAttention entmax routing.
//!
//! Delegates to `score_blocks_entmax` for query-dependent block selection.
//! Validates that VortexFlow doesn't regress DashAttention behavior.
//! Feature gate: `vortex_flow` (Plan 196, Phase 1).

use katgpt_core::types::DashAttnConfig;

use super::routing::score_blocks_entmax;
use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// EntmaxCache
// ---------------------------------------------------------------------------

/// Cache for EntmaxRouter: per-block key summaries.
///
/// Stores the summary vectors that `score_blocks_entmax` expects as input.
/// In the full DashAttention pipeline, these come from `ChunkSummaryCache`.
#[derive(Debug, Clone)]
pub struct EntmaxCache {
    /// Per-block key summaries: `[n_blocks][head_dim]`.
    pub summaries: Vec<Vec<f32>>,
    /// Head dimension.
    pub head_dim: usize,
}

impl EntmaxCache {
    /// Create a new empty cache.
    pub fn new(head_dim: usize) -> Self {
        Self {
            summaries: Vec::new(),
            head_dim,
        }
    }

    /// Create a pre-allocated cache for `n_blocks_capacity` blocks.
    pub fn with_capacity(n_blocks_capacity: usize, head_dim: usize) -> Self {
        Self {
            summaries: Vec::with_capacity(n_blocks_capacity),
            head_dim,
        }
    }

    /// Number of cached blocks.
    pub fn n_blocks(&self) -> usize {
        self.summaries.len()
    }

    /// Clear all summaries.
    pub fn clear(&mut self) {
        self.summaries.clear();
    }
}

// ---------------------------------------------------------------------------
// EntmaxRouter
// ---------------------------------------------------------------------------

/// EntmaxRouter — wraps existing `score_blocks_entmax` as a VortexFlow impl.
///
/// Uses α-entmax (α=1.5) for adaptive sparse block selection.
/// This router validates that the VortexFlow trait doesn't regress DashAttention.
#[derive(Debug)]
pub struct EntmaxRouter {
    /// DashAttention config (controls scaling_factor, alpha, etc.).
    pub config: DashAttnConfig,
}

impl EntmaxRouter {
    /// Create a new EntmaxRouter with the given DashAttention config.
    pub fn new(config: DashAttnConfig) -> Self {
        Self { config }
    }

    /// Create with default DashAttention config.
    pub fn default_router() -> Self {
        Self {
            config: DashAttnConfig::default(),
        }
    }
}

impl VortexFlow for EntmaxRouter {
    type Cache = EntmaxCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        _values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        // Extend summaries vec if needed
        if block_idx >= cache.summaries.len() {
            cache
                .summaries
                .resize_with(block_idx + 1, || vec![0.0; head_dim]);
        }

        let block_size = keys.len() / head_dim;
        if block_size == 0 {
            cache.summaries[block_idx].fill(0.0);
            return;
        }

        // Mean pooling of keys → summary (same as zero-init ChunkSummaryQuery).
        // Uses the crate SIMD add + scale kernels (matches ChannelAwareRouter's
        // mean-pool) instead of a scalar nested loop.
        let summary = &mut cache.summaries[block_idx];
        summary.resize(head_dim, 0.0);
        summary.fill(0.0);
        for t in 0..block_size {
            let k_start = t * head_dim;
            katgpt_core::simd::simd_add_inplace(summary, &keys[k_start..k_start + head_dim]);
        }
        let inv = 1.0 / block_size as f32;
        katgpt_core::simd::simd_scale_inplace(summary, inv);
    }

    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        _scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        if n_blocks == 0 {
            return RoutingDecision::new();
        }

        let n = n_blocks.min(cache.summaries.len());
        let summaries = &cache.summaries[..n];

        // Delegate to existing entmax routing
        let result = score_blocks_entmax(query, summaries, &self.config);

        // Convert RoutingResult → RoutingDecision
        // Take top_k from active indices (already sorted by entmax support)
        let k = top_k.min(result.active_indices.len());
        let mut decision = RoutingDecision::with_capacity(k);
        for &idx in &result.active_indices[..k] {
            decision.blocks.push(idx);
            decision.weights.push(result.probs[idx]);
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        EntmaxCache::with_capacity(n_blocks_capacity, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;

    fn make_router() -> EntmaxRouter {
        EntmaxRouter::default_router()
    }

    #[test]
    fn test_entmax_router_single_block() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);
        let mut scratch = VortexScratch::new(1);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(decision.blocks[0], 0);
        assert!(decision.weights[0] > 0.99);
    }

    #[test]
    fn test_entmax_router_selects_aligned_block() {
        let router = make_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        // Block 0: aligned with [1,0,0,0]
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        // Block 1: aligned with [0,1,0,0]
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];

        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);
        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(decision.blocks[0], 0);
    }

    #[test]
    fn test_entmax_router_matches_direct_call() {
        let router = make_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        let summaries_data: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let hd = 3;

        // Populate cache via forward_cache
        for (i, summary) in summaries_data.iter().enumerate() {
            // Pass keys = summary (single token block)
            router.forward_cache(&mut cache, summary, &[0.0; 3], i, hd);
        }

        let query = vec![1.0, 0.5, 0.0];

        // Direct call
        let direct = score_blocks_entmax(&query, &summaries_data, &router.config);

        // Via router
        let decision = router.forward_indexer(&query, &cache, 3, 3, &mut scratch);

        // Active indices should match (entmax support)
        assert_eq!(decision.blocks.len(), direct.active_indices.len());
        for (router_idx, &direct_idx) in decision.blocks.iter().zip(direct.active_indices.iter()) {
            assert_eq!(*router_idx, direct_idx);
        }
    }

    #[test]
    fn test_entmax_router_empty_cache() {
        let router = make_router();
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_entmax_cache_clear() {
        let mut cache = EntmaxCache::new(HEAD_DIM);
        cache.summaries.push(vec![1.0; HEAD_DIM]);
        cache.summaries.push(vec![2.0; HEAD_DIM]);
        assert_eq!(cache.n_blocks(), 2);
        cache.clear();
        assert_eq!(cache.n_blocks(), 0);
    }

    #[test]
    fn test_entmax_cache_sparse_indices() {
        let router = make_router();
        let mut cache = router.cache_new(5, HEAD_DIM);

        // Insert at index 0 and 3 (skipping 1, 2)
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let keys3 = vec![0.0, 1.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];

        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys3, &vals, 3, HEAD_DIM);

        assert_eq!(cache.summaries.len(), 4);
        assert_eq!(cache.summaries[0], vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(cache.summaries[3], vec![0.0, 1.0, 0.0, 0.0]);
        // Gap filled with zeros
        assert_eq!(cache.summaries[1], vec![0.0; HEAD_DIM]);
        assert_eq!(cache.summaries[2], vec![0.0; HEAD_DIM]);
    }
}
