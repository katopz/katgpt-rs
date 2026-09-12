//! EntmaxRouter — thin VortexFlow wrapper over existing DashAttention entmax routing.
//!
//! Delegates to `score_blocks_entmax` for query-dependent block selection.
//! Validates that VortexFlow doesn't regress DashAttention behavior.
//! Feature gate: `vortex_flow` (Plan 196, Phase 1).

use katgpt_core::types::DashAttnConfig;

#[cfg(test)]
use super::routing::score_blocks_entmax;
use super::routing::score_blocks_entmax_into;
#[cfg(feature = "asentmax_schedule")]
use super::routing::score_blocks_entmax_with_schedule_into;
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
    /// ASEntmax length-adaptive damping (Issue 747 P0.7): when `Some`, the
    /// indexer routes through `score_blocks_entmax_with_schedule_into` with
    /// the rolling-σ̂ estimator observing every raw logit row (the
    /// Bench-713-validated pattern: `est.to_schedule()` per call, σ̂ lags one
    /// row — the documented EMA warm-start). `None` (default) is the shipped
    /// Plan 106 path, bit-identical.
    #[cfg(feature = "asentmax_schedule")]
    pub asentmax: Option<crate::dash_attn::asentmax::RollingSigmaEstimator>,
}

impl EntmaxRouter {
    /// Create a new EntmaxRouter with the given DashAttention config.
    pub fn new(config: DashAttnConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "asentmax_schedule")]
            asentmax: None,
        }
    }

    /// Create with default DashAttention config.
    pub fn default_router() -> Self {
        Self {
            config: DashAttnConfig::default(),
            #[cfg(feature = "asentmax_schedule")]
            asentmax: None,
        }
    }

    /// Enable the ASEntmax derived damping schedule (rolling-σ̂ estimator,
    /// α = 0.8 EMA — the Bench 713 G2 configuration). Issue 747 P0.7.
    #[cfg(feature = "asentmax_schedule")]
    pub fn with_asentmax_schedule(mut self) -> Self {
        self.asentmax = Some(crate::dash_attn::asentmax::RollingSigmaEstimator::new(0.8));
        self
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
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        if n_blocks == 0 {
            return RoutingDecision::new();
        }

        let n = n_blocks.min(cache.summaries.len());
        let summaries = &cache.summaries[..n];

        // Delegate to existing entmax routing, reusing the scratch's
        // RoutingScratch buffers (avoids 5 Vec allocations per call).
        // With `asentmax` set (Issue 747 P0.7), the logits row is damped by
        // `est.to_schedule()` before the threshold pass — the derived
        // counter-schedule to over-sparsification at growing n_blocks.
        #[cfg(feature = "asentmax_schedule")]
        let result = match &self.asentmax {
            Some(est) => score_blocks_entmax_with_schedule_into(
                query,
                summaries,
                &[],
                &self.config,
                &est.to_schedule(),
                Some(est),
                &mut scratch.routing_scratch,
            ),
            None => score_blocks_entmax_into(
                query,
                summaries,
                &self.config,
                &mut scratch.routing_scratch,
            ),
        };
        #[cfg(not(feature = "asentmax_schedule"))]
        let result =
            score_blocks_entmax_into(query, summaries, &self.config, &mut scratch.routing_scratch);

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

    // ── Issue 747 P0.7: the schedule socket on the router ────────────────

    /// Deterministic splitmix64 → f32 uniform [0,1).
    // Gated with its only callers: at default features the schedule tests
    // compile away and an ungated `unit` is dead code in the lib test build.
    #[cfg(feature = "asentmax_schedule")]
    fn unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 40) as f32) / ((1u64 << 24) as f32)
    }

    #[cfg(feature = "asentmax_schedule")]
    #[test]
    fn asentmax_none_default_is_the_shipped_path() {
        // The default router must produce byte-identical decisions to the
        // pre-P0.7 path (score_blocks_entmax_into) — the feature-gate-audit
        // discipline: enabling the feature alone changes nothing.
        let router = EntmaxRouter::default_router();
        assert!(router.asentmax.is_none());
        let mut cache = router.cache_new(4, 3);
        let mut scratch = VortexScratch::new(4);
        let summaries: Vec<Vec<f32>> = vec![
            vec![1.0, 0.2, 0.0],
            vec![0.0, 1.0, 0.1],
            vec![0.3, 0.0, 1.0],
            vec![0.9, 0.9, 0.2],
        ];
        for (i, s) in summaries.iter().enumerate() {
            router.forward_cache(&mut cache, s, &[0.0; 3], i, 3);
        }
        let query = vec![1.0, 0.5, 0.25];
        let via_router = router.forward_indexer(&query, &cache, 4, 4, &mut scratch);
        let direct = score_blocks_entmax(&query, &summaries, &router.config);
        assert_eq!(via_router.blocks.len(), direct.active_indices.len());
        for ((&rb, &w), &di) in via_router
            .blocks
            .iter()
            .zip(via_router.weights.iter())
            .zip(direct.active_indices.iter())
        {
            assert_eq!(rb, di);
            assert_eq!(w.to_bits(), direct.probs[di].to_bits());
        }
    }

    #[cfg(feature = "asentmax_schedule")]
    #[test]
    fn asentmax_scheduled_router_holds_support_at_large_sigma() {
        // The G2 story at router level: at inflated logit scale (σ = 8) the
        // unscheduled router collapses the support (over-sparsification);
        // `with_asentmax_schedule()` holds a multi-block support.
        let n = 256_usize;
        let hd = 8_usize;
        let mut state = 0x7E57_u64;

        let mut summaries: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let s: Vec<f32> = (0..hd).map(|_| (unit(&mut state) - 0.5) * 16.0).collect();
            summaries.push(s);
        }
        let query: Vec<f32> = (0..hd).map(|_| (unit(&mut state) - 0.5) * 16.0).collect();

        let build = |sched: bool| {
            let router = if sched {
                EntmaxRouter::default_router().with_asentmax_schedule()
            } else {
                EntmaxRouter::default_router()
            };
            let mut cache = router.cache_new(n, hd);
            let vals = vec![0.0f32; hd];
            for (i, s) in summaries.iter().enumerate() {
                router.forward_cache(&mut cache, s, &vals, i, hd);
            }
            let mut scratch = VortexScratch::new(n);
            // Warm-up: converge the router's rolling σ̂ on this row scale
            // before the measured call (the estimator persists across calls;
            // single-call σ̂ would still be the warm-start 1.0).
            for _ in 0..8 {
                let _ = router.forward_indexer(&query, &cache, n, n, &mut scratch);
            }
            router.forward_indexer(&query, &cache, n, n, &mut scratch)
        };

        let raw = build(false);
        let sched = build(true);
        // σ = 8 × mean-pooled summaries ⇒ raw logits span ~±40: the raw
        // arm collapses toward 1-2 blocks, the scheduled arm holds ≥ 4×
        // that (mirrors the Bench 713 G2 measured shape at σ=8).
        assert!(
            sched.blocks.len() >= raw.blocks.len(),
            "scheduled support {} < raw {}",
            sched.blocks.len(),
            raw.blocks.len()
        );
        assert!(sched.blocks.len() >= 4, "scheduled support {} too small", sched.blocks.len());
        // Decision weights are the selected blocks' probabilities — a full
        // support selection sums to the simplex total.
        let w_sum: f32 = sched.weights.iter().sum();
        assert!(w_sum <= 1.0 + 1e-5, "weights sum {w_sum} exceeds 1");
    }
}
