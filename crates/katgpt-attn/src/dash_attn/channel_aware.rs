//! Channel-aware routing with SIMD-optimized dot products (Plan 196, Phase 2).
//!
//! Discovers routing-critical channel groups via ablation calibration, then
//! builds a compressed routing index over only those channels for faster
//! decode-time block selection.
//!
//! Feature gate: `vortex_flow` (Plan 196, Phase 2, default-OFF).

use super::block_topk::{BlockTopKRouter, argtopk_with_scratch, sigmoid};
use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// RoutingChannelMask
// ---------------------------------------------------------------------------

/// Bitset marking which channels are routing-critical.
///
/// `mask[d] == true` means dimension `d` participates in the routing dot product.
/// Dimensions not in the mask are skipped, reducing the effective routing dimension.
#[derive(Debug, Clone)]
pub struct RoutingChannelMask {
    /// Per-dimension flags: `true` if channel is routing-critical.
    pub channels: Vec<bool>,
}

impl RoutingChannelMask {
    /// Create a mask where all channels are active (full-dim routing).
    pub fn all(head_dim: usize) -> Self {
        Self {
            channels: vec![true; head_dim],
        }
    }

    /// Create a mask where no channels are active (degenerate — fallback to full).
    pub fn none(head_dim: usize) -> Self {
        Self {
            channels: vec![false; head_dim],
        }
    }

    /// Indices of routing-critical channels.
    pub fn routing_channels(&self) -> Vec<usize> {
        self.channels
            .iter()
            .enumerate()
            .filter(|&(_, active)| *active)
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of routing-critical channels.
    pub fn routing_dim(&self) -> usize {
        self.channels.iter().filter(|&&a| a).count()
    }

    /// Whether any channels are selected (false → fallback to full-dim).
    pub fn has_routing_channels(&self) -> bool {
        self.channels.iter().any(|&a| a)
    }
}

// ---------------------------------------------------------------------------
// RoutingChannelDiscovery — T10
// ---------------------------------------------------------------------------

/// One-time calibration that discovers routing-critical channel groups.
///
/// For each of 8 channel groups (g0..g7): mask the group, run routing,
/// and measure the accuracy delta. If masking a group causes >5% accuracy
/// drop, those channels are routing-critical.
///
/// This calibration is run once per model (~5 min).
pub struct RoutingChannelDiscovery {
    /// Number of channel groups to test (default: 8).
    pub n_groups: usize,
    /// Accuracy drop threshold for critical channel detection (default: 0.05 = 5%).
    pub critical_threshold: f32,
}

impl RoutingChannelDiscovery {
    /// Create with default settings: 8 groups, 5% threshold.
    pub fn new() -> Self {
        Self {
            n_groups: 8,
            critical_threshold: 0.05,
        }
    }

    /// Run calibration on synthetic calibration data.
    ///
    /// # Arguments
    /// * `head_dim` — dimension per head
    /// * `block_centroids` — `[n_blocks * head_dim]` flattened centroid vectors
    /// * `queries` — `[n_queries * head_dim]` flattened query vectors
    /// * `top_k` — number of blocks to select
    ///
    /// # Returns
    /// `RoutingChannelMask` where bit=1 if channel is routing-critical.
    pub fn calibrate(
        &self,
        head_dim: usize,
        block_centroids: &[f32],
        queries: &[f32],
        top_k: usize,
    ) -> RoutingChannelMask {
        let n_blocks = block_centroids.len() / head_dim;
        let n_queries = queries.len() / head_dim;
        if n_blocks == 0 || n_queries == 0 {
            return RoutingChannelMask::all(head_dim);
        }

        // Baseline: full-dim routing accuracy (block overlap)
        let baseline_accuracy = self.measure_accuracy(
            head_dim,
            block_centroids,
            queries,
            n_blocks,
            n_queries,
            top_k,
            &RoutingChannelMask::all(head_dim),
        );

        // Test each group
        let group_size = head_dim.div_ceil(self.n_groups);
        let mut critical = vec![false; head_dim];

        for g in 0..self.n_groups {
            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(head_dim);

            // Create mask with this group disabled
            let mut mask = RoutingChannelMask::all(head_dim);
            for slot in &mut mask.channels[g_start..g_end] {
                *slot = false;
            }

            let accuracy = self.measure_accuracy(
                head_dim,
                block_centroids,
                queries,
                n_blocks,
                n_queries,
                top_k,
                &mask,
            );

            let delta = baseline_accuracy - accuracy;
            if delta > self.critical_threshold {
                // Masking this group hurt → channels in this group are critical
                for c in critical.iter_mut().take(g_end).skip(g_start) {
                    *c = true;
                }
            }
        }

        // If no channels are critical, fall back to all (safer default)
        let has_critical = critical.iter().any(|&c| c);
        if has_critical {
            RoutingChannelMask { channels: critical }
        } else {
            RoutingChannelMask::all(head_dim)
        }
    }

    /// Measure routing accuracy as block overlap with full-dim selection.
    #[allow(clippy::too_many_arguments)]
    fn measure_accuracy(
        &self,
        head_dim: usize,
        block_centroids: &[f32],
        queries: &[f32],
        n_blocks: usize,
        n_queries: usize,
        top_k: usize,
        mask: &RoutingChannelMask,
    ) -> f32 {
        let routing_channels = mask.routing_channels();
        let routing_dim = routing_channels.len();

        // If no routing channels left, return 0 accuracy
        if routing_dim == 0 {
            return 0.0;
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut total_overlap = 0.0f32;

        // Scratch hoisted out of the per-query loop: every buffer below is
        // fully overwritten (scores) or cleared by `argtopk` (indices) on each
        // iteration, so reuse is observationally identical while dropping four
        // heap allocations per query.
        let mut full_scores = vec![0.0f32; n_blocks];
        let mut masked_scores = vec![0.0f32; n_blocks];
        let mut full_indices = Vec::with_capacity(top_k);
        let mut masked_indices = Vec::with_capacity(top_k);
        let mut topk_pairs: Vec<(usize, f32)> = Vec::new();

        for qi in 0..n_queries {
            let query = &queries[qi * head_dim..(qi + 1) * head_dim];

            // Full-dim baseline scores
            for (score, centroid) in full_scores
                .iter_mut()
                .zip(block_centroids.chunks_exact(head_dim))
            {
                *score = query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            argtopk_with_scratch(&full_scores, top_k, &mut full_indices, &mut topk_pairs);

            // Masked routing scores (only routing channels)
            for (score, centroid) in masked_scores
                .iter_mut()
                .zip(block_centroids.chunks_exact(head_dim))
            {
                let dot: f32 = routing_channels
                    .iter()
                    .map(|&d| query[d] * centroid[d])
                    .sum();
                *score = dot * scale;
            }
            argtopk_with_scratch(&masked_scores, top_k, &mut masked_indices, &mut topk_pairs);

            // Overlap fraction
            let overlap = masked_indices
                .iter()
                .filter(|idx| full_indices.contains(idx))
                .count() as f32;
            total_overlap += overlap / top_k as f32;
        }

        total_overlap / n_queries as f32
    }
}

impl Default for RoutingChannelDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ChannelAwareCache — T11
// ---------------------------------------------------------------------------

/// Cache for ChannelAwareRouter: compressed routing keys + full keys.
///
/// Stores routing-channel-only keys for fast decode routing and full keys
/// for actual attention computation. Memory overhead is ~25% additional
/// storage on centroids (routing_dim / head_dim).
#[derive(Debug, Clone)]
pub struct ChannelAwareCache {
    /// Routing-channel-only keys: flat `[n_blocks * routing_dim]`.
    pub routing_keys: Vec<f32>,
    /// Full keys for actual attention: flat `[n_blocks * head_dim]`.
    pub full_keys: Vec<f32>,
    /// Which channel indices are routing-critical.
    pub routing_channels: Vec<usize>,
    /// Number of blocks currently cached.
    pub n_blocks: usize,
    /// Full head dimension.
    pub head_dim: usize,
    /// Effective routing dimension (= routing_channels.len()).
    pub routing_dim: usize,
}

impl ChannelAwareCache {
    /// Create a pre-allocated cache.
    ///
    /// If `routing_channels` is empty, the cache falls back to full-dim routing.
    pub fn new(n_blocks_capacity: usize, head_dim: usize, routing_channels: Vec<usize>) -> Self {
        let routing_dim = routing_channels.len().max(1);
        Self {
            routing_keys: vec![0.0; n_blocks_capacity * routing_dim],
            full_keys: vec![0.0; n_blocks_capacity * head_dim],
            routing_channels,
            n_blocks: 0,
            head_dim,
            routing_dim,
        }
    }

    /// Ensure capacity for at least `n` blocks.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.routing_keys.len() < n * self.routing_dim {
            self.routing_keys.resize(n * self.routing_dim, 0.0);
        }
        if self.full_keys.len() < n * self.head_dim {
            self.full_keys.resize(n * self.head_dim, 0.0);
        }
    }

    /// Get routing key for block `i`.
    #[inline]
    pub fn routing_key(&self, i: usize) -> &[f32] {
        let start = i * self.routing_dim;
        &self.routing_keys[start..start + self.routing_dim]
    }

    /// Get full key for block `i`.
    #[inline]
    pub fn full_key(&self, i: usize) -> &[f32] {
        let start = i * self.head_dim;
        &self.full_keys[start..start + self.head_dim]
    }
}

// ---------------------------------------------------------------------------
// ChannelAwareRouter — T12
// ---------------------------------------------------------------------------

/// Channel-aware router: routes using only routing-critical channels.
///
/// When routing channels are discovered via `RoutingChannelDiscovery`,
/// this router computes dot products over only those channels,
/// achieving ~3-4x speedup on typical routing dimensions.
///
/// Falls back to `BlockTopKRouter` when no routing channels are configured.
#[derive(Debug)]
pub struct ChannelAwareRouter {
    /// Apply `1/sqrt(head_dim)` scaling to dot products.
    pub scale: bool,
    /// Fallback router for when no routing channels are set.
    #[allow(dead_code)]
    fallback: BlockTopKRouter,
}

impl ChannelAwareRouter {
    /// Create a new router with optional scaling.
    pub fn new(scale: bool) -> Self {
        Self {
            scale,
            fallback: BlockTopKRouter::new(scale),
        }
    }
}

impl VortexFlow for ChannelAwareRouter {
    type Cache = ChannelAwareCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        _values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        cache.ensure_capacity(block_idx + 1);
        let block_size = keys.len() / head_dim;

        // Compute mean centroid for full keys
        let start = block_idx * head_dim;
        let full_centroid = &mut cache.full_keys[start..start + head_dim];
        full_centroid.fill(0.0);

        if block_size == 0 {
            // Zero-length block: zero centroids
            if cache.routing_dim > 0 {
                let r_start = block_idx * cache.routing_dim;
                cache.routing_keys[r_start..r_start + cache.routing_dim].fill(0.0);
            }
        } else {
            // Mean pooling → full centroid. Use the crate SIMD kernels so the
            // accumulation and final 1/N scaling are vectorized on NEON/AVX2
            // instead of relying on the autovectorizer.
            for t in 0..block_size {
                let k_start = t * head_dim;
                katgpt_core::simd::simd_add_inplace(
                    full_centroid,
                    &keys[k_start..k_start + head_dim],
                );
            }
            let inv = 1.0 / block_size as f32;
            katgpt_core::simd::simd_scale_inplace(full_centroid, inv);

            // Extract routing channels from full centroid
            if !cache.routing_channels.is_empty() {
                let r_start = block_idx * cache.routing_dim;
                let routing_centroid =
                    &mut cache.routing_keys[r_start..r_start + cache.routing_dim];
                for (ri, &ch) in cache.routing_channels.iter().enumerate() {
                    routing_centroid[ri] = full_centroid[ch];
                }
            }
        }

        cache.n_blocks = cache.n_blocks.max(block_idx + 1);
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

        let hd = query.len();
        let scale = if self.scale {
            1.0 / (hd as f32).sqrt()
        } else {
            1.0
        };

        scratch.ensure_capacity(n_blocks);
        let scores = &mut scratch.scores[..n_blocks];

        if cache.routing_channels.is_empty() {
            for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
                let full_key = cache.full_key(i);
                *score = simd_dot_f32(query, full_key) * scale;
            }
        } else {
            let routing_channels = &cache.routing_channels;
            let routing_dim = cache.routing_dim;

            // Extract routing channels from query into scratch buffer
            scratch.routing_query_buf.resize(routing_dim, 0.0);
            for (ri, &ch) in routing_channels.iter().enumerate() {
                scratch.routing_query_buf[ri] = query[ch];
            }

            for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
                let routing_key = cache.routing_key(i);
                *score =
                    simd_dot_f32(&scratch.routing_query_buf[..routing_dim], routing_key) * scale;
            }
        }

        // Partial sort to find top-k
        let k = top_k.min(n_blocks);
        scratch.indices.clear();
        argtopk_with_scratch(scores, k, &mut scratch.indices, &mut scratch.argtopk_pairs);

        // Build routing decision with sigmoid weights
        let mut decision = RoutingDecision::with_capacity(k);
        for &idx in &scratch.indices[..k] {
            decision.blocks.push(idx);
            decision.weights.push(sigmoid(scores[idx]));
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        ChannelAwareCache::new(n_blocks_capacity, head_dim, Vec::new())
    }
}

// ---------------------------------------------------------------------------
// SIMD dot product — T13, now a DELEGATION (Issue 845)
// ---------------------------------------------------------------------------

/// SIMD dot product over the common prefix of `a` and `b`.
///
/// **Delegates to `katgpt_core::simd::simd_dot_f32`** — the shipped
/// ISA-dispatched kernel — and carries no kernel of its own. Kept as a
/// function rather than deleted because it is `pub` and takes a 2-argument
/// `(a, b)` form (length derived as `min(len)`) that this module's two call
/// sites and any external caller rely on; the shipped kernel takes an explicit
/// `len`.
///
/// # Why the ~200 lines of hand-written NEON + AVX2 that used to live here are
/// gone (Issue 845, Bench 845)
///
/// They were a same-named duplicate of a kernel already in this crate's
/// dependency graph — `katgpt-attn` depends on `katgpt-core`, which re-exports
/// `katgpt_types::simd` wholesale — and **seven other files in this crate,
/// four of them in this very `dash_attn/` directory, already call the shipped
/// one**. Two things were measured before the deletion, in both build
/// configurations, because the local AVX2 arm only existed in one of them:
///
/// | dot length | default build | `-C target-feature=+avx2` |
/// |---|---|---|
/// | 32 | local 2.72× slower | local 1.83× slower |
/// | 64 | 4.12× | 2.68× |
/// | 128 | 6.54× | 3.86× |
/// | 256 | 7.76× | 4.78× |
/// | 1024 | 6.37× | 4.00× |
///
/// The default-build column is the severe one and explains itself: the old
/// AVX2 arm was gated `#[cfg(target_feature = "avx2")]`, a **compile-time**
/// predicate that is OFF by default on x86_64, so the shipped path fell
/// through to a 4-accumulator scalar loop while the shipped kernel probed
/// CPUID once at runtime and used AVX2. Agreement was `<= 1e-6` there and
/// **bit-identical** under `+avx2`.
///
/// AGENTS.md cites this file for 30 `unsafe_op_in_unsafe_fn` findings and
/// blames *"two transcriptions of one kernel"*. Both transcriptions were of a
/// kernel that was already in the graph; removing them takes this file's
/// `unsafe` surface to zero.
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    katgpt_core::simd::simd_dot_f32(a, b, n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 8;

    #[test]
    fn test_routing_channel_mask_all() {
        let mask = RoutingChannelMask::all(HEAD_DIM);
        assert_eq!(mask.routing_dim(), HEAD_DIM);
        assert!(mask.has_routing_channels());
    }

    #[test]
    fn test_routing_channel_mask_none() {
        let mask = RoutingChannelMask::none(HEAD_DIM);
        assert_eq!(mask.routing_dim(), 0);
        assert!(!mask.has_routing_channels());
    }

    #[test]
    fn test_routing_channel_mask_partial() {
        let mut channels = vec![false; HEAD_DIM];
        channels[0] = true;
        channels[3] = true;
        channels[7] = true;
        let mask = RoutingChannelMask { channels };
        assert_eq!(mask.routing_dim(), 3);
        assert_eq!(mask.routing_channels(), vec![0, 3, 7]);
    }

    #[test]
    fn test_discovery_finds_critical_channels() {
        let discovery = RoutingChannelDiscovery::new();

        // Create synthetic data where dimension 0 is strongly routing-critical
        let n_blocks = 8;
        let mut centroids = vec![0.0f32; n_blocks * HEAD_DIM];
        for bi in 0..n_blocks {
            centroids[bi * HEAD_DIM] = (bi as f32 + 1.0) * 2.0; // dim 0 varies strongly
            centroids[bi * HEAD_DIM + 1] = 0.1; // dim 1 constant
            for d in 2..HEAD_DIM {
                centroids[bi * HEAD_DIM + d] = 0.01; // other dims near-zero
            }
        }

        // Queries aligned with dim 0
        let n_queries = 4;
        let mut queries = vec![0.0f32; n_queries * HEAD_DIM];
        for qi in 0..n_queries {
            queries[qi * HEAD_DIM] = (qi as f32 + 1.0) * 1.5;
        }

        let mask = discovery.calibrate(HEAD_DIM, &centroids, &queries, 3);

        // Dim 0 should be critical (masking it would hurt accuracy a lot)
        assert!(
            mask.channels[0],
            "dimension 0 should be routing-critical: mask = {:?}",
            mask.channels
        );
    }

    #[test]
    fn test_channel_aware_cache_new() {
        let cache = ChannelAwareCache::new(4, HEAD_DIM, vec![0, 2, 4, 6]);
        assert_eq!(cache.routing_dim, 4);
        assert_eq!(cache.routing_keys.len(), 16); // 4 blocks × 4 routing dim
        assert_eq!(cache.full_keys.len(), 32); // 4 blocks × 8 head dim
    }

    #[test]
    fn test_channel_aware_cache_ensure_capacity() {
        let mut cache = ChannelAwareCache::new(2, HEAD_DIM, vec![0, 3]);
        cache.ensure_capacity(8);
        assert!(cache.routing_keys.len() >= 16); // 8 × 2
        assert!(cache.full_keys.len() >= 64); // 8 × 8
    }

    #[test]
    fn test_channel_aware_router_forward_cache_and_indexer() {
        let router = ChannelAwareRouter::new(true);
        let routing_channels = vec![0, 1]; // Route on first 2 dimensions
        let mut cache = ChannelAwareCache::new(4, HEAD_DIM, routing_channels);
        let mut scratch = VortexScratch::new(4);

        // Block 0: centroid aligned with [1,0,...] in routing channels
        let keys0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let vals0 = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        // Block 1: centroid aligned with [0,1,...] in routing channels
        let keys1 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let vals1 = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        // Query aligned with block 0 in routing channels
        let query = vec![1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);

        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(
            decision.blocks[0], 0,
            "should select block 0 (aligned in routing channels)"
        );
    }

    #[test]
    fn test_channel_aware_router_no_routing_channels_fallback() {
        let router = ChannelAwareRouter::new(true);
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        // Without routing channels set, it should still work (full-dim fallback)
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);
        assert_eq!(decision.blocks[0], 0);
    }

    #[test]
    fn test_simd_dot_matches_scalar() {
        let a: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let simd_result = simd_dot_f32(&a, &b);
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "SIMD dot ({simd_result}) != scalar dot ({scalar_result})"
        );
    }

    #[test]
    fn test_simd_dot_small_vectors() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];

        let result = simd_dot_f32(&a, &b);
        let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_simd_dot_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_eq!(simd_dot_f32(&a, &b), 0.0);
    }

    #[test]
    fn test_channel_aware_empty_blocks() {
        let router = ChannelAwareRouter::new(true);
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_channel_aware_topk_capped() {
        let router = ChannelAwareRouter::new(true);
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys, &vals, 1, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 10, &mut scratch);
        assert_eq!(decision.blocks.len(), 2);
    }
}
