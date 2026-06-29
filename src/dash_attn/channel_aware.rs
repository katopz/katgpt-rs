//! Channel-aware routing with SIMD-optimized dot products (Plan 196, Phase 2).
//!
//! Discovers routing-critical channel groups via ablation calibration, then
//! builds a compressed routing index over only those channels for faster
//! decode-time block selection.
//!
//! Feature gate: `vortex_flow` (Plan 196, Phase 2, default-OFF).

use super::block_topk::{BlockTopKRouter, argtopk, argtopk_with_scratch, sigmoid};
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
            for d in g_start..g_end {
                mask.channels[d] = false;
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
        match has_critical {
            true => RoutingChannelMask { channels: critical },
            false => RoutingChannelMask::all(head_dim),
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

        for qi in 0..n_queries {
            let query = &queries[qi * head_dim..(qi + 1) * head_dim];

            // Full-dim baseline scores
            let mut full_scores = vec![0.0f32; n_blocks];
            for bi in 0..n_blocks {
                let centroid = &block_centroids[bi * head_dim..(bi + 1) * head_dim];
                full_scores[bi] = query
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            let mut full_indices = Vec::new();
            argtopk(&full_scores, top_k, &mut full_indices);
            let full_set: Vec<usize> = full_indices;

            // Masked routing scores (only routing channels)
            let mut masked_scores = vec![0.0f32; n_blocks];
            for bi in 0..n_blocks {
                let centroid = &block_centroids[bi * head_dim..(bi + 1) * head_dim];
                let dot: f32 = routing_channels
                    .iter()
                    .map(|&d| query[d] * centroid[d])
                    .sum();
                masked_scores[bi] = dot * scale;
            }
            let mut masked_indices = Vec::new();
            argtopk(&masked_scores, top_k, &mut masked_indices);

            // Overlap fraction
            let overlap = masked_indices
                .iter()
                .filter(|idx| full_set.contains(idx))
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

        match block_size {
            0 => {
                // Zero-length block: zero centroids
                if cache.routing_dim > 0 {
                    let r_start = block_idx * cache.routing_dim;
                    cache.routing_keys[r_start..r_start + cache.routing_dim].fill(0.0);
                }
            }
            _ => {
                // Mean pooling → full centroid. Use the crate SIMD kernels so the
                // accumulation and final 1/N scaling are vectorized on NEON/AVX2
                // instead of relying on the autovectorizer.
                for t in 0..block_size {
                    let k_start = t * head_dim;
                    katgpt_core::simd::simd_add_inplace(full_centroid, &keys[k_start..k_start + head_dim]);
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
        let scale = match self.scale {
            true => 1.0 / (hd as f32).sqrt(),
            false => 1.0,
        };

        scratch.ensure_capacity(n_blocks);
        let scores = &mut scratch.scores[..n_blocks];

        match cache.routing_channels.is_empty() {
            // Fallback: full-dim routing using full_keys
            true => {
                for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
                    let full_key = cache.full_key(i);
                    *score = simd_dot_f32(query, full_key) * scale;
                }
            }
            // Channel-aware: route using only critical channels
            false => {
                let routing_channels = &cache.routing_channels;
                let routing_dim = cache.routing_dim;

                // Extract routing channels from query into scratch buffer
                scratch.routing_query_buf.resize(routing_dim, 0.0);
                for (ri, &ch) in routing_channels.iter().enumerate() {
                    scratch.routing_query_buf[ri] = query[ch];
                }

                for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
                    let routing_key = cache.routing_key(i);
                    *score = simd_dot_f32(&scratch.routing_query_buf[..routing_dim], routing_key)
                        * scale;
                }
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
// SIMD dot product — T13
// ---------------------------------------------------------------------------

/// SIMD-optimized dot product with NEON/AVX2/scalar fallback.
///
/// For routing_dim=32: 4 NEON vectors (8 f32 each) → 4 accumulators.
/// For routing_dim=64: 8 NEON vectors → 8 accumulators.
/// Falls back to scalar for non-SIMD architectures.
#[inline]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());

    // SAFETY: we only access elements within bounds
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { simd_dot_neon(a.as_ptr(), b.as_ptr(), n) }
    }

    #[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
    {
        #[cfg(target_feature = "avx2")]
        {
            unsafe { simd_dot_avx2(a.as_ptr(), b.as_ptr(), n) }
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            simd_dot_scalar(a, b, n)
        }
    }

    #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
    {
        simd_dot_scalar(a, b, n)
    }
}

#[allow(dead_code)]
#[inline]
fn simd_dot_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    // Chunked loop for auto-vectorization (4-wide)
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        sum0 += a[base] * b[base];
        sum1 += a[base + 1] * b[base + 1];
        sum2 += a[base + 2] * b[base + 2];
        sum3 += a[base + 3] * b[base + 3];
    }

    let mut sum = sum0 + sum1 + sum2 + sum3;
    for i in (n - remainder)..n {
        sum += a[i] * b[i];
    }
    sum
}

/// NEON-optimized dot product (AArch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn simd_dot_neon(a: *const f32, b: *const f32, n: usize) -> f32 {
    use std::arch::aarch64::*;

    // SAFETY: Caller guarantees valid pointers with at least `n` elements.
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        // Process 16 elements per iteration (4 NEON vectors × 4 f32 each)
        let chunks16 = n / 16;
        let mut i = 0usize;

        for _ in 0..chunks16 {
            let v0 = vld1q_f32(a.add(i));
            let v1 = vld1q_f32(b.add(i));
            acc0 = vmlaq_f32(acc0, v0, v1);

            let v2 = vld1q_f32(a.add(i + 4));
            let v3 = vld1q_f32(b.add(i + 4));
            acc1 = vmlaq_f32(acc1, v2, v3);

            let v4 = vld1q_f32(a.add(i + 8));
            let v5 = vld1q_f32(b.add(i + 8));
            acc2 = vmlaq_f32(acc2, v4, v5);

            let v6 = vld1q_f32(a.add(i + 12));
            let v7 = vld1q_f32(b.add(i + 12));
            acc3 = vmlaq_f32(acc3, v6, v7);

            i += 16;
        }

        // Process remaining 4-element chunks
        while i + 4 <= n {
            let va = vld1q_f32(a.add(i));
            let vb = vld1q_f32(b.add(i));
            acc0 = vmlaq_f32(acc0, va, vb);
            i += 4;
        }

        // Horizontal add all accumulators
        let sum_vec = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut result = [0.0f32; 4];
        vst1q_f32(result.as_mut_ptr(), sum_vec);
        let mut sum = result[0] + result[1] + result[2] + result[3];

        // Handle remaining elements
        while i < n {
            sum += *a.add(i) * *b.add(i);
            i += 1;
        }

        sum
    }
}

/// AVX2-optimized dot product (x86_64).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn simd_dot_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    // Process 32 elements per iteration (4 AVX2 vectors × 8 f32)
    let chunks32 = n / 32;
    let mut i = 0usize;

    for _ in 0..chunks32 {
        let v0 = _mm256_loadu_ps(a.add(i));
        let v1 = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_fmadd_ps(v0, v1, acc0);

        let v2 = _mm256_loadu_ps(a.add(i + 8));
        let v3 = _mm256_loadu_ps(b.add(i + 8));
        acc1 = _mm256_fmadd_ps(v2, v3, acc1);

        let v4 = _mm256_loadu_ps(a.add(i + 16));
        let v5 = _mm256_loadu_ps(b.add(i + 16));
        acc2 = _mm256_fmadd_ps(v4, v5, acc2);

        let v6 = _mm256_loadu_ps(a.add(i + 24));
        let v7 = _mm256_loadu_ps(b.add(i + 24));
        acc3 = _mm256_fmadd_ps(v6, v7, acc3);

        i += 32;
    }

    // Process remaining 8-element chunks
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }

    // Horizontal add
    let sum256 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let hi = _mm256_extractf128_ps(sum256, 1);
    let lo = _mm256_castps256_ps128(sum256);
    let sum128 = _mm_add_ps(hi, lo);
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum128);
    let mut sum = (result[0] + result[1]) + (result[2] + result[3]);

    // Handle remaining elements
    while i < n {
        sum += *a.add(i) * *b.add(i);
        i += 1;
    }

    sum
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
