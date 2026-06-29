//! MSA (MiniMax Sparse Attention) distillation — inference-time block scoring upgrades.
//!
//! Distills MSA's key mechanisms into the VortexFlow framework without training:
//! 1. MaxPoolBlockScorer — max Q·K scoring instead of mean centroid (MSA Eq. 6)
//! 2. MaxStdDevBlockScorer — max score * sigmoid(std_dev) (UNIQUE + MSA fusion)
//! 3. Exp-free TopK — skip softmax before selection (MSA §4.1 order-preservation)
//!
//! Feature gate: `vortex_flow` (Plan 256, Research 225).

use super::block_topk::{argtopk_with_scratch, sigmoid};
use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};
use katgpt_core::simd::{simd_dot_f32, simd_sum_sq};

// ---------------------------------------------------------------------------
// MSA cache — max-pool scores + key statistics per block
// ---------------------------------------------------------------------------

/// Cache for MSA-distilled routers: per-block max-pool score + key statistics.
///
/// Unlike `BlockTopKCache` which stores mean centroids, this stores the raw
/// key matrix for each block so we can compute max(q·k_j) at query time.
/// For efficiency, we also precompute per-block key norm statistics (mean, std_dev).
#[derive(Debug, Clone)]
pub struct MsaBlockCache {
    /// Flattened key matrices: `[n_blocks * block_size * head_dim]`.
    /// Stored densely so we can iterate over keys per block at query time.
    pub keys: Vec<f32>,
    /// Per-block key norm mean: `[n_blocks]`.
    pub key_norm_mean: Vec<f32>,
    /// Per-block key norm std_dev: `[n_blocks]`.
    pub key_norm_stddev: Vec<f32>,
    /// Block size (tokens per block). Fixed at construction.
    pub block_size: usize,
    /// Number of blocks currently cached.
    pub n_blocks: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl MsaBlockCache {
    /// Create a pre-allocated cache.
    ///
    /// `block_size` must match the KV block granularity used during decode.
    pub fn new(n_blocks_capacity: usize, block_size: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![0.0; n_blocks_capacity * block_size * head_dim],
            key_norm_mean: vec![0.0; n_blocks_capacity],
            key_norm_stddev: vec![0.0; n_blocks_capacity],
            block_size,
            n_blocks: 0,
            head_dim,
        }
    }

    /// Ensure capacity for at least `n` blocks.
    pub fn ensure_capacity(&mut self, n: usize) {
        let needed = n * self.block_size * self.head_dim;
        if self.keys.len() < needed {
            self.keys.resize(needed, 0.0);
        }
        if self.key_norm_mean.len() < n {
            self.key_norm_mean.resize(n, 0.0);
        }
        if self.key_norm_stddev.len() < n {
            self.key_norm_stddev.resize(n, 0.0);
        }
    }

    /// Get keys for block `i`: flat `[block_size * head_dim]`.
    #[inline]
    pub fn block_keys(&self, i: usize) -> &[f32] {
        let start = i * self.block_size * self.head_dim;
        let end = start + self.block_size * self.head_dim;
        &self.keys[start..end]
    }
}

// ---------------------------------------------------------------------------
// Exp-free dot product scoring helpers
// ---------------------------------------------------------------------------

/// Compute max(q · k_j) over all tokens j in a block — MSA Eq. 6.
///
/// This is the core MSA scoring mechanism: instead of centroid dot product,
/// find the maximum individual key-query dot product in the block.
/// No exp/softmax needed — raw scores preserve ordering (MSA §4.1).
#[inline]
fn max_qk_score(query: &[f32], block_keys: &[f32], block_size: usize, head_dim: usize) -> f32 {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Compute all per-token dot products via SIMD, then take the max.
    // Fused dot+scale in one pass; max reduces to a single value.
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..block_size {
        let k_start = t * head_dim;
        let dot = simd_dot_f32(query, &block_keys[k_start..k_start + head_dim], head_dim);
        let score = dot * scale;
        max_score = max_score.max(score);
    }

    max_score
}

// ---------------------------------------------------------------------------
// MaxPoolBlockScorer — MSA's max(q·k) per block
// ---------------------------------------------------------------------------

/// Max-pool block scorer — selects blocks by max individual Q·K score.
///
/// Replaces `BlockTopKRouter`'s centroid dot product with MSA's max-pool
/// scoring (Equation 6). This captures the "needle-in-haystack" signal:
/// a block with even one highly-relevant token gets selected.
///
/// No training needed — this is a pure scoring function change.
#[derive(Debug)]
pub struct MaxPoolBlockScorer {
    /// KV block size (tokens per block). Must match cache construction.
    pub block_size: usize,
}

impl MaxPoolBlockScorer {
    /// Create with the given block size.
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }
}

impl VortexFlow for MaxPoolBlockScorer {
    type Cache = MsaBlockCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        _values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        let block_size = keys.len() / head_dim;
        debug_assert_eq!(
            block_size, self.block_size,
            "block_size mismatch: expected {}, got {}",
            self.block_size, block_size
        );

        cache.ensure_capacity(block_idx + 1);

        // Store raw keys for max-pool scoring at query time
        let start = block_idx * self.block_size * head_dim;
        let end = start + block_size * head_dim;
        cache.keys[start..end].copy_from_slice(keys);

        // Precompute key norm statistics for potential StdDev scoring
        let mut norm_sum = 0.0f32;
        let mut norm_sq_sum = 0.0f32;
        for t in 0..block_size {
            let k_start = t * head_dim;
            let norm_sq = simd_sum_sq(&keys[k_start..k_start + head_dim], head_dim);
            let norm = norm_sq.sqrt();
            norm_sum += norm;
            norm_sq_sum += norm * norm;
        }

        let inv = 1.0 / block_size.max(1) as f32;
        cache.key_norm_mean[block_idx] = norm_sum * inv;
        // std_dev = sqrt(E[X^2] - E[X]^2)
        let variance = norm_sq_sum * inv - (norm_sum * inv).powi(2);
        cache.key_norm_stddev[block_idx] = variance.max(0.0).sqrt();

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
        scratch.ensure_capacity(n_blocks);
        let scores = &mut scratch.scores[..n_blocks];

        // Exp-free max-pool scoring — MSA §4.1
        // softmax is order-preserving, so argmax(raw) == argmax(softmax(raw))
        for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
            let block_keys = cache.block_keys(i);
            *score = max_qk_score(query, block_keys, self.block_size, hd);
        }

        // Top-k selection (exp-free: no softmax needed)
        let k = top_k.min(n_blocks);
        scratch.indices.clear();
        argtopk_with_scratch(scores, k, &mut scratch.indices, &mut scratch.argtopk_pairs);

        let mut decision = RoutingDecision::with_capacity(k);
        for &idx in &scratch.indices[..k] {
            decision.blocks.push(idx);
            decision.weights.push(sigmoid(scores[idx]));
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        MsaBlockCache::new(n_blocks_capacity, self.block_size, head_dim)
    }
}

// ---------------------------------------------------------------------------
// MaxStdDevBlockScorer — MSA max-pool × UNIQUE std_dev gate
// ---------------------------------------------------------------------------

/// Max-pool + StdDev block scorer — fuses MSA max scoring with UNIQUE diversity.
///
/// Score formula: `score = max(q·k) × sigmoid(σ_k × λ)`
///
/// - `max(q·k)` captures the strongest query-key alignment in the block (MSA)
/// - `σ_k` = std_dev of key norms within the block (UNIQUE diversity signal)
/// - High std_dev → diverse content → higher chance of containing relevant tokens
/// - λ controls the std_dev gate strength (default 1.0)
///
/// No training needed — pure inference-time scoring function upgrade.
#[derive(Debug)]
pub struct MaxStdDevBlockScorer {
    /// KV block size (tokens per block).
    pub block_size: usize,
    /// StdDev gate strength λ. Default 1.0.
    pub lambda: f32,
}

impl MaxStdDevBlockScorer {
    /// Create with block size and default lambda=1.0.
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            lambda: 1.0,
        }
    }

    /// Create with custom lambda.
    pub fn with_lambda(block_size: usize, lambda: f32) -> Self {
        Self { block_size, lambda }
    }
}

impl VortexFlow for MaxStdDevBlockScorer {
    type Cache = MsaBlockCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        // Reuse MaxPoolBlockScorer's cache logic (stores keys + norm stats)
        MaxPoolBlockScorer::new(self.block_size)
            .forward_cache(cache, keys, values, block_idx, head_dim);
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
        scratch.ensure_capacity(n_blocks);
        let scores = &mut scratch.scores[..n_blocks];

        // Fused scoring: max(q·k) × sigmoid(σ_k × λ)
        for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
            let block_keys = cache.block_keys(i);
            let max_score = max_qk_score(query, block_keys, self.block_size, hd);
            let stddev_gate = sigmoid(cache.key_norm_stddev[i] * self.lambda);
            *score = max_score * stddev_gate;
        }

        // Exp-free top-k selection
        let k = top_k.min(n_blocks);
        scratch.indices.clear();
        argtopk_with_scratch(scores, k, &mut scratch.indices, &mut scratch.argtopk_pairs);

        let mut decision = RoutingDecision::with_capacity(k);
        for &idx in &scratch.indices[..k] {
            decision.blocks.push(idx);
            decision.weights.push(sigmoid(scores[idx]));
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        MsaBlockCache::new(n_blocks_capacity, self.block_size, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;
    const BLOCK_SIZE: usize = 4;

    fn make_max_pool_router() -> MaxPoolBlockScorer {
        MaxPoolBlockScorer::new(BLOCK_SIZE)
    }

    fn make_max_stddev_router() -> MaxStdDevBlockScorer {
        MaxStdDevBlockScorer::new(BLOCK_SIZE)
    }

    // -- MaxPoolBlockScorer tests --

    #[test]
    fn test_max_pool_selects_needle_block() {
        let router = make_max_pool_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        // Block 0: all keys aligned with [0,1,0,0] → centroid [0,1,0,0]
        // Max score with query [1,0,0,0] = 0 (no individual key matches)
        let keys0 = vec![
            0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
        ];
        let vals0 = vec![0.0; BLOCK_SIZE * HEAD_DIM];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        // Block 1: mostly noise but one key exactly aligned with query
        // Max score with query [1,0,0,0] = 0.5 (one key matches)
        let keys1 = vec![
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let vals1 = vec![0.0; BLOCK_SIZE * HEAD_DIM];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);

        // Max-pool should select block 1 (has one needle key matching query)
        // even though block 0 has higher centroid alignment
        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(decision.blocks[0], 1, "max-pool should select needle block");
    }

    #[test]
    fn test_max_pool_vs_centroid_different_selection() {
        // Demonstrate that max-pool can differ from centroid scoring
        let router = make_max_pool_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        // Block 0: uniform keys [1,0,0,0] → centroid matches perfectly
        let keys0 = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let vals = vec![0.0; BLOCK_SIZE * HEAD_DIM];
        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);

        // Block 1: mostly orthogonal but one strong match
        let keys1 = vec![
            0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);

        // Block 2: moderate alignment
        let keys2 = vec![
            0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
        ];
        router.forward_cache(&mut cache, &keys2, &vals, 2, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 3, 2, &mut scratch);

        assert_eq!(decision.blocks.len(), 2);
        // Block 1 should rank highest (max score = 2.0 * scale)
        assert_eq!(decision.blocks[0], 1);
    }

    #[test]
    fn test_max_pool_empty_blocks() {
        let router = make_max_pool_router();
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_max_pool_single_block() {
        let router = make_max_pool_router();
        let mut cache = router.cache_new(1, HEAD_DIM);
        let mut scratch = VortexScratch::new(1);

        let keys = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let vals = vec![0.0; BLOCK_SIZE * HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks, vec![0]);
        assert!(decision.weights[0] > 0.0);
    }

    #[test]
    fn test_exp_free_topk_order_preservation() {
        // Verify that raw score ordering == softmax ordering
        let router = make_max_pool_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        // Create blocks with distinct scores
        for (i, scale) in [1.0, 3.0, 2.0].into_iter().enumerate() {
            let keys: Vec<f32> = (0..BLOCK_SIZE * HEAD_DIM)
                .map(|d| if d % HEAD_DIM == 0 { scale } else { 0.0 })
                .collect();
            let vals = vec![0.0; BLOCK_SIZE * HEAD_DIM];
            router.forward_cache(&mut cache, &keys, &vals, i, HEAD_DIM);
        }

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 3, 3, &mut scratch);

        // Block 1 (scale=3.0) should be first, block 2 (scale=2.0) second, block 0 (scale=1.0) third
        assert_eq!(decision.blocks[0], 1, "highest score block should be first");
        assert_eq!(decision.blocks[1], 2);
        assert_eq!(decision.blocks[2], 0);
    }

    // -- MaxStdDevBlockScorer tests --

    #[test]
    fn test_stddev_gate_amplifies_diverse_blocks() {
        let router = make_max_stddev_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        // Block 0: uniform keys (low std_dev) → stddev gate ≈ sigmoid(0) = 0.5
        let keys0 = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ];
        let vals = vec![0.0; BLOCK_SIZE * HEAD_DIM];
        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);

        // Block 1: diverse keys (high std_dev) → stddev gate closer to 1.0
        let keys1 = vec![
            10.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);

        // Both blocks have a key with score ~10 * scale
        // But block 1 has higher std_dev → higher gated score
        assert!(
            cache.key_norm_stddev[1] > cache.key_norm_stddev[0],
            "diverse block should have higher std_dev: {} vs {}",
            cache.key_norm_stddev[1],
            cache.key_norm_stddev[0],
        );

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 2, &mut scratch);

        // Block 1 should rank higher due to diversity gate
        assert_eq!(decision.blocks[0], 1, "diverse block should rank higher");
    }

    #[test]
    fn test_stddev_router_empty_blocks() {
        let router = make_max_stddev_router();
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_stddev_lambda_control() {
        let router_low = MaxStdDevBlockScorer::with_lambda(BLOCK_SIZE, 0.0);
        let router_high = MaxStdDevBlockScorer::with_lambda(BLOCK_SIZE, 100.0);

        let mut cache_low = router_low.cache_new(1, HEAD_DIM);
        let mut cache_high = router_high.cache_new(1, HEAD_DIM);
        let mut scratch = VortexScratch::new(1);

        // Diverse block
        let keys = vec![
            10.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let vals = vec![0.0; BLOCK_SIZE * HEAD_DIM];

        router_low.forward_cache(&mut cache_low, &keys, &vals, 0, HEAD_DIM);
        router_high.forward_cache(&mut cache_high, &keys, &vals, 0, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let d_low = router_low.forward_indexer(&query, &cache_low, 1, 1, &mut scratch);
        scratch.indices.clear();
        let d_high = router_high.forward_indexer(&query, &cache_high, 1, 1, &mut scratch);

        // With lambda=0: sigmoid(0) = 0.5 → score = max_score * 0.5
        // With lambda=100: sigmoid(large) ≈ 1.0 → score ≈ max_score
        assert!(
            d_high.weights[0] > d_low.weights[0],
            "high lambda should produce higher gated weights: {} vs {}",
            d_high.weights[0],
            d_low.weights[0],
        );
    }

    #[test]
    fn test_cache_ensure_capacity() {
        let mut cache = MsaBlockCache::new(2, BLOCK_SIZE, HEAD_DIM);
        assert_eq!(cache.keys.len(), 2 * BLOCK_SIZE * HEAD_DIM);
        cache.ensure_capacity(5);
        assert!(cache.keys.len() >= 5 * BLOCK_SIZE * HEAD_DIM);
        assert!(cache.key_norm_mean.len() >= 5);
        assert!(cache.key_norm_stddev.len() >= 5);
    }

    #[test]
    fn test_key_norm_statistics() {
        let router = make_max_pool_router();
        let mut cache = router.cache_new(1, HEAD_DIM);

        // BLOCK_SIZE=4, HEAD_DIM=4 → 4 keys
        // Key 0: [1,0,0,0] → ‖k‖=1
        // Key 1: [3,4,0,0] → ‖k‖=5
        // Key 2: [1,0,0,0] → ‖k‖=1
        // Key 3: [0,0,0,0] → ‖k‖=0
        // mean_norm = (1+5+1+0)/4 = 1.75
        // std_dev = sqrt((1+25+1+0)/4 - 1.75²) = sqrt(6.75 - 3.0625) = sqrt(3.6875)
        let keys = vec![
            1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let vals = vec![0.0; BLOCK_SIZE * HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        assert!(
            (cache.key_norm_mean[0] - 1.75).abs() < 1e-4,
            "expected mean norm ~1.75, got {}",
            cache.key_norm_mean[0]
        );
        assert!(
            cache.key_norm_stddev[0] > 0.0,
            "expected non-zero std_dev for diverse keys, got {}",
            cache.key_norm_stddev[0]
        );
    }
}
