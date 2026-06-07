//! BlockTopK router — simplest VortexFlow implementation.
//!
//! Routes via mean key centroids + dot-product top-k selection.
//! Feature gate: `vortex_flow` (Plan 196, Phase 1).

use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// BlockTopKCache
// ---------------------------------------------------------------------------

/// Cache for BlockTopK router: mean key centroid per block.
#[derive(Debug, Clone)]
pub struct BlockTopKCache {
    /// Centroid vectors: flat `[n_blocks * head_dim]`.
    pub centroids: Vec<f32>,
    /// Number of blocks currently cached.
    pub n_blocks: usize,
    /// Dimension of each centroid.
    pub head_dim: usize,
}

impl BlockTopKCache {
    /// Create a pre-allocated cache.
    pub fn new(n_blocks_capacity: usize, head_dim: usize) -> Self {
        Self {
            centroids: vec![0.0; n_blocks_capacity * head_dim],
            n_blocks: 0,
            head_dim,
        }
    }

    /// Ensure the cache can hold at least `n` blocks.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.centroids.len() < n * self.head_dim {
            self.centroids.resize(n * self.head_dim, 0.0);
        }
    }

    /// Get centroid for block `i`.
    #[inline]
    pub fn centroid(&self, i: usize) -> &[f32] {
        let start = i * self.head_dim;
        &self.centroids[start..start + self.head_dim]
    }
}

// ---------------------------------------------------------------------------
// BlockTopKRouter
// ---------------------------------------------------------------------------

/// BlockTopK router — routes via mean key centroids + dot product top-k.
#[derive(Debug)]
pub struct BlockTopKRouter {
    /// Apply `1/sqrt(head_dim)` scaling to dot products.
    pub scale: bool,
}

impl BlockTopKRouter {
    /// Create a new router with optional scaling.
    pub fn new(scale: bool) -> Self {
        Self { scale }
    }
}

impl VortexFlow for BlockTopKRouter {
    type Cache = BlockTopKCache;

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
        if block_size == 0 {
            // Zero-length block: zero centroid
            let start = block_idx * head_dim;
            cache.centroids[start..start + head_dim].fill(0.0);
            cache.n_blocks = cache.n_blocks.max(block_idx + 1);
            return;
        }

        // Compute mean of keys → centroid
        let start = block_idx * head_dim;
        let centroid = &mut cache.centroids[start..start + head_dim];
        centroid.fill(0.0);
        for t in 0..block_size {
            let k_start = t * head_dim;
            for d in 0..head_dim {
                centroid[d] += keys[k_start + d];
            }
        }
        let inv = 1.0 / block_size as f32;
        for d in centroid.iter_mut() {
            *d *= inv;
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

        // Compute dot(query, centroid[i]) for each block
        for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
            let centroid = cache.centroid(i);
            let dot: f32 = query.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
            *score = dot * scale;
        }

        // Partial sort to find top-k
        let k = top_k.min(n_blocks);
        scratch.indices.clear();
        argtopk(scores, k, &mut scratch.indices);

        // Build routing decision with sigmoid weights
        let mut decision = RoutingDecision::with_capacity(k);
        for &idx in &scratch.indices[..k] {
            decision.blocks.push(idx);
            // Sigmoid normalization per-block score
            let w = sigmoid(scores[idx]);
            decision.weights.push(w);
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        BlockTopKCache::new(n_blocks_capacity, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find top-k indices from scores (partial sort, descending).
/// Uses a simple selection-based approach — O(n*k) which is fine for small k.
pub fn argtopk(scores: &[f32], k: usize, indices: &mut Vec<usize>) {
    indices.clear();
    let n = scores.len();
    let k = k.min(n);
    if k == 0 {
        return;
    }

    // Build (index, score) pairs and partial sort
    let mut pairs: Vec<(usize, f32)> = (0..n).map(|i| (i, scores[i])).collect();

    // Selection sort the top-k (in-place, O(n*k))
    for i in 0..k {
        // Find max in remaining unsorted portion
        let mut best = i;
        for j in (i + 1)..n {
            if pairs[j].1 > pairs[best].1 {
                best = j;
            }
        }
        pairs.swap(i, best);
    }

    indices.extend(pairs[..k].iter().map(|(idx, _)| *idx));
}

/// Standard sigmoid function.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;

    fn make_router() -> BlockTopKRouter {
        BlockTopKRouter::new(true)
    }

    #[test]
    fn test_known_keys_known_centroids_known_topk() {
        let router = make_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        // Block 0: keys = [[1,0,0,0], [1,0,0,0]] → centroid [1,0,0,0]
        let keys0 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        // Block 1: keys = [[0,1,0,0], [0,1,0,0]] → centroid [0,1,0,0]
        let keys1 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        // Block 2: keys = [[0,0,1,0], [0,0,1,0]] → centroid [0,0,1,0]
        let keys2 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let vals2 = vec![0.0; 8];
        router.forward_cache(&mut cache, &keys2, &vals2, 2, HEAD_DIM);

        // Query aligned with block 1 → should select block 1 first
        let query = vec![0.0, 1.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 3, 1, &mut scratch);
        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(decision.blocks[0], 1);
    }

    #[test]
    fn test_single_block_always_selected() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);
        let mut scratch = VortexScratch::new(1);

        let keys = vec![1.0, 2.0, 3.0, 4.0];
        let vals = vec![5.0, 6.0, 7.0, 8.0];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let query = vec![0.1, 0.2, 0.3, 0.4];
        let decision = router.forward_indexer(&query, &cache, 1, 1, &mut scratch);
        assert_eq!(decision.blocks, vec![0]);
    }

    #[test]
    fn test_zero_query_all_scores_equal() {
        let router = make_router();
        let mut cache = router.cache_new(3, HEAD_DIM);
        let mut scratch = VortexScratch::new(3);

        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let keys2 = vec![0.0, 0.0, 1.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];

        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);
        router.forward_cache(&mut cache, &keys2, &vals, 2, HEAD_DIM);

        // Zero query → all dot products = 0
        let query = vec![0.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 3, 2, &mut scratch);
        assert_eq!(decision.blocks.len(), 2);
        // All scores are 0.0, any 2 of 3 blocks selected is valid
        assert!(decision.blocks.iter().all(|&b| b < 3));
    }

    #[test]
    fn test_empty_blocks_returns_empty() {
        let router = make_router();
        let cache = router.cache_new(0, HEAD_DIM);
        let mut scratch = VortexScratch::new(0);

        let query = vec![1.0; HEAD_DIM];
        let decision = router.forward_indexer(&query, &cache, 0, 4, &mut scratch);
        assert!(decision.is_empty());
    }

    #[test]
    fn test_topk_capped_at_n_blocks() {
        let router = make_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys, &vals, 1, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        // top_k=10 but only 2 blocks
        let decision = router.forward_indexer(&query, &cache, 2, 10, &mut scratch);
        assert_eq!(decision.blocks.len(), 2);
    }

    #[test]
    fn test_cache_ensure_capacity_grows() {
        let mut cache = BlockTopKCache::new(2, HEAD_DIM);
        assert_eq!(cache.centroids.len(), 8);
        cache.ensure_capacity(5);
        assert!(cache.centroids.len() >= 20);
    }

    #[test]
    fn test_centroid_mean_pooling() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);

        // 3 tokens: [1,0,0,0], [0,2,0,0], [0,0,3,0] → mean = [1/3, 2/3, 1, 0]
        let keys = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0];
        let vals = vec![0.0; 12];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        let c = cache.centroid(0);
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        for (i, (&got, &exp)) in c.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "centroid mismatch at dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_argtopk_descending() {
        let scores = [0.5, 2.0, 0.1, 3.0, 1.5];
        let mut indices = Vec::new();
        argtopk(&scores, 3, &mut indices);
        assert_eq!(indices.len(), 3);
        // Descending order by score
        assert_eq!(indices[0], 3); // score 3.0
        assert_eq!(indices[1], 1); // score 2.0
        assert_eq!(indices[2], 4); // score 1.5
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }
}
