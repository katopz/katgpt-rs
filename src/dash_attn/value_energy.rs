//! ValueEnergyRouter — multi-signal routing via centroid · ‖v‖ gating.
//!
//! Combines key centroid dot product with value energy for block selection.
//! v_energy=0 gates out a block entirely; v_energy>0 passes the centroid dot product.
//! Feature gate: `vortex_flow` (Plan 196, Phase 1).

use super::block_topk::argtopk;
use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// ValueEnergyCache
// ---------------------------------------------------------------------------

/// Cache for ValueEnergyRouter: key centroids + per-block value energy.
#[derive(Debug, Clone)]
pub struct ValueEnergyCache {
    /// Key centroids per block: flat `[n_blocks * head_dim]`.
    pub centroids: Vec<f32>,
    /// Per-block value energy (mean ‖v‖): `[n_blocks]`.
    pub v_energy: Vec<f32>,
    /// Number of blocks cached.
    pub n_blocks: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl ValueEnergyCache {
    /// Create a pre-allocated cache.
    pub fn new(n_blocks_capacity: usize, head_dim: usize) -> Self {
        Self {
            centroids: vec![0.0; n_blocks_capacity * head_dim],
            v_energy: vec![0.0; n_blocks_capacity],
            n_blocks: 0,
            head_dim,
        }
    }

    /// Ensure capacity for at least `n` blocks.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.centroids.len() < n * self.head_dim {
            self.centroids.resize(n * self.head_dim, 0.0);
        }
        if self.v_energy.len() < n {
            self.v_energy.resize(n, 0.0);
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
// ValueEnergyRouter
// ---------------------------------------------------------------------------

/// ValueEnergyRouter — multi-signal routing via centroid · ‖v‖ gating.
///
/// Score formula: `score[i] = dot(centroid[i], query) * v_energy[i]`
///
/// - `v_energy[i] = 0` → block `i` is gated out (score = 0 regardless of centroid alignment)
/// - `v_energy[i] > 0` → passes the centroid dot product through
///
/// This validates that the VortexFlow trait supports multi-signal routing.
#[derive(Debug)]
pub struct ValueEnergyRouter {
    /// Apply `1/sqrt(head_dim)` scaling to dot products.
    pub scale: bool,
}

impl ValueEnergyRouter {
    /// Create a new router with optional scaling.
    pub fn new(scale: bool) -> Self {
        Self { scale }
    }
}

impl VortexFlow for ValueEnergyRouter {
    type Cache = ValueEnergyCache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        cache.ensure_capacity(block_idx + 1);
        let block_size = keys.len() / head_dim;

        // Compute key centroid (mean pooling)
        let start = block_idx * head_dim;
        let centroid = &mut cache.centroids[start..start + head_dim];
        centroid.fill(0.0);

        match block_size {
            0 => {
                cache.v_energy[block_idx] = 0.0;
            }
            _ => {
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

                // Compute value energy: mean ‖v‖ across tokens in this block
                let mut energy = 0.0f32;
                for t in 0..block_size {
                    let v_start = t * head_dim;
                    let norm_sq: f32 = (0..head_dim)
                        .map(|d| values[v_start + d] * values[v_start + d])
                        .sum();
                    energy += norm_sq.sqrt();
                }
                cache.v_energy[block_idx] = energy / block_size as f32;
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

        // Compute gated scores: dot(centroid, query) * v_energy
        for (i, score) in scores.iter_mut().enumerate().take(n_blocks) {
            let centroid = cache.centroid(i);
            let dot: f32 = query.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
            *score = dot * scale * cache.v_energy[i];
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
            let w = super::block_topk::sigmoid(scores[idx]);
            decision.weights.push(w);
        }

        decision
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        ValueEnergyCache::new(n_blocks_capacity, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIM: usize = 4;

    fn make_router() -> ValueEnergyRouter {
        ValueEnergyRouter::new(true)
    }

    #[test]
    fn test_zero_energy_gates_out_block() {
        let router = make_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        // Block 0: strong centroid alignment but zero value energy
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![0.0; HEAD_DIM]; // zero values → zero energy
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        // Block 1: weaker alignment but non-zero energy
        let keys1 = vec![0.1, 0.0, 0.0, 0.0];
        let vals1 = vec![1.0, 1.0, 1.0, 1.0]; // non-zero values
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        assert_eq!(cache.v_energy[0], 0.0);
        assert!(cache.v_energy[1] > 0.0);

        // Query aligned with block 0's centroid, but v_energy=0 gates it out
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);

        // Block 0 should NOT be selected (score = dot * 0 = 0)
        // Block 1 should be selected (score = dot * energy > 0)
        assert_eq!(decision.blocks.len(), 1);
        assert_eq!(decision.blocks[0], 1);
    }

    #[test]
    fn test_positive_energy_passes_centroid_dot() {
        let router = make_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        // Block 0: centroid [1,0,0,0], value energy = 2.0
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![2.0, 0.0, 0.0, 0.0]; // ‖v‖ = 2.0
        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);

        // Block 1: centroid [0,1,0,0], value energy = 1.0
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![1.0, 0.0, 0.0, 0.0]; // ‖v‖ = 1.0
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);

        // Query [1,0,0,0] → dot with block 0 = high * 2.0, dot with block 1 = 0 * 1.0
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 1, &mut scratch);
        assert_eq!(decision.blocks[0], 0);
    }

    #[test]
    fn test_multi_block_selection() {
        let router = make_router();
        let mut cache = router.cache_new(4, HEAD_DIM);
        let mut scratch = VortexScratch::new(4);

        // Create 4 blocks with varying alignment and energy
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let vals0 = vec![1.0, 0.0, 0.0, 0.0];

        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals1 = vec![0.5, 0.0, 0.0, 0.0];

        let keys2 = vec![0.0, 0.0, 1.0, 0.0];
        let vals2 = vec![0.0, 0.0, 0.0, 0.0]; // zero energy → gated

        let keys3 = vec![0.5, 0.5, 0.0, 0.0];
        let vals3 = vec![1.0, 1.0, 1.0, 1.0];

        router.forward_cache(&mut cache, &keys0, &vals0, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys1, &vals1, 1, HEAD_DIM);
        router.forward_cache(&mut cache, &keys2, &vals2, 2, HEAD_DIM);
        router.forward_cache(&mut cache, &keys3, &vals3, 3, HEAD_DIM);

        // Query aligned with [1,0,0,0] → block 0 and 3 should rank high
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 4, 2, &mut scratch);

        assert_eq!(decision.blocks.len(), 2);
        // Block 0 should rank highest (strong alignment + non-zero energy)
        assert_eq!(decision.blocks[0], 0);
        // Block 2 (zero energy) should NOT be in top-2 selection
        assert!(!decision.blocks.contains(&2));
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
    fn test_value_energy_computation() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);

        // Single token: v = [3.0, 4.0, 0.0, 0.0] → ‖v‖ = 5.0
        let keys = vec![1.0, 0.0, 0.0, 0.0];
        let vals = vec![3.0, 4.0, 0.0, 0.0];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        assert!(
            (cache.v_energy[0] - 5.0).abs() < 1e-5,
            "expected energy 5.0, got {}",
            cache.v_energy[0]
        );
    }

    #[test]
    fn test_value_energy_multi_token_mean() {
        let router = make_router();
        let mut cache = router.cache_new(1, HEAD_DIM);

        // 2 tokens: [3,4,0,0] → ‖v‖=5, [0,0,0,0] → ‖v‖=0 → mean=2.5
        let keys = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let vals = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        router.forward_cache(&mut cache, &keys, &vals, 0, HEAD_DIM);

        assert!(
            (cache.v_energy[0] - 2.5).abs() < 1e-5,
            "expected energy 2.5, got {}",
            cache.v_energy[0]
        );
    }

    #[test]
    fn test_cache_ensure_capacity() {
        let mut cache = ValueEnergyCache::new(2, HEAD_DIM);
        assert_eq!(cache.centroids.len(), 8);
        assert_eq!(cache.v_energy.len(), 2);

        cache.ensure_capacity(5);
        assert!(cache.centroids.len() >= 20);
        assert!(cache.v_energy.len() >= 5);
    }

    #[test]
    fn test_all_zero_energy_all_gated() {
        let router = make_router();
        let mut cache = router.cache_new(2, HEAD_DIM);
        let mut scratch = VortexScratch::new(2);

        // Both blocks have zero energy
        let keys0 = vec![1.0, 0.0, 0.0, 0.0];
        let keys1 = vec![0.0, 1.0, 0.0, 0.0];
        let vals = vec![0.0; HEAD_DIM];

        router.forward_cache(&mut cache, &keys0, &vals, 0, HEAD_DIM);
        router.forward_cache(&mut cache, &keys1, &vals, 1, HEAD_DIM);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let decision = router.forward_indexer(&query, &cache, 2, 2, &mut scratch);

        // All scores are 0 → top-k selects arbitrary blocks but weights ≈ 0.5 (sigmoid(0))
        assert_eq!(decision.blocks.len(), 2);
        for &w in &decision.weights {
            assert!(
                (w - 0.5).abs() < 1e-6,
                "expected sigmoid(0)=0.5 for zero-score blocks, got {w}"
            );
        }
    }
}
