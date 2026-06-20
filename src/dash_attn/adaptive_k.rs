//! Adaptive k budget via sigmoid gate — per-query dynamic block count selection.
//!
//! Feature gate: `msa_adaptive_k` (Plan 256 Phase 2 GOAT gate).
//!
//! The adaptive k router adjusts the number of blocks selected per query based on
//! block score variance:
//! - High variance → query has clear preference → fewer blocks needed (concentrated)
//! - Low variance → query is scattered → more blocks needed (broad attention)
//!
//! Formula: `k = k_min + (k_max - k_min) * sigmoid(w * variance + b)`
//!
//! # Threshold routing
//!
//! | k range | Path |
//! |---------|------|
//! | k ≤ 8 | SIMD-only (register top-k) |
//! | k ≤ 32 | CPU parallel (rayon) |
//! | k > 32 | GPU (documented only, not yet implemented) |

use super::block_topk::sigmoid;
use super::vortex_flow::{RoutingDecision, VortexFlow, VortexScratch};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for adaptive k budget selection.
#[derive(Debug, Clone)]
pub struct AdaptiveKConfig {
    /// Minimum k (floor for budget allocation).
    pub k_min: usize,
    /// Maximum k (ceiling for budget allocation).
    pub k_max: usize,
    /// Sigmoid weight `w` — controls sensitivity to score variance.
    /// Higher w = sharper transition between k_min and k_max.
    pub w: f32,
    /// Sigmoid bias `b` — shifts the variance threshold.
    /// Higher b = prefer higher k (more blocks).
    pub b: f32,
}

impl Default for AdaptiveKConfig {
    fn default() -> Self {
        Self {
            k_min: 4,
            k_max: 32,
            w: 5.0,
            b: 0.0,
        }
    }
}

impl AdaptiveKConfig {
    /// Create config with given k_min/k_max and default sigmoid params.
    pub fn new(k_min: usize, k_max: usize) -> Self {
        Self {
            k_min,
            k_max,
            w: 5.0,
            b: 0.0,
        }
    }

    /// Set sigmoid parameters (builder pattern).
    pub fn with_params(mut self, w: f32, b: f32) -> Self {
        self.w = w;
        self.b = b;
        self
    }
}

// ---------------------------------------------------------------------------
// Adaptive k computation
// ---------------------------------------------------------------------------

/// Compute adaptive k from block score variance.
///
/// `k = k_min + (k_max - k_min) * sigmoid(w * variance + b)`
///
/// - High variance → sigmoid→1 → k near k_max (scattered query, needs more blocks)
/// - Low variance → sigmoid→0 → k near k_min (focused query, needs fewer blocks)
#[inline]
pub fn compute_adaptive_k(scores: &[f32], n_blocks: usize, config: &AdaptiveKConfig) -> usize {
    let effective_n = n_blocks.min(scores.len());
    if effective_n == 0 {
        return config.k_min;
    }
    if effective_n <= config.k_min {
        return effective_n;
    }

    // Compute variance of scores (4-way unrolled accumulation)
    let inv = 1.0 / effective_n as f32;
    let mut m0 = 0.0f32;
    let mut m1 = 0.0f32;
    let mut m2 = 0.0f32;
    let mut m3 = 0.0f32;
    let chunks = effective_n / 4;
    for c in 0..chunks {
        let base = c * 4;
        m0 += scores[base];
        m1 += scores[base + 1];
        m2 += scores[base + 2];
        m3 += scores[base + 3];
    }
    let mut mean = (m0 + m1 + m2 + m3) * inv;
    let rem = effective_n % 4;
    for i in (effective_n - rem)..effective_n {
        mean += scores[i] * inv;
    }

    let mut v0 = 0.0f32;
    let mut v1 = 0.0f32;
    let mut v2 = 0.0f32;
    let mut v3 = 0.0f32;
    for c in 0..chunks {
        let base = c * 4;
        let d0 = scores[base] - mean;
        let d1 = scores[base + 1] - mean;
        let d2 = scores[base + 2] - mean;
        let d3 = scores[base + 3] - mean;
        v0 += d0 * d0;
        v1 += d1 * d1;
        v2 += d2 * d2;
        v3 += d3 * d3;
    }
    let mut variance = (v0 + v1 + v2 + v3) * inv;
    for i in (effective_n - rem)..effective_n {
        let d = scores[i] - mean;
        variance += d * d * inv;
    }

    // sigmoid(w * variance + b)
    let z = config.w * variance + config.b;
    let sig = sigmoid(z);

    let k_range = (config.k_max - config.k_min) as f32;
    let k = config.k_min as f32 + k_range * sig;
    let k = k.round() as usize;

    // Clamp to valid range
    k.max(config.k_min).min(config.k_max).min(effective_n)
}

// ---------------------------------------------------------------------------
// AdaptiveKRouter
// ---------------------------------------------------------------------------

/// Adaptive-k wrapper around any [`VortexFlow`] router.
///
/// Wraps an inner router and dynamically adjusts top_k per query based on
/// score variance. Instead of a fixed top_k, it:
/// 1. Calls inner router with `top_k = k_max` (ceiling) to populate scratch scores
/// 2. Computes score variance from the scratch buffer (scores persist after call)
/// 3. Determines adaptive k via sigmoid gate
/// 4. Truncates the routing decision to adaptive_k blocks
pub struct AdaptiveKRouter<R: VortexFlow> {
    /// Inner router to delegate scoring to.
    pub inner: R,
    /// Adaptive k configuration.
    pub config: AdaptiveKConfig,
}

impl<R: VortexFlow> AdaptiveKRouter<R> {
    /// Create a new adaptive-k router wrapping `inner` with given config.
    pub fn new(inner: R, config: AdaptiveKConfig) -> Self {
        Self { inner, config }
    }
}

impl<R: VortexFlow> VortexFlow for AdaptiveKRouter<R> {
    type Cache = R::Cache;

    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],
        values: &[f32],
        block_idx: usize,
        head_dim: usize,
    ) {
        self.inner
            .forward_cache(cache, keys, values, block_idx, head_dim)
    }

    fn forward_indexer(
        &self,
        query: &[f32],
        cache: &Self::Cache,
        n_blocks: usize,
        _top_k: usize,
        scratch: &mut VortexScratch,
    ) -> RoutingDecision {
        if n_blocks == 0 {
            return RoutingDecision::new();
        }

        // Step 1: Score all blocks up to k_max ceiling.
        // After this call, scratch.scores[..n_blocks] contains the raw block scores.
        let k_max = self.config.k_max.min(n_blocks);
        let mut decision_full = self
            .inner
            .forward_indexer(query, cache, n_blocks, k_max, scratch);

        // Step 2: Compute adaptive k from score variance in scratch.
        // The inner router writes scores to scratch.scores[..n_blocks] and does not clear them.
        let adaptive_k = compute_adaptive_k(&scratch.scores, n_blocks, &self.config);

        // Step 3: If adaptive_k >= what we already have, return as-is.
        if adaptive_k >= decision_full.len() {
            return decision_full;
        }

        // Step 4: Truncate to adaptive_k blocks in place — avoids a fresh
        // `RoutingDecision` allocation when the inner call already produced one.
        decision_full.blocks.truncate(adaptive_k);
        decision_full.weights.truncate(adaptive_k);
        decision_full
    }

    fn cache_new(&self, n_blocks_capacity: usize, head_dim: usize) -> Self::Cache {
        self.inner.cache_new(n_blocks_capacity, head_dim)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dash_attn::block_topk::{BlockTopKCache, BlockTopKRouter};
    use crate::dash_attn::vortex_flow::VortexScratch;

    const HEAD_DIM: usize = 64;

    // --- compute_adaptive_k unit tests ---

    #[test]
    fn test_adaptive_k_high_variance() {
        // One block dominates → high variance → k should be high (k_max range)
        let scores: Vec<f32> = vec![10.0, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01];
        let config = AdaptiveKConfig::new(4, 32);
        let k = compute_adaptive_k(&scores, scores.len(), &config);
        // High variance → sigmoid closer to 1 → k closer to k_max
        assert!(
            k > config.k_min,
            "high variance should give k > k_min, got k={k}"
        );
    }

    #[test]
    fn test_adaptive_k_low_variance() {
        // All scores similar → low variance → k should be low (k_min range)
        let scores: Vec<f32> = vec![1.0, 1.01, 0.99, 1.0, 1.01, 0.99, 1.0, 1.01];
        let config = AdaptiveKConfig::new(4, 32);
        let k = compute_adaptive_k(&scores, scores.len(), &config);
        // Low variance → sigmoid closer to 0 → k closer to k_min
        assert!(
            k <= config.k_min + 4,
            "low variance should give k near k_min, got k={k}"
        );
    }

    #[test]
    fn test_adaptive_k_respects_bounds() {
        // Extreme scores — k should still be within [k_min, k_max]
        let config = AdaptiveKConfig::new(4, 16);
        // Very high variance
        let high_scores: Vec<f32> = vec![100.0; 64];
        let k_high = compute_adaptive_k(&high_scores, high_scores.len(), &config);
        assert!(
            k_high >= config.k_min && k_high <= config.k_max,
            "k_high={k_high} out of bounds [{}, {}]",
            config.k_min,
            config.k_max
        );

        // Very low variance
        let low_scores: Vec<f32> = vec![0.001; 64];
        let k_low = compute_adaptive_k(&low_scores, low_scores.len(), &config);
        assert!(
            k_low >= config.k_min && k_low <= config.k_max,
            "k_low={k_low} out of bounds [{}, {}]",
            config.k_min,
            config.k_max
        );
    }

    #[test]
    fn test_adaptive_k_fewer_blocks_than_k_min() {
        // When n_blocks < k_min, k should equal n_blocks
        let scores: Vec<f32> = vec![1.0, 2.0];
        let config = AdaptiveKConfig::new(4, 32);
        let k = compute_adaptive_k(&scores, 2, &config);
        assert_eq!(k, 2, "should return n_blocks when n_blocks < k_min");
    }

    #[test]
    fn test_adaptive_k_empty_scores() {
        let scores: Vec<f32> = vec![];
        let config = AdaptiveKConfig::new(4, 32);
        let k = compute_adaptive_k(&scores, 0, &config);
        assert_eq!(k, config.k_min, "empty scores should return k_min");
    }

    #[test]
    fn test_adaptive_k_config_builder() {
        let config = AdaptiveKConfig::new(2, 8).with_params(10.0, -1.0);
        assert_eq!(config.k_min, 2);
        assert_eq!(config.k_max, 8);
        assert!((config.w - 10.0).abs() < f32::EPSILON);
        assert!((config.b - (-1.0)).abs() < f32::EPSILON);
    }

    // --- Integration test with BlockTopKRouter ---

    fn make_cache_and_router(
        n_blocks: usize,
        head_dim: usize,
    ) -> (BlockTopKRouter, BlockTopKCache) {
        let router = BlockTopKRouter::new(true);
        let cache = router.cache_new(n_blocks, head_dim);
        (router, cache)
    }

    #[test]
    fn test_adaptive_k_with_block_topk() {
        let block_size = 4; // tokens per block
        let n_blocks = 16;
        let (inner_router, mut cache) = make_cache_and_router(n_blocks, HEAD_DIM);

        // Insert blocks: one "needle" block with large values, rest are small
        let mut block_keys_neeedle = vec![0.0f32; block_size * HEAD_DIM];
        for v in block_keys_neeedle.iter_mut() {
            *v = 5.0; // high magnitude keys
        }

        let mut block_keys_background = vec![0.0f32; block_size * HEAD_DIM];
        for v in block_keys_background.iter_mut() {
            *v = 0.1; // low magnitude keys
        }

        // Block 0 is the needle, rest are background
        inner_router.forward_cache(&mut cache, &block_keys_neeedle, &[0.0; 64], 0, HEAD_DIM);
        for i in 1..n_blocks {
            inner_router.forward_cache(&mut cache, &block_keys_background, &[0.0; 64], i, HEAD_DIM);
        }

        // Query that matches the needle
        let query: Vec<f32> = vec![5.0; HEAD_DIM];

        // Fixed-k router (top_k = 16, selects all)
        let mut scratch_full = VortexScratch::new(n_blocks);
        let decision_fixed =
            inner_router.forward_indexer(&query, &cache, n_blocks, 16, &mut scratch_full);
        assert_eq!(decision_fixed.len(), 16);

        // Adaptive-k router — high variance should yield larger k
        let config = AdaptiveKConfig::new(2, 16);
        let adaptive_router = AdaptiveKRouter::new(inner_router, config);
        let mut cache_adaptive = adaptive_router.cache_new(n_blocks, HEAD_DIM);

        adaptive_router.forward_cache(
            &mut cache_adaptive,
            &block_keys_neeedle,
            &[0.0; 64],
            0,
            HEAD_DIM,
        );
        for i in 1..n_blocks {
            adaptive_router.forward_cache(
                &mut cache_adaptive,
                &block_keys_background,
                &[0.0; 64],
                i,
                HEAD_DIM,
            );
        }

        let mut scratch_adaptive = VortexScratch::new(n_blocks);
        let decision_adaptive = adaptive_router.forward_indexer(
            &query,
            &cache_adaptive,
            n_blocks,
            16, // top_k ignored by adaptive router
            &mut scratch_adaptive,
        );

        // Adaptive k should select a subset (needle block should be first)
        assert!(
            decision_adaptive.len() >= 2,
            "adaptive k should select at least k_min blocks, got {}",
            decision_adaptive.len()
        );
        assert!(
            decision_adaptive.len() <= 16,
            "adaptive k should select at most k_max blocks, got {}",
            decision_adaptive.len()
        );
        // Needle block (0) should be the top selection
        assert_eq!(
            decision_adaptive.blocks[0], 0,
            "needle block should be top-selected"
        );
    }

    #[test]
    fn test_adaptive_k_zero_variance_selects_min() {
        // All identical scores → zero variance → sigmoid(b) with w=5,b=0 → sigmoid(0)=0.5 → k_mid
        // But with b=-10, sigmoid(-10) ≈ 0 → k_min
        let scores: Vec<f32> = vec![1.0; 32];
        let config = AdaptiveKConfig::new(4, 32).with_params(5.0, -10.0);
        let k = compute_adaptive_k(&scores, scores.len(), &config);
        assert_eq!(
            k, config.k_min,
            "zero variance with strong negative bias should give k_min"
        );
    }

    #[test]
    fn test_adaptive_k_large_variance_selects_max() {
        // Large variance with strong positive bias → sigmoid(~large) ≈ 1 → k_max
        let mut scores = vec![0.0f32; 32];
        scores[0] = 100.0;
        let config = AdaptiveKConfig::new(4, 32).with_params(5.0, 10.0);
        let k = compute_adaptive_k(&scores, scores.len(), &config);
        assert_eq!(
            k, config.k_max,
            "large variance with strong positive bias should give k_max"
        );
    }
}

// ---------------------------------------------------------------------------
// TL;DR
// ---------------------------------------------------------------------------
//
// Adaptive k budget via sigmoid gate (Plan 256 Phase 2 GOAT gate).
//
// - `AdaptiveKConfig`: configurable k_min, k_max, sigmoid weight w, bias b
// - `compute_adaptive_k(scores, n_blocks, config)`: k = k_min + (k_max - k_min) * sigmoid(w * variance + b)
// - `AdaptiveKRouter<R: VortexFlow>`: wraps any inner router, calls it with k_max ceiling,
//   reads scratch scores to compute variance, truncates decision to adaptive k
// - Score variance drives budget: high variance = scattered query = more blocks needed
// - Feature flag: `msa_adaptive_k` (depends on `msa_sparse`)
// - Threshold routing: k≤8 → SIMD, k≤32 → CPU parallel, k>32 → GPU (documented only)
