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
#[derive(Debug, Clone, Copy)]
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
    for &s in scores[effective_n - rem..effective_n].iter() {
        mean += s * inv;
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
    for &s in scores[effective_n - rem..effective_n].iter() {
        let d = s - mean;
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
// Derived k budget (Issue 747 P1, Research 549 — Lemma 2 law)
// ---------------------------------------------------------------------------

/// Derived k budget from the Lemma-2 two-level law (Issue 747 P1).
///
/// The paper's Lemma 2 (α=1.5): k tied tokens at level-gap Δ above the bulk
/// hold attention 1/k each iff `Δ ≥ 2/√k` — inverting, the boundary block
/// size is `k̂ = 4/Δ̂²` (the α=1.5 instantiation of `((α−1)·Δ̂)^{−1/(α−1)}`).
/// **The condition has no n term** — the budget is length-independent by
/// theorem, which is the property the fitted sigmoid(`w·var+b`) arm of
/// [`compute_adaptive_k`] does not have.
///
/// # Which Δ̂ to feed
///
/// Lemma 2's Δ is the **top-block-to-bulk-level gap**. A max−mean proxy
/// (the shipped `RollingDeltaEstimator` statistic) equals it only in the
/// single-needle regime (k ≈ 1, bulk centered at 0); on multi-level rows it
/// measures top-to-center instead. Callers holding a true level-gap estimate
/// (change-point detector, frozen per-head table) should call this directly;
/// [`compute_derived_k_from_scores`] is the max−mean convenience for the
/// concentration regime.
///
/// # Direction (honest)
///
/// `k̂ = 4/Δ̂²` is the MINIMUM block size that holds concentration at gap Δ̂
/// — a concentration detector (large gap → small budget), not a coverage
/// expander. See the Bench 713 P1 section for the measured split vs the
/// sigmoid arm.
#[cfg(feature = "asentmax_schedule")]
#[inline]
pub fn compute_derived_k(delta_hat: f32, config: &AdaptiveKConfig) -> usize {
    // α = 1.5: k̂ = ((α−1)·Δ̂)^{−1/(α−1)} = (0.5·Δ̂)^{−2} = 4/Δ̂².
    let d = delta_hat.max(1e-3);
    let k_f = 4.0 / (d * d);
    // Saturating conversion: a huge k_f (tiny Δ̂) must clamp, not wrap.
    if k_f >= config.k_max as f32 {
        config.k_max
    } else {
        (k_f.round() as usize).max(config.k_min).min(config.k_max)
    }
}

/// Derived k budget from raw block scores via the max−mean Δ̂ proxy
/// (Issue 747 P1 — the one-pass convenience form).
///
/// Computes `Δ̂ = max(scores) − mean(scores)` (the `RollingDeltaEstimator`
/// statistic; 4-way unrolled to mirror [`compute_adaptive_k`]'s shape) and
/// resolves [`compute_derived_k`]. Calibrated for the single-needle /
/// concentration regime — see that function's "Which Δ̂ to feed" note.
#[cfg(feature = "asentmax_schedule")]
pub fn compute_derived_k_from_scores(
    scores: &[f32],
    n_blocks: usize,
    config: &AdaptiveKConfig,
) -> usize {
    let effective_n = n_blocks.min(scores.len());
    if effective_n == 0 {
        return config.k_min;
    }
    if effective_n <= config.k_min {
        return effective_n;
    }

    // One pass: max + sum (4-way unrolled, mirroring compute_adaptive_k).
    let inv = 1.0 / effective_n as f32;
    let mut mx0 = f32::NEG_INFINITY;
    let mut mx1 = f32::NEG_INFINITY;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let chunks = effective_n / 4;
    for c in 0..chunks {
        let base = c * 4;
        let a = scores[base];
        let b = scores[base + 1];
        let e = scores[base + 2];
        let f = scores[base + 3];
        mx0 = mx0.max(a).max(b);
        mx1 = mx1.max(e).max(f);
        s0 += a + b;
        s1 += e + f;
    }
    let mut max = mx0.max(mx1);
    let mut sum = s0 + s1;
    let rem = effective_n % 4;
    for &s in scores[effective_n - rem..effective_n].iter() {
        max = max.max(s);
        sum += s;
    }
    let mean = sum * inv;
    compute_derived_k(max - mean, config)
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
            .forward_cache(cache, keys, values, block_idx, head_dim);
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

    // --- Derived k budget (Issue 747 P1) ---

    #[cfg(feature = "asentmax_schedule")]
    #[test]
    fn test_derived_k_exact_lemma2_map() {
        let config = AdaptiveKConfig::new(1, 4096);
        // k̂ = 4/Δ̂² exactly at the Lemma-2 boundary.
        assert_eq!(compute_derived_k(2.0, &config), 1); // 4/4
        assert_eq!(compute_derived_k(1.0, &config), 4); // 4/1
        assert_eq!(compute_derived_k(0.5, &config), 16); // 4/0.25
        assert_eq!(compute_derived_k(0.25, &config), 64); // 4/0.0625
        // Monotone decreasing in Δ̂.
        let mut prev = usize::MAX;
        for i in 1..=20 {
            let d = 0.1 * i as f32;
            let k = compute_derived_k(d, &config);
            assert!(k <= prev, "k̂ must be non-increasing in Δ̂ at d={d}");
            prev = k;
        }
    }

    #[cfg(feature = "asentmax_schedule")]
    #[test]
    fn test_derived_k_clamps() {
        let config = AdaptiveKConfig::new(4, 32);
        // Tiny Δ̂ → k̂ huge → k_max (saturating, no wrap).
        assert_eq!(compute_derived_k(1e-4, &config), 32);
        assert_eq!(compute_derived_k(0.0, &config), 32);
        // Huge Δ̂ → k̂ < 1 → k_min.
        assert_eq!(compute_derived_k(100.0, &config), 4);
        // NaN Δ̂: f32::max ignores NaN → d = 1e-3 → huge k̂ → k_max (the
        // safe direction for a budget — garbage estimate, take coverage).
        assert_eq!(compute_derived_k(f32::NAN, &config), 32);
    }

    #[cfg(feature = "asentmax_schedule")]
    #[test]
    fn test_derived_k_from_scores_max_mean_proxy() {
        // Single needle at 5.0 over bulk centered at ~0 → Δ̂ ≈ 5 → k̂ = 4/25 → clamp k_min.
        let config = AdaptiveKConfig::new(4, 64);
        let scores: Vec<f32> = vec![5.0, 0.1, -0.2, 0.05, -0.1, 0.15, -0.05, 0.0];
        assert_eq!(compute_derived_k_from_scores(&scores, 8, &config), 4);
        // Bulk tightly under the top (Δ̂ small) → large budget.
        let tight: Vec<f32> = vec![1.02, 1.0, 1.01, 0.99, 1.0, 1.005, 0.995, 1.0];
        // Δ̂ ≈ 1.02 − ~1.0025 ≈ 0.0175 → k̂ = 4/0.0003 huge → k_max.
        assert_eq!(compute_derived_k_from_scores(&tight, 8, &config), 64);
        // Empty / short rows.
        assert_eq!(compute_derived_k_from_scores(&[], 0, &config), 4);
        assert_eq!(compute_derived_k_from_scores(&[1.0, 2.0], 2, &config), 2);
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
