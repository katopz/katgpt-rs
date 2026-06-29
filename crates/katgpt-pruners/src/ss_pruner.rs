//! SemiseparablePruner — Cumulative temporal influence pruning (Plan 263, Phase 3).
//!
//! Uses the semiseparable structure of diagonal SSMs: in a diagonal state-space model,
//! the influence of a token at position `from` on position `to` decays as the cumulative
//! product of per-position decay factors `a[1..=to]`. When this cumulative influence drops
//! below a threshold, the entire branch rooted at that depth is no longer relevant and can
//! be pruned from the speculative decoding tree.
//!
//! Key design: pre-compute the cumulative product into `influence_cache` at construction
//! time so that `is_valid` is a zero-allocation O(1) slice lookup.

#![allow(clippy::needless_range_loop)]

use katgpt_core::ConstraintPruner;

/// Prunes DDTree branches based on cumulative temporal influence.
///
/// Uses the semiseparable structure of the SSM: if the cumulative influence
/// of a token at depth `from` on the current depth `to` falls below
/// `threshold`, the branch is pruned. This implements the theoretical
/// result that distant tokens in a diagonal SSM have exponentially
/// decaying influence.
pub struct SemiseparablePruner {
    /// Decay factors per depth position [T]. Typically sigmoid(gate) in [0, 1].
    decay_factors: Vec<f32>,
    /// Pruning threshold. Branches with influence below this are pruned.
    threshold: f32,
    /// Pre-computed influence cache: `influence_cache[i] = prod(decay_factors[1..=i])`.
    /// `influence_cache[0] = 1.0` (identity, no decay).
    /// This avoids recomputing the product each time.
    influence_cache: Vec<f32>,
}

impl SemiseparablePruner {
    /// Create a pruner from explicit per-position decay factors.
    ///
    /// Builds the influence cache eagerly: O(T) at construction, O(1) per query.
    pub fn new(decay_factors: Vec<f32>, threshold: f32) -> Self {
        let n = decay_factors.len();
        let mut influence_cache = Vec::with_capacity(n);
        influence_cache.push(1.0); // influence_cache[0] = 1.0 (no decay at depth 0)

        let mut prod = 1.0f32;
        for i in 1..n {
            prod *= decay_factors[i];
            influence_cache.push(prod);
        }

        Self {
            decay_factors,
            threshold,
            influence_cache,
        }
    }

    /// Convenience constructor: uniform decay factor across `seq_len` positions.
    ///
    /// `decay` should be in [0, 1] (typically `sigmoid(gate)`). With `decay = 1.0`,
    /// no pruning occurs. With `decay < 1.0`, influence decays exponentially:
    /// `influence(depth) = decay^depth`.
    pub fn from_uniform(decay: f32, seq_len: usize, threshold: f32) -> Self {
        Self::new(vec![decay; seq_len], threshold)
    }

    /// Returns the pruning threshold.
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Returns the per-position decay factors.
    #[inline]
    pub fn decay_factors(&self) -> &[f32] {
        &self.decay_factors
    }

    /// Returns the cached cumulative influence from depth 0 to `depth`.
    ///
    /// This is `prod(decay_factors[1..=depth])`, equivalent to
    /// `crate::cumprodsum::influence(&decay_factors, 0, depth)`.
    ///
    /// For `depth` beyond the cache range, returns 0.0 (influence has fully decayed
    /// for any `decay < 1.0`; the sequence has ended for `decay == 1.0`).
    #[inline]
    pub fn influence_at(&self, depth: usize) -> f32 {
        if depth < self.influence_cache.len() {
            // SAFETY: bounds checked above
            unsafe { *self.influence_cache.get_unchecked(depth) }
        } else {
            // Beyond the sequence: influence is negligible for decay < 1.0.
            0.0
        }
    }
}

impl ConstraintPruner for SemiseparablePruner {
    /// Check whether the cumulative influence from depth 0 to `depth` is still
    /// above `threshold`.
    ///
    /// The influence is token-independent (depends only on depth in a diagonal SSM),
    /// so `token_idx` and `parent_tokens` are unused. If the root-to-depth influence
    /// has decayed below threshold, the entire branch is pruned — every token at this
    /// depth is rejected.
    #[inline]
    fn is_valid(&self, depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        self.influence_at(depth) >= self.threshold
    }

    /// Batch validation: since validity is depth-dependent only (not token-dependent),
    /// compute the influence once and fill all results.
    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        _parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let valid = self.influence_at(depth) >= self.threshold;
        let len = candidates.len().min(results.len());
        results[..len].fill(valid);
    }

    /// Returns the actual influence value as a soft score.
    ///
    /// High influence (near 1.0) → this depth is highly relevant.
    /// Low influence (near 0.0) → this depth should be pruned.
    fn manifold_score(&self, depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.influence_at(depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decay_allows_all() {
        // decay=1.0 means influence never drops; all depths should be valid.
        let pruner = SemiseparablePruner::from_uniform(1.0, 64, 0.5);

        for depth in 0..64 {
            assert!(
                pruner.is_valid(depth, 0, &[]),
                "depth {} should be valid with no decay",
                depth
            );
        }
    }

    #[test]
    fn test_fast_decay_prunes_far() {
        // decay=0.5, threshold=0.1
        // influence(depth) = 0.5^depth
        //   depth 0: 1.0     >= 0.1 ✓
        //   depth 1: 0.5     >= 0.1 ✓
        //   depth 2: 0.25    >= 0.1 ✓
        //   depth 3: 0.125   >= 0.1 ✓
        //   depth 4: 0.0625  <  0.1 ✗ (pruned)
        let pruner = SemiseparablePruner::from_uniform(0.5, 32, 0.1);

        for depth in 0..=3 {
            assert!(
                pruner.is_valid(depth, 0, &[]),
                "depth {} should be valid",
                depth
            );
        }
        for depth in 4..32 {
            assert!(
                !pruner.is_valid(depth, 0, &[]),
                "depth {} should be pruned",
                depth
            );
        }
    }

    #[test]
    fn test_medium_decay_partial() {
        // decay=0.9, threshold=0.3
        // influence(11) = 0.9^11 ≈ 0.3138 >= 0.3 ✓
        // influence(12) = 0.9^12 ≈ 0.2824 <  0.3 ✗
        let pruner = SemiseparablePruner::from_uniform(0.9, 64, 0.3);

        // Boundary: depth 11 valid, depth 12 pruned
        assert!(
            pruner.is_valid(11, 0, &[]),
            "depth 11 influence {:.6} should be >= 0.3",
            pruner.influence_at(11)
        );
        assert!(
            !pruner.is_valid(12, 0, &[]),
            "depth 12 influence {:.6} should be < 0.3",
            pruner.influence_at(12)
        );

        // Spot-check a few more
        assert!(pruner.is_valid(5, 0, &[]));
        assert!(!pruner.is_valid(20, 0, &[]));
    }

    #[test]
    fn test_influence_cache_correctness() {
        // Non-uniform decay factors for thorough verification
        let decay_factors: Vec<f32> = (0..32).map(|i| 0.95 - i as f32 * 0.01).collect();
        let pruner = SemiseparablePruner::new(decay_factors.clone(), 0.1);

        // influence_cache[i] should match crate::cumprodsum::influence(&decay_factors, 0, i)
        for depth in 0..32 {
            let expected = crate::cumprodsum::influence(&decay_factors, 0, depth);
            let actual = pruner.influence_at(depth);
            assert!(
                (actual - expected).abs() < 1e-6,
                "depth {}: cache {:.10} != expected {:.10}",
                depth,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_from_uniform_matches_vec() {
        let decay = 0.8f32;
        let threshold = 0.2f32;
        let seq_len = 50;

        let pruner_uniform = SemiseparablePruner::from_uniform(decay, seq_len, threshold);
        let pruner_vec = SemiseparablePruner::new(vec![decay; seq_len], threshold);

        for depth in 0..seq_len {
            assert_eq!(
                pruner_uniform.influence_at(depth),
                pruner_vec.influence_at(depth),
                "depth {} influence mismatch",
                depth
            );
            assert_eq!(
                pruner_uniform.is_valid(depth, 0, &[]),
                pruner_vec.is_valid(depth, 0, &[]),
                "depth {} validity mismatch",
                depth
            );
        }
    }

    #[test]
    fn test_batch_is_valid() {
        let pruner = SemiseparablePruner::from_uniform(0.5, 32, 0.1);

        // depth 2: influence 0.25 >= 0.1 → all valid
        let candidates = vec![10, 20, 30, 40, 50];
        let mut results = vec![false; 5];
        pruner.batch_is_valid(2, &candidates, &[], &mut results);
        assert!(
            results.iter().all(|&r| r),
            "depth 2 should allow all candidates"
        );

        // depth 10: influence 0.5^10 ≈ 0.000977 < 0.1 → all pruned
        let mut results2 = vec![true; 5];
        pruner.batch_is_valid(10, &candidates, &[], &mut results2);
        assert!(
            results2.iter().all(|&r| !r),
            "depth 10 should prune all candidates"
        );

        // Verify results length is min(candidates, results)
        let mut short_results = vec![false; 3];
        pruner.batch_is_valid(0, &candidates, &[], &mut short_results);
        assert!(short_results.iter().all(|&r| r));
    }

    #[test]
    fn test_manifold_score_matches_influence() {
        let pruner = SemiseparablePruner::from_uniform(0.7, 64, 0.05);

        for depth in 0..64 {
            let score = pruner.manifold_score(depth, 42, &[]);
            let influence = pruner.influence_at(depth);
            assert!(
                (score - influence).abs() < 1e-7,
                "depth {}: manifold_score {:.10} != influence {:.10}",
                depth,
                score,
                influence
            );
        }
    }
}
