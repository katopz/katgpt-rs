//! Region-level batching for BFCF partitions (Plan 218 Phase 3).
//!
//! Batch processes accept/reject/maybe regions from a BFCP partition:
//! - **Accept**: gather token indices from all accept regions in one pass.
//! - **Reject**: sum token counts (O(regions), zero allocation).
//! - **Maybe**: batch preimage refinement — classify tokens per region and
//!   split into accept/reject sub-regions.
//!
//! Loop structures are SIMD-friendly (contiguous iteration, branch-free inner).

use super::bfcf_types::{BorelRegion, RegionLabel};
use katgpt_speculative::ScreeningPruner;
use std::sync::Arc;

// ── RegionBatching Trait ──────────────────────────────────────────

/// Extension trait for region-level batching operations.
pub trait RegionBatching: Send + Sync {
    /// Batch process accept regions — returns token indices sampled from accept regions.
    /// SIMD-friendly: processes all accept regions in one pass.
    fn batch_accept(&self, regions: &[&BorelRegion], max_tokens: usize) -> Vec<usize>;

    /// Batch process reject regions — O(regions) count of total rejected tokens.
    fn batch_reject_count(&self, regions: &[&BorelRegion]) -> usize;

    /// Batch refine maybe regions — runs preimage refinement on all maybe regions.
    /// Returns refined sub-regions.
    fn batch_refine(
        &self,
        regions: &[&BorelRegion],
        prefix: &[usize],
        pruner: &dyn ScreeningPruner,
        vocab_size: usize,
    ) -> Vec<BorelRegion>;
}

// ── RegionBatcher ─────────────────────────────────────────────────

/// Stateless region batcher — all methods take inputs and return outputs.
/// No interior mutability needed; allocation is O(output) which is unavoidable.
pub struct RegionBatcher;

impl RegionBatcher {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RegionBatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionBatching for RegionBatcher {
    /// Gather up to `max_tokens` token indices from accept regions.
    ///
    /// Deterministic: takes first N tokens from each region in order.
    /// Token indices are a running offset (cumulative count from prior regions)
    /// plus the intra-region index `0..count`.
    fn batch_accept(&self, regions: &[&BorelRegion], max_tokens: usize) -> Vec<usize> {
        if regions.is_empty() || max_tokens == 0 {
            return Vec::new();
        }

        let mut tokens = Vec::with_capacity(max_tokens);
        let mut remaining = max_tokens;
        let mut offset = 0usize;

        for region in regions {
            // Only process accept-labeled regions.
            if region.label != RegionLabel::Accept || remaining == 0 {
                offset += region.token_count;
                continue;
            }

            let count = region.token_count.min(remaining);
            for i in 0..count {
                tokens.push(offset + i);
            }
            remaining -= count;
            offset += region.token_count;
        }

        tokens
    }

    /// Sum `token_count` across all reject regions. O(regions), zero allocation.
    fn batch_reject_count(&self, regions: &[&BorelRegion]) -> usize {
        regions
            .iter()
            .filter(|r| r.label == RegionLabel::Reject)
            .map(|r| r.token_count)
            .sum()
    }

    /// For each maybe region, classify each token via `pruner.relevance()` and
    /// split into accept/reject sub-regions. Returns a flat list of sub-regions.
    ///
    /// Non-maybe regions in the input slice are ignored.
    fn batch_refine(
        &self,
        regions: &[&BorelRegion],
        prefix: &[usize],
        pruner: &dyn ScreeningPruner,
        vocab_size: usize,
    ) -> Vec<BorelRegion> {
        let depth = prefix.len();
        // At most 2 sub-regions per maybe region (accept + reject)
        let maybe_count = regions
            .iter()
            .filter(|r| r.label == RegionLabel::Maybe)
            .count();
        let mut sub_regions = Vec::with_capacity(maybe_count * 2);

        for region in regions {
            if region.label != RegionLabel::Maybe {
                continue;
            }

            // Track running offset across regions for correct token indexing.
            let region_offset = 0usize; // per-region: tokens are 0..token_count
            let mut accept_count = 0usize;
            let mut reject_count = 0usize;

            // Classify each token in this region.
            for token_idx in 0..region.token_count {
                let global_idx = region_offset + token_idx;
                if global_idx >= vocab_size {
                    break;
                }
                let rel = pruner.relevance(depth, global_idx, prefix);
                if rel >= 0.5 {
                    accept_count += 1;
                } else {
                    reject_count += 1;
                }
            }

            // Emit accept sub-region.
            if accept_count > 0 {
                sub_regions.push(BorelRegion::from_arc(
                    RegionLabel::Accept,
                    Arc::clone(&region.constraints),
                    accept_count,
                ));
            }

            // Emit reject sub-region.
            if reject_count > 0 {
                sub_regions.push(BorelRegion::from_arc(
                    RegionLabel::Reject,
                    Arc::clone(&region.constraints),
                    reject_count,
                ));
            }
        }

        sub_regions
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bfcf_types::{BorelRegion, RegionLabel};

    /// Simple threshold pruner for testing.
    struct ThresholdPruner {
        threshold: f32,
    }

    impl ScreeningPruner for ThresholdPruner {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if (token_idx as f32) < self.threshold {
                1.0
            } else {
                0.1
            }
        }
    }

    fn make_region(label: RegionLabel, token_count: usize) -> BorelRegion {
        BorelRegion::new(label, vec![], token_count)
    }

    #[test]
    fn test_batch_accept_returns_tokens() {
        let batcher = RegionBatcher::new();
        let r1 = make_region(RegionLabel::Accept, 10);
        let r2 = make_region(RegionLabel::Accept, 20);
        let r3 = make_region(RegionLabel::Accept, 30);
        let regions: Vec<&BorelRegion> = vec![&r1, &r2, &r3];

        let tokens = batcher.batch_accept(&regions, 25);
        assert_eq!(tokens.len(), 25);
        // First 10 from r1 (offset 0)
        assert_eq!(tokens[0], 0);
        assert_eq!(tokens[9], 9);
        // Next 15 from r2 (offset 10)
        assert_eq!(tokens[10], 10);
        assert_eq!(tokens[24], 24);
    }

    #[test]
    fn test_batch_accept_respects_max() {
        let batcher = RegionBatcher::new();
        let r1 = make_region(RegionLabel::Accept, 100);
        let regions: Vec<&BorelRegion> = vec![&r1];

        let tokens = batcher.batch_accept(&regions, 5);
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_batch_accept_empty_regions() {
        let batcher = RegionBatcher::new();
        let regions: Vec<&BorelRegion> = vec![];
        let tokens = batcher.batch_accept(&regions, 100);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_batch_reject_count_sums() {
        let batcher = RegionBatcher::new();
        let r1 = make_region(RegionLabel::Reject, 10);
        let r2 = make_region(RegionLabel::Reject, 20);
        let r3 = make_region(RegionLabel::Reject, 30);
        let regions: Vec<&BorelRegion> = vec![&r1, &r2, &r3];

        assert_eq!(batcher.batch_reject_count(&regions), 60);
    }

    #[test]
    fn test_batch_reject_count_empty() {
        let batcher = RegionBatcher::new();
        let regions: Vec<&BorelRegion> = vec![];
        assert_eq!(batcher.batch_reject_count(&regions), 0);
    }

    #[test]
    fn test_batch_refine_splits_maybe() {
        let batcher = RegionBatcher::new();
        let pruner = ThresholdPruner { threshold: 5.0 };
        // 10 tokens: 0..4 → accept (relevance 1.0), 5..9 → reject (relevance 0.1)
        let region = make_region(RegionLabel::Maybe, 10);
        let regions: Vec<&BorelRegion> = vec![&region];

        let sub_regions = batcher.batch_refine(&regions, &[], &pruner, 10);
        assert_eq!(sub_regions.len(), 2);

        // One accept sub-region, one reject sub-region.
        let accept_sub = sub_regions.iter().find(|r| r.label == RegionLabel::Accept);
        let reject_sub = sub_regions.iter().find(|r| r.label == RegionLabel::Reject);
        assert!(accept_sub.is_some());
        assert!(reject_sub.is_some());
        assert_eq!(accept_sub.unwrap().token_count, 5);
        assert_eq!(reject_sub.unwrap().token_count, 5);
    }

    #[test]
    fn test_batch_refine_empty() {
        let batcher = RegionBatcher::new();
        let pruner = ThresholdPruner { threshold: 5.0 };
        let regions: Vec<&BorelRegion> = vec![];

        let sub_regions = batcher.batch_refine(&regions, &[], &pruner, 100);
        assert!(sub_regions.is_empty());
    }

    #[test]
    fn test_batch_accept_skips_non_accept() {
        let batcher = RegionBatcher::new();
        let r1 = make_region(RegionLabel::Reject, 10);
        let r2 = make_region(RegionLabel::Accept, 5);
        let regions: Vec<&BorelRegion> = vec![&r1, &r2];

        let tokens = batcher.batch_accept(&regions, 100);
        // Offset for r1 (10) + r2 tokens 0..4 → [10, 11, 12, 13, 14]
        assert_eq!(tokens, vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_batch_reject_count_ignores_non_reject() {
        let batcher = RegionBatcher::new();
        let r1 = make_region(RegionLabel::Accept, 10);
        let r2 = make_region(RegionLabel::Reject, 20);
        let r3 = make_region(RegionLabel::Maybe, 30);
        let regions: Vec<&BorelRegion> = vec![&r1, &r2, &r3];

        assert_eq!(batcher.batch_reject_count(&regions), 20);
    }

    #[test]
    fn test_batch_refine_ignores_non_maybe() {
        let batcher = RegionBatcher::new();
        let pruner = ThresholdPruner { threshold: 5.0 };
        let r1 = make_region(RegionLabel::Accept, 10);
        let r2 = make_region(RegionLabel::Reject, 10);
        let regions: Vec<&BorelRegion> = vec![&r1, &r2];

        let sub_regions = batcher.batch_refine(&regions, &[], &pruner, 20);
        assert!(sub_regions.is_empty());
    }
}
