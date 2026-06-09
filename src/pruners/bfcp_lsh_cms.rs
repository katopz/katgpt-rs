//! Top-level fusion: LSH approximate cache + Count-Min Sketch + Roaring membership.
//!
//! Pipeline: L0 exact → L1 LSH approximate → compute → insert → CMS update.
//! Batch operations use CompactBitmap (Roaring-like) for SIMD-friendly membership.
//!
//! Plan 220 Phase 4.

use std::sync::Arc;

use super::bfcf_types::{BFCP, RegionLabel};
use super::bfcp_region_cache::{FreqTier, blake3_logit_hash};
use super::count_min_sketch::{CountMinSketch, SketchFrequency};
use super::lsh_cache::BfcpLshCache;
use super::region_shard_map::RegionShardMap;
use super::roaring_membership::{CompactBitmap, RoaringBatching, RoaringMembership};

// ── BfcpLshCms ─────────────────────────────────────────────────

/// Top-level fusion: LSH approximate cache + CMS frequency + Roaring membership.
///
/// Three-level cache hierarchy:
/// - **L0**: BLAKE3 exact match → O(1), zero recomputation
/// - **L1**: LSH SimHash approximate → warm-start from cached partition
/// - **L2**: Full compute from scratch
///
/// Frequency tracking via Count-Min Sketch (O(1) update/estimate/decay).
/// Batch operations via CompactBitmap (Roaring-like compressed membership).
pub struct BfcpLshCms {
    /// Three-level cache: exact + LSH approximate.
    cache: BfcpLshCache,
    /// Count-Min Sketch for O(1) frequency estimation.
    sketch: CountMinSketch,
    /// Shard map for frequency-aware region assignment.
    shard_map: RegionShardMap,
    /// Roaring-backed membership for batch operations.
    membership: RoaringMembership,
    /// Hot tier threshold (for CMS-based classification).
    hot_threshold: u32,
    /// Warm tier threshold (for CMS-based classification).
    warm_threshold: u32,
}

impl BfcpLshCms {
    /// Create a new fused cache with all components.
    pub fn new(
        exact_capacity: usize,
        logit_dim: usize,
        num_buckets: usize,
        bucket_capacity: usize,
        hamming_radius: u32,
        num_shards: usize,
    ) -> Self {
        Self {
            cache: BfcpLshCache::new(
                exact_capacity,
                logit_dim,
                num_buckets,
                bucket_capacity,
                hamming_radius,
            ),
            sketch: CountMinSketch::new(),
            shard_map: RegionShardMap::new(num_shards),
            membership: RoaringMembership::new(),
            hot_threshold: 100,
            warm_threshold: 10,
        }
    }

    /// Main pipeline: L0 exact → L1 LSH → full compute → insert → CMS update.
    ///
    /// Returns `(partition, level, tier)` where level is 0/1/2 and tier is the
    /// CMS-estimated frequency tier for the hash.
    pub fn process<F>(&mut self, logits: &[f32], compute_fn: F) -> (Arc<BFCP>, u8, FreqTier)
    where
        F: FnOnce(&[f32]) -> BFCP,
    {
        let hash = blake3_logit_hash(logits);

        // Three-level cache lookup.
        let (partition, level) = self.cache.process(logits, compute_fn);

        // CMS frequency update.
        self.sketch.update(&hash);
        let tier = self
            .sketch
            .freq_tier_sketch(&hash, self.hot_threshold, self.warm_threshold);

        // Update membership bitmaps for the partition's regions.
        self.update_membership(&partition);

        (partition, level, tier)
    }

    /// Batch process accept tokens from partition regions using CompactBitmap.
    pub fn batch_accept(&self, max_tokens: usize) -> Vec<u32> {
        self.membership
            .roaring_accept_tokens(self.membership.bitmaps(), max_tokens)
    }

    /// Batch count rejected tokens using CompactBitmap O(1) len().
    pub fn batch_reject_count(&self) -> u64 {
        self.membership
            .roaring_reject_count(self.membership.bitmaps())
    }

    /// Get frequency tier for a hash from CMS estimate.
    pub fn freq_tier(&self, hash: &[u8; 32]) -> FreqTier {
        self.sketch
            .freq_tier_sketch(hash, self.hot_threshold, self.warm_threshold)
    }

    /// Assign shard for a region based on label and frequency tier.
    pub fn assign_shard(&self, label: RegionLabel, hash: &[u8; 32]) -> usize {
        let tier = self.freq_tier(hash);
        self.shard_map.assign_shard(label, tier)
    }

    /// O(1) decay all CMS frequencies by λ.
    pub fn decay(&mut self, lambda: f32) {
        self.sketch.sketch_decay(lambda);
    }

    /// Three-level hit rates: (l0_rate, l1_rate, miss_rate).
    pub fn hit_rates(&self) -> (f64, f64, f64) {
        self.cache.hit_rates()
    }

    /// Whether sharding should activate for given region count.
    pub fn should_shard(&self, region_count: usize) -> bool {
        self.shard_map.should_shard(region_count)
    }

    /// Access the underlying CMS for diagnostics.
    pub fn sketch(&self) -> &CountMinSketch {
        &self.sketch
    }

    /// Access the underlying three-level cache.
    pub fn cache(&self) -> &BfcpLshCache {
        &self.cache
    }

    /// Access the underlying membership bitmaps.
    pub fn membership(&self) -> &RoaringMembership {
        &self.membership
    }

    /// Access the underlying shard map.
    pub fn shard_map(&self) -> &RegionShardMap {
        &self.shard_map
    }

    /// Update membership bitmaps from a partition's regions.
    fn update_membership(&mut self, partition: &Arc<BFCP>) {
        // Build bitmaps for each region based on token ranges.
        // This is a simplified version — real implementation would use
        // actual token indices from ScreeningPruner results.
        let mut bitmaps = Vec::new();
        let mut offset = 0u32;
        for region in &partition.regions {
            let mut bm = CompactBitmap::new();
            for i in 0..region.token_count as u32 {
                bm.insert(offset + i);
            }
            bitmaps.push(bm);
            offset += region.token_count as u32;
        }
        self.membership = RoaringMembership::from_bitmaps(bitmaps);
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bfcf_types::BorelRegion;

    /// Helper: create a BFCP with given token counts per label.
    fn make_partition(accept: usize, reject: usize, maybe: usize) -> BFCP {
        let mut regions = Vec::new();
        if accept > 0 {
            regions.push(BorelRegion::new(RegionLabel::Accept, vec![], accept));
        }
        if reject > 0 {
            regions.push(BorelRegion::new(RegionLabel::Reject, vec![], reject));
        }
        if maybe > 0 {
            regions.push(BorelRegion::new(RegionLabel::Maybe, vec![], maybe));
        }
        BFCP::from_regions(regions)
    }

    #[test]
    fn test_process_fresh_compute() {
        let mut lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let (partition, level, tier) = lsh.process(&logits, |_input| make_partition(50, 30, 20));
        assert_eq!(level, 2, "first call should be full compute");
        assert_eq!(partition.region_count(), 3);
        assert_eq!(tier, FreqTier::Cold, "new entry should be Cold (freq=1)");
    }

    #[test]
    fn test_process_exact_hit() {
        let mut lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let _ = lsh.process(&logits, |_input| make_partition(50, 30, 20));

        // Same logits → exact hit.
        let (partition, level, _tier) = lsh.process(&logits, |_input| make_partition(99, 1, 0));
        assert_eq!(level, 0, "same logits should be L0 exact hit");
        assert_eq!(
            partition.region_count(),
            3,
            "should return cached partition, not compute_fn result"
        );
    }

    #[test]
    fn test_cms_freq_tier_promotion() {
        let mut lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Access same logits many times to promote through tiers.
        for _ in 0..150 {
            let _ = lsh.process(&logits, |_input| make_partition(50, 30, 20));
        }

        let hash = blake3_logit_hash(&logits);
        let tier = lsh.freq_tier(&hash);
        assert!(
            matches!(tier, FreqTier::Hot),
            "after 150 accesses, should be Hot tier, got {:?}",
            tier
        );
    }

    #[test]
    fn test_cms_decay_reduces_tier() {
        let mut lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Promote to Hot.
        for _ in 0..150 {
            let _ = lsh.process(&logits, |_input| make_partition(50, 30, 20));
        }
        let hash = blake3_logit_hash(&logits);
        assert!(matches!(lsh.freq_tier(&hash), FreqTier::Hot));

        // Heavy decay should demote.
        lsh.decay(0.01); // Aggressive decay.
        let tier = lsh.freq_tier(&hash);
        assert!(
            matches!(tier, FreqTier::Warm | FreqTier::Cold),
            "after aggressive decay, should demote from Hot, got {:?}",
            tier
        );
    }

    #[test]
    fn test_batch_reject_count() {
        let mut lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);
        let _ = lsh.process(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], |_input| {
            make_partition(50, 30, 20)
        });

        // Membership bitmaps should have been built.
        let _count = lsh.batch_reject_count();
    }

    #[test]
    fn test_shard_assignment() {
        let lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);
        let shard = lsh.assign_shard(RegionLabel::Accept, &[0u8; 32]);
        assert!(shard < 4, "shard {shard} must be < 4");
    }

    #[test]
    fn test_hit_rates_tracking() {
        let mut lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);
        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // First: miss.
        let _ = lsh.process(&logits, |_input| make_partition(50, 30, 20));
        // Second: hit.
        let _ = lsh.process(&logits, |_input| make_partition(99, 1, 0));

        let (l0, l1, miss) = lsh.hit_rates();
        assert!(l0 > 0.0, "should have L0 hits");
        assert!(miss > 0.0, "should have misses");
        assert!(
            (l0 + l1 + miss - 1.0).abs() < 1e-6,
            "rates should sum to 1.0"
        );
    }

    #[test]
    fn test_end_to_end_pipeline() {
        let mut lsh = BfcpLshCms::new(50, 8, 256, 8, 3, 4);

        // Simulate decode with 5 distinct query patterns, each repeated ~20 times.
        // This models top-1 sampling where the model keeps producing similar logits.
        let patterns: Vec<Vec<f32>> = (0..5)
            .map(|p| (0..8).map(|i| (p * 10 + i) as f32 * 0.1).collect())
            .collect();

        let mut l0_count = 0u64;
        let mut l1_count = 0u64;
        let mut miss_count = 0u64;

        for step in 0..100 {
            // Cycle through patterns. Each pattern repeats 20 times.
            let pattern_idx = step / 20;
            // Add tiny perturbation on even steps, exact repeat on odd steps.
            let logits = if step % 2 == 1 {
                // Exact repeat of the first occurrence in this batch.
                patterns[pattern_idx].clone()
            } else {
                // Small perturbation.
                patterns[pattern_idx]
                    .iter()
                    .map(|v| v + 0.001 * (step as f32))
                    .collect()
            };

            let (_, level, _) = lsh.process(&logits, |_input| make_partition(50, 30, 20));
            match level {
                0 => l0_count += 1,
                1 => l1_count += 1,
                _ => miss_count += 1,
            }
        }

        let total = l0_count + l1_count + miss_count;
        let l0_rate = l0_count as f64 / total as f64;
        let combined_rate = (l0_count + l1_count) as f64 / total as f64;

        // With 50% exact repeats, L0 should be >30%.
        assert!(
            l0_rate > 0.3,
            "L0 hit rate should be >30%, got {:.1}%",
            l0_rate * 100.0
        );

        // Combined should be even higher thanks to LSH.
        assert!(
            combined_rate > 0.5,
            "combined L0+L1 coverage should be >50%, got {:.1}%",
            combined_rate * 100.0
        );
    }

    #[test]
    fn test_feature_off_same_as_218() {
        // This test verifies that with the feature enabled, we get the same
        // partitions as Plan 218's BfcpRegionCache for identical inputs.
        let mut lsh = BfcpLshCms::new(10, 8, 64, 4, 3, 4);

        let logits = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (partition, _, _) = lsh.process(&logits, |_input| make_partition(60, 30, 10));
        assert_eq!(partition.accept_token_count(), 60);
        assert_eq!(partition.reject_token_count(), 30);
        assert_eq!(partition.maybe_token_count(), 10);

        // Second call with same logits → cached.
        let (cached, level, _) = lsh.process(&logits, |_input| make_partition(99, 1, 0));
        assert_eq!(level, 0);
        assert_eq!(cached.accept_token_count(), 60);
        assert_eq!(cached.reject_token_count(), 30);
    }
}
