//! Top-level fusion: LFU cache + shard map + batcher (Plan 218 Phase 5).
//!
//! Pipeline: lookup → cache miss → compute → insert → shard → batch.
//! Thread-safe via papaya lock-free internals in BfcpRegionCache and RegionShardMap.
//!
//! Also provides `freq_aware_complexity` for extending PerceptRouter with frequency tier.

use super::bfcf_types::{BFCP, BorelRegion};
use super::bfcp_region_cache::{BfcpRegionCache, FreqTier, blake3_logit_hash};
use super::region_batch::{RegionBatcher, RegionBatching};
use super::region_shard_map::RegionShardMap;

// ── BfcpLfuShard ─────────────────────────────────────────────────

/// Top-level fusion: LFU cache + shard map + batcher.
///
/// Pipeline: lookup → cache miss → compute → insert → shard → batch.
pub struct BfcpLfuShard {
    cache: BfcpRegionCache,
    shard_map: RegionShardMap,
    batcher: RegionBatcher,
}

impl BfcpLfuShard {
    /// Create all components with given cache capacity and shard count.
    pub fn new(cache_capacity: usize, num_shards: usize) -> Self {
        Self {
            cache: BfcpRegionCache::new(cache_capacity),
            shard_map: RegionShardMap::new(num_shards),
            batcher: RegionBatcher::new(),
        }
    }

    /// Main pipeline: hash → lookup → compute on miss → insert → return partition.
    pub fn process(&mut self, logits: &[f32], mut compute_fn: impl FnMut(&[f32]) -> BFCP) -> BFCP {
        let hash = blake3_logit_hash(logits);

        // Try cache hit first.
        if let Some(partition) = self.cache.lookup(&hash) {
            return partition;
        }

        // Cache miss — compute new partition.
        let partition = compute_fn(logits);
        self.cache.insert(hash, partition.clone());
        partition
    }

    /// Like `process` but also returns shard assignments for each region.
    ///
    /// Returns `(partition, Vec<(shard_index, FreqTier)>)`.
    pub fn process_and_shard(
        &mut self,
        logits: &[f32],
        compute_fn: impl FnMut(&[f32]) -> BFCP,
    ) -> (BFCP, Vec<(usize, FreqTier)>) {
        let partition = self.process(logits, compute_fn);
        let hash = blake3_logit_hash(logits);

        // Derive frequency tier from cache for each region.
        let tier = self.cache.freq_tier(&hash).unwrap_or(FreqTier::Cold);

        let assignments: Vec<(usize, FreqTier)> = partition
            .regions
            .iter()
            .map(|region| {
                let shard = self.shard_map.assign_shard(region.label, tier);
                (shard, tier)
            })
            .collect();

        (partition, assignments)
    }

    /// Batch accept tokens from partition regions (up to `max_tokens`).
    pub fn batch_process(&self, partition: &BFCP, max_tokens: usize) -> Vec<usize> {
        let regions: Vec<&BorelRegion> = partition.regions.iter().collect();
        self.batcher.batch_accept(&regions, max_tokens)
    }

    /// Whether sharding should activate for given region count.
    pub fn should_shard(&self, region_count: usize) -> bool {
        self.shard_map.should_shard(region_count)
    }

    /// Cache hit rate — delegates to BfcpRegionCache.
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    /// Decay cache frequencies by λ — delegates to BfcpRegionCache.
    pub fn decay(&mut self, lambda: f32) {
        self.cache.decay(lambda);
    }

    /// Access the underlying LFU cache.
    pub fn cache(&self) -> &BfcpRegionCache {
        &self.cache
    }

    /// Access the underlying shard map.
    pub fn shard_map(&self) -> &RegionShardMap {
        &self.shard_map
    }

    /// Access the underlying batcher.
    pub fn batcher(&self) -> &RegionBatcher {
        &self.batcher
    }
}

// ── Freq-Aware Complexity ────────────────────────────────────────

/// Extended complexity measure incorporating frequency tier.
///
/// `sigmoid(base_complexity × freq_factor)` where `freq_factor` depends on
/// the dominant tier of cached regions:
/// - Hot → 0.5 (cached, less work needed → lower complexity)
/// - Warm → 0.8 (partially cached → moderate)
/// - Cold → 1.2 (not cached, full recomputation → higher complexity)
pub fn freq_aware_complexity(base_complexity: f32, dominant_tier: FreqTier) -> f32 {
    let freq_factor = match dominant_tier {
        FreqTier::Hot => 0.5,
        FreqTier::Warm => 0.8,
        FreqTier::Cold => 1.2,
    };
    sigmoid(base_complexity * freq_factor)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bfcf_types::{BorelRegion, RegionLabel};

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
    fn test_process_cache_hit() {
        let mut lfu = BfcpLfuShard::new(10, 4);
        let logits = [1.0f32, 2.0, 3.0];
        // First call: miss → compute.
        let mut compute_count = 0;
        let result = lfu.process(&logits, |_input| {
            compute_count += 1;
            make_partition(50, 30, 20)
        });
        assert_eq!(compute_count, 1);
        assert_eq!(result.region_count(), 3);

        // Second call with same logits: hit → compute_fn not called.
        let result2 = lfu.process(&logits, |_input| {
            compute_count += 1;
            make_partition(99, 1, 0) // different partition — should NOT be used
        });
        assert_eq!(
            compute_count, 1,
            "compute_fn should not be called on cache hit"
        );
        // Cached partition has 3 regions, not the 2 the compute_fn would produce.
        assert_eq!(result2.region_count(), 3);
    }

    #[test]
    fn test_process_cache_miss() {
        let mut lfu = BfcpLfuShard::new(10, 4);

        let result = lfu.process(&[0.1, 0.2, 0.3], |_input| make_partition(60, 30, 10));
        assert_eq!(result.region_count(), 3);
        assert_eq!(result.accept_token_count(), 60);
    }

    #[test]
    fn test_process_and_shard_returns_assignments() {
        let mut lfu = BfcpLfuShard::new(10, 4);
        let logits = [1.0f32, 2.0, 3.0];

        let (partition, assignments) =
            lfu.process_and_shard(&logits, |_input| make_partition(50, 30, 20));

        assert_eq!(partition.region_count(), 3);
        assert_eq!(assignments.len(), 3, "one assignment per region");

        for &(shard, tier) in &assignments {
            assert!(shard < 4, "shard {shard} must be < 4");
            // Newly inserted entry has freq=1, so tier should be Cold (warm_threshold=10).
            assert_eq!(tier, FreqTier::Cold, "new entry should be Cold tier");
        }
    }

    #[test]
    fn test_batch_process_returns_tokens() {
        let lfu = BfcpLfuShard::new(10, 4);
        let partition = make_partition(10, 20, 5);

        let tokens = lfu.batch_process(&partition, 15);
        // 10 accept tokens available, capped to 15 → all 10 accepted.
        assert_eq!(tokens.len(), 10);
        // Token indices: offset 0 for first region → 0..9.
        assert_eq!(tokens[0], 0);
        assert_eq!(tokens[9], 9);
    }

    #[test]
    fn test_should_shard_delegates() {
        let lfu = BfcpLfuShard::new(10, 4);

        assert!(!lfu.should_shard(29), "below threshold should not shard");
        assert!(lfu.should_shard(30), "at threshold should shard");
        assert!(lfu.should_shard(100), "above threshold should shard");
    }

    #[test]
    fn test_freq_aware_complexity_hot_reduces() {
        let base = 1.0f32;
        let cold = freq_aware_complexity(base, FreqTier::Cold);
        let hot = freq_aware_complexity(base, FreqTier::Hot);

        assert!(
            hot < cold,
            "Hot tier ({hot}) should produce lower complexity than Cold ({cold})"
        );
        // sigmoid(1.0 * 0.5) = sigmoid(0.5)
        assert!(
            (hot - sigmoid(0.5)).abs() < 1e-6,
            "Hot should be sigmoid(0.5)"
        );
    }

    #[test]
    fn test_freq_aware_complexity_cold_increases() {
        let base = 1.0f32;
        let warm = freq_aware_complexity(base, FreqTier::Warm);
        let cold = freq_aware_complexity(base, FreqTier::Cold);

        assert!(
            cold > warm,
            "Cold tier ({cold}) should produce higher complexity than Warm ({warm})"
        );
        // sigmoid(1.0 * 1.2) = sigmoid(1.2)
        assert!(
            (cold - sigmoid(1.2)).abs() < 1e-6,
            "Cold should be sigmoid(1.2)"
        );
    }

    #[test]
    fn test_decay_integration() {
        let mut lfu = BfcpLfuShard::new(10, 4);

        // Insert and access several times to build frequency.
        let logits = [1.0f32, 2.0, 3.0];
        for _ in 0..5 {
            let _ = lfu.process(&logits, |_input| make_partition(50, 30, 20));
        }

        let hit_rate_before = lfu.cache_hit_rate();
        assert!(
            hit_rate_before > 0.0,
            "should have some hits after 5 accesses"
        );

        // Decay all frequencies.
        lfu.decay(0.5);

        // Cache should still contain the entry (not evicted, just frequency reduced).
        assert_eq!(lfu.cache().len(), 1);
    }

    // Helper for tests — mirrors the module-level sigmoid.
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}
