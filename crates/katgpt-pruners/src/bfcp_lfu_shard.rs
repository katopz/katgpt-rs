//! Top-level fusion: LFU cache + shard map + batcher (Plan 218 Phase 5).
//!
//! Pipeline: lookup → cache miss → compute → insert → shard → batch.
//! Thread-safe via papaya lock-free internals in BfcpRegionCache and RegionShardMap.
//!
//! Also provides `freq_aware_complexity` for extending PerceptRouter with frequency tier.
//!
//! With `freq_bandit` feature: `ShardTierBandit` adapts tier decisions via UCB1
//! based on cache hit rewards, overriding static frequency thresholds.

use std::sync::Arc;

use super::bfcf_types::{BFCP, BorelRegion};
use super::bfcp_region_cache::{BfcpRegionCache, FreqTier, blake3_logit_hash};
use super::region_batch::{RegionBatcher, RegionBatching};
use super::region_shard_map::RegionShardMap;

#[cfg(feature = "freq_bandit")]
use crate::freq_bandit::FrequencyBandit;

// ── ShardTierBandit (freq_bandit integration, Plan 218 Phase 5) ────

/// UCB1 bandit that adapts shard tier decisions based on cache hit rewards.
///
/// Arms: `[Hot, Warm, Cold]` (indices 0, 1, 2 matching `FreqTier` repr).
/// The bandit starts with no preference and learns which tier assignment
/// yields the best cache performance over time.
///
/// Usage: after computing the static `FreqTier` from LFU frequency counts,
/// feed the reward signal (hit/miss) to `update_reward`, then use
/// `refine_tier` to let the bandit promote or demote the tier.
#[cfg(feature = "freq_bandit")]
pub struct ShardTierBandit {
    bandit: FrequencyBandit,
}

#[cfg(feature = "freq_bandit")]
impl ShardTierBandit {
    /// Create a new shard tier bandit with default exploration constant.
    pub fn new() -> Self {
        Self {
            bandit: FrequencyBandit::new(),
        }
    }

    /// Refine the static tier using bandit knowledge.
    ///
    /// If the bandit has learned that a different tier performs better
    /// (based on accumulated reward signals), it may promote or demote.
    /// The static tier serves as a prior — the bandit can only shift by
    /// one tier level at a time to avoid erratic jumping.
    ///
    /// Returns the (possibly adjusted) tier.
    pub fn refine_tier(&self, static_tier: FreqTier) -> FreqTier {
        // If the bandit hasn't been trained yet, trust the static classification.
        if self.bandit.total_pulls() == 0 {
            return static_tier;
        }

        let best_tier = self.bandit_best_tier();

        // Allow at most one tier shift per call to prevent oscillation.
        match (static_tier, best_tier) {
            // Agreement — no change.
            (FreqTier::Hot, FreqTier::Hot)
            | (FreqTier::Warm, FreqTier::Warm)
            | (FreqTier::Cold, FreqTier::Cold) => static_tier,

            // Promotion by one level.
            (FreqTier::Warm, FreqTier::Hot) => FreqTier::Hot,
            (FreqTier::Cold, FreqTier::Warm) => FreqTier::Warm,

            // Demotion by one level.
            (FreqTier::Hot, FreqTier::Warm) => FreqTier::Warm,
            (FreqTier::Warm, FreqTier::Cold) => FreqTier::Cold,

            // Two-level shift: only move one step toward bandit recommendation.
            (FreqTier::Cold, FreqTier::Hot) => FreqTier::Warm,
            (FreqTier::Hot, FreqTier::Cold) => FreqTier::Warm,
        }
    }

    /// Feed reward signal from cache hit (1.0) or miss (0.0), or any [0, 1] value.
    pub fn update_reward(&mut self, tier: FreqTier, reward: f64) {
        self.bandit.update(tier.into(), reward);
    }

    /// Number of reward updates across all arms.
    pub fn total_pulls(&self) -> u32 {
        self.bandit.total_pulls()
    }

    /// Best tier according to accumulated Q-values.
    fn bandit_best_tier(&self) -> FreqTier {
        let best = self.bandit.best_arm();
        best.into()
    }
}

#[cfg(feature = "freq_bandit")]
impl Default for ShardTierBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── FreqTier ↔ FrequencyBand conversion ───────────────────────────

/// Map `FreqTier` arm index to `FrequencyBand` for the shared bandit core.
/// `FreqTier` repr: Hot=0, Warm=1, Cold=2 → maps to FrequencyBand indices.
#[cfg(feature = "freq_bandit")]
impl From<FreqTier> for crate::freq_bandit::FrequencyBand {
    fn from(tier: FreqTier) -> Self {
        match tier {
            FreqTier::Hot => Self::High, // Hot cache hit rate = high frequency access
            FreqTier::Warm => Self::Mid,
            FreqTier::Cold => Self::Low,
        }
    }
}

/// Map `FrequencyBand` back to `FreqTier`.
#[cfg(feature = "freq_bandit")]
impl From<crate::freq_bandit::FrequencyBand> for FreqTier {
    fn from(band: crate::freq_bandit::FrequencyBand) -> Self {
        match band {
            crate::freq_bandit::FrequencyBand::High => FreqTier::Hot,
            crate::freq_bandit::FrequencyBand::Mid => FreqTier::Warm,
            crate::freq_bandit::FrequencyBand::Low => FreqTier::Cold,
        }
    }
}

// ── BfcpLfuShard ─────────────────────────────────────────────────

/// Top-level fusion: LFU cache + shard map + batcher.
///
/// Pipeline: lookup → cache miss → compute → insert → shard → batch.
pub struct BfcpLfuShard {
    cache: BfcpRegionCache,
    shard_map: RegionShardMap,
    batcher: RegionBatcher,
    /// Optional bandit for adaptive tier refinement (Plan 189 integration).
    #[cfg(feature = "freq_bandit")]
    tier_bandit: ShardTierBandit,
}

impl BfcpLfuShard {
    /// Create all components with given cache capacity and shard count.
    pub fn new(cache_capacity: usize, num_shards: usize) -> Self {
        Self {
            cache: BfcpRegionCache::new(cache_capacity),
            shard_map: RegionShardMap::new(num_shards),
            batcher: RegionBatcher::new(),
            #[cfg(feature = "freq_bandit")]
            tier_bandit: ShardTierBandit::new(),
        }
    }

    /// Main pipeline: hash → lookup → compute on miss → insert → return partition.
    pub fn process(
        &mut self,
        logits: &[f32],
        mut compute_fn: impl FnMut(&[f32]) -> BFCP,
    ) -> Arc<BFCP> {
        self.process_with_hash(logits, &mut compute_fn).0
    }

    /// Like [`process`](Self::process) but also returns the BLAKE3 hash of `logits`
    /// so callers that need the hash for downstream cache operations (e.g. tier
    /// lookup) can avoid recomputing it (Issue 001 H-28).
    ///
    /// Returns `(partition, hash)`. The hash is computed exactly once per call.
    pub fn process_with_hash(
        &mut self,
        logits: &[f32],
        compute_fn: &mut impl FnMut(&[f32]) -> BFCP,
    ) -> (Arc<BFCP>, [u8; 32]) {
        let hash = blake3_logit_hash(logits);

        // Try cache hit first.
        if let Some(partition) = self.cache.lookup(&hash) {
            return (partition, hash);
        }

        // Cache miss — compute new partition.
        let partition = Arc::new(compute_fn(logits));
        self.cache.insert(hash, Arc::clone(&partition));
        (partition, hash)
    }

    /// Like `process` but also returns shard assignments for each region.
    ///
    /// Returns `(partition, Vec<(shard_index, FreqTier)>)`.
    ///
    /// With `freq_bandit` feature: after deriving the static tier from LFU
    /// counts, the bandit may refine the tier based on learned rewards.
    pub fn process_and_shard(
        &mut self,
        logits: &[f32],
        mut compute_fn: impl FnMut(&[f32]) -> BFCP,
    ) -> (Arc<BFCP>, Vec<(usize, FreqTier)>) {
        // Reuse the hash computed inside process_with_hash instead of
        // recomputing blake3_logit_hash a second time (Issue 001 H-28).
        let (partition, hash) = self.process_with_hash(logits, &mut compute_fn);

        // Derive frequency tier from cache for each region.
        let static_tier = self.cache.freq_tier(&hash).unwrap_or(FreqTier::Cold);

        // Bandit-refined tier (no-op without freq_bandit feature).
        #[cfg(feature = "freq_bandit")]
        let tier = self.tier_bandit.refine_tier(static_tier);
        #[cfg(not(feature = "freq_bandit"))]
        let tier = static_tier;

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

    /// Feed reward signal to the tier bandit (hit=1.0, miss=0.0, or any [0,1]).
    ///
    /// Only available with `freq_bandit` feature. No-op otherwise.
    #[cfg(feature = "freq_bandit")]
    pub fn update_bandit_reward(&mut self, tier: FreqTier, reward: f64) {
        self.tier_bandit.update_reward(tier, reward);
    }

    /// Access the tier bandit (for direct queries like `total_pulls`).
    #[cfg(feature = "freq_bandit")]
    pub fn tier_bandit(&self) -> &ShardTierBandit {
        &self.tier_bandit
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

    /// Derive `FreqTier` from cache, optionally refined by bandit.
    ///
    /// With `freq_bandit`: returns bandit-refined tier.
    /// Without: returns static tier directly.
    pub fn derive_tier(&self, hash: &[u8; 32]) -> FreqTier {
        let static_tier = self.cache.freq_tier(hash).unwrap_or(FreqTier::Cold);

        #[cfg(feature = "freq_bandit")]
        {
            self.tier_bandit.refine_tier(static_tier)
        }
        #[cfg(not(feature = "freq_bandit"))]
        {
            static_tier
        }
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

/// Frequency-adaptive manifold radius for BFCP regions.
///
/// `D_r = base_radius * sigmoid(freq / freq_scale)`
/// - Hot (high freq) → large D_r → wide manifold → more candidates pass
/// - Cold (low freq) → small D_r → tight manifold → fewer candidates pass
/// - At zero frequency: `sigmoid(0) = 0.5` → default radius = `base_radius * 0.5`
///
/// Feature-gated behind `manifold_pruner` — extends existing BFCP.
#[cfg(feature = "manifold_pruner")]
pub fn region_radius(base_radius: f32, freq: f32, freq_scale: f32) -> f32 {
    base_radius * sigmoid(freq / freq_scale)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bfcf_types::{BorelRegion, RegionLabel};

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

    // ── Bandit integration tests (freq_bandit feature) ───────────

    #[cfg(feature = "freq_bandit")]
    #[test]
    fn test_bandit_refine_tier_no_pulls_returns_static() {
        use super::ShardTierBandit;
        let bandit = ShardTierBandit::new();

        // No reward updates yet — must return the static tier unchanged.
        assert_eq!(bandit.refine_tier(FreqTier::Hot), FreqTier::Hot);
        assert_eq!(bandit.refine_tier(FreqTier::Warm), FreqTier::Warm);
        assert_eq!(bandit.refine_tier(FreqTier::Cold), FreqTier::Cold);
    }

    #[cfg(feature = "freq_bandit")]
    #[test]
    fn test_bandit_refine_tier_promotes_on_high_reward() {
        use super::ShardTierBandit;
        let mut bandit = ShardTierBandit::new();

        // Train: Hot tier gets high reward, Cold gets low reward.
        for _ in 0..10 {
            bandit.update_reward(FreqTier::Hot, 1.0);
            bandit.update_reward(FreqTier::Cold, 0.1);
        }

        // Warm should be promoted toward Hot (bandit prefers Hot).
        let refined = bandit.refine_tier(FreqTier::Warm);
        assert_eq!(refined, FreqTier::Hot, "Warm should be promoted to Hot");
    }

    #[cfg(feature = "freq_bandit")]
    #[test]
    fn test_bandit_refine_tier_demotes_on_low_reward() {
        use super::ShardTierBandit;
        let mut bandit = ShardTierBandit::new();

        // Train: Cold tier gets high reward, Hot gets low reward.
        for _ in 0..10 {
            bandit.update_reward(FreqTier::Cold, 1.0);
            bandit.update_reward(FreqTier::Hot, 0.1);
        }

        // Warm should be demoted toward Cold (bandit prefers Cold).
        let refined = bandit.refine_tier(FreqTier::Warm);
        assert_eq!(refined, FreqTier::Cold, "Warm should be demoted to Cold");
    }

    #[cfg(feature = "freq_bandit")]
    #[test]
    fn test_bandit_two_level_shift_clamped() {
        use super::ShardTierBandit;
        let mut bandit = ShardTierBandit::new();

        // Train: Hot gets high reward, Cold gets low.
        for _ in 0..10 {
            bandit.update_reward(FreqTier::Hot, 1.0);
            bandit.update_reward(FreqTier::Cold, 0.0);
        }

        // Cold → Hot is a 2-level jump; bandit should clamp to Warm (one step).
        let refined = bandit.refine_tier(FreqTier::Cold);
        assert_eq!(
            refined,
            FreqTier::Warm,
            "2-level shift should be clamped to 1 step"
        );
    }

    #[cfg(feature = "freq_bandit")]
    #[test]
    fn test_bandit_integration_with_lfu_shard() {
        let mut lfu = BfcpLfuShard::new(10, 4);
        let logits = [1.0f32, 2.0, 3.0];

        // First process — miss, inserts Cold tier.
        let (partition, assignments) =
            lfu.process_and_shard(&logits, |_| make_partition(50, 30, 20));

        assert_eq!(partition.region_count(), 3);
        for &(shard, tier) in &assignments {
            assert!(shard < 4, "shard {shard} must be < 4");
            // Bandit hasn't been trained yet, so tier is static Cold.
            assert_eq!(
                tier,
                FreqTier::Cold,
                "untrained bandit should return static tier"
            );
        }

        // Train the bandit: Hot tier gets high rewards.
        for _ in 0..5 {
            lfu.update_bandit_reward(FreqTier::Hot, 1.0);
            lfu.update_bandit_reward(FreqTier::Cold, 0.2);
        }

        assert!(
            lfu.tier_bandit().total_pulls() > 0,
            "bandit should have pulls"
        );
    }

    #[cfg(feature = "freq_bandit")]
    #[test]
    fn test_derive_tier_with_bandit() {
        let mut lfu = BfcpLfuShard::new(10, 4);
        let logits = [1.0f32, 2.0, 3.0];

        // Process to populate cache.
        let _ = lfu.process(&logits, |_| make_partition(50, 30, 20));
        let hash = blake3_logit_hash(&logits);

        // Before bandit training: derive_tier returns static classification.
        let tier = lfu.derive_tier(&hash);
        assert_eq!(tier, FreqTier::Cold, "untrained should be Cold");

        // Train bandit heavily toward Hot.
        for _ in 0..20 {
            lfu.update_bandit_reward(FreqTier::Hot, 1.0);
        }

        // derive_tier should now refine based on bandit.
        let refined = lfu.derive_tier(&hash);
        // Static is Cold, bandit prefers Hot → 2-level shift clamped to Warm.
        assert_eq!(
            refined,
            FreqTier::Warm,
            "bandit should shift Cold toward Hot, clamped to Warm"
        );
    }

    #[cfg(feature = "manifold_pruner")]
    #[test]
    fn test_region_radius_hot_wider_than_cold() {
        let base = 1.0f32;
        let scale = 10.0f32;
        let hot = region_radius(base, 100.0, scale);
        let cold = region_radius(base, 1.0, scale);
        assert!(
            hot > cold,
            "hot radius ({hot}) should be > cold radius ({cold})"
        );
    }

    #[cfg(feature = "manifold_pruner")]
    #[test]
    fn test_region_radius_zero_freq_is_half_base() {
        let base = 2.0f32;
        let r = region_radius(base, 0.0, 10.0);
        // sigmoid(0) = 0.5
        assert!(
            (r - 1.0).abs() < 1e-6,
            "zero freq should give base * 0.5 = 1.0, got {r}"
        );
    }

    // Helper for tests — mirrors the module-level sigmoid.
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}
