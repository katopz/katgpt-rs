//! Frequency-aware region sharding for BFCF partitions (Plan 218 Phase 2).
//!
//! Maps (RegionLabel × FreqTier) → shard index using a flat `[AtomicUsize; 9]` array.
//! Hot tier pins to shard 0, Cold to last shard, Warm round-robins across middle.
//! Below `min_regions_for_shard` (30), sequential processing is preferred.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::bfcf_types::RegionLabel;
use super::bfcp_region_cache::FreqTier;

// ── RegionShardMap ─────────────────────────────────────────────

/// Sentinel value for uninitialized shard entries.
const UNASSIGNED: usize = usize::MAX;
const SHARD_SLOTS: usize = 9;

/// Flat index into the 9-entry shard array: `label * 3 + tier`.
#[inline]
fn shard_index(label: RegionLabel, tier: FreqTier) -> usize {
    (label as usize) * 3 + (tier as usize)
}

/// Shard assignment: (RegionLabel × FreqTier) → preferred shard index.
///
/// Uses a flat `[AtomicUsize; 9]` instead of HashMap — O(1) array indexing
/// with no hashing overhead for the fixed 3×3 grid.
pub struct RegionShardMap {
    /// Number of available shards (worker threads).
    num_shards: usize,
    /// Flat assignment array: index = label * 3 + tier → shard index.
    assignment: [AtomicUsize; SHARD_SLOTS],
    /// Round-robin counter for warm-tier distribution.
    rr_counter: AtomicUsize,
}

/// Minimum regions before sharding activates.
const MIN_REGIONS_FOR_SHARD: usize = 30;

impl RegionShardMap {
    /// Create with default assignment rules:
    /// - Hot → shard 0 (pinned)
    /// - Warm → round-robin across shards 1..num_shards
    /// - Cold → shard num_shards-1 (lazy worker)
    pub fn new(num_shards: usize) -> Self {
        let num_shards = num_shards.max(1);
        let map = Self {
            num_shards,
            assignment: [const { AtomicUsize::new(UNASSIGNED) }; SHARD_SLOTS],
            rr_counter: AtomicUsize::new(0),
        };
        map.populate_default();
        map
    }

    /// Alternative constructor: round-robin for all tiers.
    pub fn with_round_robin(num_shards: usize) -> Self {
        let num_shards = num_shards.max(1);
        let map = Self {
            num_shards,
            assignment: [const { AtomicUsize::new(UNASSIGNED) }; SHARD_SLOTS],
            rr_counter: AtomicUsize::new(0),
        };
        map.populate_round_robin();
        map
    }

    /// Assign shard for a region given its label and frequency tier.
    /// Falls back to `label as usize % num_shards` if entry is uninitialized.
    pub fn assign_shard(&self, label: RegionLabel, tier: FreqTier) -> usize {
        let shard = self.assignment[shard_index(label, tier)].load(Ordering::Relaxed);
        if shard == UNASSIGNED {
            (label as usize) % self.num_shards
        } else {
            shard
        }
    }

    /// Rebalance shard assignments when tiers change.
    /// - Hot → shard 0
    /// - Cold → last shard
    /// - Warm → round-robin among middle shards
    pub fn rebalance(&mut self, transitions: &[(RegionLabel, FreqTier, FreqTier)]) {
        for &(label, _old_tier, new_tier) in transitions {
            let shard = match new_tier {
                FreqTier::Hot => 0,
                FreqTier::Cold => {
                    let last = self.num_shards.saturating_sub(1);
                    // Avoid shard 0 if possible (reserved for hot).
                    match last {
                        0 => 0,
                        _ => last,
                    }
                }
                FreqTier::Warm => self.next_warm_shard(),
            };
            self.assignment[shard_index(label, new_tier)].store(shard, Ordering::Relaxed);
        }
    }

    /// Minimum region count to activate sharding.
    pub fn min_regions_for_shard(&self) -> usize {
        MIN_REGIONS_FOR_SHARD
    }

    /// Number of configured shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Whether to shard given current region count.
    pub fn should_shard(&self, region_count: usize) -> bool {
        region_count >= self.min_regions_for_shard()
    }

    // ── Internal helpers ────────────────────────────────────────

    fn populate_default(&self) {
        let labels = [RegionLabel::Accept, RegionLabel::Reject, RegionLabel::Maybe];
        let tiers = [FreqTier::Hot, FreqTier::Warm, FreqTier::Cold];

        for &label in &labels {
            for &tier in &tiers {
                let shard = match tier {
                    FreqTier::Hot => 0,
                    FreqTier::Cold => self.num_shards.saturating_sub(1),
                    FreqTier::Warm => self.next_warm_shard(),
                };
                self.assignment[shard_index(label, tier)].store(shard, Ordering::Relaxed);
            }
        }
    }

    fn populate_round_robin(&self) {
        let labels = [RegionLabel::Accept, RegionLabel::Reject, RegionLabel::Maybe];
        let tiers = [FreqTier::Hot, FreqTier::Warm, FreqTier::Cold];

        for &label in &labels {
            for &tier in &tiers {
                let shard = self.next_warm_shard();
                self.assignment[shard_index(label, tier)].store(shard, Ordering::Relaxed);
            }
        }
    }

    /// Next shard for warm-tier round-robin across 1..num_shards (or 0..num_shards if single shard).
    fn next_warm_shard(&self) -> usize {
        match self.num_shards {
            0 => 0,
            1 => 0,
            n => {
                let counter = self.rr_counter.fetch_add(1, Ordering::Relaxed);
                // Distribute across shards 1..n (skip shard 0 for hot).
                1 + counter % (n - 1)
            }
        }
    }
}

// ── RegionSharding trait ───────────────────────────────────────

/// Extension trait for frequency-aware region sharding.
pub trait RegionSharding: Send + Sync {
    /// Assign shard for a region given its label and frequency tier.
    fn assign_shard(&self, label: RegionLabel, tier: FreqTier) -> usize;
    /// Rebalance shard assignments when tiers change.
    fn rebalance(&mut self, transitions: &[(RegionLabel, FreqTier, FreqTier)]);
    /// Minimum region count to activate sharding (below: sequential).
    fn min_regions_for_shard(&self) -> usize;
}

impl RegionSharding for RegionShardMap {
    fn assign_shard(&self, label: RegionLabel, tier: FreqTier) -> usize {
        self.assign_shard(label, tier)
    }

    fn rebalance(&mut self, transitions: &[(RegionLabel, FreqTier, FreqTier)]) {
        self.rebalance(transitions)
    }

    fn min_regions_for_shard(&self) -> usize {
        self.min_regions_for_shard()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_assignment_hot() {
        let map = RegionShardMap::new(4);
        for &label in &[RegionLabel::Accept, RegionLabel::Reject, RegionLabel::Maybe] {
            assert_eq!(
                map.assign_shard(label, FreqTier::Hot),
                0,
                "Hot tier should map to shard 0 for {label:?}"
            );
        }
    }

    #[test]
    fn test_default_assignment_cold() {
        let num_shards = 4;
        let map = RegionShardMap::new(num_shards);
        for &label in &[RegionLabel::Accept, RegionLabel::Reject, RegionLabel::Maybe] {
            assert_eq!(
                map.assign_shard(label, FreqTier::Cold),
                num_shards - 1,
                "Cold tier should map to last shard for {label:?}"
            );
        }
    }

    #[test]
    fn test_default_assignment_warm_round_robin() {
        let num_shards = 4;
        let map = RegionShardMap::new(num_shards);

        // Collect assignments for all (label, Warm) combos.
        let shards: Vec<usize> = [RegionLabel::Accept, RegionLabel::Reject, RegionLabel::Maybe]
            .iter()
            .map(|&label| map.assign_shard(label, FreqTier::Warm))
            .collect();

        // All warm shards should be in 1..num_shards (not 0, not last for cold).
        for &s in &shards {
            assert!(
                s > 0 && s < num_shards,
                "Warm shard {s} should be in 1..{num_shards}"
            );
        }
    }

    #[test]
    fn test_assign_shard_all_combinations() {
        let num_shards = 8;
        let map = RegionShardMap::new(num_shards);
        let labels = [RegionLabel::Accept, RegionLabel::Reject, RegionLabel::Maybe];
        let tiers = [FreqTier::Hot, FreqTier::Warm, FreqTier::Cold];

        for &label in &labels {
            for &tier in &tiers {
                let shard = map.assign_shard(label, tier);
                assert!(
                    shard < num_shards,
                    "shard {shard} must be < {num_shards} for ({label:?}, {tier:?})"
                );
            }
        }
    }

    #[test]
    fn test_rebalance_transitions() {
        let num_shards = 4;
        let mut map = RegionShardMap::new(num_shards);

        // Warm → Hot: should move to shard 0.
        map.rebalance(&[(RegionLabel::Accept, FreqTier::Warm, FreqTier::Hot)]);
        assert_eq!(map.assign_shard(RegionLabel::Accept, FreqTier::Hot), 0);

        // Hot → Cold: should move to last shard.
        map.rebalance(&[(RegionLabel::Reject, FreqTier::Hot, FreqTier::Cold)]);
        assert_eq!(
            map.assign_shard(RegionLabel::Reject, FreqTier::Cold),
            num_shards - 1
        );
    }

    #[test]
    fn test_should_shard_below_threshold() {
        let map = RegionShardMap::new(4);
        assert!(!map.should_shard(29), "29 regions should not shard");
        assert!(!map.should_shard(0), "0 regions should not shard");
    }

    #[test]
    fn test_should_shard_above_threshold() {
        let map = RegionShardMap::new(4);
        assert!(map.should_shard(30), "30 regions should shard");
        assert!(map.should_shard(100), "100 regions should shard");
    }

    #[test]
    fn test_min_regions_for_shard() {
        let map = RegionShardMap::new(4);
        assert_eq!(map.min_regions_for_shard(), 30);
    }

    #[test]
    fn test_shard_index_formula() {
        // Verify the flat index formula matches (label, tier) pairs.
        assert_eq!(shard_index(RegionLabel::Accept, FreqTier::Hot), 0);
        assert_eq!(shard_index(RegionLabel::Accept, FreqTier::Warm), 1);
        assert_eq!(shard_index(RegionLabel::Accept, FreqTier::Cold), 2);
        assert_eq!(shard_index(RegionLabel::Reject, FreqTier::Hot), 3);
        assert_eq!(shard_index(RegionLabel::Reject, FreqTier::Warm), 4);
        assert_eq!(shard_index(RegionLabel::Reject, FreqTier::Cold), 5);
        assert_eq!(shard_index(RegionLabel::Maybe, FreqTier::Hot), 6);
        assert_eq!(shard_index(RegionLabel::Maybe, FreqTier::Warm), 7);
        assert_eq!(shard_index(RegionLabel::Maybe, FreqTier::Cold), 8);
    }
}

// TL;DR: Frequency-aware region sharding for BFCF partitions. (RegionLabel × FreqTier) → shard
// via flat [AtomicUsize; 9] array indexed by label*3+tier. Hot pinned to shard 0, Cold to last
// shard, Warm round-robin. Sequential fallback when regions < 30. Plan 218 Phase 2.
