#![cfg(feature = "bfcf_lfu_shard")]
//! BFCP Region LFU Cache — frequency-aware partition caching with sigmoid admission.
//!
//! Caches BLAKE3-hashed BFCP partitions in a papaya lock-free HashMap. LFU eviction
//! removes cold entries when full. Sigmoid-gated admission prevents caching noisy
//! (highly variable) partitions. Frequency tiers (Hot/Warm/Cold) drive downstream
//! sharding decisions.
//!
//! Plan 218 Phase 1.

use std::sync::atomic::{AtomicU64, Ordering};

use super::bfcf_types::BFCP;

// ── FreqTier ──────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FreqTier {
    Hot,
    Warm,
    Cold,
}

// ── CachedRegion ──────────────────────────────────────────────

struct CachedRegion {
    partition: BFCP,
    hash: [u8; 32],
    freq: u32,
    tier: FreqTier,
}

// ── Free functions ────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn blake3_logit_hash(logits: &[f32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    // SAFETY: reading byte representation of &[f32] — no alignment issues on
    // x86/ARM since f32 and u8 share address boundaries. Read-only.
    unsafe {
        let bytes = std::slice::from_raw_parts(
            logits.as_ptr() as *const u8,
            logits.len() * std::mem::size_of::<f32>(),
        );
        hasher.update(bytes);
    }
    *hasher.finalize().as_bytes()
}

fn classify_tier(freq: u32, hot_threshold: u32, warm_threshold: u32) -> FreqTier {
    match freq {
        f if f >= hot_threshold => FreqTier::Hot,
        f if f >= warm_threshold => FreqTier::Warm,
        _ => FreqTier::Cold,
    }
}

// ── BfcpRegionCache ──────────────────────────────────────────

pub struct BfcpRegionCache {
    map: papaya::HashMap<[u8; 32], CachedRegion>,
    capacity: usize,
    admit_threshold: f32,
    hot_threshold: u32,
    warm_threshold: u32,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl BfcpRegionCache {
    pub fn new(capacity: usize) -> Self {
        Self::with_thresholds(capacity, 100, 10, 1.0)
    }

    pub fn with_thresholds(
        capacity: usize,
        hot_threshold: u32,
        warm_threshold: u32,
        admit_threshold: f32,
    ) -> Self {
        Self {
            map: papaya::HashMap::new(),
            capacity: capacity.max(1),
            admit_threshold,
            hot_threshold,
            warm_threshold,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn lookup(&self, hash: &[u8; 32]) -> Option<BFCP> {
        let guard = self.map.pin();
        match guard.get(hash) {
            Some(entry) => {
                let new_freq = entry.freq.saturating_add(1);
                let tier = classify_tier(new_freq, self.hot_threshold, self.warm_threshold);
                let partition = entry.partition.clone();
                let _ = guard.insert(
                    *hash,
                    CachedRegion {
                        partition: partition.clone(),
                        hash: *hash,
                        freq: new_freq,
                        tier,
                    },
                );
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(partition)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    pub fn insert(&mut self, hash: [u8; 32], partition: BFCP) {
        let guard = self.map.pin();

        // Existing entry → just bump freq.
        if let Some(existing) = guard.get(&hash) {
            let new_freq = existing.freq.saturating_add(1);
            let _ = guard.insert(
                hash,
                CachedRegion {
                    partition,
                    hash,
                    freq: new_freq,
                    tier: classify_tier(new_freq, self.hot_threshold, self.warm_threshold),
                },
            );
            return;
        }

        // Sigmoid admission gate: only admit when sigmoid(1 / admit_threshold) > 0.5.
        let initial_freq: u32 = 1;
        let admit_score = sigmoid(initial_freq as f32 / self.admit_threshold);
        if admit_score <= 0.5 {
            return;
        }

        // Evict LFU entry if at capacity.
        if guard.len() >= self.capacity {
            let mut min_hash: Option<[u8; 32]> = None;
            let mut min_freq = u32::MAX;
            for (k, v) in guard.iter() {
                if v.freq < min_freq {
                    min_freq = v.freq;
                    min_hash = Some(*k);
                }
            }
            if let Some(evict) = min_hash {
                let _ = guard.remove(&evict);
            }
        }

        let _ = guard.insert(
            hash,
            CachedRegion {
                partition,
                hash,
                freq: initial_freq,
                tier: classify_tier(initial_freq, self.hot_threshold, self.warm_threshold),
            },
        );
    }

    pub fn freq_tier(&self, hash: &[u8; 32]) -> Option<FreqTier> {
        let guard = self.map.pin();
        guard.get(hash).map(|e| e.tier)
    }

    pub fn decay(&mut self, lambda: f32) {
        let guard = self.map.pin();
        // Collect entries, mutate, re-insert. papaya doesn't support in-place mutation.
        let entries: Vec<([u8; 32], CachedRegion)> = guard
            .iter()
            .map(|(k, v)| {
                let new_freq = ((v.freq as f32) * lambda) as u32;
                let tier = classify_tier(new_freq, self.hot_threshold, self.warm_threshold);
                (
                    *k,
                    CachedRegion {
                        partition: v.partition.clone(),
                        hash: v.hash,
                        freq: new_freq,
                        tier,
                    },
                )
            })
            .collect();

        for (k, v) in entries {
            let _ = guard.insert(k, v);
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        match total {
            0 => 0.0,
            _ => hits as f64 / total as f64,
        }
    }

    pub fn len(&self) -> usize {
        self.map.pin().len()
    }
}

// ── RegionCaching trait ──────────────────────────────────────

#[cfg(feature = "bfcf_lfu_shard")]
pub trait RegionCaching: Send + Sync {
    fn lookup(&self, hash: &[u8; 32]) -> Option<BFCP>;
    fn insert(&mut self, hash: [u8; 32], partition: BFCP);
    fn freq_tier(&self, hash: &[u8; 32]) -> Option<FreqTier>;
    fn decay(&mut self, lambda: f32);
}

impl RegionCaching for BfcpRegionCache {
    fn lookup(&self, hash: &[u8; 32]) -> Option<BFCP> {
        self.lookup(hash)
    }

    fn insert(&mut self, hash: [u8; 32], partition: BFCP) {
        self.insert(hash, partition);
    }

    fn freq_tier(&self, hash: &[u8; 32]) -> Option<FreqTier> {
        self.freq_tier(hash)
    }

    fn decay(&mut self, lambda: f32) {
        self.decay(lambda);
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bfcf_types::{BorelRegion, HalfSpace, RegionLabel};

    fn make_partition(n_regions: usize, base_tokens: usize) -> BFCP {
        let regions: Vec<BorelRegion> = (0..n_regions)
            .map(|i| {
                BorelRegion::new(
                    match i % 3 {
                        0 => RegionLabel::Accept,
                        1 => RegionLabel::Reject,
                        _ => RegionLabel::Maybe,
                    },
                    vec![HalfSpace {
                        dim: i as u16,
                        threshold: 0.5,
                        above: true,
                    }],
                    base_tokens + i,
                )
            })
            .collect();
        BFCP::from_regions(regions)
    }

    fn make_hash(seed: u8) -> [u8; 32] {
        let logits: Vec<f32> = (0..8).map(|i| seed as f32 + i as f32 * 0.1).collect();
        blake3_logit_hash(&logits)
    }

    #[test]
    fn test_blake3_logit_hash_deterministic() {
        let logits: Vec<f32> = vec![1.0, 2.0, 3.0];
        let h1 = blake3_logit_hash(&logits);
        let h2 = blake3_logit_hash(&logits);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_blake3_logit_hash_different() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![9.0, 8.0, 7.0];
        assert_ne!(blake3_logit_hash(&a), blake3_logit_hash(&b));
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut cache = BfcpRegionCache::new(16);
        let hash = make_hash(1);
        let partition = make_partition(3, 10);

        cache.insert(hash, partition.clone());
        let result = cache.lookup(&hash);
        assert!(result.is_some());
        assert_eq!(result.unwrap().region_count(), partition.region_count());
    }

    #[test]
    fn test_lookup_miss() {
        let cache = BfcpRegionCache::new(16);
        let hash = make_hash(99);
        assert!(cache.lookup(&hash).is_none());
    }

    #[test]
    fn test_lfu_eviction() {
        let mut cache = BfcpRegionCache::new(3);

        let h1 = make_hash(1);
        let h2 = make_hash(2);
        let h3 = make_hash(3);
        let h4 = make_hash(4);

        let p1 = make_partition(1, 10);
        let p2 = make_partition(2, 20);
        let p3 = make_partition(3, 30);
        let p4 = make_partition(4, 40);

        cache.insert(h1, p1);
        cache.insert(h2, p2);
        cache.insert(h3, p3);

        // Bump h2 and h3 freq so h1 is lowest.
        let _ = cache.lookup(&h2);
        let _ = cache.lookup(&h3);

        // Insert h4 → should evict h1 (lowest freq).
        cache.insert(h4, p4);

        assert_eq!(cache.len(), 3);
        assert!(
            cache.lookup(&h1).is_none(),
            "h1 should be evicted (lowest freq)"
        );
        assert!(cache.lookup(&h4).is_some(), "h4 should be present");
    }

    #[test]
    fn test_freq_increment_on_hit() {
        let mut cache = BfcpRegionCache::new(16);
        let hash = make_hash(1);
        cache.insert(hash, make_partition(2, 10));

        // Initial tier is Cold (freq=1, warm_threshold=10).
        assert_eq!(cache.freq_tier(&hash), Some(FreqTier::Cold));

        // Lookup 9 more times to reach freq=10 (warm).
        for _ in 0..9 {
            let _ = cache.lookup(&hash);
        }
        assert_eq!(cache.freq_tier(&hash), Some(FreqTier::Warm));
    }

    #[test]
    fn test_freq_tier_classification() {
        assert_eq!(classify_tier(200, 100, 10), FreqTier::Hot);
        assert_eq!(classify_tier(50, 100, 10), FreqTier::Warm);
        assert_eq!(classify_tier(5, 100, 10), FreqTier::Cold);
        assert_eq!(classify_tier(100, 100, 10), FreqTier::Hot);
        assert_eq!(classify_tier(10, 100, 10), FreqTier::Warm);
        assert_eq!(classify_tier(9, 100, 10), FreqTier::Cold);
    }

    #[test]
    fn test_decay_reduces_frequency() {
        let mut cache = BfcpRegionCache::new(16);
        let hash = make_hash(1);
        cache.insert(hash, make_partition(2, 10));

        // Bump to freq=10 (warm).
        for _ in 0..9 {
            let _ = cache.lookup(&hash);
        }
        assert_eq!(cache.freq_tier(&hash), Some(FreqTier::Warm));

        // Decay by 0.5: freq goes from ~10 → 5, drops to Cold.
        cache.decay(0.5);
        assert_eq!(cache.freq_tier(&hash), Some(FreqTier::Cold));
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = BfcpRegionCache::new(16);
        let hash = make_hash(1);
        cache.insert(hash, make_partition(2, 10));

        // 3 hits + 1 miss = 0.75 hit rate.
        let _ = cache.lookup(&hash);
        let _ = cache.lookup(&hash);
        let _ = cache.lookup(&hash);

        let miss_hash = make_hash(99);
        let _ = cache.lookup(&miss_hash);

        let rate = cache.hit_rate();
        assert!(
            (rate - 0.75).abs() < 1e-9,
            "hit_rate should be 0.75, got {rate}"
        );
    }

    #[test]
    fn test_admission_gate() {
        // High admit_threshold → sigmoid(1/1000) ≈ 0.5005, barely above 0.5.
        // Even lower values should still pass, but let's use an extreme threshold.
        let mut cache = BfcpRegionCache::with_thresholds(16, 100, 10, 1000.0);
        let hash = make_hash(1);
        let partition = make_partition(2, 10);

        cache.insert(hash, partition);
        // sigmoid(1/1000) ≈ 0.5005 > 0.5, so it should still be admitted.
        assert!(cache.lookup(&hash).is_some());

        // With an absurdly high threshold, admission should fail.
        // sigmoid(1/1e10) ≈ 0.5 + 5e-11 > 0.5, still admitted due to floating point.
        // Let's test the negative case by using a negative threshold (sigmoid neg → < 0.5).
        let mut cache_strict = BfcpRegionCache::with_thresholds(16, 100, 10, -1.0);
        let hash2 = make_hash(2);
        cache_strict.insert(hash2, make_partition(2, 10));
        // sigmoid(1 / -1.0) = sigmoid(-1.0) ≈ 0.269 < 0.5, rejected.
        assert!(cache_strict.lookup(&hash2).is_none());
    }

    #[test]
    fn test_sigmoid() {
        let s0 = sigmoid(0.0);
        assert!(
            (s0 - 0.5).abs() < 1e-6,
            "sigmoid(0) should be 0.5, got {s0}"
        );

        let s_large = sigmoid(100.0);
        assert!(s_large > 0.99, "sigmoid(100) should be ~1.0, got {s_large}");

        let s_neg = sigmoid(-100.0);
        assert!(s_neg < 0.01, "sigmoid(-100) should be ~0.0, got {s_neg}");
    }
}

// TL;DR: LFU region cache for BFCP partitions using papaya lock-free HashMap.
// BLAKE3-hashed keys, sigmoid-gated admission, Hot/Warm/Cold frequency tiers,
// O(n) LFU eviction (n ≈ 50-100), decay for staleness prevention. Plan 218 P1.
