//! Latent Transition Cache — LRU cache for (h_t, x_{t+1}) → ĥ_{t+1} MLP predictions.
//!
//! Avoids redundant MLP forward calls when the same hidden-state + token pairs
//! recur during DDTree branch exploration. Uses papaya lock-free HashMap for
//! concurrent access with bounded eviction.
//!
//! Plan 217 Phase 4.

use std::sync::atomic::{AtomicU64, Ordering};

/// A compact key for the cache: blake3 hash of (h_t || next_emb).
/// Truncated to 16 bytes for compact storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CacheKey([u8; 16]);

impl CacheKey {
    fn from_slices(h_t: &[f32], next_emb: &[f32]) -> Self {
        let mut hasher = blake3::Hasher::new();
        // SAFETY: we're reading the byte representation of &[f32] — no alignment
        // issues on x86/ARM since f32 and u8 share address boundaries. We only
        // read, never write through these pointers.
        unsafe {
            let h_bytes =
                std::slice::from_raw_parts(h_t.as_ptr() as *const u8, std::mem::size_of_val(h_t));
            hasher.update(h_bytes);
            let emb_bytes = std::slice::from_raw_parts(
                next_emb.as_ptr() as *const u8,
                std::mem::size_of_val(next_emb),
            );
            hasher.update(emb_bytes);
        }
        let hash = hasher.finalize();
        let mut key = [0u8; 16];
        key.copy_from_slice(&hash.as_bytes()[..16]);
        CacheKey(key)
    }
}

/// Entry in the latent transition cache.
#[derive(Clone)]
struct CacheEntry {
    /// The predicted next hidden state.
    h_next: Vec<f32>,
    /// Access counter for LRU eviction (monotonically increasing).
    access_count: u64,
}

/// Lock-free LRU cache for latent transition predictions.
///
/// Uses [`papaya::HashMap`] for concurrent access. Eviction is approximate LRU
/// based on access counts. The cache has a fixed maximum capacity; when full,
/// entries with the lowest access count are evicted.
///
/// # Thread Safety
/// papaya::HashMap is lock-free and thread-safe. No `Mutex` needed.
pub struct LatentTransitionCache {
    /// The underlying lock-free hash map.
    map: papaya::HashMap<CacheKey, CacheEntry>,
    /// Maximum number of entries.
    capacity: usize,
    /// Monotonically increasing access counter for LRU approximation.
    counter: AtomicU64,
    /// Number of cache hits (for diagnostics).
    hits: AtomicU64,
    /// Number of cache misses (for diagnostics).
    misses: AtomicU64,
}

impl LatentTransitionCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            map: papaya::HashMap::new(),
            capacity: capacity.max(1),
            counter: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Look up a cached transition. Returns `Some(h_next)` on hit, `None` on miss.
    pub fn get(&self, h_t: &[f32], next_emb: &[f32]) -> Option<Vec<f32>> {
        let key = CacheKey::from_slices(h_t, next_emb);
        let map = self.map.pin();
        match map.get(&key) {
            Some(entry) => {
                self.counter.fetch_add(1, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.h_next.clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Insert a cached transition.
    /// If the cache is full, evicts an approximate-LRU entry.
    pub fn insert(&self, h_t: &[f32], next_emb: &[f32], h_next: Vec<f32>) {
        let key = CacheKey::from_slices(h_t, next_emb);
        let count = self.counter.fetch_add(1, Ordering::Relaxed);

        let map = self.map.pin();

        // Evict if over capacity
        if map.len() >= self.capacity {
            let mut min_key: Option<CacheKey> = None;
            let mut min_count = u64::MAX;
            for (k, v) in map.iter() {
                if v.access_count < min_count {
                    min_count = v.access_count;
                    min_key = Some(*k);
                }
            }
            if let Some(evict_key) = min_key {
                let _ = map.remove(&evict_key);
            }
        }

        let _ = map.insert(
            key,
            CacheEntry {
                h_next,
                access_count: count,
            },
        );
    }

    /// Get or compute a cached transition.
    /// On cache miss, calls `compute` to get the result, caches it, and returns it.
    pub fn get_or_insert<F>(&self, h_t: &[f32], next_emb: &[f32], compute: F) -> Vec<f32>
    where
        F: FnOnce() -> Vec<f32>,
    {
        match self.get(h_t, next_emb) {
            Some(h_next) => h_next,
            None => {
                let h_next = compute();
                self.insert(h_t, next_emb, h_next.clone());
                h_next
            }
        }
    }

    /// Clear the cache and reset hit/miss counters.
    pub fn clear(&self) {
        self.map.pin().clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.map.pin().len()
    }

    /// Is the cache empty?
    pub fn is_empty(&self) -> bool {
        self.map.pin().is_empty()
    }

    /// Cache hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f32 / total as f32
        }
    }

    /// Number of cache hits.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Number of cache misses.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_h(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims).map(|i| seed + i as f32 * 0.1).collect()
    }

    fn make_emb(dims: usize, seed: f32) -> Vec<f32> {
        (0..dims).map(|i| seed * 2.0 + i as f32 * 0.05).collect()
    }

    #[test]
    fn test_cache_insert_and_get() {
        let cache = LatentTransitionCache::new(16);
        let h = make_h(16, 1.0);
        let emb = make_emb(16, 2.0);
        let h_next: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();

        cache.insert(&h, &emb, h_next.clone());

        let result = cache.get(&h, &emb);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), h_next);
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let cache = LatentTransitionCache::new(16);
        let h = make_h(16, 1.0);
        let emb = make_emb(16, 2.0);

        assert!(cache.get(&h, &emb).is_none());
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
    }

    #[test]
    fn test_cache_eviction_when_full() {
        let cache = LatentTransitionCache::new(2);
        let h1 = make_h(8, 1.0);
        let emb1 = make_emb(8, 10.0);
        let h2 = make_h(8, 2.0);
        let emb2 = make_emb(8, 20.0);
        let h3 = make_h(8, 3.0);
        let emb3 = make_emb(8, 30.0);

        let val1: Vec<f32> = vec![1.0; 8];
        let val2: Vec<f32> = vec![2.0; 8];
        let val3: Vec<f32> = vec![3.0; 8];

        cache.insert(&h1, &emb1, val1);
        cache.insert(&h2, &emb2, val2);
        // Cache is now full (capacity=2). Inserting h3 should evict the oldest (h1).
        cache.insert(&h3, &emb3, val3);

        assert_eq!(cache.len(), 2);
        // h1 was evicted (lowest access count)
        assert!(cache.get(&h1, &emb1).is_none());
        // h2 and h3 should still be present
        assert!(cache.get(&h2, &emb2).is_some());
        assert!(cache.get(&h3, &emb3).is_some());
    }

    #[test]
    fn test_cache_get_or_insert() {
        let cache = LatentTransitionCache::new(16);
        let h = make_h(8, 1.0);
        let emb = make_emb(8, 2.0);
        let computed: Vec<f32> = vec![42.0; 8];

        // Miss — should call compute
        let result = cache.get_or_insert(&h, &emb, || computed.clone());
        assert_eq!(result, computed);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);

        // Hit — should NOT call compute (returns cached)
        let result2 = cache.get_or_insert(&h, &emb, || vec![999.0; 8]);
        assert_eq!(result2, computed); // not the 999.0 value
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let cache = LatentTransitionCache::new(16);
        let h = make_h(4, 1.0);
        let emb = make_emb(4, 2.0);
        let h_next = vec![0.0; 4];

        // 1 miss
        cache.get(&h, &emb);
        // Insert and get 2 hits
        cache.insert(&h, &emb, h_next);
        cache.get(&h, &emb);
        cache.get(&h, &emb);

        let rate = cache.hit_rate();
        assert!(
            (rate - 2.0 / 3.0).abs() < 1e-6,
            "hit_rate should be ~0.667, got {rate}"
        );
    }

    #[test]
    fn test_cache_clear() {
        let cache = LatentTransitionCache::new(16);
        let h = make_h(4, 1.0);
        let emb = make_emb(4, 2.0);
        let h_next = vec![0.0; 4];

        cache.insert(&h, &emb, h_next);
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_cache_different_keys() {
        let cache = LatentTransitionCache::new(16);
        let h1 = make_h(8, 1.0);
        let emb1 = make_emb(8, 10.0);
        let h2 = make_h(8, 99.0);
        let emb2 = make_emb(8, 88.0);
        let val1: Vec<f32> = vec![1.0; 8];
        let val2: Vec<f32> = vec![2.0; 8];

        cache.insert(&h1, &emb1, val1.clone());
        cache.insert(&h2, &emb2, val2.clone());

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&h1, &emb1).unwrap(), val1);
        assert_eq!(cache.get(&h2, &emb2).unwrap(), val2);
    }

    #[test]
    fn test_cache_same_key_overwrites() {
        let cache = LatentTransitionCache::new(16);
        let h = make_h(8, 1.0);
        let emb = make_emb(8, 2.0);
        let val_old: Vec<f32> = vec![1.0; 8];
        let val_new: Vec<f32> = vec![2.0; 8];

        cache.insert(&h, &emb, val_old);
        cache.insert(&h, &emb, val_new.clone());

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&h, &emb).unwrap(), val_new);
    }

    #[test]
    fn test_cache_key_deterministic() {
        let h = make_h(16, 1.0);
        let emb = make_emb(16, 2.0);

        let key1 = CacheKey::from_slices(&h, &emb);
        let key2 = CacheKey::from_slices(&h, &emb);
        assert_eq!(key1, key2, "same input must produce same key");

        let h_diff = make_h(16, 99.0);
        let key3 = CacheKey::from_slices(&h_diff, &emb);
        assert_ne!(key1, key3, "different input must produce different key");
    }
}
