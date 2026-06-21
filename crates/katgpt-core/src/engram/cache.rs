//! `ZipfianCacheHierarchy` — plasma/hot/warm/cold tiered pattern cache.
//!
//! Plan 299 Phase 6 T6.1–T6.7. Real Engram deployments have Zipfian N-gram
//! distributions (a small number of suffixes account for the vast majority
//! of lookups). The paper §2.5 + §6.4 describes a multi-level cache
//! hierarchy: HBM (plasma) → host DRAM (hot/warm) → NVMe/network (cold).
//!
//! This module implements a small, in-process version of that hierarchy:
//!
//! ```text
//! lookup(hash) →
//!   plasma tier (papaya LRU) ─────────────── HIT → return
//!                              │
//!                              MISS
//!                              ↓
//!   warm tier (warm_source.lookup_into) ──── HIT → promote to plasma, return
//!                              │
//!                              MISS
//!                              ↓
//!   cold tier (cold_fetcher.fetch) ───────── HIT → promote to plasma, return
//!                              │
//!                              MISS
//!                              ↓
//!                          return zero-filled
//! ```
//!
//! # CRITICAL — never softmax
//!
//! Per AGENTS.md this module contains **no `softmax` symbol**. It's pure
//! cache plumbing; the sigmoid gate lives in [`crate::engram::kernel`].
//!
//! # Latent vs raw boundary
//!
//! - The cached `Box<[f32]>` slot vectors are **latent** — they never sync.
//! - The [`CacheTier`] enum and [`ZipfianStats`] are diagnostics (raw, safe
//!   to surface to metrics systems).

use super::{EngramHash, EngramTable, K_MAX};
use papaya::HashMap as PapayaMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache tier that satisfied a lookup. `#[repr(u8)]` per AGENTS.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CacheTier {
    /// Plasma tier — in-process L1 / shared memory. Always checked first.
    Plasma = 0,
    /// Hot tier — HBM / fast host DRAM. Not used in the open primitive; the
    /// paper's HBM tier is host-specific. Kept for API completeness.
    Hot = 1,
    /// Warm tier — host DRAM backing source (an [`EngramTable`] trait
    /// object, typically the full frozen table).
    Warm = 2,
    /// Cold tier — NVMe / network fetch (a [`ColdFetcher`] implementation).
    Cold = 3,
}

/// Outcome of a cached lookup.
#[derive(Debug, Clone, Copy)]
pub struct CacheResult {
    /// Which tier satisfied the lookup. `None` if all tiers missed.
    pub tier: Option<CacheTier>,
    /// True iff the lookup found data (any tier). False = full miss,
    /// `out` is zero-filled.
    pub hit: bool,
}

/// Stats counters per tier. All atomic — readers/writers don't block on stats.
#[derive(Debug, Default)]
pub struct ZipfianStats {
    pub hits_plasma: AtomicU64,
    pub hits_hot: AtomicU64,
    pub hits_warm: AtomicU64,
    pub hits_cold: AtomicU64,
    pub misses: AtomicU64,
}

impl ZipfianStats {
    /// Snapshot all counters into a plain struct (for diagnostics).
    pub fn snapshot(&self) -> ZipfianStatsSnapshot {
        ZipfianStatsSnapshot {
            hits_plasma: self.hits_plasma.load(Ordering::Relaxed),
            hits_hot: self.hits_hot.load(Ordering::Relaxed),
            hits_warm: self.hits_warm.load(Ordering::Relaxed),
            hits_cold: self.hits_cold.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

/// Plain-struct snapshot of [`ZipfianStats`] (no atomics — for display/log).
#[derive(Debug, Clone, Copy, Default)]
pub struct ZipfianStatsSnapshot {
    pub hits_plasma: u64,
    pub hits_hot: u64,
    pub hits_warm: u64,
    pub hits_cold: u64,
    pub misses: u64,
}

impl ZipfianStatsSnapshot {
    /// Total lookups = sum of all hits + misses.
    pub fn total(&self) -> u64 {
        self.hits_plasma + self.hits_hot + self.hits_warm + self.hits_cold + self.misses
    }

    /// Plasma hit rate = hits_plasma / total (0.0 if no lookups yet).
    pub fn plasma_hit_rate(&self) -> f32 {
        let t = self.total();
        if t == 0 {
            return 0.0;
        }
        self.hits_plasma as f32 / t as f32
    }
}

/// Cold-tier fetcher — abstracts NVMe / network retrieval.
///
/// Implementations write the fetched slot vector into `out` (length `d`) and
/// return `true` on hit, `false` on miss. The fetcher is responsible for
/// any deserialization / network protocol; the open primitive just consumes
/// the bytes.
pub trait ColdFetcher: Send + Sync {
    /// Fetch the slot vector for `hash`. Writes into `out` (length `d`).
    /// Returns `true` on hit, `false` on miss (caller zero-fills `out`).
    fn fetch(&self, hash: EngramHash, out: &mut [f32]) -> bool;
}

/// Multi-tier cache wrapping an [`EngramTable`] warm source + optional cold fetcher.
///
/// The plasma tier is a [`papaya::HashMap`] with a generation-counter LRU
/// eviction policy. When the plasma tier is at capacity and a new entry is
/// promoted, the oldest-generation entry is evicted.
///
/// # Hot-path contract
///
/// [`lookup_cached`](Self::lookup_cached) is **mostly zero-allocation**: the
/// happy path (plasma hit) is a single hash-map lookup + slice copy. On a
/// miss that hits the warm or cold tier, a `Box<[f32]>` is allocated to
/// promote the entry into the plasma tier — this is acceptable (cache fill
/// is amortized over many future hits) but means the function is NOT
/// strictly zero-alloc. For strict zero-alloc hot paths, the caller should
/// bypass the cache and call `warm_source.lookup_into` directly with a
/// pre-allocated scratch buffer.
pub struct ZipfianCacheHierarchy {
    /// Plasma tier — lock-free hash map. Keyed by `EngramHash`, value is
    /// `(slot_vec, generation)` for LRU eviction.
    plasma: PapayaMap<EngramHash, (Box<[f32]>, u64)>,
    /// Plasma capacity (max entries before eviction).
    plasma_capacity: usize,
    /// Generation counter — incremented on each promotion. Used as the LRU
    /// timestamp (higher = more recently used).
    generation: AtomicU64,
    /// Warm tier — the backing frozen table. All slots are accessible here
    /// (the table is the source of truth).
    warm_source: Arc<dyn EngramTable>,
    /// Cold tier — optional NVMe/network fetcher. None means cold lookups
    /// always miss (which is fine if the warm_source covers the full
    /// vocabulary).
    cold_fetcher: Option<Arc<dyn ColdFetcher>>,
    /// Per-tier hit/miss counters.
    stats: ZipfianStats,
}

impl ZipfianCacheHierarchy {
    /// Construct a new cache hierarchy.
    ///
    /// - `plasma_capacity` — max entries in the plasma LRU. ~1024 is a
    ///   reasonable default for in-process use.
    /// - `warm_source` — the backing [`EngramTable`]. Wrapped in `Arc` so
    ///   multiple caches can share it.
    /// - `cold_fetcher` — optional. `None` disables the cold tier.
    pub fn new(
        plasma_capacity: usize,
        warm_source: Arc<dyn EngramTable>,
        cold_fetcher: Option<Arc<dyn ColdFetcher>>,
    ) -> Self {
        Self {
            plasma: PapayaMap::builder().build(),
            plasma_capacity,
            generation: AtomicU64::new(0),
            warm_source,
            cold_fetcher,
            stats: ZipfianStats::default(),
        }
    }

    /// Cached lookup. Writes the slot vector into `out` (length `d`).
    /// Returns the tier that satisfied the lookup (`None` on full miss).
    ///
    /// On hit at any tier below plasma, the entry is promoted to plasma
    /// (subject to capacity — oldest entry evicted if full).
    pub fn lookup_cached(&self, hash: EngramHash, d: usize, out: &mut [f32]) -> CacheResult {
        debug_assert_eq!(out.len(), d, "lookup_cached: out.len() must equal d");

        // ── Plasma tier ────────────────────────────────────────────────
        // papaya requires a `pin()` guard for all operations. `get` returns
        // a reference valid for the lifetime of the guard; we copy out and
        // drop the guard immediately.
        {
            let guard = self.plasma.pin();
            if let Some((slot, _gen_val)) = guard.get(&hash) {
                if slot.len() == d {
                    out.copy_from_slice(slot);
                    self.stats.hits_plasma.fetch_add(1, Ordering::Relaxed);
                    return CacheResult {
                        tier: Some(CacheTier::Plasma),
                        hit: true,
                    };
                }
                // Size mismatch — fall through to warm/cold. (Shouldn't
                // happen in practice; documented for robustness.)
            }
        }

        // ── Warm tier ──────────────────────────────────────────────────
        // The warm source is an EngramTable — its lookup_into fills K_MAX
        // slots. We construct a [EngramHash; K_MAX] with the requested hash
        // in slot 0 and zeros elsewhere, then read slot 0 of the output.
        let mut keys = [EngramHash(0); K_MAX];
        keys[0] = hash;
        let mut warm_out = vec![0.0f32; K_MAX * d];
        let hits = self.warm_source.lookup_into(&keys, &mut warm_out);
        if hits > 0 {
            // Slot 0 has data → copy it out + promote to plasma.
            out.copy_from_slice(&warm_out[..d]);
            self.stats.hits_warm.fetch_add(1, Ordering::Relaxed);
            self.promote_to_plasma(hash, out);
            return CacheResult {
                tier: Some(CacheTier::Warm),
                hit: true,
            };
        }

        // ── Cold tier ──────────────────────────────────────────────────
        if let Some(cold) = &self.cold_fetcher {
            if cold.fetch(hash, out) {
                self.stats.hits_cold.fetch_add(1, Ordering::Relaxed);
                self.promote_to_plasma(hash, out);
                return CacheResult {
                    tier: Some(CacheTier::Cold),
                    hit: true,
                };
            }
        }

        // ── Full miss ──────────────────────────────────────────────────
        for x in out.iter_mut() {
            *x = 0.0;
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        CacheResult {
            tier: None,
            hit: false,
        }
    }

    /// Promote a slot to the plasma tier, evicting the oldest entry if full.
    fn promote_to_plasma(&self, hash: EngramHash, slot: &[f32]) {
        let gen_val = self.generation.fetch_add(1, Ordering::Relaxed) + 1;
        let boxed: Box<[f32]> = slot.into();

        // Check capacity — if we're at the limit, evict the oldest entry.
        // The eviction scan is O(N) but only runs when we're at capacity —
        // acceptable for a cache with ~1024 entries.
        let need_evict = {
            let guard = self.plasma.pin();
            guard.len() >= self.plasma_capacity
        };
        if need_evict {
            // Find the entry with the lowest generation.
            let mut oldest_key: Option<EngramHash> = None;
            let mut oldest_gen: u64 = u64::MAX;
            let guard = self.plasma.pin();
            for (k, (_v, g)) in guard.iter() {
                if *g < oldest_gen {
                    oldest_gen = *g;
                    oldest_key = Some(*k);
                }
            }
            drop(guard);
            if let Some(k) = oldest_key {
                let guard = self.plasma.pin();
                guard.remove(&k);
            }
        }

        let guard = self.plasma.pin();
        guard.insert(hash, (boxed, gen_val));
    }

    /// Snapshot the current stats (for diagnostics).
    pub fn stats(&self) -> ZipfianStatsSnapshot {
        self.stats.snapshot()
    }

    /// Current plasma-tier entry count.
    pub fn plasma_len(&self) -> usize {
        let guard = self.plasma.pin();
        guard.len()
    }

    /// Adaptive hot-cache sizing. Grows or shrinks the plasma capacity to
    /// maintain a target plasma hit rate over the last `window` lookups.
    ///
    /// **Heuristic:** if recent plasma hit rate < target, grow by 50%. If
    /// recent rate > target + 10%, shrink by 25% (avoid thrashing at the
    /// boundary). Caps at a reasonable maximum to prevent unbounded growth.
    ///
    /// This is a coarse controller — a real production cache would use a
    /// proper additive-increase/multiplicative-decrease (AIMD) controller
    /// with hysteresis. The open primitive exposes the knob; the host tunes.
    pub fn maybe_resize(&mut self, target_hit_rate: f32) {
        let snap = self.stats.snapshot();
        let actual = snap.plasma_hit_rate();
        let diff = actual - target_hit_rate;

        if diff < -0.05 {
            // Below target — grow by 50%, capped at 1M entries.
            let new_cap = (self.plasma_capacity as f32 * 1.5) as usize;
            self.plasma_capacity = new_cap.min(1_000_000);
        } else if diff > 0.10 {
            // Above target by 10%+ — shrink by 25%, floor at 16 entries.
            let new_cap = (self.plasma_capacity as f32 * 0.75) as usize;
            self.plasma_capacity = new_cap.max(16);
        }
        // else: within tolerance, no change.
    }

    /// Current plasma capacity (for diagnostics).
    pub fn plasma_capacity(&self) -> usize {
        self.plasma_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{EngramHash, EngramTableBuilder};

    /// Build a small warm-source table with `n_populated` slots at hash 100+i.
    /// (Offsets by 100 so that the `EngramHash(0)` sentinels used internally
    /// by `lookup_cached` don't accidentally hit populated slots.)
    fn make_warm_source(d: usize, n_populated: usize) -> Arc<dyn EngramTable> {
        let mut b = EngramTableBuilder::new(1024, d);
        for i in 0..n_populated as u64 {
            let pat: Vec<f32> = (0..d).map(|j| (i as f32) * 0.1 + j as f32 * 0.01).collect();
            b.add_pattern(EngramHash(100 + i), &pat);
        }
        Arc::new(b.build())
    }

    /// A cold fetcher that "knows" about hashes 1000..1000+n.
    struct RangeColdFetcher {
        base: u64,
        n: usize,
        d: usize,
    }

    impl ColdFetcher for RangeColdFetcher {
        fn fetch(&self, hash: EngramHash, out: &mut [f32]) -> bool {
            if hash.0 >= self.base && (hash.0 - self.base) < self.n as u64 {
                let i = (hash.0 - self.base) as usize;
                for j in 0..self.d {
                    out[j] = (i as f32) * 100.0 + j as f32;
                }
                true
            } else {
                false
            }
        }
    }

    #[test]
    fn all_in_hot_yields_100_percent_plasma_hits() {
        // T6.7: all lookups are in the plasma tier after warm-up.
        let warm = make_warm_source(4, 8);
        let cache = ZipfianCacheHierarchy::new(64, warm, None);

        // Prime the plasma tier: do 8 lookups (each hits warm, promotes).
        let mut out = vec![0.0f32; 4];
        for i in 0..8u64 {
            cache.lookup_cached(EngramHash(100 + i), 4, &mut out);
        }

        // Reset stats (we can't, so just record baseline + do another round).
        let snap_before = cache.stats();

        // Second round: all should hit plasma.
        for i in 0..8u64 {
            let r = cache.lookup_cached(EngramHash(100 + i), 4, &mut out);
            assert!(r.hit, "lookup {i} must hit");
            assert_eq!(r.tier, Some(CacheTier::Plasma), "must hit plasma");
        }
        let snap_after = cache.stats();
        let plasma_delta = snap_after.hits_plasma - snap_before.hits_plasma;
        let warm_delta = snap_after.hits_warm - snap_before.hits_warm;
        assert_eq!(plasma_delta, 8, "all 8 second-round lookups hit plasma");
        assert_eq!(warm_delta, 0, "no warm hits on second round");
    }

    #[test]
    fn all_in_cold_yields_100_percent_cold_hits() {
        // T6.7: warm_source has no data for these hashes; cold_fetcher does.
        let warm = make_warm_source(4, 0); // empty warm
        let cold = Arc::new(RangeColdFetcher {
            base: 1000,
            n: 4,
            d: 4,
        });
        let cache = ZipfianCacheHierarchy::new(64, warm, Some(cold));

        let mut out = vec![0.0f32; 4];
        for i in 0..4u64 {
            let r = cache.lookup_cached(EngramHash(1000 + i), 4, &mut out);
            assert!(r.hit, "cold lookup {i} must hit");
            assert_eq!(r.tier, Some(CacheTier::Cold), "must hit cold tier");
        }
        let snap = cache.stats();
        assert_eq!(snap.hits_cold, 4, "4 cold hits");
        assert_eq!(snap.hits_plasma, 0, "0 plasma hits on first round");
    }

    #[test]
    fn promotion_populates_plasma_for_next_lookup() {
        // T6.7: cold lookup → promote → second lookup hits plasma.
        let warm = make_warm_source(4, 0);
        let cold = Arc::new(RangeColdFetcher {
            base: 2000,
            n: 1,
            d: 4,
        });
        let cache = ZipfianCacheHierarchy::new(64, warm, Some(cold));

        let mut out = vec![0.0f32; 4];
        // First lookup: cold hit.
        let r1 = cache.lookup_cached(EngramHash(2000), 4, &mut out);
        assert_eq!(r1.tier, Some(CacheTier::Cold));
        assert_eq!(&out[..], &[0.0, 1.0, 2.0, 3.0]);

        // Second lookup: should hit plasma now.
        let r2 = cache.lookup_cached(EngramHash(2000), 4, &mut out);
        assert_eq!(
            r2.tier,
            Some(CacheTier::Plasma),
            "promotion must populate plasma"
        );
        assert_eq!(&out[..], &[0.0, 1.0, 2.0, 3.0], "same data after promotion");
    }

    #[test]
    fn full_miss_zero_fills_output() {
        let warm = make_warm_source(4, 0);
        let cache = ZipfianCacheHierarchy::new(64, warm, None);

        let mut out = vec![99.0f32; 4];
        let r = cache.lookup_cached(EngramHash(42), 4, &mut out);
        assert!(!r.hit);
        assert_eq!(r.tier, None);
        assert!(out.iter().all(|&v| v == 0.0), "full miss must zero-fill");
    }

    #[test]
    fn warm_hit_data_is_correct() {
        // Warm hit returns the actual slot data from the backing table.
        let warm = make_warm_source(4, 4);
        let cache = ZipfianCacheHierarchy::new(64, warm.clone(), None);

        let mut out = vec![0.0f32; 4];
        let r = cache.lookup_cached(EngramHash(100), 4, &mut out);
        assert_eq!(r.tier, Some(CacheTier::Warm));
        // Pattern at hash 100 is [0.0, 0.01, 0.02, 0.03].
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.01).abs() < 1e-6);
        assert!((out[3] - 0.03).abs() < 1e-6);
    }

    #[test]
    fn maybe_resize_grows_on_low_hit_rate() {
        let warm = make_warm_source(4, 64);
        let mut cache = ZipfianCacheHierarchy::new(16, warm, None);

        // Do many distinct lookups — most will miss plasma (capacity=16).
        let mut out = vec![0.0f32; 4];
        for i in 0..64u64 {
            cache.lookup_cached(EngramHash(100 + i), 4, &mut out);
        }
        let initial_cap = cache.plasma_capacity();
        let initial_rate = cache.stats().plasma_hit_rate();

        // Target 90% plasma hit rate — we're way below. Resize should grow.
        cache.maybe_resize(0.90);
        assert!(
            cache.plasma_capacity() > initial_cap,
            "capacity should grow when hit rate ({:.3}) < target (0.90)",
            initial_rate
        );
    }

    #[test]
    fn maybe_resize_shrinks_on_high_hit_rate() {
        let warm = make_warm_source(4, 4);
        let mut cache = ZipfianCacheHierarchy::new(1024, warm, None); // big cache

        // Do many redundant lookups on the same 4 hashes — plasma hit rate
        // approaches 100% after warm-up.
        let mut out = vec![0.0f32; 4];
        for _ in 0..10 {
            for i in 0..4u64 {
                cache.lookup_cached(EngramHash(100 + i), 4, &mut out);
            }
        }
        let initial_cap = cache.plasma_capacity();
        cache.maybe_resize(0.50); // target 50%, actual ~100%
        assert!(
            cache.plasma_capacity() < initial_cap,
            "capacity should shrink when hit rate >> target"
        );
    }

    #[test]
    fn snapshot_total_and_hit_rate() {
        let warm = make_warm_source(4, 2);
        let cache = ZipfianCacheHierarchy::new(64, warm, None);

        let mut out = vec![0.0f32; 4];
        // 2 warm hits (and promotions), then 2 plasma hits, then 1 miss.
        cache.lookup_cached(EngramHash(100), 4, &mut out); // warm
        cache.lookup_cached(EngramHash(101), 4, &mut out); // warm
        cache.lookup_cached(EngramHash(100), 4, &mut out); // plasma
        cache.lookup_cached(EngramHash(101), 4, &mut out); // plasma
        cache.lookup_cached(EngramHash(999), 4, &mut out); // miss (999 ≠ 100..102)

        let snap = cache.stats();
        assert_eq!(snap.total(), 5);
        assert_eq!(snap.hits_warm, 2);
        assert_eq!(snap.hits_plasma, 2);
        assert_eq!(snap.misses, 1);
        // Plasma hit rate = 2/5 = 0.4.
        assert!((snap.plasma_hit_rate() - 0.4).abs() < 1e-3);
    }
}
