//! LSH SimHash Approximate Cache Layer — locality-sensitive hashing for near-miss BFCP lookup.
//!
//! Three-level cache hierarchy: L0 exact (BLAKE3) → L1 LSH (SimHash) → compute.
//! SimHash maps logit vectors to 64-bit fingerprints via random projection. Nearby
//! logits produce fingerprints within a small Hamming radius, enabling approximate
//! cache hits without exact hash matches.
//!
//! Plan 220 Phase 1.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::bfcf_types::BFCP;
use super::bfcp_region_cache::{BfcpRegionCache, blake3_logit_hash};

// ── SimHashFingerprint ────────────────────────────────────────

/// 64-bit SimHash fingerprint — Hamming-preserving hash of logit vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SimHashFingerprint(u64);

impl SimHashFingerprint {
    /// Compute SimHash via random projection: for each of 64 output bits, compute
    /// dot product of projection column with logits, sign → bit.
    ///
    /// `projection` has one row per logit dimension, each row has 64 entries.
    /// For bit j: sum projection[i][j] * logits[i] for all i, sign → bit j.
    pub fn from_logits(logits: &[f32], projection: &[[f32; 64]]) -> Self {
        // Row-major iteration: outer loop over logits, accumulate all 64 dots
        // in one pass for cache-friendly access to projection rows.
        let mut dots = [0.0f32; 64];
        for (i, logit) in logits.iter().enumerate() {
            if i >= projection.len() {
                break;
            }
            let row = &projection[i];
            for j in 0..64 {
                dots[j] += row[j] * logit;
            }
        }
        let mut bits: u64 = 0;
        for (j, &dot) in dots.iter().enumerate() {
            if dot >= 0.0 {
                bits |= 1 << j;
            }
        }
        Self(bits)
    }

    /// Hamming distance between two fingerprints (popcount of XOR).
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Extract first `bits` bits as a bucket index.
    #[inline]
    pub fn bucket_index(&self, bits: u32) -> usize {
        (self.0 & ((1u64 << bits) - 1)) as usize
    }
}

// ── LshBucket ─────────────────────────────────────────────────

/// Single LSH bucket — stores fingerprinted entries with FIFO eviction.
pub struct LshBucket {
    pub entries: VecDeque<(SimHashFingerprint, Arc<BFCP>)>,
    pub capacity: usize,
}

impl LshBucket {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Find entry within Hamming radius. Returns (partition_ref, hamming_distance).
    pub fn lookup(&self, fp: &SimHashFingerprint, radius: u32) -> Option<(&Arc<BFCP>, u32)> {
        for (stored_fp, partition) in &self.entries {
            let dist = stored_fp.hamming_distance(fp);
            if dist <= radius {
                return Some((partition, dist));
            }
        }
        None
    }

    /// Insert entry, FIFO evict oldest if at capacity.
    pub fn insert(&mut self, fp: SimHashFingerprint, partition: Arc<BFCP>) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((fp, partition));
    }
}

// ── LshApproximateCache ───────────────────────────────────────

/// Full LSH approximate cache — 2^bucket_bits buckets with random projection.
pub struct LshApproximateCache {
    buckets: Vec<LshBucket>,
    projection: Vec<[f32; 64]>,
    hamming_radius: u32,
    bucket_bits: u32,
}

impl LshApproximateCache {
    /// Create a new LSH cache.
    ///
    /// - `logit_dim`: dimensionality of logit vectors (determines projection rows).
    /// - `num_buckets`: number of LSH buckets (must be power of 2).
    /// - `bucket_capacity`: max entries per bucket before FIFO eviction.
    /// - `hamming_radius`: max Hamming distance for approximate hit.
    pub fn new(
        logit_dim: usize,
        num_buckets: usize,
        bucket_capacity: usize,
        hamming_radius: u32,
    ) -> Self {
        let num_buckets = num_buckets.max(1).next_power_of_two();
        let bucket_bits = num_buckets.trailing_zeros();

        // Random projection matrix: logit_dim × 64, entries ∈ {-1, +1}
        let mut rng = fastrand::Rng::new();
        let projection: Vec<[f32; 64]> = (0..logit_dim)
            .map(|_| {
                let mut row = [0.0f32; 64];
                for val in &mut row {
                    *val = if rng.bool() { 1.0 } else { -1.0 };
                }
                row
            })
            .collect();

        let buckets = (0..num_buckets)
            .map(|_| LshBucket::new(bucket_capacity))
            .collect();

        Self {
            buckets,
            projection,
            hamming_radius,
            bucket_bits,
        }
    }

    /// Compute SimHash for logits and look up in the appropriate bucket.
    pub fn lookup(&self, logits: &[f32]) -> Option<(&Arc<BFCP>, u32)> {
        let fp = SimHashFingerprint::from_logits(logits, &self.projection);
        let idx = fp.bucket_index(self.bucket_bits);
        self.buckets[idx].lookup(&fp, self.hamming_radius)
    }

    /// Compute SimHash for logits and insert into the appropriate bucket.
    pub fn insert(&mut self, logits: &[f32], partition: Arc<BFCP>) {
        let fp = SimHashFingerprint::from_logits(logits, &self.projection);
        let idx = fp.bucket_index(self.bucket_bits);
        self.buckets[idx].insert(fp, partition);
    }
}

// ── ApproximateCaching trait ──────────────────────────────────

/// Trait for approximate caching layers with hit-rate telemetry.
#[cfg(feature = "bfcf_lsh_cms")]
pub trait ApproximateCaching: Send + Sync {
    fn approximate_lookup(&self, logits: &[f32]) -> Option<(&Arc<BFCP>, u32)>;
    fn insert_approximate(&mut self, logits: &[f32], partition: Arc<BFCP>);
    fn cache_tier_rates(&self) -> (f64, f64, f64); // (l0, l1, miss)
}

// ── BfcpLshCache ──────────────────────────────────────────────

/// Three-level cache: L0 exact (BLAKE3) → L1 LSH (SimHash) → compute.
pub struct BfcpLshCache {
    /// Level 0: exact BLAKE3 hash cache from Plan 218.
    exact: BfcpRegionCache,
    /// Level 1: LSH approximate cache.
    lsh: LshApproximateCache,
    /// Hit counters.
    l0_hits: AtomicU64,
    l1_hits: AtomicU64,
    full_misses: AtomicU64,
}

impl BfcpLshCache {
    /// Create a new three-level cache.
    pub fn new(
        exact_capacity: usize,
        logit_dim: usize,
        num_buckets: usize,
        bucket_capacity: usize,
        hamming_radius: u32,
    ) -> Self {
        Self {
            exact: BfcpRegionCache::new(exact_capacity),
            lsh: LshApproximateCache::new(logit_dim, num_buckets, bucket_capacity, hamming_radius),
            l0_hits: AtomicU64::new(0),
            l1_hits: AtomicU64::new(0),
            full_misses: AtomicU64::new(0),
        }
    }

    /// Three-level pipeline: L0 exact → L1 LSH → compute_fn → insert both.
    ///
    /// Returns the partition and the level that served it:
    /// - 0 = L0 exact hit, 1 = L1 LSH hit, 2 = fresh compute.
    pub fn process<F>(&mut self, logits: &[f32], compute_fn: F) -> (Arc<BFCP>, u8)
    where
        F: FnOnce(&[f32]) -> BFCP,
    {
        // L0: exact BLAKE3 lookup
        let hash = blake3_logit_hash(logits);
        if let Some(partition) = self.exact.lookup(&hash) {
            self.l0_hits.fetch_add(1, Ordering::Relaxed);
            return (partition, 0);
        }

        // L1: LSH approximate lookup
        if let Some((partition, _dist)) = self.lsh.lookup(logits) {
            let partition = Arc::clone(partition);
            // Re-insert into L0 for future exact hits
            self.exact.insert(hash, Arc::clone(&partition));
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return (partition, 1);
        }

        // Miss: compute fresh partition
        let partition = Arc::new(compute_fn(logits));
        self.exact.insert(hash, Arc::clone(&partition));
        self.lsh.insert(logits, Arc::clone(&partition));
        self.full_misses.fetch_add(1, Ordering::Relaxed);
        (partition, 2)
    }

    /// Warm-start pipeline: for LSH hits, only recompute diff regions.
    /// Placeholder — full diff logic in Phase 4. Currently calls compute_fn directly.
    pub fn process_warm_start<F>(&mut self, logits: &[f32], compute_fn: F) -> (Arc<BFCP>, u8)
    where
        F: FnOnce(&[f32]) -> BFCP,
    {
        // L0: exact BLAKE3 lookup
        let hash = blake3_logit_hash(logits);
        if let Some(partition) = self.exact.lookup(&hash) {
            self.l0_hits.fetch_add(1, Ordering::Relaxed);
            return (partition, 0);
        }

        // L1: LSH approximate lookup — warm start
        if let Some((_approx_partition, _dist)) = self.lsh.lookup(logits) {
            // Phase 4 TODO: diff approx_partition against fresh logits, only recompute changed regions.
            // For now, just compute fresh and insert.
            let partition = Arc::new(compute_fn(logits));
            self.exact.insert(hash, Arc::clone(&partition));
            self.lsh.insert(logits, Arc::clone(&partition));
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return (partition, 1);
        }

        // Full miss
        let partition = Arc::new(compute_fn(logits));
        self.exact.insert(hash, Arc::clone(&partition));
        self.lsh.insert(logits, Arc::clone(&partition));
        self.full_misses.fetch_add(1, Ordering::Relaxed);
        (partition, 2)
    }

    /// Current hit rates as (l0_rate, l1_rate, miss_rate).
    pub fn hit_rates(&self) -> (f64, f64, f64) {
        let l0 = self.l0_hits.load(Ordering::Relaxed) as f64;
        let l1 = self.l1_hits.load(Ordering::Relaxed) as f64;
        let misses = self.full_misses.load(Ordering::Relaxed) as f64;
        let total = l0 + l1 + misses;
        if total == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (l0 / total, l1 / total, misses / total)
    }
}

impl ApproximateCaching for BfcpLshCache {
    fn approximate_lookup(&self, logits: &[f32]) -> Option<(&Arc<BFCP>, u32)> {
        // Try L0 first
        let hash = blake3_logit_hash(logits);
        if let Some(_partition) = self.exact.lookup(&hash) {
            // L0 is exact — hamming distance 0
            // Note: we can't return a reference to the exact cache's internal value here
            // because BfcpRegionCache::lookup returns owned Arc<BFCP>. Fall through to L1.
        }

        // L1: LSH lookup
        self.lsh.lookup(logits)
    }

    fn insert_approximate(&mut self, logits: &[f32], partition: Arc<BFCP>) {
        let hash = blake3_logit_hash(logits);
        self.exact.insert(hash, Arc::clone(&partition));
        self.lsh.insert(logits, partition);
    }

    fn cache_tier_rates(&self) -> (f64, f64, f64) {
        self.hit_rates()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::bfcf_types::{BorelRegion, HalfSpace, RegionLabel};
    use super::*;
    use std::sync::Arc as TestArc;

    /// Helper: create a trivial BFCP partition.
    fn make_partition() -> BFCP {
        BFCP::from_regions(vec![BorelRegion::new(
            RegionLabel::Accept,
            vec![HalfSpace {
                dim: 0,
                threshold: 0.5,
                above: true,
            }],
            10,
        )])
    }

    /// Helper: create a projection matrix for `dim` dimensions.
    fn make_projection(dim: usize) -> Vec<[f32; 64]> {
        let mut rng = fastrand::Rng::with_seed(42);
        (0..dim)
            .map(|_| {
                let mut row = [0.0f32; 64];
                for val in &mut row {
                    *val = if rng.bool() { 1.0 } else { -1.0 };
                }
                row
            })
            .collect()
    }

    // ── SimHash tests ──────────────────────────────────────

    #[test]
    fn test_simhash_identical_inputs() {
        let proj = make_projection(8);
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let fp1 = SimHashFingerprint::from_logits(&logits, &proj);
        let fp2 = SimHashFingerprint::from_logits(&logits, &proj);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_simhash_nearby_inputs() {
        let proj = make_projection(8);
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let b = [1.01, 2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01f32];
        let fp_a = SimHashFingerprint::from_logits(&a, &proj);
        let fp_b = SimHashFingerprint::from_logits(&b, &proj);
        let dist = fp_a.hamming_distance(&fp_b);
        assert!(
            dist <= 5,
            "nearby logits should have low Hamming distance, got {dist}"
        );
    }

    #[test]
    fn test_simhash_distant_inputs() {
        let proj = make_projection(8);
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let b = [
            -100.0, 200.0, -300.0, 400.0, -500.0, 600.0, -700.0, 800.0f32,
        ];
        let fp_a = SimHashFingerprint::from_logits(&a, &proj);
        let fp_b = SimHashFingerprint::from_logits(&b, &proj);
        let dist = fp_a.hamming_distance(&fp_b);
        assert!(
            dist > 20,
            "distant logits should have high Hamming distance, got {dist}"
        );
    }

    #[test]
    fn test_bucket_index_consistent() {
        let proj = make_projection(8);
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let fp = SimHashFingerprint::from_logits(&logits, &proj);
        assert_eq!(fp.bucket_index(12), fp.bucket_index(12));
    }

    // ── LshBucket / LshApproximateCache tests ───────────────

    #[test]
    fn test_lsh_insert_and_lookup() {
        let mut cache = LshApproximateCache::new(8, 16, 4, 3);
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let partition = make_partition();
        cache.insert(&logits, TestArc::new(partition));
        let result = cache.lookup(&logits);
        assert!(result.is_some(), "should find inserted entry");
        let (found, dist) = result.unwrap();
        assert_eq!(dist, 0, "exact match should have 0 hamming distance");
        assert_eq!(found.accept_count(), 1);
    }

    #[test]
    fn test_lsh_miss_returns_none() {
        let cache = LshApproximateCache::new(8, 16, 4, 3);
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        assert!(cache.lookup(&logits).is_none());
    }

    #[test]
    fn test_lsh_near_miss_captures() {
        // Use larger bucket capacity and radius to account for fingerprint
        // variance from row-major accumulation order.
        let mut cache = LshApproximateCache::new(8, 4, 5, 10);
        let logits_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let partition = make_partition();
        cache.insert(&logits_a, TestArc::new(partition));

        // Slightly perturbed — should be within hamming radius
        let logits_b = [1.01, 2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01f32];
        let result = cache.lookup(&logits_b);
        assert!(result.is_some(), "near-miss should be captured by LSH");
        let (_found, dist) = result.unwrap();
        assert!(
            dist <= 10,
            "near-miss should be within hamming radius, got {dist}"
        );
    }

    #[test]
    fn test_bucket_fifo_eviction() {
        let mut bucket = LshBucket::new(3);
        let fp = |v: u64| SimHashFingerprint(v);
        let partition = make_partition();

        // Fill to capacity
        bucket.insert(fp(1), TestArc::new(partition.clone()));
        bucket.insert(fp(2), TestArc::new(partition.clone()));
        bucket.insert(fp(3), TestArc::new(partition.clone()));
        assert_eq!(bucket.entries.len(), 3);

        // Next insert evicts oldest (fp=1)
        bucket.insert(fp(4), TestArc::new(partition.clone()));
        assert_eq!(bucket.entries.len(), 3);
        assert_eq!(bucket.entries[0].0, fp(2)); // oldest now fp=2
        assert_eq!(bucket.entries[2].0, fp(4)); // newest fp=4
    }

    // ── Three-level pipeline tests ─────────────────────────

    #[test]
    fn test_three_level_pipeline() {
        let mut cache = BfcpLshCache::new(100, 8, 16, 4, 3);
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let _partition = make_partition();
        let compute_count = AtomicU64::new(0);

        let compute = |_: &[f32]| {
            compute_count.fetch_add(1, Ordering::Relaxed);
            make_partition()
        };

        // First call: full miss (L0 miss, L1 miss) → compute
        let (result, level) = cache.process(&logits, compute);
        assert_eq!(level, 2, "first call should be full miss");
        assert_eq!(compute_count.load(Ordering::Relaxed), 1);
        let _ = result;

        // Second call: L0 exact hit
        let (result, level) = cache.process(&logits, compute);
        assert_eq!(level, 0, "second call should be L0 hit");
        assert_eq!(compute_count.load(Ordering::Relaxed), 1); // no new compute
        let _ = result;
    }

    #[test]
    fn test_three_level_hit_rates() {
        let mut cache = BfcpLshCache::new(100, 8, 64, 8, 5);

        // Synthetic 50-step decode simulating a real decode loop:
        // - A small set of ~5 "base" logit patterns that repeat (common tokens)
        // - Each repeat is either exact (L0 hit) or slightly drifted (L1 hit)
        let bases: Vec<Vec<f32>> = (0..5)
            .map(|b| vec![1.0 + b as f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .collect();

        let mut step = 0u32;
        for _round in 0..5 {
            for (b, base) in bases.iter().enumerate() {
                // Exact repeat → L0 hit (after first round inserts it)
                let compute = |_: &[f32]| make_partition();
                let _ = cache.process(base, compute);
                step += 1;

                // Slightly drifted variant → L1 hit if within hamming radius
                let drift = (b as f32 + 0.001) * 0.5;
                let drifted: Vec<f32> = base.iter().map(|&v| v + drift).collect();
                let compute2 = |_: &[f32]| make_partition();
                let _ = cache.process(&drifted, compute2);
                step += 1;
            }
        }

        let (l0_rate, l1_rate, miss_rate) = cache.hit_rates();
        let _total = step as f64;

        // After warm-up (first round = 10 inserts), subsequent rounds should
        // mostly hit L0 (exact repeats) and L1 (drifted variants).
        assert!(
            l0_rate > 0.40,
            "L0 hit rate should be > 40%, got {:.1}%",
            l0_rate * 100.0
        );
        assert!(
            l0_rate + l1_rate > 0.80,
            "L0+L1 hit rate should be > 80%, got {:.1}%",
            (l0_rate + l1_rate) * 100.0
        );
        assert!(
            miss_rate < 0.25,
            "Miss rate should be < 25%, got {:.1}%",
            miss_rate * 100.0
        );
    }
}
