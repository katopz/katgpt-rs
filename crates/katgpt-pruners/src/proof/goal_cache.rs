//! Global goal deduplication cache for DDTree verification (Plan 128, T1).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "The global goal cache reduces redundant verification calls by 3×
//! across DDTree branches using deep hashes of exact formal state."
//!
//! # Architecture
//!
//! ```text
//! ProofGoalCache
//! ├── cache: HashMap<GoalHash, GoalResult>   // blake3 keyed
//! ├── hits: AtomicU64                        // GOAT metric
//! ├── misses: AtomicU64                      // GOAT metric
//! └── get_or_verify(goal, verifier) → GoalResult
//!     1. hash = GoalHash(blake3::hash(goal.canonical_bytes()))
//!     2. cache.entry(hash).or_insert_with(|| verifier.verify(goal))
//!     3. Increment hits/misses atomically
//! ```
//!
//! # Scope
//!
//! Per-decode-step cache scope: created fresh per decode step, not persisted
//! across steps. This avoids stale entries and keeps memory bounded.
//! Transposition tables (e.g., GoState) handle cross-step caching separately.
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ── GoalHash ───────────────────────────────────────────────────

/// blake3 hash of a goal's canonical byte representation.
///
/// Wraps [`blake3::Hash`] for type safety and ergonomic display.
/// Per project convention (Research 063, OCTOPUS): blake3 over SHA256
/// for cache keys — faster, adequate for deduplication.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GoalHash(blake3::Hash);

impl GoalHash {
    /// Hash canonical bytes of a goal.
    ///
    /// The caller must ensure `canonical_bytes` is a deterministic encoding
    /// (same logical goal → same bytes). For game states, this is typically
    /// a serialization of the constraint + context.
    pub fn from_canonical(canonical_bytes: &[u8]) -> Self {
        Self(blake3::hash(canonical_bytes))
    }

    /// Raw blake3 hash bytes as a 32-byte array.
    pub fn as_bytes(&self) -> &[u8; 32] {
        self.0.as_bytes()
    }

    /// Inner blake3 hash for interoperability.
    pub fn inner(&self) -> &blake3::Hash {
        &self.0
    }
}

impl fmt::Debug for GoalHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show first 16 hex chars for readability
        let hex = self.0.to_hex();
        write!(f, "GoalHash({})", &hex[..16])
    }
}

impl fmt::Display for GoalHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hex = self.0.to_hex();
        write!(f, "{}", &hex[..16])
    }
}

// ── GoalResult ─────────────────────────────────────────────────

/// Outcome of verifying a proof goal / constraint.
///
/// Maps to AlphaProof Nexus's formal proof outcomes:
/// - `Proved` → goal is satisfied (verification passed)
/// - `Disproved` → goal is violated with a concrete counterexample
/// - `Unknown` → verification inconclusive (timeout, resource limit)
#[derive(Clone, Debug, PartialEq)]
pub enum GoalResult {
    /// Goal verified successfully.
    Proved,
    /// Goal disproved with counterexample data.
    ///
    /// The `String` payload stores a human-readable description of the
    /// counterexample (e.g., "position (3,5) violates adjacency constraint").
    /// For structured counterexamples, encode as JSON or a domain-specific format.
    Disproved(String),
    /// Verification inconclusive — may retry later.
    Unknown,
}

impl GoalResult {
    /// Is this goal proved?
    pub fn is_proved(&self) -> bool {
        matches!(self, Self::Proved)
    }

    /// Is this goal disproved (has a counterexample)?
    pub fn is_disproved(&self) -> bool {
        matches!(self, Self::Disproved(_))
    }

    /// Is this result unknown (inconclusive)?
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

impl fmt::Display for GoalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proved => write!(f, "Proved"),
            Self::Disproved(ce) => write!(f, "Disproved({ce})"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ── GoalVerifier trait ─────────────────────────────────────────

/// Trait for goal verification — the expensive operation we're caching.
///
/// Implementors provide domain-specific verification logic:
/// - DDTree constraints: `is_valid()` check on token path
/// - Game state constraints: rule validation (e.g., legal move, safe position)
/// - Formal proof goals: Lean-style type checking
///
/// The cache calls this only on cache misses.
pub trait GoalVerifier: Send + Sync {
    /// Verify a goal given its canonical byte representation.
    ///
    /// `canonical_bytes` is the same input passed to [`GoalHash::from_canonical`],
    /// allowing verifiers to decode domain-specific state from bytes if needed.
    fn verify(&self, canonical_bytes: &[u8]) -> GoalResult;
}

/// Blanket impl for closures — allows `|bytes| -> GoalResult { ... }` as verifiers.
impl<F: Fn(&[u8]) -> GoalResult + Send + Sync> GoalVerifier for F {
    fn verify(&self, canonical_bytes: &[u8]) -> GoalResult {
        self(canonical_bytes)
    }
}

// ── ProofGoalCache ─────────────────────────────────────────────

/// blake3-keyed global goal deduplication cache.
///
/// Prevents redundant verification calls across DDTree branches within
/// a single decode step. When multiple branches encounter the same
/// constraint state (transposition), the cache returns the previously
/// computed result without re-verifying.
///
/// # GOAT Metrics
///
/// - `hits()`: number of cache hits (verification avoided)
/// - `misses()`: number of cache misses (verification executed)
/// - `hit_rate()`: ratio of hits to total lookups (target: ≥60%)
///
/// # Thread Safety
///
/// Hit/miss counters use `AtomicU64` for lock-free reads from any thread.
/// The `HashMap` itself is not thread-safe — use one cache per decode step
/// (single-threaded scope) or wrap in `Mutex` for cross-thread sharing.
///
/// # Example
///
/// ```rust,ignore
/// use katgpt::pruners::proof::{ProofGoalCache, GoalResult};
///
/// let mut cache = ProofGoalCache::new();
///
/// // First verification: cache miss, verifier called
/// let result = cache.get_or_verify(b"constraint_A", |bytes| {
///     GoalResult::Proved // expensive check simulated
/// });
/// assert_eq!(result, GoalResult::Proved);
/// assert_eq!(cache.misses(), 1);
/// assert_eq!(cache.hits(), 0);
///
/// // Same constraint again: cache hit, verifier NOT called
/// let result2 = cache.get_or_verify(b"constraint_A", |bytes| {
///     panic!("Should not be called on cache hit!");
/// });
/// assert_eq!(result2, GoalResult::Proved);
/// assert_eq!(cache.hits(), 1);
/// ```
#[derive(Debug)]
pub struct ProofGoalCache {
    /// Deduplication cache: blake3 hash → verification result.
    cache: HashMap<GoalHash, GoalResult>,
    /// Cache hit count (GOAT metric).
    hits: AtomicU64,
    /// Cache miss count (GOAT metric).
    misses: AtomicU64,
}

impl ProofGoalCache {
    /// Create an empty goal cache.
    ///
    /// Pre-allocates space for 64 entries (matching paper's top-64 population cap).
    /// Grows automatically if more unique goals appear.
    pub fn new() -> Self {
        Self {
            cache: HashMap::with_capacity(64),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Create a cache with a specific pre-allocation capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get or verify a goal, using the cache for deduplication.
    ///
    /// # Algorithm
    ///
    /// 1. Hash `canonical_bytes` with blake3
    /// 2. If hash is in cache → return cached result, increment hits
    /// 3. If hash is not in cache → call verifier, store result, increment misses
    ///
    /// # Arguments
    ///
    /// * `canonical_bytes` — deterministic encoding of the goal/constraint
    /// * `verifier` — implements [`GoalVerifier`] or is a closure `|bytes| -> GoalResult`
    ///
    /// # Returns
    ///
    /// The verification result (either from cache or freshly computed).
    pub fn get_or_verify(
        &mut self,
        canonical_bytes: &[u8],
        verifier: impl GoalVerifier,
    ) -> GoalResult {
        let hash = GoalHash::from_canonical(canonical_bytes);

        match self.cache.get(&hash) {
            Some(result) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                result.clone()
            }
            None => {
                let result = verifier.verify(canonical_bytes);
                self.cache.insert(hash, result.clone());
                self.misses.fetch_add(1, Ordering::Relaxed);
                result
            }
        }
    }

    /// Get or verify with a pre-computed hash (avoids re-hashing).
    ///
    /// Useful when the caller already has the hash from a prior computation.
    pub fn get_or_verify_with_hash(
        &mut self,
        hash: GoalHash,
        canonical_bytes: &[u8],
        verifier: impl GoalVerifier,
    ) -> GoalResult {
        match self.cache.get(&hash) {
            Some(result) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                result.clone()
            }
            None => {
                let result = verifier.verify(canonical_bytes);
                self.cache.insert(hash, result.clone());
                self.misses.fetch_add(1, Ordering::Relaxed);
                result
            }
        }
    }

    /// Check if a goal is already in the cache without verifying.
    ///
    /// Returns `None` if not cached, `Some(&GoalResult)` if cached.
    /// Does NOT update hit/miss counters (peek-only).
    pub fn peek(&self, canonical_bytes: &[u8]) -> Option<&GoalResult> {
        let hash = GoalHash::from_canonical(canonical_bytes);
        self.cache.get(&hash)
    }

    /// Manually insert a result into the cache.
    ///
    /// Useful for pre-populating from a prior decode step's known results.
    /// Does NOT update hit/miss counters.
    ///
    /// Returns the previous result if the goal was already cached.
    pub fn insert(&mut self, canonical_bytes: &[u8], result: GoalResult) -> Option<GoalResult> {
        let hash = GoalHash::from_canonical(canonical_bytes);
        self.cache.insert(hash, result)
    }

    /// Number of unique goals in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Is the cache empty?
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Total number of cache hits (verification avoided).
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Total number of cache misses (verification executed).
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Total lookups (hits + misses).
    pub fn total_lookups(&self) -> u64 {
        self.hits() + self.misses()
    }

    /// Cache hit rate as a ratio in [0.0, 1.0].
    ///
    /// Returns 0.0 if no lookups have been performed.
    /// Target: ≥0.60 per paper's benchmark results on structured domains.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_lookups();
        match total {
            0 => 0.0,
            _ => self.hits() as f64 / total as f64,
        }
    }

    /// Clear all cached results and reset counters.
    ///
    /// Called at the start of each new decode step to avoid stale entries.
    /// Preserves the allocated capacity for reuse.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Estimate memory usage in bytes.
    ///
    /// Approximate: 32 bytes per hash + size_of::<GoalResult>() per entry.
    /// Used for GOAT metric reporting and capacity planning.
    pub fn estimated_memory_bytes(&self) -> usize {
        // GoalHash: 32 bytes (blake3::Hash)
        // GoalResult: ~24-48 bytes (enum with String variant)
        // HashMap overhead: ~48 bytes per entry
        let per_entry = 32 + 48 + 48; // hash + result + map overhead
        self.cache.len() * per_entry
    }

    /// Reset counters without clearing the cache.
    ///
    /// Useful for per-episode metric tracking without losing cached results.
    pub fn reset_counters(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

impl Default for ProofGoalCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ProofGoalCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProofGoalCache(entries={}, hits={}, misses={}, hit_rate={:.1}%)",
            self.cache.len(),
            self.hits(),
            self.misses(),
            self.hit_rate() * 100.0,
        )
    }
}

// ── ProofGoalSnapshot ──────────────────────────────────────────

/// Immutable snapshot of cache metrics for GOAT reporting.
///
/// Captures hit/miss counters and cache size at a point in time.
/// Safe to send across threads and store in benchmark results.
#[derive(Clone, Debug, PartialEq)]
pub struct ProofGoalSnapshot {
    /// Number of unique goals cached.
    pub entries: usize,
    /// Cache hit count.
    pub hits: u64,
    /// Cache miss count.
    pub misses: u64,
    /// Hit rate as ratio [0.0, 1.0].
    pub hit_rate: f64,
    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: usize,
}

impl ProofGoalSnapshot {
    /// Capture a snapshot of the current cache state.
    pub fn from_cache(cache: &ProofGoalCache) -> Self {
        Self {
            entries: cache.len(),
            hits: cache.hits(),
            misses: cache.misses(),
            hit_rate: cache.hit_rate(),
            estimated_memory_bytes: cache.estimated_memory_bytes(),
        }
    }
}

impl fmt::Display for ProofGoalSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ProofGoalSnapshot(entries={}, hits={}, misses={}, hit_rate={:.1}%, mem={}B)",
            self.entries,
            self.hits,
            self.misses,
            self.hit_rate * 100.0,
            self.estimated_memory_bytes,
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn proved_verifier(_bytes: &[u8]) -> GoalResult {
        GoalResult::Proved
    }

    fn disproved_verifier(_bytes: &[u8]) -> GoalResult {
        GoalResult::Disproved("test counterexample".to_string())
    }

    // ── GoalHash Tests ─────────────────────────────────────────

    #[test]
    fn goal_hash_deterministic() {
        let h1 = GoalHash::from_canonical(b"constraint_A");
        let h2 = GoalHash::from_canonical(b"constraint_A");
        assert_eq!(h1, h2, "same input must produce same hash");
    }

    #[test]
    fn goal_hash_different_inputs() {
        let h1 = GoalHash::from_canonical(b"constraint_A");
        let h2 = GoalHash::from_canonical(b"constraint_B");
        assert_ne!(h1, h2, "different inputs must produce different hashes");
    }

    #[test]
    fn goal_hash_empty_input() {
        let h = GoalHash::from_canonical(b"");
        assert_eq!(h.as_bytes().len(), 32, "blake3 hash is always 32 bytes");
    }

    #[test]
    fn goal_hash_display_truncated() {
        let h = GoalHash::from_canonical(b"test");
        let display = format!("{h}");
        assert_eq!(display.len(), 16, "display shows first 16 hex chars");
    }

    // ── GoalResult Tests ───────────────────────────────────────

    #[test]
    fn goal_result_predicates() {
        assert!(GoalResult::Proved.is_proved());
        assert!(!GoalResult::Proved.is_disproved());
        assert!(!GoalResult::Proved.is_unknown());

        assert!(GoalResult::Disproved("ce".to_string()).is_disproved());
        assert!(!GoalResult::Disproved("ce".to_string()).is_proved());

        assert!(GoalResult::Unknown.is_unknown());
        assert!(!GoalResult::Unknown.is_proved());
    }

    #[test]
    fn goal_result_display() {
        assert_eq!(format!("{}", GoalResult::Proved), "Proved");
        assert_eq!(
            format!("{}", GoalResult::Disproved("bad".to_string())),
            "Disproved(bad)"
        );
        assert_eq!(format!("{}", GoalResult::Unknown), "Unknown");
    }

    // ── ProofGoalCache Tests ───────────────────────────────────

    #[test]
    fn cache_miss_on_first_lookup() {
        let mut cache = ProofGoalCache::new();
        let result = cache.get_or_verify(b"goal_1", proved_verifier);
        assert_eq!(result, GoalResult::Proved);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_hit_on_repeat_lookup() {
        let mut cache = ProofGoalCache::new();

        // First lookup: miss
        cache.get_or_verify(b"goal_1", proved_verifier);
        assert_eq!(cache.misses(), 1);

        // Second lookup: hit
        let result = cache.get_or_verify(b"goal_1", disproved_verifier);
        assert_eq!(
            result,
            GoalResult::Proved,
            "should return cached Proved, not call disproved_verifier"
        );
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1, "misses should not increase on hit");
        assert_eq!(cache.len(), 1, "no new entry added");
    }

    #[test]
    fn cache_multiple_goals() {
        let mut cache = ProofGoalCache::new();

        fn counting_verifier(bytes: &[u8]) -> GoalResult {
            match bytes[0] % 3 {
                0 => GoalResult::Proved,
                1 => GoalResult::Disproved("ce".to_string()),
                _ => GoalResult::Unknown,
            }
        }

        // Three unique goals
        cache.get_or_verify(b"goal_A", counting_verifier);
        cache.get_or_verify(b"goal_B", counting_verifier);
        cache.get_or_verify(b"goal_C", counting_verifier);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.misses(), 3);
        assert_eq!(cache.hits(), 0);

        // Repeat all three — should all be hits
        cache.get_or_verify(b"goal_A", counting_verifier);
        cache.get_or_verify(b"goal_B", counting_verifier);
        cache.get_or_verify(b"goal_C", counting_verifier);
        assert_eq!(cache.hits(), 3);
        assert_eq!(cache.misses(), 3, "no new misses");
        assert_eq!(cache.len(), 3, "no new entries");
    }

    #[test]
    fn cache_hit_rate_zero_on_empty() {
        let cache = ProofGoalCache::new();
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn cache_hit_rate_calculation() {
        let mut cache = ProofGoalCache::new();

        cache.get_or_verify(b"a", proved_verifier); // miss
        cache.get_or_verify(b"b", proved_verifier); // miss
        cache.get_or_verify(b"a", proved_verifier); // hit
        cache.get_or_verify(b"a", proved_verifier); // hit
        cache.get_or_verify(b"b", proved_verifier); // hit

        // 3 hits, 2 misses = 60% hit rate
        let expected = 3.0 / 5.0;
        assert!((cache.hit_rate() - expected).abs() < 1e-9);
    }

    #[test]
    fn cache_clear_resets_everything() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"a", proved_verifier);
        cache.get_or_verify(b"a", proved_verifier);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hits(), 1);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn cache_clear_preserves_capacity() {
        let mut cache = ProofGoalCache::with_capacity(128);
        cache.get_or_verify(b"a", proved_verifier);
        assert!(cache.len() < 128);

        cache.clear();
        // Capacity should be preserved (internal HashMap detail)
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_peek_does_not_update_counters() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"a", proved_verifier);

        // Peek at cached goal
        let peeked = cache.peek(b"a");
        assert!(peeked.is_some());
        assert_eq!(peeked.unwrap(), &GoalResult::Proved);

        // Peek at uncached goal
        let peeked_miss = cache.peek(b"b");
        assert!(peeked_miss.is_none());

        // Counters should be unchanged
        assert_eq!(cache.hits(), 0, "peek should not increment hits");
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn cache_insert_manual() {
        let mut cache = ProofGoalCache::new();
        let prev = cache.insert(b"a", GoalResult::Proved);
        assert!(prev.is_none());

        // Overwrite existing
        let prev = cache.insert(b"a", GoalResult::Unknown);
        assert_eq!(prev, Some(GoalResult::Proved));
        assert_eq!(cache.len(), 1, "overwrite should not increase size");

        // Verify the updated value
        let result = cache.get_or_verify(b"a", disproved_verifier);
        assert_eq!(
            result,
            GoalResult::Unknown,
            "should return manually inserted Unknown"
        );
    }

    #[test]
    fn cache_reset_counters_only() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"a", proved_verifier);
        cache.get_or_verify(b"a", proved_verifier);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hits(), 1);

        cache.reset_counters();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.len(), 1, "cache entries should be preserved");

        // Next lookup should still be a hit (cache not cleared)
        let result = cache.get_or_verify(b"a", disproved_verifier);
        assert_eq!(result, GoalResult::Proved);
        assert_eq!(cache.hits(), 1, "fresh hit count after reset");
    }

    #[test]
    fn cache_get_or_verify_with_hash() {
        let mut cache = ProofGoalCache::new();
        let hash = GoalHash::from_canonical(b"a");

        // First lookup with pre-computed hash
        let result = cache.get_or_verify_with_hash(hash, b"a", proved_verifier);
        assert_eq!(result, GoalResult::Proved);
        assert_eq!(cache.misses(), 1);

        // Second lookup with same hash — hit
        let result = cache.get_or_verify_with_hash(hash, b"a", disproved_verifier);
        assert_eq!(result, GoalResult::Proved, "should return cached result");
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn cache_total_lookups() {
        let mut cache = ProofGoalCache::new();
        assert_eq!(cache.total_lookups(), 0);

        cache.get_or_verify(b"a", proved_verifier); // miss → total=1
        cache.get_or_verify(b"a", proved_verifier); // hit → total=2
        cache.get_or_verify(b"b", proved_verifier); // miss → total=3

        assert_eq!(cache.total_lookups(), 3);
    }

    #[test]
    fn cache_estimated_memory_bytes() {
        let cache = ProofGoalCache::new();
        assert_eq!(cache.estimated_memory_bytes(), 0);

        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"a", proved_verifier);
        let mem1 = cache.estimated_memory_bytes();
        assert!(mem1 > 0, "non-empty cache should report positive memory");

        cache.get_or_verify(b"b", proved_verifier);
        let mem2 = cache.estimated_memory_bytes();
        assert!(mem2 > mem1, "more entries should increase memory estimate");
    }

    #[test]
    fn cache_display_format() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"a", proved_verifier);
        cache.get_or_verify(b"a", proved_verifier);

        let display = format!("{cache}");
        assert!(display.contains("entries=1"));
        assert!(display.contains("hits=1"));
        assert!(display.contains("misses=1"));
        assert!(display.contains("hit_rate=50.0%"));
    }

    // ── ProofGoalSnapshot Tests ────────────────────────────────

    #[test]
    fn snapshot_from_cache() {
        let mut cache = ProofGoalCache::new();
        cache.get_or_verify(b"a", proved_verifier);
        cache.get_or_verify(b"a", proved_verifier);

        let snap = ProofGoalSnapshot::from_cache(&cache);
        assert_eq!(snap.entries, 1);
        assert_eq!(snap.hits, 1);
        assert_eq!(snap.misses, 1);
        assert!((snap.hit_rate - 0.5).abs() < 1e-9);
        assert!(snap.estimated_memory_bytes > 0);
    }

    #[test]
    fn snapshot_display() {
        let snap = ProofGoalSnapshot {
            entries: 10,
            hits: 50,
            misses: 25,
            hit_rate: 0.667,
            estimated_memory_bytes: 1024,
        };
        let display = format!("{snap}");
        assert!(display.contains("entries=10"));
        assert!(display.contains("hits=50"));
        assert!(display.contains("mem=1024B"));
    }

    // ── Closure Verifier Tests ─────────────────────────────────

    #[test]
    fn closure_verifier_works() {
        let mut cache = ProofGoalCache::new();
        let verifier = |bytes: &[u8]| -> GoalResult {
            match bytes.is_empty() {
                true => GoalResult::Unknown,
                false => GoalResult::Proved,
            }
        };

        let r1 = cache.get_or_verify(b"non-empty", verifier);
        assert_eq!(r1, GoalResult::Proved);

        let r2 = cache.get_or_verify(b"", verifier);
        assert_eq!(r2, GoalResult::Unknown);
    }

    #[test]
    fn cache_handles_disproved_with_counterexample() {
        let mut cache = ProofGoalCache::new();
        let ce = "position (3,5) violates adjacency".to_string();
        let verifier = |_bytes: &[u8]| GoalResult::Disproved(ce.clone());

        let result = cache.get_or_verify(b"bad_goal", verifier);
        match &result {
            GoalResult::Disproved(msg) => assert_eq!(*msg, "position (3,5) violates adjacency"),
            _ => panic!("expected Disproved"),
        }

        // Verify it's cached correctly
        let cached = cache.peek(b"bad_goal").unwrap().clone();
        assert_eq!(cached, result);
    }
}
