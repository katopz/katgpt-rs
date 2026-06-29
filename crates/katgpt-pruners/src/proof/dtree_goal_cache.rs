//! DDTree integration with shared goal cache for speculative decode tree verification (Plan 128, T6).
//!
//! Distilled from AlphaProof Nexus (arXiv:2605.22763):
//! "The global goal cache reduces redundant verification calls by 3×
//! across DDTree branches using deep hashes of exact formal state."
//!
//! # Architecture
//!
//! ```text
//! DTreeGoalCache
//! ├── tree_id: u64                          // identifies the decode step
//! ├── branch_count: usize                   // branches processed in this step
//! ├── cache: ProofGoalCache                 // underlying goal dedup cache
//! │   ├── HashMap<GoalHash, GoalResult>     // blake3 keyed
//! │   └── hits/misses: AtomicU64            // GOAT metrics
//! └── verify_constraint(depth, token_idx, parent_tokens, verifier)
//!     1. key = encode_constraint_key(depth, token_idx, parent_tokens)
//!     2. cache.get_or_verify(key, verifier) → GoalResult
//!     3. Deduplicates across branches sharing same constraint state
//! ```
//!
//! # DDTree Constraint Deduplication
//!
//! During speculative decoding, the DDTree builds a tree of candidate token
//! sequences. Multiple branches may encounter identical constraint states
//! (transpositions). This wrapper scopes `ProofGoalCache` per decode step
//! and encodes constraint identity as canonical bytes for deduplication.
//!
//! # Key Encoding
//!
//! `encode_constraint_key(depth, token_idx, parent_tokens)` produces a
//! deterministic byte sequence: `[depth:u64 LE][token_idx:u64 LE][parents:u16 LE each]`.
//! Same logical constraint → same bytes → cache hit → verification skipped.
//!
//! # Scope
//!
//! Per-decode-step cache scope: created fresh per decode step, not persisted
//! across steps. This avoids stale entries and keeps memory bounded.
//!
//! # Feature Gate
//!
//! Requires `proof_sketch_evolution` feature (depends on `bandit`).

use std::fmt;

use super::goal_cache::{GoalResult, ProofGoalCache, ProofGoalSnapshot};

// ── DTreeCacheSnapshot ─────────────────────────────────────────

/// Immutable snapshot of DDTree goal cache metrics for GOAT reporting.
///
/// Captures tree identity, branch progress, and cache performance at a
/// point in time. Safe to send across threads and store in benchmark results.
#[derive(Clone, Debug, PartialEq)]
pub struct DTreeCacheSnapshot {
    /// Decode step identifier this cache is scoped to.
    pub tree_id: u64,
    /// Number of branches processed so far in this decode step.
    pub branches: usize,
    /// Number of unique constraint entries in the cache.
    pub entries: usize,
    /// Cache hit count (verification avoided).
    pub hits: u64,
    /// Cache miss count (verification executed).
    pub misses: u64,
    /// Hit rate as ratio [0.0, 1.0].
    pub hit_rate: f64,
}

impl DTreeCacheSnapshot {
    /// Capture a snapshot from a `DTreeGoalCache`.
    pub fn from_cache(cache: &DTreeGoalCache) -> Self {
        Self {
            tree_id: cache.tree_id,
            branches: cache.branch_count,
            entries: cache.cache.len(),
            hits: cache.cache.hits(),
            misses: cache.cache.misses(),
            hit_rate: cache.cache.hit_rate(),
        }
    }
}

impl fmt::Display for DTreeCacheSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DTreeCacheSnapshot(tree={}, branches={}, entries={}, hits={}, misses={}, hit_rate={:.1}%)",
            self.tree_id,
            self.branches,
            self.entries,
            self.hits,
            self.misses,
            self.hit_rate * 100.0,
        )
    }
}

// ── DTreeGoalCache ─────────────────────────────────────────────

/// DDTree-scoped goal cache for constraint deduplication across draft branches.
///
/// Wraps [`ProofGoalCache`] with DDTree-specific context: a tree identifier
/// and branch counter. Each decode step creates a fresh `DTreeGoalCache`.
/// As branches are evaluated, identical constraint states (transpositions)
/// are deduplicated via blake3-keyed caching.
///
/// # Lifecycle
///
/// 1. `new()` or `new_with_capacity()` — create at start of decode step
/// 2. `start_branch()` — call before processing each branch
/// 3. `verify_constraint()` — call for each constraint check in the branch
/// 4. `snapshot()` — capture metrics after all branches processed
/// 5. `clear()` — reset for next decode step (or drop and create new)
///
/// # Thread Safety
///
/// Not thread-safe by itself. Use within single-threaded decode step scope.
/// The inner `ProofGoalCache` uses `AtomicU64` for hit/miss counters, so
/// read-only access to metrics is safe from any thread.
///
/// # Example
///
/// ```rust,ignore
/// use katgpt::pruners::proof::{DTreeGoalCache, GoalResult};
///
/// let mut cache = DTreeGoalCache::new(0);
///
/// // Branch 1: verify constraint at depth 2, token 42
/// cache.start_branch();
/// let r1 = cache.verify_constraint(2, 42, &[1, 5], |_bytes| GoalResult::Proved);
/// assert_eq!(r1, GoalResult::Proved);
///
/// // Branch 2: same constraint state → cache hit
/// cache.start_branch();
/// let r2 = cache.verify_constraint(2, 42, &[1, 5], |_bytes| GoalResult::Disproved("never".into()));
/// assert_eq!(r2, GoalResult::Proved, "should return cached result");
/// ```
#[derive(Debug)]
pub struct DTreeGoalCache {
    /// Decode step identifier (scoped per tree build).
    pub tree_id: u64,
    /// Number of branches processed in this decode step.
    branch_count: usize,
    /// Underlying goal deduplication cache.
    cache: ProofGoalCache,
}

impl DTreeGoalCache {
    /// Create a new DDTree goal cache for a given decode step.
    ///
    /// Uses default capacity (64 entries) for the underlying cache.
    pub fn new(tree_id: u64) -> Self {
        Self {
            tree_id,
            branch_count: 0,
            cache: ProofGoalCache::new(),
        }
    }

    /// Create with a specific pre-allocation capacity.
    pub fn new_with_capacity(tree_id: u64, capacity: usize) -> Self {
        Self {
            tree_id,
            branch_count: 0,
            cache: ProofGoalCache::with_capacity(capacity),
        }
    }

    /// Verify a constraint, using the cache for deduplication.
    ///
    /// # Algorithm
    ///
    /// 1. Encode (depth, token_idx, parent_tokens) into canonical bytes
    /// 2. Hash with blake3 (inside `ProofGoalCache`)
    /// 3. Cache hit → return cached result, no verifier call
    /// 4. Cache miss → call verifier, store result
    ///
    /// # Arguments
    ///
    /// * `depth` — tree depth of the constraint check
    /// * `token_idx` — token index being evaluated
    /// * `parent_tokens` — tokens placed at earlier depths in the path
    /// * `verifier` — closure or [`GoalVerifier`](super::goal_cache::GoalVerifier) impl
    ///
    /// # Returns
    ///
    /// The verification result (either from cache or freshly computed).
    pub fn verify_constraint(
        &mut self,
        depth: usize,
        token_idx: usize,
        parent_tokens: &[usize],
        verifier: impl Fn(&[u8]) -> GoalResult + Send + Sync,
    ) -> GoalResult {
        let key = encode_constraint_key(depth, token_idx, parent_tokens);
        self.cache.get_or_verify(&key, verifier)
    }

    /// Increment the branch counter before processing a new branch.
    ///
    /// Call this once per branch before any `verify_constraint` calls
    /// for that branch. Used for metrics reporting.
    pub fn start_branch(&mut self) {
        self.branch_count += 1;
    }

    /// Number of branches processed so far.
    pub fn branch_count(&self) -> usize {
        self.branch_count
    }

    /// Number of unique constraint entries in the cache.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Is the cache empty?
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.cache.hits()
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.cache.misses()
    }

    /// Cache hit rate as ratio [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    /// Total lookups (hits + misses).
    pub fn total_lookups(&self) -> u64 {
        self.cache.total_lookups()
    }

    /// Capture a metrics snapshot.
    pub fn snapshot(&self) -> DTreeCacheSnapshot {
        DTreeCacheSnapshot::from_cache(self)
    }

    /// Capture the underlying goal cache snapshot.
    pub fn goal_snapshot(&self) -> ProofGoalSnapshot {
        ProofGoalSnapshot::from_cache(&self.cache)
    }

    /// Clear all cached results and reset branch counter for next decode step.
    ///
    /// Tree ID is preserved. Use `new()` to create a fresh cache with a new ID.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.branch_count = 0;
    }

    /// Estimate memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> usize {
        self.cache.estimated_memory_bytes()
    }
}

impl fmt::Display for DTreeGoalCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DTreeGoalCache(tree={}, branches={}, entries={}, hits={}, misses={}, hit_rate={:.1}%)",
            self.tree_id,
            self.branch_count,
            self.cache.len(),
            self.cache.hits(),
            self.cache.misses(),
            self.cache.hit_rate() * 100.0,
        )
    }
}

// ── Key Encoding ───────────────────────────────────────────────

/// Encode a constraint identity as canonical bytes for deduplication hashing.
///
/// Format:
/// ```text
/// [depth as u64 LE (8 bytes)][token_idx as u64 LE (8 bytes)][parent_tokens as u16 LE each]
/// ```
///
/// Produces a deterministic byte sequence: same (depth, token_idx, parent_tokens)
/// → same bytes → same blake3 hash → cache hit.
///
/// Parent tokens are encoded as `u16` to save space (vocab sizes < 65536).
/// For larger vocabs, the encoding still works but truncates tokens > 65535.
pub fn encode_constraint_key(depth: usize, token_idx: usize, parent_tokens: &[usize]) -> Vec<u8> {
    let parent_len = parent_tokens.len();
    let mut buf = Vec::with_capacity(16 + parent_len * 2);

    // Depth as u64 LE
    buf.extend_from_slice(&(depth as u64).to_le_bytes());
    // Token index as u64 LE
    buf.extend_from_slice(&(token_idx as u64).to_le_bytes());
    // Parent tokens as u16 LE each
    for &t in parent_tokens {
        buf.extend_from_slice(&(t as u16).to_le_bytes());
    }

    buf
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

    fn unknown_verifier(_bytes: &[u8]) -> GoalResult {
        GoalResult::Unknown
    }

    // ── encode_constraint_key Tests ────────────────────────────

    #[test]
    fn key_encoding_deterministic() {
        let k1 = encode_constraint_key(2, 42, &[1, 5]);
        let k2 = encode_constraint_key(2, 42, &[1, 5]);
        assert_eq!(k1, k2, "same inputs must produce same key");
    }

    #[test]
    fn key_encoding_different_depth() {
        let k1 = encode_constraint_key(1, 42, &[1, 5]);
        let k2 = encode_constraint_key(2, 42, &[1, 5]);
        assert_ne!(k1, k2, "different depth must produce different key");
    }

    #[test]
    fn key_encoding_different_token() {
        let k1 = encode_constraint_key(2, 42, &[1, 5]);
        let k2 = encode_constraint_key(2, 43, &[1, 5]);
        assert_ne!(k1, k2, "different token must produce different key");
    }

    #[test]
    fn key_encoding_different_parents() {
        let k1 = encode_constraint_key(2, 42, &[1, 5]);
        let k2 = encode_constraint_key(2, 42, &[1, 6]);
        assert_ne!(k1, k2, "different parents must produce different key");
    }

    #[test]
    fn key_encoding_no_parents() {
        let k = encode_constraint_key(0, 0, &[]);
        assert_eq!(k.len(), 16, "no parents → 8 + 8 bytes");
    }

    #[test]
    fn key_encoding_with_parents() {
        let k = encode_constraint_key(1, 2, &[3, 4, 5]);
        assert_eq!(k.len(), 22, "3 parents → 8 + 8 + 6 bytes");
    }

    #[test]
    fn key_encoding_many_parents() {
        let parents: Vec<usize> = (0..100).collect();
        let k = encode_constraint_key(0, 0, &parents);
        assert_eq!(k.len(), 216, "100 parents → 8 + 8 + 200 bytes");
    }

    #[test]
    fn key_encoding_zero_values() {
        let k = encode_constraint_key(0, 0, &[]);
        // All zeros for depth and token_idx
        assert_eq!(&k[..8], &[0u8; 8]);
        assert_eq!(&k[8..16], &[0u8; 8]);
    }

    #[test]
    fn key_encoding_layout() {
        let k = encode_constraint_key(1, 42, &[3]);
        // depth=1 as u64 LE
        assert_eq!(u64::from_le_bytes(k[..8].try_into().unwrap()), 1);
        // token_idx=42 as u64 LE
        assert_eq!(u64::from_le_bytes(k[8..16].try_into().unwrap()), 42);
        // parent[0]=3 as u16 LE
        assert_eq!(u16::from_le_bytes(k[16..18].try_into().unwrap()), 3);
    }

    // ── DTreeGoalCache Tests ───────────────────────────────────

    #[test]
    fn new_cache_is_empty() {
        let cache = DTreeGoalCache::new(0);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.branch_count(), 0);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn new_with_capacity() {
        let cache = DTreeGoalCache::new_with_capacity(1, 128);
        assert!(cache.is_empty());
        assert_eq!(cache.tree_id, 1);
    }

    #[test]
    fn tree_id_preserved() {
        let cache = DTreeGoalCache::new(42);
        assert_eq!(cache.tree_id, 42);
    }

    #[test]
    fn verify_constraint_miss_on_first() {
        let mut cache = DTreeGoalCache::new(0);
        let result = cache.verify_constraint(0, 1, &[], proved_verifier);
        assert_eq!(result, GoalResult::Proved);
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn verify_constraint_hit_on_repeat() {
        let mut cache = DTreeGoalCache::new(0);

        // First: miss
        cache.verify_constraint(2, 42, &[1, 5], proved_verifier);
        assert_eq!(cache.misses(), 1);

        // Same constraint: hit (verifier not called)
        let result = cache.verify_constraint(2, 42, &[1, 5], disproved_verifier);
        assert_eq!(result, GoalResult::Proved, "should return cached Proved");
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1, "misses should not increase");
    }

    #[test]
    fn verify_constraint_different_depth_different_entry() {
        let mut cache = DTreeGoalCache::new(0);

        cache.verify_constraint(1, 42, &[], proved_verifier);
        cache.verify_constraint(2, 42, &[], disproved_verifier);

        assert_eq!(cache.len(), 2, "different depth = different entry");
        assert_eq!(cache.misses(), 2);
    }

    #[test]
    fn verify_constraint_different_token_different_entry() {
        let mut cache = DTreeGoalCache::new(0);

        cache.verify_constraint(0, 1, &[], proved_verifier);
        cache.verify_constraint(0, 2, &[], disproved_verifier);

        assert_eq!(cache.len(), 2, "different token = different entry");
    }

    #[test]
    fn verify_constraint_different_parents_different_entry() {
        let mut cache = DTreeGoalCache::new(0);

        cache.verify_constraint(2, 42, &[1], proved_verifier);
        cache.verify_constraint(2, 42, &[2], disproved_verifier);

        assert_eq!(cache.len(), 2, "different parents = different entry");
    }

    #[test]
    fn verify_constraint_handles_all_result_types() {
        let mut cache = DTreeGoalCache::new(0);

        let r1 = cache.verify_constraint(0, 0, &[], proved_verifier);
        assert_eq!(r1, GoalResult::Proved);

        let r2 = cache.verify_constraint(0, 1, &[], disproved_verifier);
        assert!(r2.is_disproved());

        let r3 = cache.verify_constraint(0, 2, &[], unknown_verifier);
        assert!(r3.is_unknown());

        assert_eq!(cache.len(), 3);
    }

    // ── Branch Tracking Tests ──────────────────────────────────

    #[test]
    fn start_branch_increments_count() {
        let mut cache = DTreeGoalCache::new(0);
        assert_eq!(cache.branch_count(), 0);

        cache.start_branch();
        assert_eq!(cache.branch_count(), 1);

        cache.start_branch();
        assert_eq!(cache.branch_count(), 2);

        cache.start_branch();
        assert_eq!(cache.branch_count(), 3);
    }

    #[test]
    fn branch_count_independent_of_verify() {
        let mut cache = DTreeGoalCache::new(0);
        cache.verify_constraint(0, 0, &[], proved_verifier);
        assert_eq!(
            cache.branch_count(),
            0,
            "verify does not affect branch count"
        );

        cache.start_branch();
        cache.verify_constraint(0, 1, &[], proved_verifier);
        assert_eq!(cache.branch_count(), 1);
    }

    // ── Clear Tests ────────────────────────────────────────────

    #[test]
    fn clear_resets_cache_and_branches() {
        let mut cache = DTreeGoalCache::new(5);
        cache.start_branch();
        cache.verify_constraint(0, 0, &[], proved_verifier);
        cache.verify_constraint(0, 0, &[], proved_verifier); // hit

        assert_eq!(cache.branch_count(), 1);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hits(), 1);

        cache.clear();

        assert_eq!(cache.branch_count(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.hit_rate(), 0.0);
        assert_eq!(cache.tree_id, 5, "tree_id preserved after clear");
    }

    // ── Hit Rate Tests ─────────────────────────────────────────

    #[test]
    fn hit_rate_calculation() {
        let mut cache = DTreeGoalCache::new(0);

        // 2 misses
        cache.verify_constraint(0, 0, &[], proved_verifier);
        cache.verify_constraint(0, 1, &[], proved_verifier);

        // 3 hits (same constraints)
        cache.verify_constraint(0, 0, &[], proved_verifier);
        cache.verify_constraint(0, 1, &[], proved_verifier);
        cache.verify_constraint(0, 0, &[], proved_verifier);

        // 3 hits, 2 misses = 60% hit rate
        let expected = 3.0 / 5.0;
        assert!((cache.hit_rate() - expected).abs() < 1e-9);
    }

    #[test]
    fn total_lookups() {
        let mut cache = DTreeGoalCache::new(0);
        assert_eq!(cache.total_lookups(), 0);

        cache.verify_constraint(0, 0, &[], proved_verifier);
        cache.verify_constraint(0, 0, &[], proved_verifier);
        cache.verify_constraint(0, 1, &[], proved_verifier);

        assert_eq!(cache.total_lookups(), 3);
    }

    // ── Multi-Branch Simulation ────────────────────────────────

    #[test]
    fn multi_branch_deduplication() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let verifier_calls = AtomicU64::new(0);

        let counting_verifier = |_bytes: &[u8]| -> GoalResult {
            verifier_calls.fetch_add(1, Ordering::Relaxed);
            GoalResult::Proved
        };

        let mut cache = DTreeGoalCache::new(0);

        // Branch 1: verify 3 unique constraints
        cache.start_branch();
        cache.verify_constraint(0, 1, &[], counting_verifier);
        cache.verify_constraint(1, 2, &[1], counting_verifier);
        cache.verify_constraint(2, 3, &[1, 2], counting_verifier);
        assert_eq!(cache.misses(), 3);
        assert_eq!(verifier_calls.load(Ordering::Relaxed), 3);

        // Branch 2: same 3 constraints → all hits
        cache.start_branch();
        cache.verify_constraint(0, 1, &[], counting_verifier);
        cache.verify_constraint(1, 2, &[1], counting_verifier);
        cache.verify_constraint(2, 3, &[1, 2], counting_verifier);
        assert_eq!(cache.hits(), 3);
        assert_eq!(
            verifier_calls.load(Ordering::Relaxed),
            3,
            "no new verifier calls on hits"
        );

        // Branch 3: 2 cached + 1 new
        cache.start_branch();
        cache.verify_constraint(0, 1, &[], counting_verifier); // hit
        cache.verify_constraint(1, 5, &[1], counting_verifier); // miss (new token)
        cache.verify_constraint(2, 3, &[1, 2], counting_verifier); // hit

        assert_eq!(cache.branch_count(), 3);
        assert_eq!(cache.len(), 4); // 3 original + 1 new
        assert_eq!(
            verifier_calls.load(Ordering::Relaxed),
            4,
            "only 4 total verifier calls"
        );
        assert_eq!(cache.total_lookups(), 9); // 3 + 3 + 3
    }

    // ── Snapshot Tests ─────────────────────────────────────────

    #[test]
    fn snapshot_from_cache() {
        let mut cache = DTreeGoalCache::new(7);
        cache.start_branch();
        cache.verify_constraint(0, 0, &[], proved_verifier);
        cache.verify_constraint(0, 0, &[], proved_verifier); // hit

        let snap = cache.snapshot();
        assert_eq!(snap.tree_id, 7);
        assert_eq!(snap.branches, 1);
        assert_eq!(snap.entries, 1);
        assert_eq!(snap.hits, 1);
        assert_eq!(snap.misses, 1);
        assert!((snap.hit_rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn snapshot_display() {
        let snap = DTreeCacheSnapshot {
            tree_id: 42,
            branches: 5,
            entries: 10,
            hits: 50,
            misses: 25,
            hit_rate: 0.667,
        };
        let display = format!("{snap}");
        assert!(display.contains("tree=42"));
        assert!(display.contains("branches=5"));
        assert!(display.contains("entries=10"));
        assert!(display.contains("hits=50"));
        assert!(display.contains("misses=25"));
    }

    #[test]
    fn goal_snapshot_from_cache() {
        let mut cache = DTreeGoalCache::new(0);
        cache.verify_constraint(0, 0, &[], proved_verifier);
        let snap = cache.goal_snapshot();
        assert_eq!(snap.entries, 1);
        assert_eq!(snap.misses, 1);
    }

    // ── Display Tests ──────────────────────────────────────────

    #[test]
    fn display_empty() {
        let cache = DTreeGoalCache::new(0);
        let display = format!("{cache}");
        assert!(display.contains("tree=0"));
        assert!(display.contains("branches=0"));
        assert!(display.contains("entries=0"));
    }

    #[test]
    fn display_with_data() {
        let mut cache = DTreeGoalCache::new(3);
        cache.start_branch();
        cache.verify_constraint(0, 0, &[], proved_verifier);
        let display = format!("{cache}");
        assert!(display.contains("tree=3"));
        assert!(display.contains("branches=1"));
        assert!(display.contains("entries=1"));
    }

    // ── Memory Tests ───────────────────────────────────────────

    #[test]
    fn estimated_memory_empty() {
        let cache = DTreeGoalCache::new(0);
        assert_eq!(cache.estimated_memory_bytes(), 0);
    }

    #[test]
    fn estimated_memory_with_entries() {
        let mut cache = DTreeGoalCache::new(0);
        cache.verify_constraint(0, 0, &[], proved_verifier);
        assert!(cache.estimated_memory_bytes() > 0);
    }
}
