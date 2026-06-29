//! Per-Pruner Memory — append-only ring buffer for experience accumulation.
//!
//! Each pruner accumulates edge cases and failure modes across sessions.
//! Ring buffer bounded to a power-of-2 capacity. O(1) append via atomic cursor,
//! O(k) recent-k retrieval. blake3 integrity hash for freeze/thaw.
//!
//! # MUSE Lifecycle: learn
//!
//! Episode rewards flow into `PrunerMemory::append`. Later, `AbsorbCompress`
//! reads recent entries to decide which arms to promote or demote.
//!
//! # Performance
//!
//! Target: <10ns per append, zero allocation on append.
//! All writes use `AtomicU32` cursor wrapping — no lock, no mutex.

use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::freeze::{load_frozen, save_frozen};

// ── MemoryEntry ──────────────────────────────────────────────────

/// Single experience record in the pruner's memory ring buffer.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MemoryEntry {
    /// Arm index selected in this episode.
    pub arm: u16,
    /// Reward received for this arm.
    pub reward: f32,
    /// True if reward was an outlier (e.g., >2σ from mean).
    pub is_edge_case: bool,
    /// True if reward was below failure threshold.
    pub is_failure: bool,
    /// Monotonic timestamp (episode count or similar).
    pub ts: u64,
}

impl MemoryEntry {
    /// Create a new entry with the given fields.
    #[inline]
    pub const fn new(arm: u16, reward: f32, is_edge_case: bool, is_failure: bool, ts: u64) -> Self {
        Self {
            arm,
            reward,
            is_edge_case,
            is_failure,
            ts,
        }
    }
}

// ── PrunerMemory ─────────────────────────────────────────────────

/// Per-pruner append-only experience log.
///
/// Ring buffer bounded to `capacity` entries (rounded to next power of 2).
/// Uses `AtomicU32` head for lock-free concurrent appends.
pub struct PrunerMemory {
    /// blake3 hash of pruner identity (integrity check on thaw).
    pruner_hash: [u8; 32],
    /// Ring buffer of experiences.
    entries: Box<[MemoryEntry]>,
    /// Write cursor (wraps around via masking).
    head: AtomicU32,
    /// Capacity (power of 2, used as mask).
    capacity: u32,
    /// Total entries written (monotonically increasing).
    total_written: AtomicU64,
}

impl std::fmt::Debug for PrunerMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrunerMemory")
            .field("capacity", &self.capacity)
            .field("total_written", &self.total_written)
            .field("head", &self.head)
            .finish_non_exhaustive()
    }
}

impl PrunerMemory {
    /// Create a new memory ring buffer with the given capacity (rounded up to next power of 2).
    ///
    /// `pruner_id` is hashed with blake3 for integrity checks on thaw.
    pub fn new(capacity: usize, pruner_id: &str) -> Self {
        let cap = capacity.next_power_of_two();
        let entries = vec![
            MemoryEntry {
                arm: 0,
                reward: 0.0,
                is_edge_case: false,
                is_failure: false,
                ts: 0,
            };
            cap
        ]
        .into_boxed_slice();
        Self {
            pruner_hash: compute_hash(pruner_id),
            entries,
            head: AtomicU32::new(0),
            capacity: cap as u32,
            total_written: AtomicU64::new(0),
        }
    }

    /// Append an experience entry. O(1), lock-free, zero allocation.
    #[inline]
    pub fn append(&self, entry: MemoryEntry) {
        let idx = self.head.fetch_add(1, Ordering::Relaxed) & (self.capacity - 1);
        // Safety: idx is always in bounds (masked to [0, capacity)).
        // We use a mutable pointer through UnsafeCell-like semantics on Box<[T]>.
        // The AtomicU32 head ensures each writer gets a unique slot.
        unsafe {
            let ptr = self.entries.as_ptr().add(idx as usize) as *mut MemoryEntry;
            ptr.write(entry);
        }
        self.total_written.fetch_add(1, Ordering::Relaxed);
    }

    /// Retrieve the last `n` entries in chronological order (oldest first).
    ///
    /// If fewer than `n` entries have been written, returns all available.
    pub fn recent(&self, n: usize) -> Vec<MemoryEntry> {
        let total = self.total_written.load(Ordering::Relaxed) as usize;
        // Can only return up to capacity entries (older ones are overwritten).
        let count = n.min(total).min(self.capacity as usize);
        if count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(count);
        let head = self.head.load(Ordering::Relaxed) as usize;
        // Read 'count' entries ending at head-1, going backwards.
        // Then reverse for chronological order.
        for i in 0..count {
            let idx = (head + self.capacity as usize - count + i) & (self.capacity as usize - 1);
            result.push(unsafe { *self.entries.as_ptr().add(idx) });
        }
        result
    }

    /// Total number of entries written (including overwritten).
    #[inline]
    pub fn total_entries(&self) -> u64 {
        self.total_written.load(Ordering::Relaxed)
    }

    /// blake3 hash of the pruner identity.
    #[inline]
    pub fn pruner_hash(&self) -> &[u8; 32] {
        &self.pruner_hash
    }

    /// Capacity of the ring buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }

    /// Verify integrity: check that the given pruner_id matches the stored hash.
    pub fn verify_identity(&self, pruner_id: &str) -> bool {
        compute_hash(pruner_id) == self.pruner_hash
    }
}

// ── Hash ─────────────────────────────────────────────────────────

/// Compute blake3 hash of a pruner identity string.
#[inline]
pub fn compute_hash(id: &str) -> [u8; 32] {
    blake3::hash(id.as_bytes()).into()
}

// ── FrozenPrunerMemory ───────────────────────────────────────────

/// On-disk representation of `PrunerMemory` for freeze/thaw persistence.
///
/// Uses a fixed-size array instead of `Box<[MemoryEntry]>` so the struct
/// is fully `repr(C)` safe (no pointers). Magic + version validated on load.
#[repr(C)]
pub struct FrozenPrunerMemory {
    magic: [u8; 4],
    version: u32,
    /// blake3 hash of pruner identity for integrity verification.
    pruner_hash: [u8; 32],
    /// Ring buffer capacity (must match entries.len()).
    capacity: u32,
    /// Write cursor at time of freeze.
    head: u32,
    /// Total entries written at time of freeze.
    total_written: u64,
    /// Actual number of entries stored (may be < capacity if not full).
    entry_count: u32,
    /// Fixed-size ring buffer storage (max 1024 entries).
    entries: [MemoryEntry; Self::MAX_ENTRIES],
}

impl FrozenPrunerMemory {
    const MAGIC: [u8; 4] = *b"PMEM";
    const VERSION: u32 = 1;
    const MAX_ENTRIES: usize = 1024;

    /// Validate magic and version fields.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "FrozenPrunerMemory: bad magic {:?}, expected {:?}",
                &self.magic,
                &Self::MAGIC
            ));
        }
        if self.version != Self::VERSION {
            return Err(format!(
                "FrozenPrunerMemory: version {} not supported (expected {})",
                self.version,
                Self::VERSION
            ));
        }
        Ok(())
    }
}

impl PrunerMemory {
    /// Snapshot current state into a `repr(C)` safe frozen struct.
    ///
    /// Panics if capacity exceeds `FrozenPrunerMemory::MAX_ENTRIES` (1024).
    pub fn freeze(&self) -> FrozenPrunerMemory {
        assert!(
            self.capacity as usize <= FrozenPrunerMemory::MAX_ENTRIES,
            "PrunerMemory capacity {} exceeds FrozenPrunerMemory::MAX_ENTRIES {}",
            self.capacity,
            FrozenPrunerMemory::MAX_ENTRIES
        );

        let total = self.total_written.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed) as usize;
        let entry_count = (total.min(self.capacity as u64)) as u32;

        let mut entries =
            [MemoryEntry::new(0, 0.0, false, false, 0); FrozenPrunerMemory::MAX_ENTRIES];
        // Copy entries from ring buffer to linear array (oldest first).
        if entry_count > 0 {
            let cap = self.capacity as usize;
            let start = (head + cap - entry_count as usize) & (cap - 1);
            for (i, entry) in entries.iter_mut().enumerate().take(entry_count as usize) {
                let src_idx = (start + i) & (cap - 1);
                *entry = unsafe { *self.entries.as_ptr().add(src_idx) };
            }
        }

        FrozenPrunerMemory {
            magic: FrozenPrunerMemory::MAGIC,
            version: FrozenPrunerMemory::VERSION,
            pruner_hash: self.pruner_hash,
            capacity: self.capacity,
            head: head as u32,
            total_written: total,
            entry_count,
            entries,
        }
    }

    /// Restore from a frozen snapshot.
    ///
    /// Returns a new `PrunerMemory` with entries placed back into the ring buffer
    /// in the same positions they occupied at freeze time.
    pub fn thaw(frozen: &FrozenPrunerMemory) -> Result<Self, String> {
        frozen.validate()?;

        if frozen.capacity as usize > FrozenPrunerMemory::MAX_ENTRIES {
            return Err(format!(
                "FrozenPrunerMemory capacity {} exceeds MAX_ENTRIES {}",
                frozen.capacity,
                FrozenPrunerMemory::MAX_ENTRIES
            ));
        }

        let cap = frozen.capacity.next_power_of_two() as usize;
        let mut entries = vec![MemoryEntry::new(0, 0.0, false, false, 0); cap].into_boxed_slice();

        // Restore entries into their ring-buffer positions.
        for i in 0..frozen.entry_count as usize {
            // In the frozen struct entries are stored oldest-first (0..entry_count).
            // Compute the original ring-buffer position for each.
            let ring_idx =
                (frozen.head as usize + cap - frozen.entry_count as usize + i) & (cap - 1);
            entries[ring_idx] = frozen.entries[i];
        }

        Ok(Self {
            pruner_hash: frozen.pruner_hash,
            entries,
            head: AtomicU32::new(frozen.head),
            capacity: cap as u32,
            total_written: AtomicU64::new(frozen.total_written),
        })
    }

    /// Convenience: freeze + save to disk.
    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        let frozen = self.freeze();
        save_frozen(path, &frozen)
    }

    /// Convenience: load from disk + thaw + verify identity.
    ///
    /// Fails if the loaded pruner_hash doesn't match `pruner_id`.
    pub fn load_from(path: &Path, pruner_id: &str) -> Result<Self, String> {
        let frozen: FrozenPrunerMemory = load_frozen(path)?;
        let mem = Self::thaw(&frozen)?;
        if !mem.verify_identity(pruner_id) {
            return Err(format!(
                "Identity mismatch: loaded memory does not match pruner_id {:?}",
                pruner_id
            ));
        }
        Ok(mem)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capacity_rounding() {
        let mem = PrunerMemory::new(100, "test");
        assert_eq!(mem.capacity(), 128); // next power of 2
        let mem2 = PrunerMemory::new(1024, "test");
        assert_eq!(mem2.capacity(), 1024); // already power of 2
        let mem3 = PrunerMemory::new(1, "test");
        assert_eq!(mem3.capacity(), 1);
    }

    #[test]
    fn test_append_and_recent() {
        let mem = PrunerMemory::new(16, "test_pruner");
        for i in 0..5u64 {
            mem.append(MemoryEntry::new(i as u16, i as f32, false, false, i));
        }
        assert_eq!(mem.total_entries(), 5);
        let recent = mem.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].arm, 2); // oldest of the 3
        assert_eq!(recent[2].arm, 4); // newest
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mem = PrunerMemory::new(4, "wrap_test");
        // Write 6 entries into capacity-4 buffer.
        for i in 0..6u64 {
            mem.append(MemoryEntry::new(i as u16, i as f32, false, false, i));
        }
        assert_eq!(mem.total_entries(), 6);
        // Only last 4 should be retrievable.
        let recent = mem.recent(10);
        assert_eq!(recent.len(), 4);
        // Entries 2, 3, 4, 5 (0 and 1 overwritten).
        assert_eq!(recent[0].arm, 2);
        assert_eq!(recent[3].arm, 5);
    }

    #[test]
    fn test_bounded_eviction() {
        let mem = PrunerMemory::new(8, "evict_test");
        for i in 0..20u64 {
            mem.append(MemoryEntry::new(i as u16, 1.0, i % 5 == 0, i > 15, i));
        }
        assert_eq!(mem.total_entries(), 20);
        let recent = mem.recent(8);
        assert_eq!(recent.len(), 8);
        // Should have entries 12..20 (last 8).
        assert_eq!(recent[0].arm, 12);
        assert_eq!(recent[7].arm, 19);
        // Check flags survived the wrap.
        // recent indices: 0→arm12, 1→arm13, 2→arm14, 3→arm15, 4→arm16, 5→arm17, 6→arm18, 7→arm19
        // is_edge_case when i%5==0: arm15 (15%5==0) at index 3
        assert!(recent[3].is_edge_case); // arm 15
        // arm16: 16%5==1 → not edge case
        assert!(!recent[4].is_edge_case);
        // arm 19 > 15 → is_failure=true
        assert!(recent[7].is_failure);
    }

    #[test]
    fn test_hash_identity() {
        let mem = PrunerMemory::new(16, "my_pruner");
        assert!(mem.verify_identity("my_pruner"));
        assert!(!mem.verify_identity("other_pruner"));
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let h1 = compute_hash("test");
        let h2 = compute_hash("test");
        let h3 = compute_hash("other");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_recent_empty() {
        let mem = PrunerMemory::new(16, "empty");
        assert_eq!(mem.recent(10).len(), 0);
        assert_eq!(mem.total_entries(), 0);
    }

    #[test]
    fn test_recent_more_than_available() {
        let mem = PrunerMemory::new(16, "sparse");
        mem.append(MemoryEntry::new(0, 1.0, false, false, 0));
        mem.append(MemoryEntry::new(1, 0.5, false, false, 1));
        let recent = mem.recent(10);
        assert_eq!(recent.len(), 2);
    }

    /// Benchmark: append throughput.
    /// Target: <10ns per append, zero allocation.
    /// Plan 192 Task 1.
    #[test]
    fn bench_append_throughput() {
        use std::time::Instant;

        let mem = PrunerMemory::new(1024, "bench");
        let entry = MemoryEntry::new(0, 1.0, false, false, 0);

        let iterations = 100_000u64;
        let start = Instant::now();
        for _ in 0..iterations {
            mem.append(std::hint::black_box(entry));
            std::hint::black_box(());
        }
        let elapsed = start.elapsed();

        let per_append = elapsed / iterations as u32;
        println!(
            "[bench] PrunerMemory::append: {per_append:?} per call ({} iterations)",
            iterations
        );

        assert!(
            per_append.as_nanos() < 100,
            "append should be <100ns, got {per_append:?}"
        );
    }

    // ── Freeze/Thaw Tests ──────────────────────────────────────────

    #[test]
    fn test_freeze_thaw_roundtrip() {
        let mem = PrunerMemory::new(16, "freeze_test");
        for i in 0..10u64 {
            mem.append(MemoryEntry::new(
                i as u16,
                i as f32 * 0.1,
                i == 3,
                i > 7,
                i * 100,
            ));
        }

        let frozen = mem.freeze();
        assert_eq!(frozen.magic, *b"PMEM");
        assert_eq!(frozen.version, 1);
        assert_eq!(frozen.entry_count, 10);
        assert_eq!(frozen.capacity, 16);

        let restored = PrunerMemory::thaw(&frozen).unwrap();
        assert_eq!(restored.capacity(), 16);
        assert_eq!(restored.total_entries(), 10);
        assert!(restored.verify_identity("freeze_test"));

        let original_recent = mem.recent(10);
        let restored_recent = restored.recent(10);
        assert_eq!(original_recent.len(), restored_recent.len());
        for (o, r) in original_recent.iter().zip(restored_recent.iter()) {
            assert_eq!(o.arm, r.arm);
            assert_eq!(o.reward.to_bits(), r.reward.to_bits());
            assert_eq!(o.is_edge_case, r.is_edge_case);
            assert_eq!(o.is_failure, r.is_failure);
            assert_eq!(o.ts, r.ts);
        }
    }

    #[test]
    fn test_freeze_thaw_wrap() {
        let mem = PrunerMemory::new(4, "wrap_freeze");
        // Write 7 entries into capacity-4 buffer — oldest 3 overwritten.
        for i in 0..7u64 {
            mem.append(MemoryEntry::new(i as u16, i as f32, false, false, i));
        }

        let frozen = mem.freeze();
        assert_eq!(frozen.entry_count, 4);

        let restored = PrunerMemory::thaw(&frozen).unwrap();
        assert_eq!(restored.total_entries(), 7);
        let recent = restored.recent(4);
        assert_eq!(recent.len(), 4);
        // Arms 3, 4, 5, 6 survived (0, 1, 2 overwritten).
        assert_eq!(recent[0].arm, 3);
        assert_eq!(recent[3].arm, 6);
    }

    #[test]
    fn test_save_load_roundtrip() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("memory.bin");

        let mem = PrunerMemory::new(8, "disk_test");
        for i in 0..5u64 {
            mem.append(MemoryEntry::new(i as u16, 0.5, i == 2, i == 4, i));
        }

        mem.save_to(&path).unwrap();
        assert!(path.exists());

        let loaded = PrunerMemory::load_from(&path, "disk_test").unwrap();
        assert_eq!(loaded.total_entries(), 5);
        assert_eq!(loaded.capacity(), 8);

        let recent = loaded.recent(5);
        assert_eq!(recent.len(), 5);
        assert_eq!(recent[0].arm, 0);
        assert_eq!(recent[4].arm, 4);
        assert!(recent[2].is_edge_case);
        assert!(recent[4].is_failure);
    }

    #[test]
    fn test_thaw_wrong_identity() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("memory.bin");

        let mem = PrunerMemory::new(8, "correct_id");
        mem.append(MemoryEntry::new(1, 0.9, false, false, 0));
        mem.save_to(&path).unwrap();

        let result = PrunerMemory::load_from(&path, "wrong_id");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Identity mismatch"),
            "expected identity mismatch error, got: {err}"
        );
    }
}

// TL;DR: PrunerMemory — lock-free append-only ring buffer with blake3 integrity hash. O(1) append via AtomicU32, O(k) recent retrieval. Freeze/thaw via FrozenPrunerMemory (repr(C), fixed 1024-entry array) for disk persistence. Capacity rounded to power of 2 for masking.
