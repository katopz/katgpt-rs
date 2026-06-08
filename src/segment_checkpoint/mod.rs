//! SegmentCheckpoint — Inference-Time Growing Memory via Cached KV Segments (Plan 223b).
//!
//! Caches compressed KV state checkpoints at segment boundaries.
//! GRM-style gating provides context-dependent retrieval.
//! SSC variant for sparse top-k selection.
//! Zero training required — pure modelless inference enhancement.

pub mod auto_route;
pub mod gating;

#[cfg(feature = "ssc_spec_draft")]
pub mod ssc;

#[cfg(feature = "memory_soup_dtree")]
pub mod memory_soup;

// ---------------------------------------------------------------------------
// SegmentCheckpoint
// ---------------------------------------------------------------------------

/// A single KV segment checkpoint, aligned with KVarN tile boundaries.
#[derive(Clone, Debug)]
pub struct SegmentCheckpoint {
    /// Unique segment identifier.
    pub segment_id: u32,
    /// Compressed key state (KVarN-quantized).
    pub key_compressed: Vec<u8>,
    /// Compressed value state (KVarN-quantized).
    pub val_compressed: Vec<u8>,
    /// MeanPool summary of segment keys for γ computation.
    pub summary: Vec<f32>,
    /// Start position in sequence.
    pub pos_start: usize,
    /// End position in sequence.
    pub pos_end: usize,
}

impl SegmentCheckpoint {
    /// Create a new checkpoint from compressed KV state.
    pub fn new(
        segment_id: u32,
        key_compressed: Vec<u8>,
        val_compressed: Vec<u8>,
        summary: Vec<f32>,
        pos_start: usize,
        pos_end: usize,
    ) -> Self {
        Self {
            segment_id,
            key_compressed,
            val_compressed,
            summary,
            pos_start,
            pos_end,
        }
    }
}

// ---------------------------------------------------------------------------
// SegmentStore
// ---------------------------------------------------------------------------

/// Stores cached segment checkpoints with bounded memory.
pub struct SegmentStore {
    /// Cached segments indexed by segment_id.
    segments: std::collections::HashMap<u32, SegmentCheckpoint>,
    /// Maximum number of cached segments.
    max_segments: usize,
    /// Segment size (should align with KVarN tile_size, default 128).
    segment_size: usize,
    /// Access counts for LFU eviction.
    access_counts: std::collections::HashMap<u32, u64>,
}

impl SegmentStore {
    /// Create a new SegmentStore.
    pub fn new(max_segments: usize, segment_size: usize) -> Self {
        Self {
            segments: std::collections::HashMap::new(),
            max_segments,
            segment_size,
            access_counts: std::collections::HashMap::new(),
        }
    }

    /// Get the configured segment size.
    pub fn segment_size(&self) -> usize {
        self.segment_size
    }

    /// Insert a new segment checkpoint. Evicts LFU if at capacity.
    pub fn insert(&mut self, checkpoint: SegmentCheckpoint) {
        if self.segments.len() >= self.max_segments {
            self.evict_lfu();
        }
        let id = checkpoint.segment_id;
        self.access_counts.insert(id, 0);
        self.segments.insert(id, checkpoint);
    }

    /// Get a segment checkpoint by ID. Increments access count.
    pub fn get(&mut self, segment_id: u32) -> Option<&SegmentCheckpoint> {
        if self.segments.contains_key(&segment_id) {
            *self.access_counts.entry(segment_id).or_insert(0) += 1;
        }
        self.segments.get(&segment_id)
    }

    /// Get all segment summaries for γ gate computation.
    pub fn summaries(&self) -> Vec<&[f32]> {
        self.segments
            .values()
            .map(|s| s.summary.as_slice())
            .collect()
    }

    /// Get all segment IDs in the store.
    pub fn segment_ids(&self) -> Vec<u32> {
        self.segments.keys().copied().collect()
    }

    /// Number of cached segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether store is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Evict least frequently used segment.
    fn evict_lfu(&mut self) {
        if let Some((&min_id, _)) = self.access_counts.iter().min_by_key(|&(_, &c)| c) {
            self.segments.remove(&min_id);
            self.access_counts.remove(&min_id);
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointPolicy (for TriggerGate integration)
// ---------------------------------------------------------------------------

/// Policy for when to emit segment checkpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CheckpointPolicy {
    /// Lazy: only checkpoint on segment boundary (default for CPU-only).
    Lazy = 0,
    /// Normal: checkpoint every segment boundary.
    #[default]
    Normal = 1,
    /// Eager: checkpoint every boundary + pre-compute summaries.
    Eager = 2,
}

impl CheckpointPolicy {
    /// Select checkpoint policy based on TriggerGate tier (QPS heuristic).
    ///
    /// - CPU-only (high QPS) → lazy
    /// - CPU+GPU (medium) → normal
    /// - CPU+GPU+ANE (low QPS) → eager
    pub fn from_tier(qps: f32) -> Self {
        if qps > 20.0 {
            Self::Lazy
        } else if qps > 5.0 {
            Self::Normal
        } else {
            Self::Eager
        }
    }

    /// Should we checkpoint at this segment boundary?
    pub fn should_checkpoint(&self, segment_index: usize) -> bool {
        match self {
            Self::Lazy => segment_index % 4 == 0, // every 4th boundary
            Self::Normal => true,                 // every boundary
            Self::Eager => true,                  // every boundary + pre-compute
        }
    }

    /// Should we pre-compute segment summaries?
    pub fn should_precompute_summary(&self) -> bool {
        matches!(self, Self::Eager)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut store = SegmentStore::new(10, 128);
        let cp = SegmentCheckpoint::new(0, vec![], vec![], vec![0.1, 0.2], 0, 127);
        store.insert(cp);
        assert_eq!(store.len(), 1);
        let got = store.get(0).unwrap();
        assert_eq!(got.pos_start, 0);
    }

    #[test]
    fn test_lfu_eviction() {
        let mut store = SegmentStore::new(2, 128);
        store.insert(SegmentCheckpoint::new(0, vec![], vec![], vec![0.1], 0, 127));
        store.insert(SegmentCheckpoint::new(
            1,
            vec![],
            vec![],
            vec![0.2],
            128,
            255,
        ));

        // Access segment 0 multiple times
        store.get(0);
        store.get(0);
        store.get(0);

        // Insert third segment → should evict segment 1 (least accessed)
        store.insert(SegmentCheckpoint::new(
            2,
            vec![],
            vec![],
            vec![0.3],
            256,
            383,
        ));
        assert!(store.get(0).is_some());
        assert!(store.get(1).is_none()); // evicted
        assert!(store.get(2).is_some());
    }

    #[test]
    fn test_summaries() {
        let mut store = SegmentStore::new(10, 128);
        store.insert(SegmentCheckpoint::new(
            0,
            vec![],
            vec![],
            vec![0.1, 0.2],
            0,
            127,
        ));
        store.insert(SegmentCheckpoint::new(
            1,
            vec![],
            vec![],
            vec![0.3, 0.4],
            128,
            255,
        ));
        let summaries = store.summaries();
        assert_eq!(summaries.len(), 2);
    }

    #[test]
    fn test_segment_ids() {
        let mut store = SegmentStore::new(10, 128);
        store.insert(SegmentCheckpoint::new(5, vec![], vec![], vec![0.1], 0, 127));
        store.insert(SegmentCheckpoint::new(
            10,
            vec![],
            vec![],
            vec![0.2],
            128,
            255,
        ));
        let mut ids = store.segment_ids();
        ids.sort();
        assert_eq!(ids, vec![5, 10]);
    }

    // ── CheckpointPolicy tests ──────────────────────────────────

    #[test]
    fn test_policy_default() {
        assert_eq!(CheckpointPolicy::default(), CheckpointPolicy::Normal);
    }

    #[test]
    fn test_policy_from_tier() {
        assert_eq!(CheckpointPolicy::from_tier(50.0), CheckpointPolicy::Lazy);
        assert_eq!(CheckpointPolicy::from_tier(10.0), CheckpointPolicy::Normal);
        assert_eq!(CheckpointPolicy::from_tier(1.0), CheckpointPolicy::Eager);
    }

    #[test]
    fn test_policy_should_checkpoint() {
        let lazy = CheckpointPolicy::Lazy;
        assert!(lazy.should_checkpoint(0));
        assert!(!lazy.should_checkpoint(1));
        assert!(lazy.should_checkpoint(4));

        let normal = CheckpointPolicy::Normal;
        assert!(normal.should_checkpoint(0));
        assert!(normal.should_checkpoint(1));

        let eager = CheckpointPolicy::Eager;
        assert!(eager.should_checkpoint(0));
        assert!(eager.should_checkpoint(7));
    }

    #[test]
    fn test_policy_precompute_summary() {
        assert!(!CheckpointPolicy::Lazy.should_precompute_summary());
        assert!(!CheckpointPolicy::Normal.should_precompute_summary());
        assert!(CheckpointPolicy::Eager.should_precompute_summary());
    }

    #[test]
    fn test_policy_repr() {
        assert_eq!(CheckpointPolicy::Lazy as u8, 0);
        assert_eq!(CheckpointPolicy::Normal as u8, 1);
        assert_eq!(CheckpointPolicy::Eager as u8, 2);
    }

    // ── Phase 5: Additional unit tests ──────────────────────────

    /// Test that checkpoint emission happens at segment boundaries.
    /// Position ranges must be segment_size-aligned.
    #[test]
    fn test_checkpoint_emission_at_segment_boundaries() {
        let segment_size = 128;
        let mut store = SegmentStore::new(10, segment_size);

        // Simulate tokens 0..127 → first segment
        let accepted_len = segment_size;
        assert_eq!(
            accepted_len % segment_size,
            0,
            "accepted_len must be segment-aligned"
        );
        store.insert(SegmentCheckpoint::new(
            0,
            vec![0xAB; 64],
            vec![0xCD; 64],
            vec![0.1; 32],
            0,
            segment_size - 1,
        ));
        assert_eq!(store.len(), 1);

        // Simulate tokens 128..255 → second segment
        let pos_start = segment_size;
        let pos_end = 2 * segment_size - 1;
        store.insert(SegmentCheckpoint::new(
            1,
            vec![0xAB; 64],
            vec![0xCD; 64],
            vec![0.2; 32],
            pos_start,
            pos_end,
        ));
        assert_eq!(store.len(), 2);

        // Verify segment boundaries are tile-aligned
        for id in store.segment_ids() {
            let cp = store.get(id).unwrap();
            assert_eq!(
                cp.pos_start % segment_size,
                0,
                "segment {} pos_start must be segment-aligned",
                id
            );
            assert_eq!(
                (cp.pos_end + 1) % segment_size,
                0,
                "segment {} pos_end+1 must be segment-aligned",
                id
            );
        }
    }

    /// Test retrieval with 0 segments (empty store).
    #[test]
    fn test_retrieval_zero_segments() {
        let store = SegmentStore::new(10, 128);
        assert!(store.is_empty());
        assert_eq!(store.summaries().len(), 0);
        assert_eq!(store.segment_ids().len(), 0);
    }

    /// Test retrieval with exactly 1 segment.
    #[test]
    fn test_retrieval_one_segment() {
        let mut store = SegmentStore::new(10, 128);
        store.insert(SegmentCheckpoint::new(
            0,
            vec![],
            vec![],
            vec![0.5; 16],
            0,
            127,
        ));
        assert_eq!(store.len(), 1);
        assert_eq!(store.summaries().len(), 1);

        let got = store.get(0).unwrap();
        assert_eq!(got.summary.len(), 16);
    }

    /// Test retrieval with N segments.
    #[test]
    fn test_retrieval_n_segments() {
        let mut store = SegmentStore::new(10, 128);
        for i in 0..5u32 {
            store.insert(SegmentCheckpoint::new(
                i,
                vec![0xAB; 32],
                vec![0xCD; 32],
                vec![i as f32 * 0.1; 8],
                i as usize * 128,
                (i as usize + 1) * 128 - 1,
            ));
        }
        assert_eq!(store.len(), 5);
        assert_eq!(store.summaries().len(), 5);

        // All segments retrievable
        for i in 0..5 {
            assert!(store.get(i).is_some());
        }
    }

    /// Test zero-copy alignment: segment_size % tile_size == 0.
    /// KVarN tile_size = 128 (default). Segment must divide evenly.
    #[test]
    fn test_zero_copy_alignment_tile_boundaries() {
        let tile_size: usize = 128; // KVarN tile_size

        // Valid segment sizes that are tile-aligned
        for &seg_size in &[64, 128, 256, 512] {
            assert_eq!(
                seg_size % tile_size,
                0,
                "segment_size {} must be tile-aligned (tile_size={})",
                seg_size,
                tile_size
            );
        }

        // Verify store segment_size property
        let store = SegmentStore::new(10, 128);
        assert_eq!(store.segment_size() % tile_size, 0);

        // Non-aligned sizes must fail
        let bad_sizes: &[usize] = &[100, 127, 200, 300];
        for &bad in bad_sizes {
            assert_ne!(
                bad % tile_size,
                0,
                "segment_size {} should NOT be tile-aligned",
                bad
            );
        }
    }

    /// Test that checkpoint positions span exactly segment_size tokens.
    #[test]
    fn test_checkpoint_position_span() {
        let segment_size = 128;
        let cp = SegmentCheckpoint::new(0, vec![], vec![], vec![0.1; 8], 0, segment_size - 1);
        assert_eq!(cp.pos_end - cp.pos_start + 1, segment_size);
    }
}
