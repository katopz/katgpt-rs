//! SegmentCheckpoint — Inference-Time Growing Memory via Cached KV Segments (Plan 223b).
//!
//! Caches compressed KV state checkpoints at segment boundaries.
//! GRM-style gating provides context-dependent retrieval.
//! SSC variant for sparse top-k selection.
//! Zero training required — pure modelless inference enhancement.

pub mod gating;

#[cfg(feature = "ssc_spec_draft")]
pub mod ssc;

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

/// Policy for when to emit checkpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CheckpointPolicy {
    /// Lazy: only on segment boundary.
    #[default]
    Normal = 0,
    /// Eager: every boundary + pre-compute summaries.
    Eager = 1,
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
}
