//! SSC — Sparse Selective Caching (Plan 223b Phase 2).
//!
//! Top-k segment selection for efficient retrieval.
//! Uses partial sort (heapselect) pattern.

use crate::segment_checkpoint::gating::compute_gates;
use crate::segment_checkpoint::SegmentStore;

/// Select top-k segments by relevance to query.
/// Uses gate values to rank, returns segment IDs sorted by relevance.
pub fn top_k_segments(
    store: &mut SegmentStore,
    query: &[f32],
    k: usize,
) -> Vec<(u32, f32)> {
    let summaries = store.summaries();
    if summaries.is_empty() {
        return Vec::new();
    }

    let gates = compute_gates(query, &summaries);
    let k = k.min(gates.len()).max(1);

    // Map summary index → segment_id
    let segment_ids = store.segment_ids();

    let mut pairs: Vec<(u32, f32)> = segment_ids
        .into_iter()
        .zip(gates.into_iter())
        .collect();

    // Sort by gate descending
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_checkpoint::SegmentCheckpoint;

    #[test]
    fn test_top_k_segments_basic() {
        let mut store = SegmentStore::new(10, 128);
        store.insert(SegmentCheckpoint::new(
            0,
            vec![],
            vec![],
            vec![1.0, 0.0],
            0,
            127,
        ));
        store.insert(SegmentCheckpoint::new(
            1,
            vec![],
            vec![],
            vec![0.0, 1.0],
            128,
            255,
        ));
        store.insert(SegmentCheckpoint::new(
            2,
            vec![],
            vec![],
            vec![0.5, 0.5],
            256,
            383,
        ));

        let query = vec![1.0, 0.0];
        let top = top_k_segments(&mut store, &query, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 0); // segment 0 is most aligned
    }

    #[test]
    fn test_top_k_caps_at_available() {
        let mut store = SegmentStore::new(10, 128);
        store.insert(SegmentCheckpoint::new(
            0,
            vec![],
            vec![],
            vec![1.0],
            0,
            127,
        ));

        let query = vec![1.0];
        let top = top_k_segments(&mut store, &query, 5);
        assert_eq!(top.len(), 1); // only 1 segment available
    }
}
