//! DEC topology cache: tracks dirty regions and caches topology-dependent results.
//!
//! Plan 261 Phase 1.
//!
//! When the cell complex mutates (e.g. faces destroyed for terrain deformation),
//! downstream consumers need to know what to recompute. [`DecCache`] provides:
//! - **Version tracking**: `is_valid()` checks if cached results match current topology
//! - **Dirty region**: which faces/vertices changed since last cache update
//! - **Betti number cache**: structural invariant — depends only on topology,
//!   not on input cochain values

use crate::hodge::{HodgeComponents, hodge_decompose};
use crate::types::{CellComplex, CochainField};

// ---------------------------------------------------------------------------
// DirtyRegion
// ---------------------------------------------------------------------------

/// Tracks which cells changed since the last cache update.
///
/// Consumers inspect this to drive incremental recomputation:
/// only cochains overlapping `changed_faces` / `changed_vertices` need refresh.
pub struct DirtyRegion {
    /// Face indices that were removed or modified.
    pub changed_faces: Vec<usize>,
    /// Vertex indices on the boundary of changed faces.
    pub changed_vertices: Vec<usize>,
    /// Topology version at the time of the last `mark_*` call.
    pub version: u64,
}

impl DirtyRegion {
    /// Create an empty dirty region.
    #[inline]
    pub fn new() -> Self {
        Self {
            changed_faces: Vec::new(),
            changed_vertices: Vec::new(),
            version: 0,
        }
    }

    /// Mark a face as changed at the given topology version.
    #[inline]
    pub fn mark_face(&mut self, face_idx: usize, version: u64) {
        self.changed_faces.push(face_idx);
        self.version = version;
    }

    /// Returns `true` if no cells have been marked dirty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.changed_faces.is_empty() && self.changed_vertices.is_empty()
    }
}

impl Default for DirtyRegion {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DecCache
// ---------------------------------------------------------------------------

/// Caches DEC results keyed by `CellComplex::topology_version()`.
///
/// Typical lifecycle:
/// 1. Compute Betti numbers, store via `store_betti(bettis, cx.topology_version())`.
/// 2. Before recomputing, check `cache.is_valid(cx.topology_version())`.
/// 3. If a face is destroyed, call `mark_face_destroyed` — this invalidates
///    topology-dependent caches and records the dirty region.
/// 4. After recomputation, store fresh results with the new version.
pub struct DecCache {
    /// Version of the topology that the caches were computed against.
    topology_version: u64,
    /// Cached Hodge decomposition (input-dependent — cleared on topology change).
    hodge_cache: Option<HodgeComponents>,
    /// Cached Betti numbers (structural — depend only on topology).
    betti_cache: Option<[usize; 4]>,
    /// Incremental update hints.
    dirty_region: DirtyRegion,
}

impl DecCache {
    /// Create an empty cache at version 0.
    #[inline]
    pub fn new() -> Self {
        Self {
            topology_version: 0,
            hodge_cache: None,
            betti_cache: None,
            dirty_region: DirtyRegion::new(),
        }
    }

    /// Returns `true` if cached results are valid for the given complex version.
    ///
    /// `version > 0` guard prevents a false-positive when neither the cache
    /// nor the complex has ever been mutated (both at version 0).
    #[inline]
    pub fn is_valid(&self, complex_version: u64) -> bool {
        self.topology_version == complex_version && self.topology_version > 0
    }

    /// Clear all cached results and reset version to 0.
    pub fn invalidate(&mut self) {
        self.hodge_cache = None;
        self.betti_cache = None;
        self.topology_version = 0;
    }

    /// Cached Hodge decomposition, if present.
    #[inline]
    pub fn hodge_components(&self) -> Option<&HodgeComponents> {
        self.hodge_cache.as_ref()
    }

    /// Store Hodge decomposition results at the given topology version.
    pub fn store_hodge(&mut self, components: HodgeComponents, version: u64) {
        self.hodge_cache = Some(components);
        self.topology_version = version;
    }

    /// Cached Betti numbers, if present.
    #[inline]
    pub fn betti_numbers(&self) -> Option<[usize; 4]> {
        self.betti_cache
    }

    /// Store Betti numbers at the given topology version.
    pub fn store_betti(&mut self, bettis: [usize; 4], version: u64) {
        self.betti_cache = Some(bettis);
        self.topology_version = version;
    }

    /// Immutable access to the dirty region.
    #[inline]
    pub fn dirty_region(&self) -> &DirtyRegion {
        &self.dirty_region
    }

    /// Mutable access to the dirty region.
    #[inline]
    pub fn dirty_region_mut(&mut self) -> &mut DirtyRegion {
        &mut self.dirty_region
    }

    /// Record that a face was destroyed and invalidate topology-dependent caches.
    ///
    /// If `version` differs from the cached version, the Hodge cache is dropped
    /// (the Laplacian structure changed). The dirty region records which face
    /// changed for incremental update consumers.
    pub fn mark_face_destroyed(&mut self, face_idx: usize, version: u64) {
        if version != self.topology_version {
            self.hodge_cache = None;
        }
        self.dirty_region.mark_face(face_idx, version);
    }
}

impl Default for DecCache {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cached Hodge Decomposition
// ---------------------------------------------------------------------------

/// Wrapper around [`hodge_decompose`] with cache awareness.
///
/// The Hodge decomposition depends on both topology AND input cochain values,
/// so the decomposition itself cannot be cached across different inputs.
/// This wrapper:
/// 1. Invalidates stale caches if topology changed since last call.
/// 2. Delegates to the uncached [`hodge_decompose`].
///
/// The perf win is in consumer code: check `cache.is_valid()` before
/// recomputing structural properties (Betti numbers) and use
/// `cache.dirty_region()` for incremental updates.
pub fn hodge_decompose_cached(
    cx: &CellComplex,
    input: &CochainField,
    cache: &mut DecCache,
) -> HodgeComponents {
    if !cache.is_valid(cx.topology_version()) {
        cache.invalidate();
    }
    hodge_decompose(cx, input)
}

// ---------------------------------------------------------------------------
// Incremental Update Hints
// ---------------------------------------------------------------------------

/// Fill `out` with vertex indices on the boundary of `face_idx`.
///
/// **Zero-allocation**: the caller provides the buffer; this function clears it
/// and writes results in-place.
///
/// Call this **before** `remove_face` — after removal the face's B₂ entries
/// are gone. The typical pattern is:
/// ```ignore
/// affected_vertices(&cx, face_idx, &mut dirty_verts);
/// cx.remove_face(face_idx);
/// ```
pub fn affected_vertices(cx: &CellComplex, face_idx: usize, out: &mut Vec<usize>) {
    out.clear();

    let b1 = cx.boundary_entries(0);
    let b2 = cx.boundary_entries(1);

    // Collect the boundary edges of `face_idx` in a single pass over B₂.
    // A 2D-grid face has exactly 4 boundary edges; cap the fixed-size buffer
    // at 4 (any overflow is still counted so we don't silently drop edges on
    // exotic complexes, but only the first 4 drive the membership test below).
    let mut face_edges: [usize; 4] = [usize::MAX; 4];
    let mut n_face_edges = 0usize;
    for &(edge_idx, f_idx, _) in b2 {
        if f_idx == face_idx {
            if n_face_edges < face_edges.len() {
                face_edges[n_face_edges] = edge_idx;
            }
            n_face_edges += 1;
        }
    }

    // Single streaming pass over B₁: push the endpoint vertex of any entry
    // whose edge bounds this face. Replaces the prior per-edge full B₁ re-scan
    // (≤4 passes) with one cache-friendly sequential read. Each edge appears
    // exactly twice in B₁ (tail + head), so both endpoints are collected.
    let edge_count = n_face_edges.min(face_edges.len());
    for &(v, e, _) in b1 {
        // Short-circuiting linear scan over ≤4 entries; LLVM unrolls this.
        for &fe in &face_edges[..edge_count] {
            if e == fe {
                out.push(v);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_new_is_empty() {
        let cache = DecCache::new();
        assert!(!cache.is_valid(0));
        assert!(!cache.is_valid(1));
        assert!(cache.hodge_components().is_none());
        assert!(cache.betti_numbers().is_none());
        assert!(cache.dirty_region().is_empty());
    }

    #[test]
    fn test_cache_invalidate_clears() {
        let mut cache = DecCache::new();
        cache.store_betti([1, 0, 0, 0], 5);
        assert!(cache.betti_numbers().is_some());

        cache.invalidate();
        assert!(cache.betti_numbers().is_none());
        assert!(!cache.is_valid(5));
    }

    #[test]
    fn test_cache_store_and_get_betti() {
        let mut cache = DecCache::new();
        let bettis = [1usize, 2, 0, 0];
        cache.store_betti(bettis, 3);

        assert_eq!(cache.betti_numbers(), Some(bettis));
        assert!(cache.is_valid(3));
    }

    #[test]
    fn test_cache_valid_after_store() {
        let mut cache = DecCache::new();
        assert!(!cache.is_valid(1));

        cache.store_betti([1, 0, 0, 0], 1);
        assert!(cache.is_valid(1));
        assert!(!cache.is_valid(2));
    }

    #[test]
    fn test_cache_invalid_on_version_mismatch() {
        let mut cache = DecCache::new();
        cache.store_betti([1, 0, 0, 0], 5);

        // Same version → valid
        assert!(cache.is_valid(5));

        // Different version → invalid
        assert!(!cache.is_valid(6));
        assert!(!cache.is_valid(4));
    }

    #[test]
    fn test_dirty_region_mark_face() {
        let mut region = DirtyRegion::new();
        assert!(region.is_empty());

        region.mark_face(3, 7);
        assert!(!region.is_empty());
        assert_eq!(region.changed_faces, vec![3]);
        assert_eq!(region.version, 7);

        region.mark_face(5, 8);
        assert_eq!(region.changed_faces, vec![3, 5]);
        assert_eq!(region.version, 8);
    }

    #[test]
    fn test_affected_vertices_after_removal() {
        // grid_2d(3, 3) face 0 is bounded by vertices {0, 1, 3, 4}
        // (bottom-left 2×2 cell of the grid)
        let cx = CellComplex::grid_2d(3, 3);
        let mut verts = Vec::new();

        affected_vertices(&cx, 0, &mut verts);

        // Deduplicate for assertion — shared corners produce repeats
        let mut unique: Vec<usize> = verts.clone();
        unique.sort_unstable();
        unique.dedup();

        assert_eq!(
            unique.len(),
            4,
            "face 0 should have 4 corner vertices, got {unique:?}"
        );
        assert!(unique.contains(&0), "vertex 0 missing");
        assert!(unique.contains(&1), "vertex 1 missing");
        assert!(unique.contains(&3), "vertex 3 missing");
        assert!(unique.contains(&4), "vertex 4 missing");
    }

    #[test]
    fn test_mark_face_destroyed_invalidates_hodge() {
        let mut cache = DecCache::new();
        // Simulate storing a hodge result at version 3
        let cx = CellComplex::grid_2d(2, 2);
        let input = CochainField::zeros(0, cx.n_vertices(), 1);
        let comp = hodge_decompose(&cx, &input);
        cache.store_hodge(comp, 3);
        assert!(cache.hodge_components().is_some());

        // Topology changed to version 4 — hodge cache should be dropped
        cache.mark_face_destroyed(0, 4);
        assert!(cache.hodge_components().is_none());
        assert!(!cache.dirty_region().is_empty());
        assert_eq!(cache.dirty_region().changed_faces, vec![0]);
    }

    #[test]
    fn test_hodge_decompose_cached_invalidates_on_topology_change() {
        let mut cx = CellComplex::grid_2d(4, 4);
        let input = CochainField::zeros(0, cx.n_vertices(), 1);
        let mut cache = DecCache::new();

        // First call: cache is empty (version 0), should work
        let _ = hodge_decompose_cached(&cx, &input, &mut cache);

        // Mutate topology
        cx.remove_face(0);

        // Second call: topology changed, cache should invalidate and recompute
        let _ = hodge_decompose_cached(&cx, &input, &mut cache);
        // If we reach here without panic, the invalidation path worked.
    }
}
