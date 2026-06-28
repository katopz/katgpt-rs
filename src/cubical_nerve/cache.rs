//! Bridge from CAT(0) geodesic to zone-level navigation, and cache for ⊞[L].
//!
//! T19: `NerveFlowField` converts a `GeodesicPath` (zone-level path through
//! the cubical complex) into a queryable navigation guidance structure.
//! Zero allocation on read paths — the zone→index map is built once at
//! construction.
//!
//! T20: `NerveCache` pre-computes the cubical nerve ⊞[L] on map load and
//! caches it for fast geodesic/flow-field queries. Invalidated when topology
//! changes, with version tracking.
//!
//! Plan 252 Phase 3 (T19+T20).

use std::collections::HashMap;

use super::cat0::{GeodesicPath, cat0_geodesic};
use super::nerve::{CubicalComplex, cubical_nerve};
use super::poset::{ZoneId, ZonePoset};

// ---------------------------------------------------------------------------
// NavigationHint — direction hint toward the goal
// ---------------------------------------------------------------------------

/// A navigation hint indicating progress along a geodesic.
///
/// Lightweight value — no allocation, copyable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationHint {
    /// Already at the goal — no movement needed.
    AtGoal,
    /// Moving toward the goal; `remaining` edges left.
    EnRoute { remaining: usize },
    /// Not on the path — caller should route to the nearest path vertex first.
    OffPath,
}

// ---------------------------------------------------------------------------
// NerveFlowField (T19) — zone-level navigation from a CAT(0) geodesic
// ---------------------------------------------------------------------------

/// Zone-level navigation guidance from a CAT(0) geodesic.
///
/// Built from a `GeodesicPath` and provides O(1) lookup for the next zone
/// on the path toward the goal. Zero allocation on query methods.
pub struct NerveFlowField {
    /// The geodesic path (ordered sequence of zones from start to goal).
    path: Vec<ZoneId>,
    /// Map from zone → index in `path` for O(1) lookup.
    zone_index: HashMap<ZoneId, usize>,
}

impl NerveFlowField {
    /// Build a flow field from a geodesic path.
    ///
    /// Takes ownership of the path's vertex list. Constructs the zone→index
    /// lookup map once — all subsequent queries are allocation-free.
    pub fn from_geodesic(path: GeodesicPath) -> Self {
        let zone_index: HashMap<ZoneId, usize> = path
            .vertices
            .iter()
            .enumerate()
            .map(|(i, &z)| (z, i))
            .collect();

        Self {
            path: path.vertices,
            zone_index,
        }
    }

    /// Given the current zone, return the next zone on the path toward the goal.
    ///
    /// Returns `None` if:
    /// - `current` is the goal (already there)
    /// - `current` is not on the path
    ///
    /// O(1) — single HashMap lookup + index increment.
    pub fn next_zone(&self, current: ZoneId) -> Option<ZoneId> {
        let idx = match self.zone_index.get(&current) {
            Some(&i) => i,
            None => return None,
        };

        // Already at goal?
        let next_idx = idx + 1;
        self.path.get(next_idx).copied()
    }

    /// Return a navigation hint for the current zone.
    ///
    /// Indicates whether the NPC is at the goal, en route, or off the path.
    /// O(1) — no allocation.
    pub fn direction_to_goal(&self, current: ZoneId) -> NavigationHint {
        let idx = match self.zone_index.get(&current) {
            Some(&i) => i,
            None => return NavigationHint::OffPath,
        };

        let remaining = self.path.len().saturating_sub(idx + 1);
        match remaining {
            0 => NavigationHint::AtGoal,
            n => NavigationHint::EnRoute { remaining: n },
        }
    }

    /// Returns the goal zone of this flow field, or `None` if empty.
    #[inline]
    pub fn goal(&self) -> Option<ZoneId> {
        self.path.last().copied()
    }

    /// Returns the start zone of this flow field, or `None` if empty.
    #[inline]
    pub fn start(&self) -> Option<ZoneId> {
        self.path.first().copied()
    }

    /// Returns the number of zones in the path.
    #[inline]
    pub fn len(&self) -> usize {
        self.path.len()
    }

    /// Returns `true` if the path is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }

    /// Read-only access to the underlying path vertices.
    #[inline]
    pub fn path(&self) -> &[ZoneId] {
        &self.path
    }
}

// ---------------------------------------------------------------------------
// NerveCache (T20) — cached cubical nerve ⊞[L] for a map topology
// ---------------------------------------------------------------------------

/// Cached cubical nerve for a map topology.
///
/// Pre-computed on map load. Invalidated when topology changes.
/// Read methods are zero-allocation.
pub struct NerveCache {
    /// The pre-computed cubical complex.
    complex: CubicalComplex,
    /// Zone poset used to build the nerve.
    poset: ZonePoset,
    /// Topology version — incremented on each invalidation.
    version: u64,
}

impl NerveCache {
    /// Build the cubical nerve on construction.
    ///
    /// Computes the full cubical complex from the zone poset. This is the
    /// expensive operation — subsequent reads are cheap.
    pub fn new(poset: ZonePoset) -> Self {
        let complex = cubical_nerve(&poset);
        Self {
            complex,
            poset,
            version: 1,
        }
    }

    /// Read-only access to the cached cubical complex.
    #[inline]
    pub fn complex(&self) -> &CubicalComplex {
        &self.complex
    }

    /// Read-only access to the zone poset.
    #[inline]
    pub fn poset(&self) -> &ZonePoset {
        &self.poset
    }

    /// Current topology version.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Rebuild the nerve when topology changes.
    ///
    /// Reconstructs the cubical complex from the new poset and bumps the
    /// version counter.
    pub fn invalidate(&mut self, poset: ZonePoset) {
        self.complex = cubical_nerve(&poset);
        self.poset = poset;
        self.version = self.version.saturating_add(1);
    }

    /// Convenience: compute a geodesic using the cached complex.
    ///
    /// No re-computation of the complex — reuses the cached version.
    pub fn geodesic(&self, from: ZoneId, to: ZoneId) -> GeodesicPath {
        cat0_geodesic(&self.complex, &self.poset, from, to)
    }

    /// Convenience: compute a flow field using the cached complex.
    ///
    /// Computes the geodesic, then wraps it in a `NerveFlowField`.
    pub fn flow_field(&self, from: ZoneId, to: ZoneId) -> NerveFlowField {
        let path = self.geodesic(from, to);
        NerveFlowField::from_geodesic(path)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn za(id: usize) -> ZoneId {
        ZoneId::new(id)
    }

    /// Build a chain poset 0 < 1 < ... < n-1.
    fn chain_poset(n: usize) -> ZonePoset {
        let zones: Vec<ZoneId> = (0..n).map(za).collect();
        let pairs: Vec<(ZoneId, ZoneId)> = (0..n - 1).map(|i| (za(i), za(i + 1))).collect();
        ZonePoset::from_edges(zones, pairs)
    }

    /// Build the diamond poset 0 < 1 < 3, 0 < 2 < 3.
    fn diamond_poset() -> ZonePoset {
        ZonePoset::from_edges(
            vec![za(0), za(1), za(2), za(3)],
            vec![
                (za(0), za(1)),
                (za(0), za(2)),
                (za(1), za(3)),
                (za(2), za(3)),
            ],
        )
    }

    // --- T19: NerveFlowField tests ---

    #[test]
    fn test_nerve_flow_field_follow_path() {
        // Chain: 0 — 1 — 2 — 3 — 4
        let cache = NerveCache::new(chain_poset(5));
        let ff = cache.flow_field(za(0), za(4));

        // Walk from start to goal, asking for next zone at each step.
        assert_eq!(ff.start(), Some(za(0)));
        assert_eq!(ff.goal(), Some(za(4)));

        let mut current = za(0);
        let mut visited = vec![current];

        while let Some(next) = ff.next_zone(current) {
            visited.push(next);
            current = next;
        }

        assert_eq!(visited, vec![za(0), za(1), za(2), za(3), za(4)]);

        // At goal — next_zone returns None.
        assert_eq!(ff.next_zone(za(4)), None);
        assert_eq!(ff.direction_to_goal(za(4)), NavigationHint::AtGoal);
    }

    #[test]
    fn test_nerve_flow_field_off_path_returns_none() {
        // Chain: 0 — 1 — 2
        let cache = NerveCache::new(chain_poset(3));
        let ff = cache.flow_field(za(0), za(2));

        // Zone 99 is not on the path.
        assert_eq!(ff.next_zone(za(99)), None);
        assert_eq!(ff.direction_to_goal(za(99)), NavigationHint::OffPath);
    }

    #[test]
    fn test_nerve_flow_field_en_route_hint() {
        let cache = NerveCache::new(chain_poset(5));
        let ff = cache.flow_field(za(0), za(4));

        // At start: 4 edges remaining.
        assert_eq!(
            ff.direction_to_goal(za(0)),
            NavigationHint::EnRoute { remaining: 4 }
        );
        // Midway: 2 edges remaining.
        assert_eq!(
            ff.direction_to_goal(za(2)),
            NavigationHint::EnRoute { remaining: 2 }
        );
        // At goal.
        assert_eq!(ff.direction_to_goal(za(4)), NavigationHint::AtGoal);
    }

    #[test]
    fn test_nerve_flow_field_diamond() {
        let cache = NerveCache::new(diamond_poset());
        let ff = cache.flow_field(za(1), za(2));

        // Geodesic in diamond from 1→2 goes through 0: [1, 0, 2].
        assert_eq!(ff.path(), &[za(1), za(0), za(2)]);
        assert_eq!(ff.next_zone(za(1)), Some(za(0)));
        assert_eq!(ff.next_zone(za(0)), Some(za(2)));
        assert_eq!(ff.next_zone(za(2)), None);
    }

    #[test]
    fn test_nerve_flow_field_single_zone() {
        let cache = NerveCache::new(chain_poset(1));
        let ff = cache.flow_field(za(0), za(0));

        assert_eq!(ff.path(), &[za(0)]);
        assert_eq!(ff.next_zone(za(0)), None);
        assert_eq!(ff.direction_to_goal(za(0)), NavigationHint::AtGoal);
        assert_eq!(ff.goal(), Some(za(0)));
    }

    // --- T20: NerveCache tests ---

    #[test]
    fn test_nerve_cache_build() {
        let poset = diamond_poset();
        let cache = NerveCache::new(poset);

        // Diamond: 4 vertices, 4 edges, 1 face.
        assert_eq!(cache.complex().n_vertices(), 4);
        assert_eq!(cache.complex().n_edges(), 4);
        assert_eq!(cache.complex().n_faces(), 1);
        assert_eq!(cache.version(), 1);
    }

    #[test]
    fn test_nerve_cache_invalidate() {
        // Start with chain of 3.
        let mut cache = NerveCache::new(chain_poset(3));
        assert_eq!(cache.complex().n_vertices(), 3);
        assert_eq!(cache.complex().n_edges(), 2);

        // Invalidate with diamond.
        cache.invalidate(diamond_poset());
        assert_eq!(cache.complex().n_vertices(), 4);
        assert_eq!(cache.complex().n_edges(), 4);
        assert_eq!(cache.complex().n_faces(), 1);
    }

    #[test]
    fn test_nerve_cache_version_bumps() {
        let mut cache = NerveCache::new(chain_poset(3));
        let v0 = cache.version();
        assert_eq!(v0, 1);

        cache.invalidate(chain_poset(4));
        assert_eq!(cache.version(), 2);

        cache.invalidate(chain_poset(5));
        assert_eq!(cache.version(), 3);
    }

    #[test]
    fn test_nerve_cache_geodesic_convenience() {
        let cache = NerveCache::new(chain_poset(4));

        let path = cache.geodesic(za(0), za(3));
        assert_eq!(path.vertices, vec![za(0), za(1), za(2), za(3)]);
        assert_eq!(path.length, 3);

        // Reverse direction.
        let rev = cache.geodesic(za(3), za(0));
        assert_eq!(rev.vertices, vec![za(3), za(2), za(1), za(0)]);
    }

    #[test]
    fn test_nerve_cache_flow_field_convenience() {
        let cache = NerveCache::new(diamond_poset());
        let ff = cache.flow_field(za(0), za(3));

        assert_eq!(ff.start(), Some(za(0)));
        assert_eq!(ff.goal(), Some(za(3)));

        // Path from 0 to 3 in diamond goes through either 1 or 2.
        // Verify we can follow it to the goal.
        let mut current = za(0);
        let mut steps = 0;
        while let Some(next) = ff.next_zone(current) {
            current = next;
            steps += 1;
        }
        assert_eq!(current, za(3));
        assert_eq!(steps, 2); // 0 → {1|2} → 3
    }
}
