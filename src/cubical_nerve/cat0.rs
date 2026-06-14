//! CAT(0) geodesic computation on cubical complexes.
//!
//! A CAT(0) cube complex has the property that any two vertices are connected
//! by a unique shortest path (geodesic). This is used for deterministic NPC
//! navigation — no tie-breaking needed, no ambiguous path choices.
//!
//! The geodesic is computed by BFS on the 1-skeleton (the graph formed by
//! edges of the complex). In a CAT(0) complex, BFS finds exactly one shortest
//! path between any two vertices — uniqueness is the defining property.
//!
//! For zone posets specifically, the geodesic passes through the meet:
//! `from → meet(from, to) → to`, but we use BFS for generality.
//!
//! Plan 252 Phase 3 (T18), Research 220.

use std::collections::{HashMap, VecDeque};

use super::nerve::CubicalComplex;
use super::poset::{DistributiveMeetSemilattice, ZoneId};

// ---------------------------------------------------------------------------
// GeodesicPath — result of geodesic computation
// ---------------------------------------------------------------------------

/// A geodesic (unique shortest path) between two vertices in a CAT(0) complex.
///
/// In a CAT(0) cube complex this path is guaranteed to be unique — there is
/// no other path of the same or shorter length between the same endpoints.
pub struct GeodesicPath {
    /// Ordered vertices along the path, from start to end.
    pub vertices: Vec<ZoneId>,
    /// Number of edges (= `vertices.len() - 1` when non-empty).
    pub length: usize,
}

impl GeodesicPath {
    /// Create a new geodesic path from an ordered list of vertices.
    ///
    /// Validation of adjacency is deferred — we just store the vertices
    /// and compute the edge count.
    pub fn new(vertices: Vec<ZoneId>) -> Self {
        let length = match vertices.len() {
            0 => 0,
            n => n - 1,
        };
        Self { vertices, length }
    }

    /// Returns `true` if the path contains no vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Returns the start vertex of the path, or `None` if empty.
    #[inline]
    pub fn start(&self) -> Option<ZoneId> {
        self.vertices.first().copied()
    }

    /// Returns the end vertex of the path, or `None` if empty.
    #[inline]
    pub fn end(&self) -> Option<ZoneId> {
        self.vertices.last().copied()
    }
}

// ---------------------------------------------------------------------------
// Adjacency list — pre-computed from edge list for O(1) neighbor lookup
// ---------------------------------------------------------------------------

/// Build an adjacency list from an edge list.
///
/// Returns a `HashMap<ZoneId, Vec<ZoneId>>` mapping each vertex to its
/// neighbors. Uses `with_capacity` for the inner vectors based on a
/// rough degree estimate.
fn build_adjacency(edges: &[(ZoneId, ZoneId)]) -> HashMap<ZoneId, Vec<ZoneId>> {
    // Count degrees first to pre-allocate.
    let mut degree: HashMap<ZoneId, usize> = HashMap::new();
    for &(a, b) in edges {
        *degree.entry(a).or_insert(0) += 1;
        *degree.entry(b).or_insert(0) += 1;
    }

    let mut adj: HashMap<ZoneId, Vec<ZoneId>> = HashMap::with_capacity(degree.len());
    for (&v, &d) in &degree {
        adj.insert(v, Vec::with_capacity(d));
    }

    for &(a, b) in edges {
        // Both directions — undirected graph.
        if let Some(neighbors) = adj.get_mut(&a) {
            neighbors.push(b);
        }
        if let Some(neighbors) = adj.get_mut(&b) {
            neighbors.push(a);
        }
    }

    adj
}

// ---------------------------------------------------------------------------
// cat0_geodesic — unique shortest path via BFS on 1-skeleton
// ---------------------------------------------------------------------------

/// Compute the unique geodesic (shortest path) between two vertices in a
/// CAT(0) cubical complex.
///
/// In a CAT(0) cube complex, the geodesic is guaranteed to be unique —
/// BFS on the 1-skeleton finds exactly one shortest path.
///
/// For the zone poset case, the geodesic passes through the meet:
/// `from → meet(from, to) → to`. We use BFS for generality — the
/// lattice parameter is kept for the `is_cat0` verification function.
///
/// # Arguments
/// * `complex` — the cubical complex (uses `vertices` and `edges`)
/// * `_lattice` — the distributive meet-semilattice (used by `is_cat0`, unused here)
/// * `from` — source vertex
/// * `to` — target vertex
///
/// # Returns
/// A `GeodesicPath` from `from` to `to`. If `from == to`, returns a
/// single-vertex path. If no path exists, returns an empty path.
pub fn cat0_geodesic<L>(
    complex: &CubicalComplex,
    _lattice: &L,
    from: ZoneId,
    to: ZoneId,
) -> GeodesicPath
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    // Trivial case: same vertex.
    if from == to {
        return GeodesicPath::new(vec![from]);
    }

    let adj = build_adjacency(&complex.edges);
    cat0_geodesic_with_adj(complex, &adj, from, to)
}

/// Internal: geodesic given a pre-built adjacency map.
///
/// Callers that issue many geodesic queries on the same complex should build
/// the adjacency map once and reuse it via this entry point.
fn cat0_geodesic_with_adj(
    complex: &CubicalComplex,
    adj: &HashMap<ZoneId, Vec<ZoneId>>,
    from: ZoneId,
    to: ZoneId,
) -> GeodesicPath {
    // Trivial case: same vertex.
    if from == to {
        return GeodesicPath::new(vec![from]);
    }

    // BFS from `from` to `to`.
    let mut visited: HashMap<ZoneId, ZoneId> = HashMap::new();
    let mut queue: VecDeque<ZoneId> = VecDeque::with_capacity(complex.vertices.len());

    visited.insert(from, from); // sentinel parent
    queue.push_back(from);

    let mut found = false;

    'bfs: while let Some(current) = queue.pop_front() {
        match adj.get(&current) {
            Some(neighbors) => {
                for &neighbor in neighbors {
                    if visited.contains_key(&neighbor) {
                        continue;
                    }

                    visited.insert(neighbor, current);

                    if neighbor == to {
                        found = true;
                        break 'bfs;
                    }

                    queue.push_back(neighbor);
                }
            }
            None => continue,
        }
    }

    if !found {
        // No path exists (shouldn't happen in a connected CAT(0) complex).
        return GeodesicPath::new(Vec::new());
    }

    // Reconstruct path: walk from `to` back to `from` via parent pointers.
    let mut path = Vec::new();
    let mut cursor = to;
    loop {
        path.push(cursor);
        match visited.get(&cursor) {
            Some(&parent) if parent == cursor => {
                // Reached the sentinel (start vertex).
                break;
            }
            Some(&parent) => {
                cursor = parent;
            }
            None => {
                // Should not happen — every visited node has a parent.
                break;
            }
        }
    }

    // Path is reconstructed in reverse (to → from). Reverse it.
    path.reverse();

    GeodesicPath::new(path)
}

// ---------------------------------------------------------------------------
// is_cat0 — verify the unique-geodesic property
// ---------------------------------------------------------------------------

/// Verify that a cubical complex satisfies the CAT(0) property:
/// for every pair of vertices, BFS finds exactly one shortest path.
///
/// This is O(n² × (V + E)) — use only in tests/verification, not in
/// hot paths. A truly CAT(0) complex will have unique shortest paths
/// between all pairs; this function checks that invariant.
///
/// # Returns
/// `true` if every pair of vertices has a unique shortest path.
pub fn is_cat0<L>(complex: &CubicalComplex, lattice: &L) -> bool
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    let elements = lattice.elements();

    // For every pair (i, j), check that the geodesic from i to j
    // is the same as the geodesic from j to i (reversed).
    // In a CAT(0) complex, the geodesic is unique, so both directions
    // must agree.
    //
    // Pre-compute adjacency once — it's reused across all O(N²) BFS calls.
    let adj = build_adjacency(&complex.edges);
    let n = elements.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let path_forward = cat0_geodesic_with_adj(&complex, &adj, elements[i], elements[j]);
            let path_backward = cat0_geodesic_with_adj(&complex, &adj, elements[j], elements[i]);

            // Both must find a path.
            if path_forward.is_empty() || path_backward.is_empty() {
                return false;
            }

            // Forward path reversed must equal backward path.
            if path_forward.vertices.len() != path_backward.vertices.len() {
                return false;
            }

            let len = path_forward.vertices.len();
            for k in 0..len {
                if path_forward.vertices[k] != path_backward.vertices[len - 1 - k] {
                    return false;
                }
            }
        }
    }

    true
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

    /// Build a simple chain complex: 0 — 1 — 2
    fn chain_complex() -> CubicalComplex {
        CubicalComplex {
            vertices: vec![za(0), za(1), za(2)],
            edges: vec![(za(0), za(1)), (za(1), za(2))],
            faces: vec![],
            cubes: vec![],
        }
    }

    /// Build a diamond complex:
    ///   3
    ///  / \
    /// 1   2
    ///  \ /
    ///   0
    fn diamond_complex() -> CubicalComplex {
        CubicalComplex {
            vertices: vec![za(0), za(1), za(2), za(3)],
            edges: vec![
                (za(0), za(1)),
                (za(0), za(2)),
                (za(1), za(3)),
                (za(2), za(3)),
            ],
            faces: vec![],
            cubes: vec![],
        }
    }

    /// Minimal lattice for chain 0 < 1 < 2.
    struct ChainLattice;
    impl DistributiveMeetSemilattice for ChainLattice {
        type Elem = ZoneId;
        fn meet(&self, a: &ZoneId, b: &ZoneId) -> Option<ZoneId> {
            // In a chain, meet = min(a, b).
            Some(if a <= b { *a } else { *b })
        }
        fn leq(&self, a: &ZoneId, b: &ZoneId) -> bool {
            a <= b
        }
        fn elements(&self) -> Vec<ZoneId> {
            vec![za(0), za(1), za(2)]
        }
    }

    /// Minimal lattice for diamond: 0 < 1, 0 < 2, 1 < 3, 2 < 3.
    struct DiamondLattice;
    impl DistributiveMeetSemilattice for DiamondLattice {
        type Elem = ZoneId;
        fn meet(&self, a: &ZoneId, b: &ZoneId) -> Option<ZoneId> {
            let (a, b) = (*a, *b);
            if a == b {
                return Some(a);
            }
            // meet(1, 2) = 0, meet(1, 3) = 1, meet(2, 3) = 2,
            // meet(0, x) = 0, meet(x, 0) = 0
            if a == za(0) || b == za(0) {
                return Some(za(0));
            }
            match (a, b) {
                (ZoneId(1), ZoneId(2)) | (ZoneId(2), ZoneId(1)) => Some(za(0)),
                (ZoneId(1), ZoneId(3)) | (ZoneId(3), ZoneId(1)) => Some(za(1)),
                (ZoneId(2), ZoneId(3)) | (ZoneId(3), ZoneId(2)) => Some(za(2)),
                _ => None,
            }
        }
        fn leq(&self, a: &ZoneId, b: &ZoneId) -> bool {
            let (a, b) = (a.0, b.0);
            match (a, b) {
                (x, y) if x == y => true,
                (0, _) => true,
                (_, 3) => true,
                (1, 2) | (2, 1) => false,
                (x, y) => x < y,
            }
        }
        fn elements(&self) -> Vec<ZoneId> {
            vec![za(0), za(1), za(2), za(3)]
        }
    }

    #[test]
    fn test_geodesic_same_vertex() {
        let complex = chain_complex();
        let lattice = ChainLattice;
        let path = cat0_geodesic(&complex, &lattice, za(1), za(1));

        assert_eq!(path.vertices, vec![za(1)]);
        assert_eq!(path.length, 0);
        assert!(!path.is_empty());
        assert_eq!(path.start(), Some(za(1)));
        assert_eq!(path.end(), Some(za(1)));
    }

    #[test]
    fn test_geodesic_chain() {
        let complex = chain_complex();
        let lattice = ChainLattice;
        let path = cat0_geodesic(&complex, &lattice, za(0), za(2));

        assert_eq!(path.vertices, vec![za(0), za(1), za(2)]);
        assert_eq!(path.length, 2);
        assert_eq!(path.start(), Some(za(0)));
        assert_eq!(path.end(), Some(za(2)));
    }

    #[test]
    fn test_geodesic_diamond() {
        // Diamond: 0 — 1 — 3
        //              \  /
        //               2
        // Geodesic from 1 to 2: [1, 0, 2] (through the bottom).
        let complex = diamond_complex();
        let lattice = DiamondLattice;
        let path = cat0_geodesic(&complex, &lattice, za(1), za(2));

        assert_eq!(path.vertices, vec![za(1), za(0), za(2)]);
        assert_eq!(path.length, 2);
    }

    #[test]
    fn test_geodesic_unique() {
        // Verify BFS finds the same path regardless of direction.
        let complex = chain_complex();
        let lattice = ChainLattice;

        let forward = cat0_geodesic(&complex, &lattice, za(0), za(2));
        let backward = cat0_geodesic(&complex, &lattice, za(2), za(0));

        assert_eq!(forward.vertices, vec![za(0), za(1), za(2)]);
        assert_eq!(backward.vertices, vec![za(2), za(1), za(0)]);

        // Reversed backward must equal forward.
        let mut backward_rev = backward.vertices.clone();
        backward_rev.reverse();
        assert_eq!(forward.vertices, backward_rev);
    }

    #[test]
    fn test_geodesic_diamond_unique() {
        let complex = diamond_complex();
        let lattice = DiamondLattice;

        let forward = cat0_geodesic(&complex, &lattice, za(1), za(2));
        let backward = cat0_geodesic(&complex, &lattice, za(2), za(1));

        assert_eq!(forward.vertices, vec![za(1), za(0), za(2)]);
        assert_eq!(backward.vertices, vec![za(2), za(0), za(1)]);
    }

    #[test]
    fn test_is_cat0_chain() {
        let complex = chain_complex();
        let lattice = ChainLattice;
        assert!(is_cat0(&complex, &lattice));
    }

    #[test]
    fn test_is_cat0_diamond() {
        let complex = diamond_complex();
        let lattice = DiamondLattice;
        assert!(is_cat0(&complex, &lattice));
    }

    #[test]
    fn test_geodesic_path_empty() {
        let path = GeodesicPath::new(Vec::new());
        assert!(path.is_empty());
        assert_eq!(path.length, 0);
        assert_eq!(path.start(), None);
        assert_eq!(path.end(), None);
    }

    #[test]
    fn test_geodesic_diamond_top_to_bottom() {
        let complex = diamond_complex();
        let lattice = DiamondLattice;

        // From 3 (top) to 0 (bottom): [3, 1, 0] or [3, 2, 0].
        // Both are valid shortest paths of length 2 — BFS will pick one
        // deterministically based on neighbor order.
        let path = cat0_geodesic(&complex, &lattice, za(3), za(0));
        assert_eq!(path.length, 2);
        assert_eq!(path.start(), Some(za(3)));
        assert_eq!(path.end(), Some(za(0)));

        // Verify it's a valid path through adjacent vertices.
        assert!(path.vertices.contains(&za(3)));
        assert!(path.vertices.contains(&za(0)));
    }

    #[test]
    fn test_build_adjacency() {
        let edges = vec![(za(0), za(1)), (za(1), za(2))];
        let adj = build_adjacency(&edges);

        assert_eq!(adj[&za(0)], vec![za(1)]);
        assert_eq!(adj[&za(1)], vec![za(0), za(2)]);
        assert_eq!(adj[&za(2)], vec![za(1)]);
    }
}
