//! Cubical nerve construction from distributive meet-semilattices.
//!
//! The cubical nerve functor ⊞ (arXiv:2503.13663) sends a distributive
//! meet-semilattice L to a cubical set ⊞[L] whose geometric realization
//! is a CAT(0) cube complex, guaranteeing unique geodesics.
//!
//! **Vertices (0-cubes)**: elements of L.
//! **Edges (1-cubes)**: covering pairs (a, b) where a < b and no c with a < c < b.
//! **Faces (2-cubes)**: commuting squares of four distinct zones in the Hasse diagram.
//!
//! Plan 252 Phase 3 (T17).

use std::collections::HashSet;

use super::poset::{DistributiveMeetSemilattice, ZoneId};

// ---------------------------------------------------------------------------
// CubicalCube — single cube of various dimensions
// ---------------------------------------------------------------------------

/// A single cube in the cubical complex, tagged by dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CubicalCube {
    /// 0-cube: a vertex (zone).
    Vertex { zone: ZoneId },
    /// 1-cube: an edge where `from` is covered by `to`.
    Edge { from: ZoneId, to: ZoneId },
    /// 2-cube: a face with 4 corner zones (sorted by ZoneId).
    Face { corners: [ZoneId; 4] },
    /// 3-cube: a cube with 8 corner zones (sorted by ZoneId).
    Cube { corners: [ZoneId; 8] },
}

// ---------------------------------------------------------------------------
// CubicalComplex — the full cubical complex
// ---------------------------------------------------------------------------

/// A cubical complex constructed from the cubical nerve functor ⊞.
///
/// After construction, all fields are frozen — zero allocation on read paths.
#[derive(Debug, Clone)]
pub struct CubicalComplex {
    /// All 0-cubes (zones / vertices).
    pub vertices: Vec<ZoneId>,
    /// All 1-cubes: `(from, to)` where `from` is covered by `to`.
    pub edges: Vec<(ZoneId, ZoneId)>,
    /// All 2-cubes: 4 corner zones (sorted by ZoneId for deduplication).
    pub faces: Vec<[ZoneId; 4]>,
    /// All 3-cubes: 8 corner zones (future).
    pub cubes: Vec<[ZoneId; 8]>,
}

impl CubicalComplex {
    /// Number of vertices (0-cubes).
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of edges (1-cubes).
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of faces (2-cubes).
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
}

// ---------------------------------------------------------------------------
// cubical_nerve — main construction
// ---------------------------------------------------------------------------

/// Construct the cubical nerve ⊞[L] from a distributive meet-semilattice.
///
/// # Algorithm
///
/// 1. **Vertices**: all elements of L.
/// 2. **Edges**: covering pairs (a, b) — a < b and no c exists with a < c < b.
/// 3. **Faces**: pairs of covering edges (a,b), (c,d) that form a commuting
///    square: a ≤ c and b ≤ d (or the reverse), all four corners distinct.
///
/// Faces are deduplicated by their sorted corner set.
pub fn cubical_nerve<L>(lattice: &L) -> CubicalComplex
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    let elements = lattice.elements();

    let vertices = elements.clone();

    // --- Edges: covering relations ---
    let edges = build_covering_edges(lattice, &elements);

    // --- Faces: commuting squares in the Hasse diagram ---
    let faces = build_faces(lattice, &elements, &edges);

    CubicalComplex {
        vertices,
        edges,
        faces,
        cubes: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Edge construction — covering pairs
// ---------------------------------------------------------------------------

/// Find all covering pairs in the lattice.
///
/// (a, b) is a covering pair iff a ≠ b, a ≤ b, and no element c satisfies
/// a ≤ c ≤ b with c ≠ a and c ≠ b.
fn build_covering_edges<L>(lattice: &L, elements: &[ZoneId]) -> Vec<(ZoneId, ZoneId)>
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    let n = elements.len();
    let mut edges = Vec::with_capacity(n * n / 2); // upper bound hint

    for i in 0..n {
        let a = elements[i];
        for j in 0..n {
            let b = elements[j];
            if a == b {
                continue;
            }
            if !lattice.leq(&a, &b) {
                continue;
            }
            if is_covered(lattice, elements, a, b) {
                edges.push((a, b));
            }
        }
    }

    edges
}

/// Check that `b` covers `a` — no element strictly between them.
#[inline]
fn is_covered<L>(lattice: &L, elements: &[ZoneId], a: ZoneId, b: ZoneId) -> bool
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    for &c in elements {
        if c == a || c == b {
            continue;
        }
        if lattice.leq(&a, &c) && lattice.leq(&c, &b) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Face construction — 2-cubes (commuting squares)
// ---------------------------------------------------------------------------

/// Find all 2-cubes (faces) from pairs of covering edges that form
/// commuting squares.
///
/// A face requires all four sides to be covering relations in the
/// Hasse diagram. Given covering edges (a,b) and (c,d), a face exists
/// when:
/// - All four zones {a, b, c, d} are distinct.
/// - (a,c) is a covering relation and (b,d) is a covering relation
///   (or the reverse: (c,a) covering and (d,b) covering).
///
/// This ensures the square has edges on all four sides, forming a
/// proper 2-cube in the cubical nerve.
fn build_faces<L>(_lattice: &L, _elements: &[ZoneId], edges: &[(ZoneId, ZoneId)]) -> Vec<[ZoneId; 4]>
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    // Build a set of covering pairs for O(1) lookup.
    let mut cover_set: HashSet<(ZoneId, ZoneId)> = HashSet::with_capacity(edges.len());
    for &(a, b) in edges {
        cover_set.insert((a, b));
    }

    let n_edges = edges.len();
    let mut faces = Vec::new();
    let mut seen: HashSet<[ZoneId; 4]> = HashSet::with_capacity(n_edges);

    for i in 0..n_edges {
        let (a, b) = edges[i];
        for j in (i + 1)..n_edges {
            let (c, d) = edges[j];

            // All four corners must be distinct.
            if a == c || a == d || b == c || b == d {
                continue;
            }

            // All four sides must be covering relations.
            // Orientation 1: a→b, c→d are "vertical"; a→c, b→d are "horizontal".
            let fwd_horiz = cover_set.contains(&(a, c)) && cover_set.contains(&(b, d));
            // Orientation 2: a→b, c→d are "vertical"; c→a, d→b are "horizontal".
            let rev_horiz = cover_set.contains(&(c, a)) && cover_set.contains(&(d, b));

            if !fwd_horiz && !rev_horiz {
                continue;
            }

            // Sort corners for canonical representation and deduplication.
            let mut corners = [a, c, b, d];
            corners.sort();

            if seen.insert(corners) {
                faces.push(corners);
            }
        }
    }

    faces
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::poset::{ZoneId, ZonePoset};
    use super::cubical_nerve;

    fn za(id: usize) -> ZoneId {
        ZoneId::new(id)
    }

    // Helper: build a chain poset 0 < 1 < ... < n-1.
    fn chain_poset(n: usize) -> ZonePoset {
        let zones: Vec<ZoneId> = (0..n).map(za).collect();
        let pairs: Vec<(ZoneId, ZoneId)> = (0..n - 1).map(|i| (za(i), za(i + 1))).collect();
        ZonePoset::from_edges(zones, pairs)
    }

    // Helper: build the diamond poset 0 < 1 < 3, 0 < 2 < 3.
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

    #[test]
    fn test_cubical_nerve_chain() {
        // Chain 0 < 1 < 2: 3 vertices, 2 edges, 0 faces.
        let poset = chain_poset(3);
        let complex = cubical_nerve(&poset);

        assert_eq!(complex.n_vertices(), 3);
        assert_eq!(complex.n_edges(), 2);
        assert_eq!(complex.n_faces(), 0);
        assert_eq!(complex.cubes.len(), 0);
    }

    #[test]
    fn test_cubical_nerve_diamond() {
        // Diamond: 0 < 1 < 3, 0 < 2 < 3
        // 4 vertices, 4 edges, 1 face (the square {0,1,2,3}).
        let poset = diamond_poset();
        let complex = cubical_nerve(&poset);

        assert_eq!(complex.n_vertices(), 4, "diamond should have 4 vertices");
        assert_eq!(complex.n_edges(), 4, "diamond should have 4 edges");
        assert_eq!(complex.n_faces(), 1, "diamond should have 1 face");

        let face = &complex.faces[0];
        let mut sorted = *face;
        sorted.sort();
        assert_eq!(sorted, [za(0), za(1), za(2), za(3)]);
    }

    #[test]
    fn test_cubical_nerve_single_zone() {
        // Single zone: 1 vertex, 0 edges, 0 faces.
        let poset = ZonePoset::from_edges(vec![za(0)], vec![]);
        let complex = cubical_nerve(&poset);

        assert_eq!(complex.n_vertices(), 1);
        assert_eq!(complex.n_edges(), 0);
        assert_eq!(complex.n_faces(), 0);
    }

    #[test]
    fn test_cubical_complex_dimensions() {
        let poset = diamond_poset();
        let complex = cubical_nerve(&poset);

        // Verify the accessor methods match the field lengths.
        assert_eq!(complex.n_vertices(), complex.vertices.len());
        assert_eq!(complex.n_edges(), complex.edges.len());
        assert_eq!(complex.n_faces(), complex.faces.len());
    }

    #[test]
    fn test_covering_edges_are_minimal() {
        // Chain 0 < 1 < 2 < 3: only direct covering edges, no transitive ones.
        let poset = chain_poset(4);
        let complex = cubical_nerve(&poset);

        assert_eq!(
            complex.n_edges(),
            3,
            "chain of 4 should have 3 covering edges"
        );

        // (0,1), (1,2), (2,3) — no (0,2) or (0,3) or (1,3).
        for (from, to) in &complex.edges {
            assert!(
                (from.as_usize() + 1) == to.as_usize(),
                "edge ({from:?}, {to:?}) should be a direct covering, got gap"
            );
        }
    }

    #[test]
    fn test_no_face_in_chain() {
        // Chain 0 < 1 < 2 < 3 < 4: no squares possible.
        let poset = chain_poset(5);
        let complex = cubical_nerve(&poset);

        assert_eq!(complex.n_vertices(), 5);
        assert_eq!(complex.n_edges(), 4);
        assert_eq!(complex.n_faces(), 0);
    }

    #[test]
    fn test_disconnected_poset() {
        // Three unrelated zones: no edges, no faces.
        let poset = ZonePoset::from_edges(vec![za(0), za(1), za(2)], vec![]);
        let complex = cubical_nerve(&poset);

        assert_eq!(complex.n_vertices(), 3);
        assert_eq!(complex.n_edges(), 0);
        assert_eq!(complex.n_faces(), 0);
    }

    #[test]
    fn test_two_diamonds_stacked() {
        // Two diamonds sharing an edge:
        //   0 < 1 < 3, 0 < 2 < 3  (bottom diamond)
        //   3 < 4 < 6, 3 < 5 < 6  (top diamond)
        // Expected: 7 vertices, 8 edges, 2 faces.
        let poset = ZonePoset::from_edges(
            vec![za(0), za(1), za(2), za(3), za(4), za(5), za(6)],
            vec![
                (za(0), za(1)),
                (za(0), za(2)),
                (za(1), za(3)),
                (za(2), za(3)),
                (za(3), za(4)),
                (za(3), za(5)),
                (za(4), za(6)),
                (za(5), za(6)),
            ],
        );
        let complex = cubical_nerve(&poset);

        assert_eq!(complex.n_vertices(), 7);
        assert_eq!(complex.n_edges(), 8, "stacked diamonds should have 8 edges");
        assert_eq!(complex.n_faces(), 2, "stacked diamonds should have 2 faces");
    }

    #[test]
    fn test_vertices_contain_all_elements() {
        let poset = diamond_poset();
        let complex = cubical_nerve(&poset);

        let mut sorted = complex.vertices.clone();
        sorted.sort();

        assert_eq!(sorted, vec![za(0), za(1), za(2), za(3)]);
    }

    #[test]
    fn test_face_corners_are_sorted() {
        let poset = diamond_poset();
        let complex = cubical_nerve(&poset);

        for face in &complex.faces {
            for w in face.windows(2) {
                assert!(w[0] <= w[1], "face corners should be sorted: {:?}", face);
            }
        }
    }
}
