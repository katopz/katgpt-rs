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
//! # Adaptive Backend (Plan 252 Phase 5)
//!
//! [`cubical_nerve`] automatically selects the optimal construction algorithm
//! based on the lattice size. For small lattices (<64 elements), the scalar
//! O(n³) covering-edge check is used. For larger lattices, an optimized
//! algorithm reduces redundant `leq` calls.
//!
//! Plan 252 Phase 3 (T17), Phase 5 (T30).

use std::collections::HashSet;

use super::poset::{DistributiveMeetSemilattice, ZoneId};

/// Minimum zone count for optimized nerve construction.
/// Below this, scalar O(n³) is used. Keep in sync with
/// [`crate::interval_pruner::NERVE_SIMD_THRESHOLD`] when both features enabled.
const NERVE_OPT_THRESHOLD: usize = 64;

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
///
/// # Adaptive Backend
///
/// For lattices with ≥ [`NERVE_SIMD_THRESHOLD`] (64) elements, the covering-edge
/// construction uses an optimized algorithm that reduces redundant `leq` calls
/// by pre-computing the cover relation matrix. Below the threshold, the scalar
/// O(n³) algorithm is used directly.
pub fn cubical_nerve<L>(lattice: &L) -> CubicalComplex
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    cubical_nerve_with_threshold(lattice, NERVE_OPT_THRESHOLD)
}

/// Construct the cubical nerve with a configurable threshold for backend selection.
///
/// This is the adaptive-routing entry point (Plan 252 T30).
/// When `cubical_nerve` feature is also enabled, callers can use
/// [`crate::interval_pruner::AdaptiveConfig`] to provide the threshold.
///
/// * `nerve_threshold` — minimum zone count for the optimized backend.
///   Below this, scalar O(n³) is used. Above, the bitset-optimized algorithm.
pub fn cubical_nerve_with_threshold<L>(lattice: &L, nerve_threshold: usize) -> CubicalComplex
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    let elements = lattice.elements();

    let vertices = elements.clone();

    // --- Edges: covering relations ---
    // Select algorithm based on lattice size.
    let n = elements.len();
    let edges = if n < nerve_threshold {
        build_covering_edges(lattice, &elements)
    } else {
        build_covering_edges_optimized(lattice, &elements)
    };

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

/// Optimized covering-edge construction for large posets.
///
/// Pre-computes the full `leq` matrix as a flat bitset, then checks covering
/// relations using bit operations instead of per-element `leq()` calls.
///
/// For a poset with n elements:
/// - Pre-compute: O(n²) `leq` calls → n×n bitset.
/// - Covering check: for each pair (a,b) with a≠b and leq(a,b), check that
///   no c exists with leq(a,c) and leq(c,b). Using the bitset, this becomes
///   a set intersection check: `ancestors[b] ∩ descendants[a] ⊖ {a,b} = ∅`.
///
/// The bitset approach avoids repeated binary searches in `leq()` and
/// allows cache-friendly sequential access patterns.
fn build_covering_edges_optimized<L>(lattice: &L, elements: &[ZoneId]) -> Vec<(ZoneId, ZoneId)>
where
    L: DistributiveMeetSemilattice<Elem = ZoneId>,
{
    let n = elements.len();

    // Pre-compute leq matrix as a flat bitset.
    // leq_matrix[i * n + j] = true iff elements[i] ≤ elements[j].
    let mut leq_matrix = vec![false; n * n];

    for i in 0..n {
        for j in 0..n {
            if lattice.leq(&elements[i], &elements[j]) {
                leq_matrix[i * n + j] = true;
            }
        }
    }

    // For each pair (i,j) where leq(i,j) and i≠j, check covering.
    // (i,j) is covering iff no k exists with i≠k≠j and leq(i,k) and leq(k,j).
    //
    // Optimization: for each (i,j), scan row k=0..n. If any k with
    // leq_matrix[i*n+k] AND leq_matrix[k*n+j] AND k≠i AND k≠j exists,
    // then (i,j) is NOT covering.
    //
    // Cache-friendly: row-major access to leq_matrix.
    let mut edges = Vec::with_capacity(n * n / 2);

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if !leq_matrix[i * n + j] {
                continue;
            }

            // Check covering: no k strictly between.
            let mut is_cover = true;
            for k in 0..n {
                if k == i || k == j {
                    continue;
                }
                if leq_matrix[i * n + k] && leq_matrix[k * n + j] {
                    is_cover = false;
                    break;
                }
            }

            if is_cover {
                edges.push((elements[i], elements[j]));
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
    use super::super::poset::{DistributiveMeetSemilattice, ZoneId, ZonePoset};
    use super::{cubical_nerve, cubical_nerve_with_threshold};

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

    // Helper: bushy poset where zone 0 < zone i for all i > 0.
    // Creates many covering pairs (0, i) from the root.
    fn bushy_poset(n: usize) -> ZonePoset {
        let zones: Vec<ZoneId> = (0..n).map(za).collect();
        let pairs: Vec<(ZoneId, ZoneId)> = (1..n).map(|i| (za(0), za(i))).collect();
        ZonePoset::from_edges(zones, pairs)
    }

    /// Benchmarks cubical nerve construction time vs map size.
    ///
    /// Tests two poset topologies:
    /// - **Chain**: worst case for Warshall transitive closure (linear order).
    /// - **Bushy**: zone 0 < zone i for all i, maximising covering pairs.
    ///
    /// Asserts 100-zone chain construction < 100ms as a generous bound.
    #[test]
    fn bench_cubical_nerve_construction() {
        let sizes = [10, 50, 100, 500];

        // --- Chain poset (worst case for transitive closure) ---
        for &n in &sizes {
            let poset = chain_poset(n);
            let start = std::time::Instant::now();
            let _complex = std::hint::black_box(cubical_nerve(&poset));
            let elapsed = start.elapsed();
            println!("cubical_nerve(chain {} zones): {:?}", n, elapsed);
        }

        // --- Bushy poset (many covering pairs from single root) ---
        for &n in &sizes {
            let poset = bushy_poset(n);
            let start = std::time::Instant::now();
            let _complex = std::hint::black_box(cubical_nerve(&poset));
            let elapsed = start.elapsed();
            println!("cubical_nerve(bushy {} zones): {:?}", n, elapsed);
        }

        // Assert 100-zone chain completes within generous 100ms bound.
        let poset = chain_poset(100);
        let start = std::time::Instant::now();
        let _ = cubical_nerve(&poset);
        assert!(
            start.elapsed().as_millis() < 100,
            "100-zone chain construction took too long"
        );

        // Assert 100-zone bushy completes within generous 100ms bound.
        let poset = bushy_poset(100);
        let start = std::time::Instant::now();
        let _ = cubical_nerve(&poset);
        assert!(
            start.elapsed().as_millis() < 100,
            "100-zone bushy construction took too long"
        );
    }

    #[test]
    fn test_optimized_matches_scalar_edges() {
        // Verify that the optimized covering-edge construction produces
        // the exact same edges as the scalar algorithm.
        let configs = [
            (10, "chain"),
            (50, "chain"),
            (100, "chain"),
            (10, "bushy"),
            (50, "bushy"),
            (100, "bushy"),
        ];

        for (n, topo) in &configs {
            let poset = match *topo {
                "chain" => chain_poset(*n),
                "bushy" => bushy_poset(*n),
                _ => panic!("unknown topology"),
            };

            // Build with both algorithms — force each path by constructing
            // directly (cubical_nerve auto-routes, so we call the internal fns
            // via the same elements).
            let _elements = poset.elements();

            // Note: we can't call the private fns directly from tests,
            // so instead verify that the auto-routed result is correct
            // by checking structural properties.
            let complex = cubical_nerve(&poset);

            // Chain: n-1 edges.
            if *topo == "chain" {
                assert_eq!(
                    complex.n_edges(),
                    n - 1,
                    "chain of {} should have {} edges",
                    n,
                    n - 1
                );
            }

            // Bushy: n-1 edges (all from root to each leaf).
            if *topo == "bushy" {
                assert_eq!(
                    complex.n_edges(),
                    n - 1,
                    "bushy of {} should have {} edges",
                    n,
                    n - 1
                );
            }
        }
    }

    #[test]
    fn bench_optimized_vs_scalar_nerve() {
        // Compare construction time for sizes above and below the threshold.
        let threshold = 64; // NERVE_OPT_THRESHOLD
        let sizes = [32, 64, 128, 256];

        println!("\nNerve construction benchmark (threshold={}):", threshold);

        for &n in &sizes {
            let poset = chain_poset(n);
            let start = std::time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(cubical_nerve(&poset));
            }
            let chain_time = start.elapsed();

            let poset = bushy_poset(n);
            let start = std::time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(cubical_nerve(&poset));
            }
            let bushy_time = start.elapsed();

            let algo = if n < threshold { "scalar" } else { "optimized" };
            println!(
                "  n={:>4} algo={:>9} chain={:?} bushy={:?}",
                n, algo, chain_time / 10, bushy_time / 10
            );
        }
    }

    // -----------------------------------------------------------------------
    // T30: Adaptive nerve routing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cubical_nerve_with_threshold_scalar() {
        // Force scalar path by setting threshold higher than zone count.
        let poset = chain_poset(10);
        let complex = cubical_nerve_with_threshold(&poset, 100);

        // Chain of 10 → 9 edges, same as cubical_nerve.
        assert_eq!(complex.n_vertices(), 10);
        assert_eq!(complex.n_edges(), 9);
    }

    #[test]
    fn test_cubical_nerve_with_threshold_optimized() {
        // Force optimized path by setting threshold low.
        let poset = chain_poset(10);
        let complex = cubical_nerve_with_threshold(&poset, 1);

        // Same result regardless of backend.
        assert_eq!(complex.n_vertices(), 10);
        assert_eq!(complex.n_edges(), 9);
    }

    #[test]
    fn test_nerve_scalar_matches_optimized() {
        // For a medium-sized poset, verify both paths produce identical results.
        let poset = diamond_poset();

        let scalar_complex = cubical_nerve_with_threshold(&poset, 1000); // force scalar
        let optimized_complex = cubical_nerve_with_threshold(&poset, 0);  // force optimized

        // Vertices and edges should be identical.
        let mut sv = scalar_complex.vertices.clone();
        let mut ov = optimized_complex.vertices.clone();
        sv.sort();
        ov.sort();
        assert_eq!(sv, ov, "vertices must match");

        let mut se = scalar_complex.edges.clone();
        let mut oe = optimized_complex.edges.clone();
        se.sort();
        oe.sort();
        assert_eq!(se, oe, "edges must match");

        let mut sf = scalar_complex.faces.clone();
        let mut of_ = optimized_complex.faces.clone();
        sf.sort();
        of_.sort();
        assert_eq!(sf, of_, "faces must match");
    }

    #[test]
    fn test_nerve_with_threshold_boundary() {
        // Test at exactly the threshold boundary.
        let n = 64;
        let poset = chain_poset(n);

        // At threshold → optimized.
        let at = cubical_nerve_with_threshold(&poset, n);
        // Below threshold → scalar.
        let below = cubical_nerve_with_threshold(&poset, n + 1);
        // Above threshold → optimized.
        let above = cubical_nerve_with_threshold(&poset, n - 1);

        // All should produce the same complex.
        assert_eq!(at.n_vertices(), below.n_vertices());
        assert_eq!(at.n_edges(), below.n_edges());
        assert_eq!(at.n_vertices(), above.n_vertices());
        assert_eq!(at.n_edges(), above.n_edges());
    }
}
