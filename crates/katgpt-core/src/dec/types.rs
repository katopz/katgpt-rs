//! Cell complex and cochain types for Discrete Exterior Calculus (DEC).
//!
//! Based on "Topological Neural Operators" (arXiv:2606.09806).
//! A cell complex K has cells of rank 0 (vertices), 1 (edges), 2 (faces), 3 (volumes).
//! Cochains assign feature vectors to cells of a given rank.
//! Boundary matrices Bₖ encode oriented incidence between ranks.

// ---------------------------------------------------------------------------
// Cell Complex
// ---------------------------------------------------------------------------

/// Maximum supported rank (3 = vertices, edges, faces, volumes).
pub const MAX_RANK: u8 = 3;

/// A regular cell complex: vertices, edges, faces, and volumes with oriented incidence.
///
/// Cells are indexed per-rank: cell (rank=0, idx=5) is the 6th vertex.
/// Boundary matrices `B[k]` are sparse signed incidence matrices:
/// `B[k]` has shape `[n_{k-1} × n_k]`, encoding which (k-1)-cells bound each k-cell.
///
/// The fundamental identity `B[k] * B[k+1] = 0` holds by construction
/// (boundary of boundary is zero → curl(grad)=0, div(curl)=0).
pub struct CellComplex {
    /// Number of cells per rank: `n_cells[0]` = vertices, `n_cells[1]` = edges, etc.
    n_cells: [usize; (MAX_RANK as usize) + 1],
    /// Boundary matrices: `boundaries[k]` = B_{k+1} ∈ ℝ^{n_k × n_{k+1}}.
    /// Stored as sparse triplets: each entry is (row, col, sign).
    /// `boundaries[0]` = B₁ (vertex→edge incidence).
    /// `boundaries[1]` = B₂ (edge→face incidence).
    /// `boundaries[2]` = B₃ (face→volume incidence).
    /// `boundaries[3]` is absent (no rank-4 cells).
    boundaries: [Vec<(usize, usize, i8)>; MAX_RANK as usize],
    /// Monotonically increasing version counter. Bumped on any structural mutation
    /// (cell addition/removal). Consumers compare against their cached version to
    /// detect topology changes without polling individual cells.
    topology_version: u64,
}

impl CellComplex {
    /// Create a cell complex with the given cell counts (no incidence yet).
    ///
    /// Call `add_incidence` to populate boundary relations before use.
    #[inline]
    pub fn new(n_vertices: usize, n_edges: usize, n_faces: usize, n_volumes: usize) -> Self {
        Self {
            n_cells: [n_vertices, n_edges, n_faces, n_volumes],
            boundaries: [Vec::new(), Vec::new(), Vec::new()],
            topology_version: 0,
        }
    }

    /// Create a 2D regular cubical (grid) cell complex.
    ///
    /// Grid has `w` columns and `h` rows of vertices.
    /// - Vertices: w*h
    /// - Edges: horizontal (w-1)*h + vertical w*(h-1)
    /// - Faces: (w-1)*(h-1)
    /// - Volumes: 0
    pub fn grid_2d(w: usize, h: usize) -> Self {
        let n_vertices = w * h;
        let n_h_edges = (w - 1) * h;
        let n_v_edges = w * (h - 1);
        let n_edges = n_h_edges + n_v_edges;
        let n_faces = (w - 1) * (h - 1);

        let mut cx = Self::new(n_vertices, n_edges, n_faces, 0);

        // Pre-allocate boundary vectors to exact capacity — avoids re-allocations during push.
        cx.boundaries[0].reserve_exact(2 * n_edges);
        cx.boundaries[1].reserve_exact(4 * n_faces);

        // B₁: vertex→edge incidence
        // Horizontal edges: edge e = (y, x, H) connects vertex (y*w+x) → (y*w+x+1)
        for y in 0..h {
            for x in 0..(w - 1) {
                let e_idx = y * (w - 1) + x;
                let v_tail = y * w + x;
                let v_head = y * w + x + 1;
                cx.boundaries[0].push((v_tail, e_idx, -1));
                cx.boundaries[0].push((v_head, e_idx, 1));
            }
        }
        // Vertical edges: edge e = (y, x, V) connects vertex (y*w+x) → ((y+1)*w+x)
        for y in 0..(h - 1) {
            for x in 0..w {
                let e_idx = n_h_edges + y * w + x;
                let v_tail = y * w + x;
                let v_head = (y + 1) * w + x;
                cx.boundaries[0].push((v_tail, e_idx, -1));
                cx.boundaries[0].push((v_head, e_idx, 1));
            }
        }

        // B₂: edge→face incidence
        // Face f = (y, x) is bounded by 4 edges:
        //   bottom (horizontal, y, x), right (vertical, y, x+1),
        //   top (horizontal, y+1, x), left (vertical, y, x)
        for y in 0..(h - 1) {
            for x in 0..(w - 1) {
                let f_idx = y * (w - 1) + x;
                // Bottom horizontal edge
                let e_bottom = y * (w - 1) + x;
                // Right vertical edge
                let e_right = n_h_edges + y * w + (x + 1);
                // Top horizontal edge
                let e_top = (y + 1) * (w - 1) + x;
                // Left vertical edge
                let e_left = n_h_edges + y * w + x;

                // Oriented boundary of face [bottom → right → top → left]
                cx.boundaries[1].push((e_bottom, f_idx, 1));
                cx.boundaries[1].push((e_right, f_idx, 1));
                cx.boundaries[1].push((e_top, f_idx, -1));
                cx.boundaries[1].push((e_left, f_idx, -1));
            }
        }

        cx
    }

    /// Number of cells at a given rank.
    #[inline]
    pub fn n_cells(&self, rank: u8) -> usize {
        self.n_cells[rank as usize]
    }

    /// Boundary matrix entries for rank k→(k+1): `B_{k+1}` as sparse triplets.
    ///
    /// Returns `&[(row, col, sign)]` where:
    /// - `row` indexes (k)-cells
    /// - `col` indexes (k+1)-cells
    /// - `sign` is the orientation coefficient ∈ {-1, +1}
    #[inline]
    pub fn boundary_entries(&self, k: u8) -> &[(usize, usize, i8)] {
        assert!(
            k < MAX_RANK,
            "boundary_entries: k={k} exceeds MAX_RANK={MAX_RANK}"
        );
        &self.boundaries[k as usize]
    }

    /// Number of vertices.
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.n_cells[0]
    }

    /// Number of edges.
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.n_cells[1]
    }

    /// Number of faces.
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_cells[2]
    }

    /// Number of volumes.
    #[inline]
    pub fn n_volumes(&self) -> usize {
        self.n_cells[3]
    }

    // -----------------------------------------------------------------------
    // Dynamic Topology (Plan 261 Phase 0)
    // -----------------------------------------------------------------------

    /// Current topology version. Bumped on every structural mutation.
    #[inline]
    pub fn topology_version(&self) -> u64 {
        self.topology_version
    }

    /// Returns `true` if the topology has changed since `version`.
    /// Sub-nanosecent: single integer comparison.
    #[inline]
    pub fn is_dirty_since(&self, version: u64) -> bool {
        self.topology_version > version
    }

    /// Number of active cells at a given rank.
    ///
    /// Since removal uses swap-remove (no tombstones), `n_cells` already reflects
    /// the post-removal active count. This method exists for semantic clarity at
    /// call sites that care about "alive" cells.
    #[inline]
    pub fn n_active_cells(&self, rank: u8) -> usize {
        self.n_cells(rank)
    }

    /// Remove a face (rank-2 cell) from the complex using swap-remove.
    ///
    /// - Deletes all B₂ entries for this face (its boundary edges are detached).
    /// - If the removed face is not the last, rebinds the last face's B₂ entries
    ///   to the removed slot (swap-remove avoids O(n) reindexing).
    /// - Vertices and edges are NOT removed — they may be shared with other faces.
    ///   Dangling edges represent the boundary of the destroyed region.
    pub fn remove_face(&mut self, face_idx: usize) {
        assert!(
            face_idx < self.n_cells[2],
            "remove_face: face_idx {face_idx} >= n_faces {}",
            self.n_cells[2]
        );

        let last_face = self.n_cells[2] - 1;

        // Remove the face's column from B₂ (boundaries[1]).
        self.swap_remove_from_boundary(1, face_idx, last_face, true);

        self.n_cells[2] -= 1;
        self.topology_version += 1;
    }

    /// Remove a cell of any rank using swap-remove.
    ///
    /// For each boundary matrix where the cell appears:
    /// 1. Remove all entries referencing the cell.
    /// 2. Rebind the last cell at that rank to fill the gap.
    ///
    /// Rank-specific handling:
    /// - **Vertex (0)**: removed from B₁ as `row`
    /// - **Edge (1)**: removed from B₁ as `col` AND from B₂ as `row`
    /// - **Face (2)**: delegates to [`remove_face`](Self::remove_face)
    /// - **Volume (3)**: removed from B₃ as `col`
    pub fn remove_cell(&mut self, rank: u8, cell_idx: usize) {
        match rank {
            0 => {
                assert!(
                    cell_idx < self.n_cells[0],
                    "remove_cell: vertex_idx {cell_idx} >= n_vertices {}",
                    self.n_cells[0]
                );
                let last = self.n_cells[0] - 1;
                self.swap_remove_from_boundary(0, cell_idx, last, false);
                self.n_cells[0] -= 1;
                self.topology_version += 1;
            }
            1 => {
                assert!(
                    cell_idx < self.n_cells[1],
                    "remove_cell: edge_idx {cell_idx} >= n_edges {}",
                    self.n_cells[1]
                );
                let last = self.n_cells[1] - 1;
                // Edge appears as col in B₁ (boundaries[0]) and as row in B₂ (boundaries[1])
                self.swap_remove_from_boundary(0, cell_idx, last, true);
                self.swap_remove_from_boundary(1, cell_idx, last, false);
                self.n_cells[1] -= 1;
                self.topology_version += 1;
            }
            2 => {
                self.remove_face(cell_idx);
            }
            3 => {
                assert!(
                    cell_idx < self.n_cells[3],
                    "remove_cell: volume_idx {cell_idx} >= n_volumes {}",
                    self.n_cells[3]
                );
                let last = self.n_cells[3] - 1;
                self.swap_remove_from_boundary(2, cell_idx, last, true);
                self.n_cells[3] -= 1;
                self.topology_version += 1;
            }
            _ => panic!("remove_cell: rank {rank} exceeds MAX_RANK {MAX_RANK}"),
        }
    }

    /// Swap-remove a cell from a single boundary matrix.
    ///
    /// - `boundary_idx`: index into `self.boundaries` (0=B₁, 1=B₂, 2=B₃)
    /// - `target_idx`: cell index to remove
    /// - `last_idx`: last cell index at this rank (for swap-rebind)
    /// - `is_col`: `true` if the cell appears as `col` in entries, `false` if as `row`
    ///
    /// Single-pass compact-and-rebind: drops entries referencing `target_idx` and
    /// rebinds `last_idx` → `target_idx` in one sweep. This halves cache misses vs
    /// the previous two-pass `retain()` + linear scan, which matters because the
    /// B₂ boundary of a 64×64 grid is ~384KB (larger than typical L2).
    fn swap_remove_from_boundary(
        &mut self,
        boundary_idx: usize,
        target_idx: usize,
        last_idx: usize,
        is_col: bool,
    ) {
        let boundary = &mut self.boundaries[boundary_idx];
        let needs_rebind = target_idx != last_idx;
        let mut write = 0usize;

        for read in 0..boundary.len() {
            // Copy-by-value to avoid holding a borrow into `boundary` while we write.
            let (row, col, sign) = boundary[read];

            // Drop entries referencing the removed cell.
            let cell = if is_col { col } else { row };
            if cell == target_idx {
                continue;
            }

            // Rebind last cell's entries to the freed slot (swap-remove semantics).
            let (new_row, new_col) = if needs_rebind {
                let nr = if !is_col && row == last_idx { target_idx } else { row };
                let nc = if is_col && col == last_idx { target_idx } else { col };
                (nr, nc)
            } else {
                (row, col)
            };

            boundary[write] = (new_row, new_col, sign);
            write += 1;
        }

        boundary.truncate(write);
    }
}

// ---------------------------------------------------------------------------
// Cochain Field
// ---------------------------------------------------------------------------

/// A k-cochain: assigns a feature vector to each k-cell of a cell complex.
///
/// `u: Kₖ → ℝ^{dim}` — stored as a flat slice `[n_k × dim]`, row-major.
/// The rank `k` determines the geometric type:
/// - Rank 0 (vertices): potentials, pressures, HP, scalar values
/// - Rank 1 (edges): circulations, gradients, flow, threat direction
/// - Rank 2 (faces): fluxes, vorticity, area-normalized quantities
/// - Rank 3 (volumes): densities, mass, occupancy
pub struct CochainField {
    /// Flat feature data: `[n_k × dim]`, row-major.
    /// `data[cell_idx * dim + d]` is the d-th feature of cell `cell_idx`.
    pub data: Vec<f32>,
    /// Feature dimension per cell.
    pub dim: usize,
    /// Rank k of this cochain (0=vertex, 1=edge, 2=face, 3=volume).
    pub rank: u8,
}

impl CochainField {
    /// Create a zero-initialized cochain for the given rank and dimension.
    #[inline]
    pub fn zeros(rank: u8, n_cells: usize, dim: usize) -> Self {
        Self {
            data: vec![0.0f32; n_cells * dim],
            dim,
            rank,
        }
    }

    /// Create a cochain from existing data.
    #[inline]
    pub fn from_vec(rank: u8, dim: usize, data: Vec<f32>) -> Self {
        Self { data, dim, rank }
    }

    /// Number of cells in this cochain.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.data.len() / self.dim
    }

    /// Read the feature vector for cell `idx`. Returns a slice of length `dim`.
    #[inline]
    pub fn cell_features(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Mutable access to the feature vector for cell `idx`.
    #[inline]
    pub fn cell_features_mut(&mut self, idx: usize) -> &mut [f32] {
        let start = idx * self.dim;
        &mut self.data[start..start + self.dim]
    }

    /// Read scalar value for cell `idx` (dim must be 1).
    #[inline]
    pub fn scalar(&self, idx: usize) -> f32 {
        debug_assert_eq!(self.dim, 1, "scalar() requires dim=1, got {}", self.dim);
        self.data[idx]
    }

    /// Write scalar value for cell `idx` (dim must be 1).
    #[inline]
    pub fn set_scalar(&mut self, idx: usize, val: f32) {
        debug_assert_eq!(self.dim, 1, "set_scalar() requires dim=1, got {}", self.dim);
        self.data[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec::operators::exterior_derivative;

    #[test]
    fn grid_2d_cell_counts() {
        let cx = CellComplex::grid_2d(4, 3);
        assert_eq!(cx.n_vertices(), 12); // 4*3
        assert_eq!(cx.n_edges(), 17); // (4-1)*3 + 4*(3-1) = 9+8
        assert_eq!(cx.n_faces(), 6); // (4-1)*(3-1)
        assert_eq!(cx.n_volumes(), 0);
    }

    #[test]
    fn grid_2d_boundary_b1_entries() {
        let cx = CellComplex::grid_2d(3, 2);
        // B₁ should have 2 entries per edge (tail + head)
        let b1 = cx.boundary_entries(0);
        // n_edges = (3-1)*2 + 3*(2-1) = 4+3 = 7
        assert_eq!(b1.len(), 14); // 7 edges × 2 entries each
    }

    #[test]
    fn grid_2d_boundary_b2_entries() {
        let cx = CellComplex::grid_2d(3, 2);
        // B₂ should have 4 entries per face
        let b2 = cx.boundary_entries(1);
        // n_faces = (3-1)*(2-1) = 2
        assert_eq!(b2.len(), 8); // 2 faces × 4 entries each
    }

    #[test]
    fn cochain_zeros() {
        let cf = CochainField::zeros(0, 10, 3);
        assert_eq!(cf.n_cells(), 10);
        assert_eq!(cf.dim, 3);
        assert_eq!(cf.data.len(), 30);
        assert!(cf.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn cochain_scalar_roundtrip() {
        let mut cf = CochainField::zeros(0, 5, 1);
        cf.set_scalar(2, 42.0);
        assert_eq!(cf.scalar(2), 42.0);
    }

    #[test]
    fn cochain_features_slice() {
        let mut cf = CochainField::zeros(1, 3, 2);
        cf.cell_features_mut(1)[0] = 1.0;
        cf.cell_features_mut(1)[1] = 2.0;
        let f = cf.cell_features(1);
        assert_eq!(f, &[1.0, 2.0]);
    }

    // -----------------------------------------------------------------------
    // Plan 261 Phase 0: Dynamic Topology
    // -----------------------------------------------------------------------

    #[test]
    fn test_topology_version_starts_zero() {
        let cx = CellComplex::grid_2d(4, 4);
        assert_eq!(cx.topology_version(), 0);
        assert!(!cx.is_dirty_since(0));
    }

    #[test]
    fn test_remove_face_bumps_version() {
        let mut cx = CellComplex::grid_2d(4, 4);
        assert_eq!(cx.topology_version(), 0);
        cx.remove_face(0);
        assert_eq!(cx.topology_version(), 1);
        assert!(cx.is_dirty_since(0));
        assert!(!cx.is_dirty_since(1));
    }

    #[test]
    fn test_remove_face_decrements_count() {
        let mut cx = CellComplex::grid_2d(4, 4);
        let n_before = cx.n_faces();
        cx.remove_face(0);
        assert_eq!(cx.n_faces(), n_before - 1);
        assert_eq!(cx.n_active_cells(2), n_before - 1);
    }

    #[test]
    fn test_remove_face_swap_remove_rebinds() {
        // grid_2d(3, 2) has exactly 2 faces — perfect for testing swap-rebind
        let mut cx = CellComplex::grid_2d(3, 2);
        assert_eq!(cx.n_faces(), 2);

        // Capture face 1's boundary edges before removal
        let b2_before = cx.boundary_entries(1);
        let face1_edges_before: Vec<usize> = b2_before
            .iter()
            .filter(|&&(_, col, _)| col == 1)
            .map(|&(row, _, _)| row)
            .collect();
        assert!(!face1_edges_before.is_empty());

        // Remove face 0 — face 1 (the last) should be rebound to index 0
        cx.remove_face(0);
        assert_eq!(cx.n_faces(), 1);

        let b2_after = cx.boundary_entries(1);

        // All remaining B₂ entries must reference face 0 (the swapped slot)
        for &(_, col, _) in b2_after {
            assert_eq!(
                col, 0,
                "all B₂ entries should reference face 0 after swap-remove"
            );
        }

        // The rebound edges must match face 1's original boundary edges
        let edges_after: Vec<usize> = b2_after.iter().map(|&(row, _, _)| row).collect();
        let mut sorted_before = face1_edges_before.clone();
        sorted_before.sort_unstable();
        let mut sorted_after = edges_after.clone();
        sorted_after.sort_unstable();
        assert_eq!(
            sorted_before, sorted_after,
            "rebound edges should match face 1's original edges"
        );
    }

    #[test]
    fn test_is_dirty_since() {
        let mut cx = CellComplex::grid_2d(4, 4);

        // Version 0: not dirty since 0
        assert!(!cx.is_dirty_since(0));

        cx.remove_face(2);
        // Version 1: dirty since 0, not dirty since 1
        assert!(cx.is_dirty_since(0));
        assert!(!cx.is_dirty_since(1));

        cx.remove_face(0);
        // Version 2: dirty since 0 and 1, not dirty since 2
        assert!(cx.is_dirty_since(0));
        assert!(cx.is_dirty_since(1));
        assert!(!cx.is_dirty_since(2));
    }

    #[test]
    fn test_remove_cell_face_delegates() {
        // remove_cell(rank=2, idx) must produce same result as remove_face(idx)
        let mut cx_a = CellComplex::grid_2d(4, 4);
        let mut cx_b = CellComplex::grid_2d(4, 4);

        cx_a.remove_face(3);
        cx_b.remove_cell(2, 3);

        assert_eq!(cx_a.n_faces(), cx_b.n_faces());
        assert_eq!(cx_a.topology_version(), cx_b.topology_version());

        // Compare B₂ entries
        let b2_a = cx_a.boundary_entries(1);
        let b2_b = cx_b.boundary_entries(1);
        assert_eq!(b2_a.len(), b2_b.len());
        for i in 0..b2_a.len() {
            assert_eq!(
                b2_a[i], b2_b[i],
                "B₂ entry {i} differs after remove_cell vs remove_face"
            );
        }
    }

    #[test]
    fn test_operators_correct_after_removal() {
        // The fundamental identity d₁∘d₀ = 0 must hold after face removal.
        // Removing a face deletes a column of B₂ but cannot violate B₁·B₂ = 0
        // because remaining columns were already zero in the product.
        let mut cx = CellComplex::grid_2d(4, 4);
        cx.remove_face(0);
        cx.remove_face(2);

        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            potential.set_scalar(i, (i as f32 * 0.5).sin());
        }

        let grad = exterior_derivative(&cx, &potential);
        let curl = exterior_derivative(&cx, &grad);

        assert_eq!(curl.rank, 2);
        for i in 0..cx.n_faces() {
            assert!(
                curl.scalar(i).abs() < 1e-6,
                "curl(grad) should be 0 after removal at face {i}, got {}",
                curl.scalar(i)
            );
        }
    }
}
