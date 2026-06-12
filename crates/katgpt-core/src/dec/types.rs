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
    /// Feature dimension per cell.
    pub dim: usize,
    /// Rank k of this cochain (0=vertex, 1=edge, 2=face, 3=volume).
    pub rank: u8,
    /// Flat feature data: `[n_k × dim]`, row-major.
    /// `data[cell_idx * dim + d]` is the d-th feature of cell `cell_idx`.
    pub data: Vec<f32>,
}

impl CochainField {
    /// Create a zero-initialized cochain for the given rank and dimension.
    #[inline]
    pub fn zeros(rank: u8, n_cells: usize, dim: usize) -> Self {
        Self {
            dim,
            rank,
            data: vec![0.0f32; n_cells * dim],
        }
    }

    /// Create a cochain from existing data.
    #[inline]
    pub fn from_vec(rank: u8, dim: usize, data: Vec<f32>) -> Self {
        Self { dim, rank, data }
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
}
