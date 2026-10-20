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

/// CSR-style coboundary index for rank k→(k+1) (Plan 318).
///
/// For each (k+1)-cell `c`, its boundary (k)-cells with orientation signs are:
/// ```text
/// entries[offsets[c]..offsets[c+1]]
/// ```
/// This is the transpose of `B_{k+1}`, stored in CSR layout for O(1) per-cell
/// boundary lookup. Built lazily by [`CellComplex::build_coboundary_index`] and
/// invalidated on any topology mutation.
///
/// The index is the key data structure for `boundary_flux_mass_indexed`: it
/// turns an `O(|B_{k+1}|)` full-matrix scan into an `O(|region| ×
/// boundary_per_cell)` direct lookup. Build cost is `O(|B_{k+1}|)` once per
/// topology version.
#[derive(Clone, Debug)]
pub struct CoboundaryIndex {
    /// Row pointers: length `n_cells(k+1) + 1`. `offsets[c]..offsets[c+1]` is
    /// the slice of `entries` for (k+1)-cell `c`.
    offsets: Vec<u32>,
    /// Flat entry list: `(k_cell_idx, orientation_sign)` pairs.
    entries: Vec<(u32, i8)>,
}

impl CoboundaryIndex {
    /// Returns the boundary entries for (k+1)-cell `c` as `&[(k_cell_idx, sign)]`.
    #[inline]
    pub fn cell_boundary(&self, c: usize) -> &[(u32, i8)] {
        let lo = self.offsets[c] as usize;
        let hi = self.offsets[c + 1] as usize;
        &self.entries[lo..hi]
    }

    /// Number of (k+1)-cells covered by this index (= `offsets.len() - 1`).
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

/// Regular-grid dimensionality tag for a [`CellComplex`] produced by
/// [`grid_2d`](CellComplex::grid_2d) or [`grid_3d`](CellComplex::grid_3d).
///
/// Carries the grid dimensions and discriminates the stencil fast path used
/// by `graph_laplacian_into`: 2D grids take the 5-point stencil, 3D grids take
/// the 7-point stencil. Set at construction; cleared (`grid_dims = None`) on
/// any topology mutation by `invalidate_coboundary_cache` per the `merkle_root`
/// lesson — a grid with a removed cell is no longer regular.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridDims {
    /// 2D regular grid (produced by [`CellComplex::grid_2d`]).
    Dim2 {
        /// Number of vertex columns (x-axis).
        w: usize,
        /// Number of vertex rows (y-axis).
        h: usize,
    },
    /// 3D regular cubical grid (produced by [`CellComplex::grid_3d`]).
    ///
    /// Feature-gated behind `grid_3d`; the variant is always compiled (it is
    /// part of the enum) but only constructed under the feature.
    Dim3 {
        /// Number of vertex columns (x-axis).
        w: usize,
        /// Number of vertex rows (y-axis).
        h: usize,
        /// Number of vertex depth-slices (z-axis).
        d: usize,
    },
}

/// A regular cell complex: vertices, edges, faces, and volumes with oriented incidence.
///
/// Cells are indexed per-rank: cell (rank=0, idx=5) is the 6th vertex.
/// Boundary matrices `B[k]` are sparse signed incidence matrices:
/// `B[k]` has shape `[n_{k-1} × n_k]`, encoding which (k-1)-cells bound each k-cell.
///
/// The fundamental identity `B[k] * B[k+1] = 0` holds by construction
/// (boundary of boundary is zero → curl(grad)=0, div(curl)=0).
#[derive(Clone)]
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
    /// Coboundary index cache (Plan 318). `coboundaries[k]` is the transpose of
    /// `boundaries[k]` in CSR layout. `None` = not built or invalidated. Built
    /// lazily by `build_coboundary_index`; cleared on every topology mutation
    /// (all 5 paths in `remove_face` / `remove_cell` × 4 ranks) per the
    /// `merkle_root` lesson.
    coboundaries: [Option<CoboundaryIndex>; MAX_RANK as usize],
    /// Regular-grid dimensions if this complex was produced by
    /// [`grid_2d`](Self::grid_2d) / [`grid_3d`](Self::grid_3d) and has not been
    /// mutated since. Enables the cache-friendly stencil fast path in
    /// `graph_laplacian_into` (Plan 357 G5 fix for 2D; Plan 454 for 3D).
    /// `None` for non-grid complexes or after any topology mutation
    /// (remove_face/remove_cell) — a grid with a missing face is no longer a
    /// regular grid, so the stencil would be wrong at the gap.
    grid_dims: Option<GridDims>,
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
            coboundaries: [const { None }; MAX_RANK as usize],
            grid_dims: None,
        }
    }

    /// Create a 2D regular cubical (grid) cell complex.
    ///
    /// Grid has `w` columns and `h` rows of vertices.
    /// - Vertices: w*h
    /// - Edges: horizontal (w-1)*h + vertical w*(h-1)
    /// - Faces: (w-1)*(h-1)
    /// - Volumes: 0
    ///
    /// Panics on `w == 0` or `h == 0`: the edge/face counts and loop bounds
    /// compute `w-1`/`h-1`, and a zero dimension would wrap in a release
    /// build (silent infinite push → OOM; issue 955). There is no meaningful
    /// empty grid complex — no cells of any rank, no operators apply.
    pub fn grid_2d(w: usize, h: usize) -> Self {
        assert!(
            w > 0 && h > 0,
            "grid_2d needs a non-empty grid, got {w}x{h}"
        );
        let n_vertices = w * h;
        let n_h_edges = (w - 1) * h;
        let n_v_edges = w * (h - 1);
        let n_edges = n_h_edges + n_v_edges;
        let n_faces = (w - 1) * (h - 1);

        let mut cx = Self::new(n_vertices, n_edges, n_faces, 0);
        // Mark as a regular grid so graph_laplacian_into can take the 5-point-
        // stencil fast path (Plan 357 G5 latency fix). Cleared by any mutation.
        cx.grid_dims = Some(GridDims::Dim2 { w, h });

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

    /// Create a 3D regular cubical (grid) cell complex.
    ///
    /// Grid has `w` columns, `h` rows, and `d` depth-slices of vertices.
    /// - Vertices: `w * h * d`
    /// - Edges: x-aligned `(w-1)*h*d` + y-aligned `w*(h-1)*d` + z-aligned `w*h*(d-1)`
    /// - Faces: xy-planes `(w-1)*(h-1)*d` + xz-planes `(w-1)*h*(d-1)` + yz-planes `w*(h-1)*(d-1)`
    /// - Volumes: `(w-1)*(h-1)*(d-1)`
    ///
    /// Vertex layout is row-major with z-slowest (matches [`CochainField`] flat
    /// layout): `vidx(x, y, z) = (z * h + y) * w + x`. Edges and faces are
    /// grouped by orientation into contiguous index ranges so the 7-point
    /// stencil (T3) can dispatch by orientation.
    ///
    /// Boundary matrices use canonical cubical orientation:
    /// - B₁ (vertex→edge): tail = lower-index corner, head = higher-index corner
    ///   (matches [`grid_2d`](Self::grid_2d))
    /// - B₂ (edge→face): each face oriented counterclockwise viewed from its
    ///   positive normal (xy→+z, xz→+y, yz→+x via the right-hand rule)
    /// - B₃ (face→volume): each face signed by outward-normal convention
    ///   (negative face = −1, positive face = +1)
    ///
    /// The DEC identities `B₁·B₂ = 0` and `B₂·B₃ = 0` hold by construction —
    /// verified by `grid_3d_b1_b2_composition_zero` / `grid_3d_b2_b3_composition_zero`.
    ///
    /// # Panics
    /// If `w < 2 || h < 2 || d < 2` — a grid thinner than 2 along any axis has
    /// zero volumes (degenerate). Mirrors the implicit `grid_2d` contract where
    /// `w=1` produces zero faces.
    #[cfg(feature = "grid_3d")]
    pub fn grid_3d(w: usize, h: usize, d: usize) -> Self {
        assert!(w >= 2, "grid_3d: w={w} < 2 (degenerate, zero volumes)");
        assert!(h >= 2, "grid_3d: h={h} < 2 (degenerate, zero volumes)");
        assert!(d >= 2, "grid_3d: d={d} < 2 (degenerate, zero volumes)");

        // --- Cell counts (cubical grid topology) ---
        let n_x_edges = (w - 1) * h * d;
        let n_y_edges = w * (h - 1) * d;
        let n_z_edges = w * h * (d - 1);
        let n_edges = n_x_edges + n_y_edges + n_z_edges;
        let n_xy_faces = (w - 1) * (h - 1) * d;
        let n_xz_faces = (w - 1) * h * (d - 1);
        let n_yz_faces = w * (h - 1) * (d - 1);
        let n_faces = n_xy_faces + n_xz_faces + n_yz_faces;
        let n_volumes = (w - 1) * (h - 1) * (d - 1);

        let mut cx = Self::new(w * h * d, n_edges, n_faces, n_volumes);
        cx.grid_dims = Some(GridDims::Dim3 { w, h, d });

        // Pre-allocate boundary vectors to exact capacity — no re-allocs during push.
        cx.boundaries[0].reserve_exact(2 * n_edges);
        cx.boundaries[1].reserve_exact(4 * n_faces);
        cx.boundaries[2].reserve_exact(6 * n_volumes);

        // --- Indexing helpers (inlined by the compiler; capture grid dims) ---
        let vidx = |x: usize, y: usize, z: usize| (z * h + y) * w + x;
        let e_x = |x: usize, y: usize, z: usize| (z * h + y) * (w - 1) + x;
        let e_y = |x: usize, y: usize, z: usize| n_x_edges + (z * (h - 1) + y) * w + x;
        let e_z = |x: usize, y: usize, z: usize| n_x_edges + n_y_edges + z * (w * h) + y * w + x;
        let f_xy = |x: usize, y: usize, z: usize| (z * (h - 1) + y) * (w - 1) + x;
        let f_xz = |x: usize, y: usize, z: usize| n_xy_faces + (y * (d - 1) + z) * (w - 1) + x;
        let f_yz = |x: usize, y: usize, z: usize| {
            n_xy_faces + n_xz_faces + (x * (d - 1) + z) * (h - 1) + y
        };
        let vol = |x: usize, y: usize, z: usize| (z * (h - 1) + y) * (w - 1) + x;

        // --- B₁: vertex→edge incidence (2 entries per edge) ---
        // x-edges: (x,y,z) ↔ (x+1,y,z)
        for z in 0..d {
            for y in 0..h {
                for x in 0..(w - 1) {
                    let e = e_x(x, y, z);
                    cx.boundaries[0].push((vidx(x, y, z), e, -1));
                    cx.boundaries[0].push((vidx(x + 1, y, z), e, 1));
                }
            }
        }
        // y-edges: (x,y,z) ↔ (x,y+1,z)
        for z in 0..d {
            for y in 0..(h - 1) {
                for x in 0..w {
                    let e = e_y(x, y, z);
                    cx.boundaries[0].push((vidx(x, y, z), e, -1));
                    cx.boundaries[0].push((vidx(x, y + 1, z), e, 1));
                }
            }
        }
        // z-edges: (x,y,z) ↔ (x,y,z+1)
        for z in 0..(d - 1) {
            for y in 0..h {
                for x in 0..w {
                    let e = e_z(x, y, z);
                    cx.boundaries[0].push((vidx(x, y, z), e, -1));
                    cx.boundaries[0].push((vidx(x, y, z + 1), e, 1));
                }
            }
        }

        // --- B₂: edge→face incidence (4 entries per face) ---
        // xy-faces (normal = +z), CCW loop: bottom(+1) → right(+1) → top(−1) → left(−1)
        for z in 0..d {
            for y in 0..(h - 1) {
                for x in 0..(w - 1) {
                    let f = f_xy(x, y, z);
                    cx.boundaries[1].push((e_x(x, y, z), f, 1)); // bottom
                    cx.boundaries[1].push((e_y(x + 1, y, z), f, 1)); // right
                    cx.boundaries[1].push((e_x(x, y + 1, z), f, -1)); // top (reversed)
                    cx.boundaries[1].push((e_y(x, y, z), f, -1)); // left (reversed)
                }
            }
        }
        // xz-faces (normal = +y), CCW loop in (z,x): z-fwd → x-fwd → z-bwd → x-bwd
        for y in 0..h {
            for z in 0..(d - 1) {
                for x in 0..(w - 1) {
                    let f = f_xz(x, y, z);
                    cx.boundaries[1].push((e_z(x, y, z), f, 1)); // front z-edge
                    cx.boundaries[1].push((e_x(x, y, z + 1), f, 1)); // back x-edge
                    cx.boundaries[1].push((e_z(x + 1, y, z), f, -1)); // right z-edge (reversed)
                    cx.boundaries[1].push((e_x(x, y, z), f, -1)); // front x-edge (reversed)
                }
            }
        }
        // yz-faces (normal = +x), CCW loop in (y,z): y-fwd → z-fwd → y-bwd → z-bwd
        for x in 0..w {
            for z in 0..(d - 1) {
                for y in 0..(h - 1) {
                    let f = f_yz(x, y, z);
                    cx.boundaries[1].push((e_y(x, y, z), f, 1)); // bottom y-edge
                    cx.boundaries[1].push((e_z(x, y + 1, z), f, 1)); // top z-edge
                    cx.boundaries[1].push((e_y(x, y, z + 1), f, -1)); // back y-edge (reversed)
                    cx.boundaries[1].push((e_z(x, y, z), f, -1)); // front z-edge (reversed)
                }
            }
        }

        // --- B₃: face→volume incidence (6 entries per volume) ---
        // Outward-normal signs: negative face = −1, positive face = +1.
        for z in 0..(d - 1) {
            for y in 0..(h - 1) {
                for x in 0..(w - 1) {
                    let v = vol(x, y, z);
                    cx.boundaries[2].push((f_xy(x, y, z), v, -1)); // bottom (z−)
                    cx.boundaries[2].push((f_xy(x, y, z + 1), v, 1)); // top (z+)
                    cx.boundaries[2].push((f_xz(x, y, z), v, -1)); // front (y−)
                    cx.boundaries[2].push((f_xz(x, y + 1, z), v, 1)); // back (y+)
                    cx.boundaries[2].push((f_yz(x, y, z), v, -1)); // left (x−)
                    cx.boundaries[2].push((f_yz(x + 1, y, z), v, 1)); // right (x+)
                }
            }
        }

        cx
    }

    /// Number of cells at a given rank.
    #[inline]
    pub fn n_cells(&self, rank: u8) -> usize {
        self.n_cells[rank as usize]
    }

    /// Build a 1-complex (vertices + edges only) from an edge list (Plan 370 T4.1).
    ///
    /// Each edge connects vertex `tail` → vertex `head` with orientation +1 on
    /// the head and −1 on the tail (the standard DEC convention from `grid_2d`).
    /// No faces or volumes. This fills the API gap left by the `add_incidence`
    /// method referenced in [`new`](Self::new)'s doc — arbitrary graphs (trees,
    /// DAGs, etc.) can now be constructed as a `CellComplex`.
    ///
    /// # Arguments
    /// * `n_vertices` — number of vertices (indexed `0..n_vertices`).
    /// * `edges` — `&[(tail, head)]` pairs. Edge `i` connects `tail[i]` → `head[i]`.
    ///
    /// # Panics
    /// If any vertex index in `edges` is `>= n_vertices`.
    pub fn from_edges(n_vertices: usize, edges: &[(usize, usize)]) -> Self {
        let n_edges = edges.len();
        let mut cx = Self::new(n_vertices, n_edges, 0, 0);
        cx.boundaries[0].reserve_exact(2 * n_edges);
        for (i, &(tail, head)) in edges.iter().enumerate() {
            assert!(
                tail < n_vertices,
                "from_edges: tail {tail} >= n_vertices {n_vertices} (edge {i})"
            );
            assert!(
                head < n_vertices,
                "from_edges: head {head} >= n_vertices {n_vertices} (edge {i})"
            );
            cx.boundaries[0].push((tail, i, -1));
            cx.boundaries[0].push((head, i, 1));
        }
        cx
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

    /// Build (or rebuild) the coboundary index for rank k→(k+1) (Plan 318).
    ///
    /// The coboundary index is the CSR-layout transpose of `boundaries[k]`:
    /// for each (k+1)-cell, a contiguous slice of its boundary (k)-cells with
    /// orientation signs. Enables `O(|region| × boundary_per_cell)` queries via
    /// [`coboundary_entries`](Self::coboundary_entries) instead of the
    /// `O(|B_{k+1}|)` full-matrix scan done by `boundary_entries`.
    ///
    /// # Complexity
    /// `O(|B_{k+1}|)` — a single count-then-scatter pass over the existing
    /// boundary triplets. Called once per topology version; the result is
    /// cached and invalidated automatically on any structural mutation.
    ///
    /// # Panics
    /// If `k >= MAX_RANK` (there is no `B_{k+1}` at rank `MAX_RANK`).
    pub fn build_coboundary_index(&mut self, k: u8) {
        assert!(
            k < MAX_RANK,
            "build_coboundary_index: k={k} exceeds MAX_RANK={MAX_RANK}"
        );
        let ki = k as usize;
        let n_kp1 = self.n_cells[ki + 1];

        // Pass 1: count entries per (k+1)-cell (column histogram of B_{k+1}).
        let mut offsets = vec![0u32; n_kp1 + 1];
        for &(_row, col, _sign) in &self.boundaries[ki] {
            offsets[col + 1] += 1;
        }
        // Prefix-sum → CSR row pointers (here: column pointers).
        for i in 1..=n_kp1 {
            offsets[i] += offsets[i - 1];
        }

        // Pass 2: scatter entries into their CSR slots.
        let n_entries = self.boundaries[ki].len();
        let mut entries = Vec::with_capacity(n_entries);
        entries.resize(n_entries, (0, 0));
        let mut cursor = offsets.clone(); // mutable write cursors per (k+1)-cell
        for &(row, col, sign) in &self.boundaries[ki] {
            let slot = cursor[col] as usize;
            entries[slot] = (row as u32, sign);
            cursor[col] += 1;
        }
        // cursor is now discarded; `offsets` holds the final CSR layout.

        self.coboundaries[ki] = Some(CoboundaryIndex { offsets, entries });
    }

    /// Access the pre-built coboundary index for rank k→(k+1) (Plan 318).
    ///
    /// Returns `None` if [`build_coboundary_index`](Self::build_coboundary_index)
    /// has not been called for rank `k` since the most recent topology mutation,
    /// or `k >= MAX_RANK`.
    ///
    /// Callers that need the fast path should call `build_coboundary_index`
    /// once after the topology stabilizes, then query via this accessor.
    #[inline]
    pub fn coboundary_entries(&self, k: u8) -> Option<&CoboundaryIndex> {
        if k >= MAX_RANK {
            return None;
        }
        self.coboundaries[k as usize].as_ref()
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

    /// Regular-grid dimensions `(w, h)` if this complex is an unmutated
    /// [`grid_2d`](Self::grid_2d) product. `None` for arbitrary complexes,
    /// 3D grids (use [`grid_dims_3d`](Self::grid_dims_3d)), or after any
    /// topology mutation. The graph Laplacian uses this to take a
    /// cache-friendly 5-point-stencil fast path when available (Plan 357 G5).
    ///
    /// Returns `None` for 3D grids to preserve the 2D-only contract of this
    /// accessor — callers that need to dispatch on dimensionality should use
    /// [`grid_dims_full`](Self::grid_dims_full) instead.
    #[inline]
    pub fn grid_dims(&self) -> Option<(usize, usize)> {
        match self.grid_dims {
            Some(GridDims::Dim2 { w, h }) => Some((w, h)),
            _ => None,
        }
    }

    /// Regular-grid dimensions `(w, h, d)` if this complex is an unmutated
    /// [`grid_3d`](Self::grid_3d) product. `None` for arbitrary complexes, 2D
    /// grids, or after any topology mutation.
    #[inline]
    pub fn grid_dims_3d(&self) -> Option<(usize, usize, usize)> {
        match self.grid_dims {
            Some(GridDims::Dim3 { w, h, d }) => Some((w, h, d)),
            _ => None,
        }
    }

    /// Full discriminated regular-grid dimensions, if this complex is an
    /// unmutated `grid_2d` or `grid_3d` product. `None` for arbitrary complexes
    /// or after any topology mutation.
    ///
    /// This is the accessor new code should use when it needs to dispatch on
    /// grid dimensionality. The single-feature-gated consumers (`grid_3d`
    /// stencil dispatch) match on the [`GridDims::Dim3`] variant; legacy 2D
    /// consumers keep using [`grid_dims`](Self::grid_dims).
    #[inline]
    pub fn grid_dims_full(&self) -> Option<GridDims> {
        self.grid_dims
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
        self.invalidate_coboundary_cache();
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
                self.invalidate_coboundary_cache();
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
                self.invalidate_coboundary_cache();
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
                self.invalidate_coboundary_cache();
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
                let nr = if !is_col && row == last_idx {
                    target_idx
                } else {
                    row
                };
                let nc = if is_col && col == last_idx {
                    target_idx
                } else {
                    col
                };
                (nr, nc)
            } else {
                (row, col)
            };

            boundary[write] = (new_row, new_col, sign);
            write += 1;
        }

        boundary.truncate(write);
    }

    /// Invalidate all cached coboundary indices (Plan 318).
    ///
    /// Called from every topology-mutation path (`remove_face`, `remove_cell`
    /// ranks 0/1/3; rank 2 delegates to `remove_face`). A mutation at any rank
    /// can perturb multiple boundary matrices (e.g. removing an edge touches both
    /// B₁ and B₂), so we invalidate all three CSR caches conservatively. This
    /// follows the `merkle_root` lesson: audit ALL mutation paths, not just the
    /// obvious one.
    #[inline]
    fn invalidate_coboundary_cache(&mut self) {
        self.coboundaries = [const { None }; MAX_RANK as usize];
        // A mutation breaks the regular-grid invariant (a grid with a removed
        // face/cell is no longer a regular grid), so the 5-point-stencil fast
        // path would be wrong at the gap. Following the `merkle_root` lesson:
        // every mutation path invalidates every topology-derived invariant.
        self.grid_dims = None;
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
#[derive(Clone, Debug)]
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
    use crate::operators::exterior_derivative;

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
    fn grid_2d_minimal_grid_still_works() {
        // 1x1: both subtractions give 0 — no edges, no faces, 1 vertex.
        // Must keep working (issue 955: only the ZERO dims are the trap).
        let cx = CellComplex::grid_2d(1, 1);
        assert_eq!(cx.n_vertices(), 1);
        assert_eq!(cx.n_edges(), 0);
        assert_eq!(cx.n_faces(), 0);
    }

    #[test]
    #[should_panic(expected = "grid_2d needs a non-empty grid")]
    fn grid_2d_zero_width_panics() {
        // Was: (w-1) wraps to usize::MAX in release, the B₂ outer loop pushes
        // forever → OOM kill (issue 955).
        let _ = CellComplex::grid_2d(0, 5);
    }

    #[test]
    #[should_panic(expected = "grid_2d needs a non-empty grid")]
    fn grid_2d_zero_height_panics() {
        let _ = CellComplex::grid_2d(5, 0);
    }

    #[test]
    #[should_panic(expected = "grid_2d needs a non-empty grid")]
    fn grid_2d_both_zero_panics() {
        let _ = CellComplex::grid_2d(0, 0);
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
    // Plan 370 T4.1: CellComplex::from_edges (tree-shaped 1-complexes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_edges_basic_counts() {
        // A simple path: 0→1→2→3 (4 vertices, 3 edges).
        let cx = CellComplex::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(cx.n_vertices(), 4);
        assert_eq!(cx.n_edges(), 3);
        assert_eq!(cx.n_faces(), 0);
        assert_eq!(cx.n_volumes(), 0);
    }

    #[test]
    fn from_edges_boundary_entries() {
        // A star: vertex 0 is the hub, edges to 1, 2, 3.
        let cx = CellComplex::from_edges(4, &[(0, 1), (0, 2), (0, 3)]);
        // B₁: each edge has 2 entries (tail = -1, head = +1).
        let b1 = cx.boundary_entries(0);
        assert_eq!(b1.len(), 6); // 3 edges × 2 entries
        // Edge 0: (0, 1) → entries (0, 0, -1) and (1, 0, +1)
        // Edge 1: (0, 2) → entries (0, 1, -1) and (2, 1, +1)
        // Edge 2: (0, 3) → entries (0, 2, -1) and (3, 2, +1)
        let entries: Vec<(usize, usize, i8)> = b1.to_vec();
        assert!(entries.contains(&(0, 0, -1)), "tail of edge 0");
        assert!(entries.contains(&(1, 0, 1)), "head of edge 0");
        assert!(entries.contains(&(0, 1, -1)), "tail of edge 1");
        assert!(entries.contains(&(2, 1, 1)), "head of edge 1");
    }

    #[test]
    fn from_edges_exterior_derivative_matches_manual() {
        // A tree: 0→1, 0→2 (2 edges from root 0).
        // Vertex cochain: f(0)=10, f(1)=20, f(2)=30.
        // d₀ (exterior_derivative) maps rank-0 → rank-1: df(edge) = f(head) - f(tail).
        // Edge 0 (0→1): df = 20 - 10 = 10.
        // Edge 1 (0→2): df = 30 - 10 = 20.
        let cx = CellComplex::from_edges(3, &[(0, 1), (0, 2)]);
        let mut f = CochainField::zeros(0, 3, 1);
        f.set_scalar(0, 10.0);
        f.set_scalar(1, 20.0);
        f.set_scalar(2, 30.0);
        let df = exterior_derivative(&cx, &f);
        assert_eq!(df.rank, 1);
        assert_eq!(df.n_cells(), 2);
        assert!(
            (df.scalar(0) - 10.0).abs() < 1e-5,
            "edge 0: {}",
            df.scalar(0)
        );
        assert!(
            (df.scalar(1) - 20.0).abs() < 1e-5,
            "edge 1: {}",
            df.scalar(1)
        );
    }

    #[test]
    #[should_panic(expected = "tail 5 >= n_vertices 3")]
    fn from_edges_panics_on_out_of_bounds_tail() {
        CellComplex::from_edges(3, &[(5, 1)]);
    }

    #[test]
    #[should_panic(expected = "head 99 >= n_vertices 3")]
    fn from_edges_panics_on_out_of_bounds_head() {
        CellComplex::from_edges(3, &[(0, 99)]);
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

    // -----------------------------------------------------------------------
    // Plan 318: Coboundary Index
    // -----------------------------------------------------------------------

    #[test]
    fn test_coboundary_index_not_built_returns_none() {
        let cx = CellComplex::grid_2d(4, 4);
        assert!(cx.coboundary_entries(0).is_none());
        assert!(cx.coboundary_entries(1).is_none());
        assert!(cx.coboundary_entries(2).is_none());
        assert!(cx.coboundary_entries(3).is_none(), "k >= MAX_RANK -> None");
    }

    #[test]
    fn test_coboundary_index_b2_correct_csr() {
        // grid_2d(3, 2): 2 faces, each bounded by 4 edges.
        // B₂ has 8 entries (4 per face). Coboundary index for k=1 should have:
        //   offsets = [0, 4, 8]   (each face has 4 boundary edges)
        //   entries per face match the B₂ column for that face.
        let mut cx = CellComplex::grid_2d(3, 2);
        cx.build_coboundary_index(1);
        let idx = cx.coboundary_entries(1).expect("index built for k=1");

        assert_eq!(idx.n_cells(), cx.n_faces(), "n_cells should = n_faces");

        // Cross-check: for each face, the coboundary entry set must equal the
        // B₂ column for that face (same edges, same signs).
        let b2 = cx.boundary_entries(1);
        for face in 0..cx.n_faces() {
            let cob = idx.cell_boundary(face);
            let expected: Vec<(u32, i8)> = b2
                .iter()
                .filter(|&&(_, col, _)| col == face)
                .map(|&(row, _, sign)| (row as u32, sign))
                .collect();
            let got: Vec<(u32, i8)> = cob.to_vec();
            // CSR doesn't guarantee ordering; sort both for comparison.
            let mut got_sorted = got.clone();
            got_sorted.sort_unstable_by_key(|&(e, _)| e);
            let mut exp_sorted = expected.clone();
            exp_sorted.sort_unstable_by_key(|&(e, _)| e);
            assert_eq!(
                got_sorted, exp_sorted,
                "face {face} coboundary mismatch: got {got:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn test_coboundary_index_b1_correct_csr() {
        // grid_2d(3, 2): 7 edges, each bounded by 2 vertices (tail, head).
        let mut cx = CellComplex::grid_2d(3, 2);
        cx.build_coboundary_index(0);
        let idx = cx.coboundary_entries(0).expect("index built for k=0");

        assert_eq!(idx.n_cells(), cx.n_edges());

        // Every edge should have exactly 2 coboundary vertices.
        for edge in 0..cx.n_edges() {
            let cob = idx.cell_boundary(edge);
            assert_eq!(cob.len(), 2, "edge {edge} should have 2 boundary vertices");
            // Signs should be {-1, +1} (one tail, one head).
            let signs: Vec<i8> = cob.iter().map(|&(_, s)| s).collect();
            assert!(
                signs.contains(&-1) && signs.contains(&1),
                "edge {edge} signs should be {{-1, +1}}, got {signs:?}"
            );
        }
    }

    #[test]
    fn test_coboundary_index_remove_face_invalidates() {
        // The `merkle_root` lesson: every mutation path must invalidate the cache.
        let mut cx = CellComplex::grid_2d(4, 4);
        cx.build_coboundary_index(1);
        assert!(
            cx.coboundary_entries(1).is_some(),
            "index built before mutation"
        );

        cx.remove_face(0);
        assert!(
            cx.coboundary_entries(1).is_none(),
            "remove_face must invalidate the coboundary cache"
        );
    }

    #[test]
    fn test_coboundary_index_remove_cell_invalidates_all_ranks() {
        // Audit ALL mutation paths (ranks 0/1/2/3) — per the `merkle_root` lesson.
        for rank in 0..=3u8 {
            let mut cx = CellComplex::grid_2d(4, 4);
            // Build all valid coboundary indices (k=0,1,2).
            for k in 0..3u8 {
                cx.build_coboundary_index(k);
            }
            for k in 0..3u8 {
                assert!(cx.coboundary_entries(k).is_some(), "pre-mutation k={k}");
            }

            // remove_cell at the given rank bumps topology_version and should
            // invalidate ALL coboundary caches (conservative invalidation).
            // For rank 2, we remove a face (valid). For others, remove cell 0.
            // Volumes (rank 3) don't exist in grid_2d, so guard that case.
            match rank {
                0 => cx.remove_cell(0, 0),
                1 => cx.remove_cell(1, 0),
                2 => cx.remove_cell(2, 0),
                3 => continue, // grid_2d has no volumes; skip.
                _ => unreachable!(),
            }
            for k in 0..3u8 {
                assert!(
                    cx.coboundary_entries(k).is_none(),
                    "remove_cell rank {rank} must invalidate coboundary k={k}"
                );
            }
        }
    }

    #[test]
    fn test_coboundary_index_rebuild_after_mutation() {
        // After invalidation, build_coboundary_index must produce a correct
        // index reflecting the post-mutation topology.
        let mut cx = CellComplex::grid_2d(4, 4);
        cx.remove_face(0); // 15 faces remain
        cx.build_coboundary_index(1);
        let idx = cx.coboundary_entries(1).expect("rebuilt after mutation");

        assert_eq!(
            idx.n_cells(),
            cx.n_faces(),
            "rebuilt index reflects new n_faces"
        );

        // Cross-check against the (also-mutated) B₂.
        let b2 = cx.boundary_entries(1);
        for face in 0..cx.n_faces() {
            let cob = idx.cell_boundary(face);
            let expected_count = b2.iter().filter(|&&(_, col, _)| col == face).count();
            assert_eq!(
                cob.len(),
                expected_count,
                "face {face} coboundary count mismatch after rebuild"
            );
        }
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_RANK")]
    fn test_coboundary_index_build_panics_at_max_rank() {
        let mut cx = CellComplex::grid_2d(4, 4);
        cx.build_coboundary_index(MAX_RANK); // k=3 has no B₄
    }

    // -----------------------------------------------------------------------
    // Plan 454 T2: CellComplex::grid_3d (3D cubical grid)
    // -----------------------------------------------------------------------

    /// Build a dense i8 matrix from sparse triplets (row, col, sign).
    #[cfg(feature = "grid_3d")]
    fn dense_from_triplets(
        n_rows: usize,
        n_cols: usize,
        triplets: &[(usize, usize, i8)],
    ) -> Vec<Vec<i8>> {
        let mut m = vec![vec![0i8; n_cols]; n_rows];
        for &(r, c, s) in triplets {
            m[r][c] += s;
        }
        m
    }

    /// Multiply two dense i8 matrices → dense i32 result.
    #[cfg(feature = "grid_3d")]
    fn matmul_i8(a: &[Vec<i8>], b: &[Vec<i8>]) -> Vec<Vec<i32>> {
        let n_rows = a.len();
        let n_inner = if n_rows > 0 { a[0].len() } else { 0 };
        let n_cols = if n_inner > 0 { b[0].len() } else { 0 };
        let mut c = vec![vec![0i32; n_cols]; n_rows];
        for i in 0..n_rows {
            for k in 0..n_inner {
                if a[i][k] != 0 {
                    for j in 0..n_cols {
                        c[i][j] += a[i][k] as i32 * b[k][j] as i32;
                    }
                }
            }
        }
        c
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn grid_3d_cell_counts() {
        // (3, 3, 3): Euler χ = V − E + F − C = 27 − 54 + 36 − 8 = 1 (3-ball)
        let cx = CellComplex::grid_3d(3, 3, 3);
        assert_eq!(cx.n_vertices(), 27);
        assert_eq!(cx.n_edges(), 54); // 18 + 18 + 18
        assert_eq!(cx.n_faces(), 36); // 12 + 12 + 12
        assert_eq!(cx.n_volumes(), 8); // 2*2*2

        // (4, 3, 2): χ = 24 − 46 + 29 − 6 = 1
        let cx = CellComplex::grid_3d(4, 3, 2);
        assert_eq!(cx.n_vertices(), 24);
        assert_eq!(cx.n_edges(), 46); // 18 + 16 + 12
        assert_eq!(cx.n_faces(), 29); // 12 + 9 + 8
        assert_eq!(cx.n_volumes(), 6); // 3*2*1
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn grid_3d_boundary_entry_counts() {
        let cx = CellComplex::grid_3d(3, 3, 3);
        // B₁: 2 entries per edge
        assert_eq!(cx.boundary_entries(0).len(), 2 * 54);
        // B₂: 4 entries per face
        assert_eq!(cx.boundary_entries(1).len(), 4 * 36);
        // B₃: 6 entries per volume (this is the new rank-3 matrix grid_2d leaves empty)
        assert_eq!(cx.boundary_entries(2).len(), 6 * 8);
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn grid_3d_b3_is_populated() {
        // grid_2d leaves B₃ empty (zero volumes); grid_3d must populate it.
        let cx_2d = CellComplex::grid_2d(4, 4);
        assert!(
            cx_2d.boundary_entries(2).is_empty(),
            "grid_2d B₃ must be empty (sanity check)"
        );

        let cx_3d = CellComplex::grid_3d(3, 3, 3);
        assert!(
            !cx_3d.boundary_entries(2).is_empty(),
            "grid_3d B₃ must be populated (the rank-3 boundary matrix)"
        );
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn grid_3d_vertex_degree_7_point_stencil() {
        // On a 3D grid, each vertex's degree matches the 7-point stencil neighbor
        // count: corner = 3, edge = 4, face = 5, interior = 6.
        let cx = CellComplex::grid_3d(3, 3, 3);
        let b1 = cx.boundary_entries(0);

        // Count how many times each vertex appears in B₁ (as tail or head).
        let mut degree = vec![0usize; cx.n_vertices()];
        for &(v, _edge, _sign) in b1 {
            degree[v] += 1;
        }

        // vidx(x,y,z) = (z*3+y)*3 + x for a 3×3×3 grid
        let vidx = |x: usize, y: usize, z: usize| (z * 3 + y) * 3 + x;

        // Corner (0,0,0): 3 neighbors (+x, +y, +z)
        assert_eq!(degree[vidx(0, 0, 0)], 3, "corner vertex degree");
        // Edge (1,0,0): 4 neighbors (±x, +y, +z)
        assert_eq!(degree[vidx(1, 0, 0)], 4, "edge vertex degree");
        // Face (1,1,0): 5 neighbors (±x, ±y, +z)
        assert_eq!(degree[vidx(1, 1, 0)], 5, "face vertex degree");
        // Interior (1,1,1): 6 neighbors (±x, ±y, ±z)
        assert_eq!(degree[vidx(1, 1, 1)], 6, "interior vertex degree");
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn grid_3d_b1_b2_composition_zero() {
        // DEC identity: B₁ · B₂ = 0 (boundary of boundary is zero — d∘d = 0).
        let cx = CellComplex::grid_3d(3, 3, 3);
        let b1 = dense_from_triplets(cx.n_vertices(), cx.n_edges(), cx.boundary_entries(0));
        let b2 = dense_from_triplets(cx.n_edges(), cx.n_faces(), cx.boundary_entries(1));
        let prod = matmul_i8(&b1, &b2);
        for (i, row) in prod.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert_eq!(
                    val, 0,
                    "B₁·B₂ != 0 at ({i},{j}) — orientation is inconsistent"
                );
            }
        }
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn grid_3d_b2_b3_composition_zero() {
        // DEC identity: B₂ · B₃ = 0 (boundary of boundary is zero — d∘d = 0).
        let cx = CellComplex::grid_3d(3, 3, 3);
        let b2 = dense_from_triplets(cx.n_edges(), cx.n_faces(), cx.boundary_entries(1));
        let b3 = dense_from_triplets(cx.n_faces(), cx.n_volumes(), cx.boundary_entries(2));
        let prod = matmul_i8(&b2, &b3);
        for (i, row) in prod.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert_eq!(
                    val, 0,
                    "B₂·B₃ != 0 at ({i},{j}) — orientation is inconsistent"
                );
            }
        }
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn grid_3d_grid_dims_accessors() {
        let cx = CellComplex::grid_3d(4, 3, 2);
        // grid_dims_3d returns the 3D dims
        assert_eq!(cx.grid_dims_3d(), Some((4, 3, 2)));
        // grid_dims (2D accessor) returns None for 3D grids (back-compat contract)
        assert_eq!(cx.grid_dims(), None);
        // grid_dims_full returns the discriminated enum
        assert_eq!(
            cx.grid_dims_full(),
            Some(GridDims::Dim3 { w: 4, h: 3, d: 2 })
        );
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    #[should_panic(expected = "degenerate, zero volumes")]
    fn grid_3d_panics_on_w_too_small() {
        let _ = CellComplex::grid_3d(1, 3, 3);
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    #[should_panic(expected = "degenerate, zero volumes")]
    fn grid_3d_panics_on_h_too_small() {
        let _ = CellComplex::grid_3d(3, 1, 3);
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    #[should_panic(expected = "degenerate, zero volumes")]
    fn grid_3d_panics_on_d_too_small() {
        let _ = CellComplex::grid_3d(3, 3, 1);
    }
}
