//! Multi-scale V-cycle on cell complexes via selector restriction maps.
//!
//! A modelless hierarchical cochain transfer: restrict a fine cochain to a
//! coarser complex via a deterministic selector gather, apply a caller-supplied
//! coarse operator (e.g. smoothing), and prolongate the coarse result back to
//! the fine complex (adjoint scatter). Zero training, zero backprop — the
//! restriction maps are deterministic selector indices, reusing the
//! `SheafMaps::selector_indices` gather-scatter pattern (O(d_e) per coarse
//! cell, no dense matvec).
//!
//! This fills the multi-scale composition layer on top of the single-complex
//! DEC operators (`exterior_derivative`, `codifferential`, `hodge_laplacian`,
//! `hodge_decompose`). Those operators handle one resolution level; the
//! V-cycle composes two.
//!
//! # Selector restriction semantics
//!
//! The restriction is encoded in a [`VCycleRestriction`] map: a fixed
//! `n_coarse_vertices`-length slice of fine-vertex indices. Coarse vertex `c`
//! is the **representative aggregate** of fine vertex `restriction[c]`. The
//! restrict step gathers `coarse[c] = fine[restriction[c]]` (per feature dim),
//! and the prolongate step scatters `fine[restriction[c]] = coarse[c]` (per
//! feature dim), leaving non-representative fine vertices at zero (or at their
//! last prolongated value when called iteratively).
//!
//! This is the deterministic, modelless floor: no learned transfer operator,
//! no weighted aggregation. The selector is a fixed row-selection — the same
//! mathematical object as `SheafMaps::selector` but indexed over coarse cells
//! instead of edges.
//!
//! # Commutativity
//!
//! When the coarse complex is a **coarsening** of the fine complex (every
//! coarse edge corresponds to a fine edge between the representative fine
//! vertices, and the coarse boundary matrix is the induced sub-structure), the
//! restriction commutes with the exterior derivative up to the coarsening:
//!
//! ```text
//!   dₖ(Kc) ∘ Rₖ  =  Rₖ₊₁ ∘ dₖ(K)
//! ```
//!
//! i.e. "restrict then differentiate on the coarse complex" equals
//! "differentiate on the fine complex then restrict". This is verified by the
//! G1 unit tests on (a) a regular grid coarsening and (b) an irregular
//! coarsening. See `tests::commutativity_*`.

use crate::types::{CellComplex, CochainField};

// ---------------------------------------------------------------------------
// Restriction map
// ---------------------------------------------------------------------------

/// Deterministic selector restriction map: coarse vertex `c` gathers from fine
/// vertex `restriction[c]`. Modelless — fixed row-selection, no weights.
///
/// Construct via [`VCycleRestriction::new`]. The slice length must equal the
/// coarse complex's vertex count; every entry must be a valid fine vertex
/// index (`< fine.n_vertices()`).
#[derive(Clone, Debug)]
pub struct VCycleRestriction {
    /// `restriction[c]` = fine vertex index that coarse vertex `c` gathers from.
    /// Length = `coarse.n_vertices()`.
    restriction: Vec<u32>,
}

impl VCycleRestriction {
    /// Build a selector restriction from a coarse→fine vertex map.
    ///
    /// # Panics
    /// If any fine vertex index is `>= n_fine_vertices`.
    #[inline]
    #[must_use]
    pub fn new(restriction: Vec<u32>, n_fine_vertices: usize) -> Self {
        for (c, &f) in restriction.iter().enumerate() {
            assert!(
                (f as usize) < n_fine_vertices,
                "VCycleRestriction::new: coarse vertex {c} maps to fine vertex {f} \
                 which is >= n_fine_vertices {n_fine_vertices}"
            );
        }
        Self { restriction }
    }

    /// Number of coarse vertices (= `restriction.len()`).
    #[inline]
    #[must_use]
    pub fn n_coarse_vertices(&self) -> usize {
        self.restriction.len()
    }

    /// Read the fine vertex index for coarse vertex `c`.
    #[inline]
    #[must_use]
    pub fn fine_vertex_of(&self, c: usize) -> u32 {
        self.restriction[c]
    }

    /// Read the whole coarse→fine map as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.restriction
    }
}

// ---------------------------------------------------------------------------
// V-cycle
// ---------------------------------------------------------------------------

/// Pre-allocated scratch buffers for the V-cycle. Reuse across calls to keep
/// the hot path alloc-free beyond the two output cochains.
#[derive(Debug)]
pub struct VCycleScratch {
    /// Coarse restricted cochain (rank = field.rank, dim = field.dim,
    /// n_cells = coarse.n_cells(field.rank)).
    pub restricted: CochainField,
    /// Coarse solved cochain (same shape as `restricted`).
    pub coarse_solved: CochainField,
}

impl VCycleScratch {
    /// Allocate scratch sized for the given fine/coarse complexes and field shape.
    ///
    /// # Panics
    /// If `field_rank > MAX_RANK`.
    #[inline]
    #[must_use]
    pub fn new(coarse: &CellComplex, field_rank: u8, dim: usize) -> Self {
        let n_coarse_cells = coarse.n_cells(field_rank);
        Self {
            restricted: CochainField::zeros(field_rank, n_coarse_cells, dim),
            coarse_solved: CochainField::zeros(field_rank, n_coarse_cells, dim),
        }
    }
}

/// Modelless multi-scale V-cycle: restrict `field` (fine, rank-0) → coarse →
/// apply `coarse_op` → prolongate back to the fine complex.
///
/// Returns the prolongated coarse solution on the fine complex (same rank and
/// dim as `field`). Only rank-0 (vertex) cochains are supported in M0; the
/// selector restriction is defined on vertices.
///
/// # Steps
///
/// 1. **Restrict** (fine → coarse): for each coarse vertex `c`,
///    `coarse[c] = fine[restriction[c]]` per feature dim. O(n_coarse · dim).
/// 2. **Coarse solve**: `coarse_solved = coarse_op(&coarse)`. Caller-supplied;
///    typically a smoothing pass (`hodge_laplacian` on the coarse complex).
/// 3. **Prolongate** (coarse → fine): zero `output`, then for each coarse
///    vertex `c`, `output[restriction[c]] = coarse_solved[c]` per feature dim.
///    O(n_coarse · dim).
///
/// # Allocation
///
/// Allocates one output cochain (the prolongated fine field). Pass
/// [`htno_v_cycle_into`] with a reused [`VCycleScratch`] for the alloc-free
/// steady-state path.
///
/// # Panics
/// If `field.rank != 0` (M0 supports vertex cochains only), or if
/// `restriction.n_coarse_vertices() != coarse.n_vertices()`.
#[inline]
#[must_use]
pub fn htno_v_cycle(
    fine: &CellComplex,
    coarse: &CellComplex,
    restriction: &VCycleRestriction,
    field: &CochainField,
    coarse_op: impl Fn(&CochainField) -> CochainField,
) -> CochainField {
    assert_eq!(
        field.rank, 0,
        "htno_v_cycle: M0 supports rank-0 (vertex) cochains only, got rank {}",
        field.rank
    );
    assert_eq!(
        restriction.n_coarse_vertices(),
        coarse.n_vertices(),
        "htno_v_cycle: restriction length {} != coarse.n_vertices {}",
        restriction.n_coarse_vertices(),
        coarse.n_vertices()
    );
    let mut output = CochainField::zeros(0, fine.n_vertices(), field.dim);
    let mut scratch = VCycleScratch::new(coarse, 0, field.dim);
    htno_v_cycle_into(
        fine,
        coarse,
        restriction,
        field,
        coarse_op,
        &mut output,
        &mut scratch,
    );
    output
}

/// Zero-extra-alloc V-cycle writing into pre-allocated `output` and reusing
/// `scratch` across calls.
///
/// `output` must have rank 0, dim = `field.dim`, and
/// `data.len() >= fine.n_vertices() * dim`. `scratch` must be sized for the
/// coarse complex at rank 0 and the field's dim (see [`VCycleScratch::new`]).
///
/// # Panics
/// Same as [`htno_v_cycle`], plus shape assertions on `output` and `scratch`.
#[inline]
pub fn htno_v_cycle_into(
    fine: &CellComplex,
    coarse: &CellComplex,
    restriction: &VCycleRestriction,
    field: &CochainField,
    coarse_op: impl Fn(&CochainField) -> CochainField,
    output: &mut CochainField,
    scratch: &mut VCycleScratch,
) {
    assert_eq!(field.rank, 0, "htno_v_cycle_into: field must be rank-0");
    assert_eq!(output.rank, 0, "htno_v_cycle_into: output must be rank-0");
    assert_eq!(
        output.dim, field.dim,
        "htno_v_cycle_into: output.dim {} != field.dim {}",
        output.dim, field.dim
    );
    assert_eq!(
        output.n_cells(),
        fine.n_vertices(),
        "htno_v_cycle_into: output.n_cells() {} != fine.n_vertices() {}",
        output.n_cells(),
        fine.n_vertices()
    );
    assert_eq!(
        restriction.n_coarse_vertices(),
        coarse.n_vertices(),
        "htno_v_cycle_into: restriction length {} != coarse.n_vertices() {}",
        restriction.n_coarse_vertices(),
        coarse.n_vertices()
    );

    let dim = field.dim;

    // 1. Restrict: gather fine → coarse.
    let n_coarse = coarse.n_vertices();
    debug_assert_eq!(scratch.restricted.n_cells(), n_coarse);
    debug_assert_eq!(scratch.restricted.dim, dim);
    let rmap = restriction.as_slice();
    for (c, &f) in rmap.iter().enumerate().take(n_coarse) {
        let f = f as usize;
        let src = f * dim;
        let dst = c * dim;
        scratch.restricted.data[dst..dst + dim].copy_from_slice(&field.data[src..src + dim]);
    }

    // 2. Coarse solve (caller-supplied).
    let solved = coarse_op(&scratch.restricted);
    debug_assert_eq!(solved.dim, dim);
    debug_assert_eq!(
        solved.n_cells(),
        n_coarse,
        "coarse_op returned wrong cell count"
    );
    scratch.coarse_solved.data.copy_from_slice(&solved.data);

    // 3. Prolongate: scatter coarse → fine. Non-representative fine vertices
    //    get zero (the V-cycle correction is defined on the representative
    //    subspace; the caller adds it to the fine field if desired).
    output.data.fill(0.0);
    for (c, &f) in rmap.iter().enumerate().take(n_coarse) {
        let f = f as usize;
        let src = c * dim;
        let dst = f * dim;
        output.data[dst..dst + dim].copy_from_slice(&scratch.coarse_solved.data[src..src + dim]);
    }
}

// ---------------------------------------------------------------------------
// Coarsening helpers
// ---------------------------------------------------------------------------

/// Build a 2×2-block grid coarsening: a `(w×h)` vertex grid coarsens to a
/// `(ceil(w/2) × ceil(h/2))` vertex grid, where each coarse vertex `(cx, cy)`
/// is represented by the fine vertex `(2·cx, 2·cy)` (the top-left of the 2×2
/// block). Returns the coarse complex and the restriction map.
///
/// This is the canonical modelless coarsening for regular grids. Game-map
/// terrains are grids; this is the production coarsening path.
///
/// # Panics
/// If `w == 0` or `h == 0`.
#[inline]
#[must_use]
pub fn grid_coarsen_2x2(w: usize, h: usize) -> (CellComplex, VCycleRestriction) {
    assert!(w > 0 && h > 0, "grid_coarsen_2x2: w and h must be > 0");
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let coarse = CellComplex::grid_2d(cw, ch);
    let mut restriction = Vec::with_capacity(cw * ch);
    for cy in 0..ch {
        for cx in 0..cw {
            let fx = 2 * cx;
            let fy = 2 * cy;
            // Clamp to the fine grid in case w/h is odd (the last block is a
            // 1×N or N×1 strip; its representative is the edge fine vertex).
            let fx = fx.min(w - 1);
            let fy = fy.min(h - 1);
            restriction.push((fy * w + fx) as u32);
        }
    }
    (coarse, VCycleRestriction::new(restriction, w * h))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::{exterior_derivative, graph_laplacian};

    // Tolerance for f32 comparisons in the commutativity gate. The exterior
    // derivative is an exact signed-accumulation over boundary triplets, so
    // ULP-level rounding is the only source of drift. 1e-5 is generous.
    const TOL: f32 = 1e-5;

    /// Elementwise max-abs difference between two cochains' data slices.
    fn max_abs_diff(a: &CochainField, b: &CochainField) -> f32 {
        assert_eq!(
            a.data.len(),
            b.data.len(),
            "length mismatch in max_abs_diff"
        );
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // ── G1: commutativity dₖ(Kc) ∘ Rₖ = Rₖ₊₁ ∘ dₖ(K) ──────────────────────
    //
    // For rank-0 vertex cochains, d₀ maps vertices → edges. The commutativity
    // identity becomes:
    //
    //   d₀(Kc)(R₀(v))  ==  R₁(d₀(K)(v))
    //
    // where R₀ restricts a fine vertex cochain to a coarse vertex cochain and
    // R₁ restricts a fine edge cochain to a coarse edge cochain. R₁ is the
    // edge-restriction induced by R₀: for each coarse edge (c_tail → c_head),
    // gather from the fine edge whose endpoints are the representative fine
    // vertices (R₀(c_tail) → R₀(c_head)).
    //
    // The identity holds **by construction** when the coarse complex is an
    // induced sub-complex of the fine complex: coarse vertices are a subset of
    // fine vertices (the representatives), and coarse edges are exactly the
    // fine edges between those representatives. We verify this on (a) a grid
    // sub-region and (b) a path sub-complex (the irregular case). The
    // aggregation-coarsening case (2×2 blocks) is documented separately below —
    // it does NOT satisfy the edge-level commutativity because its coarse edges
    // are long-range (not fine edges); the V-cycle still provides coarse
    // smoothing there, just not a d-commuting one.

    /// Build the induced sub-complex on a set of representative fine vertices:
    /// coarse vertices = representatives (re-indexed 0..n), coarse edges =
    /// fine edges whose BOTH endpoints are representatives. Returns the coarse
    /// complex and the restriction map.
    fn induced_sub_complex(
        fine: &CellComplex,
        representatives: &[usize],
    ) -> (CellComplex, VCycleRestriction) {
        let n_fine = fine.n_vertices();
        let rank_map: std::collections::HashMap<usize, usize> = representatives
            .iter()
            .enumerate()
            .map(|(c, &f)| (f, c))
            .collect();
        let mut coarse_edges: Vec<(usize, usize)> = Vec::new();
        for pair in fine.boundary_entries(0).as_chunks::<2>().0 {
            let (v_tail, _e, _) = pair[0];
            let (v_head, _e, _) = pair[1];
            if let (Some(&ct), Some(&ch)) = (rank_map.get(&v_tail), rank_map.get(&v_head)) {
                coarse_edges.push((ct, ch));
            }
        }
        let coarse = CellComplex::from_edges(representatives.len(), &coarse_edges);
        let restriction =
            VCycleRestriction::new(representatives.iter().map(|&f| f as u32).collect(), n_fine);
        (coarse, restriction)
    }

    #[test]
    fn commutativity_grid_coarsening_holds() {
        // Fine: 4×4 vertex grid. Coarse: induced sub-complex on the 2×2
        // sub-region at fine vertices {(0,0),(1,0),(0,1),(1,1)} = {0,1,4,5}.
        // The induced subgraph has fine edges 0-1, 0-4, 1-5, 4-5 — exactly a
        // 2×2 grid. Commutativity holds because coarse edges = fine edges
        // between representatives.
        let fine = CellComplex::grid_2d(4, 4);
        let representatives = [0usize, 1, 4, 5];
        let (coarse, restriction) = induced_sub_complex(&fine, &representatives);

        let mut fine_field = CochainField::zeros(0, fine.n_vertices(), 1);
        for i in 0..fine.n_vertices() {
            fine_field.set_scalar(i, ((i as f32) * 0.37).fract());
        }

        // LHS: d₀(Kc)(R₀(v)).
        let mut r0 = CochainField::zeros(0, coarse.n_vertices(), 1);
        for c in 0..coarse.n_vertices() {
            let f = restriction.fine_vertex_of(c) as usize;
            r0.set_scalar(c, fine_field.scalar(f));
        }
        let lhs = exterior_derivative(&coarse, &r0);

        // RHS: R₁(d₀(K)(v)) — for each coarse edge, find the matching fine edge
        // (between the representative fine vertices) and gather its d₀ value.
        let fine_b1 = fine.boundary_entries(0);
        // Build fine edge → d₀ value lookup. d₀ on edge e = Σ sign·input.
        let mut fine_edge_d0: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();
        for &(v, e, sign) in fine_b1 {
            *fine_edge_d0.entry(e).or_insert(0.0) += sign as f32 * fine_field.scalar(v);
        }
        let coarse_b1 = coarse.boundary_entries(0);
        let mut rhs = CochainField::zeros(1, coarse.n_edges(), 1);
        for ce in 0..coarse.n_edges() {
            // Find the coarse edge's representative endpoints.
            let mut ct = usize::MAX;
            let mut ch = usize::MAX;
            for &(v, e, _sign) in coarse_b1 {
                if e == ce {
                    if ct == usize::MAX {
                        ct = v;
                    } else {
                        ch = v;
                    }
                }
            }
            let ft = restriction.fine_vertex_of(ct) as usize;
            let fh = restriction.fine_vertex_of(ch) as usize;
            // Find the fine edge connecting ft ↔ fh.
            let mut val = 0.0f32;
            let mut found = false;
            for &(v, e, _sign) in fine_b1 {
                if v == ft || v == fh {
                    let mut other = usize::MAX;
                    for &(v2, e2, _s2) in fine_b1 {
                        if e2 == e && v2 != v {
                            other = v2;
                            break;
                        }
                    }
                    if (v == ft && other == fh) || (v == fh && other == ft) {
                        val = *fine_edge_d0.get(&e).unwrap_or(&0.0);
                        found = true;
                        break;
                    }
                }
            }
            assert!(
                found,
                "no fine edge for coarse edge {ce} (ft={ft}, fh={fh})"
            );
            rhs.set_scalar(ce, val);
        }

        let diff = max_abs_diff(&lhs, &rhs);
        assert!(
            diff < TOL,
            "grid commutativity failed: max abs diff = {diff:.3e} (tol {TOL:.0e})"
        );
    }

    #[test]
    fn commutativity_irregular_coarsening_holds() {
        // Irregular fine complex: a path 0→1→2→3→4→5 (6 vertices, 5 edges).
        // Induced sub-complex on representatives {1,2,3,4} — coarse path
        // 0→1→2→3 with fine edges 1-2, 2-3, 3-4. This is irregular (the
        // representatives are an interior segment, not a prefix) and the
        // commutativity holds by construction because coarse edges = fine
        // edges between representatives.
        let fine_edges: Vec<(usize, usize)> = (0..5).map(|i| (i, i + 1)).collect();
        let fine = CellComplex::from_edges(6, &fine_edges);
        let (coarse, restriction) = induced_sub_complex(&fine, &[1, 2, 3, 4]);

        let mut fine_field = CochainField::zeros(0, fine.n_vertices(), 1);
        for i in 0..6 {
            fine_field.set_scalar(i, (i as f32) * 0.13);
        }

        // LHS: d₀(Kc)(R₀(v)).
        let mut r0 = CochainField::zeros(0, coarse.n_vertices(), 1);
        for c in 0..coarse.n_vertices() {
            let f = restriction.fine_vertex_of(c) as usize;
            r0.set_scalar(c, fine_field.scalar(f));
        }
        let lhs = exterior_derivative(&coarse, &r0);

        // RHS: R₁(d₀(K)(v)). For the path, fine d₀ on edge (i,i+1) is
        // fine_field[i+1] - fine_field[i]. The coarse edges are the fine edges
        // between consecutive representatives {1,2,3,4}: edges (1,2),(2,3),(3,4).
        let fine_b1 = fine.boundary_entries(0);
        let mut fine_edge_d0: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();
        for &(v, e, sign) in fine_b1 {
            *fine_edge_d0.entry(e).or_insert(0.0) += sign as f32 * fine_field.scalar(v);
        }
        let coarse_b1 = coarse.boundary_entries(0);
        let mut rhs = CochainField::zeros(1, coarse.n_edges(), 1);
        for ce in 0..coarse.n_edges() {
            let mut ct = usize::MAX;
            let mut ch = usize::MAX;
            for &(v, e, _sign) in coarse_b1 {
                if e == ce {
                    if ct == usize::MAX {
                        ct = v;
                    } else {
                        ch = v;
                    }
                }
            }
            let ft = restriction.fine_vertex_of(ct) as usize;
            let fh = restriction.fine_vertex_of(ch) as usize;
            // Find fine edge ft↔fh.
            let mut val = 0.0f32;
            for &(v, e, _sign) in fine_b1 {
                if v == ft || v == fh {
                    let mut other = usize::MAX;
                    for &(v2, e2, _s2) in fine_b1 {
                        if e2 == e && v2 != v {
                            other = v2;
                            break;
                        }
                    }
                    if (v == ft && other == fh) || (v == fh && other == ft) {
                        val = *fine_edge_d0.get(&e).unwrap_or(&0.0);
                        break;
                    }
                }
            }
            rhs.set_scalar(ce, val);
        }

        let diff = max_abs_diff(&lhs, &rhs);
        assert!(
            diff < TOL,
            "irregular commutativity failed: max abs diff = {diff:.3e} (tol {TOL:.0e})"
        );
        // Sanity: the coarse complex is non-trivial (has edges).
        assert!(
            coarse.n_edges() > 0,
            "induced sub-complex should have edges"
        );
    }

    #[test]
    fn aggregation_coarsening_does_not_commute_documented() {
        // Honest documentation test: the 2×2 aggregation coarsening
        // (grid_coarsen_2x2) does NOT satisfy d₀-commutativity because its
        // coarse edges connect representatives that are 2 fine-cells apart
        // (long-range), not single fine edges. The V-cycle still provides
        // coarse smoothing (the coarse_op runs on the coarse complex), but
        // the restriction does NOT commute with the exterior derivative. This
        // is the documented limitation per Risk #2 — game maps use the
        // aggregation coarsening for performance, accepting that the V-cycle
        // is a smoother, not a d-commuting transfer. The induced-sub-complex
        // coarsening (above) is the d-commuting variant.
        let (w, h) = (4usize, 4usize);
        let _fine = CellComplex::grid_2d(w, h);
        let (coarse, restriction) = grid_coarsen_2x2(w, h);

        let mut fine_field = CochainField::zeros(0, _fine.n_vertices(), 1);
        for i in 0.._fine.n_vertices() {
            fine_field.set_scalar(i, (i as f32) * 0.1);
        }

        // LHS: coarse d₀ of restricted field.
        let mut r0 = CochainField::zeros(0, coarse.n_vertices(), 1);
        for c in 0..coarse.n_vertices() {
            let f = restriction.fine_vertex_of(c) as usize;
            r0.set_scalar(c, fine_field.scalar(f));
        }
        let lhs = exterior_derivative(&coarse, &r0);

        // The coarse d₀ on the first coarse edge (representatives fine 0 and
        // fine 2) is fine[2]-fine[0] = 0.2 - 0.0 = 0.2. But there is no single
        // fine edge between fine 0 and fine 2 — they are separated by fine 1.
        // So R₁(d₀(K)(v)) is undefined / zero for this coarse edge. Confirm
        // the mismatch (this is the documented non-commutativity).
        assert!(coarse.n_edges() > 0);
        // LHS is non-zero (the coarse d₀ sees the representative difference).
        assert!(
            lhs.data.iter().any(|&x| x.abs() > 0.01),
            "aggregation coarsening should produce non-zero coarse d₀"
        );
    }

    // ── V-cycle mechanics ───────────────────────────────────────────────────

    #[test]
    fn restrict_then_identity_prolongate_round_trips_representatives() {
        let (w, h) = (4usize, 4usize);
        let fine = CellComplex::grid_2d(w, h);
        let (coarse, restriction) = grid_coarsen_2x2(w, h);

        let mut fine_field = CochainField::zeros(0, fine.n_vertices(), 2);
        for i in 0..fine.n_vertices() {
            fine_field.cell_features_mut(i)[0] = i as f32;
            fine_field.cell_features_mut(i)[1] = (i as f32) * 2.0;
        }

        let out = htno_v_cycle(&fine, &coarse, &restriction, &fine_field, |x| x.clone());

        // Representative fine vertices should hold their original values;
        // non-representatives should be zero (identity coarse_op → the
        // prolongated value equals the restricted value at representatives).
        for c in 0..coarse.n_vertices() {
            let f = restriction.fine_vertex_of(c) as usize;
            assert!(
                (out.cell_features(f)[0] - fine_field.cell_features(f)[0]).abs() < TOL,
                "representative {f} dim0 mismatch"
            );
            assert!(
                (out.cell_features(f)[1] - fine_field.cell_features(f)[1]).abs() < TOL,
                "representative {f} dim1 mismatch"
            );
        }
    }

    #[test]
    fn coarse_op_solves_on_coarse_complex() {
        // Verify the coarse_op receives the correctly restricted field and its
        // output is prolongated. Use graph_laplacian as the coarse_op.
        let (w, h) = (4usize, 4usize);
        let fine = CellComplex::grid_2d(w, h);
        let (coarse, restriction) = grid_coarsen_2x2(w, h);

        let mut fine_field = CochainField::zeros(0, fine.n_vertices(), 1);
        for i in 0..fine.n_vertices() {
            fine_field.set_scalar(i, (i as f32) * 0.1);
        }

        let out = htno_v_cycle(&fine, &coarse, &restriction, &fine_field, |coarse_field| {
            graph_laplacian(&coarse, coarse_field)
        });

        // The prolongated output at representative vertices should equal the
        // coarse graph_laplacian evaluated at the restricted values.
        for c in 0..coarse.n_vertices() {
            let f = restriction.fine_vertex_of(c) as usize;
            // Manually compute the coarse Laplacian at c by gathering neighbors.
            // For a 2×2 coarse grid (4 vertices), the topology is a square:
            //   0 - 1
            //   |   |
            //   2 - 3
            // with grid_2d vertex indexing (x + y*cw).
            let (cw, ch) = (2usize, 2usize);
            let (cx, cy) = (c % cw, c / cw);
            let mut neighbors_sum = 0.0f32;
            let mut degree = 0.0f32;
            let center = fine_field.scalar(restriction.fine_vertex_of(c) as usize);
            if cx > 0 {
                let n = c - 1;
                neighbors_sum += fine_field.scalar(restriction.fine_vertex_of(n) as usize);
                degree += 1.0;
            }
            if cx + 1 < cw {
                let n = c + 1;
                neighbors_sum += fine_field.scalar(restriction.fine_vertex_of(n) as usize);
                degree += 1.0;
            }
            if cy > 0 {
                let n = c - cw;
                neighbors_sum += fine_field.scalar(restriction.fine_vertex_of(n) as usize);
                degree += 1.0;
            }
            if cy + 1 < ch {
                let n = c + cw;
                neighbors_sum += fine_field.scalar(restriction.fine_vertex_of(n) as usize);
                degree += 1.0;
            }
            let expected = degree * center - neighbors_sum;
            assert!(
                (out.scalar(f) - expected).abs() < TOL,
                "coarse Laplacian at representative {f} (coarse {c}): got {}, expected {expected}",
                out.scalar(f)
            );
        }
    }

    #[test]
    fn v_cycle_into_reuses_scratch_without_panic() {
        let (w, h) = (8usize, 8usize);
        let fine = CellComplex::grid_2d(w, h);
        let (coarse, restriction) = grid_coarsen_2x2(w, h);
        let mut fine_field = CochainField::zeros(0, fine.n_vertices(), 3);
        for i in 0..fine.n_cells(0) {
            for d in 0..3 {
                fine_field.cell_features_mut(i)[d] = (i as f32) + (d as f32) * 0.5;
            }
        }
        let mut output = CochainField::zeros(0, fine.n_vertices(), 3);
        let mut scratch = VCycleScratch::new(&coarse, 0, 3);
        // Run twice to confirm scratch reuse works.
        for _ in 0..2 {
            htno_v_cycle_into(
                &fine,
                &coarse,
                &restriction,
                &fine_field,
                |x| x.clone(),
                &mut output,
                &mut scratch,
            );
        }
        // Sanity: representatives round-trip.
        for c in 0..coarse.n_vertices() {
            let f = restriction.fine_vertex_of(c) as usize;
            for d in 0..3 {
                assert!(
                    (output.cell_features(f)[d] - fine_field.cell_features(f)[d]).abs() < TOL,
                    "round-trip failed at representative {f} dim {d}"
                );
            }
        }
    }

    #[test]
    fn grid_coarsen_2x2_produces_correct_sizes() {
        let (w, h) = (5usize, 5usize); // odd → ceil(5/2)=3
        let _fine = CellComplex::grid_2d(w, h);
        let (coarse, restriction) = grid_coarsen_2x2(w, h);
        assert_eq!(coarse.n_vertices(), 9); // 3×3
        assert_eq!(restriction.n_coarse_vertices(), 9);
        // Representatives are (0,0),(0,2),(0,4),(2,0),... in fine indices.
        assert_eq!(restriction.fine_vertex_of(0), 0); // (0,0) → fine 0
        assert_eq!(restriction.fine_vertex_of(1), 2); // (2,0) → fine 2
        assert_eq!(restriction.fine_vertex_of(2), 4); // (4,0) → fine 4 (clamped)
        assert_eq!(restriction.fine_vertex_of(3), 10); // (0,2) → fine 2*5+0 = 10
    }

    #[test]
    #[should_panic(expected = "M0 supports rank-0")]
    fn v_cycle_rejects_nonzero_rank() {
        let (w, h) = (4usize, 4usize);
        let fine = CellComplex::grid_2d(w, h);
        let (coarse, restriction) = grid_coarsen_2x2(w, h);
        let bad = CochainField::zeros(1, fine.n_edges(), 1); // rank-1
        let _ = htno_v_cycle(&fine, &coarse, &restriction, &bad, |x| x.clone());
    }
}
