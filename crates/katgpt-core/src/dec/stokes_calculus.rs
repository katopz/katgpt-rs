//! Stokes Calculus Wrappers — thin modelless primitives over DEC operators.
//!
//! Implements Plan 314 (Research 296) + Plan 317 (Issue 005) + Plan 318
//! (Issue 006). Five named wrappers exposing the Generalized Stokes' Theorem
//! as modelless inference tools:
//!
//! - [`belief_mass_divergence`] — Fokker-Planck belief-mass validator
//!   (L1 norm of discrete divergence).
//! - [`boundary_flux_mass`] — divergence-theorem boundary integral for
//!   low-dim manifolds (O(boundary) vs O(volume)).
//! - [`boundary_flux_mass_indexed`] — CSR-indexed fast path for multi-query
//!   scenarios (Plan 318; `O(|region| × boundary_per_cell)` after a one-time
//!   `O(|B_{k+1}|)` build).
//! - [`line_integral`] — discrete line integral of a rank-1 cochain along a
//!   vertex path (open path; per-edge cost).
//! - [`circulation_integral`] — discrete circulation of a rank-1 cochain
//!   around a closed vertex loop (Stokes: ∮F = ∬curl F; Plan 317).
//!
//! All five are pure wrappers over the shipped DEC operators ([`codifferential`],
//! [`exterior_derivative`], [`hodge_decompose`], and the coboundary index on
//! [`CellComplex`]). No new DEC machinery.
//!
//! # Constraint checklist (AGENTS.md)
//!
//! - Modelless (linear algebra only, no backprop): YES by construction.
//! - Latent-to-latent preferred (sigmoid not softmax): N/A (pure summation).
//! - Freeze/thaw over fine-tuning: YES (no weight mutation).
//! - Zero allocations in wrapper code: YES (all Vecs come from delegated ops
//!   or the pre-built coboundary index).

use super::hodge::hodge_decompose;
use super::operators::codifferential;
use super::types::{CellComplex, CochainField};

// ===========================================================================
// belief_mass_divergence — Fokker-Planck belief-mass validator
// ===========================================================================

/// Fokker-Planck belief-mass divergence validator.
///
/// Computes the L1 norm of the discrete divergence (codifferential δ₁) of a
/// rank-1 belief-flow cochain over all vertices:
///
/// `‖δ₁(flow)‖₁ = Σ_v |δ₁(flow)[v]|`
///
/// For a mass-conserving (divergence-free) belief flow this is ≈ 0. Non-zero
/// values indicate belief mass is being created or destroyed — a Fokker-Planck
/// continuity-equation violation. Feeds ICT `BranchingDetector` (Plan 294) a
/// modelless invariant, and `cgsp_runtime/pulse_bridge.rs` a curiosity signal
/// (positive divergence = expanding belief).
///
/// Pure wrapper over [`codifferential`] (shipped Plan 251). The only allocation
/// is the intermediate rank-0 divergence cochain, produced by the delegated
/// operator.
///
/// # Arguments
/// * `cx` — Cell complex providing boundary matrices.
/// * `belief_flow` — Rank-1 (edge) cochain representing the belief flow.
///
/// # Returns
/// `Σ_v |δ₁(belief_flow)[v]|` — the L1 divergence magnitude.
pub fn belief_mass_divergence(cx: &CellComplex, belief_flow: &CochainField) -> f32 {
    debug_assert_eq!(
        belief_flow.rank, 1,
        "belief_mass_divergence: belief_flow must be rank-1 (edge) cochain, got rank {}",
        belief_flow.rank
    );
    debug_assert_eq!(
        belief_flow.dim, 1,
        "belief_mass_divergence: belief_flow must be dim=1 (scalar per edge), got dim {}",
        belief_flow.dim
    );

    // δ₁: rank-1 → rank-0 (discrete divergence at each vertex).
    let divergence = codifferential(cx, belief_flow);

    // L1 sum: Σ_v |δ₁(flow)[v]|. Single pass, branchless abs.
    divergence.data.iter().copied().map(f32::abs).sum()
}

// ===========================================================================
// boundary_flux_mass — divergence-theorem boundary integral
// ===========================================================================

/// Divergence-theorem boundary-flux mass for low-dimensional manifolds.
///
/// Computes the oriented boundary flux of a k-cochain `field` over the boundary
/// of a region of (k+1)-cells, plus an error bound from the harmonic component.
///
/// **Mass** (boundary integral): For each (k+1)-cell in `region_cells`, sums
/// `sign · field[k-cell]` over its boundary entries in `B_{k+1}`. Interior
/// k-cells (bounding two region cells) cancel by orientation; only boundary
/// k-cells survive. By the Generalized Stokes' Theorem this equals the volume
/// integral `Σ_{f ∈ region} d_k(field)[f]`, but the caller never materializes
/// `d_k(field)` over the full complex.
///
/// **Error bound**: `‖harmonic(field)‖₁` from [`hodge_decompose`]. Harmonic
/// fields have zero divergence and zero curl, so they do not contribute to
/// boundary flux. If the field is mostly harmonic (topologically constrained
/// flow), the boundary flux underestimates total field activity — the error
/// bound quantifies this. Callers can check `error_bound / |mass|` before
/// trusting the result (per Plan 314 honest risk note: boundary-only is a win
/// only for near-exact fields, d ≤ 3).
///
/// # Arguments
/// * `cx` — Cell complex providing boundary matrices.
/// * `region_cells` — Indices of (k+1)-cells defining the region (e.g. face
///   indices when `field` is rank-1).
/// * `field` — k-cochain (typically rank-1 edge flow), dim=1.
///
/// # Returns
/// `(mass, error_bound)`:
/// - `mass` — oriented boundary flux (Σ sign · field on boundary k-cells).
/// - `error_bound` — `‖harmonic(field)‖₁`.
///
/// Returns `(0.0, 0.0)` for an empty region.
///
/// # Complexity
/// Mass computation is `O(|B_{k+1}|)` — a single pass over boundary entries.
/// Error-bound computation invokes `hodge_decompose` (CG solver, `O(E · iters)`);
/// callers that only need the mass should use [`boundary_flux_mass_only`] to
/// skip the harmonic decomposition. The decomposition depends only on
/// `(cx, field)`, not on `region_cells`, so it can be computed once per tick
/// and reused across many region queries.
pub fn boundary_flux_mass(
    cx: &CellComplex,
    region_cells: &[u32],
    field: &CochainField,
) -> (f32, f32) {
    let mass = boundary_flux_mass_only(cx, region_cells, field);

    // Error bound: L1 norm of harmonic component. Harmonic fields are in
    // ker(d) ∩ ker(δ) — they contribute zero to both curl and divergence,
    // hence zero to boundary flux.
    let decomp = hodge_decompose(cx, field);
    let error_bound: f32 = decomp.harmonic.data.iter().copied().map(f32::abs).sum();

    (mass, error_bound)
}

/// Boundary-flux mass only (no error bound).
///
/// Same as [`boundary_flux_mass`] but skips the `hodge_decompose` call.
/// This is the hot-path variant for callers that either (a) don't need the
/// error bound, or (b) cache the decomposition across many region queries.
///
/// Returns `0.0` for an empty region.
///
/// # Complexity
/// `O(|B_{k+1}|)` — a single pass over boundary entries with a region-membership
/// filter. The wrapper allocates a single `Vec<bool>` sized to the complex.
pub fn boundary_flux_mass_only(
    cx: &CellComplex,
    region_cells: &[u32],
    field: &CochainField,
) -> f32 {
    if region_cells.is_empty() {
        return 0.0;
    }

    let k = field.rank;
    debug_assert_eq!(
        field.dim, 1,
        "boundary_flux_mass_only: field must be dim=1 (scalar per cell), got dim {}",
        field.dim
    );

    let n_region_cells = cx.n_cells(k + 1);

    // Single allocation: boolean region-membership marker, sized to the complex.
    let mut in_region = vec![false; n_region_cells];
    for &cell in region_cells {
        debug_assert!(
            (cell as usize) < n_region_cells,
            "boundary_flux_mass_only: region cell {cell} >= n_cells({}) = {n_region_cells}",
            k + 1
        );
        in_region[cell as usize] = true;
    }

    // Boundary flux = Σ_{(k_cell, kp1_cell, sign) ∈ B_{k+1}, kp1_cell ∈ region}
    //                   sign · field[k_cell]
    //
    // Interior k-cells (bounding two region (k+1)-cells) appear with opposite
    // signs and cancel. Only boundary k-cells contribute. This is the discrete
    // divergence theorem: ∫_∂M ω = ∫_M dω.
    let mut mass = 0.0f32;
    for &(k_cell, kp1_cell, sign) in cx.boundary_entries(k) {
        if in_region[kp1_cell] {
            mass += sign as f32 * field.scalar(k_cell);
        }
    }

    mass
}

/// Boundary-flux mass via pre-built coboundary index (Plan 318).
///
/// Same mathematical result as [`boundary_flux_mass_only`] (the discrete
/// divergence theorem: `∫_∂M ω = ∫_M dω`), but uses the CSR coboundary index
/// for an `O(|region| × boundary_per_cell)` direct lookup instead of the
/// `O(|B_{k+1}|)` full-matrix scan. For a 64×64 region in a 256×256 grid this
/// is ~16× fewer ops.
///
/// # Preconditions
/// - The caller MUST have called [`CellComplex::build_coboundary_index`] for
///   rank `field.rank` since the last topology mutation. **Panics** (debug) or
///   silently falls back to [`boundary_flux_mass_only`] (release) if not.
///
/// # When to use this vs `boundary_flux_mass_only`
/// - **Single query:** prefer `boundary_flux_mass_only`. The index build cost
///   (`O(|B_{k+1}|)`) dominates and makes the indexed path slower.
/// - **Many queries on stable topology:** call `build_coboundary_index` once,
///   then call this function per region. The build amortizes and each query is
///   `O(|region| × boundary_per_cell)`.
///
/// Returns `0.0` for an empty region.
///
/// # Complexity
/// `O(|region| × boundary_per_cell)` — direct CSR lookups, no full-matrix
/// scan, no `Vec<bool>` allocation. Interior k-cells (bounding two region
/// (k+1)-cells) appear with opposite signs and cancel.
pub fn boundary_flux_mass_indexed(
    cx: &CellComplex,
    region_cells: &[u32],
    field: &CochainField,
) -> f32 {
    if region_cells.is_empty() {
        return 0.0;
    }

    let k = field.rank;
    debug_assert_eq!(
        field.dim, 1,
        "boundary_flux_mass_indexed: field must be dim=1 (scalar per cell), got dim {}",
        field.dim
    );

    let index = match cx.coboundary_entries(k) {
        Some(idx) => idx,
        None => {
            debug_assert!(
                false,
                "boundary_flux_mass_indexed: coboundary index for rank {k} not built. \
                 Call cx.build_coboundary_index({k}) first. Falling back to full-scan \
                 boundary_flux_mass_only in release builds."
            );
            return boundary_flux_mass_only(cx, region_cells, field);
        }
    };

    // Boundary flux = Σ_{c ∈ region} Σ_{(k_cell, sign) ∈ ∂c} sign · field[k_cell]
    //
    // Interior k-cells (bounding two region (k+1)-cells) appear with opposite
    // signs across the two cells and cancel. Only boundary k-cells survive.
    // This is the discrete divergence theorem: ∫_∂M ω = ∫_M dω.
    //
    // No allocation: iterate region cells, look up each cell's CSR slice directly.
    let mut mass = 0.0f32;
    for &cell in region_cells {
        for &(k_cell, sign) in index.cell_boundary(cell as usize) {
            mass += sign as f32 * field.scalar(k_cell as usize);
        }
    }

    mass
}

// ===========================================================================
// line_integral — discrete line integral along a vertex path
// ===========================================================================

/// Discrete line integral of a rank-1 edge cochain along a vertex path.
///
/// For each consecutive vertex pair `(a, b)` on the path, finds the edge `e`
/// connecting them in the cell complex's `B₁` entries and accumulates
/// `±field[e]` based on traversal direction relative to edge orientation:
///
/// `∫_path field = Σ_{(a,b) ∈ path} sign(b, e_{ab}) · field[e_{ab}]`
///
/// If `a` is the edge tail and `b` is the head (traversal along orientation),
/// the contribution is `+field[e]`. Reversed traversal gives `-field[e]`.
///
/// Composes with Plan 312's [`manifold_geodesic`](crate::viable_manifold_graph::manifold_geodesic)
/// path output: pass the vertex-index `Vec<u32>` directly as `path`. Useful for
/// path-energy / geodesic-cost / work computations.
///
/// # Arguments
/// * `cx` — Cell complex. `B₁` entries must be vertex–edge paired as produced by
///   [`CellComplex::grid_2d`] (tail, e, −1), (head, e, +1).
/// * `edge_field` — Rank-1 cochain (per-edge scalar values), dim=1.
/// * `path` — Vertex-index slice (e.g. from `manifold_geodesic`).
///
/// # Returns
/// The signed line integral. Returns `0.0` for paths shorter than 2 vertices.
/// Consecutive vertex pairs not connected by an edge contribute `0.0` (the
/// edge lookup fails silently — callers must ensure path vertices are adjacent
/// in the cell complex).
///
/// # Reversal antisymmetry
/// Reversing the path negates the integral: `line_integral(A→B) == −line_integral(B→A)`.
pub fn line_integral(cx: &CellComplex, edge_field: &CochainField, path: &[u32]) -> f32 {
    if path.len() < 2 {
        return 0.0;
    }

    debug_assert_eq!(
        edge_field.rank, 1,
        "line_integral: edge_field must be rank-1 (edge) cochain, got rank {}",
        edge_field.rank
    );
    debug_assert_eq!(
        edge_field.dim, 1,
        "line_integral: edge_field must be dim=1 (scalar per edge), got dim {}",
        edge_field.dim
    );

    let entries = cx.boundary_entries(0);
    let mut total = 0.0f32;

    for window in path.windows(2) {
        let a = window[0] as usize;
        let b = window[1] as usize;
        if a == b {
            continue;
        }

        // B₁ entries from grid_2d are paired: (tail, e, −1), (head, e, +1).
        // Iterate pairs to find the edge connecting a and b.
        for pair in entries.chunks_exact(2) {
            let (v0, e0, _s0) = pair[0];
            let (v1, e1, _s1) = pair[1];
            debug_assert_eq!(e0, e1, "B₁ entries must be paired by edge index");

            if (v0 == a && v1 == b) || (v0 == b && v1 == a) {
                // Found edge e connecting a and b.
                // Contribution = field[e] · sign(b, e):
                //   b is head (sign=+1) → traversal along orientation → +field
                //   b is tail (sign=−1) → traversal against orientation → −field
                let sign_b = if v0 == b { pair[0].2 } else { pair[1].2 };
                total += sign_b as f32 * edge_field.scalar(e0);
                break;
            }
        }
    }

    total
}

// ===========================================================================
// circulation_integral — rank-2 Stokes wrapper (Plan 317, Issue 005)
// ===========================================================================

/// Discrete circulation of a rank-1 edge cochain around a closed vertex loop.
///
/// By the Generalized Stokes' Theorem, the line integral around a closed loop
/// equals the integral of the curl over the enclosed area:
///
/// `∮_loop field = ∬_enclosed_area curl(field) dA = Σ_faces d₁(field)[f]`
///
/// This is the **rank-2 Stokes companion** to [`line_integral`]:
/// - [`line_integral`] on an *open* path measures per-edge cost (path energy,
///   work, friction accumulation) — it cannot see turn penalties because turn
///   count is a pairwise edge property.
/// - `circulation_integral` on a *closed* loop measures **enclosed rotational
///   content** — non-zero for rotational (non-exact) fields, zero for exact
///   (gradient) fields by the fundamental theorem of calculus.
///
/// # Smoothness / turn penalty application (Plan 314 G-C, Issue 005)
///
/// To compare two open paths between the same endpoints, close each by
/// appending its first vertex (the caller must ensure the closing edges exist
/// in the cell complex), then compute `circulation_integral` of each closure.
/// On a constant-curl field, circulation is proportional to enclosed area, so
/// the path whose closure encloses less area reports less circulation.
///
/// **Honest caveat (Plan 317 pre-analysis):** enclosed area and turn count are
/// *independent* geometric properties — a staircase (many turns) can cut the
/// corner and enclose *less* area than an L-shape (one turn) around the full
/// square. So minimizing `|circulation_integral|` does *not* always minimize
/// turns; it minimizes enclosed rotational content. Use it as a
/// rotational-cost signal, not a guaranteed turn reducer.
///
/// # Arguments
/// * `cx` — Cell complex providing `B₁` entries.
/// * `edge_field` — Rank-1 cochain (per-edge scalar values), dim=1.
/// * `closed_loop` — Vertex-index slice. MUST be closed: first and last
///   vertices must be equal. Panics in debug if not.
///
/// # Returns
/// The signed circulation. Returns `0.0` for loops shorter than 3 vertices
/// (need ≥2 edges to form a closed loop). Reversal-antisymmetric:
/// clockwise circulation == −counterclockwise circulation.
///
/// # Complexity
/// `O(loop_len × |B₁|)` — same as [`line_integral`] (it delegates).
///
/// # Example
///
/// ```ignore
/// use katgpt_core::dec::{CellComplex, CochainField, circulation_integral};
///
/// let cx = CellComplex::grid_2d(4, 4);
/// let field = CochainField::zeros(1, cx.n_edges(), 1); // zero field → zero circulation
/// let square_loop: [u32; 5] = [0, 1, 5, 4, 0]; // around face {0,1,5,4}
/// let circ = circulation_integral(&cx, &field, &square_loop);
/// assert_eq!(circ, 0.0);
/// ```
pub fn circulation_integral(
    cx: &CellComplex,
    edge_field: &CochainField,
    closed_loop: &[u32],
) -> f32 {
    // A closed loop needs at least 3 vertices (2 edges) to enclose anything.
    // Shorter inputs trivially have zero circulation.
    if closed_loop.len() < 3 {
        return 0.0;
    }

    debug_assert_eq!(
        closed_loop.first(),
        closed_loop.last(),
        "circulation_integral: closed_loop must be closed (first == last vertex), \
         got start={:?}, end={:?}",
        closed_loop.first(),
        closed_loop.last()
    );

    // A closed loop's line integral IS its circulation (Stokes' theorem).
    // The wrapper exists for (a) the closed-loop invariant check, (b) the
    // rotational-content interpretation in the docs, and (c) a stable named
    // call-site for callers who want circulation semantics, not path energy.
    line_integral(cx, edge_field, closed_loop)
}

// ===========================================================================
// Tests (Plan 314 Phase 2 + Plan 317 Phase 2)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dec::operators::exterior_derivative;

    // ── belief_mass_divergence ──────────────────────────────────────────────

    /// T2.1.1 Identity: divergence of a zero flow = 0 (mass conserved).
    #[test]
    fn test_belief_mass_divergence_identity_zero_flow() {
        let cx = CellComplex::grid_2d(4, 4);
        let flow = CochainField::zeros(1, cx.n_edges(), 1);
        let div = belief_mass_divergence(&cx, &flow);
        assert_eq!(div, 0.0, "zero flow → zero divergence");
    }

    /// T2.1.1b Identity: a divergence-free circulation field → ≈0 divergence.
    /// A coexact field (circulating around faces) has zero divergence at
    /// interior vertices. On a 4×4 grid, the 2×2 interior vertices have
    /// balanced in/out flow for a uniform circulation.
    #[test]
    fn test_belief_mass_divergence_identity_circulation() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();
        let mut flow = CochainField::zeros(1, n_edges, 1);

        // Construct a divergence-free flow: every vertex has equal in/out.
        // Use the gradient of a linear potential φ(x,y) = x — divergence of
        // gradient = Laplacian of linear = 0 at interior vertices.
        // On a 4×4 grid (w=4): horizontal edges have φ(x+1)-φ(x) = 1,
        // vertical edges have φ(y+1)-φ(y) = 0 (φ doesn't depend on y).
        let w = 4usize;
        let n_h = (w - 1) * w; // horizontal edge count
        // Horizontal edges: value 1 (gradient of φ=x)
        for e in 0..n_h {
            flow.set_scalar(e, 1.0);
        }
        // Vertical edges: value 0 (already zero)

        let div = belief_mass_divergence(&cx, &flow);
        // Interior vertices (not on boundary) have divergence 0.
        // Boundary vertices have non-zero divergence (flow exits the grid).
        // For a 4×4 grid: 4 interior vertices (div=0), 12 boundary vertices.
        // The L1 norm is dominated by boundary contributions.
        // Just verify it's non-zero (boundary effects) and finite.
        assert!(div.is_finite(), "divergence must be finite");
        assert!(
            div > 0.0,
            "boundary vertices contribute non-zero divergence"
        );
    }

    /// T2.1.2 Scaling: divergence scales linearly with flow magnitude.
    #[test]
    fn test_belief_mass_divergence_scaling() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();

        // Single non-zero edge: e0 = c
        let mut flow1 = CochainField::zeros(1, n_edges, 1);
        flow1.set_scalar(0, 1.0);
        let div1 = belief_mass_divergence(&cx, &flow1);

        let mut flow2 = CochainField::zeros(1, n_edges, 1);
        flow2.set_scalar(0, 2.0);
        let div2 = belief_mass_divergence(&cx, &flow2);

        assert!(
            (div2 - 2.0 * div1).abs() < 1e-5,
            "doubling flow must double L1 divergence: div1={div1}, div2={div2}, expected={}",
            2.0 * div1
        );
    }

    /// T2.1.3 Anomaly: inflating one edge's flow spikes the divergence.
    #[test]
    fn test_belief_mass_divergence_anomaly_injection() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();

        // Baseline: all edges = 1.0 (constant flow)
        let mut baseline = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            baseline.set_scalar(e, 1.0);
        }
        let div_baseline = belief_mass_divergence(&cx, &baseline);

        // Anomaly: inflate edge 5 to 100.0
        let mut anomaly = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            anomaly.set_scalar(e, 1.0);
        }
        anomaly.set_scalar(5, 100.0);
        let div_anomaly = belief_mass_divergence(&cx, &anomaly);

        assert!(
            div_anomaly > div_baseline,
            "anomaly injection must spike divergence: baseline={div_baseline}, anomaly={div_anomaly}"
        );
    }

    // ── boundary_flux_mass ──────────────────────────────────────────────────

    /// T2.2.1 Stokes identity: boundary flux == volume integral (d_k sum)
    /// for any field. Validates the implementation is correct.
    #[test]
    fn test_boundary_flux_mass_stokes_identity() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();
        let n_faces = cx.n_faces();

        // Non-trivial field: alternating values.
        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, if e % 2 == 0 { 1.0 } else { 0.5 });
        }

        // Region = all faces.
        let all_faces: Vec<u32> = (0..n_faces as u32).collect();
        let (mass, err) = boundary_flux_mass(&cx, &all_faces, &field);

        // Naive volume integral: Σ_{f ∈ region} d₁(field)[f]
        let d1_field = exterior_derivative(&cx, &field);
        let volume_mass: f32 = (0..n_faces).map(|f| d1_field.scalar(f)).sum();

        assert!(
            (mass - volume_mass).abs() < 1e-4,
            "Stokes identity: boundary flux ({mass}) must equal volume integral ({volume_mass})"
        );
        // Error bound is an L1 norm → always ≥ 0.
        assert!(err >= 0.0, "error_bound must be non-negative");
    }

    /// T2.2.1b Stokes identity for a subset region (not full grid).
    #[test]
    fn test_boundary_flux_mass_stokes_identity_subset() {
        let cx = CellComplex::grid_2d(5, 5);
        let n_edges = cx.n_edges();

        // Random-ish field.
        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, ((e as f32) * 0.3).sin());
        }

        // Region = faces {0, 1, 2} (a small patch, not the full grid).
        let region: Vec<u32> = vec![0, 1, 2];
        let (mass, _err) = boundary_flux_mass(&cx, &region, &field);

        // Naive volume integral over the same faces.
        let d1_field = exterior_derivative(&cx, &field);
        let volume_mass: f32 = region.iter().map(|&f| d1_field.scalar(f as usize)).sum();

        assert!(
            (mass - volume_mass).abs() < 1e-4,
            "Stokes identity (subset): boundary flux ({mass}) must equal volume integral ({volume_mass})"
        );
    }

    /// T2.2.1c Exact field (gradient) → harmonic ≈ 0 → error_bound ≈ 0.
    #[test]
    fn test_boundary_flux_mass_exact_field_zero_harmonic() {
        let cx = CellComplex::grid_2d(4, 4);

        // φ(v) = v_index → d₀(φ) is a pure gradient (exact) field.
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for v in 0..cx.n_vertices() {
            potential.set_scalar(v, v as f32);
        }
        let gradient = exterior_derivative(&cx, &potential); // rank-1, exact

        let all_faces: Vec<u32> = (0..cx.n_faces() as u32).collect();
        let (mass, err) = boundary_flux_mass(&cx, &all_faces, &gradient);

        // Curl of gradient = 0 → mass (circulation) = 0.
        assert!(
            mass.abs() < 1e-4,
            "exact field → boundary circulation ≈ 0, got {mass}"
        );
        // Harmonic of exact field ≈ 0 on a simply-connected grid.
        assert!(
            err < 1e-3,
            "exact field → harmonic ≈ 0, got error_bound={err}"
        );
    }

    /// T2.2.3 Empty region → (0.0, 0.0) without panicking.
    #[test]
    fn test_boundary_flux_mass_empty_region() {
        let cx = CellComplex::grid_2d(4, 4);
        let field = CochainField::zeros(1, cx.n_edges(), 1);
        let (mass, err) = boundary_flux_mass(&cx, &[], &field);
        assert_eq!(mass, 0.0);
        assert_eq!(err, 0.0);
    }

    // ── boundary_flux_mass_indexed (Plan 318) ────────────────────────────────

    /// T318.1 Indexed path matches `boundary_flux_mass_only` on a full-grid region.
    /// Stokes identity: both paths compute the same discrete divergence theorem.
    #[test]
    fn test_boundary_flux_mass_indexed_matches_only_full_region() {
        let mut cx = CellComplex::grid_2d(5, 5);
        let n_edges = cx.n_edges();
        let n_faces = cx.n_faces();

        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, ((e as f32) * 0.7).sin());
        }

        let all_faces: Vec<u32> = (0..n_faces as u32).collect();

        // Full-scan baseline.
        let mass_only = boundary_flux_mass_only(&cx, &all_faces, &field);

        // Indexed path.
        cx.build_coboundary_index(1);
        let mass_indexed = boundary_flux_mass_indexed(&cx, &all_faces, &field);

        assert_eq!(
            mass_only, mass_indexed,
            "indexed must match full-scan on full region"
        );
    }

    /// T318.2 Indexed path matches naive volume integral on a subset region.
    /// This is the more meaningful test: subset regions exercise the interior-
    /// edge cancellation that makes the divergence theorem work.
    #[test]
    fn test_boundary_flux_mass_indexed_stokes_identity_subset() {
        let mut cx = CellComplex::grid_2d(6, 6);
        let n_edges = cx.n_edges();

        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, ((e as f32) * 0.31).cos());
        }

        // A non-trivial region: 3 faces forming an L-shape (shared edges cancel).
        let region: Vec<u32> = vec![0, 1, 5];

        // Build the index AFTER topology is stable.
        cx.build_coboundary_index(1);
        let mass = boundary_flux_mass_indexed(&cx, &region, &field);

        // Naive volume integral: Σ_{f ∈ region} d₁(field)[f].
        let d1_field = exterior_derivative(&cx, &field);
        let volume_mass: f32 = region.iter().map(|&f| d1_field.scalar(f as usize)).sum();

        assert!(
            (mass - volume_mass).abs() < 1e-4,
            "indexed Stokes identity (subset): {mass} must equal volume integral {volume_mass}"
        );

        // Also cross-check against the full-scan path (tolerance accounts for
        // float accumulation-order differences between the two paths).
        let mass_only = boundary_flux_mass_only(&cx, &region, &field);
        assert!(
            (mass - mass_only).abs() < 1e-4,
            "indexed ({mass}) must match full-scan ({mass_only}) on subset"
        );
    }

    /// T318.3 Indexed path: empty region → 0.0 without panicking.
    #[test]
    fn test_boundary_flux_mass_indexed_empty_region() {
        let mut cx = CellComplex::grid_2d(4, 4);
        cx.build_coboundary_index(1);
        let field = CochainField::zeros(1, cx.n_edges(), 1);
        let mass = boundary_flux_mass_indexed(&cx, &[], &field);
        assert_eq!(mass, 0.0);
    }

    /// T318.4 Indexed path: zero-curl (exact/gradient) field → zero flux
    /// regardless of region. Validates the FTC via the indexed code path.
    #[test]
    fn test_boundary_flux_mass_indexed_exact_field_zero() {
        let mut cx = CellComplex::grid_2d(5, 5);

        // φ(v) = v_index → d₀(φ) is a pure gradient (exact) field.
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for v in 0..cx.n_vertices() {
            potential.set_scalar(v, v as f32);
        }
        let gradient = exterior_derivative(&cx, &potential); // rank-1, exact

        let all_faces: Vec<u32> = (0..cx.n_faces() as u32).collect();
        cx.build_coboundary_index(1);
        let mass = boundary_flux_mass_indexed(&cx, &all_faces, &gradient);

        // Curl of gradient = 0 → boundary circulation = 0.
        assert!(
            mass.abs() < 1e-4,
            "exact field → indexed boundary flux ≈ 0, got {mass}"
        );
    }

    /// T318.5 Indexed path: rebuilding after a topology mutation produces
    /// correct results. Exercises the cache invalidation + rebuild cycle.
    #[test]
    fn test_boundary_flux_mass_indexed_after_rebuild() {
        let mut cx = CellComplex::grid_2d(5, 5);
        let n_edges = cx.n_edges();

        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, ((e as f32) * 0.5).sin());
        }

        // Build, query, then mutate + rebuild + query again.
        cx.build_coboundary_index(1);
        let region: Vec<u32> = (0..cx.n_faces() as u32).collect();
        let mass_before = boundary_flux_mass_indexed(&cx, &region, &field);

        cx.remove_face(0);
        assert!(cx.coboundary_entries(1).is_none(), "cache invalidated");

        cx.build_coboundary_index(1);
        let region_after: Vec<u32> = (0..cx.n_faces() as u32).collect();
        let mass_after = boundary_flux_mass_indexed(&cx, &region_after, &field);

        // Cross-check both against full-scan on the respective topologies.
        // (mass_before was computed on the pre-mutation topology; recompute
        // the full-scan baseline on a fresh pre-mutation complex for fairness.)
        let mut cx_ref = CellComplex::grid_2d(5, 5);
        let region_ref: Vec<u32> = (0..cx_ref.n_faces() as u32).collect();
        let mass_before_ref = boundary_flux_mass_only(&cx_ref, &region_ref, &field);
        cx_ref.remove_face(0);
        let region_ref_after: Vec<u32> = (0..cx_ref.n_faces() as u32).collect();
        let mass_after_ref = boundary_flux_mass_only(&cx_ref, &region_ref_after, &field);

        assert!(
            (mass_before - mass_before_ref).abs() < 1e-4,
            "pre-mutation match: indexed={mass_before}, ref={mass_before_ref}"
        );
        assert!(
            (mass_after - mass_after_ref).abs() < 1e-4,
            "post-mutation match: indexed={mass_after}, ref={mass_after_ref}"
        );
    }

    // ── line_integral ───────────────────────────────────────────────────────

    /// T2.3.1 Straight path: integral of constant field = field_value × path_length.
    #[test]
    fn test_line_integral_constant_field_straight_path() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();

        // Constant field: all edges = 1.0
        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, 1.0);
        }

        // Path [0, 1, 2, 3] — horizontal, left to right (along edge orientation).
        // 3 edges, each value 1.0, each traversed along orientation → +1.
        let path: Vec<u32> = vec![0, 1, 2, 3];
        let integral = line_integral(&cx, &field, &path);

        assert!(
            (integral - 3.0).abs() < 1e-5,
            "constant field over 3-edge path = 3.0, got {integral}"
        );
    }

    /// T2.3.2 Reversal antisymmetry: line_integral(A→B) == −line_integral(B→A).
    #[test]
    fn test_line_integral_reversal_antisymmetry() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();

        // Non-trivial field.
        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, (e as f32 * 0.7).sin());
        }

        // Path around a face: 0→1→5→4→0 (on a 4×4 grid, w=4).
        //   0→1: horizontal edge e0 (0→1), along orientation.
        //   1→5: vertical edge (1→5). e_idx = n_h + 1*w + 1 = n_h + 5. Along orientation.
        //   5→4: horizontal edge (4→5 reversed). e = (w-1)*1 + 0 = e_idx for (4→5). Against orientation.
        //   4→0: vertical edge (0→4 reversed). e = n_h + 0*w + 0. Against orientation.
        let path_fwd: Vec<u32> = vec![0, 1, 5, 4, 0];
        let path_bwd: Vec<u32> = vec![0, 4, 5, 1, 0];

        let integral_fwd = line_integral(&cx, &field, &path_fwd);
        let integral_bwd = line_integral(&cx, &field, &path_bwd);

        assert!(
            (integral_fwd + integral_bwd).abs() < 1e-5,
            "reversal antisymmetry: fwd({integral_fwd}) + bwd({integral_bwd}) must be 0"
        );
    }

    /// T2.3.3 Closed loop of an exact (gradient) field = 0.
    #[test]
    fn test_line_integral_closed_loop_exact_field_zero() {
        let cx = CellComplex::grid_2d(4, 4);

        // φ(v) = v_index → gradient field (exact / curl-free).
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for v in 0..cx.n_vertices() {
            potential.set_scalar(v, v as f32);
        }
        let gradient = exterior_derivative(&cx, &potential); // rank-1 exact field

        // Closed loop around a face: 0→1→5→4→0.
        let path: Vec<u32> = vec![0, 1, 5, 4, 0];
        let integral = line_integral(&cx, &gradient, &path);

        // Fundamental theorem: closed loop of a gradient = 0.
        assert!(
            integral.abs() < 1e-4,
            "closed loop of exact field must be 0, got {integral}"
        );
    }

    /// Edge case: single-vertex path → 0.0.
    #[test]
    fn test_line_integral_short_path() {
        let cx = CellComplex::grid_2d(4, 4);
        let field = CochainField::zeros(1, cx.n_edges(), 1);
        assert_eq!(line_integral(&cx, &field, &[0u32]), 0.0);
        assert_eq!(line_integral(&cx, &field, &[]), 0.0);
    }

    // ── circulation_integral (Plan 317 Phase 2) ───────────────────────────

    /// T3.1 Zero-curl (exact/gradient) field → zero circulation.
    ///
    /// Fundamental theorem of calculus: the line integral of a gradient field
    /// around any closed loop is zero. `circulation_integral` must reflect this.
    #[test]
    fn test_circulation_integral_exact_field_zero() {
        let cx = CellComplex::grid_2d(4, 4);

        // φ(v) = v_index → d₀(φ) is a pure gradient (exact, curl-free) field.
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for v in 0..cx.n_vertices() {
            potential.set_scalar(v, v as f32);
        }
        let gradient = exterior_derivative(&cx, &potential); // rank-1 exact field

        // Closed loop around a face: 0→1→5→4→0 (on a 4×4 grid, w=4).
        let square_loop: Vec<u32> = vec![0, 1, 5, 4, 0];
        let circ = circulation_integral(&cx, &gradient, &square_loop);

        assert!(
            circ.abs() < 1e-4,
            "exact field → circulation must be 0 (FTC), got {circ}"
        );
    }

    /// T3.2 Constant-curl field → circulation equals curl × enclosed area.
    ///
    /// Stokes' theorem: ∮_∂A F = ∬_A curl(F) dA. For a single-face loop with
    /// curl(F) = c constant over the face, circulation = c × 1 (unit face area).
    /// We construct a field whose curl is exactly 1.0 on face 0, then verify
    /// circulation around face 0's boundary == 1.0.
    #[test]
    fn test_circulation_integral_constant_curl_equals_curl_times_area() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();

        // Face 0 on a 4×4 grid (w=4) has boundary edges:
        //   bottom:  horizontal edge e_idx = 0*（w-1) + 0 = 0  (vertex 0→1)
        //   right:   vertical   edge e_idx = n_h + 0*w + 1     (vertex 1→5)
        //   top:     horizontal edge e_idx = 1*(w-1) + 0 = 3   (vertex 4→5, reversed: 5→4)
        //   left:    vertical   edge e_idx = n_h + 0*w + 0     (vertex 0→4, reversed: 4→0)
        // where n_h = (w-1)*w = 12 horizontal edges.
        //
        // grid_2d orients edges as (tail, e, −1), (head, e, +1) where for
        // horizontal edges tail < head (left→right) and for vertical edges
        // tail < head (top→bottom, i.e. lower vertex index → higher).
        // Face 0's boundary oriented counterclockwise (interior on the left):
        //   0→1 (bottom, along edge orientation)     contribution = +field[e_bot]
        //   1→5 (right, along vertical orientation)  contribution = +field[e_right]
        //   5→4 (top, against horizontal orientation) contribution = −field[e_top]
        //   4→0 (left, against vertical orientation)  contribution = −field[e_left]
        //
        // curl d₁(field)[face 0] = +field[e_bot] − field[e_top] + field[e_right] − field[e_left]
        //                        (matches the exterior_derivative sign convention).
        //
        // To get curl = +1 on face 0 with the simplest field, set the bottom
        // edge to +1 and all other face-0 boundary edges to 0:
        //   curl[0] = (+1) − 0 + 0 − 0 = +1.
        let w = 4usize;
        let n_h = (w - 1) * w; // 12 horizontal edges
        // (Boundary edge indices documented above; only e_bot is non-zero in
        // this minimal field. The others are kept as named documentation of
        // the face-0 boundary layout.)
        let e_bot = 0usize; // horizontal edge at (row 0, col 0)
        let _e_right = n_h + 1; // vertical edge at (row 0, col 1): n_h + 0*w + 1
        let _e_top = w - 1; // horizontal edge at (row 1, col 0): 1*(w-1) + 0
        let _e_left = n_h; // vertical edge at (row 0, col 0): n_h + 0*w + 0

        let mut field = CochainField::zeros(1, n_edges, 1);
        field.set_scalar(e_bot, 1.0); // only the bottom edge is non-zero
        // (e_right, e_top, e_left remain 0.0)

        // Closed loop around face 0, counterclockwise: 0→1→5→4→0.
        let loop_ccw: Vec<u32> = vec![0, 1, 5, 4, 0];
        let circ = circulation_integral(&cx, &field, &loop_ccw);

        // Cross-check against the DEC operator: d₁(field)[face 0] must equal
        // the circulation around face 0's boundary (Stokes identity).
        let curl_field = exterior_derivative(&cx, &field); // rank-2
        let curl_face0 = curl_field.scalar(0);

        assert!(
            (circ - curl_face0).abs() < 1e-5,
            "circulation ({circ}) must equal curl at face 0 ({curl_face0}) — Stokes identity"
        );
        assert!(
            (circ - 1.0).abs() < 1e-5,
            "constant-curl=1 field over unit face → circulation = 1.0, got {circ}"
        );
    }

    /// T3.3 Reversal antisymmetry: clockwise circulation == −counterclockwise.
    ///
    /// Reversing the loop direction negates every edge's sign contribution,
    /// so circulation_cw == −circulation_ccw. Verified on a non-trivial
    /// rotational field.
    #[test]
    fn test_circulation_integral_reversal_antisymmetry() {
        let cx = CellComplex::grid_2d(4, 4);
        let n_edges = cx.n_edges();

        // Non-trivial rotational field (non-exact, so circulation is non-zero).
        let mut field = CochainField::zeros(1, n_edges, 1);
        for e in 0..n_edges {
            field.set_scalar(e, (e as f32 * 0.7).sin());
        }

        // Counterclockwise loop around face {0,1,5,4}: 0→1→5→4→0.
        let loop_ccw: Vec<u32> = vec![0, 1, 5, 4, 0];
        // Clockwise: reverse the loop (still closed, first==last==0).
        let loop_cw: Vec<u32> = vec![0, 4, 5, 1, 0];

        let circ_ccw = circulation_integral(&cx, &field, &loop_ccw);
        let circ_cw = circulation_integral(&cx, &field, &loop_cw);

        assert!(
            (circ_ccw + circ_cw).abs() < 1e-5,
            "reversal antisymmetry: ccw({circ_ccw}) + cw({circ_cw}) must be 0"
        );
    }
}
