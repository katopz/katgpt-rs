//! Discrete Exterior Calculus (DEC) operators: dₖ, δₖ, Δₖ, Hodge star.
//!
//! Based on "Topological Neural Operators" (arXiv:2606.09806).
//!
//! - `dₖ` (exterior derivative): Cₖ → Cₖ₊₁ — gradient/curl/divergence type
//! - `δₖ` (codifferential): Cₖ → Cₖ₋₁ — metric adjoint of d
//! - `Δₖ` (Hodge Laplacian): Cₖ → Cₖ — Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ
//!
//! Fundamental identity: `dₖ₊₁ ∘ dₖ = 0` (curl(grad)=0, div(curl)=0).

use super::types::{CellComplex, CochainField, MAX_RANK};

// ---------------------------------------------------------------------------
// Hodge Star Mₖ (T10)
// ---------------------------------------------------------------------------

/// Compute the Hodge star Mₖ (mass/metric matrix) scaling factor.
///
/// For uniform grids, returns identity (each cell has equal volume/area).
/// The actual Hodge star is a diagonal matrix; on uniform grids every
/// diagonal entry is the same, so we return that single scalar.
///
/// TODO: Non-uniform grids need actual metric tensor — see Plan 251 T10.
pub fn hodge_star(_cx: &CellComplex, _rank: u8) -> f32 {
    1.0f32
}

// ---------------------------------------------------------------------------
// Exterior Derivative dₖ = Bₖ₊₁ᵀ
// ---------------------------------------------------------------------------

/// Compute the discrete exterior derivative `dₖ: Cₖ → Cₖ₊₁`.
///
/// `dₖ = Bₖ₊₁ᵀ` — the transpose of the boundary matrix.
/// For scalar cochains (dim=1):
///   - d₀ = gradient (vertex → edge: signed endpoint differences)
///   - d₁ = curl (edge → face: signed circulation around faces)
///   - d₂ = divergence (face → volume: signed flux through boundaries)
///
/// For vector cochains (dim>1), applies independently per feature channel.
///
/// # Arguments
/// * `cx` — The cell complex providing boundary matrices
/// * `input` — k-cochain to differentiate (rank must be < MAX_RANK)
///
/// # Returns
/// (k+1)-cochain: the result of applying dₖ.
pub fn exterior_derivative(cx: &CellComplex, input: &CochainField) -> CochainField {
    let k = input.rank;
    assert!(
        k < MAX_RANK,
        "exterior_derivative: rank {k} has no dₖ (max rank is {MAX_RANK})"
    );

    let target_rank = k + 1;
    let n_output = cx.n_cells(target_rank);
    let dim = input.dim;
    let mut output = CochainField::zeros(target_rank, n_output, dim);

    // dₖ = Bₖ₊₁ᵀ means we iterate boundary entries and accumulate:
    // For each entry (row, col, sign) in Bₖ₊₁:
    //   output[col] += sign * input[row]
    let entries = cx.boundary_entries(k);

    // Hoist invariant chunk geometry out of the loop.
    let chunks = dim / 4;
    let remainder = dim % 4;

    // T11: SIMD hint — process inner dim loop with explicit chunking
    // so LLVM can see the unrolled 4-wide pattern for auto-vectorization.
    for &(src_cell, dst_cell, sign) in entries {
        let src_start = src_cell * dim;
        let dst_start = dst_cell * dim;
        let sign_f = sign as f32;

        for c in 0..chunks {
            let off = c * 4;
            output.data[dst_start + off] += sign_f * input.data[src_start + off];
            output.data[dst_start + off + 1] += sign_f * input.data[src_start + off + 1];
            output.data[dst_start + off + 2] += sign_f * input.data[src_start + off + 2];
            output.data[dst_start + off + 3] += sign_f * input.data[src_start + off + 3];
        }
        for d in 0..remainder {
            let off = chunks * 4 + d;
            output.data[dst_start + off] += sign_f * input.data[src_start + off];
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Codifferential δₖ = Mₖ₋₁⁻¹ Bₖ Mₖ
// ---------------------------------------------------------------------------

/// Compute the discrete codifferential `δₖ: Cₖ → Cₖ₋₁`.
///
/// `δₖ = Mₖ₋₁⁻¹ Bₖ Mₖ` — the metric adjoint of `dₖ₋₁`.
/// For uniform grids with identity Hodge stars (Mₖ = I), this simplifies to `Bₖ`.
///
/// For scalar cochains:
///   - δ₁ = divergence-like (edge → vertex: metric-weighted accumulation)
///   - δ₂ = curl-adjoint (face → edge: metric-weighted face-to-edge)
///
/// # Arguments
/// * `cx` — The cell complex providing boundary matrices
/// * `input` — k-cochain (rank must be > 0)
///
/// # Returns
/// (k-1)-cochain: the result of applying δₖ.
pub fn codifferential(cx: &CellComplex, input: &CochainField) -> CochainField {
    let k = input.rank;
    assert!(
        k > 0,
        "codifferential: rank {k} has no δₖ (rank must be > 0)"
    );

    let target_rank = k - 1;
    let n_output = cx.n_cells(target_rank);
    let dim = input.dim;
    let mut output = CochainField::zeros(target_rank, n_output, dim);

    // With identity Hodge stars (uniform grid), δₖ = Bₖ (boundary matrix applied directly).
    // For each entry (row, col, sign) in Bₖ:
    //   output[row] += sign * input[col]
    // (Note: Bₖ maps (k)-cells to (k-1)-cells, so we iterate its entries directly)
    let entries = cx.boundary_entries(target_rank);

    // Hoist invariant chunk geometry; branch-free sign via f32 multiply (matches exterior_derivative).
    let chunks = dim / 4;
    let remainder = dim % 4;

    for &(dst_cell, src_cell, sign) in entries {
        let src_start = src_cell * dim;
        let dst_start = dst_cell * dim;
        let sign_f = sign as f32;

        for c in 0..chunks {
            let off = c * 4;
            output.data[dst_start + off] += sign_f * input.data[src_start + off];
            output.data[dst_start + off + 1] += sign_f * input.data[src_start + off + 1];
            output.data[dst_start + off + 2] += sign_f * input.data[src_start + off + 2];
            output.data[dst_start + off + 3] += sign_f * input.data[src_start + off + 3];
        }
        for d in 0..remainder {
            let off = chunks * 4 + d;
            output.data[dst_start + off] += sign_f * input.data[src_start + off];
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Hodge Laplacian Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ
// ---------------------------------------------------------------------------

/// Compute the Hodge Laplacian `Δₖ: Cₖ → Cₖ`.
///
/// `Δₖ = Δ↑ₖ + Δ↓ₖ` where:
/// - `Δ↑ₖ = δₖ₊₁ ∘ dₖ` (upper: through (k+1)-cells, curl-like coupling)
/// - `Δ↓ₖ = dₖ₋₁ ∘ δₖ` (lower: through (k-1)-cells, divergence-like coupling)
///
/// For rank 0: Δ₀ = δ₁d₀ = standard graph Laplacian.
/// For rank 1: Δ₁ = δ₂d₁ + d₀δ₁ (edge coupling through faces AND vertices).
///
/// # Arguments
/// * `cx` — The cell complex
/// * `input` — k-cochain
///
/// # Returns
/// k-cochain: the result of applying Δₖ.
pub fn hodge_laplacian(cx: &CellComplex, input: &CochainField) -> CochainField {
    let k = input.rank;
    let n = input.n_cells();
    let dim = input.dim;

    // Rank-0 fast path: Δ₀ = δ₁d₀ = graph Laplacian.
    // Single-pass computation avoids 2 intermediate cochain allocations.
    if k == 0 && cx.n_edges() > 0 {
        let mut scratch = vec![0.0f32; cx.n_edges() * dim];
        return graph_laplacian(cx, input, &mut scratch);
    }

    let mut output = CochainField::zeros(k, n, dim);

    // Upper channel: Δ↑ₖ = δₖ₊₁ ∘ dₖ
    if k < MAX_RANK && cx.n_cells(k + 1) > 0 {
        let dk_input = exterior_derivative(cx, input);
        if dk_input.n_cells() > 0 {
            let delta_up = codifferential(cx, &dk_input);
            for (o, u) in output.data.iter_mut().zip(delta_up.data.iter()) {
                *o += u;
            }
        }
    }

    // Lower channel: Δ↓ₖ = dₖ₋₁ ∘ δₖ
    if k > 0 {
        let delta_k = codifferential(cx, input);
        if delta_k.n_cells() > 0 {
            let d_lower = exterior_derivative(cx, &delta_k);
            for (o, l) in output.data.iter_mut().zip(d_lower.data.iter()) {
                *o += l;
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Composite: Full Hodge Laplacian (optimized single-pass)
// ---------------------------------------------------------------------------

/// Compute the Hodge Laplacian Δ₀ = δ₁d₀ (graph Laplacian) for rank-0 cochains.
///
/// Optimized single-pass implementation avoiding intermediate allocations.
/// For a uniform grid, this is the standard 5-point stencil Laplacian (2D)
/// or 7-point stencil (3D).
///
/// # Arguments
/// * `cx` — The cell complex (must be 2D grid)
/// * `potential` — 0-cochain (vertex values)
/// * `scratch` — Pre-allocated scratch buffer of length `cx.n_edges() * dim`
///
/// # Returns
/// 0-cochain: the graph Laplacian applied to the input.
pub fn graph_laplacian(
    cx: &CellComplex,
    potential: &CochainField,
    _scratch: &mut [f32],
) -> CochainField {
    debug_assert_eq!(potential.rank, 0, "graph_laplacian requires rank-0 cochain");
    let dim = potential.dim;

    // Single-pass graph Laplacian: boundary entries are stored as adjacent pairs
    // (v_tail, e, -1), (v_head, e, +1) for each edge. Process each pair to compute
    // Δ₀[v] = degree(v)*potential[v] - Σ potential[neighbor] directly.
    let entries = cx.boundary_entries(0);
    let n_vertices = cx.n_vertices();
    let mut output = CochainField::zeros(0, n_vertices, dim);

    // Entries come in pairs for each edge: (v_tail, e, -1), (v_head, e, +1).
    // Hoist invariant chunk geometry out of the loop.
    let chunks = dim / 4;
    let remainder = dim % 4;

    for pair in entries.chunks_exact(2) {
        let (v_tail, _e, _sign_t) = pair[0];
        let (v_head, _e, _sign_h) = pair[1];
        let tail_start = v_tail * dim;
        let head_start = v_head * dim;

        for c in 0..chunks {
            let off = c * 4;
            let diff0 = potential.data[tail_start + off] - potential.data[head_start + off];
            let diff1 = potential.data[tail_start + off + 1] - potential.data[head_start + off + 1];
            let diff2 = potential.data[tail_start + off + 2] - potential.data[head_start + off + 2];
            let diff3 = potential.data[tail_start + off + 3] - potential.data[head_start + off + 3];
            output.data[tail_start + off] += diff0;
            output.data[head_start + off] -= diff0;
            output.data[tail_start + off + 1] += diff1;
            output.data[head_start + off + 1] -= diff1;
            output.data[tail_start + off + 2] += diff2;
            output.data[head_start + off + 2] -= diff2;
            output.data[tail_start + off + 3] += diff3;
            output.data[head_start + off + 3] -= diff3;
        }
        for d in 0..remainder {
            let off = chunks * 4 + d;
            let diff = potential.data[tail_start + off] - potential.data[head_start + off];
            output.data[tail_start + off] += diff;
            output.data[head_start + off] -= diff;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_of_constant_is_zero() {
        // d₀(constant) = 0 — gradient of a constant function vanishes
        let cx = CellComplex::grid_2d(4, 4);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            potential.set_scalar(i, 5.0);
        }
        let grad = exterior_derivative(&cx, &potential);
        assert_eq!(grad.rank, 1);
        assert_eq!(grad.n_cells(), cx.n_edges());
        for i in 0..grad.n_cells() {
            assert!(
                grad.scalar(i).abs() < 1e-6,
                "gradient of constant should be 0, got {} at edge {}",
                grad.scalar(i),
                i
            );
        }
    }

    #[test]
    fn curl_of_gradient_is_zero() {
        // d₁(d₀(f)) = 0 — curl of gradient vanishes (boundary of boundary is zero)
        let cx = CellComplex::grid_2d(4, 4);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for y in 0..4u16 {
            for x in 0..4u16 {
                let idx = (y as usize) * 4 + (x as usize);
                potential.set_scalar(idx, (x + y * 2) as f32);
            }
        }
        let grad = exterior_derivative(&cx, &potential);
        let curl = exterior_derivative(&cx, &grad);
        assert_eq!(curl.rank, 2);
        assert_eq!(curl.n_cells(), cx.n_faces());
        for i in 0..curl.n_cells() {
            assert!(
                curl.scalar(i).abs() < 1e-6,
                "curl(grad) should be 0, got {} at face {}",
                curl.scalar(i),
                i
            );
        }
    }

    #[test]
    fn graph_laplacian_linear_function() {
        // Δ₀(linear) = 0 — Laplacian of a linear function vanishes
        let cx = CellComplex::grid_2d(4, 4);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for y in 0..4usize {
            for x in 0..4usize {
                let idx = y * 4 + x;
                potential.set_scalar(idx, (x + y) as f32);
            }
        }
        let mut scratch = vec![0.0f32; cx.n_edges()];
        let lap = graph_laplacian(&cx, &potential, &mut scratch);

        // Interior vertices should have zero Laplacian
        for y in 1..3usize {
            for x in 1..3usize {
                let idx = y * 4 + x;
                assert!(
                    lap.scalar(idx).abs() < 1e-6,
                    "Laplacian of linear at interior ({x},{y}) should be 0, got {}",
                    lap.scalar(idx)
                );
            }
        }
    }

    #[test]
    fn gradient_direction_correct() {
        // d₀ of potential V(x,y) = x should give:
        //   horizontal edges: gradient = +1
        //   vertical edges: gradient = 0
        let cx = CellComplex::grid_2d(3, 3);
        let mut potential = CochainField::zeros(0, 9, 1);
        for y in 0..3usize {
            for x in 0..3usize {
                potential.set_scalar(y * 3 + x, x as f32);
            }
        }
        let grad = exterior_derivative(&cx, &potential);

        // Horizontal edges: (w-1)*h = 2*3 = 6 edges, each should have gradient = 1
        let n_h_edges = 2 * 3;
        for e in 0..n_h_edges {
            assert!(
                (grad.scalar(e) - 1.0).abs() < 1e-6,
                "horizontal edge {e} gradient should be 1.0, got {}",
                grad.scalar(e)
            );
        }
        // Vertical edges: w*(h-1) = 3*2 = 6 edges, each should have gradient = 0
        for e in n_h_edges..(n_h_edges + 3 * 2) {
            assert!(
                grad.scalar(e).abs() < 1e-6,
                "vertical edge {e} gradient should be 0.0, got {}",
                grad.scalar(e)
            );
        }
    }

    #[test]
    fn divergence_of_curl_is_zero() {
        // δ₂(d₁(edge_field)) should be zero for the graph Laplacian identity
        // This is equivalent to: the image of d₁ is in the kernel of δ₂ (div curl = 0)
        let cx = CellComplex::grid_2d(4, 4);

        // Create a vertex potential, compute gradient, then curl
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            potential.set_scalar(i, (i as f32 * 0.7).sin());
        }
        let grad = exterior_derivative(&cx, &potential);
        let curl = exterior_derivative(&cx, &grad);

        // div(curl) = codifferential of the face field
        // This requires rank ≥ 2 → rank 1, so we need δ₂
        if curl.rank == 2 && cx.n_faces() > 0 {
            let div_curl = codifferential(&cx, &curl);
            // This should be zero on the coexact component
            // For the full test, verify that codifferential of curl is small
            let max_val = div_curl
                .data
                .iter()
                .map(|&v: &f32| v.abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_val < 1e-4,
                "div(curl(grad(f))) should be ~0, got max {}",
                max_val
            );
        }
    }
}
