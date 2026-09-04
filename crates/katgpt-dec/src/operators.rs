//! Discrete Exterior Calculus (DEC) operators: dₖ, δₖ, Δₖ, Hodge star.
//!
//! Based on "Topological Neural Operators" (arXiv:2606.09806).
//!
//! - `dₖ` (exterior derivative): Cₖ → Cₖ₊₁ — gradient/curl/divergence type
//! - `δₖ` (codifferential): Cₖ → Cₖ₋₁ — metric adjoint of d
//! - `Δₖ` (Hodge Laplacian): Cₖ → Cₖ — Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ
//!
//! Fundamental identity: `dₖ₊₁ ∘ dₖ = 0` (curl(grad)=0, div(curl)=0).

use crate::types::{CellComplex, CochainField, GridDims, MAX_RANK};

// ---------------------------------------------------------------------------
// Hodge Star Mₖ (T10)
// ---------------------------------------------------------------------------

/// Compute the Hodge star Mₖ (mass/metric matrix) scaling factor.
///
/// For uniform grids, returns identity (each cell has equal volume/area).
/// The actual Hodge star is a diagonal matrix; on uniform grids every
/// diagonal entry is the same, so we return that single scalar.
///
/// TODO: Non-uniform grids need actual metric tensor. Plan 251 T10 shipped
/// this identity placeholder as the deliverable for uniform grids; non-uniform
/// support is standalone future work — file a plan before implementing.
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
    exterior_derivative_into(cx, input, &mut output);
    output
}

/// Zero-alloc `exterior_derivative` writing into pre-allocated `output`.
///
/// `output` must have `rank == input.rank + 1`, `dim == input.dim`, and
/// `data.len() >= cx.n_cells(input.rank + 1) * dim`. Its data is zero-filled then accumulated.
#[inline]
pub fn exterior_derivative_into(cx: &CellComplex, input: &CochainField, output: &mut CochainField) {
    let k = input.rank;
    let dim = input.dim;
    output.data.fill(0.0);

    // dₖ = Bₖ₊₁ᵀ means we iterate boundary entries and accumulate:
    // For each entry (row, col, sign) in Bₖ₊₁:
    //   output[col] += sign * input[row]
    let entries = cx.boundary_entries(k);

    // T11: slice both rows up front so the two bounds checks hoist out of the
    // inner dim loop entirely. The remaining loop is a plain equal-length
    // `axpy` over slices, which LLVM auto-vectorizes reliably (and it keeps
    // the per-element accumulation order byte-identical to the unrolled form).
    for &(src_cell, dst_cell, sign) in entries {
        let src_start = src_cell * dim;
        let dst_start = dst_cell * dim;
        let sign_f = sign as f32;

        let src_row = &input.data[src_start..src_start + dim];
        let dst_row = &mut output.data[dst_start..dst_start + dim];
        for (o, &i) in dst_row.iter_mut().zip(src_row.iter()) {
            *o += sign_f * i;
        }
    }
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
    codifferential_into(cx, input, &mut output);
    output
}

/// Zero-alloc `codifferential` writing into pre-allocated `output`.
///
/// `output` must have `rank == input.rank - 1`, `dim == input.dim`, and
/// `data.len() >= cx.n_cells(input.rank - 1) * dim`. Its data is zero-filled then accumulated.
#[inline]
pub fn codifferential_into(cx: &CellComplex, input: &CochainField, output: &mut CochainField) {
    let k = input.rank;
    let dim = input.dim;
    output.data.fill(0.0);

    // With identity Hodge stars (uniform grid), δₖ = Bₖ (boundary matrix applied directly).
    // For each entry (row, col, sign) in Bₖ:
    //   output[row] += sign * input[col]
    // (Note: Bₖ maps (k)-cells to (k-1)-cells, so we iterate its entries directly)
    let entries = cx.boundary_entries(k - 1);

    // Slice both rows up front (see `exterior_derivative_into`): the bounds
    // checks hoist out of the inner dim loop and the resulting equal-length
    // `axpy` auto-vectorizes. Branch-free sign via f32 multiply.
    for &(dst_cell, src_cell, sign) in entries {
        let src_start = src_cell * dim;
        let dst_start = dst_cell * dim;
        let sign_f = sign as f32;

        let src_row = &input.data[src_start..src_start + dim];
        let dst_row = &mut output.data[dst_start..dst_start + dim];
        for (o, &i) in dst_row.iter_mut().zip(src_row.iter()) {
            *o += sign_f * i;
        }
    }
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
        return graph_laplacian(cx, input);
    }

    let mut output = CochainField::zeros(k, n, dim);
    // Allocate scratch for the two intermediate ranks (k+1, k-1) and one result accumulator (k).
    let mut scratch_upper = CochainField::zeros(k + 1, cx.n_cells(k + 1), dim);
    let mut scratch_lower =
        CochainField::zeros(k.saturating_sub(1), cx.n_cells(k.saturating_sub(1)), dim);
    let mut scratch_result = CochainField::zeros(k, n, dim);
    hodge_laplacian_into(
        cx,
        input,
        &mut output,
        &mut scratch_upper,
        &mut scratch_lower,
        &mut scratch_result,
    );
    output
}

/// Zero-alloc `hodge_laplacian` writing into pre-allocated `output`.
///
/// Scratch buffers are reused across CG iterations:
/// - `scratch_upper`: rank k+1, capacity `cx.n_cells(k+1) * dim`
/// - `scratch_lower`: rank k-1, capacity `cx.n_cells(k-1) * dim` (unused for rank 0)
/// - `scratch_result`: rank k, capacity `n * dim` (second-stage result accumulator)
///
/// `output.data` is zero-filled then accumulated. Rank-0 delegates to `graph_laplacian_into`.
#[inline]
pub fn hodge_laplacian_into(
    cx: &CellComplex,
    input: &CochainField,
    output: &mut CochainField,
    scratch_upper: &mut CochainField,
    scratch_lower: &mut CochainField,
    scratch_result: &mut CochainField,
) {
    let k = input.rank;

    // Rank-0 fast path: Δ₀ = δ₁d₀ = graph Laplacian.
    if k == 0 && cx.n_edges() > 0 {
        graph_laplacian_into(cx, input, output);
        return;
    }

    output.data.fill(0.0);

    // Upper channel: Δ↑ₖ = δₖ₊₁ ∘ dₖ
    if k < MAX_RANK && cx.n_cells(k + 1) > 0 {
        exterior_derivative_into(cx, input, scratch_upper);
        if scratch_upper.n_cells() > 0 {
            // δₖ₊₁ maps rank k+1 → rank k. Write into scratch_result, accumulate into output.
            codifferential_into(cx, scratch_upper, scratch_result);
            for (o, u) in output.data.iter_mut().zip(scratch_result.data.iter()) {
                *o += u;
            }
        }
    }

    // Lower channel: Δ↓ₖ = dₖ₋₁ ∘ δₖ
    if k > 0 {
        codifferential_into(cx, input, scratch_lower);
        if scratch_lower.n_cells() > 0 {
            // dₖ₋₁ maps rank k-1 → rank k. Write into scratch_result, accumulate into output.
            exterior_derivative_into(cx, scratch_lower, scratch_result);
            for (o, l) in output.data.iter_mut().zip(scratch_result.data.iter()) {
                *o += l;
            }
        }
    }
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
pub fn graph_laplacian(cx: &CellComplex, potential: &CochainField) -> CochainField {
    debug_assert_eq!(potential.rank, 0, "graph_laplacian requires rank-0 cochain");
    let n_vertices = cx.n_vertices();
    let mut output = CochainField::zeros(0, n_vertices, potential.dim);
    graph_laplacian_into(cx, potential, &mut output);
    output
}

/// Zero-alloc `graph_laplacian` writing into pre-allocated `output`.
///
/// `output` must have `rank == 0`, `dim == potential.dim`, and
/// `data.len() >= cx.n_vertices() * dim`. Its data is zero-filled then accumulated,
/// unless `cx` is an unmutated `grid_2d` product — in that case the 5-point-stencil
/// fast path writes every element exactly once (no zero-fill).
#[inline]
pub fn graph_laplacian_into(cx: &CellComplex, potential: &CochainField, output: &mut CochainField) {
    debug_assert_eq!(potential.rank, 0, "graph_laplacian requires rank-0 cochain");
    // Plan 357 G5 fix: regular grids take the cache-friendly stencil fast
    // path (5-point for 2D `grid_2d` products, 7-point for 3D `grid_3d`
    // products). The generic edge-list path is correct but does scattered
    // read-modify-writes on `output` (each vertex touched degree(v) times,
    // each touch on a different cache line for large grids), which is the G5
    // bottleneck. The stencil reads vertices in row-major order and writes
    // each output element exactly once — no zero-fill, no scatter, no
    // read-modify-write store-forwarding stalls.
    //
    // Plan 454 T3: dispatch on the full `GridDims` discriminant so 3D grids
    // route to the 7-point stencil. 2D call-sites are unchanged.
    match cx.grid_dims_full() {
        Some(GridDims::Dim2 { w, h }) => {
            graph_laplacian_grid_into(w, h, potential, output);
        }
        #[cfg(feature = "grid_3d")]
        Some(GridDims::Dim3 { w, h, d }) => {
            graph_laplacian_grid_3d_into(w, h, d, potential, output);
        }
        // When the `grid_3d` feature is off, `GridDims::Dim3` cannot be
        // constructed (the `grid_3d` constructor is gated), so this arm is
        // unreachable. It exists only to keep the match exhaustive across
        // feature configurations.
        #[cfg(not(feature = "grid_3d"))]
        Some(GridDims::Dim3 { .. }) => {
            graph_laplacian_edge_list_into(cx, potential, output);
        }
        None => graph_laplacian_edge_list_into(cx, potential, output),
    }
}

/// Generic edge-list graph Laplacian (the pre-stencil path). Public via
/// [`graph_laplacian_into`] for non-grid complexes; kept separate so the grid
/// fast path has a clean dispatch point. Zero-fills `output` then accumulates
/// one `(+=diff, -=diff)` pair per edge.
#[inline]
fn graph_laplacian_edge_list_into(
    cx: &CellComplex,
    potential: &CochainField,
    output: &mut CochainField,
) {
    debug_assert_eq!(potential.rank, 0, "graph_laplacian requires rank-0 cochain");
    let dim = potential.dim;
    output.data.fill(0.0);

    // Single-pass graph Laplacian: boundary entries are stored as adjacent pairs
    // (v_tail, e, -1), (v_head, e, +1) for each edge. Process each pair to compute
    // Δ₀[v] = degree(v)*potential[v] - Σ potential[neighbor] directly.
    let entries = cx.boundary_entries(0);

    // Entries come in pairs for each edge: (v_tail, e, -1), (v_head, e, +1).
    // Hoist invariant chunk geometry out of the loop.
    let chunks = dim / 4;
    let remainder = dim % 4;

    for pair in entries.as_chunks::<2>().0 {
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
}

/// 5-point-stencil graph Laplacian for a regular `w×h` vertex grid (Plan 357 G5).
///
/// Computes `Δ₀[v] = deg(v)·potential[v] − Σ potential[neighbor]` with
/// deg(v) = 4 (interior), 3 (edge), 2 (corner). Reads vertices in row-major
/// order and writes each `output` element exactly once — no zero-fill, no
/// scattered read-modify-write. The interior loop is branch-free and
/// auto-vectorizes cleanly (4 FMA per element on the unrolled dim-chunks);
/// the boundary is `O(w+h)` and handled with explicit neighbor-count checks.
///
/// Mathematically identical to the edge-list path on the same grid (both
/// realize `δ₁d₀`); the f32 results can differ by ULP-level rounding because
/// the accumulation order differs, which is acceptable for every consumer
/// (the hodge.rs tests use `TOL = 1e-3`; the operators tests check structural
/// properties like `Δ(linear) = 0` which hold exactly under either path).
#[inline]
fn graph_laplacian_grid_into(
    w: usize,
    h: usize,
    potential: &CochainField,
    output: &mut CochainField,
) {
    debug_assert_eq!(potential.rank, 0, "graph_laplacian requires rank-0 cochain");
    let dim = potential.dim;
    let p = potential.data.as_ptr();
    let o = output.data.as_mut_ptr();
    let stride = w * dim;

    // Interior: 4 neighbors each, branch-free. The bulk path for any grid
    // larger than ~5×5; iterates (w-2)·(h-2) vertices.
    if w >= 3 && h >= 3 {
        for y in 1..(h - 1) {
            let row = y * stride;
            let up_row = row - stride;
            let down_row = row + stride;
            for x in 1..(w - 1) {
                let base = row + x * dim;
                let left = base - dim;
                let right = base + dim;
                let up = up_row + x * dim;
                let down = down_row + x * dim;
                // Safety: base, left, right are within [row, row+stride); up/down
                // are within [(y-1)*stride, (y+2)*stride) ⊂ [0, w*h*dim).
                unsafe {
                    for c in 0..dim {
                        let center = *p.add(base + c);
                        *o.add(base + c) = 4.0 * center
                            - *p.add(left + c)
                            - *p.add(right + c)
                            - *p.add(up + c)
                            - *p.add(down + c);
                    }
                }
            }
        }
    }

    // Boundary: top + bottom rows (full width). deg = 2 at corners, 3 on edges.
    for &(y, up_off, down_off, has_up, has_down) in [
        (0usize, 0usize, stride, false, true),
        (h - 1, stride, 0usize, true, false),
    ]
    .iter()
    {
        let row = y * stride;
        let up_row = row.wrapping_sub(up_off);
        let down_row = row + down_off;
        for x in 0..w {
            let base = row + x * dim;
            let has_left = x > 0;
            let has_right = x < w - 1;
            let deg = (has_left as u8 + has_right as u8 + has_up as u8 + has_down as u8) as f32;
            // wrapping_sub/add: offsets are only dereferenced when has_left/has_right
            // is true, so the underflowing values at corners (x==0 or x==w-1) are
            // never read. Raw-pointer arithmetic on out-of-bounds offsets is sound
            // as long as we don't load through them.
            let left = base.wrapping_sub(dim);
            let right = base.wrapping_add(dim);
            let up = up_row + x * dim;
            let down = down_row + x * dim;
            unsafe {
                for c in 0..dim {
                    let center = *p.add(base + c);
                    let mut acc = deg * center;
                    if has_left {
                        acc -= *p.add(left + c);
                    }
                    if has_right {
                        acc -= *p.add(right + c);
                    }
                    if has_up {
                        acc -= *p.add(up + c);
                    }
                    if has_down {
                        acc -= *p.add(down + c);
                    }
                    *o.add(base + c) = acc;
                }
            }
        }
    }

    // Boundary: left + right columns (excluding corners already written above).
    if h >= 3 {
        for &(x, left_off, right_off, has_left, has_right) in [
            (0usize, 0usize, dim, false, true),
            (w - 1, dim, 0usize, true, false),
        ]
        .iter()
        {
            for y in 1..(h - 1) {
                let row = y * stride;
                let base = row + x * dim;
                let left = base.wrapping_sub(left_off);
                let right = base.wrapping_add(right_off);
                let up = row - stride + x * dim;
                let down = row + stride + x * dim;
                unsafe {
                    for c in 0..dim {
                        let center = *p.add(base + c);
                        let mut acc = 3.0 * center;
                        if has_left {
                            acc -= *p.add(left + c);
                        }
                        if has_right {
                            acc -= *p.add(right + c);
                        }
                        acc -= *p.add(up + c);
                        acc -= *p.add(down + c);
                        *o.add(base + c) = acc;
                    }
                }
            }
        }
    }
}

/// 7-point-stencil graph Laplacian for a regular `w×h×d` vertex grid (Plan 454 T3).
///
/// Computes `Δ₀[v] = deg(v)·potential[v] − Σ potential[neighbor]` with
/// deg(v) = 6 (interior), 5 (face), 4 (edge), 3 (corner). Reads vertices in
/// row-major `(z, y, x)` order and writes each `output` element exactly once
/// — no zero-fill, no scattered read-modify-write. The interior loop is
/// branch-free and auto-vectorizes cleanly (6 FMA per element on the unrolled
/// dim-chunks); the boundary is handled with a single unified loop using
/// per-vertex `has_left/right/up/down/front/back` flags (the plan-specified
/// pattern — same correctness, simpler segmentation than 6-faces + 12-edges
/// + 8-corners special-casing).
///
/// Mathematically identical to the edge-list path on the same grid (both
/// realize `δ₁d₀`); the f32 results can differ by ULP-level rounding because
/// the accumulation order differs, which is acceptable for every consumer.
#[cfg(feature = "grid_3d")]
#[inline]
fn graph_laplacian_grid_3d_into(
    w: usize,
    h: usize,
    d: usize,
    potential: &CochainField,
    output: &mut CochainField,
) {
    debug_assert_eq!(potential.rank, 0, "graph_laplacian requires rank-0 cochain");
    let dim = potential.dim;
    let p = potential.data.as_ptr();
    let o = output.data.as_mut_ptr();
    // Stride math (Plan 454 T3): vertex index is `(z * h + y) * w + x`, so in
    // flat data layout (channel-interleaved per vertex) the per-axis strides
    // in units of f32 elements are:
    //   x-stride = dim
    //   y-stride = w * dim
    //   z-stride = w * h * dim
    let xy_plane = w * dim; // one z-slice (one xy-plane of vertices)
    let z_stride = h * xy_plane; // one full z-step = h rows

    // ── Interior: 6 neighbors each, branch-free ─────────────────────────
    // Bulk path for any grid larger than ~4×4×4; iterates (w-2)·(h-2)·(d-2)
    // vertices. deg = 6.
    if w >= 3 && h >= 3 && d >= 3 {
        for z in 1..(d - 1) {
            let plane = z * z_stride;
            let front_plane = plane - z_stride;
            let back_plane = plane + z_stride;
            for y in 1..(h - 1) {
                let row = plane + y * xy_plane;
                let up_row = row - xy_plane;
                let down_row = row + xy_plane;
                for x in 1..(w - 1) {
                    let base = row + x * dim;
                    let left = base - dim;
                    let right = base + dim;
                    let up = up_row + x * dim;
                    let down = down_row + x * dim;
                    let front = front_plane + y * xy_plane + x * dim;
                    let back = back_plane + y * xy_plane + x * dim;
                    // Safety: all offsets are within [0, w*h*d*dim) for the
                    // interior region (1 <= x < w-1, etc.).
                    unsafe {
                        for c in 0..dim {
                            let center = *p.add(base + c);
                            *o.add(base + c) = 6.0 * center
                                - *p.add(left + c)
                                - *p.add(right + c)
                                - *p.add(up + c)
                                - *p.add(down + c)
                                - *p.add(front + c)
                                - *p.add(back + c);
                        }
                    }
                }
            }
        }
    }

    // ── Boundary: unified single loop with per-vertex `has_*` flags ─────
    // Covers all 6 face planes (deg 5), 12 edges (deg 4), and 8 corners
    // (deg 3) in one pass. The `has_*` flags select which neighbor offsets
    // are valid; the corresponding `wrapping_sub`/`wrapping_add` offsets are
    // only dereferenced when their flag is true (same soundness argument as
    // the 2D boundary path — raw-pointer arithmetic on out-of-bounds offsets
    // is sound as long as we don't load through them).
    //
    // To avoid re-touching interior vertices (already written above), this
    // loop only visits vertices where at least one of x/y/z is on the
    // boundary. For grids where w<3 || h<3 || d<3 there is no interior region
    // and every vertex is boundary.
    for z in 0..d {
        let has_front = z > 0;
        let has_back = z < d - 1;
        let plane = z * z_stride;
        let front_plane = plane.wrapping_sub(z_stride);
        let back_plane = plane + z_stride;
        for y in 0..h {
            // Skip pure-interior rows when an interior region exists.
            let y_interior = y >= 1 && y < h - 1 && h >= 3;
            let row = plane + y * xy_plane;
            let up_row = row.wrapping_sub(xy_plane);
            let down_row = row + xy_plane;
            let has_up = y > 0;
            let has_down = y < h - 1;
            for x in 0..w {
                // Skip pure-interior vertices (already handled above) when a
                // full interior region exists.
                let x_interior = x >= 1 && x < w - 1 && w >= 3;
                if x_interior && y_interior && has_front && has_back {
                    continue;
                }
                let has_left = x > 0;
                let has_right = x < w - 1;
                let base = row + x * dim;
                let left = base.wrapping_sub(dim);
                let right = base.wrapping_add(dim);
                let up = up_row + x * dim;
                let down = down_row + x * dim;
                let front = front_plane + y * xy_plane + x * dim;
                let back = back_plane + y * xy_plane + x * dim;
                let deg = (has_left as u8
                    + has_right as u8
                    + has_up as u8
                    + has_down as u8
                    + has_front as u8
                    + has_back as u8) as f32;
                unsafe {
                    for c in 0..dim {
                        let center = *p.add(base + c);
                        let mut acc = deg * center;
                        if has_left {
                            acc -= *p.add(left + c);
                        }
                        if has_right {
                            acc -= *p.add(right + c);
                        }
                        if has_up {
                            acc -= *p.add(up + c);
                        }
                        if has_down {
                            acc -= *p.add(down + c);
                        }
                        if has_front {
                            acc -= *p.add(front + c);
                        }
                        if has_back {
                            acc -= *p.add(back + c);
                        }
                        *o.add(base + c) = acc;
                    }
                }
            }
        }
    }
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
        let lap = graph_laplacian(&cx, &potential);

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
                "div(curl(grad(f))) should be ~0, got max {max_val}"
            );
        }
    }

    // ── Plan 357 G5: grid-stencil fast-path equivalence ───────────────────
    //
    // The grid_dims dispatch in `graph_laplacian_into` swaps the edge-list
    // accumulation path for a 5-point-stencil direct-write path on unmutated
    // `grid_2d` complexes. The two are mathematically identical (both realize
    // δ₁d₀); the only permissible difference is ULP-level f32 rounding from
    // the changed accumulation order. These tests pin that contract.

    /// Compute the edge-list path directly (bypassing the grid dispatch) on
    /// the same grid complex. `graph_laplacian_edge_list_into` reads
    /// `cx.boundary_entries(0)` and ignores `grid_dims`, so passing a
    /// `grid_2d` complex exercises the pre-stencil accumulation path.
    fn edge_list_laplacian(cx: &CellComplex, potential: &CochainField) -> CochainField {
        let mut out = CochainField::zeros(0, cx.n_vertices(), potential.dim);
        graph_laplacian_edge_list_into(cx, potential, &mut out);
        out
    }

    #[test]
    fn graph_laplacian_grid_matches_edge_list_1ch() {
        // Single-channel: the stencil and edge-list paths must agree to within
        // 1 ULP (allow a tiny tolerance for accumulation-order rounding).
        let (w, h) = (8usize, 6usize);
        let cx = CellComplex::grid_2d(w, h);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            potential.set_scalar(i, ((i as f32) * 0.37).sin());
        }
        let lap_grid = graph_laplacian(&cx, &potential);
        let lap_edges = edge_list_laplacian(&cx, &potential);
        let mut max_diff = 0.0f32;
        for i in 0..cx.n_vertices() {
            let d = (lap_grid.scalar(i) - lap_edges.scalar(i)).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-5,
            "grid vs edge-list laplacian diverged by {max_diff:e} (expected < 1e-5)"
        );
    }

    #[test]
    fn graph_laplacian_grid_matches_edge_list_multich() {
        // Multi-channel (dim=16, the G5 workload shape): same contract.
        let (w, h) = (7usize, 5usize);
        let dim = 16usize;
        let cx = CellComplex::grid_2d(w, h);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), dim);
        for cell in 0..cx.n_vertices() {
            for ch in 0..dim {
                let v = ((cell as f32 * 0.11 + ch as f32 * 0.73).sin()) * 2.0;
                potential.data[cell * dim + ch] = v;
            }
        }
        let lap_grid = graph_laplacian(&cx, &potential);
        let lap_edges = edge_list_laplacian(&cx, &potential);
        let len = cx.n_vertices() * dim;
        let mut max_diff = 0.0f32;
        for i in 0..len {
            let d = (lap_grid.data[i] - lap_edges.data[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-4,
            "grid vs edge-list multichannel diverged by {max_diff:e} (expected < 1e-4)"
        );
    }

    #[test]
    fn graph_laplacian_grid_linear_function_is_zero() {
        // The grid path must preserve the Δ(linear)=0 identity at interior
        // vertices exactly (no rounding — 4f - 4 neighbors of a linear func
        // cancels bit-identically when f is integer-valued).
        let (w, h) = (6usize, 6usize);
        let cx = CellComplex::grid_2d(w, h);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for y in 0..h {
            for x in 0..w {
                potential.set_scalar(y * w + x, (x + y) as f32);
            }
        }
        let lap = graph_laplacian(&cx, &potential);
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let v = lap.scalar(y * w + x);
                assert!(
                    v.abs() < 1e-6,
                    "grid Δ(linear) at ({x},{y}) should be 0, got {v}"
                );
            }
        }
    }

    #[test]
    fn graph_laplacian_grid_dims_cleared_on_remove_face() {
        // The `merkle_root` lesson applied to grid_dims: any topology mutation
        // invalidates the regular-grid invariant. A grid with a removed face
        // is no longer a regular grid, so the stencil would be wrong at the
        // gap — grid_dims must be None after remove_face.
        let mut cx = CellComplex::grid_2d(5, 5);
        assert_eq!(cx.grid_dims(), Some((5, 5)));
        cx.remove_face(0);
        assert_eq!(
            cx.grid_dims(),
            None,
            "grid_dims must clear after remove_face"
        );
    }

    #[test]
    fn graph_laplacian_grid_dims_cleared_on_remove_cell() {
        // Same contract for remove_cell at every rank.
        let mut cx = CellComplex::grid_2d(5, 5);
        cx.remove_cell(0, 0); // remove vertex 0
        assert_eq!(
            cx.grid_dims(),
            None,
            "grid_dims must clear after remove_cell(0)"
        );

        let mut cx = CellComplex::grid_2d(5, 5);
        cx.remove_cell(1, 0); // remove edge 0
        assert_eq!(
            cx.grid_dims(),
            None,
            "grid_dims must clear after remove_cell(1)"
        );
    }

    // ── Plan 454 T3: 7-point-stencil fast-path for 3D grids ─────────────
    //
    // The `grid_dims_full()` dispatch in `graph_laplacian_into` routes 3D
    // `grid_3d` products to the 7-point stencil. Same contract as the 2D
    // fast path: the stencil and edge-list paths are mathematically
    // identical (both realize δ₁d₀); the only permissible difference is
    // ULP-level f32 rounding from the changed accumulation order. The
    // Δ(linear)=0 identity must hold exactly at interior vertices.

    /// Compute the edge-list path directly (bypassing the grid dispatch) on
    /// the same 3D grid complex. `graph_laplacian_edge_list_into` reads
    /// `cx.boundary_entries(0)` and ignores `grid_dims_full`, so passing a
    /// `grid_3d` complex exercises the pre-stencil accumulation path.
    #[cfg(feature = "grid_3d")]
    fn edge_list_laplacian_3d(cx: &CellComplex, potential: &CochainField) -> CochainField {
        let mut out = CochainField::zeros(0, cx.n_vertices(), potential.dim);
        graph_laplacian_edge_list_into(cx, potential, &mut out);
        out
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn graph_laplacian_grid_3d_linear_function_is_zero() {
        // Δ₀(linear) = 0 at interior vertices — the load-bearing DEC identity.
        // Must hold exactly (6f - 6 neighbors of a linear func cancels
        // bit-identically when f is integer-valued).
        let (w, h, d) = (5usize, 5usize, 5usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    // Linear in (x, y, z): f = 2x + 3y + 5z
                    let vidx = (z * h + y) * w + x;
                    potential.set_scalar(vidx, (2 * x + 3 * y + 5 * z) as f32);
                }
            }
        }
        let lap = graph_laplacian(&cx, &potential);
        // Interior vertices: 1 <= x < w-1, 1 <= y < h-1, 1 <= z < d-1
        for z in 1..(d - 1) {
            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let vidx = (z * h + y) * w + x;
                    let v = lap.scalar(vidx);
                    assert!(
                        v.abs() < 1e-6,
                        "3D grid Δ(linear) at ({x},{y},{z}) should be 0, got {v}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn graph_laplacian_grid_3d_matches_edge_list_1ch() {
        // Single-channel: the 7-point stencil and edge-list paths must agree
        // to within a tiny tolerance (ULP-level rounding from accumulation
        // order).
        let (w, h, d) = (5usize, 4usize, 4usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            potential.set_scalar(i, ((i as f32) * 0.37).sin());
        }
        let lap_grid = graph_laplacian(&cx, &potential);
        let lap_edges = edge_list_laplacian_3d(&cx, &potential);
        let mut max_diff = 0.0f32;
        for i in 0..cx.n_vertices() {
            let diff = (lap_grid.scalar(i) - lap_edges.scalar(i)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1e-4,
            "3D grid vs edge-list laplacian diverged by {max_diff:e} (expected < 1e-4)"
        );
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn graph_laplacian_grid_3d_matches_edge_list_multich() {
        // Multi-channel (dim=16, the G5 workload shape): same contract.
        let (w, h, d) = (4usize, 4usize, 3usize);
        let dim = 16usize;
        let cx = CellComplex::grid_3d(w, h, d);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), dim);
        for cell in 0..cx.n_vertices() {
            for ch in 0..dim {
                let v = ((cell as f32 * 0.11 + ch as f32 * 0.73).sin()) * 2.0;
                potential.data[cell * dim + ch] = v;
            }
        }
        let lap_grid = graph_laplacian(&cx, &potential);
        let lap_edges = edge_list_laplacian_3d(&cx, &potential);
        let len = cx.n_vertices() * dim;
        let mut max_diff = 0.0f32;
        for i in 0..len {
            let diff = (lap_grid.data[i] - lap_edges.data[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1e-4,
            "3D grid vs edge-list multichannel diverged by {max_diff:e} (expected < 1e-4)"
        );
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn graph_laplacian_grid_3d_boundary_degrees() {
        // Verify the boundary vertex degrees: corner=3, edge=4, face=5,
        // interior=6. Construct a delta function (1 at origin, 0 elsewhere);
        // the Laplacian at the origin is deg(origin)*1 - 0 = deg(origin), and
        // the Laplacian at each neighbor of the origin is 0*1 - 1 = -1.
        let (w, h, d) = (4usize, 4usize, 4usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        // Origin corner (0,0,0): degree 3
        potential.set_scalar(0, 1.0);
        let lap = graph_laplacian(&cx, &potential);
        // Corner (0,0,0): 3 neighbors, so Δ = 3*1 - 0 - 0 - 0 = 3.0
        assert_eq!(lap.scalar(0), 3.0, "corner degree should be 3");
        // Its 3 neighbors should each be -1.0:
        //   +x neighbor: (1,0,0) -> vidx = (0*h+0)*w+1 = 1
        //   +y neighbor: (0,1,0) -> vidx = (0*h+1)*w+0 = w = 4
        //   +z neighbor: (0,0,1) -> vidx = (1*h+0)*w+0 = h*w = 16
        assert_eq!(lap.scalar(1), -1.0, "+x neighbor should be -1");
        assert_eq!(lap.scalar(w), -1.0, "+y neighbor should be -1");
        assert_eq!(lap.scalar(w * h), -1.0, "+z neighbor should be -1");
        // All other vertices should be 0 (not neighbors of the origin).
        let x_n = 1usize;
        let y_n = w;
        let z_n = w * h;
        for v in 0..cx.n_vertices() {
            if v != 0 && v != x_n && v != y_n && v != z_n {
                assert_eq!(lap.scalar(v), 0.0, "non-neighbor vertex {v} should be 0");
            }
        }
    }

    #[cfg(feature = "grid_3d")]
    #[test]
    fn graph_laplacian_grid_3d_mirror_symmetry() {
        // Symmetry: Δ at (x,y,z) equals Δ at the grid-reflected point
        // (w-1-x, h-1-y, d-1-z) when the potential is mirror-symmetric about
        // the grid center. This validates that the boundary handling is
        // uniform across all 8 corners / 12 edges / 6 faces (no asymmetry bug
        // in the flag-based boundary loop).
        let (w, h, d) = (5usize, 5usize, 5usize);
        let cx = CellComplex::grid_3d(w, h, d);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let vidx = (z * h + y) * w + x;
                    // f(x,y,z) = (x - cx)² + (y - cy)² + (z - cz)² — symmetric
                    // about the center. cx = (w-1)/2, etc.
                    let cx_f = (w - 1) as f32 / 2.0;
                    let cy_f = (h - 1) as f32 / 2.0;
                    let cz_f = (d - 1) as f32 / 2.0;
                    let dx = x as f32 - cx_f;
                    let dy = y as f32 - cy_f;
                    let dz = z as f32 - cz_f;
                    potential.set_scalar(vidx, dx * dx + dy * dy + dz * dz);
                }
            }
        }
        let lap = graph_laplacian(&cx, &potential);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let vidx = (z * h + y) * w + x;
                    let mirror_vidx = ((d - 1 - z) * h + (h - 1 - y)) * w + (w - 1 - x);
                    let v = lap.scalar(vidx);
                    let vm = lap.scalar(mirror_vidx);
                    assert_eq!(
                        v.to_bits(),
                        vm.to_bits(),
                        "Δ at ({x},{y},{z})={v} should equal Δ at mirror point={vm}"
                    );
                }
            }
        }
    }
}
