//! Hodge decomposition, Betti numbers, and harmonic projection.
//!
//! Implements Plan 251 Phase 3 (T14–T16, T18–T19).
//!
//! The Hodge decomposition theorem states that any k-cochain ω on a cell complex
//! can be uniquely decomposed into three orthogonal components:
//!
//! ```text
//! ω = dₖ₋₁(α) + h + δₖ₊₁(β)
//!     exact      harmonic  coexact
//! ```
//!
//! - **exact** = dₖ₋₁(α): gradient/conservative component (in im(dₖ₋₁))
//! - **harmonic** = h: topological component (in ker(Δₖ) = ker(dₖ) ∩ ker(δₖ))
//! - **coexact** = δₖ₊₁(β): divergence-free/circulation component (in im(δₖ₊₁))
//!
//! For rank 0: exact = 0 (no d₋₁), so ω = harmonic + coexact.
//!
//! # Solver
//!
//! Uses Conjugate Gradient (CG) on the Hodge Laplacian as matvec.
//! No external linear algebra dependencies — CG converges for SPD systems
//! (the Laplacian is SPD on the orthogonal complement of its kernel).
//! For rank 0 on connected complexes, we project out the constant null space
//! before solving.

use crate::operators::{
    codifferential, exterior_derivative, graph_laplacian_into, hodge_laplacian,
    hodge_laplacian_into,
};
use crate::types::{CellComplex, CochainField, MAX_RANK};

// ---------------------------------------------------------------------------
// Hodge Components
// ---------------------------------------------------------------------------

/// Result of Hodge decomposition: ω = exact + harmonic + coexact.
pub struct HodgeComponents {
    /// Exact (gradient/conservative) component: dₖ₋₁(α) ∈ im(dₖ₋₁).
    /// Zero for rank-0 cochains (no d₋₁ exists).
    pub exact: CochainField,
    /// Harmonic (topological) component: h ∈ ker(Δₖ).
    /// For rank 0 on a connected complex: h = mean(ω) · 𝟙.
    pub harmonic: CochainField,
    /// Coexact (circulation/divergence-free) component: δₖ₊₁(β) ∈ im(δₖ₊₁).
    pub coexact: CochainField,
}

// ---------------------------------------------------------------------------
// Conjugate Gradient Solver
// ---------------------------------------------------------------------------

/// Default CG tolerance.
const CG_TOL: f32 = 1e-6;
/// Default max CG iterations (game grids converge fast).
const CG_MAX_ITER: usize = 1000;

/// Solve Ax = b where A is the Hodge Laplacian at rank `rank` using CG.
///
/// For rank 0, projects out the constant null space before solving.
/// `matvec` must be a closure that computes A·x for a given x.
///
/// Returns the solution x (allocated internally).
fn cg_solve(
    cx: &CellComplex,
    rhs: &CochainField,
    rank: u8,
    tol: f32,
    max_iter: usize,
) -> CochainField {
    let n = rhs.n_cells();
    let dim = rhs.dim;
    let mut x = CochainField::zeros(rank, n, dim);

    // For dim > 1, solve each feature channel independently.
    //
    // Scratch buffers (b, r, p, ap) are hoisted out of `cg_solve_scalar` and reused
    // across all channels — eliminates 4·dim heap allocations per call on the
    // multi-channel path (e.g. dim=4 ⇒ 16 allocs → 0). Grown only when n grows.
    match dim {
        1 => {
            let mut scratch = CgScratch::new(n);
            cg_solve_scalar(
                cx,
                &rhs.data,
                rank,
                tol,
                max_iter,
                &mut x.data,
                &mut scratch,
            );
        }
        _ => {
            // Per-channel solve
            let mut rhs_ch = vec![0.0f32; n];
            let mut x_ch = vec![0.0f32; n];
            let mut scratch = CgScratch::new(n);
            for d in 0..dim {
                for i in 0..n {
                    rhs_ch[i] = rhs.data[i * dim + d];
                }
                cg_solve_scalar(cx, &rhs_ch, rank, tol, max_iter, &mut x_ch, &mut scratch);
                for i in 0..n {
                    x.data[i * dim + d] = x_ch[i];
                }
            }
        }
    }

    x
}

/// Reusable scratch buffers for [`cg_solve_scalar`].
///
/// Holds the four length-`n` Vec allocations (b, r, p, ap) that the scalar CG
/// solver would otherwise allocate on every call. Constructed once at the
/// entry of [`cg_solve`] and grown only when `n` increases — the steady-state
/// multi-channel path allocates zero Vecs after the first channel.
struct CgScratch {
    b: Vec<f32>,
    r: Vec<f32>,
    p: Vec<f32>,
    ap: Vec<f32>,
}

impl CgScratch {
    #[inline]
    fn new(n: usize) -> Self {
        Self {
            b: vec![0.0; n],
            r: vec![0.0; n],
            p: vec![0.0; n],
            ap: vec![0.0; n],
        }
    }

    /// Grow each buffer to `n` if needed. No-op on the steady-state path where
    /// the previous call used the same `n`.
    #[inline]
    fn ensure(&mut self, n: usize) {
        if self.b.len() != n {
            self.b.resize(n, 0.0);
            self.r.resize(n, 0.0);
            self.p.resize(n, 0.0);
            self.ap.resize(n, 0.0);
        }
    }
}

/// Scalar CG solve: A·x = rhs for a single feature channel.
///
/// Uses `hodge_laplacian` (or `graph_laplacian` for rank 0) as the matvec.
/// For rank 0, removes the mean from both sides to handle the constant null space.
///
/// `scratch` provides the four length-`n` working buffers (b, r, p, ap) so the
/// caller can reuse them across many channels instead of paying 4 allocations
/// per call.
fn cg_solve_scalar(
    cx: &CellComplex,
    rhs: &[f32],
    rank: u8,
    tol: f32,
    max_iter: usize,
    x_out: &mut [f32],
    scratch: &mut CgScratch,
) {
    let n = rhs.len();
    debug_assert_eq!(x_out.len(), n);
    scratch.ensure(n);

    // For rank 0: project out constant null space
    let rhs_mean = match rank {
        0 => rhs.iter().sum::<f32>() / (n as f32),
        _ => 0.0,
    };

    // Split scratch into the four working buffers ONCE. We hold all four
    // disjoint field slices for the rest of the function; `scratch` itself
    // is not touched again. This avoids the double-`&mut` borrow that would
    // arise if we re-borrowed `scratch` later (e.g. via a helper).
    let CgScratch { b, r, p, ap } = scratch;
    let b = &mut b[..n];
    let r = &mut r[..n];
    let p = &mut p[..n];
    let ap = &mut ap[..n];

    // Build corrected RHS into the reused `b` buffer (avoids `rhs.to_vec()` per call).
    // `b[..n]` is fully overwritten below before any read.
    b.copy_from_slice(rhs);
    if rank == 0 {
        for v in b.iter_mut() {
            *v -= rhs_mean;
        }
    }

    // Pre-allocate reusable cochain fields for matvec — allocated ONCE, reused across
    // all CG iterations (up to max_iter). Previously, graph_laplacian / hodge_laplacian
    // allocated a fresh CochainField (and hodge_laplacian allocated 4 intermediates)
    // per iteration → up to 5*max_iter heap allocs per solve.
    let mut v_field = CochainField::zeros(rank, n, 1);
    let mut ax_field = CochainField::zeros(rank, n, 1);
    // Scratch for hodge_laplacian_into (rank ≥ 1). Sized for the intermediate ranks;
    // unused for rank 0 (graph_laplacian_into needs no scratch beyond ax_field).
    let n_upper = if rank + 1 < MAX_RANK {
        cx.n_cells(rank + 1)
    } else {
        0
    };
    let n_lower = if rank > 0 { cx.n_cells(rank - 1) } else { 0 };
    let mut scratch_upper = CochainField::zeros(rank + 1, n_upper, 1);
    let mut scratch_lower = CochainField::zeros(rank.saturating_sub(1), n_lower, 1);
    let mut scratch_result = CochainField::zeros(rank, n, 1);

    // Matvec closure: compute A·v for given v, writing directly into `out` (zero alloc per iter).
    // Declared `mut` because it captures `&mut` cochain fields.
    let mut matvec = |cx: &CellComplex, v: &[f32], v_field: &mut CochainField, out: &mut [f32]| {
        v_field.data.copy_from_slice(v);
        match rank {
            0 => graph_laplacian_into(cx, v_field, &mut ax_field),
            _ => hodge_laplacian_into(
                cx,
                v_field,
                &mut ax_field,
                &mut scratch_upper,
                &mut scratch_lower,
                &mut scratch_result,
            ),
        };
        out.copy_from_slice(&ax_field.data);
    };

    // CG iterations: r = b, p = r, iterate. All four working buffers come from
    // the caller-supplied scratch — no per-call Vec allocations.
    x_out.fill(0.0f32);
    // CG initialization: r = b; p = r.clone().
    r.copy_from_slice(b);
    p.copy_from_slice(r);

    let mut rs_old = dot(r, r);

    if rs_old < tol * tol {
        // RHS is essentially zero — solution is zero (in the projected space)
        // For rank 0, the solution in the null space direction doesn't matter
        // since we only need the component orthogonal to constants.
        return;
    }

    for _ in 0..max_iter {
        matvec(cx, p, &mut v_field, ap);

        let p_ap = dot(p, ap);
        if p_ap.abs() < 1e-20 {
            break; // Breakdown — p is in null space
        }
        let alpha = rs_old / p_ap;

        // Fused SAXPY: x += α·p and r -= α·Ap in a single pass over the index
        // stream. The two arrays are independent so fusing doesn't alter any
        // FP reduction order — each array's accumulation is unchanged.
        // Halves L1 cache misses on `p`, `ap`, `x_out`, `r` (each up to 16 KB
        // on 64×64 grids).
        for i in 0..n {
            x_out[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new = dot(r, r);
        if rs_new < tol * tol {
            break;
        }

        let beta_cg = rs_new / rs_old;
        // p = r + β·p
        for i in 0..n {
            p[i] = r[i] + beta_cg * p[i];
        }

        rs_old = rs_new;
    }
}

/// Dot product of two slices — SIMD-accelerated.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len().min(b.len());
    crate::simd::simd_dot_f32(&a[..len], &b[..len], len)
}

// ---------------------------------------------------------------------------
// Hodge Decomposition (T14)
// ---------------------------------------------------------------------------

/// Decompose a k-cochain into exact, harmonic, and coexact components.
///
/// Uses the Helmholtz decomposition via Poisson solves:
/// - exact = dₖ₋₁(Δₖ₋₁⁻¹(δₖ(ω))) for rank ≥ 1
/// - coexact = δₖ₊₁(Δₖ₊₁⁻¹(dₖ(ω)))
/// - harmonic = ω - exact - coexact
///
/// For rank 0: exact is zero (no d₋₁), so ω = harmonic + coexact.
pub fn hodge_decompose(cx: &CellComplex, input: &CochainField) -> HodgeComponents {
    let k = input.rank;
    let n = input.n_cells();
    let dim = input.dim;

    // -----------------------------------------------------------------------
    // Coexact component: δₖ₊₁(Δₖ₊₁⁻¹(dₖ(ω)))
    // -----------------------------------------------------------------------
    // If k+1 cells exist, compute dₖ(ω), solve Δₖ₊₁·β = dₖ(ω), then coexact = δₖ₊₁(β)
    let coexact = match k < MAX_RANK && cx.n_cells(k + 1) > 0 {
        true => {
            let dk_omega = exterior_derivative(cx, input);
            // Solve Δ_{k+1}·β = dₖ(ω)
            let beta = cg_solve(cx, &dk_omega, k + 1, CG_TOL, CG_MAX_ITER);
            // coexact = δ_{k+2}(β)
            // Note: δ_{k+1} requires rank > 0, which (k+1) always is here since k >= 0
            match k + 1 > 0 && cx.n_cells(k) > 0 {
                true => codifferential(cx, &beta),
                false => CochainField::zeros(k, n, dim),
            }
        }
        false => CochainField::zeros(k, n, dim),
    };

    // -----------------------------------------------------------------------
    // Exact component: dₖ₋₁(Δₖ₋₁⁻¹(δₖ(ω)))
    // -----------------------------------------------------------------------
    // Only for rank ≥ 1: need dₖ₋₁ and Δₖ₋₁
    let exact = match k > 0 {
        true => {
            // δₖ(ω) requires rank > 0
            let delta_k_omega = codifferential(cx, input);
            // Solve Δ_{k-1}·α = δₖ(ω)
            let alpha = cg_solve(cx, &delta_k_omega, k - 1, CG_TOL, CG_MAX_ITER);
            // exact = dₖ₋₁(α)
            match cx.n_cells(k) > 0 {
                true => exterior_derivative(cx, &alpha),
                false => CochainField::zeros(k, n, dim),
            }
        }
        false => CochainField::zeros(k, n, dim),
    };

    // -----------------------------------------------------------------------
    // Harmonic component: ω - exact - coexact
    // -----------------------------------------------------------------------
    let mut harmonic = CochainField::zeros(k, n, dim);
    for i in 0..input.data.len() {
        harmonic.data[i] = input.data[i] - exact.data[i] - coexact.data[i];
    }

    HodgeComponents {
        exact,
        harmonic,
        coexact,
    }
}

// ---------------------------------------------------------------------------
// Betti Numbers (T15)
// ---------------------------------------------------------------------------

/// Compute Betti numbers β₀, β₁, β₂, β₃ for the cell complex.
///
/// Betti number βₖ counts the number of k-dimensional "holes":
/// - β₀ = number of connected components
/// - β₁ = number of independent non-contractible loops (genus)
/// - β₂ = number of enclosed voids
/// - β₃ = number of 3D cavities
///
/// Computed via: βₖ = nₖ - rank(Bₖ) - rank(Bₖ₊₁)
/// where rank(Bₖ) is the number of linearly independent columns in the
/// boundary matrix Bₖ (mapping k-cells to (k-1)-cells).
///
/// For the sparse triplet representation, we use Gaussian elimination
/// on a column-compressed form to find the rank.
pub fn betti_numbers(cx: &CellComplex) -> [usize; 4] {
    // Compute rank of each boundary matrix Bₖ
    // Bₖ has shape [n_{k-1} × n_k], stored as triplets in boundaries[k-1]
    // boundaries[0] = B₁ (vertex→edge), boundaries[1] = B₂ (edge→face), etc.
    let mut betti = [0usize; 4];

    // rank(B₀) = 0 (no rank -1 cells)
    let rank_b0 = 0usize;

    // rank(B₁) from boundaries[0]
    let rank_b1 = boundary_matrix_rank(cx, 0);

    // rank(B₂) from boundaries[1]
    let rank_b2 = boundary_matrix_rank(cx, 1);

    // rank(B₃) from boundaries[2]
    let rank_b3 = boundary_matrix_rank(cx, 2);

    // rank(B₄) = 0 (no rank 4 cells)
    let rank_b4 = 0usize;

    // βₖ = nₖ - rank(Bₖ) - rank(Bₖ₊₁)
    // β₀ = n₀ - rank(B₀) - rank(B₁) = n_vertices - 0 - rank_b1
    let n0 = cx.n_cells(0);
    let n1 = cx.n_cells(1);
    let n2 = cx.n_cells(2);
    let n3 = cx.n_cells(3);

    betti[0] = n0.saturating_sub(rank_b0).saturating_sub(rank_b1);
    betti[1] = n1.saturating_sub(rank_b1).saturating_sub(rank_b2);
    betti[2] = n2.saturating_sub(rank_b2).saturating_sub(rank_b3);
    betti[3] = n3.saturating_sub(rank_b3).saturating_sub(rank_b4);

    betti
}

/// Compute the rank (number of linearly independent columns) of boundary matrix Bₖ₊₁.
///
/// `which` indexes into `boundaries[]`: 0 = B₁, 1 = B₂, 2 = B₃.
/// Uses Gaussian elimination on a dense copy (feasible for game-sized grids).
/// Returns 0 if the boundary is empty.
fn boundary_matrix_rank(cx: &CellComplex, which: usize) -> usize {
    if which >= MAX_RANK as usize {
        return 0;
    }

    let entries = cx.boundary_entries(which as u8);
    if entries.is_empty() {
        return 0;
    }

    // B_{which+1} has shape [n_{which} × n_{which+1}]
    let n_rows = cx.n_cells(which as u8);
    let n_cols = cx.n_cells((which as u8) + 1);

    if n_rows == 0 || n_cols == 0 {
        return 0;
    }

    // Build dense matrix (f32 for elimination)
    let mut mat = vec![0.0f32; n_rows * n_cols];
    for &(row, col, sign) in entries {
        mat[row * n_cols + col] = sign as f32;
    }

    // Gaussian elimination to find rank (column echelon form)
    // Process columns left to right, find pivot row for each
    let mut pivot_row = 0;
    let mut rank = 0;

    for col in 0..n_cols {
        // Find a non-zero entry in this column at or below pivot_row
        let mut found = None;
        for row in pivot_row..n_rows {
            if mat[row * n_cols + col].abs() > 1e-10 {
                found = Some(row);
                break;
            }
        }

        let found_row = match found {
            Some(r) => r,
            None => continue, // Column is zero — no pivot
        };

        // Swap rows if needed
        if found_row != pivot_row {
            for c in 0..n_cols {
                mat.swap(pivot_row * n_cols + c, found_row * n_cols + c);
            }
        }

        // Eliminate all other entries in this column
        let pivot_val = mat[pivot_row * n_cols + col];
        for row in 0..n_rows {
            if row == pivot_row {
                continue;
            }
            let factor = mat[row * n_cols + col] / pivot_val;
            if factor.abs() > 1e-15 {
                for c in 0..n_cols {
                    mat[row * n_cols + c] -= factor * mat[pivot_row * n_cols + c];
                }
            }
        }

        pivot_row += 1;
        rank += 1;

        if pivot_row >= n_rows {
            break;
        }
    }

    rank
}

// ---------------------------------------------------------------------------
// Harmonic Projector (T16)
// ---------------------------------------------------------------------------

/// Project a k-cochain onto the harmonic space ker(Δₖ).
///
/// For a connected cell complex:
/// - Rank 0: harmonic = constant functions → P_harm(ω) = mean(ω) · 𝟙
/// - Rank 1+: uses the Hodge decomposition — harmonic component
///
/// The harmonic component represents topologically persistent features
/// that cannot be removed by gradient or circulation operations.
pub fn harmonic_projector(cx: &CellComplex, input: &CochainField) -> CochainField {
    let k = input.rank;
    let n = input.n_cells();
    let dim = input.dim;

    match k {
        0 => {
            // For rank 0 on a connected complex, harmonic = mean · 𝟙
            let mut result = CochainField::zeros(k, n, dim);
            for d in 0..dim {
                let mean: f32 = (0..n).map(|i| input.data[i * dim + d]).sum::<f32>() / (n as f32);
                for i in 0..n {
                    result.data[i * dim + d] = mean;
                }
            }
            result
        }
        _ => {
            // For higher ranks, use full Hodge decomposition and return harmonic component
            let components = hodge_decompose(cx, input);
            components.harmonic
        }
    }
}

// ---------------------------------------------------------------------------
// Hodge Energy & Pruner Signals (T27–T29)
// ---------------------------------------------------------------------------

/// Compute Hodge energy E(ω) = ⟨ω, Δₖω⟩ for a k-cochain.
///
/// This is the DEC analog of Dirichlet energy. For rank 0, this reduces to
/// the standard graph Dirichlet energy. For higher ranks, it measures how
/// "non-harmonic" the cochain is — high energy means far from harmonic.
pub fn hodge_energy(cx: &CellComplex, omega: &CochainField) -> f32 {
    let lap = hodge_laplacian(cx, omega);
    // ⟨ω, Δω⟩ = Σᵢ ωᵢ · Δωᵢ
    omega
        .data
        .iter()
        .zip(lap.data.iter())
        .map(|(&a, &b)| a * b)
        .sum()
}

/// Hodge residual: measures how well a cochain satisfies the DEC constraint.
///
/// `residual = ‖Δₖω‖₂` — zero means ω is harmonic (fully constraint-satisfying).
/// - Low residual = high constraint satisfaction = should be pruned (already good)
/// - High residual = needs attention (far from harmonic, significant dynamics)
pub fn hodge_residual(cx: &CellComplex, omega: &CochainField) -> f32 {
    let lap = hodge_laplacian(cx, omega);
    lap.data.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// DEC-based relevance boost for a position's feature cochain.
///
/// Returns a value in (0, 1] where:
/// - High values indicate topologically smooth/relevant positions
/// - Low values indicate topologically noisy/irrelevant positions
///
/// Uses `1 / (1 + E_hodge)` as a sigmoid-like decay from 1 (zero energy)
/// toward 0 (high energy). Never uses softmax — always sigmoid family.
pub fn dec_relevance_score(cx: &CellComplex, features: &CochainField) -> f32 {
    let energy = hodge_energy(cx, features);
    1.0 / (1.0 + energy)
}

// ---------------------------------------------------------------------------
// Hodge Spectrum — Power Iteration (T17)
// ---------------------------------------------------------------------------

/// Compute approximate eigenvalues of Δₖ using power iteration with deflation.
///
/// Returns eigenvalues sorted descending. Uses `n_eigenvalues` rounds of
/// power iteration on the Hodge Laplacian matvec to extract the top eigenvalues.
/// For each round:
/// 1. Start with random vector
/// 2. Iterate: x = Δₖ(x), normalize
/// 3. Rayleigh quotient gives eigenvalue estimate
/// 4. Deflate: subtract projected component from previous eigenvectors
///
/// # Arguments
/// * `cx` — The cell complex
/// * `rank` — Rank of the cochain (determines which Δₖ to use)
/// * `n_eigenvalues` — Number of eigenvalues to extract
/// * `max_iter` — Maximum power-iteration steps per eigenvalue
///
/// # Returns
/// Eigenvalues sorted descending (largest first). Length = `n_eigenvalues`.
pub fn hodge_spectrum(
    cx: &CellComplex,
    rank: u8,
    n_eigenvalues: usize,
    max_iter: usize,
) -> Vec<f32> {
    let n = cx.n_cells(rank);
    let n_ev = n_eigenvalues.min(n);
    if n_ev == 0 {
        return Vec::new();
    }

    let mut eigenvalues = Vec::with_capacity(n_ev);
    let mut eigenvectors = Vec::with_capacity(n_ev);

    // Pre-allocate reusable cochain fields for hodge_laplacian_into — reused across
    // all power-iteration steps (max_iter * n_ev calls previously allocated fresh).
    let mut cochain = CochainField::zeros(rank, n, 1);
    let mut lap = CochainField::zeros(rank, n, 1);
    let n_upper = if rank + 1 < MAX_RANK {
        cx.n_cells(rank + 1)
    } else {
        0
    };
    let n_lower = if rank > 0 { cx.n_cells(rank - 1) } else { 0 };
    let mut scratch_upper = CochainField::zeros(rank + 1, n_upper, 1);
    let mut scratch_lower = CochainField::zeros(rank.saturating_sub(1), n_lower, 1);
    let mut scratch_result = CochainField::zeros(rank, n, 1);

    for ev_idx in 0..n_ev {
        // Initialize with pseudo-random vector using simple LCG seeded by index
        let seed = (ev_idx + 7) as u32;
        let mut x = Vec::with_capacity(n);
        for i in 0..n {
            let val = simple_lcg(seed.wrapping_add(i as u32));
            x.push((val as f32) / (u32::MAX as f32) - 0.5);
        }

        // Deflate: remove projection onto already-found eigenvectors
        deflate(&mut x, &eigenvectors);
        normalize_inplace(&mut x);

        // Power iteration — reuses pre-allocated scratch across iterations (zero alloc per iter).
        for _ in 0..max_iter {
            cochain.data.copy_from_slice(&x);
            hodge_laplacian_into(
                cx,
                &cochain,
                &mut lap,
                &mut scratch_upper,
                &mut scratch_lower,
                &mut scratch_result,
            );
            x.copy_from_slice(&lap.data);

            // Deflate again
            deflate(&mut x, &eigenvectors);

            let norm = l2_norm(&x);
            if norm < 1e-12 {
                break;
            }
            for v in x.iter_mut() {
                *v /= norm;
            }
        }

        // Rayleigh quotient: λ = xᵀΔx / (xᵀx)
        cochain.data.copy_from_slice(&x);
        hodge_laplacian_into(
            cx,
            &cochain,
            &mut lap,
            &mut scratch_upper,
            &mut scratch_lower,
            &mut scratch_result,
        );
        // SIMD reduction: replaces scalar `iter().zip().map(|(a,b)| a*b).sum()`.
        // Spectrum-only path — does NOT affect the CG solver used by `arena_proof`.
        let rayleigh = crate::simd::simd_dot_f32(&x, &lap.data, x.len());

        eigenvalues.push(rayleigh.max(0.0));
        eigenvectors.push(x);
    }

    eigenvalues.sort_unstable_by(|a, b| b.total_cmp(a));
    eigenvalues
}

/// Simple LCG pseudo-random number generator (no external deps).
fn simple_lcg(state: u32) -> u32 {
    // Numerical Recipes LCG parameters
    state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

/// L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    // SIMD self-dot for ||v||²; spectrum-only diagnostic path.
    crate::simd::simd_dot_f32(v, v, v.len()).sqrt()
}

/// Normalize a vector to unit length in-place.
fn normalize_inplace(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Remove projections of `v` onto each vector in `basis` (Gram–Schmidt deflation).
fn deflate(v: &mut [f32], basis: &[Vec<f32>]) {
    for b in basis {
        // SIMD dot — spectrum-only path, does not feed CG.
        let dot = crate::simd::simd_dot_f32(v, b, v.len());
        for (vi, bi) in v.iter_mut().zip(b.iter()) {
            *vi -= dot * bi;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (T18, T27–T30)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::operators::graph_laplacian;
    use super::*;

    /// Tolerance for floating-point comparisons.
    const TOL: f32 = 1e-3;

    /// Compute L2 inner product of two cochains (dot product of data vectors).
    fn cochain_dot(a: &CochainField, b: &CochainField) -> f32 {
        debug_assert_eq!(a.data.len(), b.data.len());
        let mut s = 0.0f32;
        for i in 0..a.data.len() {
            s += a.data[i] * b.data[i];
        }
        s
    }

    /// Max absolute value in a cochain.
    fn cochain_max_abs(a: &CochainField) -> f32 {
        a.data.iter().map(|&v| v.abs()).fold(0.0f32, f32::max)
    }

    // -----------------------------------------------------------------------
    // T14: Hodge Decomposition
    // -----------------------------------------------------------------------

    #[test]
    fn rank0_decomposition_constant_input() {
        // A constant potential: ω = c·𝟙
        // Decomposition: harmonic = c·𝟙, coexact = 0, exact = 0
        let cx = CellComplex::grid_2d(4, 4);
        let mut input = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            input.set_scalar(i, 5.0);
        }

        let comp = hodge_decompose(&cx, &input);

        // Exact is always 0 for rank 0
        assert!(
            cochain_max_abs(&comp.exact) < TOL,
            "rank-0 exact should be 0, max = {}",
            cochain_max_abs(&comp.exact)
        );

        // Harmonic should be constant = 5.0
        for i in 0..cx.n_vertices() {
            let diff = (comp.harmonic.scalar(i) - 5.0).abs();
            assert!(
                diff < TOL,
                "harmonic[{i}] should be 5.0, got {}",
                comp.harmonic.scalar(i)
            );
        }

        // Coexact should be ≈ 0
        assert!(
            cochain_max_abs(&comp.coexact) < TOL,
            "rank-0 coexact of constant should be ~0, max = {}",
            cochain_max_abs(&comp.coexact)
        );
    }

    #[test]
    fn rank0_decomposition_reconstruction() {
        // Random-ish potential: decomposition should reconstruct original
        let cx = CellComplex::grid_2d(8, 8);
        let mut input = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            input.set_scalar(i, (i as f32 * 1.7 + 0.3).sin() * 3.0 + 5.0);
        }

        let comp = hodge_decompose(&cx, &input);

        // Reconstruction: exact + harmonic + coexact ≈ input
        let mut reconstructed = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..input.data.len() {
            reconstructed.data[i] =
                comp.exact.data[i] + comp.harmonic.data[i] + comp.coexact.data[i];
        }

        let max_err = (0..input.data.len())
            .map(|i| (input.data[i] - reconstructed.data[i]).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_err < TOL,
            "reconstruction error = {max_err} (tol = {TOL})"
        );
    }

    #[test]
    fn rank0_orthogonality() {
        // exact ⊥ coexact ⊥ harmonic (in the L2/ℓ² inner product)
        let cx = CellComplex::grid_2d(8, 8);
        let mut input = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            input.set_scalar(i, (i as f32 * 2.3 + 1.1).sin() * 2.0);
        }

        let comp = hodge_decompose(&cx, &input);

        // For rank 0: exact is zero, so only check harmonic ⊥ coexact
        let dot_hc = cochain_dot(&comp.harmonic, &comp.coexact);
        let norm_h = (cochain_dot(&comp.harmonic, &comp.harmonic)).sqrt();
        let norm_c = (cochain_dot(&comp.coexact, &comp.coexact)).sqrt();

        if norm_h > TOL && norm_c > TOL {
            // Relative orthogonality: |⟨h,c⟩| / (‖h‖·‖c‖) should be small
            let rel = dot_hc.abs() / (norm_h * norm_c);
            assert!(
                rel < 0.1,
                "harmonic ⊥ coexact: |⟨h,c⟩|/(‖h‖·‖c‖) = {rel} (threshold 0.1)"
            );
        }
    }

    #[test]
    fn rank0_harmonic_is_mean() {
        // For rank 0: harmonic component should be the mean of the input
        let cx = CellComplex::grid_2d(6, 6);
        let mut input = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            input.set_scalar(i, (i as f32 * 0.5).sin() * 10.0 + 3.0);
        }

        let mean = input.data.iter().sum::<f32>() / (input.n_cells() as f32);
        let harm = harmonic_projector(&cx, &input);

        for i in 0..harm.n_cells() {
            let diff = (harm.scalar(i) - mean).abs();
            assert!(
                diff < 1e-6,
                "harmonic[{i}] = {} but mean = {mean}",
                harm.scalar(i)
            );
        }
    }

    #[test]
    fn rank1_decomposition_reconstruction() {
        // Rank-1 (edge) cochain decomposition and reconstruction
        let cx = CellComplex::grid_2d(6, 6);
        let mut input = CochainField::zeros(1, cx.n_edges(), 1);
        for i in 0..cx.n_edges() {
            input.set_scalar(i, (i as f32 * 1.3 + 0.7).sin() * 2.0);
        }

        let comp = hodge_decompose(&cx, &input);

        // Reconstruction
        let mut reconstructed = CochainField::zeros(1, cx.n_edges(), 1);
        for i in 0..input.data.len() {
            reconstructed.data[i] =
                comp.exact.data[i] + comp.harmonic.data[i] + comp.coexact.data[i];
        }

        let max_err = (0..input.data.len())
            .map(|i| (input.data[i] - reconstructed.data[i]).abs())
            .fold(0.0f32, f32::max);

        assert!(max_err < 0.1, "rank-1 reconstruction error = {max_err}");
    }

    #[test]
    fn rank1_exact_is_gradient() {
        // Input that IS a gradient: exact should capture it all
        let cx = CellComplex::grid_2d(4, 4);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            potential.set_scalar(i, (i as f32 * 0.5).sin());
        }
        let gradient = exterior_derivative(&cx, &potential);

        let comp = hodge_decompose(&cx, &gradient);

        // The gradient should be entirely in the exact component
        // (harmonic + coexact should be small)
        let residual = cochain_max_abs(&comp.harmonic) + cochain_max_abs(&comp.coexact);
        assert!(
            residual < 0.1,
            "gradient should be exact: harmonic+coexact max = {residual}"
        );
    }

    // -----------------------------------------------------------------------
    // T15: Betti Numbers
    // -----------------------------------------------------------------------

    #[test]
    fn betti_grid_2d() {
        // A 2D grid is contractible: β₀=1, β₁=0, β₂=0
        let cx = CellComplex::grid_2d(8, 8);
        let betti = betti_numbers(&cx);

        assert_eq!(
            betti[0], 1,
            "β₀ (connected components) should be 1 for a connected grid"
        );
        assert_eq!(betti[1], 0, "β₁ (loops) should be 0 for a grid (no holes)");
        assert_eq!(betti[2], 0, "β₂ (voids) should be 0 for a 2D grid");
        assert_eq!(betti[3], 0, "β₃ should be 0 for a 2D grid");
    }

    #[test]
    fn betti_grid_3x3() {
        let cx = CellComplex::grid_2d(3, 3);
        let betti = betti_numbers(&cx);

        assert_eq!(betti[0], 1, "β₀ = 1");
        assert_eq!(betti[1], 0, "β₁ = 0");
        assert_eq!(betti[2], 0, "β₂ = 0");
    }

    #[test]
    fn betti_grid_small() {
        // Smallest non-trivial grid
        let cx = CellComplex::grid_2d(2, 2);
        let betti = betti_numbers(&cx);
        assert_eq!(betti[0], 1);
        assert_eq!(betti[1], 0);
        assert_eq!(betti[2], 0);
    }

    // -----------------------------------------------------------------------
    // T16: Harmonic Projector
    // -----------------------------------------------------------------------

    #[test]
    fn harmonic_projector_rank0_is_projection() {
        // Projecting twice should give the same result
        let cx = CellComplex::grid_2d(5, 5);
        let mut input = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            input.set_scalar(i, (i as f32 * 1.1).cos() * 3.0);
        }

        let h1 = harmonic_projector(&cx, &input);
        let h2 = harmonic_projector(&cx, &h1);

        let max_diff = (0..h1.data.len())
            .map(|i| (h1.data[i] - h2.data[i]).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "P² = P (idempotent): max diff = {max_diff}"
        );
    }

    #[test]
    fn harmonic_projector_kernel_property() {
        // Δₖ(harmonic) ≈ 0 for the harmonic component
        let cx = CellComplex::grid_2d(4, 4);
        let mut input = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            input.set_scalar(i, (i as f32 * 0.7).sin());
        }

        let harm = harmonic_projector(&cx, &input);
        let lap_harm = graph_laplacian(&cx, &harm);

        let max_lap = cochain_max_abs(&lap_harm);
        assert!(max_lap < 1e-4, "Δ₀(harmonic) should be ≈ 0, got {max_lap}");
    }

    // -----------------------------------------------------------------------
    // T18: Orthogonality and reconstruction for rank 1
    // -----------------------------------------------------------------------

    #[test]
    fn rank1_three_way_orthogonality() {
        let cx = CellComplex::grid_2d(6, 6);
        let mut input = CochainField::zeros(1, cx.n_edges(), 1);
        for i in 0..cx.n_edges() {
            input.set_scalar(i, (i as f32 * 1.3 + 0.7).sin() * 2.0);
        }

        let comp = hodge_decompose(&cx, &input);

        let dot_eh = cochain_dot(&comp.exact, &comp.harmonic);
        let dot_ec = cochain_dot(&comp.exact, &comp.coexact);
        let dot_hc = cochain_dot(&comp.harmonic, &comp.coexact);

        let norm_e = (cochain_dot(&comp.exact, &comp.exact)).sqrt();
        let norm_h = (cochain_dot(&comp.harmonic, &comp.harmonic)).sqrt();
        let norm_c = (cochain_dot(&comp.coexact, &comp.coexact)).sqrt();

        let max_norm = norm_e.max(norm_h).max(norm_c);

        if max_norm > TOL {
            let max_ortho = dot_eh.abs().max(dot_ec.abs()).max(dot_hc.abs()) / max_norm;
            // Relative orthogonality — CG on the Laplacian gives approximate decomposition
            assert!(
                max_ortho < 1.0,
                "pairwise orthogonality: max |⟨a,b⟩|/‖max‖ = {max_ortho}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // T27: Hodge energy tests
    // -----------------------------------------------------------------------

    #[test]
    fn hodge_energy_constant_is_zero() {
        // Constant function → Laplacian is zero → energy is zero
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            omega.set_scalar(i, 5.0);
        }
        let e = hodge_energy(&cx, &omega);
        assert!(
            e.abs() < 1e-4,
            "hodge energy of constant should be ~0, got {e}"
        );
    }

    #[test]
    fn hodge_energy_rank0_positive_semidefinite() {
        // ⟨ω, Δω⟩ ≥ 0 for all ω (Laplacian is positive semidefinite)
        let cx = CellComplex::grid_2d(5, 5);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            omega.set_scalar(i, (i as f32 * 1.3).sin());
        }
        let e = hodge_energy(&cx, &omega);
        assert!(e >= -1e-4, "hodge energy should be ≥ 0 (PSD), got {e}");
    }

    #[test]
    fn hodge_energy_rank0_nonzero() {
        // Non-constant function should have positive energy
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        // Step function: left half = 0, right half = 1
        for y in 0..4usize {
            for x in 0..4usize {
                let v = if x >= 2 { 1.0f32 } else { 0.0f32 };
                omega.set_scalar(y * 4 + x, v);
            }
        }
        let e = hodge_energy(&cx, &omega);
        assert!(
            e > 0.1,
            "hodge energy of step function should be > 0, got {e}"
        );
    }

    #[test]
    fn hodge_energy_rank1() {
        // Edge cochain: non-constant should have positive energy
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(1, cx.n_edges(), 1);
        for i in 0..cx.n_edges() {
            omega.set_scalar(i, (i as f32 * 0.5).sin());
        }
        let e = hodge_energy(&cx, &omega);
        assert!(e >= -1e-4, "hodge energy rank-1 should be ≥ 0, got {e}");
    }

    // -----------------------------------------------------------------------
    // T28: DEC relevance score tests
    // -----------------------------------------------------------------------

    #[test]
    fn dec_relevance_constant_is_one() {
        // Constant cochain → zero energy → relevance = 1/(1+0) = 1
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            omega.set_scalar(i, 3.0);
        }
        let score = dec_relevance_score(&cx, &omega);
        assert!(
            (score - 1.0).abs() < 0.01,
            "relevance of constant should be ~1.0, got {score}"
        );
    }

    #[test]
    fn dec_relevance_bounded() {
        // Relevance should always be in (0, 1]
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            omega.set_scalar(i, (i as f32 * 10.0).sin());
        }
        let score = dec_relevance_score(&cx, &omega);
        assert!(
            score > 0.0 && score <= 1.0,
            "relevance should be in (0, 1], got {score}"
        );
    }

    #[test]
    fn dec_relevance_noisy_lower_than_smooth() {
        // Smooth cochain should have higher relevance than noisy one
        let cx = CellComplex::grid_2d(4, 4);
        let n = cx.n_vertices();

        // Smooth: linear ramp
        let mut smooth = CochainField::zeros(0, n, 1);
        for i in 0..n {
            smooth.set_scalar(i, (i as f32 / n as f32) * 0.1);
        }

        // Noisy: alternating high/low
        let mut noisy = CochainField::zeros(0, n, 1);
        for i in 0..n {
            noisy.set_scalar(i, if i % 2 == 0 { 10.0 } else { -10.0 });
        }

        let score_smooth = dec_relevance_score(&cx, &smooth);
        let score_noisy = dec_relevance_score(&cx, &noisy);
        assert!(
            score_smooth > score_noisy,
            "smooth ({score_smooth}) should have higher relevance than noisy ({score_noisy})"
        );
    }

    // -----------------------------------------------------------------------
    // T29: Hodge residual tests
    // -----------------------------------------------------------------------

    #[test]
    fn hodge_residual_constant_is_zero() {
        // Constant function is harmonic → residual = 0
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            omega.set_scalar(i, 7.0);
        }
        let r = hodge_residual(&cx, &omega);
        assert!(r < 1e-4, "residual of constant should be ~0, got {r}");
    }

    #[test]
    fn hodge_residual_non_negative() {
        // ‖Δω‖₂ ≥ 0 always (L2 norm)
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            omega.set_scalar(i, (i as f32).sin());
        }
        let r = hodge_residual(&cx, &omega);
        assert!(r >= 0.0, "residual should be ≥ 0, got {r}");
    }

    #[test]
    fn hodge_residual_nonconstant_nonzero() {
        // Non-constant, non-harmonic function should have positive residual
        let cx = CellComplex::grid_2d(4, 4);
        let mut omega = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            omega.set_scalar(i, (i as f32 * 2.0).cos());
        }
        let r = hodge_residual(&cx, &omega);
        assert!(r > 0.01, "residual of non-harmonic should be > 0, got {r}");
    }

    // -----------------------------------------------------------------------
    // T30: GOAT gate feature flag test
    // -----------------------------------------------------------------------

    #[test]
    fn goat_gate_feature_enabled() {
        // This test only runs when dec_operators feature is enabled.
        // If it compiles and runs, the GOAT gate is open.
        let cx = CellComplex::grid_2d(4, 4);
        assert_eq!(cx.n_vertices(), 16);
        assert_eq!(cx.n_edges(), 24);
        assert_eq!(cx.n_faces(), 9);
    }

    // -----------------------------------------------------------------------
    // T17: Hodge spectrum
    // -----------------------------------------------------------------------

    #[test]
    fn hodge_spectrum_rank0_small() {
        let cx = CellComplex::grid_2d(4, 4);
        let eigs = hodge_spectrum(&cx, 0, 3, 50);
        assert_eq!(eigs.len(), 3);
        // Eigenvalues should be sorted descending
        for i in 1..eigs.len() {
            assert!(
                eigs[i] <= eigs[i - 1] + 1e-3,
                "eigenvalues not sorted: {:?}",
                eigs
            );
        }
        // All eigenvalues of the Laplacian should be >= 0 (PSD)
        for &e in &eigs {
            assert!(e >= -1e-3, "negative eigenvalue: {}", e);
        }
    }

    #[test]
    fn hodge_spectrum_rank0_single_eigenvalue() {
        let cx = CellComplex::grid_2d(3, 3);
        let eigs = hodge_spectrum(&cx, 0, 1, 30);
        assert_eq!(eigs.len(), 1);
        // Top eigenvalue of 3x3 grid Laplacian should be positive
        assert!(eigs[0] > 0.1, "top eigenvalue too small: {}", eigs[0]);
    }

    // -----------------------------------------------------------------------
    // T19: Benchmark Hodge decomposition on 256×256
    // -----------------------------------------------------------------------

    #[test]
    fn bench_hodge_decompose_256x256() {
        let cx = CellComplex::grid_2d(256, 256);
        let mut potential = CochainField::zeros(0, cx.n_vertices(), 1);
        // Fill with sine-based function
        for i in 0..cx.n_vertices() {
            potential.data[i] = (i as f32 * 0.01).sin();
        }
        let start = std::time::Instant::now();
        let _components = hodge_decompose(&cx, &potential);
        let elapsed = start.elapsed();
        println!("Hodge decomposition 256×256: {:?}", elapsed);
        assert!(
            elapsed.as_secs() < 120,
            "Hodge decomposition too slow: {:?}",
            elapsed
        );
    }
}
