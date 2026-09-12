//! Spectral substrate for slice_tca (T1.2 / T1.3): per-axis Gram matrices of
//! the three unfoldings, the covariability shares, sigmoid routing, and the
//! single-class truncated-SVD fit.
//!
//! # Why the Gram reduction
//!
//! Every entry point here consumes [`crate::subspace_phase_gate::thin_svd_into`]
//! — the same spectral substrate as `linalg::tucker` — applied to the
//! *small-side* Gram matrix `G_σ = X_(σ) X_(σ)ᵀ = U Σ² Uᵀ` instead of the
//! unfolding itself. For a `d_σ × rest` unfolding with `d_σ ≪ rest`, the
//! one-sided-Jacobi cost drops from `O(rest² · d_σ)` (pairs of long columns)
//! to `O(d_σ² · rest)` (building G) + `O(d_σ³)` (a small SVD): the
//! `[64,128,32]` gate shape costs ~30M FMA total instead of seconds. The
//! left singular vectors of `X_(σ)` are read directly from `U(G_σ)`; the
//! right side is recovered by one contraction `Y_r = u_rᵀ X_(σ)`.
//!
//! Accuracy caveat (honest): the Gram squares the condition number, so tiny
//! singular values lose ~half their digits. Components below the
//! `ENERGY_FLOOR_TAU` absolute floor are trimmed anyway; the kept spectrum is
//! accurate to f32 round-off times the squared condition of the KEPT block.
//! The paper's own fitter (SGD) is far less accurate; this is a documented
//! trade, not a defect.

use crate::simd::{simd_fused_scale_acc, simd_gram_f32};
use crate::subspace_phase_gate::{numerical_rank, thin_svd_into};

use super::types::{
    NORM_EPS, SliceClass, SliceDecomposition, SliceTcaError, SliceTcaScratch, frob_sq, norm_2,
};

// ─── Unfolding + Gram builders ──────────────────────────────────────────────

/// Materialize the mode-`axis` unfolding of `x` into `scratch.unfold`:
/// a `d_σ × rest` row-major matrix whose columns enumerate the two
/// complementary axes in ascending order. Returns `(d_σ, rest)`.
pub(crate) fn unfold_into(
    x: &[f32],
    shape: [usize; 3],
    axis: usize,
    scratch: &mut SliceTcaScratch,
) -> (usize, usize) {
    let [n, t, k] = shape;
    let total = n * t * k;
    let out = &mut scratch.unfold[..total];
    match axis {
        0 => {
            // Row i is the contiguous (t × k) slab.
            out.copy_from_slice(x);
            (n, t * k)
        }
        1 => {
            // Row p, column (i·k + l).
            let rest = n * k;
            for i in 0..n {
                for p in 0..t {
                    let src = (i * t + p) * k;
                    let dst = p * rest + i * k;
                    out[dst..dst + k].copy_from_slice(&x[src..src + k]);
                }
            }
            (t, rest)
        }
        _ => {
            // Row l, column (i·t + p) — strided gather.
            let rest = n * t;
            for l in 0..k {
                let row = l * rest;
                for col in 0..rest {
                    out[row + col] = x[col * k + l];
                }
            }
            (k, rest)
        }
    }
}

/// `G_σ = X_(σ) X_(σ)ᵀ` computed directly from `x` into `scratch.gram`
/// (`d_σ × d_σ` row-major). Returns `d_σ`. Uses `simd_gram_f32` on the
/// natural contiguous views: axis 0 grams the `n × (t·k)` row layout in one
/// call; axes 1/2 accumulate per-entity Grams (axis 1 on the contiguous
/// `t × k` slabs; axis 2 on their transposes).
pub(crate) fn gram_axis_into(
    x: &[f32],
    shape: [usize; 3],
    axis: usize,
    scratch: &mut SliceTcaScratch,
) -> usize {
    let [n, t, k] = shape;
    let tk = t * k;
    let d = match axis {
        0 => n,
        1 => t,
        _ => k,
    };
    match axis {
        0 => {
            simd_gram_f32(x, n, tk, &mut scratch.gram[..n * n]);
        }
        1 => {
            let g = &mut scratch.gram[..t * t];
            g.fill(0.0);
            let buf = &mut scratch.gram_acc[..t * t];
            for i in 0..n {
                simd_gram_f32(&x[i * tk..(i + 1) * tk], t, k, buf);
                for (g_el, &b_el) in g.iter_mut().zip(buf.iter()) {
                    *g_el += b_el;
                }
            }
        }
        _ => {
            let g = &mut scratch.gram[..k * k];
            g.fill(0.0);
            let trans = &mut scratch.trans[..tk];
            let buf = &mut scratch.gram_acc[..k * k];
            for i in 0..n {
                // Transpose the (t × k) slab into (k × t), then gram it.
                for p in 0..t {
                    for l in 0..k {
                        trans[l * t + p] = x[i * tk + p * k + l];
                    }
                }
                simd_gram_f32(trans, k, t, buf);
                for (g_el, &b_el) in g.iter_mut().zip(buf.iter()) {
                    *g_el += b_el;
                }
            }
        }
    }
    d
}

/// Stash the σ² spectra of all three unfoldings into `scratch.spec`
/// (descending, clamped ≥ 0), cache `‖X‖²_F` in `scratch.norm_sq`, and cache
/// each axis's top singular-vector columns (column-major
/// `d_σ × MAX_RANK_PER_CLASS`) into `scratch.u_cache` for the fit phase.
/// One shared denominator, one fixed evaluation order.
pub(crate) fn spectra_into(x: &[f32], shape: [usize; 3], scratch: &mut SliceTcaScratch) -> f32 {
    let norm_sq = frob_sq(x);
    scratch.norm_sq = norm_sq;
    for axis in 0..3 {
        let d = gram_axis_into(x, shape, axis, scratch);
        let g_len = d * d;
        thin_svd_into(
            &scratch.gram[..g_len],
            d,
            d,
            &mut scratch.svd_result,
            &mut scratch.svd_work,
        );
        let len = scratch.svd_result.len().min(d);
        let sv = scratch.svd_result.singular_values();
        let spec = &mut scratch.spec[axis][..len];
        for (dst, &v) in spec.iter_mut().zip(sv.iter()) {
            *dst = v.max(0.0);
        }
        cache_u_columns(scratch, axis, d);
    }
    norm_sq
}

/// Copy the top-`min(d, MAX_RANK_PER_CLASS)` left singular vectors of the
/// SVD result into `u_cache[axis]` (CONTIGUOUS column-major,
/// `MAX_RANK_PER_CLASS × d`: column `r` at `[r·d .. (r+1)·d]`) and mark it
/// valid.
pub(crate) fn cache_u_columns(scratch: &mut SliceTcaScratch, axis: usize, d: usize) {
    let cols = scratch
        .svd_result
        .len()
        .min(d)
        .min(super::types::MAX_RANK_PER_CLASS);
    let cache_len = super::types::MAX_RANK_PER_CLASS * d;
    let cache = &mut scratch.u_cache[axis][..cache_len];
    cache.fill(0.0);
    for r in 0..cols {
        let col = scratch.svd_result.left_singular_vector(r);
        cache[r * d..(r + 1) * d].copy_from_slice(col);
    }
    scratch.u_cache_valid[axis] = true;
}

// ─── T1.3 — covariability shares + routing ──────────────────────────────────

/// Covariability shares (T1.3): `[EVR_n, EVR_t, EVR_k]` where
/// `EVR_σ = Σ_{r<R} σ_r²(X_(σ)) / ‖X‖²_F` — the top-R spectral energy of each
/// unfolding over the shared Frobenius denominator. Zero-alloc with a
/// pre-warmed `scratch`. The shares are NOT exclusive (they do not sum to 1);
/// that is precisely why routing is a sigmoid per class.
pub fn covariability_shares_into(
    x: &[f32],
    shape: [usize; 3],
    r: usize,
    scratch: &mut SliceTcaScratch,
) -> [f32; 3] {
    let total = shape[0] * shape[1] * shape[2];
    debug_assert_eq!(x.len(), total, "input length mismatch");
    if x.iter().all(|&v| v == 0.0) {
        scratch.norm_sq = 0.0;
        for axis in 0..3 {
            scratch.spec[axis].fill(0.0);
        }
        return [0.0; 3];
    }
    let norm_sq = spectra_into(x, shape, scratch);
    if norm_sq < NORM_EPS {
        return [0.0; 3];
    }
    let mut shares = [0.0f32; 3];
    for (axis, spec) in scratch.spec.iter().enumerate() {
        let take = r.min(spec.len());
        let mut s = 0.0f32;
        for &lambda in spec.iter().take(take) {
            s += lambda;
        }
        shares[axis] = s / norm_sq;
    }
    shares
}

/// Allocating convenience wrapper for one-shot use.
pub fn covariability_shares(x: &[f32], shape: [usize; 3], r: usize) -> [f32; 3] {
    let mut scratch = SliceTcaScratch::with_capacity(shape);
    covariability_shares_into(x, shape, r, &mut scratch)
}

/// Per-class routing gate: `route_σ = sigmoid(α·(EVR_σ − θ))` (T1.3).
///
/// SIGMOID, never softmax — the three covariability classes are not mutually
/// exclusive: a tensor can be simultaneously entity- and time-covariable
/// (that mixed case is the whole point of the joint demixer). Softmax would
/// suppress exactly the co-occurrence the classifier must surface.
pub fn route(shares: [f32; 3], theta: f32, alpha: f32) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for axis in 0..3 {
        out[axis] = crate::sigmoid(alpha * (shares[axis] - theta));
    }
    out
}

/// [`route`] with the calibrated defaults (`ROUTE_THETA`, `ROUTE_ALPHA`).
pub fn route_default(shares: [f32; 3]) -> [f32; 3] {
    route(
        shares,
        super::types::ROUTE_THETA,
        super::types::ROUTE_ALPHA,
    )
}

// ─── T1.2 — single-class fit ────────────────────────────────────────────────

/// Fit ONE class (axis) with `rank` components: the truncated SVD of the
/// mode-`axis` unfolding, expressed as `loading_r ⊗ slice_r` components with
/// unit-norm slices and amplitude in the loadings — the paper's own
/// reduction (single slice type ≡ unfolding matrix factorization), so this
/// is the Eckart–Young global optimum for the pure-class problem.
/// Deterministic trim: components are taken while
/// `σ_r² ≥ energy_floor_tau · ‖X‖²_F` (fixed order, early stop). Appends to
/// `out` (does not clear it — the joint init calls it once per active class).
///
/// Always factors this axis fresh (Gram + SVD) — safe for standalone callers
/// on a reused scratch.
pub fn fit_single_class_into(
    x: &[f32],
    shape: [usize; 3],
    axis: usize,
    rank: usize,
    energy_floor_tau: f32,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    validate_class_fit(x, shape, axis, rank)?;
    factor_axis_fresh(x, shape, axis, scratch);
    assemble_class_from_cache(
        x,
        shape,
        axis,
        rank,
        energy_floor_tau * frob_sq(x),
        scratch,
        out,
    )
}

/// Crate-internal variant for the joint fit pipeline: skips the per-axis
/// Gram+SVD when `u_cache` is valid — the caller (`fit_with_ranks_into`)
/// ran `covariability_shares_into` on the SAME tensor immediately before,
/// so the cached basis is correct by construction.
pub(crate) fn fit_single_class_cached_into(
    x: &[f32],
    shape: [usize; 3],
    axis: usize,
    rank: usize,
    energy_floor_tau: f32,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    validate_class_fit(x, shape, axis, rank)?;
    if !scratch.u_cache_valid[axis] {
        factor_axis_fresh(x, shape, axis, scratch);
    }
    assemble_class_from_cache(
        x,
        shape,
        axis,
        rank,
        energy_floor_tau * frob_sq(x),
        scratch,
        out,
    )
}

fn validate_class_fit(
    _x: &[f32],
    shape: [usize; 3],
    axis: usize,
    rank: usize,
) -> Result<(), SliceTcaError> {
    let [n, t, k] = shape;
    let d = [n, t, k][axis];
    let rest = n * t * k / d;
    let bound = d.min(rest).min(super::types::MAX_RANK_PER_CLASS);
    if rank == 0 {
        return Ok(());
    }
    if rank > bound {
        return Err(SliceTcaError::RankTooLarge { axis, rank, bound });
    }
    Ok(())
}

/// Factor an axis's Gram and cache its basis (works directly from `x`; no
/// unfolding materialization needed).
fn factor_axis_fresh(x: &[f32], shape: [usize; 3], axis: usize, scratch: &mut SliceTcaScratch) {
    let d = gram_axis_into(x, shape, axis, scratch);
    let g_len = d * d;
    thin_svd_into(
        &scratch.gram[..g_len],
        d,
        d,
        &mut scratch.svd_result,
        &mut scratch.svd_work,
    );
    cache_u_columns(scratch, axis, d);
}

/// Assemble `rank` class-`axis` components from the (re-materialized)
/// unfolding and the cached basis.
fn assemble_class_from_cache(
    x: &[f32],
    shape: [usize; 3],
    axis: usize,
    rank: usize,
    floor_sq: f32,
    scratch: &mut SliceTcaScratch,
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    let (d, rest) = unfold_into(x, shape, axis, scratch);
    let n_cols = rank.min(d);
    push_class_comps_into(
        shape,
        axis,
        n_cols,
        floor_sq,
        &scratch.u_cache[axis][..super::types::MAX_RANK_PER_CLASS * d],
        &scratch.unfold[..d * rest],
        &mut scratch.y_vec[..d * rest],
        &mut scratch.slice_buf[..d * rest],
        &mut scratch.loading_buf[..d],
        out,
    )
}

/// Shared component assembly: given the mode-`axis` unfolding in `unfold`
/// and an orthonormal `u_r` basis (`basis_cols`, column-major `d × n_cols`:
/// column `r` at `basis_cols[r·d .. (r+1)·d]`), push up to `n_cols` class-
/// `axis` components with unit-norm slices and `loading_r = σ_r · u_r`,
/// where `σ_r = ‖Y_r‖` with `Y_r = u_rᵀ · unfold` (an exact identity, so the
/// Gram and HOSVD basis paths trim identically).
/// Deterministic trim: fixed r order, stop at the first `σ_r² < floor_sq`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_class_comps_into(
    shape: [usize; 3],
    axis: usize,
    n_cols: usize,
    floor_sq: f32,
    basis_cols: &[f32],
    unfold: &[f32],
    y_vec: &mut [f32],
    slice_buf: &mut [f32],
    loading_buf: &mut [f32],
    out: &mut SliceDecomposition,
) -> Result<(), SliceTcaError> {
    let [n, t, k] = shape;
    let d = [n, t, k][axis];
    let rest = n * t * k / d;
    let class = SliceClass::from_axis(axis).ok_or(SliceTcaError::ZeroDimension)?;
    let (slice_rows, slice_cols) = complementary_dims(shape, class);
    let mut r = 0usize;
    while r < n_cols {
        let u = &basis_cols[r * d..(r + 1) * d];
        debug_assert_eq!(u.len(), d);
        // Y_r = uᵀ · unfold (one contraction; fixed row order).
        let y = &mut y_vec[..rest];
        y.fill(0.0);
        for i in 0..d {
            let s = u[i];
            simd_fused_scale_acc(y, &unfold[i * rest..(i + 1) * rest], s, rest);
        }
        // σ_r = ‖Y_r‖ exactly (u orthonormal). Normalize: slice unit-norm,
        // amplitude in the loading.
        let sigma = norm_2(y);
        if sigma * sigma < floor_sq || sigma <= NORM_EPS {
            break; // deterministic trim: fixed order, early stop
        }
        let inv = 1.0 / sigma;
        let slice = &mut slice_buf[..rest];
        for (dst, &src) in slice.iter_mut().zip(y.iter()) {
            *dst = src * inv;
        }
        let loading = &mut loading_buf[..d];
        for i in 0..d {
            loading[i] = sigma * u[i];
        }
        out.push_component(class, &loading[..d], slice, slice_rows, slice_cols, sigma)?;
        r += 1;
    }
    Ok(())
}

/// Rows/cols of a class's slice matrix for `shape` (ascending axes).
pub(crate) fn complementary_dims(shape: [usize; 3], class: SliceClass) -> (usize, usize) {
    let (a, b) = class.slice_axes();
    (shape[a], shape[b])
}

/// Effective-rank cap from the substrate's [`numerical_rank`]: the smallest r
/// capturing `eta` of the unfolding's spectral energy, computed on the σ
/// scale (square roots of the stashed σ² spectrum). Used by rank selection.
pub(crate) fn numerical_rank_cap(spec_sq: &[f32], eta: f32, sqrt_buf: &mut Vec<f32>) -> usize {
    sqrt_buf.clear();
    for &lambda in spec_sq {
        sqrt_buf.push(lambda.max(0.0).sqrt());
    }
    numerical_rank(sqrt_buf, eta)
}
