//! TILR — Trajectory-Invariant Latent Refinement (alignment-gated subspace correction).
//!
//! Distilled from Malarkkan et al., *TILR: Trajectory-Invariant Latent Refinement*
//! ([arXiv:2606.29164](https://arxiv.org/abs/2606.29164), ICML 2026 Mech Interp
//! Workshop). See `katgpt-rs/.research/408_*.md` for the open research note and
//! `katgpt-rs/.plans/425_*.md` for the execution plan.
//!
//! # The mechanism (5 steps, Research 408 §1.2)
//!
//! 1. **Contrastive differences** `δ_t = h_good − h_bad` are collected offline from a
//!    frozen reference pair (two checkpoints from different epochs). This is a
//!    consumer concern — the primitive CONSUMES a pre-computed SVD basis, it does
//!    not run the calibration.
//! 2. **Truncated SVD** `Δ ≈ U_r Σ_r V_rᵀ` retains `τ=0.90` variance → invariant
//!    subspace `S_r = span(U_r)`. Done offline via [`thin_svd_into`] (Plan 301).
//! 3. **Project** the per-instance contrastive direction `d` onto `S_r`:
//!    `d_proj = U_r (U_rᵀ d)` — an `O(d·r)` matrix-vector product.
//! 4. **Alignment gate** `γ = ‖d_proj‖ / ‖d‖ ∈ [0,1]`. When `γ→0` the direction is
//!    orthogonal to the invariant subspace and the correction vanishes.
//! 5. **Apply** `s' = s + η_base · γ · d_proj`. The step size is modulated by the
//!    alignment fraction so `γ→0` bit-recovers `s` (strict no-harm guarantee).
//!
//! # The no-harm contract (load-bearing)
//!
//! When `γ = 0` (direction orthogonal to basis), `η = 0` and `s' = s`
//! **bit-identically**. When `γ = 1` (direction ∈ span(basis)), `η = η_base` and
//! `s' = s + η_base · d`. This is enforced by clamping `‖d_proj‖² < epsilon → γ = 0`
//! exactly (not `γ ≈ 1e-38`), so `η` is exactly `0.0`.
//!
//! # Reuse map (do not duplicate)
//!
//! | Operation | Source | Notes |
//! |---|---|---|
//! | SVD → `U_r` (offline calibration) | `thin_svd_into` (Plan 301, `subspace_phase_gate`) | The basis `U_r` is an INPUT to this primitive, not computed here |
//! | Alignment ratio `γ` | `subspace_ratios` (Plan 152, `katgpt-spectral/river_valley`) | The metric is identical (`r_dom = ‖U_kᵀg‖/‖g‖`); duplicated here (~5 lines) per Plan 425 DRY decision (B) to avoid a leaf→katgpt-spectral refactor |
//! | Subspace projection `Πd` | `spectral_rewire` (Plan 423), `subspace_steering` (Plan 412) | Same projection math; TILR adds the γ-gated step size |
//! | SIMD dot products | [`simd_dot_f32`] (`katgpt-types/simd`) | Used for projection coefficients + norms |
//!
//! TILR is the **alignment-gated** member of the subspace-projection family:
//! - Plan 412 (`subspace_steering`) = ungated (fixed `α` per axis).
//! - Plan 423 (`spectral_rewire`) = ungated (fixed projection, no step modulation).
//! - TILR = γ-gated (step size scales with subspace alignment, no-harm at γ=0).
//!
//! # Allocation
//!
//! [`tilr_refine_into`] is zero-alloc on the hot path — the caller pre-allocates a
//! [`TilrScratch`] once and reuses it across calls. [`tilr_refine`] is the
//! allocating convenience wrapper for non-hot paths.
//!
//! TILR — Trajectory-Invariant Latent Refinement (alignment-gated subspace
//! correction). Plan 425, Research 408, arXiv:2606.29164 (ICML 2026 Mech Interp
//! Workshop). The alignment-gated member of the subspace-projection family:
//! projects a contrastive direction onto a frozen SVD basis, modulates the step
//! size by the alignment fraction γ = ‖Πd‖/‖d‖ so that γ→0 bit-recovers the
//! uncorrected input (strict no-harm guarantee). Pure linear algebra — flat
//! &[f32] slices + SIMD dot products, zero `crate::` deps. Consumes a
//! pre-computed SVD basis (Plan 301 thin_svd_into); does not compute it.
//! DEFAULT-ON (2026-07-09): G1–G4 ALL PASS — see .benchmarks/425_tilr_goat.md.
//!
//! Phase 3 calibration helper needs Plan 301's thin SVD — gated on both features.

use crate::simd::{simd_dot_f32, simd_fused_scale_acc};

// Phase 3 calibration helper needs Plan 301's thin SVD. Gated separately so the
// core primitive (tilr_refine_into) stays zero-`crate::`-dep. In the default
// build, `subspace_phase_gate` is transitively enabled via `tucker_factorization`
// / `viable_manifold_graph` (both default-on).
#[cfg(feature = "subspace_phase_gate")]
use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by this module's functions.
///
/// `NotOrthonormal` is returned by [`check_orthonormal`] (setup-time validation).
/// [`tilr_refine_into`] and [`tilr_refine`] only return `EtaOutOfRange` and
/// `DimensionMismatch` — they trust the basis (validated once at setup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TilrError {
    /// `basis` columns are not mutually orthonormal within `tol` (returned by
    /// [`check_orthonormal`], NOT by the hot-path refinement functions).
    NotOrthonormal = 0,
    /// `eta_base` is outside `[0.0, 1.0]`.
    EtaOutOfRange = 1,
    /// `state`, `direction`, `out`, and basis row-dimension `d` disagree, or
    /// `basis.len() != r * d`.
    DimensionMismatch = 2,
    /// `tau` is outside `(0.0, 1.0]` (returned by [`discover_invariant_subspace`]).
    InvalidTau = 3,
    /// All contrastive differences are zero — no variance to decompose (returned
    /// by [`discover_invariant_subspace`]).
    ZeroVariance = 4,
}

// ── Scratch ─────────────────────────────────────────────────────────────────

/// Scratch buffer for zero-alloc TILR refinement.
///
/// Holds the projection coefficients `U_rᵀd ∈ ℝ^r` and the projected direction
/// `d_proj ∈ ℝ^d`. Allocate once via [`TilrScratch::with_capacity`] and reuse
/// across calls — [`tilr_refine_into`] rewrites the contents without growing.
///
/// # Example
///
/// ```
/// use katgpt_core::tilr::{tilr_refine_into, TilrScratch};
///
/// let d = 8;
/// let r = 2;
/// let mut scratch = TilrScratch::with_capacity(d, r);
/// // ... reuse `scratch` across many `tilr_refine_into` calls ...
/// # let _ = (d, r, scratch);
/// ```
#[derive(Debug, Clone)]
pub struct TilrScratch {
    /// Projection coefficients `U_rᵀd`, length `r`.
    coeffs: Vec<f32>,
    /// Projected direction `d_proj = U_r (U_rᵀd)`, length `d`.
    d_proj: Vec<f32>,
}

impl TilrScratch {
    /// Pre-allocate scratch for a latent dimension `d` and basis rank `r`.
    ///
    /// The buffers are zero-initialized; [`tilr_refine_into`] rewrites them
    /// completely on each call.
    #[inline]
    #[must_use]
    pub fn with_capacity(d: usize, r: usize) -> Self {
        Self {
            coeffs: vec![0.0; r],
            d_proj: vec![0.0; d],
        }
    }

    /// The latent dimension `d` this scratch was allocated for.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.d_proj.len()
    }

    /// The basis rank `r` this scratch was allocated for.
    #[inline]
    #[must_use]
    pub fn rank(&self) -> usize {
        self.coeffs.len()
    }
}

// ── Setup-time validation ───────────────────────────────────────────────────

/// Validate that `basis` (row-major `r × d`, i.e. `basis[k*d + i]` = component `i`
/// of basis vector `k`) is a set of `r` mutually orthonormal vectors of length `d`.
///
/// Checks:
/// - Each basis vector `k` has unit norm within `tol`: `|‖u_k‖ − 1| ≤ tol`.
/// - Each pair `(i, j)` with `i ≠ j` has `|⟨u_i, u_j⟩| ≤ tol`.
///
/// **Call this once at setup** (after SVD / orthogonalization), not on the hot
/// path — the check is `O(r² · d)`. [`tilr_refine_into`] trusts the basis.
///
/// Returns [`TilrError::DimensionMismatch`] if `basis.len() != r * d` or `r == 0`
/// or `d == 0`. Returns [`TilrError::NotOrthonormal`] on a failed check.
///
/// # Example
///
/// ```
/// use katgpt_core::tilr::check_orthonormal;
///
/// // 2 orthonormal vectors of length 4 (standard basis e_0, e_1).
/// let basis = [
///     1.0, 0.0, 0.0, 0.0,
///     0.0, 1.0, 0.0, 0.0,
/// ];
/// assert!(check_orthonormal(&basis, 2, 4, 1e-5).is_ok());
/// ```
pub fn check_orthonormal(basis: &[f32], r: usize, d: usize, tol: f32) -> Result<(), TilrError> {
    if r == 0 || d == 0 || basis.len() != r * d {
        return Err(TilrError::DimensionMismatch);
    }
    // Each basis vector must be unit-norm.
    for k in 0..r {
        let row = &basis[k * d..(k + 1) * d];
        let norm = sqrt_dot(row, row);
        if (norm - 1.0).abs() > tol {
            return Err(TilrError::NotOrthonormal);
        }
    }
    // Each pair must be orthogonal.
    for i in 0..r {
        let u_i = &basis[i * d..(i + 1) * d];
        for j in (i + 1)..r {
            let u_j = &basis[j * d..(j + 1) * d];
            let dot = simd_dot_f32(u_i, u_j, d);
            if dot.abs() > tol {
                return Err(TilrError::NotOrthonormal);
            }
        }
    }
    Ok(())
}

// ── Core primitive ──────────────────────────────────────────────────────────

/// Alignment-gated subspace-projected correction (the TILR primitive).
///
/// Applies `s' = s + η_base · γ · d_proj` where:
/// - `d_proj = U_r (U_rᵀ d)` — projection of `direction` onto the invariant
///   subspace spanned by the `r` orthonormal columns of `basis`.
/// - `γ = ‖d_proj‖ / (‖d‖ + ε)` — the alignment fraction ∈ `[0, 1]`.
/// - `η = η_base · γ` — the gated step size.
///
/// **No-harm contract:** when `γ = 0` (direction orthogonal to basis), `s' = s`
/// bit-identically. When `γ = 1` (direction ∈ span(basis)), `s' = s + η_base · d`.
///
/// # Arguments
///
/// - `state` — the latent state `s ∈ ℝ^d` to correct (length `d`).
/// - `direction` — the per-instance contrastive direction `d ∈ ℝ^d` (length `d`).
/// - `basis` — `r` orthonormal column-vectors of length `d`, row-major flat
///   (`basis[k*d + i]` = component `i` of basis vector `k`). Validate once via
///   [`check_orthonormal`]; this function trusts the basis.
/// - `r` — the number of basis vectors (rank of the invariant subspace).
/// - `eta_base` — the base step size ∈ `[0, 1]`.
/// - `epsilon` — numerical guard for the norm ratio (default `1e-12`).
/// - `scratch` — pre-allocated [`TilrScratch`] (reused across calls, zero-alloc).
/// - `out` — the output buffer (length `d`). Must NOT alias `state` (the
///   safe `&[f32]` + `&mut [f32]` API enforces this). For in-place semantics,
///   use [`tilr_refine_apply`] which takes `&mut [f32]` only.
///
/// # Returns
///
/// `Ok(γ)` — the alignment fraction ∈ `[0, 1]`, for diagnostic/logging use.
///
/// # Errors
///
/// - [`TilrError::DimensionMismatch`] — `state`, `direction`, `out` lengths
///   disagree, or `basis.len() != r * d`, or `scratch` is undersized.
/// - [`TilrError::EtaOutOfRange`] — `eta_base` outside `[0, 1]`.
///
/// # Cost
///
/// `O(d · r)` for the projection (two matvecs) + `O(d)` for the final SAXPY.
/// For `d=768, r=12` this is ~9.2K FMAs — negligible vs `O(d²)` attention.
///
/// # Example
///
/// ```
/// use katgpt_core::tilr::{check_orthonormal, tilr_refine_into, TilrScratch};
///
/// let d = 4;
/// let r = 2;
/// // Basis = {e_0, e_1} (invariant subspace = first 2 dims).
/// let basis = [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0];
/// check_orthonormal(&basis, r, d, 1e-5).unwrap();
///
/// let state = [0.1, 0.2, 0.3, 0.4];
/// // Direction entirely in span(basis): γ should be 1.0.
/// let direction = [0.5, 0.0, 0.0, 0.0];
/// let mut out = [0.0f32; 4];
/// let mut scratch = TilrScratch::with_capacity(d, r);
/// let gamma = tilr_refine_into(&state, &direction, &basis, r, 0.5, 1e-12, &mut scratch, &mut out).unwrap();
/// assert!((gamma - 1.0).abs() < 1e-5);
/// // s' = s + 0.5 * 1.0 * d_proj = s + 0.5 * (0.5, 0, 0, 0)
/// assert!((out[0] - 0.35).abs() < 1e-5);
/// ```
#[inline]
#[allow(clippy::too_many_arguments)] // buffer-passing API: each arg is a distinct input/output slice or scalar
pub fn tilr_refine_into(
    state: &[f32],
    direction: &[f32],
    basis: &[f32],
    r: usize,
    eta_base: f32,
    epsilon: f32,
    scratch: &mut TilrScratch,
    out: &mut [f32],
) -> Result<f32, TilrError> {
    let d = state.len();
    // ── Dimension + range validation (O(1), no orthonormality check) ──────
    if direction.len() != d
        || out.len() != d
        || basis.len() != r * d
        || scratch.coeffs.len() < r
        || scratch.d_proj.len() < d
    {
        return Err(TilrError::DimensionMismatch);
    }
    if !(0.0..=1.0).contains(&eta_base) {
        return Err(TilrError::EtaOutOfRange);
    }

    // ── Step 1: projection coefficients coeffs[k] = ⟨u_k, d⟩ ──────────────
    // O(r · d) — one dot product per basis vector.
    let coeffs = &mut scratch.coeffs[..r];
    for k in 0..r {
        coeffs[k] = simd_dot_f32(&basis[k * d..(k + 1) * d], direction, d);
    }

    // ── Step 2: projected direction d_proj[i] = Σ_k coeffs[k] · u_k[i] ────
    // O(r · d) — reconstruct from coefficients. Written as outer SAXPY loop
    // for auto-vectorization on the inner d-loop.
    let d_proj = &mut scratch.d_proj[..d];
    d_proj.fill(0.0);
    for k in 0..r {
        let c = coeffs[k];
        let row = &basis[k * d..(k + 1) * d];
        simd_fused_scale_acc(d_proj, row, c, d);
    }

    // ── Step 3: alignment ratio γ = ‖d_proj‖ / ‖d‖ ────────────────────────
    // cf. river_valley::subspace_ratios (Plan 152) — the identical metric.
    // We compute via squared norms to avoid two sqrts, then one sqrt at the end.
    let d_proj_norm_sq = simd_dot_f32(d_proj, d_proj, d);
    let d_norm_sq = simd_dot_f32(direction, direction, d);
    // Strict no-harm: when ‖d_proj‖² < epsilon, γ = 0.0 exactly (not ≈1e-38).
    // This guarantees η = 0.0 exactly → s' = s bit-identically.
    let gamma = if d_proj_norm_sq < epsilon || d_norm_sq < epsilon {
        0.0
    } else {
        let g = (d_proj_norm_sq / (d_norm_sq + epsilon)).sqrt();
        // Numerical safety: clamp to [0, 1] (ratio can exceed 1.0 by rounding
        // when d_proj ≈ d, or when basis is not perfectly orthonormal).
        g.min(1.0)
    };

    // ── Step 4: gated step size η = η_base · γ ────────────────────────────
    let eta = eta_base * gamma;

    // ── Step 5: apply s' = s + η · d_proj ─────────────────────────────────
    // O(d) SAXPY: copy state, then accumulate η · d_proj via SIMD.
    out[..d].copy_from_slice(&state[..d]);
    simd_fused_scale_acc(&mut out[..d], d_proj, eta, d);

    Ok(gamma)
}

/// Allocating convenience wrapper for [`tilr_refine_into`] (non-hot paths).
///
/// Allocates the output `Vec<f32>` and a transient scratch internally. Use
/// [`tilr_refine_into`] with a reused [`TilrScratch`] on hot paths.
///
/// # Returns
///
/// `Ok((s', γ))` — the corrected state and the alignment fraction.
///
/// # Errors
///
/// Same as [`tilr_refine_into`].
///
/// # Example
///
/// ```
/// use katgpt_core::tilr::tilr_refine;
///
/// let state = [0.1, 0.2, 0.3, 0.4];
/// let direction = [0.0, 0.0, 0.5, 0.0]; // orthogonal to basis → γ = 0
/// let basis = [1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0]; // {e_0, e_1}
/// let (out, gamma) = tilr_refine(&state, &direction, &basis, 2, 0.5).unwrap();
/// assert_eq!(gamma, 0.0);
/// // No-harm: out == state bit-identically.
/// assert!(out.iter().zip(state.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
/// ```
#[inline]
pub fn tilr_refine(
    state: &[f32],
    direction: &[f32],
    basis: &[f32],
    r: usize,
    eta_base: f32,
) -> Result<(Vec<f32>, f32), TilrError> {
    let d = state.len();
    if direction.len() != d || basis.len() != r * d {
        return Err(TilrError::DimensionMismatch);
    }
    let mut scratch = TilrScratch::with_capacity(d, r);
    let mut out = vec![0.0; d];
    let gamma = tilr_refine_into(
        state,
        direction,
        basis,
        r,
        eta_base,
        1e-12,
        &mut scratch,
        &mut out,
    )?;
    Ok((out, gamma))
}

/// In-place alignment-gated subspace correction — modifies `state` directly.
///
/// Equivalent to [`tilr_refine_into`] with `out == state`, but takes `&mut [f32]`
/// only (no separate immutable `state` borrow) so the borrow checker allows it.
/// The correction `s' = s + η · d_proj` is applied element-wise (read-then-write
/// per index), so aliasing is safe by construction.
///
/// Use this when you want to mutate the state in-place without allocating a
/// separate output buffer. Use [`tilr_refine_into`] when you need to preserve
/// the original state.
///
/// # Returns
///
/// `Ok(γ)` — the alignment fraction ∈ `[0, 1]`.
///
/// # Errors
///
/// Same as [`tilr_refine_into`].
#[inline]
pub fn tilr_refine_apply(
    state: &mut [f32],
    direction: &[f32],
    basis: &[f32],
    r: usize,
    eta_base: f32,
    epsilon: f32,
    scratch: &mut TilrScratch,
) -> Result<f32, TilrError> {
    let d = state.len();
    if direction.len() != d
        || basis.len() != r * d
        || scratch.coeffs.len() < r
        || scratch.d_proj.len() < d
    {
        return Err(TilrError::DimensionMismatch);
    }
    if !(0.0..=1.0).contains(&eta_base) {
        return Err(TilrError::EtaOutOfRange);
    }

    // Steps 1-3: same as tilr_refine_into (compute gamma + d_proj in scratch).
    let coeffs = &mut scratch.coeffs[..r];
    for k in 0..r {
        coeffs[k] = simd_dot_f32(&basis[k * d..(k + 1) * d], direction, d);
    }
    let d_proj = &mut scratch.d_proj[..d];
    d_proj.fill(0.0);
    for k in 0..r {
        let c = coeffs[k];
        let row = &basis[k * d..(k + 1) * d];
        simd_fused_scale_acc(d_proj, row, c, d);
    }
    let d_proj_norm_sq = simd_dot_f32(d_proj, d_proj, d);
    let d_norm_sq = simd_dot_f32(direction, direction, d);
    let gamma = if d_proj_norm_sq < epsilon || d_norm_sq < epsilon {
        0.0
    } else {
        (d_proj_norm_sq / (d_norm_sq + epsilon)).sqrt().min(1.0)
    };
    let eta = eta_base * gamma;

    // Step 5: in-place SAXPY — state[i] += eta * d_proj[i].
    simd_fused_scale_acc(&mut state[..d], d_proj, eta, d);
    Ok(gamma)
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// `sqrt(⟨a, b⟩)` — dot product then sqrt, for norm computations.
#[inline]
fn sqrt_dot(a: &[f32], b: &[f32]) -> f32 {
    simd_dot_f32(a, b, a.len()).sqrt()
}

// ── Offline calibration (Phase 3, P2) ───────────────────────────────────────

/// Offline calibration: discover the invariant subspace `S_r` from contrastive
/// differences via truncated SVD.
///
/// **Runs ONCE at calibration time**, producing the basis `U_r` consumed by
/// [`tilr_refine_into`] at inference time. The online/offline split (T3.3):
/// - **Offline** (this function, once): collect `N` contrastive differences
///   `δ_t = h_good − h_bad`, stack into `Δ`, run `thin_svd_into` (Plan 301),
///   select `r` by variance retention.
/// - **Online** ([`tilr_refine_into`], per step): project the per-instance
///   direction `d` onto the frozen basis, compute `γ`, apply the gated
///   correction. `O(d·r)` per step, zero allocation.
///
/// # Arguments
///
/// - `differences` — `N` contrastive difference vectors, each of length `d`.
///   All must have the same length.
/// - `tau` — variance retention threshold ∈ `(0.0, 1.0]`. The rank `r` is the
///   smallest number of singular values capturing `tau` fraction of the total
///   variance (`Σ σ_j² / Σ_all σ²`). Paper default: `0.90`.
///
/// # Returns
///
/// `Ok((basis, r))` where `basis` is `r` orthonormal vectors of length `d`,
/// flattened row-major (`r × d`), and `r` is the selected rank. Pass `basis`
/// and `r` directly to [`tilr_refine_into`] / [`check_orthonormal`].
///
/// # Errors
///
/// - [`TilrError::DimensionMismatch`] — `differences` is empty, any difference
///   has zero length, or not all differences share the same length.
/// - [`TilrError::InvalidTau`] — `tau` not in `(0.0, 1.0]`.
/// - [`TilrError::ZeroVariance`] — all differences are zero (`Σ σ² ≈ 0`).
///
/// # Cost
///
/// `O(N · d · min(N, d))` for the one-sided Jacobi SVD. Acceptable for a
/// one-time calibration; NOT for the hot path. The right singular vectors of
/// `Δ (N×d)` are the d-dimensional invariant directions (equivalently, the
/// left singular vectors of the transposed `d×N` matrix — the paper's notation).
///
/// # Example
///
/// ```
/// use katgpt_core::tilr::discover_invariant_subspace;
///
/// // Two differences along e_0 in ℝ⁴ → rank-1 invariant subspace.
/// let diffs: Vec<&[f32]> = vec![&[2.0, 0.0, 0.0, 0.0], &[3.0, 0.0, 0.0, 0.0]];
/// let (basis, r) = discover_invariant_subspace(&diffs, 0.95).unwrap();
/// assert_eq!(r, 1);
/// assert_eq!(basis.len(), 4); // 1 vector × d=4
/// ```
#[cfg(feature = "subspace_phase_gate")]
pub fn discover_invariant_subspace(
    differences: &[&[f32]],
    tau: f32,
) -> Result<(Vec<f32>, usize), TilrError> {
    if differences.is_empty() {
        return Err(TilrError::DimensionMismatch);
    }
    if !(0.0 < tau && tau <= 1.0) {
        return Err(TilrError::InvalidTau);
    }
    let d = differences[0].len();
    if d == 0 {
        return Err(TilrError::DimensionMismatch);
    }
    for diff in differences {
        if diff.len() != d {
            return Err(TilrError::DimensionMismatch);
        }
    }

    let n = differences.len();

    // Build Δ as N×d row-major (differences as rows). The right singular vectors
    // of Δ (N×d) are d-dimensional — these span the invariant subspace S_r.
    let mut delta = Vec::with_capacity(n * d);
    for diff in differences {
        delta.extend_from_slice(diff);
    }

    // Thin SVD: Δ = U Σ Vᵀ. U is N×k, V is d×k (k = min(N, d)).
    // NOTE: SvdScratch::with_capacity takes (n_cols, m_rows) — swapped vs
    // SvdResultScratch::with_capacity(m_rows, n_cols).
    let mut work = SvdScratch::with_capacity(d, n);
    let mut result = SvdResultScratch::with_capacity(n, d);
    thin_svd_into(&delta, n, d, &mut result, &mut work);

    // Total variance = Σ σ_j².
    let sv = result.singular_values();
    let total_variance: f32 = sv.iter().map(|s| s * s).sum();
    if total_variance < 1e-30 {
        return Err(TilrError::ZeroVariance);
    }

    // Select r = smallest rank retaining tau fraction of cumulative variance.
    let mut cumul = 0.0f32;
    let mut r = sv.len();
    for (j, &s) in sv.iter().enumerate() {
        cumul += s * s / total_variance;
        if cumul >= tau {
            r = j + 1;
            break;
        }
    }

    // Extract top-r right singular vectors (each length d), flatten as r×d
    // row-major — the format consumed by tilr_refine_into / check_orthonormal.
    let mut basis = Vec::with_capacity(r * d);
    for j in 0..r {
        basis.extend_from_slice(result.right_singular_vector(j));
    }

    Ok((basis, r))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard basis vector `e_i` of length `d`, as a flat row.
    fn e(i: usize, d: usize) -> Vec<f32> {
        let mut v = vec![0.0; d];
        v[i] = 1.0;
        v
    }

    /// Construct a row-major basis from rows.
    fn basis_from_rows(rows: &[&[f32]]) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows.len() * rows[0].len());
        for row in rows {
            out.extend_from_slice(row);
        }
        out
    }

    // ── G1a: no-harm bit-identity (γ = 0) ──────────────────────────────────

    #[test]
    fn g1a_no_harm_bit_identity_when_gamma_zero() {
        // Basis = {e_0, e_1} in ℝ^4. Direction = (0, 0, 1, 0) is orthogonal
        // to the basis → γ must be 0.0 → out == state bit-identically.
        let d = 4;
        let r = 2;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let state = [0.1, 0.2, 0.3, 0.4];
        let direction = [0.0, 0.0, 1.0, 0.0];
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert_eq!(gamma, 0.0, "γ must be exactly 0.0 for orthogonal direction");
        assert!(
            out.iter()
                .zip(state.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "out must be bit-identical to state when γ=0"
        );
    }

    #[test]
    fn g1a_no_harm_zero_direction() {
        // Zero direction → ‖d‖ = 0 → γ = 0 → no-harm.
        let d = 4;
        let r = 2;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let state = [0.5, -0.3, 0.7, -0.1];
        let direction = [0.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            1.0,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert_eq!(gamma, 0.0);
        assert!(
            out.iter()
                .zip(state.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    // ── G1b: full-correction parity (γ = 1) ───────────────────────────────

    #[test]
    fn g1b_full_correction_when_direction_in_span() {
        // Basis = {e_0, e_1} in ℝ^4. Direction = (0.6, 0.8, 0, 0) ∈ span(basis).
        // γ should be 1.0, out = state + eta_base * direction.
        let d = 4;
        let r = 2;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let state = [0.1, 0.2, 0.3, 0.4];
        let direction = [0.6, 0.8, 0.0, 0.0];
        let eta_base = 0.5_f32;
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            eta_base,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(
            (gamma - 1.0).abs() < 1e-5,
            "γ must be ~1.0 for direction in span(basis), got {gamma}"
        );
        // out = state + eta_base * gamma * d_proj
        //     = state + 0.5 * 1.0 * (0.6, 0.8, 0, 0)
        let expected = [
            state[0] + eta_base * 1.0 * 0.6,
            state[1] + eta_base * 1.0 * 0.8,
            state[2],
            state[3],
        ];
        for i in 0..d {
            assert!(
                (out[i] - expected[i]).abs() < 1e-5,
                "out[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn g1b_full_correction_basis_vector_itself() {
        // Direction = e_0 (a basis vector) → γ = 1.0 exactly.
        let d = 4;
        let r = 1;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0]]);
        let state = [0.0, 0.0, 0.0, 0.0];
        let direction = [1.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            1.0,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!((gamma - 1.0).abs() < 1e-5);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);
        assert!(out[2].abs() < 1e-5);
        assert!(out[3].abs() < 1e-5);
    }

    // ── G1c: ranking preservation (subspace-mediated input invariance) ────

    #[test]
    fn g1c_subspace_invariance_components_outside_span_dont_matter() {
        // Two directions that differ only OUTSIDE span(basis) must produce
        // identical projected corrections. The "subspace-mediated input
        // invariance" property (Research 408 §1.4).
        let d = 4;
        let r = 2;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let state = [0.1, 0.2, 0.3, 0.4];
        // d_a and d_b agree on dims 0,1 (in span) but differ on dims 2,3 (out).
        let d_a = [0.5, 0.3, 0.9, 0.1];
        let d_b = [0.5, 0.3, -0.7, 0.2];
        let mut out_a = [0.0f32; 4];
        let mut out_b = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma_a = tilr_refine_into(
            &state,
            &d_a,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch,
            &mut out_a,
        )
        .unwrap();
        let gamma_b = tilr_refine_into(
            &state,
            &d_b,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch,
            &mut out_b,
        )
        .unwrap();
        // γ values differ (because ‖d_a‖ ≠ ‖d_b‖), but the projected output
        // correction should be identical since d_proj_a == d_proj_b.
        // Actually: d_proj is the same (only dims 0,1), but γ differs because
        // ‖d‖ differs. So out_a ≠ out_b. The property we test is weaker:
        // d_proj_a == d_proj_b (the projected directions are identical).
        // Verify by checking the in-span components of out match.
        // Correction = eta * gamma * d_proj. d_proj_a == d_proj_b = (0.5, 0.3, 0, 0).
        // gamma_a = ‖(0.5,0.3)‖ / ‖d_a‖, gamma_b = ‖(0.5,0.3)‖ / ‖d_b‖.
        // The invariant is: changing out-of-span components does NOT change
        // d_proj. We verify d_proj directly via a separate probe.
        let mut probe_a = TilrScratch::with_capacity(d, r);
        let mut probe_b = TilrScratch::with_capacity(d, r);
        let mut dummy = [0.0f32; 4];
        let _ = tilr_refine_into(
            &[0.0; 4],
            &d_a,
            &basis,
            r,
            0.0, // eta_base=0 so out is irrelevant; we just want d_proj in scratch
            1e-12,
            &mut probe_a,
            &mut dummy,
        );
        let _ = tilr_refine_into(
            &[0.0; 4],
            &d_b,
            &basis,
            r,
            0.0,
            1e-12,
            &mut probe_b,
            &mut dummy,
        );
        for i in 0..d {
            assert!(
                (probe_a.d_proj[i] - probe_b.d_proj[i]).abs() < 1e-6,
                "d_proj[{i}] differs: a={}, b={}",
                probe_a.d_proj[i],
                probe_b.d_proj[i]
            );
        }
        // Sanity: the in-span dims 0,1 are non-zero, out-of-span dims 2,3 are 0.
        assert!(probe_a.d_proj[0].abs() > 1e-6);
        assert!(probe_a.d_proj[1].abs() > 1e-6);
        assert!(probe_a.d_proj[2].abs() < 1e-6);
        assert!(probe_a.d_proj[3].abs() < 1e-6);
        // gamma_a and gamma_b differ because ‖d_a‖ ≠ ‖d_b‖.
        assert!((gamma_a - gamma_b).abs() > 1e-4);
    }

    // ── G1d: γ monotonicity as direction rotates into the subspace ─────────

    #[test]
    fn g1d_gamma_monotone_from_orthogonal_to_aligned() {
        // Sweep a direction from orthogonal-to-basis to within-basis.
        // γ must increase monotonically from 0 to 1.
        let d = 2;
        let r = 1;
        let basis = basis_from_rows(&[&[1.0, 0.0]]); // basis = e_0
        let state = [0.0, 0.0];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let mut out = [0.0f32; 2];
        let mut prev_gamma = -1.0_f32;
        // θ from 0 (orthogonal: e_1) to π/2 (aligned: e_0).
        // direction = (sin θ, cos θ): at θ=0 → (0,1)=e_1 orthogonal, at θ=π/2 → (1,0)=e_0 aligned.
        let steps = 32;
        for s in 0..=steps {
            let theta = (s as f32 / steps as f32) * std::f32::consts::FRAC_PI_2;
            let direction = [theta.sin(), theta.cos()];
            let gamma = tilr_refine_into(
                &state,
                &direction,
                &basis,
                r,
                1.0,
                1e-12,
                &mut scratch,
                &mut out,
            )
            .unwrap();
            assert!(
                gamma >= prev_gamma - 1e-5,
                "γ not monotone at step {s}: gamma={gamma}, prev={prev_gamma}"
            );
            prev_gamma = gamma;
        }
        // At θ=0 (direction = e_1, orthogonal): γ = 0.
        let gamma_orth = tilr_refine_into(
            &state,
            &[0.0, 1.0],
            &basis,
            r,
            1.0,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(gamma_orth.abs() < 1e-5, "γ at orthogonal = {gamma_orth}");
        // At θ=π/2 (direction = e_0, aligned): γ = 1.
        let gamma_aligned = tilr_refine_into(
            &state,
            &[1.0, 0.0],
            &basis,
            r,
            1.0,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(
            (gamma_aligned - 1.0).abs() < 1e-5,
            "γ at aligned = {gamma_aligned}"
        );
    }

    // ── G1e: orthonormality validation rejects bad basis ──────────────────

    #[test]
    fn g1e_check_orthonormal_rejects_non_unit_norm() {
        // Basis vector with norm 2.0 (not unit).
        let basis = [2.0, 0.0, 0.0, 0.0];
        let err = check_orthonormal(&basis, 1, 4, 1e-5).unwrap_err();
        assert_eq!(err, TilrError::NotOrthonormal);
    }

    #[test]
    fn g1e_check_orthonormal_rejects_non_orthogonal_pair() {
        // Two nearly-parallel vectors (not orthogonal).
        let basis = [1.0, 0.0, 0.0, 0.0, 0.9, 0.43589, 0.0, 0.0]; // 2nd row not unit either
        let result = check_orthonormal(&basis, 2, 4, 1e-5);
        assert!(matches!(result, Err(TilrError::NotOrthonormal)));
    }

    #[test]
    fn g1e_check_orthonormal_accepts_valid_basis() {
        let basis = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ];
        assert!(check_orthonormal(&basis, 3, 4, 1e-5).is_ok());
    }

    #[test]
    fn g1e_check_orthonormal_rejects_bad_dimensions() {
        // basis.len() = 8 but r*d = 2*4 = 8 — actually OK. Test r=3, d=4 → 12 ≠ 8.
        let basis = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let err = check_orthonormal(&basis, 3, 4, 1e-5).unwrap_err();
        assert_eq!(err, TilrError::DimensionMismatch);
    }

    // ── Error paths ────────────────────────────────────────────────────────

    #[test]
    fn rejects_eta_out_of_range() {
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0]]);
        let state = [0.0; 4];
        let direction = [1.0, 0.0, 0.0, 0.0];
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(4, 1);
        let err = tilr_refine_into(
            &state,
            &direction,
            &basis,
            1,
            1.5, // > 1.0
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap_err();
        assert_eq!(err, TilrError::EtaOutOfRange);
    }

    #[test]
    fn rejects_dimension_mismatch_state_direction() {
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0]]);
        let state = [0.0; 4];
        let direction = [0.0; 3]; // wrong length
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(4, 1);
        let err = tilr_refine_into(
            &state,
            &direction,
            &basis,
            1,
            0.5,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap_err();
        assert_eq!(err, TilrError::DimensionMismatch);
    }

    #[test]
    fn rejects_dimension_mismatch_basis_r_d() {
        let basis = [1.0, 0.0, 0.0, 0.0]; // 4 elements, but r=2, d=4 → need 8
        let state = [0.0; 4];
        let direction = [0.0; 4];
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(4, 2);
        let err = tilr_refine_into(
            &state,
            &direction,
            &basis,
            2, // wrong r
            0.5,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap_err();
        assert_eq!(err, TilrError::DimensionMismatch);
    }

    // ── In-place apply (tilr_refine_apply) ─────────────────────────────────

    #[test]
    fn in_place_apply_matches_separate_output() {
        let d = 4;
        let r = 2;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let mut state_inplace = [0.1, 0.2, 0.3, 0.4];
        let state_separate = [0.1, 0.2, 0.3, 0.4];
        let direction = [0.5, 0.3, 0.0, 0.0];
        let mut out_separate = [0.0f32; 4];
        let mut scratch1 = TilrScratch::with_capacity(d, r);
        let mut scratch2 = TilrScratch::with_capacity(d, r);
        let g1 = tilr_refine_apply(
            &mut state_inplace,
            &direction,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch1,
        )
        .unwrap();
        let g2 = tilr_refine_into(
            &state_separate,
            &direction,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch2,
            &mut out_separate,
        )
        .unwrap();
        assert!((g1 - g2).abs() < 1e-6);
        for i in 0..d {
            assert!((state_inplace[i] - out_separate[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn in_place_apply_no_harm_at_gamma_zero() {
        // In-place variant also satisfies the no-harm contract.
        let d = 4;
        let r = 2;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let original = [0.5, -0.3, 0.7, -0.1];
        let mut state = original;
        let direction = [0.0, 0.0, 1.0, 0.0]; // orthogonal to basis → γ = 0
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma =
            tilr_refine_apply(&mut state, &direction, &basis, r, 0.5, 1e-12, &mut scratch).unwrap();
        assert_eq!(gamma, 0.0);
        assert!(
            state
                .iter()
                .zip(original.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    // ── TilrScratch capacity ──────────────────────────────────────────────

    #[test]
    fn scratch_capacity_reports_correct_dims() {
        let s = TilrScratch::with_capacity(64, 12);
        assert_eq!(s.dim(), 64);
        assert_eq!(s.rank(), 12);
    }

    // ── Allocating wrapper ────────────────────────────────────────────────

    #[test]
    fn allocating_wrapper_matches_into() {
        let d = 4;
        let r = 2;
        let basis = basis_from_rows(&[&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]]);
        let state = [0.1, 0.2, 0.3, 0.4];
        let direction = [0.5, 0.3, 0.1, 0.2];
        let (out_alloc, gamma_alloc) = tilr_refine(&state, &direction, &basis, r, 0.5).unwrap();
        // Compare against the into variant.
        let mut out_into = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma_into = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch,
            &mut out_into,
        )
        .unwrap();
        assert!((gamma_alloc - gamma_into).abs() < 1e-6);
        for i in 0..d {
            assert!((out_alloc[i] - out_into[i]).abs() < 1e-6);
        }
    }

    // ── HLA-scale smoke test (d=8, r=3) ────────────────────────────────────

    #[test]
    fn hla_scale_smoke() {
        // Simulate HLA scale: d=8, r=3. Use first 3 standard basis vectors.
        let d = 8;
        let r = 3;
        let basis: Vec<f32> = (0..r).flat_map(|k| e(k, d)).collect();
        check_orthonormal(&basis, r, d, 1e-5).unwrap();
        let state: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1).collect();
        // Direction with components in-span (first 3) and out-of-span (last 5).
        let direction: Vec<f32> = vec![0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9];
        let mut out = vec![0.0; d];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            0.3,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        // γ = ‖proj(first 3 components)‖ / ‖direction‖
        let in_span_norm_sq: f32 = direction[..3].iter().map(|x| x * x).sum();
        let full_norm_sq: f32 = direction.iter().map(|x| x * x).sum();
        let expected_gamma = (in_span_norm_sq / full_norm_sq).sqrt();
        assert!(
            (gamma - expected_gamma).abs() < 1e-5,
            "γ = {gamma}, expected {expected_gamma}"
        );
        // dims 3..8 should be unchanged (d_proj is zero there).
        for i in 3..d {
            assert!(
                (out[i] - state[i]).abs() < 1e-6,
                "dim {i} changed unexpectedly"
            );
        }
    }

    // ── Shard-scale smoke test (d=64, r=12) ────────────────────────────────

    #[test]
    fn shard_scale_smoke() {
        let d = 64;
        let r = 12;
        // Random-ish orthonormal basis: first 12 standard basis vectors.
        let basis: Vec<f32> = (0..r).flat_map(|k| e(k, d)).collect();
        check_orthonormal(&basis, r, d, 1e-5).unwrap();
        let state: Vec<f32> = (0..d).map(|i| ((i * 7) as f32) * 0.01).collect();
        let direction: Vec<f32> = (0..d).map(|i| (((i * 13 + 1) % 10) as f32) * 0.1).collect();
        let mut out = vec![0.0; d];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(gamma > 0.0 && gamma <= 1.0, "γ out of range: {gamma}");
        // Verify dims outside span are unchanged.
        for i in r..d {
            assert!((out[i] - state[i]).abs() < 1e-6);
        }
    }

    // ── Phase 3: discover_invariant_subspace ──────────────────────────────

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn t3_2_recovers_known_2d_subspace() {
        // Two non-axis-aligned orthonormal vectors in ℝ⁶.
        let u1_raw = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]; // ‖u1‖ = √3
        let u2_raw = [1.0, -1.0, 0.0, 1.0, 1.0, -1.0]; // ‖u2‖ = √5, u1 ⊥ u2
        let n1 = (3.0f32).sqrt();
        let n2 = (5.0f32).sqrt();
        let u1: Vec<f32> = u1_raw.iter().map(|x| x / n1).collect();
        let u2: Vec<f32> = u2_raw.iter().map(|x| x / n2).collect();
        let d = 6;

        // Generate N=15 differences: δ_t = a_t·u1 + b_t·u2, all in span(u1, u2).
        let mut seed: u32 = 42;
        let lcg = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((*s >> 8) as f32) / 16777216.0 - 0.5 // [-0.5, 0.5)
        };
        let diffs: Vec<Vec<f32>> = (0..15)
            .map(|_| {
                let a = lcg(&mut seed) * 2.0;
                let b = lcg(&mut seed) * 2.0;
                (0..d).map(|i| a * u1[i] + b * u2[i]).collect()
            })
            .collect();
        let diff_refs: Vec<&[f32]> = diffs.iter().map(|v| v.as_slice()).collect();

        let (basis, r) = discover_invariant_subspace(&diff_refs, 0.99).unwrap();

        // All variance is in span(u1, u2) → r should be 2.
        assert_eq!(r, 2, "expected rank-2 subspace, got r={r}");
        assert_eq!(basis.len(), r * d);

        // The returned basis must be orthonormal.
        check_orthonormal(&basis, r, d, 1e-4).unwrap();

        // Subspace recovery: each true direction projected onto the recovered
        // basis should have projection norm ≈ 1 (principal angle ≈ 0).
        for u_true in [&u1, &u2] {
            let proj_norm_sq: f32 = (0..r)
                .map(|j| {
                    let vj = &basis[j * d..(j + 1) * d];
                    let dot = simd_dot_f32(u_true, vj, d);
                    dot * dot
                })
                .sum();
            let proj_norm = proj_norm_sq.sqrt();
            assert!(
                (proj_norm - 1.0).abs() < 0.01,
                "subspace recovery failed: proj_norm = {proj_norm} (expected ≈1.0)"
            );
        }
    }

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn discover_rejects_empty_differences() {
        let err = discover_invariant_subspace(&[], 0.9).unwrap_err();
        assert_eq!(err, TilrError::DimensionMismatch);
    }

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn discover_rejects_invalid_tau() {
        let diffs: Vec<&[f32]> = vec![&[1.0, 0.0]];
        assert_eq!(
            discover_invariant_subspace(&diffs, 0.0).unwrap_err(),
            TilrError::InvalidTau
        );
        assert_eq!(
            discover_invariant_subspace(&diffs, 1.5).unwrap_err(),
            TilrError::InvalidTau
        );
        assert_eq!(
            discover_invariant_subspace(&diffs, -0.1).unwrap_err(),
            TilrError::InvalidTau
        );
    }

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn discover_rejects_mismatched_lengths() {
        let diffs: Vec<&[f32]> = vec![&[1.0, 0.0], &[1.0, 0.0, 0.0]];
        assert_eq!(
            discover_invariant_subspace(&diffs, 0.9).unwrap_err(),
            TilrError::DimensionMismatch
        );
    }

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn discover_rejects_zero_variance() {
        let diffs: Vec<&[f32]> = vec![&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]];
        assert_eq!(
            discover_invariant_subspace(&diffs, 0.9).unwrap_err(),
            TilrError::ZeroVariance
        );
    }

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn discover_rank1_collinear_differences() {
        let diffs: Vec<&[f32]> = vec![&[2.0, 0.0, 0.0], &[3.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]];
        let (basis, r) = discover_invariant_subspace(&diffs, 0.95).unwrap();
        assert_eq!(r, 1);
        let v = &basis[..3];
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "basis not unit: {norm}");
        assert!(v[0].abs() > 0.99, "basis not aligned with e_0: {v:?}");
    }

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn discover_tau_controls_rank() {
        // δ1 = (10, 0, 0), δ2 = (0, 1, 0) → σ₁²=100, σ₂²=1, total=101.
        // tau=0.90: σ₁²/total = 0.990 ≥ 0.90 → r=1.
        // tau=0.999: 0.990 < 0.999, 1.0 ≥ 0.999 → r=2.
        let diffs: Vec<&[f32]> = vec![&[10.0, 0.0, 0.0], &[0.0, 1.0, 0.0]];
        let (_, r1) = discover_invariant_subspace(&diffs, 0.90).unwrap();
        assert_eq!(r1, 1, "tau=0.90 should give r=1");
        let (_, r2) = discover_invariant_subspace(&diffs, 0.999).unwrap();
        assert_eq!(r2, 2, "tau=0.999 should give r=2");
    }

    #[cfg(feature = "subspace_phase_gate")]
    #[test]
    fn discover_then_refine_round_trip() {
        // Discover a subspace, then use it with tilr_refine_into.
        let diffs: Vec<&[f32]> = vec![&[2.0, 0.0, 0.0, 0.0], &[0.0, 3.0, 0.0, 0.0]];
        let (basis, r) = discover_invariant_subspace(&diffs, 0.95).unwrap();
        let d = 4;
        assert_eq!(r, 2);
        check_orthonormal(&basis, r, d, 1e-4).unwrap();

        // Direction in span(basis) → γ ≈ 1.0 (full correction).
        let state = [0.1, 0.2, 0.3, 0.4];
        let direction = [0.5, 0.3, 0.0, 0.0]; // in the e_0-e_1 plane
        let mut out = [0.0f32; 4];
        let mut scratch = TilrScratch::with_capacity(d, r);
        let gamma = tilr_refine_into(
            &state,
            &direction,
            &basis,
            r,
            0.5,
            1e-12,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(
            (gamma - 1.0).abs() < 0.01,
            "expected γ≈1.0 for in-span direction, got {gamma}"
        );
    }
}
