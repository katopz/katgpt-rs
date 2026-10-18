//! Region-Conditioned Subspace Field — MFA local-geometry steering (Plan 416).
//!
//! The region-conditioned generalization of
//! [`SubspaceSteeringField`](crate::subspace_steering::SubspaceSteeringField)
//! (Plan 412). Plan 412 carries a single k-dim orthonormal block that applies
//! globally. This module generalizes it to **K regions, each with its own
//! centroid `μ_k` and its own local R-dim subspace (factor-analyzer loadings
//! `W_k`)** — the Mixture-of-Factor-Analyzers (MFA) structure from
//! [arXiv:2602.02464](https://arxiv.org/abs/2602.02464).
//!
//! Each activation `x` is decomposed into two compositional geometric objects:
//! - **Region centroid `μ_k`** — absolute position in activation space. Steered
//!   via centroid interpolation `f_μ(x) = (1−α)x + αμ_k`.
//! - **Local loadings `W_k`** — within-region variation. Steered via local
//!   subspace offset `f_w(x) = x + W_k·v`.
//!
//! The artifact `{μ_k, W_k, Ψ, π}` is either trained offline via GD on negative
//! log-likelihood (riir-train territory) or deterministically constructed via
//! K-means + per-region PCA (modelless baseline). Once frozen, all consumption
//! is closed-form linear algebra — no gradients at inference.
//!
//! # Sigmoid, not softmax
//!
//! The paper computes responsibilities `R_k(x) = p(k|x)` as a softmax over
//! Gaussian likelihoods (categorical, winner-take-all). Per the AGENTS.md
//! sigmoid mandate, this module reformulates them as **per-region independent
//! sigmoid membership gates** `g_k(x) = sigmoid(a_k(x) − τ)`. This is *more*
//! expressive: an activation can be partially in multiple regions simultaneously
//! (e.g., "70% combat, 30% fear"), consistent with
//! [`CommittedFieldBlend`](crate::committed_field_blend::CommittedFieldBlend)'s
//! sigmoid gates (Plan 321).
//!
//! # K=1 parity contract (the load-bearing gate)
//!
//! `RegionSubspaceField<D, 1, D>` with `μ_1 = 0`, `W_1 = I_D`, `log_pi = [0]`,
//! `psi_inv = [1; D]` must produce bit-identical `steer_local` output to
//! `SubspaceSteeringField<D, D>::apply` with block = identity and the same
//! offset as alphas. This is verified by the `k1_degenerate_parity_with_plan_412`
//! unit test and is the foundation of the GOAT gate.
//!
//! # Const generics
//!
//! [`RegionSubspaceField<const D, const K, const R>`] is parameterized by:
//! - `D` — the latent dimension (e.g. 8 for HLA, 64 for shards)
//! - `K` — the number of regions
//! - `R` — the per-region subspace rank (number of local axes per region)
//!
//! All storage is fixed-size arrays, so the struct is stack-only and zero-alloc
//! by construction.
//!
//! # References
//!
//! - Plan: `.plans/416_region_subspace_field_primitive.md`
//! - Research: `.research/396_MFA_Region_Conditioned_Factor_Analyzer.md`
//! - Source paper: [arXiv:2602.02464](https://arxiv.org/abs/2602.02464) —
//!   Shafran et al., "From Directions to Regions"
//! - Within-region sibling: Plan 412 (`subspace_steering.rs`, DEFAULT-ON)
//! - 1D baseline: Plan 309 (`latent_steering.rs`)
//!
//! Region-Conditioned Subspace Field — MFA local-geometry steering (Plan 416,
//! Research 396, arxiv 2602.02464 Shafran et al. "From Directions to Regions").
//! The region-conditioned generalization of Plan 412: K regions, each with a
//! centroid μ_k and a local R-dim factor-analyzer subspace W_k. Two-mode
//! steering: centroid interpolation (move toward a region) + local subspace
//! offset (walk within a region). Per-region sigmoid membership gates (reformulated
//! from the paper's softmax responsibilities per the AGENTS.md sigmoid mandate —
//! more expressive: multi-region membership). Pure modelless consumer of a frozen
//! MFA-like artifact {μ_k, W_k, Ψ, π} (trained offline via riir-train GD, or
//! deterministically constructed via K-means + per-region PCA). At the degenerate
//! limit (K=1, μ=0, W=I) steer_local is bit-identical to Plan 412. DEFAULT-ON (Plan 416 Phase 4, 2026-07-09):
//! G1–G5 ALL PASS (G1 K=1 parity is the load-bearing gate).

use blake3::Hasher;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by [`RegionSubspaceField::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionSubspaceError {
    /// A loadings axis is not unit-norm within `orthonormal_tol`, OR two axes
    /// within the same region are not mutually orthogonal (the region's
    /// `W_k · W_kᵀ` has off-diagonal entries exceeding the tolerance).
    NotOrthonormal,
    /// A noise precision value `Ψ^{-1}[d]` is not strictly positive.
    InvalidPrecision,
}

// ── Field ───────────────────────────────────────────────────────────────────

/// A region-conditioned factor-analyzer field: K regions, each with a centroid
/// `μ_k` and a local R-dim subspace (loadings `W_k`). BLAKE3-committed.
///
/// Region-conditioned generalization of
/// [`SubspaceSteeringField`](crate::subspace_steering::SubspaceSteeringField)
/// (Plan 412). The three steering-field primitives in this crate occupy a
/// strict hierarchy:
///
/// | Primitive | Dim | Regions | Mechanism |
/// |-----------|-----|---------|-----------|
/// | [`LatentSteeringVector`](crate::latent_steering::LatentSteeringVector) (P309) | 1D | — | `s + α·v` |
/// | [`SubspaceSteeringField`](crate::subspace_steering::SubspaceSteeringField) (P412) | k-dim | 1 (global) | `s + Σ α_j·u_j` |
/// | **`RegionSubspaceField`** (P416) | **R-dim per region** | **K** | **centroid + local subspace, two-mode steering** |
///
/// # Construction
///
/// Use [`new`](Self::new) to validate loadings orthonormality (per region) and
/// noise precision positivity. The projectors `Z_k` (posterior-mean operators,
/// eq. 10) are pre-computed at construction and frozen for the field's lifetime.
///
/// # Allocation
///
/// Zero heap allocations by construction — all fields are fixed-size arrays.
#[derive(Debug, Clone)]
pub struct RegionSubspaceField<const D: usize, const K: usize, const R: usize> {
    /// Region centroids `μ_k ∈ R^D`. K rows. Absolute positions in activation
    /// space. Steered via centroid interpolation
    /// ([`steer_centroid`](RegionSubspaceField::steer_centroid)).
    pub centroids: [[f32; D]; K],
    /// Per-region factor-analyzer loadings `W_k ∈ R^{R×D}`.
    /// `loadings[k][r]` is the r-th local axis (D-dim unit vector) for region k.
    /// Each region has R local axes. Steered via local subspace offset
    /// ([`steer_local`](RegionSubspaceField::steer_local)).
    pub loadings: [[[f32; D]; R]; K],
    /// Per-region mixture log-weights `log π_k` (pre-computed at construction).
    pub log_pi: [f32; K],
    /// Diagonal noise precision (inverse variance) per dimension, `Ψ^{-1}`.
    /// Must be strictly positive.
    pub psi_inv: [f32; D],
    /// Pre-computed posterior-mean projector `Z_k ∈ R^{R×D}` per region.
    /// `Z_k = (I_R + W_kᵀ Ψ^{-1} W_k)⁻¹ W_kᵀ Ψ^{-1}` (eq. 10, closed-form).
    /// Computed once at construction; frozen for the field's lifetime.
    pub projectors: [[[f32; D]; R]; K],
    /// `BLAKE3(centroids || loadings || log_pi || psi_inv)` — content commitment.
    pub commitment: [u8; 32],
}

impl<const D: usize, const K: usize, const R: usize> RegionSubspaceField<D, K, R> {
    /// Construct a region-conditioned field, validating loadings orthonormality
    /// (per region) and noise precision positivity. Pre-computes the
    /// posterior-mean projectors `Z_k` via eq. 10.
    ///
    /// Returns [`RegionSubspaceError::NotOrthonormal`] if any `loadings[k][r]`
    /// is not unit-norm within `orthonormal_tol`, or if any pair within a region
    /// has `|dot| > orthonormal_tol`. Returns
    /// [`RegionSubspaceError::InvalidPrecision`] if any `psi_inv[d] <= 0`.
    pub fn new(
        centroids: [[f32; D]; K],
        loadings: [[[f32; D]; R]; K],
        log_pi: [f32; K],
        psi_inv: [f32; D],
        orthonormal_tol: f32,
    ) -> Result<Self, RegionSubspaceError> {
        // Validate noise precision: strictly positive.
        for &p in &psi_inv {
            if p <= 0.0 {
                return Err(RegionSubspaceError::InvalidPrecision);
            }
        }
        // Validate loadings orthonormality per region.
        for region_axes in &loadings {
            // Each axis must be unit-norm.
            for axis in region_axes {
                let norm = row_norm(axis);
                if (norm - 1.0).abs() > orthonormal_tol {
                    return Err(RegionSubspaceError::NotOrthonormal);
                }
            }
            // Each pair of axes within the region must be orthogonal.
            for i in 0..R {
                for j in (i + 1)..R {
                    let dot = dot_product(&region_axes[i], &region_axes[j]);
                    if dot.abs() > orthonormal_tol {
                        return Err(RegionSubspaceError::NotOrthonormal);
                    }
                }
            }
        }
        // Pre-compute projectors Z_k = (I_R + W_k^T Ψ^{-1} W_k)^{-1} W_k^T Ψ^{-1}.
        let projectors = compute_projectors(&loadings, &psi_inv);
        let commitment = compute_field_commitment(&centroids, &loadings, &log_pi, &psi_inv);
        Ok(Self {
            centroids,
            loadings,
            log_pi,
            psi_inv,
            projectors,
            commitment,
        })
    }

    /// Construct without validation. Caller guarantees loadings orthonormality
    /// and positive noise precision. Projectors and commitment are still
    /// computed. Used when the field comes from a trusted frozen artifact.
    pub fn new_unchecked(
        centroids: [[f32; D]; K],
        loadings: [[[f32; D]; R]; K],
        log_pi: [f32; K],
        psi_inv: [f32; D],
    ) -> Self {
        let projectors = compute_projectors(&loadings, &psi_inv);
        let commitment = compute_field_commitment(&centroids, &loadings, &log_pi, &psi_inv);
        Self {
            centroids,
            loadings,
            log_pi,
            psi_inv,
            projectors,
            commitment,
        }
    }

    /// Re-check loadings orthonormality (within `tol`) AND that the stored
    /// commitment matches the current contents. Returns `false` if either fails.
    /// Note: does NOT re-check projectors (they are derived from loadings +
    /// psi_inv, so a commitment match implies projector consistency).
    #[must_use]
    pub fn verify(&self, tol: f32) -> bool {
        for region_axes in &self.loadings {
            for axis in region_axes {
                let norm = row_norm(axis);
                if (norm - 1.0).abs() > tol {
                    return false;
                }
            }
            for i in 0..R {
                for j in (i + 1)..R {
                    let dot = dot_product(&region_axes[i], &region_axes[j]);
                    if dot.abs() > tol {
                        return false;
                    }
                }
            }
        }
        compute_field_commitment(&self.centroids, &self.loadings, &self.log_pi, &self.psi_inv)
            == self.commitment
    }

    /// The latent dimension `D`.
    #[inline]
    #[must_use]
    pub const fn dim(&self) -> usize {
        D
    }

    /// The number of regions `K`.
    #[inline]
    #[must_use]
    pub const fn num_regions(&self) -> usize {
        K
    }

    /// The per-region subspace rank `R`.
    #[inline]
    #[must_use]
    pub const fn rank(&self) -> usize {
        R
    }

    // ── Operation 1: membership gates ──────────────────────────────────────

    /// Per-region sigmoid membership gates. Returns a fixed `[f32; K]` array
    /// where `out[k] = sigmoid(a_k(x) − τ)` and `a_k(x)` is the (unnormalized)
    /// Gaussian log-likelihood of `x` under region `k`.
    ///
    /// This is the sigmoid reformulation of the paper's softmax responsibilities
    /// (eq. 8). Per-region independent gates ∈ (0,1) — an activation can be
    /// partially in multiple regions simultaneously. Zero-alloc.
    ///
    /// The log-likelihood term drops the constant `-0.5·log|C_k|` (it cancels
    /// under the sigmoid when `Ψ` is shared across regions) and the `-D/2·log(2π)`
    /// constant. The Mahalanobis distance uses the diagonal-noise approximation
    /// `||x − μ_k||²_{Ψ^{-1}} = Σ_d psi_inv[d]·(x[d] − μ_k[d])²`.
    #[inline]
    #[must_use]
    pub fn membership_gates(&self, state: &[f32; D], tau: f32) -> [f32; K] {
        let mut out = [0f32; K];
        for (k, out_k) in out.iter_mut().enumerate() {
            let mut mahal = 0.0;
            for (d, &s) in state.iter().enumerate() {
                let diff = s - self.centroids[k][d];
                mahal += self.psi_inv[d] * diff * diff;
            }
            // a_k(x) = log π_k − 0.5 · Mahalanobis(x, μ_k, Ψ^{-1})
            let a_k = self.log_pi[k] - 0.5 * mahal;
            *out_k = sigmoid(a_k - tau);
        }
        out
    }

    // ── Operation 2: local coordinates ─────────────────────────────────────

    /// Posterior-mean latent vector within region `k` (eq. 9-10).
    ///
    /// Returns `ẑ_k = Z_k · (x − μ_k)` — the R-dim local coordinates of `state`
    /// within region `k`'s subspace. Zero-alloc (output is a stack `[f32; R]`).
    ///
    /// # Panics
    ///
    /// Debug-panics if `k >= K`.
    #[inline]
    #[must_use]
    pub fn local_coordinates(&self, state: &[f32; D], k: usize) -> [f32; R] {
        debug_assert!(k < K, "region index {k} >= K={K}");
        let mut out = [0f32; R];
        // ẑ_k[r] = Σ_d Z_k[r][d] · (x[d] − μ_k[d])
        for (r, out_r) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (d, &s) in state.iter().enumerate() {
                acc += self.projectors[k][r][d] * (s - self.centroids[k][d]);
            }
            *out_r = acc;
        }
        out
    }

    // ── Operation 3: centroid steering ─────────────────────────────────────

    /// Centroid interpolation toward region `k` (eq. 14). In-place.
    ///
    /// Computes `state = (1 − α)·state + α·μ_k`. At `α = 0` identity, at
    /// `α = 1` full region replacement. Zero-alloc.
    ///
    /// # Panics
    ///
    /// Debug-panics if `k >= K`.
    #[inline]
    pub fn steer_centroid(&self, state: &mut [f32; D], k: usize, alpha: f32) {
        debug_assert!(k < K, "region index {k} >= K={K}");
        let one_minus_a = 1.0 - alpha;
        for (d, s) in state.iter_mut().enumerate() {
            *s = one_minus_a * *s + alpha * self.centroids[k][d];
        }
    }

    // ── Operation 4: local subspace steering ───────────────────────────────

    /// Local subspace offset within region `k` (eq. 15). In-place.
    ///
    /// Computes `state += W_k · offset`. Region-conditioned: the loadings `W_k`
    /// are selected by region index `k`. Zero-alloc.
    ///
    /// **K=1 parity contract:** at `K=1, μ_1=0, W_1=I_D`, this reduces to
    /// `state += offset`, which is bit-identical to
    /// [`SubspaceSteeringField::apply`](crate::subspace_steering::SubspaceSteeringField::apply)
    /// with block = identity and `alphas = offset`.
    ///
    /// # Panics
    ///
    /// Debug-panics if `k >= K`.
    #[inline]
    pub fn steer_local(&self, state: &mut [f32; D], k: usize, offset: &[f32; R]) {
        debug_assert!(k < K, "region index {k} >= K={K}");
        // state[j] += Σ_r offset[r] * loadings[k][r][j]
        for (r, &v_r) in offset.iter().enumerate() {
            let axis = &self.loadings[k][r];
            for j in 0..D {
                state[j] += v_r * axis[j];
            }
        }
    }

    // ── Operation 5: full decomposition ────────────────────────────────────

    /// Full decomposition: membership gates for all regions + local coordinates
    /// for all regions. Zero-alloc (stack struct).
    ///
    /// Combines [`membership_gates`](Self::membership_gates) and
    /// [`local_coordinates`](Self::local_coordinates) for all K regions in one
    /// pass.
    #[inline]
    #[must_use]
    pub fn decompose(&self, state: &[f32; D], tau: f32) -> RegionDecomposition<K, R> {
        let gates = self.membership_gates(state, tau);
        let mut local_coords = [[0f32; R]; K];
        for (k, lc) in local_coords.iter_mut().enumerate() {
            *lc = self.local_coordinates(state, k);
        }
        RegionDecomposition {
            gates,
            local_coords,
        }
    }
}

// ── Decomposition result ────────────────────────────────────────────────────

/// Result of decomposing a state via
/// [`RegionSubspaceField::decompose`](RegionSubspaceField::decompose).
///
/// Carries the per-region membership gates and per-region local coordinates.
/// Can be passed to [`reconstruct`] to reconstruct the original state.
#[derive(Debug, Clone, Copy)]
pub struct RegionDecomposition<const K: usize, const R: usize> {
    /// Per-region sigmoid membership gates `g_k(x) ∈ (0,1)`.
    pub gates: [f32; K],
    /// Per-region local coordinates `ẑ_k(x) ∈ R^R`.
    pub local_coords: [[f32; R]; K],
}

// ── Free functions ──────────────────────────────────────────────────────────

/// Reconstruct a state from a decomposition and a field (eq. 11).
///
/// Computes `x̂ = Σ_k g_k · [μ_k + W_k · ẑ_k] / Σ_k g_k`. The normalization by
/// `Σ_k g_k` is needed because sigmoid gates don't sum to 1 (unlike softmax).
/// Zero-alloc (output is a stack `[f32; D]`).
///
/// # Panics
///
/// Debug-panics if `Σ_k g_k == 0` (all gates closed — use a lower `τ` or check
/// the gates before reconstructing).
#[inline]
#[must_use]
pub fn reconstruct<const D: usize, const K: usize, const R: usize>(
    decomp: &RegionDecomposition<K, R>,
    field: &RegionSubspaceField<D, K, R>,
) -> [f32; D] {
    let mut out = [0f32; D];
    let mut total_weight = 0.0;
    for k in 0..K {
        let g = decomp.gates[k];
        if g <= 0.0 {
            continue;
        }
        total_weight += g;
        // Add g_k · μ_k
        for (d, out_d) in out.iter_mut().enumerate() {
            *out_d += g * field.centroids[k][d];
        }
        // Add g_k · (W_k · ẑ_k)
        for r in 0..R {
            let weighted_z = g * decomp.local_coords[k][r];
            let axis = &field.loadings[k][r];
            for d in 0..D {
                out[d] += weighted_z * axis[d];
            }
        }
    }
    debug_assert!(total_weight > 0.0, "all membership gates are zero");
    // Normalize by Σ g_k.
    let inv = 1.0 / total_weight;
    for x in &mut out {
        *x *= inv;
    }
    out
}

/// Compute the BLAKE3 commitment over `centroids || loadings || log_pi || psi_inv`
/// (little-endian). Deterministic, quorum-verifiable.
#[must_use]
pub fn compute_field_commitment<const D: usize, const K: usize, const R: usize>(
    centroids: &[[f32; D]; K],
    loadings: &[[[f32; D]; R]; K],
    log_pi: &[f32; K],
    psi_inv: &[f32; D],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for row in centroids.iter() {
        for &f in row.iter() {
            hasher.update(&f.to_le_bytes());
        }
    }
    for region_axes in loadings.iter() {
        for axis in region_axes.iter() {
            for &f in axis.iter() {
                hasher.update(&f.to_le_bytes());
            }
        }
    }
    for &f in log_pi.iter() {
        hasher.update(&f.to_le_bytes());
    }
    for &f in psi_inv.iter() {
        hasher.update(&f.to_le_bytes());
    }
    let mut out = [0u8; 32];
    hasher.finalize_xof().fill(&mut out);
    out
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Compute the posterior-mean projectors `Z_k = (I_R + W_kᵀ Ψ^{-1} W_k)⁻¹ W_kᵀ Ψ^{-1}`
/// for all K regions. Each `Z_k` is `R×D`.
///
/// This is the closed-form FA posterior-mean operator (Ghahramani & Hinton 1996,
/// eq. 10). The R×R matrix `M_k = I_R + W_kᵀ Ψ^{-1} W_k` is inverted via
/// Gauss-Jordan with partial pivoting.
fn compute_projectors<const D: usize, const K: usize, const R: usize>(
    loadings: &[[[f32; D]; R]; K],
    psi_inv: &[f32; D],
) -> [[[f32; D]; R]; K] {
    let mut projectors = [[[0f32; D]; R]; K];
    for k in 0..K {
        // Step 1: compute W_k^T Ψ^{-1} — an R×D matrix.
        // (WtPsiInv)[r][d] = loadings[k][r][d] * psi_inv[d]
        let mut wt_psi_inv = [[0f32; D]; R];
        for r in 0..R {
            for d in 0..D {
                wt_psi_inv[r][d] = loadings[k][r][d] * psi_inv[d];
            }
        }
        // Step 2: compute M_k = I_R + W_k^T Ψ^{-1} W_k — an R×R matrix.
        // M_k[r1][r2] = δ(r1,r2) + Σ_d loadings[k][r1][d] * wt_psi_inv[r2][d]
        let mut m = [[0f32; R]; R];
        for r1 in 0..R {
            for r2 in 0..R {
                let mut acc = if r1 == r2 { 1.0 } else { 0.0 };
                for d in 0..D {
                    acc += loadings[k][r1][d] * wt_psi_inv[r2][d];
                }
                m[r1][r2] = acc;
            }
        }
        // Step 3: invert M_k (R×R) via Gauss-Jordan with partial pivoting.
        let m_inv = invert_rxr(&m);
        // Step 4: Z_k = M_k^{-1} · (W_k^T Ψ^{-1}) — R×D.
        for r in 0..R {
            for d in 0..D {
                let mut acc = 0.0;
                for r2 in 0..R {
                    acc += m_inv[r][r2] * wt_psi_inv[r2][d];
                }
                projectors[k][r][d] = acc;
            }
        }
    }
    projectors
}

/// Invert an R×R matrix via Gauss-Jordan elimination with partial pivoting.
///
/// Returns the inverse. For the small R values used here (typically 2–4), this
/// is numerically stable and zero-alloc (works on stack arrays).
fn invert_rxr<const R: usize>(m: &[[f32; R]; R]) -> [[f32; R]; R] {
    // Augment [m | I] into a working R×2R array, then reduce.
    // Since R is a const generic, we can't easily allocate a [R][2R] array
    // via a single type — use two separate [R][R] arrays: the source and
    // the identity that becomes the inverse.
    let mut src = *m;
    let mut inv = [[0f32; R]; R];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    // Forward elimination with partial pivoting.
    for col in 0..R {
        // Find pivot row (max abs in this column, at or below `col`).
        let mut pivot_row = col;
        let mut max_abs = src[col][col].abs();
        for (row, row_arr) in src.iter().enumerate().skip(col + 1) {
            let abs_val = row_arr[col].abs();
            if abs_val > max_abs {
                max_abs = abs_val;
                pivot_row = row;
            }
        }
        // Swap pivot row into position.
        if pivot_row != col {
            src.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }
        // Scale pivot row so src[col][col] == 1.
        let pivot = src[col][col];
        // Guard against singular matrices — for valid M_k = I + W^T Ψ^{-1} W
        // (positive definite), this should never happen.
        if pivot.abs() < f32::MIN_POSITIVE {
            // Singular: return zero matrix (projector will produce zeros).
            return [[0f32; R]; R];
        }
        let inv_pivot = 1.0 / pivot;
        for j in 0..R {
            src[col][j] *= inv_pivot;
            inv[col][j] *= inv_pivot;
        }
        // Eliminate all other rows.
        for row in 0..R {
            if row == col {
                continue;
            }
            let factor = src[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..R {
                src[row][j] -= factor * src[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    inv
}

#[inline]
fn row_norm<const D: usize>(row: &[f32; D]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn dot_product<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Delegates to the codebase-wide Cephes-backed fast_sigmoid
    // (~1 ULP accurate, ~1.7× faster than libm exp on aarch64).
    crate::simd::fast_sigmoid(x)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::subspace_steering::SubspaceSteeringField;

    // ── Helpers ────────────────────────────────────────────────────────────

    /// Build an R×D identity-ish loadings block: axis r has a 1.0 at index r
    /// and 0.0 elsewhere. Requires R <= D.
    fn identity_loadings<const D: usize, const R: usize>() -> [[f32; D]; R] {
        debug_assert!(R <= D);
        let mut block = [[0f32; D]; R];
        for (r, row) in block.iter_mut().enumerate() {
            row[r] = 1.0;
        }
        block
    }

    // ── G1: K=1 parity with Plan 412 (the load-bearing gate) ───────────────

    #[test]
    fn k1_degenerate_parity_with_plan_412() {
        // RegionSubspaceField<8, 1, 8> with μ=0, W=I_8, log_pi=[0], psi_inv=[1;8].
        // steer_local(state, 0, offset) should be bit-identical to
        // SubspaceSteeringField<8, 8>::apply with block=I_8, alphas=offset.
        type F = RegionSubspaceField<8, 1, 8>;
        const D: usize = 8;
        const K: usize = 1;
        const R: usize = 8;

        let centroids = [[0f32; D]; K];
        let loadings = [identity_loadings::<D, R>()];
        let log_pi = [0f32; K];
        let psi_inv = [1f32; D];

        let region_field = F::new_unchecked(centroids, loadings, log_pi, psi_inv);

        // Plan 412 field: block = I_8, alphas = offset.
        let block_412 = identity_loadings::<D, D>();

        // Test 100 random offsets.
        let mut rng_state: u32 = 12345;
        for _case in 0..100 {
            // Simple LCG for deterministic pseudo-random offsets.
            let mut offset = [0f32; R];
            for o in &mut offset {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                *o = ((rng_state >> 8) as f32 / 65535.0) * 2.0 - 1.0; // ∈ [-1, 1)
            }

            // Region field: steer_local.
            let mut state_region = [0f32; D];
            region_field.steer_local(&mut state_region, 0, &offset);

            // Plan 412 field: apply.
            let field_412 = SubspaceSteeringField::<D, D>::new_unchecked(block_412, offset);
            let mut state_412 = [0f32; D];
            field_412.apply(&mut state_412);

            // Bit-identical check.
            for d in 0..D {
                assert_eq!(
                    state_region[d].to_bits(),
                    state_412[d].to_bits(),
                    "bit mismatch at d={d}: region={}, plan412={}, offset={}",
                    state_region[d],
                    state_412[d],
                    offset[d]
                );
            }
        }
    }

    // ── Construction validation ────────────────────────────────────────────

    #[test]
    fn new_rejects_non_unit_norm_loadings() {
        let mut bad_axis = identity_loadings::<4, 2>();
        bad_axis[0][0] = 2.0; // norm > 1
        let result =
            RegionSubspaceField::<4, 1, 2>::new([[0f32; 4]], [bad_axis], [0f32], [1f32; 4], 1e-5);
        assert_eq!(result.unwrap_err(), RegionSubspaceError::NotOrthonormal);
    }

    #[test]
    fn new_rejects_non_orthogonal_axes() {
        // Two axes that are parallel (dot = 1).
        let mut bad = identity_loadings::<4, 2>();
        bad[1] = bad[0]; // axis 1 = axis 0 → dot = 1
        let result =
            RegionSubspaceField::<4, 1, 2>::new([[0f32; 4]], [bad], [0f32], [1f32; 4], 1e-5);
        assert_eq!(result.unwrap_err(), RegionSubspaceError::NotOrthonormal);
    }

    #[test]
    fn new_rejects_non_positive_precision() {
        let loadings = identity_loadings::<4, 2>();
        let result = RegionSubspaceField::<4, 1, 2>::new(
            [[0f32; 4]],
            [loadings],
            [0f32],
            [1.0, 1.0, 1.0, 0.0], // psi_inv[3] = 0 → invalid
            1e-5,
        );
        assert_eq!(result.unwrap_err(), RegionSubspaceError::InvalidPrecision);
    }

    #[test]
    fn new_accepts_valid_field() {
        let loadings = identity_loadings::<4, 2>();
        let field =
            RegionSubspaceField::<4, 1, 2>::new([[1f32; 4]], [loadings], [0f32], [1f32; 4], 1e-5);
        assert!(field.is_ok());
    }

    // ── Commitment determinism ─────────────────────────────────────────────

    #[test]
    fn commitment_is_deterministic() {
        let loadings = identity_loadings::<4, 2>();
        let f1 = RegionSubspaceField::<4, 1, 2>::new_unchecked(
            [[1f32; 4]],
            [loadings],
            [0f32],
            [1f32; 4],
        );
        let f2 = RegionSubspaceField::<4, 1, 2>::new_unchecked(
            [[1f32; 4]],
            [loadings],
            [0f32],
            [1f32; 4],
        );
        assert_eq!(f1.commitment, f2.commitment);
    }

    #[test]
    fn commitment_sensitive_to_tamper() {
        let loadings = identity_loadings::<4, 2>();
        let f1 = RegionSubspaceField::<4, 1, 2>::new_unchecked(
            [[1f32; 4]],
            [loadings],
            [0f32],
            [1f32; 4],
        );
        // Tamper with centroids.
        let f2 = RegionSubspaceField::<4, 1, 2>::new_unchecked(
            [[2f32; 4]], // different centroid
            [loadings],
            [0f32],
            [1f32; 4],
        );
        assert_ne!(f1.commitment, f2.commitment);
    }

    #[test]
    fn verify_round_trip_passes() {
        let loadings = identity_loadings::<4, 2>();
        let field = RegionSubspaceField::<4, 1, 2>::new_unchecked(
            [[1f32; 4]],
            [loadings],
            [0f32],
            [1f32; 4],
        );
        assert!(field.verify(1e-5));
    }

    // ── Membership gates ───────────────────────────────────────────────────

    #[test]
    fn membership_gates_near_centroid_are_high() {
        // State exactly at centroid k=0. With uniform priors (log_pi=0), the
        // sigmoid gate at the centroid is sigmoid(0) = 0.5 (the midpoint). The
        // far centroid's gate is ~0. The near gate should dominate the far gate.
        let centroids = [[0f32; 4], [10f32; 4]];
        let loadings = [identity_loadings::<4, 2>(), identity_loadings::<4, 2>()];
        let field = RegionSubspaceField::<4, 2, 2>::new_unchecked(
            centroids,
            loadings,
            [0f32, 0f32],
            [1f32; 4],
        );
        let state = [0f32; 4]; // at centroid 0
        let gates = field.membership_gates(&state, 0.0);
        // Gate 0 should be at the midpoint (0.5 for uniform priors at centroid).
        // Gate 1 should be ~0 (far centroid).
        assert!(
            (gates[0] - 0.5).abs() < 0.01,
            "gate[0] should be ~0.5 at centroid with uniform prior, got {}",
            gates[0]
        );
        assert!(gates[0] > gates[1], "near gate should exceed far gate");
        assert!(
            gates[1] < 0.01,
            "gate[1] should be near 0, got {}",
            gates[1]
        );

        // With a positive prior bias (log_pi[0]=5), the gate at the centroid
        // approaches 1 (sigmoid(5) ≈ 0.993).
        let field_biased = RegionSubspaceField::<4, 2, 2>::new_unchecked(
            centroids,
            loadings,
            [5f32, 0f32],
            [1f32; 4],
        );
        let gates_biased = field_biased.membership_gates(&state, 0.0);
        assert!(
            gates_biased[0] > 0.99,
            "biased gate[0] should be near 1, got {}",
            gates_biased[0]
        );
    }

    #[test]
    fn membership_gates_at_midpoint_are_both_open() {
        // State at the midpoint of two centroids — both gates should be
        // non-trivially open (sigmoid reformulation: multi-region membership).
        let centroids = [[0f32; 4], [2f32; 4]];
        let loadings = [identity_loadings::<4, 2>(), identity_loadings::<4, 2>()];
        let field = RegionSubspaceField::<4, 2, 2>::new_unchecked(
            centroids,
            loadings,
            [0f32, 0f32],
            [1f32; 4],
        );
        let state = [1f32; 4]; // midpoint
        let gates = field.membership_gates(&state, 0.0);
        // Both gates should be meaningfully > 0 and < 1 (the sigmoid advantage).
        assert!(
            gates[0] > 0.01 && gates[0] < 0.99,
            "gate[0]={} should be in (0.01, 0.99)",
            gates[0]
        );
        assert!(
            gates[1] > 0.01 && gates[1] < 0.99,
            "gate[1]={} should be in (0.01, 0.99)",
            gates[1]
        );
    }

    // ── Centroid steering ──────────────────────────────────────────────────

    #[test]
    fn steer_centroid_alpha_zero_is_identity() {
        let centroids = [[5f32; 4]];
        let loadings = [identity_loadings::<4, 2>()];
        let field =
            RegionSubspaceField::<4, 1, 2>::new_unchecked(centroids, loadings, [0f32], [1f32; 4]);
        let original = [1f32, 2f32, 3f32, 4f32];
        let mut state = original;
        field.steer_centroid(&mut state, 0, 0.0);
        assert_eq!(state, original);
    }

    #[test]
    fn steer_centroid_alpha_one_replaces_with_centroid() {
        let centroids = [[5f32; 4]];
        let loadings = [identity_loadings::<4, 2>()];
        let field =
            RegionSubspaceField::<4, 1, 2>::new_unchecked(centroids, loadings, [0f32], [1f32; 4]);
        let mut state = [1f32, 2f32, 3f32, 4f32];
        field.steer_centroid(&mut state, 0, 1.0);
        assert_eq!(state, [5f32; 4]);
    }

    #[test]
    fn steer_centroid_alpha_half_is_midpoint() {
        let centroids = [[10f32; 4]];
        let loadings = [identity_loadings::<4, 2>()];
        let field =
            RegionSubspaceField::<4, 1, 2>::new_unchecked(centroids, loadings, [0f32], [1f32; 4]);
        let mut state = [0f32; 4];
        field.steer_centroid(&mut state, 0, 0.5);
        // (1-0.5)*0 + 0.5*10 = 5
        assert_eq!(state, [5f32; 4]);
    }

    // ── Local steering ─────────────────────────────────────────────────────

    #[test]
    fn steer_local_adds_weighted_offset() {
        let centroids = [[0f32; 4]];
        let loadings = [identity_loadings::<4, 2>()];
        let field =
            RegionSubspaceField::<4, 1, 2>::new_unchecked(centroids, loadings, [0f32], [1f32; 4]);
        let mut state = [0f32; 4];
        let offset = [3f32, 7f32]; // only first 2 dims (R=2)
        field.steer_local(&mut state, 0, &offset);
        // state[0] += 3*1 + 0 = 3, state[1] += 7*1 = 7, state[2..4] unchanged.
        assert_eq!(state, [3f32, 7f32, 0f32, 0f32]);
    }

    // ── G2: Two-mode steering produces distinct region/local effects ────────

    #[test]
    fn k2_two_regions_centroid_steering_distinct() {
        // Two regions with different centroids — centroid steering toward
        // region 0 vs region 1 should produce distinct outputs.
        let centroids = [[10f32; 4], [-10f32; 4]];
        let loadings = [identity_loadings::<4, 2>(), identity_loadings::<4, 2>()];
        let field = RegionSubspaceField::<4, 2, 2>::new_unchecked(
            centroids,
            loadings,
            [0f32, 0f32],
            [1f32; 4],
        );
        let mut state_a = [0f32; 4];
        field.steer_centroid(&mut state_a, 0, 0.5); // toward +10
        let mut state_b = [0f32; 4];
        field.steer_centroid(&mut state_b, 1, 0.5); // toward -10
        // state_a should be all positive (+5), state_b all negative (-5).
        assert!(state_a.iter().all(|&x| x > 0.0));
        assert!(state_b.iter().all(|&x| x < 0.0));
        // Distance between them should be significant.
        let dist: f32 = state_a
            .iter()
            .zip(state_b.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(dist > 5.0, "centroid-steered states too close: dist={dist}");
    }

    #[test]
    fn k2_two_regions_local_steering_distinct_subspaces() {
        // Two regions with DIFFERENT loadings — local steering with the same
        // offset should produce different outputs because the subspaces differ.
        let centroids = [[0f32; 4], [0f32; 4]];
        let mut loadings_0 = identity_loadings::<4, 2>();
        // Region 0: axes along dims 0,1
        // Region 1: axes along dims 2,3 (different subspace)
        let mut loadings_1 = [[0f32; 4]; 2];
        loadings_1[0][2] = 1.0; // axis 0 along dim 2
        loadings_1[1][3] = 1.0; // axis 1 along dim 3
        let _ = &mut loadings_0; // suppress unused warning
        let field = RegionSubspaceField::<4, 2, 2>::new_unchecked(
            centroids,
            [loadings_0, loadings_1],
            [0f32, 0f32],
            [1f32; 4],
        );
        let offset = [1f32, 1f32];
        let mut state_a = [0f32; 4];
        field.steer_local(&mut state_a, 0, &offset); // affects dims 0,1
        let mut state_b = [0f32; 4];
        field.steer_local(&mut state_b, 1, &offset); // affects dims 2,3
        // The two outputs should be in different dimensions.
        assert!(
            state_a[0].abs() > 0.0 && state_a[2].abs() < f32::EPSILON,
            "region 0 should affect dim 0, not dim 2"
        );
        assert!(
            state_b[2].abs() > 0.0 && state_b[0].abs() < f32::EPSILON,
            "region 1 should affect dim 2, not dim 0"
        );
    }

    // ── Decompose + Reconstruct round-trip ─────────────────────────────────

    #[test]
    fn decompose_reconstruct_roundtrip_identity_field() {
        // K=1, μ=0, W=I → decompose then reconstruct should return the state
        // (since the centroid is 0 and the offset reconstructs the state).
        // With identity loadings and psi_inv=1, the projector Z_k ≈ 0.5·I
        // (since Z = (I + I)^{-1} I = 0.5 I). The reconstruction is:
        //   x̂ = g·(μ + W·ẑ) / g = μ + W·ẑ = 0 + I·(0.5·x) = 0.5·x
        // So reconstruction is 0.5× the input — a known scaling from the FA
        // posterior-mean shrinkage. Verify the SHAPE is preserved (all dims
        // scaled equally), not exact recovery.
        let centroids = [[0f32; 4]];
        let loadings = [identity_loadings::<4, 4>()]; // R=D=4
        let field =
            RegionSubspaceField::<4, 1, 4>::new_unchecked(centroids, loadings, [0f32], [1f32; 4]);
        let state = [1f32, 2f32, 3f32, 4f32];
        let decomp = field.decompose(&state, 0.0);
        let recon = reconstruct(&decomp, &field);
        // All dims should be scaled by the same factor (shape preserved).
        let scale = recon[0] / state[0];
        for d in 0..4 {
            let rel_err = ((recon[d] - scale * state[d]).abs() / state[d].abs()).max(1e-6);
            assert!(
                rel_err < 1e-4,
                "dim {d}: recon={} expected scale*state={} scale={}",
                recon[d],
                scale * state[d],
                scale
            );
        }
    }

    // ── Projector sanity ───────────────────────────────────────────────────

    #[test]
    fn projector_identity_loadings_is_half_identity() {
        // For W = I_R (R=D), Ψ^{-1} = I: Z = (I + I)^{-1} I = 0.5 I.
        let centroids = [[0f32; 4]];
        let loadings = [identity_loadings::<4, 4>()];
        let field =
            RegionSubspaceField::<4, 1, 4>::new_unchecked(centroids, loadings, [0f32], [1f32; 4]);
        for r in 0..4 {
            for d in 0..4 {
                let expected = if r == d { 0.5 } else { 0.0 };
                assert!(
                    (field.projectors[0][r][d] - expected).abs() < 1e-5,
                    "projector[0][{r}][{d}]={} expected {expected}",
                    field.projectors[0][r][d]
                );
            }
        }
    }

    #[test]
    fn local_coordinates_zero_at_centroid() {
        // When state = centroid, local coords should be zero (x − μ = 0).
        let centroids = [[5f32; 4]];
        let loadings = [identity_loadings::<4, 2>()];
        let field =
            RegionSubspaceField::<4, 1, 2>::new_unchecked(centroids, loadings, [0f32], [1f32; 4]);
        let state = [5f32; 4]; // at centroid
        let coords = field.local_coordinates(&state, 0);
        for (r, coord) in coords.iter().enumerate() {
            assert!(coord.abs() < 1e-5, "coord[{r}]={coord} should be ~0");
        }
    }
}
