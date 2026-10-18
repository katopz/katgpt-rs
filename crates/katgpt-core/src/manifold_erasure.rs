//! MANCE — Manifold-Aware Concept Erasure (local tangent + spectral weighting + trust region).
//!
//! Distilled from Avitan, Goldberg, Elazar, *MANCE: Manifold Aware Concept Erasure*
//! ([arXiv:2607.03973](https://arxiv.org/abs/2607.03973), Jul 2026). See
//! `katgpt-rs/.research/409_*.md` for the open research note and
//! `katgpt-rs/.plans/426_*.md` for the execution plan.
//!
//! # The mechanism (4 steps)
//!
//! 1. **k-NN retrieval** — find the k nearest natural reference representations
//!    `X⁽⁰⁾` to the current latent `x`. These define the local manifold patch.
//! 2. **Local tangent basis** — mean-center the k neighbors, SVD the centered
//!    matrix `S_i (k×d)`, keep the top-r right singular vectors as tangent basis
//!    `B (d×r)` with singular values `σ (r)`.
//! 3. **Spectrally-weighted erasure direction** — project the gradient `u = ∇f/‖∇f‖`
//!    onto the tangent basis, weight by `σ^α`, re-normalize: `û = B·diag(σ^α)·Bᵀu`,
//!    normalized.
//! 4. **Trust-bounded step** — compute `λ = min(λ_max, ε·r_i / <x, û>)` where
//!    `r_i` is the mean neighbor distance. Apply `x̃ = x - λ·<x, û>·û`.
//!
//! # The no-harm contract
//!
//! - **Zero gradient** → `û = 0`, `λ = 0`, `x̃ = x` bit-identically.
//! - **Gradient ⊥ tangent basis** → `Bᵀu = 0`, `û = 0`, `λ = 0`, `x̃ = x` bit-identically.
//! - **Displacement bound** → `‖x̃ - x‖ ≤ ε·r_i` (trust region).
//!
//! # The probe is a CONSUMER concern
//!
//! This primitive CONSUMES a pre-computed erasure direction (the gradient/probe).
//! It does NOT train a probe. The caller provides the direction via:
//! - **MAG** (Plan 418) — unsupervised contrastive direction mining.
//! - **CNA** (Plan 087) — contrastive neuron attribution.
//! - **HLA EmotionDirections** — pre-computed affect direction vectors.
//!
//! # Reuse map
//!
//! | Operation | Source | Notes |
//! |---|---|---|
//! | Thin SVD → tangent basis | `thin_svd_into` (Plan 301, `subspace_phase_gate`) | Local k×d SVD per sample |
//! | SIMD dot products | `simd_dot_f32` (`katgpt-types/simd`) | Distance computation + projections |
//! | k-NN selection | This module | Simple linear scan (N·d for N natural points) |
//!
//! # Allocation
//!
//! [`manifold_erasure_step_into`] is zero-alloc on the hot path — the caller
//! pre-allocates a [`ManceScratch`] once and reuses it across calls.
//! [`manifold_erasure_step`] is the allocating convenience wrapper.
//!
//! # The subspace-projection family
//!
//! | Primitive | Basis | Gating | Operation |
//! |---|---|---|---|
//! | Plan 412 `subspace_steering` | Global, k-dim | Fixed α_j | INJECTION |
//! | Plan 423 `spectral_rewire` | Global SVD of W₀ | None | DECOMPOSITION (weights) |
//! | Plan 425 `tilr` | Global U_r | γ-alignment | INJECTION |
//! | **MANCE (this)** | **Local k-NN tangent** | **σ^α + ε·r_i** | **ERASURE** |
//!
//! NOTE: `basis_split::participation_ratio` (the f64 report-only wrapper) is
//! deliberately NOT re-exported at the root — the root name is the existing
//! `subspace_phase_gate` f32 primitive; the wrapper resolves via the module
//! path only.
//!
//! Plan 426: MANCE — Manifold-Aware Concept Erasure. Local tangent + spectral
//! weighting + trust-bounded erasure. Pure modelless linear algebra (k-NN +
//! thin SVD + dot-product projections). Gated on subspace_phase_gate for the
//! local tangent SVD (Plan 301 thin_svd_into).

use crate::simd::{simd_dot_f32, simd_fused_scale_acc};

// The local tangent SVD needs Plan 301's thin_svd. Gated on both features.
// In the default build, `subspace_phase_gate` is transitively enabled via
// `viable_manifold_graph` / `tucker_factorization` (both default-on).
#[cfg(feature = "subspace_phase_gate")]
use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by this module's functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ManceError {
    /// `x`, `gradient`, `natural_pool`, or `out` dimensions disagree.
    DimensionMismatch = 0,
    /// Fewer than `k` natural points available for k-NN.
    InsufficientNeighbors = 1,
    /// Gradient is all-zero (caller should handle before calling).
    ZeroGradient = 2,
    /// Config values are invalid (e.g. `k=0`, `r=0`, `epsilon<=0`).
    InvalidConfig = 3,
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for MANCE erasure. All fields are dimensionless.
///
/// # Defaults (paper §4.1, transferred across 119 settings)
///
/// | Field | Default | Meaning |
/// |---|---|---|
/// | `epsilon` | 0.1 | Trust-region ratio (displacement / local radius) |
/// | `lambda_max` | 64.0 | Hard cap on step size λ |
/// | `alpha` | 1.0 | Spectral weighting exponent (σ^α) |
/// | `k` | 8 | Number of natural neighbors for local tangent |
/// | `r` | 8 | Tangent basis dimension (top-r SVD components) |
///
/// The key insight: ε is **dimensionless** (ratio of displacement to local
/// neighborhood radius). The local `r_i` absorbs the representation scale.
/// So ε=0.1 works for both HLA (d=8) and shards (d=64) without tuning.
#[derive(Debug, Clone, Copy)]
pub struct ManceConfig {
    pub epsilon: f32,
    pub lambda_max: f32,
    pub alpha: f32,
    pub k: usize,
    pub r: usize,
}

impl Default for ManceConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            lambda_max: 64.0,
            alpha: 1.0,
            k: 8,
            r: 8,
        }
    }
}

impl ManceConfig {
    /// Validate config values. Returns `ManceError::InvalidConfig` if any
    /// field is out of range.
    #[inline]
    pub fn validate(&self) -> Result<(), ManceError> {
        if self.epsilon <= 0.0 || !self.epsilon.is_finite() {
            return Err(ManceError::InvalidConfig);
        }
        if self.lambda_max <= 0.0 || !self.lambda_max.is_finite() {
            return Err(ManceError::InvalidConfig);
        }
        if self.alpha < 0.0 || !self.alpha.is_finite() {
            return Err(ManceError::InvalidConfig);
        }
        if self.k == 0 || self.r == 0 {
            return Err(ManceError::InvalidConfig);
        }
        Ok(())
    }
}

// ─── Step info ───────────────────────────────────────────────────────────────

/// Diagnostic output from a single MANCE erasure step.
#[derive(Debug, Clone, Copy, Default)]
pub struct ManceStepInfo {
    /// The applied step size λ.
    pub lambda: f32,
    /// Actual displacement `‖x̃ - x‖`.
    pub displacement: f32,
    /// Local neighborhood radius `r_i = mean(‖x_j - x‖)`.
    pub local_radius: f32,
    /// Gradient-tangent alignment `‖Bᵀu‖ / ‖u‖` (0 = orthogonal, 1 = in-tangent).
    pub alignment: f32,
}

// ─── Scratch ─────────────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for zero-alloc MANCE steps.
///
/// Allocate once with [`ManceScratch::with_capacity`] for the largest
/// `(d, k, r)` you will use, then reuse across calls.
pub struct ManceScratch {
    /// k-NN distances (k entries) — L2 distances from x to each selected neighbor.
    pub neighbor_distances: Vec<f32>,
    /// k-NN indices into the natural pool (k entries).
    pub neighbor_indices: Vec<usize>,
    /// Centered neighbor matrix S_i (k×d, row-major). Mean-subtracted neighbors.
    pub centered_neighbors: Vec<f32>,
    /// Tangent basis B (d×r, column-major). Top-r right singular vectors.
    pub tangent_basis: Vec<f32>,
    /// Singular values σ (r entries, descending).
    pub singular_values: Vec<f32>,
    /// Projection coordinates c = Bᵀu (r entries).
    pub projection_coords: Vec<f32>,
    /// Spectrally-weighted tangent direction û (d entries).
    pub tangent_direction: Vec<f32>,
    /// Mean of k neighbors (d entries).
    pub mean_neighbor: Vec<f32>,
    /// SVD result scratch (SOA layout, reused).
    pub svd_result: SvdResultScratch,
    /// SVD work scratch (reused).
    pub svd_work: SvdScratch,
    /// The d dimension this scratch is sized for.
    pub d: usize,
    /// The k dimension this scratch is sized for.
    pub k: usize,
    /// The r dimension this scratch is sized for.
    pub r: usize,
}

impl ManceScratch {
    /// Allocate scratch sized for dimension `d`, `k` neighbors, and `r`
    /// tangent basis components.
    pub fn with_capacity(d: usize, k: usize, r: usize) -> Self {
        Self {
            neighbor_distances: vec![0.0; k],
            neighbor_indices: vec![0; k],
            centered_neighbors: vec![0.0; k * d],
            tangent_basis: vec![0.0; d * r],
            singular_values: vec![0.0; r],
            projection_coords: vec![0.0; r],
            tangent_direction: vec![0.0; d],
            mean_neighbor: vec![0.0; d],
            // SVD of k×d matrix: m_rows=k, n_cols=d. Result stores min(k,d)
            // singular triples. We size for the worst case.
            svd_result: SvdResultScratch::with_capacity(k, d),
            svd_work: SvdScratch::with_capacity(d, k),
            d,
            k,
            r,
        }
    }
}

// ─── Tangent cache (Issue 132) ───────────────────────────────────────────────

/// Pre-allocated cache for MANCE tangent basis reuse (Issue 132).
///
/// The tangent basis `B` and singular values `σ` depend **only on the k-NN
/// neighbor positions** (rows of the natural pool), not on the query point `x`.
/// When the neighbor set is stable across iterative loop rounds (which it
/// usually is — the trust-bounded step moves x at most ε·r_i = 10% of the
/// local radius per round), the SVD can be skipped entirely.
///
/// Cache validity is determined by comparing the current k-NN neighbor
/// indices with the cached set. If any index differs, the cache is invalid
/// and the SVD must be recomputed. This is both necessary and sufficient:
/// same indices → same neighbor positions → same centered matrix → same SVD.
/// The k-NN function sorts indices by index after selection (Plan 427),
/// ensuring the same neighbor set always produces the same ordering.
///
/// The movement-threshold condition proposed in Issue 132 (recompute if
/// `‖x - x_cached‖ > 0.5·r_i`) is mathematically redundant: if indices match,
/// B/σ are identical regardless of x's position. Omitted by design.
///
/// # Allocation
///
/// Pre-allocate once with [`ManceTangentCache::with_capacity`] for the largest
/// `(d, k, r)` you will use. The hot path (cached step) does only
/// `slice::copy_from_slice` — zero allocations.
pub struct ManceTangentCache {
    /// Cached neighbor indices (k entries, sorted by index for deterministic comparison).
    neighbor_indices: Vec<usize>,
    /// Cached tangent basis B (d×r, column-major). Copied to scratch on hit.
    tangent_basis: Vec<f32>,
    /// Cached singular values σ (r entries, descending). Copied to scratch on hit.
    singular_values: Vec<f32>,
    /// Whether the cache has been populated at least once.
    valid: bool,
    /// The d dimension this cache is sized for.
    d: usize,
    /// The k dimension this cache is sized for.
    k: usize,
    /// The r dimension this cache is sized for.
    r: usize,
    /// Debug: number of cache hits (for benchmark diagnostics).
    #[cfg(feature = "manifold_erasure")]
    pub cache_hits: u64,
    /// Debug: number of cache misses (for benchmark diagnostics).
    #[cfg(feature = "manifold_erasure")]
    pub cache_misses: u64,
}

impl ManceTangentCache {
    /// Allocate a cache sized for dimension `d`, `k` neighbors, and `r`
    /// tangent basis components.
    pub fn with_capacity(d: usize, k: usize, r: usize) -> Self {
        Self {
            neighbor_indices: vec![0; k],
            tangent_basis: vec![0.0; d * r],
            singular_values: vec![0.0; r],
            valid: false,
            d,
            k,
            r,
            #[cfg(feature = "manifold_erasure")]
            cache_hits: 0,
            #[cfg(feature = "manifold_erasure")]
            cache_misses: 0,
        }
    }

    /// Check whether the cache is valid for the given neighbor indices.
    ///
    /// Returns `true` if the cache has been populated AND all k indices match.
    /// The k-NN function sorts indices by index after selection (since Plan 427),
    /// so the same neighbor set always produces the same ordering — a simple
    /// slice equality suffices.
    #[inline]
    fn is_valid_for(&self, indices: &[usize]) -> bool {
        if !self.valid {
            return false;
        }
        let k = indices.len().min(self.k);
        self.neighbor_indices[..k] == indices[..k]
    }

    /// Update the cache with fresh tangent data from scratch buffers.
    ///
    /// Copies neighbor indices, tangent basis, and singular values from the
    /// provided slices into the cache's internal buffers.
    #[inline]
    fn update(&mut self, indices: &[usize], basis: &[f32], sigma: &[f32]) {
        let k = indices.len().min(self.k);
        let d_r = basis.len().min(self.d * self.r);
        let r = sigma.len().min(self.r);
        self.neighbor_indices[..k].copy_from_slice(&indices[..k]);
        self.tangent_basis[..d_r].copy_from_slice(&basis[..d_r]);
        self.singular_values[..r].copy_from_slice(&sigma[..r]);
        self.valid = true;
    }

    /// Copy cached tangent data into scratch buffers (cache hit path).
    #[inline]
    fn copy_to(&self, basis: &mut [f32], sigma: &mut [f32]) {
        let d_r = basis.len().min(self.d * self.r);
        let r = sigma.len().min(self.r);
        basis[..d_r].copy_from_slice(&self.tangent_basis[..d_r]);
        sigma[..r].copy_from_slice(&self.singular_values[..r]);
    }

    /// Reset the cache to invalid state. Useful when the natural pool changes.
    #[inline]
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
}

// ─── Core functions ──────────────────────────────────────────────────────────

/// T1.3 — Compute L2 distances from `x` to all natural representations, select
/// the k smallest. Writes k distances + k indices into `scratch`.
///
/// O(N·d) for N natural points. Uses `simd_dot_f32` for distance computation
/// via the identity `‖a-b‖² = ‖a‖² - 2<a,b> + ‖b‖²`.
///
/// Returns the k neighbor distances slice.
#[cfg(feature = "subspace_phase_gate")]
fn knn_distances_into<'s>(
    x: &[f32],
    natural_pool: &[f32], // flat N×d, row-major
    n: usize,
    d: usize,
    k: usize,
    neighbor_distances: &'s mut [f32],
    neighbor_indices: &mut [usize],
) -> Result<&'s [f32], ManceError> {
    if n < k {
        return Err(ManceError::InsufficientNeighbors);
    }
    debug_assert_eq!(x.len(), d);
    debug_assert_eq!(natural_pool.len(), n * d);
    debug_assert!(neighbor_distances.len() >= k);
    debug_assert!(neighbor_indices.len() >= k);

    // We need the k smallest distances. Use a simple partial selection:
    // compute all distances, then selection-sort the top-k.
    //
    // For the hot path (HLA d=8, N~50), this is 400 SIMD dot products + a
    // k-element selection — trivially under budget.
    //
    // We store distances in neighbor_distances[0..n] (may need to be large
    // enough). If the pool is large, we use an in-place partial sort.
    //
    // Strategy: compute distances one-by-one, maintain a max-heap of size k.
    // For simplicity and cache-friendliness on small N, we use a flat array
    // + partial selection sort.

    // Ensure the distance buffer can hold all N distances (temporary).
    // We reuse neighbor_distances as the output (k entries) and use a
    // separate approach: compute into a temp slice, then select.
    //
    // Actually, let's be smarter: compute distances into neighbor_indices
    // (as f32 reinterpreted — no, that's UB). Let's just use a simple
    // approach: maintain the k smallest in neighbor_distances/neighbor_indices,
    // replacing the max each time.

    let dists = &mut neighbor_distances[..k];
    let idxs = &mut neighbor_indices[..k];

    // Initialize with the first k points.
    let x_norm_sq = simd_dot_f32(x, x, d);
    for i in 0..k {
        let row = &natural_pool[i * d..(i + 1) * d];
        let dot = simd_dot_f32(x, row, d);
        let row_norm_sq = simd_dot_f32(row, row, d);
        let dist_sq = x_norm_sq - 2.0 * dot + row_norm_sq;
        dists[i] = dist_sq.max(0.0); // numerical safety
        idxs[i] = i;
    }

    // Find the max in the initial k.
    let mut max_idx = 0;
    for i in 1..k {
        if dists[i] > dists[max_idx] {
            max_idx = i;
        }
    }

    // Scan the rest, replacing the max when a smaller distance is found.
    for i in k..n {
        let row = &natural_pool[i * d..(i + 1) * d];
        let dot = simd_dot_f32(x, row, d);
        let row_norm_sq = simd_dot_f32(row, row, d);
        let dist_sq = x_norm_sq - 2.0 * dot + row_norm_sq;
        let dist_sq = dist_sq.max(0.0);
        if dist_sq < dists[max_idx] {
            dists[max_idx] = dist_sq;
            idxs[max_idx] = i;
            // Re-find the max.
            max_idx = 0;
            for j in 1..k {
                if dists[j] > dists[max_idx] {
                    max_idx = j;
                }
            }
        }
    }

    // Convert squared distances to L2 distances.
    for d in dists[..k].iter_mut() {
        *d = d.sqrt();
    }

    // Sort by index (insertion sort — k is tiny, typically 8-16).
    // This ensures deterministic row ordering for the SVD: the same neighbor
    // set always produces the same centered-matrix row order, which makes the
    // SVD output reproducible and enables bit-identical tangent caching
    // (Issue 132). Without this, the max-heap replacement strategy can return
    // the same neighbor set in different orders across calls.
    for i in 1..k {
        let idx_key = idxs[i];
        let dist_key = dists[i];
        let mut j = i;
        while j > 0 && idxs[j - 1] > idx_key {
            idxs[j] = idxs[j - 1];
            dists[j] = dists[j - 1];
            j -= 1;
        }
        idxs[j] = idx_key;
        dists[j] = dist_key;
    }

    Ok(&neighbor_distances[..k])
}

/// T1.4 — Estimate the local tangent basis at `x` from its k natural neighbors.
///
/// Mean-centers the k neighbors → forms `S_i (k×d)` → SVD → keeps top-r right
/// singular vectors as tangent basis `B (d×r)` + singular values `σ (r)`.
///
/// Returns `(tangent_basis, singular_values)` where:
/// - `tangent_basis` is `d×r` column-major (column j = j-th right singular vector)
/// - `singular_values` is length `r` (descending)
#[cfg(feature = "subspace_phase_gate")]
#[allow(clippy::too_many_arguments)] // buffer-passing API: natural_pool, neighbor_indices, d, r, + 5 output slices
fn estimate_local_tangent_into(
    natural_pool: &[f32],
    neighbor_indices: &[usize],
    d: usize,
    r: usize,
    centered_neighbors: &mut [f32],
    mean_neighbor: &mut [f32],
    tangent_basis: &mut [f32],
    singular_values: &mut [f32],
    svd_result: &mut SvdResultScratch,
    svd_work: &mut SvdScratch,
) -> Result<(), ManceError> {
    let k = neighbor_indices.len();
    debug_assert!(centered_neighbors.len() >= k * d);
    debug_assert!(mean_neighbor.len() >= d);

    // Compute mean of k neighbors.
    let mean = &mut mean_neighbor[..d];
    for v in mean.iter_mut() {
        *v = 0.0;
    }
    for &idx in neighbor_indices {
        let row = &natural_pool[idx * d..(idx + 1) * d];
        // `simd_fused_scale_acc(dst, src, 1.0)` is element-wise
        // `dst[j] = fma(1.0, src[j], dst[j])`. `1.0 * src[j]` is exact, so the
        // FMA rounds exactly once on `src[j] + dst[j]` — identical to the scalar
        // `mean[j] += row[j]`. Element-wise (not a reduction), so lane order is
        // irrelevant and the accumulation order across neighbours is unchanged.
        simd_fused_scale_acc(mean, row, 1.0, d);
    }
    let inv_k = 1.0 / k as f32;
    for v in mean.iter_mut() {
        *v *= inv_k;
    }

    // Form centered matrix S_i (k×d, row-major): each row = neighbor - mean.
    let centered = &mut centered_neighbors[..k * d];
    // Pre-slice the destination row once per neighbour instead of recomputing
    // `i * d + j` (and its bounds check) per element. `centered` is exactly
    // `k * d` long and `neighbor_indices.len() == k` (asserted above), so
    // `chunks_exact_mut(d)` yields exactly as many rows as indices.
    for (&idx, dst) in neighbor_indices.iter().zip(centered.chunks_exact_mut(d)) {
        let row = &natural_pool[idx * d..(idx + 1) * d];
        for (o, (&x, &m)) in dst.iter_mut().zip(row.iter().zip(mean.iter())) {
            *o = x - m;
        }
    }

    // Thin SVD of S_i (k×d). Writes into svd_result (SOA, zero alloc).
    thin_svd_into(centered, k, d, svd_result, svd_work);

    let n_sv = svd_result.len().min(r);
    if n_sv == 0 {
        return Err(ManceError::InsufficientNeighbors);
    }

    // Extract top-r right singular vectors (columns of V) and singular values.
    // V columns are stored column-major in svd_result: column j at
    // [j * n_cols .. (j+1) * n_cols], where n_cols = d.
    let basis = &mut tangent_basis[..d * r];
    let sigmas = &mut singular_values[..r];
    for j in 0..n_sv {
        let sv = svd_result.right_singular_vector(j);
        let dest = &mut basis[j * d..(j + 1) * d];
        dest.copy_from_slice(sv);
        sigmas[j] = svd_result.singular_value(j);
    }
    // Zero out any remaining columns if r > n_sv.
    for j in n_sv..r {
        for v in basis[j * d..(j + 1) * d].iter_mut() {
            *v = 0.0;
        }
        sigmas[j] = 0.0;
    }

    Ok(())
}

/// T1.5 — Compute the spectrally-weighted tangent erasure direction.
///
/// 1. Normalize gradient: `u = ∇f / ‖∇f‖`
/// 2. Project: `c = Bᵀu` (r dot products)
/// 3. Spectrally weight: `d = B · diag(σ^α) · c` (weighted sum of basis columns)
/// 4. Normalize: `û = d / ‖d‖`
///
/// Writes the result into `scratch.tangent_direction` (d entries).
/// Returns the tangent direction slice and the alignment ratio.
///
/// # No-harm
///
/// If the gradient is zero or orthogonal to the tangent basis (‖d‖ ≈ 0),
/// the direction is set to all-zeros and alignment = 0.0. The caller should
/// check alignment and skip the step (λ=0).
#[cfg(feature = "subspace_phase_gate")]
#[allow(clippy::too_many_arguments)] // buffer-passing API: gradient, basis, sigma, 2 scalars, 2 output slices
pub fn tangent_erasure_direction_into(
    gradient: &[f32],
    basis: &[f32], // d×r, column-major
    sigma: &[f32], // r entries
    alpha: f32,
    d: usize,
    r: usize,
    projection_coords: &mut [f32],
    tangent_direction: &mut [f32],
) -> f32 {
    debug_assert_eq!(gradient.len(), d);
    debug_assert_eq!(basis.len(), d * r);
    debug_assert_eq!(sigma.len(), r);

    let coords = &mut projection_coords[..r];
    let direction = &mut tangent_direction[..d];

    // Step 0: normalize gradient.
    let grad_norm = simd_dot_f32(gradient, gradient, d).sqrt();
    if grad_norm < 1e-12 {
        // Zero gradient — no-harm.
        for v in direction.iter_mut() {
            *v = 0.0;
        }
        return 0.0;
    }
    let inv_grad_norm = 1.0 / grad_norm;

    // Step 1: c = Bᵀu (projection onto each basis vector).
    // u = gradient * inv_grad_norm. c_j = <u, basis_col_j>.
    for j in 0..r {
        let col = &basis[j * d..(j + 1) * d];
        // Compute <gradient, col> / grad_norm
        let dot = simd_dot_f32(gradient, col, d) * inv_grad_norm;
        coords[j] = dot;
    }

    let coord_sq_sum = simd_dot_f32(coords, coords, r);
    let alignment = coord_sq_sum.sqrt();

    // Step 2: d_vec = B · diag(σ^α) · c = Σ_j (σ_j^α · c_j · basis_col_j)
    for v in direction.iter_mut() {
        *v = 0.0;
    }
    // `alpha` is loop-invariant and its documented default is exactly 1.0, where
    // `powf` is an identity that still costs a full libm call per basis vector.
    // IEEE 754 requires `pow(x, 1) == x` exactly for every finite x (and the
    // sigmas here are non-negative singular values), so the fast path yields the
    // bit-identical weight.
    let alpha_is_one = alpha == 1.0;
    for j in 0..r {
        let weight = if alpha_is_one {
            sigma[j]
        } else {
            sigma[j].powf(alpha)
        };
        let coeff = weight * coords[j];
        if coeff == 0.0 {
            continue;
        }
        let col = &basis[j * d..(j + 1) * d];
        // AXPBY: direction += coeff * col. SIMD via fused multiply-add.
        simd_fused_scale_acc(direction, col, coeff, d);
    }

    // Step 3: normalize û = d / ‖d‖.
    let dir_norm = simd_dot_f32(direction, direction, d).sqrt();
    if dir_norm < 1e-12 {
        // Direction orthogonal to tangent basis — no-harm.
        for v in direction.iter_mut() {
            *v = 0.0;
        }
        return 0.0;
    }
    let inv_dir_norm = 1.0 / dir_norm;
    for v in direction.iter_mut() {
        *v *= inv_dir_norm;
    }

    alignment
}

/// T1.6 — Compute the trust-region-bounded step size λ.
///
/// `r_i = mean(‖x_j - x‖)` over k neighbors.
/// `λ = min(λ_max, ε · r_i / |<x, û>|)`.
///
/// If `<x, û> ≈ 0` (direction orthogonal to x), `λ = 0` (no-harm).
#[inline]
fn local_radius_step(
    x: &[f32],
    direction: &[f32],          // û, unit norm
    neighbor_distances: &[f32], // L2 distances to k neighbors
    epsilon: f32,
    lambda_max: f32,
    d: usize,
) -> f32 {
    debug_assert_eq!(x.len(), d);
    debug_assert_eq!(direction.len(), d);

    // Local radius: mean of neighbor distances.
    let r_i = if neighbor_distances.is_empty() {
        0.0
    } else {
        let sum: f32 = neighbor_distances.iter().sum();
        sum / neighbor_distances.len() as f32
    };

    if r_i < 1e-12 {
        return 0.0;
    }

    // Projection of x onto û.
    let x_proj = simd_dot_f32(x, direction, d);

    if x_proj.abs() < 1e-12 {
        // Direction orthogonal to x — no erasure needed.
        return 0.0;
    }

    let lambda = epsilon * r_i / x_proj.abs();
    lambda.min(lambda_max)
}

/// T1.7 — Perform a single MANCE erasure step (zero-alloc hot path).
///
/// Orchestrates T1.3→T1.6: k-NN → local tangent → spectral weighting →
/// trust region → apply `out = x - λ·<x, û>·û`.
///
/// All scratch is reused. The caller pre-allocates [`ManceScratch`] once.
///
/// # Arguments
///
/// * `x` — input latent vector (d entries)
/// * `gradient` — erasure direction / probe output (d entries, need not be unit)
/// * `natural_pool` — flat N×d natural reference representations (row-major)
/// * `n` — number of natural points
/// * `config` — MANCE configuration
/// * `scratch` — pre-allocated scratch (sized for d, k, r)
/// * `out` — output buffer (d entries, may alias `x`)
///
/// # Returns
///
/// [`ManceStepInfo`] with diagnostics, or [`ManceError`] on invalid input.
#[cfg(feature = "subspace_phase_gate")]
pub fn manifold_erasure_step_into(
    x: &[f32],
    gradient: &[f32],
    natural_pool: &[f32],
    n: usize,
    config: &ManceConfig,
    scratch: &mut ManceScratch,
    out: &mut [f32],
) -> Result<ManceStepInfo, ManceError> {
    let d = x.len();
    config.validate()?;

    if gradient.len() != d || out.len() != d || natural_pool.len() != n * d {
        return Err(ManceError::DimensionMismatch);
    }
    if n < config.k {
        return Err(ManceError::InsufficientNeighbors);
    }

    let k = config.k.min(scratch.k);
    let r = config.r.min(scratch.r);

    // Check for zero gradient early.
    let grad_norm_sq = simd_dot_f32(gradient, gradient, d);
    if grad_norm_sq < 1e-24 {
        // Zero gradient — no-harm, copy x to out.
        out.copy_from_slice(x);
        return Ok(ManceStepInfo {
            lambda: 0.0,
            displacement: 0.0,
            local_radius: 0.0,
            alignment: 0.0,
        });
    }

    // T1.3: k-NN retrieval.
    // Split borrows: pass individual scratch slices to avoid holding a
    // &mut ManceScratch across multiple calls.

    let distances = knn_distances_into(
        x,
        natural_pool,
        n,
        d,
        k,
        &mut scratch.neighbor_distances,
        &mut scratch.neighbor_indices,
    )?;

    // T1.4: local tangent basis.
    // `distances` borrows `scratch.neighbor_distances` but `neighbor_indices`
    // is a separate field — split borrow allows concurrent access.
    estimate_local_tangent_into(
        natural_pool,
        &scratch.neighbor_indices[..k],
        d,
        r,
        &mut scratch.centered_neighbors,
        &mut scratch.mean_neighbor,
        &mut scratch.tangent_basis,
        &mut scratch.singular_values,
        &mut scratch.svd_result,
        &mut scratch.svd_work,
    )?;

    // T1.5: spectrally-weighted erasure direction.
    let alignment = tangent_erasure_direction_into(
        gradient,
        &scratch.tangent_basis,
        &scratch.singular_values,
        config.alpha,
        d,
        r,
        &mut scratch.projection_coords,
        &mut scratch.tangent_direction,
    );

    // Check for orthogonal direction (no-harm).
    if alignment < 1e-12 {
        out.copy_from_slice(x);
        // Compute local radius for diagnostics.
        let r_i = if distances.is_empty() {
            0.0
        } else {
            distances.iter().sum::<f32>() / distances.len() as f32
        };
        return Ok(ManceStepInfo {
            lambda: 0.0,
            displacement: 0.0,
            local_radius: r_i,
            alignment: 0.0,
        });
    }

    // T1.6: trust-bounded step size.
    let direction = &scratch.tangent_direction;
    let lambda = local_radius_step(
        x,
        direction,
        distances,
        config.epsilon,
        config.lambda_max,
        d,
    );

    // Apply: out = x - λ · <x, û> · û
    let x_proj = simd_dot_f32(x, direction, d);
    let scale = lambda * x_proj;
    for i in 0..d {
        out[i] = x[i] - scale * direction[i];
    }

    // Diagnostics.
    let displacement = scale.abs() * simd_dot_f32(direction, direction, d).sqrt();
    let r_i = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    };

    Ok(ManceStepInfo {
        lambda,
        displacement,
        local_radius: r_i,
        alignment,
    })
}

/// T1.8 — Allocating convenience wrapper for non-hot paths.
///
/// Allocates a fresh [`ManceScratch`] internally. For hot paths, use
/// [`manifold_erasure_step_into`] with a reused scratch.
#[cfg(feature = "subspace_phase_gate")]
pub fn manifold_erasure_step(
    x: &[f32],
    gradient: &[f32],
    natural_pool: &[f32],
    n: usize,
    config: &ManceConfig,
) -> Result<(Vec<f32>, ManceStepInfo), ManceError> {
    let d = x.len();
    let mut scratch = ManceScratch::with_capacity(d, config.k, config.r);
    let mut out = vec![0.0; d];
    let info =
        manifold_erasure_step_into(x, gradient, natural_pool, n, config, &mut scratch, &mut out)?;
    Ok((out, info))
}

/// T1.7c — Cached MANCE erasure step (Issue 132).
///
/// Identical to [`manifold_erasure_step_into`] but skips the tangent SVD when
/// the k-NN neighbor set hasn't changed since the last call (cache hit).
///
/// The cache stores the tangent basis `B` and singular values `σ` keyed on
/// neighbor indices. On a cache hit, only `O(d·r)` copying is needed instead
/// of `O(k·d·min(k,d))` for the SVD. The k-NN retrieval still runs every call
/// (needed for fresh `r_i` distances).
///
/// # Correctness
///
/// Results are **bit-identical** to [`manifold_erasure_step_into`] when the
/// cache is valid: same B/σ (same neighbor positions) + same r_i (fresh k-NN
/// distances) → same direction → same step.
///
/// # Allocation
///
/// Zero allocations on the hot path. The cache is pre-allocated via
/// [`ManceTangentCache::with_capacity`].
#[cfg(feature = "subspace_phase_gate")]
#[allow(clippy::too_many_arguments)] // buffer-passing API: x, gradient, pool, 2 scalars/config, scratch, cache, out
pub fn manifold_erasure_step_cached_into(
    x: &[f32],
    gradient: &[f32],
    natural_pool: &[f32],
    n: usize,
    config: &ManceConfig,
    scratch: &mut ManceScratch,
    cache: &mut ManceTangentCache,
    out: &mut [f32],
) -> Result<ManceStepInfo, ManceError> {
    let d = x.len();
    config.validate()?;

    if gradient.len() != d || out.len() != d || natural_pool.len() != n * d {
        return Err(ManceError::DimensionMismatch);
    }
    if n < config.k {
        return Err(ManceError::InsufficientNeighbors);
    }

    let k = config.k.min(scratch.k);
    let r = config.r.min(scratch.r);

    // Check for zero gradient early.
    let grad_norm_sq = simd_dot_f32(gradient, gradient, d);
    if grad_norm_sq < 1e-24 {
        out.copy_from_slice(x);
        return Ok(ManceStepInfo {
            lambda: 0.0,
            displacement: 0.0,
            local_radius: 0.0,
            alignment: 0.0,
        });
    }

    // T1.3: k-NN retrieval (always — needed for fresh r_i distances).
    let distances = knn_distances_into(
        x,
        natural_pool,
        n,
        d,
        k,
        &mut scratch.neighbor_distances,
        &mut scratch.neighbor_indices,
    )?;

    // Cache check: skip SVD if neighbor indices match.
    let cache_hit = cache.is_valid_for(&scratch.neighbor_indices[..k]);
    if cache_hit {
        #[cfg(feature = "manifold_erasure")]
        {
            cache.cache_hits += 1;
        }
        // Reuse cached B/σ — copy into scratch.
        cache.copy_to(&mut scratch.tangent_basis, &mut scratch.singular_values);
    } else {
        #[cfg(feature = "manifold_erasure")]
        {
            cache.cache_misses += 1;
        }
        // Cache miss — recompute tangent basis via SVD.
        estimate_local_tangent_into(
            natural_pool,
            &scratch.neighbor_indices[..k],
            d,
            r,
            &mut scratch.centered_neighbors,
            &mut scratch.mean_neighbor,
            &mut scratch.tangent_basis,
            &mut scratch.singular_values,
            &mut scratch.svd_result,
            &mut scratch.svd_work,
        )?;
        // Update cache with fresh tangent data.
        cache.update(
            &scratch.neighbor_indices[..k],
            &scratch.tangent_basis,
            &scratch.singular_values,
        );
    }

    // T1.5: spectrally-weighted erasure direction.
    let alignment = tangent_erasure_direction_into(
        gradient,
        &scratch.tangent_basis,
        &scratch.singular_values,
        config.alpha,
        d,
        r,
        &mut scratch.projection_coords,
        &mut scratch.tangent_direction,
    );

    // Check for orthogonal direction (no-harm).
    if alignment < 1e-12 {
        out.copy_from_slice(x);
        let r_i = if distances.is_empty() {
            0.0
        } else {
            distances.iter().sum::<f32>() / distances.len() as f32
        };
        return Ok(ManceStepInfo {
            lambda: 0.0,
            displacement: 0.0,
            local_radius: r_i,
            alignment: 0.0,
        });
    }

    // T1.6: trust-bounded step size.
    let direction = &scratch.tangent_direction;
    let lambda = local_radius_step(
        x,
        direction,
        distances,
        config.epsilon,
        config.lambda_max,
        d,
    );

    // Apply: out = x - λ · <x, û> · û
    let x_proj = simd_dot_f32(x, direction, d);
    let scale = lambda * x_proj;
    for i in 0..d {
        out[i] = x[i] - scale * direction[i];
    }

    // Diagnostics.
    let displacement = scale.abs() * simd_dot_f32(direction, direction, d).sqrt();
    let r_i = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    };

    Ok(ManceStepInfo {
        lambda,
        displacement,
        local_radius: r_i,
        alignment,
    })
}

// ─── Phase 2: Iterative loop + closed-form preprocessing ─────────────────────

/// T2.1 — Iterative MANCE erasure loop (zero-alloc hot path).
///
/// Applies [`manifold_erasure_step_into`] for `n_rounds` rounds. The
/// `gradient_fn` provides the erasure direction at each round — the caller's
/// probe (MAG/CNA/EmotionDirections). This is the modelless analog of MANCE's
/// iterative loop with probe refit.
///
/// The output `out` is updated in-place each round, and the gradient function
/// receives the current state. If the caller wants a static direction (no
/// refit), pass a closure that ignores its input.
#[cfg(feature = "subspace_phase_gate")]
#[allow(clippy::too_many_arguments)] // buffer-passing API: x, gradient_fn, pool, n, config, n_rounds, scratch, out
pub fn manifold_erasure_loop_into<F>(
    x: &[f32],
    mut gradient_fn: F,
    natural_pool: &[f32],
    n: usize,
    config: &ManceConfig,
    n_rounds: usize,
    scratch: &mut ManceScratch,
    out: &mut [f32],
) -> Result<Vec<ManceStepInfo>, ManceError>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let d = x.len();
    if out.len() != d {
        return Err(ManceError::DimensionMismatch);
    }
    out.copy_from_slice(x);

    let mut grad_buf = vec![0.0f32; d];
    // Hoisted outside the loop: avoid per-round allocation. `current` must be a
    // distinct buffer from `out` because `manifold_erasure_step_into` reads
    // `&current` and writes `&mut out` simultaneously.
    let mut current = vec![0.0f32; d];
    let mut infos = Vec::with_capacity(n_rounds);

    for _ in 0..n_rounds {
        gradient_fn(out, &mut grad_buf);
        current.copy_from_slice(out);
        let info = manifold_erasure_step_into(
            &current,
            &grad_buf,
            natural_pool,
            n,
            config,
            scratch,
            out, // in-place: out is both input and output
        )?;
        infos.push(info);

        // Early termination: if lambda is 0, further rounds won't help.
        if info.lambda == 0.0 {
            break;
        }
    }

    Ok(infos)
}

/// T2.1c — Cached iterative MANCE erasure loop (Issue 132).
///
/// Identical to [`manifold_erasure_loop_into`] but uses a [`ManceTangentCache`]
/// to skip the tangent SVD when the k-NN neighbor set is stable across rounds.
///
/// For a 10-round loop where neighbors don't change (the common case with
/// ε=0.1 trust region), this skips ~9 of 10 SVDs.
///
/// # Correctness
///
/// Results are **bit-identical** to [`manifold_erasure_loop_into`]: the cache
/// only skips the SVD when the neighbor set is identical, in which case B/σ
/// are the same. All other computations (k-NN, spectral weighting, trust region)
/// use fresh values every round.
///
/// # Allocation
///
/// The per-call `grad_buf` and `current` buffers are hoisted outside the
/// round loop (one alloc each, not `n_rounds` allocs each). The SVD-skip
/// (the optimization) adds zero allocations.
#[cfg(feature = "subspace_phase_gate")]
#[allow(clippy::too_many_arguments)] // buffer-passing API: x, gradient_fn, pool, n, config, n_rounds, scratch, cache, out
pub fn manifold_erasure_loop_cached_into<F>(
    x: &[f32],
    mut gradient_fn: F,
    natural_pool: &[f32],
    n: usize,
    config: &ManceConfig,
    n_rounds: usize,
    scratch: &mut ManceScratch,
    cache: &mut ManceTangentCache,
    out: &mut [f32],
) -> Result<Vec<ManceStepInfo>, ManceError>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let d = x.len();
    if out.len() != d {
        return Err(ManceError::DimensionMismatch);
    }
    out.copy_from_slice(x);

    let mut grad_buf = vec![0.0f32; d];
    // Hoisted outside the loop: avoid per-round allocation. See
    // `manifold_erasure_loop_into` for the aliasing rationale.
    let mut current = vec![0.0f32; d];
    let mut infos = Vec::with_capacity(n_rounds);

    for _ in 0..n_rounds {
        gradient_fn(out, &mut grad_buf);
        current.copy_from_slice(out);
        let info = manifold_erasure_step_cached_into(
            &current,
            &grad_buf,
            natural_pool,
            n,
            config,
            scratch,
            cache,
            out, // in-place: out is both input and output
        )?;
        infos.push(info);

        // Early termination: if lambda is 0, further rounds won't help.
        if info.lambda == 0.0 {
            break;
        }
    }

    Ok(infos)
}

/// T2.2 — LEACE rank-1 closed-form erasure (MANCE+ preprocessing).
///
/// Projects out the class-mean difference direction:
/// `out = x - (<x, d_mean> / ‖d_mean‖²) · d_mean`
/// where `d_mean = μ₊ - μ₋`.
///
/// This is a zero-alloc operation using the provided scratch for the direction
/// buffer.
#[cfg(feature = "subspace_phase_gate")]
pub fn leace_first_moment_into(
    x: &[f32],
    class_mean_pos: &[f32], // μ₊
    class_mean_neg: &[f32], // μ₋
    d_mean_buf: &mut [f32],
    out: &mut [f32],
) -> Result<(), ManceError> {
    let d = x.len();
    if class_mean_pos.len() != d || class_mean_neg.len() != d || out.len() != d {
        return Err(ManceError::DimensionMismatch);
    }

    // d_mean = μ₊ - μ₋, stored in d_mean_buf.
    let d_mean = &mut d_mean_buf[..d];
    for i in 0..d {
        d_mean[i] = class_mean_pos[i] - class_mean_neg[i];
    }

    let d_mean_norm_sq = simd_dot_f32(d_mean, d_mean, d);
    if d_mean_norm_sq < 1e-24 {
        // Classes have identical means — nothing to erase.
        out.copy_from_slice(x);
        return Ok(());
    }

    let x_proj = simd_dot_f32(x, d_mean, d);
    let coeff = x_proj / d_mean_norm_sq;

    for i in 0..d {
        out[i] = x[i] - coeff * d_mean[i];
    }

    Ok(())
}

/// T2.3 — CovMatch rank-2 closed-form erasure (MANCE++ preprocessing).
///
/// Projects out the top-2 eigenvectors of ΔΣ = Σ₊ - Σ₋ (the covariance
/// asymmetry). The two eigenvectors are orthonormalized with the mean direction
/// via Gram-Schmidt before projection.
///
/// `delta_sigma_top2_eigvecs` should contain 2 eigenvectors, each of length `d`,
/// flattened as `[v1_0, v1_1, ..., v1_{d-1}, v2_0, v2_1, ..., v2_{d-1}]`.
#[cfg(feature = "subspace_phase_gate")]
pub fn covmatch_second_moment_into(
    x: &[f32],
    delta_sigma_top2_eigvecs: &[f32], // 2×d, row-major
    v2_buf: &mut [f32],
    out: &mut [f32],
) -> Result<(), ManceError> {
    let d = x.len();
    if delta_sigma_top2_eigvecs.len() != 2 * d || out.len() != d {
        return Err(ManceError::DimensionMismatch);
    }

    // Get the two eigenvectors.
    let v1 = &delta_sigma_top2_eigvecs[..d];
    let v2_raw = &delta_sigma_top2_eigvecs[d..2 * d];

    // Gram-Schmidt orthonormalize: v1 stays, v2 = v2_raw - <v2_raw, v1>·v1.
    let v2 = &mut v2_buf[..d];
    let dot_v2_v1 = simd_dot_f32(v2_raw, v1, d);
    for i in 0..d {
        v2[i] = v2_raw[i] - dot_v2_v1 * v1[i];
    }

    // Normalize v1 and v2.
    let v1_norm = simd_dot_f32(v1, v1, d).sqrt();
    let v2_norm = simd_dot_f32(v2, v2, d).sqrt();

    if v1_norm < 1e-12 && v2_norm < 1e-12 {
        // Both directions are zero — nothing to erase.
        out.copy_from_slice(x);
        return Ok(());
    }

    // Project out both directions: out = x - <x,ê1>·ê1 - <x,ê2>·ê2.
    out.copy_from_slice(x);

    if v1_norm >= 1e-12 {
        let inv_norm = 1.0 / v1_norm;
        let x_proj = simd_dot_f32(x, v1, d) * inv_norm;
        for i in 0..d {
            out[i] -= x_proj * v1[i] * inv_norm;
        }
    }

    if v2_norm >= 1e-12 {
        let inv_norm = 1.0 / v2_norm;
        let x_proj = simd_dot_f32(x, v2, d) * inv_norm;
        for i in 0..d {
            out[i] -= x_proj * v2[i] * inv_norm;
        }
    }

    Ok(())
}

/// T2.4 — MANCE+ step (LEACE preprocessing + MANCE loop).
///
/// First applies LEACE rank-1 erasure, then runs the MANCE iterative loop
/// on the preprocessed state.
#[cfg(feature = "subspace_phase_gate")]
#[allow(clippy::too_many_arguments)] // buffer-passing API: x, 2 class means, gradient_fn, pool, n, config, n_rounds, scratch, out
pub fn mance_plus_step_into<F>(
    x: &[f32],
    class_mean_pos: &[f32],
    class_mean_neg: &[f32],
    gradient_fn: F,
    natural_pool: &[f32],
    n: usize,
    config: &ManceConfig,
    n_rounds: usize,
    scratch: &mut ManceScratch,
    out: &mut [f32],
) -> Result<Vec<ManceStepInfo>, ManceError>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let d = x.len();
    if out.len() != d {
        return Err(ManceError::DimensionMismatch);
    }

    // LEACE preprocessing.
    leace_first_moment_into(
        x,
        class_mean_pos,
        class_mean_neg,
        &mut scratch.tangent_direction,
        out,
    )?;

    // MANCE loop on preprocessed state.
    // `manifold_erasure_loop_into` overwrites `out` from `x` on entry, so
    // the clone is semantically dead — but the borrow checker can't prove
    // `out` isn't aliased by the `&mut out` write inside the loop, so we
    // keep a one-time snapshot. (Single allocation per call, not per round.)
    let preprocessed = out.to_vec();
    manifold_erasure_loop_into(
        &preprocessed,
        gradient_fn,
        natural_pool,
        n,
        config,
        n_rounds,
        scratch,
        out,
    )
}

/// T2.4 — MANCE++ step (LEACE + CovMatch preprocessing + MANCE loop).
///
/// First applies LEACE rank-1 erasure, then CovMatch rank-2 erasure,
/// then runs the MANCE iterative loop on the doubly-preprocessed state.
#[cfg(feature = "subspace_phase_gate")]
#[allow(clippy::too_many_arguments)] // buffer-passing API: x, 2 class means, eigvecs, gradient_fn, pool, n, config, n_rounds, scratch, out
pub fn mance_plus_plus_step_into<F>(
    x: &[f32],
    class_mean_pos: &[f32],
    class_mean_neg: &[f32],
    delta_sigma_top2_eigvecs: &[f32],
    gradient_fn: F,
    natural_pool: &[f32],
    n: usize,
    config: &ManceConfig,
    n_rounds: usize,
    scratch: &mut ManceScratch,
    out: &mut [f32],
) -> Result<Vec<ManceStepInfo>, ManceError>
where
    F: FnMut(&[f32], &mut [f32]),
{
    let d = x.len();
    if out.len() != d {
        return Err(ManceError::DimensionMismatch);
    }

    // LEACE preprocessing.
    leace_first_moment_into(
        x,
        class_mean_pos,
        class_mean_neg,
        &mut scratch.tangent_direction,
        out,
    )?;

    // CovMatch preprocessing (in-place on the LEACE output).
    // Need a temp buffer for the Gram-Schmidt orthonormalized v2.
    // Use a local stack-allocated approach: copy out, covmatch into out.
    let out_copy = out.to_vec();
    covmatch_second_moment_into(
        &out_copy,
        delta_sigma_top2_eigvecs,
        &mut scratch.mean_neighbor,
        out,
    )?;

    // MANCE loop on preprocessed state.
    // `manifold_erasure_loop_into` overwrites `out` from `x` on entry, so
    // the clone is semantically dead — but the borrow checker can't prove
    // `out` isn't aliased by the `&mut out` write inside the loop, so we
    // keep a one-time snapshot. (Single allocation per call, not per round.)
    let preprocessed = out.to_vec();
    manifold_erasure_loop_into(
        &preprocessed,
        gradient_fn,
        natural_pool,
        n,
        config,
        n_rounds,
        scratch,
        out,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "subspace_phase_gate")]
mod tests {
    use super::*;

    /// Helper: create a simple natural pool of N points in d dimensions.
    fn make_pool(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut pool = vec![0.0f32; n * d];
        let mut s = seed;
        for i in 0..n {
            for j in 0..d {
                // Simple LCG for deterministic pseudo-random data.
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let r = ((s >> 33) as f32) / (1u64 << 31) as f32;
                pool[i * d + j] = r * 2.0 - 1.0; // [-1, 1)
            }
        }
        pool
    }

    // T1.10: Unit tests

    /// `knn_returns_correct_neighbors` — known distances, verify k smallest selected.
    #[test]
    fn knn_returns_correct_neighbors() {
        let d = 4;
        let n = 10;
        let k = 3;
        // Pool: points at known positions.
        let pool: Vec<f32> = (0..n)
            .flat_map(|i| {
                let dist = (i + 1) as f32; // point i is at distance i+1 from origin
                vec![dist, 0.0, 0.0, 0.0]
            })
            .collect();
        let x = vec![0.0; d];
        let mut scratch = ManceScratch::with_capacity(d, k, k);

        let distances = knn_distances_into(
            &x,
            &pool,
            n,
            d,
            k,
            &mut scratch.neighbor_distances,
            &mut scratch.neighbor_indices,
        )
        .unwrap();

        // The 3 nearest points should be at distances 1, 2, 3.
        let mut sorted = distances.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(sorted, vec![1.0, 2.0, 3.0]);
    }

    /// `tangent_basis_orthonormal` — verify BᵀB ≈ I_r.
    #[test]
    fn tangent_basis_orthonormal() {
        let d = 4;
        let k = 8;
        let r = 2;
        let pool = make_pool(k, d, 42);
        let mut scratch = ManceScratch::with_capacity(d, k, r);

        let indices: Vec<usize> = (0..k).collect();
        estimate_local_tangent_into(
            &pool,
            &indices,
            d,
            r,
            &mut scratch.centered_neighbors,
            &mut scratch.mean_neighbor,
            &mut scratch.tangent_basis,
            &mut scratch.singular_values,
            &mut scratch.svd_result,
            &mut scratch.svd_work,
        )
        .unwrap();
        let basis = &scratch.tangent_basis;

        // Check BᵀB ≈ I_r (each column should be unit norm and mutually orthogonal).
        for j in 0..r {
            let col_j = &basis[j * d..(j + 1) * d];
            let norm = simd_dot_f32(col_j, col_j, d).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4 || norm < 1e-6,
                "Column {j} norm = {norm}, expected ~1.0 or ~0"
            );
        }
        // Check orthogonality between columns 0 and 1.
        if r >= 2 {
            let col_0 = &basis[0..d];
            let col_1 = &basis[d..2 * d];
            let dot = simd_dot_f32(col_0, col_1, d);
            assert!(dot.abs() < 1e-4, "Columns not orthogonal: dot = {dot}");
        }
    }

    /// `spectral_weighting_prioritizes_high_sigma` — verify high-σ axes get more mass.
    #[test]
    fn spectral_weighting_prioritizes_high_sigma() {
        let d = 4;
        let r = 2;
        // Basis: standard basis vectors e1, e2 (already orthonormal).
        // Column-major: col 0 = [1,0,0,0], col 1 = [0,1,0,0]
        let basis = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // Sigma: [10.0, 1.0] — first axis has 10x the singular value.
        let sigma = vec![10.0, 1.0];
        // Gradient: equal in both directions [1, 1, 0, 0].
        let gradient = vec![1.0, 1.0, 0.0, 0.0];
        let mut scratch = ManceScratch::with_capacity(d, r, r);

        let alignment = tangent_erasure_direction_into(
            &gradient,
            &basis,
            &sigma,
            1.0,
            d,
            r,
            &mut scratch.projection_coords,
            &mut scratch.tangent_direction,
        );
        let direction = &scratch.tangent_direction;
        let _ = alignment;

        // The spectrally weighted direction should be dominated by the high-σ axis.
        // d = B · diag(σ) · Bᵀu = [10*0.707, 1*0.707, 0, 0] → normalized → mostly e1.
        let e1_component = direction[0].abs();
        let e2_component = direction[1].abs();
        assert!(
            e1_component > e2_component,
            "High-σ axis should dominate: e1={e1_component} vs e2={e2_component}"
        );
    }

    /// `trust_region_bounds_displacement` — verify `‖x̃ - x‖ ≤ ε·r_i`.
    #[test]
    fn trust_region_bounds_displacement() {
        let d = 8;
        let n = 50;
        let k = 8;
        let r = 4;
        let config = ManceConfig {
            epsilon: 0.1,
            lambda_max: 64.0,
            alpha: 1.0,
            k,
            r,
        };
        let pool = make_pool(n, d, 123);
        let x = vec![0.5; d];
        let gradient = vec![1.0; d]; // Non-zero gradient
        let mut scratch = ManceScratch::with_capacity(d, k, r);
        let mut out = vec![0.0; d];

        let info =
            manifold_erasure_step_into(&x, &gradient, &pool, n, &config, &mut scratch, &mut out)
                .unwrap();

        let _displacement = simd_dot_f32(&out, &out, d).sqrt();
        let _x_norm = simd_dot_f32(&x, &x, d).sqrt();
        // Displacement = ‖out - x‖
        let mut diff = vec![0.0; d];
        for i in 0..d {
            diff[i] = out[i] - x[i];
        }
        let actual_disp = simd_dot_f32(&diff, &diff, d).sqrt();

        let bound = config.epsilon * info.local_radius;
        assert!(
            actual_disp <= bound + 1e-5,
            "Displacement {} exceeds trust region bound {} (r_i={})",
            actual_disp,
            bound,
            info.local_radius
        );
    }

    /// `zero_gradient_no_harm` — gradient=0 → out=x bit-identically.
    #[test]
    fn zero_gradient_no_harm() {
        let d = 8;
        let n = 20;
        let k = 8;
        let r = 4;
        let config = ManceConfig {
            k,
            r,
            ..Default::default()
        };
        let pool = make_pool(n, d, 999);
        let x = vec![0.3, -0.5, 0.7, 0.1, -0.2, 0.8, -0.4, 0.6];
        let gradient = vec![0.0; d];
        let mut scratch = ManceScratch::with_capacity(d, k, r);
        let mut out = vec![0.0; d];

        let info =
            manifold_erasure_step_into(&x, &gradient, &pool, n, &config, &mut scratch, &mut out)
                .unwrap();

        assert_eq!(out, x, "Zero gradient should produce bit-identical output");
        assert_eq!(info.lambda, 0.0);
        assert_eq!(info.displacement, 0.0);
    }

    /// `orthogonal_direction_no_harm` — gradient ⊥ tangent basis → out=x bit-identically.
    #[test]
    fn orthogonal_direction_no_harm() {
        let d = 4;
        let n = 20;
        let k = 8;
        let r = 2;
        let config = ManceConfig {
            k,
            r,
            ..Default::default()
        };

        // Create a pool where the tangent basis is clearly in the e1-e2 plane.
        // Points vary only in dimensions 0 and 1.
        let mut pool = vec![0.0; n * d];
        for i in 0..n {
            pool[i * d] = (i as f32) * 0.1 - 1.0; // dim 0 varies
            pool[i * d + 1] = (i as f32) * 0.05 - 0.5; // dim 1 varies
            // dims 2, 3 stay 0
        }

        let x = vec![0.5, 0.5, 0.5, 0.5];
        // Gradient in e3 direction (orthogonal to the e1-e2 tangent plane).
        let gradient = vec![0.0, 0.0, 1.0, 0.0];
        let mut scratch = ManceScratch::with_capacity(d, k, r);
        let mut out = vec![0.0; d];

        let info =
            manifold_erasure_step_into(&x, &gradient, &pool, n, &config, &mut scratch, &mut out)
                .unwrap();

        assert_eq!(
            out, x,
            "Orthogonal gradient should produce bit-identical output"
        );
        assert_eq!(info.lambda, 0.0);
    }

    /// `erasure_reduces_target_alignment` — after step, `|<x̃, u>| < |<x, u>|`.
    #[test]
    fn erasure_reduces_target_alignment() {
        let d = 8;
        let n = 50;
        let k = 8;
        let r = 4;
        let config = ManceConfig {
            k,
            r,
            ..Default::default()
        };
        let pool = make_pool(n, d, 777);
        let x = vec![0.5; d];
        let gradient = vec![1.0, 0.5, -0.3, 0.8, -0.1, 0.4, 0.2, -0.6];
        let mut scratch = ManceScratch::with_capacity(d, k, r);
        let mut out = vec![0.0; d];

        manifold_erasure_step_into(&x, &gradient, &pool, n, &config, &mut scratch, &mut out)
            .unwrap();

        // Normalize gradient for comparison.
        let grad_norm = simd_dot_f32(&gradient, &gradient, d).sqrt();
        let mut u = vec![0.0; d];
        for i in 0..d {
            u[i] = gradient[i] / grad_norm;
        }

        let x_align = simd_dot_f32(&x, &u, d).abs();
        let out_align = simd_dot_f32(&out, &u, d).abs();

        assert!(
            out_align < x_align,
            "Erasure should reduce target alignment: before={x_align}, after={out_align}"
        );
    }

    // T2.5: Preprocessing tests

    /// `leace_removes_class_mean_difference` — after LEACE, `<x̃, d_mean> ≈ 0`.
    #[test]
    fn leace_removes_class_mean_difference() {
        let d = 8;
        let mean_pos = vec![0.5; d];
        let mean_neg = vec![-0.5; d];
        let x = vec![1.0, 0.0, 0.5, -0.3, 0.8, -0.1, 0.4, 0.2];
        let mut scratch = ManceScratch::with_capacity(d, 8, 4);
        let mut out = vec![0.0; d];

        leace_first_moment_into(
            &x,
            &mean_pos,
            &mean_neg,
            &mut scratch.tangent_direction,
            &mut out,
        )
        .unwrap();

        // d_mean = [1.0; d]
        let d_mean: Vec<f32> = (0..d).map(|i| mean_pos[i] - mean_neg[i]).collect();
        let proj = simd_dot_f32(&out, &d_mean, d);
        assert!(
            proj.abs() < 1e-5,
            "LEACE should zero the class-mean direction: proj = {proj}"
        );
    }

    /// `covmatch_removes_covariance_asymmetry` — after CovMatch, class-conditional
    /// variance asymmetry reduced.
    #[test]
    fn covmatch_removes_covariance_asymmetry() {
        let d = 4;
        // Two eigenvectors of ΔΣ: e1 and e2.
        let eigvecs = vec![
            1.0, 0.0, 0.0, 0.0, // v1 = e1
            0.0, 1.0, 0.0, 0.0, // v2 = e2
        ];
        let x = vec![1.0, 0.5, 0.3, 0.7];
        let mut scratch = ManceScratch::with_capacity(d, 8, 4);
        let mut out = vec![0.0; d];

        covmatch_second_moment_into(&x, &eigvecs, &mut scratch.mean_neighbor, &mut out).unwrap();

        // After projecting out e1 and e2, the first two components should be 0.
        assert!(
            out[0].abs() < 1e-5,
            "e1 component should be zeroed: {}",
            out[0]
        );
        assert!(
            out[1].abs() < 1e-5,
            "e2 component should be zeroed: {}",
            out[1]
        );
        // Components 2 and 3 should be preserved.
        assert!(
            (out[2] - 0.3).abs() < 1e-5,
            "e3 should be preserved: {}",
            out[2]
        );
        assert!(
            (out[3] - 0.7).abs() < 1e-5,
            "e4 should be preserved: {}",
            out[3]
        );
    }

    /// `preprocessing_preserves_orthogonal_directions` — directions ⊥ the erased
    /// directions are unchanged.
    #[test]
    fn preprocessing_preserves_orthogonal_directions() {
        let d = 4;
        let mean_pos = vec![1.0, 0.0, 0.0, 0.0];
        let mean_neg = vec![-1.0, 0.0, 0.0, 0.0];
        // d_mean = [2, 0, 0, 0] — only erases e1 direction.
        let x = vec![0.5, 0.7, -0.3, 0.4];
        let mut scratch = ManceScratch::with_capacity(d, 8, 4);
        let mut out = vec![0.0; d];

        leace_first_moment_into(
            &x,
            &mean_pos,
            &mean_neg,
            &mut scratch.tangent_direction,
            &mut out,
        )
        .unwrap();

        // e2, e3, e4 should be preserved.
        assert!((out[1] - 0.7).abs() < 1e-5, "e2 preserved: {}", out[1]);
        assert!((out[2] - (-0.3)).abs() < 1e-5, "e3 preserved: {}", out[2]);
        assert!((out[3] - 0.4).abs() < 1e-5, "e4 preserved: {}", out[3]);
    }

    // Issue 132: Tangent cache tests

    /// Cached step produces bit-identical results to uncached step.
    #[test]
    fn cached_step_matches_uncached() {
        let d = 8;
        let n = 50;
        let config = ManceConfig::default();
        let pool = make_pool(n, d, 42);
        let x = vec![0.5; d];
        let gradient = vec![1.0, 0.5, -0.3, 0.8, -0.1, 0.4, 0.2, -0.6];

        // Uncached step.
        let mut scratch_u = ManceScratch::with_capacity(d, config.k, config.r);
        let mut out_uncached = vec![0.0; d];
        let info_u = manifold_erasure_step_into(
            &x,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch_u,
            &mut out_uncached,
        )
        .unwrap();

        // Cached step (first call — cache miss, same as uncached).
        let mut scratch_c = ManceScratch::with_capacity(d, config.k, config.r);
        let mut cache = ManceTangentCache::with_capacity(d, config.k, config.r);
        let mut out_cached = vec![0.0; d];
        let info_c = manifold_erasure_step_cached_into(
            &x,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch_c,
            &mut cache,
            &mut out_cached,
        )
        .unwrap();

        // Results must be bit-identical.
        assert_eq!(
            out_uncached, out_cached,
            "Cached step output must match uncached"
        );
        assert_eq!(info_u.lambda, info_c.lambda, "lambda must match");
        assert_eq!(
            info_u.displacement, info_c.displacement,
            "displacement must match"
        );
        assert_eq!(
            info_u.local_radius, info_c.local_radius,
            "local_radius must match"
        );
        assert_eq!(info_u.alignment, info_c.alignment, "alignment must match");
    }

    /// Cached step with cache hit (same x) produces identical results.
    #[test]
    fn cached_step_cache_hit_same_result() {
        let d = 8;
        let n = 50;
        let config = ManceConfig::default();
        let pool = make_pool(n, d, 42);
        let x = vec![0.5; d];
        let gradient = vec![1.0; d];

        let mut scratch = ManceScratch::with_capacity(d, config.k, config.r);
        let mut cache = ManceTangentCache::with_capacity(d, config.k, config.r);
        let mut out1 = vec![0.0; d];
        let mut out2 = vec![0.0; d];

        // First call — cache miss, populates cache.
        let info1 = manifold_erasure_step_cached_into(
            &x,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch,
            &mut cache,
            &mut out1,
        )
        .unwrap();

        // Second call with same x — cache hit, should produce identical result.
        let info2 = manifold_erasure_step_cached_into(
            &x,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch,
            &mut cache,
            &mut out2,
        )
        .unwrap();

        assert_eq!(out1, out2, "Cache hit must produce identical output");
        assert_eq!(info1.lambda, info2.lambda);
        assert_eq!(info1.alignment, info2.alignment);
    }

    /// Cached loop produces bit-identical results to uncached loop.
    #[test]
    fn cached_loop_matches_uncached() {
        let d = 8;
        let n = 50;
        let config = ManceConfig::default();
        let pool = make_pool(n, d, 42);
        let x = vec![0.5; d];
        let gradient = vec![1.0, 0.5, -0.3, 0.8, -0.1, 0.4, 0.2, -0.6];

        let grad_ref = &gradient;
        let gf = move |_state: &[f32], buf: &mut [f32]| {
            buf.copy_from_slice(grad_ref);
        };

        // Uncached loop.
        let mut scratch_u = ManceScratch::with_capacity(d, config.k, config.r);
        let mut out_uncached = vec![0.0; d];
        let infos_u = manifold_erasure_loop_into(
            &x,
            gf,
            &pool,
            n,
            &config,
            10,
            &mut scratch_u,
            &mut out_uncached,
        )
        .unwrap();

        // Cached loop.
        let grad_ref2 = &gradient;
        let gf2 = move |_state: &[f32], buf: &mut [f32]| {
            buf.copy_from_slice(grad_ref2);
        };
        let mut scratch_c = ManceScratch::with_capacity(d, config.k, config.r);
        let mut cache = ManceTangentCache::with_capacity(d, config.k, config.r);
        let mut out_cached = vec![0.0; d];
        let infos_c = manifold_erasure_loop_cached_into(
            &x,
            gf2,
            &pool,
            n,
            &config,
            10,
            &mut scratch_c,
            &mut cache,
            &mut out_cached,
        )
        .unwrap();

        // Results must be bit-identical.
        assert_eq!(
            out_uncached, out_cached,
            "Cached loop output must match uncached"
        );
        assert_eq!(infos_u.len(), infos_c.len(), "Round count must match");
        for (i, (iu, ic)) in infos_u.iter().zip(infos_c.iter()).enumerate() {
            assert_eq!(iu.lambda, ic.lambda, "Round {i} lambda mismatch");
            assert_eq!(
                iu.displacement, ic.displacement,
                "Round {i} displacement mismatch"
            );
            assert_eq!(
                iu.local_radius, ic.local_radius,
                "Round {i} local_radius mismatch"
            );
            assert_eq!(iu.alignment, ic.alignment, "Round {i} alignment mismatch");
        }
    }

    /// Cache invalidation: when x moves far enough that neighbors change,
    /// the cache is invalidated and recomputed.
    #[test]
    fn cache_invalidation_when_neighbors_change() {
        let d = 4;
        let n = 20;
        let config = ManceConfig {
            k: 4,
            r: 2,
            ..Default::default()
        };
        // Pool with points spread across [-1, 1] in each dim.
        let pool = make_pool(n, d, 999);

        let mut scratch = ManceScratch::with_capacity(d, config.k, config.r);
        let mut cache = ManceTangentCache::with_capacity(d, config.k, config.r);
        let gradient = vec![1.0; d];
        let mut out = vec![0.0; d];

        // Step 1: x near origin.
        let x1 = vec![0.0; d];
        let info1 = manifold_erasure_step_cached_into(
            &x1,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch,
            &mut cache,
            &mut out,
        )
        .unwrap();
        assert!(cache.valid, "Cache should be valid after first step");

        // Step 2: same x — cache hit, same result.
        let mut out2 = vec![0.0; d];
        let info2 = manifold_erasure_step_cached_into(
            &x1,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch,
            &mut cache,
            &mut out2,
        )
        .unwrap();
        assert_eq!(out, out2, "Same x should produce same output");
        assert_eq!(info1.lambda, info2.lambda);

        // Step 3: x moved far — neighbors likely change, cache should still work.
        // (Even if cache is invalidated, the result must be correct.)
        let x3 = vec![0.9, -0.9, 0.9, -0.9];
        let mut out3 = vec![0.0; d];
        let _info3 = manifold_erasure_step_cached_into(
            &x3,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch,
            &mut cache,
            &mut out3,
        )
        .unwrap();

        // Verify: uncached step with x3 should match cached step.
        let mut scratch_u = ManceScratch::with_capacity(d, config.k, config.r);
        let mut out_uncached = vec![0.0; d];
        manifold_erasure_step_into(
            &x3,
            &gradient,
            &pool,
            n,
            &config,
            &mut scratch_u,
            &mut out_uncached,
        )
        .unwrap();
        assert_eq!(
            out3, out_uncached,
            "Cached step after invalidation must match uncached"
        );
    }

    /// Cache invalidate() resets to invalid state.
    #[test]
    fn cache_invalidate_resets() {
        let d = 8;
        let k = 8;
        let r = 8;
        let mut cache = ManceTangentCache::with_capacity(d, k, r);
        assert!(!cache.valid, "New cache should be invalid");

        // Populate with sorted indices (k-NN always returns sorted indices).
        cache.update(&[0, 1, 2, 3, 4, 5, 6, 7], &vec![1.0; d * r], &vec![1.0; r]);
        assert!(cache.valid, "Cache should be valid after update");
        assert!(cache.is_valid_for(&[0, 1, 2, 3, 4, 5, 6, 7]));
        // Different set — should be invalid.
        assert!(!cache.is_valid_for(&[1, 2, 3, 4, 5, 6, 7, 8]));

        // Invalidate.
        cache.invalidate();
        assert!(!cache.valid, "Cache should be invalid after invalidate()");
        assert!(
            !cache.is_valid_for(&[0, 1, 2, 3, 4, 5, 6, 7]),
            "Invalidated cache should not be valid"
        );
    }
}
