//! Spectral Rewiring — Weight Delta Purification via Base SVD Projection.
//!
//! Plan 423, Research 406. Distilled from Zhang et al., *Spectral Rewiring for
//! Exploration, Purification, and Model Merging* (arXiv:2607.03065, Tsinghua
//! AIR / ByteDance Seed, Jul 2026).
//!
//! Given base weights W₀ and a weight delta ΔW, project ΔW onto W₀'s SVD
//! spectral subspace:
//! ```text
//! W₀ = U Σ Vᵀ           (thin SVD of the base)
//! M  = U_rᵀ ΔW V_r       (rewiring matrix — r×r, cross-skill interactions)
//! ΔW* = U_r M V_rᵀ        (purified delta — on-manifold component)
//! ΔW⊥ = ΔW − ΔW*          (off-manifold residual)
//! ```
//!
//! The rewiring matrix M is compact (r×r). Its off-diagonal elements `M[i][j]`
//! (i≠j) represent cross-skill "rewiring" — many-to-one logical synthesis.
//! Its diagonal elements `M[i][i]` represent in-skill strength modulation.
//!
//! # Modelless
//!
//! SVD + matrix multiply. No training, no gradient descent. The paper's
//! headline use case (extract a reasoning core from a trained RL delta W_RL)
//! routes to riir-train; this module ships the modelless kernel that operates
//! on any weight delta — freeze/thaw deltas, LoRA overlays, consolidation
//! deltas.
//!
//! # Sibling relationship
//!
//! This is the **on-principal complement** of [`crate::off_principal`] (Plan 264).
//! `off_principal` projects a query *away* from the base SVD subspace
//! (off-principal). `spectral_rewire` projects a delta *onto* the base SVD
//! subspace (on-principal). Both reuse the same SVD substrate; together they
//! decompose any delta into on-manifold + off-manifold components (Issue 123,
//! Fusion B).
//!
//! # Feature gate
//!
//! STAYS OPT-IN — Plan 423 Phase 3 GOAT (Bench 423, 2026-07-10): all mechanism
//! gates PASS (G1a/G2/G3/G4/G5/G6), but the spectral concentration assumption
//! (G1b) does NOT hold at NPC scale (≤64×64), verified via riir-train Issue 374
//! (trained LoRA deltas produce on_manifold_fraction in [0.27, 0.58], far
//! below the SAR threshold > 0.8). The paper proves spectral concentration for
//! 1.5B–32B LLM weights; our 64×64 / 128×128 matrices are unvalidated at that
//! scale. See `.benchmarks/423_*`.

use katgpt_core::simd::dot_f32_ordered;
use katgpt_core::{SvdResultScratch, SvdScratch, thin_svd_into};

// Issue 124 (resolved 2026-07-10): the one-sided Jacobi SVD substrate in
// `katgpt_core::thin_svd_into` previously used fixed `[f32; 64]` / `[usize; 64]`
// stack arrays, capping `n_cols ≤ 64`. The argsort buffers have been moved to
// `SvdScratch` (heap-allocated, sized once), so the cap is LIFTED. Arbitrary
// `d_in` (matrix column count) is now supported — only bounded by available
// memory for the scratch buffers. The former `SVD_MAX_COLS` constant and the
// `d_in <= 64` guards have been removed.

// ---------------------------------------------------------------------------
// SpectralRewireScratch — pre-allocated reusable buffers
// ---------------------------------------------------------------------------

/// Pre-allocated scratch for [`spectral_rewire_into`]. Reuse across calls to
/// eliminate per-call allocation.
///
/// Sized via [`SpectralRewireScratch::with_capacity`] for the largest expected
/// `(d_out, d_in, rank)` triple. All internal buffers grow on demand if a
/// larger matrix is presented, but the zero-alloc contract holds only when
/// dimensions do not exceed capacity.
pub struct SpectralRewireScratch {
    /// SVD result (SOA scratch) for W₀ decomposition — stores U (column-major)
    /// and V (column-major) plus singular values.
    svd_result: SvdResultScratch,
    /// SVD working buffers (working copy of W₀, V accumulator, column norms).
    svd_work: SvdScratch,
    /// Temp buffer for `A = U_rᵀ · ΔW` (rank × d_in), row-major.
    a_buf: Vec<f32>,
    /// Rewiring matrix `M = A · V_r` (rank × rank), row-major.
    m_buf: Vec<f32>,
    /// Temp buffer for `B = U_r · M` (d_out × rank), row-major.
    b_buf: Vec<f32>,
    /// Purified delta `ΔW* = B · V_rᵀ` (d_out × d_in), row-major.
    delta_star_buf: Vec<f32>,
    /// Off-manifold residual `ΔW⊥ = ΔW − ΔW*` (d_out × d_in), row-major.
    residual_buf: Vec<f32>,
}

impl SpectralRewireScratch {
    /// Allocate scratch sized for factoring a `d_out × d_in` base matrix and
    /// projecting at up to `max_rank` spectral components.
    pub fn with_capacity(d_out: usize, d_in: usize, max_rank: usize) -> Self {
        let total = d_out * d_in;
        let rank = max_rank.min(d_out.min(d_in));
        Self {
            svd_result: SvdResultScratch::with_capacity(d_out, d_in),
            svd_work: SvdScratch::with_capacity(d_in, d_out),
            a_buf: vec![0.0; rank * d_in],
            m_buf: vec![0.0; rank * rank],
            b_buf: vec![0.0; d_out * rank],
            delta_star_buf: vec![0.0; total],
            residual_buf: vec![0.0; total],
        }
    }

    /// Grow all buffers if the requested `(d_out, d_in, rank)` exceeds current
    /// capacity. No-op (no allocation) when dimensions fit.
    fn ensure_capacity(&mut self, d_out: usize, d_in: usize, rank: usize) {
        let total = d_out * d_in;
        let r = rank.min(d_out.min(d_in));
        if r * d_in > self.a_buf.len() {
            self.a_buf.resize(r * d_in, 0.0);
        }
        if r * r > self.m_buf.len() {
            self.m_buf.resize(r * r, 0.0);
        }
        if d_out * r > self.b_buf.len() {
            self.b_buf.resize(d_out * r, 0.0);
        }
        if total > self.delta_star_buf.len() {
            self.delta_star_buf.resize(total, 0.0);
        }
        if total > self.residual_buf.len() {
            self.residual_buf.resize(total, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// SpectralRewireOutput — borrows into scratch
// ---------------------------------------------------------------------------

/// Output view of [`spectral_rewire_into`]. All slices borrow from the caller's
/// [`SpectralRewireScratch`] and are valid until the scratch is next mutated.
pub struct SpectralRewireOutput<'a> {
    /// Purified delta `ΔW* = U_r M V_rᵀ` (on-manifold component). Row-major
    /// `d_out × d_in`.
    pub delta_star: &'a [f32],
    /// Compact rewiring matrix `M = U_rᵀ ΔW V_r` (rank × rank), row-major.
    /// Off-diagonal `M[i][j]` (i≠j) = cross-skill rewiring.
    pub rewiring_matrix: &'a [f32],
    /// Off-manifold residual `ΔW⊥ = ΔW − ΔW*`. Row-major `d_out × d_in`.
    pub residual: &'a [f32],
    /// On-manifold energy fraction `‖ΔW*‖_F / ‖ΔW‖_F` ∈ [0, 1]. High values
    /// indicate spectral concentration (the paper's core assumption).
    pub on_manifold_fraction: f32,
}

// ---------------------------------------------------------------------------
// spectral_rewire_into — zero-alloc hot path
// ---------------------------------------------------------------------------

/// Project a weight delta `ΔW` onto the base weight `W₀`'s SVD spectral
/// subspace, extracting the on-manifold component `ΔW*` and the compact
/// rewiring matrix `M`.
///
/// # Arguments
///
/// - `w0`: base weights, row-major flat (`d_out × d_in`). The SVD is computed
///   over this matrix. For repeated projections against the same base, consider
///   pre-computing the SVD once (future `SpectralRewireIndex` — see Plan 423
///   open question 2).
/// - `delta`: weight delta `ΔW = W_new − W₀`, same layout as `w0`.
/// - `d_out`, `d_in`: matrix dimensions.
/// - `rank`: top-k spectral rank `r` (≤ `min(d_out, d_in)`). Controls the
///   subspace dimension — higher rank captures more of ΔW but M is larger.
/// - `scratch`: caller-owned reusable buffers.
///
/// # Returns
///
/// A [`SpectralRewireOutput`] borrowing `scratch`. The projection is exact up
/// to f32 round-off when ΔW lies entirely in the base SVD subspace.
///
/// # Zero-alloc contract
///
/// No allocation after warmup. Buffers grow only if dimensions exceed capacity
/// (see [`SpectralRewireScratch::with_capacity`]).
///
/// # Panics
///
/// Panics if `w0.len()` or `delta.len()` ≠ `d_out * d_in`, or if `rank` is 0.
pub fn spectral_rewire_into<'a>(
    w0: &[f32],
    delta: &[f32],
    d_out: usize,
    d_in: usize,
    rank: usize,
    scratch: &'a mut SpectralRewireScratch,
) -> SpectralRewireOutput<'a> {
    let total = d_out * d_in;
    assert_eq!(
        w0.len(),
        total,
        "spectral_rewire_into: w0.len() {} != d_out*d_in = {d_out}*{d_in} = {total}",
        w0.len()
    );
    assert_eq!(
        delta.len(),
        total,
        "spectral_rewire_into: delta.len() {} != d_out*d_in = {total}",
        delta.len()
    );
    assert!(
        rank >= 1,
        "spectral_rewire_into: rank must be >= 1, got {rank}"
    );
    let r = rank.min(d_out.min(d_in));

    scratch.ensure_capacity(d_out, d_in, r);

    // ── Step 1: SVD of W₀ → U_r, V_r (column-major in svd_result) ──────────
    {
        let svd_result = &mut scratch.svd_result;
        let svd_work = &mut scratch.svd_work;
        thin_svd_into(w0, d_out, d_in, svd_result, svd_work);
    }
    let svd = &scratch.svd_result;

    // Steps 2–6: shared projection core (SVD-backed accessor closures).
    project_core(
        delta,
        d_out,
        d_in,
        r,
        total,
        &mut scratch.a_buf,
        &mut scratch.m_buf,
        &mut scratch.b_buf,
        &mut scratch.delta_star_buf,
        &mut scratch.residual_buf,
        |j| svd.left_singular_vector(j),
        |j| svd.right_singular_vector(j),
    )
}

/// Shared projection core (steps 2–6 of the SAR algorithm). Takes accessor
/// closures for the top-r left (`u_col`) and right (`v_col`) singular vectors
/// of W₀, so the same code serves both the on-the-fly SVD path
/// ([`spectral_rewire_into`]) and the cached-index path
/// ([`spectral_rewire_with_index_into`]).
///
/// Writes `delta_star_buf`, `m_buf` (rewiring matrix), `residual_buf`, and
/// returns the borrowed [`SpectralRewireOutput`].
///
/// # Safety contract (caller)
///
/// `u_col(i)` must return a slice of length `d_out`; `v_col(j)` a slice of
/// length `d_in`; both for `i, j ∈ 0..r`. Scratch slices must be at least
/// `r*d_in`, `r*r`, `d_out*r`, `total`, `total` long respectively.
#[allow(clippy::too_many_arguments)]
fn project_core<'a, UF, VF>(
    delta: &[f32],
    d_out: usize,
    d_in: usize,
    r: usize,
    total: usize,
    a_buf: &'a mut [f32],
    m_buf: &'a mut [f32],
    b_buf: &'a mut [f32],
    delta_star_buf: &'a mut [f32],
    residual_buf: &'a mut [f32],
    u_col: UF,
    v_col: VF,
) -> SpectralRewireOutput<'a>
where
    UF: Fn(usize) -> &'a [f32],
    VF: Fn(usize) -> &'a [f32],
{
    // ── Step 2: A = U_rᵀ · ΔW  (r × d_in, row-major in a_buf) ──────────────
    //
    // A[i][j] = Σ_k U[k][i] · ΔW[k][j]
    // For each i: accumulate rank-1 updates over ΔW rows weighted by U column i.
    let a_len = r * d_in;
    let a_buf = &mut a_buf[..a_len];
    for v in a_buf.iter_mut() {
        *v = 0.0;
    }
    for i in 0..r {
        let u_col_i = u_col(i); // length d_out, contiguous
        let a_row_offset = i * d_in;
        for k in 0..d_out {
            let alpha = u_col_i[k]; // U[k][i]
            let delta_row = &delta[k * d_in..(k + 1) * d_in]; // contiguous
            let a_row = &mut a_buf[a_row_offset..a_row_offset + d_in];
            // a_row[j] += alpha * delta_row[j]  — contiguous axpy
            let mut j = 0;
            while j + 4 <= d_in {
                a_row[j] = alpha.mul_add(delta_row[j], a_row[j]);
                a_row[j + 1] = alpha.mul_add(delta_row[j + 1], a_row[j + 1]);
                a_row[j + 2] = alpha.mul_add(delta_row[j + 2], a_row[j + 2]);
                a_row[j + 3] = alpha.mul_add(delta_row[j + 3], a_row[j + 3]);
                j += 4;
            }
            while j < d_in {
                a_row[j] = alpha.mul_add(delta_row[j], a_row[j]);
                j += 1;
            }
        }
    }

    // ── Step 3: M = A · V_r  (r × r, row-major in m_buf) ───────────────────
    //
    // M[i][j] = Σ_k A[i][k] · V[k][j] = <A_row_i, V_col_j>
    let m_len = r * r;
    let m_buf = &mut m_buf[..m_len];
    for i in 0..r {
        let a_row_i = &a_buf[i * d_in..(i + 1) * d_in]; // contiguous, length d_in
        for j in 0..r {
            let v_col_j = v_col(j); // length d_in, contiguous
            // Issue 940 SP-3: ordered fold, not the platform lane kernel — M
            // feeds B → ΔW*, the purified weight delta that lands in
            // BLAKE3-committed freeze/thaw overlays; the platform reduction
            // would make the committed overlay node-dependent. The other
            // projection steps (2/4/5) already fold per-index through
            // single-rounding `mul_add`, so this was the one platform hole.
            m_buf[i * r + j] = dot_f32_ordered(a_row_i, v_col_j);
        }
    }

    // ── Step 4: B = U_r · M  (d_out × r, row-major in b_buf) ───────────────
    //
    // B[i][j] = Σ_k U[i][k] · M[k][j]
    // For each i: accumulate rank-1 updates over M rows weighted by U entries.
    let b_len = d_out * r;
    let b_buf = &mut b_buf[..b_len];
    for v in b_buf.iter_mut() {
        *v = 0.0;
    }
    for k in 0..r {
        let u_col_k = u_col(k); // length d_out, contiguous
        let m_row_k = &m_buf[k * r..(k + 1) * r]; // contiguous, length r
        for i in 0..d_out {
            let beta = u_col_k[i]; // U[i][k]
            let b_row = &mut b_buf[i * r..(i + 1) * r]; // contiguous, length r
            let mut j = 0;
            while j + 4 <= r {
                b_row[j] = beta.mul_add(m_row_k[j], b_row[j]);
                b_row[j + 1] = beta.mul_add(m_row_k[j + 1], b_row[j + 1]);
                b_row[j + 2] = beta.mul_add(m_row_k[j + 2], b_row[j + 2]);
                b_row[j + 3] = beta.mul_add(m_row_k[j + 3], b_row[j + 3]);
                j += 4;
            }
            while j < r {
                b_row[j] = beta.mul_add(m_row_k[j], b_row[j]);
                j += 1;
            }
        }
    }

    // ── Step 5: ΔW* = B · V_rᵀ  (d_out × d_in, row-major in delta_star_buf) ─
    //
    // ΔW*[i][j] = Σ_k B[i][k] · V[j][k]
    // Rank-1 sum: for each k, add B[:,k] ⊗ V[:,k].
    let delta_star = &mut delta_star_buf[..total];
    for v in delta_star.iter_mut() {
        *v = 0.0;
    }
    for k in 0..r {
        let v_col_k = v_col(k); // length d_in, contiguous
        for i in 0..d_out {
            let gamma = b_buf[i * r + k]; // B[i][k]
            let ds_row = &mut delta_star[i * d_in..(i + 1) * d_in]; // contiguous
            let mut j = 0;
            while j + 4 <= d_in {
                ds_row[j] = gamma.mul_add(v_col_k[j], ds_row[j]);
                ds_row[j + 1] = gamma.mul_add(v_col_k[j + 1], ds_row[j + 1]);
                ds_row[j + 2] = gamma.mul_add(v_col_k[j + 2], ds_row[j + 2]);
                ds_row[j + 3] = gamma.mul_add(v_col_k[j + 3], ds_row[j + 3]);
                j += 4;
            }
            while j < d_in {
                ds_row[j] = gamma.mul_add(v_col_k[j], ds_row[j]);
                j += 1;
            }
        }
    }

    // ── Step 6: residual + on-manifold fraction ───────────────────────────
    let residual = &mut residual_buf[..total];
    let mut norm_delta_sq = 0.0_f32;
    let mut norm_star_sq = 0.0_f32;
    for idx in 0..total {
        let d = delta[idx];
        let s = delta_star[idx];
        residual[idx] = d - s;
        norm_delta_sq = d.mul_add(d, norm_delta_sq);
        norm_star_sq = s.mul_add(s, norm_star_sq);
    }
    let on_manifold_fraction = if norm_delta_sq > 1e-30 {
        (norm_star_sq / norm_delta_sq).sqrt()
    } else {
        0.0
    };

    SpectralRewireOutput {
        delta_star,
        rewiring_matrix: m_buf,
        residual,
        on_manifold_fraction,
    }
}

// ---------------------------------------------------------------------------
// spectral_rewire — allocating convenience wrapper
// ---------------------------------------------------------------------------

/// Owned result of [`spectral_rewire`] (the allocating convenience wrapper).
///
/// For hot paths, use [`spectral_rewire_into`] with a reused
/// [`SpectralRewireScratch`] to avoid the three `Vec` allocations here.
pub struct SpectralRewireResult {
    /// Purified delta `ΔW* = U_r M V_rᵀ` (on-manifold component).
    pub delta_star: Vec<f32>,
    /// Compact rewiring matrix `M = U_rᵀ ΔW V_r` (rank × rank).
    pub rewiring_matrix: Vec<f32>,
    /// Off-manifold residual `ΔW⊥ = ΔW − ΔW*`.
    pub residual: Vec<f32>,
    /// On-manifold energy fraction `‖ΔW*‖_F / ‖ΔW‖_F` ∈ [0, 1].
    pub on_manifold_fraction: f32,
}

/// Allocating convenience wrapper around [`spectral_rewire_into`]. Allocates a
/// fresh [`SpectralRewireScratch`] internally, so this is for tests, examples,
/// and cold-path callers only.
pub fn spectral_rewire(
    w0: &[f32],
    delta: &[f32],
    d_out: usize,
    d_in: usize,
    rank: usize,
) -> SpectralRewireResult {
    let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, rank);
    let out = spectral_rewire_into(w0, delta, d_out, d_in, rank, &mut scratch);
    SpectralRewireResult {
        delta_star: out.delta_star.to_vec(),
        rewiring_matrix: out.rewiring_matrix.to_vec(),
        residual: out.residual.to_vec(),
        on_manifold_fraction: out.on_manifold_fraction,
    }
}

// ---------------------------------------------------------------------------
// SpectralRewireIndex — cached base SVD to eliminate SVD from the hot loop
// ---------------------------------------------------------------------------

/// Pre-computed top-rank SVD of a base matrix `W₀`, for repeated projections
/// against the same base (Plan 423 open question #2).
///
/// [`spectral_rewire_into`] factors `W₀` on EVERY call — the one-sided Jacobi
/// SVD dominates latency (~ms for a 64×64 base). When many deltas are projected
/// against the same base (e.g. all LoRA overlays for one layer, all freeze/thaw
/// snapshots of one shard), build an index ONCE and call
/// [`spectral_rewire_with_index_into`] per delta. The per-delta cost is then
/// just the four matmuls (steps 2–5) — typically microseconds.
///
/// Stores the top-r left singular vectors (`u_r`, column-major, `d_out × r`)
/// and right singular vectors (`v_r`, column-major, `d_in × r`) plus singular
/// values. Building the index still pays the SVD cost once; it is amortized
/// over all subsequent projections.
///
/// # Example
///
/// ```
/// use katgpt_spectral::spectral_rewire::{
///     SpectralRewireIndex, SpectralRewireScratch, spectral_rewire_with_index_into,
/// };
///
/// let (d_out, d_in, r) = (8, 8, 4);
/// let w0: Vec<f32> = (0..d_out * d_in).map(|i| (i as f32) * 0.01).collect();
/// let index = SpectralRewireIndex::new(&w0, d_out, d_in, r);
///
/// let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, r);
/// let delta: Vec<f32> = vec![0.05; d_out * d_in];
/// let out = spectral_rewire_with_index_into(&index, &delta, &mut scratch);
/// assert_eq!(out.delta_star.len(), d_out * d_in);
/// assert!(out.on_manifold_fraction >= 0.0);
/// ```
pub struct SpectralRewireIndex {
    /// Top-r left singular vectors (columns of U), column-major flat:
    /// column `i` lives at `[i * d_out .. (i+1) * d_out]`. Length `r * d_out`.
    u_r: Vec<f32>,
    /// Top-r right singular vectors (columns of V), column-major flat:
    /// column `j` lives at `[j * d_in .. (j+1) * d_in]`. Length `r * d_in`.
    v_r: Vec<f32>,
    /// Top-r singular values, descending. Length `r`.
    singular_values: Vec<f32>,
    d_out: usize,
    d_in: usize,
    rank: usize,
}

impl SpectralRewireIndex {
    /// Build an index by SVD-ing `w0` (`d_out × d_in`, row-major) and caching
    /// its top-`rank` singular vectors. Pays the SVD cost ONCE.
    ///
    /// # Panics
    ///
    /// Panics if `w0.len() != d_out * d_in` or `rank == 0`.
    pub fn new(w0: &[f32], d_out: usize, d_in: usize, rank: usize) -> Self {
        assert_eq!(
            w0.len(),
            d_out * d_in,
            "SpectralRewireIndex::new: w0.len() {} != d_out*d_in = {d_out}*{d_in}",
            w0.len()
        );
        assert!(rank >= 1, "SpectralRewireIndex::new: rank must be >= 1");
        let r = rank.min(d_out.min(d_in));

        let mut svd_result = SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = SvdScratch::with_capacity(d_in, d_out);
        thin_svd_into(w0, d_out, d_in, &mut svd_result, &mut svd_work);

        let mut u_r = vec![0.0f32; r * d_out];
        let mut v_r = vec![0.0f32; r * d_in];
        let mut singular_values = vec![0.0f32; r];
        for i in 0..r {
            u_r[i * d_out..(i + 1) * d_out].copy_from_slice(svd_result.left_singular_vector(i));
            v_r[i * d_in..(i + 1) * d_in].copy_from_slice(svd_result.right_singular_vector(i));
            singular_values[i] = svd_result.singular_value(i);
        }

        Self {
            u_r,
            v_r,
            singular_values,
            d_out,
            d_in,
            rank: r,
        }
    }

    /// The cached rank.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The `d_out` the index was built for.
    #[inline]
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// The `d_in` the index was built for.
    #[inline]
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Cached top-r left singular vector `i` (column of U). Length `d_out`.
    #[inline]
    pub fn u_col(&self, i: usize) -> &[f32] {
        &self.u_r[i * self.d_out..(i + 1) * self.d_out]
    }

    /// Cached top-r right singular vector `j` (column of V). Length `d_in`.
    #[inline]
    pub fn v_col(&self, j: usize) -> &[f32] {
        &self.v_r[j * self.d_in..(j + 1) * self.d_in]
    }

    /// Cached singular value `i` (descending).
    #[inline]
    pub fn singular_value(&self, i: usize) -> f32 {
        self.singular_values[i]
    }
}

/// Project `delta` onto the cached base SVD in `index`, skipping the SVD. The
/// hot path for repeated projections against the same base — the per-call cost
/// is just the four matmuls (no Jacobi sweep).
///
/// The `delta` shape must match the index's `(d_out, d_in)`; the scratch is
/// reused for the matmul temporaries.
///
/// # Panics
///
/// Panics if `delta.len() != index.d_out() * index.d_in()`.
pub fn spectral_rewire_with_index_into<'a>(
    index: &'a SpectralRewireIndex,
    delta: &[f32],
    scratch: &'a mut SpectralRewireScratch,
) -> SpectralRewireOutput<'a> {
    let d_out = index.d_out();
    let d_in = index.d_in();
    let r = index.rank();
    let total = d_out * d_in;
    assert_eq!(
        delta.len(),
        total,
        "spectral_rewire_with_index_into: delta.len() {} != d_out*d_in = {total}",
        delta.len()
    );

    scratch.ensure_capacity(d_out, d_in, r);

    project_core(
        delta,
        d_out,
        d_in,
        r,
        total,
        &mut scratch.a_buf,
        &mut scratch.m_buf,
        &mut scratch.b_buf,
        &mut scratch.delta_star_buf,
        &mut scratch.residual_buf,
        |i| index.u_col(i),
        |j| index.v_col(j),
    )
}

// ---------------------------------------------------------------------------
// RewiringDiagnostics — structural readout of the compact rewiring matrix M
// ---------------------------------------------------------------------------

/// Default relative threshold for declaring an off-diagonal rewiring entry
/// negligible: 1% of the largest-magnitude entry in M. Tunable via
/// [`rewiring_matrix_diagnostics_with_threshold`].
const DEFAULT_SPARSITY_REL_THRESHOLD: f32 = 0.01;

/// Structural diagnostics of a rewiring matrix `M = U_rᵀ ΔW V_r` (Plan 423
/// Phase 2).
///
/// `M` decomposes a weight delta into in-skill modulation (the diagonal) and
/// cross-skill rewiring (the off-diagonal). These readouts characterize that
/// structure without re-running the projection. All fields are derived purely
/// from the entries of `M` (no second SVD), so the cost is `O(r²)` and the call
/// is allocation-free.
///
/// - [`diagonal_energy`](RewiringDiagnostics::diagonal_energy) — fraction of
///   total `M` energy on the diagonal. Near `1.0` ⇒ the delta is mostly
///   in-skill strength modulation; near `0.0` ⇒ the delta is mostly
///   cross-skill rewiring (the paper's "many-to-one logical synthesis").
/// - [`off_diagonal_energy`](RewiringDiagnostics::off_diagonal_energy) — the
///   complement, `1.0 − diagonal_energy`.
/// - [`spectral_norm_estimate`](RewiringDiagnostics::spectral_norm_estimate) —
///   the matrix ∞-norm `max_i Σ_j |M[i][j]|`, a standard upper bound on the
///   spectral norm `‖M‖₂`. Cheap (`O(r²)`), allocation-free, and tighter than
///   the raw diagonal max. Used as a magnitude scale for the rewiring.
/// - [`rewiring_sparsity`](RewiringDiagnostics::rewiring_sparsity) — fraction
///   of off-diagonal entries whose magnitude is below
///   `rel_threshold · max|M[i][j]|`. Near `1.0` ⇒ only a few cross-skill
///   links carry the rewiring (sparse composition); near `0.0` ⇒ the
///   rewiring is diffuse across all skill pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewiringDiagnostics {
    /// `Σᵢ M[i][i]² / Σᵢⱼ M[i][j]²` — diagonal energy fraction ∈ `[0, 1]`.
    pub diagonal_energy: f32,
    /// `1.0 − diagonal_energy` — off-diagonal (cross-skill) energy fraction.
    pub off_diagonal_energy: f32,
    /// `max_i Σ_j |M[i][j]|` — ∞-norm upper bound on `‖M‖₂`.
    pub spectral_norm_estimate: f32,
    /// Fraction of off-diagonal entries below the relative negligibility
    /// threshold ∈ `[0, 1]`.
    pub rewiring_sparsity: f32,
}

/// Compute [`RewiringDiagnostics`] for a row-major `rank × rank` rewiring
/// matrix `M`, using the default negligibility threshold
/// (`DEFAULT_SPARSITY_REL_THRESHOLD`, 1% of the max-magnitude entry).
///
/// For a custom threshold, use [`rewiring_matrix_diagnostics_with_threshold`].
///
/// # Panics
///
/// Panics if `m.len() != rank * rank` or `rank == 0`.
///
/// # Example
///
/// A mixed-structure rewiring matrix: strong diagonal (in-skill modulation)
/// plus two cross-skill rewiring links.
///
/// ```
/// use katgpt_spectral::spectral_rewire::rewiring_matrix_diagnostics;
///
/// // Row-major 3×3: diagonal dominates (3, 2, 4), with one strong and one
/// // weak cross-skill link.
/// let m: &[f32] = &[
///     3.0, 0.5, 0.0,
///     0.0, 2.0, 0.0,
///     0.0, 0.0, 4.0,
/// ];
/// let d = rewiring_matrix_diagnostics(m, 3);
///
/// // Most energy is on the diagonal.
/// assert!(d.diagonal_energy > 0.9, "diagonal should dominate: {}", d.diagonal_energy);
/// assert!((d.diagonal_energy + d.off_diagonal_energy - 1.0).abs() < 1e-6);
///
/// // ∞-norm = max row sum: row 0 = |3|+|0.5|+|0| = 3.5; row 2 = 4.0 → max.
/// assert!((d.spectral_norm_estimate - 4.0).abs() < 1e-6);
/// ```
pub fn rewiring_matrix_diagnostics(m: &[f32], rank: usize) -> RewiringDiagnostics {
    rewiring_matrix_diagnostics_with_threshold(m, rank, DEFAULT_SPARSITY_REL_THRESHOLD)
}

/// Compute [`RewiringDiagnostics`] with an explicit relative negligibility
/// threshold for [`RewiringDiagnostics::rewiring_sparsity`].
///
/// `rel_threshold` scales the max-magnitude entry of `M`: an off-diagonal
/// entry counts as negligible when `|M[i][j]| < rel_threshold · max|M|`.
/// A larger threshold declares more entries negligible (higher sparsity);
/// `0.0` means only exact zeros count.
///
/// # Panics
///
/// Panics if `m.len() != rank * rank` or `rank == 0`.
pub fn rewiring_matrix_diagnostics_with_threshold(
    m: &[f32],
    rank: usize,
    rel_threshold: f32,
) -> RewiringDiagnostics {
    assert!(!m.is_empty(), "rewiring_matrix_diagnostics: m is empty");
    assert!(rank > 0, "rewiring_matrix_diagnostics: rank must be > 0");
    assert_eq!(
        m.len(),
        rank * rank,
        "rewiring_matrix_diagnostics: m.len() = {} but rank*rank = {}",
        m.len(),
        rank * rank
    );

    // Single O(r²) pass: accumulate total squared energy, diagonal squared
    // energy, the max-magnitude entry, and the ∞-norm (max absolute row sum).
    let mut total_sq: f32 = 0.0;
    let mut diag_sq: f32 = 0.0;
    let mut max_abs: f32 = 0.0;
    let mut max_row_sum: f32 = 0.0;
    for i in 0..rank {
        let row = &m[i * rank..(i + 1) * rank];
        let mut row_sum: f32 = 0.0;
        for (j, &v) in row.iter().enumerate() {
            let av = v.abs();
            total_sq += v * v;
            if i == j {
                diag_sq += v * v;
            }
            if av > max_abs {
                max_abs = av;
            }
            row_sum += av;
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    // All-zero matrix: undefined diagonal fraction and norm → report zeros,
    // vacuously fully "sparse" (no signal to rewire with).
    if total_sq == 0.0 {
        return RewiringDiagnostics {
            diagonal_energy: 0.0,
            off_diagonal_energy: 0.0,
            spectral_norm_estimate: 0.0,
            rewiring_sparsity: 1.0,
        };
    }

    let diagonal_energy = diag_sq / total_sq;
    let off_diagonal_energy = 1.0 - diagonal_energy;
    let spectral_norm_estimate = max_row_sum;

    // Off-diagonal negligibility count. rank == 1 has no off-diagonal entries
    // → sparsity is vacuously 1.0 (nothing to rewire).
    let off_diag_count = rank.saturating_sub(1) * rank;
    let rewiring_sparsity = if off_diag_count == 0 {
        1.0
    } else {
        let threshold = rel_threshold * max_abs;
        let mut negligible = 0usize;
        for i in 0..rank {
            for j in 0..rank {
                if i != j && m[i * rank + j].abs() < threshold {
                    negligible += 1;
                }
            }
        }
        (negligible as f32) / (off_diag_count as f32)
    };

    RewiringDiagnostics {
        diagonal_energy,
        off_diagonal_energy,
        spectral_norm_estimate,
        rewiring_sparsity,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG (xorshift) for reproducible test matrices.
    fn make_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        }
    }

    /// Fill a `rows × cols` row-major matrix with deterministic pseudo-random
    /// values in roughly [-0.5, 0.5).
    fn rand_matrix(rng: &mut impl FnMut() -> f32, rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols).map(|_| rng()).collect()
    }

    /// Frobenius norm of a flat row-major matrix.
    fn fro_norm(m: &[f32]) -> f32 {
        let sq: f32 = m.iter().map(|&v| v * v).sum();
        sq.sqrt()
    }

    /// Relative Frobenius error: ‖a − b‖_F / ‖b‖_F.
    fn rel_fro_err(a: &[f32], b: &[f32]) -> f32 {
        let diff: f32 = a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum();
        let base: f32 = b.iter().map(|&v| v * v).sum();
        (diff / base).sqrt()
    }

    // ── T1.7: Synthetic delta round-trip ──────────────────────────────────

    #[test]
    fn synthetic_on_manifold_delta_round_trips() {
        // Construct W₀ = random 8×8, compute its thin SVD, construct ΔW = U_r M_true V_rᵀ
        // for a known M_true. The projection must recover M_true and ΔW* = ΔW exactly.
        let mut rng = make_rng(42);
        let d_out = 8;
        let d_in = 8;
        let r = 4;

        let w0 = rand_matrix(&mut rng, d_out, d_in);

        // SVD of W₀ to get U_r, V_r for constructing the synthetic delta.
        let mut svd_result = SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = SvdScratch::with_capacity(d_in, d_out);
        thin_svd_into(&w0, d_out, d_in, &mut svd_result, &mut svd_work);

        // Random M_true (r × r).
        let m_true = rand_matrix(&mut rng, r, r);

        // ΔW = U_r · M_true · V_rᵀ, i.e. ΔW[i][j] = Σ_{a,b} U[i][a]·M[a][b]·V[j][b].
        let mut delta = vec![0.0_f32; d_out * d_in];
        for i in 0..d_out {
            for j in 0..d_in {
                let mut acc = 0.0_f32;
                for a in 0..r {
                    let u_ia = svd_result.left_singular_vector(a)[i];
                    for b in 0..r {
                        let m_ab = m_true[a * r + b];
                        let v_jb = svd_result.right_singular_vector(b)[j];
                        acc += u_ia * m_ab * v_jb;
                    }
                }
                delta[i * d_in + j] = acc;
            }
        }

        // Run spectral_rewire — should recover M_true and ΔW* ≈ ΔW.
        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        // ΔW* should equal ΔW (the delta is exactly on-manifold by construction).
        let err = rel_fro_err(&result.delta_star, &delta);
        assert!(
            err < 1e-4,
            "on-manifold round-trip: ΔW* should match ΔW within 1e-4 rel, got {err:.2e}"
        );

        // on_manifold_fraction should be ~1.0 (delta is entirely on-manifold).
        assert!(
            result.on_manifold_fraction > 0.999,
            "on_manifold_fraction should be ~1.0 for on-manifold delta, got {}",
            result.on_manifold_fraction
        );

        // Residual should be ~0.
        let res_norm = fro_norm(&result.residual);
        let delta_norm = fro_norm(&delta);
        assert!(
            res_norm / delta_norm < 1e-4,
            "residual should be ~0 for on-manifold delta, got {}/{} = {}",
            res_norm,
            delta_norm,
            res_norm / delta_norm
        );

        // M should be recoverable. Note: sign ambiguity in SVD means M may
        // differ from m_true by sign flips per singular vector. We check the
        // Frobenius norm is preserved (energy conservation).
        let m_norm = fro_norm(&result.rewiring_matrix);
        let m_true_norm = fro_norm(&m_true);
        assert!(
            (m_norm - m_true_norm).abs() / m_true_norm < 1e-3,
            "‖M‖_F should match ‖M_true‖_F, got {m_norm} vs {m_true_norm}"
        );
    }

    #[test]
    fn zero_delta_produces_zero_output() {
        let mut rng = make_rng(7);
        let d_out = 6;
        let d_in = 4;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = vec![0.0_f32; d_out * d_in];

        let result = spectral_rewire(&w0, &delta, d_out, d_in, 2);

        assert!(result.delta_star.iter().all(|&v| v.abs() < 1e-10));
        assert!(result.residual.iter().all(|&v| v.abs() < 1e-10));
        assert!(result.on_manifold_fraction.abs() < 1e-10);
    }

    #[test]
    fn on_plus_off_equals_delta() {
        // ΔW_on + ΔW_off must reconstruct ΔW exactly (orthogonal decomposition).
        let mut rng = make_rng(99);
        let d_out = 10;
        let d_in = 8;
        let r = 5;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        // delta_star + residual == delta
        for (idx, ((&delta_star, &residual), &delta)) in result
            .delta_star
            .iter()
            .zip(result.residual.iter())
            .zip(delta.iter())
            .enumerate()
        {
            let recon = delta_star + residual;
            let rel = (recon - delta).abs() / delta.abs().max(1e-10);
            assert!(
                rel < 1e-5,
                "reconstruction failed at idx {idx}: ΔW_on+ΔW_off={recon:.6e}, ΔW={delta:.6e}, rel={rel:.2e}"
            );
        }
    }

    #[test]
    fn on_manifold_fraction_in_unit_interval() {
        let mut rng = make_rng(123);
        for &(d_out, d_in, r) in &[(8, 8, 4), (16, 12, 6), (6, 4, 2)] {
            let w0 = rand_matrix(&mut rng, d_out, d_in);
            let delta = rand_matrix(&mut rng, d_out, d_in);
            let result = spectral_rewire(&w0, &delta, d_out, d_in, r);
            assert!(
                (0.0..=1.0).contains(&result.on_manifold_fraction),
                "on_manifold_fraction out of [0,1]: {} for ({d_out}×{d_in}, r={r})",
                result.on_manifold_fraction
            );
        }
    }

    #[test]
    fn higher_rank_captures_more_energy() {
        // A random delta has no preferred subspace, but increasing rank should
        // never decrease the on-manifold fraction (monotone non-decreasing).
        let mut rng = make_rng(256);
        let d_out = 12;
        let d_in = 12;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        let mut prev = 0.0_f32;
        for r in [2, 4, 6, 8, 12] {
            let result = spectral_rewire(&w0, &delta, d_out, d_in, r);
            assert!(
                result.on_manifold_fraction >= prev - 1e-5,
                "on_manifold_fraction decreased at rank {r}: {} < {prev}",
                result.on_manifold_fraction
            );
            prev = result.on_manifold_fraction;
        }
        // At full rank, the projection is identity → fraction = 1.0.
        assert!(
            prev > 0.999,
            "at full rank, on_manifold_fraction should be 1.0, got {prev}"
        );
    }

    #[test]
    fn non_square_matrix_works() {
        let mut rng = make_rng(777);
        let d_out = 16;
        let d_in = 4;
        let r = 3;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        assert_eq!(result.delta_star.len(), d_out * d_in);
        assert_eq!(result.rewiring_matrix.len(), r * r);
        assert_eq!(result.residual.len(), d_out * d_in);
        assert!((0.0..=1.0).contains(&result.on_manifold_fraction));
    }

    #[test]
    fn scratch_reuse_is_consistent() {
        let mut rng = make_rng(2024);
        let d_out = 8;
        let d_in = 8;
        let r = 4;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        // First call with fresh scratch.
        let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, r);
        let out1 = spectral_rewire_into(&w0, &delta, d_out, d_in, r, &mut scratch);
        let ds1 = out1.delta_star.to_vec();
        let m1 = out1.rewiring_matrix.to_vec();
        let frac1 = out1.on_manifold_fraction;

        // Second call reusing the same scratch — must be identical.
        let out2 = spectral_rewire_into(&w0, &delta, d_out, d_in, r, &mut scratch);
        let ds2 = out2.delta_star.to_vec();
        let m2 = out2.rewiring_matrix.to_vec();
        let frac2 = out2.on_manifold_fraction;

        assert_eq!(ds1, ds2, "delta_star must be identical on scratch reuse");
        assert_eq!(m1, m2, "rewiring_matrix must be identical on scratch reuse");
        assert_eq!(frac1, frac2, "on_manifold_fraction must be identical");
    }

    // ── T2.2: rewiring_matrix_diagnostics on identity M ──────────────────

    #[test]
    fn diagnostics_identity_matrix_is_all_diagonal() {
        // Pure diagonal M (3×3) — all energy on the diagonal, no cross-skill
        // rewiring. Every off-diagonal entry is exactly zero.
        let m: &[f32] = &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let d = rewiring_matrix_diagnostics(m, 3);

        assert!(
            (d.diagonal_energy - 1.0).abs() < 1e-6,
            "diagonal_energy = {}",
            d.diagonal_energy
        );
        assert!(
            d.off_diagonal_energy.abs() < 1e-6,
            "off_diagonal_energy = {}",
            d.off_diagonal_energy
        );
        // ∞-norm = max row sum = max(1, 2, 3) = 3.0.
        assert!(
            (d.spectral_norm_estimate - 3.0).abs() < 1e-6,
            "norm = {}",
            d.spectral_norm_estimate
        );
        // All 6 off-diagonals are zero → fully sparse.
        assert!(
            (d.rewiring_sparsity - 1.0).abs() < 1e-6,
            "sparsity = {}",
            d.rewiring_sparsity
        );
    }

    // ── T2.2: rewiring_matrix_diagnostics on pure off-diagonal M ─────────

    #[test]
    fn diagnostics_pure_off_diagonal_matrix_is_all_rewiring() {
        // Zero diagonal, all off-diagonal entries equal — pure cross-skill
        // rewiring, fully dense (no negligible off-diagonals).
        let m: &[f32] = &[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let d = rewiring_matrix_diagnostics(m, 3);

        assert!(
            d.diagonal_energy.abs() < 1e-6,
            "diagonal_energy = {}",
            d.diagonal_energy
        );
        assert!(
            (d.off_diagonal_energy - 1.0).abs() < 1e-6,
            "off_diagonal_energy = {}",
            d.off_diagonal_energy
        );
        // ∞-norm = max row sum = 2.0 (each row sums |1|+|1| on off-diagonals).
        assert!(
            (d.spectral_norm_estimate - 2.0).abs() < 1e-6,
            "norm = {}",
            d.spectral_norm_estimate
        );
        // No off-diagonal is below 1% of max_abs (1.0) → sparsity = 0.0.
        assert!(
            d.rewiring_sparsity.abs() < 1e-6,
            "sparsity = {}",
            d.rewiring_sparsity
        );
    }

    // ── T2.2: rewiring_matrix_diagnostics on mixed M + edge cases ────────

    #[test]
    fn diagnostics_mixed_and_edge_cases() {
        // Mixed: dominant diagonal (10²) plus one moderate off-diagonal (3²).
        // diagonal_energy = 100 / (100 + 9 + 9) = 100/118 ≈ 0.847.
        let m: &[f32] = &[10.0, 3.0, 3.0, 0.0];
        let d = rewiring_matrix_diagnostics(m, 2);
        assert!(
            (d.diagonal_energy - (100.0_f32 / 118.0)).abs() < 1e-5,
            "diagonal_energy = {} expected ~0.847",
            d.diagonal_energy
        );
        assert!(
            (d.diagonal_energy + d.off_diagonal_energy - 1.0).abs() < 1e-6,
            "diagonal + off-diagonal must sum to 1.0"
        );
        // ∞-norm = max(|10|+|3|, |3|+|0|) = 13.0.
        assert!((d.spectral_norm_estimate - 13.0).abs() < 1e-6);
        // Off-diagonals are both 3.0; threshold = 0.01*10 = 0.1; neither < 0.1
        // → sparsity = 0/2 = 0.0.
        assert!(
            d.rewiring_sparsity.abs() < 1e-6,
            "sparsity = {}",
            d.rewiring_sparsity
        );

        // Raising the threshold to 50% makes 3.0 < 0.5*10 = 5.0 → both negligible.
        let d2 = rewiring_matrix_diagnostics_with_threshold(m, 2, 0.5);
        assert!(
            (d2.rewiring_sparsity - 1.0).abs() < 1e-6,
            "sparsity at 50% threshold = {}",
            d2.rewiring_sparsity
        );

        // All-zero M: total energy 0 → zeroed diagnostics, vacuously sparse.
        let z: &[f32] = &[0.0, 0.0, 0.0, 0.0];
        let dz = rewiring_matrix_diagnostics(z, 2);
        assert_eq!(
            dz,
            RewiringDiagnostics {
                diagonal_energy: 0.0,
                off_diagonal_energy: 0.0,
                spectral_norm_estimate: 0.0,
                rewiring_sparsity: 1.0,
            }
        );

        // rank == 1: no off-diagonal entries → sparsity vacuously 1.0; the
        // single diagonal entry carries all energy.
        let one: &[f32] = &[5.0];
        let d1 = rewiring_matrix_diagnostics(one, 1);
        assert!((d1.diagonal_energy - 1.0).abs() < 1e-6);
        assert!((d1.spectral_norm_estimate - 5.0).abs() < 1e-6);
        assert!((d1.rewiring_sparsity - 1.0).abs() < 1e-6);

        // rank == 0 and wrong-length panic (input validation).
        assert!(std::panic::catch_unwind(|| rewiring_matrix_diagnostics(&[1.0], 0)).is_err());
        assert!(std::panic::catch_unwind(|| rewiring_matrix_diagnostics(&[1.0, 2.0], 2)).is_err());
    }

    // ── Issue 124 resolved: d_in > 64 now WORKS (cap lifted) ──
    // Previously the one-sided Jacobi SVD used fixed [f32; 64] / [usize; 64]
    // stack arrays, capping d_in at 64. After the substrate upgrade (buffers
    // moved to SvdScratch), spectral_rewire works on arbitrary d_in.
    #[test]
    fn spectral_rewire_works_for_d_in_above_old_64_cap() {
        let d_out = 8;
        let d_in = 80; // exceeds the old 64-col cap
        let rank = 4;
        let mut rng = make_rng(7171);
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        // Construct a genuinely on-manifold delta: ΔW = U_r · diag · V_rᵀ.
        let mut svd_result = katgpt_core::SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = katgpt_core::SvdScratch::with_capacity(d_in, d_out);
        katgpt_core::thin_svd_into(&w0, d_out, d_in, &mut svd_result, &mut svd_work);
        let mut delta = vec![0.0_f32; d_out * d_in];
        for j in 0..rank {
            let sv = svd_result.singular_value(j);
            let u = svd_result.left_singular_vector(j);
            let v = svd_result.right_singular_vector(j);
            for r in 0..d_out {
                for c in 0..d_in {
                    delta[r * d_in + c] += sv * u[r] * v[c];
                }
            }
        }

        let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, rank);
        let out = spectral_rewire_into(&w0, &delta, d_out, d_in, rank, &mut scratch);
        // An on-manifold delta must be recovered with high fidelity.
        assert!(
            out.on_manifold_fraction > 0.99,
            "on-manifold delta should be recovered, fraction = {}",
            out.on_manifold_fraction
        );
    }

    // ── SpectralRewireIndex: cached path matches the on-the-fly SVD path ──

    #[test]
    fn cached_index_matches_svd_path() {
        let mut rng = make_rng(303);
        let d_out = 16;
        let d_in = 12;
        let r = 5;
        let w0 = rand_matrix(&mut rng, d_out, d_in);
        let delta = rand_matrix(&mut rng, d_out, d_in);

        // On-the-fly SVD path.
        let mut scratch_a = SpectralRewireScratch::with_capacity(d_out, d_in, r);
        let out_svd = spectral_rewire_into(&w0, &delta, d_out, d_in, r, &mut scratch_a);
        let ds_svd = out_svd.delta_star.to_vec();
        let m_svd = out_svd.rewiring_matrix.to_vec();
        let frac_svd = out_svd.on_manifold_fraction;

        // Cached-index path: build index once, project with it.
        let index = SpectralRewireIndex::new(&w0, d_out, d_in, r);
        let mut scratch_b = SpectralRewireScratch::with_capacity(d_out, d_in, r);
        let out_idx = spectral_rewire_with_index_into(&index, &delta, &mut scratch_b);
        let ds_idx = out_idx.delta_star.to_vec();
        let m_idx = out_idx.rewiring_matrix.to_vec();
        let frac_idx = out_idx.on_manifold_fraction;

        // Both paths use the SAME singular vectors (copied into the index), so
        // outputs must be bit-identical.
        assert_eq!(ds_svd, ds_idx, "delta_star must match between paths");
        assert_eq!(m_svd, m_idx, "rewiring_matrix must match between paths");
        assert_eq!(frac_svd, frac_idx, "on_manifold_fraction must match");

        // Index metadata.
        assert_eq!(index.rank(), r);
        assert_eq!(index.d_out(), d_out);
        assert_eq!(index.d_in(), d_in);
        assert_eq!(index.u_col(0).len(), d_out);
        assert_eq!(index.v_col(0).len(), d_in);
    }

    // ── T2.3 (integration): diagnostics on a real spectral_rewire output ─

    #[test]
    fn diagnostics_on_synthetic_rewiring_output() {
        // Inject a known on-manifold delta: ΔW = U_r M_true V_rᵀ for a diagonal
        // M_true. The recovered rewiring matrix must be diagonal-dominant →
        // diagnostics report high diagonal_energy and full sparsity.
        let mut rng = make_rng(101);
        let d_out = 8;
        let d_in = 8;
        let r = 4;
        let w0 = rand_matrix(&mut rng, d_out, d_in);

        // Reuse the project's SVD to build a genuinely on-manifold delta.
        let mut svd_result = katgpt_core::SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = katgpt_core::SvdScratch::with_capacity(d_in, d_out);
        katgpt_core::thin_svd_into(&w0, d_out, d_in, &mut svd_result, &mut svd_work);

        // Diagonal M_true: [1, 2, 3, 4]. Build ΔW = U_r · diag · V_rᵀ.
        let m_true = [1.0_f32, 2.0, 3.0, 4.0];
        let mut delta = vec![0.0_f32; d_out * d_in];
        for i in 0..d_out {
            for j in 0..d_in {
                let mut acc = 0.0;
                for (k, &m_true_k) in m_true.iter().enumerate().take(r) {
                    let u_k = svd_result.left_singular_vector(k);
                    let v_k = svd_result.right_singular_vector(k);
                    acc += u_k[i] * m_true_k * v_k[j];
                }
                delta[i * d_in + j] = acc;
            }
        }

        let result = spectral_rewire(&w0, &delta, d_out, d_in, r);

        // On-manifold fraction should be ~1.0 (delta lives in the subspace).
        assert!(
            result.on_manifold_fraction > 0.999,
            "on_manifold_fraction = {}",
            result.on_manifold_fraction
        );

        // Diagnostics on the recovered M: diagonal should dominate.
        let d = rewiring_matrix_diagnostics(&result.rewiring_matrix, r);
        assert!(
            d.diagonal_energy > 0.95,
            "diagonal_energy = {} (expected diagonal-dominant from diag M_true)",
            d.diagonal_energy
        );
        assert!((d.diagonal_energy + d.off_diagonal_energy - 1.0).abs() < 1e-5);
        assert!((0.0..=1.0).contains(&d.rewiring_sparsity));
        assert!(d.spectral_norm_estimate > 0.0);
    }

    // ── Issue 123 modelless diagnostic: concentration measurement calibration ─
    //
    // The make-or-break for Issue 123 (Fusion B promotion) is whether REAL
    // training deltas are concentrated. That requires riir-train (no trained
    // weights exist in the 7-repo stack). But we CAN validate the MEASUREMENT
    // is trustworthy: construct deltas with KNOWN on/off mixing ratios and
    // verify the primitive measures the correct fraction. When real deltas
    // eventually arrive, this test guarantees the diagnostic is calibrated.
    //
    // Note: `on_manifold_fraction` = ‖ΔW*‖_F / ‖ΔW‖_F (a NORM ratio, the sqrt
    // of the energy ratio). For a random delta at d_out=d_in=16, r=4, the
    // incidental alignment is √(r²/(d·d)) = √(16/256) = 0.25 — not zero.
    #[test]
    fn concentration_measurement_calibrated_for_mixed_deltas() {
        let d_out = 16;
        let d_in = 16;
        let r = 4;
        let mut rng = make_rng(23456);
        let w0 = rand_matrix(&mut rng, d_out, d_in);

        // SVD of W₀ to construct a genuinely on-manifold component.
        let mut svd_result = katgpt_core::SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = katgpt_core::SvdScratch::with_capacity(d_in, d_out);
        katgpt_core::thin_svd_into(&w0, d_out, d_in, &mut svd_result, &mut svd_work);

        // On-manifold component: ΔW_on = U_r · diag(1..r) · V_rᵀ.
        let mut delta_on = vec![0.0_f32; d_out * d_in];
        for i in 0..d_out {
            for j in 0..d_in {
                let mut acc = 0.0;
                for k in 0..r {
                    let u_k = svd_result.left_singular_vector(k);
                    let v_k = svd_result.right_singular_vector(k);
                    acc += u_k[i] * ((k + 1) as f32) * v_k[j];
                }
                delta_on[i * d_in + j] = acc;
            }
        }

        // Off-manifold component: random delta (NOT aligned with W₀'s subspace).
        let delta_off = rand_matrix(&mut rng, d_out, d_in);

        // Self-calibrate: measure the incidental on-manifold energy of the pure
        // off-manifold component. This accounts for the r²/(d·d) alignment floor.
        let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, r);
        let out_off = spectral_rewire_into(&w0, &delta_off, d_out, d_in, r, &mut scratch);
        let frac_off = out_off.on_manifold_fraction; // ≈ √(r²/(d·d)) ≈ 0.25

        // Pure on-manifold should be ≈ 1.0.
        let out_on = spectral_rewire_into(&w0, &delta_on, d_out, d_in, r, &mut scratch);
        assert!(
            out_on.on_manifold_fraction > 0.999,
            "pure on-manifold should be ≈1.0, got {}",
            out_on.on_manifold_fraction
        );

        // Energies.
        let e_on: f32 = delta_on.iter().map(|v| v * v).sum();
        let e_off: f32 = delta_off.iter().map(|v| v * v).sum();

        // For each mixing ratio α, the expected on_manifold_fraction (norm ratio):
        //   fraction = sqrt( (α²·e_on + (1-α)²·e_off·frac_off²) / (α²·e_on + (1-α)²·e_off) )
        // where frac_off² is the incidental energy fraction of the random component.
        let frac_off_sq = frac_off * frac_off;
        for &alpha in &[0.25_f32, 0.5, 0.75] {
            let mut delta = vec![0.0_f32; d_out * d_in];
            for idx in 0..delta.len() {
                delta[idx] = alpha * delta_on[idx] + (1.0 - alpha) * delta_off[idx];
            }
            let out = spectral_rewire_into(&w0, &delta, d_out, d_in, r, &mut scratch);

            let a2 = alpha * alpha;
            let b2 = (1.0 - alpha) * (1.0 - alpha);
            let numerator = a2 * e_on + b2 * e_off * frac_off_sq;
            let denominator = a2 * e_on + b2 * e_off;
            let expected = (numerator / denominator).sqrt();

            let measured = out.on_manifold_fraction;
            assert!(
                (measured - expected).abs() < 0.03,
                "α={alpha}: measured={measured:.4}, expected≈{expected:.4}"
            );
        }
    }

    // ── Issue 123 modelless diagnostic: two-component separation ──
    //
    // Given a delta with known on/off structure, verify the primitive correctly
    // separates the two components: the recovered delta_star should match the
    // on-manifold component, and the residual should match the off-manifold
    // component. This validates the Fusion B decomposition MECHANISM.
    #[test]
    fn two_component_separation_recovers_known_components() {
        let d_out = 12;
        let d_in = 12;
        let r = 3;
        let mut rng = make_rng(34567);
        let w0 = rand_matrix(&mut rng, d_out, d_in);

        let mut svd_result = katgpt_core::SvdResultScratch::with_capacity(d_out, d_in);
        let mut svd_work = katgpt_core::SvdScratch::with_capacity(d_in, d_out);
        katgpt_core::thin_svd_into(&w0, d_out, d_in, &mut svd_result, &mut svd_work);

        // On-manifold: ΔW_on = U_r · diag(2,1,0.5) · V_rᵀ.
        let m_diag = [2.0_f32, 1.0, 0.5];
        let mut delta_on = vec![0.0_f32; d_out * d_in];
        for i in 0..d_out {
            for j in 0..d_in {
                let mut acc = 0.0;
                for (k, &m_diag_k) in m_diag.iter().enumerate().take(r) {
                    acc += svd_result.left_singular_vector(k)[i]
                        * m_diag_k
                        * svd_result.right_singular_vector(k)[j];
                }
                delta_on[i * d_in + j] = acc;
            }
        }

        // Off-manifold: a random component orthogonal to U_r/V_r.
        // Project out the top-r subspace to guarantee orthogonality.
        let raw_off = rand_matrix(&mut rng, d_out, d_in);
        let mut delta_off = vec![0.0_f32; d_out * d_in];
        for i in 0..d_out {
            for j in 0..d_in {
                // Subtract the projection onto each U_k V_lᵀ direction.
                let mut val = raw_off[i * d_in + j];
                for k in 0..r {
                    for l in 0..r {
                        let u_k = svd_result.left_singular_vector(k);
                        let v_l = svd_result.right_singular_vector(l);
                        // Coefficient of raw_off on U_k V_lᵀ.
                        let mut coef = 0.0;
                        for a in 0..d_out {
                            for b in 0..d_in {
                                coef += raw_off[a * d_in + b] * u_k[a] * v_l[b];
                            }
                        }
                        val -= coef * u_k[i] * v_l[j];
                    }
                }
                delta_off[i * d_in + j] = val;
            }
        }

        // Mixed delta: equal energy on and off.
        let e_on: f32 = delta_on.iter().map(|v| v * v).sum::<f32>().sqrt();
        let e_off: f32 = delta_off.iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale_on = 1.0 / e_on;
        let scale_off = 1.0 / e_off;
        let delta: Vec<f32> = (0..d_out * d_in)
            .map(|idx| scale_on * delta_on[idx] + scale_off * delta_off[idx])
            .collect();

        let mut scratch = SpectralRewireScratch::with_capacity(d_out, d_in, r);
        let out = spectral_rewire_into(&w0, &delta, d_out, d_in, r, &mut scratch);

        // The on-manifold fraction (norm ratio) for an equal-energy mix where
        // the off-component is orthogonal to the subspace should be √0.5 ≈ 0.707.
        assert!(
            (out.on_manifold_fraction - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.03,
            "equal-energy orthogonal mix: norm fraction ≈ √0.5, got {}",
            out.on_manifold_fraction
        );

        // Recomposition: delta_star + residual ≈ delta (holds by construction).
        let ds = out.delta_star;
        let res = out.residual;
        let max_err = (0..delta.len())
            .map(|i| (ds[i] + res[i] - delta[i]).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_err < 1e-4, "recomposition error = {max_err}");

        // The recovered delta_star should be close to the normalized on-manifold component.
        let star_err: f32 = (0..delta.len())
            .map(|i| (ds[i] - scale_on * delta_on[i]).powi(2))
            .sum::<f32>()
            .sqrt();
        let star_norm: f32 = ds.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            star_err / star_norm < 0.05,
            "delta_star should match on-manifold component, rel err = {}",
            star_err / star_norm
        );
    }

    /// Issue 940 SP-3 pin: `project_core`'s `M = A·V_r` folds through the
    /// ordered dot, not the platform lane kernel. Driven through the REAL
    /// `project_core` (same module — it is private by design) with a crafted
    /// rank-1 shape whose algebra makes every step bit-exact: `u_col(0) = e₀`
    /// (d_out = 1) so `A[0][:] = delta[0][:]` exactly, `v_col(0) = ones(16)`,
    /// and `delta[0][:]` the certified len-16 cancellation craft — on which
    /// the ordered dot reads 3.0 while every reassociating backend reads 6.0
    /// (the `ordered_dot_differs_from_simd_dot` craft). B = U·M and
    /// ΔW* = B·Vᵀ then propagate that single cell, so the whole output flips
    /// if the fold reverts to platform order.
    #[test]
    fn rewiring_matrix_folds_in_index_order_not_platform_order() {
        use katgpt_core::simd::{dot_f32_ordered, simd_dot_f32};

        let d_in = 16usize;
        let mut delta_row = [0.0f32; 16];
        delta_row[0] = 1e8;
        delta_row[1] = 1.0;
        delta_row[2] = 1.0;
        delta_row[3] = 1.0;
        delta_row[4] = -1e8;
        delta_row[5] = 1.0;
        delta_row[6] = 1.0;
        delta_row[7] = 1.0;

        // The craft is load-bearing: the two folds disagree on this row.
        assert_ne!(
            dot_f32_ordered(&delta_row, &[1.0f32; 16]),
            simd_dot_f32(&delta_row, &[1.0f32; 16], 16),
            "craft no longer diverges: the pin stopped being load-bearing"
        );

        let d_out = 1usize;
        let r = 1usize;
        let u = [1.0f32]; // U (1×1): u_col(0) = [1.0]
        let v = [1.0f32; 16]; // V column: ones

        let mut a_buf = vec![0.0f32; r * d_in];
        let mut m_buf = vec![0.0f32; r * r];
        let mut b_buf = vec![0.0f32; d_out * r];
        let mut delta_star_buf = vec![0.0f32; d_out * d_in];
        let mut residual_buf = vec![0.0f32; d_out * d_in];

        let out = project_core(
            &delta_row,
            d_out,
            d_in,
            r,
            d_out * d_in,
            &mut a_buf,
            &mut m_buf,
            &mut b_buf,
            &mut delta_star_buf,
            &mut residual_buf,
            |i| {
                let _ = i;
                &u[..]
            },
            |j| {
                let _ = j;
                &v[..]
            },
        );

        // Ordered fold: Σ craft = 1e8 +1 +1 +1 −1e8 +1 +1 +1 = 3.0 exactly.
        assert_eq!(
            out.rewiring_matrix[0].to_bits(),
            3.0f32.to_bits(),
            "M[0][0] is not the ordered fold (platform-kernel regression?)"
        );
        // The single M cell propagates bit-exactly: B[0][0] = 1·3.0, then
        // ΔW*[0][j] = 3.0·1.0 for every j.
        for &cell in out.delta_star {
            assert_eq!(cell.to_bits(), 3.0f32.to_bits());
        }
    }
}
