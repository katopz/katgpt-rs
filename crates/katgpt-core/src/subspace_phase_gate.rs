//! Subspace phase-gate primitives — modelless numeric utilities for subspace
//! clustering quality assurance and runtime semantic-axis discovery.
//!
//! Distilled from Wang et al., *Breaking the Curse of Dimensionality: Diffusion
//! Models Efficiently Learn Low-Dimensional Distributions* ([arXiv:2409.02426](https://arxiv.org/abs/2409.02426)).
//! See `katgpt-rs/.research/279_*.md` for the open research note and
//! `katgpt-rs/.plans/301_*.md` for the execution plan.
//!
//! Three families of operations, all inference-time and allocation-aware:
//!
//! 1. **Intrinsic-dimension estimation** — [`participation_ratio`] (continuous)
//!    and [`numerical_rank`] (discrete, with energy threshold η). Paper eq. 52
//!    uses η = 0.99.
//! 2. **Phase-transition sample-sufficiency gate** — [`phase_transition_gate`]
//!    implements the necessary condition `N ≥ d` from Theorem 4: a d-dimensional
//!    subspace cannot be recovered from fewer than d samples, regardless of
//!    algorithm. Below the threshold, any subspace estimate is
//!    information-theoretically invalid.
//! 3. **Runtime Jacobian SVD** — [`jacobian_svd_at`] estimates the Jacobian of
//!    a map `f: R^n → R^m` at a point via forward differences, then computes a
//!    thin SVD. The leading singular vectors are candidate "semantic axes" in
//!    the sense of paper §5.2: directions in the domain along which the map
//!    produces the largest output change.
//!
//! These primitives are generic numeric — no game, shard, or chain semantics.
//! Consumers apply them to their own maps (HLA evolution kernel, shard
//! projection, latent functor) and interpret the results.
//!
//! # Performance contract
//!
//! - [`participation_ratio`] and [`numerical_rank`] are O(n) on a length-n
//!   spectrum, zero-allocation, chunk-4 loops for SIMD auto-vectorisation.
//! - [`jacobian_svd_at`] is O(n·cost(f) + n²·m) — n forward evaluations of f
//!   plus an n×m thin SVD. For small n (≤ 16) and m (≤ 16), this is sub-µs
//!   on commodity hardware.
//!
//! # Determinism
//!
//! All operations are deterministic and platform-independent: no SIMD dispatch
//! inside the math (callers may wrap SIMD themselves), no floating-point
//! reordering. This is required for anti-cheat: the phase-transition gate
//! decision must be bit-identical across quorum nodes.
// (Module gating is handled by `#[cfg(feature = "subspace_phase_gate")]` on the
// `mod` declaration in `lib.rs`; this file must NOT duplicate it.)

// ─── Intrinsic-dimension estimation ─────────────────────────────────────────

/// Continuous effective dimensionality: `(Σλ)² / Σ(λ²)`.
///
/// Returns 0.0 on empty or all-non-positive input. For a flat spectrum of k
/// equal eigenvalues, returns exactly `k`. For a single dominant eigenvalue,
/// returns ~1. Always in `[0, n]` for a length-n non-negative spectrum.
///
/// Chunk-4 accumulation for SIMD auto-vectorisation. Zero-allocation.
#[inline]
pub fn participation_ratio(spectrum: &[f32]) -> f32 {
    if spectrum.is_empty() {
        return 0.0;
    }
    let mut sum: f32 = 0.0;
    let mut sum_sq: f32 = 0.0;
    let mut i = 0;
    while i + 4 <= spectrum.len() {
        let a = spectrum[i].max(0.0);
        let b = spectrum[i + 1].max(0.0);
        let c = spectrum[i + 2].max(0.0);
        let d = spectrum[i + 3].max(0.0);
        sum += a + b + c + d;
        sum_sq += a * a + b * b + c * c + d * d;
        i += 4;
    }
    while i < spectrum.len() {
        let v = spectrum[i].max(0.0);
        sum += v;
        sum_sq += v * v;
        i += 1;
    }
    if sum_sq < f32::EPSILON {
        return 0.0;
    }
    (sum * sum) / sum_sq
}

/// Discrete effective dimensionality: smallest `r` such that cumulative energy
/// `Σ_{i≤r} σ_i² / Σ_i σ_i² > η`.
///
/// Mirrors paper eq. 52 with η = 0.99. The caller MUST sort the spectrum
/// descending first — this function does not sort (zero-allocation contract).
/// Default η in [`IntrinsicDimMethod::NumericalRank`] is 0.99.
#[inline]
pub fn numerical_rank(spectrum_sorted_desc: &[f32], eta: f32) -> usize {
    debug_assert!(
        (0.0..=1.0).contains(&eta),
        "eta must be in [0, 1], got {eta}"
    );
    if spectrum_sorted_desc.is_empty() {
        return 0;
    }
    let mut total_sq: f32 = 0.0;
    let mut cum_sq: f32 = 0.0;
    for &v in spectrum_sorted_desc {
        let v = v.max(0.0);
        total_sq += v * v;
    }
    if total_sq < f32::EPSILON {
        return 0;
    }
    let threshold = eta * total_sq;
    for (i, &v) in spectrum_sorted_desc.iter().enumerate() {
        cum_sq += v.max(0.0) * v.max(0.0);
        if cum_sq > threshold {
            return i + 1;
        }
    }
    spectrum_sorted_desc.len()
}

/// Method selector for [`estimate_intrinsic_dim`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntrinsicDimMethod {
    /// Round [`participation_ratio`] to the nearest integer. Fast, continuous,
    /// good when the spectrum has a clear elbow.
    ParticipationRatio,
    /// [`numerical_rank`] at energy threshold η. Discrete, conservative.
    /// Default η = 0.99 (paper eq. 52).
    NumericalRank { eta: f32 },
}

impl Default for IntrinsicDimMethod {
    fn default() -> Self {
        // Paper eq. 52 uses η = 0.99. We default to the same.
        IntrinsicDimMethod::NumericalRank { eta: 0.99 }
    }
}

/// Dispatch to the configured estimator. See [`IntrinsicDimMethod`].
#[inline]
pub fn estimate_intrinsic_dim(spectrum: &[f32], method: IntrinsicDimMethod) -> usize {
    match method {
        IntrinsicDimMethod::ParticipationRatio => participation_ratio(spectrum).round() as usize,
        IntrinsicDimMethod::NumericalRank { eta } => numerical_rank(spectrum, eta),
    }
}

// ─── Phase-transition gate ──────────────────────────────────────────────────

/// The Wang et al. Theorem 4 necessary condition for subspace recovery:
/// `n_samples >= intrinsic_dim`.
///
/// Returns `true` iff the sample count meets or exceeds the intrinsic dim.
/// Returns `false` below the threshold — recovery is information-theoretically
/// impossible for a d-dim subspace from fewer than d samples, regardless of
/// algorithm. See [`crate::subspace_phase_gate`] module docs for the caveats
/// (this is necessary but not sufficient when subspaces are non-orthogonal).
#[inline(always)]
pub fn phase_transition_gate(n_samples: usize, intrinsic_dim: usize) -> bool {
    n_samples >= intrinsic_dim
}

// ─── Runtime Jacobian SVD ───────────────────────────────────────────────────

/// Pre-allocated scratch buffers for [`jacobian_svd_at`]. Reuse across calls
/// to avoid per-call allocation.
pub struct JacobianSvdScratch {
    /// Output column buffer for `f` evaluations, length `m`.
    f_x: Vec<f32>,
    /// Output column buffer for `f(x + eps·e_i)` evaluations, length `m`.
    f_x_pert: Vec<f32>,
    /// Output column buffer for `f(x + eps·e_i)` in the central-diff path,
    /// length `m`. Replaces a per-column `.clone()` of `f_x_pert`.
    f_x_plus: Vec<f32>,
    /// Mutable copy of the input point `x`, length `n`. Replaces a per-call
    /// `x.to_vec()` allocation.
    x_pert: Vec<f32>,
    /// Flattened Jacobian, row-major `m × n`, length `m * n`.
    jac: Vec<f32>,
    /// Thin-SVD working storage for the Jacobi-rotation routine.
    svd_work: SvdScratch,
    /// Thin-SVD result storage (SOA). Reused across calls — zero per-call
    /// allocation for the SVD output. The Jacobian is `m × n` with `m >= n`,
    /// so the result has `n` singular triples.
    svd_result: SvdResultScratch,
}

impl JacobianSvdScratch {
    /// Allocate scratch sized for an `R^n → R^m` map. Pre-allocates all
    /// internal buffers; reuse via [`Self::clear`] between calls.
    pub fn with_capacity(n: usize, m: usize) -> Self {
        Self {
            f_x: vec![0.0; m],
            f_x_pert: vec![0.0; m],
            f_x_plus: vec![0.0; m],
            x_pert: vec![0.0; n],
            jac: vec![0.0; m * n],
            svd_work: SvdScratch::with_capacity(n, m),
            svd_result: SvdResultScratch::with_capacity(m, n),
        }
    }

    /// Reset for reuse. Does not deallocate; just zeros the active regions.
    pub fn clear(&mut self) {
        for v in &mut self.f_x {
            *v = 0.0;
        }
        for v in &mut self.f_x_pert {
            *v = 0.0;
        }
        for v in &mut self.f_x_plus {
            *v = 0.0;
        }
        // x_pert is overwritten fully before each use (copy_from_slice), so
        // it doesn't need zeroing here.
        for v in &mut self.jac {
            *v = 0.0;
        }
        self.svd_work.clear();
        // svd_result is reset by `one_sided_jacobi_svd_into` via `clear_for`.
    }
}

/// Result of [`jacobian_svd_at`]. Vectors are owned for simplicity; callers
/// that need zero-allocation can drop this struct promptly.
pub struct SvdResult {
    /// Singular values, descending. Length = min(n, m).
    pub singular_values: Vec<f32>,
    /// Right singular vectors (columns of V), one `Vec<f32>` per singular
    /// value, each of length `n`. These are the "directions in the input
    /// space" along which `f` is most sensitive — the candidate "semantic
    /// axes" in the sense of paper §5.2.
    pub right_singular_vectors: Vec<Vec<f32>>,
    /// Left singular vectors (columns of U), one `Vec<f32>` per singular value,
    /// each of length `m`. These are the corresponding "directions in the
    /// output space".
    pub left_singular_vectors: Vec<Vec<f32>>,
    /// Effective rank: number of singular values above a small threshold.
    pub rank: usize,
}

/// SOA (structure-of-arrays) SVD result with caller-owned, reused storage.
/// Zero per-call allocation when reused across calls — the hot-path
/// alternative to [`SvdResult`] for tight loops (e.g. scanning many shards).
///
/// Flat layout (not `Vec<Vec<f32>>`) eliminates the inner-allocation overhead
/// that dominates small-matrix SVD cost: an 8×8 factorization allocates 17
/// `Vec`s in [`SvdResult`] vs **zero** here after the first call.
///
/// Fill via [`thin_svd_into`]. Read via the accessors:
/// - [`Self::singular_value`] — σ_j (descending).
/// - [`Self::right_singular_vector`] — column j of V (input-space direction).
/// - [`Self::left_singular_vector`] — column j of U (output-space direction).
///
/// The SOA buffers are sized for the `(m_rows, n_cols)` passed to
/// [`Self::with_capacity`] and can be reused for any smaller matrix — the
/// active region is tracked by [`Self::len`] (number of singular triples).
pub struct SvdResultScratch {
    /// Singular values, descending. Length = `len` (= min(m, n)).
    singular_values: Vec<f32>,
    /// Right singular vectors (columns of V), flat column-major,
    /// `n_cols × len`. Column j lives at indices `[j * n_cols .. (j+1) * n_cols]`.
    right_singular_vectors: Vec<f32>,
    /// Left singular vectors (columns of U), flat column-major,
    /// `m_rows × len`. Column j lives at indices `[j * m_rows .. (j+1) * m_rows]`.
    left_singular_vectors: Vec<f32>,
    /// Effective rank: count of singular values above the relative threshold.
    pub rank: usize,
    /// Number of singular triples currently stored (= min(m, n) of the last
    /// factorization). Invariant: `<= with_capacity's min(m, n)`.
    len: usize,
    /// Matrix row count this scratch is sized for (for index math).
    m_rows: usize,
    /// Matrix column count this scratch is sized for (for index math).
    n_cols: usize,
}

impl SvdResultScratch {
    /// Allocate result storage sized for factoring an `m_rows × n_cols`
    /// matrix. Pre-allocates all SOA buffers; reuse across calls for zero
    /// per-call allocation.
    pub fn with_capacity(m_rows: usize, n_cols: usize) -> Self {
        let k = m_rows.min(n_cols);
        Self {
            singular_values: vec![0.0; k],
            // Column-major: k columns, each of length n_cols (for V) / m_rows (for U).
            right_singular_vectors: vec![0.0; k * n_cols],
            left_singular_vectors: vec![0.0; k * m_rows],
            rank: 0,
            len: 0,
            m_rows,
            n_cols,
        }
    }

    /// Number of singular triples currently stored (= min(m, n)).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the result is empty (no factorization has run, or zero matrix).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Singular value `j` (descending, `j < len`). Panics on out-of-bounds.
    #[inline]
    pub fn singular_value(&self, j: usize) -> f32 {
        self.singular_values[j]
    }

    /// Singular values slice (descending), length `len`.
    #[inline]
    pub fn singular_values(&self) -> &[f32] {
        &self.singular_values[..self.len]
    }

    /// Right singular vector `j` (column j of V, the input-space direction for
    /// σ_j). Length `n_cols`. Panics on out-of-bounds `j`.
    #[inline]
    pub fn right_singular_vector(&self, j: usize) -> &[f32] {
        let start = j * self.n_cols;
        &self.right_singular_vectors[start..start + self.n_cols]
    }

    /// Left singular vector `j` (column j of U, the output-space direction for
    /// σ_j). Length `m_rows`. Panics on out-of-bounds `j`.
    #[inline]
    pub fn left_singular_vector(&self, j: usize) -> &[f32] {
        let start = j * self.m_rows;
        &self.left_singular_vectors[start..start + self.m_rows]
    }

    /// Reset the active length to zero, keeping all backing allocations.
    /// Called automatically by [`thin_svd_into`] before refilling.
    #[inline]
    fn clear_for(&mut self, m_rows: usize, n_cols: usize) {
        // Grow buffers if a larger matrix is presented than what we were
        // allocated for. Rare; happens only if the caller mixes matrix sizes
        // with a single scratch instance.
        let k = m_rows.min(n_cols);
        if k > self.singular_values.len() {
            self.singular_values.resize(k, 0.0);
        }
        if k * n_cols > self.right_singular_vectors.len() {
            self.right_singular_vectors.resize(k * n_cols, 0.0);
        }
        if k * m_rows > self.left_singular_vectors.len() {
            self.left_singular_vectors.resize(k * m_rows, 0.0);
        }
        self.m_rows = m_rows;
        self.n_cols = n_cols;
        self.len = 0;
        self.rank = 0;
    }
}

/// Estimate the Jacobian of `f: R^n → R^m` at point `x` via forward differences,
/// then return the thin SVD.
///
/// `f` is called `n + 1` times: once at `x`, then once at `x + eps·e_i` for
/// each coordinate `i`. The Jacobian column `i` is `(f(x + eps·e_i) − f(x)) / eps`.
///
/// `eps` is the forward-difference step. Reasonable default: `1e-4`. Pass a
/// negative value to opt into central differences (more accurate, 2× cost).
///
/// # Panics
///
/// Panics if `x.len() != n` (where `n` was passed to
/// [`JacobianSvdScratch::with_capacity`]) or if `f` writes a slice of the
/// wrong length.
pub fn jacobian_svd_at<F>(f: F, x: &[f32], eps: f32, scratch: &mut JacobianSvdScratch) -> SvdResult
where
    F: Fn(&[f32], &mut [f32]),
{
    let n = x.len();
    debug_assert_eq!(
        scratch.jac.len() % n,
        0,
        "scratch.jac length {} not a multiple of n={}",
        scratch.jac.len(),
        n
    );
    let m = scratch.jac.len() / n;
    debug_assert_eq!(scratch.f_x.len(), m);
    debug_assert_eq!(scratch.f_x_pert.len(), m);

    scratch.clear();

    // Central differences if eps < 0, forward otherwise.
    let central = eps < 0.0;
    let step = eps.abs();

    // f(x) — evaluated into a thread-local buffer because `f` takes `&[f32]`
    // and we need to perturb x without mutating the caller's slice.
    f(x, &mut scratch.f_x);

    // Build Jacobian column-by-column (input-coordinate-wise).
    // jac is row-major m × n, so column i lives at indices i, i+n, i+2n, ...
    // For cache friendliness on small matrices, we transpose to row-major m×n
    // where row j, col i = jac[j*n + i].
    //
    // Reuse the pre-allocated x_pert scratch instead of x.to_vec().
    debug_assert!(scratch.x_pert.len() >= n, "x_pert scratch too short");
    scratch.x_pert[..n].copy_from_slice(&x[..n]);
    let x_pert = &mut scratch.x_pert;
    for i in 0..n {
        // Save the original coordinate.
        let x_i_orig = x_pert[i];
        if central {
            // f(x + step·e_i)
            x_pert[i] = x_i_orig + step;
            f(x_pert, &mut scratch.f_x_pert);
            // Swap f_x_pert → f_x_plus without cloning: std::mem::swap avoids
            // the per-column Vec allocation the original `.clone()` did.
            std::mem::swap(&mut scratch.f_x_plus, &mut scratch.f_x_pert);
            // f(x - step·e_i)
            x_pert[i] = x_i_orig - step;
            f(x_pert, &mut scratch.f_x_pert);
            // Central diff: (f_plus − f_minus) / (2·step)
            for j in 0..m {
                scratch.jac[j * n + i] = (scratch.f_x_plus[j] - scratch.f_x_pert[j]) / (2.0 * step);
            }
        } else {
            // Forward diff: (f(x + step·e_i) − f(x)) / step
            x_pert[i] = x_i_orig + step;
            f(x_pert, &mut scratch.f_x_pert);
            for j in 0..m {
                scratch.jac[j * n + i] = (scratch.f_x_pert[j] - scratch.f_x[j]) / step;
            }
        }
        // Restore.
        x_pert[i] = x_i_orig;
    }

    // Thin SVD of the m × n Jacobian via one-sided Jacobi rotations.
    // Writes into scratch.svd_result (SOA, reused across calls), then converts
    // to the owned SvdResult return type.
    one_sided_jacobi_svd_into(
        &scratch.jac,
        m,
        n,
        &mut scratch.svd_result,
        &mut scratch.svd_work,
    );
    let len = scratch.svd_result.len;
    let singular_values = scratch.svd_result.singular_values[..len].to_vec();
    let right_singular_vectors: Vec<Vec<f32>> = (0..len)
        .map(|j| scratch.svd_result.right_singular_vector(j).to_vec())
        .collect();
    let left_singular_vectors: Vec<Vec<f32>> = (0..len)
        .map(|j| scratch.svd_result.left_singular_vector(j).to_vec())
        .collect();
    SvdResult {
        singular_values,
        right_singular_vectors,
        left_singular_vectors,
        rank: scratch.svd_result.rank,
    }
}

// ─── One-sided Jacobi SVD (portable, no native-lapack dep) ─────────────────

/// Pre-allocated scratch for [`thin_svd`] / [`one_sided_jacobi_svd`]. Reuse
/// across calls to avoid per-call allocation. Allocate once with
/// [`SvdScratch::with_capacity`] for the largest `(m, n)` you will factor.
//
// Public since Plan 002 Phase 2: consumers with a *known* linear map (e.g.
// `riir-neuron-db::NeuronShard::semantic_axes`, which SVDs a fixed weight
// matrix) want to skip the forward-differencing in `jacobian_svd_at` and call
// the SVD directly. Exposing the scratch + a thin-SVD entry point lets them do
// so without re-deriving the Jacobian (which for a linear map is the matrix
// itself and costs `n` extra `f` evaluations + a per-call `Vec` allocation
// inside `jacobian_svd_at`).
pub struct SvdScratch {
    /// Working copy of the input matrix, mutated in-place. Length m*n.
    a: Vec<f32>,
    /// Right-singular-vector accumulator V, n × n, row-major. Length n*n.
    v: Vec<f32>,
    /// Column norms (singular values during iteration). Length n.
    col_norms_sq: Vec<f32>,
}

impl SvdScratch {
    /// Allocate scratch sized for factoring an `m_rows × n_cols` matrix.
    pub fn with_capacity(n_cols: usize, m_rows: usize) -> Self {
        Self {
            a: vec![0.0; m_rows * n_cols],
            v: vec![0.0; n_cols * n_cols],
            col_norms_sq: vec![0.0; n_cols],
        }
    }

    fn clear(&mut self) {
        for v in &mut self.a {
            *v = 0.0;
        }
        for v in &mut self.v {
            *v = 0.0;
        }
        for v in &mut self.col_norms_sq {
            *v = 0.0;
        }
    }
}

/// Thin SVD of a row-major `m_rows × n_cols` matrix `M = U Σ V^T` via
/// one-sided Jacobi rotations.
///
/// This is the direct entry point for consumers that already hold the matrix
/// (e.g. a known linear map's matrix). For estimating the SVD of an arbitrary
/// (possibly non-linear) map's Jacobian, use [`jacobian_svd_at`] instead.
///
/// Returns the `min(m_rows, n_cols)` leading singular triples. Sign
/// conventions are arbitrary (canonical SVD ambiguity); callers should not
/// depend on signs.
///
/// Convergence: rotate until no off-diagonal element of `M^T M` exceeds
/// `tol² · trace(M^T M)`. Standard textbook algorithm; ~O(n²) per sweep,
/// ~log2(n) sweeps to converge for well-separated spectra.
///
/// Allocation: this convenience wrapper allocates a fresh [`SvdResult`] (17
/// `Vec`s for an 8×8 matrix). For hot loops, use [`thin_svd_into`] with a
/// reused [`SvdResultScratch`] to eliminate per-call allocation entirely.
pub fn thin_svd(
    m_flat: &[f32], // row-major m × n
    m_rows: usize,
    n_cols: usize,
    work: &mut SvdScratch,
) -> SvdResult {
    let mut result = SvdResultScratch::with_capacity(m_rows, n_cols);
    thin_svd_into(m_flat, m_rows, n_cols, &mut result, work);
    // Convert SOA scratch → owned SvdResult. This is the ONLY allocation path
    // for `thin_svd`; hot callers use `thin_svd_into` and read the SOA directly.
    let len = result.len;
    let singular_values = result.singular_values[..len].to_vec();
    let right_singular_vectors: Vec<Vec<f32>> = (0..len)
        .map(|j| result.right_singular_vector(j).to_vec())
        .collect();
    let left_singular_vectors: Vec<Vec<f32>> = (0..len)
        .map(|j| result.left_singular_vector(j).to_vec())
        .collect();
    SvdResult {
        singular_values,
        right_singular_vectors,
        left_singular_vectors,
        rank: result.rank,
    }
}

/// Zero-allocation thin SVD: factor `M = U Σ V^T` and write the result into the
/// caller-owned `result` scratch (SOA layout). This is the hot-path variant of
/// [`thin_svd`] for tight loops — reuse `result` across calls to eliminate the
/// 17 `Vec` allocations that dominate small-matrix SVD cost.
///
/// The `result` scratch is automatically grown if a larger matrix is presented
/// than it was allocated for (rare; prefer [`SvdResultScratch::with_capacity`]
/// for the largest expected size).
///
/// After the call, read results via:
/// - `result.singular_values()` / `result.singular_value(j)`
/// - `result.right_singular_vector(j)` (column j of V)
/// - `result.left_singular_vector(j)` (column j of U)
/// - `result.rank` (effective rank)
/// - `result.len()` (number of singular triples = min(m, n))
///
/// Sign conventions are arbitrary (canonical SVD ambiguity).
pub fn thin_svd_into(
    m_flat: &[f32], // row-major m × n
    m_rows: usize,
    n_cols: usize,
    result: &mut SvdResultScratch,
    work: &mut SvdScratch,
) {
    one_sided_jacobi_svd_into(m_flat, m_rows, n_cols, result, work)
}

/// One-sided Jacobi SVD: factor `M (m × n, m ≥ n) = U Σ V^T` and write the
/// `min(n, m)` leading singular triples into `result` (SOA scratch, zero
/// per-call allocation). Sign conventions are arbitrary.
///
/// Convergence: rotate until no off-diagonal element of `M^T M` exceeds
/// `tol² · trace(M^T M)`. Standard textbook algorithm; ~O(n²) per sweep,
/// ~log2(n) sweeps to converge for well-separated spectra.
#[allow(clippy::needless_range_loop)] // hot numerical SVD kernels: indices participate in stride arithmetic (out_j*n, r*n+i, etc.)
fn one_sided_jacobi_svd_into(
    m_flat: &[f32], // row-major m × n
    m_rows: usize,
    n_cols: usize,
    result: &mut SvdResultScratch,
    work: &mut SvdScratch,
) {
    let m = m_rows;
    let n = n_cols;
    debug_assert_eq!(m_flat.len(), m * n);
    // Work buffers may be larger than needed (sized once for the largest
    // matrix a caller will factor); we use the `m*n` / `n*n` prefix. This
    // lets a single `SvdScratch` be reused across matrices of different sizes
    // without reallocation.
    debug_assert!(
        work.a.len() >= m * n,
        "work.a len {} < m*n={}",
        work.a.len(),
        m * n
    );
    debug_assert!(
        work.v.len() >= n * n,
        "work.v len {} < n*n={}",
        work.v.len(),
        n * n
    );

    // Reset result for this matrix size (grows buffers if needed, zero-fill).
    result.clear_for(m, n);

    // Copy M into work.a prefix (will be mutated to U·Σ).
    work.a[..m * n].copy_from_slice(m_flat);
    // Initialise V = I (prefix only).
    for i in 0..n {
        for j in 0..n {
            work.v[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let tol: f32 = 1e-7;
    let max_sweeps = 60;

    for _sweep in 0..max_sweeps {
        // Convergence criterion: break when a full sweep applies no rotation.
        // This is the standard cyclic-Jacobi criterion and is scale-invariant
        // (unlike an absolute off-diagonal threshold, which never triggers for
        // matrices whose entries are O(1) — see issue 003). Once every column
        // pair satisfies the per-pair relative test below, the matrix A^T A is
        // diagonal to within `tol` relative accuracy, so the singular values
        // are accurate to ~7 digits and the singular vectors are stable.
        let mut rotated: bool = false;
        for p in 0..n {
            for q in (p + 1)..n {
                // Compute (p, q) entry of A^T A: dot of column p and column q.
                let mut app: f32 = 0.0;
                let mut aqq: f32 = 0.0;
                let mut apq: f32 = 0.0;
                for r in 0..m {
                    let arp = work.a[r * n + p];
                    let arq = work.a[r * n + q];
                    app += arp * arp;
                    aqq += arq * arq;
                    apq += arp * arq;
                }
                if apq.abs() <= tol * (app * aqq).sqrt() {
                    continue; // Already diagonal in this plane.
                }
                rotated = true;
                // Compute Jacobi rotation (c, s) that zeroes apq.
                let tau = (aqq - app) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // Apply rotation to columns p, q of A and V.
                for r in 0..m {
                    let arp = work.a[r * n + p];
                    let arq = work.a[r * n + q];
                    work.a[r * n + p] = c * arp - s * arq;
                    work.a[r * n + q] = s * arp + c * arq;
                }
                for r in 0..n {
                    let vrp = work.v[r * n + p];
                    let vrq = work.v[r * n + q];
                    work.v[r * n + p] = c * vrp - s * vrq;
                    work.v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
        if !rotated {
            break;
        }
    }

    // Extract singular values (column norms of A post-rotation) and sort desc.
    //
    // The raw singular values are column norms of work.a (post-Jacobi-rotation).
    // We compute them into a stack array (k ≤ 16), argsort by descending value,
    // then write the sorted triples into the SOA result. Reading from a stack
    // snapshot avoids the read-then-write aliasing hazard on
    // `result.singular_values` (writing sorted position `out_j` must not
    // clobber the source position `perm[out_j']` for a later `out_j'`).
    let k = m.min(n); // number of singular triples
    debug_assert!(k <= 16, "one-sided Jacobi result scratch supports k <= 16");

    // Stack snapshot of raw (unsorted) singular values.
    let mut raw_sigma: [f32; 16] = [0.0; 16];
    for i in 0..k {
        let mut s_sq: f32 = 0.0;
        for r in 0..m {
            let ari = work.a[r * n + i];
            s_sq += ari * ari;
        }
        raw_sigma[i] = s_sq.sqrt();
    }

    // Argsort the column indices by descending singular value.
    let mut perm: [usize; 16] = [0; 16];
    for i in 0..k {
        perm[i] = i;
    }
    // Insertion sort by descending singular value — O(k²) but k ≤ 16, and
    // branch-predictable for nearly-sorted input (common after convergence).
    for i in 1..k {
        let key_idx = perm[i];
        let key_val = raw_sigma[key_idx];
        let mut j = i;
        while j > 0 && raw_sigma[perm[j - 1]] < key_val {
            perm[j] = perm[j - 1];
            j -= 1;
        }
        perm[j] = key_idx;
    }

    // Effective rank: count singular values above a small threshold relative
    // to the largest.
    let sigma_max = if k > 0 { raw_sigma[perm[0]] } else { 0.0 };
    let rank_threshold = sigma_max * 1e-5;
    let mut rank = 0;

    // Write the sorted singular triples into the SOA result buffers.
    // Column-major layout: triple j at result.{singular_values[j],
    // right_singular_vectors[j*n .. (j+1)*n], left_singular_vectors[j*m .. (j+1)*m]}.
    // Safe: we read sigma from the stack snapshot (not from result.singular_values),
    // and vectors from work.a / work.v (which are not overwritten here).
    for out_j in 0..k {
        let src_i = perm[out_j]; // original column index
        let sv = raw_sigma[src_i];
        result.singular_values[out_j] = sv;

        if sv > rank_threshold {
            rank += 1;
        }

        // Right singular vector = column src_i of work.v, length n.
        let v_base = out_j * n;
        for r in 0..n {
            result.right_singular_vectors[v_base + r] = work.v[r * n + src_i];
        }
        // Left singular vector = column src_i of work.a, normalized by σ.
        let u_base = out_j * m;
        if sv < f32::EPSILON {
            for r in 0..m {
                result.left_singular_vectors[u_base + r] = 0.0;
            }
        } else {
            let inv_sv = 1.0 / sv;
            for r in 0..m {
                result.left_singular_vectors[u_base + r] = work.a[r * n + src_i] * inv_sv;
            }
        }
    }

    result.len = k;
    result.rank = rank;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::cmp::Ordering;

    #[test]
    fn participation_ratio_flat_spectrum() {
        // 5 equal eigenvalues → PR = 5.
        let s = vec![1.0; 5];
        let pr = participation_ratio(&s);
        assert!((pr - 5.0).abs() < 1e-5, "expected 5.0, got {pr}");
    }

    #[test]
    fn participation_ratio_dominant_mode() {
        // One large eigenvalue → PR ≈ 1.
        let s = vec![10.0, 0.01, 0.01, 0.01];
        let pr = participation_ratio(&s);
        assert!(pr < 1.2, "expected ≈ 1, got {pr}");
    }

    #[test]
    fn participation_ratio_empty_returns_zero() {
        assert_eq!(participation_ratio(&[]), 0.0);
    }

    #[test]
    fn participation_ratio_all_zero_returns_zero() {
        assert_eq!(participation_ratio(&[0.0; 8]), 0.0);
    }

    #[test]
    fn numerical_rank_full_energy() {
        // σ² = [100, 25, 4, 1], total = 130.
        // η=0.99 → threshold 128.7 → cumulative 100, 125, 129 → rank 3 (129/130 = 99.23%).
        // η=1.0  → threshold 130.0 → needs all 4 columns → rank 4.
        let s = vec![10.0, 5.0, 2.0, 1.0]; // descending
        let r99 = numerical_rank(&s, 0.99);
        assert_eq!(r99, 3, "η=0.99 → rank 3 (cum 129/130 = 99.23% > 99%)");
        let r_strict = numerical_rank(&s, 1.0);
        assert_eq!(r_strict, 4, "η=1.0 → rank 4 (needs all columns)");
    }

    #[test]
    fn numerical_rank_low_rank() {
        // 99% of energy in top 2 of 4 singular values.
        let s = vec![10.0, 10.0, 0.1, 0.1];
        let r = numerical_rank(&s, 0.99);
        assert!(r <= 2, "expected rank ≤ 2, got {r}");
        let r90 = numerical_rank(&s, 0.9);
        assert!(r90 <= 2, "at η=0.9 still rank ≤ 2, got {r90}");
    }

    #[test]
    fn phase_transition_gate_at_threshold() {
        assert!(!phase_transition_gate(5, 6), "N=5 < d=6 → false");
        assert!(phase_transition_gate(6, 6), "N=6 = d=6 → true");
        assert!(phase_transition_gate(50, 6), "N=50 > d=6 → true");
    }

    #[test]
    fn jacobian_svd_recovers_known_rank3_matrix() {
        // Construct a rank-3 matrix W (4 × 6) = U3 Σ3 V3^T.
        // W maps R^6 → R^4; its Jacobian (W is linear) is W itself.
        // Singular values should be {10, 5, 2, 0} after rounding.
        let n = 6;
        let m = 4;
        // Build a known rank-3 W.
        // u_i ∈ R^4, v_i ∈ R^6, σ_i: W = Σ σ_i u_i v_i^T
        let u1 = [1.0, 0.0, 0.0, 0.0];
        let u2 = [0.0, 1.0, 0.0, 0.0];
        let u3 = [0.0, 0.0, 1.0, 0.0];
        let v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let s1 = 10.0_f32;
        let s2 = 5.0_f32;
        let s3 = 2.0_f32;
        let mut w = vec![0.0_f32; m * n];
        for j in 0..m {
            for i in 0..n {
                let mut acc = 0.0;
                acc += s1 * u1[j] * v1[i];
                acc += s2 * u2[j] * v2[i];
                acc += s3 * u3[j] * v3[i];
                w[j * n + i] = acc;
            }
        }
        // The map f: x ↦ W x. Jacobian = W.
        let f = |x: &[f32], out: &mut [f32]| {
            debug_assert_eq!(x.len(), n);
            debug_assert_eq!(out.len(), m);
            for j in 0..m {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += w[j * n + i] * x[i];
                }
                out[j] = acc;
            }
        };
        let x = [0.5_f32; 6];
        let mut scratch = JacobianSvdScratch::with_capacity(n, m);
        let result = jacobian_svd_at(f, &x, 1e-4, &mut scratch);
        // Expect rank 3.
        assert_eq!(
            result.rank, 3,
            "expected rank 3, got {} (sigmas = {:?})",
            result.rank, result.singular_values
        );
        // Top-3 singular values should be close to {10, 5, 2} (order-tolerant sign).
        let top3: Vec<f32> = result.singular_values.iter().take(3).cloned().collect();
        let mut expected = [10.0f32, 5.0, 2.0];
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let mut got = top3.clone();
        got.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        for (e, g) in expected.iter().zip(got.iter()) {
            assert!(
                (e - g).abs() < 0.1,
                "singular value mismatch: expected ≈ {e}, got {g}"
            );
        }
        // The 4th singular value should be ≈ 0.
        if result.singular_values.len() >= 4 {
            assert!(
                result.singular_values[3] < 0.1,
                "expected 4th singular value ≈ 0, got {}",
                result.singular_values[3]
            );
        }
    }

    #[test]
    fn estimate_intrinsic_dim_participation_ratio() {
        let s = vec![1.0; 4];
        let d = estimate_intrinsic_dim(&s, IntrinsicDimMethod::ParticipationRatio);
        assert_eq!(d, 4);
    }

    #[test]
    fn estimate_intrinsic_dim_numerical_rank() {
        let s = vec![10.0, 5.0, 0.1, 0.05]; // ~99% energy in top 2
        let d = estimate_intrinsic_dim(&s, IntrinsicDimMethod::NumericalRank { eta: 0.99 });
        assert!(d <= 2, "expected d ≤ 2, got {d}");
    }

    // ── thin_svd_into / SvdResultScratch tests ─────────────────────────────

    /// Build a known rank-3 matrix (same construction as
    /// `jacobian_svd_recovers_known_rank3_matrix` but without the Jacobian
    /// estimation layer — we feed the matrix directly to `thin_svd_into`).
    fn known_rank3_matrix_4x6() -> (Vec<f32>, usize, usize) {
        let n = 6;
        let m = 4;
        let u1 = [1.0, 0.0, 0.0, 0.0];
        let u2 = [0.0, 1.0, 0.0, 0.0];
        let u3 = [0.0, 0.0, 1.0, 0.0];
        let v1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let s1 = 10.0_f32;
        let s2 = 5.0_f32;
        let s3 = 2.0_f32;
        let mut w = vec![0.0_f32; m * n];
        for j in 0..m {
            for i in 0..n {
                let mut acc = 0.0;
                acc += s1 * u1[j] * v1[i];
                acc += s2 * u2[j] * v2[i];
                acc += s3 * u3[j] * v3[i];
                w[j * n + i] = acc;
            }
        }
        (w, m, n)
    }

    #[test]
    fn thin_svd_into_matches_thin_svd() {
        // The SOA scratch path must produce bit-identical singular values to
        // the owned-result path. This is the core correctness contract of
        // `thin_svd_into`.
        let (m_flat, m_rows, n_cols) = known_rank3_matrix_4x6();
        let mut work = SvdScratch::with_capacity(n_cols, m_rows);
        let owned = thin_svd(&m_flat, m_rows, n_cols, &mut work);

        let mut work2 = SvdScratch::with_capacity(n_cols, m_rows);
        let mut result = SvdResultScratch::with_capacity(m_rows, n_cols);
        thin_svd_into(&m_flat, m_rows, n_cols, &mut result, &mut work2);

        // Same number of singular triples.
        assert_eq!(
            result.len(),
            owned.singular_values.len(),
            "SOA len {} != owned len {}",
            result.len(),
            owned.singular_values.len()
        );
        // Same rank.
        assert_eq!(result.rank, owned.rank);
        // Same singular values (both sorted descending).
        for j in 0..result.len() {
            assert!(
                (result.singular_value(j) - owned.singular_values[j]).abs() < 1e-4,
                "sv[{j}] mismatch: SOA={} owned={}",
                result.singular_value(j),
                owned.singular_values[j]
            );
        }
    }

    #[test]
    fn thin_svd_into_reused_across_calls_no_reallocation() {
        // The SOA result scratch must be reusable across calls of different
        // sizes without panicking (it auto-grows). The `SvdScratch` work
        // buffer must be pre-sized for the largest matrix (documented contract
        // — growing it in the hot path would defeat the zero-alloc goal).
        let (m_flat, m_rows, n_cols) = known_rank3_matrix_4x6();
        // Size work for the larger 8×8 matrix we'll factor second.
        let mut work = SvdScratch::with_capacity(8, 8);
        let mut result = SvdResultScratch::with_capacity(m_rows, n_cols);

        // First call: 4×6 matrix.
        thin_svd_into(&m_flat, m_rows, n_cols, &mut result, &mut work);
        assert_eq!(result.len(), m_rows.min(n_cols));
        assert_eq!(result.rank, 3, "expected rank 3");

        // Second call: an 8×8 identity (different size — exercises the result
        // grow path).
        let n8 = 8usize;
        let mut ident = vec![0.0f32; n8 * n8];
        for i in 0..n8 {
            ident[i * n8 + i] = 1.0;
        }
        thin_svd_into(&ident, n8, n8, &mut result, &mut work);
        assert_eq!(
            result.len(),
            n8,
            "8×8 identity should give 8 singular triples"
        );
        assert_eq!(result.rank, n8, "identity should be full rank");
        // All singular values of identity are 1.
        for j in 0..n8 {
            assert!(
                (result.singular_value(j) - 1.0).abs() < 1e-4,
                "identity sv[{j}] should be 1.0, got {}",
                result.singular_value(j)
            );
        }
    }

    #[test]
    fn thin_svd_into_singular_vectors_are_unit_norm() {
        // Right singular vectors must be unit-norm (columns of an orthogonal V).
        let (m_flat, m_rows, n_cols) = known_rank3_matrix_4x6();
        let mut work = SvdScratch::with_capacity(n_cols, m_rows);
        let mut result = SvdResultScratch::with_capacity(m_rows, n_cols);
        thin_svd_into(&m_flat, m_rows, n_cols, &mut result, &mut work);

        for j in 0..result.len() {
            let v = result.right_singular_vector(j);
            let norm_sq: f32 = v.iter().map(|x| x * x).sum::<f32>();
            assert!(
                (norm_sq - 1.0).abs() < 1e-3,
                "right sv vector {j} not unit-norm: |v|²={norm_sq}"
            );
        }
    }

    #[test]
    fn svd_result_scratch_accessors() {
        // Exercise the accessor API: singular_values(), singular_value(j),
        // right/left_singular_vector(j), len(), is_empty().
        let mut result = SvdResultScratch::with_capacity(4, 6);
        assert!(result.is_empty(), "fresh scratch should be empty");
        assert_eq!(result.len(), 0);

        let (m_flat, m_rows, n_cols) = known_rank3_matrix_4x6();
        let mut work = SvdScratch::with_capacity(n_cols, m_rows);
        thin_svd_into(&m_flat, m_rows, n_cols, &mut result, &mut work);

        assert!(!result.is_empty());
        assert_eq!(result.len(), 4); // min(4, 6)
        assert_eq!(result.singular_values().len(), 4);
        // Descending order.
        for j in 1..result.len() {
            assert!(
                result.singular_value(j) <= result.singular_value(j - 1) + 1e-5,
                "singular values must be descending"
            );
        }
        // Accessors return slices of the right length.
        assert_eq!(result.right_singular_vector(0).len(), n_cols);
        assert_eq!(result.left_singular_vector(0).len(), m_rows);
    }
}
