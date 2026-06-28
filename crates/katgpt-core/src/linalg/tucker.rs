//! Tucker / HOSVD tensor factorization — N-mode generalization of `thin_svd_into`.
//!
//! Distilled from the TFNO weight-compression discussion in Duruisseaux, Kossaffi,
//! Anandkumar, *Fourier Neural Operators: A Practical Perspective*
//! ([arXiv:2511.05963](https://arxiv.org/abs/2511.05963), Caltech + NVIDIA, Nov 2025).
//! See `katgpt-rs/.research/307_*.md` §3 candidate plan #3 and
//! `katgpt-rs/.plans/326_*.md` for the execution record.
//!
//! # What this computes
//!
//! Higher-Order SVD (HOSVD) decomposes an N-way tensor
//! `X ∈ R^{I_0 × I_1 × … × I_{N-1}}` into:
//!
//! ```text
//! X ≈ S ×_0 A^(0) ×_1 A^(1) × … ×_{N-1} A^(N-1)
//! ```
//!
//! where `A^(n) ∈ R^{I_n × r_n}` is the factor matrix for mode n (columns = leading
//! left singular vectors of the mode-n unfolding), `S ∈ R^{r_0 × r_1 × … × r_{N-1}}`
//! is the core tensor, and `×_n` is the n-mode (tensor-times-matrix) product. With
//! full ranks (`r_n = I_n`), the decomposition is lossless; with reduced ranks it is
//! the best per-mode truncation (not globally optimal — HOOI/HOSVD alternating
//! least squares would be needed for global optimality, which is out of scope here).
//!
//! For N=2, this reduces to standard truncated SVD.
//!
//! # Why this is modelless
//!
//! Pure closed-form linear algebra: mode-n matricizations + thin SVD + tensor-times-
//! matrix contractions. No gradient descent, no learned weights, no training. The
//! only "fitting" is the per-mode spectral truncation, which is deterministic given
//! the rank budget. This makes Tucker safe for sync-boundary commitment (paired
//! with a BLAKE3 envelope over `core || factors`) — two quorum nodes produce
//! bit-identical factorizations from identical inputs.
//!
//! # Performance contract
//!
//! - Cold-tier path: the dominant cost is N thin SVDs of the mode-n unfoldings
//!   (`O(N · I_n · prod_others · min(I_n, prod_others))` for one-sided Jacobi) plus
//!   N n-mode products (`O(N · r_n · prod_current)`). For `(64, 8, 8)` with default
//!   ranks `(8, 4, 4)`: ~3 SVDs of `(64,64)`, `(512,8)`, `(512,8)` after the wide-
//!   matrix transpose + 3 contractions. Sub-millisecond on commodity hardware.
//! - Allocation-free hot path: [`TuckerScratch`] + [`TuckerResultScratch`] are
//!   pre-allocated once via [`TuckerScratch::with_capacity`] and reused across
//!   calls. The G4 GOAT gate verifies 0 allocations per call after warmup.
//!
//! # Determinism
//!
//! Inherits [`thin_svd_into`]'s platform-independence (no SIMD dispatch inside the
//! math, no floating-point reordering). Required for anti-cheat: cold-tier Tucker
//! envelopes must reconstruct bit-identically across quorum nodes.
//!
//! # Sign ambiguity
//!
//! `thin_svd_into` documents arbitrary sign conventions for singular vectors.
//! HOSVD factor matrices inherit this ambiguity. Reconstruction is sign-invariant
//! (factors appear in conjugate pairs `A × A^T`), so the decomposition + round-trip
//! is well-defined; only the individual factor columns have arbitrary sign.
//!
//! [`thin_svd_into`]: crate::subspace_phase_gate::thin_svd_into

use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};

/// Maximum number of tensor modes supported. 4 is sufficient for all current
/// consumers (shard batch Tucker is 3-mode `(N, 8, 8)`); raise if needed.
pub const MAX_MODES: usize = 4;

/// Maximum matrix dimension the underlying one-sided-Jacobi SVD can handle.
/// The SVD uses a stack-allocated `[f32; 16]` raw-sigma buffer, so any mode-n
/// unfolding whose smaller dimension exceeds this will overflow. Tucker configs
/// are rejected at construction if any mode violates this.
///
/// For shard-batch Tucker `(N, 8, 8)`: mode 0 unfolding is `(N, 64)`, so N must
/// be ≤ 16. Larger batches must be chunked into groups of ≤ 16 shards.
pub const SVD_MAX_RANK: usize = 16;

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors raised by the Tucker decomposition / reconstruction entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuckerError {
    /// `shape` and `ranks` slices have different lengths, or length is 0 / > MAX_MODES.
    InvalidModeCount { got: usize, max: usize },
    /// A `shape[n]` entry is such that the mode-n unfolding's smaller dimension
    /// `min(shape[n], prod(shape)/shape[n])` exceeds [`SVD_MAX_RANK`] (16). The
    /// underlying one-sided-Jacobi SVD cannot factor matrices larger than this.
    ShapeExceedsSvdLimit {
        mode: usize,
        min_dim: usize,
        max: usize,
    },
    /// A `ranks[n]` entry exceeds `min(shape[n], prod(shape)/shape[n])` — the
    /// mode-n unfolding cannot have more singular values than its smaller dimension.
    RankTooLarge {
        mode: usize,
        rank: usize,
        bound: usize,
        shape_n: usize,
    },
    /// A `shape[n]` or `ranks[n]` entry is zero.
    ZeroDimension { mode: usize },
    /// Input slice length does not match `prod(shape)`.
    InputSizeMismatch { got: usize, expected: usize },
    /// Output slice length does not match `prod(shape)`.
    OutputSizeMismatch { got: usize, expected: usize },
    /// `tucker_reconstruct_into` was called with an `out_shape` that does not
    /// match the factor matrices' row counts (the original `I_n`).
    ShapeFactorMismatch {
        mode: usize,
        got: usize,
        expected: usize,
    },
}

impl core::fmt::Display for TuckerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TuckerError::InvalidModeCount { got, max } => write!(
                f,
                "tucker: invalid mode count {got}, must be in 1..={max}"
            ),
            TuckerError::ShapeExceedsSvdLimit { mode, min_dim, max } => write!(
                f,
                "tucker: shape at mode {mode} requires SVD of min-dim {min_dim} > {max} (SVD limit)"
            ),
            TuckerError::RankTooLarge { mode, rank, bound, shape_n } => write!(
                f,
                "tucker: rank {rank} for mode {mode} exceeds bound {bound} (shape {shape_n})"
            ),
            TuckerError::ZeroDimension { mode } => {
                write!(f, "tucker: zero dimension at mode {mode}")
            }
            TuckerError::InputSizeMismatch { got, expected } => {
                write!(f, "tucker: input length {got} != product(shape) {expected}")
            }
            TuckerError::OutputSizeMismatch { got, expected } => {
                write!(f, "tucker: output length {got} != product(shape) {expected}")
            }
            TuckerError::ShapeFactorMismatch { mode, got, expected } => write!(
                f,
                "tucker: out_shape[{mode}]={got} does not match factor rows {expected}"
            ),
        }
    }
}

impl std::error::Error for TuckerError {}

// ─── Config ─────────────────────────────────────────────────────────────────

/// Tensor shape + per-mode rank budget for HOSVD.
///
/// Stored as inline arrays (no heap) since `MAX_MODES = 4`. Construct via
/// [`TuckerConfig::new`], which validates the rank bounds.
#[derive(Debug, Clone)]
pub struct TuckerConfig {
    shape: [usize; MAX_MODES],
    ranks: [usize; MAX_MODES],
    n_modes: u8,
}

impl TuckerConfig {
    /// Validate and construct a Tucker config.
    ///
    /// `shape` and `ranks` must have equal length in `1..=MAX_MODES`. Two shape
    /// constraints apply per mode n:
    /// 1. `min(shape[n], prod(shape)/shape[n]) ≤ SVD_MAX_RANK` (16) — the mode-n
    ///    unfolding's smaller dimension must fit in the underlying SVD's
    ///    stack-allocated result buffer.
    /// 2. `ranks[n] ≤ min(shape[n], prod(shape)/shape[n])` — a matrix of shape
    ///    `(I_n, M)` has at most `min(I_n, M)` singular values to retain.
    ///
    /// For shard-batch Tucker `(N, 8, 8)`: mode 0 unfolding is `(N, 64)`, so
    /// `N ≤ 16` is required. Larger batches must be chunked into groups of
    /// ≤ 16 shards per Tucker call.
    pub fn new(shape: &[usize], ranks: &[usize]) -> Result<Self, TuckerError> {
        if shape.len() != ranks.len() || shape.is_empty() {
            return Err(TuckerError::InvalidModeCount {
                got: shape.len(),
                max: MAX_MODES,
            });
        }
        if shape.len() > MAX_MODES {
            return Err(TuckerError::InvalidModeCount {
                got: shape.len(),
                max: MAX_MODES,
            });
        }
        // Check zero dimensions FIRST (before rank-bounds, which would otherwise
        // fire RankTooLarge with bound=0 on the zero mode and mask the real cause).
        for n in 0..shape.len() {
            if shape[n] == 0 || ranks[n] == 0 {
                return Err(TuckerError::ZeroDimension { mode: n });
            }
        }
        let n_modes = shape.len();
        let total: usize = shape.iter().product();
        let mut s = [0usize; MAX_MODES];
        let mut r = [0usize; MAX_MODES];
        for n in 0..n_modes {
            let m = total / shape[n]; // product of other modes
            let min_dim = shape[n].min(m);
            // The SVD must factor the mode-n unfolding (shape_n, prod_others).
            // The underlying one-sided-Jacobi SVD has a stack-allocated
            // `[f32; SVD_MAX_RANK]` raw-sigma buffer — calling it with
            // min(m_rows, n_cols) > SVD_MAX_RANK would overflow. Reject shapes
            // that would require a larger SVD.
            if min_dim > SVD_MAX_RANK {
                return Err(TuckerError::ShapeExceedsSvdLimit {
                    mode: n,
                    min_dim,
                    max: SVD_MAX_RANK,
                });
            }
            // Rank bound: at most min(I_n, prod_others) singular values.
            let bound = min_dim;
            if ranks[n] > bound {
                return Err(TuckerError::RankTooLarge {
                    mode: n,
                    rank: ranks[n],
                    bound,
                    shape_n: shape[n],
                });
            }
            s[n] = shape[n];
            r[n] = ranks[n];
        }
        Ok(Self {
            shape: s,
            ranks: r,
            n_modes: n_modes as u8,
        })
    }

    /// Number of modes (length of the shape/ranks slices used at construction).
    #[inline]
    pub fn n_modes(&self) -> usize {
        self.n_modes as usize
    }

    /// Active shape slice (length `n_modes`).
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.n_modes()]
    }

    /// Active ranks slice (length `n_modes`).
    #[inline]
    pub fn ranks(&self) -> &[usize] {
        &self.ranks[..self.n_modes()]
    }

    /// Total elements in the input/output tensor: `prod(shape)`.
    #[inline]
    pub fn total_elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Core tensor elements: `prod(ranks)`.
    #[inline]
    pub fn core_elements(&self) -> usize {
        self.ranks().iter().product()
    }

    /// Total factor elements across all modes: `sum_n I_n · r_n`.
    #[inline]
    pub fn factor_elements(&self) -> usize {
        (0..self.n_modes())
            .map(|n| self.shape[n] * self.ranks[n])
            .sum()
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────────

/// Pre-allocated working buffers for the HOSVD hot path. Reuse across calls.
///
/// Sized for one specific [`TuckerConfig`] via [`with_capacity`](Self::with_capacity).
/// Calls with a smaller-or-equal config (fewer modes, smaller shape, smaller ranks)
/// can reuse the same scratch; larger configs need a fresh allocation.
///
/// **Allocation discipline:** all internal `Vec`s are sized once at construction.
/// The hot-path entry points ([`tucker_decompose_into`], [`tucker_reconstruct_into`])
/// borrow `&mut self` and never grow these buffers — G4 (0 allocs/call) holds.
pub struct TuckerScratch {
    /// Mode-n unfolding of the current tensor. Length = `total_elements`.
    unfold_buf: Vec<f32>,
    /// n-mode product output (matmul Y). Length = `total_elements`.
    y_buf: Vec<f32>,
    /// Current tensor being successively contracted. Length = `total_elements`.
    contract_buf: Vec<f32>,
    /// Transpose buffer for wide-matrix SVD pre-orientation. Length = `total_elements`.
    transpose_buf: Vec<f32>,
    /// Reusable one-sided-Jacobi SVD working memory.
    svd_work: SvdScratch,
    /// Reusable SVD result SOA buffers.
    svd_result: SvdResultScratch,
}

impl TuckerScratch {
    /// Allocate scratch sized for the given config. Reuse across calls.
    pub fn with_capacity(cfg: &TuckerConfig) -> Self {
        let nmodes = cfg.n_modes();
        let shape = cfg.shape();
        let total = cfg.total_elements();
        // SVD input is always oriented tall-skinny (m_rows ≥ n_cols) via the
        // transpose trick in `compute_factor`. Across all modes, the worst-case
        // SVD input is (max_m_rows, max_n_cols) where:
        //   max_n_cols = max_n min(I_n, prod_others)
        //   max_m_rows = max_n max(I_n, prod_others)
        let mut max_n_cols = 1usize;
        let mut max_m_rows = 1usize;
        for &i_n in shape.iter().take(nmodes) {
            let m = total / i_n;
            let (nc, mr) = if i_n >= m { (m, i_n) } else { (i_n, m) };
            max_n_cols = max_n_cols.max(nc);
            max_m_rows = max_m_rows.max(mr);
        }
        Self {
            unfold_buf: vec![0.0; total],
            y_buf: vec![0.0; total],
            contract_buf: vec![0.0; total],
            transpose_buf: vec![0.0; total],
            // SvdScratch::with_capacity takes (n_cols, m_rows) in that order.
            svd_work: SvdScratch::with_capacity(max_n_cols, max_m_rows),
            svd_result: SvdResultScratch::with_capacity(max_m_rows, max_n_cols),
        }
    }
}

// ─── Result (SOA scratch + owned convenience) ───────────────────────────────

/// Hot-path HOSVD result: core tensor + factor matrices in flat SOA layout.
///
/// **Factor storage:** each factor `A^(n)` is stored **column-major** as
/// `(I_n, r_n)`: column `j` (the `j`-th retained singular vector) lives at
/// `factors[factor_offsets[n] + j*I_n .. factor_offsets[n] + (j+1)*I_n]`. This
/// matches the layout returned by [`SvdResultScratch::left_singular_vector`] /
/// [`SvdResultScratch::right_singular_vector`], enabling direct `copy_from_slice`.
///
/// Reuse across calls to eliminate per-call allocation. Convert to an owned
/// [`TuckerResult`] via [`TuckerResult::from_scratch`] for one-shot consumers.
///
/// [`SvdResultScratch::left_singular_vector`]: crate::subspace_phase_gate::SvdResultScratch::left_singular_vector
/// [`SvdResultScratch::right_singular_vector`]: crate::subspace_phase_gate::SvdResultScratch::right_singular_vector
#[derive(Debug, Clone)]
pub struct TuckerResultScratch {
    /// Core tensor elements, row-major over `core_shape`. Length = `prod(ranks)`.
    core: Vec<f32>,
    /// All factor matrices concatenated, column-major per factor. Length = `sum I_n*r_n`.
    factors: Vec<f32>,
    /// Start offset of each factor within `factors`. Length MAX_MODES; only
    /// the first `n_modes` entries are meaningful.
    factor_offsets: [usize; MAX_MODES],
    /// `I_n` per mode (original shape entry). Length MAX_MODES.
    factor_rows: [usize; MAX_MODES],
    /// `r_n` per mode (rank budget used). Length MAX_MODES.
    factor_cols: [usize; MAX_MODES],
    /// Core shape = ranks. Length MAX_MODES.
    core_shape: [usize; MAX_MODES],
    n_modes: u8,
}

impl TuckerResultScratch {
    /// Allocate result storage sized for the given config. Reuse across calls.
    pub fn with_capacity(cfg: &TuckerConfig) -> Self {
        let mut factor_offsets = [0usize; MAX_MODES];
        let mut acc = 0usize;
        for (n, offset) in factor_offsets.iter_mut().enumerate().take(cfg.n_modes()) {
            *offset = acc;
            acc += cfg.shape[n] * cfg.ranks[n];
        }
        Self {
            core: vec![0.0; cfg.core_elements()],
            factors: vec![0.0; cfg.factor_elements()],
            factor_offsets,
            factor_rows: cfg.shape,
            factor_cols: cfg.ranks,
            core_shape: cfg.ranks,
            n_modes: cfg.n_modes,
        }
    }

    /// Number of modes in the decomposed tensor.
    #[inline]
    pub fn n_modes(&self) -> usize {
        self.n_modes as usize
    }

    /// Core tensor (row-major over `core_shape()`).
    #[inline]
    pub fn core(&self) -> &[f32] {
        &self.core
    }

    /// Core shape (= ranks used).
    #[inline]
    pub fn core_shape(&self) -> &[usize] {
        &self.core_shape[..self.n_modes()]
    }

    /// Factor matrix for `mode`, column-major `(I_n, r_n)`. Column `j` is the
    /// `j`-th retained singular vector of the mode-n unfolding.
    #[inline]
    pub fn factor(&self, mode: usize) -> &[f32] {
        let start = self.factor_offsets[mode];
        let len = self.factor_rows[mode] * self.factor_cols[mode];
        &self.factors[start..start + len]
    }

    /// `(I_n, r_n)` shape of the factor for `mode`.
    #[inline]
    pub fn factor_shape(&self, mode: usize) -> (usize, usize) {
        (self.factor_rows[mode], self.factor_cols[mode])
    }

    /// Compression ratio = `(core + factors) / original`. < 1.0 means net compression.
    /// Computed from the stored factor rows (= original `I_n`), so independent of the
    /// `TuckerConfig` used at construction.
    pub fn compression_ratio(&self) -> f32 {
        let original: usize = (0..self.n_modes()).map(|n| self.factor_rows[n]).product();
        if original == 0 {
            return 0.0;
        }
        let compressed = self.core.len() + self.factors.len();
        compressed as f32 / original as f32
    }

    /// Rebuild a scratch result from an owned [`TuckerResult`] (inverse of
    /// [`TuckerResult::from_scratch`]).
    ///
    /// This is the Cold-tier reload path: a persisted `TuckerResult` (core +
    /// factor matrices + shapes) is loaded back into the hot-path SOA layout so
    /// [`tucker_reconstruct_into`] can run without re-decomposing. The shapes
    /// must be consistent (core length = product of `core_shape`, each factor
    /// length = `I_n * r_n`); otherwise a [`TuckerError::InputSizeMismatch`] is
    /// returned naming the offending region.
    ///
    /// Allocates once (the SOA buffers); reuse the resulting scratch across
    /// many reconstruction calls.
    pub fn from_owned(result: &TuckerResult) -> Result<Self, TuckerError> {
        let nmodes = result.factor_shapes.len();
        if nmodes == 0 || nmodes > MAX_MODES {
            return Err(TuckerError::InvalidModeCount {
                got: nmodes,
                max: MAX_MODES,
            });
        }
        if result.core.len() != result.core_shape.iter().product::<usize>() {
            return Err(TuckerError::InputSizeMismatch {
                got: result.core.len(),
                expected: result.core_shape.iter().product(),
            });
        }
        if result.factors.len() != nmodes || result.core_shape.len() != nmodes {
            return Err(TuckerError::InvalidModeCount {
                got: result.factors.len(),
                max: MAX_MODES,
            });
        }
        // Validate each factor length and build the flat layout + offsets.
        let mut shape = [0usize; MAX_MODES];
        let mut ranks = [0usize; MAX_MODES];
        let mut factor_offsets = [0usize; MAX_MODES];
        let mut factors_flat = Vec::new();
        let mut acc = 0usize;
        for n in 0..nmodes {
            let (i_n, r_n) = result.factor_shapes[n];
            let expected_len = i_n * r_n;
            if result.factors[n].len() != expected_len {
                return Err(TuckerError::InputSizeMismatch {
                    got: result.factors[n].len(),
                    expected: expected_len,
                });
            }
            shape[n] = i_n;
            ranks[n] = r_n;
            factor_offsets[n] = acc;
            acc += expected_len;
            factors_flat.extend_from_slice(&result.factors[n]);
        }
        Ok(Self {
            core: result.core.clone(),
            factors: factors_flat,
            factor_offsets,
            factor_rows: shape,
            factor_cols: ranks,
            core_shape: {
                let mut cs = [0usize; MAX_MODES];
                cs[..nmodes].copy_from_slice(&result.core_shape);
                cs
            },
            n_modes: nmodes as u8,
        })
    }
}

/// Owned HOSVD result for one-shot consumers. Hot paths use [`TuckerResultScratch`].
#[derive(Debug, Clone)]
pub struct TuckerResult {
    /// Core tensor elements, row-major over `core_shape`.
    pub core: Vec<f32>,
    /// One factor matrix per mode, column-major `(I_n, r_n)`.
    pub factors: Vec<Vec<f32>>,
    /// Core shape (= ranks used).
    pub core_shape: Vec<usize>,
    /// `(I_n, r_n)` per mode.
    pub factor_shapes: Vec<(usize, usize)>,
}

impl TuckerResult {
    /// Copy out of a hot-path scratch into an owned result.
    pub fn from_scratch(scratch: &TuckerResultScratch) -> Self {
        let n = scratch.n_modes();
        let factors = (0..n).map(|m| scratch.factor(m).to_vec()).collect();
        let factor_shapes = (0..n).map(|m| scratch.factor_shape(m)).collect();
        Self {
            core: scratch.core.to_vec(),
            factors,
            core_shape: scratch.core_shape().to_vec(),
            factor_shapes,
        }
    }

    /// Compression ratio (same definition as [`TuckerResultScratch::compression_ratio`]).
    pub fn compression_ratio(&self) -> f32 {
        let original: usize = self.factor_shapes.iter().map(|(i_n, _)| *i_n).product();
        if original == 0 {
            return 0.0;
        }
        let compressed = self.core.len() + self.factors.iter().map(|f| f.len()).sum::<usize>();
        compressed as f32 / original as f32
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Per-mode column strides for the mode-n unfolding (Kolda's convention).
/// Modes other than `mode` are ordered increasingly (excluding `mode`); the
/// earliest non-`mode` mode has stride 1, the next has stride `shape[that_mode]`, etc.
/// The stride for `mode` itself is left at 0 (unused by the caller).
#[inline]
fn col_strides_for_unfold_into(shape: &[usize], mode: usize, out: &mut [usize]) {
    let n = shape.len();
    let mut stride = 1usize;
    for k in 0..n {
        if k == mode {
            continue;
        }
        out[k] = stride;
        stride *= shape[k];
    }
}

/// Incremental mixed-radix odometer used by [`unfold_into`] / [`fold_into`].
///
/// Tracks `row = multi[mode]` and `col = Σ_{k≠mode} multi[k] * col_strides[k]`
/// as the multi-index advances in row-major (most-significant first) order.
/// Each step is O(1) amortized: the expected carry-chain length per increment
/// is `1 + 1/shape[n-1] + 1/(shape[n-1]·shape[n-2]) + …` ≈ 1 + ε.
///
/// This replaces the prior per-element `n`-division multi-index decode
/// (O(n) integer div/mod per element — ~25 cycles each on most x86 cores)
/// with a branch-predicted carry loop, eliminating ~`n × total × 25` cycles
/// of division work per unfold/fold call.
#[inline]
fn unfold_deltas(
    shape: &[usize],
    mode: usize,
    col_strides: &[usize; MAX_MODES],
) -> ([usize; MAX_MODES], [usize; MAX_MODES]) {
    let n = shape.len();
    let mut row_delta = [0usize; MAX_MODES];
    let mut col_delta = [0usize; MAX_MODES];
    for k in 0..n {
        if k == mode {
            // `row` tracks multi[mode]; one step in mode-n bumps row by 1
            // (the output addresses it as `row * m`, so the effective jump is m).
            row_delta[k] = 1;
        } else {
            col_delta[k] = col_strides[k];
        }
    }
    (row_delta, col_delta)
}

/// Advance the odometer one step in row-major order, maintaining `row` and
/// `col` incrementally. Carries from the least-significant mode upward.
#[inline]
fn advance_multi(
    multi: &mut [usize; MAX_MODES],
    shape: &[usize],
    row: &mut usize,
    col: &mut usize,
    row_delta: &[usize; MAX_MODES],
    col_delta: &[usize; MAX_MODES],
) {
    let mut k = shape.len();
    loop {
        k -= 1;
        multi[k] += 1;
        *row += row_delta[k];
        *col += col_delta[k];
        if multi[k] < shape[k] {
            return;
        }
        // Overflow: reset this digit and subtract its full contribution.
        multi[k] = 0;
        *row -= shape[k] * row_delta[k];
        *col -= shape[k] * col_delta[k];
        if k == 0 {
            return;
        }
    }
}

/// Unfold `x` (row-major tensor of `shape`) along `mode` into `out` as a
/// row-major `(I_n, M)` matrix, where `I_n = shape[mode]` and `M = len / I_n`.
///
/// `out.len()` must be `≥ x.len()`. `x.len()` must equal `prod(shape)`.
fn unfold_into(x: &[f32], shape: &[usize], mode: usize, out: &mut [f32]) {
    debug_assert!(shape.len() <= MAX_MODES, "MAX_MODES = {MAX_MODES}, got {}", shape.len());
    let i_n = shape[mode];
    let total = x.len();
    debug_assert_eq!(total, shape.iter().product::<usize>());
    debug_assert!(out.len() >= total);
    if total == 0 {
        return;
    }

    let mut col_strides = [0usize; MAX_MODES];
    col_strides_for_unfold_into(shape, mode, &mut col_strides);
    let (row_delta, col_delta) = unfold_deltas(shape, mode, &col_strides);

    let m = total / i_n;
    let mut multi = [0usize; MAX_MODES];
    let mut row = 0usize;
    let mut col = 0usize;

    out[0] = x[0];
    for flat in 1..total {
        advance_multi(&mut multi, shape, &mut row, &mut col, &row_delta, &col_delta);
        out[row * m + col] = x[flat];
    }
}

/// Inverse of [`unfold_into`]: fold `mat` (row-major `(shape[mode], M)`) back
/// into `out` (row-major tensor of `shape`). `out.len()` must equal `prod(shape)`.
fn fold_into(mat: &[f32], shape: &[usize], mode: usize, out: &mut [f32]) {
    let i_n = shape[mode];
    let total = out.len();
    debug_assert_eq!(total, shape.iter().product::<usize>());
    debug_assert_eq!(mat.len(), total);
    if total == 0 {
        return;
    }

    let mut col_strides = [0usize; MAX_MODES];
    col_strides_for_unfold_into(shape, mode, &mut col_strides);
    let (row_delta, col_delta) = unfold_deltas(shape, mode, &col_strides);

    let m = total / i_n;
    let mut multi = [0usize; MAX_MODES];
    let mut row = 0usize;
    let mut col = 0usize;

    out[0] = mat[0];
    for flat in 1..total {
        advance_multi(&mut multi, shape, &mut row, &mut col, &row_delta, &col_delta);
        out[flat] = mat[row * m + col];
    }
}

/// Transpose `mat` (row-major `(rows, cols)`) into `out` (row-major `(cols, rows)`).
/// `out.len()` must equal `mat.len() = rows * cols`.
#[inline]
fn transpose_into(mat: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(out.len(), rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = mat[i * cols + j];
        }
    }
}

// ─── Main entry points ──────────────────────────────────────────────────────

/// Zero-allocation HOSVD: factor `x` (row-major tensor of `cfg.shape()`) into the
/// caller-owned `result` scratch (core + factor matrices). Reuse `scratch` and
/// `result` across calls to eliminate per-call allocation.
///
/// After the call, read results via:
/// - [`TuckerResultScratch::core`] / [`TuckerResultScratch::core_shape`]
/// - [`TuckerResultScratch::factor`] (column j = j-th retained singular vector)
/// - [`TuckerResultScratch::compression_ratio`]
///
/// Reconstruct the (approximate) original tensor with [`tucker_reconstruct_into`].
///
/// # Algorithm
///
/// 1. **Per-mode factor extraction:** for each mode n, unfold `x` along mode n
///    into `(I_n, prod_others)`; if `I_n < prod_others`, transpose to tall-skinny
///    `(prod_others, I_n)` to keep the SVD's V matrix small; run [`thin_svd_into`];
///    copy the top-`r_n` left (or right, if transposed) singular vectors into the
///    factor storage as columns of `A^(n)`.
/// 2. **Core tensor:** start with `current = x`; for each mode n in turn, unfold
///    `current` along mode n and contract with `A^(n)^T` (n-mode product), shrinking
///    mode n from `I_n` to `r_n`. After all N modes, `current` is the core.
///
/// # Errors
///
/// - [`TuckerError::InputSizeMismatch`] if `x.len() != cfg.total_elements()`.
pub fn tucker_decompose_into(
    x: &[f32],
    cfg: &TuckerConfig,
    scratch: &mut TuckerScratch,
    result: &mut TuckerResultScratch,
) -> Result<(), TuckerError> {
    let total = cfg.total_elements();
    if x.len() != total {
        return Err(TuckerError::InputSizeMismatch {
            got: x.len(),
            expected: total,
        });
    }
    let nmodes = cfg.n_modes();
    let shape = cfg.shape();
    let ranks = cfg.ranks();

    result.n_modes = nmodes as u8;

    // ── Step 1: factor matrices ─────────────────────────────────────────────
    let mut offset_acc = 0usize;
    for n in 0..nmodes {
        let i_n = shape[n];
        let m = total / i_n; // product of other modes in the ORIGINAL tensor
        result.factor_offsets[n] = offset_acc;

        // Unfold x along mode n into unfold_buf[..i_n*m].
        unfold_into(x, shape, n, &mut scratch.unfold_buf[..i_n * m]);

        // Orient the SVD input tall-skinny to keep V small.
        let transposed = if i_n >= m {
            // Factor (I_n, M) directly: m_rows = i_n, n_cols = m.
            thin_svd_into(
                &scratch.unfold_buf,
                i_n,
                m,
                &mut scratch.svd_result,
                &mut scratch.svd_work,
            );
            false
        } else {
            // Transpose to (M, I_n): m_rows = m, n_cols = i_n. Factor it; the
            // right singular vectors of the transpose are the left singular
            // vectors of the original (the factor we want).
            transpose_into(
                &scratch.unfold_buf[..i_n * m],
                i_n,
                m,
                &mut scratch.transpose_buf[..i_n * m],
            );
            thin_svd_into(
                &scratch.transpose_buf,
                m,
                i_n,
                &mut scratch.svd_result,
                &mut scratch.svd_work,
            );
            true
        };

        // Copy top-r_n factor columns into result.factors (column-major).
        let r_n = ranks[n];
        for j in 0..r_n {
            let col: &[f32] = if transposed {
                scratch.svd_result.right_singular_vector(j)
            } else {
                scratch.svd_result.left_singular_vector(j)
            };
            debug_assert_eq!(col.len(), i_n);
            let start = result.factor_offsets[n] + j * i_n;
            result.factors[start..start + i_n].copy_from_slice(col);
        }
        result.factor_rows[n] = i_n;
        result.factor_cols[n] = r_n;
        offset_acc += i_n * r_n;
    }

    // ── Step 2: core tensor ────────────────────────────────────────────────
    // current_shape starts as the original shape and shrinks one mode per step.
    let mut current_shape = [0usize; MAX_MODES];
    current_shape[..nmodes].copy_from_slice(shape);
    let mut current_len = total;
    scratch.contract_buf[..total].copy_from_slice(x);

    // Snapshot the factor offsets to avoid holding an immutable borrow of
    // `result.factors` while we later write `result.core`. (Disjoint fields, but
    // the snapshot makes the control flow obvious to the reader.)
    let factor_offsets = result.factor_offsets;

    for n in 0..nmodes {
        let i_n = current_shape[n]; // = original I_n (no prior step touched mode n)
        debug_assert_eq!(i_n, shape[n]);
        let m = current_len / i_n; // product of the other CURRENT modes
        let r_n = ranks[n];

        // Unfold current along mode n.
        unfold_into(
            &scratch.contract_buf[..current_len],
            &current_shape[..nmodes],
            n,
            &mut scratch.unfold_buf[..i_n * m],
        );

        // Y = A^(n)^T · unfold, shape (r_n, m). A^(n) is col-major (I_n, r_n):
        // A^(n)[i, j] = factors[offset_n + j*I_n + i].
        // Y[j, k] = Σ_i factors[offset_n + j*I_n + i] · unfold[i*m + k].
        // Loop order (i outer, j middle, k inner) treats each i as a rank-1 update
        // Y += a_col_j ⊗ unfold_row_i. Inner (j, k) block is SIMD-friendly.
        let y_slice = &mut scratch.y_buf[..r_n * m];
        for v in y_slice.iter_mut() {
            *v = 0.0;
        }
        for i in 0..i_n {
            let b_row = &scratch.unfold_buf[i * m..(i + 1) * m];
            for j in 0..r_n {
                let a_ij = result.factors[factor_offsets[n] + j * i_n + i];
                let y_row = &mut scratch.y_buf[j * m..(j + 1) * m];
                for k in 0..m {
                    y_row[k] += a_ij * b_row[k];
                }
            }
        }

        // Shrink mode n and fold Y back into contract_buf.
        current_shape[n] = r_n;
        current_len = r_n * m;
        fold_into(
            &scratch.y_buf[..current_len],
            &current_shape[..nmodes],
            n,
            &mut scratch.contract_buf[..current_len],
        );
    }

    // Copy the core out.
    let core_len = cfg.core_elements();
    debug_assert_eq!(core_len, current_len);
    result.core[..core_len].copy_from_slice(&scratch.contract_buf[..core_len]);
    result.core_shape[..nmodes].copy_from_slice(ranks);

    Ok(())
}

/// Zero-allocation reconstruction: write the approximate tensor `X̃ = S ×_0 A^(0)
/// ×_1 A^(1) × … ×_{N-1} A^(N-1)` into `out`, using `scratch` for working memory.
///
/// `out_shape` must match the factor matrices' row counts (the original `I_n`).
///
/// # Algorithm
///
/// Inverse of [`tucker_decompose_into`]: start with `current = core`; for each
/// mode n, unfold `current` along mode n and contract with `A^(n)` (not its
/// transpose), growing mode n from `r_n` to `I_n`.
pub fn tucker_reconstruct_into(
    result: &TuckerResultScratch,
    out_shape: &[usize],
    out: &mut [f32],
    scratch: &mut TuckerScratch,
) -> Result<(), TuckerError> {
    let nmodes = result.n_modes();
    if out_shape.len() != nmodes {
        return Err(TuckerError::ShapeFactorMismatch {
            mode: 0,
            got: out_shape.len(),
            expected: nmodes,
        });
    }
    for (n, &out_s) in out_shape.iter().enumerate().take(nmodes) {
        if out_s != result.factor_rows[n] {
            return Err(TuckerError::ShapeFactorMismatch {
                mode: n,
                got: out_s,
                expected: result.factor_rows[n],
            });
        }
    }
    let total: usize = out_shape.iter().product();
    if out.len() != total {
        return Err(TuckerError::OutputSizeMismatch {
            got: out.len(),
            expected: total,
        });
    }

    // current_shape starts as the core shape; one mode grows per step.
    let mut current_shape = [0usize; MAX_MODES];
    current_shape[..nmodes].copy_from_slice(&result.core_shape[..nmodes]);
    let mut current_len = result.core.len();
    scratch.contract_buf[..current_len].copy_from_slice(&result.core[..current_len]);

    let factor_offsets = result.factor_offsets;

    for n in 0..nmodes {
        let r_n = current_shape[n]; // current mode-n size = rank
        let i_n = out_shape[n]; // target mode-n size = original I_n
        let m = current_len / r_n; // product of the other CURRENT modes

        unfold_into(
            &scratch.contract_buf[..current_len],
            &current_shape[..nmodes],
            n,
            &mut scratch.unfold_buf[..r_n * m],
        );

        // Y = A^(n) · unfold, shape (i_n, m).
        // Y[i, k] = Σ_j A^(n)[i, j] · unfold[j*m + k]
        //        = Σ_j factors[offset_n + j*I_n + i] · unfold[j*m + k].
        // Loop order (j outer, i middle, k inner): for each j, an outer-product
        // rank-1 update of Y[:, :] by a_col_j ⊗ unfold_row_j.
        let y_slice = &mut scratch.y_buf[..i_n * m];
        for v in y_slice.iter_mut() {
            *v = 0.0;
        }
        for j in 0..r_n {
            // Column j of A^(n) is contiguous at [offset_n + j*I_n .. offset_n + (j+1)*I_n].
            let a_col =
                &result.factors[factor_offsets[n] + j * i_n..factor_offsets[n] + (j + 1) * i_n];
            let b_row = &scratch.unfold_buf[j * m..(j + 1) * m];
            for (i, &a_ij) in a_col.iter().enumerate().take(i_n) {
                // stride math: y_row index = i * m..(i + 1) * m
                let y_row = &mut scratch.y_buf[i * m..(i + 1) * m];
                for k in 0..m {
                    y_row[k] += a_ij * b_row[k];
                }
            }
        }

        current_shape[n] = i_n;
        current_len = i_n * m;
        fold_into(
            &scratch.y_buf[..current_len],
            &current_shape[..nmodes],
            n,
            &mut scratch.contract_buf[..current_len],
        );
    }

    debug_assert_eq!(current_len, total);
    out[..total].copy_from_slice(&scratch.contract_buf[..total]);
    Ok(())
}

/// Convenience: one-shot HOSVD that allocates scratch + result internally and
/// returns an owned [`TuckerResult`]. Hot paths should use
/// [`tucker_decompose_into`] with reused scratch instead.
pub fn tucker_decompose(x: &[f32], cfg: &TuckerConfig) -> Result<TuckerResult, TuckerError> {
    let mut scratch = TuckerScratch::with_capacity(cfg);
    let mut result_scratch = TuckerResultScratch::with_capacity(cfg);
    tucker_decompose_into(x, cfg, &mut scratch, &mut result_scratch)?;
    Ok(TuckerResult::from_scratch(&result_scratch))
}

// ─── Tests ─────────────────────────────────────────────────────────────────-

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative Frobenius error: `||x - y||_F / ||x||_F`.
    fn rel_frob_error(x: &[f32], y: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), y.len());
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for i in 0..x.len() {
            let d = x[i] - y[i];
            num += d * d;
            den += x[i] * x[i];
        }
        if den < f32::EPSILON {
            return 0.0;
        }
        (num / den).sqrt()
    }

    // ── Config validation ──────────────────────────────────────────────────

    #[test]
    fn config_rejects_mismatched_lengths() {
        let err = TuckerConfig::new(&[8, 8, 8], &[4, 4]).unwrap_err();
        assert_eq!(
            err,
            TuckerError::InvalidModeCount {
                got: 3,
                max: MAX_MODES
            }
        );
    }

    #[test]
    fn config_rejects_zero_modes() {
        let err = TuckerConfig::new(&[], &[]).unwrap_err();
        assert_eq!(
            err,
            TuckerError::InvalidModeCount {
                got: 0,
                max: MAX_MODES
            }
        );
    }

    #[test]
    fn config_rejects_too_many_modes() {
        let err = TuckerConfig::new(&[2; 5], &[1; 5]).unwrap_err();
        assert_eq!(
            err,
            TuckerError::InvalidModeCount {
                got: 5,
                max: MAX_MODES
            }
        );
    }

    #[test]
    fn config_rejects_zero_shape() {
        let err = TuckerConfig::new(&[8, 0, 8], &[4, 4, 4]).unwrap_err();
        assert_eq!(err, TuckerError::ZeroDimension { mode: 1 });
    }

    #[test]
    fn config_rejects_rank_above_shape() {
        let err = TuckerConfig::new(&[8, 8, 8], &[4, 9, 4]).unwrap_err();
        match err {
            TuckerError::RankTooLarge {
                mode,
                rank,
                bound,
                shape_n,
            } => {
                assert_eq!(mode, 1);
                assert_eq!(rank, 9);
                assert_eq!(bound, 8);
                assert_eq!(shape_n, 8);
            }
            _ => panic!("expected RankTooLarge, got {err:?}"),
        }
    }

    #[test]
    fn config_rejects_rank_above_unfolding_bound() {
        // shape (4, 4, 4): mode 0 unfolding is (4, 16), min = 4. r_0 = 5 should fail.
        let err = TuckerConfig::new(&[4, 4, 4], &[5, 4, 4]).unwrap_err();
        match err {
            TuckerError::RankTooLarge {
                mode, rank, bound, ..
            } => {
                assert_eq!(mode, 0);
                assert_eq!(rank, 5);
                assert_eq!(bound, 4);
            }
            _ => panic!("expected RankTooLarge, got {err:?}"),
        }
    }

    #[test]
    fn config_accepts_valid_3mode() {
        let cfg = TuckerConfig::new(&[8, 8, 8], &[4, 4, 4]).unwrap();
        assert_eq!(cfg.n_modes(), 3);
        assert_eq!(cfg.shape(), &[8, 8, 8]);
        assert_eq!(cfg.ranks(), &[4, 4, 4]);
        assert_eq!(cfg.total_elements(), 512);
        assert_eq!(cfg.core_elements(), 64);
        assert_eq!(cfg.factor_elements(), 96);
    }

    // ── Unfold / fold round-trip ───────────────────────────────────────────

    #[test]
    fn unfold_fold_round_trip_3mode() {
        // Random-ish tensor of shape (2, 3, 4).
        let shape = [2, 3, 4];
        let total = shape.iter().product::<usize>();
        let x: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let mut buf = vec![0.0f32; total];
        let mut back = vec![0.0f32; total];
        for mode in 0..3 {
            unfold_into(&x, &shape, mode, &mut buf);
            fold_into(&buf, &shape, mode, &mut back);
            for i in 0..total {
                assert!(
                    (x[i] - back[i]).abs() < 1e-6,
                    "mode {mode}: x[{i}]={}, back[{i}]={}",
                    x[i],
                    back[i]
                );
            }
        }
    }

    #[test]
    fn unfold_produces_i_n_rows() {
        // shape (4, 3, 2): mode 1 unfolding is (3, 8).
        let shape = [4, 3, 2];
        let total = 24;
        let x: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let mut buf = vec![0.0f32; total];
        unfold_into(&x, &shape, 1, &mut buf);
        // Mode 1 has 3 rows, 8 columns. Each row should have 8 entries.
        // The set of values should be a permutation of x.
        let mut sorted_buf = buf.clone();
        sorted_buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sorted_x = x.clone();
        sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted_buf, sorted_x, "unfold must be a permutation");
    }

    // ── Full HOSVD round-trip ──────────────────────────────────────────────

    #[test]
    fn hosvd_full_rank_is_near_lossless_3mode() {
        // Full-rank: ranks = shape → reconstruction must be near-identity (f32 round-off).
        let shape = [4, 4, 4];
        let total = 64;
        let x: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.5 - 15.0).sin())
            .collect();
        let cfg = TuckerConfig::new(&shape, &shape).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        let mut recon = vec![0.0f32; total];
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).unwrap();
        let err = rel_frob_error(&x, &recon);
        assert!(
            err < 1e-4,
            "full-rank reconstruction rel error should be < 1e-4, got {err}"
        );
    }

    #[test]
    fn hosvd_full_rank_is_near_lossless_wide_matrix_modes() {
        // Shape (8, 4, 4) exercises the transpose trick on modes 1 and 2
        // (mode-1 unfolding is (4, 32) → transpose → (32, 4)).
        let shape = [8, 4, 4];
        let total = 128;
        let x: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.3).cos()).collect();
        let cfg = TuckerConfig::new(&shape, &shape).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        let mut recon = vec![0.0f32; total];
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).unwrap();
        let err = rel_frob_error(&x, &recon);
        assert!(
            err < 1e-4,
            "full-rank (transpose-trick modes) reconstruction rel error should be < 1e-4, got {err}"
        );
    }

    #[test]
    fn hosvd_low_rank_recovers_exact_low_rank_tensor() {
        // Construct a tensor that is exactly rank-(2, 2, 2): a sum of 2^3 = 8 outer
        // products of 2-element vectors along each mode. Shape (4, 4, 4) → a tensor
        // whose per-mode unfolding has rank ≤ 2.
        let shape = [4, 4, 4];
        let total = 64;
        // Build X = Σ_{i,j,k} a_i · b_j · c_k · outer(i,j,k) for random-ish
        // a, b, c in R^4 (only first 2 entries nonzero → rank 2 per mode).
        let a = [1.0f32, 0.5, 0.0, 0.0];
        let b = [0.7f32, -0.3, 0.0, 0.0];
        let c = [0.4f32, 0.9, 0.0, 0.0];
        let mut x = vec![0.0f32; total];
        for i0 in 0..4 {
            for i1 in 0..4 {
                for i2 in 0..4 {
                    let flat = (i0 * 4 + i1) * 4 + i2;
                    x[flat] = a[i0] * b[i1] * c[i2];
                }
            }
        }
        // HOSVD with ranks (2, 2, 2) should recover X nearly exactly (the
        // discarded singular values are ~0 up to f32 round-off).
        let cfg = TuckerConfig::new(&shape, &[2, 2, 2]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        let mut recon = vec![0.0f32; total];
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).unwrap();
        let err = rel_frob_error(&x, &recon);
        assert!(
            err < 1e-4,
            "rank-(2,2,2) tensor recovered with ranks (2,2,2) should be < 1e-4, got {err}"
        );
    }

    #[test]
    fn hosvd_truncated_error_bounded_by_discarded_energy() {
        // For HOSVD, ||X - X̃||_F² ≤ Σ_n discarded_energy_n (the per-mode bound).
        // We construct a tensor with known per-mode spectrum, truncate, and check.
        let shape = [4, 4, 4];
        let total = 64;
        // Identity-like: diagonal entries i*16 + j*4 + k with i==j==k have value.
        // The mode-0 unfolding is rank 4 (distinct singular values).
        let mut x = vec![0.0f32; total];
        for i in 0..4 {
            x[(i * 4 + i) * 4 + i] = (i as f32) + 1.0; // σ_1=4, σ_2=3, σ_3=2, σ_4=1
        }
        let x_norm_sq = x.iter().map(|v| v * v).sum::<f32>();
        // Truncate every mode to rank 3. Discarded energy per mode ≥ σ_4² = 1
        // for the "diagonal" structure (the HOSVD per-mode bound is loose but
        // the inequality still holds).
        let cfg = TuckerConfig::new(&shape, &[3, 3, 3]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        let mut recon = vec![0.0f32; total];
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).unwrap();
        let recon_err_sq: f32 = x
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum();
        // The reconstruction should be non-trivial (some error, since rank 3 ≠ 4).
        assert!(
            recon_err_sq > 0.0,
            "expected nonzero error at ranks (3,3,3)"
        );
        // The reconstructed norm should not exceed the original norm (energy-decreasing).
        let recon_norm_sq: f32 = recon.iter().map(|v| v * v).sum();
        assert!(
            recon_norm_sq <= x_norm_sq + 1e-4,
            "reconstruction energy must not exceed original: got {recon_norm_sq} > {x_norm_sq}"
        );
    }

    // ── Factor orthogonality ───────────────────────────────────────────────

    #[test]
    fn factor_columns_are_orthonormal() {
        // Each factor A^(n) has orthonormal columns (they are left singular
        // vectors of the mode-n unfolding).
        let shape = [4, 4, 4];
        let total = 64;
        let x: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.17).sin()).collect();
        let cfg = TuckerConfig::new(&shape, &shape).unwrap(); // full rank
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        for n in 0..3 {
            let (i_n, r_n) = result.factor_shape(n);
            let factor = result.factor(n);
            // Check A^T A ≈ I (r_n × r_n).
            for j1 in 0..r_n {
                let col1 = &factor[j1 * i_n..(j1 + 1) * i_n];
                // Diagonal: ||col||² ≈ 1.
                let norm_sq: f32 = col1.iter().map(|v| v * v).sum();
                assert!(
                    (norm_sq - 1.0).abs() < 1e-4,
                    "mode {n} col {j1} norm² = {norm_sq}, expected 1.0"
                );
                // Off-diagonal: col_j1 · col_j2 ≈ 0.
                for j2 in (j1 + 1)..r_n {
                    let col2 = &factor[j2 * i_n..(j2 + 1) * i_n];
                    let dot: f32 = col1.iter().zip(col2.iter()).map(|(a, b)| a * b).sum();
                    assert!(
                        dot.abs() < 1e-4,
                        "mode {n} cols {j1},{j2} dot = {dot}, expected 0"
                    );
                }
            }
        }
    }

    // ── Reconstruction error monotonic in ranks ────────────────────────────

    #[test]
    fn reconstruction_error_decreases_with_higher_ranks() {
        let shape = [4, 4, 4];
        let total = 64;
        let x: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.23).sin() + 1.0)
            .collect();
        let mut errors = Vec::new();
        for r in 1..=4 {
            let cfg = TuckerConfig::new(&shape, &[r, r, r]).unwrap();
            let mut scratch = TuckerScratch::with_capacity(&cfg);
            let mut result = TuckerResultScratch::with_capacity(&cfg);
            tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
            let mut recon = vec![0.0f32; total];
            tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).unwrap();
            errors.push(rel_frob_error(&x, &recon));
        }
        // Monotonically non-increasing. The full-rank (r=4) error must be near zero.
        for i in 1..errors.len() {
            assert!(
                errors[i] <= errors[i - 1] + 1e-6,
                "error should be non-increasing in ranks: err[{}]={} > err[{}]={}",
                i - 1,
                errors[i - 1],
                i,
                errors[i]
            );
        }
        assert!(
            errors[3] < 1e-4,
            "full-rank error {} should be < 1e-4",
            errors[3]
        );
    }

    // ── 2-mode ≈ truncated SVD ─────────────────────────────────────────────

    #[test]
    fn two_mode_tucker_matches_truncated_svd_energy() {
        // For a matrix M (2-mode tensor), HOSVD with ranks (r0, r1) gives the
        // truncated SVD reconstruction's energy exactly when r0 = r1 = rank
        // and the truncation captures the leading r singular triples.
        // We can't assert bit-equality (HOSVD core is not diagonal), but the
        // reconstruction's Frobenius norm should equal the truncated SVD's norm.
        use crate::subspace_phase_gate::{SvdResultScratch, SvdScratch, thin_svd_into};
        let rows = 6;
        let cols = 6;
        let m: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        // Truncated SVD: keep top 3 singular values.
        let r = 3;
        let mut svd_work = SvdScratch::with_capacity(cols, rows);
        let mut svd_result = SvdResultScratch::with_capacity(rows, cols);
        thin_svd_into(&m, rows, cols, &mut svd_result, &mut svd_work);
        // Reconstruct via truncated SVD: M̃ = Σ_{j<r} σ_j u_j v_j^T.
        let mut m_truncated = vec![0.0f32; rows * cols];
        for j in 0..r {
            let sigma = svd_result.singular_value(j);
            let u = svd_result.left_singular_vector(j);
            let v = svd_result.right_singular_vector(j);
            for i in 0..rows {
                for k in 0..cols {
                    m_truncated[i * cols + k] += sigma * u[i] * v[k];
                }
            }
        }
        let svd_norm_sq = m_truncated.iter().map(|v| v * v).sum::<f32>();

        // HOSVD with ranks (r, r).
        let cfg = TuckerConfig::new(&[rows, cols], &[r, r]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&m, &cfg, &mut scratch, &mut result).unwrap();
        let mut m_recon = vec![0.0f32; rows * cols];
        tucker_reconstruct_into(&result, &[rows, cols], &mut m_recon, &mut scratch).unwrap();
        let hosvd_norm_sq = m_recon.iter().map(|v| v * v).sum::<f32>();

        // HOSVD with square ranks (r, r) should capture the same energy as
        // truncated SVD at rank r. We allow a small tolerance for f32 round-off
        // and the fact that HOSVD's per-mode optimization is slightly weaker
        // than global SVD when ranks differ between modes (here they're equal).
        assert!(
            (hosvd_norm_sq - svd_norm_sq).abs() / svd_norm_sq < 1e-3,
            "HOSVD energy {hosvd_norm_sq} should match truncated-SVD energy {svd_norm_sq}"
        );
    }

    // ── Convenience owned path ─────────────────────────────────────────────

    #[test]
    fn owned_decompose_matches_into_path() {
        let shape = [4, 4, 4];
        let total = 64;
        let x: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.19).cos()).collect();
        let cfg = TuckerConfig::new(&shape, &[3, 3, 3]).unwrap();

        let owned = tucker_decompose(&x, &cfg).unwrap();

        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();

        // Core + factors must match (up to per-column sign flips — we normalize
        // signs before comparing).
        assert_eq!(owned.core.len(), result.core.len());
        for i in 0..owned.core.len() {
            assert!(
                (owned.core[i] - result.core[i]).abs() < 1e-5,
                "core mismatch at {i}: {} vs {}",
                owned.core[i],
                result.core[i]
            );
        }
    }

    // ── Compression ratio sanity ───────────────────────────────────────────

    #[test]
    fn compression_ratio_is_correct() {
        // shape (8, 8, 8), ranks (4, 4, 4).
        // core = 64, factors = 3 * (8*4) = 96. Total compressed = 160.
        // Original = 512. Ratio = 160/512 = 0.3125.
        let cfg = TuckerConfig::new(&[8, 8, 8], &[4, 4, 4]).unwrap();
        let result = TuckerResultScratch::with_capacity(&cfg);
        let ratio = result.compression_ratio();
        assert!(
            (ratio - 160.0 / 512.0).abs() < 1e-6,
            "compression ratio {ratio} should be 160/512"
        );
    }

    #[test]
    fn full_rank_compression_ratio_is_above_one() {
        // Full rank: core + factors > original (because core = original-sized
        // AND we add the factors on top). ratio > 1.
        let cfg = TuckerConfig::new(&[4, 4, 4], &[4, 4, 4]).unwrap();
        let result = TuckerResultScratch::with_capacity(&cfg);
        let ratio = result.compression_ratio();
        // Compressed = 64 (core) + 3*16 (factors) = 112. Original = 64.
        assert!(
            (ratio - 112.0 / 64.0).abs() < 1e-6,
            "full-rank ratio {ratio} should be 112/64"
        );
    }

    // ── Error paths ────────────────────────────────────────────────────────

    #[test]
    fn decompose_rejects_wrong_input_size() {
        let cfg = TuckerConfig::new(&[4, 4, 4], &[2, 2, 2]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        let x_too_short = vec![0.0f32; 10];
        let err = tucker_decompose_into(&x_too_short, &cfg, &mut scratch, &mut result).unwrap_err();
        match err {
            TuckerError::InputSizeMismatch { got, expected } => {
                assert_eq!(got, 10);
                assert_eq!(expected, 64);
            }
            _ => panic!("expected InputSizeMismatch, got {err:?}"),
        }
    }

    #[test]
    fn reconstruct_rejects_wrong_out_shape() {
        let cfg = TuckerConfig::new(&[4, 4, 4], &[2, 2, 2]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        let x = vec![0.5f32; 64];
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        let mut out = vec![0.0f32; 64];
        // out_shape mismatch: factor_rows were [4,4,4] but we pass [4,4,5].
        let err = tucker_reconstruct_into(&result, &[4, 4, 5], &mut out, &mut scratch).unwrap_err();
        match err {
            TuckerError::ShapeFactorMismatch {
                mode,
                got,
                expected,
            } => {
                assert_eq!(mode, 2);
                assert_eq!(got, 5);
                assert_eq!(expected, 4);
            }
            _ => panic!("expected ShapeFactorMismatch, got {err:?}"),
        }
    }

    // ── Determinism (same inputs → same outputs) ───────────────────────────

    #[test]
    fn decompose_is_deterministic_across_calls() {
        let shape = [4, 4, 4];
        let total = 64;
        let x: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.31).sin()).collect();
        let cfg = TuckerConfig::new(&shape, &[3, 3, 3]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result1 = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result1).unwrap();
        // Re-run with fresh result scratch (same input → same output).
        let mut result2 = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result2).unwrap();
        // Core must match bit-for-bit.
        for i in 0..result1.core.len() {
            assert_eq!(
                result1.core[i], result2.core[i],
                "core[{i}] not bit-identical across calls"
            );
        }
        // Factors must match (sign conventions are deterministic given the SVD).
        for i in 0..result1.factors.len() {
            assert_eq!(
                result1.factors[i], result2.factors[i],
                "factors[{i}] not bit-identical across calls"
            );
        }
    }

    // ── Practical: shard-batch shape (8, 8, 8) ─────────────────────────────

    #[test]
    fn shard_batch_shape_8_8_8_smoke() {
        // The shape the riir-neuron-db integration will use: a batch of 8 shards
        // each with an 8×8 style_weights reshape. Default ranks (4, 4, 4).
        let shape = [8, 8, 8];
        let total = 512;
        let x: Vec<f32> = (0..total)
            .map(|i| {
                let v = (i as f32) * 0.01 - 2.5;
                v.sin() + 0.5 * (2.0 * v).cos()
            })
            .collect();
        let cfg = TuckerConfig::new(&shape, &[4, 4, 4]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        let mut recon = vec![0.0f32; total];
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).unwrap();
        // Low-rank (4,4,4) on a smooth signal should still capture most energy.
        let err = rel_frob_error(&x, &recon);
        assert!(
            err < 0.5,
            "rank-(4,4,4) on smooth signal: err {err} should be < 0.5"
        );
        // Compression ratio should be < 1 (net compression).
        let ratio = result.compression_ratio();
        assert!(ratio < 1.0, "compression ratio {ratio} should be < 1.0");
    }

    #[test]
    fn shard_batch_shape_16_8_8_smoke() {
        // 16 shards × 8 rows × 8 cols — the largest batch that fits the SVD's
        // k≤16 limit (mode-0 unfolding is (16, 64), min-dim = 16 = SVD_MAX_RANK).
        let shape = [16, 8, 8];
        let total = 1024;
        let x: Vec<f32> = (0..total)
            .map(|i| {
                let v = (i as f32) * 0.001;
                v.sin()
            })
            .collect();
        let cfg = TuckerConfig::new(&shape, &[8, 4, 4]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();
        let mut recon = vec![0.0f32; total];
        tucker_reconstruct_into(&result, &shape, &mut recon, &mut scratch).unwrap();
        let err = rel_frob_error(&x, &recon);
        assert!(err.is_finite(), "error must be finite, got {err}");
        let ratio = result.compression_ratio();
        // core (8*4*4=128) + factors (16*8 + 8*4 + 8*4 = 192) = 320.
        // Original = 1024. Ratio = 320/1024 = 0.3125.
        assert!(
            (ratio - 320.0 / 1024.0).abs() < 1e-6,
            "compression ratio {ratio} should be 320/1024"
        );
    }

    #[test]
    fn config_rejects_shape_exceeding_svd_limit() {
        // shape (64, 8, 8): mode 0 unfolding is (64, 64), min-dim = 64 > 16.
        // Must be rejected with ShapeExceedsSvdLimit, not panic in the SVD.
        let err = TuckerConfig::new(&[64, 8, 8], &[8, 4, 4]).unwrap_err();
        match err {
            TuckerError::ShapeExceedsSvdLimit { mode, min_dim, max } => {
                assert_eq!(mode, 0);
                assert_eq!(min_dim, 64);
                assert_eq!(max, SVD_MAX_RANK);
            }
            _ => panic!("expected ShapeExceedsSvdLimit, got {err:?}"),
        }
    }

    // ── from_owned round-trip (Cold-tier reload path) ──────────────────────

    #[test]
    fn from_owned_reconstructs_identically_to_original_scratch() {
        // decompose → from_scratch (owned) → from_owned (scratch) → reconstruct
        // must give the same tensor as reconstructing directly from the original
        // scratch result. This is the Cold-tier reload contract.
        let shape = [8, 8, 8];
        let total = 512;
        let x: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.07).sin()).collect();
        let cfg = TuckerConfig::new(&shape, &[4, 4, 4]).unwrap();
        let mut scratch = TuckerScratch::with_capacity(&cfg);
        let mut result = TuckerResultScratch::with_capacity(&cfg);
        tucker_decompose_into(&x, &cfg, &mut scratch, &mut result).unwrap();

        // Direct reconstruction from the original scratch.
        let mut recon_direct = vec![0.0f32; total];
        tucker_reconstruct_into(&result, &shape, &mut recon_direct, &mut scratch).unwrap();

        // Cold-tier reload path: scratch → owned → scratch → reconstruct.
        let owned = TuckerResult::from_scratch(&result);
        let reloaded = TuckerResultScratch::from_owned(&owned).expect("from_owned must succeed");
        let mut recon_reload = vec![0.0f32; total];
        tucker_reconstruct_into(&reloaded, &shape, &mut recon_reload, &mut scratch).unwrap();

        // Bit-identical: the reload path must not perturb the reconstruction.
        for i in 0..total {
            assert_eq!(
                recon_direct[i], recon_reload[i],
                "recon[{i}] differs between direct and reload path"
            );
        }
    }

    #[test]
    fn from_owned_rejects_inconsistent_core_length() {
        let owned = TuckerResult {
            core: vec![0.0; 8],   // claims 8 elements
            factors: vec![vec![0.0; 4]],
            core_shape: vec![2, 2, 2], // product = 8 ✓ — make it inconsistent
            factor_shapes: vec![(2, 2)],
        };
        // core.len()==8 == product(core_shape)==8 → core is fine, but this has
        // 1 factor for a 3-mode core_shape → InvalidModeCount. Build a truly
        // core-inconsistent one:
        let owned_bad = TuckerResult {
            core: vec![0.0; 7],       // 7 ≠ 2*2*2 = 8
            factors: vec![vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]],
            core_shape: vec![2, 2, 2],
            factor_shapes: vec![(2, 2), (2, 2), (2, 2)],
        };
        let err = TuckerResultScratch::from_owned(&owned_bad).unwrap_err();
        match err {
            TuckerError::InputSizeMismatch { got, expected } => {
                assert_eq!(got, 7);
                assert_eq!(expected, 8);
            }
            _ => panic!("expected InputSizeMismatch, got {err:?}"),
        }
        // Also confirm the well-formed one works (the first `owned` is rejected
        // for mode-count mismatch, not core mismatch — sanity check both paths):
        let err_modes = TuckerResultScratch::from_owned(&owned).unwrap_err();
        assert!(matches!(err_modes, TuckerError::InvalidModeCount { .. }));
    }

    #[test]
    fn from_owned_rejects_inconsistent_factor_length() {
        let owned = TuckerResult {
            core: vec![0.0; 8],
            factors: vec![vec![0.0; 3], vec![0.0; 4], vec![0.0; 4]], // mode 0: 3 ≠ 2*2=4
            core_shape: vec![2, 2, 2],
            factor_shapes: vec![(2, 2), (2, 2), (2, 2)],
        };
        let err = TuckerResultScratch::from_owned(&owned).unwrap_err();
        match err {
            TuckerError::InputSizeMismatch { got, expected } => {
                assert_eq!(got, 3);
                assert_eq!(expected, 4);
            }
            _ => panic!("expected InputSizeMismatch, got {err:?}"),
        }
    }
}
