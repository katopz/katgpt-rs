//! Decoupled types for the slice_tca module (Plan 596): the class enum, the
//! tensor container, the decomposition artifact, the fit-time scratch, the
//! config, and the error surface. Pure data + zero-alloc consumers; the fit
//! algorithms live in `svd.rs` / `als.rs` / `rank.rs`.

use crate::simd::{simd_dot_f32, simd_fused_scale_acc};
use crate::subspace_phase_gate::{SvdScratch, SvdResultScratch};

// ─── Constants (bounded sizes — no runtime-adaptive ε anywhere) ─────────────

/// Maximum number of components retained per class.
pub const MAX_RANK_PER_CLASS: usize = 8;

/// Hard bound on total components across the three classes.
pub const MAX_COMPONENTS: usize = 3 * MAX_RANK_PER_CLASS;

/// Inner-loop chunk width (fixed accumulation order; helps LLVM vectorize).
pub const CHUNK: usize = 8;

/// Default routing threshold θ: a class routes on when its EVR share exceeds
/// this. CALIBRATED on the Plan 596 synthetic fixtures, not proven — see the
/// Bench 714 caveats.
pub const ROUTE_THETA: f32 = 0.25;

/// Default routing sharpness α for `sigmoid(α·(EVR − θ))`.
pub const ROUTE_ALPHA: f32 = 20.0;

/// Default absolute energy floor τ: a component survives only when its
/// captured energy `w²` is at least `τ · ‖X‖²_F`. Calibrated (Bench 714) to
/// sit above the cross-class spectral spread of generic slice plants
/// (~2–5% per direction) and below planted components (~10%+).
pub const ENERGY_FLOOR_TAU: f32 = 0.05;

/// Default number of ALS sweeps (fixed count, fixed order — no convergence
/// test, keeping the fit deterministic and budget-bounded).
pub const DEFAULT_ALS_SWEEPS: usize = 8;

/// Denominator floor for every scalar divide in the module.
pub const NORM_EPS: f32 = 1e-12;

// ─── SliceClass ─────────────────────────────────────────────────────────────

/// Which axis a component's *loading* vector lives on. A class-σ slice
/// component is `loading ⊗ slice_matrix` where the slice matrix spans the two
/// complementary axes (ascending order).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum SliceClass {
    /// Loading on axis 0 (entity); slice matrix is `t × k`.
    #[default]
    Entity = 0,
    /// Loading on axis 1 (time); slice matrix is `n × k`.
    Time = 1,
    /// Loading on axis 2 (episode); slice matrix is `n × t`.
    Episode = 2,
}

impl SliceClass {
    /// All three classes in axis order.
    pub const ALL: [SliceClass; 3] = [SliceClass::Entity, SliceClass::Time, SliceClass::Episode];

    /// The axis this class's loading lives on.
    #[inline]
    pub fn axis(self) -> usize {
        self as usize
    }

    /// The class whose loading lives on `axis`; `None` for axis > 2.
    #[inline]
    pub fn from_axis(axis: usize) -> Option<Self> {
        match axis {
            0 => Some(SliceClass::Entity),
            1 => Some(SliceClass::Time),
            2 => Some(SliceClass::Episode),
            _ => None,
        }
    }

    /// Complementary axes (ascending) spanned by the slice matrix.
    #[inline]
    pub fn slice_axes(self) -> (usize, usize) {
        match self {
            SliceClass::Entity => (1, 2),
            SliceClass::Time => (0, 2),
            SliceClass::Episode => (0, 1),
        }
    }
}

// ─── Tensor3 ────────────────────────────────────────────────────────────────

/// Owned 3rd-order tensor, row-major `[n][t][k]` (entity × time × episode).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor3 {
    /// `(n, t, k)`.
    pub shape: [usize; 3],
    /// Row-major data, length `n·t·k`; element `(i,p,l)` at `(i·t + p)·k + l`.
    pub data: Vec<f32>,
}

impl Tensor3 {
    /// Zero tensor of the given shape. All dims must be nonzero.
    pub fn zeros(shape: [usize; 3]) -> Self {
        let total = shape[0] * shape[1] * shape[2];
        Self {
            shape,
            data: vec![0.0; total],
        }
    }

    /// Build from a flat row-major buffer, validating the length.
    pub fn from_flat(shape: [usize; 3], data: Vec<f32>) -> Result<Self, SliceTcaError> {
        let total = shape[0] * shape[1] * shape[2];
        if data.len() != total {
            return Err(SliceTcaError::InputSizeMismatch {
                got: data.len(),
                expected: total,
            });
        }
        Ok(Self { shape, data })
    }

    /// Frobenius norm `‖X‖_F` (chunk-8 fixed accumulation order).
    pub fn norm_frob(&self) -> f32 {
        frob_sq(&self.data).sqrt()
    }
}

// ─── SliceDecomposition ─────────────────────────────────────────────────────

/// A slice decomposition: `X ≈ Σ_r loading_r ⊗ slice_r` with each component
/// born into a fixed [`SliceClass`] — the loading vector lives on the class's
/// axis and the slice matrix spans the other two (unit Frobenius norm, so the
/// amplitude lives in the loading; `weights[r] = ‖loading_r‖`).
///
/// Fixed-layout, const-bounded bookkeeping (`MAX_COMPONENTS` arrays); the two
/// payload buffers are concatenations in component order. All consumers
/// (`reconstruct_into`, `entity_slice_into`, `canonical_hash`) are zero-alloc.
#[derive(Clone, Debug)]
pub struct SliceDecomposition {
    /// `(n, t, k)` of the source tensor.
    pub shape: [usize; 3],
    /// Per-class component counts `(R_n, R_t, R_k)`.
    pub ranks: [usize; 3],
    /// Live components (`ranks[0] + ranks[1] + ranks[2]`).
    pub n_components: usize,
    /// Class of each component `0..n_components`.
    pub classes: [SliceClass; MAX_COMPONENTS],
    /// Amplitude of each component (`‖loading‖`), variance-desc sorted.
    pub weights: [f32; MAX_COMPONENTS],
    /// Concatenated loading vectors (offsets in `loading_off`).
    pub loadings: Vec<f32>,
    /// Concatenated unit-norm slice matrices (offsets in `slice_off`).
    pub slices: Vec<f32>,
    loading_off: [usize; MAX_COMPONENTS + 1],
    slice_off: [usize; MAX_COMPONENTS + 1],
    slice_dims: [[usize; 2]; MAX_COMPONENTS],
}

impl SliceDecomposition {
    /// An empty decomposition of the given shape (reconstructs to zero).
    pub fn empty(shape: [usize; 3]) -> Self {
        Self {
            shape,
            ranks: [0; 3],
            n_components: 0,
            classes: [SliceClass::Entity; MAX_COMPONENTS],
            weights: [0.0; MAX_COMPONENTS],
            loadings: Vec::new(),
            slices: Vec::new(),
            loading_off: [0; MAX_COMPONENTS + 1],
            slice_off: [0; MAX_COMPONENTS + 1],
            slice_dims: [[0; 2]; MAX_COMPONENTS],
        }
    }

    /// Append one component (fit-time builder; allocates as needed).
    ///
    /// `slice` is the row-major `slice_rows × slice_cols` matrix for `class`;
    /// it is stored as given — callers pass unit-norm slices so the amplitude
    /// lands in the loading.
    pub fn push_component(
        &mut self,
        class: SliceClass,
        loading: &[f32],
        slice: &[f32],
        slice_rows: usize,
        slice_cols: usize,
        weight: f32,
    ) -> Result<(), SliceTcaError> {
        if self.n_components >= MAX_COMPONENTS {
            return Err(SliceTcaError::TooManyComponents {
                max: MAX_COMPONENTS,
            });
        }
        if slice.len() != slice_rows * slice_cols {
            return Err(SliceTcaError::SliceSizeMismatch {
                got: slice.len(),
                expected: slice_rows * slice_cols,
            });
        }
        let i = self.n_components;
        self.classes[i] = class;
        self.weights[i] = weight;
        self.loading_off[i] = self.loadings.len();
        self.slice_off[i] = self.slices.len();
        self.slice_dims[i] = [slice_rows, slice_cols];
        self.loadings.extend_from_slice(loading);
        self.slices.extend_from_slice(slice);
        self.loading_off[i + 1] = self.loadings.len();
        self.slice_off[i + 1] = self.slices.len();
        self.n_components += 1;
        self.ranks[class.axis()] += 1;
        Ok(())
    }

    /// Loading vector of component `i` (class axis).
    #[inline]
    pub fn loading(&self, i: usize) -> &[f32] {
        &self.loadings[self.loading_off[i]..self.loading_off[i + 1]]
    }

    /// Slice matrix of component `i`: `(flat row-major, rows, cols)`.
    #[inline]
    pub fn slice_matrix(&self, i: usize) -> (&[f32], usize, usize) {
        let s = &self.slices[self.slice_off[i]..self.slice_off[i + 1]];
        (s, self.slice_dims[i][0], self.slice_dims[i][1])
    }

    /// Class of component `i`.
    #[inline]
    pub fn class(&self, i: usize) -> SliceClass {
        self.classes[i]
    }

    /// Amplitude of component `i`.
    #[inline]
    pub fn weight(&self, i: usize) -> f32 {
        self.weights[i]
    }

    /// Amplitudes of all live components (variance-desc order).
    #[inline]
    pub fn weights(&self) -> &[f32] {
        &self.weights[..self.n_components]
    }

    /// Components `0..n_components` as `(class, loading, slice, rows, cols)`.
    pub fn components(&self) -> impl Iterator<Item = SliceComponent<'_>> {
        (0..self.n_components).map(move |i| SliceComponent {
            class: self.classes[i],
            weight: self.weights[i],
            loading: self.loading(i),
            slice: self.slice_matrix(i).0,
            rows: self.slice_dims[i][0],
            cols: self.slice_dims[i][1],
        })
    }

    /// Zero-alloc reconstruction: `x_hat.data ← Σ_r loading_r ⊗ slice_r`.
    ///
    /// Rank-1 GER accumulation in fixed component order (variance-desc),
    /// chunk-fixed inner loops over contiguous rows; `x_hat` is filled first,
    /// so any prior content is discarded. `x_hat.shape` must equal `self.shape`.
    pub fn reconstruct_into(&self, x_hat: &mut Tensor3) -> Result<(), SliceTcaError> {
        if x_hat.shape != self.shape {
            return Err(SliceTcaError::ShapeMismatch {
                got: x_hat.shape,
                expected: self.shape,
            });
        }
        x_hat.data.fill(0.0);
        for i in 0..self.n_components {
            let class = self.classes[i];
            let loading = self.loading(i);
            let (slice, _rows, _cols) = self.slice_matrix(i);
            add_rank1_slice(&mut x_hat.data, self.shape, class, loading, slice, 1.0);
        }
        Ok(())
    }

    /// Zero-alloc per-entity slice: `out ← Σ_r loading_r[entity] · slice_r`
    /// restricted to entity `entity`, where `out` is the row-major `t × k`
    /// slab. Sub-µs hot path for the G2 latency gate: the FIRST component
    /// writes directly (no zero pre-fill pass); the rest accumulate.
    pub fn entity_slice_into(&self, entity: usize, out: &mut [f32]) -> Result<(), SliceTcaError> {
        let [n, t, k] = self.shape;
        if entity >= n {
            return Err(SliceTcaError::EntityOutOfRange { entity, n });
        }
        if out.len() != t * k {
            return Err(SliceTcaError::SliceSizeMismatch {
                got: out.len(),
                expected: t * k,
            });
        }
        if self.n_components == 0 {
            out.fill(0.0);
            return Ok(());
        }
        let tk = t * k;
        // Component 0 WRITES (scaled, per row where needed — no zero pre-fill
        // pass); components 1.. accumulate via the fused-SIMD AXPY.
        for i in 0..self.n_components {
            let first = i == 0;
            match self.classes[i] {
                SliceClass::Entity => {
                    let (slice, _, _) = self.slice_matrix(i);
                    let a = self.loading(i)[entity];
                    if first {
                        scale_write(out, slice, a, tk);
                    } else {
                        simd_fused_scale_acc(out, slice, a, tk);
                    }
                }
                SliceClass::Time => {
                    // slice is (n × k); row `entity` is the entity's profile.
                    let (slice, _, cols) = self.slice_matrix(i);
                    let row_base = entity * cols;
                    let loading = self.loading(i);
                    for p in 0..t {
                        let s = loading[p];
                        let out_row = &mut out[p * k..(p + 1) * k];
                        let row = &slice[row_base..row_base + k];
                        if first {
                            scale_write(out_row, row, s, k);
                        } else {
                            simd_fused_scale_acc(out_row, row, s, k);
                        }
                    }
                }
                SliceClass::Episode => {
                    // slice is (n × t): out[p, l] += slice[p] · loading[l].
                    let (slice, _, _) = self.slice_matrix(i);
                    let row_base = entity * t;
                    let loading = self.loading(i);
                    for p in 0..t {
                        let m_p = slice[row_base + p];
                        let out_row = &mut out[p * k..(p + 1) * k];
                        if first {
                            scale_write(out_row, loading, m_p, k);
                        } else {
                            simd_fused_scale_acc(out_row, loading, m_p, k);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Deterministic identity of the factors: BLAKE3 over a fixed write order
    /// (shape, ranks, classes, weights, loading/slice payloads). Same input
    /// bytes + same triple/codegen ⇒ identical digest (T1.6 pin).
    pub fn canonical_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.shape[0].to_le_bytes());
        hasher.update(&self.shape[1].to_le_bytes());
        hasher.update(&self.shape[2].to_le_bytes());
        hasher.update(&(self.n_components as u32).to_le_bytes());
        for &c in &self.classes[..self.n_components] {
            hasher.update(&[c as u8]);
        }
        for &w in &self.weights[..self.n_components] {
            hasher.update(&w.to_le_bytes());
        }
        hasher.update(&(self.loadings.len() as u32).to_le_bytes());
        for &v in &self.loadings {
            hasher.update(&v.to_le_bytes());
        }
        hasher.update(&(self.slices.len() as u32).to_le_bytes());
        for &v in &self.slices {
            hasher.update(&v.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    // ── fit-time internals (crate-visible for als.rs) ──────────────────────

    /// Write a new loading for component `i` in place (ALS update). The
    /// loading length must match the existing slot.
    pub(crate) fn set_loading(&mut self, i: usize, loading: &[f32]) {
        let slot = &mut self.loadings[self.loading_off[i]..self.loading_off[i + 1]];
        slot.copy_from_slice(loading);
    }

    /// Rebuild `ranks` from the live class array (call after mutation).
    pub(crate) fn recompute_ranks(&mut self) {
        let mut ranks = [0usize; 3];
        for &c in &self.classes[..self.n_components] {
            ranks[c.axis()] += 1;
        }
        self.ranks = ranks;
    }

    /// Canonicalization (T1.5): sign rule (largest-|·| loading entry
    /// positive, first-index tie-break; slice fallback for dead loadings),
    /// weight recompute, deterministic trim (`w² < floor_sq`), and a
    /// variance-desc stable sort with a lexicographic (loading bit-pattern)
    /// tie-break. Rebuilds every bookkeeping array. Public because it is the
    /// committed-surface former: consumers canonicalize hand-built
    /// decompositions before hashing or comparing.
    pub fn canonicalize(&mut self, floor_sq: f32) {
        // Sign rule + weight recompute, in place.
        for i in 0..self.n_components {
            // Compute the weight + sign decision in a scoped borrow, then
            // mutate (assigning `weights[i]` while holding the loading view
            // would alias).
            let (w, flip_loading, flip_slice) = {
                let loading = self.loading(i);
                let w = norm_2(loading);
                if w > NORM_EPS {
                    (w, largest_abs_entry(loading) < 0.0, false)
                } else {
                    // Dead loading: sign-fix the slice instead.
                    let (slice, _, _) = self.slice_matrix(i);
                    (w, false, largest_abs_entry(slice) < 0.0)
                }
            };
            self.weights[i] = w;
            if flip_loading {
                for v in &mut self.loadings[self.loading_off[i]..self.loading_off[i + 1]] {
                    *v = -*v;
                }
                for v in &mut self.slices[self.slice_off[i]..self.slice_off[i + 1]] {
                    *v = -*v;
                }
            }
            if flip_slice {
                for v in &mut self.slices[self.slice_off[i]..self.slice_off[i + 1]] {
                    *v = -*v;
                }
            }
        }
        // Deterministic trim + stable sort by (weight desc, loading bits asc).
        let order: Vec<usize> = (0..self.n_components)
            .filter(|&i| self.weights[i] * self.weights[i] >= floor_sq)
            .collect();
        let mut order = order;
        order.sort_by(|&a, &b| {
            let wa = self.weights[a];
            let wb = self.weights[b];
            wb.total_cmp(&wa).then_with(|| {
                let la = &self.loadings[self.loading_off[a]..self.loading_off[a + 1]];
                let lb = &self.loadings[self.loading_off[b]..self.loading_off[b + 1]];
                lex_bits_cmp(la, lb)
            })
        });
        // Rebuild the payload buffers in the new order.
        let mut loadings = Vec::with_capacity(self.loadings.len());
        let mut slices = Vec::with_capacity(self.slices.len());
        let mut classes = [SliceClass::Entity; MAX_COMPONENTS];
        let mut weights = [0.0f32; MAX_COMPONENTS];
        let mut loading_off = [0usize; MAX_COMPONENTS + 1];
        let mut slice_off = [0usize; MAX_COMPONENTS + 1];
        let mut slice_dims = [[0usize; 2]; MAX_COMPONENTS];
        for (new_i, &old) in order.iter().enumerate() {
            classes[new_i] = self.classes[old];
            weights[new_i] = self.weights[old];
            loadings.extend_from_slice(self.loading(old));
            slices.extend_from_slice(self.slice_matrix(old).0);
            slice_dims[new_i] = self.slice_dims[old];
            loading_off[new_i + 1] = loadings.len();
            slice_off[new_i + 1] = slices.len();
        }
        self.n_components = order.len();
        self.classes = classes;
        self.weights = weights;
        self.loadings = loadings;
        self.slices = slices;
        self.loading_off = loading_off;
        self.slice_off = slice_off;
        self.slice_dims = slice_dims;
        self.recompute_ranks();
    }
}

/// One component view yielded by [`SliceDecomposition::components`].
#[derive(Clone, Copy)]
pub struct SliceComponent<'a> {
    pub class: SliceClass,
    pub weight: f32,
    pub loading: &'a [f32],
    pub slice: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

// ─── Config ─────────────────────────────────────────────────────────────────

/// How the joint ALS is initialized (T1.4).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum InitMode {
    /// HOSVD ([`crate::linalg::tucker_decompose_into`]) when the shape fits
    /// its per-mode SVD bound, Gram-SVD otherwise.
    #[default]
    Auto,
    /// Always the Gram-matrix reduction (`G_σ = X_(σ) X_(σ)ᵀ` factored by
    /// [`crate::subspace_phase_gate::thin_svd_into`]).
    GramSvd,
    /// Always HOSVD; errors when `TuckerConfig` rejects the shape.
    Hosvd,
}

/// All fit knobs. Every default is a const — no runtime-adaptive ε.
#[derive(Clone, Copy, Debug)]
pub struct SliceTcaConfig {
    /// Top-R spectra used by the covariability shares.
    pub share_rank: usize,
    /// Routing threshold θ.
    pub route_theta: f32,
    /// Routing sharpness α.
    pub route_alpha: f32,
    /// Absolute energy floor τ (component trim).
    pub energy_floor_tau: f32,
    /// ALS sweep count (fixed).
    pub als_sweeps: usize,
    /// Initializer selection.
    pub init: InitMode,
}

impl Default for SliceTcaConfig {
    fn default() -> Self {
        Self {
            share_rank: 2,
            route_theta: ROUTE_THETA,
            route_alpha: ROUTE_ALPHA,
            energy_floor_tau: ENERGY_FLOOR_TAU,
            als_sweeps: DEFAULT_ALS_SWEEPS,
            init: InitMode::Auto,
        }
    }
}

// ─── Scratch ────────────────────────────────────────────────────────────────

/// Reusable working memory for every slice_tca entry point. Size for one
/// shape at construction; reuse across calls for zero steady-state
/// allocation (mirrors `TuckerScratch` conventions).
pub struct SliceTcaScratch {
    /// Gram buffer, `max_d²` where `max_d = max(n, t, k)`.
    pub(crate) gram: Vec<f32>,
    /// Per-entity Gram accumulator (`max_d²`).
    pub(crate) gram_acc: Vec<f32>,
    /// Per-entity transpose buffer for the axis-2 Gram (`t·k`).
    pub(crate) trans: Vec<f32>,
    /// Mode-σ unfolding (`n·t·k`).
    pub(crate) unfold: Vec<f32>,
    /// Contraction row (`n·t·k`).
    pub(crate) y_vec: Vec<f32>,
    /// ALS residual (`n·t·k`).
    pub(crate) residual: Vec<f32>,
    /// Component assembly buffers: normalized slice (`n·t·k`) and scaled
    /// loading (`max_d`) — scratch staging for `push_component` copies.
    pub(crate) slice_buf: Vec<f32>,
    pub(crate) loading_buf: Vec<f32>,
    /// Cached top-`MAX_RANK_PER_CLASS` left-singular-vector columns of each
    /// axis's Gram (column-major `d_σ × MAX_RANK_PER_CLASS`), filled by
    /// `spectra_into` (or a standalone `fit_single_class_into`) so the fit
    /// phase never re-Grams/re-SVDs what the shares phase already factored.
    /// Valid per axis after the corresponding computation.
    pub(crate) u_cache: [Vec<f32>; 3],
    pub(crate) u_cache_valid: [bool; 3],
    /// σ² spectra of the three unfoldings (length `d_σ` each).
    pub(crate) spec: [Vec<f32>; 3],
    /// Cached `‖X‖²_F` of the last `covariability_shares_into` call.
    pub(crate) norm_sq: f32,
    /// Per-sweep relative ALS loss of the last fit (length `als_sweeps`).
    losses: Vec<f32>,
    /// SVD working memory, sized `(max_d cols, max_d rows)` — covers the
    /// square Grams and the reallocation rank-1 splits.
    pub(crate) svd_work: SvdScratch,
    pub(crate) svd_result: SvdResultScratch,
}

impl SliceTcaScratch {
    /// Allocate for the given shape. Panics on a zero dimension (use
    /// [`Tensor3`], which cannot represent one).
    pub fn with_capacity(shape: [usize; 3]) -> Self {
        let [n, t, k] = shape;
        assert!(n > 0 && t > 0 && k > 0, "zero dimension in shape {shape:?}");
        let max_d = n.max(t).max(k);
        let total = n * t * k;
        Self {
            gram: vec![0.0; max_d * max_d],
            gram_acc: vec![0.0; max_d * max_d],
            trans: vec![0.0; t * k],
            unfold: vec![0.0; total],
            y_vec: vec![0.0; total],
            residual: vec![0.0; total],
            slice_buf: vec![0.0; total],
            loading_buf: vec![0.0; max_d],
            u_cache: [
                vec![0.0; n * super::types::MAX_RANK_PER_CLASS],
                vec![0.0; t * super::types::MAX_RANK_PER_CLASS],
                vec![0.0; k * super::types::MAX_RANK_PER_CLASS],
            ],
            u_cache_valid: [false; 3],
            spec: [vec![0.0; n], vec![0.0; t], vec![0.0; k]],
            norm_sq: 0.0,
            losses: Vec::new(),
            // SvdScratch::with_capacity takes (n_cols, m_rows) in that order;
            // SvdResultScratch takes (m_rows, n_cols) — opposite, by design.
            svd_work: SvdScratch::with_capacity(max_d, max_d),
            svd_result: SvdResultScratch::with_capacity(max_d, max_d),
        }
    }

    /// Per-sweep relative losses (`‖E‖²/‖X‖²`) recorded by the last
    /// `fit_slice_into` / `fit_with_ranks_into` ALS stage.
    pub fn last_sweep_losses(&self) -> &[f32] {
        &self.losses
    }

    /// The σ² spectra (descending, clamped ≥ 0) stashed by the last
    /// `covariability_shares_into` call — consumed by rank selection.
    pub fn spectra(&self) -> [&[f32]; 3] {
        [&self.spec[0], &self.spec[1], &self.spec[2]]
    }

    /// Cached `‖X‖²_F` from the last shares call.
    pub fn norm_sq(&self) -> f32 {
        self.norm_sq
    }

    pub(crate) fn clear_losses(&mut self) {
        self.losses.clear();
    }

    pub(crate) fn push_loss(&mut self, l: f32) {
        self.losses.push(l);
    }
}

// ─── Error ──────────────────────────────────────────────────────────────────

/// Errors raised by the slice_tca fit / consumer surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceTcaError {
    /// A tensor dimension is zero.
    ZeroDimension,
    /// Flat buffer length disagrees with the shape.
    InputSizeMismatch { got: usize, expected: usize },
    /// Reconstruction target shape disagrees with the decomposition.
    ShapeMismatch { got: [usize; 3], expected: [usize; 3] },
    /// Slice buffer length disagrees with `rows·cols`.
    SliceSizeMismatch { got: usize, expected: usize },
    /// Requested rank exceeds the unfolding bound `min(d_σ, rest_σ)`.
    RankTooLarge { axis: usize, rank: usize, bound: usize },
    /// More than `MAX_COMPONENTS` live components.
    TooManyComponents { max: usize },
    /// `entity` out of range for axis 0.
    EntityOutOfRange { entity: usize, n: usize },
    /// `comp` out of range for the component array.
    ComponentOutOfRange { comp: usize, n: usize },
    /// `InitMode::Hosvd` requested on a shape `TuckerConfig` rejects.
    HosvdShapeUnsupported,
    /// Blocked CV needs at least as many episodes as folds.
    NotEnoughEpisodes { episodes: usize, folds: usize },
}

impl core::fmt::Display for SliceTcaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SliceTcaError::ZeroDimension => write!(f, "zero tensor dimension"),
            SliceTcaError::InputSizeMismatch { got, expected } => {
                write!(f, "input length {got} != expected {expected}")
            }
            SliceTcaError::ShapeMismatch { got, expected } => {
                write!(f, "shape {got:?} != expected {expected:?}")
            }
            SliceTcaError::SliceSizeMismatch { got, expected } => {
                write!(f, "slice length {got} != expected {expected}")
            }
            SliceTcaError::RankTooLarge { axis, rank, bound } => {
                write!(f, "rank {rank} on axis {axis} exceeds bound {bound}")
            }
            SliceTcaError::TooManyComponents { max } => {
                write!(f, "more than {max} components")
            }
            SliceTcaError::EntityOutOfRange { entity, n } => {
                write!(f, "entity {entity} out of range (n = {n})")
            }
            SliceTcaError::ComponentOutOfRange { comp, n } => {
                write!(f, "component {comp} out of range ({n} live)")
            }
            SliceTcaError::HosvdShapeUnsupported => {
                write!(f, "HOSVD init unsupported for this shape (TuckerConfig bound)")
            }
            SliceTcaError::NotEnoughEpisodes { episodes, folds } => {
                write!(f, "{episodes} episodes cannot form {folds} CV folds")
            }
        }
    }
}

impl std::error::Error for SliceTcaError {}

// ─── Shared fixed-order kernels ─────────────────────────────────────────────

/// `dst[i] = scale · src[i]` — element-wise map (order-independent, so no
/// chunking contract); a plain zip loop auto-vectorizes to SIMD stores.
#[inline]
fn scale_write(dst: &mut [f32], src: &[f32], scale: f32, _len: usize) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = scale * s;
    }
}

/// `‖v‖²` with a chunk-8 fixed accumulation order (8 lanes, then folded
/// left-to-right) — the same order everywhere in this module.
pub(crate) fn frob_sq(x: &[f32]) -> f32 {
    let chunks = x.len() / CHUNK;
    let rem = x.len() % CHUNK;
    let mut acc = [0.0f32; CHUNK];
    for c in 0..chunks {
        let base = c * CHUNK;
        for j in 0..CHUNK {
            let v = x[base + j];
            acc[j] += v * v;
        }
    }
    let mut total = 0.0f32;
    for &a in &acc {
        total += a;
    }
    for &v in &x[chunks * CHUNK..chunks * CHUNK + rem] {
        total += v * v;
    }
    total
}

/// `‖v‖₂` via [`frob_sq`].
pub(crate) fn norm_2(v: &[f32]) -> f32 {
    frob_sq(v).sqrt()
}

/// The signed entry of largest magnitude; ties resolved to the FIRST index
/// (strict `>` keeps the earlier winner). Deterministic sign anchor.
pub(crate) fn largest_abs_entry(v: &[f32]) -> f32 {
    let mut best = 0.0f32;
    let mut best_abs = -1.0f32;
    for &x in v {
        let a = x.abs();
        if a > best_abs {
            best_abs = a;
            best = x;
        }
    }
    best
}

/// Lexicographic comparison on raw f32 bit patterns — a deterministic total
/// order used as the canonical sort tie-break.
pub(crate) fn lex_bits_cmp(a: &[f32], b: &[f32]) -> core::cmp::Ordering {
    let n = a.len().min(b.len());
    for i in 0..n {
        let ba = a[i].to_bits();
        let bb = b[i].to_bits();
        if ba != bb {
            return ba.cmp(&bb);
        }
    }
    a.len().cmp(&b.len())
}

/// `data += scale · (loading ⊗ slice)` for one class-σ component — the
/// rank-1 GER kernel shared by reconstruction and the ALS residual updates.
/// Fixed loop order per class; contiguous inner rows. The slice's
/// `(rows, cols)` are implied by `(shape, class)`.
pub(crate) fn add_rank1_slice(
    data: &mut [f32],
    shape: [usize; 3],
    class: SliceClass,
    loading: &[f32],
    slice: &[f32],
    scale: f32,
) {
    let [n, t, k] = shape;
    match class {
        SliceClass::Entity => {
            // slice is (t × k); entity slab `i` is contiguous.
            let tk = t * k;
            for (i, row) in data.chunks_exact_mut(tk).enumerate().take(n) {
                let s = scale * loading[i];
                simd_fused_scale_acc(row, slice, s, tk);
            }
        }
        SliceClass::Time => {
            // slice is (n × k): X[i, p, :] += loading[p] · slice[i, :].
            for (flat, row) in data.chunks_exact_mut(k).enumerate() {
                let i = flat / t;
                let p = flat % t;
                let s = scale * loading[p];
                simd_fused_scale_acc(row, &slice[i * k..(i + 1) * k], s, k);
            }
        }
        SliceClass::Episode => {
            // slice is (n × t): X[i, p, :] += slice[i, p] · loading[:].
            for (flat, row) in data.chunks_exact_mut(k).enumerate() {
                let i = flat / t;
                let p = flat % t;
                let s = scale * slice[i * t + p];
                simd_fused_scale_acc(row, loading, s, k);
            }
        }
    }
}

/// Loading-axis contraction: `out[i] = ⟨E restricted to σ=i, slice⟩` for a
/// class-σ component (slice unit-norm by construction). One tensor
/// contraction per block update (T1.4); `out.len()` must be `d_σ`.
pub(crate) fn contract_axis(
    e: &[f32],
    shape: [usize; 3],
    class: SliceClass,
    slice: &[f32],
    out: &mut [f32],
) {
    let [n, t, k] = shape;
    match class {
        SliceClass::Entity => {
            let tk = t * k;
            for (i, row) in e.chunks_exact(tk).enumerate().take(n) {
                out[i] = simd_dot_f32(row, slice, tk);
            }
        }
        SliceClass::Time => {
            // slice is (n × k): out[p] = Σ_i dot(E[i,p,:], slice[i,:]).
            out[..t].fill(0.0);
            for (flat, row) in e.chunks_exact(k).enumerate() {
                let i = flat / t;
                let p = flat % t;
                out[p] += simd_dot_f32(row, &slice[i * k..(i + 1) * k], k);
            }
        }
        SliceClass::Episode => {
            // slice is (n × t): out[l] = Σ_{i,p} slice[i,p] · E[i,p,l].
            out[..k].fill(0.0);
            for (flat, row) in e.chunks_exact(k).enumerate() {
                let i = flat / t;
                let p = flat % t;
                let m = slice[i * t + p];
                simd_fused_scale_acc(out, row, m, k);
            }
        }
    }
}
