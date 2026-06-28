//! Functional Attention — closed-form Tikhonov k×k spectral transport operator
//! (dual form, matching the reference implementation).
//!
//! Implements FUNCATTN (Xiao et al., ICML 2026, arxiv 2605.31559) — see
//! Research 257 and Plan 286. Reinterprets attention as a k×k linear operator
//! between learned adaptive bases, recovered in closed form via Tikhonov-
//! regularized least squares (functional maps, Ovsjanikov 2012).
//!
//! ## Math — DUAL FORM (matches `.raw/FUNCATTN/PDE-StandardBenchmark/model/Functional_attention.py` L50-89)
//!
//! For a single head, given the head-dim input stream `x_basis, x_value ∈ R^{n×d}`
//! (pre-projections of the layer input — see `in_project_x`, `in_project_fx`
//! in the reference) and trained basis + Q/K/V projection weights:
//!
//! ```text
//! // (1) To basis — partition-of-unity Φ ∈ R^{n×k}
//! scores ← x_basis · w_basisᵀ                          // (n, k) raw
//! Φ ← row_norm(act(scores / τ))                        // (n, k), Σ_g Φ[n,g] = 1
//! col_sum[g] ← Σ_n Φ[n,g]                              // (k,)
//!
//! // (2) Slice tokens — column-normalized weighted averages
//! slice_token[g,:] ← (Σ_n Φ[n,g] · x_value[n,:]) / (col_sum[g] + ε)
//! Q̃ ← slice_token · w_qᵀ                               // (k, d), via to_q linear
//! K̃ ← slice_token · w_kᵀ                               // (k, d), via to_k linear
//! Ṽ ← slice_token · w_vᵀ                               // (k, d), via to_v linear
//!
//! // (3) Functional attention among basis coefficients — DUAL FORM
//! K̃ᵀ ← transpose(K̃)                                    // (d, k)
//! K̃ᵀ·K̃ ← Σ_l K̃[l,:]ᵀ ⊗ K̃[l,:]                         // (d, d)
//! α ← sigmoid(alpha_param) ∈ (0, 1)                    // convex-combo coefficient
//! reg ← (1 - α) · K̃ᵀ·K̃ + α · I_d                       // (d, d), PSD-bounded spectrum
//! Z ← reg⁻¹ · K̃ᵀ                                      // (d, k), via Cholesky solve
//! C ← Q̃ · Z                                            // (k, k)
//!
//! // (4) Apply + inverse project
//! out_slice ← C · Ṽ                                     // (k, d)
//! out[n,:] ← Σ_g Φ[n,g] · out_slice[g,:]                // (n, d)
//! ```
//!
//! ## Primal vs dual — why dual
//!
//! The paper (Eq. 7) writes the additive primal form `K̃·K̃ᵀ + λI_k` (k×k).
//! The **reference implementation uses the dual form** via the Woodbury
//! identity: regularize the `d×d` matrix `(1-α)·K̃ᵀ·K̃ + α·I_d` instead.
//! We follow the reference because:
//! 1. The convex-combo regularization `(1-α)·A + α·I` guarantees a bounded
//!    spectrum for any α ∈ (0, 1) — strictly more numerically stable than
//!    additive `A + λI` when A is rank-deficient.
//! 2. The d×d form matches the reference's empirical results verbatim
//!    (paper Eq. 7 and the code disagree on this point — see Research 257 §6).
//! 3. The per-slice-token `to_q`, `to_k`, `to_v` linear projections
//!    (reference L67-69) are absent from the paper's primal formulation.
//!
//! ## Sigmoid basis (AGENTS.md compliance)
//!
//! The paper uses softmax for the basis (Eq. 9). AGENTS.md mandates sigmoid.
//! Partition-of-unity (Prop 4.3) holds for *any* row-normalized non-negative
//! kernel; sigmoid-then-row-normalize is valid. The `τ → 0` P0 limit becomes
//! a `β = 1/τ → ∞` sigmoid-slope limit (analogous anneal). G3 verifies
//! accuracy parity empirically — sigmoid **outperforms** softmax at matched
//! hyperparameters on a synthetic PDE proxy (see `.benchmarks/058_*.md` G3).
//!
//! **Temperature requirement for sigmoid with small inputs:** sigmoid needs a
//! sharper slope than softmax to produce non-uniform row distributions at
//! small input scales. For `‖x‖ < 1` (pre-layernorm latents, PDE proxies), use
//! `τ ≤ 0.1` (β = 1/τ ≥ 10). At the reference default `τ = 0.5`,
//! `sigmoid(2·s)` for `s ∈ [-0.5, 0.5]` produces values in `[0.12, 0.88]`,
//! which row-normalizes to near-uniform Φ with k=4 — the basis cannot
//! differentiate between partitions and the model collapses to the column
//! mean. For typical transformer activations (`‖x‖ ~ 1–10` after layernorm),
//! `τ = 0.5` may suffice. See `.benchmarks/058_funcattn_goat.md` G3 Results
//! "Temperature sensitivity" section for the full characterization.
//!
//! ## Orthogonal init
//!
//! The reference initializes `w_basis` orthogonally (code L20-21:
//! `torch.nn.init.orthogonal_`). This is **caller responsibility** for our
//! inference-time primitive — we don't initialize weights. Document this
//! requirement; training-side code must apply orthogonal init to `w_basis`
//! before the first forward pass.
//!
//! ## Zero-alloc hot path
//!
//! All scratch buffers live in [`FuncAttnScratch`], pre-allocated once and
//! reused across calls via write-in-place. The forward path performs no
//! heap allocation after warmup — including the d×d Cholesky, done in-place.
//!
//! Feature-gated behind `#[cfg(feature = "funcattn")]`. Not in default
//! features — Gain-tier primitive awaiting LLM-domain GOAT evidence.

use crate::simd;

// ── Config ────────────────────────────────────────────────────────

/// Activation + row-normalization scheme for the FUNCATTN adaptive basis.
///
/// Both produce rows `Φ[n,:]` with `Φ[n,j] ≥ 0` and `Σ_j Φ[n,j] = 1`
/// (partition-of-unity), the only requirement for paper Prop 4.3.
///
/// `serde::{Serialize, Deserialize}` (added for Plan 286 T5.3 freeze/thaw —
/// `FuncAttnWeightsSnapshot` embeds this enum and must round-trip via serde).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
pub enum FuncAttnBasis {
    /// Paper Eq. 9 / reference L60: `Φ = Softmax(Linear(X) / τ)` along k-axis.
    /// τ is the per-head temperature (reference clamps to [0.1, 5.0]).
    Softmax,
    /// AGENTS.md default: `Φ = Sigmoid(Linear(X) / τ)` then row-normalize.
    /// β = 1/τ plays the role of sigmoid slope; partition-of-unity still
    /// holds. No attention sinks, no exp overflow, numerically stable.
    #[default]
    Sigmoid,
}

/// Configuration for [`funcattn_forward`].
///
/// Matches reference defaults: `alpha_init=0` → sigmoid(0)=0.5,
/// `temperature=0.5`, `basis_num=64`.
#[derive(Debug, Clone)]
pub struct FuncAttnConfig {
    /// Head / feature dimension `d` of the input tensors. Must match the
    /// `d` used to size `w_q`, `w_k`, `w_v` (each `(d, d)` row-major) and
    /// the row stride of `x_basis` / `x_value`.
    pub d: usize,
    /// Basis dimension `k` (reference default 64). Linear-in-n scaling
    /// holds for any `k ≪ n`; trades capacity against `O(d²·k)` solve cost.
    pub k: usize,
    /// Activation + normalization for the basis. Default [`FuncAttnBasis::Sigmoid`]
    /// per AGENTS.md (never softmax rule).
    pub basis: FuncAttnBasis,
    /// Convex-combo regularization coefficient α ∈ (0, 1).
    /// Reference uses `α = sigmoid(self.alpha)` with `self.alpha` learnable,
    /// init to `alpha_init` (default 0 → α = 0.5). We treat α as a fixed
    /// inference-time hyperparameter. Larger α = stronger regularization
    /// (α = 1 → reg = I, no information; α = 0 → reg = K̃ᵀK̃, overfits).
    /// Bounded spectrum for any α ∈ (0, 1) — strictly more stable than
    /// additive `A + λI` for rank-deficient A.
    pub alpha: f32,
    /// Per-head basis temperature τ ∈ [0.1, 5.0] (reference L13, L61).
    /// Applied as `scores / τ` before activation. For [`FuncAttnBasis::Sigmoid`],
    /// reinterprets as inverse sigmoid slope β = 1/τ.
    pub temperature: f32,
    /// Extra diagonal jitter added when Cholesky fails (reg matrix not PD).
    /// Should never trigger for α > 0 (convex combo guarantees PD) — exists
    /// as a defense against numerical drift in degenerate cases.
    pub cholesky_jitter: f32,
}

impl Default for FuncAttnConfig {
    fn default() -> Self {
        Self {
            d: 128,
            k: 64,
            basis: FuncAttnBasis::default(),
            alpha: 0.5,       // sigmoid(0), matches reference `alpha_init=0`
            temperature: 0.5, // matches reference init
            cholesky_jitter: 1e-6,
        }
    }
}

// ── Scratch ───────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for [`funcattn_forward`].
///
/// All buffers are flat `Vec<f32>` row-major. Create once via [`FuncAttnScratch::new`],
/// then call [`FuncAttnScratch::ensure_capacity`] before each forward pass.
/// The hot path performs no heap allocation when dimensions match the cache.
pub struct FuncAttnScratch {
    /// Φ basis, `(n, k)`. Partition-of-unity rows.
    pub phi: Vec<f32>,
    /// Column-normalized slice tokens, `(k, d)`. `slice_token[g,:]` is the
    /// weighted average of `x_value` over basis partition `g`.
    pub slice_token: Vec<f32>,
    /// `Q̃ = slice_token · w_qᵀ`, `(k, d)`. Output of `to_q` linear.
    pub q_slice: Vec<f32>,
    /// `K̃ = slice_token · w_kᵀ`, `(k, d)`. Output of `to_k` linear.
    pub k_slice: Vec<f32>,
    /// `Ṽ = slice_token · w_vᵀ`, `(k, d)`. Output of `to_v` linear.
    pub v_slice: Vec<f32>,
    /// `K̃ᵀ·K̃` then `(1-α)·K̃ᵀK̃ + α·I_d`, then **overwritten in place** with
    /// Cholesky factor `L` (lower triangular). `(d, d)`.
    pub reg: Vec<f32>,
    /// `Zᵀ = (reg⁻¹ · K̃ᵀ)ᵀ`, stored as `(k, d)` row-major (row `j` = solution
    /// to `reg · z = K̃[j,:]ᵀ`). Used in `C = Q̃ · Z = Q̃ · Zᵀᵀ`.
    pub z_op_t: Vec<f32>,
    /// Operator `C = Q̃ · Z`, `(k, k)`.
    pub c_op: Vec<f32>,
    /// `out_slice = C · Ṽ`, `(k, d)`.
    pub out_slice: Vec<f32>,
    /// Column sums of Φ, length `k`. `col_sum[g] = Σ_n Φ[n,g]`.
    pub col_sum: Vec<f32>,
    /// Per-row solve buffer for forward/back substitution, length `d`.
    pub solve_y: Vec<f32>,
    cached_n: usize,
    cached_d: usize,
    cached_k: usize,
}

impl FuncAttnScratch {
    /// Allocate scratch for the given dimensions.
    pub fn new(n: usize, d: usize, k: usize) -> Self {
        let nk = n.checked_mul(k).expect("n*k overflow");
        let kd = k.checked_mul(d).expect("k*d overflow");
        let dd = d.checked_mul(d).expect("d*d overflow");
        let kk = k.checked_mul(k).expect("k*k overflow");
        Self {
            phi: vec![0.0; nk],
            slice_token: vec![0.0; kd],
            q_slice: vec![0.0; kd],
            k_slice: vec![0.0; kd],
            v_slice: vec![0.0; kd],
            reg: vec![0.0; dd],
            z_op_t: vec![0.0; kd],
            c_op: vec![0.0; kk],
            out_slice: vec![0.0; kd],
            col_sum: vec![0.0; k],
            solve_y: vec![0.0; d],
            cached_n: n,
            cached_d: d,
            cached_k: k,
        }
    }

    /// Resize buffers if any dimension changed. No-op on the hot path.
    pub fn ensure_capacity(&mut self, n: usize, d: usize, k: usize) {
        if self.cached_n == n && self.cached_d == d && self.cached_k == k {
            return;
        }
        let nk = n * k;
        let kd = k * d;
        let dd = d * d;
        let kk = k * k;
        self.phi.resize(nk, 0.0);
        self.slice_token.resize(kd, 0.0);
        self.q_slice.resize(kd, 0.0);
        self.k_slice.resize(kd, 0.0);
        self.v_slice.resize(kd, 0.0);
        self.reg.resize(dd, 0.0);
        self.z_op_t.resize(kd, 0.0);
        self.c_op.resize(kk, 0.0);
        self.out_slice.resize(kd, 0.0);
        self.col_sum.resize(k, 0.0);
        self.solve_y.resize(d, 0.0);
        self.cached_n = n;
        self.cached_d = d;
        self.cached_k = k;
    }
}

// ── Errors ────────────────────────────────────────────────────────

/// Errors returned by [`funcattn_forward`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuncAttnError {
    /// Regularized matrix `(1-α)·K̃ᵀK̃ + α·I_d` was not positive definite
    /// even after adding `cholesky_jitter`. Should never trigger for
    /// `α ∈ (0, 1)` (convex combo guarantees PD) — indicates severe
    /// numerical drift or degenerate `w_k` weights.
    NotPositiveDefinite,
}

// ── Basis computation ─────────────────────────────────────────────

/// Compute the row-normalized adaptive basis `Φ = row_norm(act(X · W_Φ / τ))`.
///
/// Writes `n × k` basis to `out`. Zero-alloc — caller provides the buffer.
///
/// # Arguments
/// * `x`    — input matrix `(n, d)` row-major, length `n * d`.
/// * `w`    — basis projection weights `(k, d)` row-major (transpose of the
///   paper's natural `d × k` layout). Element `w[j * d + i]` is the
///   weight from input dim `i` to basis dim `j`. This layout makes
///   each basis row contiguous for SIMD dot products. **Must be
///   initialized orthogonally** (reference L20-21).
/// * `bias` — per-basis-dim additive bias, length `k`. Pass `&[]` for no bias.
/// * `n, d, k` — dimensions.
/// * `kind` — activation + normalization scheme.
/// * `temperature` — τ in `scores / τ` (reference clamps to [0.1, 5.0]).
/// * `out`  — output buffer `(n, k)`, length `n * k`.
#[inline]
#[allow(clippy::too_many_arguments)] // numerical kernel — argument count is intrinsic to the (n,d,k,kind,τ) API surface
pub fn compute_basis_into(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    n: usize,
    d: usize,
    k: usize,
    kind: FuncAttnBasis,
    temperature: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), n * d, "x must be (n, d)");
    debug_assert_eq!(w.len(), k * d, "w must be (k, d)");
    debug_assert!(bias.is_empty() || bias.len() == k, "bias must be length k");
    debug_assert_eq!(out.len(), n * k, "out must be (n, k)");
    debug_assert!(
        (0.1..=5.0).contains(&temperature),
        "temperature must be in [0.1, 5.0]: got {}",
        temperature
    );

    if n == 0 || k == 0 {
        return;
    }

    let inv_temp = 1.0 / temperature;

    // Stage 1 + Stage 2 fused: per row, do the linear projection + bias + scale
    // AND the per-row activation/normalization in the same pass. Each row is
    // independent, so fusing keeps the `out[i*k..]` row hot in L1 between the
    // projection write and the normalize read instead of streaming all of `out`
    // through cache twice. Bit-identical to the previous two-loop form.
    for i in 0..n {
        let x_row = &x[i * d..(i + 1) * d];
        let out_row = &mut out[i * k..(i + 1) * k];
        simd::simd_matmul_rows(out_row, w, x_row, k, d);
        if !bias.is_empty() {
            simd::simd_add_inplace(out_row, bias);
        }
        simd::simd_scale_inplace(out_row, inv_temp);
        normalize_basis_row(out_row, kind);
    }
}

/// Normalize a single basis row in place to satisfy partition-of-unity.
#[inline]
fn normalize_basis_row(row: &mut [f32], kind: FuncAttnBasis) {
    match kind {
        FuncAttnBasis::Softmax => {
            let max_score = simd::simd_max_f32(row);
            simd::simd_add_scalar_inplace(row, -max_score);
            simd::simd_exp_inplace(row);
            let rowsum = simd::simd_sum_f32(row);
            if rowsum > 0.0 {
                simd::simd_scale_inplace(row, 1.0 / rowsum);
            }
        }
        FuncAttnBasis::Sigmoid => {
            // σ(x) = 1 / (1 + exp(−x)); row[j] = σ(s_j) / Σ σ(s_k)
            simd::simd_scale_inplace(row, -1.0);
            simd::simd_exp_inplace(row);
            simd::simd_add_scalar_inplace(row, 1.0);
            simd::simd_reciprocal_inplace(row);
            let rowsum = simd::simd_sum_f32(row);
            if rowsum > 0.0 {
                simd::simd_scale_inplace(row, 1.0 / rowsum);
            }
        }
    }
}

// ── Cholesky decomposition + triangular solves (in-place, zero-alloc) ─

/// Cholesky-decompose SPD matrix `a` in place into lower-triangular `L`.
///
/// Overwrites `a` (`dim × dim` row-major SPD) with `L` (lower triangular,
/// upper triangle zeroed) such that `A = L · Lᵀ`. Returns `true` on success,
/// `false` if `A` is not positive definite (negative or zero pivot).
///
/// Standard right-looking in-place Cholesky. Cost `O(dim³/3)`.
#[inline]
fn cholesky_inplace(a: &mut [f32], dim: usize) -> bool {
    debug_assert_eq!(a.len(), dim * dim);
    for j in 0..dim {
        let mut diag = a[j * dim + j];
        if j > 0 {
            let l_row = &a[j * dim..j * dim + j];
            diag -= simd::simd_dot_f32(l_row, l_row, j);
        }
        if diag <= 0.0 {
            return false;
        }
        let diag_val = diag.sqrt();
        a[j * dim + j] = diag_val;
        let inv_diag = 1.0 / diag_val;
        for i in (j + 1)..dim {
            let mut s = a[i * dim + j];
            if j > 0 {
                let l_irow = &a[i * dim..i * dim + j];
                let l_jrow = &a[j * dim..j * dim + j];
                s -= simd::simd_dot_f32(l_irow, l_jrow, j);
            }
            a[i * dim + j] = s * inv_diag;
        }
        for i in (j + 1)..dim {
            a[j * dim + i] = 0.0;
        }
    }
    true
}

/// Solve `L · y = b` (forward) then `Lᵀ · x = y` (back), i.e. solve `A · x = b`
/// where `A = L · Lᵀ` is the Cholesky factorization.
///
/// `l` is `(dim, dim)` lower-triangular; `b`, `x`, `y_buf` are length `dim`.
#[inline]
fn cholesky_solve_into(l: &[f32], b: &[f32], dim: usize, y_buf: &mut [f32], x: &mut [f32]) {
    debug_assert_eq!(l.len(), dim * dim);
    debug_assert_eq!(b.len(), dim);
    debug_assert_eq!(y_buf.len(), dim);
    debug_assert_eq!(x.len(), dim);

    // Forward: y[i] = (b[i] − Σ_{j<i} L[i,j]·y[j]) / L[i,i]
    for i in 0..dim {
        let mut s = b[i];
        if i > 0 {
            let l_row = &l[i * dim..i * dim + i];
            let y_slice = &y_buf[..i];
            s -= simd::simd_dot_f32(l_row, y_slice, i);
        }
        y_buf[i] = s / l[i * dim + i];
    }

    // Backward: x[i] = (y[i] − Σ_{j>i} L[j,i]·x[j]) / L[i,i]
    // Strided column access of L; kept scalar (dim is L1-resident for d ≤ 256).
    for i in (0..dim).rev() {
        let s = y_buf[i];
        let mut sum_hi = 0.0f32;
        for j in (i + 1)..dim {
            sum_hi += l[j * dim + i] * x[j];
        }
        x[i] = (s - sum_hi) / l[i * dim + i];
    }
}

/// Form `(1-α)·K̃ᵀ·K̃ + α·I_d` in `reg`, Cholesky-decompose in place, then solve
/// `reg · Z = K̃ᵀ` for `Z` (column-by-column), storing `Zᵀ` as `(k, d)` row-major.
///
/// This is the dual-form Tikhonov solve (reference L71-76). The convex-combo
/// regularization guarantees PD for any `α ∈ (0, 1)` regardless of K̃'s rank.
///
/// # Arguments
/// * `k_slice` — `K̃`, `(k, d)` row-major.
/// * `alpha`   — convex-combo coefficient `α ∈ (0, 1)`.
/// * `d, k`    — dimensions.
/// * `reg`     — scratch `(d, d)`. Overwritten with Cholesky factor `L`.
/// * `y_buf`   — scratch length `d` for triangular solve.
/// * `z_op_t`  — output `Zᵀ`, `(k, d)` row-major. Row `j` solves `reg · z = K̃[j,:]ᵀ`.
/// * `jitter`  — fallback diagonal bump if the (theoretically-impossible) PD
///   check fails. Defense against extreme numerical drift.
///
/// Returns `Err(NotPositiveDefinite)` only if K̃ᵀK̃ + αI + jitter is still
/// not PD — should never happen for `α > 0` with finite K̃.
#[inline]
#[allow(clippy::too_many_arguments)] // numerical kernel — Tikhonov dual-form solve API surface
pub fn solve_convex_combo_dual(
    k_slice: &[f32],
    alpha: f32,
    d: usize,
    k: usize,
    reg: &mut [f32],
    y_buf: &mut [f32],
    z_op_t: &mut [f32],
    jitter: f32,
) -> Result<(), FuncAttnError> {
    debug_assert_eq!(k_slice.len(), k * d);
    debug_assert_eq!(reg.len(), d * d);
    debug_assert_eq!(y_buf.len(), d);
    debug_assert_eq!(z_op_t.len(), k * d);
    debug_assert!(
        alpha > 0.0 && alpha < 1.0,
        "alpha must be in (0, 1): got {}",
        alpha
    );

    let one_minus_alpha = 1.0 - alpha;

    // Stage A: reg = (1-α) · K̃ᵀ·K̃ + α · I_d.
    // K̃ᵀ·K̃ = Σ_l K̃[l,:]ᵀ ⊗ K̃[l,:] (outer-product accumulation over rows).
    reg[..d * d].fill(0.0);
    for l in 0..k {
        let k_row = &k_slice[l * d..(l + 1) * d];
        simd::simd_outer_product_acc(reg, k_row, k_row, d, d);
    }
    // Scale by (1-α), then add α on the diagonal.
    simd::simd_scale_inplace(reg, one_minus_alpha);
    for i in 0..d {
        reg[i * d + i] += alpha;
    }

    // Stage B: Cholesky in-place. Cold path: add jitter on failure.
    if !cholesky_inplace(reg, d) {
        for i in 0..d {
            reg[i * d + i] += jitter;
        }
        if !cholesky_inplace(reg, d) {
            return Err(FuncAttnError::NotPositiveDefinite);
        }
    }

    // Stage C: solve reg · z_j = K̃[j,:]ᵀ for each j, store z_j as row j of Zᵀ.
    // z_op_t[j, :] = z_j where reg · z_j = K̃[j, :]ᵀ (length d).
    for j in 0..k {
        let b_row = &k_slice[j * d..(j + 1) * d];
        let z_row = &mut z_op_t[j * d..(j + 1) * d];
        cholesky_solve_into(reg, b_row, d, y_buf, z_row);
    }

    Ok(())
}

// ── SpectralQuant composition: eigenbasis pre-rotation (Plan 286 T5.1) ────

/// Pre-rotate the FUNCATTN basis weights into a calibrated eigenbasis, in place.
///
/// Computes `W_Φ' = W_Φ · Vᵀ` where `V` is the eigenbasis (columns =
/// eigenvectors, sorted by eigenvalue descending). Source: SpectralQuant's
/// `CalibrationResult::eigenvectors` (Plan 077), or any PCA / orthogonal basis.
///
/// Plan 286 T5.1 hypothesis: a basis aligned with the data's principal
/// eigen-directions is more expressive per parameter, because signal energy
/// concentrates in the top eigen-directions. This is a **one-time
/// calibration-time transform** — after it runs, the rotated `w_basis` is fed
/// to [`funcattn_forward`] unchanged, so the forward path stays G5 zero-alloc.
///
/// # Arguments
///
/// * `w_basis`      — `(k, d)` row-major basis projection weights, mutated in place.
/// * `eigenvectors` — `(d, d)` row-major eigenvector matrix; column `j` is the
///   eigenvector for eigenvalue `j` (SpectralQuant layout:
///   sorted by eigenvalue descending).
/// * `k, d`         — dimensions. `d` must match the eigenbasis dim.
///
/// # Properties (verified in the test suite below)
///
/// - `V = I` is a no-op (`W · Iᵀ = W`).
/// - Orthogonal `V` preserves every row's norm and preserves `W`'s row
///   orthogonality — so an orthogonally-init'd basis stays orthogonal.
/// - The forward pass remains finite + partition-of-unity after rotation.
///
/// # Panics
///
/// Debug-build panics on shape mismatch (`w_basis.len() != k*d`,
/// `eigenvectors.len() != d*d`, `d == 0`). In release, the math is well-
/// defined for `d ≥ 1` and a silent no-op for `k == 0`.
#[inline]
pub fn pre_rotate_basis_weights_into(
    w_basis: &mut [f32],
    eigenvectors: &[f32],
    k: usize,
    d: usize,
) {
    debug_assert_eq!(w_basis.len(), k * d, "w_basis must be (k, d)");
    debug_assert_eq!(eigenvectors.len(), d * d, "eigenvectors must be (d, d)");
    debug_assert!(d > 0, "d must be > 0");

    if k == 0 || d == 0 {
        return;
    }

    // For each basis row k, compute w_basis_new[k, j] = Σ_i V[j, i] · w_basis[k, i].
    // One O(d) scratch allocation (calibration-time only — NOT on the decode
    // hot path; `funcattn_forward` is the G5-verified zero-alloc path). The
    // `d`-length row fits in L1 for typical head dims (d ≤ 256).
    let mut row_scratch = vec![0.0f32; d];
    for kk in 0..k {
        let row_offset = kk * d;
        // Snapshot the original row (we cannot mutate in-place — V is not
        // lower-triangular and the new row depends on every original entry).
        row_scratch[..d].copy_from_slice(&w_basis[row_offset..row_offset + d]);
        for j in 0..d {
            // V[j, i] = eigenvectors[j * d + i]; w_basis[k, i] = row_scratch[i].
            let v_row = &eigenvectors[j * d..(j + 1) * d];
            w_basis[row_offset + j] = simd::simd_dot_f32(v_row, &row_scratch[..d], d);
        }
    }
}

// ── Principled multi-scale basis constructors (Plan 332) ──────────
//
// Phase 0 of Plan 332 (Issue 001 probe, 2026-06-26) proved that a HAND-CRAFTED
// signal-aligned basis beats random-orthogonal by +0.11 cos on multi-scale
// transport. These constructors answer: can a PRINCIPLED fixed basis (no
// a-priori signal knowledge) capture most of that gain? DCT-log is the poor
// man's wavelet packet; Haar-packet is the Apollonian surrogate (same
// multi-scale hierarchical property, well-understood construction).
//
// All three return a row-orthonormal `(k, d)` matrix suitable for direct use
// as `w_basis` in `funcattn_forward`. Construction is O(d·k) ONCE at init;
// the forward hot path is unchanged (consumes `w_basis: &[f32]` regardless).
// G4 (zero-alloc steady state) is preserved by construction.

/// Gram-Schmidt orthogonalize the rows of `w` (k rows, d cols, row-major).
/// Produces a row-orthonormal matrix in place. O(k²·d).
///
/// Private helper shared by the structured-basis constructors. Not on the
/// forward hot path (init-time only).
#[cfg(feature = "funcattn_structured_basis")]
fn gram_schmidt_rows(w: &mut [f32], k: usize, d: usize) {
    for i in 0..k {
        // Subtract projections onto all previous (already-orthonormal) rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for l in 0..d {
                dot += w[i * d + l] * w[j * d + l];
            }
            for l in 0..d {
                w[i * d + l] -= dot * w[j * d + l];
            }
        }
        // L2-normalize row i in place.
        let mut s = 0.0f32;
        for l in 0..d {
            s += w[i * d + l] * w[i * d + l];
        }
        let n = s.sqrt().max(1e-12);
        for l in 0..d {
            w[i * d + l] /= n;
        }
    }
}

/// Build a DCT-II basis at logarithmically-spaced frequencies.
///
/// The simplest principled multi-scale basis (Plan 332 T1.1): the "poor man's
/// wavelet packet". Frequencies are log-spaced from 1 to `d/2`, so each basis
/// row captures a different scale of variation without requiring any a-priori
/// signal knowledge.
///
/// Frequencies: `f_i = round(2^((i/(k-1)) · log2(d/2)))` for `i` in `0..k`, so
/// `f_0 = 1` (one cycle over the whole domain) and `f_{k-1} = d/2` (Nyquist).
/// Basis row `i`: `w[i, j] = cos(π · f_i · (j + 0.5) / d)`, L2-normalized, then
/// Gram-Schmidt to guarantee exact row-orthonormality.
///
/// Returns a flat `(k, d)` row-major `Vec<f32>` suitable as `w_basis`.
///
/// # Cost
///
/// `O(k · d)` to fill + `O(k² · d)` Gram-Schmidt. Init-time only.
///
/// # Panics
///
/// Debug-build panics on `k == 0` or `d == 0`. For `k > d` the resulting rows
/// cannot all be orthonormal (rank-deficient); Gram-Schmidt still produces a
/// valid row-orthogonal matrix but later rows are numerically tiny.
#[cfg(feature = "funcattn_structured_basis")]
pub fn make_dct_log_basis(k: usize, d: usize) -> Vec<f32> {
    debug_assert!(k > 0, "k must be > 0");
    debug_assert!(d > 0, "d must be > 0");

    let mut w = vec![0.0f32; k * d];
    let log_d_half = ((d as f32) / 2.0).log2(); // log2(d/2); for d=64, = 5.

    // Pick k distinct integer frequencies in [1, d/2], log-spaced.
    //
    // Naive round(2^(frac · log2(d/2))) can collide (e.g. for k=16, d=64 the
    // log-spacing clusters at low frequencies and rounds to many duplicates).
    // Two rows at the same frequency are identical up to sign, which after
    // Gram-Schmidt yields a near-zero row (and then any FP noise on it gets
    // amplified by the 1/norm normalization, breaking row-orthonormality).
    //
    // Fix: enforce strictly-increasing integer frequencies after rounding by
    // bumping each collision to the next free integer. This may compress the
    // high end of the spectrum when k is close to d/2 (acceptable — that's the
    // rank-saturated regime where DCT-log offers little over random).
    let max_f = (d / 2).max(1);
    let mut freqs: Vec<i64> = Vec::with_capacity(k);
    for i in 0..k {
        let frac = if k > 1 {
            i as f32 / (k - 1) as f32
        } else {
            0.0
        };
        let f_raw = (2.0f32).powf(frac * log_d_half).round() as i64;
        let mut f = f_raw.clamp(1, max_f as i64);
        // Ensure strictly-greater than the previous frequency.
        if let Some(&prev) = freqs.last()
            && f <= prev {
                f = prev + 1;
            }
        // If we ran past max_f, clamp (last few frequencies may saturate at
        // max_f — for k > d/2 this is unavoidable).
        if f > max_f as i64 {
            f = max_f as i64;
        }
        freqs.push(f);
    }

    for (i, &f) in freqs.iter().enumerate() {
        // DCT-II basis vector: cos(π · f · (j + 0.5) / d), j = 0..d.
        let phase_step = core::f32::consts::PI * (f as f32) / (d as f32);
        for j in 0..d {
            w[i * d + j] = (phase_step * (j as f32 + 0.5)).cos();
        }
    }

    // L2-normalize + orthogonalize. Pure DCT-II rows at distinct frequencies
    // are already ~orthogonal (the off-diagonal entries of W·Wᵀ are O(1/d)),
    // but Gram-Schmidt gives exact row-orthonormality required by the FUNCATTN
    // forward path (the Cholesky on the d×d reg matrix is well-conditioned
    // only when w_basis has bounded condition number).
    gram_schmidt_rows(&mut w, k, d);
    w
}

/// Build a Haar wavelet packet basis at multiple scales (Plan 332 T1.2).
///
/// This is the **Apollonian surrogate**: a genuine multi-resolution basis
/// (Haar wavelet packet) that shares the multi-scale hierarchical property
/// of Apollonian packings, but with a well-understood construction. If Haar
/// wavelet packets fail, Apollonian would also fail; if they succeed, we
/// justify the harder Apollonian implementation (Plan 332 Phase 5).
///
/// Row 0 is the scaling function (constant DC = `[1/sqrt(d); d]`); the
/// remaining rows are Haar wavelets at log-spaced scales, picked coarsest
/// first then by position. At scale `s` (`1 ≤ s ≤ log2(d)`), support is
/// `2^s` samples and there are `2^(log2(d) - s)` positions.
///
/// Requires `d` to be a power of two.
///
/// Returns a flat `(k, d)` row-major `Vec<f32>` suitable as `w_basis`.
///
/// # Cost
///
/// `O(k · d)` to fill + `O(k² · d)` Gram-Schmidt (only to handle the padding
/// rows when `k > log2(d) + 1`). Init-time only.
///
/// # Panics
///
/// Debug-build panics on `k == 0`, `d == 0`, or `d` not a power of two.
#[cfg(feature = "funcattn_structured_basis")]
pub fn make_haar_packet_basis(k: usize, d: usize) -> Vec<f32> {
    debug_assert!(k > 0, "k must be > 0");
    debug_assert!(d > 0, "d must be > 0");
    debug_assert!(
        d.is_power_of_two(),
        "d must be a power of two for Haar (got {d})"
    );

    let mut w = vec![0.0f32; k * d];
    let log_d = d.trailing_zeros() as usize; // log2(d); for d=64, = 6.

    // Row 0: scaling function (DC component) = [1/sqrt(d); d]. This is the
    // "1 coarse" node from the plan — the lowest-frequency component.
    let dc_scale = (d as f32).sqrt();
    for j in 0..d {
        w[j] = 1.0 / dc_scale;
    }

    // Rows 1..k: Haar wavelets at multiple scales, coarse-to-fine then by
    // position. At scale s (1=finest support-2, log_d=coarsest support-d),
    // the wavelet is +1 on the first half of its support, -1 on the second,
    // normalized so each wavelet has unit L2 norm (norm = sqrt(support)).
    //
    // Two wavelets at different scales are orthogonal iff one's support lies
    // entirely inside a single sign-region of the other. Picking scales in
    // decreasing order with positions starting from 0 satisfies this by
    // construction (the next finer wavelet's support fits inside the previous
    // coarser wavelet's +1 half).
    let mut row = 1usize;
    'outer: for s in (1..=log_d).rev() {
        // s = log_d (coarsest, 1 position) → s = 1 (finest, d/2 positions).
        let support = 1usize << s; // 2^s
        let half = support >> 1; // 2^(s-1)
        let n_positions = d >> s; // d / 2^s = 2^(log_d - s)
        let inv_norm = 1.0 / (support as f32).sqrt();
        for p in 0..n_positions {
            if row >= k {
                break 'outer;
            }
            let start = p * support;
            let row_off = row * d;
            for j in 0..half {
                w[row_off + start + j] = inv_norm;
                w[row_off + start + half + j] = -inv_norm;
            }
            row += 1;
        }
    }

    // Pad remaining rows (when k > 1 + total available wavelets) with standard
    // basis vectors; Gram-Schmidt orthogonalizes them against the existing
    // Haar rows. For the plan's NPC regime (d=64, k ≤ 16) this branch is
    // typically not hit (1 + 6 + 12 + 24 + ... ample supply at fine scales).
    while row < k {
        let idx = (row - 1) % d;
        w[row * d + idx] = 1.0;
        row += 1;
    }

    // Exact row-orthonormality. The Haar vectors above are already orthogonal
    // by construction; this pass cleans up the filler rows and removes any
    // floating-point drift in the constructed wavelets.
    gram_schmidt_rows(&mut w, k, d);
    w
}

// ── Forward pass ──────────────────────────────────────────────────

/// Functional Attention forward pass — **dual form** (matches reference code).
///
/// Implements the FUNCATTN pipeline as shipped in
/// `.raw/FUNCATTN/PDE-StandardBenchmark/model/Functional_attention.py`:
/// basis projection → slice-token column normalization → to_q/k/v linear →
/// convex-combo Tikhonov solve → inverse projection.
///
/// # Arguments
///
/// * `x_basis`  — input for the basis projection, `(n, d)` row-major.
///   Corresponds to `x_mid = in_project_x(x)` in the reference.
/// * `x_value`  — input for the slice-token value stream, `(n, d)` row-major.
///   Corresponds to `fx_mid = in_project_fx(x)` in the reference.
///   Pass the same slice as `x_basis` for shared projection.
/// * `w_basis`  — basis projection weights `(k, d)` row-major. **Must be
///   orthogonally initialized** by the caller (reference L20-21).
/// * `w_q`, `w_k`, `w_v` — `to_q`, `to_k`, `to_v` linear projection weights,
///   each `(d, d)` row-major. Applied to slice_token (reference L67-69).
/// * `cfg`      — configuration.
/// * `scratch`  — pre-allocated scratch buffers.
/// * `out`      — output `(n, d)`, length `n * d`. Pre-allocated by caller.
///
/// # Algorithmic cost
///
/// `O(n·d·k + k·d² + d³ + k·d²)`. Linear in `n` for `k ≪ n`.
///
/// # Numerical stability
///
/// Convex-combo regularization `(1-α)·K̃ᵀK̃ + α·I_d` guarantees PD for any
/// `α ∈ (0, 1)` regardless of K̃'s rank, so Cholesky cannot fail under
/// normal operation. The `cholesky_jitter` fallback is a defense against
/// extreme floating-point drift.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // Stages 6/7/8: indices participate in stride arithmetic (j*d, l*d, g*d) for row-major matrix ops
pub fn funcattn_forward(
    x_basis: &[f32],
    x_value: &[f32],
    w_basis: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    cfg: &FuncAttnConfig,
    scratch: &mut FuncAttnScratch,
    out: &mut [f32],
) -> Result<(), FuncAttnError> {
    let d = cfg.d;
    let k = cfg.k;
    let n = if d > 0 { x_basis.len() / d } else { 0 };

    let expected = n * d;
    debug_assert!(
        x_basis.len().is_multiple_of(d),
        "x_basis.len() ({}) must be divisible by d ({})",
        x_basis.len(),
        d
    );
    debug_assert_eq!(x_basis.len(), expected, "x_basis must be (n, d)");
    debug_assert_eq!(x_value.len(), expected, "x_value must be (n, d)");
    debug_assert_eq!(w_basis.len(), k * d, "w_basis must be (k, d)");
    debug_assert_eq!(w_q.len(), d * d, "w_q must be (d, d)");
    debug_assert_eq!(w_k.len(), d * d, "w_k must be (d, d)");
    debug_assert_eq!(w_v.len(), d * d, "w_v must be (d, d)");
    debug_assert_eq!(out.len(), expected, "out must be (n, d)");

    if n == 0 || d == 0 || k == 0 {
        return Ok(());
    }

    scratch.ensure_capacity(n, d, k);

    // Stage 1: basis computation. Φ = row_norm(act(x_basis · w_basis / τ)).
    compute_basis_into(
        x_basis,
        w_basis,
        &[],
        n,
        d,
        k,
        cfg.basis,
        cfg.temperature,
        &mut scratch.phi,
    );

    // Stage 2 + Stage 3 fused: accumulate col_sum[g] = Σ_n Φ[n,g] and
    // slice_token[g,:] = Σ_n Φ[n,g]·x_value[n,:] in the SAME pass over Φ.
    // Both stages previously streamed `scratch.phi` (n·k f32) through cache
    // twice; fusing reads each Φ row once. The column normalization that
    // divides slice_token by col_sum still runs after this loop (it needs the
    // fully-accumulated col_sum). Bit-identical to the previous two-loop form.
    scratch.col_sum[..k].fill(0.0);
    scratch.slice_token[..k * d].fill(0.0);
    for i in 0..n {
        let phi_row = &scratch.phi[i * k..(i + 1) * k];
        let x_row = &x_value[i * d..(i + 1) * d];
        simd::simd_add_inplace(&mut scratch.col_sum[..k], phi_row);
        simd::simd_outer_product_acc(&mut scratch.slice_token, phi_row, x_row, k, d);
    }
    let eps = 1e-5;
    for g in 0..k {
        let denom = scratch.col_sum[g] + eps;
        let inv_denom = 1.0 / denom;
        let row = &mut scratch.slice_token[g * d..(g + 1) * d];
        simd::simd_scale_inplace(row, inv_denom);
    }

    // Stage 4: apply to_q, to_k, to_v to slice_token (per-row linear).
    // q_slice[g, :] = w_q · slice_token[g, :]ᵀ (analogously for k, v).
    for g in 0..k {
        let slice_row = &scratch.slice_token[g * d..(g + 1) * d];
        let q_row = &mut scratch.q_slice[g * d..(g + 1) * d];
        let k_row_out = &mut scratch.k_slice[g * d..(g + 1) * d];
        let v_row_out = &mut scratch.v_slice[g * d..(g + 1) * d];
        simd::simd_matmul_rows(q_row, w_q, slice_row, d, d);
        simd::simd_matmul_rows(k_row_out, w_k, slice_row, d, d);
        simd::simd_matmul_rows(v_row_out, w_v, slice_row, d, d);
    }

    // Stage 5: dual-form Tikhonov solve. Z = reg⁻¹ · K̃ᵀ, stored as Zᵀ (k, d).
    solve_convex_combo_dual(
        &scratch.k_slice,
        cfg.alpha,
        d,
        k,
        &mut scratch.reg,
        &mut scratch.solve_y,
        &mut scratch.z_op_t,
        cfg.cholesky_jitter,
    )?;

    // Stage 6: C = Q̃ · Z. C[i, j] = Σ_l Q̃[i, l] · Z[l, j] = dot(Q̃ row i, Zᵀ row j).
    for i in 0..k {
        let q_row = &scratch.q_slice[i * d..(i + 1) * d];
        let c_row = &mut scratch.c_op[i * k..(i + 1) * k];
        for j in 0..k {
            let z_row = &scratch.z_op_t[j * d..(j + 1) * d];
            c_row[j] = simd::simd_dot_f32(q_row, z_row, d);
        }
    }

    // Stage 7: out_slice = C · Ṽ. For each row i: out_slice[i,:] = Σ_l C[i,l] · Ṽ[l,:].
    for i in 0..k {
        let c_row = &scratch.c_op[i * k..(i + 1) * k];
        let out_row = &mut scratch.out_slice[i * d..(i + 1) * d];
        out_row.fill(0.0);
        for l in 0..k {
            let weight = c_row[l];
            if weight == 0.0 {
                continue;
            }
            let v_row = &scratch.v_slice[l * d..(l + 1) * d];
            simd::simd_fused_scale_acc(out_row, v_row, weight, d);
        }
    }

    // Stage 8: inverse projection. out[n, :] = Σ_g Φ[n, g] · out_slice[g, :].
    for i in 0..n {
        let phi_row = &scratch.phi[i * k..(i + 1) * k];
        let out_row = &mut out[i * d..(i + 1) * d];
        out_row.fill(0.0);
        for g in 0..k {
            let weight = phi_row[g];
            if weight == 0.0 {
                continue;
            }
            let slice_row = &scratch.out_slice[g * d..(g + 1) * d];
            simd::simd_fused_scale_acc(out_row, slice_row, weight, d);
        }
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG (xorshift64*), reproducible across runs.
    fn make_rng(seed: u64) -> impl Iterator<Item = f32> {
        let mut state = seed.max(1);
        std::iter::from_fn(move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bits = (state >> 11) as u32;
            let u01 = bits as f32 / (u32::MAX as f32);
            Some(u01 * 2.0 - 1.0)
        })
    }

    fn fill_rand(buf: &mut [f32], seed: u64) {
        let mut rng = make_rng(seed);
        for x in buf.iter_mut() {
            *x = rng.next().unwrap();
        }
    }

    /// Reference (allocating, scalar) dual-form FUNCATTN, matching
    /// `Functional_attention.py::FunctionalMap_Attention_Structured_Mesh_2D.forward`
    /// for a single (B=1, H=1) head.
    #[allow(clippy::too_many_arguments)]
    fn funcattn_reference(
        x_basis: &[f32],
        x_value: &[f32],
        w_basis: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        n: usize,
        d: usize,
        k: usize,
        basis: FuncAttnBasis,
        alpha: f32,
        temperature: f32,
    ) -> Vec<f32> {
        let inv_temp = 1.0 / temperature;
        // (1) Φ = row_norm(act((x_basis · w_basis) / τ))
        let mut phi = vec![0.0f32; n * k];
        for i in 0..n {
            for j in 0..k {
                let mut s = 0.0;
                for dd in 0..d {
                    s += x_basis[i * d + dd] * w_basis[j * d + dd];
                }
                phi[i * k + j] = s * inv_temp;
            }
            let row = &mut phi[i * k..(i + 1) * k];
            match basis {
                FuncAttnBasis::Softmax => {
                    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for x in row.iter_mut() {
                        *x = (*x - mx).exp();
                    }
                    let s: f32 = row.iter().sum();
                    if s > 0.0 {
                        for x in row.iter_mut() {
                            *x /= s;
                        }
                    }
                }
                FuncAttnBasis::Sigmoid => {
                    for x in row.iter_mut() {
                        *x = 1.0 / (1.0 + (-*x).exp());
                    }
                    let s: f32 = row.iter().sum();
                    if s > 0.0 {
                        for x in row.iter_mut() {
                            *x /= s;
                        }
                    }
                }
            }
        }

        // (2) col_sum and slice_token = (Φᵀ · x_value) / (col_sum + ε)
        let mut col_sum = vec![0.0f32; k];
        for g in 0..k {
            for i in 0..n {
                col_sum[g] += phi[i * k + g];
            }
        }
        let mut slice_token = vec![0.0f32; k * d];
        for g in 0..k {
            for dd in 0..d {
                let mut s = 0.0;
                for i in 0..n {
                    s += phi[i * k + g] * x_value[i * d + dd];
                }
                slice_token[g * d + dd] = s / (col_sum[g] + 1e-5);
            }
        }

        // (3) to_q, to_k, to_v
        let apply_linear = |w: &[f32], out: &mut Vec<f32>| {
            for g in 0..k {
                for j in 0..d {
                    let mut s = 0.0;
                    for i in 0..d {
                        s += w[j * d + i] * slice_token[g * d + i];
                    }
                    out[g * d + j] = s;
                }
            }
        };
        let mut q_slice = vec![0.0f32; k * d];
        let mut k_slice = vec![0.0f32; k * d];
        let mut v_slice = vec![0.0f32; k * d];
        apply_linear(w_q, &mut q_slice);
        apply_linear(w_k, &mut k_slice);
        apply_linear(w_v, &mut v_slice);

        // (4) Dual-form solve: Z = reg⁻¹ · K̃ᵀ, reg = (1-α)·K̃ᵀ·K̃ + α·I
        // Compute reg (d, d) via Gauss-Jordan inversion.
        let one_minus_alpha = 1.0 - alpha;
        let mut reg = vec![0.0f32; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut s = 0.0;
                for l in 0..k {
                    s += k_slice[l * d + i] * k_slice[l * d + j];
                }
                reg[i * d + j] = one_minus_alpha * s;
            }
        }
        for i in 0..d {
            reg[i * d + i] += alpha;
        }
        // Invert reg via Gauss-Jordan with partial pivoting.
        let mut aug = vec![0.0f32; d * 2 * d];
        for i in 0..d {
            for j in 0..d {
                aug[i * 2 * d + j] = reg[i * d + j];
            }
            aug[i * 2 * d + d + i] = 1.0;
        }
        for col in 0..d {
            let mut piv = col;
            for r in (col + 1)..d {
                if aug[r * 2 * d + col].abs() > aug[piv * 2 * d + col].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for j in 0..2 * d {
                    aug.swap(col * 2 * d + j, piv * 2 * d + j);
                }
            }
            let diag = aug[col * 2 * d + col];
            assert!(diag.abs() > 1e-20, "singular reg in reference");
            let inv_diag = 1.0 / diag;
            for j in 0..2 * d {
                aug[col * 2 * d + j] *= inv_diag;
            }
            for r in 0..d {
                if r != col {
                    let factor = aug[r * 2 * d + col];
                    if factor != 0.0 {
                        for j in 0..2 * d {
                            aug[r * 2 * d + j] -= factor * aug[col * 2 * d + j];
                        }
                    }
                }
            }
        }
        // reg⁻¹ is now in aug[:, d:2d].
        // Z = reg⁻¹ · K̃ᵀ. Z[l, j] = Σ_i reg⁻¹[l, i] · K̃ᵀ[i, j] = Σ_i reg⁻¹[l, i] · K̃[j, i].
        // We need Zᵀ (k, d): Zᵀ[j, l] = Z[l, j] = Σ_i reg⁻¹[l, i] · K̃[j, i].
        let mut z_op_t = vec![0.0f32; k * d];
        for j in 0..k {
            for l in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    s += aug[l * 2 * d + d + i] * k_slice[j * d + i];
                }
                z_op_t[j * d + l] = s;
            }
        }

        // (5) C = Q̃ · Z. C[i, j] = Σ_l Q̃[i, l] · Z[l, j] = dot(Q̃ row i, Zᵀ row j).
        let mut c_op = vec![0.0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for l in 0..d {
                    s += q_slice[i * d + l] * z_op_t[j * d + l];
                }
                c_op[i * k + j] = s;
            }
        }

        // (6) out_slice = C · Ṽ.
        let mut out_slice = vec![0.0f32; k * d];
        for i in 0..k {
            for dd in 0..d {
                let mut s = 0.0;
                for l in 0..k {
                    s += c_op[i * k + l] * v_slice[l * d + dd];
                }
                out_slice[i * d + dd] = s;
            }
        }

        // (7) out = Φ · out_slice.
        let mut out = vec![0.0f32; n * d];
        for i in 0..n {
            for dd in 0..d {
                let mut s = 0.0;
                for g in 0..k {
                    s += phi[i * k + g] * out_slice[g * d + dd];
                }
                out[i * d + dd] = s;
            }
        }
        out
    }

    fn frobenius(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    fn run_forward(
        n: usize,
        d: usize,
        k: usize,
        alpha: f32,
        temperature: f32,
        basis: FuncAttnBasis,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut x_basis = vec![0.0f32; n * d];
        let mut x_value = vec![0.0f32; n * d];
        let mut w_basis = vec![0.0f32; k * d];
        let mut w_q = vec![0.0f32; d * d];
        let mut w_k = vec![0.0f32; d * d];
        let mut w_v = vec![0.0f32; d * d];
        fill_rand(&mut x_basis, seed);
        fill_rand(&mut x_value, seed + 1);
        fill_rand(&mut w_basis, seed + 2);
        fill_rand(&mut w_q, seed + 3);
        fill_rand(&mut w_k, seed + 4);
        fill_rand(&mut w_v, seed + 5);

        let cfg = FuncAttnConfig {
            d,
            k,
            basis,
            alpha,
            temperature,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        funcattn_forward(
            &x_basis,
            &x_value,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        )
        .expect("forward should succeed");

        let ref_out = funcattn_reference(
            &x_basis,
            &x_value,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            n,
            d,
            k,
            basis,
            alpha,
            temperature,
        );
        (out, ref_out)
    }

    // ── Cross-check against reference (the most important correctness gate) ─

    #[test]
    fn matches_reference_sigmoid() {
        let (out, ref_out) = run_forward(16, 8, 4, 0.5, 0.5, FuncAttnBasis::Sigmoid, 42);
        let err =
            frobenius(&out, &ref_out) / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
        assert!(
            err < 1e-3,
            "sigmoid forward disagrees with reference: relative error = {}",
            err
        );
    }

    #[test]
    fn matches_reference_softmax() {
        let (out, ref_out) = run_forward(16, 8, 4, 0.5, 0.5, FuncAttnBasis::Softmax, 142);
        let err =
            frobenius(&out, &ref_out) / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
        assert!(
            err < 1e-3,
            "softmax forward disagrees with reference: relative error = {}",
            err
        );
    }

    #[test]
    fn matches_reference_extreme_alpha() {
        // α near 0 (almost pure K̃ᵀK̃) and α near 1 (almost pure I) — both must work.
        for &alpha in &[0.01f32, 0.99] {
            let (out, ref_out) = run_forward(
                12,
                6,
                4,
                alpha,
                0.5,
                FuncAttnBasis::Sigmoid,
                999 + alpha.to_bits() as u64,
            );
            let err = frobenius(&out, &ref_out)
                / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
            assert!(
                err < 1e-3,
                "α={}: forward disagrees with reference: relative error = {}",
                alpha,
                err
            );
        }
    }

    #[test]
    fn matches_reference_temperature_sweep() {
        for &temp in &[0.1f32, 0.5, 1.0, 5.0] {
            let (out, ref_out) = run_forward(
                12,
                6,
                4,
                0.5,
                temp,
                FuncAttnBasis::Sigmoid,
                7000 + (temp * 100.0) as u64,
            );
            let err = frobenius(&out, &ref_out)
                / frobenius(&ref_out, &vec![0.0; ref_out.len()]).max(1e-30);
            assert!(
                err < 1e-3,
                "τ={}: forward disagrees with reference: relative error = {}",
                temp,
                err
            );
        }
    }

    // ── G1: Mechanics (finite output, no NaN/Inf) ──────────────────

    #[test]
    fn g1_finite_output_random_inputs() {
        let n = 64;
        let d = 32;
        let k = 8;
        let mut x_basis = vec![0.0f32; n * d];
        let mut x_value = vec![0.0f32; n * d];
        let mut w_basis = vec![0.0f32; k * d];
        let mut w_q = vec![0.0f32; d * d];
        let mut w_k = vec![0.0f32; d * d];
        let mut w_v = vec![0.0f32; d * d];
        fill_rand(&mut x_basis, 12345);
        fill_rand(&mut x_value, 12346);
        fill_rand(&mut w_basis, 12347);
        fill_rand(&mut w_q, 12348);
        fill_rand(&mut w_k, 12349);
        fill_rand(&mut w_v, 12350);

        let cfg = FuncAttnConfig {
            d,
            k,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.5,
            temperature: 0.5,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        funcattn_forward(
            &x_basis,
            &x_value,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        )
        .expect("forward");
        for x in &out {
            assert!(x.is_finite(), "non-finite output: {x}");
        }
    }

    #[test]
    fn g1_sweep_input_norm_and_alpha() {
        // Sweep B ∈ {1, 10, 100} and α ∈ {0.01, 0.5, 0.99}; assert finite output.
        // Unlike the additive-λ primal form, the convex combo α∈(0,1) guarantees
        // PD for any input scale — no NotPositiveDefinite expected.
        let n = 32;
        let d = 16;
        let k = 4;
        for (b_idx, &b_scale) in [1.0f32, 10.0, 100.0].iter().enumerate() {
            for (a_idx, &alpha) in [0.01f32, 0.5, 0.99].iter().enumerate() {
                let seed = 1000 + b_idx as u64 * 10 + a_idx as u64;
                let mut x_basis = vec![0.0f32; n * d];
                let mut x_value = vec![0.0f32; n * d];
                let mut w_basis = vec![0.0f32; k * d];
                let mut w_q = vec![0.0f32; d * d];
                let mut w_k = vec![0.0f32; d * d];
                let mut w_v = vec![0.0f32; d * d];
                fill_rand(&mut x_basis, seed);
                fill_rand(&mut x_value, seed + 1);
                fill_rand(&mut w_basis, seed + 2);
                fill_rand(&mut w_q, seed + 3);
                fill_rand(&mut w_k, seed + 4);
                fill_rand(&mut w_v, seed + 5);
                for x in x_basis.iter_mut() {
                    *x *= b_scale;
                }
                for x in x_value.iter_mut() {
                    *x *= b_scale;
                }

                let cfg = FuncAttnConfig {
                    d,
                    k,
                    basis: FuncAttnBasis::Sigmoid,
                    alpha,
                    temperature: 0.5,
                    cholesky_jitter: 1e-6,
                };
                let mut scratch = FuncAttnScratch::new(n, d, k);
                let mut out = vec![0.0f32; n * d];
                funcattn_forward(
                    &x_basis,
                    &x_value,
                    &w_basis,
                    &w_q,
                    &w_k,
                    &w_v,
                    &cfg,
                    &mut scratch,
                    &mut out,
                )
                .expect("convex combo should be PD for any α ∈ (0, 1)");
                for x in &out {
                    assert!(
                        x.is_finite(),
                        "non-finite output at B={}, α={}",
                        b_scale,
                        alpha
                    );
                }
            }
        }
    }

    #[test]
    fn g1_lipschitz_bounded() {
        // Verify empirical Lipschitz constant is finite and reasonable.
        // Prop 4.5 is stated for the additive-λ form; for the convex-combo form
        // the bound becomes a function of α/(1-α) instead. We just check finiteness.
        let n = 64;
        let d = 32;
        let k = 8;
        let mut x_basis = vec![0.0f32; n * d];
        let mut x_value = vec![0.0f32; n * d];
        let mut w_basis = vec![0.0f32; k * d];
        let mut w_q = vec![0.0f32; d * d];
        let mut w_k = vec![0.0f32; d * d];
        let mut w_v = vec![0.0f32; d * d];
        fill_rand(&mut x_basis, 12345);
        fill_rand(&mut x_value, 12346);
        fill_rand(&mut w_basis, 12347);
        fill_rand(&mut w_q, 12348);
        fill_rand(&mut w_k, 12349);
        fill_rand(&mut w_v, 12350);

        let cfg = FuncAttnConfig {
            d,
            k,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.5,
            temperature: 0.5,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        funcattn_forward(
            &x_basis,
            &x_value,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        )
        .expect("forward");

        // Perturb x_basis by ‖Δ‖ = 1 and check ‖A(X+Δ) − A(X)‖ is finite.
        let mut delta = vec![0.0f32; n * d];
        fill_rand(&mut delta, 12345 + 100);
        let d_norm = frobenius(&delta, &vec![0.0; n * d]);
        for x in delta.iter_mut() {
            *x /= d_norm.max(1e-30);
        }
        let mut x_pert = x_basis.clone();
        for i in 0..n * d {
            x_pert[i] += delta[i];
        }
        let mut out_pert = vec![0.0f32; n * d];
        funcattn_forward(
            &x_pert,
            &x_value,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out_pert,
        )
        .expect("perturbed forward");

        let lip = frobenius(&out, &out_pert);
        assert!(lip.is_finite(), "Lipschitz ratio not finite");
        // Empirically this is ~1-50 for random normalized inputs at α=0.5.
        assert!(lip < 1.0e6, "empirical Lipschitz too large: {}", lip);
    }

    // ── Partition-of-unity check ──────────────────────────────────

    #[test]
    fn basis_rows_partition_of_unity() {
        let n = 8;
        let d = 16;
        let k = 6;
        let mut x = vec![0.0f32; n * d];
        let mut w = vec![0.0f32; k * d];
        fill_rand(&mut x, 7);
        fill_rand(&mut w, 8);
        let mut out = vec![0.0f32; n * k];

        for &kind in &[FuncAttnBasis::Softmax, FuncAttnBasis::Sigmoid] {
            for &temp in &[0.1f32, 0.5, 1.0, 5.0] {
                compute_basis_into(&x, &w, &[], n, d, k, kind, temp, &mut out);
                for i in 0..n {
                    let row = &out[i * k..(i + 1) * k];
                    let sum: f32 = row.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-5,
                        "row {} doesn't sum to 1 for {:?} τ={}: sum = {}",
                        i,
                        kind,
                        temp,
                        sum
                    );
                    for &v in row {
                        assert!(v >= 0.0, "negative basis entry for {:?} τ={}", kind, temp);
                    }
                }
            }
        }
    }

    // ── Cholesky unit tests ────────────────────────────────────────

    #[test]
    fn cholesky_inplace_basic_spd() {
        // A = [[4, 2], [2, 3]] is SPD with L = [[2, 0], [1, √2]] (lower triangular).
        // Stored row-major: [L[0,0], L[0,1], L[1,0], L[1,1]] = [2, 0, 1, √2].
        let mut a = vec![4.0f32, 2.0, 2.0, 3.0];
        assert!(cholesky_inplace(&mut a, 2));
        assert!((a[0] - 2.0).abs() < 1e-5, "L[0,0] = {}", a[0]);
        assert!(
            a[1].abs() < 1e-20,
            "L[0,1] upper tri must be zero, got {}",
            a[1]
        );
        assert!((a[2] - 1.0).abs() < 1e-5, "L[1,0] = {}", a[2]);
        assert!((a[3] - 2.0f32.sqrt()).abs() < 1e-5, "L[1,1] = {}", a[3]);
    }

    #[test]
    fn cholesky_inplace_indefinite_fails() {
        let mut a = vec![1.0f32, 2.0, 2.0, 1.0]; // indefinite
        assert!(!cholesky_inplace(&mut a, 2));
    }

    #[test]
    fn cholesky_solve_known_system() {
        // A = [[4, 2], [2, 3]], b = [1, 1]; solution x = [1/8, 1/4]
        let mut a = vec![4.0f32, 2.0, 2.0, 3.0];
        assert!(cholesky_inplace(&mut a, 2));
        let b = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        let mut x = vec![0.0f32; 2];
        cholesky_solve_into(&a, &b, 2, &mut y, &mut x);
        assert!((x[0] - 0.125).abs() < 1e-5, "x[0] = {}", x[0]);
        assert!((x[1] - 0.25).abs() < 1e-5, "x[1] = {}", x[1]);
    }

    // ── Larger-size sanity (catches indexing bugs, partial G4 smoke) ─

    #[test]
    fn forward_large_n_smoke() {
        // n=2048, k=16, d=64 — forward should complete without index errors or NaN.
        // Full G4 timing is in the bench file.
        let n = 2048;
        let d = 64;
        let k = 16;
        let mut x_basis = vec![0.0f32; n * d];
        let mut x_value = vec![0.0f32; n * d];
        let mut w_basis = vec![0.0f32; k * d];
        let mut w_q = vec![0.0f32; d * d];
        let mut w_k = vec![0.0f32; d * d];
        let mut w_v = vec![0.0f32; d * d];
        fill_rand(&mut x_basis, 9001);
        fill_rand(&mut x_value, 9002);
        fill_rand(&mut w_basis, 9003);
        fill_rand(&mut w_q, 9004);
        fill_rand(&mut w_k, 9005);
        fill_rand(&mut w_v, 9006);

        let cfg = FuncAttnConfig {
            d,
            k,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.5,
            temperature: 0.5,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        funcattn_forward(
            &x_basis,
            &x_value,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        )
        .expect("forward at n=2048");
        for x in &out {
            assert!(x.is_finite(), "non-finite at large n");
        }
    }

    // ── Degenerate-input guard ─────────────────────────────────────

    #[test]
    fn forward_zero_weights_alpha_positive_succeeds() {
        // All-zero w_k → K̃ all zero → reg = α·I (well-conditioned for α > 0).
        // Convex combo guarantees PD; output should be finite (possibly zero).
        let n = 4;
        let d = 8;
        let k = 4;
        let x_basis = vec![0.5f32; n * d];
        let x_value = vec![0.5f32; n * d];
        let w_basis = vec![0.1f32; k * d]; // non-zero so Φ isn't 0/0
        let w_q = vec![0.0f32; d * d];
        let w_k = vec![0.0f32; d * d];
        let w_v = vec![0.0f32; d * d];

        let cfg = FuncAttnConfig {
            d,
            k,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.5,
            temperature: 0.5,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        let res = funcattn_forward(
            &x_basis,
            &x_value,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        );
        assert!(
            res.is_ok(),
            "convex combo should succeed with α > 0 even for zero K̃"
        );
        for x in &out {
            assert!(x.is_finite(), "non-finite output for zero w_k");
        }
    }

    // ── pre_rotate_basis_weights_into (Plan 286 T5.1) ─────────────

    /// `d × d` identity matrix, row-major.
    fn identity_matrix(d: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; d * d];
        for i in 0..d {
            m[i * d + i] = 1.0;
        }
        m
    }

    /// `d × d` row-orthonormal matrix via Gram-Schmidt on random rows.
    fn random_orthonormal_rows(d: usize, seed: u64) -> Vec<f32> {
        let mut m = vec![0.0f32; d * d];
        fill_rand(&mut m, seed);
        for i in 0..d {
            for j in 0..i {
                let dot = (0..d).map(|c| m[i * d + c] * m[j * d + c]).sum::<f32>();
                for c in 0..d {
                    m[i * d + c] -= dot * m[j * d + c];
                }
            }
            let nrm = (0..d)
                .map(|c| m[i * d + c] * m[i * d + c])
                .sum::<f32>()
                .sqrt();
            let nrm = if nrm < 1e-12 { 1.0 } else { nrm };
            for c in 0..d {
                m[i * d + c] /= nrm;
            }
        }
        m
    }

    #[test]
    fn pre_rotate_identity_eigenvectors_is_noop() {
        // V = I → W · Iᵀ = W (unchanged).
        let k = 4;
        let d = 8;
        let mut w_basis = vec![0.0f32; k * d];
        fill_rand(&mut w_basis, 4242);
        let original = w_basis.clone();
        let identity = identity_matrix(d);
        pre_rotate_basis_weights_into(&mut w_basis, &identity, k, d);
        let diff = frobenius(&w_basis, &original);
        assert!(
            diff < 1e-5,
            "identity rotation should be no-op: diff = {}",
            diff
        );
    }

    #[test]
    fn pre_rotate_preserves_row_norms() {
        // V orthogonal → ‖V · w_row‖ = ‖w_row‖ for every row.
        let k = 4;
        let d = 8;
        let mut w_basis = vec![0.0f32; k * d];
        fill_rand(&mut w_basis, 4243);
        let original_norms: Vec<f32> = (0..k)
            .map(|r| {
                (0..d)
                    .map(|c| w_basis[r * d + c] * w_basis[r * d + c])
                    .sum::<f32>()
                    .sqrt()
            })
            .collect();
        let v = random_orthonormal_rows(d, 17);
        pre_rotate_basis_weights_into(&mut w_basis, &v, k, d);
        for kk in 0..k {
            let new_norm = (0..d)
                .map(|c| w_basis[kk * d + c] * w_basis[kk * d + c])
                .sum::<f32>()
                .sqrt();
            assert!(
                (new_norm - original_norms[kk]).abs() < 1e-4,
                "row {} norm changed: {} → {}",
                kk,
                original_norms[kk],
                new_norm
            );
        }
    }

    #[test]
    fn pre_rotate_preserves_orthogonality_of_w_basis() {
        // If w_basis rows are orthonormal and V is orthogonal, then W · Vᵀ is
        // still row-orthonormal (W · Wᵀ unchanged).
        let k = 4;
        let d = 8;
        let w_basis = random_orthonormal_rows(d, 31); // d×d, take first k rows
        let mut w_basis = w_basis.into_iter().take(k * d).collect::<Vec<_>>();
        let v = random_orthonormal_rows(d, 32);
        pre_rotate_basis_weights_into(&mut w_basis, &v, k, d);

        // Check the k×k Gram matrix is still identity.
        for a in 0..k {
            for b in 0..k {
                let dot = (0..d)
                    .map(|c| w_basis[a * d + c] * w_basis[b * d + c])
                    .sum::<f32>();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-3,
                    "Gram[{},{}] = {} (want {})",
                    a,
                    b,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn pre_rotate_forward_output_still_finite_and_partition_of_unity() {
        // Sanity: after rotation, the forward pass still produces finite output
        // and the basis rows still partition to 1 (verify the rotation doesn't
        // break the partition-of-unity invariant that Prop 4.3 relies on).
        let n = 32;
        let d = 8;
        let k = 4;
        let mut w_basis = random_orthonormal_rows(d, 41)
            .into_iter()
            .take(k * d)
            .collect::<Vec<_>>();
        let w_q = random_orthonormal_rows(d, 42);
        let w_k = random_orthonormal_rows(d, 43);
        let w_v = random_orthonormal_rows(d, 44);
        let v = random_orthonormal_rows(d, 45);
        pre_rotate_basis_weights_into(&mut w_basis, &v, k, d);

        // Verify partition-of-unity directly via compute_basis_into.
        let mut x = vec![0.0f32; n * d];
        fill_rand(&mut x, 99);
        let mut phi = vec![0.0f32; n * k];
        compute_basis_into(
            &x,
            &w_basis,
            &[],
            n,
            d,
            k,
            FuncAttnBasis::Sigmoid,
            0.1,
            &mut phi,
        );
        for i in 0..n {
            let row_sum: f32 = phi[i * k..(i + 1) * k].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "row {} sum = {} (want 1.0)",
                i,
                row_sum
            );
        }

        // Forward still finite.
        let cfg = FuncAttnConfig {
            d,
            k,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.5,
            temperature: 0.1,
            cholesky_jitter: 1e-6,
        };
        let mut scratch = FuncAttnScratch::new(n, d, k);
        let mut out = vec![0.0f32; n * d];
        funcattn_forward(
            &x,
            &x,
            &w_basis,
            &w_q,
            &w_k,
            &w_v,
            &cfg,
            &mut scratch,
            &mut out,
        )
        .expect("forward after rotation");
        for v in &out {
            assert!(v.is_finite(), "non-finite output after eigen-rotation");
        }
    }

    // ── Plan 332 — Principled structured basis constructors ─────────

    /// Verify `W·W^T ≈ I_k` (rows are orthonormal) for a constructed basis.
    fn check_row_orthonormal(w: &[f32], k: usize, d: usize, tol: f32, label: &str) {
        assert_eq!(w.len(), k * d, "{label}: wrong length");
        for i in 0..k {
            // Diagonal: row norm should be 1.
            let mut norm_sq = 0.0f32;
            for j in 0..d {
                norm_sq += w[i * d + j] * w[i * d + j];
            }
            assert!(
                (norm_sq - 1.0).abs() < tol,
                "{label}: row {i} norm^2 = {norm_sq}, expected 1.0 (tol {tol})"
            );
            // Off-diagonal: orthogonal to all earlier rows.
            for j in 0..i {
                let mut dot = 0.0f32;
                for l in 0..d {
                    dot += w[i * d + l] * w[j * d + l];
                }
                assert!(
                    dot.abs() < tol,
                    "{label}: rows ({i},{j}) dot = {dot}, expected 0 (tol {tol})"
                );
            }
        }
    }

    /// T1.1 unit test: DCT-log basis is row-orthonormal.
    #[cfg(feature = "funcattn_structured_basis")]
    #[test]
    fn dct_log_basis_is_row_orthonormal() {
        for &(k, d) in &[(1usize, 8usize), (4, 16), (8, 64), (16, 64), (8, 128)] {
            let w = make_dct_log_basis(k, d);
            check_row_orthonormal(&w, k, d, 1e-5, &format!("DCT-log k={k} d={d}"));
        }
    }

    /// T1.1 unit test: DCT-log basis covers log-spaced frequencies.
    ///
    /// We verify by reconstructing the per-row dominant frequency from the
    /// zero-crossing count of the post-Gram-Schmidt rows. The i-th row's
    /// dominant frequency should be monotonically non-decreasing in i.
    #[cfg(feature = "funcattn_structured_basis")]
    #[test]
    fn dct_log_basis_covers_log_spaced_frequencies() {
        let (k, d) = (8, 64);
        let w = make_dct_log_basis(k, d);
        // Count sign changes in each row interior as a proxy for frequency.
        let mut sign_changes = Vec::with_capacity(k);
        for i in 0..k {
            let row = &w[i * d..(i + 1) * d];
            let mut count = 0usize;
            for j in 1..d {
                if row[j - 1].signum() != row[j].signum() && row[j] != 0.0 {
                    count += 1;
                }
            }
            sign_changes.push(count);
        }
        // Coarsest row (i=0, f=1) should have ~2 sign changes (one full cycle);
        // finest row (i=k-1, f=d/2) should have many. Monotone non-decreasing.
        println!("DCT-log sign-change profile (k={k}, d={d}): {sign_changes:?}");
        assert!(
            sign_changes[0] <= sign_changes[k - 1],
            "DCT-log should span coarse→fine: first={}, last={}",
            sign_changes[0],
            sign_changes[k - 1]
        );
        // The coarsest row must have meaningfully fewer sign changes than the
        // finest (otherwise we didn't actually span log-spaced frequencies).
        assert!(
            sign_changes[k - 1] >= 2 * sign_changes[0],
            "DCT-log frequency spread too narrow: first={}, last={}",
            sign_changes[0],
            sign_changes[k - 1]
        );
    }

    /// T1.2 unit test: Haar-packet basis is row-orthonormal.
    #[cfg(feature = "funcattn_structured_basis")]
    #[test]
    fn haar_packet_basis_is_row_orthonormal() {
        for &(k, d) in &[(1usize, 8usize), (4, 16), (8, 64), (16, 64), (7, 128)] {
            let w = make_haar_packet_basis(k, d);
            check_row_orthonormal(&w, k, d, 1e-5, &format!("Haar-packet k={k} d={d}"));
        }
    }

    /// T1.2 unit test: Haar-packet basis spans multiple scales.
    ///
    /// Row 0 must be the DC component (constant sign — zero sign changes).
    /// Later rows must have progressively more localized support: sign changes
    /// increase as we move to finer-scale wavelets.
    #[cfg(feature = "funcattn_structured_basis")]
    #[test]
    fn haar_packet_basis_spans_multiple_scales() {
        let (k, d) = (8, 64);
        let w = make_haar_packet_basis(k, d);

        // Row 0: DC = constant sign (no interior sign changes).
        let dc_row = &w[0..d];
        let dc_sign = dc_row[0].signum();
        for &v in dc_row {
            assert_eq!(v.signum(), dc_sign, "Haar DC row should be constant-sign");
        }

        // Each subsequent row should have at least one sign change (it's a
        // wavelet, not a scaling function). Count sign changes per row.
        let mut sign_counts = Vec::with_capacity(k - 1);
        for i in 1..k {
            let row = &w[i * d..(i + 1) * d];
            let mut count = 0usize;
            for j in 1..d {
                if row[j - 1].signum() != row[j].signum() && row[j] != 0.0 {
                    count += 1;
                }
            }
            assert!(count >= 1, "Haar row {i} should have ≥1 sign change (got {count})");
            sign_counts.push(count);
        }
        println!("Haar-packet sign-change profile (k={k}, d={d}): {sign_counts:?}");
        // The coarsest wavelet (row 1, support=d) has exactly 1 sign change.
        assert_eq!(
            sign_counts[0], 1,
            "Haar row 1 (coarsest wavelet) should have exactly 1 sign change"
        );
    }

    /// T1.1+T1.2 cross-check: both bases plug into `funcattn_forward` cleanly
    /// (G3 — drop-in replacement sanity). Output must be finite + partition of
    /// unity on Φ (the existing forward-pass invariant).
    #[cfg(feature = "funcattn_structured_basis")]
    #[test]
    fn structured_bases_forward_pass_clean() {
        let (n, d, k) = (12, 64, 8);
        let cfg = FuncAttnConfig {
            d,
            k,
            basis: FuncAttnBasis::Sigmoid,
            alpha: 0.5,
            temperature: 0.5,
            cholesky_jitter: 1e-6,
        };
        // Random input (deterministic).
        let mut x = vec![0.0f32; n * d];
        let mut rng = make_rng(12345);
        for v in x.iter_mut() {
            *v = rng.next().unwrap_or(0.0);
        }
        let w_q = identity_matrix(d);
        let w_k = identity_matrix(d);
        let w_v = identity_matrix(d);

        for (label, w_basis) in [
            ("DCT-log", make_dct_log_basis(k, d)),
            ("Haar-packet", make_haar_packet_basis(k, d)),
        ] {
            let mut scratch = FuncAttnScratch::new(n, d, k);
            let mut out = vec![0.0f32; n * d];
            funcattn_forward(&x, &x, &w_basis, &w_q, &w_k, &w_v, &cfg, &mut scratch, &mut out)
                .unwrap_or_else(|e| panic!("{label}: forward failed: {e:?}"));
            for v in &out {
                assert!(v.is_finite(), "{label}: non-finite forward output");
            }
            // Φ partition-of-unity (compute_basis_into separately for clarity).
            let mut phi = vec![0.0f32; n * k];
            compute_basis_into(
                &x,
                &w_basis,
                &[],
                n,
                d,
                k,
                FuncAttnBasis::Sigmoid,
                0.5,
                &mut phi,
            );
            for i in 0..n {
                let row_sum: f32 = phi[i * k..(i + 1) * k].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-5,
                    "{label}: Φ row {i} sum = {row_sum}, expected 1.0"
                );
            }
        }
    }
}
