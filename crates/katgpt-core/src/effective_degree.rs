//! Issue 668 — `effective_degree`: a modelless **function-space** simplicity
//! metric (Research 488 / arXiv:2605.29823, Zhang, Li, Xiao, Chen & Chen,
//! *Quantifying and Optimizing Simplicity via Polynomial Representations*,
//! ICML 2026).
//!
//! # What it measures
//!
//! Restrict an arbitrary frozen function `f: ℝᵈ → ℝᵐ` to a 1-D **interpolation
//! path** between two real data points,
//!
//! ```text
//! x(α) = α·a + (1−α)·b ,   α ∈ [0, 1]
//! ```
//!
//! sample `f` at `r` nodes along that path, fit a degree-`K` Chebyshev
//! expansion `P(α) = Σₖ cₖ·Tₖ(2α−1)` by damped least squares, and report the
//! coefficient-weighted degree
//!
//! ```text
//! ED(P)      = Σₖ ‖cₖ‖·k
//! ED_norm(P) = Σₖ ‖cₖ‖·k / Σₖ ‖cₖ‖
//! ```
//!
//! ED is **Lipschitz in the coefficients** (unlike algebraic degree, which is a
//! discontinuous 0/1 read on the leading term), so it is robust to fitting
//! noise. Paper Theorem 3.1 (order preservation) is what licenses the 1-D
//! surrogate: for multivariate polynomials, degree drops under path restriction
//! occur only on a measure-zero set of directions, so averaging over random
//! data-anchored paths preserves degree ordering almost surely.
//!
//! Everything here is **modelless** — no training, no gradients. The metric
//! probes a function that is already frozen (a shard decode, a KARC readout, a
//! LoRA overlay, an NPC policy). The paper's differentiable ED *regularizer*
//! (its §7) is training-only and is explicitly out of scope — see Research 488
//! §7 for the riir-train record.
//!
//! # Four caveats that decide whether a number here means anything
//!
//! 1. **Data-manifold dependence (paper C.1).** The path endpoints MUST be real
//!    data pairs. With random-noise endpoints the ED signal collapses to
//!    baseline — the paper measures this directly. Feed real wake events, real
//!    observed states, real prompts; never `rand()` vectors.
//! 2. **Scale dependence (paper Table 12).** Raw [`EdResult::ed`] scales
//!    linearly with output magnitude (×2 outputs ⇒ ×2 ED). Compare functions
//!    only at a common output scale, or use [`EdResult::ed_norm`], which is
//!    invariant under `y ↦ s·y`. The paper's own regularizer fits post-softmax
//!    probabilities for exactly this reason.
//!
//! 3. **DC offset drags `ed_norm` down.** `ED_norm` is a magnitude-weighted
//!    mean of the degree index over **all** coefficients including `k = 0`, so
//!    a function with a large constant term reads lower than its algebraic
//!    degree — measured in Bench 665: a degree-5 restriction with a natural DC
//!    term reads `ed_norm = 1.15`, while the same coefficients with `c₀` zeroed
//!    read `1.63`. Ordering across functions is unaffected (that is the paper's
//!    claim and it holds), but do not read `ed_norm` as "the algebraic degree".
//!    For an offset-free read, zero [`EdResult::coeff_norms`]`[0]` and re-reduce
//!    through [`ed_from_coeff_norms`]; that quantity is bounded in `[1, p]` for
//!    a degree-`p` restriction.
//!
//! 4. **The sign is grain-dependent — verify it at YOUR grain.** Measured on
//!    real substrate (riir-neuron-db Bench 484, 360 shard states): ED correlates
//!    with the generalization gap at **+0.598 pooled across regimes** but
//!    **negative within every individual regime** (−0.18, −0.68, −0.35, −0.25) —
//!    a Simpson reversal reproduced on 3 disjoint seed sets. A one-sided
//!    threshold `ed_norm < τ` therefore does the right thing *between* regimes
//!    and the exactly wrong thing *inside* one, and no choice of `τ` fixes it.
//!    The paper's own correlation study measures only the across-model grain, so
//!    this inversion is unmeasured upstream. **State which grain you operate at
//!    and measure the sign there before wiring ED into any gate.**
//!
//! A fifth, softer caveat (paper §7 MNIST-CIFAR failure): ED enforces/measures
//! *simplicity*, not *correctness*. A wrong-but-simple decode passes an ED
//! gate. Any consumer gate must keep a correctness arm alongside ED.
//!
//! # Substrate reuse
//!
//! - Chebyshev evaluation: [`crate::karc::ChebyshevBasis`] (the sealed
//!   `KarcBasis` dictionary shipped for KARC forecasting — same basis, orthogonal
//!   purpose, so this module consumes it rather than re-deriving the recurrence).
//! - Damped normal equations: [`crate::linalg::cholesky_f64`] +
//!   [`crate::linalg::chol_solve_f64`]. The Gram is accumulated in **f64** for
//!   the same reason KARC's `fit_direct` does — f32 Cholesky is fragile when the
//!   damping `ε` falls below f32 epsilon relative to the matrix scale.
//!
//! Only the ED reduction and the randomized-cosine node sampler are new.
//!
//! # Why Chebyshev only
//!
//! Paper §3.2 reports Legendre works equally well and Appendix I shows the
//! degree *ordering* is preserved across the basis swap. This module ships the
//! Chebyshev path because that basis already exists in-tree; the ordering
//! invariance is a *gate*, not a runtime knob, and is exercised in
//! `tests/bench_668_effective_degree_goat.rs` by fitting an explicit Legendre
//! design matrix and reducing it through the shipped [`ed_from_coeff_norms`].
//!
//! # Allocation
//!
//! [`effective_degree_along_path`] (scalar outputs) is fully stack-resident —
//! it takes no scratch and allocates nothing. The vector-output and
//! multi-path entry points take a caller-owned [`EdScratch`], allocated once
//! and reused; steady state is zero-alloc (G4).
//!
//! Feature: `effective_degree` (opt-in, implies `karc_forecaster`). **Consumer
//! verdict is in and the gate did not change:** riir-neuron-db Issue 602's
//! freeze-gate PoC closed SCOPE-LIMITED 2026-08-17 (Bench 484). ED out-predicts
//! the incumbent `can_freeze::output_flatness` 12.6× pooled (0.598 vs 0.047) but
//! is not wirable as a one-sided freeze gate because of caveat 4, and its power
//! there traces to the DC term (caveat 3), not to shape complexity. Stays
//! **opt-in as a diagnostic surface**; the live consumer axis is cross-regime
//! triage — the KARC regime-mismatch probe of Research 488 §4, which is the
//! grain where the sign is correct.
//!
//! Effective Degree — function-space simplicity via polynomial representations
//! along data-anchored interpolation paths (Research 488 / arXiv:2605.29823,
//! Issue 668). Modelless measurement only; the paper's differentiable
//! regularizer is training-side (riir-train). Reuses `karc::ChebyshevBasis` +
//! `linalg`'s damped Cholesky, so the feature implies `karc_forecaster`.
//! Opt-in: promotion to default is blocked on a consumer verdict
//! (riir-neuron-db Issue 602 freeze-gate PoC — does ED beat `output_flatness`?).

use crate::karc::{ChebyshevBasis, KarcBasis};
use crate::linalg::{chol_solve_f64, cholesky_f64};

/// Number of Chebyshev terms the fixed-size solve supports: `T₀ … T₇`.
///
/// The paper's expensive correlation-grade configuration uses `K = 40`; its
/// *cheap* configuration — the one relevant to an inference-time probe — uses
/// `K = 3`, and the "performance" configuration `K = 7`. Capping at 8 terms
/// keeps the entire normal-equation solve in fixed-size arrays (an 8×8
/// Cholesky), which is what makes the scalar path allocation-free.
pub const MAX_ED_TERMS: usize = 8;

/// Highest Chebyshev degree representable: `MAX_ED_TERMS − 1 = 7`.
pub const MAX_ED_DEGREE: usize = MAX_ED_TERMS - 1;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Every way an ED request can be rejected before any arithmetic runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdError {
    /// `max_degree > MAX_ED_DEGREE` — the fixed-size solve cannot represent it.
    DegreeTooLarge { max_degree: usize },
    /// Fewer sample nodes than fitted terms — the fit would be rank-deficient
    /// and ED would read the damping rather than the function.
    ResolutionTooSmall { resolution: usize, n_terms: usize },
    /// `damping ≤ 0`. The damped normal equations need `ε > 0` to guarantee a
    /// positive-definite Gram (same precondition as [`crate::linalg`]).
    NonPositiveDamping,
    /// `n_pairs == 0` — nothing to average over.
    ZeroPairs,
    /// `out_dim == 0` — no outputs to fit.
    ZeroOutDim,
    /// A slice length did not match the shape implied by the config.
    LengthMismatch { expected: usize, got: usize },
    /// The supplied [`EdScratch`] was built for a smaller shape.
    ScratchTooSmall { need: usize, got: usize },
}

impl core::fmt::Display for EdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DegreeTooLarge { max_degree } => {
                write!(
                    f,
                    "max_degree {max_degree} exceeds MAX_ED_DEGREE {MAX_ED_DEGREE}"
                )
            }
            Self::ResolutionTooSmall {
                resolution,
                n_terms,
            } => write!(
                f,
                "resolution {resolution} < n_terms {n_terms} (rank-deficient fit)"
            ),
            Self::NonPositiveDamping => write!(f, "damping must be > 0"),
            Self::ZeroPairs => write!(f, "n_pairs must be > 0"),
            Self::ZeroOutDim => write!(f, "out_dim must be > 0"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::ScratchTooSmall { need, got } => {
                write!(f, "scratch too small: need {need}, got {got}")
            }
        }
    }
}

impl std::error::Error for EdError {}

// ── Config ───────────────────────────────────────────────────────────────────

/// Fit configuration: `r` nodes, degree `K`, damping `ε`, path count, seed.
///
/// Use [`EdConfig::cheap`] / [`EdConfig::precise`] rather than hand-rolling —
/// they mirror the paper's efficiency and performance configurations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdConfig {
    /// `r` — sample nodes per interpolation path. Must be `≥ max_degree + 1`.
    pub resolution: usize,
    /// `K` — highest Chebyshev degree fitted. Must be `≤ MAX_ED_DEGREE`.
    pub max_degree: usize,
    /// `ε` — ridge damping added to the Gram diagonal. Must be `> 0`.
    pub damping: f32,
    /// Number of interpolation paths (data pairs) averaged by [`ed_over_pairs`].
    pub n_pairs: usize,
    /// Seed for [`randomized_cosine_nodes`]. Same seed ⇒ bit-identical nodes.
    pub seed: u64,
}

impl EdConfig {
    /// Paper's **efficiency** configuration: `r = 4`, `K = 3`.
    ///
    /// This is the shape used by the paper's regularizer (one extra forward
    /// batch, ~2× training cost) and is the right default for an inference-time
    /// probe: four decode evaluations per path.
    pub const fn cheap() -> Self {
        Self {
            resolution: 4,
            max_degree: 3,
            damping: 1e-6,
            n_pairs: 8,
            seed: 0x0066_8EDC_4EA9,
        }
    }

    /// Paper's **performance** configuration: `r = 15`, `K = 7`.
    ///
    /// Note this is still far below the paper's *correlation-grade* estimate
    /// (`r = 200`, `K = 40`, 400 path averages), which is a research-scale
    /// measurement, not a runtime probe, and exceeds [`MAX_ED_TERMS`].
    pub const fn precise() -> Self {
        Self {
            resolution: 15,
            max_degree: 7,
            damping: 1e-6,
            n_pairs: 32,
            seed: 0x0066_8ED9_EC15,
        }
    }

    /// Number of fitted coefficients, `K + 1`.
    #[inline]
    pub const fn n_terms(&self) -> usize {
        self.max_degree + 1
    }

    /// Reject configurations the fixed-size solve cannot honour.
    pub const fn validate(&self) -> Result<(), EdError> {
        if self.max_degree > MAX_ED_DEGREE {
            return Err(EdError::DegreeTooLarge {
                max_degree: self.max_degree,
            });
        }
        let n_terms = self.n_terms();
        if self.resolution < n_terms {
            return Err(EdError::ResolutionTooSmall {
                resolution: self.resolution,
                n_terms,
            });
        }
        if self.damping <= 0.0 || self.damping.is_nan() {
            return Err(EdError::NonPositiveDamping);
        }
        Ok(())
    }
}

impl Default for EdConfig {
    fn default() -> Self {
        Self::cheap()
    }
}

// ── Result ───────────────────────────────────────────────────────────────────

/// Outcome of one ED measurement (single path, or averaged over paths).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdResult {
    /// `ED = Σₖ ‖cₖ‖·k`. **Scales with output magnitude** — see the module
    /// doc's scale-dependence caveat.
    pub ed: f32,
    /// `ED_norm = Σₖ ‖cₖ‖·k / Σₖ ‖cₖ‖`, invariant under `y ↦ s·y`. Zero when
    /// the fit is identically zero. Bounded in `[0, max_degree]`.
    pub ed_norm: f32,
    /// Per-degree coefficient magnitudes, `coeff_norms[k] = ‖cₖ‖₂` over the
    /// output dimensions. For scalar outputs this is `|cₖ|`. Entries at index
    /// `≥ n_terms` are zero. Averaged over paths by [`ed_over_pairs`].
    pub coeff_norms: [f32; MAX_ED_TERMS],
    /// `K + 1` — how many entries of `coeff_norms` are meaningful.
    pub n_terms: usize,
}

impl EdResult {
    /// All-zero result for `n_terms` coefficients (the constant-zero function).
    const fn zeroed(n_terms: usize) -> Self {
        Self {
            ed: 0.0,
            ed_norm: 0.0,
            coeff_norms: [0.0; MAX_ED_TERMS],
            n_terms,
        }
    }
}

/// Reduce per-degree coefficient magnitudes to `(ED, ED_norm)`.
///
/// `coeff_norms[k]` is the magnitude of the degree-`k` coefficient — for
/// vector-valued outputs, `‖cₖ‖₂` across output dimensions. The slice index
/// **is** the degree, so this reducer is basis-agnostic: it applies verbatim to
/// a Legendre, monomial, or any other degree-graded expansion. That is what
/// makes the paper's Appendix I basis-invariance check expressible against the
/// shipped metric rather than a test-local reimplementation.
///
/// Returns `(0.0, 0.0)` when every coefficient is zero.
#[inline]
pub fn ed_from_coeff_norms(coeff_norms: &[f32]) -> (f32, f32) {
    let mut weighted = 0.0f32;
    let mut total = 0.0f32;
    for (k, &c) in coeff_norms.iter().enumerate() {
        let c = c.abs();
        total += c;
        weighted = c.mul_add(k as f32, weighted);
    }
    if total > 0.0 {
        (weighted, weighted / total)
    } else {
        (0.0, 0.0)
    }
}

// ── Node sampling ────────────────────────────────────────────────────────────

/// splitmix64 (Vigna). Deterministic, seed-addressable, no dependency.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `[0, 1)` from a splitmix64 stream (53-bit mantissa, then narrowed).
#[inline]
fn next_u01(state: &mut u64) -> f32 {
    ((splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64) as f32
}

/// Per-path seed: mixes the base seed with the path index so paths get
/// independent node sets while the whole run stays reproducible from `cfg.seed`.
#[inline]
fn path_seed(base: u64, path: usize) -> u64 {
    let mut s = base ^ (path as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    splitmix64(&mut s)
}

/// **Randomized cosine sampling** (paper Eq. 8): stratified
/// `θᵢ ~ U[(i−1)π/r, iπ/r]`, `αᵢ = (1 − cos θᵢ)/2`.
///
/// Writes `resolution` values into `out[..resolution]` (extra capacity is left
/// untouched). Output is strictly increasing in `[0, 1]` because `θ ↦ (1−cos θ)/2`
/// is monotone on `[0, π]` and the strata are disjoint and ordered.
///
/// This is the stochastic sibling of fixed Chebyshev nodes: it keeps the
/// endpoint-clustered density that makes the Chebyshev fit well-conditioned
/// while decorrelating the node grid across paths, so averaging over paths does
/// not inherit a single grid's aliasing. Deterministic given `seed`.
///
/// Zero-allocation. Returns [`EdError::LengthMismatch`] if `out` is shorter
/// than `resolution`.
pub fn randomized_cosine_nodes(
    resolution: usize,
    seed: u64,
    out: &mut [f32],
) -> Result<(), EdError> {
    if out.len() < resolution {
        return Err(EdError::LengthMismatch {
            expected: resolution,
            got: out.len(),
        });
    }
    if resolution == 0 {
        return Ok(());
    }
    let mut state = seed;
    let step = core::f32::consts::PI / resolution as f32;
    for (i, slot) in out[..resolution].iter_mut().enumerate() {
        let u = next_u01(&mut state);
        let theta = (i as f32 + u) * step;
        *slot = 0.5 * (1.0 - theta.cos());
    }
    Ok(())
}

// ── Scratch ──────────────────────────────────────────────────────────────────

/// Caller-owned buffers for the vector-output and multi-path entry points.
///
/// Allocate once via [`EdScratch::new`], then reuse across calls — steady state
/// is allocation-free. The scalar single-path entry point
/// ([`effective_degree_along_path`]) needs none of this and takes no scratch.
#[derive(Clone, Debug)]
pub struct EdScratch {
    /// `Tᵀy`, then overwritten with nothing — shape `MAX_ED_TERMS × out_dim`.
    rhs: Vec<f64>,
    /// Solved coefficients, shape `MAX_ED_TERMS × out_dim`.
    coef: Vec<f64>,
    /// Triangular-solve intermediate, shape `MAX_ED_TERMS × out_dim`.
    z: Vec<f64>,
    /// Sampled `α` nodes, length `resolution`.
    nodes: Vec<f32>,
    /// Interpolated input point `x(α)`, length `in_dim`.
    x_buf: Vec<f32>,
    /// Decoded outputs along one path, shape `resolution × out_dim`.
    outputs: Vec<f32>,
    resolution: usize,
    in_dim: usize,
    out_dim: usize,
}

impl EdScratch {
    /// Allocate for `cfg.resolution` nodes over a `in_dim → out_dim` decode.
    ///
    /// Pass `in_dim = 0` when the scratch is only used with
    /// [`effective_degree_along_path_multi`] (no path construction).
    pub fn new(cfg: &EdConfig, in_dim: usize, out_dim: usize) -> Self {
        let n = MAX_ED_TERMS * out_dim.max(1);
        Self {
            rhs: vec![0.0; n],
            coef: vec![0.0; n],
            z: vec![0.0; n],
            nodes: vec![0.0; cfg.resolution],
            x_buf: vec![0.0; in_dim],
            outputs: vec![0.0; cfg.resolution * out_dim.max(1)],
            resolution: cfg.resolution,
            in_dim,
            out_dim: out_dim.max(1),
        }
    }

    /// `(resolution, in_dim, out_dim)` this scratch was built for.
    #[inline]
    pub const fn shape(&self) -> (usize, usize, usize) {
        (self.resolution, self.in_dim, self.out_dim)
    }

    /// The sampled `α` nodes from the most recent [`ed_over_pairs`] path — a
    /// diagnostic read, not part of the metric.
    #[inline]
    pub fn nodes(&self) -> &[f32] {
        &self.nodes
    }

    #[inline]
    fn check(&self, resolution: usize, in_dim: usize, out_dim: usize) -> Result<(), EdError> {
        match (
            self.resolution >= resolution,
            self.in_dim >= in_dim,
            self.out_dim >= out_dim,
        ) {
            (false, _, _) => Err(EdError::ScratchTooSmall {
                need: resolution,
                got: self.resolution,
            }),
            (_, false, _) => Err(EdError::ScratchTooSmall {
                need: in_dim,
                got: self.in_dim,
            }),
            (_, _, false) => Err(EdError::ScratchTooSmall {
                need: out_dim,
                got: self.out_dim,
            }),
            _ => Ok(()),
        }
    }
}

// ── The fit ──────────────────────────────────────────────────────────────────

/// Accumulate the damped normal equations `(TᵀT + εI)c = Tᵀy` and solve.
///
/// The design matrix `T[i,k] = Tₖ(2αᵢ − 1)` is never materialised: rows are
/// streamed through the `MAX_ED_TERMS`-wide basis evaluation and folded
/// straight into the Gram and RHS. That keeps the working set at
/// `n_terms² + n_terms·out_dim` regardless of `r`.
///
/// Gram/RHS accumulate in **f64** for the same reason KARC's `fit_direct` does:
/// f32 Cholesky is fragile once `ε` drops below f32 epsilon relative to the
/// matrix scale, and the default `ε = 1e-6` is squarely in that regime.
///
/// Writes `‖cₖ‖₂` (over output dims) into `coeff_norms[..n_terms]`.
#[allow(clippy::too_many_arguments)] // (data, shape, fit params, 3 scratch buffers, out) — all intrinsic to a zero-alloc solve
fn fit_coeff_norms(
    outputs: &[f32],
    out_dim: usize,
    nodes: &[f32],
    n_terms: usize,
    damping: f32,
    rhs: &mut [f64],
    coef: &mut [f64],
    z: &mut [f64],
    coeff_norms: &mut [f32; MAX_ED_TERMS],
) {
    let basis = ChebyshevBasis::<MAX_ED_TERMS>::new();
    let mut gram = [0.0f64; MAX_ED_TERMS * MAX_ED_TERMS];
    let mut chol = [0.0f64; MAX_ED_TERMS * MAX_ED_TERMS];
    let mut psi = [0.0f32; MAX_ED_TERMS];

    rhs[..n_terms * out_dim].fill(0.0);

    for (i, &alpha) in nodes.iter().enumerate() {
        // Chebyshev lives on [-1, 1]; the path parameter lives on [0, 1].
        basis.eval_into(2.0f32.mul_add(alpha, -1.0), &mut psi);
        let y = &outputs[i * out_dim..(i + 1) * out_dim];
        for j in 0..n_terms {
            let pj = psi[j] as f64;
            // Full square (not just the upper triangle): `cholesky_f64` reads
            // both `a[j*k+j]` and `a[i*k+j]` for i > j.
            for l in 0..n_terms {
                gram[j * n_terms + l] = (psi[l] as f64).mul_add(pj, gram[j * n_terms + l]);
            }
            for (o, &yv) in y.iter().enumerate() {
                rhs[j * out_dim + o] = (yv as f64).mul_add(pj, rhs[j * out_dim + o]);
            }
        }
    }

    let eps = damping as f64;
    for j in 0..n_terms {
        gram[j * n_terms + j] += eps;
    }

    cholesky_f64(&mut chol, &gram, n_terms);
    chol_solve_f64(coef, z, &chol, rhs, n_terms, out_dim);

    *coeff_norms = [0.0; MAX_ED_TERMS];
    for (k, slot) in coeff_norms.iter_mut().enumerate().take(n_terms) {
        let row = &coef[k * out_dim..(k + 1) * out_dim];
        let sq: f64 = row.iter().map(|&c| c * c).sum();
        *slot = sq.sqrt() as f32;
    }
}

/// Shared shape validation for the two `_along_path` entry points.
fn validate_path(
    outputs: &[f32],
    out_dim: usize,
    nodes: &[f32],
    cfg: &EdConfig,
) -> Result<usize, EdError> {
    cfg.validate()?;
    if out_dim == 0 {
        return Err(EdError::ZeroOutDim);
    }
    let n_terms = cfg.n_terms();
    if nodes.len() < n_terms {
        return Err(EdError::ResolutionTooSmall {
            resolution: nodes.len(),
            n_terms,
        });
    }
    let expected = nodes.len() * out_dim;
    if outputs.len() != expected {
        return Err(EdError::LengthMismatch {
            expected,
            got: outputs.len(),
        });
    }
    Ok(n_terms)
}

/// ED of a **scalar-valued** function sampled along one interpolation path.
///
/// `outputs[i]` is `f(x(nodes[i]))`; `nodes` are the `α` values (typically from
/// [`randomized_cosine_nodes`], but any `α ∈ [0,1]` grid works — fixed Chebyshev
/// nodes included). `cfg.resolution` and `cfg.n_pairs` are ignored here; the
/// node count comes from `nodes.len()`.
///
/// **Allocation-free** — the whole `(K+1)²` solve lives in fixed-size stack
/// arrays, which is what [`MAX_ED_TERMS`] buys.
///
/// # Errors
///
/// [`EdError::ResolutionTooSmall`] when `nodes.len() < K+1`,
/// [`EdError::LengthMismatch`] when `outputs.len() != nodes.len()`, plus
/// anything [`EdConfig::validate`] rejects.
pub fn effective_degree_along_path(
    outputs: &[f32],
    nodes: &[f32],
    cfg: &EdConfig,
) -> Result<EdResult, EdError> {
    let n_terms = validate_path(outputs, 1, nodes, cfg)?;
    let mut rhs = [0.0f64; MAX_ED_TERMS];
    let mut coef = [0.0f64; MAX_ED_TERMS];
    let mut z = [0.0f64; MAX_ED_TERMS];
    let mut coeff_norms = [0.0f32; MAX_ED_TERMS];
    fit_coeff_norms(
        outputs,
        1,
        nodes,
        n_terms,
        cfg.damping,
        &mut rhs,
        &mut coef,
        &mut z,
        &mut coeff_norms,
    );
    let (ed, ed_norm) = ed_from_coeff_norms(&coeff_norms[..n_terms]);
    Ok(EdResult {
        ed,
        ed_norm,
        coeff_norms,
        n_terms,
    })
}

/// ED of a **vector-valued** function sampled along one interpolation path.
///
/// `outputs` is `nodes.len() × out_dim` row-major: row `i` is
/// `f(x(nodes[i])) ∈ ℝ^out_dim`. All `out_dim` outputs share one design matrix,
/// so this is a single Cholesky with `out_dim` right-hand sides — not
/// `out_dim` independent fits. The per-degree magnitude is `‖cₖ‖₂` over the
/// output dimensions, which collapses to `|cₖ|` at `out_dim = 1` (so this
/// agrees with [`effective_degree_along_path`] exactly).
///
/// `scratch` must have been built for at least `nodes.len()` and `out_dim`.
pub fn effective_degree_along_path_multi(
    outputs: &[f32],
    out_dim: usize,
    nodes: &[f32],
    cfg: &EdConfig,
    scratch: &mut EdScratch,
) -> Result<EdResult, EdError> {
    let n_terms = validate_path(outputs, out_dim, nodes, cfg)?;
    scratch.check(nodes.len(), 0, out_dim)?;
    let mut coeff_norms = [0.0f32; MAX_ED_TERMS];
    fit_coeff_norms(
        outputs,
        out_dim,
        nodes,
        n_terms,
        cfg.damping,
        &mut scratch.rhs,
        &mut scratch.coef,
        &mut scratch.z,
        &mut coeff_norms,
    );
    let (ed, ed_norm) = ed_from_coeff_norms(&coeff_norms[..n_terms]);
    Ok(EdResult {
        ed,
        ed_norm,
        coeff_norms,
        n_terms,
    })
}

/// Average ED over `cfg.n_pairs` data-anchored interpolation paths — the
/// generic driver.
///
/// `decode` is the consumer's frozen function: `decode(&x, &mut y)` writes
/// `f(x) ∈ ℝ^out_dim` into `y`. It is deliberately a closure rather than a
/// trait so `katgpt-core` stays domain-agnostic — the same driver serves a
/// shard readout, a LoRA overlay, a KARC forecast, or an NPC policy without
/// any of those vocabularies leaking in here.
///
/// `endpoints_a` and `endpoints_b` are each `n_pairs × in_dim` row-major; path
/// `p` runs `x(α) = α·a_p + (1−α)·b_p`. **Both endpoint sets must come from the
/// real data manifold** — the paper (C.1) measures the ED signal collapsing to
/// baseline under random-noise endpoints.
///
/// Each path gets its own node draw, seeded from `cfg.seed` and the path index,
/// so the whole call is reproducible from the config alone. Returns the mean of
/// `ed`, `ed_norm`, and `coeff_norms` over paths.
///
/// Cost: `n_pairs × resolution` decode calls plus `n_pairs` `(K+1)²` solves.
/// Allocation-free after the first call given a reused `scratch`.
pub fn ed_over_pairs<F>(
    mut decode: F,
    endpoints_a: &[f32],
    endpoints_b: &[f32],
    cfg: &EdConfig,
    scratch: &mut EdScratch,
) -> Result<EdResult, EdError>
where
    F: FnMut(&[f32], &mut [f32]),
{
    cfg.validate()?;
    if cfg.n_pairs == 0 {
        return Err(EdError::ZeroPairs);
    }
    let (_, in_dim, out_dim) = scratch.shape();
    if in_dim == 0 {
        return Err(EdError::ScratchTooSmall { need: 1, got: 0 });
    }
    scratch.check(cfg.resolution, in_dim, out_dim)?;
    let expected = cfg.n_pairs * in_dim;
    for ep in [endpoints_a, endpoints_b] {
        if ep.len() != expected {
            return Err(EdError::LengthMismatch {
                expected,
                got: ep.len(),
            });
        }
    }

    let n_terms = cfg.n_terms();
    let r = cfg.resolution;
    let mut acc = EdResult::zeroed(n_terms);

    for p in 0..cfg.n_pairs {
        randomized_cosine_nodes(r, path_seed(cfg.seed, p), &mut scratch.nodes)?;
        let a = &endpoints_a[p * in_dim..(p + 1) * in_dim];
        let b = &endpoints_b[p * in_dim..(p + 1) * in_dim];
        for i in 0..r {
            let alpha = scratch.nodes[i];
            for (d, slot) in scratch.x_buf[..in_dim].iter_mut().enumerate() {
                *slot = alpha.mul_add(a[d] - b[d], b[d]);
            }
            // Disjoint field borrows: `x_buf` shared, `outputs` unique.
            let row = &mut scratch.outputs[i * out_dim..(i + 1) * out_dim];
            decode(&scratch.x_buf[..in_dim], row);
        }
        let mut coeff_norms = [0.0f32; MAX_ED_TERMS];
        fit_coeff_norms(
            &scratch.outputs[..r * out_dim],
            out_dim,
            &scratch.nodes[..r],
            n_terms,
            cfg.damping,
            &mut scratch.rhs,
            &mut scratch.coef,
            &mut scratch.z,
            &mut coeff_norms,
        );
        let (ed, ed_norm) = ed_from_coeff_norms(&coeff_norms[..n_terms]);
        acc.ed += ed;
        acc.ed_norm += ed_norm;
        for (dst, src) in acc.coeff_norms.iter_mut().zip(coeff_norms.iter()) {
            *dst += *src;
        }
    }

    let inv = 1.0 / cfg.n_pairs as f32;
    acc.ed *= inv;
    acc.ed_norm *= inv;
    for c in &mut acc.coeff_norms {
        *c *= inv;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed Chebyshev–Gauss nodes mapped to `α ∈ [0,1]`, for tests that want a
    /// deterministic grid independent of the sampler.
    fn fixed_nodes(r: usize) -> Vec<f32> {
        (0..r)
            .map(|i| {
                let theta = (2.0 * i as f32 + 1.0) * core::f32::consts::PI / (2.0 * r as f32);
                0.5 * (1.0 - theta.cos())
            })
            .collect()
    }

    #[test]
    fn cosine_nodes_are_sorted_stratified_and_in_range() {
        let cfg = EdConfig::precise();
        let mut out = vec![0.0f32; cfg.resolution];
        randomized_cosine_nodes(cfg.resolution, cfg.seed, &mut out).unwrap();
        let step = core::f32::consts::PI / cfg.resolution as f32;
        let mut prev = -1.0f32;
        for (i, &a) in out.iter().enumerate() {
            assert!((0.0..=1.0).contains(&a), "alpha {a} out of [0,1]");
            assert!(a > prev, "not strictly increasing at {i}");
            prev = a;
            // Stratum i is theta in [i*step, (i+1)*step].
            let lo = 0.5 * (1.0 - (i as f32 * step).cos());
            let hi = 0.5 * (1.0 - ((i + 1) as f32 * step).cos());
            assert!(
                a >= lo - 1e-6 && a <= hi + 1e-6,
                "alpha {a} outside stratum {i}"
            );
        }
    }

    #[test]
    fn cosine_nodes_are_deterministic_and_seed_sensitive() {
        let mut a = [0.0f32; 15];
        let mut b = [0.0f32; 15];
        let mut c = [0.0f32; 15];
        randomized_cosine_nodes(15, 42, &mut a).unwrap();
        randomized_cosine_nodes(15, 42, &mut b).unwrap();
        randomized_cosine_nodes(15, 43, &mut c).unwrap();
        assert_eq!(a, b, "same seed must reproduce bit-identical nodes");
        assert_ne!(a, c, "different seed must move the nodes");
    }

    #[test]
    fn constant_function_has_zero_effective_degree() {
        let cfg = EdConfig::cheap();
        let nodes = fixed_nodes(cfg.resolution);
        let outputs = vec![3.5f32; cfg.resolution];
        let r = effective_degree_along_path(&outputs, &nodes, &cfg).unwrap();
        assert!(r.ed < 1e-4, "constant ED = {}", r.ed);
        assert!(r.ed_norm < 1e-4, "constant ED_norm = {}", r.ed_norm);
        assert!((r.coeff_norms[0] - 3.5).abs() < 1e-3);
    }

    #[test]
    fn affine_function_ed_equals_first_coefficient() {
        // f(alpha) = 1 + 4*alpha. In u = 2*alpha-1: f = 3 + 2*u, so c1 = 2.
        let cfg = EdConfig::cheap();
        let nodes = fixed_nodes(cfg.resolution);
        let outputs: Vec<f32> = nodes.iter().map(|&a| 4.0f32.mul_add(a, 1.0)).collect();
        let r = effective_degree_along_path(&outputs, &nodes, &cfg).unwrap();
        assert!(
            (r.coeff_norms[1] - 2.0).abs() < 1e-3,
            "c1 = {}",
            r.coeff_norms[1]
        );
        assert!((r.ed - r.coeff_norms[1]).abs() < 1e-5, "ED != |c1|");
        assert!(r.coeff_norms[2] < 1e-3 && r.coeff_norms[3] < 1e-3);
    }

    #[test]
    fn pure_chebyshev_mode_has_ed_norm_equal_to_its_degree() {
        let cfg = EdConfig::precise();
        let nodes = fixed_nodes(cfg.resolution);
        let basis = ChebyshevBasis::<MAX_ED_TERMS>::new();
        for k in 1..=cfg.max_degree {
            let outputs: Vec<f32> = nodes
                .iter()
                .map(|&a| {
                    let mut psi = [0.0f32; MAX_ED_TERMS];
                    basis.eval_into(2.0f32.mul_add(a, -1.0), &mut psi);
                    psi[k]
                })
                .collect();
            let r = effective_degree_along_path(&outputs, &nodes, &cfg).unwrap();
            assert!(
                (r.ed_norm - k as f32).abs() < 5e-3,
                "T_{k}: ED_norm = {} (expected {k})",
                r.ed_norm
            );
        }
    }

    #[test]
    fn ed_scales_with_output_magnitude_but_ed_norm_does_not() {
        let cfg = EdConfig::precise();
        let nodes = fixed_nodes(cfg.resolution);
        let base: Vec<f32> = nodes.iter().map(|&a| a.powi(3) - 0.4 * a + 0.2).collect();
        let scaled: Vec<f32> = base.iter().map(|&y| 2.0 * y).collect();
        let r1 = effective_degree_along_path(&base, &nodes, &cfg).unwrap();
        let r2 = effective_degree_along_path(&scaled, &nodes, &cfg).unwrap();
        assert!(
            (r2.ed - 2.0 * r1.ed).abs() < 1e-3 * r1.ed.max(1.0),
            "ED must scale"
        );
        assert!(
            (r2.ed_norm - r1.ed_norm).abs() < 1e-4,
            "ED_norm must not scale"
        );
    }

    #[test]
    fn multi_output_agrees_with_scalar_at_out_dim_one() {
        let cfg = EdConfig::cheap();
        let nodes = fixed_nodes(cfg.resolution);
        let outputs: Vec<f32> = nodes.iter().map(|&a| a * a - 0.3 * a).collect();
        let scalar = effective_degree_along_path(&outputs, &nodes, &cfg).unwrap();
        let mut scratch = EdScratch::new(&cfg, 0, 1);
        let multi =
            effective_degree_along_path_multi(&outputs, 1, &nodes, &cfg, &mut scratch).unwrap();
        assert_eq!(scalar, multi);
    }

    #[test]
    fn multi_output_norm_is_l2_over_output_dims() {
        // Two output channels, both pure T_2 with amplitudes 3 and 4 => ||c2|| = 5.
        let cfg = EdConfig::precise();
        let nodes = fixed_nodes(cfg.resolution);
        let basis = ChebyshevBasis::<MAX_ED_TERMS>::new();
        let mut outputs = Vec::with_capacity(nodes.len() * 2);
        for &a in &nodes {
            let mut psi = [0.0f32; MAX_ED_TERMS];
            basis.eval_into(2.0f32.mul_add(a, -1.0), &mut psi);
            outputs.push(3.0 * psi[2]);
            outputs.push(4.0 * psi[2]);
        }
        let mut scratch = EdScratch::new(&cfg, 0, 2);
        let r = effective_degree_along_path_multi(&outputs, 2, &nodes, &cfg, &mut scratch).unwrap();
        assert!(
            (r.coeff_norms[2] - 5.0).abs() < 1e-2,
            "||c2|| = {}",
            r.coeff_norms[2]
        );
        assert!((r.ed_norm - 2.0).abs() < 1e-2, "ED_norm = {}", r.ed_norm);
    }

    #[test]
    fn ed_over_pairs_recovers_a_known_polynomial_degree() {
        // decode(x) = (x0)^3 along a path between two 1-D endpoints, so the
        // restriction is exactly a cubic in alpha.
        let cfg = EdConfig {
            n_pairs: 4,
            ..EdConfig::precise()
        };
        let a = vec![1.0f32, 0.9, 1.1, 0.8];
        let b = vec![-1.0f32, -0.8, -1.2, -0.7];
        let mut scratch = EdScratch::new(&cfg, 1, 1);
        let r =
            ed_over_pairs(|x, y| y[0] = x[0] * x[0] * x[0], &a, &b, &cfg, &mut scratch).unwrap();
        assert!(
            r.coeff_norms[4] < 1e-2 && r.coeff_norms[5] < 1e-2,
            "no spurious high modes"
        );
        assert!(
            r.ed_norm > 1.0 && r.ed_norm < 3.0,
            "cubic ED_norm = {}",
            r.ed_norm
        );
    }

    #[test]
    fn ed_over_pairs_is_reproducible_from_the_config_seed() {
        let cfg = EdConfig::cheap();
        let a: Vec<f32> = (0..cfg.n_pairs * 3)
            .map(|i| (i as f32) * 0.11 - 1.0)
            .collect();
        let b: Vec<f32> = (0..cfg.n_pairs * 3)
            .map(|i| 0.7 - (i as f32) * 0.09)
            .collect();
        let f = |x: &[f32], y: &mut [f32]| y[0] = x[0] * x[1] + x[2].tanh();
        let mut s1 = EdScratch::new(&cfg, 3, 1);
        let mut s2 = EdScratch::new(&cfg, 3, 1);
        let r1 = ed_over_pairs(f, &a, &b, &cfg, &mut s1).unwrap();
        let r2 = ed_over_pairs(f, &a, &b, &cfg, &mut s2).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn rejects_malformed_requests() {
        let bad_degree = EdConfig {
            max_degree: 9,
            ..EdConfig::cheap()
        };
        assert_eq!(
            bad_degree.validate(),
            Err(EdError::DegreeTooLarge { max_degree: 9 })
        );
        let thin = EdConfig {
            resolution: 2,
            max_degree: 3,
            ..EdConfig::cheap()
        };
        assert_eq!(
            thin.validate(),
            Err(EdError::ResolutionTooSmall {
                resolution: 2,
                n_terms: 4
            })
        );
        let undamped = EdConfig {
            damping: 0.0,
            ..EdConfig::cheap()
        };
        assert_eq!(undamped.validate(), Err(EdError::NonPositiveDamping));

        let cfg = EdConfig::cheap();
        let nodes = fixed_nodes(cfg.resolution);
        assert_eq!(
            effective_degree_along_path(&[1.0, 2.0], &nodes, &cfg),
            Err(EdError::LengthMismatch {
                expected: 4,
                got: 2
            })
        );
    }

    #[test]
    fn ed_reducer_matches_the_definition() {
        // |c| = [2, 1, 0, 4] => ED = 1*1 + 3*4 = 13, sum|c| = 7.
        let (ed, ed_norm) = ed_from_coeff_norms(&[2.0, 1.0, 0.0, 4.0]);
        assert!((ed - 13.0).abs() < 1e-6);
        assert!((ed_norm - 13.0 / 7.0).abs() < 1e-6);
        assert_eq!(ed_from_coeff_norms(&[0.0; 4]), (0.0, 0.0));
    }
}
