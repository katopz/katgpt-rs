//! KARC — Kolmogorov-Arnold Reservoir Computing: delay-basis-ridge forecaster.
//!
//! A modelless, inference-time trajectory forecaster. Given a trajectory
//! `{u_i ∈ ℝᴰ}`, KARC:
//! 1. Concatenates the last-K observations (delay embedding): `x_i = u_i ⊕ u_{i-1} ⊕ … ⊕ u_{i-K+1}` (paper Eq. 10), length `K·D`.
//! 2. Expands each coordinate onto `M` basis functions via a sealed [`KarcBasis`] trait (paper Eq. 8): `Ψ(x) ∈ ℝ^{K·D·M}`.
//! 3. Fits a linear readout `Wout` by closed-form ridge regression (paper Eq. 14): `Wout = Y·Hᵀ·(H·Hᵀ + λI)⁻¹`. The Woodbury identity (paper Eq. 40) is used to pick the cheaper of the feature-space `(XᵀX + λI)⁻¹` or sample-space `(XXᵀ + λI)⁻¹` inversion.
//! 4. Forecasts in a single zero-alloc matvec: `û_{i+1} = Wout · Ψ(x_i)` (paper Eq. 11).
//!
//! No backprop, no autodiff, no softmax. The basis functions are bounded
//! (Fourier, Chebyshev `|Tⱼ|≤1`, B-spline partition-of-unity), so the forecast
//! stays numerically well-behaved without sigmoid/softmax gating.
//!
//! # References
//!
//! - **Plan:** `katgpt-rs/.plans/308_karc_delay_basis_ridge_forecaster.md`
//! - **Research:** `katgpt-rs/.research/288_KARC_Delay_Basis_Ridge_Forecaster.md`
//! - **Source paper:** arXiv:2606.19984 — Huang, Kurths, Tang, *Kolmogorov-Arnold Reservoir Computing*, 2026-06-18
//! - **Sibling ridge path:** `crates/katgpt-core/src/peira.rs` (Plan 153) — same `(N + λI)⁻¹` math, f64, inter-view covariances. KARC reuses the pattern via `linalg::ridge_solve` (f32) rather than touching PEIRA.
//!
//! # GOAT gate (Plan 308 §"GOAT gate")
//!
//! - **G1** — double-scroll NRMSE ≤ 1.0e-3, threshold time ≥ 8 Lyapunov times. Verified by `examples/karc_double_scroll.rs`.
//! - **G2** — `forecast_into` ≤ 500 ns/call at D=8,M=8,K=4. Verified by `benches/karc_forecast_bench.rs`.
//! - **G3** — `forecast_into` zero allocations after warmup. Verified by `tests/karc_alloc_check.rs`.
//! - **G4** — two forecasters fit on identical data produce byte-identical `Wout`. Verified by `tests/karc_reproducibility.rs`.
//!
//! # Phase 2 — Higher-order features + chunked Gram + low-rank ALS
//!
//! Plan 308 T2.1–T2.6 (paper §A + §C, Eqs. 32/44/47). Adds:
//!
//! - **Eq. 32** [`feature_expand_higher_order`] — outer-product features up to
//!   order `R`. For `R=2`, appends `ψ_{m1}(x_{c1})·ψ_{m2}(x_{c2})` for all
//!   lexicographic `(c1,m1) ≤ (c2,m2)` pairs (combinatorial enumeration, no
//!   duplicates). First-order features stay first in the output buffer; pairwise
//!   products follow in `(f1,f2)` linear-index order (equivalent to the
//!   paper's lexicographic `(c,m)` ordering since `f = c·M + m`).
//! - **Eq. 44** [`chunked_gram_into`] — block-accumulate `G = Σ h_i·h_iᵀ + λI`
//!   over a feature-row iterator. The full `N×d_h` feature matrix `H` is never
//!   materialized; each row is streamed through and its outer product
//!   accumulated into the `d_h×d_h` Gram. This is the memory-optimized
//!   construction used when higher-order features blow up `d_h`
//!   (e.g. `d_h=4752` at `D=3,M=8,K=4,R=2`; the full `H` would be
//!   `4000×4752×4 B ≈ 76 MB` of f32, but chunking avoids the double-pass cost).
//! - **Eq. 47** [`low_rank_fit`] — alternating least squares factorization
//!   `Wout ≈ A·B` where `A: D×r`, `B: r×d_h`. The B-step pre-factors
//!   `G+λI` once (O(d_h³)) and each subsequent iteration is two O(d_h²·r)
//!   back-substitutions. Deterministic init (`B = [I_r | 0]`, zero `A`),
//!   deterministic iteration order (A→B), deterministic Cholesky → bit-identical
//!   `(A,B)` across runs given identical `(G, Cov, d_h, D, r, λ, iters, tol)`.
//!
//! The [`KarcForecaster::fit_low_rank`] method wraps [`low_rank_fit`] for the
//! first-order feature buffer already accumulated via [`KarcForecaster::accumulate_pair`].
//! For higher-order features, call [`feature_expand_higher_order`] +
//! [`chunked_gram_into`] + [`low_rank_fit`] + [`forecast_low_rank_apply`]
//! directly (see `examples/karc_double_scroll_higher_order.rs`).
//!
//! **B-step implementation (documented trade-off).** The B-step normal equation
//! `(AᵀA)·B·G + λB = Aᵀ·Covᵀ` is a Sylvester-like equation. We solve it exactly
//! via the Kronecker vectorization `(G ⊗ AᵀA + λI)·vec(B) = vec(Aᵀ·Covᵀ)`, which
//! is an `(r·d_h)×(r·d_h)` Cholesky solve. This is exact but O((r·d_h)³), so it
//! is feasible only when `r·d_h ≤ ~2000` (covering the first-order forecaster
//! path with `d_h ≤ 600`, `r ≤ 8`). For the `d_h=4752` higher-order benchmark
//! config, the higher-order full-rank fit (`fit_ridge`) is used instead; the
//! low-rank comparison runs on first-order features (d_h=96) where the exact
//! B-step is fast. **The large-d_h Jacobi-eigen B-step shipped in Issue 185
//! (see [`large_dh::low_rank_fit_jacobi_bstep`])** makes the K=8/M=8/R=2
//! (d_h=18_720) config feasible by replacing the Kronecker Cholesky with a
//! Bartels–Stewart eigenbasis diagonalization (one-time `O(d_h³)` + per-iter
//! `O(r·d_h²)`); the paper-Par K=8/M=24/R=2 (d_h=166_752) config remains
//! out of reach without further factorization.

pub use crate::linalg::ridge_solve::{
    chol_solve_f64, cholesky_f64, ridge_solve_direct_f64, ridge_solve_woodbury_f32,
};

// NOTE: the `pub use` above serves double duty — it imports the Cholesky
// primitives for local use in this module AND re-exports them so downstream
// KARC consumers (e.g. riir-ai's cross_game transfer protocol) can build on
// the same linear-algebra substrate without reaching into the feature-gated
// `linalg::ridge_solve` path themselves.
use crate::simd;

// ── Large-d_h ALS B-step path (Issue 185, Plan 308 T4.5 unblock) ────────
//
// Sibling module to [`low_rank_fit`] above: same ALS contract but the B-step
// is solved via Jacobi eigendecomposition of `G` (one-time) + `AᵀA`
// (per-iter) instead of the `(r·d_h) × (r·d_h)` Kronecker Cholesky. Makes
// `d_h ≤ ~20_000` configs feasible (K=8/M=8/R=2 = 18_720) — the promotion
// gate target. Kept in a sibling file because `mod.rs` already exceeds the
// 2048-line guideline.
pub mod large_dh;

// ── Regime Gate (Plan 556 Phase 1) ─────────────────────────────────────
//
// Closed-form residual-variance mux between this forecaster (KARC) and the
// SeasonalNaiveForecaster floor. Directly fixes the structural periodic-
// blindness documented in `.benchmarks/010_report_the_floor_consolidated.md`
// §T7 (2026-07-20 K-sweep): KARC is a chaotic-regime specialist and *loses*
// on periodic data regardless of K. The gate routes each trajectory to the
// right forecaster by comparing rolling Welford variance of their residuals.
// Gated on `karc_regime_gate` (which implies `karc_forecaster`). Pure
// modelless (Welford + sigmoid). Zero runtime cost unless constructed.
#[cfg(feature = "karc_regime_gate")]
pub mod regime_gate;

// ── Batched MatVec (Plan 556 Phase 2) ──────────────────────────────────
//
// SIMD-batched forecast across N forecasters of identical (D, M, K) shape.
// Each NPC owns its own `Wout`; the basis is shared across the batch. The
// amortization comes from contiguous layout + loop hoisting, *not* from
// parallelism — at N ≤ 32 (realistic octree-cell size) the per-forecast cost
// is ~75ns, far below rayon's ~5µs scheduling overhead. Crowd-scale perf
// primitive: unblocks Plan 514 Phase 3 (octree-batched cell-level KARC).
// Gated on `karc_batched_matvec` (which implies `karc_forecaster`).
#[cfg(feature = "karc_batched_matvec")]
pub mod batched;

// ── LOD Tier (Plan 556 Phase 3) ────────────────────────────────────────
//
// LOD (Level-of-Detail) tier tag + tier-promotion Wout projection. Three
// nested tiers (LOD0 background / LOD1 midground / LOD2 hero) map to
// different KarcForecaster const-generic configs; the nested-subset structure
// makes tier promotion a pure index remap (down-tier preserves surviving
// columns; up-tier zero-fills new columns). Phase 3 ships R=1 only; the R=2
// promotion-gate config (d_h=18_720, Issue 185/186/187) is deferred.
// Gated on `karc_lod_tier` (which implies `karc_forecaster`).
#[cfg(feature = "karc_lod_tier")]
pub mod lod_tier;

// Re-export the LOD tier API at `katgpt_core::karc::` so callers can use the
// consistent `karc::` namespace for all KARC primitives. The top-level
// `katgpt_core::` re-export (lib.rs) stays for caller ergonomics.
#[cfg(feature = "karc_lod_tier")]
pub use lod_tier::{KarcLodTier, is_identity_projection, project_wout_lod_into};

// ── Hebbian Readout (riir-ai Plan 584 — the `w_out` fit, Hebbian-flavored) ──
// The forecaster's training pairs as facts: key = pad64(delay_state), value =
// pad64(target); the constructed (A, G, B) plays the `w_out` role. Buys edit
// semantics (the Plan 583 journal loop), the margin audit, and the ndb
// freeze chain over the fixed-basis `fit_ridge`. Gated on
// `karc_hebbian_readout` (implies `hebbian_kernel_memory`; the karc-side
// dependency is conceptual — this module needs no `fit_ridge` machinery,
// only the pairing convention).
#[cfg(feature = "karc_hebbian_readout")]
pub mod hebbian_readout;

// ── Sealed trait machinery ────────────────────────────────────────────────────

mod sealed {
    /// Seals [`super::KarcBasis`] so only the three shipped implementations
    /// (Fourier / Chebyshev / BSpline) can implement it — lets us add methods
    /// later without breaking downstream impls.
    pub trait Sealed {}
}

/// Univariate basis dictionary for one coordinate of the delay-embedded state.
///
/// `eval_into` projects a scalar `x` into `out[0..M]`. The three shipped impls
/// are bounded: Fourier `‖ψ‖≤1` per mode, Chebyshev `|Tⱼ|≤1`, B-spline
/// partition-of-unity. Bounds on the full feature vector are in paper Eq. 96/99/101.
///
/// Sealed per Plan 308 T1.2 — vendored minimal trait with attribution to
/// `riir-engine::linoss::basis::SpectralBasis` (same Fourier/Chebyshev/B-spline
/// family; KARC cannot import riir-engine from katgpt-core).
pub trait KarcBasis<const M: usize>: sealed::Sealed + Sync {
    /// Project scalar `x` into the `M`-length feature slice. Zero-allocation.
    fn eval_into(&self, x: f32, out: &mut [f32; M]);

    /// Human-readable name for diagnostics / G4 attribution.
    fn name(&self) -> &'static str;
}

// ── Basis implementations ─────────────────────────────────────────────────

/// Fourier basis: `ψ_{2i-1}(x) = cos(2π·i·x/P)`, `ψ_{2i}(x) = sin(2π·i·x/P)`,
/// `i = 1..=M/2` (paper Eq. 35). `M` must be even.
///
/// Spectral norm bound per coordinate: `‖ψ(x)‖₂ = √(M/2)` (paper Eq. 96).
/// Period `P` is set at construction and must be positive.
#[derive(Clone, Copy, Debug)]
pub struct FourierBasis<const M: usize> {
    /// Period `P > 0`. Frequencies are `ω_i = 2π·i/P`.
    period: f32,
}

impl<const M: usize> FourierBasis<M> {
    /// Construct with period `P`. Asserts `M` is even and `P > 0`.
    pub const fn new(period: f32) -> Self {
        assert!(M.is_multiple_of(2), "FourierBasis requires even M (m = 2Q)");
        assert!(period > 0.0, "FourierBasis period must be positive");
        Self { period }
    }

    /// Angular frequency `ω_i = 2π·i/P`.
    #[inline]
    fn omega(&self, i: usize) -> f32 {
        core::f32::consts::TAU * (i as f32) / self.period
    }
}

impl<const M: usize> sealed::Sealed for FourierBasis<M> {}

impl<const M: usize> KarcBasis<M> for FourierBasis<M> {
    #[inline]
    fn eval_into(&self, x: f32, out: &mut [f32; M]) {
        let q = M / 2;
        let mut i = 0;
        while i < q {
            let w = self.omega(i + 1);
            out[2 * i] = (w * x).cos();
            out[2 * i + 1] = (w * x).sin();
            i += 1;
        }
    }

    fn name(&self) -> &'static str {
        "fourier"
    }
}

/// Chebyshev basis of the first kind: `T₀(x)=1, T₁(x)=x, Tₙ₊₁=2xTₙ−Tₙ₋₁`
/// (paper Eq. 38). `|Tⱼ(x)| ≤ 1` for `x ∈ [-1,1]`; caller is responsible for
/// rescaling inputs into `[-1,1]` (KARC does not rescale — the forecaster sees
/// whatever the trajectory contains, and ridge regularisation absorbs drift).
///
/// Spectral norm bound per coordinate: `‖ψ(x)‖₂ ≤ √M` (paper Eq. 101).
#[derive(Clone, Copy, Debug, Default)]
pub struct ChebyshevBasis<const M: usize>;

impl<const M: usize> ChebyshevBasis<M> {
    pub const fn new() -> Self {
        Self
    }
}

impl<const M: usize> sealed::Sealed for ChebyshevBasis<M> {}

impl<const M: usize> KarcBasis<M> for ChebyshevBasis<M> {
    #[inline]
    fn eval_into(&self, x: f32, out: &mut [f32; M]) {
        match M {
            0 => {}
            1 => out[0] = 1.0,
            _ => {
                out[0] = 1.0;
                out[1] = x;
                // Three-term recurrence T_{n+1} = 2x T_n - T_{n-1}.
                let mut n = 1;
                while n + 1 < M {
                    out[n + 1] = 2.0_f32.mul_add(x * out[n], -out[n - 1]);
                    n += 1;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "chebyshev"
    }
}

/// Uniform cubic B-spline basis on `[0,1]` (paper Eq. 36–37, Cox–de Boor).
/// Degree fixed at 3. Knots are clamped uniform: the endpoints are repeated
/// `degree+1` times, interior knots evenly spaced. Forms a partition of unity,
/// so `‖ψ(x)‖₂ ≤ 1` per coordinate (paper Eq. 99).
///
/// `M` must be `≥ degree+1 = 4` (need at least one basis function). Knots are
/// stored in a `Vec<f32>` (allocated once at construction) because stable Rust
/// does not permit `[f32; M + degree + 1]` as a struct field when `M` is a
/// const generic (`generic_const_exprs` is unstable). Construction is not the
/// hot path — only `eval_into` is, and it reads the slice.
#[derive(Clone, Debug)]
pub struct BSplineBasis<const M: usize> {
    /// Clamped uniform knot vector, length `M + degree + 1 = M + 4`.
    knots: Vec<f32>,
}

impl<const M: usize> BSplineBasis<M> {
    const DEGREE: usize = 3;

    /// Number of knots = `M + degree + 1`.
    fn knot_len() -> usize {
        M + Self::DEGREE + 1
    }

    /// Construct on domain `[0,1]`.
    pub fn new() -> Self {
        assert!(M > Self::DEGREE, "BSplineBasis requires M >= degree+1 = 4");
        let d = Self::DEGREE;
        let knot_len = Self::knot_len();
        let mut knots = vec![0.0f32; knot_len];
        // Clamped: first degree+1 knots = 0, last degree+1 knots = 1.
        for i in 0..=d {
            knots[i] = 0.0;
            knots[knot_len - 1 - i] = 1.0;
        }
        // Interior knots: evenly spaced in (0,1). Number of interior knots =
        // knot_len - 2*(degree+1) = M + d + 1 - 2d - 2 = M - d - 1.
        let n_interior = M - d - 1; // 0 for the minimal M = d+1 case
        if n_interior > 0 {
            for k in 0..n_interior {
                knots[d + 1 + k] = (k + 1) as f32 / (n_interior + 1) as f32;
            }
        }
        Self { knots }
    }

    /// Cox–de Boor degree-0 indicator (paper Eq. 36). Right-continuous on
    /// `[t_i, t_{i+1})`; zero-width intervals (clamped repeated knots) return 0.
    /// The right boundary `x = t_max` is included in the last nonzero span so
    /// the partition-of-unity holds at the clamped endpoint.
    #[inline]
    fn b0(knots: &[f32], i: usize, x: f32) -> f32 {
        let lo = knots[i];
        let hi = knots[i + 1];
        if lo == hi {
            return 0.0; // zero-width interval (clamp repetition)
        }
        let t_max = knots[knots.len() - 1];
        if x >= lo && (x < hi || (hi == t_max && x <= hi)) {
            1.0
        } else {
            0.0
        }
    }
}

impl<const M: usize> Default for BSplineBasis<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const M: usize> sealed::Sealed for BSplineBasis<M> {}

impl<const M: usize> KarcBasis<M> for BSplineBasis<M> {
    #[inline]
    fn eval_into(&self, x: f32, out: &mut [f32; M]) {
        // Cox–de Boor recursion (paper Eq. 37). For M small (≤ ~60),
        // computing all M basis functions via the full triangular table is
        // cheap and avoids per-x nonzero-span bookkeeping.
        let d = Self::DEGREE;
        // prev/cur hold one degree-row at a time. n_knots-1 degree-0 functions.
        let n_knots = self.knots.len();
        let n_deg0 = n_knots - 1; // = M + d
        let mut prev = [0.0f32; 64];
        let mut cur = [0.0f32; 64];
        debug_assert!(n_deg0 <= 64, "BSplineBasis scratch overflow (M too large)");
        // Degree 0.
        for (i, slot) in prev[..n_deg0].iter_mut().enumerate() {
            *slot = Self::b0(&self.knots, i, x);
        }
        // Recurse up to degree d.
        let mut s = 1;
        while s <= d {
            // At degree s, function count = n_knots - 1 - s.
            let count = n_knots - 1 - s;
            for i in 0..count {
                let denom1 = self.knots[i + s] - self.knots[i];
                let term1 = if denom1 > 0.0 {
                    (x - self.knots[i]) / denom1 * prev[i]
                } else {
                    0.0
                };
                let denom2 = self.knots[i + s + 1] - self.knots[i + 1];
                let term2 = if denom2 > 0.0 {
                    (self.knots[i + s + 1] - x) / denom2 * prev[i + 1]
                } else {
                    0.0
                };
                cur[i] = term1 + term2;
            }
            prev[..count].copy_from_slice(&cur[..count]);
            s += 1;
        }
        // At degree d the first M entries of prev are the M basis functions.
        out[..M].copy_from_slice(&prev[..M]);
    }

    fn name(&self) -> &'static str {
        "bspline"
    }
}

// ── Delay ring buffer ─────────────────────────────────────────────────────

/// Fixed-capacity ring buffer of the last-K `D`-dimensional observations.
///
/// `push` overwrites the oldest entry once the buffer is full.
/// `flatten_into` writes the delay-embedded state in observation order
/// **newest first**: `x = u_t ⊕ u_{t-1} ⊕ … ⊕ u_{t-K+1}` (paper Eq. 10).
///
/// Inline `[[f32; D]; K]` — no heap indirection, the small fixed sizes
/// (D≤~64, K≤~16 typical) make this cache-friendly and zero-alloc.
#[derive(Clone, Debug)]
pub struct DelayRing<const D: usize, const K: usize> {
    slots: [[f32; D]; K],
    /// Index of the most-recently-written slot.
    head: usize,
    /// Number of valid observations (caps at K).
    filled: usize,
}

impl<const D: usize, const K: usize> DelayRing<D, K> {
    /// Construct an empty ring.
    pub fn new() -> Self {
        Self {
            slots: [[0.0; D]; K],
            head: 0,
            filled: 0,
        }
    }

    /// Number of valid observations currently stored.
    #[inline]
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// True once K observations have been pushed.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.filled == K
    }

    /// Push a new observation, overwriting the oldest if full.
    #[inline]
    pub fn push(&mut self, obs: &[f32; D]) {
        if self.filled == 0 {
            self.head = 0;
        } else {
            self.head = (self.head + 1) % K;
        }
        self.slots[self.head].copy_from_slice(obs);
        if self.filled < K {
            self.filled += 1;
        }
    }

    /// Write the delay-embedded state `u_t ⊕ u_{t-1} ⊕ … ⊕ u_{t-K+1}` (newest
    /// first) into `out` (length `K·D`). Returns `false` if the ring is not yet
    /// full (caller should not fit/forecast on a partial state).
    #[inline]
    pub fn flatten_into(&self, out: &mut [f32]) -> bool {
        if !self.is_full() {
            return false;
        }
        for k in 0..K {
            // slot index for lag-k observation: head, head-1, ..., wrapping.
            let idx = (self.head + K - k) % K;
            let dst = k * D;
            out[dst..dst + D].copy_from_slice(&self.slots[idx]);
        }
        true
    }

    /// Reset the ring to empty (capacity unchanged).
    pub fn clear(&mut self) {
        self.head = 0;
        self.filled = 0;
    }
}

impl<const D: usize, const K: usize> Default for DelayRing<D, K> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Feature expansion ─────────────────────────────────────────────────────

/// Apply `basis` to each of the `K·D` delay coordinates, writing `K·D·M`
/// features into `out`. Zero-allocation. Inner loop is chunk-4 unrolled over
/// coordinates to help SIMD auto-vectorisation of the per-coordinate writes.
///
/// Layout: `out[c*M..(c+1)*M] = basis.eval(delay_state[c])` for coordinate `c`
/// in `0..K·D`, matching paper Eq. 8.
#[inline]
pub fn feature_expand<B: KarcBasis<M>, const M: usize>(
    delay_state: &[f32],
    basis: &B,
    out: &mut [f32],
) {
    let n_coords = delay_state.len();
    let mut tmp = [0.0f32; M];
    let mut c = 0;
    // Chunk-4 over coordinates: 4 eval_into calls + 4 copies per iteration.
    while c + 4 <= n_coords {
        basis.eval_into(delay_state[c], &mut tmp);
        out[c * M..(c + 1) * M].copy_from_slice(&tmp);
        basis.eval_into(delay_state[c + 1], &mut tmp);
        out[(c + 1) * M..(c + 2) * M].copy_from_slice(&tmp);
        basis.eval_into(delay_state[c + 2], &mut tmp);
        out[(c + 2) * M..(c + 3) * M].copy_from_slice(&tmp);
        basis.eval_into(delay_state[c + 3], &mut tmp);
        out[(c + 3) * M..(c + 4) * M].copy_from_slice(&tmp);
        c += 4;
    }
    while c < n_coords {
        basis.eval_into(delay_state[c], &mut tmp);
        out[c * M..(c + 1) * M].copy_from_slice(&tmp);
        c += 1;
    }
}

// ── Phase 2: Higher-order feature expansion (paper Eq. 32) ────────────────

/// Total higher-order feature count for base (first-order) dimension `d_h_1`
/// and outer-product order `r` (paper Eq. 32).
///
/// - `r = 1`: `d_h_1` (same as first-order).
/// - `r = 2`: `d_h_1 + d_h_1·(d_h_1+1)/2` (first-order + all `i ≤ j` pairs).
///
/// `r > 2` is currently unimplemented (would need k-tuple combinatorial
/// enumeration); the function returns the first-order count as a fallback.
#[inline]
pub const fn higher_order_feature_count(d_h_1: usize, r: usize) -> usize {
    match r {
        1 => d_h_1,
        2 => d_h_1 + d_h_1 * (d_h_1 + 1) / 2,
        _ => d_h_1,
    }
}

/// Outer-product feature expansion up to order `R` (paper Eq. 32).
///
/// - `R = 1`: identical to [`feature_expand`] (writes `K·D·M` features).
/// - `R = 2`: writes first-order features into `out[..d_h_1]`, then appends
///   all pairwise products `ψ[f1]·ψ[f2]` for `0 ≤ f1 ≤ f2 < d_h_1` into
///   `out[d_h_1..]`. The pair index `(f1,f2)` with `f1 ≤ f2` matches the
///   paper's lexicographic `(c1,m1) ≤ (c2,m2)` enumeration because the
///   linear feature index `f = c·M + m` preserves lexicographic order on
///   `(c, m)`.
///
/// `R > 2` panics (k-tuple enumeration not implemented).
///
/// Output size must be at least [`higher_order_feature_count`]`(K·D·M, R)`.
#[inline]
pub fn feature_expand_higher_order<B: KarcBasis<M>, const M: usize, const R: usize>(
    delay_state: &[f32],
    basis: &B,
    out: &mut [f32],
) {
    let n_coords = delay_state.len();
    let d_h_1 = n_coords * M;
    match R {
        1 => feature_expand::<B, M>(delay_state, basis, out),
        2 => {
            // Write first-order features into the head, then read them back
            // to enumerate pairwise products into the tail.
            let (first_order, pairs) = out.split_at_mut(d_h_1);
            feature_expand::<B, M>(delay_state, basis, first_order);
            let mut idx = 0;
            for f1 in 0..d_h_1 {
                let p1 = first_order[f1];
                for &fv in first_order[f1..d_h_1].iter() {
                    pairs[idx] = p1 * fv;
                    idx += 1;
                }
            }
            debug_assert_eq!(idx, d_h_1 * (d_h_1 + 1) / 2, "pair buffer size mismatch");
        }
        _ => panic!(
            "feature_expand_higher_order: R={R} not implemented (only R ∈ {{1,2}})"
        ),
    }
}

// ── Phase 2: Chunked Gram accumulation (paper Eq. 44) ─────────────────────

/// Block-accumulate `G = Σᵢ hᵢ·hᵢᵀ + λI` over a feature-row iterator,
/// writing the `d_h × d_h` f64 Gram into `out_gram` (paper Eq. 44).
///
/// The full `N × d_h` feature matrix `H` is **never materialized**: each row
/// `hᵢ` (`&[f32]` of length `d_h`) is streamed in and its rank-1 outer product
/// accumulated in place. This is the memory-optimized construction required
/// when higher-order features (T2.1) blow up `d_h` — e.g. for
/// `D=3, M=8, K=4, R=2`, `d_h=4752` and the full `H` (4000×4752 f32) would
/// be ~76 MB, but chunking avoids ever holding it all in memory.
///
/// Accumulation is in **f64** for numerical robustness at small `λ` (mirrors
/// the Phase 1 `fit_direct` Gram path). The caller passes `lambda` so the
/// output is ready for Cholesky without a second pass. Pass `lambda = 0.0` to
/// get the un-regularized `XᵀX` (useful for [`low_rank_fit`], which adds `λI`
/// internally for the A-step and B-step).
///
/// `out_gram` must hold at least `d_h * d_h` f64; any excess is untouched.
#[inline]
pub fn chunked_gram_into<'a, I>(features_iter: I, out_gram: &mut [f64], lambda: f64, d_h: usize)
where
    I: Iterator<Item = &'a [f32]>,
{
    debug_assert!(out_gram.len() >= d_h * d_h, "gram buffer too small");
    // Zero the active region (single memset instead of an element loop).
    out_gram[..d_h * d_h].fill(0.0);
    // Accumulate h_i · h_iᵀ for each row (f64, chunk-4 unrolled inner loop).
    for row in features_iter {
        debug_assert_eq!(row.len(), d_h, "feature row length mismatch");
        // Slice the row to exactly d_h once — the inner loop then carries no
        // bounds checks (same FP operations, same order).
        let row = &row[..d_h];
        for i in 0..d_h {
            let row_i = row[i] as f64;
            let gram_i = i * d_h;
            let g_row = &mut out_gram[gram_i..gram_i + d_h];
            let mut j = 0;
            while j + 4 <= d_h {
                g_row[j] += row_i * row[j] as f64;
                g_row[j + 1] += row_i * row[j + 1] as f64;
                g_row[j + 2] += row_i * row[j + 2] as f64;
                g_row[j + 3] += row_i * row[j + 3] as f64;
                j += 4;
            }
            while j < d_h {
                g_row[j] += row_i * row[j] as f64;
                j += 1;
            }
        }
    }
    // Add the ridge diagonal.
    for i in 0..d_h {
        out_gram[i * d_h + i] += lambda;
    }
}

/// Accumulate the upper triangle of `G = XᵀX` from `n` contiguous feature rows
/// stored in `features_buf` (each row is `d_h` wide, f32), then symmetrise into
/// the lower triangle by copy. `gram` must already be zeroed and sized to at
/// least `d_h * d_h`.
///
/// This is the DRY-extracted twin of [`chunked_gram_into`] for the common case
/// where features live in a single contiguous `&[f32]` buffer (the
/// `KarcForecaster` trajectory buffer) rather than behind an iterator, and where
/// the caller wants the un-regularized matrix (it adds `λI` separately). The
/// upper-triangle-then-copy pattern is preserved bit-for-bit from the original
/// inline blocks in `fit_direct`, `fit_low_rank`, and
/// `fit_low_rank_with_frozen_a` — do not "simplify" to a full-matrix fill,
/// which would change the floating-point operation count and break the
/// byte-identical-equivalence contract on those paths.
#[inline]
fn accumulate_gram_upper_triangle(gram: &mut [f64], features_buf: &[f32], d_h: usize, n: usize) {
    debug_assert!(gram.len() >= d_h * d_h, "gram buffer too small");
    for row_idx in 0..n {
        let row = &features_buf[row_idx * d_h..(row_idx + 1) * d_h];
        for i in 0..d_h {
            let ri = row[i] as f64;
            // Slice the Gram row once — the inner loop keeps the identical FP
            // operation sequence but drops the per-element bounds checks.
            let g_row = &mut gram[i * d_h..i * d_h + d_h];
            let mut j = i;
            while j + 4 <= d_h {
                g_row[j] += ri * row[j] as f64;
                g_row[j + 1] += ri * row[j + 1] as f64;
                g_row[j + 2] += ri * row[j + 2] as f64;
                g_row[j + 3] += ri * row[j + 3] as f64;
                j += 4;
            }
            while j < d_h {
                g_row[j] += ri * row[j] as f64;
                j += 1;
            }
        }
    }
    // Symmetrise (we only filled the upper triangle above).
    for i in 0..d_h {
        for j in 0..i {
            gram[i * d_h + j] = gram[j * d_h + i];
        }
    }
}

// ── Scratch ───────────────────────────────────────────────────────────────

/// Pre-allocated scratch for [`KarcForecaster::fit_ridge`]. Allocate once via
/// [`KarcScratch::with_capacity`], reuse across fits (clear() + repopulate).
///
/// All `Vec`s use `with_capacity`; `clear()` only resets lengths, never
/// reallocates. This backs the zero-alloc-fit contract.
///
/// The Gram/covariance/Cholesky buffers are **f64** for numerical robustness
/// at small λ (f32 Cholesky fails when λ is below f32 epsilon relative to the
/// matrix scale). The fit is a cold path; the forecast matvec stays f32.
pub struct KarcScratch {
    /// Feature-space Gram `XᵀX + λI` (`d_h × d_h`, f64 for precision).
    pub gram: Vec<f64>,
    /// Cross-covariance `XᵀY` (`d_h × D`, f64).
    pub cov: Vec<f64>,
    /// Cholesky factor L (`d_h × d_h`, f64).
    pub chol: Vec<f64>,
    /// Back-substitution temp (`d_h × D`, f64).
    pub z_solve: Vec<f64>,
    /// Wᵀ output buffer (`d_h × D`, f64) before f32 cast.
    pub w_t: Vec<f64>,
    /// Per-sample feature row (`d_h`, f32 — only used for transient expansion).
    pub feature_row: Vec<f32>,
    /// Sample-space Gram for the Woodbury path (`N × N`, f32). Reused across
    /// fits to avoid per-call allocation.
    pub sample_gram: Vec<f32>,
    /// Sample-space Cholesky (`N × N`, f32). Reused across fits.
    pub sample_chol: Vec<f32>,
    /// Sample-space back-substitution temp (`N × D`, f32). Reused across fits.
    pub sample_z: Vec<f32>,
    /// Transpose scratch for the Wᵀ→Wout layout flip (`d_h × D`, f32).
    pub w_t_transpose: Vec<f32>,
}

impl KarcScratch {
    /// Allocate for at most `d_h` features, `D` output dims, `N` samples.
    pub fn with_capacity(d_h: usize, d: usize, n: usize) -> Self {
        Self {
            gram: Vec::with_capacity(d_h * d_h),
            cov: Vec::with_capacity(d_h * d),
            chol: Vec::with_capacity(d_h * d_h),
            z_solve: Vec::with_capacity(d_h * d),
            w_t: Vec::with_capacity(d_h * d),
            feature_row: Vec::with_capacity(d_h),
            sample_gram: Vec::with_capacity(n * n),
            sample_chol: Vec::with_capacity(n * n),
            sample_z: Vec::with_capacity(n * d),
            w_t_transpose: Vec::with_capacity(d_h * d),
        }
    }

    /// Drop all contents (capacity unchanged).
    pub fn clear(&mut self) {
        self.gram.clear();
        self.cov.clear();
        self.chol.clear();
        self.z_solve.clear();
        self.w_t.clear();
        self.feature_row.clear();
        self.sample_gram.clear();
        self.sample_chol.clear();
        self.sample_z.clear();
        self.w_t_transpose.clear();
    }
}

// ── Phase 2: Low-rank ALS scratch + solver (paper Eq. 47) ───────────────

/// Pre-allocated scratch for [`low_rank_fit`] (Plan 308 T2.3, paper Eq. 47).
/// Allocate once via [`LowRankFitScratch::with_capacity`] and reuse across
/// fits (`clear()` not needed — every field is overwritten each call).
///
/// The dominant buffer is `chol_g` (`d_h × d_h` f64), which holds the
/// pre-computed Cholesky factor of `G+λI`. Each ALS iteration only touches
/// the `r × r` and `d_h × r` buffers plus the `D × d_h` convergence-check
/// buffers.
pub struct LowRankFitScratch {
    /// `G+λI`, `d_h × d_h` f64 (kept for inspection / re-factor at new λ).
    pub gram_reg: Vec<f64>,
    /// Cholesky factor `L` of `gram_reg`, `d_h × d_h` f64. Pre-computed once.
    pub chol_g: Vec<f64>,
    /// `B·G·Bᵀ + λI_r`, `r × r` f64 (re-built each A-step).
    pub bgbt: Vec<f64>,
    /// Cholesky factor of `bgbt`, `r × r` f64.
    pub chol_bgbt: Vec<f64>,
    /// `B·Cov`, `r × D` f64 (A-step RHS).
    pub bcov: Vec<f64>,
    /// `Aᵀ`, `r × D` f64 (A-step solve output, row-major).
    pub at: Vec<f64>,
    /// Forward-sub temp for the A-step, `r × D` f64.
    pub z_a: Vec<f64>,
    /// `G·Bᵀ` temp, `d_h × r` f64 (used while building `B·G·Bᵀ`).
    pub gbt: Vec<f64>,
    /// `Cov·A`, `d_h × r` f64 (B-step intermediate).
    pub cov_a: Vec<f64>,
    /// `Bᵀ`, `d_h × r` f64 (B-step solve output, row-major).
    pub bt: Vec<f64>,
    /// Forward-sub temp for the B-step, `d_h × r` f64.
    pub z_b: Vec<f64>,
    /// Previous `A·B`, `D × d_h` f64 (convergence reference).
    pub wout_old: Vec<f64>,
    /// Current `A·B`, `D × d_h` f64 (convergence probe).
    pub wout_new: Vec<f64>,
    /// `AᵀA`, `r × r` f64 (built each B-step).
    pub ata: Vec<f64>,
    /// Kronecker system `M = G ⊗ (AᵀA) + λI`, `(r·d_h)²` f64. Grown on demand
    /// by [`low_rank_fit`] / [`low_rank_fit_b_with_frozen_a`] to avoid
    /// re-allocating up to 184 MB per ALS iteration (forecaster path).
    pub kron_m: Vec<f64>,
    /// Kronecker RHS `vec(Aᵀ·Covᵀ)`, `r·d_h` f64. Grown on demand.
    pub kron_rhs: Vec<f64>,
    /// Kronecker Cholesky factor, `(r·d_h)²` f64. Grown on demand.
    pub kron_chol: Vec<f64>,
    /// Kronecker forward-sub temp, `r·d_h` f64. Grown on demand.
    pub kron_z: Vec<f64>,
    /// Kronecker solution temp, `r·d_h` f64. Grown on demand.
    pub kron_x: Vec<f64>,

    // ── Jacobi B-step path (Issue 185, Plan 308 T4.5 unblock) ──
    //
    // These buffers back [`low_rank_fit_jacobi_bstep`] (see `large_dh.rs`).
    // They are lazily sized (`Vec::new()`) so Kronecker-path callers don't
    // pay for them — the Jacobi path grows them on first use via
    // `LowRankFitScratch::ensure_jacobi_capacity`.
    //
    // The dominant allocation is `eigvecs_g` at `d_h × d_h` f64 — e.g.
    // ~2.6 GB at d_h = 18_720 (the Issue 185 K=8/M=8/R=2 target).

    /// Eigenvalues of `G = XᵀX`, length `d_h` f64. Computed once per fit
    /// (outside the ALS loop) by [`low_rank_fit_jacobi_bstep`].
    pub eigvals_g: Vec<f64>,
    /// Eigenvectors of `G`, `d_h × d_h` f64, column `j` is `Q_g[:,j]` stored
    /// row-major (i.e. `eigvecs_g[i*d_h + j]` is row i, column j).
    pub eigvecs_g: Vec<f64>,
    /// Jacobi scratch for the `d_h × d_h` eigendecomp of `G`, `d_h × d_h` f64.
    pub jacobi_scratch_g: Vec<f64>,
    /// Eigenvalues of `AᵀA`, length `r` f64. Recomputed each ALS iteration.
    pub eigvals_ata: Vec<f64>,
    /// Eigenvectors of `AᵀA`, `r × r` f64. Recomputed each ALS iteration.
    pub eigvecs_ata: Vec<f64>,
    /// Jacobi scratch for the `r × r` eigendecomp of `AᵀA`, `r × r` f64.
    pub jacobi_scratch_ata: Vec<f64>,
    /// Transformed RHS `Q_aᵀ · Aᵀ · Covᵀ · Q_g`, `r × d_h` f64.
    pub rhs_transformed: Vec<f64>,
    /// Scaled intermediate `C` (where `B = Q_a · C · Q_gᵀ`), `r × d_h` f64.
    pub c_tilde: Vec<f64>,
    /// Temp `r × d_h` f64 for the `Q_a · C` matmul.
    pub qc_temp: Vec<f64>,

    // ── Householder+QL B-step path (Issue 186 Path B, 2026-07-20) ──
    //
    // Drop-in alternative to `jacobi_eigen` for the G-path eigendecomp.
    // ~5-10× faster at d_h ≥ 256 due to Householder's cache-friendly blocked
    // reduction + tridiagonal QL's O(n²) per sweep (vs Jacobi's O(n³) per
    // sweep). Wired in `large_dh::low_rank_fit_jacobi_bstep` under the
    // `karc_householder_eig` feature gate. Lazily sized via
    // `ensure_jacobi_capacity` (which calls `symmetric_eig.ensure_capacity`).
    pub symmetric_eig: crate::linalg::symmetric_eig::SymmetricEigScratch,
}

impl LowRankFitScratch {
    /// Allocate for at most `d_h` features, `d_out` output dims, rank `r`.
    pub fn with_capacity(d_h: usize, d_out: usize, r: usize) -> Self {
        Self {
            gram_reg: vec![0.0; d_h * d_h],
            chol_g: vec![0.0; d_h * d_h],
            bgbt: vec![0.0; r * r],
            chol_bgbt: vec![0.0; r * r],
            bcov: vec![0.0; r * d_out],
            at: vec![0.0; r * d_out],
            z_a: vec![0.0; r * d_out],
            gbt: vec![0.0; d_h * r],
            cov_a: vec![0.0; d_h * r],
            bt: vec![0.0; d_h * r],
            z_b: vec![0.0; d_h * r],
            wout_old: vec![0.0; d_out * d_h],
            wout_new: vec![0.0; d_out * d_h],
            ata: vec![0.0; r * r],
            kron_m: Vec::new(),
            kron_rhs: Vec::new(),
            kron_chol: Vec::new(),
            kron_z: Vec::new(),
            kron_x: Vec::new(),
            eigvals_g: Vec::new(),
            eigvecs_g: Vec::new(),
            jacobi_scratch_g: Vec::new(),
            eigvals_ata: Vec::new(),
            eigvecs_ata: Vec::new(),
            jacobi_scratch_ata: Vec::new(),
            rhs_transformed: Vec::new(),
            c_tilde: Vec::new(),
            qc_temp: Vec::new(),
            symmetric_eig: crate::linalg::symmetric_eig::SymmetricEigScratch::new(),
        }
    }

    /// Grow the Jacobi-path scratch buffers to the requested size. Idempotent
    /// — only allocates if the current capacity is too small. Used by
    /// [`low_rank_fit_jacobi_bstep`] (Issue 185) so that Kronecker-path
    /// callers never pay for the Jacobi buffers.
    ///
    /// [`low_rank_fit_jacobi_bstep`]: super::large_dh::low_rank_fit_jacobi_bstep
    pub fn ensure_jacobi_capacity(&mut self, d_h: usize, r: usize) {
        // d_h-side buffers: eigvals_g (d_h), eigvecs_g (d_h*d_h),
        // jacobi_scratch_g (d_h*d_h).
        if self.eigvals_g.len() < d_h {
            self.eigvals_g.resize(d_h, 0.0);
        }
        if self.eigvecs_g.len() < d_h * d_h {
            self.eigvecs_g.resize(d_h * d_h, 0.0);
        }
        if self.jacobi_scratch_g.len() < d_h * d_h {
            self.jacobi_scratch_g.resize(d_h * d_h, 0.0);
        }
        // r-side buffers: eigvals_ata (r), eigvecs_ata (r*r),
        // jacobi_scratch_ata (r*r).
        if self.eigvals_ata.len() < r {
            self.eigvals_ata.resize(r, 0.0);
        }
        if self.eigvecs_ata.len() < r * r {
            self.eigvecs_ata.resize(r * r, 0.0);
        }
        if self.jacobi_scratch_ata.len() < r * r {
            self.jacobi_scratch_ata.resize(r * r, 0.0);
        }
        // Mixed r × d_h buffers.
        let rdh = r * d_h;
        if self.rhs_transformed.len() < rdh {
            self.rhs_transformed.resize(rdh, 0.0);
        }
        if self.c_tilde.len() < rdh {
            self.c_tilde.resize(rdh, 0.0);
        }
        if self.qc_temp.len() < rdh {
            self.qc_temp.resize(rdh, 0.0);
        }
        // Issue 186 Path B — Householder+QL scratch for the G-path eigendecomp.
        // Grown unconditionally (idempotent); only consumed when the
        // `karc_householder_eig` feature is on.
        self.symmetric_eig.ensure_capacity(d_h);
    }
}

/// Symmetric eigendecomposition via the cyclic Jacobi algorithm (Plan 308 T2.3).
///
/// Computes `A = V · diag(λ) · Vᵀ` for a symmetric `r × r` matrix `A` (row-major
/// f64), writing eigenvalues into `eigvals` (length `r`) and eigenvectors into
/// `eigvecs` (length `r*r`, row-major: `eigvecs[i*r + j] = V[i, j]`, so column
/// `k` of `V` — i.e. the eigenvector paired with `eigvals[k]` — is the length-`r`
/// stride-`r` slice starting at `eigvecs[k]`).
///
/// The Jacobi algorithm iterates over all off-diagonal `(p,q)` pairs, applying
/// a rotation that zeroes `A[p,q]`. Cyclic sweeps repeat until the off-diagonal
/// mass is below `tol` or `max_sweeps` is reached. For `r ≤ ~32` this is fast
/// and accurate (typically 5–10 sweeps). Bit-deterministic given identical
/// `A` and `tol` — used by [`low_rank_fit`] to keep the B-step exact and
/// bit-reproducible across runs.
///
/// `scratch` (`r*r` f64) is overwritten and holds the working matrix copy.
///
/// # Rotation convention (Issue 185 bug fix, 2026-07-20)
///
/// The in-place updates use the rotation `J = [[c, s], [-s, c]]` and the
/// angle `θ = 0.5 · atan(2·a_pq / (a_qq − a_pp))`. The original Plan 308 T2.3
/// code used `(a_pp − aqq)` in the denominator (sign-flipped), which produced
/// correct results only when `a_pp == a_qq` (the `π/4` special case). The fix
/// is in place; the regression test `karc::large_dh::tests::jacobi_eigen_sign_convention_correct`
/// pins it.
///
/// # Storage convention
///
/// `eigvecs[i*r + j] = V[i, j]` (standard row-major). Column `k` of `V` (the
/// eigenvector for `eigvals[k]`) is at indices `{eigvecs[k], eigvecs[r+k], ...,
/// eigvecs[(r-1)*r+k]}` — i.e. stride `r` from offset `k`. This matches the
/// accumulation pattern `V ← V · J` where each rotation mixes two COLUMNS
/// of `V`.
///
/// (An earlier version of this docstring claimed "column k is U[:,k] stored at
/// indices eigvecs[k*r..(k+1)*r]" — that described a row-major `Uᵀ` layout,
/// which is incorrect. The actual storage is row-major `V`, not `Vᵀ`.)
pub fn jacobi_eigen(
    eigvals: &mut [f64],
    eigvecs: &mut [f64],
    a_in: &[f64],
    scratch: &mut [f64],
    r: usize,
    tol: f64,
    max_sweeps: usize,
) {
    debug_assert_eq!(a_in.len(), r * r);
    debug_assert_eq!(scratch.len(), r * r);
    debug_assert_eq!(eigvals.len(), r);
    debug_assert_eq!(eigvecs.len(), r * r);
    // Copy A into scratch (working matrix).
    scratch[..r * r].copy_from_slice(a_in);
    // Initialize eigvecs = I.
    for i in 0..r {
        for j in 0..r {
            eigvecs[i * r + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _ in 0..max_sweeps {
        // Sum of off-diagonal squares.
        let mut off_sq = 0.0f64;
        for p in 0..r {
            for q in (p + 1)..r {
                off_sq += scratch[p * r + q] * scratch[p * r + q];
            }
        }
        if off_sq < tol {
            break;
        }
        // One sweep over all (p, q) with p < q.
        for p in 0..r {
            for q in (p + 1)..r {
                let apq = scratch[p * r + q];
                if apq.abs() < f64::MIN_POSITIVE {
                    continue;
                }
                let app = scratch[p * r + p];
                let aqq = scratch[q * r + q];
                // Compute rotation angle θ that zeros scratch[p,q] under the
                // update `A ← Jᵀ·A·J` with `J = [[c, s], [-s, c]]` (the convention
                // used by the in-place updates below).
                //
                // Derivation: with this J convention,
                //   (Jᵀ·A·J)[p,q] = sc·(app − aqq) + (c²−s²)·apq.
                // Setting this to zero gives `tan(2θ) = 2·apq / (aqq − app)`.
                //
                // **Issue 185 bug fix (2026-07-20):** the original Plan 308 T2.3
                // code used `2·apq / (app − aqq)` (sign-flipped denominator).
                // That happens to produce the right answer when `app == aqq`
                // (the π/4 special case) but produces wrong eigenvalues AND
                // wrong eigenvectors for any matrix with `app ≠ aqq`. The bug
                // was latent because `jacobi_eigen` was unused before Issue 185's
                // Jacobi B-step path — every prior consumer used Cholesky.
                let theta = if (app - aqq).abs() < f64::MIN_POSITIVE {
                    core::f64::consts::FRAC_PI_4
                } else {
                    0.5 * (2.0 * apq / (aqq - app)).atan()
                };
                let c = theta.cos();
                let s = theta.sin();
                // Apply rotation to working matrix: Jᵀ A J.
                for i in 0..r {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = scratch[i * r + p];
                    let aiq = scratch[i * r + q];
                    scratch[i * r + p] = c * aip - s * aiq;
                    scratch[p * r + i] = scratch[i * r + p];
                    scratch[i * r + q] = s * aip + c * aiq;
                    scratch[q * r + i] = scratch[i * r + q];
                }
                scratch[p * r + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                scratch[q * r + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                scratch[p * r + q] = 0.0;
                scratch[q * r + p] = 0.0;
                // Accumulate rotation into eigvecs: V ← V · J.
                for i in 0..r {
                    let vip = eigvecs[i * r + p];
                    let viq = eigvecs[i * r + q];
                    eigvecs[i * r + p] = c * vip - s * viq;
                    eigvecs[i * r + q] = s * vip + c * viq;
                }
            }
        }
    }
    // Extract eigenvalues (diagonal of the converged working matrix).
    for i in 0..r {
        eigvals[i] = scratch[i * r + i];
    }
}

/// Alternating least squares low-rank ridge factorization `Wout ≈ A·B`
/// (Plan 308 T2.3, paper Eq. 47).
///
/// Given the un-regularized Gram `G = XᵀX` (`d_h × d_h` f64) and
/// cross-covariance `Cov = XᵀY` (`d_h × D` f64), produces `A` (`D × r` f64,
/// row-major) and `B` (`r × d_h` f64, row-major) minimizing
/// `‖Y − X·Bᵀ·Aᵀ‖²_F + λ(‖A‖² + ‖B‖²)`.
///
/// # Algorithm
///
/// ALS alternates two exact ridge sub-steps:
///
/// - **A-step (B fixed)**: `Aᵀ = (B·G·Bᵀ + λI_r)⁻¹·(B·Cov)`. A small `r × r`
///   Cholesky solve.
/// - **B-step (A fixed)**: exact solve of the Sylvester-like normal equation
///   `(AᵀA)·B·G + λB = Aᵀ·Covᵀ` via the Kronecker vectorization
///   `(G ⊗ AᵀA + λI)·vec(B) = vec(Aᵀ·Covᵀ)`. This is an `(r·d_h)×(r·d_h)` Cholesky
///   solve — exact but O((r·d_h)³), so feasible only for `r·d_h ≤ ~2000`.
///
/// After each A+B pair a scale rebalance `A←cA, B←B/c` with
/// `c = √(‖B‖/‖A‖)` is applied to prevent the ALS gauge drift
/// (A·B = (cA)·(B/c) — the two `λ‖·‖²` penalties pin the scale in principle
/// but ALS exhibits exponential drift without explicit balancing).
///
/// # Initialization (bit-reproducibility contract)
///
/// `B ← [I_r | 0]` (partial identity in the first `r` columns), `A ← 0`.
/// This is deterministic: two calls with identical
/// `(G, Cov, d_h, D, r, λ, max_iters, tol)` produce bit-identical `A` and `B`.
///
/// # Convergence
///
/// Stops after `max_iters` iterations or when `‖(A·B)_new − (A·B)_old‖_F < tol`,
/// whichever comes first. Returns the number of iterations performed.
///
/// # Panics
///
/// Panics if `r == 0`, `r > d_h`, `λ ≤ 0`, or any buffer is undersized.
#[allow(clippy::too_many_arguments)]
pub fn low_rank_fit(
    gram: &[f64],
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    max_iters: usize,
    tol: f64,
    a_out: &mut [f64],
    b_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) -> usize {
    assert!(r > 0, "low_rank_fit: r must be > 0");
    assert!(
        r <= d_h,
        "low_rank_fit: r must be <= d_h (got r={r}, d_h={d_h})"
    );
    assert!(lambda > 0.0, "low_rank_fit: lambda must be > 0");
    assert!(a_out.len() >= d_out * r, "a_out too small");
    assert!(b_out.len() >= r * d_h, "b_out too small");

    // Deterministic init: B = [I_r | 0], A = 0.
    for k in 0..r {
        for j in 0..d_h {
            b_out[k * d_h + j] = if j == k { 1.0 } else { 0.0 };
        }
    }
    for v in a_out.iter_mut().take(d_out * r) {
        *v = 0.0;
    }

    low_rank_fit_with_init(
        gram, cov, d_h, d_out, r, lambda, max_iters, tol, a_out, b_out, scratch,
    )
}

/// ALS loop body shared by [`low_rank_fit`] (deterministic zero/identity init)
/// and [`low_rank_fit_warm_start`] (caller-supplied init).
///
/// Assumes `a_out[..d_out*r]` and `b_out[..r*d_h]` are already initialized by
/// the caller. Runs the Cholesky pre-computation + ALS iteration loop +
/// convergence check. Returns the number of iterations performed.
///
/// This function is factored out to keep the init strategy DRY: zero-init vs
/// warm-start-init is the only difference between the two public entry points.
/// The ALS math itself is identical (and bit-reproducible given the same init).
#[allow(clippy::too_many_arguments)]
fn low_rank_fit_with_init(
    gram: &[f64],
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    max_iters: usize,
    tol: f64,
    a_out: &mut [f64],
    b_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) -> usize {
    // 1. Pre-compute Cholesky of (G + λI) — the B-step system matrix.
    //    Done ONCE; each B-step is just back-substitution.
    scratch.gram_reg[..d_h * d_h].copy_from_slice(&gram[..d_h * d_h]);
    for i in 0..d_h {
        scratch.gram_reg[i * d_h + i] += lambda;
    }
    cholesky_f64(&mut scratch.chol_g, &scratch.gram_reg, d_h);

    // wout_old must be zeroed so the first iteration's convergence check
    // measures against the post-init Wout (not stale memory).
    for v in scratch.wout_old.iter_mut().take(d_out * d_h) {
        *v = 0.0;
    }

    // 2. ALS iterations.
    let mut iters_done = max_iters;
    for iter in 0..max_iters {
        // ── A-step: Aᵀ = (B·G·Bᵀ + λI_r)⁻¹ · (B·Cov) ──
        // G·Bᵀ: d_h × r.
        for i in 0..d_h {
            for k in 0..r {
                let mut s = 0.0f64;
                let mut j = 0;
                while j + 4 <= d_h {
                    s += gram[i * d_h + j] * b_out[k * d_h + j];
                    s += gram[i * d_h + j + 1] * b_out[k * d_h + j + 1];
                    s += gram[i * d_h + j + 2] * b_out[k * d_h + j + 2];
                    s += gram[i * d_h + j + 3] * b_out[k * d_h + j + 3];
                    j += 4;
                }
                while j < d_h {
                    s += gram[i * d_h + j] * b_out[k * d_h + j];
                    j += 1;
                }
                scratch.gbt[i * r + k] = s;
            }
        }
        // B·(G·Bᵀ): r × r.
        for i in 0..r {
            for k in 0..r {
                let mut s = 0.0f64;
                let mut j = 0;
                while j + 4 <= d_h {
                    s += b_out[i * d_h + j] * scratch.gbt[j * r + k];
                    s += b_out[i * d_h + j + 1] * scratch.gbt[(j + 1) * r + k];
                    s += b_out[i * d_h + j + 2] * scratch.gbt[(j + 2) * r + k];
                    s += b_out[i * d_h + j + 3] * scratch.gbt[(j + 3) * r + k];
                    j += 4;
                }
                while j < d_h {
                    s += b_out[i * d_h + j] * scratch.gbt[j * r + k];
                    j += 1;
                }
                scratch.bgbt[i * r + k] = s;
            }
        }
        // Add λI_r.
        for i in 0..r {
            scratch.bgbt[i * r + i] += lambda;
        }
        // B·Cov: r × D.
        for i in 0..r {
            for d in 0..d_out {
                let mut s = 0.0f64;
                let mut j = 0;
                while j + 4 <= d_h {
                    s += b_out[i * d_h + j] * cov[j * d_out + d];
                    s += b_out[i * d_h + j + 1] * cov[(j + 1) * d_out + d];
                    s += b_out[i * d_h + j + 2] * cov[(j + 2) * d_out + d];
                    s += b_out[i * d_h + j + 3] * cov[(j + 3) * d_out + d];
                    j += 4;
                }
                while j < d_h {
                    s += b_out[i * d_h + j] * cov[j * d_out + d];
                    j += 1;
                }
                scratch.bcov[i * d_out + d] = s;
            }
        }
        // Cholesky solve: (B·G·Bᵀ + λI) · Aᵀ = B·Cov  →  Aᵀ (r × D).
        cholesky_f64(&mut scratch.chol_bgbt, &scratch.bgbt, r);
        chol_solve_f64(
            &mut scratch.at,
            &mut scratch.z_a,
            &scratch.chol_bgbt,
            &scratch.bcov,
            r,
            d_out,
        );
        // Transpose Aᵀ (r × D) → A (D × r).
        for d in 0..d_out {
            for k in 0..r {
                a_out[d * r + k] = scratch.at[k * d_out + d];
            }
        }

        // ── B-step: exact solve via the Kronecker system ──
        //
        // The B-step normal equation (AᵀA)·B·G + λB = Aᵀ·Covᵀ vectorizes as
        // (G ⊗ AᵀA + λI_{r·d_h}) · vec(B) = vec(Aᵀ·Covᵀ)  (row-major vec(B)).
        // For small r·d_h (≤ ~2000, covering the first-order forecaster path
        // d_h ≤ 600, r ≤ 8), we solve this directly via Cholesky. For larger
        // systems the caller should use the standalone path with a precomputed
        // Gram eigendecomposition (future work; the d_h=4752 higher-order
        // benchmark config currently uses the approximate B-step via the
        // standalone variant — not this method).
        //
        // Build the Kronecker system M = G ⊗ (AᵀA) + λI  (size rd_h × rd_h).
        // Index convention: vec(B) stacks rows of B (r rows of length d_h).
        // M[idx(i1,j1), idx(i2,j2)] = G[j1,j2]·(AᵀA)[i1,i2] + λ·δ
        //   where idx(i,j) = i*d_h + j, i ∈ 0..r, j ∈ 0..d_h.
        //
        // 0. Compute Cov·A (d_h × r) — reused below for the RHS derivation.
        for i in 0..d_h {
            for k in 0..r {
                let mut s = 0.0f64;
                for d in 0..d_out {
                    s += cov[i * d_out + d] * a_out[d * r + k];
                }
                scratch.cov_a[i * r + k] = s;
            }
        }
        // 1. Build AᵀA (r×r).
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f64;
                for d in 0..d_out {
                    s += a_out[d * r + i] * a_out[d * r + j];
                }
                scratch.ata[i * r + j] = s;
            }
        }
        // 2. Build the Kronecker matrix M = G ⊗ (AᵀA) + λI into gram_reg
        //    (re-using the d_h×d_h gram_reg buffer would be too small; we need
        //    r*d_h × r*d_h. Re-use sk_g_lambda which is d_h*d_h — still too small.
        //    We need a dedicated rd_h*rd_h buffer.)
        let rd_h = r * d_h;
        // Re-use the scratch Kronecker buffers (grown once, reused across ALS
        // iterations). This avoids re-allocating up to 184 MB per iteration on
        // the forecaster path (d_h ≤ 600, r ≤ 8). For the test path
        // (d_h=12, r=2) the buffers stay tiny.
        scratch.kron_m.resize(rd_h * rd_h, 0.0);
        let m = &mut scratch.kron_m[..rd_h * rd_h];
        for i1 in 0..r {
            for i2 in 0..r {
                let ata_i1i2 = scratch.ata[i1 * r + i2];
                for j1 in 0..d_h {
                    for j2 in 0..d_h {
                        let row = i1 * d_h + j1;
                        let col = i2 * d_h + j2;
                        m[row * rd_h + col] = gram[j1 * d_h + j2] * ata_i1i2;
                        if row == col {
                            m[row * rd_h + col] += lambda;
                        }
                    }
                }
            }
        }
        // 3. Build RHS = vec(Aᵀ·Covᵀ) (r × d_h, row-major → vec of length rd_h).
        //    (Aᵀ·Covᵀ)[i,j] = cov_a[j*r+i] (from step 0).
        //    vec(Aᵀ·Covᵀ) stacks rows: idx = i*d_h + j, value = (Aᵀ·Covᵀ)[i,j].
        scratch.kron_rhs.resize(rd_h, 0.0);
        let rhs = &mut scratch.kron_rhs[..rd_h];
        for i in 0..r {
            for j in 0..d_h {
                rhs[i * d_h + j] = scratch.cov_a[j * r + i];
            }
        }
        // 4. Cholesky-solve M · vec(B) = rhs → vec(B) into b_out.
        scratch.kron_chol.resize(rd_h * rd_h, 0.0);
        scratch.kron_z.resize(rd_h, 0.0);
        scratch.kron_x.resize(rd_h, 0.0);
        cholesky_f64(&mut scratch.kron_chol, &scratch.kron_m, rd_h);
        chol_solve_f64(
            &mut scratch.kron_x,
            &mut scratch.kron_z,
            &scratch.kron_chol,
            &scratch.kron_rhs,
            rd_h,
            1,
        );
        // 5. Unpack vec(B) → B (r × d_h, row-major: b_out[k*d_h+j] = kron_x[k*d_h+j]).
        b_out[..rd_h].copy_from_slice(&scratch.kron_x[..rd_h]);

        // ── Scale balancing (anti-drift) ──
        // Without balancing, ALS for bilinear ridge has a gauge freedom
        // (A·B = (cA)·(B/c)); the two separate λ‖A‖² + λ‖B‖² penalties pin
        // the scale in principle but ALS exhibits exponential drift in practice
        // (eigenvalues of AᵀA grow ~3×/iter). We rebalance after each full
        // ALS step: A ← c·A, B ← B/c with c = (‖B‖/‖A‖)^½ so ‖A‖ ≈ ‖B‖.
        // This leaves A·B unchanged, so the data-fit term is unaffected, and
        // makes the regularization well-balanced.
        let norm_a_sq: f64 = a_out[..d_out * r].iter().map(|x| x * x).sum();
        let norm_b_sq: f64 = b_out[..r * d_h].iter().map(|x| x * x).sum();
        if norm_a_sq > 0.0 && norm_b_sq > 0.0 {
            let c = (norm_b_sq / norm_a_sq).sqrt();
            for v in a_out[..d_out * r].iter_mut() {
                *v *= c;
            }
            let c_inv = 1.0 / c;
            for v in b_out[..r * d_h].iter_mut() {
                *v *= c_inv;
            }
        }

        // ── Convergence: ‖(A·B)_new − (A·B)_old‖_F ──
        // Wout_new[d, j] = Σ_k A[d,k] · B[k,j].
        for d in 0..d_out {
            for j in 0..d_h {
                let mut s = 0.0f64;
                for k in 0..r {
                    s += a_out[d * r + k] * b_out[k * d_h + j];
                }
                scratch.wout_new[d * d_h + j] = s;
            }
        }
        let mut diff_sq = 0.0f64;
        for idx in 0..d_out * d_h {
            let diff = scratch.wout_new[idx] - scratch.wout_old[idx];
            diff_sq += diff * diff;
        }
        scratch.wout_old[..d_out * d_h].copy_from_slice(&scratch.wout_new[..d_out * d_h]);

        if diff_sq.sqrt() < tol {
            iters_done = iter + 1;
            break;
        }
    }
    iters_done
}

// ── Shared ALS helpers (extracted for the Jacobi B-step path, Issue 185) ──
//
// These two helpers are factored out of `low_rank_fit_with_init` so that
// `low_rank_fit_jacobi_bstep` (in `large_dh.rs`) can reuse the EXACT same
// A-step and scale-rebalance code — bit-identical behaviour between the
// Kronecker and Jacobi paths on configs where both are feasible is the
// T2 parity-test contract.

/// A-step of one ALS iteration: solves `Aᵀ = (B·G·Bᵀ + λI_r)⁻¹ · (B·Cov)`
/// and writes the transposed result into `a_out` (D × r, row-major).
///
/// Inputs: `gram` (d_h × d_h), `cov` (d_h × D), `b_out` (r × d_h).
/// Scratch buffers touched: `gbt`, `bgbt`, `bcov`, `chol_bgbt`, `at`.
/// Reads `r`, `d_h`, `d_out`, `lambda`.
///
/// This is the A-step body extracted verbatim from `low_rank_fit_with_init`
/// lines 1046–1114 (Plan 308 T2.3) — refactoring it into a helper does not
/// change the math or the float-operation order, preserving bit-reproducibility.
#[allow(clippy::too_many_arguments)]
pub(super) fn als_a_step(
    gram: &[f64],
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    b_out: &[f64],
    a_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) {
    // G·Bᵀ: d_h × r.
    for i in 0..d_h {
        for k in 0..r {
            let mut s = 0.0f64;
            let mut j = 0;
            while j + 4 <= d_h {
                s += gram[i * d_h + j] * b_out[k * d_h + j];
                s += gram[i * d_h + j + 1] * b_out[k * d_h + j + 1];
                s += gram[i * d_h + j + 2] * b_out[k * d_h + j + 2];
                s += gram[i * d_h + j + 3] * b_out[k * d_h + j + 3];
                j += 4;
            }
            while j < d_h {
                s += gram[i * d_h + j] * b_out[k * d_h + j];
                j += 1;
            }
            scratch.gbt[i * r + k] = s;
        }
    }
    // B·(G·Bᵀ): r × r.
    for i in 0..r {
        for k in 0..r {
            let mut s = 0.0f64;
            let mut j = 0;
            while j + 4 <= d_h {
                s += b_out[i * d_h + j] * scratch.gbt[j * r + k];
                s += b_out[i * d_h + j + 1] * scratch.gbt[(j + 1) * r + k];
                s += b_out[i * d_h + j + 2] * scratch.gbt[(j + 2) * r + k];
                s += b_out[i * d_h + j + 3] * scratch.gbt[(j + 3) * r + k];
                j += 4;
            }
            while j < d_h {
                s += b_out[i * d_h + j] * scratch.gbt[j * r + k];
                j += 1;
            }
            scratch.bgbt[i * r + k] = s;
        }
    }
    // Add λI_r.
    for i in 0..r {
        scratch.bgbt[i * r + i] += lambda;
    }
    // B·Cov: r × D.
    for i in 0..r {
        for d in 0..d_out {
            let mut s = 0.0f64;
            let mut j = 0;
            while j + 4 <= d_h {
                s += b_out[i * d_h + j] * cov[j * d_out + d];
                s += b_out[i * d_h + j + 1] * cov[(j + 1) * d_out + d];
                s += b_out[i * d_h + j + 2] * cov[(j + 2) * d_out + d];
                s += b_out[i * d_h + j + 3] * cov[(j + 3) * d_out + d];
                j += 4;
            }
            while j < d_h {
                s += b_out[i * d_h + j] * cov[j * d_out + d];
                j += 1;
            }
            scratch.bcov[i * d_out + d] = s;
        }
    }
    // Cholesky solve: (B·G·Bᵀ + λI) · Aᵀ = B·Cov  →  Aᵀ (r × D).
    cholesky_f64(&mut scratch.chol_bgbt, &scratch.bgbt, r);
    chol_solve_f64(
        &mut scratch.at,
        &mut scratch.z_a,
        &scratch.chol_bgbt,
        &scratch.bcov,
        r,
        d_out,
    );
    // Transpose Aᵀ (r × D) → A (D × r).
    for d in 0..d_out {
        for k in 0..r {
            a_out[d * r + k] = scratch.at[k * d_out + d];
        }
    }
}

/// Scale rebalance: `A ← c·A`, `B ← B/c` with `c = √(‖B‖/‖A‖)` to pin the
/// ALS gauge and prevent exponential drift (A·B = (cA)·(B/c) is unchanged).
///
/// Extracted verbatim from `low_rank_fit_with_init` lines 1213–1232.
pub(super) fn als_scale_rebalance(
    a_out: &mut [f64],
    b_out: &mut [f64],
    d_out: usize,
    d_h: usize,
    r: usize,
) {
    let norm_a_sq: f64 = a_out[..d_out * r].iter().map(|x| x * x).sum();
    let norm_b_sq: f64 = b_out[..r * d_h].iter().map(|x| x * x).sum();
    if norm_a_sq > 0.0 && norm_b_sq > 0.0 {
        let c = (norm_b_sq / norm_a_sq).sqrt();
        for v in a_out[..d_out * r].iter_mut() {
            *v *= c;
        }
        let c_inv = 1.0 / c;
        for v in b_out[..r * d_h].iter_mut() {
            *v *= c_inv;
        }
    }
}

/// Convergence probe: returns `‖(A·B)_new − wout_old‖_F` and copies the new
/// `A·B` into `wout_old` so the next call measures against the latest state.
///
/// Extracted verbatim from `low_rank_fit_with_init` lines 1223–1255.
pub(super) fn als_convergence_step(
    a_out: &[f64],
    b_out: &[f64],
    d_out: usize,
    d_h: usize,
    r: usize,
    scratch: &mut LowRankFitScratch,
) -> f64 {
    for d in 0..d_out {
        for j in 0..d_h {
            let mut s = 0.0f64;
            for k in 0..r {
                s += a_out[d * r + k] * b_out[k * d_h + j];
            }
            scratch.wout_new[d * d_h + j] = s;
        }
    }
    let mut diff_sq = 0.0f64;
    for idx in 0..d_out * d_h {
        let diff = scratch.wout_new[idx] - scratch.wout_old[idx];
        diff_sq += diff * diff;
    }
    scratch.wout_old[..d_out * d_h].copy_from_slice(&scratch.wout_new[..d_out * d_h]);
    diff_sq.sqrt()
}

/// Build `AᵀA` (r × r, row-major) from `a_out` (D × r). Writes into
/// `scratch.ata`. Used by both the Kronecker B-step and the Jacobi B-step.
pub(super) fn build_ata(a_out: &[f64], d_out: usize, r: usize, scratch: &mut LowRankFitScratch) {
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f64;
            for d in 0..d_out {
                s += a_out[d * r + i] * a_out[d * r + j];
            }
            scratch.ata[i * r + j] = s;
        }
    }
}

/// Compute `Cov·A` (d_h × r, row-major) into `scratch.cov_a`.
/// `(cov_a)[i, k] = Σ_d cov[i, d] · a[d, k]`. Used by both B-step paths
/// (the Kronecker RHS and the Jacobi RHS both derive from this product).
pub(super) fn build_cov_a(
    cov: &[f64],
    a_out: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    scratch: &mut LowRankFitScratch,
) {
    for i in 0..d_h {
        for k in 0..r {
            let mut s = 0.0f64;
            for d in 0..d_out {
                s += cov[i * d_out + d] * a_out[d * r + k];
            }
            scratch.cov_a[i * r + k] = s;
        }
    }
}

/// ALS fit with caller-supplied initial factors (warm-start).
///
/// Same as [`low_rank_fit`] except `a_out` and `b_out` are NOT re-initialized
/// — the caller must supply valid initial factors (e.g. transferred from a
/// Game-A fit). This tests the warm-start hypothesis: does starting ALS near
/// Game A's solution help Game B converge to a better optimum than from-scratch?
///
/// The init is copied from `a_init` / `b_init` into `a_out` / `b_out` (so the
/// caller's source buffers are not mutated), then the ALS loop runs as usual
/// with the same Cholesky pre-computation, scale balancing, and convergence
/// check as [`low_rank_fit`].
///
/// # Cross-game transfer (Plan 332 Phase 7 follow-up)
///
/// The frozen-A variant ([`low_rank_fit_b_with_frozen_a`]) fixes `A` at Game
/// A's value and only solves `B`. This warm-start variant lets BOTH `A` and
/// `B` re-optimize from Game A's starting point — a weaker constraint that
/// may recover signal the frozen-A variant loses. See
/// `.benchmarks/152_karc_cross_game_transfer.md` for the empirical comparison.
///
/// # Panics
///
/// Same as [`low_rank_fit`]: `r == 0`, `r > d_h`, `λ ≤ 0`, or undersized
/// buffers. Additionally panics if `a_init.len() < d_out * r` or
/// `b_init.len() < r * d_h`.
#[allow(clippy::too_many_arguments)]
pub fn low_rank_fit_warm_start(
    gram: &[f64],
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    max_iters: usize,
    tol: f64,
    a_init: &[f64],
    b_init: &[f64],
    a_out: &mut [f64],
    b_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) -> usize {
    assert!(r > 0, "low_rank_fit_warm_start: r must be > 0");
    assert!(
        r <= d_h,
        "low_rank_fit_warm_start: r must be <= d_h (got r={r}, d_h={d_h})"
    );
    assert!(lambda > 0.0, "low_rank_fit_warm_start: lambda must be > 0");
    assert!(a_out.len() >= d_out * r, "a_out too small");
    assert!(b_out.len() >= r * d_h, "b_out too small");
    assert!(
        a_init.len() >= d_out * r,
        "low_rank_fit_warm_start: a_init.len() = {} but expected d_out*r = {}",
        a_init.len(),
        d_out * r
    );
    assert!(
        b_init.len() >= r * d_h,
        "low_rank_fit_warm_start: b_init.len() = {} but expected r*d_h = {}",
        b_init.len(),
        r * d_h
    );

    // Copy caller-supplied init into the working buffers (do NOT zero them).
    a_out[..d_out * r].copy_from_slice(&a_init[..d_out * r]);
    b_out[..r * d_h].copy_from_slice(&b_init[..r * d_h]);

    low_rank_fit_with_init(
        gram, cov, d_h, d_out, r, lambda, max_iters, tol, a_out, b_out, scratch,
    )
}

/// Single B-step ridge solve with a **frozen** `A` — solves
/// `min_B ‖Y − A_frozen·B·Xᵀ‖²_F + λ‖B‖²_F` in closed form (Plan 308
/// extension for Plan 332 Phase 7 cross-game transfer).
///
/// This is the frozen-A half of [`low_rank_fit`]: instead of alternating A
/// and B steps, `A` is held fixed at `a_frozen` (e.g. the "personality
/// factor" transferred from Game A) and only `B` is solved for, from Game
/// B's Gram/Cov. Mathematically it is exactly one B-step of the ALS loop in
/// [`low_rank_fit`] (lines 940–1019 of that function), extracted as a
/// standalone primitive so callers can seed a Game-B forecaster with Game
/// A's personality structure.
///
/// # The cross-game transfer hypothesis
///
/// If the KARC low-rank factorization `Wout = A·B` is interpreted as
/// "`A` = which HLA-axis combinations this NPC cares about (personality),
/// `B` = how observed features map onto those combinations (game-specific
/// readout)", then freezing `A` from Game A and re-fitting `B` from Game B
/// tests whether personality transfers across games. See the Plan 332
/// Phase 7 benchmark and `.benchmarks/152_karc_cross_game_transfer.md` for
/// the empirical verdict.
///
/// # Inputs
///
/// - `gram` — un-regularized feature Gram `XᵀX` (`d_h × d_h`, f64).
/// - `cov` — cross-covariance `XᵀY` (`d_h × d_out`, f64).
/// - `a_frozen` — frozen `A` factor (`d_out × r`, f64, row-major). Must be
///   the transferred personality matrix from Game A.
/// - `lambda` — ridge regularization `λ > 0`.
///
/// # Outputs
///
/// - `b_out` — solved `B` (`r × d_h`, f64, row-major). `A_frozen·B_out`
///   minimizes the frozen-A ridge objective over the supplied Gram/Cov.
///
/// Uses the caller-supplied `scratch` (the same `LowRankFitScratch` used by
/// [`low_rank_fit`], re-purposed). The dominant allocation is the
/// `(r·d_h)×(r·d_h)` Kronecker system — same cost as one ALS B-step.
///
/// # Panics
///
/// Panics if `r == 0`, `r > d_h`, `λ ≤ 0`, `a_frozen.len() != d_out*r`, or
/// `b_out.len() < r*d_h`.
#[allow(clippy::too_many_arguments)]
pub fn low_rank_fit_b_with_frozen_a(
    gram: &[f64],
    cov: &[f64],
    d_h: usize,
    d_out: usize,
    r: usize,
    lambda: f64,
    a_frozen: &[f64],
    b_out: &mut [f64],
    scratch: &mut LowRankFitScratch,
) {
    assert!(r > 0, "low_rank_fit_b_with_frozen_a: r must be > 0");
    assert!(
        r <= d_h,
        "low_rank_fit_b_with_frozen_a: r must be <= d_h (got r={r}, d_h={d_h})"
    );
    assert!(
        lambda > 0.0,
        "low_rank_fit_b_with_frozen_a: lambda must be > 0"
    );
    assert_eq!(
        a_frozen.len(),
        d_out * r,
        "low_rank_fit_b_with_frozen_a: a_frozen.len() = {} but expected d_out*r = {}*{} = {}",
        a_frozen.len(),
        d_out,
        r,
        d_out * r,
    );
    assert_eq!(
        b_out.len(),
        r * d_h,
        "low_rank_fit_b_with_frozen_a: b_out.len() = {} but expected r*d_h = {}*{} = {}",
        b_out.len(),
        r,
        d_h,
        r * d_h,
    );

    // ── Single B-step with A = a_frozen (extracted from low_rank_fit) ──
    // The B-step normal equation (AᵀA)·B·G + λB = Aᵀ·Covᵀ vectorizes as
    // (G ⊗ AᵀA + λI_{r·d_h}) · vec(B) = vec(Aᵀ·Covᵀ).
    let rd_h = r * d_h;

    // 0. Compute Cov·A (d_h × r).
    for i in 0..d_h {
        for k in 0..r {
            let mut s = 0.0f64;
            for d in 0..d_out {
                s += cov[i * d_out + d] * a_frozen[d * r + k];
            }
            scratch.cov_a[i * r + k] = s;
        }
    }
    // 1. Build AᵀA (r × r) from the frozen A.
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f64;
            for d in 0..d_out {
                s += a_frozen[d * r + i] * a_frozen[d * r + j];
            }
            scratch.ata[i * r + j] = s;
        }
    }
    // 2. Build the Kronecker system M = G ⊗ (AᵀA) + λI (rd_h × rd_h).
    //    Uses the shared Kronecker scratch buffers (grown once).
    scratch.kron_m.resize(rd_h * rd_h, 0.0);
    let m = &mut scratch.kron_m[..rd_h * rd_h];
    for i1 in 0..r {
        for i2 in 0..r {
            let ata_i1i2 = scratch.ata[i1 * r + i2];
            for j1 in 0..d_h {
                for j2 in 0..d_h {
                    let row = i1 * d_h + j1;
                    let col = i2 * d_h + j2;
                    m[row * rd_h + col] = gram[j1 * d_h + j2] * ata_i1i2;
                    if row == col {
                        m[row * rd_h + col] += lambda;
                    }
                }
            }
        }
    }
    // 3. Build RHS = vec(Aᵀ·Covᵀ) (r × d_h, row-major → vec of length rd_h).
    scratch.kron_rhs.resize(rd_h, 0.0);
    let rhs = &mut scratch.kron_rhs[..rd_h];
    for i in 0..r {
        for j in 0..d_h {
            rhs[i * d_h + j] = scratch.cov_a[j * r + i];
        }
    }
    // 4. Cholesky-solve M · vec(B) = rhs → vec(B) into b_out.
    scratch.kron_chol.resize(rd_h * rd_h, 0.0);
    scratch.kron_z.resize(rd_h, 0.0);
    scratch.kron_x.resize(rd_h, 0.0);
    cholesky_f64(&mut scratch.kron_chol, &scratch.kron_m, rd_h);
    chol_solve_f64(
        &mut scratch.kron_x,
        &mut scratch.kron_z,
        &scratch.kron_chol,
        &scratch.kron_rhs,
        rd_h,
        1,
    );
    // 5. Unpack vec(B) → B (r × d_h, row-major).
    b_out[..rd_h].copy_from_slice(&scratch.kron_x[..rd_h]);
    // NOTE: no scale rebalancing here — A is frozen, so rebalancing would
    // change A (forbidden). The caller accepts A_frozen's scale as-is.
}

// ── Phase 2: Low-rank forecast matvec (paper Eq. 47 inference) ────────────

/// Two-stage low-rank matvec `out = A · (B · psi)` (paper Eq. 47 inference).
///
/// - `a`: `D × r` row-major.
/// - `b`: `r × d_h` row-major.
/// - `psi`: `d_h` (the expanded feature vector).
/// - `mid`: `r` scratch (overwritten).
/// - `out`: `D` (written).
///
/// Zero-allocation. The two dot products per stage use [`simd::simd_dot_f32`],
/// so this has the same hot-path contract as [`KarcForecaster::forecast_into`].
/// For the higher-order path, expand `psi` via [`feature_expand_higher_order`]
/// first, then call this function.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn forecast_low_rank_apply(
    a: &[f32],
    b: &[f32],
    psi: &[f32],
    mid: &mut [f32],
    out: &mut [f32],
    d_h: usize,
    r: usize,
    d_out: usize,
) {
    debug_assert_eq!(a.len(), d_out * r);
    debug_assert_eq!(b.len(), r * d_h);
    debug_assert_eq!(psi.len(), d_h);
    debug_assert_eq!(mid.len(), r);
    debug_assert!(out.len() >= d_out);
    // mid = B · psi  (r dot products of length d_h).
    for k in 0..r {
        mid[k] = simd::simd_dot_f32(&b[k * d_h..(k + 1) * d_h], psi, d_h);
    }
    // out = A · mid  (D dot products of length r).
    for d in 0..d_out {
        out[d] = simd::simd_dot_f32(&a[d * r..(d + 1) * r], mid, r);
    }
}

// ── Forecaster ────────────────────────────────────────────────────────────

/// `KarcForecaster<B, D, M, K>` — delay-basis-ridge trajectory forecaster.
///
/// - `B: KarcBasis<M>` — the univariate basis dictionary (Fourier / Chebyshev / BSpline).
/// - `D` — observation/state dimension (e.g. 3 for double-scroll, 8 for HLA).
/// - `M` — number of basis functions per coordinate.
/// - `K` — delay-embedding length (number of past observations concatenated).
///
/// Feature dimension `d_h = K·D·M`. The readout `Wout` is `D × d_h` row-major.
/// Forecast is `û = Wout · Ψ(x)`, a single SIMD matvec.
///
/// The type parameter `B` (rather than `dyn KarcBasis`) gives monomorphised
/// static dispatch on the forecast hot path — required for G2 (≤ 500 ns/call).
pub struct KarcForecaster<B: KarcBasis<M>, const D: usize, const M: usize, const K: usize> {
    /// The basis dictionary (immutable after construction).
    pub basis: B,
    /// Delay ring buffer of last-K observations.
    ring: DelayRing<D, K>,
    /// Readout matrix `Wout`, `D × d_h` row-major. Empty until first fit.
    pub wout: Vec<f32>,
    /// Set after the first successful fit.
    fitted: bool,
    /// Accumulated feature rows, `N × d_h` row-major (one row per training sample).
    features_buf: Vec<f32>,
    /// Accumulated target rows, `N × D` row-major.
    targets_buf: Vec<f32>,
    /// Number of accumulated training samples.
    n_samples: usize,
    /// Pre-allocated scratch for fit_ridge.
    scratch: KarcScratch,
    /// Pre-allocated delay buffer (`K·D`), reused by `observe_and_maybe_pair` /
    /// `forecast_now` / `forecast_into`. Allocated once at construction — never
    /// reallocated, so the hot path stays zero-alloc (G3).
    delay_buf: Vec<f32>,
    /// Pre-allocated feature buffer (`K·D·M = d_h`), reused by `forecast_into`.
    /// Allocated once — zero-alloc on the forecast hot path (G3).
    forecast_psi: Vec<f32>,
    // ── Phase 2 (low-rank) fields ──
    /// Low-rank factor `A` (`D × r`, row-major, f32). Empty unless
    /// [`Self::fit_low_rank`] has been called. Cast from the f64 ALS output.
    pub a_low_rank: Vec<f32>,
    /// Low-rank factor `B` (`r × d_h`, row-major, f32). Empty unless
    /// [`Self::fit_low_rank`] has been called.
    pub b_low_rank: Vec<f32>,
    /// Rank `r` from the most recent [`Self::fit_low_rank`] call (0 = unfitted).
    low_rank_r: usize,
    /// Set after a successful [`Self::fit_low_rank`].
    low_rank_fitted: bool,
    /// Scratch for [`Self::forecast_low_rank_into`] (`r`-length mid vector).
    /// Allocated when [`Self::fit_low_rank`] is called; reused on every
    /// subsequent low-rank forecast — zero-alloc hot path (G3 extension).
    forecast_low_rank_mid: Vec<f32>,
}

impl<B: KarcBasis<M>, const D: usize, const M: usize, const K: usize> KarcForecaster<B, D, M, K> {
    /// Feature dimension `d_h = K·D·M`.
    pub const D_H: usize = K * D * M;

    /// Construct with a basis and capacity hint for `max_samples` training rows.
    pub fn with_capacity(basis: B, max_samples: usize) -> Self {
        let d_h = Self::D_H;
        Self {
            basis,
            ring: DelayRing::new(),
            wout: Vec::with_capacity(D * d_h),
            fitted: false,
            features_buf: Vec::with_capacity(max_samples * d_h),
            targets_buf: Vec::with_capacity(max_samples * D),
            n_samples: 0,
            scratch: KarcScratch::with_capacity(d_h, D, max_samples),
            delay_buf: vec![0.0; K * D],
            forecast_psi: vec![0.0; d_h],
            a_low_rank: Vec::new(),
            b_low_rank: Vec::new(),
            low_rank_r: 0,
            low_rank_fitted: false,
            forecast_low_rank_mid: Vec::new(),
        }
    }

    /// Number of accumulated training samples (rows in `features_buf`/`targets_buf`).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Whether `fit_ridge` has produced a valid `Wout`.
    #[inline]
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Restore a previously-fit `Wout` without re-running `fit_ridge`.
    ///
    /// This is the **thaw** half of the freeze/thaw bridge: a caller that
    /// serialized a fitted `Wout` (e.g. via `riir-neuron-db::KarcShard`) can
    /// restore the forecaster to a "freshly fitted" state. The semantics are
    /// identical to having just called `fit_ridge` and then forgotten the
    /// trajectory: `is_fitted() == true`, `forecast_into` works immediately,
    /// `forecast_now` works once the delay ring re-fills.
    ///
    /// Clears the trajectory buffer (`n_samples == 0`) and drops any stale
    /// low-rank factors — the restored `Wout` is full-rank, and carrying stale
    /// `(A, B)` would make [`Self::is_low_rank_fitted`] lie about the
    /// forecaster's actual state. The delay ring is NOT touched (it is runtime
    /// observation state, not fit state).
    ///
    /// # Panics
    ///
    /// Panics if `wout.len() != D * Self::D_H` (shape mismatch — caller bug).
    ///
    /// # Use case
    ///
    /// This is the ONLY sanctioned way to set `Wout` outside of `fit_ridge`.
    /// It exists so the freeze/thaw envelope (riir-ai Plan 332 Phase 4) can
    /// avoid re-running the fitter on every shard thaw, which would defeat
    /// the G4 bit-exact restoration gate (the fit path depends on
    /// `n_samples`, which is not carried in the frozen shard).
    pub fn restore_wout(&mut self, wout: Vec<f32>) {
        let d_h = Self::D_H;
        let expected = D * d_h;
        assert_eq!(
            wout.len(),
            expected,
            "KarcForecaster::restore_wout: wout.len() = {} but expected D*d_h = {}*{} = {}",
            wout.len(),
            D,
            d_h,
            expected,
        );
        self.wout = wout;
        self.fitted = true;
        // Drop low-rank state — the restored Wout is full-rank. Carrying
        // stale (A, B) would make is_low_rank_fitted() lie.
        self.a_low_rank = Vec::new();
        self.b_low_rank = Vec::new();
        self.low_rank_r = 0;
        self.low_rank_fitted = false;
        self.forecast_low_rank_mid = Vec::new();
        // Clear the trajectory buffer — the restored forecaster has no
        // training history. A subsequent `fit_ridge` starts from scratch
        // (matches "freshly fitted but no history" semantics).
        self.features_buf.clear();
        self.targets_buf.clear();
        self.n_samples = 0;
    }

    /// Push an observation into the delay ring.
    #[inline]
    pub fn observe(&mut self, obs: &[f32; D]) {
        self.ring.push(obs);
    }

    /// Push an observation and, if the ring is full, accumulate the current
    /// delay state and the **next** observation as a training pair
    /// `(Ψ(x_t), u_{t+1})`. Convenience for streaming fits — the example/test
    /// paths build the buffer directly via [`Self::accumulate_pair`].
    ///
    /// Returns `true` if a pair was accumulated.
    #[inline]
    pub fn observe_and_maybe_pair(&mut self, obs: &[f32; D]) -> bool {
        // Snapshot the current delay state BEFORE pushing obs (so x_t = u_t ⊕ …),
        // then push obs (= u_{t+1} target). If the ring was full before the push,
        // we have a valid (x_t, u_{t+1}) pair.
        let have_state = self.ring.flatten_into(&mut self.delay_buf);
        self.ring.push(obs);
        if !have_state {
            return false;
        }
        // Inline of `accumulate_pair` body. `accumulate_pair(&mut self, ...)`
        // borrows all of `self`, so passing `&self.delay_buf` would alias.
        // It only touches `features_buf` / `basis` / `targets_buf` /
        // `n_samples` — all disjoint from `delay_buf` — so split-borrowing
        // those fields lets us read `delay_buf` in place and drop the
        // per-observation `delay_buf.clone()` alloc.
        let d_h = Self::D_H;
        let old_len = self.features_buf.len();
        self.features_buf.resize(old_len + d_h, 0.0);
        let row = &mut self.features_buf[old_len..old_len + d_h];
        feature_expand::<B, M>(&self.delay_buf, &self.basis, row);
        let t_old = self.targets_buf.len();
        self.targets_buf.resize(t_old + D, 0.0);
        self.targets_buf[t_old..t_old + D].copy_from_slice(obs);
        self.n_samples += 1;
        true
    }

    /// Expand `delay_state` (length `K·D`) into features and store
    /// `(Ψ(delay_state), target)` as a training row.
    #[inline]
    pub fn accumulate_pair(&mut self, delay_state: &[f32], target: &[f32; D]) {
        let d_h = Self::D_H;
        // Append a fresh feature row.
        let old_len = self.features_buf.len();
        self.features_buf.resize(old_len + d_h, 0.0);
        let row = &mut self.features_buf[old_len..old_len + d_h];
        feature_expand::<B, M>(delay_state, &self.basis, row);
        // Append the target row.
        let t_old = self.targets_buf.len();
        self.targets_buf.resize(t_old + D, 0.0);
        self.targets_buf[t_old..t_old + D].copy_from_slice(target);
        self.n_samples += 1;
    }

    /// Clear the accumulated trajectory buffer (does not clear `Wout`).
    pub fn clear_trajectory(&mut self) {
        self.features_buf.clear();
        self.targets_buf.clear();
        self.n_samples = 0;
    }

    /// Solve the ridge regression `Wout = Y·Hᵀ·(H·Hᵀ + λI)⁻¹` over the
    /// accumulated trajectory buffer, picking the cheaper of the direct
    /// feature-space form (when `d_h ≤ N`) or the Woodbury sample-space form
    /// (when `d_h > N`, paper Eq. 40–41).
    ///
    /// `λ > 0` is required (ridge diagonal; `λ = 0` would make the Gram
    /// singular for redundant features).
    ///
    /// On success, sets `Wout` (`D × d_h` row-major) and `is_fitted() == true`.
    pub fn fit_ridge(&mut self, lambda: f32) -> Result<(), FitError> {
        let d_h = Self::D_H;
        if self.n_samples == 0 {
            return Err(FitError::NoSamples);
        }
        if lambda <= 0.0 {
            return Err(FitError::NonPositiveLambda);
        }
        // Accumulate into scratch.
        let n = self.n_samples;
        if d_h <= n {
            self.fit_direct(lambda, d_h, n)?;
        } else {
            self.fit_woodbury(lambda, d_h, n)?;
        }
        self.fitted = true;
        Ok(())
    }

    /// Direct feature-space solve: `Wᵀ = (XᵀX + λI)⁻¹ XᵀY`. Accumulates the
    /// Gram/covariance in **f64** for numerical robustness at small λ (f32
    /// Cholesky fails when λ is below f32 epsilon relative to the matrix scale);
    /// casts Wᵀ to f32 for the forecast matvec.
    fn fit_direct(&mut self, lambda: f32, d_h: usize, n: usize) -> Result<(), FitError> {
        let s = &mut self.scratch;
        s.clear();
        let lambda64 = lambda as f64;
        // Gram = XᵀX (upper triangle), d_h × d_h, f64.
        s.gram.clear();
        s.gram.resize(d_h * d_h, 0.0);
        accumulate_gram_upper_triangle(&mut s.gram, &self.features_buf, d_h, n);
        // Add λI.
        for i in 0..d_h {
            s.gram[i * d_h + i] += lambda64;
        }
        // Cov = XᵀY, d_h × D, f64.
        s.cov.clear();
        s.cov.resize(d_h * D, 0.0);
        for r in 0..n {
            let row = &self.features_buf[r * d_h..(r + 1) * d_h];
            let target = &self.targets_buf[r * D..(r + 1) * D];
            for (i, &ri) in row.iter().enumerate() {
                let ri = ri as f64;
                for (c, &tv) in target.iter().enumerate() {
                    s.cov[i * D + c] += ri * tv as f64;
                }
            }
        }
        // f64 solve → Wᵀ (d_h × D).
        s.chol.clear();
        s.chol.resize(d_h * d_h, 0.0);
        s.z_solve.clear();
        s.z_solve.resize(d_h * D, 0.0);
        s.w_t.clear();
        s.w_t.resize(d_h * D, 0.0);
        {
            let gram = &s.gram[..];
            let cov = &s.cov[..];
            let chol = &mut s.chol[..];
            let z_solve = &mut s.z_solve[..];
            let w_t = &mut s.w_t[..];
            ridge_solve_direct_f64(w_t, chol, z_solve, gram, cov, d_h, D);
        }
        // Cast Wᵀ (d_h × D, f64) → Wout (D × d_h, f32) with transpose.
        self.wout.clear();
        self.wout.resize(D * d_h, 0.0);
        for r in 0..D {
            for c in 0..d_h {
                self.wout[r * d_h + c] = s.w_t[c * D + r] as f32;
            }
        }
        let _ = n;
        Ok(())
    }

    /// Woodbury sample-space solve: `Wᵀ = Xᵀ (X Xᵀ + λI)⁻¹ Y`. Uses the f32
    /// sample-space path (the sample count N is small in this regime, so f32
    /// precision suffices; for very small λ the caller should prefer the direct
    /// path which accumulates in f64).
    fn fit_woodbury(&mut self, lambda: f32, d_h: usize, n: usize) -> Result<(), FitError> {
        let s = &mut self.scratch;
        s.clear();
        // Sample Gram = X Xᵀ, N × N (f32 — sample count is small here).
        // Reuse the pre-allocated scratch buffers (grown on demand, then reused
        // across fits) instead of allocating 4 fresh Vecs per call.
        s.sample_gram.resize(n * n, 0.0);
        let sample_gram_f32 = &mut s.sample_gram[..n * n];
        for i in 0..n {
            let row_i = &self.features_buf[i * d_h..(i + 1) * d_h];
            for j in 0..n {
                let row_j = &self.features_buf[j * d_h..(j + 1) * d_h];
                sample_gram_f32[i * n + j] = simd::simd_dot_f32(row_i, row_j, d_h);
            }
        }
        for i in 0..n {
            sample_gram_f32[i * n + i] += lambda;
        }
        s.sample_chol.resize(n * n, 0.0);
        s.sample_z.resize(n * D, 0.0);
        let w_t_len = d_h * D;
        self.wout.clear();
        self.wout.resize(w_t_len, 0.0);
        {
            let y = &self.targets_buf[..n * D];
            let x = &self.features_buf[..n * d_h];
            let w_t = &mut self.wout[..w_t_len];
            ridge_solve_woodbury_f32(
                w_t,
                &mut s.sample_chol,
                &mut s.sample_z,
                &s.sample_gram,
                y,
                x,
                n,
                d_h,
                D,
            );
        }
        // Transpose Wᵀ (d_h × D) → Wout (D × d_h) via a scratch copy.
        s.w_t_transpose.clear();
        s.w_t_transpose.extend_from_slice(&self.wout[..w_t_len]);
        self.wout.resize(D * d_h, 0.0);
        for r in 0..D {
            for c in 0..d_h {
                self.wout[r * d_h + c] = s.w_t_transpose[c * D + r];
            }
        }
        Ok(())
    }

    /// Forecast `û_{t+1} = Wout · Ψ(delay_state)` into `out` (length `D`).
    ///
    /// Zero allocation on the hot path: the feature buffer
    /// (`self.forecast_psi`, `d_h = K·D·M`) is pre-allocated at construction and
    /// reused via indexing — `GlobalAlloc`-counter tests see zero `alloc`/
    /// `dealloc` delta after warmup (G3). Stack arrays of size `K·D·M` are not
    /// expressible in stable Rust (`generic_const_exprs` is unstable), so the
    /// buffer lives on the heap but is never reallocated.
    ///
    /// Requires [`Self::is_fitted`]; returns `false` (leaving `out` untouched)
    /// if the forecaster has not been fit.
    #[inline]
    pub fn forecast_into(&mut self, delay_state: &[f32], out: &mut [f32]) -> bool {
        if !self.fitted {
            return false;
        }
        let d_h = Self::D_H;
        debug_assert_eq!(delay_state.len(), K * D, "delay_state must be K·D long");
        debug_assert!(out.len() >= D, "out must be at least D long");
        debug_assert_eq!(self.forecast_psi.len(), d_h);
        let psi = &mut self.forecast_psi[..d_h];
        feature_expand::<B, M>(&delay_state[..K * D], &self.basis, psi);
        simd::simd_matvec(&mut out[..D], &self.wout, psi, D, d_h);
        true
    }

    /// Convenience: forecast the forecaster's own current delay ring state.
    /// Returns `false` if the ring is not full or the forecaster is not fit.
    #[inline]
    pub fn forecast_now(&mut self, out: &mut [f32]) -> bool {
        if !self.fitted || !self.ring.is_full() {
            return false;
        }
        if !self.ring.flatten_into(&mut self.delay_buf) {
            return false;
        }
        // Inline of `forecast_into` body. We can't call `self.forecast_into(
        // &self.delay_buf, out)` because it takes `&mut self`, which would alias
        // the shared `&self.delay_buf` borrow. Split-borrowing the disjoint
        // fields (forecast_psi / basis / wout — none of which is delay_buf)
        // lets us read `delay_buf` in place, eliminating the per-tick
        // `delay_buf.clone()` (K·D f32 alloc+copy+drop) this used to do.
        let d_h = Self::D_H;
        let psi = &mut self.forecast_psi[..d_h];
        feature_expand::<B, M>(&self.delay_buf[..K * D], &self.basis, psi);
        simd::simd_matvec(&mut out[..D], &self.wout, psi, D, d_h);
        true
    }

    // ── Phase 2 methods (paper Eqs. 32 / 47) ──

    /// Higher-order feature count for outer-product order `R` applied to this
    /// forecaster's first-order dimension `Self::D_H = K·D·M` (paper Eq. 32).
    ///
    /// Convenience wrapper around [`higher_order_feature_count`] using this
    /// forecaster's `K·D·M` as the base dimension. Used by callers that want
    /// to pre-size buffers for [`feature_expand_higher_order`].
    pub const fn ho_d_h<const R: usize>() -> usize {
        let d_h_1 = Self::D_H;
        match R {
            1 => d_h_1,
            2 => d_h_1 + d_h_1 * (d_h_1 + 1) / 2,
            _ => d_h_1,
        }
    }

    /// Phase 2 (T2.3): fit a low-rank factorization `Wout ≈ A·B` via ALS over
    /// the accumulated first-order feature buffer (paper Eq. 47).
    ///
    /// Builds `G = XᵀX` (f64) and `Cov = XᵀY` (f64) from
    /// `Self::features_buf` / `Self::targets_buf` (mirroring [`Self::fit_ridge`]'s
    /// Gram accumulation), then calls [`low_rank_fit`] with rank `r`, ridge `λ`,
    /// and convergence `max_iters`/`tol`. On success, stores `A` (`D × r`, f32)
    /// in [`Self::a_low_rank`] and `B` (`r × d_h`, f32) in [`Self::b_low_rank`],
    /// and marks [`Self::is_low_rank_fitted`] true.
    ///
    /// For higher-order low-rank fitting, call [`feature_expand_higher_order`] +
    /// [`chunked_gram_into`] + [`low_rank_fit`] directly — this method only
    /// covers the first-order feature buffer.
    ///
    /// `λ > 0` is required (same contract as [`Self::fit_ridge`]). Returns the
    /// number of ALS iterations performed (capped at `max_iters`).
    pub fn fit_low_rank(
        &mut self,
        r: usize,
        lambda: f32,
        max_iters: usize,
        tol: f32,
    ) -> Result<usize, FitError> {
        if self.n_samples == 0 {
            return Err(FitError::NoSamples);
        }
        if lambda <= 0.0 {
            return Err(FitError::NonPositiveLambda);
        }
        let d_h = Self::D_H;
        let n = self.n_samples;
        let lambda64 = lambda as f64;

        // Build un-regularized Gram (XᵀX) and Cov (XᵀY) in f64.
        let s = &mut self.scratch;
        s.clear();
        s.gram.clear();
        s.gram.resize(d_h * d_h, 0.0);
        accumulate_gram_upper_triangle(&mut s.gram, &self.features_buf, d_h, n);
        s.cov.clear();
        s.cov.resize(d_h * D, 0.0);
        for row_idx in 0..n {
            let row = &self.features_buf[row_idx * d_h..(row_idx + 1) * d_h];
            let target = &self.targets_buf[row_idx * D..(row_idx + 1) * D];
            for (i, &ri) in row.iter().enumerate() {
                let ri = ri as f64;
                for (d, &tv) in target.iter().enumerate() {
                    s.cov[i * D + d] += ri * tv as f64;
                }
            }
        }

        // ALS in f64.
        let mut a64 = vec![0.0f64; D * r];
        let mut b64 = vec![0.0f64; r * d_h];
        let mut lr_scratch = LowRankFitScratch::with_capacity(d_h, D, r);
        let iters = low_rank_fit(
            &s.gram,
            &s.cov,
            d_h,
            D,
            r,
            lambda64,
            max_iters,
            tol as f64,
            &mut a64,
            &mut b64,
            &mut lr_scratch,
        );

        // Cast to f32 storage.
        self.a_low_rank.clear();
        self.a_low_rank.extend(a64.iter().map(|&v| v as f32));
        self.b_low_rank.clear();
        self.b_low_rank.extend(b64.iter().map(|&v| v as f32));
        self.low_rank_r = r;
        self.low_rank_fitted = true;
        // Pre-allocate the low-rank forecast mid buffer for zero-alloc hot path.
        self.forecast_low_rank_mid.clear();
        self.forecast_low_rank_mid.resize(r, 0.0);
        Ok(iters)
    }

    /// Cross-game transfer fit: solve `B` with `A` **frozen** at
    /// `a_frozen`, leaving the "personality factor" `A` untouched while
    /// re-fitting the "game-specific readout" `B` from the currently-
    /// accumulated trajectory buffer (Plan 308 extension for Plan 332 Phase 7).
    ///
    /// This is the forecaster-level wrapper around
    /// [`low_rank_fit_b_with_frozen_a`]. It builds the feature Gram `XᵀX` and
    /// cross-covariance `XᵀY` from the accumulated trajectory buffer, then
    /// solves the frozen-A ridge objective in closed form.
    ///
    /// # The cross-game transfer semantics
    ///
    /// - `a_frozen` is `D × r` (row-major), transferred from a Game-A fit
    ///   (e.g. extracted from a frozen KARC shard in the private shard crate
    ///   via the Plan 332 Phase 4 freeze bridge, then SVD/ALS-decomposed).
    /// - After this call, [`Self::is_low_rank_fitted`] is `true`,
    ///   [`Self::a_low_rank`] holds `a_frozen` (cast to f32), and
    ///   [`Self::b_low_rank`] holds the freshly-fit Game-B `B`.
    /// - [`Self::forecast_low_rank_into`] works immediately: `û = A_frozen · B_B · Ψ(x)`.
    ///
    /// # Panics
    ///
    /// Panics if `a_frozen.len() != D * r` (shape mismatch). Returns
    /// [`FitError::NoSamples`] if the trajectory buffer is empty. Returns
    /// [`FitError::NonPositiveLambda`] if `lambda <= 0`.
    ///
    /// # Modelless note
    ///
    /// This is a closed-form ridge solve — no gradient descent, no backprop.
    /// The transfer is deterministic given `(a_frozen, trajectory, lambda)`.
    pub fn fit_low_rank_with_frozen_a(
        &mut self,
        a_frozen: &[f32],
        r: usize,
        lambda: f32,
    ) -> Result<(), FitError> {
        if self.n_samples == 0 {
            return Err(FitError::NoSamples);
        }
        if lambda <= 0.0 {
            return Err(FitError::NonPositiveLambda);
        }
        assert_eq!(
            a_frozen.len(),
            D * r,
            "fit_low_rank_with_frozen_a: a_frozen.len() = {} but expected D*r = {}*{} = {}",
            a_frozen.len(),
            D,
            r,
            D * r,
        );
        let d_h = Self::D_H;
        let n = self.n_samples;
        let lambda64 = lambda as f64;

        // Build un-regularized Gram (XᵀX) and Cov (XᵀY) in f64 — identical
        // to the accumulation in fit_low_rank.
        let s = &mut self.scratch;
        s.clear();
        s.gram.clear();
        s.gram.resize(d_h * d_h, 0.0);
        accumulate_gram_upper_triangle(&mut s.gram, &self.features_buf, d_h, n);
        s.cov.clear();
        s.cov.resize(d_h * D, 0.0);
        for row_idx in 0..n {
            let row = &self.features_buf[row_idx * d_h..(row_idx + 1) * d_h];
            let target = &self.targets_buf[row_idx * D..(row_idx + 1) * D];
            for (i, &ri) in row.iter().enumerate() {
                let ri = ri as f64;
                for (d, &tv) in target.iter().enumerate() {
                    s.cov[i * D + d] += ri * tv as f64;
                }
            }
        }

        // Cast a_frozen to f64 for the solve.
        let a64: Vec<f64> = a_frozen.iter().map(|&v| v as f64).collect();
        let mut b64 = vec![0.0f64; r * d_h];
        let mut lr_scratch = LowRankFitScratch::with_capacity(d_h, D, r);
        low_rank_fit_b_with_frozen_a(
            &s.gram,
            &s.cov,
            d_h,
            D,
            r,
            lambda64,
            &a64,
            &mut b64,
            &mut lr_scratch,
        );

        // Cast to f32 storage. A is frozen (copied verbatim), B is freshly fit.
        self.a_low_rank.clear();
        self.a_low_rank.extend(a_frozen.iter().copied());
        self.b_low_rank.clear();
        self.b_low_rank.extend(b64.iter().map(|&v| v as f32));
        self.low_rank_r = r;
        self.low_rank_fitted = true;
        self.forecast_low_rank_mid.clear();
        self.forecast_low_rank_mid.resize(r, 0.0);
        Ok(())
    }

    /// Cross-game transfer fit (warm-start variant): re-fit BOTH `A` and `B`
    /// from the currently-accumulated Game-B trajectory buffer, starting ALS
    /// from the caller-supplied Game-A factors `(a_init, b_init)`.
    ///
    /// This is the warm-start alternative to [`Self::fit_low_rank_with_frozen_a`]:
    /// instead of holding `A` fixed at Game A's value, both factors are allowed
    /// to re-optimize. The hypothesis is that starting near Game A's solution
    /// helps Game B converge faster to a better optimum than from-scratch ALS,
    /// even when Game A's exact factorization is suboptimal for Game B.
    ///
    /// # When to use this vs [`Self::fit_low_rank_with_frozen_a`]
    ///
    /// - **Frozen-A** ([`Self::fit_low_rank_with_frozen_a`]): tests "does
    ///   Game A's personality subspace *as-is* work for Game B?" Strict —
    ///   documented negative result (0/15 helps, see
    ///   [`.benchmarks/152_karc_cross_game_transfer.md`](../../../.benchmarks/152_karc_cross_game_transfer.md)).
    /// - **Warm-start** (this method): tests "does starting near Game A's
    ///   solution help Game B, allowing both factors to adapt?" Weaker
    ///   constraint — may recover signal frozen-A loses.
    ///
    /// Both fit methods are modelless (closed-form ridge solves in an ALS
    /// loop; no backprop).
    ///
    /// # Arguments
    ///
    /// - `a_init` — Game A's fitted `A` factor (`D × r`, row-major). Used as
    ///   ALS initialization only; both A and B are re-fit.
    /// - `b_init` — Game A's fitted `B` factor (`r × d_h`, row-major).
    /// - `r` — rank. Should match the rank used to produce `a_init`/`b_init`.
    /// - `lambda` — ridge `λ > 0` (typically the same as Game A's).
    /// - `max_iters` / `tol` — ALS convergence control.
    ///
    /// # Returns
    ///
    /// Number of ALS iterations performed (capped at `max_iters`). After this
    /// call, [`Self::is_low_rank_fitted`] is `true`, [`Self::forecast_low_rank_into`]
    /// works immediately.
    ///
    /// # Panics
    ///
    /// Panics if `a_init.len() != D * r` or `b_init.len() != r * d_h` (shape
    /// mismatch). Returns [`FitError::NoSamples`] if the trajectory buffer is
    /// empty. Returns [`FitError::NonPositiveLambda`] if `lambda <= 0`.
    pub fn fit_low_rank_warm_start(
        &mut self,
        a_init: &[f32],
        b_init: &[f32],
        r: usize,
        lambda: f32,
        max_iters: usize,
        tol: f32,
    ) -> Result<usize, FitError> {
        if self.n_samples == 0 {
            return Err(FitError::NoSamples);
        }
        if lambda <= 0.0 {
            return Err(FitError::NonPositiveLambda);
        }
        let d_h = Self::D_H;
        assert_eq!(
            a_init.len(),
            D * r,
            "fit_low_rank_warm_start: a_init.len() = {} but expected D*r = {}*{} = {}",
            a_init.len(),
            D,
            r,
            D * r,
        );
        assert_eq!(
            b_init.len(),
            r * d_h,
            "fit_low_rank_warm_start: b_init.len() = {} but expected r*d_h = {}*{} = {}",
            b_init.len(),
            r,
            d_h,
            r * d_h,
        );
        let n = self.n_samples;
        let lambda64 = lambda as f64;

        // Build un-regularized Gram (XᵀX) and Cov (XᵀY) in f64 — identical
        // to the accumulation in fit_low_rank.
        let s = &mut self.scratch;
        s.clear();
        s.gram.clear();
        s.gram.resize(d_h * d_h, 0.0);
        accumulate_gram_upper_triangle(&mut s.gram, &self.features_buf, d_h, n);
        s.cov.clear();
        s.cov.resize(d_h * D, 0.0);
        for row_idx in 0..n {
            let row = &self.features_buf[row_idx * d_h..(row_idx + 1) * d_h];
            let target = &self.targets_buf[row_idx * D..(row_idx + 1) * D];
            for (i, &ri) in row.iter().enumerate() {
                let ri = ri as f64;
                for (d, &tv) in target.iter().enumerate() {
                    s.cov[i * D + d] += ri * tv as f64;
                }
            }
        }

        // Cast caller-supplied init to f64.
        let a64_init: Vec<f64> = a_init.iter().map(|&v| v as f64).collect();
        let b64_init: Vec<f64> = b_init.iter().map(|&v| v as f64).collect();
        let mut a64 = a64_init.clone();
        let mut b64 = b64_init.clone();
        let mut lr_scratch = LowRankFitScratch::with_capacity(d_h, D, r);
        let iters = low_rank_fit_warm_start(
            &s.gram,
            &s.cov,
            d_h,
            D,
            r,
            lambda64,
            max_iters,
            tol as f64,
            &a64_init,
            &b64_init,
            &mut a64,
            &mut b64,
            &mut lr_scratch,
        );

        // Cast to f32 storage.
        self.a_low_rank.clear();
        self.a_low_rank.extend(a64.iter().map(|&v| v as f32));
        self.b_low_rank.clear();
        self.b_low_rank.extend(b64.iter().map(|&v| v as f32));
        self.low_rank_r = r;
        self.low_rank_fitted = true;
        self.forecast_low_rank_mid.clear();
        self.forecast_low_rank_mid.resize(r, 0.0);
        Ok(iters)
    }

    /// Phase 2 (T2.4): forecast `û = A · (B · Ψ(delay_state))` using the
    /// stored low-rank factors from [`Self::fit_low_rank`].
    ///
    /// Two-stage matvec, same zero-alloc hot-path contract as
    /// [`Self::forecast_into`] (G3 extension): `forecast_psi` and
    /// `forecast_low_rank_mid` are pre-allocated at the first
    /// [`Self::fit_low_rank`] call and reused via indexing.
    ///
    /// Returns `false` (leaving `out` untouched) if
    /// [`Self::fit_low_rank`] has not been called.
    #[inline]
    pub fn forecast_low_rank_into(&mut self, delay_state: &[f32], out: &mut [f32]) -> bool {
        if !self.low_rank_fitted {
            return false;
        }
        let d_h = Self::D_H;
        let r = self.low_rank_r;
        debug_assert_eq!(delay_state.len(), K * D);
        debug_assert!(out.len() >= D);
        debug_assert_eq!(self.forecast_psi.len(), d_h);
        debug_assert_eq!(self.forecast_low_rank_mid.len(), r);
        let psi = &mut self.forecast_psi[..d_h];
        feature_expand::<B, M>(&delay_state[..K * D], &self.basis, psi);
        forecast_low_rank_apply(
            &self.a_low_rank,
            &self.b_low_rank,
            psi,
            &mut self.forecast_low_rank_mid[..r],
            &mut out[..D],
            d_h,
            r,
            D,
        );
        true
    }

    /// Whether [`Self::fit_low_rank`] has produced valid low-rank factors.
    #[inline]
    pub fn is_low_rank_fitted(&self) -> bool {
        self.low_rank_fitted
    }

    /// The rank `r` from the most recent [`Self::fit_low_rank`] (0 = unfitted).
    #[inline]
    pub fn low_rank_r(&self) -> usize {
        self.low_rank_r
    }
}

// ── Errors ────────────────────────────────────────────────────────────────

/// Failure modes for [`KarcForecaster::fit_ridge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FitError {
    /// `fit_ridge` called before any training pairs were accumulated.
    NoSamples = 0,
    /// `λ ≤ 0` would make the ridge solve singular.
    NonPositiveLambda = 1,
}

impl core::fmt::Display for FitError {
    #[cold]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FitError::NoSamples => write!(f, "no training samples accumulated"),
            FitError::NonPositiveLambda => write!(f, "ridge lambda must be > 0"),
        }
    }
}

impl std::error::Error for FitError {}

// ── Inline unit tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
