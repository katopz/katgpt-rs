# Plan 323 — Fourier Continuation Primitive (Non-Periodic Latent Fields)

**Source paper:** "Fourier Neural Operators Explained: A Practical Perspective"
(Duruisseaux / Kossaifi / Anandkumar, Caltech + NVIDIA, arxiv 2511.05963 v2
Jan 2026), §2.3 Fourier Continuation + §6.1 TFNO spectral loss.

**Research note:** `katgpt-rs/.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md`
(candidate plan #1, Gain-tier, 0/4 novelty gate — narrow modelless gap, no
Super-GOAT).

**Status:** OPEN 2026-06-25. Opt-in until GOAT gate passes; promote to
default-on if gain is modelless (it is — closed-form least-squares).

## TL;DR

Add the one missing piece from the FNO paper's modelless toolkit that our
codebase genuinely lacks: **Fourier continuation** — a closed-form periodic
extension of a non-periodic signal so that the standard FFT (which assumes
periodicity) does not produce Gibbs ringing at the boundaries. All other
modelless FNO primitives already ship (`cross_resolution_transport`,
`funcattn`, DEC `exterior_derivative`, `fft_smooth`, `LatCalSpectralFixed`).

This is a narrow Gain-tier primitive: single file, single feature, no new
external deps (uses existing `rustfft`). It is NOT a Super-GOAT (0/4 novelty
gate) and does not change any shipped behavior — only adds a new opt-in
operator.

## Why modelless

Fourier continuation is a **purely deterministic least-squares fit**. The
canonical FC-Legendre recipe:

1. Take `x[0..N]` on a uniform grid.
2. Extend it to length `M > N` by fitting a degree-`p` polynomial to the
   first/last `p+1` samples and evaluating that polynomial on the extension
   points so the extension joins smoothly.
3. The extended signal `x_ext[0..M]` is (approximately) periodic; FFT of
   `x_ext` no longer sees a discontinuity at the wrap, so Gibbs ringing at
   the boundary is suppressed.

No gradient descent, no learned weights. The fit is closed-form. This is
exactly the freeze/thaw-friendly modelless pattern: the recipe is fixed,
the operator is inference-time.

## Where it goes

```
katgpt-rs/crates/katgpt-core/src/spectral/
├── mod.rs            (new — gated on `spectral` umbrella feature? NO —
│                      gate on `fourier_continuation` directly, mirroring
│                      how `flow` is gated on `flow_field_nav`)
└── continuation.rs   (new — the FC primitive)
```

Rationale for layout: the FNO research note also flagged
`spectral_differentiation` (candidate plan #2) as a future narrow gap. Both
belong under a `spectral/` umbrella, but each ships behind its own feature
flag (per AGENTS.md "Feature Flag Discipline" — opt-in, GOAT-gated,
independently promotable). The `mod.rs` only declares `pub mod continuation;`
under `#[cfg(feature = "fourier_continuation")]`.

## API

```rust
// crates/katgpt-core/src/spectral/continuation.rs

/// Errors returned by [`fourier_continue_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourierContinuationError {
    /// `x.len() < 2 * poly_order + 1` — too few samples for the polynomial fit.
    TooFewSamples,
    /// `extension.len() <= x.len()` — extension buffer must be strictly larger.
    ExtensionTooSmall,
    /// `poly_order == 0` — must be at least 1 (linear).
    InvalidPolyOrder,
}

/// Configuration for the Fourier continuation operator.
#[derive(Debug, Clone, Copy)]
pub struct FcConfig {
    /// Degree of the smoothing polynomial fit at each boundary (FC-Legendre
    /// uses a fixed small degree; the FNO paper recommends 2–4). Must be ≥ 1.
    pub poly_order: usize,
    /// Fraction of the input on each side to use as the fit window.
    /// FNO paper uses ~10–25% of N. Clamped to `[0.05, 0.5]`.
    pub fit_fraction: f32,
    /// How many extension samples to generate on each side. The total
    /// extension length is `2 * extension_per_side + N`.
    pub extension_per_side: usize,
}

impl FcConfig {
    /// FC-Legendre-style default: degree-3 polynomial, 20% fit window,
    /// 50% extension on each side. These match the FNO paper §2.3 defaults.
    pub const DEFAULT: Self = Self {
        poly_order: 3,
        fit_fraction: 0.20,
        extension_per_side: 0, // computed from `extension.len()` at call time
    };
}

/// Compute the Fourier-continuation extension of `x` into `extension`.
///
/// `extension` is overwritten with the periodic-friendly signal of length
/// `extension.len() >= x.len()`. The first `x.len()` entries equal `x`
/// (the original samples); the remaining entries hold the polynomial-blend
/// extension that smoothly wraps the boundary.
///
/// Uses pre-allocated `scratch` to avoid hot-path allocation.
///
/// # Errors
/// Returns [`FourierContinuationError`] on shape/order violations.
pub fn fourier_continue_into(
    x: &[f32],
    extension: &mut [f32],
    scratch: &mut FcScratch,
    cfg: &FcConfig,
) -> Result<(), FourierContinuationError> { /* ... */ }

/// Convenience wrapper that allocates — for cold paths / tests only.
pub fn fourier_continue(x: &[f32], cfg: &FcConfig)
    -> Result<Vec<f32>, FourierContinuationError> { /* ... */ }

/// Pre-allocated scratch for hot-path callers.
#[derive(Debug, Clone, Default)]
pub struct FcScratch {
    pub fit_left: Vec<f32>,   // x-coords of the left fit window
    pub fit_right: Vec<f32>,
    pub coef_left: Vec<f32>,  // polynomial coefficients (degree+1)
    pub coef_right: Vec<f32>,
    pub poly_tmp: Vec<f32>,   // Vandermonde row buffer
}

impl FcScratch {
    pub fn ensure_capacity(&mut self, n: usize, poly_order: usize) { /* ... */ }
}
```

The polynomial fit uses **closed-form least squares via the normal
equations** (`AᵀA c = Aᵀb`), solved by Gaussian elimination on the small
`(poly_order+1)²` system. For `poly_order ≤ 4` this is ≤ 25 scalar ops —
well under the FFT cost and branch-free.

## Math (FC-Legendre, abridged)

Given `x[0..N]`, fit a degree-`p` polynomial `P_L(t)` to the left
`w = max(p+1, ⌈fit_fraction·N⌉)` samples and `P_R(t)` to the right `w`
samples. Then construct the extension by blending:

- For `i ∈ [N, N + ext)`: blend the right tail `x[(N-ext)..N]` with the
  mirror-evaluated `P_R(t)` so the wrap joins smoothly.
- Mirror the left side symmetrically.

This is the FNO paper's FC-Legendre recipe (§2.3) reduced to its
least-squares core. We skip the orthonormal-Legendre basis transform (an
optional numerical-stability trick) and rely on the normal equations being
well-conditioned for `poly_order ≤ 4` — Issue 009 (below) tracks adding the
orthonormal-basis stabilization if a benchmark shows it's needed.

## Tasks

- [x] T1 — Create `src/spectral/mod.rs` + `src/spectral/continuation.rs` skeleton with the API above.
- [x] T2 — Implement `fourier_continue_into`: **final algorithm is C¹-matched linear extrapolation + x[0] wrap target** (NOT the polynomial-blend originally proposed — see G1 GOAT iteration history in `.benchmarks/322_*`).
- [x] T3 — Add the `fourier_continuation` feature flag in both `Cargo.toml` files.
- [x] T4 — Register `pub mod spectral;` in `lib.rs` under `#[cfg(feature = "fourier_continuation")]`.
- [x] T5 — Unit tests: 13 tests covering passthrough, sample preservation, interior smoothness, wrap reduction, linear-fit continuity, error paths, high-order finiteness, scratch stability, alloc-vs-into equivalence, repeated-call determinism.
- [x] T6 — Write `benches/bench_323_fourier_continuation_goat.rs` with G1–G4 + informational spectral-derivative diagnostic.
- [x] T7 — Run the GOAT gate; recorded in `.benchmarks/322_fourier_continuation_goat.md`. **All 4 gates PASS.**
- [x] T8 — G1+G2+G3+G4 all PASS → promoted `fourier_continuation` to `default` in both `Cargo.toml`s.
- N/A T9 — No gate failed; no issue opened.
- [x] T10 — Commit on `develop` with `feat:` prefix. *(Committed as `e95853a3 feat: Fourier Continuation primitive (Plan 323, DEFAULT-ON)` on 2026-06-25.)*

## GOAT gate promotion rule

Per `katgpt-rs/AGENTS.md` "Feature Flag Discipline":
1. ✅ Implemented behind `fourier_continuation = []` (opt-in).
2. ✅ Benchmark written (T6).
3. ⏳ GOAT gate: G1 (quality — Gibbs suppression, modelless) + G2 (perf) + G3 (no-regression) + G4 (alloc-free).
4. ⏳ If all gates pass AND the gain is **modelless** → promote to `default`.
5. N/A — gain does not require riir-train (pure closed-form fit).

The G1 Gibbs suppression is **modelless by construction** (no learned
weights), so a PASS gives a clean modelless gain → promote.

## Cross-references

- `katgpt-rs/.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md` — source research note, candidate plan #1.
- `katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md` + `.plans/310_*` — the existing FNO headline primitive (already shipped, DEFAULT-ON).
- `katgpt-rs/.plans/251_dec_operators_cell_complex.md` — DEC `exterior_derivative` (covers spectral differentiation in DEC vocabulary; candidate plan #2 would be a thin wrapper around this for periodic uniform grids).
- `katgpt-rs/crates/katgpt-core/src/flow/fft.rs::fft_smooth_into` — existing FFT pattern with pre-allocated scratch (template for `FcScratch`).
- `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs` — existing frozen-basis primitive pattern (template for the `*_into` + scratch API).
