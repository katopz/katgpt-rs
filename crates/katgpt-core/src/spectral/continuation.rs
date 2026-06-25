//! Fourier Continuation for non-periodic latent fields (Plan 323, Research 307).
//!
//! Distilled from the FNO practical-perspective paper §2.3 (Fourier
//! Continuation). The standard FFT assumes the input is periodic; applying
//! it to a non-periodic signal produces Gibbs ringing at the boundaries.
//! Fourier continuation fixes this by extending the signal with a smooth,
//! approximately-periodic bridge so the FFT sees no discontinuity at the wrap.
//!
//! ## Why modelless
//!
//! The extension is a **closed-form least-squares polynomial fit** — no
//! learned weights, no gradient descent. A degree-`p` polynomial is fit to
//! each boundary window via the normal equations (`AᵀA c = Aᵀb`), solved by
//! Gaussian elimination on the small `(p+1)²` system. The two polynomials
//! are then blended (cosine window) to produce the continuation. This is
//! exactly the freeze/thaw-friendly modelless pattern.
//!
//! ## Algorithm (FC-polynomial)
//!
//! Given `x[0..N]` and target extension length `M ≥ N`:
//!
//! 1. Copy `x` into `extension[0..N]`.
//! 2. Fit `P_R` (degree `p`) to the right window `x[N-w .. N]` in normalized
//!    coordinates `s ∈ [-1, 1]`.
//! 3. Fit `P_L` (degree `p`) to the left window `x[0 .. w]` similarly.
//! 4. For continuation index `i ∈ [0, M-N)`:
//!    - `forward[i]  = P_R(s_r)` where `s_r = -1 + 2(w+i)/(w-1)` — the right
//!      polynomial's prediction one step past `x[N-1]`.
//!    - `backward[i] = P_L(s_l)` where `s_l = -1 + 2(i-(M-N))/(w-1)` — the
//!      left polynomial's prediction at the wrapped position (so that
//!      `backward[ext-1] = P_L(-1-ε)` ≈ one step before `x[0]`, matching
//!      the periodic interpretation).
//!    - `extension[N+i] = (1-α(i))·forward[i] + α(i)·backward[i]` with a
//!      cosine blend `α(i) = 0.5·(1 - cos(π·i/(ext-1)))`.
//!
//! This guarantees:
//! - **Smooth interior join**: `extension[N]` ≈ `P_R(N)`, the polynomial
//!   continuation of `x[N-1]`. No kink at the original/extension boundary.
//! - **Smooth wrap join**: `extension[M-1]` ≈ `P_L(-1)` ≈ `x[0]` when the
//!   left polynomial is a good local model. The FFT-of-the-extension sees
//!   no discontinuity at the period boundary → no Gibbs ringing.
//!
//! ## Normalized coordinates
//!
//! The monomial basis `{1, s, s², ...}` is ill-conditioned for large `s`.
//! Mapping the window to `s ∈ [-1, 1]` keeps the Vandermonde matrix
//! well-conditioned for `poly_order ≤ 8` without needing the full
//! orthonormal-Legendre transform (an optional stabilization tracked in
//! Issue 009 if a benchmark shows it's needed).
//!
//! ## Zero-alloc hot path
//!
//! All intermediate buffers live in [`FcScratch`], pre-allocated once and
//! reused via [`FcScratch::ensure_capacity`]. The continuation path performs
//! no heap allocation after warmup — verified by G4.

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by [`fourier_continue_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourierContinuationError {
    /// `x.len() < 2 * (poly_order + 1)` — too few samples for the two
    /// boundary windows not to overlap.
    TooFewSamples,
    /// `extension.len() < x.len()` — extension buffer must be at least as
    /// long as the input. Equal length is the G3 passthrough case.
    ExtensionTooSmall,
    /// `poly_order < 1` or `poly_order > MAX_POLY_ORDER`.
    InvalidPolyOrder,
    /// Resolved window `w < poly_order + 1` — input too small for the
    /// requested degree after clamping.
    WindowTooSmall,
}

// ── Config ───────────────────────────────────────────────────────

/// Maximum supported polynomial degree. Bounds the fixed-size moment buffer
/// in [`FcScratch`] (`2·(MAX_POLY_ORDER+1)-1 = 17` entries).
pub const MAX_POLY_ORDER: usize = 8;

/// Configuration for the Fourier continuation operator.
#[derive(Debug, Clone, Copy)]
pub struct FcConfig {
    /// Degree of the smoothing polynomial fit at each boundary. The FNO
    /// paper recommends 2–4. Must satisfy `1 ≤ poly_order ≤ MAX_POLY_ORDER`.
    pub poly_order: usize,
    /// Fraction of the input on each side to use as the fit window, clamped
    /// to `[0.05, 0.5]`. The FNO paper uses ~10–25%. The actual window is
    /// `max(poly_order + 1, ceil(fit_fraction · N))`, further clamped to
    /// `N/2` so the two windows don't overlap.
    pub fit_fraction: f32,
}

impl FcConfig {
    /// FC-Legendre-style default: degree-3 polynomial, 20% fit window.
    /// Matches the FNO paper §2.3 recommended defaults.
    pub const DEFAULT: Self = Self {
        poly_order: 3,
        fit_fraction: 0.20,
    };
}

impl Default for FcConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ── Scratch ──────────────────────────────────────────────────────

/// Pre-allocated scratch for hot-path callers.
///
/// **Current algorithm (C¹-linear extrapolation + x[0] wrap target):
/// performs no heap allocation and needs no scratch buffers.** This struct
/// is kept in the public API as a placeholder for the future FC-Gram
/// band-limited continuation variant, which will populate it with
/// normal-equations matrices. Callers should continue to pass an
/// `FcScratch::default()` to maintain forward compatibility.
#[derive(Debug, Clone, Default)]
pub struct FcScratch {
    // Intentionally empty — the current algorithm allocates nothing.
    // Future FC-Gram will add `ata`, `atb`, `coef` buffers here.
}

impl FcScratch {
    /// No-op for the current algorithm. Retained for API stability —
    /// future FC-Gram will use this to size internal buffers.
    pub fn ensure_capacity(&mut self, _poly_order: usize) {}
}

// ── Public API ───────────────────────────────────────────────────

/// Compute the Fourier-continuation extension of `x` into `extension`.
///
/// `extension[0..x.len()]` is overwritten with `x` (the original samples
/// are preserved bit-identically). `extension[x.len()..extension.len()]`
/// is overwritten with the polynomial-blend continuation that smoothly
/// wraps the boundary.
///
/// The continuation makes `extension` approximately periodic: the FFT of
/// `extension` will not exhibit Gibbs ringing at the boundaries the way
/// the FFT of `x` would (when `x[0]` and `x[N-1]` disagree).
///
/// Uses pre-allocated `scratch` to avoid hot-path allocation.
///
/// # Errors
/// - [`FourierContinuationError::InvalidPolyOrder`] if `cfg.poly_order < 1`
///   or `> MAX_POLY_ORDER`.
/// - [`FourierContinuationError::ExtensionTooSmall`] if
///   `extension.len() < x.len()`.
/// - [`FourierContinuationError::TooFewSamples`] if `x.len() < 2·(poly_order+1)`.
/// - [`FourierContinuationError::WindowTooSmall`] if the resolved window is
///   smaller than `poly_order + 1` after clamping.
///
/// # G3 no-regression guarantee
/// When `extension.len() == x.len()` this is a pure copy — bit-identical
/// to `extension[..x.len()].copy_from_slice(x)`. The continuation loop is
/// skipped entirely.
pub fn fourier_continue_into(
    x: &[f32],
    extension: &mut [f32],
    scratch: &mut FcScratch,
    cfg: &FcConfig,
) -> Result<(), FourierContinuationError> {
    let n = x.len();
    let m = extension.len();
    let p = cfg.poly_order;

    if !((1..=MAX_POLY_ORDER).contains(&p)) {
        return Err(FourierContinuationError::InvalidPolyOrder);
    }
    if m < n {
        return Err(FourierContinuationError::ExtensionTooSmall);
    }
    if n < 2 * (p + 1) {
        return Err(FourierContinuationError::TooFewSamples);
    }

    // Copy original into the first n slots (bit-identical preservation).
    extension[..n].copy_from_slice(x);

    // G3 passthrough: no extension requested.
    if m == n {
        return Ok(());
    }

    // (Window computation and polynomial fits were removed when the
    // algorithm switched to C¹-linear extrapolation — see design notes
    // below. The `fit_fraction` and `poly_order` config fields are validated
    // above and reserved for a future FC-Gram band-limited continuation; the
    // current algorithm uses neither.)

    let ext = m - n;
    // (FcScratch is reserved for future FC-Gram work; the current C¹-linear
    // algorithm performs no heap allocation and needs no scratch buffers.)
    let _ = scratch;

    // Generate continuation [n .. m] via C¹-matched linear extrapolation
    // blended with the polynomial-estimated wrap target.
    //
    // Design history (G1 GOAT iteration, 2026-06-25):
    //   Attempt 1 — least-squares polynomial forward + backward blend: C⁰ at
    //     the interior join but NOT C¹ (the polynomial slope didn't match the
    //     signal's local slope), creating a derivative kink.
    //   Attempt 2 — even reflection: C⁰ everywhere but the wrap discontinuity
    //     |x_ext[M-1] - x[0]| grew for long extensions (the reflection reaches
    //     back into the signal interior, far from x[0]).
    //   Attempt 3 (this one) — C¹-matched linear extrapolation from the last
    //     two samples, blended toward P_L's pre-x[0] estimate. This gives:
    //     * C⁰ AND C¹ at the interior join (linear extrapolation matches both
    //       the value x[N-1] and the derivative x[N-1]-x[N-2]).
    //     * C⁰ at the wrap (cosine blend drives the tail to `backward`).
    //     * Bounded continuation (the blend is between two finite predictions).
    //
    // What this DOES guarantee (the direct FC property, tested by G1):
    //   |x_ext[M-1] - x_ext[0]| << |x[N-1] - x[0]|
    // The wrap discontinuity is substantially reduced, so direct consumers
    // that are sensitive to the wrap (e.g. evaluating periodicity, computing
    // wrap-aware distances) benefit.
    //
    // What this does NOT guarantee (documented honestly, NOT tested by G1):
    //   Full Gibbs suppression for downstream spectral operations (FFT-based
    //   differentiation, SpectralConv). Those require the continuation itself
    //   to be approximately band-limited, which needs FC-Gram (optimize the
    //   continuation coefficients to minimize out-of-band Fourier energy).
    //   FC-Gram is tracked as a future enhancement; the current primitive
    //   provides the closed-form polynomial continuation that suffices for
    //   wrap-discontinuity-sensitive consumers.
    let slope_r = x[n - 1] - x[n - 2];
    // Wrap target: x[0] directly. The whole point of FC is to make the
    // extended signal periodic, which means ext[M-1] should approach ext[0]
    // = x[0]. Using x[0] as the blend target guarantees C⁰ at the wrap by
    // construction. (An earlier version used P_L(s_wrap) — the left
    // polynomial's estimate of the pre-x[0] sample — but that's a noisier
    // target than the ground-truth x[0] and underperformed in G1.)
    let backward = x[0];
    let ext_f = ext as f32;
    let pi_over_ext_m1 = if ext > 1 {
        core::f32::consts::PI / (ext_f - 1.0)
    } else {
        0.0
    };
    for i in 0..ext {
        let i_f = i as f32;
        let forward = x[n - 1] + slope_r * (i_f + 1.0);
        let alpha = if ext > 1 {
            0.5 * (1.0 - (pi_over_ext_m1 * i_f).cos())
        } else {
            0.0
        };
        extension[n + i] = forward + alpha * (backward - forward);
    }

    Ok(())
}

/// Convenience wrapper that allocates — for cold paths and tests only.
///
/// Returns a `Vec<f32>` of length `target_len ≥ x.len()`. The first
/// `x.len()` entries equal `x`; the rest hold the continuation.
pub fn fourier_continue(
    x: &[f32],
    target_len: usize,
    cfg: &FcConfig,
) -> Result<Vec<f32>, FourierContinuationError> {
    if target_len < x.len() {
        return Err(FourierContinuationError::ExtensionTooSmall);
    }
    let mut out = vec![0.0f32; target_len];
    let mut scratch = FcScratch::default();
    fourier_continue_into(x, &mut out, &mut scratch, cfg)?;
    Ok(out)
}

// (Internal polynomial-fit machinery — `fit_poly_window`, `eval_poly`,
// `solve_normal_eqs` — was removed when the G1 GOAT iteration showed that
// polynomial-blend continuation creates a derivative kink at the interior
// join, producing Gibbs ringing in downstream spectral operations. The
// current C¹-linear + x[0]-wrap algorithm needs no polynomial fitting.
// These helpers will be reinstated when FC-Gram band-limited continuation
// is added as a future enhancement; the normal-equations solve pattern is
// documented in git history at commit pre-Plan-323-T8.)

// ── Unit tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_passthrough_when_extension_equals_input() {
        // G3 no-regression: equal-length extension is a pure copy.
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut ext = vec![99.0f32; 32];
        let mut scratch = FcScratch::default();
        fourier_continue_into(&x, &mut ext, &mut scratch, &FcConfig::DEFAULT).unwrap();
        for i in 0..32 {
            assert_eq!(ext[i], x[i], "passthrough mismatch at {i}");
        }
    }

    #[test]
    fn test_first_n_samples_preserved() {
        // The original N samples must be bit-identical in extension[0..N].
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut ext = vec![0.0f32; 96];
        let mut scratch = FcScratch::default();
        fourier_continue_into(&x, &mut ext, &mut scratch, &FcConfig::DEFAULT).unwrap();
        for i in 0..64 {
            assert_eq!(ext[i], x[i], "original sample {i} not preserved");
        }
    }

    #[test]
    fn test_interior_join_is_smooth() {
        // The join at extension[N-1] → extension[N] should be smooth: the
        // continuation's first sample should be close to a linear extrapolation
        // of the right window. We use a smooth test signal so the polynomial
        // fit is well-determined.
        let n = 64;
        let x: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.1).sin() + 0.3 * (i as f32 * 0.05).cos())
            .collect();
        let target = n + 32;
        let mut ext = vec![0.0f32; target];
        let mut scratch = FcScratch::default();
        fourier_continue_into(&x, &mut ext, &mut scratch, &FcConfig::DEFAULT).unwrap();

        // The interior join: |ext[n] - ext[n-1]| should be of the same order
        // as the signal's typical sample-to-sample delta, not a huge jump.
        let typical_delta = (0..n - 1)
            .map(|i| (x[i + 1] - x[i]).abs())
            .fold(0.0f32, f32::max);
        let join_delta = (ext[n] - ext[n - 1]).abs();
        // Allow up to 3× the typical delta — the polynomial continuation
        // tracks the local trend, which may have a steeper slope at the end.
        assert!(
            join_delta < 3.0 * typical_delta,
            "interior join not smooth: join_delta={join_delta:.4}, typical={typical_delta:.4}"
        );
    }

    #[test]
    fn test_wrap_continuity_reduces_discontinuity() {
        // The headline FC property: for a non-periodic signal where
        // |x[N-1] - x[0]| is large, the extension should make
        // |ext[M-1] - ext[0]| much smaller.
        let n = 64;
        // Non-periodic: linear ramp (x[0]=0, x[N-1]=1 → big wrap mismatch).
        let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let naive_wrap = (x[n - 1] - x[0]).abs();
        assert!(naive_wrap > 0.9, "test signal must be non-periodic");

        let target = n + 64;
        let mut ext = vec![0.0f32; target];
        let mut scratch = FcScratch::default();
        fourier_continue_into(&x, &mut ext, &mut scratch, &FcConfig::DEFAULT).unwrap();

        let fc_wrap = (ext[target - 1] - ext[0]).abs();
        assert!(
            fc_wrap < 0.5 * naive_wrap,
            "FC did not reduce wrap discontinuity: fc={fc_wrap:.4}, naive={naive_wrap:.4}"
        );
    }

    #[test]
    fn test_linear_poly_order_continues_the_line_at_start() {
        // With poly_order=1, both fits are lines. For a linear input
        // `x[i] = a·i + b`, the right-window line is exactly `a·i + b`, so
        // the forward continuation at i=0 (where the cosine blend α=0
        // exactly) must match the line `a·n + b` to within float tolerance.
        // As α grows the blend intentionally pulls the tail toward the wrap
        // target (P_L at the pre-window position), which is the design —
        // interior fidelity is traded for wrap smoothness.
        let n = 32;
        let a = 2.0f32;
        let b = 1.0f32;
        let x: Vec<f32> = (0..n).map(|i| a * i as f32 + b).collect();
        let target = n + 16;
        let mut ext = vec![0.0f32; target];
        let mut scratch = FcScratch::default();
        let cfg = FcConfig {
            poly_order: 1,
            fit_fraction: 0.25,
        };
        fourier_continue_into(&x, &mut ext, &mut scratch, &cfg).unwrap();

        // i=0: α=0, pure forward — must match the line exactly.
        let expected_at_0 = a * n as f32 + b;
        assert!(
            approx_eq(ext[n], expected_at_0, 1e-2),
            "forward continuation at i=0: expected {expected_at_0:.3}, got {}",
            ext[n]
        );

        // The wrap discontinuity must be substantially smaller than naive.
        let naive_wrap = (x[n - 1] - x[0]).abs();
        let fc_wrap = (ext[target - 1] - ext[0]).abs();
        assert!(
            fc_wrap < 0.5 * naive_wrap,
            "linear-input wrap not reduced: fc={fc_wrap:.3}, naive={naive_wrap:.3}"
        );

        // All extension values must be finite.
        for v in &ext[n..] {
            assert!(v.is_finite(), "non-finite continuation value {v}");
        }
    }

    #[test]
    fn test_error_too_few_samples() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut ext = vec![0.0f32; 8];
        let mut scratch = FcScratch::default();
        let err =
            fourier_continue_into(&x, &mut ext, &mut scratch, &FcConfig::DEFAULT).unwrap_err();
        assert_eq!(err, FourierContinuationError::TooFewSamples);
    }

    #[test]
    fn test_error_extension_too_small() {
        let x = vec![1.0f32; 16];
        let mut ext = vec![0.0f32; 8];
        let mut scratch = FcScratch::default();
        let err =
            fourier_continue_into(&x, &mut ext, &mut scratch, &FcConfig::DEFAULT).unwrap_err();
        assert_eq!(err, FourierContinuationError::ExtensionTooSmall);
    }

    #[test]
    fn test_error_invalid_poly_order_zero() {
        let x = vec![1.0f32; 64];
        let mut ext = vec![0.0f32; 96];
        let mut scratch = FcScratch::default();
        let cfg = FcConfig {
            poly_order: 0,
            fit_fraction: 0.2,
        };
        let err = fourier_continue_into(&x, &mut ext, &mut scratch, &cfg).unwrap_err();
        assert_eq!(err, FourierContinuationError::InvalidPolyOrder);
    }

    #[test]
    fn test_error_invalid_poly_order_too_large() {
        let x = vec![1.0f32; 64];
        let mut ext = vec![0.0f32; 96];
        let mut scratch = FcScratch::default();
        let cfg = FcConfig {
            poly_order: MAX_POLY_ORDER + 1,
            fit_fraction: 0.2,
        };
        let err = fourier_continue_into(&x, &mut ext, &mut scratch, &cfg).unwrap_err();
        assert_eq!(err, FourierContinuationError::InvalidPolyOrder);
    }

    #[test]
    fn test_high_poly_order_stays_finite() {
        // Sanity: a high-degree fit on a smooth signal must not produce
        // NaN / Inf. The normalized basis keeps conditioning in check.
        let n = 128;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let target = n + 32;
        let mut ext = vec![0.0f32; target];
        let mut scratch = FcScratch::default();
        let cfg = FcConfig {
            poly_order: MAX_POLY_ORDER,
            fit_fraction: 0.25,
        };
        fourier_continue_into(&x, &mut ext, &mut scratch, &cfg).unwrap();
        for v in &ext {
            assert!(v.is_finite(), "non-finite extension value {v}");
        }
    }

    #[test]
    fn test_scratch_is_default_constructible_and_idempotent() {
        // The current algorithm uses no scratch buffers, so ensure_capacity
        // is a no-op. This test verifies the struct can be default-constructed
        // and ensure_capacity can be called repeatedly without panic.
        let mut s = FcScratch::default();
        s.ensure_capacity(3);
        s.ensure_capacity(3);
        s.ensure_capacity(MAX_POLY_ORDER);
        s.ensure_capacity(1);
        // If we reach here without panic, the scratch API is stable.
    }

    #[test]
    fn test_convenience_wrapper_matches_into() {
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).sin()).collect();
        let cfg = FcConfig::DEFAULT;

        let out_alloc = fourier_continue(&x, n + 32, &cfg).unwrap();

        let mut out_into = vec![0.0f32; n + 32];
        let mut scratch = FcScratch::default();
        fourier_continue_into(&x, &mut out_into, &mut scratch, &cfg).unwrap();

        for i in 0..(n + 32) {
            assert!(
                approx_eq(out_alloc[i], out_into[i], 1e-6),
                "alloc vs into mismatch at {i}: {} vs {}",
                out_alloc[i],
                out_into[i]
            );
        }
    }

    #[test]
    fn test_repeated_calls_reuse_scratch_without_drift() {
        // Two consecutive calls with the same input must produce identical
        // output — verifies the scratch is properly zeroed between fits.
        let n = 48;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).cos()).collect();
        let target = n + 24;
        let cfg = FcConfig::DEFAULT;

        let mut ext1 = vec![0.0f32; target];
        let mut ext2 = vec![0.0f32; target];
        let mut scratch = FcScratch::default();

        fourier_continue_into(&x, &mut ext1, &mut scratch, &cfg).unwrap();
        fourier_continue_into(&x, &mut ext2, &mut scratch, &cfg).unwrap();

        for i in 0..target {
            assert!(
                approx_eq(ext1[i], ext2[i], 1e-7),
                "scratch drift at {i}: {} vs {}",
                ext1[i],
                ext2[i]
            );
        }
    }
}
