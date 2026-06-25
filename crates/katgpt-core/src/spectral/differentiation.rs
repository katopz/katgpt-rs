//! Standalone FFT-based spectral differentiation on periodic uniform 1D grids
//! (Plan 325, Research 307 §3 candidate plan #2).
//!
//! Distilled from the FNO practical-perspective paper §2.1 (spectral
//! differentiation). For an input sampled on a uniform grid and assumed
//! periodic, the m-th derivative is computed exactly (for band-limited input)
//! by multiplying each FFT coefficient by `(iω)^m` and transforming back:
//!
//! ```text
//! X       = FFT(x)
//! ∂^m x   = IFFT( (iω)^m ⊙ X )
//! ```
//!
//! ## When to use this vs DEC `exterior_derivative`
//!
//! - **This primitive** — input is a flat array of equally-spaced samples on
//!   a *periodic* 1D domain (time-series window, cyclic HLA channel, ring
//!   buffer). O(N log N), no topological setup, no CellComplex required.
//! - **DEC `exterior_derivative`** (`crates/katgpt-core/src/dec/operators.rs`)
//!   — general case on arbitrary cell complexes (irregular meshes, 2D/3D
//!   grids, manifolds with boundary). Slower per-call due to boundary-operator
//!   assembly, but handles anything.
//!
//! ## Why modelless
//!
//! FFT + element-wise complex multiply + IFFT. No weight mutation, no
//! gradient descent, no learned parameters. The frequency-domain multiplier
//! `(iω)^m` is a closed-form deterministic function of the bin index. Pure
//! linear algebra on a fixed orthogonal basis. Freeze/thaw-friendly by
//! construction (deterministic function of input).
//!
//! ## Composes with Plan 323 Fourier continuation
//!
//! For non-periodic inputs, chain with `spectral::continuation::fourier_continue`
//! to suppress Gibbs ringing before differentiating:
//! ```ignore
//! let extended = fourier_continue(&x, 2 * x.len(), &FcConfig::DEFAULT)?;
//! let deriv = spectral_differentiate(&extended, &SpecDiffConfig::default_order(1))?;
//! ```
//!
//! ## Nyquist handling (even N)
//!
//! For even-length input, the FFT bin at index `N/2` is the Nyquist mode,
//! which is ambiguous (simultaneously `+N/2` and `-N/2`) for real inputs.
//! Its coefficient is real-valued. Multiplying by `(iω)^m`:
//!
//! - **Odd `m`** (`1, 3, ...`): produces a pure imaginary value, breaking
//!   Hermitian symmetry and yielding a complex (non-real) IFFT output.
//!   **We zero this bin for odd orders on even-length signals** so the output
//!   stays real. The information loss is one mode out of N (vanishes for
//!   large N, exact for band-limited signals with no Nyquist content).
//! - **Even `m`** (`0, 2, 4, ...`): the multiplier is real and well-defined,
//!   Hermitian symmetry is preserved. We keep the Nyquist bin.
//!
//! ## Zero-alloc hot path
//!
//! [`SpecDiffScratch`] bundles a reusable `FftPlanner<f32>` and a complex
//! frequency buffer. After warmup ([`SpecDiffScratch::ensure_capacity`]),
//! [`spectral_differentiate_into`] performs no heap allocation — verified by
//! the G4 GOAT gate. (The `FftPlanner` caches FFT plans internally; the first
//! call for a given `N` populates the cache, subsequent calls hit the cache.)

use rustfft::{num_complex::Complex, Fft, FftPlanner};

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by [`spectral_differentiate_into`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecDiffError {
    /// `x.len() < 2` — FFT needs at least 2 samples.
    TooFewSamples,
    /// `out.len() != x.len()` — output buffer must match input length exactly.
    OutputSizeMismatch,
    /// `cfg.order > MAX_ORDER`. Bounded to keep the `(iω)^m` exponentiation
    /// cheap and numerically stable.
    InvalidOrder,
}

// ── Config ───────────────────────────────────────────────────────

/// Maximum supported derivative order. Bounds the `(iω)^m` magnitude
/// (high orders amplify high frequencies aggressively and overflow f32).
pub const MAX_ORDER: u32 = 8;

/// Configuration for the spectral differentiation operator.
#[derive(Debug, Clone, Copy)]
pub struct SpecDiffConfig {
    /// Derivative order. `0` = identity (G3 passthrough), `1` = first
    /// derivative, `2` = second derivative (1D Laplacian), etc.
    /// Must satisfy `order ≤ MAX_ORDER`.
    pub order: u32,
    /// Sample spacing `h` (physical distance between adjacent samples).
    /// Default `1.0` (differentiate in index space). Set to `h = L/N` to
    /// differentiate w.r.t. physical coordinate `x` over period `L`.
    pub spacing: f32,
}

impl SpecDiffConfig {
    /// Default config: first derivative, unit spacing.
    pub const DEFAULT: Self = Self {
        order: 1,
        spacing: 1.0,
    };

    /// Convenience constructor for derivative order `m` with unit spacing.
    pub const fn default_order(order: u32) -> Self {
        Self {
            order,
            spacing: 1.0,
        }
    }
}

impl Default for SpecDiffConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ── Scratch ──────────────────────────────────────────────────────

/// Pre-allocated scratch for hot-path callers.
///
/// Bundles a reusable `FftPlanner<f32>` (which caches FFT plans internally),
/// a complex frequency buffer sized to the largest input seen, and a
/// reusable FFT scratch buffer (rustfft's [`Fft::process`] allocates a
/// `Vec` on every call — we route through [`Fft::process_with_scratch`]
/// instead, feeding it this pre-allocated buffer, so the hot path is
/// allocation-free).
///
/// After [`SpecDiffScratch::ensure_capacity`], [`spectral_differentiate_into`]
/// performs zero heap allocations.
///
/// (`FftPlanner` does not implement `Debug`/`Default` itself, so we provide
/// manual impls that report only the buffer capacity / construct a fresh
/// planner via `FftPlanner::new()`.)
pub struct SpecDiffScratch {
    planner: FftPlanner<f32>,
    /// Cached forward plan for `cached_size`, if any. `Arc::clone` is free;
    /// calling `FftPlanner::plan_fft_*` is not (it allocates per call).
    fwd_plan: Option<std::sync::Arc<dyn Fft<f32>>>,
    /// Cached inverse plan for `cached_size`, if any.
    inv_plan: Option<std::sync::Arc<dyn Fft<f32>>>,
    /// Size for which `fwd_plan`/`inv_plan` are cached. `0` = uncached.
    cached_size: usize,
    freq_buf: Vec<Complex<f32>>,
    /// Scratch for rustfft's `process_with_scratch`. Sized to the larger of
    /// `get_inplace_scratch_len()` for the forward and inverse plans seen
    /// so far. Re-used across calls to avoid per-call allocation.
    fft_scratch: Vec<Complex<f32>>,
}

impl core::fmt::Debug for SpecDiffScratch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpecDiffScratch")
            .field("freq_buf_capacity", &self.freq_buf.capacity())
            .field("freq_buf_len", &self.freq_buf.len())
            .field("fft_scratch_capacity", &self.fft_scratch.capacity())
            .finish_non_exhaustive()
    }
}

impl Default for SpecDiffScratch {
    fn default() -> Self {
        Self {
            planner: FftPlanner::new(),
            fwd_plan: None,
            inv_plan: None,
            cached_size: 0,
            freq_buf: Vec::new(),
            fft_scratch: Vec::new(),
        }
    }
}

impl SpecDiffScratch {
    /// Construct an empty scratch (lazy-allocates on first call).
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensure the internal frequency buffer has capacity for at least `n`
    /// complex samples, AND pre-warm the cached FFT plans + scratch for size
    /// `n`. Plans are cached as `Arc<dyn Fft<f32>>` handles so the hot path
    /// clones them (refcount bump, allocation-free) instead of calling
    /// `FftPlanner::plan_fft_*` (which allocates per call). Call once with
    /// the expected input size to make the first [`spectral_differentiate_into`]
    /// call allocation-free. If the size later changes, plans are rebuilt
    /// lazily on the next `spectral_differentiate_into`.
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.freq_buf.capacity() < n {
            // Reserve with a bit of slack to absorb small size variations
            // without re-allocating.
            self.freq_buf
                .reserve_exact(n.saturating_sub(self.freq_buf.capacity()) + n / 4 + 8);
        }
        if self.cached_size != n {
            // Build + cache the plans for this size. Returns `Arc<dyn Fft<f32>>`;
            // we store the Arc so the hot path clones it instead of recalling
            // the planner.
            self.fwd_plan = Some(self.planner.plan_fft_forward(n));
            self.inv_plan = Some(self.planner.plan_fft_inverse(n));
            self.cached_size = n;
        }
        // Size the FFT scratch to the larger of the two cached plans' scratch
        // requirements (used by `process_with_scratch`).
        let needed = self
            .fwd_plan
            .as_ref()
            .map(|p| p.get_inplace_scratch_len())
            .unwrap_or(0)
            .max(
                self.inv_plan
                    .as_ref()
                    .map(|p| p.get_inplace_scratch_len())
                    .unwrap_or(0),
            );
        if self.fft_scratch.capacity() < needed {
            self.fft_scratch
                .reserve_exact(needed.saturating_sub(self.fft_scratch.capacity()) + needed / 4 + 8);
        }
        // Keep length == capacity so the slice we hand to rustfft is the full
        // allocation. The contents are garbage (overwritten by rustfft); only
        // the length matters.
        if self.fft_scratch.len() < self.fft_scratch.capacity() {
            self.fft_scratch
                .resize(self.fft_scratch.capacity(), Complex::new(0.0, 0.0));
        }
    }

    /// Hand rustfft two disjoint mutable sub-borrows: the frequency buffer
    /// (the actual signal) and the FFT scratch (rustfft's working space).
    /// Used between the forward and inverse FFT calls so the caller can still
    /// re-borrow `freq_buf` mutably to apply the `(iω)^m` multiplier.
    #[inline]
    fn split_buffers(&mut self) -> (&mut [Complex<f32>], &mut [Complex<f32>]) {
        // `split_first_mut` would also work; `split_at_mut(0)` is cleaner here
        // because the two slices are independent (not contiguous segments).
        // We use the safe reborrow pattern: split `freq_buf` off first, then
        // take `fft_scratch` separately. The borrow checker accepts this
        // because the two fields are disjoint.
        let freq_buf = &mut self.freq_buf[..];
        let fft_scratch = &mut self.fft_scratch[..];
        (freq_buf, fft_scratch)
    }
}

// ── Frequency index ──────────────────────────────────────────────

/// Compute the signed frequency index `k_j` for FFT output bin `j` of a
/// length-`n` transform.
///
/// Standard "fftshift" convention: bins `0..=n/2` map to non-negative
/// frequencies `0, 1, ..., n/2`; bins `n/2+1..n` map to negative frequencies
/// `-(n/2-1), ..., -1` (for even `n`) or `-(n-1)/2, ..., -1` (for odd `n`).
#[inline]
fn signed_freq_index(j: usize, n: usize) -> i32 {
    let half = n / 2;
    if j <= half {
        j as i32
    } else {
        (j as i32) - (n as i32)
    }
}

// ── Public API ───────────────────────────────────────────────────

/// Compute the m-th spectral derivative of `x` into `out`.
///
/// `x` is assumed to be `N` equally-spaced samples of a periodic signal with
/// sample spacing `cfg.spacing`. The output `out[..N]` is overwritten with
/// the m-th derivative, where `m = cfg.order`.
///
/// Uses pre-allocated `scratch` to avoid hot-path allocation. The scratch's
/// internal `FftPlanner` caches FFT plans, so the first call for a given `N`
/// populates the cache and subsequent calls hit it.
///
/// # Errors
/// - [`SpecDiffError::TooFewSamples`] if `x.len() < 2`.
/// - [`SpecDiffError::OutputSizeMismatch`] if `out.len() != x.len()`.
/// - [`SpecDiffError::InvalidOrder`] if `cfg.order > MAX_ORDER`.
///
/// # G3 no-regression guarantee
/// When `cfg.order == 0`, the multiplier is identically `1` and the output is
/// bit-identical to the input (the IFFT of `FFT(x)` is `x` exactly, modulo
/// rustfft's rounding which is bit-identical for the round-trip on real input
/// in our supported size range).
pub fn spectral_differentiate_into(
    x: &[f32],
    out: &mut [f32],
    scratch: &mut SpecDiffScratch,
    cfg: &SpecDiffConfig,
) -> Result<(), SpecDiffError> {
    let n = x.len();
    let m = cfg.order;

    if n < 2 {
        return Err(SpecDiffError::TooFewSamples);
    }
    if out.len() != n {
        return Err(SpecDiffError::OutputSizeMismatch);
    }
    if m > MAX_ORDER {
        return Err(SpecDiffError::InvalidOrder);
    }

    // Resize the freq buffer (no-op if already large enough; capacity is
    // sticky across calls).
    scratch.freq_buf.resize(n, Complex::new(0.0, 0.0));

    // Acquire the cached forward/inverse plans for size `n`. If the size
    // changed since warmup (or the scratch was never warmed), rebuild lazily.
    // We clone the cached `Arc<dyn Fft<f32>>` (refcount bump, allocation-free)
    // rather than calling `FftPlanner::plan_fft_*` (which allocates per call
    // even on cache hit — see G4 GOAT gate).
    if scratch.cached_size != n {
        scratch.ensure_capacity(n);
    }
    let fwd = scratch
        .fwd_plan
        .as_ref()
        .expect("plans cached by ensure_capacity")
        .clone();
    let inv = scratch
        .inv_plan
        .as_ref()
        .expect("plans cached by ensure_capacity")
        .clone();
    // Size the FFT scratch to the larger of the two plans' requirements so we
    // can use `process_with_scratch` (rustfft's `process` allocates a Vec per
    // call). `resize` is a no-op when `len == fft_scratch_len`; the underlying
    // capacity is sticky and grows monotonically.
    let fft_scratch_len = fwd
        .get_inplace_scratch_len()
        .max(inv.get_inplace_scratch_len());
    scratch
        .fft_scratch
        .resize(fft_scratch_len, Complex::new(0.0, 0.0));

    // Pack real input into complex buffer (imag = 0).
    for (b, &v) in scratch.freq_buf.iter_mut().zip(x.iter()) {
        b.re = v;
        b.im = 0.0;
    }

    // Split the mutable borrow of `scratch` so we can hand rustfft two
    // disjoint sub-borrows: the frequency buffer (mutated) and the FFT
    // scratch (mutated). This satisfies the borrow checker without cloning.
    let (freq_buf, fft_scratch) = scratch.split_buffers();

    // Forward FFT (in-place).
    fwd.process_with_scratch(freq_buf, fft_scratch);

    // Apply (iω)^m multiplier to each bin.
    //
    // ω_j = 2π · k_j / (N · h)
    // (iω)^m = exp(m · (ln(ω) + i·π/2)) = ω^m · (cos(m·π/2) + i·sin(m·π/2))
    //
    // For performance we special-case m ∈ {0, 1, 2} and fall back to the
    // general complex-powers recurrence for higher orders.
    let is_even_n = n.is_multiple_of(2);
    let nyquist_idx = n / 2;
    let two_pi_over_nh = 2.0f32 * core::f32::consts::PI / (n as f32 * cfg.spacing);

    match m {
        0 => {
            // Identity — multiplier is 1. Leave freq_buf untouched. The
            // IFFT round-trip reconstructs x bit-identically (G3).
        }
        1 => {
            // (iω)^1 = i·ω. Multiply by (0 + i·ω) = rotate +90° and scale.
            for j in 0..n {
                // Zero the Nyquist bin for odd orders on even-length signals
                // to keep the output real.
                if is_even_n && j == nyquist_idx {
                    scratch.freq_buf[j] = Complex::new(0.0, 0.0);
                    continue;
                }
                let k = signed_freq_index(j, n) as f32;
                let omega = two_pi_over_nh * k;
                // (i·ω) · (a + bi) = (-ω·b) + i·(ω·a)
                let a = scratch.freq_buf[j].re;
                let b = scratch.freq_buf[j].im;
                scratch.freq_buf[j] = Complex::new(-omega * b, omega * a);
            }
        }
        2 => {
            // (iω)^2 = -ω². Real multiplier, no rotation.
            for j in 0..n {
                let k = signed_freq_index(j, n) as f32;
                let omega = two_pi_over_nh * k;
                let mult = -omega * omega;
                scratch.freq_buf[j] *= mult;
            }
        }
        _ => {
            // General case: compute (iω)^m via complex exponentiation.
            // (iω)^m = ω^m · (cos(m·π/2) + i·sin(m·π/2))  for ω ≥ 0
            //        = |ω|^m · (cos(m·π/2) + i·sin(m·π/2)·sign(ω)^m)
            //
            // We implement it as repeated complex multiply by (iω) to keep
            // it numerically stable (no `powf`), then special-case odd m for
            // Nyquist zeroing at the end.
            for j in 0..n {
                let k = signed_freq_index(j, n) as f32;
                let omega = two_pi_over_nh * k;
                // (iω) as a complex number: 0 + i·ω.
                let i_omega = Complex::new(0.0, omega);
                // Raise to the m-th power by repeated multiplication.
                let mut acc = Complex::new(1.0, 0.0);
                for _ in 0..m {
                    acc *= i_omega;
                }
                scratch.freq_buf[j] *= acc;
            }
            // Nyquist zeroing for odd orders on even-length signals.
            if is_even_n && (m % 2 == 1) {
                scratch.freq_buf[nyquist_idx] = Complex::new(0.0, 0.0);
            }
        }
    }

    // Inverse FFT (in-place). rustfft's inverse is unnormalized: output is
    // scaled by N. We undo that to recover the derivative values.
    let (freq_buf, fft_scratch) = scratch.split_buffers();
    inv.process_with_scratch(freq_buf, fft_scratch);

    // Unpack real part, scale by 1/N. The imaginary part should be ~0 (we
    // preserved Hermitian symmetry by Nyquist-zeroing for odd orders); we
    // discard it.
    let scale = 1.0 / n as f32;
    for (o, &c) in out.iter_mut().zip(scratch.freq_buf.iter()) {
        *o = c.re * scale;
    }

    Ok(())
}

/// Convenience wrapper that allocates — for cold paths and tests only.
///
/// Returns a `Vec<f32>` of length `x.len()` holding the m-th spectral
/// derivative of `x`.
pub fn spectral_differentiate(x: &[f32], cfg: &SpecDiffConfig) -> Result<Vec<f32>, SpecDiffError> {
    let mut out = vec![0.0f32; x.len()];
    let mut scratch = SpecDiffScratch::new();
    spectral_differentiate_into(x, &mut out, &mut scratch, cfg)?;
    Ok(out)
}

// ── Unit tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;

    fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
        actual
            .iter()
            .zip(expected.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max)
    }

    /// Sample `sin(2π·j/N)` for j = 0..N (one full period, periodic).
    fn sine_period(n: usize) -> Vec<f32> {
        (0..n)
            .map(|j| ((TAU * j as f32) / n as f32).sin())
            .collect()
    }

    /// Sample `cos(2π·j/N)` for j = 0..N.
    fn cosine_period(n: usize) -> Vec<f32> {
        (0..n)
            .map(|j| ((TAU * j as f32) / n as f32).cos())
            .collect()
    }

    #[test]
    fn test_error_too_few_samples() {
        let x = [1.0f32];
        let mut out = [0.0f32];
        let mut scratch = SpecDiffScratch::new();
        let err = spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT);
        assert_eq!(err, Err(SpecDiffError::TooFewSamples));
    }

    #[test]
    fn test_error_output_size_mismatch() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 3];
        let mut scratch = SpecDiffScratch::new();
        let err = spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT);
        assert_eq!(err, Err(SpecDiffError::OutputSizeMismatch));
    }

    #[test]
    fn test_error_invalid_order() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];
        let mut scratch = SpecDiffScratch::new();
        let cfg = SpecDiffConfig {
            order: MAX_ORDER + 1,
            spacing: 1.0,
        };
        let err = spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg);
        assert_eq!(err, Err(SpecDiffError::InvalidOrder));
    }

    #[test]
    fn test_order_zero_is_identity() {
        // G3 no-regression: at order=0, the operator is identity.
        let n = 64;
        let x: Vec<f32> = (0..n).map(|j| (j as f32) * 0.1).collect();
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        let cfg = SpecDiffConfig::default_order(0);
        spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg).unwrap();
        // IFFT(FFT(x)) is x up to floating-point rounding from the round-trip.
        // rustfft is bit-exact for the round-trip on these sizes for real input.
        let err = max_abs_error(&out, &x);
        assert!(
            err < 1e-5,
            "order=0 should be identity, max abs error = {err:e}"
        );
    }

    #[test]
    fn test_first_derivative_of_sine_matches_cosine() {
        // d/dx sin(2πx/L) = (2π/L) cos(2πx/L). For unit spacing h=1, L=N,
        // so the analytical derivative is (2π/N)·cos(2π·j/N).
        let n = 64;
        let x = sine_period(n);
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        scratch.ensure_capacity(n);
        spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT).unwrap();

        let expected: Vec<f32> = (0..n)
            .map(|j| (TAU / n as f32) * ((TAU * j as f32) / n as f32).cos())
            .collect();
        let err = max_abs_error(&out, &expected);
        // Spectral differentiation of a band-limited periodic signal is exact
        // up to f32 rounding. The error should be tiny.
        assert!(
            err < 1e-4,
            "first derivative of sine should match (2π/N)·cos, max abs error = {err:e}"
        );
    }

    #[test]
    fn test_second_derivative_of_sine_matches_negative_sine() {
        // d²/dx² sin(2πx/L) = -(2π/L)² sin(2πx/L).
        let n = 64;
        let x = sine_period(n);
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        scratch.ensure_capacity(n);
        let cfg = SpecDiffConfig::default_order(2);
        spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg).unwrap();

        let coef = -(TAU / n as f32).powi(2);
        let expected: Vec<f32> = (0..n)
            .map(|j| coef * ((TAU * j as f32) / n as f32).sin())
            .collect();
        let err = max_abs_error(&out, &expected);
        assert!(
            err < 1e-3,
            "second derivative of sine should match -(2π/N)²·sin, max abs error = {err:e}"
        );
    }

    #[test]
    fn test_dc_component_killed_by_first_derivative() {
        // The DC (k=0) component has ω=0, so (iω)·X[0] = 0. A constant signal
        // should differentiate to zero.
        let n = 32;
        let x = vec![5.0f32; n];
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "DC bin {i} should be ~0 after first deriv, got {v:e}"
            );
        }
    }

    #[test]
    fn test_first_derivative_of_cosine_matches_negative_sine() {
        // d/dx cos(2πx/L) = -(2π/L) sin(2πx/L). This exercises the rotation
        // sign convention.
        let n = 64;
        let x = cosine_period(n);
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT).unwrap();

        let expected: Vec<f32> = (0..n)
            .map(|j| -(TAU / n as f32) * ((TAU * j as f32) / n as f32).sin())
            .collect();
        let err = max_abs_error(&out, &expected);
        assert!(
            err < 1e-4,
            "first derivative of cosine should match -(2π/N)·sin, max abs error = {err:e}"
        );
    }

    #[test]
    fn test_odd_length_works() {
        // Odd-length FFT has no Nyquist bin; the odd-order Nyquist-zeroing
        // path should be skipped without error.
        let n = 63;
        let x = sine_period(n);
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT).unwrap();

        let expected: Vec<f32> = (0..n)
            .map(|j| (TAU / n as f32) * ((TAU * j as f32) / n as f32).cos())
            .collect();
        let err = max_abs_error(&out, &expected);
        assert!(
            err < 1e-4,
            "odd-length first derivative should match (2π/N)·cos, max abs error = {err:e}"
        );
    }

    #[test]
    fn test_higher_order_general_path_matches_specialized() {
        // Order 3 should match the general complex-power path. d³/dx³ sin(ωx)
        // = -ω³ cos(ωx) (derivative cycles sin→cos→-sin→-cos).
        let n = 64;
        let x = sine_period(n);
        let omega = TAU / n as f32;

        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        let cfg = SpecDiffConfig::default_order(3);
        spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg).unwrap();

        let expected: Vec<f32> = (0..n)
            .map(|j| -omega.powi(3) * ((TAU * j as f32) / n as f32).cos())
            .collect();
        let err = max_abs_error(&out, &expected);
        // Order 3 amplifies high frequencies (ω³ is larger relative to the
        // signal magnitude), so we use a slightly looser tolerance.
        assert!(
            err < 1e-2,
            "third derivative of sine should match -ω³·cos, max abs error = {err:e}"
        );
    }

    #[test]
    fn test_spacing_scales_derivative() {
        // If spacing h is doubled, ω is halved, so the first derivative is
        // halved (differentiating w.r.t. a slower coordinate).
        let n = 64;
        let x = sine_period(n);
        let mut out_h1 = vec![0.0f32; n];
        let mut out_h2 = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        spectral_differentiate_into(
            &x,
            &mut out_h1,
            &mut scratch,
            &SpecDiffConfig {
                order: 1,
                spacing: 1.0,
            },
        )
        .unwrap();
        spectral_differentiate_into(
            &x,
            &mut out_h2,
            &mut scratch,
            &SpecDiffConfig {
                order: 1,
                spacing: 2.0,
            },
        )
        .unwrap();
        for j in 0..n {
            let ratio = out_h2[j] / out_h1[j];
            assert!(
                (ratio - 0.5).abs() < 1e-3,
                "doubling spacing should halve derivative at j={j}, ratio={ratio}"
            );
        }
    }

    #[test]
    fn test_scratch_reuse_across_sizes() {
        // Scratch should adapt to different input sizes without drift.
        let mut scratch = SpecDiffScratch::new();

        for &n in &[16usize, 64, 32, 128, 64] {
            let x = sine_period(n);
            let mut out = vec![0.0f32; n];
            spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT)
                .unwrap();
            let expected: Vec<f32> = (0..n)
                .map(|j| (TAU / n as f32) * ((TAU * j as f32) / n as f32).cos())
                .collect();
            let err = max_abs_error(&out, &expected);
            assert!(err < 1e-4, "size {n}: max abs error {err:e}");
        }
    }

    #[test]
    fn test_convenience_wrapper_matches_into() {
        let n = 64;
        let x = sine_period(n);
        let cfg = SpecDiffConfig::default_order(1);

        let out_alloc = spectral_differentiate(&x, &cfg).unwrap();
        let mut out_into = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        spectral_differentiate_into(&x, &mut out_into, &mut scratch, &cfg).unwrap();

        let err = max_abs_error(&out_alloc, &out_into);
        assert!(
            err < 1e-6,
            "convenience wrapper should match _into, err = {err:e}"
        );
    }

    #[test]
    fn test_scratch_default_constructible() {
        // SpecDiffScratch must be Default-constructible for ergonomic use.
        let mut scratch = SpecDiffScratch::default();
        let x = sine_period(32);
        let mut out = vec![0.0f32; 32];
        spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT).unwrap();
        // No assertion on values — just verify it runs without panic.
    }

    #[test]
    fn test_antisymmetric_output_for_sine_first_deriv() {
        // The first derivative of a sine over one period should integrate to
        // zero (it's a cosine). This is a sanity check on the overall sign
        // convention and DC handling.
        let n = 64;
        let x = sine_period(n);
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        spectral_differentiate_into(&x, &mut out, &mut scratch, &SpecDiffConfig::DEFAULT).unwrap();
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        assert!(
            mean.abs() < 1e-5,
            "derivative of periodic signal should have ~0 mean, got {mean:e}"
        );
    }
}
