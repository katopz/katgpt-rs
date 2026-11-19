//! Kernel-derived Fourier latent embeddings (Issue 1234; Research 682,
//! arXiv:2602.15029 — the symmetry→geometry theorem).
//!
//! When pairwise statistics are translation-invariant
//! (`K_ij = C(dist(x_i, x_j))`), the Gram is diagonalized by Fourier modes and
//! the embedding a spectral method learns is
//! `w_i = Σ_n a_n·(cos k_n x_i, sin k_n x_i)` with frequencies `k_n` and
//! amplitudes `a_n` fixed by the kernel. This module builds that embedding
//! **from the kernel** — deterministic, closed-form, no random frequencies
//! (RFF) and no hand-set ladder (sinusoidal PEs).
//!
//! # Coordinates
//!
//! The latent coordinate lives on `x ∈ [−1, 1]` (period 2 on a ring): map a
//! ring position `θ ∈ [0, 1)` to `x = 2θ`, a line segment linearly onto
//! `[−1, 1]`. Kernel: the exponential `exp(−|x − y|/σ)` (periodized on the
//! ring), the paper's fitted form.
//!
//! # Boundaries
//!
//! - **Periodic** (ring): `k_n = πn`. Amplitudes either
//!   [`PeriodicAmplitudes::Lorentzian`] — the continuum periodized kernel,
//!   `a_n = sqrt(2σ/(1 + σ²k_n²))` — or [`PeriodicAmplitudes::DiscreteExact`]
//!   for `N` evenly spaced samples: `a_n² = 2λ_n/N` from the circulant's own
//!   eigenvalues (aliasing folded in), so the full-width Gram reproduces the
//!   sampled centered kernel exactly. The Nyquist mode (`n = N/2`, N even)
//!   carries no sine component.
//! - **Open** (line): Sturm–Liouville modes of the exponential kernel on
//!   `[−1, 1]`, eigenvalue `λ = 2σ/(1 + σ²k²)`. Odd modes `sin kx` quantize
//!   by `tan k = −σk`. Even modes depend on whether the statistics are
//!   double-centered (PMI-style, the paper's form) — **measured**
//!   (fine-grid eigendecomposition, σ ∈ {0.5, 1, 4.2}, Issue 1234 note):
//!   [`OpenKernel::Raw`] `tan k = 1/(σk)`, eigenfunction `cos kx`;
//!   [`OpenKernel::Centered`] `tan k = k/(1 + σ(1+σ)k²)`, eigenfunction
//!   `cos kx − sin(k)/k` (orthogonal to constants). Every root is found by
//!   bisection inside its PINNED per-mode interval — odd
//!   `((m+½)π, (m+1)π)`, even `(mπ, (m+½)π)` — each interval holds exactly
//!   one root, so no global search can land the wrong mode.
//!
//! # Layout + allocation
//!
//! A feature row is `[f32; 2w]`, interleaved `(cos-part, sin-part)` per mode
//! pair. The spectrum is computed once in f64 into fixed arrays
//! ([`KernelSpectrum`], `w ≤ MAX_MODES`); every function writes into
//! caller-provided slices and allocates nothing.

use core::f64::consts::PI;

/// Maximum mode pairs a [`KernelSpectrum`] holds.
pub const MAX_MODES: usize = 128;

/// Refuse-loud errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelFeatureError {
    /// `w == 0` or `w > MAX_MODES`, or (DiscreteExact) `w > N/2`.
    WidthOutOfRange { w: usize },
    /// `σ` not finite and positive.
    BadSigma,
    /// An output/input slice length disagrees with the contract.
    LengthMismatch { expected: usize },
    /// A quantization bracket did not change sign (mode index named) —
    /// unreachable for finite positive σ; reported, never papered over.
    BracketNoSignChange { mode: usize },
}

impl core::fmt::Display for KernelFeatureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WidthOutOfRange { w } => write!(f, "width {w} out of range"),
            Self::BadSigma => write!(f, "sigma must be finite and > 0"),
            Self::LengthMismatch { expected } => write!(f, "slice length != {expected}"),
            Self::BracketNoSignChange { mode } => {
                write!(f, "quantization bracket for mode {mode} has no sign change")
            }
        }
    }
}

impl std::error::Error for KernelFeatureError {}

/// Periodic-boundary amplitude source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PeriodicAmplitudes {
    /// Continuum periodized exponential: `a_n = sqrt(2σ/(1+σ²k_n²))`.
    Lorentzian,
    /// `N` evenly spaced samples (`x_i = 2i/N`): exact circulant amplitudes.
    DiscreteExact { n_points: usize },
}

/// Open-boundary kernel form (see the module doc for the measured split).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenKernel {
    /// The raw exponential kernel.
    Raw,
    /// The double-centered kernel (PMI-style statistics).
    Centered,
}

/// The kernel spectrum: per mode pair `n`, a cos-part `(k, a, offset)` and a
/// sin-part `(k, a)`. Feature pair n of coordinate `x` is
/// `(a_c·(cos(k_c x) − off), a_s·sin(k_s x))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelSpectrum {
    w: usize,
    k_cos: [f64; MAX_MODES],
    a_cos: [f64; MAX_MODES],
    off_cos: [f64; MAX_MODES],
    k_sin: [f64; MAX_MODES],
    a_sin: [f64; MAX_MODES],
}

impl KernelSpectrum {
    fn empty(w: usize) -> Self {
        Self {
            w,
            k_cos: [0.0; MAX_MODES],
            a_cos: [0.0; MAX_MODES],
            off_cos: [0.0; MAX_MODES],
            k_sin: [0.0; MAX_MODES],
            a_sin: [0.0; MAX_MODES],
        }
    }

    /// Mode pairs held.
    pub fn width(&self) -> usize {
        self.w
    }

    /// `(k_cos, a_cos, k_sin, a_sin)` of pair `n` (0-based).
    pub fn mode(&self, n: usize) -> (f64, f64, f64, f64) {
        (self.k_cos[n], self.a_cos[n], self.k_sin[n], self.a_sin[n])
    }

    /// Periodic (ring) spectrum, `w` mode pairs `n = 1..=w`, `k_n = πn`.
    pub fn periodic(
        sigma: f64,
        w: usize,
        amps: PeriodicAmplitudes,
    ) -> Result<Self, KernelFeatureError> {
        check_sigma(sigma)?;
        check_width(w)?;
        let mut s = Self::empty(w);
        match amps {
            PeriodicAmplitudes::Lorentzian => {
                for i in 0..w {
                    let k = PI * (i + 1) as f64;
                    let a = lorentzian_sq(sigma, k).sqrt();
                    (s.k_cos[i], s.a_cos[i], s.k_sin[i], s.a_sin[i]) = (k, a, k, a);
                }
            }
            PeriodicAmplitudes::DiscreteExact { n_points } => {
                if n_points < 2 || w > n_points / 2 {
                    return Err(KernelFeatureError::WidthOutOfRange { w });
                }
                for i in 0..w {
                    let n = i + 1;
                    let k = PI * n as f64;
                    // λ_n = Σ_j c_j cos(2π n j / N), c_j = periodized
                    // exponential at d = 2j/N: cosh((1 − d)/σ)/sinh(1/σ).
                    let mut lam = 0.0;
                    for j in 0..n_points {
                        let d = 2.0 * j as f64 / n_points as f64;
                        let c = ((1.0 - d) / sigma).cosh() / (1.0 / sigma).sinh();
                        lam += c * (2.0 * PI * (n * j) as f64 / n_points as f64).cos();
                    }
                    let nyquist = 2 * n == n_points;
                    let a_sq = if nyquist { lam } else { 2.0 * lam } / n_points as f64;
                    let a = a_sq.max(0.0).sqrt();
                    (s.k_cos[i], s.a_cos[i], s.k_sin[i]) = (k, a, k);
                    s.a_sin[i] = if nyquist { 0.0 } else { a };
                }
            }
        }
        Ok(s)
    }

    /// Open (line) spectrum on `[−1, 1]`: the first `w` even and `w` odd
    /// Sturm–Liouville modes, paired by index in descending-eigenvalue order
    /// within each parity. Eigenfunctions are L²-normalized on `[−1, 1]`.
    pub fn open(sigma: f64, w: usize, kernel: OpenKernel) -> Result<Self, KernelFeatureError> {
        check_sigma(sigma)?;
        check_width(w)?;
        let mut s = Self::empty(w);
        for i in 0..w {
            let m = i as f64;
            // Odd: sin k + σk cos k = 0 on ((m+½)π, (m+1)π).
            let ko = bisect(
                |k| k.sin() + sigma * k * k.cos(),
                (m + 0.5) * PI,
                (m + 1.0) * PI,
                i,
            )?;
            let norm_o = 1.0 - (2.0 * ko).sin() / (2.0 * ko);
            s.k_sin[i] = ko;
            s.a_sin[i] = (lorentzian_sq(sigma, ko) / norm_o).sqrt();
            // Even.
            let (ke, off, norm_e) = match kernel {
                OpenKernel::Raw => {
                    // σk sin k − cos k = 0 on (mπ, (m+½)π).
                    let k = bisect(|k| sigma * k * k.sin() - k.cos(), m * PI, (m + 0.5) * PI, i)?;
                    (k, 0.0, 1.0 + (2.0 * k).sin() / (2.0 * k))
                }
                OpenKernel::Centered => {
                    // sin k (1 + σ(1+σ)k²) − k cos k = 0 on (m'π, (m'+½)π),
                    // m' = m + 1 (m' = 0 holds only the trivial k = 0).
                    let mp = m + 1.0;
                    let c = sigma * (1.0 + sigma);
                    let k = bisect(
                        |k| k.sin() * (1.0 + c * k * k) - k * k.cos(),
                        mp * PI,
                        (mp + 0.5) * PI,
                        i,
                    )?;
                    let off = k.sin() / k;
                    (k, off, 1.0 + (2.0 * k).sin() / (2.0 * k) - 2.0 * off * off)
                }
            };
            s.k_cos[i] = ke;
            s.off_cos[i] = off;
            s.a_cos[i] = (lorentzian_sq(sigma, ke) / norm_e).sqrt();
        }
        Ok(s)
    }

    /// Writes the `2w` feature row of coordinate `x` into `out`.
    pub fn features_into(&self, x: f32, out: &mut [f32]) -> Result<(), KernelFeatureError> {
        if out.len() != 2 * self.w {
            return Err(KernelFeatureError::LengthMismatch {
                expected: 2 * self.w,
            });
        }
        let x = x as f64;
        for n in 0..self.w {
            out[2 * n] = (self.a_cos[n] * ((self.k_cos[n] * x).cos() - self.off_cos[n])) as f32;
            out[2 * n + 1] = (self.a_sin[n] * (self.k_sin[n] * x).sin()) as f32;
        }
        Ok(())
    }
}

fn check_sigma(sigma: f64) -> Result<(), KernelFeatureError> {
    if sigma.is_finite() && sigma > 0.0 {
        Ok(())
    } else {
        Err(KernelFeatureError::BadSigma)
    }
}

fn check_width(w: usize) -> Result<(), KernelFeatureError> {
    if w == 0 || w > MAX_MODES {
        Err(KernelFeatureError::WidthOutOfRange { w })
    } else {
        Ok(())
    }
}

/// `2σ/(1 + σ²k²)` — the exponential kernel's spectral density.
fn lorentzian_sq(sigma: f64, k: f64) -> f64 {
    2.0 * sigma / (1.0 + sigma * sigma * k * k)
}

/// Bisection inside a pinned bracket `(lo, hi)` (open ends nudged inward by
/// one ulp-scale step). Deterministic: fixed 200-step cap, stops when the
/// bracket stops shrinking.
fn bisect(
    f: impl Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    mode: usize,
) -> Result<f64, KernelFeatureError> {
    let (mut a, mut b) = (lo + 1e-12 * hi, hi - 1e-12 * hi);
    let (mut fa, fb) = (f(a), f(b));
    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if fa.signum() == fb.signum() {
        return Err(KernelFeatureError::BracketNoSignChange { mode });
    }
    for _ in 0..200 {
        let mid = 0.5 * (a + b);
        if mid <= a || mid >= b {
            break;
        }
        let fm = f(mid);
        if fm == 0.0 {
            return Ok(mid);
        }
        if fm.signum() == fa.signum() {
            a = mid;
            fa = fm;
        } else {
            b = mid;
        }
    }
    Ok(0.5 * (a + b))
}

/// Writes the `n × 2w` row-major embedding of `xs` into `out`.
pub fn fourier_features_from_kernel(
    spectrum: &KernelSpectrum,
    xs: &[f32],
    out: &mut [f32],
) -> Result<(), KernelFeatureError> {
    let row = 2 * spectrum.w;
    if out.len() != xs.len() * row {
        return Err(KernelFeatureError::LengthMismatch {
            expected: xs.len() * row,
        });
    }
    for (x, dst) in xs.iter().zip(out.chunks_exact_mut(row)) {
        spectrum.features_into(*x, dst)?;
    }
    Ok(())
}

/// Smallest width `w` whose periodic Lorentzian tail-energy share
/// `Σ_{n>w} a_n² / Σ_{n≥1} a_n²` is below `tau`. The total is closed-form:
/// `Σ_{n≥1} 2σ/(1+σ²π²n²) = (2/(σπ²))·(πc·coth(πc) − 1)/(2c²)`, `c = 1/(σπ)`.
/// Returns `None` when `MAX_MODES` cannot reach `tau`.
pub fn width_for_tolerance(sigma: f64, tau: f64) -> Option<usize> {
    if !(sigma.is_finite() && sigma > 0.0) {
        return None;
    }
    let c = 1.0 / (sigma * PI);
    let total = (2.0 / (sigma * PI * PI)) * (PI * c / (PI * c).tanh() - 1.0) / (2.0 * c * c);
    let mut head = 0.0;
    for w in 1..=MAX_MODES {
        head += lorentzian_sq(sigma, PI * w as f64);
        if (total - head).max(0.0) / total < tau {
            return Some(w);
        }
    }
    None
}

/// The Prop-4 decode law `ε² ≲ r^(−1/D)` read backwards: the embedding rank
/// needed to resolve a `d_latent`-dimensional coordinate to phase error
/// `delta` is `r ≈ delta^(−2·D)`. ORDER-OF-MAGNITUDE (the law is a `≲`, its
/// constant unstated) — a sizing guide, never a guarantee. Saturates at
/// `usize::MAX`.
pub fn dims_for_phase_resolution(delta: f64, d_latent: u32) -> usize {
    if !((0.0..1.0).contains(&delta)) || d_latent == 0 {
        return 1;
    }
    let r = delta.powf(-2.0 * d_latent as f64).ceil();
    if r >= usize::MAX as f64 {
        usize::MAX
    } else {
        r as usize
    }
}

/// Phase readout: `out[i] = atan2(sin_col[i], cos_col[i]) / k` wrapped to
/// `[0, 2π/k)` — the latent coordinate (up to the conjugate pair's global
/// rotation/reflection when the columns come from an eigensolve) read off a
/// cos/sin pair at wavenumber `k`. A latent→scalar boundary readout: callers
/// sigmoid-gate it where it drives behaviour, and pass it raw where it
/// crosses sync.
pub fn phase_readout_into(
    cos_col: &[f32],
    sin_col: &[f32],
    k: f32,
    out: &mut [f32],
) -> Result<(), KernelFeatureError> {
    if sin_col.len() != cos_col.len() || out.len() != cos_col.len() {
        return Err(KernelFeatureError::LengthMismatch {
            expected: cos_col.len(),
        });
    }
    let period = core::f32::consts::TAU / k;
    for ((o, &c), &s) in out.iter_mut().zip(cos_col).zip(sin_col) {
        let x = s.atan2(c) / k;
        *o = if x < 0.0 { x + period } else { x };
    }
    Ok(())
}

/// Translation-symmetry fingerprint (report-only): over the top `top_m`
/// entries of a DESCENDING eigenvalue list, the fraction that sit in matched
/// conjugate pairs `|λ_i − λ_{i+1}| ≤ rel_tol·|λ_i|` (greedy, ascending
/// index, first-index pairing). A circulant (translation-invariant) Gram
/// scores 1.0 on its centered top modes; generic symmetric matrices score
/// near 0.
pub fn conjugate_pairing_score(eigvals_desc: &[f64], top_m: usize, rel_tol: f64) -> f64 {
    let m = top_m.min(eigvals_desc.len());
    if m == 0 {
        return 0.0;
    }
    let (mut i, mut paired) = (0usize, 0usize);
    while i + 1 < m {
        let (a, b) = (eigvals_desc[i], eigvals_desc[i + 1]);
        if (a - b).abs() <= rel_tol * a.abs() {
            paired += 2;
            i += 2;
        } else {
            i += 1;
        }
    }
    paired as f64 / m as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_roots_match_the_measured_quantization() {
        // Pinned to the exact roots (scipy brentq on the same conditions, 4
        // d.p.); the fine-grid eigendecomposition agreed to ~2e-3 (grid
        // discretization) — the open-BC GOAT test checks the eigenfunctions.
        let raw = KernelSpectrum::open(1.0, 2, OpenKernel::Raw).unwrap();
        let cen = KernelSpectrum::open(1.0, 2, OpenKernel::Centered).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() < 2e-4;
        assert!(close(raw.k_cos[0], 0.8603) && close(raw.k_cos[1], 3.4256));
        assert!(close(raw.k_sin[0], 2.0288) && close(raw.k_sin[1], 4.9132));
        assert!(close(cen.k_cos[0], 3.2860) && close(cen.k_cos[1], 6.3607));
        assert!(close(cen.k_sin[0], 2.0288));
        let c42 = KernelSpectrum::open(4.2, 2, OpenKernel::Centered).unwrap();
        assert!(close(c42.k_cos[0], 3.1560) && close(c42.k_sin[0], 1.7092));
    }

    #[test]
    fn width_selector_is_monotone_and_tail_correct() {
        let w1 = width_for_tolerance(1.0, 0.05).unwrap();
        let w2 = width_for_tolerance(1.0, 0.01).unwrap();
        assert!(w2 > w1);
        // Direct check of the tail share at the returned width.
        let total: f64 = (1..200_000)
            .map(|n| lorentzian_sq(1.0, PI * n as f64))
            .sum();
        let head: f64 = (1..=w2).map(|n| lorentzian_sq(1.0, PI * n as f64)).sum();
        assert!((total - head) / total < 0.0101);
        let head_less: f64 = (1..w2).map(|n| lorentzian_sq(1.0, PI * n as f64)).sum();
        assert!((total - head_less) / total >= 0.0099);
    }

    #[test]
    fn dims_for_phase_resolution_reads_the_law_backwards() {
        assert_eq!(dims_for_phase_resolution(0.1, 1), 100);
        assert_eq!(dims_for_phase_resolution(0.5, 2), 16);
        assert_eq!(dims_for_phase_resolution(1.5, 1), 1);
    }

    #[test]
    fn pairing_score_reads_degenerate_pairs() {
        assert_eq!(
            conjugate_pairing_score(&[5.0, 5.0, 2.0, 2.0, 1.0], 4, 1e-9),
            1.0
        );
        assert_eq!(conjugate_pairing_score(&[5.0, 4.0, 3.0, 2.0], 4, 1e-3), 0.0);
        assert_eq!(conjugate_pairing_score(&[9.0, 5.0, 5.0, 1.0], 4, 1e-9), 0.5);
    }

    #[test]
    fn refusals() {
        assert_eq!(
            KernelSpectrum::periodic(0.0, 2, PeriodicAmplitudes::Lorentzian),
            Err(KernelFeatureError::BadSigma)
        );
        assert_eq!(
            KernelSpectrum::periodic(1.0, 13, PeriodicAmplitudes::DiscreteExact { n_points: 24 }),
            Err(KernelFeatureError::WidthOutOfRange { w: 13 })
        );
        let s = KernelSpectrum::periodic(1.0, 2, PeriodicAmplitudes::Lorentzian).unwrap();
        assert_eq!(
            s.features_into(0.3, &mut [0.0; 3]),
            Err(KernelFeatureError::LengthMismatch { expected: 4 })
        );
    }
}
