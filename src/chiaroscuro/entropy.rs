//! Per-token spectral entropy via Type-II DCT (Plan 269, Research 237).
//!
//! Implements CHIAR-Former's per-token complexity signal:
//!
//! ```text
//! H(x) = -Σ_k p_k · log p_k / log d
//!      where  p_k = |DCT(x)_k|² / ‖DCT(x)‖²
//! ```
//!
//! Bounded to [0, 1]. `H ≈ 0` for smooth, predictable tokens (energy concentrated
//! in few low-frequency bins). `H ≈ 1` for complex tokens (energy spread across
//! all bins uniformly).
//!
//! # Relation to existing code
//!
//! - [`crate::irrep_pruner::spectral_flatness`] uses **FFT** on **logits** with
//!   Wiener entropy (geometric/arithmetic mean ratio). Different signal, different
//!   domain. CHIAR's H(x) is Shannon entropy of **DCT** of **embeddings**.
//! - [`crate::freq_bandit`] uses temporal DFT of token streams. CHIAR uses
//!   per-embedding DCT — same toolkit, orthogonal axis.
//!
//! # Complexity
//!
//! - DCT-II via `rustfft`: O(d log d)
//! - Entropy sum: O(d)
//! - Total per token: O(d log d) — negligible vs O(n²d) attention.

#![allow(clippy::needless_range_loop)]

use rustfft::{FftPlanner, num_complex::Complex32};

/// Numerical floor for `p log p` summands. Below this, treat p as 0.
const P_LOG_P_FLOOR: f32 = 1e-12;

/// Compute the per-token spectral entropy H(x) ∈ [0, 1].
///
/// `H(x) = -Σ_k p_k log p_k / log d` where `p_k = |DCT(x)_k|² / Σ|DCT(x)_k|²`.
///
/// Uses Type-II DCT implemented via `rustfft` (mirror of `flow::fft` pattern).
/// Allocates internal scratch — for hot loops, use [`spectral_entropy_dct_into`].
///
/// # Arguments
/// * `x` — token embedding of dimension `d` (any `d ≥ 1`).
///
/// # Returns
/// H(x) ∈ [0, 1]. Returns 0.0 for `d ≤ 1` (vacuous entropy).
pub fn spectral_entropy_dct(x: &[f32]) -> f32 {
    let d = x.len();
    if d <= 1 {
        return 0.0;
    }
    let mut planner = FftPlanner::<f32>::new();
    let mut scratch = Vec::with_capacity(d);
    spectral_entropy_dct_into(x, &mut scratch, &mut planner)
}

/// Zero-alloc variant of [`spectral_entropy_dct`].
///
/// Callers that invoke this per token should reuse `scratch` and `planner` across
/// calls to amortize allocations. `scratch` is resized internally as needed.
///
/// # Panics
/// Debug-asserts that `x` is non-empty.
pub fn spectral_entropy_dct_into(
    x: &[f32],
    scratch: &mut Vec<Complex32>,
    planner: &mut FftPlanner<f32>,
) -> f32 {
    let d = x.len();
    debug_assert!(!x.is_empty(), "spectral_entropy_dct_into: empty input");
    if d <= 1 {
        return 0.0;
    }

    // Compute Type-II DCT via the standard "mirror then FFT" trick:
    //   DCT-II(x)_k = Re(FFT(y))_k   where y = [x_0, x_1, ..., x_{d-1},
    //                                          x_{d-1}, x_{d-2}, ..., x_1]
    // (length 2d-2, even symmetry). The first d outputs are the DCT-II coefficients.
    //
    // For d=1, this degenerates; caller must guard.
    let n = if d == 2 { 2 } else { 2 * (d - 1) };
    if scratch.len() < n {
        scratch.resize(n, Complex32::new(0.0, 0.0));
    }
    let s = &mut scratch[..n];

    // Mirror: y[i] = x[i] for i < d; y[i] = x[2d-2-i] for d <= i < 2d-2.
    // Special-case d=2: y = [x[0], x[1]].
    s[0] = Complex32::new(x[0], 0.0);
    if d == 2 {
        s[1] = Complex32::new(x[1], 0.0);
    } else {
        for i in 1..d {
            s[i] = Complex32::new(x[i], 0.0);
        }
        for i in d..n {
            let src = n - i;
            s[i] = Complex32::new(x[src], 0.0);
        }
    }

    // Forward FFT of mirrored signal.
    let fft = planner.plan_fft_forward(n);
    fft.process(s);

    // Take real part of first d outputs, then |·|².
    // (Imaginary parts should be ≈ 0 by symmetry, but we use Re for stability.)
    // Reuse s[..d].re to store energies — avoids per-call Vec allocation.
    let mut total_energy: f32 = 0.0;
    for k in 0..d {
        let coef = s[k].re;
        let e = coef * coef;
        s[k].re = e;
        total_energy += e;
    }
    if total_energy <= 0.0 {
        // Degenerate (zero embedding). Treat as max-entropy (uninformative).
        return 1.0;
    }

    // H = -Σ p_k log p_k, normalized by log d.
    // Multiply by inv_total instead of dividing per bin.
    let mut h = 0.0f32;
    let inv_total = 1.0 / total_energy;
    for k in 0..d {
        let p = s[k].re * inv_total;
        if p > P_LOG_P_FLOOR {
            h -= p * p.ln();
        }
    }
    let log_d = (d as f32).ln();
    if log_d <= 0.0 {
        return 0.0;
    }
    h / log_d
}

/// Standard sigmoid. Used for any gating decision (NOT softmax — per project constraint).
///
/// Branches on sign to avoid `exp()` overflow for large negative inputs.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_vector_low_entropy() {
        // All-constant embedding → all energy in DC bin → H ≈ 0.
        let x = vec![1.0f32; 64];
        let h = spectral_entropy_dct(&x);
        assert!(h < 0.1, "constant vector H should be ≈ 0, got {h}");
    }

    #[test]
    fn test_zero_vector_handled() {
        let x = vec![0.0f32; 32];
        let h = spectral_entropy_dct(&x);
        // Zero total energy → we return 1.0 (uninformative, max entropy).
        assert!(
            (h - 1.0).abs() < 1e-6,
            "zero vector should be max-entropy sentinel, got {h}"
        );
    }

    #[test]
    fn test_uniform_random_higher_entropy_than_constant() {
        // Pseudo-random embedding → energy spread across bins → H closer to 1.
        // Use a simple LCG for reproducibility.
        let mut state: u32 = 0xCAFEBABE;
        let x: Vec<f32> = (0..128)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 8) as f32 / 16777216.0 - 0.5
            })
            .collect();
        let h_rand = spectral_entropy_dct(&x);
        let h_const = spectral_entropy_dct(&vec![0.5f32; 128]);
        assert!(
            h_rand > h_const + 0.3,
            "random H ({h_rand}) should be much > constant H ({h_const})"
        );
        assert!(h_rand <= 1.0, "H must be in [0, 1], got {h_rand}");
    }

    #[test]
    fn test_bounds_zero_to_one() {
        // Try various sizes and ensure result ∈ [0, 1].
        for &d in &[2usize, 4, 8, 16, 32, 64, 128, 256] {
            let x: Vec<f32> = (0..d).map(|i| (i as f32).sin()).collect();
            let h = spectral_entropy_dct(&x);
            assert!((0.0..=1.0).contains(&h), "d={d}: H={h} not in [0,1]");
        }
    }

    #[test]
    fn test_into_matches_allocating() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let h1 = spectral_entropy_dct(&x);
        let mut scratch = Vec::new();
        let mut planner = FftPlanner::new();
        let h2 = spectral_entropy_dct_into(&x, &mut scratch, &mut planner);
        assert!((h1 - h2).abs() < 1e-5, "allocating {h1} != into {h2}");
    }

    #[test]
    fn test_scratch_reuse_across_sizes() {
        // Verify scratch can be reused across different input sizes.
        let mut scratch = Vec::new();
        let mut planner = FftPlanner::new();
        let h1 = spectral_entropy_dct_into(&[1.0f32; 32], &mut scratch, &mut planner);
        let h2 = spectral_entropy_dct_into(&[1.0f32; 128], &mut scratch, &mut planner);
        // Both constant → both low entropy.
        assert!(h1 < 0.1 && h2 < 0.1, "constant vectors: h1={h1}, h2={h2}");
    }

    #[test]
    fn test_sigmoid_not_softmax() {
        // σ(x) + σ(y) ≠ 1 in general (it would if we used softmax).
        let s1 = sigmoid(1.0);
        let s2 = sigmoid(2.0);
        assert!((s1 - s2).abs() > 1e-6, "sigmoid outputs must differ");
        assert!(
            (s1 + s2 - 1.0).abs() > 1e-6,
            "sigmoid outputs must not sum to 1 (that would be softmax)"
        );
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(-100.0) < 1e-6);
        assert!(sigmoid(100.0) > 1.0 - 1e-6);
    }
}
