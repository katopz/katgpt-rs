//! Temporal Derivative Kernel — Dual Fast/Slow Surprise Signal (Plan 277).
//!
//! Distilled from O'Reilly 2026, "This is how the Neocortex Learns"
//! ([arXiv:2606.08720](https://arxiv.org/abs/2606.08720)). Research note:
//! [`katgpt-rs/.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md`](../.research/243_Temporal_Derivative_Kernel_Neocortical_Learning.md).
//!
//! ## Co-extraction provenance (Plan 338 Phase 2)
//!
//! The kernel struct + impls + `sigmoid_surprise_gate` were promoted to
//! `katgpt_types::temporal` so the `katgpt-sense` crate can consume the
//! kernel via the leaf only, breaking the katgpt-core cycle. This file is
//! now a thin re-export shim — `katgpt_core::temporal_deriv::*` paths are
//! preserved bit-for-bit. Tests stay here and exercise the kernel through
//! the re-export.

#![allow(clippy::needless_range_loop)]

pub use katgpt_types::temporal::{TemporalDerivativeKernel, sigmoid_surprise_gate};

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero signal → zero derivative, integrators stay at 0.
    #[test]
    fn zero_signal_yields_zero_derivative() {
        let mut k: TemporalDerivativeKernel<4> = TemporalDerivativeKernel::new(0.5, 0.05);
        for _ in 0..10 {
            let d = k.observe(&[0.0; 4]);
            assert!(d.iter().all(|x| *x == 0.0), "derivative must stay zero");
        }
        assert_eq!(k.surprise_norm(), 0.0);
    }

    /// Constant signal → derivative converges to 0 (paper's 25→25 and 50→50
    /// cases: flat → no change).
    #[test]
    fn constant_signal_converges_to_zero_derivative() {
        let mut k: TemporalDerivativeKernel<2> = TemporalDerivativeKernel::new(0.5, 0.05);
        let signal = [0.5f32, 0.5];
        // Warm up: first step spikes the derivative.
        let _ = k.observe(&signal);
        let early_norm = k.surprise_norm();
        assert!(early_norm > 0.0, "first step must produce nonzero surprise");
        // Long run: derivative decays as slow catches up.
        for _ in 0..2000 {
            k.observe(&signal);
        }
        let late_norm = k.surprise_norm();
        assert!(
            late_norm < early_norm * 0.01,
            "derivative must decay to ~0 on constant signal; early={}, late={}",
            early_norm,
            late_norm
        );
    }

    /// Step signal (0→1) → positive derivative spike that eventually decays
    /// as the slow EMA catches up to the fast EMA.
    ///
    /// Note: the derivative is NOT monotone-decreasing from t=1. The fast EMA
    /// rises quickly while the slow EMA barely moves, so `(fast − slow)` can
    /// initially *increase* for a few steps before the slow EMA catches up
    /// and the difference decays. The test therefore checks (a) the spike is
    /// positive, (b) the derivative eventually decays to <10% of its peak.
    #[test]
    fn step_up_signal_produces_positive_spike() {
        let mut k: TemporalDerivativeKernel<1> = TemporalDerivativeKernel::new(0.5, 0.05);
        // Flat at zero.
        for _ in 0..100 {
            k.observe(&[0.0]);
        }
        assert!(k.surprise_norm() < 1e-6);
        // Step up.
        let d = k.observe(&[1.0]);
        assert!(d[0] > 0.0, "step up must produce positive derivative");
        // Track the peak across the first 20 steps (may be > t=1 value).
        let mut peak = k.surprise_norm();
        for _ in 0..20 {
            k.observe(&[1.0]);
            let n = k.surprise_norm();
            if n > peak {
                peak = n;
            }
        }
        assert!(peak > 0.0, "peak must be positive after step up");
        // Continue observing; derivative must decay substantially as the slow
        // EMA converges to the same value as the fast EMA.
        for _ in 0..2000 {
            k.observe(&[1.0]);
        }
        let late = k.surprise_norm();
        assert!(
            late < peak * 0.1,
            "derivative must decay to <10% of peak; peak={}, late={}",
            peak,
            late
        );
    }

    /// Reverse step (1→0) → negative derivative spike.
    #[test]
    fn step_down_signal_produces_negative_spike() {
        let mut k: TemporalDerivativeKernel<1> = TemporalDerivativeKernel::new(0.5, 0.05);
        // Warm up at 1.0 until derivative ~ 0.
        for _ in 0..1000 {
            k.observe(&[1.0]);
        }
        assert!(k.surprise_norm() < 1e-3);
        // Step down.
        let d = k.observe(&[0.0]);
        assert!(d[0] < 0.0, "step down must produce negative derivative");
    }

    /// `alpha_fast > alpha_slow` is enforced (debug panic on violation).
    #[test]
    #[should_panic(expected = "require 0 < alpha_slow < alpha_fast <= 1")]
    #[cfg(debug_assertions)]
    fn swapped_alphas_panics_in_debug() {
        let _ = TemporalDerivativeKernel::<4>::new(0.05, 0.5);
    }

    /// `reset()` zeroes both EMA arrays.
    #[test]
    fn reset_zeroes_state() {
        let mut k: TemporalDerivativeKernel<2> = TemporalDerivativeKernel::new(0.5, 0.05);
        k.observe(&[1.0, 1.0]);
        k.observe(&[1.0, 1.0]);
        assert!(k.surprise_norm() > 0.0);
        k.reset();
        assert_eq!(k.fast, [0.0, 0.0]);
        assert_eq!(k.slow, [0.0, 0.0]);
        assert_eq!(k.surprise_norm(), 0.0);
    }

    /// `surprise_norm()` matches a manual L2 computation.
    #[test]
    fn surprise_norm_matches_manual_l2() {
        let k: TemporalDerivativeKernel<4> = TemporalDerivativeKernel::with_initial(
            [0.3, -0.7, 0.1, 0.5],
            [0.1, -0.2, 0.05, 0.1],
            0.5,
            0.05,
        );
        let mut diff = [0.0f32; 4];
        k.derivative_slice(&mut diff);
        let manual_sq: f32 = diff.iter().map(|x| x * x).sum();
        let manual = manual_sq.sqrt();
        let got = k.surprise_norm();
        assert!(
            (got - manual).abs() < 1e-5,
            "surprise_norm={} must match manual L2={}",
            got,
            manual
        );
    }

    /// `derivative_slice` matches `observe` return value.
    #[test]
    fn derivative_slice_matches_observe_output() {
        let mut k: TemporalDerivativeKernel<3> = TemporalDerivativeKernel::new(0.4, 0.04);
        let signal = [0.7, -0.3, 0.5];
        let d = k.observe(&signal);
        let mut buf = [0.0f32; 3];
        k.derivative_slice(&mut buf);
        for i in 0..3 {
            assert!((d[i] - buf[i]).abs() < 1e-6);
        }
    }

    /// `observe_simd` matches `observe` (they route through the same SIMD
    /// primitives; equivalence is structural but asserted for regression
    /// safety).
    #[test]
    fn observe_simd_matches_observe() {
        let mut a: TemporalDerivativeKernel<8> = TemporalDerivativeKernel::new(0.4, 0.04);
        let mut b = a.clone();
        let signal = [0.5f32; 8];
        for _ in 0..50 {
            let da = a.observe(&signal);
            let db = b.observe_simd(&signal);
            assert_eq!(da, db);
        }
    }

    /// `sigmoid_surprise_gate` returns a value in (0, 1) and is monotone
    /// increasing in the derivative norm.
    #[test]
    fn sigmoid_surprise_gate_is_bounded_and_monotone() {
        let zero = [0.0f32; 4];
        let small = [0.1f32; 4];
        let big = [1.0f32; 4];
        let g_zero = sigmoid_surprise_gate(&zero, 4.0);
        let g_small = sigmoid_surprise_gate(&small, 4.0);
        let g_big = sigmoid_surprise_gate(&big, 4.0);
        assert!(g_zero > 0.0 && g_zero < 1.0);
        assert!(g_small > g_zero);
        assert!(g_big > g_small);
        assert!(g_big < 1.0);
    }

    /// Default constructor matches paper's ~10× ratio.
    #[test]
    fn default_is_paper_ten_to_one_ratio() {
        let k: TemporalDerivativeKernel<4> = TemporalDerivativeKernel::default();
        assert!((k.alpha_fast - 0.3).abs() < 1e-6);
        assert!((k.alpha_slow - 0.03).abs() < 1e-6);
    }

    /// `with_initial` preserves warm-start values.
    #[test]
    fn with_initial_preserves_state() {
        let k: TemporalDerivativeKernel<2> =
            TemporalDerivativeKernel::with_initial([0.5, 0.5], [0.1, 0.1], 0.4, 0.04);
        assert_eq!(k.fast, [0.5, 0.5]);
        assert_eq!(k.slow, [0.1, 0.1]);
    }
}
