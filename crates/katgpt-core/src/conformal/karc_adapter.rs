//! KARC → [`PointForecaster`] adapter (Plan 340 Phase 2, T2.1).
//!
//! [`KarcChannelForecaster`] exposes ONE output channel of a fitted
//! [`KarcForecaster`] as a single-channel `PointForecaster`, so a KARC model
//! can be wrapped by [`ConformalIntervalCalibrator`] for type-level
//! composition and the `observe_and_update` write path.
//!
//! ## Why an adapter (not `impl PointForecaster for KarcForecaster`)
//!
//! Two mismatches prevent a direct impl:
//!
//! 1. **Arity.** `PointForecaster::forecast_into` writes ONE `f32` (the
//!    conformal overlay is per-channel). KARC's `forecast_into` writes the
//!    full `D`-channel vector in one matvec. The adapter runs the matvec once
//!    and projects out the configured channel.
//! 2. **`&mut self`.** KARC's `forecast_into` reuses a pre-allocated feature
//!    buffer (`forecast_psi`, length `d_h = K·D·M`) as scratch and therefore
//!    takes `&mut self`. The `PointForecaster` trait now takes `&mut self`
//!    too (Plan 340 Phase 2 change), so the adapter can forward directly
//!    without interior mutability — no `RefCell`, no `UnsafeCell`, zero
//!    overhead beyond KARC's own hot path.
//!
//! ## The KARC + conformal integration pattern
//!
//! KARC needs a REAL delay state (length `K·D`), but
//! [`ConformalIntervalCalibrator::interval_into`] passes an empty delay state
//! to the wrapped forecaster (it's designed for self-contained forecasters
//! like the seasonal pool). KARC callers therefore use the **point-supplied**
//! read path [`ConformalIntervalCalibrator::interval_from_point_into`]:
//!
//! ```text
//! // Per tick, per KARC model:
//! //   1. KARC forecasts all D channels in one matvec.
//! let mut point = [0.0_f32; D];
//! karc.forecast_into(delay_state, &mut point);
//! //   2. For each channel c, read the calibrated interval.
//! let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, alpha);
//! cal.interval_from_point_into(point[c], c, h, alpha, &mut iv);
//! //   3. After observing the realized actual[c], update the residual pool.
//! cal.update_residual(actual[c], point[c], c, h);
//! ```
//!
//! The adapter is still useful for two reasons:
//! - **Type-level composition** — `ConformalIntervalCalibrator<KarcChannelForecaster<..>>`
//!   is a concrete type you can store, pass around, and generic-over.
//! - **`observe_and_update`** — that method DOES forward the real `delay_state`
//!   to the wrapped forecaster, so the adapter handles the full write path in
//!   one call.
//!
//! Calling `interval_into` on a calibrator wrapping this adapter will hit the
//! `delay_state.len() == K * D` debug-assert and panic in debug builds; in
//! release builds it returns 0.0 (KARC's "not fitted" path). Use
//! `interval_from_point_into` instead — that's the documented KARC pattern.

use super::PointForecaster;
use crate::karc::{KarcBasis, KarcForecaster};

/// KARC adapter that exposes one channel of a [`KarcForecaster`] as a
/// single-channel [`PointForecaster`].
///
/// See the [module docs](self) for the integration pattern and the rationale
/// for the adapter (arity + `&mut self`).
///
/// Holds a pre-allocated `D`-length scratch for the full KARC output vector —
/// allocated once at construction, reused on every forecast (zero-alloc on
/// the hot path, matching KARC's own G3 guarantee).
pub struct KarcChannelForecaster<B, const D: usize, const M: usize, const K: usize>
where
    B: KarcBasis<M>,
{
    /// The wrapped, already-fitted KARC forecaster.
    pub karc: KarcForecaster<B, D, M, K>,
    /// Which output channel this adapter exposes (`0..D`).
    pub channel: usize,
    /// Pre-allocated scratch for KARC's full `D`-channel output. Reused on
    /// every `forecast_into` call — never reallocated after construction.
    scratch_out: Vec<f32>,
}

impl<B, const D: usize, const M: usize, const K: usize> KarcChannelForecaster<B, D, M, K>
where
    B: KarcBasis<M>,
{
    /// Wrap a fitted KARC forecaster, exposing `channel` as the single output.
    ///
    /// **Panics** if `channel >= D`.
    pub fn new(karc: KarcForecaster<B, D, M, K>, channel: usize) -> Self {
        assert!(channel < D, "channel {channel} out of range [0, {D})");
        Self {
            karc,
            channel,
            scratch_out: vec![0.0_f32; D],
        }
    }

    /// Change which channel this adapter exposes. **Panics** if `channel >= D`.
    #[inline]
    pub fn set_channel(&mut self, channel: usize) {
        assert!(channel < D, "channel {channel} out of range [0, {D})");
        self.channel = channel;
    }
}

impl<B, const D: usize, const M: usize, const K: usize> PointForecaster
    for KarcChannelForecaster<B, D, M, K>
where
    B: KarcBasis<M>,
{
    /// Forecast via KARC and return the configured channel's value.
    ///
    /// **Requires** `delay_state.len() == K * D` (KARC's delay embedding).
    /// KARC only forecasts at `h=1`; the `h` parameter is intentionally
    /// ignored — multi-horizon conformal intervals come from the residual
    /// pool's horizon-bucket indexing, not from KARC itself.
    ///
    /// If KARC is not fitted, writes `0.0` (mirroring KARC's own
    /// `forecast_into` "not fitted → false" contract).
    #[inline]
    fn forecast_into(&mut self, delay_state: &[f32], _h: usize, out: &mut f32) {
        debug_assert_eq!(
            delay_state.len(),
            K * D,
            "KARC requires delay_state of length K*D = {}, got {}",
            K * D,
            delay_state.len()
        );
        let ok = self.karc.forecast_into(delay_state, &mut self.scratch_out);
        *out = if ok { self.scratch_out[self.channel] } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::karc::ChebyshevBasis;
    use crate::{ConformalIntervalCalibrator, DecayUnit, PredictiveInterval, ResidualMode};

    /// Build a tiny fitted KARC forecaster on a sinusoid, return it + the
    /// last delay state (length K·D) + the next ground-truth value.
    fn make_fitted_karc() -> (KarcForecaster<ChebyshevBasis<4>, 2, 4, 3>, Vec<f32>, f32) {
        // D=2, M=4, K=3, λ small. Trajectory: two-channel sinusoid.
        const D: usize = 2;
        const M: usize = 4;
        const K: usize = 3;
        const N: usize = 64;
        let basis = ChebyshevBasis::<M>::new();
        let mut f = KarcForecaster::with_capacity(basis, N);

        // Build training pairs: x_t = [u_t, u_{t-1}, u_{t-2}] (flattened K·D),
        // target = u_{t+1}.
        let traj: Vec<[f32; D]> = (0..N + K + 1)
            .map(|i| {
                let t = i as f32 * 0.1;
                [t.sin(), (t * 0.7).cos()]
            })
            .collect();

        for t in (K - 1)..N + K - 1 {
            // delay_state = [u_t, u_{t-1}, u_{t-2}] flattened (K·D).
            let mut ds = [0.0_f32; K * D];
            for k in 0..K {
                let row = &traj[t - k];
                ds[k * D..(k + 1) * D].copy_from_slice(row);
            }
            let target = traj[t + 1];
            f.accumulate_pair(&ds, &target);
        }
        f.fit_ridge(1e-6).expect("fit");

        // Last delay state + next ground truth for the test.
        let t = N + K - 1;
        let mut ds = vec![0.0_f32; K * D];
        for k in 0..K {
            let row = &traj[t - k];
            ds[k * D..(k + 1) * D].copy_from_slice(row);
        }
        let next = traj[t + 1];
        (f, ds, next[0])
    }

    #[test]
    fn adapter_extracts_configured_channel() {
        let (karc, ds, _) = make_fitted_karc();

        // Forecast the full vector directly via KARC.
        let mut full = [0.0_f32; 2];
        let mut karc_mut = karc;
        assert!(karc_mut.forecast_into(&ds, &mut full));

        // Channel 0 adapter must match full[0].
        let mut adapter0 = KarcChannelForecaster::new(karc_mut, 0);
        let mut out0 = 0.0_f32;
        adapter0.forecast_into(&ds, 1, &mut out0);
        assert!(
            (out0 - full[0]).abs() < 1e-6,
            "ch0: adapter {out0} vs karc {}",
            full[0]
        );

        // set_channel(1) must then match full[1].
        adapter0.set_channel(1);
        let mut out1 = 0.0_f32;
        adapter0.forecast_into(&ds, 1, &mut out1);
        assert!(
            (out1 - full[1]).abs() < 1e-6,
            "ch1: adapter {out1} vs karc {}",
            full[1]
        );
    }

    #[test]
    fn adapter_works_with_observe_and_update() {
        // The adapter's reason to exist: ConformalIntervalCalibrator::
        // observe_and_update forwards the real delay_state, so the adapter
        // drives the write path in one call.
        let (karc, ds, actual) = make_fitted_karc();
        let adapter = KarcChannelForecaster::new(karc, 0);
        let mut cal = ConformalIntervalCalibrator::new(
            adapter,
            1, // n_channels
            1, // max_h
            1, // m
            32, // capacity
            0.0, // exp_lambda
            DecayUnit::Step,
            ResidualMode::HStep,
            false,
        );

        // observe_and_update should forecast via the adapter (real delay_state)
        // and push the residual. No panic.
        cal.observe_and_update(actual, &ds, 0, 1);
        cal.step();
        assert!(cal.tick() == 1);

        // interval_into would pass empty delay_state — that hits the adapter's
        // debug-assert. Use interval_from_point_into instead (the documented
        // KARC pattern).
        let mut point = 0.0_f32;
        cal.forecaster.forecast_into(&ds, 1, &mut point);
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
        cal.interval_from_point_into(point, 0, 1, 0.05, &mut iv);
        assert!((iv.point - point).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn channel_out_of_range_panics() {
        let (karc, _, _) = make_fitted_karc();
        // D=2, so channel=2 is out of range.
        let _ = KarcChannelForecaster::new(karc, 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "KARC requires delay_state")]
    fn empty_delay_state_panics_in_debug() {
        // Documents the "don't call interval_into with KARC" contract:
        // interval_into passes &[] which violates the K*D length assert.
        // Only runs in debug builds (release builds skip debug_assert_eq!).
        let (karc, _, _) = make_fitted_karc();
        let mut adapter = KarcChannelForecaster::new(karc, 0);
        let mut out = 0.0_f32;
        adapter.forecast_into(&[], 1, &mut out);
    }
}
