//! Time-series forecast head (Plan 628 T0.4).
//!
//! The Forecast class answered modellessly: a [`SeasonalPoolForecaster`]
//! (same-phase history, exponentially recency-weighted — the KARC special
//! case with no ridge solve) produces the point path; the Plan-340 conformal
//! overlay in [`ResidualMode::HStep`] turns the observed residuals into a
//! per-step p10/median/p90 interval over the declared horizon. Conditioning
//! does not share across the two computed classes: Forecast is time-series,
//! Estimate is cross-sectional ([`crate::estimate`]) — Proposal 016 §3.3
//! (§3.2's forecast row).
//!
//! # Observation law (single-step feedback)
//!
//! [`ForecastHead::observe`] feeds the residual at **h = 1** only: the head
//! learns from realized one-step-ahead error, exactly like the floor. Under
//! HStep bucketing, horizons `h ≤ m` all read bucket 0 — so interval
//! calibration is SOUND for `h ≤ m`. Horizons beyond `m` map to deeper
//! buckets that single-step feedback never fills; there the interval
//! honestly collapses toward the point (quantiles 0) until the observation
//! protocol is extended (a rolling-origin backtest — T0.6 scope, disclosed).
//!
//! # Floor law (Report the Floor, Research 322 / Plan 340)
//!
//! The honest baseline: a well-tuned head TiesFloor or BeatsFloor on
//! stationary seasonal data (the floor IS seasonal-naive; the same-phase
//! POOL averages many noisy samples of one phase where the floor anchors on
//! one). The G1 plumbing ([`ForecastHarness`]) runs the head through
//! `katgpt_core::conformal::run_floor_comparison`; the shipped posture is
//! documented, the gate EVALUATES at T0.6, and whatever measures is what
//! gets reported — no faked win.
//!
//! # Zero-alloc
//!
//! [`ForecastHead::observe`] and the per-step forecast loop are
//! allocation-free (scratch is owned by the substrate). The one allocation
//! class is the wire shape itself: `ForecastDistribution.steps` is a
//! caller-owned `Vec` that the head overwrites in place, resizing only when
//! the requested horizon changes (the documented cold path).

use katgpt_core::conformal::{
    ConformalIntervalCalibrator, DecayUnit, PredictiveInterval, PredictiveOutput, ResidualMode,
    SeasonalPoolForecaster, UqPrimitiveUnderTest,
};
use katgpt_core::extraction_wire::{ForecastDistribution, ForecastStep};

/// Pool mixture weight: `1.0` = PURE same-phase average (no seasonal-naive
/// anchor). The shipped posture for stationary seasonality — the pool
/// averages every noisy sample of the phase where the floor's single anchor
/// carries the full observation noise. Regime tracking (`pool_weight < 1`,
/// recency `exp_lambda > 0`) is T0.6's tuning axis, measured before any
/// promotion.
pub const FORECAST_POOL_WEIGHT: f32 = 1.0;
/// Recency decay for BOTH the pool and the residual weights: `0.0` disables
/// (all same-phase samples equal weight) — the stationary-series posture and
/// the floor-fair configuration.
pub const FORECAST_EXP_LAMBDA: f32 = 0.0;
/// Residual-ring capacity per (channel, horizon bucket) — the CSP default.
pub const FORECAST_RESIDUAL_CAPACITY: usize = 256;

/// The time-series forecast head: seasonal-pool conditioner + conformal
/// overlay in HStep residual mode, single channel.
pub struct ForecastHead {
    calibrator: ConformalIntervalCalibrator<SeasonalPoolForecaster>,
    max_h: usize,
    alpha: f32,
}

/// Sanitize a miscoverage level: non-finite falls back to 0.05, anything
/// outside [0, 0.5] clamps (the two-tailed conformal substrate's range).
pub(crate) fn sanitize_alpha(alpha: f32) -> f32 {
    if alpha.is_finite() {
        alpha.clamp(0.0, 0.5)
    } else {
        0.05
    }
}

/// Sanitize an interval LEVEL (`1 − alpha`): non-finite falls back to 0.8,
/// values below 0.5 render at 0.5 — the two-tailed substrate cannot express
/// a narrower-than-50% interval, so the declared level is served at the
/// widest honest value and `out.level` reports what was actually rendered.
pub(crate) fn sanitize_level(level: f32) -> f32 {
    if level.is_finite() {
        level.clamp(0.5, 1.0)
    } else {
        0.8
    }
}

/// Wire-shape accommodation (disclosed, shared with the estimate head): the
/// wire requires `p10 ≤ median ≤ p90` with `median == point`, but a
/// one-signed residual pool can put the true conformal interval entirely off
/// the point forecast. The interval is trimmed to include the point. A head
/// that needs the trim habitually is miscalibrated — the G1 floor comparison
/// is the instrument that catches it.
pub(crate) fn order_guarded(lo: f64, point: f64, hi: f64) -> (f64, f64) {
    (lo.min(point), hi.max(point))
}

impl ForecastHead {
    /// A head for seasonal period `m` (clamped to ≥ 1) answering up to
    /// `max_h` steps (clamped to ≥ 1; deeper horizons clamp to the last
    /// residual bucket per the substrate) at miscoverage `alpha` (the
    /// interval level is `1 − alpha`; sanitized like the estimate head).
    ///
    /// The wrapped conditioner is a [`SeasonalPoolForecaster`] at
    /// [`FORECAST_POOL_WEIGHT`] / [`FORECAST_EXP_LAMBDA`] (documented
    /// postures — [`SeasonalNaiveForecaster`] is this type at
    /// `pool_weight = 0`). History capacity `4·(m·⌈max_h/m⌉)` (≥ 8): the
    /// caller's `max_h` choice bounds the memory.
    pub fn new(m: usize, max_h: usize, alpha: f32) -> Self {
        let m = m.max(1);
        let max_h = max_h.max(1);
        let alpha = sanitize_alpha(alpha);
        let lag_max = m * max_h.div_ceil(m);
        let history_capacity = (4 * lag_max).max(8);
        let forecaster = SeasonalPoolForecaster::new(
            history_capacity,
            m,
            FORECAST_EXP_LAMBDA,
            FORECAST_POOL_WEIGHT,
        );
        let calibrator = ConformalIntervalCalibrator::new(
            forecaster,
            1, // single channel (the seasonal family is per-channel by contract)
            max_h,
            m,
            FORECAST_RESIDUAL_CAPACITY,
            FORECAST_EXP_LAMBDA,
            DecayUnit::Step,
            ResidualMode::HStep,
            false,
        );
        Self {
            calibrator,
            max_h,
            alpha,
        }
    }

    /// Observe the realized value `y` for the step that was just forecast.
    ///
    /// The single-step feedback law: the residual rides `h = 1` (bucket 0
    /// under HStep indexing — sound for every queried horizon `h ≤ m`, see
    /// the module docs). The wrapped pool's history advances, then the
    /// monotonic tick steps (a no-op at [`FORECAST_EXP_LAMBDA`] = 0, kept
    /// for lifecycle symmetry). `delay_state` is forwarded to the substrate
    /// (ignored by the seasonal family, honored by future KARC-shaped
    /// conditioners).
    pub fn observe(&mut self, y: f32, delay_state: &[f32]) {
        self.calibrator.observe_and_update(y, delay_state, 0, 1);
        self.calibrator.forecaster.observe(y);
        self.calibrator.step();
    }

    /// Forecast `horizon` steps into `out` at interval mass `level`.
    ///
    /// Per step `h = 1..=horizon` (contiguous, strictly increasing — the
    /// wire's forecast shape): `point` from the wrapped pool, `p10`/`p90`
    /// the conformal quantiles at `1 − alpha`, `median == point`.
    /// `out.level` reports the RENDERED level (a declared level below 0.5
    /// renders at 0.5 — see [`sanitize_level`]).
    ///
    /// `history_last` is the caller's most recent observation: the cold-start
    /// anchor when the head has no history yet (every step then renders
    /// `history_last` as a collapsed distribution — honest, wire-valid).
    /// `out.steps` is overwritten in place; it is reallocated only when its
    /// length differs from `horizon` (the documented cold path).
    pub fn forecast_into(
        &mut self,
        history_last: f32,
        horizon: usize,
        level: f32,
        out: &mut ForecastDistribution,
    ) {
        let horizon = horizon.max(1);
        let level = sanitize_level(level);
        let alpha = 1.0 - level;
        if out.steps.len() != horizon {
            out.steps.clear();
            out.steps.resize(
                horizon,
                ForecastStep {
                    h: 0,
                    point: 0.0,
                    p10: 0.0,
                    median: 0.0,
                    p90: 0.0,
                },
            );
        }
        out.level = level;

        if self.calibrator.forecaster.is_empty() {
            let v = f64::from(history_last);
            for (i, step) in out.steps.iter_mut().enumerate() {
                *step = ForecastStep {
                    h: i as u32 + 1,
                    point: v,
                    p10: v,
                    median: v,
                    p90: v,
                };
            }
            return;
        }

        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, alpha);
        for (i, step) in out.steps.iter_mut().enumerate() {
            let h = i + 1;
            self.calibrator.interval_into(0, h, alpha, &mut iv);
            let point = f64::from(iv.point);
            let (p10, p90) = order_guarded(f64::from(iv.lower), point, f64::from(iv.upper));
            *step = ForecastStep {
                h: h as u32,
                point,
                p10,
                median: point,
                p90,
            };
        }
    }

    /// Observations currently held by the wrapped pool.
    pub fn history_len(&self) -> usize {
        self.calibrator.forecaster.len()
    }

    /// The configured maximum horizon (deeper horizons clamp to the last
    /// residual bucket).
    pub fn max_horizon(&self) -> usize {
        self.max_h
    }

    /// The sanitized miscoverage level (the interval level is `1 − alpha`).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

// ===== G1 floor-comparison plumbing (measurement-only) =====

/// The forecast head adapted to the Report-the-Floor harness
/// ([`UqPrimitiveUnderTest`]): one-step interval from the rolling history,
/// then observe. Zero-alloc — the harness path rides the substrate's own
/// scratch, unlike the estimate harness (whose per-step refit is its
/// documented measurement cost).
pub struct ForecastHarness {
    head: ForecastHead,
}

impl ForecastHarness {
    /// A one-step harness head for seasonal period `m` at miscoverage
    /// `alpha`.
    pub fn new(m: usize, max_h: usize, alpha: f32) -> Self {
        Self {
            head: ForecastHead::new(m, max_h, alpha),
        }
    }

    /// The wrapped head (for post-run introspection in benches).
    pub fn head(&self) -> &ForecastHead {
        &self.head
    }
}

impl UqPrimitiveUnderTest for ForecastHarness {
    fn name(&self) -> &str {
        "katgpt-extract::forecast SeasonalPool + conformal HStep"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, self.head.alpha);
        self.head
            .calibrator
            .interval_into(0, 1, self.head.alpha, &mut iv);
        PredictiveOutput::from_interval(iv)
    }

    fn observe(&mut self, y: f32) {
        self.head.observe(y, &[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::TAU;
    use katgpt_core::conformal::{OverallVerdict, TrajectoryCorpus, run_floor_comparison};

    const EPS: f64 = 1e-9;

    fn deterministic_sine(n: usize, m: usize) -> Vec<f32> {
        (0..n)
            .map(|t| (TAU * (t as f32) / (m as f32)).sin())
            .collect()
    }

    #[test]
    fn steps_are_contiguous_and_quantile_ordered() {
        let mut head = ForecastHead::new(12, 8, 0.05);
        for &y in &deterministic_sine(36, 12) {
            head.observe(y, &[]);
        }
        let mut out = ForecastDistribution {
            steps: Vec::new(),
            level: 0.0,
        };
        head.forecast_into(0.0, 5, 0.8, &mut out);
        assert_eq!(out.steps.len(), 5);
        assert!((out.level - 0.8).abs() < EPS as f32);
        for (i, step) in out.steps.iter().enumerate() {
            assert_eq!(step.h, i as u32 + 1, "h contiguous from 1");
            assert!(
                step.p10 <= step.median && step.median <= step.p90,
                "step h={}: wire ordering violated: {} ≤ {} ≤ {}",
                step.h,
                step.p10,
                step.median,
                step.p90
            );
            assert_eq!(step.median, step.point, "median == point");
        }
        // The sine's next value after t=35 is sin(2π·36/12) = 0.0; a
        // pure same-phase pool over an exact sine reproduces the phase
        // exactly, so the h=1 point sits on it.
        assert!(
            (out.steps[0].point - 0.0).abs() < 1e-4,
            "noise-free seasonal pool should land on the phase, got {}",
            out.steps[0].point
        );
    }

    #[test]
    fn cold_start_falls_back_to_history_last() {
        let mut head = ForecastHead::new(4, 4, 0.05);
        assert_eq!(head.history_len(), 0);
        let mut out = ForecastDistribution {
            steps: Vec::new(),
            level: 0.0,
        };
        head.forecast_into(2.5, 3, 0.8, &mut out);
        assert_eq!(out.steps.len(), 3);
        for (i, step) in out.steps.iter().enumerate() {
            assert_eq!(step.h, i as u32 + 1);
            assert_eq!(step.point, 2.5);
            assert_eq!(step.p10, 2.5);
            assert_eq!(step.p90, 2.5);
        }
    }

    #[test]
    fn single_observation_stays_wire_ordered() {
        // One observation → a one-sample residual pool → both quantiles sit
        // on that residual; the disclosed order-guard keeps the answer
        // wire-valid instead of putting p10 above the median.
        let mut head = ForecastHead::new(4, 4, 0.05);
        head.observe(5.0, &[]);
        let mut out = ForecastDistribution {
            steps: Vec::new(),
            level: 0.0,
        };
        head.forecast_into(5.0, 2, 0.8, &mut out);
        for step in &out.steps {
            assert!(
                step.p10 <= step.median && step.median <= step.p90,
                "one-sample pool must stay ordered: {} ≤ {} ≤ {}",
                step.p10,
                step.median,
                step.p90
            );
            assert!(step.point.is_finite() && step.p10.is_finite() && step.p90.is_finite());
        }
    }

    #[test]
    fn level_below_half_clamps() {
        let mut head = ForecastHead::new(4, 4, 0.05);
        let mut out = ForecastDistribution {
            steps: Vec::new(),
            level: 0.0,
        };
        head.forecast_into(1.0, 1, 0.3, &mut out);
        assert_eq!(out.level, 0.5, "levels below 0.5 render at 0.5");
        head.forecast_into(1.0, 1, f32::NAN, &mut out);
        assert!((out.level - 0.8).abs() < EPS as f32, "non-finite → 0.8");
        head.forecast_into(1.0, 0, 0.8, &mut out);
        assert_eq!(out.steps.len(), 1, "horizon clamps to ≥ 1");
    }

    #[test]
    fn floor_comparison_stationary_seasonal_records_verdict() {
        // The honest baseline: the pool head vs the conformal-naive floor on
        // the canonical stationary-seasonal fixture. The verdict is
        // MEASURED and printed — never asserted as a win (the G1 gate
        // evaluates at T0.6).
        let corpus = TrajectoryCorpus::stationary_seasonal(12, 0.5, 400, 0xC0FF_EE1A);
        let mut harness = ForecastHarness::new(12, 1, 0.05);
        let report = run_floor_comparison(
            &mut harness,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        report.pretty_print();
        assert!(report.n_scored > 0);
        assert!(!report.is_not_applicable());
        assert!(matches!(
            report.overall,
            OverallVerdict::BeatsFloor
                | OverallVerdict::TiesFloor
                | OverallVerdict::LosesToFloor
                | OverallVerdict::Mixed
        ));
    }

    #[test]
    fn floor_comparison_white_noise_records_posture() {
        // Honest record on the degenerate corpus (the floor's worst case):
        // whatever measures, measures — the test pins the PLUMBING.
        let corpus = TrajectoryCorpus::white_noise(0.5, 400, 0xBADC_0FEE);
        let mut harness = ForecastHarness::new(1, 1, 0.05);
        let report = run_floor_comparison(
            &mut harness,
            &corpus.values,
            0.05,
            corpus.recommended_warmup,
            &corpus.name,
        );
        report.pretty_print();
        assert!(report.n_scored > 0);
        assert!(!report.is_not_applicable());
    }

    // ---- G4: allocation gate (crate::alloc_gate; see its module doc) -------

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn forecast_into_sized_out_is_zero_alloc() {
        use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
        use std::sync::atomic::Ordering;

        let mut head = ForecastHead::new(12, 8, 0.05);
        for &y in &deterministic_sine(36, 12) {
            head.observe(y, &[]);
        }
        let mut out = ForecastDistribution {
            steps: Vec::new(),
            level: 0.0,
        };
        // First call sizes `out.steps` to the horizon (the documented cold
        // path — steps are reallocated only on a length change).
        head.forecast_into(0.0, 8, 0.8, &mut out);
        assert_eq!(out.steps.len(), 8);

        // Second call at the SAME horizon: the pinned zero-alloc claim
        // (module doc — the per-step loop writes `out.steps` in place).
        let _window = MeasuringWindow::enter();
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        head.forecast_into(0.0, 8, 0.8, &mut out);
        assert_eq!(
            ALLOC_COUNT.load(Ordering::Relaxed),
            0,
            "forecast_into allocated with out.steps already at horizon length"
        );
        drop(_window);
        assert_eq!(out.steps.len(), 8);
    }
}
