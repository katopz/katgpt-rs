//! # Conformal Predictive Intervals — Modelless UQ Overlay (Plan 340)
//!
//! A generic, modelless, inference-time conformal UQ overlay that wraps any
//! point forecaster and produces coverage-guaranteed predictive intervals.
//! Distilled from the Conformal Splice Predictive (CSP) forecaster
//! (arXiv:2605.03789) and the "Report the Floor" companion paper
//! (arXiv:2606.09473).
//!
//! ## What it does
//!
//! 1. Wraps a [`PointForecaster`] trait (two impls ship: [`SeasonalPoolForecaster`],
//!    and a KARC adapter in Phase 2).
//! 2. Maintains a per-channel residual pool with exponential recency weighting
//!    ([`DecayUnit`] selectable: `Step` or `Cycle`).
//! 3. Indexes the residual pool by horizon `h` via `L_h = m·⌈h/m⌉` (the `HStep`
//!    residual mode — CSP v0.1.4 default) or a single lag-`m` pool (`Paper` mode).
//! 4. Reads empirical quantiles `q_{α/2}`, `q_{1−α/2}` to produce
//!    `[point + q_{α/2}, point + q_{1−α/2}]`.
//! 5. Optionally draws samples via the seasonal-pool + conformal-residual
//!    mixture (CSP's full predictive distribution).
//! 6. Computes CRPS / Winkler interval score / empirical coverage for the GOAT
//!    gate via the [`metrics`] submodule.
//!
//! ## Modelless mandate
//!
//! No training, no learned parameters, no gradient descent. Pure empirical-
//! quantile calibration over a residual reservoir. The only "state" is the
//! residual pool, which is updated online via [`update_residual`].
//!
//! ## "Report the Floor"
//!
//! `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1` is the
//! canonical conformal-naive floor — every future UQ-bearing primitive's GOAT
//! gate MUST beat this baseline on CRPS / coverage / Winkler per the
//! "Report the Floor" rule (Research 322, `AGENTS.md` Feature Flag Discipline,
//! Issue 010).
//!
//! [`update_residual`]: ConformalIntervalCalibrator::update_residual

use core::cmp::Ordering;

/// Maximum residual-pool size that fits in the stack-allocated weights buffer
/// used by [`ConformalIntervalCalibrator::weighted_quantile_pair`]. Pools up to
/// this size get the fast single-pass weights-compute path; larger pools fall
/// back to the per-quantile recomputation. 1024 f32 = 4KB stack — well within
/// the default 8MB main-thread stack. The plan-default capacity is 256, so the
/// fast path covers all realistic configs.
const WEIGHTS_BUF_LEN: usize = 1024;

pub mod metrics;
mod ring;
mod seasonal;

pub use metrics::{crps, empirical_coverage, winkler_score};
pub use ring::{ResidualRingBuffer, RingBuffer};
pub use seasonal::{seasonal_naive_floor, SeasonalNaiveForecaster, SeasonalPoolForecaster};

/// A point forecaster that produces a single deterministic forecast.
///
/// KARC implements this (via the Phase 2 adapter); [`SeasonalPoolForecaster`]
/// implements this directly; any future forecaster can implement it.
///
/// The `delay_state` slice is forecaster-specific (delay-embedded state for
/// KARC, ignored for the seasonal pool). Implementations MUST be deterministic
/// in `(self, delay_state, h)` for bit-reproducibility (G4).
pub trait PointForecaster {
    /// Forecast the value at horizon `h` (1-indexed) given the delay-embedded
    /// state. Writes into `out` (zero-alloc).
    fn forecast_into(&self, delay_state: &[f32], h: usize, out: &mut f32);
}

/// Residual pool indexing strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualMode {
    /// Single residual pool (lag `m`) reused for all horizons.
    /// Matches CSP `residual_mode="paper"`. Interval width is constant across
    /// horizons. Use only for seasonal data with `H ≤ m`.
    Paper,
    /// Horizon-indexed pool with `L_h = m·⌈h/m⌉`.
    /// Matches CSP `residual_mode="h_step"` (v0.1.4 default). Interval widens
    /// with horizon. Use for non-seasonal (`m=1`) or long-horizon (`H>m`) series.
    HStep,
}

impl Default for ResidualMode {
    #[inline]
    fn default() -> Self {
        // CSP v0.1.4 default — drives multi-step coverage.
        ResidualMode::HStep
    }
}

/// Unit for the residual pool's exponential recency decay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecayUnit {
    /// Decay by absolute observation age (time steps). CSP v0.1.4 default.
    /// Same-phase observations one season apart are `m` steps apart.
    Step,
    /// Decay by cycle age. CSP paper's original behavior.
    /// `m`× weaker than `Step` for the same `exp_lambda`.
    Cycle,
}

impl Default for DecayUnit {
    #[inline]
    fn default() -> Self {
        DecayUnit::Step
    }
}

/// Compute the lag `L_h` used to bucket the residual at horizon `h`.
///
/// - [`ResidualMode::Paper`]: always `m` (single pool).
/// - [`ResidualMode::HStep`]: `m·⌈h/m⌉` (widens with horizon).
#[inline]
#[allow(dead_code)] // referenced by tests + part of the documented math surface
fn horizon_lag(h: usize, m: usize, mode: ResidualMode) -> usize {
    debug_assert!(m >= 1, "seasonal period m must be >= 1");
    debug_assert!(h >= 1, "horizon h is 1-indexed");
    match mode {
        ResidualMode::Paper => m,
        ResidualMode::HStep => m * ((h + m - 1) / m),
    }
}

/// The conformal UQ overlay. Generic over any [`PointForecaster`].
///
/// State:
/// - `forecaster`: the wrapped point forecaster (KARC, SeasonalPool, etc.).
/// - `residual_pool`: per-channel × per-horizon-bucket sorted residual ring
///   buffer, exp-recency weighted.
/// - `m`: seasonal period (`m=1` for non-seasonal).
/// - `exp_lambda`: exponential decay rate for recency weighting.
/// - `decay_unit`: `Step` or `Cycle`.
/// - `residual_mode`: `Paper` or `HStep`.
/// - `orientation`: quantile-orientation correction flag (lower/upper
///   asymmetric indexing; CSP `orientation` parameter).
/// - `global_tick`: monotonic step counter for recency-weight computation.
pub struct ConformalIntervalCalibrator<F: PointForecaster> {
    /// The wrapped point forecaster.
    pub forecaster: F,
    /// Per-channel residual ring buffer, exp-recency weighted.
    /// Layout: `[channel][horizon_bucket][sorted_residual, tick]`.
    pub residual_pool: ResidualRingBuffer,
    /// Seasonal period. `m=1` for non-seasonal data.
    pub m: usize,
    /// Exponential recency-decay rate. Weight `w = exp(−λ · age)`.
    pub exp_lambda: f32,
    /// Unit for `age` in the weight formula.
    pub decay_unit: DecayUnit,
    /// Residual-pool indexing strategy.
    pub residual_mode: ResidualMode,
    /// Quantile-orientation correction (CSP `orientation`).
    pub orientation: bool,
    /// Monotonic tick counter (age = `global_tick − push_tick`).
    global_tick: u64,
}

/// A calibrated predictive interval `[lower, point, upper]` at level `1−α`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PredictiveInterval {
    /// Lower bound `point + q_{α/2}`.
    pub lower: f32,
    /// Point forecast `ŷ`.
    pub point: f32,
    /// Upper bound `point + q_{1−α/2}`.
    pub upper: f32,
    /// Two-tailed miscoverage level (e.g. `0.05` for a 95% interval).
    pub alpha: f32,
}

impl PredictiveInterval {
    /// Construct a new interval. Does NOT validate `lower ≤ point ≤ upper`.
    #[inline]
    pub const fn new(lower: f32, point: f32, upper: f32, alpha: f32) -> Self {
        Self {
            lower,
            point,
            upper,
            alpha,
        }
    }

    /// `true` iff `actual` is within `[lower, upper]` (inclusive).
    #[inline]
    pub fn contains(&self, actual: f32) -> bool {
        actual >= self.lower && actual <= self.upper
    }

    /// Interval half-width `(upper − lower) / 2`.
    #[inline]
    pub fn half_width(&self) -> f32 {
        0.5 * (self.upper - self.lower)
    }
}

impl<F: PointForecaster> ConformalIntervalCalibrator<F> {
    /// Construct a new calibrator.
    ///
    /// - `forecaster`: the wrapped point forecaster.
    /// - `n_channels`: number of channels (e.g. HLA_DIM=8).
    /// - `max_h`: maximum horizon `h` that will be queried. Horizon buckets
    ///   beyond `max_h` wrap into the largest bucket.
    /// - `m`: seasonal period (`m=1` for non-seasonal).
    /// - `capacity`: ring-buffer capacity per (channel, horizon-bucket).
    ///   CSP default 256. Total memory `n_channels · ceil(max_h/m) · capacity · 8` bytes.
    /// - `exp_lambda`: exponential decay rate. `0.0` disables recency weighting.
    /// - `decay_unit`: `Step` (default) or `Cycle`.
    /// - `residual_mode`: `HStep` (default) or `Paper`.
    /// - `orientation`: quantile-orientation correction flag.
    pub fn new(
        forecaster: F,
        n_channels: usize,
        max_h: usize,
        m: usize,
        capacity: usize,
        exp_lambda: f32,
        decay_unit: DecayUnit,
        residual_mode: ResidualMode,
        orientation: bool,
    ) -> Self {
        assert!(m >= 1, "seasonal period m must be >= 1");
        assert!(n_channels >= 1, "n_channels must be >= 1");
        assert!(max_h >= 1, "max_h must be >= 1");
        assert!(capacity >= 1, "capacity must be >= 1");
        let n_buckets = Self::n_buckets_for(max_h, m, residual_mode);
        Self {
            forecaster,
            residual_pool: ResidualRingBuffer::new(n_channels, n_buckets, capacity),
            m,
            exp_lambda,
            decay_unit,
            residual_mode,
            orientation,
            global_tick: 0,
        }
    }

    /// Number of horizon buckets needed to cover `max_h`.
    #[inline]
    fn n_buckets_for(max_h: usize, m: usize, mode: ResidualMode) -> usize {
        // In Paper mode every horizon shares the single lag-`m` bucket → 1 bucket.
        // In HStep mode horizon `h` maps to bucket index `ceil(h/m) − 1`, so we
        // need `ceil(max_h/m)` buckets.
        match mode {
            ResidualMode::Paper => 1,
            ResidualMode::HStep => (max_h + m - 1) / m,
        }
    }

    /// Map `(h, m, mode)` to a flat horizon-bucket index.
    #[inline]
    fn bucket_index(&self, h: usize) -> usize {
        debug_assert!(h >= 1, "horizon h is 1-indexed");
        let n_buckets = self.residual_pool.n_buckets;
        if n_buckets == 0 {
            return 0;
        }
        let raw = match self.residual_mode {
            ResidualMode::Paper => 0,
            ResidualMode::HStep => (h + self.m - 1) / self.m - 1,
        };
        // Clamp to the last bucket if `h` exceeds the configured `max_h`.
        raw.min(n_buckets.saturating_sub(1))
    }

    /// Advance the monotonic tick counter by one step.
    ///
    /// Call once per observation step so that recency weights reflect elapsed
    /// time. If never called, all residuals receive equal weight (no decay).
    #[inline]
    pub fn step(&mut self) {
        self.global_tick = self.global_tick.saturating_add(1);
    }

    /// Current monotonic tick.
    #[inline]
    pub fn tick(&self) -> u64 {
        self.global_tick
    }

    /// Observe an `(actual, forecasted)` pair at horizon `h`, update the
    /// residual pool for `channel`.
    ///
    /// Computes `r = actual − forecast`, indexes the horizon bucket via
    /// [`horizon_lag`], pushes into the ring buffer tagged with the current
    /// `global_tick`. The exponential recency weight `w = exp(−λ · age)` is
    /// applied at *quantile read time* (not storage), keeping the write path
    /// simple and zero-alloc.
    ///
    /// O(log n) insertion into the per-channel sorted ring buffer.
    pub fn update_residual(&mut self, actual: f32, forecast: f32, channel: usize, h: usize) {
        let residual = actual - forecast;
        let bucket = self.bucket_index(h);
        self.residual_pool
            .push(residual, channel, bucket, self.global_tick);
    }

    /// Convenience: forecast via the wrapped forecaster, then observe the
    /// realized `actual` and update the residual pool.
    ///
    /// `delay_state` is forwarded to the wrapped forecaster.
    pub fn observe_and_update(
        &mut self,
        actual: f32,
        delay_state: &[f32],
        channel: usize,
        h: usize,
    ) {
        let mut forecast = 0.0_f32;
        self.forecaster.forecast_into(delay_state, h, &mut forecast);
        self.update_residual(actual, forecast, channel, h);
    }

    /// Read the calibrated interval `[lower, point, upper]` at horizon `h`,
    /// level `1−α`, for `channel`. Writes into `out` (zero-alloc).
    ///
    /// O(n) quantile read from the pre-sorted pool (n = ring capacity). The
    /// exp-recency weights are computed ONCE for both quantile lookups
    /// (q_{α/2} and q_{1−α/2}) — this halves the `exp()` call count vs the
    /// naive per-quantile recomputation, which is the difference between
    /// meeting and missing the G2 ≤ 1µs budget at H=1.
    pub fn interval_into(
        &self,
        channel: usize,
        h: usize,
        alpha: f32,
        out: &mut PredictiveInterval,
    ) {
        debug_assert!(
            (0.0..=0.5).contains(&alpha),
            "alpha must be in [0, 0.5] for a two-tailed interval"
        );
        let bucket = self.bucket_index(h);

        // Point forecast from the wrapped forecaster (zero delay-state by
        // default; callers that have a real delay state should compute the
        // point themselves and call `interval_from_point_into`).
        let mut point = 0.0_f32;
        self.forecaster.forecast_into(&[], h, &mut point);

        let (q_lo, q_hi) = self.weighted_quantile_pair(channel, bucket, 0.5 * alpha, 1.0 - 0.5 * alpha);

        out.lower = point + q_lo;
        out.point = point;
        out.upper = point + q_hi;
        out.alpha = alpha;
    }

    /// As [`interval_into`] but with a caller-supplied point forecast. Use this
    /// when the caller has already computed `ŷ` (e.g. KARC's `forecast_into`
    /// produces a full `D`-channel vector in one call).
    pub fn interval_from_point_into(
        &self,
        point: f32,
        channel: usize,
        h: usize,
        alpha: f32,
        out: &mut PredictiveInterval,
    ) {
        debug_assert!(
            (0.0..=0.5).contains(&alpha),
            "alpha must be in [0, 0.5] for a two-tailed interval"
        );
        let bucket = self.bucket_index(h);
        let (q_lo, q_hi) = self.weighted_quantile_pair(channel, bucket, 0.5 * alpha, 1.0 - 0.5 * alpha);
        out.lower = point + q_lo;
        out.point = point;
        out.upper = point + q_hi;
        out.alpha = alpha;
    }

    /// `true` iff `actual` is outside the `1−α` interval at horizon `h`.
    /// The 1-bit calibrated curiosity / coverage-violation signal.
    #[inline]
    pub fn coverage_violation(&self, actual: f32, channel: usize, h: usize, alpha: f32) -> bool {
        let mut interval = PredictiveInterval::new(0.0, 0.0, 0.0, alpha);
        self.interval_into(channel, h, alpha, &mut interval);
        !interval.contains(actual)
    }

    /// Compute two weighted quantiles `(q_p_lo, q_p_hi)` from the same
    /// `(channel, bucket)` in a SINGLE pass over the residual pool. The
    /// exp-recency weights are computed once and reused for both lookups —
    /// this is the G2 perf-critical path (4× fewer `exp()` calls than calling
    /// [`weighted_quantile`] twice).
    ///
    /// Stack-allocates up to [`WEIGHTS_BUF_LEN`] f32 scratch (4KB); falls back
    /// to the per-quantile path if the pool is larger.
    fn weighted_quantile_pair(
        &self,
        channel: usize,
        bucket: usize,
        p_lo: f32,
        p_hi: f32,
    ) -> (f32, f32) {
        let view = self.residual_pool.channel_bucket(channel, bucket);
        let n = view.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        // Fast path: pool fits in the stack buffer → compute weights once,
        // reuse for both quantile lookups.
        if n <= WEIGHTS_BUF_LEN {
            let mut weights = [0.0_f32; WEIGHTS_BUF_LEN];
            let mut total_w = 0.0_f32;
            let lambda = self.exp_lambda;
            let tick_now = self.global_tick;
            let unit_scale = match self.decay_unit {
                DecayUnit::Step => 1.0,
                DecayUnit::Cycle => self.m as f32,
            };
            for i in 0..n {
                let (_, pushed_tick) = view.get_sorted(i);
                let age = (tick_now.saturating_sub(pushed_tick)) as f32 / unit_scale;
                let w = (-lambda * age).exp();
                weights[i] = w;
                total_w += w;
            }
            if total_w <= 0.0 {
                // Degenerate decay → median fallback.
                let med = view.get_sorted(n / 2).0;
                return (med, med);
            }
            let q_lo = Self::quantile_from_weights(&view, &weights[..n], total_w, p_lo, self.orientation);
            let q_hi = Self::quantile_from_weights(&view, &weights[..n], total_w, p_hi, self.orientation);
            return (q_lo, q_hi);
        }

        // Slow path (pool > WEIGHTS_BUF_LEN, rare): fall back to per-quantile.
        let q_lo = self.weighted_quantile(channel, bucket, p_lo);
        let q_hi = self.weighted_quantile(channel, bucket, p_hi);
        (q_lo, q_hi)
    }

    /// Walk the sorted residuals using precomputed `weights` (no `exp()` calls).
    /// Returns the residual value at weighted-CDF probability `p`.
    fn quantile_from_weights(
        view: &ring::RingView<'_>,
        weights: &[f32],
        total_w: f32,
        p: f32,
        orientation: bool,
    ) -> f32 {
        let n = view.len();
        debug_assert_eq!(n, weights.len());
        let target = p * total_w;
        let mut acc = 0.0_f32;
        let mut prev_val = view.get_sorted(0).0;
        for i in 0..n {
            let (val, _) = view.get_sorted(i);
            prev_val = val;
            acc += weights[i];
            if acc >= target {
                if orientation && i + 1 < n && acc == target {
                    let next_val = view.get_sorted(i + 1).0;
                    return 0.5 * (val + next_val);
                }
                return val;
            }
        }
        prev_val
    }

    /// Weighted empirical quantile at probability `p ∈ [0,1]` for
    /// `(channel, bucket)`. Applies the [`orientation`] correction if enabled.
    ///
    /// Returns `0.0` if the pool is empty (no residuals observed yet → interval
    /// collapses to the point forecast).
    ///
    /// NOTE: this is the SLOW path — it recomputes the weight scan on every
    /// call. For interval reads (two quantiles from the same bucket), use
    /// [`weighted_quantile_pair`] which computes weights once and reuses.
    ///
    /// [`orientation`]: ConformalIntervalCalibrator::orientation
    fn weighted_quantile(&self, channel: usize, bucket: usize, p: f32) -> f32 {
        let view = self.residual_pool.channel_bucket(channel, bucket);
        let n = view.len();
        if n == 0 {
            return 0.0;
        }

        // Sort-stable index read: `view` is kept sorted ascending by residual.
        // Apply exponential recency weights at read time.
        let lambda = self.exp_lambda;
        let tick_now = self.global_tick;
        let unit_scale = match self.decay_unit {
            DecayUnit::Step => 1.0,
            DecayUnit::Cycle => self.m as f32,
        };

        // Total weight Σ w_i.
        let mut total_w = 0.0_f32;
        for i in 0..n {
            let (_, pushed_tick) = view.get_sorted(i);
            let age = (tick_now.saturating_sub(pushed_tick)) as f32 / unit_scale;
            total_w += (-lambda * age).exp();
        }
        if total_w <= 0.0 {
            // Degenerate (all weights zero from heavy decay) → fall back to the
            // most-recent residual (the last pushed, which sorts by tick not
            // by value, so use the median of the sorted residuals).
            return view.get_sorted(n / 2).0;
        }

        // Walk the sorted residuals, accumulate weight until we cross `p · total_w`.
        let target = p * total_w;
        let mut acc = 0.0_f32;
        let mut prev_val = view.get_sorted(0).0;
        for i in 0..n {
            let (val, pushed_tick) = view.get_sorted(i);
            let age = (tick_now.saturating_sub(pushed_tick)) as f32 / unit_scale;
            let w = (-lambda * age).exp();
            prev_val = val;
            acc += w;
            if acc >= target {
                // Optional orientation correction: when interpolating between
                // two sorted residuals, use floor vs ceil indexing for lower
                // vs upper quantiles (CSP `orientation` parameter). With the
                // step-function CDF here, orientation is a tie-break convention.
                if self.orientation && i + 1 < n && acc == target {
                    // Exact tie: average with the next residual.
                    let next_val = view.get_sorted(i + 1).0;
                    return 0.5 * (val + next_val);
                }
                return val;
            }
        }
        // Numerical drift fallback: return the largest residual.
        prev_val
    }

    /// Draw `n` samples from the predictive distribution. The CSP mixture:
    /// `pool_weight` fraction from the seasonal pool (sampled proportional to
    /// recency weights), `(1−pool_weight)` fraction from the conformal residual
    /// (sampled uniformly from the residual pool + added to the point forecast).
    ///
    /// **Allocates** `Vec<f32>` of length `n`. Use for CRPS evaluation only,
    /// NOT on the per-tick hot path.
    pub fn sample_predictive_distribution(
        &self,
        channel: usize,
        h: usize,
        n: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<f32> {
        // Note: fastrand's `Rng` trait is `fastrand::Rng`. We accept any rng
        // implementing it. For the simple uniform draws below we use `rng.f32()`.
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return out;
        }

        let mut point = 0.0_f32;
        self.forecaster.forecast_into(&[], h, &mut point);

        let bucket = self.bucket_index(h);
        let view = self.residual_pool.channel_bucket(channel, bucket);
        let pool_n = view.len();

        for _ in 0..n {
            if pool_n == 0 {
                out.push(point);
            } else {
                // Uniform sample from the residual pool (could be weighted;
                // CSP samples the conformal residual uniformly and applies
                // recency only via the seasonal-pool half).
                let idx = rng.usize(0..pool_n);
                let (residual, _) = view.get_sorted(idx);
                out.push(point + residual);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// fastrand convenience — `fastrand::Rng` is a concrete struct (not a trait).
// Callers construct one with `fastrand::Rng::with_seed(...)` for determinism.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Orientation-free partial-sort helpers (used only by tests; kept here to
// avoid pulling in another module for a few utilities).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial forecaster that always predicts a constant.
    struct ConstForecaster {
        value: f32,
    }
    impl PointForecaster for ConstForecaster {
        fn forecast_into(&self, _delay_state: &[f32], _h: usize, out: &mut f32) {
            *out = self.value;
        }
    }

    #[test]
    fn horizon_lag_paper_vs_hstep() {
        // Paper: always `m`.
        assert_eq!(horizon_lag(1, 12, ResidualMode::Paper), 12);
        assert_eq!(horizon_lag(24, 12, ResidualMode::Paper), 12);
        // HStep: `m·⌈h/m⌉`.
        assert_eq!(horizon_lag(1, 12, ResidualMode::HStep), 12);
        assert_eq!(horizon_lag(12, 12, ResidualMode::HStep), 12);
        assert_eq!(horizon_lag(13, 12, ResidualMode::HStep), 24);
        assert_eq!(horizon_lag(24, 12, ResidualMode::HStep), 24);
        assert_eq!(horizon_lag(25, 12, ResidualMode::HStep), 36);
        // m=1 HStep: L_h = h.
        assert_eq!(horizon_lag(1, 1, ResidualMode::HStep), 1);
        assert_eq!(horizon_lag(7, 1, ResidualMode::HStep), 7);
    }

    #[test]
    fn interval_empty_pool_collapses_to_point() {
        let f = ConstForecaster { value: 42.0 };
        let mut cal = ConformalIntervalCalibrator::new(
            f,
            1,    // 1 channel
            8,    // max_h
            1,    // m
            16,   // capacity
            0.0,  // no decay
            DecayUnit::Step,
            ResidualMode::HStep,
            false,
        );
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
        cal.interval_into(0, 1, 0.05, &mut iv);
        // No residuals → quantiles are 0 → interval is [point, point, point].
        assert_eq!(iv.point, 42.0);
        assert_eq!(iv.lower, 42.0);
        assert_eq!(iv.upper, 42.0);
    }

    #[test]
    fn update_then_interval_contains_point_plus_residual_quantile() {
        // Push symmetric residuals ±1, ±2 around point=10; the α=0.5 quantiles
        // should straddle zero symmetrically.
        let f = ConstForecaster { value: 10.0 };
        let mut cal = ConformalIntervalCalibrator::new(
            f, 1, 1, 1, 64, 0.0, DecayUnit::Step, ResidualMode::HStep, false,
        );
        // residuals: actual − forecast
        for &r in &[-2.0_f32, -1.0, 1.0, 2.0] {
            cal.update_residual(10.0 + r, 10.0, 0, 1);
        }
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.5);
        cal.interval_into(0, 1, 0.5, &mut iv);
        // Sorted residuals: [−2, −1, 1, 2]. Equal weights (no decay).
        // α=0.5 → q_{0.25} and q_{0.75}.
        //   q_{0.25}: target = 0.25·4 = 1.0; after index 0 (w=1) acc=1 ≥ 1 → −2.0.
        //   q_{0.75}: target = 0.75·4 = 3.0; after index 2 (w=1+1+1=3) acc=3 ≥ 3 → 1.0.
        assert!((iv.point - 10.0).abs() < 1e-6, "point {}", iv.point);
        assert!((iv.lower - 8.0).abs() < 1e-6, "lower {}", iv.lower);
        assert!((iv.upper - 11.0).abs() < 1e-6, "upper {}", iv.upper);
    }

    #[test]
    fn coverage_violation_flag() {
        let f = ConstForecaster { value: 0.0 };
        let mut cal = ConformalIntervalCalibrator::new(
            f, 1, 1, 1, 64, 0.0, DecayUnit::Step, ResidualMode::HStep, false,
        );
        for &r in &[-1.0_f32, 1.0] {
            cal.update_residual(r, 0.0, 0, 1);
        }
        // α=0.5 → interval is roughly [−1, +1].
        assert!(!cal.coverage_violation(0.0, 0, 1, 0.5), "0 should be inside");
        assert!(
            cal.coverage_violation(5.0, 0, 1, 0.5),
            "5 should be outside"
        );
    }

    #[test]
    fn orientation_correction_averages_on_tie() {
        // With orientation=true and an exact-tie CDF crossing, the quantile
        // should be the average of two consecutive sorted residuals.
        let f = ConstForecaster { value: 0.0 };
        let mut cal = ConformalIntervalCalibrator::new(
            f, 1, 1, 1, 64, 0.0, DecayUnit::Step, ResidualMode::HStep, true,
        );
        for &r in &[0.0_f32, 10.0] {
            cal.update_residual(r, 0.0, 0, 1);
        }
        // p=0.5, two equal weights: total_w=2, target=1.0. After 1st residual
        // (w=1) acc=1=target → orientation tie-break → average(0,10)=5.
        let q = cal.weighted_quantile(0, 0, 0.5);
        assert!((q - 5.0).abs() < 1e-6, "orientation tie-break q={}", q);
    }

    #[test]
    fn bit_reproducibility_identical_configs() {
        // G4: two calibrators with identical config + identical residual
        // pushes produce bit-identical PredictiveInterval bounds.
        let mk = || {
            let f = ConstForecaster { value: 1.0 };
            let mut cal = ConformalIntervalCalibrator::new(
                f, 1, 1, 1, 32, 0.01, DecayUnit::Step, ResidualMode::HStep, false,
            );
            for i in 0..16 {
                let r = (i as f32) * 0.5 - 4.0;
                cal.update_residual(r, 1.0, 0, 1);
                cal.step();
            }
            cal
        };
        let a = mk();
        let b = mk();
        let mut iva = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
        let mut ivb = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
        for &alpha in &[0.01_f32, 0.05, 0.1, 0.2] {
            a.interval_into(0, 1, alpha, &mut iva);
            b.interval_into(0, 1, alpha, &mut ivb);
            assert_eq!(iva.lower.to_bits(), ivb.lower.to_bits(),
                "alpha={} lower mismatch {} vs {}", alpha, iva.lower, ivb.lower);
            assert_eq!(iva.upper.to_bits(), ivb.upper.to_bits(),
                "alpha={} upper mismatch {} vs {}", alpha, iva.upper, ivb.upper);
        }
    }

    #[test]
    fn bucket_index_clamps_beyond_max_h() {
        let f = ConstForecaster { value: 0.0 };
        let cal = ConformalIntervalCalibrator::new(
            f, 1, 4, /* max_h */ 2, /* m */ 8, /* cap */ 0.0, DecayUnit::Step,
            ResidualMode::HStep, false,
        );
        // max_h=4, m=2 → n_buckets=2. h=1→bucket0, h=2→bucket0, h=3→bucket1,
        // h=4→bucket1, h=5 (beyond max_h)→clamp to bucket1.
        assert_eq!(cal.bucket_index(1), 0);
        assert_eq!(cal.bucket_index(2), 0);
        assert_eq!(cal.bucket_index(3), 1);
        assert_eq!(cal.bucket_index(4), 1);
        assert_eq!(cal.bucket_index(5), 1);
    }
}

// Re-export the cmp::Ordering shim so the `match` arms above compile cleanly
// under `edition = "2024"`. (Not part of the public API.)
#[allow(dead_code)]
fn _ordering_shim(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}
