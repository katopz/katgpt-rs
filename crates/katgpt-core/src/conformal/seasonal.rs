//! CSP seasonal-pool forecaster — a standalone [`PointForecaster`] that does
//! pure seasonal-naive mixing with exponential recency. No learned `Wout`, no
//! ridge solve.

use super::PointForecaster;
use crate::conformal::ring::RingBuffer;

/// CSP's seasonal pool forecaster: same-phase history weighted by exponential
/// recency. No learned `Wout`, no ridge solve. Pure reservoir mixing.
///
/// This is a SPECIAL CASE of KARC (periodic delay-basis, no basis expansion,
/// no ridge). Use when:
/// (a) seasonality `m` is known,
/// (b) latency budget is tight (no ridge solve),
/// (c) the series is stationary around a stable level + seasonal pattern.
///
/// Prefer KARC otherwise.
///
/// The forecaster maintains a single-channel history ring (the caller is
/// responsible for managing per-channel forecasters if multi-channel coverage
/// is needed — each channel gets its own `SeasonalPoolForecaster`).
pub struct SeasonalPoolForecaster {
    /// History window. The caller pushes observations via [`observe`].
    ///
    /// [`observe`]: SeasonalPoolForecaster::observe
    history: RingBuffer,
    /// Seasonal period. `m=1` for non-seasonal data (forecast = last value).
    pub m: usize,
    /// Exponential recency-decay rate for the same-phase weighted average.
    /// `0.0` disables recency (simple same-phase average).
    pub exp_lambda: f32,
    /// Mixture weight `[0,1]`: `pool_weight` fraction from the same-phase
    /// average, `(1−pool_weight)` from the most-recent observation (the
    /// seasonal-naive anchor). `pool_weight = 1.0` is pure pool; `0.0` is pure
    /// seasonal-naive.
    pub pool_weight: f32,
}

impl SeasonalPoolForecaster {
    /// Construct a new seasonal-pool forecaster.
    ///
    /// - `capacity`: history ring capacity (must be `≥ m` to hold at least one
    ///   full season; recommend `≥ 4·m` for stable recency weighting).
    /// - `m`: seasonal period (`m=1` for non-seasonal).
    /// - `exp_lambda`: recency decay. `0.0` disables.
    /// - `pool_weight`: pool/anchor mixture weight in `[0,1]`.
    pub fn new(capacity: usize, m: usize, exp_lambda: f32, pool_weight: f32) -> Self {
        assert!(m >= 1, "seasonal period m must be >= 1");
        assert!(capacity >= m, "capacity {} must be >= m {}", capacity, m);
        assert!(
            (0.0..=1.0).contains(&pool_weight),
            "pool_weight must be in [0,1]"
        );
        Self {
            history: RingBuffer::with_capacity(capacity),
            m,
            exp_lambda,
            pool_weight,
        }
    }

    /// Observe a new value `y_t`, pushing it into the history ring.
    #[inline]
    pub fn observe(&mut self, y: f32) {
        self.history.push(y);
    }

    /// Current number of stored history entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Forecast the value at horizon `h` (1-indexed). The forecast is:
    ///
    /// ```text
    /// ŷ_{t+h} = pool_weight · same_phase_avg(L_h) + (1 − pool_weight) · y_{t+1−L_h}
    /// ```
    ///
    /// where `L_h = m·⌈h/m⌉` is the seasonal lag and `same_phase_avg(L_h)` is
    /// the exponentially-recency-weighted average of all history entries at
    /// offset `k·L_h` from the newest.
    ///
    /// If the history is shorter than `L_h`, the forecast falls back to the
    /// most recent observation (or `0.0` if history is empty).
    pub fn forecast(&self, h: usize) -> f32 {
        let n = self.history.len();
        if n == 0 {
            return 0.0;
        }
        let lag = self.m * ((h + self.m - 1) / self.m); // L_h = m·⌈h/m⌉
        if lag > n {
            // Not enough history — fall back to the most recent observation.
            return self.history.back(0).unwrap_or(0.0);
        }
        // Seasonal-naive anchor: y_{t+1−L_h} = the observation at offset
        // (lag − 1) back from the newest.
        let anchor = self.history.back(lag - 1).unwrap_or(0.0);
        if self.pool_weight <= 0.0 {
            return anchor;
        }

        // Same-phase average: walk back in steps of `lag`, accumulate
        // exp-recency-weighted sum.
        let mut wsum = 0.0_f32;
        let mut wtotal = 0.0_f32;
        let mut step = 0usize;
        loop {
            let back = lag.saturating_add(step * lag).saturating_sub(1);
            if back >= n {
                break;
            }
            let y = match self.history.back(back) {
                Some(v) => v,
                None => break,
            };
            let age = step as f32;
            let w = (-self.exp_lambda * age).exp();
            wsum += w * y;
            wtotal += w;
            step += 1;
            // Cap the walk to avoid pathological loops when `lag` is small.
            if step > n {
                break;
            }
        }
        let pool = if wtotal > 0.0 {
            wsum / wtotal
        } else {
            anchor
        };
        self.pool_weight * pool + (1.0 - self.pool_weight) * anchor
    }
}

impl PointForecaster for SeasonalPoolForecaster {
    /// Forecast the value at horizon `h`. The `delay_state` slice is IGNORED
    /// (the seasonal pool uses its own internal history ring as the state).
    #[inline]
    fn forecast_into(&mut self, _delay_state: &[f32], h: usize, out: &mut f32) {
        *out = self.forecast(h);
    }
}

/// Seasonal-naive forecaster — the canonical "Report the Floor" point
/// forecaster. Pure `ŷ_{t+h} = y_{t+1−L_h}` with `L_h = m·⌈h/m⌉`.
///
/// This is `SeasonalPoolForecaster` with `pool_weight = 0.0` (pure anchor).
/// The `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` with `m=1` is
/// the canonical conformal-naive floor per the "Report the Floor" rule.
pub type SeasonalNaiveForecaster = SeasonalPoolForecaster;

/// Construct the canonical conformal-naive floor forecaster.
///
/// `m=1` (non-seasonal), pure seasonal-naive (forecast = last observation),
/// capacity sized to hold the residual pool's horizon-bucket budget.
#[inline]
pub fn seasonal_naive_floor(capacity: usize) -> SeasonalNaiveForecaster {
    SeasonalPoolForecaster::new(capacity.max(1), 1, 0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seasonal_naive_forecast_is_last_value_at_h1() {
        let mut f = SeasonalNaiveForecaster::new(8, 1, 0.0, 0.0);
        f.observe(1.0);
        f.observe(2.0);
        f.observe(3.0);
        // h=1, m=1, L_h=1. Anchor = history.back(0) = 3.0.
        assert_eq!(f.forecast(1), 3.0);
        // h=2, L_h=2. Anchor = history.back(1) = 2.0.
        assert_eq!(f.forecast(2), 2.0);
    }

    #[test]
    fn seasonal_pool_m12_forecast_picks_same_phase() {
        // m=12: forecast at h=1 should anchor on the most recent same-phase
        // observation (history.back(11) for the value 12 steps ago).
        let mut f = SeasonalPoolForecaster::new(24, 12, 0.0, 0.0);
        // Push 13 values; the 1st is the "same phase as next" value.
        for v in 0..13 {
            f.observe(v as f32);
        }
        // h=1, m=12, L_h=12. anchor = history.back(11) = value at index 1 = 1.0.
        assert_eq!(f.forecast(1), 1.0);
    }

    #[test]
    fn pool_weight_averages_when_set() {
        // pool_weight=1.0, no decay → pool = average of all same-phase entries.
        let mut f = SeasonalPoolForecaster::new(8, 1, 0.0, 1.0);
        for &v in &[10.0_f32, 20.0, 30.0, 40.0] {
            f.observe(v);
        }
        // h=1, L_h=1. Same-phase entries: back(0)=40, back(1)=30, back(2)=20,
        // back(3)=10. Average = 25.0 (equal weights since exp_lambda=0).
        let got = f.forecast(1);
        assert!(
            (got - 25.0).abs() < 1e-5,
            "pool avg expected 25.0, got {}",
            got
        );
    }

    #[test]
    fn empty_forecaster_returns_zero() {
        let f = SeasonalNaiveForecaster::new(4, 1, 0.0, 0.0);
        assert_eq!(f.forecast(1), 0.0);
    }

    #[test]
    fn floor_constructor_is_non_seasonal() {
        let f = seasonal_naive_floor(8);
        assert_eq!(f.m, 1);
        assert_eq!(f.pool_weight, 0.0);
    }
}
