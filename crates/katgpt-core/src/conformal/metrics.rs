//! Conformal UQ metrics — CRPS, Winkler interval score, empirical coverage.
//!
//! These are the GOAT-gate scoring functions for any UQ-bearing primitive per
//! the "Report the Floor" rule. A primitive claiming calibrated uncertainty
//! MUST beat the conformal-naive floor
//! ([`crate::conformal::SeasonalNaiveForecaster`] with `m=1`) on these metrics
//! at its GOAT gate.
//!
//! ## Metric definitions
//!
//! ### CRPS (Continuous Ranked Probability Score)
//!
//! For a predictive sample set `{x_1, ..., x_n}` and an observed `actual`:
//!
//! ```text
//! CRPS = (1/n) · Σ_i |x_i − actual| − (1/(2n²)) · Σ_i Σ_j |x_i − x_j|
//! ```
//!
//! Lower is better. CRPS = 0 for a perfect point forecast. The negative
//! pairwise term rewards spread-out predictions (the "sharpness" penalty is
//! halved to make CRPS proper).
//!
//! For an *interval* (lower, upper) instead of a sample set, we use the
//! interval-CRPS approximation:
//!
//! ```text
//! CRPS_interval = (upper − lower) · (α − 1{actual < lower}) + 2·(lower − actual)·1{actual < lower} + 2·(actual − upper)·1{actual > upper}
//! ```
//!
//! (This is the closed-form CRPS of a uniform distribution on `[lower, upper]`
//! under the interval scoring rule; it reduces to `(upper−lower)` when the
//! actual is inside.)
//!
//! ### Winkler interval score
//!
//! For a central `1−α` interval `[lower, upper]` and an observed `actual`:
//!
//! ```text
//! W = (upper − lower) + (2/α) · (lower − actual) · 1{actual < lower}
//!                          + (2/α) · (actual − upper) · 1{actual > upper}
//! ```
//!
//! Lower is better. The interval width is penalized always; the outside-miss
//! distance is penalized by `2/α` (so a 95% interval missing by 1 unit costs
//! 40, while a 50% interval missing by 1 unit costs 4).
//!
//! ### Empirical coverage
//!
//! Fraction of actuals that fall within `[lower, upper]`. For a `1−α` interval
//! on stationary data, this should converge to `1−α` as `n → ∞`.

use crate::conformal::PredictiveInterval;

/// Compute the interval-CRPS for a single `(interval, actual)` pair.
///
/// Uses the uniform-on-interval closed form. Lower is better.
#[inline]
pub fn crps_interval(interval: &PredictiveInterval, actual: f32) -> f32 {
    let lower = interval.lower;
    let upper = interval.upper;
    let width = upper - lower;
    if actual < lower {
        width + 2.0 * (lower - actual)
    } else if actual > upper {
        width + 2.0 * (actual - upper)
    } else {
        // Inside the interval — CRPS reduces to the width (the "sharpness").
        width
    }
}

/// Compute the sample-set CRPS for `(samples, actual)`.
///
/// ```text
/// CRPS = (1/n) · Σ_i |x_i − actual| − (1/(2n²)) · Σ_i Σ_j |x_i − x_j|
/// ```
///
/// Lower is better. `O(n²)` in the number of samples — use `n ≤ 1000`.
pub fn crps(samples: &[f32], actual: f32) -> f32 {
    let n = samples.len();
    if n == 0 {
        return f32::INFINITY;
    }
    let mut abs_sum = 0.0_f32;
    for &x in samples {
        abs_sum += (x - actual).abs();
    }
    let mean_abs = abs_sum / (n as f32);

    let mut pair_sum = 0.0_f32;
    for i in 0..n {
        for j in 0..n {
            pair_sum += (samples[i] - samples[j]).abs();
        }
    }
    let mean_pair = pair_sum / (2.0 * (n as f32).powi(2));
    mean_abs - mean_pair
}

/// Compute the Winkler interval score for a single `(interval, actual)` pair.
///
/// `interval.alpha` MUST be set to the two-tailed miscoverage level
/// (e.g. `0.05` for a 95% interval). Lower is better.
#[inline]
pub fn winkler_score(interval: &PredictiveInterval, actual: f32) -> f32 {
    let lower = interval.lower;
    let upper = interval.upper;
    let alpha = interval.alpha;
    let width = upper - lower;
    let penalty = if alpha <= 0.0 {
        // Degenerate: no penalty scaling → just the width.
        0.0
    } else {
        2.0 / alpha
    };
    if actual < lower {
        width + penalty * (lower - actual)
    } else if actual > upper {
        width + penalty * (actual - upper)
    } else {
        width
    }
}

/// Compute the empirical coverage of `intervals` against `actuals`.
///
/// Returns the fraction `k/n` where `k` is the number of `actuals[i]` that
/// fall within `intervals[i].lower..=intervals[i].upper`. Both slices must be
/// the same length.
///
/// For a calibrated `1−α` interval, this should converge to `1−α` as `n → ∞`.
pub fn empirical_coverage(intervals: &[PredictiveInterval], actuals: &[f32]) -> f32 {
    debug_assert_eq!(
        intervals.len(),
        actuals.len(),
        "intervals and actuals must be the same length"
    );
    if intervals.is_empty() {
        return 0.0;
    }
    let mut hits = 0usize;
    for (iv, &a) in intervals.iter().zip(actuals.iter()) {
        if iv.contains(a) {
            hits += 1;
        }
    }
    (hits as f32) / (intervals.len() as f32)
}

/// Mean interval-CRPS over a batch.
#[inline]
pub fn mean_crps_interval(intervals: &[PredictiveInterval], actuals: &[f32]) -> f32 {
    debug_assert_eq!(intervals.len(), actuals.len());
    if intervals.is_empty() {
        return f32::INFINITY;
    }
    let mut sum = 0.0_f32;
    for (iv, &a) in intervals.iter().zip(actuals.iter()) {
        sum += crps_interval(iv, a);
    }
    sum / (intervals.len() as f32)
}

/// Mean Winkler score over a batch.
#[inline]
pub fn mean_winkler(intervals: &[PredictiveInterval], actuals: &[f32]) -> f32 {
    debug_assert_eq!(intervals.len(), actuals.len());
    if intervals.is_empty() {
        return f32::INFINITY;
    }
    let mut sum = 0.0_f32;
    for (iv, &a) in intervals.iter().zip(actuals.iter()) {
        sum += winkler_score(iv, a);
    }
    sum / (intervals.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(lower: f32, upper: f32, alpha: f32) -> PredictiveInterval {
        PredictiveInterval::new(lower, 0.5 * (lower + upper), upper, alpha)
    }

    #[test]
    fn crps_inside_reduces_to_width() {
        // Actual inside [10, 20] → CRPS = width = 10.
        let interval = iv(10.0, 20.0, 0.05);
        assert!((crps_interval(&interval, 15.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn crps_outside_adds_distance_penalty() {
        // Actual = 5 (below 10) → CRPS = 10 + 2·(10−5) = 20.
        let interval = iv(10.0, 20.0, 0.05);
        assert!((crps_interval(&interval, 5.0) - 20.0).abs() < 1e-6);
        // Actual = 25 (above 20) → CRPS = 10 + 2·(25−20) = 20.
        assert!((crps_interval(&interval, 25.0) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn crps_sample_set_zero_for_perfect_point() {
        // All samples equal to actual → CRPS = 0.
        let samples = vec![5.0_f32; 10];
        assert!(crps(&samples, 5.0).abs() < 1e-6);
    }

    #[test]
    fn crps_sample_set_positive_for_spread() {
        // Spread-out samples around actual → CRPS > 0 (sharpness penalty).
        let samples = vec![0.0_f32, 10.0];
        let got = crps(&samples, 5.0);
        // mean_abs = (5 + 5)/2 = 5; mean_pair = (|0-0|+|0-10|+|10-0|+|10-10|)/(2·4) = 20/8 = 2.5.
        // CRPS = 5 − 2.5 = 2.5.
        assert!((got - 2.5).abs() < 1e-6, "got {}", got);
    }

    #[test]
    fn winkler_inside_is_just_width() {
        let interval = iv(10.0, 20.0, 0.05);
        // Actual inside → W = width = 10.
        assert!((winkler_score(&interval, 15.0) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn winkler_outside_uses_alpha_scaling() {
        let interval = iv(10.0, 20.0, 0.05);
        // Actual = 5 (below 10), α=0.05 → penalty = 2/0.05 = 40.
        // W = 10 + 40·(10−5) = 10 + 200 = 210.
        assert!((winkler_score(&interval, 5.0) - 210.0).abs() < 1e-4, "got {}", winkler_score(&interval, 5.0));
        // Actual = 25 (above 20) → W = 10 + 40·(25−20) = 210.
        assert!((winkler_score(&interval, 25.0) - 210.0).abs() < 1e-4);
    }

    #[test]
    fn coverage_fraction_is_correct() {
        let intervals = vec![iv(0.0, 10.0, 0.05), iv(0.0, 10.0, 0.05), iv(0.0, 10.0, 0.05)];
        // 5 inside [0,10], 15 outside, 5 inside.
        let actuals = vec![5.0_f32, 15.0, 5.0];
        // 2/3 inside.
        let cov = empirical_coverage(&intervals, &actuals);
        assert!((cov - 2.0 / 3.0).abs() < 1e-6, "got {}", cov);
    }

    #[test]
    fn mean_metrics_handle_empty() {
        assert!(mean_crps_interval(&[], &[]).is_infinite());
        assert!(mean_winkler(&[], &[]).is_infinite());
        assert_eq!(empirical_coverage(&[], &[]), 0.0);
    }
}
