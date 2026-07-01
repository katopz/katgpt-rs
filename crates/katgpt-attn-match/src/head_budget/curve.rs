//! `HeadSensitivityCurve` — per-head quality-vs-ratio curve (Plan 271 Phase 3).
//!
//! Each attention head has a measured "sensitivity" curve: for a set of
//! compression ratios `{r_0 < r_1 < …}`, the quality delta `{Δ_0, Δ_1, …}`
//! (relative to the un-compacted head) is recorded. Downstream the solver
//! interpolates these to estimate the marginal quality impact of any ratio.

use crate::STABILITY_EPS;

/// Quality-vs-ratio sensitivity curve for one attention head.
///
/// `ratios` must be sorted ascending and parallel to `deltas`. `deltas[i]` is
/// the quality drop (positive = quality loss) when keeping `ratios[i]` of the
/// head's KV cache. By convention, `ratios[0]` is the most aggressive
/// compaction (smallest `r`) and the last entry is the least aggressive
/// (largest `r`, typically `r = 1.0` with `Δ = 0`).
///
/// Interpolation is linear within `[ratios[0], ratios[last]]` and flat
/// (extrapolated to the boundary value) outside.
#[derive(Clone, Debug)]
pub struct HeadSensitivityCurve {
    /// Which head this curve describes (index into the layer × head matrix).
    pub head_id: usize,
    /// Sorted ascending ratios, parallel to `deltas`.
    pub ratios: Vec<f32>,
    /// Quality delta at each ratio. Lower is better (less quality lost).
    pub deltas: Vec<f32>,
}

impl HeadSensitivityCurve {
    /// Construct a curve, validating that ratios are sorted ascending and
    /// parallel in length to deltas.
    pub fn new(head_id: usize, ratios: Vec<f32>, deltas: Vec<f32>) -> Self {
        assert_eq!(
            ratios.len(),
            deltas.len(),
            "ratios and deltas must be parallel"
        );
        assert!(!ratios.is_empty(), "curve must have at least one point");
        for w in ratios.windows(2) {
            assert!(
                w[0] <= w[1],
                "ratios must be sorted ascending (got {:?})",
                ratios
            );
        }
        Self {
            head_id,
            ratios,
            deltas,
        }
    }

    /// Linear interpolation of the quality delta at the given ratio.
    ///
    /// - Below `ratios[0]`, returns `deltas[0]` (flat extrapolation).
    /// - Above `ratios[last]`, returns `deltas[last]` (flat extrapolation).
    /// - Within range, linear interpolation between the two surrounding points.
    #[inline]
    pub fn interpolate(&self, ratio: f32) -> f32 {
        let n = self.ratios.len();
        if n == 1 {
            return self.deltas[0];
        }
        if ratio <= self.ratios[0] {
            return self.deltas[0];
        }
        if ratio >= self.ratios[n - 1] {
            return self.deltas[n - 1];
        }
        // Binary search would be faster for large curves, but the typical
        // curve has 5–10 points, so a linear scan is cache-friendly and
        // branch-predictable.
        let mut i = 0usize;
        while i + 1 < n && self.ratios[i + 1] < ratio {
            i += 1;
        }
        let r0 = self.ratios[i];
        let r1 = self.ratios[i + 1];
        let d0 = self.deltas[i];
        let d1 = self.deltas[i + 1];
        let span = r1 - r0;
        if span.abs() < STABILITY_EPS {
            return d0;
        }
        let t = (ratio - r0) / span;
        d0 + t * (d1 - d0)
    }

    /// Marginal quality gain when increasing this head's ratio from
    /// `r_from` to `r_from + step` (i.e., quality recovered by giving the
    /// head more budget). Positive = more quality recovered.
    ///
    /// `interpolate(r_from) - interpolate(r_from + step)` because deltas are
    /// quality *losses*: a larger ratio → smaller delta → less loss → gain.
    #[inline]
    pub fn marginal_gain(&self, r_from: f32, step: f32) -> f32 {
        self.interpolate(r_from) - self.interpolate(r_from + step)
    }

    /// Number of measured points.
    #[inline]
    pub fn len(&self) -> usize {
        self.ratios.len()
    }

    /// Whether the curve is empty (always false after `new` since we assert
    /// non-empty, but provided for clippy).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ratios.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_within_range() {
        let c = HeadSensitivityCurve::new(
            0,
            vec![0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            vec![0.9, 0.7, 0.4, 0.2, 0.05, 0.0],
        );
        // At r=0.3, halfway between (0.2, 0.7) and (0.4, 0.4) → 0.55.
        let v = c.interpolate(0.3);
        assert!((v - 0.55).abs() < 1e-6, "interpolate(0.3)={}, expected 0.55", v);
    }

    #[test]
    fn test_interpolate_at_knots() {
        let c = HeadSensitivityCurve::new(0, vec![0.1, 0.5, 1.0], vec![0.8, 0.4, 0.0]);
        assert!((c.interpolate(0.1) - 0.8).abs() < 1e-6);
        assert!((c.interpolate(0.5) - 0.4).abs() < 1e-6);
        assert!((c.interpolate(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_extrapolates_flat_below() {
        let c = HeadSensitivityCurve::new(0, vec![0.2, 1.0], vec![0.7, 0.0]);
        // r=0.05 is below 0.2 → clamp to deltas[0].
        assert!((c.interpolate(0.05) - 0.7).abs() < 1e-6);
        // r=0.0 too.
        assert!((c.interpolate(0.0) - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_extrapolates_flat_above() {
        let c = HeadSensitivityCurve::new(0, vec![0.1, 0.5], vec![0.9, 0.4]);
        // r=0.9 is above 0.5 → clamp to deltas[last].
        assert!((c.interpolate(0.9) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_marginal_gain_positive_for_more_budget() {
        // curve: ratios=[0.1, 0.5, 1.0], deltas=[0.8, 0.4, 0.0]
        // delta(0.2) = interp between (0.1,0.8) and (0.5,0.4) at t=0.25 → 0.7
        // delta(0.4) = interp between (0.1,0.8) and (0.5,0.4) at t=0.75 → 0.5
        // marginal_gain(0.2, 0.2) = delta(0.2) - delta(0.4) = 0.7 - 0.5 = 0.2
        let c = HeadSensitivityCurve::new(0, vec![0.1, 0.5, 1.0], vec![0.8, 0.4, 0.0]);
        let g = c.marginal_gain(0.2, 0.2);
        assert!(g > 0.0, "marginal gain should be positive, got {}", g);
        assert!((g - 0.2).abs() < 1e-5, "marginal gain={}, expected 0.2", g);
    }

    #[test]
    #[should_panic(expected = "ratios and deltas must be parallel")]
    fn test_new_rejects_mismatched_lengths() {
        let _ = HeadSensitivityCurve::new(0, vec![0.1, 0.5], vec![0.8]);
    }

    #[test]
    #[should_panic(expected = "curve must have at least one point")]
    fn test_new_rejects_empty() {
        let _ = HeadSensitivityCurve::new(0, vec![], vec![]);
    }

    #[test]
    #[should_panic(expected = "ratios must be sorted ascending")]
    fn test_new_rejects_unsorted() {
        let _ = HeadSensitivityCurve::new(0, vec![0.5, 0.1], vec![0.8, 0.9]);
    }

    #[test]
    fn test_single_point_curve() {
        let c = HeadSensitivityCurve::new(0, vec![0.5], vec![0.3]);
        assert!((c.interpolate(0.0) - 0.3).abs() < 1e-6);
        assert!((c.interpolate(1.0) - 0.3).abs() < 1e-6);
    }
}
