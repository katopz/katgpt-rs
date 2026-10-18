//! Welford online variance accumulator — the crate's ONE definition.
//!
//! Moved here (Plan: reflex 008 / Issue 055 substrate pass, 2026-09-30) from
//! `karc::regime_gate`'s feature-gated `imp` module so a second substrate
//! consumer (`perturbation_ensemble`) can reach it without dragging the
//! `karc_regime_gate` feature chain (`karc_forecaster +
//! conformal_predictive_intervals`) into consumers that want only moments.
//! `karc::regime_gate` re-exports it — both historical paths still resolve;
//! there is exactly one definition (the `rating` promotion precedent).
//!
//! Tracks `(count, mean, M2)` per Welford 1962. Variance = `M2 / (n − 1)`
//! (sample variance); returns `None` until two observations are accumulated.
//!
//! NaN inputs are silently rejected (no state change) so callers stay
//! well-defined during cold-start (one forecaster has no forecast yet, one
//! ensemble sample produced no score).
//!
//!
//! Welford online variance accumulator — the crate's ONE definition
//! (moved out of karc::regime_gate's feature-gated `imp`, Plan: reflex 008 /
//! Issue 055 substrate pass, 2026-09-30, so `perturbation_ensemble` and any
//! other moments consumer reaches it without the karc feature chain).
/// Welford online variance accumulator — closed-form, single-pass, zero-alloc.
#[derive(Clone, Copy, Debug, Default)]
pub struct WelfordVariance {
    count: usize,
    mean: f64,
    m2: f64,
}

impl WelfordVariance {
    /// New empty accumulator.
    #[inline]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Reset to empty.
    #[inline]
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }

    /// Number of observations accumulated.
    #[inline]
    pub const fn n(&self) -> usize {
        self.count
    }

    /// Push a new observation. NaN is silently rejected (state unchanged).
    /// f32 input widened to f64 for numerical robustness at small sample
    /// counts (the same widening rationale as KARC's Gram accumulation —
    /// see `linalg::ridge_solve` module doc).
    #[inline]
    pub fn observe(&mut self, x: f32) {
        if x.is_nan() {
            return;
        }
        let x = x as f64;
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.count as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Sample variance `M2 / (n − 1)`, or `None` until `n >= 2`.
    ///
    /// Captures dispersion only — NOT bias. Two streams with the same
    /// variance can have very different accuracies if their biases differ.
    /// For a "which stream has smaller error" question, use
    /// [`mse`](Self::mse) instead.
    #[inline]
    pub fn variance(&self) -> Option<f32> {
        if self.count < 2 {
            None
        } else {
            Some((self.m2 / ((self.count - 1) as f64)) as f32)
        }
    }

    /// Mean squared error vs zero target: `MSE = Var_pop + mean²`.
    ///
    /// This captures BOTH dispersion (variance) and bias (mean²): a
    /// consistently-biased stream (variance 0, large mean) gets a large MSE.
    /// Returns `None` until at least one observation has been pushed
    /// (single observation gives MSE = x²).
    ///
    /// Computed as `M2/n + mean²` (the population-variance form, which
    /// matches a residual stream's true second moment `E[r²]`). The
    /// sample-variance `M2/(n-1)` form is exposed separately as
    /// [`variance`](Self::variance) for diagnostics.
    #[inline]
    pub fn mse(&self) -> Option<f32> {
        if self.count < 1 {
            None
        } else {
            let var_pop = self.m2 / (self.count as f64);
            let mean_sq = self.mean * self.mean;
            Some((var_pop + mean_sq) as f32)
        }
    }

    /// Sample mean, or `0.0` when empty (well-defined cold-start value).
    #[inline]
    pub const fn mean(&self) -> f64 {
        self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The move pin: the accumulator's arithmetic is unchanged from the
    /// frozen karc body (which itself is Welford 1962 by definition).
    #[test]
    fn welford_matches_definition() {
        let mut w = WelfordVariance::new();
        let xs = [1.0f32, 2.0, 3.0, 4.0];
        for &x in &xs {
            w.observe(x);
        }
        assert_eq!(w.n(), 4);
        assert!((w.mean() - 2.5).abs() < 1e-12);
        let var = w.variance().unwrap();
        let expected: f32 = xs.iter().map(|x| (x - 2.5).powi(2)).sum::<f32>() / 3.0;
        assert!((var - expected).abs() < 1e-5, "{var} vs {expected}");
    }

    #[test]
    fn nan_rejected_and_cold_start_defined() {
        let mut w = WelfordVariance::new();
        assert_eq!(w.n(), 0);
        assert_eq!(w.variance(), None);
        assert_eq!(w.mse(), None);
        assert_eq!(w.mean(), 0.0);
        w.observe(f32::NAN);
        assert_eq!(w.n(), 0, "NaN must be silently rejected");
        w.observe(3.0);
        assert_eq!(w.n(), 1);
        assert_eq!(w.mse(), Some(9.0));
        assert_eq!(w.variance(), None);
        w.reset();
        assert_eq!(w.n(), 0);
    }
}
