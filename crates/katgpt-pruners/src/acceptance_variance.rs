//! AcceptanceVarianceTracker — Welford online variance + EMA smoothing for RV signal.
//!
//! Maps RAGEN-2's "reward variance = SNR proxy" to inference-time compute allocation.
//! O(1) per update, ~48 bytes per tracker. Feature-gated behind `rv_gated_routing`.

// ── AcceptanceVarianceTracker ─────────────────────────────────────

/// Online variance tracker using Welford's algorithm with EMA smoothing.
///
/// Observes boolean acceptance events (accepted/rejected from speculative decode)
/// and maintains a running variance estimate. The EMA-smoothed variance (RV) serves
/// as an SNR proxy for compute routing decisions.
///
/// High RV → model uncertain → promote to GPU + Latent thinking.
/// Low RV → model confident → CPU direct decode suffices.
pub struct AcceptanceVarianceTracker {
    /// Welford running mean of acceptance rate.
    mean: f64,
    /// Welford running M2 (sum of squared deviations).
    m2: f64,
    /// Number of observations seen.
    count: u64,
    /// EMA-smoothed variance (the RV signal).
    ema_rv: f64,
    /// EMA smoothing factor (0 < α ≤ 1). Default: 0.1.
    ema_alpha: f64,
    /// Minimum samples before RV is reported. Default: 5.
    min_samples: u64,
}

impl AcceptanceVarianceTracker {
    /// Create a new tracker with default parameters.
    ///
    /// Default: `ema_alpha = 0.1`, `min_samples = 5`.
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            ema_rv: 0.0,
            ema_alpha: 0.1,
            min_samples: 5,
        }
    }

    /// Create a tracker with custom parameters.
    pub fn with_params(ema_alpha: f64, min_samples: u64) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            ema_rv: 0.0,
            ema_alpha: ema_alpha.clamp(0.001, 1.0),
            min_samples,
        }
    }

    /// Observe an acceptance event. O(1), 3 flops.
    ///
    /// Converts `accepted` to 1.0/0.0 and updates Welford statistics,
    /// then applies EMA smoothing to the variance estimate.
    pub fn observe(&mut self, accepted: bool) {
        let x = if accepted { 1.0 } else { 0.0 };
        self.count += 1;

        // Welford online variance update
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        // EMA smoothing: blend current sample variance into running estimate
        let sample_var = self.sample_variance();
        self.ema_rv = self.ema_alpha * sample_var + (1.0 - self.ema_alpha) * self.ema_rv;
    }

    /// Current EMA-smoothed variance (the RV signal).
    ///
    /// Returns 0.0 if fewer than `min_samples` observations have been recorded.
    /// For boolean data: RV ∈ [0.0, 0.25] where 0.25 = maximum variance (p=0.5).
    pub fn rv(&self) -> f64 {
        match self.count < self.min_samples {
            true => 0.0,
            false => self.ema_rv,
        }
    }

    /// Reset all per-query state.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
        self.ema_rv = 0.0;
    }

    /// Number of observations recorded.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Current running mean of acceptance rate.
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Raw (un-smoothed) sample variance.
    fn sample_variance(&self) -> f64 {
        match self.count {
            0 | 1 => 0.0,
            _ => self.m2 / (self.count as f64 - 1.0),
        }
    }
}

impl Default for AcceptanceVarianceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_accept_rv_near_zero() {
        let mut tracker = AcceptanceVarianceTracker::new();
        // Variance of a constant (all 1.0) is 0
        for _ in 0..20 {
            tracker.observe(true);
        }
        let rv = tracker.rv();
        assert!(rv < 0.001, "all-accept RV should be ≈ 0, got {rv:.6}");
    }

    #[test]
    fn test_all_reject_rv_near_zero() {
        let mut tracker = AcceptanceVarianceTracker::new();
        // Variance of a constant (all 0.0) is 0
        for _ in 0..20 {
            tracker.observe(false);
        }
        let rv = tracker.rv();
        assert!(rv < 0.001, "all-reject RV should be ≈ 0, got {rv:.6}");
    }

    #[test]
    fn test_mixed_accept_reject_rv_positive() {
        let mut tracker = AcceptanceVarianceTracker::new();
        // 50/50 should give variance ≈ 0.25 (maximum for Bernoulli)
        for i in 0..100 {
            tracker.observe(i % 2 == 0);
        }
        let rv = tracker.rv();
        assert!(rv > 0.1, "50/50 RV should be > 0.1, got {rv:.6}");
    }

    #[test]
    fn test_ema_converges_to_true_variance() {
        let mut tracker = AcceptanceVarianceTracker::with_params(0.3, 5);
        // Alternating: true variance = 0.25 for Bernoulli(0.5)
        for i in 0..1000 {
            tracker.observe(i % 2 == 0);
        }
        let rv = tracker.rv();
        // Should be close to 0.25 (Bernoulli p=0.5 variance)
        assert!(
            (rv - 0.25).abs() < 0.02,
            "EMA should converge to ≈ 0.25, got {rv:.6}"
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut tracker = AcceptanceVarianceTracker::new();
        for i in 0..20 {
            tracker.observe(i % 2 == 0);
        }
        assert!(tracker.count() > 0);
        assert!(tracker.rv() > 0.0);

        tracker.reset();
        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.rv(), 0.0);
        assert_eq!(tracker.mean(), 0.0);
    }

    #[test]
    fn test_min_samples_gate() {
        let mut tracker = AcceptanceVarianceTracker::with_params(0.1, 10);
        // Below min_samples: RV should be 0
        for i in 0..9 {
            tracker.observe(i % 2 == 0);
        }
        assert_eq!(tracker.rv(), 0.0, "RV should be 0 below min_samples");

        // At min_samples: RV should be reported
        tracker.observe(true);
        assert!(tracker.rv() > 0.0, "RV should be > 0 at min_samples");
    }

    #[test]
    fn test_ema_alpha_clamped() {
        let tracker = AcceptanceVarianceTracker::with_params(0.0, 5);
        assert!(
            tracker.ema_alpha >= 0.001,
            "alpha should be clamped to 0.001"
        );

        let tracker = AcceptanceVarianceTracker::with_params(5.0, 5);
        assert!(tracker.ema_alpha <= 1.0, "alpha should be clamped to 1.0");
    }

    #[test]
    fn test_single_observation_zero_variance() {
        let mut tracker = AcceptanceVarianceTracker::with_params(0.1, 1);
        tracker.observe(true);
        // 1 sample → sample variance = 0
        assert_eq!(tracker.rv(), 0.0);
    }
}

// TL;DR: AcceptanceVarianceTracker — Welford O(1) online variance + EMA smoothing.
// observe(bool) → rv() returns smoothed variance signal. ~48 bytes, 3 flops/update.
// High RV = uncertain → GPU. Low RV = confident → CPU. Default-OFF behind rv_gated_routing.
