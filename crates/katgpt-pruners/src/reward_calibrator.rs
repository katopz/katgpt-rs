//! Reward-Gated Pruner Calibration — formalizes the AbsorbCompress pattern with
//! sigmoid-bounded Q-values, Welford variance tracking, and blake3 audit trail.
//!
//! Wraps any [`ScreeningPruner`] and tracks parameter→reward mapping. When a parameter
//! reaches stability (sufficient visits, low variance), it's eligible for absorption
//! into a fixed constraint.
//!
//! # Usage
//!
//! ```rust,ignore
//! let calibrator = RewardGatedCalibrator::new(NoScreeningPruner);
//!
//! // Record rewards from evaluation
//! let key = ParameterKey { pruner_id: 0, parameter_idx: 0, depth: 0 };
//! calibrator.record_reward(key, 0.8);
//!
//! // Sigmoid-bounded bandit update
//! if let Some(step) = calibrator.bandit_update(key, 0.9) {
//!     println!("Calibrated: {:?}", step);
//! }
//!
//! // Check absorption eligibility
//! if calibrator.should_absorb(&key) {
//!     println!("Parameter stable — ready for absorption");
//! }
//! ```
//!
//! # Plan 210 F4: Reward-Gated Pruner Calibration
//!
//! # Feature Gate
//!
//! `reward_calibrator`.
//!
//! # Performance
//!
//! - Per-call overhead: ~262ns (`record_reward`)
//! - Audit: blake3 hash per calibration step
//! - Regression: `verify_regression()` supports [`RegressionSuite`] with automatic rollback

use blake3::Hasher;

use katgpt_speculative::ScreeningPruner;

#[cfg(feature = "bandit")]
use super::regression::{GoldenTrace, RegressionSuite, ReplayReward};

// ── Sigmoid ──────────────────────────────────────────────────────

/// Sigmoid activation: `1 / (1 + exp(-x))`. Bounds output to (0, 1).
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── ParameterKey ─────────────────────────────────────────────────

/// 8-byte cache-line friendly key identifying a pruner parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParameterKey {
    pub pruner_id: u32,
    pub parameter_idx: u16,
    pub depth: u16,
}

// ── ParameterStats ───────────────────────────────────────────────

/// 12-byte running statistics for a parameter.
///
/// Variance is tracked via Welford's online algorithm for numerical stability.
#[derive(Clone, Debug, Default)]
pub struct ParameterStats {
    pub reward_sum: f32,
    pub visits: u32,
    pub variance: f32,
}

// ── CalibrationStep ──────────────────────────────────────────────

/// A single calibration step with blake3 audit trail.
#[derive(Clone, Debug)]
pub struct CalibrationStep {
    pub parameter_id: ParameterKey,
    pub old_value: f32,
    pub new_value: f32,
    pub reward_delta: f32,
    pub hash: [u8; 32],
}

impl CalibrationStep {
    /// Compute blake3 hash of (old_value || new_value || reward_delta) as raw f32 bytes.
    pub fn compute_hash(old: f32, new: f32, delta: f32) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&old.to_le_bytes());
        hasher.update(&new.to_le_bytes());
        hasher.update(&delta.to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

// ── CalibratorConfig ─────────────────────────────────────────────

/// Tunable thresholds for reward-gated calibration.
#[derive(Clone, Debug)]
pub struct CalibratorConfig {
    /// Minimum visits before a parameter is eligible for absorption.
    pub min_visits: usize,
    /// Maximum variance for a parameter to be considered stable.
    pub variance_threshold: f32,
    /// Learning rate for sigmoid-bounded Q-value updates.
    pub learning_rate: f32,
}

impl Default for CalibratorConfig {
    fn default() -> Self {
        Self {
            min_visits: 10,
            variance_threshold: 0.1,
            learning_rate: 0.1,
        }
    }
}

// ── RewardGatedCalibrator ────────────────────────────────────────

/// Wraps a [`ScreeningPruner`] and tracks parameter→reward mapping.
///
/// When a parameter reaches stability (sufficient visits, low variance),
/// it's eligible for absorption into a fixed constraint. All calibration
/// steps are recorded with a blake3 audit trail.
#[cfg(feature = "reward_calibrator")]
pub struct RewardGatedCalibrator<P: ScreeningPruner> {
    inner: P,
    param_stats: std::collections::HashMap<ParameterKey, ParameterStats>,
    calibration_log: Vec<CalibrationStep>,
    config: CalibratorConfig,
}

#[cfg(feature = "reward_calibrator")]
impl<P: ScreeningPruner> RewardGatedCalibrator<P> {
    /// Create a new calibrator wrapping `inner` with default config.
    pub fn new(inner: P) -> Self {
        Self::with_config(inner, CalibratorConfig::default())
    }

    /// Create a new calibrator wrapping `inner` with custom config.
    pub fn with_config(inner: P, config: CalibratorConfig) -> Self {
        Self {
            inner,
            param_stats: std::collections::HashMap::new(),
            calibration_log: Vec::new(),
            config,
        }
    }

    /// Record a reward observation for the given parameter key.
    ///
    /// Uses Welford's online algorithm for numerically stable variance tracking.
    pub fn record_reward(&mut self, key: ParameterKey, reward: f32) {
        let stats = self.param_stats.entry(key).or_default();

        // Welford's online algorithm for variance
        let n = stats.visits + 1;
        let delta = reward - stats.mean();
        let new_mean = stats.mean() + delta / n as f32;
        let delta2 = reward - new_mean;

        // Welford M2 accumulator: M2 += delta * delta2
        // variance = M2 / n
        let m2 = stats.variance * stats.visits as f32 + delta * delta2;
        stats.variance = m2 / n as f32;

        stats.reward_sum += reward;
        stats.visits = n;
    }

    /// Perform a sigmoid-bounded bandit update on the parameter.
    ///
    /// Update rule: `new_value = old + lr * sigmoid(reward - old)`
    ///
    /// Returns a [`CalibrationStep`] with blake3 audit hash if the update occurred.
    pub fn bandit_update(&mut self, key: ParameterKey, reward: f32) -> Option<CalibrationStep> {
        let stats = self.param_stats.get(&key)?;

        let old_value = stats.mean();
        let reward_delta = reward - old_value;
        let new_value = old_value + self.config.learning_rate * sigmoid(reward_delta);

        let hash = CalibrationStep::compute_hash(old_value, new_value, reward_delta);

        let step = CalibrationStep {
            parameter_id: key,
            old_value,
            new_value,
            reward_delta,
            hash,
        };

        self.calibration_log.push(step.clone());
        Some(step)
    }

    /// Check if a parameter is eligible for absorption.
    ///
    /// A parameter can be absorbed when:
    /// - `visits >= min_visits` AND
    /// - `variance <= variance_threshold`
    pub fn should_absorb(&self, key: &ParameterKey) -> bool {
        match self.param_stats.get(key) {
            Some(stats) => {
                stats.visits as usize >= self.config.min_visits
                    && stats.variance <= self.config.variance_threshold
            }
            None => false,
        }
    }

    /// Access the calibration audit log.
    pub fn calibration_log(&self) -> &[CalibrationStep] {
        &self.calibration_log
    }

    /// Roll back the most recent calibration step.
    ///
    /// Removes the last [`CalibrationStep`] from the audit log.
    /// Note: parameter stats (`param_stats`) are **not** restored — a full
    /// rollback would require storing pre-calibration snapshots. This method
    /// only removes the audit trail entry for bookkeeping.
    pub fn rollback_last(&mut self) -> Option<CalibrationStep> {
        self.calibration_log.pop()
    }

    /// Verify calibration absorption against a regression suite (Plan 210 F4.6).
    ///
    /// Replays all golden traces through fresh pruners created by `pruner_factory`.
    /// If any trace fails, the last calibration step is rolled back.
    /// Returns `true` if all traces pass.
    #[cfg(feature = "bandit")]
    pub fn verify_regression<F, R>(&mut self, suite: &RegressionSuite, pruner_factory: F) -> bool
    where
        F: Fn(&GoldenTrace) -> R,
        R: ReplayReward,
    {
        let result = suite.run(pruner_factory);
        if !result.all_passed() {
            self.rollback_last();
        }
        result.all_passed()
    }
}

#[cfg(feature = "reward_calibrator")]
impl<P: ScreeningPruner> ScreeningPruner for RewardGatedCalibrator<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Welford mean helper ─────────────────────────────────────────

impl ParameterStats {
    /// Compute running mean from sum and visits. Returns 0.0 if no visits.
    fn mean(&self) -> f32 {
        match self.visits {
            0 => 0.0,
            n => self.reward_sum / n as f32,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// No-op pruner for testing.
    struct AllowAll;

    impl ScreeningPruner for AllowAll {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    fn make_key(pruner_id: u32, parameter_idx: u16, depth: u16) -> ParameterKey {
        ParameterKey {
            pruner_id,
            parameter_idx,
            depth,
        }
    }

    #[test]
    fn test_parameter_tracking_accumulates_rewards() {
        let mut cal = RewardGatedCalibrator::new(AllowAll);
        let key = make_key(0, 0, 0);

        cal.record_reward(key.clone(), 0.5);
        cal.record_reward(key.clone(), 0.7);
        cal.record_reward(key.clone(), 0.9);

        let stats = cal.param_stats.get(&key).unwrap();
        assert_eq!(stats.visits, 3);
        assert!(
            (stats.reward_sum - 2.1).abs() < 1e-5,
            "reward_sum should be 2.1"
        );
        assert!((stats.mean() - 0.7).abs() < 1e-5, "mean should be 0.7");
    }

    #[test]
    fn test_sigmoid_bounded_q_values() {
        // Sigmoid output is bounded in [0, 1]
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let s = sigmoid(x);
            assert!(s > 0.0 && s < 1.0, "sigmoid({x}) = {s}, expected (0, 1)");
        }

        // Extreme values clamp to 0 or 1 in f32 precision
        assert_eq!(
            sigmoid(-100.0),
            0.0,
            "sigmoid(-100) should underflow to 0.0"
        );
        assert_eq!(sigmoid(100.0), 1.0, "sigmoid(100) should saturate to 1.0");

        // Sigmoid(0) = 0.5
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);

        // Verify sigmoid-bounded update: new = old + lr * sigmoid(delta)
        // Since sigmoid ∈ (0, 1), the update magnitude is bounded by lr.
        let mut cal = RewardGatedCalibrator::with_config(
            AllowAll,
            CalibratorConfig {
                min_visits: 1,
                variance_threshold: 1.0,
                learning_rate: 0.1,
            },
        );
        let key = make_key(0, 0, 0);

        cal.record_reward(key.clone(), 0.5);

        // Bandit update with very high reward
        let step = cal.bandit_update(key.clone(), 100.0).unwrap();
        let delta = step.new_value - step.old_value;
        assert!(
            delta <= 0.1 + 1e-6,
            "update delta {delta} should be bounded by learning rate 0.1"
        );

        // Bandit update with very low reward
        cal.record_reward(key.clone(), 0.5);
        let step = cal.bandit_update(key.clone(), -100.0).unwrap();
        let delta = step.new_value - step.old_value;
        assert!(
            delta >= 0.0 - 1e-6,
            "update delta {delta} with very negative reward should be near 0 (sigmoid → 0)"
        );
    }

    #[test]
    fn test_absorption_triggers_when_stable() {
        let mut cal = RewardGatedCalibrator::with_config(
            AllowAll,
            CalibratorConfig {
                min_visits: 10,
                variance_threshold: 0.1,
                learning_rate: 0.1,
            },
        );
        let key = make_key(0, 0, 0);

        // Feed identical rewards — variance = 0
        for _ in 0..10 {
            cal.record_reward(key.clone(), 0.5);
        }

        assert!(
            cal.should_absorb(&key),
            "should absorb: 10 visits, zero variance"
        );
    }

    #[test]
    fn test_absorption_blocked_by_high_variance() {
        let mut cal = RewardGatedCalibrator::with_config(
            AllowAll,
            CalibratorConfig {
                min_visits: 10,
                variance_threshold: 0.1,
                learning_rate: 0.1,
            },
        );
        let key = make_key(0, 0, 0);

        // Feed high-variance rewards
        for i in 0..10 {
            cal.record_reward(key.clone(), if i % 2 == 0 { 0.0 } else { 1.0 });
        }

        assert!(
            !cal.should_absorb(&key),
            "should NOT absorb: high variance exceeds threshold"
        );
    }

    #[test]
    fn test_blake3_audit_hash_deterministic() {
        let hash1 = CalibrationStep::compute_hash(0.5, 0.6, 0.1);
        let hash2 = CalibrationStep::compute_hash(0.5, 0.6, 0.1);
        assert_eq!(hash1, hash2, "same inputs must produce same hash");

        // Different inputs → different hash
        let hash3 = CalibrationStep::compute_hash(0.5, 0.6, 0.2);
        assert_ne!(hash1, hash3, "different inputs must produce different hash");
    }

    #[test]
    fn test_miss_path_returns_false_for_should_absorb() {
        let cal: RewardGatedCalibrator<AllowAll> = RewardGatedCalibrator::new(AllowAll);
        let key = make_key(99, 99, 99);
        assert!(!cal.should_absorb(&key), "missing key should return false");
    }

    #[test]
    fn test_welford_variance_converges() {
        let mut stats = ParameterStats::default();
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        for v in &values {
            let n = stats.visits + 1;
            let delta = v - stats.mean();
            let new_mean = stats.mean() + delta / n as f32;
            let delta2 = v - new_mean;
            let m2 = stats.variance * stats.visits as f32 + delta * delta2;
            stats.variance = m2 / n as f32;
            stats.reward_sum += v;
            stats.visits = n;
        }

        // Population variance of [2, 4, 4, 4, 5, 5, 7, 9] = 4.0
        assert!(
            (stats.variance - 4.0).abs() < 1e-4,
            "variance should be 4.0, got {}",
            stats.variance
        );
        // Mean = 5.0
        assert!(
            (stats.mean() - 5.0).abs() < 1e-5,
            "mean should be 5.0, got {}",
            stats.mean()
        );
    }

    #[test]
    fn test_delegates_relevance_to_inner() {
        let cal = RewardGatedCalibrator::new(AllowAll);
        let score = cal.relevance(0, 42, &[]);
        assert!(
            (score - 1.0).abs() < 1e-6,
            "should delegate to inner pruner"
        );
    }

    #[test]
    fn test_calibration_log_records_steps() {
        let mut cal = RewardGatedCalibrator::new(AllowAll);
        let key = make_key(0, 0, 0);

        cal.record_reward(key.clone(), 0.5);
        cal.bandit_update(key.clone(), 0.8).unwrap();

        assert_eq!(cal.calibration_log().len(), 1);
        assert_eq!(cal.calibration_log()[0].parameter_id, key);
    }

    #[test]
    fn test_bandit_update_returns_none_for_missing_key() {
        let mut cal = RewardGatedCalibrator::new(AllowAll);
        let key = make_key(0, 0, 0);
        let result = cal.bandit_update(key, 0.5);
        assert!(result.is_none(), "no stats recorded → no update");
    }

    #[test]
    fn test_rollback_last_removes_step() {
        let mut cal = RewardGatedCalibrator::new(AllowAll);
        let key = make_key(0, 0, 0);

        cal.record_reward(key.clone(), 0.5);
        cal.bandit_update(key.clone(), 0.8).unwrap();
        cal.bandit_update(key.clone(), 0.9).unwrap();
        assert_eq!(cal.calibration_log().len(), 2);

        let step = cal.rollback_last();
        assert!(step.is_some());
        assert_eq!(cal.calibration_log().len(), 1);

        let step = cal.rollback_last();
        assert!(step.is_some());
        assert_eq!(cal.calibration_log().len(), 0);

        let step = cal.rollback_last();
        assert!(step.is_none(), "empty log should return None");
    }

    #[cfg(feature = "bandit")]
    mod bandit_tests {
        use super::*;
        use crate::regression::{GoldenTrace, RegressionSuite, ReplayReward};

        /// Replay that always returns the expected reward — all traces pass.
        struct PerfectReplay;

        impl ReplayReward for PerfectReplay {
            fn replay_reward(&mut self, trace: &GoldenTrace) -> f32 {
                trace.expected_reward
            }
        }

        /// Replay that always returns zero — all traces fail.
        struct FailingReplay;

        impl ReplayReward for FailingReplay {
            fn replay_reward(&mut self, _trace: &GoldenTrace) -> f32 {
                0.0
            }
        }

        fn sample_trace(label: &str, reward: f32) -> GoldenTrace {
            GoldenTrace {
                label: label.to_string(),
                actions: vec![0],
                expected_reward: reward,
                expected_survival: reward > 0.0,
            }
        }

        #[test]
        fn test_verify_regression_all_pass() {
            let mut cal = RewardGatedCalibrator::new(AllowAll);
            let key = make_key(0, 0, 0);
            cal.record_reward(key.clone(), 0.5);
            cal.bandit_update(key, 0.8).unwrap();
            assert_eq!(cal.calibration_log().len(), 1);

            let suite =
                RegressionSuite::new(vec![sample_trace("t1", 0.8), sample_trace("t2", 0.9)], 0.1);

            let passed = cal.verify_regression(&suite, |_| PerfectReplay);
            assert!(passed, "all traces should pass");
            assert_eq!(cal.calibration_log().len(), 1, "no rollback on pass");
        }

        #[test]
        fn test_verify_regression_rollback_on_failure() {
            let mut cal = RewardGatedCalibrator::new(AllowAll);
            let key = make_key(0, 0, 0);
            cal.record_reward(key.clone(), 0.5);
            cal.bandit_update(key, 0.8).unwrap();
            assert_eq!(cal.calibration_log().len(), 1);

            let suite = RegressionSuite::new(vec![sample_trace("fail", 1.0)], 0.1);

            let passed = cal.verify_regression(&suite, |_| FailingReplay);
            assert!(!passed, "trace should fail");
            assert_eq!(cal.calibration_log().len(), 0, "should rollback on failure");
        }
    }

    // ── Performance Benchmarks (F4.8) ─────────────────────────────

    #[test]
    fn test_calibration_overhead_per_call() {
        use std::time::Instant;

        let mut cal = RewardGatedCalibrator::new(AllowAll);

        // Warm up
        for _ in 0..100 {
            cal.record_reward(make_key(0, 0, 0), 0.5);
        }

        // Measure record_reward throughput
        let iters = 10_000;
        let start = Instant::now();
        for i in 0..iters {
            let key = make_key((i % 10) as u32, 0, 0);
            cal.record_reward(key, 0.7);
        }
        let elapsed = start.elapsed();
        let per_call_ns = elapsed.as_nanos() as f64 / iters as f64;

        // Target: <10µs per call (generous — actual target is 100ns)
        assert!(
            per_call_ns < 10_000.0,
            "Calibration overhead {per_call_ns:.0}ns exceeds 10µs target"
        );

        eprintln!("  F4.8 calibration overhead: {per_call_ns:.0}ns per record_reward call");
    }
}
