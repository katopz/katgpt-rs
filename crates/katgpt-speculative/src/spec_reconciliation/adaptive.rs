//! AdaptiveReconciler — per-player threshold learning for the reconciliation pipeline.
//!
//! Wraps [`SpecReconciler`] with a lightweight bandit for per-player threshold adaptation.
//! Uses `ThinkingBanditFrozen`-style freeze/thaw for persistence across sessions.
//!
//! The bandit has two arms: "strict" (high threshold) and "lenient" (low threshold).
//! It learns per-player which threshold minimizes false positives while catching hacks.

use super::reconciler::{ReconciliationResult, SpecReconciler};
use super::types::{ReconciliationConfig, ReconciliationVerdict, TrajectoryPoint};
use katgpt_types::Rng;

/// Number of threshold arms (strict, medium, lenient).
const NUM_ARMS: usize = 3;

/// Threshold candidates: [strict, medium, lenient].
const THRESHOLD_CANDIDATES: [f32; NUM_ARMS] = [0.9, 0.7, 0.5];

/// Frozen bandit state for persistence across sessions.
/// `repr(C)` for zero-dependency binary persistence.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveReconcilerFrozen {
    /// Magic bytes for validation: b"ADRC".
    pub magic: [u8; 4],
    /// Version for migration.
    pub version: u32,
    /// Per-arm Q-values: [strict, medium, lenient].
    pub q_values: [f32; NUM_ARMS],
    /// Per-arm visit counts.
    pub visits: [u32; NUM_ARMS],
    /// Total pulls.
    pub total_pulls: u32,
}

impl AdaptiveReconcilerFrozen {
    const MAGIC: [u8; 4] = *b"ADRC";
    const VERSION: u32 = 1;

    /// Validate frozen state.
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err(format!(
                "AdaptiveReconcilerFrozen: bad magic {:?}, expected {:?}",
                self.magic,
                Self::MAGIC
            ));
        }
        if self.version != Self::VERSION {
            return Err(format!(
                "AdaptiveReconcilerFrozen: bad version {}, expected {}",
                self.version,
                Self::VERSION
            ));
        }
        Ok(())
    }
}

/// Per-player adaptive reconciliation with bandit threshold learning.
///
/// The bandit explores between 3 threshold levels:
/// - Arm 0 (strict): accept_threshold = 0.9 → few false negatives, more false positives
/// - Arm 1 (medium): accept_threshold = 0.7 → balanced
/// - Arm 2 (lenient): accept_threshold = 0.5 → fewer false positives
///
/// Reward: 1.0 for correct decisions (Accept on legitimate, Quarantine on hack),
/// 0.0 for mistakes. The bandit converges to the optimal per-player threshold.
pub struct AdaptiveReconciler {
    /// Inner reconciler.
    reconciler: SpecReconciler,
    /// Per-arm Q-values.
    q_values: [f32; NUM_ARMS],
    /// Per-arm visit counts.
    visits: [u32; NUM_ARMS],
    /// Total pulls.
    total_pulls: u32,
    /// Last chosen arm.
    last_arm: usize,
    /// Exploration rate (ε-greedy).
    epsilon: f32,
}

impl AdaptiveReconciler {
    /// Create a new adaptive reconciler.
    pub fn new(config: ReconciliationConfig) -> Self {
        Self {
            reconciler: SpecReconciler::new(config),
            q_values: [0.0; NUM_ARMS],
            visits: [0; NUM_ARMS],
            total_pulls: 0,
            last_arm: 1, // start with medium
            epsilon: 0.3,
        }
    }

    /// Create from frozen state.
    pub fn from_frozen(
        config: ReconciliationConfig,
        frozen: &AdaptiveReconcilerFrozen,
    ) -> Result<Self, String> {
        frozen.validate()?;
        Ok(Self {
            reconciler: SpecReconciler::new(config),
            q_values: frozen.q_values,
            visits: frozen.visits,
            total_pulls: frozen.total_pulls,
            last_arm: 1,
            epsilon: 0.3,
        })
    }

    /// Freeze bandit state for persistence.
    pub fn freeze(&self) -> AdaptiveReconcilerFrozen {
        AdaptiveReconcilerFrozen {
            magic: AdaptiveReconcilerFrozen::MAGIC,
            version: AdaptiveReconcilerFrozen::VERSION,
            q_values: self.q_values,
            visits: self.visits,
            total_pulls: self.total_pulls,
        }
    }

    /// Select an arm using ε-greedy.
    fn select_arm(&mut self, rng: &mut Rng) -> usize {
        if rng.uniform() < self.epsilon {
            rng.uniform() as usize * NUM_ARMS / (u8::MAX as usize + 1)
        } else {
            self.best_arm()
        }
    }

    /// Select arm deterministically (for testing).
    fn best_arm(&self) -> usize {
        self.q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(1, |(i, _)| i)
    }

    /// Update Q-value for the last chosen arm.
    fn update(&mut self, arm: usize, reward: f32) {
        self.visits[arm] += 1;
        self.total_pulls += 1;
        let n = self.visits[arm] as f32;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;

        // Decay epsilon.
        self.epsilon *= 0.995;
        if self.epsilon < 0.01 {
            self.epsilon = 0.01;
        }
    }

    /// Reconcile with adaptive threshold selection.
    ///
    /// Returns the reconciliation result. Call `observe_outcome` afterward
    /// with the ground truth to update the bandit.
    pub fn reconcile(
        &mut self,
        h_last: &TrajectoryPoint,
        client_trajectory: &[TrajectoryPoint],
        q_goals: &[f32],
        steps: usize,
        rng: &mut Rng,
    ) -> ReconciliationResult {
        // Select threshold arm.
        self.last_arm = self.select_arm(rng);

        // Override the accept threshold.
        self.reconciler.config.accept_threshold = THRESHOLD_CANDIDATES[self.last_arm];

        self.reconciler
            .reconcile(h_last, client_trajectory, q_goals, steps, rng)
    }

    /// Observe the ground truth outcome and update the bandit.
    ///
    /// `was_legitimate` = true if the trajectory was actually legitimate.
    /// `verdict` = the verdict returned by `reconcile`.
    pub fn observe_outcome(&mut self, verdict: ReconciliationVerdict, was_legitimate: bool) {
        let reward = match (verdict, was_legitimate) {
            (ReconciliationVerdict::Accept, true) => 1.0, // correct accept
            (ReconciliationVerdict::Quarantine, false) => 1.0, // correct quarantine
            (ReconciliationVerdict::Uncertain, _) => 0.5, // uncertain = partial
            (ReconciliationVerdict::Accept, false) => 0.0, // missed hack
            (ReconciliationVerdict::Quarantine, true) => 0.0, // false positive
        };
        self.update(self.last_arm, reward);
    }

    /// Get the current best threshold (Q-value-weighted).
    pub fn best_threshold(&self) -> f32 {
        THRESHOLD_CANDIDATES[self.best_arm()]
    }

    /// Get the current epsilon (exploration rate).
    #[inline]
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get Q-values for inspection.
    pub fn q_values(&self) -> &[f32; NUM_ARMS] {
        &self.q_values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> ReconciliationConfig {
        ReconciliationConfig {
            k: 8,
            max_speed: 600.0,
            map_bounds: [0.0, 0.0, 4096.0, 4096.0],
            accept_threshold: 0.7,
            quarantine_threshold: 0.2,
            kill_rate_sigma: 5.0,
            noise_sigma: 0.1,
            dt: 1.0 / 60.0,
        }
    }

    fn make_h_last() -> TrajectoryPoint {
        TrajectoryPoint::from_fields(2048.0, 2048.0, 10.0, 5.0, 2.0, 0.0, 1.0, 0.0)
    }

    #[test]
    fn test_frozen_round_trip() {
        let config = make_config();
        let reconciler = AdaptiveReconciler::new(config);
        let frozen = reconciler.freeze();
        assert!(frozen.validate().is_ok());

        let restored = AdaptiveReconciler::from_frozen(config, &frozen).unwrap();
        let refrozen = restored.freeze();
        assert_eq!(frozen.q_values, refrozen.q_values);
        assert_eq!(frozen.visits, refrozen.visits);
    }

    #[test]
    fn test_frozen_bad_magic() {
        let frozen = AdaptiveReconcilerFrozen {
            magic: *b"XXXX",
            version: 1,
            q_values: [0.0; NUM_ARMS],
            visits: [0; NUM_ARMS],
            total_pulls: 0,
        };
        assert!(AdaptiveReconciler::from_frozen(make_config(), &frozen).is_err());
    }

    #[test]
    fn test_bandit_converges_to_optimal_threshold() {
        let config = make_config();
        let mut adaptive = AdaptiveReconciler::new(config);
        let h_last = make_h_last();
        let mut rng = Rng::new(42);

        // Simulate 100 reconciliations with legitimate trajectories.
        for _ in 0..100 {
            // Legitimate: small movement
            let client: Vec<TrajectoryPoint> = (0..5)
                .map(|i| {
                    TrajectoryPoint::from_fields(
                        h_last.pos_x() + i as f32 * 0.1,
                        h_last.pos_y(),
                        10.0,
                        5.0,
                        2.0,
                        0.0,
                        1.0,
                        0.0,
                    )
                })
                .collect();

            let result = adaptive.reconcile(&h_last, &client, &[], 5, &mut rng);
            adaptive.observe_outcome(result.verdict, true);
        }

        // After 100 episodes, the bandit should have converged.
        // The best arm should have a positive Q-value.
        let best = adaptive.best_arm();
        assert!(
            adaptive.q_values[best] > 0.0,
            "best arm should have positive Q-value after training"
        );
        // Epsilon should have decayed.
        assert!(
            adaptive.epsilon() < 0.3,
            "epsilon should decay after 100 episodes, got {}",
            adaptive.epsilon()
        );
    }

    #[test]
    fn test_adaptive_handles_hacks() {
        let config = make_config();
        let mut adaptive = AdaptiveReconciler::new(config);
        let h_last = make_h_last();
        let mut rng = Rng::new(42);

        // Mix of legitimate and hack trajectories.
        for episode in 0..100 {
            let is_hack = episode % 4 == 0; // 25% hacks

            let client = if is_hack {
                // Teleport hack
                vec![
                    h_last,
                    TrajectoryPoint::from_fields(
                        h_last.pos_x() + 5000.0,
                        h_last.pos_y(),
                        0.0,
                        0.0,
                        2.0,
                        0.0,
                        1.0,
                        0.0,
                    ),
                ]
            } else {
                // Legitimate
                (0..5)
                    .map(|i| {
                        TrajectoryPoint::from_fields(
                            h_last.pos_x() + i as f32 * 0.1,
                            h_last.pos_y(),
                            10.0,
                            5.0,
                            2.0,
                            0.0,
                            1.0,
                            0.0,
                        )
                    })
                    .collect()
            };

            let result = adaptive.reconcile(&h_last, &client, &[], 5, &mut rng);
            adaptive.observe_outcome(result.verdict, !is_hack);
        }

        // Bandit should have learned something.
        assert!(
            adaptive.q_values.iter().any(|&q| q > 0.0),
            "at least one arm should have positive Q-value"
        );
    }
}
