//! SenseBandit — trial log for sense module quality feedback.

use crate::types::{SenseKind, SenseModule};

/// Number of SenseKind variants tracked by the aggregate table.
const AGGREGATE_KINDS: usize = 8;

/// Per-kind aggregate for O(1) average_reward.
#[derive(Clone, Copy, Debug, Default)]
struct KindAggregate {
    sum: f32,
    count: u32,
}

/// A single sense trial for bandit feedback.
#[derive(Clone, Debug)]
pub struct SenseTrial {
    pub npc_id: u32,
    pub action_taken: u32,
    pub activation: f32,
    pub reward: f32,
    pub sense_kind: SenseKind,
}

/// Trial log for sense module self-learning.
/// Maintains per-kind aggregates for O(1) average_reward queries.
#[derive(Clone, Debug, Default)]
pub struct SenseTrialLog {
    pub trials: Vec<SenseTrial>,
    /// Per-kind running aggregates — O(1) average_reward by kind.
    aggregates: [KindAggregate; AGGREGATE_KINDS],
}

impl SenseTrialLog {
    /// Create a new trial log with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            trials: Vec::with_capacity(capacity),
            aggregates: [KindAggregate::default(); AGGREGATE_KINDS],
        }
    }

    pub fn record(&mut self, trial: SenseTrial) {
        let idx = trial.sense_kind as usize;
        if idx < AGGREGATE_KINDS {
            self.aggregates[idx].sum += trial.reward;
            self.aggregates[idx].count += 1;
        }
        self.trials.push(trial);
    }

    /// Compute average reward for a sense kind — O(1) via pre-computed aggregates.
    #[inline]
    pub fn average_reward(&self, kind: SenseKind) -> f32 {
        let idx = kind as usize;
        if idx < AGGREGATE_KINDS {
            let agg = &self.aggregates[idx];
            if agg.count == 0 {
                0.0
            } else {
                agg.sum / agg.count as f32
            }
        } else {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for t in &self.trials {
                if t.sense_kind == kind {
                    sum += t.reward;
                    count += 1;
                }
            }
            if count == 0 { 0.0 } else { sum / count as f32 }
        }
    }
}

/// Compute exploration-weighted reward for sense trial.
/// Dimensions with low precision get boosted exploration reward.
/// Pre-computes max_lambda once — O(8) scan instead of O(8×N).
#[cfg(feature = "bake_precision")]
pub fn precision_weighted_reward(
    base_reward: f32,
    precision: &[f32; 8],
    activated_dims: &[usize],
) -> f32 {
    if activated_dims.is_empty() {
        return base_reward;
    }
    let max_lam = crate::sense::bake::max_lambda(precision);
    let mut sum = 0.0f32;
    for &d in activated_dims {
        sum += crate::sense::bake::exploration_priority_with_max(precision, d, max_lam);
    }
    let avg_priority = sum / activated_dims.len() as f32;
    base_reward * (1.0 + avg_priority)
}

/// EMA weight update for direction decay based on bandit feedback.
/// Never updates GM-pinned modules.
pub fn decay_direction(module: &mut SenseModule, trial: &SenseTrial, alpha: f32) {
    if module.kind != trial.sense_kind {
        return;
    }
    // Adjust confidence via EMA
    let target = trial.reward;
    module.confidence = alpha * module.confidence + (1.0 - alpha) * target;
    module.commit(); // re-commit with new confidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_reward_increases_confidence() {
        let mut module = SenseModule {
            confidence: 0.3,
            ..Default::default()
        };
        module.commit();

        let trial = SenseTrial {
            npc_id: 1,
            action_taken: 0,
            activation: 0.5,
            reward: 0.9,
            sense_kind: module.kind,
        };
        decay_direction(&mut module, &trial, 0.5);
        assert!(module.confidence > 0.3);
    }

    #[test]
    fn test_low_reward_decreases_confidence() {
        let mut module = SenseModule {
            confidence: 0.8,
            ..Default::default()
        };
        module.commit();

        let trial = SenseTrial {
            npc_id: 1,
            action_taken: 0,
            activation: 0.5,
            reward: 0.1,
            sense_kind: module.kind,
        };
        decay_direction(&mut module, &trial, 0.5);
        assert!(module.confidence < 0.8);
    }

    #[test]
    fn test_average_reward_zero_alloc() {
        let mut log = SenseTrialLog::default();
        log.record(SenseTrial {
            npc_id: 1,
            action_taken: 0,
            activation: 0.5,
            reward: 0.8,
            sense_kind: SenseKind::FighterSense,
        });
        log.record(SenseTrial {
            npc_id: 2,
            action_taken: 1,
            activation: 0.3,
            reward: 0.6,
            sense_kind: SenseKind::FighterSense,
        });
        log.record(SenseTrial {
            npc_id: 3,
            action_taken: 0,
            activation: 0.4,
            reward: 0.2,
            sense_kind: SenseKind::SpatialSense,
        });

        let avg = log.average_reward(SenseKind::FighterSense);
        assert!((avg - 0.7).abs() < 1e-6);
        assert!((log.average_reward(SenseKind::SpatialSense) - 0.2).abs() < 1e-6);
        assert!((log.average_reward(SenseKind::CommonSense)).abs() < 1e-6);
    }
}
