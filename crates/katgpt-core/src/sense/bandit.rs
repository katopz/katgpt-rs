//! SenseBandit — trial log for sense module quality feedback.

use crate::types::{SenseKind, SenseModule};

/// A single sense trial for bandit feedback.
#[derive(Clone, Debug)]
pub struct SenseTrial {
    pub sense_kind: SenseKind,
    pub activation: f32,
    pub reward: f32,
    pub npc_id: u32,
    pub action_taken: u32,
}

/// Trial log for sense module self-learning.
#[derive(Clone, Debug, Default)]
pub struct SenseTrialLog {
    pub trials: Vec<SenseTrial>,
}

impl SenseTrialLog {
    pub fn record(&mut self, trial: SenseTrial) {
        self.trials.push(trial);
    }

    /// Compute average reward for a sense kind — single pass, zero allocation.
    pub fn average_reward(&self, kind: SenseKind) -> f32 {
        let (sum, count) = self
            .trials
            .iter()
            .filter(|t| t.sense_kind == kind)
            .fold((0.0f32, 0usize), |(s, c), t| (s + t.reward, c + 1));
        if count == 0 { 0.0 } else { sum / count as f32 }
    }
}

/// Compute exploration-weighted reward for sense trial.
/// Dimensions with low precision get boosted exploration reward.
#[cfg(feature = "bake_precision")]
pub fn precision_weighted_reward(
    base_reward: f32,
    precision: &[f32; 8],
    activated_dims: &[usize],
) -> f32 {
    if activated_dims.is_empty() {
        return base_reward;
    }
    let avg_priority: f32 = activated_dims
        .iter()
        .map(|&d| crate::sense::bake::exploration_priority(precision, d))
        .sum::<f32>()
        / activated_dims.len() as f32;
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
        let mut module = SenseModule::default();
        module.confidence = 0.3;
        module.commit();

        let trial = SenseTrial {
            npc_id: 1,
            sense_kind: module.kind,
            activation: 0.5,
            action_taken: 0,
            reward: 0.9,
        };
        decay_direction(&mut module, &trial, 0.5);
        assert!(module.confidence > 0.3);
    }

    #[test]
    fn test_low_reward_decreases_confidence() {
        let mut module = SenseModule::default();
        module.confidence = 0.8;
        module.commit();

        let trial = SenseTrial {
            npc_id: 1,
            sense_kind: module.kind,
            activation: 0.5,
            action_taken: 0,
            reward: 0.1,
        };
        decay_direction(&mut module, &trial, 0.5);
        assert!(module.confidence < 0.8);
    }

    #[test]
    fn test_average_reward_zero_alloc() {
        let mut log = SenseTrialLog::default();
        log.record(SenseTrial {
            npc_id: 1,
            sense_kind: SenseKind::FighterSense,
            activation: 0.5,
            action_taken: 0,
            reward: 0.8,
        });
        log.record(SenseTrial {
            npc_id: 2,
            sense_kind: SenseKind::FighterSense,
            activation: 0.3,
            action_taken: 1,
            reward: 0.6,
        });
        log.record(SenseTrial {
            npc_id: 3,
            sense_kind: SenseKind::SpatialSense,
            activation: 0.4,
            action_taken: 0,
            reward: 0.2,
        });

        let avg = log.average_reward(SenseKind::FighterSense);
        assert!((avg - 0.7).abs() < 1e-6);
        assert!((log.average_reward(SenseKind::SpatialSense) - 0.2).abs() < 1e-6);
        assert!((log.average_reward(SenseKind::CommonSense)).abs() < 1e-6);
    }
}
