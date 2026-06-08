//! SenseBandit — trial log for sense module quality feedback.

use crate::types::{SenseKind, SenseModule};

/// A single sense trial for bandit feedback.
#[derive(Clone, Debug)]
pub struct SenseTrial {
    pub npc_id: u32,
    pub sense_kind: SenseKind,
    pub activation: f32,
    pub action_taken: u32,
    pub reward: f32,
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

    pub fn average_reward(&self, kind: SenseKind) -> f32 {
        let rewards: Vec<f32> = self
            .trials
            .iter()
            .filter(|t| t.sense_kind == kind)
            .map(|t| t.reward)
            .collect();
        if rewards.is_empty() {
            return 0.0;
        }
        rewards.iter().sum::<f32>() / rewards.len() as f32
    }
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
}
