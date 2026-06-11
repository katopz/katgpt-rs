//! `MuxBanditWidth` — adaptive superposition width via multi-armed bandit.
//!
//! Treats each candidate width (1..=K) as a bandit arm and selects the
//! expansion width that has historically yielded the best reward.

/// A single arm in the bandit, representing a candidate width.
#[derive(Debug, Clone)]
struct Arm {
    width: usize,
    total_reward: f32,
    pulls: u32,
}

impl Arm {
    fn new(width: usize) -> Self {
        Self {
            width,
            total_reward: 0.0,
            pulls: 0,
        }
    }

    #[inline]
    fn update(&mut self, reward: f32) {
        self.total_reward += reward;
        self.pulls += 1;
    }
}

/// Bandit-based width selector for MUX superposition expansion.
#[derive(Debug, Clone)]
pub struct MuxBanditWidth {
    arms: Vec<Arm>,
    /// Exploration factor (higher = more exploration).
    pub exploration: f32,
}

impl MuxBanditWidth {
    /// Create a new bandit with arms for widths 1..=k.
    pub fn new(k: usize) -> Self {
        let arms = (1..=k).map(Arm::new).collect();
        Self {
            arms,
            exploration: 1.0,
        }
    }

    /// Select the best width arm using upper confidence bound.
    pub fn select_width(&self, total_steps: u32) -> usize {
        let ln_n = if total_steps > 0 {
            (total_steps as f32).ln()
        } else {
            1.0
        };

        let mut best_width = 1;
        let mut best_ucb = f32::NEG_INFINITY;
        for arm in &self.arms {
            let ucb = if arm.pulls == 0 {
                f32::INFINITY
            } else {
                let avg = arm.total_reward / arm.pulls as f32;
                avg + self.exploration * (2.0 * ln_n / arm.pulls as f32).sqrt()
            };
            if ucb > best_ucb {
                best_ucb = ucb;
                best_width = arm.width;
            }
        }
        best_width
    }

    /// Update the arm for `width` with the observed `reward`.
    pub fn update(&mut self, width: usize, reward: f32) {
        if width >= 1 && width <= self.arms.len() {
            // Arms are created as (1..=k), so arm index = width - 1.
            self.arms[width - 1].update(reward);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_bandit_has_k_arms() {
        let bandit = MuxBanditWidth::new(4);
        assert_eq!(bandit.arms.len(), 4);
    }

    #[test]
    fn selects_unexplored_arm_first() {
        let bandit = MuxBanditWidth::new(4);
        // With no pulls, should return some width (all are unexplored)
        let width = bandit.select_width(0);
        assert!((1..=4).contains(&width));
    }

    #[test]
    fn update_and_select() {
        let mut bandit = MuxBanditWidth::new(4);
        // Pull all arms so none are unexplored, but give high reward to width 2
        bandit.update(2, 10.0);
        bandit.update(2, 10.0);
        bandit.update(1, 0.1);
        bandit.update(3, 0.1);
        bandit.update(4, 0.1);
        // Width 2 should dominate after equal pulls
        let width = bandit.select_width(5);
        assert_eq!(width, 2);
    }
}
