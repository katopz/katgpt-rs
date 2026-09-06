//! Bandit environments (extracted from mod.rs by Issue 177).
//!
//! BanditEnv trait + BernoulliEnv + GaussianEnv — the test/training harness
//! for bandit arm-selection strategies.

use katgpt_types::Rng;

// ── Environment ─────────────────────────────────────────────────

/// A multi-armed bandit environment that generates stochastic rewards.
///
/// Each arm has a hidden reward distribution. The agent's goal is to
/// identify the arm with the highest expected reward while minimizing
/// cumulative regret.
pub trait BanditEnv: Send + Sync {
    /// Pull an arm and receive a stochastic reward in [0.0, 1.0].
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32;

    /// Expected (mean) reward for a specific arm.
    fn expected_reward(&self, arm: usize) -> f32;

    /// Expected reward of the optimal arm.
    fn optimal_reward(&self) -> f32;

    /// Number of arms.
    fn num_arms(&self) -> usize;

    /// Index of the optimal arm (highest expected reward).
    fn optimal_arm(&self) -> usize;
}

// ── Bernoulli Environment ───────────────────────────────────────

/// Bernoulli bandit: each arm returns 1.0 with probability `p`, 0.0 otherwise.
///
/// Classic MAB setting. Optimal for Thompson Sampling with Beta posteriors.
#[derive(Clone)]
pub struct BernoulliEnv {
    probs: Vec<f32>,
    optimal_arm: usize,
    optimal_reward: f32,
}

impl BernoulliEnv {
    /// Create a Bernoulli bandit with per-arm success probabilities.
    pub fn new(probs: &[f32]) -> Self {
        let optimal_arm = probs
            .iter()
            .enumerate()
            // float_order: a NaN prob must never define the optimal arm.
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i);
        let optimal_reward = probs[optimal_arm];
        Self {
            probs: probs.to_vec(),
            optimal_arm,
            optimal_reward,
        }
    }

    /// Success probability for each arm.
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }
}

impl BanditEnv for BernoulliEnv {
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32 {
        if arm >= self.probs.len() || rng.uniform() >= self.probs[arm] {
            0.0
        } else {
            1.0
        }
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        self.probs.get(arm).copied().unwrap_or(0.0)
    }

    #[inline]
    fn optimal_reward(&self) -> f32 {
        self.optimal_reward
    }

    fn num_arms(&self) -> usize {
        self.probs.len()
    }

    #[inline]
    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}

// ── Gaussian Environment ────────────────────────────────────────

/// Gaussian bandit: each arm returns a reward sampled from N(μ, σ²).
///
/// Rewards are clamped to [0.0, 1.0]. Useful for continuous reward settings.
#[derive(Clone)]
pub struct GaussianEnv {
    means: Vec<f32>,
    std: f32,
    optimal_arm: usize,
    optimal_reward: f32,
}

impl GaussianEnv {
    /// Create a Gaussian bandit with per-arm means and shared standard deviation.
    pub fn new(means: &[f32], std: f32) -> Self {
        let optimal_arm = means
            .iter()
            .enumerate()
            // float_order: a NaN mean must never define the optimal arm.
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i);
        let optimal_reward = means[optimal_arm];
        Self {
            means: means.to_vec(),
            std,
            optimal_arm,
            optimal_reward,
        }
    }

    /// Mean reward for each arm.
    pub fn means(&self) -> &[f32] {
        &self.means
    }
}

impl BanditEnv for GaussianEnv {
    fn pull(&self, arm: usize, rng: &mut Rng) -> f32 {
        if arm >= self.means.len() {
            return 0.0;
        }
        // Box-Muller transform for Gaussian sampling
        let u1 = rng.uniform().max(f32::EPSILON);
        let u2 = rng.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        (self.means[arm] + self.std * z).clamp(0.0, 1.0)
    }

    fn expected_reward(&self, arm: usize) -> f32 {
        self.means.get(arm).copied().unwrap_or(0.0)
    }

    #[inline]
    fn optimal_reward(&self) -> f32 {
        self.optimal_reward
    }

    fn num_arms(&self) -> usize {
        self.means.len()
    }

    #[inline]
    fn optimal_arm(&self) -> usize {
        self.optimal_arm
    }
}
