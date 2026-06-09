//! FoldBandit — self-tuning fold budget via Thompson sampling — Plan 195 T6.
//!
//! Selects the optimal fold budget (fraction of steps to keep) using a
//! multi-armed bandit with Thompson sampling. Each arm represents a
//! different fold aggressiveness level.

use crate::types::Rng;

/// Available fold budget arms.
const BUDGET_ARMS: [f32; 5] = [0.3, 0.5, 0.7, 0.9, 1.0];

/// Default arm index (0.7 = moderate folding).
const DEFAULT_ARM: usize = 2;

/// Bandit self-tuning for fold budget selection.
///
/// Uses Thompson sampling with Beta posteriors to balance exploration
/// (trying different fold budgets) and exploitation (using the best-known
/// budget). Reward = acceptance_rate * token_reduction_ratio.
#[derive(Debug, Clone)]
pub struct FoldBandit {
    /// Beta distribution α (successes) per arm.
    alphas: [f32; 5],
    /// Beta distribution β (failures) per arm.
    betas: [f32; 5],
    /// Total pulls per arm.
    pulls: [u32; 5],
    /// Decay factor for aging observations.
    decay: f32,
}

impl FoldBandit {
    /// Create a new fold bandit with uniform priors.
    pub fn new() -> Self {
        Self {
            alphas: [1.0; 5],
            betas: [1.0; 5],
            pulls: [0; 5],
            decay: 0.995,
        }
    }

    /// Create a fold bandit with a specific decay rate.
    pub fn with_decay(decay: f32) -> Self {
        Self {
            decay: decay.clamp(0.9, 1.0),
            ..Self::new()
        }
    }

    /// Select a fold budget using Thompson sampling.
    ///
    /// Samples from Beta(α, β) for each arm and returns the budget
    /// corresponding to the highest sample.
    pub fn select_budget(&mut self, rng: &mut Rng) -> f32 {
        let mut best_arm = DEFAULT_ARM;
        let mut best_sample = f32::NEG_INFINITY;

        for i in 0..5 {
            let sample = sample_beta(self.alphas[i], self.betas[i], rng);
            if sample > best_sample {
                best_sample = sample;
                best_arm = i;
            }
        }

        self.pulls[best_arm] += 1;
        BUDGET_ARMS[best_arm]
    }

    /// Record the reward signal for a chosen budget.
    ///
    /// - `accepted`: whether the fold verification passed.
    /// - `tokens_saved_ratio`: fraction of tokens saved (0.0–1.0).
    ///
    /// Reward combines acceptance and savings; penalty applies when
    /// verification fails.
    pub fn record_reward(&mut self, budget: f32, accepted: bool, tokens_saved_ratio: f32) {
        let arm = match budget_to_arm(budget) {
            Some(idx) => idx,
            None => return,
        };

        let reward = match accepted {
            true => tokens_saved_ratio.max(0.0),
            false => 0.0,
        };

        // Update Beta posterior.
        self.alphas[arm] += reward;
        self.betas[arm] += 1.0 - reward;

        // Apply decay to all arms to age out old observations.
        self.apply_decay();
    }

    /// Apply exponential decay to all arm observations.
    fn apply_decay(&mut self) {
        for i in 0..5 {
            self.alphas[i] = 1.0 + (self.alphas[i] - 1.0) * self.decay;
            self.betas[i] = 1.0 + (self.betas[i] - 1.0) * self.decay;
        }
    }

    /// Get total pulls across all arms.
    pub fn total_pulls(&self) -> u32 {
        self.pulls.iter().sum()
    }

    /// Get the arm index with the highest expected reward.
    pub fn best_arm(&self) -> usize {
        let mut best = DEFAULT_ARM;
        let mut best_mean = 0.0_f32;

        for i in 0..5 {
            let mean = self.alphas[i] / (self.alphas[i] + self.betas[i]);
            if mean > best_mean {
                best_mean = mean;
                best = i;
            }
        }

        best
    }

    /// Get the best fold budget based on observed rewards.
    pub fn best_budget(&self) -> f32 {
        BUDGET_ARMS[self.best_arm()]
    }

    /// Get the number of pulls for a specific arm.
    pub fn pulls(&self, arm: usize) -> u32 {
        self.pulls.get(arm).copied().unwrap_or(0)
    }

    /// Scale fold budget by precision confidence from BAKE embeddings.
    ///
    /// High confidence (precise embeddings) → can fold more aggressively → lower budget.
    /// Low confidence (uncertain embeddings) → keep more steps → higher budget.
    ///
    /// Bridge: `fold_budget = bandit.best_budget() * (1.0 - α * confidence)`
    /// where α controls how much precision influences folding (default 0.3).
    #[cfg(feature = "bake_precision")]
    pub fn precision_gated_budget(&self, precision_confidence: f32, alpha: f32) -> f32 {
        let base = self.best_budget();
        let gated = base * (1.0 - alpha * precision_confidence);
        gated.clamp(0.1, 1.0) // always keep at least 10%
    }
}

impl Default for FoldBandit {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a budget value to the nearest arm index.
fn budget_to_arm(budget: f32) -> Option<usize> {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for (i, &arm_budget) in BUDGET_ARMS.iter().enumerate() {
        let dist = (arm_budget - budget).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    Some(best_idx)
}

/// Sample from Beta(α, β) distribution using Jöhnk's algorithm.
fn sample_beta(alpha: f32, beta: f32, rng: &mut Rng) -> f32 {
    // Uniform prior: return uniform sample.
    if (alpha - 1.0).abs() < f32::EPSILON && (beta - 1.0).abs() < f32::EPSILON {
        return rng.uniform();
    }

    for _ in 0..256 {
        let u1 = rng.uniform().max(f32::EPSILON);
        let u2 = rng.uniform().max(f32::EPSILON);

        let x = u1.powf(1.0 / alpha);
        let y = u2.powf(1.0 / beta);

        if x + y <= 1.0 {
            return x / (x + y + f32::EPSILON);
        }
    }

    // Fallback: return mean estimate.
    alpha / (alpha + beta)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fold_bandit_new() {
        let bandit = FoldBandit::new();
        assert_eq!(bandit.total_pulls(), 0);
        // With uniform priors, best_arm returns whichever has highest mean.
        // All start equal, so the first arm wins.
        assert!(bandit.best_arm() < 5);
    }

    #[test]
    fn test_select_budget() {
        let mut bandit = FoldBandit::new();
        let mut rng = Rng::new(42);

        let budget = bandit.select_budget(&mut rng);
        assert!(BUDGET_ARMS.contains(&budget));
        assert_eq!(bandit.total_pulls(), 1);
    }

    #[test]
    fn test_record_reward_accepted() {
        let mut bandit = FoldBandit::new();
        bandit.record_reward(0.7, true, 0.5);

        // Arm 2 (budget 0.7) should have higher alpha.
        assert!(bandit.alphas[2] > 1.0);
    }

    #[test]
    fn test_record_reward_rejected() {
        let mut bandit = FoldBandit::new();
        bandit.record_reward(0.7, false, 0.5);

        // Rejection should increase beta for arm 2.
        assert!(bandit.betas[2] > 1.0);
    }

    #[test]
    fn test_convergence_to_best_arm() {
        let mut bandit = FoldBandit::new();
        let mut rng = Rng::new(123);

        // Simulate: budget 0.7 always succeeds with high savings.
        for _ in 0..100 {
            let budget = bandit.select_budget(&mut rng);
            let arm_idx = budget_to_arm(budget).unwrap();
            let accepted = arm_idx == 2;
            let savings = match arm_idx {
                2 => 0.5,
                _ => 0.1,
            };
            bandit.record_reward(budget, accepted, savings);
        }

        // After 100 episodes, arm 2 should dominate.
        assert_eq!(bandit.best_arm(), 2);
    }

    #[test]
    fn test_best_budget() {
        let mut bandit = FoldBandit::new();
        // Bias arm 4 (budget 1.0) heavily.
        for _ in 0..50 {
            bandit.record_reward(1.0, true, 0.9);
        }
        assert!((bandit.best_budget() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_budget_to_arm() {
        assert_eq!(budget_to_arm(0.3), Some(0));
        assert_eq!(budget_to_arm(0.5), Some(1));
        assert_eq!(budget_to_arm(0.7), Some(2));
        assert_eq!(budget_to_arm(0.9), Some(3));
        assert_eq!(budget_to_arm(1.0), Some(4));
        // Nearest match for non-exact values.
        assert_eq!(budget_to_arm(0.4), Some(0)); // closer to 0.3
        assert_eq!(budget_to_arm(0.6), Some(2)); // 0.7 slightly closer due to f32 rounding
    }

    #[test]
    fn test_sample_beta_bounds() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let sample = sample_beta(2.0, 5.0, &mut rng);
            assert!(
                (0.0..=1.0).contains(&sample),
                "Beta sample {sample} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_decay() {
        let mut bandit = FoldBandit::with_decay(0.9);
        bandit.alphas[0] = 10.0;
        bandit.betas[0] = 2.0;
        bandit.record_reward(0.3, true, 0.5);

        // After decay, alpha should be between 1.0 and 10.0.
        assert!(bandit.alphas[0] > 1.0);
        assert!(bandit.alphas[0] < 10.0);
    }

    #[test]
    fn test_pulls_tracking() {
        let mut bandit = FoldBandit::new();
        let mut rng = Rng::new(42);

        bandit.select_budget(&mut rng);
        bandit.select_budget(&mut rng);
        bandit.select_budget(&mut rng);

        assert_eq!(bandit.total_pulls(), 3);
    }

    #[test]
    fn test_default() {
        let bandit = FoldBandit::default();
        assert_eq!(bandit.total_pulls(), 0);
    }

    // ── Precision-Gated Budget Tests (Plan 236) ──────────────────────

    #[cfg(feature = "bake_precision")]
    #[test]
    fn test_precision_gated_budget_zero_confidence_returns_base() {
        let bandit = FoldBandit::new();
        let base = bandit.best_budget();
        let gated = bandit.precision_gated_budget(0.0, 0.3);
        assert!((gated - base).abs() < f32::EPSILON,
            "confidence=0 should return base budget, got {gated} vs {base}");
    }

    #[cfg(feature = "bake_precision")]
    #[test]
    fn test_precision_gated_budget_full_confidence_reduces_budget() {
        let bandit = FoldBandit::new();
        let base = bandit.best_budget();
        let gated = bandit.precision_gated_budget(1.0, 0.3);
        assert!(gated < base,
            "confidence=1 should reduce budget, got {gated} >= {base}");
    }

    #[cfg(feature = "bake_precision")]
    #[test]
    fn test_precision_gated_budget_clamps_to_minimum() {
        // Bias arm 4 (budget 1.0) so base is 1.0, then extreme alpha+confidence
        let mut bandit = FoldBandit::new();
        for _ in 0..50 {
            bandit.record_reward(1.0, true, 0.9);
        }
        assert!((bandit.best_budget() - 1.0).abs() < f32::EPSILON);
        let gated = bandit.precision_gated_budget(1.0, 10.0);
        assert!((gated - 0.1).abs() < f32::EPSILON,
            "should clamp to 0.1 minimum, got {gated}");
    }

    #[cfg(feature = "bake_precision")]
    #[test]
    fn test_precision_gated_budget_zero_alpha_returns_base() {
        let bandit = FoldBandit::new();
        let base = bandit.best_budget();
        let gated = bandit.precision_gated_budget(0.8, 0.0);
        assert!((gated - base).abs() < f32::EPSILON,
            "alpha=0 should return base budget unchanged, got {gated} vs {base}");
    }
}
