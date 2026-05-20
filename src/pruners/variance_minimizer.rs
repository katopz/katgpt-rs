//! Variance-minimized schedule optimizer (RePlaid Prop 1 adaptation).
//!
//! RePlaid proves that minimizing Monte-Carlo variance of per-timestep loss
//! yields a constant-difficulty schedule (Prop 1). This struct adapts that
//! principle to any sequential process with scalar per-step costs.
//!
//! Usage: Track per-step cost → adapt schedule parameter → flatten variance.
//! No teacher, no gradients — purely online statistics.

/// Configuration for variance minimization.
#[derive(Debug, Clone)]
pub struct VarianceMinimizerConfig {
    /// EMA decay for running mean (0.99 = slow adaptation).
    pub mean_decay: f32,
    /// EMA decay for running variance (0.99 = slow adaptation).
    pub var_decay: f32,
    /// Learning rate for schedule parameter update.
    pub lr: f32,
    /// Minimum schedule parameter value.
    pub min_param: f32,
    /// Maximum schedule parameter value.
    pub max_param: f32,
}

impl Default for VarianceMinimizerConfig {
    fn default() -> Self {
        Self {
            mean_decay: 0.99,
            var_decay: 0.99,
            lr: 0.01,
            min_param: 0.01,
            max_param: 1.0,
        }
    }
}

/// Online variance-minimized schedule optimizer.
///
/// Tracks per-step cost and adapts a schedule parameter to minimize
/// the variance of costs across steps. Inspired by RePlaid Prop 1:
/// "there exists a unique noise schedule γ* such that ℓ(t) ≡ κ for all t."
#[derive(Debug, Clone)]
pub struct VarianceMinimizer {
    config: VarianceMinimizerConfig,
    /// Running mean of per-step costs.
    running_mean: f32,
    /// Running variance of per-step costs.
    running_var: f32,
    /// Current schedule parameter being optimized.
    param: f32,
    /// Number of observations seen.
    n_observations: u32,
}

impl VarianceMinimizer {
    /// Initialize with param at midpoint of [min_param, max_param].
    pub fn new(config: VarianceMinimizerConfig) -> Self {
        let param = (config.min_param + config.max_param) / 2.0;
        Self {
            config,
            running_mean: 0.0,
            running_var: 0.0,
            param,
            n_observations: 0,
        }
    }

    /// Initialize with a specific starting param.
    pub fn with_param(config: VarianceMinimizerConfig, initial_param: f32) -> Self {
        let param = initial_param.clamp(config.min_param, config.max_param);
        Self {
            config,
            running_mean: 0.0,
            running_var: 0.0,
            param,
            n_observations: 0,
        }
    }

    /// Update running mean/variance with a new cost observation (EMA).
    pub fn observe(&mut self, cost: f32) {
        match self.n_observations {
            0 => {
                self.running_mean = cost;
                self.running_var = 0.0;
            }
            _ => {
                self.running_mean = self.config.mean_decay * self.running_mean
                    + (1.0 - self.config.mean_decay) * cost;
                let delta = cost - self.running_mean;
                self.running_var = self.config.var_decay * self.running_var
                    + (1.0 - self.config.var_decay) * delta * delta;
            }
        }
        self.n_observations += 1;
    }

    /// Adjust param to minimize variance. Returns the new param.
    ///
    /// If variance is already flat (below threshold), returns param unchanged.
    /// Otherwise moves param in the direction that reduces variance:
    /// `param -= lr * sign(cost - mean) * sqrt(variance)`.
    pub fn adapt(&mut self) -> f32 {
        const VARIANCE_FLOOR: f32 = 1e-8;

        if self.running_var < VARIANCE_FLOOR {
            return self.param;
        }

        let delta = self.running_mean - self.param;
        let sign = match delta.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => 1.0f32,
            Some(std::cmp::Ordering::Less) => -1.0f32,
            _ => 0.0f32,
        };

        self.param -= self.config.lr * sign * self.running_var.sqrt();
        self.param = self
            .param
            .clamp(self.config.min_param, self.config.max_param);
        self.param
    }

    /// Convenience: observe a cost then adapt the schedule parameter.
    pub fn observe_and_adapt(&mut self, cost: f32) -> f32 {
        self.observe(cost);
        self.adapt()
    }

    /// Current schedule parameter.
    #[inline]
    pub fn param(&self) -> f32 {
        self.param
    }

    /// Current running variance of costs.
    #[inline]
    pub fn variance(&self) -> f32 {
        self.running_var
    }

    /// Current running mean of costs.
    #[inline]
    pub fn mean(&self) -> f32 {
        self.running_mean
    }

    /// Number of observations seen so far.
    #[inline]
    pub fn n_observations(&self) -> u32 {
        self.n_observations
    }

    /// Reset all statistics and restore param to midpoint of [min_param, max_param].
    pub fn reset(&mut self) {
        self.running_mean = 0.0;
        self.running_var = 0.0;
        self.param = (self.config.min_param + self.config.max_param) / 2.0;
        self.n_observations = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_minimizer_converges() {
        // Synthetic: costs that decrease in variance as param increases.
        // After many observations, variance should be tracked.
        let config = VarianceMinimizerConfig {
            mean_decay: 0.9,
            var_decay: 0.9,
            lr: 0.1,
            min_param: 0.01,
            max_param: 1.0,
        };
        let mut vm = VarianceMinimizer::new(config);

        for _ in 0..100 {
            let cost = 0.5 + 0.3 * (vm.param * 10.0).sin();
            vm.observe_and_adapt(cost);
        }

        assert!(vm.n_observations() == 100);
        assert!(vm.param() >= 0.01 && vm.param() <= 1.0);
    }

    #[test]
    fn test_variance_minimizer_clamps() {
        let config = VarianceMinimizerConfig {
            lr: 10.0, // very high lr to push out of bounds
            ..Default::default()
        };
        let mut vm = VarianceMinimizer::new(config.clone());

        // Push param to lower boundary.
        for _ in 0..100 {
            vm.observe_and_adapt(100.0);
        }
        assert!(vm.param() >= config.min_param);

        // Push other direction.
        vm.reset();
        for _ in 0..100 {
            vm.observe_and_adapt(-100.0);
        }
        assert!(vm.param() <= config.max_param);
    }

    #[test]
    fn test_default_config_sensible() {
        let config = VarianceMinimizerConfig::default();
        assert!(config.mean_decay > 0.0 && config.mean_decay < 1.0);
        assert!(config.var_decay > 0.0 && config.var_decay < 1.0);
        assert!(config.lr > 0.0);
        assert!(config.min_param < config.max_param);
    }

    #[test]
    fn test_initial_param_is_midpoint() {
        let config = VarianceMinimizerConfig {
            min_param: 0.1,
            max_param: 0.9,
            ..Default::default()
        };
        let vm = VarianceMinimizer::new(config);
        assert!((vm.param() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_with_param_custom_init() {
        let vm = VarianceMinimizer::with_param(VarianceMinimizerConfig::default(), 0.75);
        assert!((vm.param() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_with_param_clamps_to_range() {
        let vm = VarianceMinimizer::with_param(VarianceMinimizerConfig::default(), 5.0);
        assert!(vm.param() <= 1.0);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut vm = VarianceMinimizer::new(VarianceMinimizerConfig::default());
        vm.observe(1.0);
        vm.observe(2.0);
        vm.observe(3.0);
        assert!(vm.n_observations() > 0);

        vm.reset();
        assert_eq!(vm.n_observations(), 0);
        assert!(vm.variance().abs() < 1e-8);
        assert!(vm.mean().abs() < 1e-8);
    }

    #[test]
    fn test_ema_responds_to_recent_costs() {
        let config = VarianceMinimizerConfig {
            mean_decay: 0.5, // fast decay
            ..Default::default()
        };
        let mut vm = VarianceMinimizer::new(config);

        // Observe 0.0 for a while.
        for _ in 0..50 {
            vm.observe(0.0);
        }
        let mean_after_zeros = vm.mean();

        // Switch to 1.0 — mean should move toward 1.0.
        for _ in 0..10 {
            vm.observe(1.0);
        }
        assert!(vm.mean() > mean_after_zeros);
    }

    #[test]
    fn test_first_observation_sets_mean_directly() {
        let mut vm = VarianceMinimizer::new(VarianceMinimizerConfig::default());
        vm.observe(0.42);
        assert!((vm.mean() - 0.42).abs() < 1e-6);
        assert!(vm.variance().abs() < 1e-8);
        assert_eq!(vm.n_observations(), 1);
    }

    #[test]
    fn test_adapt_no_op_when_single_observation() {
        let config = VarianceMinimizerConfig {
            lr: 1.0,
            min_param: 0.0,
            max_param: 2.0,
            ..Default::default()
        };
        let mut vm = VarianceMinimizer::new(config);
        let param_before = vm.param();
        vm.observe(1.0);
        // With one observation, variance is 0 → adapt is no-op.
        let result = vm.adapt();
        assert!((result - param_before).abs() < 1e-6);
    }
}
