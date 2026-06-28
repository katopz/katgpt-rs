//! Simulation Gate — routes between analytical and simulation-based evaluation.
//!
//! Wolfram's key insight: when a game is computationally reducible (low irreducibility),
//! analytical shortcuts exist and expensive simulation can be skipped. When irreducible,
//! full simulation (bandit/MCTS/rollout) is required.
//!
//! This module provides [`SimulationGate`] which wraps [`IrreducibilityGate`] to make
//! routing decisions: `reducible → shortcut`, `irreducible → full simulation`.
//!
//! Plan 188 Phase 4 integration.

use crate::ruliology::irreducibility::IrreducibilityGate;
use crate::ruliology::types::WinMatrix;

// ── SimulationStrategy ─────────────────────────────────────────────

/// Decision made by the simulation gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimulationStrategy {
    /// Game is reducible — use analytical shortcut (e.g., Nash equilibrium lookup).
    /// Skip expensive simulation. The cached payoff matrix is sufficient.
    AnalyticalShortcut,
    /// Game is irreducible — full simulation required.
    /// Use bandit/MCTS/rollout for accurate payoff estimation.
    FullSimulation,
    /// Borderline — game has moderate irreducibility.
    /// Use lightweight simulation (fewer rounds) as a compromise.
    LightweightSimulation,
}

// ── SimulationGateConfig ──────────────────────────────────────────

/// Configuration for the simulation gate.
#[derive(Debug, Clone)]
pub struct SimulationGateConfig {
    /// Compression ratio threshold for "definitely reducible" (skip simulation).
    /// Below this → AnalyticalShortcut.
    pub reducible_threshold: f32,
    /// Compression ratio threshold for "definitely irreducible" (full simulation).
    /// Above this → FullSimulation.
    pub irreducible_threshold: f32,
    /// Number of rounds for lightweight simulation (between analytical and full).
    pub lightweight_rounds: u32,
    /// Number of rounds for full simulation.
    pub full_rounds: u32,
}

impl Default for SimulationGateConfig {
    fn default() -> Self {
        Self {
            reducible_threshold: 0.3,
            irreducible_threshold: 0.7,
            lightweight_rounds: 20,
            full_rounds: 100,
        }
    }
}

// ── SimulationGateResult ──────────────────────────────────────────

/// Result of simulation gate analysis.
#[derive(Debug, Clone)]
pub struct SimulationGateResult {
    /// Recommended simulation strategy.
    pub strategy: SimulationStrategy,
    /// Recommended number of tournament rounds.
    pub recommended_rounds: u32,
    /// Compression ratio from irreducibility analysis.
    pub compression_ratio: f32,
    /// Whether the game is computationally irreducible.
    pub is_irreducible: bool,
    /// Mean absolute payoff (indicator of game dynamics).
    pub mean_abs_payoff: f64,
    /// Payoff variance (high variance = complex dynamics).
    pub payoff_variance: f64,
}

// ── SimulationGate ────────────────────────────────────────────────

/// Gate that routes between analytical and simulation-based evaluation.
///
/// Uses [`IrreducibilityGate`] as the underlying irreducibility detector,
/// then maps the compression ratio to a [`SimulationStrategy`]:
///
/// - `ratio < reducible_threshold` → [`AnalyticalShortcut`](SimulationStrategy::AnalyticalShortcut)
/// - `ratio ≥ irreducible_threshold` → [`FullSimulation`](SimulationStrategy::FullSimulation)
/// - Otherwise → [`LightweightSimulation`](SimulationStrategy::LightweightSimulation)
///
/// # Usage
///
/// ```rust,ignore
/// use katgpt_rs::ruliology::{FsmEnumerator, matching_pennies, SimulationGate};
///
/// let strategies = FsmEnumerator::enumerate(2);
/// let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
/// let gate = SimulationGate::default();
/// let result = gate.route(&matrix);
///
/// match result.strategy {
///     SimulationStrategy::AnalyticalShortcut => { /* skip simulation */ }
///     SimulationStrategy::LightweightSimulation => { /* use 20 rounds */ }
///     SimulationStrategy::FullSimulation => { /* use 100 rounds */ }
/// }
/// ```
pub struct SimulationGate {
    /// Inner irreducibility gate.
    irreducibility: IrreducibilityGate,
    /// Routing configuration.
    config: SimulationGateConfig,
}

impl Default for SimulationGate {
    /// Default gate with default thresholds.
    fn default() -> Self {
        Self::new(SimulationGateConfig::default())
    }
}

impl SimulationGate {
    /// Create a new simulation gate with the given config.
    pub fn new(config: SimulationGateConfig) -> Self {
        let irreducibility = IrreducibilityGate::new(config.irreducible_threshold);
        Self {
            irreducibility,
            config,
        }
    }

    /// Analyze a win matrix and recommend a simulation strategy.
    ///
    /// Returns the recommended strategy, number of rounds, and irreducibility metrics.
    pub fn route(&self, matrix: &WinMatrix) -> SimulationGateResult {
        let result = self.irreducibility.analyze(matrix);

        let (strategy, recommended_rounds) =
            if result.compression_ratio < self.config.reducible_threshold {
                // Low irreducibility → game has predictable structure.
                // Analytical shortcuts exist: skip expensive simulation.
                (SimulationStrategy::AnalyticalShortcut, 0)
            } else if result.compression_ratio >= self.config.irreducible_threshold {
                // High irreducibility → must simulate.
                // Full bandit/MCTS/rollout required.
                (SimulationStrategy::FullSimulation, self.config.full_rounds)
            } else {
                // Borderline → lightweight simulation as compromise.
                (
                    SimulationStrategy::LightweightSimulation,
                    self.config.lightweight_rounds,
                )
            };

        SimulationGateResult {
            strategy,
            recommended_rounds,
            compression_ratio: result.compression_ratio,
            is_irreducible: result.is_irreducible,
            mean_abs_payoff: result.mean_abs_payoff,
            payoff_variance: result.payoff_variance,
        }
    }

    /// Quick check: should we skip simulation?
    ///
    /// Returns `true` if the game is reducible and analytical shortcuts are available.
    pub fn should_skip_simulation(&self, matrix: &WinMatrix) -> bool {
        let result = self.irreducibility.analyze(matrix);
        result.compression_ratio < self.config.reducible_threshold
    }

    /// Quick check: do we need full simulation?
    ///
    /// Returns `true` if the game is irreducible and full bandit/MCTS is required.
    pub fn needs_full_simulation(&self, matrix: &WinMatrix) -> bool {
        let result = self.irreducibility.analyze(matrix);
        result.compression_ratio >= self.config.irreducible_threshold
    }

    /// Get the routing configuration.
    pub fn config(&self) -> &SimulationGateConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliology::{FsmEnumerator, matching_pennies, prisoners_dilemma};

    #[test]
    fn test_simulation_gate_matching_pennies_reducible() {
        let strategies = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
        let gate = SimulationGate::default();
        let result = gate.route(&matrix);

        // Matching pennies with 2-state FSMs has moderate irreducibility → not full simulation.
        assert!(
            matches!(
                result.strategy,
                SimulationStrategy::AnalyticalShortcut | SimulationStrategy::LightweightSimulation
            ),
            "matching pennies should be reducible or lightweight, got {:?} (ratio={:.4})",
            result.strategy,
            result.compression_ratio
        );
        // Verify it's not recommending full simulation.
        assert!(
            result.recommended_rounds < 100,
            "should not recommend full simulation rounds"
        );
    }

    #[test]
    fn test_simulation_gate_pd_reducible() {
        let strategies = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&strategies, 100, &|a, b| prisoners_dilemma(a, b).0);
        let gate = SimulationGate::default();
        let result = gate.route(&matrix);

        // PD with 2-state FSMs is very structured → shortcut.
        assert!(
            matches!(result.strategy, SimulationStrategy::AnalyticalShortcut),
            "PD should be reducible, got {:?} (ratio={:.4})",
            result.strategy,
            result.compression_ratio
        );
    }

    #[test]
    fn test_simulation_gate_random_irreducible() {
        // Build a matrix with pseudo-random payoffs.
        let n = 22;
        let mut payoffs = Vec::with_capacity(n);
        let mut state: u64 = 42;
        for _ in 0..n {
            let mut row = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                row.push(val);
            }
            payoffs.push(row);
        }

        let ids: Vec<u64> = (0..n as u64).collect();
        let matrix = WinMatrix::new(payoffs, ids);
        let gate = SimulationGate::default();
        let result = gate.route(&matrix);

        assert!(
            matches!(result.strategy, SimulationStrategy::FullSimulation),
            "random matrix should require full simulation, got {:?} (ratio={:.4})",
            result.strategy,
            result.compression_ratio
        );
        assert_eq!(result.recommended_rounds, 100);
    }

    #[test]
    fn test_simulation_gate_uniform_reducible() {
        let n = 10;
        let payoffs = vec![vec![0.5; n]; n];
        let ids: Vec<u64> = (0..n as u64).collect();
        let matrix = WinMatrix::new(payoffs, ids);

        let gate = SimulationGate::default();
        let result = gate.route(&matrix);

        assert!(
            matches!(result.strategy, SimulationStrategy::AnalyticalShortcut),
            "uniform matrix should be shortcut, got {:?}",
            result.strategy
        );
    }

    #[test]
    fn test_should_skip_simulation() {
        // PD with 2-state FSMs is very structured → should skip.
        let strategies = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&strategies, 100, &|a, b| prisoners_dilemma(a, b).0);
        let gate = SimulationGate::default();

        assert!(
            gate.should_skip_simulation(&matrix),
            "PD should be reducible"
        );
    }

    #[test]
    fn test_needs_full_simulation_random() {
        let n = 22;
        let mut payoffs = Vec::with_capacity(n);
        let mut state: u64 = 42;
        for _ in 0..n {
            let mut row = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                row.push(val);
            }
            payoffs.push(row);
        }
        let ids: Vec<u64> = (0..n as u64).collect();
        let matrix = WinMatrix::new(payoffs, ids);

        let gate = SimulationGate::default();
        assert!(gate.needs_full_simulation(&matrix));
    }

    #[test]
    fn test_simulation_gate_config_custom_thresholds() {
        let config = SimulationGateConfig {
            reducible_threshold: 0.5,
            irreducible_threshold: 0.9,
            lightweight_rounds: 10,
            full_rounds: 50,
        };
        let gate = SimulationGate::new(config);

        // Verify config is accessible.
        assert_eq!(gate.config().lightweight_rounds, 10);
        assert_eq!(gate.config().full_rounds, 50);
    }

    #[test]
    fn test_simulation_gate_result_fields() {
        let payoffs = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let ids = vec![1u64, 2];
        let matrix = WinMatrix::new(payoffs, ids);

        let gate = SimulationGate::default();
        let result = gate.route(&matrix);

        // All fields should be populated.
        assert!(result.compression_ratio >= 0.0);
        assert!(result.mean_abs_payoff > 0.0);
        assert!(result.payoff_variance >= 0.0);
        assert!(matches!(
            result.strategy,
            SimulationStrategy::AnalyticalShortcut
                | SimulationStrategy::LightweightSimulation
                | SimulationStrategy::FullSimulation
        ));
    }
}

// TL;DR: SimulationGate — routes between AnalyticalShortcut (skip simulation when reducible), LightweightSimulation (borderline), and FullSimulation (irreducible). Wraps IrreducibilityGate with configurable thresholds. Plan 188 Phase 4 integration.
