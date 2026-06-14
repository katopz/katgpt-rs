//! DendriticGate — NMDA-inspired adaptive tree branching (Plan 260).
//!
//! Physics-inspired deterministic gate that modulates DDTree expansion budget
//! using entropy + candidate coincidence. Zero parameters, zero training.
//!
//! Modeled on dendritic NMDA Mg²⁺ voltage-dependent coincidence detection:
//! high entropy (uncertainty) → gate opens → more compute.
//! Low coincidence (novel context) → gate opens → more exploration.
//! High coincidence + low entropy → gate closes → proximal dendrite sufficient.

/// NMDA-gated adaptive expansion budget.
///
/// `effective_budget = base_budget * nmda_gate`
/// where `nmda_gate = sigmoid(sensitivity * (entropy - threshold)) * coincidence`
///
/// Stack-only, zero-allocation, deterministic.
///
/// Field order: usize (8B) → f32 (4B) → f32 (4B) — no padding, 16 bytes total.
#[derive(Debug, Clone, Copy)]
pub struct DendriticGate {
    /// Top-K agreement span for coincidence scoring (default: 4).
    pub coincidence_window: usize,
    /// Entropy threshold for gate activation (default: 1.5 nats).
    pub threshold: f32,
    /// Sigmoid steepness — controls gate sharpness (default: 2.0).
    pub voltage_sensitivity: f32,
}

impl Default for DendriticGate {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DendriticGate {
    /// Const constructor with default parameters.
    #[inline]
    pub const fn new() -> Self {
        Self {
            coincidence_window: 4,
            threshold: 1.5,
            voltage_sensitivity: 2.0,
        }
    }

    /// Constructor with custom parameters.
    #[inline]
    pub const fn with_params(
        threshold: f32,
        voltage_sensitivity: f32,
        coincidence_window: usize,
    ) -> Self {
        Self {
            coincidence_window,
            threshold,
            voltage_sensitivity,
        }
    }

    /// Compute NMDA gate value from entropy and coincidence signals.
    ///
    /// Returns `sigmoid(sensitivity * (entropy - threshold)) * coincidence`
    /// in [0, 1]. When `entropy >> threshold` and `coincidence ≈ 1.0`,
    /// gate opens fully (expand aggressively). When `entropy << threshold`
    /// or `coincidence ≈ 0.0`, gate closes (early exit).
    #[inline]
    pub fn compute_gate(&self, entropy: f32, coincidence: f32) -> f32 {
        dendritic_sigmoid(self.voltage_sensitivity * (entropy - self.threshold)) * coincidence
    }

    /// Check if the gate is effectively closed (below early-exit threshold).
    /// When gate < 0.1, proximal dendrite is sufficient — no need to expand further.
    #[inline]
    pub fn should_exit_early(&self, entropy: f32, coincidence: f32) -> bool {
        self.compute_gate(entropy, coincidence) < 0.1
    }
}

/// Sigmoid function for dendritic gate computation.
///
/// `σ(x) = 1 / (1 + exp(-x))`
///
/// Uses sigmoid (not softmax) per project rules.
/// Delegates to shared crate::simd::fast_sigmoid (bounded (0,1), Cephes-exp).
#[inline]
pub fn dendritic_sigmoid(x: f32) -> f32 {
    crate::simd::fast_sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dendritic_gate_deterministic() {
        let gate = DendriticGate::new();
        let a = gate.compute_gate(2.0, 0.8);
        let b = gate.compute_gate(2.0, 0.8);
        assert!(
            (a - b).abs() < 1e-10,
            "same inputs must produce same output: {a} vs {b}"
        );
    }

    #[test]
    fn test_dendritic_gate_high_entropy_expands() {
        let gate = DendriticGate::new();
        // High entropy (3.0 >> threshold 1.5), full coincidence
        let gate_val = gate.compute_gate(3.0, 1.0);
        assert!(
            gate_val > 0.5,
            "high entropy should expand: gate={gate_val}"
        );
    }

    #[test]
    fn test_dendritic_gate_low_entropy_contracts() {
        let gate = DendriticGate::new();
        // Low entropy (0.5 << threshold 1.5), full coincidence
        let gate_val = gate.compute_gate(0.5, 1.0);
        assert!(
            gate_val < 0.5,
            "low entropy should contract: gate={gate_val}"
        );
    }

    #[test]
    fn test_dendritic_gate_coincidence_and() {
        let gate = DendriticGate::new();
        // High entropy but zero coincidence → suppressed
        let gate_val = gate.compute_gate(3.0, 0.1);
        assert!(
            gate_val < 0.15,
            "low coincidence should suppress even high entropy: gate={gate_val}"
        );
    }

    #[test]
    fn test_dendritic_gate_early_exit() {
        let gate = DendriticGate::new();
        // Very low entropy → gate should be < 0.1 (early exit trigger)
        let gate_val = gate.compute_gate(0.0, 1.0);
        assert!(
            gate_val < 0.1,
            "very low entropy should trigger early exit: gate={gate_val}"
        );
    }

    #[test]
    fn test_dendritic_sigmoid_symmetry() {
        let x = 1.5f32;
        let pos = dendritic_sigmoid(x);
        let neg = dendritic_sigmoid(-x);
        assert!(
            (pos + neg - 1.0).abs() < 1e-6,
            "sigmoid(x) + sigmoid(-x) should be 1.0"
        );
    }

    #[test]
    fn test_dendritic_sigmoid_clamps() {
        assert!((dendritic_sigmoid(100.0) - 1.0).abs() < 1e-10);
        assert!(dendritic_sigmoid(-100.0).abs() < 1e-10);
    }

    #[test]
    fn test_dendritic_gate_at_threshold() {
        let gate = DendriticGate::new();
        // entropy == threshold → sigmoid(0) = 0.5
        let gate_val = gate.compute_gate(1.5, 1.0);
        assert!(
            (gate_val - 0.5).abs() < 1e-6,
            "at threshold with full coincidence, gate should be 0.5: got {gate_val}"
        );
    }

    #[test]
    fn test_dendritic_gate_custom_params() {
        let gate = DendriticGate::with_params(2.0, 4.0, 8);
        // threshold=2.0, sensitivity=4.0: entropy=3.0 → sigmoid(4.0 * 1.0) = sigmoid(4.0) ≈ 0.982
        let gate_val = gate.compute_gate(3.0, 1.0);
        assert!(
            gate_val > 0.95,
            "high sensitivity should make gate very steep: gate={gate_val}"
        );
        assert_eq!(gate.coincidence_window, 8);
    }
}
