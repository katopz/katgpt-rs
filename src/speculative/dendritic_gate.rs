//! DendriticGate — NMDA-inspired adaptive tree branching (Plan 260).
//!
//! Uses entropy + candidate coincidence as a deterministic signal for branch budget
//! allocation. Zero parameters, zero training, physics-based adaptive compute.
//! Modeled on dendritic NMDA Mg²⁺ voltage-dependent coincidence detection.

/// NMDA-inspired gate that modulates DDTree expansion budget based on entropy and coincidence.
///
/// The gate computes: `sigmoid(sensitivity * (entropy - threshold)) * coincidence`
///
/// - High entropy (> threshold) + high coincidence → gate opens → expand more
/// - Low entropy (< threshold) + any coincidence → gate closes → expand less
/// - High entropy + low coincidence → gate suppressed → don't over-expand
///
/// This is deterministic: same inputs always produce the same output.
/// No RNG, no bandit, no learned parameters.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DendriticGate {
    /// Entropy threshold for gate activation (default: 1.5).
    /// Below this, the sigmoid output is < 0.5 → constriction.
    pub threshold: f32,
    /// Sigmoid steepness / voltage sensitivity (default: 2.0).
    /// Higher = sharper transition between open/closed.
    pub voltage_sensitivity: f32,
    /// Top-K agreement span for coincidence scoring (default: 4).
    pub coincidence_window: usize,
}

impl Default for DendriticGate {
    fn default() -> Self {
        Self::new()
    }
}

impl DendriticGate {
    /// Const constructor with default parameters.
    #[inline]
    pub const fn new() -> Self {
        Self {
            threshold: 1.5,
            voltage_sensitivity: 2.0,
            coincidence_window: 4,
        }
    }

    /// Const constructor with custom parameters.
    #[inline]
    pub const fn with_params(
        threshold: f32,
        voltage_sensitivity: f32,
        coincidence_window: usize,
    ) -> Self {
        Self {
            threshold,
            voltage_sensitivity,
            coincidence_window,
        }
    }

    /// Compute the NMDA gate value from entropy and coincidence signals.
    ///
    /// Returns `sigmoid(sensitivity * (entropy - threshold)) * coincidence` ∈ [0, 1].
    ///
    /// - `entropy` — entropy of the marginal distribution at this depth
    /// - `coincidence` — agreement fraction between top-K candidates and parent path ∈ [0, 1]
    ///
    /// Zero allocation, stack-only, deterministic.
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

/// Dendritic sigmoid: numerically stable sigmoid for gate computation.
/// Uses the identity `sigmoid(x) = 1 / (1 + exp(-x))`.
#[inline]
pub fn dendritic_sigmoid(x: f32) -> f32 {
    // Numerically stable: split into positive/negative to avoid overflow
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gate_values() {
        let g = DendriticGate::default();
        assert_eq!(g.threshold, 1.5);
        assert_eq!(g.voltage_sensitivity, 2.0);
        assert_eq!(g.coincidence_window, 4);
    }

    #[test]
    fn const_new_matches_default() {
        assert_eq!(
            DendriticGate::new().threshold,
            DendriticGate::default().threshold
        );
        assert_eq!(
            DendriticGate::new().voltage_sensitivity,
            DendriticGate::default().voltage_sensitivity
        );
        assert_eq!(
            DendriticGate::new().coincidence_window,
            DendriticGate::default().coincidence_window
        );
    }

    #[test]
    fn sigmoid_symmetry() {
        // sigmoid(x) + sigmoid(-x) = 1
        for x in [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0] {
            let s = dendritic_sigmoid(x);
            let s_neg = dendritic_sigmoid(-x);
            assert!(
                (s + s_neg - 1.0).abs() < 1e-5,
                "sigmoid({}) + sigmoid(-{}) = {} ≠ 1",
                x,
                x,
                s + s_neg
            );
        }
    }

    #[test]
    fn high_entropy_high_coincidence_opens_gate() {
        let g = DendriticGate::new();
        // entropy=3.0 > threshold=1.5, coincidence=1.0
        let val = g.compute_gate(3.0, 1.0);
        assert!(val > 0.8, "gate should be open, got {val}");
    }

    #[test]
    fn low_entropy_closes_gate() {
        let g = DendriticGate::new();
        // entropy=0.5 < threshold=1.5
        let val = g.compute_gate(0.5, 1.0);
        assert!(val < 0.2, "gate should be closed, got {val}");
    }

    #[test]
    fn low_coincidence_suppresses() {
        let g = DendriticGate::new();
        let val = g.compute_gate(3.0, 0.1);
        assert!(val < 0.1, "low coincidence should suppress gate, got {val}");
    }

    #[test]
    fn early_exit_detects_closed_gate() {
        let g = DendriticGate::new();
        assert!(g.should_exit_early(0.5, 0.5));
        assert!(!g.should_exit_early(3.0, 1.0));
    }

    #[test]
    fn gate_is_deterministic() {
        let g = DendriticGate::new();
        let a = g.compute_gate(2.3, 0.7);
        let b = g.compute_gate(2.3, 0.7);
        assert_eq!(a, b);
    }
}
