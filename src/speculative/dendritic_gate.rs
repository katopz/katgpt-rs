//! DendriticGate — NMDA-inspired adaptive tree branching (Plan 260).
//!
//! Re-exports from `katgpt_core::dendritic_gate` for use in speculative module.
//! The canonical implementation lives in `katgpt-core` so both the main crate
//! and `katgpt-core` (MuxBfs) can share it.

pub use katgpt_core::dendritic_gate::{DendriticGate, dendritic_sigmoid};

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
        let val = g.compute_gate(3.0, 1.0);
        assert!(val > 0.8, "gate should be open, got {val}");
    }

    #[test]
    fn low_entropy_closes_gate() {
        let g = DendriticGate::new();
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
