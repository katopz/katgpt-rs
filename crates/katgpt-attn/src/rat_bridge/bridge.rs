//! Bridge state — wraps GDN2 recurrent state for dilated attention.
//!
//! Reuses existing GDN2 readout as a projection vector for bridge attention.
//! No new parameters — pure inference-time reuse of recurrent state.

use katgpt_core::types::DilationConfig;

/// State for RAT+ bridge, wrapping recurrent attention state.
#[derive(Debug, Clone)]
pub struct RatBridgeState {
    /// Dilation factor for KV cache access.
    pub dilation: DilationConfig,
    /// Gate value: α = sigmoid(⟨q, gdn2_readout⟩)
    /// Controls blend between dilated KV attention and bridge readout.
    pub alpha: f32,
    /// Bridge projection vector (reuses GDN2 readout).
    pub projection: Vec<f32>,
}

impl RatBridgeState {
    /// Create a new bridge state with given dilation and dimension.
    pub fn new(dilation: DilationConfig, dim: usize) -> Self {
        Self {
            dilation,
            alpha: 0.5,
            projection: vec![0.0; dim],
        }
    }

    /// Compute bridge gate: α = sigmoid(dot(query, gdn2_readout)).
    ///
    /// Uses sigmoid (not softmax) per project constraints.
    /// Returns the computed gate value in [0, 1].
    pub fn compute_gate(&mut self, query: &[f32], gdn2_readout: &[f32]) -> f32 {
        let dot: f32 = query
            .iter()
            .zip(gdn2_readout.iter())
            .map(|(q, r)| q * r)
            .sum();
        // sigmoid, not softmax
        self.alpha = 1.0 / (1.0 + (-dot).exp());
        self.alpha
    }

    /// Reset dilation based on QPS load signal.
    ///
    /// Delegates to `DilationConfig::from_qps` which maps QPS thresholds
    /// to dilation strides (low QPS → dense, high QPS → aggressive dilation).
    pub fn set_dilation_from_qps(&mut self, qps: f32) {
        self.dilation = DilationConfig::from_qps(qps);
    }

    /// Update projection from GDN2 readout (no new parameters — reuses GDN2 state).
    ///
    /// The bridge projection is the GDN2 readout vector itself, used for
    /// bridge attention: `bridge_out = projection · query`.
    pub fn update_projection(&mut self, gdn2_readout: &[f32]) {
        self.projection.clear();
        self.projection.extend_from_slice(gdn2_readout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_state_new() {
        let state = RatBridgeState::new(DilationConfig::D16, 64);
        assert_eq!(state.dilation, DilationConfig::D16);
        assert_eq!(state.alpha, 0.5);
        assert_eq!(state.projection.len(), 64);
    }

    #[test]
    fn test_compute_gate_range() {
        let mut state = RatBridgeState::new(DilationConfig::D4, 8);
        let query = vec![1.0; 8];
        let readout = vec![1.0; 8];
        let alpha = state.compute_gate(&query, &readout);
        assert!((0.0..=1.0).contains(&alpha));
        // dot=8 → sigmoid(8) ≈ 0.9997
        assert!(alpha > 0.99);
    }

    #[test]
    fn test_compute_gate_orthogonal() {
        let mut state = RatBridgeState::new(DilationConfig::D1, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let readout = vec![0.0, 1.0, 0.0, 0.0];
        let alpha = state.compute_gate(&query, &readout);
        // dot=0 → sigmoid(0) = 0.5
        assert!((alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_set_dilation_from_qps() {
        let mut state = RatBridgeState::new(DilationConfig::D1, 8);
        // Low QPS → dense
        state.set_dilation_from_qps(0.5);
        assert_eq!(state.dilation, DilationConfig::D1);
        // High QPS → aggressive dilation
        state.set_dilation_from_qps(50.0);
        assert!(state.dilation.stride() > 1);
    }

    #[test]
    fn test_update_projection() {
        let mut state = RatBridgeState::new(DilationConfig::D4, 4);
        let readout = vec![1.0, 2.0, 3.0, 4.0];
        state.update_projection(&readout);
        assert_eq!(state.projection, readout);
    }
}
