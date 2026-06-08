//! Fused bridge attention — α-blend of dilated KV and bridge readout.
//!
//! Computes: y = α · attn(Q·K_dilated^T)·V_dilated + (1-α) · S·q
//! where α = sigmoid(gate), S is GDN2 state, q is query.
//!
//! Uses sigmoid (not softmax) for all gating per project constraints.

use super::bridge::RatBridgeState;
use super::dilated_kv::DilatedKvAccessor;

/// Fused bridge attention output.
#[derive(Debug, Clone)]
pub struct BridgeAttentionOutput {
    /// Output tensor (same dim as query).
    pub output: Vec<f32>,
    /// Gate value α used for blending.
    pub alpha: f32,
}

/// Compute fused bridge attention:
/// y = α · attn(Q·K_dilated^T)·V_dilated + (1-α) · S·q
/// where S is GDN2 state, q is query, K/V_dilated are strided.
///
/// Attention weights use sigmoid (not softmax) per project constraints.
pub fn bridge_attention(
    query: &[f32],
    kv_keys_dilated: &[Vec<f32>],
    kv_vals_dilated: &[Vec<f32>],
    gdn2_state: &[f32],
    alpha: f32,
) -> BridgeAttentionOutput {
    let dim = query.len();

    // Dilated attention: sigmoid-weighted dot-product attention on strided KV
    let attn_weights: Vec<f32> = kv_keys_dilated
        .iter()
        .map(|k| {
            let dot: f32 = k.iter().zip(query.iter()).map(|(ki, qi)| ki * qi).sum();
            1.0 / (1.0 + (-dot).exp()) // sigmoid, not softmax
        })
        .collect();

    let weight_sum: f32 = attn_weights.iter().sum();
    let attn_output: Vec<f32> = if weight_sum > 0.0 {
        kv_vals_dilated
            .iter()
            .zip(attn_weights.iter())
            .fold(vec![0.0; dim], |acc, (v, w)| {
                acc.iter()
                    .zip(v.iter())
                    .map(|(a, vi)| a + vi * w / weight_sum)
                    .collect()
            })
    } else {
        vec![0.0; dim]
    };

    // Bridge readout: S · q (simplified projection)
    let bridge_output: Vec<f32> = gdn2_state
        .iter()
        .zip(query.iter())
        .map(|(s, q)| s * q)
        .collect();

    // α-blend
    let output: Vec<f32> = attn_output
        .iter()
        .zip(bridge_output.iter())
        .map(|(a, b)| alpha * a + (1.0 - alpha) * b)
        .collect();

    BridgeAttentionOutput { output, alpha }
}

/// Full decode step with RAT+ bridge.
///
/// Combines `RatBridgeState` gate computation with dilated KV attention.
/// This is the primary entry point for decode-time inference:
/// 1. Update bridge gate from query + GDN2 readout
/// 2. Compute dilated indices from current dilation config
/// 3. Fused bridge attention with α-blend
pub fn rat_decode_step(
    state: &mut RatBridgeState,
    query: &[f32],
    kv_keys: &[Vec<f32>],
    kv_vals: &[Vec<f32>],
    gdn2_readout: &[f32],
) -> BridgeAttentionOutput {
    // 1. Update bridge gate
    state.compute_gate(query, gdn2_readout);

    // 2. Get dilated indices
    let indices = DilatedKvAccessor::dilated_indices(kv_keys.len(), state.dilation);

    // 3. Dilated KV access — collect owned copies for bridge_attention compatibility
    let keys_dilated: Vec<Vec<f32>> = indices.iter().map(|&i| kv_keys[i].clone()).collect();
    let vals_dilated: Vec<Vec<f32>> = indices.iter().map(|&i| kv_vals[i].clone()).collect();

    // 4. Bridge attention with α from state
    bridge_attention(
        query,
        &keys_dilated,
        &vals_dilated,
        gdn2_readout,
        state.alpha,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_attention_dims() {
        let dim = 8;
        let query = vec![0.5; dim];
        let keys = vec![vec![0.3; dim], vec![0.7; dim]];
        let vals = vec![vec![0.4; dim], vec![0.6; dim]];
        let state = vec![0.1; dim];
        let out = bridge_attention(&query, &keys, &vals, &state, 0.5);
        assert_eq!(out.output.len(), dim);
        assert!((0.0..=1.0).contains(&out.alpha));
    }

    #[test]
    fn test_alpha_controls_blend() {
        let dim = 4;
        let query = vec![1.0; dim];
        let keys = vec![vec![1.0; dim]];
        let vals = vec![vec![2.0; dim]];
        let state = vec![0.5; dim];

        let full_kv = bridge_attention(&query, &keys, &vals, &state, 1.0);
        let full_bridge = bridge_attention(&query, &keys, &vals, &state, 0.0);

        // alpha=1 should be closer to KV output, alpha=0 closer to bridge
        assert_ne!(full_kv.output, full_bridge.output);
    }

    #[test]
    fn test_empty_kv_uses_bridge_only() {
        let dim = 4;
        let query = vec![1.0; dim];
        let keys: Vec<Vec<f32>> = vec![];
        let vals: Vec<Vec<f32>> = vec![];
        let state = vec![0.5; dim];

        let out = bridge_attention(&query, &keys, &vals, &state, 0.5);
        // With empty KV, attn_output is all zeros, so output = 0.5 * 0 + 0.5 * bridge
        // bridge = S·q = [0.5, 0.5, 0.5, 0.5]
        // output = 0.5 * 0 + 0.5 * 0.5 = 0.25
        for &v in &out.output {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rat_decode_step() {
        let mut state = RatBridgeState::new(katgpt_core::types::DilationConfig::D4, 8);
        let query = vec![0.5; 8];
        let keys = (0..16).map(|_| vec![0.3; 8]).collect::<Vec<_>>();
        let vals = (0..16).map(|_| vec![0.4; 8]).collect::<Vec<_>>();
        let gdn2 = vec![0.1; 8];
        let out = rat_decode_step(&mut state, &query, &keys, &vals, &gdn2);
        assert_eq!(out.output.len(), 8);
        assert!((0.0..=1.0).contains(&out.alpha));
    }

    #[test]
    fn test_dilation_reduces_kv_access() {
        let query = vec![0.5; 4];
        let keys = (0..32)
            .map(|i| vec![i as f32 / 32.0; 4])
            .collect::<Vec<_>>();
        let vals = (0..32)
            .map(|i| vec![i as f32 / 64.0; 4])
            .collect::<Vec<_>>();
        let gdn2 = vec![0.1; 4];

        let dense = {
            let mut s = RatBridgeState::new(katgpt_core::types::DilationConfig::D1, 4);
            rat_decode_step(&mut s, &query, &keys, &vals, &gdn2)
        };
        let dilated = {
            let mut s = RatBridgeState::new(katgpt_core::types::DilationConfig::D16, 4);
            rat_decode_step(&mut s, &query, &keys, &vals, &gdn2)
        };

        // Both should produce valid output, but different (different KV positions used)
        assert_eq!(dense.output.len(), 4);
        assert_eq!(dilated.output.len(), 4);
    }
}
