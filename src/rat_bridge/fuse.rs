//! Fused bridge attention — α-blend of dilated KV and bridge readout.
//!
//! Computes: y = α · attn(Q·K_dilated^T)·V_dilated + (1-α) · S·q
//! where α = sigmoid(gate), S is GDN2 state, q is query.
//!
//! Uses sigmoid (not softmax) for all gating per project constraints.

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
}
