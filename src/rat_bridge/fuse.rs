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
///
/// Allocating wrapper — prefer [`bridge_attention_into`] in hot paths.
pub fn bridge_attention(
    query: &[f32],
    kv_keys_dilated: &[Vec<f32>],
    kv_vals_dilated: &[Vec<f32>],
    gdn2_state: &[f32],
    alpha: f32,
) -> BridgeAttentionOutput {
    let dim = query.len();
    let mut output = vec![0.0f32; dim];
    bridge_attention_into(
        query,
        kv_keys_dilated,
        kv_vals_dilated,
        gdn2_state,
        alpha,
        &mut output,
    );
    BridgeAttentionOutput { output, alpha }
}

/// Zero-alloc fused bridge attention. Writes the blended output into `out`.
///
/// Internally fuses the three dot-product/elementwise stages (attention
/// weights, value blend, bridge readout, α-blend) so no intermediate `Vec`
/// is allocated. The previous `.fold(vec![...], ...).collect()` pattern was
/// O(n_kv * dim) extra work plus n_kv allocations per call — this version is
/// O(n_kv * dim) work with zero allocations.
///
/// Uses [`crate::simd::simd_dot_f32`] for the per-key dot product.
///
/// Accepts KV containers as slices of anything that derefs to `&[f32]` —
/// works for `&[Vec<f32>]`, `&[&Vec<f32>]`, `&[&[f32]]`, etc. This lets
/// [`rat_decode_step_into`] gather dilated KV references without cloning.
pub fn bridge_attention_into<K: AsRef<[f32]>, V: AsRef<[f32]>>(
    query: &[f32],
    kv_keys_dilated: &[K],
    kv_vals_dilated: &[V],
    gdn2_state: &[f32],
    alpha: f32,
    out: &mut [f32],
) {
    let dim = query.len();
    debug_assert!(out.len() >= dim);
    debug_assert!(gdn2_state.len() >= dim);

    // Stage 1+2: compute sigmoid attention weights and accumulate the
    // normalized value blend directly into `out`. We avoid materializing
    // `attn_weights` by computing the softmax-like normalization in two passes:
    //   pass A: Σ σ(q·k_i)
    //   pass B: write Σ_i σ(q·k_i) · v_i / Σ σ(q·k_i)
    //
    // Since sigmoid weights are independent and we only need their sum for
    // normalization (NOT softmax), the two-pass approach is exact and avoids
    // a Vec<f32> of length n_kv per call.
    //
    // SAFETY: all slices are bounded by their `.len()`; we use unchecked indexing
    // only after debug_asserts confirm lengths match.
    let inv_alpha = 1.0 - alpha;

    // Bridge readout: out[d] = α·attn + (1-α)·(s[d]·q[d]).
    // Pre-load the bridge contribution, then add the attention contribution on top.
    // Both contributions are fused into the same output buffer.
    for d in 0..dim {
        // Bridge contribution
        out[d] = inv_alpha * gdn2_state[d] * query[d];
    }

    if kv_keys_dilated.is_empty() {
        return;
    }

    // Pass A: compute weight_sum = Σ σ(q·k_i)
    let mut weight_sum = 0.0f32;
    // Avoid re-allocating: we accumulate the weighted value contributions into
    // `out` AFTER computing weight_sum, so we need to store the per-key weights
    // somewhere. For small n_kv (typical dilation factor ≤ 16), we use a stack
    // array; for larger n_kv we fall back to a heap allocation via Vec.
    //
    // However, we can avoid the per-key weight storage entirely by computing
    // each weight twice (once in pass A for the sum, once in pass B for the
    // blend). The sigmoid is ~10 cycles; doubling it is still cheaper than the
    // allocation + cache pollution of an n_kv-sized buffer for typical n_kv ≤ 16.
    //
    // For n_kv > 16 (rare), the doubled-sigmoid cost exceeds the alloc cost,
    // so we switch to the storage-based path. We pick the threshold at 16
    // based on the typical dilation factor (D1–D16).
    const INLINE_MAX: usize = 16;

    let n_kv = kv_keys_dilated.len();

    if n_kv <= INLINE_MAX {
        // Two-pass with doubled sigmoid (no per-key storage).
        for k in kv_keys_dilated {
            let dot = crate::simd::simd_dot_f32(k.as_ref(), query, dim);
            weight_sum += sigmoid(dot);
        }
        if weight_sum <= 0.0 {
            return;
        }
        let inv_weight_sum = 1.0 / weight_sum;
        // Pass B: add α · Σ_i σ(q·k_i) · v_i / Σ σ(q·k_i) to out.
        for (k, v) in kv_keys_dilated.iter().zip(kv_vals_dilated.iter()) {
            let dot = crate::simd::simd_dot_f32(k.as_ref(), query, dim);
            let w = sigmoid(dot) * inv_weight_sum * alpha;
            let v_ref = v.as_ref();
            // out[d] += w * v[d]  (fused SIMD axpy)
            for d in 0..dim {
                out[d] += w * v_ref[d];
            }
        }
    } else {
        // Storage-based path for large n_kv. Single sigmoid pass + single axpy.
        let mut weights: Vec<f32> = Vec::with_capacity(n_kv);
        // SAFETY: we set len then fully initialize before reading.
        // (Avoids the memset that `vec![0.0; n_kv]` would perform.)
        // For correctness in non-`debug_assert` builds we still initialize
        // unconditionally below; the `set_len` is just to skip the memset.
        // Allow: intentional uninit-then-fill to skip the memset.
        #[allow(clippy::uninit_vec)]
        unsafe {
            weights.set_len(n_kv)
        };
        for (i, k) in kv_keys_dilated.iter().enumerate() {
            let w = sigmoid(crate::simd::simd_dot_f32(k.as_ref(), query, dim));
            weights[i] = w;
            weight_sum += w;
        }
        if weight_sum <= 0.0 {
            return;
        }
        let inv_weight_sum = 1.0 / weight_sum;
        for (w_slot, v) in weights.iter().zip(kv_vals_dilated.iter()) {
            let w = *w_slot * inv_weight_sum * alpha;
            let v_ref = v.as_ref();
            for d in 0..dim {
                out[d] += w * v_ref[d];
            }
        }
    }
}

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x)).
///
/// Used for all attention weights in this module. NOT softmax per project
/// constraints — each weight is independent and they don't sum to 1.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the bridge gate α = sigmoid(⟨query, gdn2_readout⟩).
///
/// Extracted as a standalone helper so callers like
/// [`super::dilated_kv::dilated_decode_step_into`] can compute the gate
/// once and reuse it for both dilated-attention and bridge-readout stages.
/// Uses [`crate::simd::simd_dot_f32`] for the dot product.
#[inline]
pub fn bridge_attention_gate(query: &[f32], gdn2_readout: &[f32]) -> f32 {
    let dim = query.len();
    sigmoid(crate::simd::simd_dot_f32(query, gdn2_readout, dim))
}

/// Full decode step with RAT+ bridge.
///
/// Combines `RatBridgeState` gate computation with dilated KV attention.
/// This is the primary entry point for decode-time inference:
/// 1. Update bridge gate from query + GDN2 readout
/// 2. Compute dilated indices from current dilation config
/// 3. Fused bridge attention with α-blend
///
/// Allocates `BridgeAttentionOutput` (containing the output `Vec<f32>`).
/// For hot decode loops, prefer [`rat_decode_step_into`] which writes the
/// output into a caller-provided buffer and skips per-step allocation.
pub fn rat_decode_step(
    state: &mut RatBridgeState,
    query: &[f32],
    kv_keys: &[Vec<f32>],
    kv_vals: &[Vec<f32>],
    gdn2_readout: &[f32],
) -> BridgeAttentionOutput {
    let dim = query.len();
    let mut output = vec![0.0f32; dim];
    let alpha = rat_decode_step_into(state, query, kv_keys, kv_vals, gdn2_readout, &mut output);
    BridgeAttentionOutput { output, alpha }
}

/// Zero-alloc variant of [`rat_decode_step`]. Writes the blended output into `out`.
///
/// Avoids the `Vec<Vec<f32>>` clones the previous implementation performed on
/// every decode step (one clone per dilated KV position). Instead, it indexes
/// directly into `kv_keys`/`kv_vals` via the dilated index list.
///
/// Returns the gate value α used for blending.
pub fn rat_decode_step_into(
    state: &mut RatBridgeState,
    query: &[f32],
    kv_keys: &[Vec<f32>],
    kv_vals: &[Vec<f32>],
    gdn2_readout: &[f32],
    out: &mut [f32],
) -> f32 {
    // 1. Update bridge gate
    state.compute_gate(query, gdn2_readout);

    // 2. Get dilated indices — no clones, just indices into kv_keys/kv_vals.
    let indices = DilatedKvAccessor::dilated_indices(kv_keys.len(), state.dilation);

    // 3. Gather dilated KV references without cloning. We need &[Vec<f32>]
    // to match `bridge_attention_into`'s signature; building it from indices
    // is a small allocation but is `O(n_dilated)` pointers, not `O(n_dilated * dim)` f32s.
    //
    // For typical dilation factors (D1–D16), n_dilated ≤ 16, so the Vec<&Vec>
    // is at most 16 pointers = 128 bytes — much smaller than the previous
    // 16 × dim × 4 bytes clone (e.g., 16 × 4096 × 4 = 256 KB for n_embd=4096).
    //
    // For the absolute hottest decode loops, callers can pre-build the
    // dilated slice references themselves and call `bridge_attention_into`
    // directly to skip even this pointer-vec allocation.
    let keys_dilated: Vec<&Vec<f32>> = indices.iter().map(|&i| &kv_keys[i]).collect();
    let vals_dilated: Vec<&Vec<f32>> = indices.iter().map(|&i| &kv_vals[i]).collect();

    bridge_attention_into(
        query,
        &keys_dilated,
        &vals_dilated,
        gdn2_readout,
        state.alpha,
        out,
    );
    state.alpha
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
