//! Dilated KV accessor — strided views into KV cache.
//!
//! Zero-copy accessor for dilated sparse attention during decode.
//! Accesses every D-th token from the KV cache to reduce attention FLOPs.

use katgpt_core::types::DilationConfig;

/// Sigmoid activation — used instead of softmax per project constraints.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dilated decode step — replace full KV scan with D-strided access + bridge readout.
/// Returns attention-weighted output from dilated KV positions.
///
/// Uses sigmoid (not softmax) for all attention weights.
pub fn dilated_decode_step(
    query: &[f32],
    kv_cache_keys: &[Vec<f32>],
    kv_cache_vals: &[Vec<f32>],
    dilation: DilationConfig,
    gdn2_readout: &[f32],
) -> (Vec<f32>, f32) {
    // 1. Get dilated KV positions
    let indices = DilatedKvAccessor::dilated_indices(kv_cache_keys.len(), dilation);
    let keys_dilated: Vec<&Vec<f32>> = indices.iter().map(|&i| &kv_cache_keys[i]).collect();
    let vals_dilated: Vec<&Vec<f32>> = indices.iter().map(|&i| &kv_cache_vals[i]).collect();

    // 2. Compute sigmoid gate α = sigmoid(⟨q, gdn2_readout⟩)
    let alpha = sigmoid(
        query
            .iter()
            .zip(gdn2_readout.iter())
            .map(|(q, r)| q * r)
            .sum(),
    );

    // 3. Dilated attention with sigmoid (not softmax)
    let weights: Vec<f32> = keys_dilated
        .iter()
        .map(|k| sigmoid(k.iter().zip(query.iter()).map(|(ki, qi)| ki * qi).sum()))
        .collect();
    let w_sum: f32 = weights.iter().sum();

    let dim = query.len();
    let attn_out: Vec<f32> = if w_sum > 0.0 {
        vals_dilated
            .iter()
            .zip(weights.iter())
            .fold(vec![0.0; dim], |acc, (v, w)| {
                acc.iter()
                    .zip(v.iter())
                    .map(|(a, vi)| a + vi * w / w_sum)
                    .collect()
            })
    } else {
        vec![0.0; dim]
    };

    // 4. Bridge readout: S · q
    let bridge_out: Vec<f32> = gdn2_readout
        .iter()
        .zip(query.iter())
        .map(|(s, q)| s * q)
        .collect();

    // 5. α-blend: α · attn + (1-α) · bridge
    let output: Vec<f32> = attn_out
        .iter()
        .zip(bridge_out.iter())
        .map(|(a, b)| alpha * a + (1.0 - alpha) * b)
        .collect();

    (output, alpha)
}

/// Zero-copy accessor for dilated KV cache views.
pub struct DilatedKvAccessor;

impl DilatedKvAccessor {
    /// Access every D-th token from KV cache. Returns collected references.
    ///
    /// For true zero-copy in hot paths, prefer `dilated_indices()` + direct indexing
    /// to avoid the Vec allocation.
    pub fn stride_access<T>(kv_cache: &[T], d: DilationConfig) -> Vec<&T> {
        kv_cache.iter().step_by(d.stride()).collect()
    }

    /// Get dilated indices for a given cache length and dilation.
    ///
    /// Use these indices for direct array access without allocation in the hot loop.
    pub fn dilated_indices(len: usize, d: DilationConfig) -> Vec<usize> {
        (0..len).step_by(d.stride()).collect()
    }

    /// Number of elements accessed at a given dilation.
    ///
    /// Useful for pre-allocating output buffers.
    #[inline]
    pub fn dilated_len(len: usize, d: DilationConfig) -> usize {
        (len + d.stride() - 1) / d.stride()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stride_access() {
        let cache = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let accessed = DilatedKvAccessor::stride_access(&cache, DilationConfig::D2);
        assert_eq!(accessed, vec![&10, &30, &50, &70]);
    }

    #[test]
    fn test_dilated_indices() {
        let indices = DilatedKvAccessor::dilated_indices(8, DilationConfig::D4);
        assert_eq!(indices, vec![0, 4]);
    }

    #[test]
    fn test_dilated_len() {
        assert_eq!(DilatedKvAccessor::dilated_len(64, DilationConfig::D1), 64);
        assert_eq!(DilatedKvAccessor::dilated_len(64, DilationConfig::D4), 16);
        assert_eq!(DilatedKvAccessor::dilated_len(64, DilationConfig::D64), 1);
        assert_eq!(DilatedKvAccessor::dilated_len(65, DilationConfig::D16), 5);
    }

    #[test]
    fn test_stride_access_dense() {
        let cache = vec![1, 2, 3];
        let accessed = DilatedKvAccessor::stride_access(&cache, DilationConfig::D1);
        assert_eq!(accessed, vec![&1, &2, &3]);
    }

    #[test]
    fn test_dilated_decode_step() {
        let query = vec![0.5; 8];
        let keys = (0..16).map(|_| vec![0.3; 8]).collect::<Vec<_>>();
        let vals = (0..16).map(|_| vec![0.4; 8]).collect::<Vec<_>>();
        let gdn2 = vec![0.1; 8];

        let (output, alpha) = dilated_decode_step(&query, &keys, &vals, DilationConfig::D4, &gdn2);

        assert_eq!(output.len(), 8);
        assert!((0.0..=1.0).contains(&alpha));
    }

    #[test]
    fn test_dilated_decode_d4_accesses_fewer_positions() {
        let query = vec![0.5; 4];
        let keys = (0..16).map(|i| vec![i as f32; 4]).collect::<Vec<_>>();
        let vals = (0..16).map(|i| vec![i as f32 * 2.0; 4]).collect::<Vec<_>>();
        let gdn2 = vec![0.1; 4];

        // D=1 accesses all 16 positions
        let indices_dense = DilatedKvAccessor::dilated_indices(16, DilationConfig::D1);
        assert_eq!(indices_dense.len(), 16);

        // D=4 accesses 4 positions
        let indices_d4 = DilatedKvAccessor::dilated_indices(16, DilationConfig::D4);
        assert_eq!(indices_d4.len(), 4);

        // Both should produce valid output
        let (out_dense, _) = dilated_decode_step(&query, &keys, &vals, DilationConfig::D1, &gdn2);
        let (out_d4, _) = dilated_decode_step(&query, &keys, &vals, DilationConfig::D4, &gdn2);

        assert_eq!(out_dense.len(), 4);
        assert_eq!(out_d4.len(), 4);
    }
}
