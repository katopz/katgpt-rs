//! Dilated KV accessor — strided views into KV cache.
//!
//! Zero-copy accessor for dilated sparse attention during decode.
//! Accesses every D-th token from the KV cache to reduce attention FLOPs.

use katgpt_core::types::DilationConfig;


/// Dilated decode step — replace full KV scan with D-strided access + bridge readout.
/// Returns attention-weighted output from dilated KV positions.
///
/// Uses sigmoid (not softmax) for all attention weights.
///
/// Allocating wrapper — prefer [`dilated_decode_step_into`] in hot paths.
pub fn dilated_decode_step(
    query: &[f32],
    kv_cache_keys: &[Vec<f32>],
    kv_cache_vals: &[Vec<f32>],
    dilation: DilationConfig,
    gdn2_readout: &[f32],
) -> (Vec<f32>, f32) {
    let dim = query.len();
    let mut output = vec![0.0f32; dim];
    let alpha = dilated_decode_step_into(
        query,
        kv_cache_keys,
        kv_cache_vals,
        dilation,
        gdn2_readout,
        &mut output,
    );
    (output, alpha)
}

/// Zero-alloc variant of [`dilated_decode_step`]. Writes the blended output
/// into `out` and returns the gate value α.
///
/// Internally delegates to [`super::fuse::bridge_attention_into`] which fuses
/// the attention/bridge/α-blend stages into a single pass. The previous
/// implementation used `.fold(vec![...], ...).collect()` which allocated a
/// `Vec<f32>` per fold step (O(n_kv) allocations + O(n_kv * dim) extra work
/// per call).
pub fn dilated_decode_step_into(
    query: &[f32],
    kv_cache_keys: &[Vec<f32>],
    kv_cache_vals: &[Vec<f32>],
    dilation: DilationConfig,
    gdn2_readout: &[f32],
    out: &mut [f32],
) -> f32 {
    // 1. Compute sigmoid gate α = sigmoid(⟨q, gdn2_readout⟩)
    //
    // (Independent of dilation — compute once.)
    let alpha = super::fuse::bridge_attention_gate(query, gdn2_readout);

    // 2. Gather dilated KV references. Small alloc (n_dilated pointers) — for
    //    the hottest decode loops, callers should call bridge_attention_into
    //    directly with pre-built dilated slice references.
    let indices = DilatedKvAccessor::dilated_indices(kv_cache_keys.len(), dilation);
    let keys_dilated: Vec<&Vec<f32>> = indices.iter().map(|&i| &kv_cache_keys[i]).collect();
    let vals_dilated: Vec<&Vec<f32>> = indices.iter().map(|&i| &kv_cache_vals[i]).collect();

    // 3. Fused attention + bridge + α-blend, written directly into `out`.
    super::fuse::bridge_attention_into(
        query,
        &keys_dilated,
        &vals_dilated,
        gdn2_readout,
        alpha,
        out,
    );

    alpha
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
        len.div_ceil(d.stride())
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
