//! Dilated KV accessor — strided views into KV cache.
//!
//! Zero-copy accessor for dilated sparse attention during decode.
//! Accesses every D-th token from the KV cache to reduce attention FLOPs.

use katgpt_core::types::DilationConfig;

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
}
