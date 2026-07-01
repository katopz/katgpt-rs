//! Compact KV cache storage and compaction metadata.

use crate::still_kv::beta_bias::BetaBias;
use half::f16;
use std::fmt;

/// Strategy used for KV cache compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompactionStrategy {
    /// k-means-style cluster representatives.
    ClusterCentroids,
    /// Attention-score-weighted importance sampling.
    AttentionWeighted,
    /// PCA/SVD low-rank projection.
    SpectralProjection,
    /// BFCF region-weighted blending.
    BfcfRegionBlend,
    /// Multiplexed superposition encoding.
    MuxSuperposition,
}

impl fmt::Display for CompactionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompactionStrategy::ClusterCentroids => write!(f, "cluster_centroids"),
            CompactionStrategy::AttentionWeighted => write!(f, "attention_weighted"),
            CompactionStrategy::SpectralProjection => write!(f, "spectral_projection"),
            CompactionStrategy::BfcfRegionBlend => write!(f, "bfcf_region_blend"),
            CompactionStrategy::MuxSuperposition => write!(f, "mux_superposition"),
        }
    }
}

/// Metadata about a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionMeta {
    /// Which strategy was used.
    pub strategy: CompactionStrategy,
    /// Compression ratio (original_len / compacted_len).
    pub compression_ratio: f32,
    /// Quality score (0..1, cosine similarity proxy).
    pub quality_score: f32,
}

/// Compacted KV cache stored in f16 for memory efficiency.
///
/// Keys and values are stored as flat f16 buffers in `[seq_len, head_dim]` layout.
/// `position_offset` tracks the original position range for RoPE re-application.
/// `beta` holds the additive attention bias per latent for attention calibration.
#[derive(Debug, Clone)]
pub struct CompactKVCache {
    /// Compacted key cache — flat f16, shape `[compact_len * num_heads * head_dim]`.
    pub keys: Vec<f16>,
    /// Compacted value cache — flat f16, shape `[compact_len * num_heads * head_dim]`.
    pub values: Vec<f16>,
    /// Number of heads in the original cache.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Original start position (pre-compaction).
    pub position_offset: usize,
    /// Compaction metadata.
    pub meta: CompactionMeta,
    /// Additive attention bias per latent — calibrates attention to synthetic entries.
    /// Shape: `[compact_len]` (one scalar per latent token).
    /// Without β, the frozen model has no mechanism to upweight/downweight compact slots.
    pub beta: BetaBias,
}

impl CompactKVCache {
    /// Create a new empty compact cache with the given dimensions.
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            num_heads,
            head_dim,
            position_offset: 0,
            meta: CompactionMeta {
                strategy: CompactionStrategy::ClusterCentroids,
                compression_ratio: 1.0,
                quality_score: 1.0,
            },
            beta: BetaBias::zeros(0, 0),
        }
    }

    /// Returns the number of compacted tokens (sequence length).
    pub fn compact_len(&self) -> usize {
        let elements_per_token = self.num_heads * self.head_dim;
        match elements_per_token {
            0 => 0,
            n => self.keys.len() / n,
        }
    }

    /// Returns total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f16>()
    }

    /// Returns the effective compression ratio from metadata.
    pub fn compression_ratio(&self) -> f32 {
        self.meta.compression_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_display() {
        assert_eq!(
            CompactionStrategy::ClusterCentroids.to_string(),
            "cluster_centroids"
        );
        assert_eq!(
            CompactionStrategy::MuxSuperposition.to_string(),
            "mux_superposition"
        );
    }

    #[test]
    fn test_compact_cache_new() {
        let cache = CompactKVCache::new(8, 64);
        assert_eq!(cache.compact_len(), 0);
        assert_eq!(cache.num_heads, 8);
        assert_eq!(cache.head_dim, 64);
    }
}
