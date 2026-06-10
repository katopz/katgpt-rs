//! StillKV: Perceiver-based KV cache compaction — modelless (Plan 245).
//!
//! Compacts KV caches via learned cross-attention without model-specific knowledge.
//! Key insight: position-free compaction — un-rotate RoPE, compact in latent space,
//! re-rotate on retrieval.
//!
//! Strategies:
//! - **ClusterCentroids**: k-means-style cluster representatives
//! - **AttentionWeighted**: attention-score-weighted importance sampling
//! - **SpectralProjection**: PCA/SVD low-rank projection
//! - **BfcfRegionBlend**: BFCF region-weighted blending
//! - **MuxSuperposition**: multiplexed superposition encoding

pub mod compact_cache;
pub mod iterative;
pub mod perceiver;
pub mod position_free;
pub mod query_bank;

pub use compact_cache::{CompactKVCache, CompactionMeta, CompactionStrategy};
pub use iterative::{IterativeChunkCompactor, KVChunk};
pub use perceiver::{StillPerceiver, StillPerceiverConfig};
pub use position_free::PositionFreeCompactor;
pub use query_bank::QueryBank;

/// Compute cosine similarity between two flat f32 vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in (0..a.len()).step_by(4) {
        let end = (i + 4).min(a.len());
        for j in i..end {
            dot += a[j] * b[j];
            norm_a += a[j] * a[j];
            norm_b += b[j] * b[j];
        }
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use half::f16;

    /// T19: Position-free compaction round-trip.
    /// un-rotate → compact → re-rotate should approximately preserve semantics.
    #[test]
    fn test_position_free_compaction_roundtrip() {
        let head_dim = 16;
        let num_heads = 2;
        let seq_len = 32;
        let rope_theta = 10000.0;

        let compactor = PositionFreeCompactor::new(rope_theta, head_dim);

        // Create synthetic keys at position 0
        let original_f32: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let original_f16: Vec<f16> = original_f32.iter().map(|&v| f16::from_f32(v)).collect();

        // Un-rotate at pos 0 (should be identity since angle=0)
        let position_free = compactor.un_rotate_keys(&original_f16, 0);

        // Compact: just take first 16 tokens (simple truncation budget)
        let budget = 16;
        let compact_f32 = position_free[..budget * head_dim].to_vec();

        // Re-rotate at new position
        let new_start_pos = 0;
        let re_rotated_f16 = compactor.re_rotate_keys(&compact_f32, new_start_pos);

        // Verify: at pos 0, rotation is identity, so f16 round-trip should be exact
        for i in 0..budget * head_dim {
            let original = f16::from_f32(original_f32[i]);
            assert_eq!(re_rotated_f16[i], original, "Mismatch at index {}", i);
        }
    }

    /// T19b: Non-trivial position round-trip.
    /// un-rotate at pos 100, re-rotate at pos 100 should recover original.
    #[test]
    fn test_position_free_compaction_roundtrip_nontrivial_pos() {
        let head_dim = 16;
        let seq_len = 8;
        let rope_theta = 10000.0;
        let start_pos = 100;

        let compactor = PositionFreeCompactor::new(rope_theta, head_dim);

        let original_f32: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| (i as f32 * 0.3).cos())
            .collect();
        let original_f16: Vec<f16> = original_f32.iter().map(|&v| f16::from_f32(v)).collect();

        // Un-rotate at start_pos
        let position_free = compactor.un_rotate_keys(&original_f16, start_pos);

        // Re-rotate at same position
        let recovered_f16 = compactor.re_rotate_keys(&position_free, start_pos);

        // Should recover original within f16 precision
        for i in 0..seq_len * head_dim {
            let diff = (recovered_f16[i].to_f32() - original_f16[i].to_f32()).abs();
            assert!(
                diff < 0.01,
                "Round-trip error too large at index {}: {}",
                i,
                diff
            );
        }
    }

    /// T20: compact_into produces correct budget size via IterativeChunkCompactor.
    #[test]
    fn test_compact_into_correct_budget() {
        let chunk_size = 16;
        let num_heads = 2;
        let head_dim = 8;
        let compression_ratio = 4;
        let rope_theta = 10000.0;
        let tokens_per_elem = num_heads * head_dim;

        let compactor = IterativeChunkCompactor::new(
            chunk_size,
            0,
            num_heads,
            head_dim,
            CompactionStrategy::ClusterCentroids,
            rope_theta,
            compression_ratio,
        );

        // Create 32 tokens of data (2 chunks)
        let total_tokens = 32;
        let keys = vec![f16::from_f32(1.0); total_tokens * tokens_per_elem];
        let values = vec![f16::from_f32(2.0); total_tokens * tokens_per_elem];

        let chunks = compactor.split_into_chunks(&keys, &values, 0);
        assert_eq!(chunks.len(), 2);

        let budget = compactor.compact_budget();
        assert_eq!(budget, chunk_size / compression_ratio);

        // Compact first chunk
        let compacted = compactor.compact_chunk(&chunks[0], None, budget);
        assert_eq!(compacted.len, budget);
    }

    /// T21: Iterative compaction produces linear growth at rate 1/c.
    #[test]
    fn test_iterative_linear_growth() {
        let chunk_size = 16;
        let num_heads = 2;
        let head_dim = 8;
        let compression_ratio = 4;
        let rope_theta = 10000.0;
        let tokens_per_elem = num_heads * head_dim;

        let compactor = IterativeChunkCompactor::new(
            chunk_size,
            0,
            num_heads,
            head_dim,
            CompactionStrategy::ClusterCentroids,
            rope_theta,
            compression_ratio,
        );

        // Create 64 tokens (4 chunks)
        let total_tokens = 64;
        let keys: Vec<f16> = (0..total_tokens * tokens_per_elem)
            .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
            .collect();
        let values: Vec<f16> = (0..total_tokens * tokens_per_elem)
            .map(|i| f16::from_f32((i as f32 * 0.2).cos()))
            .collect();

        let chunks = compactor.split_into_chunks(&keys, &values, 0);
        assert_eq!(chunks.len(), 4);

        let stream_result = compactor.compact_stream(chunks);

        // Each compacted chunk should have budget = 16/4 = 4 tokens
        let budget = chunk_size / compression_ratio;
        for (i, chunk) in stream_result.iter().enumerate() {
            assert_eq!(
                chunk.len, budget,
                "Chunk {} has {} tokens, expected {}",
                i, chunk.len, budget
            );
        }

        // Total compact tokens = 4 chunks × 4 tokens = 16
        let total_compact: usize = stream_result.iter().map(|c| c.len).sum();
        assert_eq!(total_compact, 16);

        // Compression ratio: 64 original → 16 compact = 4x
        assert_eq!(total_tokens / total_compact, compression_ratio);
    }

    /// End-to-end: full pipeline with all strategies.
    #[test]
    fn test_full_pipeline_all_strategies() {
        let strategies = [
            CompactionStrategy::ClusterCentroids,
            CompactionStrategy::AttentionWeighted,
            CompactionStrategy::SpectralProjection,
            CompactionStrategy::BfcfRegionBlend,
            CompactionStrategy::MuxSuperposition,
        ];

        for strategy in strategies {
            let compactor = IterativeChunkCompactor::new(16, 0, 2, 8, strategy, 10000.0, 4);

            let keys = vec![f16::from_f32(1.0); 16 * 16];
            let values = vec![f16::from_f32(2.0); 16 * 16];
            let chunks = compactor.split_into_chunks(&keys, &values, 0);

            let budget = compactor.compact_budget();
            let compacted = compactor.compact_chunk(&chunks[0], None, budget);

            assert_eq!(
                compacted.len, budget,
                "Strategy {:?} produced wrong budget",
                strategy
            );
            assert!(!compacted.keys.is_empty());
            assert!(!compacted.values.is_empty());
        }
    }
}
