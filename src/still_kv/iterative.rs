//! Iterative chunk-based KV cache compaction.
//!
//! Processes KV cache in fixed-size chunks with a lookahead buffer,
//! enabling streaming compaction for very long sequences.
//!
//! Pipeline per chunk:
//! 1. Un-rotate RoPE via `PositionFreeCompactor`
//! 2. Generate queries via `QueryBank`
//! 3. Cross-attend via `StillPerceiver::forward`
//! 4. Project to compact keys/values via `forward_projected`
//! 5. Re-rotate compacted keys with new positions
//! 6. Convert values back to f16

use half::f16;

use super::compact_cache::CompactionStrategy;
use super::perceiver::{StillPerceiver, StillPerceiverConfig};
use super::position_free::PositionFreeCompactor;
use super::query_bank::create_query_bank;

/// A chunk of KV cache data for iterative processing.
#[derive(Debug, Clone)]
pub struct KVChunk {
    /// Key data for this chunk — flat f16, shape `[chunk_size * num_heads * head_dim]`.
    pub keys: Vec<f16>,
    /// Value data for this chunk — flat f16, shape `[chunk_size * num_heads * head_dim]`.
    pub values: Vec<f16>,
    /// Starting position of this chunk.
    pub start_pos: usize,
    /// Number of tokens in this chunk.
    pub len: usize,
}

impl KVChunk {
    /// Create a new empty chunk.
    pub fn new(start_pos: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            start_pos,
            len: 0,
        }
    }

    /// Returns true if the chunk has no tokens.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Iterative chunk-based KV cache compactor.
///
/// Processes the KV cache in fixed-size chunks, maintaining a lookahead buffer
/// for context-aware compaction decisions. This enables memory-bounded
/// compaction for arbitrarily long sequences.
#[derive(Debug, Clone)]
pub struct IterativeChunkCompactor {
    /// Number of tokens per processing chunk.
    pub chunk_size: usize,
    /// Number of lookahead tokens for context awareness.
    pub lookahead_buffer: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Compaction strategy for query bank generation.
    pub strategy: CompactionStrategy,
    /// RoPE base frequency (theta).
    pub rope_theta: f32,
    /// Compression ratio: original tokens / compact tokens per chunk.
    pub compression_ratio: usize,
}

impl IterativeChunkCompactor {
    /// Create a new iterative compactor.
    ///
    /// # Arguments
    /// * `chunk_size` - Tokens per processing chunk
    /// * `lookahead_buffer` - Lookahead tokens for context awareness
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `strategy` - Compaction strategy for query bank generation
    /// * `rope_theta` - RoPE base frequency
    /// * `compression_ratio` - Compression ratio (e.g., 4 means 4x compression)
    pub fn new(
        chunk_size: usize,
        lookahead_buffer: usize,
        num_heads: usize,
        head_dim: usize,
        strategy: CompactionStrategy,
        rope_theta: f32,
        compression_ratio: usize,
    ) -> Self {
        Self {
            chunk_size,
            lookahead_buffer,
            num_heads,
            head_dim,
            strategy,
            rope_theta,
            compression_ratio: compression_ratio.max(1),
        }
    }

    /// Compute the budget (number of compact tokens) per chunk.
    ///
    /// `budget = chunk_size / compression_ratio`
    pub fn compact_budget(&self) -> usize {
        if self.compression_ratio == 0 {
            return self.chunk_size;
        }
        self.chunk_size / self.compression_ratio
    }

    /// Split a full KV cache into chunks for iterative processing.
    ///
    /// # Arguments
    /// * `keys` - Flat f16 key buffer
    /// * `values` - Flat f16 value buffer
    /// * `start_pos` - Starting position
    ///
    /// # Returns
    /// Iterator-friendly Vec of KVChunk.
    pub fn split_into_chunks(
        &self,
        keys: &[f16],
        values: &[f16],
        start_pos: usize,
    ) -> Vec<KVChunk> {
        let tokens_per_element = self.num_heads * self.head_dim;
        let total_tokens = match tokens_per_element {
            0 => return Vec::new(),
            t => keys.len() / t,
        };

        let mut chunks = Vec::new();
        let mut pos = start_pos;
        let mut offset = 0;

        while offset < total_tokens {
            let chunk_len = self.chunk_size.min(total_tokens - offset);
            let elem_start = offset * tokens_per_element;
            let elem_end = (offset + chunk_len) * tokens_per_element;

            chunks.push(KVChunk {
                keys: keys[elem_start..elem_end].to_vec(),
                values: values[elem_start..elem_end].to_vec(),
                start_pos: pos,
                len: chunk_len,
            });

            offset += chunk_len;
            pos += chunk_len;
        }

        chunks
    }

    // -----------------------------------------------------------------------
    // T15: Per-chunk compaction using perceiver + query bank
    // -----------------------------------------------------------------------

    /// Compact a single chunk using the perceiver pipeline.
    ///
    /// Pipeline:
    /// 1. Un-rotate RoPE via PositionFreeCompactor
    /// 2. Generate queries via QueryBank
    /// 3. Cross-attend via StillPerceiver::forward_projected
    /// 4. Re-rotate compacted keys with new positions
    /// 5. Convert values back to f16
    ///
    /// # Arguments
    /// * `chunk` - Current chunk to compact
    /// * `lookahead` - Optional lookahead chunk for context (T16)
    /// * `budget` - Target number of compact tokens
    ///
    /// # Returns
    /// Compacted chunk.
    pub fn compact_chunk(
        &self,
        chunk: &KVChunk,
        lookahead: Option<&KVChunk>,
        budget: usize,
    ) -> KVChunk {
        let kv_dim = self.num_heads * self.head_dim;
        if chunk.is_empty() || kv_dim == 0 || budget == 0 {
            return KVChunk::new(chunk.start_pos);
        }

        // T16: If lookahead is present, concatenate chunk keys with lookahead keys
        // for richer context during compaction.
        let (combined_keys_f16, combined_start_pos) = match lookahead {
            Some(la) if !la.is_empty() => {
                let mut k = Vec::with_capacity(chunk.keys.len() + la.keys.len());
                k.extend_from_slice(&chunk.keys);
                k.extend_from_slice(&la.keys);
                (k, chunk.start_pos)
            }
            _ => (chunk.keys.clone(), chunk.start_pos),
        };

        // Step 1: Un-rotate RoPE from keys (position-free space)
        let pos_free_compactor = PositionFreeCompactor::new(self.rope_theta, kv_dim);
        let unrotated_keys =
            pos_free_compactor.un_rotate_keys(&combined_keys_f16, combined_start_pos);

        // Step 2: Generate queries via query bank
        let query_bank = create_query_bank(self.strategy, kv_dim);
        let queries = query_bank.generate_queries(&unrotated_keys, budget);

        // If query bank returned nothing, fall back to truncated output
        if queries.is_empty() {
            return self.truncate_chunk(chunk, budget, kv_dim);
        }

        // Step 3+4: Cross-attend and project to compact keys/values
        let perceiver = self.build_perceiver(budget, kv_dim);
        let (compact_keys_f32, compact_values_f32) =
            perceiver.forward_projected(&unrotated_keys, &queries);

        // Step 4: Re-rotate compacted keys with new positions
        // The compacted tokens start at `new_start_pos` which the caller sets
        // For now we use the chunk's start_pos; compact_stream() adjusts this.
        let compact_keys_f16 =
            pos_free_compactor.re_rotate_keys(&compact_keys_f32, chunk.start_pos);

        // Step 5: Convert compact values to f16 (values don't need RoPE rotation)
        let compact_values_f16: Vec<f16> = compact_values_f32
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();

        // T16: When lookahead was used, we compacted combined data but only
        // keep budget tokens from the original chunk portion.
        // The perceiver already produced exactly `budget` tokens of output,
        // so the output is already sized correctly.
        let actual_len = compact_keys_f16.len() / kv_dim;

        KVChunk {
            keys: compact_keys_f16,
            values: compact_values_f16,
            start_pos: chunk.start_pos,
            len: actual_len,
        }
    }

    // -----------------------------------------------------------------------
    // T17: compact_stream — multi-chunk compaction with position tracking
    // -----------------------------------------------------------------------

    /// Compact a stream of chunks with correct position offset tracking.
    ///
    /// Each chunk is compacted independently (with optional lookahead from
    /// the next chunk). Position offsets are accumulated so that the
    /// compacted chunks form a contiguous sequence.
    ///
    /// # Arguments
    /// * `chunks` - Ordered chunks to compact
    ///
    /// # Returns
    /// Compacted chunks with correct start_pos values.
    pub fn compact_stream(&self, chunks: Vec<KVChunk>) -> Vec<KVChunk> {
        if chunks.is_empty() {
            return Vec::new();
        }

        let budget = self.compact_budget();
        let mut result = Vec::with_capacity(chunks.len());
        let mut accumulated_compact_len: usize = 0;

        for i in 0..chunks.len() {
            let lookahead = if i + 1 < chunks.len() {
                Some(&chunks[i + 1])
            } else {
                None
            };

            let mut compacted = self.compact_chunk(&chunks[i], lookahead, budget);

            // T17: Update start_pos to account for accumulated compaction.
            // new_start_pos = accumulated_compact_len
            compacted.start_pos = accumulated_compact_len;
            accumulated_compact_len += compacted.len;

            result.push(compacted);
        }

        result
    }

    // -----------------------------------------------------------------------
    // T18: compact_with_checkpoint — integration point for segment checkpoints
    // -----------------------------------------------------------------------

    /// Compact chunks with position tracking, suitable for integration with
    /// segment checkpoint systems.
    ///
    /// This is a convenience wrapper around `compact_stream()` that accepts
    /// an iterator of chunks. The position offsets are tracked internally.
    ///
    /// # Arguments
    /// * `chunks` - Ordered chunks to compact
    ///
    /// # Returns
    /// Compacted chunks with monotonically increasing start_pos.
    pub fn compact_with_checkpoint(
        &self,
        chunks: impl IntoIterator<Item = KVChunk>,
    ) -> Vec<KVChunk> {
        self.compact_stream(chunks.into_iter().collect())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Build a StillPerceiver configured for this compactor's dimensions.
    fn build_perceiver(&self, budget: usize, kv_dim: usize) -> StillPerceiver {
        let config = StillPerceiverConfig::with_kv_dim(kv_dim, budget, kv_dim);
        StillPerceiver::new(config)
    }

    /// Fallback: truncate chunk to budget tokens (original stub behavior).
    fn truncate_chunk(&self, chunk: &KVChunk, budget: usize, kv_dim: usize) -> KVChunk {
        let actual_budget = budget.min(chunk.len);
        let elem_end = actual_budget * kv_dim;

        KVChunk {
            keys: chunk.keys[..elem_end].to_vec(),
            values: chunk.values[..elem_end].to_vec(),
            start_pos: chunk.start_pos,
            len: actual_budget,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a compactor with default test params.
    fn test_compactor() -> IterativeChunkCompactor {
        IterativeChunkCompactor::new(
            8, // chunk_size
            4, // lookahead_buffer
            2, // num_heads
            4, // head_dim
            CompactionStrategy::ClusterCentroids,
            10000.0,
            2, // compression_ratio
        )
    }

    /// Helper: create a chunk filled with a constant value.
    fn make_chunk(
        start_pos: usize,
        len: usize,
        num_heads: usize,
        head_dim: usize,
        val: f32,
    ) -> KVChunk {
        let total = len * num_heads * head_dim;
        KVChunk {
            keys: vec![f16::from_f32(val); total],
            values: vec![f16::from_f32(val + 1.0); total],
            start_pos,
            len,
        }
    }

    #[test]
    fn test_kv_chunk_new() {
        let chunk = KVChunk::new(10);
        assert!(chunk.is_empty());
        assert_eq!(chunk.start_pos, 10);
    }

    #[test]
    fn test_split_into_chunks() {
        let compactor = IterativeChunkCompactor::new(
            4,
            2,
            2,
            4,
            CompactionStrategy::ClusterCentroids,
            10000.0,
            2,
        );
        // 8 tokens × 2 heads × 4 dim = 64 elements
        let keys = vec![f16::from_f32(1.0); 64];
        let values = vec![f16::from_f32(2.0); 64];
        let chunks = compactor.split_into_chunks(&keys, &values, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len, 4);
        assert_eq!(chunks[1].len, 4);
    }

    // -----------------------------------------------------------------------
    // T15 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_budget() {
        let compactor = test_compactor();
        // chunk_size=8, compression_ratio=2 → budget=4
        assert_eq!(compactor.compact_budget(), 4);
    }

    #[test]
    fn test_compact_budget_no_panic_on_zero_ratio() {
        let mut compactor = test_compactor();
        compactor.compression_ratio = 0;
        // Should return chunk_size (no compression)
        assert_eq!(compactor.compact_budget(), compactor.chunk_size);
    }

    #[test]
    fn test_compact_chunk_produces_budget_size() {
        let compactor = test_compactor();
        // chunk_size=8, num_heads=2, head_dim=4 → kv_dim=8
        // compression_ratio=2 → budget=4
        let chunk = make_chunk(0, 8, 2, 4, 1.0);
        let budget = compactor.compact_budget();

        let compacted = compactor.compact_chunk(&chunk, None, budget);

        // Compacted chunk should have exactly budget tokens
        assert_eq!(compacted.len, budget, "compacted len should equal budget");
        assert_eq!(
            compacted.keys.len(),
            budget * compactor.num_heads * compactor.head_dim,
            "keys buffer should match budget × kv_dim"
        );
        assert_eq!(
            compacted.values.len(),
            budget * compactor.num_heads * compactor.head_dim,
            "values buffer should match budget × kv_dim"
        );
    }

    #[test]
    fn test_compact_chunk_empty_input() {
        let compactor = test_compactor();
        let empty = KVChunk::new(0);
        let result = compactor.compact_chunk(&empty, None, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compact_chunk_zero_budget() {
        let compactor = test_compactor();
        let chunk = make_chunk(0, 8, 2, 4, 1.0);
        let result = compactor.compact_chunk(&chunk, None, 0);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // T16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_chunk_with_lookahead() {
        let compactor = test_compactor();
        let budget = compactor.compact_budget();

        let chunk = make_chunk(0, 8, 2, 4, 1.0);
        let lookahead = make_chunk(8, 8, 2, 4, 2.0);

        let without_la = compactor.compact_chunk(&chunk, None, budget);
        let with_la = compactor.compact_chunk(&chunk, Some(&lookahead), budget);

        // Both should produce budget-sized output
        assert_eq!(without_la.len, budget);
        assert_eq!(with_la.len, budget);

        // With lookahead, the perceiver attends to combined data,
        // so the output should differ from without lookahead
        let any_different = without_la
            .keys
            .iter()
            .zip(with_la.keys.iter())
            .any(|(a, b)| (a.to_f32() - b.to_f32()).abs() > 1e-6);
        assert!(
            any_different,
            "lookahead should produce different compaction than without"
        );
    }

    #[test]
    fn test_compact_chunk_lookahead_size_still_budget() {
        let compactor = test_compactor();
        let budget = compactor.compact_budget();

        let chunk = make_chunk(0, 8, 2, 4, 1.0);
        let lookahead = make_chunk(8, 4, 2, 4, 5.0);

        let result = compactor.compact_chunk(&chunk, Some(&lookahead), budget);

        // Output should still be exactly budget tokens, not budget + lookahead
        assert_eq!(result.len, budget);
    }

    // -----------------------------------------------------------------------
    // T17 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_stream_linear_growth() {
        let compactor = test_compactor();
        // chunk_size=8, compression_ratio=2 → budget=4 per chunk
        let chunks = vec![
            make_chunk(0, 8, 2, 4, 1.0),
            make_chunk(8, 8, 2, 4, 2.0),
            make_chunk(16, 8, 2, 4, 3.0),
        ];

        let compacted = compactor.compact_stream(chunks);
        assert_eq!(
            compacted.len(),
            3,
            "should produce one output per input chunk"
        );

        // Each compacted chunk should have budget tokens
        for (i, c) in compacted.iter().enumerate() {
            assert_eq!(c.len, 4, "chunk {} should have budget tokens", i);
        }
    }

    #[test]
    fn test_compact_stream_position_offsets() {
        let compactor = test_compactor();
        // chunk_size=8, compression_ratio=2 → budget=4 per chunk
        let chunks = vec![
            make_chunk(0, 8, 2, 4, 1.0),
            make_chunk(8, 8, 2, 4, 2.0),
            make_chunk(16, 8, 2, 4, 3.0),
        ];

        let compacted = compactor.compact_stream(chunks);

        // Position offsets should accumulate:
        // chunk 0: start_pos=0, len=4 → next starts at 4
        // chunk 1: start_pos=4, len=4 → next starts at 8
        // chunk 2: start_pos=8, len=4
        assert_eq!(compacted[0].start_pos, 0, "first chunk starts at 0");
        assert_eq!(
            compacted[1].start_pos, 4,
            "second chunk starts after first compacted"
        );
        assert_eq!(
            compacted[2].start_pos, 8,
            "third chunk starts after two compacted"
        );
    }

    #[test]
    fn test_compact_stream_empty_input() {
        let compactor = test_compactor();
        let result = compactor.compact_stream(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_compact_stream_single_chunk() {
        let compactor = test_compactor();
        let chunks = vec![make_chunk(0, 8, 2, 4, 1.0)];
        let compacted = compactor.compact_stream(chunks);

        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].start_pos, 0);
        assert_eq!(compacted[0].len, compactor.compact_budget());
    }

    // -----------------------------------------------------------------------
    // T18 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_with_checkpoint() {
        let compactor = test_compactor();
        let chunks = vec![make_chunk(0, 8, 2, 4, 1.0), make_chunk(8, 8, 2, 4, 2.0)];

        let compacted = compactor.compact_with_checkpoint(chunks);

        assert_eq!(compacted.len(), 2);
        assert_eq!(compacted[0].start_pos, 0);
        assert_eq!(compacted[1].start_pos, 4);
    }

    #[test]
    fn test_compact_with_checkpoint_position_offsets() {
        let compactor = test_compactor();
        let chunks: Vec<KVChunk> = (0..5)
            .map(|i| make_chunk(i * 8, 8, 2, 4, i as f32 + 1.0))
            .collect();

        let compacted = compactor.compact_with_checkpoint(chunks);

        // budget=4 per chunk, so positions: 0, 4, 8, 12, 16
        for (i, c) in compacted.iter().enumerate() {
            assert_eq!(
                c.start_pos,
                i * 4,
                "chunk {} start_pos should be {}",
                i,
                i * 4
            );
            assert_eq!(c.len, 4, "each chunk should have budget tokens");
        }
    }
}
