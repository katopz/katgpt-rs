//! Bridge function for wiring `MixedPrefillSequence` into `forward_prefill`.
//!
//! This module provides `forward_prefill_with_compression` — a wrapper that
//! decomposes a mixed raw+latent prefill sequence into raw token IDs suitable
//! for the existing `forward_prefill` path, plus metadata tracking which
//! positions correspond to latent slots for future mid-layer injection.
//!
//! This is the Phase 3 bridge: it doesn't modify `transformer.rs` internals,
//! but prepares everything needed for the deep integration.

use crate::mux_latent::inject::{
    CompressionSummary, LatentPrefillAdapter, MixedPrefillSequence, PrefillEntry,
};

/// Metadata about which positions in the prefill are latent slots.
///
/// When `forward_prefill_with_compression` returns, this struct tells the caller
/// which positions in the logit output correspond to compressed spans rather than
/// raw tokens. Future mid-layer injection can use this to substitute superposition
/// weights at the correct positions.
#[derive(Debug, Clone)]
pub struct CompressionMetadata {
    /// Indices in the token array that correspond to latent entries.
    pub latent_indices: Vec<usize>,
    /// For each latent index, the number of original tokens it represents.
    pub original_token_counts: Vec<usize>,
    /// For each latent index, the segment ID for EXPAND retrieval.
    pub segment_ids: Vec<u32>,
    /// For each latent index, the superposition weights.
    pub weights: Vec<Vec<f32>>,
    /// Summary of the compression benefit.
    pub summary: CompressionSummary,
}

/// Result of calling `forward_prefill_with_compression`.
///
/// Contains the raw token IDs to pass to `forward_prefill`, plus metadata
/// about which positions are latent slots.
#[derive(Debug, Clone)]
pub struct CompressedPrefillPlan {
    /// Token IDs to pass to `forward_prefill`.
    /// For `PrefillEntry::Raw`: the actual token ID.
    /// For `PrefillEntry::Latent`: the anchor token (first original token).
    pub token_ids: Vec<usize>,
    /// Metadata about latent positions.
    pub compression: CompressionMetadata,
}

/// Decompose a `MixedPrefillSequence` into a plan for `forward_prefill`.
///
/// This is the main bridge function. It:
/// 1. Extracts raw token IDs from the mixed sequence (using anchor tokens for latent slots)
/// 2. Builds `CompressionMetadata` noting which positions are latent
/// 3. Returns a `CompressedPrefillPlan` that can be fed directly to `forward_prefill`
///
/// The caller then:
/// 1. Passes `plan.token_ids` to `forward_prefill` as the `tokens` parameter
/// 2. Uses `plan.compression.latent_indices` for mid-layer injection
/// 3. Uses `plan.compression.summary` for telemetry/metrics
pub fn forward_prefill_with_compression(seq: &MixedPrefillSequence) -> CompressedPrefillPlan {
    let token_ids = LatentPrefillAdapter::raw_token_ids(seq);
    let token_ids_usize: Vec<usize> = token_ids.iter().map(|&t| t as usize).collect();

    let mut latent_indices = Vec::new();
    let mut original_token_counts = Vec::new();
    let mut segment_ids = Vec::new();
    let mut weights = Vec::new();

    for (i, entry) in seq.entries.iter().enumerate() {
        if let PrefillEntry::Latent {
            segment_id,
            weights: w,
            original_tokens,
            ..
        } = entry
        {
            latent_indices.push(i);
            original_token_counts.push(original_tokens.len());
            segment_ids.push(*segment_id);
            weights.push(w.clone());
        }
    }

    let summary = CompressionSummary::from_sequence(seq);

    CompressedPrefillPlan {
        token_ids: token_ids_usize,
        compression: CompressionMetadata {
            latent_indices,
            original_token_counts,
            segment_ids,
            weights,
            summary,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux_latent::config::{CompressionRatio, MuxLatentConfig};
    use crate::mux_latent::encoder::MuxLatentEncoder;

    #[test]
    fn test_compressed_prefill_plan_all_latent() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config.clone());
        let tokens: Vec<u32> = (0..64).collect();
        let ctx = encoder.encode(&tokens);

        let adapter = LatentPrefillAdapter::new(config);
        let seq = adapter.to_prefill_sequence(&ctx);

        let plan = forward_prefill_with_compression(&seq);

        // 64 tokens at X8 → 8 latent slots → 8 token IDs (anchors)
        assert_eq!(plan.token_ids.len(), 8);
        assert_eq!(plan.compression.latent_indices.len(), 8);
        assert_eq!(plan.compression.summary.latent_slots, 8);
        assert_eq!(plan.compression.summary.raw_tokens, 0);

        // Each latent index should have weights
        for w in &plan.compression.weights {
            assert!(!w.is_empty());
            let sum: f32 = w.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Weights should sum to ~1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_compressed_prefill_plan_mixed() {
        let config = MuxLatentConfig {
            window_size: 8,
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: true,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config.clone());
        let tokens: Vec<u32> = (0..24).collect();
        let ctx = encoder.encode(&tokens);

        let adapter = LatentPrefillAdapter::new(config);
        let seq = adapter.to_prefill_sequence(&ctx);

        let plan = forward_prefill_with_compression(&seq);

        // First 8 entries are raw, rest are latent
        assert!(plan.compression.latent_indices.len() > 0);
        // Latent indices should all be >= 8
        for &idx in &plan.compression.latent_indices {
            assert!(idx >= 8, "Latent index {idx} should be >= 8");
        }
        // Token IDs for raw entries should match original tokens
        assert_eq!(plan.token_ids[0], 0);
        assert_eq!(plan.token_ids[7], 7);
    }

    #[test]
    fn test_compressed_prefill_plan_4k() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: true,
            ..Default::default()
        };

        let tokens: Vec<u32> = (0..4096).map(|t| t % 32000).collect();
        let encoder = MuxLatentEncoder::new(config.clone());
        let ctx = encoder.encode(&tokens);

        let adapter = LatentPrefillAdapter::new(config);
        let seq = adapter.to_prefill_sequence(&ctx);

        let plan = forward_prefill_with_compression(&seq);

        // Should have significantly fewer token IDs than 4096
        assert!(
            plan.token_ids.len() < 2048,
            "Expected < 2048 token IDs, got {}",
            plan.token_ids.len()
        );

        // Summary should show good compression
        assert!(plan.compression.summary.kv_savings > 0.5);
        assert!(plan.compression.summary.estimated_ttft_reduction < 0.5);
    }
}
