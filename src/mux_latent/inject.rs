//! Decoder-side injection of compressed latent context into the prefill path.
//!
//! This module bridges `CompressedContext` (from the MUX-Latent encoder) with
//! the existing transformer prefill path. It provides:
//!
//! - `LatentPrefillAdapter`: Prepares compressed segments for injection into KV cache
//! - Mixed raw+latent prefill: Raw tokens get standard prefill, latent slots get
//!   mid-layer injection (similar to `DomainLatent` but per-slot)
//!
//! The key insight is that latent slots don't need KV cache entries for every
//! original token — they only need one cache entry per compressed span. This
//! is where the memory savings come from.

use crate::mux_latent::config::MuxLatentConfig;
use crate::mux_latent::context::{CompressedContext, LatentSegment};

/// A pre-mixed sequence of prefill entries: some raw (standard tokens),
/// some latent (compressed spans represented as single KV entries).
#[derive(Debug, Clone)]
pub struct MixedPrefillSequence {
    /// Entries in order of prefill processing.
    pub entries: Vec<PrefillEntry>,
    /// Original total token count (before compression).
    pub original_token_count: usize,
    /// Effective prefill length (raw tokens + latent slots).
    /// This is the actual length that will be prefilled into the KV cache.
    pub effective_prefill_len: usize,
}

/// A single entry in the mixed prefill sequence.
#[derive(Debug, Clone)]
pub enum PrefillEntry {
    /// Standard raw token — processed normally.
    Raw {
        /// Token ID.
        token_id: u32,
        /// Original position in the uncompressed sequence.
        original_pos: usize,
    },
    /// Latent slot — injected at mid-layer with superposition weights.
    /// Represents N compressed tokens as a single KV cache entry.
    Latent {
        /// Segment ID for EXPAND retrieval.
        segment_id: u32,
        /// Superposition weights (one per original token in the span).
        weights: Vec<f32>,
        /// Original tokens in this compressed span.
        original_tokens: Vec<u32>,
        /// Position this slot occupies in the compressed sequence.
        compressed_pos: usize,
    },
}

/// Adapter that converts `CompressedContext` into a `MixedPrefillSequence`
/// suitable for the prefill path.
#[derive(Debug)]
pub struct LatentPrefillAdapter {
    config: MuxLatentConfig,
}

impl LatentPrefillAdapter {
    /// Create a new adapter with the given configuration.
    pub fn new(config: MuxLatentConfig) -> Self {
        Self { config }
    }

    /// Convert a `CompressedContext` into a mixed prefill sequence.
    ///
    /// Raw segments are expanded into individual `PrefillEntry::Raw` entries.
    /// Compressed segments become single `PrefillEntry::Latent` entries.
    ///
    /// The effective prefill length is the count of entries, which should be
    /// much smaller than `original_token_count` when compression is active.
    pub fn to_prefill_sequence(&self, ctx: &CompressedContext) -> MixedPrefillSequence {
        let mut entries = Vec::new();
        let mut original_pos = 0usize;
        let mut compressed_pos = 0usize;

        for segment in &ctx.segments {
            match segment {
                LatentSegment::Raw { tokens } => {
                    for &token_id in tokens {
                        entries.push(PrefillEntry::Raw {
                            token_id,
                            original_pos,
                        });
                        original_pos += 1;
                        compressed_pos += 1;
                    }
                }
                LatentSegment::Compressed {
                    segment_id,
                    weights,
                    original_tokens,
                } => {
                    entries.push(PrefillEntry::Latent {
                        segment_id: *segment_id,
                        weights: weights.clone(),
                        original_tokens: original_tokens.clone(),
                        compressed_pos,
                    });
                    original_pos += original_tokens.len();
                    compressed_pos += 1;
                }
            }
        }

        let effective_prefill_len = entries.len();

        MixedPrefillSequence {
            entries,
            original_token_count: ctx.original_token_count,
            effective_prefill_len,
        }
    }

    /// Extract only the raw token IDs from a mixed sequence.
    ///
    /// Useful for the embedding lookup — latent slots need special handling.
    pub fn raw_token_ids(seq: &MixedPrefillSequence) -> Vec<u32> {
        seq.entries
            .iter()
            .filter_map(|e| match e {
                PrefillEntry::Raw { token_id, .. } => Some(*token_id),
                PrefillEntry::Latent {
                    original_tokens, ..
                } => {
                    // Use the first original token as the embedding anchor
                    original_tokens.first().copied()
                }
            })
            .collect()
    }

    /// Count the KV cache savings compared to uncompressed prefill.
    ///
    /// Returns (original_kv_entries, compressed_kv_entries, savings_fraction).
    pub fn kv_savings(seq: &MixedPrefillSequence) -> (usize, usize, f32) {
        let original = seq.original_token_count;
        let compressed = seq.effective_prefill_len;
        let savings = if original > 0 {
            1.0 - (compressed as f32 / original as f32)
        } else {
            0.0
        };
        (original, compressed, savings)
    }

    /// Get indices of latent entries in the sequence (for mid-layer injection).
    pub fn latent_indices(seq: &MixedPrefillSequence) -> Vec<usize> {
        seq.entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| match e {
                PrefillEntry::Latent { .. } => Some(i),
                PrefillEntry::Raw { .. } => None,
            })
            .collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &MuxLatentConfig {
        &self.config
    }
}

/// Summary of the compression benefit for prefill.
#[derive(Debug, Clone)]
pub struct CompressionSummary {
    /// Original token count.
    pub original_tokens: usize,
    /// Effective prefill entries (raw + latent slots).
    pub effective_entries: usize,
    /// Number of latent slots.
    pub latent_slots: usize,
    /// Number of raw tokens.
    pub raw_tokens: usize,
    /// KV cache savings as fraction [0, 1).
    pub kv_savings: f32,
    /// Estimated TTFT reduction factor (heuristic).
    /// Approximate: TTFT scales roughly linearly with prefill length.
    pub estimated_ttft_reduction: f32,
}

impl CompressionSummary {
    /// Compute summary from a mixed prefill sequence.
    pub fn from_sequence(seq: &MixedPrefillSequence) -> Self {
        let mut latent_slots = 0usize;
        let mut raw_tokens = 0usize;

        for entry in &seq.entries {
            match entry {
                PrefillEntry::Latent { .. } => latent_slots += 1,
                PrefillEntry::Raw { .. } => raw_tokens += 1,
            }
        }

        let (original, compressed, kv_savings) = LatentPrefillAdapter::kv_savings(seq);

        let estimated_ttft_reduction = if original > 0 {
            compressed as f32 / original as f32
        } else {
            1.0
        };

        Self {
            original_tokens: original,
            effective_entries: compressed,
            latent_slots,
            raw_tokens,
            kv_savings,
            estimated_ttft_reduction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux_latent::config::CompressionRatio;
    use crate::mux_latent::encoder::MuxLatentEncoder;

    #[test]
    fn test_prefill_adapter_basic() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };

        let encoder = MuxLatentEncoder::new(config.clone());
        let tokens: Vec<u32> = (0..16).collect();
        let ctx = encoder.encode(&tokens);

        let adapter = LatentPrefillAdapter::new(config);
        let seq = adapter.to_prefill_sequence(&ctx);

        // 16 tokens at X4 → 4 latent slots
        assert_eq!(seq.original_token_count, 16);
        assert_eq!(seq.effective_prefill_len, 4);
        assert_eq!(seq.entries.len(), 4);

        // All entries should be latent (no raw preserved)
        for entry in &seq.entries {
            assert!(matches!(entry, PrefillEntry::Latent { .. }));
        }
    }

    #[test]
    fn test_prefill_adapter_preserve_instructions() {
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

        // First 8 tokens preserved as raw, rest compressed
        assert_eq!(seq.original_token_count, 24);

        // First 8 entries should be Raw
        for i in 0..8 {
            assert!(
                matches!(seq.entries[i], PrefillEntry::Raw { .. }),
                "Entry {i} should be Raw"
            );
        }

        // Remaining entries should be Latent
        let latent_count = seq
            .entries
            .iter()
            .filter(|e| matches!(e, PrefillEntry::Latent { .. }))
            .count();
        assert!(latent_count > 0);
    }

    #[test]
    fn test_prefill_adapter_kv_savings() {
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

        let (original, compressed, savings) = LatentPrefillAdapter::kv_savings(&seq);
        assert_eq!(original, 64);
        assert_eq!(compressed, 8);
        assert!(
            (savings - 0.875).abs() < 0.01,
            "Expected ~87.5% savings, got {savings:.2}"
        );
    }

    #[test]
    fn test_prefill_adapter_latent_indices() {
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

        let indices = LatentPrefillAdapter::latent_indices(&seq);
        // Indices should start at 8 (after the raw instruction tokens)
        assert!(indices[0] >= 8);
    }

    #[test]
    fn test_compression_summary() {
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

        let summary = CompressionSummary::from_sequence(&seq);
        assert_eq!(summary.original_tokens, 64);
        assert_eq!(summary.effective_entries, 8);
        assert_eq!(summary.latent_slots, 8);
        assert_eq!(summary.raw_tokens, 0);
        assert!(summary.kv_savings > 0.8);
        assert!(
            (summary.estimated_ttft_reduction - 0.125).abs() < 0.01,
            "Expected TTFT reduction ~12.5%, got {:.1}% ({})",
            summary.estimated_ttft_reduction * 100.0,
            summary.estimated_ttft_reduction,
        );
    }

    #[test]
    fn test_raw_token_ids_extraction() {
        let config = MuxLatentConfig {
            window_size: 4,
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: true,
            ..Default::default()
        };

        let encoder = MuxLatentEncoder::new(config.clone());
        let tokens: Vec<u32> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let ctx = encoder.encode(&tokens);

        let adapter = LatentPrefillAdapter::new(config);
        let seq = adapter.to_prefill_sequence(&ctx);

        let raw_ids = LatentPrefillAdapter::raw_token_ids(&seq);
        // First 4 are raw tokens [100, 200, 300, 400]
        // Then latent slot uses first token [500] as anchor
        assert_eq!(raw_ids.len(), seq.effective_prefill_len);
        assert_eq!(raw_ids[0], 100);
        assert_eq!(raw_ids[3], 400);
    }

    #[test]
    fn test_integration_4k_prefill_sequence() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: true,
            ..Default::default()
        };

        let mut tokens = Vec::with_capacity(4096);
        for i in 0..4096u32 {
            tokens.push(i % 32000);
        }

        let encoder = MuxLatentEncoder::new(config.clone());
        let ctx = encoder.encode(&tokens);

        let adapter = LatentPrefillAdapter::new(config);
        let seq = adapter.to_prefill_sequence(&ctx);

        let summary = CompressionSummary::from_sequence(&seq);

        // 1024 raw + 384 latent = 1408 effective entries
        assert_eq!(summary.original_tokens, 4096);
        assert!(
            summary.effective_entries < 2048,
            "Should be much less than 4096, got {}",
            summary.effective_entries
        );
        assert!(summary.kv_savings > 0.5);
    }
}
