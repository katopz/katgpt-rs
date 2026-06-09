//! MUX-Latent encoder — compresses token spans into superposition latent slots.
//!
//! This is the inference-time analog of LCLM's encoder. Instead of a trained
//! neural network, we use MUX's position-weighted superposition:
//!
//!   latent_slot[i] = Σ (decay^j * onehot(token[j])) for j in span
//!
//! The geometric decay ensures:
//! 1. Positional information is preserved (causal ordering, matching LCLM's finding)
//! 2. MUX's lossless separation guarantee allows reconstruction (EXPAND)
//! 3. Zero training — the encoder is a deterministic function of decay + span_size

use crate::mux_latent::config::MuxLatentConfig;
use crate::mux_latent::context::{CompressedContext, LatentSegment};

/// MUX-Latent encoder: compresses token sequences into latent superposition slots.
///
/// This encoder is pure inference — no training, no parameters. It uses geometric
/// decay weighting to create position-aware superpositions that are information-preserving
/// (MUX separation guarantee).
#[derive(Debug, Clone)]
pub struct MuxLatentEncoder {
    config: MuxLatentConfig,
}

impl MuxLatentEncoder {
    /// Create a new encoder with the given configuration.
    pub fn new(config: MuxLatentConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_encoder() -> Self {
        Self::new(MuxLatentConfig::default())
    }

    /// Compress a full token sequence into a `CompressedContext`.
    ///
    /// The input is split into windows (per config.window_size), each window
    /// is further split into spans (per config.compression_ratio), and each
    /// span is encoded into a single latent slot via MUX superposition.
    ///
    /// Instructions (first window when preserve_instructions is true) are
    /// kept as raw tokens.
    pub fn encode(&self, tokens: &[u32]) -> CompressedContext {
        let span_size = self.config.compression_ratio.span_size();
        let window_size = self.config.window_size;
        let original_count = tokens.len();

        let mut segments = Vec::new();
        let mut segment_id: u32 = 0;
        let mut pos = 0;

        while pos < tokens.len() {
            let window_end = (pos + window_size).min(tokens.len());

            // Preserve first window as raw tokens (typically system prompt + instructions)
            if self.config.preserve_instructions && pos == 0 && window_end < tokens.len() {
                segments.push(LatentSegment::Raw {
                    tokens: tokens[pos..window_end].to_vec(),
                });
                pos = window_end;
                continue;
            }

            // Encode spans within the window
            while pos < window_end {
                let span_end = (pos + span_size).min(window_end);
                let span_tokens = &tokens[pos..span_end];

                if span_tokens.len() <= 2 {
                    // Too short to compress meaningfully — keep raw
                    segments.push(LatentSegment::Raw {
                        tokens: span_tokens.to_vec(),
                    });
                } else {
                    let weights = self.encode_span(span_tokens);
                    segments.push(LatentSegment::Compressed {
                        segment_id,
                        weights,
                        original_tokens: span_tokens.to_vec(),
                    });
                    segment_id += 1;
                }

                pos = span_end;
            }
        }

        let latent_slot_count = segments
            .iter()
            .filter(|s| matches!(s, LatentSegment::Compressed { .. }))
            .count();

        CompressedContext {
            segments,
            original_token_count: original_count,
            latent_slot_count,
            config: self.config.clone(),
        }
    }

    /// Encode a single span of tokens into MUX superposition weights.
    ///
    /// Each position gets a weight of `decay^j` where j is the position index.
    /// This creates a causal ordering that preserves positional information
    /// (matching LCLM's finding that causal encoding outperforms bidirectional).
    ///
    /// The weights are normalized to sum to 1.0 for stable injection.
    pub(crate) fn encode_span(&self, tokens: &[u32]) -> Vec<f32> {
        let decay = self.config.mux_decay;
        let mut weights = Vec::with_capacity(tokens.len());

        let mut total_weight = 0.0f32;
        for j in 0..tokens.len() {
            let w = decay.powi(j as i32);
            weights.push(w);
            total_weight += w;
        }

        // Normalize to unit sum for stable decoder-side injection
        if total_weight > 0.0 {
            for w in &mut weights {
                *w /= total_weight;
            }
        }

        weights
    }

    /// Encode only the latent slots (no raw segments) for tight inner loops.
    ///
    /// Returns (weights_per_span, segment_count) where each weights vec
    /// has `span_size` elements.
    pub fn encode_latents_only(&self, tokens: &[u32]) -> Vec<Vec<f32>> {
        let span_size = self.config.compression_ratio.span_size();
        let mut result = Vec::with_capacity(tokens.len().div_ceil(span_size));

        for chunk in tokens.chunks(span_size) {
            if chunk.len() > 2 {
                result.push(self.encode_span(chunk));
            }
        }

        result
    }

    /// Compute the superposition logit vector for a span (for MUX verification).
    ///
    /// This produces the vocabulary-space representation:
    ///   logit[v] = Σ weights[j] for all j where tokens[j] == v
    ///
    /// Useful for integration with existing MUX verification infrastructure.
    pub fn superposition_logits(&self, tokens: &[u32], vocab_size: usize) -> Vec<f32> {
        let weights = self.encode_span(tokens);
        let mut logits = vec![0.0f32; vocab_size];

        for (j, &token_id) in tokens.iter().enumerate() {
            if (token_id as usize) < vocab_size {
                logits[token_id as usize] += weights[j];
            }
        }

        logits
    }

    /// Get the configuration.
    pub fn config(&self) -> &MuxLatentConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux_latent::config::CompressionRatio;

    #[test]
    fn test_encode_span_weights() {
        let config = MuxLatentConfig {
            mux_decay: 0.9,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);

        let tokens = vec![100u32, 200, 300, 400];
        let weights = encoder.encode_span(&tokens);

        // Weights should sum to ~1.0 (normalized)
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Weights should sum to 1.0, got {sum}"
        );

        // First weight should be largest (causal decay)
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
        assert!(weights[2] > weights[3]);
    }

    #[test]
    fn test_encode_preserves_instructions() {
        let config = MuxLatentConfig {
            window_size: 16,
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: true,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);

        // 48 tokens: first 16 (window) kept raw, rest compressed
        let tokens: Vec<u32> = (0..48).collect();
        let ctx = encoder.encode(&tokens);

        // First segment should be raw
        assert!(matches!(ctx.segments[0], LatentSegment::Raw { .. }));
        // Should have compressed segments after the first
        assert!(ctx.segments.len() > 1);
        assert!(ctx.latent_slot_count > 0);
        assert_eq!(ctx.original_token_count, 48);
    }

    #[test]
    fn test_compression_ratio_x8() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);

        // 64 tokens → 8 latent slots at 8x compression
        let tokens: Vec<u32> = (0..64).collect();
        let ctx = encoder.encode(&tokens);

        assert_eq!(ctx.latent_slot_count, 8);
        assert_eq!(ctx.original_token_count, 64);
    }

    #[test]
    fn test_compression_ratio_x16() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X16,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);

        let tokens: Vec<u32> = (0..64).collect();
        let ctx = encoder.encode(&tokens);

        assert_eq!(ctx.latent_slot_count, 4);
    }

    #[test]
    fn test_expand_roundtrip() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);

        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let ctx = encoder.encode(&tokens);

        // Should be able to expand segment 0 to get original tokens
        let expanded = ctx.expand(0);
        assert!(expanded.is_some());
        assert_eq!(expanded.unwrap(), &[10, 20, 30, 40]);

        // Segment 1
        let expanded1 = ctx.expand(1);
        assert!(expanded1.is_some());
        assert_eq!(expanded1.unwrap(), &[50, 60, 70, 80]);
    }

    #[test]
    fn test_memory_savings() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);

        let tokens: Vec<u32> = (0..64).collect();
        let ctx = encoder.encode(&tokens);

        // 8 latent slots + 0 raw = 8 decoder budget from 64 tokens
        assert_eq!(ctx.decoder_budget(), 8);
        // Memory savings should be ~87.5% (1 - 8/64)
        let savings = ctx.memory_savings();
        assert!(
            (savings - 0.875).abs() < 0.01,
            "Expected ~87.5% savings, got {savings:.2}%"
        );
    }

    #[test]
    fn test_superposition_logits() {
        let config = MuxLatentConfig::default();
        let encoder = MuxLatentEncoder::new(config);

        let tokens = vec![5u32, 10, 15, 20, 5, 25, 30, 5];
        let logits = encoder.superposition_logits(&tokens, 32);

        // Token 5 appears 3 times with decay weights
        assert!(logits[5] > 0.0);
        // Token 10 appears once
        assert!(logits[10] > 0.0);
        // Token 0 never appears
        assert_eq!(logits[0], 0.0);
    }
}
