//! EXPAND(i) tool — selective decompression of latent context segments.
//!
//! This is the inference-time analog of LCLM's agent scaffolding where the
//! model can request expansion of specific compressed segments. Since MUX
//! superposition stores original tokens alongside weights, expansion is O(1).

use crate::mux_latent::context::CompressedContext;

/// Result of expanding a compressed segment.
#[derive(Debug, Clone)]
pub struct ExpandedSegment {
    /// The segment ID that was expanded.
    pub segment_id: u32,
    /// The original tokens recovered from the compressed segment.
    pub tokens: Vec<u32>,
    /// The superposition weights that were used for this segment.
    pub weights: Vec<f32>,
}

/// Expand a single compressed segment by ID.
///
/// Returns None if the segment doesn't exist or is already raw.
/// This is O(n) where n is the number of segments (typically small).
pub fn expand_segment(ctx: &CompressedContext, segment_id: u32) -> Option<ExpandedSegment> {
    for seg in &ctx.segments {
        if let crate::mux_latent::context::LatentSegment::Compressed {
            segment_id: id,
            weights,
            original_tokens,
        } = seg
        {
            if *id == segment_id {
                return Some(ExpandedSegment {
                    segment_id: *id,
                    tokens: original_tokens.clone(),
                    weights: weights.clone(),
                });
            }
        }
    }
    None
}

/// Expand all compressed segments back to raw tokens.
///
/// This fully decompresses the context, useful for verification or fallback.
pub fn expand_all(ctx: &CompressedContext) -> Vec<u32> {
    let mut result = Vec::with_capacity(ctx.original_token_count);
    for seg in &ctx.segments {
        match seg {
            crate::mux_latent::context::LatentSegment::Compressed {
                original_tokens, ..
            } => result.extend_from_slice(original_tokens),
            crate::mux_latent::context::LatentSegment::Raw { tokens } => {
                result.extend_from_slice(tokens)
            }
        }
    }
    result
}

/// Select which segments to expand based on a relevance query.
///
/// Given a set of query token IDs and a relevance threshold, finds segments
/// that have high overlap with the query. This is a simple baseline; more
/// sophisticated methods can use embedding similarity.
///
/// Returns segment IDs to expand, sorted by relevance (highest first).
pub fn select_segments_to_expand(
    ctx: &CompressedContext,
    query_tokens: &[u32],
    top_k: usize,
) -> Vec<u32> {
    let query_set: std::collections::HashSet<u32> = query_tokens.iter().copied().collect();

    let mut scored: Vec<(u32, usize)> = Vec::new();

    for seg in &ctx.segments {
        if let crate::mux_latent::context::LatentSegment::Compressed {
            segment_id,
            original_tokens,
            ..
        } = seg
        {
            let overlap = original_tokens
                .iter()
                .filter(|t| query_set.contains(t))
                .count();
            if overlap > 0 {
                scored.push((*segment_id, overlap));
            }
        }
    }

    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored.into_iter().take(top_k).map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux_latent::config::{CompressionRatio, MuxLatentConfig};
    use crate::mux_latent::encoder::MuxLatentEncoder;

    #[test]
    fn test_expand_segment() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);
        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let ctx = encoder.encode(&tokens);

        let expanded = expand_segment(&ctx, 0).unwrap();
        assert_eq!(expanded.tokens, vec![10, 20, 30, 40]);
        assert_eq!(expanded.segment_id, 0);
    }

    #[test]
    fn test_expand_all_roundtrip() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);
        let tokens: Vec<u32> = (0..32).collect();
        let ctx = encoder.encode(&tokens);

        let expanded = expand_all(&ctx);
        assert_eq!(expanded, tokens);
    }

    #[test]
    fn test_select_segments_by_query() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);

        // 8 segments of 4 tokens each: [0-3], [4-7], [8-11], ...
        let tokens: Vec<u32> = (0..32).collect();
        let ctx = encoder.encode(&tokens);

        // Query contains tokens from segments 2 and 5
        let query = vec![9u32, 10, 20, 21];
        let to_expand = select_segments_to_expand(&ctx, &query, 3);

        assert!(to_expand.contains(&2)); // segment 2 has tokens [8,9,10,11]
        assert!(to_expand.contains(&5)); // segment 5 has tokens [20,21,22,23]
    }

    #[test]
    fn test_expand_nonexistent_returns_none() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };
        let encoder = MuxLatentEncoder::new(config);
        let tokens: Vec<u32> = (0..8).collect();
        let ctx = encoder.encode(&tokens);

        assert!(expand_segment(&ctx, 999).is_none());
    }
}
