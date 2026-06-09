//! Compressed context representation — the output of MUX-Latent encoding.
//!
//! A `CompressedContext` holds both latent (compressed) and raw (uncompressed)
//! segments. This hybrid structure preserves high-value tokens (instructions,
//! system prompts) while compressing bulk context.

use crate::mux_latent::config::MuxLatentConfig;

/// A single segment of the compressed context.
#[derive(Debug, Clone)]
pub enum LatentSegment {
    /// Compressed span: one latent slot representing N input tokens.
    /// The superposition weights encode positional information via geometric decay.
    Compressed {
        /// Index of this segment for EXPAND retrieval.
        segment_id: u32,
        /// Superposition weights — one f32 per position in the span.
        /// Ordered by token position, decay-weighted.
        weights: Vec<f32>,
        /// Token IDs in the original span (kept for EXPAND recovery).
        original_tokens: Vec<u32>,
    },
    /// Uncompressed span: raw tokens kept as-is.
    Raw {
        /// Token IDs in original order.
        tokens: Vec<u32>,
    },
}

impl LatentSegment {
    /// Number of latent slots this segment occupies.
    pub fn latent_count(&self) -> usize {
        match self {
            Self::Compressed { .. } => 1,
            Self::Raw { tokens } => tokens.len(),
        }
    }

    /// Number of original tokens represented by this segment.
    pub fn original_token_count(&self) -> usize {
        match self {
            Self::Compressed {
                original_tokens, ..
            } => original_tokens.len(),
            Self::Raw { tokens } => tokens.len(),
        }
    }

    /// Segment ID, if this is a compressed segment.
    pub fn segment_id(&self) -> Option<u32> {
        match self {
            Self::Compressed { segment_id, .. } => Some(*segment_id),
            Self::Raw { .. } => None,
        }
    }
}

/// Compressed context with mixed latent and raw segments.
///
/// This is the output of the MUX-Latent encoder and the input to the
/// decoder-side injection pipeline.
#[derive(Debug, Clone)]
pub struct CompressedContext {
    /// Segments in order (some compressed, some raw).
    pub segments: Vec<LatentSegment>,

    /// Total number of original tokens before compression.
    pub original_token_count: usize,

    /// Total number of latent slots (compressed segments only).
    pub latent_slot_count: usize,

    /// The config used to produce this compression.
    pub config: MuxLatentConfig,
}

impl CompressedContext {
    /// Effective compression ratio achieved.
    pub fn compression_ratio(&self) -> f32 {
        if self.latent_slot_count == 0 {
            return 1.0;
        }
        // Ratio of original tokens to (latent slots + raw token count)
        let total_after = self
            .segments
            .iter()
            .map(|s| s.latent_count())
            .sum::<usize>();
        self.original_token_count as f32 / total_after.max(1) as f32
    }

    /// Retrieve original tokens for a compressed segment by ID (EXPAND).
    ///
    /// This is the inference-time analog of LCLM's EXPAND(i) tool.
    /// Returns None if segment_id doesn't exist or segment is raw.
    pub fn expand(&self, segment_id: u32) -> Option<&[u32]> {
        self.segments.iter().find_map(|seg| match seg {
            LatentSegment::Compressed {
                segment_id: id,
                original_tokens,
                ..
            } if *id == segment_id => Some(original_tokens.as_slice()),
            _ => None,
        })
    }

    /// Total decoder-side token budget (latent slots + raw tokens).
    pub fn decoder_budget(&self) -> usize {
        self.segments.iter().map(|s| s.latent_count()).sum()
    }

    /// Memory savings: original tokens vs decoder budget.
    pub fn memory_savings(&self) -> f32 {
        let budget = self.decoder_budget();
        if budget == 0 {
            return 0.0;
        }
        1.0 - (budget as f32 / self.original_token_count as f32)
    }
}
