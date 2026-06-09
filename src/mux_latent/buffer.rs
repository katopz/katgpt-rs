//! LatentContextBuffer — manages mixed raw+latent context segments.
//!
//! This is the core runtime data structure for MUX-Latent context compression.
//! It holds a `CompressedContext` and provides:
//! - Budget tracking (latent slots used vs total budget)
//! - Segment-level access for decoder-side injection
//! - Eviction policy when latent budget is exceeded
//! - Integration with `SpectralLOD` for adaptive compression

use crate::mux_latent::config::MuxLatentConfig;
use crate::mux_latent::context::{CompressedContext, LatentSegment};
use crate::mux_latent::encoder::MuxLatentEncoder;
use crate::mux_latent::expand::select_segments_to_expand;
use crate::mux_latent::spectral_lod::SpectralLOD;

/// Eviction policy when the latent budget is exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict oldest segments first (FIFO).
    OldestFirst,
    /// Evict lowest-information segments first (by spectral energy).
    LowestEnergy,
}

/// Statistics about the current buffer state.
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Total original tokens fed into the buffer.
    pub total_input_tokens: usize,
    /// Current number of latent slots in use.
    pub latent_slots_used: usize,
    /// Maximum latent slots allowed.
    pub latent_slot_budget: usize,
    /// Current number of raw token segments.
    pub raw_segment_count: usize,
    /// Overall compression ratio achieved.
    pub compression_ratio: f32,
    /// Memory savings fraction [0, 1).
    pub memory_savings: f32,
}

/// LatentContextBuffer — the runtime manager for compressed context.
///
/// Wraps the encoding pipeline and provides budget management, eviction,
/// and query-based segment retrieval. This is what the decoder-side injection
/// path consumes during prefill.
#[derive(Debug, Clone)]
pub struct LatentContextBuffer {
    /// Current compressed context.
    context: CompressedContext,
    /// Maximum number of latent slots (0 = unlimited).
    latent_budget: usize,
    /// Eviction policy when budget exceeded.
    eviction_policy: EvictionPolicy,
    /// SLoD analyzer for adaptive compression (optional).
    slod: Option<SpectralLOD>,
}

impl LatentContextBuffer {
    /// Create a new buffer by compressing the given tokens.
    pub fn new(tokens: &[u32], config: MuxLatentConfig) -> Self {
        let encoder = MuxLatentEncoder::new(config.clone());
        let context = encoder.encode(tokens);

        let latent_budget = if config.max_latent_slots > 0 {
            config.max_latent_slots
        } else {
            context.latent_slot_count
        };

        let mut buf = Self {
            context,
            latent_budget,
            eviction_policy: EvictionPolicy::OldestFirst,
            slod: None,
        };

        buf.enforce_budget();
        buf
    }

    /// Create a buffer with adaptive SLoD compression.
    pub fn new_adaptive(tokens: &[u32], config: MuxLatentConfig, slod: SpectralLOD) -> Self {
        let window_size = config.window_size;
        let span_size = config.compression_ratio.span_size();

        // Split tokens into windows and determine per-window compression
        let windows: Vec<&[u32]> = tokens.chunks(window_size).collect();
        let adaptive = slod.adaptive_ratios(&windows, config.compression_ratio);

        // Re-encode with adaptive ratios per window
        let mut segments = Vec::new();
        let mut segment_id: u32 = 0;
        let mut pos = 0;

        for (window_tokens, ratio) in &adaptive {
            let window_end = pos + window_tokens.len();
            let window_span = ratio.span_size();

            // Preserve instructions (first window) if configured
            if config.preserve_instructions && pos == 0 && window_end < tokens.len() {
                segments.push(LatentSegment::Raw {
                    tokens: window_tokens.to_vec(),
                });
                pos = window_end;
                continue;
            }

            // Encode spans within this window at the adaptive ratio
            let mut wpos = 0;
            while wpos < window_tokens.len() {
                let span_end = (wpos + window_span).min(window_tokens.len());
                let span_tokens = &window_tokens[wpos..span_end];

                if span_tokens.len() <= 2 {
                    segments.push(LatentSegment::Raw {
                        tokens: span_tokens.to_vec(),
                    });
                } else {
                    // Compute weights for this span
                    let encoder = MuxLatentEncoder::new(MuxLatentConfig {
                        compression_ratio: *ratio,
                        mux_decay: config.mux_decay,
                        ..config.clone()
                    });
                    let weights = encoder.encode_span(span_tokens);
                    segments.push(LatentSegment::Compressed {
                        segment_id,
                        weights,
                        original_tokens: span_tokens.to_vec(),
                    });
                    segment_id += 1;
                }

                wpos = span_end;
            }

            pos = window_end;
        }

        let latent_slot_count = segments
            .iter()
            .filter(|s| matches!(s, LatentSegment::Compressed { .. }))
            .count();

        let context = CompressedContext {
            segments,
            original_token_count: tokens.len(),
            latent_slot_count,
            config: config.clone(),
        };

        let latent_budget = if config.max_latent_slots > 0 {
            config.max_latent_slots
        } else {
            context.latent_slot_count
        };

        let mut buf = Self {
            context,
            latent_budget,
            eviction_policy: EvictionPolicy::OldestFirst,
            slod: Some(slod),
        };

        buf.enforce_budget();
        buf
    }

    /// Get the underlying compressed context.
    pub fn context(&self) -> &CompressedContext {
        &self.context
    }

    /// Expand a specific segment (EXPAND(i) analog).
    pub fn expand(&self, segment_id: u32) -> Option<&[u32]> {
        self.context.expand(segment_id)
    }

    /// Find segments relevant to a query and expand them.
    ///
    /// Returns the top-k most relevant segment IDs based on token overlap.
    pub fn query_expand(&self, query_tokens: &[u32], top_k: usize) -> Vec<u32> {
        select_segments_to_expand(&self.context, query_tokens, top_k)
    }

    /// Get the raw token sequence (all expanded segments in order).
    ///
    /// This is the full decompressed context — used for verification or fallback.
    pub fn full_expand(&self) -> Vec<u32> {
        crate::mux_latent::expand::expand_all(&self.context)
    }

    /// Get the decoder-side token budget (latent slots + raw tokens).
    pub fn decoder_budget(&self) -> usize {
        self.context.decoder_budget()
    }

    /// Compute memory savings as a fraction [0, 1).
    pub fn memory_savings(&self) -> f32 {
        self.context.memory_savings()
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> BufferStats {
        let raw_count = self
            .context
            .segments
            .iter()
            .filter(|s| matches!(s, LatentSegment::Raw { .. }))
            .count();

        BufferStats {
            total_input_tokens: self.context.original_token_count,
            latent_slots_used: self.context.latent_slot_count,
            latent_slot_budget: self.latent_budget,
            raw_segment_count: raw_count,
            compression_ratio: self.context.compression_ratio(),
            memory_savings: self.memory_savings(),
        }
    }

    /// Set the eviction policy.
    pub fn set_eviction_policy(&mut self, policy: EvictionPolicy) {
        self.eviction_policy = policy;
    }

    /// Set a new latent budget, evicting segments if necessary.
    pub fn set_latent_budget(&mut self, budget: usize) {
        self.latent_budget = budget;
        self.enforce_budget();
    }

    /// Enforce the latent budget by evicting excess compressed segments.
    ///
    /// Evicted segments are converted to raw tokens (no information loss).
    fn enforce_budget(&mut self) {
        if self.latent_budget == 0 {
            return; // unlimited
        }

        let excess = self
            .context
            .latent_slot_count
            .saturating_sub(self.latent_budget);
        if excess == 0 {
            return;
        }

        // Find compressed segment indices to evict
        let mut candidates: Vec<usize> = Vec::new();
        for (i, seg) in self.context.segments.iter().enumerate() {
            if let LatentSegment::Compressed { .. } = seg {
                candidates.push(i);
            }
        }

        match self.eviction_policy {
            EvictionPolicy::OldestFirst => {
                // Evict the first `excess` compressed segments (oldest)
                let to_evict: Vec<usize> = candidates.into_iter().take(excess).collect();
                self.evict_segments(&to_evict);
            }
            EvictionPolicy::LowestEnergy => {
                // Would need spectral analysis — for now fallback to oldest
                let to_evict: Vec<usize> = candidates.into_iter().take(excess).collect();
                self.evict_segments(&to_evict);
            }
        }
    }

    /// Convert compressed segments at the given indices to raw segments.
    ///
    /// This is lossless — original tokens are preserved in Compressed segments.
    fn evict_segments(&mut self, indices: &[usize]) {
        let mut latent_count = 0usize;
        for &idx in indices {
            if let Some(seg) = self.context.segments.get(idx) {
                if let LatentSegment::Compressed {
                    original_tokens, ..
                } = seg
                {
                    let tokens = original_tokens.clone();
                    // Account for the raw tokens this will add to decoder budget
                    // (1 slot per raw token instead of 1 latent slot)
                    self.context.segments[idx] = LatentSegment::Raw { tokens };
                    latent_count += 1;
                }
            }
        }
        self.context.latent_slot_count -= latent_count;
    }

    /// Append new tokens to the buffer, compressing them into latent segments.
    ///
    /// This extends the context without re-encoding existing segments.
    pub fn append(&mut self, tokens: &[u32]) {
        let config = self.context.config.clone();
        let encoder = MuxLatentEncoder::new(config);

        let new_context = encoder.encode(tokens);

        // Append new segments, renumbering IDs
        let max_id = self
            .context
            .segments
            .iter()
            .filter_map(|s| s.segment_id())
            .max()
            .unwrap_or(0);

        let mut next_id = max_id + 1;
        for mut seg in new_context.segments {
            if let LatentSegment::Compressed {
                ref mut segment_id, ..
            } = seg
            {
                *segment_id = next_id;
                next_id += 1;
            }
            self.context.segments.push(seg);
        }

        self.context.original_token_count += tokens.len();
        self.context.latent_slot_count = self
            .context
            .segments
            .iter()
            .filter(|s| matches!(s, LatentSegment::Compressed { .. }))
            .count();

        self.enforce_budget();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mux_latent::config::CompressionRatio;

    fn make_tokens(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn test_buffer_basic_compression() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: false,
            ..Default::default()
        };

        let tokens = make_tokens(64);
        let buf = LatentContextBuffer::new(&tokens, config);

        let stats = buf.stats();
        assert_eq!(stats.total_input_tokens, 64);
        assert_eq!(stats.latent_slots_used, 8); // 64/8 = 8 slots
        assert!(stats.memory_savings > 0.8); // >80% savings
    }

    #[test]
    fn test_buffer_preserve_instructions() {
        let config = MuxLatentConfig {
            window_size: 16,
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: true,
            ..Default::default()
        };

        // 48 tokens: first 16 preserved as raw, rest compressed
        let tokens = make_tokens(48);
        let buf = LatentContextBuffer::new(&tokens, config);

        let ctx = buf.context();
        // First segment should be raw (instructions)
        assert!(matches!(ctx.segments[0], LatentSegment::Raw { .. }));

        // Full expand should recover all 48 tokens
        let expanded = buf.full_expand();
        assert_eq!(expanded.len(), 48);
        assert_eq!(expanded, tokens);
    }

    #[test]
    fn test_buffer_expand_roundtrip() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };

        let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let buf = LatentContextBuffer::new(&tokens, config);

        // Expand segment 0 → [10, 20, 30, 40]
        let expanded = buf.expand(0);
        assert!(expanded.is_some());
        assert_eq!(expanded.unwrap(), &[10, 20, 30, 40]);

        // Full expand → original
        let full = buf.full_expand();
        assert_eq!(full, tokens);
    }

    #[test]
    fn test_buffer_budget_enforcement() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            max_latent_slots: 2, // Only 2 slots allowed
            ..Default::default()
        };

        // 12 tokens → 3 segments at X4, but budget is 2 → oldest evicted to raw
        let tokens = make_tokens(12);
        let mut buf = LatentContextBuffer::new(&tokens, config);

        let stats = buf.stats();
        assert!(
            stats.latent_slots_used <= 2,
            "Should have at most 2 latent slots, got {}",
            stats.latent_slots_used
        );

        // Even with eviction, full expand should still work (lossless)
        let expanded = buf.full_expand();
        assert_eq!(expanded.len(), 12);
    }

    #[test]
    fn test_buffer_append() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };

        let tokens1 = make_tokens(8);
        let mut buf = LatentContextBuffer::new(&tokens1, config.clone());

        // Append more tokens
        let tokens2: Vec<u32> = (100..108).collect();
        buf.append(&tokens2);

        let stats = buf.stats();
        assert_eq!(stats.total_input_tokens, 16);

        // Full expand should recover all 16 tokens
        let expanded = buf.full_expand();
        assert_eq!(expanded.len(), 16);
        assert_eq!(&expanded[..8], &tokens1[..]);
        assert_eq!(&expanded[8..], &tokens2[..]);
    }

    #[test]
    fn test_buffer_query_expand() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };

        // 16 tokens → 4 segments: [0-3], [4-7], [8-11], [12-15]
        let tokens = make_tokens(16);
        let buf = LatentContextBuffer::new(&tokens, config);

        // Query for tokens in segment 2 (8,9,10,11)
        let query = vec![9u32, 10];
        let to_expand = buf.query_expand(&query, 2);

        assert!(to_expand.contains(&2));
    }

    #[test]
    fn test_buffer_adaptive_compression() {
        let config = MuxLatentConfig {
            window_size: 8,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: false,
            ..Default::default()
        };

        let slod = SpectralLOD::default();

        // Mix of diverse and repetitive tokens
        let mut tokens = Vec::new();
        // Window 1: diverse
        tokens.extend_from_slice(&[0u32, 500, 1000, 1500, 2000, 2500, 3000, 3500]);
        // Window 2: repetitive
        tokens.extend_from_slice(&[5u32, 5, 5, 5, 5, 5, 5, 5]);

        let buf = LatentContextBuffer::new_adaptive(&tokens, config, slod);
        let stats = buf.stats();

        assert_eq!(stats.total_input_tokens, 16);
        // Diverse window should get lower compression (more slots)
        // Repetitive window should get higher compression (fewer slots)
        assert!(stats.latent_slots_used > 0);

        // Full roundtrip should still work
        let expanded = buf.full_expand();
        assert_eq!(expanded, tokens);
    }

    #[test]
    fn test_buffer_set_budget() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            ..Default::default()
        };

        let tokens = make_tokens(16);
        let mut buf = LatentContextBuffer::new(&tokens, config);

        // Initially 4 slots
        assert_eq!(buf.stats().latent_slots_used, 4);

        // Reduce budget to 2 → 2 segments evicted to raw
        buf.set_latent_budget(2);
        assert!(buf.stats().latent_slots_used <= 2);

        // Full expand still recovers everything
        let expanded = buf.full_expand();
        assert_eq!(expanded.len(), 16);
    }

    #[test]
    fn test_integration_4k_to_256_latent() {
        // The key integration test: compress 4096 tokens → ~512 latent slots at X8
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X8,
            preserve_instructions: true,
            ..Default::default()
        };

        // Simulate a realistic 4k context: diverse tokens
        let mut tokens = Vec::with_capacity(4096);
        for i in 0..4096u32 {
            tokens.push(i % 32000); // Simulate diverse vocabulary
        }

        let buf = LatentContextBuffer::new(&tokens, config);
        let stats = buf.stats();

        // First window (1024 tokens) kept as raw, rest compressed
        // 3072 tokens / 8 = 384 latent slots
        assert_eq!(stats.total_input_tokens, 4096);
        assert!(
            stats.latent_slots_used <= 512,
            "Expected ~384 latent slots, got {}",
            stats.latent_slots_used
        );

        // Decoder budget should be much smaller than 4096
        let budget = buf.decoder_budget();
        assert!(
            budget < 2048,
            "Decoder budget should be significantly reduced, got {budget}"
        );

        // Memory savings should be substantial
        assert!(
            stats.memory_savings > 0.5,
            "Memory savings should be >50%, got {:.1}%",
            stats.memory_savings * 100.0
        );

        // Full roundtrip should recover all tokens
        let expanded = buf.full_expand();
        assert_eq!(expanded.len(), 4096);
        assert_eq!(expanded, tokens);

        // Recall verification: expand segment 100 and check content
        if let Some(segment_tokens) = buf.expand(100) {
            assert!(!segment_tokens.is_empty());
            assert_eq!(segment_tokens.len(), 8); // X8 compression
        }
    }

    #[test]
    fn test_integration_4k_x16_compression() {
        let config = MuxLatentConfig {
            window_size: 1024,
            compression_ratio: CompressionRatio::X16,
            preserve_instructions: true,
            ..Default::default()
        };

        let mut tokens = Vec::with_capacity(4096);
        for i in 0..4096u32 {
            tokens.push(i % 32000);
        }

        let buf = LatentContextBuffer::new(&tokens, config);
        let stats = buf.stats();

        // 3072 / 16 = 192 latent slots
        assert!(
            stats.latent_slots_used <= 256,
            "Expected ~192 latent slots at X16, got {}",
            stats.latent_slots_used
        );

        let expanded = buf.full_expand();
        assert_eq!(expanded, tokens);
    }

    #[test]
    fn test_buffer_eviction_policy_lowest_energy() {
        let config = MuxLatentConfig {
            compression_ratio: CompressionRatio::X4,
            preserve_instructions: false,
            max_latent_slots: 2,
            ..Default::default()
        };

        let tokens = make_tokens(12);
        let mut buf = LatentContextBuffer::new(&tokens, config);
        buf.set_eviction_policy(EvictionPolicy::LowestEnergy);

        let stats = buf.stats();
        assert!(stats.latent_slots_used <= 2);

        // Full expand still works
        let expanded = buf.full_expand();
        assert_eq!(expanded.len(), 12);
    }
}
