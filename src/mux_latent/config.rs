//! Configuration for MUX-Latent Context Compression.

/// Compression ratio determines how many input tokens map to one latent slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionRatio {
    /// 4 input tokens → 1 latent slot
    X4 = 4,
    /// 8 input tokens → 1 latent slot
    #[default]
    X8 = 8,
    /// 16 input tokens → 1 latent slot
    X16 = 16,
}

impl CompressionRatio {
    /// Number of input tokens per latent slot.
    pub fn span_size(&self) -> usize {
        *self as usize
    }
}

/// Configuration for the MUX-Latent context compression pipeline.
///
/// Mirrors LCLM's architecture: windowed encoding, configurable compression,
/// and decoder-side injection. All parameters are inference-time (no training).
#[derive(Debug, Clone)]
pub struct MuxLatentConfig {
    /// Number of input tokens processed per encoder window.
    /// LCLM paper found W=1024 optimal. Smaller windows use less memory.
    pub window_size: usize,

    /// Compression ratio (4x, 8x, or 16x).
    pub compression_ratio: CompressionRatio,

    /// Geometric decay rate for MUX superposition weights.
    /// Higher = more weight on first token in span, lower = more uniform.
    /// LCLM paper found causal (sequential) encoding optimal — we model this
    /// via positional decay in the superposition weights.
    pub mux_decay: f32,

    /// Maximum number of latent slots to keep in memory.
    /// Prevents unbounded memory growth for very long contexts.
    /// 0 = unlimited.
    pub max_latent_slots: usize,

    /// Whether to keep system/instruction tokens uncompressed.
    /// These are typically short and high-value, so no compression.
    pub preserve_instructions: bool,

    /// Layer index for domain_latent injection.
    /// Defaults to mid-layer (half of total layers).
    pub injection_layer: Option<usize>,
}

impl Default for MuxLatentConfig {
    fn default() -> Self {
        Self {
            window_size: 0,
            compression_ratio: CompressionRatio::default(),
            mux_decay: 0.9,
            max_latent_slots: 0,
            preserve_instructions: false,
            injection_layer: None,
        }
    }
}

impl MuxLatentConfig {
    /// Creates a config optimized for speed (aggressive compression).
    pub fn fast() -> Self {
        Self {
            compression_ratio: CompressionRatio::X16,
            mux_decay: 0.85,
            ..Default::default()
        }
    }

    /// Creates a config optimized for quality (conservative compression).
    pub fn quality() -> Self {
        Self {
            compression_ratio: CompressionRatio::X4,
            mux_decay: 0.95,
            ..Default::default()
        }
    }

    /// Number of latent slots produced from a given input length.
    pub fn latent_slot_count(&self, input_len: usize) -> usize {
        let total_spans = input_len.div_ceil(self.compression_ratio.span_size());
        if self.max_latent_slots > 0 {
            total_spans.min(self.max_latent_slots)
        } else {
            total_spans
        }
    }

    /// Number of windows for the encoder (each window processes window_size tokens).
    pub fn window_count(&self, input_len: usize) -> usize {
        input_len.div_ceil(self.window_size)
    }
}
