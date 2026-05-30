// katgpt-rs types: re-exports from katgpt-core + project-specific items.
//
// All shared types (Config, Rng, InferenceOverrides, math utilities, LoRA,
// DomainLatent) are defined in katgpt-core and re-exported here.
// This module adds only katgpt-rs-specific items.

// Re-export all shared types from core
pub use katgpt_core::types::*;

// ---------------------------------------------------------------------------
// QuantizedKVCache — katgpt-rs only
// ---------------------------------------------------------------------------

/// Shared interface for quantized KV caches.
///
/// Enables [`crate::transformer::forward_quantized`] to work with any
/// compression backend (TurboQuant, SpectralQuant, or future methods).
pub trait QuantizedKVCache {
    /// Quantize and store a key vector at given layer and position.
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]);
    /// Quantize and store a value vector at given layer and position.
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]);
    /// Dequantize a key into a pre-allocated buffer (zero-alloc hot path).
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    /// Dequantize a value into a pre-allocated buffer (zero-alloc hot path).
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    /// Reset cache for a new sequence.
    fn reset(&mut self);
    /// Current write position.
    fn pos(&self) -> usize;
    /// Set the current write position.
    fn set_pos(&mut self, pos: usize);
}

// ---------------------------------------------------------------------------
// AsymmetricKVConfig — asymmetric K/V cache compression (Plan 123)
// ---------------------------------------------------------------------------

/// Asymmetric KV cache configuration.
///
/// Research 081 proves V-side compression is quality-free (softmax amplifies K errors
/// O(e^ε) but V errors only scale linearly O(w·ε)). Recommended: key_bits=8, val_bits=3.
///
/// Plan 123: Asymmetric K/V Cache Compression — GOAT proof.
#[derive(Clone, Debug)]
pub struct AsymmetricKVConfig {
    /// Bits for key quantization (precision-critical due to softmax amplification).
    pub key_bits: u8,
    /// Bits for value quantization (quality-free compression opportunity).
    pub val_bits: u8,
}

impl Default for AsymmetricKVConfig {
    fn default() -> Self {
        Self {
            key_bits: 8,
            val_bits: 3,
        }
    }
}

impl AsymmetricKVConfig {
    /// Create a new asymmetric config.
    pub fn new(key_bits: u8, val_bits: u8) -> Self {
        Self { key_bits, val_bits }
    }

    /// Symmetric config (same bits for K and V).
    pub fn symmetric(bits: u8) -> Self {
        Self {
            key_bits: bits,
            val_bits: bits,
        }
    }

    /// Whether this config is asymmetric (key_bits ≠ val_bits).
    #[inline]
    pub fn is_asymmetric(&self) -> bool {
        self.key_bits != self.val_bits
    }

    /// Theoretical compression ratio vs fp32 (32 bits per element).
    /// Returns ratio of fp32 size to quantized size.
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        let fp32_bits = 32.0;
        let avg_bits = (self.key_bits as f32 + self.val_bits as f32) / 2.0;
        fp32_bits / avg_bits
    }

    /// Total bits per KV pair.
    #[inline]
    pub fn total_bits(&self) -> u8 {
        self.key_bits + self.val_bits
    }
}
