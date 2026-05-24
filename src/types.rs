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
