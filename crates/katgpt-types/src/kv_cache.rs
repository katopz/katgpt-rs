//! Quantized KV cache trait — shared extension point for all backends.
//!
//! Originally lived in `katgpt-rs/src/types.rs` (Plan 123 / Issue 015 Phase 1).
//! Promoted to `katgpt-types` so that every KV backend crate
//! (`katgpt-kv`, sibling engine crates, future downstream) can implement
//! against a stable leaf-crate interface without depending on the root
//! `katgpt-rs` crate.
//!
//! The `compact_into` extension (which historically gated on
//! `crate::still_kv::CompactionStrategy`) is intentionally NOT in this
//! trait. It lives in `katgpt-kv` behind the `still_kv` feature as the
//! `CompactableKVCache` extension trait, so this leaf crate stays free
//! of any KV-storage-concrete type coupling.

/// Shared interface for quantized KV caches.
///
/// Enables `transformer::forward_quantized` to work with any compression
/// backend (TurboQuant, SpectralQuant, OscKV, ShardKV, KVarN, or future
/// methods). Backends implement this trait; the inference loop stays
/// backend-agnostic.
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
