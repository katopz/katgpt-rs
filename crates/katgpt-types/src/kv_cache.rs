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
    /// Dequantize a value and add a per-element `bias` row:
    /// `out[i] = dequant(v)[i] + bias[i]` (zero-alloc hot path).
    ///
    /// The contract is **bit-identity with the default**: dequantize, then
    /// one f32 `+` per element (no FMA contraction of the add). A backend
    /// may override it to fold the add into its own dequant epilogue so the
    /// row is written once (KVarN does — Issue 883 P1, Bench 895
    /// addendum), but the result must equal this default bitwise.
    fn dequantize_value_add_into(
        &mut self,
        layer: usize,
        pos: usize,
        out: &mut [f32],
        bias: &[f32],
    ) {
        debug_assert_eq!(out.len(), bias.len());
        self.dequantize_value_into(layer, pos, out);
        for (o, &b) in out.iter_mut().zip(bias) {
            *o += b;
        }
    }
    /// Reset cache for a new sequence.
    fn reset(&mut self);
    /// Current write position.
    fn pos(&self) -> usize;
    /// Set the current write position.
    fn set_pos(&mut self, pos: usize);
}
