//! Q-K=V projection sharing for inference-time KV cache halving (Plan 185).

use katgpt_core::types::{AttentionProjection, CacheLayout};

/// Merge K and V projection weights post-hoc.
/// W_kv = (W_k + W_v) / 2
///
/// Returns the merged weight matrix. Call this once at model load time
/// when `AttentionProjection::SharedKV` is configured.
pub fn merge_kv_weights(w_k: &[f32], w_v: &[f32]) -> Vec<f32> {
    assert_eq!(
        w_k.len(),
        w_v.len(),
        "K and V weight matrices must have same size"
    );
    w_k.iter()
        .zip(w_v.iter())
        .map(|(&k, &v)| (k + v) / 2.0)
        .collect()
}

/// Merge bias vectors for K=V sharing.
pub fn merge_kv_bias(b_k: &[f32], b_v: &[f32]) -> Vec<f32> {
    assert_eq!(
        b_k.len(),
        b_v.len(),
        "K and V bias vectors must have same size"
    );
    b_k.iter()
        .zip(b_v.iter())
        .map(|(&k, &v)| (k + v) / 2.0)
        .collect()
}

/// Compute the cache layout from attention projection config.
pub const fn cache_layout(projection: AttentionProjection) -> CacheLayout {
    match projection {
        AttentionProjection::Full => CacheLayout::KV,
        AttentionProjection::SharedKV => CacheLayout::K,
    }
}

/// Compute cache slots per layer based on layout.
/// K-only layout gets 2× the slots (V is derived from K).
#[inline]
pub fn cache_slots_per_layer(layout: CacheLayout, base_slots: usize) -> usize {
    match layout {
        CacheLayout::KV => base_slots,
        CacheLayout::K => base_slots * 2,
    }
}

/// Compute memory per token for given layout.
#[inline]
pub fn memory_per_token(layout: CacheLayout, head_dim: usize) -> usize {
    match layout {
        CacheLayout::KV => 2 * head_dim * std::mem::size_of::<f32>(),
        CacheLayout::K => head_dim * std::mem::size_of::<f32>(),
    }
}

/// Apply SharedKV projection: returns K to use as both K and V.
/// When `SharedKV`, V output is simply K output (no V projection needed).
/// Returns `(k, v)` tuple where `v` is either the original V or a clone of K.
#[inline]
pub fn shared_kv_project<'a>(
    k: &'a [f32],
    v: &'a [f32],
    projection: AttentionProjection,
) -> (&'a [f32], &'a [f32]) {
    match projection {
        AttentionProjection::Full => (k, v),
        AttentionProjection::SharedKV => (k, k),
    }
}

/// Compute attention with SharedKV: skip V projection, use K as V.
/// Returns attention output scaled by whether V projection is needed.
#[inline]
pub fn attention_flops_factor(projection: AttentionProjection) -> f32 {
    match projection {
        AttentionProjection::Full => 1.0,
        AttentionProjection::SharedKV => 2.0 / 3.0,
    }
}
