//! Forward pass using TurboQuant compressed KV cache.
//!
//! The main [`forward_turboquant`] function lives in [`crate::transformer`] because
//! [`ForwardContext`] fields are private to that module. This file provides
//! helper functions used by the TurboQuant attention path.
//!
//! Architecture:
//! 1. Standard embedding + RMSNorm + QKV projection (same as baseline)
//! 2. Store K,V via [`TurboQuantKVCache::store_key`] / [`store_value`]
//! 3. Dequantize K,V on-the-fly during attention scoring
//! 4. Standard MLP + residual (same as baseline)

use super::kv_cache::TurboQuantKVCache;

#[cfg(test)]
use crate::types;

/// Dequantize key vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[block_size * kv_dim]` row-major, compatible with the
/// [`attention_head`] kernel's expected `key_cache` layout.
///
/// Returns `(flat_keys, pos_count)` where `flat_keys[pos * kv_dim..]`
/// holds the reconstructed key for that position.
///
/// **Note:** Allocates a new Vec per call. For hot-path code, prefer
/// [`dequantize_keys_flat_into`] which reuses a pre-allocated buffer.
pub fn dequantize_keys_flat(
    cache: &mut TurboQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    dequantize_keys_flat_into(cache, layer, pos, kv_dim, &mut flat);
    flat
}

/// Dequantize value vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[block_size * kv_dim]` row-major, compatible with the
/// [`attention_head`] kernel's expected `value_cache` layout.
///
/// **Note:** Allocates a new Vec per call. For hot-path code, prefer
/// [`dequantize_values_flat_into`] which reuses a pre-allocated buffer.
pub fn dequantize_values_flat(
    cache: &mut TurboQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    dequantize_values_flat_into(cache, layer, pos, kv_dim, &mut flat);
    flat
}

/// Zero-alloc variant of [`dequantize_keys_flat`].
///
/// Writes into `buf`, which must have capacity `>= (pos + 1) * kv_dim`.
/// Uses `dequantize_key_into` per position to avoid per-position Vec allocation.
pub fn dequantize_keys_flat_into(
    cache: &mut TurboQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
    buf: &mut [f32],
) {
    let total = (pos + 1) * kv_dim;
    debug_assert!(
        buf.len() >= total,
        "buffer too small: {} < {total}",
        buf.len()
    );
    for t in 0..=pos {
        cache.dequantize_key_into(layer, t, &mut buf[t * kv_dim..(t + 1) * kv_dim]);
    }
}

/// Zero-alloc variant of [`dequantize_values_flat`].
///
/// Writes into `buf`, which must have capacity `>= (pos + 1) * kv_dim`.
/// Uses `dequantize_value_into` per position to avoid per-position Vec allocation.
pub fn dequantize_values_flat_into(
    cache: &mut TurboQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
    buf: &mut [f32],
) {
    let total = (pos + 1) * kv_dim;
    debug_assert!(
        buf.len() >= total,
        "buffer too small: {} < {total}",
        buf.len()
    );
    for t in 0..=pos {
        cache.dequantize_value_into(layer, t, &mut buf[t * kv_dim..(t + 1) * kv_dim]);
    }
}

/// Compute per-head attention scores using dequantized KV cache.
///
/// This is the hot-path for TurboQuant inference. For each query head,
/// it scores against the reconstructed key cache and accumulates weighted
/// reconstructed values.
///
/// Buffers `scores_buf` and `attn_out` should be pre-allocated by the caller.
#[allow(clippy::too_many_arguments)]
pub fn attention_turboquant(
    q: &[f32],
    flat_keys: &[f32],
    flat_values: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    head_dim: usize,
    pos: usize,
    scale: f32,
) {
    let t_n = pos + 1;

    // Pass 1: Q·K scores + find max
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), head_dim);
            let k_slice = std::slice::from_raw_parts(flat_keys.as_ptr().add(k_off), head_dim);
            crate::simd::simd_dot_f32(q_slice, k_slice, head_dim)
        };
        let score = dot * scale;
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        max_score = max_score.max(score);
    }

    // Pass 2: exp(scores - max) + sum (SIMD batch)
    let scores_slice = unsafe { std::slice::from_raw_parts_mut(scores_buf.as_mut_ptr(), t_n) };
    crate::simd::simd_add_scalar_inplace(scores_slice, -max_score);
    crate::simd::simd_exp_inplace(scores_slice);
    let sum = crate::simd::simd_sum_f32(scores_slice);

    // Pass 3: normalize + weighted value accumulation
    // Loop order: t outer, d inner — contiguous value_cache row access, better cache locality.
    // Previous d-outer/t-inner order touched a different cache line per t for each d.
    let inv_sum = 1.0 / sum;
    attn_out[q_head_offset..q_head_offset + head_dim].fill(0.0);
    for t in 0..t_n {
        let weight = unsafe { *scores_buf.get_unchecked(t) * inv_sum };
        let v_row = unsafe {
            std::slice::from_raw_parts(
                flat_values.as_ptr().add(t * kv_dim + kv_group_offset),
                head_dim,
            )
        };
        crate::simd::simd_fused_scale_acc(
            &mut attn_out[q_head_offset..q_head_offset + head_dim],
            v_row,
            weight,
            head_dim,
        );
    }
}

/// Compression-aware cosine similarity metric for quality validation.
///
/// Measures reconstruction fidelity of the quantize→dequantize round-trip.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let dot = crate::simd::simd_dot_f32(a, b, len);
    let na = crate::simd::simd_dot_f32(a, a, len).sqrt();
    let nb = crate::simd::simd_dot_f32(b, b, len).sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    dot / (na * nb)
}

// ── MaxSim Late-Interaction Scoring on Compressed KV (Research 45, Plan 080) ──

/// MaxSim scoring directly on TurboQuant-compressed KV cache.
///
/// Computes `score = Σ_i max_j dot(q_i, dequantize_key(j))` without allocating
/// the full dequantized key matrix. Each position is lazy-dequantized inside the
/// inner loop, keeping peak memory at O(dim) instead of O(Ld × dim).
///
/// This is the TurboQuant counterpart to [`maxsim_score`](crate::simd::maxsim_score).
/// The principle is identical (running max over doc tokens), but here each doc
/// token's vector comes from the compressed KV cache rather than a flat buffer.
///
/// # Relationship to SpectralQuant (Research 39)
///
/// SpectralQuant's `spectralquant_attention.wgsl` already implements fused
/// dequantize + scoring with softmax-sum reduction. Adding `ScoreReduction::MaxSim`
/// to that kernel is the GPU equivalent of this function. This CPU path covers:
/// - TurboQuant-only scenarios (no SpectralQuant calibration)
/// - Testing / correctness validation for the GPU path
/// - Small batch sizes where GPU dispatch overhead dominates
///
/// # Feature flag
/// Requires both `turboquant` and `maxsim` features.
///
/// # GOAT proof (Plan 080 T9)
/// Must match uncompressed `maxsim_score` within 1e-3.
/// Latency overhead vs `attention_turboquant` softmax-sum mode must be ≤5%.
#[cfg(all(feature = "turboquant", feature = "maxsim"))]
pub fn maxsim_score_turboquant(
    queries: &[f32],
    cache: &mut super::kv_cache::TurboQuantKVCache,
    layer: usize,
    pos_range: std::ops::Range<usize>,
    dim: usize,
) -> f32 {
    let lq = queries.len() / dim;
    if lq == 0 || pos_range.is_empty() {
        return 0.0;
    }

    // Pre-allocate a single scratch buffer for lazy dequantization.
    // Avoids allocating a Vec<f32> per position in the inner loop.
    let mut key_buf = vec![0.0f32; cache.kv_dim()];

    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for t in pos_range.clone() {
            // Lazy dequantize into pre-allocated buffer: O(dim) peak memory,
            // zero heap allocation per position. Matches maxsim.metal streaming pattern.
            cache.dequantize_key_into(layer, t, &mut key_buf);
            let dot = crate::simd::simd_dot_f32(q_row, &key_buf[..dim], dim);
            my_max = my_max.max(dot);
        }
        score += my_max;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::TransformerWeights;
    use crate::types::{Config, Rng};

    #[test]
    fn test_cosine_similarity_identical() {
        let v = [1.0f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_flat_roundtrip() {
        let config = Config::micro();
        let kv_dim = types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let _weights = TransformerWeights::new(&config, &mut rng);
        let mut cache = super::super::kv_cache::TurboQuantKVCache::new(&config, 4, 4);

        // Create a synthetic key from weight projection
        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);
        cache.store_key(0, 1, &key);

        let flat = dequantize_keys_flat(&mut cache, 0, 1, kv_dim);
        assert_eq!(flat.len(), 2 * kv_dim);
        // Positions should reconstruct to similar vectors
        let cos = cosine_similarity(&flat[0..kv_dim], &flat[kv_dim..]);
        assert!(
            cos > 0.99,
            "Same key should reconstruct identically, cos={cos}"
        );
    }

    #[test]
    fn test_attention_turboquant_produces_finite() {
        let config = Config::micro();
        let kv_dim = types::kv_dim(&config);
        let head_dim = config.head_dim;
        let _n_head = config.n_head;
        let n_embd = config.n_embd;

        let mut rng = Rng::new(99);
        let _weights = TransformerWeights::new(&config, &mut rng);
        let mut tq_cache = super::super::kv_cache::TurboQuantKVCache::new(&config, 4, 4);

        // Store some synthetic KV entries
        let kv: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.05).sin()).collect();
        for pos in 0..4 {
            tq_cache.store_key(0, pos, &kv);
            tq_cache.store_value(0, pos, &kv);
        }

        let flat_keys = dequantize_keys_flat(&mut tq_cache, 0, 3, kv_dim);
        let flat_values = dequantize_values_flat(&mut tq_cache, 0, 3, kv_dim);

        let q: Vec<f32> = (0..n_embd).map(|_| rng.normal()).collect();
        let mut attn_out = vec![0.0f32; n_embd];
        let mut scores = vec![0.0f32; config.block_size];

        attention_turboquant(
            &q,
            &flat_keys,
            &flat_values,
            &mut attn_out,
            &mut scores,
            0,
            0,
            kv_dim,
            head_dim,
            3,
            1.0 / (head_dim as f32).sqrt(),
        );

        // All outputs should be finite
        for (i, &v) in attn_out.iter().enumerate() {
            assert!(v.is_finite(), "attn_out[{i}] = {v} is not finite");
        }
    }
}
