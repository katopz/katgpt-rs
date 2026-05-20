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
pub fn dequantize_keys_flat(
    cache: &TurboQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    for t in 0..=pos {
        let recon = cache.dequantize_key(layer, t);
        flat[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&recon);
    }
    flat
}

/// Dequantize value vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[block_size * kv_dim]` row-major, compatible with the
/// [`attention_head`] kernel's expected `value_cache` layout.
pub fn dequantize_values_flat(
    cache: &TurboQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    for t in 0..=pos {
        let recon = cache.dequantize_value(layer, t);
        flat[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&recon);
    }
    flat
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
        if score > max_score {
            max_score = score;
        }
    }

    // Pass 2: exp(scores - max) + sum
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation
    let inv_sum = 1.0 / sum;
    for d in 0..head_dim {
        let mut val = 0.0f32;
        for t in 0..t_n {
            unsafe {
                val += *scores_buf.get_unchecked(t)
                    * inv_sum
                    * *flat_values.get_unchecked(t * kv_dim + kv_group_offset + d);
            }
        }
        unsafe {
            *attn_out.get_unchecked_mut(q_head_offset + d) = val;
        }
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
#[allow(dead_code)] // Stub — wired in Plan 080 T9
pub fn maxsim_score_turboquant(
    queries: &[f32],
    cache: &super::kv_cache::TurboQuantKVCache,
    layer: usize,
    pos_range: std::ops::Range<usize>,
    dim: usize,
) -> f32 {
    let lq = queries.len() / dim;
    if lq == 0 || pos_range.is_empty() {
        return 0.0;
    }

    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for t in pos_range.clone() {
            // Lazy dequantize: only one key vector in memory at a time.
            // This is the "streaming over doc tokens" pattern from maxsim.metal —
            // same O(dim) peak memory, but on CPU with simd_dot_f32 instead of
            // Metal simdgroup_matrix.
            let key = cache.dequantize_key(layer, t);
            let dot = crate::simd::simd_dot_f32(q_row, &key, dim);
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

        let flat = dequantize_keys_flat(&cache, 0, 1, kv_dim);
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

        let flat_keys = dequantize_keys_flat(&tq_cache, 0, 3, kv_dim);
        let flat_values = dequantize_values_flat(&tq_cache, 0, 3, kv_dim);

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
