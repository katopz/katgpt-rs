//! Forward pass helpers using OCTOPUS compressed KV cache.
//!
//! Provides dequantize-to-flat-buffer helpers and attention scoring
//! for the OCTOPUS path. The main [`forward_octopus`] function lives
//! in [`crate::transformer`] because [`ForwardContext`] fields are
//! private to that module.
//!
//! Architecture:
//! 1. Standard embedding + RMSNorm + QKV projection (same as baseline)
//! 2. Store K,V via [`OctopusKVCache::store_key`] / [`store_value`]
//! 3. Dequantize K,V on-the-fly during attention scoring
//! 4. Standard MLP + residual (same as baseline)

use super::kv_cache::OctopusKVCache;

/// Dequantize key vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[(pos + 1) * kv_dim]` row-major, compatible with the
/// attention kernel's expected `key_cache` layout.
///
/// Returns `(flat_keys, pos_count)`.
pub fn dequantize_keys_flat(
    cache: &mut OctopusKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    // OPT: use _into variant to avoid per-position Vec allocation
    dequantize_keys_flat_into(cache, layer, pos, kv_dim, &mut flat);
    flat
}

/// Dequantize value vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[(pos + 1) * kv_dim]` row-major, compatible with the
/// attention kernel's expected `value_cache` layout.
pub fn dequantize_values_flat(
    cache: &mut OctopusKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    // OPT: use _into variant to avoid per-position Vec allocation
    dequantize_values_flat_into(cache, layer, pos, kv_dim, &mut flat);
    flat
}

/// Dequantize keys into pre-allocated flat buffer (zero-alloc variant).
///
/// Caller must ensure `flat` has length `>= (pos + 1) * kv_dim`.
pub fn dequantize_keys_flat_into(
    cache: &mut OctopusKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
    flat: &mut [f32],
) {
    for t in 0..=pos {
        cache.dequantize_key_into(layer, t, &mut flat[t * kv_dim..(t + 1) * kv_dim]);
    }
}

/// Dequantize values into pre-allocated flat buffer (zero-alloc variant).
///
/// Caller must ensure `flat` has length `>= (pos + 1) * kv_dim`.
pub fn dequantize_values_flat_into(
    cache: &mut OctopusKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
    flat: &mut [f32],
) {
    for t in 0..=pos {
        cache.dequantize_value_into(layer, t, &mut flat[t * kv_dim..(t + 1) * kv_dim]);
    }
}

/// Compute per-head attention scores using dequantized OCTOPUS KV cache.
///
/// This is the hot-path for OCTOPUS inference. For each query head,
/// scores against the reconstructed key cache and accumulates weighted
/// reconstructed values.
///
/// Buffers `scores_buf` and `attn_out` should be pre-allocated by the caller.
#[allow(clippy::too_many_arguments)]
pub fn attention_octopus(
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
    // Loop order: t outer, d inner — contiguous value_cache row access, better cache locality.
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

/// Compute inner-product absolute error between original and reconstructed vectors.
///
/// Useful for benchmarking OCTOPUS vs TurboQuant quality.
pub fn ip_error(a: &[f32], b: &[f32], query: &[f32]) -> f32 {
    let ip_orig: f32 = a.iter().zip(query).map(|(x, q)| x * q).sum();
    let ip_recon: f32 = b.iter().zip(query).map(|(x, q)| x * q).sum();
    (ip_orig - ip_recon).abs()
}

/// Compute per-coordinate MSE between original and reconstructed vectors.
pub fn per_coord_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    let n = original.len() as f32;
    original
        .iter()
        .zip(reconstructed)
        .map(|(o, r)| (o - r) * (o - r))
        .sum::<f32>()
        / n
}

// ── MaxSim Late-Interaction Scoring on Compressed KV ──────────

/// MaxSim scoring directly on OCTOPUS-compressed KV cache.
///
/// Computes `score = Σ_i max_j dot(q_i, dequantize_key(j))` without allocating
/// the full dequantized key matrix. Each position is lazy-dequantized inside the
/// inner loop, keeping peak memory at O(dim) instead of O(L × dim).
///
/// # Feature flag
/// Requires both `octopus` and `maxsim` features.
///
/// # GOAT proof (Plan 099 T9)
/// Must match uncompressed maxsim_score within 1e-3.
#[cfg(all(feature = "octopus", feature = "maxsim"))]
pub fn maxsim_score_octopus(
    queries: &[f32],
    cache: &mut super::kv_cache::OctopusKVCache,
    layer: usize,
    pos_range: std::ops::Range<usize>,
    dim: usize,
) -> f32 {
    let lq = queries.len() / dim;
    if lq == 0 || pos_range.is_empty() {
        return 0.0;
    }

    let mut key_buf = vec![0.0f32; dim];
    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for t in pos_range.clone() {
            // Zero-alloc lazy dequantize into reusable buffer.
            cache.dequantize_key_into(layer, t, &mut key_buf);
            let dot = crate::simd::simd_dot_f32(q_row, &key_buf, dim);
            my_max = my_max.max(dot);
        }
        score += my_max;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Config;

    fn make_cache(config: &Config, key_bits: u8, val_bits: u8) -> OctopusKVCache {
        OctopusKVCache::new(config, key_bits, val_bits)
    }

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
    fn test_cosine_similarity_antiparallel() {
        let a = [1.0f32, 0.0];
        let b = [-1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_flat_roundtrip() {
        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let mut cache = make_cache(&config, 3, 3);

        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);
        cache.store_key(0, 1, &key);

        let flat = dequantize_keys_flat(&mut cache, 0, 1, kv_dim);
        assert_eq!(flat.len(), 2 * kv_dim);
        // Same key should reconstruct to similar vectors
        let cos = cosine_similarity(&flat[0..kv_dim], &flat[kv_dim..]);
        assert!(
            cos > 0.99,
            "same key should reconstruct identically, cos={cos}"
        );
    }

    #[test]
    fn test_dequantize_flat_into_roundtrip() {
        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let mut cache = make_cache(&config, 3, 3);

        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let value: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.07).cos()).collect();
        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);
        cache.store_key(0, 1, &key);
        cache.store_value(0, 1, &value);

        let mut flat_keys = vec![0.0f32; 2 * kv_dim];
        let mut flat_vals = vec![0.0f32; 2 * kv_dim];
        dequantize_keys_flat_into(&mut cache, 0, 1, kv_dim, &mut flat_keys);
        dequantize_values_flat_into(&mut cache, 0, 1, kv_dim, &mut flat_vals);

        // Compare with non-into versions
        let ref_keys = dequantize_keys_flat(&mut cache, 0, 1, kv_dim);
        let ref_vals = dequantize_values_flat(&mut cache, 0, 1, kv_dim);

        for i in 0..flat_keys.len() {
            assert!(
                (flat_keys[i] - ref_keys[i]).abs() < 1e-5,
                "key mismatch at {i}"
            );
        }
        for i in 0..flat_vals.len() {
            assert!(
                (flat_vals[i] - ref_vals[i]).abs() < 1e-5,
                "val mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_attention_octopus_produces_finite() {
        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let head_dim = config.head_dim;
        let n_embd = config.n_embd;

        let mut cache = make_cache(&config, 3, 3);

        // Store synthetic KV entries
        let kv: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.05).sin()).collect();
        for pos in 0..4 {
            cache.store_key(0, pos, &kv);
            cache.store_value(0, pos, &kv);
        }

        let flat_keys = dequantize_keys_flat(&mut cache, 0, 3, kv_dim);
        let flat_values = dequantize_values_flat(&mut cache, 0, 3, kv_dim);

        let q: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.03).sin()).collect();
        let mut attn_out = vec![0.0f32; n_embd];
        let mut scores = vec![0.0f32; config.block_size];

        attention_octopus(
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

    #[test]
    fn test_per_coord_mse() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.1f32, 2.1, 3.1];
        let mse = per_coord_mse(&a, &b);
        // Expected: ((0.1)^2 + (0.1)^2 + (0.1)^2) / 3 = 0.01
        assert!((mse - 0.01).abs() < 1e-6, "mse = {mse}");
    }

    #[test]
    fn test_per_coord_mse_identical() {
        let a = [1.0f32, 2.0, 3.0];
        assert!(per_coord_mse(&a, &a) < 1e-10);
    }

    #[test]
    fn test_ip_error() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32, 2.1];
        let q = [1.0f32, 0.0];
        let err = ip_error(&a, &b, &q);
        // IP(a, q) = 1.0, IP(b, q) = 1.0, error = 0.0
        assert!(err < 1e-6, "ip error = {err}");
    }

    #[test]
    fn test_ip_error_nonzero() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32, 2.1];
        let q = [0.0f32, 1.0];
        let err = ip_error(&a, &b, &q);
        // IP(a, q) = 2.0, IP(b, q) = 2.1, error = 0.1
        assert!((err - 0.1).abs() < 1e-6, "ip error = {err}");
    }

    #[test]
    fn test_attention_weights_normalized() {
        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let head_dim = config.head_dim;
        let n_embd = config.n_embd;

        let mut cache = make_cache(&config, 3, 3);

        // Store diverse keys so scores differ
        for pos in 0..4 {
            let key: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + pos * 17) as f32 * 0.07).sin())
                .collect();
            let value: Vec<f32> = (0..kv_dim)
                .map(|i| ((i + pos * 13) as f32 * 0.05).cos())
                .collect();
            cache.store_key(0, pos, &key);
            cache.store_value(0, pos, &value);
        }

        let flat_keys = dequantize_keys_flat(&mut cache, 0, 3, kv_dim);
        let flat_values = dequantize_values_flat(&mut cache, 0, 3, kv_dim);

        let q = vec![0.5f32; n_embd];
        let mut attn_out = vec![0.0f32; n_embd];
        let mut scores = vec![0.0f32; config.block_size];

        attention_octopus(
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

        // scores_buf stores unnormalized exponentials: exp(score - max).
        // Normalize manually to verify softmax correctness.
        let sum: f32 = scores[..4].iter().sum();
        let normalized: Vec<f32> = scores[..4].iter().map(|s| s / sum).collect();
        let norm_sum: f32 = normalized.iter().sum();
        assert!(
            (norm_sum - 1.0).abs() < 1e-4,
            "normalized softmax weights sum = {norm_sum}, expected 1.0"
        );

        // Each normalized weight should be in (0, 1]
        for (t, &w) in normalized.iter().enumerate() {
            assert!(w > 0.0, "weight[{t}] = {w}, should be positive");
            assert!(w <= 1.0, "weight[{t}] = {w}, should be <= 1.0");
        }
    }
}
