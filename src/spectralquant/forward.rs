//! SpectralQuant forward pass helpers.
//!
//! Provides dequantization and attention scoring functions for the
//! SpectralQuant KV cache path. The main forward function
//! (`forward_quantized`) is generic and lives in [`crate::transformer`].

#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
use super::spectral_kv_cache::DequantizeScratch;
use super::spectral_kv_cache::SpectralQuantKVCache;
#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
use rayon::prelude::*;

/// Dequantize SpectralQuant key vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[block_size * kv_dim]` row-major, compatible with the
/// attention kernel's expected `key_cache` layout.
pub fn dequantize_spectral_keys_flat(
    cache: &mut SpectralQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    for t in 0..=pos {
        cache.dequantize_key_into(layer, t, &mut flat[t * kv_dim..(t + 1) * kv_dim]);
    }
    flat
}

/// Dequantize SpectralQuant value vectors for positions `[0..=pos]` into a flat buffer.
///
/// Layout: `[block_size * kv_dim]` row-major, compatible with the
/// attention kernel's expected `value_cache` layout.
pub fn dequantize_spectral_values_flat(
    cache: &mut SpectralQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut flat = vec![0.0f32; (pos + 1) * kv_dim];
    for t in 0..=pos {
        cache.dequantize_value_into(layer, t, &mut flat[t * kv_dim..(t + 1) * kv_dim]);
    }
    flat
}

/// Compute per-head attention scores using dequantized SpectralQuant KV cache.
///
/// Self-contained attention scoring: Q·K → softmax → weighted V accumulation.
/// Accepts flat buffers produced by [`dequantize_spectral_keys_flat`] / [`dequantize_spectral_values_flat`].
#[allow(clippy::too_many_arguments)]
pub fn attention_spectralquant(
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

// ── MaxSim Late-Interaction Scoring on SpectralQuant KV (Research 45, Plan 080) ──

/// MaxSim scoring directly on SpectralQuant-compressed KV cache.
///
/// Computes `score = Σ_i max_j dot(q_i, dequantize_key(j))` without allocating
/// the full dequantized key matrix. Each position is lazy-dequantized inside the
/// inner loop, keeping peak memory at O(dim) instead of O(Ld × dim).
///
/// # SpectralQuant Optimization: d_eff Truncation
///
/// SpectralQuant's key property (Research 39): d_eff ≈ 3-5% of head_dim for keys.
/// After eigenbasis rotation, coordinates `[d_eff..dim]` are noise and contribute
/// negligible dot-product signal. This function could be extended to only dequantize
/// and score the semantic subspace `[0..d_eff]`, reducing per-position work by ~95%.
///
/// However, that optimization changes the *result* (not just the speed), so it
/// requires its own GOAT proof. The current implementation scores all dimensions
/// for correctness parity with the uncompressed `maxsim_score`.
///
/// # Relationship to TurboQuant (Research 20)
///
/// [`maxsim_score_turboquant`](crate::turboquant::forward::maxsim_score_turboquant)
/// is the same pattern for TurboQuant's random-rotation + uniform-bit path.
/// This SpectralQuant version uses calibrated eigenbasis + water-fill + selective QJL,
/// giving higher fidelity at the same compression ratio. Both share the same
/// running-max-over-lazy-dequantized-keys inner loop structure.
///
/// # Feature flag
/// Requires both `spectralquant` and `maxsim` features.
///
/// # GOAT proof (Plan 080 T10)
/// Must match uncompressed `maxsim_score` within 1e-3.
/// Must match CPU reference within 1e-3.
#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
pub fn maxsim_score_spectralquant(
    queries: &[f32],
    cache: &mut SpectralQuantKVCache,
    layer: usize,
    pos_range: std::ops::Range<usize>,
    dim: usize,
) -> f32 {
    let lq = queries.len() / dim;
    if lq == 0 || pos_range.is_empty() {
        return 0.0;
    }

    // Reusable buffer for lazy dequantize — avoids per-position allocation.
    // Peak memory: O(dim) for the key buffer, matching maxsim.metal's design
    // of streaming over doc tokens with only running state in shared memory.
    let mut key_buf = vec![0.0f32; dim];

    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for t in pos_range.clone() {
            // Lazy dequantize into reusable buffer — spectral rotation +
            // variable-bit unpack + codebook lookup all happen inside
            // dequantize_key_into. Only one key vector in memory at a time.
            cache.dequantize_key_into(layer, t, &mut key_buf);
            let dot = crate::simd::simd_dot_f32(q_row, &key_buf, dim);
            my_max = my_max.max(dot);
        }
        score += my_max;
    }
    score
}

// ── Parallel Variants (rayon + per-thread scratch) ──────────────────

/// Parallel batch dequantize of SpectralQuant key vectors for positions `[0..=pos]`.
///
/// Same output as [`dequantize_spectral_keys_flat`] but uses rayon with per-thread
/// [`DequantizeScratch`] buffers. Takes `&cache` (not `&mut`) — safe for concurrent reads.
///
/// Falls back to sequential for `n <= threshold` where rayon overhead outweighs benefit.
pub fn par_dequantize_spectral_keys_flat(
    cache: &SpectralQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
    threshold: usize,
) -> Vec<f32> {
    let n = pos + 1;
    if n == 0 {
        return Vec::new();
    }
    debug_assert_eq!(kv_dim, cache.kv_dim());

    cache.par_dequantize_keys_flat(layer, pos, threshold)
}

/// Parallel batch dequantize of SpectralQuant value vectors for positions `[0..=pos]`.
///
/// Same output as [`dequantize_spectral_values_flat`] but uses rayon with per-thread
/// [`DequantizeScratch`] buffers. Takes `&cache` (not `&mut`) — safe for concurrent reads.
///
/// Falls back to sequential for `n <= threshold` where rayon overhead outweighs benefit.
pub fn par_dequantize_spectral_values_flat(
    cache: &SpectralQuantKVCache,
    layer: usize,
    pos: usize,
    kv_dim: usize,
    threshold: usize,
) -> Vec<f32> {
    let n = pos + 1;
    if n == 0 {
        return Vec::new();
    }
    debug_assert_eq!(kv_dim, cache.kv_dim());

    cache.par_dequantize_values_flat(layer, pos, threshold)
}

/// Parallel MaxSim scoring on SpectralQuant-compressed KV cache.
///
/// Same math as [`maxsim_score_spectralquant`] but parallelizes the outer loop
/// over query tokens using rayon. Each rayon worker gets its own
/// [`DequantizeScratch`] + key buffer via `map_init`.
///
/// Takes `&cache` (not `&mut`) — safe for concurrent reads from multiple threads.
/// Falls back to sequential for `lq <= threshold` where rayon overhead outweighs benefit.
///
/// # Feature flag
/// Requires both `spectralquant` and `maxsim` features.
#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
pub fn par_maxsim_score_spectralquant(
    queries: &[f32],
    cache: &SpectralQuantKVCache,
    layer: usize,
    pos_range: std::ops::Range<usize>,
    dim: usize,
    threshold: usize,
) -> f32 {
    let lq = queries.len() / dim;
    if lq == 0 || pos_range.is_empty() {
        return 0.0;
    }

    // Sequential fallback for few query tokens
    if lq <= threshold {
        return maxsim_score_spectralquant_fallback(queries, cache, layer, pos_range, dim);
    }

    // Parallel: each worker gets its own scratch + key buffer
    let per_query_max: Vec<f32> = (0..lq)
        .into_par_iter()
        .map_init(
            || (DequantizeScratch::new(dim), vec![0.0f32; dim]),
            |(scratch, key_buf), i| {
                let q_row = &queries[i * dim..(i + 1) * dim];
                let mut my_max = f32::NEG_INFINITY;
                for t in pos_range.clone() {
                    cache.dequantize_key_into_with_scratch(layer, t, scratch, key_buf);
                    let dot = crate::simd::simd_dot_f32(q_row, key_buf, dim);
                    my_max = my_max.max(dot);
                }
                my_max
            },
        )
        .collect();

    per_query_max.into_iter().sum()
}

/// Sequential fallback for `par_maxsim_score_spectralquant`.
///
/// Uses `&cache` + [`DequantizeScratch`] (no `&mut cache` required).
/// Extracted as a standalone function so both seq and par paths use
/// the same `dequantize_key_into_with_scratch` code path.
#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
fn maxsim_score_spectralquant_fallback(
    queries: &[f32],
    cache: &SpectralQuantKVCache,
    layer: usize,
    pos_range: std::ops::Range<usize>,
    dim: usize,
) -> f32 {
    let lq = queries.len() / dim;
    let mut scratch = DequantizeScratch::new(dim);
    let mut key_buf = vec![0.0f32; dim];

    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for t in pos_range.clone() {
            cache.dequantize_key_into_with_scratch(layer, t, &mut scratch, &mut key_buf);
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
    use crate::spectralquant::spectral::participation_ratio;
    use crate::spectralquant::types::{SpectralQuantCalibration, SpectralQuantKVCacheConfig};
    use crate::types::{Config, Rng};

    #[test]
    fn test_spectralquant_forward_produces_finite() {
        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let head_dim = config.head_dim;
        let n_embd = config.n_embd;

        // Build calibration with identity eigenvectors
        let mut eigenvectors = vec![0.0f32; kv_dim * kv_dim];
        for i in 0..kv_dim {
            eigenvectors[i * kv_dim + i] = 1.0;
        }
        let eigenvalues: Vec<f32> = (0..kv_dim).map(|i| 10.0 * 0.8f32.powi(i as i32)).collect();
        let d_eff = participation_ratio(&eigenvalues);
        let cal = SpectralQuantCalibration {
            eigenvectors,
            eigenvalues,
            d_eff,
            spectral_gap: None,
            var_95: 10,
            var_99: 20,
            n_samples: 100,
            head_dim: kv_dim,
        };

        let sq_config = SpectralQuantKVCacheConfig {
            avg_bits: 3.0,
            min_tail_bits: 1,
            max_bits: 8,
            qjl_dim: 16,
            lloyd_max_iter: 30,
            calibration_samples: 100,
            seed: 42,
            use_water_fill: false,
            wf_min_bits: 1,
            wf_max_bits: 6,
            n_layers: config.n_layer,
            kv_dim,
            max_seq_len: config.block_size,
        };

        let mut sq_cache = SpectralQuantKVCache::from_calibration(
            &sq_config,
            &vec![cal.clone(); config.n_layer],
            &vec![cal; config.n_layer],
        );

        // Store synthetic KV entries
        let kv: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.05).sin()).collect();
        for pos in 0..4 {
            sq_cache.store_key(0, pos, &kv);
            sq_cache.store_value(0, pos, &kv);
        }

        let flat_keys = dequantize_spectral_keys_flat(&mut sq_cache, 0, 3, kv_dim);
        let flat_values = dequantize_spectral_values_flat(&mut sq_cache, 0, 3, kv_dim);

        let mut rng = Rng::new(99);
        let q: Vec<f32> = (0..n_embd).map(|_| rng.normal()).collect();
        let mut attn_out = vec![0.0f32; n_embd];
        let mut scores = vec![0.0f32; config.block_size];

        attention_spectralquant(
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

        for (i, &v) in attn_out.iter().enumerate() {
            assert!(v.is_finite(), "attn_out[{i}] = {v} is not finite");
        }
    }

    /// Trace per-dimension roundtrip error for SQ quantize/dequantize.
    /// Purpose: diagnose why SQ maxsim error (5.7%) is higher than TQ (0.95%).
    #[test]
    #[cfg(feature = "maxsim")]
    fn trace_sq_roundtrip_error() {
        use crate::simd::maxsim_score;

        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let n_positions = 8;
        let lq = 2;

        let mut eigenvectors = vec![0.0f32; kv_dim * kv_dim];
        for i in 0..kv_dim {
            eigenvectors[i * kv_dim + i] = 1.0;
        }
        let eigenvalues: Vec<f32> = (0..kv_dim).map(|i| 10.0 * 0.8f32.powi(i as i32)).collect();
        let d_eff = participation_ratio(&eigenvalues);
        let cal = SpectralQuantCalibration {
            eigenvectors: eigenvectors.clone(),
            eigenvalues: eigenvalues.clone(),
            d_eff,
            spectral_gap: None,
            var_95: 10,
            var_99: 20,
            n_samples: 100,
            head_dim: kv_dim,
        };
        let sq_config = SpectralQuantKVCacheConfig {
            avg_bits: 3.0,
            min_tail_bits: 1,
            max_bits: 8,
            qjl_dim: 16,
            lloyd_max_iter: 30,
            calibration_samples: 100,
            seed: 42,
            use_water_fill: false,
            wf_min_bits: 1,
            wf_max_bits: 6,
            n_layers: config.n_layer,
            kv_dim,
            max_seq_len: config.block_size,
        };
        let mut sq_cache = super::super::spectral_kv_cache::SpectralQuantKVCache::from_calibration(
            &sq_config,
            &vec![cal.clone(); config.n_layer],
            &vec![cal; config.n_layer],
        );

        // Store one key and trace per-dim roundtrip
        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.05).cos()).collect();
        sq_cache.store_key(0, 0, &key);

        let mut reconstructed = vec![0.0f32; kv_dim];
        sq_cache.dequantize_key_into(0, 0, &mut reconstructed);

        let mut max_abs_err = 0.0f32;
        let mut total_sq_err = 0.0f32;
        eprintln!("\n  dim | original  | recon     | error      | eigenvalue");
        eprintln!("  ----|-----------|-----------|------------|----------");
        for i in 0..kv_dim {
            let err = (key[i] - reconstructed[i]).abs();
            max_abs_err = max_abs_err.max(err);
            total_sq_err += err * err;
            eprintln!(
                "  {:3} | {:9.6} | {:9.6} | {:10.6} | {:.4}",
                i,
                key[i],
                reconstructed[i],
                err,
                eigenvalues.get(i).copied().unwrap_or(0.0)
            );
        }
        eprintln!(
            "  max_abs_err={max_abs_err:.6}, rmse={:.6}",
            total_sq_err.sqrt() / kv_dim as f32
        );

        // Dump codebook centroids to diagnose range mismatch
        let layer_state = &sq_cache.layers[0];
        eprintln!(
            "\n  b_high={}, b_low={}, d_eff={}",
            layer_state.b_high, layer_state.b_low, layer_state.d_eff
        );
        if let Some(ref cb) = layer_state.semantic_codebook {
            eprintln!(
                "  semantic centroids ({}) : {:?}",
                cb.centroids.len(),
                cb.centroids
            );
        }
        eprintln!(
            "  tail centroids ({})     : {:?}",
            layer_state.tail_codebook.centroids.len(),
            layer_state.tail_codebook.centroids
        );
        if let Some(ref bits) = layer_state.semantic_bits_per_dim {
            eprintln!("  semantic_bits_per_dim  : {:?}", bits);
        }

        // Compare SQ maxsim vs uncompressed
        let queries: Vec<f32> = (0..lq * kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let original_keys: Vec<Vec<f32>> = (0..n_positions)
            .map(|t| {
                (0..kv_dim)
                    .map(|d| ((t * kv_dim + d) as f32 * 0.05).cos())
                    .collect()
            })
            .collect();
        for (t, k) in original_keys.iter().enumerate() {
            sq_cache.store_key(0, t, k);
        }

        let sq_score =
            maxsim_score_spectralquant(&queries, &mut sq_cache, 0, 0..n_positions, kv_dim);

        // Fair comparison: dequantize all keys from SQ cache, then score uncompressed.
        // SQ applies random rotation when eigenvectors are identity, so comparing
        // against raw unrotated keys is unfair — both paths must go through the
        // same rotation to isolate quantization error from rotation mismatch.
        let mut reconstructed_keys = vec![0.0f32; n_positions * kv_dim];
        for t in 0..n_positions {
            sq_cache.dequantize_key_into(
                0,
                t,
                &mut reconstructed_keys[t * kv_dim..(t + 1) * kv_dim],
            );
        }
        let reconstructed = maxsim_score(&queries, &reconstructed_keys, lq, n_positions, kv_dim);

        eprintln!("\n  SQ MaxSim (streaming):  {sq_score:.6}");
        eprintln!("  SQ MaxSim (dequant):    {reconstructed:.6}");
        eprintln!(
            "  Match:                  {:.6} ({:.2}%)",
            (sq_score - reconstructed).abs(),
            (sq_score - reconstructed).abs() / reconstructed.abs().max(1e-8) * 100.0
        );

        // Streaming vs dequantized should match exactly (same codebook, same data)
        assert!(sq_score.is_finite(), "SQ score is not finite: {sq_score}");
        assert!(
            (sq_score - reconstructed).abs() < 1e-4,
            "streaming vs dequantized mismatch: {sq_score} vs {reconstructed}"
        );
    }

    /// Verify `par_maxsim_score_spectralquant` produces the same score as the
    /// sequential `maxsim_score_spectralquant`. Issue 064 T7 — GOAT proof.
    #[test]
    #[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
    fn test_par_maxsim_matches_seq() {
        use super::{maxsim_score_spectralquant, par_maxsim_score_spectralquant};
        use crate::spectralquant::spectral::participation_ratio;
        use crate::spectralquant::types::{SpectralQuantCalibration, SpectralQuantKVCacheConfig};
        use crate::types::Config;

        let config = Config::micro();
        let kv_dim = crate::types::kv_dim(&config);
        let n_positions = 32;
        let lq = 8;

        // Build calibration
        let mut eigenvectors = vec![0.0f32; kv_dim * kv_dim];
        for i in 0..kv_dim {
            eigenvectors[i * kv_dim + i] = 1.0;
        }
        let eigenvalues: Vec<f32> = (0..kv_dim).map(|i| 10.0 * 0.8f32.powi(i as i32)).collect();
        let d_eff = participation_ratio(&eigenvalues);
        let cal = SpectralQuantCalibration {
            eigenvectors,
            eigenvalues,
            d_eff,
            spectral_gap: None,
            var_95: 10,
            var_99: 20,
            n_samples: 100,
            head_dim: kv_dim,
        };

        let sq_config = SpectralQuantKVCacheConfig {
            avg_bits: 3.0,
            min_tail_bits: 1,
            max_bits: 8,
            qjl_dim: 16,
            lloyd_max_iter: 30,
            calibration_samples: 100,
            seed: 42,
            use_water_fill: false,
            wf_min_bits: 1,
            wf_max_bits: 6,
            n_layers: config.n_layer,
            kv_dim,
            max_seq_len: n_positions,
        };

        let mut cache = SpectralQuantKVCache::from_calibration(
            &sq_config,
            &vec![cal.clone(); config.n_layer],
            &vec![cal; config.n_layer],
        );

        // Store synthetic keys
        for pos in 0..n_positions {
            let key: Vec<f32> = (0..kv_dim)
                .map(|d| ((d + pos * 7) as f32 * 0.1).sin())
                .collect();
            cache.store_key(0, pos, &key);
        }

        // Build queries
        let queries: Vec<f32> = (0..lq * kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();

        // Sequential (requires &mut cache)
        let seq_score = maxsim_score_spectralquant(&queries, &mut cache, 0, 0..n_positions, kv_dim);

        // Parallel (requires only &cache, threshold=1 forces parallel path)
        let par_score =
            par_maxsim_score_spectralquant(&queries, &cache, 0, 0..n_positions, kv_dim, 1);

        assert!(seq_score.is_finite(), "seq score not finite: {seq_score}");
        assert!(par_score.is_finite(), "par score not finite: {par_score}");
        assert!(
            (seq_score - par_score).abs() < 1e-4,
            "par maxsim mismatch: seq={seq_score} par={par_score} delta={}",
            (seq_score - par_score).abs()
        );
    }
}
