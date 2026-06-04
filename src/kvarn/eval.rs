//! Pseudo-decode evaluation harness for KVarN (Research 159).
//!
//! Simulates autoregressive KV-cache quantization error accumulation.
//! Splits a sequence into tiles, quantizes each tile's KV cache, and
//! tracks per-tile MSE, cosine similarity, cumulative error, and max
//! magnitude error.

use super::kv_cache::{KVarNConfig, KVarNKVCache};
use super::var_norm::VarNormConfig;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a pseudo-decode evaluation.
#[derive(Clone, Debug)]
pub struct PseudoDecodeResult {
    /// Per-tile MSE (quantize + dequantize error).
    pub per_tile_mse: Vec<f32>,
    /// Cumulative MSE up to each tile.
    pub cumulative_mse: Vec<f32>,
    /// Per-tile cosine similarity.
    pub per_tile_cosine: Vec<f32>,
    /// Worst-case token error (magnitude error) per tile.
    pub per_tile_max_magnitude_error: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Run pseudo-decode evaluation on a sequence of key/value vectors.
///
/// Simulates autoregressive KV-cache quantization error accumulation:
/// 1. Split sequence into tiles of `tile_size` tokens
/// 2. After each tile, quantize the entire KV cache up to that point
/// 3. Compute MSE / cosine / max-error for the newly quantized tile
/// 4. Track cumulative error as tiles accumulate
///
/// Returns per-tile metrics.
pub fn pseudo_decode_eval(
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    tile_size: usize,
    bits: u8,
    config: &VarNormConfig,
) -> PseudoDecodeResult {
    let seq_len = keys.len();
    assert_eq!(
        values.len(),
        seq_len,
        "keys and values must have same length"
    );
    if seq_len == 0 {
        return PseudoDecodeResult {
            per_tile_mse: vec![],
            cumulative_mse: vec![],
            per_tile_cosine: vec![],
            per_tile_max_magnitude_error: vec![],
        };
    }

    let kv_dim = keys[0].len();
    let n_layers = 1;
    let max_seq = seq_len;

    let cache_config = KVarNConfig {
        n_layers,
        kv_dim,
        max_seq_len: max_seq,
        bits,
        tile_size,
        var_norm: config.clone(),
    };

    let n_tiles = (seq_len + tile_size - 1) / tile_size;
    let mut per_tile_mse = Vec::with_capacity(n_tiles);
    let mut cumulative_mse = Vec::with_capacity(n_tiles);
    let mut per_tile_cosine = Vec::with_capacity(n_tiles);
    let mut per_tile_max_mag = Vec::with_capacity(n_tiles);

    let mut total_sq_error = 0.0f32;
    let mut total_count = 0usize;

    let mut cache = KVarNKVCache::with_config(&cache_config);
    let mut out = vec![0.0f32; kv_dim];

    // Process tiles
    for tile_start in (0..seq_len).step_by(tile_size) {
        let tile_end = (tile_start + tile_size).min(seq_len);

        // Store all positions in this tile
        for pos in tile_start..tile_end {
            cache.store_key(0, pos, &keys[pos]);
            cache.store_value(0, pos, &values[pos]);
        }

        // Evaluate error for this tile's positions
        let mut tile_sq_error = 0.0f32;
        let mut tile_count = 0usize;
        let mut tile_cosine_sum = 0.0f32;
        let mut tile_max_mag_error = 0.0f32;

        for pos in tile_start..tile_end {
            // Key error
            cache.dequantize_key_into(0, pos, &mut out);
            let key_mse = per_coord_mse(&keys[pos], &out);
            tile_sq_error += key_mse * kv_dim as f32;
            tile_cosine_sum += cosine_sim(&keys[pos], &out);
            tile_max_mag_error = tile_max_mag_error.max(max_magnitude_error(&keys[pos], &out));

            // Value error
            cache.dequantize_value_into(0, pos, &mut out);
            let val_mse = per_coord_mse(&values[pos], &out);
            tile_sq_error += val_mse * kv_dim as f32;
            tile_cosine_sum += cosine_sim(&values[pos], &out);
            tile_max_mag_error = tile_max_mag_error.max(max_magnitude_error(&values[pos], &out));

            tile_count += 2; // key + value
        }

        let tile_mse = if tile_count > 0 {
            tile_sq_error / (tile_count * kv_dim) as f32
        } else {
            0.0
        };
        let tile_cosine = if tile_count > 0 {
            tile_cosine_sum / tile_count as f32
        } else {
            1.0
        };

        per_tile_mse.push(tile_mse);
        per_tile_cosine.push(tile_cosine);
        per_tile_max_mag.push(tile_max_mag_error);

        total_sq_error += tile_sq_error;
        total_count += tile_count * kv_dim;
        cumulative_mse.push(if total_count > 0 {
            total_sq_error / total_count as f32
        } else {
            0.0
        });
    }

    PseudoDecodeResult {
        per_tile_mse,
        cumulative_mse,
        per_tile_cosine,
        per_tile_max_magnitude_error: per_tile_max_mag,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    dot / (na * nb)
}

#[inline]
fn per_coord_mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        / a.len() as f32
}

#[inline]
fn max_magnitude_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_random_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i32 as f32) / (1i32 << 31) as f32
            })
            .collect()
    }

    #[test]
    fn test_pseudo_decode_eval() {
        let kv_dim = 64;
        let seq_len = 32;
        let tile_size = 8;
        let bits = 4;
        let config = VarNormConfig {
            tile_size,
            iterations: 8,
            ..Default::default()
        };

        let keys: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| make_random_vec(kv_dim, i as u64 * 100 + 1))
            .collect();
        let values: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| make_random_vec(kv_dim, i as u64 * 100 + 2))
            .collect();

        let result = pseudo_decode_eval(&keys, &values, tile_size, bits, &config);

        let n_tiles = (seq_len + tile_size - 1) / tile_size;
        assert_eq!(result.per_tile_mse.len(), n_tiles);
        assert_eq!(result.cumulative_mse.len(), n_tiles);
        assert_eq!(result.per_tile_cosine.len(), n_tiles);
        assert_eq!(result.per_tile_max_magnitude_error.len(), n_tiles);

        // All MSE values should be finite and non-negative
        for &mse in &result.per_tile_mse {
            assert!(
                mse.is_finite() && mse >= 0.0,
                "MSE should be finite and non-negative: {mse}"
            );
        }

        // All cosine similarities should be in [0, 1] (positive for random data)
        for &cos in &result.per_tile_cosine {
            assert!(
                cos >= -0.1 && cos <= 1.01,
                "cosine should be in [-0.1, 1.0]: {cos}"
            );
        }

        // Cumulative MSE should be monotonically related to per-tile MSE
        // (may not be monotonic since it's a running average)
        for &cmse in &result.cumulative_mse {
            assert!(cmse.is_finite() && cmse >= 0.0);
        }

        // With 4-bit quantization, cosine similarity should be reasonable
        let avg_cosine: f32 = result.per_tile_cosine.iter().sum::<f32>() / n_tiles as f32;
        assert!(
            avg_cosine > 0.7,
            "average cosine similarity should be reasonable: {avg_cosine}"
        );
    }

    #[test]
    fn test_pseudo_decode_empty_sequence() {
        let config = VarNormConfig::default();
        let result = pseudo_decode_eval(&[], &[], 8, 4, &config);
        assert!(result.per_tile_mse.is_empty());
        assert!(result.cumulative_mse.is_empty());
    }

    #[test]
    fn test_pseudo_decode_single_token() {
        let kv_dim = 16;
        let config = VarNormConfig {
            tile_size: 1,
            ..Default::default()
        };
        let key = make_random_vec(kv_dim, 42);
        let val = make_random_vec(kv_dim, 43);

        let result = pseudo_decode_eval(&[key.clone()], &[val.clone()], 1, 4, &config);
        assert_eq!(result.per_tile_mse.len(), 1);
        assert_eq!(result.per_tile_cosine.len(), 1);
        assert!(result.per_tile_mse[0].is_finite());
    }

    #[test]
    fn test_pseudo_decode_higher_bits_lower_error() {
        let kv_dim = 32;
        let seq_len = 8;
        let tile_size = 4;
        let keys: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| make_random_vec(kv_dim, i as u64 * 100 + 1))
            .collect();
        let values: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| make_random_vec(kv_dim, i as u64 * 100 + 2))
            .collect();

        let config_2bit = VarNormConfig {
            tile_size,
            ..Default::default()
        };
        let config_8bit = VarNormConfig {
            tile_size,
            ..Default::default()
        };

        let result_2bit = pseudo_decode_eval(&keys, &values, tile_size, 2, &config_2bit);
        let result_8bit = pseudo_decode_eval(&keys, &values, tile_size, 8, &config_8bit);

        // Higher bits should give lower (or equal) MSE
        let avg_mse_2bit: f32 =
            result_2bit.per_tile_mse.iter().sum::<f32>() / result_2bit.per_tile_mse.len() as f32;
        let avg_mse_8bit: f32 =
            result_8bit.per_tile_mse.iter().sum::<f32>() / result_8bit.per_tile_mse.len() as f32;
        assert!(
            avg_mse_8bit <= avg_mse_2bit + 0.01,
            "8-bit MSE should be <= 2-bit MSE: 8bit={avg_mse_8bit}, 2bit={avg_mse_2bit}"
        );

        // Higher bits should give higher cosine similarity
        let avg_cos_2bit: f32 = result_2bit.per_tile_cosine.iter().sum::<f32>()
            / result_2bit.per_tile_cosine.len() as f32;
        let avg_cos_8bit: f32 = result_8bit.per_tile_cosine.iter().sum::<f32>()
            / result_8bit.per_tile_cosine.len() as f32;
        assert!(
            avg_cos_8bit >= avg_cos_2bit - 0.05,
            "8-bit cosine should be >= 2-bit cosine: 8bit={avg_cos_8bit}, 2bit={avg_cos_2bit}"
        );
    }
}
