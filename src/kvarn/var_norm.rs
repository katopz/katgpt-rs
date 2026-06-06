//! Variance normalization via iterative Sinkhorn-style dual-scaling (Research 159).
//!
//! Equalizes per-row and per-column standard deviations of a 2D tile, reducing
//! quantization error from heterogenous magnitude distributions. The algorithm
//! iteratively adjusts log-space column and row scales until the imbalance
//! (max/min ratio of row and column standard deviations) converges.

/// Scales produced by variance normalization.
///
/// The normalized tile is: `tile[i,j] = original[i,j] / s_row[i] / s_col[j]`.
/// Reconstruction: `original[i,j] = normalized[i,j] * s_row[i] * s_col[j]`.
#[derive(Clone, Debug)]
pub struct VarianceNormScales {
    /// Column scales `[cols]`.
    pub s_col: Vec<f32>,
    /// Row scales `[rows]`.
    pub s_row: Vec<f32>,
}

/// Variance normalization configuration.
#[derive(Clone, Debug)]
pub struct VarNormConfig {
    /// Number of Sinkhorn iterations (default: 8).
    pub iterations: usize,
    /// Log clamp lower bound (default: -0.3).
    pub log_clamp_lo: f32,
    /// Log clamp upper bound (default: 10.0).
    pub log_clamp_hi: f32,
    /// Tile size — tokens per tile (default: 128).
    pub tile_size: usize,
}

impl Default for VarNormConfig {
    fn default() -> Self {
        Self {
            iterations: 8,
            log_clamp_lo: -0.3,
            log_clamp_hi: 10.0,
            tile_size: 128,
        }
    }
}

/// Perform variance normalization on a tile `[rows, cols]` stored row-major.
///
/// Returns the scales used for normalization. The tile is modified in-place:
/// after the call, `tile[i,j] = original[i,j] / s_row[i] / s_col[j]`.
///
/// # Panics
///
/// Panics if `tile.len() != rows * cols`.
pub fn variance_normalize(
    tile: &mut [f32],
    rows: usize,
    cols: usize,
    config: &VarNormConfig,
) -> VarianceNormScales {
    assert_eq!(tile.len(), rows * cols, "tile size mismatch");

    if rows == 0 || cols == 0 {
        return VarianceNormScales {
            s_col: vec![],
            s_row: vec![],
        };
    }

    let log_clamp_lo = config.log_clamp_lo;
    let log_clamp_hi = config.log_clamp_hi;

    // Initialize log scales to zero (linear scale = 1.0).
    let mut log_s_col = vec![0.0f32; cols];
    let mut log_s_row = vec![0.0f32; rows];

    // Compute initial current = tile / exp(log_s_col) / exp(log_s_row)
    let mut cur = tile.to_vec();
    apply_dual_scale(&mut cur, rows, cols, &log_s_row, &log_s_col);

    // Track best imbalance
    let mut col_s = col_stds(&cur, rows, cols);
    let mut row_s = row_stds(&cur, rows, cols);
    let mut imb_best = imbalance(&col_s, &row_s);
    let mut log_s_col_best = log_s_col.clone();
    let mut log_s_row_best = log_s_row.clone();

    for _k in 0..config.iterations {
        // Column step: update log_s_col based on column std devs
        for (j, &s) in col_s.iter().enumerate() {
            let log_s = s.ln();
            let clamped = log_s.clamp(log_clamp_lo, log_clamp_hi);
            log_s_col[j] = (log_s_col[j] + clamped).clamp(log_clamp_lo, log_clamp_hi);
        }

        // Recompute current
        cur.copy_from_slice(tile);
        apply_dual_scale(&mut cur, rows, cols, &log_s_row, &log_s_col);

        // Row step: update log_s_row based on row std devs
        row_s = row_stds(&cur, rows, cols);
        for (i, &s) in row_s.iter().enumerate() {
            let log_s = s.ln();
            let clamped = log_s.clamp(log_clamp_lo, log_clamp_hi);
            log_s_row[i] = (log_s_row[i] + clamped).clamp(log_clamp_lo, log_clamp_hi);
        }

        // Recompute current
        cur.copy_from_slice(tile);
        apply_dual_scale(&mut cur, rows, cols, &log_s_row, &log_s_col);

        // Check imbalance
        col_s = col_stds(&cur, rows, cols);
        row_s = row_stds(&cur, rows, cols);
        let imb_cur = imbalance(&col_s, &row_s);

        if imb_cur <= imb_best {
            imb_best = imb_cur;
            log_s_col_best = log_s_col.clone();
            log_s_row_best = log_s_row.clone();
        }
    }

    // Apply best scales to the tile in-place
    let s_col: Vec<f32> = log_s_col_best.iter().map(|&l| l.exp()).collect();
    let s_row: Vec<f32> = log_s_row_best.iter().map(|&l| l.exp()).collect();
    apply_scales_into(tile, rows, cols, &s_row, &s_col);

    VarianceNormScales { s_col, s_row }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply dual scale to a tile: `out[i,j] = cur[i,j] / s_row[i] / s_col[j]`.
#[inline]
fn apply_dual_scale(
    cur: &mut [f32],
    rows: usize,
    cols: usize,
    log_s_row: &[f32],
    log_s_col: &[f32],
) {
    // Precompute exp of log scales
    let inv_row: Vec<f32> = log_s_row.iter().map(|&l| 1.0 / l.exp()).collect();
    let inv_col: Vec<f32> = log_s_col.iter().map(|&l| 1.0 / l.exp()).collect();

    for i in 0..rows {
        let row_scale = inv_row[i];
        for j in 0..cols {
            cur[i * cols + j] *= row_scale * inv_col[j];
        }
    }
}

/// Apply scales to a tile in-place: `tile[i,j] /= s_row[i] * s_col[j]`.
#[inline]
fn apply_scales_into(tile: &mut [f32], rows: usize, cols: usize, s_row: &[f32], s_col: &[f32]) {
    for i in 0..rows {
        let inv_row = 1.0 / s_row[i];
        for j in 0..cols {
            tile[i * cols + j] *= inv_row / s_col[j];
        }
    }
}

/// Compute standard deviation of each column in a `[rows, cols]` row-major tile.
#[inline]
pub(crate) fn col_stds(tile: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; cols];
    if rows == 0 {
        return result;
    }
    // Two-pass: mean then variance
    let mut mean = vec![0.0f32; cols];
    for i in 0..rows {
        for j in 0..cols {
            mean[j] += tile[i * cols + j];
        }
    }
    let inv_rows = 1.0 / rows as f32;
    for m in mean.iter_mut() {
        *m *= inv_rows;
    }
    for i in 0..rows {
        for j in 0..cols {
            let d = tile[i * cols + j] - mean[j];
            result[j] += d * d;
        }
    }
    for r in result.iter_mut() {
        *r = (*r * inv_rows).sqrt();
    }
    result
}

/// Compute standard deviation of each row in a `[rows, cols]` row-major tile.
#[inline]
pub(crate) fn row_stds(tile: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; rows];
    if cols == 0 {
        return result;
    }
    let inv_cols = 1.0 / cols as f32;
    for (i, res) in result.iter_mut().enumerate() {
        let mut mean = 0.0f32;
        let off = i * cols;
        for j in 0..cols {
            mean += tile[off + j];
        }
        mean *= inv_cols;
        let mut var = 0.0f32;
        for j in 0..cols {
            let d = tile[off + j] - mean;
            var += d * d;
        }
        *res = (var * inv_cols).sqrt();
    }
    result
}

/// Compute imbalance metric: max/min ratio of column stds + max/min ratio of row stds.
///
/// Lower is better (perfectly equal = 2.0).
#[inline]
pub(crate) fn imbalance(col_s: &[f32], row_s: &[f32]) -> f32 {
    let col_ratio = ratio_max_min(col_s);
    let row_ratio = ratio_max_min(row_s);
    col_ratio + row_ratio
}

/// max/min ratio, with epsilon guard to avoid division by zero.
#[inline]
fn ratio_max_min(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let lo = vals.iter().copied().fold(f32::MAX, f32::min).max(1e-8);
    let hi = vals.iter().copied().fold(f32::MIN, f32::max).max(1e-8);
    hi / lo
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_normalize_reduces_imbalance() {
        // Create a tile with very different magnitudes across rows and columns
        let rows = 4;
        let cols = 8;
        let mut tile = vec![0.0f32; rows * cols];
        // Row 0: large values, Row 3: small values
        for j in 0..cols {
            tile[j] = 100.0 * (j as f32 + 1.0);
            tile[cols + j] = 10.0 * (j as f32 + 1.0);
            tile[2 * cols + j] = 1.0 * (j as f32 + 1.0);
            tile[3 * cols + j] = 0.1 * (j as f32 + 1.0);
        }

        let original = tile.clone();
        let col_s_before = col_stds(&original, rows, cols);
        let row_s_before = row_stds(&original, rows, cols);
        let imb_before = imbalance(&col_s_before, &row_s_before);

        let config = VarNormConfig {
            iterations: 20,
            ..Default::default()
        };
        let scales = variance_normalize(&mut tile, rows, cols, &config);

        let col_s_after = col_stds(&tile, rows, cols);
        let row_s_after = row_stds(&tile, rows, cols);
        let imb_after = imbalance(&col_s_after, &row_s_after);

        assert!(
            imb_after < imb_before,
            "imbalance should decrease: before={imb_before}, after={imb_after}"
        );
        // Scales should be non-trivial (not all 1.0)
        assert!(
            scales.s_row.iter().any(|&s| (s - 1.0).abs() > 0.01),
            "row scales should differ from 1.0"
        );
    }

    #[test]
    fn test_variance_normalize_roundtrip() {
        let rows = 4;
        let cols = 4;
        let mut tile = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let original = tile.clone();

        let config = VarNormConfig::default();
        let scales = variance_normalize(&mut tile, rows, cols, &config);

        // Reconstruct: tile_normalized * s_row[i] * s_col[j]
        let mut reconstructed = tile.clone();
        for i in 0..rows {
            for j in 0..cols {
                reconstructed[i * cols + j] *= scales.s_row[i] * scales.s_col[j];
            }
        }

        for (a, b) in reconstructed.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-2,
                "roundtrip mismatch: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_col_stds() {
        let tile = vec![
            1.0f32, 2.0, // row 0
            3.0, 4.0, // row 1
        ];
        let stds = col_stds(&tile, 2, 2);
        // col 0: [1, 3], mean=2, std=1
        assert!((stds[0] - 1.0).abs() < 1e-5, "col 0 std: {}", stds[0]);
        // col 1: [2, 4], mean=3, std=1
        assert!((stds[1] - 1.0).abs() < 1e-5, "col 1 std: {}", stds[1]);
    }

    #[test]
    fn test_row_stds() {
        let tile = vec![
            1.0f32, 3.0, // row 0
            2.0, 4.0, // row 1
        ];
        let stds = row_stds(&tile, 2, 2);
        // row 0: [1, 3], mean=2, std=1
        assert!((stds[0] - 1.0).abs() < 1e-5, "row 0 std: {}", stds[0]);
        // row 1: [2, 4], mean=3, std=1
        assert!((stds[1] - 1.0).abs() < 1e-5, "row 1 std: {}", stds[1]);
    }

    #[test]
    fn test_imbalance_uniform() {
        // Uniform stds → imbalance = 1.0 + 1.0 = 2.0
        let col_s = vec![1.0f32, 1.0, 1.0];
        let row_s = vec![1.0f32, 1.0];
        let imb = imbalance(&col_s, &row_s);
        assert!(
            (imb - 2.0).abs() < 1e-5,
            "uniform imbalance should be 2.0, got {imb}"
        );
    }

    #[test]
    fn test_empty_tile() {
        let config = VarNormConfig::default();
        let mut tile: Vec<f32> = vec![];
        let scales = variance_normalize(&mut tile, 0, 4, &config);
        assert!(scales.s_col.is_empty());
        assert!(scales.s_row.is_empty());
    }

    #[test]
    fn test_variance_normalize_benchmark_128x128() {
        // Generate random 128×128 tile
        let rows = 128;
        let cols = 128;
        let mut tile = vec![0.0f32; rows * cols];
        let mut seed: u64 = 42;
        for v in tile.iter_mut() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((seed >> 33) as i32 as f32) / (1i32 << 31) as f32;
        }

        let config = VarNormConfig {
            iterations: 8,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        let _scales = variance_normalize(&mut tile, rows, cols, &config);
        let elapsed = start.elapsed();
        let elapsed_us = elapsed.as_secs_f64() * 1e6;

        eprintln!("VarN 128×128 8 iters: {elapsed_us:.0}μs");

        // Relaxed CI bound — actual target is ≤50μs on Apple M2 SIMD
        assert!(
            elapsed.as_secs() < 5,
            "VarN took too long: {elapsed_us:.0}μs"
        );
    }
}
