//! Walsh-Hadamard transform for KVarN (Research 159).
//!
//! Self-contained implementation — the `shard_kv` feature is optional, so we
//! keep our own copy rather than depending on it. Power-of-2 lengths only.
//!
//! Uses the orthogonal normalization (1/√2 per butterfly step), making the
//! transform self-inverse: H(H(x)) = x.

/// In-place orthogonal Walsh-Hadamard transform on a power-of-2-length buffer.
///
/// O(n log n), no allocations. Each butterfly step multiplies by 1/√2,
/// so the total normalization per application is 1/√n. This makes the
/// transform self-inverse: H(H(x)) = x.
///
/// Uses unsafe pointer arithmetic to help LLVM auto-vectorize the butterfly
/// loop by proving non-aliasing between the two halves.
#[inline]
pub fn hadamard_transform_inplace(x: &mut [f32]) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    // Only power-of-2 lengths are supported.
    if !n.is_power_of_two() {
        return;
    }

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    let ptr = x.as_mut_ptr();

    // Safety: n is power-of-2 (verified above), all indices are in bounds.
    unsafe {
        let mut step = 2;
        while step <= n {
            let half = step / 2;
            let mut block_start = 0;
            while block_start < n {
                for i in 0..half {
                    let a = *ptr.add(block_start + i);
                    let b = *ptr.add(block_start + half + i);
                    *ptr.add(block_start + i) = (a + b) * inv_sqrt2;
                    *ptr.add(block_start + half + i) = (a - b) * inv_sqrt2;
                }
                block_start += step;
            }
            step *= 2;
        }
    }
}

/// Apply Hadamard to each row of a 2D tile `[rows, cols]` stored row-major.
///
/// Each row must have power-of-2 length (`cols`). This is the common case
/// since kv_dim is typically 64, 128, 256, etc.
pub fn hadamard_rows(tile: &mut [f32], cols: usize) {
    if cols == 0 {
        return;
    }
    for row in tile.chunks_exact_mut(cols) {
        hadamard_transform_inplace(row);
    }
}

/// Apply Hadamard to each column of a 2D tile `[rows, cols]` stored row-major.
///
/// Each column must have power-of-2 length (`rows`). This is the common case
/// since kv_dim is typically 64, 128, 256, etc.
///
/// Equivalent to per-column Hadamard: mixes values across rows within each column.
///
/// Allocates a per-column scratch buffer. For hot paths, prefer
/// [`hadamard_cols_into`] which reuses a caller-owned scratch buffer.
pub fn hadamard_cols(tile: &mut [f32], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 || !rows.is_power_of_two() {
        return;
    }
    let mut col_buf = vec![0.0f32; rows];
    hadamard_cols_into(tile, rows, cols, &mut col_buf);
}

/// Zero-allocation variant of [`hadamard_cols`].
///
/// `col_buf` is caller-owned scratch with length `>= rows`; contents are overwritten.
/// Reuse the same buffer across many tiles to eliminate per-tile allocation.
#[inline]
pub fn hadamard_cols_into(tile: &mut [f32], rows: usize, cols: usize, col_buf: &mut [f32]) {
    if rows == 0 || cols == 0 || !rows.is_power_of_two() {
        return;
    }
    debug_assert!(col_buf.len() >= rows, "col_buf must hold at least `rows` elements");
    let buf = &mut col_buf[..rows];
    // Process one column at a time: gather strided, transform, scatter strided.
    for j in 0..cols {
        for i in 0..rows {
            buf[i] = tile[i * cols + j];
        }
        hadamard_transform_inplace(buf);
        for i in 0..rows {
            tile[i * cols + j] = buf[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_roundtrip() {
        let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = buf.clone();
        // Orthogonal Hadamard is self-inverse: H(H(x)) = x
        hadamard_transform_inplace(&mut buf);
        hadamard_transform_inplace(&mut buf);
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "roundtrip mismatch: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_hadamard_unit_vector() {
        // Hadamard preserves L2 norm (it's orthogonal with 1/√2 factors).
        let mut buf = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let norm_before: f32 = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
        hadamard_transform_inplace(&mut buf);
        let norm_after: f32 = buf.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-5,
            "norm not preserved: before={norm_before}, after={norm_after}"
        );
    }

    #[test]
    fn test_hadamard_rows() {
        let mut tile = vec![
            1.0f32, 2.0, 3.0, 4.0, // row 0
            5.0f32, 6.0, 7.0, 8.0, // row 1
        ];
        hadamard_rows(&mut tile, 4);
        // Each row should be transformed independently.
        let expected_row0 = {
            let mut r = vec![1.0f32, 2.0, 3.0, 4.0];
            hadamard_transform_inplace(&mut r);
            r
        };
        let expected_row1 = {
            let mut r = vec![5.0f32, 6.0, 7.0, 8.0];
            hadamard_transform_inplace(&mut r);
            r
        };
        for i in 0..4 {
            assert!(
                (tile[i] - expected_row0[i]).abs() < 1e-5,
                "row 0 mismatch at {i}"
            );
            assert!(
                (tile[4 + i] - expected_row1[i]).abs() < 1e-5,
                "row 1 mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_hadamard_empty_and_single() {
        let mut empty: Vec<f32> = vec![];
        hadamard_transform_inplace(&mut empty);
        assert!(empty.is_empty());

        let mut single = vec![1.5f32];
        hadamard_transform_inplace(&mut single);
        assert!((single[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_hadamard_non_power_of_two_noop() {
        let mut buf = vec![1.0f32, 2.0, 3.0]; // length 3, not power of 2
        let original = buf.clone();
        hadamard_transform_inplace(&mut buf);
        assert_eq!(buf, original, "non-power-of-2 should be no-op");
    }
}
