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
    let mut step = 2;
    while step <= n {
        let half = step / 2;
        for block_start in (0..n).step_by(step) {
            for i in 0..half {
                let a = x[block_start + i];
                let b = x[block_start + half + i];
                x[block_start + i] = (a + b) * inv_sqrt2;
                x[block_start + half + i] = (a - b) * inv_sqrt2;
            }
        }
        step *= 2;
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

        let mut single = vec![3.14f32];
        hadamard_transform_inplace(&mut single);
        assert!((single[0] - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_hadamard_non_power_of_two_noop() {
        let mut buf = vec![1.0f32, 2.0, 3.0]; // length 3, not power of 2
        let original = buf.clone();
        hadamard_transform_inplace(&mut buf);
        assert_eq!(buf, original, "non-power-of-2 should be no-op");
    }
}
