//! Bit-plane packed ternary weights.

/// Bit-plane packed ternary weights: each element is {-1, 0, +1}.
///
/// 64 weights per block stored as two u64 bitmasks:
/// - pos_bits[block] bit k set → weight[row][k] = +1
/// - neg_bits[block] bit k set → weight[row][k] = -1
/// - both zero → weight = 0 (implicit skip, no storage needed)
///
/// `row_scale[r]` rescales the accumulated sum back toward original float magnitudes.
/// Memory: ~1.58 bits/weight (log₂3), plus one f32 per row for scale.
#[cfg(feature = "plasma_path")]
#[derive(Clone, Debug)]
pub struct TernaryWeights {
    pub rows: usize,
    pub cols: usize,
    pub blocks64: usize,     // (cols + 63) / 64
    pub pos_bits: Vec<u64>,  // [rows * blocks64]
    pub neg_bits: Vec<u64>,  // [rows * blocks64]
    pub row_scale: Vec<f32>, // [rows]
}

#[cfg(feature = "plasma_path")]
impl TernaryWeights {
    /// Create zeroed ternary weights.
    pub fn new(rows: usize, cols: usize) -> Self {
        let blocks64 = cols.div_ceil(64);
        Self {
            rows,
            cols,
            blocks64,
            pos_bits: vec![0u64; rows * blocks64],
            neg_bits: vec![0u64; rows * blocks64],
            row_scale: vec![1.0f32; rows],
        }
    }

    /// Set a single ternary value at (row, col). Panics if out of bounds or value not in {-1, 0, +1}.
    pub fn set(&mut self, row: usize, col: usize, value: i8) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        assert!(
            (-1..=1).contains(&value),
            "ternary value must be -1, 0, or +1"
        );
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        match value {
            1 => {
                self.pos_bits[idx] |= mask;
                self.neg_bits[idx] &= !mask;
            }
            -1 => {
                self.pos_bits[idx] &= !mask;
                self.neg_bits[idx] |= mask;
            }
            0 => {
                self.pos_bits[idx] &= !mask;
                self.neg_bits[idx] &= !mask;
            }
            _ => unreachable!(),
        }
    }

    /// Get the ternary value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> i8 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let block = col >> 6;
        let bit = col & 63;
        let mask = 1u64 << bit;
        let idx = row * self.blocks64 + block;
        let pos = (self.pos_bits[idx] & mask) != 0;
        let neg = (self.neg_bits[idx] & mask) != 0;
        pos as i8 - neg as i8
    }

    /// Quantize f32 weights to ternary with row-wise error compensation.
    ///
    /// For each row:
    ///   scale = mean(|row|)
    ///   threshold = 0.5 * scale
    ///   for each weight: adjusted = value + carry
    ///     if adjusted > threshold → +1
    ///     if adjusted < -threshold → -1
    ///     else → 0
    ///     carry = adjusted - (q * scale)
    pub fn quantize_from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(
            weights.len(),
            rows * cols,
            "weights slice must be rows*cols"
        );
        let mut tw = Self::new(rows, cols);

        for r in 0..rows {
            let row_start = r * cols;
            let row = &weights[row_start..row_start + cols];

            // Compute scale = mean(|row|)
            let abs_sum = crate::simd::simd_sum_abs_f32(row);
            let scale = abs_sum / cols as f32;
            tw.row_scale[r] = if scale > 0.0 { scale } else { 1.0 };

            let threshold = 0.5 * tw.row_scale[r];
            let mut carry = 0.0f32;

            // Inline bit manipulation to avoid per-element bounds checks in set()
            let row_base = r * tw.blocks64;
            for (c, &val) in row.iter().enumerate() {
                let adjusted = val + carry;
                let q = if adjusted > threshold {
                    1i8
                } else if adjusted < -threshold {
                    -1i8
                } else {
                    0i8
                };
                let block = c >> 6;
                let bit = c & 63;
                let mask = 1u64 << bit;
                let idx = row_base + block;
                // Branch-free: clear both bits, then set the one that matches q
                tw.pos_bits[idx] &= !mask;
                tw.neg_bits[idx] &= !mask;
                // q is 1 or -1 or 0; only set the relevant bit
                tw.pos_bits[idx] |= (q == 1) as u64 * mask;
                tw.neg_bits[idx] |= (q == -1) as u64 * mask;
                carry = adjusted - (q as f32 * tw.row_scale[r]);
            }
        }

        tw
    }

    /// Compute a checksum over all values (sum of row_scale[r] * sum of signs in row r).
    /// Used for cross-implementation verification.
    pub fn checksum(&self) -> f32 {
        let mut total = 0.0f32;
        for r in 0..self.rows {
            // Accumulate as integer to avoid per-element f32 conversion overhead.
            let mut row_sum: i32 = 0;
            let row_base = r * self.blocks64;
            for b in 0..self.blocks64 {
                let idx = row_base + b;
                row_sum += self.pos_bits[idx].count_ones() as i32;
                row_sum -= self.neg_bits[idx].count_ones() as i32;
            }
            total += self.row_scale[r] * row_sum as f32;
        }
        total
    }
}
