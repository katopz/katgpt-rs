//! Channel SIMD Alignment — cache-line-aligned storage for ternary weights.
//! Pads weight rows to 64-byte boundaries for optimal SIMD throughput.
//! Feature-gated behind `channel_simd_align`.

/// Cache line size in bytes.
const CACHE_LINE: usize = 64;

/// Aligned weight matrix with cache-line-padded rows.
/// Each row is padded to a multiple of CACHE_LINE bytes.
#[derive(Debug, Clone)]
pub struct AlignedWeightMatrix {
    /// Padded data. Row i starts at offsets[i].
    pub data: Vec<f32>,
    /// Row offsets into data (for O(1) row access).
    pub offsets: Vec<usize>,
    /// Original (unpadded) row dimension.
    pub row_dim: usize,
    /// Padded row dimension (multiple of CACHE_LINE / sizeof(f32)).
    pub padded_dim: usize,
    /// Number of rows.
    pub num_rows: usize,
}

impl AlignedWeightMatrix {
    /// Create a new aligned matrix from row-major data.
    /// Pads each row to CACHE_LINE boundary.
    pub fn from_rows(rows: &[Vec<f32>]) -> Self {
        if rows.is_empty() {
            return Self {
                data: Vec::new(),
                offsets: Vec::new(),
                row_dim: 0,
                padded_dim: 0,
                num_rows: 0,
            };
        }

        let row_dim = rows[0].len();
        let padded_dim = Self::pad_dim(row_dim);

        let mut data = Vec::with_capacity(padded_dim * rows.len());
        let mut offsets = Vec::with_capacity(rows.len());

        for row in rows {
            offsets.push(data.len());
            data.extend_from_slice(row);
            // Pad with zeros
            let padding = padded_dim - row.len();
            data.extend(std::iter::repeat(0.0f32).take(padding));
        }

        Self {
            data,
            offsets,
            row_dim,
            padded_dim,
            num_rows: rows.len(),
        }
    }

    /// Pad dimension to cache line boundary.
    fn pad_dim(dim: usize) -> usize {
        let f32_per_line = CACHE_LINE / std::mem::size_of::<f32>();
        ((dim + f32_per_line - 1) / f32_per_line) * f32_per_line
    }

    /// Get a pointer to the start of row i (aligned to cache line).
    pub fn row_ptr(&self, i: usize) -> *const f32 {
        self.data[self.offsets[i]..].as_ptr()
    }

    /// Get row data (padded).
    pub fn row(&self, i: usize) -> &[f32] {
        let start = self.offsets[i];
        &self.data[start..start + self.padded_dim]
    }

    /// Dot product of a vector with row i, using only the original (unpadded) dimensions.
    pub fn dot_row(&self, vec: &[f32], row_idx: usize) -> f32 {
        let row = self.row(row_idx);
        let len = self.row_dim.min(vec.len());
        crate::simd::simd_dot_f32(&vec[..len], &row[..len], len)
    }

    /// Matrix-vector multiply: y = A * x.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        (0..self.num_rows).map(|i| self.dot_row(x, i)).collect()
    }

    /// Quantize a float row into the aligned matrix.
    /// Uses cache-line-aligned writes for SIMD-friendly layout.
    pub fn quantize_row(&mut self, row_idx: usize, data: &[f32]) {
        debug_assert!(row_idx < self.num_rows, "row index out of bounds");
        let start = self.offsets[row_idx];
        let copy_len = data.len().min(self.row_dim);
        self.data[start..start + copy_len].copy_from_slice(&data[..copy_len]);
        // Zero-pad remainder
        self.data[start + copy_len..start + self.padded_dim].fill(0.0);
    }

    /// Dequantize a row from the aligned matrix back to a target buffer.
    /// Only copies the original (unpadded) dimensions.
    pub fn dequantize_row(&self, row_idx: usize, out: &mut [f32]) {
        debug_assert!(row_idx < self.num_rows, "row index out of bounds");
        let start = self.offsets[row_idx];
        let copy_len = out.len().min(self.row_dim);
        out[..copy_len].copy_from_slice(&self.data[start..start + copy_len]);
    }

    /// Batch matvec with pre-allocated output buffer (zero-alloc on repeated calls).
    pub fn matvec_into(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.num_rows, "output length mismatch");
        for i in 0..self.num_rows {
            out[i] = self.dot_row(x, i);
        }
    }

    /// Convert from ternary weight representation to aligned float matrix.
    /// Each ternary value {-1, 0, +1} is converted to f32, then aligned.
    ///
    /// Writes directly into a pre-sized slice (no per-element `push`/growth),
    /// and pads each row's tail with a single `fill(0.0)` call.
    pub fn from_ternary(
        pos_bits: &[u64],
        neg_bits: &[u64],
        row_scale: &[f32],
        rows: usize,
        cols: usize,
        blocks64: usize,
    ) -> Self {
        let row_dim = cols;
        let padded_dim = Self::pad_dim(row_dim);

        let total_len = padded_dim
            .checked_mul(rows)
            .expect("padded_dim * rows overflows");
        let mut data = Vec::with_capacity(total_len);
        // SAFETY: we will fully initialize `total_len` f32s below before any read.
        // Each row writes exactly `cols` ternary values then `padding` zeros.
        // Allow: intentional uninit-then-fill to skip the memset.
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(total_len)
        };
        let mut offsets = Vec::with_capacity(rows);

        let mut write_pos = 0usize;
        for r in 0..rows {
            offsets.push(write_pos);
            let scale = unsafe { *row_scale.get_unchecked(r) };
            let row_base = r * blocks64;

            // Decode ternary bits into the destination row slice directly.
            // Branch-free inner: compute `val = sign * scale` where sign ∈ {-1, 0, +1}.
            //
            // We walk one 64-bit block at a time, draining the block bit-by-bit
            // via `>>= 1`. This avoids the per-element `1u64 << (c & 63)` shift
            // and the per-element `idx = row_base + (c >> 6)` recomputation —
            // both of which LLVM rarely hoists cleanly across the bounds-checked
            // load. Bit order is LSB-first, matching the original
            // `bit = 1u64 << (c & 63)` indexing.
            let row_dst = &mut data[write_pos..write_pos + cols];
            let mut c = 0usize;
            for block in 0..blocks64 {
                let mut pos_w = unsafe { *pos_bits.get_unchecked(row_base + block) };
                let mut neg_w = unsafe { *neg_bits.get_unchecked(row_base + block) };
                let block_end = (c + 64).min(cols);
                while c < block_end {
                    let pos = pos_w & 1 != 0;
                    let neg = neg_w & 1 != 0;
                    pos_w >>= 1;
                    neg_w >>= 1;
                    let sign = (pos as i32) - (neg as i32);
                    // sign ∈ {-1, 0, +1}; multiply by scale once.
                    unsafe {
                        *row_dst.get_unchecked_mut(c) = (sign as f32) * scale;
                    }
                    c += 1;
                }
            }

            // Zero-pad the tail of this row in one shot.
            let pad = padded_dim - cols;
            if pad > 0 {
                data[write_pos + cols..write_pos + padded_dim].fill(0.0);
            }
            write_pos += padded_dim;
        }

        Self {
            data,
            offsets,
            row_dim,
            padded_dim,
            num_rows: rows,
        }
    }

    /// Memory overhead from padding.
    pub fn padding_overhead(&self) -> f32 {
        let original = self.row_dim * self.num_rows * std::mem::size_of::<f32>();
        let padded = self.padded_dim * self.num_rows * std::mem::size_of::<f32>();
        if original == 0 {
            return 0.0;
        }
        (padded - original) as f32 / original as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_dim_alignment() {
        let dim = AlignedWeightMatrix::pad_dim(10);
        assert_eq!(dim % (CACHE_LINE / std::mem::size_of::<f32>()), 0);
        assert!(dim >= 10);
    }

    #[test]
    fn test_from_rows() {
        let rows = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mat = AlignedWeightMatrix::from_rows(&rows);
        assert_eq!(mat.num_rows, 2);
        assert_eq!(mat.row_dim, 3);
        assert!(mat.padded_dim >= 3);
    }

    #[test]
    fn test_dot_row() {
        let rows = vec![vec![1.0, 2.0, 3.0]];
        let mat = AlignedWeightMatrix::from_rows(&rows);
        let vec = vec![1.0, 1.0, 1.0];
        let dot = mat.dot_row(&vec, 0);
        assert!((dot - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec() {
        let rows = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mat = AlignedWeightMatrix::from_rows(&rows);
        let x = vec![3.0, 5.0];
        let y = mat.matvec(&x);
        assert!((y[0] - 3.0).abs() < 1e-6);
        assert!((y[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_padding_overhead() {
        let rows = vec![vec![0.0; 16]];
        let mat = AlignedWeightMatrix::from_rows(&rows);
        let overhead = mat.padding_overhead();
        assert!(overhead >= 0.0);
        // 16 f32 = 64 bytes = exactly one cache line, so no padding needed
        assert!(
            overhead < 0.01,
            "Expected near-zero overhead for exact cache line fit, got {}",
            overhead
        );
    }

    #[test]
    fn test_empty_matrix() {
        let mat = AlignedWeightMatrix::from_rows(&[]);
        assert_eq!(mat.num_rows, 0);
    }
}
