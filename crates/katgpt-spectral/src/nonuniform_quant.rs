//! Non-uniform quantizer combining two-regime allocation + per-dim water-fill.
//!
//! Two paths:
//! - v1 uniform: single shared semantic codebook, b_high/b_low
//! - v2 water-fill: per-semantic-dim codebooks with water-fill bit distribution

use super::spectral::{BitAllocator, LloydMaxQuantizer, waterfill_bits};
use super::types::WaterfillAllocation;

/// End-to-end non-uniform quantizer.
///
/// Operates on pre-rotated vectors where first d_eff coords are semantic
/// (high-energy) and the rest are tail regime.
pub struct NonUniformQuantizer {
    eigenvalues: Vec<f32>,
    avg_bits: f32,
    head_dim: usize,
    max_lloyd_iter: usize,
    seed: u64,
    use_water_fill: bool,
    wf_min_bits: u8,
    wf_max_bits: Option<u8>,
    // Fitted state:
    allocator: BitAllocator,
    d_eff_int: usize,
    b_high: u8,
    b_low: u8,
    semantic_quantizer: Option<LloydMaxQuantizer>,
    tail_quantizer: Option<LloydMaxQuantizer>,
    per_dim_semantic_quantizers: Option<Vec<LloydMaxQuantizer>>,
    semantic_bits_per_dim: Option<Vec<u8>>,
    waterfill_allocation: Option<WaterfillAllocation>,
    is_fitted: bool,
}

/// Compressed vector representation.
///
/// Stores indices as u32 (not bit-packed) — bit-packing is in spectral_kv_cache.rs.
pub struct CompressedVector {
    pub semantic_indices: Vec<u32>,
    pub tail_indices: Vec<u32>,
    pub d_eff: usize,
    pub head_dim: usize,
    pub b_high: u8,
    pub b_low: u8,
    pub semantic_bits_per_dim: Option<Vec<u8>>,
    pub actual_bits_used: f64,
    pub mse: f32,
}

impl NonUniformQuantizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eigenvalues: Vec<f32>,
        avg_bits: f32,
        head_dim: usize,
        max_lloyd_iter: usize,
        seed: u64,
        use_water_fill: bool,
        wf_min_bits: u8,
        wf_max_bits: Option<u8>,
    ) -> Self {
        let d_eff_float = super::spectral::participation_ratio(&eigenvalues);
        let d_eff_int = (d_eff_float.ceil() as usize).max(1).min(head_dim);
        let allocator = BitAllocator::new(1, 8);

        Self {
            eigenvalues,
            avg_bits,
            head_dim,
            max_lloyd_iter,
            seed,
            use_water_fill,
            wf_min_bits,
            wf_max_bits,
            allocator,
            d_eff_int,
            b_high: 0,
            b_low: 0,
            semantic_quantizer: None,
            tail_quantizer: None,
            per_dim_semantic_quantizers: None,
            semantic_bits_per_dim: None,
            waterfill_allocation: None,
            is_fitted: false,
        }
    }

    /// Fit Lloyd-Max codebooks from rotated data.
    pub fn fit(&mut self, rotated_data: &[Vec<f32>], d_eff: Option<f32>) -> &Self {
        // Update d_eff if provided
        if let Some(de) = d_eff {
            self.d_eff_int = (de.ceil() as usize).max(1).min(self.head_dim);
        }

        // Step 1: Bit allocation
        let (b_high, b_low) =
            self.allocator
                .allocate(self.d_eff_int as f32, self.avg_bits, self.head_dim);
        self.b_high = b_high;
        self.b_low = b_low;

        // Step 2: Water-fill (if v2 path)
        if self.use_water_fill && b_high > 0 {
            let first_d_eff_ev: Vec<f64> = self
                .eigenvalues
                .iter()
                .take(self.d_eff_int)
                .map(|&x| x as f64)
                .collect();
            let total_semantic = b_high as usize * self.d_eff_int;
            let bits_per_dim = waterfill_bits(
                &first_d_eff_ev,
                total_semantic,
                self.wf_min_bits,
                self.wf_max_bits,
            );
            self.semantic_bits_per_dim = Some(bits_per_dim.to_vec());
            self.waterfill_allocation = Some(WaterfillAllocation {
                use_water_fill: true,
                eigenvalues: self
                    .eigenvalues
                    .iter()
                    .take(self.d_eff_int)
                    .copied()
                    .collect(),
                bits_per_dim: bits_per_dim.to_vec(),
                d_eff: self.d_eff_int,
                total_semantic_bits: total_semantic,
                min_bits: self.wf_min_bits,
                max_bits: self.wf_max_bits,
                formula_version: 2,
            });
        }

        // Step 3: Fit Lloyd-Max codebooks
        if rotated_data.is_empty() {
            self.is_fitted = true;
            return self;
        }

        // Collect semantic and tail data
        let mut semantic_data = Vec::new();
        let mut tail_data = Vec::new();
        for sample in rotated_data {
            for (i, &v) in sample.iter().enumerate() {
                if i < self.d_eff_int {
                    semantic_data.push(v);
                } else {
                    tail_data.push(v);
                }
            }
        }

        // Tail quantizer (shared, both paths)
        let mut tail_q =
            LloydMaxQuantizer::new(b_low.max(1), self.max_lloyd_iter, self.seed.wrapping_add(1));
        tail_q.fit(&tail_data);
        self.tail_quantizer = Some(tail_q);

        if self.use_water_fill {
            // v2: per-dim semantic quantizers
            let mut per_dim = Vec::with_capacity(self.d_eff_int);
            let bits = self.semantic_bits_per_dim.as_ref().unwrap();
            for dim in 0..self.d_eff_int {
                let dim_data: Vec<f32> = rotated_data.iter().map(|s| s[dim]).collect();
                let bits_for_dim = bits.get(dim).copied().unwrap_or(b_high).max(1);
                let mut q = LloydMaxQuantizer::new(
                    bits_for_dim,
                    self.max_lloyd_iter,
                    self.seed.wrapping_add((dim + 10) as u64),
                );
                q.fit(&dim_data);
                per_dim.push(q);
            }
            self.per_dim_semantic_quantizers = Some(per_dim);
        } else {
            // v1: single shared semantic quantizer
            let mut sem_q = LloydMaxQuantizer::new(b_high.max(1), self.max_lloyd_iter, self.seed);
            sem_q.fit(&semantic_data);
            self.semantic_quantizer = Some(sem_q);
        }

        self.is_fitted = true;
        self
    }

    /// Compress a pre-rotated vector.
    ///
    /// # Panics
    ///
    /// Panics if not fitted.
    pub fn compress(&self, x: &[f32]) -> CompressedVector {
        assert!(self.is_fitted, "not fitted");
        assert_eq!(x.len(), self.head_dim, "dimension mismatch");

        let mut semantic_indices = Vec::with_capacity(self.d_eff_int);
        let mut tail_indices = Vec::with_capacity(self.head_dim.saturating_sub(self.d_eff_int));

        if self.use_water_fill {
            let quantizers = self.per_dim_semantic_quantizers.as_ref().unwrap();
            for (i, q) in quantizers.iter().enumerate() {
                let idx = q.quantize(&[x[i]])[0];
                semantic_indices.push(idx);
            }
        } else {
            let q = self.semantic_quantizer.as_ref().unwrap();
            for &v in x.iter().take(self.d_eff_int) {
                semantic_indices.push(q.quantize(&[v])[0]);
            }
        }

        let tail_q = self.tail_quantizer.as_ref().unwrap();
        for &v in x.iter().skip(self.d_eff_int) {
            tail_indices.push(tail_q.quantize(&[v])[0]);
        }

        let actual_bits = if self.use_water_fill {
            let bits = self.semantic_bits_per_dim.as_ref().unwrap();
            bits.iter().map(|&b| b as f64).sum::<f64>()
                + (self.head_dim - self.d_eff_int) as f64 * self.b_low as f64
        } else {
            self.d_eff_int as f64 * self.b_high as f64
                + (self.head_dim - self.d_eff_int) as f64 * self.b_low as f64
        };

        CompressedVector {
            semantic_indices,
            tail_indices,
            d_eff: self.d_eff_int,
            head_dim: self.head_dim,
            b_high: self.b_high,
            b_low: self.b_low,
            semantic_bits_per_dim: self.semantic_bits_per_dim.clone(),
            actual_bits_used: actual_bits,
            mse: 0.0,
        }
    }

    /// Decompress a compressed vector back to approximate original.
    ///
    /// # Panics
    ///
    /// Panics if not fitted.
    pub fn decompress(&self, compressed: &CompressedVector) -> Vec<f32> {
        assert!(self.is_fitted, "not fitted");

        let mut result = vec![0.0f32; self.head_dim];

        if self.use_water_fill {
            let quantizers = self.per_dim_semantic_quantizers.as_ref().unwrap();
            for (i, q) in quantizers.iter().enumerate() {
                if i < compressed.semantic_indices.len() {
                    result[i] = q.dequantize(&[compressed.semantic_indices[i]])[0];
                }
            }
        } else {
            let q = self.semantic_quantizer.as_ref().unwrap();
            for (i, &idx) in compressed.semantic_indices.iter().enumerate() {
                result[i] = q.dequantize(&[idx])[0];
            }
        }

        let tail_q = self.tail_quantizer.as_ref().unwrap();
        for (i, &idx) in compressed.tail_indices.iter().enumerate() {
            result[self.d_eff_int + i] = tail_q.dequantize(&[idx])[0];
        }

        result
    }

    /// Compression ratio: original_bits / actual_bits.
    pub fn compression_ratio(&self) -> f32 {
        let original = self.head_dim as f32 * 32.0; // f32 = 32 bits
        let used = self.d_eff_int as f32 * self.b_high as f32
            + (self.head_dim - self.d_eff_int) as f32 * self.b_low as f32;
        if used < 1.0 {
            return 1.0;
        }
        original / used
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_eigenvalues(dim: usize) -> Vec<f32> {
        // Exponential decay eigenvalues
        (0..dim).map(|i| 10.0f32 * 0.8f32.powi(i as i32)).collect()
    }

    fn make_rotated_data(n: usize, head_dim: usize, d_eff: usize) -> Vec<Vec<f32>> {
        let mut rng = katgpt_core::types::Rng::new(42);
        (0..n)
            .map(|_| {
                (0..head_dim)
                    .map(|i| {
                        let scale = if i < d_eff { 1.0 } else { 0.1 };
                        rng.normal() * scale
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_nonuniform_v1_roundtrip() {
        let head_dim = 32;
        let eigenvalues = make_test_eigenvalues(head_dim);
        let data = make_rotated_data(200, head_dim, 6);

        let mut q = NonUniformQuantizer::new(eigenvalues, 3.0, head_dim, 30, 42, false, 1, None);
        q.fit(&data, None);

        for sample in &data[..5] {
            let compressed = q.compress(sample);
            let decompressed = q.decompress(&compressed);
            assert_eq!(decompressed.len(), head_dim);
        }
    }

    #[test]
    fn test_nonuniform_v2_roundtrip() {
        let head_dim = 32;
        let eigenvalues = make_test_eigenvalues(head_dim);
        let data = make_rotated_data(200, head_dim, 6);

        let mut q = NonUniformQuantizer::new(eigenvalues, 3.0, head_dim, 30, 42, true, 2, Some(6));
        q.fit(&data, None);

        for sample in &data[..5] {
            let compressed = q.compress(sample);
            let decompressed = q.decompress(&compressed);
            assert_eq!(decompressed.len(), head_dim);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let head_dim = 128;
        let eigenvalues = make_test_eigenvalues(head_dim);
        let mut q = NonUniformQuantizer::new(eigenvalues, 3.0, head_dim, 10, 42, false, 1, None);
        // Before fitting, ratio is based on b_high=0, b_low=0 — degenerate
        // After fitting:
        let data = make_rotated_data(50, head_dim, 6);
        q.fit(&data, None);
        let ratio = q.compression_ratio();
        // avg 3 bits/coord → ~32/3 ≈ 10.67× compression
        assert!(
            ratio > 5.0 && ratio < 20.0,
            "compression ratio should be ~10x, got {ratio}"
        );
    }
}
