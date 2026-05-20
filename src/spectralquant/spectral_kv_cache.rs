//! SpectralQuant KV cache with per-dim variable-bit packing.
//!
//! Stores K and V tensors in compressed format:
//! - Semantic dimensions (top d_eff): per-dim variable-bit codebooks
//! - Tail dimensions: uniform b_low-bit codebook
//!
//! Each KV vector: normalize → rotate → quantize → variable-bit pack.
//! Reconstruction: unpack → dequantize → inverse rotate → rescale.

use super::spectral::{
    BitAllocator, LloydMaxQuantizer, generate_selective_qjl_signs, waterfill_bits,
};
use super::spectral_rotation::SpectralRotation;
use super::types::{
    LloydMaxCodebook, SpectralQuantCalibration, SpectralQuantKVCacheConfig, SpectralQuantLayer,
    WaterfillAllocation,
};
use crate::simd::simd_scale_inplace;

/// Compressed KV cache using SpectralQuant quantization.
///
/// Two-regime storage:
/// - Semantic (first d_eff dims after rotation): variable-bit packed indices
/// - Tail (remaining dims): uniform low-bit packed indices
///
/// Zero-alloc hot path via scratch buffers.
pub struct SpectralQuantKVCache {
    /// Per-layer calibration + codebooks.
    pub layers: Vec<SpectralQuantLayer>,
    /// Packed key indices: [layer][position] → variable-bit packed bytes.
    key_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position key norms.
    key_norms: Vec<Vec<f32>>,
    /// Packed value indices.
    val_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position value norms.
    val_norms: Vec<Vec<f32>>,
    /// Current write position.
    pos: usize,
    n_layers: usize,
    kv_dim: usize,
    max_seq_len: usize,
    // ── Scratch buffers (zero-alloc hot path) ──
    scratch_normalized: Vec<f32>,
    scratch_rotated: Vec<f32>,
    scratch_unrotated: Vec<f32>,
    scratch_semantic_indices: Vec<u8>,
    scratch_tail_indices: Vec<u8>,
    scratch_all_indices: Vec<u8>,
    scratch_all_bits: Vec<u8>,
}

impl SpectralQuantKVCache {
    /// Create from pre-computed calibration data and config.
    ///
    /// Fits Lloyd-Max codebooks by generating synthetic rotated data from the
    /// eigenvalue distribution: in the spectral basis, dimension `i` has variance `λ_i`,
    /// so we sample `N(0, λ_i)` to build representative codebooks.
    pub fn from_calibration(
        config: &SpectralQuantKVCacheConfig,
        key_calibrations: &[SpectralQuantCalibration],
        val_calibrations: &[SpectralQuantCalibration],
    ) -> Self {
        let n_layers = config.n_layers;
        let kv_dim = config.kv_dim;
        let max_seq_len = config.max_seq_len;

        let mut layers: Vec<SpectralQuantLayer> =
            key_calibrations
                .iter()
                .zip(val_calibrations.iter())
                .enumerate()
                .map(|(layer_idx, (key_cal, _val_cal))| {
                    let d_eff = (key_cal.d_eff.ceil() as usize).max(1).min(kv_dim);
                    let head_dim = key_cal.head_dim;

                    let allocator = BitAllocator::new(config.min_tail_bits, config.max_bits);
                    let (b_high, b_low) =
                        allocator.allocate(key_cal.d_eff, config.avg_bits, head_dim);

                    let qjl_signs = generate_selective_qjl_signs(
                        config.qjl_dim,
                        d_eff,
                        config.seed.wrapping_add(layer_idx as u64 * 100),
                    );

                    let (
                        semantic_bits_per_dim,
                        per_dim_codebooks,
                        semantic_codebook,
                        _waterfill_alloc,
                    ) = if config.use_water_fill && b_high > 0 {
                        let first_ev: Vec<f64> = key_cal
                            .eigenvalues
                            .iter()
                            .take(d_eff)
                            .map(|&x| x as f64)
                            .collect();
                        let total_semantic = b_high as usize * d_eff;
                        let bits = waterfill_bits(
                            &first_ev,
                            total_semantic,
                            config.wf_min_bits,
                            Some(config.wf_max_bits),
                        );
                        let bits_u8 = bits.to_vec();
                        let alloc = WaterfillAllocation {
                            use_water_fill: true,
                            eigenvalues: key_cal.eigenvalues.iter().take(d_eff).copied().collect(),
                            bits_per_dim: bits_u8.clone(),
                            d_eff,
                            total_semantic_bits: total_semantic,
                            min_bits: config.wf_min_bits,
                            max_bits: Some(config.wf_max_bits),
                            formula_version: 2,
                        };
                        let codebooks: Vec<LloydMaxCodebook> = bits_u8
                            .iter()
                            .map(|&b| LloydMaxCodebook {
                                centroids: vec![0.0; 1 << b],
                                n_bits: b,
                            })
                            .collect();
                        (Some(bits_u8), Some(codebooks), None, Some(alloc))
                    } else {
                        let codebook = LloydMaxCodebook {
                            centroids: vec![0.0; 1 << b_high],
                            n_bits: b_high,
                        };
                        (None, None, Some(codebook), None)
                    };

                    let tail_codebook = LloydMaxCodebook {
                        centroids: vec![0.0; 1 << b_low.max(1)],
                        n_bits: b_low.max(1),
                    };

                    SpectralQuantLayer {
                        calibration: key_cal.clone(),
                        qjl_signs,
                        d_eff,
                        b_high,
                        b_low,
                        semantic_bits_per_dim,
                        per_dim_semantic_codebooks: per_dim_codebooks,
                        semantic_codebook,
                        tail_codebook,
                    }
                })
                .collect();

        // Fit codebooks from eigenvalue distribution.
        // In the spectral (rotated) domain, dim i has variance λ_i.
        // Generate synthetic samples and fit Lloyd-Max centroids.
        let n_synthetic = config.calibration_samples.max(64);
        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            let eigenvalues = &layer.calibration.eigenvalues;
            let d_eff = layer.d_eff;
            let head_dim = layer.calibration.head_dim;
            let b_high = layer.b_high;
            let b_low = layer.b_low;
            let mut rng = crate::types::Rng::new(config.seed.wrapping_add(layer_idx as u64 * 31));

            // Generate synthetic rotated data: rotated[i] ~ N(0, λ_i)
            let synthetic_rotated: Vec<Vec<f32>> = (0..n_synthetic)
                .map(|_| {
                    (0..head_dim)
                        .map(|i| {
                            let var = eigenvalues.get(i).copied().unwrap_or(1e-4).max(1e-8);
                            rng.normal() * var.sqrt()
                        })
                        .collect()
                })
                .collect();

            // Fit tail codebook (shared across all tail dims)
            let tail_data: Vec<f32> = synthetic_rotated
                .iter()
                .flat_map(|s| s.iter().skip(d_eff).copied())
                .collect();
            let mut tail_q = LloydMaxQuantizer::new(
                b_low.max(1),
                config.lloyd_max_iter,
                config.seed.wrapping_add(layer_idx as u64 * 51 + 1),
            );
            tail_q.fit(&tail_data);
            layer.tail_codebook.centroids = tail_q.centroids().to_vec();

            if let Some(ref mut cb) = layer.semantic_codebook {
                // v1: shared semantic codebook — collect all semantic dims
                let semantic_data: Vec<f32> = synthetic_rotated
                    .iter()
                    .flat_map(|s| s.iter().take(d_eff).copied())
                    .collect();
                let mut sem_q = LloydMaxQuantizer::new(
                    b_high.max(1),
                    config.lloyd_max_iter,
                    config.seed.wrapping_add(layer_idx as u64 * 51 + 2),
                );
                sem_q.fit(&semantic_data);
                cb.centroids = sem_q.centroids().to_vec();
            } else if let Some(ref mut per_dim) = layer.per_dim_semantic_codebooks {
                // v2: per-dim semantic codebooks
                let bits = layer.semantic_bits_per_dim.as_ref();
                for (dim, cb) in per_dim.iter_mut().enumerate() {
                    let dim_data: Vec<f32> = synthetic_rotated.iter().map(|s| s[dim]).collect();
                    let bits_for_dim = bits
                        .and_then(|b| b.get(dim).copied())
                        .unwrap_or(b_high)
                        .max(1);
                    let mut q = LloydMaxQuantizer::new(
                        bits_for_dim,
                        config.lloyd_max_iter,
                        config.seed.wrapping_add((dim + 10) as u64),
                    );
                    q.fit(&dim_data);
                    cb.centroids = q.centroids().to_vec();
                }
            }
        }

        // Conservative packed size: 1 byte per dim covers all variable-bit layouts
        let max_packed = kv_dim;

        Self {
            layers,
            key_indices: vec![vec![vec![0u8; max_packed]; max_seq_len]; n_layers],
            key_norms: vec![vec![0.0f32; max_seq_len]; n_layers],
            val_indices: vec![vec![vec![0u8; max_packed]; max_seq_len]; n_layers],
            val_norms: vec![vec![0.0f32; max_seq_len]; n_layers],
            pos: 0,
            n_layers,
            kv_dim,
            max_seq_len,
            scratch_normalized: vec![0.0f32; kv_dim],
            scratch_rotated: vec![0.0f32; kv_dim],
            scratch_unrotated: vec![0.0f32; kv_dim],
            scratch_semantic_indices: vec![0u8; kv_dim],
            scratch_tail_indices: vec![0u8; kv_dim],
            scratch_all_indices: vec![0u8; kv_dim],
            scratch_all_bits: vec![0u8; kv_dim],
        }
    }

    /// Quantize and store a key vector at given layer and position.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let d_eff = layer_state.d_eff;
        let b_low = layer_state.b_low.max(1);

        // Compute norm
        let norm = simd_norm(key);
        if norm < 1e-8 {
            self.key_norms[layer][pos] = 0.0;
            return;
        }
        self.key_norms[layer][pos] = norm;

        // Normalize into scratch buffer
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..key.len()].copy_from_slice(key);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // Rotate using eigenvectors
        let rotation = SpectralRotation::new(
            layer_state.calibration.eigenvectors.clone(),
            layer_state.calibration.head_dim,
        );
        rotation.rotate(&self.scratch_normalized, &mut self.scratch_rotated);

        // Quantize semantic dims
        if let Some(cb) = &layer_state.semantic_codebook {
            // v1: shared semantic codebook
            for i in 0..d_eff {
                self.scratch_semantic_indices[i] =
                    quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            // v2: per-dim codebooks
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                self.scratch_semantic_indices[i] =
                    quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
            }
        }

        // Quantize tail dims
        let tail_cb = &layer_state.tail_codebook;
        for (i, &v) in self.scratch_rotated.iter().enumerate().skip(d_eff) {
            self.scratch_tail_indices[i - d_eff] = quantize_to_idx(v, &tail_cb.centroids);
        }

        // Build combined bits-per-dim array
        let all_bits = &mut self.scratch_all_bits;
        if let Some(ref bits) = layer_state.semantic_bits_per_dim {
            all_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
        } else {
            all_bits[..d_eff].fill(layer_state.b_high);
        }
        all_bits[d_eff..self.kv_dim].fill(b_low);

        // Build combined indices array
        let all_indices = &mut self.scratch_all_indices;
        all_indices[..d_eff].copy_from_slice(&self.scratch_semantic_indices[..d_eff]);
        let tail_len = self.kv_dim - d_eff;
        all_indices[d_eff..self.kv_dim].copy_from_slice(&self.scratch_tail_indices[..tail_len]);

        // Pack variable bits into storage
        pack_variable_bits(
            &all_indices[..self.kv_dim],
            &all_bits[..self.kv_dim],
            &mut self.key_indices[layer][pos],
        );
    }

    /// Quantize and store a value vector at given layer and position.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let d_eff = layer_state.d_eff;
        let b_low = layer_state.b_low.max(1);

        let norm = simd_norm(value);
        if norm < 1e-8 {
            self.val_norms[layer][pos] = 0.0;
            return;
        }
        self.val_norms[layer][pos] = norm;

        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..value.len()].copy_from_slice(value);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        let rotation = SpectralRotation::new(
            layer_state.calibration.eigenvectors.clone(),
            layer_state.calibration.head_dim,
        );
        rotation.rotate(&self.scratch_normalized, &mut self.scratch_rotated);

        if let Some(cb) = &layer_state.semantic_codebook {
            for i in 0..d_eff {
                self.scratch_semantic_indices[i] =
                    quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                self.scratch_semantic_indices[i] =
                    quantize_to_idx(self.scratch_rotated[i], &cb.centroids);
            }
        }

        let tail_cb = &layer_state.tail_codebook;
        for (i, &v) in self.scratch_rotated.iter().enumerate().skip(d_eff) {
            self.scratch_tail_indices[i - d_eff] = quantize_to_idx(v, &tail_cb.centroids);
        }

        let all_bits = &mut self.scratch_all_bits;
        if let Some(ref bits) = layer_state.semantic_bits_per_dim {
            all_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
        } else {
            all_bits[..d_eff].fill(layer_state.b_high);
        }
        all_bits[d_eff..self.kv_dim].fill(b_low);

        let all_indices = &mut self.scratch_all_indices;
        all_indices[..d_eff].copy_from_slice(&self.scratch_semantic_indices[..d_eff]);
        let tail_len = self.kv_dim - d_eff;
        all_indices[d_eff..self.kv_dim].copy_from_slice(&self.scratch_tail_indices[..tail_len]);

        pack_variable_bits(
            &all_indices[..self.kv_dim],
            &all_bits[..self.kv_dim],
            &mut self.val_indices[layer][pos],
        );
    }

    /// Dequantize a key at position into a new vector.
    pub fn dequantize_key(&self, layer: usize, pos: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.kv_dim];
        // We need &mut self for scratch buffers, so use a temporary approach
        // by reconstructing directly without scratch
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[layer][pos];
        if norm < 1e-8 {
            return out;
        }

        let d_eff = layer_state.d_eff;
        let b_low = layer_state.b_low.max(1);

        // Build bits array
        let mut all_bits = vec![0u8; self.kv_dim];
        if let Some(ref bits) = layer_state.semantic_bits_per_dim {
            all_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
        } else {
            all_bits[..d_eff].fill(layer_state.b_high);
        }
        all_bits[d_eff..self.kv_dim].fill(b_low);

        let mut all_indices = vec![0u8; self.kv_dim];
        unpack_variable_bits(
            &self.key_indices[layer][pos],
            &all_bits,
            self.kv_dim,
            &mut all_indices,
        );

        let mut rotated = vec![0.0f32; self.kv_dim];
        if let Some(cb) = &layer_state.semantic_codebook {
            for i in 0..d_eff {
                rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            let limit = d_eff.min(per_dim.len());
            for i in 0..limit {
                rotated[i] = dequantize_idx(all_indices[i], &per_dim[i].centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for i in d_eff..self.kv_dim {
            rotated[i] = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let rotation = SpectralRotation::new(
            layer_state.calibration.eigenvectors.clone(),
            layer_state.calibration.head_dim,
        );
        let mut normalized = vec![0.0f32; self.kv_dim];
        rotation.unrotate(&rotated, &mut normalized);

        for v in &mut normalized {
            *v *= norm;
        }
        out.copy_from_slice(&normalized);
        out
    }

    /// Dequantize a value at position into a new vector.
    pub fn dequantize_value(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[layer][pos];
        if norm < 1e-8 {
            return vec![0.0f32; self.kv_dim];
        }

        let d_eff = layer_state.d_eff;
        let b_low = layer_state.b_low.max(1);

        let mut all_bits = vec![0u8; self.kv_dim];
        if let Some(ref bits) = layer_state.semantic_bits_per_dim {
            all_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
        } else {
            all_bits[..d_eff].fill(layer_state.b_high);
        }
        all_bits[d_eff..self.kv_dim].fill(b_low);

        let mut all_indices = vec![0u8; self.kv_dim];
        unpack_variable_bits(
            &self.val_indices[layer][pos],
            &all_bits,
            self.kv_dim,
            &mut all_indices,
        );

        let mut rotated = vec![0.0f32; self.kv_dim];
        if let Some(cb) = &layer_state.semantic_codebook {
            for i in 0..d_eff {
                rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            let limit = d_eff.min(per_dim.len());
            for i in 0..limit {
                rotated[i] = dequantize_idx(all_indices[i], &per_dim[i].centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for i in d_eff..self.kv_dim {
            rotated[i] = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let rotation = SpectralRotation::new(
            layer_state.calibration.eigenvectors.clone(),
            layer_state.calibration.head_dim,
        );
        let mut normalized = vec![0.0f32; self.kv_dim];
        rotation.unrotate(&rotated, &mut normalized);

        normalized.iter().map(|x| x * norm).collect()
    }

    /// Dequantize key into pre-allocated buffer. Zero-alloc hot path.
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    /// Reconstruction: unpack → dequantize → inverse rotate → scale by norm.
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let d_eff = layer_state.d_eff;
        let b_low = layer_state.b_low.max(1);

        // Build bits array in scratch
        let all_bits = &mut self.scratch_all_bits;
        if let Some(ref bits) = layer_state.semantic_bits_per_dim {
            all_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
        } else {
            all_bits[..d_eff].fill(layer_state.b_high);
        }
        all_bits[d_eff..self.kv_dim].fill(b_low);

        // Unpack variable bits into scratch
        let all_indices = &mut self.scratch_all_indices;
        unpack_variable_bits(
            &self.key_indices[layer][pos],
            &all_bits[..self.kv_dim],
            self.kv_dim,
            all_indices,
        );

        // Dequantize into scratch_rotated
        if let Some(cb) = &layer_state.semantic_codebook {
            for (i, c) in self.scratch_rotated.iter_mut().enumerate().take(d_eff) {
                *c = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                self.scratch_rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for (i, r) in self.scratch_rotated.iter_mut().enumerate().skip(d_eff) {
            *r = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        // Inverse rotate
        let rotation = SpectralRotation::new(
            layer_state.calibration.eigenvectors.clone(),
            layer_state.calibration.head_dim,
        );
        rotation.unrotate(&self.scratch_rotated, &mut self.scratch_unrotated);

        // Scale by norm → output
        out.copy_from_slice(&self.scratch_unrotated);
        simd_scale_inplace(out, norm);
    }

    /// Dequantize value into pre-allocated buffer. Zero-alloc hot path.
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    /// Reconstruction: unpack → dequantize → inverse rotate → scale by norm.
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let d_eff = layer_state.d_eff;
        let b_low = layer_state.b_low.max(1);

        let all_bits = &mut self.scratch_all_bits;
        if let Some(ref bits) = layer_state.semantic_bits_per_dim {
            all_bits[..d_eff].copy_from_slice(&bits[..d_eff.min(bits.len())]);
        } else {
            all_bits[..d_eff].fill(layer_state.b_high);
        }
        all_bits[d_eff..self.kv_dim].fill(b_low);

        let all_indices = &mut self.scratch_all_indices;
        unpack_variable_bits(
            &self.val_indices[layer][pos],
            &all_bits[..self.kv_dim],
            self.kv_dim,
            all_indices,
        );

        if let Some(cb) = &layer_state.semantic_codebook {
            for (i, r) in self.scratch_rotated.iter_mut().enumerate().take(d_eff) {
                *r = dequantize_idx(all_indices[i], &cb.centroids);
            }
        } else if let Some(per_dim) = &layer_state.per_dim_semantic_codebooks {
            for (i, cb) in per_dim.iter().enumerate().take(d_eff) {
                self.scratch_rotated[i] = dequantize_idx(all_indices[i], &cb.centroids);
            }
        }
        let tail_cb = &layer_state.tail_codebook;
        for (i, r) in self.scratch_rotated.iter_mut().enumerate().skip(d_eff) {
            *r = dequantize_idx(all_indices[i], &tail_cb.centroids);
        }

        let rotation = SpectralRotation::new(
            layer_state.calibration.eigenvectors.clone(),
            layer_state.calibration.head_dim,
        );
        rotation.unrotate(&self.scratch_rotated, &mut self.scratch_unrotated);

        out.copy_from_slice(&self.scratch_unrotated);
        simd_scale_inplace(out, norm);
    }

    /// Reset cache for a new sequence.
    pub fn reset(&mut self) {
        self.pos = 0;
        for layer in 0..self.n_layers {
            for p in 0..self.max_seq_len {
                self.key_indices[layer][p].fill(0);
                self.key_norms[layer][p] = 0.0;
                self.val_indices[layer][p].fill(0);
                self.val_norms[layer][p] = 0.0;
            }
        }
    }

    /// Current write position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set the current write position.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// KV dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Compression ratio vs f32 uncompressed (32 bits per coordinate).
    pub fn compression_ratio(&self) -> f32 {
        if self.layers.is_empty() {
            return 1.0;
        }
        let original = self.kv_dim as f32 * 32.0;
        let layer0 = &self.layers[0];
        let used = layer0.d_eff as f32 * layer0.b_high as f32
            + (self.kv_dim - layer0.d_eff) as f32 * layer0.b_low.max(1) as f32;
        if used < 1.0 {
            return 1.0;
        }
        original / used
    }
}

impl crate::types::QuantizedKVCache for SpectralQuantKVCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        self.store_key(layer, pos, key);
    }

    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        self.store_value(layer, pos, value);
    }

    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_key_into(layer, pos, out);
    }

    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        self.dequantize_value_into(layer, pos, out);
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn pos(&self) -> usize {
        self.pos()
    }

    fn set_pos(&mut self, pos: usize) {
        self.set_pos(pos);
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Compute L2 norm of a vector.
fn simd_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Find nearest centroid index for a value.
fn quantize_to_idx(value: f32, centroids: &[f32]) -> u8 {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (value - *a)
                .abs()
                .partial_cmp(&(value - *b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0) as u8
}

/// Dequantize an index back to centroid value.
fn dequantize_idx(idx: u8, centroids: &[f32]) -> f32 {
    centroids.get(idx as usize).copied().unwrap_or(0.0)
}

/// Pack variable-bit indices into bytes.
///
/// Each index uses `bits_per_dim[i]` bits. Output is written LSB-first.
fn pack_variable_bits(indices: &[u8], bits_per_dim: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0u32;

    for (i, &idx) in indices.iter().enumerate() {
        let bits = bits_per_dim.get(i).copied().unwrap_or(1) as u32;
        bit_buffer |= (idx as u64) << bits_in_buffer;
        bits_in_buffer += bits;

        while bits_in_buffer >= 8 {
            out.push((bit_buffer & 0xFF) as u8);
            bit_buffer >>= 8;
            bits_in_buffer -= 8;
        }
    }
    if bits_in_buffer > 0 {
        out.push((bit_buffer & 0xFF) as u8);
    }
}

/// Unpack variable-bit indices from bytes.
///
/// Reads LSB-first, each dim consumes `bits_per_dim[i]` bits.
fn unpack_variable_bits(packed: &[u8], bits_per_dim: &[u8], n_dims: usize, out: &mut [u8]) {
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0u32;
    let mut byte_idx = 0;

    for (i, o) in out.iter_mut().enumerate().take(n_dims) {
        let bits = bits_per_dim.get(i).copied().unwrap_or(1) as u32;
        while bits_in_buffer < bits && byte_idx < packed.len() {
            bit_buffer |= (packed[byte_idx] as u64) << bits_in_buffer;
            bits_in_buffer += 8;
            byte_idx += 1;
        }
        let mask = (1u64 << bits) - 1;
        *o = (bit_buffer & mask) as u8;
        bit_buffer >>= bits;
        bits_in_buffer -= bits;
    }
}

#[cfg(test)]
mod tests {
    use super::super::spectral::participation_ratio;
    use super::*;

    fn make_test_calibration(head_dim: usize) -> SpectralQuantCalibration {
        let mut eigenvectors = vec![0.0f32; head_dim * head_dim];
        for i in 0..head_dim {
            eigenvectors[i * head_dim + i] = 1.0;
        }
        let eigenvalues: Vec<f32> = (0..head_dim)
            .map(|i| 10.0 * 0.8f32.powi(i as i32))
            .collect();
        let d_eff = participation_ratio(&eigenvalues);
        SpectralQuantCalibration {
            eigenvectors,
            eigenvalues,
            d_eff,
            spectral_gap: None,
            var_95: 10,
            var_99: 20,
            n_samples: 100,
            head_dim,
        }
    }

    fn make_test_config(
        n_layers: usize,
        kv_dim: usize,
        max_seq_len: usize,
    ) -> SpectralQuantKVCacheConfig {
        SpectralQuantKVCacheConfig {
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
            n_layers,
            kv_dim,
            max_seq_len,
        }
    }

    /// No-op: codebooks are now fitted during `from_calibration()`.
    #[allow(dead_code)]
    fn init_test_centroids(_cache: &mut SpectralQuantKVCache) {
        // Codebooks are already fitted from eigenvalue distribution in from_calibration().
    }

    #[test]
    fn test_pack_unpack_roundtrip_2bit() {
        let indices = vec![0u8, 1, 2, 3, 0, 2, 1, 3];
        let bits = vec![2u8; 8];
        let mut packed = Vec::new();
        pack_variable_bits(&indices, &bits, &mut packed);
        let mut unpacked = vec![0u8; 8];
        unpack_variable_bits(&packed, &bits, 8, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_variable() {
        let indices = vec![3u8, 7, 15, 1, 0];
        let bits = vec![2u8, 3, 4, 1, 1];
        let mut packed = Vec::new();
        pack_variable_bits(&indices, &bits, &mut packed);
        let mut unpacked = vec![0u8; 5];
        unpack_variable_bits(&packed, &bits, 5, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_pack_unpack_roundtrip_1bit() {
        let indices = vec![1u8, 0, 1, 1, 0, 0, 1, 0];
        let bits = vec![1u8; 8];
        let mut packed = Vec::new();
        pack_variable_bits(&indices, &bits, &mut packed);
        let mut unpacked = vec![0u8; 8];
        unpack_variable_bits(&packed, &bits, 8, &mut unpacked);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn test_kv_cache_store_dequantize() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let key: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).sin())
            .collect();
        cache.store_key(0, 0, &key);

        let mut recovered = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut recovered);

        let orig_norm: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rec_norm: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            rec_norm > 0.1,
            "reconstructed norm too small: {rec_norm}"
        );
        // 3-bit quantization with synthetic codebooks: allow up to 2x norm distortion
        assert!(
            (orig_norm - rec_norm).abs() / orig_norm < 1.0,
            "norm changed too much: {orig_norm} -> {rec_norm}"
        );
    }

    #[test]
    fn test_kv_cache_value_roundtrip() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let value: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).cos())
            .collect();
        cache.store_value(0, 0, &value);

        let mut recovered = vec![0.0f32; kv_dim];
        cache.dequantize_value_into(0, 0, &mut recovered);

        let orig_norm: f32 = value.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rec_norm: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            rec_norm > 0.1,
            "reconstructed norm too small: {rec_norm}"
        );
        assert!(
            (orig_norm - rec_norm).abs() / orig_norm < 1.0,
            "norm changed too much: {orig_norm} -> {rec_norm}"
        );
    }

    #[test]
    fn test_compression_ratio() {
        let kv_dim = 128;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 64);
        let cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let ratio = cache.compression_ratio();
        assert!(
            ratio > 5.0 && ratio < 20.0,
            "compression ratio should be ~10x, got {ratio}"
        );
    }

    #[test]
    fn test_reset_clears() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        let key: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0).sin()).collect();
        cache.store_key(0, 0, &key);
        assert!(cache.key_norms[0][0] > 0.0);

        cache.reset();
        assert_eq!(cache.pos(), 0);
        assert_eq!(cache.key_norms[0][0], 0.0);
        assert_eq!(cache.val_norms[0][0], 0.0);
    }

    #[test]
    fn test_zero_vector_handling() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        let zero_key = vec![0.0f32; kv_dim];
        cache.store_key(0, 0, &zero_key);
        assert_eq!(cache.key_norms[0][0], 0.0);

        let mut recovered = vec![1.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut recovered);
        assert!(recovered.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_multi_position() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        for pos in 0..4 {
            let key: Vec<f32> = (0..kv_dim)
                .map(|i| cal.eigenvalues[i].sqrt() * ((i + pos) as f32 + 1.0).sin())
                .collect();
            cache.store_key(0, pos, &key);
        }

        for pos in 0..4 {
            let original: Vec<f32> = (0..kv_dim)
                .map(|i| cal.eigenvalues[i].sqrt() * ((i + pos) as f32 + 1.0).sin())
                .collect();
            let mut recovered = vec![0.0f32; kv_dim];
            cache.dequantize_key_into(0, pos, &mut recovered);
            let orig_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
            let rec_norm: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                rec_norm > 0.1,
                "pos {pos}: reconstructed norm too small: {rec_norm}"
            );
            assert!(
                (orig_norm - rec_norm).abs() / orig_norm < 1.0,
                "pos {pos}: norm changed too much: {orig_norm} -> {rec_norm}"
            );
        }
    }

    #[test]
    fn test_multi_layer_independence() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(2, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            &[cal.clone(), cal.clone()],
            &[cal.clone(), cal.clone()],
        );
        let key0: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).sin())
            .collect();
        let key1: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 2.0).cos())
            .collect();
        cache.store_key(0, 0, &key0);
        cache.store_key(1, 0, &key1);

        let mut rec0 = vec![0.0f32; kv_dim];
        let mut rec1 = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut rec0);
        cache.dequantize_key_into(1, 0, &mut rec1);

        // Layers should produce different reconstructions from different inputs
        let diff: f32 = rec0
            .iter()
            .zip(rec1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "layers should produce different outputs, diff={diff}"
        );
    }

    #[test]
    fn test_dequantize_key_allocating() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let key: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).sin())
            .collect();
        cache.store_key(0, 0, &key);

        let recovered = cache.dequantize_key(0, 0);
        assert_eq!(recovered.len(), kv_dim);
    }

    #[test]
    fn test_dequantize_value_allocating() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );
        let value: Vec<f32> = (0..kv_dim)
            .map(|i| cal.eigenvalues[i].sqrt() * (i as f32 + 1.0).cos())
            .collect();
        cache.store_value(0, 0, &value);

        let recovered = cache.dequantize_value(0, 0);
        assert_eq!(recovered.len(), kv_dim);
    }

    #[test]
    fn test_set_pos() {
        let kv_dim = 16;
        let cal = make_test_calibration(kv_dim);
        let config = make_test_config(1, kv_dim, 32);
        let mut cache = SpectralQuantKVCache::from_calibration(
            &config,
            std::slice::from_ref(&cal),
            std::slice::from_ref(&cal),
        );

        assert_eq!(cache.pos(), 0);
        cache.set_pos(10);
        assert_eq!(cache.pos(), 10);
    }
}
