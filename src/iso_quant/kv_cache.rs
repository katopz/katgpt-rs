//! IsoQuant KV cache: 4D quaternion rotation + scalar quantization.
//!
//! Pipeline per vector:
//!   normalize → quaternion rotate → quantize → bit-pack
//! Reconstruction:
//!   unpack → dequantize → inverse rotate → rescale
//!
//! Complexity: O(d) vs TurboQuant's O(d²) for rotation.
//! d=128: IsoQuant = 512–1024 FMAs, TurboQuant = 16,384 FMAs.

use super::rotation::{apply_inverse_rotation, apply_rotation, generate_unit_quaternions};
use super::types::{IsoQuantConfig, IsoQuantLayer, IsoQuantMode};
use crate::simd::simd_scale_inplace;
use crate::turboquant::codebook::compute_codebook;
use crate::turboquant::types::TurboQuantCodebook;
use crate::types;

/// Compressed KV cache using IsoQuant 4D quaternion rotation.
///
/// Each KV vector is: normalized → quaternion rotated → quantized → bit-packed.
/// Reconstruction: unpack → dequantize → inverse rotate → rescale.
pub struct IsoQuantKVCache {
    /// Per-layer quaternion state.
    pub layers: Vec<IsoQuantLayer>,
    /// Key codebook (shared across layers, same kv_dim/key_bits).
    key_codebook: TurboQuantCodebook,
    /// Value codebook (shared across layers).
    val_codebook: TurboQuantCodebook,
    /// Bit-packed key indices: layers × positions × packed_coords.
    key_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position key norms.
    key_norms: Vec<Vec<f32>>,
    /// Bit-packed value indices.
    val_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position value norms.
    val_norms: Vec<Vec<f32>>,
    /// Current position.
    pos: usize,
    /// Number of layers.
    n_layers: usize,
    /// KV dimension.
    kv_dim: usize,
    /// Key bits per coordinate.
    key_bits: u8,
    /// Value bits per coordinate.
    val_bits: u8,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Rotation mode: Full (6 DOF) or Fast (3 DOF).
    mode: IsoQuantMode,
    /// Scratch: normalized input [kv_dim]. Reused across store/dequantize calls.
    scratch_normalized: Vec<f32>,
    /// Scratch: rotation output [kv_dim].
    scratch_rotated: Vec<f32>,
    /// Scratch: quantized indices [kv_dim].
    scratch_indices: Vec<u8>,
}

impl IsoQuantKVCache {
    /// Create a new compressed KV cache from config.
    pub fn new(config: &IsoQuantConfig) -> Self {
        let n_groups = config.kv_dim.div_ceil(4);

        let key_codebook = compute_codebook(config.kv_dim, config.key_bits);
        let val_codebook = compute_codebook(config.kv_dim, config.val_bits);

        // Generate per-layer quaternions (different seeds per layer for diversity)
        let layers: Vec<IsoQuantLayer> = (0..config.n_layers)
            .map(|layer_idx| {
                let base_seed = config.seed.wrapping_add(layer_idx as u64 * 1000);
                let key_q_left = generate_unit_quaternions(n_groups, base_seed);
                let val_q_left = generate_unit_quaternions(n_groups, base_seed.wrapping_add(100));

                let (key_q_right, val_q_right) = match config.mode {
                    IsoQuantMode::Full => (
                        Some(generate_unit_quaternions(
                            n_groups,
                            base_seed.wrapping_add(200),
                        )),
                        Some(generate_unit_quaternions(
                            n_groups,
                            base_seed.wrapping_add(300),
                        )),
                    ),
                    IsoQuantMode::Fast => (None, None),
                };

                IsoQuantLayer {
                    key_q_left,
                    key_q_right,
                    val_q_left,
                    val_q_right,
                }
            })
            .collect();

        let packed_key_len = packed_len(config.kv_dim, config.key_bits);
        let packed_val_len = packed_len(config.kv_dim, config.val_bits);

        Self {
            layers,
            key_codebook,
            val_codebook,
            key_indices: vec![vec![vec![0u8; packed_key_len]; config.max_seq_len]; config.n_layers],
            key_norms: vec![vec![0.0f32; config.max_seq_len]; config.n_layers],
            val_indices: vec![vec![vec![0u8; packed_val_len]; config.max_seq_len]; config.n_layers],
            val_norms: vec![vec![0.0f32; config.max_seq_len]; config.n_layers],
            pos: 0,
            n_layers: config.n_layers,
            kv_dim: config.kv_dim,
            key_bits: config.key_bits,
            val_bits: config.val_bits,
            max_seq_len: config.max_seq_len,
            mode: config.mode,
            scratch_normalized: vec![0.0f32; config.kv_dim],
            scratch_rotated: vec![0.0f32; config.kv_dim],
            scratch_indices: vec![0u8; config.kv_dim],
        }
    }

    /// Quantize and store a key vector at given layer and position.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        let layer_state = &self.layers[layer];

        // Compute norm via SIMD (avoids scalar iteration)
        let norm = crate::simd::simd_sum_sq(key, key.len()).sqrt();
        self.key_norms[layer][pos] = norm;

        if norm < 1e-8 {
            return;
        }

        // Normalize into scratch buffer
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..key.len()].copy_from_slice(key);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // Quaternion rotate: normalized → scratch_rotated
        apply_rotation(
            &layer_state.key_q_left,
            layer_state.key_q_right.as_deref(),
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        // Quantize using codebook
        let cb = &self.key_codebook;
        for (i, &v) in self.scratch_rotated.iter().enumerate() {
            unsafe {
                *self.scratch_indices.get_unchecked_mut(i) = cb.quantize(v);
            }
        }

        // Bit-pack into storage
        pack_indices_into(
            &self.scratch_indices,
            self.key_bits,
            &mut self.key_indices[layer][pos],
        );
    }

    /// Quantize and store a value vector at given layer and position.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        let layer_state = &self.layers[layer];

        let norm = crate::simd::simd_sum_sq(value, value.len()).sqrt();
        self.val_norms[layer][pos] = norm;

        if norm < 1e-8 {
            return;
        }

        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..value.len()].copy_from_slice(value);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        apply_rotation(
            &layer_state.val_q_left,
            layer_state.val_q_right.as_deref(),
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        let cb = &self.val_codebook;
        for (i, &v) in self.scratch_rotated.iter().enumerate() {
            unsafe {
                *self.scratch_indices.get_unchecked_mut(i) = cb.quantize(v);
            }
        }

        pack_indices_into(
            &self.scratch_indices,
            self.val_bits,
            &mut self.val_indices[layer][pos],
        );
    }

    /// Dequantize key at position. Returns reconstructed key vector.
    ///
    /// Only available in tests — prefer `dequantize_key_into` for production.
    #[cfg(test)]
    pub fn dequantize_key(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[layer][pos];

        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let indices = unpack_indices(&self.key_indices[layer][pos], self.key_bits, self.kv_dim);
        let centroids = &self.key_codebook.centroids;
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| centroids[i as usize])
            .collect();

        let mut normalized = vec![0.0f32; self.kv_dim];
        apply_inverse_rotation(
            &layer_state.key_q_left,
            layer_state.key_q_right.as_deref(),
            &rotated,
            &mut normalized,
        );

        normalized.iter().map(|x| x * norm).collect()
    }

    /// Dequantize value at position. Returns reconstructed value vector.
    ///
    /// Only available in tests — prefer `dequantize_value_into` for production.
    #[cfg(test)]
    pub fn dequantize_value(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[layer][pos];

        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let indices = unpack_indices(&self.val_indices[layer][pos], self.val_bits, self.kv_dim);
        let centroids = &self.val_codebook.centroids;
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| centroids[i as usize])
            .collect();

        let mut normalized = vec![0.0f32; self.kv_dim];
        apply_inverse_rotation(
            &layer_state.val_q_left,
            layer_state.val_q_right.as_deref(),
            &rotated,
            &mut normalized,
        );

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

        // Unpack into scratch_indices
        unpack_indices_into(
            &self.key_indices[layer][pos],
            self.key_bits,
            self.kv_dim,
            &mut self.scratch_indices,
        );

        // Dequantize into scratch_rotated using shared codebook centroids
        let key_centroids = &self.key_codebook.centroids;
        for (i, &idx) in self.scratch_indices.iter().enumerate() {
            unsafe {
                *self.scratch_rotated.get_unchecked_mut(i) =
                    *key_centroids.get_unchecked(idx as usize);
            }
        }

        // Inverse rotate into scratch_normalized
        apply_inverse_rotation(
            &layer_state.key_q_left,
            layer_state.key_q_right.as_deref(),
            &self.scratch_rotated,
            &mut self.scratch_normalized,
        );

        // Scale by norm → output
        out.copy_from_slice(&self.scratch_normalized);
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

        unpack_indices_into(
            &self.val_indices[layer][pos],
            self.val_bits,
            self.kv_dim,
            &mut self.scratch_indices,
        );

        let val_centroids = &self.val_codebook.centroids;
        for (i, &idx) in self.scratch_indices.iter().enumerate() {
            unsafe {
                *self.scratch_rotated.get_unchecked_mut(i) =
                    *val_centroids.get_unchecked(idx as usize);
            }
        }

        apply_inverse_rotation(
            &layer_state.val_q_left,
            layer_state.val_q_right.as_deref(),
            &self.scratch_rotated,
            &mut self.scratch_normalized,
        );

        out.copy_from_slice(&self.scratch_normalized);
        simd_scale_inplace(out, norm);
    }

    /// Reset cache for new sequence.
    pub fn reset(&mut self) {
        for layer in 0..self.n_layers {
            for pos in 0..self.max_seq_len {
                self.key_indices[layer][pos].fill(0);
                self.key_norms[layer][pos] = 0.0;
                self.val_indices[layer][pos].fill(0);
                self.val_norms[layer][pos] = 0.0;
            }
        }
        self.pos = 0;
    }

    /// Bytes stored per token (K + V, all layers).
    pub fn bytes_per_token(&self) -> usize {
        let packed_key = packed_len(self.kv_dim, self.key_bits);
        let packed_val = packed_len(self.kv_dim, self.val_bits);
        let per_layer = packed_key + packed_val + 8; // +8 for two f32 norms
        per_layer * self.n_layers
    }

    /// Compression ratio vs f32 KV cache.
    pub fn compression_ratio(&self) -> f64 {
        let flat_bytes = self.kv_dim * 4 * 2 * self.n_layers; // f32, K+V
        flat_bytes as f64 / self.bytes_per_token() as f64
    }

    /// Get current position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set current position (for manual position management).
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Get KV dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Get rotation mode.
    pub fn mode(&self) -> IsoQuantMode {
        self.mode
    }
}

impl types::QuantizedKVCache for IsoQuantKVCache {
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

// ── Bit packing ──────────────────────────────────────────────

/// Packed byte length for n values at given bits per value.
fn packed_len(n: usize, bits: u8) -> usize {
    match bits {
        2 => n.div_ceil(4),
        3 | 4 => n.div_ceil(2), // 3-bit stored as 4-bit (2 per u8)
        8 => n,
        _ => (n * bits as usize).div_ceil(8),
    }
}

/// Pack indices into pre-allocated buffer (zero-alloc).
fn pack_indices_into(indices: &[u8], bits: u8, out: &mut [u8]) {
    match bits {
        2 => {
            out.fill(0);
            for (i, &idx) in indices.iter().enumerate() {
                let byte = i / 4;
                let shift = (i % 4) * 2;
                unsafe {
                    *out.get_unchecked_mut(byte) |= (idx & 0x3) << shift;
                }
            }
        }
        3 | 4 => {
            out.fill(0);
            for (i, &idx) in indices.iter().enumerate() {
                let byte = i / 2;
                let shift = (i % 2) * 4;
                unsafe {
                    *out.get_unchecked_mut(byte) |= (idx & 0xF) << shift;
                }
            }
        }
        8 => {
            let copy_len = out.len().min(indices.len());
            out[..copy_len].copy_from_slice(&indices[..copy_len]);
        }
        _ => {
            out.fill(0);
            let mut bit_pos = 0usize;
            for &idx in indices {
                let byte_pos = bit_pos / 8;
                let shift = bit_pos % 8;
                unsafe {
                    *out.get_unchecked_mut(byte_pos) |= idx << shift;
                    if shift + bits as usize > 8 {
                        *out.get_unchecked_mut(byte_pos + 1) |= idx >> (8 - shift);
                    }
                }
                bit_pos += bits as usize;
            }
        }
    }
}

/// Unpack bytes back to indices.
#[allow(dead_code)]
fn unpack_indices(packed: &[u8], bits: u8, n: usize) -> Vec<u8> {
    match bits {
        2 => {
            let mut indices = vec![0u8; n];
            for (i, out) in indices.iter_mut().enumerate() {
                let byte = i / 4;
                let shift = (i % 4) * 2;
                if byte < packed.len() {
                    *out = (packed[byte] >> shift) & 0x3;
                }
            }
            indices
        }
        3 | 4 => {
            let mut indices = vec![0u8; n];
            for (i, out) in indices.iter_mut().enumerate() {
                let byte = i / 2;
                let shift = (i % 2) * 4;
                if byte < packed.len() {
                    *out = (packed[byte] >> shift) & 0xF;
                }
            }
            indices
        }
        8 => {
            let mut indices = vec![0u8; n];
            let copy_len = n.min(packed.len());
            indices[..copy_len].copy_from_slice(&packed[..copy_len]);
            indices
        }
        _ => {
            let mut indices = vec![0u8; n];
            let mask = (1u8 << bits) - 1;
            let mut bit_pos = 0usize;
            for idx in indices.iter_mut().take(n) {
                let byte_pos = bit_pos / 8;
                let shift = bit_pos % 8;
                if byte_pos < packed.len() {
                    *idx = (packed[byte_pos] >> shift) & mask;
                    if shift + bits as usize > 8 && byte_pos + 1 < packed.len() {
                        *idx |= (packed[byte_pos + 1] << (8 - shift)) & mask;
                    }
                }
                bit_pos += bits as usize;
            }
            indices
        }
    }
}

/// Unpack indices into pre-allocated buffer (zero-alloc).
fn unpack_indices_into(packed: &[u8], bits: u8, n: usize, out: &mut [u8]) {
    debug_assert!(out.len() >= n);
    match bits {
        2 => {
            for i in 0..n {
                let byte = i / 4;
                let shift = (i % 4) * 2;
                if byte < packed.len() {
                    unsafe {
                        *out.get_unchecked_mut(i) = (*packed.get_unchecked(byte) >> shift) & 0x3;
                    }
                } else {
                    unsafe {
                        *out.get_unchecked_mut(i) = 0;
                    }
                }
            }
        }
        3 | 4 => {
            for i in 0..n {
                let byte = i / 2;
                let shift = (i % 2) * 4;
                if byte < packed.len() {
                    unsafe {
                        *out.get_unchecked_mut(i) = (*packed.get_unchecked(byte) >> shift) & 0xF;
                    }
                } else {
                    unsafe {
                        *out.get_unchecked_mut(i) = 0;
                    }
                }
            }
        }
        8 => {
            let copy_len = n.min(packed.len());
            out[..copy_len].copy_from_slice(&packed[..copy_len]);
            out[copy_len..n].fill(0);
        }
        _ => {
            let mask = (1u8 << bits) - 1;
            let mut bit_pos = 0usize;
            for i in 0..n {
                let byte_pos = bit_pos / 8;
                let shift = bit_pos % 8;
                if byte_pos < packed.len() {
                    unsafe {
                        *out.get_unchecked_mut(i) =
                            (*packed.get_unchecked(byte_pos) >> shift) & mask;
                        if shift + bits as usize > 8 && byte_pos + 1 < packed.len() {
                            *out.get_unchecked_mut(i) |=
                                (*packed.get_unchecked(byte_pos + 1) << (8 - shift)) & mask;
                        }
                    }
                } else {
                    unsafe {
                        *out.get_unchecked_mut(i) = 0;
                    }
                }
                bit_pos += bits as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b).max(1e-8)
    }

    fn make_test_config(mode: IsoQuantMode) -> IsoQuantConfig {
        IsoQuantConfig {
            key_bits: 4,
            val_bits: 4,
            seed: 42,
            n_layers: 2,
            kv_dim: 64,
            max_seq_len: 32,
            mode,
        }
    }

    #[test]
    fn test_kv_cache_roundtrip_full() {
        let config = make_test_config(IsoQuantMode::Full);
        let mut cache = IsoQuantKVCache::new(&config);

        let key: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let value: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.1).cos()).collect();

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let mut reconstructed_key = vec![0.0f32; config.kv_dim];
        cache.dequantize_key_into(0, 0, &mut reconstructed_key);

        let mut reconstructed_val = vec![0.0f32; config.kv_dim];
        cache.dequantize_value_into(0, 0, &mut reconstructed_val);

        let key_cos = cosine_sim(&key, &reconstructed_key);
        let val_cos = cosine_sim(&value, &reconstructed_val);

        assert!(key_cos > 0.95, "key cosine = {key_cos}, expected > 0.95");
        assert!(val_cos > 0.95, "value cosine = {val_cos}, expected > 0.95");
    }

    #[test]
    fn test_kv_cache_roundtrip_fast() {
        let config = make_test_config(IsoQuantMode::Fast);
        let mut cache = IsoQuantKVCache::new(&config);

        let key: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let value: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.1).cos()).collect();

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let mut reconstructed_key = vec![0.0f32; config.kv_dim];
        cache.dequantize_key_into(0, 0, &mut reconstructed_key);

        let mut reconstructed_val = vec![0.0f32; config.kv_dim];
        cache.dequantize_value_into(0, 0, &mut reconstructed_val);

        let key_cos = cosine_sim(&key, &reconstructed_key);
        let val_cos = cosine_sim(&value, &reconstructed_val);

        assert!(key_cos > 0.95, "key cosine = {key_cos}, expected > 0.95");
        assert!(val_cos > 0.95, "value cosine = {val_cos}, expected > 0.95");
    }

    #[test]
    fn test_compression_ratio() {
        let config = make_test_config(IsoQuantMode::Full);
        let cache = IsoQuantKVCache::new(&config);

        // 4-bit: kv_dim/2 bytes per K + kv_dim/2 bytes per V + 8 bytes norms = 72 per layer
        // 2 layers = 144 bytes per token
        // f32: 64*4*2*2 = 1024 bytes per token
        // ratio = 1024 / 144 ≈ 7.1
        let ratio = cache.compression_ratio();
        assert!(ratio > 4.0, "compression ratio = {ratio}, expected > 4.0");
        assert!(ratio < 20.0, "compression ratio = {ratio}, expected < 20.0");
    }

    #[test]
    fn test_full_vs_fast_quality() {
        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let config_full = make_test_config(IsoQuantMode::Full);
        let mut cache_full = IsoQuantKVCache::new(&config_full);
        cache_full.store_key(0, 0, &key);
        let mut recon_full = vec![0.0f32; 64];
        cache_full.dequantize_key_into(0, 0, &mut recon_full);
        let cos_full = cosine_sim(&key, &recon_full);

        let config_fast = make_test_config(IsoQuantMode::Fast);
        let mut cache_fast = IsoQuantKVCache::new(&config_fast);
        cache_fast.store_key(0, 0, &key);
        let mut recon_fast = vec![0.0f32; 64];
        cache_fast.dequantize_key_into(0, 0, &mut recon_fast);
        let cos_fast = cosine_sim(&key, &recon_fast);

        // Both should be good quality
        assert!(cos_full > 0.95, "full cosine = {cos_full}");
        assert!(cos_fast > 0.95, "fast cosine = {cos_fast}");
    }

    #[test]
    fn test_reset_clears() {
        let config = make_test_config(IsoQuantMode::Full);
        let mut cache = IsoQuantKVCache::new(&config);

        let key: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);
        assert_eq!(cache.pos(), 0);

        cache.set_pos(5);
        assert_eq!(cache.pos(), 5);

        cache.reset();
        assert_eq!(cache.pos(), 0);

        // After reset, dequantize should return zeros (norm is 0)
        let mut out = vec![1.0f32; config.kv_dim];
        cache.dequantize_key_into(0, 0, &mut out);
        assert!(out.iter().all(|&x| x == 0.0), "expected zeros after reset");
    }

    #[test]
    fn test_zero_vector_handling() {
        let config = make_test_config(IsoQuantMode::Full);
        let mut cache = IsoQuantKVCache::new(&config);

        let zeros = vec![0.0f32; config.kv_dim];
        cache.store_key(0, 0, &zeros);

        let mut out = vec![1.0f32; config.kv_dim];
        cache.dequantize_key_into(0, 0, &mut out);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "expected zeros for zero input"
        );
    }

    #[test]
    fn test_bytes_per_token() {
        let config = make_test_config(IsoQuantMode::Full);
        let cache = IsoQuantKVCache::new(&config);

        let bpt = cache.bytes_per_token();
        // 4-bit: kv_dim/2 bytes per K + kv_dim/2 bytes per V + 8 bytes norms = 72 per layer
        // 2 layers = 144
        assert!(bpt > 0, "bytes_per_token should be positive");
        assert!(
            bpt < config.kv_dim * 4 * 2 * config.n_layers,
            "should be smaller than f32"
        );
    }

    #[test]
    fn test_multi_layer_independence() {
        let config = make_test_config(IsoQuantMode::Full);
        let mut cache = IsoQuantKVCache::new(&config);

        let key0: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let key1: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.2).cos()).collect();

        cache.store_key(0, 0, &key0);
        cache.store_key(1, 0, &key1);

        let mut out0 = vec![0.0f32; config.kv_dim];
        let mut out1 = vec![0.0f32; config.kv_dim];
        cache.dequantize_key_into(0, 0, &mut out0);
        cache.dequantize_key_into(1, 0, &mut out1);

        let cos0 = cosine_sim(&key0, &out0);
        let cos1 = cosine_sim(&key1, &out1);

        assert!(cos0 > 0.95, "layer 0 cosine = {cos0}");
        assert!(cos1 > 0.95, "layer 1 cosine = {cos1}");
    }

    #[test]
    fn test_dequantize_allocating() {
        // Test the allocating dequantize_key / dequantize_value variants
        let config = make_test_config(IsoQuantMode::Full);
        let mut cache = IsoQuantKVCache::new(&config);

        let key: Vec<f32> = (0..config.kv_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);

        let recon = cache.dequantize_key(0, 0);
        let cos = cosine_sim(&key, &recon);
        assert!(cos > 0.95, "allocating dequantize cosine = {cos}");
    }

    #[test]
    fn test_2bit_roundtrip() {
        let config = IsoQuantConfig {
            key_bits: 2,
            val_bits: 2,
            seed: 42,
            n_layers: 1,
            kv_dim: 64,
            max_seq_len: 16,
            mode: IsoQuantMode::Full,
        };
        let mut cache = IsoQuantKVCache::new(&config);

        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);

        let mut recon = vec![0.0f32; 64];
        cache.dequantize_key_into(0, 0, &mut recon);

        let cos = cosine_sim(&key, &recon);
        // 2-bit is lower quality but should still be reasonable
        assert!(cos > 0.80, "2-bit cosine = {cos}, expected > 0.80");
    }

    #[test]
    fn test_non_multiple_of_4_dim() {
        let config = IsoQuantConfig {
            key_bits: 4,
            val_bits: 4,
            seed: 42,
            n_layers: 1,
            kv_dim: 50, // Not a multiple of 4
            max_seq_len: 16,
            mode: IsoQuantMode::Full,
        };
        let mut cache = IsoQuantKVCache::new(&config);

        let key: Vec<f32> = (0..50).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);

        let mut recon = vec![0.0f32; 50];
        cache.dequantize_key_into(0, 0, &mut recon);

        let cos = cosine_sim(&key, &recon);
        assert!(cos > 0.95, "non-multiple-of-4 cosine = {cos}");
    }

    #[test]
    fn test_quantized_kv_cache_trait() {
        // Verify the trait impl compiles and works
        fn use_trait(cache: &mut dyn types::QuantizedKVCache) {
            assert_eq!(cache.pos(), 0);
            cache.set_pos(3);
            assert_eq!(cache.pos(), 3);
            cache.reset();
            assert_eq!(cache.pos(), 0);
        }

        let config = make_test_config(IsoQuantMode::Full);
        let mut cache = IsoQuantKVCache::new(&config);
        use_trait(&mut cache);
    }
}
