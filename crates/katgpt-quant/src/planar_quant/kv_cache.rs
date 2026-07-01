//! Compressed KV cache using PlanarQuant 2D Givens rotation quantization.
//!
//! Replaces TurboQuant's O(d²) random rotation with O(d) block-diagonal 2D rotations.
//! Each adjacent pair of elements is independently rotated by a random angle.
//! Storage: bit-packed indices + per-position norms, same as TurboQuant.

use super::rotation::{apply_inverse_rotation, apply_rotation, generate_givens_rotations};
use super::types::{PlanarQuantConfig, PlanarQuantLayer};
use katgpt_core::simd::simd_scale_inplace;
use crate::turboquant::codebook::compute_codebook;

/// Compressed KV cache using PlanarQuant 2D Givens rotation quantization.
///
/// Each KV vector is: normalized → 2D rotate → quantized → bit-packed.
/// Reconstruction: unpack → dequantize → inverse 2D rotate → rescale.
pub struct PlanarQuantKVCache {
    /// Per-layer quantization state (2D rotations + codebook centroids/boundaries).
    pub layers: Vec<PlanarQuantLayer>,
    /// Bit-packed key indices: layers × positions × packed_coords.
    key_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position key norms (for reconstruction).
    key_norms: Vec<Vec<f32>>,
    /// Bit-packed value indices.
    val_indices: Vec<Vec<Vec<u8>>>,
    /// Per-position value norms.
    val_norms: Vec<Vec<f32>>,
    /// Current write position.
    pos: usize,
    /// Number of layers.
    n_layers: usize,
    /// KV dimension (n_kv_head * head_dim).
    kv_dim: usize,
    /// Key bits per coordinate.
    key_bits: u8,
    /// Value bits per coordinate.
    val_bits: u8,
    /// Maximum sequence length.
    #[allow(dead_code)] // future: bounded reset, overflow checks
    max_seq_len: usize,
    /// Highest position ever written (for efficient reset).
    max_used_pos: usize,
    // ── Scratch buffers for zero-alloc hot path ──
    /// Normalized input: [kv_dim].
    scratch_normalized: Vec<f32>,
    /// Rotation output: [kv_dim].
    scratch_rotated: Vec<f32>,
    /// Quantized/unpacked indices: [kv_dim].
    scratch_indices: Vec<u8>,
}

impl PlanarQuantKVCache {
    /// Create from explicit config struct.
    pub fn with_config(config: &PlanarQuantConfig) -> Self {
        let kv_dim_padded = (config.kv_dim + 1) & !1; // round up to even
        let n_groups = kv_dim_padded / 2;

        let key_codebook = compute_codebook(config.kv_dim, config.key_bits);
        let val_codebook = compute_codebook(config.kv_dim, config.val_bits);

        let layers: Vec<PlanarQuantLayer> = (0..config.n_layers)
            .map(|layer_idx| {
                let seed = config
                    .seed
                    .wrapping_add(layer_idx as u64 * 1000)
                    .wrapping_add(1);
                let val_seed = seed.wrapping_add(500);
                PlanarQuantLayer {
                    key_rotations: generate_givens_rotations(n_groups, seed),
                    val_rotations: generate_givens_rotations(n_groups, val_seed),
                    key_centroids: key_codebook.centroids.clone(),
                    key_boundaries: key_codebook.boundaries.clone(),
                    val_centroids: val_codebook.centroids.clone(),
                    val_boundaries: val_codebook.boundaries.clone(),
                }
            })
            .collect();

        let packed_key_len = packed_len(config.kv_dim, config.key_bits);
        let packed_val_len = packed_len(config.kv_dim, config.val_bits);

        Self {
            layers,
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
            max_used_pos: 0,
            scratch_normalized: vec![0.0f32; kv_dim_padded],
            scratch_rotated: vec![0.0f32; kv_dim_padded],
            scratch_indices: vec![0u8; kv_dim_padded],
        }
    }

    /// Quantize and store a key vector at given layer and position.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        let layer_state = &self.layers[layer];

        // Track highest used position for efficient reset
        if pos > self.max_used_pos {
            self.max_used_pos = pos;
        }

        // Compute norm via SIMD (avoids scalar iteration)
        let norm = katgpt_core::simd::simd_sum_sq(key, key.len()).sqrt();
        self.key_norms[layer][pos] = norm;

        if norm < 1e-8 {
            return;
        }

        // Normalize into scratch buffer (copy + SIMD scale)
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..key.len()].copy_from_slice(key);
        // Zero-pad if kv_dim is odd (shouldn't happen in practice, but safe).
        // Only the tail slot needs zeroing — the [..kv_dim] range is fully
        // overwritten by the copy above.
        if key.len() < self.scratch_normalized.len() {
            self.scratch_normalized[key.len()..].fill(0.0);
        }
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // 2D Givens rotation → scratch_rotated
        apply_rotation(
            &layer_state.key_rotations,
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        // Quantize using boundaries
        for (i, &v) in self.scratch_rotated[..self.kv_dim].iter().enumerate() {
            unsafe {
                *self.scratch_indices.get_unchecked_mut(i) =
                    quantize_index(v, &layer_state.key_boundaries);
            }
        }

        // Pack into existing buffer (zero-alloc)
        pack_indices_into(
            &self.scratch_indices[..self.kv_dim],
            self.key_bits,
            &mut self.key_indices[layer][pos],
        );
    }

    /// Quantize and store a value vector at given layer and position.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        let layer_state = &self.layers[layer];

        // Track highest used position for efficient reset
        if pos > self.max_used_pos {
            self.max_used_pos = pos;
        }

        let norm = katgpt_core::simd::simd_sum_sq(value, value.len()).sqrt();
        self.val_norms[layer][pos] = norm;

        if norm < 1e-8 {
            return;
        }

        // Normalize into scratch buffer
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..value.len()].copy_from_slice(value);
        if value.len() < self.scratch_normalized.len() {
            self.scratch_normalized[value.len()..].fill(0.0);
        }
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // 2D Givens rotation → scratch_rotated
        apply_rotation(
            &layer_state.val_rotations,
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        // Quantize using boundaries
        for (i, &v) in self.scratch_rotated[..self.kv_dim].iter().enumerate() {
            unsafe {
                *self.scratch_indices.get_unchecked_mut(i) =
                    quantize_index(v, &layer_state.val_boundaries);
            }
        }

        // Pack into existing buffer
        pack_indices_into(
            &self.scratch_indices[..self.kv_dim],
            self.val_bits,
            &mut self.val_indices[layer][pos],
        );
    }

    /// Dequantize key at position. Returns reconstructed key vector.
    pub fn dequantize_key(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.key_norms[layer][pos];
        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let indices = unpack_indices(&self.key_indices[layer][pos], self.key_bits, self.kv_dim);
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| dequantize_index(i, &layer_state.key_centroids))
            .collect();

        // Pad for inverse rotation if needed
        let padded_len = (self.kv_dim + 1) & !1;
        let mut padded = vec![0.0f32; padded_len];
        padded[..self.kv_dim].copy_from_slice(&rotated);
        let mut normalized = vec![0.0f32; padded_len];

        apply_inverse_rotation(&layer_state.key_rotations, &padded, &mut normalized);
        normalized[..self.kv_dim].iter().map(|x| x * norm).collect()
    }

    /// Dequantize value at position. Returns reconstructed value vector.
    pub fn dequantize_value(&self, layer: usize, pos: usize) -> Vec<f32> {
        let layer_state = &self.layers[layer];
        let norm = self.val_norms[layer][pos];
        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let indices = unpack_indices(&self.val_indices[layer][pos], self.val_bits, self.kv_dim);
        let rotated: Vec<f32> = indices
            .iter()
            .map(|&i| dequantize_index(i, &layer_state.val_centroids))
            .collect();

        let padded_len = (self.kv_dim + 1) & !1;
        let mut padded = vec![0.0f32; padded_len];
        padded[..self.kv_dim].copy_from_slice(&rotated);
        let mut normalized = vec![0.0f32; padded_len];

        apply_inverse_rotation(&layer_state.val_rotations, &padded, &mut normalized);
        normalized[..self.kv_dim].iter().map(|x| x * norm).collect()
    }

    /// Dequantize key into pre-allocated buffer. Zero-alloc hot path.
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    /// Reconstruction: unpack → dequantize → inverse rotate → scale by norm.
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let norm = self.key_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let layer_state = &self.layers[layer];

        // Unpack into scratch_indices
        unpack_indices_into(
            &self.key_indices[layer][pos],
            self.key_bits,
            self.kv_dim,
            &mut self.scratch_indices,
        );

        // Dequantize → scratch_rotated (only clear padding, the kv_dim range
        // will be fully overwritten by the dequantize loop)
        if self.kv_dim < self.scratch_rotated.len() {
            self.scratch_rotated[self.kv_dim..].fill(0.0);
        }
        for i in 0..self.kv_dim {
            unsafe {
                *self.scratch_rotated.get_unchecked_mut(i) = dequantize_index(
                    *self.scratch_indices.get_unchecked(i),
                    &layer_state.key_centroids,
                );
            }
        }

        // Inverse 2D rotation → scratch_normalized
        apply_inverse_rotation(
            &layer_state.key_rotations,
            &self.scratch_rotated,
            &mut self.scratch_normalized,
        );

        // Scale by norm → output
        out.copy_from_slice(&self.scratch_normalized[..self.kv_dim]);
        simd_scale_inplace(out, norm);
    }

    /// Dequantize value into pre-allocated buffer. Zero-alloc hot path.
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let norm = self.val_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let layer_state = &self.layers[layer];

        // Unpack into scratch_indices
        unpack_indices_into(
            &self.val_indices[layer][pos],
            self.val_bits,
            self.kv_dim,
            &mut self.scratch_indices,
        );

        // Dequantize → scratch_rotated (only clear padding)
        if self.kv_dim < self.scratch_rotated.len() {
            self.scratch_rotated[self.kv_dim..].fill(0.0);
        }
        for i in 0..self.kv_dim {
            unsafe {
                *self.scratch_rotated.get_unchecked_mut(i) = dequantize_index(
                    *self.scratch_indices.get_unchecked(i),
                    &layer_state.val_centroids,
                );
            }
        }

        // Inverse 2D rotation → scratch_normalized
        apply_inverse_rotation(
            &layer_state.val_rotations,
            &self.scratch_rotated,
            &mut self.scratch_normalized,
        );

        // Scale by norm → output
        out.copy_from_slice(&self.scratch_normalized[..self.kv_dim]);
        simd_scale_inplace(out, norm);
    }

    /// Reset cache for new sequence.
    pub fn reset(&mut self) {
        // Only clear positions that were actually used
        let limit = self.max_used_pos + 1;
        for layer in 0..self.n_layers {
            for pos in 0..limit {
                self.key_indices[layer][pos].fill(0);
                self.key_norms[layer][pos] = 0.0;
                self.val_indices[layer][pos].fill(0);
                self.val_norms[layer][pos] = 0.0;
            }
        }
        self.pos = 0;
        self.max_used_pos = 0;
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
}

impl katgpt_core::types::QuantizedKVCache for PlanarQuantKVCache {
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

// ── Quantize / Dequantize helpers ─────────────────────────────

/// Quantize a value to an index using decision boundaries.
///
/// Returns index in `[0, boundaries.len()]`.
/// For small codebooks (2-4 bits = 4-16 levels), a branchless linear scan
/// beats binary search due to cache locality and reduced overhead.
#[inline]
fn quantize_index(value: f32, boundaries: &[f32]) -> u8 {
    // Fast path for small codebooks (2-4 bits = 4-16 levels)
    if boundaries.len() <= 15 {
        let mut idx = 0u8;
        for &b in boundaries {
            idx += (value >= b) as u8;
        }
        return idx;
    }
    boundaries.partition_point(|&b| value >= b) as u8
}

/// Dequantize an index back to centroid value.
/// Uses unchecked access since indices are validated during quantization.
#[inline]
fn dequantize_index(index: u8, centroids: &[f32]) -> f32 {
    unsafe { *centroids.get_unchecked(index as usize) }
}

// ── Bit packing ──────────────────────────────────────────────

/// Packed byte length for n values at given bits per value.
fn packed_len(n: usize, bits: u8) -> usize {
    match bits {
        2 => n.div_ceil(4),
        // 3-bit quantization is stored as 4-bit (wastes 1 bit per index but
        // avoids the complexity of non-power-of-2 packing).
        3 | 4 => n.div_ceil(2),
        8 => n,
        _ => (n * bits as usize).div_ceil(8),
    }
}

/// Pack indices into pre-allocated buffer (zero-alloc).
fn pack_indices_into(indices: &[u8], bits: u8, out: &mut [u8]) {
    match bits {
        2 => {
            out.fill(0);
            let n = indices.len();
            let full_quads = n / 4;
            for q in 0..full_quads {
                let base = q * 4;
                unsafe {
                    let i0 = *indices.get_unchecked(base) & 0x3;
                    let i1 = *indices.get_unchecked(base + 1) & 0x3;
                    let i2 = *indices.get_unchecked(base + 2) & 0x3;
                    let i3 = *indices.get_unchecked(base + 3) & 0x3;
                    *out.get_unchecked_mut(q) = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6);
                }
            }
            let remainder = n % 4;
            if remainder > 0 {
                let base = full_quads * 4;
                let mut byte = 0u8;
                for i in 0..remainder {
                    unsafe {
                        byte |= (*indices.get_unchecked(base + i) & 0x3) << (i * 2);
                    }
                }
                unsafe {
                    *out.get_unchecked_mut(full_quads) = byte;
                }
            }
        }
        3 | 4 => {
            out.fill(0);
            let n = indices.len();
            let full_pairs = n / 2;
            for p in 0..full_pairs {
                let base = p * 2;
                unsafe {
                    let lo = *indices.get_unchecked(base) & 0xF;
                    let hi = *indices.get_unchecked(base + 1) & 0xF;
                    *out.get_unchecked_mut(p) = lo | (hi << 4);
                }
            }
            if !n.is_multiple_of(2) {
                unsafe {
                    *out.get_unchecked_mut(full_pairs) = *indices.get_unchecked(n - 1) & 0xF;
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
fn unpack_indices(packed: &[u8], bits: u8, n: usize) -> Vec<u8> {
    match bits {
        2 => {
            let mut indices = vec![0u8; n];
            let full_quads = n / 4;
            for q in 0..full_quads {
                if q < packed.len() {
                    let b = packed[q];
                    let base = q * 4;
                    indices[base] = b & 0x3;
                    indices[base + 1] = (b >> 2) & 0x3;
                    indices[base + 2] = (b >> 4) & 0x3;
                    indices[base + 3] = (b >> 6) & 0x3;
                }
            }
            let remainder = n % 4;
            if remainder > 0 {
                let base = full_quads * 4;
                if full_quads < packed.len() {
                    let b = packed[full_quads];
                    for i in 0..remainder {
                        indices[base + i] = (b >> (i * 2)) & 0x3;
                    }
                }
            }
            indices
        }
        3 | 4 => {
            let mut indices = vec![0u8; n];
            let full_pairs = n / 2;
            for p in 0..full_pairs {
                if p < packed.len() {
                    let b = packed[p];
                    let base = p * 2;
                    indices[base] = b & 0xF;
                    indices[base + 1] = (b >> 4) & 0xF;
                }
            }
            if !n.is_multiple_of(2) && full_pairs < packed.len() {
                indices[n - 1] = packed[full_pairs] & 0xF;
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
            let mask = (1u8 << bits) - 1;
            let mut indices = vec![0u8; n];
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
            // Hoist `q < packed.len()` out of the inner loop: process the
            // available full bytes in a branch-free body, then a single fill(0)
            // covers any indices whose source byte is missing.
            let n_full_bytes = packed.len().min(n / 4);
            for q in 0..n_full_bytes {
                let base = q * 4;
                unsafe {
                    let b = *packed.get_unchecked(q);
                    *out.get_unchecked_mut(base) = b & 0x3;
                    *out.get_unchecked_mut(base + 1) = (b >> 2) & 0x3;
                    *out.get_unchecked_mut(base + 2) = (b >> 4) & 0x3;
                    *out.get_unchecked_mut(base + 3) = (b >> 6) & 0x3;
                }
            }
            let consumed = n_full_bytes * 4;
            if consumed < n {
                if n_full_bytes < packed.len() {
                    let base = consumed;
                    let remainder = n - consumed;
                    unsafe {
                        let b = *packed.get_unchecked(n_full_bytes);
                        for i in 0..remainder {
                            *out.get_unchecked_mut(base + i) = (b >> (i * 2)) & 0x3;
                        }
                    }
                } else {
                    out[consumed..n].fill(0);
                }
            }
        }
        3 | 4 => {
            // Same hoisting pattern as the 2-bit case.
            let n_full_bytes = packed.len().min(n / 2);
            for p in 0..n_full_bytes {
                let base = p * 2;
                unsafe {
                    let b = *packed.get_unchecked(p);
                    *out.get_unchecked_mut(base) = b & 0xF;
                    *out.get_unchecked_mut(base + 1) = (b >> 4) & 0xF;
                }
            }
            let consumed = n_full_bytes * 2;
            if consumed < n {
                if n_full_bytes < packed.len() {
                    unsafe {
                        *out.get_unchecked_mut(consumed) =
                            *packed.get_unchecked(n_full_bytes) & 0xF;
                    }
                } else {
                    out[consumed..n].fill(0);
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

    /// Cosine similarity between two vectors.
    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            return 0.0;
        }
        dot / (na * nb)
    }

    fn make_test_config() -> PlanarQuantConfig {
        PlanarQuantConfig {
            key_bits: 4,
            val_bits: 4,
            seed: 42,
            n_layers: 2,
            kv_dim: 64,
            max_seq_len: 32,
        }
    }

    fn make_random_vec(kv_dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = katgpt_core::types::Rng::new(seed);
        (0..kv_dim).map(|_| rng.normal()).collect()
    }

    #[test]
    fn test_kv_cache_roundtrip() {
        let config = make_test_config();
        let mut cache = PlanarQuantKVCache::with_config(&config);

        for layer in 0..config.n_layers {
            for pos in 0..8 {
                let key = make_random_vec(config.kv_dim, 100 + layer as u64 * 1000 + pos as u64);
                let value = make_random_vec(config.kv_dim, 200 + layer as u64 * 1000 + pos as u64);
                cache.store_key(layer, pos, &key);
                cache.store_value(layer, pos, &value);
            }
        }

        for layer in 0..config.n_layers {
            for pos in 0..8 {
                let key = make_random_vec(config.kv_dim, 100 + layer as u64 * 1000 + pos as u64);
                let value = make_random_vec(config.kv_dim, 200 + layer as u64 * 1000 + pos as u64);

                let reconstructed_key = cache.dequantize_key(layer, pos);
                let reconstructed_val = cache.dequantize_value(layer, pos);

                let key_sim = cosine_sim(&key, &reconstructed_key);
                let val_sim = cosine_sim(&value, &reconstructed_val);

                assert!(
                    key_sim > 0.90,
                    "key cosine sim too low at layer={layer} pos={pos}: {key_sim}"
                );
                assert!(
                    val_sim > 0.90,
                    "val cosine sim too low at layer={layer} pos={pos}: {val_sim}"
                );
            }
        }
    }

    #[test]
    fn test_dequantize_into_matches_dequantize() {
        let config = make_test_config();
        let mut cache = PlanarQuantKVCache::with_config(&config);

        let key = make_random_vec(config.kv_dim, 999);
        let value = make_random_vec(config.kv_dim, 888);
        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let key_alloc = cache.dequantize_key(0, 0);
        let val_alloc = cache.dequantize_value(0, 0);

        let mut key_buf = vec![0.0f32; config.kv_dim];
        let mut val_buf = vec![0.0f32; config.kv_dim];
        cache.dequantize_key_into(0, 0, &mut key_buf);
        cache.dequantize_value_into(0, 0, &mut val_buf);

        for i in 0..config.kv_dim {
            assert!(
                (key_alloc[i] - key_buf[i]).abs() < 1e-6,
                "key mismatch at [{i}]: {} vs {}",
                key_alloc[i],
                key_buf[i]
            );
            assert!(
                (val_alloc[i] - val_buf[i]).abs() < 1e-6,
                "val mismatch at [{i}]: {} vs {}",
                val_alloc[i],
                val_buf[i]
            );
        }
    }

    #[test]
    fn test_compression_ratio() {
        let config = make_test_config();
        let cache = PlanarQuantKVCache::with_config(&config);

        // 64 dim × 4 bytes × 2 (K+V) × 2 layers = 1024 bytes
        let flat_bytes = config.kv_dim * 4 * 2 * config.n_layers;
        let ratio = cache.compression_ratio();

        // 4-bit packing: ~8× compression (plus norm overhead)
        assert!(ratio > 4.0, "compression ratio too low: {ratio}");
        assert!(ratio < 20.0, "compression ratio suspiciously high: {ratio}");
        assert!(
            (flat_bytes as f64 / cache.bytes_per_token() as f64 - ratio).abs() < 0.01,
            "ratio mismatch"
        );
    }

    #[test]
    fn test_bytes_per_token() {
        let config = make_test_config();
        let cache = PlanarQuantKVCache::with_config(&config);

        let bpt = cache.bytes_per_token();
        // 4-bit: kv_dim/2 bytes per K + kv_dim/2 bytes per V + 8 for norms = 72 per layer
        // × 2 layers = 144
        let expected_per_layer = config.kv_dim / 2 + config.kv_dim / 2 + 8;
        assert_eq!(bpt, expected_per_layer * config.n_layers);
    }

    #[test]
    fn test_reset_clears() {
        let config = make_test_config();
        let mut cache = PlanarQuantKVCache::with_config(&config);

        let key = make_random_vec(config.kv_dim, 42);
        cache.store_key(0, 0, &key);
        assert!(cache.key_norms[0][0] > 0.0);

        cache.reset();
        assert_eq!(cache.pos(), 0);
        assert_eq!(cache.key_norms[0][0], 0.0);
        assert_eq!(cache.val_norms[0][0], 0.0);
    }

    #[test]
    fn test_zero_vector_handling() {
        let config = make_test_config();
        let mut cache = PlanarQuantKVCache::with_config(&config);

        let zeros = vec![0.0f32; config.kv_dim];
        cache.store_key(0, 0, &zeros);
        cache.store_value(0, 0, &zeros);

        let key = cache.dequantize_key(0, 0);
        let val = cache.dequantize_value(0, 0);
        assert!(key.iter().all(|&x| x.abs() < 1e-8));
        assert!(val.iter().all(|&x| x.abs() < 1e-8));
    }

    #[test]
    fn test_multi_layer_independence() {
        let config = PlanarQuantConfig {
            key_bits: 4,
            val_bits: 4,
            seed: 42,
            n_layers: 3,
            kv_dim: 32,
            max_seq_len: 8,
        };
        let mut cache = PlanarQuantKVCache::with_config(&config);

        let key = make_random_vec(config.kv_dim, 777);

        // Store same key in all layers
        for layer in 0..config.n_layers {
            cache.store_key(layer, 0, &key);
        }

        // Each layer should have different rotations → different packed data
        // but same norm
        for layer in 0..config.n_layers {
            assert!(
                (cache.key_norms[layer][0] - cache.key_norms[0][0]).abs() < 1e-5,
                "norms should match across layers"
            );
        }
    }

    #[test]
    fn test_pos_management() {
        let config = make_test_config();
        let mut cache = PlanarQuantKVCache::with_config(&config);

        assert_eq!(cache.pos(), 0);
        cache.set_pos(10);
        assert_eq!(cache.pos(), 10);
        cache.reset();
        assert_eq!(cache.pos(), 0);
    }

    #[test]
    fn test_pack_unpack_roundtrip_2bit() {
        let n = 16;
        let indices: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        let mut packed = vec![0u8; packed_len(n, 2)];
        pack_indices_into(&indices, 2, &mut packed);
        let recovered = unpack_indices(&packed, 2, n);
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_pack_unpack_roundtrip_4bit() {
        let n = 16;
        let indices: Vec<u8> = (0..n).map(|i| (i % 16) as u8).collect();
        let mut packed = vec![0u8; packed_len(n, 4)];
        pack_indices_into(&indices, 4, &mut packed);
        let recovered = unpack_indices(&packed, 4, n);
        assert_eq!(indices, recovered);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let cb = crate::turboquant::codebook::compute_codebook(64, 4);
        let boundaries = &cb.boundaries;
        let centroids = &cb.centroids;

        // Values near the distribution center should roundtrip well
        for v in [-0.3f32, -0.1, 0.0, 0.1, 0.3] {
            let idx = quantize_index(v, boundaries);
            let reconstructed = dequantize_index(idx, centroids);
            assert!(
                (reconstructed - v).abs() < 0.5,
                "roundtrip failed for {v} -> idx {idx} -> {reconstructed}"
            );
        }
    }
}
