//! OCTOPUS KV cache: stores key/value vectors in octahedral-triplet quantized form.
//!
//! Encoding pipeline (per vector):
//! 1. Normalize to unit length, store L2 norm separately
//! 2. Apply random rotation (same approach as TurboQuant)
//! 3. Decompose rotated vector into ⌈d/3⌉ triplets
//! 4. Encode each triplet via octahedral map + quantize (ξ, η, ρ)
//! 5. Bit-pack triplet indices into contiguous byte buffer
//!
//! Decoding pipeline (reverse):
//! 1. Unpack triplet indices from byte buffer
//! 2. Decode each triplet: dequantize → oct_decode → reconstruct 3-vector
//! 3. Recompose into d-dimensional rotated vector (truncate zero-pad)
//! 4. Inverse rotate, scale by stored norm

use super::encode::{
    decode_vector_into, encode_vector_into, pack_triplet_indices_into, unpack_triplet_indices,
    unpack_triplet_indices_into,
};
use super::triplet::n_triplets;
use super::types::{OctopusCodebook, OctopusConfig, OctopusLayer, TripletIndices};

/// OCTOPUS compressed KV cache.
///
/// Each (layer, position) stores:
/// - One `f32` norm (the original vector's L2 norm)
/// - Packed byte buffer of triplet indices (2·dir_bits + nrm_bits bits per triplet)
///
/// Internally uses scratch buffers for the encode/decode hot path to minimize
/// allocations during attention scoring.
pub struct OctopusKVCache {
    /// Per-layer rotation matrices + dual codebooks.
    pub layers: Vec<OctopusLayer>,
    /// Packed key triplet indices: [layer][pos][packed_bytes].
    key_packed: Vec<Vec<Vec<u8>>>,
    /// Per-position key L2 norms: [layer][pos].
    key_norms: Vec<Vec<f32>>,
    /// Packed value triplet indices: [layer][pos][packed_bytes].
    val_packed: Vec<Vec<Vec<u8>>>,
    /// Per-position value L2 norms: [layer][pos].
    val_norms: Vec<Vec<f32>>,
    // ── Scratch buffers for hot path ──
    /// [kv_dim] — normalized vector / inverse rotation output.
    scratch_normalized: Vec<f32>,
    /// [n_triplets * 3] — rotated vector / decoded triplet vectors.
    scratch_workspace: Vec<f32>,
    /// [n_triplets] — reusable buffer for decomposed triplets (avoids alloc in store_key/store_value).
    scratch_triplets: Vec<super::triplet::Triplet>,
    /// [n_triplets] — reusable buffer for unpacked triplet indices.
    scratch_indices: Vec<TripletIndices>,
    // ── usize fields (grouped for alignment) ──
    /// Current write position.
    pos: usize,
    /// Highest position ever written (for efficient reset).
    max_used_pos: usize,
    /// Number of transformer layers.
    n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads).
    kv_dim: usize,
    /// Maximum sequence length.
    #[allow(dead_code)]
    max_seq_len: usize,
    /// Number of triplets: ⌈kv_dim/3⌉.
    n_triplets: usize,
    // ── Small fields at end to minimize padding ──
    /// Nominal bits per key coordinate.
    key_bits: u8,
    /// Nominal bits per value coordinate.
    val_bits: u8,
    /// Use joint 3×3 rounding in encoder.
    use_joint_rounding: bool,
}

impl OctopusKVCache {
    /// Create a new OCTOPUS KV cache from transformer config.
    ///
    /// Uses the same rotation matrix for all layers (deterministic from seed 42).
    /// Codebooks are shared across layers since they depend only on (kv_dim, bits).
    pub fn new(config: &crate::types::Config, key_bits: u8, val_bits: u8) -> Self {
        let n_layers = config.n_layer;
        let kv_dim = crate::types::kv_dim(config);
        let max_seq_len = config.block_size;
        let oct_config = OctopusConfig {
            key_bits,
            val_bits,
            seed: 42,
            n_layers,
            kv_dim,
            max_seq_len,
            use_qjl_residual: false,
            use_joint_rounding: true,
        };
        Self::with_config(&oct_config)
    }

    /// Create from explicit OCTOPUS config.
    pub fn with_config(cfg: &OctopusConfig) -> Self {
        let n_tri = n_triplets(cfg.kv_dim);

        // Build shared rotation + codebooks
        let rotation = generate_rotation_matrix(cfg.kv_dim, cfg.seed);
        let key_codebook = OctopusCodebook::build(cfg.kv_dim, cfg.key_bits);
        let val_codebook = OctopusCodebook::build(cfg.kv_dim, cfg.val_bits);
        let qjl_matrix = if cfg.use_qjl_residual {
            Some(generate_qjl_matrix(cfg.kv_dim, cfg.seed.wrapping_add(1)))
        } else {
            None
        };

        let layer = OctopusLayer {
            rotation,
            key_codebook,
            val_codebook,
            qjl_matrix,
        };
        let layers = vec![layer; cfg.n_layers];

        let packed_key_len = packed_triplet_len(n_tri, cfg.key_bits);
        let packed_val_len = packed_triplet_len(n_tri, cfg.val_bits);

        Self {
            layers,
            key_packed: vec![vec![vec![0u8; packed_key_len]; cfg.max_seq_len]; cfg.n_layers],
            key_norms: vec![vec![0.0f32; cfg.max_seq_len]; cfg.n_layers],
            val_packed: vec![vec![vec![0u8; packed_val_len]; cfg.max_seq_len]; cfg.n_layers],
            val_norms: vec![vec![0.0f32; cfg.max_seq_len]; cfg.n_layers],
            pos: 0,
            max_used_pos: 0,
            n_layers: cfg.n_layers,
            kv_dim: cfg.kv_dim,
            key_bits: cfg.key_bits,
            val_bits: cfg.val_bits,
            max_seq_len: cfg.max_seq_len,
            use_joint_rounding: cfg.use_joint_rounding,
            n_triplets: n_tri,
            scratch_normalized: vec![0.0f32; cfg.kv_dim],
            scratch_workspace: vec![0.0f32; n_tri * 3],
            scratch_triplets: Vec::with_capacity(n_tri),
            scratch_indices: Vec::with_capacity(n_tri),
        }
    }

    /// Quantize and store a key vector at given layer and position.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        self.max_used_pos = self.max_used_pos.max(pos + 1);
        // SIMD norm computation (avoids scalar iteration)
        let norm = crate::simd::simd_sum_sq(key, self.kv_dim).sqrt();
        self.key_norms[layer][pos] = norm;

        if norm < 1e-8 {
            self.key_packed[layer][pos].fill(0);
            return;
        }

        // Normalize
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..key.len()].copy_from_slice(key);
        scale_inplace(&mut self.scratch_normalized, inv_norm);

        // Rotate: R · normalized → workspace
        mat_vec_into(
            &self.layers[layer].rotation,
            &self.scratch_normalized,
            &mut self.scratch_workspace[..self.kv_dim],
        );

        // Encode triplets into scratch_indices (zero-alloc)
        let cb = &self.layers[layer].key_codebook;
        super::triplet::decompose_into(&self.scratch_workspace[..self.kv_dim], &mut self.scratch_triplets);
        encode_vector_into(
            &self.scratch_triplets,
            cb,
            self.use_joint_rounding,
            &mut self.scratch_indices,
        );
        // Pack into pre-allocated buffer (zero-alloc)
        pack_triplet_indices_into(
            &self.scratch_indices,
            cb.dir_bits,
            cb.nrm_bits,
            &mut self.key_packed[layer][pos],
        );
    }

    /// Quantize and store a value vector at given layer and position.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        self.max_used_pos = self.max_used_pos.max(pos + 1);
        // SIMD norm computation (avoids scalar iteration)
        let norm = crate::simd::simd_sum_sq(value, self.kv_dim).sqrt();
        self.val_norms[layer][pos] = norm;

        if norm < 1e-8 {
            self.val_packed[layer][pos].fill(0);
            return;
        }

        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..value.len()].copy_from_slice(value);
        scale_inplace(&mut self.scratch_normalized, inv_norm);

        mat_vec_into(
            &self.layers[layer].rotation,
            &self.scratch_normalized,
            &mut self.scratch_workspace[..self.kv_dim],
        );

        // Encode triplets into scratch_indices (zero-alloc)
        let cb = &self.layers[layer].val_codebook;
        super::triplet::decompose_into(&self.scratch_workspace[..self.kv_dim], &mut self.scratch_triplets);
        encode_vector_into(
            &self.scratch_triplets,
            cb,
            self.use_joint_rounding,
            &mut self.scratch_indices,
        );
        // Pack into pre-allocated buffer (zero-alloc)
        pack_triplet_indices_into(
            &self.scratch_indices,
            cb.dir_bits,
            cb.nrm_bits,
            &mut self.val_packed[layer][pos],
        );
    }

    /// Dequantize key at position. Returns reconstructed key vector.
    pub fn dequantize_key(&self, layer: usize, pos: usize) -> Vec<f32> {
        let norm = self.key_norms[layer][pos];
        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let cb = &self.layers[layer].key_codebook;
        let indices = unpack_triplet_indices(
            &self.key_packed[layer][pos],
            self.n_triplets,
            cb.dir_bits,
            cb.nrm_bits,
        );

        let mut decoded = vec![0.0f32; self.n_triplets * 3];
        decode_vector_into(&indices, cb, &mut decoded);

        let normalized = mat_vec_t(&self.layers[layer].rotation, &decoded[..self.kv_dim]);
        normalized.iter().map(|x| x * norm).collect()
    }

    /// Dequantize value at position. Returns reconstructed value vector.
    pub fn dequantize_value(&self, layer: usize, pos: usize) -> Vec<f32> {
        let norm = self.val_norms[layer][pos];
        if norm < 1e-8 {
            return vec![0.0; self.kv_dim];
        }

        let cb = &self.layers[layer].val_codebook;
        let indices = unpack_triplet_indices(
            &self.val_packed[layer][pos],
            self.n_triplets,
            cb.dir_bits,
            cb.nrm_bits,
        );

        let mut decoded = vec![0.0f32; self.n_triplets * 3];
        decode_vector_into(&indices, cb, &mut decoded);

        let normalized = mat_vec_t(&self.layers[layer].rotation, &decoded[..self.kv_dim]);
        normalized.iter().map(|x| x * norm).collect()
    }

    /// Dequantize key into pre-allocated buffer. Zero-alloc hot path.
    ///
    /// Uses internal scratch buffers — requires `&mut self`.
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let norm = self.key_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let cb = &self.layers[layer].key_codebook;
        unpack_triplet_indices_into(
            &self.key_packed[layer][pos],
            self.n_triplets,
            cb.dir_bits,
            cb.nrm_bits,
            &mut self.scratch_indices,
        );

        // Decode triplets into workspace
        decode_vector_into(&self.scratch_indices, cb, &mut self.scratch_workspace);

        // Inverse rotate: R^T · workspace[..kv_dim] → scratch_normalized
        mat_vec_t_into(
            &self.layers[layer].rotation,
            &self.scratch_workspace[..self.kv_dim],
            &mut self.scratch_normalized,
        );

        // Scale by norm → output
        out.copy_from_slice(&self.scratch_normalized);
        scale_inplace(out, norm);
    }

    /// Dequantize value into pre-allocated buffer. Zero-alloc hot path.
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let norm = self.val_norms[layer][pos];

        if norm < 1e-8 {
            out.fill(0.0);
            return;
        }

        let cb = &self.layers[layer].val_codebook;
        unpack_triplet_indices_into(
            &self.val_packed[layer][pos],
            self.n_triplets,
            cb.dir_bits,
            cb.nrm_bits,
            &mut self.scratch_indices,
        );

        decode_vector_into(&self.scratch_indices, cb, &mut self.scratch_workspace);

        mat_vec_t_into(
            &self.layers[layer].rotation,
            &self.scratch_workspace[..self.kv_dim],
            &mut self.scratch_normalized,
        );

        out.copy_from_slice(&self.scratch_normalized);
        scale_inplace(out, norm);
    }

    /// Reset cache for a new sequence.
    pub fn reset(&mut self) {
        let used = self.max_used_pos;
        for layer in 0..self.n_layers {
            for pos in 0..used {
                self.key_packed[layer][pos].fill(0);
                self.key_norms[layer][pos] = 0.0;
                self.val_packed[layer][pos].fill(0);
                self.val_norms[layer][pos] = 0.0;
            }
        }
        self.pos = 0;
        self.max_used_pos = 0;
    }

    /// Bytes stored per token (K + V, all layers).
    pub fn bytes_per_token(&self) -> usize {
        let packed_key = packed_triplet_len(self.n_triplets, self.key_bits);
        let packed_val = packed_triplet_len(self.n_triplets, self.val_bits);
        let per_layer = packed_key + packed_val + 8; // +8 for two f32 norms
        per_layer * self.n_layers
    }

    /// Compression ratio vs f32 KV cache.
    pub fn compression_ratio(&self) -> f64 {
        let flat_bytes = self.kv_dim * 4 * 2 * self.n_layers; // f32, K+V
        flat_bytes as f64 / self.bytes_per_token() as f64
    }

    /// Current write position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set current position.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// Get KV dimension.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
}

impl crate::types::QuantizedKVCache for OctopusKVCache {
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

// ── Rotation matrix generation ───────────────────────────────
// Same algorithm as TurboQuant: random orthogonal via QR decomposition.
// Kept local to avoid cross-feature dependency.

/// Generate a random orthogonal matrix via QR decomposition (column-major).
fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = crate::types::Rng::new(seed);
    let mut mat = vec![0.0f32; dim * dim];
    for val in mat.iter_mut() {
        *val = rng.normal();
    }

    // QR via modified Gram-Schmidt (column-major)
    let mut q = vec![0.0f32; dim * dim];
    let mut v: Vec<Vec<f32>> = (0..dim).map(|_| vec![0.0f32; dim]).collect();

    for col in 0..dim {
        for row in 0..dim {
            v[col][row] = mat[row * dim + col];
        }
    }

    for i in 0..dim {
        for j in 0..i {
            let dot: f32 = (0..dim).map(|k| q[k * dim + j] * v[i][k]).sum();
            for k in 0..dim {
                v[i][k] -= dot * q[k * dim + j];
            }
        }
        let norm: f32 = v[i].iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for k in 0..dim {
                q[k * dim + i] = v[i][k] / norm;
            }
        }
    }
    q
}

/// Generate QJL projection matrix (i.i.d. N(0,1) entries).
fn generate_qjl_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = crate::types::Rng::new(seed);
    let mut mat = vec![0.0f32; dim * dim];
    for val in mat.iter_mut() {
        *val = rng.normal();
    }
    mat
}

// ── Matrix-vector operations ─────────────────────────────────
// Column-major storage: mat[col * dim + row].

/// Matrix-vector multiply: out = mat · in (column-major).
///
/// Optimized with 4-column-at-a-time processing for register reuse
/// and unsafe access to eliminate bounds checks in the inner loop.
fn mat_vec_into(mat: &[f32], input: &[f32], out: &mut [f32]) {
    let dim = input.len();
    out[..dim].fill(0.0);
    let chunks4 = dim / 4;

    // Process 4 columns at a time — keeps 4 column scalars in registers
    // and amortizes the row-loop overhead across 4 FMAs per iteration.
    for c in 0..chunks4 {
        let base = c * 4;
        unsafe {
            let x0 = *input.get_unchecked(base);
            let x1 = *input.get_unchecked(base + 1);
            let x2 = *input.get_unchecked(base + 2);
            let x3 = *input.get_unchecked(base + 3);
            let col0 = base * dim;
            let col1 = (base + 1) * dim;
            let col2 = (base + 2) * dim;
            let col3 = (base + 3) * dim;
            for r in 0..dim {
                *out.get_unchecked_mut(r) +=
                    *mat.get_unchecked(col0 + r) * x0
                    + *mat.get_unchecked(col1 + r) * x1
                    + *mat.get_unchecked(col2 + r) * x2
                    + *mat.get_unchecked(col3 + r) * x3;
            }
        }
    }

    // Handle remaining columns (0..3)
    for c in (chunks4 * 4)..dim {
        let col_base = c * dim;
        unsafe {
            let x = *input.get_unchecked(c);
            for r in 0..dim {
                *out.get_unchecked_mut(r) += *mat.get_unchecked(col_base + r) * x;
            }
        }
    }
}

/// Transpose matrix-vector multiply: out = mat^T · in (column-major).
fn mat_vec_t(mat: &[f32], input: &[f32]) -> Vec<f32> {
    let dim = input.len();
    let mut out = vec![0.0f32; dim];
    mat_vec_t_into(mat, input, &mut out);
    out
}

/// Transpose matrix-vector multiply into pre-allocated buffer.
///
/// Uses SIMD dot product for the inner loop where possible.
fn mat_vec_t_into(mat: &[f32], input: &[f32], out: &mut [f32]) {
    let dim = input.len();
    debug_assert_eq!(out.len(), dim);
    for row in 0..dim {
        let m_row = &mat[row * dim..(row + 1) * dim];
        out[row] = crate::simd::simd_dot_f32(m_row, input, dim);
    }
}

/// Scale buffer in-place using SIMD.
fn scale_inplace(buf: &mut [f32], s: f32) {
    crate::simd::simd_scale_inplace(buf, s);
}

/// Compute packed byte length for n_triplets with given nominal bits.
fn packed_triplet_len(n_tri: usize, nominal_bits: u8) -> usize {
    let dir_bits = OctopusConfig::dir_bits(nominal_bits) as usize;
    let nrm_bits = OctopusConfig::nrm_bits(nominal_bits) as usize;
    let bits_per_triplet = 2 * dir_bits + nrm_bits;
    (n_tri * bits_per_triplet).div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(kv_dim: usize, key_bits: u8, val_bits: u8) -> OctopusKVCache {
        let cfg = OctopusConfig {
            key_bits,
            val_bits,
            seed: 42,
            n_layers: 2,
            kv_dim,
            max_seq_len: 32,
            use_qjl_residual: false,
            use_joint_rounding: true,
        };
        OctopusKVCache::with_config(&cfg)
    }

    // ── Basic roundtrip ──────────────────────────────────────

    #[test]
    fn test_kv_cache_roundtrip() {
        let mut cache = make_cache(64, 2, 2);
        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let value: Vec<f32> = (0..64).map(|i| (i as f32 * 0.07).cos()).collect();

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let recon_key = cache.dequantize_key(0, 0);
        let recon_val = cache.dequantize_value(0, 0);

        let cos_key = cosine_sim(&key, &recon_key);
        let cos_val = cosine_sim(&value, &recon_val);

        assert!(cos_key > 0.7, "key cosine = {cos_key}");
        assert!(cos_val > 0.7, "value cosine = {cos_val}");
    }

    #[test]
    fn test_kv_cache_roundtrip_dim128() {
        let mut cache = make_cache(128, 3, 3);
        let key: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let value: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).cos()).collect();

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let recon_key = cache.dequantize_key(0, 0);
        let recon_val = cache.dequantize_value(0, 0);

        let cos_key = cosine_sim(&key, &recon_key);
        let cos_val = cosine_sim(&value, &recon_val);

        assert!(cos_key > 0.8, "key cosine = {cos_key}");
        assert!(cos_val > 0.8, "value cosine = {cos_val}");
    }

    #[test]
    fn test_kv_cache_roundtrip_multi_pos() {
        let mut cache = make_cache(64, 3, 3);
        for pos in 0..4 {
            let key: Vec<f32> = (0..64)
                .map(|i| ((i + pos * 7) as f32 * 0.1).sin())
                .collect();
            let value: Vec<f32> = (0..64)
                .map(|i| ((i + pos * 11) as f32 * 0.1).cos())
                .collect();
            cache.store_key(0, pos, &key);
            cache.store_value(0, pos, &value);
        }

        for pos in 0..4 {
            let orig_key: Vec<f32> = (0..64)
                .map(|i| ((i + pos * 7) as f32 * 0.1).sin())
                .collect();
            let recon = cache.dequantize_key(0, pos);
            let cos = cosine_sim(&orig_key, &recon);
            assert!(cos > 0.8, "pos {pos} key cosine = {cos}");
        }
    }

    // ── dequantize_into ──────────────────────────────────────

    #[test]
    fn test_dequantize_key_into() {
        let mut cache = make_cache(64, 2, 2);
        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);

        let mut out = vec![0.0f32; 64];
        cache.dequantize_key_into(0, 0, &mut out);

        let recon = cache.dequantize_key(0, 0);
        for i in 0..64 {
            assert!(
                (out[i] - recon[i]).abs() < 1e-5,
                "mismatch at [{i}]: into={}, vec={}",
                out[i],
                recon[i]
            );
        }
    }

    #[test]
    fn test_dequantize_value_into() {
        let mut cache = make_cache(64, 3, 3);
        let value: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).cos()).collect();
        cache.store_value(0, 0, &value);

        let mut out = vec![0.0f32; 64];
        cache.dequantize_value_into(0, 0, &mut out);

        let recon = cache.dequantize_value(0, 0);
        for i in 0..64 {
            assert!((out[i] - recon[i]).abs() < 1e-5, "mismatch at [{i}]");
        }
    }

    // ── Zero vector handling ─────────────────────────────────

    #[test]
    fn test_zero_vector_handling() {
        let mut cache = make_cache(64, 2, 2);
        let zero = vec![0.0f32; 64];
        cache.store_key(0, 0, &zero);
        cache.store_value(0, 0, &zero);

        let recon_key = cache.dequantize_key(0, 0);
        let recon_val = cache.dequantize_value(0, 0);

        for (i, &v) in recon_key.iter().enumerate() {
            assert!(v.abs() < 1e-6, "key[{i}] = {v}, expected ~0");
        }
        for (i, &v) in recon_val.iter().enumerate() {
            assert!(v.abs() < 1e-6, "val[{i}] = {v}, expected ~0");
        }
    }

    // ── Reset ────────────────────────────────────────────────

    #[test]
    fn test_reset_clears() {
        let mut cache = make_cache(64, 2, 2);
        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        cache.store_key(0, 0, &key);
        assert!(cache.key_norms[0][0] > 0.0);

        cache.reset();
        assert_eq!(cache.pos, 0);
        assert!(cache.key_norms[0][0].abs() < 1e-6);
    }

    // ── Multi-layer independence ─────────────────────────────

    #[test]
    fn test_multi_layer_independence() {
        let mut cache = make_cache(64, 2, 2);
        let key0: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let key1: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).cos()).collect();

        cache.store_key(0, 0, &key0);
        cache.store_key(1, 0, &key1);

        // Same rotation/codebook per layer, but stored data is independent
        let recon0 = cache.dequantize_key(0, 0);
        let recon1 = cache.dequantize_key(1, 0);

        let cos0 = cosine_sim(&key0, &recon0);
        let cos1 = cosine_sim(&key1, &recon1);
        assert!(cos0 > 0.7, "layer 0 cosine = {cos0}");
        assert!(cos1 > 0.7, "layer 1 cosine = {cos1}");
    }

    // ── Compression metrics ──────────────────────────────────

    #[test]
    fn test_bytes_per_token() {
        let cache = make_cache(64, 2, 2);
        let bpt = cache.bytes_per_token();
        assert!(bpt > 0, "bytes per token should be positive");

        // For dim=64, n_triplets=22, key_bits=2 (dir=3,nrm=1)
        // bits per triplet = 2*3+1 = 7, total = 22*7 = 154 bits = 20 bytes per side
        // Per layer: 20 + 20 + 8 = 48 bytes
        // 2 layers: 96 bytes
        let expected = 96;
        assert_eq!(
            bpt, expected,
            "bytes per token: got {bpt}, expected {expected}"
        );
    }

    #[test]
    fn test_compression_ratio() {
        let cache = make_cache(64, 2, 2);
        let ratio = cache.compression_ratio();
        // f32 baseline: 64 * 4 * 2 * 2 = 1024 bytes
        // compressed: 96 bytes
        // ratio: 1024 / 96 ≈ 10.67
        assert!(ratio > 5.0, "compression ratio = {ratio}, expected > 5.0");
    }

    #[test]
    fn test_compression_ratio_3bit() {
        let cache = make_cache(128, 3, 3);
        let ratio = cache.compression_ratio();
        // f32 baseline: 128 * 4 * 2 * 2 = 2048 bytes
        // dir=4, nrm=2, bits_per_triplet=10, n_triplets=43
        // packed = 43*10/8 = 54 bytes per side, per layer: 54+54+8 = 116
        // 2 layers: 232 bytes
        // ratio: 2048 / 232 ≈ 8.83
        assert!(ratio > 4.0, "compression ratio = {ratio}, expected > 4.0");
    }

    // ── with_config matches new ──────────────────────────────

    #[test]
    fn test_pos_set_pos() {
        let mut cache = make_cache(64, 2, 2);
        assert_eq!(cache.pos(), 0);
        cache.set_pos(5);
        assert_eq!(cache.pos(), 5);
    }

    #[test]
    fn test_kv_dim_accessor() {
        let cache = make_cache(128, 2, 2);
        assert_eq!(cache.kv_dim(), 128);
    }

    // ── QuantizedKVCache trait ───────────────────────────────

    #[test]
    fn test_trait_store_and_dequantize() {
        let mut cache = make_cache(64, 2, 2);
        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        // Use trait methods
        let trait_cache = &mut cache as &mut dyn crate::types::QuantizedKVCache;
        trait_cache.store_key(0, 0, &key);

        let mut out = vec![0.0f32; 64];
        trait_cache.dequantize_key_into(0, 0, &mut out);

        let cos = cosine_sim(&key, &out);
        assert!(cos > 0.7, "trait key cosine = {cos}");
    }

    // ── Bit width variations ─────────────────────────────────

    #[test]
    fn test_4bit_quality() {
        let mut cache = make_cache(128, 4, 4);
        let key: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();

        cache.store_key(0, 0, &key);
        let recon = cache.dequantize_key(0, 0);
        let cos = cosine_sim(&key, &recon);

        // 4-bit should be very high quality
        assert!(cos > 0.95, "4-bit key cosine = {cos}");
    }

    #[test]
    fn test_2bit_quality() {
        let mut cache = make_cache(128, 2, 2);
        let key: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();

        cache.store_key(0, 0, &key);
        let recon = cache.dequantize_key(0, 0);
        let cos = cosine_sim(&key, &recon);

        // 2-bit is aggressive but should still be reasonable
        assert!(cos > 0.6, "2-bit key cosine = {cos}");
    }

    #[test]
    fn test_dim_not_multiple_of_3() {
        // kv_dim=65 → n_triplets=22, recomposed length=66, truncate to 65
        let mut cache = make_cache(65, 3, 3);
        let key: Vec<f32> = (0..65).map(|i| (i as f32 * 0.1).sin()).collect();

        cache.store_key(0, 0, &key);
        let recon = cache.dequantize_key(0, 0);

        assert_eq!(recon.len(), 65);
        let cos = cosine_sim(&key, &recon);
        assert!(cos > 0.8, "dim=65 cosine = {cos}");
    }

    // ── Helper ───────────────────────────────────────────────

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            return 0.0;
        }
        dot / (na * nb)
    }

    // ── SP-KV + OCTOPUS Composition Proof ────────────────────
    // Verifies SpKvQuantCache<OctopusKVCache> compiles and works end-to-end.
    // Acceptance criterion: "SpKvQuantCache<OctopusKVCache> compiles (composition proof)".

    #[cfg(all(feature = "sp_kv", feature = "octopus"))]
    #[test]
    fn test_sp_kv_octopus_composition_compiles() {
        use crate::sp_kv::types::{SpKvConfig, SpKvQuantCache};

        let octopus = make_cache(64, 3, 3);
        let sp_cfg = SpKvConfig::default();
        let mut hybrid: SpKvQuantCache<OctopusKVCache> =
            SpKvQuantCache::new(sp_cfg, octopus, 2, 32);

        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 - 3.2).sin()).collect();
        let val: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();

        // High utility → should be retained and written
        let written = hybrid.write_gated(0, &key, &val, 0.9, 0, false, 0.5);
        assert!(written, "high utility should be retained");
        assert!(hybrid.meta[0].retained[0]);
        assert_eq!(hybrid.meta[0].retained_count, 1);

        // Low utility, not in window → should be pruned
        let written = hybrid.write_gated(0, &key, &val, 0.1, 1, false, 0.5);
        assert!(!written, "low utility should be pruned");
        assert!(!hybrid.meta[0].retained[1]);
        assert_eq!(hybrid.meta[0].retained_count, 1);

        // In window → should be retained regardless of utility
        let written = hybrid.write_gated(0, &key, &val, 0.01, 2, true, 0.5);
        assert!(written, "in-window should always be retained");
        assert!(hybrid.meta[0].retained[2]);
        assert_eq!(hybrid.meta[0].retained_count, 2);
    }

    #[cfg(all(feature = "sp_kv", feature = "octopus"))]
    #[test]
    fn test_sp_kv_octopus_roundtrip() {
        use crate::sp_kv::types::{SpKvConfig, SpKvQuantCache};

        let octopus = make_cache(64, 3, 3);
        let sp_cfg = SpKvConfig::default();
        let mut hybrid: SpKvQuantCache<OctopusKVCache> =
            SpKvQuantCache::new(sp_cfg, octopus, 2, 32);

        let key: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 - 3.2).sin()).collect();
        let val: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05).cos()).collect();

        // Write with high utility so it's retained
        hybrid.write_gated(0, &key, &val, 0.99, 0, false, 0.5);

        // Dequantize through the quant backend
        let mut recon_key = vec![0.0f32; 64];
        hybrid.quant.dequantize_key_into(0, 0, &mut recon_key);

        let cos = cosine_sim(&key, &recon_key);
        assert!(cos > 0.95, "SP-KV + OCTOPUS roundtrip cosine = {cos}");

        let mut recon_val = vec![0.0f32; 64];
        hybrid.quant.dequantize_value_into(0, 0, &mut recon_val);

        let cos_v = cosine_sim(&val, &recon_val);
        assert!(
            cos_v > 0.95,
            "SP-KV + OCTOPUS value roundtrip cosine = {cos_v}"
        );
    }
}
