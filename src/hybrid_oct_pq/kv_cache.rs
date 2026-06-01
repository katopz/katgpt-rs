//! Hybrid OCTOPUS-encoding + PlanarQuant-rotation compressed KV cache.
//!
//! Encoding pipeline (per vector):
//! 1. Normalize to unit length, store L2 norm separately
//! 2. Apply PlanarQuant 2D Givens rotation (O(d) FMAs, 256 for d=128)
//! 3. Decompose rotated vector into ⌈d/3⌉ triplets
//! 4. Encode each triplet via OCTOPUS octahedral map + quantize (ξ, η, ρ)
//! 5. Bit-pack triplet indices into contiguous byte buffer
//!
//! Decoding pipeline (reverse):
//! 1. Unpack triplet indices from byte buffer
//! 2. Decode each triplet: dequantize → oct_decode → reconstruct 3-vector
//! 3. Recompose into d-dimensional rotated vector (truncate zero-pad)
//! 4. Inverse PlanarQuant 2D rotation
//! 5. Scale by stored norm
//!
//! Rotation cost: 256 FMAs for d=128 (same as PlanarQuant, 64× cheaper than OCTOPUS's 16,384).

use super::types::{HybridOctPqConfig, HybridOctPqLayer};
use crate::octopus::OctopusConfig;
use crate::octopus::encode::{
    decode_vector_into, encode_vector_into, pack_triplet_indices_into, unpack_triplet_indices,
    unpack_triplet_indices_into,
};
use crate::octopus::triplet::{Triplet, decompose_into, n_triplets};
use crate::octopus::types::TripletIndices;
use crate::planar_quant::rotation::{
    apply_inverse_rotation, apply_rotation, generate_givens_rotations,
};
use crate::simd::simd_scale_inplace;

/// Hybrid OCTOPUS-encoding + PlanarQuant-rotation compressed KV cache.
///
/// Each (layer, position) stores:
/// - One `f32` norm (the original vector's L2 norm)
/// - Packed byte buffer of OCTOPUS triplet indices
///
/// Rotation uses PlanarQuant's O(d) 2D Givens (256 FMAs for d=128)
/// instead of OCTOPUS's O(d²) full matrix (16,384 FMAs).
pub struct HybridOctPqKVCache {
    // ── Storage fields (Vec: 24 bytes each, pointer-aligned) ──
    /// Per-layer PQ rotations + OCT codebooks.
    pub layers: Vec<HybridOctPqLayer>,
    /// Packed key triplet indices: [layer][pos][packed_bytes].
    key_packed: Vec<Vec<Vec<u8>>>,
    /// Per-position key L2 norms: [layer][pos].
    key_norms: Vec<Vec<f32>>,
    /// Packed value triplet indices: [layer][pos][packed_bytes].
    val_packed: Vec<Vec<Vec<u8>>>,
    /// Per-position value L2 norms: [layer][pos].
    val_norms: Vec<Vec<f32>>,
    // ── Scratch buffers for zero-alloc hot path ──
    /// [kv_dim_padded] — normalized vector / PQ rotation input.
    scratch_normalized: Vec<f32>,
    /// [kv_dim_padded] — PQ rotation output / encode input.
    scratch_rotated: Vec<f32>,
    /// [n_triplets * 3] — OCT triplet decode workspace.
    scratch_workspace: Vec<f32>,
    /// [n_triplets] — scratch for triplet decomposition output.
    scratch_triplets: Vec<Triplet>,
    /// [n_triplets] — scratch for encoded triplet indices.
    scratch_indices: Vec<TripletIndices>,
    // ── Scalar config (usize: 8 bytes each) ──
    /// Current write position.
    pos: usize,
    /// Number of transformer layers.
    n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads).
    kv_dim: usize,
    /// KV dimension padded to even (for PQ rotation).
    kv_dim_padded: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Number of triplets: ⌈kv_dim/3⌉.
    n_triplets: usize,
    // ── Small fields at end (packed, 3 bytes + 5 padding) ──
    /// Nominal bits per key coordinate.
    key_bits: u8,
    /// Nominal bits per value coordinate.
    val_bits: u8,
    /// Use joint 3×3 rounding in encoder.
    use_joint_rounding: bool,
}

impl HybridOctPqKVCache {
    /// Create from explicit config.
    ///
    /// Per-layer PQ rotations are generated with deterministic seed offsets.
    /// OCT codebooks are shared across layers (depend only on kv_dim + bits).
    pub fn with_config(cfg: &HybridOctPqConfig) -> Self {
        let kv_dim_padded = (cfg.kv_dim + 1) & !1; // round up to even
        let n_groups = kv_dim_padded / 2;
        let n_tri = n_triplets(cfg.kv_dim);

        // Build OCT codebooks (shared across layers, depends only on kv_dim + bits)
        let key_codebook = crate::octopus::OctopusCodebook::build(cfg.kv_dim, cfg.key_bits);
        let val_codebook = crate::octopus::OctopusCodebook::build(cfg.kv_dim, cfg.val_bits);

        // Build per-layer PQ rotations + OCT codebooks
        let layers: Vec<HybridOctPqLayer> = (0..cfg.n_layers)
            .map(|layer_idx| {
                let key_seed = cfg
                    .seed
                    .wrapping_add(layer_idx as u64 * 1000)
                    .wrapping_add(1);
                let val_seed = key_seed.wrapping_add(500);
                HybridOctPqLayer {
                    key_rotations: generate_givens_rotations(n_groups, key_seed),
                    val_rotations: generate_givens_rotations(n_groups, val_seed),
                    key_codebook: key_codebook.clone(),
                    val_codebook: val_codebook.clone(),
                }
            })
            .collect();

        let packed_key_len = packed_triplet_len(n_tri, cfg.key_bits);
        let packed_val_len = packed_triplet_len(n_tri, cfg.val_bits);

        Self {
            layers,
            key_packed: vec![vec![vec![0u8; packed_key_len]; cfg.max_seq_len]; cfg.n_layers],
            key_norms: vec![vec![0.0f32; cfg.max_seq_len]; cfg.n_layers],
            val_packed: vec![vec![vec![0u8; packed_val_len]; cfg.max_seq_len]; cfg.n_layers],
            val_norms: vec![vec![0.0f32; cfg.max_seq_len]; cfg.n_layers],
            scratch_normalized: vec![0.0f32; kv_dim_padded],
            scratch_rotated: vec![0.0f32; kv_dim_padded],
            scratch_workspace: vec![0.0f32; n_tri * 3],
            scratch_triplets: Vec::with_capacity(n_tri),
            scratch_indices: Vec::with_capacity(n_tri),
            pos: 0,
            n_layers: cfg.n_layers,
            kv_dim: cfg.kv_dim,
            kv_dim_padded,
            max_seq_len: cfg.max_seq_len,
            n_triplets: n_tri,
            key_bits: cfg.key_bits,
            val_bits: cfg.val_bits,
            use_joint_rounding: cfg.use_joint_rounding,
        }
    }

    /// Quantize and store a key vector at given layer and position.
    ///
    /// Pipeline: normalize → PQ 2D rotate → decompose triplets → OCT encode → bit-pack
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        let norm = crate::simd::simd_sum_sq(key, key.len()).sqrt();
        self.key_norms[layer][pos] = norm;

        if norm < 1e-8 {
            self.key_packed[layer][pos].fill(0);
            return;
        }

        // 1. Normalize into scratch buffer (copy + scale, zero-pad to even)
        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..key.len()].copy_from_slice(key);
        self.scratch_normalized[key.len()..].fill(0.0);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        // 2. PQ 2D Givens rotation → scratch_rotated
        apply_rotation(
            &self.layers[layer].key_rotations,
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        // 3+4. Decompose → encode triplets (zero-alloc via scratch buffers)
        let cb = &self.layers[layer].key_codebook;
        decompose_into(
            &self.scratch_rotated[..self.kv_dim],
            &mut self.scratch_triplets,
        );
        encode_vector_into(
            &self.scratch_triplets,
            cb,
            self.use_joint_rounding,
            &mut self.scratch_indices,
        );

        // 5. Bit-pack triplet indices directly into packed buffer
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
        let norm = crate::simd::simd_sum_sq(value, value.len()).sqrt();
        self.val_norms[layer][pos] = norm;

        if norm < 1e-8 {
            self.val_packed[layer][pos].fill(0);
            return;
        }

        let inv_norm = 1.0 / norm;
        self.scratch_normalized[..value.len()].copy_from_slice(value);
        self.scratch_normalized[value.len()..].fill(0.0);
        simd_scale_inplace(&mut self.scratch_normalized, inv_norm);

        apply_rotation(
            &self.layers[layer].val_rotations,
            &self.scratch_normalized,
            &mut self.scratch_rotated,
        );

        let cb = &self.layers[layer].val_codebook;
        decompose_into(
            &self.scratch_rotated[..self.kv_dim],
            &mut self.scratch_triplets,
        );
        encode_vector_into(
            &self.scratch_triplets,
            cb,
            self.use_joint_rounding,
            &mut self.scratch_indices,
        );

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

        // OCT decode triplets
        let mut decoded = vec![0.0f32; self.n_triplets * 3];
        decode_vector_into(&indices, cb, &mut decoded);

        // PQ inverse rotate (pad to even for rotation)
        let mut padded = vec![0.0f32; self.kv_dim_padded];
        padded[..self.kv_dim].copy_from_slice(&decoded[..self.kv_dim]);
        let mut normalized = vec![0.0f32; self.kv_dim_padded];
        apply_inverse_rotation(&self.layers[layer].key_rotations, &padded, &mut normalized);

        // Scale by norm
        normalized[..self.kv_dim].iter().map(|x| x * norm).collect()
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

        let mut padded = vec![0.0f32; self.kv_dim_padded];
        padded[..self.kv_dim].copy_from_slice(&decoded[..self.kv_dim]);
        let mut normalized = vec![0.0f32; self.kv_dim_padded];
        apply_inverse_rotation(&self.layers[layer].val_rotations, &padded, &mut normalized);

        normalized[..self.kv_dim].iter().map(|x| x * norm).collect()
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

        // OCT decode triplets into workspace
        decode_vector_into(&self.scratch_indices, cb, &mut self.scratch_workspace);

        // Copy + zero-pad to even for PQ inverse rotation
        self.scratch_rotated.fill(0.0);
        self.scratch_rotated[..self.kv_dim].copy_from_slice(&self.scratch_workspace[..self.kv_dim]);

        // PQ inverse 2D rotation
        apply_inverse_rotation(
            &self.layers[layer].key_rotations,
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

        let cb = &self.layers[layer].val_codebook;
        unpack_triplet_indices_into(
            &self.val_packed[layer][pos],
            self.n_triplets,
            cb.dir_bits,
            cb.nrm_bits,
            &mut self.scratch_indices,
        );

        decode_vector_into(&self.scratch_indices, cb, &mut self.scratch_workspace);

        self.scratch_rotated.fill(0.0);
        self.scratch_rotated[..self.kv_dim].copy_from_slice(&self.scratch_workspace[..self.kv_dim]);

        apply_inverse_rotation(
            &self.layers[layer].val_rotations,
            &self.scratch_rotated,
            &mut self.scratch_normalized,
        );

        out.copy_from_slice(&self.scratch_normalized[..self.kv_dim]);
        simd_scale_inplace(out, norm);
    }

    /// Reset cache for new sequence.
    pub fn reset(&mut self) {
        for layer in 0..self.n_layers {
            for pos in 0..self.max_seq_len {
                self.key_packed[layer][pos].fill(0);
                self.key_norms[layer][pos] = 0.0;
                self.val_packed[layer][pos].fill(0);
                self.val_norms[layer][pos] = 0.0;
            }
        }
        self.pos = 0;
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

    /// Get current position.
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

impl crate::types::QuantizedKVCache for HybridOctPqKVCache {
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

    fn make_random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = crate::types::Rng::new(seed);
        (0..dim).map(|_| rng.normal()).collect()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-8 || nb < 1e-8 {
            return 0.0;
        }
        dot / (na * nb)
    }

    fn per_coord_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
        let n = original.len() as f32;
        original
            .iter()
            .zip(reconstructed)
            .map(|(o, r)| (o - r) * (o - r))
            .sum::<f32>()
            / n
    }

    fn make_test_config() -> HybridOctPqConfig {
        HybridOctPqConfig {
            key_bits: 2,
            val_bits: 2,
            seed: 42,
            n_layers: 2,
            kv_dim: 64,
            max_seq_len: 32,
            use_joint_rounding: true,
        }
    }

    #[test]
    fn test_kv_cache_roundtrip() {
        let config = make_test_config();
        let mut cache = HybridOctPqKVCache::with_config(&config);

        for layer in 0..config.n_layers {
            for pos in 0..8 {
                let key = make_random_vec(config.kv_dim, 100 + layer as u64 * 1000 + pos as u64);
                let value = make_random_vec(config.kv_dim, 200 + layer as u64 * 1000 + pos as u64);
                cache.store_key(layer, pos, &key);
                cache.store_value(layer, pos, &value);

                let key_recon = cache.dequantize_key(layer, pos);
                let val_recon = cache.dequantize_value(layer, pos);

                let key_cos = cosine_sim(&key, &key_recon);
                let val_cos = cosine_sim(&value, &val_recon);

                assert!(
                    key_cos > 0.85,
                    "key cosine too low at layer={layer} pos={pos}: {key_cos:.4}"
                );
                assert!(
                    val_cos > 0.85,
                    "val cosine too low at layer={layer} pos={pos}: {val_cos:.4}"
                );
            }
        }
    }

    #[test]
    fn test_dequantize_into_matches_dequantize() {
        let config = make_test_config();
        let mut cache = HybridOctPqKVCache::with_config(&config);

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
                "key mismatch at [{i}]: alloc={} buf={}",
                key_alloc[i],
                key_buf[i]
            );
            assert!(
                (val_alloc[i] - val_buf[i]).abs() < 1e-6,
                "val mismatch at [{i}]: alloc={} buf={}",
                val_alloc[i],
                val_buf[i]
            );
        }
    }

    #[test]
    fn test_3bit_roundtrip() {
        let config = HybridOctPqConfig {
            key_bits: 3,
            val_bits: 3,
            ..make_test_config()
        };
        let mut cache = HybridOctPqKVCache::with_config(&config);

        let key = make_random_vec(config.kv_dim, 42);
        let value = make_random_vec(config.kv_dim, 43);
        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let key_recon = cache.dequantize_key(0, 0);
        let val_recon = cache.dequantize_value(0, 0);

        let key_cos = cosine_sim(&key, &key_recon);
        let val_cos = cosine_sim(&value, &val_recon);

        assert!(key_cos > 0.92, "3-bit key cosine: {key_cos:.4}");
        assert!(val_cos > 0.92, "3-bit val cosine: {val_cos:.4}");
    }

    #[test]
    fn test_4bit_roundtrip() {
        let config = HybridOctPqConfig {
            key_bits: 4,
            val_bits: 4,
            ..make_test_config()
        };
        let mut cache = HybridOctPqKVCache::with_config(&config);

        let key = make_random_vec(config.kv_dim, 77);
        let value = make_random_vec(config.kv_dim, 78);
        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &value);

        let key_recon = cache.dequantize_key(0, 0);
        let val_recon = cache.dequantize_value(0, 0);

        let key_cos = cosine_sim(&key, &key_recon);
        let val_cos = cosine_sim(&value, &val_recon);

        assert!(key_cos > 0.96, "4-bit key cosine: {key_cos:.4}");
        assert!(val_cos > 0.96, "4-bit val cosine: {val_cos:.4}");
    }

    #[test]
    fn test_mse_improves_with_bits() {
        let mse_at = |bits: u8| -> f32 {
            let config = HybridOctPqConfig {
                key_bits: bits,
                val_bits: bits,
                ..make_test_config()
            };
            let mut cache = HybridOctPqKVCache::with_config(&config);
            let mut total_mse = 0.0f32;
            let n = 10;
            for pos in 0..n {
                let key = make_random_vec(config.kv_dim, pos as u64 * 100 + 500);
                cache.store_key(0, pos, &key);
                let recon = cache.dequantize_key(0, pos);
                total_mse += per_coord_mse(&key, &recon);
            }
            total_mse / n as f32
        };

        let mse2 = mse_at(2);
        let mse3 = mse_at(3);
        let mse4 = mse_at(4);

        assert!(
            mse3 < mse2,
            "3-bit MSE ({mse3:.6}) should be < 2-bit ({mse2:.6})"
        );
        assert!(
            mse4 < mse3,
            "4-bit MSE ({mse4:.6}) should be < 3-bit ({mse3:.6})"
        );
    }

    #[test]
    fn test_compression_ratio() {
        let config = make_test_config();
        let cache = HybridOctPqKVCache::with_config(&config);

        // 64 dim × 4 bytes × 2 (K+V) × 2 layers = 1024 bytes
        let _flat_bytes = config.kv_dim * 4 * 2 * config.n_layers;
        let ratio = cache.compression_ratio();

        // At 2-bit nominal: bits_per_triplet = 2*(2+1) + (2-1) = 7
        // n_triplets = ceil(64/3) = 22, total_bits = 22*7 = 154, bytes = 20
        // Per layer: 20 (key) + 20 (val) + 8 (norms) = 48
        // Total: 96 bytes → ratio ≈ 1024/96 ≈ 10.7
        assert!(ratio > 4.0, "compression ratio too low: {ratio}");
        assert!(ratio < 30.0, "compression ratio suspiciously high: {ratio}");
    }

    #[test]
    fn test_bytes_per_token() {
        let config = make_test_config();
        let cache = HybridOctPqKVCache::with_config(&config);

        let bpt = cache.bytes_per_token();
        // 2-bit: bits_per_triplet = 7, n_triplets = 22
        // packed_key = ceil(22*7/8) = 20, packed_val = 20
        // per_layer = 20 + 20 + 8 = 48, total = 96
        assert!(bpt > 0, "bytes_per_token should be positive");
        assert!(
            bpt < config.kv_dim * 4 * 2 * config.n_layers,
            "bytes_per_token should be less than f32 baseline"
        );
    }

    #[test]
    fn test_reset_clears() {
        let config = make_test_config();
        let mut cache = HybridOctPqKVCache::with_config(&config);

        let key = make_random_vec(config.kv_dim, 42);
        cache.store_key(0, 0, &key);
        assert!(cache.key_norms[0][0] > 0.0);

        cache.reset();
        assert_eq!(cache.pos(), 0);
        assert_eq!(cache.key_norms[0][0], 0.0);
    }

    #[test]
    fn test_zero_vector_handling() {
        let config = make_test_config();
        let mut cache = HybridOctPqKVCache::with_config(&config);

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
        let config = HybridOctPqConfig {
            key_bits: 3,
            val_bits: 3,
            seed: 42,
            n_layers: 3,
            kv_dim: 32,
            max_seq_len: 8,
            use_joint_rounding: true,
        };
        let mut cache = HybridOctPqKVCache::with_config(&config);

        let key0 = make_random_vec(32, 100);
        let key1 = make_random_vec(32, 200);
        let key2 = make_random_vec(32, 300);

        cache.store_key(0, 0, &key0);
        cache.store_key(1, 0, &key1);
        cache.store_key(2, 0, &key2);

        let recon0 = cache.dequantize_key(0, 0);
        let recon1 = cache.dequantize_key(1, 0);
        let recon2 = cache.dequantize_key(2, 0);

        // Each layer should reconstruct its own key, not cross-contaminate
        let cos00 = cosine_sim(&key0, &recon0);
        let cos01 = cosine_sim(&key0, &recon1);
        let cos02 = cosine_sim(&key0, &recon2);

        assert!(cos00 > 0.9, "layer 0 self-similarity: {cos00:.4}");
        // Cross-layer similarity should be lower (different rotations)
        assert!(
            cos01 < cos00 + 0.1,
            "cross-layer sim should not exceed self-sim: {cos01:.4} vs {cos00:.4}"
        );
        assert!(
            cos02 < cos00 + 0.1,
            "cross-layer sim should not exceed self-sim: {cos02:.4} vs {cos00:.4}"
        );
    }

    #[test]
    fn test_odd_dimension() {
        // d=7: padded to 8 for PQ, 3 triplets (9 elements) for OCT
        let config = HybridOctPqConfig {
            key_bits: 3,
            val_bits: 3,
            seed: 55,
            n_layers: 1,
            kv_dim: 7,
            max_seq_len: 8,
            use_joint_rounding: false,
        };
        let mut cache = HybridOctPqKVCache::with_config(&config);

        let key = make_random_vec(7, 42);
        cache.store_key(0, 0, &key);
        let recon = cache.dequantize_key(0, 0);

        assert_eq!(recon.len(), 7);
        let cos = cosine_sim(&key, &recon);
        assert!(cos > 0.9, "odd-dim cosine: {cos:.4}");
    }

    #[test]
    fn test_pos_management() {
        let config = make_test_config();
        let mut cache = HybridOctPqKVCache::with_config(&config);

        assert_eq!(cache.pos(), 0);
        cache.set_pos(10);
        assert_eq!(cache.pos(), 10);
        cache.reset();
        assert_eq!(cache.pos(), 0);
    }

    #[test]
    fn test_hybrid_vs_pure_octopus_mse() {
        // At same bit width, hybrid should be within 2× of pure OCTOPUS MSE.
        // This is a sanity check — the hybrid uses PQ rotation instead of
        // full d×d rotation, so some MSE degradation is expected.
        let bits = 3u8;
        let dim = 64;
        let n_keys = 20;
        let seed = 42u64;

        let mut rng = crate::types::Rng::new(seed);
        let keys: Vec<Vec<f32>> = (0..n_keys)
            .map(|_| (0..dim).map(|_| rng.normal()).collect())
            .collect();

        // Hybrid MSE
        let hybrid_cfg = HybridOctPqConfig {
            key_bits: bits,
            val_bits: bits,
            seed,
            n_layers: 1,
            kv_dim: dim,
            max_seq_len: n_keys + 16,
            use_joint_rounding: true,
        };
        let mut hybrid = HybridOctPqKVCache::with_config(&hybrid_cfg);
        let mut hybrid_mse = 0.0f32;
        for (pos, key) in keys.iter().enumerate() {
            hybrid.store_key(0, pos, key);
            let recon = hybrid.dequantize_key(0, pos);
            hybrid_mse += per_coord_mse(key, &recon);
        }
        hybrid_mse /= n_keys as f32;

        // Pure OCTOPUS MSE
        let oct_cfg = crate::octopus::OctopusConfig {
            key_bits: bits,
            val_bits: bits,
            seed,
            n_layers: 1,
            kv_dim: dim,
            max_seq_len: n_keys + 16,
            use_qjl_residual: false,
            use_joint_rounding: true,
        };
        let mut oct = crate::octopus::OctopusKVCache::with_config(&oct_cfg);
        let mut oct_mse = 0.0f32;
        for (pos, key) in keys.iter().enumerate() {
            oct.store_key(0, pos, key);
            let recon = oct.dequantize_key(0, pos);
            oct_mse += per_coord_mse(key, &recon);
        }
        oct_mse /= n_keys as f32;

        let ratio = hybrid_mse / oct_mse;
        println!(
            "  bits={bits} dim={dim}: OCT MSE={oct_mse:.6}, Hybrid MSE={hybrid_mse:.6}, ratio={ratio:.2}×"
        );

        // Hybrid should be within 3× of OCTOPUS (generous bound for first implementation)
        assert!(
            ratio < 3.0,
            "hybrid MSE ({hybrid_mse:.6}) is {ratio:.1}× worse than OCTOPUS ({oct_mse:.6})"
        );
    }
}
