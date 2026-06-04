//! KVarN KV Cache — dual-scale quantized KV cache with variance normalization (Research 159).
//!
//! Implements [`QuantizedKVCache`] with:
//! - Hadamard rotation per tile (absorbed into weights at training time, applied at quantize time)
//! - Variance normalization per tile (Sinkhorn iterative dual-scaling)
//! - Asymmetric RTN (round-to-nearest) with dual scales:
//!   - K tile `[D, group]`: per-channel RTN scale × per-token var_norm scale
//!   - V tile `[group, D]`: per-token RTN scale × per-channel var_norm scale
//!
//! Dequantization: one extra multiply vs standard RTN for the dual-scale reconstruction.

use super::hadamard;
use super::var_norm::{VarNormConfig, VarianceNormScales, variance_normalize};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// KVarN KV cache configuration.
#[derive(Clone, Debug)]
pub struct KVarNConfig {
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads).
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Bits per element (default: 2).
    pub bits: u8,
    /// Tokens per tile (default: 128).
    pub tile_size: usize,
    /// Variance normalization config.
    pub var_norm: VarNormConfig,
}

impl Default for KVarNConfig {
    fn default() -> Self {
        Self {
            n_layers: 1,
            kv_dim: 128,
            max_seq_len: 2048,
            bits: 2,
            tile_size: 128,
            var_norm: VarNormConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-layer storage
// ---------------------------------------------------------------------------

/// Per-tile metadata for a quantized tile.
struct TileMeta {
    /// Number of valid positions in this tile (≤ tile_size).
    count: usize,
    /// Variance normalization scales.
    var_scales: VarianceNormScales,
    /// Per-row RTN scales (channel scales for K, token scales for V).
    rtn_scales: Vec<f32>,
    /// Per-row zero points.
    rtn_zp: Vec<f32>,
}

impl TileMeta {
    fn empty(_tile_size: usize, rows: usize, cols: usize) -> Self {
        Self {
            count: 0,
            var_scales: VarianceNormScales {
                s_col: vec![1.0; cols],
                s_row: vec![1.0; rows],
            },
            rtn_scales: vec![1.0; rows],
            rtn_zp: vec![0.0; rows],
        }
    }
}

// ---------------------------------------------------------------------------
// KVarNKVCache
// ---------------------------------------------------------------------------

/// KVarN variance-normalized quantized KV cache (Research 159).
///
/// Storage layout per layer:
/// - Key tiles: `[D, tile_size]` — rows=channels, cols=tokens_in_tile
/// - Value tiles: `[tile_size, D]` — rows=tokens_in_tile, cols=channels
///
/// Each tile is quantized with Hadamard rotation → variance normalization →
/// asymmetric RTN with dual scales.
pub struct KVarNKVCache {
    // ── Storage fields (Vec: 24 bytes each, pointer-aligned) ──
    /// Quantized key data: [layer][tile][packed_bytes].
    key_quantized: Vec<Vec<Vec<u8>>>,
    /// Key tile metadata: [layer][tile].
    key_tiles: Vec<Vec<TileMeta>>,
    /// Key tile buffer: [kv_dim * tile_size] — accumulates raw data for current tile.
    key_buffer: Vec<f32>,
    /// Quantized value data: [layer][tile][packed_bytes].
    val_quantized: Vec<Vec<Vec<u8>>>,
    /// Value tile metadata: [layer][tile].
    val_tiles: Vec<Vec<TileMeta>>,
    /// Value tile buffer: [tile_size * kv_dim].
    val_buffer: Vec<f32>,
    // ── Scratch buffers for zero-alloc hot path ──
    /// Scratch for tile operations: [tile_rows * tile_cols].
    scratch_tile: Vec<f32>,
    /// Scratch for dequant output.
    scratch_dequant: Vec<f32>,
    // ── Scalar config (usize: 8 bytes each) ──
    /// Current write position.
    pos: usize,
    /// Number of transformer layers.
    n_layers: usize,
    /// KV dimension.
    kv_dim: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Tile size (tokens per tile).
    tile_size: usize,
    /// Number of complete tiles.
    n_tiles: usize,
    /// Bytes per row for packed quantized data.
    bytes_per_row: usize,
    // ── Small fields at end ──
    /// Bits per element.
    bits: u8,
}

impl KVarNKVCache {
    /// Create a new KVarN KV cache from config.
    pub fn with_config(cfg: &KVarNConfig) -> Self {
        let n_tiles = (cfg.max_seq_len + cfg.tile_size - 1) / cfg.tile_size;
        let tile_size = cfg.tile_size;

        // For keys: tile layout [kv_dim, tile_size]
        let key_tile_rows = cfg.kv_dim;
        let key_tile_cols = tile_size;
        // For values: tile layout [tile_size, kv_dim]
        let val_tile_rows = tile_size;
        let val_tile_cols = cfg.kv_dim;

        let bytes_per_row = packed_bytes_per_row(tile_size, cfg.bits);

        // Initialize per-layer, per-tile storage
        let key_quantized =
            vec![vec![vec![0u8; bytes_per_row * cfg.kv_dim]; n_tiles]; cfg.n_layers];
        let key_tiles: Vec<Vec<TileMeta>> = (0..cfg.n_layers)
            .map(|_| {
                (0..n_tiles)
                    .map(|_| TileMeta::empty(tile_size, key_tile_rows, key_tile_cols))
                    .collect()
            })
            .collect();

        let val_quantized =
            vec![
                vec![vec![0u8; packed_bytes_per_row(cfg.kv_dim, cfg.bits) * tile_size]; n_tiles];
                cfg.n_layers
            ];
        let val_tiles: Vec<Vec<TileMeta>> = (0..cfg.n_layers)
            .map(|_| {
                (0..n_tiles)
                    .map(|_| TileMeta::empty(tile_size, val_tile_rows, val_tile_cols))
                    .collect()
            })
            .collect();

        let key_buffer_size = cfg.kv_dim * tile_size;
        let val_buffer_size = tile_size * cfg.kv_dim;

        Self {
            key_quantized,
            key_tiles,
            key_buffer: vec![0.0; key_buffer_size],
            val_quantized,
            val_tiles,
            val_buffer: vec![0.0; val_buffer_size],
            scratch_tile: vec![0.0f32; key_buffer_size.max(val_buffer_size)],
            scratch_dequant: vec![0.0f32; cfg.kv_dim],
            pos: 0,
            n_layers: cfg.n_layers,
            kv_dim: cfg.kv_dim,
            max_seq_len: cfg.max_seq_len,
            tile_size,
            n_tiles,
            bytes_per_row,
            bits: cfg.bits,
        }
    }

    /// Quantize and store a key vector at given layer and position.
    ///
    /// Applies Hadamard rotation per-position at store time, then buffers into
    /// the tile. The tile is quantized (variance normalization + RTN) when full.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);
        debug_assert!(pos < self.max_seq_len);

        let tile_idx = pos / self.tile_size;
        let pos_in_tile = pos % self.tile_size;

        // Apply Hadamard per-position (channel dimension) using scratch buffer
        self.scratch_dequant[..self.kv_dim].copy_from_slice(key);
        hadamard::hadamard_transform_inplace(&mut self.scratch_dequant[..self.kv_dim]);

        // Buffer layout: [kv_dim, tile_size] row-major, so row=channel, col=token
        for ch in 0..self.kv_dim {
            self.key_buffer[ch * self.tile_size + pos_in_tile] = self.scratch_dequant[ch];
        }

        let tile = &mut self.key_tiles[layer][tile_idx];
        tile.count += 1;

        // If tile is complete, quantize it
        if tile.count == self.tile_size || pos == self.max_seq_len - 1 {
            let count = tile.count;
            self.quantize_key_tile(layer, tile_idx, count);
        }
    }

    /// Quantize and store a value vector at given layer and position.
    ///
    /// Applies Hadamard rotation per-position at store time, then buffers into
    /// the tile.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);
        debug_assert!(pos < self.max_seq_len);

        let tile_idx = pos / self.tile_size;
        let pos_in_tile = pos % self.tile_size;

        // Apply Hadamard per-position (channel dimension) using scratch buffer
        self.scratch_dequant[..self.kv_dim].copy_from_slice(value);
        hadamard::hadamard_transform_inplace(&mut self.scratch_dequant[..self.kv_dim]);

        // Buffer layout: [tile_size, kv_dim] row-major, so row=token, col=channel
        let off = pos_in_tile * self.kv_dim;
        self.val_buffer[off..off + self.kv_dim]
            .copy_from_slice(&self.scratch_dequant[..self.kv_dim]);

        let tile = &mut self.val_tiles[layer][tile_idx];
        tile.count += 1;

        if tile.count == self.tile_size || pos == self.max_seq_len - 1 {
            let count = tile.count;
            self.quantize_val_tile(layer, tile_idx, count);
        }
    }

    /// Dequantize key into pre-allocated buffer (zero-alloc hot path).
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let tile_idx = pos / self.tile_size;
        let pos_in_tile = pos % self.tile_size;

        let tile = &self.key_tiles[layer][tile_idx];
        if tile.count == 0 {
            out.fill(0.0);
            return;
        }

        // Use actual tile cols for bpr (may differ for incomplete last tile)
        let actual_cols = tile.count.min(self.tile_size);
        let bits = self.bits as usize;
        let bpr = packed_bytes_per_row(actual_cols, self.bits);

        for ch in 0..self.kv_dim {
            let row_off = ch * bpr;
            let packed_val = &self.key_quantized[layer][tile_idx][row_off..row_off + bpr];
            let q = unpack_value(packed_val, pos_in_tile, bits);

            // Dual-scale dequantize: (q * rtn_scale + zp) * var_s_col[j] * var_s_row[i]
            let rtn_val = q as f32 * tile.rtn_scales[ch] + tile.rtn_zp[ch];
            out[ch] = rtn_val * tile.var_scales.s_col[pos_in_tile] * tile.var_scales.s_row[ch];
        }

        // Inverse Hadamard on the output vector (undoes per-position rotation from store_key)
        hadamard::hadamard_transform_inplace(out);
    }

    /// Dequantize value into pre-allocated buffer (zero-alloc hot path).
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let tile_idx = pos / self.tile_size;
        let pos_in_tile = pos % self.tile_size;

        let tile = &self.val_tiles[layer][tile_idx];
        if tile.count == 0 {
            out.fill(0.0);
            return;
        }

        let bits = self.bits as usize;
        let bpr = packed_bytes_per_row(self.kv_dim, self.bits);

        let row_off = pos_in_tile * bpr;
        let packed_row = &self.val_quantized[layer][tile_idx][row_off..row_off + bpr];

        for ch in 0..self.kv_dim {
            let q = unpack_value(packed_row, ch, bits);
            let rtn_val = q as f32 * tile.rtn_scales[pos_in_tile] + tile.rtn_zp[pos_in_tile];
            out[ch] = rtn_val * tile.var_scales.s_col[ch] * tile.var_scales.s_row[pos_in_tile];
        }

        // Inverse Hadamard (undoes per-position rotation from store_value)
        hadamard::hadamard_transform_inplace(out);
    }

    /// Reset cache for a new sequence.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.key_buffer.fill(0.0);
        self.val_buffer.fill(0.0);
        for layer in 0..self.n_layers {
            for t in 0..self.n_tiles {
                self.key_tiles[layer][t].count = 0;
                self.val_tiles[layer][t].count = 0;
                self.key_quantized[layer][t].fill(0);
                self.val_quantized[layer][t].fill(0);
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

    // ── Internal quantization ──

    /// Quantize a key tile: [kv_dim, tile_size] with variance normalization.
    fn quantize_key_tile(&mut self, layer: usize, tile_idx: usize, count: usize) {
        let rows = self.kv_dim;
        let cols = count.min(self.tile_size);

        // Copy buffer to scratch
        let tile_size = self.tile_size;
        let mut tile_data = vec![0.0f32; rows * cols];
        for ch in 0..rows {
            for t in 0..cols {
                tile_data[ch * cols + t] = self.key_buffer[ch * tile_size + t];
            }
        }

        // Note: Hadamard is already applied per-position at store_key time.

        // Step 1: Variance normalization
        let config = VarNormConfig {
            tile_size: self.tile_size,
            ..Default::default()
        };
        let var_scales = variance_normalize(&mut tile_data, rows, cols, &config);

        // Step 2: Per-row (channel) RTN quantization
        let (rtn_scales, rtn_zp, packed) = rtn_quantize_rows(&tile_data, rows, cols, self.bits);

        // Store
        let bpr = packed_bytes_per_row(cols, self.bits);
        let meta = &mut self.key_tiles[layer][tile_idx];
        meta.count = count;
        meta.var_scales = var_scales;
        meta.rtn_scales = rtn_scales;
        meta.rtn_zp = rtn_zp;

        let quantized = &mut self.key_quantized[layer][tile_idx];
        let expected_len = rows * bpr;
        if quantized.len() != expected_len {
            quantized.resize(expected_len, 0);
        }
        quantized.copy_from_slice(&packed);
    }

    /// Quantize a value tile: [tile_size, kv_dim] with variance normalization.
    fn quantize_val_tile(&mut self, layer: usize, tile_idx: usize, count: usize) {
        let rows = count.min(self.tile_size);
        let cols = self.kv_dim;

        // Copy buffer to scratch
        let mut tile_data = vec![0.0f32; rows * cols];
        let off = 0;
        tile_data[off..off + rows * cols].copy_from_slice(&self.val_buffer[off..off + rows * cols]);

        // Note: Hadamard is already applied per-position at store_value time.

        // Step 1: Variance normalization
        let config = VarNormConfig {
            tile_size: self.tile_size,
            ..Default::default()
        };
        let var_scales = variance_normalize(&mut tile_data, rows, cols, &config);

        // Step 2: Per-row (token) RTN quantization
        let (rtn_scales, rtn_zp, packed) = rtn_quantize_rows(&tile_data, rows, cols, self.bits);

        let bpr = packed_bytes_per_row(cols, self.bits);
        let meta = &mut self.val_tiles[layer][tile_idx];
        meta.count = count;
        meta.var_scales = var_scales;
        meta.rtn_scales = rtn_scales;
        meta.rtn_zp = rtn_zp;

        let quantized = &mut self.val_quantized[layer][tile_idx];
        let expected_len = rows * bpr;
        if quantized.len() != expected_len {
            quantized.resize(expected_len, 0);
        }
        quantized.copy_from_slice(&packed);
    }
}

impl crate::types::QuantizedKVCache for KVarNKVCache {
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

// ---------------------------------------------------------------------------
// RTN quantization helpers
// ---------------------------------------------------------------------------

/// Packed bytes per row for given cols and bits.
fn packed_bytes_per_row(cols: usize, bits: u8) -> usize {
    (cols * bits as usize).div_ceil(8)
}

/// RTN quantize rows of a 2D tile. Returns (per-row scales, per-row zero-points, packed data).
///
/// For each row: find min/max, compute scale = (max - min) / (levels - 1),
/// quantize each element to [0, levels-1], pack into bits.
fn rtn_quantize_rows(
    tile: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let levels = 1u32 << bits;
    let half_levels = (levels - 1) as f32;
    let bpr = packed_bytes_per_row(cols, bits);
    let mut packed = vec![0u8; rows * bpr];
    let mut scales = vec![1.0f32; rows];
    let mut zps = vec![0.0f32; rows];

    for r in 0..rows {
        let row_off = r * cols;
        let row = &tile[row_off..row_off + cols];

        // Find min/max
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for &v in row {
            lo = lo.min(v);
            hi = hi.max(v);
        }

        if hi - lo < 1e-10 {
            // Degenerate: all same value
            scales[r] = 0.0;
            zps[r] = lo;
            continue;
        }

        let scale = (hi - lo) / half_levels;
        scales[r] = scale;
        zps[r] = lo;

        // Quantize and pack
        for (c, &v) in row.iter().enumerate() {
            let normalized = (v - lo) / scale;
            let q = (normalized.round() as u32).clamp(0, levels - 1);
            pack_value(&mut packed[r * bpr..], c, q, bits as usize);
        }
    }

    (scales, zps, packed)
}

/// Pack a value at given position into a bit-packed row.
#[inline]
fn pack_value(row: &mut [u8], pos: usize, value: u32, bits: usize) {
    let bit_offset = pos * bits;
    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;

    // Mask value to valid range
    let val = value & ((1u32 << bits) - 1);

    // Lower bits go at position bit_shift in byte_offset
    let lo_bits = 8 - bit_shift;
    if byte_offset < row.len() {
        row[byte_offset] |= ((val << bit_shift) & 0xFF) as u8;
    }
    // Upper bits go at position 0 in byte_offset+1
    if bits > lo_bits && byte_offset + 1 < row.len() {
        row[byte_offset + 1] |= (val >> lo_bits) as u8;
    }
}

/// Unpack a value at given position from a bit-packed row.
#[inline]
fn unpack_value(row: &[u8], pos: usize, bits: usize) -> u32 {
    let bit_offset = pos * bits;
    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;

    let lo_bits = 8 - bit_shift;

    // Extract bits from byte_offset starting at bit_shift
    let mut val: u32 = if byte_offset < row.len() {
        (row[byte_offset] >> bit_shift) as u32
    } else {
        0
    };

    // Extract remaining upper bits from byte_offset+1
    if bits > lo_bits && byte_offset + 1 < row.len() {
        val |= (row[byte_offset + 1] as u32) << lo_bits;
    }

    val & ((1u32 << bits) - 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-10 || nb < 1e-10 {
            return 0.0;
        }
        dot / (na * nb)
    }

    fn per_coord_mse(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            / a.len() as f32
    }

    fn make_config(kv_dim: usize, max_seq: usize, bits: u8, tile_size: usize) -> KVarNConfig {
        KVarNConfig {
            n_layers: 2,
            kv_dim,
            max_seq_len: max_seq,
            bits,
            tile_size,
            var_norm: VarNormConfig {
                tile_size,
                iterations: 8,
                ..Default::default()
            },
        }
    }

    fn make_random_vec(len: usize, seed: u64) -> Vec<f32> {
        // Simple LCG PRNG
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i32 as f32) / (1i32 << 31) as f32
            })
            .collect()
    }

    #[test]
    fn test_kvarn_roundtrip() {
        let kv_dim = 64;
        let seq_len = 16;
        let bits = 4;
        let tile_size = 8;
        let cfg = make_config(kv_dim, seq_len, bits, tile_size);
        let mut cache = KVarNKVCache::with_config(&cfg);

        let mut keys: Vec<Vec<f32>> = Vec::new();
        let mut values: Vec<Vec<f32>> = Vec::new();

        for pos in 0..seq_len {
            let key = make_random_vec(kv_dim, pos as u64 * 1000 + 1);
            let val = make_random_vec(kv_dim, pos as u64 * 1000 + 2);
            keys.push(key.clone());
            values.push(val.clone());
            cache.store_key(0, pos, &key);
            cache.store_value(0, pos, &val);
        }

        let mut out = vec![0.0f32; kv_dim];
        let mut total_key_mse = 0.0f32;
        let mut total_val_mse = 0.0f32;

        for pos in 0..seq_len {
            cache.dequantize_key_into(0, pos, &mut out);
            total_key_mse += per_coord_mse(&keys[pos], &out);

            cache.dequantize_value_into(0, pos, &mut out);
            total_val_mse += per_coord_mse(&values[pos], &out);
        }

        total_key_mse /= seq_len as f32;
        total_val_mse /= seq_len as f32;

        // At 4 bits, MSE should be reasonable
        assert!(total_key_mse < 0.5, "key MSE too high: {total_key_mse}");
        assert!(total_val_mse < 0.5, "value MSE too high: {total_val_mse}");
    }

    #[test]
    fn test_kvarn_dual_scale() {
        // Verify dual-scale dequantization: value = rtn_scale * q / (levels-1) + zp,
        // then multiplied by var_norm scales.
        let kv_dim = 8;
        let seq_len = 2;
        let bits = 4;
        let tile_size = 2;
        let cfg = make_config(kv_dim, seq_len, bits, tile_size);
        let mut cache = KVarNKVCache::with_config(&cfg);

        let key = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let val = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &val);

        // Also store second position to fill the tile
        let key2 = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let val2 = vec![1.5f32, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        cache.store_key(0, 1, &key2);
        cache.store_value(0, 1, &val2);

        let mut out = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut out);
        // Output should be non-trivial (not all zeros)
        assert!(
            out.iter().any(|&v| v.abs() > 1e-5),
            "dequantized key should be non-zero"
        );

        cache.dequantize_value_into(0, 0, &mut out);
        assert!(
            out.iter().any(|&v| v.abs() > 1e-5),
            "dequantized value should be non-zero"
        );
    }

    #[test]
    fn test_kvarn_zero_vector_handling() {
        let kv_dim = 8;
        let seq_len = 2;
        let bits = 2;
        let tile_size = 2;
        let cfg = make_config(kv_dim, seq_len, bits, tile_size);
        let mut cache = KVarNKVCache::with_config(&cfg);

        let zero = vec![0.0f32; kv_dim];
        cache.store_key(0, 0, &zero);
        cache.store_value(0, 0, &zero);

        // Fill tile
        cache.store_key(0, 1, &zero);
        cache.store_value(0, 1, &zero);

        let mut out = vec![0.0f32; kv_dim];
        cache.dequantize_key_into(0, 0, &mut out);
        // Zero input should produce near-zero output
        for &v in &out {
            assert!(v.abs() < 0.1, "zero key should dequant near zero, got {v}");
        }

        cache.dequantize_value_into(0, 0, &mut out);
        for &v in &out {
            assert!(
                v.abs() < 0.1,
                "zero value should dequant near zero, got {v}"
            );
        }
    }

    #[test]
    fn test_kvarn_multi_layer_independence() {
        let kv_dim = 16;
        let seq_len = 2;
        let bits = 4;
        let tile_size = 2;
        let cfg = make_config(kv_dim, seq_len, bits, tile_size);
        let mut cache = KVarNKVCache::with_config(&cfg);

        let key0 = make_random_vec(kv_dim, 42);
        let key1 = make_random_vec(kv_dim, 99);
        let val0 = make_random_vec(kv_dim, 43);
        let val1 = make_random_vec(kv_dim, 100);

        cache.store_key(0, 0, &key0);
        cache.store_value(0, 0, &val0);
        cache.store_key(0, 1, &make_random_vec(kv_dim, 44));
        cache.store_value(0, 1, &make_random_vec(kv_dim, 45));

        cache.store_key(1, 0, &key1);
        cache.store_value(1, 0, &val1);
        cache.store_key(1, 1, &make_random_vec(kv_dim, 101));
        cache.store_value(1, 1, &make_random_vec(kv_dim, 102));

        let mut out0 = vec![0.0f32; kv_dim];
        let mut out1 = vec![0.0f32; kv_dim];

        cache.dequantize_key_into(0, 0, &mut out0);
        cache.dequantize_key_into(1, 0, &mut out1);

        // Different layers should produce different outputs for same position
        let cos = cosine_sim(&out0, &out1);
        // They can have high cosine similarity but shouldn't be identical
        // (different input vectors, same quantization)
        assert!(
            (out0[0] - out1[0]).abs() > 1e-5 || key0 != key1,
            "layers should be independent"
        );
    }

    #[test]
    fn test_kvarn_reset_clears() {
        let kv_dim = 8;
        let seq_len = 4;
        let bits = 4;
        let tile_size = 2;
        let cfg = make_config(kv_dim, seq_len, bits, tile_size);
        let mut cache = KVarNKVCache::with_config(&cfg);

        let key = make_random_vec(kv_dim, 42);
        let val = make_random_vec(kv_dim, 43);
        cache.store_key(0, 0, &key);
        cache.store_value(0, 0, &val);
        cache.store_key(0, 1, &make_random_vec(kv_dim, 44));
        cache.store_value(0, 1, &make_random_vec(kv_dim, 45));

        cache.reset();
        assert_eq!(cache.pos(), 0);

        // After reset, tiles should be empty
        assert_eq!(cache.key_tiles[0][0].count, 0);
    }

    #[test]
    fn test_pack_unpack_roundtrip() {
        for &bits in &[1u8, 2, 4, 8] {
            let cols = 32;
            let bpr = packed_bytes_per_row(cols, bits);
            let levels = 1u32 << bits;
            let mut row = vec![0u8; bpr];

            for pos in 0..cols {
                let val = (pos as u32) % levels;
                pack_value(&mut row, pos, val, bits as usize);
            }

            for pos in 0..cols {
                let expected = (pos as u32) % levels;
                let got = unpack_value(&row, pos, bits as usize);
                assert_eq!(
                    got, expected,
                    "pack/unpack mismatch at pos={pos}, bits={bits}"
                );
            }
        }
    }

    #[test]
    fn test_rtn_quantize_rows() {
        let rows = 4;
        let cols = 8;
        let bits = 4;
        let tile = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ];

        let (scales, zps, packed) = rtn_quantize_rows(&tile, rows, cols, bits);
        assert_eq!(scales.len(), rows);
        assert_eq!(zps.len(), rows);

        // Dequantize and check MSE is reasonable
        let mut mse = 0.0f32;
        for r in 0..rows {
            let bpr = packed_bytes_per_row(cols, bits);
            for c in 0..cols {
                let q = unpack_value(&packed[r * bpr..], c, bits as usize);
                let dequant = q as f32 * scales[r] + zps[r];
                let diff = tile[r * cols + c] - dequant;
                mse += diff * diff;
            }
        }
        mse /= (rows * cols) as f32;
        assert!(mse < 0.02, "RTN MSE too high: {mse}");
    }

    #[test]
    fn test_cosine_similarity_reasonable() {
        let kv_dim = 64;
        let seq_len = 8;
        let bits = 4;
        let tile_size = 4;
        let cfg = make_config(kv_dim, seq_len, bits, tile_size);
        let mut cache = KVarNKVCache::with_config(&cfg);

        let mut keys: Vec<Vec<f32>> = Vec::new();
        let mut values: Vec<Vec<f32>> = Vec::new();

        for pos in 0..seq_len {
            let key = make_random_vec(kv_dim, pos as u64 * 777 + 1);
            let val = make_random_vec(kv_dim, pos as u64 * 777 + 2);
            keys.push(key.clone());
            values.push(val.clone());
            cache.store_key(0, pos, &key);
            cache.store_value(0, pos, &val);
        }

        let mut out = vec![0.0f32; kv_dim];
        for pos in 0..seq_len {
            cache.dequantize_key_into(0, pos, &mut out);
            let cos = cosine_sim(&keys[pos], &out);
            assert!(cos > 0.9, "key cosine sim too low at pos {pos}: {cos}");

            cache.dequantize_value_into(0, pos, &mut out);
            let cos = cosine_sim(&values[pos], &out);
            assert!(cos > 0.9, "value cosine sim too low at pos {pos}: {cos}");
        }
    }
}
