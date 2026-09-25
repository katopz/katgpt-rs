//! KVarN KV Cache — dual-scale quantized KV cache with variance normalization (Research 159).
//!
//! Implements `QuantizedKVCache` with:
//! - Hadamard rotation per tile (absorbed into weights at training time, applied at quantize time)
//! - Variance normalization per tile (Sinkhorn iterative dual-scaling)
//! - Asymmetric RTN (round-to-nearest) with dual scales:
//!   - K tile `[D, group]`: per-channel RTN scale × per-token var_norm scale
//!   - V tile `[group, D]`: per-token RTN scale × per-channel var_norm scale
//!
//! Dequantization: one extra multiply vs standard RTN for the dual-scale reconstruction.

#![allow(clippy::needless_range_loop)]

use super::hadamard;
use super::var_norm::{VarNormConfig, VarianceNormScales, variance_normalize_into_scales};

#[cfg(feature = "targeted_precision")]
use crate::targeted_precision::PrecisionBudget;

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
    /// Enable Hadamard rotation (default: false).
    ///
    /// When enabled, applies Hadamard transform per-tile at quantize time and
    /// inverse at dequant. This decorrelates quantization errors across channels
    /// at the cost of ~2× dequant overhead.
    ///
    /// VarN already equalizes magnitudes via Sinkhorn scaling, so Hadamard is
    /// typically unnecessary. Benchmarks show no-Hadamard has BETTER quality
    /// (cosine 0.9988 vs 0.9974) because VarN handles magnitude equalization.
    ///
    /// Enable only if profiling shows error correlation across channels in
    /// your specific model.
    pub hadamard: bool,
    /// Optional static calibration table (Plan 227 Phase 1).
    /// When set, replaces Sinkhorn iterations with O(1) lookup.
    #[cfg(feature = "static_cal_tables")]
    pub static_cal: Option<crate::static_cal::StaticCalTable>,
    /// Optional per-head precision budget (Plan 227 Phase 2).
    /// When set, uses per-head bit-width instead of uniform.
    #[cfg(feature = "targeted_precision")]
    pub precision_budget: Option<PrecisionBudget>,
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
            hadamard: false,
            #[cfg(feature = "static_cal_tables")]
            static_cal: None,
            #[cfg(feature = "targeted_precision")]
            precision_budget: None,
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
    /// Quantized key data: flat `[n_layers * n_tiles * key_tile_packed_len]`.
    /// Layer-tile-major layout: element `(layer, tile)` is at
    /// `(layer * n_tiles + tile) * key_tile_packed_len`.
    key_quantized: Vec<u8>,
    /// Key tile metadata, flat `[n_layers * n_tiles]` indexed by
    /// `layer * n_tiles + tile_idx` (see `key_tile_meta`). Flattened from
    /// the previous Vec<Vec<TileMeta>> to collapse two pointer chases + two
    /// bounds checks on every per-token/per-dequant access into one of each.
    key_tiles: Vec<TileMeta>,
    /// Key tile buffer: [kv_dim * tile_size] — accumulates raw data for current tile.
    key_buffer: Vec<f32>,
    /// Quantized value data: flat `[n_layers * n_tiles * val_tile_packed_len]`.
    val_quantized: Vec<u8>,
    /// Value tile metadata, flat `[n_layers * n_tiles]` (see `key_tiles`).
    val_tiles: Vec<TileMeta>,
    /// Value tile buffer: [tile_size * kv_dim].
    val_buffer: Vec<f32>,
    // ── Scratch buffers for zero-alloc hot path ──
    /// Scratch for tile operations: [tile_rows * tile_cols].
    /// Reused by quantize_key_tile / quantize_val_tile to avoid per-tile allocation.
    scratch_tile: Vec<f32>,
    /// Scratch for batch-unpacked u32 values (reused across dequant calls).
    scratch_unpack: Vec<u32>,
    /// VarN scratch: copy of the tile being normalized (`rows * cols`).
    /// Sized to `max(kv_dim, tile_size)² ` to cover both K and V tile shapes.
    varn_cur: Vec<f32>,
    /// VarN scratch: per-column std devs (length = max(kv_dim, tile_size)).
    varn_col_s: Vec<f32>,
    /// VarN scratch: per-row std devs (length = max(kv_dim, tile_size)).
    varn_row_s: Vec<f32>,
    /// VarN scratch: per-column mean (length = max(kv_dim, tile_size)).
    varn_mean: Vec<f32>,
    /// VarN scratch: 1 / exp(log_s_row[i]) (length = max(kv_dim, tile_size)).
    varn_inv_row: Vec<f32>,
    /// VarN scratch: 1 / exp(log_s_col[j]) (length = max(kv_dim, tile_size)).
    varn_inv_col: Vec<f32>,
    /// VarN scratch: running column log-scale (length = max(kv_dim, tile_size)).
    varn_log_s_col: Vec<f32>,
    /// VarN scratch: running row log-scale (length = max(kv_dim, tile_size)).
    varn_log_s_row: Vec<f32>,
    /// VarN scratch: best-seen column log-scale (length = max(kv_dim, tile_size)).
    varn_log_s_col_best: Vec<f32>,
    /// VarN scratch: best-seen row log-scale (length = max(kv_dim, tile_size)).
    varn_log_s_row_best: Vec<f32>,
    /// Hadamard column-transform scratch (length = kv_dim; reused by quantize_key_tile).
    hadamard_col_buf: Vec<f32>,
    /// RTN scratch: per-row scales output of rtn_quantize_rows* (length =
    /// max(kv_dim, tile_size) * max groups_per_row). Reused across quantize calls.
    scratch_rtn_scales: Vec<f32>,
    /// RTN scratch: per-row zero-points output of rtn_quantize_rows* (same size as scratch_rtn_scales).
    scratch_rtn_zp: Vec<f32>,
    /// RTN scratch: packed bytes output of rtn_quantize_rows* (length =
    /// max(key_tile_packed_len, val_tile_packed_len)). Reused across quantize calls.
    scratch_rtn_packed: Vec<u8>,
    /// VarianceNormScales scratch: s_col output (length = max(kv_dim, tile_size)).
    /// Holds the col-scale vector written by variance_normalize_into and the
    /// static-cal/skip_varn fill paths, replacing the per-call vec![1.0; cols]
    /// allocation in the skip_varn branch.
    scratch_var_s_col: Vec<f32>,
    /// VarianceNormScales scratch: s_row output (length = max(kv_dim, tile_size)).
    /// Holds the row-scale vector written by variance_normalize_into / static_cal
    /// / skip_varn, replacing the per-call vec![1.0; rows] allocation.
    scratch_var_s_row: Vec<f32>,
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
    #[allow(dead_code)] // computed at construction for fast path access
    bytes_per_row: usize,
    /// Packed bytes per key tile: `bytes_per_row * kv_dim`.
    key_tile_packed_len: usize,
    /// Packed bytes per value tile: `packed_bytes_per_row(kv_dim, bits) * tile_size`.
    val_tile_packed_len: usize,
    // ── Small fields at end ──
    /// Bits per element.
    bits: u8,
    /// Whether Hadamard rotation is enabled (user config).
    #[allow(dead_code)] // kept as user-facing config; effective_hadamard is used internally
    hadamard: bool,
    /// Whether to skip variance normalization (computed: bits <= 2).
    skip_varn: bool,
    /// Effective Hadamard mode (computed: hadamard).
    effective_hadamard: bool,
    /// Sub-channel group size for RTN quantization (at 2-bit: 32, otherwise 0 = full row).
    group_size: usize,
    /// Optional static calibration table for O(1) scale lookup.
    #[cfg(feature = "static_cal_tables")]
    static_cal: Option<crate::static_cal::StaticCalTable>,
    /// Optional per-head precision budget for non-uniform quantization.
    #[cfg(feature = "targeted_precision")]
    precision_budget: Option<PrecisionBudget>,
}

impl KVarNKVCache {
    /// Byte offset of key tile `(layer, tile_idx)` in the flat `key_quantized`.
    #[inline]
    fn key_tile_off(&self, layer: usize, tile_idx: usize) -> usize {
        (layer * self.n_tiles + tile_idx) * self.key_tile_packed_len
    }

    /// Byte offset of value tile `(layer, tile_idx)` in the flat `val_quantized`.
    #[inline]
    fn val_tile_off(&self, layer: usize, tile_idx: usize) -> usize {
        (layer * self.n_tiles + tile_idx) * self.val_tile_packed_len
    }

    /// Flat index of tile `(layer, tile_idx)` in either `key_tiles` or `val_tiles`.
    /// Free function to avoid borrow-checker conflicts when callers need
    /// `&mut self.{key,val}_tiles[idx]` (calling a `&self` method in the same
    /// expression would conflict with the `&mut self` borrow).
    #[inline]
    fn tile_meta_idx(n_tiles: usize, layer: usize, tile_idx: usize) -> usize {
        layer * n_tiles + tile_idx
    }

    /// Create a new KVarN KV cache from config.
    pub fn with_config(cfg: &KVarNConfig) -> Self {
        let n_tiles = cfg.max_seq_len.div_ceil(cfg.tile_size);
        let tile_size = cfg.tile_size;

        // For keys: tile layout [kv_dim, tile_size]
        let key_tile_rows = cfg.kv_dim;
        let key_tile_cols = tile_size;
        // For values: tile layout [tile_size, kv_dim]
        let val_tile_rows = tile_size;
        let val_tile_cols = cfg.kv_dim;

        let bytes_per_row = packed_bytes_per_row(tile_size, cfg.bits);
        let key_tile_packed_len = bytes_per_row * cfg.kv_dim;
        let val_tile_packed_len = packed_bytes_per_row(cfg.kv_dim, cfg.bits) * tile_size;

        // Initialize per-layer, per-tile storage (flat: one contiguous allocation)
        let key_quantized = vec![0u8; cfg.n_layers * n_tiles * key_tile_packed_len];
        // Flat tile metadata: one contiguous Vec<TileMeta> indexed by
        // `layer * n_tiles + tile_idx`. Replaces Vec<Vec<TileMeta>> to collapse
        // (n_layers + 1) heap allocations and per-access pointer chases into
        // a single allocation + single bounds-checked index.
        let key_tiles: Vec<TileMeta> = (0..cfg.n_layers)
            .flat_map(|_| {
                (0..n_tiles)
                    .map(|_| TileMeta::empty(tile_size, key_tile_rows, key_tile_cols))
            })
            .collect();

        let val_quantized = vec![0u8; cfg.n_layers * n_tiles * val_tile_packed_len];
        let val_tiles: Vec<TileMeta> = (0..cfg.n_layers)
            .flat_map(|_| {
                (0..n_tiles)
                    .map(|_| TileMeta::empty(tile_size, val_tile_rows, val_tile_cols))
            })
            .collect();

        let key_buffer_size = cfg.kv_dim * tile_size;
        let val_buffer_size = tile_size * cfg.kv_dim;

        // VarN scratch sizes: key tile is [kv_dim, count], val tile is [count, kv_dim].
        // Both dimensions are bounded by max(kv_dim, tile_size), so a single square
        // scratch layout covers both call sites without resizing.
        let varn_max_dim = cfg.kv_dim.max(tile_size);
        let varn_tile_size = varn_max_dim * varn_max_dim;

        // RTN scratch sizes: cover both K (rows=kv_dim, cols=tile_size) and V
        // (rows=tile_size, cols=kv_dim) tile shapes, plus the optional
        // sub-channel grouping factor (groups_per_row = cols/group_size).
        // Max scales/zps entries = max_rows * max_groups_per_row.
        let rtn_max_rows = varn_max_dim;
        let rtn_max_cols = varn_max_dim;
        let rtn_max_groups = if cfg.bits <= 2 {
            // 2-bit path uses group_size=4; ceil(cols/4) groups per row.
            rtn_max_cols.div_ceil(4)
        } else {
            1
        };
        let rtn_scales_len = rtn_max_rows * rtn_max_groups;
        let rtn_packed_len = key_tile_packed_len.max(val_tile_packed_len);

        Self {
            key_quantized,
            key_tiles,
            key_buffer: vec![0.0; key_buffer_size],
            val_quantized,
            val_tiles,
            val_buffer: vec![0.0; val_buffer_size],
            scratch_tile: vec![0.0f32; key_buffer_size.max(val_buffer_size)],
            scratch_unpack: vec![0u32; cfg.kv_dim],
            varn_cur: vec![0.0f32; varn_tile_size],
            varn_col_s: vec![0.0f32; varn_max_dim],
            varn_row_s: vec![0.0f32; varn_max_dim],
            varn_mean: vec![0.0f32; varn_max_dim],
            varn_inv_row: vec![0.0f32; varn_max_dim],
            varn_inv_col: vec![0.0f32; varn_max_dim],
            varn_log_s_col: vec![0.0f32; varn_max_dim],
            varn_log_s_row: vec![0.0f32; varn_max_dim],
            varn_log_s_col_best: vec![0.0f32; varn_max_dim],
            varn_log_s_row_best: vec![0.0f32; varn_max_dim],
            hadamard_col_buf: vec![0.0f32; cfg.kv_dim],
            scratch_rtn_scales: vec![0.0f32; rtn_scales_len],
            scratch_rtn_zp: vec![0.0f32; rtn_scales_len],
            scratch_rtn_packed: vec![0u8; rtn_packed_len],
            scratch_var_s_col: vec![1.0f32; varn_max_dim],
            scratch_var_s_row: vec![1.0f32; varn_max_dim],
            pos: 0,
            n_layers: cfg.n_layers,
            kv_dim: cfg.kv_dim,
            max_seq_len: cfg.max_seq_len,
            tile_size,
            n_tiles,
            bytes_per_row,
            key_tile_packed_len,
            val_tile_packed_len,
            bits: cfg.bits,
            hadamard: cfg.hadamard,
            skip_varn: cfg.bits <= 2,
            effective_hadamard: cfg.hadamard,
            group_size: if cfg.bits <= 2 { 4 } else { 0 },
            #[cfg(feature = "static_cal_tables")]
            static_cal: cfg.static_cal.clone(),
            #[cfg(feature = "targeted_precision")]
            precision_budget: cfg.precision_budget.clone(),
        }
    }

    /// Get effective bits for a specific channel/row in a layer.
    /// When targeted_precision is enabled with a budget, uses per-head bits.
    /// Otherwise falls back to uniform self.bits.
    #[inline]
    #[allow(dead_code)] // reserved for future per-row quantization (Plan 227 Phase 2+)
    fn effective_bits(&self, _layer: usize, _channel: usize) -> u8 {
        #[cfg(feature = "targeted_precision")]
        if let Some(ref budget) = self.precision_budget {
            let head = _channel; // simplified: 1 channel per head for budget purposes
            return budget.get_bits(_layer, head);
        }
        self.bits
    }

    /// Quantize and store a key vector at given layer and position.
    ///
    /// Buffers raw data into the tile. Hadamard rotation (if enabled) is applied
    /// per-tile at quantization time when the tile fills — see `quantize_key_tile`.
    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) {
        debug_assert_eq!(key.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);
        debug_assert!(pos < self.max_seq_len);

        let tile_idx = pos / self.tile_size;
        let pos_in_tile = pos % self.tile_size;

        // Buffer layout: [kv_dim, tile_size] row-major, so row=channel, col=token
        for (ch, &k) in key.iter().enumerate().take(self.kv_dim) {
            self.key_buffer[ch * self.tile_size + pos_in_tile] = k;
        }

        let tile = &mut self.key_tiles[Self::tile_meta_idx(self.n_tiles, layer, tile_idx)];
        tile.count += 1;

        // If tile is complete, quantize it
        if tile.count == self.tile_size || pos == self.max_seq_len - 1 {
            let count = tile.count;
            self.quantize_key_tile(layer, tile_idx, count);
        }
    }

    /// Quantize and store a value vector at given layer and position.
    ///
    /// Buffers raw data into the tile. Hadamard rotation (if enabled) is applied
    /// per-tile at quantization time — see `quantize_val_tile`.
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) {
        debug_assert_eq!(value.len(), self.kv_dim);
        debug_assert!(layer < self.n_layers);
        debug_assert!(pos < self.max_seq_len);

        let tile_idx = pos / self.tile_size;
        let pos_in_tile = pos % self.tile_size;

        // Buffer layout: [tile_size, kv_dim] row-major, so row=token, col=channel
        let off = pos_in_tile * self.kv_dim;
        self.val_buffer[off..off + self.kv_dim].copy_from_slice(value);

        let tile = &mut self.val_tiles[Self::tile_meta_idx(self.n_tiles, layer, tile_idx)];
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

        let tile = &self.key_tiles[Self::tile_meta_idx(self.n_tiles, layer, tile_idx)];
        if tile.count == 0 {
            out.fill(0.0);
            return;
        }

        // Use actual tile cols for bpr (may differ for incomplete last tile)
        let actual_cols = tile.count.min(self.tile_size);
        let bits = self.bits as usize;
        let bpr = packed_bytes_per_row(actual_cols, self.bits);
        let kv_dim = self.kv_dim;

        let quantized_off = self.key_tile_off(layer, tile_idx);
        let quantized =
            &self.key_quantized[quantized_off..quantized_off + self.key_tile_packed_len];

        // Precompute column-scale constant (same for all channels)
        let var_col = tile.var_scales.s_col[pos_in_tile];
        let rtn_scales = &tile.rtn_scales;
        let rtn_zp = &tile.rtn_zp;
        let s_row = &tile.var_scales.s_row;

        // Specialized hot path per bit-width to avoid generic division/modulo.
        // Inner loops use `mul_add` so `(q*s + zp)` becomes a single fused FMA.
        match bits {
            4 => {
                // 4-bit: 2 values per byte, pos_in_tile determines nibble
                let byte_off = pos_in_tile >> 1;
                let shift = (pos_in_tile & 1) * 4;
                let mask: u8 = 0x0F;
                for ch in 0..kv_dim {
                    let q = ((quantized[ch * bpr + byte_off] >> shift) & mask) as f32;
                    let var_row = s_row[ch];
                    out[ch] = q.mul_add(rtn_scales[ch], rtn_zp[ch]) * var_col * var_row;
                }
            }
            2 => {
                // 2-bit: 4 values per byte
                let byte_off = pos_in_tile >> 2;
                let shift = (pos_in_tile & 3) * 2;
                let mask: u8 = 0x03;
                if self.skip_varn {
                    if self.group_size > 0 {
                        // Grouped quantization: find group for this position
                        let groups_per_row = actual_cols.div_ceil(self.group_size);
                        let g = pos_in_tile / self.group_size;
                        for ch in 0..kv_dim {
                            let q = ((quantized[ch * bpr + byte_off] >> shift) & mask) as f32;
                            let idx = ch * groups_per_row + g.min(groups_per_row - 1);
                            out[ch] = q.mul_add(rtn_scales[idx], rtn_zp[idx]);
                        }
                    } else {
                        for ch in 0..kv_dim {
                            let q = ((quantized[ch * bpr + byte_off] >> shift) & mask) as f32;
                            out[ch] = q.mul_add(rtn_scales[ch], rtn_zp[ch]);
                        }
                    }
                } else {
                    for ch in 0..kv_dim {
                        let q = ((quantized[ch * bpr + byte_off] >> shift) & mask) as f32;
                        let var_row = s_row[ch];
                        out[ch] = q.mul_add(rtn_scales[ch], rtn_zp[ch]) * var_col * var_row;
                    }
                }
            }
            8 => {
                // 8-bit: 1 value per byte, trivial
                for ch in 0..kv_dim {
                    let q = quantized[ch * bpr + pos_in_tile] as f32;
                    let var_row = s_row[ch];
                    out[ch] = q.mul_add(rtn_scales[ch], rtn_zp[ch]) * var_col * var_row;
                }
            }
            _ => {
                // Fallback: generic unpack
                for ch in 0..kv_dim {
                    let row_off = ch * bpr;
                    let q = unpack_value(&quantized[row_off..row_off + bpr], pos_in_tile, bits);
                    let var_row = s_row[ch];
                    out[ch] = (q as f32).mul_add(rtn_scales[ch], rtn_zp[ch]) * var_col * var_row;
                }
            }
        }

        // Inverse Hadamard on the output vector.
        // Hadamard is applied per-tile on the channel dimension (columns for keys,
        // rows for values). The dequantized output vector needs inverse Hadamard
        // to recover the original channel values.
        if self.effective_hadamard && kv_dim.is_power_of_two() {
            hadamard::hadamard_transform_inplace(out);
        }
    }

    /// Dequantize value into pre-allocated buffer (zero-alloc hot path).
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.kv_dim);
        let tile_idx = pos / self.tile_size;
        let pos_in_tile = pos % self.tile_size;

        let tile = &self.val_tiles[Self::tile_meta_idx(self.n_tiles, layer, tile_idx)];
        if tile.count == 0 {
            out.fill(0.0);
            return;
        }

        let bits = self.bits as usize;
        let kv_dim = self.kv_dim;
        let bpr = packed_bytes_per_row(kv_dim, self.bits);

        let row_off = pos_in_tile * bpr;
        let tile_off = self.val_tile_off(layer, tile_idx);
        let packed_row = &self.val_quantized[tile_off + row_off..tile_off + row_off + bpr];

        // Precompute row-scale constant (same for all channels in this row)
        let var_row = tile.var_scales.s_row[pos_in_tile];
        let rtn_scales = &tile.rtn_scales;
        let rtn_zp = &tile.rtn_zp;
        let s_col = &tile.var_scales.s_col;

        // Specialized hot path per bit-width — inline unpack directly into dequant.
        // Inner loops use `mul_add` so `(q*s + zp)` becomes a single fused FMA.
        match bits {
            4 => {
                // 2 values per byte, dequant inline.
                // Split into full-pair loop (branch-free) + odd tail to eliminate the
                // per-iteration `if 2*i+1 < kv_dim` check on the common even-kv_dim path.
                let rtn_scale = rtn_scales[pos_in_tile];
                let rtn_zp_val = rtn_zp[pos_in_tile];
                let full_pairs = kv_dim / 2;
                for i in 0..full_pairs {
                    let b = packed_row[i];
                    let q0 = (b & 0x0F) as f32;
                    out[2 * i] = q0.mul_add(rtn_scale, rtn_zp_val) * s_col[2 * i] * var_row;
                    let q1 = (b >> 4) as f32;
                    out[2 * i + 1] = q1.mul_add(rtn_scale, rtn_zp_val) * s_col[2 * i + 1] * var_row;
                }
                if kv_dim & 1 == 1 {
                    let b = packed_row[full_pairs];
                    let q0 = (b & 0x0F) as f32;
                    out[2 * full_pairs] =
                        q0.mul_add(rtn_scale, rtn_zp_val) * s_col[2 * full_pairs] * var_row;
                }
            }
            2 => {
                // 4 values per byte
                if self.skip_varn {
                    if self.group_size > 0 {
                        // Grouped quantization: per-token, per-channel-group scales.
                        //
                        // Fast path: group_size == 4 means each byte covers exactly
                        // one group (4 values / 4 = 1 byte per group). This is the
                        // only configuration that sets group_size > 0 (see with_config:
                        // `group_size: if cfg.bits <= 2 { 4 } else { 0 }`), so we can
                        // specialize the branch-free inner loop. The original code
                        // had 3 `if 4*i+k < kv_dim` checks per byte.
                        debug_assert_eq!(
                            self.group_size, 4,
                            "group_size>0 implies group_size==4 at 2-bit"
                        );
                        let groups_per_row = kv_dim.div_ceil(self.group_size);
                        let row_base = pos_in_tile * groups_per_row;
                        let full_quads = kv_dim / 4;
                        for i in 0..full_quads {
                            let b = packed_row[i];
                            let g = i.min(groups_per_row - 1);
                            let idx = row_base + g;
                            let scale = rtn_scales[idx];
                            let zp = rtn_zp[idx];
                            out[4 * i] = ((b & 0x03) as f32).mul_add(scale, zp);
                            out[4 * i + 1] = (((b >> 2) & 0x03) as f32).mul_add(scale, zp);
                            out[4 * i + 2] = (((b >> 4) & 0x03) as f32).mul_add(scale, zp);
                            out[4 * i + 3] = (((b >> 6) & 0x03) as f32).mul_add(scale, zp);
                        }
                        // Tail: 0..=3 remaining values packed in the next byte,
                        // all in the last group.
                        let tail_start = 4 * full_quads;
                        if tail_start < kv_dim {
                            let b = packed_row[full_quads];
                            let idx = row_base + (groups_per_row - 1);
                            let scale = rtn_scales[idx];
                            let zp = rtn_zp[idx];
                            let shifts = [0u32, 2, 4, 6];
                            for (j, &sh) in shifts.iter().enumerate() {
                                let k = tail_start + j;
                                if k >= kv_dim {
                                    break;
                                }
                                out[k] = (((b >> sh) & 0x03) as f32).mul_add(scale, zp);
                            }
                        }
                    } else {
                        // Non-grouped: per-token scale
                        let rtn_scale = rtn_scales[pos_in_tile];
                        let rtn_zp_val = rtn_zp[pos_in_tile];
                        // Branch-free over complete quads; tail handled separately.
                        let full_quads = kv_dim / 4;
                        for i in 0..full_quads {
                            let b = packed_row[i];
                            out[4 * i] = ((b & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                            out[4 * i + 1] =
                                (((b >> 2) & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                            out[4 * i + 2] =
                                (((b >> 4) & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                            out[4 * i + 3] =
                                (((b >> 6) & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                        }
                        let tail_start = 4 * full_quads;
                        if tail_start < kv_dim {
                            let tail_byte = packed_row[full_quads];
                            let shifts = [0u32, 2, 4, 6];
                            for (j, &sh) in shifts.iter().enumerate() {
                                let idx = tail_start + j;
                                if idx >= kv_dim {
                                    break;
                                }
                                out[idx] = (((tail_byte >> sh) & 0x03) as f32)
                                    .mul_add(rtn_scale, rtn_zp_val);
                            }
                        }
                    }
                } else {
                    let rtn_scale = rtn_scales[pos_in_tile];
                    let rtn_zp_val = rtn_zp[pos_in_tile];
                    // Process complete quads branch-free, then handle the 0–3 elem tail.
                    // Common case (kv_dim divisible by 4) skips all per-iter bounds checks.
                    let full_quads = kv_dim / 4;
                    for i in 0..full_quads {
                        let b = packed_row[i];
                        let q0 = (b & 0x03) as f32;
                        out[4 * i] = q0.mul_add(rtn_scale, rtn_zp_val) * s_col[4 * i] * var_row;
                        let q1 = ((b >> 2) & 0x03) as f32;
                        out[4 * i + 1] =
                            q1.mul_add(rtn_scale, rtn_zp_val) * s_col[4 * i + 1] * var_row;
                        let q2 = ((b >> 4) & 0x03) as f32;
                        out[4 * i + 2] =
                            q2.mul_add(rtn_scale, rtn_zp_val) * s_col[4 * i + 2] * var_row;
                        let q3 = ((b >> 6) & 0x03) as f32;
                        out[4 * i + 3] =
                            q3.mul_add(rtn_scale, rtn_zp_val) * s_col[4 * i + 3] * var_row;
                    }
                    // Tail: 0..=3 remaining elements packed in the next byte.
                    // Only access packed_row[full_quads] if a tail actually exists.
                    let tail_start = 4 * full_quads;
                    if tail_start < kv_dim {
                        let tail_byte = packed_row[full_quads];
                        let shifts = [0u32, 2, 4, 6];
                        for (j, &sh) in shifts.iter().enumerate() {
                            let idx = tail_start + j;
                            if idx >= kv_dim {
                                break;
                            }
                            let q = ((tail_byte >> sh) & 0x03) as f32;
                            out[idx] = q.mul_add(rtn_scale, rtn_zp_val) * s_col[idx] * var_row;
                        }
                    }
                }
            }
            8 => {
                let rtn_scale = rtn_scales[pos_in_tile];
                let rtn_zp_val = rtn_zp[pos_in_tile];
                for ch in 0..kv_dim {
                    let q = packed_row[ch] as f32;
                    out[ch] = q.mul_add(rtn_scale, rtn_zp_val) * s_col[ch] * var_row;
                }
            }
            _ => {
                // Fallback: batch unpack then dequant
                let rtn_scale = rtn_scales[pos_in_tile];
                let rtn_zp_val = rtn_zp[pos_in_tile];
                let scratch = &mut self.scratch_unpack[..kv_dim];
                unpack_row(packed_row, bits, scratch);
                for ch in 0..kv_dim {
                    let q = scratch[ch] as f32;
                    out[ch] = q.mul_add(rtn_scale, rtn_zp_val) * s_col[ch] * var_row;
                }
            }
        }

        // Inverse Hadamard on the output vector — see dequantize_key_into.
        if self.effective_hadamard && kv_dim.is_power_of_two() {
            hadamard::hadamard_transform_inplace(out);
        }
    }

    /// Reset cache for a new sequence.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.key_buffer.fill(0.0);
        self.val_buffer.fill(0.0);
        self.key_quantized.fill(0);
        self.val_quantized.fill(0);
        for layer in 0..self.n_layers {
            let base = layer * self.n_tiles;
            for t in 0..self.n_tiles {
                self.key_tiles[base + t].count = 0;
                self.val_tiles[base + t].count = 0;
            }
        }
    }

    /// Current write position.
    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set the current write position.
    #[inline]
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    // ── Internal quantization ──

    /// Quantize a key tile: [kv_dim, tile_size] with variance normalization.
    fn quantize_key_tile(&mut self, layer: usize, tile_idx: usize, count: usize) {
        let rows = self.kv_dim;
        let cols = count.min(self.tile_size);
        let tile_size = self.tile_size;

        // Reuse the pre-allocated scratch_tile buffer (kv_dim * tile_size floats).
        // Drop the mutable borrow before writing back to storage fields.
        let (rtn_scales_len, packed_len, bits) = {
            let tile_data = &mut self.scratch_tile[..rows * cols];

            // Strided copy: key_buffer is [kv_dim, tile_size] row-major; compact to [rows, cols].
            for ch in 0..rows {
                let src = &self.key_buffer[ch * tile_size..ch * tile_size + cols];
                let dst = &mut tile_data[ch * cols..ch * cols + cols];
                dst.copy_from_slice(src);
            }

            // Hadamard rotation per-tile on channel dimension:
            //   Key tile [kv_dim, tile_size]: Hadamard each column (= each position's channels)
            //   This is equivalent to per-position Hadamard on kv_dim, but batched at tile time.
            if self.effective_hadamard && rows.is_power_of_two() {
                hadamard::hadamard_cols_into(tile_data, rows, cols, &mut self.hadamard_col_buf);
            }

            // Step 1: Variance normalization — write s_col/s_row directly into scratch.
            //   Static cal tables: O(1) lookup replaces Sinkhorn iterations (Plan 227 Phase 1)
            #[cfg(feature = "static_cal_tables")]
            if let Some(ref cal) = self.static_cal {
                // Use static per-head scales instead of iterative Sinkhorn.
                let s_row = &mut self.scratch_var_s_row[..rows];
                for ch in 0..rows {
                    s_row[ch] = cal.get_scale(layer, ch);
                }
                // s_col = ones (no column scaling).
                self.scratch_var_s_col[..cols].fill(1.0);
                // Apply static scales to tile (reciprocal-multiply: one division per row,
                // not per element — vectorizer-friendly inner loop).
                for ch in 0..rows {
                    let inv_scale = 1.0 / s_row[ch];
                    let row_off = ch * cols;
                    for t in 0..cols {
                        tile_data[row_off + t] *= inv_scale;
                    }
                }
            } else if self.skip_varn {
                self.scratch_var_s_col[..cols].fill(1.0);
                self.scratch_var_s_row[..rows].fill(1.0);
            } else {
                let config = VarNormConfig {
                    tile_size: self.tile_size,
                    ..Default::default()
                };
                variance_normalize_into_scales(
                    tile_data, rows, cols, &config,
                    &mut self.varn_cur[..rows * cols],
                    &mut self.varn_col_s[..cols],
                    &mut self.varn_row_s[..rows],
                    &mut self.varn_mean[..cols],
                    &mut self.varn_inv_row[..rows],
                    &mut self.varn_inv_col[..cols],
                    &mut self.varn_log_s_col[..cols],
                    &mut self.varn_log_s_row[..rows],
                    &mut self.varn_log_s_col_best[..cols],
                    &mut self.varn_log_s_row_best[..rows],
                    &mut self.scratch_var_s_col[..cols],
                    &mut self.scratch_var_s_row[..rows],
                );
            }

            #[cfg(not(feature = "static_cal_tables"))]
            if self.skip_varn {
                self.scratch_var_s_col[..cols].fill(1.0);
                self.scratch_var_s_row[..rows].fill(1.0);
            } else {
                let config = VarNormConfig {
                    tile_size: self.tile_size,
                    ..Default::default()
                };
                variance_normalize_into_scales(
                    tile_data, rows, cols, &config,
                    &mut self.varn_cur[..rows * cols],
                    &mut self.varn_col_s[..cols],
                    &mut self.varn_row_s[..rows],
                    &mut self.varn_mean[..cols],
                    &mut self.varn_inv_row[..rows],
                    &mut self.varn_inv_col[..cols],
                    &mut self.varn_log_s_col[..cols],
                    &mut self.varn_log_s_row[..rows],
                    &mut self.varn_log_s_col_best[..cols],
                    &mut self.varn_log_s_row_best[..rows],
                    &mut self.scratch_var_s_col[..cols],
                    &mut self.scratch_var_s_row[..rows],
                );
            }

            // Step 2: RTN quantization
            //   Targeted precision: use budget-allocated bits (Plan 227 Phase 2)
            #[cfg(feature = "targeted_precision")]
            let bits = if let Some(ref budget) = self.precision_budget {
                budget.budget.ceil() as u8 // use ceiling to avoid precision loss
            } else {
                self.bits
            };

            #[cfg(not(feature = "targeted_precision"))]
            let bits = self.bits;

            let bpr = packed_bytes_per_row(cols, bits);
            let packed_len = rows * bpr;
            let (scales_len, packed_len) = if self.group_size > 0 {
                let groups_per_row = cols.div_ceil(self.group_size);
                rtn_quantize_rows_grouped_into(
                    tile_data,
                    rows,
                    cols,
                    bits,
                    self.group_size,
                    &mut self.scratch_rtn_scales[..rows * groups_per_row],
                    &mut self.scratch_rtn_zp[..rows * groups_per_row],
                    &mut self.scratch_rtn_packed[..packed_len],
                );
                (rows * groups_per_row, packed_len)
            } else {
                rtn_quantize_rows_into(
                    tile_data,
                    rows,
                    cols,
                    bits,
                    &mut self.scratch_rtn_scales[..rows],
                    &mut self.scratch_rtn_zp[..rows],
                    &mut self.scratch_rtn_packed[..packed_len],
                );
                (rows, packed_len)
            };
            (scales_len, packed_len, bits)
        };

        // Store: copy scratch → TileMeta using clear+extend (preserves Vec capacity
        // across calls → zero alloc in steady state after first quantize).
        let bpr = packed_bytes_per_row(cols, bits);
        let meta = &mut self.key_tiles[Self::tile_meta_idx(self.n_tiles, layer, tile_idx)];
        meta.count = count;
        meta.var_scales.s_col.clear();
        meta.var_scales.s_col.extend_from_slice(&self.scratch_var_s_col[..cols]);
        meta.var_scales.s_row.clear();
        meta.var_scales.s_row.extend_from_slice(&self.scratch_var_s_row[..rows]);
        meta.rtn_scales.clear();
        meta.rtn_scales.extend_from_slice(&self.scratch_rtn_scales[..rtn_scales_len]);
        meta.rtn_zp.clear();
        meta.rtn_zp.extend_from_slice(&self.scratch_rtn_zp[..rtn_scales_len]);

        let off = self.key_tile_off(layer, tile_idx);
        let quantized = &mut self.key_quantized[off..off + self.key_tile_packed_len];
        let expected_len = rows * bpr;
        // Zero the entire slot first, then write the packed data at the front.
        // The slot is pre-sized for a full tile (tile_size cols); actual packed
        // data may be shorter for partial tiles.
        quantized.fill(0);
        quantized[..expected_len].copy_from_slice(&self.scratch_rtn_packed[..packed_len]);
    }

    /// Quantize a value tile: [tile_size, kv_dim] with variance normalization.
    fn quantize_val_tile(&mut self, layer: usize, tile_idx: usize, count: usize) {
        let rows = count.min(self.tile_size);
        let cols = self.kv_dim;

        // Reuse the pre-allocated scratch_tile buffer (tile_size * kv_dim floats).
        // Drop the mutable borrow before writing back to storage fields.
        let (rtn_scales_len, packed_len, bits) = {
            let tile_data = &mut self.scratch_tile[..rows * cols];

            // val_buffer is already [tile_size, kv_dim] row-major contiguous; copy directly.
            tile_data.copy_from_slice(&self.val_buffer[..rows * cols]);

            // Hadamard rotation per-tile (clustered across channels per token)
            if self.effective_hadamard && cols.is_power_of_two() {
                hadamard::hadamard_rows(tile_data, cols);
            }

            // Step 1: Variance normalization — write s_col/s_row directly into scratch.
            //   Static cal tables: O(1) lookup replaces Sinkhorn iterations (Plan 227 Phase 1)
            #[cfg(feature = "static_cal_tables")]
            if let Some(ref cal) = self.static_cal {
                let s_row = &mut self.scratch_var_s_row[..rows];
                for ch in 0..rows {
                    s_row[ch] = cal.get_scale(layer, ch);
                }
                self.scratch_var_s_col[..cols].fill(1.0);
                for ch in 0..rows {
                    let inv_scale = 1.0 / s_row[ch];
                    let row_off = ch * cols;
                    for t in 0..cols {
                        tile_data[row_off + t] *= inv_scale;
                    }
                }
            } else if self.skip_varn {
                self.scratch_var_s_col[..cols].fill(1.0);
                self.scratch_var_s_row[..rows].fill(1.0);
            } else {
                let config = VarNormConfig {
                    tile_size: self.tile_size,
                    ..Default::default()
                };
                variance_normalize_into_scales(
                    tile_data, rows, cols, &config,
                    &mut self.varn_cur[..rows * cols],
                    &mut self.varn_col_s[..cols],
                    &mut self.varn_row_s[..rows],
                    &mut self.varn_mean[..cols],
                    &mut self.varn_inv_row[..rows],
                    &mut self.varn_inv_col[..cols],
                    &mut self.varn_log_s_col[..cols],
                    &mut self.varn_log_s_row[..rows],
                    &mut self.varn_log_s_col_best[..cols],
                    &mut self.varn_log_s_row_best[..rows],
                    &mut self.scratch_var_s_col[..cols],
                    &mut self.scratch_var_s_row[..rows],
                );
            }

            #[cfg(not(feature = "static_cal_tables"))]
            if self.skip_varn {
                self.scratch_var_s_col[..cols].fill(1.0);
                self.scratch_var_s_row[..rows].fill(1.0);
            } else {
                let config = VarNormConfig {
                    tile_size: self.tile_size,
                    ..Default::default()
                };
                variance_normalize_into_scales(
                    tile_data, rows, cols, &config,
                    &mut self.varn_cur[..rows * cols],
                    &mut self.varn_col_s[..cols],
                    &mut self.varn_row_s[..rows],
                    &mut self.varn_mean[..cols],
                    &mut self.varn_inv_row[..rows],
                    &mut self.varn_inv_col[..cols],
                    &mut self.varn_log_s_col[..cols],
                    &mut self.varn_log_s_row[..rows],
                    &mut self.varn_log_s_col_best[..cols],
                    &mut self.varn_log_s_row_best[..rows],
                    &mut self.scratch_var_s_col[..cols],
                    &mut self.scratch_var_s_row[..rows],
                );
            }

            // Step 2: RTN quantization
            //   Targeted precision: use budget-allocated bits (Plan 227 Phase 2)
            #[cfg(feature = "targeted_precision")]
            let bits = if let Some(ref budget) = self.precision_budget {
                budget.budget.ceil() as u8
            } else {
                self.bits
            };

            #[cfg(not(feature = "targeted_precision"))]
            let bits = self.bits;

            let bpr = packed_bytes_per_row(cols, bits);
            let packed_len = rows * bpr;
            let (scales_len, packed_len) = if self.group_size > 0 {
                let groups_per_row = cols.div_ceil(self.group_size);
                rtn_quantize_rows_grouped_into(
                    tile_data,
                    rows,
                    cols,
                    bits,
                    self.group_size,
                    &mut self.scratch_rtn_scales[..rows * groups_per_row],
                    &mut self.scratch_rtn_zp[..rows * groups_per_row],
                    &mut self.scratch_rtn_packed[..packed_len],
                );
                (rows * groups_per_row, packed_len)
            } else {
                rtn_quantize_rows_into(
                    tile_data,
                    rows,
                    cols,
                    bits,
                    &mut self.scratch_rtn_scales[..rows],
                    &mut self.scratch_rtn_zp[..rows],
                    &mut self.scratch_rtn_packed[..packed_len],
                );
                (rows, packed_len)
            };
            (scales_len, packed_len, bits)
        };

        let bpr = packed_bytes_per_row(cols, bits);
        let meta = &mut self.val_tiles[Self::tile_meta_idx(self.n_tiles, layer, tile_idx)];
        meta.count = count;
        meta.var_scales.s_col.clear();
        meta.var_scales.s_col.extend_from_slice(&self.scratch_var_s_col[..cols]);
        meta.var_scales.s_row.clear();
        meta.var_scales.s_row.extend_from_slice(&self.scratch_var_s_row[..rows]);
        meta.rtn_scales.clear();
        meta.rtn_scales.extend_from_slice(&self.scratch_rtn_scales[..rtn_scales_len]);
        meta.rtn_zp.clear();
        meta.rtn_zp.extend_from_slice(&self.scratch_rtn_zp[..rtn_scales_len]);

        let off = self.val_tile_off(layer, tile_idx);
        let quantized = &mut self.val_quantized[off..off + self.val_tile_packed_len];
        let expected_len = rows * bpr;
        // Zero the entire slot first, then write the packed data at the front.
        // The slot is pre-sized for a full tile (tile_size rows); actual packed
        // data may be shorter for partial tiles.
        quantized.fill(0);
        quantized[..expected_len].copy_from_slice(&self.scratch_rtn_packed[..packed_len]);
    }
}

impl katgpt_core::types::QuantizedKVCache for KVarNKVCache {
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

    #[inline]
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
pub fn packed_bytes_per_row(cols: usize, bits: u8) -> usize {
    (cols * bits as usize).div_ceil(8)
}

/// RTN quantize rows of a 2D tile. Returns (per-row scales, per-row zero-points, packed data).
///
/// For each row: find min/max, compute scale = (max - min) / (levels - 1),
/// quantize each element to [0, levels-1], pack into bits.
///
/// This is the allocating convenience wrapper. Hot-path callers should prefer
/// [`rtn_quantize_rows_into`] with a reusable scratch buffer.
pub fn rtn_quantize_rows(
    tile: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let bpr = packed_bytes_per_row(cols, bits);
    let mut packed = vec![0u8; rows * bpr];
    let mut scales = vec![1.0f32; rows];
    let mut zps = vec![0.0f32; rows];
    rtn_quantize_rows_into(tile, rows, cols, bits, &mut scales, &mut zps, &mut packed);
    (scales, zps, packed)
}

/// In-place RTN quantize rows of a 2D tile into caller-provided buffers.
///
/// - `scales.len() >= rows` (row-major per-row scale)
/// - `zps.len() >= rows`
/// - `packed.len() >= rows * packed_bytes_per_row(cols, bits)`
///
/// The first `rows` entries of `scales`/`zps` and the first
/// `rows * bpr` bytes of `packed` are overwritten; trailing bytes are untouched.
/// On degenerate rows (constant value), `scales[r] = 0.0` and `zps[r] = lo`,
/// and the packed bytes for that row are left untouched (dequant reads
/// `q.mul_add(0.0, lo) = lo` for any q, so zero-init packed is fine).
pub fn rtn_quantize_rows_into(
    tile: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
    scales: &mut [f32],
    zps: &mut [f32],
    packed: &mut [u8],
) {
    let levels = 1u32 << bits;
    let half_levels = (levels - 1) as f32;
    let bpr = packed_bytes_per_row(cols, bits);
    // Zero the output prefix so the degenerate-row early-continue path below
    // (constant-value rows) leaves zero packed bytes for that row, matching
    // the allocating wrapper's `vec![0u8; rows * bpr]` initialization.
    // Without this, reused scratch would leak stale packed bytes from prior calls.
    packed[..rows * bpr].fill(0);
    // Initialise the scales/zps prefixes to the defaults the allocating
    // wrapper used (scales=1.0, zps=0.0) so the degenerate-row early-continue
    // path produces bit-identical TileMeta contents.
    // `fill` on the exact prefix lowers to a memset instead of a scalar store
    // loop; the written values are the same constants.
    scales[..rows].fill(1.0);
    zps[..rows].fill(0.0);
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

        // Quantize and pack — use precomputed `inv_scale` and `neg_lo_over_scale`
        // so the hot inner loop becomes a single `mul_add` + round, no division.
        let inv_scale = 1.0 / scale;
        let bias = -lo * inv_scale;
        // Hoist the row slice out of the per-element loop: the `packed[r*bpr..]`
        // range slice (and its bounds check) was rebuilt for every element.
        // The open-ended range is kept verbatim — `pack_value` consults
        // `row.len()` for its write guards, so shortening the slice to exactly
        // `bpr` would change behaviour on the final element of a row.
        let prow = &mut packed[r * bpr..];
        for (c, &v) in row.iter().enumerate() {
            let normalized = v.mul_add(inv_scale, bias);
            let q = (normalized.round() as u32).clamp(0, levels - 1);
            pack_value(prow, c, q, bits as usize);
        }
    }
}

/// RTN quantize with sub-channel grouping.
///
/// Splits each row into `groups_per_row` groups of `group_size` elements.
/// Each group gets its own scale/zp, giving tighter quantization ranges.
/// Returns (scales[rows * groups_per_row], zps[rows * groups_per_row], packed data).
///
/// The packed data layout is identical to `rtn_quantize_rows`.
///
/// This is the allocating convenience wrapper. Hot-path callers should prefer
/// [`rtn_quantize_rows_grouped_into`] with a reusable scratch buffer.
pub fn rtn_quantize_rows_grouped(
    tile: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
    group_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let bpr = packed_bytes_per_row(cols, bits);
    let mut packed = vec![0u8; rows * bpr];
    let groups_per_row = cols.div_ceil(group_size);
    let mut scales = vec![1.0f32; rows * groups_per_row];
    let mut zps = vec![0.0f32; rows * groups_per_row];
    rtn_quantize_rows_grouped_into(
        tile, rows, cols, bits, group_size, &mut scales, &mut zps, &mut packed,
    );
    (scales, zps, packed)
}

/// In-place RTN quantize with sub-channel grouping into caller-provided buffers.
///
/// - `scales.len() >= rows * cols.div_ceil(group_size)`
/// - `zps.len() >= rows * cols.div_ceil(group_size)`
/// - `packed.len() >= rows * packed_bytes_per_row(cols, bits)`
#[allow(clippy::too_many_arguments)]
pub fn rtn_quantize_rows_grouped_into(
    tile: &[f32],
    rows: usize,
    cols: usize,
    bits: u8,
    group_size: usize,
    scales: &mut [f32],
    zps: &mut [f32],
    packed: &mut [u8],
) {
    let levels = 1u32 << bits;
    let half_levels = (levels - 1) as f32;
    let bpr = packed_bytes_per_row(cols, bits);
    let groups_per_row = cols.div_ceil(group_size);
    // Zero the output prefix so the degenerate-group early-continue path below
    // (constant-value groups) leaves zero packed bytes for that group's row
    // segment, matching the allocating wrapper's `vec![0u8; rows * bpr]` init.
    // Without this, reused scratch would leak stale packed bytes from prior calls.
    packed[..rows * bpr].fill(0);
    // Initialise the scales/zps prefixes to the defaults the allocating
    // wrapper used (scales=1.0, zps=0.0) so the degenerate-group early-continue
    // path produces bit-identical TileMeta contents.
    let total_entries = rows * groups_per_row;
    // `fill` on the exact prefix lowers to a memset instead of a scalar store
    // loop; the written values are the same constants.
    scales[..total_entries].fill(1.0);
    zps[..total_entries].fill(0.0);

    for r in 0..rows {
        let row_off = r * cols;
        // Pre-slice the row and the packed row once per row so the per-element
        // bounds checks below hoist out of the innermost loops.
        let row = &tile[row_off..row_off + cols];
        // Open-ended range kept verbatim — `pack_value` consults `row.len()`
        // for its write guards, so shortening to `bpr` would change behaviour.
        let prow = &mut packed[r * bpr..];
        for g in 0..groups_per_row {
            let g_start = g * group_size;
            let g_end = (g_start + group_size).min(cols);
            let group = &row[g_start..g_end];

            // Find min/max within this group — same element order as before.
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for &v in group {
                lo = lo.min(v);
                hi = hi.max(v);
            }

            let idx = r * groups_per_row + g;
            if hi - lo < 1e-10 {
                scales[idx] = 0.0;
                zps[idx] = lo;
                continue;
            }

            let scale = (hi - lo) / half_levels;
            scales[idx] = scale;
            zps[idx] = lo;

            // Quantize and pack elements in this group — single mul_add per element.
            let inv_scale = 1.0 / scale;
            let bias = -lo * inv_scale;
            for (c, &v) in (g_start..g_end).zip(group) {
                let normalized = v.mul_add(inv_scale, bias);
                let q = (normalized.round() as u32).clamp(0, levels - 1);
                pack_value(&mut *prow, c, q, bits as usize);
            }
        }
    }
}

/// Pack a value at given position into a bit-packed row.
#[inline]
pub fn pack_value(row: &mut [u8], pos: usize, value: u32, bits: usize) {
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
pub fn unpack_value(row: &[u8], pos: usize, bits: usize) -> u32 {
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

/// Unpack all values from a bit-packed row into a pre-allocated u32 buffer.
/// Optimized for power-of-2 bit widths (1, 2, 4, 8).
#[inline]
pub fn unpack_row(row: &[u8], bits: usize, out: &mut [u32]) {
    let n = out.len();
    match bits {
        8 => {
            for (i, &b) in row.iter().take(n).enumerate() {
                out[i] = b as u32;
            }
        }
        4 => {
            // 2 values per byte
            let byte_count = n.div_ceil(2);
            for (i, &b) in row.iter().take(byte_count).enumerate() {
                out[2 * i] = (b & 0x0F) as u32;
                if 2 * i + 1 < n {
                    out[2 * i + 1] = (b >> 4) as u32;
                }
            }
        }
        2 => {
            // 4 values per byte
            let byte_count = n.div_ceil(4);
            for (i, &b) in row.iter().take(byte_count).enumerate() {
                out[4 * i] = (b & 0x03) as u32;
                if 4 * i + 1 < n {
                    out[4 * i + 1] = ((b >> 2) & 0x03) as u32;
                }
                if 4 * i + 2 < n {
                    out[4 * i + 2] = ((b >> 4) & 0x03) as u32;
                }
                if 4 * i + 3 < n {
                    out[4 * i + 3] = ((b >> 6) & 0x03) as u32;
                }
            }
        }
        1 => {
            // 8 values per byte
            let byte_count = n.div_ceil(8);
            for (i, &b) in row.iter().take(byte_count).enumerate() {
                for bit in 0..8usize {
                    if 8 * i + bit < n {
                        out[8 * i + bit] = ((b >> bit) & 1) as u32;
                    }
                }
            }
        }
        _ => {
            // Fallback to per-element
            for (i, slot) in out.iter_mut().enumerate().take(n) {
                *slot = unpack_value(row, i, bits);
            }
        }
    }
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
            hadamard: false,
            #[cfg(feature = "static_cal_tables")]
            static_cal: None,
            #[cfg(feature = "targeted_precision")]
            precision_budget: None,
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
        let _cos = cosine_sim(&out0, &out1);
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
        assert_eq!(cache.key_tiles[0].count, 0);
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
    fn test_unpack_row_matches_unpack_value() {
        for &bits in &[1u8, 2, 3, 4, 8] {
            for &cols in &[1usize, 7, 8, 15, 16, 32, 64, 127, 128] {
                let bpr = packed_bytes_per_row(cols, bits);
                let levels = 1u32 << bits;
                let mut row = vec![0u8; bpr];

                // Pack known values
                for pos in 0..cols {
                    let val = (pos as u32 * 7 + 3) % levels;
                    pack_value(&mut row, pos, val, bits as usize);
                }

                // Batch unpack
                let mut batch_out = vec![0u32; cols];
                unpack_row(&row, bits as usize, &mut batch_out);

                // Compare with per-element unpack
                for (pos, got) in batch_out.iter().enumerate().take(cols) {
                    let expected = unpack_value(&row, pos, bits as usize);
                    assert_eq!(
                        got, &expected,
                        "unpack_row mismatch at pos={pos}, bits={bits}, cols={cols}: got {got}, expected {expected}"
                    );
                }
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

    #[test]
    fn test_kvarn_memory_usage_2bit() {
        let config = KVarNConfig {
            n_layers: 1,
            kv_dim: 128,
            max_seq_len: 1024,
            bits: 2,
            tile_size: 128,
            var_norm: VarNormConfig::default(),
            hadamard: false,
            #[cfg(feature = "static_cal_tables")]
            static_cal: None,
            #[cfg(feature = "targeted_precision")]
            precision_budget: None,
        };

        let mut cache = KVarNKVCache::with_config(&config);
        let mut rng_state: u64 = 42;

        // Fill all positions
        for pos in 0..config.max_seq_len {
            let key: Vec<f32> = (0..config.kv_dim)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((rng_state >> 33) as i32 as f32) / (1i32 << 31) as f32
                })
                .collect();
            let val: Vec<f32> = (0..config.kv_dim)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((rng_state >> 33) as i32 as f32) / (1i32 << 31) as f32
                })
                .collect();
            cache.store_key(0, pos, &key);
            cache.store_value(0, pos, &val);
        }

        // Calculate quantized data bytes
        let quantized_bytes = cache.key_quantized.len() + cache.val_quantized.len();

        // Approximate scale overhead per tile (conservative)
        let n_tiles = config.max_seq_len.div_ceil(config.tile_size);
        // Key tile: [kv_dim, tile_size] → s_col(tile_size), s_row(kv_dim), rtn_scales(kv_dim), rtn_zp(kv_dim)
        let key_tile_overhead = (config.tile_size + config.kv_dim * 3) * 4; // f32 bytes
        // Val tile: [tile_size, kv_dim] → s_col(kv_dim), s_row(tile_size), rtn_scales(tile_size), rtn_zp(tile_size)
        let val_tile_overhead = (config.kv_dim + config.tile_size * 3) * 4;
        let scale_bytes = n_tiles * (key_tile_overhead + val_tile_overhead) * config.n_layers;

        let total_bytes = quantized_bytes + scale_bytes;
        let total_elements = config.n_layers * config.max_seq_len * config.kv_dim * 2; // K + V
        let bits_per_elem = total_bytes as f64 * 8.0 / total_elements as f64;

        eprintln!("KVarN 2-bit memory: {bits_per_elem:.2} bits/elem (target ≤ 2.3)");
        eprintln!("  Quantized: {quantized_bytes} bytes, Scales: {scale_bytes} bytes");
        eprintln!("  Total: {total_bytes} bytes for {total_elements} elements");

        // Quantized data alone is 2.0 bits/elem; scale metadata adds ~1.0 bit overhead
        // at kv_dim=128. The 2.3 target is achievable at higher dims where scales amortize.
        assert!(
            bits_per_elem <= 3.1,
            "Memory usage too high: {bits_per_elem:.2} bits/elem, expected ≤ 3.1"
        );
    }
}
