//! KVarN per-row / per-column dequant kernels (Issue 894).
//!
//! `KVarNKVCache::{dequantize_value_into, dequantize_key_into}` resolve a tile
//! and hand the inputs of ONE dequant to these kernels as a read-only view.
//! The views are public so an oracle (the pre-894 indexed loops, kept
//! verbatim in `tests/common/kvarn_dequant_oracle.rs`) can be driven over
//! the exact inputs the shipped kernel sees — that is the bit-identity gate
//! and the paired A/B timing gate (`tests/bench_894_kvarn_dequant_zip_goat.rs`).
//!
//! ## Why the loops are shaped like this
//!
//! The pre-894 loops indexed `out[2*i]`, `s_col[2*i]`, `packed_row[i]`, …
//! inside `for i in 0..n`. No slice length was pinned to `n`, so every access
//! kept its bounds check and the loop ran scalar (Issue 894: ~0.75 ns/element
//! at 4 bits on an M3 Max). The rewrite walks the same elements with
//! `zip` / `as_chunks` over slices pre-cut to the loop length, which
//! removes the checks and lets LLVM vectorise.
//!
//! **Per-element arithmetic is unchanged, op for op and in the same order**:
//! `fma(q, scale, zp) * s_a * s_b`, left-associated exactly as before, with
//! `mul_add` in exactly the places the old code had it. No reassociation, no
//! new contraction — so the output is bit-identical by construction, and the
//! oracle gates assert it (0 differing bits).
//!
//! Every pre-cut slice spans exactly the elements the old loop read, so a
//! slice that is too short panics under the same condition the old indexing
//! did (up front instead of mid-loop).

#![allow(clippy::needless_range_loop)] // the two unchanged generic-bits fallbacks

/// Inputs of one value-row dequant (value tile layout `[tile_size, kv_dim]`:
/// one row = one token, per-token RTN scale × per-channel var-norm scale).
#[derive(Clone, Copy, Debug)]
pub struct KVarNValueRowView<'a> {
    /// Packed row bytes for this token (`packed_bytes_per_row(kv_dim, bits)`).
    pub packed_row: &'a [u8],
    /// The tile's RTN scales (per token, or per token × channel group at 2-bit grouped).
    pub rtn_scales: &'a [f32],
    /// The tile's RTN zero points (same layout as `rtn_scales`).
    pub rtn_zp: &'a [f32],
    /// Per-channel var-norm scales (length `kv_dim`).
    pub s_col: &'a [f32],
    /// This token's var-norm row scale.
    pub var_row: f32,
    /// Token index inside its tile.
    pub pos_in_tile: usize,
    /// KV dimension (output length).
    pub kv_dim: usize,
    /// Bits per element.
    pub bits: u8,
    /// Var-norm skipped (scales not applied at dequant).
    pub skip_varn: bool,
    /// Sub-channel group size (0 = ungrouped; `with_config` assigns only 0 or 4).
    pub group_size: usize,
}

/// Inputs of one key-column dequant (key tile layout `[kv_dim, tile_size]`:
/// one output = one token's column, per-channel RTN scale × per-token var-norm scale).
#[derive(Clone, Copy, Debug)]
pub struct KVarNKeyColView<'a> {
    /// The whole packed key-tile slot (rows = channels, `bpr` bytes each).
    pub quantized: &'a [u8],
    /// Packed bytes per channel row for this tile (`packed_bytes_per_row(actual_cols, bits)`).
    pub bpr: usize,
    /// Valid tokens in this tile (`count.min(tile_size)`).
    pub actual_cols: usize,
    /// The tile's RTN scales (per channel, or per channel × token group at 2-bit grouped).
    pub rtn_scales: &'a [f32],
    /// The tile's RTN zero points (same layout as `rtn_scales`).
    pub rtn_zp: &'a [f32],
    /// Per-channel var-norm row scales (length `kv_dim`).
    pub s_row: &'a [f32],
    /// This token's var-norm column scale.
    pub var_col: f32,
    /// Token index inside its tile.
    pub pos_in_tile: usize,
    /// KV dimension (output length).
    pub kv_dim: usize,
    /// Bits per element.
    pub bits: u8,
    /// Var-norm skipped (scales not applied at dequant).
    pub skip_varn: bool,
    /// Sub-channel group size (0 = ungrouped; `with_config` assigns only 0 or 4).
    pub group_size: usize,
}

/// `s[start], s[start + stride], …` — exactly `n` elements.
///
/// The slice is cut to end at the last element read, so it panics under the
/// same condition as the old `s[start + (n-1)*stride]` access, never reads
/// past it, and never silently yields fewer than `n` (which a `zip` would
/// otherwise truncate to).
#[inline(always)]
fn strided<T>(
    s: &[T],
    start: usize,
    stride: usize,
    n: usize,
) -> std::iter::StepBy<std::slice::Iter<'_, T>> {
    match n {
        0 => s[..0].iter().step_by(1),
        _ => s[start..start + (n - 1) * stride + 1]
            .iter()
            .step_by(stride),
    }
}

/// Dequantize one value row. `scratch` (≥ `kv_dim`) is used only by the
/// generic-bits fallback.
#[inline]
pub(crate) fn dequant_value_row(v: &KVarNValueRowView<'_>, scratch: &mut [u32], out: &mut [f32]) {
    let kv_dim = v.kv_dim;
    let packed_row = v.packed_row;
    let s_col = v.s_col;
    let var_row = v.var_row;
    let pos_in_tile = v.pos_in_tile;
    match v.bits {
        4 => {
            // 2 values per byte: full pairs, then the odd tail.
            let rtn_scale = v.rtn_scales[pos_in_tile];
            let rtn_zp_val = v.rtn_zp[pos_in_tile];
            let full_pairs = kv_dim / 2;
            let n = 2 * full_pairs;
            for ((o, &b), sc) in out[..n]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(&packed_row[..full_pairs])
                .zip(s_col[..n].as_chunks::<2>().0)
            {
                o[0] = ((b & 0x0F) as f32).mul_add(rtn_scale, rtn_zp_val) * sc[0] * var_row;
                o[1] = ((b >> 4) as f32).mul_add(rtn_scale, rtn_zp_val) * sc[1] * var_row;
            }
            if kv_dim & 1 == 1 {
                let b = packed_row[full_pairs];
                let q0 = (b & 0x0F) as f32;
                out[n] = q0.mul_add(rtn_scale, rtn_zp_val) * s_col[n] * var_row;
            }
        }
        2 => {
            // 4 values per byte: full quads, then the 0..=3 tail.
            let full_quads = kv_dim / 4;
            let n = 4 * full_quads;
            let shifts = [0u32, 2, 4, 6];
            match (v.skip_varn, v.group_size > 0) {
                (true, true) => {
                    // Grouped: per-token, per-channel-group scales. `with_config`
                    // assigns group_size = 4 whenever it is nonzero, so each byte is
                    // exactly one group, and the old `g = i.min(groups_per_row - 1)`
                    // is `i` for every full quad (i < kv_dim/4 ≤ kv_dim.div_ceil(4)).
                    debug_assert_eq!(
                        v.group_size, 4,
                        "group_size>0 implies group_size==4 at 2-bit"
                    );
                    let groups_per_row = kv_dim.div_ceil(v.group_size);
                    let row_base = pos_in_tile * groups_per_row;
                    for (((o, &b), &scale), &zp) in out[..n]
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(&packed_row[..full_quads])
                        .zip(&v.rtn_scales[row_base..row_base + full_quads])
                        .zip(&v.rtn_zp[row_base..row_base + full_quads])
                    {
                        o[0] = ((b & 0x03) as f32).mul_add(scale, zp);
                        o[1] = (((b >> 2) & 0x03) as f32).mul_add(scale, zp);
                        o[2] = (((b >> 4) & 0x03) as f32).mul_add(scale, zp);
                        o[3] = (((b >> 6) & 0x03) as f32).mul_add(scale, zp);
                    }
                    if n < kv_dim {
                        // Tail values are all in the last group.
                        let b = packed_row[full_quads];
                        let idx = row_base + (groups_per_row - 1);
                        let scale = v.rtn_scales[idx];
                        let zp = v.rtn_zp[idx];
                        for (o, &sh) in out[n..kv_dim].iter_mut().zip(&shifts) {
                            *o = (((b >> sh) & 0x03) as f32).mul_add(scale, zp);
                        }
                    }
                }
                (true, false) => {
                    // Ungrouped, no var-norm: per-token scale.
                    let rtn_scale = v.rtn_scales[pos_in_tile];
                    let rtn_zp_val = v.rtn_zp[pos_in_tile];
                    for (o, &b) in out[..n]
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(&packed_row[..full_quads])
                    {
                        o[0] = ((b & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                        o[1] = (((b >> 2) & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                        o[2] = (((b >> 4) & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                        o[3] = (((b >> 6) & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                    }
                    if n < kv_dim {
                        let tail_byte = packed_row[full_quads];
                        for (o, &sh) in out[n..kv_dim].iter_mut().zip(&shifts) {
                            *o = (((tail_byte >> sh) & 0x03) as f32).mul_add(rtn_scale, rtn_zp_val);
                        }
                    }
                }
                (false, _) => {
                    // Var-norm on: per-token scale × per-channel s_col × var_row.
                    let rtn_scale = v.rtn_scales[pos_in_tile];
                    let rtn_zp_val = v.rtn_zp[pos_in_tile];
                    for ((o, &b), sc) in out[..n]
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(&packed_row[..full_quads])
                        .zip(s_col[..n].as_chunks::<4>().0)
                    {
                        let q0 = (b & 0x03) as f32;
                        o[0] = q0.mul_add(rtn_scale, rtn_zp_val) * sc[0] * var_row;
                        let q1 = ((b >> 2) & 0x03) as f32;
                        o[1] = q1.mul_add(rtn_scale, rtn_zp_val) * sc[1] * var_row;
                        let q2 = ((b >> 4) & 0x03) as f32;
                        o[2] = q2.mul_add(rtn_scale, rtn_zp_val) * sc[2] * var_row;
                        let q3 = ((b >> 6) & 0x03) as f32;
                        o[3] = q3.mul_add(rtn_scale, rtn_zp_val) * sc[3] * var_row;
                    }
                    if n < kv_dim {
                        let tail_byte = packed_row[full_quads];
                        for ((o, &sc), &sh) in out[n..kv_dim]
                            .iter_mut()
                            .zip(&s_col[n..kv_dim])
                            .zip(&shifts)
                        {
                            let q = ((tail_byte >> sh) & 0x03) as f32;
                            *o = q.mul_add(rtn_scale, rtn_zp_val) * sc * var_row;
                        }
                    }
                }
            }
        }
        8 => {
            let rtn_scale = v.rtn_scales[pos_in_tile];
            let rtn_zp_val = v.rtn_zp[pos_in_tile];
            for ((o, &b), &sc) in out[..kv_dim]
                .iter_mut()
                .zip(&packed_row[..kv_dim])
                .zip(&s_col[..kv_dim])
            {
                *o = (b as f32).mul_add(rtn_scale, rtn_zp_val) * sc * var_row;
            }
        }
        bits => {
            // Fallback: batch unpack then dequant (unchanged from pre-894).
            let rtn_scale = v.rtn_scales[pos_in_tile];
            let rtn_zp_val = v.rtn_zp[pos_in_tile];
            let scratch = &mut scratch[..kv_dim];
            super::kv_cache::unpack_row(packed_row, bits as usize, scratch);
            for ch in 0..kv_dim {
                let q = scratch[ch] as f32;
                out[ch] = q.mul_add(rtn_scale, rtn_zp_val) * s_col[ch] * var_row;
            }
        }
    }
}

/// Dequantize one key column (one token across every channel row).
#[inline]
pub(crate) fn dequant_key_col(v: &KVarNKeyColView<'_>, out: &mut [f32]) {
    let kv_dim = v.kv_dim;
    let bpr = v.bpr;
    let quantized = v.quantized;
    let pos_in_tile = v.pos_in_tile;
    let var_col = v.var_col;
    match v.bits {
        4 => {
            // 2 values per byte; pos_in_tile picks the nibble.
            let shift = (pos_in_tile & 1) * 4;
            let mask: u8 = 0x0F;
            let (rtn_scales, rtn_zp) = (&v.rtn_scales[..kv_dim], &v.rtn_zp[..kv_dim]);
            for ((((o, &b), &sc), &zp), &var_row) in out[..kv_dim]
                .iter_mut()
                .zip(strided(quantized, pos_in_tile >> 1, bpr, kv_dim))
                .zip(rtn_scales)
                .zip(rtn_zp)
                .zip(&v.s_row[..kv_dim])
            {
                let q = ((b >> shift) & mask) as f32;
                *o = q.mul_add(sc, zp) * var_col * var_row;
            }
        }
        2 => {
            let bytes = strided(quantized, pos_in_tile >> 2, bpr, kv_dim);
            let shift = (pos_in_tile & 3) * 2;
            let mask: u8 = 0x03;
            match (v.skip_varn, v.group_size > 0) {
                (true, true) => {
                    // Grouped: this token's group index is the same for every channel.
                    let groups_per_row = v.actual_cols.div_ceil(v.group_size);
                    let g = (pos_in_tile / v.group_size).min(groups_per_row - 1);
                    for (((o, &b), &sc), &zp) in out[..kv_dim]
                        .iter_mut()
                        .zip(bytes)
                        .zip(strided(v.rtn_scales, g, groups_per_row, kv_dim))
                        .zip(strided(v.rtn_zp, g, groups_per_row, kv_dim))
                    {
                        let q = ((b >> shift) & mask) as f32;
                        *o = q.mul_add(sc, zp);
                    }
                }
                (true, false) => {
                    let (rtn_scales, rtn_zp) = (&v.rtn_scales[..kv_dim], &v.rtn_zp[..kv_dim]);
                    for (((o, &b), &sc), &zp) in out[..kv_dim]
                        .iter_mut()
                        .zip(bytes)
                        .zip(rtn_scales)
                        .zip(rtn_zp)
                    {
                        let q = ((b >> shift) & mask) as f32;
                        *o = q.mul_add(sc, zp);
                    }
                }
                (false, _) => {
                    let (rtn_scales, rtn_zp) = (&v.rtn_scales[..kv_dim], &v.rtn_zp[..kv_dim]);
                    for ((((o, &b), &sc), &zp), &var_row) in out[..kv_dim]
                        .iter_mut()
                        .zip(bytes)
                        .zip(rtn_scales)
                        .zip(rtn_zp)
                        .zip(&v.s_row[..kv_dim])
                    {
                        let q = ((b >> shift) & mask) as f32;
                        *o = q.mul_add(sc, zp) * var_col * var_row;
                    }
                }
            }
        }
        8 => {
            let (rtn_scales, rtn_zp) = (&v.rtn_scales[..kv_dim], &v.rtn_zp[..kv_dim]);
            for ((((o, &b), &sc), &zp), &var_row) in out[..kv_dim]
                .iter_mut()
                .zip(strided(quantized, pos_in_tile, bpr, kv_dim))
                .zip(rtn_scales)
                .zip(rtn_zp)
                .zip(&v.s_row[..kv_dim])
            {
                *o = (b as f32).mul_add(sc, zp) * var_col * var_row;
            }
        }
        bits => {
            // Fallback: generic unpack (unchanged from pre-894).
            let bits = bits as usize;
            let (rtn_scales, rtn_zp, s_row) = (v.rtn_scales, v.rtn_zp, v.s_row);
            for ch in 0..kv_dim {
                let row_off = ch * bpr;
                let q = super::kv_cache::unpack_value(
                    &quantized[row_off..row_off + bpr],
                    pos_in_tile,
                    bits,
                );
                let var_row = s_row[ch];
                out[ch] = (q as f32).mul_add(rtn_scales[ch], rtn_zp[ch]) * var_col * var_row;
            }
        }
    }
}
