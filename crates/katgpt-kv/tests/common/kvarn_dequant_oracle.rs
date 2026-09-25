//! The PRE-Issue-894 KVarN dequant loops, kept VERBATIM as the oracle.
//!
//! These are the indexed `for ch in 0..kv_dim` / `for i in 0..full_pairs`
//! bodies of `KVarNKVCache::{dequantize_key_into, dequantize_value_into}` as
//! of `30b28e3be`, transcribed mechanically: only the preamble changed (the
//! locals the old method computed from `self` now come from the public
//! `KVarN{KeyCol,ValueRow}View` the shipped method builds), plus
//! `self.skip_varn` / `self.group_size` → `v.…` and `self.scratch_unpack` →
//! the caller's `scratch`. Hadamard is applied by the caller, as the shipped
//! kernels leave it to the method.
//!
//! Shared by the in-crate bit-identity oracle (`src/kvarn/dequant_oracle_tests.rs`,
//! which can also force the var-norm/grouping modes `with_config` never
//! produces) and the paired A/B gate (`tests/bench_894_kvarn_dequant_zip_goat.rs`).
//! The includer must have `KVarNKeyColView`, `KVarNValueRowView`,
//! `unpack_row` and `unpack_value` in scope.
//!
//! Do NOT "tidy" this file: its only job is to be the old code.

#![allow(dead_code, clippy::needless_range_loop, clippy::all)]

use super::{KVarNKeyColView, KVarNValueRowView, unpack_row, unpack_value};

/// Pre-894 `dequantize_key_into` kernel body (no Hadamard).
#[inline]
#[rustfmt::skip] // verbatim pre-894 text — keep the old line breaks
pub fn old_dequantize_key(v: &KVarNKeyColView<'_>, out: &mut [f32]) {
    let pos_in_tile = v.pos_in_tile;
    let actual_cols = v.actual_cols;
    let bits = v.bits as usize;
    let bpr = v.bpr;
    let kv_dim = v.kv_dim;
    let quantized = v.quantized;
    let var_col = v.var_col;
    let rtn_scales = v.rtn_scales;
    let rtn_zp = v.rtn_zp;
    let s_row = v.s_row;

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
            if v.skip_varn {
                if v.group_size > 0 {
                    // Grouped quantization: find group for this position
                    let groups_per_row = actual_cols.div_ceil(v.group_size);
                    let g = pos_in_tile / v.group_size;
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
}

/// Pre-894 `dequantize_value_into` kernel body (no Hadamard).
#[inline]
#[rustfmt::skip] // verbatim pre-894 text — keep the old line breaks
pub fn old_dequantize_value(v: &KVarNValueRowView<'_>, scratch: &mut [u32], out: &mut [f32]) {
    let pos_in_tile = v.pos_in_tile;
    let bits = v.bits as usize;
    let kv_dim = v.kv_dim;
    let packed_row = v.packed_row;
    let var_row = v.var_row;
    let rtn_scales = v.rtn_scales;
    let rtn_zp = v.rtn_zp;
    let s_col = v.s_col;

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
            if v.skip_varn {
                if v.group_size > 0 {
                    // Grouped quantization: per-token, per-channel-group scales.
                    //
                    // Fast path: group_size == 4 means each byte covers exactly
                    // one group (4 values / 4 = 1 byte per group). This is the
                    // only configuration that sets group_size > 0 (see with_config:
                    // `group_size: if cfg.bits <= 2 { 4 } else { 0 }`), so we can
                    // specialize the branch-free inner loop. The original code
                    // had 3 `if 4*i+k < kv_dim` checks per byte.
                    debug_assert_eq!(
                        v.group_size, 4,
                        "group_size>0 implies group_size==4 at 2-bit"
                    );
                    let groups_per_row = kv_dim.div_ceil(v.group_size);
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
            let scratch = &mut scratch[..kv_dim];
            unpack_row(packed_row, bits, scratch);
            for ch in 0..kv_dim {
                let q = scratch[ch] as f32;
                out[ch] = q.mul_add(rtn_scale, rtn_zp_val) * s_col[ch] * var_row;
            }
        }
    }
}
