//! Issue 894 T1 — bit-identity of the zip-rewritten dequant kernels against
//! the pre-894 indexed loops (kept verbatim in `tests/common/kvarn_dequant_oracle.rs`).
//!
//! Coverage: bits {2, 4, 8} (+ 3 as the unchanged generic fallback) × every
//! quantize/dequant mode — including the var-norm-on 2-bit and var-norm-off
//! 4/8-bit arms that `with_config` never selects, reached via
//! `set_quant_mode_for_test` — × odd and even `kv_dim` × full and PARTIAL
//! tiles (a `max_seq_len` that is not a tile multiple) × Hadamard on/off,
//! for both value rows and key columns. The rule is `to_bits` equality on
//! every element: 0 differing bits.
//!
//! Positions of a counted-but-not-yet-quantized tile used to read the tile's
//! EMPTY metadata (zeros; a panic in the grouped 2-bit arms). Since Issue 896
//! such a tile has no view (`key_col_view` / `value_row_view` return `None`)
//! and is served from the layer's raw buffer, so those positions are asserted
//! to read back the STORED input exactly (`to_bits` equality) instead of
//! being compared against the old loops. Every quantized-tile read is still
//! bit-identical to the oracle: this is the layer-major pin Issue 896 T1
//! requires (the stores below are layer-major, n_layers = 2).

#![allow(clippy::needless_range_loop)] // `p` indexes the cache as well as `inputs`

use super::hadamard::hadamard_transform_inplace;
use super::kv_cache::{KVarNConfig, KVarNKVCache, unpack_row, unpack_value};
use super::{KVarNKeyColView, KVarNValueRowView};

#[path = "../../tests/common/kvarn_dequant_oracle.rs"]
mod oracle;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Heavy-tailed-ish values incl. exact zeros, negatives and constant rows
    /// (degenerate-range RTN), so scale/zp edge cases are exercised.
    fn val(&mut self, ch: usize) -> f32 {
        let r = self.next();
        match r % 17 {
            0 => 0.0,
            1 => -0.0,
            2 => 1.5,
            _ => {
                let u = (r >> 11) as f64 / (1u64 << 53) as f64;
                ((u - 0.5) * 8.0 * (1.0 + (ch % 7) as f64)) as f32
            }
        }
    }
}

/// Every mode the dequant arms branch on: (skip_varn, group_size).
const MODES: [(bool, usize); 4] = [(true, 4), (true, 0), (false, 0), (false, 4)];

fn run_case(
    bits: u8,
    kv_dim: usize,
    tile: usize,
    max_seq: usize,
    stored: usize,
    mode: (bool, usize),
    hadamard: bool,
) -> (usize, usize) {
    let mut c = KVarNKVCache::with_config(&KVarNConfig {
        n_layers: 2,
        kv_dim,
        max_seq_len: max_seq,
        bits,
        tile_size: tile,
        hadamard,
        ..KVarNConfig::default()
    });
    c.set_quant_mode_for_test(mode.0, mode.1);
    let mut rng = Rng(0x894 ^ ((bits as u64) << 32) ^ ((kv_dim as u64) << 16) ^ tile as u64);
    let mut v = vec![0.0f32; kv_dim];
    // Stored inputs, `[layer][pos] -> (key, value)`, for the in-progress reads.
    let mut inputs: Vec<Vec<(Vec<f32>, Vec<f32>)>> = (0..2).map(|_| Vec::with_capacity(stored)).collect();
    for (layer, inp) in inputs.iter_mut().enumerate() {
        for p in 0..stored {
            let constant = p % 11 == 5;
            for (ch, x) in v.iter_mut().enumerate() {
                *x = if constant { 0.25 } else { rng.val(ch) };
            }
            c.store_key(layer, p, &v);
            let key = v.clone();
            for x in v.iter_mut() {
                if !constant {
                    *x = rng.val(kv_dim - 1) * 0.5;
                }
            }
            c.store_value(layer, p, &v);
            inp.push((key, v.clone()));
        }
    }
    let had = hadamard && kv_dim.is_power_of_two();
    let (mut got, mut want) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
    let mut scratch = vec![0u32; kv_dim];
    let (mut compared, mut differing) = (0usize, 0usize);
    let mut cmp = |got: &[f32], want: &[f32]| {
        compared += got.len();
        differing += got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    };
    for layer in 0..2 {
        for p in 0..stored {
            let (key_in, val_in) = &inputs[layer][p];
            // ── key column ──
            match c.key_col_view(layer, p) {
                Some(view) => {
                    let view: KVarNKeyColView<'_> = view;
                    oracle::old_dequantize_key(&view, &mut want);
                    if had {
                        hadamard_transform_inplace(&mut want);
                    }
                }
                // In-progress tile (Issue 896): exact raw read, no transform.
                None => want.copy_from_slice(key_in),
            }
            got.fill(f32::NAN);
            c.dequantize_key_into(layer, p, &mut got);
            cmp(&got, &want);
            // ── value row ──
            let Some(view) = c.value_row_view(layer, p) else {
                want.copy_from_slice(val_in);
                got.fill(f32::NAN);
                c.dequantize_value_into(layer, p, &mut got);
                cmp(&got, &want);
                continue;
            };
            let view: KVarNValueRowView<'_> = view;
            oracle::old_dequantize_value(&view, &mut scratch, &mut want);
            if had {
                hadamard_transform_inplace(&mut want);
            }
            got.fill(f32::NAN);
            c.dequantize_value_into(layer, p, &mut got);
            cmp(&got, &want);
        }
    }
    (compared, differing)
}

#[test]
fn issue_894_t1_zip_dequant_is_bit_identical_to_the_old_loops() {
    let mut total = 0usize;
    let mut cases = 0usize;
    for bits in [2u8, 3, 4, 8] {
        for kv_dim in [1usize, 2, 3, 5, 7, 36, 37, 38, 39, 64, 128, 130] {
            // (tile, max_seq, stored): full tiles; partial LAST tile quantized
            // at max_seq − 1; a counted-but-unquantized trailing tile.
            for (tile, max_seq, stored) in [
                (16usize, 64usize, 64usize),
                (16, 45, 45),
                (8, 64, 29),
                (128, 128, 128),
            ] {
                for mode in MODES {
                    // 4/8-bit dequant ignores grouping; quantizing grouped there
                    // would only change the scale layout the arm never reads.
                    if bits != 2 && mode.1 != 0 {
                        continue;
                    }
                    for hadamard in [false, true] {
                        let (n, d) = run_case(bits, kv_dim, tile, max_seq, stored, mode, hadamard);
                        assert_eq!(
                            d, 0,
                            "bits={bits} kv_dim={kv_dim} tile={tile} max_seq={max_seq} stored={stored} \
                             mode={mode:?} hadamard={hadamard}: {d} of {n} elements differ bitwise"
                        );
                        total += n;
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(
        cases >= 400 && total > 1_000_000,
        "oracle coverage collapsed: {cases} cases, {total} elements"
    );
    println!("Issue 894 T1: {cases} cases, {total} elements, 0 differing bits");
}
