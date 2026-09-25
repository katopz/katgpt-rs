//! Issue 896 regression tests — per-layer raw tile buffers, and exact reads of
//! the in-progress tile.
//!
//! Before 896, `KVarNKVCache` had ONE raw tile buffer shared by every layer,
//! so decode-order stores (for each position, for each layer) quantized the
//! LAST layer's data into every layer; and a tile that had not been quantized
//! yet was read through its empty (or, after `reset()`, a previous
//! sequence's) scale metadata — zeros, or an index panic for 2-bit keys.
//!
//! The reference for a quantized tile is the SAME cache fed layer-major, the
//! flow that was always correct and is pinned bit-for-bit to the verbatim
//! pre-894 oracle by `dequant_oracle_tests` (n_layers = 2, layer-major).
//! A tile that is not quantized yet must read back the stored input exactly
//! (`to_bits` equality): the raw buffer holds it before Hadamard / var-norm /
//! RTN, so no inverse transform applies.

#![allow(clippy::needless_range_loop)] // positions index the cache as well as the inputs

use super::kv_cache::{KVarNConfig, KVarNKVCache};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Values incl. exact zeros, negatives and a per-layer offset, so a
    /// cross-layer mix-up cannot hide behind similar magnitudes.
    fn val(&mut self, layer: usize) -> f32 {
        let r = self.next();
        match r % 13 {
            0 => 0.0,
            1 => -0.0,
            _ => {
                let u = (r >> 11) as f64 / (1u64 << 53) as f64;
                ((u - 0.5) * 6.0 + 10.0 * layer as f64) as f32
            }
        }
    }
}

/// Every mode the dequant arms branch on: (skip_varn, group_size).
const MODES: [(bool, usize); 4] = [(true, 4), (true, 0), (false, 0), (false, 4)];

fn modes_for(bits: u8) -> impl Iterator<Item = (bool, usize)> {
    // 4/8-bit dequant ignores grouping (same filter as the 894 oracle).
    MODES.into_iter().filter(move |m| bits == 2 || m.1 == 0)
}

#[allow(clippy::too_many_arguments)]
fn cache(
    bits: u8,
    kv_dim: usize,
    tile: usize,
    max_seq: usize,
    n_layers: usize,
    mode: (bool, usize),
    hadamard: bool,
) -> KVarNKVCache {
    let mut c = KVarNKVCache::with_config(&KVarNConfig {
        n_layers,
        kv_dim,
        max_seq_len: max_seq,
        bits,
        tile_size: tile,
        hadamard,
        ..KVarNConfig::default()
    });
    c.set_quant_mode_for_test(mode.0, mode.1);
    c
}

/// `[layer][pos] -> (key, value)`.
type Inputs = Vec<Vec<(Vec<f32>, Vec<f32>)>>;

fn inputs(n_layers: usize, stored: usize, kv_dim: usize, seed: u64) -> Inputs {
    let mut rng = Rng(seed);
    (0..n_layers)
        .map(|l| {
            (0..stored)
                .map(|_| {
                    let k = (0..kv_dim).map(|_| rng.val(l)).collect();
                    let v = (0..kv_dim).map(|_| rng.val(l) * 0.5).collect();
                    (k, v)
                })
                .collect()
        })
        .collect()
}

fn store_layer_major(c: &mut KVarNKVCache, inp: &Inputs) {
    for (l, rows) in inp.iter().enumerate() {
        for (p, (k, v)) in rows.iter().enumerate() {
            c.store_key(l, p, k);
            c.store_value(l, p, v);
        }
    }
}

fn differing(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// Decode-order stores into `a`; after EVERY position, read every stored
/// position of every layer. Quantized tiles must equal the layer-major
/// reference `b` bitwise; in-progress tiles must equal the input exactly.
/// Returns `(elements compared, differing)`.
#[allow(clippy::too_many_arguments)]
fn decode_order_case(
    bits: u8,
    kv_dim: usize,
    tile: usize,
    max_seq: usize,
    stored: usize,
    n_layers: usize,
    mode: (bool, usize),
    hadamard: bool,
) -> (usize, usize) {
    let inp = inputs(
        n_layers,
        stored,
        kv_dim,
        0x896 ^ ((bits as u64) << 40) ^ kv_dim as u64,
    );
    let mut b = cache(bits, kv_dim, tile, max_seq, n_layers, mode, hadamard);
    store_layer_major(&mut b, &inp);
    let mut a = cache(bits, kv_dim, tile, max_seq, n_layers, mode, hadamard);
    let (mut got, mut want) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
    let (mut n, mut diff) = (0usize, 0usize);
    for p in 0..stored {
        for (l, rows) in inp.iter().enumerate() {
            a.store_key(l, p, &rows[p].0);
            a.store_value(l, p, &rows[p].1);
        }
        for (l, rows) in inp.iter().enumerate() {
            for q in 0..=p {
                // A tile is quantized exactly when it has filled, or at the
                // final position of the sequence.
                let expect_q = (q / tile + 1) * tile <= p + 1 || p == max_seq - 1;
                let kq = a.key_col_view(l, q).is_some();
                let vq = a.value_row_view(l, q).is_some();
                assert_eq!(
                    (kq, vq),
                    (expect_q, expect_q),
                    "quantized-state mismatch l={l} q={q} after p={p}"
                );
                // ── key ──
                if kq {
                    b.dequantize_key_into(l, q, &mut want);
                } else {
                    want.copy_from_slice(&rows[q].0);
                }
                got.fill(f32::NAN);
                a.dequantize_key_into(l, q, &mut got);
                diff += differing(&got, &want);
                // ── value ──
                if vq {
                    b.dequantize_value_into(l, q, &mut want);
                } else {
                    want.copy_from_slice(&rows[q].1);
                }
                got.fill(f32::NAN);
                a.dequantize_value_into(l, q, &mut got);
                diff += differing(&got, &want);
                n += 2 * kv_dim;
            }
        }
    }
    (n, diff)
}

#[test]
fn issue_896_t3_decode_order_multi_layer_matches_layer_major_bitwise() {
    let (mut total, mut cases) = (0usize, 0usize);
    for n_layers in [2usize, 3] {
        for bits in [2u8, 3, 4, 8] {
            for kv_dim in [16usize, 37] {
                // (tile, max_seq, stored): partial LAST tile quantized at
                // max_seq − 1; full tiles then a trailing in-progress tile;
                // full tiles only.
                for (tile, max_seq, stored) in
                    [(8usize, 29usize, 29usize), (8, 64, 29), (16, 48, 48)]
                {
                    for mode in modes_for(bits) {
                        for hadamard in [false, true] {
                            let (n, d) = decode_order_case(
                                bits, kv_dim, tile, max_seq, stored, n_layers, mode, hadamard,
                            );
                            assert_eq!(
                                d, 0,
                                "n_layers={n_layers} bits={bits} kv_dim={kv_dim} tile={tile} \
                                 max_seq={max_seq} stored={stored} mode={mode:?} \
                                 hadamard={hadamard}: {d} of {n} elements differ bitwise"
                            );
                            total += n;
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        cases >= 200 && total > 5_000_000,
        "coverage collapsed: {cases} cases, {total} elements"
    );
    println!("Issue 896 T3 decode-order: {cases} cases, {total} elements, 0 differing bits");
}

#[test]
fn issue_896_t3_full_tile_then_partial() {
    for n_layers in [1usize, 2, 3] {
        for bits in [2u8, 3, 4, 8] {
            for mode in modes_for(bits) {
                for hadamard in [false, true] {
                    let (kv_dim, tile, max_seq, stored) = (16, 8, 64, 8 + 5);
                    let inp = inputs(n_layers, stored, kv_dim, 0xF7 ^ bits as u64);
                    // Reference: a single-tile sequence of just tile 0.
                    let mut r = cache(bits, kv_dim, tile, max_seq, n_layers, mode, hadamard);
                    let head: Inputs = inp.iter().map(|rows| rows[..tile].to_vec()).collect();
                    store_layer_major(&mut r, &head);
                    let mut c = cache(bits, kv_dim, tile, max_seq, n_layers, mode, hadamard);
                    for p in 0..stored {
                        for (l, rows) in inp.iter().enumerate() {
                            c.store_key(l, p, &rows[p].0);
                            c.store_value(l, p, &rows[p].1);
                        }
                    }
                    let (mut got, mut want) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
                    for (l, rows) in inp.iter().enumerate() {
                        for p in 0..stored {
                            let full = p < tile;
                            assert_eq!(c.key_col_view(l, p).is_some(), full);
                            assert_eq!(c.value_row_view(l, p).is_some(), full);
                            for key in [true, false] {
                                match (full, key) {
                                    (true, true) => r.dequantize_key_into(l, p, &mut want),
                                    (true, false) => r.dequantize_value_into(l, p, &mut want),
                                    (false, true) => want.copy_from_slice(&rows[p].0),
                                    (false, false) => want.copy_from_slice(&rows[p].1),
                                }
                                got.fill(f32::NAN);
                                if key {
                                    c.dequantize_key_into(l, p, &mut got);
                                } else {
                                    c.dequantize_value_into(l, p, &mut got);
                                }
                                assert_eq!(
                                    differing(&got, &want),
                                    0,
                                    "n_layers={n_layers} bits={bits} mode={mode:?} \
                                     hadamard={hadamard} l={l} p={p} key={key}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn issue_896_t3_reset_then_partial_tile_never_reads_stale_scales() {
    let (kv_dim, tile, max_seq, n_layers) = (16usize, 8usize, 64usize, 2usize);
    for bits in [2u8, 3, 4, 8] {
        for mode in modes_for(bits) {
            for hadamard in [false, true] {
                let mut c = cache(bits, kv_dim, tile, max_seq, n_layers, mode, hadamard);
                // Sequence 1: two quantized tiles + an in-progress one, decode order.
                let s1 = inputs(n_layers, 20, kv_dim, 0x51);
                for p in 0..20 {
                    for (l, rows) in s1.iter().enumerate() {
                        c.store_key(l, p, &rows[p].0);
                        c.store_value(l, p, &rows[p].1);
                    }
                }
                c.reset();
                // Sequence 2: a partial tile only — every read must be exact,
                // unstored slots read 0, and no tile may expose old scales.
                let s2 = inputs(n_layers, 20, kv_dim, 0x52);
                for p in 0..5 {
                    for (l, rows) in s2.iter().enumerate() {
                        c.store_key(l, p, &rows[p].0);
                        c.store_value(l, p, &rows[p].1);
                    }
                }
                let mut got = vec![0.0f32; kv_dim];
                let zero = vec![0.0f32; kv_dim];
                for (l, rows) in s2.iter().enumerate() {
                    for p in 0..20 {
                        assert!(
                            c.key_col_view(l, p).is_none(),
                            "stale key scales l={l} p={p}"
                        );
                        assert!(
                            c.value_row_view(l, p).is_none(),
                            "stale value scales l={l} p={p}"
                        );
                        let (wk, wv) = if p < 5 {
                            (&rows[p].0, &rows[p].1)
                        } else {
                            (&zero, &zero)
                        };
                        got.fill(f32::NAN);
                        c.dequantize_key_into(l, p, &mut got);
                        assert_eq!(
                            differing(&got, wk),
                            0,
                            "bits={bits} {mode:?} key l={l} p={p}"
                        );
                        got.fill(f32::NAN);
                        c.dequantize_value_into(l, p, &mut got);
                        assert_eq!(
                            differing(&got, wv),
                            0,
                            "bits={bits} {mode:?} val l={l} p={p}"
                        );
                    }
                }
                // Continue sequence 2 through two full tiles: the re-quantized
                // tiles must equal a fresh cache fed sequence 2 alone.
                for p in 5..20 {
                    for (l, rows) in s2.iter().enumerate() {
                        c.store_key(l, p, &rows[p].0);
                        c.store_value(l, p, &rows[p].1);
                    }
                }
                let mut fresh = cache(bits, kv_dim, tile, max_seq, n_layers, mode, hadamard);
                store_layer_major(&mut fresh, &s2);
                let mut want = vec![0.0f32; kv_dim];
                for l in 0..n_layers {
                    for p in 0..20 {
                        fresh.dequantize_key_into(l, p, &mut want);
                        c.dequantize_key_into(l, p, &mut got);
                        assert_eq!(differing(&got, &want), 0, "post-reset key l={l} p={p}");
                        fresh.dequantize_value_into(l, p, &mut want);
                        c.dequantize_value_into(l, p, &mut got);
                        assert_eq!(differing(&got, &want), 0, "post-reset val l={l} p={p}");
                    }
                }
            }
        }
    }
}

/// The measured Finding-1 probe: 2 layers, kv_dim 16, tile 128, 8-bit,
/// decode-order stores; layer 0 in [1, 13], layer 1 in [−54, −50]. Pre-896,
/// layer 0 served layer 1's data (L0 p3 value read −54.01, −50.02, …).
#[test]
fn issue_896_probe_decode_order_layer0_serves_its_own_data() {
    let (kv_dim, tile) = (16usize, 128usize);
    let l0 = |p: usize, ch: usize| 1.0 + ((p + ch) % 13) as f32;
    let l1 = |p: usize, ch: usize| -50.0 - ((p + ch) % 5) as f32;
    let mut c = KVarNKVCache::with_config(&KVarNConfig {
        n_layers: 2,
        kv_dim,
        max_seq_len: tile,
        bits: 8,
        tile_size: tile,
        ..KVarNConfig::default()
    });
    for p in 0..tile {
        for (l, f) in [(0usize, &l0 as &dyn Fn(usize, usize) -> f32), (1, &l1)] {
            let row: Vec<f32> = (0..kv_dim).map(|ch| f(p, ch)).collect();
            c.store_key(l, p, &row);
            c.store_value(l, p, &row);
        }
    }
    let mut out = vec![0.0f32; kv_dim];
    for (l, f) in [(0usize, &l0 as &dyn Fn(usize, usize) -> f32), (1, &l1)] {
        for p in [3usize, 64, 127] {
            assert!(c.key_col_view(l, p).is_some() && c.value_row_view(l, p).is_some());
            for key in [true, false] {
                if key {
                    c.dequantize_key_into(l, p, &mut out);
                } else {
                    c.dequantize_value_into(l, p, &mut out);
                }
                for (ch, &got) in out.iter().enumerate() {
                    let want = f(p, ch);
                    assert!(
                        (got - want).abs() < 0.25,
                        "layer {l} p{p} key={key} ch{ch}: got {got}, want {want} \
                         (Issue 896: a shared raw buffer serves the last layer's data)"
                    );
                }
            }
        }
    }
}

/// The measured Finding-2 probe: kv_dim 8, 10 of 128 positions stored,
/// values 1..8. Pre-896, 2- and 4-bit value reads at p5 returned all zeros
/// and a 2-bit key read panicked (index out of range).
#[test]
fn issue_896_probe_in_progress_tile_reads_exact_at_2_and_4_bits() {
    let kv_dim = 8usize;
    for bits in [2u8, 3, 4, 8] {
        let mut c = KVarNKVCache::with_config(&KVarNConfig {
            kv_dim,
            bits,
            ..KVarNConfig::default()
        });
        let row: Vec<f32> = (1..=kv_dim).map(|x| x as f32).collect();
        for p in 0..10 {
            c.store_key(0, p, &row);
            c.store_value(0, p, &row);
        }
        let mut out = vec![0.0f32; kv_dim];
        c.dequantize_value_into(0, 5, &mut out);
        assert_eq!(
            out, row,
            "bits={bits}: in-progress value read (pre-896: all zeros)"
        );
        c.dequantize_key_into(0, 5, &mut out);
        assert_eq!(
            out, row,
            "bits={bits}: in-progress key read (pre-896: 2-bit panicked)"
        );
        c.dequantize_key_into(0, 10, &mut out);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "bits={bits}: unstored slot must read 0"
        );
    }
}
