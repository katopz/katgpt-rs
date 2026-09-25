//! Issue 894 — KVarN dequant kernels rewritten as bounds-check-free zips.
//!
//! The pre-894 `dequantize_{value,key}_into` loops indexed `out[2*i]`,
//! `s_col[2*i]`, … with no slice length pinned, so every access kept its
//! bounds check and the loops ran scalar. The rewrite (`src/kvarn/dequant.rs`)
//! keeps every per-element op and its order; this gate holds it to that and
//! measures what it buys. The OLD loops are the oracle, kept verbatim in
//! `tests/common/kvarn_dequant_oracle.rs` and driven through the same public
//! `KVarN{ValueRow,KeyCol}View` the shipped kernel reads.
//!
//! G3  bit-identity, old vs new, over the configurations `with_config`
//!     produces: bits {2, 4, 8} × Hadamard on/off × kv_dim {128, 37} × a
//!     partial last tile, value rows AND key columns. 0 differing bits
//!     (`to_bits`). The var-norm/grouping modes `with_config` never selects
//!     are covered by the in-crate T1 oracle
//!     (`src/kvarn/dequant_oracle_tests.rs`).
//! G4  the shipped dequant passes allocate 0 times.
//! G2  the plain decode pass (T=4096, kv_dim 128, dequant + axpy), paired
//!     interleaved `ab_median_ratio`, candidate = shipped, baseline = old:
//!     the 4-bit VALUE arm (the issue's) must be ≥ 10% faster
//!     (median new/old ≤ 0.90), and every other rewritten arm (value 2/8,
//!     key 2/4/8) must not regress (median ≤ 1.05). An A/A control reports
//!     the protocol's floor.
//!
//! Run (perf numbers need --release):
//!   cargo test -p katgpt-kv --release --features kvarn \
//!     --test bench_894_kvarn_dequant_zip_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::RefCell;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

#[path = "common/kvarn_dequant_oracle.rs"]
mod oracle;

use ab_timing::ab_median_ratio;
use katgpt_kv::kvarn::hadamard::hadamard_transform_inplace;
use katgpt_kv::kvarn::kv_cache::KVarNConfig;
use katgpt_kv::kvarn::{
    KVarNKVCache, KVarNKeyColView, KVarNValueRowView, unpack_row, unpack_value,
};

// ── G4: counting allocator ────────────────────────────────────────────────

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

static FAILS: AtomicUsize = AtomicUsize::new(0);

fn gate(name: &str, ok: bool, detail: String) {
    println!("  [{}] {name}: {detail}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        FAILS.fetch_add(1, Ordering::Relaxed);
    }
}

// ── fixture ───────────────────────────────────────────────────────────────

const D: usize = 128;
const TILE: usize = 128;
const T: usize = 4096; // 32 full tiles

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn gauss(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

fn filled(bits: u8, kv_dim: usize, max_seq: usize, hadamard: bool, seed: u64) -> KVarNKVCache {
    let mut c = KVarNKVCache::with_config(&KVarNConfig {
        n_layers: 1,
        kv_dim,
        max_seq_len: max_seq,
        bits,
        tile_size: TILE,
        hadamard,
        ..KVarNConfig::default()
    });
    let mut rng = Rng(seed);
    let mut v = vec![0.0f32; kv_dim];
    for p in 0..max_seq {
        for (ch, x) in v.iter_mut().enumerate() {
            *x = rng.gauss() * (1.0 + (ch % 5) as f32);
        }
        c.store_key(0, p, &v);
        for x in v.iter_mut() {
            *x = rng.gauss();
        }
        c.store_value(0, p, &v);
    }
    c
}

fn main() {
    println!("Issue 894 — KVarN zip dequant GOAT");
    g3_bit_identity();
    g4_allocs();
    g2_kernel();
    let fails = FAILS.load(Ordering::Relaxed);
    println!(
        "\n{}",
        if fails == 0 {
            "ALL GATES PASS".to_string()
        } else {
            format!("{fails} GATE(S) FAILED")
        }
    );
    if fails != 0 {
        std::process::exit(1);
    }
}

// ── G3 ────────────────────────────────────────────────────────────────────

fn g3_bit_identity() {
    println!("\nG3 — old vs new, bitwise (with_config modes)");
    for bits in [2u8, 4, 8] {
        for hadamard in [false, true] {
            for (kv_dim, max_seq) in [(D, 3 * TILE + 45), (37, 2 * TILE + 1)] {
                let mut c = filled(bits, kv_dim, max_seq, hadamard, 0x894 ^ bits as u64);
                let had = hadamard && kv_dim.is_power_of_two();
                let (mut got, mut want) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
                let mut scratch = vec![0u32; kv_dim];
                let (mut n, mut diff) = (0usize, 0usize);
                for p in 0..max_seq {
                    let kv: KVarNKeyColView<'_> = c.key_col_view(0, p).expect("stored");
                    oracle::old_dequantize_key(&kv, &mut want);
                    if had {
                        hadamard_transform_inplace(&mut want);
                    }
                    c.dequantize_key_into(0, p, &mut got);
                    diff += got
                        .iter()
                        .zip(&want)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let vv: KVarNValueRowView<'_> = c.value_row_view(0, p).expect("stored");
                    oracle::old_dequantize_value(&vv, &mut scratch, &mut want);
                    if had {
                        hadamard_transform_inplace(&mut want);
                    }
                    c.dequantize_value_into(0, p, &mut got);
                    diff += got
                        .iter()
                        .zip(&want)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    n += 2 * kv_dim;
                }
                gate(
                    &format!("G3 bits={bits} hadamard={hadamard} kv_dim={kv_dim}"),
                    diff == 0,
                    format!(
                        "{diff} of {n} elements differ bitwise (K+V, {max_seq} positions incl. a partial tile)"
                    ),
                );
            }
        }
    }
}

// ── G4 ────────────────────────────────────────────────────────────────────

fn g4_allocs() {
    println!("\nG4 — shipped dequant allocations");
    for bits in [2u8, 4, 8] {
        let mut c = filled(bits, D, 2 * TILE, false, 7);
        let mut buf = vec![0.0f32; D];
        let mut sink = 0.0f32;
        let b0 = allocs();
        for p in 0..2 * TILE {
            c.dequantize_key_into(0, p, &mut buf);
            sink += buf[p % D];
            c.dequantize_value_into(0, p, &mut buf);
            sink += buf[p % D];
        }
        let n = allocs() - b0;
        black_box(sink);
        gate(
            &format!("G4 bits={bits}"),
            n == 0,
            format!("{n} allocs over {} K+V dequants", 4 * TILE),
        );
    }
}

// ── G2 ────────────────────────────────────────────────────────────────────

/// One plain decode pass: dequant every position, weighted-accumulate it.
/// `OLD` runs the verbatim pre-894 loop over the shipped view; `KEY` picks
/// key columns instead of value rows. Const params keep the arm choice out
/// of the timed loop.
fn pass<const OLD: bool, const KEY: bool>(
    c: &RefCell<KVarNKVCache>,
    w: &[f32],
    i: usize,
    bufs: &mut Bufs,
) -> f32 {
    let mut c = c.borrow_mut();
    let Bufs { buf, acc, scratch } = bufs;
    acc.fill(0.0);
    let s = 1.0 + (i % 3) as f32 * 1e-3;
    for (p, &wv) in w.iter().enumerate() {
        let p = black_box(p);
        match (KEY, OLD) {
            (false, true) => {
                let v = c.value_row_view(0, p).expect("stored");
                oracle::old_dequantize_value(black_box(&v), scratch, black_box(&mut *buf));
            }
            (false, false) => c.dequantize_value_into(0, p, black_box(&mut *buf)),
            (true, true) => {
                let v = c.key_col_view(0, p).expect("stored");
                oracle::old_dequantize_key(black_box(&v), black_box(&mut *buf));
            }
            (true, false) => c.dequantize_key_into(0, p, black_box(&mut *buf)),
        }
        let wp = black_box(wv * s);
        for (a, &x) in acc.iter_mut().zip(buf.iter()) {
            *a += wp * x;
        }
    }
    black_box(acc[i % D])
}

struct Bufs {
    buf: Vec<f32>,
    acc: Vec<f32>,
    scratch: Vec<u32>,
}

fn bufs() -> Bufs {
    Bufs {
        buf: vec![0.0; D],
        acc: vec![0.0; D],
        scratch: vec![0; D],
    }
}

type PassFn = fn(&RefCell<KVarNKVCache>, &[f32], usize, &mut Bufs) -> f32;

fn g2_kernel() {
    println!("\nG2 — plain decode pass (dequant + axpy), T={T}, kv_dim={D}; ratio = new / old");
    let w: Vec<f32> = (0..T).map(|p| 1.0 / (1.0 + (p % 97) as f32)).collect();
    let (mut ka, mut kb) = (0.0f32, 0.0f32);
    let arms: [(&str, u8, PassFn, PassFn); 6] = [
        ("value", 4, pass::<true, false>, pass::<false, false>),
        ("value", 2, pass::<true, false>, pass::<false, false>),
        ("value", 8, pass::<true, false>, pass::<false, false>),
        ("key", 4, pass::<true, true>, pass::<false, true>),
        ("key", 2, pass::<true, true>, pass::<false, true>),
        ("key", 8, pass::<true, true>, pass::<false, true>),
    ];
    for (side, bits, old, new) in arms {
        let c = RefCell::new(filled(bits, D, T, false, 0x0894_0002 ^ bits as u64));
        let (mut ba, mut bb) = (bufs(), bufs());
        let r = ab_median_ratio(
            21,
            10,
            3,
            |i| ka += old(&c, &w, i, &mut ba),
            |i| kb += new(&c, &w, i, &mut bb),
        );
        let name = format!("{side} {bits}-bit");
        r.report(&format!("G2 {name} new/old"));
        let detail = format!(
            "median {:.4} = {:.2}x faster (rounds {:.4}–{:.4}; old {:.1} µs, new {:.1} µs per {T}-position pass)",
            r.median,
            1.0 / r.median,
            r.min(),
            r.max(),
            r.a_ns_per_iter() / 1e3,
            r.b_ns_per_iter() / 1e3,
        );
        match (side, bits) {
            ("value", 4) => {
                gate(
                    "G2 value 4-bit ≥ 10% faster (≤ 0.90)",
                    r.median <= 0.90,
                    detail,
                );
                let (mut b2, mut b3) = (bufs(), bufs());
                let aa = ab_median_ratio(
                    21,
                    10,
                    3,
                    |i| ka += new(&c, &w, i, &mut b2),
                    |i| kb += new(&c, &w, i, &mut b3),
                );
                aa.report("G2 report A/A control (new/new, value 4-bit)");
            }
            _ => gate(
                &format!("G2 {name} no regression (≤ 1.05)"),
                r.median <= 1.05,
                detail,
            ),
        }
    }
    black_box(ka + kb);
}
