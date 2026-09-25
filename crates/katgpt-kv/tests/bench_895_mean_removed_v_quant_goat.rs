//! Issue 883 P1 — token-mean-removed V-cache quantization over the REAL
//! KVarN backend (Bench 895, primitive half).
//!
//! `MeanRemovedValueCache<KVarNKVCache>` stores `q(V − E^V_l[s])` and reads
//! `dequant + E^V_l[s]`; the table is fitted by the P0 substrate
//! (`LayeredVkCalibration` → `FittedTokenTable::from_calibration`) on a
//! calibration split and every gate reads a held-out split.
//!
//! G1a prediction vs measurement (the issue's arbiter): per-token planted
//!     means `μ_s ~ N(0, σ_μ²)` (centered under the sampling law) + noise
//!     `N(0, 1)`; the dashboard's variance reduction predicts the quant-MSE
//!     ratio `removed/plain = 1 − ρ_V` (scale-equivariant RTN on Gaussian
//!     rows ⇒ MSE ∝ row variance). Tolerance ±10% relative, every cell of
//!     ρ ∈ {0.25, 0.5, 0.75} × bits ∈ {2, 4, 8}.
//! G1c the absmax caveat — an OFF-MEAN occurrence (a token whose occurrences
//!     are bimodal: 75% carry a +20σ spike on 4 channels, 25% do not; the
//!     mean sits between). Mean removal GROWS the no-spike rows' range.
//!     REPORT (recorded whichever way it goes), with the prediction beside.
//! G1d trap 3 — a sink token (the first position of every tile, a
//!     CONSISTENT +20σ massive activation): the table absorbs the sink mean.
//!     Gate: neither the sink rows nor the non-sink rows of sink-bearing
//!     tiles regress (MSE ratio ≤ 1.0 for both).
//! G2  the decode V-aggregation kernel (dequant + weighted accumulate over
//!     T=4096 positions, 4-bit): fused `accumulate_value` vs the plain
//!     KVarN dequant+axpy, paired interleave; bar ≤ 1.01 (the issue's
//!     "fused dequant+add ≤ +1% kernel time"). The two-pass decorator read
//!     (`dequantize_value_into` then axpy) is reported beside it, plus
//!     three REPORT arms that locate the cost: an A/A control (the
//!     protocol's floor on this box), a lookup-only miss-table arm (the
//!     per-position token → row resolution alone), and the deferred
//!     restore `Σ_p w_p·v̂_p + Σ_s W_s·E[s]` (the named next lever).
//! G3  a table that misses every token, and an all-`+0.0` table, dequantize
//!     bit-identically to the undecorated backend (bits 2 and 4).
//! G3b a non-zero table: decorated read + accumulate bitwise vs the unfused
//!     reference, bits 2/4/8 + Hadamard.
//! G4  the decorator adds 0 allocations over the backend's own store+read
//!     loop.
//!
//! The PPL-at-matched-bits + per-family retention walk halves of G1 need a
//! model and are riir-infer Issue 013's.
//!
//! Run:
//!   cargo test -p katgpt-kv --release --features kvarn,fitted_value_tables \
//!     --test bench_895_mean_removed_v_quant_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_core::fitted_anchor_table::LayeredVkCalibration;
use katgpt_core::fitted_value_table::{FittedTokenTable, MeanRemovedValueCache, VkSignal};
use katgpt_core::types::QuantizedKVCache;
use katgpt_kv::kvarn::KVarNKVCache;
use katgpt_kv::kvarn::kv_cache::KVarNConfig;

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

// ── fixtures ──────────────────────────────────────────────────────────────

const D: usize = 128; // kv_dim
const TILE: usize = 128;
const T: usize = 4096; // held-out positions = 32 full tiles
const U: usize = 64; // vocab (planted tokens)
const SINK: usize = U; // extra token id for G1d
const N_CAL: usize = 100_000;
const SPIKE: f32 = 20.0;
const SPIKE_CH: [usize; 4] = [3, 41, 77, 110];

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

/// Zipf(1) over `U` tokens.
fn zipf_cdf() -> Vec<f64> {
    let w: Vec<f64> = (1..=U).map(|r| 1.0 / r as f64).collect();
    let z: f64 = w.iter().sum();
    let mut acc = 0.0;
    w.iter()
        .map(|x| {
            acc += x / z;
            acc
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    /// Gaussian planted means only (G1a).
    Plain,
    /// Token 0 (the most frequent) is bimodal: 75% spike, 25% none (G1c).
    OffMean,
    /// Position 0 of every tile is the SINK token with a consistent spike (G1d).
    Sink,
}

struct World {
    cdf: Vec<f64>,
    mu: Vec<f32>, // (U+1) × D
    arm: Arm,
}

impl World {
    fn new(sigma_mu: f32, arm: Arm, seed: u64) -> Self {
        let cdf = zipf_cdf();
        let mut prev = 0.0;
        let probs: Vec<f64> = cdf
            .iter()
            .map(|&c| {
                let p = c - prev;
                prev = c;
                p
            })
            .collect();
        let mut rng = Rng(seed);
        let mut mu: Vec<f32> = (0..(U + 1) * D).map(|_| sigma_mu * rng.gauss()).collect();
        for j in 0..D {
            let m: f64 = (0..U).map(|s| probs[s] * f64::from(mu[s * D + j])).sum();
            for s in 0..U {
                mu[s * D + j] -= m as f32;
            }
        }
        Self { cdf, mu, arm }
    }

    /// Draw the token at position `pos` + its value row. Returns (token,
    /// spike_flag) — spike_flag marks the rows the G1c/G1d reports split on.
    fn draw(&self, rng: &mut Rng, pos: usize, v: &mut [f32]) -> (usize, bool) {
        let s = if self.arm == Arm::Sink && pos.is_multiple_of(TILE) {
            SINK
        } else {
            let x = rng.uniform();
            self.cdf.partition_point(|&c| c < x).min(U - 1)
        };
        for (x, &m) in v.iter_mut().zip(&self.mu[s * D..(s + 1) * D]) {
            *x = m + rng.gauss();
        }
        let spike = match self.arm {
            Arm::Plain => false,
            Arm::OffMean => s == 0 && rng.uniform() < 0.75,
            Arm::Sink => s == SINK,
        };
        if spike {
            for &c in &SPIKE_CH {
                v[c] += SPIKE;
            }
        }
        (s, spike)
    }
}

/// Fit the E^V table on a calibration split; return (table, dashboard ρ_V).
fn fit(world: &World, seed: u64) -> (FittedTokenTable, f64) {
    let mut rng = Rng(seed);
    let mut toks = Vec::with_capacity(N_CAL);
    let mut vs = vec![0.0f32; N_CAL * D];
    let mut counts = vec![0u64; U + 1];
    for t in 0..N_CAL {
        let (s, _) = world.draw(&mut rng, t, &mut vs[t * D..(t + 1) * D]);
        counts[s] += 1;
        toks.push(s);
    }
    let zeros = [0.0f32; D];
    let mut cal = LayeredVkCalibration::from_counts(1, D, counts, U + 1);
    for t in 0..N_CAL {
        cal.observe_layer(0, toks[t], &zeros, &vs[t * D..(t + 1) * D]);
    }
    let rho = f64::from(cal.layers[0].v.r_squared().aggregate);
    (
        FittedTokenTable::from_calibration(&cal, VkSignal::ValueMean, 0.0),
        rho,
    )
}

fn kvarn(bits: u8) -> KVarNKVCache {
    KVarNKVCache::with_config(&KVarNConfig {
        n_layers: 1,
        kv_dim: D,
        max_seq_len: T,
        bits,
        tile_size: TILE,
        ..KVarNConfig::default()
    })
}

struct HeldOut {
    toks: Vec<u32>,
    vals: Vec<f32>,
    spike: Vec<bool>,
}

fn held_out(world: &World, seed: u64) -> HeldOut {
    let mut rng = Rng(seed);
    let mut vals = vec![0.0f32; T * D];
    let mut toks = Vec::with_capacity(T);
    let mut spike = Vec::with_capacity(T);
    for p in 0..T {
        let (s, f) = world.draw(&mut rng, p, &mut vals[p * D..(p + 1) * D]);
        toks.push(s as u32);
        spike.push(f);
    }
    HeldOut { toks, vals, spike }
}

/// Per-position squared error of (plain, mean-removed) through KVarN.
fn sq_errors(bits: u8, table: &FittedTokenTable, ho: &HeldOut) -> (Vec<f64>, Vec<f64>) {
    let mut plain = kvarn(bits);
    let mut dec = MeanRemovedValueCache::new(kvarn(bits), table, T);
    for p in 0..T {
        let v = &ho.vals[p * D..(p + 1) * D];
        plain.store_value(0, p, v);
        dec.set_token(p, ho.toks[p]);
        dec.store_value(0, p, v);
    }
    let mut out = vec![0.0f32; D];
    let (mut ep, mut ed) = (Vec::with_capacity(T), Vec::with_capacity(T));
    for p in 0..T {
        let v = &ho.vals[p * D..(p + 1) * D];
        plain.dequantize_value_into(0, p, &mut out);
        ep.push(
            out.iter()
                .zip(v)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum(),
        );
        dec.dequantize_value_into(0, p, &mut out);
        ed.push(
            out.iter()
                .zip(v)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum(),
        );
    }
    (ep, ed)
}

fn ratio_where(ep: &[f64], ed: &[f64], keep: impl Fn(usize) -> bool) -> (f64, usize) {
    let (mut a, mut b, mut n) = (0.0, 0.0, 0usize);
    for i in 0..ep.len() {
        if keep(i) {
            a += ep[i];
            b += ed[i];
            n += 1;
        }
    }
    (b / a, n)
}

fn main() {
    println!("Bench 895 — Issue 883 P1 mean-removed V quant over KVarN (katgpt-kv)\n");
    g1a_prediction();
    g1c_off_mean();
    g1d_sink();
    g3_bit_identity();
    g3b_fold_bit_identity();
    g4_allocs();
    g2_kernel();
    let fails = FAILS.load(Ordering::Relaxed);
    if fails > 0 {
        println!("\n{fails} gate(s) FAILED");
        std::process::exit(1);
    }
    println!("\nALL GATES PASS");
}

fn g1a_prediction() {
    println!("G1a — dashboard-predicted (1 − ρ_V) vs measured KVarN quant-MSE ratio");
    for &rho_t in &[0.25f32, 0.5, 0.75] {
        let sigma_mu = (rho_t / (1.0 - rho_t)).sqrt();
        let world = World::new(sigma_mu, Arm::Plain, 0x0883_001a ^ rho_t.to_bits() as u64);
        let (table, rho) = fit(&world, 0xCA1 ^ rho_t.to_bits() as u64);
        let ho = held_out(&world, 0x40 ^ rho_t.to_bits() as u64);
        for &bits in &[2u8, 4, 8] {
            let (ep, ed) = sq_errors(bits, &table, &ho);
            let (measured, _) = ratio_where(&ep, &ed, |_| true);
            let predicted = 1.0 - rho;
            let rel = measured / predicted - 1.0;
            let mse_plain = ep.iter().sum::<f64>() / (T * D) as f64;
            gate(
                &format!("G1a ρ≈{rho_t} bits={bits}"),
                rel.abs() <= 0.10,
                format!(
                    "ρ_V={rho:.4} ⇒ predicted {predicted:.4}; measured {measured:.4} ({:+.1}% rel, \
                     tol ±10%); plain MSE/elem {mse_plain:.3e}",
                    100.0 * rel
                ),
            );
        }
    }
}

fn g1c_off_mean() {
    println!("\nG1c — absmax caveat: bimodal token (75% +20σ spike, 25% none) — REPORT");
    let world = World::new(1.0, Arm::OffMean, 0x0883_001c);
    let (table, rho) = fit(&world, 0xCA1C);
    let ho = held_out(&world, 0x41C);
    for &bits in &[2u8, 4, 8] {
        let (ep, ed) = sq_errors(bits, &table, &ho);
        let (all, _) = ratio_where(&ep, &ed, |_| true);
        let (spk, n1) = ratio_where(&ep, &ed, |i| ho.toks[i] == 0 && ho.spike[i]);
        let (nos, n2) = ratio_where(&ep, &ed, |i| ho.toks[i] == 0 && !ho.spike[i]);
        let (rest, _) = ratio_where(&ep, &ed, |i| ho.toks[i] != 0);
        println!(
            "    bits={bits}: overall {all:.4} vs predicted {:.4} ({:+.1}% rel) | token-0 spike rows \
             (n={n1}) {spk:.3} · token-0 NO-spike rows (n={n2}) {nos:.3} · other tokens {rest:.4}",
            1.0 - rho,
            100.0 * (all / (1.0 - rho) - 1.0)
        );
    }
}

fn g1d_sink() {
    println!("\nG1d — trap 3: sink token at every tile start (consistent +20σ)");
    let world = World::new(1.0, Arm::Sink, 0x0883_001d);
    let (table, rho) = fit(&world, 0xCA1D);
    let ho = held_out(&world, 0x41D);
    for &bits in &[2u8, 4, 8] {
        let (ep, ed) = sq_errors(bits, &table, &ho);
        let (sink, _) = ratio_where(&ep, &ed, |i| ho.toks[i] as usize == SINK);
        let (rest, _) = ratio_where(&ep, &ed, |i| ho.toks[i] as usize != SINK);
        let (all, _) = ratio_where(&ep, &ed, |_| true);
        gate(
            &format!("G1d sink bits={bits}"),
            sink <= 1.0 && rest <= 1.0,
            format!(
                "sink rows {sink:.4}, non-sink rows of sink tiles {rest:.4} (both ≤ 1.0); overall \
                 {all:.4} vs predicted {:.4}",
                1.0 - rho
            ),
        );
    }
}

fn g3_bit_identity() {
    println!("\nG3 — miss table / E=0 table bitwise vs the plain backend");
    let world = World::new(1.0, Arm::Plain, 0x0883_0003);
    let (table, _) = fit(&world, 0xCA13);
    let ho = held_out(&world, 0x413);
    let zeros = table.zeros_like();
    let miss = FittedTokenTable::from_rows(1, D, vec![u32::MAX; U + 1], Vec::new());
    for &bits in &[2u8, 4] {
        for (label, t) in [("miss", &miss), ("E=0", &zeros)] {
            let mut plain = kvarn(bits);
            let mut dec = MeanRemovedValueCache::new(kvarn(bits), t, T);
            for p in 0..T {
                let v = &ho.vals[p * D..(p + 1) * D];
                plain.store_value(0, p, v);
                dec.set_token(p, ho.toks[p]);
                dec.store_value(0, p, v);
            }
            let (mut a, mut b) = (vec![0.0f32; D], vec![0.0f32; D]);
            let mut diff = 0usize;
            for p in 0..T {
                plain.dequantize_value_into(0, p, &mut a);
                dec.dequantize_value_into(0, p, &mut b);
                diff += a
                    .iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
            }
            gate(
                &format!("G3 {label} bits={bits}"),
                diff == 0,
                format!("{diff} differing elements of {}", T * D),
            );
        }
    }
}

/// G3b — a REAL (non-zero) table: the decorator's read and fused
/// accumulate must equal the backend's plain dequant followed by the
/// unfused `+ E[s]` (and `w · (x + m)`) BITWISE. Covers every bit-width
/// arm plus the Hadamard configuration. It is the gate any future fold of
/// the add-back into a backend's dequant epilogue must pass (the
/// bit-identical KVarN fold measured at `60f1e7baa` did; its FMA-contracted
/// variant failed it on ~29% of elements — Bench 895 addendum).
fn g3b_fold_bit_identity() {
    println!("\nG3b — non-zero table: decorated read / accumulate vs unfused reference, bitwise");
    let world = World::new(1.0, Arm::Plain, 0x0883_0033);
    let (table, _) = fit(&world, 0xCA33);
    let ho = held_out(&world, 0x433);
    for &(bits, hadamard) in &[(2u8, false), (4, false), (8, false), (4, true)] {
        let mk = || {
            KVarNKVCache::with_config(&KVarNConfig {
                n_layers: 1,
                kv_dim: D,
                max_seq_len: T,
                bits,
                tile_size: TILE,
                hadamard,
                ..KVarNConfig::default()
            })
        };
        let mut dec = MeanRemovedValueCache::new(mk(), &table, T);
        for p in 0..T {
            dec.set_token(p, ho.toks[p]);
            dec.store_value(0, p, &ho.vals[p * D..(p + 1) * D]);
        }
        let (mut got, mut want) = (vec![0.0f32; D], vec![0.0f32; D]);
        let (mut acc_got, mut acc_want) = (vec![0.0f32; D], vec![0.0f32; D]);
        let (mut diff_read, mut diff_acc, mut hit) = (0usize, 0usize, 0usize);
        for p in 0..T {
            let row = table.row(0, ho.toks[p]);
            hit += usize::from(row.is_some());
            dec.dequantize_value_into(0, p, &mut got);
            dec.inner_mut().dequantize_value_into(0, p, &mut want);
            let w = 1.0 / (1.0 + (p % 97) as f32);
            for (k, (a, &x)) in acc_want.iter_mut().zip(&want).enumerate() {
                *a += w * row.map_or(x, |e| x + e[k]);
            }
            if let Some(e) = row {
                for (x, &m) in want.iter_mut().zip(e) {
                    *x += m;
                }
            }
            dec.accumulate_value(0, p, w, &mut acc_got);
            diff_read += got
                .iter()
                .zip(&want)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
        }
        diff_acc += acc_got
            .iter()
            .zip(&acc_want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        gate(
            &format!("G3b bits={bits} hadamard={hadamard}"),
            diff_read == 0 && diff_acc == 0 && hit > T / 2,
            format!(
                "read {diff_read} of {} differ, accumulate {diff_acc} of {D} differ ({hit} of {T} \
                 positions hit a row)",
                T * D
            ),
        );
    }
}

fn g4_allocs() {
    println!("\nG4 — decorator allocations over the backend's own loop");
    let world = World::new(1.0, Arm::Plain, 0x0883_0004);
    let (table, _) = fit(&world, 0xCA14);
    let ho = held_out(&world, 0x414);
    let mut plain = kvarn(4);
    let mut dec = MeanRemovedValueCache::new(kvarn(4), &table, T);
    let mut out = vec![0.0f32; D];
    let mut acc = vec![0.0f32; D];
    let run_plain = |c: &mut KVarNKVCache, out: &mut [f32]| {
        c.reset();
        for p in 0..T {
            c.store_value(0, p, &ho.vals[p * D..(p + 1) * D]);
        }
        for p in 0..T {
            c.dequantize_value_into(0, p, out);
        }
    };
    run_plain(&mut plain, &mut out); // warm
    let b0 = allocs();
    run_plain(&mut plain, &mut out);
    let n_plain = allocs() - b0;
    let run_dec =
        |c: &mut MeanRemovedValueCache<'_, KVarNKVCache>, out: &mut [f32], acc: &mut [f32]| {
            c.reset();
            for p in 0..T {
                c.set_token(p, ho.toks[p]);
                c.store_value(0, p, &ho.vals[p * D..(p + 1) * D]);
            }
            for p in 0..T {
                c.dequantize_value_into(0, p, out);
                c.accumulate_value(0, p, 0.5, acc);
            }
        };
    run_dec(&mut dec, &mut out, &mut acc);
    let b1 = allocs();
    run_dec(&mut dec, &mut out, &mut acc);
    let n_dec = allocs() - b1;
    black_box((&out, &acc));
    gate(
        "G4 decorator",
        n_dec == n_plain,
        format!(
            "backend loop {n_plain} allocs, decorated loop {n_dec} allocs (Δ {}) over {T} stores + \
             {} reads",
            n_dec as i64 - n_plain as i64,
            2 * T
        ),
    );
}

fn g2_kernel() {
    println!("\nG2 — decode V-aggregation kernel, T={T}, kv_dim={D}, 4-bit");
    let world = World::new(1.0, Arm::Plain, 0x0883_0002);
    let (table, _) = fit(&world, 0xCA12);
    let ho = held_out(&world, 0x412);
    let mut plain = kvarn(4);
    let mut dec = MeanRemovedValueCache::new(kvarn(4), &table, T);
    for p in 0..T {
        let v = &ho.vals[p * D..(p + 1) * D];
        plain.store_value(0, p, v);
        dec.set_token(p, ho.toks[p]);
        dec.store_value(0, p, v);
    }
    let w: Vec<f32> = (0..T).map(|p| 1.0 / (1.0 + (p % 97) as f32)).collect();
    let (mut buf_a, mut buf_b) = (vec![0.0f32; D], vec![0.0f32; D]);
    let (mut acc_a, mut acc_b) = (vec![0.0f32; D], vec![0.0f32; D]);
    let (mut ka, mut kb) = (0.0f32, 0.0f32);
    let r = ab_median_ratio(
        21,
        10,
        3,
        |i| {
            acc_a.fill(0.0);
            let s = 1.0 + (i % 3) as f32 * 1e-3;
            for (p, &wv) in w.iter().enumerate() {
                plain.dequantize_value_into(0, p, &mut buf_a);
                let wp = black_box(wv * s);
                for (a, &x) in acc_a.iter_mut().zip(&buf_a) {
                    *a += wp * x;
                }
            }
            ka += black_box(acc_a[i % D]);
        },
        |i| {
            acc_b.fill(0.0);
            let s = 1.0 + (i % 3) as f32 * 1e-3;
            for (p, &wv) in w.iter().enumerate() {
                dec.accumulate_value(0, p, black_box(wv * s), &mut acc_b);
            }
            kb += black_box(acc_b[i % D]);
        },
    );
    r.report("G2 fused accumulate_value / plain dequant+axpy");
    gate(
        "G2 fused dequant+restore ≤ +1%",
        r.median <= 1.01,
        format!(
            "median {:.4} ({:+.2}%, range {:.4}–{:.4}; plain {:.1} µs per {T}-position pass)",
            r.median,
            r.overhead_pct(),
            r.min(),
            r.max(),
            r.a_ns_per_iter() / 1e3
        ),
    );
    let r2 = ab_median_ratio(
        21,
        10,
        3,
        |i| {
            acc_a.fill(0.0);
            let s = 1.0 + (i % 3) as f32 * 1e-3;
            for (p, &wv) in w.iter().enumerate() {
                plain.dequantize_value_into(0, p, &mut buf_a);
                let wp = black_box(wv * s);
                for (a, &x) in acc_a.iter_mut().zip(&buf_a) {
                    *a += wp * x;
                }
            }
            ka += black_box(acc_a[i % D]);
        },
        |i| {
            acc_b.fill(0.0);
            let s = 1.0 + (i % 3) as f32 * 1e-3;
            for (p, &wv) in w.iter().enumerate() {
                dec.dequantize_value_into(0, p, &mut buf_b);
                let wp = black_box(wv * s);
                for (a, &x) in acc_b.iter_mut().zip(&buf_b) {
                    *a += wp * x;
                }
            }
            kb += black_box(acc_b[i % D]);
        },
    );
    r2.report("G2 report two-pass decorator read / plain");

    // ── REPORT arms (Bench 895 addendum): the per-position floor and the
    // next lever. None of these gate; they locate where the +x% lives.
    //
    // A/A: a second plain cache over the same data — the protocol's own
    // floor on this box (the harness always runs a before b in a round).
    let mut plain2 = kvarn(4);
    // Lookup-only: a decorator whose table misses every token — it pays the
    // per-position token → row resolution and nothing else.
    let miss = FittedTokenTable::from_rows(1, D, vec![u32::MAX; U + 1], Vec::new());
    let mut dmiss = MeanRemovedValueCache::new(kvarn(4), &miss, T);
    for p in 0..T {
        let v = &ho.vals[p * D..(p + 1) * D];
        plain2.store_value(0, p, v);
        dmiss.set_token(p, ho.toks[p]);
        dmiss.store_value(0, p, v);
    }
    let r3 = ab_median_ratio(
        21,
        10,
        3,
        |i| ka += plain_pass(&mut plain, &w, i, &mut buf_a, &mut acc_a),
        |i| kb += plain_pass(&mut plain2, &w, i, &mut buf_b, &mut acc_b),
    );
    r3.report("G2 report A/A control (second plain cache) / plain");
    let r4 = ab_median_ratio(
        21,
        10,
        3,
        |i| ka += plain_pass(&mut plain, &w, i, &mut buf_a, &mut acc_a),
        |i| {
            acc_b.fill(0.0);
            let s = 1.0 + (i % 3) as f32 * 1e-3;
            for (p, &wv) in w.iter().enumerate() {
                dmiss.accumulate_value(0, p, black_box(wv * s), &mut acc_b);
            }
            kb += black_box(acc_b[i % D]);
        },
    );
    r4.report("G2 report lookup-only (miss-table accumulate_value) / plain");
    // Deferred restore — the NEXT lever, by linearity of the aggregation:
    //   Σ_p w_p·(v̂_p + E[s_p]) = Σ_p w_p·v̂_p + Σ_s (Σ_{p: s_p = s} w_p)·E[s]
    // so the per-position work is one scalar bucket add, and the table rows
    // are touched once per DISTINCT token at the end. Not bit-identical to
    // the per-position form (the sum is reassociated); test-local only.
    let toks = &ho.toks;
    let mut wtok = vec![0.0f32; U + 1];
    let r5 = ab_median_ratio(
        21,
        10,
        3,
        |i| ka += plain_pass(&mut plain, &w, i, &mut buf_a, &mut acc_a),
        |i| {
            acc_b.fill(0.0);
            wtok.fill(0.0);
            let s = 1.0 + (i % 3) as f32 * 1e-3;
            for (p, &wv) in w.iter().enumerate() {
                dec.inner_mut().dequantize_value_into(0, p, &mut buf_b);
                let wp = black_box(wv * s);
                for (a, &x) in acc_b.iter_mut().zip(&buf_b) {
                    *a += wp * x;
                }
                wtok[toks[p] as usize] += wp;
            }
            for (t, &wt) in wtok.iter().enumerate() {
                if wt == 0.0 {
                    continue;
                }
                if let Some(row) = table.row(0, t as u32) {
                    for (a, &m) in acc_b.iter_mut().zip(row) {
                        *a += wt * m;
                    }
                }
            }
            kb += black_box(acc_b[i % D]);
        },
    );
    r5.report("G2 report deferred restore (per-token weight bucket) / plain");
    black_box(ka + kb);
}

/// The plain G2 arm: KVarN dequant + axpy over every position.
fn plain_pass(c: &mut KVarNKVCache, w: &[f32], i: usize, buf: &mut [f32], acc: &mut [f32]) -> f32 {
    acc.fill(0.0);
    let s = 1.0 + (i % 3) as f32 * 1e-3;
    for (p, &wv) in w.iter().enumerate() {
        c.dequantize_value_into(0, p, buf);
        let wp = black_box(wv * s);
        for (a, &x) in acc.iter_mut().zip(buf.iter()) {
            *a += wp * x;
        }
    }
    black_box(acc[i % D])
}
