//! Issue 883 P2 + P3 + P4 — fitted K=V+ retrofit, V-cache-halving
//! reconstruction, and the serving laws (Bench 895, primitive half).
//!
//! P2 (fitted K=V+, `v_from_k_plus`):
//!   G1 refund law — synthetic `V = K + E*[s] + noise`, table fitted by the
//!      P0 substrate on a calibration split, measured on a held-out split:
//!      `MSE(λ)/MSE(0)` vs the dashboard prediction `1 − (2λ − λ²)·ρ(V−K)`,
//!      tolerance ±3% relative at λ ∈ {0.5, 1}. (Primitive law only — the
//!      model-level PPL/NIAH ladder is riir-infer Issue 013.)
//!   G2 `v_from_k_plus` vs the `W_V` GEMV it deletes (gemma-2-2b shape,
//!      d=2304 → d_v=1024), paired interleave; bar ≤ 0.05×.
//!   G3 λ=0 bit-identical to the `V := K` copy (incl. `−0.0`, a huge row).
//!   G4 0 allocs.
//! P3 (reconstruction, `reconstruct_v_from_rope_k` / `read_v`):
//!   G1 exactness: `G(−θp)·(G(θp)·K) + λE` vs `K + λE` over positions to
//!      131072, gemma head width 256; stated bound: per element
//!      `|err| ≤ 8ε·(‖k_pair‖₂ + |result|)`, ε = f32::EPSILON. Trap 2 pinned
//!      as a measured negative: 64 forward/inverse round-trips (reported).
//!   G2 one decode-attention head (online softmax, fused score + V
//!      accumulate) at T=32768, hd=256: reconstruct-from-K (RopeAction, the
//!      shipped action) vs the full-cache control; bar ≤ 1.00×. A
//!      table-driven PositionGroupAction adapter (test-local) is reported
//!      beside it.
//!   G3 `VReadPath::FullCache` bit-identical to the stored V.
//!   G4 0 allocs.
//! P4: FLOP law `ΔF_V = L·S·d_v·(2d−1)` — line 1 counted vs analytic, line
//!   2 the measured GEMV time → implied GFLOP/s + GB/s (is the saving
//!   FLOP- or byte-shaped?). Storage dial + cache fraction printed.
//!
//! Run:
//!   cargo test -p katgpt-core --release --features fitted_v_reconstruct \
//!     --test bench_895_fitted_value_table_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_core::fitted_anchor_table::LayeredVkCalibration;
use katgpt_core::fitted_value_table::{
    FittedTokenTable, VReadPath, VkSignal, read_v, reconstruct_v_from_rope_k, storage_bytes,
    v_cache_fraction, v_from_k_plus, v_projection_flops,
};
use katgpt_core::position_group_action::{PositionGroupAction, RopeAction};

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

// ── fixtures ──────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // splitmix64
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

/// Zipf(1) sampler over `u` tokens via the CDF.
struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    fn new(u: usize) -> Self {
        let w: Vec<f64> = (1..=u).map(|r| 1.0 / r as f64).collect();
        let z: f64 = w.iter().sum();
        let mut acc = 0.0;
        let cdf = w
            .iter()
            .map(|x| {
                acc += x / z;
                acc
            })
            .collect();
        Self { cdf }
    }
    fn probs(&self) -> Vec<f64> {
        let mut prev = 0.0;
        self.cdf
            .iter()
            .map(|&c| {
                let p = c - prev;
                prev = c;
                p
            })
            .collect()
    }
    fn sample(&self, rng: &mut Rng) -> usize {
        let x = rng.uniform();
        self.cdf.partition_point(|&c| c < x).min(self.cdf.len() - 1)
    }
}

/// Planted per-token table `E*[s] ~ N(0, σ²)`, then CENTERED under the
/// sampling distribution so the grand mean is ≈ 0 (the dashboard ρ is a
/// centered statistic; an uncentered grand mean would add a term the
/// prediction does not carry — stated, not hidden).
fn planted(u: usize, d: usize, sigma: f32, probs: &[f64], rng: &mut Rng) -> Vec<f32> {
    let mut e: Vec<f32> = (0..u * d).map(|_| sigma * rng.gauss()).collect();
    for j in 0..d {
        let m: f64 = (0..u).map(|s| probs[s] * f64::from(e[s * d + j])).sum();
        for s in 0..u {
            e[s * d + j] -= m as f32;
        }
    }
    e
}

static FAILS: AtomicUsize = AtomicUsize::new(0);

fn gate(name: &str, ok: bool, detail: String) {
    println!("  [{}] {name}: {detail}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        FAILS.fetch_add(1, Ordering::Relaxed);
    }
}

const EPS: f32 = f32::EPSILON;

fn main() {
    println!("Bench 895 — Issue 883 P2/P3/P4 primitive GOAT (katgpt-core)\n");
    p2_g1_refund_law();
    p2_g3_bit_identity();
    p3_g1_exactness();
    p3_g3_kill_switch();
    g4_allocs();
    p2_g2_and_p4_flop_law();
    p3_g2_decode_head();

    let fails = FAILS.load(Ordering::Relaxed);
    if fails > 0 {
        println!("\n{fails} gate(s) FAILED");
        std::process::exit(1);
    }
    println!("\nALL GATES PASS");
}

// ── P2 G1 — the refund law ────────────────────────────────────────────────

fn p2_g1_refund_law() {
    println!("P2 G1 — refund law MSE(λ)/MSE(0) vs 1 − (2λ − λ²)·ρ(V−K)");
    let (u, d) = (64usize, 64usize);
    let zipf = Zipf::new(u);
    let probs = zipf.probs();
    for &(sigma_e, label) in &[(0.5f32, "ρ≈0.20"), (1.0, "ρ≈0.50"), (2.0, "ρ≈0.80")] {
        let mut rng = Rng(0x883_0002 ^ sigma_e.to_bits() as u64);
        let e_star = planted(u, d, sigma_e, &probs, &mut rng);
        let gen_tok = |rng: &mut Rng, k: &mut [f32], v: &mut [f32]| -> usize {
            let s = zipf.sample(rng);
            for j in 0..d {
                k[j] = rng.gauss();
                v[j] = k[j] + e_star[s * d + j] + rng.gauss();
            }
            s
        };
        // calibration split — the P0 substrate
        let n_cal = 200_000usize;
        let mut counts = vec![0u64; u];
        let mut toks = Vec::with_capacity(n_cal);
        let mut ks = vec![0.0f32; n_cal * d];
        let mut vs = vec![0.0f32; n_cal * d];
        for t in 0..n_cal {
            let (kr, vr) = (&mut ks[t * d..(t + 1) * d], &mut vs[t * d..(t + 1) * d]);
            let s = gen_tok(&mut rng, kr, vr);
            counts[s] += 1;
            toks.push(s);
        }
        let mut cal = LayeredVkCalibration::from_counts(1, d, counts, u);
        for t in 0..n_cal {
            cal.observe_layer(0, toks[t], &ks[t * d..(t + 1) * d], &vs[t * d..(t + 1) * d]);
        }
        let rho = cal.layers[0].vk.r_squared().aggregate as f64;
        let table = FittedTokenTable::from_calibration(&cal, VkSignal::Residual, 0.0);
        // held-out split
        let n_ho = 50_000usize;
        let lambdas = [0.0f32, 0.5, 1.0];
        let mut mse = [0.0f64; 3];
        let (mut k, mut v, mut out) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d]);
        for _ in 0..n_ho {
            let s = gen_tok(&mut rng, &mut k, &mut v);
            let row = table.row(0, s as u32);
            for (i, &lam) in lambdas.iter().enumerate() {
                v_from_k_plus(&k, row, lam, &mut out);
                mse[i] += out
                    .iter()
                    .zip(&v)
                    .map(|(a, b)| f64::from(a - b).powi(2))
                    .sum::<f64>();
            }
        }
        for (i, &lam) in lambdas.iter().enumerate().skip(1) {
            let lam = f64::from(lam);
            let measured = mse[i] / mse[0];
            let predicted = 1.0 - (2.0 * lam - lam * lam) * rho;
            let rel = measured / predicted - 1.0;
            gate(
                &format!("P2 G1 {label} λ={lam}"),
                rel.abs() <= 0.03,
                format!(
                    "dashboard ρ(V−K)={rho:.4}; measured MSE ratio {measured:.4} vs predicted \
                     {predicted:.4} ({:+.2}% rel, tol ±3%)",
                    100.0 * rel
                ),
            );
        }
        // James–Stein shrinkage ladder (report): rare-token rows shrink.
        let mut line = String::new();
        for &ljs in &[10.0f32, 100.0, 1000.0] {
            let ts = FittedTokenTable::from_calibration(&cal, VkSignal::Residual, ljs);
            let mut r2 = Rng(0xAB ^ ljs.to_bits() as u64);
            let mut acc = 0.0f64;
            let mut acc0 = 0.0f64;
            for _ in 0..20_000 {
                let s = gen_tok(&mut r2, &mut k, &mut v);
                v_from_k_plus(&k, ts.row(0, s as u32), 1.0, &mut out);
                acc += out
                    .iter()
                    .zip(&v)
                    .map(|(a, b)| f64::from(a - b).powi(2))
                    .sum::<f64>();
                v_from_k_plus(&k, table.row(0, s as u32), 1.0, &mut out);
                acc0 += out
                    .iter()
                    .zip(&v)
                    .map(|(a, b)| f64::from(a - b).powi(2))
                    .sum::<f64>();
            }
            line.push_str(&format!(" λ_js={ljs}: {:.4}", acc / acc0));
        }
        println!("    report {label} JS-shrunk/plain MSE at λ=1 (n_s ≫ λ_js ⇒ ≈1):{line}");
    }
}

// ── P2 G3 ─────────────────────────────────────────────────────────────────

fn p2_g3_bit_identity() {
    println!("\nP2 G3 — λ=0 ≡ V:=K (bitwise)");
    let d = 1024;
    let mut rng = Rng(0x883_0003);
    let mut k: Vec<f32> = (0..d).map(|_| rng.gauss() * 3.0).collect();
    k[0] = -0.0;
    k[1] = f32::MIN_POSITIVE;
    let mut data: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    data[0] = f32::MAX; // 0·MAX = 0, but a λ=0 multiply must still never run
    let table = FittedTokenTable::from_rows(1, d, vec![0], data);
    let mut out = vec![1.0f32; d];
    v_from_k_plus(&k, table.row(0, 0), 0.0, &mut out);
    let same = out.iter().zip(&k).all(|(a, b)| a.to_bits() == b.to_bits());
    let mut out_miss = vec![1.0f32; d];
    v_from_k_plus(&k, table.row(0, 7), 0.9, &mut out_miss);
    let same_miss = out_miss
        .iter()
        .zip(&k)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    gate(
        "P2 G3 λ=0 / missing row",
        same && same_miss,
        format!("λ=0 bitwise {same}, miss bitwise {same_miss} (incl. −0.0 and a MAX row entry)"),
    );
}

// ── P3 G1 — exactness + trap 2 ────────────────────────────────────────────

fn p3_g1_exactness() {
    println!("\nP3 G1 — reconstruct(G(θp)K) vs K + λE, bound 8ε·(‖k_pair‖+|result|)");
    let (hd, n_heads) = (256usize, 4usize);
    let d = hd * n_heads;
    for &theta in &[10_000.0f32, 1_000_000.0] {
        let rope = RopeAction::with_theta(hd, theta);
        let mut rng = Rng(0x883_0031 ^ theta.to_bits() as u64);
        let mut worst = 0.0f64;
        let mut worst_abs = 0.0f32;
        let (mut kc, mut out, mut expect) = (vec![0.0f32; d], vec![0.0f32; d], vec![0.0f32; d]);
        let row: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
        let positions: Vec<u32> = (0..512)
            .map(|i| match i {
                0 => 0,
                1 => 1,
                2 => 131_071,
                _ => (rng.next_u64() % 131_072) as u32,
            })
            .collect();
        for &p in &positions {
            // pre-RoPE K with occasional massive-activation channels
            let k_pre: Vec<f32> = (0..d)
                .map(|j| rng.gauss() * if j % 97 == 0 { 60.0 } else { 1.0 })
                .collect();
            for (x, o) in k_pre.chunks_exact(hd).zip(kc.chunks_exact_mut(hd)) {
                rope.apply_at(p as f32, x, o); // the cache write (model's own)
            }
            for &lam in &[0.0f32, 1.0, 0.37] {
                reconstruct_v_from_rope_k(&rope, p as f32, &kc, Some(&row), lam, &mut out);
                v_from_k_plus(&k_pre, Some(&row), lam, &mut expect); // P2's served V
                for i in 0..d {
                    let pair = i & !1;
                    let pn = (k_pre[pair].powi(2) + k_pre[pair + 1].powi(2)).sqrt();
                    let err = (out[i] - expect[i]).abs();
                    let scale = f64::from(pn + expect[i].abs()).max(f64::MIN_POSITIVE);
                    worst = worst.max(f64::from(err) / scale / f64::from(EPS));
                    worst_abs = worst_abs.max(err);
                }
            }
        }
        gate(
            &format!("P3 G1 θ={theta}"),
            worst <= 8.0,
            format!(
                "max |err|/(‖k_pair‖+|result|) = {worst:.2}ε (bound 8ε), max |err| = {worst_abs:.3e}, \
                 positions 0..131071 × λ∈{{0,0.37,1}}"
            ),
        );
    }
    // Trap 2 pinned: the round-trip the primitive never does.
    let rope = RopeAction::new(hd);
    let mut rng = Rng(0x2);
    let k0: Vec<f32> = (0..hd).map(|_| rng.gauss()).collect();
    let (mut a, mut b) = (k0.clone(), vec![0.0f32; hd]);
    let mut line = String::new();
    for trips in 1..=64 {
        rope.apply_at(100_003.0, &a, &mut b);
        rope.apply_inverse_at(100_003.0, &b, &mut a);
        if matches!(trips, 1 | 8 | 64) {
            let e = a
                .iter()
                .zip(&k0)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            line.push_str(&format!(" {trips}×: {e:.2e}"));
        }
    }
    println!("    report trap 2 — max |err| after N forward/inverse round-trips:{line}");
}

// ── P3 G3 ─────────────────────────────────────────────────────────────────

fn p3_g3_kill_switch() {
    println!("\nP3 G3 — FullCache kill switch bitwise");
    let hd = 256;
    let d = 4 * hd;
    let rope = RopeAction::new(hd);
    let mut rng = Rng(0x883_0033);
    let v: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let k: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let row: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let mut out = vec![0.0f32; d];
    read_v(
        VReadPath::FullCache,
        &rope,
        777.0,
        &k,
        Some(&v),
        Some(&row),
        &mut out,
    );
    let same = out.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits());
    gate(
        "P3 G3 FullCache",
        same,
        format!("bitwise copy of stored V: {same}"),
    );
}

// ── G4 ────────────────────────────────────────────────────────────────────

fn g4_allocs() {
    println!("\nG4 — steady-state allocations");
    let hd = 256;
    let d = 4 * hd;
    let rope = RopeAction::new(hd);
    let mut rng = Rng(0x883_0004);
    let k: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let v: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let data: Vec<f32> = (0..8 * d).map(|_| rng.gauss()).collect();
    let table = FittedTokenTable::from_rows(1, d, (0..8).collect(), data);
    let mut out = vec![0.0f32; d];
    let mut sink = 0.0f32;
    let before = allocs();
    for i in 0..10_000usize {
        let row = table.row(0, (i % 11) as u32);
        v_from_k_plus(black_box(&k), row, 0.5, &mut out);
        sink += out[i % d];
        reconstruct_v_from_rope_k(&rope, i as f32, &k, row, 1.0, &mut out);
        sink += out[i % d];
        read_v(
            VReadPath::FullCache,
            &rope,
            i as f32,
            &k,
            Some(&v),
            row,
            &mut out,
        );
        sink += out[i % d];
    }
    let n = allocs() - before;
    black_box(sink);
    gate(
        "G4 P2+P3",
        n == 0,
        format!("{n} allocations over 30k primitive calls"),
    );
}

// ── P2 G2 + P4 FLOP law ───────────────────────────────────────────────────

fn gemv(w: &[f32], x: &[f32], out: &mut [f32], d: usize) {
    for (r, o) in out.iter_mut().enumerate() {
        *o = katgpt_core::simd::simd_dot_f32(&w[r * d..(r + 1) * d], x, d);
    }
}

fn p2_g2_and_p4_flop_law() {
    println!("\nP2 G2 + P4 — v_from_k_plus vs the deleted W_V GEMV (gemma-2-2b d=2304 → d_v=1024)");
    let (d_model, d_v, n_layer) = (2304usize, 1024usize, 26u64);
    let mut rng = Rng(0x883_0022);
    let w: Vec<f32> = (0..d_v * d_model).map(|_| rng.gauss() * 0.02).collect();
    let x: Vec<f32> = (0..d_model).map(|_| rng.gauss()).collect();
    let k: Vec<f32> = (0..d_v).map(|_| rng.gauss()).collect();
    let data: Vec<f32> = (0..64 * d_v).map(|_| rng.gauss()).collect();
    let table = FittedTokenTable::from_rows(1, d_v, (0..64).collect(), data);
    let (mut oa, mut ob) = (vec![0.0f32; d_v], vec![0.0f32; d_v]);
    let (mut ka, mut kb) = (0.0f32, 0.0f32);
    let mut xv = x.clone();
    let r = ab_median_ratio(
        15,
        20,
        5,
        |i| {
            xv[i % d_model] += 1e-6;
            gemv(black_box(&w), black_box(&xv), &mut oa, d_model);
            ka += black_box(oa[i % d_v]);
        },
        |i| {
            let row = table.row(0, (i % 64) as u32);
            v_from_k_plus(black_box(&k), black_box(row), black_box(0.8), &mut ob);
            kb += black_box(ob[i % d_v]);
        },
    );
    r.report("P2 G2 v_from_k_plus / W_V GEMV");
    black_box(ka + kb);
    let gemv_ns = r.a_ns_per_iter();
    let plus_ns = r.b_ns_per_iter();
    gate(
        "P2 G2 v_from_k_plus ≤ 0.05× W_V GEMV",
        r.median <= 0.05,
        format!(
            "median {:.5} (GEMV {:.1} µs, K+λE {:.1} ns per token-layer)",
            r.median,
            gemv_ns / 1e3,
            plus_ns
        ),
    );
    // V:=K copy vs K+λE (report: the cost of the table refund over V:=K)
    {
        let (mut oc, mut od) = (vec![0.0f32; d_v], vec![0.0f32; d_v]);
        let (mut kc, mut kd) = (0.0f32, 0.0f32);
        let r2 = ab_median_ratio(
            15,
            20_000,
            100,
            |i| {
                oc.copy_from_slice(black_box(&k));
                kc += black_box(oc[i % d_v]);
            },
            |i| {
                let row = table.row(0, (i % 64) as u32);
                v_from_k_plus(black_box(&k), black_box(row), black_box(0.8), &mut od);
                kd += black_box(od[i % d_v]);
            },
        );
        r2.report("P2 report K+λE / V:=K copy");
        black_box(kc + kd);
    }
    // P4 line 1: counted FLOPs of a scalar reference GEMV == the law.
    let (dm, dv) = (37usize, 11usize); // small, exact count
    let (mut muls, mut adds) = (0u64, 0u64);
    for _ in 0..dv {
        let mut acc = None::<f32>;
        for _ in 0..dm {
            muls += 1;
            acc = Some(match acc {
                None => 1.0,
                Some(a) => {
                    adds += 1;
                    a + 1.0
                }
            });
        }
    }
    let counted = muls + adds;
    let law = v_projection_flops(1, 1, dv as u64, dm as u64);
    gate(
        "P4 FLOP law line 1 (counted == d_v·(2d−1))",
        counted == law,
        format!("counted {counted} (mul {muls} + add {adds}) vs law {law}"),
    );
    // P4 line 2: measured.
    let flops_tok_layer = v_projection_flops(1, 1, d_v as u64, d_model as u64) as f64;
    let bytes_tok_layer = (d_v * d_model * 4) as f64;
    let saved_ns = gemv_ns - plus_ns;
    println!(
        "    P4 line 2: ΔF_V per token-layer = {:.3} MFLOP; W_V GEMV {:.1} µs ⇒ {:.1} GFLOP/s, \
         {:.1} GB/s of f32 weights; saved {:.1} µs/token-layer ⇒ {:.2} ms/token over L={n_layer} \
         (ΔF_V @S=128 = {:.2} GFLOP)",
        flops_tok_layer / 1e6,
        gemv_ns / 1e3,
        flops_tok_layer / gemv_ns,
        bytes_tok_layer / gemv_ns,
        saved_ns / 1e3,
        saved_ns * n_layer as f64 / 1e6,
        v_projection_flops(n_layer, 128, d_v as u64, d_model as u64) as f64 / 1e9
    );
    println!(
        "    P4 storage dial P(K)=4·L·K·d_v: K=8192 → {:.2} GiB, K=1024 → {:.1} MiB, K=256k(full \
         vocab) → {:.1} GiB; V-cache fraction n_v/(n_kv+n_v) = {:.3} (GQA 8q:4kv, hd 256)",
        storage_bytes(4, 26, 8192, 1024) as f64 / (1u64 << 30) as f64,
        storage_bytes(4, 26, 1024, 1024) as f64 / (1u64 << 20) as f64,
        storage_bytes(4, 26, 256_000, 1024) as f64 / (1u64 << 30) as f64,
        v_cache_fraction(4 * 256, 4 * 256)
    );
}

// ── P3 G2 — one decode-attention head ─────────────────────────────────────

/// Test-local table-driven action (cos/sin per position precomputed) — a
/// REPORT arm showing the generic primitive's cost without transcendentals.
struct TableRope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl TableRope {
    fn new(hd: usize, positions: usize, theta: f32) -> Self {
        let rope = RopeAction::with_theta(hd, theta);
        let half = hd / 2;
        let mut cos = vec![0.0f32; positions * half];
        let mut sin = vec![0.0f32; positions * half];
        for p in 0..positions {
            for (i, &w) in rope.omegas().iter().enumerate() {
                let a = p as f32 * w;
                cos[p * half + i] = a.cos();
                sin[p * half + i] = a.sin();
            }
        }
        Self { cos, sin, half }
    }
}

impl PositionGroupAction for TableRope {
    fn apply_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        let p = n as usize;
        let (c, s) = (&self.cos[p * self.half..], &self.sin[p * self.half..]);
        for i in 0..self.half {
            let (x0, x1) = (x[2 * i], x[2 * i + 1]);
            out[2 * i] = c[i].mul_add(x0, -s[i] * x1);
            out[2 * i + 1] = s[i].mul_add(x0, c[i] * x1);
        }
    }
    fn apply_inverse_at(&self, n: f32, x: &[f32], out: &mut [f32]) {
        let p = n as usize;
        let (c, s) = (&self.cos[p * self.half..], &self.sin[p * self.half..]);
        for i in 0..self.half {
            let (x0, x1) = (x[2 * i], x[2 * i + 1]);
            out[2 * i] = c[i].mul_add(x0, s[i] * x1);
            out[2 * i + 1] = (-s[i]).mul_add(x0, c[i] * x1);
        }
    }
    fn dim(&self) -> usize {
        2 * self.half
    }
}

#[allow(clippy::too_many_arguments)]
fn attn_head<F: FnMut(usize, &[f32], &mut [f32])>(
    q: &[f32],
    kc: &[f32],
    t: usize,
    hd: usize,
    vbuf: &mut [f32],
    acc: &mut [f32],
    mut v_of: F,
) -> f32 {
    let scale = 1.0 / (hd as f32).sqrt();
    let (mut m, mut l) = (f32::NEG_INFINITY, 0.0f32);
    acc.fill(0.0);
    for p in 0..t {
        let k = &kc[p * hd..(p + 1) * hd];
        let s = katgpt_core::simd::simd_dot_f32(q, k, hd) * scale;
        let m_new = m.max(s);
        let c = (m - m_new).exp();
        let w = (s - m_new).exp();
        l = l * c + w;
        v_of(p, k, vbuf);
        for (a, &v) in acc.iter_mut().zip(vbuf.iter()) {
            *a = a.mul_add(c, w * v);
        }
        m = m_new;
    }
    acc[0] / l
}

fn p3_g2_decode_head() {
    println!("\nP3 G2 — decode head T=32768 hd=256: reconstruct-from-K vs full cache");
    let (t, hd) = (32_768usize, 256usize);
    let mut rng = Rng(0x883_0032);
    let kc: Vec<f32> = (0..t * hd).map(|_| rng.gauss()).collect();
    let vc: Vec<f32> = (0..t * hd).map(|_| rng.gauss()).collect();
    let toks: Vec<u32> = (0..t).map(|_| (rng.next_u64() % 512) as u32).collect();
    let data: Vec<f32> = (0..256 * hd).map(|_| rng.gauss() * 0.3).collect();
    let table = FittedTokenTable::from_rows(1, hd, (0..256).collect(), data);
    let q: Vec<f32> = (0..hd).map(|_| rng.gauss()).collect();
    let rope = RopeAction::new(hd);
    let trope = TableRope::new(hd, t, 10_000.0);
    println!(
        "    DRAM bytes/position: full cache K+V = {} B; reconstruct = {} B of K only (the table \
         row is Zipf-hot: 256 rows = {} KiB, L2-resident); TableRope streams +{} B/position of cos/sin",
        2 * hd * 4,
        hd * 4,
        256 * hd * 4 / 1024,
        hd * 4
    );
    let (mut va, mut vb) = (vec![0.0f32; hd], vec![0.0f32; hd]);
    let (mut aa, mut ab) = (vec![0.0f32; hd], vec![0.0f32; hd]);
    let (mut sa, mut sb) = (0.0f32, 0.0f32);
    let (mut qa, mut qb) = (q.clone(), q.clone());
    let r = ab_median_ratio(
        9,
        1,
        1,
        |i| {
            qa[i % hd] += 1e-6;
            sa += attn_head(
                black_box(&qa),
                black_box(&kc),
                t,
                hd,
                &mut va,
                &mut aa,
                |p, _k, out| {
                    read_v(
                        VReadPath::FullCache,
                        &rope,
                        p as f32,
                        _k,
                        Some(&vc[p * hd..(p + 1) * hd]),
                        None,
                        out,
                    );
                },
            );
        },
        |i| {
            qb[i % hd] += 1e-6;
            sb += attn_head(
                black_box(&qb),
                black_box(&kc),
                t,
                hd,
                &mut vb,
                &mut ab,
                |p, k, out| {
                    let row = table.row(0, toks[p]);
                    reconstruct_v_from_rope_k(&rope, p as f32, k, row, 1.0, out);
                },
            );
        },
    );
    r.report("P3 G2 reconstruct(RopeAction)/full-cache");
    gate(
        "P3 G2 reconstruct(RopeAction) ≤ 1.00× full cache",
        r.median <= 1.0,
        format!(
            "median {:.3} (full {:.2} ms, reconstruct {:.2} ms per head-step, T={t})",
            r.median,
            r.a_ns_per_iter() / 1e6,
            r.b_ns_per_iter() / 1e6
        ),
    );
    let r2 = ab_median_ratio(
        9,
        1,
        1,
        |i| {
            qa[i % hd] += 1e-6;
            sa += attn_head(
                black_box(&qa),
                black_box(&kc),
                t,
                hd,
                &mut va,
                &mut aa,
                |p, _k, out| {
                    out.copy_from_slice(&vc[p * hd..(p + 1) * hd]);
                },
            );
        },
        |i| {
            qb[i % hd] += 1e-6;
            sb += attn_head(
                black_box(&qb),
                black_box(&kc),
                t,
                hd,
                &mut vb,
                &mut ab,
                |p, k, out| {
                    let row = table.row(0, toks[p]);
                    reconstruct_v_from_rope_k(&trope, p as f32, k, row, 1.0, out);
                },
            );
        },
    );
    r2.report("P3 report reconstruct(TableRope)/full-cache");
    black_box(sa + sb);
}
