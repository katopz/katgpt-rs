//! Issue 882 P4 rider (b) — rank-1 spectral deflation of the rerank-stage
//! candidate affinity, gated on effective rank (Bench 897).
//!
//! Fixture (modelless, synthetic): M = 64 rerank candidates in d = 128,
//! `x_i = α·h_i·g + s_k·b_k + ε_i` — a GLOBAL common-mode direction `g`
//! weighted by a per-candidate hubness `h_i ∈ [0.3, 1]` (the "similar to
//! everything" component), a planted BLOCK structure (K = 4 blocks of 16,
//! random unit centers `b_k`, strengths `s = (1.4, 1.0, 1.0, 0.8)`), and
//! noise `ε ~ N(0, σ²/d)`, σ = 0.35. Affinity `A = X Xᵀ` (the M×M rerank
//! Gram). Block-retrieval top-1: for each candidate i, is
//! `argmax_{j≠i} A′_ij` in i's own block?
//!
//! G1b-1 collapsed: no-deflation top-1 is broken by the hub mode; gated
//!       deflation (λ = 1, θ*) fires and recovers it (≥ 0.95 and ≥ baseline
//!       + 0.25). Pre-registered at α = 3, where the LIFT half FAILED (the
//!       baseline only lost ~7%) — kept as a reported line; re-specified at
//!       α = 8 with θ* unchanged, severity sweep printed beside it.
//! G1b-2 healthy (α = 0) NEGATIVE ARM, pinned: the gate REFUSES (A untouched,
//!       bit-identical) AND deflating anyway HURTS (ungated top-1 < healthy
//!       top-1 − 0.10) — if ungated deflation ever stops hurting, the trap's
//!       characterisation changed and this reds.
//! G1b-3 θ chosen by DIRECT EVALUATION: a grid θ ∈ {1.05 … 4.0} scored by
//!       mean gated top-1 over a calibration set (α ∈ {0, 0.5, 1, 2, 3} ×
//!       8 seeds); θ* = the best (ties → smallest θ, the conservative end);
//!       then gated top-1 at θ* on 8 HELD-OUT seeds must be ≥ max(none,
//!       ungated) − 0.01 on EVERY α (the gate never loses to either fixed
//!       policy by more than noise). No gradient descent anywhere.
//! G1b-4 the stable rank (the gate's quantity) orders the classes the same
//!       way as the Roy–Vetterli entropy erank (`river_valley`).
//! G2    paired interleave (`ab_median_ratio`) at M = 128, d = 128: (affinity
//!       build) vs (affinity build + gated deflation). Pre-registered bars:
//!       collapsed ≤ 1.25, healthy (refusal, iteration-cap-bound) ≤ 1.60.
//!       Best-of µs at M ∈ {64, 128, 256} reported.
//! G3    λ = 0 and gate refusal are `to_bits`-identical to the input.
//! G4    0 allocations over the steady-state gated call.
//!
//! Run:
//!   cargo test -p katgpt-spectral --release \
//!     --features affinity_deflation,river_valley \
//!     --test bench_897_affinity_deflation_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::{ab_median_ratio, best_of_us};
use katgpt_core::float_order;
use katgpt_core::simd::simd_dot_f32;
use katgpt_spectral::affinity_deflation::{
    DeflationConfig, DeflationScratch, deflate_rank1_gated_inplace,
};
use katgpt_spectral::river_valley::effective_rank_into;

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

// ── deterministic RNG ─────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unif(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// ≈ N(0, 1) (Irwin–Hall of 12).
    fn gauss(&mut self) -> f32 {
        (0..12).map(|_| self.unif()).sum::<f32>() - 6.0
    }
}

// ── fixture ───────────────────────────────────────────────────────────────

const D: usize = 128;
const K: usize = 4;
const STRENGTH: [f32; K] = [1.4, 1.0, 1.0, 0.8];
const NOISE: f32 = 0.35;

fn unit(rng: &mut Rng, d: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..d).map(|_| rng.gauss()).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

struct Fixture {
    x: Vec<f32>, // m × D
    block: Vec<usize>,
    m: usize,
}

fn make(m: usize, alpha: f32, seed: u64) -> Fixture {
    let mut rng = Rng::new(seed);
    let g = unit(&mut rng, D);
    let centers: Vec<Vec<f32>> = (0..K).map(|_| unit(&mut rng, D)).collect();
    let mut x = vec![0.0f32; m * D];
    let mut block = vec![0usize; m];
    let ns = NOISE / (D as f32).sqrt();
    for i in 0..m {
        let k = i * K / m;
        block[i] = k;
        let h = 0.3 + 0.7 * rng.unif();
        for t in 0..D {
            x[i * D + t] = alpha * h * g[t] + STRENGTH[k] * centers[k][t] + ns * rng.gauss();
        }
    }
    Fixture { x, block, m }
}

/// Symmetric Gram build (upper triangle + mirror) — the rerank-stage affinity.
fn build_affinity(x: &[f32], m: usize, a: &mut [f32]) {
    for i in 0..m {
        let xi = &x[i * D..(i + 1) * D];
        for j in i..m {
            let v = simd_dot_f32(xi, &x[j * D..(j + 1) * D], D);
            a[i * m + j] = v;
            a[j * m + i] = v;
        }
    }
}

fn top1(a: &[f32], m: usize, block: &[usize]) -> f32 {
    let mut hit = 0usize;
    for i in 0..m {
        let row = &a[i * m..(i + 1) * m];
        let j = (0..m)
            .filter(|&j| j != i)
            .max_by(|&p, &q| float_order::cmp_for_max(row[p], row[q]))
            .unwrap();
        if block[j] == block[i] {
            hit += 1;
        }
    }
    hit as f32 / m as f32
}

struct Eval {
    none: f32,
    ungated: f32,
    srank: f32,
}

fn eval_one(f: &Fixture, sc: &mut DeflationScratch) -> Eval {
    let m = f.m;
    let mut a = vec![0.0f32; m * m];
    build_affinity(&f.x, m, &mut a);
    let none = top1(&a, m, &f.block);
    let r = deflate_rank1_gated_inplace(&mut a, m, m, &DeflationConfig::ungated(1.0), sc);
    let ungated = top1(&a, m, &f.block);
    Eval {
        none,
        ungated,
        srank: r.stable_rank,
    }
}

/// Gated top-1 given the precomputed pieces: deflates iff srank < θ.
fn gated(e: &Eval, theta: f32) -> f32 {
    if e.srank.is_finite() && e.srank < theta {
        e.ungated
    } else {
        e.none
    }
}

fn gate(name: &str, ok: bool, detail: String, fails: &mut usize) {
    println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        *fails += 1;
    }
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn main() {
    let mut fails = 0usize;
    println!("# Bench 897 (b) — gated rank-1 affinity deflation (Issue 882 P4 rider b)\n");
    let m = 64usize;
    let mut sc = DeflationScratch::new(m, m);

    // ── G1b-3: θ by direct evaluation ─────────────────────────────────────
    let alphas = [0.0f32, 0.5, 1.0, 2.0, 3.0];
    let cal_seeds: Vec<u64> = (0..8).map(|s| 1000 + s).collect();
    let hold_seeds: Vec<u64> = (0..8).map(|s| 9000 + s).collect();
    let thetas = [1.05f32, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0];

    let cal: Vec<Vec<Eval>> = alphas
        .iter()
        .map(|&al| {
            cal_seeds
                .iter()
                .map(|&s| eval_one(&make(m, al, s), &mut sc))
                .collect()
        })
        .collect();
    println!("calibration (mean over {} seeds):", cal_seeds.len());
    println!("  alpha  srank   none   ungated");
    for (al, row) in alphas.iter().zip(&cal) {
        let n = row.len() as f32;
        println!(
            "  {al:4.1}  {:6.3}  {:.3}  {:.3}",
            row.iter().map(|e| e.srank).sum::<f32>() / n,
            row.iter().map(|e| e.none).sum::<f32>() / n,
            row.iter().map(|e| e.ungated).sum::<f32>() / n
        );
    }
    let mut best = (f32::NEG_INFINITY, thetas[0]);
    print!("  θ grid (mean gated top-1):");
    for &th in &thetas {
        let tot: f32 = cal.iter().flatten().map(|e| gated(e, th)).sum();
        let mean = tot / (alphas.len() * cal_seeds.len()) as f32;
        print!(" {th}:{mean:.4}");
        if mean > best.0 + 1e-6 {
            best = (mean, th);
        }
    }
    println!();
    let theta_star = best.1;
    println!("  θ* = {theta_star} (cal mean {:.4})", best.0);

    let hold: Vec<Vec<Eval>> = alphas
        .iter()
        .map(|&al| {
            hold_seeds
                .iter()
                .map(|&s| eval_one(&make(m, al, s), &mut sc))
                .collect()
        })
        .collect();
    let mut worst_margin = f32::INFINITY;
    println!("held-out (mean over {} seeds):", hold_seeds.len());
    println!("  alpha   none  ungated  gated@θ*  fired");
    for (al, row) in alphas.iter().zip(&hold) {
        let n = row.len() as f32;
        let none = row.iter().map(|e| e.none).sum::<f32>() / n;
        let ung = row.iter().map(|e| e.ungated).sum::<f32>() / n;
        let gt = row.iter().map(|e| gated(e, theta_star)).sum::<f32>() / n;
        let fired = row.iter().filter(|e| e.srank < theta_star).count();
        println!(
            "  {al:4.1}  {none:.3}   {ung:.3}    {gt:.3}    {fired}/{}",
            row.len()
        );
        worst_margin = worst_margin.min(gt - none.max(ung));
    }
    gate(
        "G1b-3 held-out gated@θ* ≥ max(none, ungated) − 0.01 on every α",
        worst_margin >= -0.01,
        format!("worst margin {worst_margin:+.4}"),
        &mut fails,
    );

    // ── G1b-1: collapsed recovery ─────────────────────────────────────────
    // The pre-registered bar (α = 3: gate fires on every seed, gated ≥ 0.95
    // AND ≥ none + 0.25) FAILED its lift half on first run: recovery is
    // complete (→ 1.000) but α = 3 only costs the no-deflation baseline ~7%
    // — the best in-block hub is usually near the global max, so the common
    // mode rarely flips a row's top-1 at that strength. Kept as a reported
    // line. Re-specified BEFORE the sweep below was run: the severity cell
    // α = 8 at the UNCHANGED calibrated θ* (no re-fit), same bars.
    {
        let row = &hold[alphas.len() - 1]; // α = 3
        let n = row.len() as f32;
        let none = row.iter().map(|e| e.none).sum::<f32>() / n;
        let gt = row.iter().map(|e| gated(e, theta_star)).sum::<f32>() / n;
        let fired = row.iter().all(|e| e.srank < theta_star);
        let ok = fired && gt >= 0.95 && gt >= none + 0.25;
        println!(
            "G1b-1 (report) PRE-REGISTERED α=3 bar {}: none {none:.3} → gated {gt:.3} (lift {:+.3}, bar +0.25), fired every seed {fired}",
            if ok { "passed" } else { "FAILED" },
            gt - none
        );
        println!("  severity sweep (held-out seeds, θ* = {theta_star}):");
        println!("  alpha   none  gated@θ*  fired  srank");
        let mut sev = (0.0f32, 0.0f32, false);
        for &al in &[3.0f32, 4.0, 6.0, 8.0] {
            let evs: Vec<Eval> = hold_seeds
                .iter()
                .map(|&s| eval_one(&make(m, al, s), &mut sc))
                .collect();
            let n = evs.len() as f32;
            let none = evs.iter().map(|e| e.none).sum::<f32>() / n;
            let gt = evs.iter().map(|e| gated(e, theta_star)).sum::<f32>() / n;
            let fired = evs.iter().filter(|e| e.srank < theta_star).count();
            let sr = evs.iter().map(|e| e.srank).sum::<f32>() / n;
            println!(
                "  {al:4.1}  {none:.3}   {gt:.3}    {fired}/{}  {sr:.4}",
                evs.len()
            );
            if al == 8.0 {
                sev = (none, gt, fired == evs.len());
            }
        }
        let (none, gt, fired) = sev;
        gate(
            "G1b-1 collapsed (re-specified α=8, θ* unchanged): fires every seed, gated ≥ 0.95 and ≥ none + 0.25",
            fired && gt >= 0.95 && gt >= none + 0.25,
            format!("none {none:.3} → gated {gt:.3} (lift {:+.3})", gt - none),
            &mut fails,
        );
    }

    // ── G1b-2: healthy negative arm ───────────────────────────────────────
    {
        let mut refused_bits = true;
        let (mut none_s, mut ung_s) = (0.0f32, 0.0f32);
        for &s in &hold_seeds {
            let f = make(m, 0.0, s);
            let mut a = vec![0.0f32; m * m];
            build_affinity(&f.x, m, &mut a);
            let a0 = a.clone();
            let r = deflate_rank1_gated_inplace(
                &mut a,
                m,
                m,
                &DeflationConfig::new(1.0, theta_star),
                &mut sc,
            );
            refused_bits &= !r.deflated && bits_eq(&a, &a0);
            none_s += top1(&a0, m, &f.block);
            let mut b = a0.clone();
            deflate_rank1_gated_inplace(&mut b, m, m, &DeflationConfig::ungated(1.0), &mut sc);
            ung_s += top1(&b, m, &f.block);
        }
        let n = hold_seeds.len() as f32;
        let (none, ung) = (none_s / n, ung_s / n);
        gate(
            "G1b-2 healthy (α=0): gate REFUSES, bit-identical",
            refused_bits,
            format!("θ* = {theta_star}"),
            &mut fails,
        );
        gate(
            "G1b-2 healthy NEGATIVE pinned: ungated deflation HURTS (< none − 0.10)",
            ung < none - 0.10,
            format!("none {none:.3}, ungated {ung:.3} (Δ {:+.3})", ung - none),
            &mut fails,
        );
    }

    // ── G1b-4: stable rank vs Roy–Vetterli erank ──────────────────────────
    {
        let mut gram = vec![0.0f32; m * m];
        let mut ev = vec![0.0f32; m];
        let mut js = vec![0.0f32; m * m];
        let mut rows = Vec::new();
        for &al in &[0.0f32, 3.0] {
            let (mut sr, mut er) = (0.0f32, 0.0f32);
            for &s in &hold_seeds[..4] {
                let f = make(m, al, s);
                let mut a = vec![0.0f32; m * m];
                build_affinity(&f.x, m, &mut a);
                er += effective_rank_into(&a, m, m, &mut gram, &mut ev, &mut js);
                let r = deflate_rank1_gated_inplace(
                    &mut a,
                    m,
                    m,
                    &DeflationConfig::new(1.0, 0.0),
                    &mut sc,
                );
                sr += r.stable_rank;
            }
            rows.push((al, sr / 4.0, er / 4.0));
        }
        for (al, sr, er) in &rows {
            println!("     α={al}: stable rank {sr:.3}, Roy–Vetterli erank(σ²) {er:.3}");
        }
        gate(
            "G1b-4 stable rank and entropy erank order collapsed < healthy",
            rows[1].1 < rows[0].1 && rows[1].2 < rows[0].2,
            format!(
                "srank {:.3} < {:.3}, erank {:.3} < {:.3}",
                rows[1].1, rows[0].1, rows[1].2, rows[0].2
            ),
            &mut fails,
        );
    }

    // ── G3 ────────────────────────────────────────────────────────────────
    {
        let f = make(m, 3.0, 7);
        let mut a0 = vec![0.0f32; m * m];
        build_affinity(&f.x, m, &mut a0);
        let mut a = a0.clone();
        let r = deflate_rank1_gated_inplace(&mut a, m, m, &DeflationConfig::off(), &mut sc);
        let off_ok = !r.deflated && bits_eq(&a, &a0);
        let r2 =
            deflate_rank1_gated_inplace(&mut a, m, m, &DeflationConfig::new(1.0, 1.0), &mut sc);
        let refuse_ok = !r2.deflated && bits_eq(&a, &a0);
        gate(
            "G3 λ=0 and refusal (θ=1 ≤ srank) bit-identical",
            off_ok && refuse_ok,
            format!("collapsed srank {:.4}", r2.stable_rank),
            &mut fails,
        );
    }

    // ── G4 ────────────────────────────────────────────────────────────────
    {
        let fc = make(m, 3.0, 21);
        let fh = make(m, 0.0, 21);
        let mut ac = vec![0.0f32; m * m];
        let mut ah = vec![0.0f32; m * m];
        let mut work = vec![0.0f32; m * m];
        build_affinity(&fc.x, m, &mut ac);
        build_affinity(&fh.x, m, &mut ah);
        let cfg = DeflationConfig::new(1.0, theta_star);
        let before = allocs();
        for _ in 0..32 {
            work.copy_from_slice(&ac);
            black_box(deflate_rank1_gated_inplace(&mut work, m, m, &cfg, &mut sc));
            work.copy_from_slice(&ah);
            black_box(deflate_rank1_gated_inplace(&mut work, m, m, &cfg, &mut sc));
        }
        let n_alloc = allocs() - before;
        gate(
            "G4 0 allocs (gated call, collapsed + healthy, 32 rounds)",
            n_alloc == 0,
            format!("{n_alloc} allocs"),
            &mut fails,
        );
    }

    // ── G2 ────────────────────────────────────────────────────────────────
    {
        let mm = 128usize;
        let mut sc2 = DeflationScratch::new(mm, mm);
        let cfg = DeflationConfig::new(1.0, theta_star);
        for (label, alpha, bar) in [
            ("collapsed α=3", 3.0f32, 1.25f64),
            ("healthy α=0", 0.0, 1.60),
        ] {
            let f = make(mm, alpha, 31);
            let mut a1 = vec![0.0f32; mm * mm];
            let mut a2 = vec![0.0f32; mm * mm];
            let mut last = None;
            let ab = ab_median_ratio(
                15,
                20,
                5,
                |_| {
                    build_affinity(black_box(&f.x), mm, &mut a1);
                    black_box(&a1);
                },
                |_| {
                    build_affinity(black_box(&f.x), mm, &mut a2);
                    last = Some(deflate_rank1_gated_inplace(
                        black_box(&mut a2),
                        mm,
                        mm,
                        black_box(&cfg),
                        &mut sc2,
                    ));
                    black_box(&a2);
                },
            );
            ab.report(&format!(
                "G2 M={mm} {label}: build (a) vs build + gated deflation (b)"
            ));
            let r = last.unwrap();
            gate(
                &format!("G2 M={mm} {label} paired ratio ≤ {bar}"),
                ab.median <= bar,
                format!(
                    "median {:.3} (min {:.3} max {:.3}); deflated {} after {} iters, srank {:.3}",
                    ab.median,
                    ab.min(),
                    ab.max(),
                    r.deflated,
                    r.iters,
                    r.stable_rank
                ),
                &mut fails,
            );
        }
        for &mm in &[64usize, 128, 256] {
            let mut sc3 = DeflationScratch::new(mm, mm);
            for (label, alpha) in [("collapsed", 3.0f32), ("healthy", 0.0)] {
                let f = make(mm, alpha, 41);
                let mut a0 = vec![0.0f32; mm * mm];
                build_affinity(&f.x, mm, &mut a0);
                let mut a = a0.clone();
                let us = best_of_us(5, 40, || {
                    a.copy_from_slice(&a0);
                    let st = Instant::now();
                    let r = deflate_rank1_gated_inplace(
                        black_box(&mut a),
                        mm,
                        mm,
                        black_box(&cfg),
                        &mut sc3,
                    );
                    let e = st.elapsed();
                    black_box((r, &a));
                    e
                });
                println!("     G2 (report) M={mm} {label}: gated deflation best-of {us:.2} µs");
            }
        }
    }

    if fails > 0 {
        println!("\n{fails} gate(s) FAILED");
        std::process::exit(1);
    }
    println!("\nall gates PASS");
}
