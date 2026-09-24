//! Bench 885 — calibrated-mass router gate GOAT (Issue 880, Bench 884
//! promotion candidate (a)).
//!
//! `gate_sigmoid_topk_mass_into` = the incumbent's logits + exact-k selection
//! sort **plus** an `exact_mass_admit` τ bisection, so its weights sum to k.
//! Bench 884's 0.906× compared the two operators as ALTERNATIVES at k = N/10;
//! the upgrade COMPOSES them, so this bench measures the composition against
//! the incumbent in the router's own regimes rather than inheriting that
//! ratio.
//!
//! Gates:
//! - **G1**: katgpt-spectral unit suite (t15-t19: selection equivalence,
//!   sum-to-k, τ coupling, σ(z − τ) identity, budget extremes) + the spot
//!   calibration assert here.
//! - **G2**: `b/a` time ratio (a = incumbent, b = calibrated) via the shared
//!   interleaved `ab_median_ratio`, per regime; plus the calibration column
//!   `|Σw − k|` for BOTH arms — the quantity only `b` controls. The ratio is
//!   RECORDED; the verdict against the issue's ≤ 1.05× bar is printed, and a
//!   gross ceiling (a de-optimization catch, not a measurement) is asserted.
//! - **G3**: katgpt-spectral t14 pins the incumbent byte-identical across the
//!   helper extraction; every pre-existing router test passes unchanged.
//! - **G4**: steady-state `_into` calls allocate nothing (counting allocator,
//!   canary-checked).
//!
//! # Run
//!
//! ```bash
//! cargo test --features calibrated_mass_gate \
//!   --test bench_885_calibrated_mass_gate_goat --release -- --nocapture --test-threads=1
//! ```

#![cfg(feature = "calibrated_mass_gate")]

#[path = "common/ab_timing.rs"]
mod ab_timing;

#[path = "../crates/katgpt-core/tests/common/mod.rs"]
#[allow(dead_code, unused_macros, unused_imports)]
mod common;
counting_allocator!();

use ab_timing::ab_median_ratio;
use katgpt_spectral::manifold_power_iter_router::{
    gate_sigmoid_topk_into, gate_sigmoid_topk_mass_into,
};
use std::hint::black_box;

/// Deterministic SplitMix64 (the workspace-bench idiom — no rand dep).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_signed(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

const D_MODEL: usize = 16;
const BETA: f32 = 1.3;
const X_POOL: usize = 16;

struct Regime {
    name: &'static str,
    n: usize,
    k: usize,
    iters: usize,
}

const REGIMES: [Regime; 3] = [
    Regime { name: "hot N=64 k=4", n: 64, k: 4, iters: 64 },
    Regime { name: "game N=256 k=8", n: 256, k: 8, iters: 32 },
    Regime { name: "bench884 N=1000 k=100", n: 1000, k: 100, iters: 8 },
];

struct Fixture {
    r: Vec<f32>,
    xs: Vec<Vec<f32>>,
}

fn fixture(n: usize, seed: u64) -> Fixture {
    let mut rng = SplitMix64(seed);
    let r = (0..n * D_MODEL).map(|_| rng.next_signed()).collect();
    let xs = (0..X_POOL)
        .map(|_| (0..D_MODEL).map(|_| rng.next_signed()).collect())
        .collect();
    Fixture { r, xs }
}

fn sum_f64(v: &[f32]) -> f64 {
    v.iter().map(|&m| m as f64).sum()
}

#[test]
fn g2_calibrated_gate_vs_incumbent() {
    println!("\nBench 885 — calibrated-mass gate vs gate_sigmoid_topk_into (T = 1, β = {BETA}, d = {D_MODEL})");
    println!("| regime | b/a median | min..max | a ns/call | b ns/call | |Σw−k| incumbent | |Σw−k| calibrated |");
    println!("|---|---|---|---|---|---|---|");
    let mut worst = 0.0f64;
    for (ri, g) in REGIMES.iter().enumerate() {
        let f = fixture(g.n, 0x880 + ri as u64);
        let (n, k) = (g.n, g.k);

        // Calibration column (G1 spot + the quantity only b controls).
        let mut s_a = vec![0.0f32; n];
        let mut i_a = vec![0usize; n];
        let mut z_b = vec![0.0f32; n];
        let mut m_b = vec![0.0f32; n];
        let mut i_b = vec![0usize; n];
        let (mut cal_a, mut cal_b) = (0.0f64, 0.0f64);
        for x in &f.xs {
            gate_sigmoid_topk_into(x, &f.r, n, D_MODEL, BETA, k, &mut s_a, &mut i_a);
            gate_sigmoid_topk_mass_into(
                x, &f.r, n, D_MODEL, BETA, k, 1.0, &mut z_b, &mut m_b, &mut i_b,
            );
            cal_a = cal_a.max((sum_f64(&s_a) - k as f64).abs());
            cal_b = cal_b.max((sum_f64(&m_b) - k as f64).abs());
        }
        let tol = 1e-3 + 2.0e-7 * n as f64;
        assert!(cal_b <= tol, "{}: calibrated |Σm − k| = {cal_b} > {tol}", g.name);

        let (mut sink_a, mut sink_b) = (0.0f32, 0.0f32);
        let ab = ab_median_ratio(
            31,
            g.iters,
            3,
            |i| {
                let x = &f.xs[i % X_POOL];
                let kk = gate_sigmoid_topk_into(
                    black_box(x), &f.r, n, D_MODEL, BETA, k, &mut s_a, &mut i_a,
                );
                sink_a += s_a[i_a[kk - 1]];
            },
            |i| {
                let x = &f.xs[i % X_POOL];
                let (kk, tau) = gate_sigmoid_topk_mass_into(
                    black_box(x), &f.r, n, D_MODEL, BETA, k, 1.0, &mut z_b, &mut m_b, &mut i_b,
                );
                sink_b += m_b[i_b[kk - 1]] + tau * 1e-30;
            },
        );
        black_box(sink_a + sink_b);
        println!(
            "| {} | {:.2}× | {:.2}..{:.2} | {:.0} | {:.0} | {:.3e} | {:.3e} |",
            g.name,
            ab.median,
            ab.min(),
            ab.max(),
            ab.a_ns_per_iter(),
            ab.b_ns_per_iter(),
            cal_a,
            cal_b
        );
        worst = worst.max(ab.median);
    }
    let verdict = if worst <= 1.05 { "PASS" } else { "FAIL" };
    println!("G2 vs the issue's ≤ 1.05× bar: {verdict} (worst regime median {worst:.2}×)");
    // Gross ceiling: the composition is incumbent + ≤ 50 bisection passes, so
    // anything past 200× is a de-optimization, not box noise.
    assert!(worst < 200.0, "calibrated gate {worst:.1}× the incumbent — de-optimized");
}

#[test]
fn g4_calibrated_gate_steady_state_is_alloc_free() {
    assert_counter_is_live();
    let n = 256usize;
    let f = fixture(n, 0x8804);
    let mut z = vec![0.0f32; n];
    let mut m = vec![0.0f32; n];
    let mut idx = vec![0usize; n];
    let ((), allocs) = alloc_delta(|| {
        let mut acc = 0.0f32;
        for i in 0..2000usize {
            let k = 1 + i % 32;
            let (kk, tau) = gate_sigmoid_topk_mass_into(
                black_box(&f.xs[i % X_POOL]),
                &f.r,
                n,
                D_MODEL,
                BETA,
                k,
                1.0,
                &mut z,
                &mut m,
                &mut idx,
            );
            acc += m[idx[kk - 1]] + tau * 1e-30;
        }
        black_box(acc);
    });
    assert_eq!(allocs, 0, "steady-state gate_sigmoid_topk_mass_into allocated {allocs}×");
}
