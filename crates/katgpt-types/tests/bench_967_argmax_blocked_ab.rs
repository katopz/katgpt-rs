//! Issue 967 — blocked two-level NEON argmax vs the lane-wise single pass.
//!
//! Found by katgpt-rs Issue 966 G2: at a 151 936 vocab the top-16 pool
//! extraction (a 256-wide `simd_max_f32` block-skip) ran FASTER than
//! `simd_argmax_f32` itself — the lane-wise kernel is one dependent
//! `max → compare → select` chain per 4 floats, so it is latency-bound while
//! an ILP max reduce is throughput-bound.
//!
//! Pre-registered bars (written before the first run, 2026-10-27):
//!
//! - **G1 (gate, every profile):** `simd_argmax_blocked_f32` returns the same
//!   index as `simd_argmax_lanewise_f32` and a value `==` to it, over seeded
//!   uniform / tie-heavy / peaked / signed-zero / all-equal / NaN-sprinkled /
//!   ±inf inputs at ragged lengths straddling the 512 dispatch floor and the
//!   256 block (NaN inputs: index AND value bits identical — the fallback).
//! - **G2 (gate, `--release`):** interleaved median ratio blocked / lane-wise
//!   `≤ 0.80` at n ∈ {32 000, 151 936} for the `iid` shape (max position
//!   uniform), AND no shape (`iid`/`early`/`mid`/`late`) at any n ∈ {512,
//!   4096, 32 000, 151 936} above `1.10` (no-regression bar; n < 512 is the
//!   lane-wise kernel itself, so it is not measured).
//! - **Loud zero:** both arms' outputs feed sinks asserted non-zero; inputs
//!   churn per iteration (fat-LTO anti-hoist).
//!
//! Off aarch64 both functions are `simd_argmax_f32`, so G1 is trivially green
//! and G2 reads ~1.00 — the gate is an aarch64 claim and says so.
//!
//! ```sh
//! cargo test -p katgpt-types --release --test bench_967_argmax_blocked_ab \
//!   -- --nocapture --test-threads=1
//! ```

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_types::simd::{simd_argmax_blocked_f32, simd_argmax_lanewise_f32};
use std::hint::black_box;

const G2_WIN_MAX: f64 = 0.80;
const G2_NOREG_MAX: f64 = 1.10;

fn check(x: &[f32], what: &str) {
    let (ia, va) = simd_argmax_lanewise_f32(x);
    let (ib, vb) = simd_argmax_blocked_f32(x);
    assert_eq!(
        ia,
        ib,
        "{what} n={}: index lane-wise {ia} vs blocked {ib}",
        x.len()
    );
    if x.iter().any(|v| v.is_nan()) {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "{what}: NaN fallback value bits"
        );
    } else {
        assert!(va == vb, "{what} n={}: value {va} vs {vb}", x.len());
    }
}

#[test]
fn g1_blocked_matches_lanewise() {
    let mut rng = fastrand::Rng::with_seed(967);
    let lens = [
        0usize, 1, 3, 4, 7, 255, 256, 257, 511, 512, 513, 767, 768, 1000, 1024, 4095, 4096, 4097,
        32_000, 151_936,
    ];
    for &n in &lens {
        for rep in 0..6 {
            let uni: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            check(&uni, "uniform");
            let ties: Vec<f32> = (0..n).map(|_| rng.u32(0..4) as f32).collect();
            check(&ties, "4-level ties");
            let mut peak = vec![0.0f32; n];
            if n > 0 {
                let p = rng.usize(0..n);
                peak[p] = 1.0;
                if rep % 2 == 0 {
                    // a second, later copy of the max: first index must win
                    let q = rng.usize(p..n);
                    peak[q] = 1.0;
                }
            }
            check(&peak, "peaked");
            let zeros: Vec<f32> = (0..n)
                .map(|_| if rng.bool() { 0.0 } else { -0.0 })
                .collect();
            check(&zeros, "signed zeros");
            check(&vec![-1.5f32; n], "all equal");
            let mut inf: Vec<f32> = (0..n).map(|_| rng.f32() - 0.5).collect();
            if n > 1 {
                inf[rng.usize(0..n)] = f32::NEG_INFINITY;
                inf[rng.usize(0..n)] = f32::INFINITY;
            }
            check(&inf, "±inf");
            check(&vec![f32::NEG_INFINITY; n], "all -inf");
            let mut nan = uni.clone();
            if n > 0 {
                nan[rng.usize(0..n)] = f32::NAN;
            }
            check(&nan, "NaN-sprinkled");
        }
    }
}

/// Max-position shapes (the Issue-817 axis): uniform, first 5%, n/2, last 5%.
fn shaped(n: usize, shape: &str, rng: &mut fastrand::Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n).map(|_| rng.f32() * 0.5).collect();
    let five = (n / 20).max(1);
    let pos = match shape {
        "iid" => return (0..n).map(|_| rng.f32()).collect(),
        "early" => rng.usize(0..five),
        "mid" => n / 2,
        _ => n - 1 - rng.usize(0..five),
    };
    v[pos] = 1.0;
    v
}

#[test]
fn g2_blocked_vs_lanewise() {
    if cfg!(debug_assertions) {
        eprintln!("967 G2: debug build — timing not read; run with --release");
        return;
    }
    let mut rng = fastrand::Rng::with_seed(9670);
    let mut fails = Vec::new();
    eprintln!("\n967 G2 | n | shape | median B/A | [min, max] | lane ns | blocked ns | bar");
    for &n in &[512usize, 4096, 32_000, 151_936] {
        for shape in ["iid", "early", "mid", "late"] {
            // 8 churned copies so neither arm sees one hot, hoistable input.
            let bufs: Vec<Vec<f32>> = (0..8).map(|_| shaped(n, shape, &mut rng)).collect();
            let iters = (2_000_000 / n).clamp(16, 4096);
            let mut sink_a = 0usize;
            let mut sink_b = 0usize;
            let ab = ab_median_ratio(
                21,
                iters,
                3,
                |i| {
                    let (k, _) = simd_argmax_lanewise_f32(black_box(&bufs[i % 8]));
                    sink_a = sink_a.wrapping_add(k + 1);
                },
                |i| {
                    let (k, _) = simd_argmax_blocked_f32(black_box(&bufs[i % 8]));
                    sink_b = sink_b.wrapping_add(k + 1);
                },
            );
            assert!(sink_a != 0 && sink_b != 0, "a sink stayed zero");
            let gated_win = shape == "iid" && n >= 32_000;
            let bar = if gated_win { G2_WIN_MAX } else { G2_NOREG_MAX };
            eprintln!(
                "967 G2 | {n:>7} | {shape:>5} | {:.3} | [{:.3}, {:.3}] | {:.1} | {:.1} | ≤ {bar}",
                ab.median,
                ab.min(),
                ab.max(),
                ab.a_ns_per_iter(),
                ab.b_ns_per_iter()
            );
            if ab.median > bar {
                fails.push(format!("n={n} {shape}: {:.3} > {bar}", ab.median));
            }
        }
    }
    if cfg!(target_arch = "aarch64") {
        assert!(fails.is_empty(), "967 G2 FAIL: {fails:?}");
    }
}
