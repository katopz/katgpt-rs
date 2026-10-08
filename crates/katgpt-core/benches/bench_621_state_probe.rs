//! Plan 621 Phase 1 — `state_probe` G2 micro-bench (T1.4).
//!
//! The bound math (pass-rate + Miller–Madow histogram scan + Wilson CI) must
//! complete in < 1 µs per call at release. Median of 10⁴ runs; inputs are
//! `black_box`ed per call so the pure kernel cannot be hoisted into a
//! constant (the absent-work class — a deleted loop reports a fictional
//! sub-ns time and a well-formed pass).
//!
//! Run: `cargo bench -p katgpt-core --features state_probe --bench
//! bench_621_state_probe` (cargo bench is release by construction — the bar
//! is a release-profile claim, never gated on `debug_assertions`).

#![cfg(feature = "state_probe")]

use katgpt_core::state_probe::{ProbeInput, miller_madow_entropy, probe};
use std::hint::black_box;

/// Median over `iterations` runs, ns. The closure returns an f64 the timer
/// folds into the sample vector so the call cannot be dead-code-eliminated.
fn time_median_ns(f: &mut dyn FnMut() -> f64, iterations: usize) -> f64 {
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let r = f();
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000_000.0;
        times.push((elapsed, r));
    }
    times.sort_by(|a, b| a.0.total_cmp(&b.0));
    times[times.len() / 2].0
}

fn main() {
    // K=64 bins, n=256: the classifier-scale shape (a prompt-sized answer
    // alphabet, a 4-continuation-per-answer ensemble).
    let hist: [u16; 64] = {
        let mut h = [0u16; 64];
        for (i, slot) in h.iter_mut().enumerate() {
            *slot = match i {
                0 => 96,
                1..=6 => 16,
                7..=38 => 2,
                _ => 0, // unobserved alphabet tail — the K′ < K posture
            };
        }
        h
    };
    assert_eq!(hist.iter().map(|&c| c as usize).sum::<usize>(), 256);

    let z95 = 1.959_963_984_540_054;

    // Full probe(): scalars black-boxed per call (hoist defence).
    let mut probe_call = || {
        let input = ProbeInput {
            pass_count: black_box(97u32),
            n: black_box(256u32),
            histogram: black_box(&hist),
        };
        let est = probe(&input, black_box(z95));
        est.v_hat + est.h_mm + est.wilson_lo + est.wilson_hi
    };
    let probe_ns = time_median_ns(&mut probe_call, 10_000);

    // The entropy half alone (the histogram scan dominates the math).
    let mut entropy_call = || miller_madow_entropy(black_box(&hist));
    let entropy_ns = time_median_ns(&mut entropy_call, 10_000);

    // G2 bar: bound math ≤ 1 µs/call at release (Plan 621 T1.4). The
    // expected magnitude is low-hundreds of ns; 1 µs leaves the headroom
    // for shared-box noise while still catching a real regression (a
    // hidden allocation or an accidental O(K²) scan).
    let pass = probe_ns < 1_000.0 && entropy_ns < 1_000.0;

    println!("── Plan 621 T1.4 G2 micro-bench (median of 10_000) ──");
    println!("probe()  full kernel : {probe_ns:8.1} ns/call");
    println!("miller_madow_entropy : {entropy_ns:8.1} ns/call");
    println!("bar                  : < 1000 ns/call");
    if pass {
        println!("   G2 ✓");
    } else {
        println!("   G2 ✗ — bound math exceeded the 1 µs bar");
        std::process::exit(1);
    }
}
