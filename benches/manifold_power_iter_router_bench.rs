//! Manifold Power Iteration MoE Router — N/D sweep benchmark (Plan 279 Phase 3).
//!
//! Uses `std::time::Instant` (NOT criterion — matches `attn_match_router_bench.rs`).
//!
//! Run:
//! ```bash
//! cargo run --release --bench manifold_power_iter_router_bench \
//!            --features manifold_power_iter_router
//! ```
//!
//! Sweeps `N ∈ {8, 32, 64, 256}` × `D ∈ {64, 256, 1024}` and measures:
//! - Gram compute time (warm tier, once per snapshot).
//! - MPI recondition time (paper Eq. 4–5, once per snapshot).
//! - Sigmoid gate time (per-token — should be flat / identical to vanilla).
//! - λ_alignment and maxvio before → after for each (N, D).

#![cfg(feature = "manifold_power_iter_router")]

use katgpt_rs::manifold_power_iter_router::{
    compute_diagnostics, compute_expert_gram_into, gate_sigmoid_topk, manifold_power_iter_router,
};
use katgpt_rs::spectral_retract::PowerRetractScratch;
use std::time::{Duration, Instant};

/// Deterministic xorshift64 PRNG.
fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0);
    }
    v
}

/// Best-of-N wall-clock microseconds for a closure.
fn bench_us(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut best = Duration::from_secs(60);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed();
        if dt < best {
            best = dt;
        }
    }
    best.as_secs_f64() * 1e6
}

fn main() {
    println!("=== Manifold Power Iteration MoE Router Benchmark (Plan 279) ===\n");

    let n_values: &[usize] = &[8, 32, 64, 256];
    let d_values: &[usize] = &[64, 256, 1024];

    println!(
        "{:>5} {:>5} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "N", "D", "gram_us", "mpi_us", "gate_us", "total_us", "λ_before", "λ_after"
    );
    println!("{}", "-".repeat(95));

    for &n in n_values {
        for &d in d_values {
            // Build router + gate weights + grams.
            let r_seed: Vec<f32> = seeded_vec(42 + (n * d) as u64, n * d);
            let w_g: Vec<Vec<f32>> = (0..n)
                .map(|i| seeded_vec(100 + i as u64 + (d * d) as u64, d * d))
                .collect();

            // Gram compute (warm tier — measured separately).
            let gram_us = bench_us(2, 5, || {
                let mut grams: Vec<Vec<f32>> = Vec::with_capacity(n);
                for w in &w_g {
                    let mut g = vec![0.0f32; d * d];
                    compute_expert_gram_into(w, d, &mut g);
                    grams.push(g);
                }
                std::hint::black_box(&grams);
            });

            // Pre-build grams for MPI measurement.
            let grams: Vec<Vec<f32>> = w_g
                .iter()
                .map(|w| {
                    let mut g = vec![0.0f32; d * d];
                    compute_expert_gram_into(w, d, &mut g);
                    g
                })
                .collect();
            let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();

            let target_norm = 1.0f32 / (n as f32).sqrt();
            let (lambda_before, _) = compute_diagnostics(&r_seed, &grams_ref, n, d, target_norm);

            // MPI recondition (once per snapshot).
            let mut r_work = r_seed.clone();
            let mut scratch = PowerRetractScratch::new(d);
            let mpi_us = bench_us(3, 10, || {
                r_work.copy_from_slice(&r_seed);
                let _res = manifold_power_iter_router(
                    &mut r_work,
                    &grams_ref,
                    n,
                    d,
                    1.0,
                    1,
                    &mut scratch,
                );
                std::hint::black_box(&_res);
            });

            // Capture "after" λ from the last MPI run.
            let mut r_after = r_seed.clone();
            let mut s2 = PowerRetractScratch::new(d);
            let res = manifold_power_iter_router(&mut r_after, &grams_ref, n, d, 1.0, 1, &mut s2);

            // Sigmoid gate (per-token — should be ~flat across N×D since it's
            // just a matvec; the point of G3 is identical timing R vs R').
            let x = seeded_vec(7, d);
            let mut scores = vec![0.0f32; n];
            let gate_us = bench_us(5, 200, || {
                let _topk = gate_sigmoid_topk(&x, &r_after, n, d, 1.0, 3.min(n), &mut scores);
                std::hint::black_box(&_topk);
            });

            let total_us = gram_us + mpi_us;

            println!(
                "{:>5} {:>5} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>10.3} {:>10.3}",
                n,
                d,
                gram_us,
                mpi_us,
                gate_us,
                total_us,
                lambda_before,
                res.lambda_alignment
            );
        }
    }
    println!();

    // Game-scale focus: N=8, D=256 — must be sub-ms total (G4).
    println!("G4 game-scale focus (N=8, D=256):");
    let n = 8usize;
    let d = 256usize;
    let r = seeded_vec(42, n * d);
    let w_g: Vec<Vec<f32>> = (0..n).map(|i| seeded_vec(100 + i as u64, d * d)).collect();
    let mut grams: Vec<Vec<f32>> = Vec::with_capacity(n);
    let t0 = Instant::now();
    for w in &w_g {
        let mut g = vec![0.0f32; d * d];
        compute_expert_gram_into(w, d, &mut g);
        grams.push(g);
    }
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();
    let mut r_work = r.clone();
    let mut scratch = PowerRetractScratch::new(d);
    let _res = manifold_power_iter_router(&mut r_work, &grams_ref, n, d, 1.0, 1, &mut scratch);
    let total = t0.elapsed();
    println!("  total (gram + MPI) = {:?}", total);
    println!(
        "  G4 (sub-ms): {}",
        if total.as_secs_f64() < 1e-3 {
            "PASS"
        } else {
            "FAIL"
        }
    );
}
