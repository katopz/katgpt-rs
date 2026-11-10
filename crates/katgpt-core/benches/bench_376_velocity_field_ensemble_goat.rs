//! Plan 376 Phase 3 — Velocity-Field Ensemble GOAT gate G4 (latency) bench.
//!
//! Measures the three latency-sensitive operations and compares against the
//! Plan 376 T3.4 targets:
//!
//! - `fit_into` (N=50, P=8, D=8) ≤ 50 µs
//! - single `eval_into` ≤ 200 ns
//! - `eval_batch_into` for 1000 states ≤ 5 ms
//!
//! Also re-verifies G3 (zero-alloc) on the same config via the
//! `CountingAllocator` (mirrors the bench_370 pattern: latency + alloc in one
//! bench file with `harness = false`).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/vfe_376 cargo build --release -p katgpt-core \
//!     --features velocity_field_ensemble --bench bench_376_velocity_field_ensemble_goat
//! /tmp/vfe_376/release/deps/bench_376_velocity_field_ensemble_goat-* --nocapture
//! ```

use katgpt_core::velocity_field_ensemble::{
    ClosureField, EnsembleFitScratch, VelocityFieldEnsemble,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Config ────────────────────────────────────────────────────────────────

const D: usize = 8;
const P: usize = 8;
const N_FIT_PAIRS: usize = 50;
const N_BATCH: usize = 1000;

// Latency targets (Plan 376 T3.4).
const TARGET_FIT_US: u64 = 50;
const TARGET_EVAL_NS: u64 = 200;
const TARGET_BATCH_MS: u64 = 5;

const WARMUP_ITERS: usize = 1000;
const MEASURE_ITERS: usize = 10_000;

// ── Linear fields (P=8 distinct named fns; same type for [F; P]) ──────────

fn field_0(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[k] * 0.1;
    }
}
fn field_1(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 1) % D] * 0.15;
    }
}
fn field_2(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 2) % D] * 0.2 - x[k] * 0.05;
    }
}
fn field_3(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[D - 1 - k] * 0.12;
    }
}
fn field_4(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[k % 4] * 0.18 + x[(k + 4) % D] * 0.07;
    }
}
fn field_5(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 3) % D] * 0.11 - x[(k + 5) % D] * 0.04;
    }
}
fn field_6(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = (x[k] + x[(k + 1) % D]) * 0.09;
    }
}
fn field_7(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 6) % D] * 0.13 + x[(k + 7) % D] * 0.06;
    }
}

type FieldFn = fn(&[f32], &mut [f32; D]);

fn build_ensemble() -> VelocityFieldEnsemble<ClosureField<D, FieldFn>, P, D> {
    let fields = [
        ClosureField::<D, FieldFn>::new(0, field_0),
        ClosureField::<D, FieldFn>::new(1, field_1),
        ClosureField::<D, FieldFn>::new(2, field_2),
        ClosureField::<D, FieldFn>::new(3, field_3),
        ClosureField::<D, FieldFn>::new(4, field_4),
        ClosureField::<D, FieldFn>::new(5, field_5),
        ClosureField::<D, FieldFn>::new(6, field_6),
        ClosureField::<D, FieldFn>::new(7, field_7),
    ];
    VelocityFieldEnsemble::new(fields)
}

// ── Gate result ───────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

// ── Median helper ─────────────────────────────────────────────────────────

fn median_u64(v: &mut [u64]) -> u64 {
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    }
}

// ── Gates ─────────────────────────────────────────────────────────────────

fn gate_g4_fit_latency() -> GateResult {
    println!("\n--- G4: fit_into latency (N={N_FIT_PAIRS}, P={P}, D={D}) ---");

    // Pre-build pairs (one-time alloc, outside the measured region).
    let xs: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut x = [0.0f32; D];
            for (k, slot) in x.iter_mut().enumerate().take(D) {
                *slot = ((i + k) as f32) * 0.01;
            }
            x
        })
        .collect();
    let ys: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut y = [0.0f32; D];
            for (k, slot) in y.iter_mut().enumerate().take(D) {
                *slot = ((i * 2 + k) as f32) * 0.005;
            }
            y
        })
        .collect();
    let x_refs: Vec<&[f32]> = xs.iter().map(|v| &v[..]).collect();
    let y_refs: Vec<&[f32]> = ys.iter().map(|v| &v[..]).collect();

    let mut ensemble = build_ensemble();
    let mut scratch = EnsembleFitScratch::<P, D>::new();

    // Warmup.
    for _ in 0..100 {
        ensemble.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);
    }

    // Measure: batch timing for sub-µs resolution.
    const BATCH: usize = 100;
    let iters = MEASURE_ITERS / BATCH;
    let mut per_call_ns = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            ensemble.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);
        }
        let dt = t0.elapsed();
        per_call_ns.push((dt.as_nanos() as u64) / (BATCH as u64));
    }
    let med = median_u64(&mut per_call_ns);

    println!(
        "  fit_into p50: {} ns  (target ≤ {} µs = {} ns)",
        med,
        TARGET_FIT_US,
        TARGET_FIT_US * 1000
    );

    let passed = med <= TARGET_FIT_US * 1000;
    if passed {
        GateResult {
            name: "G4 fit_into latency",
            passed: true,
            detail: format!("{} ns ≤ {} ns target", med, TARGET_FIT_US * 1000),
        }
    } else {
        GateResult {
            name: "G4 fit_into latency",
            passed: false,
            detail: format!("{} ns > {} ns target", med, TARGET_FIT_US * 1000),
        }
    }
}

fn gate_g4_eval_latency() -> GateResult {
    println!("\n--- G4: eval_into latency (single call, P={P}, D={D}) ---");

    let mut ensemble = build_ensemble();
    let mut scratch = EnsembleFitScratch::<P, D>::new();
    // Minimal fit so eta is non-uniform.
    let xs_fit: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut x = [0.0f32; D];
            for (k, slot) in x.iter_mut().enumerate().take(D) {
                *slot = ((i + k) as f32) * 0.01;
            }
            x
        })
        .collect();
    let ys_fit: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut y = [0.0f32; D];
            for (k, slot) in y.iter_mut().enumerate().take(D) {
                *slot = ((i * 2 + k) as f32) * 0.005;
            }
            y
        })
        .collect();
    let x_refs: Vec<&[f32]> = xs_fit.iter().map(|v| &v[..]).collect();
    let y_refs: Vec<&[f32]> = ys_fit.iter().map(|v| &v[..]).collect();
    ensemble.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);

    let x = [0.5f32; D];
    let mut out = [0.0f32; D];
    let mut eval_scratch = [0.0f32; D];

    // Warmup.
    for _ in 0..WARMUP_ITERS {
        {
            ensemble.eval_into(
                black_box(&x),
                black_box(&mut out),
                black_box(&mut eval_scratch),
            );
            black_box(());
        }
    }

    // Measure.
    const BATCH: usize = 1000;
    let iters = MEASURE_ITERS / BATCH;
    let mut per_call_ns = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        for _ in 0..BATCH {
            {
                ensemble.eval_into(
                    black_box(&x),
                    black_box(&mut out),
                    black_box(&mut eval_scratch),
                );
                black_box(());
            }
        }
        let dt = t0.elapsed();
        per_call_ns.push((dt.as_nanos() as u64) / (BATCH as u64));
    }
    let med = median_u64(&mut per_call_ns);

    println!("  eval_into p50: {med} ns  (target ≤ {TARGET_EVAL_NS} ns)");

    let passed = med <= TARGET_EVAL_NS;
    GateResult {
        name: "G4 eval_into latency",
        passed,
        detail: format!(
            "{} ns {} {} ns target",
            med,
            if passed { "≤" } else { ">" },
            TARGET_EVAL_NS
        ),
    }
}

fn gate_g4_batch_latency() -> GateResult {
    println!("\n--- G4: eval_batch_into latency (N_batch={N_BATCH}, P={P}, D={D}) ---");

    let mut ensemble = build_ensemble();
    let mut scratch = EnsembleFitScratch::<P, D>::new();
    let xs_fit: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut x = [0.0f32; D];
            for (k, slot) in x.iter_mut().enumerate().take(D) {
                *slot = ((i + k) as f32) * 0.01;
            }
            x
        })
        .collect();
    let ys_fit: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut y = [0.0f32; D];
            for (k, slot) in y.iter_mut().enumerate().take(D) {
                *slot = ((i * 2 + k) as f32) * 0.005;
            }
            y
        })
        .collect();
    let x_refs: Vec<&[f32]> = xs_fit.iter().map(|v| &v[..]).collect();
    let y_refs: Vec<&[f32]> = ys_fit.iter().map(|v| &v[..]).collect();
    ensemble.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);

    // Batch buffers (one-time alloc, outside measured region).
    let batch_x: Vec<[f32; D]> = (0..N_BATCH)
        .map(|i| {
            let mut x = [0.0f32; D];
            for (k, slot) in x.iter_mut().enumerate().take(D) {
                *slot = ((i + k) as f32) * 0.001;
            }
            x
        })
        .collect();
    let mut batch_out: Vec<[f32; D]> = vec![[0.0f32; D]; N_BATCH];
    let batch_x_refs: Vec<&[f32]> = batch_x.iter().map(|v| &v[..]).collect();
    let mut batch_out_refs: Vec<&mut [f32; D]> = batch_out.iter_mut().collect();
    let mut eval_scratch = [0.0f32; D];

    // Warmup.
    for _ in 0..10 {
        ensemble.eval_batch_into(&batch_x_refs, &mut batch_out_refs, &mut eval_scratch);
    }

    // Measure.
    const RUNS: usize = 100;
    let mut per_run_us = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t0 = Instant::now();
        ensemble.eval_batch_into(&batch_x_refs, &mut batch_out_refs, &mut eval_scratch);
        let dt = t0.elapsed();
        per_run_us.push(dt.as_micros() as u64);
    }
    let med = median_u64(&mut per_run_us);
    let med_dur = Duration::from_micros(med);

    println!("  eval_batch_into(N={N_BATCH}) p50: {med_dur:?}  (target ≤ {TARGET_BATCH_MS} ms)");

    let passed = med <= TARGET_BATCH_MS * 1000;
    GateResult {
        name: "G4 eval_batch_into latency",
        passed,
        detail: format!(
            "{} µs {} {} µs target",
            med,
            if passed { "≤" } else { ">" },
            TARGET_BATCH_MS * 1000
        ),
    }
}

fn gate_g3_zero_alloc() -> GateResult {
    println!("\n--- G3: zero-alloc hot path (re-verify in bench) ---");

    let mut ensemble = build_ensemble();
    let mut scratch = EnsembleFitScratch::<P, D>::new();
    let xs_fit: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut x = [0.0f32; D];
            for (k, slot) in x.iter_mut().enumerate().take(D) {
                *slot = ((i + k) as f32) * 0.01;
            }
            x
        })
        .collect();
    let ys_fit: Vec<[f32; D]> = (0..N_FIT_PAIRS)
        .map(|i| {
            let mut y = [0.0f32; D];
            for (k, slot) in y.iter_mut().enumerate().take(D) {
                *slot = ((i * 2 + k) as f32) * 0.005;
            }
            y
        })
        .collect();
    let x_refs: Vec<&[f32]> = xs_fit.iter().map(|v| &v[..]).collect();
    let y_refs: Vec<&[f32]> = ys_fit.iter().map(|v| &v[..]).collect();
    ensemble.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);

    let x = [0.5f32; D];
    let mut out = [0.0f32; D];
    let mut eval_scratch = [0.0f32; D];

    // Warmup.
    for _ in 0..100 {
        {
            ensemble.eval_into(&x, &mut out, &mut eval_scratch);
            black_box(());
        }
    }

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..1000 {
        {
            ensemble.eval_into(&x, &mut out, &mut eval_scratch);
            black_box(());
        }
    }
    let delta = ALLOC_COUNT.load(Ordering::Relaxed) - before;

    println!("  eval_into allocs/1000 calls: {delta}  (target: 0)");

    let passed = delta == 0;
    GateResult {
        name: "G3 zero-alloc eval_into",
        passed,
        detail: format!("{delta} allocs in 1000 calls (target 0)"),
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("==============================================================");
    println!("  Plan 376 Phase 3 — Velocity-Field Ensemble GOAT Gate (G3+G4)");
    println!("==============================================================");
    println!("  (G1 mechanics: unit tests in velocity_field_ensemble.rs)");
    println!("  (G2 cross-domain quality: bench_376_velocity_field_ensemble_poc.rs)");

    let gates = [
        gate_g3_zero_alloc(),
        gate_g4_fit_latency(),
        gate_g4_eval_latency(),
        gate_g4_batch_latency(),
    ];

    println!("\n=== Gate Verdicts ===");
    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    println!("G3 (no-regression combo check): verified via `cargo check --all-features`");
    println!("    and `cargo check --no-default-features` (the merkle_root lesson).");
    println!();

    if all_pass {
        println!("=== ALL G3+G4 GATES PASS ===");
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED ===");
        std::process::exit(1);
    }
}
