//! Plan 432 Phase 2 T2.4 — VFD latency GOAT gate G4 bench.
//!
//! Measures `vfd_score_into` latency at the paper-default config:
//! - M=2 members, D=8 (HLA dim), N_s=10 ODE steps, B=5 Monte Carlo batch.
//!
//! Target: ≤ 50 µs per call (plasma-tier budget per Plan 432 T2.4).
//! Failure does NOT block ship — it constrains the deployment regime.
//!
//! Also re-verifies G3 (zero-alloc) on the same config via the
//! `CountingAllocator` (mirrors bench_376).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/vfd_432 cargo build --release -p katgpt-core \
//!     --features velocity_field_disagreement --bench bench_432_vfd_goat
//! /tmp/vfd_432/release/deps/bench_432_vfd_goat-* --nocapture
//! ```

use katgpt_core::velocity_field_disagreement::{VfdScratch, vfd_score_into};
use katgpt_core::velocity_field_ensemble::{ClosureField, Schedule, VelocityField};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Config ────────────────────────────────────────────────────────────────

/// D=8: the HLA velocity-field output dimension (the production target).
const D: usize = 8;
/// M=2: paper default (sufficient per §6.2).
const M: usize = 2;
/// N_s=10: paper default ODE step count.
const N_STEPS: usize = 10;
/// B=5: paper default Monte Carlo batch size.
const BATCH: usize = 5;

/// Latency target: ≤ 50 µs per `vfd_score_into` call.
const TARGET_US: u64 = 50;

const WARMUP_ITERS: usize = 1000;
const MEASURE_ITERS: usize = 10_000;

// ── Linear fields (two distinct named fns; same type for [F; M]) ──────────

fn field_0(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[k] * 0.1 + 0.01;
    }
}
fn field_1(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[k] * 0.15 - 0.02;
    }
}

type FieldFn = fn(&[f32], &mut [f32; D]);

fn build_fields() -> [ClosureField<D, FieldFn>; M] {
    [
        ClosureField::<D, FieldFn>::new(0, field_0),
        ClosureField::<D, FieldFn>::new(1, field_1),
    ]
}

// ── Deterministic RNG (xorshift32 — fast, no allocations) ────────────────

struct FastRng {
    state: u32,
}

impl FastRng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
    /// Box-Muller standard normal.
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_uniform().max(f32::MIN_POSITIVE);
        let u2 = self.next_uniform();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
    fn next_uniform(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        (x >> 8) as f32 / ((1u32 << 24) as f32)
    }
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

fn gate_g4_vfd_latency() -> bool {
    println!("\n=== G4: vfd_score_into latency (M={M}, D={D}, N_s={N_STEPS}, B={BATCH}) ===");
    println!("Target: ≤ {TARGET_US} µs per call");

    let fields = build_fields();
    let field_refs: [&dyn VelocityField<D>; M] = [&fields[0], &fields[1]];
    let mut scratch: VfdScratch<M, D> = VfdScratch::new();
    let mut rng = FastRng::new(42);

    // Warmup.
    for _ in 0..WARMUP_ITERS {
        let _ = black_box(vfd_score_into(
            black_box(&field_refs),
            black_box(Schedule::Linear),
            black_box(N_STEPS),
            black_box(BATCH),
            black_box(&mut scratch),
            black_box(&mut || rng.next_normal()),
        ));
    }

    // Measure: batch timing for sub-µs resolution.
    const BATCH_ITERS: usize = 50;
    let outer = MEASURE_ITERS / BATCH_ITERS;
    let mut per_call_ns: Vec<u64> = Vec::with_capacity(outer);
    for _ in 0..outer {
        let t0 = Instant::now();
        for _ in 0..BATCH_ITERS {
            let _ = black_box(vfd_score_into(
                &field_refs,
                Schedule::Linear,
                N_STEPS,
                BATCH,
                &mut scratch,
                &mut || rng.next_normal(),
            ));
        }
        let dt = t0.elapsed();
        per_call_ns.push((dt.as_nanos() as u64) / (BATCH_ITERS as u64));
    }
    let med_ns = median_u64(&mut per_call_ns);
    let med_us = med_ns as f64 / 1000.0;

    println!("  vfd_score_into p50: {med_ns} ns ({med_us:.2} µs)");
    println!("  samples: {outer} (each = {BATCH_ITERS} calls)");

    let passed = med_ns <= TARGET_US * 1000;
    if passed {
        println!("  ✅ PASS: {med_us:.2} µs ≤ {TARGET_US} µs target");
    } else {
        println!("  ❌ FAIL: {med_us:.2} µs > {TARGET_US} µs target");
        println!(
            "  (Failure does NOT block ship — constrains deployment regime. See Plan 432 Risk Note #2.)"
        );
    }
    passed
}

fn gate_g3_zero_alloc() -> bool {
    println!("\n=== G3 (re-verify): vfd_score_into zero-alloc on score path ===");

    let fields = build_fields();
    let field_refs: [&dyn VelocityField<D>; M] = [&fields[0], &fields[1]];
    let mut scratch: VfdScratch<M, D> = VfdScratch::new();
    let mut rng = FastRng::new(7);

    // Reset counters after construction.
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);

    // Run a batch of VFD score calls.
    for _ in 0..1000 {
        let _ = black_box(vfd_score_into(
            &field_refs,
            Schedule::Linear,
            N_STEPS,
            BATCH,
            &mut scratch,
            &mut || rng.next_normal(),
        ));
    }

    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed);
    println!("  allocs:   {allocs}");
    println!("  deallocs: {deallocs}");

    let passed = allocs == 0 && deallocs == 0;
    if passed {
        println!("  ✅ PASS: zero allocations on the score path");
    } else {
        println!("  ❌ FAIL: {allocs} allocs detected (expected 0)");
    }
    passed
}

fn main() {
    let g4 = gate_g4_vfd_latency();
    let g3 = gate_g3_zero_alloc();

    println!("\n=== Plan 432 Phase 2 G3/G4 Summary ===");
    println!(
        "G3 (zero-alloc):    {}",
        if g3 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "G4 (≤{TARGET_US} µs):   {}",
        if g4 {
            "✅ PASS"
        } else {
            "❌ FAIL (non-blocking)"
        }
    );
}
