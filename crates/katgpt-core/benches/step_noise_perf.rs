//! Issue 975 T4 — the `step_noise` perf arm: the pre-registered bar is
//! **≤ 100 ns per `observe()` at N = 8** (`--release`, macOS/arm64), with
//! the **G4 zero-steady-state-alloc canary** riding the same binary.
//!
//! Groups:
//! - **g2_observe_latency** — criterion over `StepNoiseMonitor::<8>::observe`
//!   on a live monitor (warm, steady state) fed a deterministic walk; the
//!   pre-registered bar is asserted (an over-budget regression FAILS the
//!   bench, never just prints — the loud-zero discipline: absent work or a
//!   10× regression both red here).
//! - **g4_alloc_canary** — the counting allocator over a steady-state
//!   `observe()` loop: ZERO allocations after warmup (the module is fixed
//!   `[f32; N]` arrays + scalar EMAs by construction; this canary proves
//!   the construction).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features step_noise --bench step_noise_perf
//! ```
//!
//! Box state beside every number: `riir-reflex/scripts/bench_preflight.sh`
//! (the PROVENANCE line is quoted in the Issue-975 T4 record; battery /
//! power-mode / load refusals are the box's own gate).

#![cfg(feature = "step_noise")]

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

use std::sync::atomic::Ordering;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::step_noise::{StepNoiseConfig, StepNoiseMonitor};

/// The T4 monitor dimension (the pre-registered bar's N).
const N: usize = 8;
/// The pre-registered per-observe bar (ns) at N = 8, release.
const BAR_NS: f64 = 100.0;

/// Deterministic LCG (no global RNG — the unseeded-global-RNG gate).
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

/// A live, warmed monitor + its deterministic input walk (rebuilt per
/// measurement so the measured loop is pure `observe`).
fn bench_setup() -> (StepNoiseMonitor<N>, Vec<([f32; N], f32)>) {
    let mut rng = Lcg(0x5E70_5E17);
    let mut state = [0.0_f32; N];
    for x in state.iter_mut() {
        *x = rng.next_f32() * 0.1;
    }
    let mut inputs = Vec::with_capacity(2048);
    let mut obj = 0.5_f32;
    for _ in 0..2048 {
        for (j, x) in state.iter_mut().enumerate() {
            // Mixed regime walk: drift + direction churn keeps every EMA
            // branch live (churn, slope, dispersion rotation, divergence).
            *x += 0.02 * ((rng.next_f32() * 2.0) - 0.5) + 0.01 * (j as f32).sin();
        }
        obj += 0.001 * (rng.next_f32() - 0.4);
        inputs.push((state, obj));
    }
    let mut mon = StepNoiseMonitor::<N>::new(StepNoiseConfig::DEFAULT);
    for (s, o) in &inputs {
        mon.observe(s, *o);
    }
    (mon, inputs)
}

fn g2_observe_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_noise_g2_observe_latency");
    // Warmup + measurement over the SAME live monitor: steady state only.
    let (mut mon, inputs) = bench_setup();
    group.bench_function("observe_n8_warm", |b| {
        b.iter(|| {
            let mut acc = 0.0_f32;
            for (s, o) in &inputs {
                let read = mon.observe(black_box(s), black_box(*o));
                acc += read.churn + read.progress_slope;
            }
            black_box(acc)
        })
    });
    drop(inputs);

    // The pre-registered bar, asserted on a fresh calibration run (mean of
    // the loop above, measured here independently of criterion's report —
    // a regression FAILS the bench).
    let (mut mon, inputs) = bench_setup();
    let mut idx2 = 0_usize;
    let iters = 200_000_usize;
    let t0 = std::time::Instant::now();
    let mut acc = 0.0_f32;
    while idx2 < iters {
        let (s, o) = &inputs[idx2 & 2047];
        let read = mon.observe(black_box(s), black_box(*o));
        acc += read.churn;
        idx2 += 1;
    }
    let elapsed = t0.elapsed();
    black_box(acc);
    let ns_per = elapsed.as_nanos() as f64 / iters as f64;
    println!(
        "[step_noise][t4] observe() N={N}: {ns_per:.1} ns/op over {iters} iters (bar ≤ {BAR_NS:.0} ns)"
    );
    assert!(
        ns_per <= BAR_NS,
        "T4 bar FAILED: {ns_per:.1} ns/observe > {BAR_NS:.0} ns — the zero-alloc O(N) contract regressed"
    );
    group.finish();
}

fn g4_alloc_canary(c: &mut Criterion) {
    // No criterion row: the canary's verdict is the assertion below (it
    // runs at every bench invocation, criterion sampling can't drop it).
    let _ = c;
    let (mut mon, inputs) = bench_setup();
    // Warmup OUTSIDE the measured window (settle any lazy state).
    for (s, o) in &inputs {
        mon.observe(s, *o);
    }
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut acc = 0.0_f32;
    for _ in 0..10_000 {
        for (s, o) in &inputs {
            let read = mon.observe(s, *o);
            acc += read.churn + read.ddr;
        }
    }
    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    black_box(acc);
    let delta = alloc_after - alloc_before;
    assert_eq!(
        delta, 0,
        "G4 FAILED: {delta} allocations in the steady-state observe() loop — the fixed-array contract broke"
    );
    println!("[step_noise][t4] G4: 0 allocations over 20,480 steady-state observe() calls");
}

criterion_group!(benches, g2_observe_latency, g4_alloc_canary);
criterion_main!(benches);
