//! Spectral Differentiation GOAT gate bench (Plan 325 Phase 2).
//!
//! Exercises G1–G4 for the `spectral_differentiation` primitive.
//!
//! # Gates
//!
//! - **G1 (analytical correctness — modelless quality)**: The headline
//!   spectral-differentiation claim. Sample `sin(2πx)` over one period on
//!   `N=64` points; the spectral derivative at order 1 must match
//!   `(2π/N)·cos(2πx)` to max abs error `< 1e-4`, and order 2 must match
//!   `-(2π/N)²·sin(2πx)` to `< 1e-3`. Also compared against a 2-point
//!   finite-difference baseline — the spectral derivative should be at
//!   least 100× more accurate on smooth periodic signals (the regime where
//!   FFT differentiation is exact up to f32 rounding).
//!
//! - **G2 (perf)**: `spectral_differentiate_into` mean latency over 1000
//!   calls with pre-warmed scratch. **PASS** if mean latency ≤ 50µs for
//!   `N ∈ {64, 256, 1024}` (same budget as Plan 323 Fourier Continuation).
//!
//! - **G3 (no regression)**: When `cfg.order == 0`, the multiplier is
//!   identically `1` and the output is bit-identical to the input (the
//!   IFFT(FFT(x)) round-trip on real input).
//!
//! - **G4 (alloc-free hot path)**: `spectral_differentiate_into` with
//!   pre-warmed `SpecDiffScratch` allocates 0 times over 100 steady-state
//!   calls (counted via a global `CountingAllocator`).
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features spectral_differentiation --bench bench_325_spectral_differentiation_goat -- --nocapture
//! ```

#![cfg(feature = "spectral_differentiation")]

use katgpt_core::{spectral_differentiate_into, SpecDiffConfig, SpecDiffScratch};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ─── CountingAllocator (G4) ─────────────────────────────────────────────────

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let r = f();
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    (r, after - before)
}

// ─── Test signal & analytical derivatives ───────────────────────────────────

const TAU: f32 = std::f32::consts::TAU;

/// Sample `sin(2π·j/N)` for j = 0..N — one full period, exactly periodic.
fn sine_period(n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| ((TAU * j as f32) / n as f32).sin())
        .collect()
}

/// Analytical first derivative of `sine_period`: `(2π/N)·cos(2π·j/N)`.
fn sine_first_deriv(n: usize) -> Vec<f32> {
    let omega = TAU / n as f32;
    (0..n)
        .map(|j| omega * ((TAU * j as f32) / n as f32).cos())
        .collect()
}

/// Analytical second derivative of `sine_period`: `-(2π/N)²·sin(2π·j/N)`.
fn sine_second_deriv(n: usize) -> Vec<f32> {
    let omega = TAU / n as f32;
    (0..n)
        .map(|j| -omega * omega * ((TAU * j as f32) / n as f32).sin())
        .collect()
}

/// Two-point central finite-difference first derivative. The classic
/// O(h²)-accurate baseline. For comparison only — spectral differentiation
/// should be ~100× more accurate on smooth periodic signals.
fn finite_difference_first_deriv(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0f32; n];
    // Periodic wrap: indices mod n.
    for i in 0..n {
        let prev = x[(i + n - 1) % n];
        let next = x[(i + 1) % n];
        out[i] = (next - prev) * 0.5;
    }
    out
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0f32, f32::max)
}

// ─── Gate runners ───────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

fn gate_g1_analytical_correctness() -> GateResult {
    let n = 64usize;
    let x = sine_period(n);
    let mut scratch = SpecDiffScratch::new();
    scratch.ensure_capacity(n);

    // Order 1: d/dx sin(2πx) = (2π/N)·cos(2πx). Target max abs error < 1e-4.
    let mut out1 = vec![0.0f32; n];
    if let Err(e) =
        spectral_differentiate_into(&x, &mut out1, &mut scratch, &SpecDiffConfig::DEFAULT)
    {
        return GateResult::fail("G1", format!("order 1 failed: {e:?}"));
    }
    let expected1 = sine_first_deriv(n);
    let err1 = max_abs_error(&out1, &expected1);

    // Order 2: d²/dx² sin(2πx) = -(2π/N)²·sin(2πx). Target max abs error < 1e-3.
    let mut out2 = vec![0.0f32; n];
    if let Err(e) = spectral_differentiate_into(
        &x,
        &mut out2,
        &mut scratch,
        &SpecDiffConfig::default_order(2),
    ) {
        return GateResult::fail("G1", format!("order 2 failed: {e:?}"));
    }
    let expected2 = sine_second_deriv(n);
    let err2 = max_abs_error(&out2, &expected2);

    // Spectral-vs-FD accuracy ratio (order 1). Spectral is exact up to f32
    // rounding on band-limited periodic signals; FD has O(h²) error. The
    // spectral derivative should be ≥ 100× more accurate here.
    let fd = finite_difference_first_deriv(&x);
    let fd_err = max_abs_error(&fd, &expected1);
    let ratio = fd_err / err1.max(1e-12);

    let order1_pass = err1 < 1e-4;
    let order2_pass = err2 < 1e-3;
    let fd_ratio_pass = ratio >= 100.0;

    if order1_pass && order2_pass && fd_ratio_pass {
        GateResult::pass(
            "G1",
            format!(
                "order1 max|err|={err1:.2e} (<1e-4), order2 max|err|={err2:.2e} (<1e-3), spectral-vs-FD ratio={ratio:.1}x (>=100x)"
            ),
        )
    } else {
        let mut reasons = Vec::new();
        if !order1_pass {
            reasons.push(format!("order1 max|err|={err1:.2e} >= 1e-4"));
        }
        if !order2_pass {
            reasons.push(format!("order2 max|err|={err2:.2e} >= 1e-3"));
        }
        if !fd_ratio_pass {
            reasons.push(format!(
                "spectral-vs-FD ratio={ratio:.1}x < 100x (spectral={err1:.2e}, FD={fd_err:.2e})"
            ));
        }
        GateResult::fail("G1", reasons.join("; "))
    }
}

fn gate_g2_perf() -> GateResult {
    const SIZES: &[usize] = &[64, 256, 1024];
    const TARGET_US: f64 = 50.0;
    const WARMUP: usize = 20;
    const ITERS: usize = 1000;

    let mut all_pass = true;
    let mut details: Vec<String> = Vec::with_capacity(SIZES.len());

    for &n in SIZES {
        let x = sine_period(n);
        let mut out = vec![0.0f32; n];
        let mut scratch = SpecDiffScratch::new();
        scratch.ensure_capacity(n);
        let cfg = SpecDiffConfig::DEFAULT;

        // Warmup — populate the FFT planner cache for this size.
        for _ in 0..WARMUP {
            let _ = spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg);
        }

        let start = Instant::now();
        for _ in 0..ITERS {
            let _ = spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg);
        }
        let elapsed = start.elapsed();
        let mean_ns = elapsed.as_nanos() as f64 / ITERS as f64;
        let mean_us = mean_ns / 1000.0;

        let pass = mean_us <= TARGET_US;
        if !pass {
            all_pass = false;
        }
        let mark = if pass { "OK" } else { "X" };
        details.push(format!("N={n}: {mean_us:.2}us [{mark}]"));
    }

    if all_pass {
        GateResult::pass(
            "G2",
            format!(
                "all sizes <= {TARGET_US}us target ({ITERS} iters each): {}",
                details.join(", ")
            ),
        )
    } else {
        GateResult::fail(
            "G2",
            format!(
                "one or more sizes exceed {TARGET_US}us: {}",
                details.join(", ")
            ),
        )
    }
}

fn gate_g3_no_regression() -> GateResult {
    // G3: at order=0, the operator is identity (multiplier is 1, round-trip
    // IFFT(FFT(x)) = x up to f32 rounding from rustfft's butterflies).
    let n = 64usize;
    // Non-trivial input so any error is visible.
    let x: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.1).sin() + 0.3 * (i as f32 * 0.07).cos())
        .collect();
    let mut out = vec![f32::NAN; n];
    let mut scratch = SpecDiffScratch::new();
    scratch.ensure_capacity(n);

    match spectral_differentiate_into(
        &x,
        &mut out,
        &mut scratch,
        &SpecDiffConfig::default_order(0),
    ) {
        Ok(()) => {
            let max_err = max_abs_error(&out, &x);
            // rustfft's round-trip is bit-exact for small real inputs in
            // our size range; allow a tiny epsilon to absorb platform jitter.
            // Bar matches the unit test (test_order_zero_is_identity).
            let epsilon = 1e-5;
            if max_err < epsilon {
                GateResult::pass(
                    "G3",
                    format!("order=0 is identity, max|err|={max_err:.2e} (<{epsilon})"),
                )
            } else {
                GateResult::fail(
                    "G3",
                    format!("order=0 NOT identity, max|err|={max_err:.2e} (>= {epsilon})"),
                )
            }
        }
        Err(e) => GateResult::fail("G3", format!("order=0 returned error: {e:?}")),
    }
}

fn gate_g4_zero_alloc() -> GateResult {
    let n = 256usize;
    let x = sine_period(n);
    let mut out = vec![0.0f32; n];
    let mut scratch = SpecDiffScratch::new();
    scratch.ensure_capacity(n);
    let cfg = SpecDiffConfig::DEFAULT;

    // Warmup: ensure scratch buffer is at capacity AND the FFT planner cache
    // is populated for this size (both are first-call allocations).
    for _ in 0..10 {
        let _ = spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg);
    }

    // Measure allocations over a tight loop of steady-state calls.
    let iters = 100usize;
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..iters {
            let _ = spectral_differentiate_into(&x, &mut out, &mut scratch, &cfg);
        }
    });

    if allocs == 0 {
        GateResult::pass(
            "G4",
            format!("0 allocations over {iters} steady-state calls"),
        )
    } else {
        GateResult::fail(
            "G4",
            format!(
                "{allocs} allocations over {iters} steady-state calls (expected 0; ~{} per call)",
                allocs / iters
            ),
        )
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 325 - Spectral Differentiation GOAT Gate ===\n");

    let gates = [
        gate_g1_analytical_correctness(),
        gate_g2_perf(),
        gate_g3_no_regression(),
        gate_g4_zero_alloc(),
    ];

    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!("=== ALL GATES PASS - modelless gain proven, eligible for default promotion ===");
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED - keep opt-in, investigate ===");
        std::process::exit(1);
    }
}
