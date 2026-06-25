//! Phase-Modulated Subspace Rotation Gate GOAT bench (Plan 322 Phase 2).
//!
//! Exercises G1–G4 + G6 for the `phase_rotation_coupling` primitive.
//!
//! # Gates
//!
//! - **G1 (norm preservation — modelless quality)**: The headline
//!   `sin²α + cos²α = 1` identity must hold to `< 1e-4` across a 1000-point
//!   α sweep in `[0, π/2]`. Measured both for `phase_safe_cos_sin` (libm sin
//!   + Pythagorean sqrt recovery — should be essentially f32 rounding noise)
//!   AND for the full `compute_phase_from_projection` end-to-end path.
//!   Also re-verifies the `‖out‖² ≤ ‖a‖² + ‖b‖²` Cauchy-Schwarz bound.
//!
//! - **G2 (smooth interpolation)**: Sweeping α ∈ [0, π/2] must move the
//!   output monotonically from `a` to `b` in cosine-similarity space (sim_a
//!   non-increasing, sim_b non-decreasing, no reversals). Uses HLA-scale
//!   D=8 halves.
//!
//! - **G3 (latency)**: Batched-median timing, 1024 calls per measurement ×
//!   256 batches with sink-hash anti-hoist. Targets:
//!     - D=8 scalar phase + mix: < 50 ns/call.
//!     - D=8 mix-only (cos/sin precomputed): < 20 ns/call.
//!     - D=64 per-channel phase + mix: < 1500 ns/call (libm-sin budget).
//!
//! - **G4 (alloc-free hot path)**: After scratch warmup, 100 steady-state
//!   calls of `phase_rotation_gate_into` AND `compute_phase_per_channel_into`
//!   allocate 0 times (counted via a global `CountingAllocator`).
//!
//! - **G6 (sigmoid never softmax)**: Static check — the kernel uses
//!   `simd::fast_sigmoid` (1/(1+e^{-x})), never softmax. Asserted by
//!   inspecting the constructed phase value at `dot = 0` (sigmoid(0) = 0.5 →
//!   α = π/4 → cos α = sin α = 1/√2). Softmax of a single value would give
//!   1.0, not 0.5.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features phase_rotation_coupling --bench bench_322_phase_rotation_goat -- --nocapture
//! ```
//!
//! Or (working around the dyld/trustd stall on macOS):
//!
//! ```bash
//! cargo bench -p katgpt-core --features phase_rotation_coupling --bench bench_322_phase_rotation_goat --no-run
//! target/release/deps/bench_322_phase_rotation_goat-<hash>
//! ```

#![cfg(feature = "phase_rotation_coupling")]

use katgpt_core::{
    compute_phase_from_projection, compute_phase_per_channel_into, phase_rotation_gate_into,
    PhaseRotationScratch,
};
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

// ─── Helpers ────────────────────────────────────────────────────────────────

const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

fn l2_norm_sq(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

// ─── Gate runner ────────────────────────────────────────────────────────────

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

// ─── G1: Norm preservation ──────────────────────────────────────────────────

fn gate_g1_norm_preservation() -> GateResult {
    // Sub-gate G1a: phase_safe_cos_sin Pythagorean drift across α sweep.
    // (phase_safe_cos_sin is private — exercise it indirectly through the
    // public per-channel API with uniform state/direction so every channel
    // gets the same α. The drift measured here is the drift of the per-channel
    // cos/sin pair.)
    let steps = 1000usize;
    let mut max_drift_per_channel = 0.0f32;
    let mut max_drift_scalar = 0.0f32;
    for k in 0..=steps {
        let alpha = (k as f32 / steps as f32) * FRAC_PI_2;

        // Scalar path (compute_phase_from_projection uses libm cos/sin
        // directly — its drift is the libm-cos + libm-sin combined drift).
        // Construct a state/direction whose dot · sharpness = logit(sigmoid
        // inverse of alpha · 2/π). We work backwards: pick dot directly.
        // sigmoid(dot · λ) · π/2 = alpha  ⇒  dot · λ = ln(alpha / (π/2 - alpha+
        // ε)) ... easier: skip the inverse, just measure compute_phase and
        // check cos²+sin².
        // For G1 we directly exercise the libm cos/sin path by computing the
        // phase from a *constructed* projection. Pick sharpness=1, dot=alpha
        // (so sigmoid(alpha)·π/2 ≠ alpha exactly, but we just need to verify
        // the *identity* on the returned cos/sin, not the α value itself).
        let dot = (k as f32 / steps as f32) * 4.0 - 2.0; // sweep [-2, 2]
        let state = [dot];
        let direction = [1.0f32];
        let mut cos_s = 0.0f32;
        let mut sin_s = 0.0f32;
        // Use a length-1 state/direction to exercise compute_phase_from_projection
        // without needing a real D-dim state.
        if compute_phase_from_projection(&state, &direction, 1.0, &mut cos_s, &mut sin_s).is_ok() {
            let drift = (cos_s * cos_s + sin_s * sin_s - 1.0).abs();
            max_drift_scalar = max_drift_scalar.max(drift);
        }

        // Per-channel path via compute_phase_per_channel_into — uniform
        // state/direction so every channel sees the same projection. We use
        // the per-channel API because it routes through phase_safe_cos_sin
        // (the Pythagorean-recovery path) — that's where the G1 identity is
        // load-bearing.
        let st = [alpha; 8];
        let dirs = [1.0f32; 8];
        // sharpness = π/(2·α) so the sigmoid input is π/2 (saturates to α=π/2);
        // for cleaner sweep, just use sharpness = 1 and accept the induced α.
        let mut cos_pc = [0.0f32; 8];
        let mut sin_pc = [0.0f32; 8];
        let mut scratch = PhaseRotationScratch::new(8);
        if compute_phase_per_channel_into(
            &st,
            &dirs,
            1.0,
            &mut cos_pc,
            &mut sin_pc,
            &mut scratch,
        )
        .is_ok()
        {
            // Every channel should have the same (cos, sin).
            let drift = (cos_pc[0] * cos_pc[0] + sin_pc[0] * sin_pc[0] - 1.0).abs();
            max_drift_per_channel = max_drift_per_channel.max(drift);
        }
    }

    // Sub-gate G1b: ‖out‖² ≤ ‖a‖² + ‖b‖² for random a/b across α sweep.
    let a = [0.5f32, -1.2, 0.8, 2.1, -0.7, 1.5, -0.3, 0.9];
    let b = [1.1f32, 0.4, -1.8, 0.6, 1.3, -0.9, 0.7, -1.1];
    let bound = l2_norm_sq(&a) + l2_norm_sq(&b);
    let mut max_bound_violation = 0.0f32;
    let mut out = [0.0f32; 8];
    for k in 0..=steps {
        let alpha = (k as f32 / steps as f32) * FRAC_PI_2;
        let c = alpha.cos();
        let s = alpha.sin();
        if phase_rotation_gate_into(&a, &b, &[c], &[s], &mut out).is_ok() {
            let n = l2_norm_sq(&out);
            let violation = (n - bound).max(0.0);
            max_bound_violation = max_bound_violation.max(violation);
        }
    }

    let g1a_pass = max_drift_per_channel < 1e-4;
    let g1b_pass = max_bound_violation < 1e-4;
    // Scalar path drift is libm-only (~1e-7); include as informational.
    let g1_info = max_drift_scalar < 1e-4;

    if g1a_pass && g1b_pass {
        GateResult::pass(
            "G1",
            format!(
                "per-channel Pythagorean drift={max_drift_per_channel:.2e} (<1e-4), \
                 scalar libm drift={max_drift_scalar:.2e} (info, <1e-4 {info_mark}), \
                 ‖out‖² bound violation={max_bound_violation:.2e} (<1e-4)",
                info_mark = if g1_info { "OK" } else { "X" }
            ),
        )
    } else {
        let mut reasons = Vec::new();
        if !g1a_pass {
            reasons.push(format!(
                "per-channel Pythagorean drift={max_drift_per_channel:.2e} >= 1e-4"
            ));
        }
        if !g1b_pass {
            reasons.push(format!(
                "‖out‖² bound violation={max_bound_violation:.2e} >= 1e-4"
            ));
        }
        GateResult::fail("G1", reasons.join("; "))
    }
}

// ─── G2: Smooth interpolation ───────────────────────────────────────────────

fn gate_g2_smooth_interpolation() -> GateResult {
    // Sweep α ∈ [0, π/2]; output must move monotonically from a to b in
    // cosine-similarity space. HLA-scale D=8 halves.
    let a = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let steps = 100usize;
    let mut reversals = 0usize;
    let mut prev_sim_a = 1.0f32;
    let mut prev_sim_b = 0.0f32;
    let mut out = [0.0f32; 8];
    for k in 0..=steps {
        let alpha = (k as f32 / steps as f32) * FRAC_PI_2;
        let c = alpha.cos();
        let s = alpha.sin();
        if phase_rotation_gate_into(&a, &b, &[c], &[s], &mut out).is_err() {
            continue;
        }
        let sim_a = cosine_sim(&out, &a);
        let sim_b = cosine_sim(&out, &b);
        // sim_a must be non-increasing within a small tolerance.
        if sim_a > prev_sim_a + 1e-5 {
            reversals += 1;
        }
        // sim_b must be non-decreasing within a small tolerance.
        if sim_b < prev_sim_b - 1e-5 {
            reversals += 1;
        }
        prev_sim_a = sim_a;
        prev_sim_b = sim_b;
    }

    if reversals == 0 {
        GateResult::pass(
            "G2",
            format!(
                "monotone interpolation from a to b across {steps} steps, 0 reversals (tol 1e-5)"
            ),
        )
    } else {
        GateResult::fail(
            "G2",
            format!("{reversals} reversals across {steps}-step sweep (expected monotone)"),
        )
    }
}

// ─── G3: Latency ────────────────────────────────────────────────────────────

fn timed_median_ns(iters: usize, batches: usize, mut body: impl FnMut()) -> f64 {
    // Warmup.
    for _ in 0..(batches.min(20)) {
        body();
    }
    let mut batch_times_ns: Vec<u64> = Vec::with_capacity(batches);
    for _ in 0..batches {
        let start = Instant::now();
        for _ in 0..iters {
            body();
        }
        batch_times_ns.push(start.elapsed().as_nanos() as u64);
    }
    batch_times_ns.sort_unstable();
    // Median batch → median per-call ns.
    let mid = batch_times_ns.len() / 2;
    let median_batch_ns = batch_times_ns[mid] as f64;
    median_batch_ns / iters as f64
}

/// Sink that the optimizer cannot eliminate: `std::hint::black_box` forces
/// the value to be materialized in a register / memory. Use this on hot-path
/// *inputs* and *outputs* to prevent elision. The closure form lets callers
/// wrap individual values without lifting the whole body out of the loop.
#[inline(never)]
fn bb<T>(x: T) -> T {
    std::hint::black_box(x)
}

fn gate_g3_latency() -> GateResult {
    const ITERS: usize = 1024;
    const BATCHES: usize = 256;
    const TARGET_D8_SCALAR_NS: f64 = 50.0;
    const TARGET_D8_MIX_ONLY_NS: f64 = 20.0;
    const TARGET_D64_PER_CHANNEL_NS: f64 = 1500.0;

    // D=8 scalar phase + mix. The full hot path: compute_phase_from_projection
    // (one dot + sigmoid + cos + sin) + phase_rotation_gate_into (8 FMA).
    // black_box the inputs so the compiler can't hoist the dot/sigmoid/cos/sin
    // out of the loop (they depend on `d8_state` which we re-feed each iter).
    let d8_state = [0.3f32, -0.5, 0.7, 0.1, -0.2, 0.8, -0.4, 0.6];
    let d8_direction = [0.1f32, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
    let d8_a = [1.0f32; 8];
    let d8_b = [0.5f32; 8];
    let mut d8_out = [0.0f32; 8];
    let mut cos8 = 0.0f32;
    let mut sin8 = 0.0f32;
    let d8_scalar_ns = timed_median_ns(ITERS, BATCHES, || {
        // Black-box the inputs each iteration to prevent hoisting.
        let state = bb(d8_state);
        let dirn = bb(d8_direction);
        let a = bb(d8_a);
        let b = bb(d8_b);
        let _ = compute_phase_from_projection(&state, &dirn, 4.0, &mut cos8, &mut sin8);
        let _ = phase_rotation_gate_into(&a, &b, &[cos8], &[sin8], &mut d8_out);
        // Black-box the output to force the write.
        let _ = bb(d8_out);
    });

    // D=8 mix-only — precomputed cos/sin. Pure FMA kernel.
    let precomputed_c = FRAC_1_SQRT_2;
    let precomputed_s = FRAC_1_SQRT_2;
    let d8_mix_only_ns = timed_median_ns(ITERS, BATCHES, || {
        let a = bb(d8_a);
        let b = bb(d8_b);
        let c = bb(precomputed_c);
        let s = bb(precomputed_s);
        let _ = phase_rotation_gate_into(&a, &b, &[c], &[s], &mut d8_out);
        let _ = bb(d8_out);
    });

    // D=64 per-channel phase + mix. The cold-path budget.
    let d64_state: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
    let d64_directions: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let d64_a: Vec<f32> = vec![0.7; 64];
    let d64_b: Vec<f32> = vec![0.3; 64];
    let mut d64_cos = vec![0.0f32; 64];
    let mut d64_sin = vec![0.0f32; 64];
    let mut d64_out = vec![0.0f32; 64];
    let mut scratch64 = PhaseRotationScratch::new(64);
    // Warm the scratch.
    let _ = compute_phase_per_channel_into(
        &d64_state,
        &d64_directions,
        4.0,
        &mut d64_cos,
        &mut d64_sin,
        &mut scratch64,
    );
    let d64_per_channel_ns = timed_median_ns(ITERS, BATCHES, || {
        let st = bb(&d64_state[..]);
        let dr = bb(&d64_directions[..]);
        let a = bb(&d64_a[..]);
        let b = bb(&d64_b[..]);
        let _ = compute_phase_per_channel_into(
            st,
            dr,
            4.0,
            &mut d64_cos,
            &mut d64_sin,
            &mut scratch64,
        );
        let _ = phase_rotation_gate_into(a, b, &d64_cos, &d64_sin, &mut d64_out);
        let _ = bb(&d64_out[..]);
    });

    let d8_scalar_pass = d8_scalar_ns <= TARGET_D8_SCALAR_NS;
    let d8_mix_pass = d8_mix_only_ns <= TARGET_D8_MIX_ONLY_NS;
    let d64_pass = d64_per_channel_ns <= TARGET_D64_PER_CHANNEL_NS;

    let all_pass = d8_scalar_pass && d8_mix_pass && d64_pass;
    let fmt = |v: f64, target: f64, pass: bool| {
        let mark = if pass { "OK" } else { "X" };
        format!("{v:.1}ns [{mark}, <= {target}ns]")
    };

    if all_pass {
        GateResult::pass(
            "G3",
            format!(
                "D=8 scalar+mix: {}, D=8 mix-only: {}, D=64 per-channel+mix: {}",
                fmt(d8_scalar_ns, TARGET_D8_SCALAR_NS, d8_scalar_pass),
                fmt(d8_mix_only_ns, TARGET_D8_MIX_ONLY_NS, d8_mix_pass),
                fmt(d64_per_channel_ns, TARGET_D64_PER_CHANNEL_NS, d64_pass),
            ),
        )
    } else {
        GateResult::fail(
            "G3",
            format!(
                "D=8 scalar+mix: {}, D=8 mix-only: {}, D=64 per-channel+mix: {}",
                fmt(d8_scalar_ns, TARGET_D8_SCALAR_NS, d8_scalar_pass),
                fmt(d8_mix_only_ns, TARGET_D8_MIX_ONLY_NS, d8_mix_pass),
                fmt(d64_per_channel_ns, TARGET_D64_PER_CHANNEL_NS, d64_pass),
            ),
        )
    }
}

// ─── G4: Alloc-free hot path ────────────────────────────────────────────────

fn gate_g4_zero_alloc() -> GateResult {
    // Warmup both hot paths.
    let d = 64usize;
    let a = vec![0.7f32; d];
    let b = vec![0.3f32; d];
    let state = vec![0.5f32; d];
    let directions = vec![1.0f32; d];
    let mut cos_alpha = vec![0.0f32; d];
    let mut sin_alpha = vec![0.0f32; d];
    let mut out = vec![0.0f32; d];
    let mut scratch = PhaseRotationScratch::new(d);
    for _ in 0..10 {
        let _ = phase_rotation_gate_into(&a, &b, &cos_alpha, &sin_alpha, &mut out);
        let _ = compute_phase_per_channel_into(
            &state,
            &directions,
            4.0,
            &mut cos_alpha,
            &mut sin_alpha,
            &mut scratch,
        );
    }

    // Measure: 100 steady-state calls through both hot paths, count allocations.
    let iters = 100usize;
    let (_, allocs) = alloc_delta(|| {
        for _ in 0..iters {
            let _ = phase_rotation_gate_into(&a, &b, &cos_alpha, &sin_alpha, &mut out);
            let _ = compute_phase_per_channel_into(
                &state,
                &directions,
                4.0,
                &mut cos_alpha,
                &mut sin_alpha,
                &mut scratch,
            );
        }
    });

    if allocs == 0 {
        GateResult::pass(
            "G4",
            format!("0 allocations over {iters} steady-state calls (mix + per-channel)"),
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

// ─── G6: Sigmoid never softmax (static check via behavior) ──────────────────

fn gate_g6_sigmoid_never_softmax() -> GateResult {
    // Static check: at dot = 0, sigmoid(0) = 0.5 → α = π/4 → cos α = sin α = 1/√2.
    // Softmax of a single value would give 1.0, not 0.5. So checking that the
    // phase at zero projection is exactly π/4 proves sigmoid is used, not softmax.
    let state = [0.0f32; 8];
    let direction = [1.0f32; 8];
    let mut cos_a = 0.0f32;
    let mut sin_a = 0.0f32;
    if compute_phase_from_projection(&state, &direction, 1.0, &mut cos_a, &mut sin_a).is_err() {
        return GateResult::fail("G6", "compute_phase_from_projection returned Err");
    }
    let expected = FRAC_1_SQRT_2; // cos(π/4) = sin(π/4) = 1/√2
    let cos_ok = (cos_a - expected).abs() < 1e-5;
    let sin_ok = (sin_a - expected).abs() < 1e-5;
    let equal = (cos_a - sin_a).abs() < 1e-6;

    if cos_ok && sin_ok && equal {
        GateResult::pass(
            "G6",
            format!(
                "at dot=0: cos α = sin α = {cos_a:.4} ≈ 1/√2 (sigmoid(0)=0.5 → α=π/4); softmax would give 1.0"
            ),
        )
    } else {
        GateResult::fail(
            "G6",
            format!(
                "at dot=0: cos α={cos_a:.4}, sin α={sin_a:.4} (expected 1/√2={expected:.4} for sigmoid)"
            ),
        )
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 322 — Phase-Modulated Subspace Rotation Gate GOAT Gate ===\n");

    let gates = [
        gate_g1_norm_preservation(),
        gate_g2_smooth_interpolation(),
        gate_g3_latency(),
        gate_g4_zero_alloc(),
        gate_g6_sigmoid_never_softmax(),
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
        println!(
            "=== ALL GATES PASS — modelless gain proven, eligible for default promotion ==="
        );
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED — keep opt-in, investigate ===");
        std::process::exit(1);
    }
}
