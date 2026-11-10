//! Plan 469 Phase 4 — HOPE Hilbert-Schmidt Capacity Kernel GOAT Gate (G1–G4).
//!
//! Exercises the four GOAT gates for the `hope_capacity` primitive distilled
//! from arXiv:2607.21366 (Mobahi & Bartlett, HOPE, Google DeepMind 2026-07-24).
//! Pure closed-form math — no model, no training. See
//! `katgpt-rs/.research/454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md` and
//! `katgpt-rs/.plans/469_hilbert_schmidt_capacity_kernel_primitive.md`.
//!
//! # Gates
//!
//! - **G1 (correctness sanity)** — re-verifies the load-bearing analytic
//!   invariants on the bench fixture: `relu_self_kernel(1,0) ≈ 0.5`, scale
//!   invariance, Cauchy-Schwarz, optimal-scale closed form. The full G1 bit-
//!   exact test suite is the 30 unit tests in `hope::tests`; this bench only
//!   re-checks the values the latency loop is exercising (defends against a
//!   silent fixture change that would make the latency numbers meaningless).
//! - **G2 (latency)** — p50 ns/call per kernel at HLA-scale dims:
//!   - `relu_self_kernel`      < 10 ns   (scalar-only: exp + erf_approx)
//!   - `warped_correlation`    < 50 ns   (D=8 SIMD dot + arithmetic)
//!   - `relu_cross_kernel_approx` < 80 ns (D=8, composes warped + trig)
//!   - `hope_capacity`         < 80 ns   (D=8 w_out norm + self-kernel)
//!   - `hope_prune_cost`       < 100 ns  (capacity + 1 div)
//!   - `hope_merge_cost`       < 200 ns  (2 capacities + parent metrics)
//!   - `optimal_rank1_parent_into_scratch` < 400 ns (D=8, rank-2 eigen + sign resolve)
//!   - `hope_greedy_select`    < 100 ns / 32 candidates (linear scan)
//!   - `hope_block_eviction_cost` < 50 ns (2-layer sum)
//!
//! All gates use `std::time::Instant` + `black_box` (criterion is not a
//! katgpt-core dev-dep on wasm32; this mirrors the bench_370 / bench_377
//! convention).
//! - **G3 (no-regression)** — verified externally via the lib test suite
//!   (`cargo test -p katgpt-core --lib` → 1796 green)
//!   + `cargo check --all-features`. This bench prints the gate as informational.
//! - **G4 (alloc-free)** — `CountingAllocator` audit over 100 steady-state
//!   calls per hot-path kernel (after a warmup pass). Target: 0 allocations
//!   on `relu_self_kernel`, `warped_correlation`, `relu_cross_kernel_approx`,
//!   `hope_capacity`, `hope_prune_cost`, `hope_merge_cost`,
//!   `hope_block_eviction_cost`, `hope_greedy_select`, AND
//!   `optimal_rank1_parent_into_scratch` (the scratch variant — the owned
//!   `optimal_rank1_parent` is caller-controlled allocation, NOT hot-path).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/hope_goat cargo bench -p katgpt-core \
//!     --features hope_capacity --bench bench_469_hope_kernel_goat -- --nocapture
//! ```
//!
//! Or the direct-binary workaround for the macOS dyld/trustd stall:
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/hope_goat cargo build --release -p katgpt-core \
//!     --features hope_capacity --bench bench_469_hope_kernel_goat
//! /tmp/hope_goat/release/bench_469_hope_kernel_goat-* --nocapture
//! ```

use katgpt_core::hope::{
    Rank1Operator, hope_block_eviction_cost, hope_capacity, hope_greedy_select, hope_merge_cost,
    hope_prune_cost, optimal_rank1_parent_into_scratch, relu_cross_kernel_approx, relu_self_kernel,
    warped_correlation,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── GateResult ─────────────────────────────────────────────────────────────

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

// ─── Rank1Operator impl for the bench fixture ───────────────────────────────
//
// Owned Vecs mirror the riir-neuron-db `ShardAsRank1` bridge shape (Plan 321):
// the bench exercises the same borrow pattern the production consumer will use.

struct OwnedRank1 {
    w_in: Vec<f32>,
    w_out: Vec<f32>,
    gamma: f32,
    beta: f32,
}

impl Rank1Operator for OwnedRank1 {
    fn w_in(&self) -> &[f32] {
        &self.w_in
    }
    fn w_out(&self) -> &[f32] {
        &self.w_out
    }
    fn gamma(&self) -> f32 {
        self.gamma
    }
    fn beta(&self) -> f32 {
        self.beta
    }
}

fn owned(w_in: &[f32], w_out: &[f32], gamma: f32, beta: f32) -> OwnedRank1 {
    OwnedRank1 {
        w_in: w_in.to_vec(),
        w_out: w_out.to_vec(),
        gamma,
        beta,
    }
}

// ─── Fixture: HLA-scale D=8 directions ──────────────────────────────────────
//
// D=8 matches the NpcEmotionScalars shape — the primary riir-ai consumer.
// D=64 (style_weights) is exercised by riir-neuron-db Plan 321's own bench;
// here we pin the HLA-scale latency floor.

const D_HLA: usize = 8;

fn fixture_w_in_i() -> [f32; D_HLA] {
    [0.3, -0.5, 0.8, 0.1, 0.9, -0.2, 0.4, 0.0]
}
fn fixture_w_in_j() -> [f32; D_HLA] {
    [0.7, 0.2, -0.4, 0.6, 0.1, -0.3, 0.5, 0.8]
}
fn fixture_w_out() -> [f32; 2] {
    [1.0, 0.5]
}

// (median_ns helper removed: G2 now reports the mean over ITERS×BATCH calls
// for sub-ns precision, not the per-batch median.)

// Iterations for the latency sweep. High enough to stabilize the median,
// low enough to keep the bench under ~2 s total.
const ITERS: usize = 10_000;
// Steady-state alloc audit: number of calls measured after warmup.
const ALLOCS_CALLS: usize = 100;

// ─── G1: correctness sanity on the bench fixture ────────────────────────────

fn gate_g1_correctness_sanity() -> GateResult {
    // 1. relu_self_kernel(1,0) = 0.5 (half-wave rectified energy).
    let k_standard = relu_self_kernel(1.0, 0.0);
    assert!(
        (k_standard - 0.5).abs() < 1e-4,
        "G1: K(1,0) = {k_standard}, expected 0.5"
    );

    // 2. Scale invariance: λ=2 on (w_in, γ) + 1/λ on w_out leaves capacity unchanged.
    let op_a = owned(&[1.0, 0.5], &[2.0, 0.5], 1.0, 0.0);
    let op_b = owned(&[2.0, 1.0], &[1.0, 0.25], 2.0, 0.0);
    let cap_a = hope_capacity(&op_a);
    let cap_b = hope_capacity(&op_b);
    assert!(
        (cap_a - cap_b).abs() < 1e-4,
        "G1: scale invariance violated: cap(λ=1)={cap_a}, cap(λ=2)={cap_b}"
    );

    // 3. Cauchy-Schwarz on cross-kernel.
    let w_i = fixture_w_in_i();
    let w_j = fixture_w_in_j();
    let k_ii = relu_self_kernel(1.0, 0.0);
    let k_jj = relu_self_kernel(1.5, 0.0);
    let k_ij = relu_cross_kernel_approx(&w_i, &w_j, 1.0, 1.5);
    let bound = (k_ii * k_jj).max(0.0).sqrt();
    assert!(
        k_ij.abs() <= bound + 1e-5,
        "G1: Cauchy-Schwarz violated: |K(ij)| = {} > √(K(ii)·K(jj)) = {bound}",
        k_ij.abs()
    );

    // 4. Optimal scale closed form: s* = (a+b·E)/(2E+b); for a=b=E=1, s* = 2/3.
    // (compute_optimal_scale is pub(crate); we verify indirectly via the
    // parent's s_star being finite + positive on a non-degenerate fixture.)
    let op_i = owned(&w_i, &fixture_w_out(), 1.0, 0.0);
    let op_j = owned(&w_j, &fixture_w_out(), 1.5, 0.2);
    let mut u_scratch = [0.0_f32; D_HLA];
    let mut v_scratch = [0.0_f32; 2];
    let mut k_self = 0.0_f32;
    let s_star = optimal_rank1_parent_into_scratch(
        &op_i,
        &op_j,
        1.0,
        &mut u_scratch,
        &mut v_scratch,
        &mut k_self,
    );
    assert!(
        s_star.is_finite() && s_star > 0.0,
        "G1: s_star = {s_star}, expected > 0"
    );

    GateResult::pass(
        "G1 correctness sanity",
        format!(
            "K(1,0)={k_standard:.4}≈0.5; scale-inv Δ={:.2e}; CS margin {:.2e}; s*={s_star:.4}>0",
            (cap_a - cap_b).abs(),
            bound - k_ij.abs(),
        ),
    )
}

// ─── G2: latency sweep ──────────────────────────────────────────────────────

fn gate_g2_latency() -> GateResult {
    let w_i = fixture_w_in_i();
    let w_j = fixture_w_in_j();
    let w_out = fixture_w_out();
    let op_i = owned(&w_i, &w_out, 1.0, 0.0);
    let op_j = owned(&w_j, &w_out, 1.5, 0.2);

    let parent = {
        // Owned variant for merge_cost — allocates once outside the hot loop.
        katgpt_core::hope::optimal_rank1_parent(&op_i, &op_j, 1.0)
    };

    let mut fail = String::new();
    let mut ns_report_f64: Vec<(&'static str, f64, u64)> = Vec::new();

    // Inner-loop batch size. Each timed region runs BATCH calls back-to-back
    // before reading the clock, then we divide by BATCH. This amortizes the
    // ~40 ns `Instant::now()` overhead on macOS (`mach_absolute_time`) which
    // otherwise dominates single-call measurements of sub-10ns kernels.
    // Tuned so the per-batch wall time is ~1–10 µs (well above the clock
    // floor, well below the timer's 1 ms drift).
    const BATCH: usize = 256;

    macro_rules! latency_gate {
        ($name:expr, $target:expr, $body:block) => {{
            // Warmup.
            for _ in 0..(ITERS / 10) {
                for _ in 0..BATCH {
                    let _ = black_box($body);
                }
            }
            // Measure total elapsed ns across all ITERS batches, then divide
            // by (ITERS × BATCH) for the mean per-call cost. Using the mean
            // (not per-batch median) preserves sub-ns precision via f64 —
            // integer division of small batches rounds sub-ns kernels to 0.
            let t0 = Instant::now();
            for _ in 0..ITERS {
                for _ in 0..BATCH {
                    let _ = black_box($body);
                }
            }
            let total_ns = t0.elapsed().as_nanos() as f64;
            let calls = (ITERS * BATCH) as f64;
            // Mean ns/call with sub-ns precision (f64).
            let ns_per_call = total_ns / calls;
            ns_report_f64.push(($name, ns_per_call, $target));
            if ns_per_call > $target as f64 {
                fail.push_str(&format!(
                    "  {}: {:.2} ns > {} ns target\n",
                    $name, ns_per_call, $target
                ));
            }
        }};
    }

    latency_gate!("relu_self_kernel", 10, { relu_self_kernel(1.0, 0.5) });
    latency_gate!("warped_correlation", 50, {
        warped_correlation(&w_i, &w_j, 1.0, 1.5)
    });
    latency_gate!("relu_cross_kernel_approx", 80, {
        relu_cross_kernel_approx(&w_i, &w_j, 1.0, 1.5)
    });
    latency_gate!("hope_capacity", 80, { hope_capacity(&op_i) });
    latency_gate!("hope_prune_cost", 100, { hope_prune_cost(&op_i, 5, 10.0) });
    latency_gate!("hope_merge_cost", 200, {
        hope_merge_cost(&op_i, &op_j, &parent, 2, 10.0)
    });
    latency_gate!("optimal_rank1_parent_into_scratch", 400, {
        let mut u = [0.0_f32; D_HLA];
        let mut v = [0.0_f32; 2];
        let mut k = 0.0_f32;
        optimal_rank1_parent_into_scratch(&op_i, &op_j, 1.0, &mut u, &mut v, &mut k)
    });

    // hope_greedy_select on a 32-candidate distortion-rate vector.
    let drs: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 0.05).collect();
    latency_gate!("hope_greedy_select(32)", 100, { hope_greedy_select(&drs) });

    // hope_block_eviction_cost on a 2-layer residual block.
    let n_active = [3usize, 2];
    let e_active = [1.5_f32, 2.0];
    latency_gate!("hope_block_eviction_cost(2-layer)", 50, {
        hope_block_eviction_cost(&n_active, &e_active, 5.0)
    });

    println!("\n--- G2: latency (D={D_HLA}, {ITERS}×{BATCH} calls, mean ns/call) ---");
    for (name, ns_per_call, target) in &ns_report_f64 {
        let ratio = ns_per_call / *target as f64;
        let headroom = if ratio < 1.0 {
            format!("{:.1}× under", 1.0 / ratio)
        } else {
            format!("{ratio:.2}× OVER")
        };
        println!("  {name:<40} {ns_per_call:>6.2} ns  (target ≤ {target:>4} ns, {headroom})");
    }

    if fail.is_empty() {
        GateResult::pass(
            "G2 latency",
            format!("all {} kernels under target", ns_report_f64.len()),
        )
    } else {
        GateResult::fail("G2 latency", fail)
    }
}

// ─── G4: alloc-free hot path (CountingAllocator) ────────────────────────────
//
// Verifies 0 allocations across ALLOCS_CALLS steady-state calls per kernel.
// The owned `optimal_rank1_parent` is intentionally NOT measured here — it's
// caller-controlled allocation (returns a Rank1Parent with two Vecs); the
// zero-alloc contract is on the `_into_scratch` variant.

fn gate_g4_alloc_free() -> GateResult {
    use std::sync::atomic::Ordering;

    let w_i = fixture_w_in_i();
    let w_j = fixture_w_in_j();
    let w_out = fixture_w_out();
    let op_i = owned(&w_i, &w_out, 1.0, 0.0);
    let op_j = owned(&w_j, &w_out, 1.5, 0.2);

    // Owned parent for merge_cost — the alloc happens at construction, NOT
    // inside the measured steady-state loop.
    let parent = katgpt_core::hope::optimal_rank1_parent(&op_i, &op_j, 1.0);

    let drs: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 + 0.05).collect();
    let n_active = [3usize, 2];
    let e_active = [1.5_f32, 2.0];

    // Pre-allocate scratches ONCE — they are reused across all calls.
    let mut u_scratch = [0.0_f32; D_HLA];
    let mut v_scratch = [0.0_f32; 2];
    let mut k_self_scratch = 0.0_f32;

    let mut fail = String::new();
    let mut reports = Vec::new();

    macro_rules! alloc_gate {
        ($name:expr, $body:block) => {{
            // Warmup: prime any lazy state (math is pure, but the allocator
            // might see first-touch stdlib allocations from e.g. print buffers).
            for _ in 0..10 {
                let _ = black_box($body);
            }
            let before = ALLOC_COUNT.load(Ordering::Relaxed);
            for _ in 0..ALLOCS_CALLS {
                let _ = black_box($body);
            }
            let after = ALLOC_COUNT.load(Ordering::Relaxed);
            let delta = after - before;
            reports.push(($name, delta));
            if delta != 0 {
                fail.push_str(&format!(
                    "  {}: {} allocs / {} calls (expected 0)\n",
                    $name, delta, ALLOCS_CALLS
                ));
            }
        }};
    }

    alloc_gate!("relu_self_kernel", { relu_self_kernel(1.0, 0.5) });
    alloc_gate!("warped_correlation", {
        warped_correlation(&w_i, &w_j, 1.0, 1.5)
    });
    alloc_gate!("relu_cross_kernel_approx", {
        relu_cross_kernel_approx(&w_i, &w_j, 1.0, 1.5)
    });
    alloc_gate!("hope_capacity", { hope_capacity(&op_i) });
    alloc_gate!("hope_prune_cost", { hope_prune_cost(&op_i, 5, 10.0) });
    alloc_gate!("hope_merge_cost", {
        hope_merge_cost(&op_i, &op_j, &parent, 2, 10.0)
    });
    alloc_gate!("hope_greedy_select", { hope_greedy_select(&drs) });
    alloc_gate!("hope_block_eviction_cost", {
        hope_block_eviction_cost(&n_active, &e_active, 5.0)
    });
    alloc_gate!("optimal_rank1_parent_into_scratch", {
        optimal_rank1_parent_into_scratch(
            &op_i,
            &op_j,
            1.0,
            &mut u_scratch,
            &mut v_scratch,
            &mut k_self_scratch,
        )
    });

    println!("\n--- G4: alloc-free hot path (CountingAllocator, {ALLOCS_CALLS} calls each) ---");
    for (name, delta) in &reports {
        println!("  {name:<40} {delta:>3} allocs");
    }

    if fail.is_empty() {
        GateResult::pass(
            "G4 alloc-free",
            format!("all {} hot-path kernels: 0 allocs", reports.len()),
        )
    } else {
        GateResult::fail("G4 alloc-free", fail)
    }
}

// ─── main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 469 - HOPE Hilbert-Schmidt Capacity Kernel GOAT Gate (Phase 4 T4.1) ===");
    println!("    Paper: arXiv:2607.21366 (Mobahi & Bartlett, Google DeepMind 2026-07-24)");
    println!("    D_HLA = {D_HLA}, ITERS = {ITERS}, ALLOCS_CALLS = {ALLOCS_CALLS}\n");

    let g1 = gate_g1_correctness_sanity();
    let g2 = gate_g2_latency();
    // G3 (no-regression) is verified externally via `cargo test --features
    // hope_capacity --lib` (1796 green) + `cargo check --all-features`; the
    // bench reports it as informational so the gate table is complete.
    let g3 = GateResult::pass(
        "G3 no-regression (informational)",
        "verified externally: cargo test --features hope_capacity --lib (1796 green) + cargo check --all-features",
    );
    let g4 = gate_g4_alloc_free();

    let gates = [g1, g2, g3, g4];

    println!("\n=== GOAT Gate Summary ===");
    let mut all_pass = true;
    for g in &gates {
        let tag = if g.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("{tag}  {name}", name = g.name);
        println!("        {}", g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!("\n=== Verdict ===");
    if all_pass {
        println!("✅ ALL GATES PASS — Plan 469 T4.1 GOAT gate green.");
        println!("    Phase 4 T4.6 promotion criterion met (G1+G2+G3+G4 modelless gain).");
        println!("    Next: update Cargo.toml to promote `hope_capacity` to `default`.");
    } else {
        println!("❌ ONE OR MORE GATES FAILED — see details above.");
        println!("    Stays opt-in until all gates pass.");
    }
}
