//! Bench 922 — loop-straightness GOAT gates (Issue 922).
//!
//! - **G2 (latency)** — one `action_efficiency` over a K=8 knot trajectory
//!   (dim 4, the deliberation-cadence workload shape) against the sub-µs
//!   budget; K=64 (dim 32, the cgsp bridge shape) reported with a
//!   linear-scaling disclosure. Best-of-N, `black_box` on inputs and sink.
//! - **G4 (zero-alloc)** — 0 allocations in steady state over 1000 calls
//!   (counting allocator, the bench_407/775 mirror).
//!
//! G1 (η=1 bit-exact on dyadic straight paths, A/B/A = 0, floor sweep over
//! 500 seeded walks, knot-refinement identity, degenerate pins) lives in the
//! in-module test suite — `cargo test -p katgpt-dec --features
//! loop_straightness --lib`.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/bench922 cargo bench -p katgpt-dec \
//!   --features loop_straightness --no-default-features \
//!   --bench bench_922_loop_straightness_goat -- --nocapture
//! ```

#![cfg(feature = "loop_straightness")]

// Shared CountingAllocator macro (the bench_407/775 mirror).
#[path = "../tests/common/counting_allocator.rs"]
mod counting_allocator;

use katgpt_dec::{action_efficiency, path_action};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

counting_allocator!();

const BUDGET_NS: f64 = 1_000.0; // sub-µs class, the issue's G2 bar
const BEST_OF: usize = 7;

fn bench(label: &str, points: &[f32], stride: usize, weights: &[f32]) -> f64 {
    let mut best = f64::INFINITY;
    let mut sink = 0.0f32;
    for _ in 0..BEST_OF {
        let t = Instant::now();
        for _ in 0..1000 {
            sink += black_box(action_efficiency(
                black_box(points),
                black_box(stride),
                black_box(weights),
            ));
        }
        best = best.min(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    println!("  {label}: {best:.1} ns/call (budget {BUDGET_NS:.0} ns) — sink {sink}");
    best
}

fn main() {
    println!("bench 922 — loop-straightness GOAT (Issue 922)");

    // K=8 knots, dim 4 — the deliberation-cadence workload shape.
    let mut points = [0.0f32; 9 * 4];
    for k in 1..9 {
        for d in 0..4 {
            points[k * 4 + d] = (k as f32) * 0.125 + 0.01 * d as f32;
        }
    }
    let weights = [0.125f32; 8];
    let small = bench("G2 K=8 dim=4", &points, 4, &weights);
    assert!(
        small < BUDGET_NS,
        "G2 FAIL: {small:.1} ns >= {BUDGET_NS:.0} ns budget"
    );

    // K=64 knots, dim 32 — the cgsp-bridge shape; reported with the scaling
    // disclosure (linear in K·dim), no separate bar (the primitive's cost is
    // one pass over the log; a straight-line projection of the K=8 rate is
    // the honest expectation, not a gate).
    let mut big = vec![0.0f32; 65 * 32];
    for k in 1..65 {
        for d in 0..32 {
            big[k * 32 + d] = (k as f32) * 0.015625 + 0.001 * d as f32;
        }
    }
    let bw = vec![1.0f32 / 64.0; 64];
    let large = bench("G2 K=64 dim=32 (disclosed)", &big, 32, &bw);
    let ratio = (64.0 * 32.0) / (8.0 * 4.0);
    println!(
        "  scaling: K·dim grew {ratio:.0}x, latency {:.1}x (linear expectation)",
        large / small
    );

    // G4 — zero allocations in steady state (the bench_775 pattern).
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..1000 {
        let (w, c, s) = path_action(&points, 4, &weights);
        black_box((w, c, s));
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    println!("  G4: {allocs} allocations / 1000 path_action calls");
    assert!(allocs == 0, "G4 FAIL: {allocs} allocations in steady state");

    println!("✓ bench 922 PASS — G2 {small:.1} ns < {BUDGET_NS:.0} ns, G4 0 allocs");
}
