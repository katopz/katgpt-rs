//! Plan 294 Phase 5 T5.2 — GOAT Gate G5: zero-alloc hot path.
//!
//! Uses the existing `katgpt_rs::alloc::TrackingAllocator` (debug-only,
//! per-thread counters — see `src/alloc.rs`) to verify that
//! `BranchingDetector::observe_and_detect_into` allocates **0 bytes** after
//! warmup. The warmup allows the detector to populate its pre-allocated
//! scratch fields; the measurement loop then asserts no further growth.
//!
//! ## Contract
//!
//! - Warmup: 100 calls (allocates any first-touch buffers — there shouldn't
//!   be any since `new()` pre-allocates everything, but we keep the warmup
//!   symmetric with other G5 gates).
//! - Measure: 1000 calls. Assert: 0 allocations on this thread during the
//!   measured window.
//!
//! ## Run
//!
//! ```text
//! cargo test --features ict_branching --test bench_294_ict_g5 -- --nocapture
//! ```
//!
//! **Note:** `TrackingAllocator` is `#[cfg(debug_assertions)]` only. This
//! test compiles but no-ops on release builds — the contract is enforced
//! in debug test runs, which is where CI gates live.

#![cfg(feature = "ict_branching")]

use katgpt_core::ict::{BranchingDetector, BranchingReport};

const K_TRAJECTORIES: usize = 8;
const ACTION_DIM: usize = 32;
const WARMUP_ITERS: usize = 100;
const MEASURE_ITERS: usize = 1000;
const TOLERANCE: usize = 0; // Strict: 0 allocs/call expected.

/// Build a deterministic K=8 × action_dim=32 trajectory set.
fn make_trajectories() -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(K_TRAJECTORIES);
    let mut seed = 0xABCDEF01u64;
    for _ in 0..K_TRAJECTORIES {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let dom = (seed >> 32) as usize % ACTION_DIM;
        let dom_mass = 0.5 + 0.2 * ((seed & 0xFFFF) as f32 / 65535.0);
        let mut p = vec![0.0_f32; ACTION_DIM];
        p[dom] = dom_mass;
        let rest = (1.0 - dom_mass) / (ACTION_DIM - 1) as f32;
        for j in 0..ACTION_DIM {
            if j != dom {
                p[j] = rest;
            }
        }
        out.push(p);
    }
    out
}

#[cfg(debug_assertions)]
#[test]
fn g5_zero_alloc_hot_path() {
    use katgpt_rs::alloc::{get_alloc_stats, reset_alloc_stats};

    let trajectories = make_trajectories();
    let traj_refs: Vec<&[f32]> = trajectories.iter().map(|v| v.as_slice()).collect();

    let mut det = BranchingDetector::new(K_TRAJECTORIES, ACTION_DIM, 0.10, 0.05);
    let mut report = BranchingReport {
        mask: vec![false; K_TRAJECTORIES],
        beta_per_step: vec![0.0; K_TRAJECTORIES],
        uniqueness_scores: vec![0.0; K_TRAJECTORIES],
    };

    // ── Warmup (any first-touch allocation should happen here). ──
    for _ in 0..WARMUP_ITERS {
        det.observe_and_detect_into(&traj_refs, &mut report);
    }

    // ── Reset counters and measure. ──
    reset_alloc_stats();
    let mut sink = 0.0_f32;
    for _ in 0..MEASURE_ITERS {
        det.observe_and_detect_into(&traj_refs, &mut report);
        sink += report.uniqueness_scores[0];
    }
    let (count, bytes) = get_alloc_stats();

    // Prevent the compiler from folding `sink` away.
    if sink.is_nan() {
        eprintln!("impossible: sink nan");
    }

    println!("\n=== G5 — Zero-alloc hot path ===");
    println!(
        "K={}, action_dim={}, warmup={}, measured={}",
        K_TRAJECTORIES, ACTION_DIM, WARMUP_ITERS, MEASURE_ITERS
    );
    println!("Allocations during measured window: count = {count}, bytes = {bytes}");
    println!("Tolerance: {TOLERANCE} allocs/call.");

    let per_call = count as f64 / MEASURE_ITERS as f64;
    let verdict = if count == TOLERANCE { "PASS" } else { "FAIL" };
    println!(
        "G5 {verdict}: {per_call:.3} allocs/call (mean), {count} total across {MEASURE_ITERS} calls."
    );

    assert!(
        count == TOLERANCE,
        "G5 FAIL: expected ≤ {TOLERANCE} allocs across {MEASURE_ITERS} calls, got {count} \
         ({per_call:.3}/call). Hot path is not zero-alloc — see BranchingDetector::observe_and_detect_into."
    );
}

#[cfg(not(debug_assertions))]
#[test]
fn g5_zero_alloc_hot_path_noop_in_release() {
    // TrackingAllocator is debug-only. This test exists so the test file
    // compiles in release builds (where the contract is not enforceable).
    // The contract is enforced in debug CI runs.
    println!("\nG5: skipped in release build (TrackingAllocator is debug-only).");
}
