//! Plan 607 T3 G4 — zero-alloc steady state for the fitted head's DECISION
//! path (FittedHead::pick + score over a live decision set). Separate
//! single-path binary so the counting allocator picks up ONLY this path's
//! allocations (parallel lib tests would corrupt the deltas — the
//! bench-655 convention).
//!
//! Scope note (honest by construction): `HeadFitter` allocates its D×D
//! scratch ONCE at construction — that is the cold fit path, deliberately
//! OUT of this window. The G4 claim is the per-decision hot loop: the
//! head's score/pick are stack-local f64 folds.

#![cfg(feature = "distance_abstain")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::state_option_scoring::head::HeadFitter;
use std::sync::atomic::Ordering;

const D: usize = 12; // the first consumer's design width (11 features + intercept)
const K: usize = 34; // the widest real decision set
const CALLS: usize = 100;

/// Seeded LCG (no global RNG).
fn lcg_row(seed: u64) -> [f64; D] {
    let mut s = seed;
    let mut v = [0.0f64; D];
    for x in v.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x = ((s >> 40) as f64) / (1u64 << 24) as f64 * 2.0 - 1.0;
    }
    v
}

/// G4: 0 allocations across 100 steady-state decision sets at the widest
/// real option count — pick (argmax over K dot products) and score are
/// stack-local f64 folds.
#[test]
fn g4_zero_alloc_head_pick_score() {
    // One cold fit OUTSIDE the measured window (the scratch construction
    // and the fit itself are build-time, not per-decision).
    let mut fitter = HeadFitter::<D>::new();
    let train: Vec<[f64; D]> = (0..256).map(|i| lcg_row(0x0877_a000 + i)).collect();
    let y: Vec<f64> = train.iter().map(|r| r.iter().sum()).collect();
    let head = fitter.fit_into(&train, &y, 1e-2);

    let rows: Vec<[f64; D]> = (0..K).map(|i| lcg_row(0x0877_b000 + i as u64)).collect();

    let run = || {
        let pick = head.pick(&rows, K);
        let s = head.score(&rows[pick]);
        (pick, s)
    };

    // Warmup: settle any lazy runtime state before the measured window.
    for _ in 0..5 {
        let _ = run();
    }
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut last = (0usize, 0.0f64);
    for _ in 0..CALLS {
        last = run();
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    std::hint::black_box((last, &rows));
    assert_eq!(
        (allocs, deallocs),
        (0, 0),
        "G4 FAIL: {allocs} allocs / {deallocs} deallocs across {CALLS} steady-state head decision sets"
    );
}
