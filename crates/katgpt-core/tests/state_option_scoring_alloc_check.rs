//! Plan 607 T1 G4 — zero-alloc steady state for the `state_option_scoring`
//! hot path (CentroidTable::new + pick + score_into). Separate single-fn
//! binary so the counting allocator picks up ONLY this path's allocations
//! (parallel lib tests would corrupt the deltas — the bench-655
//! convention).

#![cfg(feature = "distance_abstain")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::state_option_scoring::CentroidTable;
use std::sync::atomic::Ordering;

const D: usize = 64;
const K: usize = 34; // the widest real decision set
const CALLS: usize = 100;

/// Seeded LCG (no global RNG).
fn lcg_vector(seed: u64) -> [f32; D] {
    let mut s = seed;
    let mut v = [0.0f32; D];
    for x in v.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x = ((s >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0;
    }
    v
}

/// G4: 0 allocations across 100 steady-state decision sets at the widest
/// real option count — table build (normalize + stack copy), pick, and
/// score_into are all stack-local folds.
#[test]
fn g4_zero_alloc_build_pick_score() {
    let mut options = [[0.0f32; D]; K];
    for (i, o) in options.iter_mut().enumerate() {
        *o = lcg_vector(0x0000_0607_u64.wrapping_add(i as u64));
    }
    let state = lcg_vector(0x05EE_D607);
    let mut out = [0.0f32; K];

    let mut run = || {
        let table = CentroidTable::<D, K>::new(&options);
        let pick = table.pick(&state);
        table.score_into(&state, 8.0, &mut out);
        (pick, out[pick])
    };

    // Warmup: settle any lazy runtime state before the measured window.
    for _ in 0..5 {
        let _ = run();
    }
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut last = (0usize, 0.0f32);
    for _ in 0..CALLS {
        last = run();
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    std::hint::black_box((last, &out));
    assert_eq!(
        (allocs, deallocs),
        (0, 0),
        "G4 FAIL: {allocs} allocs / {deallocs} deallocs across {CALLS} steady-state decision sets"
    );
}
