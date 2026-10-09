//! Plan 623 T1.3 G4 — zero-alloc steady state for the `event_state_windows`
//! hot path (population_state_summary at N=10k with pre-allocated scratch +
//! the window-eval loop). Separate single-fn binary so the counting
//! allocator picks up ONLY this path's allocations (parallel lib tests
//! would corrupt the deltas — the bench-655 convention).

#![cfg(feature = "event_state_windows")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::event_state_window::{event_state_window, population_state_summary};
use std::sync::atomic::Ordering;

const N: usize = 10_000;
const CALLS: usize = 100;
const WINDOW_EVALS: usize = 10_000;

/// Seeded LCG (no global RNG).
fn lcg_values(seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(N);
    for _ in 0..N {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((s >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0);
    }
    v
}

/// G4: 0 allocations across 100 steady-state summaries (scratch allocated
/// BEFORE the measured window) + 10k window evals — copy, total_cmp sort,
/// the f64 folds, and the sigmoid product are all caller-buffer work.
#[test]
fn g4_zero_alloc_summary_and_window() {
    let values = lcg_values(0x0000_0623);
    let mut scratch = vec![0.0f32; N];

    let mut run = || {
        let s = population_state_summary(&values, &mut scratch);
        let mut acc = 0.0f32;
        let mut i = 0usize;
        while i < WINDOW_EVALS {
            let t = (i % 97) as f32;
            acc += event_state_window(t, 0.0, 100.0, 10.0, 5.0);
            i += 7; // deterministic stride, well inside the support
        }
        (s.mean + s.p50, acc)
    };

    // Warmup: settle any lazy runtime state before the measured window.
    for _ in 0..5 {
        let _ = run();
    }
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut last = (0.0f32, 0.0f32);
    for _ in 0..CALLS {
        last = run();
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    std::hint::black_box((last, &scratch));
    assert_eq!(
        (allocs, deallocs),
        (0, 0),
        "G4 FAIL: {allocs} allocs / {deallocs} deallocs across {CALLS} summaries + {WINDOW_EVALS} window evals"
    );
}
