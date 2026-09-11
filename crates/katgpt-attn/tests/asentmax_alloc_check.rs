//! Issue 747 P0 T0.5 — ASEntmax G4 zero-allocation gate.
//!
//! Steady-state hot path must allocate 0 bytes:
//! - `apply_asentmax_inplace` (1000 calls)
//! - `RollingSigmaEstimator::observe_row` + `resolve_sigma` + `to_schedule`
//!   (1000 cycles)
//! - `AsentmaxSchedule::multiplier` (1000 calls)
//!
//! Separate test binary (the `karc_alloc_check` convention): the global
//! CountingAllocator must not pick up allocations from parallel tests.
//! The allocator macro is included from katgpt-core's shared test commons
//! via a filesystem `#[path]` include — one source of truth; katgpt-core
//! is a guaranteed-present path dep of this crate.

#![cfg(feature = "asentmax_schedule")]

#[path = "../../katgpt-core/tests/common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_attn::dash_attn::asentmax::{
    AsentmaxSchedule, RollingSigmaEstimator, apply_asentmax_inplace,
};

#[test]
fn asentmax_steady_state_is_alloc_free() {
    use std::sync::atomic::Ordering;

    assert_counter_is_live();

    let n = 4_096_usize;
    let log_n = (n as f32).ln();
    let mut scores: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37) % 7.0) - 3.5).collect();
    let sched = AsentmaxSchedule::Derived { sigma_hat: 2.0 };
    let est = RollingSigmaEstimator::default();

    // Warm-up (first-touch effects, if any).
    for _ in 0..10 {
        apply_asentmax_inplace(&mut scores, &sched, log_n);
        est.observe_row(&scores);
        let _ = est.to_schedule();
    }

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..1_000 {
        apply_asentmax_inplace(&mut scores, &sched, log_n);
        est.observe_row(&scores);
        let s = est.to_schedule();
        black_box(sched.multiplier(log_n));
        black_box(s);
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "steady-state asentmax path allocated {} bytes-worth of calls",
        after - before
    );
}

#[inline]
fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}
