//! Issue 747 P3 — G4 zero-allocation gate (below-τ steady state).
//!
//! Steady-state decode (stable support, candidates below τ) must allocate
//! 0 bytes: `push` on the Lemma-1 fast path is one compare + one `0.0`
//! write into pre-sized `probs`. Support-entry pushes may grow the
//! candidate list (rare by construction — that is the primitive's cost
//! model); the steady state is what must be exactly free.
//!
//! Separate test binary (the `karc_alloc_check` convention): the global
//! CountingAllocator must not pick up allocations from parallel tests.
//! The allocator macro is included from katgpt-core's shared test commons
//! via a filesystem `#[path]` include — one source of truth.

#![cfg(feature = "asentmax_schedule")]

#[path = "../../katgpt-core/tests/common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_attn::dash_attn::entmax_incremental::IncrementalEntmax1p5;

#[test]
fn incremental_below_threshold_steady_state_is_alloc_free() {
    use std::sync::atomic::Ordering;

    assert_counter_is_live();

    // Realistic decode shape: a few spikes form the stable support, the
    // bulk is N(0,1)-scale — never toppling τ again. probs is pre-sized
    // for the WHOLE measured window (capacity = warm-up + 1000): the
    // steady-state claim is "within the reservation, 0 allocs".
    let warmup = 3_096_usize;
    let measured = 1_000_usize;
    let capacity = warmup + measured;
    let mut inc = IncrementalEntmax1p5::new(capacity);
    for &spike in &[7.3_f32, 6.9, 6.4] {
        inc.push(spike);
    }
    // Warm-up to `warmup` (first-touch, candidate-list growth from the
    // seeding spikes — all outside the measured window).
    let mut state = 0xBEEF_u64;
    while inc.len() < warmup {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let s = (((state >> 40) as f32) / ((1u64 << 24) as f32)) * 4.0 - 2.0;
        let event = inc.push(s);
        assert!(!event, "warm-up stream must stay below τ");
    }

    // Measured window: 1000 more below-τ pushes, still inside the
    // pre-sized reservation.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for _ in 0..measured {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let s = (((state >> 40) as f32) / ((1u64 << 24) as f32)) * 4.0 - 2.0;
        black_box(inc.push(s));
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "below-τ steady state allocated {} bytes-worth of calls",
        after - before
    );
    assert_eq!(inc.len(), capacity);
}

#[inline]
fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}
