//! Plan 410 — Linking-Fold G4 zero-alloc GOAT gate (separate test binary).
//!
//! The hot-path fold functions `fold_projection_into` and `fold_gelu_into`
//! MUST NOT allocate. They are per-tick, in-place, `#[inline]` corrections
//! applied to latent vectors when the (cold-path) detector fires. This is the
//! G4 gate: 0 allocations over 1000 steady-state calls.
//!
//! Separate binary from `bench_410_linking_fold_goat.rs` so the
//! `CountingAllocator` global picks up only the fold's allocations (matches
//! the `sleep_time_alloc_check.rs` / `karc_alloc_check.rs` /
//! `analytic_lattice_alloc_check.rs` pattern — a CountingAllocator in the
//! bench binary would skew the `Instant::now()` timing loops).
//!
//! The detector (`detect_linking` / `detect_linking_into`) is cold-path and
//! explicitly allowed to allocate — see `linking_detector.rs` module doc. It
//! is NOT gated here.
//!
//! Single test function so both checks run serially against the shared global
//! `CountingAllocator` (parallel test execution would corrupt the deltas —
//! the alloc counter is process-wide).

use katgpt_core::linking_fold::{fold_gelu_into, fold_projection_into};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

// Shared call counts (defined at module level so all four blocks reference
// the same constants — single test function = serial execution against the
// shared global CountingAllocator).
const N_WARMUP: usize = 200;
const N_CALLS: usize = 1000;

/// G4 zero-alloc gate for both `fold_projection_into` and `fold_gelu_into`.
///
/// Single test function so both checks run serially against the shared global
/// `CountingAllocator` (parallel test execution would corrupt the deltas —
/// the alloc counter is process-wide).
///
/// Uses stack-fixed arrays for state + center so no fixture-construction
/// allocation can leak into the measured window. The fold operates in-place
/// on `&mut [f32]`; with stack inputs there is structurally nothing to
/// allocate, and this test confirms that empirically.
#[test]
fn g4_zero_alloc_after_warmup_both_folds() {
    const D_HLA: usize = 8; // HLA tick-budget dim (the hot-path target).
    const D_SHARD: usize = 64; // NeuronShard style_weights dim.

    // ── fold_projection_into @ D_HLA ────────────────────────────────────────
    {
        let mut state = [0.0_f32; D_HLA];
        let center = [0.0_f32; D_HLA];
        // Seed with a non-trivial value so the fold does real work (reflects
        // the negative half-line) rather than being a no-op.
        for (i, s) in state.iter_mut().enumerate() {
            *s = (i as f32) * 0.1 - 0.4;
        }

        // Warmup: settle any lazy allocations (SIMD dispatcher Once, etc.).
        for _ in 0..N_WARMUP {
            // Re-seed each call so the fold isn't idempotent after the first.
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.013 - 0.4;
            }
            fold_projection_into(&mut state, &center);
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        let mut sink = 0.0_f32;
        for k in 0..N_CALLS {
            // Vary the input slightly each call so the compiler can't hoist
            // the fold out of the loop as a constant-fold no-op.
            let tweak = (k as f32) * 1e-6;
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.013 - 0.4 + tweak;
            }
            fold_projection_into(&mut state, &center);
            sink += state[0];
        }
        std::hint::black_box(sink);

        let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
        let alloc_delta = alloc_after - alloc_before;
        let dealloc_delta = dealloc_after - dealloc_before;

        assert_eq!(
            alloc_delta, 0,
            "G4 FAIL: fold_projection_into (D={D_HLA}) allocated {alloc_delta} times in {N_CALLS} calls (expected 0)"
        );
        assert_eq!(
            dealloc_delta, 0,
            "G4 FAIL: fold_projection_into (D={D_HLA}) deallocated {dealloc_delta} times in {N_CALLS} calls (expected 0)"
        );
    }

    // ── fold_projection_into @ D_SHARD ──────────────────────────────────────
    {
        let mut state = [0.0_f32; D_SHARD];
        let center = [0.0_f32; D_SHARD];
        for (i, s) in state.iter_mut().enumerate() {
            *s = (i as f32) * 0.01 - 0.3;
        }

        for _ in 0..N_WARMUP {
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.011 - 0.3;
            }
            fold_projection_into(&mut state, &center);
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        let mut sink = 0.0_f32;
        for k in 0..N_CALLS {
            let tweak = (k as f32) * 1e-7;
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.011 - 0.3 + tweak;
            }
            fold_projection_into(&mut state, &center);
            sink += state[0];
        }
        std::hint::black_box(sink);

        let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
        let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
        assert_eq!(
            alloc_delta, 0,
            "G4 FAIL: fold_projection_into (D={D_SHARD}) allocated {alloc_delta} times in {N_CALLS} calls"
        );
        assert_eq!(
            dealloc_delta, 0,
            "G4 FAIL: fold_projection_into (D={D_SHARD}) deallocated {dealloc_delta} times in {N_CALLS} calls"
        );
    }

    // ── fold_gelu_into @ D_HLA ──────────────────────────────────────────────
    {
        let mut state = [0.0_f32; D_HLA];
        let center = [0.0_f32; D_HLA];
        for (i, s) in state.iter_mut().enumerate() {
            *s = (i as f32) * 0.1 - 0.4;
        }

        for _ in 0..N_WARMUP {
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.013 - 0.4;
            }
            fold_gelu_into(&mut state, &center, 10.0);
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        let mut sink = 0.0_f32;
        for k in 0..N_CALLS {
            let tweak = (k as f32) * 1e-6;
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.013 - 0.4 + tweak;
            }
            fold_gelu_into(&mut state, &center, 10.0);
            sink += state[0];
        }
        std::hint::black_box(sink);

        let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
        let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
        assert_eq!(
            alloc_delta, 0,
            "G4 FAIL: fold_gelu_into (D={D_HLA}) allocated {alloc_delta} times in {N_CALLS} calls"
        );
        assert_eq!(
            dealloc_delta, 0,
            "G4 FAIL: fold_gelu_into (D={D_HLA}) deallocated {dealloc_delta} times in {N_CALLS} calls"
        );
    }

    // ── fold_gelu_into @ D_SHARD ────────────────────────────────────────────
    {
        let mut state = [0.0_f32; D_SHARD];
        let center = [0.0_f32; D_SHARD];
        for (i, s) in state.iter_mut().enumerate() {
            *s = (i as f32) * 0.01 - 0.3;
        }

        for _ in 0..N_WARMUP {
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.011 - 0.3;
            }
            fold_gelu_into(&mut state, &center, 10.0);
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        let mut sink = 0.0_f32;
        for k in 0..N_CALLS {
            let tweak = (k as f32) * 1e-7;
            for (i, s) in state.iter_mut().enumerate() {
                *s = (i as f32) * 0.011 - 0.3 + tweak;
            }
            fold_gelu_into(&mut state, &center, 10.0);
            sink += state[0];
        }
        std::hint::black_box(sink);

        let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
        let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
        assert_eq!(
            alloc_delta, 0,
            "G4 FAIL: fold_gelu_into (D={D_SHARD}) allocated {alloc_delta} times in {N_CALLS} calls"
        );
        assert_eq!(
            dealloc_delta, 0,
            "G4 FAIL: fold_gelu_into (D={D_SHARD}) deallocated {dealloc_delta} times in {N_CALLS} calls"
        );
    }
}
