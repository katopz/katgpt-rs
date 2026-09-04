//! Functional Attention (FUNCATTN) — zero-allocation steady-state test
//! (Plan 286 T2.3 / G5).
//!
//! Pattern mirrors `tests/bench_275_swir_goat.rs::g7_step_zero_allocation_debug`:
//! - Debug-only: `katgpt_rs::alloc::TrackingAllocator` is installed as the
//!   global allocator only under `#[cfg(debug_assertions)]`. The thread-local
//!   counters give per-test isolation.
//! - Release: just exercises the hot path with a timing sanity check (no
//!   allocation audit possible — TrackingAllocator is debug-only).
//!
//! Run (debug for the allocation audit):
//! ```bash
//! cargo test --features funcattn --test funcattn_g5_zero_alloc
//! ```

#![cfg(feature = "funcattn")]

// Issue 721 T3: install the tracking allocator in THIS test binary. The root
// lib no longer registers a `#[global_allocator]` as a library (that chose
// the process allocator for every downstream binary and conflicted with any
// consumer's own registration). Replaces the Issue-682 force-link, which
// existed only to keep the root's library-level shim linked.
#[path = "common/alloc_tracking.rs"]
mod alloc_tracking;

/// Liveness sentinel (Issue 682): FAIL the audit if the TrackingAllocator
/// is not actually installed (debug builds only).
#[cfg(debug_assertions)]
fn assert_alloc_tracking_live() {
    katgpt_core::alloc::reset_alloc_stats();
    let _probe: Vec<u8> = vec![0u8; 64];
    let (count, _) = katgpt_core::alloc::get_alloc_stats();
    assert!(
        count >= 1,
        "TrackingAllocator not installed in this binary — alloc gate is VACUOUS (Issue 682)"
    );
    katgpt_core::alloc::reset_alloc_stats();
}

#[cfg(not(debug_assertions))]
fn assert_alloc_tracking_live() {}

use katgpt_core::funcattn::{FuncAttnConfig, FuncAttnScratch, funcattn_forward};

/// Deterministic xorshift64* PRNG, matching `funcattn.rs::tests::make_rng`.
fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let bits = (s >> 11) as u32;
        let u01 = bits as f32 / (u32::MAX as f32);
        v.push(u01 * 2.0 - 1.0);
    }
    v
}

/// G5 — zero-allocation steady state for `funcattn_forward`.
///
/// Pre-allocates inputs, weights, output, and `FuncAttnScratch`, runs 50
/// warmup calls to absorb any one-time `ensure_capacity` allocations, then
/// measures 100 forward calls. Asserts that the calling thread performs
/// **zero** heap allocations during the measured window.
///
/// This is the formal gate that the hot path is allocation-free after
/// warmup — the property Plan 286 T2.3 promises and that the `_into` /
/// `ensure_capacity` design exists to deliver.
#[test]
fn g5_zero_alloc_steady_state() {
    let n = 512usize;
    let d = 128usize;
    let k = 64usize;
    let cfg = FuncAttnConfig {
        d,
        k,
        ..FuncAttnConfig::default()
    };

    // Pre-allocate all inputs and scratch. These allocations happen BEFORE
    // the measured window — they are the "build the engine" phase, not the
    // hot path.
    let x_basis = seeded_vec(0x1234_5678u64, n * d);
    let x_value = seeded_vec(0x2345_6789u64, n * d);
    let w_basis = seeded_vec(0x3456_789Au64, k * d);
    let w_q = seeded_vec(0x4567_89ABu64, d * d);
    let w_k = seeded_vec(0x5678_9ABCu64, d * d);
    let w_v = seeded_vec(0x6789_ABCDu64, d * d);
    let mut out = vec![0.0f32; n * d];
    let mut scratch = FuncAttnScratch::new(n, d, k);

    // ── Debug: allocation audit ───────────────────────────────────────────
    #[cfg(debug_assertions)]
    {
        // Warm up: any one-time lazy init or `ensure_capacity` resize settles
        // here. After 50 calls, the cached (n, d, k) matches and
        // `ensure_capacity` is a no-op.
        const MEASURED_ITERS: usize = 100;

        for _ in 0..50 {
            funcattn_forward(
                &x_basis,
                &x_value,
                &w_basis,
                &w_q,
                &w_k,
                &w_v,
                &cfg,
                &mut scratch,
                &mut out,
            )
            .expect("warmup forward should succeed");
        }

        assert_alloc_tracking_live();
        katgpt_core::alloc::reset_alloc_stats();
        for _ in 0..MEASURED_ITERS {
            funcattn_forward(
                &x_basis,
                &x_value,
                &w_basis,
                &w_q,
                &w_k,
                &w_v,
                &cfg,
                &mut scratch,
                &mut out,
            )
            .expect("measured forward should succeed");
        }
        // Sink output so the optimizer cannot elide the call.
        std::hint::black_box(&out);

        let (count, bytes) = katgpt_core::alloc::get_alloc_stats();
        println!(
            "G5 FUNCATTN: {count} allocations, {bytes} bytes over {MEASURED_ITERS} forward calls \
             (d={d}, k={k}, n={n})"
        );

        // Strict zero — the design's whole point is `ensure_capacity` is a
        // no-op once dimensions match, and every internal stage writes into
        // pre-sized scratch buffers. If this fires, something in the hot path
        // is allocating and the bench/doc claims are wrong.
        assert!(
            count == 0,
            "G5 FAIL: funcattn_forward allocated {count} times ({bytes} bytes) over \
             {MEASURED_ITERS} calls. Expected zero — the hot path must be allocation-free \
             after warmup (all scratch is pre-allocated, ensure_capacity is a no-op when \
             cached dimensions match)."
        );
        assert!(
            bytes == 0,
            "G5 FAIL: funcattn_forward allocated {bytes} bytes ({count} allocations) over \
             {MEASURED_ITERS} calls. Expected zero bytes."
        );
        println!("G5 PASS: zero allocations on the steady-state hot path.");
    }

    // ── Release: timing sanity check (TrackingAllocator is debug-only) ───
    #[cfg(not(debug_assertions))]
    {
        use std::time::Instant;
        // Warm up.
        for _ in 0..50 {
            let _ = funcattn_forward(
                &x_basis,
                &x_value,
                &w_basis,
                &w_q,
                &w_k,
                &w_v,
                &cfg,
                &mut scratch,
                &mut out,
            );
        }
        std::hint::black_box(&out);

        // Best-of-50 timing in release. No alloc assertion possible —
        // TrackingAllocator is debug-only. The point of running this in
        // release is to confirm the hot path completes without panicking and
        // to give a ballpark per-call latency.
        const REL_ITERS: usize = 50;
        let mut best_us = u128::MAX;
        for _ in 0..REL_ITERS {
            let t0 = Instant::now();
            funcattn_forward(
                &x_basis,
                &x_value,
                &w_basis,
                &w_q,
                &w_k,
                &w_v,
                &cfg,
                &mut scratch,
                &mut out,
            )
            .expect("release forward should succeed");
            let dt = t0.elapsed().as_micros();
            if dt < best_us {
                best_us = dt;
            }
        }
        std::hint::black_box(&out);
        println!(
            "G5 (release): TrackingAllocator is debug-only — no allocation audit. \
             Best-of-{REL_ITERS} forward at n={n}, d={d}, k={k}: {best_us} µs/call. \
             Run `cargo test --features funcattn --test funcattn_g5_zero_alloc` in \
             debug for the allocation audit."
        );
    }
}
