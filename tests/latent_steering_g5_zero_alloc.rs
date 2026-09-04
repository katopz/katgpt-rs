//! Latent Field Steering — G5 zero-allocation steady-state gate (Plan 309 T2.5).
//!
//! Pattern mirrors `tests/cross_res_g4_zero_alloc.rs`:
//! - Debug-only: `katgpt_rs::alloc::TrackingAllocator` is the global allocator
//!   only under `#[cfg(debug_assertions)]`. Thread-local counters give per-test
//!   isolation.
//! - Release: timing sanity only (TrackingAllocator is debug-only).
//!
//! ## Gate
//!
//! - **PASS (debug):** 0 allocations over 1000 crowd-applies after warmup.
//! - **PASS (release):** timing sanity (no hard gate — just not pathologically slow).
//!
//! ## Run
//!
//! ```bash
//! # Debug — formal allocation audit.
//! cargo test --features latent_field_steering \
//!   --test latent_steering_g5_zero_alloc -- --nocapture
//!
//! # Release — timing sanity.
//! cargo test --features latent_field_steering --release \
//!   --test latent_steering_g5_zero_alloc -- --nocapture
//! ```

#![cfg(feature = "latent_field_steering")]

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

use katgpt_core::latent_steering::{
    FieldSupport, LatentField, LatentSteeringVector, apply_field_to_crowd,
};

const D: usize = 8;
const N_NPCS: usize = 5000;

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    fn next_f32(&mut self) -> f32 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        let bits = (self.s >> 11) as u32;
        bits as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

#[test]
fn g5_zero_alloc_steady_state() {
    let mut rng = Rng::new(0xA110C);

    let mut dir: Vec<f32> = (0..D).map(|_| rng.next_f32()).collect();
    let norm = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut dir {
        *x /= norm;
    }
    let steering = LatentSteeringVector::new(dir, 0.3, 1e-4).unwrap();
    let field = LatentField {
        steering,
        support: FieldSupport::Global,
    };

    let mut states: Vec<f32> = (0..N_NPCS * D).map(|_| rng.next_f32()).collect();
    let positions = vec![None; N_NPCS];
    let zones = vec![None; N_NPCS];

    // ── Debug: allocation audit ───────────────────────────────────────
    #[cfg(debug_assertions)]
    {
        // Warmup.
        const MEASURED_ITERS: usize = 1000;

        for _ in 0..10 {
            apply_field_to_crowd(&mut states, D, &positions, &zones, &field);
        }
        std::hint::black_box(&states);

        assert_alloc_tracking_live();
        katgpt_core::alloc::reset_alloc_stats();
        for _ in 0..MEASURED_ITERS {
            apply_field_to_crowd(&mut states, D, &positions, &zones, &field);
        }
        std::hint::black_box(&states);

        let (count, bytes) = katgpt_core::alloc::get_alloc_stats();
        println!(
            "G5 latent_steering: {count} allocations, {bytes} bytes over {MEASURED_ITERS} \
             crowd-applies ({N_NPCS} NPCs × {D}d, global field)"
        );
        assert!(
            count == 0,
            "G5 FAIL: apply_field_to_crowd allocated {count} times ({bytes} bytes) over \
             {MEASURED_ITERS} calls. Expected zero — the hot path must be allocation-free \
             after warmup (all inputs are borrowed slices, no internal buffering)."
        );
        assert!(bytes == 0);
        println!("G5 PASS: zero allocations on the steady-state hot path.");
    }

    // ── Release: timing sanity ─────────────────────────────────────────
    #[cfg(not(debug_assertions))]
    {
        use std::time::Instant;
        const N: usize = 10_000;

        for _ in 0..10 {
            apply_field_to_crowd(&mut states, D, &positions, &zones, &field);
        }
        std::hint::black_box(&states);
        let t0 = Instant::now();
        for _ in 0..N {
            apply_field_to_crowd(&mut states, D, &positions, &zones, &field);
        }
        let elapsed = t0.elapsed();
        std::hint::black_box(&states);
        let us_per = elapsed.as_nanos() as f64 / N as f64 / 1000.0;
        println!(
            "G5 release timing: {us_per:.1} µs/crowd-apply ({N_NPCS} NPCs × {D}d); \
             {N} iters in {elapsed:?}"
        );
        // Sanity: should be well under 5ms even on a slow build.
        assert!(
            us_per < 5000.0,
            "G5 timing FAIL: {us_per:.1} µs/crowd-apply is pathologically slow"
        );
        println!("G5 release timing OK: {us_per:.1} µs/crowd-apply.");
    }
}
