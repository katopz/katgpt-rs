//! Issue 879 G4 — zero-allocation steady state for the MAttr budget
//! primitives: `exact_mass_admit_into` (the zero-alloc out-param form; the
//! allocating wrapper is cold-path by design) and `LogFrontier`
//! `sample`/`observe`.
//!
//! Separate single-fn binary (the `*_alloc_check` convention: the counting
//! allocator is a process global; parallel tests in one binary corrupt each
//! other's deltas).
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features exact_mass_admit \
//!     --test exact_mass_admit_g4_alloc --release -- --nocapture
//! ```

#![cfg(feature = "exact_mass_admit")]

use katgpt_core::exact_mass_admit::exact_mass_admit_into;
use katgpt_core::log_frontier::LogFrontier;
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn g4_zero_alloc_steady_state() {
    let n = 4096usize;
    let scores: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.618_033_9) % 6.0) - 3.0)
        .collect();
    let mut mask = vec![0.0f32; n];

    // ── Measured: the operator + the controller, 0 allocations ──────────
    let ((), allocs) = alloc_delta(|| {
        let mut acc = 0.0f32;
        for i in 0..500u32 {
            let k = 1.0 + (i % 4096) as f32;
            let tau = exact_mass_admit_into(black_box(&scores), k, 1.0, &mut mask);
            acc += mask[(i as usize) & (n - 1)] + tau * 1e-30;
        }
        let mut lf = LogFrontier::new(1024, 0.9, 0.05, 0.25, 8);
        let mut u = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..100_000u32 {
            u = u.wrapping_mul(6364136223846793005).wrapping_add(1);
            let k = lf.sample(black_box((u >> 40) as f32 / 16_777_216.0));
            acc += lf.observe(black_box(k * 1e-6)) * 1e-30;
        }
        black_box(acc);
    });
    assert_eq!(
        allocs, 0,
        "steady-state exact_mass_admit/log_frontier paths allocated {allocs}×"
    );
}
