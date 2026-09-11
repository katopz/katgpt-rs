//! Issue 746 G4 — zero-allocation gate for the `marginal_rewind` operator.
//!
//! `rewind_into` (the hot path — one call per recovery) must allocate 0
//! bytes after warmup: caller-supplied state/noise/buffer slices, no
//! interior allocation. `rewind_plan` / `gamma_dial_plan` are `Copy`
//! struct returns — stack-only by construction. Mirrors the
//! `saddle_escape_alloc_check` convention (separate binary:
//! `#[global_allocator]` is crate-binary-unique).

#![cfg(feature = "marginal_rewind")]

use katgpt_core::marginal_rewind::{gamma_dial_plan, rewind_into, rewind_plan};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn rewind_operator_zero_alloc_after_warmup() {
    const D: usize = 64;
    let x = [0.5_f32; D];
    let eps = [0.1_f32; D];
    let mut out = [0.0_f32; D];

    // Warmup (first-touch /assert paths — assert machinery allocates on
    // panic formatting only, which the counted window must not hit).
    for k in 1..=64usize {
        let plan = rewind_plan(0.9, 0.9 - k as f32 * 0.01);
        rewind_into(&x, plan, &eps, &mut out);
    }

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for k in 1..=4096usize {
        // s wraps in (0.051, 0.699] — stays inside the (0, t] contract.
        let plan = rewind_plan(0.7, 0.7 - (k % 650) as f32 * 0.001);
        rewind_into(&x, plan, &eps, &mut out);
        let dial = gamma_dial_plan(0.7, 1.5, 0.1);
        rewind_into(&x, dial, &eps, &mut out);
        out[0] += out[D - 1]; // sink so the loop cannot be collapsed away
    }
    let delta = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        delta, 0,
        "rewind hot path allocated {delta} bytes over 8192 calls"
    );
}
