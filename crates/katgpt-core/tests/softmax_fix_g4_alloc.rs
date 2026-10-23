//! G4 — alloc-free `fixed_log_softmax` (Issue 959 T5).
//!
//! Both arms (f32 + f64) of the full form and the single-slot target
//! form must be allocation-free in steady state: the primitive is pure
//! arithmetic over caller-provided slices — the G4 claim is structural,
//! this gate proves it. 1000 mixed calls per arm after warmup.
//!
//! The counting allocator (tests/common) is per-thread — this test is
//! single-threaded by construction.

#![cfg(feature = "fixed_log_softmax")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::softmax_fix::{
    fixed_log_softmax, fixed_log_softmax_f64, fixed_log_softmax_target,
    fixed_log_softmax_target_f64,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;

#[test]
fn fixed_log_softmax_is_alloc_free() {
    // Margins past the fp32 cliff so the exercised path includes the
    // subnormal-tail exp terms (the band the primitive exists for).
    let z: Vec<f32> = (0..64).map(|i| -((i % 17) as f32) * 6.0).collect();
    let z64: Vec<f64> = z.iter().map(|&v| f64::from(v)).collect();
    let mut out = vec![0.0_f32; z.len()];
    let mut out64 = vec![0.0_f64; z.len()];

    // Warmup (first-touch TLS of the counter itself must not count).
    fixed_log_softmax(&z, &mut out);
    fixed_log_softmax_f64(&z64, &mut out64);
    black_box(fixed_log_softmax_target(&z, 1));
    black_box(fixed_log_softmax_target_f64(&z64, 1));

    ALLOC_COUNT.store(0, Ordering::Relaxed);
    for i in 0..1000 {
        let t = i % z.len();
        fixed_log_softmax(&z, &mut out);
        fixed_log_softmax_f64(&z64, &mut out64);
        black_box(fixed_log_softmax_target(&z, t));
        black_box(fixed_log_softmax_target_f64(&z64, t));
    }
    black_box((&out, &out64));
    assert_eq!(
        ALLOC_COUNT.load(Ordering::Relaxed),
        0,
        "fixed_log_softmax steady state must be allocation-free"
    );
}
