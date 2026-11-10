//! MAG G5 — zero-allocation gate (Plan 418 Phase 2 T2.5).
//!
//! Verifies the hot-path `_into` variants allocate 0 bytes after warmup:
//! - `mine_direction_into` — writes the unit-normalized direction into a
//!   pre-allocated `&mut [f32]` buffer. No BLAKE3 commit, no MagDirection.
//! - `transfer_score_into` — computes centroid-based transfer scores using
//!   pre-allocated scratch buffers.
//!
//! The allocating wrappers (`mine_direction`, `transfer_score`) are cold-path
//! convenience APIs — they allocate per call for the result artifact. The G5
//! gate tests the zero-alloc hot-path variants only.

use katgpt_core::mag::{DataSet, TransferMetric, mine_direction_into, transfer_score_into};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const D: usize = 64;
const N: usize = 100;

#[inline]
fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(1e-10);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

#[test]
fn g5_zero_alloc_hot_path() {
    let mut rng = fastrand::Rng::with_seed(0xA500_0001);

    // Build synthetic paired data.
    let mut with = vec![[0.0_f32; D]; N];
    let mut without = vec![[0.0_f32; D]; N];
    for i in 0..N {
        for j in 0..D {
            without[i][j] = gaussian(&mut rng);
            with[i][j] = without[i][j] + if j == 0 { 2.0 } else { 0.0 };
        }
    }

    // Build a candidate + target dataset for transfer_score_into.
    let mut acts = vec![[0.0_f32; D]; 40];
    let labels = {
        let mut l = vec![false; 40];
        for x in l.iter_mut().take(20) {
            *x = true;
        }
        l
    };
    for i in 0..40 {
        for val in &mut acts[i] {
            *val = gaussian(&mut rng) + if labels[i] { 1.0 } else { -1.0 };
        }
    }
    let candidate = DataSet::new(&acts, &labels);
    let target = DataSet::new(&acts, &labels);

    // Pre-allocate reusable buffers.
    let mut dir_buf = vec![0.0_f32; D];
    let mut scratch = vec![0.0_f32; 2 * D];

    // Warmup: settle any lazy allocations.
    for _ in 0..10 {
        let _ = mine_direction_into(&with, &without, &mut dir_buf);
        let _ = transfer_score_into(
            &candidate,
            &target,
            TransferMetric::CentroidCosine,
            &mut scratch,
        );
        let _ = transfer_score_into(
            &candidate,
            &target,
            TransferMetric::ClassConditionalCosineBenign,
            &mut scratch,
        );
    }

    // Measure.
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    let mut sink = 0.0_f32;
    for _ in 0..1000 {
        let norm = mine_direction_into(&with, &without, &mut dir_buf).unwrap();
        sink += norm;
        let s1 = transfer_score_into(
            &candidate,
            &target,
            TransferMetric::CentroidCosine,
            &mut scratch,
        )
        .unwrap();
        sink += s1;
        let s2 = transfer_score_into(
            &candidate,
            &target,
            TransferMetric::ClassConditionalCosineBenign,
            &mut scratch,
        )
        .unwrap();
        sink += s2;
    }
    std::hint::black_box(sink);

    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_delta = alloc_after - alloc_before;
    let dealloc_delta = dealloc_after - dealloc_before;

    println!("G5 zero-alloc hot path (1000 iterations):");
    println!("  allocs:   {alloc_delta} (target 0)");
    println!("  deallocs: {dealloc_delta} (target 0)");

    assert_eq!(
        alloc_delta, 0,
        "G5 FAIL: hot path allocated {alloc_delta} times in 1000 iterations"
    );
    assert_eq!(
        dealloc_delta, 0,
        "G5 FAIL: hot path deallocated {dealloc_delta} times in 1000 iterations"
    );
}
