//! Velocity-Field Ensemble zero-allocation test — GOAT gate G3 (Plan 376).
//!
//! `eval_into` and `eval_batch_into` must not allocate heap memory after the
//! ensemble is constructed and fit (the only allocations are the one-time
//! `EnsembleFitScratch` Vec buffers in `new()`, which live for the lifetime of
//! the scratch). We verify this with a manual `GlobalAlloc` counter — same
//! pattern as `tests/karc_alloc_check.rs`.
//!
//! `fit_into` itself performs zero allocations on the pair-accumulation path
//! (it reuses the scratch buffers); the only allocations happen in
//! `EnsembleFitScratch::new()` (the three P×P Vecs). Those are one-time and
//! outside the hot path. This test verifies the steady-state hot path.

use katgpt_core::velocity_field_ensemble::{
    ClosureField, EnsembleFitScratch, VelocityFieldEnsemble,
};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

// Constants matching the plan's G4 latency config.
const D: usize = 8;
const P: usize = 8;
const N_PAIRS: usize = 50;

// Named fn items so all P fields share the same `fn(&[f32], &mut [f32; D])`
// type (required for `[F; P]`). Each produces a distinct linear drift so the
// Gram is well-conditioned.
fn field_0(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[k] * 0.1;
    }
}
fn field_1(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 1) % D] * 0.15;
    }
}
fn field_2(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 2) % D] * 0.2 - x[k] * 0.05;
    }
}
fn field_3(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[D - 1 - k] * 0.12;
    }
}
fn field_4(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[k % 4] * 0.18 + x[(k + 4) % D] * 0.07;
    }
}
fn field_5(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 3) % D] * 0.11 - x[(k + 5) % D] * 0.04;
    }
}
fn field_6(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = (x[k] + x[(k + 1) % D]) * 0.09;
    }
}
fn field_7(x: &[f32], out: &mut [f32; D]) {
    for k in 0..D {
        out[k] = x[(k + 6) % D] * 0.13 + x[(k + 7) % D] * 0.06;
    }
}

#[allow(clippy::type_complexity)] // test fixture builder: generic ensemble return type
fn build_ensemble() -> VelocityFieldEnsemble<ClosureField<D, fn(&[f32], &mut [f32; D])>, P, D> {
    let fields = [
        ClosureField::<D, _>::new(0, field_0 as fn(&[f32], &mut [f32; D])),
        ClosureField::<D, _>::new(1, field_1 as fn(&[f32], &mut [f32; D])),
        ClosureField::<D, _>::new(2, field_2 as fn(&[f32], &mut [f32; D])),
        ClosureField::<D, _>::new(3, field_3 as fn(&[f32], &mut [f32; D])),
        ClosureField::<D, _>::new(4, field_4 as fn(&[f32], &mut [f32; D])),
        ClosureField::<D, _>::new(5, field_5 as fn(&[f32], &mut [f32; D])),
        ClosureField::<D, _>::new(6, field_6 as fn(&[f32], &mut [f32; D])),
        ClosureField::<D, _>::new(7, field_7 as fn(&[f32], &mut [f32; D])),
    ];
    VelocityFieldEnsemble::new(fields)
}

// NOTE: both checks live in ONE test function. The `#[global_allocator]`
// counting pattern is fragile under parallel test execution (cargo test runs
// tests on multiple threads by default; the shared ALLOC_COUNT catches
// allocations from sibling tests' setup phases). A single test function is
// inherently serial — matching the karc_alloc_check.rs convention (one test,
// one alloc-counted region).
#[test]
fn g3_eval_and_batch_zero_alloc_after_warmup() {
    const N_CALLS: usize = 1000;
    const N_BATCHES: usize = 100;

    let mut ensemble = build_ensemble();
    let mut scratch = EnsembleFitScratch::<P, D>::new();

    // Build N_PAIRS synthetic pairs (deterministic; content doesn't matter for
    // the alloc check).
    let xs: Vec<[f32; D]> = (0..N_PAIRS)
        .map(|i| {
            let mut x = [0.0f32; D];
            for (k, slot) in x.iter_mut().enumerate().take(D) {
                *slot = ((i + k) as f32) * 0.01;
            }
            x
        })
        .collect();
    let ys: Vec<[f32; D]> = (0..N_PAIRS)
        .map(|i| {
            let mut y = [0.0f32; D];
            for (k, slot) in y.iter_mut().enumerate().take(D) {
                *slot = ((i * 2 + k) as f32) * 0.005;
            }
            y
        })
        .collect();
    let x_refs: Vec<&[f32]> = xs.iter().map(|v| &v[..]).collect();
    let y_refs: Vec<&[f32]> = ys.iter().map(|v| &v[..]).collect();

    // Fit (one-time; allocates nothing on the pair loop).
    ensemble.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);

    // ── Part 1: eval_into zero-alloc ────────────────────────────────────────
    let mut out = [0.0f32; D];
    let mut eval_scratch = [0.0f32; D];
    for _ in 0..10 {
        ensemble.eval_into(&xs[0], &mut out, &mut eval_scratch);
    }

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut total: f32 = 0.0;
    for i in 0..N_CALLS {
        let x = &xs[i % N_PAIRS];
        ensemble.eval_into(x, &mut out, &mut eval_scratch);
        total += out[0];
    }
    std::hint::black_box(total);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: eval_into allocated {alloc_delta} times in {N_CALLS} calls"
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: eval_into deallocated {dealloc_delta} times in {N_CALLS} calls"
    );

    // ── Part 2: eval_batch_into zero-alloc ──────────────────────────────────
    const BATCH: usize = 100;
    let batch_x: Vec<[f32; D]> = (0..BATCH)
        .map(|i| {
            let mut x = [0.0f32; D];
            for (k, slot) in x.iter_mut().enumerate().take(D) {
                *slot = (i as f32) * 0.1 + (k as f32) * 0.01;
            }
            x
        })
        .collect();
    let mut batch_out: [[f32; D]; BATCH] = [[0.0; D]; BATCH];
    let batch_x_refs: Vec<&[f32]> = batch_x.iter().map(|v| &v[..]).collect();
    let mut batch_out_refs: Vec<&mut [f32; D]> = batch_out.iter_mut().collect();

    // Warmup.
    for _ in 0..3 {
        ensemble.eval_batch_into(&batch_x_refs, &mut batch_out_refs, &mut eval_scratch);
    }

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut total: f32 = 0.0;
    for _ in 0..N_BATCHES {
        ensemble.eval_batch_into(&batch_x_refs, &mut batch_out_refs, &mut eval_scratch);
        // Read through the live mutable borrow (avoids reborrowing batch_out).
        total += batch_out_refs[0][0];
    }
    std::hint::black_box(total);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: eval_batch_into allocated {alloc_delta} times in {N_BATCHES} batches of {BATCH}"
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: eval_batch_into deallocated {dealloc_delta} times in {N_BATCHES} batches of {BATCH}"
    );
}
