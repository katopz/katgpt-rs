//! Conformal zero-allocation test — GOAT gate G3 (Plan 340).
//!
//! `update_residual` and `interval_into` MUST perform zero allocations after
//! warmup. We verify this with a manual `GlobalAlloc` counter (matches the
//! `karc_alloc_check` / `analytic_lattice_alloc_check` pattern).

use katgpt_core::{
    ConformalIntervalCalibrator, DecayUnit, PredictiveInterval, ResidualMode,
    SeasonalPoolForecaster,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            System.alloc(layout)
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            System.dealloc(ptr, layout);
        }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[test]
fn g3_interval_into_zero_alloc_after_warmup() {
    let forecaster = SeasonalPoolForecaster::new(64, 12, 0.0, 0.0);
    let mut cal = ConformalIntervalCalibrator::new(
        forecaster,
        8,  // 8 channels (HLA_DIM)
        8,  // max_h
        12, // m
        64, // capacity per (channel, bucket)
        0.0,
        DecayUnit::Step,
        ResidualMode::HStep,
        false,
    );

    // Warmup: fill the residual pool.
    for i in 0..200 {
        let r = (i as f32) * 0.01 - 1.0;
        for ch in 0..8 {
            cal.update_residual(r + (ch as f32), 0.0, ch, 1);
        }
    }

    // Warmup some interval reads to settle any lazy allocations.
    // Sweep alpha values too so every code path is warmed.
    let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.05);
    for &alpha in &[0.01_f32, 0.05, 0.1, 0.2] {
        for _ in 0..50 {
            for ch in 0..8 {
                for h in 1..=8 {
                    cal.interval_into(ch, h, alpha, &mut iv);
                }
            }
        }
    }
    std::hint::black_box(iv.lower);

    // Measure.
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    const N_CALLS: usize = 1000;
    let mut sink = 0.0_f32;
    for _ in 0..N_CALLS {
        for ch in 0..8 {
            cal.interval_into(ch, 1, 0.05, &mut iv);
            sink += iv.lower;
        }
    }
    std::hint::black_box(sink);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: interval_into allocated {} times in {} calls × 8 channels",
        alloc_delta, N_CALLS
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: interval_into deallocated {} times in {} calls × 8 channels",
        dealloc_delta, N_CALLS
    );
}

#[test]
fn g3_update_residual_zero_alloc_after_warmup() {
    let forecaster = SeasonalPoolForecaster::new(64, 12, 0.0, 0.0);
    let mut cal = ConformalIntervalCalibrator::new(
        forecaster,
        8,
        8,
        12,
        64,
        0.0,
        DecayUnit::Step,
        ResidualMode::HStep,
        false,
    );

    // Warmup: fill the residual pool to capacity.
    for i in 0..200 {
        let r = (i as f32) * 0.01 - 1.0;
        for ch in 0..8 {
            cal.update_residual(r + (ch as f32), 0.0, ch, 1);
        }
    }

    // Measure update_residual at steady state (pool full → eviction path).
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    const N_CALLS: usize = 1000;
    let mut sink = 0.0_f32;
    for i in 0..N_CALLS {
        let r = (i as f32) * 0.001;
        for ch in 0..8 {
            cal.update_residual(r + (ch as f32), 0.0, ch, 1);
        }
        sink += r;
    }
    std::hint::black_box(sink);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: update_residual allocated {} times in {} calls × 8 channels",
        alloc_delta, N_CALLS
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: update_residual deallocated {} times in {} calls × 8 channels",
        dealloc_delta, N_CALLS
    );
}
