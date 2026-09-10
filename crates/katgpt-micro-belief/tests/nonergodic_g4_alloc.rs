#![cfg(feature = "nonergodic_belief")]
//! G4 alloc gate (Plan 592 T2.3): construction + hot path allocate ZERO.
//!
//! Own test binary → own `#[global_allocator]` (the CountingAllocator only
//! sees this target's allocations; matches the canon_goat / procrustes_bench
//! convention of inlining the allocator rather than sharing it).

use katgpt_micro_belief::{ComponentModel, NonergodicFilter, SyntheticBlock};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[inline]
fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let r = f();
    (r, ALLOC_COUNT.load(Ordering::Relaxed) - before)
}

#[test]
fn g4_hot_path_is_alloc_free() {
    // Setup (heap allowed): K=8 synthetic D=32 components off-stack.
    let blocks: Vec<SyntheticBlock> = (0..8).map(|n| SyntheticBlock::new(32, n)).collect();
    let models: [&dyn ComponentModel; 8] = std::array::from_fn(|n| &blocks[n] as &dyn ComponentModel);

    // Construction: fully inline (η + weights + fat pointers) → 0 allocs.
    let (mut filter, build_allocs) =
        alloc_delta(|| NonergodicFilter::<8, 32>::new(models, [0.125; 8]));
    assert_eq!(build_allocs, 0, "construction allocated {build_allocs} times");

    // Setup token stream (heap allowed, outside the measured region).
    let mut rng = fastrand::Rng::with_seed(592);
    let tokens: Vec<u8> = (0..1000).map(|_| rng.u8(0..2)).collect();

    // Hot path: 1000 ticks → 0 allocs.
    let (_, tick_allocs) = alloc_delta(|| {
        for &t in &tokens {
            filter.tick(t);
        }
    });
    assert_eq!(tick_allocs, 0, "tick allocated {tick_allocs} times over 1000 ticks");

    // Telescoping readout: 100 calls → 0 allocs.
    let mut tele = [0.0f32; 8 * 32];
    let (_, tele_allocs) = alloc_delta(|| {
        for _ in 0..100 {
            filter.telescope_into(&mut tele);
        }
    });
    assert_eq!(tele_allocs, 0, "telescope_into allocated {tele_allocs} times");

    // Readout accessors → 0 allocs.
    let (_, read_allocs) = alloc_delta(|| {
        let _ = filter.committed();
        let _ = filter.revive_margin();
        let _ = filter.revive_gate();
        let _ = filter.weights();
        let _ = filter.component(3);
    });
    assert_eq!(read_allocs, 0, "readouts allocated {read_allocs} times");
}
