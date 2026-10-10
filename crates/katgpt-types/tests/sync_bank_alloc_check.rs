//! Issue 930 T6 (G4) — SyncBank observe/readout zero-allocation gate.
//!
//! The observe path must not allocate heap memory after warmup: all work is
//! arithmetic over the pre-sized `sums`/`means` slabs (one allocation each at
//! [`SyncBank::new`]). The readouts stream the slabs with no scratch at all.
//!
//! Separate test binary (the `subspace_phase_gate_alloc_check` /
//! `karc_alloc_check` convention): `#[global_allocator]` is crate-binary-
//! unique and would collide with other test modules in the lib test binary.
//! The macro is re-declared here (not shared from katgpt-core) because
//! katgpt-types is the leaf — a dev-dependency on katgpt-core would invert
//! the dependency direction.

use katgpt_types::sync_bank::{SyncBank, SYNC_BANK_LADDER};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// G4: one test per alloc-check binary (the house convention) — the tests in
/// one binary run in PARALLEL threads over the same global counter, so two
/// measuring tests would count each other's warmup/drops. Both shapes run
/// sequentially here, each with its own before/after window.
#[test]
fn sync_bank_observe_and_readouts_zero_alloc() {
    // ── D=8/K=3 production shape: observe + readouts, 10k iterations ──
    let mut bank = SyncBank::<8, 3>::new(SYNC_BANK_LADDER);
    let z = [0.5f32; 8];

    // Warmup: construction allocs land here.
    for _ in 0..16 {
        bank.observe(&z);
    }
    let mut level = [0.0f32; 3];
    let mut argmax = [0.0f32; 3];
    let mut surprise = [0.0f32; 3];
    bank.relation_level_into(&mut level, &mut argmax);
    bank.relation_surprise_into(&mut surprise);

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    let mut sink: f32 = 0.0;
    for tick in 0..10_000u32 {
        let mut zz = [0.0f32; 8];
        for (i, v) in zz.iter_mut().enumerate() {
            *v = 0.5 + ((tick + i as u32) % 7) as f32 * 0.01;
        }
        bank.observe(&zz);
        bank.relation_level_into(&mut level, &mut argmax);
        bank.relation_surprise_into(&mut surprise);
        sink += level[0] + surprise[2] + argmax[0];
    }
    std::hint::black_box(sink);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    assert_eq!(
        alloc_delta, 0,
        "observe+readouts allocated {alloc_delta} times in 10k iterations (expected 0)"
    );
    assert_eq!(
        dealloc_delta, 0,
        "observe+readouts deallocated {dealloc_delta} times in 10k iterations (expected 0)"
    );

    // ── D=64/K=3: the large-belief shape stays allocation-free too ──
    let mut big = SyncBank::<64, 3>::new(SYNC_BANK_LADDER);
    let zb = [0.5f32; 64];
    for _ in 0..16 {
        big.observe(&zb);
    }
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut sink = 0.0f32;
    for tick in 0..1_000u32 {
        let mut zz = [0.0f32; 64];
        for (i, v) in zz.iter_mut().enumerate() {
            *v = 0.5 + ((tick + i as u32) % 11) as f32 * 0.005;
        }
        big.observe(&zz);
        sink += big.surprise_max();
    }
    std::hint::black_box(sink);
    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    assert_eq!(alloc_delta, 0, "D=64 observe allocated {alloc_delta} times (expected 0)");
}
