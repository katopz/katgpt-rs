//! Issue 976 G4 — zero-allocation steady state for the outcome cache.
//!
//! Separate single-fn binary (a CountingAllocator would pick up allocations
//! from parallel tests). Audited per-round path, after a frozen Pod exists:
//! `accept_count_distribution` → `outcome_scores` → `select_top_b` →
//! `select_conformal` → `hit` (×10 000).
//!
//! ```sh
//! cargo test -p katgpt-speculative --features outcome_cache \
//!   --test outcome_cache_alloc_check --release
//! ```

use katgpt_speculative::outcome_cache::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

struct CountingAllocator {
    allocated: AtomicU64,
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocated
            .fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.allocated.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator {
    allocated: AtomicU64::new(0),
};

#[test]
fn outcome_cache_round_is_alloc_free() {
    let mut b = HistogramBuilder::new();
    for i in 0..512u32 {
        b.observe((i % 5) as usize, i % 11);
    }
    let pod = match b.try_freeze(1.0, true) {
        FreezeVerdict::Converged(p, _) | FreezeVerdict::Forced(p, _) => p,
        v => panic!("{v:?}"),
    };
    let conf = [0.9f32, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    let mut pk = [0.0f32; K_SLOTS];
    let mut scores = [0.0f32; GRID];
    let mut sel = [0u16; GRID];
    let mut hits = 0usize;

    let before = GLOBAL.allocated.load(Ordering::SeqCst);
    for i in 0..10_000usize {
        let n = accept_count_distribution(std::hint::black_box(&conf[..(i % GAMMA) + 1]), &mut pk);
        outcome_scores(&pk, &pod, &mut scores);
        let m = select_top_b(&scores, 8, &mut sel);
        hits += usize::from(hit(&sel[..m], cell(n.min(2), i % A_SLOTS)));
        let m = select_conformal(&scores, 1e-3, 8, &mut sel);
        hits += m;
    }
    let after = GLOBAL.allocated.load(Ordering::SeqCst);
    std::hint::black_box(hits);
    assert!(hits > 0, "loud zero: the measured loop did no work");
    assert_eq!(
        after - before,
        0,
        "per-round path allocated {} B",
        after - before
    );
}
