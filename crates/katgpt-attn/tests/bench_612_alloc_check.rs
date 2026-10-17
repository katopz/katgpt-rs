//! Plan 612 T2.4 — G4 zero-alloc gate for the pyramid selection path.
//!
//! Own test binary (the `asentmax_alloc_check` precedent): a global
//! counting allocator cannot share a binary with tests that allocate in
//! parallel — their allocations would land inside this test's delta and
//! flake the bar.
//!
//! Synthetic keys are in-bounds here (the random-key BAN covers
//! selection-QUALITY claims, Plan 612's binding constraint); allocation
//! behaviour is data-independent — only shapes matter.

#![cfg(feature = "pyramid_topk")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use katgpt_attn::dash_attn::pyramid_topk::{
    PyramidKeyHierarchy, PyramidScoreMode, PyramidScorer, PyramidScratch, coarse_to_fine_select,
};

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn g4_zero_alloc_selection_steady_state() {
    // Real gate geometry: head_dim 256, 8192 keys (128 leaves, 8 levels),
    // the gate's K=8 and GQA group of 6.
    let (n, d, group) = (8192usize, 256usize, 6usize);
    let mut keys = Vec::with_capacity(n * d);
    let mut x = 0x612_9e37u64;
    for _ in 0..n * d {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        keys.push((x >> 40) as i32 as f32 / (1 << 22) as f32);
    }
    let mut storage = vec![0.0f32; PyramidKeyHierarchy::required_len(n, d) + 8];
    let hier = PyramidKeyHierarchy::build(&keys, n, d, &mut storage);
    let qs: Vec<Vec<f32>> = (0..group)
        .map(|h| {
            (0..d)
                .map(|i| (i as f32 * 0.01) - (h as f32 + 1.2))
                .collect()
        })
        .collect();
    let heads: Vec<&[f32]> = qs.iter().map(|q| q.as_slice()).collect();
    let mut scratch = PyramidScratch::new();
    let scorer = PyramidScorer {
        mode: PyramidScoreMode::ExactLse,
        scale: 1.0 / 16.0,
    };

    // Warmup: grow every scratch Vec to steady state (u/cand/nonforced/
    // retained/indices/pairs/out all reach their working capacity).
    for _ in 0..3 {
        coarse_to_fine_select(&hier, &keys, &heads, n - 1, 8, scorer, &mut scratch);
    }
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut blackhole = 0usize;
    for _ in 0..10 {
        blackhole += coarse_to_fine_select(&hier, &keys, &heads, n - 1, 8, scorer, &mut scratch);
    }
    let delta = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    let _ = blackhole;
    println!("[612] G4: {delta} allocations across 10 warmed selections (bar: 0)");
    assert_eq!(
        delta, 0,
        "selection path must be allocation-free in steady state"
    );
}
