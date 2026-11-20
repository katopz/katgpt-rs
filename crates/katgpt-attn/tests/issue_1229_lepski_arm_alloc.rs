//! Issue 1229 G4 (consumer half) — the Lepski arm adds ZERO allocations over
//! its inner router: the stack grid, the in-place bulk-MAD σ̂ and the in-place
//! truncation allocate nothing; the only allocation is the inner router's own
//! `RoutingDecision`.
//!
//! Own test binary: a global counting allocator cannot share a binary with
//! tests that allocate in parallel (the `bench_612_alloc_check` precedent).

#![cfg(feature = "msa_adaptive_k_lepski")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use katgpt_attn::dash_attn::adaptive_k_lepski::{LepskiKConfig, LepskiKRouter};
use katgpt_attn::dash_attn::block_topk::BlockTopKRouter;
use katgpt_attn::dash_attn::vortex_flow::{VortexFlow, VortexScratch};

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
fn g4_lepski_arm_adds_zero_allocations() {
    const HD: usize = 64;
    let n = 256;
    let lepski = LepskiKRouter::new(BlockTopKRouter::new(true), LepskiKConfig::new(4, 32));
    let fixed = BlockTopKRouter::new(true);
    let mut cache = fixed.cache_new(n, HD);
    let values = vec![0.0f32; 16 * HD];
    for b in 0..n {
        let keys: Vec<f32> = (0..16 * HD)
            .map(|i| (((b * 131 + i * 17) % 251) as f32 / 251.0) - 0.5)
            .collect();
        fixed.forward_cache(&mut cache, &keys, &values, b, HD);
    }
    let queries: Vec<Vec<f32>> = (0..16)
        .map(|q| {
            (0..HD)
                .map(|d| ((q * 29 + d * 3) % 23) as f32 - 11.0)
                .collect()
        })
        .collect();
    let mut scratch = VortexScratch::new(n);

    // Warm both paths (scratch capacity growth is a first-call cost).
    for q in &queries {
        drop(fixed.forward_indexer(q, &cache, n, 32, &mut scratch));
        drop(lepski.forward_indexer(q, &cache, n, 32, &mut scratch));
    }

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    for q in &queries {
        drop(fixed.forward_indexer(q, &cache, n, 32, &mut scratch));
    }
    let inner = ALLOC_COUNT.load(Ordering::Relaxed) - before;

    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut ks = 0usize;
    for q in &queries {
        let d = lepski.forward_indexer(q, &cache, n, 32, &mut scratch);
        ks += d.len();
        drop(d);
    }
    let arm = ALLOC_COUNT.load(Ordering::Relaxed) - before;

    assert!(ks > 0, "the arm must select blocks (loud zero)");
    assert!(
        inner > 0,
        "the inner router's decision allocation must be counted (blindness floor)"
    );
    assert_eq!(
        arm, inner,
        "Lepski arm allocations {arm} != inner-router allocations {inner}"
    );
}
