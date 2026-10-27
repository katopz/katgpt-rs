#![cfg(feature = "grid_reselect")]
//! Issue 966 G4 — zero-allocation steady state for grid_reselect.
//!
//! Separate single-fn binary (the `entropy_bounded_lane_alloc_check`
//! convention: a CountingAllocator would pick up allocations from parallel
//! tests).
//!
//! Audited paths, on caller-owned buffers after warmup:
//! - `topk_pools_into` over an 8-step × 256-vocab block, K = 16 (×1000)
//! - `reselect_block_into` over the same pools (×1000)
//!
//! ```sh
//! cargo test -p katgpt-speculative --features grid_reselect \
//!   --test grid_reselect_alloc_check --release
//! ```

use katgpt_speculative::bigram_markov::BigramMarkovBuilder;
use katgpt_speculative::grid_reselect::{reselect_block_into, topk_pools_into};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

struct CountingAllocator {
    inner: System,
    allocated: AtomicU64,
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocated
            .fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { self.inner.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.inner.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        self.allocated.fetch_add(new_size as u64, Ordering::Relaxed);
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator {
    inner: System,
    allocated: AtomicU64::new(0),
};

fn allocated_bytes() -> u64 {
    GLOBAL.allocated.load(Ordering::SeqCst)
}

#[test]
fn grid_reselect_steady_state_is_alloc_free() {
    let steps = 8usize;
    let vocab = 256usize;
    let k = 16usize;
    let slot_len = k; // vocab > k

    // Table + buffers are built BEFORE the measured window (builder allocates
    // by design; the audited surface is the extraction + walk).
    let mut b = BigramMarkovBuilder::with_capacity(4096);
    let mut rng = fastrand::Rng::with_seed(966);
    for _ in 0..4096 {
        let p = rng.u32(0..vocab as u32);
        let n = rng.u32(0..vocab as u32);
        b.add_sequence(&[p, n]);
    }
    let table = b.build(vocab, k);

    let mut m = vec![0.0f32; steps * vocab];
    for s in m.iter_mut() {
        *s = rng.f32();
    }
    let mut pools = vec![(0u32, 0f32); steps * slot_len];
    let mut out = vec![0u32; steps];

    // Warmup (also proves the call shape populates everything).
    for _ in 0..100 {
        let n = topk_pools_into(&m, steps, vocab, k, &mut pools);
        assert_eq!(n, steps);
        let w = reselect_block_into(&table, 0, steps, slot_len, &pools, 0.5, &mut out);
        assert_eq!(w, steps);
    }

    let before = allocated_bytes();
    for _ in 0..1000 {
        let n = topk_pools_into(&m, steps, vocab, k, &mut pools);
        assert_eq!(n, steps);
        let w = reselect_block_into(&table, 0, steps, slot_len, &pools, 0.5, &mut out);
        assert_eq!(w, steps);
    }
    let after = allocated_bytes();
    assert_eq!(
        after,
        before,
        "grid_reselect steady state allocated {} bytes over 1000 cycles",
        after - before
    );
}
