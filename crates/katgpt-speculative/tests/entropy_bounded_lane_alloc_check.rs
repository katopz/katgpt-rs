#![cfg(feature = "entropy_bounded_commit")]
//! Issue 917 T3 G4 — zero-allocation steady state for the wired EB lanes.
//!
//! Separate single-fn binary (the `entropy_bounded_commit_alloc_check`
//! convention: a CountingAllocator would pick up allocations from parallel
//! tests).
//!
//! Audited paths, on caller-owned scratch after warmup:
//! - `build_dd_tree_eb_into` over a 4-depth × 64-vocab block (×200)
//! - `dflash_block_commit_eb_with` over a 8-step × 64-vocab block (×2000)
//!
//! ```sh
//! cargo test -p katgpt-speculative --features entropy_bounded_commit \
//!   --test entropy_bounded_lane_alloc_check --release
//! ```

use katgpt_core::entropy_bounded_commit::{EbCommitConfig, ErrorProxy};
use katgpt_speculative::NoPruner;
use katgpt_speculative::entropy_bounded::{
    EbBuildScratch, EbCommitScratch, build_dd_tree_eb_into, dflash_block_commit_eb_with,
};
use katgpt_types::Config;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
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

fn marginals(depths: usize, vocab: usize) -> Vec<Vec<f32>> {
    let mut rng = fastrand::Rng::with_seed(917);
    (0..depths)
        .map(|d| {
            // Mixed sharpness rows so the commit prefix is non-trivial.
            (0..vocab)
                .map(|i| {
                    let sharp = ((i + d) % 4) as f32 * 1.5;
                    rng.f32() * (0.1 + sharp)
                })
                .collect::<Vec<f32>>()
        })
        .collect()
}

#[test]
fn eb_lane_wiring_steady_state_is_alloc_free() {
    const DEPTHS: usize = 4;
    const VOCAB: usize = 64;
    const STEPS: usize = 8;

    let mut config = Config::draft();
    config.vocab_size = VOCAB;
    config.tree_budget = 64;
    config.early_exit_patience = 0;
    config.early_exit_gap = 0.0;

    let eb_cfg = EbCommitConfig {
        gamma: 0.5,
        max_commit: 8,
        proxy: ErrorProxy::Entropy,
    };

    let rows = marginals(DEPTHS, VOCAB);
    let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();

    let mut flat: Vec<f32> = Vec::with_capacity(STEPS * VOCAB);
    for d in 0..STEPS {
        flat.extend(rows[d % DEPTHS].iter());
    }

    // Warmup (and liveness): the lanes must commit something.
    let mut tree_scratch = EbBuildScratch::default();
    let tree = build_dd_tree_eb_into(
        &mut tree_scratch,
        &refs,
        &config,
        &NoPruner,
        black_box(&eb_cfg),
    );
    assert!(!tree.is_empty(), "DDTree EB build liveness");
    assert!(
        tree_scratch.children_committed >= 1,
        "build no-stall liveness"
    );

    let mut block_scratch = EbCommitScratch::default();
    let mut tokens = vec![0usize; STEPS];
    let k0 = dflash_block_commit_eb_with(
        black_box(&flat),
        STEPS,
        VOCAB,
        &mut block_scratch,
        &mut tokens,
        black_box(&eb_cfg),
    );
    assert!(k0 >= 1, "block commit no-stall liveness");

    // Allocation audit: plain loops, no timing inside the counted region.
    let mut acc = 0usize;
    let before = allocated_bytes();
    for _ in 0..200 {
        let tree = build_dd_tree_eb_into(
            &mut tree_scratch,
            black_box(&refs),
            &config,
            &NoPruner,
            black_box(&eb_cfg),
        );
        acc += tree.len();
    }
    for _ in 0..2000 {
        let k = dflash_block_commit_eb_with(
            black_box(&flat),
            STEPS,
            VOCAB,
            &mut block_scratch,
            &mut tokens,
            black_box(&eb_cfg),
        );
        acc += k;
    }
    let after = allocated_bytes();
    black_box(acc);
    assert!(
        acc >= 2200,
        "every pass committed (liveness of the audited loops)"
    );
    assert_eq!(after - before, 0, "EB lane steady state must not allocate");
}
