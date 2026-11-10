#![cfg(feature = "closure_instrument")]
//! Plan 604 G4 — zero-allocation steady state for
//! `certified_ptg_to_motif_embedding_into` (Proposal 005 Phase 1).
//!
//! Separate single-fn binary (the `regime_probe_alloc_check` convention): a
//! counting allocator is process-global per test binary, so sharing a target
//! with parallel tests would count their allocations against this gate.
//!
//! Audited path, after warmup:
//! - `certified_ptg_to_motif_embedding_into` × 1000 on pre-built fixtures
//!   (caller-owned scratch, zeroed each call by the kernel itself).
//!
//! Run with:
//! ```sh
//! cargo test -p katgpt-core --features closure_instrument \
//!   --test bridge_certified_alloc_check -- --nocapture
//! ```

use katgpt_core::closure::{
    MotifDirections, OperatorKind, PrimitiveKind, PrimitiveTransitionGraph, PtgRecorder,
    certified_ptg_to_motif_embedding_into,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

struct CountingAllocator {
    inner: System,
    allocated: AtomicU64,
}

static ALLOCATOR: CountingAllocator = CountingAllocator {
    inner: System,
    allocated: AtomicU64::new(0),
};

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
    ALLOCATOR.allocated.load(Ordering::SeqCst)
}

#[test]
fn g4_certified_bridge_into_is_alloc_free_in_steady_state() {
    const K: usize = 32;
    const N: usize = 64;
    const NODES: usize = 48;
    const ITERS: usize = 1000;

    // ── Fixtures (pre-warmup; allocations here are not audited) ─────────
    let mut x: u64 = 0x604A_110C;
    let mut next = move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let directions: Vec<f32> = (0..K * N)
        .map(|_| ((next() >> 40) as f32 / 16_777_216.0 - 0.5) * 4.0)
        .collect();
    let dirs = MotifDirections::from_flat(directions, K, N).expect("shape");

    let mut rec = PtgRecorder::new(1);
    let mut prev: Option<u32> = None;
    for j in 0..NODES {
        let prim = (next() % 256) as u32;
        let node = rec.enter(PrimitiveKind::UserDefined(prim), j as u32, None);
        if let Some(p) = prev {
            rec.exit(p, node, OperatorKind::Sequence);
        }
        prev = Some(node);
    }
    let ptg: PrimitiveTransitionGraph = rec.finish();

    let mut feature = vec![0.0f32; N];
    let mut out = vec![0.0f32; K];

    // ── Warmup (one call so lazy state, if any, lands before the audit) ──
    let k = certified_ptg_to_motif_embedding_into(&ptg, &dirs, &mut feature, &mut out);
    assert_eq!(k, K);
    black_box(&out);

    // ── Audited steady state ─────────────────────────────────────────────
    let before = allocated_bytes();
    for _ in 0..ITERS {
        let k = certified_ptg_to_motif_embedding_into(&ptg, &dirs, &mut feature, &mut out);
        black_box((&k, out.as_ptr()));
    }
    let after = allocated_bytes();
    assert_eq!(
        before,
        after,
        "certified bridge hot path allocated {}/{ITERS} bytes across {ITERS} calls",
        after - before
    );
}
