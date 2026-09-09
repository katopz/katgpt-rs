#![cfg(feature = "regime_probe")]
//! Issue 740 G4 — zero-allocation steady state for the regime probes.
//!
//! Separate single-fn binary (the `karc_alloc_check` /
//! `sleep_time_alloc_check` convention): a CountingAllocator would pick up
//! allocations from parallel GOAT tests, so the audit gets its own target.
//!
//! Audited paths (after warmup):
//! - `conditional_entropies_into` — batch entropy over caller scratch (×1000)
//! - `entropy_gap_into` on a warmed report (×100)
//! - `basin_probe_into` on warmed scratch + report (×100)
//!
//! Run with:
//! ```sh
//! cargo test -p katgpt-core --features regime_probe \
//!   --test regime_probe_alloc_check -- --nocapture
//! ```

use katgpt_core::regime_probe::{
    BasinScratch, BasinReport, FrozenRenovator, basin_probe_into, conditional_entropies_into,
    entropy_gap_into,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── CountingAllocator (the repo-standard shape) ────────────────────────────

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
        self.allocated.fetch_add(layout.size() as u64, Ordering::Relaxed);
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

/// Zero-alloc test renovator: one-hot at a pinned answer table.
struct Oracle {
    answer: Vec<usize>,
}
impl FrozenRenovator for Oracle {
    fn len(&self) -> usize {
        self.answer.len()
    }
    fn alphabet(&self) -> usize {
        6
    }
    fn posterior_into(&self, i: usize, _x: &[usize], out: &mut [f32]) {
        out.fill(0.0);
        out[self.answer[i]] = 1.0;
    }
}

#[test]
fn g4_zero_alloc_steady_state() {
    const POSITIONS: usize = 256;
    const VOCAB: usize = 32;
    const SEQ: usize = 96;

    // ── Fixtures (pre-warmup; allocations here are not audited) ──────────
    let logits: Vec<f32> = (0..POSITIONS * VOCAB)
        .map(|i| ((i * 41) % 97) as f32 / 5.0 - 8.0)
        .collect();
    let mut ents = vec![0.0f32; POSITIONS];

    let mut ref_ents: Vec<f32> = Vec::with_capacity(POSITIONS);
    let mut gen_ents: Vec<f32> = Vec::with_capacity(POSITIONS);

    let original: Vec<usize> = (0..SEQ).map(|i| (i * 5) % 6).collect();
    let ren = Oracle { answer: original.clone() };
    let mut scratch = BasinScratch::new(SEQ, 6);
    let mut report = BasinReport::default();
    let mut gap_report = katgpt_core::regime_probe::EntropyGapReport::default();

    // ── Warmup (fills capacities; allocations from here are audited) ─────
    conditional_entropies_into(&logits, POSITIONS, VOCAB, &mut ents);
    ref_ents.extend_from_slice(&ents[..POSITIONS]);
    gen_ents.extend_from_slice(&ents[..POSITIONS]);
    entropy_gap_into(&ref_ents, &gen_ents, &mut gap_report);
    basin_probe_into(&ren, &original, 0.3, 2, 0x7402, &mut scratch, &mut report);

    // ── Audited steady state ─────────────────────────────────────────────
    let before = allocated_bytes();

    for _ in 0..1000 {
        conditional_entropies_into(&logits, POSITIONS, VOCAB, &mut ents);
    }
    for _ in 0..100 {
        entropy_gap_into(&ref_ents, &gen_ents, &mut gap_report);
    }
    for _ in 0..100 {
        basin_probe_into(&ren, &original, 0.3, 2, 0x7402, &mut scratch, &mut report);
    }

    let after = allocated_bytes();
    let delta = after - before;
    assert_eq!(
        delta, 0,
        "G4 FAIL: steady-state probes allocated {delta} bytes across \
         1000 entropy batches + 100 gap measurements + 100 basin probes"
    );
    println!("G4 PASS: 0 bytes allocated in steady state (audit window: {before} → {after})");
}
