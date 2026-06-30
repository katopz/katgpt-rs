//! Group Invariance Probe — GOAT gate G4 (zero-allocation after warmup).
//!
//! `discover_subgroup_into` must allocate zero bytes on the heap after the
//! caller's scratch buffers are initialized. We verify this with a manual
//! `GlobalAlloc` counter (same pattern as `karc_alloc_check.rs`).
//!
//! Plan 356 Phase 1 T1.7.

use katgpt_core::{
    GroupAction, SubgroupClass, discover_subgroup_into, invariance_score,
};
use katgpt_core::group_invariance_probe::Rng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

// ── Minimal in-house RNG (splitmix64 — matches the in-crate test) ──────────

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Rng for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

// ── C₈ action on 8-dim indicator vectors (matches in-crate test) ───────────

struct C8Action;

impl GroupAction for C8Action {
    type Elem = u8;

    fn act(&self, g: &Self::Elem, q: &[f32], out: &mut [f32]) {
        let k = (*g as usize) % 8;
        let n = q.len();
        for i in 0..n {
            out[i] = q[(i + k) % n];
        }
    }

    fn sample(&self, rng: &mut impl Rng) -> Self::Elem {
        (rng.next_u64() % 8) as u8
    }
}

fn normalized_l1_indicator(a: &[f32], b: &[f32]) -> f32 {
    let l1: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    l1 * 0.25
}

#[test]
fn g4_discover_subgroup_into_zero_alloc_after_warmup() {
    let q = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let mut scores = [0.0_f32; 256];
    let mut rotated = [0.0_f32; 8];

    // Warmup: one call to allocate any lazy internals (there are none in
    // this primitive — it's pure float arithmetic — but the warmup pattern
    // matches `karc_alloc_check.rs` for consistency).
    let mut rng_warmup = SplitMix64::new(0xCAFE_BABE_0000_0042);
    let _ = discover_subgroup_into(
        &C8Action,
        &q,
        256,
        normalized_l1_indicator,
        10.0,
        0.5,
        &mut rng_warmup,
        &mut scores,
        &mut rotated,
    );

    // Reset counters.
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);

    // Measured call.
    let mut rng = SplitMix64::new(0x356C_C4FE_ED53_56C4);
    let report = discover_subgroup_into(
        &C8Action,
        &q,
        256,
        normalized_l1_indicator,
        10.0,
        0.5,
        &mut rng,
        &mut scores,
        &mut rotated,
    );

    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed);

    // Sanity: the probe still works (not a no-op).
    assert_eq!(
        report.class,
        SubgroupClass::Discrete,
        "G4 test should still classify C₄ ⊂ C₈ as Discrete"
    );

    assert_eq!(
        allocs, 0,
        "discover_subgroup_into allocated {allocs} bytes after warmup (expected 0); \
         deallocations: {deallocs}"
    );
}

#[test]
fn g4_invariance_score_and_classify_zero_alloc() {
    // The leaf functions (invariance_score, score_variance via classify)
    // should also allocate zero bytes — they're pure float arithmetic.
    let distances = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 0.1, 0.9, 1.1];
    let scores: [f32; 8] = std::array::from_fn(|i| invariance_score(distances[i], 10.0));

    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);

    let _class = katgpt_core::classify_subgroup(&scores, 0.5);

    let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        allocs, 0,
        "classify_subgroup allocated {allocs} bytes (expected 0)"
    );
}
