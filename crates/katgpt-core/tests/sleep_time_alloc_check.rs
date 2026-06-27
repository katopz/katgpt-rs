//! Plan 334 — Sleep-Time G5 zero-alloc GOAT gate (separate test binary).
//!
//! The wake-time hot path `consume()` MUST NOT allocate. This is the G5 gate
//! (paper §5.3: amortization only pays off if the per-consumer wake-time cost
//! is dominated by the gate hit, not by allocation/indirection overhead).
//!
//! Separate binary from `sleep_time_goat.rs` so the `CountingAllocator` global
//! picks up only `consume()` allocations (matches the `karc_alloc_check.rs`
//! and `analytic_lattice_alloc_check.rs` pattern).
//!
//! `anticipate()` is allowed to allocate (it builds the output artifact). The
//! per-direction compute itself is zero-alloc (caller provides scratch).
//!
//! Both checks live in ONE test function because tests within a single binary
//! still run in parallel by default, and they share the global
//! `CountingAllocator`. One function = serial by construction (matches the
//! `analytic_lattice_alloc_check.rs` pattern).

#![cfg(feature = "sleep_time_anticipation")]

use katgpt_core::sleep_time::{
    consume, consume_gate, AnticipatedQueryDir, DotPredictabilityScorer, IdentityFunctorOp,
    SleepTimeAnticipator, SleepTimeScratch,
};
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

/// G5 zero-alloc gate for both `consume()` and `consume_gate()`.
///
/// Single test function so both checks run serially against the shared global
/// `CountingAllocator` (parallel test execution would corrupt the deltas —
/// the alloc counter is process-wide).
///
/// The plan specifies 100 warmup calls + 100 measured calls. We do 200 warmup
/// + 1000 measured to be extra strict — the gate is "0 allocs over 100 calls",
/// so 1000 calls with 0 allocs is a 10× stronger guarantee.
#[test]
fn g5_zero_alloc_after_warmup_both_paths() {
    const D: usize = 8; // HLA dim (matches paper's NPC HLA scale).
    const K: usize = 4; // small catalog (ambient NPC scale).

    // Build a c' artifact once (outside the measured window — anticipate may alloc).
    let dirs = [
        AnticipatedQueryDir::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        AnticipatedQueryDir::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let anticipator = SleepTimeAnticipator::<D, K, IdentityFunctorOp, DotPredictabilityScorer> {
        op: IdentityFunctorOp,
        scorer: DotPredictabilityScorer::default(),
        budgets: [100; K],
        tau: 0.5,
        beta: 4.0,
    };
    let c = [0.3; D];
    let mut scratch = SleepTimeScratch::new();
    let artifact = anticipator.anticipate(&c, &dirs, &mut scratch);

    let q = [0.7; D];
    // Function pointer (ZST) — guarantees no closure-capture allocation.
    fn fresh_think_fn(fq: &[f32; 8]) -> [f32; 8] {
        let mut z = [0.0f32; 8];
        for j in 0..8 {
            z[j] = fq[j] * 0.5 + 1.0;
        }
        z
    }
    let fresh_think: fn(&[f32; 8]) -> [f32; 8] = fresh_think_fn;

    // ── consume() G5 gate ─────────────────────────────────────────────────
    {
        // Warmup: settle any lazy allocations (SIMD dispatcher Once, etc.).
        const N_WARMUP: usize = 200;
        let mut warmup_sink = 0.0f32;
        for _ in 0..N_WARMUP {
            let out = consume(&q, &artifact, 0.5, 4.0, fresh_think);
            warmup_sink += out[0];
        }
        std::hint::black_box(warmup_sink);

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        // Measured window.
        const N_CALLS: usize = 1000;
        let mut sink = 0.0f32;
        for _ in 0..N_CALLS {
            let out = consume(&q, &artifact, 0.5, 4.0, fresh_think);
            sink += out[0];
        }
        std::hint::black_box(sink);

        let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
        let alloc_delta = alloc_after - alloc_before;
        let dealloc_delta = dealloc_after - dealloc_before;

        assert_eq!(
            alloc_delta, 0,
            "G5 FAIL: consume() allocated {} times in {} calls (expected 0)",
            alloc_delta, N_CALLS
        );
        assert_eq!(
            dealloc_delta, 0,
            "G5 FAIL: consume() deallocated {} times in {} calls (expected 0)",
            dealloc_delta, N_CALLS
        );
    }

    // ── consume_gate() G5 gate ────────────────────────────────────────────
    {
        // Warmup.
        const N_WARMUP: usize = 200;
        let mut warmup_sink = 0usize;
        for _ in 0..N_WARMUP {
            let (best_i, _gate) = consume_gate(&q, &artifact, 0.5, 4.0);
            warmup_sink = warmup_sink.wrapping_add(best_i);
        }
        std::hint::black_box(warmup_sink);

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        const N_CALLS: usize = 1000;
        let mut sink = 0usize;
        for _ in 0..N_CALLS {
            let (best_i, _gate) = consume_gate(&q, &artifact, 0.5, 4.0);
            sink = sink.wrapping_add(best_i);
        }
        std::hint::black_box(sink);

        let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
        let alloc_delta = alloc_after - alloc_before;
        let dealloc_delta = dealloc_after - dealloc_before;

        assert_eq!(
            alloc_delta, 0,
            "G5 FAIL: consume_gate() allocated {} times in {} calls",
            alloc_delta, N_CALLS
        );
        assert_eq!(
            dealloc_delta, 0,
            "G5 FAIL: consume_gate() deallocated {} times in {} calls",
            dealloc_delta, N_CALLS
        );
    }
}
