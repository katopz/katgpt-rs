//! Plan 330 — Analytic Lattice G5 zero-alloc gate (separate test binary).
//!
//! This is a SEPARATE test binary from `analytic_lattice_goat.rs` because the
//! `CountingAllocator` global would pick up allocations from other tests if
//! they ran in the same binary in parallel. By isolating it, we get clean
//! alloc/dealloc deltas.
//!
//! All three zero-alloc checks live in ONE test function because the tests
//! within a single binary still run in parallel by default, and they share
//! the global `CountingAllocator`. One function = serial by construction.
//!
//! Pattern matches `tests/karc_alloc_check.rs` (Plan 308 G3).

#![cfg(feature = "analytic_lattice")]

use katgpt_core::analytic_lattice::{
    LatticeVector, TransportOperator, batch_compose_chain_into, compose_chain_into,
    direction_vector_decode,
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

/// G5 zero-alloc gate for all three hot-path primitives.
///
/// Single function so the three checks run serially (they share the global
/// `CountingAllocator` — parallel execution would corrupt the deltas).
#[test]
fn g5_zero_alloc_after_warmup_all_primitives() {
    // ── compose_chain_into ──
    {
        const K: usize = 8;
        const N_OPS: usize = 3;

        let mut seed: u64 = 999;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (1u64 << 31) as f32 * 0.5
        };
        let ops: Vec<TransportOperator> = (0..N_OPS)
            .map(|_| {
                let data: Vec<f32> = (0..K * K).map(|_| rng()).collect();
                TransportOperator::from_row_major(K, data).unwrap()
            })
            .collect();

        let mut scratch = Vec::new();
        let mut out = TransportOperator::zeros(K);

        // Warmup: settle any lazy allocations (SIMD dispatcher Once, etc.).
        for _ in 0..5 {
            compose_chain_into(&ops, &mut scratch, &mut out).unwrap();
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        const N_CALLS: usize = 1000;
        let mut sink = 0.0f32;
        for _ in 0..N_CALLS {
            compose_chain_into(&ops, &mut scratch, &mut out).unwrap();
            sink += out.as_slice()[0];
        }

        let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
        let alloc_delta = alloc_after - alloc_before;
        let dealloc_delta = dealloc_after - dealloc_before;
        std::hint::black_box(sink);

        assert_eq!(
            alloc_delta, 0,
            "G5 FAIL: compose_chain_into allocated {} times in {} calls",
            alloc_delta, N_CALLS
        );
        assert_eq!(
            dealloc_delta, 0,
            "G5 FAIL: compose_chain_into deallocated {} times in {} calls",
            dealloc_delta, N_CALLS
        );
    }

    // ── batch_compose_chain_into ──
    {
        const K: usize = 8;
        const N: usize = 16;

        let mut seed: u64 = 777;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (1u64 << 31) as f32 * 0.5
        };

        let prefix: Vec<f32> = (0..K * K).map(|_| rng()).collect();
        let suffixes: Vec<f32> = (0..N * K * K).map(|_| rng()).collect();
        let mut out = vec![0.0f32; N * K * K];

        for _ in 0..5 {
            batch_compose_chain_into(&prefix, &suffixes, &mut out, K, N);
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        const N_CALLS: usize = 1000;
        let mut sink = 0.0f32;
        for _ in 0..N_CALLS {
            batch_compose_chain_into(&prefix, &suffixes, &mut out, K, N);
            sink += out[0];
        }

        let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
        let alloc_delta = alloc_after - alloc_before;
        let dealloc_delta = dealloc_after - dealloc_before;
        std::hint::black_box(sink);

        assert_eq!(
            alloc_delta, 0,
            "G5 FAIL: batch_compose_chain_into allocated {} times in {} calls",
            alloc_delta, N_CALLS
        );
        assert_eq!(
            dealloc_delta, 0,
            "G5 FAIL: batch_compose_chain_into deallocated {} times in {} calls",
            dealloc_delta, N_CALLS
        );
    }

    // ── direction_vector_decode ──
    {
        const N: usize = 8;

        let mut seed: u64 = 555;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (1u64 << 31) as f32 * 2.0 - 1.0
        };

        let state = LatticeVector::<N>::new([
            rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng(),
        ]);
        let direction = LatticeVector::<N>::new([
            rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng(),
        ]);

        for _ in 0..5 {
            let _ = direction_vector_decode(&state, &direction, 1.0);
        }

        let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

        const N_CALLS: usize = 1000;
        let mut sink = 0.0f32;
        for _ in 0..N_CALLS {
            sink += direction_vector_decode(&state, &direction, 1.0);
        }

        let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
        let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
        let alloc_delta = alloc_after - alloc_before;
        let dealloc_delta = dealloc_after - dealloc_before;
        std::hint::black_box(sink);

        assert_eq!(
            alloc_delta, 0,
            "G5 FAIL: direction_vector_decode allocated {} times in {} calls",
            alloc_delta, N_CALLS
        );
        assert_eq!(
            dealloc_delta, 0,
            "G5 FAIL: direction_vector_decode deallocated {} times in {} calls",
            dealloc_delta, N_CALLS
        );
    }
}
