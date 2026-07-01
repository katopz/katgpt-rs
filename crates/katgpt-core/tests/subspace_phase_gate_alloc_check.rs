//! Plan 301 T4.1 — `jacobian_svd_at_into` zero-allocation gate.
//!
//! `jacobian_svd_at_into` must not allocate heap memory after warmup: all work
//! happens in the pre-sized `JacobianSvdScratch` buffers (`f_x`, `f_x_pert`,
//! `f_x_plus`, `x_pert`, `jac`, `svd_work`, `svd_result`). This is the
//! allocation-elimination that closes the plan's T3.4 latency gap (the prior
//! `jacobian_svd_at` allocating path spent ~45% of its time on the 17-`Vec`
//! SOA→owned conversion — see `.benchmarks/301_*.md` Phase 3).
//!
//! Separate test binary (matches the `karc_alloc_check` /
//! `analytic_lattice_alloc_check` convention): `#[global_allocator]` is
//! crate-binary-unique and would collide with other test modules in the lib
//! test binary.

use katgpt_core::{JacobianSvdScratch, jacobian_svd_at_into};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Wrapping allocator that counts `alloc`/`dealloc` calls. Single global
/// counter is sufficient for these single-threaded tests.
struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            System.alloc(layout)
        }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            System.dealloc(ptr, layout);
        }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[test]
fn jacobian_svd_at_into_zero_alloc_after_warmup() {
    // Deterministic rank-3 linear map R^8 → R^8 (diagonal, σ = {10, 5, 2}).
    // The alloc gate doesn't care about the spectrum shape — any rank-3 8×8
    // map exercises the same Jacobi sweeps + SOA writes. A diagonal map is
    // the simplest deterministic choice and avoids reproducing the lib-test
    // `known_rank3_map_r8x8` helper (private to the lib's test module).
    let mut w_diag = [0.0f32; 64];
    w_diag[0] = 10.0;
    w_diag[9] = 5.0;
    w_diag[18] = 2.0;
    let f = |x: &[f32], out: &mut [f32]| {
        for j in 0..8 {
            let mut acc = 0.0f32;
            for i in 0..8 {
                acc += w_diag[j * 8 + i] * x[i];
            }
            out[j] = acc;
        }
    };
    let x = [0.5f32; 8];
    let mut scratch = JacobianSvdScratch::with_capacity(8, 8);

    // Warmup: first call grows scratch buffers + primes caches / Once inits.
    for _ in 0..10 {
        jacobian_svd_at_into(f, &x, 1e-4, &mut scratch);
    }

    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    const N_CALLS: usize = 1000;
    let mut sink: f32 = 0.0;
    for _ in 0..N_CALLS {
        jacobian_svd_at_into(f, &x, 1e-4, &mut scratch);
        sink += scratch.svd_result().singular_value(0);
    }
    std::hint::black_box(sink);

    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_delta = alloc_after - alloc_before;
    let dealloc_delta = dealloc_after - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "jacobian_svd_at_into allocated {} times in {} calls (expected 0)",
        alloc_delta, N_CALLS
    );
    assert_eq!(
        dealloc_delta, 0,
        "jacobian_svd_at_into deallocated {} times in {} calls (expected 0)",
        dealloc_delta, N_CALLS
    );
}
