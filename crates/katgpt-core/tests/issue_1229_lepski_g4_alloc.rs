//! Issue 1229 — G4 zero-alloc census over the Lepski cascade hot path.
//!
//! Separate test binary (the crate-binary-unique `#[global_allocator]`
//! convention). Steady state with warm caller-owned buffers: grid build,
//! prefix means, τ̂, σ̂ (in-place selection), the cascade, Lambert W and the
//! critical radius must allocate ZERO. Gated by `required-features = ["estimators"]`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRACKING: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) == 1 {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

use katgpt_core::lepski_cascade::{
    A_DEFAULT, critical_radius_sq, geometric_grid_into, lambert_w0, lepski_cascade,
    network_volume_v, prefix_means_into, sigma_hat_consecutive, sigma_hat_disagreement, tau_full,
    tau_hat_into,
};

#[test]
fn g4_hot_path_allocates_zero() {
    let ys: Vec<f64> = (0..1024).map(|i| ((i as f64) * 0.37).sin()).collect();
    let mut grid = [0usize; 16];
    let mut est = [0.0f64; 16];
    let mut tau = [0.0f64; 16];
    let mut scratch = vec![0.0f64; ys.len()];
    let widths = [1usize, 3, 1];
    let mut sink = 0.0f64;

    for round in 0..2 {
        // Round 0 warms; round 1 is measured.
        if round == 1 {
            ALLOC_COUNT.store(0, Ordering::SeqCst);
            TRACKING.store(1, Ordering::SeqCst);
        }
        let jn = geometric_grid_into(4, 512, &mut grid);
        let s = sigma_hat_consecutive(&ys, &mut scratch).unwrap();
        tau_hat_into(s, ys.len(), &grid[..jn], &mut tau);
        let m = prefix_means_into(&ys, &grid[..jn], &mut est);
        let sel = lepski_cascade(&est[..m], &tau[..m], A_DEFAULT).unwrap();
        let v = network_volume_v(&widths);
        let d2 = critical_radius_sq(grid[sel.index], 10, 1, v);
        sink += tau_full(0.1, d2, grid[sel.index], jn, 0.05)
            + lambert_w0(1e6)
            + sigma_hat_disagreement(3, 10).unwrap();
        if round == 1 {
            TRACKING.store(0, Ordering::SeqCst);
        }
    }
    std::hint::black_box(sink);
    let n = ALLOC_COUNT.load(Ordering::SeqCst);
    println!("[i1229 G4] steady-state allocations = {n}");
    assert_eq!(n, 0, "Lepski hot path allocated {n} times");
}
