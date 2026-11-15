#![cfg(feature = "estimators")]
//! Plan 638 T2.6 — the G4 zero-alloc canary over the conditional-geometry
//! steady-state call paths.
//!
//! Separate test binary (the `#[global_allocator]` convention —
//! crate-binary-unique). The counting allocator wraps System; after one
//! warm call per path (scratch capacities settled), the steady-state
//! `ball_counts` + `cond_id` + `cond_imbalance` paths must allocate ZERO
//! over 1000 cycles — every buffer lives in the caller's scratch, sized
//! once at `with_capacity` (the plan's D6/no-alloc law).
//!
//! NOT counted (calibration by contract): `BallCountScratch::with_capacity`
//! and `ImbalanceScratch::with_capacity` (the one-time sizing), fixture
//! population construction.

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

#[path = "fixture_cond_geometry/mod.rs"]
mod fixture;

use fixture::Lcg;
use katgpt_core::data_probe::cond_geometry::{
    ball_counts, cond_id, cond_imbalance, BallCountScratch, ImbalanceScratch,
};

const N: usize = 512;
const DIM: usize = 16;
const CYCLES: usize = 1000;

#[test]
fn g4_zero_alloc_steady_state_1000_cycles() {
    // ── setup (uncounted): fixture + warm every path once ──
    let mut rng = Lcg::new(0x638_6004);
    let pts: Vec<f32> = (0..N * DIM).map(|_| rng.next_gauss()).collect();
    let to: Vec<f32> = (0..N * DIM).map(|_| rng.next_gauss()).collect();
    let all: Vec<u32> = (0..N as u32).collect();
    let group: Vec<u32> = (0..128u32).collect();

    let mut counts = BallCountScratch::with_capacity(N);
    let mut imb = ImbalanceScratch::with_capacity(N);
    ball_counts(&pts, N, DIM, &all, 0.9, &mut counts).expect("warm counts");
    cond_id(&counts, &group).expect("warm estimate");
    cond_imbalance(&pts, &to, N, DIM, &group, &mut imb).expect("warm imbalance");

    // ── the counted window ──
    TRACKING.store(1, Ordering::Relaxed);
    for _ in 0..CYCLES {
        ball_counts(&pts, N, DIM, &all, 0.9, &mut counts).expect("counts");
        let _ = cond_id(&counts, &group).expect("estimate");
        cond_imbalance(&pts, &to, N, DIM, &group, &mut imb).expect("imbalance");
    }
    TRACKING.store(0, Ordering::Relaxed);

    let leaked = ALLOC_COUNT.load(Ordering::Relaxed);
    assert_eq!(
        leaked, 0,
        "G4 FAILED: {leaked} allocations over {CYCLES} steady-state cycles — a buffer escaped the scratch"
    );
}
