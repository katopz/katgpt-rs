#![cfg(feature = "nonergodic_belief")]
//! Plan 592 T2.3 — G2 (ns/tick) + G4 (alloc count) over the
//! K ∈ {2, 8, 16} × D ∈ {8, 32, 64} grid.
//!
//! `harness = false` + `std::time::Instant` + inlined CountingAllocator —
//! the repo GOAT-bench convention (canon_goat / procrustes_bench); criterion
//! is deliberately not a dep because the CountingAllocator needs a bare
//! binary to count without harness noise. Runs at the bench (release) profile.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan592 cargo bench -p katgpt-micro-belief \
//!   --features nonergodic_belief --bench nonergodic_perf -- --nocapture
//! ```
//!
//! Gates: tick < 1000 ns at K=8/D=8 (G2); 0 allocations across every grid
//! point (G4). CPU-only primitive — no GPU gate needed.

use katgpt_micro_belief::{ComponentModel, NonergodicFilter, SyntheticBlock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ─── CountingAllocator (inlined — canon_goat convention) ───────────────────

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[inline]
fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let r = f();
    (r, ALLOC_COUNT.load(Ordering::Relaxed) - before)
}

// ─── bench ──────────────────────────────────────────────────────────────────

const WARM_TICKS: usize = 2_000;
const TIMED_TICKS: usize = 20_000;

/// Build a K×D filter over seeded SyntheticBlocks, warm it, then time
/// TIMED_TICKS ticks while counting allocations. Returns (ns/tick, allocs).
fn bench_case<const K: usize, const D: usize>() -> (f64, usize) {
    let blocks: Vec<SyntheticBlock> = (0..K).map(|n| SyntheticBlock::new(D, n as u64)).collect();
    let models: [&dyn ComponentModel; K] =
        std::array::from_fn(|n| &blocks[n] as &dyn ComponentModel);
    let mut filter = NonergodicFilter::<K, D>::new(models, [1.0f32 / K as f32; K]);
    for t in 0..WARM_TICKS {
        filter.tick((t & 1) as u8);
    }
    let start = Instant::now();
    let ((), allocs) = alloc_delta(|| {
        for t in 0..TIMED_TICKS {
            filter.tick((t & 1) as u8);
        }
    });
    // Keep the final state observable so the loop cannot be optimized away.
    if filter.weights()[0] > 2.0 {
        println!();
    }
    let elapsed = start.elapsed();
    (elapsed.as_nanos() as f64 / TIMED_TICKS as f64, allocs)
}

fn main() {
    println!(
        "nonergodic_perf — Plan 592 T2.3 (warm {WARM_TICKS} + timed {TIMED_TICKS} ticks, bench profile)"
    );
    println!("CPU-only primitive — no GPU gate needed (modelless f32 fixed-array math).");
    let grid: [(usize, usize); 9] = [
        (2, 8),
        (2, 32),
        (2, 64),
        (8, 8),
        (8, 32),
        (8, 64),
        (16, 8),
        (16, 32),
        (16, 64),
    ];
    println!("{:>4} {:>4} {:>12} {:>8}", "K", "D", "ns/tick", "allocs");
    let mut k8d8 = 0.0f64;
    for &(k, d) in &grid {
        let (ns, allocs) = match (k, d) {
            (2, 8) => bench_case::<2, 8>(),
            (2, 32) => bench_case::<2, 32>(),
            (2, 64) => bench_case::<2, 64>(),
            (8, 8) => bench_case::<8, 8>(),
            (8, 32) => bench_case::<8, 32>(),
            (8, 64) => bench_case::<8, 64>(),
            (16, 8) => bench_case::<16, 8>(),
            (16, 32) => bench_case::<16, 32>(),
            _ => bench_case::<16, 64>(),
        };
        println!("{k:>4} {d:>4} {ns:>12.1} {allocs:>8}");
        if k == 8 && d == 8 {
            k8d8 = ns;
        }
        assert_eq!(
            allocs, 0,
            "G4: tick allocated {allocs} times at K={k} D={d}"
        );
    }
    assert!(
        k8d8 < 1000.0,
        "G2: K=8/D=8 tick {k8d8} ns >= 1000 ns gate"
    );
    println!("\nG2 PASS: K=8/D=8 tick = {k8d8:.1} ns (< 1000 ns gate)");
    println!("G4 PASS: 0 allocations at all 9 grid points");
}
