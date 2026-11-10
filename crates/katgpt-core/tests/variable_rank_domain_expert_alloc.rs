//! Plan 558 T3.4 — Variable-Rank Domain Expert G4 alloc-free hot path audit.
//!
//! Mirrors the convention of `tests/bench_453_variable_rank_domain_expert.rs`
//! and the Plan 321 CommittedFieldBlend alloc audit. Uses a CountingAllocator
//! to verify `VariableRankRouter::tick` performs ZERO heap allocations in the
//! steady state (after warmup).
//!
//! Run with:
//! ```sh
//! cargo test -p katgpt-core --features committed_field_blend \
//!   --test variable_rank_domain_expert_alloc -- --nocapture
//! ```

#![allow(clippy::float_cmp)]

use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
use katgpt_core::variable_rank_domain_expert::{ClusterHolder, VariableRankRouter};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── CountingAllocator ──────────────────────────────────────────────────────

struct CountingAllocator {
    inner: System,
    allocated: AtomicU64,
    deallocated: AtomicU64,
}

static ALLOCATOR: CountingAllocator = CountingAllocator {
    inner: System,
    allocated: AtomicU64::new(0),
    deallocated: AtomicU64::new(0),
};

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocated
            .fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { self.inner.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocated
            .fetch_add(layout.size() as u64, Ordering::Relaxed);
        unsafe { self.inner.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator {
    inner: System,
    allocated: AtomicU64::new(0),
    deallocated: AtomicU64::new(0),
};

fn reset_alloc_counters() {
    ALLOCATOR.allocated.store(0, Ordering::SeqCst);
    ALLOCATOR.deallocated.store(0, Ordering::SeqCst);
}

fn allocated_bytes() -> u64 {
    ALLOCATOR.allocated.load(Ordering::SeqCst)
}

// ─── Minimal fixture (reused from the lib tests shape) ──────────────────────

struct FixtureField<const D: usize> {
    direction: [f32; D],
    blake3: [u8; 32],
}

impl<const D: usize> ArchetypeFieldSource<D> for FixtureField<D> {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        let dot: f32 = (0..D).map(|i| z[i] * self.direction[i]).sum();
        for (i, slot) in dz_scratch.iter_mut().enumerate().take(D) {
            *slot = self.direction[i] * dot;
        }
        &mut dz_scratch[..D]
    }
    fn commitment(&self) -> [u8; 32] {
        self.blake3
    }
    fn lipschitz_bound(&self) -> f32 {
        1.0
    }
}

fn fixture_field<const D: usize>(seed: usize) -> Box<FixtureField<D>> {
    let mut direction = [0.0f32; D];
    for (i, slot) in direction.iter_mut().enumerate() {
        let x = (seed * 37 + i * 13) as f32;
        *slot = ((x * 0.1).sin() + (x * 0.07).cos()) * 0.5;
    }
    let norm: f32 = direction
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .max(1e-8);
    for v in direction.iter_mut() {
        *v /= norm;
    }
    let mut blake3 = [0u8; 32];
    for (i, b) in blake3.iter_mut().enumerate() {
        *b = ((seed * 251 + i) & 0xFF) as u8;
    }
    Box::new(FixtureField { direction, blake3 })
}

fn make_router() -> VariableRankRouter<2, 4, 2> {
    // Domain 0: L=4, K=4 (no projection)
    let mut move_blend = CommittedFieldBlend::<4, 4>::uncommitted();
    move_blend.pi = [0.5, -0.3, 0.8, 0.1];
    move_blend.tau = 1.0;
    let move_fields: [Box<dyn ArchetypeFieldSource<4>>; 4] = [
        fixture_field::<4>(100),
        fixture_field::<4>(200),
        fixture_field::<4>(300),
        fixture_field::<4>(400),
    ];
    let move_cluster = Box::new(ClusterHolder::<4, 4>::new(move_blend, move_fields));

    // Domain 1: L=2, K=2 (project to dims [0,1])
    let mut combat_blend = CommittedFieldBlend::<2, 2>::uncommitted();
    combat_blend.pi = [0.6, -0.2];
    combat_blend.tau = 1.0;
    let combat_fields: [Box<dyn ArchetypeFieldSource<2>>; 2] =
        [fixture_field::<2>(500), fixture_field::<2>(600)];
    let combat_cluster = Box::new(ClusterHolder::<2, 2>::new(combat_blend, combat_fields));

    let domain_directions: [[f32; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
    let projection_indices: [Vec<usize>; 2] = [vec![0, 1, 2, 3], vec![0, 1]];

    VariableRankRouter::<2, 4, 2>::new(
        [move_cluster, combat_cluster],
        projection_indices,
        domain_directions,
    )
}

// ─── G4: 0 allocations in steady-state tick ─────────────────────────────────

#[test]
fn g4_zero_alloc_steady_state_tick() {
    let router = make_router();
    let z = [1.0f32, 2.0, 3.0, 4.0];
    let activity = [0.7, 0.3];

    // Warmup: 1 tick (may allocate for debug prints / first-call paths).
    let mut scratch = [0.0f32; 16];
    let mut dz_out = [0.0f32; 4];
    let _ = router.tick(&z, &activity, &mut scratch, &mut dz_out);

    // Reset counters AFTER warmup.
    reset_alloc_counters();

    // 1000 ticks — should allocate 0 bytes.
    for _ in 0..1000 {
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        let _ = router.tick(&z, &activity, &mut scratch, &mut dz_out);
    }

    let bytes = allocated_bytes();
    assert_eq!(
        bytes, 0,
        "G4 FAIL: VariableRankRouter::tick allocated {bytes} bytes across 1000 steady-state calls (expected 0)"
    );
    println!("G4 PASS: 0 bytes allocated across 1000 steady-state ticks");
}
