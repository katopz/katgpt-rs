//! Issue 189 T2 — Variable-Rank Domain Expert macro router G4 alloc-free audit.
//!
//! Mirrors `tests/variable_rank_domain_expert_alloc.rs` but exercises the
//! monomorphized `variable_rank_router_static!` macro router instead of the
//! dynamic `VariableRankRouter`. Verifies the zero-vtable tick path performs
//! ZERO heap allocations in the steady state (after warmup).
//!
//! Run with:
//! ```sh
//! cargo test -p katgpt-core --features committed_field_blend \
//!   --test variable_rank_domain_expert_macro_alloc -- --nocapture
//! ```

#![allow(clippy::float_cmp)]

use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
use katgpt_core::variable_rank_domain_expert::ClusterHolder;
use katgpt_core::variable_rank_router_static;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

// ─── CountingAllocator ──────────────────────────────────────────────────────

struct CountingAllocator {
    inner: System,
    allocated: AtomicU64,
    deallocated: AtomicU64,
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator {
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

fn reset_alloc_counters() {
    GLOBAL.allocated.store(0, Ordering::SeqCst);
    GLOBAL.deallocated.store(0, Ordering::SeqCst);
}

fn allocated_bytes() -> u64 {
    GLOBAL.allocated.load(Ordering::SeqCst)
}

// ─── Minimal fixture (mirrors the lib tests shape) ──────────────────────────

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

// ─── Macro router definition (2-domain, D_FULL=4, A=2) ─────────────────────

variable_rank_router_static! {
    /// 2-domain alloc-test router: move (K=4, L=4) + combat (K=2, L=2).
    struct AllocTestRouter<2, 4, 2>;

    0 => move_cluster:   ClusterHolder<4, 4> => [0, 1, 2, 3];
    1 => combat_cluster: ClusterHolder<2, 2> => [0, 1];
}

fn make_macro_router() -> AllocTestRouter {
    let mut move_blend = CommittedFieldBlend::<4, 4>::uncommitted();
    move_blend.pi = [0.5, -0.3, 0.8, 0.1];
    move_blend.tau = 1.0;
    let move_fields: [Box<dyn ArchetypeFieldSource<4>>; 4] = [
        fixture_field::<4>(100),
        fixture_field::<4>(200),
        fixture_field::<4>(300),
        fixture_field::<4>(400),
    ];
    let move_cluster = ClusterHolder::<4, 4>::new(move_blend, move_fields);

    let mut combat_blend = CommittedFieldBlend::<2, 2>::uncommitted();
    combat_blend.pi = [0.6, -0.2];
    combat_blend.tau = 1.0;
    let combat_fields: [Box<dyn ArchetypeFieldSource<2>>; 2] =
        [fixture_field::<2>(500), fixture_field::<2>(600)];
    let combat_cluster = ClusterHolder::<2, 2>::new(combat_blend, combat_fields);

    AllocTestRouter::new(move_cluster, combat_cluster, [[1.0, 0.0], [0.0, 1.0]])
}

// ─── G4: 0 allocations in steady-state tick ─────────────────────────────────
//
// Note: we keep this to a SINGLE test to avoid the global-allocator bleed
// between concurrent test threads (the test harness runs tests in a thread
// pool; a second CountingAllocator test in the same binary picks up
// background allocations from the first test's thread teardown).

#[test]
fn g4_macro_router_zero_alloc_all_paths() {
    let mut router = make_macro_router();
    let z = [1.0f32, 2.0, 3.0, 4.0];
    let activity = [0.7, 0.3];
    let pi_move = [0.5f32, -0.3, 0.8, 0.1];
    let pi_combat = [0.6f32, -0.2];

    // Warmup: 20 ticks covering both the plain tick path + the override_pi
    // path (prime all code paths / JIT-equivalent first-call paths).
    for _ in 0..20 {
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        router.override_cluster_pi(0, &pi_move);
        router.override_cluster_pi(1, &pi_combat);
        let _ = router.tick(&z, &activity, &mut scratch, &mut dz_out);
    }

    reset_alloc_counters();

    // Phase 1: 1000 plain ticks (no override_pi) — should allocate 0 bytes.
    for _ in 0..1000 {
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        let _ = router.tick(&z, &activity, &mut scratch, &mut dz_out);
    }
    let plain_bytes = allocated_bytes();
    assert_eq!(
        plain_bytes, 0,
        "G4 FAIL (plain tick): macro router allocated {plain_bytes} bytes across 1000 calls (expected 0)"
    );
    println!("G4 PASS (plain tick): 0 bytes across 1000 ticks");

    // Reset + Phase 2: 1000 ticks with per-NPC pi override — should also allocate 0.
    reset_alloc_counters();
    for _ in 0..1000 {
        let mut scratch = [0.0f32; 16];
        let mut dz_out = [0.0f32; 4];
        router.override_cluster_pi(0, &pi_move);
        router.override_cluster_pi(1, &pi_combat);
        let _ = router.tick(&z, &activity, &mut scratch, &mut dz_out);
    }
    let override_bytes = allocated_bytes();
    assert_eq!(
        override_bytes, 0,
        "G4 FAIL (override_pi): macro router allocated {override_bytes} bytes across 1000 calls (expected 0)"
    );
    println!("G4 PASS (override_pi): 0 bytes across 1000 ticks");
}
