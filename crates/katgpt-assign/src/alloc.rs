//! Allocation tracking for the G4 gate (the katgpt-core `alloc` module's
//! pattern, re-implemented locally to keep the zero-dep posture).
//!
//! The gate predicate is `any(debug_assertions, feature = "alloc_tracking")`
//! — Issue 741's lesson carried over: gating a MEASUREMENT on
//! `debug_assertions` alone couples "can I measure?" to "am I optimised?"
//! and makes every `--release` gate run compile to an empty binary. With
//! the feature, `--release --features alloc_tracking` runs the zero-alloc
//! assertion against the optimised code that actually ships.
//!
//! Counters are per-thread (thread-local `Cell`) — the solve loop is
//! single-threaded, so attribution is exact.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

#[derive(Clone, Copy)]
struct AllocStats {
    count: usize,
    bytes: usize,
}

impl AllocStats {
    const ZERO: Self = Self { count: 0, bytes: 0 };
}

thread_local! {
    static STATS: Cell<AllocStats> = const { Cell::new(AllocStats::ZERO) };
}

/// Counting allocator wrapper (`#[global_allocator]` is installed by the
/// test harness below, test builds only).
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        STATS.with(|s| {
            let mut cur = s.get();
            cur.count += 1;
            cur.bytes += layout.size();
            s.set(cur);
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Reset the calling thread's counters.
pub fn reset() {
    STATS.with(|s| s.set(AllocStats::ZERO));
}

/// Read the calling thread's counters `(count, bytes)`.
pub fn get() -> (usize, usize) {
    STATS.with(|s| {
        let cur = s.get();
        (cur.count, cur.bytes)
    })
}

// Test-only global registration (mirrors katgpt-core's TEST_GLOBAL_ALLOC):
// does not exist when the crate is consumed as a library dep, so no
// double-declare conflict with a consumer's own allocator.
#[cfg(all(test, any(debug_assertions, feature = "alloc_tracking")))]
#[global_allocator]
static TEST_GLOBAL_ALLOC: TrackingAllocator = TrackingAllocator;

#[cfg(all(test, any(debug_assertions, feature = "alloc_tracking")))]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::{Limits, Seed};

    /// G4: the solve hot loop allocates nothing. `Solver::new` allocates
    /// all scratch up front; every scan/apply reuses cleared buffers.
    #[test]
    fn solve_hot_loop_is_alloc_free() {
        let problem = fixtures::random_instance(99, 200, 10, 2, 50);
        // Build the solver OUTSIDE the measured window.
        let solver = crate::Solver::new(&problem, Seed(99), Limits::default()).unwrap();
        reset();
        let solution = solver.run();
        let (count, bytes) = get();
        assert_eq!(
            (count, bytes),
            (0, 0),
            "solve() allocated: {} allocations, {} bytes (stopped_by={:?}, evals={}, accepted={})",
            count,
            bytes,
            solution.stopped_by,
            solution.moves_evaluated,
            solution.moves_accepted,
        );
    }

    /// G4, Issue 964: the GroupCount heat arm allocates nothing on
    /// recompute — the sorted/deduped membership is precomputed at compile
    /// and the count scratch is reused. The pre-fix shape (a `members
    /// .clone()` + `vec![0i64; b]` per `refresh_heat`) fired exactly here.
    #[test]
    fn group_count_solve_hot_loop_is_alloc_free() {
        let mut problem = fixtures::random_instance(97, 300, 12, 2, 50);
        problem.specs = vec![
            crate::Spec::capacity(0, 6_000),
            crate::Spec::group_count((0..200).collect()),
        ];
        // Build the solver OUTSIDE the measured window.
        let solver = crate::Solver::new(&problem, Seed(97), Limits::default()).unwrap();
        reset();
        let solution = solver.run();
        let (count, bytes) = get();
        assert_eq!(
            (count, bytes),
            (0, 0),
            "GroupCount solve allocated: {} allocations, {} bytes (stopped_by={:?}, evals={})",
            count,
            bytes,
            solution.stopped_by,
            solution.moves_evaluated,
        );
    }

    /// Read-only heat path (the `explain` surface) allocates nothing —
    /// `heat_of_container` consumed the same per-call clone before 964.
    #[test]
    fn heat_of_container_is_alloc_free() {
        let mut problem = fixtures::random_instance(96, 300, 12, 2, 50);
        problem.specs = vec![
            crate::Spec::capacity(0, 6_000),
            crate::Spec::group_count((0..200).collect()),
        ];
        let solver = crate::Solver::new(&problem, Seed(96), Limits::default()).unwrap();
        reset();
        let mut total = 0i64;
        for c in 0..problem.num_containers {
            total += solver.heat_of_container(c as u32);
        }
        let (count, bytes) = get();
        assert_eq!(
            (count, bytes),
            (0, 0),
            "heat_of_container loop allocated: {} allocations, {} bytes (total={})",
            count,
            bytes,
            total,
        );
    }
}
