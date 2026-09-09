//! Allocation tracking for the alloc gates (G4 / G5 / G7).
//!
//! Counters are **per-thread** (thread-local `Cell`), not process-global. This
//! lets parallel tests measure allocation-free hot paths without bleeding
//! sibling-test allocations into each other's counts. Every existing caller
//! follows `reset → measure-on-calling-thread → get`, which is exactly the
//! per-thread semantic; switching from global atomics to thread-local `Cell`
//! is both more correct (natural attribution) and faster on the hot path
//! (no `lock cmpxchg`, no cross-core cache-line bouncing).
//!
//! `const` initialization in `thread_local!` avoids lazy-init allocation on
//! the fast path — the TLS slot holds the `Cell` directly.
//!
//! **Whole-module gate, carried ONCE on the `pub mod alloc;` declaration in
//! `lib.rs`** (Issue 741) — `any(debug_assertions, feature = "alloc_tracking")`.
//! Every item below exists exactly when that predicate holds, so no item
//! repeats it; a per-item `#[cfg]` here could drift out of agreement with the
//! module gate and with the `#[global_allocator]` registrations that depend on
//! it, which is why there is one gate and not ten.
//!
//! **Why the predicate is not just `debug_assertions` (Issue 741).** It was,
//! and the cost was that every alloc gate in the workspace could only ever run
//! in a profile nobody ships. `debug_assertions` is not a knob: it is ON in dev
//! and OFF in release, so a `--release` run of an alloc gate compiled the whole
//! target to an **empty binary** and printed `test result: ok. 0 passed`, exit
//! 0 — indistinguishable from a pass, and `--release` is the profile
//! `AGENTS.md` mandates for gates. Measured on `kimi_k3_g4_alloc_free`: 1
//! passed in dev, `0 passed` in release.
//!
//! The `alloc_tracking` feature makes the axis **opt-in and profile-free**:
//!
//! - dev, feature off — unchanged from before, tracking on, zero cost to add.
//! - `--release --features alloc_tracking` — the gates compile and RUN, so the
//!   zero-alloc claim is verified against the **optimised** code that actually
//!   ships. This is the configuration the alloc gates are meant to be read in.
//! - shipped release, feature off — the module does not exist, `System` is the
//!   allocator, and there is no TLS read on the allocation path. Byte-identical
//!   to the pre-741 release build.
//!
//! A profile is not a capability; a feature is. Gating a *measurement* on
//! `debug_assertions` couples "can I measure this?" to "am I optimised?", and
//! those are independent questions.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// Aggregate per-thread allocation stats (count + bytes). `Copy` so it can
/// live in a `Cell` (single load + store per `alloc`, no `RefCell` overhead).
#[derive(Clone, Copy)]
struct AllocStats {
    count: usize,
    bytes: usize,
}

impl AllocStats {
    /// `const` constructor so the `thread_local!` initializer is const-evaluable.
    const ZERO: Self = Self { count: 0, bytes: 0 };
}

thread_local! {
    /// Single TLS key for both counters — one TLS address computation per
    /// `alloc`, not two.
    static THREAD_ALLOC: Cell<AllocStats> = const { Cell::new(AllocStats::ZERO) };
}

/// Allocator wrapper that tracks allocation count and bytes on the **calling
/// thread**. Install via `#[global_allocator]` in the binary crate. Present
/// under `debug_assertions` OR the `alloc_tracking` feature — see module docs
/// for why the profile alone was the wrong axis (Issue 741).
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        THREAD_ALLOC.with(|cell| {
            let mut s = cell.get();
            s.count = s.count.wrapping_add(1);
            s.bytes = s.bytes.wrapping_add(layout.size());
            cell.set(s);
        });
        // Safety: delegated to the system allocator, layout is valid.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Safety: delegated to the system allocator, ptr+layout are valid.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Reset the **calling thread's** allocation counters to zero. Does not affect
/// other threads' counters — each thread's `Cell` is independent.
pub fn reset_alloc_stats() {
    THREAD_ALLOC.with(|cell| cell.set(AllocStats::ZERO));
}

/// Get the **calling thread's** allocation stats as `(count, total_bytes)`.
/// Returns only allocations performed on the current thread since the last
/// [`reset_alloc_stats`] on this thread.
pub fn get_alloc_stats() -> (usize, usize) {
    THREAD_ALLOC.with(|cell| {
        let s = cell.get();
        (s.count, s.bytes)
    })
}

// Plain `cfg(test)`: the enclosing module already carries the
// `any(debug_assertions, feature = "alloc_tracking")` gate (see module docs),
// so these tests exist exactly when the accessors they call do. Re-stating the
// predicate here is what made it possible for the two to disagree.
#[cfg(test)]
mod tests {
    use super::*;

    // No serialization mutex needed: each test thread's counters are isolated
    // by the thread-local `Cell`. Tests run fully parallel without interference.

    #[test]
    fn test_reset_clears_stats() {
        // Reset and immediately read on THIS thread. With thread-local
        // counters, concurrent tests on other threads cannot inflate our
        // count. Between reset and read this thread performs no allocations,
        // so count should be exactly 0. A small tolerance is kept only to
        // absorb any runtime bookkeeping the test harness itself might do on
        // this thread before reaching the assertion.
        reset_alloc_stats();
        let (count, bytes) = get_alloc_stats();
        assert!(
            count <= 5,
            "count should be near-zero after reset, got {count}"
        );
        assert!(
            bytes <= 4096,
            "bytes should be near-zero after reset, got {bytes}"
        );
    }

    #[test]
    fn test_alloc_increments_count() {
        reset_alloc_stats();
        let _v: Vec<u8> = vec![0u8; 1024];
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "at least one allocation should have occurred");
        assert!(bytes >= 1024, "bytes should be at least 1024, got {bytes}");
    }

    #[test]
    fn test_multiple_allocs_accumulate() {
        reset_alloc_stats();
        let _v1: Vec<u8> = vec![0u8; 64];
        let _v2: Vec<u8> = vec![0u8; 128];
        let (count, bytes) = get_alloc_stats();
        assert!(count >= 2, "at least two allocations, got {count}");
        assert!(bytes >= 192, "bytes should be at least 192, got {bytes}");
    }

    #[test]
    fn test_string_allocation() {
        reset_alloc_stats();
        let _s = String::from("hello world test allocation");
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "string allocation should increment count");
        assert!(bytes > 0, "string allocation should increment bytes");
    }

    #[test]
    fn test_box_allocation() {
        reset_alloc_stats();
        let _b = Box::new(42u64);
        let (count, bytes) = get_alloc_stats();
        assert!(count > 0, "box allocation should increment count");
        assert!(
            bytes >= 8,
            "box allocation should account for u64, got {bytes}"
        );
    }

    /// Thread isolation: allocations on another thread are not visible on
    /// this thread. This is the property G7 (and every other alloc-audit
    /// gate) relies on for parallel-safe measurement.
    #[test]
    fn test_thread_isolation() {
        // The worker thread allocates 4 KiB on its own thread and reports its
        // own observed byte count via the channel.
        const WORKER_ALLOC_BYTES: usize = 4096;
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            reset_alloc_stats();
            let _v: Vec<u8> = vec![0u8; WORKER_ALLOC_BYTES];
            let (_count, bytes) = get_alloc_stats();
            let _ = tx.send(bytes);
        });
        // Reset on THIS thread AFTER spawning (thread spawn itself allocates
        // bookkeeping on the spawning thread — that's legitimate runtime
        // overhead, not the property under test).
        reset_alloc_stats();
        let worker_bytes = rx.recv().expect("worker should report");
        handle.join().expect("worker thread panicked");
        let (_main_count, main_bytes) = get_alloc_stats();
        // The worker saw its own ~4 KiB allocation (modulo Vec bookkeeping).
        assert!(
            worker_bytes >= WORKER_ALLOC_BYTES,
            "worker thread should have seen its own {WORKER_ALLOC_BYTES}-byte allocation, got {worker_bytes}"
        );
        // The main thread must NOT see the worker's allocation. Some runtime
        // bookkeeping from `recv`/`join` may allocate on this thread, but it
        // is tiny (hundreds of bytes) — nowhere near 4 KiB. The defining
        // property: the worker's 4 KiB does not leak into our counter.
        assert!(
            main_bytes < WORKER_ALLOC_BYTES,
            "main thread should not see the worker's {WORKER_ALLOC_BYTES}-byte \
             allocation, but observed {main_bytes} bytes on this thread — \
             thread-local isolation is broken"
        );
    }
}
