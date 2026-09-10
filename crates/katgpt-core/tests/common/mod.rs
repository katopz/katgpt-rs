//! Shared `CountingAllocator` test/bench infrastructure (Issue 044 T3).
//!
//! Provides a `counting_allocator!()` macro that emits a global allocator
//! tracking alloc/dealloc counts, plus an `alloc_delta` helper. Eliminates
//! ~25 lines of boilerplate per G3/G4 alloc-check test and bench.
//!
//! # Usage
//!
//! In a test file under `tests/`:
//! ```ignore
//! #[path = "common/mod.rs"]
//! mod common;
//! counting_allocator!();
//! ```
//!
//! In a bench file under `benches/`:
//! ```ignore
//! #[path = "../tests/common/mod.rs"]
//! mod common;
//! counting_allocator!();
//! ```
//!
//! After macro invocation, the following names are in scope at the crate
//! root (where the macro was invoked):
//! - `ALLOC_COUNT` — `AtomicUsize` of total allocations
//! - `DEALLOC_COUNT` — `AtomicUsize` of total deallocations
//! - `alloc_delta(f)` — runs `f()` and returns `(result, alloc_count_delta)`
//!
//! Callers that read counters directly (most tests/benches) should add their
//! own `use std::sync::atomic::Ordering;` — the macro does NOT emit `use`
//! statements, to avoid import conflicts at the call site.
//!
//! # Why `#[macro_export]` despite the API pollution concern?
//!
//! `macro_rules!` macros without `#[macro_export]` are NOT path-accessible
//! from sibling modules — only from descendants of the defining module. Since
//! test/bench files invoke the macro from the CRATE ROOT (not from inside
//! `mod common`), `#[macro_export]` is required to make the macro visible.
//! The export goes to the consuming test/bench crate's root (each test/bench
//! is its own crate), NOT to `katgpt_core`'s public API — `tests/common/mod.rs`
//! is not compiled into the `katgpt_core` library crate.

/// Install a global `CountingAllocator` that counts **this thread's**
/// alloc/dealloc calls.
///
/// # Per-thread, not per-process (Issue 714)
///
/// The counters were process-global `AtomicUsize` until 2026-09-03, and
/// `cargo test` runs a binary's tests on **parallel threads by default** — so
/// every allocation a SIBLING test made inside a gate's measured window landed
/// in its `after - before`. A gate was measuring its hot path plus whatever
/// else the binary happened to be doing.
///
/// It is not a subtle failure to reason about after the fact and it is not
/// rare: `plan414_hla_committed_belief_probe_goat::g4_zero_alloc` reported
/// `6 allocs in 1000 probe calls` against a product path that allocates zero
/// times, and passed 3/3 under `--test-threads=1`. 14 katgpt-core binaries
/// have two or more tests and were exposed.
///
/// Per-thread is not a workaround for the harness — it is what an alloc gate
/// has always MEANT: *did this code path, on this thread, allocate*. A
/// process-global count answers a question no gate here has ever wanted asked.
///
/// ## Why a `Cell` and `try_with`, and not `thread_local!` defaults
///
/// The hook runs INSIDE `alloc`. A lazily-initialised TLS slot can allocate on
/// first touch and recurse forever, so the slot is `const`-initialised.
/// `try_with` (not `with`) because a thread tearing down its TLS would
/// otherwise panic from inside the allocator; a dropped count during teardown
/// is the right trade.
///
/// ## The call-site API is unchanged
///
/// `ALLOC_COUNT.load(Ordering)` / `.fetch_add(n, Ordering)` still compile —
/// 37 test binaries use them. The `Ordering` argument is accepted and ignored:
/// a thread-local `Cell` needs no ordering, and taking it keeps every call site
/// working rather than trading one defect for a 37-file churn.
///
/// Emits, at the call site:
/// - `struct CountingAllocator;`
/// - `static ALLOC_COUNT: ThreadCounter` (per-thread; atomic-shaped API)
/// - `static DEALLOC_COUNT: ThreadCounter`
/// - `unsafe impl GlobalAlloc for CountingAllocator`
/// - `#[global_allocator] static A: CountingAllocator`
/// - `fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize)`
/// - `fn assert_counter_is_live()` — the canary; see below.
///
/// # ALWAYS call `assert_counter_is_live()` in an alloc gate
///
/// A counter that silently becomes a no-op passes **every** alloc gate in the
/// repo at once. That is a strictly worse failure than the one this fix
/// removes, and it is invisible: 37 green gates over zero measurement. The
/// canary forces a known allocation through the counter and asserts it moved.
#[macro_export]
macro_rules! counting_allocator {
    () => {
        struct CountingAllocator;

        std::thread_local! {
            // `const`-initialised: a lazy TLS slot can allocate on first touch,
            // from inside `alloc`, and recurse forever.
            static TL_ALLOCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
            static TL_DEALLOCS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        }

        /// A per-thread counter wearing the `AtomicUsize` API the 37 existing
        /// call sites already use. The `Ordering` is accepted and ignored —
        /// a thread-local `Cell` has no cross-thread ordering to specify.
        #[derive(Debug)]
        struct ThreadCounter(&'static std::thread::LocalKey<std::cell::Cell<usize>>);

        #[allow(dead_code)]
        impl ThreadCounter {
            #[inline]
            fn load(&self, _: std::sync::atomic::Ordering) -> usize {
                self.0.try_with(std::cell::Cell::get).unwrap_or(0)
            }
            #[inline]
            fn fetch_add(&self, n: usize, _: std::sync::atomic::Ordering) -> usize {
                // `try_with`, not `with`: a thread tearing down its TLS must
                // not panic from inside the allocator.
                self.0
                    .try_with(|c| {
                        let prev = c.get();
                        c.set(prev.wrapping_add(n));
                        prev
                    })
                    .unwrap_or(0)
            }
            #[inline]
            fn store(&self, v: usize, _: std::sync::atomic::Ordering) {
                let _ = self.0.try_with(|c| c.set(v));
            }
        }

        static ALLOC_COUNT: ThreadCounter = ThreadCounter(&TL_ALLOCS);
        static DEALLOC_COUNT: ThreadCounter = ThreadCounter(&TL_DEALLOCS);

        unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
                ALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                unsafe { std::alloc::System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
                DEALLOC_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                unsafe { std::alloc::System.dealloc(ptr, layout) }
            }
        }

        #[global_allocator]
        static A: CountingAllocator = CountingAllocator;

        #[inline]
        #[allow(dead_code)]
        fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
            let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            let r = f();
            let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            (r, after - before)
        }

        /// Force a known heap allocation through the counter and assert it
        /// moved (Issue 714).
        ///
        /// Call this in every alloc gate, BEFORE the measured window. A
        /// counter that has become a no-op reports `0` for everything and
        /// makes all 37 alloc gates pass over zero measurement — a silent,
        /// repo-wide vacuous green, and a strictly worse failure than the
        /// sibling-race this fix removes.
        ///
        /// `black_box` on the *input* size and on the vec itself: with a
        /// literal capacity and an unused result, LLVM is entitled to delete
        /// the allocation, and then the canary fails on correct code.
        #[inline(never)]
        #[allow(dead_code)]
        fn assert_counter_is_live() {
            let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            let n = std::hint::black_box(64usize);
            let v: Vec<u8> = Vec::with_capacity(n);
            std::hint::black_box(&v);
            let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
            assert!(
                after > before,
                "alloc counter is DEAD: a {n}-byte Vec allocation did not move it \
                 ({before} -> {after}). Every alloc gate in this binary is vacuous."
            );
            drop(v);
        }
    };
}
