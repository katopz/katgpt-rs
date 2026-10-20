//! The G4 alloc-gate machinery — the TLS-gated counting allocator
//! (Plan 628 T0.2, consolidated at T0.6).
//!
//! ONE `#[global_allocator]` per test binary is a hard rustc rule
//! (`cannot define multiple global allocators`, verified empirically), and
//! every module-internal `#[cfg(test)]` gate in this crate links into the
//! SAME lib test binary — so the T0.2 machinery that lived inside
//! `span_pointer::tests` moves here verbatim and every module's gate drives
//! it through `crate::alloc_gate` (the katgpt-core `TEST_GLOBAL_ALLOC` /
//! katgpt-assign `alloc.rs` precedent). Test-only: this module does not
//! exist outside `cfg(test)`, so consumers and shipped builds never see it.
//!
//! The gate predicate for the per-module gate TESTS is
//! `#[cfg(any(debug_assertions, feature = "alloc_tracking"))]` (the
//! repo-wide law): they run in debug test builds and in
//! `--release --features alloc_tracking`, and compile away at
//! release-default. The allocator itself stays unconditional under
//! `cfg(test)` — matching the T0.2 posture it was moved from byte for byte.

// At the flag-OFF test posture no gate uses the shared machinery (the
// feature-gated modules are compiled out); the items below are still the
// one allocator this binary runs on.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counts allocations made by the MEASURING THREAD only: the TLS flag
/// (const-init — accessing it never allocates) is set around the
/// measured window, and the allocator increments only when the CURRENT
/// thread is inside a window. Sibling test threads allocating in
/// parallel are invisible to the counter, so the gate is deterministic
/// at any --test-threads value (the retry-arm band-aid this replaced
/// false-armed under --all-features: shared-process noise failed all
/// four attempts). A REAL hot-path allocation still fails — it happens
/// on the measured thread, inside the window.
pub(crate) static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

/// RAII measured window: allocations made between `enter()` and the drop
/// (on the entering thread only) are counted into [`ALLOC_COUNT`].
pub(crate) struct MeasuringWindow;

impl MeasuringWindow {
    pub(crate) fn enter() -> Self {
        MEASURING.with(|m| m.set(true));
        Self
    }
}

impl Drop for MeasuringWindow {
    fn drop(&mut self) {
        MEASURING.with(|m| m.set(false));
    }
}

impl MeasuringWindow {
    fn is_measuring() -> bool {
        MEASURING.with(Cell::get)
    }
}

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MeasuringWindow::is_measuring() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if MeasuringWindow::is_measuring() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if MeasuringWindow::is_measuring() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static COUNTING: CountingAlloc = CountingAlloc;
