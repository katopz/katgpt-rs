//! Per-target tracking allocator for alloc-gate test binaries (katgpt-rs
//! Issue 721 T3; profile axis widened to a feature in Issue 741).
//!
//! The root crate used to register `TrackingAllocator` behind a bare
//! `#[cfg(debug_assertions)]` **as a library**, which chose the process
//! allocator for every downstream binary that linked it: any consumer's own
//! registration was a hard compile conflict, and consumers that did not
//! register one silently depended on this crate staying linked. T3 moved the
//! registration out of the library (`src/lib.rs` now gates it on
//! `cfg(all(test, debug_assertions))` — unit tests only, the katgpt-core
//! house pattern), and every integration test / example that asserts on
//! `katgpt_core::alloc` counters installs its own copy via this module:
//!
//! ```ignore
//! #[path = "common/alloc_tracking.rs"]
//! mod alloc_tracking;
//! ```
//!
//! (Examples use `#[path = "../tests/common/alloc_tracking.rs"]`.)
//!
//! This replaces the Issue-682 force-link pattern (`extern crate
//! katgpt_rs;`), which existed only to keep the root's library-level shim
//! from being dropped by the linker — there is no library-level shim anymore.

// Issue 741: `any(debug_assertions, feature = "alloc_tracking")`, matching the
// gate on katgpt-core's `alloc` module. It was `debug_assertions` alone, which
// meant an alloc gate could only ever be measured in a profile nobody ships:
// under `--release` the target compiled to an empty binary and printed
// `ok. 0 passed`, exit 0. A consumer of this module opts into release
// measurement by enabling `alloc_tracking` and widening its own `#![cfg]` to
// the same predicate.
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[global_allocator]
static GLOBAL_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;
