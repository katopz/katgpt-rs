//! Per-target tracking allocator for alloc-gate test binaries (katgpt-rs
//! Issue 721 T3).
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

#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;
