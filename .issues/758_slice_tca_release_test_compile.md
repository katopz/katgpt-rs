# Issue 758: slice_tca test module breaks `cargo test --release` — `crate::alloc` is debug-gated (Issue 741 profile×feature class)

**Status:** Open — found 2026-09-12 while running Issue 757's release-mode profile harness (`cargo test --release -p katgpt-core --features linking_fold`).
**Class:** the Issue 741 lesson — "a MEASUREMENT gated on debug_assertions is impossible in the configuration that ships" — here as a compile break rather than a silent skip: the profile is part of the claim, and `--release` without `alloc_tracking` compiles the lib test harness to a **compile error**.

`crates/katgpt-core/src/slice_tca/tests.rs:23` does `use crate::alloc::{get_alloc_stats, reset_alloc_stats};` — the `alloc` module is gated at `lib.rs` `#[cfg(any(debug_assertions, feature = "alloc_tracking"))]` (sliceTCA + bmr landed Phase 31, 2026-09-12). Under `cargo test --release` (debug_assertions OFF) without `alloc_tracking`:

```
error[E0432]: unresolved import `crate::alloc`
```

Consequences: no release-profile test run of katgpt-core's lib works on that configuration — anything needing release-mode test execution (perf-sensitive ignored harnesses, future release-test compile checks like riir-ai's Issue 869 lane) is blocked or silently unexercised. Not caught by the standard gates: dev-profile clippy/test lanes compile it (debug_assertions on), and `--all-features` supplies `alloc_tracking`.

## Tasks

- [ ] **T1** Pick the fix shape (either is Issue-741-compliant — capability, not profile property):
  - (a) move the alloc-stats asserts behind `#[cfg(any(debug_assertions, feature = "alloc_tracking"))]` within the test module (the measurement stays where its substrate exists), or
  - (b) make the test module's alloc use unconditional by making `crate::alloc` available via the `slice_tca` feature (feature-implies-`alloc_tracking`), keeping the tests runnable in release.
- [ ] **T2** Add the lane that pins it: `cargo test --release -p katgpt-core --lib` (or the crate-wide equivalent) must COMPILE — the same shape as riir-ai's release-test compile check (Issue 869) — so the profile axis is asserted, not remembered.
- [ ] **T3** Verify both arms: `cargo test --release -p katgpt-core --lib` and `cargo test -p katgpt-core --lib` (dev) both green.

## Provenance

Found via Issue 757 T0.1 (the release profile harness died on this before any linking code compiled — E0432 in `slice_tca/tests.rs`, unrelated to the linking work).
