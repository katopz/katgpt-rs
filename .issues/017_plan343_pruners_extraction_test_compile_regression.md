# Issue 017 — Plan 343 (`katgpt-pruners` extraction) test-compile regression

**Status:** RESOLVED 2026-06-30 (same session — fix applied + verified).
**Discovered:** 2026-06-30, while re-verifying Issue 011 closure.
**Blocking:** `cargo test --lib --all-features` (and any `--all-features` test gate). The lib itself compiles clean (`cargo build --lib --all-features` ✅); only `#[cfg(test)]` modules break.
**Root-cause commit:** `d4a86187 feat: extract katgpt-pruners crate (Plan 343)`.

## TL;DR

Plan 343 extracted the pruners substrate into a new leaf crate
`katgpt-pruners`. The extraction intentionally **duplicated** the
`ComputeTier` enum (root `trigger_gate::ComputeTier` vs pruners
`thicket_variance_probe::ComputeTier`) to keep the pruners crate
leaf-clean (no dep on the root crate). Values cross the boundary as plain
`u8` via a private bridge fn `tier_to_kp` in `src/inference_router/router_tvp.rs`.

Two test-wiring mistakes slipped through (the lib passed because prod code
uses the bridge; the tests bypass it):

1. **E0433** — `src/pruners/bomber/bomber_state.rs:749` kept the pre-extraction
   import path `use crate::bomber::arena::EMPTY_ARENA;`. Post-extraction the
   module lives at `crate::pruners::bomber::arena`. The sibling
   `replay_backward.rs:233` already uses the correct relative path
   (`super::super::arena::EMPTY_ARENA`) — `bomber_state.rs` was missed.
2. **10× E0308** — `src/inference_router/router_tests.rs` calls the pruners
   `tvp_tier_decision(...)` directly (bypassing the bridge) at 10 sites but
   passes root `ComputeTier` literals. The function wants pruners `ComputeTier`.

## The three `ComputeTier` enums (audit, for the record)

Not all duplicates are the same smell. Three distinct enums exist; only #1
and #2 are involved here.

| # | Path | Variants | Role | Verdict |
|---|---|---|---|---|
| 1 | `crate::trigger_gate::ComputeTier` | `{CpuOnly, CpuGpu, CpuGpuAne}` | Inference-routing tier (root) | source of truth |
| 2 | `katgpt_pruners::thicket_variance_probe::ComputeTier` | `{CpuOnly, CpuGpu, CpuGpuAne}` | Pruners-side mirror of #1 | intentional dupe (leaf-clean), bridged via `tier_to_kp` + u8 |
| 3 | `katgpt_pruners::spec_compile::router::ComputeTier` | `{Cpu, Simd, Gpu, Ane}` | Constraint-validation backend | unrelated domain — NOT a dupe, leave alone |

#1↔#2 duplication is the documented Plan 343 trade-off (pruners is a leaf
crate consumed cross-repo; it cannot import the root `ComputeTier` without
a circular dep). The bridge pattern (`tier_to_kp` in `router_tvp.rs`) is
the contract. This is acceptable DRY-with-justification, not a smell to
"fix" by collapsing — collapsing would re-couple the leaf to root.

## Fix

### Fix 1 — `bomber_state.rs:749` (E0433, one line)

```diff
-    use crate::bomber::arena::EMPTY_ARENA;
+    use crate::pruners::bomber::arena::EMPTY_ARENA;
```

Aligns with the sibling `replay_backward.rs:233` convention.

### Fix 2 — `router_tests.rs` (10× E0308)

Promote the bridge fn `tier_to_kp` from private to `pub(crate)` so test
code reuses the **same** prod bridge (DRY: one conversion site, not a
test-local reimplementation).

Two call patterns in the test file:

- **`tvp_tier_decision_branches` (pure unit test of the pruners fn, lines
  ~701-745):** uses pruners-side semantics directly. Cleanest is to import
  the pruners `ComputeTier` under a local alias and use its literals — the
  test is exercising the pruners function in isolation, so asserting on
  the pruners-side enum is the honest framing. No bridge needed here.
- **`simulate_cascade` (cascade helper, line ~800):** returns root
  `ComputeTier` (it tests the router cascade end-to-end). The mid-cascade
  call to `tvp_tier_decision` wraps its arg with `tier_to_kp(...)` —
  mirrors exactly what `InferenceRouter::observe_tvp_decision` does in prod.

## Verification

```bash
# Lib still compiles clean (unchanged):
cargo build --lib --all-features

# Test compile now clean + Issue 011 subset runs:
cargo test --lib --all-features --no-run
cargo test --lib --all-features -- \
  iso_quant::rotation::tests::test_non_multiple_of_4 \
  speculative::flashar_anchor::tests::test_anchor_then_fill_reduces_steps \
  still_kv::integration_tests::goat_t24_compact_cache_quality \
  inference_router::router_tests \
  --test-threads=1
```

## Lessons

- **Extraction audit must cover `#[cfg(test)]` modules, not just prod.**
  Plan 343's extraction was correct for the lib (prod uses the bridge), but
  test code that called the extracted fn directly was missed. The
  `cargo check --all-features` gate catches lib compile; `cargo test
  --all-features --no-run` catches test compile. Both must be in CI.
- **Mirror the `merkle_root` / `can_freeze` lesson from sibling repos:**
  when a type is duplicated across an extraction boundary (here:
  `ComputeTier`), audit ALL call sites — prod AND test — for which side of
  the boundary each call lives on. The bridge fn is the contract; tests
  that bypass it silently break.
