# Issue 013 — Collapse the speculative/DDTree fork between katgpt-rs and riir-engine

**Date:** 2026-06-29
**Status:** Phase A complete (riir-engine fork collapsed). Root convergence = follow-up.
**Severity:** DRY violation (user rule: "DRY, Modular, Generic, Decouple")

## Problem

`riir-engine/src/{dd_tree,dflash}.rs` are local reimplementations of
`katgpt-rs/src/speculative/{dd_tree,dflash}.rs`. Improvements to one side
don't propagate to the other.

## Investigation findings (2026-06-29)

### Already DRY (Plan 008 Phase 2.5/2.6 did this)

- `katgpt-rs/src/speculative/types.rs` → **shim** re-exports `katgpt_core::speculative::types`
- `katgpt-rs/src/speculative/sampling.rs` → **shim** re-exports `katgpt_core::speculative::sampling`
- `riir-engine/src/spec_types.rs` → **shim** re-exports `katgpt_core::{traits, speculative::types}`

Types + sampling + traits are shared via `katgpt_core`. No work needed there.

### Still forked (this issue's scope)

| File | katgpt-rs | riir-engine | Notes |
|------|-----------|-------------|-------|
| `dd_tree.rs` | 6575 lines (full: core + feature-gated variants) | 2273 lines (core subset only) | riir-engine has `build_dd_tree`, `_pruned`, `_screened`, `_balanced`, `TreeBuilder`. katgpt-rs adds `_belief`, `_speculative`, `_kurtosis`, `_domino`, `_manifold`, `_lodestar`, `_gdsd`, etc. |
| `dflash.rs` | 1726 lines | 689 lines | Both call `forward`. Needs parameterization. |

### Dependency analysis

**dd_tree core functions** (the subset both sides have):
- Depend on: `katgpt_core::speculative::types::*`, `katgpt_core::traits::*`, `katgpt_types::{Config, Rng, InferenceResult}`, `rayon`
- Do NOT depend on: `forward`, sibling modules
- → **Cleanly movable to a shared leaf**

**dd_tree feature-gated variants** (katgpt-rs only):
- Depend on: `super::belief_drafter`, `super::spec_generator`, `super::kurtosis_gate`, `super::best_buddies`, `super::correlation_budget`, `super::nf_flow_budget`, `super::domino`, `crate::pruners::*`
- → **Must stay in katgpt-rs root** (they reference root-only modules)

**dflash:**
- Depends on `crate::transformer::forward` (not in katgpt-transformer leaf)
- → **Deferred to Issue 014** (needs forward trait parameterization)

## Plan

### Phase A — `katgpt-speculative` leaf (core dd_tree only)

- [x] 0. Scaffold `crates/katgpt-speculative/` workspace member ✅
- [x] 1. Move core dd_tree functions to `katgpt-speculative::dd_tree` ✅
      (24/24 tests pass — pure algorithm tests kept, dflash integration tests
      deferred to riir-engine's dflash.rs test module)
- [-] 2. katgpt-rs root `speculative/dd_tree.rs` → DEFERRED re-export.
      Root has DIVERGED with optimizations not in the leaf yet:
      - `TreeBuilder`: extra `log_marginals` cache + `cache_log_marginals()`
      - `extract_best_path_into`: two-pass, `>=` last-wins-on-tie, full f32 precision
        (leaf uses single-pass with `(score*1e6) as i64` quantization)
      - `build_inference_result`: `&str` args (leaf uses `impl Into<String>`)
      - `merge_retrieved_branches`: incremental O(D) (leaf uses fold O(D²))
      Forcing re-export now would silently lose these optimizations.
      TODO: port root's optimizations upstream into the leaf, then flip re-export on.
- [x] 3. riir-engine: delete `src/dd_tree.rs`, import from `katgpt-speculative` ✅
      (2387/2387 lib tests pass)
- [x] 4. `cargo check -p katgpt-speculative` → clean ✅
- [x] 5. `cargo test -p katgpt-speculative --lib` → 24/24 pass ✅
- [x] 6. riir-engine: `cargo check -p riir-engine` → clean, 2387 tests pass ✅

### Phase A.5 — Root convergence (follow-up)

- [-] Port root's dd_tree optimizations into `katgpt-speculative/src/dd_tree.rs`:
      log_marginals cache, incremental path reconstruction, &str args, two-pass
      extract_best_path_into. Then flip re-export on at root and delete the
      root's duplicate core definitions. After this, BOTH sides import from
      the leaf — full DRY.

### Phase B — dflash (deferred to Issue 014)

- [-] dflash needs `forward` parameterization (trait or fn pointer).
      The base `forward` signatures are identical between katgpt-rs and
      riir-engine, but the trait needs design. Separate issue.

### Phase C — KV/attention/quant re-org (deferred)

- [-] User explicitly deferred: "there's a lot to re-org there e.g. kv
      related, attention, quant — when re-group done we dry later"

## Why a new leaf (not katgpt-core)

User preference: "new crate not core for now — there's a lot to re-org
there". katgpt-core is slated for broader re-org. The new leaf keeps
speculative concerns isolated. When katgpt-core re-orgs, the leaf can
merge or stay independent.

## Why NOT promote to katgpt-core despite types being there

katgpt_core::speculative already has types + sampling. Adding the dd_tree
algorithm there is tempting (keeps namespace unified). But the user
explicitly said "not core for now". The new leaf re-exports types from
katgpt_core, so the namespace split is temporary and clean.

## IP boundary

All moved code is generic inference mechanics (textbook speculative
decoding + search tree). No game/chain/shard IP. Already public in
katgpt-rs today. Moving to a leaf changes location, not exposure.
