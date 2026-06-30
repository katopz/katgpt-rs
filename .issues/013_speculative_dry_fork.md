# Issue 013 — Collapse the speculative/DDTree fork between katgpt-rs and riir-engine

**Date:** 2026-06-29
**Status:** Complete. Phase A + Phase A.5 + Phase B done (root + riir-engine both import the shared cores). Phase C (KV/attention/quant re-org) remains user-deferred.
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
- → Resolved Phase B via a `DflashCtx` + `DflashCache` trait pair + `forward_fn` closure
  in `katgpt-speculative::dflash` (see Phase B below). The shared core is generic
  over `Ctx`, `Cache`, `W`; each consumer impls the two traits and passes its own
  `forward` (via a small `Output=()` trampoline — the shared core reads logits
  back via `DflashCtx::logits_slice` because the generic `F` bound can't return a
  borrow tied to `&mut Ctx`).

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

- [x] Port root's dd_tree optimizations into `katgpt-speculative/src/dd_tree.rs`:
      log_marginals cache, incremental path reconstruction, &str args, two-pass
      extract_best_path_into. Then flip re-export on at root and delete the
      root's duplicate core definitions. After this, BOTH sides import from
      the leaf — full DRY.

  **Completed 2026-06-29.** All four optimizations ported to the leaf;
  root's duplicate core *free functions* deleted and replaced with
  `pub use katgpt_speculative::dd_tree::*`. Root's `TreeBuilder` struct +
  methods MUST stay in the root because three feature-gated inherent methods
  (`build_screened_progressive`, `build_screened_with_depth_budgets`,
  `build_screened_recfm`) depend on root-only sibling types
  (`PositionWeightedBudget`, `CorrelationBudgetAllocator`, `CrossScaleConfig`)
  and need `&mut self` access to private fields — inherent methods cannot
  span crates. The local struct shadows the glob-reexported leaf
  `TreeBuilder`; the core free functions (`build_dd_tree`, `_pruned`,
  `_screened`, `_balanced`, `extract_best_path*`, `build_inference_result`,
  `merge_retrieved_branches`, `inject_sde_noise*`, `find_valid_sequence`,
  `par_find_valid_sequence`, `build_slices_view`) all come from the leaf.
  Tests: katgpt-speculative 24/24, katgpt-rs 3875/3875, riir-engine 2387/2387.

### Phase B — dflash (completed 2026-06-30)

- [x] Added `katgpt-speculative::dflash` module exporting two backend traits
      and three generic zero-alloc `_with` cores:
      - `trait DflashCache { reset / invalidate_position / seed_layers }`
      - `trait DflashCtx<Weights> { logits_slice / apply_mtp_conditioning }`
      - `pub fn dflash_predict_with`, `dflash_predict_ar_with`,
        `dflash_predict_conditioned_with` — generic over `Ctx, Cache, W, F`.
      Both crates impl the traits for their own `ForwardContext` /
      `MultiLayerKVCache` and delegate their `_with` bodies to the shared
      core via a disjoint field borrow on `SpeculativeContext`. The thin
      wrappers (`dflash_predict`, `_ar`, `_conditioned`, `_parallel`) and
      feature-gated variants (`_domino`, `_routing`, `_fusion` in katgpt-rs)
      stay local.
      **Free win:** the shared core carries the Issue 053 selective
      `invalidate_position` optimization, which riir-engine previously
      lacked — riir-engine gains it transparently (`test_*_matches_original`
      confirms identical marginals).

      **Verification:**
      - `katgpt-speculative`: 24/24 lib tests pass
      - `katgpt-rs`: 659/659 speculative lib tests pass (incl. 23 dflash)
      - `riir-engine`: 24/24 dflash tests pass (incl. 3 `*_matches_original`)
      - `cargo check --workspace` clean on both repos (default features)

      **Design notes:**
      - katgpt-rs needed a `CacheAdapter<'a>` newtype to satisfy the orphan
        rule (`MultiLayerKVCache` is foreign — defined in
        `katgpt-transformer`). riir-engine could impl directly because its
        `MultiLayerKVCache` is local.
      - Both crates needed a tiny `forward_void`/`forward_via_adapter`
        trampoline because `forward` returns `&mut [f32]` but the shared
        core's `F: Fn(...) -> ()` bound pins the output. The trampoline
        discards the return; the shared core reads logits via
        `DflashCtx::logits_slice` (verified both crates' `forward` return
        `&mut ctx.logits`).

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
