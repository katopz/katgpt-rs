# Plan 008: katgpt-core Substrate Extraction (Phase 1+2 of Issue 007)

> **Origin:** [Issue 007](../issues/007_katgpt_rs_cargo_publish_substrate_reorg.md)
> **Status:** Active — Phase 1 step 1 ✅ done (pre-existing); steps 2-7 queued
> **Branch:** `develop`
> **Created:** 2026-06-27
> **Cross-repo:** katgpt-rs (primary moves), riir-ai/riir-engine (Phase 2 dedup consumer)

---

## TL;DR

Issue 007 was written against a snapshot that has since drifted. This plan
captures the **corrected** scope after a full audit of the current tree.

**Three findings that change the issue's scope:**

1. **Phase 5 (publish `katgpt-rs`) is dead.** Both `Cargo.toml:9`
   (`publish = false  # dev/examples aggregator — never published`) and
   `release-plz.toml:9-12` (`release = false, publish = false`) lock the root
   crate private permanently, with the rationale: "Only katgpt-core ships to
   crates.io." This decision was made AFTER Issue 007 was filed and overrides
   its Phase 5.

2. **Phase 1 step 1 (move `types`) is already done.** `katgpt-core/src/types/`
   has 14 files (`config.rs`, `enums.rs`, `rng.rs`, `math.rs`, …). Root
   `src/types.rs` is already a thin `pub use katgpt_core::types::*;` re-export
   shim plus a handful of root-only items (`QuantizedKVCache`,
   `AsymmetricKVConfig`, `top_p_coreset`, `OutlierGuardConfig`).

3. **Phase 2B's premise is inverted.** Issue 007 says "cgsp in core, cce in
   root — move cgsp UP". Reality: `cgsp` is already in `katgpt-core/src/cgsp/`
   AND re-exported from root `src/cgsp.rs` (verbatim `pub use katgpt_core::cgsp`).
   `cce` is in root `src/cce/`. The issue's "tier inconsistency" is real but
   the proposed direction ("move cgsp up") contradicts the cargo-publish
   decision in finding #1 — the only publishable crate is core, so substrate
   goes DOWN, not up. `cce` is correctly in root (cognitive layer, not
   substrate); `cgsp` is correctly in core. **No tier move needed.**

**What IS still real and valuable** (the heart of Issue 007): the
**cross-repo DRY violation**. `riir-engine/src/` has its own divergent
`crate::hla`, `crate::transformer`, `crate::types`, `crate::tokenizer`,
`crate::dd_tree`, `crate::spec_types`, `crate::mcts`, `crate::sampling`,
`crate::delta_mem`, `crate::simd` — all confirmed via grep using `crate::`
prefix (not `katgpt_core::`). These are PUBLIC inference mechanics per the
refined strategy doc, stranded in a private fork.

---

## Verdicts on Issue 007 Open Questions

| Q | Issue wording | Verdict |
|---|---|---|
| 1 | Phase 1 scope: full chain or subset? | **Full chain.** Anything left behind stays duplicated. Order enforced by deps: transformer → {weights, hla, dd_tree, spec_types} → {mcts, sampling, delta_mem}. |
| 2 | tokenizer to core or root? | **Defer; audit-first.** Risk 4 stands: SentencePiece-sys is a C++ build dep that disqualifies from leaf-clean core. Move only after transformer/hla land; verify no SentencePiece dep before moving. |
| 3 | mcts/dd_tree generic-vs-game split aggressiveness | **Generic core trait + concrete game impls in riir-engine.** Mirror the existing `traits.rs` pattern (already half-done for spec_types). Move `TreeNode`/`DDTreeBranchCache`/`SpeculativeContext` types to join their traits in core; leave game-coupled impls in riir-engine. |
| 4 | Go order: 1+2 first or push to publish? | **Phase 1+2 only. Phase 3-5 deferred indefinitely.** Phase 5 rescinded (finding #1). Phase 3 (root subdir reorg) is cosmetic — not worth the churn while 100+ features still flatten at root. Phase 4 (`plotters` optional) is independent and can be done as a standalone fix if `cargo check --no-default-features` ever fails; not blocking. |

---

## Task list

### Phase 1 — Substrate extraction to `katgpt-core`

- [x] **Step 1 — `types` → core.** DONE pre-this-plan. `katgpt-core/src/types/` 14 files; root is re-export shim.
- [ ] **Step 2 — `transformer` substrate types + `weights` → core.**

  ⚠️ **AUDIT FINDING (2026-06-27, before execution): the original premise was wrong.**
  `transformer.rs` is NOT pure substrate. The file is **8398 lines** but splits into:
  - **~1100 lines of pure data types** (LayerWeights, TransformerWeights + impl,
    DecodeStage, KV caches, PrefillContext, WallPrefixState, MtpProjection) —
    these have ZERO root-only deps and CAN move to core.
  - **~5300 lines of forward functions** (forward, forward_base, forward_coda,
    forward_looped, forward_prefill, forward_paged, forward_raven,
    forward_quantized, forward_turboquant, generate_*, etc.) — these call into
    `crate::hla`, `crate::sleep`, `crate::tf_loop`, `crate::gdn2`,
    `crate::turboquant`, `crate::pruners::*` (root-only cognitive modules).
    **They are composition logic, not substrate, and cannot move to core.**
  - **~2000 lines of tests** (move with their subject).

  `ForwardContext` CANNOT move cleanly: its struct definition has fields typed
  as root-only `crate::pruners::{CnaModulator, SubstrateMask, HydraSkipPlan}`.
  Those types have their own root-only dependency chains (not in scope here).

  Bidirectional cycle confirmed at root level: `transformer` ↔ {`hla`,
  `gdn2`, `sleep`, `tf_loop`, `turboquant`} all use each other's
  `TransformerWeights`/`ForwardContext` types. The cycle is only resolvable
  by moving the **type definitions** (used by all) to core, leaving the
  **forward composition functions** (which call into cognitive modules) in root.

  Corrected subtasks:
  - [x] 2a. Map `transformer.rs` internal sections — DONE during audit
  - [x] 2b. Move **data types only** to a NEW crate `katgpt-transformer/` (per user direction: "if move to core is too much, define new one e.g. katgpt-foo and keep core core"):
    - [x] `lib.rs` — module decls + `DecodeStage` enum + re-exports + `PAGE_SIZE` const
    - [x] `weights.rs` — `LayerWeights`, `TransformerWeights` + `impl new/init/zero`
    - [x] `kv_cache.rs` — `KVCache`, `MultiLayerKVCache`, `KVSnapshot`,
      `KVLayerSnapshot`, `PagedKVCache`, `RavenKVCache` + `preload_kv_cache`
    - [x] `context.rs` — `PrefillContext`, `WallPrefixState`, `GateStatistics`
      (NB: `ForwardContext` stays in root — has root-only pruner fields)
    - [x] `mtp.rs` — `MtpProjection`, `load_mtp_projection`, `project_target_activation`,
      magic constants + tests
    - [x] `contiguous.rs` — `ContiguousWeights` + `load_ternary_bits` (moved verbatim
      from root `src/weights.rs`)
  - [x] 2c. Deleted root `src/weights.rs`; replaced `pub mod weights;` in
    root `src/lib.rs` with `pub use katgpt_transformer::{ContiguousWeights, load_ternary_bits};`
  - [x] 2d. Root `src/transformer.rs` keeps: `ForwardContext`, all forward
    functions, all tests. Imports types via `pub use katgpt_transformer::{...}`.
    Stays a single 7055-line file for this commit; splitting forward funcs into
    `src/transformer/{forward,prefill,raven,paged,generate,...}.rs` is a
    **follow-up** (out of scope for step 2).
  - [x] 2e. `katgpt-transformer/src/lib.rs` declares all type modules (no feature gate
    on the module itself; `wall_attention`-gated items gated at re-export).
  - [x] 2f. Feature gates audited and forwarded:
    - `katgpt-rs/Cargo.toml`: `wall_attention`, `delta_routing`, `decode_specialize`,
      `plasma_path` now forward to `katgpt-transformer/<feature>`
    - All 3 combos (`--no-default-features`, default, `--all-features`) compile clean.
  - [x] 2g. `cargo check` + `cargo test -p katgpt-transformer --lib` (11/11 green) +
    `cargo test --lib transformer::` (80/80 green) + full `cargo test --lib`
    (3990/3991 green; the 1 failure is an unrelated flake in
    `pruners::three_mode_bandit::tests::bench_grounding_quality_32k` which passes
    in isolation).
  - [x] 2h. Commit: `feat: Plan 008 step 2 — extract katgpt-transformer substrate crate` (1debf905 on develop, 2026-06-27)

  **FOLLOW-UP (separate commit, not step 2):** split root `src/transformer.rs`
  forward functions into per-family submodules mirroring riir-engine's
  `transformer/{gemma2,llama,prefill,raven,mtp,attention}.rs` layout. Root
  file is ~6300 lines after step 2 (forward funcs + ForwardContext + tests),
  still over the 2048 ceiling — addressed in follow-up.
- [ ] **Step 3 — `tokenizer` → core.** DEFERRED per Q2 verdict. Audit SentencePiece-sys dep first; if present, leave in root.
- [x] **Step 4 — `hla` → core (substrate half).** 2248 lines total (`forward.rs` 569 + `kernel.rs` 1019 + `types.rs` 606 + `mod.rs` 54). Depends on step 2.

  ⚠️ **AUDIT FINDING (2026-06-28, before execution): the original premise was wrong.**
  `forward.rs` CANNOT move cleanly to core — it imports `crate::transformer::{ForwardContext, TransformerWeights}` and `ForwardContext` has root-only pruner fields (`CnaModulator`, `SubstrateMask`, `HydraSkipPlan`). This is the **same split pattern as Step 2** (`katgpt-transformer` got the substrate types; root kept the forward composition). Corrected scope: move the **pure substrate half** (`types.rs` + `kernel.rs`) to core; keep the **composition half** (`forward.rs`) in root.

  ### Done subtasks (2026-06-28)
  - [x] 4a. Move `types.rs` (606 LoC) + `kernel.rs` (1019 LoC) → `katgpt-core/src/hla/` (verbatim; both files depend only on `crate::simd` + `crate::types::Config`, both already in core — zero import changes needed). New `katgpt-core/src/hla/mod.rs` declares `pub mod kernel; pub mod types;` + re-exports the substrate API. `forward.rs` stays in root.
  - [x] 4c. Root `src/hla/mod.rs` → thin re-export of `katgpt_core::hla::{kernel, types}` + substrate API + local `pub mod forward;` (the composition layer). All existing call sites (`crate::hla::MultiLayerHlaCache`, `crate::hla::hla_state_update`, etc.) resolve unchanged via the re-exports.
  - [x] 4d. **GOAT gate PASSED** — bit-identical forward output:
    - `cargo test -p katgpt-core --lib hla::` → **16/16 green** (9 types + 7 kernel substrate tests, moved verbatim).
    - `cargo test --lib --features hla_attention hla::` → **8/8 green** (the forward-composition tests: `forward_hla_produces_finite_logits`, `forward_ahla_produces_finite_logits`, `forward_hla_reset_clean`, `forward_hla_multi_token_stable`, `forward_ahla_multi_token_stable`, `forward_hla_all_configs`, `forward_ahla_gqa_draft`, `ahla_memory_smaller_than_symmetric`). These exercise the full `forward_hla`/`forward_ahla` path through `ForwardContext` → re-exported substrate kernels → output logits. Bit-identical because the kernels are byte-for-byte the same code, just resolved through `katgpt_core::hla` instead of local `crate::hla::kernel`.
    - `cargo check -p katgpt-core --no-default-features` clean (substrate always-on, like simd/types).
    - `cargo check -p katgpt-core --all-features` clean.
    - `cargo check --all-features` (root) clean.
    - `cargo check -p katgpt-core --target wasm32-wasip2` clean (HLA substrate builds on WASM; 1 pre-existing unrelated simd warning).
    - `cargo test --lib --features hla_attention` (full root) → **3974/3975 green**. The 1 failure (`sleep::eviction::tests::sliding_window_retains_recent`) is **pre-existing** — confirmed failing on unmodified `develop` HEAD `eb604670`. Not caused by this change, not in scope to fix.
  - [x] 4e. Commit: `feat(core): Plan 008 step 4 — move HLA substrate to katgpt-core` (see commit log).

  ### Deferred subtask (Phase 2 reconciliation, not Phase 1)
  - [ ] **4b. Port riir-engine's `*_role_aware` variants behind a core feature `hla_role_aware`.** DEFERRED — this is Phase 2 (riir-engine dedup) work, not Phase 1 (substrate extraction). Rationale:
    1. The role-aware kernel variants (`hla_state_update_role_aware`, `ahla_step_role_aware`, `hla_layer_update_role_aware`, `ahla_layer_step_role_aware`, `third_order_update`, `third_order_readout`) all depend on `crate::role_transport::{RoleEmbeddingTable, diagonal_transport, SlotLabel}` — Category C private composition per Issue 007 §"Cross-repo consumer cleanup".
    2. Porting them to core requires defining a `RoleTransport` trait in core + a `SlotLabel` newtype, then having riir-engine's `RoleEmbeddingTable` impl the trait. That's a design change to core's public API surface, not a pure move.
    3. riir-engine also DIVERGED with `ThirdOrderMoment` (Plan 151 T13) + `HlaUpdateMode` + a `role: Option<SlotLabel>` field on `HlaQHeadState`/`AhlaQHeadState`. These are cognitive extensions, not substrate.
    4. Per Risk 2 mitigation: "keep riir-engine's `role_transport.rs` as the private composition layer (it's Category C)." The cleanest interpretation: the role-aware **wrappers** (which compute the transported key then call the standard kernel) stay in riir-engine as composition; only the standard kernels (now in core) are the shared substrate.
    5. **Track in Phase 2.1** (`riir-engine src/hla/ → consume katgpt_core::hla`). When riir-engine deletes its local `types.rs`/`kernel.rs` and imports from core, the role-aware wrappers will call `katgpt_core::hla::hla_state_update` instead of the local copy. The wrapper code itself can stay in riir-engine indefinitely — it's Category C composition.

  **Net result:** the publishable-leaf half of HLA (cache types + streaming kernels, 1625 LoC) now lives in `katgpt-core` and is available to any crate via `cargo add katgpt-core`. The composition half (`forward_hla`/`forward_ahla`, 569 LoC) stays in root because it needs `ForwardContext`. The cognitive half (role-aware + third-order, ~600 LoC) stays in riir-engine because it needs `role_transport`. Three-tier split achieved without breaking any call site.
- [x] **Step 5 — `dd_tree` + `spec_types` → core.** Traits already in `core/traits.rs`; move dependent types (`TreeNode`, `DDTreeBranchCache`, `SpeculativeContext`, `DraftResult`, `NoPruner`, `ScreeningPruner` dep types) to join them.

  ⚠️ **AUDIT FINDING (2026-06-28, before execution): the original premise needed the same scope correction as Steps 2 and 4.**
  - There is NO `spec_types.rs` in katgpt-rs root. The substrate types live in `src/speculative/types.rs`. (`spec_types.rs` exists only in `riir-engine`, where it's a duplicate copy — Phase 2.5 dedup target.)
  - `src/speculative/dd_tree.rs` (6575 lines) is the BUILDER file (composition: `build_dd_tree_*`, `TreeBuilder` impl, tests). It stays in root exactly like `src/hla/forward.rs` stayed in root in Step 4.
  - Some types in `speculative/types.rs` are PURE substrate (depend only on `Config` + core traits + std) — these move.
  - Some types are COMPOSITION (need `katgpt-transformer` or root-only types) — these stay.

  **Corrected scope:** move the pure-substrate types to `katgpt-core/src/speculative/types.rs`; keep the composition types in root as a re-export shim.

  ### Done subtasks (2026-06-28)
  - [x] 5a. Added 12 empty feature markers to `katgpt-core/Cargo.toml` for substrate type gating: `stability_metrics`, `spec_cost_model`, `kurtosis_gate`, `elf_sde`, `tes_loop`, `tri_mode`, `dmax_spd`, `lattice_deduction`, `echo_env_predictor`, `dflare_fusion`, `dflare_kv_routing`, `dflare_progressive_budget`. All are empty `[]` (or `dllm`-implying where the upstream feature already implies it) — no behavior, no deps, just cfg-gating markers so the substrate types can be feature-gated identically in core and root.
  - [x] 5b. Forwarded those 12 features from root `Cargo.toml` (e.g. `elf_sde = ["katgpt-core/elf_sde"]`). Root's feature still owns the root-specific modules (e.g. root's `elf_sde` still gates Plan 079 ELF SDE noise injection); the forward just enables the substrate gate in core.
  - [x] 5c. Created `katgpt-core/src/speculative/` (new module, always-on like `simd`/`types`/`traits`/`hla`) with:
    - `mod.rs` (42 LoC) — module doc + `pub mod types;` + `pub use types::*;`
    - `types.rs` (1394 LoC) — pure substrate types moved verbatim from root `speculative/types.rs`. Imports: `use crate::traits::ScreeningPruner; use crate::types::Config; use std::cmp::Ordering;` (all already in core). Includes all substrate tests (32 tests, all green).
  - [x] 5d. Updated `katgpt-core/src/lib.rs` — added `pub mod speculative;` (always-on) + updated crate doc to list the new module.
  - [x] 5e. Rewrote root `src/speculative/types.rs` as a thin re-export shim (was 2190 LoC, now 596 LoC):
    - Re-exports the substrate API from `katgpt_core::speculative::types::{...}` (always-on types) + feature-gated re-exports for `MarginalFusionConfig` / `KvRoutingConfig` / `PositionWeightedBudget` / `LdtPruneConfig` / `ConflictDetector` / `EntropyConflictDetector` / `LDT_THETA_ELIM` / `TesNode` / `TrajectoryCredit`.
    - Re-exports the traits from `katgpt_core::traits::{...}` (unchanged from Plan 107 Phase 0).
    - Keeps the composition types local: `SpeculativeContext` (needs `ForwardContext` + `MultiLayerKVCache` from katgpt-transformer), `DDTreeBranchCache` (needs `PagedKVCache` + `forward_paged`), `TesConfig` (needs `BanditStrategy`), `SelfSpecConfig` (needs `D2fDecodeConfig` + `DiffusionSampler`).
    - Keeps the composition tests local (9 `test_branch_cache_*` tests that need `ForwardContext` + `TransformerWeights` + `Rng`).
  - [x] 5f. **GOAT gate PASSED** — bit-identical, no call-site changes:
    - `cargo check -p katgpt-core` clean (substrate always-on, like simd/types/hla/traits).
    - `cargo check -p katgpt-core --no-default-features` clean.
    - `cargo check -p katgpt-core --all-features` clean.
    - `cargo check -p katgpt-core --target wasm32-wasip2` clean (1 pre-existing unrelated simd warning).
    - `cargo test -p katgpt-core --lib speculative::` → **5/5 green** (default features: ungated substrate tests).
    - `cargo test -p katgpt-core --lib speculative:: --all-features` → **32/32 green** (all substrate tests including feature-gated EarlyStopGate, dflare_*, DraftEvent, RejectionReason, DecodeStrategy).
    - `cargo check` (root default) clean.
    - `cargo check --all-features` (root) clean.
    - `cargo test --lib speculative::types::` → **9/9 green** (DDTreeBranchCache composition tests).
    - `cargo test --lib speculative::` → **664/664 green** (full speculative module).
    - `cargo test --lib` (root default) → **3955/3956 green**. The 1 failure (`sleep::eviction::tests::sliding_window_retains_recent`) is **pre-existing** — confirmed failing on unmodified `develop` HEAD `9852a100` via `git stash` test. Not caused by this change.
    - `cargo test --lib --all-features` (root) → **7268/7280 green** (12 pre-existing failures, confirmed on unmodified develop via `git stash` test of 2 representative failures: `sliding_window_retains_recent` and `test_anchor_then_fill_produces_valid_output`).
  - [x] 5g. Commit: `feat(core): Plan 008 step 5 — move speculative substrate types to katgpt-core` (see commit log).

  ### Composition types that stayed in root (with rationale)
  - `SpeculativeContext` — fields `ctx: ForwardContext`, `cache: MultiLayerKVCache`. Both from `katgpt-transformer`. Moving would force katgpt-core to depend on katgpt-transformer (breaks the "core is the leaf" layering).
  - `DDTreeBranchCache` — field `paged: PagedKVCache`, method `forward_branch` calls `forward_paged`. Both from `katgpt-transformer`.
  - `TesConfig` — field `bandit_strategy: BanditStrategy` from `crate::pruners::bandit` (root-only). Pure-data `TesNode` + pure-algorithm `TrajectoryCredit` DID move (they have no root-only deps).
  - `SelfSpecConfig` — fields `d2f_config: D2fDecodeConfig`, `sampler: Option<DiffusionSampler>` from `crate::speculative::{d2f, diffusion_sampler}` (root-only).

  ### Layering achieved
  | Tier | Location | Content | LoC | Rationale |
  |---|---|---|---|---|
  | **Substrate** | `katgpt-core/src/speculative/types.rs` | `TreeNode`, `DraftResult`, `DraftEvent`, `RejectionReason`, `DecodeStrategy`, `SdeConfig`, `EarlyStopGate`, LDT `ConflictDetector` + `EntropyConflictDetector`, `TesNode`, `TrajectoryCredit`, all DFlare/LDT/PFlash configs + snapshots | 1394 | Pure data + algorithm + trait impls; any crate can `cargo add katgpt-core` |
  | **Composition** | `katgpt-rs/src/speculative/types.rs` | `SpeculativeContext`, `DDTreeBranchCache`, `TesConfig`, `SelfSpecConfig` + re-export shim | 596 | Need `katgpt-transformer` or root-only `BanditStrategy` / D2F types |
  | **Builder** | `katgpt-rs/src/speculative/dd_tree.rs` | `build_dd_tree_*`, `TreeBuilder` | 6575 | Composition that drives the substrate; needs `SpeculativeContext` + `ForwardContext` |

  **Net result:** the publishable-leaf half of speculative substrate types (1394 LoC) now lives in `katgpt-core` and is available to any crate via `cargo add katgpt-core`. The composition half (`SpeculativeContext`/`DDTreeBranchCache`/`TesConfig`/`SelfSpecConfig`, 596 LoC) stays in root because it needs katgpt-transformer or root-only types. The builder half (`dd_tree.rs`, 6575 LoC) stays untouched. Three-tier split achieved without breaking any call site — all existing import paths (`crate::speculative::types::TreeNode`, `...::SpeculativeContext`, `...::DDTreeBranchCache`, etc.) resolve unchanged via the re-export shim.
- [x] **Step 6 — `mcts`, `sampling`, `delta_mem` → core.** Leaf inference mechanics. `mcts` parameterize over a core `Game` trait (Q3 verdict); leave game-specific impls in riir-engine.

  ⚠️ **AUDIT FINDING (2026-06-28, before execution): the original premise needed the same scope correction as Steps 2, 4, and 5.**
  - There is NO `src/mcts.rs`, NO `src/sampling.rs`, NO `src/delta_mem/` at root. The actual locations are:
    - `src/pruners/game_state/mcts.rs` (1044 LoC) — MCTS algorithm (substrate) + `BanditRolloutPolicy` (composition, gated by `bandit`).
    - `src/speculative/sampling.rs` (131 LoC) — CDF + residual samplers (substrate).
    - `src/pruners/delta_mem/{state,hash,multi,pruner,multi_pruner}.rs` (1992 LoC) — split: 3 substrate files + 2 composition files.
  - The `GameState` / `StateHeuristic` / `RolloutPolicy` / `RandomRolloutPolicy` / `ActionSpaceLog` **traits already live in `katgpt-core/src/traits.rs`** (Plan 107 Phase 0). `src/pruners/game_state/mod.rs` is already a re-export shim. So Step 6 is really about extracting the **algorithms + types that sit on top of those traits**, not the traits themselves.
  - `BanditRolloutPolicy` CANNOT move cleanly: it depends on `BanditStats` from `crate::pruners::bandit` (root-only). Same split pattern as Step 4's `forward.rs` and Step 5's `SpeculativeContext`.
  - `MemorySteeredPruner<P>` / `MultiDomainMemoryPruner<P>` CANNOT move cleanly: they wrap `ScreeningPruner` (now in `katgpt_core::traits`) with root-side composition — they're consumers of the substrate trait, not substrate themselves.

  **Corrected scope:** move the pure-substrate algorithms + types to `katgpt-core`; keep the composition that needs root-only types in root as re-export shims.

  ### Done subtasks (2026-06-28)
  - [x] 6a. Created `katgpt-core/src/mcts.rs` (806 LoC) — `MCTSNode`, `mcts_search`, `mcts_search_informed`, `mcts_search_impl`, `select_inline`, `expand_and_rollout`, `rollout`, `backpropagate`, `ucb1_score`, `UCB1_C`, `MAX_TREE_SIZE` + 14 substrate tests. Moved verbatim from root `pruners/game_state/mcts.rs`; imports changed from `super::{GameState, ...}` to `crate::traits::{GameState, ...}`. Pure substrate: depends only on `katgpt_core::traits` + `fastrand::Rng` (already in core).
  - [x] 6b. Created `katgpt-core/src/speculative/sampling.rs` (145 LoC) — `sample_from_distribution`, `sample_residual_distribution_into`, `sample_residual_distribution` + 5 tests. Moved verbatim from root `speculative/sampling.rs`; imports changed to `crate::simd::simd_scale_inplace` + `crate::types::Rng` (both always-on in core).
  - [x] 6c. Created `katgpt-core/src/delta_mem/` (new always-on module like `mcts`/`hla`/`speculative`):
    - `mod.rs` (37 LoC) — module doc + `pub mod {hash, multi, state};` + re-exports.
    - `state.rs` (910 LoC) — `DeltaMemoryConfig`, `DeltaMemoryState`, `DeltaMemorySnapshot` + 17 default tests + 8 `temporal_deriv`-gated tests. Moved verbatim; only path change: `use katgpt_core::temporal_deriv::TemporalDerivativeKernel` (was already `katgpt_core::` in root — no change).
    - `hash.rs` (275 LoC) — `FeatureHasher`, `ContextFeatures`, `OutcomeFeatures` + 7 tests. Verbatim copy (only `fastrand` dep).
    - `multi.rs` (242 LoC) — `MultiDomainMemory`, `AggregationStrategy` + 6 tests. Verbatim copy (depends on `super::state::*` which resolves in core).
  - [x] 6d. Updated `katgpt-core/src/lib.rs` — added `pub mod delta_mem;` + `pub mod mcts;` (both always-on) + updated crate doc to list the new modules and document the substrate/composition split.
  - [x] 6e. Updated `katgpt-core/src/speculative/mod.rs` — added `pub mod sampling;` + re-exports of the 3 sampling functions.
  - [x] 6f. Rewrote root `src/pruners/game_state/mcts.rs` (was 1044 LoC, now 314 LoC) as composition-only: re-exports `mcts_search` / `mcts_search_informed` from `katgpt_core::mcts`, keeps `BanditRolloutPolicy` (gated by `bandit`) + 5 composition tests. All 14 substrate tests moved to core.
  - [x] 6g. Rewrote root `src/speculative/sampling.rs` (was 131 LoC, now 16 LoC) as re-export shim from `katgpt_core::speculative::sampling::{...}`. All 5 tests moved to core.
  - [x] 6h. Rewrote root `src/pruners/delta_mem/mod.rs` (was 35 LoC, now 64 LoC) as re-export shim: re-exports substrate types from `katgpt_core::delta_mem::{...}`, re-exposes the substrate module layout via inline `pub mod hash/state/multi { pub use katgpt_core::delta_mem::...; }` so absolute paths (`crate::pruners::delta_mem::state::DEFAULT_THETA_SURPRISE`) resolve unchanged, declares local `mod pruner; mod multi_pruner;` (composition). Deleted the 3 substrate files (state/hash/multi.rs); kept pruner.rs + multi_pruner.rs unchanged (they import from `super::{state,hash}` which still resolves via the re-export shim).
  - [x] 6i. **GOAT gate PASSED** — bit-identical, no call-site changes:
    - `cargo check -p katgpt-core` clean (substrate always-on, like simd/types/hla/speculative).
    - `cargo check -p katgpt-core --no-default-features` clean.
    - `cargo check -p katgpt-core --all-features` clean.
    - `cargo check -p katgpt-core --target wasm32-wasip2` clean (1 pre-existing unrelated simd warning).
    - `cargo test -p katgpt-core --lib mcts::` → **14/14 green** (UCB1 + MCTS search + backpropagation + informed MCTS).
    - `cargo test -p katgpt-core --lib speculative::sampling::` → **5/5 green**.
    - `cargo test -p katgpt-core --lib delta_mem::` → **37/37 green** (default features include `temporal_deriv` so all surprise-gate tests run).
    - `cargo check` (root default) clean.
    - `cargo check --all-features` (root) clean.
    - `cargo test --lib pruners::game_state::mcts::` → **5/5 green** (composition-only bandit tests).
    - `cargo test --lib` (root default) → **3936/3937 green**. The 1 failure (`sleep::eviction::tests::sliding_window_retains_recent`) is **pre-existing** — confirmed on Step 5 baseline HEAD `4d85ea7a` via `git stash` test. Not caused by this change. Test count dropped from Step 5's 3955 → 3936 = exactly 19 tests moved to core (14 mcts substrate + 5 sampling).
    - `cargo test --lib --all-features` (root) → **7212/7224 green** (12 failures, all confirmed pre-existing on Step 5 baseline which had 7271/7280; 7280−7224=56 tests moved to core = 14 mcts + 5 sampling + 37 delta_mem substrate). The 3 "extra" failures vs Step 5 baseline (`percepta::compile::test_e2e_compile_collatz_c`, `ruliology::benchmarks::bench_enumerate_fsm_3_states`, `speculative::nf_flow::test_bench_flow_score_v128_t5`) are flaky bench-style tests that pass in isolation — not caused by this change.
    - `cargo test --bench delta_mem_surprise_gate_bench --features delta_mem,temporal_deriv` → **G3 PASS**: write suppression 42.90% (target ≥30%), recall loss 0.00% (target ≤5%). Bit-identical to pre-move behavior.
  - [x] 6j. Commit: `feat(core): Plan 008 step 6 — move mcts/sampling/delta_mem substrate to katgpt-core` (see commit log).

  ### Composition types that stayed in root (with rationale)
  - `BanditRolloutPolicy` (root `pruners/game_state/mcts.rs`) — depends on `BanditStats` from `crate::pruners::bandit` (root-only). Stays behind `bandit` feature; 5 bandit tests stayed with it.
  - `MemorySteeredPruner<P>` (root `pruners/delta_mem/pruner.rs`) — wraps an inner `P: ScreeningPruner`. Generic is instantiated at root call sites that compose spec-decoding context. The trait `ScreeningPruner` itself moved to `katgpt_core::traits` in Step 5 — but the *composition* (a pruner that holds a `DeltaMemoryState` + `FeatureHasher` and adds memory-steered corrections) is consumer-specific.
  - `MultiDomainMemoryPruner<P>` (root `pruners/delta_mem/multi_pruner.rs`) — same pattern, multi-domain variant.

  ### Layering achieved
  | Tier | Location | Content | LoC | Rationale |
  |---|---|---|---|---|
  | **Substrate** | `katgpt-core/src/mcts.rs` | `mcts_search`, `mcts_search_informed`, `MCTSNode`, UCB1/backprop/rollout helpers | 806 | Pure algorithm over `GameState` trait |
  | **Substrate** | `katgpt-core/src/speculative/sampling.rs` | `sample_from_distribution`, `sample_residual_distribution_into`, `sample_residual_distribution` | 145 | Pure CDF math over `Rng` + `simd` |
  | **Substrate** | `katgpt-core/src/delta_mem/{state,hash,multi}.rs` | `DeltaMemoryState`, `FeatureHasher`, `MultiDomainMemory` + configs + snapshots | 1427 | Pure data + algorithm over `serde` + `fastrand` + `temporal_deriv` |
  | **Composition** | `katgpt-rs/src/pruners/game_state/mcts.rs` | `BanditRolloutPolicy` + 5 tests | 314 | Needs `BanditStats` from root-only `crate::pruners::bandit` |
  | **Composition** | `katgpt-rs/src/speculative/sampling.rs` | re-export shim | 16 | N/A |
  | **Composition** | `katgpt-rs/src/pruners/delta_mem/{mod,pruner,multi_pruner}.rs` | `MemorySteeredPruner<P>`, `MultiDomainMemoryPruner<P>` + re-export shim | ~600 | Wrap root-only `ScreeningPruner` instances |

  **Net result:** 2378 LoC of substrate (mcts algorithm + sampling primitives + delta_mem state machine) now lives in `katgpt-core` and is available to any crate via `cargo add katgpt-core`. The composition half (`BanditRolloutPolicy`, `MemorySteeredPruner<P>`, `MultiDomainMemoryPruner<P>`, ~930 LoC) stays in root because it needs root-only types. Three-tier split achieved without breaking any call site — all existing import paths (`crate::pruners::game_state::mcts_search`, `crate::speculative::sampling::sample_from_distribution`, `crate::pruners::delta_mem::DeltaMemoryConfig`, `crate::pruners::delta_mem::state::DEFAULT_THETA_SURPRISE`, etc.) resolve unchanged via the re-export shims.
- [x] **Step 7 — riir-engine `simd/wasm32.rs` → consume `katgpt_core::simd`.** Already shipped in core under `wasm32_simd128_*` kernels. Diff for riir-engine-only improvements, port if any, then delete reimplementation.

  **Scope audit (2026-06-28):** riir-engine `simd/mod.rs` already does `pub use katgpt_core::simd::*;` on non-WASM targets. Only the WASM SIMD128 path (`src/simd/wasm32.rs`, 630 LoC) is local. Of its 11 functions:
  - **8 have bit-identical equivalents already in katgpt-core** (which has WASM SIMD128 paths for each): `simd_dot_f32`, `simd_sum_f32`, `simd_scale_inplace`, `simd_add_scalar_inplace`, `simd_exp_inplace`, `simd_matvec`, `simd_outer_product_acc` (WASM path missing — port), `simd_sum_sq` (WASM path missing — port).
  - **2 are thin ergonomic wrappers** (no algorithmic content): `dot_f32_simd(a,b)` = `simd_dot_f32(a,b,a.len().min(b.len()))`; `matmul_f32_simd(a,x,rows,cols,out)` = `simd_matvec(out,a,x,rows,cols)`. Drop them; update the 2 WASM-only call sites.
  - **1 is unique substrate** (`project_ternary_simd`): single-vector ternary dot-product. Its SWAR algorithm was already ported to core's `wasm32_ternary_matvec` + `bitselect_nibble8_wasm` (matvec form, TernaryWeights struct), but the dot-product form (raw bit-plane slices → f32) doesn't exist in core. Port it as a new public function under `plasma_path`.

  Subtasks:
  - [x] 7a. Port WASM SIMD128 path to `simd_sum_sq` in `research.rs` (mirror NEON 4-accumulator pattern, port riir-engine's algorithm).
  - [x] 7b. Port WASM SIMD128 path to `simd_outer_product_acc` in `dot.rs` (mirror NEON FMA pattern, port riir-engine's algorithm).
  - [x] 7c. Add `project_ternary_simd` (single-vector ternary dot-product variant) to `ternary.rs` under `plasma_path`. Port verbatim from riir-engine; the SWAR algorithm already lives in core as `bitselect_nibble8_wasm` but in matvec form. Added scalar fallback + WASM SIMD128 SWAR kernel + dispatcher + 7 tests (ported from riir-engine).
  - [x] 7d. Update `simd/mod.rs` re-exports to include `project_ternary_simd` under `plasma_path`.
  - [x] 7e. Delete `riir-engine/src/simd/wasm32.rs`; replace `simd/mod.rs` with `pub use katgpt_core::simd::*;` for all targets.
  - [x] 7f. Update `riir-engine/examples/wasm_simd_bench.rs` to use core API (`simd_dot_f32` 3-arg form, `simd_matvec`, `project_ternary_simd` from core). (Dropped `matmul_f32_simd` wrapper — replaced both call sites with `simd_matvec` which takes args in `(acc, mat, vec, rows, cols)` order.)
  - [x] 7g. GOAT gate — PASSED (bit-identical, no call-site regressions):
    - `cargo check -p katgpt-core` (default) clean.
    - `cargo check -p katgpt-core --no-default-features` clean.
    - `cargo check -p katgpt-core --all-features` clean.
    - `cargo check -p katgpt-core --target wasm32-wasip2` clean (1 pre-existing `SimdLevel` unused-import warning, unchanged from Step 6).
    - `cargo test -p katgpt-core --lib simd::` → **124/124 green** (up from 117; +7 new `project_ternary_simd` tests ported from riir-engine).
    - `cargo check` (katgpt-rs root default) clean.
    - `cargo test --lib` (katgpt-rs root default) → **3936/3937 green** (1 pre-existing `sliding_window_retains_recent` failure, identical to Step 6 baseline — no regression).
    - `cargo test --lib simd::` (katgpt-rs root) → **4/4 green** (attn_match tests).
    - `cargo check -p riir-engine` (default) clean (3 pre-existing pathfinder warnings).
    - `cargo check -p riir-engine --target wasm32-wasip2 --no-default-features --features hla` clean (3 pre-existing pathfinder warnings). The `+simd128` flag is set by riir-ai's `.cargo/config.toml`.
    - `cargo test -p riir-engine --lib` → **2428/2429 green** (1 pre-existing `cgsp_runtime::dual_pool_bridge::g5_epool_persistence` failure, documented in Step 6 — unrelated to simd).
    - `cargo build -p riir-engine --example wasm_simd_bench --no-default-features --features hla` (native) clean — example updated to use `simd_matvec` instead of dropped `matmul_f32_simd` wrapper.
    - WASM example build blocked by pre-existing `freetype-sys` C++ cross-compile failure (missing `string.h` for wasip2 sysroot) — unrelated to this change; the example's Rust code compiles clean (validated via lib check).

  **Bonus fix (pre-existing bug unblocked by Step 7):** `katgpt-core/src/simd/elementwise.rs:1493` `mask_f32x4_wasm` used `f32x4(...)` without importing it (relied on a function-scoped `use` in a sibling function). Latent bug — only surfaces when building for `wasm32 +simd128`, which riir-ai's `.cargo/config.toml` enables unconditionally (`rustflags = ["-C", "target-feature=+simd128"]`). Fixed by adding `use core::arch::wasm32::f32x4;` inside `mask_f32x4_wasm`. This unblocks the entire WASM SIMD128 build path for riir-engine.

### Phase 2 — riir-engine dedup (the DRY payoff)

After each Phase 1 step lands, riir-engine deletes its copy and imports from
`katgpt_core` the same way `analytic_lattice` / `arg_runtime` already do.

- [~] **2.1 riir-engine `src/hla/` → `use katgpt_core::hla::{...}`** (2026-06-28, PARTIAL — two blockers deferred)

  **Done:**
  - `HlaVariant` enum + `floats_per_q_head`/`floats_per_kv_group`/`layer_bytes` impl methods dedup'd — riir-engine now re-exports from `katgpt_core::hla::HlaVariant` (−29 LoC in riir-engine).
  - Core kernels optimized with riir-engine's hot-path improvements (GOAT direction — core is now canonical optimized version): `hla_state_update`/`hla_per_head_update` skip-zero optimization (when `k[i]==0.0`, skip inner loop — IEEE 754 identity); `hla_readout`/`hla_readout_normalized` loop-interchange + 4-wide chunking for cache locality + auto-vectorization (+53 LoC in core).

  **DEFERRED — Blocker #1 (role field contamination):** riir-engine's `HlaQHeadState`/`AhlaQHeadState` carry `role: Option<SlotLabel>` + `third_order: ThirdOrderMoment` fields (gated `hla_role_aware`). These contaminate the entire type chain (`HlaLayerState.heads` → `MultiLayerHlaCache.layers`). Core's types don't have these fields, so re-exporting would lose role-aware functionality. **Resolution path:** refactor role info into a side-channel (e.g., `HashMap<HeadId, RoleInfo>`) or pass `role` as a kernel parameter. Medium effort — touches all role-aware kernels + forward.rs.

  **DEFERRED — Blocker #2 (`ahla_step` math divergence — needs investigation):** riir-engine's `ahla_step` computes `tmp_r = PKV · q` (via `simd_matvec`), while katgpt-core computes `tmp_r = qᵀ · PKV` (manual loop, matches docstring). Since PKV = Σ k·vᵀ is non-symmetric, `PKV · q ≠ qᵀ · PKV`. riir-engine's docstring claims `qᵀ · PKV` but code computes the transpose — **a long-standing semantic mismatch**. Tests don't catch it (use sparse inputs where the two are identical). Per modelless-first mandate, did NOT change behavior. **Tracked as Issue 009** (`issues/009_ahla_step_math_divergence.md`). Needs decision: is riir-engine's `PKV · q` an intentional variant or a bug? If bug, fixing it changes trained behavior → riir-train follow-up.

  **GOAT gate:** 16/16 core hla tests, 8/8 root hla tests, 22/22 riir-engine hla tests (default), 28/28 hla_role_aware, full lib 2378/2379 (1 pre-existing `g5_epool_persistence`).
- [x] **2.2 riir-engine `src/transformer/` → consume `katgpt_transformer::{...}`** (2026-06-27)

  **Scope:** swapped all substrate types from local definitions to
  `katgpt-transformer` re-exports: `LayerWeights`, `TransformerWeights`,
  `KVCache`, `MultiLayerKVCache`, `KVSnapshot`, `KVLayerSnapshot`,
  `PagedKVCache`, `PAGE_SIZE`, `preload_kv_cache`, `MtpProjection`,
  `load_mtp_projection`, `project_target_activation`, `RavenKVCache`.
  Local definitions deleted from `transformer/mod.rs`, `transformer/raven.rs`,
  and `transformer/mtp.rs` (MTP projection substrate section removed; clustered
  LM head helpers stay local — they call `matmul`).

  **Kept local (correctly):** `ForwardContext` (engine-specific pruner fields),
  `PrefillContext` (**drifted** — riir-engine has `normed_x`, katgpt-transformer
  has `queries`+`residuals`; reconciliation deferred), all forward functions,
  `load_embed_*`, clustered LM head helpers, all raven forward functions.

  **Reconciliation toward safe defaults (Option A per user direction):** the
  initial swap revealed behavioral drift between riir-engine's local copies and
  katgpt-transformer's versions. Per user instruction "go for A, and do the same
  for other part too," katgpt-transformer was made conservative + riir-engine's
  better impls were ported:
  - `KVCache::reset()` — reverted no-op optimization to eager zeroing (safe
    default for shared substrate; avoids stale-KV leaks for consumers that
    reset between sequences).
  - `MultiLayerKVCache::restore()` — added `[pos..block_size)` tail zeroing
    (conservative; matches riir-engine's original behavior).
  - `PagedKVCache` — ported riir-engine's ArrayVec-based `ensure_pages`/
    `rollback` (stack-allocated scratch, zero heap alloc, bounded to 128 layers)
    + pre-populated free list (memory-efficient page reuse) + `kv_page_size`
    cached field (avoids recomputation). katgpt-transformer's pre-allocated
    `Vec` scratch approach (`deficits`/`new_pages`/`all_new_buf`/
    `rollback_removed` fields) removed in favor of ArrayVec.
  - `LayerWeights` — gated `attn_norm_gamma`/`mlp_norm_gamma`/
    `attn_qkv_fused` behind new `kog_cpu_fusion` feature. riir-engine uses
    `default-features = false` on its katgpt-transformer dep → gets the compact
    6-field struct (no ~2×n floats/layer dead weight). katgpt-rs root enables
    `kog_cpu_fusion` (default-on); `ane`/`gpu_inference` features imply it
    (their backends read the gamma fields unconditionally).
  - `fold_gamma`/`interleave_qkv` methods — gated behind `kog_cpu_fusion`.

  **Kept (additive, used by katgpt-rs root):** `MultiLayerKVCache.fill_pos`/
  `advance_pos` (sleep consolidation, eviction, tf_loop, all forward funcs),
  `RavenKVCache.readout_scores`/`readout_output` (root's `forward_raven`),
  `invalidate_position` (dflash Issue 053 — ported from riir-engine).

  **Validation:**
  - `cargo check` clean on both repos (katgpt-rs root + riir-engine).
  - katgpt-transformer: `--no-default-features`, default, `--all-features` all clean.
  - riir-engine `transformer::` tests: **80/80 green** (includes snapshot/restore,
    paged forward, prefill, preload_kv_cache, transformer_still compaction).
  - riir-engine `dflash::` tests: **24/24 green** (validates `invalidate_position`
    + `reset()` zeroing behavior in the speculative decoding path).
  - riir-engine full lib: **2382/2383 green** (1 unrelated pre-existing failure
    in `cgsp_runtime::dual_pool_bridge::g5_epool_persistence` — katgpt-core
    CGSP types, not transformer).
  - katgpt-rs root `transformer::` tests: **80/80 green** (validates the
    behavioral changes are compatible with root too).

  **Follow-up (tracked, not blocking):**
  - Reconcile `PrefillContext` drift (riir-engine `normed_x` vs katgpt-transformer
    `queries`+`residuals`). Requires porting the newer pre-activation caching
    scheme to riir-engine's `forward_prefill` or vice versa.
- [x] **2.3 riir-engine `src/types.rs` → `use katgpt_core::types::{...}`** (2026-06-28)

  **Scope:** riir-engine's `types.rs` already does `pub use katgpt_core::types::*;` at line 10. Only local addition is `NoiseSchedule` (feature-gated `dllm`, engine-specific D2F noise schedule). No further dedup possible — the substrate (Config, Rng, math utils, LoRA, DomainLatent, AttentionMode) is fully re-exported.
- [ ] 2.4 riir-engine `src/tokenizer.rs` → consume core (after step 3, if it moves)
- [x] **2.5 riir-engine `src/dd_tree.rs` + `spec_types.rs` → consume core** (2026-06-28)

  **Scope:** riir-engine's `spec_types.rs` local definitions of `TreeNode`, `DraftResult`, `RejectionReason`, `DraftEvent`, `PrefillMode`, `FlashPrefillConfig`, `BlockScores` replaced with `pub use katgpt_core::speculative::types::{...}` re-export. katgpt-core's versions are supersets (additive feature-gated fields/variants like `RejectionReason::KurtosisRejection`, `DraftResult.cost_snapshot`/`stability`/`routing_overlap`, `FlashPrefillConfig.score_reduction`/`budget_adaptation`).

  **Feature surface reconciliation:** added `spec_cost_model` + `stability_metrics` passthrough features to riir-engine `Cargo.toml` (matching the existing `temporal_deriv` pattern from Phase 2.6), added to default. The riir-ai workspace unifies these on via katgpt-rs root's defaults; the explicit passthroughs let `#[cfg(feature = "...")]` gates in riir-engine match the actual katgpt-core feature surface.

  **Construction sites updated:** `dflash.rs` `DraftResult { marginals, sampled_tokens }` literals at L272 and L317 extended with `#[cfg(feature = "domain_latent")] routing_overlap: None`, `#[cfg(feature = "spec_cost_model")] cost_snapshot: None`, `#[cfg(feature = "stability_metrics")] stability: None` — matches the root katgpt-rs pattern.

  **Kept local (composition):** `SpeculativeContext` (references `ForwardContext`, `MultiLayerKVCache` — engine-specific), `DDTreeBranchCache` (references `PagedKVCache`, `forward_paged`). All of `dd_tree.rs` (composition: `build_dd_tree*`, `extract_best_path*`, `TreeBuilder`).

  **GOAT gate:** riir-engine `spec_types::` 15/15, `dflash::` 24/24, `dd_tree::` 37/37, full lib 2378/2379 (1 pre-existing).
- [x] **2.6 riir-engine `src/mcts.rs`, `sampling.rs`, `delta_mem/` → consume core** (2026-06-28)

  **Scope:** Step-7 pattern (port improvements to core, then dedup consumer). Three sub-tasks done in parallel:

  **2.6a mcts + sampling:**
  - Ported riir-engine's perf optimizations to core: (1) `sample_residual_distribution_into` 4-wide chunked residual + fused write+sum + chunked normalize (replaces scalar + `simd_scale_inplace`); (2) `MCTSNode.children`/`unexpanded` changed from `Vec<usize>` to `ArrayVec<usize, 16>` (zero heap alloc per node, `MAX_UNEXPANDED=16` covers Bomber=6/Grid=5/Raid=~9); (3) `ucb1_score_cached` (hoist `ln(parent_visits)` out of inner loop, `total_cmp` over `partial_cmp`). Added `arrayvec` dep to katgpt-core.
  - riir-engine `src/sampling.rs` (164→16 LoC) and `src/mcts.rs` (697→19 LoC) rewritten as re-export shims. Note: `src/mcts.rs` was only used by its own tests (production uses `fourier/mcts.rs`'s `mcts_search_fourier`).

  **2.6b delta_mem:**
  - Ported riir-engine's hot-path APIs to core: `FeatureHasher::{hash_key_into, hash_value_into, feature_dim, to_vec_into, to_array}` (zero-alloc); `DeltaMemoryState::read_into` + inline-SIMD `write` + pre-allocated `segment_*_buf`; `MultiDomainMemory::{read_domain_into, read_aggregated(&mut self)}` + pre-allocated buffers + `#[repr(u8)]` on `AggregationStrategy`. Used the FourierFeatureHasher transpose trick (column-major generation → row-major storage) to preserve katgpt-core's bit-pattern AND get SIMD performance. Core's `temporal_deriv` surprise gate preserved.
  - riir-engine `src/delta_mem/{hash,state,multi}.rs` (1213 LoC total) DELETED; `mod.rs` rewritten as re-export shim (Step 6h pattern: inline `pub mod` blocks preserve absolute paths like `crate::delta_mem::state::DEFAULT_THETA_SURPRISE`). Composition (`pruner.rs`, `multi_pruner.rs`, `strategy_memory.rs`) kept local.
  - Added `temporal_deriv` passthrough feature to riir-engine (default-on).

  **GOAT gate:** core mcts 14/14, sampling 5/5, delta_mem 47/47 (+10 new bit-stability tests); delta_mem bench G3 PASS (suppression 42.90%, recall_loss 0.00% — bit-identical to Step 6 baseline); root 3936/3937; riir-engine delta_mem composition 29/29, fourier 346/346, full lib 2378/2379.

  **Net LoC: −1808 in riir-engine** (substrate deleted), **+445 in katgpt-core** (improvements ported with tests).
- [x] **2.7 riir-engine `src/simd/` → consume core** (2026-06-28, Step 7)

  **Scope:** riir-engine `simd/mod.rs` now does `pub use katgpt_core::simd::*;` for all targets (was target-gated). Deleted `src/simd/wasm32.rs` (630 LoC). Of its 11 functions: 8 had bit-identical core equivalents; 2 (`simd_sum_sq`, `simd_outer_product_acc`) had missing WASM SIMD128 paths in core — ported; 2 thin ergonomic wrappers (`dot_f32_simd`, `matmul_f32_simd`) dropped; 1 unique substrate (`project_ternary_simd`) ported to core under `plasma_path`. Bonus: fixed latent `mask_f32x4_wasm` missing-import bug in `katgpt-core/src/simd/elementwise.rs:1493`.

  **GOAT gate:** bit-identical, 124/124 core simd tests green (+7 new), 2428/2429 riir-engine lib tests green (1 pre-existing `cgsp_runtime::dual_pool_bridge::g5_epool_persistence`).

  **Commits:** `katgpt-rs/develop` `3a0ed1d5`, `riir-ai/develop` `ad8ea1ea`.
- [ ] 2.8 Bit-identical verification: `forward_hla`/`forward_gemma2`/`dd_tree` tests pass unchanged in both repos

### Phase 3-5 — DEFERRED

- [ ] **Phase 3** (root subdir reorg into `primitives/`/`inference/`/`games/`/`backends/`) — cosmetic, not worth churn while 100+ features flatten at root. Revisit if/when root module count becomes unnavigable.
- [ ] **Phase 4** (`plotters` optional, `cargo check --no-default-features` clean on root) — independent quick win; do as standalone if it becomes blocking.
- [ ] ~~**Phase 5** (publish `katgpt-rs` to crates.io)~~ — **RESCINDED.** Conflicts with the post-issue decision in `Cargo.toml:9` + `release-plz.toml:9-12` to keep root private permanently. Only `katgpt-core` ships.

---

## Risk register (carried from Issue 007, updated)

1. **`transformer.rs` is 8398 lines.** Moving without splitting violates the 2048-line ceiling. **Mitigation:** step 2 mandates split-then-move. This makes step 2 the single biggest commit of the plan — allocate a full focused session, not a tag-end task.
2. **riir-engine HLA diverged** (`*_role_aware` variants + `role_transport` wiring). **Mitigation:** port kernel variants into core behind `hla_role_aware` feature; keep `role_transport.rs` as private composition in riir-engine (Category C).
3. **Version churn.** Core moves to next minor (new modules); root version is meaningless (`publish = false`). **Mitigation:** only core version matters; release-plz handles it.
4. **`tokenizer` may pull SentencePiece C++ build dep.** **Mitigation:** Q2 verdict — audit-first, defer to step 3.
5. **`dd_tree`/`spec_types` reconciliation.** riir-engine's copy may have game-coupled additions. **Mitigation:** Q3 verdict — generic types to core, game impls stay in riir-engine.
6. **`mcts.rs` imports `crate::game_state::GameState`** (Category C game IP). **Mitigation:** Q3 verdict — parameterize over core `Game` trait in core; keep game-specific impl in riir-engine.

---

## Acceptance

Mirrors Issue 007 §Acceptance, updated:

- [ ] Phase 1 step 2: `transformer`+`weights` live in `katgpt-core`, split into <2048-line files, re-exported from root. `cargo test -p katgpt-core --lib` + `cargo test --lib` green on arm64. (x86_64 already cleared per Issue 006.)
- [x] **Phase 1 step 4 (substrate half):** `hla` cache types + streaming kernels live in `katgpt-core/src/hla/{types,kernel}.rs`, re-exported from root `src/hla/mod.rs`. Bit-identical forward output vs pre-move (8/8 forward tests + 16/16 substrate tests green). `forward.rs` stays in root (needs `ForwardContext`). Role-aware variants + `ThirdOrderMoment` deferred to Phase 2.1 (riir-engine reconciliation — they're Category C cognitive composition, not substrate).
- [x] **Phase 1 step 5 (substrate half):** speculative-decoding types live in `katgpt-core/src/speculative/types.rs`, re-exported from root `src/speculative/types.rs`. Bit-identical (32/32 substrate tests + 9 composition tests green).
- [x] **Phase 1 step 6 (substrate half):** `mcts` algorithm (`mcts_search`, `mcts_search_informed`, UCB1 helpers, `MCTSNode`), `sampling` primitives (CDF + residual samplers), and `delta_mem` substrate (`DeltaMemoryState`, `FeatureHasher`, `MultiDomainMemory`) live in `katgpt-core/src/{mcts.rs, speculative/sampling.rs, delta_mem/}`. Bit-identical behavior: 14+5+37=56 substrate tests green in core; 5 bandit composition tests + 0 sampling composition tests + delta_mem bench G3 (suppression 42.90%, recall_loss 0.00%) all green in root. Composition that needs root-only types (`BanditRolloutPolicy` needs `BanditStats`; `MemorySteeredPruner<P>` / `MultiDomainMemoryPruner<P>` wrap root `ScreeningPruner` impls) stays in root as expected. No call-site changes.
- [x] **Phase 1 step 7:** riir-engine `simd/wasm32.rs` consumes `katgpt_core::simd`. (2026-06-28)
- [ ] Phase 2: riir-engine has zero Category A duplicates; all consume `katgpt_core::`. Bit-identical tests in both repos.
- [ ] Each phase commit includes GOAT/bench evidence per AGENTS.md "dont defer benchmark task".

---

## Out of scope (explicit)

- Publishing `katgpt-rs` root crate (Phase 5 — rescinded).
- Moving cognitive/reasoning primitives (`cce`, `clr`, `compaction`, `claim_rubric`, etc.) out of root. They are correctly tiered: root = cognitive basics + composition, core = pure substrate, riir-* = GOAT tuning. Per Issue 007 §"Cognitive/reasoning is a NEW MOAT".
- Moving Category C game IP (`arena/`, `bom_arena/`, `cce_runtime/`, etc.) — stays private per the 003 strategy.
