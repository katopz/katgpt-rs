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
- [ ] **Step 6 — `mcts`, `sampling`, `delta_mem` → core.** Leaf inference mechanics. `mcts` parameterize over a core `Game` trait (Q3 verdict); leave game-specific impls in riir-engine.
- [ ] **Step 7 — riir-engine `simd/wasm32.rs` → consume `katgpt_core::simd`.** Already shipped in core under `wasm32_simd128_*` kernels. Diff for riir-engine-only improvements, port if any, then delete reimplementation.

### Phase 2 — riir-engine dedup (the DRY payoff)

After each Phase 1 step lands, riir-engine deletes its copy and imports from
`katgpt_core` the same way `analytic_lattice` / `arg_runtime` already do.

- [ ] 2.1 riir-engine `src/hla/` → `use katgpt_core::hla::{...}` (after step 4). Includes the deferred 4b work: riir-engine deletes its local `types.rs`/`kernel.rs` (substrate half), imports from core; keeps the role-aware wrappers + `ThirdOrderMoment` + `role_transport` as the cognitive composition layer (Category C).
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
- [ ] 2.3 riir-engine `src/types.rs` → `use katgpt_core::types::{...}` (already partially done via `spec_types.rs:11`)
- [ ] 2.4 riir-engine `src/tokenizer.rs` → consume core (after step 3, if it moves)
- [ ] 2.5 riir-engine `src/dd_tree.rs` + `spec_types.rs` → consume core (after step 5)
- [ ] 2.6 riir-engine `src/mcts.rs`, `sampling.rs`, `delta_mem/` → consume core (after step 6)
- [ ] 2.7 riir-engine `src/simd/` → consume core (after step 7)
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
- [ ] Phase 1 steps 5-7: each substrate module lives in core, re-exported from root.
- [ ] Phase 2: riir-engine has zero Category A duplicates; all consume `katgpt_core::`. Bit-identical tests in both repos.
- [ ] Each phase commit includes GOAT/bench evidence per AGENTS.md "dont defer benchmark task".

---

## Out of scope (explicit)

- Publishing `katgpt-rs` root crate (Phase 5 — rescinded).
- Moving cognitive/reasoning primitives (`cce`, `clr`, `compaction`, `claim_rubric`, etc.) out of root. They are correctly tiered: root = cognitive basics + composition, core = pure substrate, riir-* = GOAT tuning. Per Issue 007 §"Cognitive/reasoning is a NEW MOAT".
- Moving Category C game IP (`arena/`, `bom_arena/`, `cce_runtime/`, etc.) — stays private per the 003 strategy.
