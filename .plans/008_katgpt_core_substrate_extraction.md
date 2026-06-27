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
- [ ] **Step 4 — `hla` → core.** 2248 lines total (`forward.rs` 569 + `kernel.rs` 1019 + `types.rs` 606 + `mod.rs` 54). Depends on step 2.
  - [ ] 4a. Move `hla/{mod,types,kernel,forward}.rs` → `katgpt-core/src/hla/`
  - [ ] 4b. Port riir-engine's `*_role_aware` variants behind a new core feature `hla_role_aware` (consolidation, not blind copy — per Issue 007 §"Cross-repo consumer cleanup")
  - [ ] 4c. Root `src/hla/` → thin re-export
  - [ ] 4d. GOAT gate: bit-identical forward output vs pre-move (sigmoid, never softmax)
  - [ ] 4e. Commit: `feat(core): Plan 008 step 4 — move HLA substrate to katgpt-core + port role-aware variants`
- [ ] **Step 5 — `dd_tree` + `spec_types` → core.** Traits already in `core/traits.rs`; move dependent types (`TreeNode`, `DDTreeBranchCache`, `SpeculativeContext`, `DraftResult`, `NoPruner`, `ScreeningPruner` dep types) to join them.
- [ ] **Step 6 — `mcts`, `sampling`, `delta_mem` → core.** Leaf inference mechanics. `mcts` parameterize over a core `Game` trait (Q3 verdict); leave game-specific impls in riir-engine.
- [ ] **Step 7 — riir-engine `simd/wasm32.rs` → consume `katgpt_core::simd`.** Already shipped in core under `wasm32_simd128_*` kernels. Diff for riir-engine-only improvements, port if any, then delete reimplementation.

### Phase 2 — riir-engine dedup (the DRY payoff)

After each Phase 1 step lands, riir-engine deletes its copy and imports from
`katgpt_core` the same way `analytic_lattice` / `arg_runtime` already do.

- [ ] 2.1 riir-engine `src/hla/` → `use katgpt_core::hla::{...}` (after step 4)
- [ ] 2.2 riir-engine `src/transformer/` → `use katgpt_core::transformer::{...}` (after step 2)
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
- [ ] Phase 1 step 4: `hla` lives in `katgpt-core` with role-aware variants behind feature flag. Bit-identical forward output vs pre-move.
- [ ] Phase 1 steps 5-7: each substrate module lives in core, re-exported from root.
- [ ] Phase 2: riir-engine has zero Category A duplicates; all consume `katgpt_core::`. Bit-identical tests in both repos.
- [ ] Each phase commit includes GOAT/bench evidence per AGENTS.md "dont defer benchmark task".

---

## Out of scope (explicit)

- Publishing `katgpt-rs` root crate (Phase 5 — rescinded).
- Moving cognitive/reasoning primitives (`cce`, `clr`, `compaction`, `claim_rubric`, etc.) out of root. They are correctly tiered: root = cognitive basics + composition, core = pure substrate, riir-* = GOAT tuning. Per Issue 007 §"Cognitive/reasoning is a NEW MOAT".
- Moving Category C game IP (`arena/`, `bom_arena/`, `cce_runtime/`, etc.) — stays private per the 003 strategy.
