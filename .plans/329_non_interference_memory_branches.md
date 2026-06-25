# Plan 329: Non-Interference Memory Branches — Continual Adaptation Primitive

**Date:** 2026-06-26
**Research:** [katgpt-rs/.research/310_RIZZ_Non_Interference_Memory_Branches.md](../.research/310_RIZZ_Non_Interference_Memory_Branches.md)
**Source paper:** [arxiv 2606.20638](https://arxiv.org/abs/2606.20638) — RIZZ (Goel et al., Oxford, Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/branching/` (new module) + Cargo feature `non_interference_branches`
**Status:** Active — Phase 0 (this plan). Super-GOAT fusion of BAKE × CLR × MCGS × Engram × ARG × closure-instrument × Salience.

---

## Goal

Ship five generic open primitives that, composed together, implement RIZZ's verifier-gated non-interference memory branches over a latent substrate:

1. **`BranchBank<B>`** — bounded bank of persistent `CognitiveBranch`es (zero-interference zones)
2. **`BranchRouter<E>`** — dot-product snap router (no LLM judge; uses pre-normalized latent embeddings)
3. **`VerifierGate<V>`** — gates memory writes on verifier score; composes with CLR `should_write_memory`
4. **`NonInterferenceProjection`** — orthogonal projection directions per branch; updates projected onto branch direction don't contaminate others
5. **`BudgetCompiler`** — priority-cascade context compiler under fixed byte/token budget

GOAT gate: G1 correctness (orthogonality preserves non-interference), G2 perf (router < 1µs per query at ≤64 branches), G3 no-regression (default-off), G4 alloc-free hot path, G5 modelless (no riir-train dep).

---

## Phase 1 — Skeleton: `BranchBank` + `BranchRouter` + `VerifierGate` (CORE)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/branching/` module directory with `mod.rs` (feature-gated under `non_interference_branches` in `crates/katgpt-core/Cargo.toml`). ✅ 2026-06-26
- [x] **T1.2** `types.rs` — decoupled structs: `BranchId(u32)`, `EpisodicEntry<E>`, `ProceduralRule`, `FailureEntry<E>`, `BranchStats`, `BranchLifecycle`, `CognitiveBranch<E>`. All `#[derive(Clone, Debug)]`, `#[repr(C)]` / `#[repr(transparent)]` where Pod-compatible. `BranchLifecycle` re-exports ARG `LifecycleState` when `arg_protocol` is on. ✅ 2026-06-26
- [x] **T1.3** `bank.rs` — `BranchBank<E: Clone>` with pre-allocated capacity, free-list slot reuse, `spawn`/`merge`/`prune`/`merge_sweep`/`prune_sweep`/`write_episodic`. ✅ 2026-06-26
- [x] **T1.4** `router.rs` — `BranchRouter` with dot-product snap + Jaccard fallback (`route` + `route_with_tokens`), SIMD-friendly max reduction. ✅ 2026-06-26
- [x] **T1.5** `verifier.rs` — `VerifierGate` with Write/Quarantine/Reject + `should_write_composed` for CLR composition. ✅ 2026-06-26
- [x] **T1.6** Unit/property tests in each file (56 tests total: types 10, bank 17, router 13, verifier 16). ✅ 2026-06-26
- [x] **T1.7** Wire `branching` module into `crates/katgpt-core/src/lib.rs` behind `#[cfg(feature = "non_interference_branches")]`. ✅ 2026-06-26

**Phase 1 exit:** `cargo test -p katgpt-core --features non_interference_branches --lib` green (56/56); `cargo check --features non_interference_branches` clean; `cargo check --no-default-features` clean; `cargo check --all-features` clean; `cargo check --features non_interference_branches,arg_protocol` clean (BranchLifecycle type-alias composition verified). ✅ 2026-06-26

---

## Phase 2 — `NonInterferenceProjection` + `BudgetCompiler`

### Tasks

- [x] **T2.1** `projection.rs` — `NonInterferenceProjection<const D: usize = 8>` with row-major `Vec<[f32; D]>`, const-generic dimensionality (matches `DelayRing<D,K>` idiom). Methods: `project(branch_id, vector) -> Option<f32>` (dot-product onto branch direction), `interference(b1, b2) -> f32` (=`|dot(dir_i, dir_j)|`), `assign_direction(branch_id, direction) -> AssignResult` (validates dimension/magnitude/interference, L2-normalizes in place), `max_orthogonal_branches() -> usize` (=D, frame-theory limit), `is_non_interfering_with_all(branch_id)`, `clear_direction`, `grow`, `assigned_directions` iterator. `AssignError` enum (`WrongDimension`/`ZeroMagnitude`/`Interferes`), `AssignResult` struct with conflict_branch + interference. 24 unit tests including G1-critical `orthogonal_property_invariant_8_branches_in_d8` and `ninth_direction_in_d8_must_interfere`. ✅ 2026-06-26
- [x] **T2.2** `compiler.rs` — `BudgetCompiler { budget_bytes }` with 7-tier priority cascade (`ScopeCtx > Procedural > Episodic > CrossBranchPositive > Failures > WorkingMemory > Query`). `RetrievedMaterials<E,F,W,Q,S>` aggregates heterogeneous materials (scope_ctx, procedural rules, episodic entries, cross-branch positives + parallel byte array, failures, working memory + parallel byte array, query). `CompiledContext<O>` with `clear()`+reuse discipline, per-tier `tier_counts`/`tier_dropped` arrays. `compile()` takes 14 closures (7 byte-cost + 7 payload-convert). Atomic admission (never partial); budget invariant `bytes_used <= budget_bytes` enforced + `debug_assert!`-ed. Zero-alloc steady-state (reuse same `out`). 21 unit tests including cascade priority, budget invariant, atomic admission, parallel-byte-array override. ✅ 2026-06-26
- [x] **T2.3** Property tests: `orthogonal_property_invariant_8_branches_in_d8` (G1-critical: all 8×7 ordered pairs have interference < 1e-6), `ninth_direction_in_d8_must_interfere` (frame-theory: 9th direction in D=8 interferes by ≥ 1/sqrt(D)), `compile_respects_budget_invariant` (100 items × 30 bytes under 50-byte budget → 1 admitted, 99 dropped, within_budget), `scope_ctx_never_dropped_before_working_memory` (priority cascade), `compile_atomic_admission_never_partially_admits`. ✅ 2026-06-26

**Phase 2 exit:** all Phase 1+2 unit tests green (101/101: types 10, bank 17, router 13, verifier 16, projection 24, compiler 21). ✅ 2026-06-26

---

## Phase 3 — GOAT Gate + Promotion

### Tasks

- [x] **T3.1** `benches/bench_329_non_interference_branches_goat.rs` — GOAT gate. ✅ 2026-06-26
  - **G1 (correctness):** G1a — spawn N=8 branches with orthogonal directions in D=8 space; verify `interference(b_i, b_j) < 1e-6` for all 8×7=56 ordered pairs (max observed = 0.00e0). G1b — write to branch 0; verify branch 1..7's episodic/procedural/failure stores byte-for-byte unchanged (non-interference by construction). G1c — 9th direction in D=8 (normalized all-ones) correctly rejected: interferes by 0.3536 ≥ 1/sqrt(8)=0.3536 > threshold 0.1.
  - **G2 (perf):** `router.route()` on 64-branch bank = 301.5ns < 1µs target (3.3× margin) over 10,000 iters (release, `std::time::Instant`, `black_box`).
  - **G3 (no-regression):** `cargo check --all-features`, `cargo check --no-default-features`, `cargo check -p katgpt-core` (default), `cargo check` (root) — all clean.
  - **G4 (alloc-free):** `router.route()` and `verifier.should_write()` allocate 0 bytes over 100 steady-state calls (CountingAllocator). G4b companion gate verifies the 0-alloc result is non-degenerate (correct WriteDecision variants for known inputs).
  - **G5 (modelless):** `non_interference_branches = []` deps in Cargo.toml — no `riir_train`/`riir_gpu`. Pure dot-product + budget arithmetic.
- [x] **T3.2** Promoted `non_interference_branches` to `default` in `crates/katgpt-core/Cargo.toml` + added passthrough `non_interference_branches = ["katgpt-core/non_interference_branches"]` to root `katgpt-rs/Cargo.toml` default list. ✅ 2026-06-26
- [x] **T3.3** Recorded benchmark in `katgpt-rs/.benchmarks/329_non_interference_branches_goat.md`. ✅ 2026-06-26

**Phase 3 exit:** all gates PASS; feature promoted to default-on with pure modelless gain (structural geometric orthogonality, not learned). ✅ 2026-06-26

---

## Phase 4 — Composition Tests with Existing Primitives

### Tasks

- [ ] **T4.1** Composition test: `BranchBank` + `arg_protocol` — verify `CognitiveBranch.lifecycle` round-trips through ARG `LifecycleState` when both features are on. Branch spawn = ARG `Active`; merge = source → `Deprecated` + RedirectTable; prune = `Removed`.
- [ ] **T4.2** Composition test: `VerifierGate` + `clr` — verify `should_write` composes with CLR `should_write_memory(r_k, S_LP)` (CLR is upstream; VerifierGate adds branch-centroid check downstream).
- [ ] **T4.3** Composition test: `BranchRouter` + `engram` — verify router can snap to branches whose `spawn_anchor` is derived from Engram hash-address embeddings.
- [ ] **T4.4** Composition test: `NonInterferenceProjection` + `closure` (Plan 290) — verify closure motifs can populate `ProceduralRule.direction` from PTG node signatures.

**Phase 4 exit:** all composition tests green; the five primitives compose cleanly with the four existing systems they're designed to fuse with.

---

## Out of scope (deferred)

- **riir-ai runtime wiring** — covered in `riir-ai/.plans/338_cognitive_branch_runtime_wiring.md`. Composes these open primitives with HLA + Entity Cognition Stack + CLR runtime + Engram runtime to give each NPC its own `BranchBank`.
- **riir-neuron-db freeze/thaw per branch** — each branch's state could be frozen into a `MerkleFrozenEnvelope` for tamper-evident persistence. Separate work item; the open primitive doesn't depend on it.
- **Cross-NPC branch transfer** — sharing a branch across NPCs (e.g., a "combat tactics" branch shared by all guards). Separate work item.
- **LLM-judge-based hierarchical labeling** — RIZZ uses an LLM judge to propose `(function, application)` labels. Our reframing uses pure dot-product snapping on latent embeddings (no LLM). If a future use case needs richer labels, an LLM judge can be added in riir-ai without changing the open primitive.

---

## Risks

1. **Sparse-branch failure** (RIZZ §4 DS-1000) — mitigation: `merge_sweep` with `min_examples_per_branch` floor. Documented in research note §5.1.
2. **Orthogonal capacity limit** — in D=8 HLA space, ≤8 fully-orthogonal branches. Mitigation: near-orthogonal (interference < ε) for more branches; `NonInterferenceProjection.max_orthogonal_branches(d)` documents the limit.
3. **Verifier quality** — CLR reward signal may be noisy. Mitigation: compose CLR `S_LP` (curiosity) as secondary gate.
4. **Vocabulary collision** — "branch" is overloaded. Mitigation: namespace `branching::`, use `CognitiveBranch` not `Branch`.

---

## References

- **Paper:** [RIZZ arxiv 2606.20638](https://arxiv.org/abs/2606.20638)
- **Research note:** [katgpt-rs/.research/310_RIZZ_Non_Interference_Memory_Branches.md](../.research/310_RIZZ_Non_Interference_Memory_Branches.md)
- **Private guide:** [riir-ai/.research/161_Per_NPC_Cognitive_Branch_Continual_Adaptation_Guide.md](../../riir-ai/.research/161_Per_NPC_Cognitive_Branch_Continual_Adaptation_Guide.md)
- **Fusion cousins:** Plan 236 (BAKE), Plan 284/316 (CLR), progressive_mcgs/ (branch spawning), Plan 299 (Engram), Plan 327 (ARG), Plan 290 (closure-instrument), Plan 303 (Salience)
