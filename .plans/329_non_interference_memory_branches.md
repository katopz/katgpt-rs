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

- [ ] **T1.1** Create `crates/katgpt-core/src/branching/` module directory with `mod.rs` (feature-gated under `non_interference_branches` in `crates/katgpt-core/Cargo.toml`).
- [ ] **T1.2** `types.rs` — decoupled structs: `BranchId(u32)`, `EpisodicEntry<E> { embedding: Vec<f32>, payload: E, reward: f32, scope: Option<u64>, tick: u64 }`, `ProceduralRule { direction: Vec<f32>, antecedent: [u8;32], strategy: [u8;32], helpful: u32, harmful: u32 }`, `FailureEntry<E> { embedding: Vec<f32>, payload: E, tick: u64 }`, `BranchStats { n_writes: u32, n_reads: u32, avg_reward: f32, last_touch_tick: u64 }`, `BranchLifecycle` (re-export ARG `LifecycleState` when `arg_protocol` feature is on; else local `Active|Shadow|Deprecated|Removed` enum). `CognitiveBranch<E>` aggregates these. All `#[derive(Clone, Debug)]`, `#[repr(C)]` where Pod-compatible.
- [ ] **T1.3** `bank.rs` — `BranchBank<E: Clone>` with `Vec<CognitiveBranch<E>>` (pre-allocated capacity `max_branches`), `max_branches: usize`, `n_active: usize`. Methods: `new(max_branches)`, `get(BranchId)`, `active_branches() -> impl Iterator`, `spawn(anchor: Vec<f32>) -> Option<BranchId>` (returns None if at capacity), `merge(b1, b2) -> BranchId` (concatenates episodic, sums stats, ors anchors into normalized sum), `prune(b)` (sets lifecycle=Removed, decrements n_active). Zero-alloc: `Vec::with_capacity(max_branches)` once, `clear()` + reuse on prune.
- [ ] **T1.4** `router.rs` — `BranchRouter { tau_snap: f32, tau_jaccard: f32, tau_spawn: f32 }`. Method `route(&self, query_embedding: &[f32], bank: &BranchBank<E>) -> RouteResult { branch: Option<BranchId>, mode: RouteMode }`. `RouteMode ∈ {Reuse, Spawn, Frozen}`. Scoring: `s(b) = dot(query, b.spawn_anchor)`; if `max s(b) >= tau_snap` → Reuse; else if any same-family sibling has Jaccard ≥ tau_jaccard on hash-token overlap → Reuse (lower score); else if capacity remains → Spawn; else Frozen (no write). Pre-normalized embeddings assumed (caller responsibility). Hot path: branch-free max reduction over `spawn_anchor` dot products (SIMD-friendly).
- [ ] **T1.5** `verifier.rs` — `VerifierGate { tau_write: f32, tau_curiosity: f32, quarantine_centroid_thresh: f32 }`. Method `should_write(&self, reward: f32, curiosity: f32, branch_centroid_sim: f32) -> WriteDecision`. `WriteDecision ∈ {Write, Quarantine, Reject}`. Rule: `Write` iff `reward > tau_write ∧ curiosity > tau_curiosity ∧ branch_centroid_sim >= quarantine_centroid_thresh`. Composes with CLR's `should_write_memory(r_k, S_LP)` — CLR is the upstream gate; VerifierGate adds the branch-centroid quarantine check.
- [ ] **T1.6** Unit tests in each file (property tests): bank spawn/merge/prune roundtrip; router returns Spawn when bank empty, Reuse when snap match, Frozen at capacity; verifier Write/Quarantine/Reject boundaries.
- [ ] **T1.7** Wire `branching` module into `crates/katgpt-core/src/lib.rs` behind `#[cfg(feature = "non_interference_branches")]`.

**Phase 1 exit:** `cargo test -p katgpt-core --features non_interference_branches --lib` green; `cargo check --features non_interference_branches` clean.

---

## Phase 2 — `NonInterferenceProjection` + `BudgetCompiler`

### Tasks

- [ ] **T2.1** `projection.rs` — `NonInterferenceProjection { directions: Vec<[f32; D]> }` (fixed `D` const generic or config). Method `project(&self, branch_id: BranchId, vector: &[f32]) -> Vec<f32>` — projects vector onto branch's direction. Method `interference(&self, b1: BranchId, b2: BranchId) -> f32` — returns `abs(dot(directions[b1], directions[b2]))` (0 = orthogonal/non-interfering, 1 = identical). Method `assign_direction(&mut self, branch_id, direction)` — sets direction, should verify near-orthogonal to existing (warn if interference > epsilon). Method `max_orthogonal_branches(d: usize) -> usize` — returns `d` (the dimension count). Property test: `interference(b_i, b_j) ≈ 0` for all i≠j when directions are orthogonal.
- [ ] **T2.2** `compiler.rs` — `BudgetCompiler { budget_bytes: usize }`. Method `compile(&self, materials: &RetrievedMaterials<E>) -> CompiledContext<E>` — applies fixed priority cascade (scoped_ctx → procedural → episodic → failures → working_memory → query), drops lowest-priority first when over budget. `RetrievedMaterials<E>` aggregates branch-local + cross-branch positive + working memory. `CompiledContext<E>` is the bounded output. Zero-alloc: writes into pre-allocated `Vec` with `clear()` + reuse.
- [ ] **T2.3** Property tests: projection orthogonality invariant; compiler respects budget (output ≤ budget_bytes); compiler priority cascade (scoped_ctx never dropped before working_memory).

**Phase 2 exit:** all Phase 1+2 unit tests green.

---

## Phase 3 — GOAT Gate + Promotion

### Tasks

- [ ] **T3.1** `tests/bench_329_non_interference_branches_goat.rs` — GOAT gate:
  - **G1 (correctness):** spawn N=8 branches with orthogonal directions in D=8 space; verify `interference(b_i, b_j) < 1e-6` for all i≠j. Write to branch i; verify branch j's episodic/procedural stores unchanged (non-interference by construction).
  - **G2 (perf):** `router.route()` on 64-branch bank < 1µs (release). Measure with criterion or `std::time::Instant` over 10K iterations.
  - **G3 (no-regression):** `cargo check --all-features` and `cargo check --no-default-features` both clean.
  - **G4 (alloc-free):** `router.route()` and `verifier.should_write()` allocate 0 bytes on the hot path (inspect with `#[global_allocator]` counter or assert no `Vec::new()` / `Box::new()` in the path).
  - **G5 (modelless):** no `riir_train` / `riir_gpu` dependency. Pure closed-form arithmetic + dot products.
- [ ] **T3.2** If G1–G5 all PASS → promote `non_interference_branches` to `default` in `crates/katgpt-core/Cargo.toml` and `katgpt-rs/Cargo.toml`.
- [ ] **T3.3** Record benchmark in `katgpt-rs/.benchmarks/329_non_interference_branches_goat.md`.

**Phase 3 exit:** all gates PASS; feature promoted to default-on (if modelless gain proven) OR kept opt-in with documented reason.

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
