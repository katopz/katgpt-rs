# Plan 327: ARG Protocol Primitives — Open Skeleton (Phase 1)

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/309_ARG_Latent_Substrate_Synthesis.md](../.research/309_ARG_Latent_Substrate_Synthesis.md)
**Private guide:** [riir-ai/.research/160_ARG_Over_Latent_State_Runtime_Guide.md](../../riir-ai/.research/160_ARG_Over_Latent_State_Runtime_Guide.md)
**Private wiring plan:** [riir-ai/.plans/337_arg_runtime_wiring.md](../../riir-ai/.plans/337_arg_runtime_wiring.md)
**Source protocol:** [ARG Standard](https://protocol.airistech.ai/arg-core.html) — Iris Technologies, 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/arg/` (new module) + Cargo feature `arg_protocol`
**Status:** Done — all phases shipped, promoted to DEFAULT-ON.

---

## Goal

Ship the **generic protocol primitives** that close the five gaps identified in Research 309, behind the `arg_protocol` feature flag (opt-in). These are pure types + traits — no game IP, no chain IP, no shard IP. The riir-ai runtime composes them with existing shipped systems into the ARG-over-Latent-Substrate pipeline (private Super-GOAT, Guide 160).

**GOAT gate (mandatory before promotion to default-on):**
- G1 Correctness (property tests)
- G2 Perf (≤50ns PolicyEnvelope eval; O(K) InfoRegistry lookup, K≤20)
- G3 No-regression (`cargo check --all-features`, `--each-feature`)
- G4 Alloc-free hot path (bounded-N case)
- G5 Silence-bias (OfflineCandidateScorer strictly penalizes low-confidence evidence)

If G1–G5 all pass AND gain is modelless → promote `arg_protocol` to default-on. If any fails → keep opt-in, file `.issues/` with the gap.

---

## Phase 1 — Unblocking Skeleton (CORE — this session)

The three smallest, most foundational primitives. Ships first so the open adoption hook exists.

### Tasks

- [x] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/arg/` with `mod.rs` declaring submodules.
- [x] **T1.2** Add Cargo feature `arg_protocol = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml`. Default-off.
- [x] **T1.3** Wire `#[cfg(feature = "arg_protocol")] pub mod arg;` in `katgpt-rs/crates/katgpt-core/src/lib.rs`.
- [x] **T1.4** Write `arg/policy.rs` — `PolicyEnvelope`, `PolicyState`, `ResponseMode`, `PolicyConstraints`. ≤100 lines.
  - `PolicyState ∈ {Allow, AllowWithRefocus, Restrict, Block}` `#[repr(u8)]` enum
  - `ResponseMode ∈ {Normal, Prudent, Refocus, Refusal}` `#[repr(u8)]` enum
  - `PolicyConstraints { allowed_labels: &[LabelId], forbidden_labels: &[LabelId], max_hops: u8, max_depth: u8, max_complexity: u16 }`
  - `PolicyEnvelope { state: PolicyState, constraints: PolicyConstraints, response_mode: ResponseMode }`
  - `PolicyEnvelope::evaluate(&self, ctx: &EvalCtx) -> PolicyDecision` — zero-alloc, returns decision + whether to short-circuit
- [x] **T1.5** Write `arg/taxonomy.rs` — `TaxonomyNode`, `TaxonomyKind`, `LabelId`, `LabelSet`, `TaxonomyValidator`. ≤200 lines.
  - `LabelId` = `u32` (stable identity, never recycled)
  - `TaxonomyKind ∈ {Cluster, Label, Leaf}` `#[repr(u8)]` enum
  - `TaxonomyNode { id: LabelId, kind: TaxonomyKind, parent_id: Option<LabelId>, incompatible_with: &[LabelId] }`
  - `LabelSet` = smallvec-like bounded set of `LabelId` (cap 32)
  - `TaxonomyValidator` — owns `&[TaxonomyNode]` (sorted by id for binary-search lookup)
  - `TaxonomyValidator::validate_label_set(&self, candidates: &LabelSet, scratch: &mut ValidationScratch) -> ValidationResult` — enforces existence, cluster↔label compatibility, parent/child coherence, explicit incompatibilities. Zero-alloc when scratch is preallocated.
  - `TaxonomyValidator::expand_ascending(&self, leaf_set: &LabelSet, scratch: &mut ValidationScratch) -> LabelSet` — child→parent→root expansion only (no descending).
- [x] **T1.6** Write `arg/lifecycle.rs` — `LifecycleState`, `RedirectTable`. ≤100 lines.
  - `LifecycleState ∈ {Active, Shadow, Deprecated, Removed}` `#[repr(u8)]` enum
  - `RedirectTable` — papaya lock-free `HashMap<LabelId, LabelId>` (deprecated → replacement); `redirect(&self, id: LabelId) -> LabelId` follows chains; `redirect_chain(&self, id) -> Vec<LabelId>` for audit
  - `RedirectTable::insert_redirect(old, new)` — chain compression on insert (avoid redirect chains longer than 3)
- [x] **T1.7** Write `arg/lib.rs`-style facade re-exports in `arg/mod.rs`.

### Phase 1 GOAT gate

- [x] **T1.G1** Property tests for `TaxonomyValidator`:
  - rejects non-existent label
  - rejects cluster↔label incompatibility
  - enforces parent/child coherence (a child without parent fails)
  - ascending expansion preserves `child ⊆ expanded_parent`
  - ascending expansion never descends
- [-] **T1.G2** Criterion bench: `PolicyEnvelope::evaluate` median ≤ 50ns; `TaxonomyValidator::validate_label_set` median ≤ 200ns (taxonomy of 256 nodes, candidate set of 8). — *Deferred to Phase 4 (covers all primitives in one bench).*
- [x] **T1.G3** `cargo check --all-features` passes; `cargo check` (default) unchanged.
- [x] **T1.G4** `PolicyEnvelope::evaluate` and `TaxonomyValidator::validate_label_set` zero-alloc verified via `cargo test --features arg_protocol` (assert no `Vec::new()` / `Box::new()` / `String` in hot path; use scratch buffers).
- [x] **T1.G5** N/A in Phase 1 (silence-bias scorer ships in Phase 2).

---

## Phase 2 — Typed Offline Candidates + Silence-Bias Scorer

- [x] **T2.1** Write `arg/candidate.rs` — `TypedOfflineCandidate`, `CandidateIntent`. ≤150 lines.
  - `CandidateKind ∈ {Split, Merge, Edge, Taxonomy, NewNode, RegistryDedup}` `#[repr(u8)]` enum
  - `CandidateIntent { kind, target_label: LabelId, before: LabelSet, after: LabelSet, evidence_refs: &[EvidenceId] }`
  - `TypedOfflineCandidate { intent: CandidateIntent, score: Option<f32> }`
- [x] **T2.2** Write `arg/scorer.rs` — `OfflineCandidateScorer`, `InfoOutcomeStatus`, `Evidence`. ≤200 lines.
  - `InfoOutcomeStatus ∈ {InfoConfirmedSuccess, InfoUncertainSuccess, InfoLowConfidence}` `#[repr(u8)]` enum
  - `Evidence { outcome: InfoOutcomeStatus, weight: f32 }`
  - `OfflineCandidateScorer::score(&self, candidate: &TypedOfflineCandidate, evidence: &[Evidence]) -> f32` — computes `Gain_info_confirmed`, `Gain_info_uncertain`, `Gain_info_lowconf` separately, applies `Penalty_silent(C)` if `uncertain + lowconf > threshold`.
  - `OfflineCandidateScorer::can_auto_commit(scored: &ScoredCandidate, threshold: f32) -> bool` — refuses auto-commit when low-confidence-dominated.
- [x] **T2.3** Property tests for G5 silence-bias:
  - Same nominal gain, all-confirmed evidence → score X
  - Same nominal gain, all-low-confidence evidence → score Y < X (strict)
  - Same nominal gain, 50/50 confirmed/lowconf → score Z, X > Z > Y
  - Auto-commit threshold refuses when lowconf fraction > threshold

---

## Phase 3 — Info Registry

- [x] **T3.1** Write `arg/registry.rs` — `InfoRegistry`, `InfoUnit`, `InfoKey`, `InfoType`, `AccessScope`, `CompareResult`. ≤250 lines.
  - `InfoType = u8` (controlled category)
  - `AccessScope = u64` (tenant/workspace id)
  - `LabelSignature = [u8; 32]` (BLAKE3 of `L_final_ids`)
  - `InfoKey { signature: LabelSignature, info_type: InfoType, scope: AccessScope }` — derives `Ord + Hash + Eq`
  - `InfoUnit { key: InfoKey, payload_hash: [u8;32], c_info: f32, outcome: InfoOutcomeStatus, provenance: Provenance, ts: u64 }`
  - `InfoRegistry` — papaya lock-free `HashMap<InfoKey, Vec<InfoUnit>>` (canonical unit + grey-zone candidates)
  - `InfoRegistry::canonicalize(&self, unit: InfoUnit, scratch: &mut MatchScratch) -> MatchResult`
    - Phase 1: hard filter by `InfoKey` exact
    - Phase 2: bounded recall on Top-K via lexical/vector (slot reserved, not implemented — gateway trait)
    - Phase 3: grey-zone `CompareResult ∈ {Same, Different, Unsure}` via pluggable `CompareFn` trait
  - `MatchResult ∈ {StrongMatch(InfoUnit), GreyZone(Vec<InfoUnit>), NoMatch}`
- [x] **T3.2** Property tests:
  - Two units with same `InfoKey` → `StrongMatch`
  - Two units with different `InfoKey` but same payload hash → `GreyZone`
  - Two units with different `InfoKey` and different payload → `NoMatch`
  - `InfoKey` order is deterministic (Ord derived from BLAKE3 bytes)

---

## Phase 4 — GOAT Gate + Promotion

- [x] **T4.1** Run `cargo test -p katgpt-core --features arg_protocol --lib` — all property tests pass. (61 tests: 26 P1 + 20 P2 + 15 P3.)
- [x] **T4.2** Run `cargo check --all-features` and `cargo hack check --each-feature` (if cargo-hack available). (all-features + default + no-default all clean; cargo-hack 0.6.45 available.)
- [x] **T4.3** Run criterion bench G2; record in `katgpt-rs/.benchmarks/327_arg_protocol_goat.md`. (G2a PolicyEnvelope ~0.4ns<50ns, G2b TaxonomyValidator ~170ns<200ns steady-state; G4 0 allocs/100 calls after the zero-alloc fix.)
- [x] **T4.4** If G1–G5 all PASS:
  - Moved `arg_protocol = []` from opt-in to `default` in `katgpt-core/Cargo.toml`.
  - Updated lib.rs comment (DEFAULT-ON).
- [x] **T4.5** N/A — all gates passed; no issue filed.

### Phase 4 zero-alloc fix (T4.3 finding)

The Phase 1 `TaxonomyValidator::validate_label_set` allocated 3× per call (local
`accepted` Vec + `mem::take` defeating scratch reuse). Fixed by:
1. Adding `accepted: Vec<LabelId>` to `ValidationScratch` (reusable buffer).
2. Replacing `mem::take` with `scratch.rejections.clone()` (scratch retains
capacity; 0 allocs on the no-rejection hot path).

After fix: G4 PASS (0 allocs / 100 calls). The rejection path (cold/error)
still allocates 1 Vec — acceptable (not steady-state).

---

## Out of scope (deferred)

- **riir-ai runtime wiring** — covered in `riir-ai/.plans/337_arg_runtime_wiring.md`. Composes these open primitives with HLA + Entity Cognition Stack + VMG + Sub-Goal Compaction.
- **riir-neuron-db `InfoKey` view on `NeuronShard`** — Phase 3 follow-up. `NeuronShard` already has the BLAKE3 commitment; just needs the `InfoKey` projection layer.
- **riir-chain LatCal commit of `LabelSignature`** — covered by existing chain infrastructure. No new work for v1.
- **Bounded LLM proposer** (ARG OW-3.2) — explicitly rejected for the runtime hot path. Reserved for offline candidate generation only, if ever.

---

## Risks

1. **Vocabulary collision** — `policy` already means many things. Mitigation: namespace under `arg::*`; use `PolicyEnvelope` not `Policy`.
2. **Premature unification** — risk of over-constraining future primitives. Mitigation: Phase 1 ships only types + validators; no runtime; riir-ai stays free to compose however.
3. **G5 gaming** — silence-bias penalty is easy to get wrong. Mitigation: G5 is a property test with strict inequalities, not a benchmark.
4. **Scope creep** — five primitives is a lot. Phase 1 ships three; Phase 2/3 ship the rest. Don't try to do all five in one session.

---

## TL;DR

Plan 327 ships five generic ARG protocol primitives (`PolicyEnvelope`, `TaxonomyValidator`, `TypedOfflineCandidate`, `LifecycleState`+`RedirectTable`, `InfoRegistry`) in `katgpt-rs/crates/katgpt-core/src/arg/` behind the `arg_protocol` feature flag. Phase 1 (this session) ships the three smallest with property tests + criterion bench. GOAT gate G1–G5 must all pass before promotion to default-on. Private moat composition lives in riir-ai Guide 160 + Plan 337.
