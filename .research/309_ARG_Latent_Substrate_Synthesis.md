# Research 309: ARG × Latent Substrate — Protocol Synthesis (Super-GOAT Fusion)

> **Source:** [ARG Standard](https://protocol.airistech.ai/arg-core.html) + [Context Weaver](https://protocol.airistech.ai/context-weaver.html) + [Policy Manager](https://protocol.airistech.ai/policy-manager.html) + [airistech.ai blog](https://airistech.ai/from-deterministic-to-adaptive-reasoning-graphs/) — Iris Technologies, 2026. (Companion HF posts by TeamAIris were unavailable; canonical protocol pulled from `protocol.airistech.ai`.)
> **Date:** 2026-06-25
> **Status:** Active — Super-GOAT (fusion)
> **Classification:** Public — open synthesis (no game IP, no chain IP, no shard IP)
> **Related Research:** 010 (riir-ai KG × HLA × Role Transport), 141 (riir-ai KG Triple Typology), 146 (riir-ai Entity Cognition Stack), 154 (riir-ai Viable Manifold Graph runtime guide), 155 (riir-ai Per-NPC Sub-Goal Compaction guide), 084 (ActiveGraph event-sourced graphs), 234 (DenseMesh latent node network), 294 (Viable Manifold Graph primitive), 249 (DecentMem dual-pool reachable router)
> **Related Plans:** 327 (this — open ARG protocol primitives), 312 (VMG), 320 (Closed-Unit Compaction Gate), 092 (Freeze/Thaw), 251 (DEC operators)
> **Cross-ref (riir-ai):** Research 160 (ARG-over-Latent-State Runtime Guide — private Super-GOAT moat), Plan 337 (private runtime wiring)

---

## TL;DR

ARG (Adaptive/Deterministic Reasoning Graph) is a **protocol**, not a mechanism: an ontology (taxonomy `cluster → label root → parent → child` + leaf-node graph) navigated by a **read-only online loop** (Policy Pre-Check → Classification → Context Weaver → Landing Point → Bounded Traversal → Info/Action → Episodic MemoryWrite) and evolved by a **governed offline loop** (Collection → Typed Candidates → Scoring with silence-bias penalty → Validation → Lifecycle `ACTIVE → DEPRECATED → REMOVED` with redirect/alias tables). The whole thing is bound by hard invariants: vectors are approximators, taxonomy validity always wins, silence ≠ confirmed success, snapshot pinned per request.

**Direct-mapping ARG onto this codebase is mostly Pass** — ARG is LLM-text-centric (Policy Manager shapes refusals, Context Weaver routes LLM labels, Info nodes are text chunks), and would violate the modelless-first / latent-to-latent / freeze-thaw-over-fine-tuning mandates. **But fusing ARG's protocol discipline with our latent-state substrate is Super-GOAT.** We already ship ~70% of the pieces — they're distributed across eight non-ARG-vocabulary notes (R010/R141/R146/R154/R155/R294/R084/R234). What's missing is (a) the unifying protocol-contract framing, and (b) five concrete open primitives that close the remaining gaps.

**Distilled for katgpt-rs (modelless, inference-time, open):**

Five generic protocol primitives, no game/chain/shard semantics:

1. **`PolicyEnvelope`** — `policy_state ∈ {ALLOW, ALLOW_WITH_REFOCUS, RESTRICT, BLOCK}` + `PolicyConstraints { allowed_labels, forbidden_labels, max_hops, max_depth, max_complexity }` + `response_mode`. Hard-gate consumed by binding/traversal/writes.
2. **`TaxonomyValidator`** — deterministic checker over `TaxonomyNode { id, kind ∈ {Cluster, Label, Leaf}, parent_id, incompatible_with }` enforcing existence, cluster↔label compatibility, parent/child coherence, explicit incompatibilities. Produces `L_valid` from `L_union`.
3. **`TypedOfflineCandidate`** — enum `Split | Merge | Edge | Taxonomy | NewNode | RegistryDedup` with `before/after_intent` + `evidence_refs`. The typed structural-change vocabulary ARG's offline loop Step B mandates.
4. **`LifecycleState`** — `Active | Shadow | Deprecated | Removed` for ontology leaves + `RedirectTable` mapping deprecated→replacement so episodic records remain interpretable under split/merge.
5. **`InfoRegistry`** — canonicalization with `InfoKey = (LabelSignature, InfoType, AccessScope)` + two-phase matching (hard filter on `InfoKey` exact, bounded recall on Top-K) + grey-zone `CompareResult ∈ {Same, Different, Unsure}`. The `ShardIndex` cousin, ARG-shaped.

All five are pure engine plumbing — no IP leak.

---

## 1. Paper Core Findings (ARG protocol summary)

### 1.1 The ontology (§0.1)

- **Context typologies** are high-level combinable dimensions (role × product × channel × environment × time). They DO NOT carry knowledge — they only parameterize constraints.
- **Taxonomy**: `cluster` (stable top-level domain root) ↔ `label` (hierarchical chain `root → parent → child`). Taxonomy validity is enforced deterministically by the Context Weaver's validator.
- **Ontological graph ("branches & leaves")**: nodes are TERMINAL operational leaves (concept, action, info, memory behavior). Each node carries `title + chunk`. Nodes attach M2M to clusters/labels/edges. Vector indexes are runtime artifacts, NOT persistent fields.
- **Access vs domain scope — do not mix**: `AccessScope` = tenant/workspace boundary (Policy Manager, RBAC/ABAC); `RetrievalType ∈ {USER, DOMAIN, EXTERNAL}` = semantic/content boundary (taxonomy + Context Weaver).

### 1.2 Online loop — read-only over fixed snapshot (§2)

| Step | What | ARG-mandated outputs |
|------|------|----------------------|
| 1 | Policy Pre-Check | `policy_state`, `policy_constraints`. BLOCK → stop; REFOCUS → refocus + stop; RESTRICT → continue under constraints. |
| 2 | Initial classification | `L_raw` (lexical candidate labels) + policy-filtered candidates. NOT retrieval. |
| 3 | Context Weaver | `L_final`, `L_final_ids`, `PrimaryLabelID`, `LabelSignature`, `RetrievalType`, `confidence_global`, flags (`OOD`/`LOW_MARGIN`/`VECTOR_AMBIGUITY`/`ABSTAIN_RECOMMENDED`/`NEW_INTENT_CANDIDATE`/`UNKNOWN_LABEL_CANDIDATE`/`CONFLICT_POLICY_STRONG_SIGNAL`/`LLM_ESCALATED`/`COLD_START_BUFFER_CANDIDATE`). |
| 4 | Landing point | `N_eligible := N_scope ∩ N(L_final)`. All later steps stay inside. Action vs Info binding decision. CLARIFY when uncertain. |
| 5 | Neighbor scoring | Bounded ranked candidates, policy+taxonomy gated. |
| 6 | Fast Path | Short-circuit when stable direct target exists. Still gated by policy + taxonomy + N_eligible. |
| 7 | Deterministic traversal | Bounded depth/hops/tokens/latency, inside N_eligible. Outputs target node + bounded `InternalContextBundle`. |
| 8 | Action | Re-check authorization. ABSTAIN/CLARIFY fallback. Never invent actions online. |
| 9 | Info | Grounded response. **`C_info ∈ [0,1]` + `InfoOutcomeStatus ∈ {INFO_CONFIRMED_SUCCESS, INFO_UNCERTAIN_SUCCESS, INFO_LOW_CONFIDENCE}`. Silence ≠ confirmed success.** Merge internal/external/registry bundles. |
| 10 | MemoryWrite | Episodic only. `InfoKey = (LabelSignature, InfoType, AccessScope)` canonicalization. Two-phase dedup. Grey-zone NLI compare. PII/access invariants enforced. |

### 1.3 Offline loop — controlled evolution (§3)

A→E gated pipeline, versioned, taxonomy-guided, outcome-driven, auditable, memory-continuity-safe.

- **A Collection**: episodic registries + action outcomes + memory events + coverage/abstention signals. Strong vs weak signal separation preserved.
- **B Candidates**: typed (`split | merge | edge | taxonomy | new-node | registry_dedup`), taxonomy-anchored, evidence-referenced, bounded budget per batch.
- **C Scoring**: replay-based when possible. **Silence-bias penalty** — `Penalty_silent(C)` penalizes candidates whose gains are dominated by unverified outcomes. Auto-commit forbidden if gains are dominated by `Gain_info_uncertain + Gain_info_lowconf`.
- **D Validation**: taxonomy coherence, policy/access, structural integrity (no uncontrolled M2M explosion), memory continuity (no orphaned episodic records).
- **E Lifecycle**: `ACTIVE → DEPRECATED → REMOVED`. Split/merge MUST preserve continuity via redirect/migration. Indexes/validators/connectors updated at publish.

### 1.4 Invariants (§6.2 anti-patterns)

- No online mutation of ontology.
- Vectors are approximators, never structure-definers, never policy-bypassing.
- No duplication of Policy Manager responsibilities inside Context Weaver or ARG Core.
- Silence/absence of errors ≠ success.
- No external content promoted to ontology online.
- No publish without validation + versioning + lifecycle continuity.

---

## 2. Distillation — what we already ship (the 70%)

**Canonical failure mode I almost committed**: a paper-vocabulary grep for `taxonomy | Context Weaver | Policy Manager | L_final | RetrievalType | InfoUnit | landing point` returns ~zero hits across `katgpt-rs/.research/` + `riir-ai/.research/`, suggesting "ARG is novel here". **This is the R269/R296 failure mode.** The corpus already ships ARG patterns under codebase vocabulary. The mapping:

| ARG concept | Codebase vocabulary | Note |
|---|---|---|
| Ontology: taxonomy cluster↔label | `SenseKind` (7 clusters) + `ACTION_*` namespaces (relation labels) + `KgSlotLabel` (structural slots) | R141 (riir-ai) §1.2 explicitly frames this as "the literature's heterogeneous/multiplex graph axis" — that *is* ARG's taxonomy. |
| Leaf-node graph; M2M attachments | Entity Cognition Stack (9-layer: SPECIES/PERSON/KIN/COMPANIONS/COMMUNITY/NORMS/STRATEGY/SITUATION/SENSE); layer-count-agnostic | R146 (riir-ai) + Plan 327 ✅ implemented. "Branches & leaves" over per-NPC cognition. |
| Context Weaver (taxonomy arbitration) | KG × HLA × Role Transport — KnowFormer Q-RMPNN/V-RMPNN + Repository-Attention + Journey-Based Role Transport | R010 (riir-ai). Structure-aware queries route via KG topology into attention. |
| Landing point + bounded traversal | Viable Manifold Graph (`manifold_geodesic`, `manifold_random_walk` on safe latent subgraph) | R294 (open primitive) + R154 (riir-ai Super-GOAT guide). ARG traversal, latent-space. |
| Neighbor scoring + fast path | `polytope_router`, `zone_gating`, plasma-tier fast-track | Multiple. |
| Episodic MemoryWrite (Step 10) | `CompactionAuditRecord` + Raven slots (`consolidation.rs`) + `episode_buffer.rs` | R155 (riir-ai) §2.3 has the exact "stays local vs crosses sync" table. |
| Offline consolidation (Step A→E) | Per-NPC Sub-Goal Compaction C1/C2/C3/N1 rubric + Raven/δ-Mem sleep + `can_freeze` two-sided gate | R155 + riir-neuron-db/.research/007. |
| Policy Pre-Check / constraints | `DualSignalGate` (`kg_gate.rs`) — latent conviction AND raw corroboration required before social triple commits | Gates at emission, the right ARG place. |
| Silence ≠ success / `C_info` | Two-sided freeze gate (`FreezeGateReport`: input N≥d, output flatness<0.3, `can_freeze`) + G1 correctness gates everywhere | Per-primitive, not unified. |
| Event-sourced log + replay | R084 ActiveGraph maps the pattern, identifies the gap, lays out `EventLog<A>` + content-addressed eval cache | Gap acknowledged. |
| Snapshot publish boundary | `MerkleFrozenEnvelope`, atomic Arc swap, `commit_snapshot_via_quorum` | Strong direct match. |
| Reachability guarantee (no trapping) | DecentMem dual-pool router (E-pool/X-pool with X always nonzero → irreducible Markov chain, O(log T) regret) | R249 + Plan 282. |

**So ARG-shaped behavior already emerges from the composition of these eight notes.** What's missing is the protocol-level naming/contract that lets them interop as one ARG-style pipeline, plus five concrete gap primitives.

---

## 3. The honest gaps (the 30%)

Q1 grep (this session, all five repos): zero hits for `InfoRegistry | InfoKey | policy_state | TaxonomyValidator | TypedOfflineCandidate | OfflineCandidate | ArgProtocol`. The five gaps:

| # | Gap primitive | Why ARG-mandated | Why generic (no IP) |
|---|---|---|---|
| G1 | `PolicyEnvelope` (`policy_state` + `PolicyConstraints`) | Step 1 hard gate. We have many gates but no unified envelope crossing binding/traversal/writes. | Pure enum + small constraint record. |
| G2 | `TaxonomyValidator` (deterministic validator) | Step 3 produces `L_final` via strict validation. We have `DualSignalGate` (emission gate) but no label-set validator producing `L_valid` from `L_union`. | Pure tree-walk over `TaxonomyNode` records. |
| G3 | `TypedOfflineCandidate` enum | Step B mandates typed candidates. Our consolidation produces a weight delta, not typed structural changes with before/after intents. | Enum + intent record. |
| G4 | `LifecycleState` + `RedirectTable` | Step E mandates `ACTIVE→DEPRECATED→REMOVED` with redirect/alias preserving episodic interpretability under split/merge. `MerkleFrozenEnvelope` versions but no per-leaf lifecycle. | Enum + hashmap. (Existing `seal-online-remaster` `LifecycleState{Spawned,Owned,...}` is game-state, NOT ontology evolution — different concept, no conflict.) |
| G5 | `InfoRegistry` (`InfoKey` canonicalization + two-phase dedup + grey-zone compare) | Step 10 mandates canonicalization with stable key + two-phase match + grey-zone. `ShardIndex` is zone→shard lookup, not dedup. | Pure registry over `InfoUnit` records. |

---

## 4. Latent-space reframing (mandatory per workflow §1.5 step 3)

The fusion: **ARG protocol discipline × latent-state substrate**. ARG operates over text chunks and LLM tokens; we operate over:

- **HLA per-NPC 8-dim latent state** (`sense/`) → becomes the "chunk" attached to a leaf. A persona is an ARG Info node whose `chunk` is the HLA vector + provenance.
- **`latent_functor/` operations** → become the typed edge semantics. `arithmetic.rs` vector arithmetic IS the movement grammar between leaves.
- **`NeuronShard` `style_weights[64]` + `MerkleFrozenEnvelope`** → becomes the canonicalized InfoUnit. `InfoKey = (LabelSignature, InfoType, AccessScope)` maps to `(shard BLAKE3, SenseKind, zone_hash)`.
- **`cgsp_runtime/` curiosity** → drives Step 5 neighbor scoring. Curiosity-weighted edge selection.
- **DEC Stokes operators** (`exterior_derivative`, `codifferential`, `hodge_decompose`) → become the silence-bias detector. `belief_mass_divergence > 0` since last compaction IS the "no silence" predicate (positive divergence = mass creation = real signal).
- **LatCal fixed-point commitment** → becomes the publish boundary. Offline candidate → LatCal-validated raw scalar → chain commit → next online snapshot.

**Adapter routing / KV compression framings are NOT the primary reframing here** — this is the R269 anti-pattern. The latent-substrate reframing is the primary; the protocol discipline is the multiplier.

---

## 5. Verdict — **Super-GOAT (fusion)**

| Tier | Criteria | This work |
|------|----------|-----------|
| Q1 No prior art? | ✓ (verified this session: zero hits on `InfoRegistry`/`TaxonomyValidator`/`PolicyEnvelope`/`TypedOfflineCandidate`/`LifecycleState` for ontology / `ArgProtocol` across all five repos. Pieces shipped but unsynthesized; gap primitives genuinely missing.) | PASS |
| Q2 New class of behavior? | ✓ ARG-grade deterministic lifecycle-governed protocol over latent state at MMO scale — neither LLM agents (slow, costly) nor scripted AI (brittle, no emergence) can do this. | PASS |
| Q3 Product selling point? | ✓ "Our NPCs reason on a deterministic, auditable, lifecycle-governed latent-state graph at 20Hz × thousands of entities — ARG-grade auditability without LLM cost, with freeze/thaw-versioned persona divergence." | PASS |
| Q4 Force multiplier? | ✓ Touches ≥6 pillars: HLA, latent_functor, neuron-shard, chain commitment, DEC, cgsp, engram, KG. | PASS |

**All 4 YES → Super-GOAT.** Selling point: *ARG-grade deterministic lifecycle-governed reasoning over a freeze/thaw-versioned latent-state graph at MMO scale.*

Per skill rule, mandatory outputs in this session:
1. **Open primitive** → this note + `katgpt-rs/.plans/327_arg_protocol_primitives.md` + `katgpt-rs/crates/katgpt-core/src/arg/` module (Phase 1 unblocking skeleton).
2. **Architectural GUIDE (private moat)** → `riir-ai/.research/160_ARG_Over_Latent_State_Runtime_Guide.md`.
3. **Plans** → `katgpt-rs/.plans/327_*` (open) + `riir-ai/.plans/337_*` (private runtime wiring).

---

## 6. Latent vs raw boundary (per AGENTS.md)

| Artifact | Space | Crosses sync? |
|---|---|---|
| HLA 8-dim affect per NPC | latent | no |
| `TaxonomyNode { id, kind, parent_id, ... }` | raw | yes (ontology publish) |
| `LabelSignature` (BLAKE3 of L_final_ids) | raw | yes |
| `PolicyEnvelope { state, constraints }` | raw | yes (constraints are deterministic) |
| `InfoKey = (LabelSignature, InfoType, AccessScope)` | raw | yes |
| `TypedOfflineCandidate` intents | raw | yes (publish trail) |
| `LifecycleState` + `RedirectTable` | raw | yes |
| NeuronShard `style_weights[64]` | latent | no (only BLAKE3 commitment crosses) |
| 5 synced affect scalars (valence/arousal/...) | raw | yes (bridge from latent) |

**Bridge rule**: latent → raw is one-way (project to scalar via sigmoid, clamp to range). Never reconstruct latent from raw.

---

## 7. What stays open vs private

- **Open (katgpt-rs)**: the five generic protocol primitives (`PolicyEnvelope`, `TaxonomyValidator`, `TypedOfflineCandidate`, `LifecycleState`+`RedirectTable`, `InfoRegistry`). Pure protocol plumbing.
- **Private (riir-ai)**: the wiring that turns HLA + Entity Cognition Stack + VMG + Sub-Goal Compaction + DualSignalGate + KG triples into one ARG-shaped per-NPC pipeline at MMO scale. The selling point.
- **Private (riir-chain)**: LatCal commitment of `LabelSignature` + `InfoKey` for tamper-evident publish trail (already covered by existing chain infrastructure — no new chain work needed for v1).
- **Private (riir-neuron-db)**: `NeuronShard` as the canonicalized InfoUnit physical representation — already shipped; just needs the `InfoKey` view layered on top.

---

## 8. Validation protocol — GOAT gate (G1–G5)

Per AGENTS.md feature-flag discipline, the open plan (`katgpt-rs/.plans/327`) ships behind `arg_protocol` feature, opt-in. GOAT gate:

- **G1 Correctness** — `TaxonomyValidator` enforces: missing label rejected, cluster↔label incompatibility rejected, parent/child coherence enforced, ascending-only expansion preserves invariants. Property test.
- **G2 Perf** — `PolicyEnvelope` evaluation ≤ 50 ns (parity with `SalienceTriGate`, plasma tier). `InfoRegistry` two-phase lookup O(K) where K ≤ 20. Criterion bench.
- **G3 No-regression** — `cargo check --all-features` passes; `--each-feature` passes; default features unchanged.
- **G4 Alloc-free hot path** — `PolicyEnvelope::evaluate` and `TaxonomyValidator::validate_label_set` zero-alloc in the bounded-N case (input slice + scratch buffer).
- **G5 Silence-bias** — `OfflineCandidateScorer::score` returns strictly lower score for candidates whose evidence is dominated by `InfoOutcomeStatus::INFO_LOW_CONFIDENCE` vs `INFO_CONFIRMED_SUCCESS`, given equal nominal gain.

If all G1–G5 pass AND gain is modelless → promote `arg_protocol` to default-on.

---

## 9. Fusion genealogy

This note synthesizes eight prior notes into one protocol framing:

```
ARG (Iris Technologies, 2026)
   │
   ├──× R010 (KG × HLA × Role Transport) ──────────── Context-Weaver-shaped routing in latent space
   ├──× R141 (KG Triple Typology) ─────────────────── Ontology as heterogeneous/multiplex graph
   ├──× R146 (Entity Cognition Stack) ─────────────── Branches & leaves over per-NPC cognition
   ├──× R154 (Viable Manifold Graph runtime guide) ── Bounded traversal on safe latent subgraph
   ├──× R155 (Per-NPC Sub-Goal Compaction guide) ──── Offline consolidation + episodic MemoryWrite
   ├──× R084 (ActiveGraph event-sourced) ──────────── Event log + replay + fork-and-diff substrate
   ├──× R294 (Viable Manifold Graph primitive) ────── Open traversal primitive
   └──× R249 (DecentMem dual-pool reachable router) ─ Reachability guarantee (no trapping)
              │
              ▼
       ARG-over-Latent-Substrate
       (Super-GOAT fusion — this note)
```

---

## 10. Risks (honest)

1. **Scope creep** — five gap primitives is a lot for one plan. Mitigation: Phase 1 ships only `PolicyEnvelope` + `TaxonomyValidator` + `LifecycleState` (small, foundational). `InfoRegistry` and `TypedOfflineCandidate` are Phase 2/3.
2. **Vocabulary collision** — "policy" already means many things in this codebase. Mitigation: namespace under `arg::*` and use `PolicyEnvelope` (not `Policy`) to avoid collision.
3. **Premature unification** — synthesizing eight notes into one protocol risks over-constraining future primitives. Mitigation: protocol primitives are pure types + traits, not a runtime. Game runtime stays free to compose them however it wants.
4. **GOAT gate gaming** — silence-bias penalty (G5) is easy to get wrong. Mitigation: G5 must be a property test, not a benchmark.

---

## TL;DR

ARG is a protocol (not a mechanism) for deterministic, auditable, lifecycle-governed agent reasoning. Direct mapping onto this codebase is mostly Pass (LLM-text-centric, would violate modelless-first). **Fusing ARG's protocol discipline with our latent-state substrate is Super-GOAT** — all four novelty-gate questions pass honestly (Q1 verified this session: zero code hits for the five gap primitives). We already ship ~70% of the pieces across R010/R141/R146/R154/R155/R294/R084/R249 — they're distributed under non-ARG vocabulary. The 30% gap is five generic open primitives: `PolicyEnvelope`, `TaxonomyValidator`, `TypedOfflineCandidate`, `LifecycleState`+`RedirectTable`, `InfoRegistry`. Mandated outputs in this session: this note + private guide (`riir-ai/.research/160`) + open plan (`katgpt-rs/.plans/327`) + private wiring plan (`riir-ai/.plans/337`) + Phase 1 implementation + GOAT gate G1–G5.
