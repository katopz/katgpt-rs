# Research 264: A Compositional Framework for Open-Ended Intelligence

> **Source:** [A Compositional Framework for Open-ended Intelligence](https://arxiv.org/abs/2606.15386) — Ida Momennejad (MSR NYC) & Roberta Raileanu (Google DeepMind), 17 Jun 2026
> **Date:** 2026-06-18
> **Status:** Active
> **Related Research:** 172 (MUSE skill lifecycle), 191 (Prism capability substrate), 211 (Bayesian posterior skill evolution), 171 (FrontierCS open-ended arena), 116 (LLM sleep consolidation), 240 (SGS self-play), 194 (CaDDTree)
> **Related Plans:** 215 (Regime-Transition w/ MDL gate), 274 (CGSP self-play), 282 (Dual-Pool CGSP), 191 (Open-ended arena), 094 (TIES merging)
> **Cross-ref (riir-ai):** Research 059 (MUSE validators), Plan 299 (NPC curiosity runtime), `riir-ai/crates/riir-engine/src/cgsp_runtime/cross_game_transfer.rs` (TaR — exact paper primitive already shipped)
> **PASS-Redirects (synthesis):** He, Li, Wang, Vitercik [arXiv:2609.19465 "Compositional Reasoning in Language Models under Reinforcement Learning Post-Training"] — the evaluation-side sibling of this note's PTG framework: dependency DAGs over skills (three levels: horizon / skill-switching / non-local dependency) plus a coverage-factor transfer bound; distilled at [Research 674](674_Compositional_RL_Asymmetry_Coverage.md) (Gain — the bound substrate POC + the training-data-design law).
> **Classification:** Public

---

## TL;DR

A *theory/framework* paper formalizing open-ended intelligence as **compositional closure** `L(P, C)` of a minimal primitive set `P` + composition operators `C`. Solutions are **Primitive Transition Graphs (PTGs)** — directed graphs of primitive invocations with branching/recurrence. Self-play in imagined counterfactual possible-worlds expands `L(P, C)` under a parsimony (MDL) constraint; primitives/motifs that survive recomposition are promoted into the library. Evaluation shifts from "behavioral performance" to "what persists across possible worlds".

**Distilled for katgpt-rs (modelless, inference-time):** Three runtime/inference-time contributions we don't yet ship, each fusable with existing pillars:
1. **Primitive Transition Graph (PTG)** as an explicit runtime data structure — every executed trace becomes a DAG of `(primitive, operator)` nodes; motifs are mined and promoted as higher-order primitives.
2. **Primitive Reuse Index (PRI)** + **Compositional Depth Generalization (CDG)** as runtime evaluation metrics — measure *closure expansion* and *compositional reuse* instead of (only) win-rate.
3. **Motif wrapping/promotion** — recurring subgraphs in PTGs get wrapped as new primitives and admitted through the existing MDL gate (Plan 215).

The paper's training-side contribution (Next Primitive Prediction / NPP objective) is a training paradigm → `riir-train`. The paper's transfer-side contribution (Transfer-as-Recomposition / TaR) is **already shipped** in `riir-ai/crates/riir-engine/src/cgsp_runtime/cross_game_transfer.rs` as the `AnchorProfile` 8-axis pivoting primitive (private Super-GOAT moat, do not duplicate publicly).

---

## 1. Paper Core Findings

### 1.1 The Core Tuple `(P, C, L)`

| Object | Definition | Examples |
|--------|------------|----------|
| `P` (Primitives) | Minimal reusable units: *representational* (objects, fields) + *algorithmic* (comparison, retrieval, verification) | `compare_distance`, `verify_threshold`, `branch`, `nearest_neighbor` |
| `C` (Composition Operators) | Sequencing, recursion, branching — type-safe chains between primitives | `Sequence(A→B)`, `Recurse(A)`, `Branch(cond, A, B)` |
| `L(P, C)` (Closure) | All computations reachable by repeated application of `C` to `P`. Open-ended ⟺ `|L| = ∞` and parsimonious | The set of all solvable task instances |

**Proposition 4.1 (Unbounded Closure):** Any `(P, C)` with ≥1 recursive operator + type-consistency + intermediate-output reuse generates an unbounded closure.

### 1.2 Primitive Transition Graph (PTG)

Every solution is a directed graph `G = (P, C)` where:
- Nodes = primitive invocations `p ∈ P`
- Edges = operator instances `c ∈ C` (state-transforming transitions)
- Recurring subgraphs = **motifs** (e.g., `Search → Interact → Verify`, "castling in chess", "hero's journey")

### 1.3 The Parsimony Constraint (MDL)

Primitive admission is gated by **Minimum Description Length**:
> "What is the smallest set of functions `P` that can reconstruct all successful traces in the training set?"

A candidate primitive enters `P` only if it *reduces* total description length by more than its admission cost. This is what separates "open-ended exploration" from "novelty-search noise accumulation" — without parsimony, the library fills with brittle, task-specific fragments (the compositional analogue of the "noisy TV" problem).

### 1.4 Next Primitive Prediction (NPP) — TRAINING OBJECTIVE → riir-train

> `L_NPP = -log P(p_{t+1}, c_{t+1} | p_{1:t}, c_{1:t}, W)`

A transformer/graph-predictor trained to emit the next `(primitive, operator)` pair given a partial PTG traversal and a "world vector" `W`. Forces the architecture to learn a compact, role-flexible basis that remains stable across counterfactual world variations.

**Routing:** NPP is a training paradigm requiring backprop through the predictor. It belongs in **riir-train**. We do not distill it here.

### 1.5 Transfer-as-Recomposition (TaR)

Open-endedness = competence that survives controlled counterfactual variation. Vary world dynamics, swap reward structure, reassign object roles → keep primitives/motifs that still compose. **Example:** Minecraft `move_to` rebinds to "swim to" in an underwater world; the high-level motif `Verify → Act → Verify` is unchanged.

### 1.6 Evaluation Metrics (all modelless, all runtime)

| Metric | Definition |
|--------|------------|
| **PRI** (Primitive Reuse Index) | Frequency of each primitive `p ∈ P` appearing across distinct task families. High → general axiom. Low → brittle fragment. |
| **CDG** (Compositional Depth Generalization) | Success rate on PTGs of depth `d_test > d_train`. Tests mastery of closure `L(P, C)`. |
| **PDY** (Primitive Discovery Yield) | Rate at which newly discovered primitives improve utility + sample efficiency. |
| **OCR** (Open-ended Curriculum Robustness) | Robustness of primitive composition under shifted task distributions. |
| **TaR** (Transfer-as-Recomposition) | Role-flexibility — hold `P` fixed, vary environmental constants. Low TaR → primitives were memorized. |

### 1.7 Collective Compositionality

Primitives should be portable objects retrievable by *functional role*, transmissible across agents/modules/time. Heterogeneous agents achieve algorithmic interoperability via overlapping primitive inventories + shared compositional language. (Examples cited: V(D)J recombination, Hox genes, Krebs cycle — minimal basis + composition = unbounded antibody/body-plan/metabolic diversity.)

---

## 2. Distillation

### 2.1 What Already Ships (Prior Art in Our Codebase)

This is the most important section. **A naïve direct-map of this paper would create three duplicate research notes and one duplicate plan.** The vocabulary-translation grep (per skill protocol §1 step 2) reveals the paper's mechanisms already shipped under different names:

| Paper Concept | Shipped Cousin | Location | Status |
|---------------|----------------|----------|--------|
| **Algorithmic primitive** `p ∈ P` (minimal reusable unit) | `ConstraintPruner` trait (arm in BanditPruner) + `SkillSpec` (civ) + WASM validator + LoRA adapter | `katgpt-rs/src/pruners/mod.rs`, `mmorpg-remaster/crates/mmorpg-edge-worker/src/persistence/skill.rs`, `riir-engine/src/adapters/` | ✅ Multiple |
| **Composition operators `C`** (sequence/branch/recurse, type-safe) | `crates/katgpt-pruners/src/lattice_operad/composed_pruner.rs` (operadic AND/OR composition), `TIES merge` (task vector sign election + disjoint merge) | `katgpt-rs/src/lattice_operad/`, `katgpt-rs/.plans/094_*.md` | ✅ Production |
| **Parsimony / MDL admission gate** | `RegimeTransitionGate::evaluate()` — accept iff `DL_new < DL_old - AdmissionCost` (default 32 bits/pruner) | `katgpt-rs/.plans/215_regime_transition_inference.md` T2 | ✅ **DEFAULT ON**, GOAT 8/8 |
| **Discovery as vocabulary change** (regime collapse → new primitive admission) | `RegimeCollapseClassifier` → `Discovery` regime in Four-Regime Router | Plan 215 T1 + T5 | ✅ Shipped |
| **Transfer-as-Recomposition (TaR)** across worlds | `AnchorProfile` (8-axis game-agnostic personality vocabulary) + `translate_priorities(source, target)` — pivots priorities through 8 canonical anchors, rebinds same personality to different game | `riir-ai/crates/riir-engine/src/cgsp_runtime/cross_game_transfer.rs` | ✅ **Super-GOAT private moat** (already classified PRIVATE, do not duplicate) |
| **Self-play in imagined worlds** (counterfactual neighborhood expansion) | `CgspLoop` + `DualPoolBandit` (E-pool successes + X-pool fresh candidates) + `ProblemMutator` trait | `katgpt-rs/crates/katgpt-core/src/cgsp/`, Plan 274/282, Plan 191 | ✅ Production |
| **Skill lifecycle** (create → memory → manage → eval → refine) | `skill_lifecycle` feature in `AbsorbCompress` + `BanditPruner` + `SkillCatalog` + `BomberTestGate` | `katgpt-rs/src/pruners/skill_*.rs`, `katgpt-rs/src/pruners/bomber/skill_lifecycle_player.rs` | ✅ Production |
| **Vector-based primitive steering** (function vectors, persona vectors) | `EmotionDirections` (valence/arousal/desperation/calm direction vectors), CNA Steering (Plan 087), SubstrateGate (Plan 216), Sparse Off-Principal Task Vector (Plan 264) | `katgpt-rs/crates/katgpt-pruners/src/emotion_vector.rs`, Plan 087/216/264 | ✅ Multiple |
| **Collective compositionality** (primitives portable across agents) | `KnowledgePayload { skill_id, confidence, cross_zone }` in civ signals + distillation-to-server pipeline | `riir-games/src/civ/{node_sync,signal,skill}.rs` | ✅ Production |
| **Verification primitives** (proof-gated admission) | WASM validator + `PrunerTestGate::run()` before bandit promotion | `katgpt-rs/crates/katgpt-pruners/src/skill_test.rs`, riir-ai validator pipeline | ✅ Production |
| **Provenance chain / replay** | `ProvenanceChain` (BLAKE3-hashed) + Kan-transport replay on regime change | Plan 215 T3 | ✅ Shipped |

**The single most important prior-art hit:** the paper's central *runtime* mechanism — "discover new primitives when current set cannot express the answer, admit through MDL gate, replay provenance through new vocabulary" — is **Plan 215's exact architecture**, shipped and GOAT-proved 8/8 + 4/4 real, under the names "Regime-Transition", "MDL gate", "AdmissionCost", "vocabulary change", "RegimeCollapseClassifier", "ProvenanceChain", "Kan-transport". The paper provides the *theoretical framing*; we have the *implementation*.

### 2.2 What Does NOT Ship (Genuine Gaps)

Three contributions from the paper have **no shipped equivalent** under any vocabulary:

1. **Primitive Transition Graph (PTG) as explicit runtime data structure.** We have *pruners* and *skill catalogs* and *operadic composition of pruners*, but no place where an *executed trace* is materialized as a directed graph of `(primitive, operator)` nodes for downstream mining. The closest thing is `EventLog` (Plan 124) and `TrialLog` (BanditPruner), both flat append-only logs without graph structure or motif mining.

   **✅ CONCRETELY INSTANTIABLE (2026-06-25):** Plan 324 (`bisimulation_operator_inference`, Research 308) ships `TransitionGraph` — a sorted, deduped, indexed observed-transition set that IS the PTG data structure. Combined with `partition_refine` (bisimulation quotient), it closes the motif-mining loop: recurring sub-paths collapse into single quotient classes, and `infer_operators` lifts them to abstract operator schemas. The CEI (Plan 290) wraps this for full motif-mining + metrics; Plan 324 supplies the deterministic core.

2. **Motif mining → motif wrapping → higher-order primitive promotion.** Plan 215 admits *new primitives* through the MDL gate but does not *consolidate recurring subgraphs of past executions* into new composite primitives. The paper's "turn recurring block into reusable function" loop is missing.

   **✅ CONCRETELY INSTANTIABLE (2026-06-25):** Plan 324's `BisimulationQuotient` + `OperatorSchema` provide the deterministic half — bisimulation collapses recurring motifs into equivalence classes, and operator inference produces the abstract "wrapped" primitive (one `OperatorDef` per recurring label). The MDL-gated admission (Plan 215) wraps around this: when a quotient class's BLAKE3 commitment stabilizes across N observations, admit the corresponding `OperatorDef` as a first-class primitive. The non-deterministic half (LLM-grade symbolic abstraction, ASP solver) remains the CWM path (Plan 296).

3. **Closure-expansion metrics (PRI, CDG, PDY, OCR, TaR) as runtime evaluation surface.** We measure win-rate, latency, win-rate-improvement, GOAT gate pass-rate. We do not measure:
   - *How often each primitive recurs across task families* (PRI)
   - *Success on traces deeper than any seen in training* (CDG)
   - *Rate of useful primitive discovery* (PDY)
   - *Win-rate variance under task-distribution shift* (OCR)
   - *Role-flexibility: hold primitive set, vary environment, does the solution still compose?* (TaR) — note: `AnchorProfile` *performs* transfer but does not *measure* it as an evaluation metric.

### 2.3 Fusion (the value-add over direct mapping)

**Fusion target: Closure-Expansion Instrument (CEI)** — a measurement + data-structure layer that retrofits the paper's PTG/PRI/CDG metrics onto our existing MDL-gated regime-transition + CGSP + AnchorProfile + SkillCatalog stack. It is NOT a new capability class; it is the **evaluation lens** that turns "we ship open-ended inference" from a claim into a measured property.

**Three closest cousins to fuse:**

- **Cousin A — Plan 215 (Regime-Transition / MDL gate):** provides the parsimony-gated primitive-admission primitive. CEI wraps it: every successful execution writes a PTG; every `RegimeTransitionGate::evaluate()` call is annotated with `(candidate_primitive, motif_it_completes)`.
- **Cousin B — Plan 172/MUSE skill lifecycle + Plan 211 Bayesian posterior skill evolution:** provides per-skill memory and lifecycle actions (explore/patch/split/compress/retire). CEI adds a new action: **`promote_motif`** — when a subgraph recurs ≥ N times across distinct task families (high PRI), wrap it as a new primitive and admit through MDL gate.
- **Cousin C — `cross_game_transfer.rs` AnchorProfile (riir-ai private):** already performs TaR operationally. CEI adds **TaR as a metric**: hold `P` fixed, perturb game config (`ProblemMutator`), measure how often the same PTG motif still solves the perturbed instance. This converts the private Super-GOAT moat into a *measurable* moat — important for selling.

**Fusion hypothesis:** "Fusing the paper's PTG data structure × Plan 215's MDL gate × AnchorProfile's cross-game pivoting produces a *closure-expansion dashboard* — for any NPC, at any tick, we can answer: how diverse is its compositional repertoire? how transferable is it across counterfactual worlds? what is the marginal value of the last primitive discovered? None of the three alone can answer these questions."

This is a **measurement + explainability** layer, not a new capability class. So it is **GOAT**, not Super-GOAT. See §3.

---

## 3. Verdict

**GOAT — Closure-Expansion Instrument (CEI) is a provable-gain measurement + data-structure layer, not a new capability class.**

**One-line reasoning:** The paper's *capability-class* contributions are (a) MDL-gated primitive admission → **already shipped as Plan 215 (DEFAULT ON, GOAT 8/8)**, (b) Transfer-as-Recomposition → **already shipped as private Super-GOAT `AnchorProfile` in `riir-ai/crates/riir-engine/src/cgsp_runtime/cross_game_transfer.rs`**, (c) Next Primitive Prediction → **training paradigm → riir-train**. What remains genuinely novel — PTG as runtime data structure, motif mining + promotion, closure-expansion metrics (PRI/CDG/TaR) — is *measurement and data-structure* layering over what we already ship. Provably useful for explainability, transfer-quality scoring, and "open-endedness" as a selling-point metric, but does not unlock a new capability the existing stack cannot do.

### 3.1 Novelty gate (§1.5 of skill protocol)

| Question | Answer |
|----------|--------|
| **Q1 No prior art?** | NO. Plan 215 ships the MDL-gated primitive admission (the paper's §4.3 + §5 minus NPP objective). `cross_game_transfer.rs` ships TaR (the paper's §5.2). Skill lifecycle (172/211) ships creation/memory/eval/refine. The genuinely missing pieces are PTG-as-data-structure, motif mining, and PRI/CDG/TaR-as-metrics — incremental over shipped foundations, not greenfield. |
| **Q2 New class of behavior?** | NO. CEI is a measurement/explainability layer. The capabilities it measures (open-ended primitive discovery, cross-world transfer) already exist in our stack; CEI makes them *observable*. |
| **Q3 Product selling point?** | Partial. "Our NPCs have a measurable compositional-closure diversity index" is interesting but not headline. Better as supporting metric for the existing Super-GOAT claims (Plan 215 + AnchorProfile). |
| **Q4 Force multiplier ≥2 pillars?** | YES. Connects ConstraintPruner lifecycle × AnchorProfile transfer × CGSP self-play × SkillCatalog × freeze/thaw × MUSE memory. Single highest-leverage unification opportunity for our existing open-ended stack. |

**1 NO → not Super-GOAT.** Proceed to GOAT.

### 3.2 Why not Pass

The paper itself names the missing pieces explicitly (PTG, motif mining, PRI/CDG/TaR metrics), and these are *concrete, modelless, runtime-implementable* additions that fuse with ≥4 existing pillars. Pass would be wrong — there is genuine shippable value here, just not a moat.

### 3.3 Routing

| Deliverable | Repo | Action |
|-------------|------|--------|
| Research note (this file) | `katgpt-rs/.research/264_*.md` | ✅ Created |
| Plan: Closure-Expansion Instrument | `katgpt-rs/.plans/` | ✅ Will create as Plan 286 (next free slot) |
| Open primitive: PTG data structure + motif miner + PRI/CDG/TaR metrics | `katgpt-rs/crates/katgpt-core/src/closure/` | Plan phase 1–4 |
| riir-ai guide | — | **NOT created** (verdict ≠ Super-GOAT; the private selling-point doc for AnchorProfile already exists in `riir-ai/crates/riir-engine/src/cgsp_runtime/cross_game_transfer.rs` doc comments) |
| riir-train routing | `riir-train/.research/` | NPP training objective → "note redirect to riir-train", do not create in this session |

---

## 4. Latent vs Raw Boundary (for the future plan)

The PTG data structure has a clear raw-vs-latent split per AGENTS.md:

| PTG Field | Space | Synced? | Why |
|-----------|-------|---------|-----|
| `primitive_id: u32` (enum variant index) | Raw (discrete tag) | YES — if PTG snapshots are committed for replay/audit | Bit-identical replay requires deterministic primitive enumeration |
| `operator_id: u8` (Sequence/Branch/Recurse) | Raw (discrete tag) | YES | Same — anti-cheat replay |
| `execution_tick: u32` | Raw | YES | Deterministic ordering |
| `blake3_hash` of motif subgraph | Raw (32-byte commitment) | YES (audit) | Tamper-evident |
| `motif_embedding: [f32; K]` (latent motif vector for similarity) | Latent | NO — local only | Used for PRI computation + motif clustering, not for game-state reconstruction |
| `reuse_count: u32` per primitive (for PRI) | Raw (counter) | YES (aggregate) | Counter is deterministic given the same execution history |
| `TaR_score: f32` (transfer metric output) | Latent (statistical) | NO — local-only diagnostic | Statistical, not bit-reproducible |

**Rule:** the *structure* of the PTG (which primitive, what operator, what order) is raw and syncable. The *latent embeddings used for similarity/metric computation* stay local. Bridge functions: `ptg_to_motif_embedding()` (raw→latent, dot-product projection onto motif direction vectors, sigmoid-bounded, zero-allocation) and `motif_embedding_to_tar_score()` (latent→raw scalar, clamp to [0,1]).

**Anti-pattern to avoid:** never sync the `motif_embedding` vector itself. The scalar `TaR_score` is the only thing that crosses the sync boundary, and only as a diagnostic aggregate (not for anti-cheat validation — TaR is not a movement claim).

---

## 5. Implementation Sketch (defer to Plan 286)

```
katgpt-rs/crates/katgpt-core/src/closure/    NEW MODULE, feature = "closure_instrument"
├── mod.rs              PrimitiveKind (enum), OperatorKind (enum), PtgNode, PtgEdge, PrimitiveTransitionGraph
├── trace.rs            PtgRecorder — wraps any ConstraintPruner execution, materializes PTG
├── motif.rs            MotifMiner — frequent-subgraph mining over recent PTGs (gSpan-lite, bounded depth)
├── metrics.rs          PrimitiveReuseIndex, CompositionalDepthGeneralization, TransferAsRecomposition
├── admit.rs            MotifAdmitter — wraps RegimeTransitionGate (Plan 215) for motif-as-primitive admission
└── tests.rs            property tests + corpus fixture (Bomber/Go/Civ trace corpus)
```

**Feature flag:** `closure_instrument` (off by default; promote to default-on only if GOAT-gate passes).

**GOAT gate (defer to Plan 286):**
- G1: PRI computation < 100µs per 1K-trace corpus (Hot-tier)
- G2: Motif mining adds < 5% overhead to regime-transition admission path
- G3: TaR metric correlates ≥ 0.5 with measured cross-game transfer acceleration (validate against `AnchorProfile` benchmarks in riir-ai)
- G4: PTG snapshot of 10K traces serializes to < 1MB (for cold-tier commitment)
- G5 (demotion): if metrics don't correlate with any existing quality/transfer benchmark, demote to opt-in diagnostic only

**CPU/GPU/ANE tier routing (per AGENTS.md):**
- PTG construction: plasma (µs, CPU, single-thread, lock-free `push`)
- Motif mining: warm (ms, CPU rayon, batched at sleep-cycle boundaries like AutoDreamer)
- PRI/CDG/TaR computation: hot (sub-ms, CPU SIMD, dot-products over motif embeddings)
- PTG cold-tier commitment: cold (BLAKE3 hash + Merkle-octree — reuse Plan 280 infra)

---

## TL;DR

**Paper = theory/framework; not a single primitive.** The capability-class mechanisms it describes — MDL-gated primitive admission and Transfer-as-Recomposition — **already ship** in Plan 215 (`RegimeTransitionGate`, DEFAULT-ON, GOAT 8/8) and `riir-ai/crates/riir-engine/src/cgsp_runtime/cross_game_transfer.rs` (`AnchorProfile`, private Super-GOAT moat) respectively. The paper's training-side contribution (Next Primitive Prediction) routes to **riir-train**. The genuinely missing runtime pieces — PTG as explicit data structure, motif mining with promotion, PRI/CDG/TaR evaluation metrics — fuse naturally with our existing stack (Plan 215 MDL gate × MUSE skill lifecycle × AnchorProfile transfer × CGSP self-play) into a **Closure-Expansion Instrument (CEI)** that makes "open-ended inference" a *measurable* property of our NPCs. **Verdict: GOAT** (measurement/explainability layer, not new capability class) — research note created, Plan 286 to follow, **no riir-ai guide** (verdict ≠ Super-GOAT; private AnchorProfile selling-point doc already exists in code).
