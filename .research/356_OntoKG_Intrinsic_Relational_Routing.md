# Research 356: OntoKG — Ontology-Oriented KG Construction with Intrinsic-Relational Routing

> **Source:** [OntoKG: Ontology-Oriented Knowledge Graph Construction with Intrinsic-Relational Routing](https://arxiv.org/abs/2604.02618) — Yitao Li, Zhanlin Liu, Anuranjan Pandey, Muni Srikanth (ProRata.ai), 3 Apr 2026
> **Date:** 2026-07-01
> **Status:** Done
> **Related Research:** katgpt-rs `141` (KG Triple Typology — DIRECT prior art, §1.3 raw/latent = intrinsic/relational, §3.3 #7 rates LPG LOW), `208` (SLoD), `209` (BAKE); riir-ai `010` (KG × HLA Role Transport), `141` (KG Triple Typology reference)
> **Related Plans:** riir-ai `323` (hyperedge variable-arity — the higher-priority structural gap)
> **Classification:** Public

---

## TL;DR

OntoKG organizes a raw open-domain KG (Wikidata, 100M entities) into a typed property graph by **classifying every property as either intrinsic** (node attribute for tabular lookup, e.g. birth date) **or relational** (traversable graph edge, e.g. employer), routing each to a typed schema module. The result is a declarative, portable schema of 8 categories × 94 modules (56 intrinsic, 38 relational) covering 34M entities at 93.3% category coverage, with 5 downstream applications (ontology analysis, benchmark auditing, entity disambiguation +2.4pp on BLINK, domain customization, LLM-guided extraction).

**Verdict: GAIN for katgpt-rs.** The core mechanism (intrinsic/relational routing = the property graph model, Angles 2018) is well-known prior art AND is *already implicit* in our raw-vs-latent sync boundary rule (Research 141 §1.3: physical → TxDelta = "intrinsic", semantic/social → KG triple = "relational"). Research 141 §3.3 #7 explicitly rated the property graph model (LPG) as **LOW priority** for a top-down MMO. The paper is well-executed KG engineering at Wikidata scale, but its transferable insight for us is modest: a *declarative* schema layer that makes the implicit intrinsic/relational split explicit, inspectable, and committable — plus cross-cutting relational modules (a minor structural refinement to our per-SenseKind `ACTION_*` namespaces). No new capability class; no Super-GOAT; the agentic-LLM refinement loop is training/external-LLM-flavored and doesn't fit modelless runtime.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is the **declarative intrinsic/relational routing decision as a schema-level type assignment** — `τ(m) ∈ {Intrinsic, Relational}` per module, made at schema design time rather than implicitly at the bridge-rule layer. This is a schema-engineering insight, not a new inference primitive.

---

## 1. Paper Core Findings

### 1.1 The intrinsic-relational routing mechanism (§3.1)

A schema `S = {(C_i, G_i, M_i)}` partitions entities into k **categories** via gate matching (type-assertion property values), then assigns each entity zero-or-more **modules** within its category. Each module `m` carries:

- **Type** `τ(m) ∈ {Intrinsic, Relational}` — the edge-boundary decision: intrinsic → node attribute (tabular lookup); relational → graph edge (traversal). Made explicitly at schema design time.
- **Indicators** `I(m)` — property-value conditions that trigger module assignment (presence-based or value-based).
- **Value properties** `Π(m)` — properties extracted for matched entities.

Intrinsic modules define **what an entity IS** (identity-providing, category-confined: chemical formula, birth date). Relational modules define **what an entity CONNECTS TO** (cross-cutting domains: military spans 6 categories, religion spans 7). An entity receives exactly one category, zero-or-more intrinsic modules, zero-or-more relational modules.

### 1.2 Iterative schema refinement with agentic LLM oracles (§3.2, §4.3)

Three decision oracles drive refinement: `δ_c` (category assignment for unclassified types), `δ_m` (module assignment for new gate values), and a refinement oracle (create/merge/split modules). OntoKG implements these as an **agentic Claude Opus 4.6 workflow** with grounding tools (LMDB label lookup, live SPARQL P31 query, tag validator, coverage analysis, unclassified-hub analysis) to eliminate hallucinated QIDs/PIDs. Convergence target: θ_c ≈ θ_m ≈ 0.9.

### 1.3 Wikidata case study results (§4)

- **34.6M core entities** (from 100M after rule-based cleaning: structural classification, source signature matching, curation score, ratio-based safety net).
- **94 modules** (56 intrinsic, 38 relational) across 8 categories (people, places, creative_works_media, knowledge, science, organizations, events_actions, products_artifacts).
- **93.3% category coverage**, 98.0% module-assignment rate among classified.
- **34.0M nodes, 61.2M edges** across 38 relationship types.
- Rust classifier: ~110K entities/sec on M4 Max; full 34.6M classification in ~5 min.

### 1.4 Five downstream applications (§5)

1. **Ontology structure analysis** — bipartite category × relational-module view reveals domain clusters (governance, cultural, economic).
2. **Benchmark annotation auditing** — ontological classifier as independent tiebreaker between AIDA-YAGO and CleanCoNLL (7.8:1 ratio favoring CleanCoNLL corrections).
3. **Entity disambiguation** — module-based type profiles (2.52 labels/entity vs YAGO's 1.42) → +2.4pp macro accuracy on BLINK controlled-candidate subset.
4. **Domain customization** — declarative YAML schema as parameterized generator; education module decomposed into 9 facets.
5. **LLM-guided extraction** — schema as prompt instructions for zero-training entity extraction.

---

## 2. Distillation

### 2.1 What we ALREADY have (prior art — Research 141 mapping)

Research 141 (`riir-ai/.research/141_KG_Triple_Typology_Reference_and_Structural_Gaps.md`) is the **direct prior-art reference**. The mapping is near-complete:

| OntoKG concept | Our shipped equivalent | Status |
|---|---|---|
| **Category** `C_i` (people/places/orgs) | **`SenseKind`** (`crates/katgpt-types/src/lib.rs`) — 6 always-on semantic domains (Common/Fighter/GameTheory/Spatial/Social/Skill) + feature-gated SpectralThreat | ✅ shipped — but oriented to NPC cognitive domains, not encyclopedic entity types |
| **Module** `m` (grouping of related properties) | **`ACTION_*` namespace** (`kg.rs`) — 1000s=physical, 2000s=combat, 3000s=economic, 4000s=spatial, 5000s=social, 6000s=skill, 7000s=reputation | ✅ shipped — but confined to ONE SenseKind each (no cross-cutting) |
| **Gate matching** (type-assertion → category) | **`classify_sense(transition) → SenseKind`** (`kg.rs`) — `match` on `transition.action` | ✅ shipped |
| **Module indicators** `I(m)` | The `match` arms in `classify_sense` (value-based routing by action ID) | ✅ shipped |
| **Value properties** `Π(m)` + extraction | **`extract_triples(transition)`** → `KgTriple { head, relation, tail, confidence }` | ✅ shipped |
| **Intrinsic module** (node attribute, tabular) | **Physical domain → TxDelta** (raw, synced, NOT a KG triple) — per Research 141 §1.3 raw-vs-latent rule | ✅ shipped **implicitly** — the intrinsic/relational split IS our raw/latent boundary |
| **Relational module** (graph edge, traversal) | **Semantic/Social domain → KG triple** (latent similarity → triple emission) | ✅ shipped |
| **Schema as portable artifact** | **`freeze_with_merkle` / `thaw_verify_merkle`** (`kg.rs`) — BLAKE3-committed KG snapshots | ✅ shipped — **stronger** than OntoKG's YAML (tamper-evident, committable) |
| **Cross-cutting relational module** (spans ≥2 categories) | ❌ **gap** — each `ACTION_*` routes to exactly ONE SenseKind via `classify_sense` returning a single `SenseKind` | ❌ minor gap |
| **Declarative YAML schema** | ❌ not shipped — our typology is in Rust consts + match arms, not a declarative config | ❌ minor gap (design-time tooling) |

**The critical prior-art finding:** Research 141 §1.3 already establishes that the **intrinsic/relational distinction IS our raw-vs-latent sync boundary rule**:

> | Domain | Examples | KG triple? |
> |--------|----------|------------|
> | **Physical** (intrinsic) | position, distance, HP, wallet | **NO** — TxDelta |
> | **Semantic** (relational) | emotion, mood, curiosity, style | **YES** |
> | **Social** (relational) | encounters, relationships, factions | **YES** |

OntoKG's `τ(m) = Intrinsic` ≡ our "physical → TxDelta (raw, synced, not a triple)". OntoKG's `τ(m) = Relational` ≡ our "semantic/social → KG triple (latent, local until committed)". **The distinction ships; it's just implicit in the bridge rule, not declarative in a schema.**

Research 141 §3.3 #7 explicitly rates the property graph model (LPG) as **LOW priority** for a top-down MMO:

> 7. **[LOW]** Hierarchical ontology, **property graph (LPG)**, concept lattices (FCA), quivers, scene/molecular graphs — not GOAT-relevant for a top-down MMO.

### 2.2 What OntoKG adds that we DON'T have

1. **Declarative `τ(m) ∈ {Intrinsic, Relational}` type assignment.** We make this decision implicitly at the bridge-rule layer (physical → TxDelta, semantic → triple). OntoKG makes it explicit and inspectable per-module in a YAML config. *Engineering refinement, not a new capability.*

2. **Cross-cutting relational modules.** OntoKG's "military" spans 6 categories; "religion" spans 7. Our `classify_sense` returns exactly ONE `SenseKind` per action — an `ACTION_ATTACK` is always `FighterSense`. A cross-cutting module would let one action route to multiple senses (e.g., attack → FighterSense + SocialSense for reputation impact). *Minor structural gap — see §2.3 fusion.*

3. **Iterative schema refinement with agentic LLM.** Uses Claude Opus 4.6 + grounding tools. **Modelless check (§3.5):** the three oracles (δ_c category, δ_m module, refinement) require semantic judgment that no deterministic construction provides. Path 1 (freeze/thaw) — N/A (schema design, not weight correction). Path 2 (deterministic LoRA) — N/A (symbolic, not weight-space). Path 3 (latent projection) — N/A (schema design is symbolic). → **genuine external-LLM dependency**, but this is **design-time tooling, not runtime**. Does not fit the modelless runtime mandate; would belong in an offline schema-design CLI if pursued.

4. **Entity disambiguation via module profiles.** N/A for us — our entity IDs are deterministic `u64` (no disambiguation needed at runtime).

5. **Domain customization via YAML.** Declarative parameterized schema generator — interesting for game design (different zones → different relational module sets) but design-time, not runtime.

### 2.3 Fusion (novelty TBD → Issue, NOT committed as Super-GOAT)

Three fusion angles, in priority order:

**Fusion A — Cross-cutting relational modules × Research 141 hyperedge gap (#2).** OntoKG's cross-cutting relational module (spans ≥2 categories) is structurally analogous to Research 141's variable-arity hyperedge gap (Plan 323). A relational module connecting 6 categories ≈ a hyperedge connecting 6 entity types. The fusion: extend `classify_sense` to return a bitset of SenseKinds (or emit multiple triples with different `relation` IDs), enabling an action to carry multi-sense semantics. *This is the only fusion with a genuine structural-gap target; file as Issue, not Plan 323 replacement.*

**Fusion B — Declarative schema × `KgTripleTemplate` × BLAKE3 commitment.** `vibe.rs` already ships `KgTripleTemplate { subject, predicate, object: [u8; 32] }`. Extending it with a `τ ∈ {Intrinsic, Relational}` field per template slot, frozen via `freeze_with_merkle`, yields a committable, inspectable schema artifact. *Engineering refinement; GAIN.*

**Fusion C — OntoKG intrinsic/relational × SLoD (Research 208) continuous zoom.** At coarse SLoD σ, only relational modules visible (cross-cutting social skeleton); at fine σ, intrinsic attributes also visible (per-NPC belief scalars). Hierarchical schema resolution. *Speculative; needs SLoD to ship first.*

**Why NOT Super-GOAT:** the intrinsic/relational distinction is (a) well-known prior art (property graph model, Angles 2018, Hogan et al. 2021 — both cited in OntoKG §2.4), (b) *already implicit* in our raw/latent boundary rule per Research 141 §1.3, and (c) explicitly rated LOW priority for our use case per Research 141 §3.3 #7. The novelty gate Q1 (no prior art) fails on both axes (global literature + our codebase). Q2 (new class of behavior) fails — it's making explicit what's implicit. Q3 (product selling point) is weak. Q4 (force multiplier) is partial. Per the no-candidate-escape-hatch rule, fusion angles with novelty TBD are filed as **Issues**, not committed as Super-GOAT in this note.

---

## 3. Verdict

**GAIN — useful schema-engineering refinement; not a new capability class.**

| Tier | Criteria | This paper |
|------|----------|-----------|
| ~~Super-GOAT~~ | Novel mechanism + new capability class + selling point + force multiplier | ❌ intrinsic/relational = property graph model (prior art); implicit in our raw/latent rule; LOW priority per Research 141 |
| ~~GOAT~~ | Provable gain over existing approach, promotes to default | ❌ no provable runtime gain; the gain is design-time inspectability |
| **GAIN** | Incremental improvement, useful but not headline-worthy | ✅ declarative schema + cross-cutting modules are useful refinements |
| ~~Pass~~ | Not relevant | ❌ has *some* transferable value for KG schema layer |

**One-line reasoning:** The intrinsic/relational routing IS our raw/latent sync boundary rule (Research 141 §1.3), already shipped implicitly; making it declarative is an engineering refinement, and the property graph model is explicitly LOW priority for a top-down MMO (Research 141 §3.3 #7).

### MOAT gate per domain (§1.6)

| Domain | Verdict | Reasoning |
|--------|---------|-----------|
| **katgpt-rs** (public engine) | **Neutral GAIN** | A generic `KgSchemaModule { tau: Intrinsic/Relational, indicators, value_props }` trait is a paper-derived fundamental primitive, but it's a schema-engineering utility, not an inference primitive. Ships behind feature flag if pursued; does NOT strengthen the engine moat. |
| **riir-ai** (private runtime) | **Neutral GAIN** | The game instantiation (SenseKind × ACTION_* as declarative modules) touches the KG system but doesn't reach pillar-level. Research 141 already owns this space and prioritized hyperedges (#2) and reification (#1) above the property graph model (#7 LOW). |
| **riir-neuron-db** | **Out of scope** | Schema commitment is a runtime/chain concern, not a shard-storage concern. `KgTripleTemplate` in `vibe.rs` is the closest hook but it's a vibe-schedule template, not a schema layer. |
| **riir-chain** | **Out of scope** | No commitment/transport angle. |
| **riir-train** | **N/A** | The agentic-LLM refinement is design-time tooling, not training-method research. |

**Recommended follow-up: Issue, not Plan.** Per the global rule ("Create issue at ./issues for optimization or refactor task, do not create plan"), the cross-cutting relational module gap (Fusion A) is an optimization/refinement of the existing KG system, not a new primitive. File as `katgpt-rs/.issues/NNN_cross_cutting_relational_modules.md` if pursued. Research 141's existing Issue queue (edge reification, event-centric KG, bitemporal, RCC8, node typing) remains higher priority.

---

## 4. Honest Assessment

### Strengths of the paper
1. **Clean formalization** — the `S = {(C_i, G_i, M_i)}` schema with `τ(m) ∈ {Intrinsic, Relational}` is a well-executed operationalization of the property graph model at Wikidata scale.
2. **Agentic grounding tools** — LMDB label lookup + live SPARQL + tag validator to eliminate LLM hallucination at the source is a sound engineering pattern.
3. **Five downstream apps** demonstrate the schema is a reusable artifact, not just a construction byproduct.
4. **Rust + Python implementation** — 110K entities/sec is respectable.

### Why the gain is modest for us
1. **Domain mismatch.** OntoKG targets encyclopedic KG construction (Wikidata: 100M entities, ad hoc types, no schema enforcement). Our KG is **runtime-emitted from game self-play** with deterministic entity IDs and a fixed `SenseKind × ACTION_*` typology. We don't have the "unconstrained open editing" problem OntoKG solves.
2. **Implicit prior art.** Research 141 §1.3 already maps intrinsic/relational to our raw/latent boundary. The paper's contribution is making it declarative — useful for inspectability, but not a capability gain.
3. **Research 141 prioritization.** The property graph model is explicitly LOW priority (#7) below hyperedges (#2), reification (#1), event-centric KG (#3), bitemporal (#4), RCC8 (#5), and node typing (#6). This paper doesn't change that ranking.
4. **Modelless constraint.** The agentic-LLM refinement loop doesn't fit runtime; it's design-time tooling at best.

### What we'd build (if pursued, GAIN-tier, behind feature flag)

Phase 1: `kg_schema` feature flag in katgpt-core
- `KgSchemaModule` trait with `tau: ModuleType { Intrinsic, Relational }`, `indicators`, `value_props`
- `KgSchema` struct holding categories + modules, serializable to a committable format
- `route_property(schema, property) -> ModuleType` — the declarative router

Phase 2: riir-ai instantiation
- Map `SenseKind` → category, `ACTION_*` → module, with explicit `tau` per module
- Extend `classify_sense` to optionally return a bitset for cross-cutting modules (Fusion A)

Phase 3: Benchmark
- Before/after: schema inspectability (can a designer query "which properties are intrinsic?"), cross-cutting module coverage (how many actions gain multi-sense semantics)

*Not planned in this session — file as Issue if pursued.*

---

## TL;DR

OntoKG organizes Wikidata (100M entities) into a typed property graph by routing each property as **intrinsic** (node attribute) or **relational** (graph edge), producing a declarative 8-category × 94-module schema at 93.3% coverage. **Verdict: GAIN.** The intrinsic/relational distinction is the property graph model (Angles 2018, well-known prior art), is *already implicit* in our raw-vs-latent sync boundary rule (Research 141 §1.3: physical → TxDelta = intrinsic, semantic → KG triple = relational), and Research 141 §3.3 #7 explicitly rated the property graph model as **LOW priority** for a top-down MMO. The transferable insights — declarative schema layer, cross-cutting relational modules, domain customization — are useful engineering refinements but not new capability classes. The agentic-LLM refinement loop is design-time tooling that doesn't fit modelless runtime. **Recommended follow-up: Issue for cross-cutting relational modules (Fusion A with Research 141 hyperedge gap), not Plan.** Research 141's existing higher-priority gaps (hyperedges #2, reification #1, event-centric KG #3) remain the active KG structural work.

> **PASS-Redirects (synthesis):** NVIDIA [github.com/NVIDIA/auto-ontology "Auto Ontology"] (Apache-2.0) — LLM-GENERATED enterprise semantic layer over SQL metadata (concepts/mappings/joins/SQL-metrics/lineage/certification compiled by reasoning+embed+rerank model roles over pgvector, MCP-served text-to-SQL; BEAVER #1, BIRD-Interact #2) — a cousin of this note's SCHEMA side only: generation is model-based end to end, so nothing distills into a modelless lane; the doctrine mapping (their `check_answerable` coverage gate ≡ `CorpusDistanceGate` `σ(scale·(max_sim−mid))` abstention, answer-returns-SQL ≡ decision receipts, certify-before-serve ≡ golden fixtures/GOAT, fold-corrections-back ≡ self_evolve) + the topogram/gram-family relation verdict (related by INTENT, unrelated by MECHANISM) live on riir-reflex Research 015. Verdict PASS 2026-11-17 (README-level read, no clone — Python enterprise text-to-SQL is outside every riir-refine domain and is not a decision-index entrant).
