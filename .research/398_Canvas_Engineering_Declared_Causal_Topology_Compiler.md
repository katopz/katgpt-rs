# Research 398: Canvas Engineering — Declared Causal Macrostructure Compiler for Latent Dynamics

> **Source:** Jacob Valdez (CommandAGI). *Canvas Engineering: Declared Causal Macrostructure for Reverse-Diffusion Latent Dynamics*. July 2026. http://commandagi.com/research/canvas-engineering.pdf · code: github.com/commandAGI/canvas-engineering (Apache 2.0)
> **Date:** 2026-07-09
> **Status:** Active
> **Related Research:** 311 (Analytic Lattice Encoder/Decoder — closest cousin), 229 (ProgramAsWeights / spec→compile), 295 (AC-GPT arbitrary-conditional attention), 097 (Training-Free Looped Transformers), 266 (FPRM damped fixed-point halting), 219 (DEC operators / Stokes substrate), 280 (RTDC resolution-tiered commitment), 353 (program-synthesized head surrogates), 396 (MFA region-conditioned steering)
> **Related Plans:** 330 (analytic lattice primitive), 313 (AC-Prefix mask builder), 416 (region subspace field), 108/136 (looped transformer modes), 419 (this note's plan — canvas schema compiler)
> **PASS-Redirects (synthesis):** Li, Yang, Chen, Xu, Li, Su [arXiv:2605.22566 "GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving"] (ICML 2026) — PASS. GraphFlow's wGraph (global operation DAG unifying atomic operations across workflows, with task-conditioned subgraph instantiation via GNN + MLP edge scoring) is Canvas Schema (this note / Plan 419) applied at agent-workflow granularity. Canvas already ships `CanvasSchema → CanvasTopology (directed connection graph) → attention mask + loss weight mask` with reachability semantics; GraphFlow's unification-across-workflows (one global DAG vs Canvas's per-instance topology) is a refactor opportunity, not a new primitive. The GNN-based edge scoring is offline supervision (training-side); the modelless analog (task-conditioned dot-product routing via Set Attention P354 + manifold_bandit P370) ships. See also Research 368 (AutoMem — workflow orchestration as decision structure) + Research 193 (BFCF — topology-aware KV sharding).
> **Classification:** Public

---

## TL;DR

Canvas engineering declares a **typed schema** (regions, their connectivity, temporal frequencies, loss roles) and a **compiler** lowers it to attention masks + loss weights over a stock pretrained backbone. The headline empirical results (1.73× parameter efficiency, 350K-frozen beats 11.7M-unfrozen, cortical R²=0.825) are **training-dependent** — they require training the DiT *within* the declared structure. The **modelless-distillable value** is narrower and architectural:

1. **The compiler itself** — `CanvasSchema → CanvasLayout (region index sets) + CanvasTopology (directed connection graph) → attention mask + loss weight mask`. Pure structure compilation, zero gradient descent.
2. **Reachability-as-causal-semantics** — for a binary mask, an absent edge `a ↛ b` is an **exact marginal independence** in the generative dynamics (the empty-conditioning case of d-separation); one denoiser pass with L blocks moves information along paths of length ≤ L; K sampling steps compose to horizon K·L. This is a graph-theoretic guarantee that holds *by construction*, not by regularization.
3. **`transfer_distance`** — semantic-type compatibility via frozen embeddings (camera vs joint-angles distance), making modality compatibility a computable scalar instead of a judgment call.
4. **Schema-mediated latent exchange** — two models that share a schema can exchange latent state directly (no re-encoding) because the schema fixes what every position means. This is a freeze/thaw pattern.

**Distilled for katgpt-rs (modelless, inference-time):** a generic `CanvasSchema` compiler + `reachability_horizon` primitive + `transfer_distance` scalar. No game IP, no chain IP, no shard IP. The compiler emits an `AttentionMaskSpec` (consumable by the existing AC-Prefix / VortexFlow mask builders) and a `LossWeightMask` (consumable by training-time callers; at inference only the mask is used). The looped-attention half of the paper is **already distilled** (Research 097 → Plan 136 `LoopMode::TrainingFree`; Research 266 → FPRM Gain) and is NOT re-distilled here.

**Verdict: GOAT.** The compiler + reachability semantics is a novel modelless primitive connecting ≥3 pillars (DEC topology, HLA per-entity latent state, latent_functor typed programs, AC-Prefix masks, freeze/thaw exchange). But (a) the constituent sub-primitives largely ship (`region_subspace_bridge` Plan 416, AC-Prefix Plan 313, StillPerceiver, `LoopMode::TrainingFree`); (b) the headline empirical value is training-dependent and requires riir-train to validate; (c) the reachability semantics, while elegant, is a reframing of "sparse attention = causal graph" that we implicitly already use. It is a **unifying abstraction** (a type system for latent state), not a new empirical capability class at the modelless level. A fusion idea (canvas × DEC reachability × region_subspace × freeze/thaw schema exchange) is flagged as a potential Super-GOAT **if** a defend-wrong PoC proves the compiled canvas improves per-NPC behavior modellessly over the un-unified constituents — tested in a PoC (resolved 2026-07-09, §7) that could not isolate a canvas-attributable gain from the tuned classifier. The modelless compiler ships; the empirical Super-GOAT claim waits for a cleaner PoC.

---

## 1. Paper Core Findings

### 1.1 The canvas (§2.1)

A canvas is a 3D spatiotemporal grid `(T, H, W)` of `d`-dim latent positions, flattened into the token sequence of a diffusion transformer (DiT). A `CanvasLayout` partitions it into named regions; each `RegionSpec` carries bounds, a temporal period (so a `period=4` "thought" region coexists with a `period=1` "perception" region on one canvas), `is_output` + `loss_weight` (type-directed loss codegen), an optional `semantic_type` with a frozen embedding, and a `default_attn` function family.

### 1.2 Topology = the causal graph = the attention mask (§2.2 — the load-bearing claim)

A `CanvasTopology` is a directed multigraph of cross-attention operations. Each `Connection(src, dst)` licenses `src` tokens to query `dst` keys/values. **The absence of an edge is a hard prohibition, not a soft prior.** Reverse diffusion applies the denoiser to the full canvas at every step, so which regions can influence which under iteration is exactly the topology's transitive closure. Declaring the topology therefore declares a causal interaction graph over the generative dynamics.

Convenience constructors: `dense` (fully connected, the degenerate standard-transformer case), `isolated` (block-diagonal), `hub_spoke` (shared coordinator), `causal_chain` (A→B→C), `causal_temporal` (same-frame self + previous-frame cross, no future leakage).

### 1.3 Core mathematics (§2.3 — four small pieces)

1. **Regions are index sets.** Region `r` with bounds `(t0,t1,h0,h1,w0,w1)` owns `I_r = {tHW+hW+w : ...}`. Struct-offset arithmetic.
2. **Topology compiles to a mask.** Edges lower to `M ∈ R^{N×N}_{≥0}`: `M_ij = max_k w_k · 1[i∈I_{r_k}] · 1[j∈I_{s_k}] · A_τ(t(i),t(j))` where `A_τ` is the temporal alignment predicate. Per-edge cross-attention: `Y_{I_r} += w · softmax(Q_{I_r} K^T_{I_s}/√d) V_{I_s}`. Binary mask (w∈{0,1}) coincides with the monolithic form (add `log M` to logits with `log 0 = -∞`).
3. **Loss participation is a weight vector.** `ω_i = Σ_r 1[i∈I_r] · loss_weight_r · 1[is_output_r]`. Training minimizes position-weighted denoising loss; context positions are clamped (inpainting-style conditioning).
4. **Reachability is the causal statement.** Information-flow graph `G` has arc `s→r` for every connection (`r` queries `s`). One denoiser pass with L blocks moves info along paths ≤ L; K sampling steps compose to horizon K·L. **Load-bearing direction (exact, by construction):** if `G` has no directed path `a→b`, then `a` cannot influence `b`'s denoised value — exact marginal independence (empty-conditioning d-separation). Converse is bounded, not guaranteed.

### 1.4 A type system, literally (§2.4 — Table 1)

| Type concept | Canvas equivalent | Implementation |
|---|---|---|
| Struct field (offset+size) | Region bounds | `region_indices()` |
| Field annotation (period, loss_weight, sem. type) | `RegionSpec` | — |
| Pointer / reference | `Connection` | `CanvasTopology` |
| Function signature | Topology + fn | `attention_ops()` |
| Type-directed codegen | Loss mask | `loss_weight_mask()` |
| ABI compatibility | Schema compat | `compatible_regions()` |
| Coercion cost | Transfer distance | `transfer_distance()` |

Two models sharing a schema can exchange latent state directly — no tokenization, no re-encoding — because the schema fixes what every position means. Across differing schemas, a region can declare its modality meaning as a string + fixed embedding, making modality compatibility a computable distance.

### 1.5 Attention function types (§2.5)

17 registered function families: dot-product (`cross_`, `linear_`, `sigmoid_attention`), gating (`gated`), compression (`perceiver`, `pooling`), transfer (`copy`), state-space (`mamba`, `rwkv`), convolution (`hyena`), sparse (`local_`, `sparse_attention`), meta (`none`, `random_fixed`, `mixture`). Resolution: `connection.fn → region.default_attn → global cross_attention`.

### 1.6 Compilation: entity schema → canvas (§3)

The compiler accepts typed entity declarations (nested Python dataclasses) and flattens the entity/relationship graph into layout + topology. **Every nested type automatically receives a coarse-grained field** — a compressed representation that bottlenecks all cross-level attention. 50 vehicles under dense cross-attention would cost O(50² × fields²); under coarse-graining each interacts through its 4×4 summary → O(50 × 16). Deep nesting chains the bottlenecks: `us.macro.gdp → cn.macro.inflation` is forced through `us.macro → us → regime → cn → cn.macro`. Hierarchical abstraction is a topological consequence, not an emergent hope.

The program layer (v2) adds process semantics per region: family (observation/state/memory/residual/action), carrier (deterministic/diffusive/filter/memory/residual), clock (periodic or event-triggered with composable firing rules like `Or(periodic(4), on("err.prediction", gt=0.5))`), compile mode (runtime/freeze/constant/export).

### 1.7 Empirical record (§5 — HONESTLY TRAINING-DEPENDENT)

CogVideoX-2B on BridgeData V2: 26 experiments, 236 runs.
- **Looped attention:** 3-loop frozen (350K trainable params) beats every unfrozen condition (11.7M). 1.73× parameter efficiency (p<0.001). Loop representations converge toward fixed points (cosine to loop 1: 0.926→0.996).
- **Calibration findings (honest, not narrative):** (a) looped-attention benefit at this scale is weight-sharing regularization, NOT detectable iterative reasoning; (b) flat-canvas co-residence **degraded** joint prediction by 19% (p<0.0001) — binding must be declared, not hoped for; (c) loss is nearly insensitive to token allocation (α=0.011: doubling a region's tokens moves loss 0.8%) — canvas design is forgiving.
- **Cortical world model:** 23 brain regions (Destrieux atlas) wired by 42 known cortical pathways; recovers TRIBE-v2-estimated cortical dynamics at R²=0.825 using 19.6% of possible connections. Ablation: at 135 features a fully dense model matches the cortical one — topology is a convergence prior, not a capacity advantage.

### 1.8 Open problems (§6)

- **Representation stability (the linchpin):** everything interoperable rests on a platonic-representation claim that identical declared structure induces predictably aligned latent geometry across seeds/datasets/backbones. Plausible, unproven.
- **Binding:** whether declared topology recovers/exceeds independent-model performance is the decisive ablation.
- **Scale:** all evidence at 2B params and robot-video scale.
- **Learned topology:** today the graph is authored; propose/prune edges under sparsity pressure is the natural continuation.

---

## 2. Distillation (modelless, inference-time)

### 2.1 Training-dependent vs modelless split (the honest cut)

| Paper component | Modelless? | Why |
|---|---|---|
| The compiler (`schema → layout + topology → mask + loss weights`) | **YES** | Pure structure compilation. Zero gradient descent. |
| Reachability = exact marginal independence | **YES** | Graph-theoretic guarantee, holds by construction for binary masks. |
| `transfer_distance` (semantic type compatibility) | **YES** | Frozen-embedding cosine; no training. |
| Schema-mediated latent exchange (same-schema models swap latents) | **YES** | A freeze/thaw pattern; structure is fixed, content is exchanged. |
| Coarse-grained bottleneck construction | **YES (structure)** / **NO (content)** | The bottleneck *layout* is declared; the *summary content* is learned. |
| Looped attention (frozen DiT blocks iterated K× with zero-init learned embeddings) | **PARTIAL** | The zero-init learned embeddings require training. The frozen-block iteration itself is `LoopMode::TrainingFree` (Plan 136). |
| 1.73× parameter efficiency | **NO** | Requires training the DiT within the declared structure. |
| 350K-frozen beats 11.7M-unfrozen | **NO** | Training result. |
| Cortical R²=0.825 | **NO** | Training result. |
| Flat-canvas 19% degradation without declared topology | **NO (it's a loss)** | Applying a mask to an untrained-for-it backbone degrades performance. |

**The modelless value is the compiler + reachability semantics + transfer_distance + schema exchange. The empirical headline value is training-dependent.** Applying a declared-topology mask to a frozen backbone that was NOT trained within it is a documented **loss** (paper §5 calibration finding #2). The modelless primitive therefore ships the *compilation* and the *guarantee*; the *behavioral gain* requires riir-train.

### 2.2 §3.5 modelless-unblock check (mandatory before any riir-train deferral)

The paper's headline gains appear to need training. Exhausting the three modelless paths:

1. **Freeze/thaw snapshot correction** — can a frozen snapshot + thaw recover the gain? **NO.** The gain comes from training the network to *use* the declared topology. A frozen untrained-for-it backbone degrades by 19% (paper §5). Freeze/thaw can exchange schema-compatible latents but cannot make an untrained backbone respect a topology it never learned.
2. **Raw/lora reader-writer hot-swap** — can a deterministically constructed LoRA enforce the topology? **PARTIAL.** The topology is a *mask* (hard prohibition), not a weight. A reader-LoRA could *approximate* the mask softly, but the paper's reachability guarantee requires a *binary* mask (absent edge = exact independence). A soft LoRA approximation breaks the guarantee. So the LoRA path gives an approximation, not the exact guarantee.
3. **Latent-space correction** — can a dot-product projection + sigmoid gate enforce region structure? **PARTIAL.** This is essentially what `region_subspace_bridge` (Plan 416) already does — it projects NPC latent state onto region centroids + local axes. But that's region-conditioned *steering*, not declared *causal topology* with reachability guarantees.

**Conclusion:** the modelless paths can APPLY declared structure and can EXCHANGE schema-compatible latents, but they cannot ACHIEVE the paper's empirical parameter-efficiency / cortical-fit gains without training the network within the declared topology. The training dependency is genuine for the empirical headline. The modelless primitive (compiler + reachability) ships regardless, because it is structure compilation, not a behavioral claim.

### 2.3 Vocabulary translation (paper ↔ codebase) — MANDATORY before novelty claim

| Paper term | Codebase-equivalent | Verified shipped? |
|---|---|---|
| "canvas" / "structured latent space" | typed latent state, HLA 8-dim, `ZoneGeometryPod` 8-lane, `RegionSubspaceField` K-region | **YES (pieces)** — no unified "canvas" type |
| "region" / `RegionSpec` / `region_indices()` | `RegionSubspaceBridge::apply_to_zones_batch(region_indices)`, `ZoneGeometryPod` lanes | **YES (pieces)** — region-conditioned steering ships (Plan 416) |
| "topology" / `Connection` / `CanvasTopology` | sparse attention pattern, `AttentionMaskSpec`, AC-Prefix mask, VortexFlow routing | **YES (pieces)** — no unified topology-as-graph compiler |
| "attention mask" / "causal interaction graph" | AC-Prefix `AcPrefixMask`, DFlash non-causal mask, causal masks | **YES (pieces)** — mask builders ship (Plan 313) |
| "coarse-grained field" / "hierarchical bottleneck" | Egg/Shell bridge (pillar 1), StillPerceiver (Plan 245), perceiver compression | **YES (specific instances)** — no generic "bottleneck-at-every-nesting-level" primitive |
| "schema" / "type system" / "ABI" | `types.rs` decoupled structs, latent_functor typed programs, Analytic Lattice entity encoders | **YES (pieces)** — no unified schema compiler |
| "transfer_distance" / "semantic_type" | schema-centroid cosine (Plan 237), Rosetta neurons best-buddies, dot-product projection | **NO** — no semantic-type-distance primitive |
| "looped attention" / "universal transformer" | `LoopMode::WeightShared` (Plan 108), `LoopMode::TrainingFree` (Plan 136), FPRM (R266) | **YES — fully distilled** |
| "reachability" / "d-separation" / "marginal independence" | DEC `exterior_derivative`, `CellComplex` paths, Viable Manifold Graph transitive closure | **YES (graph substrate)** — no reachability-on-attention-mask primitive |
| "compile_schema" / "compiler" | ProgramAsWeights spec-compile (R229), Percepta C→WASM→weights, Analytic Lattice entity→lattice | **YES (parallel axes)** — no schema→layout+topology compiler |
| "schema-mediated latent exchange" | freeze/thaw (`MerkleFrozenEnvelope`), `LoRAWeightVersion`, `CommittedFieldBlend` BLAKE3 commit | **YES (freeze/thaw substrate)** — no schema-keyed exchange |

### 2.4 Prior-art surface — what already ships (verified grep + read)

1. **`riir-ai/crates/riir-engine/src/latent_functor/region_subspace_bridge.rs`** (Issue 424, wires Plan 416) — zone-conditioned two-mode NPC personality steering. `apply_to_zones_batch(region_indices, centroid_alpha, local_offsets)`. Centroid interpolation + local subspace offset. Pre-discovered MFA artifact `{μ_k, W_k, log π_k, Ψ^{-1}}` applied modellessly. **This is the region concept, at per-NPC latent-state granularity.** Ships declared *regions* but NOT declared *causal topology* between regions.
2. **Plan 313 (AC-Prefix)** — `AcPrefixMask` builder, three-region attention rule, branch-free `attends(i,j)`, bit-packed. Ships the mask-builder primitive for a specific topology (xc-bidirectional | causal-everywhere-else).
3. **Plan 136 (`LoopMode::TrainingFree`)** + Research 097 — K-stage Runge-Kutta β=0.5 sub-stepping on frozen blocks. The looped-attention half of the paper is fully distilled and shipped.
4. **Plan 245 (StillKV StillPerceiver)** — perceiver bottleneck as one of the 17 function families.
5. **Research 311 / Plan 330 (Analytic Lattice Encoder/Decoder)** — compile typed entity schema (Quest/Player/Boss/Zone) → typed-lattice coordinate → compose functional-attention operators. Closest cousin. Encoder half found redundant (`FourierEncoder::encode_*_into` ships). Novel: `compose_chain`, `batch_compose_chain`, ASOC cascade.
6. **katgpt-core `dec/`** — `exterior_derivative` (d), `codifferential` (δ), `hodge_decompose`, `CellComplex`, `CochainField`. The graph/topology substrate for reachability semantics.
7. **Research 229 (ProgramAsWeights)** — spec→compile. We compile specs into symbolic constraints (bitmap), they compile into neural weights. Parallel axis, not the same primitive.
8. **`MerkleFrozenEnvelope` / `CommittedFieldBlend`** — freeze/thaw + BLAKE3 commitment substrate for schema-mediated latent exchange.

**The gap (Q1):** no unified `CanvasSchema → CanvasLayout + CanvasTopology → AttentionMaskSpec + LossWeightMask` compiler; no `reachability_horizon(topology, L, K)` primitive; no `transfer_distance(region_a, region_b)` scalar. These three are genuinely missing.

### 2.5 Fusion — what novel combination does this enable?

**Fusion idea (novelty TBD — needs a PoC with a non-designer-tuned discriminator before Super-GOAT verdict; the first PoC could not isolate a canvas-attributable gain, see §7):**

> Canvas Engineering compiler × DEC reachability × `region_subspace_bridge` × freeze/thaw schema-mediated exchange × HLA per-NPC latent state → "A typed NPC cognitive stack where each NPC's latent state is a **compiled canvas** with declared causal topology (perception → affect → action, memory ↔ affect), reachability guarantees (perception cannot influence action without traversing affect, by construction), and schema-mediated freeze/thaw exchange (two NPCs with the same cognitive schema can swap latent state directly)."

The NEW capability this fusion would produce: **declared causal topology with reachability guarantees on per-NPC latent state, plus schema-keyed latent exchange.** None of the constituents alone does this:
- `region_subspace_bridge` does region-conditioned steering, NOT declared causal topology between regions.
- AC-Prefix does mask building, NOT schema compilation.
- DEC does topology/reachability, NOT on per-NPC attention masks.
- Freeze/thaw does exchange, NOT schema-keyed.

**But:** whether the compiled canvas + reachability semantics improves per-NPC behavior modellessly over the un-unified constituents is **unproven**. The paper's evidence is that declared topology helps *when trained within*; the modelless application (apply mask to frozen backbone) is a documented 19% loss. So the fusion's behavioral value at the modelless level is a hypothesis, not a proven gain. Per §3.6, a defend-wrong PoC in `riir-ai/crates/riir-poc/` is the right next step before any Super-GOAT re-evaluation — NOT a Super-GOAT verdict now.

### 2.6 Compute-unit translation (the R368 lesson — does NOT trigger here)

The paper has no "N LLM calls/step" structure. Its compute unit is "one DiT denoiser pass over the canvas." For us, the analog compute unit is "one latent_functor application over the NPC's typed latent state." No false-PASS risk from conflating LLM-as-implementation with LLM-as-mechanism — the paper is architecture, not agent orchestration.

---

## 3. Verdict — GOAT

**Tiers (high → low):**

| Tier | Criteria | Routing |
|---|---|---|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | Open primitive + private guide + plans |
| **GOAT** | Provable gain over existing approach, but not a new class of capability | Plan + implement, feature flag + benchmark |
| **Gain** | Incremental improvement, useful but not headline-worthy | Plan only, behind feature flag |
| **Pass** | Not relevant, OR training-only (→ riir-train note, stop) | One-line note |

**Verdict: GOAT.**

**One-line reasoning:** The `CanvasSchema` compiler + `reachability_horizon` + `transfer_distance` is a novel modelless primitive (structure compilation, zero gradient descent) with a provable correctness property (absent edge = exact marginal independence by construction) that unifies ≥3 existing pillars (DEC topology, region-conditioned latent state, AC-Prefix masks, freeze/thaw exchange); but (a) the constituent sub-primitives largely ship, (b) the headline empirical value (parameter efficiency, cortical fit) is training-dependent and requires riir-train to validate, (c) the reachability semantics is an elegant reframing of sparse-attention-as-causal-graph rather than a new mechanism. It is a unifying type-system abstraction for latent state, not a new empirical capability class at the modelless level.

### 3.1 Novelty gate (Q1–Q4)

- **Q1 (No prior art for the unified compiler?): YES.** Grep across all 5 repos (both paper vocab and codebase vocab) confirms: no `CanvasSchema`/`CanvasLayout`/`CanvasTopology`/`compile_schema`/`transfer_distance`/`reachability_horizon` type. Pieces ship (`region_subspace_bridge`, AC-Prefix, DEC, StillPerceiver, Analytic Lattice); the unified compiler does not.
- **Q2 (New class of behavior?): PARTIAL → NO at the modelless level.** The compiler enables "pointability" (typed, pointable latent state) and "declared causal topology with reachability guarantees." But the runtime behavior (sparse attention over typed regions) is something the constituents already do; the NEW thing is the compilation + guarantee, which is a structuring/refactoring gain, not a behavioral one. Critically, the paper's own evidence shows modelless application (mask on untrained backbone) is a 19% LOSS — so the modelless behavioral claim is unproven and currently negative.
- **Q3 (Product selling point?): YES (architectural), NO (empirical, modelless).** "Our NPC cognitive stack has a typed schema compiler with declared causal topology and reachability guarantees" is a real architectural selling point. "And it improves NPC behavior modellessly" is unproven.
- **Q4 (Force multiplier?): YES.** Connects to DEC topology, HLA per-entity latent state, latent_functor typed programs, AC-Prefix masks, freeze/thaw schema exchange, region_subspace steering. ≥4 pillars/systems.

**Q2 fails at the modelless level → not Super-GOAT now.** The fusion idea (§2.5) is flagged as a potential Super-GOAT if a PoC proves modelless behavioral gain — the first PoC (§7, resolved 2026-07-09) could not isolate a canvas-attributable gain from the tuned classifier, so Super-GOAT remains un-awarded. Per §1.5, "candidate" language is avoided: this is a GOAT with a tracked fusion follow-up, not a deferred Super-GOAT commitment.

### 3.2 MOAT gate per domain (§1.6)

- **katgpt-rs (public engine):** the compiler + reachability + transfer_distance is a **paper-derived fundamental primitive** (generic math, no game/chain/shard semantics). Ships behind feature flag `canvas_schema`. GOAT-gate decides promote/demote. **In scope.**
- **riir-ai (private runtime):** the fusion (typed NPC cognitive stack) is **pillar-level if the PoC passes**. Until then, deferred — the guide is NOT created now (GOAT verdict, not Super-GOAT).
- **riir-chain / riir-neuron-db:** schema-mediated latent exchange touches freeze/thaw, but the primary value is the compiler (katgpt-rs). No reroute needed.
- **riir-train:** the empirical validation (train DiT within declared topology, measure parameter efficiency at our scale) is a genuine riir-train follow-up. Noted, not blocked.

### 3.3 §3.6 defend-wrong PoC — NOT triggered now

The §3.6 PoC rule triggers for PASS verdicts that downgrade on "runtime analog already ships" OR for quality-parity claims. This verdict is **GOAT** (not PASS) and makes **no quality-parity claim** — it explicitly states the modelless behavioral gain is unproven and currently negative (paper's 19% degradation finding). The compiler primitive ships on its structural/correctness merits (reachability guarantee by construction), not on a behavioral-parity claim. A PoC is the right *follow-up* for the fusion's potential Super-GOAT re-evaluation (§7 records the first one, inconclusive), not a blocker for the GOAT.

---

## 4. Distilled primitive — what ships in katgpt-rs

Open primitive (Plan 419), feature flag `canvas_schema`, opt-in until GOAT gate:

```rust
// crates/katgpt-core/src/canvas/mod.rs (new module)

/// A typed region of the canvas: bounds + temporal period + loss role + semantic type.
pub struct RegionSpec {
    pub name: &'static str,
    pub bounds: CanvasBounds,        // (t0,t1,h0,h1,w0,w1)
    pub period: u32,                 // temporal update frequency
    pub is_output: bool,
    pub loss_weight: f32,
    pub semantic_type: Option<SemanticType>,  // frozen embedding for transfer_distance
    pub default_attn: AttentionFnFamily,
}

/// A directed connection: src queries dst. Absence = hard prohibition (exact marginal independence).
pub struct Connection {
    pub src: RegionId,
    pub dst: RegionId,
    pub weight: f32,                 // binary {0,1} for the reachability guarantee
    pub t_src: Option<i32>,          // temporal offset
    pub t_dst: Option<i32>,
    pub fn_family: Option<AttentionFnFamily>,  // overrides region default
}

pub struct CanvasSchema {
    pub layout: CanvasLayout,        // T, H, W, d_model, regions
    pub topology: CanvasTopology,    // connections
}

/// THE COMPILER. Pure structure, zero gradient descent.
/// schema → (region index sets, attention mask spec, loss weight mask)
pub fn compile_schema(schema: &CanvasSchema) -> CompiledCanvas {
    let region_indices: Vec<IndexSet> = schema.layout.regions.iter()
        .map(|r| region_indices(r)).collect();
    let mask = build_attention_mask(&schema.topology, &region_indices);
    let loss_mask = build_loss_weight_mask(&schema.layout);
    CompiledCanvas { region_indices, mask, loss_mask }
}

/// THE REACHABILITY GUARANTEE.
/// Returns the causal horizon: max path length reachable in K sampling steps × L attention blocks.
/// For binary masks: if no directed path a→b in topology, a cannot influence b (exact marginal independence).
pub fn reachability_horizon(topology: &CanvasTopology, n_blocks: usize, n_steps: usize) -> usize {
    n_blocks * n_steps
}

/// Returns true iff `from` can reach `to` within the horizon (transitive closure check).
pub fn can_reach(topology: &CanvasTopology, from: RegionId, to: RegionId, horizon: usize) -> bool {
    // BFS/DFS on the information-flow graph (arc s→r for connection r queries s)
    transitive_closure_reaches(topology, from, to, horizon)
}

/// Semantic-type compatibility via frozen embeddings (modelless routing scalar).
pub fn transfer_distance(a: &SemanticType, b: &SemanticType) -> f32 {
    1.0 - cosine(a.frozen_embedding(), b.frozen_embedding())
}
```

**Consumers:** the `CompiledCanvas.mask` is an `AttentionMaskSpec` consumable by AC-Prefix / VortexFlow / any sparse-attention path. The `reachability_horizon` / `can_reach` primitives compose with DEC's `CellComplex` path queries. The `transfer_distance` is a standalone routing scalar.

**What does NOT ship here (training-dependent, → riir-train):**
- Training a DiT within the declared topology (the empirical parameter-efficiency path).
- Looped-attention zero-init learned embeddings (already covered by `LoopMode::WeightShared` Plan 108).
- Representation-stability validation across seeds/backbones (paper §6 linchpin, open).

---

## 5. Risks and honest caveats

1. **The modelless behavioral gain is unproven (and currently negative).** The paper's own calibration finding #2: flat-canvas co-residence without declared topology degrades by 19%. Applying a declared-topology mask to a frozen untrained-for-it backbone is a documented loss. The compiler + reachability ships on structural/correctness merits; the behavioral gain requires riir-train. Do NOT claim modelless behavioral parity.
2. **The reachability guarantee requires a binary mask.** Soft masks (weighted edges, LoRA approximations) break the exact marginal independence. The guarantee is `w ∈ {0,1}` only.
3. **The "pointability" selling point assumes representation stability.** Paper §6 names this as the linchpin: identical declared structure must induce predictably aligned latent geometry across seeds/datasets/backbones. Plausible, unproven. Without it, "schema-mediated latent exchange" degrades to "exchange and hope."
4. **Most constituent primitives ship.** The compiler's value is unification + the reachability guarantee, not new constituent capability. A reviewer who greps `region_subspace_bridge` + AC-Prefix + StillPerceiver + `LoopMode::TrainingFree` will correctly observe that the pieces exist; the novel claim is the compiler that binds them + the reachability semantics.
5. **The paper is small-scale (2B params, robot video).** All empirical claims are calibration, not narrative (paper §5 is explicit about this). Do not over-claim.
6. **Looped attention is already distilled.** Research 097 → Plan 136 ships `LoopMode::TrainingFree`. Research 266 (FPRM) is Gain. The looped-attention half of this paper adds nothing over those. This note deliberately does NOT re-distill it.

---

## 6. Plan

→ `katgpt-rs/.plans/419_canvas_schema_compiler.md` (open primitive: compiler + reachability + transfer_distance, feature flag `canvas_schema`, opt-in until GOAT gate).

**Fusion PoC follow-up:** → `riir-ai/.benchmarks/043_canvas_npc_cognitive_stack_modelless.md` (defend-wrong PoC for the §2.5 fusion: does compiled canvas + reachability improve per-NPC behavior modellessly over un-unified constituents? If YES → re-evaluate for Super-GOAT; if NO → stays GOAT, compiler ships on structural merits). **RESOLVED 2026-07-09 — see §7 PoC Addendum below.** Result: could not isolate a canvas-attributable gain from the tuned classifier; stays GOAT. The correct re-PoC path is routing through `FaithfulnessProbe` (Plan 278, already wired in riir-poc via `jlens_concept_readout_goat.rs`) — see §7.

**riir-train follow-up (noted, not blocked):** train a small DiT within a declared NPC-cognitive-stack topology; measure parameter efficiency vs flat baseline at our scale. This is the genuine training dependency for the empirical headline; it lives in riir-train.

---

## 7. PoC Addendum (Issue 043, resolved 2026-07-09)

**Bench:** `riir-ai/crates/riir-poc/benches/canvas_npc_cognitive_stack_modelless.rs`
**Record:** `riir-ai/.benchmarks/043_canvas_npc_cognitive_stack_modelless.md`

**Result: COULD NOT ISOLATE A CANVAS-ATTRIBUTABLE GAIN FROM THE TUNED CLASSIFIER. Stays GOAT.**

The canvas `can_reach` gate enforces the declared perception→affect→action
topology correctly (T0 passes: `can_reach(perception,action,1)=false`,
`can_reach(perception,action,2)=true`). But the behavioral metrics do **not**
attributably support a canvas-attributable gain, for three reasons:

1. **Designer-tuned affect classifier (the load-bearing confound).** The affect
   layer that distinguishes real vs fake threats was hand-tuned to correctly
   classify the toy domain's threats (bench honesty note #3). The canvas gate
   FORCES action through this perfect discriminator; the leaky shortcut uses raw
   perception which cannot discriminate. So the observed MinDFake difference
   (canvas 1.93 vs leaky 3.03–3.46) and the MinDReal difference (canvas 3.87 vs
   leaky 3.28–3.40) are attributable to the **tuned discriminator**, not to the
   canvas structure. The canvas contributed the *forcing* (the `can_reach` gate),
   not the *discrimination*. Strip the hand-tuning and the gain likely vanishes.
2. **Noisy flee-rate metric.** The floor variant (no flee response at all)
   reports the HIGHEST TrueFlee rate (37.8%) — pure displacement-alignment noise
   (bench honesty note #1). A metric that reports the no-response baseline as the
   top flee-rate cannot ground the T2 "TrueFlee -0.7pp PASS" gate; that gate is
   measuring noise.
3. **Canvas has lower TrueFlee than Leaky** (22.6% vs 23.3–24.6%). By the rate
   metric, canvas flees real threats *slightly less* often. The bench leans on
   MinDReal to claim better avoidance, but lower flee-rate + higher min-distance
   is consistent with the canvas NPC simply not entering the flee trigger band
   (positioning), not with superior fleeing.

A more honest framing of the PoC: *the canvas gate's forcing function is only
beneficial when a downstream discriminator already exists; the PoC could not
isolate a canvas-attributable behavioral gain from the hand-tuned affect
classifier, so it does not answer whether canvas structure alone produces a
behavioral gain.*

**Verdict: Research 398 stays GOAT (not Super-GOAT).** The verdict rests on
structural/correctness merits (reachability-by-construction, Plan 419 G1–G6
PASS), **not** on this PoC. The PoC does not award Super-GOAT, and its inability
to isolate a canvas-attributable gain strengthens (not weakens) the
"not Super-GOAT" conclusion. `canvas_schema` stays opt-in pending a production
consumer.

### The correct re-PoC path: route through `FaithfulnessProbe` (we ship the tool)

The ad-hoc displacement-alignment flee metric was the **wrong tool**. We ship
the right one: `FaithfulnessProbe` (Plan 278,
`crates/katgpt-core/src/faithfulness/probe.rs`) — a **causal-intervention**
probe, not a correlational metric. It runs a 5-variant intervention suite
(`Empty`, `Shuffle`, `Corrupt`, `Irrelevant`, `Filler`) on an injected memory
segment and returns a `FaithfulnessProfile` whose `is_faithfully_used(t)`
verdict requires a small `empty_delta` (graceful absence) plus large deltas on
the meaningful interventions.

**Infrastructure is already wired.** `riir-poc` already depends on
`faithfulness_probe` (`crates/riir-poc/Cargo.toml` line 23) and already has a
precedent bench using it — `riir-ai/crates/riir-poc/benches/jlens_concept_readout_goat.rs` imports
`DefaultFaithfulnessProbe` + `FaithfulnessProbe::faithfulness_profile` and runs
the verdict at a 0.5 threshold. The canvas PoC lives in the same crate and could
have followed this exact pattern; it reinvented a noisy correlational metric
instead.

**Mapping onto the canvas question:**
- `ConsumerContext` = the NPC action function (canvas variant reads affect only;
  leaky variant reads affect OR perception via the shortcut).
- `Memory` = the affect signal.
- `Empty` intervention on affect: canvas variant loses its sole input
  (`can_reach` blocks perception→action at horizon 1) → large `empty_delta` →
  proves the gate is structurally load-bearing. Leaky variant still has the
  shortcut → small `empty_delta` → proves redundancy.
- `Irrelevant` intervention on affect: canvas variant degrades (depends on
  affect quality); leaky variant may not (perception compensates). Isolates how
  much each variant relies on affect quality.

**What this settles and what it does not:**

| Question | `FaithfulnessProbe` settles it? |
|---|---|
| Is the canvas gate causally load-bearing (affect faithfully used)? | Yes — cleanly, via causal intervention |
| Is the behavioral delta canvas-attributable (not classifier-artifact)? | Mostly — the `Empty` delta is canvas-attributable by construction; the `Irrelevant` delta exposes classifier-quality dependence |
| Does canvas produce *better* behavior? | No — still needs a validated affect signal (non-designer-tuned); the probe characterizes the dependency but does not prove quality |

So a re-PoC routing through `DefaultFaithfulnessProbe` would cleanly settle the
structural-attribution half (the part the ad-hoc PoC botched) and sharply
characterize the classifier-quality dependence (the `Irrelevant` delta). The
behavioral-quality claim still ultimately needs a non-designer-tuned
discriminator, but the probe is the correct first instrument — it replaces
guesswork with causal isolation.

---

## 8. Cross-repo context — what is default-on, what has consumers, where the showcase lives

> **Why this section exists:** the canvas story's value hinges on whether its *constituents* are already
> shipped/default-on/showcased. Reconstructing that state required grepping across 5 repos and **missed
> things** (the `faithfulness_probe` wiring in riir-poc; the Plan 426 cookbook bench). This section
> captures the verified state inline so the next reader does not re-grep and re-miss. All facts below
> were grep-verified on 2026-07-10.

### The constituents canvas "unifies" — and their real status

| Constituent | Primitive | Feature / status | Runtime consumer (non-test/bench)? | Showcase |
|---|---|---|---|---|
| Region-conditioned steering | `RegionSubspaceField` (Plan 416) | **DEFAULT-ON** in katgpt-core (`region_subspace_steering`) | **YES** — `riir-ai/crates/riir-engine/src/latent_functor/region_subspace_bridge.rs` (`apply_to_zones_batch`) | Plan 426 S3 (measured) |
| Additive 1D steering | `LatentSteeringVector` (Plan 309) | **DEFAULT-ON** | **YES** — `latent_functor` bridges | Plan 426 S1/S2/S4/S7 |
| Slerp / norm-preserving | Plan 405 | opt-in→default bridge | `spherical_steering_bridge` | Plan 426 S2 (0.0 drift) |
| AC-Prefix mask builder | `AcPrefixMask` (Plan 313) | **DEFAULT-ON** (canonical G1 lesson) | riir-engine attention path | `.benchmarks/313_*` |
| DEC operators | `exterior_derivative`/`codifferential`/`hodge_decompose` | **DEFAULT-ON** (manifold substrate) | DEC consumers in riir-engine | Plan 426 S5/S6 + `.benchmarks/313_*` |
| Looped attention | `LoopMode::TrainingFree` (Plan 136) | distilled | — | — |
| Freeze/thaw exchange | `MerkleFrozenEnvelope` / `CommittedFieldBlend` | **DEFAULT-ON** | riir-neuron-db | — |

### canvas_schema's status (the compiler this research distills)

- **Feature flag `canvas_schema`: OPT-IN**, NOT default-on. Plan 419 Phase 1–6 done, G1–G6 PASS.
  Promotion explicitly **deferred** (Plan 419 Phase 5 decision + §7 here).
- **Runtime consumers: ZERO.** Only the riir-poc bench `canvas_npc_cognitive_stack_modelless.rs`
  (the inconclusive Issue 043 PoC). AC-Prefix/VortexFlow wiring is a deferred follow-up (Plan 419 T6.3).
- Grep-verified: no call to `compile_schema`/`CanvasSchema`/`can_reach`/`reachability_horizon` exists
  in riir-ai / riir-chain / riir-neuron-db outside tests/benches.

### The flagship showcase (what to point at for "modelless works, here are the numbers")

**`riir-ai/.benchmarks/426_steering_geometry_cookbook.md`** — 7 canonical MMORPG scenarios, head-to-head,
measured, re-verified 2026-07-10 (numbers reproduce). Quality metrics exact (S1 bit-identical 0.392628,
S2 norm drift 29.2×, S3 intrinsic dim 2, S6 mass 0.0161); latencies within ±20% variance, ratios stable.
Two scenarios (S4, S5) honestly contradict the tentative verdict (G-honest flags). Run:

```bash
CARGO_TARGET_DIR=/tmp/cookbook_426 cargo bench -p riir-poc --bench steering_geometry_cookbook -- --nocapture
```

Other showcase benches in the same crate (`riir-poc/benches/`): `subspace_swap_goat.rs` (3-way quality
head-to-head), `jlens_concept_readout_goat.rs` (`DefaultFaithfulnessProbe` precedent — see below).

### The right tool for canvas's behavioral question (don't grep for this)

`FaithfulnessProbe` (Plan 278, `crates/katgpt-core/src/faithfulness/probe.rs`) is the causal-intervention probe
that should settle the §7 question. **It is already wired into riir-poc** (`Cargo.toml` line 23 feature
dep) with a precedent bench (`jlens_concept_readout_goat.rs` imports `DefaultFaithfulnessProbe` +
`faithfulness_profile`, verdict at 0.5 threshold). The canvas PoC lives in the same crate and could have
followed that pattern; it reinvented a noisy correlational flee metric instead. See §7 for the mapping.

### Bottom line (captured here so it doesn't need re-deriving)

The canvas compiler's constituents are **already DEFAULT-ON with runtime consumers and a measured
7-scenario showcase**. canvas_schema is the opt-in unification layer with **zero consumers** and an
**inconclusive behavioral PoC**. It parks as a correctness-class primitive (reachability-by-construction,
like DEC `d∘d=0`) until a real runtime path wants declared-topology-as-causal-graph. The steering + DEC
stack (Plan 426) is the story to show off, not the canvas fusion.

> **Update 2026-07-10 (R403):** the "needs training" conclusion above is a paper-framing
> artifact (it compiled onto a DiT's attention positions). Research 403 dissolves it by
> distilling the canvas onto the **latent-functor substrate** (gated functor graph instead of
> attention mask) — same reachability guarantee, no transformer backbone, no training. The gated
> graph is R398's first potential runtime consumer. See
> [`R403`](403_Topology_Constrained_Latent_Functor_Composition.md) and
> `Issue 122` (the FaithfulnessProbe-attributed PoC).


---

## TL;DR

Canvas engineering declares a typed schema and compiles it to attention masks + loss weights. The **modelless value** is the compiler + reachability-as-exact-marginal-independence + transfer_distance + schema-mediated latent exchange. The **headline empirical value** (1.73× parameter efficiency, cortical R²=0.825) is training-dependent. Most constituent primitives ship (`region_subspace_bridge`, AC-Prefix, StillPerceiver, `LoopMode::TrainingFree`). **Verdict: GOAT** — the unified `CanvasSchema` compiler is novel and modelless, connects ≥4 pillars, and gives a provable correctness property (reachability by construction); but it is a unifying type-system abstraction, not a new empirical capability class at the modelless level, and the behavioral gain requires riir-train. A fusion PoC (record `riir-ai/.benchmarks/043_*`, §7) could not isolate a canvas-attributable gain from the tuned classifier; the ad-hoc flee metric was the wrong tool and a re-PoC should route through `FaithfulnessProbe` (Plan 278, already wired in riir-poc). Super-GOAT re-evaluation stays open pending that causal-intervention re-PoC. Plan 419 ships the open compiler primitive.
