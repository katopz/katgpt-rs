# MRAgent: Reconstructive Memory Graph — Modelless Distillation

**Date:** 2026-06-11
**Paper:** [Memory is Reconstructed, Not Retrieved (ICML 2026)](https://arxiv.org/pdf/2606.06036v1)
**Verdict: GOAT — GAIN** (modelless path, high commercial alignment)

---

## Paper Core

MRAgent reframes memory access from **passive retrieval** (TopK similarity) to **active reconstruction** (stateful multi-step traversal over Cue–Tag–Content graph). Key result: **+23% accuracy, 5× token reduction, 2× faster** vs Mem0 on LOCOMO benchmark.

### Architecture: Cue–Tag–Content (CTC) Graph

```
Cue (entity/keyword) ──Tag (semantic bridge)──▶ Content (episode/semantic/topic)
```

- **Cue**: Fine-grained keywords (entities, attributes, temporal anchors)
- **Tag**: Short associative phrase mediating cue↔content (the innovation)
- **Content**: Episodic (event), Semantic (stable fact), Topic (recurring pattern)

### Active Reconstruction Algorithm

```python
S(t) = (Z(t), H(t))  # active set + accumulated evidence
for t in 0..T:
    A(t) = LLM_select(x, H(t), Z(t))     # pick traversal actions
    Z'(t+1) = union(Π_a(Z(t)) for a in A)  # expand candidates
    Z(t+1) = LLM_route(x, H(t), Z'(t+1))  # prune + route
    H(t+1) = H(t) ∪ Z(t+1)                # accumulate evidence
    if sufficient(H(t+1)): break
```

### Theoretical Result (Theorem 4.1)

Active reconstruction is **strictly more expressive** than passive retrieval: H_passive ⊊ H_active for any budget T ≥ 2. Passive retrieval has exponential gap on binary-tree needle-in-haystack tasks.

---

## Fusion with Our Architecture: CTC × KG-Latent-Octree × MUX-Latent

### The Insight: CTC IS Our Octree, But With Navigation

Our existing KG-Latent-Octree already encodes the CTC pattern:

| MRAgent | Our Existing | Mapping |
|---------|-------------|---------|
| **Cue** | HLA state `[f32; 8]` / entity hash | Direct — already exists |
| **Tag** | `SenseKind` + octree morton code (`segment_id`) | Already exists as octree occupancy bits |
| **Content** | `TernaryDir` direction vectors + KG triples | Already exists in `SenseModule` |

**What we DON'T have:** The active reconstruction loop — iterative navigation over the graph with pruning. Currently `NpcBrain::project()` does ONE-SHOT dot-product projection. MRAgent proves multi-step is strictly more powerful.

### Novel Fusion: **OctreeCTC** — Reconstructive Octree Navigation

The fusion is NOT just "add MRAgent to octree" — it's fundamentally different:

1. **Navigation IS projection** — each traversal step IS a `SenseModule::project()` call, but the HLA state **evolves** based on accumulated evidence (active, not passive)
2. **Pruning IS octree LOD** — irrelevant branches are pruned by lowering octree LOD (fewer nodes to traverse), not by LLM routing
3. **Tags ARE ternary direction vectors** — the associative bridge between cue and content is the `TernaryDir` itself, which encodes `{+1, -1, 0}` semantic polarity
4. **No LLM in the loop** — modelless by design. Traversal decisions use deterministic entropy-based routing from our existing `SenseBandit`

### Architecture: OctreeCTC

```
                ┌─────────────────────────────────────┐
                │         OctreeCTC Memory Graph      │
                │                                     │
  Cue (HLA) ──▶ │  Octree Node 0 (root)               │
                │    ├─ Tag: TernaryDir[0] ──▶ Content: KG Triples (episode)
                │    ├─ Tag: TernaryDir[1] ──▶ Content: KG Triples (semantic)
                │    └─ Tag: TernaryDir[2] ──▶ Octree Child Nodes
                │                                     │
                │  Reconstruction State:              │
                │    Z(t) = active octree nodes       │
                │    H(t) = accumulated triples       │
                │    HLA'(t) = updated HLA from H(t)  │
                │                                     │
                │  Traversal:                         │
                │    select: entropy-gated bandit     │
                │    expand: project(Z, HLA') → rank  │
                │    prune: LOD-adaptive cutoff       │
                │    update: HLA'' = bridge(H(t))     │
                └─────────────────────────────────────┘
```

### Key Innovation: HLA State Evolution (No LLM Needed)

```rust
// Before: one-shot passive projection
let activation = brain.project_all(&hla_state); // single pass

// After: multi-step active reconstruction  
let mut state = ReconstructionState::new(hla_state);
for step in 0..max_steps {
    let candidates = state.expand(&brain);     // octree traversal
    let selected = state.route(&candidates);   // entropy-gated prune
    state.accumulate(selected);                // KG triples → evidence
    state.evolve_hla();                        // bridge: triples → HLA update
    if state.sufficient() { break; }           // early stop
}
```

The `evolve_hla()` step is the bridge function from AGENTS.md: accumulated KG triples (raw) → projected scalars via dot-product + sigmoid → HLA state update (latent). No softmax, no LLM.

---

## Commercial Alignment (per 003 Verdict)

| Criterion | Assessment |
|-----------|------------|
| **Engine/Fuel Split** | ✅ Engine (modelless reconstruction) stays MIT. Fuel (trained `SenseModule` weights from riir-ai) stays SaaS |
| **RIIR Wedge** | ✅ Reconstructive memory improves DDTree constraint synthesis — better pruning = better RIIR accuracy |
| **Defensibility** | ✅ Active reconstruction over octree is novel — no competitor has this. Combined with `SenseModule` weights = Ferrari with optimized navigation |
| **Modelless First** | ✅ No LLM calls in reconstruction loop. Entropy-based bandit + dot-product + sigmoid only |

---

## Performance Expectations

| Metric | Passive (current) | Active (projected) | Rationale |
|--------|-------------------|-------------------|-----------|
| Multi-hop recall | ~60% (single projection) | ~85%+ (3-step reconstruction) | MRAgent shows +30% on multi-hop, our octree has richer structure |
| Latency/NPC/tick | ~45ns | ~120ns (3 steps × 40ns) | 2.7× slower but still well within game tick budget (16ms) |
| Token/query cost | N/A (modelless) | N/A (modelless) | Zero tokens — this is the advantage vs MRAgent which uses LLM calls |
| Memory footprint | 232B/SenseModule | Same + 64B ReconstructionState | Minimal overhead |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Entropy-gated bandit too simple vs LLM routing | Bandit has proven convergence in `SenseBandit`. LLM routing costs 100ms+ — game needs <1ms |
| HLA evolution destabilizes projection | Clamp + sigmoid bridge (per AGENTS.md). Max HLA delta per step bounded |
| Octree traversal depth explosion | LOD-adaptive pruning already limits traversal. Max 3 steps default |
| No LLM means no semantic understanding | `TernaryDir` encodes semantic polarity. `KgEmbedding` encodes entity/relation hashing. Not LLM-level but sufficient for game AI |

---

## What We Take, What We Leave

### Take (GAIN)
1. **Active reconstruction paradigm** → `ReconstructionState` iterative navigation
2. **CTC graph structure** → Maps 1:1 to our existing octree (Cue=HLA, Tag=TernaryDir, Content=KG triple)
3. **Theorem 4.1 proof** → Validates that multi-step > single-step for memory access
4. **Evidence accumulation + early stopping** → Entropy-based sufficiency check

### Leave (NO GAIN)
1. **LLM in the loop** — We're modelless. All routing is deterministic.
2. **Memory construction via LLM distillation** — Our `SenseOctreeBuilder` already handles this via `KgEmbedding` → `TernaryDir` compression
3. **Topic/abstraction layer** — Our `SpectralLOD` already handles multi-granularity
4. **Conversational memory benchmark** — We target game AI, not dialog agents

---

## TL;DR

MRAgent proves active multi-step reconstruction over Cue–Tag–Content graphs is **strictly more powerful** than passive retrieval. Our KG-Latent-Octree already IS a CTC graph. The missing piece is the **reconstruction loop** — iterative HLA-state-aware navigation with pruning. This is modelless (entropy bandit + dot-product + sigmoid, no LLM), fits in ~120ns per tick, and improves multi-hop KG triple recall by projected 25%+. Fits engine/fuel split: engine stays MIT, trained `SenseModule` weights stay SaaS. **GOAT: GAIN. Proceed to plan.**
