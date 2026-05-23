# Research 58: MeMo — Memory as a Model

> **Paper:** [MeMo: Memory as a Model](https://arxiv.org/abs/2605.15156) — Quek, Lee, Leong, Verma et al. (NUS/MIT/SMART), May 2026
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 06 (Raven RSM), 21 (G-Zero), 24 (δ-Mem), 37 (REAP Model-Based/Modelless), 51 (Deep Manifold)
> **Related Plans:** 092 (Freeze/Thaw), 094 (MeMo Reflections + TIES Merging)
> **Verdict: CONCEPTUAL ALIGNMENT — Validates Raven RSM O(1) retrieval. Two extractable techniques: Reflection QA pipeline (data synthesis) + TIES model merging (continual integration). No architectural changes needed.**

---

## TL;DR

MeMo trains a small dedicated MEMORY model on corpus-derived "reflection QA pairs", then an EXECUTIVE model queries it via a 3-stage multi-turn protocol at inference. O(1) retrieval cost, noise-robust, black-box compatible. Our system already captures the core ideas via Raven RSM (O(1) slots), G-Zero phases (multi-turn protocol), and WASM validator SDK (black-box). Two concrete techniques extractable: (1) 5-step reflection QA data synthesis pipeline, (2) TIES model merging at ρ=0.3 for continual knowledge integration.

---

## Paper Architecture

### Two Components

1. **MEMORY model Mφ** — Small model (1.5B–14B params) trained on reflection QA pairs derived from target corpus. Answers queries from parametric knowledge alone (no document access at inference).
2. **EXECUTIVE model Mθ** — Frozen LLM (32B+) that queries MEMORY model via structured protocol. Treats MEMORY model as black-box knowledge oracle.

### 5-Step Data Synthesis Pipeline (Algorithm 1)

| Step | Name | What | Output |
|------|------|------|--------|
| 1 | Fact Extraction | Direct + indirect extraction per chunk | Q_dir, Q_indir |
| 2 | Consolidation | Merge related QA pairs into multi-fact questions | Q_mrg |
| 3 | Verification | Rewrite/discarded non-self-contained pairs | Q_ver |
| 4 | Entity Surfacing | Generate entity-from-description QA pairs (reversal curse mitigation) | Q_ent |
| 5 | Cross-Document Synthesis | Converging clues + parallel properties across document groups | Q_cross |

**Final dataset:** Q_final = Q_ver ∪ Q_ent ∪ Q_cross

**Critical finding (ablation):** Step 5 is the most important — removal causes accuracy collapse (6.37% from 24.00% on NarrativeQA).

### 3-Stage Inference Protocol

| Stage | Budget | What |
|-------|--------|------|
| 1: Grounding | 1 turn | Decompose query into atomic sub-questions → MEMORY model answers each independently |
| 2: Entity Identification | 7 turns | Iteratively narrow candidate entities via targeted follow-ups |
| 3: Answer Synthesis | 8 turns | Seek supporting facts for identified entity → synthesize final answer |

### Model Merging for Continual Integration

- Train separate MEMORY models per corpus → compute task vectors τ_i = φ_i − φ_0
- Merge: φ_merged = φ_0 + Merge({τ_i})
- **TIES at ρ=0.3** is best: trim top-30% magnitude entries, sign-conflict resolution, disjoint merge
- **33% compute saving** at K=2; Θ(K) vs Θ(K²) for full retrain
- Accuracy gap: -11.0pp (Qwen) to -19.1pp (Gemini) vs full retrain, but still beats all retrieval baselines

### Key Properties

| Property | MeMo | RAG | Fine-tuning | Latent Memory |
|----------|------|-----|-------------|---------------|
| Frozen base LLM | ✅ | ✅ | ❌ | ✅ |
| No retrieval index | ✅ | ❌ | ✅ | ✅ |
| Black-box compatible | ✅ | ✅ | ❌ | ❌ |
| No catastrophic forgetting | ✅ | ✅ | ❌ | ✅ |
| Constant-size memory | ✅ | ❌ | ✅ | ✅ |
| Cross-LLM transferable | ✅ | ✅ | ❌ | ❌ |

---

## Mapping to Our Stack

### What We Already Have (Conceptual Alignment)

| MeMo Concept | Our Implementation | Match |
|--------------|-------------------|-------|
| O(1) retrieval (fixed-size memory model) | Raven RSM: O(1) slot-based attention | ✅ Exact |
| Black-box compatibility (no weights/logits access) | WASM Validator SDK: sandboxed `.wasm` validators | ✅ Exact |
| Model-based/modelless spectrum | `ConstraintPruner` → `ScreeningPruner` → `BanditPruner` → `SpeculativeVerifier` | ✅ Captured |
| Multi-turn protocol (3 stages) | G-Zero phases (modelless → model-based) + multi-episode bandit | ✅ Captured |
| Frozen base model + trained adapter | LoRA fine-tuning + `HotSwapPruner` zero-downtime swap | ✅ Captured |
| Continual knowledge integration | Freeze/Thaw pipeline (Plan 092) + `repr(C)` bandit state | ✅ Partial |
| Cross-document reasoning | DDTree branch exploration with screening pruner | ✅ Partial |
| Entity surfacing (reversal curse) | Not yet implemented | ❌ Gap |

### What We DON'T Have (Technique Extraction)

#### 1. Reflection QA Pipeline

MeMo's 5-step synthesis is a concrete data generation technique. Our closest equivalent is:
- GFlowNet distillation (Plan 052) — generates trajectory-based training data
- ROPD rubric (Plan 071) — multi-criterion reward vectors
- Game replay training data (Plan 039) — raw game state → action pairs

None of these do **consolidation + verification + entity surfacing** on raw corpora. This is a new technique.

**Potential use:** Generate reflection QA pairs from game replay corpora (Bomber, Go, FFT). Instead of raw (state, action) pairs, create compositional questions that require integrating multiple game situations.

#### 2. TIES Model Merging

Concrete merging technique: trim → sign elect → disjoint merge at ρ=0.3 density.

Our freeze/thaw pipeline (Plan 092) stores `repr(C)` bandit state, but doesn't do model merging. If we ever have multiple domain-specific LoRA adapters, TIES merging is the way to combine them.

**Potential use:** Merge per-game LoRA adapters (Bomber LoRA + Go LoRA) into a single game-domain adapter.

---

## Experimental Results (From Paper)

### Main Results (Table 2)

| Method | BrowseComp-Plus | NarrativeQA | MuSiQue |
|--------|----------------|-------------|---------|
| BM25 | 1.11 / 27.00 | 10.24 / 14.33 | 20.00 / 23.20 |
| NV-Embed-V2 | 50.67 / 57.00 | 20.59 / 26.62 | 37.47 / 46.60 |
| HippoRAG2 | 56.11 / 66.33 | 21.39 / 23.21 | 42.17 / 57.00 |
| **MeMo (14B)** | **54.22 / 66.67** | **26.85 / 53.58** | **48.30 / 60.20** |
| Perfect Retrieval (UB) | 79.67 / 88.33 | 51.42 / 60.41 | 62.83 / 73.00 |

(Accuracy %; Qwen2.5-32B / Gemini-3-Flash as Executive)

### Noise Robustness (Table 3)

| Method | BrowseComp-Plus (0N→1N) | MuSiQue (0N→1N) |
|--------|------------------------|------------------|
| NV-Embed-V2 | ↓ 6.22% | ↓ 4.83% |
| HippoRAG2 | ↓ 6.22% | ↓ 5.16% |
| **MeMo** | **↑ 0.55%** | **↓ 1.77%** |

MeMo is robust to retrieval noise — near-zero degradation vs 5-6% drops for retrieval methods.

### Model Merging (Table 6)

| Method | Compute (8×H100 GPU-h) | Qwen Acc | Gemini Acc |
|--------|------------------------|----------|------------|
| Full retrain | 72h | 26.85% | 53.58% |
| **Merge-TIES ρ=0.3** | **48h (-33%)** | 15.81% (-11.0pp) | 34.47% (-19.1pp) |

Merged model still beats all retrieval baselines (BM25 10.24%, NV-Embed 20.59%, HippoRAG2 21.39%).

---

## Distillations for Our Stack

### D1: Reflection QA Game Data (Modelless)

**What:** Apply MeMo's 5-step pipeline to game replay data. Instead of raw (state, action) pairs, generate compositional game-knowledge QA:

```
Step 1: Extract facts from game situations (direct: "Where should I place bomb?" + indirect: "Why did this position fail?")
Step 2: Consolidate related facts ("When trapped on three sides, bomb placement patterns")
Step 3: Verify self-containment (rewrite ambiguous references)
Step 4: Entity surfacing ("What entity has the pattern: moves toward center, avoids corners, places bombs on timer?")
Step 5: Cross-game synthesis ("In both Bomber and Go, corner strategies vs center strategies")
```

**Why modelless:** QA pairs become heuristic knowledge for `BanditPruner` + `AbsorbCompress`. No gradient update needed. The QA format is directly consumable by our `ScreeningPruner::relevance()`.

**Implementation:** `src/pruners/reflection.rs` with `fn synthesize_reflections(game_replay: &[GameState]) -> Vec<ReflectionQA>` behind feature gate `memo_reflections`.

### D2: TIES Model Merging (Model-Based)

**What:** Implement TIES merging for combining multiple domain-specific LoRA adapters.

```rust
fn ties_merge(base: &LoRAWeights, task_vectors: &[TaskVector], density: f32) -> LoRAWeights {
    // 1. Trim each τ_i to top-ρ% magnitude entries
    // 2. Elect sign at each coordinate via magnitude-weighted majority vote
    // 3. Disjoint merge: only keep entries that agree with elected sign
    // 4. Sum and add to base
}
```

**Why model-based:** Requires trained LoRA adapters (from `riir-gpu`). This is the Phase 2 path — when modelless plateaus, merge multiple domain adapters.

**Implementation:** In `riir-ai/crates/riir-gpu/src/merging.rs` behind feature gate `ties_merge`. Uses existing `export_lora` / `load_lora` infrastructure.

### D3: Entity Surfacing for Reversal Curse (Modelless)

**What:** Generate entity-from-description QA pairs to mitigate the reversal curse (model knows "A is B" but not "B is A").

**Why modelless:** In game domains, this means training the bandit to recognize patterns from descriptions, not just from state observations. E.g., bandit knows "corner strategy" → Q=0.8, but also "the strategy that avoids center engagement" → Q=0.8.

**Implementation:** Low priority — our bandits don't use natural language descriptions.

---

## What NOT To Do

1. **Don't build a separate MEMORY model service.** Our system operates at token-level (pruning, speculation) not at knowledge-base level. Raven RSM already gives O(1) slot-based retrieval.
2. **Don't implement the full multi-turn protocol as a new abstraction.** G-Zero's phase system already handles multi-episode learning with budgets.
3. **Don't use MeMo's exact data synthesis for general LLM knowledge.** We're focused on game domains and code transpilation, not open-domain QA.
4. **Don't replace Freeze/Thaw with model merging.** Our `repr(C)` binary persistence is simpler and more appropriate for bandit state. TIES merging is for LoRA adapters, not bandit arrays.

---

## Relationship to Existing Research

| Research | Overlap | Delta |
|----------|---------|-------|
| 06 (Raven RSM) | O(1) retrieval via fixed-size slots | MeMo validates this approach at corpus scale |
| 21 (G-Zero) | Multi-phase self-play, δ signal | MeMo's multi-turn protocol is analogous to G-Zero phases |
| 24 (δ-Mem) | Online memory updates | MeMo uses SFT training (not online), δ-Mem uses delta rule |
| 37 (REAP) | Model-based/modelless spectrum | MeMo is model-based (trained memory), our pruners are modelless |
| 51 (Deep Manifold) | Fixed-point boundary conditions | MeMo's reflections = "boundary conditions" on corpus knowledge |
| 52 (GFlowNet) | Trajectory-based data generation | MeMo uses LLM-synthesized QA, GFlowNet uses flow-based sampling |

---

## References

- Paper: https://arxiv.org/abs/2605.15156
- Code: Supplementary materials (not yet public)
- Related: HippoRAG2, Cartridges, Memory Decoder, AutoCompressor