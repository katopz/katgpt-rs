# Research 111: Emergent Analogical Reasoning in Transformers

**Paper:** arXiv:2602.01992 (ICML 2026)
**Authors:** Minegishi, Feng, Furuta, Kojima, Iwasawa, Matsuo (UTokyo / Google DeepMind)
**Date:** 2026-05-26
**Verdict:** ⭐ **ADOPT — high-value mechanistic insight for inference-time geometry + LoRA distillation**

---

## TL;DR

Analogical reasoning in Transformers decomposes into two mechanistic components:
1. **Structural alignment** — embeddings of entities across categories geometrically align (measured by Dirichlet Energy decrease)
2. **Functor application** — attention writes source entity info into functor position, residual connection adds it as `e_target ≈ e_source + f` (vector arithmetic)

This emerges in a three-stage training dynamic: memorization → composition → analogy. Analogy is **qualitatively different** from composition — it's sensitive to weight decay, batch size, relation diversity, and does **not** improve monotonically with model size.

---

## Key Findings

### 1. Three-Stage Training Dynamics
| Stage | What | Sensitivity |
|-------|------|-------------|
| Memorization | Fits in-distribution facts | Robust |
| Composition | Chains 2-hop relations OOD | Robust, scales with width/depth |
| Analogy | Maps across categories via functor | Highly fragile — data, optimizer, scale sensitive |

**Implication for us:** Our modelless distillation pipelines (GFlowNet, ROPD, SDAR) are doing composition/memorization. True cross-domain transfer (analogy) requires specific conditions that we aren't engineering for.

### 2. Dirichlet Energy as Structural Alignment Metric
$$E(\mathbf{E}) = \sum_{e_i, e_j \in E} A_{ij} \|h_{e_i} - h_{e_j}\|^2$$

Where $A_{ij} = 1$ if entities $i,j$ are related via functor. **Lower = more aligned.**

- In toy models: energy drops along **training-step axis**
- In pretrained LLMs (Gemma-2 2B/9B, LLaMA): energy drops along **layer axis** (in-context learning)
- Energy drop **precedes** analogical accuracy improvement

**Implication for us:** We can compute Dirichlet Energy over our KV cache embeddings or LoRA weight subspace to measure cross-domain alignment quality. This is a direct diagnostic for whether our LoRA domain adapters are learning structural correspondences or just memorizing domain-specific patterns.

### 3. Functor as Vector Addition via Residual Connection
```
e_target ≈ e_source + f
```
Where `f` is the functor token representation. The attention mechanism:
1. Functor token `f` attends to source entity `e_s`
2. Attention writes `e_s` info into `f`'s representation
3. Residual connection: `h_f' = h_f + Attn(h_f, ...)` ≈ `h_f + h_{e_s}`
4. Unembedding: target entity ≈ `h_{e_s} + h_f`

**Implication for us:** This validates the linear representation hypothesis for cross-domain mapping. Our `DomainLatent` injection and Fourier-MLA LoRA already exploit linear subspaces — analogy suggests we should explicitly learn **functor directions** between game domains.

### 4. Critical Data/Optimization Conditions for Analogy
| Factor | Analogy Emerges? |
|--------|-----------------|
| Too few relations (`|R| < 1000`) | ❌ Fails |
| Weight decay = 0 | Slow/fragile |
| Weight decay = 0.01–0.1 | ✅ Best |
| Weight decay = 1.0 | ❌ Fails |
| Batch size ↑ | Faster emergence |
| Model too small (`d=64`) | ❌ Fails |
| Model too large (`d=512`) | ⚠️ Degraded (memorization wins) |
| Graph too sparse | ❌ Analogy degrades fast |

**Implication for us:** Our LoRA training pipeline (riir-ai) uses SGD-like optimization. We should check: is weight decay configured in the "analogy sweet spot" (0.01–0.1)? Is relation diversity sufficient in game domains?

### 5. LLM Layer-Axis Confirmation
In Gemma-2 and LLaMA:
- Dirichlet Energy drops in **later layers** (layers 15–25 in Gemma-2 2B)
- Target probability surges at the same layers
- More entities → requires deeper layers for alignment

**Implication for us:** Our Percepta (transformer-vm) and GDN2 already do multi-layer processing. The layer where alignment occurs could be a natural early-exit signal for analogical vs. non-analogical queries.

---

## Distillation to Our Architecture

### katgpt-rs (Open Engine)

| Our Component | Paper Concept | Action |
|---------------|--------------|--------|
| `spectralquant` KV cache | Dirichlet Energy over embeddings | Add `dirichlet_energy()` diagnostic to measure cross-position structural alignment |
| `ScreeningPruner` | Graded relevance | Analogy suggests relevance should encode **relational role similarity**, not just token probability |
| `ConfiguratorContext` (SR²AM) | Data/optimizer sensitivity | Add analogy-readiness signal: "is the current training regime likely to produce structural alignment?" |
| `GameState` trait | Category structure | Games ARE categories — entities = positions, relations = valid moves. Analogical transfer = "same tactic works across game domains" |
| `data_probe` | Markov chain diagnostics | Already have Dirichlet sampling in `markov.rs` — extend to Dirichlet Energy computation over embedding adjacency |
| Early exit (Plan 026) | Layer-axis alignment | Dirichlet Energy at layer L could be an early-exit signal for "this query is analogical" |

### riir-ai (Private/Game Domain)

| Our Component | Paper Concept | Action |
|---------------|--------------|--------|
| **Fourier MCTS** (Research 001) | Relational structure encoding | Games have rich relational structure (position → position via move). Fourier encoding already captures periodic patterns — this IS structural alignment. **Fourier is riir-ai private IP** |
| LoRA training (wgpu) | Weight decay sweet spot | Verify wd ∈ [0.01, 0.1] for analogy emergence. Add Dirichlet Energy probe during LoRA training to measure cross-domain alignment |
| SHINE hypernet | Context→LoRA mapping | SHINE is literally a functor: maps context to LoRA weights. The paper validates that functors emerge as linear vector additions |
| NPC Dialog | Cross-domain analogy | NPC quest packs could be "categories" — analogous quests across different game domains |
| Self-play episodes | Training dynamics | The 3-stage dynamics (memorize→compose→analogy) should be observable in our self-play training curves |

---

## Feature Gate Strategy

| Feature | Gate | Domain | Why |
|---------|------|--------|-----|
| `dirichlet_energy` diagnostic | katgpt-rs (open) | KV cache analysis | Generic embedding diagnostic — not game-specific |
| Functor direction probe | katgpt-rs (open) | Inference-time | General mechanistic interpretability |
| LoRA analogy training config | riir-ai (private) | LoRA training | Game-specific weight decay and relation diversity tuning |
| Cross-game analogy detection | riir-ai (private) | Game AI | Super-GOAT: detect cross-domain tactical transfer via functor directions. Details → riir-ai/.research/123 |
| Structural alignment early exit | katgpt-rs (open) | Inference | General early-exit signal |

---

## GOAT Pillar Mapping

From `27_mmo_goat_pillars_decision_matrix.md`:

| Pillar | Analogy Connection |
|--------|-------------------|
| **Pillar 1: Fourier Spatial AI** | Direct hit. Fourier encoding IS relational structure encoding. Dirichlet Energy measures whether position embeddings align across game maps. This validates Fourier as the right approach — it creates the structural alignment the paper identifies as prerequisite for analogy. **Note:** Fourier is riir-ai private IP (Research 001) — the Dirichlet Energy diagnostic is open, but the Fourier-specific alignment proof stays private. |
| **Pillar 2: WASM Validators** | Indirect. Validators encode game rules (relations). The "relational diversity" finding (need `|R| ≥ 1000`) suggests validators should be **diverse** — not just "is this valid?" but "what relational role does this entity play?" |
| **Pillar 3: NPC Dialog** | Direct hit. Dialog FSM defines categories of NPC behavior. Analogical reasoning = "this NPC's quest structure matches that NPC's" → enables quest generation by structural analogy. |
| **Pillar 4: Frame-Sampling** | Indirect. Frame decimation is about computational efficiency, not relational structure. |

**Key insight:** The paper's finding that analogy requires **dense relational graphs** validates our decision to make games the domain — games have inherently dense, well-defined relational structures (every position has valid moves to other positions). Natural language analogy is hard because graphs are sparse. Game analogy should be easier.

---

## What NOT to Do

1. **Don't implement the full synthetic task.** It's a toy. The value is the Dirichlet Energy metric and the vector-addition insight, not the task itself.
2. **Don't add analogy as a "reasoning mode".** The paper shows analogy emerges from training dynamics, not from architecture changes. We can't bolt it on.
3. **Don't make this a LoRA bet.** The paper shows analogy is fragile — if LoRA is underperforming (our Risk Assessment from Pillars doc), analogy won't save it.
4. **Don't expose cross-game analogy as a product feature.** If it works, it's a secret advantage (Super-GOAT). Keep it in riir-ai.

---

## Research Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ First mechanistic decomposition of analogy in Transformers |
| Rigor | ⭐⭐⭐⭐⭐ Toy → LLM validation, multiple model families, E-KAR benchmark |
| Relevance to us | ⭐⭐⭐⭐ Directly validates Fourier encoding, explains LoRA training dynamics |
| Actionability | ⭐⭐⭐⭐ Dirichlet Energy is trivial to implement, functor direction is testable |
| Risk | ⭐⭐ Low — even if analogy doesn't emerge in our training, the diagnostic is useful |

**Bottom line:** Adopt Dirichlet Energy as a generic diagnostic in katgpt-rs (open). Investigate functor directions and Fourier-specific alignment proofs in riir-ai (private, Research 001). The paper validates that games are a natural domain for analogical reasoning due to dense relational structure — this strengthens Pillar 1 (Fourier Spatial AI), but the Fourier-specific work stays private.

---

## Implementation Status (2026-06-15)

Both open-engine primitives identified above are now built in `crates/katgpt-core/src/dirichlet.rs` (feature `dirichlet_energy`):

1. **Dirichlet Energy diagnostic** — Plan 149 ✅ SHIPPED (`dirichlet_energy`, `kv_cache_dirichlet_energy`, adjacency helpers). Public-safe: pure measurement, not a latent operation.
2. **Functor direction probe** (`e_target ≈ e_source + f`) — Plan 273 📐 **SPEC (audited, corrected).** API signatures + GOAT G1–G5 spec are valid. Target repo corrected from katgpt-rs to riir-ai per [`003`](../.research/003_Commercial_Open_Source_Strategy_Verdict.md) (latent-operation internals are private). Implemented by [riir-ai Plan 303](../../riir-ai/.plans/303_latent_functor_runtime_npc_relational_learning.md). Game-specific runtime (NPC functor tables, coherence re-estimation, cross-game transfer) → **riir-ai Research 123** (Latent Functor Runtime Guide).

## Related Internal Research

| Research | Connection |
|----------|-----------|
| Research 051 (Deep Manifold) | Fixed-point boundary conditions → geometric alignment is a related concept |
| Research 058 (GRAM) | Recursive reasoning composition → analogy is "beyond composition" |
| Research 037 (REAP Model-Based/Modelless) | Directly related — analogy is the "modelless transfer" mechanism |
| Research 062 (SHINE) | Context→LoRA hypernet IS a functor in category-theory terms |
| Research 070 (GDN2) | Linear attention recurrence could implement functor application |
| Research 039 (SpectralQuant) | Eigenbasis alignment ≈ structural alignment in spectral domain |
| riir-ai Research 010 (KG × HLA × Role Transport) | **Direct hit** — role transport operators ARE functors, HLA higher-order moments capture "relations between relations". Dirichlet Energy is the quality diagnostic for KG training. Plan 151 (riir-ai) implements the full pipeline. |
| riir-ai Research 123 (Latent Functor Runtime Guide) | **Super-GOAT guide** — runtime NPC relational learning via functor directions, coherence as collapse detection, freeze/thaw re-estimation, cross-game transfer path. The private counterpart to this open research note. |

## External References

- Code: https://github.com/gouki510/Analogy_in_Transformer
- Park et al. (2025) ICLR — In-Context Learning of Representations (Dirichlet Energy formulation)
- Hendel et al. (2023) — In-context learning creates task vectors (linear subspace)
- Gentner (1983) — Structure-Mapping Theory (cognitive science foundation)
