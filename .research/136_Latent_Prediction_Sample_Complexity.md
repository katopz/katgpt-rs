# Research 136: Learn from Your Own Latents — Sample-Complexity Theory

**Source:** arXiv:2605.27734 (Korchinski, Favero, Wyart — EPFL/Cambridge/JHU)
**Date:** 2026-05-28
**Verdict:** ⚠️ THEORETICAL ONLY — Training paradigm, no inference gain

## Core Finding

Token-level SSL (MLM, diffusion) requires O(m^(L+1)) samples to learn hierarchical latent structure.
Latent-prediction SSL (data2vec, JEPA) requires only O(m³) samples — **exponential improvement**, independent of hierarchy depth L.

| Method | Sample Complexity | Depth Dependence |
|--------|------------------|-----------------|
| Supervised classification | O(m^L) | Exponential |
| Token-level SSL (MLM, diffusion) | O(m^(L+1)) | Exponential |
| Latent prediction (data2vec, ILC) | O(m³) | **Independent of L** |

## Key Mechanisms

### 1. Iterative Latent Clustering (ILC)
- Level-by-level: cluster synonym tuples by their "cousin context vectors"
- Synonyms (same parent) have identical context vectors
- Each level costs O(vm³) — same at every depth
- k-means clustering sufficient

### 2. Stacked Latent-Clustering (SLC)
- Neural implementation of ILC: predictor + clusterer modules
- Predictor predicts cousin tokens (cross-entropy)
- Clusterer assigns soft cluster labels (contrastive loss)
- Teacher-student with EMA prevents collapse
- **Local learning suffices** — stop-gradients between modules still achieve O(m³)

### 3. data2vec Analysis
- data2vec **implicitly performs** ILC/SLC's hierarchical clustering
- Phase-by-phase: level-1 latents → enter teacher target → level-2 latents → ...
- EMA teacher acts as "refreshed target" carrying learned latents
- **H-JEPA stacking is redundant** — single data2vec already hierarchical

## Why No Gain for katgpt-rs

1. **Training paradigm, not inference**: The paper proves sample efficiency for *learning* representations. katgpt-rs is an inference engine — it doesn't train models. The training impact is in riir-ai (wgpu LoRA, ROPD, SDAR, SHINE pipeline).
2. **No inference path impact**: Whether a model was trained with token-level or latent-prediction SSL doesn't change how inference works. The KV cache, attention, and speculation pipelines are identical.
3. **Self-distillation already covered**: Our SDAR (Plan 073) and ROPD (Plan 072) in riir-ai already implement teacher-student distillation with latent targets. The paper validates this design but doesn't improve inference.
4. **Screening pruner**: Synonym clustering could theoretically improve `ScreeningPruner::relevance()` by grouping semantically identical candidates, but the overhead of computing cousin context vectors at inference time violates optimization.md (no allocation in hot loops).

## What We Already Have That This Validates

| Paper Concept | katgpt-rs Equivalent | Status |
|--------------|---------------------|--------|
| Teacher-student EMA | SDAR gated distillation (Plan 072) | ✅ Implemented |
| Hierarchical latent clustering | ROPD rubric criteria (Plan 071) | ✅ Implemented |
| Contrastive clustering loss | Bradley-Terry pairwise ranking | ✅ Implemented |
| Stop-gradient local learning | Freeze/Thaw pipeline (Plan 092) | ✅ Implemented |
| data2vec = implicit hierarchy | VPD variational distillation | ✅ Implemented |

## Theoretical Value

- **Validates SDAR sigmoid gate**: The paper proves latent targets are strictly better than token targets. SDAR's sigmoid-gated teacher representation is a latent target → validates the design.
- **Validates ROPD multi-criterion**: The ILC algorithm clusters by multi-dimensional context vectors. ROPD rubrics are multi-criterion evaluation → same principle.
- **Validates local learning**: Stop-gradients between modules still work → our Freeze/Thaw's per-layer approach is theoretically sound.

## Open/Close Split

- No code in katgpt-rs — purely theoretical validation
- Game training applications → riir-ai Research 025 (private)

## References

- Related: Research 036 (ROPD), Research 038 (SDAR), Research 040 (Bradley-Terry), Research 080 (VPD)
- Cross-ref: riir-ai Research 025 (game training application)
- Paper Table 1: data2vec achieves m³ scaling confirmed experimentally for L=3..7
