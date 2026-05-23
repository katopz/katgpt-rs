# Lighthouse Attention: Distilled & Mapped to Our Stack

**Paper:** [Long Context Pre-Training with Lighthouse Attention](https://arxiv.org/pdf/2605.06554) — Peng, Ghosh, Quesnelle (Nous Research), May 2026
**Code:** https://github.com/ighoshsubho/lighthouse-attention

## TL;DR

Lighthouse Attention is a **training-only** hierarchical selection mechanism that wraps around stock FlashAttention. It pools Q, K, V symmetrically into a multi-resolution pyramid, selects top-K entries via parameter-free ℓ2-norm scoring, then runs dense attention on the gathered sub-sequence. After pretraining, a brief dense-SDPA resume recovers full attention quality. Result: **1.4–1.7× wall-clock speedup** at 98K+ context, scales to 1M tokens on 32 B200 GPUs.

**Our verdict:** Lighthouse's algorithmic patterns (pyramid pooling, parameter-free scoring, top-K selection → dense sub-sequence, non-differentiable pruning) directly validate and extend our existing PFlash + Raven + TurboQuant stack. The key new insight is **symmetric Q/K/V pooling** at training time, which we don't do but could inspire a training-time mode for `riir-burner`.

---

## Core Algorithm (4-Stage Pipeline)

```
Input: X ∈ R^{N×d_model}

Stage 1 — Pyramid Pool (symmetric):
  Q, K, V = X @ W_Q, X @ W_K, X @ W_V
  For ℓ = 0..L-1: mean-pool Q, K, V by factor p^ℓ → (Q^(ℓ), K^(ℓ), V^(ℓ))
  Total pyramid entries: Σ N/p^ℓ ≤ N·p/(p-1)  →  O(N)

Stage 2 — Hierarchical Selector (parameter-free):
  Score: s_QK[ℓ,i] = max(||Q_j||_2 for j in window)   // max-pooled L2 norms
         s_KQ[ℓ,i] = max(||K_j||_2 for j in window)
  Select: Chunked-bitonic top-K across all levels → indices I
  No learnable parameters. No gradient through selection.

Stage 3 — Dense Sub-Sequence Attention:
  Gather: (Q^(ℓ), K^(ℓ), V^(ℓ)) indexed by I → contiguous [S, d]
  Attend: eO = FlashAttention(eQ, eK, eV)   // stock FA, no custom sparse kernel
  S = N/p^{L-1} + (L-1)·p·k  ≈  65K at N=1M, L=4, p=4, k=4096

Stage 4 — Scatter-Back:
  Write eO to shifted ranges R(ℓ,i) = [i·p^ℓ + p^ℓ - 1, i·p^ℓ + 2·p^ℓ - 2]
  Sum contributions across levels → O ∈ R^{N×d}
```

**Key properties:**
- **No new parameters** — pyramid is fixed pooling, scorer is parameter-free
- **No custom attention kernel** — selection is outside attention, uses stock FlashAttention
- **No straight-through estimator** — top-K is non-differentiable, gradients flow through values only
- **Two-stage training** — pretrain with Lighthouse, then brief dense SDPA resume

---

## Key Results

### SDPA Recoverability (Central Claim)
| Recipe | Steps (LH→SDPA) | Final Loss | Tok/s (k) |
|--------|:---:|:---:|:---:|
| Dense SDPA from scratch | 0→16k | 0.7237 | 45.6 |
| **LH → SDPA (10k+6k)** | **10k→16k** | **0.6980** | **75.0** |
| LH → SDPA (11k+5k) | 11k→16k | 0.7001 | 75.4 |
| LH → SDPA (12k+4k) | 12k→16k | 0.7102 | 74.7 |

All Lighthouse resumes **match or beat** dense-from-scratch at matched token budget.

### Scaling vs Context Length (single B200, L=3, p=4, sparsity 1:64)
| Context | SDPA Fwd | LH Fwd | Speedup |
|---------|----------|--------|:---:|
| 8K | baseline | baseline | ~1× |
| 32K | ~4× slower | ~1.2× slower | ~3.3× |
| 128K | ~16× slower | ~2× slower | ~8× |
| 512K | ~64× slower | ~3× slower | **21×** |

### Best Ablation Cell
**L=3, p=2, k=1536, dilated scorer** → 0.6825 loss (best in grid), 1.69× wall-clock speedup.

### Needle-in-a-Haystack (98K, post-resume)
k=2048 dilated: **0.76 mean retrieval** (vs 0.72 dense baseline). Lighthouse-trained models retrieve *better* after resume.

---

## Mapping to Our Stack

### What We Already Have (Validated by Lighthouse)

| Lighthouse Concept | Our Implementation | Status |
|---|---|---|
| Parameter-free block scoring | `PFlash` `BlockAttentionScorer` (mean-Q × mean-K dot) | ✅ We use dot-product; they use L2 norms. Both parameter-free. |
| Top-K selection → dense sub-sequence | `block_select()` → `compress_prompt_blocks()` → target prefill | ✅ Same pattern: score → select → gather → dense attention |
| Chunked GPU selection kernel | `flashprefill_block_select.wgsl` | ✅ We have 4 WGSL kernels for PFlash GPU pipeline |
| Non-differentiable pruning | `ConstraintPruner` / `ScreeningPruner` — all non-differentiable | ✅ Same philosophy: projections learn to be useful when selected |
| Sparsity ≈ 1:64 | PFlash 4K→192 tokens (21.3× reduction ≈ 1:21 sparsity) | ✅ Our sparsity is more aggressive |
| Multi-level hierarchy | `Raven RSM` fixed-size slots with Top-K routing | ⚡ Raven is flat Top-K; Lighthouse adds pyramid levels |
| O(S²d) on sub-sequence | `flashprefill_sparse_forward.wgsl` (online softmax over selected blocks) | ✅ Same concept |

### What's New from Lighthouse (Potential Extensions)

| Lighthouse Innovation | Our Gap | Opportunity |
|---|---|---|
| **Symmetric Q/K/V pooling** | PFlash only pools K (mean-K scoring), not Q or V | Could extend PFlash to pool Q+K+V symmetrically for richer multi-scale representation |
| **L-level pyramid** (multi-resolution) | PFlash is single-resolution blocks | Pyramid adds coarse-to-fine scoring: cheaper, better coverage |
| **Scatter-back reconstruction** | PFlash compresses input only, no output reconstruction | For training-time use in `riir-burner`, scatter-back enables gradient flow to all positions |
| **Training-time sparse attention** | We only do inference-time (PFlash for TTFT reduction) | `riir-burner` could use Lighthouse for faster LoRA training on long sequences |
| **Two-stage training** (sparse → dense resume) | N/A — we don't pretrain | If `riir-burner` ever does continued pretraining, this recipe applies directly |
| **Context parallelism** (ring attention on gathered sub-sequence) | N/A — single-GPU training | Future: multi-GPU LoRA training with CP via `riir-gpu` |
| **Max-pooled L2-norm scorer** | PFlash uses dot-product scorer | L2 norms are cheaper (no cross-head interaction); could benchmark against our dot-product scorer |

### Architecture Overlap Diagram

```
Lighthouse Attention                    Our Stack (microgpt-rs + riir-ai)
─────────────────────                   ─────────────────────────────────

Pyramid Pool (Q,K,V)                    Raven RSM (flat slot memory)
  └─ L levels, factor p                   └─ Fixed slots, Top-K routing
                                         
Hierarchical Selector                   PFlash Block Selector
  ├─ L2-norm scoring                      ├─ mean-Q × mean-K dot scoring
  ├─ Chunked-bitonic top-K                ├─ Heuristic rules (sink+window+α)
  └─ No parameters                        └─ `flashprefill_block_select.wgsl`
                                         
Dense Gather + FlashAttention            PFlash Compressed Prefill
  └─ Stock FA on sub-sequence             └─ Target prefill on compressed tokens
                                         
Scatter-Back                             (none — inference-only, no reconstruction)
  └─ Write to shifted ranges              
                                         
TurboQuant (2-4 bit KV)                 TurboQuant KV Cache
  └─ N/A (orthogonal technique)           └─ Same: random rotation + Lloyd-Max
                                         
Two-Stage Training                       riir-burner LoRA Training
  └─ Sparse pretrain → dense resume       └─ Could adopt: sparse attention → dense LoRA
```

---

## Design Decisions Worth Noting

### 1. Why Symmetric Q/K/V Pooling?
Prior work (NSA, HISA, InfLLM-v2) pools only K,V — natural for inference (one query at a time). Training exposes all queries in parallel, so pooling Q too:
- Reduces dense attention from O(N·S·d) to O(S²·d) at training time
- Puts pooled Q and pooled K in same representation space
- Creates coherent (Q,K,V) triples at each level
- **No holes** in scattered output (critical for stable training gradients)

### 2. Why Parameter-Free Scoring?
L2 norms are strictly weaker than any learned scorer. Any positive result is a **lower bound** on what richer scorers can achieve. This avoids:
- Scorer collapse (learned scorer overpowers attention)
- Scorer-attention misalignment
- Auxiliary loss tuning
- Extra backward-pass computation

### 3. Why Selection Outside the Attention Kernel?
Lighthouse gathers a dense sub-sequence, then calls stock FlashAttention:
- Same kernel at training and inference (no train/serve divergence)
- Inherits FA improvements automatically (FA-3, FA-4)
- Ring attention works without sparse-aware collectives
- Disabling selection recovers dense baseline exactly

### 4. Why Non-Differentiable Top-K (No STE)?
Projections learn to produce **values useful when selected**, not scores good at selecting. This is philosophically identical to our `ConstraintPruner`/`ScreeningPruner` design — the pruner doesn't learn, the model learns to produce tokens that survive pruning.

### 5. Counter-Intuitive: Smaller k → Better Loss
At 50B token budget, k=1536 beats k=4096 (0.6825 vs 0.6951). Hierarchical selection appears to **regularize** at small budgets. Whether this reverses at larger budgets is unknown.

---

## Complexity Comparison

| Method | Per-Layer Compute | Notes |
|--------|:---:|---|
| Dense Softmax | Θ(T²·d) | Quadratic wall |
| Log-Linear Attention | Θ(T·log T·d) | Super-linear |
| **Lighthouse (bounded k)** | **Θ(T·d)** | Linear! At bounded k |
| Linear Attention / SSMs | Θ(T·d) | Same class, but compresses entire past |
| **Our PFlash** | Θ(S²·d) where S≪T | Inference-time only, same sub-sequence pattern |

Key insight: Lighthouse is **linear in T** at bounded k because only S tokens attend to each other, while linear-cost stages (projection, pooling, scoring, scatter) dominate. The log factor lives in S, not total compute.

---

## Applicable Techniques for Each Project

### microgpt-rs
- **Pyramid scoring for PFlash**: Replace single-level block scoring with L-level pyramid. Coarser levels cost almost nothing (fewer entries) and guarantee every region contributes.
- **L2-norm scorer**: Benchmark against our dot-product scorer. L2 norms are cheaper (no mean-Q × mean-K dot) and Lighthouse shows they work.
- **Stratified top-K**: Lighthouse's chunked-bitonic naturally produces stratified selection (every region gets some tokens). Our `block_select` uses heuristic rules for similar effect.

### riir-ai / riir-gpu
- **Multi-level GPU kernels**: Current PFlash is 4 kernels (mean_K → block_score → block_select → sparse_forward). Pyramid adds L levels but each level is smaller — kernel design scales.
- **Scatter-back kernel**: New `lighthouse_scatter_back.wgsl` would be needed for training-time use.

### riir-burner
- **Long-context LoRA training**: If training on long sequences (code corpus, RAG chunks), Lighthouse-style sparse attention could accelerate training.
- **Two-stage recipe**: Pretrain LoRA with sparse attention, then brief dense resume for inference-ready adapter.

### gist/anyrag
- **Long-document ingestion**: RAG ingestion often processes long documents. Lighthouse's pyramid scoring could improve chunk selection for retrieval.

---

## What We Should NOT Adopt

1. **Training-time scatter-back** — We're inference-focused. PFlash's compress-only approach is correct for TTFT reduction.
2. **Two-stage training** — We don't pretrain. riir-burner does LoRA fine-tuning, not continued pretraining.
3. **Context parallelism** — We're single-GPU (Apple Silicon). Multi-node ring attention is not our target.
4. **Dilated softmax scorer** — 9% more expensive than L2-norm, within noise of parameter-free. Not worth the complexity.

---

## Key Takeaways

1. **Parameter-free scoring works** — L2 norms match learned scorers within 0.01 loss. Our PFlash dot-product scorer is already in this spirit.
2. **Selection outside the kernel is the right architecture** — Our PFlash already does this (score → select → compress → target prefill). Lighthouse validates this pattern at pretraining scale.
3. **Symmetric Q/K/V pooling is the novel insight** — We could extend PFlash to pool Q alongside K for richer scoring, especially at training time.
4. **Smaller selection budgets may be better** — k=1536 beats k=4096 at our token scales. Agrees with our PFlash finding that aggressive compression (21.3×) preserves NIAH retrieval.
5. **Dense resume recovers quality** — Sparse training doesn't hollow out the model. Critical validation for anyone considering training-time sparse attention.

---

## Reference

```bibtex
@article{peng2026lighthouse,
  title     = {Long Context Pre-Training with Lighthouse Attention},
  author    = {Peng, Bowen and Ghosh, Subho and Quesnelle, Jeffrey},
  year      = {2026},
  note      = {Nous Research. Symmetric Q/K/V pyramid, parameter-free L2-norm scoring,
               chunked-bitonic top-K, stock FlashAttention on gathered sub-sequence,
               1.4-1.7× wall-clock speedup at 98K+ context on B200}
}
```
