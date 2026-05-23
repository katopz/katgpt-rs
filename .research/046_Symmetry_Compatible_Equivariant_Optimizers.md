# Research: Symmetry-Compatible Principle for Optimizer Design (46)

> Source: [Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers](https://arxiv.org/abs/2605.18106) by Tim Tsz-Kit Lau & Weijie Su (UPenn), May 2026
> Local: `.raw/equivariant_optimizers/` (PyTorch training code)
> Date: 2026-05 (paper), distilled 2026-05
> **Verdict: HIGH VALUE — Layerwise symmetry-compatible optimizer assignments directly applicable to our LoRA training stack in `riir-gpu`. The architecture–optimizer co-design principle maps cleanly to our parameter routing system.**

## TL;DR

The paper proves that coordinate-wise optimizers (AdamW) are **geometrically mismatched** for matrix-valued parameters. Different layer types have different symmetry groups, and optimizers should match those symmetries. They derive a layerwise optimizer stack:

| Layer Type | Symmetry Group | Optimizer Class |
|---|---|---|
| Linear / attention weights | O_d_out × O_d_in (bi-orthogonal) | Full spectral (Muon/PolarGrad) |
| Embedding / LM head | P_v × O_d (permutation × orthogonal) | RowNorm / RightPolarGrad / Hybrid |
| SwiGLU gate/up projections | P_dff × O_d (neuron permutation) | Row-aware LPRO variants |
| SwiGLU down projection | Transposed neuron geometry | Column-aware LPRO variants |
| MoE router | P_e × shift-invariant (expert perm + shared offset) | Centered RowNorm / LeftPolarGrad |

**Key empirical result:** RowNormM for embedding/LM head consistently beats AdamW across all tested architectures (Qwen3-0.6B, Gemma 3 1B, OLMoE-1B-7B, gpt-oss). Gains scale with vocabulary size.

## Why This Matters for Our Stack

### 1. We Already Have the Parameter Routing Infrastructure

Our `riir-gpu` already distinguishes parameter types by role:
- LoRA adapters on attention layers (Q/K/V/O projections)
- LoRA adapters on MLP layers (gate/up/down projections)
- Separate handling for embedding and LM head in the inference path

The paper's `build_transformer_param_groups()` routing logic is conceptually identical to what we need. Their roles (`matrix_attention`, `mlp_gate_up`, `mlp_down`, `embedding`, `lm_head`, `moe_router`) map directly to our layer-wise LoRA adapter groups.

### 2. LoRA Training Benefits from Symmetry-Compatible Updates

We train LoRA adapters (low-rank A, B matrices) via wgpu. The paper's row-norm updates are **computationally cheap** — just row-wise scaling of momentum. This is perfect for our WGSL kernel pipeline:

- **RowNormM**: `η(‖M_{i:}‖₂) * M_{i:}` per row — trivial WGSL kernel
- **Centered RowNormM** (router): Same but with `Π⊥ = I - (1/e)·1·1ᵀ` centering
- **HybridPolarGradM**: Row-norm + right-spectral — needs Gram inverse-square-root (Newton-Schulz)

### 3. Architecture–Optimizer Co-Design = Our Model-Based/Modelless Pattern

The paper's philosophy mirrors our model-based/modelless duality (Research 37):

| Paper Concept | Our Equivalent |
|---|---|
| Layer symmetry group determines optimizer | Parameter role determines training strategy |
| Bi-orthogonal → spectral (Muon) | Model-based → GRPO/DPO (gradient updates) |
| LPRO → row-norm (cheap) | Modelless → BanditPruner (no gradients) |
| Hybrid (row-norm + spectral) | Phase bridge (modelless signal → model-based training) |
| Architecture–optimizer co-design | Domain-specific inference budget |

The symmetry principle provides a **theoretically grounded criterion** for which parameters need expensive (spectral/model-based) vs cheap (row-norm/modelless) updates.

## Distillable Ideas

### D1: RowNormM for LoRA Adapters (Modelless Path)

The cheapest symmetry-compatible optimizer. For LoRA adapters on embedding/LM head:

```text
For momentum M (EMA of gradients):
  D_η = Diag(η(‖M_{1:}‖₂), ..., η(‖M_{v:}‖₂))
  update = D_η · M
  where η(t) = 1/(t + ε)  (smoothed row normalization)
```

This costs O(v*d) — just row norm + scale. No SVD, no matrix decomposition. Perfect for our WGSL training kernels.

**Applicable to:** LoRA adapters on SwiGLU gate/up projections (row-aware), down projection (column-aware via transpose).

### D2: Router Centering for Expert Routing (Modelless Path)

For MoE routers, the paper derives that expert-permutation symmetry + shared-logit-shift invariance requires **centering** before updates:

```text
Π⊥ = I_e - (1/e) · 1_e · 1_eᵀ   (project out shared mean)
M_c = Π⊥ · M                      (center momentum)
update = Diag(η(‖M_c_{i:}‖₂)) · M_c
```

This is directly applicable to our `riir-router` prompt routing — expert/domain scores should be centered before bandit updates.

### D3: Hybrid Row-Norm/Spectral for LoRA (Model-Based Path)

For model-based training (GRPO/DPO in `riir-gpu`), combining row normalization with spectral updates:

```text
Right-spectral/row-norm order:
  Z = M · (MᵀM + εI)^{-1/2}   // right-spectral (Newton-Schulz, ~5 steps)
  update = Diag(η(‖Z_{i:}‖₂)) · Z   // then row-norm
```

The Gram inverse-square-root can be done in WGSL via Polar Express or Newton-Schulz iteration. The paper uses 5 inner steps with ε=1e-7.

### D4: Per-Role Learning Rate Strategy

The paper demonstrates that different parameter roles need different learning rates:

| Role | LR (Qwen3) | LR (Gemma 3 1B) |
|---|---|---|
| Scalar/vector (AdamW) | 0.05 | 0.05 |
| Matrix hidden/attn (Muon) | 0.02 | 0.02 |
| Embedding (RowNormM) | 0.50 | 0.0025 |
| LM head (RowNormM) | 0.005 | 0.0025 |
| SwiGLU MLP (Muon) | 0.02 | 0.02 |

Key insight: embedding LRs are **25× higher** than matrix LRs with RowNormM. This makes sense — row normalization constrains update magnitude, so higher LR is safe.

### D5: SwiGLU Intermediate-Neuron Permutation Symmetry

The paper proves that SwiGLU gate/up/down projections have **intermediate-neuron permutation symmetry** (Proposition 3.4), NOT full left-orthogonal symmetry. This means:

- Gate/up projections: row-aware updates (permutation on rows = neurons)
- Down projection: column-aware updates (permutation on columns = neurons)
- The same permutation applied to all three projections leaves the function unchanged

This is critical for our LoRA adapter placement on SwiGLU blocks.

## What We Don't Need

1. **Full Muon/PolarGrad for our use case** — We train LoRA adapters (low-rank A, B), not full weight matrices. Spectral optimizers shine for full matrix training. For low-rank adapters, the row-norm updates are sufficient and much cheaper.

2. **Distributed Muon/Dion** — Our training runs on single GPU (Apple M-series). No need for distributed orthogonalization.

3. **Polar Express / QDWH / ZOLO-PD** — These are heavy numerical methods for accurate polar decomposition. For LoRA training, Newton-Schulz (5 steps) is sufficient, and even that may be overkill — RowNormM alone already beats AdamW.

## Implementation Priority

| Priority | Component | Path | Complexity |
|---|---|---|---|
| **P1** | RowNormM WGSL kernel | Modelless (riir-gpu) | Low — just row norm + scale |
| **P1** | Per-role LR configuration | Both | Low — config change |
| **P2** | Router centering in bandit | Modelless (microgpt-rs) | Low — project out mean |
| **P2** | SwiGLU row/column-aware updates | Model-based (riir-gpu) | Medium — need transpose handling |
| **P3** | Newton-Schulz Gram invsqrt | Model-based (riir-gpu) | Medium — ~5 iteration WGSL kernel |
| **P3** | Hybrid row-norm/spectral | Model-based (riir-gpu) | High — compose P1 + P3 |

## Key Numbers (Cross-Model Results)

| Model | Params | Vocab | RowNormM val loss | AdamW val loss | Gap |
|---|---|---|---|---|---|
| Qwen3-0.6B | 626M | 152K | **4.1991** | 4.2084 | −0.0093 |
| Gemma 3 1B | 1.1B | 262K | **4.0552** | 4.0862 | −0.0310 |
| OLMoE-1B-7B | 2.8B | 50K | **4.0814** | 4.1155 | −0.0341 |
| gpt-oss | 3.5B | 201K | **4.3090** | 4.3704 | −0.0614 |

Trend: **Larger vocabulary → bigger gain** from symmetry-compatible updates.

## References

- Paper: https://arxiv.org/abs/2605.18106
- Code: https://github.com/timlautk/equivariant_optimizers
- Local: `.raw/equivariant_optimizers/`
- Related research: `37_REAP_Model-Based_Modelless_Duality.md` (role-based routing), `21_G-Zero_Self-Play_Open-Ended_Generation.md` (δ signal), `38_SDAR_Self_Distilled_Agentic_RL.md` (gated distillation)
- Related plans: `049_g_zero_self_play.md` (Phase 2 gradient training), `071_ropd_rubric_modelless.md` (modelless rubric)