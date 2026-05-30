# Parallax: Parameterized Local Linear Attention

**Paper:** [arXiv 2605.29157](https://arxiv.org/abs/2605.29157) — Zuo, Pai, Zeng, Dewulf, Hu, Wang (May 2026)
**Verdict:** ⚠️ **Conditional gain — CPU inference limited, GPU training-dependent**
**Domain:** katgpt-rs (inference-side attention mechanism)

---

## Core Idea

Parallax adds a **learned R projection** alongside Q, K, V that probes the KV covariance:

```
o_PLX = o_SA − Σ_KV · ρ     where ρ = W_R · x
```

- `o_SA` = standard softmax attention output (intercept)
- `Σ_KV` = KV cross-covariance under softmax weights
- `ρ` = learned probe from layer input via extra projection W_R

**Theoretical basis:** Upgrading softmax's local constant estimator (Nadaraya-Watson) to local linear → strictly better bias-variance tradeoff (Theorem 2.1: R(f_GL) ≫ R(f_NW) ≫ R(f_LL)).

## Key Results

| Metric | Parallax vs Transformer |
|--------|------------------------|
| 0.6B perplexity (LAMBADA, Muon) | 18.56 vs 22.15 (**-16.2%**) |
| 1.7B perplexity (LAMBADA, Muon) | 10.80 vs 13.07 (**-17.4%**) |
| Downstream accuracy avg (0.6B, Muon) | 55.99 vs 54.54 (**+1.05**) |
| Downstream accuracy avg (1.7B, Muon) | 62.45 vs 61.43 (**+1.02**) |
| Decode kernel (H200) | Matches/beats FA2/FA3 |
| Arithmetic intensity | 2× FA (compute-bound regime) |

## Critical Finding: Optimizer Dependence

| Optimizer | Parallax advantage | COR (correction-to-output ratio) |
|-----------|-------------------|-------------------------------|
| **Muon + WSD** | **+1.05 avg accuracy** | 8–12 in deep layers |
| AdamW + Cosine | -0.89 avg accuracy (**worse**) | < 4 in all layers |
| AdamW + WSD | +0.07 avg accuracy (marginal) | < 4 |

**Root cause:** Muon maintains high stable rank in W_R (134.0) vs AdamW (9.3–11.1). The gate test shows AdamW learns to *suppress* the correction branch (gate → 0.26), while Muon opens it (gate → 0.95).

**Implication:** Without Muon optimizer, Parallax provides no gain. Our LoRA training pipeline (riir-ai/riir-gpu) uses AdamW-style optimizers.

## Streaming Algorithm (Algorithm 1)

Two parallel branches sharing KV stream:
- **Softmax branch:** QK^T → softmax → PV (standard FA)
- **Covariance branch:** RK^T → fused with softmax weights → PV

Extra state: (d2, O2) alongside FA's (m, d1, O1). No extra HBM I/O per iteration.

**Decode optimization:** Joint QK+RK in same WGMMA accumulator on Hopper GPUs. The R row fills otherwise-idle accumulator rows during single-query decode.

## Applicability to katgpt-rs

### What We Have

| Component | katgpt-rs equivalent | Status |
|-----------|---------------------|--------|
| Attention mechanism | SDPA (default), HLA (O(1) cache), GDN2 (O(1) recurrent), DashAttention (sparse) | GOAT proved |
| Streaming algorithm | PFlash block-sparse prefill, tiled_attention (CPU SIMD) | Available |
| R projection | None | — |
| Muon optimizer | `newton_schulz` feature flag (Plan 152) | GOAT proved 25/25 (Bench 050), default-on |

### Distillation Opportunities

1. **CPU SIMD R projection:** The W_R · x matmul is a standard linear projection. On CPU SIMD, this is ~O(d²) FLOPs with no kernel launch overhead. Our `coda_fusion` could fuse R projection with existing Q projection.

2. **Covariance correction in AHLA/HLA:** Our AHLA already maintains Σ_K = K^T K (second-order sufficient statistics). The Parallax covariance Σ_KV = E_p[(v - v̄)(k - k̄)^T] could potentially be maintained as additional running statistics in AHLA's O(1) state. This would give Parallax-style correction at O(1) decode cost.

3. **Post-training adaptation:** Setting W_R = 0 recovers exact softmax behavior. Any pretrained transformer checkpoint can be converted by adding W_R and fine-tuning. This aligns with our LoRA pipeline — W_R could be a LoRA adapter.

### Why NOT Default-On

| Reason | Evidence |
|--------|----------|
| **Muon-dependent** | AdamW makes Parallax *worse* or marginal. Our training uses AdamW |
| **CPU inference limited** | The 2× arithmetic intensity advantage is GPU-specific (WGMMA sharing). On CPU, the extra matmul is pure overhead |
| **No pretrained model uses it** | Requires training from scratch or fine-tuning with W_R initialization |
| **Adds 1 linear projection** | ~d² extra parameters per layer, ~2× attention FLOPs |

### What IS Extractable (No Training Dependency)

1. **Attention sink reduction insight:** Parallax's correction branch absorbs the routing role of attention sinks. Our `dash_attn` already handles sparsity, but the insight that covariance correction reduces sink dependency is useful for future attention design.

2. **Score range extension:** Allowing negative attention weights (via the correction) provides expressiveness. Our `dash_attn` with α-entmax already allows some sparsity, but negative weights are a different expressiveness axis.

3. **AHLA covariance branch:** Maintaining Σ_KV alongside AHLA's Σ_K is a potential future O(1) extension that doesn't require Parallax's specific formulation.

## Alignment with Optimization Guidelines

From `.contexts/optimization.md`:

| Guideline | Parallax compatibility |
|-----------|----------------------|
| "GPU for microsecond workloads is net negative" | ✅ Parallax decode is GPU-optimized, but our inference is CPU. The extra R·K matmul (~O(L·d)) per decode step is avoidable overhead on CPU |
| "Pre-compute lookup tables once" | ❌ Covariance is query-dependent (softmax-weighted), can't pre-compute |
| "Verify with release build" | Would need release benchmarks with/without the feature |
| "Feature flags affect binary layout" | ⚠️ The extra projection changes attention path — must benchmark cold cache impact |

## Verdict

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| **GOAT potential** | ⏳ Unproven | Requires Muon optimizer + from-scratch training. Cannot prove gain with AdamW |
| **Inference gain** | ❌ No | CPU decode adds ~2× attention FLOPs for marginal quality gain without Muon-trained weights |
| **Architecture insight** | ✅ Yes | Local linear > local constant is fundamental. Covariance correction as additive branch is clean |
| **Feature gate** | `parallax_attn` (opt-in) | Only useful with Muon-trained models. NOT default-on |
| **Game relevance** | ❌ None | Pure language modeling attention mechanism. Stays in katgpt-rs domain |

### Final Verdict: **NO GAIN for current stack**

The core idea (learned covariance correction) is sound and theoretically grounded. However:
1. **Muon optimizer is a hard prerequisite** — without it, Parallax degrades
2. **CPU inference gets no compute advantage** — the WGMMA sharing trick is GPU-only
3. **No pretrained Parallax models exist** — can't validate quality at our scale
4. **The AHLA covariance extension** is the only extractable idea, but it's speculative

**Recommendation:** `newton_schulz` is now default-on. Re-evaluate Parallax when a Muon-trained model is available. The post-training adaptation path (W_R = 0 init + LoRA fine-tune) is viable for future LoRA distillation targets.

**Plan:** [`.plans/135_parallax_attn.md`](../.plans/135_parallax_attn.md)
