# Research: MoA — Mixture of Activations for Token-Adaptive FFN

**Date:** 2026-05-27
**Status:** Verdict → Plan
**Context:** katgpt-rs inference engine, activation functions, FFN layers
**Paper:** "More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations" (arXiv:2605.26647) — ByteDance Seed + PKU

---

## TL;DR

MoA replaces the single fixed activation in FFN layers (ReLU, SiLU, GELU) with a token-adaptive mixture. A lightweight sigmoid gate `π_k(x) = sigmoid(u_k^T x)` produces per-token mixing weights over a small dictionary of activations (ReLU, ReLU², GELU, SiLU, LeakyReLU, tanh, Id). The same linear projections W1, W2 are shared — only O(d) extra gating parameters per activation entry. Strict expressivity hierarchy proven: Fixed-activation ⊊ LA (learnable, input-independent) ⊊ MoA (input-dependent). Best variant: Type-II bi-MoA with sigmoid gating, applied to both branches of SwiGLU. Consistent loss reduction across 0.12B-2B dense and MoE models. Overhead: 1.03-1.13× wall-clock, memory unchanged.

---

## Key Ideas

### 1. Expressivity Hierarchy (Theorem 4.1, 4.2)

```
Fixed-activation FFN  ⊊  LA (learnable coefficients)  ⊊  MoA (input-dependent gates)
```

- **LA** adds scalar coefficients `α_k` per activation — same mix for all tokens. Width-1 LA can represent `ReLU(x) + ReLU²(x)`, which no fixed-activation FFN can at any finite width.
- **MoA** makes coefficients input-dependent via `π_k(x) = sigmoid(u_k^T x)`. Width-1 MoA can represent `tanh(λx₁) · ReLU(x₂)` — a ridge singularity with non-constant amplitude. LA can only produce constant-amplitude ridges.

### 2. Type-II bi-MoA (Best Variant)

For SwiGLU-style FFN `g(x) = W₃(σ(W₁x) ⊙ W₂x)`:

```
g_bi-MoA(x) = W₃ [ Σ_k ρ_k(x) σ_k(W₁x) ] ⊙ [ Σ_ℓ π_ℓ(x) σ_ℓ(W₂x) ]
```

Both branches get token-adaptive mixing. Dictionary: `{Id, GELU, SiLU, ReLU, LeakyReLU, ReLU², tanh}`.

### 3. Overhead Analysis (Table 4)

| Model | Wall-clock | Memory |
|-------|-----------|--------|
| Type-I MoA | 1.03× | 1.00× |
| Type-II MoA | 1.13× | 1.00× |

Extra params: O(d · |K|) where |K| = 7 activations. For d=4096, that's ~28K params vs millions in FFN.

### 4. Key Experimental Findings

- **Sigmoid gating > softmax > tanh** (Table 2). Theory uses tanh for proofs, but sigmoid (sigmoid(0) = 0.5) gives better optimization dynamics.
- **MoA tolerates larger learning rates** — 1.5-2× the baseline peak lr.
- **Scaling law stable** — gap persists across 0.25B-2B MoE models (Figure 3).
- **Vision transfer** — works on MAE ViT-B too (Figure 5).

---

## Distillation for katgpt-rs

### What We Already Have

1. **`GateActivation` enum** (`coda.rs`): ReLU, SiLU, GeGELU-Tanh, GeGELU — applies single activation to entire FFN output. This is the "fixed-activation" baseline.
2. **`swiglu()`** (`types.rs:1408`): `hidden[i] = silu(gate[i]) * up[i]` — standard SwiGLU, both branches fixed.
3. **`simd_matmul_rmsnorm_swiglu()`** (`coda.rs:238`): Fused matmul + RMSNorm + SwiGLU — optimized for the fixed-activation path.
4. **Sparse MLP** (feature `sparse_mlp`): Runtime index packing for ReLU-sparsity — compatible with MoA since MoA uses soft gating (all activations contribute, no hard routing).

### What MoA Adds at Inference Time

For **inference support** of MoA-trained models:

```
// Current: fixed SwiGLU
output = W₃(silu(W₁x) ⊙ W₂x)

// MoA bi-MoA with sigmoid gating:
y = W₁x,  z = W₂x
ρ_k = sigmoid(u_k^T x)   // gate params per activation (both branches)
π_ℓ = sigmoid(v_ℓ^T x)
output = W₃ [ Σ_k ρ_k · σ_k(y) ] ⊙ [ Σ_ℓ π_ℓ · σ_ℓ(z) ]
```

The activation dictionary `σ_k` includes: Id, ReLU, ReLU², LeakyReLU, GELU, SiLU, tanh — all already implementable from our existing `GateActivation`.

### Compatibility Assessment

| Component | Impact | Change Needed |
|-----------|--------|---------------|
| `GateActivation::activate()` | Extend to MoA dictionary | Add ReLU², LeakyReLU, Id to enum |
| `swiglu()` | Replace with `moa_swiglu()` | Token-adaptive mixing loop |
| `simd_matmul_rmsnorm_swiglu()` | Add MoA variant | Fused: matmul + MoA mixing + RMS |
| Weight loading | Add MoA gate params `u_k, v_ℓ` | New fields in weight struct |
| Sparse MLP | Compatible | Soft gating = all activations contribute |

---

## Verdict

### ✅ GAIN — Inference Support (Feature Gate: `moa_inference`)

**Why:** Supporting MoA models at inference is a **free lunch** — 1.03-1.13× overhead for strictly more expressive FFN. If major model families adopt MoA (ByteDance already ships it), we need to support it.

**What:** Feature-gated MoA forward pass. When `moa_inference` is ON and model weights contain MoA gating params, use token-adaptive mixing. Otherwise, fall back to existing fixed-activation path (zero perf impact).

**GOAT proof needed:**
1. MoA forward produces identical output to paper's reference (correctness)
2. MoA overhead ≤ 1.15× vs fixed SwiGLU (perf bound from paper Table 4)
3. Fallback to fixed activation when no MoA weights present (zero regression)

### Scope: katgpt-rs (open, inference only)

This is an **inference engine** change — we don't train models. The MoA gating weights come from the model file. Our job is to efficiently execute the MoA forward pass. This is the same relationship we have with SwiGLU, GeGLU, etc.

### NOT a Pillar (for decision matrix)

MoA is a **model architecture feature** — it improves model quality, but doesn't create game-specific IP. It's a "support more models" capability, not a "unique game AI capability." Listed in the "What Is NOT a Pillar" section.

### Cross-reference: riir-ai

If MoA improves LoRA training quality (game LoRA adapters converge better), that's a riir-ai win. But that's a **training-time** change in the wgpu pipeline, not an inference concern. Tracked separately in riir-ai research.

---

## References

- Paper: arXiv 2605.26647 (ByteDance Seed + PKU, May 2026)
- Related: Our Sparse MLP (R008, Plan 022), GateActivation enum, SwiGLU implementation
- riir-ai companion: R017 (MoA Training for Game LoRA)
