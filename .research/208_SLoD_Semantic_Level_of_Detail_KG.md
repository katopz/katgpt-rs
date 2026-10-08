# Research 208: SLoD — Semantic Level of Detail for Knowledge Graphs via Spectral Heat Diffusion

**Date:** 2026-06
**Status:** GOAT Verdict — GAIN (Modelless Fusion)
**Domain:** Modelless (inference-time KG resolution control)
**Relates To:** Plans 156 (Spectral Hierarchy), 196 (KG Latent Octree), 209 (FOL Rules), 210 (INSIGHT); Research 121, 196, 207
**Paper:** arxiv 2603.08965 — Izgorodin, Mnemoverse.AI, GRAAI 2026

---

## Executive Summary

**SLoD (Semantic Level of Detail)** is a modelless framework for continuous-resolution zoom on knowledge graphs via heat kernel diffusion on a graph Laplacian induced by hyperbolic (Poincaré ball) kNN embeddings. Key innovations:

1. **Continuous zoom operator** Φ_σ: scales from point detail (σ→0) to uniform averaging (σ→∞)
2. **Emergent scale boundaries** from spectral gaps in the graph Laplacian — no manual Leiden γ sweep
3. **Hierarchical coherence guarantee** (Theorem 1): Lipschitz continuity of Fréchet centroid trajectory with O(σ) error
4. **Three-signal boundary detection**: representation velocity V(σ), weight divergence D_w(σ), neighborhood churn C_k(σ)
5. **Validated**: macro ARI=1.00 at high SNR (HSBM), Kendall τ=0.79 on WordNet (82K synsets)

### GOAT Verdict: GAIN for katgpt-rs (Modelless)

**Why GAIN:**
- We already have `spectral_hierarchy` module with eigenspace alignment, Haar wavelets, Cauchy interlacing
- SLoD extends this to a **continuous zoom operator** on KG structures — modelless, no training
- Fréchet mean in tangent space is pure linear algebra — SIMD-acceleratable
- Spectral gap detection is O(N·K²_eigs) with Lanczos — linear in N
- Maps directly to our existing `ConstraintPruner` → DDTree pipeline: SLoD boundaries select which abstraction level to prune at
- Hyperbolic kNN graph construction is the key geometric step (5.7pp meso ARI gain over Euclidean per ablation)
- The composite boundary score (V + D_w + C_k) is z-score normalized — modelless, no learned weights

**Why NOT for riir-ai (model-based):**
- SLoD is fundamentally modelless — it operates on pre-existing embeddings, no training required
- The hyperbolic embeddings can come from LoRA-trained models (fuel), but the SLoD operator itself is pure math
- No LoRA training component needed — this is inference-time resolution control

---

## Paper Distillation

### Core Algorithm (Algorithm 1: SLoD via Tangent-Space Aggregation)

```
Input: Embeddings V ⊂ B^d, focus x_0, scale σ
1. Compute heat kernel weights: w_i = K_σ(x_0, v_i) / Σ_j K_σ(x_0, v_j)
2. Initialize μ^(0) = v_{i*} where i* = argmax w_i
3. Iterate (max T=15 steps):
   a. u_i = Log_{μ^(t)}(v_i)  — map to tangent space
   b. ū = Σ_i w_i · u_i        — weighted average
   c. μ^(t+1) = Exp_{μ^(t)}(η·ū) — map back to manifold
   d. Early exit if d_H(μ^(t+1), μ^(t)) < tol
4. Return μ^(t+1) = Fréchet mean at scale σ
```

### Boundary Detection (Algorithm 2: BoundaryScan)

Three complementary signals z-score normalized and combined:
- **V(σ)** = d_H(Φ_{σ+Δσ}, Φ_σ) / Δσ — Fréchet centroid velocity
- **D_w(σ)** = JSD(w(σ) || w(σ+Δσ)) — weight distribution shift  
- **C_k(σ)** = 1 - |N_k(σ) ∩ N_k(σ+Δσ)| / |N_k(σ) ∪ N_k(σ+Δσ)| — neighborhood churn

Peaks in composite S(σ) = α₁·V̂ + α₂·D̂ + α₃·Ĉ detected via MAD threshold.

### Key Theoretical Results

**Theorem 1 (Hierarchical Coherence):**
d_H(Φ_σ₁, Φ_σ₂) ≤ C · |σ₂ - σ₁| · (1+ε)
where C = ℓ · ||L||_{1→1} · D_T(v), independent of N for fixed subtree depth.

**Lemma A.1:** Fréchet mean is (D/2)-Lipschitz in weights on every Hadamard manifold (sharp at N=2).

**Lemma A.2:** Heat kernel weights are 2·||L||_{1→1}-Lipschitz in σ on the simplex.

### Ablation Takeaways (50-seed bootstrap)

| Finding | Detail |
|---------|--------|
| **(a)** Laplacian construction is load bearing | kNN-of-Poincaré-Laplacian: macro ARI 0.42 at r=20 vs direct 0.00 |
| **(b)** Hyperbolic kNN matters | Poincaré vs Euclidean: +5.7pp meso ARI (Wilcoxon p < 10⁻¹⁵) |
| **(c)** Binary kNN matches Gaussian | ≤3pp difference — can drop edge weights |
| **(d)** D_w is metric-agnostic | V and C_k reward hyperbolic structure |
| **(e)** Three indicators locate distinct diffusion events | D_w earliest, V middle, C_k latest |
| **(f)** MAD defaults robust | β ∈ {1,2,3} doesn't change top peak |

### Default Parameters (transferred HSBM → WordNet unchanged)

- k = max(10, min(⌊√N⌋, 50)) for kNN
- (α₁, α₂, α₃) = (1/3, 1/3, 1/3)
- β = 2 (MAD multiplier)
- R = 2 (gap threshold)
- T = 15 iterations, η = 1.0, tol = 10⁻⁶

---

## Fusion Ideas — Creative Applications to katgpt-rs

### Fusion 1: SLoD-Adaptive DDTree Budget Selection
**Idea:** Use SLoD boundary detection to automatically select the DDTree exploration depth. Instead of fixed-depth DDTree, let spectral gaps in the constraint graph determine how deep to search.

**Mapping:**
- KG nodes → DDTree branches (token choices)
- Heat kernel σ → DDTree depth budget
- Spectral boundaries → natural cut points for pruning
- Fréchet centroid → "average semantic direction" at each DDTree level

**Gain:** Automatic depth selection removes the manual `max_depth` parameter. The DDTree explores until the spectral boundary says "coarser search is sufficient."

### Fusion 2: Hyperbolic Constraint Graph for Multi-Domain Pruners
**Idea:** Embed constraint pruners (SynPruner, BomberWasmPruner, domain validators) as nodes in a hyperbolic graph. Use SLoD to zoom between "syntax-only" (coarse) and "full semantic validation" (fine).

**Mapping:**
- Coarse σ → ConstraintPruner (bracket/keyword only)
- Medium σ → SynPruner (AST-level syntax)
- Fine σ → CompilerFeedback (cargo check)
- SLoD boundary → automatic router between validation tiers

**Gain:** Replaces manual tier routing with continuous, data-driven resolution. The boundary scan finds where syntax→semantics transition matters.

### Fusion 3: KG-Adaptive BeliefDrafter Resolution
**Idea:** Use SLoD to control BeliefDrafter's latent prediction resolution. At coarse σ, predict at region level; at fine σ, predict per-token.

**Mapping:**
- BeliefDrafter hidden state → point on manifold
- Latent prediction σ → SLoD diffusion scale
- Spectral gaps → natural prediction resolution transitions

**Gain:** Variable-resolution speculative decoding that adapts to query complexity.

---

## Existing Infrastructure We Can Reuse

| Component | Location | Reuse |
|-----------|----------|-------|
| `eigenspace_alignment()` | spectral_hierarchy.rs | Eigenvalue decomposition for spectral gap detection |
| `top_k_eigenvectors()` | spectral_hierarchy.rs | Jacobi iteration for Laplacian eigenvectors |
| `haar_wavelet_basis()` | spectral_hierarchy.rs | Wavelet modes for boundary detection |
| `cauchy_interlacing_check()` | spectral_hierarchy.rs | Validates hierarchical eigenvalue structure |
| `simd_dot_f32()` | simd.rs | SIMD-accelerated tangent-space aggregation |
| `KgTriple` extraction | riir-engine | KG construction for domain embeddings |
| `NeuronShard` | riir-chain | Fixed 368B Pod with BLAKE3 commitment |
| `BeliefDrafter` MLP | katgpt-core | Latent dynamics prediction |
| `ConstraintPruner` trait | katgpt-core | Pluggable validation tiers |
| `LodestarPruner` | katgpt-core | Completion-distance budget pruning |

---

## Honest Assessment

### Strengths
1. **Principled theory**: Lipschitz guarantees, bounded approximation error, curvature-independent sharp constant
2. **Modelless**: No training required — pure spectral methods on pre-computed embeddings
3. **Default transfer**: Parameters work unchanged across HSBM and WordNet
4. **Efficient**: O(N·K²_eigs) with Lanczos, linear in N

### Risks
1. **Hyperbolic embedding quality**: Needs good Poincaré embeddings; flat/random KGs won't benefit
2. **Dense graph degradation**: Tree assumption breaks for cyclic KGs (§8 Limitations)
3. **Scalability**: Lanczos is practical to ~100K nodes; beyond needs Chebyshev/Nyström approximation
4. **Focus dependency**: Results depend on focus node x_0; multi-focus aggregation is open (Open Question 8)
5. **Novelty**: Not yet battle-tested on real GraphRAG systems (Open Question 6)

### What We'd Build (Modelless First)

Phase 1: `slod` feature flag in katgpt-core
- `SlodOperator` struct with heat kernel diffusion on kNN graph
- `BoundaryScan` with three-signal composite detection
- `SlodPruner` implementing `ConstraintPruner` that adapts validation depth to spectral boundaries

Phase 2: Integration with existing pruners
- Hyperbolic embedding of pruner domains (syntax, game, compiler)
- Automatic tier routing via SLoD boundaries
- SIMD-accelerated tangent-space Fréchet mean

Phase 3: Benchmark proof
- HSBM synthetic hierarchy recovery
- WordNet-equivalent taxonomy on our KG structures
- Before/after: SLoD-adaptive DDTree vs fixed-depth DDTree

---

## TL;DR

SLoD is a **modelless continuous-zoom operator for KG resolution** via spectral heat diffusion. It discovers abstraction boundaries from Laplacian spectral gaps without manual parameter tuning. Strong theoretical guarantees (Lipschitz coherence), validated on synthetic + real hierarchies, default parameters transfer across domains.

**GOAT verdict: GAIN for katgpt-rs (modelless).** Extends our existing `spectral_hierarchy` module to continuous KG resolution control. Key fusion: SLoD-adaptive DDTree budget selection, hyperbolic constraint graph routing, and variable-resolution BeliefDrafter. Build as `slod` feature flag, benchmark against fixed-depth DDTree baseline.

> **PASS-Redirects (synthesis):** Nakayama et al. [arXiv:2610.05816 "Level-of-Token Diffusion"] — the lattice-rectangular, generation-side cousin of SLoD's continuous zoom; its quadtree layout is a discrete special case of SLoD's continuous spectral zoom (no lattice needed in KG latent space), and its trained asymmetric-flow half is out of scope (no image/video diffusion in the stack). Verdict PASS; its one surviving modelless residue (Procrustes-fitted extent pooling) routed to `.issues/925_lot_fitted_pooling_routing_probe.md` on the attention-routing slot, not KG.
