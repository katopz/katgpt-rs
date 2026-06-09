# 207: ManifoldE — Point-to-Manifold Principle for Inference-Time Constraints

**Paper**: "From One Point to A Manifold: Knowledge Graph Embedding For Precise Link Prediction"
**Authors**: Xiao, Huang, Zhu (Tsinghua, 2016) | **arXiv**: 1512.04792
**Status**: Research | **Date**: 2026-06-09

---

## 1. Paper Summary

ManifoldE solves **imprecise link prediction** in KG embeddings by replacing point-wise geometric targets (TransE's `h+r=t`) with manifold-wise targets (sphere/hyperplane). Point-wise mapping creates an ill-posed algebraic system: T·d equations with (E+R)·d variables where T >> E+R, forcing multiple entities to compete for a single point. By relaxing each triple to a scalar manifold equation (`||h+r-t||² = D_r²`), the system becomes nearly well-posed. Results: +162.7% HITS@1 over TransE on FB15K, +215.7% on N-N relations.

The fundamental principle is transferable beyond KG: **replace point-wise decisions with manifold-wise scoring wherever multiple candidates compete for a single target**.

---

## 2. Core Ideas Distilled

1. **Ill-posed algebraic system**: Translation-based models (h+r=t) produce d equations per triple. With T >> E+R, the system is massively over-determined → unstable solutions.
2. **Manifold relaxation**: Each triple → one scalar equation (`||M(h,r,t) - D_r²||² = 0`). Reduces equations by factor d, making system nearly well-posed when d ≥ T/(E+R).
3. **Sphere manifold**: `||h + r - t||² = D_r²` — tails lie on a sphere, not a point. Radius D_r is relation-specific.
4. **Hyperplane manifold**: `|(h + r_head)^T(t + r_tail)| = D_r²` — two hyperplanes intersect more often than two spheres → better for multi-relational composition.
5. **Kernel trick to RKHS**: Map embeddings to Reproducing Kernel Hilbert Space for more expressive manifold shapes (Gaussian, Polynomial kernels).
6. **Relation-specific geometry**: Different relations need different D_r → adaptive manifold radius per semantic context.

---

## 3. Existing Infrastructure

| Component | Location / Feature Gate | Relevance |
|-----------|------------------------|-----------|
| `ManifoldResidual` trait | `deep_manifold` | L2ResidualScorer, KlResidualScorer — distance to fixed-point manifold |
| `ResidualRelevanceScorer` | `deep_manifold` | Blends residual with relevance score |
| `BoundaryAlignment` trait | `federation` | KlBoundaryAligner — federated KL coupling between domain experts |
| `ConstraintPruner` trait | core | `is_valid()` binary, `batch_is_valid()` batch — currently point-wise |
| `ScreeningPruner` trait | core | `relevance()` scalar score per candidate — linear scoring |
| BFCP Region partitioning | `bfcf_lfu_shard` (Plan 218) | ~50 manifold regions in logit space, LFU cache, Hot/Warm/Cold tiers |
| KG Latent Octree | Research 196 | Ternary octree over KG embedding space, WASM pods |
| `BeliefDrafter` | Plan 217 | Latent dynamics MLP for self-speculative decoding |
| `NeuronShard` | core | 368B fixed Pod: style_weights[64] + hla_moments[8] + BLAKE3 |

---

## 4. Creative Fusion Ideas

### Fusion 1: ManifoldPruner — Soft Validity Scoring

**What**: Replace binary `is_valid()` with continuous `manifold_score()` ∈ [0, 1].

**How it maps**: ManifoldE's `||h+r-t||² = D_r²` → `sigmoid(-||metric(h,r,t) - D_r²|| / temperature)`. Tokens inside manifold ≈ 1.0, outside ≈ 0.0, boundary ≈ 0.5. Temperature = manifold radius controller (analogous to D_r). Modelless — the "manifold" is the pruner's decision boundary reinterpreted geometrically.

**Trait extension**:
```rust
// ConstraintPruner trait — backward compatible addition
fn manifold_score(&self, depth: u32, token_idx: u32, prefix: &[TokenId]) -> f32 {
    // Default: binary → {0.0, 1.0}
    if self.is_valid(depth, token_idx, prefix) { 1.0 } else { 0.0 }
}
```

**Expected gain**: MEDIUM — recovers boundary tokens currently discarded by hard accept/reject. Risk: LOW — backward compatible.

### Fusion 2: ManifoldExpansion — Multi-Child DDTree Growth

**What**: At each DDTree depth, expand ALL candidates above manifold threshold, not just top-1.

**How it maps**: ManifoldE shows point-wise creates ill-posed systems. DDTree expanding to one child per branch is point-wise. When base model logits are uncertain, forcing a single path creates instability. Expanding all above-threshold candidates is nearly well-posed — each depth adds equations proportional to candidate count. Already partially enabled by `batch_is_valid`; novel part is ManifoldE's sphere equation to set threshold dynamically per-context.

**Expected gain**: LOW-MEDIUM — marginal over existing batch expansion. Risk: LOW.

### Fusion 3: HyperplanePruner — Geometric Constraint Intersection

**What**: Each pruner defines a half-space `{tokens : (h + r_head)^T(token + r_tail) >= D_r}`. Composition = intersection of half-spaces, not boolean AND.

**How it maps**: ManifoldE's key insight: two hyperplanes intersect more often than two spheres. When multiple pruners compose (SynPruner + DomainPruner + GamePruner), boolean AND of point-wise checks is geometrically restrictive. Half-space intersection yields a convex polytope of valid tokens — more solutions, fewer false negatives.

```rust
// Optional trait method — backward compatible
fn constraint_vector(&self, ctx: &PruneContext) -> Option<(Vec<f32>, f32)> {
    None // default: fall back to is_valid()
}
// Composition: intersect all non-None constraint vectors
```

**Expected gain**: HIGH — fundamentally better multi-constraint composition. This is the ConstraintPruner moat improvement the commercial strategy depends on. Risk: MEDIUM — needs SIMD for perf, new trait methods.

### Fusion 4: Kernel-Tricked Relevance Scoring

**What**: Replace linear `dot(query, candidate)` with `K(query, candidate)` using Gaussian or Polynomial kernel.

**How it maps**: ManifoldE maps to RKHS for expressive manifolds. Our `ScreeningPruner::relevance()` is linear. Gaussian kernel `exp(-||q-c||²/σ²)` = natural distance-to-manifold scoring. σ = manifold width (analogous to D_r). Modelless — just a scoring function change, no training.

**SIMD-friendly**: Gaussian kernel is element-wise f32 ops.

```rust
enum KernelKind {
    Linear,                           // dot(q, c) — current
    Gaussian { sigma: f32 },          // exp(-||q-c||²/σ²)
    Polynomial { degree: f32, c: f32 }, // (dot(q,c) + c)^degree
}
```

**Expected gain**: MEDIUM — non-linear relevance catches what linear misses. Risk: LOW.

### Fusion 5: Region-Level Manifold Radius Adaptation

**What**: BFCP regions get frequency-adapted manifold radius D_r.

**How it maps**: ManifoldE's relation-specific D_r → our region-specific radius. Hot regions (high LFU frequency): large D_r (wide manifold, many valid tokens). Cold regions: small D_r (tight manifold, conservative). Self-adapting via existing LFU frequency — no training.

**Expected gain**: LOW — minor parametric adaptation. Risk: LOW.

---

## 5. Verdict (per 003 Strategy)

### Engine / Fuel Split

| Layer | Ownership | Fusion Fit |
|-------|-----------|------------|
| **Engine** (MIT) | katgpt-rs | ManifoldPruner, HyperplanePruner, KernelRelevance, BFCP radius |
| **Fuel** (commercial) | domain configs | Manifold parameters (D_r, σ, temperature) per domain |

All fusions are **modelless** — geometric transformations of existing signals. No `lora.bin`, no training, pure inference-time geometry.

### GOAT / Gain Assessment

| Fusion | Expected Gain | Risk | Verdict | Priority |
|--------|--------------|------|---------|----------|
| 3. HyperplanePruner | HIGH | MEDIUM | **GAIN** | P0 — highest value |
| 1. ManifoldPruner (soft validity) | MEDIUM | LOW | **GAIN** | P1 |
| 4. Kernel-Tricked Relevance | MEDIUM | LOW | **GAIN** | P1 |
| 2. ManifoldExpansion (multi-child) | LOW-MEDIUM | LOW | **GAIN** | P2 — marginal over batch |
| 5. Region Radius Adaptation | LOW | LOW | **GAIN** | P2 — minor parametric |

### Feature Gate

```
manifold_pruner   // opt-in, GOAT-gate for default promotion
```

### Implementation Sequence

1. Add `manifold_score()` default method to `ConstraintPruner` trait (backward compatible)
2. Add `constraint_vector()` optional method to `ConstraintPruner` trait
3. Implement `HyperplanePruner` that intersects constraint vectors
4. Add `KernelKind` enum to `ScreeningPruner` with Gaussian/Polynomial variants
5. Wire BFCP region radius to LFU frequency
6. Benchmark: `cargo bench --features manifold_pruner` vs baseline

---

## 6. Cross-Repo Alignment

| Target | What | Why |
|--------|------|-----|
| `riir-ai` | `KernelKind` enum + kernel scoring functions | Reusable inference-time scoring — no dependency on katgpt-rs specifics |
| `seal-online-remaster` | Manifold-wise NPC behavior validation | Replace binary "is_action_valid" with soft manifold scoring for smoother AI |
| `katgpt-rs` | Trait extensions, HyperplanePruner, BFCP radius wiring | Core implementation |

---

## 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| SIMD perf regression from soft scoring | MEDIUM | Benchmark before/after; keep binary fast-path as default |
| `constraint_vector()` allocation in hot loop | HIGH | Pre-allocate per-pruner, return `&[f32]` slice, not `Vec<f32>` |
| Gaussian kernel σ tuning per domain | LOW | Start with σ=1.0, make configurable in domain config (fuel layer) |
| HyperplanePruner composition blowup with many pruners | MEDIUM | Cap intersection at N pruners; fall back to boolean AND for remaining |
| Trait method proliferation | LOW | Both new methods have sensible defaults; zero cost if not overridden |

---

## 8. TL;DR

**ManifoldE's core principle**: point-wise → manifold-wise. Instead of binary accept/reject (TransE's single point), score distance to a validity manifold (sphere/hyperplane). This makes the algebraic system well-posed and dramatically improves precision.

**Highest-value fusion**: HyperplanePruner (Fusion 3) — compose multiple constraint pruners as half-space intersection instead of boolean AND. Geometrically yields more valid candidates, fewer false negatives. Directly strengthens the ConstraintPruner moat that the commercial strategy depends on.

**All fusions are modelless**: geometric reinterpretations of existing signals, no training, no lora.bin, pure inference-time. Feature-gate as `manifold_pruner`, benchmark against baseline, promote to default if GOAT-gate passes.

**Promote to Plan**: Yes. Create Plan 219 with P0 = HyperplanePruner, P1 = ManifoldPruner + KernelRelevance.
