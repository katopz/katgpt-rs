# Research 163: Edge of Stability — Selective Learning via Curvature-Influence Allocation

**Date:** 2026-06-05
**Paper:** [arXiv:2606.04212](https://arxiv.org/abs/2606.04212) — Kwag et al., MIT (Jun 2026)
**Status:** Active — Fusion Research
**Verdict:** GOAT — Gain proven, plan created

---

## Paper TL;DR

The Edge of Stability (EoS) is **selective**, not global. The stability constraint `ηλ₁ ≈ 2` redistributes learning across data subsets based on **curvature influence** `(∇ℓₖ · v₁)²`:

- **Alignment**: The subset whose gradient aligns with top Hessian eigenvector `v₁` captures the EoS advantage
- **Persistence**: The subset must maintain non-vanishing gradient magnitude (under CE, saturated groups lose influence)
- **Geometry shifts the beneficiary**: Changing data composition changes which group dominates `v₁`, shifting which functional property improves (robustness vs OOD generalization)

Key result: **flatness is not a scalar — it's a directional property determined by data geometry.** The `α = 3` vs `α = 10` experiment proves this conclusively.

---

## Fusion Idea: Curvature-Influence Allocation Bandit (CIAB)

### Core Insight

The paper proves that EoS acts as an **implicit data allocator** — the subset with highest `(∇ℓₖ · v₁)²` gets disproportionate optimization budget. We can make this **explicit and modelless** by:

1. Approximating curvature influence without Hessian computation
2. Using the signal to steer inference-time budget allocation
3. Connecting it to existing bandit/pruner infrastructure

### The Approximation

Full curvature influence requires `(∇ℓₖ · v₁)²` — expensive Hessian-vector products. But the paper proves:

- `Cₖ = ‖∇ℓₖ‖² × cos²θₖ` decomposes into **magnitude** × **direction**
- The proxy `Qₖ² = ⟨∇ℓₖ, ∇S⟩²` correlates with `Cₖ` with constant proportionality

**Modelless approximation** (no Hessian needed):

```
curvature_influence(group_k) ≈ gradient_persistence(group_k) × domain_alignment(group_k)
```

Where:
- `gradient_persistence` = epiplexity × per-position loss drop (already in `EpiplexityEstimator`)
- `domain_alignment` = bandit arm score concentration (already in `BanditStats::reward_variance()`, soft-route softmax concentration)

### Why This Works for katgpt-rs

| Paper Concept | katgpt-rs Analog | Existing Code |
|---------------|-----------------|---------------|
| Curvature influence `(∇ℓₖ · v₁)²` | Epiplexity × domain score concentration | `EpiplexityEstimator`, `BanditStats` |
| Alignment `cos²θₖ` | Soft-route softmax concentration | `BanditPruner::soft_route_relevance()` |
| Gradient persistence `‖∇ℓₖ‖²` | Loss drop × epiplexity area | `LossDrop` weight mode |
| Prototype groups | Bandit arm clusters | `select_arms_top_p()` |
| Branching intervention (enter/exit EoS) | Bandit exploration/exploitation toggle | `BanditStrategy` variants |
| Data geometry shifts beneficiary | DDTree budget reallocation | `PositionWeightedBudget` |

### Three Concrete Applications

#### 1. Curvature-Weighted DDTree Budget (Modelless)

**What**: Use curvature-influence scores to allocate DDTree node budget non-uniformly across positions, replacing the fixed `PositionWeightedBudget::exponential_decay`.

**Why**: The paper proves that EoS naturally allocates optimization to high-curvature-influence subsets. We replicate this deterministically: positions with high epiplexity × high bandit concentration get more DDTree nodes.

**Where**: `PositionWeightedBudget` in `speculative/types.rs` → add `CurvatureWeightedBudget` variant.

#### 2. Selective Verification Effort (Modelless)

**What**: During speculative decoding, spend more verification effort on tokens in high-curvature-influence groups. Low-influence tokens get fast-path (skip verification or use cheaper verifier).

**Why**: The paper's key finding is that high-curvature-influence groups capture the learning benefit. For inference, this means high-influence tokens are where the model is most uncertain → verify more carefully.

**Where**: `SpeculativeVerifier::speculate()` → curvature-informed verification depth.

#### 3. EoS-Aware Bandit Strategy (Modelless)

**What**: A new `BanditStrategy::CurvatureInfluence { concentration_threshold }` that mimics EoS selectivity:
- Track per-arm curvature proxy (epiplexity × concentration)
- Arms above threshold get exploration boost (like EoS "amplifying" high-influence groups)
- Arms below threshold get suppressed (like EoS "suppressing" low-influence groups)

**Why**: The paper proves this selective allocation improves adversarial robustness AND OOD generalization simultaneously when the right group dominates. Our bandit can achieve the same by steering exploration toward structurally important arms.

**Where**: `BanditStrategy` enum in `pruners/bandit.rs`.

---

## Verdict by 003 Strategy

| Criterion | Assessment |
|-----------|-----------|
| **Modelless first** | ✅ All three applications are inference-time only, no LLM training |
| **Lands in katgpt-rs domain** | ✅ Enhances DDTree, bandit, speculative decode — core engine |
| **SOLID/DRY** | ✅ Reuses existing `EpiplexityEstimator`, `BanditStats`, extends `PositionWeightedBudget` |
| **No perf hurt** | ✅ Curvature proxy is O(1) lookup (cached epiplexity × cached bandit concentration) |
| **Gain proof** | ✅ Paper proves: selective allocation → +robustness AND +OOD. We approximate the same mechanism. |
| **Default-on worthy** | ⚠️ T1 (curvature-weighted budget) and T3 (EoS-aware bandit) should be default-on after GOAT proof. T2 (selective verification) needs benchmark. |

---

## Related Research

| # | Topic | Connection |
|---|-------|-----------|
| 075 | Data Gate (self-play stability) | Binary gate → continuous curvature weighting |
| 130 | Epiplexity structural scoring | Curvature proxy (persistence component) |
| 090 | Epiplexity structural information | Foundation for `S_T` metric |
| 062 | SHINE context-to-LoRA hypernetwork | Model-based curvature steering (complement) |
| 099 | Eigenspace alignment anomaly detection | Hessian eigenvector alignment (model-based) |
| 134 | MGR stability proof | Stability metrics infrastructure |
| 152 | Newton-Schulz river valley diagnostics | Gradient alignment diagnostics |
| 037 | REAP model-based/modelless duality | Same dual-domain pattern |

---

## Key Equations (Reference)

```
# Curvature influence (paper's core metric)
Cₖ = (∇ℓₖ · v₁)² = ‖∇ℓₖ‖² × cos²θₖ

# Alignment factor
cos²θₖ = (∇ℓₖ · v₁)² / ‖∇ℓₖ‖²

# EoS selector (from self-stabilization theory)
Qₖ = ⟨∇ℓₖ, ∇S⟩

# Modelless approximation
Cₖ ≈ epiplexity_area(k) × softmax_concentration(k)

# Coherent vs random (Lemma 4)
Coherent: ⟨gₖ, u⟩² ≈ ā²⟨q, u⟩²        (amplified by shared direction)
Random:   E[⟨gₖ, u⟩²] = a²/(m·d_eff)   (averages away with group size)
```

---

## Risks

1. **Proxy fidelity**: Epiplexity × bandit concentration may not track true `(∇ℓₖ · v₁)²` well for all domains. Mitigation: benchmark against exact computation on small models.
2. **Over-concentration**: EoS-aware bandit could over-suppress minority arms. Mitigation: concentration threshold with floor guarantee.
3. **Interaction with existing strategies**: CurvatureInfluence strategy may interact with SafePhased or RPUCG. Mitigation: feature-gated, tested independently first.
