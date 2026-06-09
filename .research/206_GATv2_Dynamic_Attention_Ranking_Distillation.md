# GATv2 Dynamic Attention Ranking — Dual-Project Distillation Verdict

**Source:** [How Attentive Are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) — Brody et al., ICLR 2022
**Date:** 2026-06-09
**Related Plans:** 021 (ScreeningPruner), 030 (BanditPruner), 079 (BtRank), 197 (DominoPruner), 243 (Polytope LoRA Router)
**Cross-Project:** riir-ai `.research/051_dMoE_Block_Level_LoRA_Routing.md`

---

## 1. Paper Core Insight (3 Sentences)

Standard GAT computes **static attention**: the ranking (argsort) of attention scores across keys is identical for every query node, because the linear projection `aᵀ·W·hⱼ` creates a global key ranking independent of the query. GATv2 fixes this by reordering composition — concat inputs first, then apply a shared weight matrix, then nonlinearity, then attention vector — which produces a true MLP over the (query, key) pair. This ordering change is the difference between a model that **cannot** discriminate contexts (GAT) and one that can (GATv2), with GATv2 outperforming GAT by 1.4–11.5% across 12 benchmarks.

---

## 2. Principle Extraction: Ordering of Composition

The deeper pattern is **not** about graphs — it's about when composition order matters:

```
Static (GAT):    score = f(linear(query), linear(key))
                  → key ranking is independent of query

Dynamic (GATv2): score = f(linear(concat(query, key)))
                  → key ranking is conditioned on query
```

The **litmus test**: if you permute the query while holding keys fixed, does the argsort of scores change?

- **YES** → dynamic (expressive)
- **NO** → static (limited)

This test applies anywhere you have a scoring function over pairs.

---

## 3. Mapping to katgpt-rs (Modelless): Static Ranking Detection in Pruners

### 3.1 Current State Analysis

The `ScreeningPruner::relevance()` signature already accepts `parent_tokens`:

```rust
fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
```

**However**, many implementations effectively ignore `parent_tokens`:

| Pruner | Uses `parent_tokens`? | Static? |
|--------|----------------------|---------|
| `NoScreeningPruner` | No (returns 1.0) | Trivially static |
| `BinaryScreeningPruner<P>` | Delegates to `P::is_valid()` | Depends on P |
| `BanditPruner` (base) | No — Q-values indexed by arm only | **Static** |
| `DominoPruner::causal_correction` | Yes — explicit prefix conditioning | **Dynamic** |
| `NarrowingPruner` | Yes — parent determines valid set | **Dynamic** |

The BanditPruner is the most important case: it scores tokens via per-arm Q-values (`q_values[arm]`), which are **independent of parent context**. Two different parent prefixes that both query arm 3 get the same relevance score. This is exactly the GAT static attention problem.

### 3.2 Fusion Idea: `DynamicRankPruner` — Diagnostic + Correction

Not a new pruner trait, but a **wrapper** that detects static ranking and applies context-dependent correction:

```rust
/// Wrapper that detects and corrects static ranking in ScreeningPruner.
///
/// GATv2 insight: if argsort(relevance(token_j)) is invariant across
/// different parent contexts, the inner pruner is "static" — it cannot
/// discriminate between contexts. This wrapper:
/// 1. Diagnoses static ranking by comparing argsort across sampled parents
/// 2. If static: applies a context-dependent correction (DominoPruner-style prefix lookup)
/// 3. If dynamic: passes through unchanged (zero overhead)
///
/// Feature-gated under `dynamic_rank`.
pub struct DynamicRankPruner<P: ScreeningPruner> {
    inner: P,
    /// Detected static ranking: argsort of relevance for a canonical parent.
    /// If None, not yet diagnosed. If Some, the static ranking to correct.
    static_ranking: Option<Vec<usize>>,
    /// Correction table: prefix hash → score adjustment vector.
    /// Same pattern as DominoPruner::PrefixCorrectionTable.
    correction_table: Papaya<HashMap<u64, Vec<f32>>>,
    /// Diagnosis sample count before declaring static/dynamic.
    diagnosis_rounds: usize,
    /// Whether the inner pruner has been diagnosed.
    diagnosed: AtomicBool,
}
```

**Why this is NOT just "use parent_tokens more"**: The GATv2 insight is that you can have the *parameter* for context conditioning (the `parent_tokens` argument) but still produce static rankings because your *implementation* doesn't actually condition on it. The wrapper detects this programmatically — it's a **diagnostic**, not just an API.

### 3.3 Connection to Existing Infrastructure

This maps cleanly to existing patterns:

| GATv2 Concept | katgpt-rs Analog | Status |
|---------------|------------------|--------|
| Static attention detection | BanditPruner Q-values (per-arm, no parent) | **Exists** (the problem) |
| Dynamic attention fix | DominoPruner::causal_correction (prefix-conditioned) | **Exists** (the solution pattern) |
| Diagnostic (argsort comparison) | QuestBench entropy scoring | **Exists** (the measurement tool) |
| Nonlinear pair transform | AbsorbCompress sigmoid gating | **Exists** (the activation function) |

The key observation: **katgpt-rs already has all the building blocks** — the gap is in detecting when BanditPruner is statically ranking and automatically bridging to DominoPruner-style prefix correction.

### 3.4 What This Actually Changes

For `BanditPruner::relevance()`:
```
Current:  relevance = q_values[token_idx]           // Static: same for all parents
Fixed:    relevance = q_values[token_idx] + correction(parent_hash, token_idx)
          // Dynamic: correction is nonzero only when static ranking detected
```

The correction is zero-cost when the pruner is already dynamic (diagnosis confirms, no table lookup). It's only nonzero when static ranking is detected, and the correction is a prefix-aware delta from the existing `PrefixCorrectionTable` pattern.

---

## 4. Mapping to riir-ai (Model-Based): Dynamic LoRA Attention Routing

### 4.1 Current State Analysis

riir-ai has two routing mechanisms:

1. **`PolytopeLoraRouter`** (Plan 243): routes via BFCP polytope membership
   ```rust
   fn route(&self, state: &[f32]) -> (usize, f32)
   ```
   Routes based on **input state alone** — this is static routing in GAT terms.

2. **`FrameExpertCoreset`** (Plan 203): frame-level LoRA expert coreset selection
   Also routes based on input features alone.

Both are GAT-style: `route(input)`, not `route(input, target)`.

### 4.2 The GATv2 Insight for LoRA Routing

In MoE / multi-LoRA settings, the routing decision should condition on **both** the source context AND the target domain:

```
Static (current):  route(source_state) → adapter_idx
                   → Same state always maps to same adapter, regardless of what we're optimizing for

Dynamic (proposed): route(source_state, target_signal) → adapter_idx
                    → Routing changes based on what we're trying to achieve
```

**Concrete example for game AI:**
- NPC in "combat zone" with health=50% → currently always routes to "combat LoRA"
- But if the target is "survival" (low health, enemies nearby) vs "aggression" (high health, enemies nearby) → should route differently
- The **same state** maps to different LoRA adapters depending on the objective

### 4.3 Fusion Idea: `DynamicPairRouter` trait

```rust
/// LoRA router that conditions on both source state AND target signal.
///
/// GATv2 principle: routing based on input alone is static (same global ranking
/// of experts regardless of objective). Dynamic routing scores the (state, target)
/// PAIR through a nonlinear transform, producing context-dependent expert selection.
///
/// This is NOT PolytopeRouter with more dimensions — it's a fundamentally different
/// scoring function that produces different expert rankings for different targets
/// given the same input state.
pub trait DynamicPairRouter {
    /// Route based on source state AND target signal.
    /// Returns (adapter_idx, confidence).
    ///
    /// The GATv2 litmus test: for fixed `state`, changing `target_signal`
    /// should change the argsort of adapter scores.
    fn route_dynamic(
        &self,
        source_state: &[f32],
        target_signal: &[f32],
    ) -> (usize, f32);

    /// Number of managed adapters.
    fn adapter_count(&self) -> usize;

    /// Verify dynamic property: route_dynamic(s, t1) != route_dynamic(s, t2)
    /// for at least one s, t1, t2 triple.
    fn verify_dynamic(&self, test_states: &[Vec<f32>], test_targets: &[Vec<f32>]) -> DynamicReport;
}
```

### 4.4 Practical LoRA Implementation

For LoRA adapters `W_lora = W_base + α·B·A`:

```
Static routing (current):
  expert_idx = argmax_i score_i(state)           // PolytopeLoraRouter
  output = W_base · x + α · B_{expert} · A_{expert} · x

Dynamic routing (proposed):
  // Score the (state, target) PAIR
  pair_repr = LeakyReLU(W_pair · [state ‖ target_signal])
  expert_idx = argmax_i aᵀ · pair_repr_i
  output = W_base · x + α · B_{expert(state,target)} · A_{expert(state,target)} · x
```

This is LoRA-only (no full training). The routing weights `W_pair` and `a` are small additional LoRA parameters trained alongside the main adapters.

---

## 5. GOAT Analysis

### 5.1 katgpt-rs: DynamicRankPruner

| Criterion | Assessment | Reasoning |
|-----------|-----------|-----------|
| **Implementation Complexity** | **Low** (~200 LOC) | Wrapper pattern already exists (BinaryScreeningPruner, BanditPruner wrapping inner). Diagnosis is argsort comparison. Correction table reuses PrefixCorrectionTable pattern. No new dependencies. |
| **Performance Gain Potential** | **Medium** (5–15% acceptance rate) | Paper shows GATv2 beats GAT by 1.4–11.5%. Our BanditPruner has known static ranking — fixing it should improve token selection quality. But: our negative results on reward modulation (SDAR ELO 954≈955) suggest the ceiling may be lower than the paper's graph-domain results. |
| **Alignment with Strategy** | **High** | Modelless, inference-time only. Feature-gated (`dynamic_rank`). Diagnoses existing pruners (engine) without requiring riir-ai (fuel). Clean engine/fuel split. |
| **Risk** | **Low** | Wrapper is additive. Zero overhead when pruner is already dynamic. Diagnosis is read-only observation. Correction table is opt-in. Can be toggled off via feature gate. |

**GOAT Score: 6/10** — Low cost, clean alignment, but gain potential is uncertain given our string of near-zero modelless improvements (SDAR, GFlowNet). The diagnostic value alone (knowing which pruners are static) may be worth the implementation.

### 5.2 riir-ai: DynamicPairRouter

| Criterion | Assessment | Reasoning |
|-----------|-----------|-----------|
| **Implementation Complexity** | **Medium** (~500 LOC) | New trait, new router implementation, training integration with existing GZeroLoop. Requires extending `PolytopeLoraRouter` or building alongside it. Small additional LoRA parameters for routing weights. |
| **Performance Gain Potential** | **High** (10–25% task accuracy) | Game AI has clear objectives (survival, aggression, economy). Static routing means the same NPC state always uses the same LoRA regardless of strategic goal. Dynamic routing should meaningfully improve multi-objective game AI. |
| **Alignment with Strategy** | **Perfect** | Model-based (training time). riir-ai's commercial value proposition is intelligent game AI. LoRA-only (constraint satisfied). Extends existing PolytopeLoraRouter rather than replacing it. |
| **Risk** | **Medium** | Training-time change means regression potential is higher. Requires GPU testing. May interact with existing curvature curriculum (Plan 203). New routing weights need training budget. |

**GOAT Score: 7/10** — Higher potential gain for riir-ai's commercial value, but also higher cost and risk. The game AI domain is exactly where GATv2's insight (context-dependent ranking) should shine — NPC behavior should change based on objectives, not just current state.

---

## 6. Verdict

### ✅ GOAT — with Conditions

**Both projects should proceed, but with different priority and scope.**

#### Priority 1: katgpt-rs Diagnostic (Low Risk, Quick Value)

The `DynamicRankPruner` wrapper is worth implementing primarily as a **diagnostic tool**:

1. It tells us which pruners are statically ranking (BanditPruner almost certainly is)
2. It provides the correction mechanism as a follow-up if the diagnostic reveals significant static behavior
3. The diagnostic itself is the GOAT proof: compare acceptance rates with/without correction on the same pruner

**GOAT gate**: `dynamic_rank` feature flag. Proof = acceptance rate delta on bomber/go benchmarks with BanditPruner wrapped in DynamicRankPruner vs unwrapped.

#### Priority 2: riir-ai DynamicPairRouter (High Potential, Needs Proof)

The `DynamicPairRouter` should be implemented as an extension to `PolytopeLoraRouter`:

1. Start with the `verify_dynamic()` method — prove that current routing is actually static
2. Implement pair routing with small additional LoRA parameters
3. Benchmark on FFT Tactics Arena (existing infrastructure) with survival vs aggression objectives

**GOAT gate**: `dynamic_pair_routing` feature flag. Proof = game win rate with dynamic routing vs static routing on the same NPC.

#### What NOT to Do

- **Do NOT** replace existing pruners — the wrapper pattern is additive
- **Do NOT** change the `ScreeningPruner` trait signature — it already has `parent_tokens`
- **Do NOT** make dynamic routing the default — feature-gate both, prove gain first
- **Do NOT** implement the full GATv2 architecture — we're distilling the *principle*, not the mechanism

---

## 7. Implementation Tasks

### katgpt-rs (Plan 232 — `dynamic_rank` feature)

- [ ] **T1: Static ranking diagnostic** — function that takes `dyn ScreeningPruner`, samples N parent contexts, computes argsort of relevance for each, checks if argsort is invariant. ~50 LOC.
- [ ] **T2: `DynamicRankPruner<P>` wrapper** — wraps any `ScreeningPruner`, runs diagnostic on first call, applies prefix-correction if static detected. ~100 LOC.
- [ ] **T3: Integration with `BanditPruner`** — add `with_dynamic_rank()` builder method. Forward `relevance()` through wrapper. ~30 LOC.
- [ ] **T4: GOAT proof test** — `tests/bench_dynamic_rank_goat.rs`. Compare BanditPruner acceptance rate with/without wrapper on bomber arena. Must show ≥2% improvement. ~80 LOC.
- [ ] **T5: Feature gate** — add `dynamic_rank = []` to Cargo.toml, default-off until proof passes.

### riir-ai (Plan 090 — `dynamic_pair_routing` feature)

- [ ] **T1: `DynamicPairRouter` trait** — define trait with `route_dynamic(source, target)` and `verify_dynamic()`. ~30 LOC.
- [ ] **T2: `DynamicPolytopeLoraRouter`** — extend `PolytopeLoraRouter` with target-conditioned scoring. `pair_repr = LeakyReLU(W_pair · [state ‖ target])`. ~200 LOC.
- [ ] **T3: Training integration** — add pair routing to GZeroLoop. Small LoRA params for W_pair trained alongside main adapters. ~150 LOC.
- [ ] **T4: Dynamic verification test** — prove that current PolytopeLoraRouter is static (argsort invariant to target). Then prove DynamicPolytopeLoraRouter is dynamic (argsort changes with target). ~100 LOC.
- [ ] **T5: GOAT proof benchmark** — FFT Tactics Arena with survival vs aggression objectives. Win rate delta with dynamic routing vs static. ~80 LOC.
- [ ] **T6: Feature gate** — `polytope_lora_routing` feature already exists. Add `dynamic_pair_routing` sub-feature.

---

## 8. Relationship to Existing Plans

| Plan | Status | GATv2 Connection |
|------|--------|-----------------|
| 021 ScreeningPruner | ✅ Done | Base trait that DynamicRankPruner wraps |
| 030 BanditPruner | ✅ Done | **Primary target** — BanditPruner Q-values are static ranking |
| 079 BtRank | ✅ GOAT | BT ranking is pairwise (already dynamic in GATv2 terms) |
| 197 DominoPruner | ✅ GOAT | DominoPruner is already dynamic — the correction pattern to reuse |
| 243 Polytope Router | ✅ Done | **Base router** for DynamicPairRouter — G1 proved it's static |
| 203 FrameExpertCoreset | Done | Frame-level coreset — could also benefit from dynamic routing |
| 221 KG Confidence Bridge | ✅ Done | **Prerequisite for DynamicPairRouter** — KG confidence now flows from KgEmbedding → SenseModule.confidence → project() scaling. Same HLA + different confidence → different sense ranking (GATv2 dynamic property proven in T5). |

---

## 9. Honest Caveats

1. **Negative result history**: Our modelless improvements (SDAR ELO 954≈955, GFlowNet no gain) suggest that inference-time modifications to the selection mechanism may have limited impact when the base model quality is the bottleneck. GATv2's gains were in training-time attention, not post-hoc scoring.

2. **Domain mismatch**: GATv2 was tested on node classification, link prediction, and graph property prediction. Our domain (speculative decoding token selection) is a different optimization landscape. The principle transfers, but the magnitude of gain may not.

3. **BanditPruner may already be dynamic enough**: While Q-values are per-arm (static), the `soft_route` blending across arms introduces some context-dependence. The diagnostic will reveal whether this is sufficient.

4. **LoRA routing gain depends on adapter diversity**: If all LoRA adapters learn similar behaviors, dynamic routing has nothing to discriminate. The gain requires meaningful adapter specialization, which depends on training quality.

5. **KG confidence bridge is infrastructure, not payoff**: The confidence-weighted projection (`confidence * sigmoid(dot)`) is a static scaling — the same triple always has the same confidence. The GATv2 dynamic property emerges when DIFFERENT zones/contexts have different confidence vectors, flipping the argsort. This was proven in bench_221 T5. **DynamicPairRouter (Plan 260 T1-T4+T7) now proves the dynamic property at the LoRA routing layer** — G1 confirms static base, G2 confirms dynamic extension.

---

## 10. Acceptance Rate Benchmark Results (2026-06-09)

**Verdict: NO GAIN — Diagnostic only, do NOT promote to default.**

| Metric | BanditPruner Baseline | DynamicRankPruner Wrapped | Delta |
|--------|----------------------|---------------------------|-------|
| Tree nodes | 16,000 | 16,000 | +0 |
| Accepted tokens | 1,212 | 1,211 | -1 |
| **Acceptance rate** | **7.58%** | **7.57%** | **-0.01pp** |
| Total reward | 680.1 | 678.2 | -1.9 |

**Root cause:** Marginals dominate tree structure, not pruner scores. The `0.01` correction learning rate is too small to shift which branches survive. Static detection is correct, but correction is too weak.

### Promotion Decision

| Gate | Criterion | Result |
|------|-----------|--------|
| Detection GOAT | Diagnostic identifies static pruners | ✅ PASS (5/5) |
| Acceptance rate gain | ≥2% improvement | ❌ FAIL (-0.01pp) |
| Safe (no regression) | ≤10% degradation | ✅ PASS (-0.08% relative) |

**Outcome: DEMOTE to diagnostic tool.** Keep `dynamic_rank` feature gate, default-OFF. Do NOT add to `full` or `default`. The diagnostic value (knowing which pruners are static) is real, but the correction mechanism does not produce measurable gain.

---

## TL;DR

**GATv2's insight**: composition order (linear-then-nonlinearity vs nonlinearity-then-linear) determines whether scoring is static (same ranking for all queries) or dynamic (ranking conditioned on query). Apply the diagnostic: does argsort change when you change the query?

**For katgpt-rs**: BanditPruner is likely static (per-arm Q-values, no parent conditioning). Wrap it with a diagnostic that detects this and applies DominoPruner-style prefix correction. Low cost (~200 LOC), uncertain gain (our modelless track record is mixed), high diagnostic value.

**For riir-ai**: PolytopeLoraRouter routes on input alone (static). Add target-conditioned pair routing via small LoRA parameters. Higher potential gain (game AI has clear multi-objective structure), but higher cost and risk.

**Verdict: GOAT** — both are worth implementing behind feature gates. katgpt-rs diagnostic first (quick win / learning), riir-ai pair routing second (higher ceiling). Neither changes existing APIs.

KG confidence bridge (Plan 221 T13) closes the extraction→inference gap — confidence now flows to projection output. Unblocks DynamicPairRouter.

**DynamicPairRouter (Plan 260 G1-G2 ✅)**: `PolytopeLoraRouter` confirmed static (G1). `DynamicPolytopeLoraRouter` confirmed dynamic — 3 distinct argsort patterns for same state + different objectives (G2). Xavier-initialized weights also produce dynamic ranking (G2b: 3/4 states). Feature-gated `dynamic_pair_routing`, default-OFF. T5 (training integration), T6 (arena benchmark), T8 (CPU/GPU auto-route) pending.
