# Plan 183: Curvature-Influence Allocation Bandit (CIAB) — Modelless

> **Research:** 163 (EoS Selective Learning — Curvature Allocation)
> **Paper:** [arXiv:2606.04212](https://arxiv.org/abs/2606.04212) — Kwag et al., Jun 2026
> **Depends:** Plan 130 (Epiplexity ✅), Plan 030 (Bandit ✅), Plan 131 (SpecHop ✅)
> **Feature Gate:** `curvature_alloc = ["bandit", "epiplexity"]`
> **Status:** ✅ Complete — GOAT 6/6 — Default ON
> **Default-on:** ✅ Yes — GOAT proof passed, promoted to default

---

## Summary

Implement curvature-influence allocation inspired by the EoS selective learning paper. The key insight: EoS implicitly allocates optimization budget to high-curvature-influence data subsets. We make this explicit and modelless using epiplexity × bandit concentration as a proxy, then use it to steer DDTree budget, bandit exploration, and speculative decode verification.

---

## Tasks

- [x] T1: Implement `CurvatureInfluenceScorer` trait + `EosProxyScorer` in `src/pruners/curvature_alloc.rs`
- [x] T2: Implement `CurvatureWeightedBudget` variant for DDTree node allocation
- [x] T3: Implement `BanditStrategy::CurvatureInfluence` — EoS-aware arm selection
- [x] T4: Wire curvature scoring into speculative decode verification depth
- [x] T5: GOAT proof — curvature-weighted vs uniform DDTree budget benchmark (6/6 tests)
- [x] T6: GOAT proof — EoS-aware bandit vs standard strategies (unit tests inline)
- [x] T7: Feature gate `curvature_alloc` with `#[cfg]` on all new code
- [x] T8: Update README, .docs, .research references

---

## T1: `CurvatureInfluenceScorer` Trait

**File:** `crates/katgpt-core/src/types.rs`

```rust
/// Curvature influence scorer — modelless proxy for (∇ℓₖ · v₁)².
///
/// Approximates the paper's curvature influence metric using:
/// - gradient persistence (loss residual / epiplexity area)
/// - domain alignment (softmax concentration / bandit score spread)
///
/// Reference: arXiv:2606.04212, Eq. (2) and Appendix F.4
pub trait CurvatureInfluenceScorer {
    /// Curvature influence proxy for group k ∈ [0, num_groups).
    /// Returns value in [0, 1] where 1 = highest influence.
    fn curvature_influence(&self, group: usize) -> f32;

    /// Number of tracked groups.
    fn num_groups(&self) -> usize;

    /// Update persistence component from loss signal.
    fn update_persistence(&mut self, group: usize, loss: f32);

    /// Update alignment component from score distribution.
    fn update_alignment(&mut self, group: usize, scores: &[f32]);
}

/// EoS-proxy scorer using epiplexity × softmax concentration.
pub struct EosProxyScorer {
    persistence: Vec<f32>,    // EMA loss residual per group
    alignment: Vec<f32>,      // Score concentration per group
    ema_rate: f32,            // Exponential moving average rate
    influence: Vec<f32>,      // Cached influence = persistence × alignment
}
```

**Key methods:**
- `curvature_influence(k)` → `persistence[k] * alignment[k]` (normalized to [0,1])
- `update_persistence(k, loss)` → EMA update of `|loss - running_mean|`
- `update_alignment(k, scores)` → softmax entropy of scores → `1 - normalized_entropy`

---

## T2: `CurvatureWeightedBudget`

**File:** `src/speculative/types.rs` — extend `PositionWeightedBudget` enum

```rust
pub enum BudgetMode {
    /// Exponential decay: weight(d) = exp(-d/γ)
    Exponential { gamma: f32 },
    /// Curvature-influence weighted: budget proportional to curvature_influence(position)
    /// Falls back to uniform when curvature data unavailable
    CurvatureWeighted { floor_ratio: f32 },
}
```

**Behavior:**
- For each DDTree depth `d`, weight = `curvature_influence(d) + floor_ratio`
- Normalize weights to sum to total_budget
- Floor_ratio (default 0.1) ensures no position gets zero nodes

---

## T3: `BanditStrategy::CurvatureInfluence`

**File:** `src/pruners/bandit.rs` — add to `BanditStrategy` enum

```rust
/// EoS-aware arm selection inspired by arXiv:2606.04212.
/// Arms with high curvature influence get exploration boost;
/// arms with low influence get suppressed.
CurvatureInfluence {
    /// Minimum exploration weight for suppressed arms
    floor: f32,
    /// Concentration threshold above which arms get boosted
    concentration_threshold: f32,
}
```

**Scoring:**
- Compute `concentration = max(scores) / sum(scores)` for all arms
- If `concentration > threshold`: boost top arm's score by `(concentration / threshold)`
- Floor guarantee: all arms get at least `floor * max_score`

---

## T4: Speculative Decode Verification Depth

**File:** `src/speculative/` — curvature-informed verification

When `curvature_alloc` feature is enabled:
- Tokens at positions with high curvature influence → full verification (speculate all branches)
- Tokens at positions with low curvature influence → fast-path (accept top-1 or verify only top-k=1)

---

## T5: GOAT Proof — Curvature-Weighted DDTree Budget

**File:** `tests/proof_curvature_alloc_goat.rs`

| Proof | Target | Metric |
|-------|--------|--------|
| G1: No panics | All tests pass |cargo test|
| G2: Curvature budget ≥ uniform | Valid node rate on hard positions | ≥ uniform rate |
| G3: No regression on easy positions | Valid node rate on easy positions | ≥ 95% of uniform |
| G4: Budget sum preserved | Total allocated nodes | = total_budget ± 1 |

---

## T6: GOAT Proof — EoS-Aware Bandit

**File:** `tests/proof_eos_bandit_goat.rs`

| Proof | Target |
|-------|--------|
| G1: No panics | All tests pass |
| G2: CurvatureInfluence converges to optimal arm | Within 2× steps of UCB1 |
| G3: Floor guarantee | All arms pulled ≥ 1 time in N rounds |
| G4: Arena improvement | Bomber HL score ≥ baseline bandit |

---

## T7: Feature Gate

```toml
[features]
curvature_alloc = ["bandit", "epiplexity"]
```

All new code behind `#[cfg(feature = "curvature_alloc")]`.

---

## T8: Documentation Updates

- Add `Curvature Allocation` section to `README.md` under Key Features
- Update `.docs/01_overview.md` feature flag table
- Update `.docs/15_paper_feature_comparison.md`
- Cross-reference Research 163 and Plan 183

---

## Expected Performance

| Metric | Overhead |
|--------|---------|
| Curvature proxy computation | O(1) per position (cached EMA) |
| CurvatureWeightedBudget | Same as PositionWeightedBudget (array scan) |
| EoS-aware bandit scoring | +1 division + 1 comparison per arm |
| Verification depth selection | O(1) lookup |
| **Total inference overhead** | **< 0.1%** |

---

## Why Default-On After GOAT

The paper proves the principle (selective allocation → +robustness AND +OOD). Our proxy has:
1. **Zero allocation** (all cached values)
2. **O(1) per-position** computation
3. **Floor guarantees** prevent starvation
4. **Graceful degradation** (falls back to uniform when no curvature data)

The only reason not to default-on is if the proxy doesn't track true curvature influence — which the GOAT proofs will verify.
