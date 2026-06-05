# Plan 186: NDS Curvature Proxy — Modelless Inference-Time Budget Control

> **Research:** 166 (Muon Curvature Perspective — NDS)
> **Paper:** [arXiv:2606.04662](https://arxiv.org/abs/2606.04662) — Wang et al., Jun 2026
> **Depends:** Plan 183 (CIAB 📋), Plan 030 (Bandit ✅), Plan 152 (Newton-Schulz ✅)
> **Feature Gate:** `nds_proxy = ["bandit"]`
> **Status:** Active
> **Default-on:** Yes (after GOAT proof) — zero perf overhead, theoretically grounded

---

## Summary

Implement inference-time NDS (Normalized Directional Sharpness) proxy from paper arXiv:2606.04662. The paper proves Muon's 2× speedup over Adam comes from lower NDS — the curvature encountered along the update direction. We distill this to a modelless proxy computed from token marginals, then use it to:

1. Modulate DDTree budget (high NDS = confident = less budget; low NDS = uncertain = more budget)
2. Add spectral balance score for DDTree branch selection (Muon's "equal energy" principle)
3. Apply layer-weighted verification depth from paper's 70/28/2 within-layer decomposition

All modelless — no Hessian, no training, no new allocations.

---

## Tasks

- [x] T1: Implement `nds_proxy()` function in `src/pruners/nds_proxy.rs`
- [x] T2: Implement `NdsBudgetModifier` trait + `SpectralFlatnessBudget` in `src/pruners/nds_proxy.rs`
- [x] T3: Implement `spectral_balance_score()` for DDTree branch visit counts
- [x] T4: Implement `layer_weighted_verification_depth()` from 70/28/2 heuristic
- [x] T5: Wire NDS proxy into DDTree budget allocation (compose with `budget_adaptation`)
- [x] T6: Wire spectral balance into DDTree branch scoring (compose with `bandit`)
- [x] T7: Add `nds_proxy` to `CurvatureInfluenceScorer` trait (enhance Plan 183)
- [x] T8: Feature gate `nds_proxy` with `#[cfg]` on all new code
- [x] T9: GOAT proof — NDS proxy tests (13/13 passing)
- [x] T10: GOAT proof — spectral balance, layer depth, budget modifier tests inline
- [x] T11: Update README, .docs, .research references

---

## T1: `nds_proxy()` Function

**File:** `crates/katgpt-core/src/types.rs`

```rust
/// Inference-time NDS (Normalized Directional Sharpness) proxy.
///
/// Paper (arXiv:2606.04662) proves Muon's advantage comes from lower NDS —
/// the curvature encountered along the update direction. At inference time,
/// we approximate NDS from token marginal distributions:
/// - High NDS ≈ peaked distribution (few tokens dominate) → confident
/// - Low NDS ≈ flat distribution (many tokens compete) → uncertain
///
/// Modelless: computed from existing marginals, no Hessian needed.
/// Complexity: O(K) where K = number of top-K marginals.
#[inline]
pub fn nds_proxy(top_k_probs: &[f32]) -> f32 {
    if top_k_probs.is_empty() { return 0.5; }
    let n = top_k_probs.len() as f32;
    let am = top_k_probs.iter().sum::<f32>() / n;
    if am <= 0.0 { return 0.5; }
    let ln_sum: f32 = top_k_probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|p| p.ln())
        .sum();
    let gm = (ln_sum / n).exp();
    // Spectral flatness = gm/am ∈ [0, 1]
    // NDS proxy = 1 - flatness. High = peaked, Low = flat.
    (1.0 - gm / am).clamp(0.0, 1.0)
}
```

---

## T2: `NdsBudgetModifier` Trait

```rust
/// Budget modifier based on NDS proxy.
///
/// Modulates DDTree budget inversely with confidence:
/// - High NDS (peaked marginals) → less budget needed
/// - Low NDS (flat marginals) → more budget needed
pub trait NdsBudgetModifier {
    /// Compute budget multiplier from NDS proxy value.
    fn budget_scale(&self, nds: f32) -> f32;
}

/// Spectral flatness budget modifier.
/// Scale = 1.0 + (1 - NDS) * max_boost ∈ [1.0, 1.0 + max_boost]
pub struct SpectralFlatnessBudget {
    pub max_boost: f32,  // default 0.5
}

impl NdsBudgetModifier for SpectralFlatnessBudget {
    fn budget_scale(&self, nds: f32) -> f32 {
        1.0 + (1.0 - nds) * self.max_boost
    }
}
```

---

## T3: Spectral Balance Score

```rust
/// Spectral balance score for DDTree branch visit distribution.
///
/// Inspired by Muon's "equal energy across eigenmodes" principle.
/// Balanced exploration → lower curvature penalty (paper's NDS result).
/// Returns ∈ [0, 1]: 1.0 = perfectly balanced, 0.0 = all on one branch.
pub fn spectral_balance_score(visit_counts: &[u32]) -> f32 {
    let total: u32 = visit_counts.iter().sum();
    if total == 0 { return 1.0; }
    let n = visit_counts.len() as f32;
    if n <= 1.0 { return 1.0; }
    let entropy: f32 = visit_counts.iter()
        .filter(|&&v| v > 0)
        .map(|&v| {
            let p = v as f32 / total as f32;
            -p * p.log2()
        })
        .sum();
    (entropy / n.log2()).clamp(0.0, 1.0)
}
```

---

## T4: Layer-Weighted Verification Depth

```rust
/// Layer-weighted verification depth from paper's within-layer NDS decomposition.
///
/// Paper finding: 70% of Muon's NDS advantage from boundary layers (L1, L12),
/// 28% from deep layers (L8-L11), 2% from middle layers (L2-L7).
/// → Verify boundary layers more deeply, middle layers minimally.
#[derive(Debug, Clone, Copy)]
pub enum LayerDepth {
    Boundary = 3,  // 70% NDS → verify deeply
    Deep = 2,      // 28% NDS
    Middle = 1,    // 2% NDS
}

pub fn layer_nds_depth(layer_idx: usize, total_layers: usize) -> LayerDepth {
    let is_boundary = layer_idx == 0 || layer_idx == total_layers - 1;
    let is_deep = !is_boundary && layer_idx >= total_layers * 7 / 10;
    match (is_boundary, is_deep) {
        (true, _) => LayerDepth::Boundary,
        (false, true) => LayerDepth::Deep,
        _ => LayerDepth::Middle,
    }
}
```

---

## T9-T10: GOAT Proofs

### T9: NDS Budget Modulation Benchmark

```
Test: Generate synthetic marginals (peaked, uniform, bimodal).
Measure: DDTree quality (valid token rate, diversity) with NDS budget vs uniform budget.
Expect: NDS-modulated budget maintains quality while reducing total nodes for peaked queries.
```

### T10: Spectral Balance Arena

```
Test: Run bomber arena with spectral balance branch scoring vs greedy.
Measure: Win rate, kill rate, decision diversity.
Expect: Balanced exploration ≥ greedy for hard maps, no regression on easy maps.
```

---

## Constraints

1. **Modelless only** — no Hessian, no training, no weight access
2. **Zero new allocations** — NDS proxy computed from existing marginals slice
3. **Feature-gated** — `nds_proxy = ["bandit"]`, all new code behind `#[cfg]`
4. **Default-on after GOAT** — if gain proven, must be on by default
5. **Composes with existing** — works alongside `budget_adaptation`, `bandit`, CIAB

---

## Relationship to Other Plans

| Plan | Relation |
|------|----------|
| **183 (CIAB)** | NDS proxy adds signal to `CurvatureInfluenceScorer` |
| **167 (Budget Adaptation)** | NDS is an additional budget modulation factor |
| **152 (Newton-Schulz)** | River-valley diagnostics already measure spectral structure |
| **030 (Bandit)** | Spectral balance becomes a new arm selection signal |
| **riir-ai Plan 208** | Model-based NDS monitoring for LoRA training |
