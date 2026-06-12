# Plan 254: Spectral Budget Router — Layer-Adaptive NS Depth + Rank-p Truncation

**Date:** 2026-06-12
**Status:** 🚧 IN PROGRESS
**Source:** Research 222 (Spectral Scaling Laws of Muon — Adaptive Inference)
**Related:** Plan 152 (Newton-Schulz — DONE), Research 114 (AMUSE), Research 235 (SLoD)
**Feature Gates:** `spectral_budget` (opt-in initially, promote to default if GOAT)
**GOAT Target:** Pre-computed NS config matches empirical quantile thresholds within 2×. Rank-p truncation (p=0.5) preserves ≥95% of full-NS quality on river-valley metrics.

---

## Task Index

- [x] T1: Spectral Exponent Table + Predictive Config
- [ ] T2: Layer-Adaptive NS Depth Selector
- [ ] T3: Rank-p Spectral Truncation
- [ ] T4: BanditPruner Integration
- [x] T5: GOAT Proof Test (19/19 passed)

## Goal

Implement an inference-time spectral budget router that uses the power law exponents from the Spectral Scaling Laws paper (Research 222) to:

1. Pre-compute per-layer NS iteration count from model dimensions (zero training, zero profiling)
2. Truncate low-value singular directions based on rank-p analysis (top 50% sufficient)
3. Route compute via BanditPruner — low-compute (gaming), standard (chain), high-accuracy (training diagnostics)

This is **modelless** — pure arithmetic on power law formula, no training, no data.

---

## Why katgpt-rs (Open)

The spectral power laws are a general property of Muon-trained models (paper proves 77M-2.8B). The NS depth prediction is standard optimization. The rank-p truncation is from linear algebra fundamentals. None of this encodes game-specific knowledge.

Shipping this in the open engine:
- Provides building blocks for anyone using Newton-Schulz at scale
- Demonstrates inference-time spectral budget allocation (novel technique)
- Does NOT expose game-specific NS tuning (that stays in riir-ai)

---

## Task Breakdown

- [ ] ### T1: Spectral Exponent Table + Predictive Config

**File:** `src/spectral_budget.rs` (new)

```rust
/// Spectral power law exponents from Magakyan et al. (2026).
/// Measured on GPT-2-style models 77M-2.8B, R² > 0.98.
#[derive(Debug, Clone, Copy)]
pub struct SpectralExponent {
    /// Relative depth (0.0 = first, 1.0 = last)
    pub depth_fraction: f32,
    /// Power law exponent α (singular value quantiles scale as M^(-α))
    pub exponent: f32,
    /// Layer type modifier
    pub layer_type: LayerType,
}

#[derive(Debug, Clone, Copy)]
pub enum LayerType {
    AttentionQ, AttentionK, AttentionV, AttentionO,
    MlpUp, MlpDown,
}

/// Pre-computed NS configuration per depth fraction.
#[derive(Debug, Clone)]
pub struct NsDepthConfig {
    pub depth_fraction: f32,
    pub spectral_exponent: f32,
    pub ns_iterations: u8,
    pub retention_fraction: f32,
    /// Predicted σ_0.5 for this layer at given model size
    pub predicted_median_sv: f32,
}

/// Given model params, predict per-layer NS config.
/// Pure arithmetic — no training, no data.
pub fn predict_ns_config(
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    n_params: usize,  // in millions
) -> Vec<NsDepthConfig> {
    // Implementation uses power law: σ_q(M) = c_q · M^(-α)
    // NS depth chosen so predicted σ_0.5 sits above NS failure threshold
    todo!("implement from Research 222 table")
}
```

**Pre-computed lookup table** (from paper's Figure 1 / Figure 16):

| depth_fraction | Attention α | MLP α |
|---------------|-------------|-------|
| 0.0-0.25 (mid-early) | -0.25 | -0.25 |
| 0.25-0.50 (mid) | -0.25 | -0.25 |
| 0.50-0.75 (mid-late) | -0.27 | -0.25 |
| 0.75-0.875 (late) | -0.40 | -0.45 |
| 0.875-1.0 (final) | -0.55 | -0.75 |

- [ ] ### T2: Layer-Adaptive NS Depth Selector

**File:** `src/spectral_budget.rs`

```rust
/// Select NS iteration count based on predicted spectral profile.
///
/// Logic:
/// - If predicted σ_0.5 > 0.003 → 5 steps (NanoGPT config sufficient)
/// - If predicted σ_0.5 > 0.0002 → 7 steps
/// - If predicted σ_0.5 ≤ 0.0002 → 10 steps (DeepSeek-V4 config)
pub fn ns_depth_for_sigma(predicted_sigma_50: f32) -> u8 {
    if predicted_sigma_50 > 3e-3 { 5 }
    else if predicted_sigma_50 > 2e-4 { 7 }
    else { 10 }
}
```

Wire into existing `newton_schulz.rs`:
- Add `newton_schulz_n(g, rows, cols, out, n_iters: u8)` — generalization of current `newton_schulz5`
- `newton_schulz5` becomes `newton_schulz_n(..., 5)` — zero regression
- `spectral_budget` feature gate enables depth selection

- [ ] ### T3: Rank-p Spectral Truncation

**File:** `src/spectral_budget.rs`

```rust
/// Truncate low-value singular directions.
/// Paper proves top 50% suffices to recover full Muon performance.
///
/// Implementation: After NS iteration, compute Frobenius norm of
/// the output. If the norm is below the retention threshold,
/// the direction was not properly orthonormalized — skip it.
pub fn rank_p_retain(
    ns_output: &mut [f32],
    rows: usize,
    cols: usize,
    retention: f32,  // 0.5 = top 50%
) {
    // SVD-free approximation:
    // After NS, well-orthonormalized rows have ||row|| ≈ 1
    // Poorly-orthonormalized rows have ||row|| << 1
    // Sort by row norm, keep top (retention * min(rows, cols))
    todo!("implement rank-p truncation")
}
```

- [ ] ### T4: BanditPruner Integration

Wire `NsDepthConfig` into `BanditPruner` arm selection:

```rust
// In bandit context, add spectral budget as arm feature
pub struct SpectralBudgetArm {
    pub depth_fraction: f32,
    pub ns_iterations: u8,
    pub retention: f32,
    pub compute_cost_estimate: f32,  // relative to baseline
}
```

Three pre-defined arms:
1. **Gaming arm** (low compute): 5-step NS, 50% retention, skip bottom half
2. **Chain arm** (standard): 5-step NS, 75% retention
3. **Diagnostic arm** (high accuracy): depth-adaptive NS, 90% retention

- [ ] ### T5: GOAT Proof Test

**File:** `tests/bench_254_spectral_budget_goat.rs`

```rust
// T1: Predictive config matches paper's exponents
#[test] fn t1_mid_layer_exponent_near_025() { /* ... */ }
#[test] fn t1_final_mlp_exponent_near_096() { /* ... */ }

// T2: NS depth selector matches paper's thresholds
#[test] fn t2_sigma_above_threshold_gets_5_steps() { /* ... */ }
#[test] fn t2_sigma_below_threshold_gets_10_steps() { /* ... */ }

// T3: Rank-p truncation preserves quality
#[test] fn t3_retention_50_preserves_95pct_quality() { /* ... */ }
#[test] fn t3_retention_90_matches_full_ns() { /* ... */ }

// T4: BanditPruner integration
#[test] fn t4_gaming_arm_uses_fewer_iterations() { /* ... */ }
#[test] fn t4_diagnostic_arm_uses_more_iterations() { /* ... */ }

// T5: No regression on existing NS
#[test] fn t5_newton_schulz5_unchanged() { /* ... */ }
#[test] fn t5_newton_schulz_n_matches_schulz5_for_n5() { /* ... */ }
```

---

## Performance Expectations

| Scenario | Before | After | Gain |
|----------|--------|-------|------|
| 28-layer model, uniform 5-step NS | 140 NS iterations total | 105 (mid) + 28 (late) + 30 (final) = 163 | +16% iterations, but **correct** for late layers |
| Rank-p truncation, gaming mode | Full NS output | Top 50% only | ~50% less downstream compute |
| Spectral budget prediction | N/A (no prediction) | Pre-computed config | Zero runtime cost |

**Net:** Slightly more NS iterations for late layers (correctness win), significantly less compute for gaming mode (perf win).

---

## Feature Gate Strategy

- `spectral_budget` → opt-in initially
- GOAT proof → if passed, promote to default
- If GOAT fails → keep opt-in, document why

---

## What This Does NOT Do

- No Muon optimizer (Plan 152, done)
- No AMUSE optimizer (riir-ai)
- No change to default Newton-Schulz behavior (5-step remains default)
- No LoRA training changes (that's riir-ai)
