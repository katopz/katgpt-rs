# Plan 129: OPUS-Inspired Boltzmann + Redundancy Selection

**Research**: 089_OPUS_Optimizer_Induced_Projected_Utility_Selection.md
**Status**: 📋 Planned
**Feature Gate**: `opus_selection = ["bandit"]`

---

## Motivation

From OPUS paper (arXiv:2602.05400): Boltzmann sampling with redundancy penalty outperforms greedy top-k by +1.26 avg points on real benchmarks (Table 7). Current `BanditPruner` uses Thompson/UCB/EpsilonGreedy but lacks:
1. Explicit redundancy penalty against already-selected arms
2. Boltzmann (softmax) temperature-controlled sampling
3. Low-dimensional sketch for efficient inner-product estimation

This is the **highest-value distillation** from OPUS — composable, simple, directly improves existing bandit infrastructure without requiring pre-training scale.

## Scope

- [x] **In scope**: OpusBanditPruner<P>, CountSketch primitive, Boltzmann sampler, GOAT proofs
- [ ] **Out of scope**: Full OPUS pre-training pipeline, Muon optimizer, Bench-proxy construction, AdamW preconditioner

## Tasks

### T1: CountSketch Primitive
- [ ] Create `src/pruners/opus/count_sketch.rs`
- [ ] Implement `CountSketch` struct with hash/sign pairs
- [ ] `fn sketch(&self, vec: &[f32]) -> Vec<f32>` — O(d) → O(m) projection
- [ ] `fn inner_product_estimate(&self, a: &[f32], b: &[f32]) -> f32` — unbiased estimator
- [ ] Unit tests: unbiasedness, variance bounds
- [ ] Micro-bench: sketch speed vs full inner product

### T2: Boltzmann Sampler
- [ ] Create `src/pruners/opus/boltzmann.rs`
- [ ] `fn boltzmann_sample(utilities: &[f32], temperature: f32, rng: &mut Rng) -> usize`
- [ ] `fn boltzmann_sample_batch(utilities: &[f32], temperature: f32, k: usize, rng: &mut Rng) -> Vec<usize>`
- [ ] Temperature τ controls exploration: τ→0 greedy, τ→∞ uniform
- [ ] Unit tests: probability distribution validity, edge cases (τ=0, τ=∞, single arm)

### T3: OpusBanditPruner<P>
- [ ] Create `src/pruners/opus/types.rs` with `OpusConfig`, `OpusBanditPruner<P>`
- [ ] Create `src/pruners/opus/mod.rs` (index only)
- [ ] Implement `ScreeningPruner` for `OpusBanditPruner<P>`
- [ ] Core scoring: `U_z = alignment - redundancy_weight * ⟨ϕ(z), Φ_selected⟩`
- [ ] Maintain running history `Φ_selected` of sketch features
- [ ] Use Boltzmann sampling instead of Thompson/UCB for arm selection
- [ ] Delegate domain relevance to inner `BanditPruner<P>`

### T4: OpusBanditEnv for Standalone Testing
- [ ] Implement `BanditEnv` for a configurable test environment
- [ ] Redundant arms: some arms give same reward (test diversity)
- [ ] Run `BanditSession` with `OpusBanditPruner` vs `BanditPruner`
- [ ] Metric: cumulative reward, regret, diversity (unique arms pulled)

### T5: GOAT Proof — Bandit Benchmark
- [ ] Add `examples/bandit_08_opus.goat.rs`
- [ ] Compare: Thompson vs UCB vs OpusBandit on `bandit_01_basic` scenario
- [ ] Metric: regret convergence, cumulative reward, arm diversity
- [ ] Expected: Opus maintains ≥ Thompson reward + higher diversity

### T6: GOAT Proof — DDtree Quality
- [ ] Add opus option to `build_dd_tree_screened()` integration
- [ ] Compare tree quality with OpusBanditPruner vs BanditPruner
- [ ] Metric: tree coverage, depth efficiency, unique leaves
- [ ] Expected: Better coverage from redundancy penalty avoiding duplicate branches

### T7: Feature Gate + Cargo.toml
- [ ] Add `opus_selection = ["bandit"]` to `katgpt-rs/Cargo.toml` features
- [ ] Add `[[example]]` entries for opus examples
- [ ] Gate `src/pruners/opus/` module behind `#[cfg(feature = "opus_selection")]`
- [ ] Update README with OPUS section under 🪐 Gated Features

### T8: Documentation + Benchmark
- [ ] Add `.benchmarks/0XX_opus_boltzmann_bandit.md`
- [ ] Update `README.md` Feature Flags section
- [ ] Update `.research/089_...md` with actual benchmark results

## Key Types

```rust
/// OPUS configuration (paper defaults: τ=0.9, m=8192, ρ=0.5, b_t=64).
pub struct OpusConfig {
    pub temperature: f32,        // τ = 0.9
    pub redundancy_weight: f32,  // λ scaling for redundancy penalty
    pub sketch_dim: usize,       // m = 8192
    pub buffer_size: usize,      // N = 64
    pub selection_ratio: f32,    // ρ = 0.5
}

/// OPUS-inspired BanditPruner with Boltzmann sampling + redundancy penalty.
pub struct OpusBanditPruner<P: ScreeningPruner> {
    inner: BanditPruner<P>,
    config: OpusConfig,
    sketch: CountSketch,
    /// Running history of selected sketch features Φ(t,r).
    selected_history: Vec<Vec<f32>>,
    /// Per-arm last sketch for redundancy computation.
    arm_sketches: Vec<Vec<f32>>,
}
```

## Module Structure

```
src/pruners/opus/
├── mod.rs           # Index only — pub mod count_sketch, boltzmann, types;
├── types.rs         # OpusConfig, OpusBanditPruner<P>, impl ScreeningPruner
├── count_sketch.rs  # CountSketch projection (standalone, reusable)
└── boltzmann.rs     # Boltzmann sampling with redundancy-aware batch selection
```

## GOAT Proof Targets

| Proof | Metric | Target |
|-------|--------|--------|
| P1: Bandit reward | Cumulative reward | ≥ Thompson sampling |
| P2: Bandit diversity | Unique arms pulled | > Thompson sampling |
| P3: Regret convergence | Steps to 95% optimal | ≤ Thompson sampling |
| P4: DDtree coverage | Unique leaves / total | > BanditPruner baseline |
| P5: CountSketch accuracy | Inner product MSE | < 0.01 vs exact |

## References

- Research 089: `.research/089_OPUS_Optimizer_Induced_Projected_Utility_Selection.md`
- OPUS paper: arXiv:2602.05400v2
- Existing bandit: `src/pruners/bandit.rs` (Plan 030)
- CountSketch: Cormode & Muthukrishnan 2005