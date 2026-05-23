# Plan 112: SR²AM Configurator Bandit — Learned Per-Turn Planning Regulation

> **Research:** 076 (SR²AM Self-Regulated Simulative Reasoning)
> **Paper:** [arXiv:2605.22138](https://arxiv.org/pdf/2605.22138) — Deng, Hou, Sá Neves et al., May 2026
> **Depends:** Plan 030 (Multi-Armed Bandit ✅), Plan 049 (G-Zero ✅), Plan 026 (Early Exit ✅)
> **Feature Gate:** `sr2am_configurator = ["bandit"]`
> **Status:** Phase 1–3 Complete ✅ (T1–T14)

## Tasks

### Phase 1: Configurator Types & Bandit Arm (Modelless)

- [x] T1: Define `PlanningDecision` enum in `microgpt-core/src/types.rs`
  - `PlanNew` — reset tree, full budget allocation
  - `PlanExtend` — keep tree, add one depth level
  - `PlanSkip` — early exit, direct token sampling
- [x] T2: Define `ConfiguratorContext` struct with `domain: usize` + `entropy_bin: usize`
- [x] T3: Add `ConfiguratorBandit` struct in `src/pruners/configurator_bandit.rs`
  - Three arms: `PlanNew`, `PlanExtend`, `PlanSkip`
  - UCB1 selection from existing bandit infrastructure
  - Q-values keyed by `(domain, entropy_bin)` — context-aware arm selection
- [x] T4: Add `sr2am_configurator` feature gate in `Cargo.toml` with `#[cfg(feature = "sr2am_configurator")]` on all new code
- [x] T5: Unit tests for `ConfiguratorBandit` arm selection, Q-value updates, context binning

### Phase 2: DDTree Integration (Modelless)

- [x] T6: Add `planning_decision: Option<PlanningDecision>` to `InferenceResult`
- [x] T7: Wire `ConfiguratorBandit` into speculative decode path
  - Before DDTree `build()`: query configurator → get `PlanningDecision`
  - `PlanNew` → reset tree, allocate full `tree_budget`
  - `PlanExtend` → keep existing tree, `draft_lookahead += 1`
  - `PlanSkip` → bypass tree, use `sample_token()` directly (early exit)
- [x] T8: Implement entropy-based context binning
  - `entropy_bin = (entropy * 10.0) as usize` — coarse 10-bin discretization
  - Q-values tracked per `(domain, entropy_bin)` pair
- [x] T9: Implement reward signal for Q-value updates
  - `quality_gain = screening_relevance - prev_relevance`
  - `token_cost = tokens_used / tree_budget`
  - `reward = quality_gain - β * token_cost` (β configurable, default 0.1)
- [x] T10: Integration tests: configurator correctly gates DDTree builds

### Phase 3: Uncertainty-Aware Horizon Truncation (Modelless)

- [x] T11: Add `max_plan_horizon: Option<usize>` to `InferenceOverrides`
  - When set, caps `draft_lookahead` to this value regardless of domain config
- [x] T12: Implement `entropy_truncate_horizon()` helper
  - If `entropy > threshold` (default 2.5 nats), cap `draft_lookahead` at 2
  - Maps directly to SR²AM's finding that web tasks benefit from short horizons
- [x] T13: Add `plan_horizon_used` metric to `InferenceResult`
- [x] T14: Tests for horizon truncation edge cases (entropy boundary, override precedence)

### Phase 4: GOAT Proof (Bomber/Go Arena)

- [ ] T15: Create `examples/bomber_12_sr2am_tournament.rs`
  - Players: Random, HL, GZero, SR2AM (ConfiguratorBandit)
  - Feature flags: `bomber,sr2am_configurator,g_zero`
- [ ] T16: Create `examples/go_11_sr2am_arena.rs`
  - Compare MCTS with vs without configurator bandit
  - Feature flags: `go,sr2am_configurator`
- [ ] T17: Run Bomber arena (1000 rounds) and record:
  - Win rate vs baseline (expect: same or better)
  - Tokens/pruners evaluated per game (expect: fewer with configurator)
  - `plan_skip` savings percentage
  - `plan_depth_used` distribution
- [ ] T18: Run Go arena (20 games, 9×9) and record:
  - Win rate vs baseline MCTS
  - MCTS node expansions per game (expect: fewer with configurator)
  - Configurator decision distribution (PlanNew/Extend/Skip %)

### Phase 5: Horizon-Deepening Reward for GZeroLoop (Model-Based Bridge)

- [ ] T19: Add `plan_depth_reward` to `GZeroLoop` reward shaping
  - When configurator chose `PlanNew` or `PlanExtend`:
    `bonus = α * (actual_depth / max_depth)` (α configurable, default 0.1)
  - When configurator chose `PlanSkip`:
    `bonus = 0` (no depth reward for reactive path)
- [ ] T20: Add `configurator_decision_history` tracking in `GZeroLoop` round metrics
  - Count PlanNew/Extend/Skip per round
  - Track average plan depth when planning was chosen
- [ ] T21: Wire reward into existing `loss_grpo.rs` advantage computation
  - No GRPO architecture changes — just richer reward signal
  - Verify existing CISPO loss handles the new reward shape
- [ ] T22: Feature gate: `sr2am_configurator` enables reward shaping in GZeroLoop
- [ ] T23: Benchmark: GZeroLoop with vs without horizon-deepening reward (100 rounds)

### Documentation

- [ ] T24: Update `README.md` — add SR²AM Configurator section under 🎯 G-Zero
- [ ] T25: Update `.docs/09_heuristic-learning.md` — configurator as HL pattern
- [ ] T26: Update `Cargo.toml` feature flags documentation
- [ ] T27: Add benchmark results to `.benchmarks/012_sr2am_configurator_goat.md`

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│         Inference Pipeline with SR²AM Configurator               │
│                                                                  │
│  Observation → Entropy Estimation                                │
│                    │                                             │
│                    ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ConfiguratorBandit (new)                                 │   │
│  │                                                          │   │
│  │  Context: (domain, entropy_bin)                          │   │
│  │  Arms: PlanNew | PlanExtend | PlanSkip                   │   │
│  │  Selection: UCB1 (existing bandit infra)                 │   │
│  │  Reward: quality_gain - β * token_cost                   │   │
│  └─────────────────┬────────────────────────────────────────┘   │
│                    │                                             │
│         ┌──────────┼──────────┐                                  │
│         ▼          ▼          ▼                                   │
│     PlanNew    PlanExtend   PlanSkip                             │
│         │          │          │                                   │
│   Reset tree   Extend tree  Bypass tree                          │
│   Full budget  +1 depth     Direct sample                        │
│         │          │          │                                   │
│         └────┬─────┘          │                                   │
│              ▼                ▼                                   │
│     DDTree::build()    sample_token()                            │
│     DDTree::extend()   + ConstraintPruner                        │
│              │                │                                   │
│              ▼                ▼                                   │
│     ScreeningPruner    BanditPruner<P>                           │
│     (relevance())      (explore/exploit)                         │
│              │                │                                   │
│              └───────┬────────┘                                   │
│                      ▼                                           │
│              InferenceResult                                      │
│              + planning_decision                                  │
│              + plan_depth_used                                    │
│              + plan_skip_savings                                  │
│              + plan_horizon_used                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Configurator as Bandit Arm, Not Separate Module

SR²AM implements the configurator as a separate prompted LLM module. We implement it as a **bandit arm selection** because:
- Zero inference cost (no LLM call needed)
- Integrates with existing `BanditPruner` infrastructure
- Q-values provide interpretable planning decisions
- UCB1 naturally balances exploration (try new planning depths) vs exploitation (use known-good depth)

### 2. Context-Aware via Entropy Binning

The configurator's decision should depend on the current uncertainty:
- Low entropy → `PlanSkip` (confident, no need for tree search)
- Medium entropy → `PlanExtend` (reasonable path, deepen slightly)
- High entropy → `PlanNew` (uncertain, explore fresh)

Entropy binning (10 bins) provides coarse context without overfitting to specific entropy values.

### 3. Horizon Truncation for High-Uncertainty Domains

SR²AM finds that web tasks (high environmental uncertainty) benefit from planning horizon capped at 2 steps. For our game domains:
- High-uncertainty states → cap `draft_lookahead` at 2 (avoid overcommitting)
- Low-uncertainty states → use full `draft_lookahead` from domain config

This is a simple heuristic that doesn't require learning — entropy threshold is configurable.

### 4. Feature-Gated, Always Available

Following project convention: all new code behind `sr2am_configurator = ["bandit"]` feature gate. The feature enables the bandit-based configurator decision; without it, existing `early_exit_patience` + `early_exit_gap` static thresholds continue to work unchanged.

### 5. Horizon-Deepening Reward (Model-Based Bridge)

The key SR²AM finding: RL should increase planning **depth** not **frequency**. For `GZeroLoop`:
- Add bonus reward when configurator chooses to plan AND plan depth is high
- No reward for planning frequency — only for depth when planning is chosen
- This shapes GRPO to learn: "when you plan, plan deep; when you don't need to plan, skip it"

## Key Types

```rust
/// SR²AM Configurator decision — learned per-turn planning regulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanningDecision {
    /// Reset tree, full budget allocation (high uncertainty, new sub-problem)
    PlanNew,
    /// Keep tree, extend depth by one level (moderate uncertainty, continuing)
    PlanExtend,
    /// Skip tree search, direct token sampling (low uncertainty, confident)
    PlanSkip,
}

/// Context key for configurator bandit — coarse entropy binning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConfiguratorContext {
    pub domain: usize,      // Domain index from bandit
    pub entropy_bin: usize, // floor(entropy * 10.0), clamped to 0..9
}
```

## Benchmark Plan

### D1: Configurator Decision Distribution (T15-T16)

Run 1000 games Bomber, 20 games Go. Report:
- `% PlanNew / PlanExtend / PlanSkip` per domain
- Average plan depth when PlanNew chosen
- Average plan depth when PlanExtend chosen
- Correlation between entropy bin and decision

### D2: Efficiency Metrics (T17-T18)

Compare with/without configurator:

| Metric | Baseline (no configurator) | With Configurator | Δ |
|--------|---------------------------|-------------------|---|
| Win rate (Bomber) | baseline | ≥ baseline | ≥0 |
| Pruners evaluated/game | baseline | < baseline | ↓ |
| MCTS nodes expanded (Go) | baseline | < baseline | ↓ |
| plan_skip savings % | 0% | >0% | ↑ |

### D3: Horizon-Deepening Reward (T23)

Run GZeroLoop 100 rounds with/without reward shaping:

| Metric | No Horizon Reward | With Horizon Reward | Δ |
|--------|-------------------|---------------------|---|
| Average plan depth when planning | baseline | > baseline | ↑ |
| Planning frequency | baseline | ≈ baseline | ≈0 |
| Win rate | baseline | ≥ baseline | ≥0 |

## File Changes Summary

| File | Change | Lines (est.) |
|------|--------|-------------|
| `crates/microgpt-core/src/types.rs` | `PlanningDecision`, `ConfiguratorContext` | +36 |
| `crates/microgpt-core/src/lib.rs` | Feature-gated re-exports | +3 |
| `crates/microgpt-core/Cargo.toml` | Feature gate | +1 |
| `src/pruners/mod.rs` | Add `configurator_bandit` module + re-exports | +6 |
| `src/pruners/configurator_bandit.rs` | `ConfiguratorBandit` struct + impl + tests | +570 |
| `crates/microgpt-core/src/types.rs` | `InferenceResult` addition | +3 |
| `src/feedback.rs` | `planning_decision: None` in test structs | +6 |
| `src/speculative/dd_tree.rs` | `entropy_truncate_horizon` + `build_inference_result` update | +29 |
| `src/speculative/mod.rs` | Re-export `entropy_truncate_horizon` | +2 |
| `Cargo.toml` | Feature gate + default/full | +3 |
| `tests/test_sr2am_configurator_goat.rs` | Integration tests (29 tests) | +487 |
| **Total** | | **~1146** |

## References

- SR²AM paper: [arXiv:2605.22138](https://arxiv.org/pdf/2605.22138)
- SR²AM code: https://github.com/sailing-lab/sr2am
- Our Research 076: `.research/076_SR2AM_Self_Regulated_Simulative_Reasoning.md`
- Our G-Zero Plan: `.plans/049_g_zero_self_play.md`
- Our Bandit Plan: `.plans/030_multi_armed_bandit.md`
- Our Early Exit: `.plans/026_ppot_logit_resampling.md`
