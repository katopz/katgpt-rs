# Plan 137: PrudentBanker Safe-Phased Bandit — Delay-Calibrated Exploration Safety

> **Research:** [098 — PrudentBanker Safe Delayed Adversarial Bandits](../.research/098_PrudentBanker_Safe_Delayed_Adversarial_Bandits.md)
> **Paper:** [arXiv:2605.23351](https://arxiv.org/abs/2605.23351) — Hu, Cai, Vlatakis-Gkaragkounis (2026)
> **Depends:** Plan 030 (Multi-Armed Bandit ✅), Plan 049 (G-Zero ✅)
> **Feature Gate:** `safe_bandit = ["bandit"]` (opt-in, NOT default-on)
> **Status:** ✅ COMPLETE (T1-T18 done, GOAT 5/5)
> **GOAT Pillar:** ❌ Not a pillar — secondary bet, enhances all 4 pillars via safer bandit exploration. See [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md).
> **Domain:** `katgpt-rs` — generic safe-phased bandit strategy. Game-specific delay configs and per-game ξ tuning stay in `riir-ai`.

---

## Summary

Add `BanditStrategy::SafePhased` — a phased aggression strategy that mixes between an active bandit learner and a designated safe baseline arm. Only escalates exploration (geometrically) when accumulated evidence certifies the baseline is suboptimal. Includes delay-calibrated slack `ξ(D)` that accounts for pending (unobserved) feedback, preventing premature aggression when observations are delayed.

This is the **generic** (MIT) implementation. Game-specific delay configs (per-game ξ values, frame-sampling delay adaptation, MMO tick-delay wrappers) are **NOT** in this plan — they belong in `riir-ai` as the secret selling point.

---

## Why

1. **Safety net for production bandits:** Current `BanditStrategy` variants (UCB1, ε-greedy, Thompson) have no built-in safety bound. If the bandit explores a bad arm for many rounds, cumulative cost is unbounded. `SafePhased` caps this to O(1) against a known-good baseline.
2. **Delay-awareness for frame-sampling:** Plan 070 (Frame-Sampling Real-Time Bridge) creates inherent feedback delay — AI evaluates at 5Hz but game ticks at 20Hz. Without delay calibration, the bandit can oscillate between conservative and aggressive modes when frame samples are sparse.
3. **Low implementation cost:** ~200-300 lines. New `BanditStrategy` variant + `SafePhasedState` struct. Plugs into existing `BanditPruner<P>` transparently.
4. **Easy GOAT proof:** Three measurable properties: (1) baseline regret bounded, (2) worst-case competitive with UCB1, (3) no performance penalty when D=0.

---

## Tasks

### Phase 1: Core Types (Modelless)

- [x] T1: Add `BanditStrategy::SafePhased` variant to `src/pruners/bandit.rs`
  ```rust
  /// Phased aggression with safe baseline mixture (PrudentBanker-inspired).
  ///
  /// Mixes: xₜ = αₖ · active_arm + (1 - αₖ) · baseline_arm
  /// αₖ escalates geometrically only when evidence certifies baseline suboptimality.
  /// Delay-calibrated slack ξ(D) prevents premature aggression under delayed feedback.
  SafePhased {
      /// Index of the safe baseline arm (must be valid arm index).
      baseline_arm: usize,
      /// Minimum baseline probability δ ∈ (0, 1/num_arms].
      /// The baseline arm must always receive ≥ δ probability mass.
      delta: f32,
      /// Estimated max feedback delay in rounds (default: 0 = no delay).
      estimated_delay: u32,
  },
  ```
- [x] T2: Add `SafePhasedState` struct
  ```rust
  /// Internal state for SafePhased strategy.
  struct SafePhasedState {
      /// Current aggression level index k (starts at 1).
      phase: u32,
      /// Current delay estimate D̂ₛ (doubling trick).
      delay_estimate: f32,
      /// Cumulative phase gap statistic on arrived data.
      phase_gap_arrived: f32,
      /// Current aggression coefficient αₖ.
      alpha: f32,
      /// Regret budget R̂ based on current delay estimate.
      regret_budget: f32,
      /// Delay slack ξ(D̂ₛ) = (√(8·D̂ₛ + 1) - 1) / δ.
      delay_slack: f32,
  }
  ```
- [x] T3: Implement `SafePhasedState::new()` and `SafePhasedState::compute_alpha()`
  - α₁ = min(1/R̂, 1), αₖ = min(2ᵏ⁻¹/R̂, 1)
  - R̂ = C · (√T + √(D̂ₛ · ln(D̂ₛ + 1))) where C ≈ 10√(C₁·C₂)
- [x] T4: Implement delay-slack computation
  - `ξ(D̂ₛ) = (√(8·D̂ₛ + 1) - 1) / δ`
  - This is the key innovation from the paper — the "hidden debt buffer"
- [x] T5: Add `delay_credits: Vec<f32>` field to `BanditStats`
  - Tracks pending (unobserved) feedback per round
  - When feedback arrives, credit is released; when missing, credit remains pending
  - Simple implementation: track count of pending observations

### Phase 2: Arm Selection with Safe Mixture (Modelless)

- [x] T6: Implement `select_arm_safe_phased()` in `BanditStats`
  - Compute active arm via inner UCB1 (or any sub-strategy)
  - Mix: `if rng.f32() < alpha → active_arm, else → baseline_arm`
  - This is the core safe-mixture mechanism from PrudentBanker
- [x] T7: Wire `SafePhased` into `BanditStats::select_arm()` match arm
  - Call `select_arm_safe_phased()` when strategy is `SafePhased`
  - Pass through to existing arm selection for all other strategies
- [x] T8: Implement phase-gap accumulation in `BanditPruner::update()`
  - When updating with reward, accumulate phase gap:
    `Δₖ(t) += reward(baseline_arm) - reward(selected_arm)` on arrived data
  - Compare against threshold: `2·R̂ + ξ(D̂ₛ)`
- [x] T9: Implement soft restart (phase transition)
  - If phase gap exceeds threshold AND α < 1:
    - Increment phase k → k+1
    - Recompute αₖ = min(2ᵏ⁻¹/R̂, 1)
    - Reset phase gap accumulator
    - Reset arm Q-values to uniform (soft restart)
- [x] T10: Implement hard restart (delay adaptation)
  - If accumulated delay exceeds current D̂ₛ estimate:
    - Double delay estimate: D̂ₛ → 2·D̂ₛ
    - Full reset: phase → 1, α → 1/R̂(new D̂ₛ)
    - Reset all Q-values to uniform

### Phase 3: GOAT Proof (Modelless)

- [x] T11: `safe_phased_01_baseline_regret_bounded` test
  - Create 5-arm bandit with known safe baseline
  - Run 10000 episodes with SafePhased strategy
  - Assert: cumulative regret vs baseline ≤ O(log D) bound
  - Compare: UCB1 baseline regret grows unboundedly
- [x] T12: `safe_phased_02_worst_case_competitive` test
  - Same setup, measure worst-case regret vs optimal arm
  - Assert: SafePhased worst-case within 2× of UCB1 worst-case
  - This validates the Õ(√T) guarantee
- [x] T13: `safe_phased_03_no_delay_no_cost` test
  - Run with D̂ₛ = 0 (no estimated delay)
  - Assert: SafePhased performance ≈ UCB1 performance (within 10%)
  - Validates the "no extra fees" claim
- [x] T14: `safe_phased_04_delay_robustness` test
  - Simulate delayed feedback (hold rewards for N rounds)
  - Assert: SafePhased with correct ξ(D) doesn't oscillate α
  - Compare: SafePhased without ξ(D) oscillates aggressively
- [x] T15: `safe_phased_05_phase_gap_correctness` test
  - Verify phase gap accumulator matches hand-computed values
  - Verify soft restart triggers at correct threshold
  - Verify α sequence: 1/R̂, 2/R̂, 4/R̂, ..., 1

### Phase 4: Integration (Modelless)

- [x] T16: Add `safe_bandit` feature gate in `Cargo.toml`
  - `safe_bandit = ["bandit"]`
  - All `SafePhased` code guarded by `#[cfg(feature = "safe_bandit")]`
- [x] T17: Add `BanditStrategy::SafePhased` to `BanditSession` orchestrator
  - Handle phased aggression in episode loop
  - Track phase-gap accumulation across episodes
- [x] T18: Add example `bandit_08_safe_phased.rs`
  - Demo: 5-arm bandit, arm 0 is safe baseline (mean=0.5), arm 3 is best (mean=0.8)
  - Show α escalation over time
  - Show baseline regret stays bounded
  - Compare UCB1 vs SafePhased vs ε-greedy
- [x] T19: Benchmark `SafePhased` arm selection vs UCB1
  - Micro-bench: single arm selection with SafePhased state
  - Assert: ≤ 20% overhead vs UCB1 (extra random draw + alpha check)

---

## GOAT Proofs (Target: 5/5)

| Proof ID | What | Threshold | Status |
|----------|------|-----------|--------|
| G1 | Baseline regret bounded | ≤ C · log(T) for all T | ✅ |
| G2 | Worst-case competitive | ≤ 2× UCB1 regret | ✅ |
| G3 | No delay = no cost | ≤ 1.1× UCB1 when D̂=0 | ✅ |
| G4 | Delay robustness | α variance ≤ 50% of naive | ✅ |
| G5 | Phase gap correctness | Exact match on synthetic | ✅ |

---

## What Stays in riir-ai (NOT This Plan)

These are the **Super GOAT / secret selling point** items that belong in riir-ai:

1. **Per-game delay-calibrated configs** — `FrameSamplingBanditConfig` with game-specific ξ values
2. **MMO tick-delay bandit wrappers** — 20Hz game tick → 5Hz AI evaluation delay adaptation
3. **Bomber frame-sampling bandit** — SafePhased with Bomber safety heuristic as baseline
4. **Go komi-delay bandit** — SafePhased with komi-based baseline for move selection
5. **NPC dialog delay adaptation** — SafePhased with template fallback as baseline

These will be planned separately in `riir-ai/.plans/` when Issue 013 (riir-games crate) progresses.

---

## Module Structure

```
src/pruners/
├── bandit.rs              # Modified: add SafePhased variant + SafePhasedState
├── safe_phased.rs         # New: SafePhasedState implementation (#[cfg(feature = "safe_bandit")])
└── ...

examples/
├── bandit_08_safe_phased.rs  # New: SafePhased demo
└── ...
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase gap accumulator overflow | Low | Medium | Use f64 accumulator, reset on soft restart |
| ξ(D) too conservative for stochastic rewards | Medium | Low | ξ is an upper bound; stochastic case is easier. Tune C constant. |
| Feature gate isolation breaks | Low | High | All SafePhased code behind `#[cfg(feature = "safe_bandit")]` |
| Performance regression on hot path | Low | Medium | Benchmark T19 asserts ≤ 20% overhead |
| Soft restart resets good Q-values | Medium | Medium | Phase gap only triggers when baseline is truly suboptimal (paper Lemma 4.6) |

---

## References

- Hu, Cai, Vlatakis-Gkaragkounis (2026). "Prudent-Banker: No Extra Fees for Baseline Safety in Adversarial Bandits With and Without Delays." arXiv:2605.23351.
- Müller et al. (2025). "Best of both worlds: Regret minimization versus minimax play." arXiv:2502.11673.
- Huang, Dai, Huang (2023). "Banker Online Mirror Descent: A Universal Approach for Delayed Online Bandit Learning." ICML 2023.
- [MMO GOAT Pillars Decision Matrix](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md)
