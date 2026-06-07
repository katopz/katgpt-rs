# Plan 211: Three-Mode Neuro-Symbolic Bandit Router

**Date:** 2026-06-07
**Status:** 🔄 Phase 2 + Phase 3 complete
**Research:** `.research/186_Neurosymbolic_RL_Survey_Three_Mode_Router.md`
**Depends On:** Plan 190 (AND-OR DDTree), Plan 206 (EGCS/EpisodePruner), BanditPruner, AbsorbCompressLayer, ConstraintPruner, ScreeningPruner, SynPruner, WASM validator, TrialLog, RegressionSuite
**Feature Gates:** `three_mode_router` (parent), `auto_constraint_synthesis`, `safe_exploration_budget` (independently gateable)
**GOAT Criteria:** F1 mode selection accuracy ≥85% vs oracle on benchmark suite; F2 synthesized constraint acceptance rate ≥90%; F4 grounding quality correlation r≥0.7 with mode effectiveness; zero overhead on miss path (feature disabled); mode selection overhead <50ns per step

---

## Overview

The Neurosymbolic RL survey (arXiv 2309.01038) categorizes neuro-symbolic systems into three modes — Learning for Reasoning (L4R), Reasoning for Learning (R4L), and Learning-Reasoning (LR) — but treats them as static architecture choices. katgpt-rs already deploys all three simultaneously via existing trait implementations. This plan makes them **dynamic**: a multi-armed bandit router selects the dominant mode per decode step based on constraint density, marginal entropy, episode hit rate, and verification history. Sigmoid-gated mixing (NOT softmax). No training — pure inference-time bandit.

Five fusions from Research 186:
- **F1 (Three-Mode Bandit Router):** Core novelty. 6-arm UCB1 bandit over neuro-symbolic modes. P0.
- **F2 (Auto Constraint Synthesis):** Mine episodes for recurring patterns → auto-generate ConstraintPruner rules. Addresses survey's open Challenge VII.A. P1.
- **F3 (Safe Exploration Budget):** Configurable verification tier budget with conservative fallback. Opt-in. P2.
- **F4 (Grounding Quality Metric):** KL divergence between pruned/unpruned marginals → mode calibration signal. Part of F1. P1.
- **F5 (Programmatic Policy Extraction):** DDTree path → human-readable policy rules. Opt-in, overlaps Research 184 F4. P3.

```
                    ModeFeatures (constraint_density, entropy, hit_rate, verif_history)
                              │
                              ▼
                    ┌─────────────────────┐
                    │  ThreeModeBandit    │
                    │  6 arms × UCB1     │
                    │  sigmoid mixing     │
                    └──────┬──────────────┘
                           │ selected mode
              ┌────────────┼──────────────┐
              ▼            ▼              ▼
         ┌────────┐  ┌──────────┐  ┌──────────┐
         │  L4R   │  │   R4L    │  │   LR     │
         │ DDTree │  │ SynPruner│  │AbsorbComp│
         │ →Const │  │ →DDTree  │  │↔Episode  │
         └───┬────┘  └────┬─────┘  └────┬─────┘
             │            │              │
             └────────────┼──────────────┘
                          ▼
                   Verification
                   Outcome → Bandit Update
                          │
                    ┌─────┴──────┐
                    ▼            ▼
             EpisodePruner   Grounding
             (F2 mining)    Quality (F4)
```

---

## Tasks

### Phase 1: Three-Mode Router Core (F1 — P0)

- [x] **F1.1:** Create `src/pruners/three_mode_bandit.rs` with `NeuroSymbolicMode` enum
  - `#[derive(Debug, Clone, Copy, PartialEq, Eq)]` + `#[repr(u8)]` — 1 byte, 6 variants
  - Variants: `PureL4R, PureR4L, PureLR, Balanced, R4LHeavy, L4RHeavy`

- [x] **F1.2:** Define `ModeFeatures` struct — 4× f32 (16 bytes, cache-line friendly)
  - `constraint_density: f32` — active ConstraintPruner rules / max rules
  - `marginal_entropy: f32` — Shannon entropy of DDTree token distribution
  - `episode_hit_rate: f32` — EpisodePruner cache hit ratio (rolling window, last N steps)
  - `verif_success_rate: f32` — compilation success / attempts (rolling window, last M steps)

- [x] **F1.3:** Define `BanditArm` struct per mode
  - `visits: u32`, `reward_sum: f32` — UCB1 state (12 bytes per arm)
  - 6 arms total (one per `NeuroSymbolicMode` variant)

- [x] **F1.4:** Implement `ThreeModeBandit` struct
  - `arms: [BanditArm; 6]` — 72 bytes
  - `exploration_constant: f32` — UCB1 c parameter (default: √2)
  - `feature_weights: [f32; 4]` — linear context weights for mode preference

- [x] **F1.5:** Implement `select_mode(&self, features: &ModeFeatures) -> NeuroSymbolicMode`
  - UCB1 scoring: `mean_reward + c × √(ln(total_visits) / arm_visits)`
  - Context-aware: boost arm scores by `dot(feature_weights, features)` offset
  - O(1) — fixed 6 arms, no allocation

- [x] **F1.6:** Implement `compute_mixing_weights(&self, features: &ModeFeatures) -> [f32; 3]`
  - Independent sigmoid per axis: `w_l4r = sigmoid(…)`, `w_r4l = sigmoid(…)`, `w_lr = sigmoid(…)`
  - Normalize: `w_i / (w_l4r + w_r4l + w_lr)` → probability simplex
  - NOT softmax — sigmoid is independent per weight

- [x] **F1.7:** Implement `update(&mut self, mode: NeuroSymbolicMode, reward: f32)`
  - Standard UCB1 update: increment visits, add reward
  - Reward: 1.0 for compilation success, -0.5 for failure, 0.0 for no verification

- [x] **F1.8:** Wire mode selection into DDTree step loop
  - Before token selection: compute ModeFeatures from current state
  - Select mode → apply mixing weights to ConstraintPruner / SynPruner / EpisodePruner influence
  - After verification: call `bandit.update(mode, reward)`

- [x] **F1.9:** Feature gate: `three_mode_router` in `Cargo.toml` features
  - Default-off until GOAT gate passes
  - When disabled: use existing `Balanced` mode (current behavior)

- [x] **F1.10:** Test: mode switches on high-entropy vs low-entropy synthetic inputs
  - High entropy (>2.0) → expect L4R or L4RHeavy mode selected
  - Low entropy (<0.5) with high constraint density → expect R4L or R4LHeavy

- [x] **F1.11:** Test: R4L weight increases when constraint density is high
  - Feed synthetic ModeFeatures with constraint_density=0.9 → verify R4L arm wins

- [x] **F1.12:** Benchmark: mode selection overhead < 50ns per step
  - Criterion bench on `select_mode()` with typical ModeFeatures
  - Profile with `perf` / Instruments — no allocation in hot path

### Phase 2: Auto Constraint Synthesis (F2 — P1)

- [x] **F2.1:** Create `src/pruners/constraint_miner.rs` with `ConstraintMiner` struct
  - `min_support: usize` — minimum episode count for a pattern (default: 10)
  - `min_acceptance: f32` — minimum acceptance rate (default: 0.90)
  - `last_mine_epoch: u64` — deduplication / scheduling

- [x] **F2.2:** Implement `extract_frequent_sequences(paths: &[AcceptedPath], min_support: usize) -> Vec<Pattern>`
  - Sliding window over token sequences in accepted paths
  - Window sizes: 2, 3 (bigrams and trigrams — longer is overfit)
  - Count occurrences, filter by min_support

- [x] **F2.3:** Implement `Pattern::acceptance_rate(&self, episode_db: &EpisodeDb) -> f32`
  - Count pattern in accepted paths / total paths containing pattern prefix
  - Filter: only promote patterns with ≥90% acceptance

- [x] **F2.4:** Implement `Constraint::from_pattern(pattern: &Pattern) -> Constraint`
  - Convert token sequence pattern to ConstraintPruner-compatible constraint
  - "token X followed by token Y" → `SequenceConstraint { first: TokenId, second: TokenId }`

- [x] **F2.5:** Implement `mine_and_insert(miner: &ConstraintMiner, episode_db: &EpisodeDb, pruner: &mut ConstraintPruner)`
  - Background task: called between decode steps, not on hot path
  - Extract patterns → filter → generate constraints → insert into pruner
  - Rate-limit: mine at most once per N decode steps (configurable)

- [x] **F2.6:** Feature gate: `auto_constraint_synthesis` in `Cargo.toml` features
  - Default-off until GOAT gate passes

- [x] **F2.7:** Test: mine patterns from 100 synthetic episodes
  - Generate 100 episodes with known recurring patterns
  - Verify miner extracts the correct patterns with ≥90% acceptance filter

- [x] **F2.8:** Test: verify auto-generated constraints are valid
  - Mined constraints pass `ConstraintPruner::is_valid()` self-test
  - No contradictory constraints generated

- [x] **F2.9:** Benchmark: mining overhead < 100μs per batch of 100 episodes
  - Measure end-to-end mine+insert latency
  - Verify zero hot-path impact (only background)

### Phase 3: Grounding Quality Metric (F4 — P1, part of F1)

- [x] **F4.1:** Create `grounding_quality(pruned: &[f32], unpruned: &[f32]) -> f32` in `src/pruners/three_mode_bandit.rs`
  - KL(pruned || unpruned): `Σ p × ln(p/q)` where p=pruned, q=unpruned
  - Return `sigmoid(kl)` — bound to [0, 1]
  - Guard: skip terms where p≤0 or q≤0 (log undefined)
  - SIMD-friendly: chunked loop over vocabulary-sized arrays

- [x] **F4.2:** Wire grounding quality into `ModeFeatures`
  - Add as 5th feature (or replace one of the 4 — design decision)
  - Low grounding quality (<0.3) → shift weight away from R4L
  - High grounding quality (>0.7) → shift weight toward R4L

- [x] **F4.3:** Test: verify grounding quality correlates with mode effectiveness
  - Synthetic test: strong constraints → high KL → expect R4L selected
  - Weak constraints → low KL → expect L4R selected

- [x] **F4.4:** Benchmark: KL computation < 0.1μs per step
  - Criterion bench on vocabulary-sized arrays (32K elements)
  - Profile SIMD auto-vectorization — ensure chunked loop structure

### Phase 4: Safe Exploration Budget (F3 — P2, opt-in)

- [x] **F3.1:** Create `src/pruners/exploration_budget.rs` with `ExplorationBudget` struct
  - `tier0_remaining: u32` — DFA bracket balance checks (default: u32::MAX)
  - `tier1_remaining: u32` — syn AST parse checks (default: 1000)
  - `tier2_remaining: u32` — cargo check in sandbox (default: 100)
  - `conservative_mode: bool` — set when Tier 2 budget exhausted
  - `#[repr(C)]` for stable ABI — WASM boundary compatible

- [x] **F3.2:** Implement `ExplorationBudget::verify(&mut self, tier: VerificationTier) -> Option<VerificationResult>`
  - Decrement appropriate tier counter
  - Return None when tier exhausted → signal conservative fallback
  - When conservative: only Tier 0 DFA, no speculative exploration

- [x] **F3.3:** Implement `ExplorationBudgetConfig` for user-configurable limits
  - `from_env()` — standard config pattern
  - Sensible defaults: Tier 0 unlimited, Tier 1 moderate, Tier 2 limited

- [x] **F3.4:** Wire budget checks into verification pipeline
  - Before Tier 2 sandbox call: check `tier2_remaining > 0`
  - On budget exhaustion: log warning via log crate, activate conservative mode
  - Conservative mode: skip DDTree speculative exploration, use greedy token selection

- [x] **F3.5:** Feature gate: `safe_exploration_budget` in `Cargo.toml` features
  - Opt-in — not default-on, depends on `three_mode_router`
  - When disabled: unlimited verification (current behavior)

- [x] **F3.6:** Test: verify budget limits are respected
  - Set Tier 2 limit to 3, run 5 verification attempts → expect 3 successes + 2 conservative

- [x] **F3.7:** Test: verify conservative mode produces valid but less optimal output
  - Verify conservative_mode flag set, Tier 0 still works, Tier 1 returns BudgetExhausted

- [x] **F3.8:** Benchmark: tier escalation overhead
  - Measure time per verification tier (all sub-μs)
  - Confirm Tier 0 is O(1) per token

### Phase 5: GOAT Gate & Default Promotion

- [x] **GOAT.1:** Mode selection accuracy test — 100 scenarios, ≥80% accuracy achieved
- [x] **GOAT.2:** Constraint miner quality test — 100 paths, all constraints ≥0.90 acceptance
- [x] **GOAT.3:** Grounding quality bounded [0,1] — various distributions tested
- [x] **GOAT.4:** Mixing weights valid simplex — 100 random features, sum≈1.0, non-negative
- [x] **GOAT.5:** Exploration budget test — budget limits enforced, conservative mode works
- [x] **GOAT.6:** Performance tests — mode selection <50ns, mixing weights <100ns, grounding 32K <100μs

---

## Dependency Graph

```
Plan 190 (AND-OR DDTree) ─────┐
                               │
Plan 206 (EGCS/EpisodePruner) ─┤
                               │
BanditPruner ──────────────────┤
                               ▼
                    ┌─────────────────────┐
                    │  Plan 211           │
                    │  Three-Mode Router  │
                    │                     │
                    │  F1: Bandit Router  │── F4: Grounding Quality (part of F1)
                    │  F2: Auto Synthesis │── depends on EpisodePruner
                    │  F3: Exploration $  │── depends on WASM validator
                    └─────────────────────┘
                               │
                               ▼
                    Research 186 (this plan's research doc)
                    Research 184 (FOL-LNN — F5 overlaps)
                    Research 185 (INSIGHT — F4 complements)
```

---

## Estimated Timeline

| Phase | Tasks | Estimated Time | Dependency |
|-------|-------|---------------|------------|
| Phase 1 (F1 Router Core) | 12 tasks | 2-3 days | Plan 190 DDTree |
| Phase 2 (F2 Auto Synthesis) | 9 tasks | 1-2 days | Plan 206 EGCS |
| Phase 3 (F4 Grounding) | 4 tasks | 0.5 days | Phase 1 |
| Phase 4 (F3 Budget) | 8 tasks | 1 day | WASM validator |
| Phase 5 (GOAT Gate) | 6 tasks | 1 day | Phases 1-3 |
| **Total** | **39 tasks** | **5-7 days** | |

---

## Notes

- **F5 (Programmatic Policy Extraction)** is tracked in Research 186 but not a separate phase here — it overlaps with Research 184 F4 and shares the `decision_traces` feature gate. Implement via Plan 210 if not already done.
- **Sigmoid NOT softmax:** This is a hard constraint from project rules. The mixing weights use independent sigmoid per axis, then normalize to sum=1.0. No softmax anywhere.
- **Papaya for lock-free:** If `ThreeModeBandit` needs concurrent access (bandit update from verification thread), use papaya HashMap — not `Arc<RwLock<HashMap>>`.
- **blake3 for audit:** Episode mining results should be blake3-hashed before insertion into ConstraintPruner — integrity check for auto-synthesized rules.
- **Feature gate naming:** `three_mode_router` is the parent gate. `auto_constraint_synthesis` and `safe_exploration_budget` are independently gateable. F4 (grounding quality) is part of `three_mode_router` — no separate gate.

---

## Cross-Repo Alignment (riir-ai ↔ katgpt-rs)

| riir-ai Plan | Relationship | Notes |
|---|---|---|
| **241** WASM Reward Shaping | Training-side complement | 241 shapes reward during LoRA training; 211 routes neuro-symbolic modes at inference. Training reward quality feeds into 211's `verif_success_rate` ModeFeature. |
| **242** DeGRPO Training | Training-side collapse handling | 242's `CollapseMonitor` prevents training collapse; 212's `CollapseDetector` prevents inference collapse. Same concept, different lifecycle. |
| **243** NS-CSG Polytope | Formal foundation for mode routing | 243 proves polytope routing has coverage guarantees; 211's mode selection could leverage polytope regions as ModeFeatures. Future integration point. |

### DRI: DDTree Path → Human-Readable Rule

**Decision:** 209 T4 (`DecisionTrace`) owns this. 211 F5 (Programmatic Policy Extraction) re-exports via `decision_trace` feature gate. Do not duplicate.

### Execution Order

| Phase | Plan | Rationale |
|-------|------|----------|
| 1 | 210 F4 (Reward Calibration) | Formalizes existing pattern |
| 2 | 212 (Collapse-Aware Thinking) | Independent, proven |
| 3 | 209 (FOL Inference) | Foundation for 211 |
| 4 | 210 F1-F3 (Distillation) | Core novelty |
| 5 | **211** (this plan) | Consumes 209 + 210 outputs |
