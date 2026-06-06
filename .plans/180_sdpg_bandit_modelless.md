# Plan 180: SDPG Bandit — Modelless Self-Distilled Policy Gradient

**Branch:** `develop/feature/180_sdpg_bandit_modelless`
**Depends on:** Plan 030 (Bandit), Plan 049 (G-Zero Phase 1), Plan 071 (ROPD Rubric), Plan 072 (SDAR Gate)
**Research:** `.research/160_SDPG_Self_Distilled_Policy_Gradient.md`
**Goal:** Distill SDPG's Proposition 3.1 (full-vocab OPD ≡ policy gradient with centered log-ratio advantage) into modelless bandit credit assignment. Dense per-arm signal from oracle-informed teacher Q-values, positive-advantage gating, β warmup-decay schedule, and unnormalized KL anchoring. No neural model — pure bandit computation.

**Key Insight:** SDPG proves OPD is equivalent to REINFORCE with `A_dist(a) = SG[D̄ - log(p̄/q̄)]`. In modelless land, `p̄ = softmax(Q_oracle/τ)` and `q̄ = softmax(Q_student/τ)`. Arena replay data IS the privileged context — oracle knows which arms won, student doesn't. The centered log-ratio provides dense per-arm credit where current bandits only get sparse game-outcome reward.

**Why now:** SDPG shows +35% on hard benchmarks from dense credit assignment. Our bandits have the same sparse→dense gap. The mathematical foundation (Proposition 3.1) is proven — we're implementing a theorem, not a heuristic.

---

## Tasks

### Phase 0: Benchmark Baseline (MUST DO FIRST)

- [x] T1: Create benchmark test — `tests/bench_sdpg_bandit_modelless.rs`
  - Baseline: existing `BanditPruner` with UCB1 (scalar δ reward)
  - Compare: `SdpgBanditPruner` with oracle-informed centered log-ratio advantage
  - Metrics: bandit regret convergence, optimal arm selection rate, Q-value stability over 1000 games
  - Domains: Bomber arena (action space ~5-10 arms)
  - Oracle: load arena replay data from `bomber_04_replay_gen` output
  - **Gate:** SDPG Bandit must converge in ≤ same episodes OR show higher final reward

### Phase 1: Core Computation — Centered Log-Ratio Advantage

The mathematical heart of SDPG — Proposition 3.1 translated to bandits.

- [x] **T2: Implement `SdpgAdvantage` trait + `centered_log_ratio` function** — `src/pruners/sdpg/advantage.rs`
  ```rust
  //! SDPG Proposition 3.1: centered log-ratio advantage for bandits.
  //!
  //! A(a) = D̄ - log(p̄(a)/q̄(a)) where:
  //! - p̄ = softmax(teacher_q / τ)  [oracle-informed distribution]
  //! - q̄ = softmax(student_q / τ)  [bandit-learned distribution]
  //! - D̄ = KL(p̄ || q̄)            [centering constant]

  /// Compute centered log-ratio advantage for each arm.
  ///
  /// Returns Vec<f32> of advantages. Positive = student underestimates arm
  /// relative to oracle (should explore more). Negative = overestimates.
  ///
  /// # Arguments
  /// * `student_q` - Bandit Q-values (student distribution)
  /// * `teacher_q` - Oracle Q-values from replay (teacher distribution)
  /// * `temperature` - Softmax temperature τ (lower = sharper distribution)
  pub fn centered_log_ratio(
      student_q: &[f32],
      teacher_q: &[f32],
      temperature: f32,
  ) -> Vec<f32> {
      assert_eq!(student_q.len(), teacher_q.len());
      let n = student_q.len();

      // Softmax with temperature
      let p_bar = softmax_scaled(teacher_q, temperature); // teacher
      let q_bar = softmax_scaled(student_q, temperature); // student

      // KL divergence: D̄ = Σ p̄(a) * log(p̄(a)/q̄(a))
      let d_bar: f32 = p_bar.iter().zip(q_bar.iter())
          .map(|(&p, &q)| {
              if p > 0.0 && q > 0.0 { p * (p / q).ln() } else { 0.0 }
          })
          .sum();

      // Centered log-ratio: A(a) = D̄ - log(p̄(a)/q̄(a))
      p_bar.iter().zip(q_bar.iter())
          .map(|(&p, &q)| {
              let log_ratio = if p > 0.0 && q > 0.0 {
                  (p / q).ln()
              } else if p > q {
                  f32::MAX // teacher much higher → strong positive advantage
              } else {
                  f32::MIN // student much higher → strong negative advantage
              };
              d_bar - log_ratio
          })
          .collect()
  }

  fn softmax_scaled(logits: &[f32], temperature: f32) -> Vec<f32> {
      let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
      let exps: Vec<f32> = logits.iter()
          .map(|&v| ((v - max_val) / temperature).exp())
          .collect();
      let sum: f32 = exps.iter().sum();
      exps.iter().map(|&e| e / sum).collect()
  }
  ```

- [x] **T3: Implement `BetaSchedule` struct** — `src/pruners/sdpg/schedule.rs`
  ```rust
  /// β warmup-decay schedule from SDPG paper.
  ///
  /// Warmup: linearly ramp β from 0 to β_base over warmup_steps.
  /// Decay: linearly decay β from β_base to 0 after warmup until decay_steps.
  /// After decay_steps: β = 0 (teacher fully phased out).
  pub struct BetaSchedule {
      pub beta_base: f32,
      pub warmup_steps: usize,
      pub decay_steps: usize,
      pub current_step: usize,
  }

  impl BetaSchedule {
      pub fn new(beta_base: f32, warmup_steps: usize, decay_steps: usize) -> Self {
          Self { beta_base, warmup_steps, decay_steps, current_step: 0 }
      }

      pub fn beta(&self) -> f32 { /* ... see R160 sketch ... */ }
      pub fn step(&mut self) { self.current_step += 1; }
      pub fn reset(&mut self) { self.current_step = 0; }
  }
  ```

### Phase 2: SDPG Bandit Pruner

Wrap existing `BanditPruner` with oracle-informed teacher and centered log-ratio advantage.

- [x] **T4: Implement `SdpgBanditPruner<P>`** — `src/pruners/sdpg/mod.rs`
  ```rust
  //! SDPG Bandit Pruner — modelless self-distilled policy gradient.
  //!
  //! Wraps BanditPruner with:
  //! - Oracle-informed teacher Q-values (from arena replay)
  //! - Centered log-ratio advantage (Proposition 3.1)
  //! - Positive-advantage gating (only when arena outcome > 0)
  //! - β warmup-decay schedule (phase out teacher influence)

  pub struct SdpgBanditPruner<P: ScreeningPruner> {
      inner: BanditPruner<P>,
      teacher_q: Vec<f32>,
      ref_q: Vec<f32>,
      schedule: BetaSchedule,
      anchor: KlAnchor,
      temperature: f32,
  }
  ```
  - `update(arm, reward, arena_outcome)`: compute centered log-ratio advantage, gate by `arena_outcome > 0`, modulate by `β * advantage`, apply KL anchor, update inner bandit
  - `select()`: delegate to inner bandit's UCB1
  - `relevance()`: delegate to inner bandit (ScreeningPruner impl)
  - Teacher Q-values loaded from replay data at construction time
  - Reference Q-values snapshot at construction (for KL anchoring)

- [x] **T5: Implement positive-advantage gating** — in `SdpgBanditPruner::update`
  - `m_i = 1[arena_outcome > 0]` — only apply SDPG signal when game was won
  - When `m_i = 0`: only apply KL anchor (stability), no teacher signal
  - Reuse arena outcome signal from existing bomber replay infrastructure

### Phase 3: KL Anchoring

Unnormalized KL regularization for bandit Q-values.

- [x] **T6: Implement `KlAnchor` enum** — `src/pruners/sdpg/anchor.rs`
  ```rust
  //! Unnormalized KL anchoring for bandit Q-values.
  //!
  //! From SDPG paper: anchors policy to frozen reference via UFKL or URKL.
  //! Prevents Q-value collapse (mode collapse analog) in long self-play.

  pub enum KlAnchor {
      /// Unnormalized Forward KL — mass-corrected anchoring.
      /// L = β * Σ [Q_ref(a)/Q(a) + log(Q(a)/Q_ref(a))]
      Ufkl { beta: f32 },

      /// Unnormalized Reverse KL — variance-bounded, mode-seeking.
      /// L = β * 0.5 * Σ [log(Q(a)/Q_ref(a))]²
      Urkl { beta: f32 },
  }
  ```
  - `anchor_loss(q, q_ref) -> Vec<f32>`: per-arm anchoring adjustment
  - Default: `Urkl { beta: 0.01 }` — variance-bounded, mode-seeking (paper recommends this variant)
  - Applied every update as Q-value adjustment (subtract from Q)

### Phase 4: Feature Gate + Module Wiring

- [x] **T7: Add feature flag `sdpg_bandit`** — `Cargo.toml` + `src/pruners/mod.rs`
  ```toml
  [features]
  sdpg_bandit = []  # SDPG bandit + KL anchoring (default-on candidate)
  ```
  - All new types behind `#[cfg(feature = "sdpg_bandit")]`
  - New module: `src/pruners/sdpg/` with `mod.rs`, `advantage.rs`, `schedule.rs`, `anchor.rs`
  - Export: `SdpgBanditPruner`, `SdpgAdvantage`, `BetaSchedule`, `KlAnchor`

### Phase 5: Arena Example

- [x] T8: Create bomber arena example — `examples/bomber_18_sdpg_tournament.rs`
  - Players: `SdpgPlayer` (SDPG Bandit), `HLPlayer`, `GZeroPlayer`, `RubricPlayer`, `SdarPlayer`, `RandomPlayer`
  - Oracle: load replay data from `bomber_04_replay_gen` output
  - 50 games × 15 matchups = 750 total games
  - Metrics: ELO rating, win rate, head-to-head record, Q-value convergence speed
  - Compare SDPG Bandit convergence vs HL vs GZero vs Rubric vs SDAR
  - **Gate:** SDPG > HL > Random (must beat heuristic learning baseline)

### Phase 6: Unit Tests

- [x] **T9: Unit tests for all components** — `src/pruners/sdpg/` inline test modules (21/21 passing)
  - `centered_log_ratio` tests:
    - Identical Q-values → all advantages ≈ 0
    - Teacher strongly prefers arm A → positive advantage for A
    - Student strongly prefers arm B → negative advantage for B
    - Sum of advantages ≈ 0 (centering property)
    - Temperature sensitivity: low τ → sharper, high τ → more uniform
  - `BetaSchedule` tests:
    - At step 0: β = 0
    - At warmup_steps: β = β_base
    - At decay_steps: β = 0
    - Mid-decay: 0 < β < β_base
  - `KlAnchor` tests:
    - UFKL: Q = Q_ref → loss = 0
    - UFKL: Q diverges from Q_ref → loss > 0
    - URKL: Q = Q_ref → loss = 0
    - URKL: numerical stability with Q near zero
  - `SdpgBanditPruner` tests:
    - Converges to optimal arm (oracle-endorsed)
    - Positive-advantage gating: no teacher signal on losing outcomes
    - KL anchor prevents Q-value collapse
    - β schedule phases out teacher influence

### Phase 7: GOAT Proof

- [x] T10: Run arena, verify SDPG > HL > Random — **NEGATIVE RESULT**: SDPG 12% win rate < HL 29.6%. Root cause: uniform teacher Q-values (no oracle data). Keep as opt-in infrastructure. See `.benchmarks/011_sdpg_bandit_arena.md`

### Phase 8: Sigmoid + RawDelta Fix (post-mortem)

Root cause analysis showed softmax-based KL has poor resolution for 5-10 bandit arms. Two alternative advantage modes added:

- [x] T11: Add `sigmoid_advantage()` — per-arm σ(teacher/τ) - σ(student/τ), no cross-arm normalization
- [x] T12: Add `raw_delta_advantage()` — simplest teacher_q - student_q per arm
- [x] T13: Add `AdvantageMode` enum with `Sigmoid` as default (per AGENTS.md: use sigmoid not softmax)
- [x] T14: Wire `AdvantageMode` into `SdpgBanditPruner::update()` dispatch
- [x] T15: Add `from_replay()` constructor — loads teacher Q from replay JSONL via `ReplaySample::from_json`
- [x] T16: Add `SdpgPlayer::with_replay()` — constructs player with oracle replay data
- [x] T17: All 30 SDPG lib tests + 9 sdpg_player tests passing

### Phase 9: Oracle Pipeline + Bug Fix

- [x] T18: Fix critical bug — `update_if_sdpg` missing from `arena_runner.rs`, SDPG bandit never learned from outcomes. This alone improved SDPG from 12% → 15.3%.
- [x] T19: Add `template_id: u8` to `ReplaySample` (backward compat, serde default=255)
- [x] T20: Add `SdpgPlayer::with_teacher_q()` constructor for oracle Q injection
- [x] T21: Add `SdpgPlayer::sdpg_bandit()` accessor for Q-value extraction
- [x] T22: Create `bomber_19_sdpg_replay_gen` — two-phase burn-in + GOAT gate example
- [x] T23: Run oracle pipeline — **STILL NEGATIVE**: SDPG(oracle) 14% < HL 28%
  - Root cause: all 8 templates converge to Q~0.88 (variance <0.04) — domain has no meaningful template-level differentiation
  - Bomber outcomes depend on action execution (safety filter, bomb placement), not template choice
  - SDPG's template-level oracle signal is the wrong abstraction for this domain
  - **update_if_sdpg fix** was the real win (12% → 15.3%)

---

## Files Modified

| File | Changes |
|------|---------|
| `src/pruners/sdpg/mod.rs` | **New:** `SdpgBanditPruner<P>` wrapper |
| `src/pruners/sdpg/advantage.rs` | **New:** `centered_log_ratio` + `softmax_scaled` |
| `src/pruners/sdpg/schedule.rs` | **New:** `BetaSchedule` |
| `src/pruners/sdpg/anchor.rs` | **New:** `KlAnchor` (UFKL + URKL) |
| `src/pruners/mod.rs` | Add `pub mod sdpg;` behind `#[cfg(feature = "sdpg_bandit")]` |
| `Cargo.toml` | Add `sdpg_bandit = []` feature |
| `examples/bomber_11_sdpg_tournament.rs` | **New:** Arena benchmark |
| `tests/bench_sdpg_bandit_modelless.rs` | **New:** Component benchmarks |

## Feature Gate

```toml
[features]
sdpg_bandit = []  # SDPG bandit + KL anchoring (default-on candidate)
```

All new code behind `#[cfg(feature = "sdpg_bandit")]`. Composes with existing features: `bandit`, `g_zero`, `ropd_rubric`, `sdar_gate`.

## Hyperparameter Guide

| Parameter | Default | Range | Effect |
|---|---|---|---|
| temperature (τ) | 1.0 | [0.1, 10.0] | Softmax sharpness. Low = sharper distribution matching, high = softer |
| β_base | 0.1 | [0.01, 1.0] | Max teacher influence strength |
| warmup_steps | 100 | [50, 500] | Games before teacher at full strength |
| decay_steps | 1000 | [500, 5000] | Games until teacher fully phased out |
| KlAnchor beta (UFKL) | 0.01 | [0.001, 0.1] | KL anchoring strength (forward) |
| KlAnchor beta (URKL) | 0.01 | [0.001, 0.1] | KL anchoring strength (reverse, recommended) |

## Design Decisions

1. **Wrap, don't replace** — `SdpgBanditPruner<P>` wraps `BanditPruner<P>`. Same pattern as `RubricBanditPruner`, `DeltaBanditPruner`. Opt-in, revertible.
2. **Oracle from replay** — Teacher Q-values loaded from arena replay data (already generated by `bomber_04_replay_gen`). No new data pipeline needed.
3. **Softmax conversion** — Q-values aren't distributions, so we softmax them before computing KL. Temperature τ controls sharpness.
4. **UFKL vs URKL default** — URKL is variance-bounded (no division by Q), recommended as default. UFKL available for mass-correction scenarios.
5. **Reference Q-table = snapshot** — Frozen copy of initial Q-values. Simplest anchor. Can be upgraded to heuristic-derived or cross-validated later.

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Bandit regret convergence | ≤ same episodes as baseline | T9 unit tests |
| Arena ELO | SDPG > HL > Random | T10 GOAT proof |
| Q-value stability | No collapse over 5000 games | T9 anchor test |
| Latency impact | ≤1% increase per game | T1 benchmark |
| Composability | Works with bandit, g_zero, ropd_rubric features | T7 feature gate |

## Failure Mode

If SDPG Bandit shows no arena improvement over HL baseline:
- Document negative result in `.research/160_SDPG_Self_Distilled_Policy_Gradient.md`
- Keep `sdpg_bandit` module as infrastructure — centered log-ratio may help in other contexts (e.g., DDTree relevance matching)
- The KL anchoring (`KlAnchor`) may still be independently useful for bandit stability
- The mathematical structure (Proposition 3.1) is correct — the issue would be in the modelless approximation, not the theorem

## Relationship to Related Plans

| Plan | Relationship |
|------|-------------|
| Plan 071 (ROPD Rubric) | Rubric vectors are per-criterion; SDPG is full-distribution. Complementary: Rubric for quality axes, SDPG for credit assignment |
| Plan 072 (SDAR Gate) | SDAR gates signal intensity; SDPG provides the signal structure. Composable: SDPG advantage × SDAR gate |
| Plan 052 (GFlowNet) | GFlowNet adds flow bonus to bandit reward; SDPG adds oracle-informed advantage. Both improve bandit convergence, different mechanisms |
| Plan 049 (G-Zero Phase 1) | G-Zero provides δ-reward; SDPG adds teacher-informed advantage on top. SDPG Bandit uses δ as arena_outcome signal for gating |

## Timeline

- Phase 0 (T1): 1 day
- Phase 1 (T2-T3): 1 day
- Phase 2 (T4-T5): 1.5 days
- Phase 3 (T6): 0.5 day
- Phase 4 (T7): 0.5 day
- Phase 5 (T8): 1 day
- Phase 6 (T9): 1 day
- Phase 7 (T10): 1 day
- **Total: ~7.5 days**

## References

- SDPG paper: https://arxiv.org/abs/2606.04036
- Research doc: `.research/160_SDPG_Self_Distilled_Policy_Gradient.md`
- SDAR research: `.research/038_SDAR_Self_Distilled_Agentic_RL.md`
- ROPD research: `.research/036_ROPD_Rubric_OnPolicy_Distillation.md`
- Bandit infrastructure: `src/pruners/bandit.rs`
- Arena replay: `examples/bomber_04_replay_gen.rs`
