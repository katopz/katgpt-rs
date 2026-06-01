# Plan 120: VPD EM-Style Modelless Distillation

> **Status:** ✅ COMPLETE (T1-T12 all implemented, GOAT proofs passing)
> **Branch:** `develop/feature/120_vpd_em_distill`
> **Depends on:** Plan 072 (SDAR gate ✅), Plan 079 (BT rank ✅), Plan 111 (data gate ✅), Plan 030 (BanditPruner ✅)
> **Research:** `.research/080_VPD_Variational_Policy_Distillation.md`
> **Source:** arXiv:2605.15113 — Variational Policy Distillation (Salesforce AI Research, 2026)
> **Feature gate:** `vpd_em_distill` (opt-in, depends on `sdar_gate` + `bandit`)
> **Goal:** Implement VPD's co-evolutionary EM loop for modelless distillation: actively train the feedback-conditioned "teacher" bandit (E-step) instead of treating it as passive, then distill back to the student (M-step) with dynamic trust-region anchoring.

## Summary

VPD identifies a critical flaw in existing self-distillation (SDPO): the feedback-conditioned teacher is **never explicitly trained** — it relies on zero-shot feedback interpretation, which plateaus as the student improves. VPD fixes this with a **Variational EM** cycle:

| Phase | What | Our Analog |
|-------|------|-----------|
| E-step | Train teacher to distinguish success/failure given feedback | Update `SdarBanditPruner` with BCO-style unpaired preference |
| M-step | Distill refined teacher back to student | `SdarGatedAbsorbCompress` KL-gated absorb phase |
| Dynamic prior | Anchor E-step to current student (not frozen baseline) | Use current bandit Q-values as reference, not initial Q₀ |

**Why this matters for modelless:** Our current `sdar_gate` treats the sigmoid-gated "teacher" as a passive signal processor. VPD proves this leads to plateau. By actively refining the teacher's discriminative ability via BCO, we extract more signal from the same feedback.

### What We Already Have (DO NOT reimplement)

| Component | Location | Role |
|-----------|----------|------|
| `SdarBanditPruner<P>` | `src/pruners/sdar/mod.rs` | Sigmoid-gated reward bandit — **becomes E-step target** |
| `SdarGatedAbsorbCompress<P>` | `src/pruners/sdar/mod.rs` | Sigmoid-gated absorb — **becomes M-step distiller** |
| `sdar_gate()` / `sdar_modulate()` | `src/pruners/sdar_gate.rs` | σ(β·x) gate function — reuse as-is |
| `BtConfig` / `BtComparison` | `src/pruners/bt_rank.rs` | BT pairwise ranking — **extend for BCO unpaired** |
| `SdarPlayer` | `src/pruners/bomber/sdar_player.rs` | SDAR bomber player — **add E/M cycle** |
| `BanditPruner<P>` with UCB1 | `src/pruners/bandit.rs` | Core bandit — reuse Q-value tracking |
| `DataGatePlayer` | `src/pruners/bomber/` | Stability gating — VPD dynamic prior is related |
| `AbsorbCompressLayer<P>` | `src/pruners/absorb_compress.rs` | Compression/distillation — reuse as M-step core |

### What's New (Implement)

| Component | Description |
|-----------|-------------|
| `BcoSample` | Unpaired preference sample (reward + implicit reward + feedback embedding) |
| `BcoOptimizer` | BCO-style unpaired preference: σ(r̃ - δ) for positive, σ(-(r̃ - δ)) for negative |
| `VpdConfig` | EM config: E-step frequency F, BCO temperature β, reward shift δ |
| `VpdEmCycle` | The EM loop: alternating E-step (teacher refine) / M-step (student distill) |
| `VpdPlayer` | Bomber arena player using VPD EM cycle (extends SdarPlayer) |
| `DynamicPrior` | Trait for anchoring BCO implicit reward to current student Q-values |

## Tasks

- [x] T1: Implement `BcoSample` and `BcoOptimizer` in `src/pruners/vpd_em.rs`
- [x] T2: Implement `VpdConfig` with paper-validated defaults (F=5, β=0.1, λ=0.01)
- [x] T3: Implement `VpdEmCycle` — the core EM loop with asymmetric frequency
- [x] T4: Implement `DynamicPrior` — replace static π_ref with current π_θ anchoring
- [x] T5: Implement `VpdPlayer` for bomber arena — extends SdarPlayer with E/M phases
- [x] T6: Add feature gate `vpd_em_distill` to `Cargo.toml` (depends on `sdar_gate`, `bandit`)
- [x] T7: Add `bomber_15_vpd_tournament` example — VPD vs SDAR vs GZero vs Random
- [x] T8: GOAT proof — VPD ≥ SDAR on bomber arena (1000 rounds, 4-player)
- [x] T9: GOAT proof — Dynamic prior ≥ Fixed prior (ablation)
- [x] T10: GOAT proof — F=5 ≥ F=1 and F=10 (frequency ablation)
- [x] T11: Update README.md with VPD section
- [x] T12: Update research 080 with benchmark results

## Test Coverage

- `tests/test_120_vpd_em_goat.rs` — 10 GOAT proof tests (BCO loss, shift convergence, E-step frequency, dynamic prior, config defaults, softmax/KL)
- `src/pruners/vpd_em.rs` — 31 internal unit tests (log_sigmoid, softmax, KL divergence, BCO optimizer, VpdConfig, VpdEmCycle)
- `src/pruners/bomber/vpd_player.rs` — 10 internal unit tests (reward, player init, action selection, outcome, reset)
- Run: `cargo test --features vpd_em_distill --test test_120_vpd_em_goat`
- Run: `cargo test --features "vpd_em_distill,bomber" -p katgpt-rs --lib vpd`

## Architecture

### Module Structure

```
src/pruners/
├── vpd_em.rs          # BcoSample, BcoOptimizer, VpdConfig, VpdEmCycle
├── vpd_gate.rs        # Feature gate re-exports
└── bomber/
    └── vpd_player.rs  # VpdPlayer for bomber arena

tests/
└── test_120_vpd_em_distill.rs  # GOAT proofs
```

### Key Types

```rust
// src/pruners/vpd_em.rs

/// BCO unpaired preference sample.
/// Unlike BtComparison (paired winner/loser), BCO operates on individual samples.
pub struct BcoSample {
    /// Index of the action/template being evaluated.
    pub action_idx: usize,
    /// Binary outcome reward: 1.0 for success, 0.0 for failure.
    pub outcome: f32,
    /// Implicit reward: β · log(q_φ / π_θ) with dynamic prior.
    pub implicit_reward: f32,
    /// Feedback signal (scalar, from sdar_gate or similar).
    pub feedback_signal: f32,
}

/// BCO optimizer for unpaired preference learning.
/// Implements: L = -E_{y+}[log σ(r̃ - δ)] - E_{y-}[log σ(-(r̃ - δ))]
pub struct BcoOptimizer {
    /// BCO temperature (paper: β=0.1).
    pub temperature: f32,
    /// Moving average reward shift δ.
    pub reward_shift: f32,
    /// EMA momentum for δ update.
    pub shift_momentum: f32,
}

/// VPD EM configuration.
/// Defaults from paper Table C.1 and Table C.2.
pub struct VpdConfig {
    /// E-step frequency: 1 E-step per F M-steps (paper: F=5).
    pub e_step_frequency: usize,
    /// BCO temperature β (paper: 0.1).
    pub bco_temperature: f32,
    /// KL penalty strength (paper: β in Eq. 7, typically 0.1-1.0).
    pub kl_penalty: f32,
    /// Whether to use dynamic prior (π_θ) vs fixed prior (π_ref).
    /// Paper ablation: dynamic 74.34 vs fixed 67.84 — always use dynamic.
    pub dynamic_prior: bool,
}

/// VPD EM cycle state machine.
/// Alternates between E-step (teacher refinement) and M-step (student distillation).
pub struct VpdEmCycle<P: ScreeningPruner> {
    config: VpdConfig,
    /// E-step optimizer (BCO).
    bco: BcoOptimizer,
    /// M-step counter — triggers E-step every F M-steps.
    m_step_count: usize,
    /// Current student Q-values (dynamic prior).
    student_q: Vec<f32>,
    /// Teacher Q-values (feedback-conditioned, updated in E-step).
    teacher_q: Vec<f32>,
    /// Reference Q-values (frozen from initial state, for fixed-prior ablation).
    reference_q: Vec<f32>,
    /// Collected samples for E-step batch.
    e_step_buffer: Vec<BcoSample>,
    /// Phantom for ScreeningPruner.
    _phantom: PhantomData<P>,
}
```

### E-Step: Teacher Refinement (BCO)

```rust
impl BcoOptimizer {
    /// Compute BCO loss for a batch of unpaired samples.
    /// L = -Σ_{y+} log σ(r̃ - δ) - Σ_{y-} log σ(-(r̃ - δ))
    pub fn compute_loss(&self, samples: &[BcoSample]) -> f32 {
        let mut loss = 0.0f32;
        for s in samples {
            let r_tilde = s.implicit_reward - self.reward_shift;
            if s.outcome > 0.5 {
                // Positive sample: log σ(r̃ - δ)
                loss -= log_sigmoid(r_tilde / self.temperature);
            } else {
                // Negative sample: log σ(-(r̃ - δ))
                loss -= log_sigmoid(-r_tilde / self.temperature);
            }
        }
        loss / samples.len().max(1) as f32
    }

    /// Update reward shift δ via EMA.
    /// δ = 0.5 · (E[r̃(y+)] + E[r̃(y-)])
    pub fn update_shift(&mut self, samples: &[BcoSample]) {
        let (pos_sum, neg_sum, pos_n, neg_n) = samples.iter().fold(
            (0.0f32, 0.0f32, 0usize, 0usize),
            |(ps, ns, pn, nn), s| {
                if s.outcome > 0.5 {
                    (ps + s.implicit_reward, ns, pn + 1, nn)
                } else {
                    (ps, ns + s.implicit_reward, pn, nn + 1)
                }
            },
        );
        let pos_avg = if pos_n > 0 { pos_sum / pos_n as f32 } else { 0.0 };
        let neg_avg = if neg_n > 0 { neg_sum / neg_n as f32 } else { 0.0 };
        let target = 0.5 * (pos_avg + neg_avg);
        self.reward_shift = self.shift_momentum * self.reward_shift
            + (1.0 - self.shift_momentum) * target;
    }
}
```

### M-Step: Student Distillation (KL-Gated Absorb)

The M-step reuses existing `SdarGatedAbsorbCompress` but with teacher Q-values from the E-step as the target distribution. The KL divergence is computed at the **action level** (not token level, since we're modelless):

```rust
impl<P: ScreeningPruner> VpdEmCycle<P> {
    /// M-step: Distill teacher → student via KL-gated absorb-compress.
    /// Action-level KL: D_KL(π_student || q_teacher) at each action.
    pub fn m_step(&mut self, action_idx: usize, reward: f32, absorb: &mut SdarGatedAbsorbCompress<P>) {
        // Compute action-level KL divergence as gating signal
        let student_prob = softmax(&self.student_q);
        let teacher_prob = softmax(&self.teacher_q);
        let kl_div = kl_divergence(&student_prob, &teacher_prob);

        // Gate the distillation signal using SDAR sigmoid
        let gate = sdar_gate(kl_div, SDAR_BETA);

        // Absorb with gated signal (teacher as target)
        let teacher_q_val = self.teacher_q.get(action_idx).copied().unwrap_or(0.0);
        absorb.observe_with_q(action_idx, reward * gate, teacher_q_val);

        // Update student Q-values (dynamic prior for next E-step)
        self.student_q[action_idx] = self.student_q[action_idx]
            .lerp(teacher_q_val, 0.1); // η=0.1 soft update

        self.m_step_count += 1;
    }
}
```

### VpdPlayer for Bomber Arena

```rust
// src/pruners/bomber/vpd_player.rs

/// Bomber arena player using VPD EM-style co-evolutionary distillation.
///
/// Architecture (extends SdarPlayer with E/M cycle):
/// VpdPlayer
///   ├── BomberTemplateProposer     (UCB1 template selection — shared with GZero/Rubric/SDAR)
///   ├── VpdEmCycle                 (E-step BCO + M-step KL-gated absorb)
///   │   ├── BcoOptimizer           (unpaired preference for teacher refinement)
///   │   └── SdarGatedAbsorbCompress (KL-gated distillation for student)
///   ├── Cross-round Q-values       (action-level bandit memory)
///   └── DynamicPrior               (current student Q-values as E-step anchor)
///
/// The EM cycle runs within the self-play loop:
/// 1. Select template → choose action → observe outcome
/// 2. M-step: update student via KL-gated absorb (every round)
/// 3. E-step: refine teacher via BCO (every F=5 rounds)
pub struct VpdPlayer {
    _id: u8,
    // Game state tracking (same as SdarPlayer)
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<KnownOpponent>,
    last_dir: Option<BomberAction>,
    alive: bool,
    powerups_collected: u32,
    // VPD components
    template_proposer: BomberTemplateProposer,
    em_cycle: VpdEmCycle<NoScreeningPruner>,
    // Cross-round memory
    round_actions: Vec<(BomberAction, f32)>,
    round_template_ids: Vec<usize>,
    round_count: usize,
}
```

## GOAT Proof Strategy

### Proof 1: VPD ≥ SDAR (Main Result)

```
Hypothesis: VpdPlayer win rate ≥ SdarPlayer win rate in 4-player bomber arena.

Setup:
- 1000 rounds, 4-player games
- Players: [VpdPlayer, SdarPlayer, GZeroPlayer, RandomPlayer]
- Metric: win rate, ELO, survival rate

Expected: VPD ≥ SDAR by 3-8% (paper shows ~4% average gain across benchmarks).
Reasoning: E-step actively improves teacher discriminative ability → better signal quality → faster convergence.
```

### Proof 2: Dynamic Prior ≥ Fixed Prior (Ablation)

```
Hypothesis: VpdPlayer with dynamic_prior=true ≥ dynamic_prior=false.

Setup:
- Same as Proof 1 but compare VPD-dynamic vs VPD-fixed
- Paper shows 74.34 vs 67.84 on SciKnowEval (1.7B)
- Expect similar magnitude gain in bomber arena

Expected: Dynamic prior wins by 5-10% (larger than paper due to faster game rounds).
```

### Proof 3: F=5 Optimal (Frequency Ablation)

```
Hypothesis: VPD with F=5 ≥ F=1 and F=5 ≥ F=10.

Setup:
- Compare three VpdPlayer variants: F=1, F=5, F=10
- Same arena setup, 1000 rounds

Expected: F=5 > F=1 (volatile teacher) and F=5 > F=10 (stale teacher).
Paper validates: 74.34 (F=5) > 70.21 (F=1) > 69.27 (F=10).
```

## Feature Gate

```toml
# Cargo.toml
[features]
vpd_em_distill = ["sdar_gate", "bandit"]  # VPD EM-style co-evolutionary distillation (Plan 120, Research 080)
```

```rust
// src/pruners/mod.rs
#[cfg(feature = "vpd_em_distill")]
pub mod vpd_em;

// src/pruners/bomber/mod.rs
#[cfg(feature = "vpd_em_distill")]
pub mod vpd_player;
```

## Implementation Notes

### Key Differences from Paper

1. **Action-level KL instead of token-level KL.** We operate on discrete action spaces (≤20 templates), not continuous token distributions. KL divergence is computed over the template/action probability vector, not token sequences.

2. **Bandit Q-values as distribution proxy.** We don't have π_θ(·|x) explicitly. Instead, we use the bandit's Q-value vector (one Q per template/action) as the distribution. Softmax(Q) approximates π_θ.

3. **No shared-weight optimization.** The paper's φ=θ trick (same network, different conditioning) is trivially satisfied because teacher and student ARE the same bandit — just different Q-value snapshots (current vs reference).

4. **Scalar feedback instead of language feedback.** Our "diagnostic feedback" is a scalar reward signal (survival, powerups, danger) rather than textual critique. The BCO framework still applies — we just use `outcome_reward` instead of `C` (language critique).

### Numerical Stability

- Use numerically stable `log_sigmoid(x) = -softplus(-x)` to avoid overflow
- Clamp implicit reward r̃ to [-10, 10] before BCO loss computation
- Use EMA for reward shift δ (momentum=0.9) instead of raw batch average

### Dependencies

| Crate | Usage |
|-------|-------|
| `fastrand` | RNG for softmax sampling |
| Existing `sdar` module | Reuse `SdarBanditPruner`, `SdarGatedAbsorbCompress` |
| Existing `sdar_gate` | Reuse `sdar_gate()`, `SDAR_BETA` |
| Existing `bt_rank` | Extend `BtConfig` pattern for BCO config |

## Benchmark Plan

```
# T8: Main GOAT proof
cargo run --example bomber_15_vpd_tournament --features vpd_em_distill,sdar_gate,ropd_rubric,g_zero,bomber

# Expected output:
# ┌──────────────────────────────────────────────────────────┐
# │  VPD EM Distillation Tournament (Plan 120)              │
# │  Players: VPD, SDAR, GZero, Rubric, HL, Random         │
# │  1000 rounds × 6 matchups                               │
# ├──────────────────────────────────────────────────────────┤
# │  VPD ≥ SDAR (win rate)  → GOAT ✅                       │
# │  Dynamic ≥ Fixed prior  → GOAT ✅                       │
# │  F=5 ≥ F=1, F=10        → GOAT ✅                       │
# └──────────────────────────────────────────────────────────┘
```

## References

- **Paper:** arXiv:2605.15113 — Learning from Language Feedback via Variational Policy Distillation
- **BCO:** arXiv:2505.xxxxx — Binary Classifier Optimization for LLM Alignment
- **SDPO:** arXiv:2601.20802 — Reinforcement Learning via Self-Distillation (our SDAR base)
- **GRPO:** arXiv:2402.03300 — DeepSeekMath (our bandit UCB1 analog)
- **Related plans:** Plan 072 (SDAR gate), Plan 079 (BT rank), Plan 111 (data gate), Plan 071 (ROPD rubric)