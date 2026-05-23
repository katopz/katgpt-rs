# Research 075: Survive or Collapse — Data Gating in Self-Play RL

> Source: [Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL](https://arxiv.org/abs/2605.22217) — Pu et al., May 2026
> Raw code: `.raw/survive-or-collapse/`
> Date: 2026-05, distilled 2026-05
> **Verdict: HIGH VALUE — Data Gate is the Missing Piece in GZeroLoop**

## Summary

The paper proves that self-play RL stability is governed by two **asymmetric** levers:

1. **Data-level gate** `F: T → {0,1}` — decides which proposer-generated tasks enter the training pool
2. **Reward signal** `R(a, τ)` — updates the policy on admitted tasks

**Key finding**: The gate is the *binding constraint*. A strict gate is sufficient for stability under **every** reward variant (including self-consistency with no ground truth). No reward variant is sufficient once the gate is removed. This is not obvious — one would expect the reward to naturally down-weight bad data, but self-consistency reward is *maximized* by corrupted data once the solver converges to spurious consensus.

### Table 1 Results (Coding + DSL twin)

| Config | Proposer | Solver | Gate | Coding | DSL | Outcome |
|--------|----------|--------|------|--------|-----|---------|
| GG+exec | grounded | grounded | exec | 0.71 | 0.61 | ✅ stable |
| GI+exec | grounded | intrinsic | exec | 0.67 | 0.63 | ✅ stable |
| II+exec | intrinsic | intrinsic | exec | 0.67 | 0.60 | ✅ stable |
| GG+off | grounded | grounded | off | 0.002 | 0.50 | ❌ collapse (coding) |
| IG+off | intrinsic | grounded | off | 0.006 | 0.58 | ❌ collapse (coding) |
| GI+off | grounded | intrinsic | off | 0.002 | 0.38 | ❌ collapse (both) |
| II+off | intrinsic | intrinsic | off | 0.007 | 0.18 | ❌ collapse (both) |

**Pattern**: Gate-on → always stable. Gate-off → always collapse (coding). DSL grounded-solver holds at baseline because deterministic interpreter acts as implicit gate.

---

## Core Concepts

### 1. Data Gate `F_ε(τ)`

Binary gate with continuous relaxation:

```text
F_ε(τ) = 1            if exec(τ) = 1   (deterministic execution)
F_ε(τ) = Bernoulli(ε) if exec(τ) = 0   (failed/nondeterministic)
```

- `ε = 0`: strict gate (only deterministic tasks admitted) — **optimal**
- `ε = 1`: gate off (everything admitted) — collapse
- Paper sweeps ε ∈ {0.00, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00}

### 2. Reward Variants

**Solver reward**:
- **Grounded**: `R_S^g(a, τ) = 1[eval(a) = eval(o*(q))]` — checks against executor ground truth
- **Intrinsic (self-consistency)**: `R_S^i(a^(i), A(q)) = (1/n) Σ 1[κ(a^(j)) = κ(a^(i))]` — intra-group agreement, no ground truth needed

**Proposer reward**:
- `R_P(τ) = 1 - α̂(τ, π_θ)` — inversely proportional to solver pass rate
- Grounded proposer uses executor output as reference
- Intrinsic proposer uses self-claimed output

### 3. Intrinsic-Grounded Gap

Key diagnostic metric: difference between self-consistency reward and grounded accuracy. When gap ≈ 1.0, the solver has reached a spurious self-consistent attractor decoupled from correctness.

### 4. Grounded Proposer Paradox

Counter-intuitive finding: a **grounded** proposer accelerates collapse *faster* than an ungrounded one when paired with an intrinsic solver. Mechanism: grounded proposer produces clean, well-structured tasks that form the lowest-resistance path to the spurious self-consistent attractor. The proposer doesn't bias toward truth — it sharpens the corridor to failure.

### 5. Two-Stage Phase Transition

Continuous ε sweep reveals:
- **Stage 1** (ε ≈ 0.05): Training-side metrics decouple (gap jumps from 0.16 → 0.44), but validation holds
- **Stage 2** (ε ≈ 0.40): In-domain validation collapses. Mixed aggregate masks this until ε ≈ 0.70

**Practical implication**: Training metrics can look fine while the model is already degrading. Need in-domain probes, not just aggregate metrics.

### 6. Proposer-Capacity Ceiling

Even with strict gate (ε=0), system stagnates when proposer can't generate novel problems. Dataset eligibility drops to ~0.7%. Solution: orthogonal curriculum design, not gate relaxation.

---

## Distillation to Our Architecture

### What We Have (Maps Directly)

| Paper Concept | Our Code | Status |
|---|---|---|
| GRPO group advantage | `loss_grpo::group_advantage()` | ✅ Plan 059 |
| GRPO loss variants | `GrpoLossVariant` (PpoClip, Cispo) | ✅ Plan 093 |
| Proposer trait | `proposer::Proposer` | ✅ Plan 059 |
| Self-play loop | `GZeroLoop` in `riir-gpu` | ✅ Plan 059 |
| Preference pair filter | `DeltaFilter` (6-stage) | ✅ Plan 059 |
| Intrinsic reward (δ) | `HintDelta` via log-prob shift | ✅ Plan 049 |
| Self-consistency | Implicit in GRPO group agreement | ⚠️ Not exposed as reward mode |
| Grounded verification | `Validator` trait, game executors | ✅ Multiple plans |

### What We're Missing (The Gap)

| Paper Concept | Gap | Priority |
|---|---|---|
| **Task-level data gate** `F_ε` | We filter *preference pairs* (DeltaFilter) but never gate *task admission* before training pool | 🔴 Critical |
| **Grounded vs Intrinsic solver reward mode** | `GrpoConfig` has no reward mode enum | 🟡 High |
| **Intrinsic-grounded gap metric** | No diagnostic for self-consistency vs correctness decoupling | 🟡 High |
| **Continuous gate ε** | Binary on/off only, no Bernoulli relaxation | 🟢 Medium |
| **Training pool with replay** | GZeroLoop generates per-round, no persistent pool with FIFO eviction | 🟢 Medium |
| **DSL twin task** | No deterministic controlled environment for ablation | 🟢 Nice-to-have |

### Key Insight for Our System

Our `DeltaFilter` operates at the **wrong level** for the paper's finding. DeltaFilter filters preference pairs *after* the solver has already attempted them. The paper's gate `F_ε` operates *before* — it decides whether a proposer-generated task should even enter the training pool. This is a fundamentally different intervention point:

```text
Paper: Proposer → [GATE F_ε] → Training Pool → Solver → Reward → Update
Us:    Proposer → Solver → Reward → [DeltaFilter] → DPO pairs → Update
```

Both are needed. The gate prevents bad data from ever reaching the solver. DeltaFilter prevents bad preference pairs from reaching DPO. But the paper proves the **gate** is the binding constraint, not the downstream filter.

---

## Architecture Proposal

### DataGate Trait

```rust
/// Task-level admission gate for self-play training pool.
///
/// Decides whether a proposer-generated task should enter the training pool
/// BEFORE the solver attempts it. This is the binding constraint for self-play
/// stability (Survive or Collapse, Pu et al. 2026).
pub trait DataGate {
    /// Admit or reject a proposed task.
    ///
    /// Returns `Admit` if the task passes the gate, `Reject(reason)` if not.
    fn admit(&self, task: &ProposerTask) -> GateDecision;
    
    /// Current leak rate ε (fraction of failed tasks admitted).
    /// ε=0 means strict gate. ε=1 means gate off.
    fn leak_rate(&self) -> f32;
}

pub enum GateDecision {
    Admit,
    Reject(String),
}
```

### Reward Mode Enum

```rust
/// Solver reward grounding mode.
///
/// Asymmetric finding: gate matters more than reward mode,
/// but reward mode affects collapse speed when gate is off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverRewardMode {
    /// Check against executor ground truth (R_S^g)
    Grounded,
    /// Intra-group self-consistency agreement (R_S^i)
    IntrinsicSelfConsistency,
}
```

### Gate Implementations

1. **ExecutionGate** — execute task in sandbox, check determinism (paper's primary gate)
2. **DeterminismGate** — two repeated executions, reject if outputs differ
3. **CompositeGate** — chain multiple gates, all must pass
4. **LeakyGate<G>** — wrap any gate with ε-Bernoulli relaxation for phase diagram experiments

---

## Verdict

**HIGH VALUE — Directly extends GZeroLoop with the missing binding constraint.**

The paper provides strong empirical evidence (7 configurations × 2 tasks + continuous sweep) that:

1. Our current `DeltaFilter` is necessary but insufficient — it operates downstream
2. A task-level `DataGate` is the binding constraint the paper proves is critical
3. The grounded/intrinsic reward axis maps cleanly to our `GrpoConfig`
4. The intrinsic-grounded gap is a cheap diagnostic we should track

**Risk**: Low. The gate is a pure filter — it can only help, never hurt (paper proves ε=0 is optimal).

**Effort**: Medium. Need new trait + impls, wire into `GZeroLoop`, add reward mode to `GrpoConfig`, add gap metric. No GPU kernel changes.

**GOAT proof**: Can be proven on existing Bomber/Go arenas by showing gate-on runs don't collapse while gate-off runs with intrinsic reward do.

---

## References

- Paper: [arXiv:2605.22217](https://arxiv.org/abs/2605.22217)
- Code: `.raw/survive-or-collapse/`
- Builds on: Absolute Zero Reasoner, verl GRPO trainer
- Related our plans: 049 (G-Zero), 059 (GZeroLoop), 093 (CISPO GRPO)
- Related our research: 021 (G-Zero), 037 (REAP Model-Based/Modelless), 061 (SLIME)