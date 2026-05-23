# Plan 111: Data Gate — Self-Play Stability via Task-Level Filtering

> **Research:** 075 (Survive or Collapse — Data Gating in Self-Play RL)
> **Paper:** [arXiv:2605.22217](https://arxiv.org/abs/2605.22217) — Pu et al., May 2026
> **Depends:** Plan 059 (GZeroLoop ✅), Plan 093 (CISPO GRPO ✅), Plan 049 (G-Zero ✅)
> **Feature Gate:** `data_gate = ["bandit"]` (modelless path uses bandit infrastructure)
> **Status:** Planning

## Tasks

- [ ] T1: Add `DataGate` trait + `GateDecision` enum to `microgpt-core/src/types.rs`
- [ ] T2: Add `SolverRewardMode` enum to `riir-gpu/src/loss_grpo.rs` (Grounded, IntrinsicSelfConsistency)
- [ ] T3: Implement `ExecutionGate` (sandbox exec + determinism check) in `riir-gpu/src/data_gate.rs`
- [ ] T4: Implement `LeakyGate<G: DataGate>` (ε-Bernoulli relaxation) for phase diagram experiments
- [ ] T5: Wire `DataGate` into `GZeroLoop` — gate tasks BEFORE solver attempts them
- [ ] T6: Add `intrinsic_grounded_gap` metric tracking to `GZeroLoop` round metrics
- [ ] T7: Add `data_gate` feature gate with `#[cfg(feature = "data_gate")]` on all new code
- [ ] T8: GOAT proof — Bomber arena: gate-on vs gate-off with intrinsic solver reward (1000 rounds)
- [ ] T9: Update README, .docs, .research references

---

## Motivation

The paper proves that self-play stability is governed by two **asymmetric** levers:

1. **Data-level gate** `F_ε` — decides which proposer tasks enter training pool
2. **Reward signal** `R` — updates policy on admitted tasks

**The gate is the binding constraint.** A strict gate (ε=0) is sufficient for stability under every reward variant. No reward variant prevents collapse without the gate.

Our current `DeltaFilter` operates at the **wrong level** — it filters preference pairs *after* the solver has attempted them. The paper's gate operates *before* — preventing bad tasks from ever reaching the solver.

```text
Paper:  Proposer → [GATE F_ε] → Training Pool → Solver → Reward → Update
Ours:   Proposer → Solver → Reward → [DeltaFilter] → DPO pairs → Update
After:  Proposer → [DataGate] → Solver → Reward → [DeltaFilter] → DPO pairs → Update
```

Both filters are needed. But the paper proves the **upstream gate** is the binding constraint.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                    GZeroLoop with DataGate                       │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │ Proposer │───▸│ DataGate │───▸│  Solver  │───▸│  Reward   │  │
│  │          │    │  F_ε(τ)  │    │          │    │  R(a, τ)  │  │
│  └──────────┘    └────┬─────┘    └──────────┘    └─────┬─────┘  │
│                       │                                │        │
│                  Admit/Reject                    Grounded/SC     │
│                       │                                │        │
│                       ▼                                ▼        │
│               ┌──────────────┐               ┌───────────────┐  │
│               │ TrainingPool │               │ DeltaFilter   │  │
│               │ (FIFO, cap B)│               │ (6-stage)     │  │
│               └──────┬───────┘               └───────┬───────┘  │
│                      │                               │          │
│                      ▼                               ▼          │
│               ┌──────────────────────────────────────────┐      │
│               │         GRPO / DPO Training              │      │
│               │  (CISPO default, group advantage)        │      │
│               └──────────────────────────────────────────┘      │
│                                                                  │
│  Metrics: intrinsic_grounded_gap, gate_admission_rate,          │
│           pool_size, validation_accuracy                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## T1: `DataGate` Trait

**File:** `microgpt-core/src/types.rs` (shared between both crates)

```rust
/// Task-level admission gate for self-play training pool.
///
/// Decides whether a proposer-generated task should enter the training pool
/// BEFORE the solver attempts it. This is the binding constraint for self-play
/// stability (Survive or Collapse, Pu et al. 2026).
///
/// Key finding: a strict gate is sufficient for stability under every reward
/// variant; no reward variant is sufficient once the gate is removed.
pub trait DataGate {
    /// Admit or reject a proposed task.
    ///
    /// Returns `Admit` if the task passes the gate, `Reject(reason)` if not.
    fn admit(&self, task: &ProposerTask) -> GateDecision;

    /// Current leak rate ε (fraction of failed tasks admitted).
    /// ε=0 means strict gate (optimal). ε=1 means gate off (collapse).
    fn leak_rate(&self) -> f32;
}

/// A task proposed by the self-play proposer, before solver evaluation.
#[derive(Debug, Clone)]
pub struct ProposerTask {
    /// Task identifier for diagnostics
    pub id: usize,
    /// The problem/query text
    pub query: String,
    /// Optional code or DSL expression to execute
    pub program: Option<String>,
    /// Optional input for the program
    pub program_input: Option<String>,
    /// Task type discriminator
    pub task_type: TaskType,
}

/// Discriminator for different self-play task types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Python code output prediction
    CodeIO,
    /// DSL expression evaluation
    DslExpr,
    /// Game action (Bomber, Go, FFT, Monopoly)
    GameAction,
    /// Open-ended generation
    OpenEnded,
}

/// Gate admission decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Task passes the gate — admitted to training pool
    Admit,
    /// Task rejected with reason
    Reject(String),
}
```

---

## T2: `SolverRewardMode` Enum

**File:** `riir-gpu/src/loss_grpo.rs`

```rust
/// Solver reward grounding mode.
///
/// Controls what the solver reward measures:
/// - Grounded: checks answer against executor ground truth (R_S^g)
/// - IntrinsicSelfConsistency: intra-group agreement, no ground truth (R_S^i)
///
/// Paper finding: gate matters more than reward mode, but intrinsic SC
/// with gate-off collapses fastest (Grounded Proposer Paradox).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverRewardMode {
    /// Grounded reward: 1[eval(a) = eval(o*(q))]
    /// Requires deterministic executor. Default and safest.
    #[default]
    Grounded,
    /// Intrinsic self-consistency: (1/n) Σ 1[κ(a^(j)) = κ(a^(i))]
    /// No ground truth needed, but requires strict gate for stability.
    IntrinsicSelfConsistency,
}
```

Add to `GrpoConfig`:

```rust
pub struct GrpoConfig {
    // ... existing fields ...
    /// Solver reward mode: Grounded (default) or IntrinsicSelfConsistency.
    pub solver_reward_mode: SolverRewardMode,
}
```

---

## T3: `ExecutionGate` Implementation

**File:** `riir-gpu/src/data_gate.rs` (new file)

Implements the paper's primary gate: execute the program, check determinism.

```rust
/// Execution-based data gate (paper's primary gate).
///
/// Admits a task only if:
/// 1. Program executes successfully (no crash/timeout)
/// 2. Output is deterministic across two repeated runs
///
/// This is the gate that the paper proves is the binding constraint.
/// ε=0 (strict) is optimal — no reward variant is stable without it.
pub struct ExecutionGate {
    /// Sandbox executor for running programs
    executor: Box<dyn TaskExecutor>,
    /// Whether to check determinism (two runs must agree)
    check_determinism: bool,
    /// Timeout in milliseconds per execution
    timeout_ms: u64,
}

/// Leaky gate wrapper with ε-Bernoulli relaxation.
///
/// Wraps any `DataGate` and admits failed tasks with probability ε.
/// Used for phase diagram experiments (paper Section 4).
///
/// ε=0: strict gate (only passing tasks admitted)
/// ε=0.05: 5% of failed tasks leak through
/// ε=1.0: gate effectively off
pub struct LeakyGate<G: DataGate> {
    inner: G,
    epsilon: f32,
    rng: Rng,
}
```

For game domains (Bomber, Go, FFT), the gate validates:
- Action is syntactically valid (existing `ConstraintPruner::is_valid`)
- Game state transition is legal (executor check)
- No nondeterminism (deterministic game rules)

---

## T4: `LeakyGate<G>` — Phase Diagram Experiments

Wraps any `DataGate` with Bernoulli(ε) relaxation for failed tasks:

```text
F_ε(τ) = 1             if inner.admit(τ) = Admit
F_ε(τ) = Bernoulli(ε)  if inner.admit(τ) = Reject
```

Sweep ε ∈ {0.00, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00} to reproduce paper's phase diagram.

---

## T5: Wire into `GZeroLoop`

**File:** `riir-gpu/crates/riir-gpu/src/gzero_loop.rs`

Modify `GZeroLoop` to gate tasks BEFORE solver attempts:

```text
// Before (current):
for task in proposer.generate(batch) {
    solver.attempt(task) → reward → DeltaFilter
}

// After (with gate):
for task in proposer.generate(batch) {
    match gate.admit(&task) {
        GateDecision::Admit => {
            solver.attempt(task) → reward → DeltaFilter
        }
        GateDecision::Reject(reason) => {
            log::debug!("Gate rejected task {id}: {reason}");
            // Task never reaches solver or training pool
        }
    }
}
```

Add to `GZeroLoop` struct:

```rust
pub struct GZeroLoop {
    // ... existing fields ...
    /// Task-level data gate (binding constraint for stability)
    data_gate: Box<dyn DataGate>,
}
```

---

## T6: `intrinsic_grounded_gap` Metric

**File:** `riir-gpu/src/gzero_loop.rs` (extend `RoundMetrics`)

```rust
/// Difference between self-consistency reward and grounded accuracy.
/// Gap ≈ 1.0 indicates spurious self-consistent attractor (collapse).
/// Gap ≈ 0.0 indicates reward is well-calibrated (stable).
pub intrinsic_grounded_gap: f32,
```

For each round, when using `IntrinsicSelfConsistency` reward:
- Compute intrinsic reward (group agreement)
- Compute grounded accuracy (against executor) as diagnostic
- Report gap = intrinsic_reward - grounded_accuracy
- Alert if gap > 0.5 (early warning of collapse)

---

## T7: Feature Gate

**File:** `riir-gpu/Cargo.toml`

```toml
[features]
default = ["coda_fusion", "asft_loss"]
data_gate = ["bandit"]  # DataGate trait + ExecutionGate + LeakyGate
# ...
```

All new code gated:

```rust
#[cfg(feature = "data_gate")]
pub mod data_gate;

#[cfg(feature = "data_gate")]
pub use data_gate::{ExecutionGate, LeakyGate};
```

The `DataGate` trait itself goes in `microgpt-core/src/types.rs` (ungated — both projects need it, like `ScreeningPruner`).

---

## T8: GOAT Proof — Bomber Arena

**File:** `microgpt-rs/tests/data_gate_bomber.rs` (new test)

Prove the paper's finding in our domain:

```text
Experiment: 1000 rounds Bomber self-play
  A: gate-on  + IntrinsicSelfConsistency reward → should remain stable
  B: gate-off + IntrinsicSelfConsistency reward → should show gap growth
  C: gate-on  + Grounded reward → should remain stable (control)
  D: gate-off + Grounded reward → should collapse (paper: GG+off)

GOAT criteria (≥4/6 pass):
  1. A.win_rate > B.win_rate at round 1000
  2. A.intrinsic_grounded_gap < 0.3 at round 1000
  3. B.intrinsic_grounded_gap > 0.5 at some point during run
  4. C.win_rate ≥ A.win_rate (grounded ≥ intrinsic when both gated)
  5. D shows degradation vs C (gate-off < gate-on)
  6. A.gate_admission_rate > 0 (gate actually filters something)
```

### Phase Diagram Mini-Experiment

Sweep ε ∈ {0.0, 0.2, 0.5, 1.0} with II config:
- Show gap grows with ε
- Show validation holds until ε is high
- Reproduces paper Figure 4 in game domain

---

## Benchmark Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Gate overhead per task | < 1ms | Sandbox execution is the bottleneck |
| Gap metric computation | < 100μs | Simple arithmetic |
| No regression in existing GZeroLoop tests | 0 failures | All existing tests pass with gate always-admitting |

---

## Key Design Decisions

1. **Gate BEFORE solver, not after** — the paper proves this is the binding constraint, not downstream filtering
2. **ε=0 is default** — paper proves strict gate is optimal; no reason to leak bad tasks
3. **Grounded reward is default** — safest option; intrinsic SC only with gate-on
4. **Trait in microgpt-core** — like `ScreeningPruner`, both projects need it
5. **Feature gated** — always gated regardless of GOAT outcome (per project convention)
6. **No training pool yet** — T5 adds gate to per-round flow; persistent pool with FIFO eviction is a future extension (the paper's pool with cap B=16,384)
7. **Game domains get implicit gate** — deterministic game rules provide natural gate; explicit gate catches edge cases

---

## File Changes Summary

| File | Action | Scope |
|------|--------|-------|
| `microgpt-core/src/types.rs` | Add `DataGate`, `GateDecision`, `ProposerTask`, `TaskType` | T1 |
| `riir-gpu/src/loss_grpo.rs` | Add `SolverRewardMode`, extend `GrpoConfig` | T2 |
| `riir-gpu/src/data_gate.rs` | New file: `ExecutionGate`, `LeakyGate<G>` | T3-T4 |
| `riir-gpu/src/gzero_loop.rs` | Wire gate, add gap metric to `RoundMetrics` | T5-T6 |
| `riir-gpu/src/lib.rs` | Add `mod data_gate` + re-exports | T7 |
| `riir-gpu/Cargo.toml` | Add `data_gate` feature | T7 |
| `microgpt-rs/tests/data_gate_bomber.rs` | GOAT proof test | T8 |
| `microgpt-rs/README.md` | Update with DataGate section | T9 |
| `riir-ai/README.md` | Update with DataGate section | T9 |

---

## Relation to Existing Work

| Plan | Relationship |
|------|-------------|
| 049 (G-Zero) | Phase 1 modelless + Phase 2 model-based — gate applies to both |
| 059 (GZeroLoop) | Direct extension — add gate to existing loop |
| 093 (CISPO GRPO) | Orthogonal — gate is data-level, CISPO is loss-level |
| 071 (ROPD) | Orthogonal — rubric distillation, not self-play |
| 073 (SDAR) | Complementary — SDAR is teacher-student, gate is self-play stability |
| 092 (Freeze/Thaw) | Complementary — freeze/thaw manages knowledge, gate manages data quality |
| 034 (Bomber WASM) | Provides sandbox executor for `ExecutionGate` |

---

## References

- Paper: [Survive or Collapse](https://arxiv.org/abs/2605.22217) — Pu et al., 2026
- Raw code: `.raw/survive-or-collapse/`
- Our research: `075_Survive_Or_Collapse_Data_Gating_Self_Play_RL.md`
- Related: Absolute Zero Reasoner, verl GRPO trainer