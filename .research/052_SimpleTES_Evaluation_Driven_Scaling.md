# Research 52: SimpleTES — Evaluation-Driven Scaling for Modelless Architecture

> **Paper:** [Evaluation-driven Scaling for Scientific Discovery (SimpleTES)](https://arxiv.org/abs/2604.19341) — arXiv:2604.19341
> **Date:** 2026-04, distilled 2026-07
> **Related Research:** 21 (G-Zero δ Signal), 23 (GFlowNet Paths), 37 (REAP Model-Based/Modelless), 35 (Attractor Fixed-Point)
> **Related Plans:** Plan 042 (TTT Feedback Loop), Plan 048 (Self-Improving Loop)
> **Feature Gate:** `tes_loop = ["bandit"]`

---

## TL;DR

SimpleTES organizes test-time compute as (C, L, K, Φ) — global width, refinement depth, local sample size, and proposal constructor from history. Default: C=32, L=100, K=16, budget N=C×L×K=51.2K. Using open-source `gpt-oss-120b`, it achieves SOTA across 21 scientific problems, beating frontier model baselines. Key results: 2× LASSO speedup, 24.5% quantum circuit routing improvement, new Erdős minimum overlap constructions.

Post-training uses trajectory-level histories as supervision with max-trajectory-score credit assignment (not per-step reward), producing generalizable discovery behaviors.

**Verdict: HIGH VALUE for modelless stack. SimpleTES proves evaluation-driven loops with simple policies (no neural training at inference) beat frontier models. Their (C, L, K) allocation is our `BanditPruner` at trajectory granularity. Four actionable distillations: RPUCG bandit variant, TES loop trait, trajectory-level pruning, credit assignment bridge.**

---

## 1. The (C, L, K, Φ) Decomposition

| Dimension | Symbol | Role | Default |
|-----------|--------|------|---------|
| Global width | C | Parallel trajectories | 32 |
| Refinement depth | L | Iterations per trajectory | 100 |
| Local sample size | K | Candidates per step | 16 |
| Proposal constructor | Φ | History → inspiration selection | RPUCG |

Total budget: N = C × L × K = 51,200 evaluator queries.

This maps directly to our speculative decoding and bandit architecture:

| SimpleTES | Our Stack |
|-----------|-----------|
| C (global width) | `speculate()` draft batch size |
| L (refinement depth) | `max_depth` / tree depth |
| K (local sample size) | `BanditPruner` arms per node |
| Φ (proposal selector) | `BanditPruner<P>` strategy selection |

---

## 2. RPUCG — Graph-Based Bandit (maps to Φ)

Their Relative Propagation Upper Confidence Graph is our `BanditPruner` at trajectory level:

```microgpt-rs/.research/052_SimpleTES_Evaluation_Driven_Scaling.md#L50-58
// Propagated value: max(r_i, γ·max_child_U) — like our AbsorbCompress
// Exploration bonus: λ·ρ·√(1+|S|)/(1+n_i) — like our UCB bandit
// Greedy selection excluding one-hop neighbors — diversity enforcement
//
// UCB_i = V_i + λ · ρ · √(1 + |S|) / (1 + n_i)
// where V_i = max(r_i, γ · max(U_child))
```

### Mapping to Our Components

| RPUCG Component | Our Equivalent |
|-----------------|----------------|
| Propagated value `V_i = max(r_i, γ·max_child_U)` | `AbsorbCompress` heuristic promotion |
| Exploration bonus `λ·ρ·√(1+|S|)/(1+n_i)` | `BanditStrategy::Ucb` with exploration weight |
| One-hop neighbor exclusion | Diversity penalty in `ScreeningPruner::relevance()` |
| Trajectory-level graph | Our arena tree structure |

---

## 3. Trajectory-Level Credit Assignment (maps to G-Zero)

Their post-training assigns credit by **max trajectory score** to ALL nodes in that trajectory:

- w = 1 for all nodes in best trajectory
- w = 0 for all nodes in worst trajectory

This contrasts with our current per-step δ in `DeltaBanditPruner`. The trajectory-level approach is coarser but more robust to sparse rewards.

### Alignment with G-Zero Phases

| Phase | Our Signal | SimpleTES Analog |
|-------|-----------|------------------|
| G-Zero Phase 1 (modelless) | Per-step δ → BanditPruner | Per-step evaluator scores |
| G-Zero Phase 2 (model-based) | Per-step δ → DPO/GRPO | Trajectory-level max score → SFT |
| **New: Trajectory bridge** | Max-trajectory δ → all steps | SimpleTES credit assignment |

---

## 4. Best-Solution Restart (maps to AbsorbCompress)

Their restart strategy reinitializes from the best discovered solution, which is exactly our `AbsorbCompress` layer promotion pattern: take the highest-scoring observation, promote it as the new seed for further exploration.

---

## 5. Trajectory-Level Pruning (early stopping)

Maps to our `early_exit_patience` / `early_exit_gap` but applied at **trajectory** level, not node level. Kill underperforming chains early to reallocate budget.

---

## 6. Actionable Distillations

### ✅ Already Distilled (No New Code)

- Minimalist prompt engineering — we already use this
- Async execution model — our arena handles this
- Per-step scoring → our `BanditPruner` already does this

### 🔧 New Distillations

#### Distillation A: `TesLoop` Trait (~80 LOC, Low Effort, High Value)

```microgpt-rs/.research/052_SimpleTES_Evaluation_Driven_Scaling.md#L115-128
// Feature-gated: tes_loop = ["bandit"]

pub struct TesConfig {
    pub global_width: usize,     // C: parallel trajectories
    pub refinement_depth: usize, // L: iterations per trajectory
    pub local_sample_size: usize, // K: candidates per step
}

pub trait TesLoop {
    fn config(&self) -> &TesConfig;
    fn budget(&self) -> usize {
        self.config().global_width * self.config().refinement_depth * self.config().local_sample_size
    }
    fn select_inspirations(&self, history: &[TesNode]) -> Vec<usize>; // Φ
}
```

#### Distillation B: RPUCG Bandit Variant (~120 LOC, Medium Effort, High Value)

Add graph-based propagation to `BanditPruner`:
- Propagated value `V_i` with discount `γ`
- UCB exploration with set-size normalization
- One-hop diversity exclusion

Builds on existing `BanditStrategy::Ucb`, adds parent-child value propagation.

#### Distillation C: Trajectory-Level Pruning (~60 LOC, Low Effort, High Value)

Kill underperforming trajectories early based on running score gap. Maps directly to arena infrastructure with `early_exit_patience` applied at chain level.

#### Distillation D: Trajectory Credit Bridge (~40 LOC, Low Effort, Medium Value)

Bridge from trajectory-level max score to per-step credit for G-Zero Phase 2. Used in DPO training when we have sparse trajectory rewards.

---

## 7. What We Don't Need

| Item | Reason |
|------|--------|
| Hacking analysis (§4.3) | Production concern, not research scope |
| LLM-specific prompt templates | We use our own minimalist prompts |
| Multi-domain evaluator design | Our domain evaluators already exist |
| Front-end visualization | Out of scope |

---

## 8. Verdict

| Distillation | Effort | Value | Priority |
|-------------|--------|-------|----------|
| TesLoop trait + config | Low | High | P1 |
| RPUCG bandit variant | Medium | High | P2 |
| Trajectory-level pruning | Low | High | P3 |
| Trajectory credit bridge | Low | Medium | P4 |

**Feature gate:** `tes_loop = ["bandit"]` — builds on bandit infrastructure, no new model dependencies.

SimpleTES validates that modelless evaluation-driven scaling beats frontier models with complex pipelines. Our `BanditPruner` + `AbsorbCompress` + `ScreeningPruner` stack is already the right architecture — SimpleTES adds trajectory-level granularity and graph-based propagation as incremental improvements.

---

## References

- Paper: https://arxiv.org/abs/2604.19341
- Related: `21_G-Zero_Self-Play_Open-Ended_Generation.md` (δ signal, credit assignment)
- Related: `23_GFlowNet_Shortest_Paths.md` (flow-based trajectory selection)
- Related: `37_REAP_Model-Based_Modelless_Duality.md` (model-based/modelless spectrum)