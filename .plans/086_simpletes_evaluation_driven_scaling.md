# Plan 086: SimpleTES Evaluation-Driven Scaling

> **Research:** [52_SimpleTES_Evaluation_Driven_Scaling.md](../.research/052_SimpleTES_Evaluation_Driven_Scaling.md)
> **Source:** [SimpleTES](https://arxiv.org/abs/2604.19341) — Evaluation-Driven Scaling
> **Feature Gate:** `tes_loop = ["bandit"]`
> **Type:** Modelless

## Tasks

- [x] **T1: TesConfig + TesNode types** — Add core types to `types.rs`
- [x] **T2: TesLoop trait** — Core trait in `src/pruners/tes_loop.rs` with default RPUCG selection
- [x] **T3: RPUCG bandit variant** — Graph-based propagation in `BanditStrategy::Rpucg`
- [x] **T4: Trajectory-level pruning** — Chain-level early stopping in arena infrastructure
- [x] **T5: GOAT proof — Simulated TES loop** — Prove RPUCG beats greedy (4/4 proofs passed, Bench 016)
- [x] **T6: Feature gate audit** — Zero impact on default build
- [x] **T7: Trajectory Credit Bridge** — Max-trajectory-score credit assignment for G-Zero Phase 2 (4/4 proofs, Bench 017)
- [x] **T8: SimpleTesLoop struct** — Concrete `SimpleTesLoop<E>` implementing full C×L×K loop (4/4 proofs, Bench 017)
- [x] **T9: Budget Scaling Benchmark** — Wide(24×5×8)=0.9988 vs Narrow(2×8×30)=0.8266, spread=0.1722 (Bench 017)
- [x] **T10: Cross-Strategy GOAT proof** — RPUCG≅UCB1 flat, Thompson best Gaussian, all converge (Bench 017)

---

## Context

SimpleTES (arXiv:2604.19341) proves evaluation-driven loops with (C=32, L=100, K=16) beat frontier models using open-source gpt-oss-120b. Their RPUCG is graph-based UCB — exactly our `BanditPruner` with parent-child value propagation.

The key insight: **modelless evaluation-driven scaling is already our architecture**. SimpleTES adds trajectory-level granularity and graph-based propagation to our existing `BanditPruner` + `AbsorbCompress` + `ScreeningPruner` stack.

### Architecture

```
┌─────────────────────────────────────────────────┐
│ TesLoop<C, L, K, Φ>                              │
│                                                   │
│  C trajectories × L steps × K candidates          │
│  Φ = RPUCG (graph-based UCB)                      │
│                                                   │
│  Per-step: BanditPruner (existing)                 │
│  Per-trajectory: RPUCG propagation (new)           │
│  Across-trajectories: pruning (new)                │
└─────────────────────────────────────────────────┘
```

---

## T1: TesConfig + TesNode Types

### What

Core configuration and node types for the TES loop. `TesConfig` holds the (C, L, K) hyperparameters. `TesNode` tracks per-candidate state including graph propagation metadata.

### Where

`src/speculative/types.rs` — add types behind feature gate

### Implementation

```rust
// src/speculative/types.rs

#[cfg(feature = "tes_loop")]
use crate::pruners::bandit::BanditStrategy;

/// SimpleTES configuration (C, L, K) hyperparameters.
///
/// C = global_width: parallel trajectories (default 32)
/// L = refinement_depth: iterations per trajectory (default 100)
/// K = local_sample_size: candidates per step (default 16)
///
/// Budget = C × L × K total evaluations per solve.
#[cfg(feature = "tes_loop")]
#[derive(Clone, Debug)]
pub struct TesConfig {
    pub global_width: usize,       // C: parallel trajectories
    pub refinement_depth: usize,   // L: iterations per trajectory
    pub local_sample_size: usize,  // K: candidates per step
    pub bandit_strategy: BanditStrategy,
}

#[cfg(feature = "tes_loop")]
impl Default for TesConfig {
    fn default() -> Self {
        Self {
            global_width: 32,
            refinement_depth: 100,
            local_sample_size: 16,
            bandit_strategy: BanditStrategy::Rpucg { gamma: 0.8, lambda: 1.0 },
        }
    }
}

#[cfg(feature = "tes_loop")]
impl TesConfig {
    pub fn budget(&self) -> usize {
        self.global_width * self.refinement_depth * self.local_sample_size
    }
}

/// Node in the TES evaluation graph.
///
/// Each node represents a candidate solution with:
/// - Direct evaluation score r
/// - Graph-propagated value U (max of own score and children's values)
/// - Visit count for UCB exploration
#[cfg(feature = "tes_loop")]
#[derive(Clone, Debug)]
pub struct TesNode {
    pub solution: Vec<usize>,      // The candidate tokens
    pub score: f32,                // Evaluator score r
    pub metadata: String,          // Feedback text m
    pub parent_idx: Option<usize>, // For graph propagation
    pub visit_count: usize,        // For RPUCG exploration
    pub propagated_value: f32,     // U_i = max(r_i, γ·max_child_U)
}

#[cfg(feature = "tes_loop")]
impl TesNode {
    pub fn new(solution: Vec<usize>, parent_idx: Option<usize>) -> Self {
        Self {
            solution,
            score: 0.0,
            metadata: String::new(),
            parent_idx,
            visit_count: 0,
            propagated_value: 0.0,
        }
    }
}
```

---

## T2: TesLoop Trait

### What

Core trait for the TES evaluation loop. Provides the loop skeleton: select inspirations from history, evaluate candidates, propagate values back through the graph.

### Where

`src/pruners/tes_loop.rs` — new file, feature-gated

### Implementation

```rust
// src/pruners/tes_loop.rs

//! SimpleTES evaluation-driven scaling loop.
//!
//! Feature-gated under `tes_loop` (requires `bandit`).

#[cfg(feature = "tes_loop")]
use crate::speculative::types::{TesConfig, TesNode};

/// Core trait for the TES evaluation loop.
///
/// Implementors provide the evaluation function; the trait provides
/// default RPUCG selection and value propagation.
#[cfg(feature = "tes_loop")]
pub trait TesLoop: Send + Sync {
    /// Get the TES configuration.
    fn config(&self) -> &TesConfig;

    /// Total evaluation budget: C × L × K.
    fn budget(&self) -> usize {
        self.config().budget()
    }

    /// Select `count` inspirations from history using RPUCG greedy selection.
    ///
    /// Default: greedy by propagated_value, excluding one-hop neighbors
    /// for diversity (SimpleTES Section 3.3).
    fn select_inspirations(&self, history: &[TesNode], count: usize) -> Vec<usize> {
        if history.is_empty() || count == 0 {
            return Vec::new();
        }

        let mut selected: Vec<usize> = Vec::with_capacity(count);
        let mut excluded: HashSet<usize> = HashSet::new();

        while selected.len() < count {
            let best = history.iter().enumerate()
                .filter(|(i, _)| !selected.contains(i) && !excluded.contains(i))
                .max_by(|(_, a), (_, b)| {
                    a.propagated_value.partial_cmp(&b.propagated_value)
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(i, _)| i);

            match best {
                Some(idx) => {
                    selected.push(idx);
                    // Exclude one-hop neighbors for diversity
                    excluded.insert(idx);
                    if let Some(parent) = history[idx].parent_idx {
                        excluded.insert(parent);
                    }
                    for (child_idx, node) in history.iter().enumerate() {
                        if node.parent_idx == Some(idx) {
                            excluded.insert(child_idx);
                        }
                    }
                }
                None => break,
            }
        }

        selected
    }

    /// Backpropagate values through the evaluation graph.
    ///
    /// U_i = max(r_i, γ · max(U_child_j for j in children(i)))
    fn update_propagated_values(&self, history: &mut [TesNode], gamma: f32) {
        // Process in reverse order (children before parents)
        for i in (0..history.len()).rev() {
            let own_score = history[i].score;

            let max_child_value = history.iter()
                .filter(|node| node.parent_idx == Some(i))
                .map(|node| node.propagated_value)
                .fold(0.0f32, |acc, v| acc.max(v));

            history[i].propagated_value = own_score.max(gamma * max_child_value);
        }
    }
}
```

---

## T3: RPUCG Bandit Variant

### What

Add `Rpucg` variant to `BanditStrategy` enum. RPUCG (Rooted Propagation UCB on Graph) is SimpleTES's graph-based UCB that propagates values from children to parents.

### Where

`src/pruners/bandit.rs` — add enum variant and selection logic

### Implementation

```rust
// Add to BanditStrategy enum in src/pruners/bandit.rs

pub enum BanditStrategy {
    EpsilonGreedy { epsilon: f32 },
    Ucb { exploration_weight: f32 },
    ThompsonSampling,
    #[cfg(feature = "tes_loop")]
    Rpucg { gamma: f32, lambda: f32 }, // NEW: graph-based UCB
}
```

### RPUCG Selection Formula

The RPUCG scoring for node i:

- `U_i = max(r_i, γ · max(U_child_j for j in children(i)))`
- `score_i = U_i + λ · ρ_i · √(1 + |S|) / (1 + n_i)`

Where:
- `r_i` = direct evaluation score
- `γ` = propagation discount (default 0.8)
- `λ` = exploration weight (default 1.0)
- `ρ_i` = parent influence factor
- `|S|` = total visits across all nodes
- `n_i` = visits to node i

Select top-k by score. Exclude one-hop neighbors for diversity.

---

## T4: Trajectory-Level Pruning

### What

Extend arena infrastructure with chain-level early stopping. At checkpoints (L/4, L/2, 3L/4), rank trajectories by current best score and kill the bottom X%.

### Where

`src/pruners/arena/` — extend existing arena with trajectory pruning
`src/speculative/types.rs` — `TesConfig` already has `refinement_depth`

### Implementation

```rust
// Trajectory-level pruning checkpoint logic

#[cfg(feature = "tes_loop")]
pub struct TrajectoryPruner {
    /// Checkpoint fractions (e.g., [0.25, 0.5, 0.75])
    pub checkpoints: Vec<f32>,
    /// Fraction of trajectories to kill at each checkpoint
    pub kill_fraction: f32,
}

#[cfg(feature = "tes_loop")]
impl TrajectoryPruner {
    pub fn new() -> Self {
        Self {
            checkpoints: vec![0.25, 0.5, 0.75],
            kill_fraction: 0.3,
        }
    }

    /// Check if current step is a checkpoint.
    pub fn is_checkpoint(&self, step: usize, total_steps: usize) -> bool {
        self.checkpoints.iter().any(|&frac| {
            let checkpoint_step = (frac * total_steps as f32) as usize;
            step == checkpoint_step
        })
    }

    /// Prune bottom trajectories, return indices to kill.
    pub fn prune(&self, chains: &[TesNode], budget_to_redistribute: usize) -> Vec<usize> {
        let kill_count = ((chains.len() as f32 * self.kill_fraction) as usize)
            .min(chains.len().saturating_sub(1));

        let mut indexed: Vec<(usize, f32)> = chains.iter()
            .enumerate()
            .map(|(i, node)| (i, node.propagated_value))
            .collect();

        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        indexed.into_iter()
            .take(kill_count)
            .map(|(i, _)| i)
            .collect()
    }
}
```

---

## T5: GOAT Proof — Bomber Arena

### What

Prove RPUCG beats vanilla UCB in the existing bomber arena. This is the empirical validation that graph-based propagation outperforms flat bandit selection.

### Setup

Use existing bomber arena (`src/pruners/arena/`).

| Config | Strategy | Episodes |
|--------|----------|----------|
| Baseline | `BanditStrategy::Ucb { exploration_weight: 1.0 }` | 1000 |
| RPUCG-g08 | `BanditStrategy::Rpucg { gamma: 0.8, lambda: 1.0 }` | 1000 |
| RPUCG-g09 | `BanditStrategy::Rpucg { gamma: 0.9, lambda: 1.0 }` | 1000 |

### Metrics

1. **Cumulative reward** — total score across episodes
2. **Convergence speed** — episodes to reach 90% of peak performance
3. **Best-arm identification** — frequency of selecting optimal action
4. **Regret** — cumulative vs oracle

### Benchmark File

`.benchmarks/012_simpletes_rpucg_goat.md`

---

## T6: Feature Gate Audit

### What

Ensure all TES code is behind `tes_loop` feature gate with zero impact on default build.

### Checklist

- [x] `tes_loop = ["bandit"]` added to `Cargo.toml` features
- [x] All new types: `#[cfg(feature = "tes_loop")]`
- [x] All new files: `tes_loop.rs` conditionally included in `mod.rs`
- [x] `BanditStrategy::Rpucg` variant: `#[cfg(feature = "tes_loop")]`
- [x] `cargo build` — zero new code compiled, no warnings
- [x] `cargo build --features tes_loop` — all TES code active
- [x] `cargo test` (no features) — all existing tests pass
- [x] `cargo test --features tes_loop` — all tests pass including new
- [x] `cargo clippy --features tes_loop` — no warnings
- [x] No performance regression on default build

---

## File Changes Summary

| File | Change | Feature Gate |
|------|--------|-------------|
| `Cargo.toml` | Add `tes_loop = ["bandit"]` feature | — |
| `src/speculative/types.rs` | `TesConfig`, `TesNode` structs | `tes_loop` |
| `src/pruners/tes_loop.rs` | New file: `TesLoop` trait + default impl | `tes_loop` |
| `src/pruners/mod.rs` | Conditional `mod tes_loop` | `tes_loop` |
| `src/pruners/bandit.rs` | `BanditStrategy::Rpucg` variant + selection | `tes_loop` |
| `src/pruners/arena/` | `TrajectoryPruner` for chain-level pruning | `tes_loop` |
| `tests/test_simpletes.rs` | New test file: RPUCG selection, propagation proofs | `tes_loop` |

---

## References

- **Paper**: arXiv:2604.19341 — SimpleTES: Evaluation-Driven Scaling
- **Research**: `.research/052_SimpleTES_Evaluation_Driven_Scaling.md`
- **Related**: Plan 030 (BanditPruner), Plan 050 (Feature Gate Audit), Plan 033 (Bomber Arena)
- **Key files**: `src/pruners/bandit.rs`, `src/speculative/types.rs`, `src/pruners/arena/`
