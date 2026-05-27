# Research 118: LEO — Learn Everything All at Once (Goal-Conditioned RL)

**Date:** 2026-05-27
**Paper:** arXiv:2605.23551 — "Goal-Conditioned Agents that Learn Everything All at Once"
**Code:** `riir-ai/.raw/purejaxgcrl/` (JAX reference implementation)
**Verdict:** ⭐ SUPER GOAT — Cross-cutting game AI architecture, feature-gated

---

## Executive Summary

LEO reparameterizes goal-conditioned Q-learning by currying goals from input to output: `Q(s) → R^{G×A}`. This enables **all-goals updates** in a single forward/backward pass — 250× faster than naïve relabelling. Dual LEO pairs a LEO teacher (all-goals, rough) with a UVFA student (single-goal, precise), creating a teacher-student bootstrapping loop.

**For katgpt-rs:** LEO provides the generic open-source trait framework for all-goals learning. The game-specific implementations (goal sets, network architectures, α schedules) stay in riir-ai as the commercial selling point.

---

## Core Innovation

### Currying Goals from Input to Output

**UVFA (standard):** `Q(s, g) → R^A` — goal concatenated to state as input
**LEO:** `Q(s) → R^{G×A}` — goal moved to output dimension

This allows **vectorized Bellman update** across all goals:
```
L = (R(s') + γ · max_a' Q(a'|s') - Q(a|s))²
```
Where `R(s') ∈ R^G` is the reward vector for all goals.

### Dual LEO Architecture

```
State s ──→ [LEO Network] ──→ Q_LEO(s,a,g) for ALL g
                │                    │
                │              α · Q_LEO
                │                    │
State s + Goal g ──→ [UVFA Network] ──→ (1-α) · Q_UVFA
                                     │
                                     ▼
                              Q_combined = α·Q_LEO + (1-α)·Q_UVFA
```

Teacher-Student dynamic:
1. LEO learns rough Q for goal G from incidental experience
2. UVFA has zero estimates for G
3. When G commanded: Q_combined ≈ Q_LEO (α dominates)
4. Positive examples → UVFA learns → (1-α)·Q_UVFA dominates
5. Natural handoff from modelless teacher to model-based student

---

## Key Results (Paper)

| Metric | Value |
|--------|-------|
| Craftax (512 goals) success rate | Dual LEO (PPO): 0.22 vs best baseline: 0.08 (2.75×) |
| Speed (512 goals) | LEO: 37K steps/sec vs naïve: 140 steps/sec (264× faster) |
| Overhead vs single-goal | Only 34% slower for 512 goals |
| Partial update tolerance | 60% of heads sufficient for Dual LEO |
| Continuous control (Ant U Maze) | SAC+LEO: 0.45 vs SAC+HER: 0.35 |

---

## Open-Source Trait Framework (katgpt-rs)

### Proposed Traits

```rust
/// All-goals Q-value output head
pub trait LeoHead {
    /// Output Q-values for all goals × all actions: R^{G×A}
    fn all_goals_q(&self, state: &[f32]) -> Vec<Vec<f32>>; // [goals][actions]
    
    /// Number of goals in the output
    fn goal_count(&self) -> usize;
    
    /// Number of actions per goal
    fn action_count(&self) -> usize;
}

/// Dual LEO mixing between teacher (LEO) and student (UVFA)
pub trait DualLeoMixer {
    /// Mix LEO and UVFA Q-values for acting
    fn mix(&self, q_leo: &[f32], q_uvfa: &[f32], goal: usize) -> Vec<f32>;
    
    /// Get current mixing coefficient α
    fn alpha(&self) -> f32;
}

/// Vectorized all-goals Bellman update
pub trait AllGoalsUpdate {
    /// Compute all-goals TD target: R(s') + γ · max_a' Q(a'|s')
    fn td_target(&self, rewards: &[f32], next_q: &[Vec<f32>], gamma: f32) -> Vec<f32>;
    
    /// Compute all-goals loss
    fn loss(&self, predicted: &[Vec<f32>], target: &[Vec<f32>]) -> f32;
}

/// Autocurriculum goal sampling
pub trait AutocurriculumSampler {
    /// Sample a goal from previously observed goals
    fn sample_goal(&self, rng: &mut impl Rng) -> usize;
    
    /// Mark a goal as observed
    fn observe_goal(&mut self, goal: usize);
    
    /// Get observed goal count
    fn observed_count(&self) -> usize;
}
```

### Feature Gate

```toml
[features]
leo_all_goals = []           # Generic LEO trait framework
dual_leo = ["leo_all_goals"] # Dual LEO mixing + autocurriculum
```

---

## Relationship to Existing katgpt-rs Features

| Feature | How LEO Interacts |
|---------|-------------------|
| `ConstraintPruner` | Defines R(s,g) — the reward function LEO learns |
| `ScreeningPruner` | Can use LEO Q-values as relevance scores |
| `SpeculativeVerifier` | LEO Q-values guide speculation toward high-value goals |
| `Multi-Armed Bandit` | α = bandit mixing parameter for model vs modelless |
| `DDTree` | LEO Q-values populate tree nodes for all goals |
| `G-Zero (Plan 049)` | LEO = G-Zero's value learning formalization |
| `GFlowNet (Plan 052)` | LEO × GFlowNet = all-goals updates from sampled paths |
| `Data Gate (Plan 111)` | LEO autocurriculum = data gate for goal sampling |
| `SR²AM (Plan 112)` | α-mixing = SR²AM's model vs modelless decision |
| `SpecHop (Plan 131)` | LEO Q-values guide hop-level speculation |
| `Parallel Probe (Plan 133)` | LEO all-goals = parallel probe across goal space |

---

## Honest Assessment

| Pro | Con |
|-----|-----|
| 250× speedup over naïve all-goals | Requires finite goal set |
| Teacher-student natural modelless→model bridge | Late fusion limits easy-goal performance (Dual LEO fixes) |
| 34% overhead for 512 goals | Continuous actions need O(\|G\|) policy updates |
| Cross-game transfer via shared goal representations | Needs sigmoid bounding to prevent divergence |
| Matches our modelless→model pipeline exactly | Not yet validated on our specific games |

**Verdict:** The architecture maps perfectly to our existing modelless→model pipeline. The open traits provide plug sockets; riir-ai provides the game-specific plugs. The 2.75× improvement on Craftax (512 goals) strongly suggests similar gains for our games with 100+ goals each.

---

## References

- Matthews et al. (2026). "Goal-Conditioned Agents that Learn Everything All at Once." ICML 2026.
- See riir-ai Research 012 for full analysis and game-specific mapping.
