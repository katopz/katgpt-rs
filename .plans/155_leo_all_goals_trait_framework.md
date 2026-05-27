# Plan 155: LEO All-Goals Trait Framework (Open — MIT)

**Date:** 2026-05-27
**Research:** katgpt-rs Research 118, riir-ai Research 012
**Verdict:** ⭐ SUPER GOAT — Open trait framework, feature-gated
**Ref:** 27_mmo_goat_pillars_decision_matrix.md (open/close boundary)

---

# Task

- [x] T1: Add `LeoHead` trait to `katgpt-rs-core/src/traits.rs`
- [x] T2: Add `DualLeoMixer` trait + default impl
- [x] T3: Add `AllGoalsUpdate` trait + vectorized Bellman
- [x] T4: Add `AutocurriculumSampler` trait + default impl
- [x] T5: Add `sigmoid_bounded_q` utility
- [x] T6: Feature gate: `leo_all_goals`, `dual_leo`
- [x] T7: Unit tests for all traits
- [x] T8: GOAT proof — trait compilation + micro-bench

---

## T1: `LeoHead` Trait

```rust
/// All-goals Q-value output head (LEO architecture).
/// 
/// Instead of conditioning on a goal (UVFA-style), this outputs
/// Q-values for ALL goals simultaneously: Q(s) → R^{G×A}.
/// 
/// Ref: Matthews et al. (2026) "Learn Everything All at Once"
pub trait LeoHead {
    /// Compute Q-values for all goals × all actions from state.
    /// Returns [goals][actions] flattened.
    fn all_goals_q(&self, state: &[f32]) -> Vec<f32>;
    
    /// Number of goals in the output head.
    fn goal_count(&self) -> usize;
    
    /// Number of discrete actions per goal.
    fn action_count(&self) -> usize;
    
    /// Extract Q-values for a specific goal by indexing.
    fn q_for_goal(&self, all_q: &[f32], goal: usize) -> &[f32] {
        let start = goal * self.action_count();
        &all_q[start..start + self.action_count()]
    }
}
```

---

## T2: `DualLeoMixer` Trait

```rust
/// Dual LEO mixing between teacher (LEO) and student (UVFA).
///
/// Q_combined(g) = α·Q_LEO(s,a,g) + (1-α)·Q_UVFA(s,a,g)
///
/// α controls modelless→model trust transfer:
/// - High α: trust LEO teacher (modelless, broad)
/// - Low α: trust UVFA student (model-based, precise)
pub trait DualLeoMixer {
    /// Mix LEO and UVFA Q-values for acting on goal.
    fn mix(
        &self,
        q_leo: &[f32],   // [actions] for specific goal
        q_uvfa: &[f32],  // [actions] for specific goal
        alpha: f32,
    ) -> Vec<f32> {
        q_leo.iter()
            .zip(q_uvfa.iter())
            .map(|(&ql, &qu)| alpha * ql + (1.0 - alpha) * qu)
            .collect()
    }
    
    /// Default α = 0.3 (from paper sweep on Craftax).
    fn default_alpha(&self) -> f32 { 0.3 }
}
```

---

## T3: `AllGoalsUpdate` Trait

```rust
/// Vectorized all-goals Bellman update.
///
/// L = (R(s') + γ · max_a' Q(a'|s') - Q(a|s))²
///
/// Where R(s') ∈ R^G is the reward vector across ALL goals.
/// Single forward pass updates all |G| Q-value heads simultaneously.
pub trait AllGoalsUpdate {
    /// Compute all-goals TD target.
    /// rewards: [goals] — R(s',g) for all g
    /// next_q: [goals][actions] — Q(s',a',g) for all g,a
    /// Returns: [goals] — TD target per goal
    fn td_target(
        &self,
        rewards: &[f32],
        next_q: &[Vec<f32>],
        gamma: f32,
    ) -> Vec<f32> {
        rewards.iter()
            .zip(next_q.iter())
            .map(|(&r, q_next)| {
                let max_q = q_next.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                r + gamma * max_q
            })
            .collect()
    }
    
    /// Compute all-goals TD loss (MSE).
    fn loss(
        &self,
        predicted: &[Vec<f32>], // [goals] chosen action Q-values
        target: &[f32],          // [goals] TD targets
    ) -> f32 {
        predicted.iter()
            .zip(target.iter())
            .map(|(q_pred, &q_tgt)| {
                let chosen = q_pred[0]; // simplified: take first action
                0.5 * (chosen - q_tgt).powi(2)
            })
            .sum::<f32>() / predicted.len() as f32
    }
}
```

---

## T4: `AutocurriculumSampler` Trait

```rust
/// Goal sampling from previously observed goals only.
///
/// "We sample goals only from goals observed at least once in the past,
/// to prevent completely out-of-reach goals being sampled."
/// — Matthews et al. (2026)
pub trait AutocurriculumSampler {
    /// Sample a goal uniformly from previously observed goals.
    fn sample_goal(&self, rng: &mut impl Rng) -> usize;
    
    /// Mark a goal as observed (first time seen in any trajectory).
    fn observe_goal(&mut self, goal: usize);
    
    /// Number of unique goals observed so far.
    fn observed_count(&self) -> usize;
    
    /// Total goals in the goal set.
    fn total_goal_count(&self) -> usize;
}
```

---

## T5: `sigmoid_bounded_q` Utility

```rust
/// Bound Q-value estimates with sigmoid to prevent divergence.
/// CRITICAL: Without this, LEO's Q-values frequently diverge
/// due to highly off-policy updates (paper Section 5.1).
pub fn sigmoid_bounded_q(raw_q: f32) -> f32 {
    1.0 / (1.0 + (-raw_q).exp())
}
```

---

## T6: Feature Gate

```toml
[features]
default = []
leo_all_goals = []            # LeoHead + AllGoalsUpdate + sigmoid_bounded_q
dual_leo = ["leo_all_goals"]  # + DualLeoMixer + AutocurriculumSampler
```

---

## T7-T8: Tests + GOAT Proof

- Unit tests for all trait default implementations
- Micro-bench: `LeoHead::all_goals_q` with 512 goals × 8 actions
- GOAT proof: trait compilation passes, no runtime errors, feature gate works

---

## Priority

**MEDIUM** — Framework only. Depends on riir-ai Plan 155 for game-specific implementations that prove the value. Ship the trait sockets first, let riir-ai fill in the plugs.

---

## References

- katgpt-rs Research 118 (full analysis)
- riir-ai Research 012 (game-specific mapping + Super GOAT rationale)
- 27_mmo_goat_pillars_decision_matrix.md (open/close boundary)
