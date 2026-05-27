# Plan 155: LEO All-Goals Trait Framework (Open — MIT)

**Date:** 2026-05-27
**Research:** katgpt-rs Research 118, riir-ai Research 012
**Source:** `.raw/purejaxgcrl/` (JAX reference implementation)
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
- [x] T9: Refine traits from source code audit (see § Source Code Corrections)

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
    /// Returns `[goals * actions]` flattened (row-major: goal-major).
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

### Source Code Notes

The JAX `LEONetworkConvSymbolicCraftax` outputs `qs.reshape((batch, num_goals, action_dim))` then applies `jax.nn.sigmoid(qs)` **unconditionally** (default `normalise_output=True`). The final layer is a single `Dense(num_goals * action_dim)` — one big linear layer, reshaped to `[G, A]`. This is elegant: no per-goal heads, just one fat output.

For our trait, `all_goals_q()` should return the **post-sigmoid** values (already bounded), matching the JAX implementation.

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
    fn mix(&self, q_leo: &[f32], q_uvfa: &[f32], alpha: f32) -> Vec<f32> {
        q_leo.iter()
            .zip(q_uvfa.iter())
            .map(|(&ql, &qu)| alpha * ql + (1.0 - alpha) * qu)
            .collect()
    }
    
    /// Default α = 0.3 (from paper sweep on Craftax, sweep config `lc_leo_weight`).
    fn default_alpha(&self) -> f32 { 0.3 }
}
```

### Source Code Notes — 5 Acting Modes

The JAX `dual_leo_pqn.py` supports **5 acting modes** via `DUAL_LEO_ACTING_MODE`:

| Mode | Formula | Use Case |
|------|---------|----------|
| `leo_act` | `Q = Q_LEO[:,:,g]` | LEO-only ablation |
| `uvfa_act` | `Q = Q_UVFA[:,g]` | UVFA-only ablation |
| `lc` | `Q = (1-α)·Q_UVFA + α·Q_LEO` | **Default (sweep winner)** |
| `max` | `Q = max(Q_LEO, Q_UVFA)` | Optimistic combining |
| `min` | `Q = min(Q_LEO, Q_UVFA)` | Pessimistic combining |

The sweep uses `lc` with `lc_leo_weight=0.3` and `anneal_lc_leo=false`. When annealing IS enabled:
```python
p = n_updates / num_updates
coef = p * LC_LEO_ANNEAL_END + (1 - p) * LC_LEO_ANNEAL_START
```
This linearly interpolates α from `ANNEAL_START` to `ANNEAL_END` over training.

**Our trait should add:** `fn acting_mode() -> ActingMode` enum to support all 5 modes.

### Source Code Notes — Dual LEO PPO (BC Regularization)

The `dual_leo_ppo.py` variant adds **behavioral cloning regularization**:
- `bc_coef_policy = 0.1` — PPO policy is regularized toward LEO's argmax action
- `bc_coef_value = 0.0` — no value BC (disabled in sweep)
- `bc_policy_target = "argmax"` — target is LEO's greedy action
- `anneal_bc = true` — BC coefficient decays to 0 over training

This means the UVFA student doesn't just learn from its own TD targets — it's also pulled toward LEO's policy early in training. This is critical for bootstrapping and should be captured in the trait framework.

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

### Source Code Notes — Actual TD Target Computation

The JAX source computes TD targets differently than the trait suggests:

```python
# From dual_leo_pqn.py L450-477
q_targets = transitions.reward_all_goals + config["GAMMA"] * max_qs * (
    1 - jnp.logical_or(transitions.done_all_goals, transitions.done_ep[:, :, None])
)
```

Key details:
1. **`reward_all_goals`** = `goals_achieved * 1.0` — binary reward (0 or 1 per goal)
2. **Terminal masking**: When `done_all_goals[g]` OR `done_ep`, the target is just the reward (no bootstrap)
3. **`max_qs`** comes from the **stored Q-values** (`q_vals_all` from the forward pass during acting), NOT a fresh forward pass on `next_obs`. The last timestep's Q is computed fresh from `last_obs`.

For UVFA, the source uses **Q(λ) lambda returns** (L608-634):
```python
lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
```
This is a multi-step TD method with eligibility traces, not single-step Bellman. The `LAMBDA` parameter controls the trace decay.

**Our trait should add:** `fn td_target_lambda()` for Q(λ) variant used in UVFA update.

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

### Source Code Notes — Autocurriculum Details

The JAX autocurriculum is more nuanced than the trait suggests:

1. **`goals_seen`** is a boolean mask over ALL goals, updated via `get_goals_seen()` which checks if any observation in the batch matches a goal's observation pattern (union matching: `match_sum > 0`).

2. **`sample_goal()`** uses `jax.random.choice(rng, arange(len(seen_goals)), p=seen_goals)` — samples proportionally from the boolean mask. This is effectively **uniform over observed goals**.

3. **`ONLY_SAMPLE_FROM_SEEN_GOALS`** config flag — when `True`, only samples from seen; when `False`, samples uniformly from all goals.

4. **Goal resampling triggers**: A new goal is sampled when:
   - Acting goal is achieved (`acting_goals_achieved`)
   - Episode ends (`new_done`)
   - Both conditions use `jnp.lax.select(done | achieved, new_goal, old_goal)`

5. **`LIVE_SUCCESS_RATE_DECAY`** — Success/failure counters are decayed each update step: `counter *= LIVE_SUCCESS_RATE_DECAY`. This gives an exponential moving average of per-goal success rates, used for logging and potentially for adaptive goal sampling.

6. **`num_goals_completed`** — Tracks how many goals were completed within the current episode (resets to 0 on episode end). This enables "first return then explore" — after achieving one goal, immediately sample another.

**Our trait should capture:** The `seen_goals` boolean mask model, the decay mechanism, and `num_goals_completed` tracking.

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

### Source Code Notes — Sigmoid Is ALWAYS ON

In `LEONetworkConvSymbolicCraftax` (L262-263):
```python
if self.normalise_output:
    qs = jax.nn.sigmoid(qs)
```
And in `LEONetworkFlat` (L328):
```python
qs = jax.nn.sigmoid(qs)  # Always applied, no conditional
```

The sweep config uses `NETWORK_SIGMOID_VALUE=True` (default). The `normalise_output` parameter defaults to `True` in the network constructor. This means Q-values are **always** bounded to [0, 1], which maps to goal achievement probability.

This is not just a stability trick — it's fundamental to how LEO interprets Q-values. Since rewards are binary (goal achieved = 1, not achieved = 0), Q-values represent **probability of eventual goal achievement**, and sigmoid is the natural bounding function.

---

## T6: Feature Gate

```toml
[features]
leo_all_goals = []            # LeoHead + AllGoalsUpdate + sigmoid_bounded_q
dual_leo = ["leo_all_goals"]  # + DualLeoMixer + AutocurriculumSampler
```

### Source Code Notes — Dual LEO PPO vs PQN Variants

The source has two Dual LEO variants:
- `dual_leo_pqn.py` — UVFA is a Q-network (PQN-style), uses ε-greedy
- `dual_leo_ppo.py` — UVFA is an actor-critic (PPO-style), uses stochastic policy + BC regularization

Both share the same LEO network. The trait framework should be agnostic to which UVFA variant is used.

---

## T9: Source Code Corrections (New Task)

After auditing `.raw/purejaxgcrl/`, these refinements are needed:

### 9a. Add `ActingMode` enum to `DualLeoMixer`

```rust
#[cfg(feature = "dual_leo")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActingMode {
    LeoOnly,    // leo_act — LEO only (ablation)
    UvfaOnly,   // uvfa_act — UVFA only (ablation)
    Lc,         // lc — linear combination (default, sweep winner)
    Max,        // max — optimistic combining
    Min,        // min — pessimistic combining
}
```

### 9b. Add `AlphaSchedule` to `DualLeoMixer`

```rust
pub enum AlphaSchedule {
    Fixed(f32),                                          // Constant α
    LinearAnneal { start: f32, end: f32 },               // Linear from start→end
}
```

### 9c. Add `BcRegularization` to `DualLeoMixer` (for PPO variant)

```rust
pub struct BcConfig {
    pub policy_coef: f32,      // Default: 0.1
    pub value_coef: f32,       // Default: 0.0
    pub target: BcTarget,      // Argmax | QWeighted
    pub anneal: bool,          // Decay to 0 over training
}

pub enum BcTarget {
    Argmax,       // Follow LEO's greedy action
}
```

### 9d. Add Q(λ) to `AllGoalsUpdate`

The UVFA update uses eligibility traces. The trait should support:
```rust
fn td_target_lambda(
    &self,
    rewards: &[f32],
    next_q_max: &[f32],
    done: &[bool],
    gamma: f32,
    lambda: f32,
) -> Vec<f32>;
```

### 9e. Refine `AutocurriculumSampler` — seen_goals mask model

The source tracks `goals_seen` as a boolean mask over all goals, with `get_goals_seen()` computing which goals were observed in a batch of transitions. This is fundamentally different from incrementally adding to a `HashSet`:

```rust
/// Update observed goals from a batch of observations.
/// Returns updated boolean mask over all goals.
fn update_goals_seen(&self, obs_batch: &[Vec<f32>], all_goals: &[Vec<f32>]) -> Vec<bool>;
```

### 9f. Document BatchRenorm requirement

The source uses **Batch Renormalization** (not standard BatchNorm or LayerNorm). `BatchRenorm` is critical for stability with highly off-policy data. It uses running statistics with constrained corrections (r_max=3, d_max=5, warmup=1000 steps). This should be documented in the trait framework as a recommendation.

---

## Priority

**MEDIUM** — Framework only. Depends on riir-ai Plan 155 for game-specific implementations that prove the value. Ship the trait sockets first, let riir-ai fill in the plugs.

---

## References

- katgpt-rs Research 118 (full analysis)
- riir-ai Research 012 (game-specific mapping + Super GOAT rationale)
- 27_mmo_goat_pillars_decision_matrix.md (open/close boundary)
- `.raw/purejaxgcrl/dual_leo_pqn.py` — Dual LEO PQN (Q-learning UVFA)
- `.raw/purejaxgcrl/dual_leo_ppo.py` — Dual LEO PPO (actor-critic UVFA + BC)
- `.raw/purejaxgcrl/leo.py` — Standalone LEO (ablation)
- `.raw/purejaxgcrl/models/pqn_models_gc.py` — Network architectures
- `.raw/purejaxgcrl/envs/craftax/craftax_goals.py` — 512 goals for Craftax
- `.raw/purejaxgcrl/sweeps/craftax/dual_leo_pqn.yaml` — Sweep config
