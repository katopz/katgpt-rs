# Plan 093: CISPO Loss Variant + GRPO Group Wiring

> **Parent**: Research 57 (ART Agent Reinforcement Trainer Distillation)
> **Depends**: Plan 059 (G-Zero DPO/GRPO in `riir-gpu`) ✅
> **Scope**: Add CISPO loss variant to `riir-gpu/src/loss_grpo.rs`, wire trajectory grouping into `GZeroLoop`
> **Feature gate**: `cispo_loss` (off by default, proof via GOAT benchmark)

## Tasks

- [ ] T1: Add `GrpoLossVariant` enum (`PpoClip`, `Cispo`) to `loss_grpo.rs`
- [ ] T2: Implement CISPO loss function (detached ratio, wider clip, new_logprob multiply)
- [ ] T3: Add CISPO GPU kernel (`cispo_loss.wgsl`) to `riir-gpu`
- [ ] T4: Wire trajectory grouping into `GZeroLoop` (group_size rollouts → advantage)
- [ ] T5: GOAT benchmark: CISPO vs PPO-clip on bomber arena (1000 rounds)
- [ ] T6: If GOAT passes, document; if not, keep as opt-in feature gate

## Objective

Distill the one genuinely useful idea from ART (OpenPipe's Agent Reinforcement Trainer): the **CISPO** (Clipped Importance Sampling Policy Optimization) loss variant. Our current GRPO uses standard PPO-clip. CISPO detaches the importance ratio before clipping, uses a wider clip range (ε=1.0 vs 0.2), and multiplies by `new_logprobs` directly. ART reports this is more stable for agent training.

Additionally, wire the existing `GrpoConfig::group_size` into `GZeroLoop` so rollouts are properly grouped for advantage computation — ART's `TrajectoryGroup` pattern.

## Background: Why CISPO

From ART's `loss.py`:

```python
# Standard PPO-clip (our current approach):
policy_loss = -torch.min(
    prob_ratio * advantages,
    torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
)

# CISPO (ART's default, ppo=False):
policy_loss = -(
    torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)  # detached!
    * advantages
    * new_logprobs  # gradient flows through logprobs, not ratio
)
```

Key differences:
1. **Detached ratio** — gradient only through `new_logprobs`, prevents ratio explosion
2. **Wider clip** — default ε=1.0, ε_high=4.0 (vs PPO's 0.2) — allows larger policy shifts
3. **Logprob multiplication** — `clip(ratio) * advantage * logprob` instead of `min(ratio*adv, clip(ratio)*adv)`
4. ART found this works better for multi-step agent training than standard PPO-clip

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO Loss Pipeline                           │
│                                                                 │
│  ┌──────────────────┐     ┌────────────────────────────────┐   │
│  │  GrpoLossVariant  │     │  Existing (PpoClip)            │   │
│  │  enum             │     │                                │   │
│  │                   │     │  ratio = exp(new - old)        │   │
│  │  PpoClip ─────────┼────▸│  clipped = clamp(ratio, ...)  │   │
│  │  Cispo  ──────────┼──┐  │  loss = -min(r*adv, c*adv)    │   │
│  │                   │  │  └────────────────────────────────┘   │
│  └──────────────────┘  │                                        │
│                        │  ┌────────────────────────────────┐   │
│                        └─▸│  New (Cispo)                    │   │
│                           │                                │   │
│                           │  ratio = exp(new - old)        │   │
│                           │  clipped = clamp(detach(r),.)  │   │
│                           │  loss = -clip * adv * new_log  │   │
│                           └────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Trajectory Grouping (wire into GZeroLoop)               │   │
│  │                                                          │   │
│  │  Proposer → K rollouts per context                       │   │
│  │           → group_rewards: Vec<Vec<f32>>                  │   │
│  │           → group_advantage(rewards[group]) within-group  │   │
│  │           → GRPO loss with per-group advantages           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## T1: `GrpoLossVariant` Enum

**File**: `riir-ai/crates/riir-gpu/src/loss_grpo.rs`

```rust
/// GRPO loss variant selection.
///
/// CISPO (Clipped Importance Sampling Policy Optimization) is ART's default
/// loss for agent training. It detaches the importance ratio before clipping,
/// preventing ratio explosion during multi-step rollouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpoLossVariant {
    /// Standard PPO-clip: min(ratio * adv, clamp(ratio) * adv)
    PpoClip,
    /// CISPO: clamp(detach(ratio)) * adv * new_logprob
    /// Wider clip range, gradient only through logprobs.
    Cispo,
}

impl Default for GrpoLossVariant {
    fn default() -> Self {
        Self::PpoClip
    }
}
```

Add to `GrpoConfig`:

```rust
pub struct GrpoConfig {
    // ... existing fields ...
    /// Loss variant: PpoClip (standard) or Cispo (ART-style).
    pub loss_variant: GrpoLossVariant,
    /// CISPO lower clip bound (default: 1.0 — wider than PPO's 0.2).
    pub cispo_epsilon: f32,
    /// CISPO upper clip bound (default: 4.0 — wider than PPO's 1.2).
    pub cispo_epsilon_high: f32,
}
```

## T2: CISPO Loss Function (CPU)

**File**: `riir-ai/crates/riir-gpu/src/loss_grpo.rs`

```rust
/// CISPO loss computation (CPU version for testing parity).
///
/// Key differences from PPO-clip:
/// 1. Importance ratio is detached (gradient only through new_logprobs)
/// 2. Wider clip range (epsilon=1.0, epsilon_high=4.0 by default)
/// 3. Loss = -clip(detach(ratio)) * advantage * new_logprob
pub fn cispo_loss(
    old_logprobs: &[f32],
    new_logprobs: &[f32],
    advantages: &[f32],
    epsilon: f32,
    epsilon_high: f32,
) -> (f32, GrpoMetrics) {
    assert_eq!(old_logprobs.len(), new_logprobs.len());
    assert_eq!(old_logprobs.len(), advantages.len());

    let n = old_logprobs.len();
    if n == 0 {
        return (0.0, GrpoMetrics::default());
    }

    let mut total_loss = 0.0f32;
    let mut total_advantage = 0.0f32;
    let mut clip_count = 0usize;
    let mut ratio_sum = 0.0f32;
    let mut ratio_sq_sum = 0.0f32;

    for i in 0..n {
        // Importance ratio (detached — no gradient through this)
        let ratio = (new_logprobs[i] - old_logprobs[i]).exp();
        let clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon_high);

        // Track clip fraction
        if (ratio - clipped_ratio).abs() > 1e-6 {
            clip_count += 1;
        }

        // CISPO: -clip(detach(ratio)) * advantage * new_logprob
        // Gradient flows ONLY through new_logprobs[i], not through ratio
        let loss = -clipped_ratio * advantages[i] * new_logprobs[i];
        total_loss += loss;

        total_advantage += advantages[i].abs();
        ratio_sum += ratio;
        ratio_sq_sum += ratio * ratio;
    }

    let mean_loss = total_loss / n as f32;
    let mean_advantage = total_advantage / n as f32;
    let clip_fraction = clip_count as f32 / n as f32;

    // Entropy of importance ratios (exploration measure)
    let mean_ratio = ratio_sum / n as f32;
    let var_ratio = (ratio_sq_sum / n as f32) - (mean_ratio * mean_ratio);
    let entropy = (var_ratio + 1e-8).ln();

    let metrics = GrpoMetrics {
        loss: mean_loss,
        mean_advantage,
        clip_fraction,
        entropy,
    };

    (mean_loss, metrics)
}
```

Update `grpo_loss` dispatch:

```rust
pub fn grpo_loss(
    old_logprobs: &[f32],
    new_logprobs: &[f32],
    advantages: &[f32],
    clip_epsilon: f32,
) -> (f32, GrpoMetrics) {
    // ... existing PPO-clip implementation ...
}

/// Dispatch to the appropriate loss variant.
pub fn grpo_loss_with_variant(
    old_logprobs: &[f32],
    new_logprobs: &[f32],
    advantages: &[f32],
    config: &GrpoConfig,
) -> (f32, GrpoMetrics) {
    match config.loss_variant {
        GrpoLossVariant::PpoClip => grpo_loss(old_logprobs, new_logprobs, advantages, config.clip_epsilon),
        GrpoLossVariant::Cispo => cispo_loss(
            old_logprobs,
            new_logprobs,
            advantages,
            config.cispo_epsilon,
            config.cispo_epsilon_high,
        ),
    }
}
```

## T3: CISPO GPU Kernel

**File**: `riir-ai/crates/riir-gpu/src/kernels/cispo_loss.wgsl` (new)

```wgsl
// CISPO loss kernel: Clipped Importance Sampling Policy Optimization
// Based on ART's loss.py (OpenPipe)

struct CispoUniforms {
    num_elements: u32,
    epsilon: f32,
    epsilon_high: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> old_logprobs: array<f32>;
@group(0) @binding(1) var<storage, read> new_logprobs: array<f32>;
@group(0) @binding(2) var<storage, read> advantages: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: CispoUniforms;
@group(0) @binding(4) var<storage, read_write> output_loss: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_clipped: array<f32>; // clip indicator per element

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.num_elements) {
        return;
    }

    let ratio = exp(new_logprobs[idx] - old_logprobs[idx]);
    let clipped_ratio = clamp(ratio, 1.0 - uniforms.epsilon, 1.0 + uniforms.epsilon_high);

    // CISPO: -clip(detach(ratio)) * advantage * new_logprob
    // On GPU, "detach" means no backward pass through ratio — handled by kernel design
    output_loss[idx] = -clipped_ratio * advantages[idx] * new_logprobs[idx];

    // Track clipping for metrics
    let diff = abs(ratio - clipped_ratio);
    output_clipped[idx] = select(0.0, 1.0, diff > 1e-6);
}
```

**File**: `riir-ai/crates/riir-gpu/src/kernels/cispo_reduce.wgsl` (new)

```wgsl
// Reduction kernel for CISPO metrics: mean loss, clip fraction
// Same tree-reduction pattern as loss_reduce.wgsl

struct ReduceUniforms {
    num_elements: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> losses: array<f32>;
@group(0) @binding(1) var<storage, read> clipped_flags: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: ReduceUniforms;
@group(0) @binding(3) var<storage, read_write> result: array<f32>; // [0]=mean_loss, [1]=clip_fraction

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.num_elements) {
        return;
    }

    // Atomic-style reduction — in practice use workgroup shared memory
    // Simplified: each thread accumulates its portion
    let n = uniforms.num_elements;
    result[0] = result[0] + losses[idx] / f32(n);
    result[1] = result[1] + clipped_flags[idx] / f32(n);
}
```

Wire into `GpuPipelines` with `cispo_loss` and `cispo_reduce` pipeline registration.

## T4: Trajectory Grouping in GZeroLoop

**File**: `riir-ai/crates/riir-gpu/src/gzero_loop.rs`

The existing `GrpoConfig::group_size` (default: 16) exists but isn't used in the self-play loop. Wire it:

```rust
impl GZeroLoop {
    /// Run one GRPO training step with trajectory grouping.
    ///
    /// Groups rollouts by context, computes within-group advantages,
    /// then applies GRPO loss (PPO-clip or CISPO).
    pub fn train_grpo_grouped(
        &mut self,
        rollouts: &[RolloutResult],  // (context_id, reward, logprobs)
    ) -> Result<GrpoMetrics, GpuError> {
        let group_size = self.grpo_config.group_size;

        // Group by context_id
        let mut groups: HashMap<u64, Vec<&RolloutResult>> = HashMap::new();
        for r in rollouts {
            groups.entry(r.context_id).or_default().push(r);
        }

        let mut all_old_logprobs = Vec::new();
        let mut all_new_logprobs = Vec::new();
        let mut all_advantages = Vec::new();

        for (_ctx_id, group) in &groups {
            if group.len() < 2 {
                continue; // Need at least 2 for advantage
            }

            let rewards: Vec<f32> = group.iter().map(|r| r.reward).collect();
            let (advantages, _) = group_advantage(&rewards);

            for (i, rollout) in group.iter().enumerate() {
                all_old_logprobs.extend_from_slice(&rollout.old_logprobs);
                all_new_logprobs.extend_from_slice(&rollout.new_logprobs);
                all_advantages.push(advantages[i]);
            }
        }

        // Dispatch to appropriate loss variant
        grpo_loss_with_variant(
            &all_old_logprobs,
            &all_new_logprobs,
            &all_advantages,
            &self.grpo_config,
        )
    }
}
```

## T5: GOAT Benchmark

**Command**: `cargo test -p microgpt-rs --features "g_zero,bomber,cispo_loss" --test bench_gzero_cispo -- --nocapture`

**Benchmark plan**:
1. Run bomber arena 1000 rounds with PPO-clip (baseline)
2. Run bomber arena 1000 rounds with CISPO (variant)
3. Compare: survival rate, total score, δ mean, training stability (loss variance)

**GOAT criteria** (must pass ≥4/6):
- [ ] Survival rate ≥ baseline
- [ ] Total score ≥ baseline
- [ ] Training loss variance ≤ baseline (more stable)
- [ ] Clip fraction in reasonable range [0.05, 0.40]
- [ ] No NaN/Inf in loss
- [ ] `select_action` latency ≤ baseline

**Benchmark file**: `microgpt-rs/.benchmarks/018_cispo_vs_ppoclip.md`

## T6: Decision

If GOAT passes ≥4/6:
- Keep `cispo_loss` feature gate, document in README
- Consider making it default in future if more benchmarks confirm

If GOAT fails:
- Keep as opt-in feature gate
- Document as "experimental, no proven gain" (like `stepcode`)
- Do NOT make default

## Feature Gate Definition

**File**: `microgpt-rs/Cargo.toml` and `riir-ai/crates/riir-gpu/Cargo.toml`

```toml
[features]
cispo_loss = []  # CISPO loss variant for GRPO (ART-style, off by default)
```

**File**: `microgpt-rs/src/lib.rs`

```rust
#[cfg(feature = "cispo_loss")]
pub mod cispo;
```

## Module Structure

```text
riir-ai/crates/riir-gpu/src/
├── loss_grpo.rs          # + GrpoLossVariant enum, cispo_loss(), grpo_loss_with_variant()
├── kernels/
│   ├── cispo_loss.wgsl   # NEW: CISPO compute kernel
│   └── cispo_reduce.wgsl # NEW: CISPO reduction kernel
├── gzero_loop.rs         # + train_grpo_grouped() method
└── ...

microgpt-rs/src/
├── pruners/g_zero/       # No changes (modelless layer unaffected)
├── benchmark.rs          # + bench_cispo() behind feature gate
└── ...
```

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| CISPO worse than PPO-clip | Feature gate off by default, GOAT benchmark first |
| GPU kernel parity issues | CPU test first, GPU parity test second |
| Trajectory grouping memory | Group_size capped at 64, streaming groups |
| Widened clip causes instability | Default ε=1.0 from ART, tunable via config |

## Related

- Research 57: ART distillation verdict
- Research 21: G-Zero (Hint-δ)
- Research 25: StepCodeReasoner (Bi-Level GRPO)
- Plan 059: G-Zero DPO/GRPO in riir-gpu ✅
- Plan 049: G-Zero self-play (Phase 1 modelless, Phase 2 model-based)