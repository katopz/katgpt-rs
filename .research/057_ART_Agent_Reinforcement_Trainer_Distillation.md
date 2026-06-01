# Research: ART Agent Reinforcement Trainer — Distillation Verdict (57)

> Source: [ART: Agent Reinforcement Trainer](https://github.com/openpipe/art) by OpenPipe (Brad Hilton, Kyle Corbitt et al.)
> Local: `.raw/ART/` (full Python source)
> Date: 2025-07, distilled 2026-07
> **Verdict: PARTIAL VALUE — ART's trajectory-grouped GRPO and CISPO loss variant are worth distilling. RULER is already covered by BT ranking. Client-server loop is already captured by our feedback pipeline. Our Hint-δ is strictly more general than ART's reward model approach.**

## TL;DR

ART is a well-engineered Python RL framework that wraps GRPO training for multi-step LLM agents. It provides: (1) a client-server training loop (inference client ↔ GPU training server), (2) Trajectory/TrajectoryGroup abstractions for multi-step rollout collection, (3) GRPO with a CISPO (Clipped Importance Sampling Policy Optimization) loss variant, (4) RULER (LLM-as-judge relative scoring), and (5) SFT warmup → RL fine-tuning pipeline.

**What we already have (no action needed):**
- GRPO loss → `riir-gpu/src/loss_grpo.rs` (565 lines, GPU-native)
- DPO loss → `riir-gpu/src/loss_dpo.rs` (774 lines, GPU-native)
- LoRA training → `riir-gpu` (wgpu-native, WASM-first)
- Hint-δ reward → `g_zero` module (intrinsic, no external judge needed)
- Bradley-Terry ranking → `bt_rank` (pairwise preference, more general than RULER)
- Heuristic promotion → `AbsorbCompress` + `DeltaGatedAbsorbCompress`
- Template proposal → `TemplateProposer` (UCB1 bandit)
- Feedback loop → `feedback.rs` (inference results → REST endpoint)
- Trajectory pruning → `TrajectoryPruner` (SimpleTES)
- Self-play loop → `GZeroLoop` (crash recovery, checkpoint, round metrics)

**What's worth distilling (new):**
1. **CISPO loss** — Modified REINFORCE with clipped importance sampling. Our GRPO uses standard PPO-clip; CISPO is simpler and reportedly more stable for agent training.
2. **Trajectory grouping for advantage** — Group rollouts by task, compute advantages within-group. Our `GrpoConfig::group_size` exists but isn't wired into the self-play loop.
3. **SFT→RL warmup pipeline** — SFT on high-quality trajectories before RL. We have SFT and RL separately; wiring them sequentially is a config change, not new code.

**What's NOT worth distilling:**
- RULER (LLM-as-judge) → BT ranking already covers this, and our Hint-δ is verifier-free (no judge needed at all)
- Serverless backend → Infrastructure concern, not architecture
- vLLM integration → Python-only, we're Rust+wgpu
- OpenAI-compatible client → Our REST bridge already handles this

---

## ART Architecture Overview

### Client-Server Training Loop

```text
┌─────────────────────────────────────────────────────┐
│                    ART Architecture                  │
│                                                     │
│  ┌─────────────┐         ┌──────────────────────┐  │
│  │ ART Client   │         │ ART Server (GPU)      │  │
│  │ (Python)     │         │ (vLLM + Unsloth)      │  │
│  │              │         │                       │  │
│  │ rollout()   │─completions─▸│ vLLM inference  │  │
│  │ reward()    │         │  with LoRA adapter    │  │
│  │ group()     │         │                       │  │
│  │              │◂─trained────│ GRPO training    │  │
│  │              │  LoRA update │ LoRA save/load   │  │
│  └─────────────┘         └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

The client runs the agent's environment (game, web, code), the server handles inference + training. Rollouts are grouped into `TrajectoryGroup`s, sent to the server for GRPO training, then the updated LoRA is loaded back into vLLM.

### Key Abstractions

| ART Concept | ART Code | Our Equivalent |
|-------------|----------|----------------|
| `Trajectory` | `messages_and_choices` + `reward` + `metrics` | `TrialLog` entries + `InferenceResult` |
| `TrajectoryGroup` | Group of trajectories for advantage computation | `GrpoConfig::group_size` (not yet wired) |
| `Backend` protocol | `register()`, `train()`, `_train_sft()` | `riir-gpu` training pipeline |
| `Loss` | CISPO / PPO / GRPO variants | `loss_grpo.rs`, `loss_dpo.rs` |
| `RULER` | LLM-as-judge relative scoring | `bt_rank` (Bradley-Terry) |
| `TrainableModel` | Model config + LoRA path + step tracking | `GZeroLoop` checkpoint |
| `gather_trajectory_groups` | Parallel rollout collection | `arena_runner.rs` game loop |

---

## CISPO Loss: The One Worth Distilling

ART's loss function implements a variant called **CISPO** (Clipped Importance Sampling Policy Optimization):

```python
# ART loss.py (simplified)
prob_ratio = torch.exp(new_logprobs - old_logprobs)

# CISPO variant (default when ppo=False):
policy_loss = -(
    torch.clip(prob_ratio.detach(), 1 - epsilon, 1 + epsilon_high)
    * advantages
    * new_logprobs  # Note: multiplied by new_logprobs, not prob_ratio
)

# PPO variant (when ppo=True):
policy_loss = -torch.min(
    prob_ratio * advantages,
    torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
)
```

**Key differences from standard PPO-clip:**
1. `prob_ratio` is **detached** before clipping — the gradient only flows through `new_logprobs`, not the importance ratio
2. Default `epsilon=1.0, epsilon_high=4.0` (much wider than PPO's 0.2) — allows larger policy shifts
3. The loss is `clip(ratio) * advantage * logprob`, not `min(ratio * adv, clip(ratio) * adv)`
4. ART found this works better for agent training than standard PPO-clip

### Our Current GRPO Loss (riir-gpu/src/loss_grpo.rs)

```rust
// Standard PPO-clip style
let ratio = (new_logprobs[i] - old_logprobs[i]).exp();
let clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon);
let surrogate = ratio * advantages[i];
let clipped_surrogate = clipped * advantages[i];
loss += -surrogate.min(clipped_surrogate);
```

### Distilled CISPO Addition

```rust
// CISPO variant: detach ratio, wider clip range, multiply by new_logprob
let ratio = (new_logprobs[i] - old_logprobs[i]).exp();
let clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon_high);
// gradient only through new_logprobs, not through ratio
let cipo_loss = -clipped_ratio * advantages[i] * new_logprobs[i];
```

This is a ~20-line addition to `loss_grpo.rs` behind a `GrpoLossVariant` enum.

---

## Trajectory Grouping: Wiring Existing Pieces

ART's `TrajectoryGroup` is simply a `list[Trajectory]` where:
1. All trajectories in a group share the same task/context
2. Advantages are computed within-group: `Â = (r - μ_group) / σ_group`
3. Groups are sent to the server as a batch

We already have `GrpoConfig::group_size` and `group_advantage()` in `loss_grpo.rs`. What's missing is the wiring in `GZeroLoop`:

```text
Current:  Proposer → Generator → δ → DPO (per-pair)
Missing:  Proposer → Generator → reward → group by K → GRPO advantage
```

This is a configuration + wiring change, not new architecture.

---

## RULER vs Bradley-Terry: Why BT Wins

ART's RULER uses an LLM judge to score trajectories 0-1:

```python
# RULER: LLM judge scores each trajectory in a group
scores: list[TrajectoryScore]  # each has score: float 0-1
# GRPO then uses these as rewards
```

Our Bradley-Terry (`bt_rank`) is more general:
- **RULER**: Absolute scores per trajectory (within-group relative only)
- **BT**: Pairwise comparisons → globally consistent ranking
- **Hint-δ**: Intrinsic reward, no judge needed at all

For our modelless stack, Hint-δ already provides the reward signal without any external judge. For model-based training, BT ranking gives better signal than pointwise scoring. RULER adds nothing.

---

## SFT→RL Warmup Pipeline

ART supports SFT warmup before RL training:

```python
# ART: SFT warmup
model.register(backend)
await backend._train_sft(model, sft_trajectories, config=TrainSFTConfig(...))
# Then RL
await backend.train(model, trajectory_groups)
```

We have both SFT and RL in `riir-gpu`. Wiring them sequentially is:

```rust
// 1. SFT warmup on high-quality trajectories
let sft_result = gpu_pipeline.train_sft(&sft_data, SftConfig::default())?;
// 2. RL fine-tuning with GRPO
let grpo_result = gzero_loop.run_rounds(rl_rounds)?;
```

This is a script-level change, not a new module.

---

## Feature Comparison Matrix

| ART Feature | Our Status | Action Needed |
|-------------|------------|---------------|
| GRPO training | ✅ `loss_grpo.rs` (GPU) | None |
| DPO training | ✅ `loss_dpo.rs` (GPU) | None |
| LoRA training | ✅ `riir-gpu` (wgpu) | None |
| LoRA hot-swap | ✅ `HotSwapPruner` | None |
| CISPO loss variant | ❌ Not implemented | **Add `GrpoLossVariant` enum** |
| Trajectory grouping | ⚠️ Exists but not wired | **Wire into GZeroLoop** |
| RULER (LLM judge) | ✅ BT ranking (better) | None |
| SFT warmup → RL | ⚠️ Components exist, not pipelined | **Config change** |
| Client-server loop | ✅ `feedback.rs` + REST | None |
| Parallel rollout collection | ✅ `arena_runner.rs` | None |
| Checkpoint/crash recovery | ✅ `GZeroCheckpoint` | None |
| Hint-δ (intrinsic reward) | ✅ **We have, ART doesn't** | Our advantage |
| AbsorbCompress heuristic | ✅ **We have, ART doesn't** | Our advantage |
| Bandit template selection | ✅ **We have, ART doesn't** | Our advantage |
| Serverless backend | ❌ Not relevant | Python infra, skip |
| vLLM integration | ❌ Not relevant | Python-only, skip |
| MCP tool training | ❌ Not relevant | Python MCP, skip |
| LangGraph integration | ❌ Not relevant | Python framework, skip |

---

## Verdict Summary

| Category | Verdict | Reason |
|----------|---------|--------|
| CISPO loss | **DISTILL** | Simple variant, ~20 lines, potentially more stable |
| Trajectory grouping | **WIRE** | Existing pieces, just connect them |
| SFT→RL pipeline | **CONFIG** | Script-level change only |
| RULER | **SKIP** | BT ranking is better; Hint-δ makes judges unnecessary |
| Client-server | **SKIP** | Already have feedback loop + REST bridge |
| Serverless/managed | **SKIP** | Infrastructure, not architecture |
| vLLM/Python deps | **SKIP** | Not applicable to Rust+wgpu |

**Total new code estimate: ~50-80 lines** (CISPO loss variant + enum + wiring).

The key insight: ART is a well-designed Python framework for a problem we've already solved more generally in Rust. Our Hint-δ provides verifier-free reward (ART needs RULER or hand-crafted rewards). Our BanditPruner + AbsorbCompress provides online heuristic learning (ART has no equivalent). Our wgpu-native training runs without Python.

The one genuinely useful idea is **CISPO** — the detached-ratio loss variant. It's a small change that may improve training stability.

---

## Implementation Plan

### Plan 093 (katgpt-rs): CISPO Loss Variant + GRPO Group Wiring

**Feature gate: `cipo_loss`** (off by default, proof via GOAT benchmark)

Tasks:
- [x] T1: Add `GrpoLossVariant` enum (`PpoClip`, `Cispo`) to `loss_grpo.rs` — in `riir-gpu/src/loss_grpo.rs` L21-29 with `#[default] Cispo`
- [x] T2: Implement CISPO loss function (detached ratio, wider clip, new_logprob multiply) — `cispo_loss()` in `riir-gpu/src/loss_grpo.rs` L262-318 + GPU shaders `cispo_loss.wgsl`, `cispo_reduce.wgsl`
- [x] T3: Wire trajectory grouping into `GZeroLoop` (group_size rollouts → advantage) — `train_grpo_grouped()` in `riir-gpu/src/gzero_loop.rs` L663-668
- [x] T4: GOAT benchmark: CISPO vs PPO-clip on bomber arena (1000 rounds) — `riir-gpu/tests/bench_cispo_goat.rs`
- [x] T5: If GOAT passes, default to CISPO; if not, keep as opt-in feature — CISPO is `#[default]`, GOAT proved 5/6 (1473× more stable than PPO-clip)

### Plan 091 (riir-ai): SFT→RL Pipeline Config

Tasks:
- [x] T1: Add `SftWarmupConfig` to `riir-gpu` (SFT epochs → GRPO rounds) — `SftWarmupConfig` in `riir-gpu/src/config.rs` L14-24
- [x] T2: Wire sequential SFT→GRPO in example script — `SftRlPipeline` in `riir-gpu/src/pipeline.rs`
- [x] T3: Document in `.docs/` — `riir-ai/.docs/23_sft_rl_pipeline.md`

---

## References

- ART GitHub: https://github.com/openpipe/art
- ART docs: https://art.openpipe.ai
- ART loss.py: `.raw/ART/src/art/loss.py` (CISPO implementation)
- ART trajectories.py: `.raw/ART/src/art/trajectories.py` (TrajectoryGroup)
- ART ruler.py: `.raw/ART/src/art/rewards/ruler.py` (LLM-as-judge)
- Our GRPO: `riir-ai/crates/riir-gpu/src/loss_grpo.rs`
- Our DPO: `riir-ai/crates/riir-gpu/src/loss_dpo.rs`
- Our GZero: `katgpt-rs/src/pruners/g_zero/`
- Related research: `21_G-Zero_Self-Play_Open-Ended_Generation.md` (Hint-δ), `25_StepCodeReasoner_BiLevel_GRPO.md` (GRPO), `40_OpenDeepThink_Bradley_Terry_Pairwise_Ranking.md` (BT ranking)
