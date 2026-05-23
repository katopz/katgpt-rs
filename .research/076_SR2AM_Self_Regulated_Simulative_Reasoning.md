# Research 76: SR²AM — Self-Regulated Simulative Reasoning Agentic LLM

> Source: [Efficient Agentic Reasoning Through Self-Regulated Simulative Planning](https://arxiv.org/pdf/2605.22138) by Mingkai Deng, Jinyu Hou, Lara Sá Neves, Varad Pimpalkhute, Taylor W. Killian, Zhengzhong Liu, Eric P. Xing (IFM + CMU), May 2026
> Code: https://github.com/sailing-lab/sr2am
> Date: 2026-05, distilled 2026-07
> **Verdict: MEDIUM VALUE — Configurator concept (learned per-turn planning regulation) distills into our existing BanditPruner + early_exit stack. Simulative planning (System II) is partially covered by DDTree + ScreeningPruner; full world-model planning is out of scope for inference-time Rust. The key distillable insight: RL should deepen planning HORIZON, not increase planning FREQUENCY.**

## TL;DR

SR²AM decomposes agentic reasoning into three systems: reactive execution (System I), simulative planning via world model (System II), and a learned configurator that decides when/how deeply to plan (System III). The key empirical finding: **RL with self-regulation increases average planning horizon by 22.8% while planning frequency grows only 2.0%** — the model learns to plan *further ahead*, not *more often*. This is directly applicable to our `BanditPruner` Q-value regulation and `early_exit_patience`/`early_exit_gap` parameters.

**What we already have (no action needed):**
- System I (reactive execution) → `ConstraintPruner`, `NoScreeningPruner`, direct token sampling
- Partial System III (static regulation) → `early_exit_patience` + `early_exit_gap` in `Config`
- Partial System III (domain budget) → `tree_budget`, `draft_lookahead` per domain in riir-ai
- Partial System II (forward model) → `GameState` forward model (STRATEGA), `GoState` (AutoGo)
- Model-based signal → Hint-δ (`DeltaBanditPruner`, `DeltaGatedAbsorbCompress`)
- RL training loop → `GZeroLoop` + `loss_grpo.rs` + `loss_dpo.rs` in riir-gpu
- Plan reconstruction → Already do this for game traces (Plan 039, Plan 056)

**What's worth distilling (new):**
1. **Configurator enum** — Learned per-turn `PlanningDecision { Plan, Continue, Skip }` added to our `BanditPruner` as a third action alongside Explore/Exploit
2. **Horizon-deepening reward** — RL reward shaping that incentivizes longer DDTree chains when configurator chooses Plan, not more frequent planning. Directly applicable to `GZeroLoop` reward function.
3. **Plan horizon truncation** — For high-uncertainty domains (web), truncate planning to 2 steps. Maps to our `draft_lookahead` per-domain config in riir-ai.

**What's NOT worth distilling:**
- Full simulative planning with LLM-as-world-model → We don't run LLM planning at inference time in Rust; this is a Python-side orchestration pattern
- Multi-module inference (v0.1) → Overengineered; our `TemplateProposer` already covers rule-based query generation
- GRPO with asymmetric clipping → Already in `loss_grpo.rs` (Plan 093 CISPO)
- Web search / browser tools → Infrastructure concern, not architecture

---

## Core Architecture: Three-System Decomposition

```
┌─────────────────────────────────────────────────────────┐
│                  SR²AM Three Systems                     │
│                                                          │
│  System III: CONFIGURATOR κ                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Input: belief state ŝ_t                          │    │
│  │ Output: decision u_t ∈ {Plan, Continue, Skip}    │    │
│  │ Learned via RL — decides WHEN and HOW DEEP       │    │
│  └──────────────────────┬──────────────────────────┘    │
│                         │                                │
│           ┌─────────────┼─────────────┐                  │
│           ▼             ▼             ▼                  │
│     u=Plan         u=Continue    u=Skip                  │
│           │             │             │                  │
│  System II:          System II:    System I:             │
│  SIMULATIVE          CONTINUE      REACTIVE              │
│  PLANNER π_f         PLANNING      EXECUTION α           │
│  ┌───────────┐       ┌──────────┐  ┌──────────┐        │
│  │New plan:  │       │Extend    │  │Direct    │        │
│  │ŝ_t,a'_t,  │       │existing  │  │action    │        │
│  │ŝ_{t+1},   │       │plan by   │  │a_t ~     │        │
│  │a'_{t+1},  │       │one step  │  │p_α(·|ŝ_t)│        │
│  │...ŝ_{T'}  │       │          │  │          │        │
│  └───────────┘       └──────────┘  └──────────┘        │
│                                                          │
│  World Model f: predicts ŝ_{t+1} given ŝ_t and a'_t     │
│  (LLM itself serves as world model in language space)     │
└─────────────────────────────────────────────────────────┘
```

### Formal Decomposition (Equation 4)

The key equation showing the three-system action distribution:

```text
p_π(a_t | ŝ_t) = Σ_{u_t, c_t}  p_α(a_t | ŝ_t, c_t)     ← System I (actor)
                                × p_{π_f}(c_t | ŝ_t, u_t) ← System II (planner)
                                × p_κ(u_t | ŝ_t)          ← System III (configurator)
```

Where:
- `u_t` = configurator decision (Plan / Continue / Skip)
- `c_t` = structured plan = (ŝ_t, a'_t, ŝ_{t+1}, a'_{t+1}, ..., ŝ_{T'})
- `a_t` = concrete action from actor

**Contrast with unregulated deliberation (Equation 3):**

```text
p_π(a_t | ŝ_t) = Σ_{z_t}  p_π(a_t | ŝ_t, z_t) × p_π(z_t | ŝ_t)
                                     ↑
                        unstructured latent z_t (no state prediction, no planning control)
```

---

## Mapping to Our Architecture

### System I → Reactive Execution (Already Covered)

| SR²AM Component | Our Equivalent | Notes |
|----------------|---------------|-------|
| Actor α | `sample_token()` + `ConstraintPruner` | Direct action selection |
| Free-form reasoning z_t | `NoScreeningPruner` path | Skip screening, direct output |
| Tool execution | `validator` module (WASM) | External action execution |

**Verdict: No new code needed.**

### System II → Simulative Planning (Partially Covered)

| SR²AM Component | Our Equivalent | Gap |
|----------------|---------------|-----|
| World model f | `GameState` forward model (STRATEGA) | ✅ Game domains covered |
| World model f | `GoState` + MCTS (AutoGo) | ✅ Go covered |
| Plan structure c_t | DDTree node sequences | ⚠️ Implicit — no explicit state-action-state encoding |
| Plan reconstruction | Game replay training (Plan 039) | ✅ Trace reconstruction exists |
| LLM-as-world-model | N/A | ❌ Not applicable — we don't run LLM planning in Rust |

**Key gap:** SR²AM's plans encode `(belief_state, proposed_action, predicted_next_state)` tuples. Our DDTree nodes encode `(token_sequence, score)` pairs without explicit state prediction. However, for our game domains (Bomber, Go, FFT), the forward model already provides next-state prediction — the plan structure is implicit in the tree.

**Verdict: For game domains, System II is already covered by MCTS + forward model. For LLM inference, simulative planning is a Python-side orchestration concern — out of scope for Rust runtime.**

### System III → Self-Regulation (Key Distillable Idea)

| SR²AM Component | Our Equivalent | Gap |
|----------------|---------------|-----|
| Configurator κ (learned) | `BanditPruner` Q-values | ⚠️ Q-values decide arm selection, not planning depth |
| Configurator decision u_t | `early_exit_patience` + `early_exit_gap` | ❌ Static threshold, not learned |
| Planning frequency control | `tree_budget` | ❌ Fixed budget, not per-turn adaptive |
| Horizon control T' | `draft_lookahead` | ❌ Fixed depth, not uncertainty-adaptive |
| Per-turn reassessment | N/A | ❌ We don't reassess planning need mid-sequence |

**This is the actionable insight.** Our `early_exit_patience` is a static threshold. SR²AM's configurator is a **learned** per-turn decision. The mapping:

```text
SR²AM                          Our Stack (proposed)
─────────                      ────────────────────
u_t = Plan (new)          →   BanditPruner arm: "plan_new"     (reset tree, full budget)
u_t = Continue (extend)   →   BanditPruner arm: "plan_extend" (keep tree, add depth)
u_t = Skip (react only)   →   BanditPruner arm: "plan_skip"   (early_exit, direct sample)
```

The configurator decision becomes a third bandit dimension alongside the existing Explore/Exploit arms.

---

## Key Empirical Findings

### 1. Horizon Deepening, Not Frequency Increase

**The single most important result.** After RL training:

| Metric | Pre-RL | Post-RL | Change |
|--------|--------|---------|--------|
| Average planning horizon | 1.67 steps | 2.05 steps | **+22.8%** |
| Planning frequency (% turns with plan) | 15.6% | 13.6% | **+2.0pp** (decreased!) |
| 2-3+ step plans | 5.3% of turns | 14.9% of turns | **+9.6pp** |
| 0-step (skip planning) | 15.6% | 13.6% | Stable |

**Implication for us:** When we train `GZeroLoop` with GRPO, the reward should incentivize **deeper DDTree chains when planning is chosen**, not more frequent tree building. This is a reward shaping insight, not an architecture change.

### 2. Component Ablation Confirms All Three Systems Matter

| Ablation | Pass@1 | Tokens | What's Removed |
|----------|--------|--------|----------------|
| Full SR²AM | 66.6 | 4,925 | — |
| − Free-form reasoning (System I) | **46.8** | 1,188 | Largest drop — structured + free-form are complementary |
| − Simulative planning (System II) | 65.2 | 4,602 | Moderate drop — state prediction helps |
| − Selective planning (System III) | 65.2 | **5,451** | Token waste — configurator controls efficiency |
| − Plan horizon control (System III) | 65.3 | 4,829 | Minor drop — uncertainty-aware truncation |
| Original teacher CoT (no structure) | 65.3 | 3,844 | Baseline — our structure beats it by 1.3 pp |
| Full + RL | **72.8** | 5,414 | RL lifts +6.2 pp with moderate token growth |

**Key insight:** Removing System I (free-form reasoning) causes the *largest* accuracy drop. Removing System III (selective planning) causes the *largest* token increase. Both are important — one for quality, one for efficiency.

### 3. Unregulated Deliberation Diverges Under RL

When comparing SR²AM-v0.1-8B (self-regulated) vs Qwen3-8B (unregulated) over 400 RL steps:

| Metric | Self-Regulated | Unregulated |
|--------|---------------|-------------|
| Tokens at step 400 | ~3,600 | ~6,200+ (growing) |
| Pass@1 at step 400 | 56.2 | 47.6 |
| Out-of-context rate | 5.3% | 22.4% |

Unregulated deliberation increases token consumption with **diminishing accuracy returns** and rising context overflow. Self-regulated channels improvement through planning quality.

### 4. 8B Competitive with 120-355B

SR²AM-v0.1-8B achieves overall Pass@1 of 57.0, competitive with:
- GPT-OSS-120B-high (120B, 60.3)
- Qwen3-235B (235B, 57.0)
- GLM-4.6 (357B, 60.7)

While using only 3,698 reasoning tokens per trajectory (fewer than most 7-8B baselines at 601-11,206).

---

## Distillable Ideas Ranked by Applicability

### Tier 1: Directly Applicable (Implement in Plan 112)

1. **Configurator as Bandit Arm** — Add `PlanningDecision` enum to `BanditPruner`. Three arms: `PlanNew`, `PlanExtend`, `PlanSkip`. Q-values learned from reward signal. Maps to existing bandit infrastructure.

2. **Horizon-Deepening Reward Shaping** — In `GZeroLoop`, shape reward to incentivize deeper DDTree chains when `PlanNew` or `PlanExtend` is chosen. Penalize token waste without quality gain. Concrete formula: `reward += α * (plan_depth_when_chosen / max_depth) - β * (tokens_used / budget)`.

3. **Uncertainty-Aware Horizon Truncation** — For high-uncertainty domains, limit `draft_lookahead` to 2. Already configurable per-domain in riir-ai's TOML configs. Just needs the heuristic: if `entropy > threshold`, cap `draft_lookahead` at 2.

### Tier 2: Worth Exploring (Feature-Gated Proof)

4. **Plan Reconstruction from Traces** — Given game replay traces (Plan 039), reconstruct structured `(state, action, predicted_next_state)` plans. Annotate existing MCTS traces with explicit state predictions. Use as SFT data for configurator training.

5. **Configurator Accuracy Metric** — Track `configurator_agreement = % of turns where configurator's decision matches hindsight-optimal`. Add to `InferenceResult` metrics. Requires post-hoc analysis of whether planning was beneficial.

### Tier 3: Out of Scope (Not Applicable)

6. **LLM-as-World-Model** — Using the LLM itself to predict future states in language space. We don't run LLM planning at inference time in Rust. This is a Python-side orchestration pattern for agentic workflows.

7. **Multi-Module Inference (v0.1)** — Separate prompted LLMs for planning, reflection, summary. Overengineered for our use case. Our `TemplateProposer` already covers rule-based generation.

8. **Web Search / Browser Tools** — Infrastructure concern. Not architecture.

---

## Proposed Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│         Our Stack with SR²AM Configurator (Plan 112)         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Configurator Bandit (new)                            │   │
│  │                                                      │   │
│  │  Arms: PlanNew | PlanExtend | PlanSkip               │   │
│  │  Signal: δ from DeltaBanditPruner (existing)         │   │
│  │  Reward: quality_gain / tokens_used                  │   │
│  │  Q-value update: UCB1 (existing bandit infra)        │   │
│  └─────────────┬────────────────────────────────────────┘   │
│                │                                             │
│      ┌─────────┼─────────┐                                  │
│      ▼         ▼         ▼                                   │
│  PlanNew   PlanExtend  PlanSkip                              │
│      │         │         │                                   │
│  Reset tree  Add depth  Early exit                           │
│  Full budget  +1 level   Direct sample                       │
│      │         │         │                                   │
│  DDTree     DDTree     sample_token()                        │
│  build()    extend()   + ConstraintPruner                    │
│      │         │         │                                   │
│      └────┬────┘         │                                   │
│           ▼              ▼                                   │
│    ScreeningPruner   BanditPruner<P>                         │
│    (relevance())     (explore/exploit)                       │
│           │              │                                   │
│           └──────┬───────┘                                   │
│                  ▼                                           │
│           InferenceResult                                    │
│           + configurator_decision                            │
│           + plan_depth_used                                  │
│           + plan_skip_savings                                │
└─────────────────────────────────────────────────────────────┘
```

### Feature Gate

```toml
[features]
sr2am_configurator = ["bandit"]  # Configurator bandit arm
```

All new code gated behind `#[cfg(feature = "sr2am_configurator")]`.

---

## Verdict Summary

| Aspect | Verdict | Rationale |
|--------|---------|-----------|
| System I (Reactive) | ✅ Already covered | `ConstraintPruner` + `sample_token()` |
| System II (Simulative) | ⚠️ Partially covered | MCTS + forward model for games; LLM planning out of scope |
| System III (Configurator) | 🔧 **Key distillable idea** | Replace static `early_exit_patience` with learned bandit arm |
| Horizon deepening reward | 🔧 **Directly applicable** | Shape `GZeroLoop` reward to prefer deeper plans over more plans |
| Plan reconstruction | 📋 Worth exploring | Annotate game traces with state predictions for SFT |
| LLM-as-world-model | ❌ Out of scope | Python-side orchestration, not Rust inference |

**Overall: MEDIUM VALUE.** The configurator concept (learned per-turn planning regulation) is the most novel and applicable part. It maps cleanly onto our existing `BanditPruner` infrastructure and can be feature-gated for GOAT proof. The simulative planning (System II) is already covered by our MCTS + forward model for game domains, and LLM-as-world-model is out of scope for Rust runtime.

**Risk:** Low. The configurator is additive — it wraps existing DDTree + early_exit logic behind a bandit decision. No existing code changes; all new code behind feature gate.

**GOAT proof strategy:** Run Bomber/Go arena with and without configurator. Metric: same win rate, fewer tokens (efficiency gain from smarter planning decisions).