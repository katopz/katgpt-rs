# Research 363: Finding the Time to Think — Learning Planning Budgets in Real-Time RL

> **Source:** Muppidi, Darwish, Cope, Henriques, Foerster — "Finding the Time to Think: Learning Planning Budgets in Real-Time RL" (Oxford BOLD + VGG, 2026-06-30). [arXiv:2606.26463](https://arxiv.org/abs/2606.26463) · [code](https://github.com/Aneeshers/realtime-rl-code) · [project page](https://aneeshers.github.io/realtime-rl/)
> **Date:** 2026-07-02
> **Status:** Done — **Pass**. Training-only contribution; the runtime analog (state-dependent compute budget) already ships modellessly under different vocabulary. Two fusion ideas tracked as issues.
> **Related Research:** 149 (per-NPC gain/cost reasoning depth guide — the **direct analog**), 282 (LoopCoder-v2 gain/cost halting distillation), 218 (breakeven complexity router), 350 (density-aware compute scheduling), 136 (per-NPC runtime test-time scaling), 163/318 (sleep-time query anticipation), 212 (collapse-aware adaptive thinking)
> **Related Plans:** 304 (`GainCostLoopHalter` open primitive), 194 (adaptive CoT bandit), 212 (collapse-aware), 263 (cumprodsum freshness-driven thinking budget), 231 (PathwayTracker — 85% thinking-budget savings, GOAT-promoted)
> **Cross-ref (riir-ai):** [Research 149](../../riir-ai/.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md) — the per-NPC selling-point guide that *is* the modelless analog.
> **Classification:** Public (katgpt-rs)

---

## TL;DR

The paper introduces **variable-delay real-time RL**: the environment progresses every frame regardless, and a PPO-trained gating policy chooses a state-dependent MCTS budget `k ∈ K` at each decision point. The agent emits `k−1` cheap "reflex" actions while MCTS runs, then applies the planned action. Across Pac-Man, real-time Tetris, Snake, Speed Hex, and Speed Go, the gating policy beats every fixed-budget and heuristic baseline (+10–65% over best fixed-`k`), and transfers cleanly to a two-GPU asynchronous deployment with no retraining.

**Distilled for katgpt-rs (modelless, inference-time):**

The paper's headline contribution is a **training loop** — PPO with variable-duration GAE (`γ^k` per-step discount correction) over budgeted options (semi-MDP over `k`-long anytime-compute options). The gating policy, the frozen AlphaZero base, and the GAE discount correction are all training-bound → **riir-train** (one-line note, stop).

What survives as a *modelless primitive* is the paper's **meta-reasoning insight** — *the cost of deliberation is endogenous to the environment, so the budget should be state-dependent* — but that insight **already ships modellessly** under the codebase's own vocabulary, in three independent implementations:

1. **Per-NPC Gain/Cost Reasoning Depth** (R149/282, Plan 304 — `GainCostLoopHalter`). Each NPC auto-selects its own reasoning depth via `Halt when Gain(r) < Cost(r) × τ`. This *is* the modelless analog of the paper's state-dependent budget gate: the budget is selected per decision from runtime signals (effective rank delta, coherence decay, update-direction alignment), with no PPO, no learned policy, no gradient descent.
2. **Breakeven Complexity Router** (R218). The paper's "value of computation" framing (Russell & Wefald 1991 VOC = expected quality gain − cost) is exactly R218's breakeven `N* = B / (C_δB − C_inf)`, already distilled for LLM inference routing (full-attention vs sparse vs speculative).
3. **Density-Aware Compute Scheduling** (R350). The paper's "Tetris board density drives budget" finding is the per-zone analog of R350's `(mobility_weight, tier_class, cache_key)` classifier — sparse zones get full recompute (high movement freedom → high event entropy), dense zones get LRU-cached projections.

The only genuine novelty the paper adds beyond shipped primitives is the **real-time interaction protocol** (environment runs concurrently on a second GPU while planner computes) — and that is a systems/deployment concern, not a research-primitive concern.

---

## 1. Paper Core Findings

### 1.1 Variable-delay real-time RL (the formalism)

Generalizes Ramstedt & Pal (2019)'s fixed-1-step-delay real-time MDP: the agent is a *procedure that runs in time*, not a function. The environment advances one frame per `t` regardless; whatever action the agent has submitted by frame `t` is applied, with a fallback no-op if none. The agent picks `k` per decision point.

This is encoded as a **semi-MDP over budgeted options**: each option `o_k` runs computation `c_k` for exactly `k` frames, emits `k−1` reflex actions during the wait window, applies `c_k`'s output on the `k`-th frame. Meta-Bellman:

```
V(s_t) = E_{k~π_gate}[ Σ_{j=0}^{k-1} γ^j r_{t+j}  +  γ^k V(s_{t+k}) ]
```

The key correction vs ordinary GAE: the per-step discount is `γ^k`, not `γ` (Appendix C). Without it, value function is biased toward many short budgets.

### 1.2 The adaptive gating policy (the training contribution)

A lightweight network (1×1 conv → 3 residual blocks → global avg pool → MLP) takes three inputs at each meta-step:
1. Raw game observation
2. Frozen planner's intermediate spatial features (trunk output)
3. Frozen planner's scalar value estimate `V(s_t)` (and remaining clock for clock envs)

Produces a distribution over `k` and a baseline value. Trained with PPO on top of the **frozen** AlphaZero base (no joint training). Two-phase training: train base planners via self-play → freeze → train gate.

### 1.3 When does the gate plan deeply?

Empirically (Figure 5), the gate allocates compute where *the consequence of a suboptimal action is large* AND *additional search is likely to change the decision*:

| Env | Trigger for deep `k` |
|-----|----------------------|
| Pac-Man | Nearest ghost *far* (can afford to plan); becomes reactive when ghost is *close* |
| Tetris RT | High board density (placement precision matters); bimodal "react or plan deeply" — `k=3` never selected |
| Snake | Spatial constraint (low reachability, body-growth events) |
| Speed Hex/Go | Budget-conditioned: small clock → cheap option; large clock → spread mass |

### 1.4 Real-time deployment transfer

The committed-action training protocol (MCTS simulates the same `k−1` reflex steps the agent will execute) **transfers to a two-GPU deployment with no architectural change**: GPU0 runs env+reflex, GPU1 runs MCTS in parallel. 45 deployment cells (3 envs × 3 GPUs × 5 FPS) confirm simulation-trained policies transfer cleanly; H100 stays reliably in-budget, A40 breaks down only at tightest FPS.

### 1.5 Limitations (paper §8)

- Requires perfect simulator (commits `k−1` steps inside MCTS tree); MuZero-style learned-dynamics planner is open.
- Frozen base planner (no joint optimization); Appendix F shows base-checkpoint choice matters.
- Hand-calibrated discrete budget set `K` per environment.

---

## 2. Distillation

### 2.1 What's training-only (→ riir-train)

- The PPO gating-policy training loop (variable-duration GAE, `γ^k` discount correction, PPO clip).
- The AlphaZero base-planner self-play training.
- The frozen-base-then-train-gate two-phase recipe.
- Any variant that jointly optimizes planner + gate.

**All of the paper's empirical wins are gated behind this training.** A reproduction requires GPU-days on H100. → riir-train one-line note, stop.

### 2.2 What's modelless (stays in katgpt-rs / riir-ai)

The **insight** — state-dependent compute budget where the cost of deliberation is endogenous — is fully modelless in the codebase. Three existing implementations, each a different vocabulary for the same Russell & Wefald (1991) value-of-computation idea:

| Codebase (modelless) | Paper analog | Mechanism | Status |
|---|---|---|---|
| **Per-NPC Gain/Cost Halter** (R149, R282, P304) | Gating policy selects `k` per state | `halt when Gain(r) < Cost(r) × τ`; gain = effective-rank delta / output shift / coherence improvement; cost = staleness / drift / coherence decay. **Per-NPC**, not per-agent. | Plan 304 scoped; selling-point guide R149 shipped |
| **Breakeven Complexity Router** (R218) | VOC = expected quality gain − cost | `N* = B / (C_δB − C_inf)` — counts forward solves before an approximate method becomes cost-effective vs an error-equivalent exact one | R218 GOAT verdict: HIGH GAIN |
| **Density-Aware Compute Scheduling** (R350) | "Tetris board density drives budget" | `sigmoid(-β·(ρ−ρ₀))` → `(mobility_weight, tier_class, cache_key)` per zone; sparse recompute, dense cache | R350 active |
| **PathwayTracker** (P231, GOAT-promoted) | "React when converged, plan when divergent" | Intrinsic pathway stability detection; **85% thinking-budget savings, 100% convergence-detection accuracy, 0 false early exits** | Default-on |
| **Cumprodsum Freshness Gate** (P263) | "React on stale, plan on fresh" | `context_freshness = mean(cumprodsum(decay_factors))`; `thinking_budget = base + max_extra × sigmoid(β·(freshness − threshold))`; stale=7 blocks, fresh=1 block | Default-on |
| **Adaptive CoT Bandit** (P194, R283) | "When to think" bandit | FrequencyBandit learns when to think; collapse-aware extension (P212) adds early-exit on divergence | Default-on |

### 2.3 Latent-space reframing (mandatory before verdict)

The paper gates on **raw game features**: ghost distance, board fill fraction, reachable cells, remaining clock. Our reframe: gate on **HLA latent state** (per-NPC 8-dim affective: valence/arousal/desperation/calm/fear + 3).

```text
// Paper (raw):
gate_k = MLP([raw_obs, planner_trunk_features, V(s), clock])
       → softmax over K

// Codebase analog (latent, modelless, per-NPC, sigmoid):
k_npc = halt_when( gain(r) < cost(r) × τ )
  where:
    gain(r) = sigmoid( dot(Δh_r, d_curiosity) )  // HLA curiosity direction
    cost(r) = sigmoid( dot(h_r,    d_staleness) ) // coherence-decay direction
    h_r     = HLA belief at loop r                 // per-NPC, latent, fog-of-war gated
```

The latent reframing is **strictly stronger** than the paper's raw-feature gate for three reasons:

1. **Crowd-scale**: the paper trains one gate per agent. The HLA reframing scales to 10,000 concurrent NPCs each with independent depth selection from a single frozen artifact (R149 selling point).
2. **Fog-of-war compatible**: the gate operates on the *think brain* (per-NPC `SpatialBelief`, latent, NOT synced). The info brain (real `MapPos`, synced) is untouched. The paper's gate assumes the agent sees the true state.
3. **No training**: gain/cost curves are deterministic functions of (synced) input hidden states — no PPO, no GPU-days, no frozen base planner. The "frozen base" the paper requires is replaced by freeze/thaw-committed direction vectors (BLAKE3-checked, atomic Arc swap).

**Sync boundary**: only the resulting halt count `L` (a raw scalar) crosses sync, exactly as R149 §3 documents. Anti-cheat and deterministic replay unaffected.

### 2.4 Fusion opportunities (the angle worth recording)

The paper contributes one thing the codebase primitives do NOT have: the **two-GPU asynchronous deployment protocol** (env on GPU0, planner on GPU1, committed reflex actions bridge the latency gap). That is a systems/deployment concern — not a research primitive — but it suggests two concrete fusion ideas worth tracking:

| Fusion | What it gains | Mechanism | Track as |
|---|---|---|---|
| **Asynchronous Plasma-tier NPC cognition** | Bridge the 20Hz tick budget for hard NPC decisions by running deep cognition on a warm-tier worker while the plasma-tier reflex policy holds the NPC's actions for `k` ticks | Mirror the paper's committed-action protocol: reflex policy = HLA evolve_hla single-step (already sub-µs); planner = full `latent_functor` application + CLR vote; on tick `t+k`, apply planner output. The `k` is selected per-NPC by `GainCostLoopHalter` (P304). | **Issue 364** (perf engineering, not research) — [riir-ai/.issues/364](../../../riir-ai/.issues/364_async_plasma_tier_npc_cognition.md) |
| **Variable-duration CGSP** | Generalize CGSP from fixed-`k` rollouts to per-NPC-`k` rollouts where `k` comes from gain/cost halting, with `γ^k` discount correction (Appendix C) carried through GAE | **Honesty correction (2026-07-02):** the original framing ("replace γ with γ^k in the advantage computation") overstated the scope — CGSP's update path is bandit/priority-based (`PriorityTableBandit` + `hint_delta`), NOT a γ-discounted advantage. There is no γ to replace. The actual ship is a **staleness weight** `exp(-λ·(k_npc−1))` on the priority hint_delta (the bandit-update analog of γ^k); the literal γ^k correction is a deferred hook for any future value/advantage path. See Issue 365 §"The correction". | **Issue 365** (runtime extension to Plan 304) — [riir-ai/.issues/365](../../../riir-ai/.issues/365_variable_duration_cgsp_staleness_weight.md) |

Neither fusion is novel enough to be a Super-GOAT (both are combinations of shipped primitives applied to a deployment concern the paper introduced). Both belong as `.issues/` follow-ups, not plans.

### 2.5 §3.6 Defend-wrong PoC reasoning

I am **not** claiming quality parity between the shipped modelless primitives and the paper's PPO-trained gate. The claims I am making:

| Claim | Type | Evidence | Status |
|---|---|---|---|
| The runtime analog (state-dependent compute budget) exists | Architectural | R149, R282, R218, R350, P231, P263, P304 (all shipped or scoped) | **Proven** by grep + read |
| The modelless analog is sub-µs / zero-GPU | Latency | P231 GOAT: update 123 ns, stability 2.7 µs; P263: 1-block vs 7-block clamping | **Proven** by existing benchmarks |
| The modelless analog **matches or beats** the paper's +10–65% gain over best fixed-`k` | **Quality** | Not measured head-to-head on Pac-Man/Tetris/Snake/Hex/Go | **Unproven — not claimed** |

The third claim would require a PoC in `riir-poc/` running `GainCostLoopHalter` + HLA gating vs fixed-depth on a controlled toy domain. I am NOT making that claim — the verdict is Pass on architectural + training-only grounds, not on quality-parity grounds. If a future plan wanted to claim "our modelless gate beats the paper's PPO gate on the paper's own benchmark", §3.6 mandates the PoC. **Tracked as an optional follow-up, not a blocking one.**

### 2.6 §3.5 Modelless unblock check (for completeness)

If someone proposed "train a gating policy like the paper to manage NPC planning budgets", the §3.5 protocol returns MODELLESS-VALIDABLE before reaching riir-train:

1. **Freeze/thaw** — can a frozen snapshot fix the "which budget for which state" question? → YES: the gain/cost curves freeze into per-archetype direction vectors (R149 §3). **Path 1 succeeds.**
2. **Raw/lora hot-swap** — can a deterministically constructed reader-LoRA fix it? → N/A; no weight correction needed, the gate is a runtime decision not a weight update.
3. **Latent-space correction** — can a sigmoid projection fix it? → YES: `gate = sigmoid(dot(h, d_curiosity) − dot(h, d_staleness))` (§2.3 above). **Path 3 succeeds.**

Conclusion: a "trained gating policy" is NOT required. The modelless paths cover the mechanism. → No riir-train dependency for the runtime primitive; riir-train only if someone specifically wants to reproduce the paper's PPO loop as a training-method artifact.

---

## 3. Verdict

**Pass.**

One-line reasoning: **the paper's headline contribution is a training loop (PPO + variable-duration GAE over budgeted options); the modelless insight it rests on — state-dependent compute budget with endogenous deliberation cost — already ships in the codebase under different vocabulary (gain/cost halting R149/P304, breakeven complexity R218, density-aware scheduling R350, PathwayTracker P231, cumprodsum freshness P263), and the latent reframing on HLA state is strictly stronger (crowd-scale, fog-of-war compatible, no training).**

### Tiers (high → low) — applied

| Tier | Criteria | Routing | This paper |
|---|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + force multiplier | Open primitive + private guide + plans | ✗ — mechanism ships (R149/R218/R350), no new capability class |
| GOAT | Provable gain over existing approach | Plan + feature flag + benchmark | ✗ — gain is on the paper's training loop, not over our shipped primitives; would need PoC (§2.5) |
| Gain | Incremental improvement | Plan only, feature flag | ✗ — the deployment-protocol fusion (§2.4) is a perf-engineering issue, not a primitive |
| **Pass** | Training-only OR not relevant OR already ships | One-line note, no files in session | **✓ — training loop → riir-train; runtime analog ships; deployment protocol → issue** |

### Novelty gate (Q1–Q4)

1. **No prior art?** NO. Vocabulary-translated grep across all 5 repos returns direct hits: R149 (`gain/cost halting`), R282 (`GainCostLoopHalter`), R218 (breakeven complexity = VOC), R350 (density-aware compute scheduling), P231 (PathwayTracker — 85% thinking-budget savings), P263 (cumprodsum freshness gate). The mechanism ships under at least six different names.
2. **New class of behavior?** NO. State-dependent compute budget selection is the codebase's bread and butter — every per-NPC reasoning-depth decision is exactly this.
3. **Product selling point?** NO (for the paper's mechanism). The codebase's selling point ("10,000 NPCs each at individually-optimal reasoning depth from a single frozen artifact", R149) is *stronger* than the paper's ("one agent with a trained gate").
4. **Force multiplier?** Partial — the deployment-protocol fusion (§2.4) connects plasma-tier reflex + warm-tier planner + gain/cost halting, which is a real combination. But it is perf engineering, not a new capability class.

→ 1 NO, 2 NO, 3 NO, 4 partial. Not Super-GOAT.

### MOAT gate per domain (§1.6)

- `katgpt-rs` MOAT (paper-derived fundamental primitive passing GOAT via fusion): the paper's *insight* is fundamental, but it already ships — no new primitive to add. The deployment-protocol fusion is perf engineering, not a research-grade primitive.
- `riir-ai` MOAT (pillar-level / Super-GOAT connecting ≥2 pillars): R149 *is* the pillar-level selling point; the paper adds nothing to it.

→ Neutral for both repos. Pass is the correct tier.

---

## 4. What ships where (5-repo discipline)

| Repo | What | Why |
|---|---|---|
| `katgpt-rs` | Nothing new | All modelless primitives the paper's insight reduces to already ship (P231, P263, P304-scoped) |
| `riir-ai` | Two `.issues/` follow-ups (§2.4): asynchronous plasma-tier cognition; variable-duration CGSP with `γ^k` | Perf-engineering extensions to shipped pillars, not new research |
| `riir-chain` | Nothing | No sync-boundary angle |
| `riir-neuron-db` | Nothing | No shard/freeze angle beyond what R149 already covers |
| `riir-train` | One-line note: "PPO + variable-duration GAE over budgeted options (Muppidi et al. 2026, arXiv:2606.26463) — a training-method reference for state-dependent compute-budget gates; runtime analog covered modellessly in katgpt-rs/R149+P304." | The paper's actual contribution is here |

---

## 5. Open questions / risks

- **Q: Does the modelless gain/cost halt actually beat a fixed budget on the paper's five environments?** Unknown — would require a `riir-poc/` PoC (§2.5). Not blocking the Pass verdict (the verdict rests on architectural coverage + training-only redirect, not quality parity).
- **Q: Is the two-GPU asynchronous protocol worth adopting for riir-ai's hot/plasma tier?** Possibly — the paper's "reflex holds the agent while planner computes asynchronously" pattern maps cleanly to "HLA evolve_hla holds the NPC while latent_functor computes on a warm worker". Tracked as **Issue 364**.
- **Q: Does the `γ^k` GAE correction improve CGSP value estimates?** **Honesty correction:** CGSP has no γ-discounted value estimates — it's bandit/priority-based. The modelless analog (staleness weight on the priority delta) is tracked as **Issue 365**; the literal γ^k correction is a deferred hook, not a one-line change.

---

## TL;DR

**Verdict: Pass.** The paper introduces variable-delay real-time RL with a PPO-trained gating policy that selects state-dependent MCTS budgets. The training loop is → riir-train (one-line note). The modelless insight — state-dependent compute budget with endogenous deliberation cost — already ships in six independent implementations (R149 gain/cost halting, R218 breakeven complexity, R350 density-aware scheduling, P231 PathwayTracker, P263 cumprodsum freshness, P304 `GainCostLoopHalter`), and the latent reframing on HLA state is strictly stronger (crowd-scale, fog-of-war compatible, no training). The only genuine novelty — the two-GPU asynchronous committed-action deployment protocol — is a systems concern tracked as two `.issues/` follow-ups (asynchronous plasma-tier cognition; variable-duration CGSP with `γ^k` discount), not a research primitive. No files created in katgpt-rs / riir-ai / riir-chain / riir-neuron-db beyond this note.
