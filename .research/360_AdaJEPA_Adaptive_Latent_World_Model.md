# Research 360: AdaJEPA — An Adaptive Latent World Model

> **Source:** [AdaJEPA: An Adaptive Latent World Model](https://arxiv.org/abs/2606.32026) — Ying Wang, Oumayma Bounou, Yann LeCun, Mengye Ren (NYU + AMI Labs), arXiv:2606.32026v1, 30 Jun 2026
> **Date:** 2026-07-01
> **Status:** Done — verdict locked (**PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db**)
> **Classification:** Public (this note). Training recipe → riir-train.
> **Related Research:** 358 (SMWM — **identical verdict, same author Balestriero, same JEPA domain, runtime analog already ships**), 138 (LeJEPA — same author Balestriero, LOW-MODERATE GAIN precedent), 123 (Latent Functor Runtime — **ships the runtime version of AdaJEPA's core insight as Super-GOAT**), 275 (Induced CWM — the `g_φ` forward-model + `BeliefInferenceFn` encoder), 318/riir-ai 163 (Sleep-Time — offline consolidation, the "frozen world model rolls forward" half), 359/riir-ai 168 (Motor-Gated DEC World Model — closest selling-point cousin, written the same day), 243 (Temporal Deriv Kernel — prediction-error curiosity, DEFAULT-ON)
> **Domain:** katgpt-rs (this note, public). The distilled RUNTIME primitive already ships — no new public or private file.

---

## TL;DR

AdaJEPA adapts a JEPA latent world model **inside closed-loop MPC**: at every replan step, plan with the current model → execute the first action → observe the next transition `(o_t, a_t, o_{t+1})` → perform **one gradient step** on a small subset of encoder/predictor parameters using the self-supervised latent prediction loss → replan with the updated model. A small online buffer (`recent-N` or `hard-N`) keeps the latest transitions. Across shape/visual/dynamics/layout distribution shifts, AdaJEPA recovers substantial planning success with as few as one GD step per MPC replan (e.g., ~2× on unseen PushObj shapes, +25% on unseen PointMaze layouts). Stop-gradient on the target branch is the default anti-collapse stabilizer; LoRA gives a similar boost.

**Verdict: PASS for modelless/runtime (katgpt-rs / riir-ai / riir-chain / riir-neuron-db). Training recipe → riir-train (one-line note).**

The paper's distilled primitive — "the action-conditional latent predictor is recalibrated online from its own rollout consequences, then atomically reused for the next planning cycle" — **is already shipped modellessly** as `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs::ReestimationScheduler` (Super-GOAT Research 123, Plans 303/317): `ObservationBuffer` = the online `recent-N` buffer; `tau_reest` = the recalibration trigger; `tick()` = the recalibration step (`extract_functor` re-estimates the action→latent-displacement map from the buffer, no gradient); atomic swap = the next planning cycle reuses the updated functors. This is plan-execute-adapt-replan, modellessly, with no backprop. The training-only parts (GD on JEPA encoder+predictor at every MPC step) belong in `riir-train` per the precedent set by Research 358 (SMWM, same author Balestriero, identical PASS verdict). **Honest downgrade — the runtime analog was the DiPOD canonical vocabulary-mismatch failure case documented in the research skill itself; the structure is identical under different vocabulary.**

---

## 1. Paper Core Findings

### 1.1 The closed-loop plan-execute-adapt-replan recipe (Algorithm 1)

After pretraining a JEPA world model `(ℰ_s, ℰ_a, f_θ)` on reward-free offline data, AdaJEPA runs the following loop at test time:

1. **Plan** — MPC optimizes an action chunk minimizing `Σ α_k ‖ẑ_{t+k} − z_g‖²` by rolling out `f_θ`.
2. **Execute** — run the first action `a_t`, observe `o_{t+1}`.
3. **Adapt** — append `(o_t, a_t, o_{t+1})` to online buffer ℬ; perform `U` gradient steps on `Ω ⊆ {φ, ψ, θ}` minimizing `L_ada = (1/|ℬ|) Σ ℓ(f_θ(z_i, ℰ_a(a_i)), sg(z_{i+1}))`.
4. **Replan** with the updated model.

The adapted model is *immediately reused* for the next planning problem. Each episode starts from the same pretrained model and maintains its own copy + buffer.

### 1.2 What to adapt + how

- **Parameters Ω**: by default, the predictor's last transformer block + final LayerNorm, plus the encoder's last stage (the projection head). Adapting earlier predictor blocks (`predfirst+enclast`) helps on layout shifts; LoRA (rank 8) gives a similar boost but does not consistently beat direct selected-layer updates.
- **Hyperparameters**: one GD step per MPC replan step at the training learning rate (`η_pred = 5e-4`, `η_enc = 1e-5`) is a strong practical default. Larger LRs + multiple steps help in low-stability regimes but overshoot; smaller LRs need more steps and more latency.
- **Online buffer ℬ**: cap to size N. Two strategies — `recent-N` (most recent transitions, focuses on local dynamics) and `hard-N` (the N transitions with the largest prediction errors). Both substantially outperform frozen; `recent-N` is the more stable default. **Buffer design has a smaller effect than expected — even no-buffer beats frozen.**
- **Anti-collapse**: stop-gradient on the target representation `z_{i+1}` is the default. Restricted last-layer updates already limit collapse; removing stop-gradient while updating only last layers gives similar performance.

### 1.3 Empirical results

- **In-distribution**: safe to apply. Yields large gains when the frozen model is suboptimal (PushObj seen shapes: +20%), does no harm when already near-optimal (default PushT/DINO-WM validation).
- **Shape shifts** (PushObj, unseen shapes I/smallT/square): AdaJEPA nearly doubles planning success.
- **Visual shifts** (blur, salt-and-pepper, dark, red-agent/block/anchor): clear gains except for color shifts that break the anchor-vs-block distinction.
- **Dynamics shifts** (PointMaze low-mass, high-damping): the frozen baseline is already strong (in-context learning over 3-frame history); AdaJEPA adds consistent small gains.
- **Layout shifts** (unseen PointMaze layouts): `predfirst+enclast` gives +21.3% (CEM) / +25.3% (GD).
- **Latency**: negligible overhead — +0.01–0.03 s per MPC replan step on an H200 (one GD step on a small parameter subset).
- **Data-scale ablation**: AdaJEPA is most valuable in low-data regimes. A 1k-trajectory single-shape model with adaptation (60.8% seen success) beats a 16k-trajectory-per-shape frozen model (43.5%). Diversity (K shapes) matters more than per-shape trajectory count (N).

### 1.4 The structural insight (§4.4, the transferable nugget)

The paper frames the result as: *"latent world models should continue to be trained at deployment, rather than kept frozen."* The mechanism that makes this work is **(a) the online transition is a free self-supervised signal** (the model caused the action, so `(o_t, a_t, o_{t+1})` is always available), **(b) a tiny last-layer update is enough** because most of the pretrained representation generalizes, and **(c) the closed loop keeps the model pinned to the local dynamics** currently being encountered. The `recent-N` buffer's superiority over `hard-N` in many regimes shows that **locality** (recalibrate to the dynamics at hand) often beats **difficulty** (recalibrate to the hardest transitions).

---

## 2. Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Where it ships |
|---|---|---|
| latent state `z_t` | HLA per-NPC 8-dim state, belief state, sense projection | `riir-engine/src/hla/`, `katgpt-core` HLA kernels |
| JEPA encoder `ℰ_s` + predictor `f_θ` | `InducedCwmKernel: GameState` (frozen forward model) + `BeliefInferenceFn` (observation→belief sampler) | `katgpt-core/src/induced_cwm/`, Research 275 / Plan 296 |
| action encoder `ℰ_a` | `extract_functor` (estimate displacement from (source, target) pairs) | `riir-engine/src/latent_functor/arithmetic.rs` |
| plan-execute-adapt-replan (Algorithm 1) | CGSP/MCTS plan → NPC acts → `ReestimationScheduler::observe()` → `tick()` re-estimates functors → next planning cycle reuses | `riir-engine/src/latent_functor/reestimation.rs`, `cgsp_runtime/` |
| online buffer ℬ (capacity N) | `ObservationBuffer` (capacity-capped ring buffer) | `reestimation.rs::ObservationBuffer` |
| `recent-N` vs `hard-N` strategy | FIFO ring buffer (`head`/`len` wraps oldest) — the `recent-N` default; `JsUniquenessTrigger` + TEMP `sleep_diverse` cover the `hard-N` spirit (rare/diverse preservation) | `reestimation.rs`, `riir-neuron-db/.plans/005` |
| adaptation loss `L_ada` | `extract_functor` MSE over `(source, target)` pairs = `(1/N) Σ_k (target_k − source_k)` mean displacement | `latent_functor/arithmetic.rs` |
| stop-gradient anti-collapse | freeze/thaw atomicity (readers keep old snapshot; writers commit new); coherence gate `functor_gate(coherence, β, τ) = sigmoid(β·(c−τ))` | `MerkleFrozenEnvelope`, `latent_functor/arithmetic.rs` |
| adapted parameters Ω (last layer) | functor direction vectors (the only mutable state); base operator frozen | `LatentSteeringVector`, `apply_functor` |
| `coherence < threshold` trigger (implicit: prediction error grows → adapt) | `coherence < tau_reest` re-estimation trigger | `reestimation.rs::ReestimationScheduler::tick` |
| MPC replan | MCTS rollout / CGSP cycle / decision stage | `cgsp_runtime/`, `mcts_collapse_bridge.rs` |
| distribution shift → frozen model fails | coherence decay → staleness → re-estimation trigger | `reestimation.rs`, DiPOD-class pattern |
| one gradient step per replan | one `tick()` per decision stage (modelless, no gradient) | `reestimation.rs::tick` |
| single GD step on last layers | `extract_functor` re-derives the displacement from the buffer (closed-form, no gradient) | `latent_functor/arithmetic.rs` |

---

## 3. Distillation — duplicate detection vs our corpus

### 3.1 Plan-execute-adapt-replan at runtime — **already shipped as Super-GOAT**

The paper's headline loop is **exactly** the DiPOD-class pattern that the research skill flags as the canonical vocabulary-mismatch failure case:

> *"DiPOD's 'self-distillation when ELBO drifts' is shipped as `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` 'coherence-driven re-estimation scheduler when coherence < tau_reest'."* — SKILL §1.5

The runtime structure is:

```
plan(MCTS/CGSP, current functors) → NPC executes first action →
  ReestimationScheduler::observe(source, target) →
    buffer.push(...) (capacity-capped, wraps oldest = recent-N) →
      tick() →
        if coherence < tau_reest: extract_functor over buffer →
          atomically swap updated functors into the table →
            next planning cycle uses updated functors
```

This is plan-execute-adapt-replan, modellessly, with **zero gradient steps**. The "adaptation loss" is the closed-form `extract_functor` mean-displacement estimate; the "stop-gradient" is replaced by atomic freeze/thaw (readers keep the old functor snapshot, writers commit the new one); the "last-layer update" is the functor direction vector (the only mutable state, the operator base is frozen); the "online buffer" is `ObservationBuffer` (capacity-capped ring buffer, exactly `recent-N`). The `JsUniquenessTrigger` adds a rare-but-diverse preservation path (the spirit of `hard-N`).

**Shipped as Super-GOAT**: Research 123 (Latent Functor Runtime Guide) + Plans 303 / 317 / 357.

### 3.2 The frozen JEPA world model — **already shipped**

AdaJEPA's pretrained `(ℰ_s, ℰ_a, f_θ)` maps onto the already-shipped **InducedCwmKernel + BeliefInferenceFn** (Research 275 / Plan 296): `InducedCwmKernel: GameState` is a verifiable, BLAKE3-committable, atomic-hot-swappable forward model; `BeliefInferenceFn<S>` is the observation-history → belief-distribution sampler. The "frozen pretrained world model reused for planning" pitch is the CWM pitch verbatim.

### 3.3 Offline rehearsal through the frozen model — **already shipped**

AdaJEPA's per-episode adaptation (each episode starts from the pretrained model, maintains its own adapted copy) is the *online* analog of what `sleep_time` (Plan 341, **DEFAULT-ON 2026-06-27**) does *offline*: precompute `z_i = c + dir_i` during idle ticks, then `consume(gate, precomputed, fresh)` blends at wake time. The riir-ai guide 168 (Motor-Gated DEC World Model, written the same day as this note) extends this to *action-trajectory rehearsal through a frozen spatial field* — strictly broader than AdaJEPA's per-episode encoder/predictor adaptation.

### 3.4 Prediction-error-driven recalibration — **already DEFAULT-ON**

AdaJEPA's signal is the latent prediction error `ℓ(ẑ_{t+1}, z_{t+1})`. The Temporal Derivative Kernel (Research 243, Plan 277, **DEFAULT-ON**) ships curiosity = prediction-error signal. `latent_functor/reestimation.rs::tick` triggers re-estimation on coherence decay, which is the same signal under a different name (coherence is `1 − normalized prediction error`).

### 3.5 Closest cousins across all 5 repos

| Cousin | Domain | Verdict / status | Overlap with AdaJEPA |
|---|---|---|---|
| **Research 123 (Latent Functor Runtime)** | riir-ai | **Super-GOAT, shipped (Plans 303/317/357)** | Ships plan-execute-adapt-replan modellessly via `ReestimationScheduler` — AdaJEPA's loop without gradients |
| **Research 358 (SMWM, same author Balestriero)** | katgpt-rs | **PASS** (identical verdict, today) | Same JEPA world-model domain, same "runtime analog already ships" conclusion — sets the precedent |
| **Research 138 (LeJEPA, same author Balestriero)** | katgpt-rs | **LOW-MODERATE GAIN** | Same JEPA domain — downgrade precedent |
| **Research 275 (Induced CWM)** | katgpt-rs | Shipped | `InducedCwmKernel` + `BeliefInferenceFn` — the frozen `g_φ` + encoder target |
| **Research 318 / riir-ai 163 (Sleep-Time)** | katgpt-rs / riir-ai | **DEFAULT-ON 2026-06-27** | Offline frozen-model rehearsal — the "frozen world model rolls forward" half |
| **riir-ai 168 (Motor-Gated DEC World Model)** | riir-ai | Super-GOAT guide (today) | Closest selling-point cousin — frozen world model + per-NPC belief evolution + offline rehearsal, strictly broader |
| **Plan 277 (Temporal Deriv Kernel)** | katgpt-rs | **DEFAULT-ON** | Curiosity = prediction-error signal (the AdaJEPA adaptation trigger) |
| **riir-neuron-db Plan 005 (TEMP `sleep_diverse`)** | riir-neuron-db | **DEFAULT-ON 2026-06-29** | Lipschitz-bound diversity pre-filter on wake events — the `hard-N` spirit |

---

## 4. Mandatory latent-space reframing (per SKILL §1.5 step 3)

| Target substrate | AdaJEPA reframing | Status |
|---|---|---|
| **(a) HLA per-NPC latent state** | "Recalibrate the HLA projection direction vectors from recent transitions" — the committed-personality + civ_emotion path, already the latent-functor pitch | Already shipped |
| **(b) `latent_functor/` (the JEPA predictor + adapter)** | "Adaptation = `extract_functor` re-derivation when coherence < `tau_reest`; predictor = `apply_functor`" — verbatim, modelless | Already shipped as Super-GOAT (Research 123, Plans 303/317/357) |
| **(c) `cgsp_runtime/` (the MPC replan loop)** | "Each CGSP cycle = one plan-execute-adapt-replan iteration; `ReestimationScheduler::observe` feeds the online buffer; `tick` is the adapt step" | Already shipped |
| **(d) LatCal fixed-point commitment (sync boundary)** | AdaJEPA's adaptation is per-episode-local (not synced); only the resulting action crosses the boundary — same discipline as HLA's 5 synced scalars | Boundary discipline inherited, no new bridge needed |
| **(e) `NeuronShard` / `MerkleFrozenEnvelope` / Raven consolidation** | The pretrained model is a frozen, BLAKE3-committed `InducedCwmKernel`; per-episode adapted functors are local latent state (never synced); sleep-time consolidation is the cross-episode integration | Already shipped (Plan 296, Plan 341) |
| **(f) DEC Stokes operators** | No reframing — AdaJEPA is point-state-conditional, not field/divergence/curl-centric | N/A |

Every substrate either already ships the equivalent or is orthogonal. **No new latent-to-latent operation is suggested by AdaJEPA that the codebase does not already have.**

---

## 5. §3.5 Modelless-unblock check

The paper IS training-only (gradient descent on JEPA encoder + predictor at every MPC step). Per §3.5 the question is whether the distilled primitive can be implemented modellessly. **It already is** — no unblock needed:

1. **Freeze/thaw path** — N/A as a *gate failure*. The primitive IS the runtime pattern: `ReestimationScheduler` atomically swaps updated functors when coherence decays; readers keep the old snapshot until the swap completes (the modelless analog of AdaJEPA's "adapted model is immediately reused for the next planning problem").
2. **Raw/lora reader-writer hot-swap** — N/A. `apply_functor` (deterministic displacement addition) is *more* modelless than a constructed LoRA overlay. AdaJEPA's own ablation shows LoRA gives a similar boost but does not consistently beat direct selected-layer updates — and our selected-"layer" update is `extract_functor`, which is closed-form (no GD at all).
3. **Latent-space correction** — N/A. The prediction error `e = z_observed − z_predicted` is folded into the next functor via `extract_functor` over the buffer (a leaky-integrator displacement estimate), gated by `functor_gate(coherence, β, τ) = sigmoid(β·(c−τ))`. This is the modelless analog of AdaJEPA's `Ω ← Ω − η∇_Ω L_ada`.

No deferral to riir-train is needed from the modelless side because **there is nothing to unblock** — the runtime primitive already ships and is strictly more modelless than the paper's mechanism. The training recipe itself (JEPA pretraining + per-step GD adaptation) belongs in riir-train per the Research 358 (SMWM) precedent, as a refinement of the existing RLVR/GRPO + auxiliary-loss objectives — not a new research line.

---

## 6. Novelty gate (§1.5) — all four NO

| Q | Answer | Evidence |
|---|---|---|
| **1. No prior art?** | NO | Research 123 (ships runtime plan-execute-adapt-replan as Super-GOAT via `ReestimationScheduler`); Research 358 (identical verdict, same author Balestriero); Research 275 (frozen world model); Plan 341 (offline frozen-model rehearsal, DEFAULT-ON); Plan 277 (prediction-error curiosity, DEFAULT-ON); riir-neuron-db Plan 005 (`hard-N`-spirit diversity filter, DEFAULT-ON) |
| **2. New capability class?** | NO | "Recalibrate the action-conditional predictor online from rollout consequences" already ships modellessly |
| **3. Product selling point?** | NO | "NPCs recalibrate their world model from their own rollout consequences" is the latent-functor re-estimation + sleep-time consolidation pitch (Research 123, Plans 303/317/341) |
| **4. Force multiplier (≥2 pillars)?** | WEAK | Touches functor + sleep_time + InducedCwm + HLA, but all already integrated |

**Verdict: PASS for modelless/runtime.** Not Super-GOAT, not GOAT, not Gain.

---

## 7. MOAT gate per domain

| Repo | In-scope? | MOAT contribution | Decision |
|---|---|---|---|
| `katgpt-rs` (public) | Marginal | None — primitive already shipped as `latent_functor` (private) + sleep_time (public). No new open primitive to add. | **No file created** (this note is the only output) |
| `riir-ai` (private runtime) | In-scope | None — Research 123 (Super-GOAT) + Research 168 (Motor-Gated DEC, today) already cover the runtime IP, strictly more broadly | **No guide created** |
| `riir-chain` (private chain) | Out of scope | N/A — AdaJEPA's adaptation is per-episode-local latent state, never crosses sync | — |
| `riir-neuron-db` (private shards) | Out of scope | N/A — the frozen `InducedCwmKernel` already commits via BLAKE3; `MerkleFrozenEnvelope` already wraps frozen trajectories | — |
| `riir-train` (private training) | In-scope | Gain — per-MPC-step GD adaptation with `recent-N`/`hard-N` buffers is a worth-trying refinement of existing JEPA pretraining + RLVR objectives; one-line note | **→ riir-train** (see §8) |

---

## 8. → riir-train (one-line redirect per SKILL §"Redirect to riir-train")

If prioritized, file a plan in `riir-train/.plans/` extending Research 358 (SMWM) and the existing JEPA-pretraining recipe: add **per-MPC-step last-block GD adaptation with stop-gradient** as an online refinement loop, with `recent-N` and `hard-N` buffer strategies as A/B variants, tested on the Bomber/Go/Civ arenas against the frozen-pretrained baseline. Hypothesis (per AdaJEPA §4.4): the gain is largest in low-training-data regimes and under dynamics/layout distribution shift; the gain is smallest when the pretrained model is already near-optimal in-distribution. **Not pursued here — out of scope for this workflow.**

The only genuinely transferable *runtime* nugget — AdaJEPA's `hard-N` buffer (keep the N transitions with the largest prediction errors, vs `recent-N`) — is already covered in spirit by `JsUniquenessTrigger` (rare-signal preservation in `reestimation.rs`) and TEMP `sleep_diverse` (Lipschitz-bound diversity pre-filter, riir-neuron-db Plan 005, **DEFAULT-ON 2026-06-29**). If a future `ObservationBuffer` refactor ever prioritizes prediction-error-magnitude curation over FIFO, it is a one-line option, not a feature flag.

---

## TL;DR

**Paper:** *AdaJEPA: An Adaptive Latent World Model* (Wang, Bounou, LeCun, Ren, arXiv:2606.32026, 2026-06-30).

**Verdict:** **PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db.** The paper's distilled runtime primitive — "the action-conditional latent predictor is recalibrated online from its own rollout consequences, then atomically reused for the next planning cycle" — **is already shipped modellessly** as `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs::ReestimationScheduler` (Super-GOAT Research 123, Plans 303/317/357): `ObservationBuffer` = the online `recent-N` buffer; `tau_reest` = the recalibration trigger; `tick()` = the recalibration step (`extract_functor` re-estimates the action→latent-displacement map from the buffer, no gradient); atomic swap = the next planning cycle reuses the updated functors. The training recipe (per-MPC-step GD on JEPA encoder+predictor) is a refinement of the same mechanism Research 358 (SMWM, same author Balestriero) already evaluated as training-only; it belongs in riir-train as a one-line refinement, not in the modelless/runtime repos. This is the DiPOD canonical vocabulary-mismatch failure case documented in the research skill — the structure is identical under different vocabulary.

**Files created this session:** `katgpt-rs/.research/360_AdaJEPA_Adaptive_Latent_World_Model.md` (this note — the only output).

**Recommended next step:** None for katgpt-rs / riir-ai / riir-chain / riir-neuron-db. The riir-train follow-up is optional and out of scope for this workflow.

---

## 9. PoC Addendum — empirical check of the "parity" claim (2026-07-01)

A "defend-wrong" PoC was added at `riir-ai/crates/riir-poc/benches/adajepa_modelless_goat.rs`
to test whether the shipped modelless pattern (closed-form refit + coherence-
triggered re-estimation) actually matches an AdaJEPA-style per-step adaptation
loop on a planning-under-shift task. Toy domain: 2D point-mass navigation,
`z_{t+1} = z_t + mass·a`, model prior `R=I` (mass=1), MPC sampling planner
(horizon=4, n_samples=64), 200 episodes per shift, goal_radius=0.3, max_steps=25.

**Results (raw, from the bench run):**

| Shift | Frozen | PerStepGd (AdaJEPA-analog) | CoherenceTriggeredRefit (shipped) |
|---|---|---|---|
| in-dist (mass=1.0)  | 68.5% | 68.5% | 68.5% |
| mild (mass=0.7)     | 73.5% | **87.0% ↑** | 73.5% (tie, **0 updates**) |
| moderate (mass=1.5) | 61.0% | 42.0% ↓ | 57.5% ↓ |
| severe (mass=0.4)   | 55.0% | **87.5% ↑** | **87.5% ↑** |
| severe (mass=2.5)   | 40.5% | 25.5% ↓ | 22.5% ↓ |

Latency per replan (all three ~940 ns, planner-dominated): frozen 937 ns,
per_step_gd 945 ns, coherence_triggered 967 ns. **Latency parity confirmed** —
sub-µs adaptation overhead, no autograd, the planner's 64-sample rollout is
the bottleneck.

**Honest revision of the verdict's "parity" claim:**

- ✅ **Latency parity confirmed.** All three strategies ~940 ns/replan;
  adaptation overhead is +8–30 ns. Sub-µs, modelless, no GD.
- ✅ **Capability coverage confirmed.** Both adaptive strategies recover
  success on severe *undershoot* shifts (mass<1.0): 55%→87.5% at mass=0.4.
- ❌ **Quality parity REFUTED on two axes:**
  1. **Coherence trigger too conservative for mild shifts.** At mass=0.7,
     `mean_updates=0` — the coherence gate never fires because prediction
     error stays below the threshold. PerStepGd (always updates) catches
     mild shifts the coherence gate misses (73.5%→87.0%). The shipped
     pattern needs a more sensitive threshold or a small-step always-on
     background update for the mild-shift regime.
  2. **All adaptation strategies HURT on overshoot shifts (mass>1.0).**
     At mass=1.5 and 2.5, both adaptive strategies score *below* frozen.
     Root cause: rank-1 GD / 2×2 LS refit diverges when actions are
     correlated (the planner keeps picking goal-directed actions, so the
     normal-equations matrix `AᵀA` is ill-conditioned). AdaJEPA's stop-
    gradient + restricted-layer discipline would also struggle here on a
     2×2 system, but the paper's neural-net predictor has more capacity
     to absorb the correction stably.

**Net:** The Research 360 verdict stands on the *architectural* claim (the
runtime analog of plan-execute-adapt-replan ships, modellessly, at parity
latency) but is **partially wrong** on the *quality* claim. The shipped pattern
is a base layer that needs: (a) a more sensitive coherence threshold (or
background-update path) for mild shifts, and (b) a stabilization mechanism
(action decorrelation, larger buffer, or damping) for overshoot shifts.

These are tracked as follow-ups in `riir-ai/.issues/` (Issue: AdaJEPA PoC —
coherence trigger tuning + overshoot-shift stabilization). The PoC is kept
as a permanent regression check in `riir-poc` ("defend-wrong" crate) — its
*job* was to defend or refute the verdict's parity claim, and it refuted
the quality half honestly.
