# Research 358: SMWM — Sensorimotor World Models via Inverse Dynamics Regularization

> **Source:** [Sensorimotor World Models: Perception for Action via Inverse Dynamics](https://arxiv.org/abs/2606.20104) — Ivashkov, Balestriero, Schölkopf (MPI-IS / Brown / ETH), arXiv:2606.20104v1, 18 Jun 2026
> **Date:** 2026-07-01
> **Status:** Done — verdict locked (**PASS for katgpt-rs / riir-ai**)
> **Classification:** Public (this note). Training recipe → riir-train.
> **Related Research:** 138 (LeJEPA, same author Balestriero — same-domain precedent, LOW-MODERATE GAIN), 127 (EnvRL — **identical mechanism, training-only precedent**), 123 (Latent Functor Runtime — **ships the runtime version of SMWM's core insight as Super-GOAT**), 115 (PEIRA), 298 (Inverting Bellman — sibling closed-form world-model extraction), 318 (Sleep-Time — covers "offline reward-free"), 462 (INTACT — **extends SMWM's `L_inv` with a goal-conditioned second branch + shared operator; identical PASS verdict, same runtime-already-ships conclusion**)
> **Domain:** katgpt-rs (this note, public). The distilled RUNTIME primitive already ships — no new public or private file.
> **PASS-Redirects (synthesis):** Sun, Zhao, Zhang [arXiv:2607.26056 "INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models"] — extends SMWM's `L_inv` auxiliary with a stop-gradient goal-conditioned second call through the same operator (asymmetric gradient routing). Verdict: **PASS** (not Gain). Same canonical JEPA-world-model PASS class as SMWM/AdaJEPA/Temporal Straightening: the deployment primitive (`m = z_g − z_t` → `apply_functor_edge_into`) already ships modellessly; the training recipe (joint encoder + dual-call operator + forward predictor) belongs in riir-train as a multi-loss refinement of the existing JEPA-pretraining recipe. See [Research 462](462_INTACT_Isomorphic_Intent_To_Action_Search_Free_World_Models.md).

---

## TL;DR

SMWM trains a JEPA-style latent world model end-to-end with a single **inverse-dynamics regularizer** `L_inv = ‖h_ψ(z_t, z_{t+1}) − a_t‖²` that (a) prevents representation collapse and (b) induces action-aligned latent structure. Empirically the encoder learns compact representations that track the controllable degrees of freedom, filter uncontrollable distractors, and approximately satisfy **translation-equivariance** `g_a(z) ≈ z + ρ(a)` with action composition `g_{a₂∘a₁} = g_{a₂} ∘ g_{a₁}` (Theorem 1, a semigroup homomorphism into latent transformations). The recipe is reward-free, offline, no frozen encoder, no EMA.

**Verdict: PASS for modelless/runtime (katgpt-rs / riir-ai). Training recipe → riir-train.**

The paper's distilled primitive — "the action between two latent states is recoverable as a latent displacement vector" — **is already shipped at runtime** as `crates/katgpt-percepta/src/wasm/interpreter/arithmetic.rs` (Plan 273, Super-GOAT Research 123): `extract_functor` estimates `f = mean(target − source)`, `apply_functor` predicts `out = source + f`, `functor_gate` is the sigmoid trust gate. This is `z_{t+1} ≈ z_t + ρ(a)` verbatim, modelless, no training. `sleep_time::HlaSleepTimeOp` (Plan 341, **DEFAULT-ON since 2026-06-27**) inlines the same `z_i = c + dir_i` elementwise add. The training-only parts (end-to-end JEPA + inverse-dynamics loss) are a refinement of EnvRL's auxiliary ID head (Research 127, training-only verdict) and belong in `riir-train` per the precedent set there. **Honest downgrade — same-domain precedent (Research 138, same author Balestriero) was LOW-MODERATE GAIN; SMWM's runtime contribution is even smaller because Research 123 already shipped it.**

---

## 1. Paper Core Findings

### 1.1 The single regularizer

Joint objective `L = L_fwd + λ · L_inv` over an offline dataset `D = {(o_t, a_t, o_{t+1})}`:
- `L_fwd = ‖g_φ(z_t, a_t) − z_{t+1}‖²` — forward dynamics in latent space
- `L_inv = ‖h_ψ(z_t, z_{t+1}) − a_t‖²` — inverse dynamics: predict action from two consecutive embeddings

The inverse head propagates gradients into the encoder. A constant (collapsed) encoder cannot reduce `L_inv` below `Var(a)` because `(z_t, z_{t+1})` would carry no information about `a_t`. So inverse prediction is the sole anti-collapse mechanism — no EMA, no frozen encoder, no SIGReg/VICReg distributional prior.

### 1.2 The distilled structural insight (§4) — translation-equivariance

Empirically the learned forward model approximately satisfies `g_a(z) ≈ z + ρ(a)` where `ρ(a)` is approximately independent of `z`. **Theorem 1**: if the encoder is equivariant (`f(a(o)) = g_a(f(o))`) and actions form a semigroup, then `a ↦ g_a` is a homomorphism: `g_{a₂∘a₁}(z) = g_{a₂}(g_{a₁}(z))`. The inverse-dynamics loss biases toward the additive regime because the simplest way to make `a_t` recoverable from `(z_t, z_{t+1})` is to encode it in the displacement `z_{t+1} − z_t ≈ ρ(a_t)`.

### 1.3 Action-aligned / controllable-DoF representation

PCA spectra on dot-world configurations show the encoder allocates significant variance to exactly the controllable degrees of freedom and filters out uncontrollable distractors (random-moving dot ignored even though visible). Sprite-world reconstruction shows the model "sees" the same object differently depending on which pose axes are exposed as actions.

### 1.4 Planning results (§5)

CEM + MPC planning in the frozen latent space. On 4 tasks (TwoRoom, Reacher, Push-T, OGBench-Cube) SMWM matches SIGReg on the 2D tasks and beats it on OGBench-Cube (84% vs 59%). Forward-only (`λ=0`) collapses and fails.

---

## 2. Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Where it ships |
|---|---|---|
| latent state / embedding `z_t` | HLA per-NPC state, belief state, sense projection | `riir-engine/src/hla/`, `katgpt-core` HLA kernels |
| forward model `g_φ` | `GameState` trait, `InducedCwmKernel` (Plan 296) | `katgpt-core/src/induced_cwm/`, `riir-ai/crates/riir-engine/src/game_state.rs` |
| inverse model `h_ψ(z_t, z_{t+1}) → a_t` | `extract_functor` (estimate displacement from pairs) | `crates/katgpt-percepta/src/wasm/interpreter/arithmetic.rs` |
| `g_a(z) ≈ z + ρ(a)` latent translation | `apply_functor: out = source + f`; `HlaSleepTimeOp: z_i = c + dir_i` | `crates/katgpt-percepta/src/wasm/interpreter/arithmetic.rs`, `riir-ai/crates/riir-engine/src/sleep_time/hla_op.rs` |
| Theorem 1 homomorphism `g_{a₂∘a₁} = g_{a₂}∘g_{a₁}` | functor composition, functor table reuse | `latent_functor/`, Research 123 §1.4 |
| action-aligned representation | committed personality direction vectors, archetype blend | `riir-engine/src/committed_blend/`, `riir-neuron-db/src/archetype_blend_shard.rs` |
| controllable DoF projection | direction-vector projection (sigmoid, per AGENTS.md constraint #2) | HLA emotion extraction (`civ_emotion` Plan 175), zone attention |
| filter uncontrollable distractors | curiosity = prediction-error signal (Pathak-style) | Plan 277 temporal-derivative kernel (**DEFAULT-ON**), CGSP (Plan 274) |
| anti-collapse / no EMA / no frozen encoder | freeze/thaw snapshots replace EMA | `riir-neuron-db/src/freeze.rs::MerkleFrozenEnvelope`, Plan 341 |
| reward-free offline trajectories | sleep-time anticipation, CGSP self-play consolidation | `riir-engine/src/sleep_time/` (DEFAULT-ON), `cgsp_runtime/` |
| compact interpretable latent space | HLA 8-dim per-NPC, SpectralQuant eigenbasis | HLA, Research 039 |

---

## 3. Distillation — duplicate detection vs our corpus

### 3.1 Inverse dynamics as auxiliary loss — **already evaluated, training-only**

**Research 127 (EnvRL, 2026-06-18)** evaluated the *exact* same mechanism — `L_ID(θ) = -log p_θ(a_t | s_t, s_{t+1})` as an auxiliary loss — and reached the verdict: *"Pass for modelless/runtime. ID is a genuinely novel auxiliary (zero grep hits) but it is purely a training-loop credit-assignment refinement — no modelless/runtime distillation worth shipping to katgpt-rs or riir-ai."* SMWM differs only in framing (anti-collapse for JEPA world models vs credit-assignment for GRPO). The mechanism is identical; the verdict applies. SMWM's "novelty" is using ID as the **sole** regularizer rather than one of many — a training-recipe choice, not a runtime primitive.

### 3.2 The distilled runtime primitive — **already shipped as Super-GOAT**

**Research 123 (Latent Functor Runtime Guide)** documents the SMWM-equivalent insight at runtime, **as a Super-GOAT**:

> "Analogy `A:B :: C:D` decomposes into [...] **functor application as residual-stream addition**. `e_target ≈ e_source + f`" — verified in pretrained LLMs along the **layer axis** during inference (no weight updates).

This is `z_{t+1} ≈ z_t + ρ(a)` verbatim. The paper's Theorem 1 homomorphism is functor composition (Research 123 §1.4). The paper's "filter uncontrollable distractors" is Pathak-style curiosity (Plan 277, DEFAULT-ON). The paper's "action-aligned representation" is HLA committed personality direction vectors. The paper's "no EMA / no frozen encoder" is the freeze/thaw snapshot discipline (MerkleFrozenEnvelope, Plan 341). The paper's "reward-free offline" is sleep-time anticipation (DEFAULT-ON) and CGSP self-play.

The latent-functor arithmetic API ships exactly the paper's primitives modellessly:
```rust
// crates/katgpt-percepta/src/wasm/interpreter/arithmetic.rs (Plan 273, Super-GOAT Research 123)
extract_functor(sources, targets, dim) -> (functor_dir, coherence)
//   f = (1/N) Σ_k (target_k − source_k)   ← mean displacement = ρ(a) estimate
apply_functor(source, functor, out)         ← out = source + f   = z + ρ(a)
functor_gate(coherence, beta, tau)          ← sigmoid(β·(c − τ)) trust gate
```

`sleep_time::HlaSleepTimeOp` (Plan 341, **DEFAULT-ON 2026-06-27**) inlines the same `z_i = c + dir_i` elementwise add and was explicitly noted in Issue 005 as replacing the `apply_functor` dispatch — so the SMWM primitive is not only shipped, it's been **promoted to default-on** as the canonical modelless latent-translation op.

### 3.3 Closest cousins across all 5 repos

| Cousin | Domain | Verdict / status | Overlap with SMWM |
|---|---|---|---|
| **Research 127 (EnvRL)** | riir-train | Pass (training-only) | Identical mechanism (inverse dynamics auxiliary) — sets the precedent |
| **Research 123 (Latent Functor)** | riir-ai | **Super-GOAT, shipped (Plan 273/303)** | Ships `e_target ≈ e_source + f` at runtime — SMWM's `g_a(z) ≈ z + ρ(a)` |
| **Research 138 (LeJEPA, same author)** | katgpt-rs | LOW-MODERATE GAIN (theoretical only) | Same author, same JEPA domain — downgrade precedent |
| **Research 298 (Inverting Bellman)** | katgpt-rs | GOAT | Sibling "extract world model from frozen signal" — different path, same target |
| **Plan 277 (Temporal Deriv Kernel)** | katgpt-rs | **DEFAULT-ON** | Ships curiosity = prediction-error signal (Pathak-style distractor filter) |
| **Plan 341 (Sleep-Time)** | riir-ai | **DEFAULT-ON 2026-06-27** | Ships `z_i = c + dir_i` (SMWM's translation op) modellessly |
| **Plan 275 / 296 (Induced CWM)** | katgpt-rs | Shipped | `InducedCwmKernel: GameState` — the `g_φ` forward-model target |

---

## 4. Mandatory latent-space reframing (per SKILL §1 step 3)

| Target substrate | SMWM reframing | Status |
|---|---|---|
| **(a) HLA per-NPC latent state** | "Action-aligned HLA direction vectors" — already the committed-personality pitch (civ_emotion Plan 175) | Already shipped |
| **(b) `latent_functor/` action application** | "Inverse dynamics = `extract_functor`; forward dynamics = `apply_functor`" — verbatim, modelless | Already shipped as Super-GOAT (Research 123, Plan 273) |
| **(c) `cgsp_runtime/` curiosity signals** | "Filter uncontrollable distractors" — already the Pathak-style prediction-error curiosity channel (Plan 277 DEFAULT-ON) | Already shipped |
| **(d) freeze/thaw snapshots ("no EMA" claim)** | SMWM eliminates EMA via inverse-dynamics regularizer; we eliminate EMA via atomic freeze/thaw with BLAKE3 commitment — different solution, same outcome | Already shipped (Plan 341 DEFAULT-ON, MerkleFrozenEnvelope) |
| **(e) `sleep_time/` offline consolidation ("reward-free" claim)** | "Offline reward-free trajectory consolidation" — exactly the sleep-cycle compute pattern | Already shipped (DEFAULT-ON 2026-06-27) |
| **(f) DEC Stokes operators** | No reframing — SMWM is action/displacement-centric, not divergence/curl-centric | N/A |

Every substrate either already ships the equivalent or is orthogonal. No new latent-to-latent operation is suggested by SMWM that the codebase does not already have.

---

## 5. §3.5 Modelless unblock check

The paper IS training-only (end-to-end gradient descent through encoder + forward + inverse heads). Per §3.5 the question is whether the distilled primitive can be implemented modellessly. **It already is** — no unblock needed:

1. **Freeze/thaw path** — N/A (the primitive is not a gate failure; it's a runtime pattern).
2. **Raw/lora reader-writer hot-swap** — N/A (no weight correction needed).
3. **Latent-space correction** — N/A (the latent-displacement operation is already shipped as `apply_functor`).

No deferral to riir-train is needed from the modelless side because there is nothing to unblock. The training recipe itself (JEPA + inverse-dynamics loss) does belong in riir-train per the EnvRL Research 127 precedent, but as a one-line refinement of the existing `L_GRPO + λ_env·L_ECHO + λ_sdpg·L_SDPG` three-loss objective — not a new research line.

---

## 6. Novelty gate (§1.5) — all four NO

| Q | Answer | Evidence |
|---|---|---|
| **1. No prior art?** | NO | Research 127 (identical mechanism, training-only verdict); Research 123 (ships runtime version as Super-GOAT); Plan 277 (DEFAULT-ON distractor filter); Plan 341 (DEFAULT-ON `z_i = c + dir_i`) |
| **2. New capability class?** | NO | Runtime analog (NPCs learn action→latent-displacement without training) already ships |
| **3. Product selling point?** | NO | "Action-aligned latent state" is already the HLA committed-personality + latent-functor pitch |
| **4. Force multiplier (≥2 pillars)?** | WEAK | Touches functor + sleep_time + HLA, but all already integrated |

**Verdict: PASS for modelless/runtime.** Not Super-GOAT, not GOAT, not Gain.

---

## 7. MOAT gate per domain

| Repo | In-scope? | MOAT contribution | Decision |
|---|---|---|---|
| `katgpt-rs` (public) | Marginal | None — primitive already shipped as `latent_functor` (private) + sleep_time (public). No new open primitive to add. | **No file created** (this note is the only output) |
| `riir-ai` (private runtime) | In-scope | None — Research 123 (Super-GOAT) already covers the runtime IP | **No guide created** |
| `riir-chain` (private chain) | Out of scope | N/A | — |
| `riir-neuron-db` (private shards) | Out of scope | N/A | — |
| `riir-train` (private training) | In-scope | Gain — ID-as-sole-regularizer is a worth-trying refinement of EnvRL's auxiliary; one-line note there | **→ riir-train** (see §8) |

---

## 8. → riir-train (one-line redirect per SKILL §"Redirect to riir-train")

If prioritized, file a plan in `riir-train/.plans/` extending Research 127: add `L_inv` as a JEPA-world-model anti-collapse regularizer (or as a fourth term in the Research 102 three-loss objective `L_GRPO + λ_env·L_ECHO + λ_sdpg·L_SDPG + λ_id·L_ID`) with cosine decay, and A/B-test on Bomber/Civ arenas. Hypothesis: ID helps most in noisy-observation / contact-rich tasks (per SMWM Fig 5 OGBench-Cube + EnvRL WebShop ablation). Not pursued here — out of scope for this workflow.

> **INTACT extension (Research 408, 2026-07-31):** [riir-train/.research/408_intact_shared_operator_dual_call_auxiliary.md](../../riir-train/.research/408_intact_shared_operator_dual_call_auxiliary.md) captures INTACT's (arXiv:2607.26056, [R462](462_INTACT_Isomorphic_Intent_To_Action_Search_Free_World_Models.md)) refinement of this `L_inv` line: the local branch *is* `L_inv` reformulated as displacement `m_local = z_{t+1} − z_t`; INTACT adds a second call `L_goal = −log p_η(a_t | z_t, sg(z_g) − z_t, a_{t-1})` through the **same** operator with stop-gradient on `z_g`. The sharing + asymmetric routing are the novel structural commitments; the gauge experiment (§5.2) is the causal proof. Consumer tracked in `riir-ai Issue 572`.

---

## TL;DR

**Paper:** *Sensorimotor World Models: Perception for Action via Inverse Dynamics* (Ivashkov, Balestriero, Schölkopf, arXiv:2606.20104, 2026-06-18).

**Verdict:** **PASS for katgpt-rs / riir-ai.** The paper's distilled runtime primitive — "the action between two latent states is recoverable as a latent displacement vector" (`z_{t+1} ≈ z_t + ρ(a)`) — **is already shipped modellessly** as `crates/katgpt-percepta/src/wasm/interpreter/arithmetic.rs::apply_functor` (Super-GOAT Research 123, Plan 273) and `sleep_time::HlaSleepTimeOp::z_i = c + dir_i` (Plan 341, **DEFAULT-ON since 2026-06-27**). The training recipe (end-to-end JEPA + inverse-dynamics loss) is the same mechanism EnvRL Research 127 already evaluated as training-only; it belongs in riir-train as a one-line refinement, not in the modelless/runtime repos.

**Files created this session:** `katgpt-rs/.research/358_Sensorimotor_World_Model_Inverse_Dynamics_Regularizer.md` (this note — the only output).

**Recommended next step:** None for katgpt-rs / riir-ai / riir-chain / riir-neuron-db. The riir-train follow-up is optional and out of scope for this workflow.
