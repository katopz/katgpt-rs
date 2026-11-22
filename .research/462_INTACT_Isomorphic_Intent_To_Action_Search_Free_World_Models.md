# Research 462: INTACT — Isomorphic Intent-to-Action Learning for Search-Free World Models

> **Source:** [INTACT: Intent-To-Action Learning for Search-Free World Models](https://arxiv.org/abs/2607.26056) — Junhan Sun, Hao Zhao, Guofeng Zhang (Zhejiang U + Tsinghua AIR + RoboParty Lab), arXiv:2607.26056v1, 28 Jul 2026
> **Date:** 2026-07-31
> **Status:** Done — verdict locked (**PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db**)
> **Classification:** Public (this note). Training recipe → riir-train.
> **Related Research:** 358 (SMWM — **identical mechanism, identical PASS verdict**; INTACT cites SMWM [16] and lifts its `L_inv` as the "local physical intent" branch), 426 (Temporal Straightening — **same authors LeCun/Ren/Balestriero lineage, same LeWM benchmark, same PASS pattern**), 360 (AdaJEPA — same authors Wang/Bounou/LeCun/Ren, same JEPA PASS precedent), 138 (LeJEPA — same author Balestriero, LOW-MODERATE GAIN downgrade precedent), 123 (Latent Functor Runtime — **ships the INTACT deployment primitive as Super-GOAT**), 324 (GC-IDM distillation in riir-ai — **GC-IDM is INTACT's direct cited prior art [34]**, same amortized-planning pattern, Gain verdict)
> **Domain:** katgpt-rs (this note, public). The distilled runtime primitives already ship — no new public or private file.
> **PASS-Redirects (synthesis):** Zhang et al. [arXiv:2611.25927 "Fast-JEPA: Advancing the Success–Latency Frontier for Latent World Models"] — third-party INTACT-lineage follow-up: anchored multi-horizon substrate + terminal-only candidate scoring (2.41× vs the INTACT baseline here) + DynaGoal deadline/delay benchmark; PASS modelless (the substrate is trained; the rollout-free-evaluation class is published — 2611.05072/WoTE), training half completes deferred riir-train R408 with the dual-call λ replicated identically — see katgpt-rs Research 687.

---

## TL;DR

INTACT trains an end-to-end JEPA world model jointly with one **shared conditional action operator** `G_η(z, m, a₋₁) → p(a | z, m, a₋₁)` called twice per transition: once with attached **local intent** `m_local = z_{t+1} − z_t` (physical successor), once with stop-gradient **goal intent** `m_goal = sg(z_g) − z_t` (deployable anchor). The two calls share parameters and a 4-slot input grammar `[z; m; z ⊙ m; A(a₋₁)]` but route endpoint gradients asymmetrically — physical successor stays attached (representation shaping), future goal is detached (deployment conditional). At inference, the goal call's conditional mean **directly emits an action chunk** (Direct mode, zero search); optional Guarded A centers a small 128×3 raw-action CEM around the Direct plan as a verifier. The paper formalizes a **conditional action quotient**: at fixed state `z`, two endpoint conditions `y, y'` are equivalent iff they induce the same expert action law `p*_E(a | z, y) = p*_E(a | z, y')`; this quotient is bijective to the realizable action-law image (Proposition 1), and proper-NLL minimization recovers it on supported conditions (Proposition 3). On the 4 official LeWM tasks, one-epoch Direct reaches 85.78/100.00/97.67/97.89 % SR with 2.9–5.5 ms latency (≈300× under CEM 300×30); Guarded A reaches 96.86 % macro with 384 candidates (23.44× fewer than CEM's 9000).

**Verdict: PASS for modelless/runtime (katgpt-rs / riir-ai / riir-chain / riir-neuron-db). Training recipe → riir-train (one-line) + a tracked `riir-ai` follow-up for the Motor-Gated DEC consumer policy head training recipe (Issue 572).**

**Substrate mismatch, not "we already ship this"** — the honest framing. The paper's value is concentrated in the *training-side* mechanism: a **shared conditional operator** jointly trained across two condition families (local physical intent + goal deployment intent) with **asymmetric gradient routing** (attached on `z_{t+1}`, stop-gradient on `z_g`). This is what makes the learned representation action-semantic. Our stack has the *runtime deployment primitive* — `crates/katgpt-core/src/closure/functor_edge.rs::apply_functor_edge_into` (Plan 040, Super-GOAT R123): cosine-gated `out = state + sigmoid(β·(cos−τ)) · direction` — and the *closed-form displacement estimator* `extract_functor` (R123, Plan 303/317) which constructs INTACT's `m_local = z_{t+1} − z_t` from a transition buffer without gradient. `sleep_time::HlaSleepTimeOp` (Plan 341, **DEFAULT-ON**) inlines the same `z_i = c + dir_i` add. **But these are constructed, not jointly learned.** INTACT's gauge experiment (Full-backbone → LeWM-actor stays at 5.08% SR; §5.2, 21,600-episode audit) is the causal proof that the shared-operator training buys something independent construction does not — a coordinate map can restore a learned conditional but cannot create one absent from the actor. We have no analog of this experiment, and no shipped primitive that was *jointly trained* across two condition families. The honest statement is: "we construct at runtime what INTACT jointly learns during training; the two are not synonyms, but the training-side mechanism lands on a future consumer (Motor-Gated DEC policy head, Issue 572), not on the modelless/runtime repos."

INTACT's **stop-gradient on the goal anchor** maps to the freeze/thaw discipline (MerkleFrozenEnvelope) at the *deployment* level (the frozen goal latent serves as a fixed anchor); the *training-time* asymmetric routing has no runtime analog and is correctly → riir-train. INTACT's **conditional action quotient** (Definition 1) is *related to* but not identical to the `functor_gate` coherence check — coherence says "displacement is consistent across the buffer"; the quotient says "different endpoints map to the same action law". The quotient is the richer equivalence; we capture a useful operational subset via coherence, not the full relation. The training-only pieces (joint encoder + INTACT predictor + forward + asymmetric gradient routing + shared graph-isomorphic dual call) refine SMWM's `L_inv` auxiliary (R358, training-only) with a second goal-conditioned branch — tracked in `riir-ai/.issues/572` as the concrete training recipe for the Motor-Gated DEC consumer policy head (extends Issue 565's runtime-side scope).

---

## 1. Paper Core Findings

### 1.1 The forward-action gap (§1, §3.1)

A forward latent world model learns `F_ϕ(z, a) → ẑ+` but recovers actions only via expensive test-time search (CEM/MPPI). The asymmetry: actions shape the latent during training, yet deployment search begins from generic Gaussian proposals. The paper's core observation: every action-labeled transition reveals (state, motion-intent, action) triples; across many trajectories these reveal *families* of state-conditional intents + their corresponding action families. Train one operator to interpret both the realized physical intent and a deployable goal intent.

### 1.2 The conditional action quotient (Definition 1, Propositions 1 + 3)

At fixed state `z`, define `y ~_z y' ⇐⇒ p*_E(a | z, y) = p*_E(a | z, y')`. The quotient `Y/~_z` is bijective (Prop 1) to the realizable action-law image. Proper-NLL minimization recovers the quotient on supported conditions (Prop 3): for any endpoint `y` with positive probability, `E_{a∼p*_E}[−log p_η(a | z, y)] = H[p*_E] + D_KL(p*_E ‖ p_η)`, minimized iff `p_η = p*_E`. **Equivalence is defined at fixed state by equality of the conditional action law**, not by Euclidean proximity of endpoints. The paper emphasizes: this is *not* metric learning on observations; it is an output-distribution geometry.

### 1.3 The dual-call architecture (§3.2, §3.3, Algorithm 1)

```
G_η : (z, m, a₋₁) → p_η(a | z, m, a₋₁)        # shared conditional operator
m_local_t = z_{t+1} − z_t                          # attached (physical successor)
m_goal_t  = sg(z_g) − z_t                          # stop-gradient (deployment anchor)

# 4-slot input grammar (Eq 16):
h_Δ_t(m_t) = [ z_t;  m_t;  z_t ⊙ m_t;  A_ω(a_{t-1}) ]
# w^⊤(z_t ⊙ m_t) = z_t^⊤ diag(w) m_t  — cheap state-intent bilinear interaction

L_ST = L_world + λ_inv · L_z+ + λ_goal · L_goal    # Eq 15
```

The state-intent bilinear feature `z_t ⊙ m_t` is **not a learned second-order dynamics model** — it's a fixed input feature. The controlled A–G grammar study (Table A2) shows the matched pair `[m_t, z_t ⊙ m_t]` adds +3.89 points over the ungrammared variants; the same displacement gets interpreted differently across contact, pose, and obstacle states.

### 1.4 Deployment — Direct + optional Guarded A (§3.3, Algorithm 2)

Direct mode: recurrently compute `m_k = z_g − ẑ_k`, `ā_k = μ_η(ẑ_k, m_k, ā_{k-1})`, `ẑ_{k+1} = F_ϕ(ẑ_k, ā_k)`; execute the first block; replan from the next real observation. **Zero candidates, zero terminal-cost calls, 2.9–5.5 ms.**

Guarded A: center a 128×3 raw-action CEM at the Direct plan with `σ₀ = 0.25`; preserve the Direct reference + global best across refits. Reaches 96.86 % macro SR with **384 candidates** (vs CEM 300×30's 9000) — 23.44× fewer samples, +16.00 macro points over pure CEM.

### 1.5 Empirical results (§5, Tables 2–3, Figure 4)

- **Single-task (one epoch, Direct, no search):** 85.78 / 100.00 / 97.67 / 97.89 % SR on PushT / Cube / Reacher / TwoRoom (3 training seeds × 3 eval seeds × 100 episodes).
- **Multi-task (one shared ViT-Tiny/14 encoder, E5 Direct):** 89.39 ± 0.77 % macro SR, +23.22-point macro gain over matched shared-encoder LeWM with CEM 300×30.
- **kNN / CKA predict SR:** across 45 E1–E5 checkpoints, predicted–expert action-family kNN correlates with Direct SR at r = 0.954 (CI [.928, .969]); CKA at r = 0.897; pointwise action R² at r = 0.815. **Family-level preservation > pointwise recovery.**
- **Gauge intervention (21,600 episodes):** correct pair calibration recovers 68.04 % CLEAR Moderate SR vs 9.46 % shuffled (+58.58 pp). Reverse swaps (Full-backbone → LeWM-actor) stay at 5.08 % — a coordinate map cannot create an untrained deployment conditional.
- **Goal displacement > waypoint coordinate** under all inference interfaces (Direct, pure CEM, actor-on CEM, Guarded A). Waypoint wins at E1 (optimization speed); goal displacement wins at E5 (converged coordinate).

### 1.6 What is NOT the contribution

- The JEPA architecture (LeCun 2022), the LeWM encoder/predictor, the SIGReg regularizer — all from prior work [4, 9].
- Inverse dynamics as auxiliary representation-shaping loss — SMWM [16] ships this exact mechanism (already evaluated in Research 358, identical PASS verdict).
- Goal-conditioned imitation (GCSL [18]) — INTACT's goal branch is GCSL-like hindsight action likelihood; the paper explicitly identifies goal-intent-only as the GCSL ablation in its factorial.
- The CEM planner — standard; Guarded A is a small-budget variant centered on the Direct plan.
- Cross-task transfer — explicitly fails (leave-one-task-out, cross-task actor-output transfer all negative, §6).

---

## 2. Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Where it ships |
|---|---|---|
| latent state `z_t`, goal `z_g` | HLA per-NPC state, belief state, sense projection | `riir-engine/src/hla/`, `katgpt-core` HLA + sense kernels |
| forward model `F_ϕ(z, a) → ẑ+` | `GameState` trait, `InducedCwmKernel::advance` (Plan 296) | `katgpt-core/src/induced_cwm/`, `riir-engine/src/game_state.rs` |
| **local intent `m_local = z_{t+1} − z_t`** | **`extract_functor`** (estimate displacement from transition pairs: `f = mean_k(target_k − source_k)`) | `riir-ai/crates/riir-engine/src/latent_functor/arithmetic/` (Super-GOAT R123, Plan 303/317) |
| **goal intent `m_goal = sg(z_g) − z_t`** | `extract_functor` from (current, goal) pair — same op, different anchor | same |
| **`G_η(z, m, a₋₁) → a` (shared operator, deployment)** | **`apply_functor_edge_into: out = state + sigmoid(β·(cos−τ)) · direction`** | `katgpt-rs/crates/katgpt-core/src/closure/functor_edge.rs` (Plan 040) |
| INTACT predictor's bilinear feature `z_t ⊙ m_t` | cosine-similarity gate (the `cos` term in `functor_gate`); optional Hadamard is a ≤1-line variant | `closure/functor_edge.rs`, `latent_functor/arithmetic/` |
| **stop-gradient on goal anchor** (asymmetric gradient routing) | **freeze/thaw discipline** — the goal latent is a frozen `MerkleFrozenEnvelope`-committed anchor when serving as deployment target; plastic when it's a real successor | `riir-neuron-db/src/freeze.rs`, Plan 341 |
| conditional action quotient `y ~_z y'` (Def 1) | coherence gate: two endpoints with the same projection onto `dir` are action-equivalent by construction | `functor_gate(coherence, β, τ) = sigmoid(β·(c − τ))` |
| state-intent interaction "action-semantic" representation | committed personality direction vectors, archetype blend, HLA emotion extraction | `riir-engine/src/committed_blend/`, `riir-neuron-db/src/archetype_blend_shard.rs`, `civ_emotion` (Plan 175) |
| Direct mode (zero-search control) | closed-form functor application — single matvec, no inner optimization | `apply_functor`, `apply_functor_edge_into` |
| Guarded A (small-budget CEM around Direct plan) | MCTS / CGSP / `ismcts_search_with_inference` (turn-based) — different planning paradigm; bounded local search is a known variant | `cgsp_runtime/`, `mcts.rs`, R145 |
| kNN / CKA family-conversion diagnostics | `eigenspace_alignment`, `cauchy_interlacing_check`, within-class effective rank (Plan 415), LatentTrajectoryGeometry (Plan 342) | `katgpt-core/src/spectral_hierarchy/`, `latent_trajectory_geometry.rs` |
| reward-free offline trajectories | sleep-time anticipation, CGSP self-play consolidation | `riir-engine/src/sleep_time/` (DEFAULT-ON), `cgsp_runtime/` |
| `L_world + λ_inv·L_z+ + λ_goal·L_goal` joint training | training recipe — not in katgpt-rs / riir-ai (modelless mandate) | → riir-train |

---

## 3. Distillation — duplicate detection vs our corpus

### 3.1 The dual-call inverse-dynamics + goal-conditioning recipe — **already evaluated, training-only**

**Research 358 (SMWM, 2026-07-01)** evaluated the *exact* same mechanism — `L_inv = ‖h_ψ(z_t, z_{t+1}) − a_t‖²` as auxiliary JEPA regularizer — and reached the verdict: *"Pass for modelless/runtime. ID is a genuinely novel auxiliary (zero grep hits) but it is purely a training-loop credit-assignment refinement — no modelless/runtime distillation worth shipping to katgpt-rs or riir-ai."* INTACT cites SMWM as [16] and lifts `L_inv` verbatim as the local physical-intent branch. The novel piece vs SMWM is the *second* (goal-conditioned) call through the same operator — but Research 324 (GC-IDM, the paper's [34]) already evaluated goal-conditioned inverse dynamics as **Gain** (actionable refinement to Motor-Gated DEC consumer policy, captured in `riir-ai/.issues/565`). INTACT's "shared operator + asymmetric gradient routing" is a training-loop architectural choice that does not produce a new runtime primitive beyond what SMWM + GC-IDM together already imply.

### 3.2 The distilled runtime primitive — **already shipped as Super-GOAT**

**Research 123 (Latent Functor Runtime Guide)** documents the INTACT-equivalent insight at runtime, **as a Super-GOAT**:

> "Analogy `A:B :: C:D` decomposes into [...] **functor application as residual-stream addition**. `e_target ≈ e_source + f`"

This is `z_{t+1} ≈ z_t + ρ(a)` verbatim — INTACT's local-intent construction. The latent-functor API ships exactly INTACT's deployment primitive modellessly:

```rust
// riir-ai/crates/riir-engine/src/latent_functor/arithmetic/ (Super-GOAT Research 123, Plans 303/317)
extract_functor(sources, targets, dim) -> (functor_dir, coherence)
//   f = (1/N) Σ_k (target_k − source_k)   ← mean displacement = INTACT's m_local estimate
apply_functor(source, functor, out)         ← out = source + f  = z + ρ(a)  (INTACT Direct mode)
functor_gate(coherence, beta, tau)          ← sigmoid(β·(c − τ)) trust gate

// katgpt-rs/crates/katgpt-core/src/closure/functor_edge.rs (Plan 040, edge-fused variant)
apply_functor_edge_into(state, params, direction, dim, out)
//   out = state + sigmoid(β·(cos(state,dir) − τ)) · direction
//   (the closed-form "conditional mean" Direct-mode action, zero search)
```

`sleep_time::HlaSleepTimeOp` (Plan 341, **DEFAULT-ON 2026-06-27**) inlines the same `z_i = c + dir_i` elementwise add and was explicitly noted in Issue 005 as replacing the `apply_functor` dispatch — so the INTACT primitive is not only shipped, it's been **promoted to default-on** as the canonical modelless latent-translation op.

### 3.3 The conditional action quotient — **operationally already captured by the coherence gate**

INTACT's Definition 1 (`y ~_z y' ⇐⇒ p*_E(a | z, y) = p*_E(a | z, y')`) is a *definitional* equivalence class — it tells us when two endpoints are action-equivalent. Operationally, this is what the `functor_gate` coherence discipline enforces: endpoints that project onto the same displacement direction (high cosine) get the same action; endpoints that don't (low cosine) get gated to identity. Proposition 3's recovery guarantee (proper-NLL minimization recovers the quotient on supported conditions) is the training-time analog of `extract_functor`'s closed-form displacement estimate — both say "the action law at a supported condition is recoverable from the displacement". The quotient is a useful conceptual framing but translates to no new runtime primitive.

### 3.4 The state-intent bilinear feature `z_t ⊙ m_t` — **related to, but not identical to, the cosine gate**

INTACT's `w^⊤(z_t ⊙ m_t) = z_t^⊤ diag(w) m_t` is a fixed bilinear interaction between state and intent, fed through a learned per-channel weight `w`. Our `functor_gate` computes the cosine similarity `cos(z, m)` — a *normalized scalar* bilinear interaction used as a single gate value. These are related (both are bilinear in z and m) but **not the same feature**: `z ⊙ m` is an elementwise product feeding a per-channel projection; cosine is a single normalized scalar. The controlled A–G ablation (Table A2) shows the matched grammar `[m_t, z_t ⊙ m_t]` adds +3.89 points over ungrammared variants — a measurable gain from the richer feature. **Why it doesn't land as runtime code:** our `apply_functor_edge_into` has no learned per-channel weight to consume the elementwise product; we'd need a learned policy head (the Motor-Gated DEC consumer, R168/Issue 565/572) for the bilinear feature to have a consumer. If/when that head ships, INTACT's grammar is the input-feature recipe. Until then, it stays a training-side detail.

### 3.5 The kNN / CKA family-conversion diagnostics — **related to, but not shipped as, our spectral diagnostics**

INTACT's `r = 0.954` correlation between predicted–expert **action-family kNN overlap** and Direct SR measures something specific: do the k-nearest neighbors in *predicted-action* space match the k-nearest in *expert-action* space, per checkpoint? Our `eigenspace_alignment` (Plan 156, Research 121) + `cauchy_interlacing_check` + within-class effective rank (Plan 415) measure eigenvector / eigenvalue structure of representation matrices — a related but different family of diagnostics. We do **not** ship an action-family kNN-overlap metric. The paper's "family-level preservation > pointwise recovery" finding (kNN 0.954 > action R² 0.815) is a genuine empirical insight about what predicts closed-loop control success; it's consistent with our spectral-diagnostic posture but is not literally covered by it. Adding a kNN-overlap diagnostic would be a small Gain if a learned policy head ever needs it; today no consumer exists.

### 3.6 Closest cousins across all 5 repos

| Cousin | Domain | Verdict / status | Overlap with INTACT |
|---|---|---|---|
| **Research 358 (SMWM)** | katgpt-rs | **PASS** (2026-07-01) | Identical local-intent mechanism (`L_inv` auxiliary); INTACT cites SMWM as [16]; same PASS precedent |
| **Research 324 (GC-IDM)** | riir-ai | **Gain** (2026-07-25) | GC-IDM is INTACT's [34]; same amortized-planning pattern (small MLP head on frozen JEPA latents → action); both Goal-conditioned IDM |
| **Research 426 (Temporal Straightening)** | katgpt-rs | **PASS** (2026-07-15) | Same authors LeCun/Ren, same LeWM benchmark, same JEPA PASS pattern; INTACT's "goal displacement > waypoint" extends paper's "linear A=I is the straight regime" |
| **Research 360 (AdaJEPA)** | katgpt-rs | **PASS** (2026-07-01) | Same authors Wang/Bounou/LeCun/Ren, same JEPA domain; same "runtime analog already ships" conclusion |
| **Research 138 (LeJEPA)** | katgpt-rs | **LOW-MODERATE GAIN** | Same author Balestriero; "we don't train JEPAs in Rust" downgrade precedent |
| **Research 123 + Plans 303/317 (Latent Functor Runtime)** | riir-ai | **Super-GOAT, shipped** | `extract_functor` constructs INTACT's `m_local` closed-form; `apply_functor` is Direct mode verbatim |
| **Plan 040 (Functor Edge)** | katgpt-rs | **Shipped** | `apply_functor_edge_into` is the closed-form INTACT Direct action with cosine-gated trust |
| **Plan 341 (Sleep-Time)** | riir-ai | **DEFAULT-ON 2026-06-27** | Ships `z_i = c + dir_i` (INTACT's translation op) modellessly |
| **Plan 296 (Induced CWM Kernel)** | katgpt-rs | **Shipped** | Frozen, BLAKE3-committed world model `advance(state, action)` — the deployment `F_ϕ` target |
| **R168 + R359 + Plan 357 (Motor-Gated DEC)** | riir-ai / katgpt-rs | **Super-GOAT, shipped** | Offline-rehearsal policy head over a frozen spatial-field world model — the full INTACT capability (frozen model + amortized policy + optional verification) at NPC scale |
| **Issue 565 (AdaLN-Zero + commit_window=1)** | riir-ai | **✅ DONE (Plan 519, 2026-07-31; G7b PASS)** | Captured the actionable refinements from GC-IDM for the Motor-Gated DEC consumer policy head. File removed per noise-reduction rule; resolution lives in Plan 519 + R168 |

---

## 4. Mandatory latent-space reframing (per SKILL §1 step 3)

| Target substrate | INTACT reframing | Status |
|---|---|---|
| **(a) HLA per-NPC latent state** | "Goal-displacement policy head reads HLA direction vectors" — already the committed-personality pitch (civ_emotion Plan 175) | Already shipped |
| **(b) `latent_functor/` action application** | "Inverse dynamics = `extract_functor`; Direct-mode action = `apply_functor`/`apply_functor_edge_into`" — verbatim, modelless | Already shipped as Super-GOAT (Research 123, Plan 303/317/040) |
| **(c) `cgsp_runtime/` curiosity + search** | "Filter uncontrollable distractors via prediction error" — already Pathak-style curiosity (Plan 277 DEFAULT-ON); "CEM as optional verifier" — already a known planner variant, our analog is MCTS/CGSP | Already shipped |
| **(d) LatCal fixed-point commitment (sync boundary)** | INTACT's intent coordinates are per-NPC latent, never cross sync; only the resulting raw action crosses — same discipline as existing HLA's 5 synced scalars | Boundary discipline inherited, no new bridge needed |
| **(e) `NeuronShard` / `MerkleFrozenEnvelope` / Raven consolidation** | Frozen encoder + INTACT predictor snapshot = BLAKE3-committed `InducedCwmKernel` (Plan 296); stop-gradient goal anchor = freeze/thaw atomic-swap discipline | Already shipped |
| **(f) DEC Stokes operators (`katgpt-dec`)** | No reframing — INTACT is action/displacement-centric, not divergence/curl/flux-centric | N/A |

Every substrate either already ships the equivalent or is orthogonal. **No new latent-to-latent operation is suggested by INTACT that the codebase does not already have.**

---

## 5. §3.5 Modelless-unblock check

The paper IS training-only (joint encoder + INTACT predictor + forward predictor + asymmetric gradient routing, all via backprop). Per §3.5 the question is whether the distilled primitive can be implemented modellessly. **It already is** — all three correction paths are shipped:

1. **Freeze/thaw path** — N/A as a *gate failure*. The primitive IS the runtime pattern: `extract_functor` constructs the displacement atomically; the stop-gradient goal anchor maps to `MerkleFrozenEnvelope` atomic swap (readers keep the old snapshot until the swap completes). The frozen INTACT predictor at deployment is just `apply_functor_edge_into` — a deterministic single matvec.
2. **Raw/lora reader-writer hot-swap** — N/A. INTACT's "Conditional operator learns the supported family mapping" is realized modellessly by `extract_functor` re-estimating the displacement direction from a transition buffer. No learned overlay needed.
3. **Latent-space correction** — N/A. The goal-displacement coordinate `m = z_g − z_t` is a closed-form latent subtraction; `apply_functor_edge_into` is the closed-form conditional mean; `functor_gate` is the closed-form trust gate. The entire Direct-mode inference path is modelless.

No deferral to riir-train is needed from the modelless side because **the runtime primitives already cover all three modelless paths** INTACT's mechanism could be realized through. The training recipe itself (joint encoder + INTACT predictor + forward optimization with asymmetric gradient routing + shared graph-isomorphic dual call) belongs in riir-train per the SMWM Research 358 precedent — and is a multi-loss refinement of the existing JEPA-pretraining + RLVR recipe, not a new research line.

---

## 6. Novelty gate (§1.5) — all four NO

| Q | Answer | Evidence |
|---|---|---|
| **1. No prior art?** | NO | Research 358 (SMWM, identical mechanism, PASS); Research 324 (GC-IDM = paper's [34], Gain); Research 426 (Temporal Straightening, same authors, PASS); Research 360 (AdaJEPA, same authors, PASS); Research 123 + Plan 303/317/040 (Latent Functor Super-GOAT, ships the deployment primitive); Plan 341 (DEFAULT-ON `z_i = c + dir_i`); R168 + R359 + Plan 357 (Motor-Gated DEC, ships the full frozen-model + amortized-policy + optional-verification capability) |
| **2. New capability class?** | NO | "Frozen world model + amortized forward-policy head with optional bounded verifier, no search at inference" already ships as Motor-Gated DEC + Induced CWM (R168, R145) |
| **3. Product selling point?** | NO | "NPCs act from goal-displacement latents without search" is already the latent-functor + sleep-time + Motor-Gated DEC selling point cluster |
| **4. Force multiplier (≥2 pillars)?** | WEAK | Touches functor + sleep_time + HLA + Induced CWM + Motor-Gated DEC, but all already integrated |

**Verdict: PASS for modelless/runtime.** Not Super-GOAT, not GOAT, not Gain.

---

## 7. §1.55 value-extraction scan (mandatory even on Pass)

**Actionable improvements check (re-examined honestly after user pushback on the first-draft "we already ship this" framing):**
- Does INTACT's data contradict a current config default? **No.** INTACT's "goal displacement > waypoint coordinate" finding (§5.1, E5 reversal: goal-displacement 89.39 vs waypoint 76.42 macro SR) is consistent with our `extract_functor`'s use of raw mean displacement (not normalized). We happened to pick the paper's E5 winner by default — but we never *measured* this on our substrate, so the alignment is luck, not validation.
- Does INTACT expose a failure mode in shipped code with no existing mitigation? **No direct failure mode.** But it exposes a *capability gap*: we have no shipped primitive that was **jointly trained across two condition families** (local physical + goal deployment) via a shared operator with asymmetric gradient routing. Our `apply_functor_edge_into` is a closed-form displacement op, not a learned shared conditional. The paper's gauge experiment (§5.2, Full-backbone → LeWM-actor = 5.08%) is causal proof that sharing buys something independent construction doesn't. This is not a *bug* (we don't currently need a learned shared operator), but it is a genuine gap relative to the paper's contribution — not the synonym I first claimed.
- Does INTACT unblock a known deferred task? **Yes — extends Issue 565.** INTACT's "shared operator across local+goal with asymmetric gradient routing" + "goal displacement > waypoint coordinate" findings are *training-side* guidance for the Motor-Gated DEC consumer policy head (R168 G7). Issue 565 (AdaLN-Zero + commit_window=1) covers the *runtime consumer* architecture but explicitly defers "training the encoder" as out of scope (line 66). INTACT fills the training-side half that Issue 565 left open: the dual-call loss `L_world + λ_inv·L_z+ + λ_goal·L_goal` with shared operator + stop-gradient-on-goal + the 4-slot grammar `[z; m; z⊙m; A(a₋₁)]`. **Filed as `riir-ai/.issues/572`** — extends Issue 565's scope from "runtime policy-head architecture" to "runtime architecture + training recipe".
- Does INTACT validate our design? **Partially** — the "family-level kNN > pointwise R²" finding (r = 0.954 vs 0.815) is *consistent with* our spectral-diagnostic posture, but we don't ship the kNN-overlap metric itself, so this is post-hoc rationalization, not validation.

**Verdict: PASS for modelless/runtime repos (the runtime primitives genuinely do cover the deployment path), BUT with two honest corrections to the first draft:**
1. The framing "we already ship this" was wrong — the honest framing is **substrate mismatch**: we construct at runtime what INTACT jointly learns during training; the shared-operator coupling is a real training-side contribution with no runtime analog, and it lands on a tracked consumer (Issue 572), not on the modelless/runtime repos.
2. The first draft under-weighted the genuinely novel pieces (shared-operator coupling, conditional action quotient as a richer equivalence than coherence, kNN-overlap diagnostic, the gauge causal experiment). These are real contributions; they just don't translate to new code in `katgpt-rs`/`riir-ai` because (a) we don't train JEPAs, (b) we don't have a learned policy head that would consume the bilinear feature or the kNN diagnostic, (c) the conditional action quotient is a definitional framing captured only approximately by `functor_gate` coherence.

**No Gain-level primitive to ship to katgpt-rs/riir-ai/riir-chain/riir-neuron-db.** The training-side guidance lands on Issue 572 (riir-ai, blocked on R168 G7). The one-line riir-train note (§9) stands.

---

## 8. MOAT gate per domain

| Repo | In-scope? | MOAT contribution | Decision |
|---|---|---|---|
| `katgpt-rs` (public) | Marginal | None — Direct-mode primitive already shipped as `apply_functor_edge_into` (Plan 040) + `extract_functor` (Super-GOAT R123). No new open primitive to add. | **No file created** (this note is the only output) |
| `riir-ai` (private runtime) | In-scope | None — Research 123 (Super-GOAT) + R168 (Motor-Gated DEC) + R145 (Induced CWM) already cover the runtime IP, strictly more broadly. Issue 565 already captures the actionable policy-head refinements. | **No guide created** |
| `riir-chain` (private chain) | Out of scope | N/A — INTACT's intent coordinates are encoder-local latent geometry, never cross sync | — |
| `riir-neuron-db` (private shards) | Out of scope | N/A — the frozen encoder + INTACT predictor already commit via BLAKE3 (InducedCwmKernel); stop-gradient anchor maps to existing MerkleFrozenEnvelope | — |
| `riir-train` (private training) | In-scope | Marginal — the joint encoder + INTACT predictor + forward training with asymmetric gradient routing is a multi-loss refinement of existing JEPA-pretraining + RLVR objectives. SMWM's `L_inv` was already noted there (R358 §8); INTACT adds the goal-conditioned branch + shared operator + stop-gradient-on-goal as a multi-task extension. **One-line note in `riir-train/.research/` if prioritized.** | **→ riir-train** (see §9) |

---

## 9. → riir-train (one-line redirect per SKILL §"Redirect to riir-train")

If prioritized, file a plan in `riir-train/.plans/` extending Research 358's SMWM note + the existing JEPA-pretraining recipe: add **`L_goal = −log p_η(a_t | z_t, sg(z_g) − z_t, a_{t-1})`** as a second-branch auxiliary alongside `L_inv` (R358 §8), sharing the same operator `G_η` with the 4-slot grammar `[z; m; z ⊙ m; A(a_{t-1})]` and asymmetric gradient routing (attached on `z_{t+1}`, stop-gradient on `z_g`). A/B-test on Bomber/Civ/Go arenas against the SMWM-`L_inv`-only baseline. Hypothesis (per INTACT §5.1 single-task gain of +8.78 macro points at E5 from adding the goal branch): the goal-conditioned call helps most in contact-rich tasks (PushT-analog) where the local inverse alone underdetermines the action. Test on a 2-D navigation + manipulation toy (Wall/PushT analog) where the kNN-family-conversion metric can be compared against expert ground truth — mirroring INTACT's Figure 5 setup. **Not pursued here — out of scope for this workflow.**

The only genuinely transferable *runtime* observation — INTACT's note that "the INTACT predictor's diagonal-Gaussian mean can hide multimodality at junctions" (§6) and that "mixture actors / uncertainty-triggered verification are natural extensions" — is already the codebase's posture: `functor_gate` is sigmoid-bounded (not softmax — per AGENTS.md constraint #2), and the optional MCTS/CGSP search is the multimodal-fallback path. No new primitive is implied.

---

## TL;DR

**Paper:** *INTACT: Intent-To-Action Learning for Search-Free World Models* (Sun, Zhao, Zhang, arXiv:2607.26056, 2026-07-28).

**Verdict:** **PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db.** The paper's distilled runtime primitive — "the action realizing a latent displacement `m = z_g − z_t` is recoverable as a frozen conditional policy `a = μ_η(z, m, a₋₁)` evaluated in a single forward pass (Direct mode), with optional bounded CEM verification (Guarded A)" — **is already shipped modellessly** as:

- `katgpt-rs/crates/katgpt-core/src/closure/functor_edge.rs::apply_functor_edge_into` (Plan 040, Super-GOAT R123) — cosine-gated `out = state + sigmoid(β·(cos−τ)) · direction` (Direct mode verbatim, zero search).
- `riir-ai/crates/riir-engine/src/latent_functor/arithmetic/::extract_functor` (Super-GOAT R123, Plan 303/317) — constructs INTACT's `m_local = z_{t+1} − z_t` displacement closed-form from a transition buffer, no gradient.
- `riir-ai/crates/riir-engine/src/sleep_time/hla_op.rs::HlaSleepTimeOp` (Plan 341, **DEFAULT-ON 2026-06-27**) — `z_i = c + dir_i` elementwise add, the canonical modelless latent-translation op.
- `riir-ai/crates/riir-engine/src/game_state.rs::InducedCwmKernel` (Plan 296) — the frozen, BLAKE3-committed world-model `F_ϕ` target.
- `riir-ai` R168 + R359 + Plan 357 (Motor-Gated DEC) — the full Super-GOAT capability: offline-rehearsal policy over a frozen spatial-field world model, with optional verification — strictly more integrated than INTACT's robotics-domain POC.

The training recipe (joint encoder + INTACT predictor + forward predictor with asymmetric gradient routing + shared graph-isomorphic dual call) is the same family EnvRL (R127) already evaluated as training-only, refined by SMWM (R358) as `L_inv`-auxiliary, and extended by INTACT with the goal-conditioned second branch. It belongs in `riir-train` as a multi-loss variant of the existing JEPA-pretraining + RLVR recipe, not in the modelless/runtime repos. The conditional action quotient (Definition 1) is a definitional equivalence captured operationally by the existing `functor_gate` coherence discipline; the state-intent bilinear `z ⊙ m` is a ≤1-line extension of the cosine gate if any consumer ever needs it; the kNN/CKA family-conversion diagnostics are already covered by `eigenspace_alignment` + within-class effective rank.

This is the **same canonical JEPA-world-model PASS class as SMWM (R358), Temporal Straightening (R426), and AdaJEPA (R360)** — but the first draft's framing ("we already ship this") was too dismissive and has been corrected above to **substrate mismatch**: we construct what they jointly learn, and the shared-operator training-side contribution is genuinely novel (proven causally by the paper's gauge experiment), it just doesn't land as runtime code. Research 324 (GC-IDM, the paper's direct cited prior art [34]) set the Gain precedent for the runtime-side policy-head refinements (AdaLN-Zero horizon modulation, commit_window=1) — those land on Issue 565; INTACT extends the scope to the *training-side* recipe (shared operator + dual-call loss + asymmetric routing), tracked in `riir-ai/.issues/572`.

**Files created this session:** `katgpt-rs/.research/462_INTACT_Isomorphic_Intent_To_Action_Search_Free_World_Models.md` (this note — the only katgpt-rs output). PASS-Redirects cross-references added to R358 (SMWM), R426 (Temporal Straightening), and `riir-ai/.research/324` (GC-IDM). **Tracked follow-up:** `riir-ai/.issues/572` (INTACT × GC-IDM combined training recipe for Motor-Gated DEC consumer policy head — extends Issue 565 from runtime architecture to runtime + training).

**Recommended next step:** None for katgpt-rs / riir-ai / riir-chain / riir-neuron-db. The riir-train follow-up (add `L_goal` second-branch auxiliary alongside `L_inv`) is optional and out of scope for this workflow.

---

## 10. PoC-scope note (per SKILL §3.6)

A "defend-wrong" PoC at `riir-poc/` is **not required** for this verdict. §3.6 mandates a PoC when a verdict *downgrades a paper on the grounds that "the runtime analog already ships" or achieves "parity"* — i.e. when an architectural-evidence-only claim asserts quality parity. This verdict makes **no quality-parity claim**:

- It does not claim the shipped `apply_functor_edge_into` "matches" INTACT's trained `G_η` on the LeWM benchmark.
- It does not claim the shipped `extract_functor` "performs as well as" INTACT's joint encoder training.
- It claims only **architectural coverage** (the runtime pieces ship separately under different vocabulary) + **substrate sufficiency** (the modelless Direct-mode path is realizable in our stack as `extract_functor` + `apply_functor_edge_into` + `functor_gate` + freeze/thaw).

The verdict is a PASS, not a parity-backed downgrade of a quality claim. The §3.6 PoC mandate triggers on the latter; it does not trigger on a structural-coverage PASS where the paper's training-time machinery (joint encoder + INTACT predictor optimization) is explicitly out of scope for the modelless/runtime repos.

If a future plan *does* consume INTACT's framing for a runtime change (e.g. adding the state-intent Hadamard feature `z ⊙ m` to `apply_functor_edge_into`, or wiring the Motor-Gated DEC consumer policy to use goal-displacement coordinates per Issue 565), THAT plan would carry its own quality-gate PoC. This research note does not.
