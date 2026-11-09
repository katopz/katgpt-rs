# Research 662: LARC — Latent Reasoning in Continuous Space (Info-Gain Trace Reward, Test-Time Latent Budget)

> **Source:** "Latent Reasoning in Continuous Space for Unified Multimodal Models" (LARC) — Jeon, Jeong, Lee, Cho, Shin; KAIST/POSTECH/RLWRLD; arXiv:2610.42149, NeurIPS 2026. Project: rootyjeon.github.io/larc-umm. Code: github.com/rootyJeon/LARC @ `8e7e4a9479877a723e5c7b2856d727b73e627737` — **README-only placeholder at read time** (inference code "expected 11/9", training/eval code unreleased, no LICENSE file); every quote below cites the paper text, not the repo.
> **Date:** 2026-11-09
> **Status:** Active
> **Related Research:** 660 (Diffusion-Similarity Curriculum Diversity — the batch-level diversity cousin, filed same day), 622 (HybridAgent — Plan 461's research), 496/500 (hint_regret sources), 322 (Report-the-Floor)
> **Related Plans:** katgpt-rs 634 (diffusion_diversity), riir-train 461 (hybrid-link latent bridge), riir-train 059 (G-Zero DPO/GRPO), riir-ai 646 (CGSP diffusion-diversity consumer)
> **Classification:** Public

---

## TL;DR

LARC makes a unified multimodal model (BAGEL) reason in the same continuous space its image module generates in: after a `<latent>` token the **last hidden state is fed back as the next input embedding**, alternating short text spans with continuous latent steps; a curriculum-SFT stage progressively activates latent steps, then a self-evolving GRPO stage rewards each trace by its **information gain about the target under a frozen reference** — `r_IG = log p_ref(x|c,h(τ)) − log p_ref(x|c)`, estimated as a paired flow-matching loss difference under **shared noise** — plus a pairwise-cosine diversity term that prevents latent collapse. Result: +7pp GenEval over BAGEL, and the first clear **test-time scaling** in a reasoning-augmented UMM — more latent steps at ~0.12 s/step vs best-of-N at ~46 s/sample (≈380× cheaper marginal compute), monotonic gains to L=10.

**Distilled for katgpt-rs (modelless, inference-time):** three transferable fragments survive contact with the shipped stack — (1) the **fixed-point early-exit lever** on attractor-iteration latent kernels (break the K-loop once the step-to-step residual falls below ε; LARC's own step-collapse finding is the motivating cousin only — its failure channel needs a per-step consumer this stack does not have) → katgpt-rs Issue 991; (2) three **training recipes** with named GRPO/latent-loop consumers (r_IG self-reward, curriculum staged latent activation, trajectory-internal diversity reward) → riir-train Issue 681; (3) everything else signal-diffs into shipped cousins and is **audited-discard**ed below — the latent-step mechanism, the CRN paired estimator, and the escalation-budget axis each have a live home already.

---

## 1. Paper Core Findings

1. **Modality-mismatch diagnosis:** reasoning-augmented UMMs reason in discrete tokens while generation runs in continuous space; token-level CoT for image generation collapses into prompt rephrasing and *degrades* visual reasoning vs the no-reasoning baseline (Table 6: MIRA 7.5 baseline → 7.1 token-CoT → 9.7 latent).
2. **Interleaved latent trajectory:** τ = (s₁,l₁,…,s_N,l_N); in latent mode `E_{t+1} = [E_t; h_t]` (last hidden state as next input embedding); the trajectory is read by the image module through cross-attention. Interleaved beats Coconut-naive pure-latent (+0.02 OneIG, +0.44 LC-T editing) — text spans keep rollout stochasticity, latent steps keep dense representations.
3. **Stage 1 curriculum SFT:** phase 0 trains text-only; phase k activates the first k latent steps (LR decayed per phase); loss = NTP over text + flow-matching on the target image; gradient flows through latent steps with **no direct latent supervision**.
4. **Stage 2 self-evolving GRPO:** reward `r_IG(τ) = log p_ref(x|c,h(τ)) − log p_ref(x|c)`; E[r_IG] = I(x; h(τ)|c) ≥ 0 (conditional MI, Jensen); estimated as `L_flow(x,c) − L_flow(x,c,h(τ))` with **both evaluations sharing the same noise sample and time step** so the sample-wise constant C(x) cancels exactly (unbiased, low variance, no reward model, no ground-truth chains, no external critic). Plus `r_div = mean pairwise (1 − cos(l_n, l_n'))`, λ_div = 0.1. GRPO: group 8, clip 0.2, KL β=1e-3. Cost: 8×A100, ~300 GPU-h (SFT) + ~96 GPU-h (RL).
5. **Anti-collapse evidence (Table 12):** λ_div = 0 → pairwise cos **0.97** (collapsed) with GenEval 0.83 / MIRA 9.8; λ_div = 0.1 → cos 0.79, GenEval 0.88 / MIRA 11.7. The collapse regime is a *measured* failure signature, not a hypothetical.
6. **Test-time scaling:** budget-forcing latent steps L=1→10 improves all three task families monotonically and beats best-of-N at every matched budget; marginal cost 46.8→48.0 s (L=1→10) vs best-of-N 46.5→469.0 s (N=1→10).
7. **Why not DPO:** no human-interpretable preference exists over hidden-state traces; external reward models score rendered images, not traces — structurally misaligned. (Their argument for self-reward.)
8. **Results:** GenEval 0.88 (+7pp over BAGEL, SOTA), DPG 90.04 (+5.0), OneIG 0.866 (+10.2pp), CompBench LC-T 22.51 (+2.1), MIRA 11.7 (+4.2); understanding does not regress (MME/MMB/MMMU all up). Data: 1M interleaved CoT samples (FLUX-Reason 500K + WEAVE 100K + ScaleEdit 400K).

## 2. Distillation

### 2.1 Published prior art (web sweep, agent-verified)

| Work | What it does | What it does NOT do vs LARC |
|---|---|---|
| MCOUT, arXiv:2508.12587 (NeurIPS 2025) | Reuses LM last hidden state as continuous thought (same mechanism), multimodal | Understanding-only; no generation, no RL, no TTS |
| Geiping et al., arXiv:2502.05171 | Latent test-time scaling via recurrent depth (text LM, 562 cites) | Text-only |
| UniT (Meta, CVPR 2026) | Sequential CoT TTS **beats best-of-N for generation** in a unified model | Text CoT, not latent |
| EndoCoT, arXiv:2603.12252 (ECCV 2026) | Iterative latent thought states bridged to a DiT | Two-model pipeline, text-grounded, no info-gain RL |
| MG-Select, arXiv:2510.05681 | Verifier-free model-internal selection signal (the reward's ancestor) | Test-time selection for VLAs, not an RL reward |
| DiverseGRPO (CVPR 2026) | Diversity-aware GRPO for T2I mode collapse | Image-level diversity, not trajectory-internal latent steps |

Verdict: LARC's *combination* (hidden-state-feedback latent reasoning inside a unified **generation** model + info-gain RL reward + latent-budget TTS) has no direct prior; its "first TTS for generation in UMMs" headline is too broad (UniT holds the sequential-TTS-beats-best-of-N result in text CoT); the defensible novelties are the continuous-latent instantiation and the ~380× cost ratio.

### 2.2 Shipped-cousin signal-diffs (§3.6 — every "covered" claim carries the formula-level diff)

| Paper component | Shipped cousin (verified this session) | Signal the cousin consumes | Signal the paper consumes | Delta that survives |
|---|---|---|---|---|
| Hidden-state feedback latent steps | riir-train **Plan 461** hybrid-link lane (`hybrid_link_loop`: `run_latent_span` feeds `R(last_hidden)` back through the embedding-input forward, exits at `</latent>`; E0-verified on Kimi-K3 0.4B; Phase 2 E1 complete on frozen Bonsai-27B — stage-2 map PASS, **G2 structurally fails** (LM-head cost ceiling), Phase 3 owner-gated, no production consumer) | trained latent-continuation bridge, teacher-forced stage-1 + stage-2 map | same feedback loop, curriculum-staged training + RL on the loop | curriculum **staged activation** of the loop → riir-train 681 |
| (inference twin) | riir-infer `issue929_bonsai2_loop_probe.rs` — K looped passes advance GDN state; `B929_REINJECT=1` decodes *from* the loop-refined state (measurement lane, Phase-2 adjudicated) | K-step loop refinement of recurrent state at inference | same at UMM scale | none new — posture parity already measured |
| (game-kernel twin) | katgpt-micro-belief `LatentThoughtKernel` (K-iteration attractor over `MicroRecurrentBeliefState`; K=1 bit-identical to single step; `evolve_belief` per-tick routing in the stack) | K inner matvec+sigmoid iterations per belief step | N transformer-layer hidden-state replays | none new at this layer's scale ([f32;8..32]) |
| Info-gain paired estimator | katgpt-core `engram/privilege.rs` (Issue 656, Bench 656 GOAT): `δ_k = score(state + contribution_k) − score(state)`, outcome-weighted EMA, sigmoid gate `σ((Δ−m)/s)` | with-vs-without counterfactual advantage of a *memory contribution*, runtime selection gate | with-vs-without log-density delta of a *reasoning trace*, RL training reward | the REWARD consumer (training-side) → riir-train 681; runtime gate already ships |
| Shared-noise pairing | katgpt-core `hint_regret` (Plan 576, Research 496/500): paired-rollout VoI under **CRN — common random numbers**, "the shared noise source cancels in the paired difference", empirical-Bernstein bound "SEES the CRN variance reduction" (measured 2.76× variance ratio per Bench 576 G2) | paired return difference under shared seed, curriculum triage statistic | paired loss difference under shared noise, GRPO reward | none new at the estimator level — CRN is shipped and measured; the MI non-negativity floor (E[r] ≥ 0) is a derivation nicety our binary-outcome verifiers don't need |
| Diversity / collapse | katgpt-core `cgsp`: `EntropyCollapse`/`CollapseSignal`/`CollapseDetector` (entropy-threshold detection + exploration injection; Bench 911 rewind) | output-distribution entropy of the sampling loop | pairwise cosine of *latent step vectors* within one trajectory | **no consumer survives the signal-diff** — verdict round 1: our latent-thought kernels are *attractor iterations* (`LatentThoughtKernel` module doc: "move the belief further toward a fixed point"), successive cosine ≈ 1 is designed convergence, and LARC's collapse channel requires a consumer reading every step separately, which the stack does not have; the surviving fragment is a convergence-residual early-exit lever → katgpt-rs Issue 991 (reframed) |
| (batch-level diversity) | Research **660** family (same day): Plan 634 `diffusion_diversity` (Div_P^τ graph-diffusion spread), riir-ai Plan 646 CGSP consumer (open), reflex 115 / refine 222 arms | distribution-level spread over a *similarity graph* for curriculum **batch selection** | within-trajectory step-level repulsion as an RL **reward** | different granularity + different consumer; complementary, cross-ref'd in riir-train 681 |
| Test-time think-budget scaling | riir-rethink ESC lane (`escalating_backend.rs`, `esc_voi.rs` VoI observer Issue 047; `esc_utility.rs` **Bench 097 close-negative**: pre-escalation features predict escalation value only via cheap confidence — falsifier mass 1.000 on both suites) | cross-model escalation value from pre-escalation features | same-model latent-step budget, measured monotonic gains | regime mismatch (cross-model escalation vs same-model internal steps) + Bench 097's standing negative → **audited discard** |
| Width-vs-depth compute arms | Bench 898 guided width rollouts; riir-ai Bench 955 multi-hypothesis width GOAT (N parallel `evolve_belief`, width-vs-depth paired) | width (parallel hypotheses) vs depth at matched compute | latent-step depth vs best-of-N samples | no best-of-N consumer lane to order against → audited discard (design guidance only) |
| Curriculum staging | katgpt-core `ladder_gate.rs` (ATC-style stage controller: advance iff held-out accuracy passes, Issue 887) | quality-gated stage advance on held-out accuracy | schedule-based phase advance + LR decay | schedule-vs-quality-gated variant → riir-train 681 recipe 2 |

### 2.3 Path 0 inventory (adversarial panel merged; every row ends in a/b/c)

Panel: No-GD advocate extracted five modelless candidates; Model-based advocate extracted five recipe components; both merged below with coordinator-verified facts (every code fact above grepped this session).

| # | Component | Track | Endpoint |
|---|---|---|---|
| 1 | Interleaved hidden-state feedback loop | a | **(c) audited discard** — mechanism ships three times (Plan 461 trained lane; Issue 929 reinject; LatentThoughtKernel game kernel); no UMM-generation readout exists to port, and none is wanted (no image lane) |
| 2 | r_IG info-gain estimator, modelless use | a | **(c) audited discard** — runtime selection gates already consume the counterfactual signal (privilege δ) and the CRN pairing (hint_regret); a new runtime consumer would need a probabilistic frozen scorer our binary verifiers don't provide — no signal source, no primitive |
| 3 | Successive-step geometry on latent kernels | a | **(b) issue, REFRAMED at verdict round 1** — round-1 premise (cosine-collapse monitor) was WRONG: `LatentThoughtKernel` is an attractor iteration, so high successive cosine is convergence, not collapse, and no module reads intermediate steps (the loss channel LARC's collapse needs). What survives is the standard **fixed-point early exit** (residual ‖s_{t+1} − s_t‖ < ε for w steps ⇒ break the K-loop, tolerance-declared) → katgpt-rs Issue 991; LARC demoted to motivating cousin |
| 4 | Test-time latent-budget forcing on ESC | a | **(c) audited discard** — Bench 097 close-negative is standing counter-evidence on this lane; the modelless cheap leg has no latent-step knob; regime mismatch (same-model internal steps vs cross-model escalation) |
| 5 | +step-vs-+sample cost ordering law | a | **(c) audited discard** — width arms already measured (Bench 898/955); no best-of-N-vs-step consumer exists; recorded as design guidance in §2.2 only |
| 6 | r_IG self-reward for GRPO lanes | c | **(b) issue** — riir-train 681 recipe 1 (consumer: `loss_grpo.rs` + G-Zero challenger; replaces/augments delta_filter heuristics) |
| 7 | Curriculum staged latent activation | c | **(b) issue** — riir-train 681 recipe 2 (consumer: Plan 461 hybrid-loop training stages; ladder_gate is the quality-gated sibling) |
| 8 | r_div trajectory-internal diversity reward | b/c | **(b) issue** — riir-train 681 recipe 3 (evidence: Table 12; cross-ref 680's batch-level Div gate — complementary granularity) |
| 9 | Flow-matching image head / UMM training loop | c | **(c) audited discard** — no image-generation training loop exists in the stack (refab ships oracle fixtures, not trained heads); mechanism-level: no consumer surface at any margin |
| 10 | DPO-over-latent-traces critique | c | **(c) audited discard** — no latent-trace DPO lane exists; the critique transfers as a caution row inside riir-train 681 |

### 2.4 Novelty gate (§1.5) — not claimed

No GOAT/Super-GOAT filing is made. Pinned claim, verdict-round-2 form: *a fixed-point early-exit test for attractor-iteration latent kernels (residual ‖s_{t+1} − s_t‖ < ε for w steps ⇒ break the K-loop), consuming step-to-step residuals, distinguished from the precomputed-input matvec reuse by cutting ITERATION COUNT rather than per-step compute.* Q1: the early-exit technique itself is standard convergence detection — no novelty claimed; the issue files it as a small latency lever with a census-first premise check. Q2: **fails** — truncation/early-exit is an existing behavior class, and the round-1 collapse framing failed harder (the attractor's high successive cosine is designed convergence; the collapse channel has no consumer in this stack). Per the no-candidate-escape-hatch rule: the lever goes to Issue 991, not a GOAT note.

### 2.5 Weakest points (named)

1. **The early-exit premise is unmeasured.** Whether real bench inputs converge before K is unknown (the census is T1/T3; the negative path — "never converges within K, close the issue" — is the likely outcome and is pre-registered). The round-1 stronger claim (collapse detection) was REFUTED at verdict round 1: the attractor kernel's high successive cosine is the designed behavior, and LARC's collapse needs a per-step consumer this stack does not have.
2. **The training recipes ride lanes of uncertain activity.** G-Zero's last burst was July-era; Plan 461 Phase 3 is owner-gated with a structurally-failing G2 on its serving half. Issue 681 therefore carries the 680-style promotion trigger (GPU window + consumer lane commits) instead of a plan.
3. **Code provenance is paper-only.** The repo is a README placeholder with no license at read time; if inference code lands (announced 11/9), a follow-up read should re-verify the reward implementation details (Algorithm 1) against the actual code before any implementation.

## 3. Verdict

| Track | Verdict | Routing |
|---|---|---|
| a — modelless/runtime | **Gain** (one small convergence early-exit lever idea, reframed at verdict round 1; rest discarded with audited signal-diffs) | katgpt-rs Issue 991; discards recorded in §2.3 |
| b — self-adaptive | **Gain** (folded into the training-recipe issue — no runtime RL-trace consumer exists) | riir-train Issue 681 recipe 3 |
| c — training (Path 0.5) | **Gain** (applicable recipes, issue vehicle per ungoat discipline — no committed GPU budget or active consumer lane; promotion trigger named) | riir-train Issue 681 |

**MOAT gate:** the detector idea is a katgpt-rs base primitive (generic, no game/chain/shard IP) — in scope as an issue; the training recipes are riir-train moat — in scope. No public/private tier hazards: both destinations are the correct tiers; nothing rethink-class lands public.

## 4. Routing summary

- `katgpt-rs/.research/662_LARC_Latent_Reasoning_Continuous_Space.md` (this note)
- `katgpt-rs/.issues/991_attractor_fixed_point_early_exit.md` (track-a lever idea, reframed at verdict round 1 — the round-1 collapse framing was refuted; the old filename `991_latent_step_cosine_collapse_monitor.md` never committed)
- `riir-train/.issues/681_larc_self_reward_training_recipes.md` (track-b/c recipes)
- Cousins referenced by name+number (660, 634, 646, 461, 059, 680, 991/Issue 656 record in Bench 656, hint_regret Plan 576) so future greps land here.

## 5. PASS-Redirects

None (verdict is not PASS). External kin recorded for future greps: MCOUT arXiv:2508.12587 "Multimodal Chain of Continuous Thought" · Geiping arXiv:2502.05171 "Scaling up Test-Time Compute with Latent Reasoning" · UniT (Meta CVPR 2026, unified multimodal CoT test-time scaling) · EndoCoT arXiv:2603.12252 · MG-Select arXiv:2510.05681 · DiverseGRPO (CVPR 2026).
