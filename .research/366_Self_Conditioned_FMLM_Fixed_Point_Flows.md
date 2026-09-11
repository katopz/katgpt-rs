# Research 366: Self-Conditioned FMLM via Fixed-Point Flows

> **Source:** "Self-Conditioned Flow Map Language Models via Fixed-Point Flows" — Yoo, Kim, Eijkelboom, Lee, Boffi, Hong, Kim (KAIST / UvA / CMU). [arXiv:2607.00714](https://arxiv.org/abs/2607.00714). Jul 2026.
> **Reference impl:** https://github.com/Ugness/self-conditioned-fmlm (PyTorch; ELF, ELF*, FMLM* checkpoints on HuggingFace)
> **Date:** 2026-07-02
> **Status:** Done
> **Related Research:** 035 (Attractor Models / DEQ), 041 (RePlaid — self-conditioning), 073 (LT2 Looped), 097 (Training-Free Looped), 265 (CoFRe / FP-MGM — **direct sibling paper**), 273 (ELT Elastic Looped), 344 (Implicit FP RNN — **closed this design space**)
> **Related Plans:** 066 (D2F), 079 (ELF modelless), 108 (LT2 `LoopMode::WeightShared` — *is* the FP block), 136 (TF-Loop), 222 (`self_cond_draft` — 2-pass SC), 258 (RCD — closest cousin to warm-start carry), 291 (D2F 3SR warm-start)
> **Classification:** Public
> **Update (2026-09-11):** superseded-in-scope by "Thinking with Looped Flows" ([arXiv:2609.11801](https://arxiv.org/abs/2609.11801)) — which positions self-conditioning as the special case z=x̂ of stateful denoising, and trains the recurrence along flow timesteps with shared noise (measurably better: 58.8 vs this line's ARC numbers). Distill + Path 0 merge: `riir-train/.research/452_Looped_Flows_Training_Recipe_Distill.md`; execution plan: `riir-train/.plans/395_looped_flows_recipe_arm.md`.

---

## TL;DR

FMLM★ proves that **self-conditioning** (the `z = sg(D(x, 0))` trick from Analog Bits, Chen et al. 2022) in continuous flow language models implicitly learns a **fixed-point Picard iteration** `z_{j+1} = D_t(x, z_j)` that refines the denoiser estimate toward the Bayes-optimal prediction. Under a contractivity assumption (η < 1), the iteration converges exponentially to a unique fixed point z⋆ (Props 3.1–3.5), and the **fixed-point velocity** `b⋆(x) = (D⋆(x) − x)/(1−t)` is autonomous — so it defines an ordinary flow with a valid flow map. The paper then distills this into FMLM★ (a few-step flow map) via fixed-point distillation (CDEQ) + two-time-denoiser semigroup distillation, achieving SOTA one-/few-step generation on OpenWebText.

**Distilled for katgpt-rs (modelless, inference-time):** ~95% of the inference surface is **already shipped**. `LoopMode::WeightShared { loop_count }` (Plan 108, default-on, GOAT 8/8) *is* the iterated fixed-point block; `LoopMode::TrainingFree` (Plan 136) is a strictly more sophisticated ODE-motivated variant; `ResidualRelevanceScorer.is_converged` (Plan 085, default-on) is the halt; Plan 222 `self_cond_draft` is the 2-pass self-conditioned draft; CoFRe/Research 265 already distilled the direct sibling paper (FP-MGM + three-state warm-start). The only modelless delta is the **cold-start equivalence** empirical finding (warm-start across flow steps is merely a heuristic for better FP initialization; cold-start z₀=0 with enough iterations reaches the same frontier — their Fig. 3) — which is a config insight on a non-hot path, not a primitive. The training recipes (CDEQ, semigroup flow-map distillation, FMLM★) → **riir-train**.

**Verdict: Gain** (leaning Pass — heavily saturated prior art). No plan, no Super-GOAT guide, no open primitive. The note exists to (a) record the "self-conditioning = fixed-point" theorem as already-covered territory, (b) route training recipes to riir-train, (c) prevent re-evaluation by a future agent.

---

## 1. Paper Core Findings

### 1.1 The fixed-point view of self-conditioning (§3.1–3.2)

A self-conditioned flow denoiser `D̂_t(x, z)` takes its own previous estimate `z` as conditioning. Training mixes two terms (Eq. 7): the usual denoising loss (z=0) and a self-correcting loss (z = sg(D̂_t(x, 0))). **Proposition 3.1** proves the Bayes-optimal minimizer is independent of z (z ⊥ x₁ | I_t) — so the loss trains the model to map *every* on-distribution z to the ideal denoiser target.

**Key insight:** the self-conditioned denoiser defines a Picard iteration `z_{j+1} = D̂_t(x, z_j)` (Eq. 10). Under **contractivity** (Def 3.2, |D̂(z) − D̂(z')| ≤ η|z − z'|, η < 1), **Proposition 3.3** gives:
- (i) exponential convergence to a unique fixed point z⋆;
- (ii) per-iteration error bound `|z_j − D_t(x)| ≤ |D̂(x,z₀) − D_t(x)| + (η − ηʲ)/(1−η) · |z₁ − z₀|`;
- (iii) fixed-point error bound `|z⋆ − D_t(x)| ≤ |D̂(x,z₀) − D_t(x)| + η/(1−η) · |z₁ − z₀|`.

Appendix A.2 shows self-conditioned training *approximately* induces contractivity (the loss confines the denoiser to a contractive set above a scale w, with leakage ≤ ε).

### 1.2 Fixed-point flows (§3.3–3.4)

Replacing the self-conditioning state by its fixed point yields the **fixed-point velocity** `b⋆_t(x) = (D⋆_t(x) − x)/(1−t)` (Eq. 14), which depends only on (t, x) — an ordinary ODE. **Proposition 3.4:** when D⋆ = D (Bayes-optimal), b⋆ recovers the true velocity b. The **fixed-point flow map** X⋆_{s,t} (Prop 3.6) satisfies the semigroup condition X⋆_{s,t} = X⋆_{u,t} ∘ X⋆_{s,u}.

### 1.3 Cold-start vs warm-start (§3.3, the modelless insight)

Conventional self-conditioned generation (Eq. 9) warm-starts: `ẑ^0_{t_i} = ẑ⋆_{t_{i−1}}` (carry the previous FP estimate across flow steps). The paper shows this is **merely a sampling heuristic**:
- **Cold-start** (z₀ = 0 at every flow step) with 1 fixed-point iteration (FPI) is worse than warm-start (their Fig. 3).
- **Cold-start with 100 FPIs matches warm-start with 100 FPIs** — initialization becomes irrelevant once enough iterations are run (Prop 3.5: `j ≥ log(|z₀−z⋆|/ε) / log(1/η)` iterations suffice).

**Implication:** warm-start is not a defining property of self-conditioned flows; it is a single-step approximation that happens to land near the fixed point. The flow-state / self-conditioning-state coupling in conventional SC generation (Eq. 9) is a byproduct of warm starts, not a structural feature.

### 1.4 Distillation recipes → riir-train (§3.3–3.4, §4.2)

- **Fixed-point distillation** (Eq. 19): regress D⋆ onto the converged fixed point z⋆ (estimated by iterating from z₀=0). Implemented via **CDEQ** (Consistency Deep Equilibrium, Lin et al. 2026) — compresses the N-step FP iteration into a single forward. Produces ELF★ (self-conditioning-free, autonomous velocity).
- **Flow map distillation** (Eq. 27): learn a **two-time denoiser** δ_{s,t}(x) = x + (1−s)·v_{s,t}(x) satisfying (i) diagonal δ_{t,t} = D⋆_t (Eq. 25) and (ii) semigroup condition δ_{s,t} = γ·δ_{s,u} + (1−γ)·δ_{u,t}(X⋆_{s,u}) (Eq. 26). Produces FMLM★ (few-step flow map).
- **Offline vs online:** offline trains ELF★ then FMLM★ (two stages, 82h); online distills FMLM★ directly from the self-conditioned teacher using a fixed number of cold-start FPIs per training step (saturates ~9 FPIs, ~24h, 0.3× the cost).

### 1.5 Headline results (OpenWebText, GPT-2-Large tokenizer, 105M params)

FMLM★ achieves SOTA among entropy-preserving (≈5.44 nats) few-step baselines:

| Steps | Duo·DCD | MDLM·SDTT | FMLM | DFM·ESD | **FMLM★** |
|---|---|---|---|---|---|
| 1 (gPPL↓) | 5743 | 1261 | 168 | 5.33† | **112.52** |
| 2 (gPPL↓) | 891 | 877 | 133 | 109 | **94.74** |
| 4 (gPPL↓) | 251 | 340 | 111 | 77 | **75.22** |

†DFM·ESD 1-step collapses to entropy 0.26 (mode collapse — disqualified on the entropy-preservation criterion).

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface)

| FMLM mechanism | Shipped cousin | File / Plan |
|---|---|---|
| **Iterated fixed-point block** (the entire §3.2 mechanism — `z_{j+1} = D̂(x, z_j)`) | `LoopMode::WeightShared { loop_count }` — default-on, GOAT 8/8 | Plan 108, `crates/katgpt-types/src/lib.rs`, `forward_looped` |
| ODE-motivated iterated block (strictly more general than Picard FP) | `LoopMode::TrainingFree` + `TrainingFreeLoopConfig` (K-stage RK β=0.5) — default-on, GOAT 4/4 | Plan 136, `tf_loop` |
| **FP convergence halt** (Prop 3.5 iteration count) | `ResidualRelevanceScorer.is_converged` — default-on, GOAT 6/6 | Plan 085, `manifold_residual.rs` |
| **Self-conditioning** (the z = sg(D(x,0)) trick) | `self_cond_draft` — 2-pass self-conditioned speculative draft (pass 1 → estimate → pass 2 with SC) | Plan 222, `critical_interval_gate` + `self_cond_draft` feature |
| Self-conditioning in continuous flow LMs (distilled sampling techniques) | ELF modelless path — SDE noise injection, logit-normal schedule | Plan 079, Research 041 (RePlaid §4.2 self-conditioning draft-refine loop) |
| **Warm-start across flow/denoising steps** (Eq. 9 carry) | RCD Residual Context Diffusion — entropy-weighted residual embedding carry for "still masked" positions across D2F steps | Plan 258 (`rcd_residual`), `dllm.rs::denoise_loop_rcd` |
| Token-type-aware warm-start (3SR — the closest cousin to FMLM's warm-start) | D2F Three-State Warm-Start — unchanged-visible/still-masked/newly-revealed coefficients | Plan 291, Research 265 (CoFRe) |
| Contractivity / DEQ comparison / attractor FP basins | Attractor Models — implicit gradient barrier confines training to contractive regime ρ(J) < 1 | Research 035 |
| **Implicit FP RNN** (FP-halting design space closure) | "Implicit LMs are RNNs" — §3.5 modelless-unblock FAILS on all three paths; novelty gate Q1–Q4 all NO | Research 344 |
| Cubic fixed-point (Newton-Schulz orthogonalization) | 5-iteration cubic FP for Muon-family optimizer weight matrices — default-on, GOAT 25/25 | Plan 152, `newton_schulz.rs` |
| L2/KL residual fixed-point scoring | `deep_manifold` — ResidualRelevanceScorer — default-on, GOAT 6/6 | Plan 085 |
| Loop halting / gain-cost | FPRM (266), LoopCoder-V2 (282), ELT (273), Self-Advantage Gate (283), PathwayTracker (231), River-Valley Diagnostics (152) | — |

**The FP-iteration-on-a-denoiser/transformer design space is one of the most saturated in this codebase.** Research 344 (2026-06-30) just closed it with: *"the FP-halting + looped-block design space is one of the best-covered in the codebase"* and *"§3.5 modelless-unblock protocol fails on all three paths — the paper's RNN-equivalence is the absence of a dynamics in the modelless regime, not a correctable bias."* CoFRe (Research 265, 2026-06-18) reached the same conclusion for the masked-diffusion variant: *"~80% of the paper's inference surface is already shipped."*

### 2.2 The modelless delta (cold-start equivalence) — narrow, config-level

The ONE finding in FMLM not explicitly stated in CoFRe (R265) or Research 344:

> **Cold-start equivalence (§3.3, Fig. 3):** warm-start (carrying z across flow steps) is *not* a defining property of self-conditioned flows. Cold-start (z₀=0) with sufficient FPIs reaches the same gPPL–entropy frontier as warm-start. The flow-state / SC-state coupling in conventional SC generation (Eq. 9) is a byproduct of single-step warm-start approximation, not a structural feature.

**Why this is a config insight, not a primitive:**
1. It applies only to self-conditioned flow/masked-diffusion models — katgpt-rs's primary stack is autoregressive LLMs (D2F/`self_cond_draft` is opt-in research, not a hot path).
2. "Run more iterations from cold-start" is a knob on `LoopMode::WeightShared { loop_count }` + initialization, not a new mechanism.
3. It is already implied by the contractivity analysis (Props 3.3–3.5): convergence is independent of initialization under η < 1.
4. CoFRe's 3SR (Plan 291) already explored the warm-start design space on D2F and found it a narrow refinement of RCD (Plan 258).

**Where it COULD matter (narrow):** if D2F's `self_cond_draft` (Plan 222) were ever extended from 2-pass to N-iteration, the cold-start finding says the carry-across-denoising-steps (RCD/3SR) is optional — you can re-cold-start the SC state at each D2F step if you run enough inner iterations. But this is a config trade-off (more inner compute vs carry state), not a new capability.

### 2.3 Latent-space reframing (mandatory per skill §1 step 3) — no Super-GOAT angle

Re-casting the FP mechanism as a latent-to-latent op on the seven Super-GOAT factory modules:

- **(a) HLA per-NPC latent state:** running HLA's `evolve_hla` as a FP iteration = exactly what Plan 276 (AttractorKernel) tried and **null-resulted** (Research 344 §3.2: *"random-init attractors flip-flop, G2.1 coherence FAIL 569× flip-flops vs leaky"*, *"the paper's RNN-equivalence requires trained dynamics; per-NPC deliberation via FP would require a trained attractor per NPC, which doesn't scale to 10k NPCs"*). Dead end.
- **(b) `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs`:** already ships "coherence-driven re-estimation when coherence < tau_reest" — a *triggered* re-derivation, not a FP iteration. Different mechanism; no fusion gain.
- **(c) `cgsp_runtime/` curiosity:** not a FP-on-a-denoiser mechanism.
- **(d) LatCal fixed-point:** a *numeric format* for deterministic chain commitment (raw scalar bridge), NOT a functional FP iteration. Completely different sense of "fixed-point."
- **(e) NeuronShard consolidation:** sleep-cycle consolidation toward a stable shard is FP-*ish* but already shipped (Raven/δ-Mem) and is a different mechanism (experience replay averaging, not Picard iteration on a function).
- **(f) DEC operators:** the harmonic component (kernel of the Hodge Laplacian) is a FP of the Laplacian operator, but already shipped (`hodge_decompose`, `harmonic_projector`) and is a different mechanism (Helmholtz decomposition, not self-conditioning refinement).

**No Super-GOAT angle.** The FP-iteration-on-latent-state space is already explored (HLA attractor null result, latent_functor re-estimation, neuron-shard consolidation). FMLM's specific contribution (self-conditioning = FP) is about a training trick in continuous flow LMs that does not transfer to our latent kernels in a novel way.

### 2.4 → riir-train (out of scope, noted for completeness)

The following are unambiguously training-side and route to `riir-train/.research/` if pursued:

- **CDEQ fixed-point distillation** (Lin et al. 2026) — compresses the N-step FP iteration into a single forward via consistency distillation with Anderson acceleration (K=20 teacher iterations, history 3, mixing 0.9). Produces ELF★.
- **Two-time-denoiser semigroup distillation** (Eq. 27) — learns δ_{s,t} satisfying diagonal + semigroup conditions. Produces FMLM★. Offline (two-stage, 82h on 8×B200) or online (one-stage, ~24h, cold-start FPIs at each training step).
- **Online vs offline trade-off** — online saturates at ~9 FPIs, 0.3× the cost of offline, near-matching quality.
- **Self-conditioning guidance weight** w ∈ [0.5, 5.0] log-uniform sampling during distillation.

If riir-train ever produces a distilled few-step flow-map artifact, the *inference* path is already shipped here (`LoopMode::WeightShared` + `ResidualRelevanceScorer.is_converged` halt + Euler/γ-sampling). The artifact would load via the existing model loader; no new katgpt-rs primitive needed.

---

## 3. Verdict

**Gain** (leaning Pass — heavily saturated prior art).

One-line reasoning: FMLM's entire inference-time surface (iterated FP block, convergence halt, self-conditioning, warm-start carry) is ~95% already shipped (`LoopMode::WeightShared` = the FP block; `ResidualRelevanceScorer.is_converged` = the halt; Plan 222 `self_cond_draft` = self-conditioning; Plan 258 RCD + Plan 291 3SR = warm-start carry); the only modelless delta is the cold-start equivalence finding (a config insight on an opt-in path, not a primitive); the training recipes (CDEQ, semigroup flow-map distillation, FMLM★) → riir-train.

**Why not Super-GOAT:** fails novelty gate Q1 (saturated prior art — CoFRe R265 is a direct sibling paper; Research 344 just closed the FP-for-LM design space with "Q1–Q4 all NO"), Q2 (no new capability class — the FP iteration IS what `LoopMode::WeightShared` already does), Q3 (no product selling point — katgpt-rs is AR-LLM-focused; continuous flow LMs are not our primary stack; the FMLM★ distillation is training), Q4 (no force multiplier — the FP family already multiplies the reasoning/freeze-thaw pillars; this paper adds no new connection).

**Why not GOAT:** no provable latency/quality/security gain over the existing shipped approach on a modelless path. Every empirical result in the paper is a *training* result (CDEQ distillation, semigroup distillation, 8×B200 × 5 epochs). The cold-start equivalence is an empirical finding about an existing mechanism, not a gain over a shipped primitive.

**Why not Pass:** the paper is high-profile (KAIST/CMU, SOTA on OpenWebText) and a future agent might re-evaluate it; the "self-conditioning = fixed-point" theorem (Props 3.1–3.5) is worth recording as already-covered territory; the training recipes need explicit → riir-train routing; and the cold-start equivalence, while narrow, IS a genuinely transferable finding (warm-start is optional under sufficient iterations) worth documenting. Matches the CoFRe (R265) and Research 344 pattern.

| Tier | Criteria | Routing |
|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + force multiplier | — |
| GOAT | Provable gain over existing approach, new default candidate | — |
| **Gain** ← | Incremental, useful, not headline-worthy | **Note only. No plan, no guide, no open primitive.** |
| Pass | Not relevant / training-only | — |

### 3.1 §3.5 modelless-unblock protocol — N/A (no gate to unblock)

There is no failing GOAT gate to defer. The paper's value is overwhelmingly training-side (CDEQ, semigroup distillation, FMLM★ artifact). The modelless inference path (iterate a denoiser to a FP with a residual halt) is already shipped. No §3.5 check required — there is no "this needs training" deferral to challenge.

### 3.2 MOAT gate (§1.6) — katgpt-rs domain

- **In scope?** Borderline. The paper is about continuous flow language models; katgpt-rs is AR-LLM-focused. The modelless primitives it touches (FP iteration, self-conditioning, warm-start) are in the transformer-stack / DEC substrate domain.
- **Strengthens moat?** No. The FP-halting family already saturates this slot of the transformer stack; FMLM adds no new moat to the public engine.
- **Verdict:** Neutral Gain. Document for completeness and re-evaluation prevention. No feature flag, no benchmark, no promote/demote tracking needed.

---

## 4. What this note prevents (canonical failure modes averted)

1. **False Super-GOAT on "self-conditioning = fixed-point is a novel insight."** It is a *theorem* about an existing mechanism, not a new primitive. The FP iteration pattern is shipped as `LoopMode::WeightShared` (Plan 108) and the design space is closed by Research 344.

2. **False Super-GOAT on "cold-start sampling is a new inference primitive."** It is a config finding (run more iterations from z₀=0) on an opt-in path (D2F / self-conditioned flow LMs). CoFRe (R265) already explored the warm-start variant (3SR, Plan 291) and found it a narrow refinement of RCD.

3. **Mis-routing the paper's value to katgpt-rs.** The paper's value is overwhelmingly training-side (CDEQ fixed-point distillation, two-time-denoiser semigroup distillation, FMLM★ artifact, 8×B200 GPU-hours). Routing any of it here would violate the modelless-first mandate (constraint #1).

4. **Re-evaluation by a future agent.** This note + the prior-art table (§2.1) + the connection to CoFRe (R265) and Research 344 should prevent any future session from re-running the novelty gate on this paper or its close cousins (ELF, LangFlow, MDLM, DFM, FMLM, CoFRe, FP-MGM, DEQ, Attractor Models).

5. **Duplicate distillation.** The continuous-flow-LM line (ELF/MDLM/DFM/FMLM/CoFRe) is a single design space; CoFRe (R265) and this note cover it. A future paper in this family (e.g., a new flow-map distillation variant) should be checked against R265 + R366 before any new note is created.

---

## 5. Action items

- [x] **Document the paper** (this note) — record findings, prior-art surface, cold-start insight, → riir-train routing.
- [-] **No plan in katgpt-rs.** The modelless delta (cold-start equivalence) is a config insight on an opt-in path; no new primitive warranted.
- [-] **No Super-GOAT guide in riir-ai / riir-chain / riir-neuron-db.** Latent-space reframing (§2.3) finds no novel latent-op angle; the HLA FP angle is null-resulted (Plan 276 AttractorKernel).
- [-] **→ riir-train** (if pursued): CDEQ fixed-point distillation, two-time-denoiser semigroup flow-map distillation, online vs offline FMLM★ recipe, self-conditioning guidance weight schedule. Out of scope for this workflow; noted for completeness.
- [-] **Track as context, not action:** the cold-start equivalence (Fig. 3) is empirical validation that warm-start is optional under sufficient iterations — consistent with the contractivity analysis (Props 3.3–3.5) and with our shipped `LoopMode::WeightShared` + `ResidualRelevanceScorer.is_converged`. No code change.

---

## TL;DR

**Verdict: Gain** (leaning Pass). Yoo et al. (KAIST/UvA/CMU, arXiv:2607.00714) prove that self-conditioning in continuous flow language models implicitly learns a fixed-point Picard iteration `z_{j+1} = D̂_t(x, z_j)` converging exponentially to the Bayes-optimal denoiser under contractivity (Props 3.1–3.5), formalize this as a "fixed-point flow" with a valid autonomous velocity `b⋆(x) = (D⋆(x)−x)/(1−t)`, and distill it into FMLM★ (SOTA few-step flow map on OpenWebText) via CDEQ fixed-point distillation + two-time-denoiser semigroup distillation. **~95% of the inference surface is already shipped** in katgpt-rs: `LoopMode::WeightShared` (Plan 108) *is* the iterated FP block; `ResidualRelevanceScorer.is_converged` (Plan 085) *is* the halt; Plan 222 `self_cond_draft` *is* self-conditioning; Plan 258 RCD + Plan 291 3SR *are* the warm-start carry. CoFRe (Research 265, arXiv:2605.31215) is a **direct sibling paper** already distilled as Gain. Research 344 (Implicit FP RNN, 2026-06-30) **just closed this design space** with "novelty gate Q1–Q4 all NO" and "§3.5 modelless-unblock fails on all three paths." The only modelless delta is the **cold-start equivalence** finding (warm-start across flow steps is optional — cold-start z₀=0 with enough iterations matches; their Fig. 3), which is a config insight on an opt-in path, not a primitive. The training recipes (CDEQ, semigroup flow-map distillation, FMLM★) → **riir-train**. Latent-space reframing (§2.3) confirms no Super-GOAT angle — the HLA FP angle is null-resulted (Plan 276 AttractorKernel: 569× flip-flops at random init). **No plan, no Super-GOAT guide, no open primitive created in this session.** The note exists to document the saturated prior art, route training to riir-train, and prevent re-evaluation.
