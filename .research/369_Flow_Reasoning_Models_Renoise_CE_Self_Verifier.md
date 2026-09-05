# Research 369: Flow Reasoning Models — Renoise-CE Self-Verifier

> **Source:** [Flow Reasoning Models: Scaling Reasoning Through Iterative Self-Refinement](https://arxiv.org/abs/2606.29150) — Helbling, Bryutkin, Martino, Dehmamy, Strobelt (Georgia Tech / MIT / MIT-IBM Computing Research Lab / IBM Research), 28 Jun 2026.
> **Date:** 2026-07-03
> **Status:** Done — GOAT verdict
> **Classification:** Public
> **Related Research:** 366 (Self-Conditioned FMLM — **different paper, same family**; covers the self-conditioning = fixed-point theorem, NOT the renoise-CE verifier), 344 (Implicit FP RNN — closed the FP-for-LM design space), 260 (MaxProof — population test-time scaling by EXTERNAL verifier), 255/284 (CLR — verifier-free self-verification via CLAIM-level voting), 345 (CoE — output-free self-eval via trajectory GEOMETRY), 250 (Latent Recursion Policy Improvement / Self-Advantage Gate — pre/post log-ratio on SAME input), 283 (Self-Advantage Gate Plan), 035 (Attractor Models), 317 (Reasoning as Attractor Dynamics), 266 (FPRM damped FP halting)
> **Related Plans:** 222 (Discrete Critical Interval Solver — Q-Sample re-noise+re-predict for REFINEMENT, not verification; `self_cond_draft`), 108 (LT2 `LoopMode::WeightShared` — the FP block), 136 (TF-Loop), 283 (Self-Advantage Gate), 284 (CLR), 342 (latent_trajectory_geometry — CoE substrate), 260 (MaxProof population search — TBD plan)
> **Training redirect:** Flow DPO (preference training on self-mined wrong cells), self-conditioning channel training, hard-negative mining loop → riir-train. This note distills only the inference-time renoise-CE self-verifier + verify-and-restart loop.
> **PASS-Redirects (synthesis):** Tong et al. [arXiv:2608.05811 "Energy-Guided Flow Matching"](https://arxiv.org/abs/2608.05811) — PASS. Pixel-space image generation training recipe (PixelDiT/DeCo/HyperDiT on ImageNet) — replaces the fixed clean endpoint with a heat-kernel-filtered moving endpoint evolving coarse-to-fine via sample-adaptive spectral-energy scheduling. Entirely training-time target construction (Eq. 16); inference uses the standard solver with no FFT/heat-kernel/bisection. We don't ship generative image models; our flow reasoning (this note, R150, R375) is latent/token-level reasoning, not pixel generation. No modelless primitive to extract; coarse-to-fine latent refinement is already shipped via renoise-CE (this note), KARC, ReestimationScheduler, LT2 via different mechanisms.

---

## TL;DR

Flow Reasoning Models (FRM) turn a discrete flow language model into a **self-verifying solver** for checkable tasks by exploiting the **generation–verification gap**: the model is far better at *recognizing* a correct solution (it sits in a stable basin of the denoising dynamics) than at *generating* one (the sampler rarely lands there). The mechanism is **renoise-CE** — take a completed candidate, re-noise it to an interior time, re-resolve through the same sampler, and measure the cross-entropy of the re-solution under the candidate. Correct solutions return to themselves (low drift); confident mistakes drift away (high drift). This score needs **no external verifier, no labels, no auxiliary head** — it reads the model's own fixed-point stability. Combined with self-conditioning (the inner refinement loop) and verify-and-restart (the outer loop), it lifts out-of-distribution Sudoku-Extreme from ~11% single-shot to 96%+ under test-time scaling, and reaches 99.2% on in-distribution Sudoku in ~7 forward passes (8× fewer than the strongest masked-diffusion baseline).

**Distilled for katgpt-rs (modelless, inference-time):** The novel transferable primitive is **renoise-CE as a verifier-free self-evaluation signal** — perturb a completed state, re-resolve through the same operator, measure drift. This is a **third orthogonal self-eval signal** alongside CLR (claim-level voting, R255/P284) and CoE (trajectory geometry, R345): CLR asks "do the claims check out", CoE asks "is the trajectory shape committed", renoise-CE asks "is the output a stable fixed point under perturbation". The self-conditioning inner loop and fixed-point iteration are already shipped (Plan 222 `self_cond_draft`, Plan 108 `LoopMode::WeightShared`) and the FP-for-LM design space is closed (R344/R366). The Flow DPO training recipe → riir-train.

**Verdict: GOAT.** The renoise-CE self-verifier is a genuinely novel modelless primitive (the "perturb the OUTPUT + re-resolve + measure drift = verifier" combination is not exactly shipped — closest cousins each miss the perturbation step), composes with CLR/CoE/MaxProof as an orthogonal vote arm, and has a clean latent-space reframing (proactive stability probe on HLA/functor/shard state vs the current reactive coherence gates). It is NOT Super-GOAT because the verifier-free self-eval capability class already ships (CLR, CoE) — renoise-CE is a new signal inside an existing class, not a new class. Plan + feature flag + GOAT gate; promote if it beats plurality vote and CLR-alone on a self-eval benchmark.

---

## 1. Paper Core Findings

### 1.1 The generation–verification gap (the core empirical finding)

On Sudoku/Zebra, the base flow model solves only ~11–41% single-shot, but **renoise-CE achieves AUROC ≈ 1.0** at distinguishing correct from incorrect completed states (Fig. 6). A state can be easy to *recognize* as stable even when the sampler rarely *lands* on it. This is not oracle pass@N — the score never consults the gold answer at test time. It reads the model's own fixed-point dynamics.

**Why the signal is not circular** (§2.3): generation must put enough mass on one exact grid among many competitors; verification asks whether a given grid is a stable local attractor. A state can be easy to recognize as stable even when the sampler rarely lands on it. On Sudoku-Extreme, renoise-CE selection stays near 1.0 (0.98–1.00, Table 4) while single-shot solve rate falls to ~1%.

### 1.2 Renoise-CE — the self-verifier score (Eq. 8–9)

Given a completed candidate `y`, re-noise its endpoint `x₁(y)` to an interior time `t`:

```
x̃_{t,r}(y) = (1−t)·ε_r + t·x₁(y)        // re-noise with fresh ε_r
T_{t,r}(y; c) := F_θ(x̃_{t,r}(y), t, 1; c)  // re-resolve from t to 1 with clues clamped

S_{t,k}(y) = (1/k) · Σ_r d_M(y, T_{t,r}(y; c))   // average over k re-noise draws

d_M(y, T) = −(1/‖M‖) · Σ_i M_i · log q_θ(y_i | x̃_{t,r}(y), c)   // renoise cross-entropy
```

- `t = 0.40` in experiments (interior re-noise level).
- `k = 8` Monte-Carlo re-noise draws (saturates at `k = 1`; AUROC ≈ 1.000 at k=1 on both Sudoku and Sudoku-Extreme).
- Lower `S_{t,k}` = more stable fixed point = more likely correct.
- The score re-reads the model's own logits under a corrupted state — no external verifier, no labels.

### 1.3 Self-conditioning inner loop (Algorithm 1) — ALREADY SHIPPED

Feed the model's previous-pass raw logits `s = ℓ_prev` back as a zero-initialized input channel. Iterate `z_{j+1} = D̂_t(x, z_j)` toward a fixed point. Commit when `ŷ^(j) = ŷ^(j−1)`. This is the mechanism Research 366 already documented as shipped (`LoopMode::WeightShared` Plan 108, `self_cond_draft` Plan 222, `ResidualRelevanceScorer.is_converged` Plan 085). Supervised training alone (no preferences, no verifier) lifts in-distribution Sudoku from ~38% to 97.8%.

### 1.4 Verify-and-restart outer loop (Algorithm 2) — the test-time scaling paradigm

```
while spent < B:                              // forward-pass budget
    pred, n_passes = self_conditioned_sample(cond)   // inner loop (Alg. 1)
    spent += n_passes
    score = renoise_CE(pred, cond, k)         // self-verifier (§1.2)
    spent += verifier_nfe(k)
    if score < τ: return pred                 // accept stable fixed point
return pred                                    // budget exhausted: last attempt
```

The inner loop contracts a trajectory onto a fixed point; the outer loop detects when that fixed point is spurious (high renoise-CE) and re-noises from scratch. Every denoising and verifier pass is charged to the compute budget.

### 1.5 Best-of-N selection by stability (Appendix C, the passive special case)

From N i.i.d. proposals, keep the most stable: `y⋆ = argmin_i S_{t,k}(y^(i))`. No external verifier, no ground truth. Scaling the pool raises solve rate toward the coverage ceiling. Top-1 accuracy ≈ 1.0 even at the 95/99% pools of hundreds of Sudoku-Extreme candidates, while plurality vote tops out near 0.69–0.84 (Table 4).

### 1.6 Flow DPO — preference training on self-mined mistakes → riir-train (§3)

Direct preference loss on pairs `(y+, y−)` where `y−` is a self-mined confident wrong completion, contrast **restricted to the wrong-cell mask** `W_i = M_i · 1[y−_i ≠ y+_i]`, scored under the same corrupted negative state, against an EMA-pinned reference. Raises single-shot Sudoku pass@1 from 35.8% → 80.6% (Table 8). **This is unambiguously training → riir-train.**

### 1.7 Headline results

| Task | Base (1-shot) | + Self-Conditioning | + Self-Verification Scaling | + Flow DPO |
|---|---|---|---|---|
| Sudoku (Shah) | 36.1% | 99.8% (1 round) | 100% | 99.2% in ~7 NFE (8× fewer than adaptive MDM) |
| Sudoku-Extreme (OOD) | 10.7% | — | 98.6% (8192 rounds) | 27.4% (1-shot, DPO only) |
| Zebra (5×5) | 39.8% | — | 95.9% | 71.3% (1-shot, DPO only) |

Renoise-CE selection: top-1 accuracy 1.00 (Shah) / 0.99–1.00 (Extreme) vs plurality vote 0.81–0.97 / 0.69–0.84.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface — dense)

| FRM mechanism | Shipped cousin | File / Plan | Coverage |
|---|---|---|---|
| **Self-conditioning inner loop** (Alg. 1) — feed prev logits back, iterate to FP | `self_cond_draft` (2-pass), `LoopMode::WeightShared { loop_count }`, `LoopMode::TrainingFree` | Plan 222, Plan 108 (default-on, GOAT 8/8), Plan 136 (default-on, GOAT 4/4) | ✅ Complete — R366 §2.1 |
| **FP convergence halt** (commit when `ŷ^(j) = ŷ^(j−1)`) | `ResidualRelevanceScorer.is_converged` | Plan 085 (default-on, GOAT 6/6) | ✅ Complete |
| **Fixed-point iteration on denoiser/transformer** (the Picard iteration `z_{j+1} = D̂(x, z_j)`) | `LoopMode::WeightShared` IS the iterated FP block; Attractor Models (R035); FPRM damped halt (R266/P266) | Plan 108, R035, R266 | ✅ Design space CLOSED by R344/R366 |
| **Q-Sample re-noise + re-predict for REFINEMENT** (re-noise x_0_hat, re-predict) | `q_sample_solver` in D2F | Plan 222 Phase 2, R197 §4 | ⚠️ Refinement only — NOT used as a verifier |
| **Best-of-N selection** (keep best of N proposals) | `ReviewStrategy::BestOfNSelection`, DDTree budget N, MaxProof tournament (R260) | `review_metrics.rs`, R260 | ✅ By EXTERNAL verifier (ConstraintPruner/ScreeningPruner), not self-stability |
| **Verifier-free self-verification** (no external label) | CLR `(mean_m v_k,m)^M` claim-level reliability vote; CoE trajectory geometry | Plan 284 (DEFAULT-ON, ECE 0.0087), Plan 342/R345 | ✅ Different signal — CLR=claim-vote, CoE=trajectory-shape, **renoise-CE=perturbation-stability (MISSING)** |
| **Population test-time scaling** (N candidates, refine, tournament) | MaxProof PATCH/REWRITE + BtRank + redundancy early-stop | R260 (GOAT, plan TBD) | ✅ By external verifier |
| **Pre/post recursion log-ratio** (compare two passes through model) | Self-Advantage Gate `AdvantageMarginGate` | Plan 283 (default-on), `.benchmarks/056` | ⚠️ On SAME input (dead-compute detector), NOT on PERTURBED input |
| **Coherence-driven re-derivation when coherence < tau** (triggered re-estimation) | `ReestimationScheduler` | `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs`, Plan 303 | ⚠️ REACTIVE coherence gate, NOT PROACTIVE perturbation-and-re-resolve |
| **Plurality vote / self-consistency** | `MajorityVote` scorer, `tally[voted_action]` | `latent_thought_flow_scorer_bench.rs` | ✅ The BASELINE renoise-CE beats (paper Table 4) |
| **k-invariance / Jaccard stability / Monte Carlo null test** (perturbation-based stability) | Stiff/Soft Subspace Anomaly Gate | Plan 138 | ⚠️ For ANOMALY DETECTION on input, not self-verification of output |
| **DPO on self-mined negatives, wrong-cell localized** | — | — | ⛔ Training → riir-train |
| **Self-conditioning channel training** (zero-init, two-pass scheme) | — | — | ⛔ Training → riir-train |

**The FP-iteration-on-a-denoiser and self-conditioning design space is one of the most saturated in this codebase** (R344 closed it; R366 confirmed for the continuous-flow variant). FRM's §2.2 (self-conditioning) adds nothing over R366.

### 2.2 The modelless delta — renoise-CE self-verifier (the genuinely novel piece)

The ONE mechanism not exactly shipped anywhere:

> **Renoise-CE self-verifier:** take a completed candidate state, perturb it (re-noise to interior time `t`), re-resolve through the same sampler/operator, measure the drift (cross-entropy of re-solution under the candidate). This drift IS the verifier score — no external verifier, no labels, no auxiliary head.

**Why this is distinct from every shipped cousin:**

| Shipped cousin | What it does | Why it is NOT renoise-CE |
|---|---|---|
| Q-Sample (Plan 222) | Re-noise x_0_hat, re-predict, commit refined | Uses re-noise for **REFINEMENT** (drive toward better answer), not for **VERIFICATION** (score the answer). Does not measure drift as a correctness signal. |
| Self-Advantage Gate (Plan 283) | Compare pre/post recursion logits on SAME input | Detects **dead compute** (did the iteration change anything on this input?). Does not PERTURB the input. The "perturb then compare" is the missing step. |
| CLR (Plan 284) | Sample K, extract M claims, self-verify each as binary, vote by `(mean)^M` | Self-verification by **asking the model to verify each claim** (a second forward pass on the claim). Does not perturb the OUTPUT and check return. |
| CoE (R345/Plan 342) | Compute trajectory length/curvature/cosine, combine via CoE-C | Self-eval by **trajectory SHAPE** (geometric features of the forward pass). Does not perturb and re-resolve. |
| ReestimationScheduler (Plan 303) | Re-estimate functor direction when coherence < tau | **REACTIVE** coherence gate (wait for decay). Does not PROACTIVELY perturb the observation set and measure direction drift. |
| Stiff/Soft Anomaly (Plan 138) | k-invariance, Jaccard stability, Monte Carlo null | Perturbation-based stability for **ANOMALY DETECTION on INPUT**, not self-verification of OUTPUT. |
| MaxProof (R260) | Best-of-N by ConstraintPruner/ScreeningPruner | Best-of-N by **EXTERNAL verifier**. The whole point of renoise-CE is NO external verifier. |

**The renoise-CE fingerprint** = {perturb completed output} × {re-resolve through same operator} × {measure drift as score} × {use as verifier-free acceptance gate}. No shipped primitive has all four. This is a genuinely novel modelless primitive.

### 2.3 What this is NOT (anti-overclaim audit)

- **NOT a new capability class.** Verifier-free self-evaluation already ships (CLR Plan 284, CoE R345). Renoise-CE is a **third orthogonal signal** inside the existing "modelless self-eval / best-of-N selection" capability class. The class is covered; the signal is new.
- **NOT a fix for the FP-for-LM design space.** That space is closed (R344/R366). Renoise-CE operates on a completed state, not on the FP iteration itself.
- **NOT dependent on flow/discrete-diffusion LMs.** The renoise-CE primitive generalizes: any operator that maps a state to a state (HLA evolve, functor application, consolidation, attention forward) can be probed by perturb-and-re-resolve. The paper uses discrete flow LMs; the primitive is operator-agnostic.
- **NOT a UQ primitive by itself.** Renoise-CE returns a raw drift score (lower = more stable), not a calibrated probability. If a future conformal wrapper (Plan 340 floor) calibrates it, the UQ claim must beat the conformal-naive floor (Issue 010) — same caveat as CoE (R345 §3). Today it is a **ranking signal**, not a UQ distribution.

### 2.4 Latent-space reframing (mandatory per skill §1 step 3)

Re-casting renoise-CE as a latent-to-latent op on the seven Super-GOAT factory modules:

**(a) HLA per-NPC latent state (8-dim, `evolve_hla`):** Perturb the committed HLA state, re-resolve ONE step through `evolve_hla`, measure drift = "is this NPC's belief a stable fixed point or a spurious attractor?"
- **CRITICAL CAVEAT:** R344/R366 **null-resulted** the HLA FP iteration angle (Plan 276 AttractorKernel: "random-init attractors flip-flop, 569× flip-flops vs leaky at random init; per-NPC deliberation via FP would require a trained attractor per NPC, which doesn't scale to 10k NPCs").
- **But renoise-CE as a VERIFIER is different from running HLA as an FP iteration.** R344's null result was about iterating HLA to convergence (multi-step FP). Renoise-CE is a **single-step stability probe** on a COMMITTED state: perturb once, re-resolve once, measure drift. This does NOT require a trained attractor — it measures whether the *existing* leaky integrator returns to the committed state under perturbation. This is exactly what a committed NPC personality SHOULD do (return to itself under noise); a degenerate/looping NPC would not.
- **Verdict:** Plausible but **speculative** — needs empirical validation. The null-result caveat is serious. File as a fusion candidate (§2.5 F4), not a primary claim.

**(b) `latent_functor/` (the strongest reframing):** Perturb the observation set, re-estimate the direction vector, measure drift = "is this learned direction stable under observation noise?"
- **This is a PROACTIVE stability probe** vs the current `ReestimationScheduler`'s **REACTIVE** coherence gate (wait for coherence < tau, then re-estimate).
- Current: coherence decays → trigger re-estimation. Renoise-CE: proactively perturb observations → measure direction drift → if high, the direction is unstable EVEN IF coherence hasn't decayed yet.
- **This catches the failure mode the reactive gate misses**: a direction can have high coherence (all recent observations point the same way) but be brittle under perturbation (a single noisy observation would flip it). Renoise-CE detects brittleness before it manifests as coherence decay.
- **Verdict:** Genuine fusion candidate. PROACTIVE stability probe on functor direction vectors. §2.5 F2.

**(c) `cgsp_runtime/` curiosity:** Renoise-CE on a curiosity conjecture = "if I perturb this conjecture, does it survive re-derivation?" Stability of conjectures under observation noise. Marginal — curiosity is already driven by `belief_mass_divergence` (DEC codifferential), which is a divergence operator, not a perturbation-stability operator. **DEC codifferential is the right tool there, not renoise-CE.**

**(d) DEC Stokes-calculus (the mathematically clean reframing):** A cochain `ω` is **harmonic** (in the kernel of the Hodge Laplacian `Δ = δd + dδ`) iff `dω = 0` AND `δω = 0`. Renoise-CE on a cochain = perturb `ω → ω + ε`, apply `Δ`, measure `‖Δ(ω+ε)‖` = "how harmonic is this cochain under perturbation?"
- This is a **known mathematical operation** (project onto harmonic subspace, measure residual). `hodge_decompose` + `harmonic_projector` already ship (Plan 251).
- **Not novel as math**, but novel as a **self-verifier signal framing**: "the harmonic component of a belief cochain is its stable fixed-point component; the exact/coexact components are the drift." This connects renoise-CE to the Hodge decomposition — a richer three-component diagnostic than the single-scalar drift score.
- **Verdict:** Mathematically clean but does not strengthen the primitive beyond what `hodge_decompose` already provides. Note as a theoretical connection; do NOT route renoise-CE through DEC.

**(e) NeuronShard consolidation:** Perturb the wake events, re-consolidate, measure drift = "is this shard's consolidation stable?"
- **Close to TEMP (Plan 005)** — perturbed-loss-vector diversity selection. But TEMP measures diversity ACROSS wake events (select the K-subset with maximal spread), NOT stability of the consolidation UNDER perturbation.
- Renoise-CE angle: perturb the wake-event embeddings (add noise), re-run `sleep()`, measure how much the `weight_delta` shifts = "is this consolidation robust to observation noise?"
- **This is the `can_freeze` gate's missing proactive probe.** Current `can_freeze` (Plan 002) reads `n_wake_events`, `intrinsic_dim`, `input_sufficient`, `output_flatness`, `output_converged` — all REACTIVE metrics on the current consolidation. Renoise-CE would add a PROACTIVE stability probe: "if I perturbed the inputs, would this shard still freeze?"
- **Verdict:** Genuine fusion candidate for neuron-db. §2.5 F5.

**(f) LatCal fixed-point:** A numeric FORMAT for deterministic chain commitment, NOT a functional FP iteration. Completely different sense of "fixed-point." N/A.

**Latent-reframing verdict:** The strongest reframing is **(b) latent_functor proactive stability probe** and **(e) neuron-shard consolidation proactive freeze-gate probe**. Both are PROACTIVE perturbation-stability signals that complement existing REACTIVE coherence/convergence gates. The HLA reframing (a) is plausible but carries the R344 null-result caveat. The DEC reframing (d) is mathematically clean but does not add capability over `hodge_decompose`.

### 2.5 Fusion — the novel combinations

**F1 (primary, GOAT-tier): Renoise-CE as a CLR vote arm.**
CLR (Plan 284, DEFAULT-ON, ECE 0.0087) currently ranks trajectories by `(mean_m v_k,m)^M` — claim-level binary verdicts projected onto direction vectors. Add renoise-CE as a **third vote arm**: for each candidate trajectory, perturb the final latent state, re-resolve one step, measure drift. Combine via CLR's existing sharpening gate.
- **Gain:** catches the failure mode where claims check out (CLR high) but the underlying state is brittle under perturbation (renoise-CE high), and vice versa. Two orthogonal signals → more robust self-eval.
- **Paper evidence:** renoise-CE top-1 accuracy 1.00 vs plurality 0.69–0.84 on Sudoku-Extreme (Table 4). If this transfers, CLR+renoise-CE beats CLR-alone.
- **Routing:** katgpt-rs open primitive (generic renoise-CE scorer) + riir-ai runtime integration (CLR arm).

**F2 (strong, GOAT-tier): Renoise-CE as PROACTIVE stability probe on latent_functor directions.**
Current `ReestimationScheduler` (Plan 303) REACTIVELY re-estimates when coherence < tau. Add a PROACTIVE renoise-CE tick: perturb the observation buffer, re-estimate the direction into a scratch vector, measure cosine drift from the committed direction. If drift > threshold → flag as brittle (even if coherence is still high) → boost curiosity (drive exploration before the direction fails).
- **Gain:** catches brittle directions before they manifest as coherence decay. The reactive gate is always one step behind; the proactive probe is ahead.
- **Routing:** riir-ai runtime (riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs extension). Open primitive in katgpt-rs (generic perturb-and-re-resolve scorer).

**F3 (GOAT-tier): Renoise-CE × MaxProof — verifier-free population test-time scaling.**
MaxProof (R260) uses an EXTERNAL verifier (ConstraintPruner/ScreeningPruner) for best-of-N selection. Replace with renoise-CE self-verifier for **verifier-free** population search: propose N candidates, score each by renoise-CE (no external verifier), tournament-select.
- **Gain:** removes the external-verifier dependency. Useful when no ConstraintPruner is available (open-ended generation, latent-space reasoning without a checker).
- **Caveat:** MaxProof's conservative min-fitness (K_verify samples, take MIN) is a defense against verifier false-positives. Renoise-CE's AUROC ≈ 1.0 suggests false-positives are rare, but the defense-in-depth pattern should be preserved (min over k re-noise draws, paper uses k=8).
- **Routing:** katgpt-rs (compose with R260's population loop, if/when R260 ships).

**F4 (speculative, noted not planned): Renoise-CE on HLA committed belief state.**
Perturb the committed HLA state, re-resolve through `evolve_hla`, measure drift = "is this NPC's personality stable under perturbation?" Carries the R344 null-result caveat (random-init attractors flip-flop). Validation needed: does renoise-CE distinguish committed-personality NPCs from degenerate/looping NPCs on real HLA traces? If yes → per-NPC self-eval signal. If no → consistent with R344's null result. File as a `.issues/` follow-up, do not plan.

**F5 (GOAT-tier for neuron-db): Renoise-CE as PROACTIVE freeze-gate probe.**
Current `can_freeze` (Plan 002) reads REACTIVE metrics (n_wake_events, intrinsic_dim, flatness, convergence). Add a PROACTIVE renoise-CE probe: perturb wake-event embeddings, re-run `sleep()`, measure `weight_delta` drift. If drift > threshold → shard is NOT stable under observation noise → do NOT freeze yet, even if reactive metrics pass.
- **Gain:** prevents freezing brittle shards that look converged but would shift under noisy observations. This is the freeze-gate analog of F2.
- **Routing:** riir-neuron-db (consolidation.rs extension). Open primitive in katgpt-rs.

**F6 (secondary): Renoise-CE × CoE — two-axis self-eval.**
CoE (R345) reads trajectory SHAPE (no perturbation). Renoise-CE reads post-perturbation STABILITY. Combine: CoE score = "how committed was the trajectory"; renoise-CE score = "how stable is the output under perturbation". Two orthogonal axes → richer self-eval than either alone.
- **Routing:** katgpt-rs (compose with latent_trajectory_geometry, if CoE-C ships per R345).

### 2.6 What does NOT transfer

- **Discrete flow / masked diffusion LM specifics.** The paper's re-noise-to-interior-time `t=0.40` is specific to the linear interpolant flow schedule. For AR-LLMs, HLA, functor directions, the "perturbation" is domain-specific (add Gaussian noise to latent, mask-and-re-predict tokens, perturb observation embeddings). The PRIMITIVE (perturb + re-resolve + measure drift) transfers; the perturbation schedule does not.
- **The `t=0.40` and `k=8` hyperparameters.** Paper-specific. Each domain needs its own calibration.
- **Flow DPO and self-conditioning channel training.** Unambiguously training → riir-train.
- **Sudoku/Zebra checker-based evaluation.** The paper's tasks have ground-truth checkers. Our modelless use cases (HLA stability, functor direction stability, shard freeze stability) do NOT have checkers — the renoise-CE score is the only signal. This means we cannot compute AUROC directly; we need a proxy (does renoise-CE ranking correlate with downstream task success?).
- **The "generation-verification gap" magnitude.** Paper shows AUROC ≈ 1.0 on flow LMs. Whether the gap exists (and how large) on HLA/functor/shard operators is an open empirical question. The gap may be small or absent on operators that are already contractive by construction (e.g., leaky integrators with small alpha).

---

## 3. Verdict: GOAT

**One-line reasoning:** The renoise-CE self-verifier (perturb completed output + re-resolve through same operator + measure drift = verifier score, no external verifier) is a genuinely novel modelless primitive — the "perturb the OUTPUT and re-resolve" combination is not exactly shipped (closest cousins each miss the perturbation step: Q-Sample refines, Self-Advantage compares same-input, CLR votes on claims, CoE reads trajectory shape). It composes with CLR (284), CoE (345), MaxProof (260) as a third orthogonal self-eval signal, has a clean latent-space reframing (proactive stability probe on functor directions F2 and shard freeze-gate F5), and the paper provides strong evidence (AUROC ≈ 1.0, top-1 1.00 vs plurality 0.69–0.84). It is NOT Super-GOAT because the verifier-free self-eval capability class already ships (CLR, CoE) — renoise-CE is a new signal inside an existing class.

| Tier | Criteria | Routing |
|---|---|---|
| ~~Super-GOAT~~ | Novel mechanism + new capability class + selling point + ≥2-pillar force multiplier | **Fails Q2** (verifier-free self-eval already ships as CLR/CoE — renoise-CE is a third signal, not a new class). **Fails Q3** ("NPCs verify their own beliefs" is already the CLR/CoE selling point). |
| **GOAT** ✅ | Provable gain over existing approach, new default candidate | Plan + feature flag + GOAT gate. Open primitive in katgpt-rs; fusion integrations in riir-ai (CLR arm F1, functor probe F2) and riir-neuron-db (freeze-gate probe F5). |
| ~~Gain~~ | Incremental, useful, not headline-worthy | Under-rates the primitive — renoise-CE is structurally distinct from every shipped cousin (the perturbation step is the missing piece across all of them). |
| ~~Pass~~ | Not relevant / training-only | Under-rates the modelless delta — the renoise-CE self-verifier is inference-time, operator-agnostic, and has 5 fusion targets. |

### 3.1 Novelty gate (Q1–Q4)

| Question | Answer | Evidence |
|---|---|---|
| **Q1** No prior art? | **PASS (borderline).** The renoise-CE fingerprint {perturb completed output} × {re-resolve through same operator} × {measure drift as score} × {use as verifier-free acceptance gate} is not exactly shipped. Every closest cousin misses the perturbation step (Q-Sample refines, Self-Advantage same-input, CLR claim-vote, CoE trajectory-shape, MaxProof external-verifier). R366 covered the FMLM★ paper (different paper, same family) — its modelless delta was cold-start equivalence (config insight), NOT renoise-CE. | §2.1–2.2 prior-art table |
| **Q2** New class of behavior? | **FAIL.** Verifier-free self-evaluation / best-of-N selection is an existing capability class shipped via CLR (Plan 284, DEFAULT-ON) and CoE (R345/Plan 342). Renoise-CE is a third orthogonal signal inside this class, not a new class. | §2.3 anti-overclaim |
| **Q3** Product selling point? | **FAIL.** "Our NPCs/systems verify their own outputs without an external verifier" IS the CLR + CoE selling point (R255 §3, R345 §2.5). Adding renoise-CE as a third arm refines the selling point; it does not create a new one. | §2.3 |
| **Q4** Force multiplier? | **YES.** Connects to CLR (284), CoE (345), MaxProof (260), Self-Advantage (283), latent_functor reestimation (303), neuron-db freeze-gate (002). Five fusion targets identified (F1–F5). | §2.5 |

**Q1 PASS + Q2/Q3 FAIL + Q4 YES → GOAT.** No Super-GOAT guide created (per the "no candidate escape hatch" rule: Q2/Q3 fail means this is NOT Super-GOAT, full stop).

### 3.2 MOAT gate (§1.6) — katgpt-rs domain

- **In scope?** YES. Renoise-CE is a generic modelless inference primitive (perturb + re-resolve + measure drift). No game IP, no chain IP, no shard IP. The open primitive is pure math on operator outputs.
- **Strengthens moat?** Moderately. The self-eval / best-of-N slot of the transformer stack currently has CLR (claim-level) + CoE (trajectory-level). Renoise-CE adds perturbation-stability as a third axis. This strengthens the public engine's self-eval story (three orthogonal signals > two), which is the adoption hook for the private runtime's per-NPC test-time scaling (riir-ai R136).
- **Per-stack promote/demote tracking:** renoise-CE lands in the **self-eval / verification stack slot**. If it beats plurality vote AND CLR-alone on a self-eval benchmark → promote toward default. If CLR+renoise-CE fusion beats both → promote the fusion. Demote plurality vote (the loser the paper already shows is dominated 0.69–0.84 vs 1.00).

### 3.3 §3.5 modelless-unblock protocol — N/A

There is no failing GOAT gate to defer. The renoise-CE primitive is inference-time only. The training parts (Flow DPO, self-conditioning channel training) are unambiguously training-side and route to riir-train without a §3.5 check (they require backprop through base weights — gradient descent on preference pairs, which no deterministic construction provides).

### 3.4 §3.6 defend-wrong PoC — conditionally required

Any **parity claim** ("renoise-CE matches/beats CLR or plurality vote at self-eval") is a **quality claim** requiring a head-to-head PoC on a controlled toy benchmark, NOT just architectural reasoning. The paper provides this evidence on Sudoku/Zebra (AUROC ≈ 1.0, top-1 1.00 vs 0.69–0.84), but those are flow-LM tasks with checkers. For our modelless use cases (HLA stability, functor direction stability, shard freeze stability — no checkers), a PoC is **mandatory before any parity claim**.

**PoC scope (if/when planned):**
- Three competitors: (1) renoise-CE self-verifier, (2) plurality vote baseline, (3) CLR-alone baseline.
- Controlled toy domain: bomber_arena or a synthetic latent-state task where ground-truth "correct vs incorrect" is known.
- Verdict table: renoise-CE top-1 accuracy vs plurality vs CLR, at coverage 90/95/99%.
- Lives in `riir-ai/crates/riir-poc/` per §3.6. Use `CARGO_TARGET_DIR=/tmp/...`, clean up when done.
- The PoC defends OR refutes. If renoise-CE does NOT beat plurality/CLR on our domain, the verdict is honestly revised and the follow-up is tracked in `.issues/`.

**When PoC is NOT required for this note:** the note itself makes no quality parity claim — it claims the primitive is NOVEL (architectural) and a GOAT CANDIDATE (pending gate). The GOAT gate (in the plan) will include the PoC.

---

## 4. Open primitive (target: `katgpt-rs/crates/katgpt-core/src/renoise_ce.rs` or `katgpt-rs/crates/katgpt-core/src/renoise_ce.rs`)

```rust
//! Renoise-CE self-verifier — perturb a completed state, re-resolve through
//! the same operator, measure drift as a verifier-free correctness score.
//!
//! Distilled from Flow Reasoning Models (Helbling et al., arXiv:2606.29150).
//! Modelless: inference-time only, operator-agnostic, no external verifier.
//!
//! NOT a UQ primitive — returns a raw drift score (lower = more stable), not a
//! calibrated probability. Conformal wrapping (Plan 340 floor) required for any
//! UQ claim (Issue 010).

use crate::sampling::Operator;  // any state -> state map

/// Configuration for the renoise-CE probe.
#[derive(Clone, Debug)]
pub struct RenoiseCeConfig {
    /// Perturbation magnitude (paper: t=0.40 for flow LMs; domain-specific).
    pub perturbation_level: f32,
    /// Number of re-noise draws to average (paper: k=8; saturates at k=1).
    pub k_draws: u8,
    /// Acceptance threshold τ (lower = stricter; paper: tuned per task).
    pub tau: f32,
}

/// A single renoise-CE probe result.
#[derive(Clone, Debug)]
pub struct RenoiseCeScore {
    /// Mean cross-entropy drift across k draws (lower = more stable).
    pub drift: f32,
    /// Per-draw drifts (for variance/min aggregation).
    pub per_draw: [f32; 8],  // fixed k=8 max; paper uses k=8
    /// Acceptance decision: drift < tau.
    pub accepted: bool,
}

/// Trait for operators that can be probed by renoise-CE.
///
/// The operator maps a state to a state (denoiser, HLA evolve, functor
/// application, consolidation, attention forward). The probe perturbs the
/// input state and measures how much the output drifts.
pub trait RenoiseCeProbe {
    type State: Clone + AsRef<[f32]> + AsMut<[f32]>;

    /// Re-resolve through the operator from a (possibly perturbed) state.
    fn re_resolve(&self, state: &Self::State) -> Self::State;

    /// Perturb the state in-place (domain-specific: Gaussian noise, mask, etc.).
    fn perturb(&self, state: &mut Self::State, level: f32, rng: &mut impl rand::Rng);

    /// Cross-entropy of `candidate` under the re-resolved distribution.
    /// For continuous states: negative log-likelihood under a Gaussian centered
    /// at the re-resolved state. For discrete: token CE.
    fn drift_ce(candidate: &Self::State, re_resolved: &Self::State) -> f32;
}

/// Compute the renoise-CE score for a completed candidate.
///
/// `candidate` is the completed state to verify. The probe perturbs it,
/// re-resolves through the same operator, and measures drift.
pub fn renoise_ce_score<O: RenoiseCeProbe>(
    operator: &O,
    candidate: &O::State,
    config: &RenoiseCeConfig,
    rng: &mut impl rand::Rng,
) -> RenoiseCeScore {
    let mut per_draw = [0.0f32; 8];
    let mut sum = 0.0f32;
    let k = config.k_draws.min(8) as usize;

    for i in 0..k {
        let mut perturbed = candidate.clone();
        operator.perturb(&mut perturbed, config.perturbation_level, rng);
        let re_resolved = operator.re_resolve(&perturbed);
        let drift = O::drift_ce(candidate, &re_resolved);
        per_draw[i] = drift;
        sum += drift;
    }

    let drift = sum / k as f32;
    RenoiseCeScore {
        drift,
        per_draw,
        accepted: drift < config.tau,
    }
}

/// Verify-and-restart outer loop (Algorithm 2).
///
/// Propose via `proposer`, verify via renoise-CE, restart if unstable,
/// accept if stable, under a forward-pass budget.
pub fn verify_and_restart<P, O>(
    proposer: &P,
    operator: &O,
    config: &RenoiseCeConfig,
    budget: usize,
    rng: &mut impl rand::Rng,
) -> Option<P::Output>
where
    P: Proposer<State = O::State>,
    O: RenoiseCeProbe,
{
    let mut spent = 0;
    let mut best: Option<(f32, P::Output)> = None;
    while spent < budget {
        let (candidate, n_passes) = proposer.propose();
        spent += n_passes + config.k_draws as usize; // charge verifier NFE
        let score = renoise_ce_score(operator, &candidate, config, rng);
        if score.accepted {
            return Some(candidate.into());
        }
        // track best (lowest drift) for budget-exhausted fallback
        match &best {
            None => best = Some((score.drift, candidate.into())),
            Some((d, _)) if score.drift < *d => best = Some((score.drift, candidate.into())),
            _ => {}
        }
    }
    best.map(|(_, o)| o)
}

pub trait Proposer {
    type State: Clone;
    type Output;
    /// Propose one candidate, return (state, forward passes consumed).
    fn propose(&self) -> (Self::State, usize);
}

/// Best-of-N selection by renoise-CE stability (Appendix C, passive case).
///
/// Keep the most stable proposal from N i.i.d. samples.
pub fn best_of_n_stability<P, O>(
    proposer: &P,
    operator: &O,
    config: &RenoiseCeConfig,
    n: usize,
    rng: &mut impl rand::Rng,
) -> Option<P::Output>
where
    P: Proposer<State = O::State>,
    O: RenoiseCeProbe,
{
    (0..n)
        .filter_map(|_| {
            let (candidate, _) = proposer.propose();
            let score = renoise_ce_score(operator, &candidate, config, rng);
            Some((score.drift, candidate))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .map(|(_, c)| c.into())
}
```

**Design notes:**
- Zero-allocation hot path: `per_draw` is a fixed `[f32; 8]`, `perturb` operates in-place, `re_resolve` returns owned state (one allocation per draw, unavoidable).
- Operator-agnostic: `RenoiseCeProbe` trait works for any state→state operator (denoiser, HLA, functor, consolidation, attention).
- Conformal-floor caveat documented in module docstring (NOT a UQ primitive; raw ranking signal).
- `verify_and_restart` charges every verifier NFE to the budget (paper §B).

---

## 5. GOAT gate criteria (for the plan)

| Gate | Criterion | Target | Validation |
|---|---|---|---|
| **G1** | Renoise-CE selection accuracy > plurality vote | top-1 ≥ 0.95 vs plurality ≤ 0.85 at 99% coverage | Synthetic toy domain (bomber_arena or latent-state task with known correct/incorrect) |
| **G2** | Renoise-CE + CLR fusion > CLR-alone | ≥ +5% top-1 over CLR-alone | Same domain; CLR arm + renoise-CE arm combined |
| **G3** | No regression on existing self-eval | CLR + CoE benchmarks unchanged when renoise-CE is opt-in | Feature-isolation test |
| **G4** | Zero-allocation hot path | `renoise_ce_score` allocates 0 (fixed `[f32; 8]`, in-place perturb) | Bench: ≤ 1µs at HLA scale (8-dim) |
| **G5** | Latency | `renoise_ce_score` p50 < 100µs at vocab=128 (1 re-resolve + k=8 perturbs) | Criterion bench |
| **G6** | Feature isolation clean | All existing tests pass with `renoise_ce` feature on and off | CI feature guard |

**UQ floor rule (Issue 010):** if any conformal-wrapped UQ claim is made (renoise-CE as calibrated correctness probability), it MUST beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1) on CRPS/coverage/Winkler. Until Plan 340 ships, renoise-CE is a **ranking signal**, not a UQ distribution.

**Promotion rule:** if G1–G6 pass AND G2 shows CLR+renoise-CE > CLR-alone → promote `renoise_ce` toward default (alongside CLR). Demote plurality vote (the documented loser). If G2 fails → keep `renoise_ce` opt-in as an alternative signal.

---

## 6. What stays where (5-repo discipline)

| Component | Repo | Why |
|---|---|---|
| `renoise_ce_score` open primitive (generic perturb + re-resolve + drift) | `katgpt-rs` (MIT) | Generic modelless inference primitive, no game/chain/shard IP |
| `verify_and_restart` / `best_of_n_stability` open loop framework | `katgpt-rs` (MIT) | Generic test-time scaling, composes with DDTree/CLR/MaxProof |
| CLR + renoise-CE fusion arm (F1) | `riir-ai` (private) | Game-runtime IP: per-NPC test-time scaling integration |
| Proactive functor stability probe (F2) | `riir-ai` (private) | riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs extension — runtime IP |
| Proactive shard freeze-gate probe (F5) | `riir-neuron-db` (private) | consolidation.rs `can_freeze` extension — shard IP. **LANDED 2026-07-06** as Plan 406 P5.3 (`renoise_freeze_probe` opt-in, riir-neuron-db commit `725eed1`; `proactive_freeze_drift` + `can_freeze_proactive`, strict downgrade-only). |
| HLA renoise-CE probe (F4) | `riir-ai` (private, speculative) | Per-NPC self-eval — needs R344 null-result re-validation. **RESOLVED 2026-07-09** as Plan 414 (tracked as `.issues/048`, resolved + removed): R344 unblocked by construction — `CommittedBlendState` is a one-shot closed-form map, NOT an iterative attractor, so the Plan-276 flip-flop pathology is impossible. Landed as the opt-in `hla_committed_belief_probe` π-Lipschitz diagnostic in katgpt-core (public, GOAT ALL PASS); per-NPC runtime consumer deliberately deferred (no-default-consumer rule). |
| Flow DPO / self-conditioning channel training | `riir-train` (private) | Training — preference optimization on self-mined negatives |

---

## 7. What this note prevents (canonical failure modes averted)

1. **False Super-GOAT on "renoise-CE is a new capability class."** It is NOT — verifier-free self-eval ships as CLR (claim-level) and CoE (trajectory-level). Renoise-CE is a third orthogonal signal (perturbation-stability) inside the existing class. Q2/Q3 fail → GOAT, not Super-GOAT.

2. **False duplicate claim vs Research 366.** R366 covers a DIFFERENT paper (arXiv:2607.00714, FMLM★, KAIST/UvA/CMU) — its modelless delta was cold-start equivalence (a config insight). This note covers arXiv:2606.29150 (FRM, MIT-IBM) — its modelless delta is the renoise-CE self-verifier (a genuine primitive). Same research family (self-conditioned flow LMs, fixed-point view), different papers, different modelless deltas. Both notes are needed.

3. **False "already ships" PASS verdict.** The renoise-CE fingerprint (perturb OUTPUT + re-resolve + measure drift = verifier) is NOT exactly shipped. Every closest cousin misses the perturbation step: Q-Sample (Plan 222) refines, Self-Advantage (Plan 283) compares same-input, CLR (Plan 284) votes on claims, CoE (R345) reads trajectory shape, MaxProof (R260) uses external verifier. The architectural coverage is partial; the perturbation-and-re-resolve-as-verifier is the novel delta. A PASS verdict backed only by "self-conditioning ships" would be the §3.6 false-PASS failure mode.

4. **Mis-routing training value to katgpt-rs.** Flow DPO (preference training on self-mined wrong cells, wrong-cell localized contrast, EMA-pinned reference), self-conditioning channel training (zero-init two-pass scheme), hard-negative mining loop — all unambiguously training-side → riir-train.

5. **False UQ claim without conformal floor.** Renoise-CE returns a raw drift score, not a calibrated probability. Any UQ claim (correctness probability, confidence interval) MUST be conformal-wrapped and beat the floor (Plan 340, Issue 010). Until then, it is a ranking signal.

6. **Re-evaluation by a future agent.** This note + the prior-art table (§2.1) + the distinction from R366 (§2.2) + the five fusion targets (§2.5) should prevent any future session from re-running the novelty gate on this paper or conflating it with R366.

---

## 8. Action items

- [x] **Document the paper** (this note) — record findings, prior-art surface, renoise-CE primitive, fusion targets, → riir-train routing.
- [-] **No Super-GOAT guide in riir-ai / riir-chain / riir-neuron-db.** Q2/Q3 fail (verifier-free self-eval capability class already ships). No private guide warranted.
- [x] **Plan in katgpt-rs** (next session or follow-up): `katgpt-rs/.plans/369_renoise_ce_self_verifier.md` — implement the open primitive behind `renoise_ce` feature flag, run GOAT gate G1–G6, include the §3.6 defend-wrong PoC. — **DONE as Plan 406 (2026-07-06): G1+G2 PASS, promoted to DEFAULT-ON.**
- [-] **→ riir-train** (if pursued): Flow DPO recipe (wrong-cell localized preference contrast, EMA-pinned reference, self-mined hard negatives, priority replay buffer), self-conditioning channel training (zero-init two-pass scheme), online vs offline FMLM★-style distillation. Out of scope for this workflow; noted for completeness.
- [x] **Track HLA renoise-CE (F4) as a `.issues/` follow-up** — needs R344 null-result re-validation before planning. The single-step stability probe on a committed HLA state is different from the multi-step FP iteration that null-resulted, but the caveat is serious. — **RESOLVED: tracked as `.issues/048_hla_committed_belief_probe_blocked_on_r344.md` (resolved + removed per noise-reduction), executed as Plan 414 (2026-07-09). The R344 caveat dissolved BY CONSTRUCTION: `CommittedBlendState` is a one-shot closed-form map, NOT an iterative attractor — the Plan-276 flip-flop pathology cannot occur. Shipped as opt-in `hla_committed_belief_probe` (katgpt-core π-Lipschitz sensitivity diagnostic; GOAT G1+G1b+G2+G2b+G3+G4+G5 ALL PASS; Plan 414 also corrected the issue's proposed gate — the probe perturbs π, so the relevant bound is the π-Lipschitz constant, not the cached z-sensitivity one). Per-NPC runtime consumer deliberately not wired (no-default-consumer rule).**
- [x] **Track proactive freeze-gate probe (F5) for riir-neuron-db** — extends `can_freeze` (Plan 002) with a renoise-CE stability probe. Cross-repo note in `riir-neuron-db/.research/` when pursued. — **DONE as Plan 406 P5.3 (2026-07-06; riir-neuron-db commit `725eed1`): opt-in `renoise_freeze_probe` feature — `proactive_freeze_drift()` (k-draw perturb-and-re-average → MSE drift vs the clean delta) + `can_freeze_proactive()` companion (STRICT downgrade-only: never upgrades a reactive fail), DRY `average_embeddings()` shared by `sleep()` + probe; 6 tests (`consolidation/renoise_probe_tests.rs`), clippy + all feature combos clean. Durable record = riir-neuron-db `.docs/04_consolidation_retrieval/consolidation_pipeline.md` §`proactive_freeze_drift` — supersedes the planned `.research/` note.**

---

## TL;DR

**Verdict: GOAT.** Flow Reasoning Models (Helbling et al., MIT-IBM, arXiv:2606.29150) introduces the **renoise-CE self-verifier** — perturb a completed state, re-resolve through the same operator, measure drift as a verifier-free correctness score (no external verifier, no labels, no auxiliary head; AUROC ≈ 1.0, top-1 1.00 vs plurality 0.69–0.84 on Sudoku-Extreme). The self-conditioning inner loop and fixed-point iteration are already shipped (Plan 222/108) and the FP-for-LM design space is closed (R344/R366); the Flow DPO training recipe → riir-train. The genuinely novel modelless delta is renoise-CE: the {perturb OUTPUT + re-resolve + measure drift = verifier} fingerprint is not exactly shipped — every closest cousin misses the perturbation step (Q-Sample refines, Self-Advantage same-input, CLR claim-vote, CoE trajectory-shape, MaxProof external-verifier). It composes with CLR (284) / CoE (345) / MaxProof (260) as a **third orthogonal self-eval signal** and has a clean latent-space reframing as a **proactive stability probe** (vs the current reactive coherence gates) on functor directions (F2) and shard freeze-gates (F5). NOT Super-GOAT because the verifier-free self-eval capability class already ships (CLR, CoE) — Q2/Q3 fail. Open primitive in `katgpt-rs` (generic `renoise_ce_score` + `verify_and_restart` + `best_of_n_stability`, feature flag `renoise_ce`, GOAT-gated with defend-wrong PoC per §3.6). Five fusion targets: CLR vote arm (F1), proactive functor probe (F2), verifier-free MaxProof (F3), HLA committed-belief probe (F4, speculative — R344 caveat), proactive shard freeze-gate (F5) — **all five since landed; F4/F5 dispositions recorded in §6 + §8.** This note is distinct from Research 366 (different paper, same family — R366's delta was cold-start equivalence; this note's delta is the renoise-CE verifier).
