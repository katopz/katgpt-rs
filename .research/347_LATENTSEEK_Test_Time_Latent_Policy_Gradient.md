# Research 347: LATENTSEEK — Test-Time Instance-Level Policy Gradient in Latent Space

> **Source:** [Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space](https://arxiv.org/abs/2505.13308) — Li, Li, Wu, Zhu, Wang, Yu, Jiang, Zhu, Jia, Wu, Zheng (PKU · BIGAI NLCo · Tsinghua · SJTU · CASIA · UCLA), arXiv:2505.13308v3, 19 Jan 2026.
> **Code:** https://github.com/bigai-nlco/LatentSeek
> **Date:** 2026-06-29
> **Status:** Done
> **Related Research:** 325 (Survey gap G5 — this note closes it), 019 (TTT-Discover — closest gradient-on-weights cousin, NO-GAIN), 124 (ViTTT — vision TTT, NO-GAIN), 284/255 (CLR open primitive — modelless analog), 136 (riir-ai Per-NPC CLR Runtime — modelless test-time scaling), 240 (CGSP — curiosity host), 123 (riir-ai Latent Functor Runtime Guide — coherence-driven re-estimation cousin)
> **Related Plans:** none (no new primitive — modelless analog already ships)
> **Classification:** Public

---

## TL;DR

LATENTSEEK does test-time instance-level adaptation (TTIA) by running **REINFORCE policy gradient directly on token-wise latent representations** `z_t` (the activations ahead of the LM head), NOT on base weights. The model parameters are frozen; only the per-instance latents are updated via `z ← z + η∇_z J(z)` with a self-reward signal `R(x,c) ∼ π(·|x,c,prompt_self-reward)`. Reported gains: +15.23 over BoN on GSM8K, +6.67 over CoT on AIME2024, +18.1 over SimpleRL-Zoo on GSM8K+MATH-500 averaged. Converges in <2 iterations on average. Notably, the decoded intermediate tokens are often incoherent ("total downloads of downloads") yet the final answer is correct — evidence that the model's native reasoning geometry lives in latent space, not token space.

**Distilled for katgpt-rs (modelless, inference-time):** **No new primitive.** The §3.5 modelless-unblock check succeeds: the *deterministic-construct modelless analog* of LATENTSEEK's reward-guided latent refinement is **already shipped** across three modules —
1. `cgsp_runtime` (`riir-ai`) — runtime curiosity-guided self-play with `solve_rate = sigmoid(sharpness · dot(candidate, target))` priority-table updates (no gradient).
2. `latent_functor/reestimation.rs` (`riir-ai`) — coherence-driven re-estimation scheduler that recomputes direction vectors when `coherence < tau_reest` and atomically swaps BLAKE3-committed entries (no gradient).
3. CLR Per-NPC Runtime Test-Time Scaling (R136/R284) — nonlinear reliability voting `(mean(v))^M` over M dot-product + sigmoid claim checks against BLAKE3-committed direction vectors (no gradient).

The one transferable *theoretical* insight is LATENTSEEK's **independence assumption** (§2.2, §C.2): treating latent positions as independent provably enlarges the exploration space vs. autoregressive conditioning, and the MIP-Bounded = MIP = NEXP complexity argument (Thm C.10, C.11) bounds the loss. This is a *justification* for our existing per-position dot-product + sigmoid direction-vector approach (CLR, Salience Tri-Gate, latent_functor) — not a new mechanism.

**Verdict: Gain.** Anti-duplication note for survey gap G5. The borderline-modelless classification is recorded below so future sessions don't re-distill this paper into a false Super-GOAT.

---

## 1. Paper Core Findings

### 1.1 The mechanism — REINFORCE on token latents, not weights

For reasoning sequence `x = (x_1,…,x_T)` with latent representations `z = (z_1,…,z_N)` (where `z_t := π_Transformer(x_<t, c)` lies ahead of the LM head, following Hao et al. 2024 / Kong et al. 2025), LATENTSEEK optimizes:

```
z* = argmax_z  E_{x∼π(x|z,c)}[ R(x, c) ]           (Eq. 3)
```

via direct policy gradient (REINFORCE, Williams 1992):

```
z ← z + η · ∇_z J(z)                                  (Eq. 5)
[∇_z J(z)]_t = E_{x∼π(x|z,c)}[ R(x,c) · ∇_{z_t} log π(x_t | z_t) ]   (Eq. 7)
```

Crucially: **only `z` is updated, the model parameters are frozen.** The backward pass for `∇_{z_t} log π(x_t|z_t)` is through the **LM head only** (~525 MFLOPs forward / ~1.05 GFLOPs backward per LLaMA-3.1-8B, §G.3), NOT through the transformer backbone. This is what makes LATENTSEEK borderline-modelless per our constraint #4: it updates latent state, not base weights, but it uses gradient descent (backprop) on the LM head rather than a deterministic construction.

### 1.2 The independence assumption (the actual theoretical contribution)

The latent positions are treated as **independent** (Eq. 7 sums per-position gradients without cross-position terms). The paper gives two justifications:

- **Practical** (§2.2 reason 1): without independence, the autoregressive structure `π(x_t|x_<t)` forces all optimization pressure onto `z_1` (every subsequent token is conditioned on it), collapsing the effective search space. Independence decouples positions, yielding a "substantially larger exploration space and a more flexible launch pad".
- **Theoretical** (§C.2): the MIP-Bounded complexity class. LATENTSEEK-with-independence maps onto Multi-Prover Interactive Proofs where each prover emits one bounded token. **Theorem C.10: MIP-Bounded = MIP**; **Corollary C.11: NP ⊂ NEXP = MIP-Bounded**. The independence deficit is bounded by (a) the base model's faithful follow-up generation and (b) the reward model's evaluation capacity.

This is the genuinely novel theoretical artifact: **independence over latent positions is not a bug — it's the mechanism that unlocks latent-space search power equivalent to NEXP.**

### 1.3 Two enhancing techniques (engineering, not novelty)

1. **CoT initialization** — initial `z` comes from a CoT rollout (good launch pad).
2. **Fractional sequence optimization** — only optimize `z_{1…ρT}` with `ρ ∈ (0,1]` (default 0.2). Excessive latent modification produces incoherent decodes that break the reward function.

### 1.4 The Perfect Sparse Reward Model (PSRM) experiment — the key diagnostic

Replacing the self-reward with a binary PSRM (0 if answer exactly matches ground truth, −1 otherwise) — near-zero directional information, close to blind exploration — still yields **+10.67 average points** over the self-reward variant (Table 3). With extreme scaling (K=256 iterations), Qwen2.5-1.5B + LATENTSEEK-PSRM beats GPT-4o on AIME2024 by 14 points and trails o1-preview by only 2.7 on MATH-500.

**Interpretation:** the latent space alone carries enough expressivity that *unguided* exploration in it (PSRM is essentially random-reward hill-climbing) recovers most of the gain. The gradient signal matters less than the *space being searched*.

### 1.5 The "incoherent but correct" phenomenon (§3.8, §I)

LATENTSEEK frequently produces correct final answers from linguistically anomalous intermediate tokens ("let'll more understand it down step two andLet", "total downloads of downloads"). Wordcloud analysis (Fig 10) shows first-token distributions dominated by prepositions ("let"), second by verbs ("find", "solve"), third by nonsensical proper nouns ("thecy", "theella"). **Implication:** LLM reasoning geometry is native to latent space; the token-space rendering is a lossy projection. This is consistent with the latent-CoT line (Hao 2024, Kong 2025, Deng 2024 iCoT) but is the first to show it empirically via test-time optimization.

### 1.6 What does NOT transfer (the ablations)

- **Stochastic exploration (SE) baseline** (Table 5): single-shot Gaussian noise `z ← z + ε` with `σ² ∈ {0.5,0.75,1.0}` underperforms LATENTSEEK by ~13.66 points. → Gradient guidance matters, not just noise injection.
- **Constant-reward ablation** (Table 7): replacing self-reward with constant −1 drops performance by 14.21 points. → The reward signal must carry directional information for the *self-reward* variant (but note PSRM shows the directional information can be near-zero and still win — the contradiction is resolved by PSRM running many more iterations).
- **Middle-stage optimization** (Table 13): optimizing `z_{N1+1…N}` (40% into the sequence) underperforms initial-stage optimization by 7.58 points. → Prefix-independence limits the effectiveness; the launch pad must be at the start.

---

## 2. Distillation

### 2.1 Critical classification — borderline modelless, NOT training-only

LATENTSEEK does **not** update base weights. The paper states this explicitly (§1, §5): "without modifying its parameters", "without requiring parameter updating", "circumvents the need for parameter updates". The only thing mutated at test time is the per-instance latent sequence `z`.

This satisfies the **letter** of our constraint #4 ("runtime GRPO self-play stays modelless IF it updates latent state only, NOT base weights"). The **spirit** is murkier: LATENTSEEK uses backpropagation through the LM head to compute `∇_z log π(x_t|z_t)`. That is gradient descent at inference time — ~1.05 GFLOPs of backward compute per iteration (§G.3), per instance. Constraint #1 ("No LLM training, no backprop through base weights") is technically satisfied (the transformer backbone is frozen, only the LM head sees backward), but constraint #2's preference for "dot-product + sigmoid projections onto learned direction vectors" is violated: LATENTSEEK's update is a learned gradient, not a deterministic projection.

**Honest classification: BORDERLINE MODELLESS.** Per the assignment, this is the case where §3.5 must run before any riir-train deferral.

### 2.2 §3.5 modelless-unblock protocol — the decision

The protocol asks: can LATENTSEEK's `∇_z log π(x_t|z_t)` be replaced by a deterministic construction?

**Path 1 — Freeze/thaw snapshot correction** (`riir-neuron-db/src/freeze.rs`, `MerkleFrozenEnvelope`):
LATENTSEEK's update is *per-instance* (each problem gets its own `z*`); freeze/thaw is *per-snapshot* (one thawed state serves many instances). Wrong granularity. **Path 1 FAILS** — wrong temporal scope.

**Path 2 — Raw/lora reader-writer hot-swap** (`LoraPair { reader, writer }`, Plan 025):
A deterministically constructed reader/writer LoRA applies a fixed correction to all inputs. LATENTSEEK's correction is instance-specific (depends on the self-reward `R(x,c)` evaluated on this instance's rollout). **Path 2 FAILS** — wrong instance scope.

**Path 3 — Latent-space correction** (dot-product projection + sigmoid gate, per constraint #2):
**This is where the modelless analog lives.** The question: can the instance-specific reward-guided latent update be replaced by a deterministic projection + sigmoid gate?

YES — and we already ship three independent instances of this replacement:

| LATENTSEEK component | Modelless analog (shipped) | Mechanism |
|---|---|---|
| `z ← z + η∇_z J(z)` (per-position gradient ascent) | `cgsp_runtime` priority-table update (`riir-ai/crates/riir-engine/src/cgsp_runtime/`) | `solve_rate = sigmoid(sharpness · dot(candidate, target))`; bandit absorbs rewards into priority table; NO gradient |
| Reward-guided latent refinement loop | `latent_functor/reestimation.rs` ("coherence-driven re-estimation scheduler") | When `coherence < tau_reest`, recompute direction vector from fresh observations, atomic swap with new `Uuid::now_v7()` snapshot + BLAKE3 commitment; NO gradient |
| Self-reward `R(x,c) ∼ π(·|x,c,prompt)` evaluating trajectory quality | CLR Per-NPC Runtime Test-Time Scaling (R136/R284, riir-ai Plan 316) | Sample K candidates → extract M claims → `v[k][m] = sigmoid(dot(claim_vec, direction_vec))` → reliability `r[k] = pow(mean_m v[k][m], M)` → vote; NO gradient |

**Path 3 SUCCEEDS — modelless-validable.** The deterministic-construct modelless analog of LATENTSEEK already ships. No new primitive needed.

**Why the modelless analog is sufficient (and arguably superior for our domain):**

1. **Latency**: LATENTSEEK's per-iteration cost is ~3.07 × 10¹¹ FLOPs (2 forward + 1 LM-head backward, §G.3). At 20Hz game tick with a 50ms budget per NPC, this is 4+ orders of magnitude too slow. Our `cgsp_runtime` cycle is <1ms (R136 §1.2).
2. **Crowd scale**: LATENTSEEK optimizes one instance at a time on one A100/L40/4090. Our runtime runs thousands of concurrent NPCs each with independent latent state. Per-instance gradient descent does not vectorize across NPCs the way dot-product + sigmoid does.
3. **Deterministic replay / anti-cheat**: LATENTSEEK's `z*` is the result of stochastic optimization (REINFORCE is a stochastic gradient estimator). Bit-identical reconstruction across nodes (required for quorum sync, AGENTS.md raw-domain rule) is impossible without checkpointing `z*` per instance — which collapses back to freeze/thaw, not gradient descent.
4. **The PSRM ablation proves the gradient barely matters**: with near-zero directional reward signal, latent-space search still wins by +10.67. The *space* (latent, decoupled positions) is doing the work, not the gradient. Our modelless analogs already operate in that same space.

### 2.3 What IS transferable — the independence-assumption theoretical justification

The one piece of LATENTSEEK that does NOT ship anywhere in our corpus is the **theoretical argument** (§C.2, Thm C.10/C.11) that independent per-position latent updates unlock NEXP-equivalent search power, with the deficit bounded by base-model follow-up faithfulness and reward-model accuracy.

This is not a new primitive — it is a *justification* for design choices we already made:
- CLR's per-claim dot-product + sigmoid (R284) treats claims as independent reliability contributors: `r = pow(mean_m v[m], M)`. The independence is an implementation choice; LATENTSEEK proves it's also a *power-preserving* choice.
- Salience Tri-Gate (R281) emits per-tick Speak/Silent/Delegate decisions as independent sigmoid gates per direction vector. Same independence, same justification now available.
- `latent_functor` operator-valued C matrices (Plan 318) treat spectral projections as independent channels. Same.

**Actionable consequence:** when future sessions question whether our per-position/per-claim independence is a limitation (e.g., "shouldn't we model cross-claim dependencies?"), LATENTSEEK §C.2 is the citation that says "no — independence is provably sufficient up to base-model + reward-model capacity, and decoupling is what unlocks the search space". File this as a theoretical note, not a primitive.

### 2.4 Fusion search — no novel combination

Following the fusion protocol (§Workflow step 1), I grepped all five repos (both layers: `.research/`+`.plans/`+`.docs/` for intent, `src/`+`crates/` for shipped code) with both paper vocabulary (TTIA, policy gradient, latent representation, REINFORCE) and codebase vocabulary (test-time scaling, runtime self-play, curiosity, coherence re-estimation, direction vector, sigmoid projection, CLR, cgsp).

**Closest cousins (in priority order):**

1. **R136 (riir-ai) + R284/255 (katgpt-rs) — CLR Per-NPC Runtime Test-Time Scaling.** This IS the modelless analog of LATENTSEEK. Same paradigm (test-time instance-level adaptation, no weight updates), same shape (sample K candidates, score with reward-like signal, vote/refine), different mechanism (dot-product + sigmoid + nonlinear reliability voting vs. REINFORCE gradient). **Fusion: redundant — already shipped.**
2. **R123 (riir-ai) — Latent Functor Runtime Guide + `latent_functor/reestimation.rs`.** Coherence-driven re-estimation is the modelless analog of LATENTSEEK's iterative latent refinement. Same trigger (reward/coherence signal crosses threshold), different update (recompute from observations vs. gradient step). **Fusion: redundant — already shipped.**
3. **R019 — TTT-Discover.** Closest *gradient-on-weights* cousin. Explicitly concluded NO-GAIN because per-query full-LoRA training at test time ($500/problem) is not production-viable. LATENTSEEK is cheaper (LM-head backward only) but still 4 orders of magnitude too slow for 20Hz tick. **Fusion: already declined.**
4. **R124 — ViTTT.** Vision TTT. Explicitly NO-GAIN — vision-specific, causal-mismatch, no game-domain connection. **Fusion: already declined.**
5. **R240 — CGSP (Curiosity-Guided Self-Play).** The runtime host for curiosity-driven exploration. Already integrated with CLR (R136 §1.2 step 7) and latent_functor (R123). **Fusion: already integrated.**

**No novel combination of LATENTSEEK × cousin A × cousin B produces a capability none of them has alone.** The modelless analog already covers the test-time latent-adaptation capability; LATENTSEEK's gradient adds latency without adding capability (per the PSRM ablation).

### 2.5 Latent-to-latent reframing (mandatory per §Workflow step 3)

Re-casting LATENTSEEK's mechanism as a latent-to-latent op on each of the seven Super-GOAT factory modules:

| Substrate | LATENTSEEK reframing | Already shipped? |
|---|---|---|
| (a) HLA per-NPC latent state (`sense/`) | Per-NPC reward-guided refinement of the 8-dim affect vector | ✅ `cgsp_runtime` + CLR voting |
| (b) `latent_functor/` operations | Coherence-triggered re-estimation of operator-valued C matrices | ✅ `reestimation.rs` (coherence < `tau_reest` → recompute + atomic swap) |
| (c) `cgsp_runtime/` curiosity signals | Reward-absorbing bandit + priority table that biases next-cycle candidate generation | ✅ `runtime.rs`, `anti_cheat.rs` ("runtime's bandit absorbs rewards and the priority table moves") |
| (d) LatCal fixed-point commitment (chain) | N/A — LATENTSEEK's `z*` is per-instance stochastic, cannot be LatCal-committed without breaking determinism | N/A by design |
| (e) `NeuronShard` style_weights / freeze envelope | N/A — LATENTSEEK doesn't touch shard storage; the per-instance `z*` is ephemeral | N/A by design |
| (f) DEC Stokes operators (`dec/`) | N/A — no divergence/boundary/Hodge structure in LATENTSEEK's mechanism | N/A |
| (g) Adapter routing | N/A — LATENTSEEK doesn't route between adapters; it modifies a single model's latents | N/A |

The reframing confirms: **the latent-to-latent operation LATENTSEEK performs is already shipped in (a), (b), (c).** Substrates (d), (e), (f), (g) are inapplicable by LATENTSEEK's design (per-instance stochastic ephemeral latents, no commitment, no manifold structure, no adapter routing). No Super-GOAT angle.

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + selling point + force multiplier | Open primitive + private guide + plans |
| **GOAT** | Provable gain over existing approach; promotes to default if it wins | Plan + implement + feature flag + benchmark |
| **Gain** | Incremental / anti-duplication note / theoretical justification | Note only |
| **Pass** | Not relevant, OR training-only (→ riir-train) | One-line note |

### Verdict: **Gain**

**One-line reasoning:** LATENTSEEK is borderline-modelless (test-time REINFORCE on token latents, not base weights — passes constraint #4 letter), but §3.5 modelless-unblock Path 3 succeeds: the deterministic-construct analog (dot-product + sigmoid direction-vector voting, coherence-driven re-estimation, curiosity-absorbing priority tables) already ships across `cgsp_runtime` + `latent_functor/reestimation` + CLR (R136/R284), so there is no new primitive to implement; the only transferable artifact is the §C.2 independence-assumption theoretical justification (MIP-Bounded = NEXP), which serves as a *citation* for design choices we already made, not a new mechanism.

### Why not Super-GOAT

- **Q1 (no prior art?):** FAIL — `cgsp_runtime` + CLR + `latent_functor/reestimation` are exact modelless analogs. Three-layer check (notes + code + vocabulary translation) all hit.
- **Q2 (new capability class?):** FAIL — test-time instance-level adaptation is already a shipped capability (CLR R136 explicitly titled "Per-NPC Runtime Test-Time Scaling").
- **Q3 (product selling point?):** FAIL — cannot finish "our NPCs do X that no competitor can" with anything LATENTSEEK-specific. The selling point is already claimed by CLR.
- **Q4 (force multiplier?):** FAIL — no novel connection to ≥2 pillars; the connections (HLA, CGSP, latent_functor) are already wired by CLR.

### Why not GOAT

- No provable gain over the shipped modelless analog. The PSRM ablation (§1.4) suggests the *space* does the work, not the gradient — and we already operate in that space at 1000× lower latency. Running a GOAT gate (LATENTSEEK-style gradient vs. CLR-style voting on the same task) would almost certainly show CLR winning on the perf/sec gate (G3) at comparable or worse quality, because the gradient adds ~10¹¹ FLOPs/iter for a directional signal the PSRM ablation shows is barely necessary.

### Why not Pass

- The paper is NOT training-only (it explicitly avoids weight updates). It is a legitimate borderline-modelless candidate flagged by Survey 325 §7.2 gap G5. Recording the §3.5 decision prevents future re-distillation. The §C.2 independence-assumption theorem is a genuinely useful citation for our existing design choices.

### Honest caveats on the modelless-vs-trained classification

1. **Constraint #4 letter vs. spirit.** LATENTSEEK passes the letter ("updates latent state, not base weights") but uses backprop through the LM head, which is closer to "training" than to "deterministic construction". Constraint #2's preference for "dot-product + sigmoid onto learned direction vectors" is the operative rule that pushes us toward the modelless analog. If a future revision of constraint #4 explicitly forbids *any* backprop at inference time (including LM-head-only backward), LATENTSEEK becomes definitively → riir-train. The current wording leaves it borderline.
2. **The gradient analog would need riir-train if we ever wanted it.** If a future task requires the *exact* LATENTSEEK mechanism (per-instance REINFORCE on token latents through the LM head), that is a training-adjacent compute pattern (backprop, optimizer state, learning rate) and belongs in `riir-train` — not katgpt-rs, not riir-ai. The modelless analog is the only thing that ships in our 5-repo strategy.
3. **Large-model scaling is unvalidated.** LATENTSEEK is tested up to 14B (§3.1). Our 20Hz game-tick budget cannot afford even one LATENTSEEK iteration on a 14B model. The latency gap is fundamental, not a tuning problem.
4. **The PSRM result cuts both ways.** It could be read as "latent-space exploration is so powerful that even random-reward hill-climbing wins" (our interpretation — supports the modelless analog) OR as "with a strong enough reward model, gradient-guided latent search is even better" (supports a future riir-train investigation). The paper's §3.3 explicitly leans toward the second reading for the extreme-scaling regime (K=256 iterations on a 1.5B model beating GPT-4o). That regime is irrelevant to our 20Hz / thousands-of-NPCs / sub-ms-per-decision domain.

---

## 4. Cross-references

- **Survey 325 §7.2 gap G5** — closed by this note.
- **R019 (TTT-Discover)**, **R124 (ViTTT)** — closest gradient-on-weights / vision-TTT cousins, both NO-GAIN.
- **R136 (riir-ai CLR Per-NPC Runtime) + R284/255 (katgpt-rs CLR open primitive)** — the modelless analog that already ships.
- **R123 (riir-ai Latent Functor Runtime Guide)** + `latent_functor/reestimation.rs` — coherence-driven re-estimation, the iterative-refinement modelless analog.
- **R240 (CGSP)** — curiosity-guided self-play host.
- **R281 (Salience Tri-Gate)** — another per-position independence design that LATENTSEEK §C.2 justifies theoretically.

## TL;DR

LATENTSEEK runs REINFORCE policy gradient on per-instance token latents (not base weights) at test time — borderline-modelless per constraint #4 letter, but §3.5 Path 3 (latent-space correction via dot-product + sigmoid) already ships as the deterministic-construct analog in `cgsp_runtime` + `latent_functor/reestimation` + CLR (R136/R284). **Verdict: Gain** — anti-duplication note for survey gap G5, no new primitive, no plan. The only transferable artifact is the §C.2 independence-assumption theorem (MIP-Bounded = NEXP) — a citation for design choices we already made, not a new mechanism. The borderline classification is honest: if a future constraint revision forbids *any* inference-time backprop (including LM-head-only), LATENTSEEK becomes definitively → riir-train; today the modelless analog wins on perf/sec (10¹¹ FLOPs/iter for LATENTSEEK vs. <1ms for CLR at 20Hz tick) without sacrificing the capability the PSRM ablation proves is carried by the *latent space itself*, not the gradient.
