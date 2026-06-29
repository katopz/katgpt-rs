# Research 325: A Survey on Latent Reasoning — Unifying Taxonomy Map

> **Source:** [A Survey on Latent Reasoning](https://arxiv.org/pdf/2507.06203) — Zhu, Peng, Cheng, Qu, Huang, Zhu, Wang, Xue, Zhang, Shan, Cai, Kergan, Kembay, Smith, Lin, Nguyen, Pan, Chou, Cai, Wu, Zhao, Liu, Yang, Zhou, Zheng, Li, Zhou, Li, Zhang, Liu, Zhang, Huang, Eshraghian (UCSC, FDU, NJU, PKU, RUC, UoM, UW-Madison, PolyU, M-A-P), arXiv:2507.06203v2, Jul 2025
> **Date:** 2026-06-29
> **Status:** Done
> **Related Research:** 028 (HLA), 034 (D2F), 035 (Attractor), 048 (HRM), 073 (LT2), 097 (Training-Free Looped), 158 (MUX), 175 (ThoughtFold), 192 (NextLat), 230 (SSD duality), 241 (SwiR switch), 242 (Topological recurrent belief), 263 (Latent Thought Flow), 265 (CoFRe FP-MGM), 266 (FPRM damped halting), 273 (ELT), 282 (LoopCoder-V2), 317 (Reasoning as attractor)
> **Related Plans:** 025 (bidirectional prefill), 066 (D2F), 108 (LT2 looped), 136 (TF Loop), 217 (NextLat drafter), 276 (MicroRecurrentBeliefState), 291 (D2F 3SR warm-start)
> **Classification:** Public
> **Verdict: Gain** — survey, not a new mechanism. Value is a unifying taxonomy that maps the codebase's scattered latent-reasoning corpus to a single frame, prevents future false-Super-GOAT claims in this saturated corner, and surfaces two narrow fusion gaps (bidirectional dKV-Cache for diffusion; explicit "depth-from-optimization-over-time" bridge framing for `latent_functor/reestimation.rs`).

---

## TL;DR

This is a **survey**, not a new method paper. Its value to us is **organizational, not mechanical**: it provides a clean two-axis taxonomy (vertical / activation-based vs horizontal / hidden-state-based recurrence) plus a third "infinite-depth via diffusion" axis and a mechanistic-interpretability chapter, all of which map onto primitives we already ship. The bandwidth framing (≈40,960 bits per FP16 hidden state vs ≈15 bits per token, ~2.7×10³-fold gap) is a useful one-liner for selling Pillar 8 (Reasoning Pack).

**Distilled for katgpt-rs (modelless, inference-time):**

The survey confirms — at the literature-aggregation level — that **every major latent-reasoning family is already represented in our corpus**. The deliverable is therefore not code; it is **this map**, which future research sessions should grep before claiming novelty on any "latent CoT", "continuous thought", "looped transformer", "hidden-state recurrence", "fast-weight", or "diffusion reasoning" paper. The canonical failure mode this prevents is exactly the skill's standing warning: paper-vocabulary-only greps that miss shipped code under codebase vocabulary (HLA = horizontal recurrence, AHLA-looped = vertical recurrence, evolve_hla = per-NPC recurrent belief kernel with no `.research/` framing).

---

## 1. Paper Core Findings

### 1.1 The unifying framework (§2)

State at layer `l`, time `t`: `x_l^t ∈ R^d`, hidden state `S_l^t`. Three forms of `S`:
- **KV cache** (standard Transformer): `S = (K, V) ∈ R^{n×d}`, grows with sequence.
- **Linear attention state**: `S ∈ R^{d×d}`, fixed-size matrix.
- **Recurrent state**: `S ∈ R^d`, fixed-size vector.

Spatial + temporal update: `x_{l+1}^t = f(x_l^t, g(S_l^t, x_l^t))`. Latent CoT drops the decode-to-token step: `z_{t+1} = Transform(z_t, S_t)` (no `Decode()`).

### 1.2 Vertical recurrence — activation-based (§3.1)

Iteratively refine activations within fixed layers, gaining effective depth.

| Sub-family | Representative | Key idea |
|---|---|---|
| Architectural loops | Universal Transformer, CoTFormer, Recursive Transformer, AlgoFormer, Recurrent-Depth | Same layer(s) re-applied; ACT / early-exit / fixed-point halting |
| Hidden-state feedback | Coconut, CoTFormer | Last-layer hidden state re-injected as input position |
| Training-induced recurrence | Coconut, CODI, CCOT, PCCOT, System-1.5, Pause/Filler/Planning tokens, Lightthinker, Decomposes Reasoning | Standard architecture, training objective creates the loop |
| Training strategies | MIDAS, Looping-Inspired Reg, Stepwise Internalization, RELAY | Curriculum / regularization to induce or stabilize recurrence |

**Architectural convergence (Table 1):** modern designs converge on **Pre/Loop/Coda** structure (Prelude input encode → Loop blocks → Coda decode). Depth embeddings are deprecated (UT had them, recent models drop them). Dynamic stopping simplifies (UT's ACT → simple early-exit on `max_t Δh < ε` or fixed iterations).

### 1.3 Horizontal recurrence — hidden-state-based (§3.2)

Compress prior context into a fixed-size state, expand temporal capacity.

| Sub-family | Representative | Update rule |
|---|---|---|
| Linear-state | Linear Attn, RetNet, GLA, RWKV-6, HGRN-2, Mamba-2 | `S_t = S_{t-1} ⊙ M_t + k_t v_tᵀ` (associative scan) |
| Gradient-state | TTT, Titans, ATLAS, Gated Delta, Lattice, Moneta/Yaad/Memora | `S_t = α_t S_{t-1} − η_t ∇_S ℓ(S_{t-1}; k_t, v_t)` (online optimization) |
| Training-induced conversion | SUPRA, MOHAWK, Llamba, LoLCATs, Liger | Distill pretrained Transformer → recurrent student |

**Key unifying insight (§3.2.1, DeltaNet duality):** the linear-state closed-form `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t k_t v_tᵀ` is **mathematically equivalent** to one gradient-descent step on `L(S) = ½‖S k_t − v_t‖²`. This reframes temporal recurrence as iterative optimization — bridges horizontal (state) and vertical (depth) recurrence conceptually.

**Parallelization pattern:** chunk-wise — intra-chunk parallel gradient w.r.t. same initial state, inter-chunk sequential recurrence.

### 1.4 Mechanistic interpretability (§4)

Layer specialization theory: shallow = feature/syntactic, intermediate = reasoning sub-circuits (the "core of latent CoT"), deep = output refinement (but deep layers often degenerate — Pre-LN variance growth, attention matrices collapsing to rank-1).

Information-flow diagnostics: causal mediation analysis, "back attention" (top-down info flow), "Chain-of-Embedding" trajectory geometry distinguishes correct from incorrect answers (output-free self-eval).

Turing completeness: vanilla Transformer is TC under arbitrary precision + positional encoding + hard-max (Pérez 2019); achievable under constant precision (Li & Wang 2025); CoT enables fixed-depth TC (Li 2024, Qiu 2024).

### 1.5 Infinite-depth via diffusion (§5)

Spatial infinite reasoning: text diffusion models refine the **entire output in parallel** with bidirectional context (vs AR's irreversible left-to-right commitment).

| Family | Update | Cache |
|---|---|---|
| Masked diffusion (temporal-only) | `x_{t+1}^l(i) = f(x_t^l(i))` if mask, else unchanged | none |
| Masked diffusion + cache | `x_{l+1}^t = f_τ(x_l^t, S_l^t)`; selective cache refresh by confidence threshold τ | bidirectional KV (`dKV-Cache`, `dLLM-Cache`) |
| Embedding diffusion | `x_{t+1}^l = f(x_t^l, ε_t)` (Gaussian noise) | none |
| Hybrid AR-Diffusion | diffusion refinement + AR prefix caching | AR prefix + diffusion cache |

**Optimization-as-depth (§5.2):** all three "infinite-depth" strategies (Infini-attention compressive memory, TTT/Titans/Atlas fast weights, implicit fixed-point RNNs) embody one principle — **depth emerges from optimization over time**. The hidden state is a fast-weight layer refined per token; longer sequences = more optimization iterations = deeper effective reasoning.

---

## 2. Distillation — map to shipped code

This is the load-bearing section. **Every survey family is already represented.** This map exists to prevent the next research session from writing a duplicate note when grepping paper vocabulary only.

### 2.1 Vocabulary crosswalk (paper ↔ codebase)

| Paper term | Codebase term(s) | Where it ships |
|---|---|---|
| latent CoT / continuous thought | latent drafter, belief state, micro-belief, latent thought | `katgpt-core/src/speculative/`, `sense/`, `latent_thought/` |
| vertical recurrence / looped transformer | LT2 looped, training-free loop, ELT, LoopCoder-V2 | `forward_looped`, `LoopMode`, `tf_loop` feature |
| horizontal recurrence / hidden state | HLA, AHLA, Raven RSM, δ-Mem, MicroBelief | `katgpt-core/src/sense/`, Raven RSM slot memory |
| linear-state recurrence | HLA / AHLA second-order SK accumulator | Research 028; `AHLAState` |
| gradient-state recurrence / fast weights | LoRA reader-writer hot-swap, raw/lora (deterministic) | `LoraPair`, `dispatch_lora_merge` (riir-ai) |
| hidden state as fast weights / TTT | `latent_functor/reestimation.rs` coherence-driven re-estimation | riir-ai (the canonical vocabulary-mismatch example) |
| Pre/Loop/Coda | `LoopMode::{None, Count}`, hybrid SDPA+AHLA 1:4 | Plan 108, `.benchmarks/033_lt2_looped_goat.md` |
| early-exit / `max_t Δh < ε` | FPRM damped halting, LoopCoder-V2 gain-cost halting | Research 266, 282; `GainCostLoopHalter` |
| attractor / fixed-point | Attractor kernel (Family A), FPRM | Research 035, 266, 317 |
| infinite-depth via diffusion | D2F, ColaDLM, Nemotron TriMode, DMax SPD | Research 034, 010, 055; Plans 066, 109 |
| bidirectional KV for diffusion | bidirectional prefill + LoRA switch | **Plan 025** (partial; no dedicated dKV-Cache primitive) |
| pause / filler / planning tokens | salience tri-gate speak/silent/delegate | Research 281; `SalienceTriGate` |
| chain-of-embedding trajectory | depth-invariance diagnostic, CNA neuron attribution | Research 286, 053; `classify_chain` |
| layer specialization (shallow/mid/deep) | reasoning pack composition, cognitive branch | Pillar 8; riir-ai Research 161 |
| Coconut continuous-thought feedback | NextLat belief-state drafter | Research 192; Plan 217 |
| MUX/CCOT compressed reasoning | MUX-Latent context compression | Research 158; Plan 238 |
| explicit↔latent switch | SwiR switch-thinking | Research 241; Plan 275 |

### 2.2 Vertical recurrence — what we ship

- **LT2 Looped (Research 073, Plan 108, `.benchmarks/033`)** — T loops give rank-T state upgrade on AHLA; hybrid SDPA+AHLA 1:4 ratio is the flagship. Memory stays at 640 B/layer regardless of T (vs 5120 B naive). This is the survey's "Pre/Loop/Coda + simplified halting" converged design, shipped.
- **Training-Free Loop (Research 097, Plan 136)** — ODE-motivated damped Euler sub-stepping on a frozen checkpoint. Maps to survey's Recurrent-Depth family.
- **ELT (Research 273)** — Elastic Looped Transformers for any-time inference.
- **LoopCoder-V2 (Research 282, `.benchmarks/304`)** — gain/cost loop halting; G2 crowd-NPC savings 76.7%, G4 oscillation detection.
- **FPRM (Research 266)** — damped fixed-point halting.
- **ThoughtFold (Research 175)** — chain folding, inference-time.
- **NextLat (Research 192, Plan 217)** — belief-state latent dynamics; this IS Coconut's "continuous thought" distilled to inference.
- **Attractor Models (Research 035)** + MicroRecurrentBeliefState Family A (Plan 276, `.benchmarks/276`) — attractor-family recurrent kernel; **honest null result** documented (attractor hysteresis requires trained weights; random-init flip-flops — Plan 276 G2.1 FAIL is the canonical "modelless-unblock protocol §3.5 matters" reminder).

### 2.3 Horizontal recurrence — what we ship

- **HLA / AHLA (Research 028)** — higher-order linear attention, second-order SK accumulator, O(d·dv) state. This IS the survey's "linear-state recurrence" family, shipped before the survey.
- **Raven RSM (Research 006)** — O(1) routing slot memory.
- **δ-Mem / Dual-Pool Reachable Router (Research 024, 249; Plan 282)** — online associative memory, non-trapping router.
- **MicroBelief / LeakyIntegrator (Plan 276)** — Family C, byte-identical to `evolve_hla` (`katgpt-core/src/sense/reconstruction.rs`).
- **Topological Recurrent Belief (Research 242, Plan 276)** — Mozer et al. taxonomy (recurrence axis × tokens-per-step). This is the closest cousin to the new survey's recurrent-belief subset; verdict was revised Super-GOAT → GOAT after the HLA prior-art check (the canonical "grep shipped code, not just notes" lesson).
- **SSD Duality (Research 230)** — semiseparable state-space duality, Mamba-2 algebra.

### 2.4 Diffusion / infinite-depth — what we ship

- **D2F (Research 034, Plan 066)** — discrete diffusion forcing, block-causal attention, bidirectional within block. This is the survey's "hybrid AR-Diffusion" family, shipped.
- **ColaDLM (Research 010)** — continuous latent diffusion, d=16 optimal, 16 denoising steps.
- **Nemotron TriMode (Research 055)** — tri-mode AR/diffusion/mixed.
- **DMax SPD (Plan 109)** — aggressive parallel decoding for dLLMs.
- **D2F 3SR Warm-Start (Research 265, Plan 291)** — three-state reuse × LT2-looped × RCD fusion.
- **Bidirectional Prefill + LoRA Switch (Plan 025)** — bidirectional attention in prefill, zero-copy; **partial coverage** of survey's "MDM with cache" formula `x_{l+1}^t = f_τ(x_l^t, S_l^t)`.

### 2.5 Mechanistic interpretability — what we ship

- **Depth-Invariance Diagnostic (Research 286, `.benchmarks/306`)** — `classify_chain`, SIMD-vectorized, comparable to survey's "Chain-of-Embedding trajectory" tool.
- **CNA — Contrastive Neuron Attribution (Research 053)** — neuron-level attribution, aligns with survey's mechanistic-circuit literature.
- **Reasoning Pack composition (Pillar 8, riir-ai Research 146/149/151/161)** — the "layer specialization" thesis operationalized as per-NPC cognitive-branch composition.

---

## 3. Fusion opportunities (genuine gaps)

The survey surfaces **two narrow gaps** worth a future plan. Neither is Super-GOAT — both are incremental refinements of shipped machinery.

### 3.1 Bidirectional dKV-Cache primitive (gap)

The survey's §5.1.1 + Table 4 describe a **dedicated diffusion KV cache** with confidence-thresholded selective refresh:

```
S_l^{t+1}(i) = g_τ(x_l^t(i), S_l^t(i))   if c_l^t(i) ≥ τ   else unchanged
x_{l+1}^t   = f_τ(x_l^t, S_l^t)          (bidirectional block using cache)
```

**Our coverage:** Plan 025 ships bidirectional prefill attention, and Plan 066 (D2F) ships block-causal attention. Neither exposes the **confidence-thresholded selective cache refresh** as a standalone primitive. dKV-Cache (Ma 2025) and dLLM-Cache (Liu 2025) report 2–10× and up to 9.1× speedup respectively on LLaDA.

**Why this is a Gain, not GOAT:** our D2F path already gets most of the speedup from block-parallelism + causal-across-blocks KV reuse; the confidence-threshold refinement is a 1.5–2× additional speedup on top, gated behind whether we ever ship a real dLLM inference path (currently micro-scale research per Plan 066). Defer until dLLM inference is on the product roadmap.

### 3.2 "Depth from optimization over time" explicit framing for `reestimation.rs` (framing gap)

The survey's §5.2 unification — "depth emerges from optimization over time, hidden state = fast-weight layer refined per token" — is the **conceptual bridge** between vertical and horizontal recurrence. Our `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` ships exactly this pattern under the name "coherence-driven re-estimation scheduler when coherence < tau_reest" — DiPOD's "self-distillation when ELBO drifts" in codebase vocabulary.

**Gap:** no `.research/` note frames `reestimation.rs` in the survey's "fast-weight optimization over time" vocabulary. This is the canonical vocabulary-mismatch failure the skill warns about. A future note (or a one-paragraph addendum to an existing riir-ai functor note) closing this vocabulary gap would prevent the next paper-vocabulary-only grep from missing it.

**Why this is a Gain, not a plan:** it's a documentation/framing fix, not a code change. No feature flag, no benchmark.

### 3.3 Fusion idea (novelty TBD — NOT a Super-GOAT claim)

Survey's "gradient-state recurrence = optimization over time" × our `latent_functor/reestimation.rs` × our KARC reservoir (Plan 308/332, `KarcShard`) → a per-NPC primitive where the KARC reservoir's delay-basis ridge update is **driven by the functor's coherence signal** as the online "loss", unifying horizontal recurrence (reservoir state) with vertical (functor re-estimation trigger) under one optimization-over-time frame.

**Novelty TBD.** This crosses the §3.5 modelless-unblock boundary cautiously: the reservoir update is deterministic ridge regression (modelless), the functor coherence signal is latent (modelless), but the *coupling* (coherence as loss) needs a closed-form construction before it qualifies. Do NOT promote to Super-GOAT without running Q1–Q4 of the novelty gate. Track in `.issues/` if pursued.

---

## 4. Verdict

**Gain.**

**One-line reasoning:** Survey, not a new mechanism; value is the unifying taxonomy that maps the codebase's saturated latent-reasoning corpus to a single frame, plus two narrow gaps (dKV-Cache, reestimation-vocabulary bridge) neither of which is a new capability class.

**Why not Super-GOAT:**
- Q1 (no prior art?): **FAIL** — every family is shipped (see §2 map).
- Q2 (new class of behavior?): **FAIL** — the survey aggregates; it doesn't introduce a new capability.
- Q3 (product selling point?): **PARTIAL** — the bandwidth framing (≈40,960 bits vs ≈15 bits) is a nice Pillar 8 one-liner but not a new moat.
- Q4 (force multiplier?): **PARTIAL** — the map connects existing primitives, but the connections are mostly already documented in the individual notes.

**Why not GOAT:** no new primitive to benchmark, no provable gain over an existing approach. The map itself is the deliverable.

**Why not Pass:** the unification is genuinely useful given how scattered this corner of the corpus is (15+ existing notes across vertical/horizontal/diffusion/interpretability). Future research sessions that grep paper vocabulary only will miss shipped code; this note is the prophylactic.

---

## 5. What this note prevents (canonical failure modes averted)

1. **False Super-GOAT on the next "looped transformer" paper.** Any future paper in the vertical-recurrence family must check §2.2 before claiming novelty. We have LT2 (073), TF-Loop (097), ELT (273), LoopCoder-V2 (282), FPRM (266), ThoughtFold (175), MicroBelief (276), Attractor (035) — eight notes covering the family.
2. **False Super-GOAT on the next "hidden state" / "fast weights" paper.** Any future paper in the horizontal-recurrence family must check §2.3. We have HLA (028), Raven RSM (006), δ-Mem (024/249), MicroBelief (276), Topological Recurrent Belief (242), SSD Duality (230) — six notes. The DiPOD/reestimation vocabulary-mismatch lesson (Research 123) is the standing warning.
3. **False Super-GOAT on the next "diffusion reasoning" paper.** Any future paper in the diffusion family must check §2.4. We have D2F (034), ColaDLM (010), Nemotron TriMode (055), D2F-3SR (265/291), Plan 025 (bidirectional prefill) — five artifacts. The dKV-Cache gap in §3.1 is the only documented unshipped piece.
4. **Paper-vocabulary-only grep on "latent CoT".** §2.1 is the vocabulary crosswalk. Use it.

---

## 6. Action items

- [ ] **None in this session.** This note is the deliverable. No code, no feature flag, no benchmark — Gain verdict.
- [-] **Deferred:** dKV-Cache primitive (§3.1) — track when dLLM inference hits the product roadmap. Not Super-GOAT; not GOAT; not now.
- [-] **Deferred:** reestimation-vocabulary bridge addendum (§3.2) — documentation fix in riir-ai, not blocking.
- [-] **Deferred:** §3.3 fusion idea — file in `.issues/` if pursued; do NOT promote without Q1–Q4 novelty gate.

---

## TL;DR

Survey, not method. **Verdict: Gain.** Every latent-reasoning family it catalogs is already shipped in our corpus; this note is the anti-duplication map + vocabulary crosswalk that future research sessions grep before claiming novelty. Two narrow gaps (dKV-Cache, reestimation-vocabulary bridge) are documented but deferred — neither is a new capability class. Pillar 8 (Reasoning Pack) gets a cleaner selling-point one-liner (the bandwidth framing); no moat change. Commit and stop.
