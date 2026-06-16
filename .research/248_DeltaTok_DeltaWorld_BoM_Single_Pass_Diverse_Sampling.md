# Research 248: DeltaTok / DeltaWorld — Delta-Token Compression + Best-of-Many Single-Pass Diverse Sampling

> **Source:** [A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](https://arxiv.org/pdf/2604.04913) — Kerssies, Berton, He, Yu, Ma, de Geus, Dubbelman, Chen (Amazon + TU Eindhoven + JHU), arXiv:2604.04913, Apr 2026
> **Code/weights:** deltatok.github.io
> **Date:** 2026-06-16
> **Status:** Done
> **Related Research:** 192 (NextLat belief residual = delta encoding), 215 (ECHO env prediction), 242 (MicroRecurrentBeliefState — `evolve_hla` prior-art lesson), 245 (Mirage latent spatial memory — PASS), 018 (Free Transformer Z-sampling — closest BoM cousin), 058 (GRAM SDE noise), 079 (EqR noise injection)
> **Related Plans:** 276 (MicroRecurrentBeliefState — ships the delta-encoding kernel), 277 (Temporal Derivative Kernel — temporal delta cousin), 247 (EnvPredictorPruner — inference-time ECHO), 281 (this paper's plan — BoM single-pass sampling)
> **Cross-ref (riir-train):** `riir-train/.plans/272_echo_env_prediction_lora_training.md`, `riir-train/.benchmarks/288_echo_bomber_arena_v4_2layer.md` — **the ECHO failure analysis explicitly identifies "delta-token encoding" as the fix; this paper is the literature backup for that fix**
> **Classification:** Public

---

## TL;DR

DeltaTok compresses the VFM-feature difference between consecutive video frames into a single continuous "delta" token (1024× token reduction at 512×512), and Best-of-Many (BoM) training samples K Gaussian noise queries per step, supervising only the closest prediction to ground truth, which at inference yields **K diverse plausible futures in a single forward pass**. DeltaWorld (the combination) achieves 35× fewer params and 2000× fewer FLOPs than Cosmos while producing better best-of-20 predictions on dense forecasting.

**For katgpt-rs (modelless, inference-time):** the paper splits cleanly into two primitives. (1) **Delta-token compression of state change is already shipped** — `evolve_hla` (`sense/reconstruction.rs:623`) is a gated additive delta update of the 8-dim HLA state; `MicroRecurrentBeliefState` (Plan 276) generalizes it into a learned `(s_{t-1}, x_t) → s_t` kernel (attractor + leaky families), structurally identical to DeltaTok's encoder `z_t = g(x_{t-1}, x_t)`. NextLat's residual MLP `ĥ = f(h,x) + h` (Research 192) IS a delta encoding. The spatial-token-reduction part of DeltaTok (H×W → 1 token) is video-specific and has no 2D-top-down game analog — our belief vectors are already single tokens. (2) **Best-of-Many single-pass K-hypothesis sampling is the novel inference primitive** — no shipped code does stochastic multi-hypothesis evolution of `MicroRecurrentBeliefState`. The closest cousins (Free Transformer Z-sampling Research 018 = hypothetical; DDTree branching = sequential K forwards, not single-pass; GRAM SDE Research 058 = distillation-time) do not cover single-pass K-query batched sampling on a per-entity belief kernel.

**Verdict: Gain.** The delta-token compression is prior art (shipped via `evolve_hla` / `MicroRecurrentBeliefState` / NextLat residual). The BoM single-pass sampling is a novel *mechanism* but not a new *capability class* (DDTree already provides decision-level diversity). The paper's biggest leverage for our trio is **literature backup for the ECHO T1 redesign** (riir-train benchmark 288 already identified "delta-token encoding + dedicated obs head" as the fix for ECHO's env-prediction/action-vocab conflation failure). Plan 281 creates the BoM primitive as an opt-in `MicroRecurrentBeliefState` variant; the ECHO training fix is noted here as a riir-train cross-ref and is out of scope for this workflow.

---

## 1. Paper Core Findings

### 1.1 DeltaTok — compress frame difference into one token

The tokenizer is a continuous autoencoder (not a VAE). The **encoder** takes both the previous and current VFM (DINOv3) feature maps and produces a single delta token:

```
z_t = g(x_{t-1}, x_t, z_init) ∈ R^D          (Eq. 8)
```

The **decoder** reconstructs the current frame by transforming the previous frame using the delta token:

```
x̂_t = h(x_{t-1}, z_t)                         (Eq. 9)
```

Both `g` and `h` are ViT-B stacks. Trained separately (50K iters, MSE loss `‖x_t − x̂_t‖²`, batch 1024) before the world model. The first frame is prepended with a black frame so `z_1` encodes absolute features. Key property: **a single delta token suffices because consecutive frames differ in structured, low-dimensional ways**; when temporal redundancy is low, the token reverts to absolute compression.

This collapses video from 3D spatio-temporal (H×W×T tokens) to 1D temporal (T tokens) — **1024× token reduction at 512×512**.

### 1.2 Best-of-Many (BoM) — single-pass diverse generation

The predictor `f` cross-attends from a single learnable query `q` to the context. To make the model *generative* (multi-hypothesis) without diffusion's iterative cost, BoM draws K noise queries and supervises only the best:

```
q_k ~ N(μ, Σ),  k = 1..K
x̂^k_{t+1,h,w} = f(q_k, X_{1:t}, T_{1:t}, τ_{t+1}, h, w)
k★ = argmin_k Σ_{h,w} ℓ(x_{t+1}, x̂^k_{t+1})
L_BoM = Σ_{h,w} ℓ(x_{t+1}, x̂^{k★}_{t+1})        (Eq. 4)
```

At inference, different noise queries yield diverse futures **in a single forward pass** (no iterative denoising). K=256 during training; 20 samples at eval.

### 1.3 DeltaWorld — the combination + results

The predictor operates entirely on delta-token sequences `Z_{1:t} = (z_1, ..., z_t)`, predicting `ẑ_{t+1} = f(q_k, Z_{1:t}, T_{1:t}, τ_{t+1})`. The BoM loss is computed in delta-token space (no decode needed during training). The decoder is applied separately to recover spatial features.

**Results (Table 3, dense forecasting benchmark):**
- DeltaWorld-0.3B: **50.1 / 46.7** VSPAW mid mIoU (best / mean) vs Cosmos-12B **47.7 / 45.5** — better best AND better mean, with **35× fewer params and 2000× fewer FLOPs**.
- The gap between best and mean is consistently larger for DeltaWorld than Cosmos → more meaningful sample diversity.
- Mean mIoU recovers to the discriminative baseline level → "predicting no change preserves the previous frame" is a natural strong prior.

### 1.4 What the training actually is (→ riir-train if we cared)

- DeltaTok tokenizer training (autoencoder MSE, 50K iters, batch 1024, AdamW lr=1e-3).
- DeltaWorld predictor training (BoM objective, smooth L1 loss β=0.1, 300K iters, batch 1024, K=256 noise queries, AdamW lr=1e-4).
- 2D RoPE → 1D RoPE simplification (single token per frame, no spatial axes).
- Noise query distribution `N(0, 0.02²I)`.
- Task heads (linear seg head, DPT-style depth head) trained separately on frozen VFM features.

All training-side → riir-train. The inference-time artifacts (frozen tokenizer + frozen predictor + BoM sampling) are what distills to katgpt-rs / riir-ai.

### 1.5 Limitations (paper §D)

- **No explicit distributional objective.** BoM lacks diffusion's principled data-distribution connection; coverage is bounded by K. No mechanism encourages diverse query-space utilization.
- **Error accumulation.** Delta decoding compounds errors across rollout steps. Mitigation: tokenizer operates on its own reconstructions (sequential relative to decoded, not parallel from ground truth).

---

## 2. Distillation

### 2.1 The transferable primitives (two, split by novelty)

**Primitive A — Delta-token compression of state change.** The insight: encoding only the *change* between consecutive latent states requires less information than re-encoding the full state, and "predict no change = preserve prior" is a natural regularizer. This is a latent-to-latent operation (no decode to raw in the compression step).

**Primitive B — Best-of-Many single-pass K-hypothesis sampling.** The insight: inject K noise queries at a single attention/kernel site, evaluate K outputs in one batched forward, and (at training) supervise only the best. At inference, K queries → K diverse outputs with no iterative cost.

### 2.2 Prior-art check — Primitive A is already shipped (the decisive part)

The mandatory two-layer novelty check (notes + shipped code, per the Research 242 `evolve_hla` lesson) found Primitive A covered in **four** places:

| Paper concept | Shipped prior art | Match |
|---|---|---|
| Delta encoder `z_t = g(x_{t-1}, x_t)` | `MicroRecurrentBeliefState::step(s_{t-1}, x_t)` (Plan 276, `micro_belief/attractor.rs`) — `state[i] = clamp(2·σ(W_s·s + W_x·x + b) − 1, ±clamp)`. A learned `(state, input) → next_state` kernel. **Structurally identical to DeltaTok's encoder.** | ✅ Encoder ships |
| Additive delta with clamp | `ReconstructionState::evolve_hla()` (`sense/reconstruction.rs:623`) — `hla[i] = (hla[i] + clamped_delta).clamp(-1, 1)` where `clamped_delta = clamp(lr·(normalized − half_total)·scale, max_delta)`. Family C leaky integrator. | ✅ Additive delta ships |
| Residual = delta | NextLat belief drafter (Research 192, Plan 217) — `ĥ_{t+1} = f_ψ(h_t, x_{t+1}) + h_t`. The residual `f_ψ` IS the delta. | ✅ Residual delta ships |
| Temporal derivative as signal | Plan 277 Temporal Derivative Kernel — dual fast/slow surprise signal from temporal deltas. | ✅ Temporal delta cousin ships |
| "Predict no change = preserve prior" | Implicit in all three above: zero delta / zero residual / zero `f_ψ` ⇒ state unchanged. `evolve_hla`'s `total < 1e-8` early-return is the explicit no-op guard. | ✅ Prior ships |

**The spatial-token-reduction part of DeltaTok (H×W feature map → 1 token) does not transfer.** Our game AI has no spatial feature maps per NPC — the belief vector (8-dim HLA, or D-dim `MicroRecurrentBeliefState`) is already a single token. DeltaTok's 1024× spatial compression is solving a problem we don't have. The encoder mechanism itself (`(prev, curr) → delta`) is what transfers, and it is already shipped.

**Conclusion for Primitive A:** prior art blocks novelty. No new primitive to ship.

### 2.3 Prior-art check — Primitive B (BoM) is the novel piece

| Paper concept | Shipped prior art | Match |
|---|---|---|
| K noise queries → K diverse outputs in ONE forward pass | **None shipped.** | ❌ Novel mechanism |
| Multi-sample diverse generation (sequential) | DDTree branching — explores K branches but via K sequential forward passes through the model, not a single batched pass. Diversity comes from the tree structure (logit-based), not from noise injection at a kernel site. | ⚠️ Capability cousin, different mechanism |
| Noise injection for diversity (hypothetical) | Free Transformer Z-sampling (Research 018) — "sample multiple Z values, generate with each, pick the best" — explicitly flagged as hypothetical ("If a Free Transformer base model becomes available"). | ⚠️ Closest cousin, unshipped |
| SDE noise injection (distillation) | GRAM (Research 058), ELF (Research 044/079), EqR (Research 079) — all training/distillation-time noise injection, not inference-time single-pass K-hypothesis sampling. | ⚠️ Different regime |
| Best-of-N selection | P-UCB sketch sampler (Benchmark 039), `SketchSampler` ε-greedy + diversity injection — selects best from a population, but population is generated sequentially. | ⚠️ Selection cousin, not generation |

**Conclusion for Primitive B:** the specific mechanism (inject K noise vectors at a single kernel/attention site, batch-evaluate K outputs in one forward, select best for training / use all for diverse inference) has **no direct shipped prior art** in our stack. The closest cousins either (a) do sequential K-pass diversity (DDTree), (b) are hypothetical (Free Transformer Z), or (c) operate in a different regime (distillation-time SDE).

### 2.4 Fusion — BoM × MicroRecurrentBeliefState (Plan 276)

**The combination:** add a `sample_k_states(&self, s_prev, x, queries: &[QueryState; K]) -> [BeliefState; K]` method to `MicroRecurrentBeliefState`. Each query `q_k ~ N(0, σ²I)` is concatenated/added to the kernel input; the kernel is evaluated K times in a single batched matvec (SIMD-friendly: K-row matrix multiply). Returns K diverse next-belief-states.

**What this produces that no incumbent alone can:**

| Incumbent alone | What it can't do | What the fusion adds |
|---|---|---|
| `MicroRecurrentBeliefState::step()` (deterministic) | Only ONE next-state per tick. NPC has no uncertainty about its own belief evolution. | K diverse plausible next-beliefs per tick in one batched call → NPC "imagines" K futures |
| DDTree (decision-level diversity) | Explores K actions but against ONE predicted world state. Belief is deterministic; only the action branches. | Belief-level diversity — NPC considers K possible world evolutions, THEN plans actions against each |
| NextLat residual drafter | Drafts K tokens sequentially via recursive MLP composition. Each draft is a separate forward. | Single-pass K hypotheses — batched, no sequential composition |

**Capability increment (over shipped `MicroRecurrentBeliefState`):** stochastic multi-hypothesis belief evolution. Currently an NPC's belief state evolves deterministically (`s_{t+1} = f(s_t, x_t)`). With BoM, it evolves into a *distribution* over K hypotheses (`s_{t+1}^{(k)} = f(s_t, x_t, q_k)`). The NPC can then plan against the most threatening/plausible hypothesis (minimax over the K beliefs) or against the mean (risk-averse).

**Honest capability-class assessment:** this is a refinement, not a new class. DDTree already gives the NPC diverse futures at the action level. Moving diversity from action-level to belief-level is a meaningful architectural shift (beliefs inform actions, so belief diversity is "upstream" of action diversity), but it is not a capability no competitor can match — a competitor with enough compute could run K separate belief kernels. The novelty is doing it *cheaply* (single batched matvec vs K forwards), which is a perf claim, not a capability claim.

### 2.5 Connection to the ECHO failure (riir-train cross-ref)

**`riir-train/.benchmarks/288_echo_bomber_arena_v4_2layer.md`** is the smoking gun. ECHO (env-prediction auxiliary loss, `riir-train/.plans/272`) FAILED GOAT (−6.0pp vs baseline). The root-cause analysis states verbatim:

> "The fix is the T1 design: encode game-state deltas as a **separate small observation vocabulary** (5–6 tokens: position-delta, HP-delta, resource-delta, etc.) projected from a dedicated `obs_head`, not reusing the policy head's logits over the action vocabulary."

**DeltaTok is the literature backup for exactly this fix.** DeltaTok's encoder `z_t = g(x_{t-1}, x_t)` compresses state change into a dedicated token space, separate from any policy/action head. The paper's ablation (Table 2, step 2→3) shows delta compression beats full-frame compression precisely because "the delta captures only the information needed to transform x_{t-1} into x_t."

This is a **riir-train** insight (training the ECHO obs head with delta-token encoding). It is out of scope for this workflow per the skill's training-redirect rule. Recorded here as a cross-reference so the riir-train ECHO redesign (Plan 272 T1) can cite this paper as the methodological basis.

---

## 3. Verdict

**Tier: Gain.**

**One-line reasoning:** Delta-token compression is already shipped (`evolve_hla` / `MicroRecurrentBeliefState` / NextLat residual); BoM single-pass K-hypothesis sampling is a novel *mechanism* but not a new *capability class* (DDTree already provides diversity); the paper's biggest leverage (ECHO T1 delta-token fix) is training-side → riir-train.

**Novelty gate (honest, post prior-art check — applying the Research 242 `evolve_hla` lesson):**

| Gate | Question | Honest answer |
|---|---|---|
| **Q1 Novelty** | No prior art in shipped code? | **FAILS for Primitive A** (delta-token compression shipped via `evolve_hla` / `MicroRecurrentBeliefState` / NextLat). **PASSES for Primitive B** (BoM single-pass K-query sampling has no shipped equivalent). Net: mixed → not Super-GOAT. |
| **Q2 New capability class** | New behavior, not better numbers? | **FAILS.** BoM gives belief-level diversity, but DDTree already gives decision-level diversity. Moving diversity upstream (belief → action) is a refinement, not a new class. A competitor with K× compute could replicate. |
| **Q3 Selling point** | "Our NPCs/systems do X no competitor can"? | **WEAK.** "NPCs imagine K futures per tick in one forward pass" is a perf claim (single-pass vs K-pass), not a capability claim. |
| **Q4 Force multiplier** | Connects to ≥2 existing pillars? | **PASSES.** Connects to MicroRecurrentBeliefState (Plan 276), NextLat (192/217), ECHO (215/247), DDTree, Temporal Derivative Kernel (277), Freeze/Thaw (version the noise query distribution as a snapshot). But Q4 alone ≠ Super-GOAT. |

**Not Super-GOAT (Q1 mixed, Q2/Q3 fail) → not GOAT (no provable quality gain over DDTree diversity; the latency gain of single-pass vs K-pass is provable in principle but unverified for our small belief kernels where K forwards are already cheap) → Gain.**

**Per skill Gain protocol:** research note + plan (Plan 281), behind feature flag, GOAT gate before any promotion.

### Routing

- **katgpt-rs (public, this repo):** Plan 281 — `BoMSampler` trait + `MicroRecurrentBeliefState::sample_k_states()` opt-in variant. Generic primitive, no game semantics.
- **riir-ai (private):** Game-AI integration deferred — if Plan 281's GOAT gate passes (K-hypothesis belief evolution improves planning quality on a benchmark), then riir-ai wires it into NPC tick dispatch. No guide created (Gain, not Super-GOAT).
- **riir-train (private, out of scope):** ECHO T1 redesign (`riir-train/.plans/272`) should cite this paper as the methodological basis for the delta-token observation head. Not actioned in this session.

---

## 4. Closest cousins (for the fusion protocol record)

Across both repos, both layers (notes + code):

- `katgpt-rs/.research/192_NextLat_Belief_State_Latent_Dynamics.md` + `katgpt-rs/.plans/217_nextlat_belief_state_drafter.md` — **closest cousin.** NextLat's residual MLP `ĥ = f(h,x) + h` IS delta encoding; the drafter is deterministic. BoM adds stochastic K-hypothesis to the same residual structure.
- `katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md` + `katgpt-rs/.plans/276_micro_recurrent_belief_state.md` — **the prior-art-check lesson.** `evolve_hla` ships delta encoding; Plan 276 generalizes it. BoM would be an opt-in stochastic variant of the same kernel.
- `katgpt-rs/.plans/277_temporal_derivative_kernel.md` — temporal delta as a surprise signal (fast/slow dual). Cousin on the delta axis; orthogonal on the sampling axis.
- `katgpt-rs/.research/018_The_Free_Transformer_Latent_Injection.md` — **closest BoM cousin.** Free Transformer Z-sampling = "sample multiple Z, pick best" — explicitly flagged hypothetical. BoM is the realized, trainable version.
- `katgpt-rs/.research/215_ECHO_Environment_Prediction_Inference_Time.md` + `katgpt-rs/.plans/247_echo_env_predictor_pruner.md` — ECHO inference-time env prediction. DeltaTok's encoding is the literature fix for ECHO's training failure (riir-train benchmark 288).
- `katgpt-rs/.research/245_Latent_Spatial_Memory_Video_World_Models.md` — Mirage (PASS). Same "video world model" domain; Mirage's latent spatial memory is already shipped as `SpatialMemory`. Neither video paper's spatial mechanism transfers to 2D top-down.
- `riir-train/.plans/272_echo_env_prediction_lora_training.md` + `riir-train/.benchmarks/288_echo_bomber_arena_v4_2layer.md` — **the ECHO failure that explicitly calls for delta-token encoding.** Training-side; cross-ref only.

---

## 5. What does NOT transfer

- **Spatial token reduction (H×W → 1 token).** Video-specific. Our belief vectors are already single tokens.
- **DeltaTok tokenizer training (autoencoder MSE).** Training-side → riir-train.
- **DeltaWorld predictor training (BoM objective, K=256).** Training-side → riir-train.
- **DINOv3 VFM feature space.** Tied to the vision backbone. Our latent space is the belief vector, not VFM features.
- **Diffusion-baseline comparison (Cosmos).** We have no diffusion world model to beat.
- **Pinhole-camera / z-buffer readout (Mirage-style).** 3D perspective; no 2D-top-down analog (already established in Research 245).

---

## 6. Open questions / risks

- **R1 — Is single-pass BoM actually cheaper than K DDTree forwards for our small kernels?** For an 8-dim HLA kernel, K=8 noise queries is an 8-row matvec (~64 FLOPs) vs 8 separate `evolve_hla` calls (~8×20 FLOPs each). The gain is marginal at this scale. BoM's leverage grows with kernel dimension and K, but our belief kernels are deliberately small (plasma-tier budget). **The GOAT gate must measure actual latency, not assume it.**
- **R2 — Does belief-level diversity improve planning quality?** This is the actual GOAT question. DDTree already gives action-level diversity. If planning against K diverse beliefs doesn't improve arena win rate / HL score over planning against 1 deterministic belief + K diverse actions, BoM is not worth the complexity. **Must benchmark before promotion.**
- **R3 — Noise query distribution provenance.** Where does `N(0, 0.02²I)` come from for our belief space? The paper tunes this for DINOv3 features. For our 8-dim HLA space (range `[-1, 1]`), `σ=0.02` may be too small. Needs calibration or bandit-tuned σ per NPC class.
- **R4 — Sync boundary.** The K belief hypotheses are local (think-brain). Only the selected belief (or the mean) projects to synced scalars via the existing bridge. Never sync the K-vector distribution — that would leak the noise query distribution. Same rule as Research 242 R3.
- **R5 — ECHO fix is training-side.** This paper provides the *method* for the ECHO T1 redesign, but implementing it requires riir-train work (train a delta-token obs head). Not actioned here.

---

## TL;DR

DeltaTok/DeltaWorld (arXiv:2604.04913) compresses consecutive-frame VFM feature differences into single delta tokens and uses Best-of-Many training (K noise queries, supervise best) to generate K diverse futures in one forward pass — 35× fewer params, 2000× fewer FLOPs than Cosmos, better best-of-20 predictions. **Prior-art check (notes + code, per the Research 242 `evolve_hla` lesson) shows delta-token compression is already shipped** via `evolve_hla` (additive delta), `MicroRecurrentBeliefState` (Plan 276, learned `(s_{t-1}, x_t) → s_t` kernel = DeltaTok's encoder), and NextLat's residual MLP (Research 192). The spatial-token-reduction part is video-specific with no 2D game analog. **The only novel inference primitive is Best-of-Many single-pass K-hypothesis sampling** — no shipped code does stochastic multi-hypothesis evolution of `MicroRecurrentBeliefState` (closest cousins: Free Transformer Z-sampling = hypothetical; DDTree = sequential K-pass). **Verdict: Gain** — Q1 mixed (delta=shipped, BoM=novel), Q2/Q3 fail (capability exists via DDTree, just cheaper mechanism). Plan 281 creates `BoMSampler` as an opt-in `MicroRecurrentBeliefState` variant behind a feature flag; GOAT gate requires proof that K-hypothesis beliefs improve planning quality over deterministic beliefs + DDTree action diversity. **Cross-ref to riir-train:** this paper is the literature backup for the ECHO T1 redesign (benchmark 288 explicitly identifies "delta-token encoding + dedicated obs head" as the fix for ECHO's env-prediction/action-vocab conflation failure) — training-side, out of scope for this workflow.
