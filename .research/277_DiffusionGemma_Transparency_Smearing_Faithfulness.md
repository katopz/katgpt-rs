# Research 277: How Transparent is DiffusionGemma? — Smearing, Opaque Serial Depth, and the Top-k Bottleneck

> **Source:** Engels, McDougall, Chughtai et al., "How Transparent is DiffusionGemma?", Google DeepMind, [arXiv:2606.20560](https://arxiv.org/abs/2606.20560), 18 Jun 2026.
> **Date:** 2026-06-21
> **Status:** Done
> **Related Research:** 244 (Faithfulness / Cognitive Integrity), 158 (MUX), 248 (BoM), 228 (RCD), 192 (NextLat), 242 (MicroBelief), 250 (Self-Advantage / latent policy improvement)
> **Related Plans:** 278 (FaithfulnessProbe — the host for the distilled primitive), 178 (MUX), 281 (BoM), 258 (RCD), 276 (MicroBelief)
> **Cross-ref (riir-ai):** `.research/129_Cognitive_Integrity_Layer_Guide.md` (the selling-point doc this paper validates), `.research/123_Latent_Functor_Runtime_Guide.md` (re-estimation = retroactive self-correction)
> **Classification:** Public

---

## TL;DR

This is an **interpretability analysis paper**, not a new mechanism. DeepMind audits DiffusionGemma (a text-diffusion LLM) along four axes — opaque serial depth, variable transparency, monitorability, algorithmic transparency — and concludes that **forcing the latent self-conditioning matrix through a top-k / top-p token bottleneck between denoising steps preserves performance while collapsing opaque serial depth from 28.6× to 1.1× that of the autoregressive counterpart**. They also name and visualize four diffusion-specific phenomena: **non-chronological reasoning, token smearing, sequence smearing, and intermediate-context reasoning**.

**Distilled for katgpt-rs (modelless, inference-time):** the paper is *confirmatory*, not novel relative to our codebase. Every one of its load-bearing mechanisms already ships under codebase vocabulary:

| Paper mechanism | Shipped cousin (notes + code) | Match |
|---|---|---|
| Top-k / top-p token bottleneck between denoising steps (the central design prescription) | `MicroRecurrentBeliefState::project_to_scalars` bridge (Plan 276, `micro_belief/types.rs`) — 8-dim HLA latent → 5 committed scalars (valence/arousal/desperation/calm/fear); `evolve_hla` (`sense/reconstruction.rs`) is the recurrent carry | ✅ Exact pattern |
| Token smearing (one token's probability mass spread across adjacent positions) | `BoMSampler::sample_k_states` (Plan 281, `micro_belief/bom.rs`) — K Gaussian noise queries at one kernel site → K diverse next-belief-states in one batched matvec. Stochastic version of the same "hypothesis spread across positions" phenomenon. | ✅ Stochastic analog |
| Sequence smearing (multiple semantically distinct candidate sequences in superposition) | **MUX Multiplexed Latent Reasoning** (Plan 178, Research 158, `src/mux/`) — single latent token encodes K reasoning steps as geometrically-weighted superposition of one-hot vectors in the vocabulary simplex, losslessly recoverable via `mux_demux`. Deterministic version of sequence smearing. | ✅ Deterministic analog |
| Retroactive self-correction (model commits answer, then refines reasoning, then revises answer) | `latent_functor::reestimation::ReestimationScheduler` + `cce_runtime::reestimation_trigger::CceReestimationTrigger` (riir-ai) — "when coherence < `tau_reest`, re-estimate functors". This is the DiPOD canonical-miss pattern documented in the skill. | ✅ Exact pattern, different vocab |
| Early response length prediction (model predicts its own padding-token CDF before deciding content) | Zone attention routing: `dot(NPC_preference, zone_embedding) → sigmoid → "how much does this NPC care about this zone"` decides computational budget allocation *before* tactical computation. Plus SenseLoD (`sense/lod.rs`) — predicts LOD level before full perception. | ✅ Capability cousin |
| Monitorability (CoT → downstream monitor predicts behavior) | **FaithfulnessProbe** (Plan 278, Research 244) + **Cognitive Integrity Layer** (riir-ai `.research/129`) — already ships exactly this: causal intervention diagnostic that audits whether injected latent memory is faithfully reflected in behavior. GOAT 4/4 passed. | ✅ Exact capability |
| Intermediate-context reasoning (latent tokens causally necessary but absent from final output) | The HLA belief state itself — the 8-dim latent drives behavior but only 5 scalars cross the sync boundary. The "intermediate context" is the 3 latent dims that never sync. By design. | ✅ Architectural |

**The one extractable primitive** is a *diagnostic extension* to FaithfulnessProbe: a **smear classifier** that distinguishes (a) coherent single hypothesis, (b) benign token smear (positional uncertainty), (c) potentially-unfaithful sequence smear (multi-hypothesis superposition requiring disambiguation). This extends the existing binary faithful/unfaithful probe to ternary. See §2.2.

**Verdict: Gain.** The paper is a strong external validation of our architecture (top-k scalar bridge, coherence-triggered re-estimation, FaithfulnessProbe) and supplies a vocabulary ("smearing", "opaque serial depth") we should adopt in the Cognitive Integrity Layer doc. It does not introduce a new capability class.

---

## 1. Paper Core Findings (verified by reading the full PDF)

### 1.1 Architecture: DiffusionGemma's denoising loop

DiffusionGemma adapts Gemma 4 26B A4B into a **text diffusion** model. Inference = setup + T iterations of a denoising loop. Each step takes a canvas `o_{t-1} ∈ [V]^C` (C token positions) plus a **self-conditioning matrix** `S_{t-1} ∈ ℝ^{C×d_model}` and produces updated `o_t`, `S_t`. The self-conditioning matrix is the latent carry between steps:

```
S_t = softmax(ℓ̂_t) · W_E      (Eq. 1 — biased towards embedding directions)
```

This is the paper's "information bottleneck between denoising steps" — the analog of our HLA latent carry between ticks.

### 1.2 Opaque serial depth (§2)

**Definition** (Brown-Cohen et al., 2026): the longest serial computation path in the model's boolean circuit that does not pass through an interpretable node.

**Key result (Table 1):**
- If `S_t` is treated as uninterpretable: opaque serial depth = **608,016** (28.6× Gemma 4's 21,235).
- If `S_t` is treated as interpretable (via top-k projection): opaque serial depth = **23,571** (1.1× Gemma 4).

**Asymptotically:** uninterpretable = `O(T·L·(log N + log D + log C))`; interpretable = `O(L·(log N + log D + log C))` — the `T` factor disappears.

**The entire argument for latent-reasoning transparency rests on whether the bottleneck is interpretable.**

### 1.3 Variable transparency — the top-k/top-p ablation (§3)

**Method:** replace `S_t = softmax(ℓ̂_t)·W_E` with `S'_t = W_E · softmax(f(ℓ̂_t))` where `f` is either:
- `f_k`: keep top-k logits, redistribute rest uniformly (Algorithm 1).
- `f_p`: keep logits with prob ≥ p, redistribute rest uniformly.

**Result (Fig. 2):** `k=8` and `p=0.03` attain baseline performance on N2C / LCB / AMC-AIME / IMO / GPQA. **Restricting the latent carry to 8 tokens per position preserves capability.**

**Token identity analysis (Fig. 3):** under `p>0.03`, >85% of retained tokens are either (a) the final token at that position, (b) a final token at an adjacent position (±10), or (c) semantically similar (top-50 cosine neighbor) to a final token. **The latent carry is mostly "guesses for nearby final tokens" — interpretable.**

### 1.4 Monitorability (§4)

Adapts Guan et al. (2025)'s monitorability evals (intervention / process / outcome-property). **DiffusionGemma is similarly monitorable to Gemma 4** (Fig. 4), despite shorter CoT (Fig. 5).

### 1.5 Algorithmic transparency case studies (§5) — the phenomena

| Phenomenon | Description | Interpretability concern |
|---|---|---|
| **Early response length prediction** (§5.1.1) | After step 0, padding-token probability as a function of position forms a CDF; expected length `E_t[L] = Σ_i (1 - p^t_pad(i))` is accurate by step 2. | Low — interpretable as length budget. |
| **Retroactive self-correction** (§5.1.2) | Model commits answer "9" at step 4, fills in reasoning steps 5–7, then corrects answer to "8" with higher confidence. Impossible for AR models. | Medium — earlier tokens causally depend on later tokens. |
| **Non-autoregressive code generation** (§5.1.3) | "Skeleton-first": model commits function signature + core loop logic, *then* backfills docstring/variable names that appear earlier in the source. | Medium — commit order ≠ source order. |
| **Token smearing** (§5.2.1) | A single token's probability mass spreads across multiple adjacent positions simultaneously (e.g. "bracket" predicted at positions 12–18 before converging to 15). | Low — interpretable as "model knows the token, not the position". |
| **Sequence smearing** (§5.2.2) | Two or more semantically distinct candidate sequences held in superposition with probability mass split across both (e.g. "answer is 9" vs "answer is 12" coexist at step 3 before one wins at step 4). | **High** — "the model can perform a kind of beam search, holding multiple multi-token chunks in prediction space at once." The authors explicitly flag this as concerning and note they did *not* find semantically distinct sequence smearing, only spacing-related. |
| **Intermediate context reasoning** (§5.3) | Fibonacci-with-"3"→"Gold" replacement: model generates "3" as a scratch token, uses it to compute subsequent terms, then overwrites it with "Gold". The "3" is causally necessary but absent from final output. | Medium — final output is insufficient for understanding the algorithm. |

### 1.6 Open problems (§7) — research agenda

The paper's §7 lists ~25 open problems. The ones most relevant to us:
- §7.2.1 Adapting Lens and Patchscope techniques (we have FaithfulnessProbe attribution).
- §7.2.4 When do latent vector interventions fail? (We have Plan 278 G3 saturation results.)
- §7.3.6 Using top intermediate tokens for monitoring (we have FaithfulnessProbe + Cognitive Integrity Layer).
- §7.4 Model organism research (finetuning to hide reasoning from final answer — directly relevant to our anti-cheat + Cognitive Integrity Layer).

---

## 2. Distillation

### 2.1 The transferable insight (stripped of DiffusionGemma specifics)

**Principle:** *Latent reasoning is monitorable iff the latent carry between stages is forced through a low-rank interpretable bottleneck (top-k or top-p projection onto a vocabulary/direction basis), and the residual non-bottlenecked computation is bounded by a per-stage opaque-depth budget.*

This principle is **architecture-agnostic**. It applies to:
- Text diffusion (the paper's domain).
- Loop-layer transformers (LT2 — Plan 108).
- Continuous-CoT (NFCoT — Plan 229).
- **Per-NPC recurrent belief state evolution** (HLA / MicroRecurrentBeliefState — our domain).

For us: between two `project_to_scalars` calls (HLA → 5 synced scalars), the NPC's latent state can drift through `n_functor_applications × dim(latent_space)` opaque computation. The "opaque serial depth" of a single NPC tick is bounded by `(cgsp_cycles_per_tick) × (hla_dim)`. The top-k bottleneck is our bridge function. **We already ship the paper's prescription.**

### 2.2 The one extractable primitive — Smear Classification for FaithfulnessProbe

The paper's §5.2 names two distinct phenomena that our current **FaithfulnessProbe** (Plan 278) treats as a single binary "faithful / unfaithful" signal:

- **Token smearing** — mass on one token spread across adjacent *positions*. Benign: model knows the token, not the position. Faithful.
- **Sequence smearing** — mass split across *semantically distinct tokens at the same position*. Potentially unfaithful: model is hiding a multi-hypothesis computation that may not survive commitment.

**Fusion:** FaithfulnessProbe (Plan 278) × MUX (Plan 178, superposition weights) × this paper's smear taxonomy = **ternary smear classifier**:

```rust
pub enum SmearClass {
    /// One dominant hypothesis. Mass concentrated on single token at single position.
    CoherentSingle,
    /// Token smearing — mass on one token spread across adjacent positions.
    /// Benign: positional uncertainty. Faithful.
    TokenSmear { span: usize },
    /// Sequence smearing — mass split across semantically distinct tokens at one position.
    /// Requires disambiguation before commitment. Potentially unfaithful.
    SequenceSmear { n_hypotheses: usize, semantic_distance: f32 },
}

pub trait SmearClassifier {
    /// Given a [K][D] slice of MUX superposition weights (or BoM K-hypothesis beliefs),
    /// classify the mass distribution.
    fn classify(&self, weights: &[f32], k: usize, d: usize) -> SmearClass;
}
```

**Why this is a real gain, not a duplicate:**
- Plan 278's FaithfulnessProbe is **binary** (faithful / unfaithful) via causal intervention + attribution.
- MUX (Plan 178) **generates** superposition but does not **classify** it.
- BoMSampler (Plan 281) **samples** K hypotheses but does not **classify** the resulting distribution.
- **No shipped code classifies a superposition distribution into smear types.** This is the gap.

**The classifier uses the paper's operational definition (§5.2.1 vs §5.2.2):**
- Token smear ⟺ high mass on one token id, spread across >1 position.
- Sequence smear ⟺ high mass split across ≥2 distinct token ids at one position, with semantic distance (cosine in embedding space) above a threshold.

**Sync boundary compliance:** the classifier reads MUX/BoM weights (local latent state, never synced) and emits a `SmearClass` enum (a u8 — raw, deterministic, syncable). The enum can drive a Cognitive Integrity flag in `SyncBlock` without leaking the latent distribution. ✅ Raw-at-boundary, latent-locally.

### 2.3 Fusion opportunities (for the record, not all planned)

| Fusion | Paper × Note A × Note B | Novel capability? |
|---|---|---|
| **Smear classifier** (§2.2 above) | This paper × Plan 278 (FaithfulnessProbe) × Plan 178 (MUX) | Ternary diagnostic where binary existed. **Gain.** |
| **Opaque-depth budget gate** | This paper × Plan 276 (MicroBelief) × Plan 283 (Self-Advantage recursion gate) | Gate NPC latent computation by `(cycles × dim)` budget; halt when opaque depth exceeds threshold. Capability exists implicitly via `tau_reest`; this makes it explicit. **Gain at best** — likely duplicative with re-estimation trigger. |
| **Top-p salient scalar bridge** | This paper × Plan 276 (`project_to_scalars`) × Plan 237 (Schema Centroid informed init) | Narrow the HLA bridge from 5 scalars to "top-p salient directions" (those with `sigmoid(dot) > p`). NPCs "speak when they have something to say." **Gain** — incremental bridge variant. |
| **Length-budget early prediction** | This paper × Plan 240 (Spectral NPC Perception) × zone attention | NPC predicts its own response/action-length CDF early via padding-analog in latent space. **Speculative** — no clear game-AI analog. |
| **Intermediate-context scratch tokens** | This paper × Plan 172 (RiM Reasoning Buffer Slots) × Plan 217 (NextLat) | Fixed latent workspace tokens that are causally necessary but never decoded. RiM slots already ship this; the paper validates the pattern. **Already shipped, just framed differently.** |

---

## 3. Verdict

**Tier: Gain.**

### One-line reasoning

The paper is an interpretability analysis that **validates mechanisms we already ship** (top-k scalar bridge = HLA `project_to_scalars`; retroactive self-correction = `latent_functor/reestimation`; monitorability = FaithfulnessProbe + Cognitive Integrity Layer; sequence smearing = MUX; token smearing = BoM). The one extractable primitive — a ternary smear classifier extending FaithfulnessProbe — is a useful diagnostic but not a new capability class. Q1 (no prior art) FAILS across all six core mechanisms; Q2 (new capability) FAILS (analysis paper); Q3 (selling point) PARTIAL (validates existing selling point, no new one); Q4 (force multiplier) PARTIAL (cross-connects existing systems, doesn't multiply into new class).

### Why not Super-GOAT

- **Q1 FAIL:** Every load-bearing mechanism has shipped prior art under codebase vocabulary. The six-row mapping table in §TL;DR is dispositive. The closest the paper gets to novelty is *naming* "smearing" — but MUX (Plan 178, shipped) and BoM (Plan 281, shipped) are the deterministic and stochastic versions of exactly that phenomenon.
- **Q2 FAIL:** This is an analysis paper. It does not introduce a new mechanism, training recipe, or capability. It audits an existing architecture.
- **Q3 PARTIAL:** The paper *validates* our existing selling point ("our NPCs are monitorable because we ship a top-k scalar bridge + Cognitive Integrity Layer"). It does not give us a *new* selling point.
- **Q4 PARTIAL:** The smear classifier fuses FaithfulnessProbe × MUX × this paper, but the result is a diagnostic extension, not a force multiplier that produces a new pillar.

Per the skill's "no candidate escape hatch" rule: I am NOT writing "Super-GOAT candidate". This is a committed Gain verdict.

### Why not Pass

The paper is directly relevant to our latent-reasoning-monitorability pillar (FaithfulnessProbe, Cognitive Integrity Layer, MUX, BoM). It supplies:
1. **External validation** for the top-k scalar bridge design (use in the riir-ai `.research/129` selling-point doc).
2. **A vocabulary** ("opaque serial depth", "variable vs algorithmic transparency", "token vs sequence smearing") we should adopt.
3. **One small primitive** (smear classifier) worth planning as a Plan 278 extension.

### Routing

- **katgpt-rs (public, this repo):**
  - Research note 277 (this file).
  - Plan 298 — `SmearClassifier` trait + impl, opt-in behind `smear_classifier` feature, extending `faithfulness_probe`. Generic primitive, no game semantics.
- **riir-ai (private):**
  - **Update `.research/129_Cognitive_Integrity_Layer_Guide.md`** to cite this paper as external validation of the top-k scalar bridge design and to adopt the "opaque serial depth" + "smearing" vocabulary. This is a doc update, not a new guide — Super-GOAT-guide-mandatory rule not triggered (Gain, not Super-GOAT).
  - No new private research note needed.
- **riir-train:** N/A — paper contains no training method.

### Latent vs raw boundary (per AGENTS.md)

The smear classifier respects the sync boundary by construction:
- **Input** (MUX superposition weights / BoM K-hypothesis beliefs): local latent state, never synced.
- **Output** (`SmearClass` enum, `#[repr(u8)]`): raw, deterministic, syncable. Can flow into `SyncBlock` → `ChainConsensus` as a Cognitive Integrity flag.
- **Bridge function:** zero-allocation (operates on caller-provided `&[f32]` slices), gateable by feature flag (`smear_classifier`), introduces no sync dependency.

---

## 4. Closest cousins (for the fusion protocol record)

1. **`katgpt-rs/.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md` + `.plans/278_faithfulness_probe_modelless.md`** — **closest cousin.** Plan 278 ships the binary faithful/unfaithful probe; this paper supplies the vocabulary to extend it to ternary (single / token-smear / sequence-smear).
2. **`katgpt-rs/.research/158_MUX_Multiplexed_Latent_Reasoning.md` + `.plans/178_mux_multiplexed_latent_reasoning.md`** — MUX *is* deterministic sequence smearing. The paper is external evidence that the phenomenon is real and load-bearing in production latent reasoners.
3. **`katgpt-rs/.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md` + `.plans/281_bom_single_pass_diverse_sampling.md`** — BoM *is* stochastic token smearing (K noise queries → K diverse hypotheses at adjacent positions).
4. **`riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md`** — the private selling-point doc this paper validates. **Should be updated to cite arXiv:2606.20560** as external validation.
5. **`riir-ai/.research/123_Latent_Functor_Runtime_Guide.md` + `latent_functor/reestimation.rs`** — the DiPOD-canonical "coherence < tau_reest → re-estimate" pattern. Maps to the paper's "retroactive self-correction" (§5.1.2).
6. **`katgpt-rs/.research/228_RCD_Residual_Context_Diffusion.md` + `.plans/258_rcd_residual_context_diffusion.md`** — RCD's entropy-weighted residual carry between D2F denoising steps is the closest cousin of DiffusionGemma's self-conditioning matrix `S_t` (Eq. 1).

---

## 5. What this paper does NOT give us

Be honest about the boundaries:

1. **No new training method.** DiffusionGemma is pretrained by DeepMind; the paper audits it. Nothing to route to riir-train.
2. **No new inference primitive beyond the smear classifier.** The top-k bottleneck is already our bridge function; the case studies are observations, not mechanisms.
3. **No new game-AI capability.** The phenomena (non-chronological reasoning, smearing) are interesting but our NPCs already exhibit analogs (zone attention predicts budget before tactical compute; MUX holds multi-hypothesis superposition; BoM samples K diverse beliefs).
4. **No proof that smear classification improves downstream behavior.** The paper shows smearing *exists*; it does not show that *detecting* it improves anything. Plan 298's GOAT gate must prove the ternary classification produces a measurably better intervention than the existing binary probe (e.g., fewer false-positive faithfulness flags on benign token smearing).

---

## TL;DR

**arXiv:2606.20560** ("How Transparent is DiffusionGemma?", DeepMind, Jun 2026) is an **interpretability analysis paper** that audits text-diffusion LLM transparency along four axes (opaque serial depth, variable transparency, monitorability, algorithmic transparency) and concludes that **forcing the latent carry through a top-k/p token bottleneck between denoising steps collapses opaque serial depth from 28.6× to 1.1× the autoregressive baseline with no capability loss**. It names four diffusion-specific phenomena: non-chronological reasoning, token smearing, sequence smearing, intermediate-context reasoning. **Verdict: Gain** — the paper is *confirmatory* for our codebase, not novel. Every load-bearing mechanism already ships under codebase vocabulary: top-k scalar bridge = `MicroRecurrentBeliefState::project_to_scalars` (Plan 276); retroactive self-correction = `latent_functor/reestimation` + `cce_runtime/reestimation_trigger` (the DiPOD canonical-miss pattern); monitorability = FaithfulnessProbe (Plan 278) + Cognitive Integrity Layer (riir-ai `.research/129`); sequence smearing = MUX (Plan 178); token smearing = BoMSampler (Plan 281). **Q1 (no prior art) FAILS across all six mechanisms; Q2 (new capability) FAILS (analysis paper); Q3/Q4 PARTIAL.** The one extractable primitive is a **ternary smear classifier** extending Plan 278's binary FaithfulnessProbe to distinguish `CoherentSingle` / `TokenSmear` (benign positional uncertainty) / `SequenceSmear` (potentially unfaithful multi-hypothesis superposition) — routed to Plan 298 behind `smear_classifier` feature flag. **Action items:** (1) Plan 298 — smear classifier in katgpt-rs; (2) update riir-ai `.research/129` to cite this paper as external validation of the top-k scalar bridge design and adopt "opaque serial depth" + "smearing" vocabulary. **No Super-GOAT guide created** — Gain verdict, mandatory-guide rule not triggered.
