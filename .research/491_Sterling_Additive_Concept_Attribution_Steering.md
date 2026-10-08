# Research 491: Steerling — Additive Concept Attribution & Calibrated Steering

> **Source:** "Scaling Inherently Interpretable Language Models" — [arXiv:2608.07594](https://arxiv.org/abs/2608.07594) — Guide Labs Team, Madsen, Ismail, Nguyen, ..., Adebayo — 2026-08-06 (Steerling-8B)
> **Date:** 2026-08-19
> **Status:** IMPLEMENTED (opt-in) — Issue 672 T1–T5 shipped behind `sterling_primitives` + the ungated `noisy_or` core util (civ site delegates); GOAT G1–G4 PASS ([Bench 667](../.benchmarks/667_sterling_primitives_goat.md)). The first consumer LANDED 2026-08-20: riir-ai Issue 732 (`exact_emotion_attribution` — the additive goal-scoring variant + `NpcDecisionAttribution` ledger, GOAT G1/G2/G3/G4/G8 ALL PASS incl. the +10 ns/NPC G2 + the gemv bit-identity oracle; riir-ai Bench 702). Both features stay opt-in pending a game-level A/B; the consumer GOAT named as 672's promotion gate is now on record.
> **Related Research:** 290 (latent field steering — the shipped inference twin), 393 (subspace concept primitive), 397 (MAG mining + `calibrate_alpha`), 409 (MANCE concept erasure), 388 (jacobian lens concept readout), 144 (functional emotions / linear representations → behavior control), 301 (indicator probe bank), 382 (spherical steering), 357 (activation steering)
> **Cross-ref:** riir-ai Issue 732 (exact NPC emotion attribution — the white space), riir-train Research 425 + Issue 467 (training recipe / dllm-defaults insurance)
> **PASS-Redirects (synthesis):** Hill, Xinyu, Prasetio [arXiv:2508.02995 "The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet"] — claims dual-stream (ventral identity / dorsal pose) manifold disentanglement but never measures it; this note's measure-only disentanglement gauge `‖ΨᵀΦ‖²_F/(d²(M−1))` (`katgpt-core/src/sterling.rs`, Issue 672) measures what the paper only asserts, and the dual-stream split itself is our two-brain model + `DualSignalEvidenceCollector` binding (riir-ai Plan 309). PASS.
> **Classification:** Public

---

## TL;DR

Steerling-8B trains an 8B diffusion LM with interpretability as a training constraint: an **additive concept bottleneck** `h̄ = k̂ + û + ε` (sigmoid activations, linear LM head) between backbone and head, so **every output logit decomposes exactly into per-concept contributions + residual** — attribution is read off the forward pass, not estimated. Steering injects the model's own concept direction with **per-direction calibration `γ = τ/peak(e_c)`**, suppression uses a **ReLU-gated logit mask** (naive subtraction promotes anti-aligned tokens), and — the paper's hardest lesson — **injection must be trained (respond + express losses) or the model treats it as OOD and does not respond**. Headline result: across 3 orders of magnitude of compute the concept module costs a small fixed offset, and all four interpretability metrics *improve* with scale.

For this stack: the inference-time half is almost entirely consumable **modellessly** (9/12 mechanisms by the adversarial panel). The codebase already ships calibrated steering (`latent_steering` + `calibrate_alpha`), noisy-OR aggregation (civ salience gate), and counterfactual attribution (`causal_validation` patching, `step_attribution` Δ gates). The **white space**: no decision layer in the stack offers *exact additive per-concept attribution* — the per-NPC goal selector max-fuses drive contributions and never decomposes them — and the paper's mask-schedule ablation **contradicts our RePlaid adaptive-schedule class** (uniform beat a curriculum on knowledge benchmarks at pretraining scale).

**Distilled for katgpt-rs (modelless, inference-time):**
1. **Exact additive attribution**: a linear consumer over additive components ⇒ `Σ parts + residual == fused output` exactly. A decomposed-GEMV readout helper makes "why this score" a free byproduct of the computation it explains. (Issue 672 T2.)
2. **ReLU-gated suppression**: `ℓ_v ← ℓ_v − s·ReLU(a_c[v])`, never `−s·a_c[v]` — the naive form *promotes* anti-aligned tokens. One branch-free op; the output-space complement of MANCE erasure (R409). (Issue 672 T1.)
3. **`γ = τ/peak(e_c)` per-direction steering calibration**: normalize injection strength by the direction's peak logit effect. Logit-space cousin of shipped `calibrate_alpha(τ)` (activation-norm space, R397) — commensurates *output effect* rather than *input magnitude*.
4. **Lift sets**: `lift(w,c) = P(w|c-chunks)/P(w)` — a pure corpus statistic; steering targets for `latent_field_steering` and bias tables for `TernaryDraftModel`. Zero training. (Issue 672 T3.)
5. **Noisy-OR span aggregation** `1 − Π(1−k_t)` — already ships literally in `riir-games-civ salience_gate mod.rs:1215`; generalize as a core util.
6. **HSIC-style cross-covariance as a *metric*** (measure-only): channel-disentanglement gauge for shard/affect audits without the training loss.
7. **Trained-absence validity condition**: counterfactual baselines must be in-distribution states (dllm `[MASK]`, fog-of-war nulls) — a *gate* we can check, not something we can retrofit.
8. **Injection must be exercised or it's OOD** — validates our deterministic-construction + train-time-exposure paths (GDSD, reader-LoRA) and warns against bolting steering onto a frozen model with no exposure.

---

## 1. Paper core findings

- **Recipe = 5 conditions.** Causal faithfulness: *Nativeness* (attributed variables on the compute path), *Agreement* (|Δℓ_observed − Δℓ_predicted| ≤ ξ), *Validity* (interventions in a training-supported family). Semantic faithfulness: *Interpretation* (semantic cards describe the variables), *Coverage* (attributed fraction ζ, residual reported). Post-hoc methods (probes/SAE/gradients/SHAP/CoT) fail one or both by construction — the model was never trained to make the explanation an interface.
- **Architecture**: causal diffusion (block-causal attention, b=64; bidirectional in-block, causal across blocks; trained on a single corrupted sequence — no Block-Diffusion clean-copy concat ⇒ ~half its training cost, KV-cacheable inference) + concept module `h̄ = k̂ + û + ε` where `k = σ(f(h))` over n=33,732 Atlas concepts, `u = σ(g(h))` over m=3n unknown concepts with **factorized low-rank U=A·B (R=256, ~15× smaller)**, residual ε with dropout p_ε pressing it toward 0. **Sigmoid, not softmax** — the paper's architecture is our house rule.
- **Losses**: `L = L_MDM + λ_concept·L_concept + λ_rec·L_rec + λ_indep·L_indep`; L_concept = chunk-level BCE with **noisy-OR** OR-aggregation over masked tokens (positive-only labels); L_rec = MSE(û, h − k̂_GT) detached; L_indep = normalized cross-covariance (linear-HSIC) keeping known/unknown disentangled (two terms at mid-training).
- **Steering**: amplify `h ← h + γ·e_c` from layer L_inj, `γ = τ/peak(e_c)`, `peak = max_y e_cᵀW_y`; suppress via negative injection **plus** ReLU-gated logit subtraction. **Steering training**: 4 interleaved mid-training phases over token-level concept-tagged data with `L_respond` (bottleneck must activate) + `L_express` (probability mass on the concept's lifted token set). Without it, steering scored *worse than prompting* on harmonic mean and ~⅓ of concepts never activated.
- **Scaling**: IsoFLOP across 3 OOM, 4 families (AR/CDLM ± concept). Concept module shifts exponents by a small fixed per-backbone offset; **all 4 interpretability metrics improve with compute** (concept loss ↓, independence loss ↓, contribution ↑ 0.62→0.85, alignment ↑ ~3.7/5). Joint Chinchilla form predicts 8B validation loss within 0.11 nats; 3/4 metrics extrapolate from small-scale fits. Overhead: 89% of params at 10M → 4% at 8B → <1% at frontier (a *scale trap* for our 0.5–2B sizes — low-rank + small n mandatory).
- **Pretraining lessons**: moving-Gaussian mask curriculum (0.2→0.8) hurt knowledge benchmarks once past ~0.5 (MMLU/WinoGrande declined); uniform restored it. Teacher-forcing floor **0.5, never 0** (floor 0 collapses alignment ~30% for zero capability gain). Mid-training (150B, code-augmented, uniform masking, TF→0, sparsified heads top-32/128) rescued every benchmark (+10pp avg) and improved all interpretability metrics.
- **Atlas data pipeline**: 6.6M docs → 44M chunks → 500M LLM tags → embed → k-means k=80k → coherence filter → LLM label → Louvain graph dedup → 33,732 concepts → trained annotator (Qwen3-Embedding-0.6B + PU loss) → 1.5T tokens annotated; IVFPQ index (64× compression, 808 GB mmap, recall@10 96.8%) for training-data attribution as **transduced retrieval** (mean-pool → MLP transducer → ANN). "Similarity, not causation."

## 2. Path 0 inventory (three-track panel, merged)

| Component | Coverage | Extraction (modelless?) |
|---|---|---|
| Additive bottleneck decomposition | PARTIAL — sigmoid concept layers + max-fused goal scoring ship; no additive decision head | **YES** — pure algebra; decomposed-GEMV readout |
| γ=τ/peak steering calibration | PARTIAL — `calibrate_alpha(τ)` ships in activation-norm space (R397) | **YES** — one GEMV+max over frozen W at freeze time |
| ReLU-gated suppression | NONE (MANCE erasure covers intent in activation space) | **YES** — masking algebra |
| Steering training (respond/express) | PARTIAL — GDSD noise exposure, deterministic reader-LoRA | **NO** — the responder is genuinely trained |
| Atlas concept mining | PARTIAL — kg_clustering centroids, riir-rag embed/BM25/KNN | PARTIAL — lift sets are closed-form; tagging pipeline is model-based |
| Trained-absence baseline | PARTIAL — decay_confidence, fog-of-war, dllm [MASK] | PARTIAL — validity *gate* yes; baseline itself is a training property |
| TDA as transduced retrieval | PARTIAL — mech_attribution influence proxy, DenseEmbedIndex exact KNN | PARTIAL — deterministic-embedder stand-in for the transducer (gap reported) |
| HSIC independence | NONE as loss; `cross_covariance` exists in riir-poc | **YES as metric** (measure-only) |
| Noisy-OR aggregation | **COVERED** — civ salience gate literal | — |
| Bounded-metric scaling fits | PARTIAL — `ScalingLawAllocator` Chinchilla fit (Bench 046) | **YES** — OLS/log-linear forms closed-form |
| IG with [MASK] baseline | NONE | NO (needs ∇F) — forward-only path probes + #1 replace it |

Paths 1–3 checked per row before any training deferral: freeze/thaw snapshot + deterministic reader-LoRA + latent gate cover the *correction* use-cases; the genuinely-trained residue (f/g concept heads, steering responder, transducer) is routed to riir-train Research 425.

## 3. Coverage map (shipped cousins + signal-diff)

| Ships | Signal it consumes | vs Steerling |
|---|---|---|
| `latent_steering.rs` (Plan 309) + `latent_steering_bridge` (riir-ai, DEFAULT-ON) | unit direction + α; steers real Gemma-2 residual stream | injection mechanism ≡; direction provenance = mined (MAG) not first-class trained param |
| `calibrate_alpha(τ)` (MAG, R397) | **prefix-activation norm** | Steerling's γ consumes **peak logit shift** — commensurates output effect, not input magnitude |
| `subspace_steering.rs` (R393), `manifold_erasure.rs` (R409) | orthonormal blocks; k-NN tangent erasure | k-dim concepts; suppression in activation space (ReLU gate is the output-space complement) |
| `causal_validation/` (riir-ai) + `step_attribution` (DEFAULT-ON) | counterfactual patching deltas; replay A/B Δ | *estimation* via interventions; Steerling gets the same class **exact by construction** — but only because the head is additive |
| civ `salience_gate` noisy-OR; `EmotionKick`/`tamed_aura`/CLR | additive clamped kicks, rank-gated suppression, crowd amplification | emotion steering COVERED; per-emotion **decision attribution NONE** (goal_prior max-fused, never decomposed) |
| `mech_attribution` (influence proxy), `DenseEmbedIndex` (exact KNN + rerank) | activation influence; cosine ANN | TDA without transducer/IVFPQ |
| dllm (Plan 068) + `gemma2_d2f` + GDSD | block-causal single-seq masked CE (no clean copy) | paper **validates our default** as the 2×-cheaper path; block_size default 16 vs paper 64; **RePlaid (Plan 078) adaptive schedule = the penalized class** |
| Plan 244 `concept_extractor`/`concept_lora` (riir-train) | 7 hand concepts, sigmoid zone detection, per-concept LoRA | the natural upgrade target for a learned dictionary |
| `ScalingLawAllocator` (Bench 046, 571ns) | Chinchilla form on bench points | the fit exists; IsoFLOP-slice harness is the missing half |

**White space (grep-verified zero):** no decision layer offers exact additive per-concept attribution. The stack has both halves of Steerling's thesis *separately* — calibrated steering and counterfactual attribution — but nowhere do they unify: per-drive contributions are computed every tick (`sigmoid(dot(stat_vec, dir_vec))`) and then **max-fused**, never surfaced as a decomposition.

## 4. Fusion

**Exact-Emotion-Ledger NPCs** = paper × R290 (latent steering) × R144 (functional emotions): make the per-NPC decision layer *additive* over the 5 affect scalars with a (near-)linear consumption head ⇒ (a) every decision ships with an exact per-emotion contribution ledger ("fled: fear +0.62, desperation +0.21, calm −0.07, residual 0.10") — no estimator, no backprop, near-free; (b) the same directions are γ-calibrated amplify/suppress knobs via the already-shipped `EmotionKick`/`latent_steering` substrate (one global τ, commensurate per-emotion strength); (c) ReLU-gated suppression enables one-sided crowd calming without promoting the opposite pole — the CLR-demotion whiplash lesson as a reusable op.

**Novelty gate scoring:** Q1 prior art — YES (grep-verified white space; additive-exact attribution mechanism itself novel-ish, see §5). Q2 new behavior class — **NO**: it is observability of existing behavior (the NPCs act the same; we explain them exactly) — fails the "structurally sloppy without it" pillar test. Q3 selling point — YES ("every NPC decision ships with an exact why" — GM dashboard/audit). Q4 force multiplier — YES (HLA affect × motivation runtime × GM tools × latent steering). **3/4 → Gain, not Super-GOAT.** Filed as riir-ai Issue 732 (no "candidate" escape hatch — the decision is recorded, the issue carries it).

## 5. Prior art (§4 mandatory search)

- **Anticipates the architecture (both uncited by the paper):** CB-LLM "Concept Bottleneck Large Language Models" (arXiv:2412.07992, ICLR 2025 — from-scratch interpretable LLM, LLM-annotated concept bottleneck); CB-pLM "Concept Bottleneck Language Models for Protein Design" (arXiv:2411.06090, ICLR 2025 — *generative masked* LM + concept layer at 24M→3B). Steerling reads as the CBGM lineage (Ismail on both papers) scaled to 8B NL pretraining.
- **Concurrent closest cousin:** PRISM "Prototype Language Models" (arXiv:2607.00510, Jul 2026) — sparse prototype mixtures, TDA ~500× faster than influence baselines, **calibrated linear prototype controllers**, targeted suppression without finetuning. Narrows Steerling's novelty to scale + the scaling *trend* + the two calibration/gate mechanisms.
- **Novel per available evidence:** (i) interpretability improves with capability across ~3 OOM within one by-design pretraining family (PRISM spans 1 OOM; CB-pLM reports no trend; Anthropic's Scaling Monosemanticity is post-hoc SAE); (ii) `γ = τ/peak` per-direction training-time calibration (nearest: Linear Accessibility Profile arXiv:2604.15557 uses logit-lens to *predict* steering effectiveness, not calibrate); (iii) trained ReLU gate on logit suppression (nearest: Arditi et al. arXiv:2406.11717 one-sided directional ablation/clamp).
- **Established, not claimable:** TDA-as-retrieval (kNN-LM ICLR 2020, RETRO 2021, TRAK/Datamodels 2023, Google PAIR 8B-scale TDA 2025); [MASK]/in-distribution IG baselines (Kim et al. ICML 2020 Input Marginalization; Sturmfels Distill 2020); concept-library construction pipelines (Anthropic auto-interp, Transluce neuron description). Causal-mask diffusion concurrent: CARD arXiv:2601.22031.

## 6. Verdict

**GAIN** — multiple actionable items + one white-space fusion; not Super-GOAT (Q2 fails: observability, not new capability class). One-line reasoning: the paper's inference-time half decomposes into closed-form primitives we mostly ship or can ship modellessly; the one genuinely-trained half routes to riir-train; the mask-schedule finding contradicts a live default (RePlaid).

| Mechanism | Verdict | Routing |
|---|---|---|
| Exact additive attribution (decomposed readout) | Gain — open primitive | katgpt-rs Issue 672 T2; consumer riir-ai Issue 732 |
| ReLU-gated suppression | Gain — open primitive | katgpt-rs Issue 672 T1 |
| Lift-set steering targets | Gain — closed-form corpus statistic | katgpt-rs Issue 672 T3 |
| γ=τ/peak logit-space calibration | Gain — variant of shipped `calibrate_alpha` | fold into Issue 672 T3 consumers |
| Noisy-OR as core util | Gain — generalize shipped civ instance | Issue 672 (rider) |
| HSIC disentanglement metric | Gain — measure-only audit gauge | Issue 672 (rider) |
| Per-emotion decision ledger (game layer) | Gain — the white-space fusion | riir-ai Issue 732 |
| Concept-bottleneck training program | Path 0.5 — genuinely trained | riir-train Research 425 (flagship H-lite→A→D, 165–315 4090-hrs, owner call) |
| Mask-schedule + block-size ablations | Gain — contradicts/extends dllm defaults | riir-train Issue 467 (35–65 GPU-hrs) |
| IVFPQ/transducer TDA | Pass-for-now — substrate suffices at our scale (DenseEmbedIndex exact KNN); revisit if corpus ≥1B chunks | — |

**MOAT gate:** open primitives (attribution decomposition, ReLU-gate, lift sets) → katgpt-rs ✓ (generic math, no game semantics). Game selling point → riir-ai issue ✓. Training how → riir-train ✓. No chain/shard novelty claimed.

## 7. Actionable — files created

- katgpt-rs Issue 672 — Sterling-derived modelless primitives (relu-gated suppression, exact-decomposition readout, lift sets).
- riir-ai Issue 732 — Exact per-emotion attribution of NPC decisions (additive scoring variant + ledger readout).
- riir-train Research 425 — concept-bottleneck training recipe (Path 0.5 program, pending owner decision).
- riir-train Issue 467 — mask-schedule (RePlaid insurance) + block-size/per-block-noise ablations vs dllm defaults.

**Config guards recorded for future work** (no issue — recipe rules): teacher-forcing floors anneal to **0.5, never 0** for any future aux-head trainer; concept-module dictionaries at our scale must be low-rank + small-n (the 89%-overhead-at-10M trap).
