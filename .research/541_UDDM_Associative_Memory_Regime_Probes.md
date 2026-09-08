# Research 541: UDDMs are Associative Memories — Regime Probes for Frozen Predictors

> **Source:** "Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data" — Bao Pham, Mohammed J. Zaki, Luca Ambrogioni, Dmitry Krotov, Matteo Negri. EMNLP 2026 (Main). [arXiv:2604.26841](https://arxiv.org/abs/2604.26841)
> **Code:** official repo `Lemon-cmd/Associative-Memory-and-Language-Diffusion` (MIT) @ `39b8b970bcc12b62ad2df0d72ca48ea1f0c60959`, audited in `.raw/am-ld/` before cleanup; builds on `s-sahoo/duo` (Apache-2.0). Key file read directly: `eval_entropy.py::get_cond_entropy` (perturb x0 → xt at t≈1e-5 → forward → per-position `-Σ p·log p`).
> **Date:** 2026-09-08
> **Status:** Active
> **Classification:** Public
> **Related Research:** 455 (Hebbian Kernel Memory — margin/capacity cousin), 466 (CP^(d-1) Hopfield — capacity-law cousin), 24 (δ-Mem — write-side AM), 79 (EqR — attractor-residual-as-signal), 536 (EBT — energy/curvature lens)
> **Related Plans:** 559 (hebbian_kernel_memory primitive), Bench 276 (AttractorKernel honest-null record)
> **Verdict:** **GAIN** — three-track split filed; modelless instrumentation suite is the PRIMARY track, training-recipe items SECONDARY (serving-envelope rule).

---

## TL;DR

UDDMs (uniform-state discrete diffusion) behave as associative memories whose basins of attraction form via **conditional-likelihood maximization alone** (no energy function): cross-entropy training implicitly factors as Hebbian storage × margin maximization (eq 11), and the achieved margin certifies basin size (`κ > 2√ρ`, Gardner capacity eq 26). Sweeping dataset size reveals a **sharp memorization→generalization transition**: basins around training samples shrink while basins around unseen samples expand, and the transition is detectable **with a training-free probe** — the conditional entropy of the model's own predicted token distributions (memorization ⇒ vanishing entropy; generalization ⇒ finite entropy; the train-vs-synthetic entropy-gap closes exactly at the transition).

**Distilled for the stack (modelless, inference-time):** every probe in the paper consumes only a frozen predictor's output categoricals — which every serving path here already produces. The deliverable is a **regime-probe suite** (per-position conditional entropy, entropy-gap two-sample detector, corrupt-recovery basin protocol, Gardner capacity LUT, margin spectrum), not a training loop. The paper's own training loop is stock Duo/UDDM (NELBO, AdamW, EMA — advocate-read) and should be adopted nowhere.

---

## 1. Paper Core Findings

| Finding | Form | Where |
|---|---|---|
| AM without energy | Basins form via pseudo-likelihood/conditional-likelihood maximization; couplings need not be symmetric | §3, [d2025pseudo] |
| Cross-entropy = Hebbian × margin | `dL/dW ∝ −(xℓxᵐ) · (1 − tanh M)`; penalty `≈ 2e^{−2M}` concentrates on smallest-margin samples; CE implicitly solves max-margin (Soudry 2018) | eq 10–11 |
| Margin ⇒ basin size | `κ > 2√ρ` (CLT: ρ flipped inputs perturb a unit-variance pre-activation by ≈2√ρ); Gardner capacity `1/γ_c(κ) = (1+κ²)Φ(κ) + κφ(κ)` inverted for `κ_max(γ)` | Appx E, eq 26, Fig 6 |
| Memorization→generalization transition | As dataset size grows at fixed model size/steps: training-sample recovery falls, test-sample recovery rises, they converge; larger models DELAY the transition | Figs 1–2 |
| **Conditional-entropy probe** | Per-token `H(xℓ|z_t)` (eq 13); recovered tokens ≈ 0 entropy; sequence-level train-vs-synthetic entropy distributions converge at the transition; no external critic needed | Figs 3–4, eq 13 |
| Curvature readout | `H ≈ −½ log det H(essian)` at the mode (Laplace approximation) — entropy as inverse basin sharpness; discrete instantiation left open | Appx D |
| Micro-attractors | A persistent class of low-entropy tokens survives generalization (function words / syntactic skeletons) | Fig 3, §6 |
| Perplexity trap | External-critic perplexity is NON-monotonic across the transition — lowest in the memorization regime (duplication artifact). Low gen-ppl ≠ good; checkpoint selection on gen-ppl alone selects memorized checkpoints | Fig 5 |

---

## 2. Path 0 Inventory (three-track merge)

Advocates: No-GD (12-row inventory, kept in full) + Model-based (4-row recipe analysis). Discard ledger: **no advocate finding discarded**; one (margin regularizer) is marked NOT-RECOMMENDED with a kill-gate, one (contamination audit) carries the Min-K% prior-art caveat.

### Modelless track (PRIMARY — fits the serving envelope)

| # | Extractable | Closed-form/modelless? | Consumer surface (what ships there) | Routing |
|---|---|---|---|---|
| M1 | Per-position conditional entropy of a categorical (eq 13) | Yes — one max-shift/log-sum-exp pass | `katgpt-core/src/breakeven/fidelity.rs` already computes per-position CE from forwards; `cgsp/types.rs` has `entropy_nats`; engine decode path emits categoricals | Kernel EXISTS, **instrument does not** — see signal-diff §3 |
| M2 | **Entropy-gap regime detector** (train-vs-synthetic entropy distributions; gap statistic: mean gap / KS / 1-D Wasserstein) | Yes — sampling is inference | New probe beside `katgpt-core/src/data_probe/` (synthetic-probe harness precedent); `riir-clippy/src/score_bench/` OOD axis; engine serving-health audit | **MISSING — file** (katgpt-rs Issue 740) |
| M3 | **Corrupt-recovery basin probe** (eq 12: corrupt fraction ρ → denoise → recovery rate on corrupted positions) | Yes — protocol over any frozen renovator; `ugc_schedule.rs` ships the corrupt+denoise machinery shape (`UgcDenoiser`, `bernoulli_unmask_with_grid`) | Same probe module; ground-truth-regime validation on deterministically constructed predictors (exact-LUT "memorizer" vs kernel-smoothed "generalizer" — both GD-free, regime known by construction) | **MISSING — file** (Issue 740) |
| M4 | **Gardner capacity LUT** (`κ_max(γ)` via eq 26; basin bound `κ > 2√ρ`) | Yes — precomputed γ→κ_max table; std-normal CDF/PDF | **Index capacity planner**: `riir-neuron-db` ShardIndex/ItemEmbedIndex/DenseEmbedIndex load planning (currently folklore); `hebbian_kernel_memory` basin-radius certification (it has decoding margin γ_min, NOT input-corruption tolerance); `riir-clippy` RuleIndex KNN load | **MISSING — file** (Issue 740; the one genuinely new closed-form primitive) |
| M5 | Margin spectrum + `e^{−2M}` mining law (concentrate effort on smallest-margin items) | Yes — margin = observed-logit − top-competitor-logit, offline | `riir-clippy` corpus/frontier mining — rank rules/examples by margin; consolidation priority (low-margin memories most plastic) | **MISSING — file** (riir-clippy Issue 077) |
| M6 | Self-entropy as critic-free quality metric (tracks external-critic perplexity across the transition, Fig 5 alignment) | Yes — one reduction per candidate | score_bench pre-filter; engine generation QA | Part of M2's detector; folded |
| M7 | Curvature bridge `effective_support = exp(H)` | Modeled analogy (paper's Appx D is continuous-only; flagged honest) | FidelityMatcher adaptive compression; collapse canaries | Recorded, NOT filed — needs its own utility gate; low confidence |
| M8 | UDDM closed-form reverse posterior (eq 14) + T-operator LUT (eq 19) + β(t) schedule | Yes but **schedule-family-specific** (Sahoo duality UDDM) | `ugc_schedule.rs` UgcDenoiser seam; `entropic_tilt` β-solver instantiates it | Recorded, NOT filed — no consumer needs uniform-state diffusion today |

### Training track (SECONDARY per serving-envelope fit — outside the hot path)

| # | Extractable | Pipeline | Cost | Routing |
|---|---|---|---|---|
| T1 | **Critic-ppl early-stopping trap**: low gen-ppl selects memorized checkpoints (Fig 5 non-monotonicity); never trust gen-ppl trend without a regime probe | EVERY riir-train checkpoint loop + distill_attention (currently loss-curve-only selection) | 0 GPU-h — procedural | **File — riir-train Issue 528** (highest value/cost ratio in the whole paper) |
| T2 | Entropy-gap checkpoint selection + data-mix validation (per-source gap never closing ⇒ contaminated/isolated source) | Stage 3 LoRA loop; Stage 2 `training.jsonl` filter | ≤1–3% training wall-clock at checkpoint intervals | File with T1 |
| T3 | Transition-ratio sweep (N\*(size)) for from-scratch drafters (`TernaryDraftModel .bits` corpus budget; "smaller drafters exit memorization sooner" corollary) | Tiny-only replication: 8 corpus fractions × 150K steps ≈ **~50 GPU-h on the idle 4090** (the paper's full 162-model sweep ≈ 4–5 GPU-years — REJECTED) | ~50 GPU-h | **File as owner-gated option** in Issue 528; do NOT run now |
| T4 | Margin regularizer (anti-entropy-collapse) | None justified — eq 11 says CE already max-margins; adding it double-pays | 12–25 GPU-h if ever | **NOT RECOMMENDED** — kill-gate: Appx-E binary-AM toy (CPU-cheap) must show basin gain before any GPU |

**Serving-envelope ranking:** T1/T2 protect every existing training run for free but run OUTSIDE the serving path; M2–M5 run INSIDE it (serving health, score-bench, drafter selection). Modelless track is PRIMARY per the TTPO rule; the training items are still filed because they are near-zero-cost and protect live pipelines.

---

## 3. Signal-diff vs closest cousins (§3.6 — no name-match coverage claims)

| Cousin | What it ships (its core formula's signal) | What this paper adds | Verdict |
|---|---|---|---|
| `hebbian_kernel_memory` (R455/P559) | **Decoding margin** `γ_min = min⟨v_f(i) − v_j, MLP(k_i)⟩` — value-space retrieval SEPARATION | **Input-corruption tolerance**: `κ > 2√ρ` bounds how many bit-flips a stored pattern survives — a basin RADIUS, not a separation margin. Different signal, composable: margin certifies separation, Gardner certifies perturbation tolerance | Gap is real → M4 |
| `cp_hopfield` (R466) | BBP **capacity** transition α_c — load (patterns/dim) at which recall collapses | **Dataset-size memorization→generalization** transition — a different axis (how many DISTINCT samples before novel inputs become attractors). Also ships a regime PROBE (entropy gap) where cp_hopfield ships a capacity constant | Different axis → M2/M4 stand |
| `breakeven/fidelity.rs::cross_entropy` | CE **kernel** (logits+target → scalar) for compression-fidelity deltas | The **instrument**: per-position entropy over the PREDICTED distribution + two-sample gap detector + recovery protocol. Kernel exists; protocol missing | M1 kernel exists → M2/M3 are the missing layer |
| Plan 276 `AttractorKernel` (demoted) | Random-Xavier recurrent kernel — honest null: "hysteresis is a property of TRAINED attractor networks, not randomly-initialised ones" | §3 explains exactly why: basins require conditional-likelihood/Hebbian construction. The modelless fix is R455's closed-form construction (already shipped for MLPs) — a Hebbian-constructed attractor variant is now UNBLOCKED if a consumer demands it | Recorded; no consumer today — do not reopen 276 speculatively |
| `data_probe/markov.rs` | Synthetic Markov-sequence probe harness | Basin/regime probes are siblings in the same harness family | M2/M3 land beside it |

**Reverse-grep for documented gaps:** Plan 276's null result is a documented, unresolved limitation this paper's theory directly explains (fidelity.rs has no regime notion; score_bench has no OOD axis of this shape; riir-train selects checkpoints on loss curves alone — T1). Gain confirmed.

---

## 4. Novelty gate (pinned claim)

**Claim:** "A predictor-agnostic regime-probe suite (conditional-entropy gap + corrupt-recovery basin protocol + Gardner capacity LUT) shipped as modelless katgpt-core primitives, consuming any serving-path categorical, with the healer corpus + riir-train checkpoint loops as measured consumers."

| Q | Answer | Evidence |
|---|---|---|
| Q1 No prior art for the TECHNIQUES? | **NO** — individually established: entropy-based membership/contamination detection (Min-K%, Carlini 2021, Shi 2023); memorization↔generalization phase transition via AM lens (arXiv:2505.21777 — same group, continuous domain; arXiv:2508.17689 "Edge of Memorization"; arXiv:2505.16959 early-stopping phase diagrams); Gardner capacity is 1988 textbook | §4 searches, 2026-09-08 |
| Q2 New behavior class? | Partial — the STACK gains regime detection it lacks, but the class exists in the literature | — |
| Q3 Product selling point? | Weak-moderate ("our healer knows when its corpus is overfit"; "our training loop can't silently pick memorized checkpoints") | — |
| Q4 Force multiplier? | YES — connects hebbian_kernel_memory, ugc_schedule, score_bench, riir-train loops, neuron-db indices | §2 |

**Not all 4 YES ⇒ NOT Super-GOAT.** Gain, filed as issues (novelty-of-integration, not novelty-of-technique). No "candidate" escape hatch used.

---

## 5. GOAT gates (sketch, for the filed issues)

- **M2/M3 (regime probes):** G1 discriminative validity — separates the deterministically-constructed memorizer/generalizer pair (regime known by construction, zero training); G2 convergent validity — detected transition agrees with external-critic perplexity inflection on a probe set (paper's own Fig 5/Fig 17 protocol); G3 bit-determinism, BLAKE3-able artifacts; G4 zero-alloc scratch-buffer protocol. **UQ floor rule:** if the probe ever emits a calibrated probability/interval, it must beat the conformal-naive floor (Research 322 / Plan 340) on CRPS/coverage/Winkler — a bare detector does not trigger this.
- **M4 (Gardner LUT):** G1 LUT vs direct numeric inversion (rel-err < 1e-6, golden vectors); G2 bound-holds test — flip-fraction ρ_c at 0.99-overlap on a constructed Hebbian memory sits above the κ_achieved-derived bound across loads (Fig 6 shape, offline); G3 the bound is a FLOOR (never reds valid loads); O(1) lookup, precomputed at build time.
- **M5 (margin mining):** G1 margin sign/monotonicity contract; G2 yield-per-review — `e^{−2M}`-weighted mining beats uniform-ordered mining on measured score-bench heal rate; G3 zero-alloc ranking pass.
- **T1 (checkpoint trap):** G1 documented + assertable: no checkpoint selected in the intermediate regime without a regime probe; G3 no regression vs val-loss stop.
- **T3 (sweep, if ever run):** G1 drafter trained at corpus ≥ N\*(size) beats sub-transition baseline on held-out fix acceptance; G2 ≤ 50 GPU-h measured; G3 zero regression on the certified-replay lane; G4 `.bits` inference contract unchanged.

---

## 6. Latent↔raw boundary check

The probes consume **model output categoricals** (latent-space confidences) and emit **raw scalars** (entropy nats, gap statistics, margins, ρ_c) — exactly the sanctioned bridge direction (latent read → scalar out). No probe output crosses a sync boundary; regime labels are local instrumentation. If wired into NPC affect (e.g. entropy-of-action-distribution as an arousal-adjacent signal), the scalar crosses via the existing 5-scalar affect bridge, never the embedding — compliant with the sync-boundary rule. Sigmoid (not softmax) wherever a probe output gates a blend.

## 7. P0–P3

- **P0** — riir-train Issue 528 T1 (critic-ppl trap + entropy-gap checkpoint signal): zero GPU, protects every live training loop. Procedural change + harness assertion.
- **P1** — katgpt-rs Issue 740 M2+M3+M4 behind a feature flag (`regime_probe`): the probe module + Gardner LUT with the constructed-pair GOAT gate. Small, self-contained, unblocks the healer + index-planner consumers.
- **P2** — riir-clippy Issue 077 M5: margin-weighted mining + entropy-gap OOD axis on score_bench (rides P1's primitives).
- **P3** — T3 sweep (owner-gated, ~50 GPU-h, only if the drafter corpus budget becomes a live decision); M7 curvature bridge; M8 uniform-state diffusion schedules (dormant until a uniform-state consumer exists).

## 8. Provenance & hygiene

- Paper read in full (arXiv HTML v2). `get_cond_entropy` verified in the cloned official repo @ `39b8b970` (MIT); training-loop details (AdamW 3e-4, EMA 0.9999, NELBO, DiT) are advocate-read of the same clone. Clone deleted after this note per §0.5 — re-clone at the pinned sha if code-level re-verification is needed.
- Prior-art searches 2026-09-08: class landscape + component techniques (results in conversation log). Headline class is published; no technique-novelty claimed.
- Adversarial panel: 2 advocates (No-GD, Model-based), one spawn round, briefs did not leak the coordinator's routing; merged table above; no finding discarded without reason (two carry honest caveats: M7 modeled-analogy, T4 not-recommended).
