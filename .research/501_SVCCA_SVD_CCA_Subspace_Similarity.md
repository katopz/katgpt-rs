# Research 501: SVCCA — SVD+CCA Affine-Invariant Subspace Similarity

> **Source:** [SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability](https://arxiv.org/abs/1706.05806) — Raghu, Gilmer, Yosinski, Sohl-Dickstein (Google Brain / Cornell / Uber AI), NIPS 2017
> **Date:** 2026-08-24
> **Status:** DISTILLED — pending owner decision (Issue 684 primitive + riir-train Plan 349)
> **Related Research:** 115 (PEIRA — learned CCA, different mechanism), 178 (Rosetta — axis-aligned correlation mining, the baseline SVCCA beats), 397 (MAG — ships `cka_linear`), 462 (INTACT — CKA-checkpoint correlation), 238 (LoRA-Muon spectral manifold — `ns_inv_sqrt_psd` substrate)
> **Related Plans:** riir-train [Plan 349](../../riir-train/.plans/349_svcca_freeze_training_recipes.md)
> **Cross-ref (riir-neuron-db):** Research 001 (subspace consolidation quality gate — `can_freeze` reads absolute stats), 007/011 (CUCG freeze-gate isomorphism)
> **Classification:** Public (the primitive); consumers private

---

## TL;DR

SVCCA is a **closed-form, affine-invariant similarity operator between two representation subspaces**: SVD each (keep directions explaining 99% of variance — denoising), then CCA between the reduced subspaces (an eigenproblem on the whitened cross-covariance). Mean canonical correlation ρ̄ ∈ [0,1] is the similarity scalar. **Every linalg component already ships in katgpt-core** (`thin_svd_into`, `ns_inv_sqrt_psd`, `symmetric_eig`, ridge, numerical-rank η=0.99) — the composition (~200 LOC) and its runtime gates do not ship anywhere. Verdict: **Gain** — implement the `svd_cca` primitive (Issue 684) + riir-train recipes (Plan 349). Two Super-GOAT-shaped fusions (semantic-integrity tier, affine-invariant belief alignment) are filed as **fusion ideas, novelty TBD** pending the primitive + defend-wrong PoC.

**Distilled for katgpt-rs (modelless, inference-time):** the operator is pure fixed-size f32 linear algebra — no training, no data-dependent iteration, deterministic (fixed Jacobi sweeps + fixed Newton–Schulz iterations → bit-stable), zero-alloc via caller-owned scratch (the `SvdScratch`/`NewtonSchulzScratch` house pattern). It answers a question nothing in the stack can answer today: **"are these two representation snapshots the same function, up to invertible linear re-mixing?"** BLAKE3/Merkle prove same *bytes*; CCA proves same *representation*.

---

## 1. Paper core findings

1. **The operator.** A neuron's representation = its activation vector over a dataset; a layer = the subspace spanned by its neurons. SVCCA(L1, L2): (a) thin SVD of each, keep the smallest k with Σᵢ≤k sᵢ² ≥ 0.99·Σsᵢ²; (b) CCA on the reduced sets — solve max corr(aᵀX′, bᵀY′) → canonical correlations ρ₁≥ρ₂≥…; ρ̄ = mean. Invariant to invertible linear maps, permutations, axis scaling (CCA invariance proof in Appendix B).
2. **Why SVD before CCA (the load-bearing detail).** Naive CCA cannot distinguish "50 aligned + 150 noise dims" from "50 aligned + 150 useful-but-different dims" — both give ρ=1×50, 0×150. The 99%-variance pre-reduction removes noise directions before they dilute ρ̄. This is the failure mode that makes raw CCA unusable as a similarity gate.
3. **Layers are low-rank.** Projecting a 512-neuron fc layer onto its top ~25 SVCCA directions retains near-full accuracy, no retraining → compression Wx → WPᵀ(Px).
4. **Bottom-up convergence.** Lower layers reach their final representations early in training; top layers drift much longer (convnets, resnets, and stacked LSTMs on PTB). → **Freeze Training**: sequentially freeze lower layers (paper: linear schedule i/L); saves backward compute and slightly *improved* CIFAR-10 generalization (per-layer early stopping).
5. **DFT block-diagonalization.** For translation-equivariant layers + translation-invariant datasets, per-channel 2D-DFT makes the covariance exactly block-diagonal (circulant-matrix lemmas) → exact SVCCA at kn·log n + n²k^2.5 instead of (kn²)^2.5.
6. **Class-information localization.** CCA similarity between a layer and a class logit traces *where* class-specific information forms (firetruck separates from dog breeds early; similar breeds track together).

## 2. Distillation

### 2.1 The primitive (what Issue 684 implements)

```
svcca_into(x, y, dx, dy, n, var_keep=0.99, ridge, scratch) -> CcaReport
  1. column-center X (dx×n), Y (dy×n)
  2. thin_svd_into(X) -> (Ux, sx); kx = numerical_rank(sx, 0.99); X̃ = Ux[:,kx]ᵀ X
     (same for Y)                        — the η=0.99 estimator already in phase_gate.rs
  3. Cx = X̃X̃ᵀ/(n-1) + λI, Cy likewise    — λ above the format noise floor,
                                          capped by the damping budget (riir-clippy
                                          Batch-54 rule; guard `!is_finite() ||`)
  4. Wx = ns_inv_sqrt_psd(Cx), Wy = ...   — shipped (newton_schulz.rs)
  5. M = Wx·Cxy·Cy⁻¹·Cxyᵀ·Wx              — symmetric PSD, kx×kx
  6. symmetric_eig(M) -> λi; ρi = √clamp(λi, 0, 1); ρ̄ = mean
     kx==0 -> degenerate flag (that flag IS the collapse signal — see 2.3)
```

Sample-space variant (n×n eigenproblem) when d > ~256 (the Jacobi ceiling in `data_probe/geometry.rs`) — relevant for gemma2 d=2304 activations, NOT for our 8/16/32/64-dim latents.

### 2.2 Signal-diff vs shipped cousins (mechanism level, read from code)

| Shipped cousin | What it consumes | What SVCCA consumes | Verdict |
|---|---|---|---|
| `cka_linear` (`mag/transfer.rs:537`) | feature Grams, `tr(Cx·Cy)/‖·‖‖·‖` — **orthogonal**-invariance only, single scalar, no denoise, no spectrum | whitened cross-cov eigen-spectrum — **full affine** invariance, ρ spectrum + aligned directions, SVD denoise separates noise-aligned from useful-different | cousin, not coverage (the CKA paper's own critique of CCA cuts both ways: CKA keeps functional sensitivity CCA discards — keep both, different gates) |
| PEIRA (`peira.rs`, Research 115) | two views, **learned** alignment (EMA covariance + aux loss, ~500 iterations) | two fixed activation matrices, **closed-form** snapshot compare | different mechanism (iterative vs one-shot); PEIRA is training-shaped, SVCCA is gate-shaped |
| `ProcrustesAdapter` / `SubspaceAdapter` (katgpt-canon, Research 459/406) | two snapshots at **fit time**, orthogonal transform / joint SVD | any two populations, runtime, affine-invariant | orthogonal-only ⊂ affine; canon adapts, SVCCA measures |
| Rosetta (Research 178) | per-neuron Pearson + best-buddies — axis-aligned | subspace-level | Rosetta is literally the neuron-aligned baseline SVCCA Fig.2 beats |
| erank / gaussianity / dist_guard (743) | **one** population vs absolute floors | **two** representations vs each other | the missing comparative axis; nothing diffs audit t vs t−k today |
| `can_freeze` gate (ndb `phase_gate.rs`) | n_wake_events, intrinsic_dim (same η=0.99!), spectral_flatness < 0.3 — all absolute, one snapshot | ρ̄(before, after) — convergence measured directly | complementary: gate v2 = absolute ∨ comparative |

### 2.3 Fusion (the Super-GOAT candidates — novelty TBD, gated on Issue 684 + PoC)

- **Fusion A — Semantic-integrity tier** (`MerkleFrozenEnvelope` × `svd_cca`): BLAKE3 says *same bytes*; ρ̄ over a fixed BLAKE3-seeded probe battery says *same function*. Today a byte-different semantically-fine migration and a byte-different semantically-broken corruption are indistinguishable at thaw/hot-swap time (verified: `species_transition.rs` is a threshold classifier, not an equivalence check). Selling point if it PoCs: *"adapter hot-swap with mathematical proof the mind survived."*
- **Fusion B — Converged-representation freeze** (`can_freeze` × bottom-up ordering): replace/augment the flatness heuristic with ρ̄(pre-consolidation, post-consolidation); commit cognition layers to Cold tier **bottom-up** (perception stabilizes before personality — the paper's law transplanted onto a pipeline that never trained anything).
- **Fusion C — Affine-invariant belief alignment** (`evolve_belief`/`GenericSpatialBelief` × `svd_cca` × KG triples): two NPCs with permuted internal bases still align (ρ̄ high) where per-neuron cosine fails → `aligned_with`/`divergent_from` KG triples from subspace overlap. Prior-art agent: **zero published CCA-in-game-AI hits** (closest: RSA of deep-RL agents, bioRxiv 2021).
- **Fusion D — DFT-equivariant field CCA** (`katgpt-dec` cochains / heightfield tiles × Theorem 1): per-frequency blocks decouple → exact subspace comparison of spatial fields at O(d² per frequency).
- **Fusion E — Retrieval-space drift** (riir-clippy corpus centroids): quantify how much the retrieval space moved on corpus growth instead of blind floor re-pinning.

### 2.4 Game-context reframe

Bottom-up convergence is a **cognition-stack staging law**: sense → HLA → emotion → policy layers converge at different rates; per-layer ρ̄ trajectories give (a) staged commitment order, (b) a class-information profile (`cca(layer, event-label)` — "where does threat-info form?") that says which layers are safe to freeze/compact, (c) a crowd-level emergent: alignment clusters from Fusion C.

## 3. Path 0 decomposition (§3.5, two questions per component)

| Component | Coverage | Extraction without GD? |
|---|---|---|
| SVCCA operator | PARTIAL (2.2 table) | **YES — closed-form**; strongest finding class (open-primitive candidate) |
| Variance-keep truncation (WPᵀPx) | EXISTS (numerical_rank η=0.99, `semantic_axes`, spectral_rewire) | covered; new use is comparative |
| Bottom-up convergence ordering | ABSENT as measurement | YES — measured law → derived schedule |
| Freeze-training regime | ABSENT in-repo (whole-base freeze only; FreezeOut/AutoFreeze published) | criterion modelless; regime = training → **dual-track** (ndb staged commitment + riir-train Plan 349) |
| DFT block-diagonalization | PARTIAL (DFT embedder precedent; DEC equivariant fields) | YES — exact algebra |
| Class-info localization | PARTIAL (`batch_quality_gate` scores directions, not layers) | YES |

## 4. Prior art (verified IDs; full report in session log)

- **Successors:** CKA [1905.00414] (orthogonal-only invariance, no hyperparameters, identifies cross-init correspondence — the honest tradeoff), PWCCA [1806.05759] (usage-weighted ρ̄), critiques [2210.16156, 2202.00095], survey [2305.06329], Platonic [2405.07987].
- **Freeze training:** FreezeOut [1706.04983] (contemporaneous, fixed schedule), AutoFreeze [2102.01386] (gradient-triggered), Yuan et al. [2209.11204] (similarity-informed). **No published CCA-as-live-freeze-trigger found** — the concept space is crowded though.
- **Low-rank truncation:** decade-deep (1404.0736 → DLRT 2205.13571 / 2305.19059 / 2410.18720, TensorGPT 2307.00526, SVD-LLM 2403.07378). Covered.
- **Runtime/inference-time representation similarity:** adjacent-but-not-CCA (git re-basin 2209.04836, FedMA 1905.12022 — matching; SN-Net 2302.06586 — learned stitching; production drift monitors — cosine/PSI, none CCA). **Adapter-equivalence validation + CCA-gated operations: zero hits.**
- **Game AI / agent belief:** zero hits.

## 5. Verdict

**Gain.** The operator doesn't ship, is actionable (substrate 100% present, ~200 LOC), and unblocks gates the stack lacks (semantic integrity, convergence comparison, cross-time monitoring). Not Super-GOAT this session: Q1 is partial (the math is published 2017; the runtime/game applications are novel-but-adjacent), Q2/Q3 (new behavior class / selling point) are **unproven until the primitive lands and a defend-wrong PoC runs** — per the no-candidate rule, Fusions A/C stay "fusion idea, novelty TBD" until then.

**MOAT gate:** primitive → katgpt-rs (fundamental base math, leaf-clean, no game semantics ✓); training recipes → riir-train (active moat, Plan 349 ✓); downstream consumers (ndb gate v2, riir-ai belief alignment) file in their own repos when the primitive exists.

**Latent/raw boundary:** ρ̄ and kx are scalars — may cross sync as belief-health stats; the subspaces themselves stay local (latent). **Not UQ-bearing** (similarity, not a predictive distribution) — the conformal-floor rule does not bind.

## 6. What ships where

| Output | File |
|---|---|
| Open primitive (opt-in `svd_cca` feature, G1–G4 gates) | katgpt-rs `Issue 684` |
| Training recipes (monitor → adaptive freeze → measured ranks → distill selection) | riir-train [Plan 349](../../riir-train/.plans/349_svcca_freeze_training_recipes.md) |
| Consumer follow-ups (file on primitive landing) | ndb `can_freeze` v2 + staged bottom-up commitment; riir-ai hot-swap equivalence gate + belief-alignment PoC |

**Hazards (all mitigated by shipped substrate):** rank-deficient whitening (Batch-54 ε rule + `!is_finite()||` guard); n ≥ d sample gate (already `input_sufficient`, Wang Thm 4); degenerate inputs are the collapse *signal*, not an error; determinism via fixed iteration counts; all consumers sit at Warm/Glacial cadence (sleep cycle, swap boundary, checkpoint, corpus commit) — nothing on the 20 Hz tick.
