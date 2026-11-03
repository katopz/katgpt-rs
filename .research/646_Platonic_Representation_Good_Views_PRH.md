# Research 646: Platonic Representation Theorem — Good Views, Bad Views, and the ρ Alignability Floor

**Status:** Done — verdict Gain (laws adopted; PoC + recipe-applicability issues filed)
**Source:** [arXiv:2610.39302](https://arxiv.org/abs/2610.39302) — *Proving and Improving the Platonic Representation Hypothesis with JEPAs*, Randall Balestriero & Yann LeCun (Brown/NYU/AMI Labs), 2026-10-29
**Date:** 2026-11-03
**Related Research:** 540 (vec2vec unpaired embedding translation — the closest cousin; this paper is its theory companion), 498 (LeVLJEPA SIGReg → `gaussianity_probe`), 504 (Orthogonal JEPA → `orthogonal_factorization`), 459 (katgpt-canon adapters), 371 (gw_alignment)
**Related Plans:** riir-train Plan 391 (vec2vec translator — this paper adjudicates its modelless arm), katgpt-rs Plan 625 (`basis_split`)
**Classification:** Public

---

## TL;DR

The Platonic Representation Hypothesis is now a theorem for JEPAs: separately-trained encoders of one world agree up to one rotation **iff their positive pairs relate the same world states** ("good views"); positives that corrupt one observation ("bad views") spend embedding dimensions on modality nuisance no rotation can recover, and next-token prediction is a bad-view JEPA identified only up to **GL(k)** (invertible linear), not O(k). Every derived quantity is closed-form measurement-time linear algebra — the modelless posture is the paper's native one. Internal novelty on the headline coupling is **killed/weakened by concurrent published art** (the ρ measurement and ranking are the paper's own results; Schnaus et al. 2610.09411 predicts alignability pre-fit from cross-set statistics). Verdict **Gain**: four workspace laws adopted (§3), one operational PoC filed (katgpt-rs Issue 978 — ρ alignability gate), one recipe-applicability gate filed (riir-train Issue 663 — RIDT q measurement + laya pair-free probe that adjudicates Plan 391).

**Distilled for katgpt-rs (modelless, measurement-time):**
ρ(f) = tr(Cov(E[f(x)|z])^{1/2})/k over grouped multi-view embeddings — the state-explained variance fraction — lower-bounds best-rotation cross-space alignment error by 2k(1−min ρ) with no pairs, no labels, no training. Plus the two identification laws: no unpaired first-two-moment statistic identifies the rotation (only k pairs or higher-order finite-world geometry), and next-token spaces are only linearly mappable (GL(k)), never rotation-alignable.

---

## 1. Paper Core Findings

| Result | Content |
|---|---|
| **Thm 1 (Platonic Representation Theorem)** | M ≥ 2 modalities sharing one positive-pair distribution π, minimizers unique up to rotation ⇒ every separately-trained encoder satisfies f*_m ∘ g_m = Q_m·h* — one world embedding, one rotation each. |
| **Prop 1 (spectral form)** | Whitened MSE + spectral gap ⇒ minimizers = top-k eigenvectors of the normalized positive-pair adjacency K_π (Laplacian-eigenmaps); min L = 2Σ_{i≤k}(1−μ_i); unique up to rotation iff μ_k > μ_{k+1}. |
| **Thm 2 (kernel mismatch bound)** | Best-rotation RMS distance between two spectral embeddings ≤ (2√2/δ)·min(√k·‖K₁−K₂‖₂, ‖K₁−K₂‖_F), δ = eigengap (Davis–Kahan variant). |
| **Prop 2 (identification)** | (i) NO statistic of unpaired samples' first two moments identifies the rotation — CKA, centroid cosine, and every whitening-invariant second-moment metric are structurally blind to misalignment; (ii) k pairs in general position determine it (orthogonal Procrustes, closed-form SVD). Finite-world caveat: higher-order geometry CAN reveal it pair-free (Schnaus et al. validate). |
| **Thm 3 (nuisance costs alignment)** | ρ(f) = tr(Cov(E[f(x)|z])^{1/2})/k ∈ [0,1] (scene-explained fraction); best-rotation error ≥ 2k(1−min ρ); ρ = 1 iff f(x) is a function of the state alone. Empirically ρ ranks encoders exactly as measured alignment does (Spearman .98). |
| **Prop 3 (good vs bad views)** | Good views (two observations of one state, independent nuisances) ⇒ ρ = 1; bad views (two corruptions of ONE observation, sharing its nuisance) spend r of k directions on nuisance whenever the corruption keeps nuisance more than the state's k-th direction (β₁ > α_k), losing ≥ 2r alignment. |
| **Thm 4 (views + data set the kernel)** | A spectral JEPA learns the world kernel π_W induced by its views and data; good views induce the world's own kernel (τ = 0); bad views get their kernel from data density (weak-corruption overlap chains scale Δ ∝ log N/N). |
| **Prop 4 (next-token = bad-view JEPA)** | NTP's positives = (prefix, continuation) within one document; anti-collapse = softmax log-partition; minimizers identified only up to **GL(k)**, not O(k). Contexts sharing a continuation become implicit positives — very weak views needing very dense data, plus spill through frequent tokens. |
| **Empirics** | Good views need 42× fewer samples than NTP on the same corpus; 109M text JEPA beats LLMs up to 72B on the PRH benchmark (.184 vs .154); trimming is THE knob (whole captions −25% alignment; everything else ±.016 ≈ seed spread); alignment scaling law predicts unseen runs within seed noise; captioner whose only pair-fitted part is one rotation (CIDEr .838 COCO); pair-free alignment succeeds on good-view encoders and FAILS on NTP encoders (FOSCTTM .287 vs .018). |

Paper's own caveat (load-bearing): the scale comparisons ride Recap-DataComp texts where 2 of 3 views per image are LLaVA recaptions — paired information enters **through the definition of positives**, never a cross-modal loss. The between-views comparisons (good vs bad vs NTP on identical texts) do not.

## 2. Modelless extraction ledger (No-GD advocate merge)

All nine derived quantities are closed-form over precomputed embeddings/kernels/pair sets — 9/9 modelless-feasible:

| # | Quantity | Inputs | Ships where it would live |
|---|---|---|---|
| 1 | ρ state-explained fraction (Thm 3) | embeddings + group labels, whitening | new measurement primitive (Issue 978) |
| 2 | Alignment floor 2k(1−min ρ) | ρ of both spaces | same |
| 3 | Post-Procrustes residual (Prop 2ii) | k paired embeddings | `katgpt-spectral::procrustes`, `katgpt-canon::ProcrustesAdapter` — SHIP |
| 4 | Kernel-mismatch bound (Thm 2) | K₁, K₂, eigengap δ | advisory; needs kernel access (weaker extraction) |
| 5 | Spectral floor 2Σ(1−μᵢ), gap-uniqueness | positive-pair graph | `gw_alignment`/MAG-adjacent; diagnostic |
| 6 | View-class test β₁ > α_k (Prop 3) | corruption vs state spectra | `latent_confounder_audit`-adjacent |
| 7 | GL(k)-vs-O(k) residual gap (Prop 4iii) | two embedding sets, shared items | adapter-selection rule (§3 Law 2) |
| 8 | τ = view-kernel vs world-kernel drift (Thm 4) | proxy world kernel | corpus-design rule (§3 Law 4) |
| 9 | Alignment scaling-law coefficients | measured R@1 grid | go/no-go instrument (riir-train Issue 663 T1) |

**The ρ recipe** (adapted from the paper's Table 6 estimator; the advocate's form, corrected in review): whiten f over the population (Cov → I); per-group mean m_g over views; between-group covariance CORRECTED for finite-group spread — subtract the mean within-group covariance / n_g (the paper subtracts a fifth of the mean within-image covariance for its 5-view groups; healer format-pairs have n_g = 2, where the uncorrected estimator overstates Cov(E[f|z]) by ≈ half the within-variance); ρ = Σᵢ√λᵢ(C)/k. One eigendecomposition. Caveats pinned: ρ REQUIRES grouping labels (where none exist, Prop 2(i) forbids the unpaired fallback — correctly undefined, never silently computed); the whitening convention must be pinned per space or ρ values are incomparable; tiny δ makes Thm 2 vacuous — report δ beside every bound.

## 3. The four workspace laws (the durable distillation)

1. **The paired-alignment metric law (Prop 2i).** No unpaired first-two-moment statistic identifies the rotation: MAG's `CentroidCosine`/`CkaLinear` and every whitening-invariant second-moment metric can DETECT gross mismatch but can never CERTIFY alignment — any alignment claim must be paired (Procrustes on k anchors) or rest on higher-order structure (the finite-world caveat, which is exactly `gw_alignment`'s design premise and `leakage_probe`'s honest `InsufficientAlignment` verdict). TransferMetric's headline should be residual-after-Procrustes, with CKA demoted to invariant-baseline. Explains at the mechanism level why Bench 424–427's four hidden-state methods could not certify cross-arch agreement.
2. **The GL(k) law for LLM-derived spaces (Prop 4iii).** Next-token-trained spaces are identified only up to GL(k): the BINDING direction is cross-objective ⇒ a rotation-only adapter (`ProcrustesAdapter`) is structurally INSUFFICIENT — refuse it when the two sides' training objectives differ. The converse is NOT claimed: same-objective does not imply rotation-sufficient (two NTP checkpoints are same-objective yet GL(k)-identified); rotation-only sufficiency for same-lineage pairs is katgpt-canon's own empirical validation (Bench 562, same-arch snapshot swap), not this paper's theorem. Cross-objective pairs need invertible linear maps (SubspaceAdapter class) or retraining. Corollary for the sync law: cross-objective spaces cannot be rotation-canonicalized, so wire crossings keep scalar projections — never raw vectors.
3. **The nuisance-carrier diagnosis of NTP embeddings (Prop 3+4).** LLM embeddings carry document nuisance (author wording/style) because their positives share it — the mechanism behind anisotropy lines, behind vec2vec needing a trained translator for LLM-space alignment, and behind the Bench-115 finding that the 8-D rule embedder has no transferable formatting subspace (formatting IS nuisance; nuisances don't rotate across spaces, they must be excluded or the embedder must be selected for ρ).
4. **The good-view corpus principle (Thm 4 + Prop 3).** What a corpus's positive pairs relate determines the kernel its representations induce. The healer's (finding, fix) pairs are same-file — bad-view-shaped (shared file nuisance); cross-file same-rule pairs are good views. Eval fixtures for the withheld-pair OOD axis should prefer withheld CROSS-FILE pairs; the ρ measurement (per embedder, grouped by rule) is the cheap instrument that ranks candidate embedders by how much rule vs formatting variance they carry — complementary to FLINT (riir-refine Research 309, the architecture remedy) and to `basis_split` (paired perm-delta subspace leakage — a targeted two-space measurement where ρ is the one-space, arbitrary-grouping generalization).

## 4. Prior art + novelty verdict

Web search verdict (agent-run, arXiv-abstract level; Scholar rate-limited): **WEAKENED, borderline KILLED as an internal novelty claim.**

- The ρ definition, the 2k(1−ρ) floor, and the encoder-ranking validation ARE arXiv:2610.39302's own results — not claimable.
- **Schnaus et al. arXiv:2610.09411** (concurrent, Oct 2026 — the paper's own §5.2 tool): unpaired Wasserstein-Procrustes alignment + "standard geometric alignment metrics accurately predict when this is possible" — from CROSS-set statistics. Our remaining delta: one-space-only computation (ρ needs no foreign sample).
- **CMC (Tian et al. arXiv:1906.05849)** published the good-view pair design (independent views of one scene vs corruptions of one observation).
- **ILO (arXiv:2609.04819)**: within-model multilingual sharing metric correlating with cross-lingual transfer (Spearman 0.90) — nearest published neighbor; within-one-model, no theoretical floor.
- Moschella et al. (relative representations) + ASIF demonstrate pair-free alignability without predicting it from one-space statistics.
- Landscape: Huh et al. 2405.07987 (position), Ziyin 2507.01098 (deep-linear proof), Doshi et al. 2610.07168 (multilingual theory), dense 2026 follow-up field. The JEPA proof is first in 2610.39302 (confirmed).

Surviving delta: the **operational coupling** — a go/no-go gate spending Procrustes/transport compute only when ρ clears a threshold, and pipeline embedder selection by ρ. No published work found on that exact deployment; it is engineering, not a research-novelty claim. Internal novelty gate Q1 = FAIL ⇒ **not Super-GOAT, not GOAT-titled**; Gain with issue filing.

## 5. Coverage — workspace signal-diffs (all read, none kill the extractions)

| Cousin | Ships | Signal-diff vs ρ/floor |
|---|---|---|
| `katgpt-spectral::procrustes` + `katgpt-canon::ProcrustesAdapter` (Bench 562) | paired orthogonal rotation, validated | covers extraction #3 fully; no predictor of when rotation suffices |
| `gw_alignment` (Issue 743) | structure-only GW quotient alignment of two distance matrices | pair-free alignment METHOD (the finite-world-geometry branch); no one-space predictor, no floor |
| `leakage_probe` (Issue 736, Research 540) | unpaired OT transport + kNN attribute-leak score | end-to-end two-space ATTACK simulation with labels; ρ is the one-space, label-free, pre-transport predictor — a front-end, not a competitor |
| `basis_split` (Plan 625; reflex 086) | paired perm-delta projection onto whitened eigvecs | targeted nuisance (presentation deltas) in a specific eigenspace; ρ is grouping-general, no pairs, whole-space scalar |
| `loop_alignment_probe` (Issue 929) | per-token state-vs-readout cos scalar predicting composition failure | the alignment-scalar-predicts-failure PATTERN precedent; different object entirely |
| `latent_confounder_audit` (Bench 194) | direction-vector confounder diagnostics (R₀/R_shift/L) | audits individual direction vectors for shortcut leakage; ρ is a whole-space variance split over groups |
| MAG TransferMetric suite | CentroidCosine/CKA/Wasserstein1d etc. | unpaired within/across statistics — per Law 1, cannot certify alignment; gain a headline metric (post-Procrustes residual) |

## 6. Fusion lines (for follow-up lanes, not committed here)

- **ρ-gate × leakage_probe front-end**: skip the OT transport entirely when ρ floor already proves misalignment — a measured go/no-go that saves the transport cost and prevents reading noise off `InsufficientAlignment` runs.
- **ρ × healer embedder selection**: rank candidate embedders for the retrieval channel by rule-explained fraction over grouped fixtures — the cheap instrument beside FLINT's architecture remedy (Bench-115 axis).
- **Law 2 × katgpt-canon adapter selection**: a check that refuses `ProcrustesAdapter` when the two sides' training objectives differ (GL(k) class) — currently implicit operator knowledge.
- **Good views × healer mining + reflex distillation**: prefer cross-file same-rule pairs (eval) and measure q on RIDT paraphrase corpora (before any decision-encoder GPU spend).
- **σ-noise matching (R1)**: any future decoder trained over frozen embeddings sets its training noise analytically from the residual cosine σ = (c⁻²−1)^½ — deletes a hyperparameter sweep; zero-cost adopt (riir-train Issue 663 records it).

## 7. Three-track verdict + MOAT gate

- **Modelless track → Gain.** Files: this note + katgpt-rs Issue 978 (ρ primitive PoC with pre-registered falsification arms; promotion consumer-gated per the `latent_confounder_audit` precedent — no consumer, no promotion).
- **Model-based track → Gain (conditional).** riir-train Issue 663: measure q on RIDT corpora + run the laya pair-free probe (adjudicates Plan 391's modelless arm — if rotation+OT matches the trained translator, Plan 391 must beat it or die per modelless-first). Good-view decision-encoder training (R3) has NO current consumer — the advocate's honest finding — so it stays behind Issue 663's T3 decision gate, not a plan filed on spec. Recipes recorded: LeJEPA+SIGReg (λ=.02, 1024 slices), trimming-is-the-knob, clip 1.0 + collapse-window tripwire, ~50–150 A100-h at 33M.
- **Security track → no new surface.** `leakage_probe` + `gw_alignment` already carry the unpaired-transport threat model; this paper strengthens their theory (Law 1) without changing their contracts.

**MOAT gate (katgpt-rs):** fundamental measurement primitive family (spectral/variance diagnostics beside `gaussianity_probe`) — in scope. The theory is public paper content; nothing game/chain/shard/train-private crosses. Consumers are opt-in feature-forwards on existing edges (the `basis_cert` precedent).

## 8. Honest caveats

- ρ needs grouping labels; groups must share one world state with independent nuisances — on single-view corpora ρ is undefined, not zero.
- The paper's scale claims ride captioner-generated positives (paired information via the positives' definition) — quote only the between-views comparisons as objective-free.
- Concurrent art (Schnaus 2610.09411) may claim the operational gate in its own follow-ups; Issue 978's gate is framed as engineering adoption, not novelty.
- Whitening convention and eigengap δ must be reported beside every ρ/bound number.
- No PoC ran this session — the planted-ρ fixture and real-fixture ranking are Issue 978's pre-registered proof path; nothing here claims a measured gain.
