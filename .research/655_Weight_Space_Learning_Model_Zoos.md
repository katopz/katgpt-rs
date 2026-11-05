# Research 655: Weight Space Learning — Populations of Neural Networks (GeoSANE + WeightCLIP)

> **Source:** Damian Borth (Universität St. Gallen), "Weight Space Learning: Learning from Populations of Neural Networks" — keynote, Weight Symmetry Workshop @ ICML 2026. Featured papers: **GeoSANE** — Hanna, Falk, Yu, Borth, CVPR 2026, arXiv:2603.23408 (code: `github.com/HSG-AIML/GeoSANE`); **WeightCLIP** — Asefaw, Tzevelekakis, Falk, Meynent, Borth, ICML 2026, arXiv:2607.03551. Field context: ICLR 2025 Workshop on Weight Space Learning; survey arXiv:2603.10090 (already cited in Research 648).
> **Date:** 2026-11-05
> **Status:** Active
> **Related Research:** 648 (weight-space programming — the read/write-weights cousin), 178 (Rosetta Neurons — model-zoo unit mining), riir-train 067 (cross-game LoRA alignment), riir-infer 022 (delta transport; Git Re-Basin naming disambiguation), riir-train 490 (init-confound protocol — the audit tool for GeoSANE-class claims)
> **Related Plans:** riir-train Plan 487 (population warm-start lane — Phase 0 active, Phase 2 conditional)
> **Files created this session:** `riir-train/.plans/487_wsl_population_warm_start.md`. (A v1 draft also filed katgpt-rs issue 984 for symmetry canonicalization; verdict round 1 found its named consumers are same-anchor + already basis-free — verified in source this session — so the issue was FOLDED into Plan 487 T1.2 as a conditional dependency and the file removed pre-commit. Never landed.)
> **Classification:** Public

---

## TL;DR

Weight Space Learning (WSL) treats trained weights θ as a data modality: a **model zoo** (population of nets trained under varying dataset/architecture/hyperparameters/task/seed) → encoder g: Θ→Z → decoder h: Z→Θ → downstream tasks (property prediction from raw weights; weight synthesis / initialization generation). GeoSANE generates task-conditioned weight *initializations* from a zoo latent (+18–25% over training-from-scratch); WeightCLIP contrastively aligns *dataset* embeddings with *weight* embeddings so a new dataset's prompt yields an initialization, and refines in latent Z rather than weight space. **The field is published** (survey arXiv:2603.10090, ICLR-2025 workshop, both featured papers) — this note makes no mechanism-novelty claim. What is actionable HERE: our stack owns real, growing, metadata-rich weight populations (specialist heads, LoRA adapters, checkpoint archives) that nobody has pointed a population-level method at. Filed: a **riir-train plan** (zoo hygiene + low-rank population warm-start for specialist heads, gated against a retrieval-over-zoo modelless floor and Research-490 init-confound discipline) and a **katgpt-rs issue** (weight-symmetry canonicalization before cross-checkpoint comparison). The weights-only *audit* line remains the standing Research-648 Track C discard.

**Distilled (no mechanism claim — adoption):**
The zoo view is the transferable frame: every training lane that persists an artifact is implicitly building a model zoo, and a zoo without unified metadata (dataset, task, hyperparameters, seed, eval, digest) is a zoo nobody can learn from. Symmetry (hidden-unit permutation, ReLU scale) makes coordinate-wise comparison of two trained weight vectors basis-arbitrary — canonicalize before any cross-checkpoint metric.

---

## 1. Keynote core content (compressed)

1. **Three knowledge locations:** dataset (distillation/synthesis), model parameters (compression/distillation/inference of latent factors), context (RNN state, attention, prompts). WSL is the parameters axis.
2. **Formal frame:** population θ ~ P(D, A, λ, τ, E); mechanistic view (raw params), functionalist view (activations), dynamics view (gradients/momentum). Canonical pipeline: model zoos → encoder g_θ / decoder h_ψ → discriminative (performance prediction, epoch/lineage tracking, dataset attribution) + generative (weight synthesis, initialization generation, compression, ensembles) tasks.
3. **Symmetries:** hidden-unit permutation (swap units + in/out weights ⇒ same function) and scale invariance (ReLU: layer l ×α, layer l+1 ×1/α). θ₁ ≠ θ₂ with f_θ₁ = f_θ₂ — coordinate-wise comparison is basis-arbitrary.
4. **GeoSANE (CVPR 2026):** "geospatial model foundry" — transformer encoder over a zoo of remote-sensing foundation models; task-prompt + KDE sampling in Z; decoder emits weights for a target architecture. +18.5% (RESISC-45 ViT-L), +24.8% (fMoW), +20.4% (BigEarthNet) vs from-scratch; competitive with data-centric RSFMs (SatMAE/CROMA/Scale-MAE/TerraFM) without pretraining on pixels.
5. **WeightCLIP (ICML 2026):** contrastive alignment of dataset embedding e_D and weight embedding z_w; same-dataset models cluster tightly; unseen-dataset prompt ⇒ usable init; **latent-space refinement** (gradient steps in Z, decode once) beats weight-space fine-tuning at equal step budgets (+8.3%, +12.0%).
6. **Method taxonomy:** weight statistics (StatNN, LinDet, PhyDet, MoTHer — modelless class); autoencoders (SANE line); equivariant nets (DWSNet, NFN, NFT, ScaleGMN); graph (G.pt, GMN); diffusion (HyperDiff, P-diff). **Open challenges:** (1) zoo scaling/standardization, (2) symmetry-aware backbones, (3) unified benchmarking.

## 2. Prior art

### 2.1 Internal-first (grepped this session, all `.research/` + `.plans/` + `.issues/`)

| Hit | Relation |
|---|---|
| **648 Weight-Space Programming** (katgpt-rs) | The closest cousin. Covers LLM-authored weight *edits/reads*; cites the WSL survey (2603.10090) + Weight Oracles. Its **Track C (weights-only audit) audited discard** — Weight Oracles prior art covers property-from-weights; our frozen brains are 8-dim scale. This keynote is *generation/population*, not audit → discard STANDS (no new evidence). |
| **178 Rosetta Neurons + riir-train 067** | Model-zoo MINING (cross-model/cross-LoRA unit alignment via Pearson + best buddies) — the discriminative/population half in modelless form. GeoSANE's delta is the *generative decode*; WeightCLIP's delta is the *dataset-conditioned* axis. |
| **riir-infer 022 delta transport** | Weight-space *transport*, with the explicit Git Re-Basin disambiguation (permutation sync ≠ delta sync) — feeds the deferred canonicalization dependency (Plan 487 T1.2). |
| **riir-train 490 pre-pretraining** | The init-confound protocol: a "5.4% method gain" closed entirely by fixing the baseline's init. Directly binding on GeoSANE's headline (init-vs-scratch comparison). |
| 588 AWQ, 439 NVFP4, 312 BabelTele | Weight *formats/codecs* — adjacent substrate, not population learning. |

### 2.2 Published (searched 2026-11-05)

- GeoSANE: verified published — CVPR 2026 (openaccess + poster), arXiv:2603.23408, code `HSG-AIML/GeoSANE`. Builds on the SANE line (SANE/SANE-B/SANE-HF, Hanna et al.) — the autoencoder-over-zoo mechanism is published.
- WeightCLIP: verified published — arXiv:2607.03551, ICML 2026 poster; OpenReview PDF.
- Field: ICLR 2025 Workshop on Weight Space Learning (weight-space-learning.github.io); survey arXiv:2603.10090; method families per the taxonomy all published (p-diff, HyperDiff, G.pt, NFN, ScaleGMN, MoTHer, StatNN).
- Conclusion: **Q1 (no prior art) = NO, deliberately** — no mechanism-novelty claim is made; the filing is adoption + gating, scored as Gain.

## 3. Path 0 inventory (three-track panel; advocate rows merged, coverage verdicts are coordinator's)

| # | Component | Nature | Coverage (shipped analog + signal-diff) | Outcome |
|---|---|---|---|---|
| 1 | Learned weight encoder over zoo (both papers' backbone) | training | Modelless analog: weight-stat fingerprints (advocate row) — **converges with 648 Track C audited discard** (Weight Oracles prior art; 8-dim brain scale; re-open triggers unchanged, GeoSANE adds no audit need) | audited discard (Track C stands) |
| 2 | Decoder → warm-start initialization (GeoSANE core) | training | Modelless floor: **retrieval-over-zoo** — pick the closest stored head (suite-similarity via signed `corpus_centroid` simhash, instinct) and transplant its weights. Generation must BEAT this floor or demote | riir-train Plan 487 Phase 2 (GOAT-gated, retrieval as pre-registered baseline arm) |
| 3 | Contrastive dataset↔weight alignment (WeightCLIP) | training | Dataset half SHIPS: signed simhash `corpus_centroid` (instinct, Bench-003 GOAT). Weight half: modelless stats only; learned alignment needs GD | Plan 487 Phase 3 (DEFERRED, secondary) |
| 4 | Latent-Z refinement (GD in Z, decode once) | training | Modelless recast changes the mechanism class (pool reweighting ≠ GD); the recipe itself is training. Serving-envelope check: training-time lane, not hot path — 4–8 GPU-h affordable but the consumer gain is unproven | Plan 487 Phase 3 (DEFERRED, secondary) |
| 5 | Discriminative tasks from raw weights (perf/lineage/attribution prediction) | modelless class | Same row as #1 — 648 Track C discard + Weight Oracles. Lineage half ships (`MerkleFrozenEnvelope` v2 `parent_commitment`, epoch tags) | audited discard / ships |
| 6 | Permutation + scale canonicalization (challenge #2) | modelless math | Mechanism PUBLISHED (Git Re-Basin class). **v1 correction (verdict round 1, source-verified): the named consumers do NOT have the problem** — `sst_cross_tail` prices "two same-anchor tail factors" via "basis-free alignment SCALARS in uncentered-linear-CKA form" over "sorted (stable, total_cmp) eigenpairs and sign-canonicalized columns" (same-anchor ⇒ shared basis; linear-CKA is invariant to orthogonal transforms), and `sst_merge_admission` prices "against each same-anchor sibling". Same-anchor lineages inherit the anchor's unit order; the primary zoo population (instinct specialist heads) is LINEAR bag-feature classifiers — no hidden units, no symmetry at all. **No live consumer today** | folded into Plan 487 T1.2 (DEFERRED conditional dependency, with the corrected mechanism taxonomy: sort-key orbit representative ≠ Re-Basin pairwise matching; order-independent tie-break; per-unit α_i; power-of-two-only bit-identity) |
| 6b | Closed-form population PCA warm-start | modelless math | A rank-k linear autoencoder trained on reconstruction loss converges to PCA (Baldi–Hornik) — the decoder's first rung has a closed form: SVD over the zoo, no GD, no GPU-hours. At N ≤ 50 heads the decode is an affine combination of zoo members (≤ N−1 components) — barely past retrieval | Plan 487 T2.1 (modelless arm, conditional on T2.0 bar); generic primitive could later lift to katgpt-core if it clears |
| 7 | Zoo-hygiene metadata protocol (challenge #1) | modelless protocol | PARTIAL: riir-train `data/` archive + `data_backup.sh` (BLAKE3-verified HF lanes), instinct winners dir + manifest A5/A6 pins, freeze-envelope provenance sidecars. Gap: no unified (D, τ, λ, E, eval) metadata row per trained artifact | Plan 487 Phase 0 (0 GPU-h precondition) |
| 8 | Weight-space learning at 27B / laya-Q8 scale | training | **Non-viable on our compute** (advocate honesty row): encoder training over 27B params exceeds one GPU; laya Q8 checkpoints get statistics-only encoding at most | scoped OUT of Plan 487, recorded |

**Panel discards (with auditable reasons):**
- Weight-stat property prediction as a *new* katgpt-rs primitive — discarded: 648 Track C's published-prior-art + scale discard covers the same signal with unchanged re-open triggers; this keynote demonstrates generation, not audit, so it provides no new in-stack failure to fill.
- "Prompt-conditioned generation recast as pure retrieval" as the *primary* filing — discarded as the mechanism claim (retrieval is a baseline arm inside Plan 487's gate, not a replacement filing): if retrieval wins the gate the decoder demotes, which is the pre-registered outcome, not a verdict swap.

## 4. Fusion (what keynote × stack produces that neither alone has)

- **GeoSANE's task-conditioned generation × instinct's arsenal manifest** (suite → winner-head seating, A5/A6 byte-pinned): *warm-start-on-arrival* — a new suite's head initialized from the population latent instead of trained from scratch. The serving family onboards suites continuously; every new suite is a warm-start consumer.
- **WeightCLIP's dataset side × `corpus_centroid`** (signed simhash, already shipped and GOAT-tested): the dataset embedding half of the contrastive pair EXISTS; only the weight half + alignment are new — that is why Phase 3 is cheap-to-defer rather than a from-zero build.
- **Zoo view × freeze/thaw discipline**: `MerkleFrozenEnvelope` + `AnchorSidecar` + `data_backup.sh` already commit artifact bytes with provenance; the missing piece is a unified metadata SCHEMA (Phase 0), not new infrastructure.
- **Symmetry challenge × weight-comparison instruments — CORRECTED to a non-consumer row:** `sst_cross_tail` and `sst_merge_admission` are same-anchor instruments that are already basis-free (uncentered-linear-CKA over Grams; sorted + sign-canonicalized eigenbases) — verified in source this session (verdict round 1). The canonicalization work item survives only as Plan 487 T1.2, conditional on a future hidden-unit population joining the zoo.
- **Closed-form first × the generative-init lane:** a linear autoencoder IS PCA (Baldi–Hornik), so the GeoSANE-shaped decoder's first rung is a modelless SVD — Plan 487 schedules the closed-form arm before any GD arm, and the T2.0 arithmetic (span{zoo members} at N ≤ 50) predicts it will likely read "premature" today. That recorded outcome prices WHEN the lane re-opens, which is the useful deliverable at small N.
- **490's init-confound protocol × GeoSANE's init-based headline**: our own protocol is the audit tool for exactly this claim class — "gains vs from-scratch" must ablate init on BOTH arms. Plan 487's gate carries it.

**Two-sided priority check (fusion priority #1):** *Rulewright* — riir-refine has no weight population (corpus/Elo/trajectories are non-weight; `TernaryDraftModel .bits` is a single artifact line, no population) → no load-bearing consumer, honestly noted. *Decision-serving family* — instinct specialist heads are a real, growing per-suite weight population seated by the manifest → Plan 487 aims at its training cost (suite onboarding). Priority-1 surface served via its training lane.

**Stationary-surface check:** the warm-start lane is training-time (mutated optimizer/AE state exists — dynamics legitimate); no ESC/exact-gate surface is touched.

## 5. Verdict

**Pinned claim (one sentence):** *population-level warm-start for the stack's specialist-head populations (zoo metadata hygiene → symmetry-canonicalized population autoencoder → decoded init), gated to beat BOTH a retrieval-over-zoo modelless floor AND the strongest closed-form init under Research-490 init-confound discipline — the WSL field and both featured papers are published; no mechanism novelty is claimed.*

**Novelty gate:** Q1 no-prior-art = **No** (field published; both papers published; SANE/p-diff/NFN lineages published). Q2 new behavior class = No (warm-start init is an old idea; population-conditioning is the field's own idea). ⇒ **Not Super-GOAT, not GOAT-mechanism.**

**Per-track verdicts:**
- **Track A (modelless): Gain** — files: this note + the MODELLESS arms inside Plan 487 (T2.0 arithmetic stop, T2.1 closed-form PCA decode, T1.1 stat vectors, T1.2 deferred canonicalization with corrected taxonomy). No new inference primitive ships now; the PCA decode is modelless-validable and could lift to katgpt-core only if it clears the gate. The audit line stays discarded; the canonicalization issue was folded after round-1 verification (no live consumer).
- **Track C (model-based): Gain → Plan** — `riir-train/.plans/487_wsl_population_warm_start.md` (Path 0.5: applicable, actively pursued). Phase 0 zoo hygiene ACTIVE (0 GPU-h); Phase 2 CONDITIONAL — the T2.0 sufficiency arithmetic (N ≥ 40 + ≥3 label-compatible suites) is expected to read `premature` at today's N ≈ 10–50, with closed-form PCA first and the MLP GD arm gated behind PCA; Phase 3 deferred secondary.
- **MOAT gate:** katgpt-rs note = public area map, zero game/chain/shard IP ✓; riir-train Plan = active training moat ✓; issue = generic public math ✓. No rethink-class content in any public file ✓.

**Demotion law (pre-registered):** if Phase 2's learned arm fails to beat BOTH baseline arms, the negative is recorded and the lane demotes; Phase 0 (metadata) and the deferred canonicalization dependency (Plan 487 T1.2) stand on their own regardless — they are infrastructure, not claims.

**Verdict negotiation record:** round 1 REVISE (claude-sub-agent-verdict, session bd7cee20) — findings all addressed with source evidence: (1) `sst_cross_tail`/`sst_merge_admission` re-read this session; both same-anchor + already basis-free → removed as canonicalization consumers, issue 984 folded into Plan 487 T1.2, file removed pre-commit; (2) linear AE = closed-form PCA (Baldi–Hornik) → PCA promoted to the first learned arm, modelless track; (3) T2.0 sufficiency arithmetic added (span-of-zoo argument; N ≥ 40 bar; `premature` as expected outcome); (4) head architecture pinned as a verified premise + T0.6 label-space rule; (5) canonicalization corrections carried (order-independent tie-break, per-unit α_i, power-of-two-only bit-identity, sort-key ≠ Re-Basin).

## 6. Scale-transfer caveat (the honest limit)

GeoSANE's zoo is thousands of HuggingFace models; our specialist-head population is O(10–50) today — and it is LINEAR (bag-feature classifiers), so it carries no hidden-unit symmetry at all. Two consequences the v1 draft missed and the verdict caught: (1) with k ≤ N−1 components, any decoded init is an affine combination of zoo members — the interpolation argument puts the learned decode barely past the retrieval floor at this N; (2) the genuinely-GD rung (nonlinear MLP AE) is both gated behind the closed-form PCA arm AND likely premature. The keynote's own open challenge #1 (zoo scaling) puts us on the wrong side of the curve — which is exactly why the plan is (a) metadata hygiene now, (b) a pre-computed sufficiency bar (T2.0: N ≥ 40) that records `premature` as the expected measured outcome, and (c) a demotion law that keeps the metadata layer as the standing deliverable. A failed or premature feasibility arm prices WHEN the lane re-opens.
