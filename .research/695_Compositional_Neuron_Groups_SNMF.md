# Research 695: Compositional Neuron Groups — SNMF Activation Decomposition, Input-Mapped Features, Recursive Hierarchy (Shafran/Geiger/Geva)

> **Source:** arXiv:2506.10920 "Constructing Interpretable Features from Compositional Neuron Groups" (Or Shafran, Atticus Geiger, Mor Geva — ACL 2026 main; v2 2026-05-02; code `github.com/ordavid-s/snmf-mlp-decomposition`). Surfaced by @GurYoav's public thread noting the Anthropic neuron-basis posts under-credit this line; the three-cousin sweep lives in research 693 §8.
> **Date:** 2026-11-26
> **Status:** Active — **GAIN** (modelless track: this note + riir-infer issue 1255 — a group-granularity census lane with pre-registered bars. No training-track row: the paper trains nothing — SNMF is offline unsupervised factorization of captured activations, not weight training; zero riir-train backlog rows, no Path 0.5 plan).
> **Related Research:** 693 (the Anthropic neuron-basis pair — the observational sibling; its F2 facet accretion and F3 multi-path facts are exactly the sparse-combination and hierarchy shapes this paper constructs), 203 (ROTATE — the weights-only disentangler; SNMF is its data-side complement: ROTATE needs no activations, SNMF needs no weights-only constraint and returns the input map), 180 (Rosetta kurtosis), 388 (J-lens), 666 (neuron-lens lane posture), 286 (depth-invariance — recursive-STATE magnitude; SNMF's hierarchy is a feature TREE, not state recursion — different object)
> **Cross-refs:** `riir-infer/.issues/1255_snmf_group_feature_census_lane.md` · katgpt-spectral `ica_lens` (Plan 475 — the FastICA cousin this lane A/Bs against) · katgpt-pruners `census_screen` (1237) · katgpt-pruners `vocab_channel_pruner` (Plan/Bench 228) · riir-infer 1241 + 1250
> **Classification:** Public

---

## Pinned claim (pre-§4, one sentence)

SNMF over captured MLP activations (A ≈ Z·Y with Y ≥ 0, hard winner-take-all-sparse Z) is a PUBLISHED unsupervised decomposition class whose value here is one lane-local fusion — group-granularity census candidates with free input mapping (the coefficient matrix Y names each feature's activating tokens) and a constructive test for 693's observational hierarchy claims — filed as riir-infer 1255; no mechanism novelty is claimed (the NMF/FastICA lineage is published, and the paper itself is the class-establishing citation).

## TL;DR

The paper decomposes per-layer MLP activations A ∈ R^{d_a×n} with **semi-nonnegative matrix factorization** (Ding et al. 2010): A ≈ Z·Y, where Z ∈ R^{d_a×k} holds the FEATURES — each a sparse linear combination of the MLP's own neurons (sparsity via a hard winner-take-all operator keeping the top p% = 1% of entries by absolute value per column) — and Y ∈ R^{k×n}≥0 holds the COEFFICIENTS: Y[i,j] says how strongly feature i contributed to token j. Multiplicative updates for Y (positive/negative-part split), a closed-form ridge step for Z (`Z ← A Yᵀ (YYᵀ + λI)⁻¹`), column re-normalization each step; converges in ~1.5–3.3k iterations; initialization-robust (Random/SVD/K-Means all land within noise). Features map to the residual stream via `f = W_out·z`, so every feature is by construction inside the model's own neuron basis — no learned-from-scratch dictionary.

Results (Llama-3.1-8B, Gemma-2-2B, GPT-2): **concept steering beats both SAEs (GemmaScope/LlamaScope on MLP-out, and capacity-matched SAEs on MLP activations) and the supervised difference-in-means baseline** — the first unsupervised-in-model-basis method to do so; concept detection is comparable-to-better. The structural findings: (1) **recursive SNMF** (re-decomposing Z at smaller k, then joint fine-tune) exposes a feature hierarchy — weekday features merge into weekend → day-of-week; parent features cover the union of their children's promoted tokens (coverage 0.46 Llama / 0.37 Gemma vs 0.05/0.00 random-parent control) while children stay diverse (0.99); (2) **semantically related features share core neurons** (the binarized Z·Zᵀ overlap census: all seven weekdays share a core set, each day adds exclusive neurons that promote its own token and suppress the others; the shared core raises ALL weekday logits when amplified).

The ROTATE paper (arXiv:2604.06005, our Research 203) cites exactly this work as its activation-side counterpart: "Shafran et al. used MLP activations to discover neuron combinations that capture concepts and outperform SAEs on causal steering. Unlike these works, ROTATE achieves data-free decomposition." The two are complementary by the authors' own framing.

---

## 1. What is genuinely new in the paper (mechanism level)

1. **In-model-basis features by constrained factorization** — features restricted to span{existing neuron directions}, sparsity enforced by a hard top-p% operator rather than an L1 prior. SAEs learn directions from scratch over residual activations; SNMF composes the neurons the model already has.
2. **The coefficient matrix Y as a free input map** — every feature ships with its activating-token attribution computed by the factorization itself, not by a second attribution pass. This is the property none of our direction-acquisition lanes return (FastICA returns directions; NNLS-over-lens-atoms returns per-unit verbalizability, not per-group token attribution).
3. **Recursive SNMF hierarchy** — smaller-k re-decomposition of Z yields parent features; the parent-coverage metric (parent's promoted-token set covers the union of children's) is a quantified, falsifiable hierarchy test, with diversity (1 − child overlap) as its companion.
4. **Causal steering parity with a supervised baseline** — unsupervised combos matching/beating DiffMeans on steering while staying interpretable; SAEs fail the same comparison.

## 2. Path 0 inventory (three-track; per-track verdicts never pooled)

| Component | Coverage here | Extraction (modelless?) | Disposition |
|---|---|---|---|
| SNMF factorization of activations (offline, no weight GD) | NONE — `ica_lens` FastICA maximizes non-Gaussianity across SAMPLES with orthogonal components (no nonnegativity, no input map); `readout_jshare` NNLS decomposes ONE unit over a FIXED lens dictionary | YES (captured activations + multiplicative updates; deterministic seeding required — SplitMix64 keyed on the capture digest, never unseeded-global draws) | **Extracted** → riir-infer 1255 (lane); primitive home proposal: katgpt-spectral, `ica_lens` sibling, opt-in feature |
| Y input mapping (per-feature activating tokens) | PARTIAL — census_screen carries per-unit labels; no per-GROUP token attribution ships | YES | **Extracted** → 1255 T1 |
| Recursive hierarchy + parent-coverage metric | NONE | YES (one cell, verdict recorded regardless of outcome) | **Extracted** → 1255 T2 |
| Shared-core census (Z̄·Zᵀ overlap bands) | NONE — 693 F2 reports facet accretion observationally; no overlap census ships | YES | **Extracted** → 1255 T2 |
| Steering with neuron-combo directions | PARTIAL — `latent_steering` + `calibrate_alpha` (491) is the normalization family; combos are a new direction SOURCE, not a new steering mechanism | YES | **Corroboration** — direction-source option for existing lanes |
| Group-level census pushes (scale a sparse combo, not one column) | NONE — 1241 pushes single columns | YES (mechanical extension of the activation-scale push) | **Extracted** → 1255 T3, gated on the 1241 sweep |

**Training track: empty** — the paper proposes no training recipe (its SAE baselines are baselines it trained for comparison). No Path 0.5 plan, no riir-train backlog row. **Modelless track: GAIN** (1255 + this note).

## 3. Coverage map — §3.6 signal-diffs (every dismissal read at source)

| Shipped cousin | What it consumes/returns | This paper's instrument | Verdict |
|---|---|---|---|
| katgpt-spectral `ica_lens` (FastICA, Plan 475, DEFAULT-ON) | data matrix of observations → orthogonal directions maximizing cross-SAMPLE non-Gaussianity; no input mapping; no nonnegativity | SNMF: factors over the NEURON axis with nonnegative TOKEN-axis coefficients; parts-based additive; Y = the input map | **Different objective, axis, and output shape — not coverage.** The A/B is a T3 ARM in 1255 with the bar pre-stated (SNMF ≥ 1.25× the ica_lens arm decides together with the census_screen comparison; one clear failure closes NEGATIVE) — the paper's related-work claim that PCA/ICA underperform causally is measured on OUR checkpoint, never assumed |
| `readout_jshare` (riir-infer 1250 T2) | ONE unit at a time decomposed over a FIXED lens-direction dictionary (NNLS); returns verbalizability bucket per unit | SNMF learns BOTH factors from activations; returns GROUPS, not per-unit buckets | **Adjacent tier, not coverage** — J-share is the weights-only cheap screen; SNMF is the activation-side group tier between screen and sweep |
| `vocab_channel_pruner` (203/Bench 228) | weight-space, data-free Householder rotations maximizing vocabulary kurtosis; per-neuron token reachability | activation-space, weights-free; needs forward passes, returns combos + input map | **Complementary pair** — ROTATE for data-free disentangling, SNMF when activations exist and input attribution is wanted |
| riir-infer 1241 census | single-column activation-scale pushes, dose-response ladders | group features = multi-column sparse combos | **Granularity extension** — a combo push is a mechanical extension of the existing push seam; gated on the sweep's outcome, never a replacement |
| katgpt-pruners `census_screen` (1237) | per-unit AUROC over captured activations | consumes the SAME captures (1255 T0 is pinned to reuse the census_screen harvest where one exists — one harvest, two consumers), outputs combos of units | **Complementary consumer of one capture** — the lane shares the activation harvest, not the unit; the census_screen top-k arm is T3's deciding baseline (beating random columns proves nothing 1237/1250 have not already established) |
| 301 probe bank construction | contrastive-pair directions in residual space (data-derived, supervised-by-pairing) | learned unsupervised combos in NEURON space mapped to residual via W_out·z | **Different construction source** — recorded as an alternative bank source, not filed |
| `surgical_read_patch` read directions / deletion-mask lanes | CONSTRUCTED rank-1 / mask overlays (closed-form or scored, never learned) | learned factorization | **Different provenance class** — constructed overlays stay the deployment lane; SNMF is analysis-tier |

## 4. Published prior art (§4)

The paper IS the published class-establishing citation (ACL 2026 main); its own lineage: NMF/CRAFT in vision (Lee & Seung 1999; Fel 2023), dictionary learning over contextualized embeddings (Yun 2021), the SAE line it benchmarks against. Nothing to novelty-check on the mechanism — our claim is lane-local fusion only, and this note makes no mechanism-novelty claim. The hierarchy result's closest published cousin inside our corpus is 693's F2 (facet accretion) — observational there, constructive here.

## 5. Fusion (the two-sided priority-ladder check)

**Priority #1 (self-evolve product stack): honest none.** The rulewright's text-rule corpus has no activation-census consumer; the decision family's gates are exact/stationary (stationary-surface law — SNMF is an offline one-shot artifact, no dynamics signal proposed anywhere). Recorded, not forced.

**Priority #3 (inference-perf league): the real consumer.** riir-infer 1255 — group-granularity candidates for the 1241 census with the input map for free, A/B'd against the best per-unit tier we ship (census_screen top-k) and the classical decomposition (ica_lens), not against random columns alone. Cost posture carried honestly: SNMF needs forward passes (the capture tier), so it is NOT a cheaper pre-screen than 1250's weights-only readout — its claim is GRANULARITY (groups + attribution), sitting between census_screen (per-unit) and the interventional sweep. Transfer caveat (GGUF-verified 2026-11-26): the fixture's 24 layers ALL carry MLP/FFN blocks (`blk.N.ffn_{gate,up,down}` at the header), so the lane's scope is the MLP blocks in every layer; the paper's statistics are dense-SwiGLU (Llama/Gemma/GPT-2), and whether the per-layer mixer type (attention vs GDN) shifts the activation statistics is measured by 1255's T0, never assumed.

## 6. What we do NOT claim

No mechanism novelty (published class; the paper is the citation). No Super-GOAT: Q1 fails (class published), the fusion value is lane-local. No game-side filing (the two-brain/latent rules bind game AI; this object is LLM MLP activations). No serving path — offline per-model artifact tier only, same posture line as 666/1250.

## 7. Fusion ideas recorded here (novelty TBD, no lanes beyond 1255)

1. **Group candidates ahead of the sweep** — filed as 1255 with bars (≥2× confirmation density at equal budget, bootstrap LB95 > 1.0× over ≥30 confirmations, one clear failure closes NEGATIVE).
2. **Hierarchy validator for 693's F2** — the parent-coverage metric run on OUR checkpoint would turn 693's fixed-basis facet-accretion observation into a measured structure; rides 1255 T2's cell, never a standalone claim.
3. **Neuron-space direction source for the 301 bank** — SNMF combos mapped by W_out·z as bank entries; only worth a cell if the bank's next consumer lane opens.
4. **Shared-core overlap as a census-side summary** — Z̄·Zᵀ bands as a per-layer artifact column (which units participate in how many features) — a cheap rider on 1255 T2's outputs.
