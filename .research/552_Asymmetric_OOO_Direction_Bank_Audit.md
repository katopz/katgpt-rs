# Research 552: Asymmetric OOO + Cross-OOO — Direction-Bank Audit & Curation Gate

> **Source:** Issa, Liu, Ballé, Klindt — "High-dimensional population codes reveal interpretable and diverse features underlying visual perception" [bioRxiv 2026.09.05.748439](https://www.biorxiv.org/content/10.64898/2026.09.05.748439v1) (CSHL/NYU, posted 2026-09-11)
> **Date:** 2026-09-12
> **Status:** SUPER-GOAT (upgraded 2026-09-12 — the PoC pass condition of §3 was met and the mandatory outputs filed: open primitive `ooo_audit` shipped, GOAT gate Bench 758 G1+G2+G4 ALL PASS, this note's addendum records the measurement, the riir-ai guide landed as Research 378, the riir-clippy axis filed as an issue). Honest scope: what the PoC PROVED is codebank/bank-level redundancy detection + curation on shipped substrate (F3) and the metric-granularity law; the crowd-scale NPC premise (F1) remains a projection until the riir-ai consumer lands — the guide is the plan for that. Primitive stays OPT-IN per the no-default-consumer rule.
> **Related Research:** 397 (MAG — closest cousin, the acquisition half), 144 (Functional Emotions — direction bank), 276/321 (direction consumers PWC/CFB), 409 (MANCE — projection-ablation lineage), 020 (TurboQuant — VQ cousin), 078 (MTP Cluster), 453 (Variable-Rank Domain Expert centroids), 467 (RRQ), 527 (TPR binding), 390 (Expand Neurons — superposition), 475 (ICA Lens — direction discovery cousin)
> **Cross-ref (riir-ai):** 316 (MAG game-runtime guide — this is its missing curation stage)
> **Classification:** Public (metrics + greedy curation are generic math); game wiring → riir-ai, healer wiring → riir-clippy, both private.

---

## TL;DR

The paper audits **representation banks** (neurons, PCA axes, SAE latents, K-means template codes) with two automated, human-perception-aligned metrics and one greedy curation loop — all modelless, all offline, all matrix reductions:

1. **Asymmetric Odd-One-Out (OOO) interpretability score** — per unit: take its K maximally-activating exemplars (MEIs), threshold = mean pairwise similarity among those MEIs; score = fraction of all other exemplars whose mean similarity to the MEIs falls *below* that threshold. Asymmetric by design (fair to sparse/non-negative activations that a symmetric 2-AFC metric penalizes).
2. **Cross-OOO diversity score** — per unit *pair*: fraction of the 2K MEIs correctly attributed to their unit of origin (more similar to own unit's other MEIs than to the other unit's MEIs). Measures **redundancy across the bank**, not within a unit.
3. **Greedy curation** — sort units by OOO descending, drop OOO < 0.8, greedily append a unit only if its Cross-OOO vs *every* kept unit ≥ 0.8. List length = the bank's count of **unique interpretable features**.

Headline results: K-means distance-to-centroid codes ("unsupervised template matching", UTM) beat neurons, PCA, *and* SAEs on both metrics across macaque V4/IT and ResNet-50/CLIP/DINOv2; UTM recovers up to **8× more unique features than neurons** (superposition — non-orthogonal codes), the advantage grows with unit count and *stimulus diversity* (categories, not exemplar count), and projection-ablation of a UTM code selectively removes exactly its concept from downstream behavior while neuron ablation does nothing.

**Why it matters here:** we ship a growing **direction-vector ecosystem** — MAG mines directions unsupervised (R397/Plan 418, default-on), EmotionDirections (P162) extracts them supervised, PWC/CFB/LFS/Spherical *inject* them (Σ sigmoid(w_i/τ)·d_i), freeze/thaw *commits* them. **Nothing audits or curates the bank.** Two NPCs mining near-identical "avoid faction X" directions, or one NPC mining the same concept from five quests, are invisible today — and because blends **sum** sigmoid-weighted directions, duplicated concepts **double-count** (an effective-weight inflation bug class, not just waste). This paper supplies the missing stage: *acquire (MAG) → inject (LFS/PWC) → **audit/curate (this)** → commit (freeze/thaw)*.

## 1. Paper Core (what was measured)

- **Data:** macaque IT (Vinken et al. 2023, 449 units × 1,379 images; Majaj et al. 2015, 168 IT + 88 V4 × 3,200), AM face patch (159 × 2,100), plus ResNet-50 L3/L4, CLIP ViT-B/16, DINOv2 ViT-S/14 on 10k CIFAR-100.
- **Extraction:** N×I activation matrix → Z×I population matrix via (a) PCA, (b) SAE (L1+MSE, λ sweep, 1× expansion), (c) UTM = K-means over image-response columns; each cluster center is a "template"; row z of the population matrix = **negative Euclidean distance** of every image response to template z.
- **OOO (Eq. 6–8):** threshold from MEI self-similarity, intruder-fraction scoring, DreamSim as the human-aligned similarity metric (Wasserstein Distortion as the local/receptive-field-aware alternative).
- **Cross-OOO (Eq. 9–11):** symmetric U×U matrix; 0.5 = chance = indistinguishable preferences.
- **Null control:** duplicated-best-neuron + matched Gaussian noise → UTM gains vanish ⇒ genuine population synergy, not a K-means artifact.
- **Ablation:** `x ← x − (x·ĉ)ĉ` on UTM codes selectively degrades exactly the relevant class (sunflower → confused with rose/poppy); neuron ablation ≈ no effect. Established lineage (INLP → RLACE → LEACE → directional ablation) — the paper's contribution is applying it to *discovered* codes as causal validation.

## 2. Distillation (modelless primitives for this stack)

Everything below is closed-form matrix reduction over an activation matrix `A ∈ R^{U×N}` plus a **pluggable** exemplar-similarity matrix `S ∈ R^{N×N}` (DreamSim in the paper; for us: latent cosine on exemplar embeddings, span embeddings from `ModellessEmbedder`, or any kernel). No training, no gradients; K-means substrate already ships (`fit_codebook_kmeans_into` — deterministic Lloyd's + k-means++).

- `ooo_score(unit)` — asymmetric intruder fraction. O(K·(N−K)) per unit, reducible to two matvecs against S.
- `cross_ooo(a, b)` — attribution fraction over 2K MEIs; fills a symmetric U×U matrix.
- `greedy_curate(units, ooo_cut = 0.8, cross_cut = 0.8)` → curated index list + **unique-feature count** (a bank-quality scalar).
- **Unique-feature count as an elbow detector**: the paper's Fig. 4 scaling (features vs Z ∈ {N, 2N, 4N, 8N}) is a modelless saturation curve — pick codebook K where the curated count saturates. Direct K-selection signal for `EffectCodebook` (factorized_action), `cluster_map` (katgpt-forward), shard VQ (katgpt-kv).
- **Projection-ablation validation** — already our pattern (R409 MANCE lineage); consume, don't rebuild.

## 3. Novelty gate (§1.5, scored honestly)

- **Q1 no prior art:** PASS. `odd_one_out|OddOneOut` grep = **0 hits** workspace-wide; no diversity/redundancy bank audit ships (only selection-time penalties — alien_sampler local-redundancy bench — and bench-time non-redundancy GOAT checks). Published prior art: components are derivative (word intrusion — Chang et al. NeurIPS 2009; symmetric MEI metric — Klindt et al. 2023 arXiv:2310.11431; MIS NeurIPS 2024; K-means codes — Coates & Ng 2012; ablation — INLP/LEACE), but the **asymmetric OOO variant** and especially **Cross-OOO + greedy curation of a bank** have no close published ancestor (verified by prior-art sweep, 2026-09-12). Signal-diff vs closest shipped cousin **MAG (R397)**: MAG *consumes verdict-conditioned mean activation shifts* to **acquire** directions (diagnostic: linearity ϵ_Q, LOO separability — steerability signals); OOO/Cross-OOO *consume top-K activating exemplars + an external similarity matrix* to **audit any existing bank**. Different signal, different pipeline stage, one read of `mag_mining`'s kernel suffices — no overlap.
- **Q2 new behavior class:** PARTIAL. New *capability class* (self-auditing, self-curating direction banks — no incumbent here or in the paper's lineage has it), but it **hardens existing pillars rather than adding a new one**, and whether it is load-bearing (vs. hygiene) rests on an unmeasured premise: that runtime-mined MAG banks actually accumulate redundancy at crowd scale. That measurement is the PoC's job (Issue 759 T3). Not confident ⇒ no Super-GOAT this pass, no "candidate" hedging either — the claim is explicitly deferred to the PoC.
- **Q3 product selling point:** conditionally YES — *"every direction in an NPC's frozen personality/emotion bank provably earns its slot: self-consistent (OOO) and non-redundant (Cross-OOO), audited modellessly at freeze time."*
- **Q4 force multiplier:** YES — MAG banks (R397/316), EmotionDirections (P162), CFB archetype banks (P321), `EffectCodebook`, `cluster_map`, shard-VQ centroids, healer rule corpus (§F2), neuron-db `style_weights[64]` at freeze. ≥2 pillars trivially.

**Verdict (superseded 2026-09-12 — see Status + PoC Addendum): Gain — GOAT-tier fusion, deferred Super-GOAT.** The deferral condition has since been MET: the PoC measured real redundancy on shipped substrate and curation recovered it exactly. Files: this note + `.issues/759` (closed, resolution recorded) + Bench 758 + the `ooo_audit` primitive + riir-ai guide (Research 378) + the riir-clippy axis issue.

## 4. Fusion (what none of the cousins alone can do)

- **F1 — Self-curating NPC direction banks (riir-ai):** MAG × OOO/Cross-OOO × freeze/thaw. MAG mines candidate directions from verdict-labeled shifts; the audit gate runs **before** `MerkleFrozenEnvelope` commit — freeze only what earns its slot. Crowd correctness angle: PWC blends sum sigmoid-gated directions, so duplicated concepts inflate effective weight and drag personality toward duplicated concepts; Cross-OOO dedup is the guard. Natural cadence: consolidation/sleep-cycle windows (cross-ref R116 sleep-time compute, R317 feeling-brain consolidation — biologically apt: sleep consolidates and prunes).
- **F2 — Healer rule-corpus dedup (riir-clippy, fusion priority #2):** rules = units; each rule's top-K oracle-triggering spans = MEIs; span embeddings (riir-rag `ModellessEmbedder`, already shipped) = S; Cross-OOO = the **absorption/splitting detector** the corpus has never had (SAE-latent redundancy's exact analog in rule space); greedy curation = corpus dedup instrument; feeds `frontier_report` as a new per-rule axis beside r/f/n — a rule whose top spans are indistinguishable from another rule's is a merge candidate, measurable today.
- **F3 — Codebook K-selection (katgpt-rs):** unique-feature count (§2) as saturation curve for `fit_codebook_kmeans_into` K, `cluster_map` K, shard-VQ group count — replaces hand-pinned K with the bank's measured distinct-content capacity.

## 5. Latent vs raw boundary

Audit is **offline and latent-only**: activation matrices + similarity matrices, both local. Outputs are scalar scores + an index list; the curated bank commits through the existing freeze envelope (BLAKE3, atomic swap). Nothing new crosses sync; zero game-tick cost; the 5 synced affect scalars are untouched. Healer-side: audit runs on fixture spans at bench time, never in the heal path.

## 6. Validation protocol (Issue 759)

- **G1 correctness (synthetic):** planted ground-truth bank — distinct directions + planted near-duplicates + noise units; audit must recover the planted unique count ±1, prune duplicates, drop noise; curated subset preserves OOO ranking.
- **G2 perf:** U×U Cross-OOO + U×N reductions, zero-alloc, chunked SIMD; µs–ms budget per audit run.
- **G3 no-regression:** feature flag `direction_bank_audit`, off by default; consumers unchanged.
- **G4:** alloc-free reductions (offline gate, same discipline).
- **Defend-wrong PoC (§3.6, the kill condition):** run the audit on a *real* bank (Plan 418 MAG fixtures or a live `EffectCodebook`). If measured redundancy is below the metric's noise floor, the gate is hygiene-only → demote to opt-in, record the negative result here. UQ floor rule: N/A (no distributional claim).

## PoC Addendum (2026-09-12 — the §3.6 measurement, Bench 758)

The defend-wrong PoC ran on REAL substrate (shipped deterministic k-means,
`factorized_action::fit_codebook_kmeans_into`) reading an over-complete
K=12 codebook over 4 planted clusters as a UTM bank (activation =
−Euclidean distance, the paper's construction):

- **Redundancy measured + recovered:** under concept-level similarity (the
  DreamSim role), the 12-centroid bank curates to **exactly 4** distinct
  features; exact-fit K=4 also yields 4 (elbow anchor). Axes confirmed:
  architectural (grep + shipped substrate), latency (G2 1120 µs full cycle at
  U=64/N=1024), quality (G1 exact planted-count recovery across 3 seeds).
- **The metric-granularity law (novel measured finding):** under geometric
  RBF (γ=1) the SAME bank keeps ~11 — Cross-OOO detects redundancy at the
  similarity metric's CONCEPT GRANULARITY; the metric defines the concept.
  Consumers must pick a semantic-granularity metric (span embeddings, latent
  cosine) or the audit honestly reports micro-features. Pinned as a test.
- **G4 caught a real bug pre-ship:** `reserve_for`'s len-blind `Vec::reserve`
  doubled each scratch buffer exactly once (2 steady-state allocs); fixed,
  gate now 0/100 cycles.
- Primitive GOAT: G1+G2+G4 PASS; OPT-IN per the no-default-consumer rule.
- **First consumer LANDED (riir-clippy `248be9a5`, 2026-09-12):** the F2 healer axis (Issue 098, Bench 086) — measured PASS on the 100-span real oracle-labeled clippy pool (TF-IDF cosine over op-bigram tokens; the granularity law governed the metric choice); 3 absorption candidates incl. the calibration anchor (`single_match↔single_match_else`, clippy's own split, detected unprompted); `unique_feature_count = 12/17`. Status: DISTILLED — F2 consumed; F1 (game MAG bank) remains with the cadence work.

## 7. Priority

P2 — small offline primitive; does not compete with the 4090 prefill lane. Healer axis (F2) is the cheapest first consumer (fixtures + embeddings already exist); game wiring (F1) lands with the MAG bank cadence work.
