# Research 675: Conditional Geometry — Subset-Conditional Intrinsic Dimension + Information Imbalance

> **Source:** "The Changing Geometry of Grammar: Dimensionality and Neighborhood Reorganization across Transformer Layers" — Vallisa*, Ravenda*, Palominos, He, Raballo, Mira, Homan, Hinzen; arXiv:2608.25166, 2026-08-25, cs.CL, 11 pp. (0 citations per OpenAlex 2026-11-14; near-contemporary for the scientific claim: Domenichelli, Brunato & Dell'Orletta, CoNLL 2026, "Linguistic Profiling of Transformer Embedding Geometry")
> **Date:** 2026-11-15
> **Status:** Active — Plan 638 filed
> **Related Research:** 279 (spectral intrinsic dim + N≥d gate), 394 (within-class effective rank), 409 (TwoNN sketch), 449 (M1 mutual-kNN probe), 437 (knn_entropy source line)
> **Related Plans:** 301 (subspace_phase_gate — shipped), 415 (within_class_effective_rank — shipped), 426 (manifold erasure), **638 (this filing: cond_geometry)**
> **Classification:** Public

---

## TL;DR

The paper conditions two closed-form geometric measures on a token's grammatical class: (1) a **PoS-conditional intrinsic-dimension estimator** — the ABIDE/BIDE two-ball binomial model whose likelihood factorizes across points, so restricting the maximizer to an arbitrary labeled subset G is a legitimate conditional estimator with closed form `d̂_G = log(k̄_A/k̄_B)/log(τ)`; (2) a **token-level conditional Information Imbalance** — the rank of one space's nearest neighbor inside another space, normalized, per subset — which localizes *where* a token's neighborhood is reorganized between layers. Findings: function words expand earlier and collapse sooner than content words; ID expansions/contractions are explained by identifiable neighborhood reorganizations; encoders anchor II to the last layer, decoders to the first; per-token geometric trajectories alone classify grammatical role (F1 0.878 binary, 0.540 fine PoS — no lexical input).

**Distilled for katgpt-rs (modelless, inference-time):** the paper's value here is NOT grammar — it is two closed-form, label-conditional, spectral-free measures of *any* embedding population: "how many degrees of freedom does THIS labeled subset sit on" and "did THIS subset's neighborhood structure get rebuilt between two spaces/times". The stack owns set-level spectral ID (`subspace_phase_gate::estimate_intrinsic_dim` — participation ratio / numerical rank, Plan 301) and per-class spectral rank (`data_probe::geometry::within_class_effective_rank`, Plan 415); it owns NOTHING that (i) conditions an ID estimate on an arbitrary labeled subset in closed form without an eigensolver, and (ii) detects cross-space neighborhood reorganization. That pair is filed as Plan 638 (`cond_id` + `cond_imbalance`), gated on the existing `estimators` family umbrella.

---

## 1. Paper Core Findings

1. **PoS-conditional ID (cID), closed form.** Adapts ABIDE (Di Noia et al., arXiv:2405.15132, v5 2026-04-01): two concentric balls with radius ratio τ; under local density homogeneity, `k_A,i | k_B,i ~ Binomial(k_B,i, τ^d)` — the local density ρ_i cancels in the ratio, so each per-point factor depends only on local geometry. The likelihood **factorizes across points**, so restricting it to a subset G is a legitimate conditional likelihood whose maximizer is closed form:
   `d̂_G = log(k_A^G / k_B^G) / log(τ)`, with `k_A^G = (1/|G|) Σ_{i∈G} k_A,i`.
   **Neighbor counts include tokens of ANY class** — the binomial conditional holds regardless of the labels of points inside the balls; per-point neighborhood sizes are determined globally, and only the final average is restricted to the group. Groups with fewer than **10 tokens are discarded** (paper Appendix A.1, verified against the fetched PDF) to respect the estimator's asymptotic guarantees.
2. **PoS-conditional Information Imbalance (cII).** Glielmo et al. 2022 (PNAS Nexus; arXiv:2104.15079) adapted to token level: for layer pair A→B, take each token's nearest neighbor in A, record its rank in B, normalize by N/2, average over the subset `S_p`. Conditioning enters ONLY through the averaging set — neighbor search stays over the full population, so **cross-category reattachment remains visible** (a determiner's nearest neighbor shifting from other determiners to its head noun). Asymmetry carries the signal: Δ_p(A→B) ≈ 0 means A's neighborhoods predict B's. A crossover between Δ(ℓ→first) rising and Δ(ℓ→last) falling localizes the layer range where a category detaches from its input identity and converges to its contextual role.
3. **Role-dependent trajectories.** Closed-class items (DET/PRON/ADP) expand into higher-dimensional neighborhoods earlier and collapse sooner than open-class items, tracking dependency resolution; PROPN is an outlier (stable neighborhoods — rigid designators).
4. **Architecture dissociation.** Encoders couple cID to cII(ℓ→last); decoders to cII(ℓ→first) — bidirectional attention makes the dependency available from the input; causal attention establishes it only after the complement is processed. **Practical guidance for any checkpoint audit: cII-vs-last is the informative axis for encoders, cII-vs-first for decoders.**
5. **Geometry alone recovers role.** Logistic regression on per-token ID+II trajectories (point-wise MLE + point-wise II as features) separates function vs content words (F1 0.878) and recovers fine PoS (F1 0.540), far above shuffled controls (0.389/0.167) — with no lexical identity. Appendix B: function-word averaged representations converge toward content-word topic-predictivity exactly at reorganization layers ("semantic absorption"), and layer-mean cID correlates NEGATIVELY with classifier accuracy — the layers where a category's manifold is most compressed are where its role is most decodable.

## 2. Distillation

### 2.1 Published prior art (§4 searches — 2 parallel agents, 26 queries + 11 fetches)

| Item | Status | Bearing on the claim |
|---|---|---|
| BIDE/ABIDE closed form | PUBLISHED (arXiv:2405.15132 v5) | The estimator math is public — our claim is never "new estimator". ABIDE names heterogeneous-ID conditioning **future work**; the subset restriction is unpublished as such. |
| Information Imbalance | PUBLISHED (Glielmo 2022; DADApy Python/JAX-DII) | Measure + Python impl public. No Rust implementation found; no production/diagnostic deployment found; first transformer-layer application appears to be this paper. |
| Per-class / per-PoS ID analyses | PUBLISHED but all RECOMPUTE per subset with standard estimators — Eser et al. arXiv:2511.10475 (per-class ID, class imbalance); Domenichelli et al. CoNLL 2026 (per-PoS per-layer ID — the near-contemporary for the scientific finding; no closed-form conditional estimator, no II); Valeriani et al. arXiv:2302.00294 (global layer-wise ID + neighbor composition). | Differentiation targets, cited. |
| ID-as-diagnostic landscape | Research-crowded (TwoNN token-embedding training dynamics arXiv:2506.01034; text-complexity; AI-text detection), **deployed-open** — vendor drift stacks are centroid/distance/distribution-based; nobody ships ID-shift or neighborhood-reorganization as a live signal. | Application lane open. |
| Rust implementation | **No Rust ID/II crate exists.** kNN substrate candidates: kiddo v6.3 / linfa-nn (not needed at our scales — see 2.4). | First-of-kind artifact. |
| Estimator-validity caveat | L2N2 (arXiv:2603.10493): TwoNN/MLE/GRIDE systematically underestimate on spheres. | Gate material: synthetic gates MUST include a sphere arm. Survey anchor: arXiv:2509.15517 (8-estimator comparative). |

Agent-reported, coordinator-unverified: ABIDE journal pin, exact citation counts (OpenAlex 0-citation read was one oracle; S2 rate-limited). The under-10-token discard and the closed form were verified against the fetched PDF this session.

### 2.2 In-stack prior art — §3.6 signal-diffs on every coverage dismissal (code-verified this session)

1. **`subspace_phase_gate` (Plan 301, shipped)** — `participation_ratio` (L65), `numerical_rank` (L100), `estimate_intrinsic_dim` (L149), `phase_transition_gate` (L167). Consumes a SPECTRUM (eigenvalues of a fitted covariance); set-level; no labels; no subset conditioning. cID consumes RAW DISTANCES (neighbor counts), is label-conditional in closed form, and needs no eigensolver. Different estimator family → not coverage.
2. **`data_probe::geometry::within_class_effective_rank` (Plan 415, shipped)** — entropy rank of the within-class RESIDUAL COVARIANCE per class group; one set-level scalar per group; eigensolver-bound; no point-wise decomposition. cID's factorized per-point likelihood yields point-wise features (the paper's classification granularity). Different estimand + granularity → not coverage.
3. **`depth_invariance` (katgpt-types)** — per-timestep spectral flatness + `effective_rank_slope`, `Collapsed` detection on recursive chains: the trajectory-of-geometry-across-depth SHAPE exists, class-agnostic, single-chain, spectral. Partial cousin → not coverage.
4. **"Behavior rank preservation" (Plans 309/310 G2 gates)** — ACTION-preference ranks under state perturbation. Different signal class from embedding-neighborhood ranks; all workspace "rank preservation" grep hits are this → not coverage.
5. **TwoNN (R409 sketch; zero code hits)** — per-point local ID as a tangent-rank floor; not conditional; not shipped → not coverage (and demoted to secondary reference in Plan 638's comparisons, since there is no shipped TwoNN to beat).
6. **R449 M1 mutual-kNN alignment** — paper-analysis probe, not shipped; binary overlap vs II's normalized rank prediction → not coverage.
7. **`nuisance_fraction` (Issue 978)** — corrected BETWEEN-group variance fraction ρ; explicitly documented as "Neighbor, not duplicate" of within_class_effective_rank. Different axis (group separation vs subset manifold dimensionality) → not coverage; cited as family context.
8. **`knn_entropy` (Issue 708, data_probe::entropy)** — the STRUCTURAL SIBLING: brute-force O(n²·d) kNN over point-latent populations, offline audit cadence, bounded scratch, f64 accumulators, zero steady-state allocs, duplicates ⇒ collapse signal. Produces nats, not dimension ranks, and no II. Adopted as the implementation template (scratch discipline, gate shapes, opt-in posture).

**Reverse-grep:** the consumers below fill DOCUMENTED gaps — riir-refine Issue 222 batch_diversity names axis (a) "a richer rule embedding" as the pending unblock; no II analog exists anywhere (grep: `information imbalance|info_imbalance|neighborhood rank` → zero code hits).

### 2.3 Reframes (both mandatory)

- **Game context:** II across TICKS on per-NPC belief populations = a **social reorganization detector**: when a zone's belief-embedding kNN structure rebuilds (kin→foe reattachment, party regrouping), cII spikes before individual behavior changes. Natural consumer: gate KG-triple emission cadence (riir-ai `vibe.rs` emits latent-similarity triples) on reorganization events — "cadence follows observation". cID per archetype group = crowd-state complexity (how many independent factors condition this crowd's state). FUSION IDEA ONLY — unmeasured; the riir-ai issue filing is DEFERRED (repo [ahead 1, behind 75] + dirty lockfile this session). **Re-open trigger: file the riir-ai issue when riir-ai develop is synced and clean.**
- **Consumer context (priority-#1 self-evolve stack):**
  (a) **riir-refine corpus health probe** — fit a linear probe on geometry-only features (per-rule embedding statistics over the frozen embed) vs domain labels: if geometry alone recovers the domain, the embedding is category-structured (the R394 "geometric features recover class" result transplanted as an EMBEDDING-QUALITY AUDIT). Actionable now on `assets/rule_embed_frozen_v1.bin`. Filed as riir-refine Issue 239 axis (i) — **cond-ID only** (single space; II has no consumer until a second space exists).
  (b) **batch_diversity rider** — Issue 222 axis (a): per-batch ID ratio (batch cID vs corpus cID) as a diversity diagnostic that does not reduce to spread ratios. Honest caveat: ID inherits embedding saturation — it is a richer FUNCTIONAL of the future richer embedding, not a fix for its absence. Filed as Issue 239 axis (ii).
  (c) **Checkpoint geometry audits** — an offline `geometry_audit` lane in riir-infer (the `weight_audit`/`sink_audit` shape: opt-in, report-only, zero serving cost), reading the encoder/decoder II-anchoring guidance (§1 finding 4). Consumer-side, not filed this session.
  (d) The reflex MODELLESS lane has no layers — the paper's layer-wise mechanism has no consumer there; honest scoping: consumers are checkpoint-side and corpus-side only.

**Stationary-surface check (§1 step 4a):** batch corpus mutates per batch (dynamics valid); KG ledger mutates per tick (dynamics valid); checkpoints are static → cID/cII profiles are LEVEL readouts on a fixed artifact — permitted, never wired as dynamics monitors.

### 2.4 Fusion

**The filing:** `cond_id` (closed-form BIDE restriction over labeled subsets, neighbor counts over the full population) + `cond_imbalance` (cross-space neighbor-rank prediction, subset-averaged, first-index tie rule) as ONE kit under the `estimators` umbrella — Plan 638.

**Fusion candidates this enables (documented, not measured):** cID-ratio batch diversity × Issue 222 · II-gated KG emission × riir-ai vibe · II(ℓ→first)-anchored decoder audits × riir-infer geometry_audit · cID as the sample-sufficiency `d` for the SHIPPED `phase_transition_gate(n, d)` (Plan 301's gate currently takes an externally-estimated d — cID provides the label-conditional, spectra-free estimator).

**kNN substrate decision:** brute-force distance counts over reusable scratch (the `knn_entropy` discipline) — our populations are document/sentence scale (n ≤ a few thousand; the paper's own posture is ~1,218 tokens/document), O(n²·d) is the shipped precedent for offline cadence. kiddo/linfa-nn recorded as the upgrade path if populations grow; no new deps (katgpt-core leaf law).

## 3. Verdict

**Tiers:** GOAT on the estimator pair (track a+b). Track (c): nothing to file — the paper trains nothing; closed-form analysis of frozen checkpoints. Adversarial panel: SKIPPED — no training-adjacent framing anywhere (analogous to the skill's inference-side skip clause).

**Novelty gate (Q1–Q4):**
- Q1 prior art: in-stack — none computes conditional closed-form ID or any II (8 signal-diffs above). Published — technique halves published; conditional restriction unpublished; applied runtime-diagnostic lane open. YES (scoped as in-stack artifact).
- Q2 new behavior class: subset-conditional no-spectra geometry + cross-space reorganization detection = capabilities no incumbent has. YES-for-stack.
- Q3 selling point: "our systems measure how many independent factors condition any labeled subset of embeddings in closed form, and detect when a population's neighborhood structure reorganizes between two spaces or ticks — no training, no spectra." Finishable per domain. YES.
- Q4 force multiplier: PROPOSED (≥2 pillar connections listed in 2.4) but UNMEASURED. This is the honest bar → **GOAT, not Super-GOAT** (R394 precedent: analogous fusion graded GOAT for exactly this shape).

**MOAT gate (§1.6):** katgpt-rs row — generic latent-space math, no game/chain/shard semantics → open primitive here; consumers route per domain (2.3). Feature discipline: **no new feature key** — feature_cap_gate measures katgpt-core at 290/300 (10 headroom); per the Plan 635 growth guard and verdict round-2 amendment, the module gates on the existing `estimators` family umbrella (pins grant row exists).

**Weakest point (named):** marginal value over the shipped spectral incumbents on OUR data is unproven — our embeddings are small-dim (8-D hashed spans, 64-dim style_weights) where spectra are cheap. Plan 638 carries the kill-switch: incumbent comparison (`within_class_effective_rank` + `subspace_phase_gate::estimate_intrinsic_dim`) on real data shapes; no advantage → demote to Gain (diagnostic reference implementation). Degenerate inputs are first-class: minimum subset-support floor with loud refusal (paper's <10-token discard cited), pinned tie + zero-distance rule (determinism law), and an explicit statement that NO interval/coverage is claimed — nothing UQ-bearing is emitted, so the conformal-naive floor rule does not apply. `cond_imbalance` is a LOCAL-ONLY instrument that never crosses the sync boundary; its tie rule serves reproducibility, not quorum.

## 4. What is NOT distilled

- The linguistics (PoS typology, PROPN rigidity, head directionality) — domain science, not stack primitive.
- The downstream-topic-classification experiment design (Claude-labeled PubMed specialties) — an analysis pattern, recorded as inspiration only for consumer (a).
- Anything training-shaped: the paper has no training loop. No riir-train deferral exists to document (no Path 0.5 trigger).
- Multi-model comparison infrastructure — our audits are per-checkpoint; the paper's cross-family claims inform §1 finding 4's guidance only.
