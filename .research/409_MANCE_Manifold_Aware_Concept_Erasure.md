# Research 409: MANCE — Manifold-Aware Concept Erasure

> **Source:** [MANCE: Manifold Aware Concept Erasure](https://arxiv.org/abs/2607.03973) — Avitan, Goldberg, Elazar (Bar-Ilan University / AI2), Jul 2026. Code: [github.com/MatanAvitan/mance](https://github.com/MatanAvitan/mance)
> **Date:** 2026-07-11
> **Status:** Active — GOAT verdict, plan filed
> **Related Research:** 408 (TILR — alignment-gated subspace correction, the closest cousin), 406 (Spectral Rewiring — weight-delta SVD projection), 393 (Block-Sparse Featurizer → Plan 412 Subspace Steering), 294 (Viable Manifold Graph — safe-manifold navigation), 290 (Latent Field Steering — 1D direction injection), 310 (RIZZ — non-interference branches), 397 (MAG — unsupervised direction mining, the probe replacement)
> **Related Plans:** 425 (TILR — alignment-gated subspace correction), 423 (spectral_rewire — weight-delta purification), 412 (subspace_steering — k-dim block), 329 (non_interference_branches — orthogonal direction allocation), 418 (MAG — direction mining), 309 (latent_field_steering — 1D), 426 (this primitive's plan)
> **Classification:** Public
> **PASS-Redirects (synthesis):** Gong et al. [arXiv:2607.22531 "Twins: Learn to Predict Unified Representations with Focal Loss"] — PASS. The paper's second diagnostic (Two-NN intrinsic dimensionality: SigLIP ID≈15 despite D=768, VAE ID≈35 despite D=128 — the "dimensionality paradox" where the semantic manifold is more compressed than the pixel manifold) is exactly the Two-NN local intrinsic-dim estimation MANCE already consumes (Step 1, k-NN + local PCA + TwoNN floor). The Twins representation + Focal Loss are image-diffusion training infrastructure we do not ship. Diagnostic stage → covered here + Research 394 (within-class effective rank, the conditional-alignment axis) + Research 279 (phase-transition N≥d gate); training stage → riir-train (image diffusion, out of scope).
> **PASS-Redirects (synthesis):** Fu et al. [arXiv:2608.03913 "Sparse Weight Decomposition for Efficient Circuit Extraction"] — PASS. SWD ablates *circuits* (sparse addressable units in weight matrices); MANCE erases *concepts* (directions in latent state). Different layers (weights vs activations), different mechanisms (sparse factorization vs tangent-projected erasure). SWD's circuit ablation is the mech-interp analog of MANCE's concept erasure, but operates on a layer (full weight matrices) our runtime doesn't expose for ablation (NPC personality lives in `style_weights[64]` latent vectors + direction projections, not weight-matrix circuits). Sparse two-factor decomposition mechanism → covered by R231 (`SparseTaskVector`, rank-`nnz` factorization for deltas); mech-interp circuit extraction → R444 (IMIR, stronger closed-form). Not our commercial domain.
> **PASS-Redirects (synthesis):** Quillan-Ronin [github.com/leeex1/Quillan-Ronin — "Building Foundation Models via Weight-Space Programming (WSP): Analytical Alignment, Multi-Donor Cognitive Grafting, and Closed-Form Refusal Abliteration without Backpropagation" (no arXiv; HF CrashOverrideX/Quillan-Ronin; Nov 2026)] — PASS. Its §3 refusal ablation (contrastive centroid → W − α(vvᵀ)W, α∈[0.8,1.0]) is the published abliteration weight-orthogonalization line (Arditi et al. arXiv:2406.11717; Labonne 2024) — the refusal-direction member of exactly the erasure family this note maps; the stack serves no RLHF chat surface to ablate (refusal handling here is modelless abstention), and published evidence that abliteration damages capability (promptfoo "Abliteration Cripples Math", 2026-03) contradicts the paper's unmeasured "zero degradation" claim (its §7 has no quality benchmark). Re-open trigger: if an RLHF chat model is ever SERVED by the stack (e.g., a future qwen3.8 chat endpoint), re-evaluate abliteration as a deterministic rank-1 `LoraPair` overlay behind a capability-retention gate (the lossy-surface per-family rule) — until then there is no consumer.

---

## TL;DR

MANCE proposes the **Manifold Constraint Hypothesis (MCH)**: natural representations concentrate on a lower-dimensional manifold, and interventions constrained to that manifold preserve other encoded concepts better than unconstrained edits of the same magnitude. MANCE operationalizes MCH for concept erasure by (1) estimating a **local tangent basis** per-sample via k-NN + local PCA on **natural (unedited) representations** X⁽⁰⁾, (2) projecting the erasure gradient onto that basis with **spectral weighting** (σ^α, α=1), and (3) bounding the step by a **per-sample local-radius trust region** (ε·r_i, where r_i is the mean distance to k natural neighbors). MANCE++ adds LEACE (1st-moment) + CovMatch (2nd-moment rank-2) closed-form preprocessing. State-of-the-art on 119 settings (13 LLMs × 3 NLP concepts + 40 CelebA-CLIP attributes), outperforming Obliviator on surgicality-leakage tradeoff.

**Distilled for katgpt-rs (modelless, inference-time):**

The MLP probe training is the "LLM-as-implementation" pattern (R368) — it computes the *decision* "which direction to erase"; our substrate computes the same decision modellessly via MAG (Plan 418) or CNA (Plan 087) or HLA `EmotionDirections`. The **mechanism** — local tangent projection + spectral weighting + per-sample trust region — is pure linear algebra (k-NN, local SVD, weighted projection, bounded step). The novel combination not yet shipped:

| MANCE component | Closest shipped cousin | Gap |
|---|---|---|
| Local tangent basis (per-sample k-NN + local PCA on natural X⁽⁰⁾) | `spectral_rewire` (Plan 423, global SVD of W₀), `tilr` (Plan 425, global invariant subspace U_r) | **GAP** — both use a GLOBAL pre-computed basis; MANCE estimates a LOCAL per-sample tangent from natural neighbors, re-estimated every round as the edit moves |
| Spectral weighting (σ^α reweighting of tangent coordinates) | — | **NOT SHIPPED** — no primitive reweights projection coordinates by local singular values |
| Per-sample local-radius trust region (ε·r_i, dimensionless, transfers across settings) | `trust_region_spec` (Plan 182, speculative decode trust region — different domain) | **GAP** — trust region exists for spec decode, not for latent edits; the per-sample ε·r_i cap is not shipped |
| Natural-manifold anchoring (always estimate from X⁽⁰⁾, not edited state) | — | **NOT SHIPPED** — the "moving query, static manifold" pattern (edits move the point, but the tangent is always estimated from natural geometry) is genuinely novel |

---

## 1. Paper Core Findings

### 1.1 The Manifold Constraint Hypothesis (MCH)

Natural representations produced by a model on natural inputs do not fill the full representation space uniformly — they concentrate on a structured, lower-dimensional manifold M ⊂ ℝ^d. MCH predicts that among interventions with matched effect on the target concept, **manifold-constrained interventions preserve other concepts better than unconstrained interventions**. The paper validates this empirically: the unconstrained ablation (AmbCE++, same effective step magnitude, no tangent projection) leaves 6–10pp leakage vs MANCE++'s 0–1.6pp.

### 1.2 The MANCE algorithm (3 ingredients)

**Step 1: Estimate the manifold locally.** For each representation x_i, find k nearest neighbors N_k(x_i) among the **natural (unedited)** representations X⁽⁰⁾. Mean-center the neighbors, form local PCA matrix S_i, compute SVD: `SVD(S_i) = L_i diag(σ_i,1,...,σ_i,k) V_i^T`. Keep top-r right singular vectors as tangent basis: `B_i = [v_i^(1), ..., v_i^(r)]`. The rank r is estimated once via TwoNN intrinsic dimension (floor k_min=8).

**Key design choice:** the neighborhood is always drawn from the fixed natural X⁽⁰⁾, but the tangent is estimated at the sample's **current (edited)** position. As a point is edited, it queries a new neighborhood of natural representations — the tangent reflects how natural representations vary near the edited location, not how already-edited points vary.

**Step 2: Build the tangent erasure direction.** Normalize the probe gradient: `u_i = ∇f(x_i) / ||∇f(x_i)||`. Project onto tangent basis: `c_i = B_i^T u_i`. Spectrally weight: `d_i = B_i diag(σ_i^α) c_i` (α=1 default — high-σ axes get more step mass, thin axes get less). Normalize: `û_i = d_i / ||d_i||`. Apply erasure: `x̃_i = x_i - λ_i · <x_i, û_i> · û_i`.

**Step 3: Per-sample local-radius cap.** The step size λ_i is chosen so `||x̃_i - x_i|| ≤ ε · r_i`, where `r_i = (1/k) Σ_{j∈N_k(x_i)} ||x_j - x_i||` (mean distance to k natural neighbors). Closed-form: `λ_i = min(λ_max, ε·r_i / <x_i, û_i>)`. The key insight: **ε is dimensionless** (ratio of displacement to local neighborhood radius), so ε=0.1 transfers across all 119 settings without per-setting tuning — the local r_i absorbs the panel's representation scale.

### 1.3 MANCE+ and MANCE++

- **MANCE+** = LEACE (rank-1 first-moment linear erasure) prepended before the MANCE loop.
- **MANCE++** = LEACE + CovMatch (rank-2 second-moment covariance asymmetry erasure, the top-2 eigenvectors of ΔΣ = Σ₊ - Σ₋) prepended. Total closed-form preprocessing removes effective rank ≤3, negligible vs d ∈ [768, 5376].

### 1.4 Results (119 settings: 39 NLP + 80 CelebA-CLIP)

- MANCE improves every prior erasure method (INLP, LEACE, IGBP) when applied on top: e.g. LEACE leakage drops from 19.1→1.5pp at ΔY≤1pp.
- MANCE++ is SOTA on nonlinear erasure: 0.0–1.6pp leakage at all budgets, reaching chance on 19–35/39 NLP settings vs Obliviator's 13–17/39.
- The gain is largest where erasure is hardest (gender, where profession strongly correlates): MANCE++ reaches chance on 12/13 models at ΔY≤5pp vs 0/13 for Obliviator.
- **The unconstrained ablation (AmbCE++) is the key control**: same effective step magnitude (λ=29.31, the mean per-sample λ_i from MANCE++), no tangent projection. It leaves 6–10pp leakage — proving the gain comes from the manifold constraint, not the probe loop or preprocessing.

### 1.5 Latency

MANCE takes ~8 min/panel (NVIDIA B200, 458–475s across variants). ~50% runtime is per-round local SVDs, ~40% CPU-GPU transfers. A fully GPU-resident implementation would reduce this severalfold. This is a one-time offline fit-and-apply cost; inference uses the edited representations directly.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | ≥2 codebase equivalents |
|---|---|
| concept erasure / remove concept | **direction removal**, **orthogonal projection** (`orthogonal_projection_into` in riir-poc), **subspace projection** (`spectral_rewire`, `tilr`), **latent steering** (Plan 309 — the inverse, injection) |
| manifold constraint / tangent space / local PCA | **tangent basis**, **local SVD** (`thin_svd_into`, Plan 301), **neighborhood PCA** (NOT shipped — SLoD uses tangent space for Fréchet mean, not for editing), **manifold** (`viable_manifold_graph`, Plan 312 — discrete navigation, not tangent projection) |
| spectral weighting (σ^α) | **spectral weighting**, **eigenvalue weighting** (NOT shipped as a step-mass reweighting; `spectral_hierarchy` uses eigenvalues for diagnostics, not for reweighting projection coordinates) |
| local-radius trust region (ε·r_i) | **trust region** (`trust_region_spec`, Plan 182 — speculative decode, different domain), **per-sample step** (NOT shipped for latent edits) |
| probe / classifier (finds the concept direction) | **direction vector** (HLA `EmotionDirections`), **MAG** `mine_contrast_direction` (Plan 418), **CNA** contrastive pair (Plan 087), **ConstraintPruner** |
| surgicality / preserve other concepts | **non-interference** (Plan 329 — orthogonal branch allocation), **no-harm guarantee** (TILR Plan 425 — γ→0 bit-recovers input), **branch-local** |
| natural representations X⁽⁰⁾ | **natural manifold**, **reference representations**, **baseline state** (freeze/thaw pre-edit snapshot, shard pre-consolidation state) |
| iterative probe refit | **re-estimation** (`riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` — coherence-driven re-estimation), **consolidation cycle** (`src/sleep/consolidation.rs`) |
| LEACE (1st-moment erasure) | **mean removal**, **centering** (shipped as a preprocessing step in many primitives), **orthogonal projection** onto mean direction |
| CovMatch (2nd-moment, rank-2) | **covariance asymmetry projection** (NOT shipped; `dual_gram_pca` ships PCA but not class-conditional ΔΣ eigenvector projection) |

### 2.2 Fusion grep results (both layers, all 5 repos, both vocabularies)

Paper-vocabulary grep (`tangent|local_pca|local_svd|manifold_constrain|concept_eras|trust_region|spectral_weight`) returned **ZERO** codebase hits for the MANCE-specific combination (local tangent + spectral weighting + trust region for latent edits). The codebase-vocabulary grep surfaced **10 close cousins**:

| Cousin | Mechanism | Overlap with MANCE | Difference |
|---|---|---|---|
| **Plan 425 TILR** (R408) | Alignment-gated subspace projection: `s' = s + η·γ·Πd` | Subspace-projected correction + no-harm guarantee (γ→0) | **GLOBAL** invariant subspace U_r (from contrastive SVD); MANCE uses **LOCAL** per-sample tangent. TILR gates by alignment fraction γ; MANCE gates by local radius ε·r_i. TILR is INJECTION; MANCE is ERASURE (subtracts direction). |
| **Plan 423 spectral_rewire** (R406) | SVD-project weight delta ΔW onto base U_r V_r^T | On-manifold/off-manifold decomposition + `on_manifold_fraction` | Operates on **WEIGHT deltas** (offline); MANCE on **latent state** (runtime). **GLOBAL** SVD of W₀; MANCE uses **LOCAL** k-NN PCA. No spectral weighting, no trust region. |
| **Plan 412 subspace_steering** (R393) | k-dim block steering `s + Σ α_j u_j` | Subspace-projected correction on latent state | `α_j` are **fixed per-axis strengths**, NOT adaptive to local geometry. No spectral weighting, no trust region. INJECTION, not ERASURE. |
| **Plan 416 region_subspace** (R396) | MFA region-conditioned subspaces | Per-region subspace projection | Sigmoid membership gates g_k(x), not local-radius trust region. Region-conditioned (K regions), not per-sample local. |
| **Plan 329 non_interference_branches** (R310) | Orthogonal direction allocation for branches | Non-interference guarantee (branches don't affect each other) | Allocates ORTHOGONAL directions for WRITING; MANCE ERASES from shared state. Different mechanism — orthogonality vs tangent projection. |
| **Plan 312 viable_manifold_graph** (R294) | Discrete safe-manifold navigation via pullback volume | Safe-manifold constraint on latent state | DISCRETE graph walk, not continuous tangent projection. No spectral weighting, no erasure. |
| **Plan 309 latent_field_steering** (R290) | 1D `s + α·v` injection | Additive latent correction | Single direction, fixed α, no subspace, no local tangent, INJECTION not ERASURE. |
| **Plan 405 spherical_steering** (R382) | Slerp toward single target | Additive correction | Single-target Slerp, no subspace, no local tangent. |
| **Plan 418 MAG** (R397) | Unsupervised direction mining via contrast | Direction extraction (the probe replacement) | Direction MINING, not gated correction. No subspace, no tangent, no trust region. |
| **Plan 235 SLoD** (R208) | Fréchet mean via tangent-space aggregation | Tangent-space math (Log/Exp maps) | Uses tangent space for KG AGGREGATION (Fréchet mean), not for ERASURE. No spectral weighting, no trust region. Different application. |

**Conclusion of the fusion grep:** every MANCE component has a shipped cousin, but the specific integration — *local per-sample tangent basis estimated from natural representations + spectral σ^α weighting + per-sample ε·r_i trust region, applied as a concept ERASURE (subtraction) on latent state* — is **not shipped**. The gap is threefold: (1) local vs global basis estimation, (2) spectral weighting of projection coordinates, (3) per-sample local-radius trust region. TILR (Plan 425) is the closest — it adds the γ-alignment gate to subspace projection but on a GLOBAL basis and for INJECTION, not ERASURE.

### 2.3 Latent-space reframing (mandatory per workflow §1 step 3)

The paper operates on LLM residual streams (d ∈ [768, 5376]) and CLIP image embeddings (d=768). Re-cast on the codebase's latent-state kernels:

**(a) HLA per-NPC latent state (8-dim, `riir-ai/crates/riir-engine/src/hla/`)**

The natural representations X⁽⁰⁾ are the HLA states of all NPCs in a zone at baseline (pre-event). The concept to erase is an unwanted emotion direction (e.g., "excessive fear" after a predator encounter). The local tangent basis at NPC i is estimated from the k nearest NPCs' HLA states — capturing how natural affect varies near NPC i. The spectral weighting prioritizes well-supported affect axes (high-σ directions where the NPC population varies most). The trust region ensures the edit doesn't push the NPC's affect outside the natural range for its neighborhood. This is **surgical per-NPC emotion erasure**: remove "fear" without damaging "curiosity" or "loyalty". Composes with `EmotionDirections::project` (read-side) and `CommittedFieldBlend` (Plan 321, personality commitment).

**(b) `latent_functor/` operations (`riir-ai/crates/riir-engine/src/latent_functor/`)**

The natural manifold is the set of latent states the functor has historically processed (a reference buffer of past states). The erasure direction is the functor's drift signal (the direction in which the functor is diverging from its intended behavior). The local tangent constrains the correction to directions that are natural for the functor's recent state distribution. The trust region prevents over-correction. This is a **manifold-constrained re-estimation** — a principled replacement for the scalar `tau_reest` threshold in `reestimation.rs`: instead of binary "coherence < tau → re-estimate", it's "re-estimate along the local tangent, bounded by the neighborhood radius, spectrally weighted by local support." Composes with `reestimation.rs` and `reestimation_steerer.rs`.

**(c) `cgsp_runtime/` curiosity signals (`riir-ai/crates/riir-engine/src/cgsp_runtime/`)**

Curiosity = prediction error. MANCE reframing: erase the "expected" component of the prediction error (the part that lies on the natural manifold of past prediction errors), leaving only the "surprising" component (off-manifold residual). This is a **curiosity novelty filter**: `curiosity_novel = ||error_off_manifold||`, `curiosity_expected = ||error_on_manifold||`. The NPC explores based on novelty, not raw error. Composes with `curiosity_class_router.rs` and the `pulse_bridge.rs`.

**(d) `NeuronShard` style_weights / freeze envelope / consolidation (`riir-neuron-db/src/`)**

The natural representations X⁽⁰⁾ are the pre-consolidation `style_weights[64]` of all shards in a zone. The concept to erase is a biased direction introduced during consolidation (e.g., a direction that over-represents one archetype). The local tangent basis from neighboring shards constrains the correction to what's natural for the shard population. The trust region prevents the correction from pushing the shard outside the natural style distribution. This is **surgical shard debiasing** — remove a biased direction from a shard's style_weights while preserving the shard's core style. Composes with `consolidation.rs` (Raven/δ-Mem), `freeze.rs` (`MerkleFrozenEnvelope`), and the `can_freeze` gate.

**(e) AnyRAG escalation gateway (`riir-neuron-db/src/gateway.rs`)**

Before retrieval, erase query-irrelevant concepts from the query embedding (e.g., remove "temporal" concept when querying for "spatial" patterns). The local tangent from natural queries constrains the erasure to preserve query-relevant information. This is **surgical query pre-filtering for retrieval**. Composes with `gateway.rs`.

**(f) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-dec/`)**

The local tangent basis is the discrete exterior derivative's local structure — the directions in which a cochain field varies naturally on the cell complex. Spectral weighting by local singular values corresponds to weighting by the local metric (edge lengths, face areas). The trust region ensures the edit doesn't violate the cochain's closure properties. This connects MANCE to the `hodge_decompose` (exact/coexact/harmonic split) — the harmonic component is the "natural manifold" of the flow field. **Curse-of-dimensionality caveat:** valid only for d ≤ 3 (2D maps, 3D belief regions). Do NOT apply to high-dim shards (d=64) or HLA (d=8 is borderline).

### 2.4 Fusion opportunities (the highest-value combinations)

- **F1: MANCE × TILR (Plan 425) × spectral_rewire (Plan 423).** TILR provides the γ-alignment gate (global invariant subspace); spectral_rewire provides the on-manifold/off-manifold decomposition; MANCE provides the local tangent basis + spectral weighting + trust region. Fusion: a unified manifold-constrained editing primitive that uses BOTH a global invariant subspace (γ gate: "is this correction in the right direction?") AND a local tangent basis (trust region: "is this step safe for the local manifold?"). The global gate filters direction; the local tangent filters magnitude.

- **F2: MANCE × non_interference_branches (Plan 329) × MAG (Plan 418).** MAG mines the concept direction to erase (replacing the MLP probe); non_interference_branches allocates the erased direction as a branch; MANCE performs the surgical erasure constrained to the local manifold. Fusion: mine → branch → erase, with double protection (branch orthogonality + manifold constraint).

- **F3: MANCE × HLA × CommittedFieldBlend (Plan 321).** When an NPC's committed personality has an unwanted trait (e.g., excessive fear after trauma), MANCE surgically removes it while preserving the rest of the personality. The committed blend's BLAKE3 commitment ensures the corrected state is versioned. The local tangent from neighboring NPCs' HLA states constrains the correction to what's natural for the NPC population.

- **F4: MANCE × freeze/thaw × NeuronShard.** When freezing a shard, use MANCE to remove unwanted concepts (e.g., a biased direction from training) before committing to the `MerkleFrozenEnvelope`. The local tangent from neighboring shards' style_weights constrains the correction. The freeze envelope's BLAKE3 commitment ensures the corrected shard is tamper-evident.

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. Architectural guide → riir-ai/.research/. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only (→ riir-train note, stop). | One-line note. No files created in this session. |

### Verdict: **GOAT** (not Super-GOAT)

**One-line reasoning:** MANCE's local tangent basis + spectral weighting + per-sample trust region combination is genuinely not shipped, but it is a principled refinement of the existing subspace-projection family (Plan 412/423/425) — the "local, spectrally-weighted, trust-bounded" counterpart to TILR's "global, alignment-gated" approach, applied to ERASURE rather than INJECTION.

**Novelty gate (Q1–Q4):**

1. **No prior art?** ⚠️ **PARTIAL YES.** The specific combination (local per-sample tangent from natural k-NN + σ^α spectral weighting + ε·r_i trust region + natural-manifold anchoring) is NOT shipped as a single primitive. BUT every component has a close cousin: `tilr` (Plan 425) does alignment-gated subspace projection (global basis), `spectral_rewire` (Plan 423) does on-manifold/off-manifold decomposition (weight-space), `subspace_steering` (Plan 412) does k-dim block correction. The gap is local-vs-global basis estimation + spectral weighting + trust region — a 3-component gap, not a fundamentally new mechanism. → Q1 = **YES (but borderline — it's a novel combination of existing pieces, not a new mechanism class)**.

2. **New class of behavior?** **BORDERLINE → NO.** MANCE enables "surgical concept erasure" — removing a concept while preserving other information. This IS a new capability (no shipped primitive can erase with a preservation guarantee), but it's in the same CAPABILITY CLASS as "latent state editing" — it's the "remove" counterpart to the existing "add" primitives (latent_field_steering, subspace_steering, TILR). It doesn't open a new family of behavior that was previously impossible; it makes the existing family surgical. Compare to Viable Manifold Graph (R294, Super-GOAT), which enabled "NPCs random-walk their affect" — a fundamentally new behavior. MANCE makes existing editing safer, not fundamentally new. → Q2 = **NO**.

3. **Product selling point?** **WEAK YES.** "Our NPCs can selectively remove personality traits without collateral damage to other cognitive capabilities" is a selling point, but it's an extension of the existing personality composition story (CommittedFieldBlend, PersonalityWeightedComposition), not a new pillar. Cannot finish "our NPCs do X no competitor can" — a competitor with orthogonal projection + non-interference branches has approximately the same capability, just less principled. → Q3 = **WEAK**.

4. **Force multiplier?** **YES.** Connects to ≥4 pillars/systems: HLA (AI pillar — per-NPC affect erasure), freeze/thaw (neuron-db pillar — shard debiasing), AnyRAG (neuron-db pillar — query pre-filtering), non_interference_branches (reasoning pillar — orthogonal allocation + surgical erasure), TILR (reasoning — alignment gate + local tangent), MAG (reasoning — probe replacement). But force multiplication alone does not make Super-GOAT.

**Q2=NO → verdict is GOAT (not Super-GOAT).** No private architectural guide needed; no Super-GOAT mandatory outputs triggered.

### MOAT gate per domain

| Domain | In scope? | MOAT contribution |
|--------|-----------|-------------------|
| **katgpt-rs** (public engine) | ✅ YES | Generic manifold-constrained erasure primitive — local tangent projection + spectral weighting + trust region. Research-grade primitive for the adoption funnel. **Correct home for the open primitive.** |
| **riir-ai** (private runtime) | Conditional | The latent-state application (HLA surgical emotion erasure, functor manifold-constrained re-estimation) is a *consumer* of the open primitive — land in riir-ai only if a GOAT-gate validates quality. |
| **riir-neuron-db** (private shards) | Conditional | The shard-level application (surgical concept removal from style_weights) is a *consumer* — land only if the shard-level GOAT gate validates. |
| **riir-chain** (private chain) | NO | No chain/LatCal/sync-boundary angle. |
| **riir-train** | NO | 100% modelless after probe replacement — no training dependency. |

### Why not Super-GOAT (the honest demotion)

The closest the codebase comes to MANCE is the trio **Plan 425 (TILR) + Plan 423 (spectral_rewire) + Plan 412 (subspace_steering)**. TILR already ships alignment-gated subspace projection with a no-harm guarantee; spectral_rewire already ships on-manifold/off-manifold decomposition; subspace_steering already ships k-dim block correction on latent state. A developer who reads those three plans, replaces the global basis with a local k-NN PCA, adds σ^α weighting, and adds an ε·r_i cap has built MANCE in ~80 lines on top of existing primitives. The *idea* is not moat-worthy; the *validated integration with local tangent estimation + spectral weighting + trust region* is a GOAT-tier quality primitive, not a Super-GOAT moat. Claiming Super-GOAT here would be claiming novelty over a combination that is 70% assembled from shipped pieces — the false-Super-GOAT failure mode.

---

## 4. Implementation routing

- **Open primitive** → `katgpt-rs/crates/katgpt-core/src/` (new module `manifold_erasure.rs`). Feature flag `manifold_erasure`. Generic math (k-NN, local SVD, weighted projection, bounded step), no game/chain/shard semantics.
- **Plan** → `katgpt-rs/.plans/426_manifold_concept_erasure_primitive.md`.
- **GOAT gate:** G1 (correctness — erasure reduces target-direction energy, preserves orthogonal directions), G2 (perf — local SVD + projection < budget), G3 (no regression), G4 (alloc-free hot path via scratch buffers), G5 (modelless — no training deps). **Not UQ-bearing** → no conformal floor needed.
- **Consumer wiring (deferred to follow-up issues, not this plan):**
  - riir-ai: HLA surgical emotion erasure (consumes the open primitive on 8-d HLA state).
  - riir-neuron-db: shard surgical concept removal (consumes on 64-d `style_weights`).
  - riir-ai: AnyRAG query pre-filtering (consumes on query embeddings).

---

## 5. Modelless-unblock protocol check (§3.5)

MANCE fits a nonlinear MLP probe (h=512, 800 SGD steps per refit, every τ=8 rounds). This is technically training. The §3.5 check:

1. **Freeze/thaw snapshot correction** — can a frozen snapshot fix the probe? **NO** — the probe is a diagnostic, not a correction. The probe FINDS the direction; the edit MECHANISM (tangent projection + spectral weighting + trust region) is what corrects. The mechanism is modelless.
2. **Raw/lora reader-writer hot-swap** — can a deterministically constructed adapter replace the probe? **PARTIALLY** — a reader-LoRA could project the latent state onto a concept direction, but this is just a linear projection. The probe's value is finding NONLINEAR concept directions. However, our substrate has modelless alternatives: MAG (unsupervised contrastive direction mining), CNA (contrastive neuron attribution), HLA `EmotionDirections` (pre-computed affect directions). These replace the probe modellessly.
3. **Latent-space correction** — the edit mechanism IS a latent-space correction (tangent projection + spectral weighting + trust region). ✅

**Verdict: MODELLESS-VALIDABLE.** The probe is the "LLM-as-implementation" pattern (R368) — it computes the DECISION "which direction to erase"; our substrate computes the same decision modellessly via MAG/CNA/EmotionDirections. The mechanism (the actual edit) is 100% modelless linear algebra. No riir-train dependency.

**The R368 guard check:** "N LLM/probe calls per step" → ask "what decision is each probe call computing?" → answer: "which direction in latent space carries the target concept" → our substrate computes this via MAG (contrastive mining) or CNA (contrastive attribution) or EmotionDirections (pre-computed). The probe is one *instantiation* of computing this decision; MAG/CNA/EmotionDirections is ours. This is the decision-structure case, NOT the LLM-dependent-process case. The R169 guard does NOT trigger.

---

## 6. The open primitive sketch

```rust
/// Manifold-constrained concept erasure — local tangent projection + spectral
/// weighting + per-sample trust region. Distilled from MANCE (arXiv:2607.03973).
///
/// The probe (which finds the concept direction) is a CONSUMER concern — this
/// primitive CONSUMES a pre-computed erasure direction, it does not train a
/// probe. Replace the MLP probe with MAG (Plan 418), CNA (Plan 087), or HLA
/// EmotionDirections for a modelless pipeline.

/// Configuration (all dimensionless, transfer across settings per the paper).
pub struct ManceConfig {
    /// Local-radius fraction (default 0.1). Dimensionless: displacement / r_i.
    pub epsilon: f32,
    /// Safety cap on per-sample step (default 64).
    pub lambda_max: f32,
    /// Spectral exponent (default 1.0). α=0: isotropic, α=1: σ-weighted, α=2: Mahalanobis-ish.
    pub alpha: f32,
    /// k-NN neighborhood size (default max(8, TwoNN floor)).
    pub k: usize,
    /// Tangent basis rank (default: TwoNN intrinsic dimension, floor 8).
    pub r: usize,
}

/// Estimate local tangent basis at x from natural neighbors.
/// Returns (B: [d×r] tangent basis, σ: [r] singular values).
fn estimate_local_tangent(
    x: &[f32],
    natural_neighbors: &[&[f32]],  // k nearest from X⁽⁰⁾
    r: usize,
    scratch: &mut TangentScratch,
) -> (TangentBasis<'_>, &[f32]);

/// Build spectrally-weighted tangent erasure direction.
/// d = B · diag(σ^α) · Bᵀ · u, then normalize.
fn tangent_erasure_direction(
    x: &[f32],
    gradient: &[f32],       // the concept direction to erase (from MAG/CNA/probe)
    basis: &TangentBasis<'_>,
    sigma: &[f32],
    alpha: f32,
    scratch: &mut ProjectionScratch,
) -> ErasureDirection<'_>;

/// Per-sample local-radius step size.
/// λ = min(λ_max, ε·r_i / <x, û>), where r_i = mean dist to k natural neighbors.
fn local_radius_step(
    x: &[f32],
    direction: &[f32],      // û (normalized)
    natural_neighbors: &[&[f32]],
    epsilon: f32,
    lambda_max: f32,
) -> f32;

/// One MANCE erasure step: x̃ = x - λ·<x, û>·û.
/// Zero-alloc: writes into `out`. Reuses `scratch` across calls.
fn manifold_erasure_step_into(
    x: &[f32],
    gradient: &[f32],
    natural_neighbors: &[&[f32]],
    config: &ManceConfig,
    scratch: &mut ManceScratch,
    out: &mut [f32],
) -> ManceStepInfo;
```

**Reuse map (do not duplicate):**

| Operation | Source | Notes |
|---|---|---|
| SVD → local tangent basis | `thin_svd_into` (Plan 301, `subspace_phase_gate`) | The local PCA SVD reuses the existing SVD scratch infrastructure |
| k-NN query | New (simple — L2 distance to k natural neighbors) | O(k·d) per sample; can use existing `simd_dot_f32` for distance |
| Subspace projection | `spectral_rewire::project_core` (Plan 423), `tilr` (Plan 425) | Same projection math; MANCE adds σ^α weighting |
| SIMD dot products | `simd_dot_f32` (`katgpt-types/simd`) | Used for projection coefficients, norms, distances |

**Family relationship (the subspace-projection family):**

| Primitive | Basis | Gating | Operation | Domain |
|---|---|---|---|---|
| Plan 412 `subspace_steering` | Global, k-dim block | Fixed α_j | INJECTION `s + Σ α_j u_j` | Latent state |
| Plan 423 `spectral_rewire` | Global SVD of W₀ | None (fixed projection) | DECOMPOSITION `ΔW = ΔW* + ΔW⊥` | Weight deltas |
| Plan 425 `tilr` | Global invariant subspace U_r | γ-alignment gate `η·γ` | INJECTION `s + η·γ·Πd` | Latent state |
| **MANCE (this)** | **Local k-NN tangent B_i** | **Spectral σ^α + trust region ε·r_i** | **ERASURE `x - λ·<x,û>·û`** | **Latent state** |

MANCE is the **local, spectrally-weighted, trust-bounded erasure** member of the family.

---

## 7. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ The mechanism (local tangent + spectral weighting + trust region) is pure linear algebra. The probe is replaced by MAG/CNA/EmotionDirections (modelless). No gradient descent at inference. |
| Latent-to-latent preferred | ✅ Operates entirely in latent space. Never crosses to tokens. |
| Use sigmoid not softmax | ✅ The spectral weighting uses σ^α (power weighting), not softmax. The trust region is a hard cap (min), not a soft gate. Consistent with the codebase's hard-gate + sigmoid pattern. |
| Freeze/thaw over fine-tuning | ✅ The natural reference X⁽⁰⁾ is a frozen snapshot (pre-edit state). The tangent basis is estimated from this frozen reference. No weight mutation. |
| 5-repo discipline | ✅ Open primitive → katgpt-rs; HLA wiring → riir-ai (deferred); shard wiring → riir-neuron-db (deferred). |
| Raw scalars at sync boundary | ✅ The tangent basis B_i and singular values σ_i are fixed-size raw f32 arrays — deterministic, bit-identical across quorum if X⁽⁰⁾ is synced. The erasure direction (from MAG/CNA) is also a fixed-size vector. No variable-length embedding crosses sync. |
| Zero-alloc hot path | ✅ `manifold_erasure_step_into` uses pre-allocated `ManceScratch` (tangent basis buffer, projection buffer, distance buffer). All writes in-place. |
| CPU/GPU/ANE auto-route | ✅ At HLA scale (d=8, k=8, r≤8), SIMD CPU. At shard scale (d=64, k=16, r≤16), SIMD CPU or GPU matmul. k-NN distance computation is embarrassingly parallel. |
| Keep files < 2048 lines | ✅ The primitive module should be ~400-600 lines. |

---

## TL;DR

MANCE = manifold-constrained concept erasure via local tangent projection + spectral σ^α weighting + per-sample ε·r_i trust region, anchored to natural (unedited) representations. The MLP probe is the "LLM-as-implementation" — replace with MAG/CNA/EmotionDirections for a modelless pipeline. The mechanism is 100% linear algebra. **GOAT** — the local tangent + spectral weighting + trust region combination is genuinely not shipped (TILR/spectral_rewire use global bases), but it's a principled refinement of the subspace-projection family, not a new capability class. Plan filed at `.plans/426_*.md`, feature flag `manifold_erasure`, open primitive in katgpt-rs. Not Super-GOAT because Q2 (new capability class) is NO — it's the "surgical remove" counterpart to existing "add" primitives, not a new family.
