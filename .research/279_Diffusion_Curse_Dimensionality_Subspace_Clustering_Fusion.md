# Research 279: Diffusion ≡ Subspace Clustering — Phase Transition Gate & Runtime Jacobian SVD

> **Source:** Wang, Zhang, Zhang, Chen, Ma, Qu. *Breaking the Curse of Dimensionality: Diffusion Models Efficiently Learn Low-Dimensional Distributions*. [arXiv:2409.02426](https://arxiv.org/abs/2409.02426). U-Michigan / UC Berkeley. v5, 9 Jun 2026.
> **Date:** 2026-06-22
> **Status:** Active — Super-GOAT guide spawned at `riir-neuron-db/.research/001_*.md`.
> **Related Research:** 039 (SpectralQuant — closest shipped cousin), 111 (Emergent Analogical / Dirichlet Energy), 121 (Hierarchical Concept Geometry), 136 (Latent Prediction Sample Complexity), 138 (LeJEPA Identifiability), 269 (Variable-Width Stage-Gated HLA), 257 (Functional Attention Spectral Transport).
> **Related Plans:** 077 (SpectralQuant), 138 (Stiff/Soft Subspace Anomaly Gate), 156 (Spectral Hierarchy Diagnostic), 246 (Spectral Irrep Pruner), 276 (MicroRecurrentBeliefState), 301 (this paper's open primitive), `riir-neuron-db/.plans/002` (private consolidation gate).
> **Cross-ref (riir-neuron-db):** Research 001 — *Subspace Consolidation Quality Gate* Super-GOAT guide (private).
> **Classification:** Public (katgpt-rs engine note). The theoretical paper is public; its distillation into runtime primitives is open (this note + Plan 301). The private Super-GOAT selling point lives at `riir-neuron-db/.research/001_*.md`.
> **PASS-Redirects (synthesis):** Gong et al. [arXiv:2607.22531 "Twins: Learn to Predict Unified Representations with Focal Loss"] — PASS. The paper's intrinsic-dimension analysis (Two-NN: SigLIP ID≈15 vs VAE ID≈35) + the resulting "DiT underfits high-ID VAE" optimization-imbalance diagnosis reinforce the N≥d sample-sufficiency gate this note ships (Plan 301). The Twins channel-concat representation + Focal Loss are image-diffusion training infrastructure (DiT/VAE/ViT/ImageNet) we do not ship. Diagnostic stage → covered here (N≥d gate) + Research 394 (within-class effective rank, conditional alignment) + Research 409 (Two-NN local estimation); training stage → riir-train (image diffusion, out of scope).

---

## TL;DR

Wang et al. prove that training a diffusion model on a **mixture of low-rank Gaussians (MoLRG)** is *exactly equivalent* to solving the canonical **subspace clustering** problem, and that the minimal number of samples for the model to generalise scales **linearly with the intrinsic dimension d** (Theorem 4: N ≥ d → recover subspace, N < d → fail with constant probability). They also show the **Jacobian of the denoising autoencoder (DAE) at t ≈ 0.8 is low-rank**, and its leading singular vectors are **semantic task vectors** — directions in latent space whose traversal produces controlled, interpretable changes (Section 5.2).

**Distilled for katgpt-rs (modelless, inference-time):**

1. **Phase-transition gate primitive**: `N ≥ d` is a *sample-sufficiency* check, complementary to our existing `spectral_flatness()` *output-convergence* check in `katgpt-rs/src/sleep/consolidation.rs`. Together they form a two-sided quality gate: "do we have enough inputs?" (N ≥ d) AND "is the output converged?" (flatness < τ). Ships as a generic numeric primitive — no game/shard semantics.
2. **Runtime Jacobian SVD primitive**: given any differentiable map `f: R^n → R^n` (HLA evolution kernel, latent functor, projection head), estimate its Jacobian at a point via forward differences, SVD it, and expose the leading singular vectors as candidate "task directions". This is the modelless, inference-time analog of the paper's Section 5.2 procedure — no training, no diffusion model required.
3. **Theoretical justification for SpectralQuant (R039)**: the paper proves that for MoLRG-structured data, eigenbasis rotation + truncation to the top-d eigenvectors is *optimal* — exactly what SpectralQuant's `calibrate_eigenbasis()` does offline. This elevates SpectralQuant from "empirically works" to "provably optimal under MoLRG".

The **private Super-GOAT selling point** — NPCs that self-discover their own emotional/cognitive axes from runtime experience, shards that prove they have enough wake events to consolidate, KG triples formally grounded in latent subspace geometry — is documented at `riir-neuron-db/.research/001_*.md`.

---

## 1. Paper Core Findings

### 1.1 The MoLRG model

Data `x_0 ∈ R^n` is drawn from a mixture of K low-rank Gaussians:

```
x_0 ~ Σ_k π_k N(μ*_k, Σ*_k),   rank(Σ*_k) = d_k < n
```

Each component lives on a d_k-dimensional linear subspace spanned by the orthonormal columns of `U*_k ∈ O_{n×d_k}`. **Motivation**: real image datasets approximately lie on a union of low-dimensional manifolds; locally each manifold ≈ its tangent space (a linear subspace). Empirically validated on MNIST / CIFAR-10 / FFHQ (Fig 1).

### 1.2 The equivalence theorems

Under the parameterisation `x_θ(x_t, t) = Σ_k ŵ_k(θ; x_0) · U_k U_k^T x_t` (a mixture-of-experts with linear encoder `U_k^T` + decoder `U_k`, weighted by hard-max assignment), the diffusion training loss (5) becomes:

- **K = 1 (Theorem 1)**: equivalent to PCA — `max_U Σ_i ‖U^T x^(i)‖²  s.t. U^T U = I_d`.
- **K > 1 (Theorem 3)**: equivalent to **K-subspace clustering** — assign each sample to its best-fit subspace, then maximise projected energy.

### 1.3 Sample complexity (Theorem 4)

Under Assumptions 1–3 (orthogonal subspaces, equal dims, hard-max):

- **If N_k ≥ d for all k**: ∃ permutation Π such that with prob ≥ 1 − 2K²N⁻¹ − Σ_k (½N_k^{−d+1} + exp(−c₂N_k)):
  `‖Û_Π(k) Û_Π(k)^T − U*_k U**_k^T‖_F ≤ c₁ · ‖E‖_F / (√N_k − √d − 1)`
- **If N_k < d for some k**: ∃ k such that the recovery error is bounded *below* by `β − c₁'·‖E‖_F/(√d − √N_k − 1)`, β = √2·min(d − N_k, n − d).

**Implication**: there is a sharp phase transition at N = d. Below it, recovery is information-theoretically impossible; above it, recovery error decays as O(‖E‖_F/√N).

### 1.4 Phase transition in real diffusion models (Section 5.1)

On CIFAR-10, CelebA, FFHQ, AFHQ with U-Net DDPM: the generalisation score (GL, fraction of generated samples distinct from training set) follows a sigmoid in `log₂(N / ID)`, where ID is the estimated intrinsic dimension. The threshold `N_min ≈ 630 · ID` for real images (vs 50 · ID for synthetic MoLRG). **Linear in intrinsic dim, not ambient dim.**

### 1.5 Jacobian SVD → semantic task vectors (Section 5.2)

For a pretrained DDPM, the DAE satisfies a first-order Taylor expansion `x_θ(x_t + δ_t, t) ≈ x_θ(x_t, t) + J_t · δ_t`, where `J_t = ∇_{x_t} x_θ(x_t, t)`. SVD: `J_t = P Σ Q^T`. The paper shows:

- `J_t` is **low-rank at certain timesteps** (U-shaped curve, minimum rank at t ≈ 0.815 across CIFAR-10 / CelebA / FFHQ / AFHQ).
- The leading right singular vectors `q_1, …, q_r` are **semantic task vectors**: steering `x_t → x_t + α·q_i` produces interpretable edits (gender, hairstyle, color) on the generated image.
- Random directions on the unit sphere produce no perceptible change → the subspace spanned by `P = [p_1, …, p_r]` is non-trivial and semantically aligned.

---

## 2. Distillation

### 2.1 What is transferable (stripped of the training setup)

The paper's value to a **modelless inference** codebase is threefold, in decreasing order of novelty:

| # | Primitive | Transferable insight | Existing closest cousin |
|---|-----------|---------------------|------------------------|
| **A** | **Runtime Jacobian SVD → task directions** | For any latent map `f`, `J_f(x)` at a carefully-chosen "operating point" has low rank, and its singular vectors are interpretable axes. No training needed — just forward differences + SVD. | None shipped. SpectralQuant (R039) does *offline* covariance eigendecomposition, not *online* Jacobian SVD. EmotionDirections (P162) uses *pre-computed* direction vectors, not *runtime-discovered*. |
| **B** | **Phase-transition sample-sufficiency gate** | Given N samples and an estimated intrinsic dim d (via participation ratio), `N ≥ d` is a hard gate: below it, any subspace estimate is information-theoretically invalid. | `consolidation.rs::spectral_convergence_check()` measures *output* flatness, not *input* sufficiency. Stiff/Soft Subspace Anomaly Gate (P138) measures subspace stability, not sample count. |
| **C** | **Theoretical optimality of eigenbasis rotation** | Under MoLRG, the top-d eigenvectors of the covariance are the *optimal* subspace basis — SpectralQuant's `calibrate_eigenbasis()` is provably optimal, not just empirically good. | SpectralQuant (R039, P077) — direct theoretical validation. |

### 2.2 Latent-space reframing (mandatory per workflow §1 step 3)

The paper operates on raw image pixels `x_0 ∈ R^n`. We re-cast each mechanism on the codebase's latent-state kernels:

**(a) HLA per-NPC latent state (8-dim, `riir-ai/crates/riir-engine/src/hla/`)**

HLA's 8-dim state IS a projection `U_k U_k^T x_t` onto an 8-dim subspace — but the basis `U_k` is currently *hand-crafted* (valence/arousal/desperation/calm/fear + 3 unused). The paper says: at runtime, the Jacobian of the HLA evolution kernel `evolve_hla()` has low rank, and its singular vectors reveal the *actual* semantic axes for this NPC's recent experience. Each NPC's emotional geometry becomes unique, data-derived, and evolves over time. **This is the private Super-GOAT angle — see `riir-neuron-db/.research/001_*.md`.**

**(b) `latent_functor/` operations (`riir-ai/crates/riir-engine/src/latent_functor/`)**

Each functor application is one "DAE step". The functor's Jacobian at the current latent state, when SVD'd, reveals which directions the functor is *sensitive* to. Reframing: a functor has "converged" on a latent state when its Jacobian rank drops below a threshold — the *range* of `J_f` is the active subspace. This generalises `reestimation.rs`'s coherence threshold (currently scalar cosine) to a *rank-k Frobenius residual fit* — already the direction Plan 318 (rank-k upgrade) is moving.

**(c) `cgsp_runtime/` curiosity (`riir-ai/crates/riir-engine/src/cgsp_runtime/`)**

Curiosity = prediction error in latent space. The paper's reframing: a query is "novel" iff it lies *outside* the span of recently-seen subspaces. Operationally: project the query onto the runtime SVD basis `U_k` (from recent wake events); if the projection residual `‖x − U_k U_k^T x‖` is large, the query is novel. This is a *geometric* curiosity signal — complementing the existing *entropy* signal.

**(d) LatCal fixed-point commitment (`riir-chain/src/encoding/latcal*.rs`)**

The paper's `D_k = diag(s_t² λ_{k,i} / (γ_t² + s_t² λ_{k,i}))` coefficients are bounded `[0, 1]` scalars — one per (subspace, eigenvalue) pair. Reframing: commit the `D_k` coefficients (raw scalars, ~64 f32 per shard) to chain via LatCal fixed-point; keep the `U_k` bases local (latent, large, n×d f32). This is the textbook sync-boundary bridge pattern from AGENTS.md: raw scalars cross, full embeddings stay.

**(e) `NeuronShard` `style_weights[64]` / freeze envelope / consolidation (`riir-neuron-db/src/`)**

- `style_weights[64]` IS `U_k` — a 64-dim orthonormal-ish basis (or its flattened representation). The freeze envelope commits this basis + its eigenvalues. The paper proves this commitment is *sufficient* to reconstruct the MoLRG component (Lemma 1: posterior mean is a closed-form function of `U_k`, `Λ_k`, `μ_k`).
- `consolidation.rs` (Raven/δ-Mem) IS subspace clustering over shards — the paper proves it converges with `N ≥ d` samples. The existing `spectral_convergence_check()` measures *output* flatness; the paper's `N ≥ d` rule adds the *input* sufficiency gate.
- AnyRAG escalation gateway: use SVD of the local shard's `style_weights` to determine if a query lies in-span. If out-of-span (large projection residual), escalate to external knowledge. This is a *geometric* escalation trigger, complementing the existing confidence-based trigger.
- Vibe KG triple emission: "entity has trait X" ↔ entity's representation lies in subspace `U_X`. The paper's correspondence between basis vectors and semantic attributes gives a *principled* foundation for KG triple emission from latent geometry — currently KG triples are emitted from cosine/dot-product thresholds without theoretical grounding.

### 2.3 Fusion

The highest-value Super-GOAT is a **three-way fusion**:

```
Paper 2409.02426 (phase transition + Jacobian SVD)
    ×
R039 SpectralQuant (offline eigenbasis calibration, shipped)
    ×
riir-neuron-db consolidation.rs (Raven/δ-Mem, shipped)
    =
"Self-certifying shards": a shard that (1) proves it has enough wake events
via N ≥ d, (2) proves its output is converged via spectral flatness, and
(3) exposes its semantic axes via runtime Jacobian SVD — all modelless,
all inference-time, all BLAKE3-committed.
```

**Second fusion** (HLA angle, private to riir-ai):

```
Paper 2409.02426 §5.2 (runtime Jacobian SVD → semantic task vectors)
    ×
R123 Latent Functor Runtime Guide (existing Super-GOAT)
    ×
R269 Variable-Width Stage-Gated HLA Subspace (recent fusion)
    =
"Self-discovering NPCs": each NPC's HLA axes are not hand-coded but
*discovered* from its own experience via runtime Jacobian SVD of its
evolve_hla() kernel. Two NPCs in the same zone develop different
emotional geometries because their experience trajectories differ.
```

**Third fusion** (LatCal angle, private to riir-chain):

```
Paper 2409.02426 (D_k coefficients are bounded [0,1] scalars)
    ×
R212 Gemini Fourier × LatCal (canonical LatCal Super-GOAT precedent)
    ×
riir-neuron-db freeze.rs (MerkleFrozenEnvelope)
    =
"Committed subspace coefficients": freeze the D_k scalars (one per
eigenvalue) into the Merkle envelope as raw LatCal fixed-point values,
while the U_k basis stays local. Syncs the *summary* of the latent
state across quorum without syncing the full embedding.
```

---

## 3. Verdict

### Novelty gate (Q1–Q4)

| Q | Answer | Evidence |
|---|--------|----------|
| **Q1: No prior art?** | **YES** (for the combination). Closest cousins: SpectralQuant (R039, offline covariance eigendecomposition — different), `consolidation.rs::spectral_convergence_check` (output flatness, not input sufficiency), EmotionDirections (P162, pre-computed directions, not runtime-discovered). Three-layer grep (notes + code + vocabulary translation) confirms no shipped primitive does runtime Jacobian SVD for semantic direction discovery, nor N ≥ d as a sample-sufficiency gate. |
| **Q2: New class of behavior?** | **YES**. "Self-certifying shards" (input + output quality gates) and "self-discovering NPCs" (runtime-derived emotional axes) are capability classes no incumbent has. Existing shards are either committed-or-not; existing HLA axes are hand-coded. |
| **Q3: Product selling point?** | **YES**. "Our NPCs discover their own emotional axes from experience, validated by phase-transition theory, instead of being hand-coded with valence/arousal/fear." "Our shards prove they have enough samples to be stable, via N≥d phase transition." |
| **Q4: Force multiplier?** | **YES**. Touches 5 of 6 Super-GOAT factory modules: HLA (riir-ai), latent_functor (riir-ai), cgsp_runtime (riir-ai), neuron-db shards/freeze/consolidation (riir-neuron-db), LatCal fixed-point (riir-chain). Only `sense/` (katgpt-rs) is untouched, and even there the Jacobian SVD primitive is reusable. |

**All 4 YES → Super-GOAT.** Selling point: *"Shards and NPCs that mathematically prove their own stability and self-discover their semantic axes — no hand-coded emotional dimensions, no heuristic consolidation thresholds."*

### Mandatory outputs (this session)

Per workflow §1.5, all three are produced in this session:

1. **Open primitive** → `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs` (Plan 301). Generic numeric: participation-ratio-based intrinsic dim estimate, N ≥ d phase-transition gate, forward-difference Jacobian SVD. **No game semantics, no shard semantics.**
2. **Private Super-GOAT guide** → `riir-neuron-db/.research/001_Subspace_Consolidation_Quality_Gate_Guide.md`. Selling-point doc with connection map, latent-vs-raw boundary, validation protocol (G1–G5), implementation priority.
3. **Plans** → `katgpt-rs/.plans/301_runtime_subspace_phase_gate_primitive.md` (open) + `riir-neuron-db/.plans/002_phase_transition_consolidation_gate.md` (private).

### Tier justification

| Tier | Criteria | Met? |
|------|----------|------|
| Super-GOAT | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | ✅ all 4 |
| GOAT | Provable gain, not new class | n/a |
| Gain | Incremental | n/a |
| Pass | Not relevant / training-only | n/a — the paper's *training* theory is a justification, but the *distilled primitives* (Jacobian SVD, N≥d gate) are modelless and inference-time. |

**One-line reasoning**: the paper provides (a) theoretical proof that our existing SpectralQuant + consolidation approach is optimal under MoLRG, AND (b) two novel modelless primitives (runtime Jacobian SVD, N ≥ d phase-transition gate) that fuse across 5 Super-GOAT factory modules to enable self-certifying shards and self-discovering NPCs — a capability class no competitor has.

---

## 4. What stays open vs private

| Artifact | Repo | Visibility | Why |
|----------|------|-----------|-----|
| `subspace_phase_gate.rs` primitive | katgpt-rs | **Open (MIT)** | Generic numeric math: participation ratio, N ≥ d check, Jacobian SVD. No game IP, no shard IP. Reusable beyond our stack. |
| Plan 301 | katgpt-rs | **Open** | Generic primitive plan. |
| Research 279 (this note) | katgpt-rs | **Open** | Distillation of a public paper. |
| Research 001 Super-GOAT guide | riir-neuron-db | **Private** | Selling point: self-certifying shards, semantic basis extraction from `style_weights[64]`. |
| Plan 002 | riir-neuron-db | **Private** | Consolidation gate + semantic basis extraction plan. |
| (Future) HLA self-discovery | riir-ai | **Private** | The runtime Jacobian SVD applied to `evolve_hla()` — flagged as follow-up, not implemented this session. |

---

## 5. Risks and honest caveats

1. **The Jacobian SVD at t ≈ 0.8 requires a "DAE-like" map.** The paper's result is specifically about the denoising autoencoder of a *trained* diffusion model. For our HLA evolution kernel or latent functors, the "right operating point" (analog of t ≈ 0.8) is unknown and must be empirically discovered. Mitigation: the primitive ships a generic `jacobian_svd_at` that takes the map and the point as parameters; the *choice* of point is the caller's responsibility (private riir-neuron-db / riir-ai code).
2. **Participation ratio ≠ exact intrinsic dimension.** The paper uses numerical rank at a specific energy threshold (eq. 52, η = 0.99). SpectralQuant already ships `participation_ratio()` (continuous) — we use it as a fast proxy, but the gate's threshold may need calibration per domain. Mitigation: expose both `participation_ratio()` and `numerical_rank(η)` in the primitive.
3. **The MoLRG assumption (orthogonal subspaces) is strong.** Real NPC behaviour subspaces (combat / dialog / social / economic) are *not* orthogonal. The paper's Theorem 4 requires orthogonality; the non-orthogonal case is open. Mitigation: the N ≥ d gate remains a *necessary* condition (you can't learn a d-dim subspace from fewer than d samples regardless of orthogonality); it's not a *sufficient* condition in the non-orthogonal case, but that's fine for a quality gate.
4. **"Self-discovering NPCs" is a claim about riir-ai, not katgpt-rs.** The open primitive ships the *math*; the *claim* that NPCs self-discover emotional axes requires applying the math to `evolve_hla()` in riir-ai, which is out of scope for this session. The private Super-GOAT guide (riir-neuron-db/.research/001) documents the claim and validation protocol; riir-ai implementation is a follow-up plan.
5. **Paper is about diffusion models, we don't train diffusion models.** The theoretical results (Theorems 1–4) are about diffusion training loss. We use them as *justification* for the modelless primitives, not as directly-implementable algorithms. The Jacobian SVD result (§5.2) is the directly-transferable part.

---

## 6. References

- **Source paper**: [arXiv:2409.02426](https://arxiv.org/abs/2409.02426) — Wang et al., v5 Jun 2026.
- **Closest internal cousin (shipped)**: `katgpt-rs/.research/039_SpectralQuant_Calibrated_Eigenbasis_KV_Compression.md` + `katgpt-rs/crates/katgpt-spectral/src/spectral.rs` (`participation_ratio`, `calibrate_eigenbasis`).
- **Closest internal cousin (consolidation)**: `katgpt-rs/src/sleep/consolidation.rs::spectral_convergence_check` + `riir-neuron-db/src/spectral_flatness.rs`.
- **Latent functor rank-k**: `riir-ai/.plans/318_latent_functor_rank_k_upgrade.md`.
- **Stage-gated HLA**: `katgpt-rs/.research/269_Variable_Width_Shape_Adapter_Fusion.md`.
- **Canonical LatCal Super-GOAT precedent**: `katgpt-rs/.research/212_Gemini_Fourier_LatCal_Fusion_Verdict.md`.
- **Private Super-GOAT guide**: `riir-neuron-db/.research/001_Subspace_Consolidation_Quality_Gate_Guide.md`.

> **PASS-Redirects (synthesis):** Feng et al. [arXiv:2609.28473 "On the Diffusibility of High-Dimensional Latents"] — adds the empirical half of this note's subspace story: reconstruction-tuned encoders measurably collapse effective dimensionality (R90 672→129 of 1024 on finetuned DINOv2-L; MAE R90=103 of 768), and x0-prediction — not a dimensionality-reducing adapter — restores diffusibility in the FULL ambient space. Our own surfaces already measure + exploit the same collapse (KV-key d_eff≈4–6, style_weights rank≈3 per riir-train Plan 316), as compression headroom rather than training obstacle.
