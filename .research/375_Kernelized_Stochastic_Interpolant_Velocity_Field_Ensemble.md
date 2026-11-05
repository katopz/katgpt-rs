# Research 375: Kernelized Stochastic Interpolants — Velocity-Field Ensemble Combination

> **Source:** *Generative Modeling via Kernelized Stochastic Interpolants* — Coeurdoux, Lempereur, Cuvelle-Magar, Mallat, Vanden-Eijnden. ICML 2026 SPIGM Workshop. [arxiv 2602.20070](https://arxiv.org/abs/2602.20070)
> **Date:** 2026-07-04
> **Status:** Active
> **Related Research:** 288 (KARC — delay-basis ridge), 302 (FAME CommittedFieldBlend — sigmoid projection blend), 291 (Cross-Resolution Spectral Transport), 276 (PersonalityWeightedComposition), 115 (PEIRA — predictor_with_scratch), 257 (FuncAttn — Tikhonov spectral transport), 218 (Breakeven Complexity Router), 322 (Conformal Seasonal Pools — UQ overlay)
> **Related Plans:** 376 (this primitive — open), riir-ai 385 (runtime wiring), 308 (KARC), 321 (CommittedFieldBlend), 310 (Cross-Resolution), 286 (FuncAttn)
> **Cross-ref (riir-ai):** Research 170 (Per-NPC Velocity-Field Ensemble Composition Guide)
> **Classification:** Public
> **PASS-Redirects (synthesis):** Tong et al. [arXiv:2608.05811 "Energy-Guided Flow Matching"](https://arxiv.org/abs/2608.05811) — PASS. Pixel-space image generation training recipe — replaces the fixed clean endpoint with a heat-kernel-filtered moving endpoint `y_t(x) = F⁻¹(exp(-a·h(x,t)·ρ²)·b_x)` releasing low→high image-frequency content via sample-adaptive spectral-energy scheduling. Mechanism is entirely training-time velocity-target construction (Eq. 16: `v_t = y_t - ε + t·∂_t y_t`); the paper explicitly states inference uses the same solver with no FFT/heat-kernel/bisection at runtime. Out of scope for our model-based track too (Plan 318/059/066 train LLM/game-AI/linear-attention, not image DiT). No modelless primitive to extract (there is nothing at inference). Closest cousins: this note (stochastic interpolant integrator), R150 (RecFM cross-scale flow matching), R369 (renoise-CE flow reasoning verifier) — all latent-reasoning, none pixel-space.
> **PASS-Redirects (synthesis):** Saied, Sheebaelhamd, Vernade, Fleuret [arXiv:2610.40271 "Gaussian Distillation of Flow Policies"](https://arxiv.org/abs/2610.40271) — PASS. One-pass Gaussian readout distillation of continuous flow policies (mean identity `ε + v(ε,0|c) = E[m|c]`; variance-from-velocity-slope; frozen-feature readouts on 0.02% of states; closed-form likelihoods; 1.91×/1.57× latency at near-teacher success). The identities would land HERE — this module is the workspace's only continuous velocity-field substrate (ridge-solved frozen-field combination) — but no continuous flow sampler ships as a serving consumer today, so nothing is filed. RE-OPEN trigger: a continuous flow or velocity-field sampler landing as a serving consumer (e.g. out of the d2f/velocity-field lanes) re-opens this PASS against the paper's moment identities.

---

## TL;DR

The paper replaces neural-network training for stochastic-interpolant generative models with a **P×P kernel linear system over pre-trained velocity fields**. Given P frozen velocity fields `{b_i(x)}` (any architecture, any training stage, any source domain), the combined drift is `b̂_t(x) = Σ_i η_t^i · b_i(x)`, where `η_t ∈ R^P` is **solved once per time grid** from data pairs via `K_t η_t = r_t` (`K_t` = Gram of velocity-field outputs, `r_t` = cross-correlation with target derivative). No gradient descent. No architecture constraint. Cross-domain composition works: models trained on F-MNIST + EMNIST + K-MNIST combine into a better MNIST generator than any single MNIST model — purely via linear algebra.

**Distilled for katgpt-rs (modelless, inference-time):**
A `VelocityFieldEnsemble<P, D>` primitive that takes P frozen `VelocityField` impls (a trait wrapping any forward drift `b_i(x) -> R^d`), accumulates the P×P Gram matrix and P-dim right-hand-side from N data pairs `(z_n, a_n)`, and solves `η = K^{-1} r` reusing the existing `crates/katgpt-core/src/linalg/ridge_solve.rs`. The combined drift `b̂(x) = Σ_i η_i b_i(x)` is then a single zero-alloc evaluation. **No game IP, no chain IP, no shard IP — generic algebraic ensemble combination.**

---

## 1. Paper Core Findings

### 1.1 The primitive (Proposition 2.1)

The drift of a stochastic interpolant `I_t = α_t z + β_t a` (z = noise, a = data) is the minimizer of the regression loss `L_b[b̂] = E[|b̂(I_t) − İ_t|²]`. Restricting `b̂` to the span of P feature-gradient functions `{∇φ_i(x)}` (the paper's "feature map"), the unique minimizer under positive-definite Gram is:

```
b̂_t(x) = Σ_i η_t^i · ∇φ_i(x),    K_t η_t = r_t
K_t[i,j] = E[∇φ_i(I_t) · ∇φ_j(I_t)]      (P×P Gram, P independent of d)
r_t[i]   = E[∇φ_i(I_t) · İ_t]             (P-dim RHS)
```

**Empirical system** (eq. 7): given N sample pairs `(z_n, a_n)`, set `I_t^n = α_t z_n + β_t a_n`, `İ_t^n = α̇_t z_n + β̇_t a_n`, and solve the empirical P×P system. Solved **once per time-grid point** `t_k = k/K` before generation; the `{η_{t_k}}` are then reused for any number of samples.

### 1.2 The killer application (§2.5, §3.3, Appendix E)

If `{b_i^t(x)}` are themselves **pre-trained velocity fields** (any source: flow matching, diffusion, different architecture, different training dataset, different training stage), they can be used DIRECTLY as the feature gradients by setting `∇φ_i(x) := b_i^t(x)`. Then:

- The combined drift is `b̂_t(x) = Σ_i η_t^i b_i^t(x)` — a **weighted ensemble of pre-trained models**.
- Weights `η_t` are **solved from data pairs** via the P×P system — NOT trained, NOT averaged, NOT evolved online.
- The paper demonstrates (§3.3, Appendix E): 20 weak MNIST U-Nets (50–100 SGD steps each, individually producing noise) compose via Algorithm 1 into a generator that produces **recognizable digits** with no additional training. Cross-domain composition (F-MNIST + EMNIST + K-MNIST + MNIST, 40 models) **beats** the 10-MNIST-only ensemble — source-domain velocity fields provide useful low-level feature gradients the linear system repurposes.

This is "Model Soup" / "Diffusion Soup" generalized: works across **architectures** and **training stages** (the paper's Contribution 4), not just same-architecture weight averaging.

### 1.3 Optimal diffusion coefficient D*_t (Proposition 2.2)

Because feature maps are finite-dimensional, the kernel is rank-≤P and the drift estimate is approximate. The diffusion coefficient `D_t` matters. The pointwise minimizer of the Girsanov path-KL bound (eq. 10) is:

```
D*_t = α_t γ_t / β_t,    γ_t := α_t β̇_t − α̇_t β_t > 0
```

`D*_t → ∞` as `t → 0` (strong Gaussian-regime diffusion) and `D*_t → 0` as `t → 1` (pure ODE transport near data). The integrator (eq. 14) handles both limits seamlessly via a trapezoidal scheme that resamples from Gaussian at the singular endpoint — no clamping. **The time-reversed SDE under `D*_t` is an OU-type process independent of the target μ** (Appendix B) — a notable structural property.

### 1.4 The Hilbert-space theorem (Appendix A.1)

Under a *characteristic* kernel (Gaussian RBF, Laplacian, inverse-multiquadric — all infinite-dim), the drift ansatz recovers the true velocity field `b_t` exactly. **The finite-P / finite-feature regime we ship is an approximation; the theorem bounds the gap.** This is the principled basis for "more features → better approximation".

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface — verify before any novelty claim)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| Closed-form P×P ridge solve `η = (K + λI)^{-1} r` | **`linalg::ridge_solve`** — `ridge_solve_direct_f32`, `ridge_solve_direct_f64`, `ridge_solve_woodbury_f32`, `chol_solve_f32` | Plan 308 T1.6, `crates/katgpt-core/src/linalg/ridge_solve.rs` (the canonical P×P Cholesky path KARC + PEIRA + FuncAttn all consume) |
| Per-NPC trajectory forecaster via delay-basis ridge | **KARC** — `KarcForecaster<D,M,K>::fit_direct / fit_woodbury` | Plan 308, Research 288, `riir-ai/crates/riir-games-civ/src/civ/map_tick/karc.rs` |
| Closed-form ridge over inter-view covariances `P* = Σ(N+λI)^{-1}` | **PEIRA** — `predictor_with_scratch` | Plan 153, `crates/katgpt-core/src/peira.rs` |
| Closed-form Tikhonov `(1-α)K̃ᵀK̃ + αI_d` spectral transport | **FuncAttn** — `solve_convex_combo_dual` | Plan 286, Research 257, `crates/katgpt-core/src/funcattn/mod.rs` |
| Per-entity FIXED MoE blend (sigmoid projection) | **CommittedFieldBlend** — `π_k = sigmoid(g_k(s)/τ)` computed once from trajectory summary, frozen | Plan 321, Research 302, `crates/katgpt-core/src/committed_field_blend.rs` |
| Per-layer sigmoid composition with per-tick DRIFT | **PersonalityWeightedComposition** | Plan 297, Research 276 |
| Asymmetric basis projection (cross-resolution transport) | **CrossResolutionTransport** | Plan 310, Research 291, `crates/katgpt-core/src/cross_resolution.rs` |
| Pre-trained velocity-field drafter (Fourier-mode token drafting) | **LinOSS ModalSpecDrafter** | Plan 189, `crates/katgpt-core/src/linoss.rs:561` |
| BLAKE3-committed frozen artifact pool + atomic Arc-swap | **LoRAHotSwap**, **EmotionDirections loader**, **MerkleFrozenEnvelope** | riir-neuron-db/src/freeze.rs, riir-ai/crates/riir-engine/src/snapshot.rs |
| Raw scalar list crossing sync boundary via fixed-point | **LatCal** 2×2-block commitment | riir-chain/src/encoding/latcal.rs |

### 2.2 What the paper adds that none of the above does alone

The fusion is the novelty, not any single component:

1. **P pre-trained NEURAL velocity fields as the basis** — KARC's basis is delay-embedded + spectral (Fourier/Walsh/Haar/Chebyshev). FuncAttn's basis is a sigmoid-normalized partition of one input. CommittedFieldBlend's basis is K learned *direction vectors*. **None of them treats K pre-trained forward models as the regression basis.** This is the gap: "the basis functions ARE other people's models".

2. **Combination weights SOLVED via P×P linear system** — CommittedFieldBlend's `π` comes from `sigmoid(g_k(s)/τ)` projection onto direction vectors (a *voting* operation). PersonalityWeightedComposition's `w` *drifts per-tick* from reward surprise. **Neither computes the LEAST-SQUARES-OPTIMAL combination weights `η = K^{-1} r` from data pairs.** The paper's weights are regression-optimal for the target distribution; the alternatives are heuristic or evolved.

3. **Cross-domain / cross-architecture composition without retraining** — KARC, PEIRA, FuncAttn all combine features of the SAME underlying model. CommittedFieldBlend's archetype library is trained offline on the SAME domain. **The paper's killer demo (Appendix E) composes models trained on DIFFERENT domains** (MNIST + F-MNIST + EMNIST + K-MNIST) into a strictly better MNIST generator. **No shipped primitive does this** — every shipped combiner assumes common architecture or common domain.

4. **Heterogeneous-output velocity fields** — The paper's `b_i(x) ∈ R^d` are all same-d, but combined with the Cross-Resolution asymmetric extension (Research 291), the velocity fields can have **different output dimensions** `d_i`. This fusion produces a velocity-field ensemble that spans heterogeneous model classes (a 7B LLM drafter, a 1B LLM drafter, an HLA forecaster, a LinOSS drafter) all combined into one algebraic super-drafter.

5. **Optimal diffusion schedule `D*_t`** — None of our UQ primitives (BoMSampler R281/Plan 281, Sleep-Time Anticipator R318/Plan 334, KARC+conformal-overlay R308+340, Conformal Seasonal Pools R322/Plan 340) derives the diffusion schedule from a Girsanov path-KL bound. **This is the first primitive in the corpus with a principled "how much noise to inject at time t" derived from a KL-divergence minimization.** It is UQ-bearing → subject to the §"Report the Floor" rule (Issue 010).

### 2.3 Fusion (the Super-GOAT move)

| Fusion partner | What it ships | What paper adds | Fusion product |
|---|---|---|---|
| **R288 KARC** | Per-NPC delay-basis ridge forecaster | P **pre-trained velocity fields** as an alternative basis (replace delay-embedding with model-pool-embedding); same ridge solve | "KARC where the basis is *other models* — combine P frozen forecasters into one optimal per-target forecaster" |
| **R302 CommittedFieldBlend (FAME)** | Per-entity FIXED MoE blend with sigmoid projection `π` | **Regression-optimal** `η = K^{-1} r` (not sigmoid projection); the *alternative weight-derivation rule* | "Two weight modes for CommittedFieldBlend: (a) sigmoid projection `π` for fast commitment, (b) ridge-solved `η` for data-pair-optimal commitment — switchable per use case" |
| **R291 Cross-Resolution Spectral Transport** | Asymmetric `Φ_src / Ψ_dst` basis projection (train-small-deploy-large) | Heterogeneous-d `b_i(x)` (each velocity field its own output dim) → project via Cross-Resolution first, then ensemble-combine | "Ensemble across **model classes** (LLM drafter + HLA forecaster + LinOSS drafter), each at its native d, projected to common d, then ridge-combined" |
| **R257 FuncAttn** | Closed-form Tikhonov spectral transport | The velocity-field pool AS the FuncAttn basis (replace Φ partition with K model outputs) | "FuncAttn where each basis function is a frozen model's forward pass" |
| **R115 PEIRA** | Zero-alloc scratch + EMA tracking for incremental ridge solves | Streaming / online ensemble refit (new data pairs accumulate into K, r without re-solving from scratch) | "Online ensemble: as new NPC trajectory accumulates, η updates incrementally via PEIRA's EMA machinery — no full re-solve" |
| **R276 PersonalityWeightedComposition** | Per-layer sigmoid-gated composition with drift | The ridge-solved `η` as an alternative *commitment mode* for the composition weights | "Per-NPC personality weights solved from data, not drifted — adds a `WeightDerivation::RidgeSolve` mode" |
| **freeze/thaw runtime (riir-ai)** | Atomic Arc-swap of frozen adapter pool | The velocity-field pool IS the freeze/thaw pool; η commits per-target as K floats | "Per-target ensemble weights frozen as a `VelocityFieldEnsembleShard`, atomic-swap when target distribution shifts" |
| **LatCal commitment (riir-chain)** | 2×2-block fixed-point linear-op commitment | The K solved weights as raw fixed-point scalars crossing sync | "Two nodes agree bit-for-bit on the ensemble weights for a given target — quorum-reproducible ensemble" |
| **DEC Stokes (R219/R296)** | `belief_mass_divergence`, `boundary_flux_mass` | The combined drift `b̂_t`'s divergence can be inspected via DEC operators | "Ensemble drift that's mass-conservation-validatable — `belief_mass_divergence(b̂) ≈ 0` is a sanity check on the combination" |
| **BoMSampler / KARC+conformal (UQ)** | K-hypothesis belief sampling, conformal intervals | Optimal diffusion `D*_t` from Girsanov path-KL → the principled noise schedule for ensemble sampling | "Ensemble sampler with principled noise schedule + conformal coverage — passes the §Report-the-Floor gate" |

### 2.4 Latent-space reframing (mandatory per fusion protocol §1.3)

Operating on each Super-GOAT factory module:

(a) **HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): The velocity-field ensemble's basis is P **per-NPC HLA evolution kernels** (each NPC's `evolve_hla` instance after freeze/thaw divergence). The ensemble drift `b̂_t(hla_state)` is the *combined* HLA-update direction, ridge-optimal for the NPC's recent trajectory. Replaces "one global `evolve_hla`" with "per-NPC ridge-optimal blend of P archetype HLA kernels".

(b) **latent_functor** (`riir-engine/src/latent_functor/`): The functor's direction vector `f` is generalized from rank-1 (single direction) to rank-P (P frozen functor instances combined via ridge solve). `extract_functor` becomes `fit_velocity_field_ensemble(trajectory)`; `apply_functor` becomes `apply_ensemble_functor(η, source_state)`. The `ReestimationScheduler` is the natural trigger for η-refit.

(c) **cgsp_runtime curiosity** (`riir-engine/src/cgsp_runtime/`): Curiosity becomes `curiosity_t = ‖actual_hla_t − b̂_t(hla_{t-1})‖` — surprise against the **ensemble-optimal forecast**, not a single forecaster. Crowd-scale: each NPC's ensemble is its own; quorum-reproducible because η is solved from raw trajectory pairs (deterministic).

(d) **LatCal fixed-point commitment** (`riir-chain/src/encoding/`): The K solved weights `η ∈ R^P` (P=3–8 typical) cross the sync boundary as K LatCal-committed fixed-point scalars. **Never commit the velocity field definitions** — those are library artifacts referenced by shard hash. The sync artifact is exactly K floats per (NPC, target-distribution) pair.

(e) **NeuronShard / freeze envelope** (`riir-neuron-db/src/`): `VelocityFieldEnsembleShard` subtype. Layout: `[zone_hash(32) | η_flat(P·4) | field_library_hash(32) | schedule_hash(16) | version(4) | blake3(32) | merkle_root(32)]` ≈ 116 + 4P bytes, P=8 → 148 bytes, padded to 192. `MerkleFrozenEnvelope` wraps it for atomic hot-swap. **The velocity-field library itself is a separate frozen artifact** (P NeuronShards), referenced by hash — not duplicated per ensemble.

(f) **DEC Stokes-calculus** (`katgpt-dec/src/`): The combined drift `b̂_t` is a vector field on the latent manifold. `codifferential(b̂_t)` measures its divergence; `belief_mass_divergence(b̂_t) ≈ 0` is a modelless sanity check that the ensemble preserves belief mass. The Girsanov-derived `D*_t` is the *raw scalar* that crosses sync to keep the ensemble's stochastic transport mass-conserving. **Curse-of-dimensionality caveat (R296):** DEC ops win only for d ≤ 3; HLA (d=8) and shards (d=64) do NOT benefit from boundary-only computation. The DEC mapping is for the *mass-conservation property*, not perf.

---

## 3. §3.5 Modelless Unblock Protocol (MANDATORY — passed)

Before any riir-train deferral, exhaust the three modelless paths:

**Path 1 (freeze/thaw snapshot correction):** **PASS.** The P velocity fields `{b_i}` are frozen snapshot artifacts (pre-trained offline once in riir-train, then frozen forever — the canonical freeze/thaw substrate). The per-target weights `η` are computed once per target distribution and frozen. Both are freeze/thaw artifacts — `MerkleFrozenEnvelope` wraps the ensemble, the field library is a frozen shard set. No runtime weight mutation.

**Path 2 (raw/lora reader-writer hot-swap):** **PASS (with caveat).** If the velocity fields are LoRA pairs `{reader_i, writer_i}`, the combined LoRA `L_η = Σ_i η_i · L_i` is a **deterministic linear combination in LoRA space** (LoRA pairs form a vector space — sum of low-rank matrices is low-rank up to rank-P·r). Constructing `L_η` from P frozen LoRA pairs + P ridge-solved weights is modelless (weight addition, no backprop). Caveat: this loses the per-time-grid structure (η depends on t); the runtime form is "η for the *current* target distribution, applied as a merged LoRA".

**Path 3 (latent-space correction):** **PASS.** The ridge solve `η = K^{-1} r` IS a latent-space operation: K and r are computed from velocity-field *outputs* (latent vectors), and η is a low-dim latent direction. No gradient descent. No backprop through base weights. The optimization is closed-form, in latent space, applied as a linear combination of frozen latents.

**Decision protocol result:** All three paths pass → **MODELLESS-VALIDABLE.** The primitive ships in katgpt-rs without any riir-train dependency for the per-target weight computation. The P velocity fields themselves are pre-trained offline (riir-train's job, once, for the library) — but that is the freeze/thaw substrate, not a per-target training dependency. **No riir-train deferral.**

---

## 4. Verdict

### Tier: **Super-GOAT — via fusion**

| Q | Answer | Evidence |
|---|--------|----------|
| **Q1: No prior art?** | **YES (for the combination)** | The fusion — P pre-trained velocity fields as the regression basis × ridge-solved combination weights × cross-domain/cross-architecture composition × freeze/thaw library × LatCal commitment — has zero shipped prior art. Each *component* has a cousin (KARC ridge solve, CommittedFieldBlend sigmoid projection, Cross-Resolution asymmetric basis, LoRAHotSwap frozen pool, LatCal commit), but **no shipped primitive combines P frozen forward models into a regression-optimal ensemble solved from data pairs**. Vocabulary check passed: grep on paper terms (`stochastic interpolant`, `velocity field`, `kernel method`, `flow matching`, `feature gradient`) AND codebase terms (`ridge solve`, `frozen.*adapter`, `LoRA pool`, `model ensemble`, `linear system`, `weighted composition`) was performed across all 5 repos at both `.md` and `.rs` layers. |
| **Q2: New capability class?** | **YES** | "Algebraic ensemble of P frozen heterogeneous pre-trained models, with regression-optimal combination weights derived from data pairs, no architecture constraint, no retraining" is a new capability class. No current primitive does this — KARC ridge-solves delay features (not model outputs); CommittedFieldBlend sigmoid-projects onto direction vectors (not regression-solved); Cross-Resolution transports between two bases (not combines P); LoRAHotSwap swaps one adapter at a time (not solves optimal combination). |
| **Q3: Product selling point?** | **YES** | "Combine pre-trained NPCs/personalities/forecasters from any source — different training stages, different games, different architectures, even different domains — into one optimal behavior via a single linear-system solve. No fine-tuning, no online drift, regression-optimal for THIS target distribution." Concrete, demoable (the paper's cross-domain MNIST composition is the killer demo — F-MNIST+EMNIST+K-MNIST+MNIST ensemble beats MNIST-only), hard to replicate without our full stack. |
| **Q4: Force multiplier?** | **YES (≥7 systems)** | Connects: KARC (ridge solve + per-NPC forecaster), CommittedFieldBlend (commit-once pattern + archetype library), Cross-Resolution Spectral Transport (heterogeneous-d velocity fields), PEIRA (zero-alloc scratch + EMA incremental refit), FuncAttn (alternative basis interpretation), PersonalityWeightedComposition (commitment mode extension), LatCal (sync-boundary commitment), DEC Stokes (mass-conservation sanity), UQ primitives (`D*_t` schedule). |

**Mandatory outputs (this session):**
1. **Open primitive** → `katgpt-rs/.plans/376_velocity_field_ensemble_primitive.md` (generic math, no game IP — `VelocityFieldEnsemble<P, D>` + `VelocityField` trait + ridge solve reuse + zero-alloc scratch).
2. **Private guide** → `riir-ai/.research/170_per_npc_velocity_field_ensemble_composition_guide.md` (selling point: per-NPC algebraic ensemble composition + cross-game personality transfer + committed ensemble shards — game runtime is the dominant pillar).
3. **Cross-ref guides** (deferred — file after Plan 376 GOAT gate passes): `riir-neuron-db/.research/013_velocity_field_ensemble_shard_crossref.md` (freeze substrate), `riir-chain/.research/008_latcal_committed_ensemble_weights.md` (sync-boundary bridge).
4. **Private plan** (deferred — file after Plan 376 GOAT gate passes): `riir-ai/.plans/385_velocity_field_ensemble_runtime_integration.md` (runtime wiring: HLA hook, latent_functor interop, field-library loader, KarcShard/ArchetypeBlendShard freeze integration).

**One-line reasoning:** The paper's value is not the stochastic-interpolant framework (which is field-specific) and not the Girsanov `D*_t` derivation (which is theoretical scaffolding); it is the *primitive* that **the basis functions of a ridge solve can be other pre-trained models' forward passes**, with weights derived as `η = K^{-1} r` from data pairs — making ensemble combination a closed-form algebraic operation, valid across architectures and domains. That primitive, fused with our KARC ridge machinery + CommittedFieldBlend commit-once pattern + Cross-Resolution heterogeneous-d projection + freeze/thaw library + LatCal commitment, is the Super-GOAT.

---

## 5. Caveats and known risks

1. **KARC overlap is real and IS the foundation.** The closed-form `(K + λI)^{-1} r` is **identical** to KARC's `fit_direct` math. The contribution is the *basis construction* — P velocity-field outputs as features, not delay-embedded basis-expanded observations. **Do NOT re-ship the ridge solve — reuse `linalg::ridge_solve::{ridge_solve_direct_f32, ridge_solve_woodbury_f32}` directly.** Anyone reviewing this verdict should grep `ridge_solve_direct_f32|ridge_solve_woodbury_f32` and confirm.

2. **CommittedFieldBlend overlap is real and ORTHOGONAL.** CommittedFieldBlend computes `π` via sigmoid projection `π_k = sigmoid(g_k(s)/τ)` (a *voting* operation); the paper computes `η` via least-squares regression `η = K^{-1} r` (an *optimality* operation). The two are **alternative weight-derivation modes for the same commit-once pattern** — both produce a frozen K-weight vector for the NPC's lifetime. Plan 376 MUST position ridge-solve as a *mode* of `CommittedFieldBlend`, not a competing primitive. Default mode = sigmoid projection (cheap, no data-pair requirement); opt-in mode = ridge solve (requires N data pairs, regression-optimal).

3. **The optimal `D*_t` requires a Girsanov-style argument that doesn't trivially map to discrete-tick game AI.** The schedule `D*_t = α_t γ_t / β_t` is derived for continuous-time interpolants `t ∈ [0,1]`. Mapping to a 20Hz game tick requires choosing an interpolant schedule per "generation episode" — non-trivial. **Mitigation:** ship the integrator (eq. 14) as an open primitive in katgpt-rs; defer the per-game schedule tuning to riir-ai runtime integration (Plan 385). The `D*_t` derivation is a *transferable theoretical result*, not a directly-shipped runtime constant.

4. **UQ-bearing primitive → subject to the §"Report the Floor" rule (Issue 010).** The ensemble claims a probability distribution (the generated density `ρ_{D_t}`). Per the rule adopted 2026-06-28, the GOAT gate MUST benchmark against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340, m=1) on CRPS / coverage / Winkler score. If the ensemble cannot beat the conformal-naive floor on a UQ benchmark, the GOAT gate FAILS on the UQ axis (the primitive can still ship as a non-UQ algebraic combiner, but the UQ claim is dropped). The floor is now enforceable.

5. **Cross-domain composition is empirically demonstrated for IMAGE GENERATION, not for game AI.** The paper's cross-domain result (F-MNIST+EMNIST+K-MNIST+MNIST) is on U-Nets generating images. Whether velocity fields from *different games* combine into a better target-game NPC is **unproven** and may require a defend-wrong PoC (§3.6) before any quality-parity claim. **Mitigation:** Plan 376 GOAT gate G2 measures cross-domain ensemble quality on a controlled toy benchmark (bomber/go/monopoly primitives). Architectural coverage is sufficient for the open-primitive ship; quality parity needs the PoC.

6. **Per-NPC memory cost.** A `VelocityFieldEnsembleShard` of 192 bytes × 10,000 NPCs = 1.92 MB. Fits comfortably in Warm tier. Not a constraint.

7. **Closed-form solve cost.** P×P Cholesky for P=8 is ~512 ops, sub-µs on SIMD. Accumulating K (P×P) and r (P) over N=50 data pairs is ~50·P² = 3,200 ops, also sub-µs. The full fit is well under 5µs — fits comfortably in plasma tier (µs budget) for the per-NPC regime.

8. **Numerical conditioning of the velocity-field Gram.** If P velocity fields are highly correlated (e.g., same architecture, similar training), `K_t` is ill-conditioned. **Mitigation:** ridge regularization `λI` (already in the solve); diagnostic via condition number of `K_t` logged at fit time. If `cond(K_t) > 1e8`, log a warning and increase `λ`.

9. **The "feature gradient" interpretation is a re-description.** The paper motivates `b_i(x)` as `∇φ_i(x)` (gradient of some feature map). For our purposes, `b_i(x)` is just a frozen model's forward output — we don't need to construct `φ_i`. The ridge solve works identically on raw `b_i(x)` outputs. **Document this** so future readers don't go looking for a `φ` to construct.

---

## 6. Next steps (see Plan 376)

**Phase 1 (open primitive, katgpt-rs):** ship `VelocityFieldEnsemble<P, D>` + `VelocityField` trait (sealed, with a blanket impl for any `Fn(&[f32], &mut [f32])`) in `crates/katgpt-core/src/velocity_field_ensemble.rs` behind `velocity_field_ensemble` feature. Reuse `linalg::ridge_solve::ridge_solve_direct_f32` for the P×P solve; reuse the `FuncAttnScratch` pattern for zero-alloc scratch. Zero game IP. GOAT gate G1–G4 on:
- G1 (mechanics): synthetic — 3 fixed linear velocity fields, known optimal `η`, verify solve recovers it bit-for-bit.
- G2 (cross-domain): bomber-go-monopoly primitive drafter ensemble — does the ridge-combined drafter beat any single drafter on a held-out target game? Architectural coverage is sufficient; quality parity is the open question.
- G3 (no-regression): `--features velocity_field_ensemble` adds zero warnings, zero new allocations on the no-op path.
- G4 (latency): full fit + 1000 combined-drift evals ≤ 100µs on SIMD.

**Phase 2–4 (private runtime, riir-ai Plan 385, deferred):** runtime wiring — HLA hook (each NPC's recent trajectory → ensemble fit → combined HLA-update direction), latent_functor interop (ridge-solve as `WeightDerivation::RidgeSolve` mode), field-library loader (P archetype shards frozen + referenced by hash), `VelocityFieldEnsembleShard` freeze integration.

**Phase 5 (chain commitment, riir-chain, deferred):** LatCal commit the K solved weights as K fixed-point scalars crossing sync. Two nodes agree bit-for-bit on the ensemble for a given target.

**Phase 6 (UQ floor, deferred):** benchmark ensemble+`D*_t` against conformal-naive floor (Plan 340, m=1) on CRPS / coverage / Winkler. Per Issue 010, this is mandatory before any UQ claim.

---

## TL;DR (one-line)

Kernelized Stochastic Interpolants = **the basis functions of a ridge solve can be other pre-trained models' forward passes**, with weights `η = K^{-1} r` derived from data pairs (not trained, not evolved, not projected) — making ensemble combination a closed-form algebraic operation valid across architectures and domains; the math pieces all ship (KARC ridge solve, CommittedFieldBlend commit-once pattern, Cross-Resolution heterogeneous-d, LoRAHotSwap frozen pool, LatCal commit); the Super-GOAT is the *combination* as the first per-NPC algebraic ensemble composition that combines heterogeneous frozen forecasters into a regression-optimal super-forecaster, fits in a `VelocityFieldEnsembleShard`, and crosses the LatCal sync boundary as K=8 floats.
