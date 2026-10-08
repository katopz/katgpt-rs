# Research 288: Kolmogorov-Arnold Reservoir Computing (KARC) — Delay-Basis-Ridge Forecaster

> **Source:** [Kolmogorov-Arnold Reservoir Computing](https://arxiv.org/pdf/2606.19984) — Huang, Kurths, Tang (UESTC / PIK Potsdam / Humboldt / Fudan), arXiv:2606.19984v1, 2026-06-18
> **Date:** 2026-06-22
> **Status:** Active — Super-GOAT via fusion; primitive + plan + private guide created this session
> **Related Research:** 095 (LinOSS oscillatory SSM), 100 (EGA spectral salience), 153→003-doc (PEIRA closed-form ridge), 212 (Gemini Fourier × LatCal), 230 (Semiseparable SSD), 242 (topological recurrent belief), 246 (manifold power-iter MoE router), 257 (Functional Attention closed-form Tikhonov), 265 (FP-MGM weight-shared loop), 276 (MicroRecurrentBeliefState — HLA substrate), 281 (Per-Tick Salience Tri-Gate)
> **Cross-ref (riir-ai):** Research 152 (Per-NPC Delay-Basis Ridge Forecaster Guide — the selling point)
> **Cross-ref (riir-chain):** Research 003 (LatCal-Committed KarcShard Readout — sync-boundary bridge)
> **Cross-ref (riir-neuron-db):** Research 003 (KarcShard Storage Crossref — freeze substrate)
> **Related Plans:** katgpt-rs 308 (this research's open primitive)
> **Classification:** Public (katgpt-rs = open math primitive); the *selling-point guide* is private in riir-ai.
> **Verdict: Super-GOAT — the fusion (delay-embedding × basis expansion × closed-form ridge × per-NPC HLA trajectory forecaster × LatCal commitment × KarcShard freeze) is a new capability class with no shipped prior art for the COMBINATION.**
> **PASS-Redirects (synthesis):** "World Model Anomaly Detection with a Latent Linear Prior" [TMLR openreview VLIzLK3CfR, code: anonymous.4open.science/r/lp-dreamer] — LP-DreamerV3's "linear prior A on successive latent states" is exactly the KARC forecast (this note's `Wout·φ(z_t)` linear model). The paper's residual-based anomaly detector `‖z_{t+1}−A·z_t‖>τ` is already wired as `mcts_collapse_bridge` (`‖MCTS_rollout−KARC_forecast‖>τ`, Bench 152) + `ReestimationScheduler` coherence<tau_reest. PASS: KARC is the linear prior; the imagined-rollout anomaly detection ships; only the value-update-suppression-in-MBRL-training half is new (→ riir-train if we ever ship DreamerV3-style value backups from imagination).
> **PASS-Redirects (synthesis):** Heightman, Orlova, Mantrov, Ustimenko [arXiv:2608.11911 "Hamilton-Zero: A Neural Tensor-Network Foundation Model for Ground States of Arbitrary Quadratic Qubit Hamiltonians"] — the fidelity-susceptibility phase-transition witness (Fig 12: χ_F/N peaks at g_c=1 zero-shot from ONE frozen checkpoint, no retraining) is the same capability class as KARC's surprise spike on regime change (`karc_surprise_spikes_on_regime_change`, Bench 332 G2 — spike within 1 tick of the regime boundary, then decay); and the zero-variance calibration (relative gap ≈ q/(1+q), q = κV with κ=0.2825, held-out size transfer N≤18 → 19–22) is the physics twin of our conformal-UQ "report the floor" overlay (Research 322 / Plan 340 — variance-derived point estimate of error, calibrated on a held-out regime). The VMC machinery (SU(2)^N manifold ansatz, Lie-derivative Hamiltonians, KFAC-3F stacked-weight natural gradient, replica-exchange Langevin on quaternions) has no consumer in the stack → PASS.
> **PASS-Redirects (synthesis):** Hill, Xinyu, Prasetio [arXiv:2508.02995 "The Geometry of Cortical Computation: Manifold Disentanglement and Predictive Dynamics in VCNet"] — top-down predictive coding ε = ReLU(V1_bu − AIT_td) is a rectified forecast residual: same mechanism as KARC surprise (riir-ai `karc_bridge.rs`) + `curiosity_bridge.rs` ‖dz‖₂ + `offline_scored_wake.rs` surprise_norm, shipped modelless; its learnable channel "neuromodulatory" gain is covered by riir-ai `mop_homeostasis` (Plan 581). PASS.

---

## TL;DR

KARC replaces a recurrent reservoir with **delay-coordinate univariate basis expansion + closed-form ridge regression readout**. Given a trajectory `{u_i}`, it forms delay states `x_i = u_i ⊕ u_{i-1} ⊕ ... ⊕ u_{i-k+1}`, projects each coordinate onto m basis functions (Fourier, Chebyshev, B-spline), and fits `Wout = YH^T(HH^T + λI)^{-1}` in closed form. The paper proves this is a lightweight Kolmogorov-Arnold realization; we don't care about the KAN interpretation — we care that it is **the first per-NPC forecaster primitive that is (a) closed-form (no backprop), (b) bit-reproducible from raw trajectory alone, (c) frozen into a fixed-size shard, (d) committable through LatCal as a linear readout.**

**Distilled for katgpt-rs (modelless, inference-time):**
A generic `KarcForecaster<D, M, K>` struct: delay-embedding window of length `K` over a `D`-dim state, basis-expanded to `M` features per coordinate, fit by ridge regression `Wout ∈ R^{D×(K·D·M)}` with Woodbury + chunked Gram + optional low-rank factorization for high-D settings. The basis is a sealed trait `KarcBasis` with three shipped instances (`Fourier`, `Chebyshev`, `BSpline`) — already half-shipped in `riir-ai/crates/riir-engine/src/linoss/basis.rs`. The ridge fit reuses `crates/katgpt-core/src/peira.rs`'s closed-form `(N + λI)^{-1}` machinery. The "novelty" is the **combination** — the closest cousin (latent_functor) learns a *single direction vector per relation*; KARC learns a *full readout matrix over basis-expanded delay-embedded coordinates*, generalizing latent_functor from `rank-1` to `rank-m × delay-k`.

---

## 1. Paper Core Findings (verified by full PDF read)

### 1.1 The primitive
- **Delay embedding** (Eq. 10): `x_i = u_i ⊕ u_{i-1} ⊕ ... ⊕ u_{i-k+1}` — flat concat of last-k observations, length `n = k·d`.
- **Basis expansion** (Eq. 8): per-coordinate projection `Ψ(x) = [ψ_1(x_1), ..., ψ_m(x_1), ψ_1(x_2), ..., ψ_m(x_n)]^T`, length `n·m`.
- **Closed-form readout** (Eq. 14): `Wout = YH^T(HH^T + λI)^{-1}` where `H ∈ R^{(n·m)×N}` is the feature matrix, `Y ∈ R^{d×N}` the targets.
- **One-step forecast**: `û_{i+1} = Wout · Ψ(x_i)`. Autonomous rollout by feeding `û_{i+1}` back as the next observation.
- **Higher-order** (Eq. 32): outer products `∏_ℓ ψ_{j_ℓ}(x_{p_ℓ})` up to order `R`. Feature dim `D_R = Σ_{r=1}^R C(nm+r-1, r)`.

### 1.2 The basis dictionary (Methods §B)
- **Fourier**: `ψ_{2i-1}(x) = cos(2π i x/P)`, `ψ_{2i}(x) = sin(2π i x/P)` — `m = 2Q`. Spectral norm bound `‖Ψ(x)‖₂ ≤ √(nm/2)`.
- **Chebyshev**: `T_0(x)=1, T_1(x)=x, T_{n+1}(x)=2xT_n(x)−T_{n-1}(x)` — `|T_j(x)| ≤ 1` on `[-1,1]`. Bound `√(nm)`.
- **B-spline**: Cox-de Boor recursion, partition-of-unity → `‖Ψ(x)‖₂ ≤ √n`.

### 1.3 Memory-optimized training (Methods §C — important for crowd-scale)
- **Woodbury identity**: swap `(HH^T + λI)^{-1}` (`d_h × d_h`) for `(H^T H + λI)^{-1}` (`N × N`) when `d_h > N`. Big win when feature dim ≫ sample count (the per-NPC setting: `n·m = 256` features, `N = 50` ticks observed).
- **Chunked Gram**: `H^T H = Σ_i H_i^T H_i` accumulated block-by-block — never materialize full `H`.
- **Low-rank factorization** (Eq. 46): `Wout ≈ AB`, `A ∈ R^{d×d_l}`, `B ∈ R^{d_l×d_h}`, `d_l ≪ min(d, d_h)`. Alternating least squares. **This is the variant that fits in a `NeuronShard`** — `d_l` slots of `style_weights[64]`.

### 1.4 Error bound (Appendix D — Eq. 103)
`e_tot ≤ L_F√d·w^k + ε_Ψ(k)·(1 + N·B_Ψ^2/(σ_min(H)^2+λ)) + λ·B_W·B_Ψ/(σ_min(H)^2+λ)`
Four terms: time-delay truncation `L_F√d·w^k` (geometric in delay length `k`), basis-dictionary residual `ε_Ψ(k)`, ridge regularization bias, and Gram conditioning. Not directly actionable, but informs hyperparameter tradeoffs (larger `k` reduces truncation but enlarges feature space).

### 1.5 Headline empirical wins
- **Double-scroll** (3-dim chaotic ODE): KARC NRMSE `5.3×10^{-4}` vs RC `2.1×10^{-2}` (40× lower), threshold time `16.7 LT` vs `10.7 LT` (1.56× longer horizon), **train time 0.12s** (RC: 0.36s, NG-RC: 0.20s).
- **Kuramoto-Sivashinsky** (`L=22`, 64 grid): KARC threshold `11.9 LT` vs RC `2.4 LT` (~5× longer), train time `0.79s` vs VolterraRC `978s` (CPU).
- **Shallow water** (2D, 64×64 grid): KARC threshold 40 steps vs VolterraRC 26, RC 6; relative mass error slowest growth.
- **FLUX diffusion sampling acceleration** (§4): KARC-Fourier and KARC-B-spline match Spectrum (Chebyshev) within 1 PSNR point at 4.6× speedup — basis choice is modular.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface — verify before any novelty claim)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| Closed-form ridge readout `Wout = YH^T(HH^T+λI)^{-1}` | **PEIRA** — `P* = Σ(N + λI)^{-1}` zero-alloc | Plan 153, `crates/katgpt-core/src/peira.rs` (lines 875–917 — `predictor_with_scratch`, `predict_and_loss`) |
| Closed-form Tikhonov solve `(1-α)·K̃ᵀK̃ + α·I_d` | **FuncAttn** regression operator | Plan 286, Research 257, `crates/katgpt-core/src/funcattn/mod.rs` |
| Fourier / Walsh / Haar basis dictionary with `SpectralBasis` trait | **riir-ai/crates/riir-engine/src/linoss/basis.rs** — exact same family | `riir-ai/crates/riir-engine/src/linoss/basis.rs` — `FourierBasis`, `WalshBasis`, `HaarBasis` |
| Spectral eigenbasis projection (`V^T @ x`) | **TurboQuant SpectralRotation** | Plan 077, `crates/katgpt-spectral/src/spectral_rotation.rs` |
| Oscillatory SSM forecaster + Fourier-mode drafter | **LinOSS / ModalSpec** — drafter reconstructs via Fourier coeffs | Plan 189, `crates/katgpt-core/src/linoss.rs:561` (`ModalSpecDrafter`) |
| Per-NPC learned direction vector from observation pairs, with coherence gate + re-estimation | **latent_functor** — `extract_functor`, `predict_stance`, `ReestimationScheduler` | Plan 303, `riir-engine/src/latent_functor/` (especially `arithmetic.rs`, `reestimation.rs`) |
| Per-NPC recurrent belief kernel (attractor + leaky families) | **MicroRecurrentBeliefState** | Plan 276, Research 242, `crates/katgpt-core/src/micro_belief/` |
| HLA recurrent update (leaky integrator step) | **`evolve_hla`** | `crates/katgpt-sense/src/reconstruction.rs`, `leaky_core.rs` |
| Crowdscale curiosity signal from coherence decay + JS-uniqueness | **cgsp_runtime** curiosity boosts | `riir-engine/src/cgsp_runtime/`, `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` |
| MCTS collapse detection baseline (cheap predictor) | **mcts_collapse_bridge** | `riir-ai/crates/riir-engine/src/cgsp_runtime/mcts_collapse_bridge.rs` |
| Per-shard frozen latent state (Pod, BLAKE3, dendritic branch) | **NeuronShard** | `riir-neuron-db/src/shard/mod.rs` (`style_weights[64]`, `hla_moments[8]`) |
| Deterministic linear-op commitment + 2×2 matrix arithmetic | **LatCal** | `riir-chain/src/encoding/latcal.rs` (`LatCalMatrix`, `multiply`, `to_fixed`) |
| Spectral commitment of Fourier coefficients | **LatCal Fixed-Point Fourier Coefficients** | Plan 265 (riir-ai) |
| Forensic fingerprint of a fixed-point committed blob | **Forensic Watermark** | Plan 322 (riir-ai), `riir-chain/src/forensic/` |

### 2.2 What KARC adds that none of the above does alone

The fusion is the novelty, not any single component:

1. **Delay-embedding** — `x_i = u_i ⊕ u_{i-1} ⊕ ... ⊕ u_{i-k+1}` is NOT shipped anywhere in the corpus (grep for `delay_len|n_delays|history_window|k_delay|delay_state` returns zero inference-time hits — only `simulate_network_delay` test helpers in an unrelated project). LinOSS uses recurrence (`LinOSSState{y,z}` carried forward); latent_functor uses single-pair regression (`(source, target)` → `f`); neither concatenates the last-k observations as a feature vector.

2. **Closed-form ridge over basis-expanded delay features** — PEIRA does closed-form ridge but on **inter-view covariances**, not delay-embedded features. FuncAttn does closed-form Tikhonov but on **spectral kernel matrices**, not delay-embedded features. latent_functor does regression but **single direction per relation** (rank-1), not a `D × (K·D·M)` readout matrix (rank-m × delay-k).

3. **Bit-reproducible learned forecaster** — the `Wout` of KARC is **fully determined** by `(basis_config, k, λ, trajectory_bytes)`. Two nodes with the same trajectory produce identical `Wout` bit-for-bit. latent_functor commitments are hashes of `f` vectors (the *result*), not the *procedure*; PEIRA commitments are hashes of `(P*, Q*)`. KARC's commitment is **the trajectory itself** — the readout is reproducible. This is the substrate for quorum-agreed "this NPC behaved surprisingly at tick T" without a model in the loop.

4. **Low-rank readout fits in a shard** — §C of the paper: `Wout ≈ AB` with `A ∈ R^{D×d_l}`, `B ∈ R^{d_l×d_h}`. For HLA `D=8`, `d_h=256`, `d_l=8` gives `A` as 8×8 = 64 floats = exactly `style_weights[64]`. **This is the natural `KarcShard` subtype of `NeuronShard`** — the existing Pod layout already has the slot.

### 2.3 Fusion (the Super-GOAT move)

| Fusion partner | What it ships | What KARC adds | Fusion product |
|---|---|---|---|
| **R276 MicroRecurrentBeliefState / HLA `evolve_hla`** | Per-NPC 8-dim HLA state with *fixed* leaky integrator update — no per-NPC learned dynamics | Closed-form ridge fit of next-HLA from delay-embedded basis-expanded HLA — **per-NPC learned dynamics** | "Every NPC has its own learned personality forecaster, fit from its own trajectory, replacing the one-size-fits-all leaky integrator." |
| **R303 latent_functor** | Per-(source,target) relation: single direction vector `f` + coherence-gated apply | Full readout matrix over basis-expanded delay coordinates (rank-`m × delay-k` generalization) | KARC = "rank-k latent_functor over time" — predict the **next full latent state**, not just the stance toward one relation |
| **R153 PEIRA** | Closed-form ridge `P* = Σ(N+λI)^{-1}` over inter-view covariances | Delay-embedding + basis expansion as the feature construction; the SAME ridge math | PEIRA's machinery (zero-alloc scratch, EMA tracking) directly reusable; KARC widens the feature space from raw `x` to `Ψ(delay_embedding(x))` |
| **R257 FuncAttn** | Closed-form Tikhonov `(1-α)K̃ᵀK̃ + αI_d` over a basis-partition Φ | The KARC forecaster as an alternative basis-partition regression operator with delay-aware features | KARC and FuncAttn share the regression-solve primitive; KARC's basis is temporal (delay), FuncAttn's is spatial (token partition) |
| **R095 LinOSS / riir-ai/crates/riir-engine/src/linoss/basis.rs** | `SpectralBasis` trait + `FourierBasis` / `WalshBasis` / `HaarBasis` + `ModalSpecDrafter` (Fourier-mode token drafting) | The basis functions are the *same*; KARC uses them for **latent-state forecasting**, LinOSS uses them for **token drafting** | KARC reuses the basis infrastructure verbatim — the `SpectralBasis` trait becomes the KARC basis trait with one extra method (`mode_count_per_dim`) |
| **R242 Topological Recurrent Belief / R276 MicroBelief** | Per-NPC belief kernel + BLAKE3 snapshot | Deterministic reproducible-from-trajectory forecaster; curiosity = `‖actual − forecast‖` | **Curiosity gets a deterministic baseline** — quorum-agreed surprise signal at the sync boundary, replacing model-in-the-loop curiosity divergence |
| **R212 Gemini Fourier × LatCal (Plan 242)** | LatCal commitment of Fourier-smoothed potential fields | A learned (not hand-tuned) Fourier-coefficient readout, fit per NPC | "Per-NPC learned Fourier personality, committed via LatCal, quorum-verifiable" |
| **R265 LatCal Fixed-Point Fourier Coefficients (Plan 265)** | Fixed-point commitment of Fourier coefficients | The coefficients are now *fit from trajectory* via ridge, not preset | LatCal commits the `Wout` of KARC as a 2×2-block linear-op chain |
| **NeuronShard / MerkleFrozenEnvelope** | Fixed-size Pod with `style_weights[64]`, BLAKE3 commitment, freeze/thaw envelope | A `KarcShard` subtype that stores the low-rank `AB` factorization of `Wout` | **Per-NPC personality frozen into a shard, replicated via chain, restorable on any node** |
| **R281 Per-Tick Salience Tri-Gate** | Per-tick autonomous emit decision from latent state | A predicted-next-HLA as additional context signal to the salience gate | "NPC decides whether to speak *now* partly based on its KARC-forecast next emotional state" |
| **mcts_collapse_bridge** | Cheap predictor baseline for MCTS collapse detection | KARC forecast is that cheap predictor — deterministic, closed-form | Crowd-scale collapse detection: `‖MCTS_rollout − KARC_forecast‖ > τ` triggers deeper search |

### 2.4 Latent-space reframing (mandatory per fusion protocol §1.3)

Operating on each Super-GOAT factory module:

(a) **HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): KARC's delay vector is the last-k HLA snapshots, basis expansion is the KARC feature map, ridge readout predicts next HLA. The forecast HLA can either *replace* `evolve_hla` (per-NPC learned update) or *augment* it (forecast as input to a gate that decides whether to invoke the expensive LLM/CoT path).

(b) **latent_functor** (`riir-engine/src/latent_functor/`): KARC = functor class with **closed-form ridge fit** of the operator matrix. `extract_functor` becomes `fit_karc_functor(trajectory)`; `apply_functor` becomes `karc_forecast(delay_state)`. The `ReestimationScheduler` already implements "drift-triggered re-fit"; KARC fits into this scheduler as a higher-rank alternative to the rank-1 displacement fit.

(c) **cgsp_runtime curiosity** (`riir-engine/src/cgsp_runtime/`): curiosity signal becomes `curiosity_t = ‖actual_hla_t − karc_forecast_t‖`. Reproducible across nodes → quorum-agreed "this NPC was surprised at tick T". This is the **first deterministic curiosity signal** in the corpus — existing curiosity is coherence-decay or JS-uniqueness, both of which need shared model state; KARC only needs shared raw trajectory.

(d) **LatCal fixed-point commitment** (`riir-chain/src/encoding/`): `Wout` is a linear op. LatCal commits linear ops over 2×2 fixed-point blocks. **A LatCal-committed KARC forecaster = deterministic, quorum-reproducible HLA forecast.** The forecast itself (a 5-scalar emotion projection) is the bridge artifact that crosses sync — never the full `Wout` matrix.

(e) **NeuronShard / freeze envelope** (`riir-neuron-db/src/`): `KarcShard` subtype. Layout: `[zone_hash(32) | A_flat(64) | B_row_pointers(8) | basis_config(8) | k(1) | lambda(1) | commitment(32) | merkle_root(32)]` ≈ 178 bytes, padded to 256. `MerkleFrozenEnvelope` wraps it for self-play freeze/thaw.

---

## 3. Verdict

### Tier: **Super-GOAT** — via fusion

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | **YES (for the combination)** | The fusion — delay-embedding × basis expansion × closed-form ridge × per-NPC HLA forecaster × LatCal commitment × KarcShard freeze — has zero shipped prior art. Each *component* has a cousin (PEIRA ridge, linoss basis, latent_functor per-NPC learn, MicroBelief snapshot, LatCal commit), but no shipped primitive combines them. Vocabulary check passed: grep on paper terms (`reservoir`, `KARC`, `NG-RC`, `Kolmogorov`) AND codebase terms (`ridge`, `closed-form`, `Fourier basis`, `evolve_hla`, `delay embedding`, `leaky integrator`, `functor`, `LatCal`, `fixed-point`) was performed across all 5 repos at both `.md` and `.rs` layers. |
| **Q2: New capability class?** | **YES** | "Per-NPC deterministic trajectory forecaster, frozen into a shard, quorum-reproducible from raw trajectory alone" is a new capability class. No current primitive does this — PEIRA doesn't forecast, latent_functor doesn't delay-embed, LinOSS doesn't fit per-NPC, evolve_hla doesn't learn. |
| **Q3: Product selling point?** | **YES** | "Every NPC has a unique personality encoded as a closed-form ridge-fit forecaster over its own trajectory, frozen via BLAKE3 into a KarcShard, quorum-verifiable across nodes. Crowd-scale curiosity and collapse detection without a learned world model — two nodes agree bit-for-bit that an NPC was surprised at tick T." |
| **Q4: Force multiplier?** | **YES (≥5 pillars)** | Connects: HLA (latent substrate), latent_functor (rank-k generalization), cgsp_runtime (curiosity + MCTS collapse), NeuronShard/freeze (persistence), LatCal (commitment bridge), linoss/basis (Fourier dictionary reuse), Per-Tick Salience (forecast as context). |

**Mandatory outputs (this session):**
1. **Open primitive** → `katgpt-rs/.plans/308_karc_delay_basis_ridge_forecaster.md` (generic math, no game IP).
2. **Private guide** → `riir-ai/.research/152_Per_NPC_Karc_Forecaster_Guide.md` (selling point: per-NPC personality forecasting + crowd curiosity + frozen shards — game runtime is the dominant pillar).
3. **Cross-ref guides** → `riir-neuron-db/.research/003_KarcShard_Storage_Crossref.md` (freeze substrate), `riir-chain/.research/003_LatCal_Committed_Karc_Readout.md` (sync-boundary bridge).
4. **Private plan** → `riir-ai/.plans/332_karc_runtime_npc_integration.md` (runtime wiring: HLA hook, latent_functor interop, curiosity bridge, KarcShard freeze).

**One-line reasoning:** KARC's value is not the KAN interpretation (which is re-description) and not the reservoir-computing framing (which is field-specific); it is the *combination* of delay-embedding + basis expansion + closed-form ridge as a **per-NPC reproducible forecaster** that fits in a shard and crosses the LatCal sync boundary as a 5-scalar emotion projection. That combination is the Super-GOAT.

---

## 4. Caveats and known risks

1. **PEIRA overlap is real.** Anyone reading this verdict should verify the ridge math in `peira.rs:875-917` and confirm that KARC's contribution is the *feature construction* (delay × basis), not the ridge solve. The closed-form `(N + λI)^{-1}` is identical. **Do not re-ship the ridge solve — reuse `predictor_with_scratch`.**
2. **latent_functor overlap is real.** latent_functor already learns per-NPC per-relation direction vectors with re-estimation. KARC is *strictly more general* (matrix readout vs vector direction), but the integration story must position KARC as a **rank-k temporal extension** of latent_functor, not a replacement. latent_functor handles relational stance; KARC handles temporal trajectory.
3. **LinOSS ModalSpec overlap is real.** LinOSS already does Fourier-basis forecasting (for token drafting). KARC's *temporal-delay* feature construction is the delta; the basis functions themselves are reusable. **Do not duplicate `riir-ai/crates/riir-engine/src/linoss/basis.rs`** — KARC should consume `SpectralBasis` directly.
4. **Per-NPC memory cost.** A `KarcShard` of 256 bytes × 10,000 NPCs = 2.5 MB. Fits comfortably inWarm tier. Not a constraint, but worth stating.
5. **Closed-form solve cost.** Woodbury reduces the per-fit cost to `O(N^2 · d_h + N^3)`. For `N=50, d_h=256`: ~3.3M ops, ~5µs on SIMD. Fits in Warm tier. Re-fit cadence: once per `tau_reest` ticks (same trigger as latent_functor's `ReestimationScheduler`).
6. **The error bound (Eq. 103) is qualitative, not actionable.** Treat it as design guidance (larger `k` reduces truncation but enlarges feature space; basis choice trades off feature bound `B_Ψ` vs residual `ε_Ψ(k)`); not as a runtime check.

## 5. Next steps (see Plan 308)

Phase 1: ship `KarcForecaster<D, M, K>` + `KarcBasis` trait (sealed, three impls) in `riir-ai/crates/riir-games-civ/src/civ/map_tick/karc.rs` behind `karc_forecaster` feature. Reuse `predictor_with_scratch` for the ridge solve; reuse `SpectralBasis::eval` for the basis. Zero game IP. GOAT gate G1–G3 on synthetic chaotic systems (Lorenz-63, double-scroll) reproducing paper Table I within 2×.

Phase 2–4: runtime integration in riir-ai (Plan 332), shard subtype in riir-neuron-db (Plan to file separately), LatCal commitment in riir-chain (Plan to file separately).

## TL;DR (one-line)

KARC = delay-embedding × basis expansion × closed-form ridge; the math is mostly shipped (PEIRA ridge, linoss basis, latent_functor per-NPC learn); the Super-GOAT is the *combination* as the first per-NPC reproducible trajectory forecaster that fits in a KarcShard and crosses the LatCal sync boundary as a 5-scalar emotion projection.
