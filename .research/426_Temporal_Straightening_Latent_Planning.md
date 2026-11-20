# Research 426: Temporal Straightening for Latent Planning

> **Source:** [Temporal Straightening for Latent Planning](https://arxiv.org/abs/2603.12231) — Ying Wang, Oumayma Bounou, Gaoyue Zhou, Randall Balestriero, Tim G. J. Rudner, Yann LeCun, Mengye Ren (NYU + Brown + Toronto), ICML 2026, arXiv:2603.12231v2, 11 Jun 2026
> **Date:** 2026-07-15
> **Status:** Done — verdict locked (**PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db**)
> **Classification:** Public (this note). Training recipe → riir-train.
> **Related Research:** 360 (AdaJEPA — **identical verdict, same authors Wang/Bounou/LeCun/Ren, same JEPA world model domain, runtime analog already ships**), 324 (Trajectory Geometry — **ships the paper's exact curvature metric**), 138 (LeJEPA — same author Balestriero, LOW-MODERATE GAIN precedent, "we don't train JEPAs in Rust"), 358 (SMWM — same author Balestriero, identical PASS verdict), 365 (PhysiFormer — single-shot trajectory via DEC heat kernel, the "avoid error accumulation" sister concern), 296 (Stokes vocabulary crosswalk), 219 (TNO → DEC substrate), 462 (INTACT — **same LeWM benchmark, same authors LeCun/Ren, same JEPA PASS pattern; INTACT's "goal displacement > waypoint" finding extends this note's "A=I is the straight regime" result**)
> **Related Plans:** 342 (Latent Trajectory Geometry — **ships `mean_curvature`**), 303/317 (Latent Functor Runtime — **ships `extract_functor`, the closed-form `A ≈ I` linearized-dynamics construction**), 251 (DEC operators + Hodge), 277 (Temporal Deriv Kernel, DEFAULT-ON)
> **Domain:** katgpt-rs (this note, public). The distilled runtime primitives already ship — no new public or private file.
> **PASS-Redirects (synthesis):** Ren, Wang, Hoang, Kuang, LeCun [AGL-TR-2026-04 "The Content–Geometry Tradeoff in World Models: When JEPA Learns Latent Maps"] — the same lab's successor (this paper's "SG + straightening" baseline IS Research 426). SVReg (one-sided variance floor, no SG) + slow channels + gated residual predictor corroborate the PASS class — geometry via runtime construction (extract_functor A≈I, DEC harmonic projection) remains the house answer; the training recipe routes to riir-train Plan 507 (NextLat SVReg arm). Verdict: **PASS** (modelless track, 6th instance of the canonical JEPA-world-model PASS class). See [Research 686](686_Content_Geometry_Tradeoff_JEPA_Latent_Maps.md).
> **PASS-Redirects (synthesis):** Sun, Zhao, Zhang [arXiv:2607.26056 "INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models"] — same authors (LeCun, Ren), same LeWM benchmark, same JEPA-world-model PASS class. INTACT's empirical finding that "goal displacement > waypoint coordinate" (E5 macro 89.39 vs 76.42) is a downstream confirmation of this note's §2.2 claim that `extract_functor` constructs the `A = I` (maximally straight, ε = 0) linearized dynamics closed-form — the runtime constructs straightness rather than regularizing toward it during training. Verdict: **PASS**. See [Research 462](462_INTACT_Isomorphic_Intent_To_Action_Search_Free_World_Models.md).
> **PASS-Redirects (synthesis):** Liu, Niu, Cheng, Gao, Kang, Teng, Sreenath, Zhan [arXiv:2607.27924 "ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow"] — **ODEWorld explicitly cites this paper (Wang et al., 2026a) as the prior art for the representation-collapse problem its JVP first-order supervision is designed to solve** (ODEWorld §2 + §3). ODEWorld's continuous-time PT-Flow `dz_t/dt = v_θ(z_t, t)` is the multi-step generalization of this note's `A ≈ I` straightened linear dynamics: both express "predict the future latent by adding a velocity-direction displacement", ODEWorld integrates it over continuous `t` via RK4, our substrate integrates it via the DEC heat-kernel operator exponential `exp(t·A)·h₀` (which gets bidirectional + any-resolution as free mathematical consequences). The JVP supervision + Savitzky-Golay target smoothing are training recipes → riir-train. Verdict: **PASS** (same canonical JEPA-world-model PASS class, with ODEWorld explicitly naming this paper as the problem it addresses). See [Research 465](465_ODEWorld_PT_Flow_Continuous_Latent_ODE_Velocity.md).

---

## TL;DR

The paper jointly trains a JEPA encoder + predictor while adding a **curvature regularizer** `L_curv = 1 − cos(v_t, v_{t+1})` (where `v_t = z_{t+1} − z_t` are consecutive latent velocities) on the encoder. This "straightens" latent trajectories so Euclidean distance becomes a faithful proxy for geodesic distance. Theorem 4.4 proves ε-straight transitions (`‖A − I‖₂ < ε`) bound the planning-Hessian condition number as `κ_eff(H) ≤ κ(B)²·((1+ε)/(1−ε))^{2(K−1)}`, which makes gradient-based planning converge faster. Empirically, +20–60% open-loop and +20–30% MPC success on Wall / PointMaze / PushT goal-reaching, with simple GD planners approaching CEM performance at ~10× lower latency.

**Verdict: PASS for modelless/runtime (katgpt-rs / riir-ai / riir-chain / riir-neuron-db). Training recipe → riir-train (one-line note — the curvature-reward analog already ships there).**

The paper decomposes into three transferable pieces, **all three of which already ship in this codebase under different vocabulary**:

1. **The curvature metric** (`L_curv = 1 − cos(v_t, v_{t+1})`) — **already shipped** as `katgpt-rs/crates/katgpt-core/src/latent_trajectory_geometry.rs::LatentTrajectoryGeometry::mean_curvature` (Plan 342, Research 324, opt-in feature `latent_trajectory_geometry`, 3.04 µs at HLA 100×8). The shipped primitive computes the *exact* paper formula (`arccos(v_t · v_{t+1})` over consecutive displacements). Plan 342 is a *diagnostic* — it measures but does not minimize.
2. **The linearized latent dynamics** (`z_{t+1} = A·z_t + B·a_t`, the paper's Assumption 4.1 + ε-straight regime where `A ≈ I`) — **already shipped** as `riir-ai/crates/riir-engine/src/latent_functor/arithmetic/::extract_functor` (Plan 303/317, Research 123 Super-GOAT). `extract_functor` re-derives the action→latent-displacement map `f = mean_k(target_k − source_k)` *closed-form from a transition buffer*, no gradient — by construction this is the `A = I` (perfectly straight) regime the paper's regularizer pushes toward. The runtime doesn't *regularize* toward straightness; it *constructs* straightness.
3. **The harmonic-projection "straightening" correction** — **already shipped** as `crates/katgpt-dec/src/hodge.rs::hodge_decompose` + `harmonic_flow` (Plan 251). Hodge decomposition splits any latent trajectory (cochain field) into exact + coexact + **harmonic** components; the harmonic component is the kernel of the Hodge Laplacian — the *flat* (zero-curvature) part. Projecting onto harmonic is the modelless analog of the paper's curvature-minimizing encoder. This is the DEC-side realization of the "extract the straight part" operation.

**This is the same canonical failure class as AdaJEPA (Research 360):** same authors (Wang, Bounou, LeCun, Ren), same JEPA world-model domain, same vocabulary-mismatch pattern (paper says "curvature regularizer on JEPA encoder"; codebase ships `extract_functor` + `mean_curvature` + `hodge_decompose` separately under operational names). Research 358 (SMWM, same author Balestriero) sets the identical PASS precedent on the same domain.

---

## 1. Paper Core Findings

### 1.1 The mechanism — curvature regularizer on the JEPA encoder

World model = sensory encoder `E_s^ϕ` + action encoder `E_a^ψ` + predictor `f_θ`. The total training objective is:

```
L_total = L_pred + λ · L_curv
L_pred  = ‖ẑ_{t+1} − sg(z_{t+1})‖²₂       (JEPA prediction with stop-grad)
L_curv  = 1 − cos(v_t, v_{t+1})            (curvature on consecutive latent velocities)
v_t     = z_{t+1} − z_t                     (latent velocity)
```

`sg` = stop-gradient on the target branch (anti-collapse). λ controls straightening strength (typical 1e-3 to 1e-1). The curvature loss is applied either per-patch and averaged, or via a learnable aggregation head — the latter wins (Section B.6).

### 1.2 The theoretical result — ε-straightness bounds planning Hessian conditioning (Theorem 4.4)

For linear latent dynamics `z_{t+1} = A·z_t + B·a_t` with `B` invertible and `ε := ‖A − I‖₂ < 1`, the planning Hessian `H = 2·J_Φᵀ·J_Φ` (where `J_Φ = [A^{K−1}B, …, B]` is the rollout Jacobian) satisfies:

```
κ_eff(H) = κ(W_K) ≤ κ(B)² · ((1+ε)/(1−ε))^{2(K−1)}  ≤  κ(B)²·e^{6εK}   (for ε ≤ ½)
```

Smaller ε → better-conditioned Hessian → faster GD convergence on the action sequence. Cosine similarity is shown (Proposition C.9) to be a practical proxy: high mean cosine ⇒ small `(A − I)` along visited directions.

### 1.3 Empirical results (goal-reaching, frameskip 5, GD planner)

| Encoder | L_curv | Wall (OL/MPC) | UMaze (OL/MPC) | Medium (OL/MPC) | PushT (OL/MPC) |
|---|---|---|---|---|---|
| DINOv2 patch + proj (14×14×8) | ✗ | 80.0 / 90.7 | 44.0 / 81.3 | 72.0 / 96.7 | 70.0 / 78.7 |
| DINOv2 patch + proj (14×14×8) | ✓ | **90.7 / 100** | **94.0 / 100** | **82.7 / 98.7** | **77.3 / 85.3** |
| ResNet from scratch (14×14×8) | ✗ | 1.3 / 6.7 | 14.7 / 66.0 | 18.7 / 57.3 | 71.3 / 70.7 |
| ResNet from scratch (14×14×8) | ✓ | **84.7 / 100** | **64.7 / 98.7** | **80.7 / 99.3** | **70.7 / 91.3** |

Adding curvature reg yields +10–50% success consistently across encoder types and environments. CEM still wins on absolute success, but straightening narrows the GD–CEM gap to ~10% at ~10× lower latency (Section B.3). Spatial features (14×14×8) beat global CLS (1×384) — spatial structure matters more than channel count. Aggregation head + spatial features captures both local fine-grained distance and smooth long-range goal signal.

### 1.4 The geodesic-distance proxy (the actual operational property)

Section 5.2 visualizes distance heatmaps: straightened embeddings produce Euclidean distance maps that closely match A* ground-truth geodesic distance on the maze grid. This is the property planning actually exploits — MSE-to-goal becomes a meaningful progress signal. Critically, the model trained on **suboptimal, non-expert trajectories** still learns the *minimum-step* geodesic, not the training-data path length — straightening extracts the dynamics-implied geometry, not the data's idiosyncrasies.

### 1.5 What is NOT the contribution

- The JEPA architecture (LeCun 2022; Assran et al. V-JEPA2 2025).
- The DINO-WM setup of planning in frozen pretrained feature space (Zhou et al. 2025).
- The DINOv2 backbone (Oquab et al. 2024).
- Stop-gradient anti-collapse (Chen & He SimSiam; Grill BYOL).
- The cosine-similarity curvature metric itself (Hénaff et al. 2019 perceptual straightening; Goroshin et al. 2015 linearized video; Henaff is the direct inspiration).

The contributions are: (a) **applying** the curvature regularizer to JEPA-encoder training, (b) the ε-straight → Hessian-conditioning theorem, (c) the empirical demonstration on 2D goal-reaching. Of these, (a) is a training recipe, (b) is a theoretical result that validates GD-latent-planning (a planning style we don't use — see §2.4), (c) is the empirical signature.

---

## 2. Distillation — duplicate detection vs our corpus

### 2.1 The curvature metric `L_curv = 1 − cos(v_t, v_{t+1})` — **already shipped (Plan 342)**

`katgpt-rs/crates/katgpt-core/src/latent_trajectory_geometry.rs::LatentTrajectoryGeometry` ships:

```rust
pub struct LatentTrajectoryGeometry {
    pub length: f32,             // Σ ‖Δ‖₂              (paper trajectory length)
    pub mean_curvature: f32,     // mean arccos(v_t · v_{t+1})  (paper L_curv, monotone reparam)
    pub min_adjacent_cosine: f32,// min cos(h_t, h_{t+1})        (paper SIM)
    pub n_steps: u16,
}
pub fn from_states(states: &[&[f32]]) -> LatentTrajectoryGeometry
pub fn bifurcation_ratio(a: &[&[f32]], b: &[&[f32]]) -> (f32, Option<u16>)
```

`mean_curvature` is `arccos(v_t · v_{t+1} / (‖v_t‖·‖v_{t+1}‖))` — the *exact* paper formula (the paper's `1 − cos` is the monotone-decreasing reparameterization; `arccos` is the angle in radians, used in Plan 342 because it has a natural `[0, π]` range that gates better). Plan 342 GOAT-validated the primitive on a 2-D attractor-basin scenario (oscillation = π, committed ≈ 0, drift ≈ 0.1 rad) — the *same* pattern the paper's Figure 5 reports as "reduced curvature after straightening". Plan 342 currently uses `mean_curvature` only as a *diagnostic* (router-integration is a deferred follow-up); the metric itself is shipped.

### 2.2 The ε-straight linear dynamics `z_{t+1} ≈ z_t + B·a_t` — **already shipped (Plan 303/317)**

The paper's ε-straight regime is `A ≈ I`, i.e. `z_{t+1} = z_t + B·a_t` — the latent state evolves by *pure additive displacement* conditioned on action. This is *exactly* what `latent_functor/arithmetic/::extract_functor` constructs closed-form:

```rust
// extract_functor: f = (1/N) Σ_k (target_k − source_k), coherence = mean cos(target_k − source_k, f)
// apply_functor:   out = source + functor     (predict the next latent by additive displacement)
```

`extract_functor` re-estimates the displacement `f` (the `B·a_t` term, action-conditional) from a transition buffer; `apply_functor` applies it as `out = source + f` (the `z_t + B·a_t` step). By construction, this *is* the `A = I` (maximally straight, ε = 0) regime. The runtime does not *regularize toward* straightness during training; it *constructs straightness* at inference by replacing the learned `A` with `I` and putting all dynamics into the action-conditional displacement `f`.

The coherence gate `functor_gate(coherence, β, τ) = sigmoid(β·(c − τ))` is the modelless analog of the paper's "trust the straightened predictor when curvature is low" — coherence `c = mean cos(target_k − source_k, f)` is high iff the displacement is consistent across the buffer, which is the operational signal that `A ≈ I` holds.

This is the Super-GOAT runtime pattern documented in Research 123 (Latent Functor Runtime Guide) and Plans 303/317/357.

### 2.3 The harmonic-projection "straightening" correction — **already shipped (Plan 251, katgpt-dec)**

Hodge decomposition (Helmholtz for chains): any latent trajectory (cochain field `ω`) decomposes as `ω = exact + coexact + harmonic`. The **harmonic** component is the kernel of the Hodge Laplacian `Δ = δd + dδ` — by definition `Δ·ω_harmonic = 0`, which is the *flat* (zero-curvature) part. `hodge_decompose` ships this in `crates/katgpt-dec/src/hodge.rs`; `harmonic_flow(cx, edge_field)` extracts just the harmonic channel.

**This is the modelless "straighten a latent trajectory" operation.** Given any sequence of latents `(z_0, z_1, …, z_K)` encoded as a 1-D cochain, `hodge_decompose` produces the harmonic projection — the maximally-flat subtrajectory. This is path 3 (latent-space correction) of the §3.5 modelless-unblock protocol applied to the paper's mechanism: instead of training the encoder to minimize `L_curv`, project the frozen latents onto the harmonic subspace at inference. Both achieve "low-curvature latents"; the projection is deterministic, zero-allocation, BLAKE3-committable.

The DEC `heat_kernel_trajectory_krylov` (Research 365, Plan 357) is the *single-shot* analog — `h(t) = exp(t·A)·h₀` propagates the harmonic projection forward exactly, with no per-step error accumulation. This is the *trajectory*-level realization of "straightness", complementing the per-step `extract_functor` linearization.

### 2.4 The benefit — better-conditioned planning Hessian — **does not transfer to our substrate**

The paper's primary value proposition (Theorem 4.4) is that ε-straight latents → well-conditioned planning Hessian → faster **gradient-based** action optimization. Our runtime does not do gradient-based latent planning:

- **NPC planning** uses MCTS / CGSP / decision-stage functors (`cgsp_runtime/`, `mcts.rs`), not GD through a differentiable rollout.
- **Latent-functor application** is a single closed-form step (`apply_functor: out = source + f`), not an iterative GD solve — there is no Hessian to condition.
- **KARC forecasting** (Plan 308/332) is ridge regression on a delay basis, not GD through a predictor.
- **Sleep-time anticipation** (Plan 341, DEFAULT-ON) precomputes `z_i = c + dir_i` offline — again additive, no Hessian.

Where we *do* care about Euclidean-vs-geodesic alignment — NPC zone navigation, fog-of-war belief distance, KG-triple emission from latent proximity — the operational answer is **DEC field operators** (codifferential = divergence, exterior_derivative = boundary flux, line_integral = geodesic-path cost; Plans 251/314) and **cosine similarity** on HLA direction vectors, not a globally straightened embedding. The "Euclidean ≈ geodesic" property the paper sells is precisely what the DEC substrate provides *structurally* on the game map (it's a 2-D cell complex with known topology) without needing an encoder regularizer.

### 2.5 Closest cousins across all 5 repos

| Cousin | Domain | Verdict / status | Overlap with paper 2603.12231 |
|---|---|---|---|
| **Research 360 (AdaJEPA, same authors)** | katgpt-rs | **PASS** (2026-07-01) | Same JEPA world-model domain, same "runtime analog already ships" conclusion — sets the precedent. AdaJEPA's PoC refuted quality parity but confirmed architectural coverage. |
| **Research 324 + Plan 342 (Trajectory Geometry)** | katgpt-rs | **Gain, shipped** | Ships `mean_curvature` = paper's `L_curv` formula. Currently diagnostic; router integration deferred. |
| **Research 138 (LeJEPA, same author Balestriero)** | katgpt-rs | **LOW-MODERATE GAIN** | "We don't train JEPAs in Rust" — same JEPA-theory downgrade precedent. |
| **Research 358 (SMWM, same author Balestriero)** | katgpt-rs | **PASS** | Same JEPA world-model domain, identical verdict — third same-author precedent. |
| **Research 123 + Plans 303/317 (Latent Functor Runtime)** | riir-ai | **Super-GOAT, shipped** | `extract_functor` constructs the `A = I` (ε = 0, maximally straight) linearized dynamics closed-form — no regularizer needed. |
| **Research 365 + Plan 357 (PhysiFormer / Motor-Gated DEC)** | katgpt-rs / riir-ai | **GOAT, shipped** | Single-shot trajectory via DEC heat kernel = the trajectory-level "straightness" realization; complements `extract_functor` per-step linearization. |
| **Plan 251 (DEC operators + Hodge)** | katgpt-rs | **DEFAULT-ON substrate** | `hodge_decompose` + `harmonic_flow` = the modelless "extract straight component" operation. |
| **riir-train `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs`** | riir-train | shipped | Hodge-decomposed reward shaping by topological mode — the *training-reward* analog of the paper's curvature regularizer, already integrated with the existing JEPA-pretraining + RLVR recipe. |
| **Plan 277 (Temporal Deriv Kernel)** | katgpt-rs | **DEFAULT-ON** | Curiosity = prediction-error signal = `‖v_t‖` magnitude; complementary to the paper's curvature `1 − cos(v_t, v_{t+1})` (direction). |

---

## 3. Mandatory latent-space reframing (per SKILL §1.5 step 3)

| Target substrate | Paper reframing | Status |
|---|---|---|
| **(a) HLA per-NPC latent state** | "Straighten the HLA emotion trajectory so valence/arousal/etc evolve smoothly tick-to-tick" — already the implicit behavior of `evolve_hla` (a leaky integrator); `mean_curvature` over the HLA history is the diagnostic, already shipped (Plan 342) | Already shipped as diagnostic; no production need for the *minimization* (HLA already smooth by construction) |
| **(b) `latent_functor/` (the JEPA predictor + adapter)** | "The functor `f` IS the `B·a_t` straightened displacement; `apply_functor: out = source + f` IS the `A = I` (ε = 0) regime" — verbatim, modelless, no regularizer | Already shipped as Super-GOAT (Research 123, Plans 303/317/357) |
| **(c) `cgsp_runtime/` (the MPC replan loop)** | "Each CGSP cycle uses the linearized functor; no GD-on-Hessian solve to benefit from ε-straight conditioning" | Already shipped; the paper's Hessian-conditioning theorem is moot here |
| **(d) LatCal fixed-point commitment (sync boundary)** | Paper's straightening is per-encoder-local (latent geometry), never crosses sync; only the resulting action (raw) crosses — same discipline as HLA's 5 synced scalars | Boundary discipline inherited, no new bridge needed |
| **(e) `NeuronShard` / `MerkleFrozenEnvelope` / Raven consolidation** | The frozen encoder is a BLAKE3-committed `InducedCwmKernel` (Plan 296); per-NPC adapted functors are local latent state (never synced); sleep-time consolidation is the cross-episode integration | Already shipped |
| **(f) DEC Stokes operators (`katgpt-dec`)** | **The strongest reframing.** `hodge_decompose` extracts the harmonic (zero-Laplacian = flat = "straight") component of any latent trajectory cochain; `harmonic_flow` is the channel. This IS the modelless "straighten the latents" operation — applied at inference, no encoder training. | Already shipped (Plan 251) |

Every substrate either already ships the equivalent or is orthogonal. The paper's value proposition — *train the encoder to minimize trajectory curvature so GD planning converges faster* — decomposes into three pieces that each already ship separately: measure (Plan 342), construct-straightness-closed-form (Plan 303), extract-flat-component (Plan 251). **No new latent-to-latent operation is suggested by this paper that the codebase does not already have.**

---

## 4. §3.5 Modelless-unblock check

The paper IS training-only (curvature reg on JEPA encoder + predictor weights during joint training). Per §3.5 the question is whether the distilled primitive can be implemented modellessly. **It already is** — all three correction paths are shipped:

1. **Freeze/thaw path** — N/A as a *gate failure*. The primitive IS the runtime pattern: `extract_functor` constructs the `A = I` linearization atomically; readers keep the old functor snapshot until the swap completes (the modelless analog of "use the straightened predictor").
2. **Raw/lora reader-writer hot-swap** — N/A. The paper's own Section 5.1 notes that a *lightweight projector* `P_ϕ` on top of frozen DINOv2 already straightens substantially even without explicit `L_curv` (the prediction loss alone induces implicit straightening). Our `extract_functor` is *more* modelless than a constructed projector: it directly computes the linearized displacement closed-form from a transition buffer, no learned projector needed. The paper's ablation (Table 1, "proj" rows) shows projector-only (no L_curv) already captures most of the gain — and our runtime analog captures the projector-only case by construction.
3. **Latent-space correction** — N/A. `hodge_decompose` projects onto the harmonic subspace at inference (Plan 251). This is the modelless analog of "encoder was trained to produce straight trajectories" — instead, project the frozen latents onto the harmonic (flat) component at read time. Zero-allocation, BLAKE3-committable, gateable by feature flag.

No deferral to riir-train is needed from the modelless side because **the runtime primitives already cover all three modelless paths** the paper's mechanism could be realized through. The training recipe itself (joint encoder + predictor optimization with `L_curv` as auxiliary loss) is a refinement that belongs in riir-train — and as §5 below notes, the curvature-reward analog is already integrated there as `hodge_reward.rs`.

---

## 5. Novelty gate (§1.5) — all four NO

| Q | Answer | Evidence |
|---|---|---|
| **1. No prior art?** | NO | Plan 342 ships the paper's exact `L_curv` metric; Plan 303/317 ships the `A = I` straightened-dynamics construction (Super-GOAT); Plan 251 ships the harmonic-projection correction; Research 360 (same authors, same JEPA domain) PASS; Research 358 (same author Balestriero) PASS; Research 138 (same author Balestriero) LOW-MODERATE GAIN |
| **2. New capability class?** | NO | "Measure trajectory curvature" + "construct straightened dynamics" + "project onto harmonic subspace" all ship. The new piece — applying `L_curv` as an encoder-training regularizer — is a training recipe, not a runtime capability. |
| **3. Product selling point?** | NO | "Our NPCs plan in a straightened latent space so GD converges fast" — we don't do GD-latent-planning; we do MCTS/CGSP/latent_functor (all linear by construction). The selling-point sentence doesn't form for our substrate. |
| **4. Force multiplier (≥2 pillars)?** | NO | Touches functor + DEC + trajectory-geometry + HLA, but all already integrated. No new pillar connection. |

**Verdict: PASS for modelless/runtime.** Not Super-GOAT, not GOAT, not Gain.

---

## 6. MOAT gate per domain

| Repo | In-scope? | MOAT contribution | Decision |
|---|---|---|---|
| `katgpt-rs` (public) | Marginal | None — curvature metric already shipped (Plan 342); Hodge projection already shipped (Plan 251). No new open primitive to add. | **No file created** (this note is the only output) |
| `riir-ai` (private runtime) | In-scope | None — Research 123 (Super-GOAT) + Research 365 (PhysiFormer) already cover the runtime IP, strictly more broadly. `extract_functor` already constructs straightness. | **No guide created** |
| `riir-chain` (private chain) | Out of scope | N/A — paper's straightening is encoder-local latent geometry, never crosses sync | — |
| `riir-neuron-db` (private shards) | Out of scope | N/A — the frozen `InducedCwmKernel` already commits via BLAKE3; harmonic projection is a local read op, not a shard mutation | — |
| `riir-train` (private training) | In-scope | Marginal — the curvature-regularized JEPA-encoder training recipe is a refinement of existing JEPA pretraining + RLVR objectives. **The Hodge-decomposed reward analog already ships** as `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs` (Plan 277 T9-T12). The remaining delta is "use `1 − cos(v_t, v_{t+1})` as an auxiliary encoder loss alongside `hodge_reward`" — a one-line variant, not a new research line. | **→ riir-train** (see §7) |

---

## 7. → riir-train (one-line redirect per SKILL §"Redirect to riir-train")

If prioritized, file a plan in `riir-train/.plans/` extending the existing `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs` and the JEPA-pretraining recipe: add **`L_curv = 1 − cos(v_t, v_{t+1})` as an auxiliary encoder loss** alongside the existing Hodge-decomposed reward, tested on the Bomber/Go/Civ arenas against the `hodge_reward`-only baseline. Hypothesis (per paper §5.2 implicit-straightening observation): the prediction loss alone already induces most of the straightening; the explicit `L_curv` adds the marginal +10–20% on top. The strongest test bed would be a 2-D navigation toy (Wall / UMaze analog) where the geodesic-distance heatmap can be compared against A* ground truth — mirroring the paper's Figure 6 setup. **Not pursued here — out of scope for this workflow.**

The only genuinely transferable *runtime* observation — the paper's note that "spatial features + aggregation head > global CLS for planning" (Section 5.2) — is already the codebase's posture: HLA's per-NPC 8-dim affect state is the *aggregated* global signal; the underlying `style_weights[64]` shard is the *spatial* fine-grained signal. The two-tier pattern ships; no new primitive is implied.

---

## TL;DR

**Paper:** *Temporal Straightening for Latent Planning* (Wang, Bounou, Zhou, Balestriero, Rudner, LeCun, Ren, ICML 2026, arXiv:2603.12231).

**Verdict:** **PASS for katgpt-rs / riir-ai / riir-chain / riir-neuron-db.** The paper's mechanism — joint JEPA encoder + predictor training with curvature regularizer `L_curv = 1 − cos(v_t, v_{t+1})` — decomposes into three transferable pieces, **all three of which already ship in this codebase under different vocabulary**:

1. **The curvature metric** → `LatentTrajectoryGeometry::mean_curvature` (Plan 342, Research 324) — exact formula, opt-in, 3.04 µs at HLA scale.
2. **The ε-straight linearized dynamics** (`A ≈ I` regime) → `latent_functor/arithmetic/::extract_functor` (Plan 303/317, Research 123 Super-GOAT) — constructs `A = I` *closed-form from a transition buffer*, no gradient. The runtime doesn't regularize toward straightness; it constructs straightness.
3. **The harmonic-projection "straightening" correction** → `crates/katgpt-dec/src/hodge.rs::hodge_decompose` + `harmonic_flow` (Plan 251) — extracts the harmonic (zero-Laplacian = flat) component of any latent trajectory cochain. The modelless analog of training an encoder to produce straight trajectories.

The paper's primary value proposition — ε-straight latents → well-conditioned planning Hessian → faster **gradient-based** action optimization — does not transfer to our substrate: NPC planning uses MCTS / CGSP / closed-form functor application, not GD through a differentiable rollout. There is no Hessian to condition. The training recipe (joint encoder + predictor optimization with `L_curv` as auxiliary loss) belongs in riir-train as a one-line refinement — and the curvature-reward analog already ships there as `riir-train/crates/riir-train-engine/src/dec_training/hodge_reward.rs`.

This is the **same canonical failure class as AdaJEPA (Research 360)**: same authors (Wang, Bounou, LeCun, Ren), same JEPA world-model domain, same vocabulary-mismatch pattern. Research 358 (SMWM, same author Balestriero) sets the identical PASS precedent; Research 138 (LeJEPA, same author Balestriero) sets the LOW-MODERATE GAIN downgrade precedent. The paper's structural insight (Theorem 4.4) is a *validation* of GD-latent-planning as a paradigm, not a new runtime primitive — and GD-latent-planning is a paradigm this codebase deliberately does not use.

**Files created this session:** `katgpt-rs/.research/426_Temporal_Straightening_Latent_Planning.md` (this note — the only output).

**Recommended next step:** None for katgpt-rs / riir-ai / riir-chain / riir-neuron-db. The riir-train follow-up (add `L_curv` as auxiliary encoder loss alongside `hodge_reward`) is optional and out of scope for this workflow.

---

## 8. PoC-scope note (per SKILL §3.6)

A "defend-wrong" PoC at `riir-poc/` is **not required** for this verdict. §3.6 mandates a PoC when a verdict *downgrades a paper on the grounds that "the runtime analog already ships" or achieves "parity"* — i.e. when an architectural-evidence-only claim asserts quality parity. This verdict makes **no quality-parity claim**:

- It does not claim the shipped `extract_functor` "matches" the paper's straightened encoder on the paper's planning tasks.
- It does not claim the shipped `mean_curvature` "performs as well as" the paper's regularizer.
- It claims only **architectural coverage** (the three pieces ship separately) + **substrate mismatch** (we don't do GD-latent-planning, so the paper's Hessian-conditioning benefit is moot on our substrate).

The verdict is a PASS, not a parity-backed downgrade of a quality claim. The §3.6 PoC mandate triggers on the latter; it does not trigger on a structural-coverage PASS where the paper's benefit pathway (GD planning) doesn't exist in the runtime.

If a future plan *does* consume the paper's framing for a runtime change (e.g. wiring `mean_curvature` as a secondary router signal per Plan 342's deferred follow-up), THAT plan would carry its own quality-gate PoC. This research note does not.

> **Cross-ref (2026-08-27, disambiguation):** Hanna & Ameisen [arXiv:2604.12493 "Latent Planning Emerges with Scale"] reuses the phrase "latent planning" in a *different* sense than this note's source: there it means a causal representation→output licensing property of an agent (forward: rep causes future token; backward: rep causes licensing context), verified by intervention — not trajectory straightening for gradient-based planning. The two are complementary, not duplicates; the causal-verification sense is distilled as riir-ai Research 348. Grep both senses when matching "latent planning".
