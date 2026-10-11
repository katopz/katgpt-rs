# Research 582: PFD — Probability-Flow Distillation (Exact Wasserstein Gradient Flow)

> **Source:** [Probability-Flow Distillation: Exact Wasserstein Gradient Flow for High-Fidelity 3D Generation](https://arxiv.org/abs/2605.09071) — Rohith Ramanan, A. N. Rajagopalan (IIT Madras), arXiv:2605.09071, cs.CV, 2026-05-09.
> **Date:** 2026-09-22
> **Status:** DISTILLED — pending owner decision
> **Classification:** Public (generic math) + private routing (see §5)
> **Related Research:** 517 (CDM twisted SMC — closest time cousin, discrete diffusion), 505 (Mean-Field Distributional Steering — ships the MeasureReward/FkStepper substrate), 290 (Latent Field Steering), 236 (QGF), 369 (Renoise-CE — the round-trip probe lineage), 197 (Plan 222 solver switching — ships DPM-Solver++(2M)), 034 (D2F discrete-diffusion decode), 561 (TSD — the bounded-loss lesson that constrains the KL recipe)
> **Related Issues:** katgpt-rs 875 (horizon weighting + target-anchored probe), riir-train 569 (exact-integration dLLM recipes), riir-ai 996 (crowd target-density fusion)
> **Panel:** 2 advocates (No-GD + Model-based), same-batch with §4 searches. Both tracks produced plans/issues; one discard class recorded in §3.

---

## TL;DR

PFD improves text-to-3D score distillation by replacing SDI's posterior-mean estimator — proven equivalent to a **single forward-Euler step of the reverse DDIM PF-ODE spanning the entire remaining noise interval** (Δσ = −σ_t) — with **exact PF-ODE integration in both directions** (deterministic DDIM inversion forward, DPM++ reverse), and proves the resulting discrepancy update `x0 ← x0 − η(x0 − x̂0)` is an **exact Wasserstein gradient flow** of a time-averaged KL functional under linear drift + stop-gradient (Theorem 1). The headline transfer for this stack is NOT 3D: (a) the algorithm is itself No-GD (stop-gradient, forward evaluations only, particle-state not weight-state); (b) the Fubini time-weighting law — averaging uniformly-sampled-t partial integrals yields effective weight (T−t) — is a closed-form schedule primitive nothing in the workspace ships; (c) the frozen-surrogate result (unconditional score ≈ particle-distribution score near the data manifold) is theory backing for the freeze-over-finetune stance the `distributional_steering` substrate already takes. Measure-level distribution matching against a target **already ships** as `MmdReward` (−MMD²(μ,ν), same fixed point as KL, no density estimation) — PFD's residual modelless delta is thin (weighting schedules + a target-anchored probe mode). The model-based arm files a staged PoC for the dLLM training lane (exact-integration rollouts + time-averaged bounded-KL objective).

## 1. Paper core findings

1. **Single-step bias identity (their Claim):** the DDIM posterior mean `E[x0|xt] = x̃t − σt·εφ(xt,t)` is exactly one forward-Euler step of the reverse DDIM PF-ODE in σ-parameterization (`dx̃/dσ = εφ`) with `Δσ = −σt` — the whole remaining interval in one step. This bias is the mechanism of SDI's mode-seeking/incomplete-distribution behavior (2D concentric-circles toy: SDS → single mode, SDI → partial, PFD → full multimodal recovery).
2. **PFD algorithm (their Alg. 1):** iterate `x0 ← x0 − η·Δt` where `Δt = x0 − x̂0`, `x̂0` obtained by forward-integrating the PF-ODE of the current particle distribution (DDIM inversion in practice) then reverse-integrating the target prior's PF-ODE. Stop-gradient throughout — no backprop through the solver, gradients not propagated through score evaluations.
3. **Theorem 1 (exactness):** under linear drift `f(x,t)=a(t)x` (VP/VE class) + stop-gradient (Lemma 3: the flow Jacobian collapses to a scaled identity), `E_t[Δt]` EXACTLY equals the Wasserstein gradient of `F[q0] = E_t[w(t)·DKL(qt‖pt)]` with `w(t) = ½(T−t)g(t)²c(t,0)²`, `c(t,0)=exp(−∫a)`. The **(T−t) factor is a Fubini swap** on the triangular integration domain: sampling t uniformly and integrating the discrepancy from 0 to t gives effective weight (T−s) — max weight at low noise, decaying linearly to **zero at t=T**.
4. **Negative CFG algebra:** `ε_src + γ(ε_tgt−ε_src) ≡ ε_tgt + (1−γ)(ε_src−ε_tgt)` — swapping source/target roles ≡ `γ_cfg = 1−γ`; the forward (inversion) pass needs NEGATIVE guidance (−6.5) because it must estimate the SOURCE (current-sample) distribution's score.
5. **Frozen-surrogate single-particle regime:** the frozen unconditional diffusion score substitutes for the particle distribution's own score when samples stay near the data manifold — eliminating VSD's LoRA adaptation; memory ≈ SDI, ~2h/object on A100. Time annealing (sampling range [0.02T,0.98T] → [0.02T,0.70T] late) improves fine detail.

## 2. Path-0 decomposition + coverage (signal-diff per §3.6)

| PFD component | Ships? | Where / mechanism-level delta |
|---|---|---|
| Multi-step PF-ODE solving (the correction the bias identity motivates) | **YES** | `dllm_solver.rs` (Plan 222): `DpmSolver2M` is the DEFAULT `SolverKind` (2nd-order multistep); QSample re-noise+re-predict partial steps; entropy-triggered switching. The shipped default already avoids the one-step pattern; the identity lands as documentation, not a gap. |
| Measure-level distribution matching to a target | **YES** | `distributional_steering.rs` (Plan 577/Bench 682): `MmdReward` = `−MMD²(μ,ν)` against a frozen target particle set — same fixed point as PFD's KL (both vanish iff μ=ν), works on empirical measures with NO density estimation. |
| KL reward row | **NO — documented non-goal** | Module doc: "the entropy row is a documented non-goal (needs density estimation — Research 505 risk 3; approximate via MMD-to-uniform instead)". Covers the PFD KL row identically: plug-in KL between particle populations needs densities. Residual delta = the w(t) weighting only → Issue 875. |
| Opaque-reward steering / frozen scorer as surrogate | **YES** | `FkStepper` assumes a caller-supplied frozen scorer; `Closure` reward row (Plan 581 T1.1) covers black-box scorers; `X0ProxyReward` (twist_cache) is the x̂₀ posterior-mean proxy. PFD adds the *validity condition* (near-manifold) — theory for shipped behavior. |
| Perturb→re-resolve→drift scoring operator | **YES** | `renoise_ce.rs` (R369/Plan 406): perturb a completed state, re-resolve, cross-entropy drift. `re_resolve` is an open trait seam — PFD's *target-anchored* variant (resolve against a target prior, not self) is a new probe mode, not new machinery → Issue 875 T4. |
| (T−t) horizon weighting / w(t) closed-form schedule | **NO** | Nothing in the workspace weights time-averaged accumulation by remaining horizon: `twist_step_into` is span-normalized β/KL-budget; renoise k_draws average flat; solver switching is entropy- (state-) driven, not horizon-driven. → Issue 875 T1/T2. |
| Time-annealed t-sampling range | **NO** (as schedule) | Plan 222 switches on entropy (state), PFD anneals on iteration count (time). Orthogonal axis → Issue 875 T3. |
| SMC/ESS/resampling shell | **YES** | `twist_cache.rs` + `distributional_steering.rs` (systematic/residual resample, ESS guard) — R517 lineage. |
| Wasserstein/OT machinery | **YES** | `gw_alignment` (Gromov-Wasserstein, bit-deterministic), `leakage_probe` (entropic Sinkhorn + Procrustes), `mag::transfer::Wasserstein1d`, `katgpt-attn/static_cal.rs` (Sinkhorn→O(1) committed table — the exact pattern a w(t) table would follow). |
| Continuity-equation validator (WGF's governing equation) | **YES** | `katgpt-dec/stokes_calculus.rs` `belief_mass_divergence` (Fokker-Planck on belief cochains, Plan 314); `flow.rs` Helmholtz exact/coexact steering fields. |
| Particle update loop x0 ← x0 − ηΔt | **YES (shape)** | `FkStepper` damped fixed-point iteration on `WeightedPopulation` (k_fp Picard, damping). |

## 3. Adversarial panel + discard ledger

**No-GD advocate** (22 candidates) — headline: PFD is a No-GD algorithm wearing distillation clothes (stop-gradient, forward-evals only, particle-state not weight-state); Theorem 1 is a license to COMPUTE Wasserstein gradients without GD. **Model-based advocate** (11 candidates, 9 substantive) — recipes for the dLLM/GDSD lane with 4090 GPU-hour estimates; verified `riir-train-gpu/src/{loss_grpo,loss_dpo}.rs` + the `dllm/` directory module (`riir-train-gpu/src/dllm/mod.rs`), `loss_uopsd`, TSD harness (`riir-train-gpu/src/distill_harness.rs`), and that Plan 361's twist head is plan-level only (no `.rs` yet).

**Coordinator discards (mechanism-level reasons):**

- **KL `MeasureReward` row** — blocked by the module's own documented non-goal (density estimation); `MmdReward` is the shipped measure-level analog with the same fixed point. The residual (w(t)-weighted matching across noise levels) is Issue 875 T2.
- **Negative-CFG algebra helper in katgpt-core** — no consumer: our guidance surfaces are measure-level (MMD/moment first-variations), not prediction-extrapolation between conditional/unconditional scores; the identity is recorded for the dLLM training lane instead (riir-train 569 C4) where guidance-carrying passes exist.
- **Frozen-surrogate validity gate** — the behavior already ships (`FkStepper` frozen scorer, `Closure` rewards); the paper supplies the near-manifold validity condition but there is no measured failure mode to defend against; recorded here as theory, no file.
- **Lemma-3 `k_fp=1` fast-path license** — converts a tuning knob into a theorem-scoped decision, but K_FP=3 has no measured cost and no consumer pain; recorded here, no file.
- **Healer trajectory-level PFD** — no noise ladder exists in code space; the round-trip-through-frozen-operator shape is already the renoise-CE lineage (R369); the mapping is a loose analogy. Discarded.
- **Single-step-bias identity as shipped artifact** — the correction it motivates (multi-step higher-order solving) already ships as the `dllm_solver` DEFAULT; the identity's value here is documentation/lens. Discarded as a gap.

**Survivors** (filed): horizon-weighting utility + w(t) table + time-annealing + target-anchored probe mode (katgpt-rs Issue 875); dLLM training recipes C1/C2/C4/C5/C6 + C9 toy harness (riir-train Issue 569); crowd target-density composition (riir-ai Issue 996).

## 4. Fusion

**Closest cousins:** R505/Plan 577 (`MmdReward`+`FkStepper` — measure-reward population steering), R369/Plan 406 (renoise round-trip), R197/Plan 222 (solver switching), R517/Plan 581 (twist SMC + x̂₀ proxy). **Fusion this paper triggers:** *crowd/zone population steering toward a designer-specified TARGET DENSITY via the shipped distributional-steering first-variation field, with PFD's (T−t) horizon law as the accumulation schedule* — i.e., R505's substrate × PFD's weighting law × the zone-graph game surface (consumers of `katgpt-core/src/flow/mod.rs` LeoPotentialGrid and `katgpt-dec/src/coulomb.rs` — both katgpt-rs files; see riir-ai Issue 996 for the game wiring). **In-workspace prior art first:** Bench 825 `CoulombFlowField`/`CrowdRouter` (opt-in `coulomb_flow`) already transports μ0→μ1 EXACTLY on a zone graph — the fusion's honest residual is the annealed per-tick MMD-field dynamics, not target-density matching itself. External prior art is DENSE and must not be claimed against: DM-Count (NeurIPS, OT for crowd density matching), MBOT evacuation (ACM 2023), OT-M crowd localization, "Trajectory-Optimized Density Control with Flow Matching" (arXiv 2025), plus the VSD/ProlificDreamer WGF lineage and the NeurIPS "Are We Really Learning the Score Function?" WGF framing. Our angle is only the composition of shipped substrate at zone-graph scale (ns/NPC/tick, 0-alloc, unarmed bit-identical) — novelty TBD by PoC, hence an issue not a plan (riir-ai 996). Secondary fusion recorded, unplanned: OT-coupled particle updates (rank-pair discrepancies via `mag::transfer`'s sorted-quantile core) inside `FkStepper`.

## 5. Per-track verdicts + routing

- **Modelless track: Gain.** Genuinely-unshipped thin primitives (horizon weighting, w(t) table, time-annealing, target-anchored probe) → katgpt-rs Issue 875 (poc tier, opt-in feature, GOAT gate per task). Not Super-GOAT: Q1 fails (dense prior art incl. our own shipped MMD-to-target), Q3 unproven without PoC.
- **Model-based track: Gain.** Staged PoC for the dLLM lane → riir-train Issue 569 (C9 toy harness FIRST — concentric-circles mode-coverage gate; then time-averaged bounded-KL objective with the TSD reverse-KL-forbidden caveat, exact-integration rollouts for GRPO/GDSD, negative-CFG sign convention, time-annealing curriculum, solver-consistent draft training). Envelope note: neither track lands on the AR-decode hot path; the dLLM lane is secondary — PoC-first, no 4090 spend before the toy gate passes.
- **Files created this session:** this note; katgpt-rs `.issues/875_pfd_horizon_weighting_target_anchored_probe.md`; riir-train `.issues/569_pfd_exact_integration_dllm_recipes.md`; riir-ai `.issues/996_crowd_target_density_matching_fusion.md`.

## 6. What does NOT transfer (honest caveats)

- The 3D application, NeRF/hash-grid pipeline, CLIP/IQA metrics — out of scope for every repo.
- The exactness theorem is continuous-space, linear-drift + stop-gradient scoped; the discrete/cochain analog would be OUR derivation (not claimed).
- Discrete-time DDIM inversion in dLLMs is lossy (no exact discrete inverse) — C1's "exact integration" degrades to "higher-fidelity integration" in the discrete lane; the toy harness must measure that gap, not assume it away.
- No quality-parity claim is made for any row here vs the paper's numbers (text-to-3D has no in-stack counterpart); §3.6 PoC obligations ride the filed issues, not this note.
