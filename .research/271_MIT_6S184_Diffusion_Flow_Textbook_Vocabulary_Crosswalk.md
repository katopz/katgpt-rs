# Research 271: MIT 6.S184 — Diffusion & Flow Matching Textbook (Vocabulary Crosswalk)

> **Source:** [MIT 6.S184 "Generative AI With Stochastic Differential Equations" lecture notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) — Peter Holderrieth & Ezra Erives (MIT CSAIL, 2026 edition, 84 pp.)
> **Date:** 2026-06-20
> **Status:** Reference (not a verdict on a novel mechanism)
> **Related Research:** 010 (ColaDLM), 034 (D2F), 041 (RePlaid), 044 (ELF), 055 (Nemotron TriMode), 119 (PiD), 131 (DiffusionBlocks), 150 (RecFM), 151 (GDSD), 154 (DFlare), 215 (ECHO), 228 (RCD), 236 (QGF), 003 (Commercial Strategy)
> **Classification:** Public (katgpt-rs)

---

## TL;DR

This is a **textbook**, not a single-mechanism paper. **Verdict: Pass on novelty** — every chapter (flow matching, score matching, classifier-free guidance, VAEs, discrete diffusion/CTMCs, generator matching) is already covered by 12+ research notes in the corpus and is shipping in `katgpt-rs/src/dllm.rs`, ELF (Plan 079), D2F (Plan 066), DiffusionSampler (Plan 116), RecFM (150), QGF (236), and GDSD (151/186).

The artifact value here is **not a new plan**. It is a **vocabulary crosswalk** between MIT 6.S184's mathematical notation and our codebase's mechanism names. This crosswalk exists to prevent the three documented false-Super-GOAT failure modes:

- **#1 `evolve_hla` (no notes framing):** a per-NPC recurrent belief-state kernel ships in `katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` with no `.research/` note framing it as a denoising/belief kernel. Notes-only grep misses it.
- **#2 `latent_functor/reestimation.rs` (different vocabulary):** DiPOD's "interleave self-distillation when ELBO drifts" ships as "coherence-driven re-estimation when coherence < τ_reest". Paper-vocabulary grep misses BOTH notes AND code.
- **#3 R269 (adapter-routing default):** defaulting to adapter routing when a latent-functor/HLA/LatCal reframing is available.

Future diffusion/score/flow-matching research notes should grep this crosswalk first.

---

## 1. Textbook Structure (1-line per chapter)

| § | Topic | Our coverage |
|---|---|---|
| 1 | Generation as sampling from p_data | implicit in all dLLM work |
| 2 | Flow models (ODE) / Diffusion models (SDE) | shipped: ELF (044), D2F (034) |
| 3 | Flow matching — simulation-free training | shipped: ELF (044), RecFM (150) |
| 4 | Score functions & score matching | covered: QGF (236) reframes policy score |
| 5 | Guidance — classifier-free | covered: QGF (236), ELF (044 L54) |
| 6 | VAEs & latent diffusion | covered: ColaDLM (010) rate-distortion |
| 7 | Discrete diffusion / CTMCs / MDLM | shipped: `dllm.rs` NoiseSchedule, denoise_loop |
| D | VAE rate-distortion Pareto frontier | covered: ColaDLM (010) "Three-Curve Framework" |

---

## 2. Vocabulary Crosswalk (paper term → codebase term → shipped artifact)

**Use this table when grepping for prior art on any diffusion/flow-matching/score idea.**

| MIT 6.S184 term | Codebase equivalent | Where it ships / is framed |
|---|---|---|
| Vector field `u_θ_t(x)` | `SpeculativeGenerator::generate()` velocity | `katgpt-rs/src/speculative/` |
| Flow / ODE simulation (Euler) | Drafter step / dflash predict | `katgpt-rs/src/speculative/dflash.rs` |
| SDE / Euler-Maruyama (`+ σ_t √h ε`) | SDE noise injection (overhead benchmarked, NOT a runtime knob) | `katgpt-rs/tests/bench_elf_modelless.rs::bench_sde_noise_injection_overhead` |
| Score function `∇ log p_t(x)` | Latent direction vector (HLA projection / functor) | `katgpt-rs/crates/katgpt-core/src/sense/`, `riir-ai/.../latent_functor/` |
| Score matching (denoising) | `NoiseSchedule` mask ratios, `denoise_loop` | `katgpt-rs/src/dllm.rs` |
| Conditional flow matching loss `L_CFM` | D2F training loss (`L = ‖ε_θ(α_t z + β_t ε) − ε‖²`) | `katgpt-rs/src/dllm.rs::denoise_loop` |
| Classifier-free guidance `(1-w)u(∅) + w·u(y)` | QGF reference + scaled critic gradient; LoRA hot-swap linear interp | QGF (236), `riir-ai/.../adapters/`, `polytope_router.rs` |
| Denoiser `D_t(x) = E[z\|x]` (Remark 16) | `BakeStillState::from_posterior` | `riir-ai/.../lora_still_continual.rs` |
| Noise schedulers `α_t, β_t` | `ScheduleKind::{Uniform, LogitNormal{mean,std}}` | `katgpt-rs/src/speculative/d2f.rs::D2fDecodeConfig` |
| Gaussian CondOT path `p_t(x\|z) = N(tz, (1-t)²)` | `NoiseSchedule::monotonic_ratios` (discrete analog κ_t) | `katgpt-rs/src/dllm.rs` |
| Marginalization trick `u(x) = ∫ u(x\|z) p_1\|t(z\|x) dz` | MoE/dMoE router (router = Bayesian posterior over expert identity) | 161 dMoE, 246 Manifold Power Iteration MoE Router |
| CTMC rate matrix `Q_t(y\|x)` | Token re-mask rate per position | `katgpt-rs/src/dllm.rs::corrupt_block` |
| Factorized mixture path (Bernoulli mask) | Block-causal mask + remask | D2F (034) |
| MDLM `[MASK]^d → data` | D2F masked diffusion decode | `katgpt-rs/src/speculative/d2f.rs` |
| VAE encoder `q_φ(z\|x)` / decoder `p_θ(x\|z)` | LatCal raw↔latent bridge (deterministic) + HLA projection (latent) | `riir-ai/.../encoding/latcal*.rs`, `katgpt-rs/.../sense/` |
| VAE KL-to-prior regularizer | (no direct analog — our latents are committed raw via LatCal, not Gaussian-regularized) | gap |
| VAE rate-distortion Pareto frontier | "5 scalars across sync boundary" heuristic (not yet framed as R-D optimization) | gap → future issue |
| Fokker-Planck / continuity equation `∂_t p = -div(pu) + (σ²/2)Δp` | (not yet a runtime invariant validator) | gap |
| Generator Matching (Remark 40, unified discrete+continuous) | (not framed; CTMC and flow both ship but not unified) | gap |
| GLASS Flows (Remark 21, stochastic via ODE) | (not framed; adjacent to mcts_collapse_bridge) | gap → Issue 038 |
| Langevin dynamics `dX = (σ²/2)∇log p dt + σ dW` | (not directly shipped; bandits + decayed absorb used for exploration instead) | gap → Issue 037 |
| Diffusion coefficient `σ_t` (runtime knob) | `bench_sde_noise_injection_overhead` measures cost; not yet exposed as determinism/exploration knob | gap → Issue 037 |

**Rows marked "gap" are not yet shipped or framed.** They are candidate fusion angles (§3), **not** novel mechanisms until Q1–Q4 novelty gate passes.

---

## 3. Latent-space reframings (mandatory per workflow §1.5 step 3)

For each of the five Super-GOAT factory modules, what does this textbook's machinery look like?

### 3.1 HLA (`katgpt-rs/crates/katgpt-core/src/sense/` + `riir-ai/.../hla/`)

**HLA is NOT a diffusion denoiser.** Verified by reading `hla/kernel.rs`: HLA is a **second-order linear-attention streaming recurrence** with leaky decay γ (`SK += kkᵀ`, `CQV += qvᵀ`, `mQ += q`, exponential decay). It tracks second-order moments, not Bayesian posteriors. Treating HLA as `D_t(x) = E[z|x]` would be the R269 failure mode (defaulting to a plausible-sounding but wrong reframe).

The **correct** mapping: HLA is closer to a **linear-attention state-space model** (cf. 070 Gated DeltaNet, 230 Semiseparable SSD), where the recurrence plays the role of an ODE integrator. The denoiser analogy applies only to `BakeStillState::from_posterior` in LoRA continual training, not to runtime HLA.

### 3.2 `latent_functor/` (`zone_gating`, `reestimation`, `arithmetic`, `cross_game`, `k_selector`, `quality_gate`)

The textbook's **marginalization trick** (Thm 9: `u(x) = ∫ u(x|z) p_1|t(z|x) dz`) is the formal justification for treating `latent_functor/zone_gating.rs` as a posterior-weighted mixture of zone-specific vector fields. Each zone = a "conditional vector field", the gate = the posterior. This is consistent with 161 dMoE and 246 Manifold Power Iteration MoE Router framings. **Not novel — just clarifying vocabulary.**

`latent_functor/reestimation.rs`'s "coherence-driven re-estimation when coherence < τ_reest" is the codebase-vocabulary equivalent of DiPOD's "interleave self-distillation when ELBO drifts". Already documented as failure mode #2.

### 3.3 `cgsp_runtime/` (curiosity-guided self-play)

Verified by reading `cgsp_runtime/runtime.rs`: curiosity is modeled as a **decayed-absorb priority bandit** (`p ← p·decay + reward`, decay=0.7, capped at 1.0). This is **NOT** Langevin dynamics or SDE-driven exploration. The textbook's `σ_t` as a noise-injection knob for exploration is a genuinely different mechanism — see Issue 037.

### 3.4 LatCal (`riir-ai/.../encoding/latcal*.rs`)

LatCal is the **deterministic raw↔latent bridge**. Textbook VAE §6.2 and §D give the rate-distortion framework for choosing how much to compress. Currently we heuristically commit "5 scalars across sync boundary" (valence/arousal/desperation/calm/fear per AGENTS.md); this should be reframed as a Pareto-optimal rate-distortion point. The textbook's Figure 22 "knee of the frontier" is exactly the design principle.

**LatCal stays deterministic (raw) — VAE stochasticity does NOT cross the sync boundary.** The textbook's VAE machinery applies to the *local* latent representation, not the committed values.

### 3.5 Adapter routing (`adapters/`, `polytope_router.rs`, dMoE)

Classifier-free guidance `(1-w)u(∅) + w·u(y)` is mathematically identical to **freeze/thaw adapter linear interpolation**: each frozen adapter is a "guided vector field", the unconditioned baseline is `u(∅)`, and `w` is the routing weight. This is the R269 warning — adapter routing is the GOAT-tier framing, **not** the Super-GOAT framing. The latent-functor/LatCal reframing is stronger when available.

---

## 4. Verdict

**Pass on novelty.** Every chapter of this textbook ships or is covered by existing research notes. No new mechanism. No new capability class. No Super-GOAT, no GOAT, no Gain, no plan.

**Reference value:** this note exists as a **vocabulary crosswalk** (§2) for future diffusion/flow-matching/score research. The three documented false-Super-GOAT failure modes are caused by vocabulary mismatch between paper and codebase; this crosswalk is the prophylactic. **Cite this note** when future research touches: score functions, flow matching, classifier-free guidance, denoising, noise schedules, VAE rate-distortion, CTMCs, MDLM, generator matching, Fokker-Planck, or Langevin dynamics.

**One-line reasoning:** Super-GOAT requires novel mechanism + new capability class + product selling point + force multiplier. This textbook has none of those relative to our existing corpus — it is foundational material we already ship.

---

## 5. Future fusion candidates (tracked as issues, NOT plans)

Per AGENTS.md ("Create issue at ./issues for optimization task, do not create plan") and the workflow's anti-escape-hatch rule ("do not write 'Super-GOAT candidate' — write 'fusion idea, novelty TBD, needs Q1-Q4' and create an issue"), the four "gap" rows from §2 that look most promising are tracked as:

- **Issue 037 — SDE Extension σ as runtime determinism/exploration knob.** Theorem 17 says given a trained flow `u_θ_t`, any `σ_t ≥ 0` produces a valid SDE at inference time. Currently `cgsp_runtime` uses decayed-absorb bandits for curiosity, NOT SDE noise injection. Fusion idea: σ=0 for sync-critical NPCs (bit-identical replay), σ>0 for exploring NPCs. **Novelty TBD — needs Q1–Q4.**

- **Issue 038 — GLASS Flows (Remark 21) for MCTS collapse bridge.** Stochastic-looking dynamics via ODEs using sampling tricks, allowing search algorithms while keeping ODE efficiency. Adjacent to `cgsp_runtime/mcts_collapse_bridge.rs` and 215 ECHO. **Novelty TBD — needs Q1–Q4.**

Two smaller reframings are mentioned in §3 but **not** worth issues (too small to track):
- "MoE router = Bayesian posterior over expert identity" (theoretical clarification of 161/246, not new mechanism)
- "Fokker-Planck as runtime invariant validator" (small implementation detail, ~20 LOC sanity check on HLA updates — pick up next time HLA integrity is touched)

---

## TL;DR

MIT 6.S184 is the canonical diffusion/flow-matching textbook. Every chapter ships or is covered in our corpus. **Verdict: Pass on novelty.** The artifact here is a vocabulary crosswalk (§2) to prevent future false-Super-GOAT claims caused by paper-vs-codebase vocabulary mismatch. Two promising "gap" angles (σ-as-runtime-knob, GLASS-flows-for-MCTS) are tracked as Issues 037/038 with explicit "novelty TBD, needs Q1–Q4" status — not promoted to plans or Super-GOAT candidates until the gate passes.
