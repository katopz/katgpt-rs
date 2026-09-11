# Research 271: MIT 6.S184 — Diffusion & Flow Matching Textbook (Vocabulary Crosswalk)

> **Source:** [MIT 6.S184 "Generative AI With Stochastic Differential Equations" lecture notes](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) — Peter Holderrieth & Ezra Erives (MIT CSAIL, 2026 edition, 84 pp.)
> **Date:** 2026-06-20
> **Status:** Reference (not a verdict on a novel mechanism)
> **Related Research:** 010 (ColaDLM), 034 (D2F), 041 (RePlaid), 044 (ELF), 055 (Nemotron TriMode), 119 (PiD), 131 (DiffusionBlocks), 150 (RecFM), 151 (GDSD), 154 (DFlare), 215 (ECHO), 228 (RCD), 236 (QGF), 003 (Commercial Strategy)
> **Classification:** Public (katgpt-rs)
> **Cross-ref (Gain):** Lee/Coeurdoux/Potaptchik/Du/Albergo/Vanden-Eijnden [arXiv:2608.01692 "Beckmann Transport Models: From Autonomous Flows to One-Step Maps"] — flow-matching variant using a time-independent drift `b(x)` (this crosswalk's §3 flow matching, but with the `t`-dependence dropped). **Revised from PASS to Gain** (Research 468): BTM's divergence equation `∇·(νb)=μ0−μ1` is the Beckmann OT flux constraint that our shipped CCE Moderator (Plan 295) documents as missing ("MFG dynamics / occupation-measure flow constraints"). The training aspects (learning `b`/`T` via regression/map loss) → riir-train; the framework recovers Poisson Flow (ν=1) and corrects Equilibrium Matching's biased loss (image FID only). The modelless value is the divergence-as-CCE-feasibility-constraint fusion.
> **PASS-Redirects (synthesis):** Lai/Song/Kim/Mitsufuji/Ermon [arXiv:2510.21890v3 "The Principles of Diffusion Models"] — the field-originators' definitive monograph (Sony + OpenAI + Stanford; MIT Press 2027; v3 2026-08-27). Verdict Pass — superset of this crosswalk's coverage; v3-specific rows (discrete-diffusion "Road Ahead" chapter, revised solvers chapter, flow maps, Diff-DPO, OT/Wasserstein) mapped in §6 below. Cite §6 alongside §2 when triaging diffusion papers.

---

## TL;DR

This is a **textbook**, not a single-mechanism paper. **Verdict: Pass on novelty** — every chapter (flow matching, score matching, classifier-free guidance, VAEs, discrete diffusion/CTMCs, generator matching) is already covered by 12+ research notes in the corpus and is shipping in `riir-ai/crates/riir-engine/src/transformer/dllm.rs`, ELF (Plan 079), D2F (Plan 066), DiffusionSampler (Plan 116), RecFM (150), QGF (236), and GDSD (151/186).

The artifact value here is **not a new plan**. It is a **vocabulary crosswalk** between MIT 6.S184's mathematical notation and our codebase's mechanism names. This crosswalk exists to prevent the three documented false-Super-GOAT failure modes:

- **#1 `evolve_hla` (no notes framing):** a per-NPC recurrent belief-state kernel ships in `katgpt-rs/crates/katgpt-sense/src/reconstruction.rs` with no `.research/` note framing it as a denoising/belief kernel. Notes-only grep misses it.
- **#2 `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` (different vocabulary):** DiPOD's "interleave self-distillation when ELBO drifts" ships as "coherence-driven re-estimation when coherence < τ_reest". Paper-vocabulary grep misses BOTH notes AND code.
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
| Flow / ODE simulation (Euler) | Drafter step / dflash predict | `crates/katgpt-forward/src/dflash.rs` |
| SDE / Euler-Maruyama (`+ σ_t √h ε`) | SDE noise injection (overhead benchmarked, NOT a runtime knob) | `katgpt-rs/tests/bench_elf_modelless.rs::bench_sde_noise_injection_overhead` |
| Score function `∇ log p_t(x)` | Latent direction vector (HLA projection / functor) | `katgpt-rs/crates/katgpt-core/src/sense/`, `riir-ai/.../latent_functor/` |
| Score matching (denoising) | `NoiseSchedule` mask ratios, `denoise_loop` | `riir-ai/crates/riir-engine/src/transformer/dllm.rs` |
| Conditional flow matching loss `L_CFM` | D2F training loss (`L = ‖ε_θ(α_t z + β_t ε) − ε‖²`) | `riir-ai/crates/riir-engine/src/transformer/dllm.rs::denoise_loop` |
| Classifier-free guidance `(1-w)u(∅) + w·u(y)` | QGF reference + scaled critic gradient; LoRA hot-swap linear interp | QGF (236), `riir-ai/.../adapters/`, `polytope_router.rs` |
| Denoiser `D_t(x) = E[z\|x]` (Remark 16) | `BakeStillState::from_posterior` | `riir-ai/.../lora_still_continual.rs` |
| Noise schedulers `α_t, β_t` | `ScheduleKind::{Uniform, LogitNormal{mean,std}}` | `katgpt-rs/src/speculative/d2f.rs::D2fDecodeConfig` |
| Gaussian CondOT path `p_t(x\|z) = N(tz, (1-t)²)` | `NoiseSchedule::monotonic_ratios` (discrete analog κ_t) | `riir-ai/crates/riir-engine/src/transformer/dllm.rs` |
| Marginalization trick `u(x) = ∫ u(x\|z) p_1\|t(z\|x) dz` | MoE/dMoE router (router = Bayesian posterior over expert identity) | 161 dMoE, 246 Manifold Power Iteration MoE Router |
| CTMC rate matrix `Q_t(y\|x)` | Token re-mask rate per position | `riir-ai/crates/riir-engine/src/transformer/dllm.rs::corrupt_block` |
| Factorized mixture path (Bernoulli mask) | Block-causal mask + remask | D2F (034) |
| MDLM `[MASK]^d → data` | D2F masked diffusion decode | `katgpt-rs/src/speculative/d2f.rs` |
| VAE encoder `q_φ(z\|x)` / decoder `p_θ(x\|z)` | LatCal raw↔latent bridge (deterministic) + HLA projection (latent) | `riir-ai/.../encoding/latcal*.rs`, `katgpt-rs/.../sense/` |
| VAE KL-to-prior regularizer | (no direct analog — our latents are committed raw via LatCal, not Gaussian-regularized) | gap |
| VAE rate-distortion Pareto frontier | "5 scalars across sync boundary" heuristic (not yet framed as R-D optimization) | gap → future issue |
| Fokker-Planck / continuity equation `∂_t p = -div(pu) + (σ²/2)Δp` | `belief_mass_divergence` (L1 norm `Σ_v |δ₁(flow)[v]|`) | **shipped** (Plan 314, 2026-06-24; opt-in `stokes_calculus` in `katgpt-dec`, re-exported `katgpt_core::dec`) — G3 conservation validator in `motor_gated.rs`, `PcaGlobalFn::BeliefMassDivergence` arm in `pca.rs`. Honest gate record: G-A branching-detector use FAILED (9.5× slower, 36% lower F1 than JS-divergence at `action_dim=8`, riir-ai Bench 334) — retained as a mass-conservation checker, NOT a branching detector |
| Generator Matching (Remark 40, unified discrete+continuous) | (not framed; CTMC and flow both ship but not unified) | gap |
| GLASS Flows (Remark 21, stochastic via ODE) | (not applicable — GLASS is narrowly reward-alignment for diffusion models, see Issue 038 closure; `mcts_collapse_bridge.rs` uses MCTS visit statistics, not flow sampling) | closed → Issue 038 |
| Langevin dynamics `dX = (σ²/2)∇log p dt + σ dW` | (tested by PTRM, zero gain over plain Gaussian — see Issue 037 closure) | closed → Issue 037 |
| Diffusion coefficient `σ_t` (runtime knob) | `TrdConfig::elf_noise_scale` (default 0.1) + `inject_sde_noise` (ELF Plan 079) | shipped — Issue 037 closed |

**Rows marked "gap" are not yet shipped or framed.** They are candidate fusion angles (§3), **not** novel mechanisms until Q1–Q4 novelty gate passes.

---

## 3. Latent-space reframings (mandatory per workflow §1.5 step 3)

For each of the five Super-GOAT factory modules, what does this textbook's machinery look like?

### 3.1 HLA (`katgpt-rs/crates/katgpt-core/src/sense/` + `riir-ai/.../hla/`)

**HLA is NOT a diffusion denoiser.** Verified by reading `riir-ai/crates/riir-engine/src/hla/kernel.rs`: HLA is a **second-order linear-attention streaming recurrence** with leaky decay γ (`SK += kkᵀ`, `CQV += qvᵀ`, `mQ += q`, exponential decay). It tracks second-order moments, not Bayesian posteriors. Treating HLA as `D_t(x) = E[z|x]` would be the R269 failure mode (defaulting to a plausible-sounding but wrong reframe).

The **correct** mapping: HLA is closer to a **linear-attention state-space model** (cf. 070 Gated DeltaNet, 230 Semiseparable SSD), where the recurrence plays the role of an ODE integrator. The denoiser analogy applies only to `BakeStillState::from_posterior` in LoRA continual training, not to runtime HLA.

### 3.2 `latent_functor/` (`zone_gating`, `reestimation`, `arithmetic`, `cross_game`, `k_selector`, `quality_gate`)

The textbook's **marginalization trick** (Thm 9: `u(x) = ∫ u(x|z) p_1|t(z|x) dz`) is the formal justification for treating `riir-ai/crates/riir-engine/src/latent_functor/zone_gating.rs` as a posterior-weighted mixture of zone-specific vector fields. Each zone = a "conditional vector field", the gate = the posterior. This is consistent with 161 dMoE and 246 Manifold Power Iteration MoE Router framings. **Not novel — just clarifying vocabulary.**

`riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs`'s "coherence-driven re-estimation when coherence < τ_reest" is the codebase-vocabulary equivalent of DiPOD's "interleave self-distillation when ELBO drifts". Already documented as failure mode #2.

### 3.3 `cgsp_runtime/` (curiosity-guided self-play)

Verified by reading `riir-ai/crates/riir-engine/src/cgsp_runtime/runtime.rs`: curiosity is modeled as a **decayed-absorb priority bandit** (`p ← p·decay + reward`, decay=0.7, capped at 1.0). This is **NOT** Langevin dynamics or SDE-driven exploration. The textbook's `σ_t` as a noise-injection knob for exploration is a genuinely different mechanism — see Issue 037.

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

## 5. Future fusion candidates (issues status)

Both fusion candidates from the initial version of this note have been **closed** after running the Q1–Q4 novelty gate:

- **Issue 037 — SDE Extension σ as runtime determinism/exploration knob.** ❌ **CLOSED NOT NOVEL.** Vocabulary translation revealed `TrdConfig::elf_noise_scale` (default 0.1) + `inject_sde_noise` (ELF Plan 079) **already ship the exact mechanism**. The stronger version (gradient-guided Langevin) was tested by PTRM (Research 049 §6.1, §7.4) and gave **zero improvement** over plain Gaussian — explicit negative result. Lesson: original Issue 037 was created from a paper-vocabulary-only grep that missed `noise_scale`/`inject_sde_noise`/`elf_noise_scale`. This is exactly the workflow §1.5 step 1 #2 failure mode the crosswalk was meant to prevent.

- **Issue 038 — GLASS Flows (Remark 21) for MCTS collapse bridge.** ❌ **CLOSED NOT APPLICABLE.** Reading the actual GLASS Flows paper (arxiv 2509.25170) revealed the lecture-note Remark over-generalized. GLASS is narrowly scoped to **reward alignment in diffusion models** (SMC/search/guidance that already sample from `pt′|t` of a flow model). `mcts_collapse_bridge.rs` operates on MCTS visit statistics (δmg discriminator), not flow sampling — GLASS doesn't apply. GLASS would only become relevant if we ship a continuous-flow NPC behavior policy that needs SMC reward steering; we don't have one today.

Two smaller reframings remain mentioned in §3 but **not** worth issues (too small to track):
- "MoE router = Bayesian posterior over expert identity" (theoretical clarification of 161/246, not new mechanism)
- "Fokker-Planck as runtime invariant validator" — ✅ **LANDED via Plan 314** (2026-06-24, four days after this note): `belief_mass_divergence` in `katgpt-dec/src/stokes_calculus.rs`, opt-in, zero-alloc wrapper over `codifferential`. The G-A branching-detector gate honestly FAILED (Bench 334: 9.5× slower + 36% lower F1); the primitive stands as a mass-conservation checker feeding `motor_gated` G3 + `pca.rs`.

---

## 6. v3 Addendum — arXiv:2510.21890 "The Principles of Diffusion Models" (2026-08-29)

[Lai/Song/Kim/Mitsufuji/Ermon](https://arxiv.org/abs/2510.21890) (Sony + OpenAI + Stanford; MIT Press 2027) is the field-originators' definitive monograph — the superset of MIT 6.S184 that this crosswalk anticipated. **Verdict: Pass** on the same grounds as the parent note: a synthesis of principles the corpus already ships or covers (12+ diffusion notes; `dllm.rs`, D2F, DFlare, GDSD, RecFM, QGF, dflash/d2f/dspark decoders), no new mechanism, no capability class we lack. **Reverse-grep found no unmapped documented gap**: §3's two gaps (Fokker-Planck validator ~20 LOC; VAE rate-distortion framing) were already adjudicated in §5 and the book covers both topics as reference material, not as unblocking mechanisms. Reverse-grep of the workspace for `2510.21890` before this addendum: zero hits (no prior note).

v3-specific content (v3 = 2026-08-27: new "Road Ahead" discrete-diffusion chapter, restructured + revised solvers chapter, reader guides; v2 = 2026-05-27 discrete-diffusion walkthrough + errata) and where each maps:

| v3 book topic | Codebase equivalent | Where it ships / is framed |
|---|---|---|
| Unified three views (variational §2 / score §6 / flow §5 → one time-dependent velocity field) | the drafter+denoiser stack as a whole | this crosswalk §2 rows 1–8 |
| Guidance chapter incl. **Diff-DPO** (p.257: winner-loser log-sigmoid over diffusion rollouts; errata-corrected surrogate keeps the BT log-sigmoid) | preference alignment on diffusion-style models | riir-train 064 ThoughtFold Mask-DPO (discrete-token preference margins), 429 Self-Rewarding Iterative DPO, 432 DiffusionOPSD — no recipe delta for our dLLM trainers |
| **Revised solvers chapter** (DPM-Solver-class continuous ODE integrators) | the discrete "solver" layer already ships under a matching name: `katgpt_core::dllm_solver` (RCD Research 228, 3SR) consumed by `denoise_loop_rcd*` / `denoise_loop_scheduled` in `crates/katgpt-forward/src/denoise_loops.rs` | different substrate: book solvers consume continuous-field derivative evaluations; ours consume residual-context/reuse state. Continuous latent flows (150 RecFM, 465 ODEWorld, 375 interpolants) run few explicit steps, matvec-bound — solver order is not a documented gap |
| **Flow-map models** (direct x_s→x_t mappings; consistency/shortcut class) | depth-step shortcut analog | 343 System 1.5 depth-step shortcuts, 490 DFlash2 pair-scored paths, 316 DSpark confidence-scheduled spec decode, `dflash_predict_*` in `crates/katgpt-speculative/src/dflash.rs` |
| **"Road Ahead" NEW chapter: discrete diffusion** | the shipped dLLM stack | `riir-engine/src/transformer/dllm.rs` (NoiseSchedule, denoise_loop), GDSD 151/186, D2F 034 / Plan 066, DFlare 154, DMax 072, CDM 517 (amortized twist SMC discrete diffusion) |
| Ch 7 optimal transport / Wasserstein | Beckmann transport flux constraint | 468 (cross-referenced in this note's header), 314 f-divergences Fisher-Rao |
| Ch 6 score-SDE / reverse-time Fokker-Planck (Prop 6.4.1, errata-fixed sign) / probability flow | QGF policy score; DEC divergence operator | QGF 236, 296 Stokes crosswalk (`codifferential` δ) |
| Ch 2 variational view incl. hierarchical VAE (v2/v3 errata: layerwise-KL collapse corrected to raw expectations) | VAE row in §2 | ColaDLM 010 rate-distortion |
| Appendices: Itô integration, Langevin, Green's identity | closed Issue 037 | see §5 |

**Signal-diff notes** (the §3.6 defense for each coverage dismissal above): (1) flow maps consume trajectory time `t`; our 343 shortcuts consume depth-step position — same direct-jump shape, different axis, both shipped; (2) the book's solvers consume derivative evaluations of a continuous field; `dllm_solver` consumes residual-context/reuse state — no discrete solver we lack; (3) Diff-DPO consumes winner-loser margins over continuous diffusion rollouts; riir-train 064 consumes masked-token preference margins — same family, ours discrete, no actionable delta.

**Adversarial-panel note:** not triggered — no riir-train deferral is being made, and the abstract carries no optimizer/loss/RL/backprop framing (views/guidance/solvers/flow-maps). Every training-objective chapter maps to an existing riir-train note (039 GDSD, 432 DiffusionOPSD, 433 RVM, 419 LOPD); the three-track grep requirement (pre-flight #5) is satisfied by those notes + the shipped `dllm.rs`/`denoise_loops.rs`/quest_grammar/TernaryDraftModel training surfaces.

---

## TL;DR

MIT 6.S184 is the canonical diffusion/flow-matching textbook. Every chapter ships or is covered in our corpus. **Verdict: Pass on novelty.** The artifact here is a vocabulary crosswalk (§2) to prevent future false-Super-GOAT claims caused by paper-vs-codebase vocabulary mismatch. The two initial "gap" angles (σ-as-runtime-knob, GLASS-flows-for-MCTS) were tracked as Issues 037/038, ran through the Q1–Q4 novelty gate, and **both closed**: 037 because `elf_noise_scale`/`inject_sde_noise` already ship (plus PTRM proved gradient-guided Langevin adds nothing), 038 because GLASS Flows is narrowly reward-alignment for diffusion models and doesn't apply to `mcts_collapse_bridge.rs`. The vocabulary crosswalk itself was the durable output — and the closures proved its worth (Issue 037's failure mode is exactly what the crosswalk was meant to catch, once it was extended with the missing rows). **§6 (2026-08-29) extends the crosswalk to arXiv:2510.21890 "The Principles of Diffusion Models" (v3)** — the field-originators' MIT-Press monograph and the superset reference; same Pass verdict, no new files, v3-specific rows mapped there.

---

## 7. Re-verification addendum (2026-09-11) — same URL, unchanged PDF

The source URL was re-fetched and re-examined end-to-end. **The document is unchanged since this note's distillation**: PDF `ModDate 2026-03-18` (predates the 2026-06-20 distillation), 84 pages (as cited), and all content anchors present (Remark 16 score reparameterization, Remark 21 GLASS Flows, Remark 40 Generator Matching; Appendices A–E incl. the Fokker-Planck proof and the literature guide). **Verdict stands: Pass.**

Crosswalk maintenance from the re-run — gap-row dispositions:
- §2 **Fokker-Planck row: gap → SHIPPED** (Plan 314 `belief_mass_divergence`; see updated row + §5). Plan 314's Goal line explicitly cites closing this note's gap.
- §2 **VAE R-D Pareto row: gap → FILED** as riir-ai `.issues/923_sync_scalar_rate_distortion_frontier.md` (commit `c4f0e453e`, 2026-09-11) — the "gap → future issue" promise kept. Bench protocol: rate = bits/NPC/sync (set size × width; current 5 × f32 = 160 bits via `SyncRecord`) vs distortion = behavior divergence (Kendall τ on cgsp rankings, CCE bucket disagreement, personality-ledger drift); verdict at the knee. Sibling-bridge + Lean-proof cost axes anchored in the issue.
- §2 **VAE KL-to-prior: by design, not a gap** — our latents commit raw via LatCal, not Gaussian-regularized.
- §2 **Generator Matching unification: deliberately NOT filed** (adjudication call, 2026-09-11): no consumer surface — the stack's discrete half (dLLM/CTMC denoise loops) and continuous half (velocity fields/steering) never need one generator interface today, and the two-brain model already keeps discrete belief (KG triples) and continuous affect separate by design. Filing would be YAGNI; R271 §5's original call stands. Revisit ONLY if a mixed discrete-continuous generation surface appears (e.g. personality drift as a jump-diffusion over style vectors).
- No new mechanisms in the unchanged document warrant rows; the successor-monograph superset is already mapped in §6.
