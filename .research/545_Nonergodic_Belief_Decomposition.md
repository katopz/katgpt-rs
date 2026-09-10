# Research 545: Nonergodic Belief Decomposition — Telescoping Geometry

> **Status:** Active — Super-GOAT verdict; outputs filed this session (Plan 592 kernel + riir-ai Guide 373)
> **Source:** "The geometry of nonergodic composition" — Simplex blog, https://simplex.pub/nonergodic-geometry/ (Ray, Riechers, Shai et al., 2026-09-09). Prior-work lineage: Shai et al. [arXiv:2405.15943](https://arxiv.org/abs/2405.15943) (NeurIPS 2024, single-process belief geometry), Riechers et al. [arXiv:2505.18373](https://arxiv.org/abs/2505.18373), Piotrowski et al. [arXiv:2502.01954](https://arxiv.org/abs/2502.01954), Xie et al. [arXiv:2111.02080](https://arxiv.org/abs/2111.02080) (ICL as implicit Bayesian inference), Crutchfield [arXiv:2507.07343](https://arxiv.org/abs/2507.07343) (structural mixtures, Entropy 2026), Engels et al. [arXiv:2405.14860](https://arxiv.org/abs/2405.14860) (multidim features), Bhalla et al. [arXiv:2604.28119](https://arxiv.org/abs/2604.28119) (SAE concept manifolds), Shai et al. [arXiv:2602.02385](https://arxiv.org/abs/2602.02385) (factored representations).
> **Date:** 2026-09-10
> **Related Research:** 248 (BoM K-hypothesis), 302 + 158 (FAME / CommittedFieldBlend), 393 (Block-Sparse Featurizer), 242 (MicroRecurrentBeliefState), 543 (HMM homeostatic control), 192 (NextLat), 121 (hierarchical concept geometry), 246 + 161 (MoE routers), 577 (FK hypothesis weights)
> **Related Plans:** 592 (`nonergodic_belief` kernel, this repo); riir-ai `.research/373` (think-brain game-runtime guide)
> **Classification:** Public

---

## TL;DR

LLM training data is a **nonergodic composition**: each sequence comes from ONE of N latent generators (chosen at sequence start, never revisited). The optimal predictor's belief then provably decomposes as `η = (w₁η₁, …, w_Nη_N)` — a posterior weight `w_n` over WHICH generator is active, times each generator's own inner belief `η_n` — and this geometry (per-component simplices scaled toward origin by `w_n`, "telescoping cones") is linearly readable from transformer residual streams (R² ≈ 0.985). The feature picture becomes **eventually sparse, multi-dimensional**: sparse at the component level, dense within the active component.

**Distilled for katgpt-rs (modelless, inference-time):** the block-diagonal Bayes filter is a closed-form, zero-training runtime primitive — per-component beliefs updated independently by their own likelihoods (block numerator), coupled only by one scalar normalizer (denominator), with the telescoping view formed at readout. It slots directly onto the BoM `[K·D]` substrate (`katgpt-micro-belief`, Plan 281) and the FK weight solver (`hypothesis_weights_into`, Plan 577), and upgrades per-entity tracking from single-belief-with-decay to commit-with-revision multi-hypothesis tracking (game guide: riir-ai 373).

---

## 1. Paper Core Findings

1. **Nonergodic composition = direct sum.** Given component HMMs with token operators `T_n(x)`, the composition is `T(x) = ⊕_n T_n(x)` (block-diagonal — probability mass NEVER moves between blocks) with initial belief `η^(∅) = (μ₁η₁, …, μ_Nη_N)`. The generator is permanently confined to its block; **the observer's belief about which block is not**.
2. **Two-level belief decomposition.** The belief update numerator acts block-by-block (each `η_n` updated by its own operator, as if alone); the denominator is a single scalar `Z` summing over all blocks. Every reachable belief factorizes as `η = (w₁η₁, …, w_Nη_N)` with `Σw_n = 1`: mixture posterior over generators + per-generator conditional inner beliefs.
3. **Telescoping geometry.** Each component's belief simplex is scaled toward the origin by `w_n`; weights are coupled (`Σw=1`), so one cone grows while others shrink. Beliefs live in the `(Σ_n |S_n| − 1)`-simplex.
4. **Eventually sparse multi-dimensional features.** Early context: several components carry weight. As evidence accumulates `w → δ_{n,n*}`: geometry collapses onto the active block. Sparse at the component level, DENSE (multi-dimensional) within — a theory-derived refinement of the sparse-autoencoder "sparse 1-d features" picture toward **sparse dense subspaces**.
5. **Empirical verification.** Transformers trained on a two-Mess3 composition show this geometry linearly embedded in the residual stream (linear probe R² ≈ 0.985 vs ≈ 0.45 untrained); it emerges from next-token cross-entropy alone, and it is **strictly richer than the next-token distribution** (5-dim beliefs vs 2-dim next-token) — it encodes the full future, not just the next symbol.

## 2. Distillation

The transferable primitive is the **two-level sequential Bayes filter over a block-diagonal generator**, stripped of all interpretability apparatus:

```
State:  η_n (per-component belief, dim D each)  +  w_n (component posterior, K of them)
Tick:   for each n:  ℓ_n = likelihood_n(η_n, token)          // block numerator, independent
                     η_n ← normalize(update_n(η_n, token))   // inner Bayes step
        log w_n += ln ℓ_n ;  Z = LSE(log w)                    // the ONLY coupling: one scalar
        w_n ← exp(log w_n − Z)                                 // (normalize per-tick for stability)
Readout (telescope): out[K·D] = w_n · η_n written in place   // the paper's (w₁η₁,…,w_Nη_N)
```

Properties that make it a runtime primitive, not just a derivation:

- **Exact, modelless, closed-form.** No training, no gradients. Pure multiply + one LSE per tick.
- **Unnormalized accumulation is optional.** Numerically safer form renormalizes `w` per tick; the telescope view `w_n·η_n` is formed only at readout (zero extra state).
- **Commit-with-revision.** `w` sharpens toward one-hot with consistent evidence (like belief synchronization in Z1R) and — the load-bearing asymmetry — **revives** a collapsed hypothesis when evidence contradicts the commitment. Single-belief kernels (leaky integrators, `SpatialBelief` decay) cannot represent this; frozen blends (FAME) deliberately refuse it.
- **Confidence is principled.** `1 − max_n w_n` (or the posterior entropy) is the natural uncertainty signal — no hand-tuned decay constant.

**Fusion** (what paper × A × B produces that none can alone):

| Ingredient | Brings | Missing |
|---|---|---|
| This paper | two-level decomposition, telescoping layout, sparsification law, geometry grounding | an implementation substrate |
| BoM `[K·D]` slots + FK weights (R248/R577, `bom::hypothesis_weights_into`) | batched single-pass layout (right slot shape) + a normalized weight vector over hypotheses | per-slot generator likelihoods, online Z-coupled recursion (weights are one-shot cold-path tilt) |
| FAME / CommittedFieldBlend (R302/R158) | per-entity weight vector over K expert kernels | live posterior (frozen at commitment), inner beliefs (one blended kernel), normalization (sigmoid coexistence by design) |
| Think-brain `SpatialBelief` + sigmoid decay (riir-engine `ns_csg.rs`) | per-entity observation→belief plumbing, confidence bridge | any alternative-hypothesis set (single belief, scalar decay) |
| SubspaceSteeringField (R393 consumer, `steerable_axes`) | the sparse-DENSE-subspace geometry, shipped | runtime derivation (BSF blocks are trained/static, top-k selected) |

**Synthesis:** `NonergodicFilter` = BoM slots whose K hypotheses are **generator-conditioned inner beliefs** updated per-block, whose weights are an **online posterior** coupled by the scalar Z, whose readout is the **telescoping layout**, and whose collapse/revive gates are **sigmoid-thresholded** for behavior semantics. BoM supplies ~80% of the mechanical substrate; the missing 20% is exactly the paper's contribution.

**Numerics note (stack-rule honesty):** the posterior itself requires LSE-normalized products — probability semantics, not a learned-direction projection; the "sigmoid, never softmax" rule governs *bridges*, and here sigmoid appears where it belongs: the confidence-decay bridge to existing `SpatialBelief` consumers and the commit/revive behavior gates. The GOAT gate compares exact-Bayes vs sigmoid-approx variants so the choice is measured, not assumed.

## 3. Prior Art (mandatory disclosures)

- **Classical algorithmic ancestor — cite, don't claim:** Multiple Hypothesis Tracking (Reid 1979) and IMM filtering (Blom & Bar-Shalom 1988) already maintain posterior-over-models + per-model state estimates with model-conditioned updates at runtime, as do mixture-Kalman/Rao-Blackwellized filters. **A claim of "novel algorithm" for the raw decomposition would not survive.** The defensible deltas: (a) inner beliefs are **latent vectors with sigmoid-projected readouts**, not Kalman state estimates; (b) **geometry-grounded** — the transformer verification (R²≈0.985) makes the representation a falsifiable design substrate, a bridge the literature does not make; (c) **modelless zero-alloc SIMD** runtime on the BoM substrate; (d) **game-runtime per-entity integration + sync boundary** (only committed scalars cross); (e) sigmoid-gated commit/**revive** semantics (MHT prunes hypotheses; revival as behavior is ours).
- **Crutchfield arXiv:2507.07343** (structural mixtures) is the closest math prior on the mixture side — no networks, no telescoping geometry, no runtime.
- **The blog's own headline** (derivation + transformer verification of the nonergodic case) checked NOVEL via a 13-query sweep (2026-09-10); "eventually sparse multi-dimensional features" has zero prior published hits.
- Workspace greps: `nonergodic` → **0 hits** across all repos; two-level structure (posterior-over-generators + per-generator inner beliefs) → **0 hits** under 6 name variants (agent audit: 7 SINGLE-LEVEL, 2 K-PARTICLE, 0 TWO-LEVEL surfaces).

## 4. Consumer Reframes (both mandatory)

**Game context (priority #1 — full development in riir-ai Guide 373):** a player's **archetype is a nonergodic generator by construction** — chosen at character creation, never revisited. An NPC observer carries `w` over archetypes + per-archetype inner behavioral beliefs; actions are observation tokens; contradictory actions revive collapsed hypotheses (**emergent suspicion**). Same machinery tracks NPC agendas (forage/fight/flee/trade), zone archetypes (combat/market/social), and the mind-reading channel — where the observer/generator asymmetry is exactly the privacy boundary: the truth never syncs, the observer's committed scalars may.

**Healer context (priority #2 — fusion idea recorded, not planned):** spans within one file come from ONE domain "generator" (kernel_opt / clippy_lints / rust_perf / sec …), yet `KernelExpertRouter::pick_domain` is stateless per-span argmax. A file-level domain posterior + per-domain inner state is the nonergodic upgrade of the healer's router. Deferred pending the kernel's GOAT — filed as follow-up, not gold-plated here.

**Training track (audited defer):** no training-loop value (analysis paper); the optional experiment "verify telescoping geometry in our trained ternary models" is recorded but NOT filed — no active training run on multi-source data would change a decision based on that measurement today. Revisit when a multi-source curriculum lands in riir-train.

## 5. Verdict

**Super-GOAT (the fusion).** Scored per §1.5 on the narrowed claim — *runtime modelless two-level belief filter, geometry-grounded, latent-space-integrated, per-entity*:

1. **No prior art for the synthesis:** MHT/IMM cited as ancestor; the geometry-grounded + latent + game-runtime + revive combination has no published or shipped instance (0 TWO-LEVEL surfaces in-workspace).
2. **New behavior class:** commit-with-revision multi-hypothesis tracking — the missing middle between single-belief kernels (can't hold ambiguity) and frozen blends (refuse collapse). Revival-on-contradiction is a capability no incumbent has.
3. **Product selling point:** "Our NPCs infer your play-style from your actions, commit as evidence accumulates, and grow suspicious when you act against type."
4. **Force multiplier:** ≥2 pillars — belief-kernel family (sense/micro-belief) + per-entity cognition runtime (think-brain) + per-entity MoE (FAME) + commitment plane (LatCal) + the interpretability story.

**Mandatory outputs (this session):** open primitive → Plan 592 (`katgpt-micro-belief`, feature `nonergodic_belief`); architectural guide → riir-ai `.research/373`; MOAT gates below.

**MOAT gate per domain:**

| Domain | Fit | Ruling |
|---|---|---|
| katgpt-rs | Fundamental belief-kernel primitive, generic (ComponentModel trait, zero game semantics) | ✅ in scope — Plan 592 |
| riir-ai | Pillar-level: think-brain upgrade, per-entity cognition | ✅ in scope — Guide 373 (wiring plan AFTER kernel GOAT) |
| riir-chain | Commitment of observer scalars via LatCal | deferred — guide §boundary only |
| riir-neuron-db | hypothesis-trajectory persistence | deferred — not load-bearing for the verdict |
| riir-train | no training content | out of scope (audited defer above) |
| riir-clippy | domain-posterior routing | fusion idea recorded (§4) |

**Per-stack ledger slot:** belief-kernel family (`katgpt-micro-belief`), feature `nonergodic_belief` (opt-in), benchmark + GOAT gate before any promotion; demotion rule: if it loses to the BoM-K baseline on the identification bench, it stays opt-in and the note records the loss.

## 6. Validation Protocol

- G1: exact-posterior property tests vs brute-force enumeration on random small block-diagonal HMMs (bit-level agreement to 1e-5); Σw = 1; collapse monotonicity under consistent evidence; revival under contradiction.
- G2/G4: ns/tick at K=8/D=8, zero alloc (target sub-µs; expect O(K·D) SIMD dots + one K-length LSE).
- G3: existing `bom_sampling` paths untouched (feature-independent).
- Quality axis: two-generator identification streams — accuracy / log-loss / calibration vs (a) single-belief leaky integrator, (b) BoM K-particle baseline. Report the Floor: the filter exposes a posterior over hypotheses, not forecast intervals; the UQ floor rule binds only if a predictive-interval surface is added — decision documented in Plan 592 T2.6.

## 7. P0–P3

- **P0** — Plan 592 kernel + GOAT gate (this repo). *Blocker for everything below.*
- **P1** — riir-ai defend-wrong PoC: archetype-tracking vs single-belief on behavior streams (`riir-poc`, three competitors per §3.6).
- **P2** — think-brain wiring behind `riir-engine` feature flag + LatCal-committed observer scalars (≤4 archetype posteriors = ≤4 scalars; inner beliefs stay latent-local).
- **P3** — healer domain-posterior experiment (riir-clippy); trained-model geometry probe (riir-train, only when a multi-source curriculum exists).
