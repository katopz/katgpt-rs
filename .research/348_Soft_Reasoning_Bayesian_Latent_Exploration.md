# Research 348: Soft Reasoning — Bayesian Latent Exploration over the First-Token Embedding

> **Source:** *Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration* — Zhu, Zhao, Yan, He, Chen, Gui. arXiv:2505.24688 (May 2025, ICML 2025). Code: https://github.com/alickzhu/Soft-Reasoning
> **Date:** 2026-06-29
> **Status:** Active
> **Related Research:** 281 (Salience Tri-Gate), 248 (BoM), 290 (Latent Field Steering), 098 (PrudentBanker), 192 (NextLat), 322 (Conformal UQ floor), 325 (Survey §7.2 gap G6)
> **Related Plans:** 281 (BoM sampler), 309 (Latent Field Steering primitive), 340 (Conformal-naive floor)
> **Classification:** Public

---

## TL;DR

Soft Reasoning treats the **first-token embedding as a controllable latent variable** and runs **closed-loop Bayesian optimization (Expected-Improvement, Gaussian-kernel surrogate, dim-reduced random projection) against a verifier+coherence reward** to pick the next perturbation. It converges in 2–4 iterations (paper Table 2) and beats Self-Consistency / FIRE / CoT-Decoding / RAP on accuracy while using **6–14% of RAP's tokens/time** (paper Table 3). It is pure inference-time, model-agnostic, needs no parameter access — the verifier is the same model that generates.

**Distilled for katgpt-rs (modelless, inference-time):**
A **closed-loop verifier-guided Bayesian exploration primitive over a latent perturbation vector**. It fuses three shipped mechanisms that have never been wired into one loop:
1. **BoM-style K-query perturbation** (`MicroRecurrentBeliefState::sample_k_states`, Plan 281) — inject `K` Gaussian perturbations at one latent site, batch-evaluate.
2. **Bandit acquisition function** (`PrudentBanker` P-UCB, `SketchSampler`, `CuratorBandit` Thompson) — pick the next perturbation site by EI/UCB, regret-bounded.
3. **Verifier+coherence reward** (`CLR` Multi-Generate, `ConstraintPruner`, NextLat belief residual as coherence) — score the resulting trajectory, feed back to the surrogate.

**Verdict: GOAT (not Super-GOAT).** Honest reasoning below (§3).

---

## 1. Paper Core Findings

### 1.1 The mechanism (§4, verified by full read)

For a generative model `g_θ` and prompt `q`, greedy-decode the first token `w^(1)`, take its embedding `z ∈ R^D`. Treat `z` as the prior "correct start" and **perturb it**:

```
x_i = z + σ·ε_i,    ε_i ~ N(0, I),    i = 1..k
```

Each `x_i` is added to the vocab as a special token placed at the **end of the prompt** (paper Table 6: "Last" placement dominates). Because decoding after `x_i` is **greedy**, `x_i` one-to-one determines the entire output `y_i := w^(1..L)_i`. So `x` is a deterministic control knob for the trajectory.

The objective over perturbations is

```
f(x) = r_verifier(y) + r_coherence(y)
r_verifier(y) = 1{y_v == y}            // Multi-Generate: verifier regenerates the answer; match ⇒ correct
r_coherence(y) = Σ log P(w^(i))        // token-log-prob fluency
```

Bayesian optimization fits a Gaussian-kernel posterior `f(x) | f(x_{1:k}) ~ N(μ_k(x), σ_k²(x))` and selects `x_{k+1} = argmax EI_k(x)` where the **Expected-Improvement** has a closed form (Eq. 3, integration by parts):

```
EI_k(x) = (μ_k(x) − f*_k)_+ + σ_k(x)·φ(z) − (μ_k(x) − f*_k)·Φ(z),  z = (μ_k(x) − f*_k)/σ_k(x)
```

with `φ/Φ` the standard-normal pdf/cdf. An **adaptive EI** (Appendix A.4) scales `σ_k` by `ω_k = √(γ_k + 1 + ln(1/δ))` to absorb verifier noise, and Theorem A.1 gives sublinear regret `R_T = O(γ_T √T)`.

**Dimension reduction (Appendix A.5, Theorem A.2):** for `D = 768..8192`, BO collapses. Random projection `g(u) = f(Au)` with `A ∈ R^{D×d}`, `d ≈ 50`, recovers the effective dimension. Figure 6 shows `d = 50` is the sweet spot — variance is stable across 50 random projections (Figure 7).

### 1.2 What ships in the paper and what doesn't

- **Ships (open source):** the BO loop, the EI closed form, the dim-reduction projection, the four verifier prompts (Single-Judge / Multi-Judge / Single-Generate / Multi-Generate, Appendix B.6).
- **Headline results:** Table 1 — `+5pp` on GSM8K zero-shot vs best baseline across LLaMA-3-8B / Qwen2-7B / Mistral-7B / Qwen2-70B. Table 3 — `6.19%` of RAP's input tokens, `63.28%` of RAP's output tokens, `14.3%` of RAP's time.
- **What doesn't ship in the paper:** multi-token extension (Table 5: accuracy collapses past k=5 first tokens — `52%` at k=20 vs `79%` at k=1). The mechanism is **single-token-perturbation**, not trajectory-shaped.

### 1.3 Mechanistic claim (§5.2, Figure 3–4) — the part that actually transfers

The perturbation activates 3–4% more MLP neurons per layer than SC sampling (Figure 3), and the **critical-neuron activation rate climbs monotonically across BO iterations** (Figure 4). The targeted-masking experiment drops accuracy `62% → 13%` for critical neurons vs `62% → 42%` for random — confirming causal role. **This is the empirical justification that latent perturbation beats token-level temperature**: it touches neural pathways temperature cannot reach.

### 1.4 What does NOT transfer

- **LLM-text decoding specifics** (CoT prompts, GSM8K, Multi-Generate prompt template). Our target is per-NPC latent state, not text generation.
- **Verifier = same model** — for NPCs the verifier is `CLR`/`ConstraintPruner`/`claim verifier`, not a re-decode.
- **Multi-token optimization** (Table 5) — known to fail; we will not pursue.

---

## 2. Distillation — fuse, don't direct-map

### 2.1 Vocabulary translation (paper ↔ codebase)

| Paper term | Codebase term(s) | Where it ships |
|---|---|---|
| Bayesian optimization / EI / GP surrogate | UCB1 / P-UCB / Thompson sampling | `PrudentBanker` (098), `SketchSampler` P-UCB (039), `CuratorBandit` Thompson (`curator.rs:295`), `FeedbackBandit` (`bench_310_t33`) |
| First-token embedding as controllable latent variable | latent prefill adapter, domain-latent injection | `Latent Field Steering` `apply_latent_steering` (R290, Plan 309) |
| Gaussian noise injection `x = z + σε` | K-query perturbation `q_k ~ N(0, σ²I)` | BoM `sample_k_states` (Plan 281), `MicroRecurrentBeliefState` (Plan 276), curiosity pulse (041) |
| Random projection dim-reduction `g(u) = f(Au)` | spectral basis, subspace projection | `NeuronShard::semantic_axes` (SVD), SpectralQuant eigenbasis (039) |
| Verifier reward `r_verifier` (Multi-Generate) | CLR Multi-Generate, claim verifier | CLR (Plan 284 / R255), `ConstraintPruner::is_valid` |
| Coherence `r_coherence = Σ log P(w)` | belief residual, coherence `tau_reest` | NextLat residual `ĥ = f(h,x) + h` (R192), `latent_functor/reestimation.rs` |
| Adaptive EI noise scaling `ω_k` | conformal-naive floor, calibrated UQ | Plan 340 `ConformalIntervalCalibrator`, R322 |
| Critical-neuron activation | neuron attribution | CNA (053), depth-invariance `classify_chain` (286) |

### 2.2 Latent-space reframing (mandatory)

Re-cast the paper as a **latent-to-latent op on HLA / `latent_functor` / DEC state**. The acquisition function does **not** operate on decoded tokens — it operates on the **perturbation vector** `δ ∈ R^d_latent` (HLA: d=8; latent_functor zone-gating direction; DEC cochain channel).

```
δ* = argmax_δ  EI_k(δ)
     s.t.   s'_i = s + sigmoid(α·A·δ_i)          // sigmoid gate per AGENTS.md constraint #2
            y_i  = forward(s'_i, x_t)             // NPC forward pass — single, batched across K
            r_i  = CLR.verify(y_i) + coherence(y_i)
            posterior_k+1 = GP_update(δ_{1:k}, r_{1:k})
```

Crucial design points (vs the paper):
- **`sigmoid` not softmax** for the projection `δ → s'` (constraint #2; the paper adds raw `σ·ε`, we add `sigmoid`-gated `α·A·δ` to bound the perturbation in `[-1, 1]` HLA range).
- **Random projection `A` lives in a `LatentSubspace` Pod**, BLAKE3-committed (freeze/thaw-compatible) — not regenerated each call. Matches the paper's Figure 7 stability (50 random `A`s give negligible variance).
- **`GP_update` is the EI surrogate**: Gaussian kernel `k(δ_i, δ_j) = exp(-‖δ_i−δ_j‖² / 2ℓ²)` on the low-dim subspace. d=8 (HLA) or d=16 (DEC cochain) — well inside the paper's "d ≤ 50" sweet spot, so the curse-of-dimensionality caveats don't bite.
- **Verifier is CLR Multi-Generate** (the paper's best verifier strategy) over the K candidate trajectories — already shipped infrastructure.

This reframing gives the **Super-GOAT factory module angle**:
- **HLA (`sense/`):** explore per-NPC belief-state perturbations. A frightened NPC can Bayesian-explore which latent valence-arousal combination yields the best survival trajectory.
- **`latent_functor/zone_gating.rs`:** explore which zone-gating direction maximizes a "did I reach the goal zone" verifier reward — runtime zone-attention discovery.
- **DEC (`dec/`):** the `f*_k` incumbent in EI is naturally a Hodge-harmonic scalar (the harmonic component is "what survives perturbation"), so EI explores the exact/coexact channels. **Genuinely novel angle:** EI on the coexact (solenoidal) channel is a modelless analog of "explore circulation patterns".
- **`cgsp_runtime/`:** curiosity pulse (041) becomes the **exploration term** in EI (the `σ_k(x)·φ(z)` term), regret-bounded — replaces the heuristic curiosity schedule with a principled one.

### 2.3 Closest existing cousins (fusion-protocol record)

| Cousin | What it ships | What's missing vs Soft Reasoning |
|---|---|---|
| **Latent Field Steering (R290, Plan 309)** | Direction-vector injection into latent state, designer-top-down | No acquisition function; no verifier reward loop; one-shot, not closed-loop. **Strongest cousin.** |
| **BoM Sampler (R248, Plan 281)** | K Gaussian queries, single-pass, batched | No EI/UCB acquisition — random `q_k ~ N(0,σ²I)`; no posterior update across iterations |
| **PrudentBanker / SketchSampler (098 / 039)** | P-UCB / Thompson acquisition over a **population of arms** | Acquisition is over arm IDs, not over a continuous latent-perturbation vector |
| **Salience Tri-Gate (R281)** | 3-way Speak/Silent/Delegate decision | Emit decision, not exploration; no acquisition |
| **Sigmoid-Graded Reject (310 T1)** | Soft reject with retry | No acquisition, single retry direction |
| **CLR Multi-Generate (R255, P284)** | Multi-candidate verify | Pure verifier — no generation-side perturbation |
| **Viable Manifold Graph (R294)** | Discrete safe-manifold navigation | Discrete graph, not continuous BO |

**The gap each cousin leaves open:** none of them **closes the loop** — perturb → forward → verify → surrogate-update → next-perturbation. The fusion is what's novel.

### 2.4 Fusion — Soft Reasoning × Latent Field Steering × BoM × CLR

**The combination (primary fusion target):** add an `explore_latent` method to the existing `LatentField` (R290) or `MicroRecurrentBeliefState` (P276):

```
pub fn explore_latent<V: Verifier>(
    &self,
    initial: &BeliefState,
    context: &Context,
    verifier: &V,
    budget: ExplorationBudget,         // K queries per round, max rounds, ε-convergence
    subspace: &LatentSubspace,          // frozen BLAKE3-committed projection A
) -> ExplorationOutcome {
    // Round 0: sample K perturbations, evaluate, build GP posterior
    // Round k: EI-max pick next K perturbations, evaluate, update posterior
    // Converge: |f*_k − f*_{k-1}| < ε
}
```

**What this produces that none of the cousins alone can:**

| Incumbent alone | What it can't do | What the fusion adds |
|---|---|---|
| Latent Field Steering | Designer must hand-pick the direction vector | NPC **discovers** its own optimal direction vector via verifier feedback |
| BoM Sampler | Random queries — no regret bound, no convergence | Regret-bounded exploration (Theorem A.1: `R_T = O(γ_T √T)`) |
| CLR Multi-Generate | Verifies already-generated candidates | Closes the loop — verification **drives** the next generation |
| Curiosity pulse (041) | Heuristic exploration schedule | **Principled** exploration schedule from the EI acquisition |

**Capability increment:** runtime NPC **self-directed latent exploration with regret-bounded convergence**. Today NPCs explore via designer-set curiosity schedules (041) or random BoM sampling (281). With this, an NPC can run `~2–4` BO rounds per decision (paper Table 2) and converge on a latent state that maximizes its own verifier (e.g., "did I survive?" / "did I reach the goal?").

---

## 3. Verdict

### Tier: **GOAT (not Super-GOAT)**

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | Open primitive + private guide + plans |
| **GOAT** | Provable gain over existing approach, not a new class. Promotes to default if it wins | Plan + implement, feature flag + benchmark |
| **Gain** | Incremental improvement | Plan only, behind feature flag |
| **Pass** | Not relevant | One-line note |

**One-line reasoning:** The closed-loop verifier-guided Bayesian latent exploration is a genuinely novel **mechanism** (no shipped prior art for the loop — see §2.3), but it is **not a new capability class**: each of its three components (BoM perturbation, bandit acquisition, CLR verify) is already shipped, and the fusion is a tighter combination rather than something no competitor could assemble. Promotes to default only if GOAT gate beats the existing BoM + curiosity-pulse baseline on regret-bounded exploration efficiency.

### Why NOT Super-GOAT (honest demotion)

The novelty gate Q1–Q4 fails on Q2 (new class of behavior):

- **Q1 No prior art? YES.** The three-layer check (notes + code + vocabulary translation, §2.3) confirms no shipped closed-loop verifier-guided BO over latent perturbations. The **mechanism** is novel.
- **Q2 New class of behavior? NO.** A competitor with BoM (Plan 281) + CLR (Plan 284) + UCB1 bandits (PrudentBanker) could approximate the loop by chaining them sequentially. Soft Reasoning's contribution is the **EI closed-form + regret bound** — a tighter integration, not a capability no incumbent can match. The paper's own framing ("[our method] is able to control and optimise the reasoning without accessing model parameters") describes a *better optimization*, not a new task.
- **Q3 Product selling point? PARTIAL.** "NPCs self-discover optimal latent states through verifier-guided exploration" is a selling point, but it overlaps with the existing curiosity-pulse selling point (R041, R240). Not a clean increment.
- **Q4 Force multiplier? YES.** Touches HLA + latent_functor + CLR + conformal-UQ floor + DEC (Hodge-harmonic-incumbent reframe). ≥5 systems.

Failing Q2 ⇒ GOAT per the skill's rule ("If YES to all 4 → Super-GOAT"). No `riir-ai/.research/` guide required (the Super-GOAT anti-deferral rule does not trigger because we do not write "all 4 YES").

### What the GOAT gate must prove (before promotion to default)

The UQ-bearing-primitive GOAT-gate extension ("Report the Floor" rule, adopted 2026-06-28 per R322 / Plan 340) **applies**: EI claims a calibrated posterior `N(μ_k, σ_k²)`. **Mandatory baselines:**

1. **Conformal-naive floor** (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, m=1, plain split conformal) on CRPS / coverage / Winkler score. If Soft Reasoning's EI cannot beat the floor, the GOAT gate FAILS — it's the floor in disguise.
2. **BoM-with-UCB1 ablation** (single-pass K=20 BoM sampling + UCB1 selection across rounds, no GP surrogate). This isolates whether the GP posterior + EI closed form adds value over the cheaper bandit baseline.
3. **Curiosity-pulse-only baseline** (current default — heuristic schedule). G3 no-regression: replacing the curiosity schedule with EI must not regress crowd-scale NPC survival / goal-reach.
4. **Latency gate G2**: paper's `~2–4 BO rounds × K=5 queries × forward` = `10–25 forwards per NPC decision`. **At 20Hz tick this is borderline-infeasible** for thousands of NPCs — the GOAT plan must scope this to **offline / cold-tier / single-hero-NPC** decisions, NOT the per-tick crowd hot path. This is the headline risk.

### Plan-only routing (no Super-GOAT guide)

- **`katgpt-rs/.plans/348_latent_bayesian_exploration_primitive.md`** (open) — opt-in feature flag `latent_bayesian_exploration`, ~150 LOC over `LatentField` / `MicroRecurrentBeliefState`:
  - `EI<Subspace>` struct with closed-form `EI_k(δ)` (Appendix A.4 adaptive variant)
  - `GaussianProcessSurrogate` (kernel `exp(-‖·‖²/2ℓ²)`, posterior update)
  - `explore_latent<V: Verifier>(initial, context, verifier, budget, subspace) -> ExplorationOutcome`
  - BLAKE3-committed `LatentSubspace` (random projection `A`, frozen at init)
  - GOAT gate benchmarks (1–4 above) + conformal floor comparison
- **No private `riir-ai/.research/` guide** — verdict is GOAT, not Super-GOAT.
- **No `riir-chain` / `riir-neuron-db` cross-ref** — pure inference-time latent op, no chain commitment, no shard storage.

---

## 4. What this note prevents (canonical failure modes averted)

1. **Paper-vocabulary-only grep would miss prior art.** The paper says "Bayesian optimization / Expected-Improvement"; we ship UCB1/P-UCB/Thompson under "bandit" vocabulary. Grepping only paper terms ⇒ false "no prior art" → false Super-GOAT. The §2.1 crosswalk is the defense.
2. **False "novel capability" over the BoM + bandit + CLR stack.** Each component ships; the closed loop is the only novel piece, and a competitor could chain the components sequentially. Honest Q2 = NO demotes to GOAT.
3. **Perf-gate skipping.** The paper runs at `~15–25 forwards per query` — fine for LLM benchmarks, **infeasible at 20Hz tick**. The GOAT plan must scope to offline/cold-tier decisions, and the latency gate is mandatory.
4. **UQ claim without the conformal floor.** EI's posterior is a calibrated-uncertainty claim → the "Report the Floor" rule (R322 / Plan 340) applies from initial gate. Recorded in §3 so the plan does not skip it.

---

## TL;DR

Soft Reasoning (arXiv:2505.24688) is closed-loop verifier-guided Bayesian optimization (Expected-Improvement, GP surrogate, dim-reduced random projection) over the first-token embedding, converging in 2–4 iterations and beating SC/FIRE/CoT-Decoding/RAP at 6–14% of their token/time cost. **Verdict: GOAT, not Super-GOAT** — the mechanism (closed loop of BoM-perturbation × bandit-acquisition × CLR-verify) is novel (no shipped prior art for the loop; verified by three-layer check + vocabulary crosswalk §2.3), but it is **not a new capability class** over the existing BoM + bandit + CLR stack — failing novelty gate Q2 demotes to GOAT. Plan-only, opt-in `latent_bayesian_exploration` feature, with mandatory conformal-naive-floor comparison (EI claims a calibrated posterior — the UQ-floor rule applies), mandatory BoM+UCB1 ablation, and a **latency-gate-mandated offline/cold-tier scope** (the paper's `~15–25 forwards/query` is infeasible at 20Hz tick). No Super-GOAT guide required.
