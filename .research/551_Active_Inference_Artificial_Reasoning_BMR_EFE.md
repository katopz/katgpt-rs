# Research 551: Active Inference and Artificial Reasoning — BMR + EFE-over-Models

> **Source:** "Active inference and artificial reasoning" — Friston, Da Costa, Tschantz, Heins, Buckley, Verbelen, Parr. Nature Communications 2026 (published 2026-09-08), DOI [10.1038/s41467-026-77209-5](https://doi.org/10.1038/s41467-026-77209-5); arXiv:2512.21129 (2025-12-24).
> **Date:** 2026-09-12
> **Status:** Active — distilled; open primitive filed as [Plan 597](../.plans/597_bmr_efe_model_gain.md); game-runtime guide at riir-ai `.research/376`; fusion issue riir-ai `.issues/925`.
> **Related Research:** 478 (MOP — the anti-FEP ledger this note must engage), 543 (HMM control), 240 (CGSP), 334 (sleep-time), 336 (best_belief); riir-ai 341 (EVPI gate), 370 (homeostat).
> **Classification:** Public — the distilled math is closed-form Bayesian model selection, no game semantics. The selling-point guide is private (riir-ai 376).

---

## TL;DR

The paper extends active inference from states/parameters to **structure**: the expected free energy gains an **expected information gain over MODELS** term — the KL between the posterior over rival generative models *with* vs *without* the anticipated outcome of an action — evaluated in closed form via **Bayesian Model Reduction** (Beta-function ratios over accumulated Dirichlet counts). Commitment is an **Occam's razor** threshold (log Bayes factor of best model vs rest > 16 nats), after which the agent replaces its posterior with the selected model and suppresses novelty (counts ×512). On the three-ball rule-discovery paradigm, active model disambiguation cuts discovery time 28.5 → 19.4 trials and lifts score 89.2 → 150; without any info-gain term ~half of runs never discover the rule in 64 trials.

**Distilled for katgpt-rs (modelless, inference-time):** the entire mechanism is closed-form (digamma/lgamma/Beta functions on Dirichlet count tensors — zero gradient descent anywhere). Two components have **zero shipped analog** workspace-wide: (1) BMR closed-form model evidence, (2) the EFE info-gain-over-models term. Both are modelless inference primitives for `katgpt-core`. The published implementation landscape is MATLAB (SPM), Python (pymdp), Julia (ActiveInference.jl/RxInfer) — **Rust is an empty niche** (GitHub: 0–1★ hobby repos only).

---

## 1. Paper Core Findings

### 1.1 Three unknowns, three timescales

Active inference distinguishes **latent states** (inference, fast), **parameters** (learning, slow — Dirichlet count accumulation), and **structure** (selection, post-hoc — BMR + Occam commit). Selection is *deliberately* post-hoc: evidence accumulates under a uniform prior over the hypothesis space; the posterior over models is evaluated offline from accumulated counts; a discrete commit (Eq 12–13) approximates continuous structure learning (appendix proves the post-hoc scheme lower-bounds full model inference via Jensen on the concave digamma).

### 1.2 The new term: expected information gain over models (Eq 10–11)

EFE already decomposes into risk + ambiguity (states) + novelty (parameters). The paper adds:

```
G_models(u) = E_Q(o|u) [ KL( Q(m | s,o,u) ‖ Q(m | s,u) ) ]
```

The predictive posterior over models is computed **without re-running inference per model**, via BMR on the accumulated counts `a` and the anticipated count increment `Δa = ŝ ⊗ ô` (expected state ⊗ anticipated outcome):

```
ln P(m | o)  ∝  Σ_cols [ ln B(ã_c) + ln B(a_c) − ln B(ã_m,c) − ln B(ã_m,c + a_c − ã_c) ]     (Eq 7/9)
ln B(x)     =   Σ_i ln Γ(x_i) − ln Γ(Σ_i x_i)
predictive:  same with a → a + Δa                                                              (Eq 11)
```

where `ã` = full prior counts, `a` = accumulated posterior counts, `ã_m` = the reduced prior defining model m (shrinkage: counts zeroed or set to a small value like 1/32). **Sparsity is the perf story**: `Δa` touches exactly one column (the anticipated (state, outcome) pair), so the incremental per-action evaluation is O(#models) column-ratio recomputes, not O(#columns × #models).

### 1.3 Occam's razor commit (Eq 12–13)

```
log_bayes_factor = ln [ P(m*|o) / (1 − P(m*|o)) ]      (m* = argmax P(m|o))
commit when log_bayes_factor > 16 nats  →  posterior ← selected model's average;
optionally multiply its Dirichlet counts by 512 to suppress novelty (exploitation mode).
```

### 1.4 Model-space generation (Eq 14)

Candidate rules are generated automatically from the **factorial structure** of the (known) generative model: identify a controllable *choice* factor sharing annotated states with non-controlled *criterion* factors; candidate *context* factors whose levels uniquely specify the criterion. Rule = "if context state X then criterion state Y is the correct choice". For three-ball: 3 × 3³ = **81 hypotheses** (79 unique). This is probabilistic inductive logic programming (PILP) flavor — structure priors from symmetry/isomorphism constraints.

### 1.5 Results (contribution analysis)

- Info gain over models+params: discovery 28.5 → **19.4** trials, score 89.2 → **150** (64 paired sims).
- States-only info gain: fails to disambiguate two plausible hypotheses after 32 trials.
- No info gain (random): Occam ≈ 0 at trial 32; ~half of sims never discover in 64 trials.
- Failure mode: **jumping to conclusions** — 1/64 sims committed to the wrong model at trial 6 (threshold too low for that evidence path). The paper notes the schizophrenia-research reading of this dial.
- The "aha moment" is measurable: policy-posterior precision (negative entropy) spikes at commit; dopamine analog.

---

## 2. Distillation

### 2.1 Path 0 component table (coverage × extraction)

| Paper component | Shipped analog (verified file-level) | Verdict |
|---|---|---|
| EFE epistemic term over **states** (salience/ambiguity) | CuriosityPulse (`cgsp_runtime`), fog curiosity (`riir-games-quest/quest/curiosity.rs`), EVPI gate | exists |
| EFE novelty term over **parameters** | `delta_mem` surprise-gated δ-rule; `best_belief` Beta-Bernoulli; bandits | partial (Beta yes; Dirichlet-tensor no) |
| **EFE info gain over models** (Eq 10) | — | **no analog → open primitive** |
| **BMR closed-form evidence** (Eq 7/9) | — (`dirichlet.rs` is Dirichlet *energy*, a name trap) | **no analog → open primitive** |
| Occam commit statistic (Eq 12) | `contrastive_scope` EvidenceTier (log-odds, tier demotion); freeze/thaw commit machinery | partial (statistic absent, commit machinery exists) |
| Model-space generation (Eq 14) | quest grammar KG — authored + frozen, not generated | absent (by design, game-side) |
| Sleep/offline arbitration timing | Raven/δ-Mem sleep cycle; `arg_runtime/offline.rs` (default-on) | exists (timing), absent (arbitration semantics) |
| Confidence/precision readout | MCTS/argmax policy posteriors | partial |

**Path 0 verdict: MODELLESS-VALIDABLE.** No component requires gradient descent; every formula is digamma/lgamma arithmetic on count tensors. The two no-analog rows are the strongest finding class → `katgpt-core` open primitives (Plan 597).

### 2.2 Signal-diff on the partial rows (§3.6 discipline)

- **`best_belief.rs`** (Beta-Bernoulli ε-quantile LCB): consumes win/fail counts for a *single* hypothesis's track record. BMR consumes the *joint* accumulated Dirichlet posterior and scores *rival structures* against each other via shared counts. Different signal: per-hypothesis history vs cross-model evidence ratios. Not covered.
- **`contrastive_scope` EvidenceTier**: two-corpus log-odds scope score gating *one* rule's applicability. BMR arbitrates N rival models. Scope gating ≠ hypothesis arbitration. Not covered (the log-odds *shape* is shared; the *object* differs).
- **Raven/δ-Mem sleep**: consumes event embeddings → produces weight deltas (compression). Paper's sleep arbitrates hypotheses via BMR on counts. Different signal entirely; the timing substrate is reusable, the semantics are not. Not covered.
- **EVPI gate (`evpi_gate.rs`)**: consumes the NPC's decision fn + plausible-set extremes of stale *state* unknowns; emits re-observe actions for flips. The paper's term consumes a *model posterior* and emits disambiguating probes. **Complementary, and the composition point**: a model-EVPI ("would resolving *which model* flip the decision?") is literally Eq 10 dressed in our gate vocabulary. Note `evpi_gate.rs` already cites Friston 2015 and names itself the epistemic/pragmatic split.

### 2.3 Fusion (paper × shipped substrate)

- **F1 — P0, katgpt-core (Plan 597):** `bmr` module — `ln_beta`, per-column BMR log-evidence (Eq 7/9), posterior over models, predictive variant with sparse `Δa` (Eq 11), Occam log Bayes factor (Eq 12), and the generic Eq-14 model-space enumerator. Feature `bmr`, default-off until GOAT.
- **F2 — P1, riir-ai (guide 376 / issue 925):** **scientist NPC** — extend `evpi_gate.rs` with model-posterior flips; Dirichlet count state accumulates per-NPC (latent, Warm tier); sleep-cycle BMR arbitration in `arg_runtime/offline.rs`; Occam commit → freeze (selected rule persisted with boosted counts). Exploitation stays MOP/HMM-control (see 2.4).
- **F3 — P2, riir-clippy (recorded):** healer hypothesis disambiguation — the rule corpus *is* a model space; heal-span selection by expected info gain over which rule applies; Occam commit = corpus promotion. Wiring deferred until F1 lands (fusion-priority ladder: game #1, healer #2 — recorded here so it is not re-derived).
- **F4 — P2, riir-neuron-db (recorded):** sleep arbitration pass over Raven slots (counts already accumulate); **crowd-pooled evidence** — 1000 NPCs merging zone-level Dirichlet counts via committed shard merges. The paper is single-agent; pooling is our delta.

**Game-context reframe:** the per-entity scalars are *discovery time* and *commit confidence*; they drive epistemic foraging cadence (which experiment this tick), consolidation window (when sleep arbitrates), and personality (Occam threshold divergence — a cautious NPC uses >16 nats, an impulsive one jumps to conclusions: the paper's failure mode becomes a temperament dial). **Consumer reframe (healer):** rule ↔ code-shape binding corpus + oracle labels = the feedback modality; spans chosen to disambiguate rules = active data selection over the corpus.

### 2.4 Anti-FEP reconciliation (load-bearing — read before touching this)

The workspace has a standing **anti-FEP ledger**: Research 478 adopted MOP over EFE for behavioral variability; Research 543 adopted exact HMM control over variational free-energy loops; Research 370's incumbent table marks FEP "avoid". **Those rejections govern the exploitation/control axis** (policy extraction in known/frozen models — where EFE collapses to deterministic policies). This distillation adopts the **discovery axis only**: EFE-over-models for epistemic foraging under unknown likelihood structure, a job with no incumbent (MOP needs a frozen kernel; HMM control a frozen HMM; EVPI a known decision fn). The paper's own timescale separation licenses exactly this split: after the Occam commit, planning reverts to the committed model — where MOP/HMM-control remain the policy extractors. Do not cite Research 551 as "we adopted EFE" without this paragraph.

### 2.5 Name traps (grep hygiene)

- `katgpt-core/src/dirichlet.rs` = Dirichlet **energy** (graph-Laplacian smoothness), not the distribution.
- Plan 268's "active inference" = inference-engine scheduling sense, not Friston.
- "Sleep" in `sleep_time/` (riir-ai) = anticipator substrate; the BMR arbitration seam is `arg_runtime/offline.rs` + Raven.

---

## 3. Verdict

**Super-GOAT (fusion-class).**

| Gate question | Answer |
|---|---|
| Q1 prior art | Core math = the group's established 2015–2026 line (EFE 2015; three-ball/insight 2017; BMR 2018; discrete synthesis 2020; supervised structure learning 2023; AXIOM 2025 is the closest game-side prior). The **composition** — BMR/EFE-over-models × EVPI decision-flip gating × Raven sleep arbitration × freeze/commit × crowd pooling — has no published match (verified: headline + component + framing searches; Rust niche empty). Novelty is the fusion, not the equations. |
| Q2 new behavior class | NPCs that design experiments to discover *rules* and commit during sleep. Shipped curiosity resolves state/parameter uncertainty; nothing resolves structure uncertainty. |
| Q3 selling point | "Our NPCs figure out how your world works by experimenting — measurably sample-efficient, with visible aha moments" (paper-measured analog: 19.4 vs 28.5 trials; ablation: no info-gain ⇒ half never discover). |
| Q4 force multiplier | Connects EVPI gate + CuriosityPulse + Raven/δ-Mem + freeze/thaw + quest grammar (hypothesis-space authoring) + vibe KG (belief triples) + healer corpora. ≥2 pillars trivially. |

**MOAT gate:** katgpt-rs gets the generic closed-form math (fundamental/principle ✓, no game semantics); riir-ai gets the runtime guide (376); riir-clippy consumer recorded (2.3 F3); riir-neuron-db pooling recorded (2.4 F4). Selling-point doc lives private-side per the commercial strategy.

**Honest caveats:** (1) this is a port + composition of published math, not new theory — the moat is the integration and the Rust/modelless/20Hz engineering; (2) the paper's numbers come from their priors/setup — our gate must reproduce the *qualitative ablation* (info-gain arm wins, no-info-gain arm fails), not clone their trial counts; (3) premature-commit (1/64) is real and becomes a tunable, not a bug.

---

## 4. Prior-art ledger (verified via Crossref/arXiv/GitHub/crates.io)

- **EFE foundations:** Friston et al. 2015 "Active inference and epistemic value" (Cogn Neurosci); 2016 "Active inference and learning"; 2017 "A process theory"; Parr & Friston 2019; Da Costa et al. 2020 "Active inference on discrete state-spaces: a synthesis" (JMP 99:102447 — the algorithmic spec); Friston et al. 2021 "Sophisticated Inference"; Millidge, Tschantz, Buckley 2021 "Whence the Expected Free Energy?"; Friston et al. 2023 "Path integrals, particular kinds, and strange things".
- **BMR:** Dickey 1971 (Savage–Dickey); Friston & Penny 2011 "Post hoc Bayesian model selection"; Friston, Parr, Zeidman 2018 "Bayesian model reduction" (arXiv:1805.07092) — already framed as statistical structure learning.
- **Structure-learning line (same group):** bioRxiv 2019 concept learning; "Supervised structure learning" 2023 (arXiv:2311.10300 — EFE-weighted model selection predates 2026); Da Costa et al. 2025 "Possible Principles for Aligned Structure Learning Agents" (Neural Comput); **Heins et al. 2025 AXIOM** (arXiv:2505.24784 — BMR structure refinement inside games, ~10k steps; closest published prior).
- **OED lineage:** Lindley 1956; MacKay 1992; Chaloner & Verdinelli 1995; Parr, Friston, Zeidman 2024 "Active Data Selection and Information Seeking" (Algorithms 17:118); Tsividis et al. 2021 (theory-based RL exploration).
- **Insight/sleep:** Friston et al. 2017 "Active Inference, Curiosity and Insight" (Neural Comput 29:2633–2683 — the three-ball paradigm's origin; 2026 makes the data-gathering active); Hinton et al. 1995 wake-sleep.
- **Implementations:** SPM (MATLAB, the paper's reference); pymdp (Python, JOSS 2022, 734★); ActiveInference.jl (Entropy 2025); RxInfer.jl (Julia, 413★). **Rust: none established** — GitHub `active inference language:Rust` = 14 repos, all ≤3★, dormant; crates.io: one marginal `symthaea-fep` v0.1.0.
- **EFE ↔ curiosity:** Biehl et al. 2018; Schwartenbeck et al. 2019; Pezzulo & Friston 2019; ICM/RND/VIME on the RL side. The paper's taxonomy delta: curiosity over **structure** as a third kind (states/parameters/models).

---

## 5. Implementation sketch (consumed by Plan 597)

```rust
// crates/katgpt-core/src/bmr.rs  (feature = "bmr", default-off)
// All log-space, f64 internally, zero-alloc hot paths, fixed-size columns.

pub fn ln_beta(x: &[f64]) -> f64;                       // Σ lnΓ(x_i) − lnΓ(Σ x_i)
pub fn ln_beta_delta(base: &ColumnSums, col: usize, delta: &[f64]) -> f64;  // cached-Σ incremental

/// Eq 7/9 — per-column BMR log-evidence of reduced model m vs full model.
/// Counts are per-column Dirichlet parameters (likelihood tensor columns).
pub fn bmr_log_evidence(prior: &Counts, post: &Counts, reduced_prior: &Counts) -> f64;

/// Eq 9 — posterior over the model space (uniform model prior).
pub fn posterior_over_models(models: &[Counts], post: &Counts, out: &mut [f64]);

/// Eq 11 — predictive variant: post + Δa where Δa = ŝ ⊗ ô (ONE column touched).
/// O(#models) after O(#cols) init — the sparse-Δ trick.
pub fn predictive_model_posterior(models: &[Counts], post: &Counts,
                                  state: usize, outcome: usize, out: &mut [f64]);

/// Eq 10 — expected info gain over models for candidate action u:
/// E_{Q(o|u)} [ KL(Q(m|·,o,u) ‖ Q(m|·,u)) ] — sum over anticipated outcomes.
pub fn efe_model_gain(/* action-anticipated (state,outcome,prob) triples */) -> f64;

/// Eq 12 — Occam commit statistic.
pub fn occam_log_bayes_factor(model_posterior: &[f64]) -> f64;   // ln p*/(1−p*)

/// Eq 14 — generic model-space enumerator over choice/criterion/context factors.
pub fn enumerate_isomorphic_rules(factors: &FactorLayout) -> Vec<Counts>;
```

Needs a modelless `ln_gamma` (Lanczos, ~30 LOC — statrs is dev-dep only). Numerical care: log-space only, never exponentiate ratios; counts reach 512× post-commit so f64 sums with cached column Σ. The softmax over policies (Eq 4) stays a genuine categorical posterior (house sigmoid rule applies to scalar projections, not categorical sampling).

**GOAT gate shape (full spec in Plan 597):** G1 = BMR matches brute-force marginal-likelihood enumeration on small tensors (fp-tolerance) + three-ball qualitative ablation reproduction; G2 = sparse-Δ eval ≥100× vs naive full recompute on the 81-model space, µs-scale; G3 = feature-flagged, no regression; G4 = alloc-free hot path. **UQ floor rule: considered, not applicable** — this is a model-selection belief, not a predictive interval/coverage claim; the honest gate is discovery-rate + premature-commit-rate vs the paper's ablation, not CRPS/Winkler.
