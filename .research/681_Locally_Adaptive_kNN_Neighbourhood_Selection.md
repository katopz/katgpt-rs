# Research 681: Locally Adaptive kNN Neighbourhood Selection (Lepski cascade)

> **Source:** [arXiv:2611.22198v1](https://arxiv.org/abs/2611.22198) — *Training Neural Networks to be Locally Adaptive*, Banerjee & Sarkar (NUS / ISI), 2026-11-17, math.ST.
> **Date:** 2026-11-18
> **Status:** Active — verdict GOAT-tier (selection layer, PoC-gated); filed as poc issue per the poc-first law, promotion talk after the gate runs.
> **Related Research:** 549 (ASEntmax — the sibling closed-form adaptive-k arm, `compute_derived_k`), 392 (SSMax rolling Δ̂ — the level-gap statistic the derived arm consumes)
> **Related Plans:** 256 (MSA adaptive k — the shipped sigmoid arm)
> **Cross-ref:** riir-train `.issues/701` @ `11f52889` (training-track probe cells) · riir-ai `.issues/1239` @ `e4c6cc75a` (DEC-field agreement fusion, novelty TBD) · this note + katgpt-rs `.issues/1229`: commit `4fcc5013c`
> **Classification:** Public

---

## TL;DR

For each query, fit the model on only the k nearest neighbours of the query — the training
sample AND the fitted function become query-dependent — and pick k per query with a **Lepski
agreement cascade**: scan a geometric grid upward, keep the largest k whose estimate still
agrees with every smaller-grid estimate within a **computable** noise scale τ̂_k = σ̂·√(log n/k),
where σ̂ is a nearest-neighbour MAD. Theory: pointwise risk n^(−2α/(2α+d*)) on a d*-manifold is
pointwise minimax optimal, the adaptive rule attains it WITHOUT knowing the local Hölder
exponent α_x or the intrinsic dimension d*, and the log factor is priced by a lower bound
(Lepski–Spokoiny) — zero empirical tuning knobs.

**Distilled for katgpt-rs (modelless, inference-time):** the adaptivity is a **selection layer,
separable from the estimator**. The paper's per-query BFGS-trained micro-network is incidental —
the theory only needs the bias-grows-with-k / noise-decays-with-k profile that every modelless
kNN aggregation (vote, kernel mean, fused score) already has. What transfers: the agreement
cascade, the MAD σ̂, the oracle inequality as a *testable property gate*, and the kNN-radius
auto-adaptation design law. Consumes nested neighbourhoods = **prefixes of ONE top-K retrieval**
→ the cascade costs one retrieval + O(log n) scalar folds, zero-alloc friendly.

---

## 1. Paper core findings (pinned from the fetched full text, v1)

1. **Localized estimator.** For query x: fit a small ReLU net (class F(L,p,s), sparse, bounded
   Lipschitz K) on the k nearest neighbours N_k(x) only; predict f̂(x) = ĝ_x(x).
   Motivation (Doppler): global fit smooths away the oscillations near the origin; no single k
   is right everywhere — k=5 tracks the rough region but is noisy on the smooth part, k=20 the
   reverse; the localized best-k beats the global network 7.6× in median MSE.
2. **Localization theory.** Pointwise risk E|f̂(x) − f0(x)|² ≲ (k/n)^(2α/d*) + C_n/k; balancing
   gives k ≍ C_n^(d*/(2α+d*))·n^(2α/(2α+d*)) and rate n^(−2α/(2α+d*)) (up to logs) — **matching
   Le Cam lower bound** ⇒ pointwise minimax optimal, with d* (intrinsic) not d (ambient). The
   analysis handles the data-dependent subsample by conditioning on the covariates (the
   neighbourhood is selected without reference to the responses, so the noise stays i.i.d.)
   and a local Gaussian complexity G_n(δ; ∂F) with a critical radius from metric entropy:
   **δ̄²_k = 2048(s+1)/k · W(k(L+1)V²/(16(s+1)))** (Lambert W), computable from architecture alone.
3. **Lepski adaptive k (the actionable part).** Grid K_n = {k0, 2k0, …}, k0 = ⌈8 log(n/η)⌉.
   Select k̂(x) = max{k : |f̂_k(x) − f̂_ℓ(x)| ≤ A·τ_ℓ ∀ℓ ≤ k} with A ≥ 4 and
   **τ_k = 6√κ_σ·max{δ̄²_k, 2·log(|K_n|/η)/k}^(1/2)** — the stochastic scale DECAYS in k and is
   computable from architecture; the bias b_k(x) = C_M(L_F+3L_x)(k/n)^(α_x/d*) GROWS in k and is
   unknown. Oracle inequality: |f̂_ad(x) − f0(x)| ≤ (A+2)·τ_(k*_x) on a 1−2η event, where
   k*_x = max{k : b_k ≤ τ_k} is the oracle balance — the rule never stops before the oracle
   neighbourhood and pays at most a constant over it. Practical form: τ̂_k = σ̂·√(log n/k),
   **σ̂ = median_i |Y_i − Y_nn(i)| / (0.6745√2)** (nearest-neighbour MAD), sequential scan
   upward, stop at first rejection.
4. **Adaptation theorem.** |f̂_ad(x) − f0(x)|² ≲ (D_n/n)^(2α_x/(2α_x+d*))·(L_F+3L_0)^(2d*/(2α_x+d*))
   — adapts to the LOCAL exponent α_x and intrinsic d* simultaneously, neither known. The log
   factor in D_n is the unavoidable price of pointwise adaptation (Lepski & Spokoiny 1997 lower
   bound) — the schedule ships with no empirical tuning knob for it.
5. **Experiments (honest limits).** 1D denoising: local nets beat every global net tested;
   trend filtering often best overall; LDNN–Lepski selects the largest grid candidate 52–88% of
   the time in 1D (upper end often binding). Manifold (d*=3 in R^30): local nets beat global
   for Blocks/Bumps variants, LOSE for HeaviSine/Doppler variants; smooth activations beat
   ReLU on the manifold (opposite of 1D). n ≤ 10⁴, synthetic only.

## 2. Distillation

### 2.1 Path 0 inventory (training-target decomposition)

The value is the MATH (selection layer + rates), not the training loop (per-query BFGS fitting
is incidental — the paper's own baselines run the same rule with local-linear and kNN estimators).

| Paper component | Disposition | Where |
|---|---|---|
| Per-query local network fit | **Deferred** — structurally unaffordable on serving lanes (reflex G2 p99 ≤ 1 ms/decision-set; G4 zero-alloc; runtime weight mutation banned). Frozen per-cluster conversion = two pre-registered probe cells | riir-train `.issues/701` |
| Lepski agreement cascade (§4.4) | **Extracted** — estimator-agnostic selection layer over nested kNN-prefix estimates | katgpt-rs `.issues/1229` |
| MAD σ̂ noise scale | **Extracted** — pure order statistics; discrete-label analog = neighbour disagreement rate (derived, not copied — the MAD form assumes continuous response noise) | katgpt-rs `.issues/1229` |
| Critical radius δ̄²_k (Lambert W, entropy-derived) | **Extracted (secondary)** — closed-form capacity-vs-evidence floor; turns a measured K-floor into a derived floor that predicts how it moves when retrieval dimensionality changes. Lambert W = one fixed-point scalar fn (~6 iterations), no dep | katgpt-rs `.issues/1229` |
| kNN radius (k/n)^(1/d*) auto-adaptation | **Covered by construction** (Kpotufe 2011 — k-NN adapts to local intrinsic dimension with nothing estimating it; the paper credits this) | design-law citation |
| Pointwise minimax rate + log-factor lower bound | **Theory citation** — design justification for corpus-is-the-model serving (see §2.4) | this note |
| Oracle inequality | **Extracted as a property gate** — testable meta-gate on any modelless estimator; a violation means σ̂ or the agreement statistic is mis-specified, not that the rule is wrong | gate in `.issues/1229` |

### 2.2 Signal-diff vs shipped substrate (§3.6 discipline)

| Shipped cousin | File | Signal it consumes | This paper's signal | Verdict |
|---|---|---|---|---|
| `compute_adaptive_k` (Plan 256, `msa_adaptive_k`) | `katgpt-attn/src/dash_attn/adaptive_k.rs` | **dispersion of one score vector** — k = k_min+(k_max−k_min)·sigmoid(w·variance+b), one pass, heuristic, no oracle property | **cross-k estimate agreement** of nested candidates vs a computable noise scale | **Partial** — same output variable (adaptive neighbourhood size), different signal, different guarantee |
| `compute_derived_k` (Issue 747 P1, Research 549) | same file, `asentmax_schedule` | **level gap Δ̂** (top-vs-bulk); closed form k̂ = 4/Δ̂², length-independent by theorem | agreement; no gap proxy needed; oracle inequality | **Partial** — third arm: selection-by-agreement, completes the family {heuristic gate, closed-form law, agreement cascade} |
| `CorpusDistanceGate` | `katgpt-core::distance_abstain`, consumed by reflex `DomainExpert` (gate fitted at cal-slice 30th percentile — a GLOBAL threshold) | query→corpus distance vs global threshold | **per-query** cascade instability — abstain where no k achieves agreement (kNN aggregation provably unreliable there) | **Complementary** — adaptive second abstention signal, fusable with the distance gate |
| Published: Adaptive-k (Taguchi et al. 2025, megagonlabs/adaptive-k-retrieval) | external | similarity-distribution decay of the retrieved list | agreement of nested estimates; oracle bound | **Class prior art for "adaptive retrieval depth"** — kills the class-level novelty claim |
| Published: Pre-retrieval Query Clustering (arXiv:2609.13489) | external | depth needs estimated from query features | same class, different signal | class prior art |

**Cost insight (the 7-calls/query objection dissolves):** the grid is NESTED — k-neighbourhoods
are prefixes of one top-K retrieval. The cascade = one retrieval at the grid max + O(log n)
scalar folds over prefix aggregates. No second pass over the corpus, no per-candidate retrieval
(the paper's own §5 implementation refits per k, but that is its micro-fit, not the rule).

### 2.3 Consumer reframe (skill §1 step 4 — both halves)

**Game context:** per-NPC `GenericSpatialBelief` aggregates neighbour evidence — the cascade
selects per query how many neighbours to trust; the agreement statistic for a belief FIELD is
the DEC divergence (`codifferential`) of the field between successive k — flux change below τ ⇒
stable belief; divergence explosion ⇒ fog/contested region ⇒ confidence decays through the
existing sigmoid law. Zero training; the operators ship in `katgpt-dec`. → fusion idea, novelty
TBD: **riir-ai `.issues/1239`**.

**Consumer (rulewright) context:** (a) retrieval index — per-span adaptive K in the fan-out
(the measured K≥6 floor is global; the prefix-form keeps ONE retrieval at grid-max so the
Bench-126 call-count floor is not re-paid); (b) fix-trajectory memory — `elo_k_eff`'s global
n_floor=16 could become a per-rule adaptive evidence floor (widen while successive Beta-LCB
estimates shift ≤ τ̂) — SECONDARY, watch row; (c) score-bench — the oracle inequality as a
meta-gate on any retrieval-depth change; (d) drafter-pruner selection — **KILLED, audited
discard**: three measured negatives collide (`kernel_route_narrow` cost-axis negative Bench 126;
`calibrated_dispatch` value leg FAIL bench 156, drops 0.0095% of real pools; the LSA/HGA
crossover law — dense-write wins below ~32k buckets), plus per-query fitting in the fix path is
runtime weight mutation; (e) rustc_errors playbooks — nothing maps.

**Serving context (priority #1 in the fusion ladder):** reflex `Lz4FlexDrafter` scores option
bytes against the routed domain's whole corpus today. The adaptive-depth formulation: score
against the top-k retrieved corpus docs, cascade over prefix-k. **Honest open question (the
weakest point, named):** whether prefix-truncated corpus scoring degrades gracefully the way
kNN regression does — the bias-grows/τ-decays profile is a HYPOTHESIS on this surface until the
poc's G1 measures it. If truncation degrades non-monotonically, the cascade's premise (agreement
= smoothness detector) is refuted on that surface and the consumer stays at the attention/belief
surfaces only.

### 2.4 The theory as a design citation (free value)

The pointwise minimax result is a THEORY-ANALOG blessing for the corpus-is-the-model serving
doctrine: query-conditional local scoring (the corpus slice IS the fitted model) is provably
rate-optimal pointwise with adaptive neighbourhood size, the intrinsic dimension (not ambient)
governs the rate, and the log factor is priced by a lower bound — cite Lepski & Spokoiny 1997
instead of adding an empirical knob. Kpotufe 2011 additionally licenses "budget by k, never by
radius" — every per-domain distance threshold in the stack is redundant with the radius law if
the consumer requests k neighbours and lets the ball find its own scale (a refutation gate for
the threshold-tuning habit, not a code change by itself).

### 2.5 Fusion map (paper × stack)

- × `adaptive_k.rs` → **third arm** beside the variance-sigmoid and the Lemma-2 closed form.
- × `CorpusDistanceGate` → adaptive abstention: abstain-on-cascade-failure, fires exactly where
  kNN aggregation is provably unreliable (τ can no longer absorb the bias).
- × `katgpt-dec` → belief-field agreement via DEC divergence (riir-ai issue).
- × `elo_k_eff` → per-rule adaptive evidence floors (watch row).
- × quantization (training track) → per-block fitted pre-compensation as a candidate third
  head-class for the q4 re-open bar — riir-train issue, kill-on-tie screen only.

## 3. Verdict

| Tier | Result | Reason |
|---|---|---|
| **Super-GOAT** | ✗ | Fails Q1 at the literature level: the adaptive-bandwidth CLASS is 35-year-old classical statistics (Lepski 1990/91; Lepski et al. 1997; Goldenshluger–Lepski 2011) and "per-query adaptive retrieval depth" is PUBLISHED (Adaptive-k, Taguchi et al. 2025; arXiv:2609.13489). Fails Q2: adaptive neighbourhood size already ships (`compute_adaptive_k`, `compute_derived_k`, K≥6 floor) — the agreement SIGNAL is new in-stack, the behaviour class is not. |
| **GOAT** | ✓ (PoC-gated) | Provable-gain mechanism over the incumbent heuristic arms: the agreement cascade carries an oracle inequality the shipped arms lack, the MAD σ̂ is a principled noise scale, and the log factor is lower-bound-priced (zero tuning knobs). The gain is HYPOTHESIS until the poc's gates run — filed as poc issue per the poc-first law (proof path below), promotion talk strictly after. |
| ~~Gain~~ | — | The mechanism is more than incremental on the surfaces where the profile holds (oracle property is a capability the incumbent arms lack). |
| ~~Pass~~ | — | Not already shipped: the agreement cascade exists nowhere in the stack (signal-diff table). |

**MOAT gate (katgpt-rs):** generic modelless selection-layer primitives (cascade, σ̂, Lambert-W
floor) — public, no game/chain/shard semantics. Fits the per-stack ledger: feature flag +
benchmark + GOAT gate, slot = pruning/routing (the adaptive-k family). Consumers in reflex /
refine / riir-ai consume via the katgpt-core primitive (substrate-first, never fork).

**Per-track separation (no cascade):** the training-track kill of per-query fitting does NOT
touch the modelless selection layer (different tracks, independent verdicts); the modelless
GOAT does NOT excuse the training cells (they carry their own pre-registered kill bars).

## 4. PoC protocol (the proof path — pre-registered)

- **Fixture:** the paper's own Doppler function + a piecewise-constant fixture, planted noise σ.
  Estimator = kNN mean (continuous) and majority vote (discrete — exercises the
  disagreement-rate σ̂ analog).
- **G1 (structure):** the cascade's selected k must track planted smoothness within ONE grid
  step of the oracle k*_x (rough region → small k, smooth region → large k). Falsifier: no
  tracking ⇒ the agreement-is-smoothness premise is refuted ⇒ demote to watch row.
- **G1 (property gate):** |estimate − truth| ≤ (A+2)·τ̂ on ≥ 99% of queries at planted σ, A
  pinned; a violation indicts σ̂ (or the agreement statistic), never the rule.
- **G1 (discrete):** disagreement-rate σ̂ analog on a labeled fixture — derived, not copied.
- **G1 (known answers):** Lambert-W vectors (W(e)=1, W(1)≈0.567143, monotone, branch checks).
- **G2:** latency parity vs fixed-K incumbent (expected ≤ fixed-K_max: one retrieval either way;
  the cascade adds log-n folds).
- **G3 (the promotion gate):** paired accuracy on the reflex harness suite slice (cal + test,
  paired LB95, ONE frozen read) — cascade-adaptive depth vs incumbent fixed-depth scoring.
  No promotion talk before this runs.
- **G4:** alloc census zero at the serving posture.

**Constant hygiene:** δ̄²_k = 2048(s+1)/k·W(k(L+1)V²/(16(s+1))), τ_k = 6√κ_σ·max{δ̄²_k,
2 log(|K_n|/η)/k}^½, σ̂ = median|Y_i − Y_nn(i)|/(0.6745√2), A ≥ 4, k0 = ⌈8 log(n/η)⌉ — read from
the paper's own §3.14/§4.3/§4.4/§5 text at implementation time; never from this note's prose.

## 5. Prior art (the searches that pinned the claim)

Pinned claim before searching: *"agreement-cascade adaptive neighbourhood size (largest k whose
nested-estimate disagreement stays under a computable noise scale) for corpus/kNN decision
serving, distinguished from the shipped variance-sigmoid and level-gap closed-form arms by
consuming only cross-k agreement and carrying an oracle inequality."*

- Classical: Lepski 1990/91 (the rule); Lepski, Mammen & Spokoiny 1997 (variable-bandwidth
  kernels, spatial adaptivity); Lepski & Spokoiny 1997 (pointwise adaptation lower bound — the
  log factor); Goldenshluger & Lepski 2010/2011 (oracle inequalities); Kpotufe 2011 (k-NN
  adapts to local intrinsic dimension); Kpotufe & Garg 2013 (local smoothness + dimension);
  Hütter & Rigollet 2017 (notes on Lepski's method).
- Published class prior art (adaptive retrieval depth): Adaptive-k (Taguchi et al. 2025,
  megagonlabs/adaptive-k-retrieval — context size from similarity decay); Pre-retrieval Query
  Clustering for Adaptive Top-k (arXiv:2609.13489 — depth needs from query features); Adaptive-RAG
  family (query complexity → retrieval strategy).
- The paper's OWN novelty (and the residual novelty of the fusion): the local model is a
  NETWORK, the guarantee is POINTWISE (integrated-risk adaptivity says nothing about a rough
  point), and the stochastic scale is ENTROPY-derived (computable from architecture) — that
  combination is new in the literature per its Table 1, and the agreement-cascade-with-oracle-
  gate is new in OUR stack per the signal-diff table.

## References

- `katgpt-rs/.issues/1229` — poc: agreement cascade + MAD σ̂ as the adaptive_k third arm (gates §4).
- `riir-train/.issues/701` — training-track probe cells (frozen per-cluster NPC brain grids;
  per-block quant pre-compensation screen, kill-on-tie).
- `riir-ai/.issues/1239` — DEC-field agreement fusion (novelty TBD).
- Shipped cousins: `katgpt-attn/src/dash_attn/adaptive_k.rs`; Research 549 (Lemma-2 derived k);
  Plan 256 (sigmoid arm); `katgpt-core::distance_abstain::CorpusDistanceGate`;
  `katgpt-core::compression_drafter::Lz4FlexDrafter`; riir-refine `elo_k_eff` (n_floor 16).
