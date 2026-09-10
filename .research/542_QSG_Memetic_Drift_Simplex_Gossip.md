# Research 542: QSG — Quantized Simplex Gossip, Memetic Drift & Crowd-Consensus Scaling Laws

> **Source:** "When Is Collective Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in LLMs" — Hidenori Tanaka (Harvard/NTT Physics of Intelligence), [arXiv:2603.24676](https://arxiv.org/abs/2603.24676), Mar 2026 (19 pp). Companion blog: physicsintelligence.org "What Shapes Collective Belief Collapse in AI Swarms?" (Sep 2026).
> **Date:** 2026-09-10
> **Status:** Active
> **Related Research:** 497 (signed-coupling opinion dynamics), 371 (mean-field regime classifier), 469 (collective payoff schemes), 499 (jagged judges), 505 (mean-field distributional steering)
> **Related Plans:** [Plan 589 — QSG gossip primitive](../.plans/589_qsg_gossip_primitive.md) (this repo)
> **Cross-ref (riir-ai):** Research 314 (sheaf-ADMM consensus), 167 (cross-NPC set attention), 135 (cascade belief octree), **369 (crowd-consensus guide)**; Issue 907 (swarm consumer POC)
> **Classification:** Public

---

## TL;DR

Tanaka introduces **Quantized Simplex Gossip (QSG)**: N agents each hold a belief = a probability
distribution over K options (a point on the simplex); each interaction a random speaker **samples** a
length-m message from its belief and a random listener blends toward it at adaptation rate α. Because
the communicated message is quantized (sampled), sampling variance is *injected into the population
even when every agent is identical* — the population becomes its own evolving data source ("mutual
in-context learning"), and under neutrality consensus is a lottery ("**memetic drift**", by analogy
with neutral evolution). The paper derives and experimentally validates (GPT-4o, Claude Haiku 4.5
naming games) closed-form scaling laws: collapse time `t ~ mN²/α²`, bandwidth law `1/m`, and a
drift-vs-selection crossover `Γh = mNh/α` with logistic fixation.

**Distilled for katgpt-rs (modelless, inference-time):** a complete, closed-form dynamics module for
multi-agent belief interaction — zero training, zero weights, pure array math. It drops in beside
`signed_coupling` (which owns the *temperature* axis of crowd opinion dynamics) as the
*bandwidth/adaptation* axis, generalizes our crowd order parameters from binary stances to K-option
beliefs, and gives the first **predictive** (not merely detector-style) crowd-collapse law in the
corpus. Super-GOAT: new capability class (predictive crowd-consensus scaling), product selling point,
force multiplier across ≥4 shipped systems.

---

## 1. Paper Core Findings

**Model.** Agent i holds `x_i ∈ Δ^{K−1}` (simplex). Each step: ordered speaker–listener pair
uniformly at random; speaker emits message y; only the listener updates:

```
x_L ← (1−α)·x_L + α·y          α ∈ (0,1]   (adaptation rate = persona plasticity)
Hard   (m=1):  y = e_k*,  k* ~ Cat(x_S)
Top-m:         y(m) = (1/m) Σ_j e_{k_j},  k_j ~iid~ Cat(x_S)
Soft   (m=∞):  y = x_S   (DeGroot-style full-belief exchange — analytic baseline)
```

**Observables.** Population mean `x̄`, **polarization** `U = ‖x̄‖²₂ ∈ [1/K, 1]` (the order
parameter: 1/K = symmetric, 1 = collapsed/consensus), disagreement energy `V = Σ_i ‖x_i − x̄‖²`,
coordination rate `S = U − V/(N(N−1))`. Identities: `V = N(q − U)`, `E‖x_S − x_L‖² = 2V/(N−1)`.

**Theorem 1 (variance injection).** `E[ΔU|X]_hard = E[ΔU|X]_soft + (α²/N²)·E[1−‖x_S‖²|X]`.
Soft exchange *preserves the mean in expectation* (martingale) and *contracts V*
(`E[ΔV|X]_soft = −2α/(N−1)·(1−α+αN)·V ≤ 0`) — full-belief exchange can NEVER break symmetry.
At perfect symmetry, Soft stays frozen; Hard drifts at `α²(1−1/K)/N² > 0` — **the speaker's
uncertainty `1−‖x_S‖²` is the fuel of memetic drift** (zero at one-hot vertices, max at center).

**Theorem 2 (bandwidth law).** `E[ΔU|X]_topm = E[ΔU|X]_soft + (α²/(mN²))·E[1−‖x_S‖²|X]` —
extra collapse pressure scales `1/m`. Short messages = lossy quantization = faster collapse.

**Mean-field collapse law.** `dU/dt = (α²/mN²)(1−U)` →
`U(t) = 1 − (1−1/K)·exp(−α²t/(mN²))`, `t_cons(U⋆) ≈ (mN²/α²)·log((1−1/K)/(1−U⋆))`.
Validated on GPT-4o + Claude Haiku 4.5 naming games: early drift ∝ 1/N², consensus time ∝ N²,
Top-m drift ∝ 1/m — all three laws confirmed across two LLM families.

**Drift–selection crossover (K=2).** Tilt the speaker channel by small bias h. Diffusion
approximation collapses fixation statistics onto one parameter **`Γh = mNh/α`**:

```
Pr(label 1 fixes) ≈ 1/(1+exp(−Γh))     (logistic)
Nc ~ α/(m|h|)                          (crossover population size)
```

|Γh| ≪ 1 → near-neutral winners (lottery); |Γh| ≫ 1 → weak bias decisively amplified.
**Larger N or m suppress drift and make the same bias more decisive; larger α strengthens drift
relative to bias.** Tempered-sampling variant: `ΓT = (mN/α)·|1/T − 1|`, crossover at ΓT ≈ 1,
also validated. α=1 reduces to the K-state voter/Moran process (winner prob = x̄_k(0), a martingale
result); α<1 is expectation-level identities + mean-field only.

**Safety reading (the blog's framing):** collective belief collapse in agent swarms (METR/OpenAI-HF
message-board incidents) is steerable by three physical dials — plasticity α, bandwidth m, population
N. Agreement alone is NOT evidence of collective reasoning; it may be amplified sampling noise.

---

## 2. Distillation

The transferable primitive is **not** about LLMs at all: it is a closed-form statistical mechanics of
*any* population of agents that (a) hold normalized belief distributions over K options, (b) communicate
**samples** (not full beliefs) under a per-message budget, (c) blend toward what they hear at rate α.
Every ingredient is modelless array math — categorical sampling, convex blend, mean/reducer, logistic —
the exact class `signed_coupling` already ships (Glauber resampling, order parameters, sweep harness).

Three conceptual upgrades over our current crowd machinery:

1. **Persistent simplex beliefs vs memoryless stances.** `signed_coupling` recomputes a Bernoulli
   parameter from the graph each tick (heat-bath); QSG agents carry persistent state that EMA-blends —
   beliefs have *inertia*, which is what makes drift *accumulate* into conventions.
2. **The m axis (message quantization) is new to the corpus.** Our closest budget surface,
   `NpcCommsBus`'s `K(ca)` message budget (`DensityBudget::k_for`), already ships
   "how much of the speaker's state fits in the message" as an **engineering** constraint; QSG reveals
   it is also a **dynamical** control: bandwidth sets collapse speed `1/m` and the crossover `Nc ~ α/(m|h|)`.
3. **Predictive laws, not detectors.** Our collapse machinery is detector-style (`χ = N·Var_t(|n|)`
   sweep, δmg discriminator in R135, anti-common-mode gate); QSG gives *closed-form predictions* —
   time-to-consensus from (N, m, α) before the crowd runs, and the logistic fixation curve from one
   dimensionless group.

### Latent-space reframing

- Per-agent belief `x ∈ Δ^{K−1}` is **local latent state** (K-dim, K = option count, e.g. 3 rumor
  variants, 5 faction candidates) — never synced raw.
- Zone-level order parameter `U` (one f32) is the projection that leaves the entity: exactly the
  "sync the scalars, not the embedding" rule; it feeds the existing `MeanFieldOverlap` reducer row and
  crowd-regime classifier as one more aggregate over the same pass (Plan 371).
- Sigmoid usage: blend gating stays sigmoid (`σ(α·freshness)` in `blend_from_indexed_slice` already);
  the softmax appears only as categorical *sampling* over the simplex (that is a distribution draw,
  not a projection — allowed; the "sigmoid not softmax" rule governs projections onto direction
  vectors, which unchanged).
- DEC bridge (d ≤ 3 caveat): for K ≤ 3 option beliefs, the gossip flow is a rank-1 cochain flow and
  `belief_mass_divergence` (`stokes_calculus.rs`, Plan 314) validates mass conservation of the
  belief distribution under sampling (sampling preserves E[y] = x_S — mass conserved in expectation;
  δ-diagnostic makes it checkable). NOT for high-K.

---

## 3. Fusion — what paper × corpus produces that neither has alone

Vocabulary translation ran both directions (paper vocab → operator names); the sweep hit 17 shipped
mechanisms, 4 of them genuine multi-agent belief interaction. Closest cousins and the fusions:

| # | Cousin | Ships | Fusion with QSG |
|---|---|---|---|
| 1 | `signed_coupling` (katgpt-core, R497, Bench 672) | binary Glauber opinion dynamics, `net_opinion`/`crowd_conviction`/χ(T_c) sweep | Generalize stance ∈ {±1} → simplex belief ∈ Δ^{K−1} keeping sampled exchange; χ-sweep harness gains the (N, m, α) axes; R497's "awaits production consumer" gets one: **collapse-time prediction**. Relationship: signed_coupling = temperature axis (T), QSG = bandwidth/adaptation axes (m, α) — sibling dials of the same crowd panel. |
| 2 | `MeanFieldOverlap` + Hopf regime classifier (Plan 371, default-on) | population order parameters (κ, κ_a, Q) + Calm/Oscillating/Irregular/Panic | `U` and `V` become two more reducers over the same aggregation pass; the regime classifier gains a **consensus/collapse** state grounded in a validated law. |
| 3 | `NpcCommsBus` + `blend_from_indexed_slice` + `K(ca)` budget | speaker→listener partial-belief message, σ(α·freshness) EMA blend | This IS QSG's listener update with a budgeted message — except the slice is deterministic top-K, not a random sample. QSG predicts what that difference *does*: sampled messages inject collapse pressure α²/(mN²)·E[1−‖x_S‖²], deterministic slices do not. The m axis gets a physical meaning. |
| 4 | `agreement_contagion` (R364/Plan 588 family) | relay strength `c(α) = 2σ(κ(α−1))−1` vs witness count | Complementary laws, different quantities: contagion = propagation *strength* vs witnesses; QSG = consensus *timescale* vs (N, m, α). Together: how fast a rumor spreads AND how fast a crowd settles on it. |
| 5 | Sheaf-ADMM consensus (R314/Plan 394, default-on) | per-agent (x_i, z_i, u_i) diffusion to consensus | QSG is the *stochastic* consensus process the sheaf's z-projection deterministically seeks; the dual `u_i` (disagreement fingerprint) doubles as a per-agent drift-vs-selection estimator (which NPCs wander neutrally vs are being pulled by the majority). |
| 6 | Payoff schemes (R469) | imitation weighted by feedback payoff (selection pole, Lyapunov) | QSG is the drift pole; **the Γh crossover IS the fusion**: CLR feedback payoffs act as the bias h — in large zones even tiny payoff asymmetries decide the convention, in small camps consensus is a fashion lottery. Two designer knobs: drift-only zones vs payoff-weighted zones. |
| 7 | Cross-NPC set attention (R167/Plan 355) + `social_pressure` + R135 cascade + crowd_mcgs gossip events | full-belief DeGroot blending / persuasion / cascade / gossip events | QSG supplies the population-level law these micro-mechanisms embed into; the γ/N N-invariance guard in set_attention is the *anti*-collapse dial QSG's U-monotonicity explains. |

Consumer-context (priority-#2 healer check, honest): the healer fleet's shared corpus/trajectory store
is a message board where rule-frequency compounds across mining batches — QSG's selection regime warns
fleet-shared fix *conventions* can amplify tiny early biases, and `score_history.jsonl` could carry a
U-style concentration statistic. But no per-agent belief state exists to gossip with, so there is no
direct implementable surface today — recorded here, not filed.

### Game-context reframe (the selling point)

- **Predictable rumor collapse:** "In our 1000-NPC zones, the time for a crowd to settle on one rumor
  follows a measured law t ~ mN²/α² — tuned per-zone by NPC agreeableness (α from temperament) and
  gossip bandwidth (m from the message budget)." Small camps → consensus is a lottery → emergent
  faction names/fashions differ per shard. Large cities → any seeded bias (a rumor-mongering NPC, a
  fear-primed zone) deterministically wins → designers steer crowd monoculture with two dials.
- **New crowd regime:** U crossing toward 1 = monoculture (mob/panic/hype); the anti-common-mode gate
  and Hopf classifier get a physics-grounded trigger.
- **Persona plasticity reads as sycophancy:** Tanaka's "What Is the Agent Condition?" — an individual
  agent's collaborativeness (high α) is exactly what accelerates collective collapse. Per-NPC α becomes
  a *social* trait with population-level consequences — a novel emergent-behavior axis no competitor
  ships with calibrated laws.

---

## 4. Classification (Path 0) + prior art

**Path 0 (trivial):** the paper contains no training loop, no loss, no gradient — its value is 100%
closed-form math computed at inference/simulation time. MODELLESS-VALIDABLE, no riir-train deferral,
no three-track adversarial panel required (classification never touches training; pre-flight #5
confirmed no training-track surface is implicated).

**Pinned novelty claim (§4 precondition):** *Sampled-message belief gossip (QSG) with closed-form
collapse scaling laws (t ~ mN²/α², 1/m bandwidth law, Γh = mNh/α drift-vs-selection crossover) for
the crowd/swarm runtime surface, consuming per-NPC K-option belief state + per-zone message budget,
distinguished from shipped `signed_coupling` (binary memoryless Glauber, temperature axis only) by
persistent simplex beliefs, EMA adaptation, and bandwidth/adaptation scaling laws.*

**Published prior art (web search, 7 queries):** the (a) simplex-gossip + (b) closed-form
N/m/α scaling laws + (c) drift-vs-selection crossover combination ships nowhere except the paper
itself (also circulated as "Scaling Laws for Consensus Under Pluralistic Uncertainty", ICML 2026
workshop). Closest: Ashery et al. 2025 (rewarded naming game — no beliefs, no laws), Flint et al. 2026
PNAS (group-size empirics — no model), quantized-gossip control theory (Frasca 2008/Koloskova 2019 —
consensus guaranteed, never a lottery; no LLM/belief semantics). Concurrent neighbor: Zou et al.
2608.16578 (Ising stat-mech of AI communities) — already distilled as R497's source; no overlap with
the QSG combination. ~2 citations, no critiques yet.

**In-corpus prior art:** binary-opinion gossip + order parameters + critical temperature are
already claimed (R497); peer belief refinement claimed twice (R167, R314). **Unclaimed in the corpus:
K-simplex beliefs, sampled-message semantics, the drift/selection dichotomy, and a predictive
collapse-time law.** Additive, not duplicative.

---

## 5. Verdict

**Super-GOAT.** All four gates YES:

1. **No prior art** — published combination novel (above); in-corpus gaps precisely identified
   (simplex beliefs, sampling semantics, predictive laws).
2. **New behavior class** — from detector-style collapse detection to *predictive* crowd-consensus
   dynamics with designer-steerable dials; chance-driven convention formation in small populations is
   a behavior class no shipped system exhibits by construction.
3. **Product selling point** — "crowds that obey physics": rumor-consensus times follow measured
   scaling laws tuned by per-NPC plasticity and per-zone bandwidth; small camps show shard-unique
   fashion/faction lotteries, large cities deterministically amplify seeded biases.
4. **Force multiplier** — connects ≥4 shipped systems: `signed_coupling`, `mean_field` regime
   classifier, `NpcCommsBus` budget, `agreement_contagion`, plus sheaf-ADMM (drift estimator) and
   R469 payoffs (selection pole).

**Selling point (explicit):** our NPC crowds exhibit physically calibrated collective belief dynamics —
consensus collapse times follow published, validated scaling laws, per-zone steerable, emergent faction
naming included.

**MOAT gate:** katgpt-rs — fundamental base primitive via fusion (crowd-dynamics family; belief kernel
+ sigmoid mechanics are in-scope per §1.6; sibling slot to `signed_coupling`, not a demotion — new
axes m/α, new state class). riir-ai — pillar-level: fuses crowd_mcgs + npc_comms + swarm emotions +
mean_field into a new selling point (guide: riir-ai/.research/369). Feature `qsg_gossip` ships
opt-in; promotion to default requires the Plan 589 GOAT gate (physics-law reproduction + perf).

**Outputs (this session):** Plan 589 (primitive + physics GOAT gate, this repo) ·
riir-ai Research 369 (architectural guide) · riir-ai Issue 907 (swarm consumer POC, runs the
goat-audit before its implementation plan).

### Pre-plan cherry-pick note (§1.7)

The katgpt-rs plan is primitive-internal (no cross-repo consumption). The riir-ai consumer issue
consumes `signed_coupling`/`mean_field`/`NpcCommsBus` substrate — its implementation plan must run the
goat-audit + substrate-first skills first (noted in Issue 907).
