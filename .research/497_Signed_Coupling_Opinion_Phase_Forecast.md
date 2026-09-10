# Research 497: Signed-Coupling Opinion Dynamics — Crowd Phase Forecasting

> **Source:** [arXiv:2608.16578](https://arxiv.org/abs/2608.16578) — "Physics of Agents: Statistical Mechanics Predicts Collective Behavior of AI Agents" — Batu El, Jinhee Paeng, Fatih Dinc, Shiye Su, Mete Erdogan, Aneesh Pappu, Haotian Ye, Wanjia Zhao, Surya Ganguli, James Zou (Stanford), 17 Aug 2026, 51pp
> **Date:** 2026-08-21
> **Status:** SHIPPED 2026-08-22 — `signed_coupling_dynamics` (opt-in) in `katgpt-core/src/signed_coupling.rs`; GOAT G1–G4 ALL PASS, [Bench 672](../.benchmarks/672_signed_coupling_goat.md). Issue 680 resolved-and-removed. Two measured findings the §10 spec did not anticipate: (i) at the fitted-range MIDPOINTS a discordant tie is net ATTRACTIVE (`β₀ − β⁻ = +0.15`), so a frustrated graph converges — polarization needs the `β⁻ > β₀` corner of the same ranges; (ii) the `|J|` channel is not independent (`Σ|J|s = P + D`), so the three-coupling sum collapses to two channel sums weighted once per NODE — the per-edge weight-table form of the §10 spec measured 1.5× SLOWER. Promotion to default still waits on a production consumer (§7: swarm emotions first).
> **Related Research:** 470 (Lonely Runner phase separation — the in-stack statistical-mechanics ancestor), 354 (NPT cross-NPC set attention), 468 (Beckmann MFG transport — the other crowd-dynamics formalism), 478 (MOP crowd occupancy)
> **Related Plans:** none yet — gated on the primitive issue below
> **Cross-ref (riir-ai):** swarm `emotion.rs` (tick_swarm_emotions / tamed_aura / CLR collective — the closest shipped propagation substrate); Sheaf-ADMM conviction (vocabulary collision, §5)
> **Classification:** Public

---

## TL;DR

The paper fits an **Ising/Glauber model on a *signed* social graph** to 10,000+ LLM-agent communities and shows the fitted rule predicts collective opinion dynamics (75–86% balanced accuracy) from initial opinions alone. The core update rule is our house shape — **sigmoid of a signed-graph weighted sum**:

```
P(s_i = +1) = σ( β⁺·Σ_j J⁺_ij·s_j  +  β⁻·Σ_j J⁻_ij·s_j  +  β₀·Σ_j |J_ij|·s_j  +  g_i )
```

with separate couplings for concordant ties (β⁺), discordant ties (β⁻), and mere connection (β₀), plus a per-agent intrinsic field `g_i = w^T·φ_i` (persona × question embedding — **exactly our direction-vector dot-product pattern**). Three findings transfer modellessly: (i) communities operate **below a critical social temperature** T_c (located by the peak of χ = N·Var_t(|n(t)|)), which explains why conviction builds; (ii) concordant ties outweigh discordant ones (β⁺ > β⁻ everywhere), which biases crowds toward consensus over polarization; (iii) **correct neighbors pull harder** (5-coupling truth asymmetry), which drives truth-seeking.

**Distilled for katgpt-rs (modelless, inference-time):** a `signed_coupling` opinion-propagation kernel + three crowd order-parameter reducers (`net_opinion`, `conviction = mean(s²)`, susceptibility accumulator). The paper's gradient-descent fit is only needed to *predict real LLM agents*; a game crowd *authors* its couplings — the paper's own fitted values (β⁺≈0.9–2.4, β⁻≈0.2–1.1, β₀≈0.6–1.0, truth-gap β_T⁺−β_F⁺≈0.1–0.3) become designer-facing defaults. Zero-GD, zero-alloc, O(edges), sigmoid-not-softmax.

---

## 1. Paper core findings

1. **Setup.** N=32 LM agents, personas + binary questions (objective MATH / subjective political), signed communication network J ∈ {−1,0,+1}^N×N (concordant/discordant/none), T=8 synchronous rounds of message exchange. ~10,000 communities across 4 models × 60 questions × 10 graphs × 4 episodes.

2. **Regimes.** Crowd state is summarized by **net opinion** n(t) = mean(s) and **conviction** c(t) = mean(s²). Three characteristic regimes: **indifference** (|n| low, c low), **polarization** (|n| low, c high), **consensus** (|n| high, c high). Conviction rises monotonically under interaction — crowds order themselves.

3. **Archetypes.** Individuals: Frozen / Switcher / Intermittent / Oscillator (by flip count). Groups: Persistent Split / Convergence / Divergence / Majority Switch / Persistent Majority (via split band δ=1/K around n=0). Divergence + Majority Switch reach 11–12% — crowds overturn initial majorities regularly.

4. **Truth-seeking.** On objective questions, wrong→correct majority switches outnumber correct→wrong. On subjective questions, 3 of 4 models drift rightward politically — interaction can amplify directional bias.

5. **Model.** Energy `E(s) = −½ ΣΣ J_ij s_i s_j − Σ g_i s_i`; Glauber dynamics give the sigmoid update above. Fitted by GD on one-step transitions (16–19 params). Beats Persistence / Interaction-Free / Mean-Field baselines in every cell; generalizes to unseen graph families (lattices, low-rank).

6. **Mechanics.** (i) fitted operating point sits **below T_c** in every model×dataset cell → conviction buildup is the ordered phase asserting itself; (ii) β⁺ > β⁻ in all four models (effective concordant weight 0.99–3.03 vs discordant ≤0.73, sometimes negative) → consensus favored; (iii) truth-split couplings: correct neighbors pull harder on the concordant channel AND wrong neighbors push harder on the discordant channel — both are truth-seeking.

## 2. Path 0 decomposition (training-target → modelless inventory)

The paper's only GD is the offline fit of ~19 scalars to observed LLM transitions. Every runtime component is closed-form:

| # | Component | Paper form | Shipped analog (audited) | Extraction |
|---|-----------|-----------|--------------------------|------------|
| 1 | Local field sum | Σ_j J_ij·s_j | CLR weighted set attention (`katgpt-core/src/set_attention.rs`, σ(q·k/√k·β)-gated weighted sum) — **unsigned, single-channel**; swarm `emotion.rs` fear propagation | closed-form; needs the sign/tie-type split |
| 2 | Sigmoid gate | σ(h_i) | house rule everywhere (`sigmoid_latent`, functor_gate `σ(β·(coherence−τ))`, analytic-lattice decoder `σ((dot/N)·temp)`) | **exists** |
| 3 | Intrinsic field | g_i = w^T·φ_i (persona×question) | per-NPC `style_weights[64]` (NeuronShard), personality direction vectors, DriveSpec dir_vecs | **exists** — it is our dot-product+sigmoid pattern verbatim |
| 4 | 3-coupling split | β⁺/β⁻/β₀ over J⁺/J⁻/\|J\| | swarm 2-channel precedent: threat kick + rank-gated aura reduction `(1−aura)` — tie-typed but magnitude modulation, not signed couplings on a state; no β₀ channel | **missing** — the load-bearing new substrate |
| 5 | Net opinion n = mean(s) | crowd scalar | crowd centroid / crowd_cost mean (`cce_runtime/crowd_attention_bridge.rs`) | **exists** |
| 6 | Conviction c = mean(s²) | order parameter | **NOT SHIPPED** (no mean-square crowd reducer anywhere) | trivial reducer |
| 7 | Susceptibility χ = N·Var_t(\|n\|) | criticality diagnostic | **NOT SHIPPED** (no order-parameter variance diagnostic; τ-as-steepness ships, sweeps don't) | running-moment accumulator |
| 8 | Critical temperature T_c | peak of χ over temperature sweep (offline, 41 log-spaced points × 500 steps) | — | offline example/bench, not runtime |
| 9 | Truth-asymmetric coupling | κ_j = 1{s_j = correct} × split β_T/β_F | first_offered_scored_hunts-style informed-scoring precedent | closed-form (informed-NPC indicator) |
| 10 | Continuous relaxation | τ·dm_i/dt = −m_i + tanh(β(Jm)_i + g_i); ε-clock discretization | — | optional; the discrete rule is the primary |

**Path 0 verdict: MODELLESS-VALIDABLE.** No riir-train deferral — the GD fit predicts *real* LLM agents; game crowds *author* couplings (deterministic construction, hand-tuned or paper-default constants).

## 3. Published prior art (§4 gate — run before this verdict)

The **framing-level claim is taken**. Two works a reviewer would cite against the paper — and against us if we overclaim:

- **De Marzo et al., [arXiv:2605.10721](https://arxiv.org/abs/2605.10721)** (11 May 2026, 3 months prior): "Conformity Generates Collective Misalignment in AI Agents Societies" — fitted statistical-physics theory of LLM agent opinion dynamics with conformity coupling + intrinsic bias (structurally the β·ΣJ·s + g decomposition), predicts regime trapping and tipping points, at 9 LLMs × 100 opinion pairs. Covers: the general "stat-mech theory predicts LLM crowd regimes" headline.
- **De Nobili, Iyer, Codello, Burioni, [arXiv:2608.02178](https://arxiv.org/abs/2608.02178)** (3 Aug 2026, 2 weeks prior): mean-field theory + **critical line / critical exponents / temperature as control parameter** for multi-agent LLM consensus (naming games). Covers: the criticality machinery for LLM agent collectives.
- Background classics the paper itself sits on: Glauber 1963 (the update rule is textbook heat-bath dynamics — 60+ years old), Brock & Durlauf 2001 / Castellano et al. 2009 (Ising-opinion sociophysics), Li et al. 2019 (binary opinion dynamics on signed networks), Tsarev et al. 2019 (critical *social* temperature with valence/conviction-like order parameters), Chuang et al. 2024 (LLM opinion-dynamics simulation), MF-LLM [arXiv:2504.21582](https://arxiv.org/abs/2504.21582) (mean-field LLM populations).

**What survives as defensible increment** (and is what THIS note claims): (a) the **signed 3-coupling taxonomy** fitted to transitions; (b) the **truth-asymmetry 5-coupling split**; (c) **individual-trajectory prediction** with graph-family generalization (prior work predicts population-level trapping, not paths); (d) χ-peak social T_c on **signed opinion graphs**. For OUR stack the application target is different again — game NPC crowd phase forecasting — for which no shipped or published game-AI analog surfaced (residual uncertainty: the search covered LLM-agent literature, not game crowd-sim literature; scoped below to our substrate audit as the load-bearing novelty check).

## 4. Signal-diff vs closest shipped cousins (§3.6, granularity rule applied)

| Cousin | Its signal | Paper's signal | Verdict |
|--------|-----------|----------------|---------|
| CLR weighted set attention | α_ij = σ(q_i·k_j) — **relevance** (query-key similarity), positive | J_ij — **relation** (fixed signed adjacency), tie-typed β split | different class; CLR is the propagation *substrate*, not the signed coupling |
| Swarm emotion (threat kick + aura) | fear **magnitude** modulation; per-observer gates | signed coupling on an opinion **state** s_j, outer sigmoid → probability; β₀ mere-existence channel | partial precedent (2 tie-typed channels exist); the signed-state + 3-coupling + probability form does not ship |
| Sheaf-ADMM conviction | per-dim **resistance** to consensus (diag of quadratic; scalar 1.0 in prod, vectors in bench) | conviction c(t) = mean(s²) — crowd **order parameter** | **vocabulary collision — same word, different meaning.** See §5. |
| Beckmann/MFG transport (R468) | continuous flow/divergence on spatial fields | discrete binary spin dynamics on signed graphs | complementary formalism, not overlapping |
| Phase separation (R470/R334) | periodic circular distances (Lonely Runner) | thermodynamic phase transition / critical temperature | different "phase"; R470 is the in-stack stat-mech ancestor, no T_c machinery |

One level up (the Le Critique rule): the expression *combining* CLR with other signals in production is `tick_swarm_emotions_collective` = reliability gate `sigmoid(intensity)^M × calmness^panic_exp` → CLR propagate. That is a per-**observer** gate; the paper's split is per-**tie-type** couplings on the edge. Confirmed different insertion point.

## 5. Vocabulary collision warning (load-bearing for future greps)

"**Conviction**" now means two things in the stack:

1. **Sheaf-ADMM conviction** (riir-agents `multi_agent.rs`, Sheaf maps): per-agent/per-dim resistance in the consensus quadratic — how strongly an agent *holds its ground*. Semantically closest to the paper's **λ personal-pressure weight** (or an inverse-temperature per agent).
2. **Paper conviction** c(t) = mean(s²): a *crowd-level order parameter* — how strongly the crowd overall holds opinions, regardless of direction.

Any future grep for "conviction" must disambiguate. Proposed naming for the new reducer: `crowd_conviction` (order parameter) vs the existing sheaf `conviction` (resistance). Nice fusion hook, though: **Sheaf conviction vectors are a natural per-agent g_i / λ source** — agents with high sheaf conviction get a stronger intrinsic field in the opinion update.

## 6. Game-context reframe (§1 step 4)

- **Per-NPC behavior signal**: the local field h_i *is* social pressure — an NPC's push toward an action stance given friends'/rivals' stances and its own personality. Below T_c, stances crystallize (the crowd commits); near/above T_c, milling indecision.
- **Crowd patterns = the three regimes**: indifference (town-square chatter, market hagglers), consensus (mob panic, market runs, stampedes), polarization (faction standoffs, dueling guilds, split elections). **Forecasting which regime a crowd enters** — from initial stances + the grudge graph — is a capability our stack does not have: swarm emotions *react*, they do not *predict phase*.
- **Combinatorial structure already ships**: the signed J_ij is derivable from grudge intensity × engagement sign (per-pair) or latent-grudge direction vectors; KG triples (entity fears entity → J=−1, entity likes entity → J=+1) are exactly the social-domain emission we already produce.
- **Social temperature as a designer dial**: per-zone "social climate" T — high T = indecisive crowds, low T = decisive mobs. One scalar knob moving a crowd between apathy and riot. Nobody ships this.
- **Truth asymmetry → informed-NPC weighting**: rumors/quest-hints from NPCs who actually completed the content spread with coupling β_T⁺ > β_F⁺ — a closed-form explanation of "ask the veteran, not the tourist".
- **Fairness/coverage**: majority-switch and divergence rates (11–12%) are measurable crowd-liveliness metrics — a GM observability panel: is this crowd dead (frozen), alive (switching), or stampeding (consensus + high conviction)?

**Selling point (if promoted later):** *"Our crowds have a social temperature."* Crowd phase (panic / split / apathy) is forecastable from initial emotional state + the social grudge graph, at zero-alloc Plasma-tier cost, with a single per-zone designer dial. Not claimed as Super-GOAT here — see verdict.

## 7. Fusion

- **CLR weighted set attention** — the signed coupling update is CLR's sibling: same σ(gated weighted sum) shape with signed, tie-typed weights. Implement as a `clr_weighted_set_attention_into`-style caller, not a parallel system.
- **Swarm emotions (riir-games)** — the natural first consumer: fear propagation generalized to signed opinion stances; tamed_aura already proves the rank-gated negative-coupling pattern.
- **Sheaf-ADMM** — conviction vectors as per-agent intrinsic fields (g_i / λ); the two "convictions" compose.
- **Phase separation (R470/R334)** — same stat-mech family; phase_separation drives *when* an NPC is unique, social temperature drives *how the crowd orders*. Orthogonal, composable.
- **Salience tri-gate** — predicted-regime as a salience input: a crowd heading into consensus should delegate more (fewer speakers needed); a polarizing crowd should speak more.
- **MCTS/ARG rollouts** — the fitted rule is a cheap forward model for crowd rollouts (500-step sweeps ran offline in the paper; a 32-agent rollout is trivially fast) — planning through predicted crowd phase.
- **Karma engine / grudge** — the J graph source. KG-triple emission from latent proximity (social domain rule) feeds the adjacency for free.

## 8. Latent vs raw boundary

- The opinion update, forecasts, T_c, and regime classification are **latent/think-brain** — computed locally, never synced, consumed by GM tooling / salience / planning. Fog-of-war compliant.
- Crowd summary scalars (n, c per zone) may sync **as summaries** (the flock-centroid precedent: a summary is not per-entity truth).
- If any crowd decision commits through chain (e.g. a faction-vote event), the *event* is a raw TxDelta; the *dynamics that produced it* stay latent. Never sync h_i fields.

## 9. Verdict

**Gain.** Per §1.5:

| Gate | Score | Reasoning |
|------|-------|-----------|
| Q1 no prior art | **NO** | The framing-level claim ("statistical mechanics predicts LLM agent collective behavior") is published — De Marzo arXiv:2605.10721 (May 2026) is the direct precedent; De Nobili arXiv:2608.02178 covers LLM-agent criticality two weeks prior. Only the specific package (signed 3-coupling, truth asymmetry, trajectory prediction, χ-peak T_c on signed graphs) survives. |
| Q2 new behavior class | yes (for our stack) | Crowd-phase *forecasting* (predict consensus/polarization/indifference before it happens) — swarm emotions react, they don't predict phase. Substrate audit confirms nothing ships. |
| Q3 selling point | yes | "Crowds have a social temperature" — designer dial + GM observability + panic anticipation. |
| Q4 force multiplier | yes | CLR, swarm emotions, Sheaf conviction, grudge/karma, salience, MCTS rollouts (≥2 pillars, really ~6). |

Not all 4 → **not Super-GOAT** (Q1 fails on the framing; the no-candidate rule applies). Doesn't ship + modelless → Gain. The shipped-substrate coverage is partial (CLR shape + 2-channel precedent exist; the signed 3-coupling kernel, mean(s²), and χ diagnostics do not ship).

**MOAT gate (katgpt-rs):** in-scope — generic signed-graph math + order-parameter diagnostics, no game/chain/shard semantics. The riir-ai consumer wiring (crowd-phase forecasting in swarm/salience) is downstream, gated on this primitive's GOAT + a goat-audit before any riir-ai plan consumes it.

**UQ honesty note:** the kernel emits a Bernoulli parameter P(s_i=+1) — it is a *dynamics rule*, not a calibrated forecaster. If anyone later claims prediction quality (the paper's 75–86% numbers), the conformal-naive floor rule (Report the Floor) applies to that claim — not to the primitive itself.

## 10. Primitive spec (implementation issue: katgpt-rs `.issues/680`)

```rust
// katgpt-core, feature `signed_coupling_dynamics` (opt-in)
// Row-compressed signed adjacency (J+ and J- neighbor lists per node).

/// Glauber-style signed-coupling opinion update.
/// h_i = beta_plus·Σ J+_ij·s_j + beta_minus·Σ J-_ij·s_j + beta_zero·Σ|J_ij|·s_j + g_i
/// P_i = sigmoid(h_i).  O(edges), zero-alloc (writes into caller scratch).
pub fn signed_coupling_update_into(
    graph: &SignedGraph, states: &[f32], // s_j ∈ {-1,+1} (or conviction-weighted [-1,1])
    couplings: &Couplings, intrinsic: &[f32], // g_i
    out_probs: &mut [f32],
)

/// Truth-asymmetric variant: coupling weight per neighbor depends on an
/// informed-indicator κ_j (the paper's 5-coupling split, composed from
/// (beta_true, beta_false) × (J+, J-) at call sites).
pub fn signed_coupling_update_informed_into(...)

// Order-parameter reducers (always-on with the feature):
pub fn net_opinion(states: &[f32]) -> f32;          // n = mean(s)
pub fn crowd_conviction(states: &[f32]) -> f32;     // c = mean(s²)  — NEW (nothing ships today)
pub struct SusceptibilityAccumulator;               // running Var_t(|n|) over ticks → χ; T_c = argmax over a sweep (offline bench/example only)
```

GOAT gate at implementation: G1 determinism (seeded stochastic rollout reproduces the paper's three regimes on synthetic signed graphs), G2 latency vs a hand-rolled baseline (target: Plasma tier, µs for 32-agent graphs), G3 no-regression (default features untouched), G4 alloc-free steady state. Defaults from the paper's fitted values; promotion to default only if a production consumer lands (the CLR precedent: default-on only after the swarm consumer adopted it).

## 11. Open questions

- Is the continuous tanh relaxation (component 10) worth shipping alongside the discrete rule? The paper's async experiments favor it; our ticks are synchronous — defer until an async-consumer need appears.
- Multi-question extension (Potts/vector opinions) — the paper's own future work; our emotion axes (valence/arousal/fear/calm/desperation) are natural vector stances. Defer with the primitive's first consumer decision.
- Crowd-phase *forecast quality* on real game crowds (vs the paper's LLM crowds) is an unproven quality claim — any such claim requires the defend-wrong PoC + the conformal floor.

## See also

- Prior art: [arXiv:2605.10721](https://arxiv.org/abs/2605.10721) (De Marzo — the framing precedent), [arXiv:2608.02178](https://arxiv.org/abs/2608.02178) (De Nobili — LLM-agent criticality), [arXiv:2504.21582](https://arxiv.org/abs/2504.21582) (MF-LLM)
- In-stack: `.research/470` (phase separation), `riir-ai/.research/334` (phase separation guide), `.research/354` (NPT set attention), `.research/468` (Beckmann MFG)
- Substrate: `katgpt-rs/crates/katgpt-core/src/set_attention.rs` (CLR), `riir-ai/crates/riir-games/src/swarm/emotion.rs` (fear/aura channels), `riir-ai/crates/riir-agents/src/multi_agent.rs` (Sheaf conviction — mind the collision, §5)
- Sibling primitive (landed): `.research/542` + `katgpt-rs/crates/katgpt-core/src/qsg_gossip.rs` (Plan 589, arXiv:2603.24676 — Quantized Simplex Gossip, memetic drift on K-option simplex beliefs). Axis split — do not blur, compose: signed_coupling owns the TEMPERATURE axis T on binary stance over a persistent SIGNED GRAPH (memoryless heat-bath resample); qsg_gossip owns the BANDWIDTH m + ADAPTATION α axes on persistent simplex state (message sampling + EMA blend, graph-free uniform pairing). Same promotion posture: opt-in until a production consumer (riir-ai Issue 907).
