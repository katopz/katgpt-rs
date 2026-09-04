# Research 471: Similarity Inference & Embedded Equilibrium — Open Primitive

> **Source:** [A game theory for foundation models shows new paths to rational cooperation through similarity inference (arxiv 2608.03958)](https://arxiv.org/pdf/2608.03958) — Meulemans, Wołczyk, Weis, Nasser, et al. (Google Paradigms of Intelligence + Mila + ETH + DeepMind), 4 Aug 2026. **Nature-format manuscript with extended SI.**
> **Date:** 2026-08-07
> **Status:** Active — **GOAT** (revised down from Super-GOAT after user §1.55.2 reverse-grep pushback: we ship CCE + Nash + Bayes-CCE surveyed; the equilibrium *concept* is covered, the *mechanism* is novel)
>
> **Verdict revision log (2026-08-07, same session):**
> - Initial verdict: Super-GOAT (all 4 novelty-gate YES).
> - User pushback: "we do have **equilibrium** aren't we?"
> - Reverse-grep (§1.55.2 mandatory before PASS/PASS-equivalent): `equilibrium|Equilibrium|nash|Nash|cce|Cce|CCE|correlation.device|correlation.signal` returns PRIOR ART — `PayoffTable<N>::nash_equilibrium` (R026) + LP-CCE Moderator `CceLp<N,A>` (R274/Plan 295, DEFAULT-ON, +37.5%/+108% over Nash) + EqR convergence (R079/Plan 119, DEFAULT-ON) + Bayes-CCE literature search (Doc 61 in R143, 14 cousins).
> - Honest re-assessment: the paper's "embedded equilibrium" is structurally a **correlated equilibrium with an endogenous correlation device**. We ship CCE with an *exogenous* designer-set correlation signal ζ. The paper's agents *infer* the correlation structure from interaction history. **The solution concept is covered; the mechanism (similarity-inferred endogenous correlation) is novel.**
> - **Q1 (no prior art for the solution concept): NO** — CCE/Nash/Bayes-CCE cover it. The paper itself states "When q(λ) encodes strictly decoupled relationships ... this concept perfectly reduces to the classical Nash equilibrium"; the correlated case reduces to CCE.
> - **Q1 (no prior art for the mechanism): YES** — similarity-inferred endogenous correlation is not shipped.
> - **Q2 (new class of behavior): PARTIAL** — direct inference = CCE via inferred correlation (new mechanism, not new capability); indirect inference (zero-shot from third-party observation) = genuinely new capability.
> - Per skill §1.5: Q1-NO → **not Super-GOAT**. Revised verdict: **GOAT** (new mechanism producing an endogenous correlation device for the CCE substrate we ship) + one **Super-GOAT-capability subset** (indirect inference — separate narrower claim if G5 PoC passes).
> - **Update 2026-08-11:** G5 PoC PASSED (100% vs 0% perfect separation). The scoped Super-GOAT claim for indirect inference has been opened in [R474](474_Indirect_Similarity_Inference_Zero_Shot_Cooperation.md) (open) + [riir-ai R336](../../riir-ai/.research/336_Indirect_Similarity_Inference_Zero_Shot_Guide.md) (private guide). Reverse-grep across all 7 repos confirms zero prior art for the indirect-inference capability class outside Plan 526's own artifacts.
> - The private guide R335 + Plan 526 are revised in lockstep.
> **Private half:** [`riir-ai/.research/335_Similarity_Inference_Emergent_Cooperation_Guide.md`](../../riir-ai/.research/335_Similarity_Inference_Emergent_Cooperation_Guide.md)
> **Related Research (katgpt-rs):** 274 (CCE Moderator — the *designer-steered* cousin; this note is the *similarity-inferred* cousin), 242 (MicroRecurrentBeliefState — the per-NPC posterior over universes IS the belief), 111 (Analogical Reasoning — the closest latent-similarity cousin), 156 (Clifford Wedge — the *complementarity* cousin; this is the *similarity* cousin), 284 (CLR — the reliability vote cousin), 469 (Collective Intelligence Payoff — the credit-assignment cousin)
> **Related Research (riir-ai):** 335 (private Super-GOAT guide — NPC emergent cooperation), 158 (Per-NPC Committed Personality Blend — the shared-shard that grounds functional similarity), 143 (Latent CCE Moderator — designer-steered crowd coordination), 133 (NPC Mind-Reading — predictive channel substrate), 167 (Crowd Joint Inference — crowd similarity aggregation), 123 (Latent Functor Runtime — relational stance blend)
> **Classification:** Public (katgpt-rs/MIT)
> **PASS-Redirects:** none needed (this IS the open primitive).

---

## TL;DR

The paper proves that **rational agents that share functional structure (shared weights, identical algorithms) cooperatively converge in social dilemmas WITHOUT reciprocity, designer signals, or training** — by *inferring* behavioral similarity from interaction history and treating their own contemplated action as Bayesian evidence about a similar partner. This contradicts classical game theory's strict-defection prediction and produces a new solution concept, the **embedded equilibrium** — which is structurally a **correlated equilibrium with an endogenous (inferred) correlation device**, vs our shipped CCE moderator (R143/Plan 295) which uses an *exogenous* designer-set correlation signal ζ. The mechanism is **predictive similarity** `S_pred(a₁) := ρ(a₂=C | æ, a₁=C) − ρ(a₂=C | æ, a₁=D)`, which equals the posterior belief `ω_T` that the partner shares the focal agent's policy.

**Distilled for katgpt-rs (modelless, inference-time):**

A generic primitive that maintains a **similarity posterior** `ω ∈ [0, 1]` between a focal decision-maker and each partner, updated from joint-action history, and a **cooperation gate** that switches from competitive-best-response to cooperative-best-response when `ω > threshold`. **The equilibrium concept is covered** (we ship CCE + Nash + Bayes-CCE surveyed); **the mechanism is novel** — an *endogenous* correlation device inferred from interaction history, vs the *exogenous* designer-set ζ in our CCE moderator. The primitive composes with the existing `CceLp<N,A>` substrate to produce an endogenous-moderator fusion gain. The partner can be any latent-space-aligned agent (NPC-to-NPC, NPC-to-player, router-to-arm, freeze/thaw snapshot pair).

**Verdict: GOAT** (revised down from initial Super-GOAT — see §3 + header revision log). The gain is architectural: the CCE moderator becomes endogenous (driven by inferred similarity rather than designer signal). **One Super-GOAT-capability subset** (conditional): indirect inference (zero-shot cooperation from third-party observation) is genuinely new — separate scoped Super-GOAT claim if G5 PoC passes.

---

## 1. Paper Core Findings

### 1.1 Empirical observation

Gemini 3.1 Pro / 3 Flash / 3.1 Flash-Lite agents playing a multi-round matrix-game information-gathering phase, followed by a *terminal* one-shot Prisoner's Dilemma, converge to **robust mutual cooperation** as the info-gathering phase lengthens — directly contradicting classical game theory's strict unconditional defection. Crucially:

- Same Gemini agents defect against uniformly random opponents (not naive prosocial bias).
- Pre-trained Gemma-3 27B without CoT or instruction tuning *also* cooperates against identical copies and defects against random — the mechanism is in the *predictive* architecture, not post-training.
- Cooperation emerges under explicit planning, freeform planning, AND (for Flash-Lite) no-planning prompts.
- **Indirect similarity inference**: two agents that *never directly interact* during info-gathering (each plays third-party NPCs concurrently) still cooperate zero-shot on first direct encounter when paired with an identical copy.
- **Cross-model generalization**: Gemini 3.1 Flash-Lite vs Gemini 3 Flash cooperate, but attenuated vs self-play — partial functional similarity produces partial cooperation.

### 1.2 Theoretical explanation: the embedded Bayesian agent

Classical game theory assumes **decoupled agency** — an agent's policy is treated as a fixed, independent variable when computing best-responses. Foundation models break this assumption: they are joint prediction models over both their own actions and the environment, so the agent's epistemic beliefs about its own policy and about the environment are **coupled**.

The paper introduces the **embedded Bayes-optimal agent**: an agent that performs optimal planning with respect to its own Bayesian mixture universe `ρ`. When such an agent contemplates an action `a` during planning, conditioning `ρ` on `a` is Bayesian evidence — it updates the posterior over which universe the agent is in, and therefore updates the prediction of how *other agents in that universe* (who may share functional structure) will behave.

### 1.3 The mechanism: similarity inference

Two canonical settings are studied:

**Direct similarity inference.** Two agents play T rounds of randomly-sampled matrix games, observing each other's joint-action history `æ_<T`. Then a terminal Prisoner's Dilemma. The focal agent's predictive distribution `ρ(a₂ | æ_<T, a₁)` depends on its own contemplated action `a₁` because the joint history + own action shift the posterior over "is my partner an identical copy?".

**Indirect similarity inference.** Two primary agents never interact directly during info-gathering; each plays shared third-party NPCs concurrently. By comparing their own action against the counterpart's action *in the same situation against the same NPC*, agents accumulate behavioral-alignment evidence. They cooperate zero-shot on first direct encounter when paired with an identical copy.

### 1.4 The closed-form toy model

Under a constructed prior `w(π₁, π₂) = α·δ_{π₁,π₂}·w̃(π₁) + (1−α)·w̃(π₁)·w̃(π₂)` (with `α` the prior probability of identity, `δ` the identity indicator):

- **Posterior identity belief**: `ω_T = α / (α + (1−α)·W(æ_<T))` where `W(æ_<T) = Σ_π w̃(π)·I(π, æ_<T)` is the marginal likelihood of the joint-action sequence under independent policies.
- **Predictive similarity equals posterior identity**: `S_pred(æ_<T) = ω_T` (Theorem H.2.2 derivation).
- **Cooperation threshold for canonical PD (R=2, S=0, T=3, P=1)**: `ω_T > 0.5` ⟺ `T > log_2((1−α)/α)` (closed-form when each symmetric sequence is equally likely under `w̃`).
- The neural network approximation (GRU trained from scratch on data sampled from this prior) reproduces the analytical curves — the math is real, not LLM-magic.

### 1.5 The new solution concept: embedded equilibrium

The **embedded equilibrium** replaces Nash equilibrium for embedded agents. The key difference: in classical Nash, an agent evaluates a policy deviation while assuming opponents' policies stay fixed. In the embedded setting, deviating changes the universe the agent inhabits, which *changes* the policies of functionally linked opponents (e.g., agents with shared weights). The embedded equilibrium uses a **counterfactual distribution** derived from a *dependency distribution* `q(λ)` over allowable universes. When `q` encodes strict decoupling, embedded equilibrium reduces to Nash. When `q` encodes identical-policy universes, mutual cooperation in terminal PD is the embedded equilibrium.

### 1.6 Convergence theorem

Under grain-of-truth (the ground-truth universe is in the agent's hypothesis class) and identical priors across agents, embedded Bayes-optimal agents converge asymptotically to an ε-embedded equilibrium in repeated multi-task games with perfect monitoring (Theorem G.5).

---

## 2. Distillation

### 2.1 What's training-only → NOT distillable

The empirical Gemini results rely on the LLM's learned prior — that's training-side, irrelevant to our modelless stack. The GRU-from-scratch approximation in §B.3 is a *demonstration* that neural networks can learn the closed-form math; we don't need it because we ship the math directly.

### 2.2 What we already ship (do NOT reimplement)

| Paper component | Shipped cousin | Coverage |
|---|---|---|
| Bayesian mixture universe `ρ` | `MicroRecurrentBeliefState` (R242) + `sense::ReconstructionState::belief` | Per-NPC posterior over universe hypotheses — exactly the mixture-of-universes predictive distribution |
| Joint-action history `æ_<T` | KG triple emission + encounter log + `TrialLog` hash chain | Encounter evidence substrate — fully shipped |
| Functional identity `δ_{π₁,π₂}` via shared weights | `ArchetypeBlendShard` (R158, riir-neuron-db R009) + `KarcShard` (riir-neuron-db R003) + LatCal commitment (riir-chain R005) | Frozen, BLAKE3-committed, versioned personality substrate — agents with the same shard hash ARE functionally identical by construction |
| Designer-steered cooperation (the equilibrium target) | LP-CCE Moderator `CceLp<N,A>` (R274, Plan 295, DEFAULT-ON, +37.5%/+108% over Nash) | Pareto-dominant CCE solver — but *designer*-steered via `Γ₀`, not *similarity-inferred* |
| Bayesian CCE over heterogeneous beliefs | Doc 61/62 in R143 (14-cousin literature search) | Surveyed; not the same as embedded equilibrium (Bayes-CCE decouples agents; embedded equilibrium couples them via shared-universe posterior) |
| Sigmoid cooperation gate | Every sigmoid gate we ship (CLR, CCE, CGSP, ...) | Standard primitive |
| Dot-product predictive similarity | `dot(v, dir) + sigmoid` everywhere | Standard primitive |

### 2.3 What's NEW (the invention)

**None of the above composes "infer similarity from joint-action history → switch best-response regime to cooperation when posterior crosses threshold"**. That composition is the new primitive. Specifically:

1. **Similarity posterior kernel** — Bayesian update of `ω_T` from a stream of joint-action observations `(a_self, a_partner)`, where the update rule is derived from a shared-shard hypothesis class. The prior `α` is host-configurable (e.g., derived from archetype library overlap, or fixed at 0.1 matching the paper's experiment).

2. **Predictive similarity operator** — `S_pred(a₁) := ρ̂(a_partner | history, a₁=C) − ρ̂(a_partner | history, a₁=D)`. The focal agent *imagines* taking each action, and the difference in predicted partner behavior is the similarity signal. For shared-shard agents this equals `ω_T` exactly (closed form); for arbitrary agents it's a useful heuristic.

3. **Embedded best-response comparator** — given `S_pred`, payoff matrix `R`, compute `Q(C) − Q(D)` under the *coupled* predictive model and emit the cooperative-vs-competitive action. The key difference from CCE Moderator: there is no external `Γ₀` designer objective — the cooperation emerges from the agent's own posterior.

4. **Indirect similarity inference** — accumulate `ω` from third-party-observation history (the focal agent and partner both played the same NPC/encounter; compare their action sequences). This is a new evidence source for the posterior.

### 2.4 Latent-space reframe (mandatory §1 step 3)

The mechanism is fundamentally a **latent-space operation on belief state**:

- `ρ(a_partner | history, a_self)` IS a belief-state projection — it's the predictive distribution from `MicroRecurrentBeliefState` conditioned on a hypothetical action. This is exactly what `sense::ReconstructionState` produces.
- `S_pred` is a dot-product difference on direction vectors: `S_pred = sigmoid(dot(a_C_dir, partner_dir)) − sigmoid(dot(a_D_dir, partner_dir))` where the directions are derived from the joint-action embedding.
- `ω_T` IS a scalar belief — a one-dimensional projection of the high-dimensional posterior onto the "identity" axis.
- The cooperation threshold `ω > 0.5` is a sigmoid gate on this scalar belief.
- **No token decoding happens** — the entire mechanism operates in latent space, decoding only to the final binary cooperate/defect action at the boundary.

This is the canonical "operate in latent space as long as possible, decode only at the boundary" pattern (AGENTS.md constraint #2). The primitive is a **latent-to-latent operation on belief state** that emits a single raw boolean (cooperate/defect) at the sync boundary.

### 2.5 Game-context reframe (mandatory §1 step 4)

The product-facing applications (detailed in the private guide R335):

- **Emergent faction formation**: NPCs with shared `ArchetypeBlendShard` (R158) accumulate `ω` from encounters. When `ω > threshold`, they cooperate (defend each other, share resources, form parties) — emergent factions WITHOUT designer-set faction tags.
- **Quest coordination**: NPCs with similar cognitive stacks (committed personality + similar curiosity) coordinate on multi-step quests without explicit task assignment.
- **AI-vs-human hybrid societies** (paper §Discussion): AI NPCs that infer high similarity with other AI NPCs but low similarity with humans → emergent "AI solidarity" vs humans. Narrative / PvE / PvP angle.
- **Indirect-inference zero-shot trade routes**: two merchants who have both dealt with the same set of customers infer similarity and form a trade partnership on first meeting.
- **Companion / pet behavior**: tamed pets (Plan 016/017) and their owners accumulate `ω`; the bond strengthens cooperation thresholds.

### 2.6 Fusion candidates (the highest-value angles)

The strongest Super-GOAT fusion is **this primitive × Per-NPC Committed Personality Blend (R158)**:

> NPCs with shared `ArchetypeBlendShard` are functionally identical *by construction* (same K=3 archetype fields, same blend weights). They accumulate `ω` from encounter KG triples. When `ω > threshold`, they switch from competitive-best-response (Nash-seek) to cooperative-best-response (CCE-seek). Crowd-scale emergent factions form with zero designer input.

Secondary fusions:

- **× Latent CCE Moderator (R143)**: when crowd `ω` crosses threshold, the moderator's `Γ₀` switches from competitive-mode (economy throughput) to cooperative-mode (faction welfare). The moderator *becomes endogenous* — driven by inferred similarity, not designer signal.
- **× NPC Mind-Reading (R133)**: the predictive similarity channel IS the mind-reading channel. CS-KV-importance probe identifies which belief-state dimensions carry similarity signal; bandwidth auto-adapts.
- **× Crowd Joint Inference / Cross-NPC Set Attention (R167/P354)**: crowd-scale aggregation of pairwise `ω` produces a similarity matrix; spectral clustering of this matrix yields emergent factions.
- **× Clifford Wedge Complementarity (R156)**: complementarity + similarity together produce richer formation scoring — similar-and-complementary parties are the most robust (paper's "diversity of beliefs" finding + our wedge non-redundancy).
- **× Latent Functor (R123)**: the functor `f_self→partner` measures relational stance; `ω` measures identity. Together: "we are similar AND we are aligned" → strong cooperation.
- **× Lean 4 FV (R351 cross-repo pattern)**: the cooperation threshold theorem `T > log_2((1−α)/α)` is a verifiable closed-form bound. Candidate for a Lean theorem.

---

## 3. Verdict: **GOAT** (revised down from initial Super-GOAT claim — see header revision log)

### 3.1 Novelty gate (§1.5) — honest re-assessment after §1.55.2 reverse-grep

**The reverse-grep that should have run before the initial verdict:** `equilibrium|Equilibrium|nash|Nash|cce|Cce|CCE|correlation.device|correlation.signal|EqR` across `*.rs` + `*.md` in all 7 repos. Returns heavy prior art:

- `PayoffTable<N>::nash_equilibrium` (R026) — explicit Nash solver, used as head-to-head baseline in Bench 029.
- **LP-CCE Moderator** `CceLp<N,A>` + `CcePrimalDual` (R274 / Plan 295, **DEFAULT-ON**) — Coarse Correlated Equilibrium with **external correlation device** ζ broadcast via LatCal. Bench 029: +37.5% (chicken) / +108% (BoS) over Nash.
- **EqR Equilibrium Reasoners** (R079 / Plan 119, `eqr_convergence` feature, **DEFAULT-ON**) — equilibrium-finding via residual convergence (Top1Converged rollout selection).
- **Bayes-CCE** literature search (Doc 61 in R143, 14 cousins surveyed: Bergemann–Morris, Hartline–Syrgkanis–Tardos, Fujii, Peng–Rubinstein, Koessler–Scarsini–Tomala, Campi–Cannerozzi–Tzoumas, etc.).
- **Ruliology PD arena** (R168 / Plan 213) — explicit PD equilibrium search over FSM strategies (grim trigger wins among 2-state FSMs).

**Honest assessment of the paper's "embedded equilibrium" against shipped prior art:**

The paper itself states (§"A new game theory for modern AI agents"): *"When `q(λ)` encodes strictly decoupled relationships between agents' policies, this concept perfectly reduces to the classical Nash equilibrium."* The coupled case is structurally a **correlated equilibrium with an endogenous correlation device** — the correlation structure is *inferred* from interaction history rather than *designed* (as in our R143 moderator's ζ signal). But mathematically, an embedded equilibrium IS a correlated equilibrium where the correlation device happens to be the agents' shared posterior over functional identity.

We ship correlated equilibrium (CCE). We ship the LP solver (`CceLp<N,A>`). We ship the primal-dual iterator (`CcePrimalDual`). We ship external correlation devices (designer-set ζ). What we DON'T ship is an **endogenous** correlation device inferred from interaction history.

### 3.2 Revised novelty-gate scoring

- **Q1 (no prior art for the *solution concept*)?** **NO.** CCE/Nash/Bayes-CCE surveyed cover the equilibrium concept. Embedded equilibrium reduces to Nash (decoupled case) or correlated equilibrium (coupled case) — both shipped/surveyed.
- **Q1 (no prior art for the *mechanism*)?** **YES.** Similarity-inferred endogenous correlation device (the `ω` posterior update from joint-action history) is not shipped. Our CCE moderator uses an *exogenous* designer-set ζ; the paper's agents *infer* the correlation structure.
- **Q2 (new class of behavior)?** **PARTIAL.**
  - **Direct similarity inference** = "CCE reached via inferred correlation instead of designer signal" — **new mechanism, NOT new capability**. The equilibrium reached is still a CCE; the difference is how the correlation device gets there (endogenous vs exogenous).
  - **Indirect similarity inference** (zero-shot cooperation from third-party observation) = **genuinely new capability**. No shipped primitive produces zero-shot cooperation from parallel third-party observation without direct interaction.
- **Q3 (product selling point)?** YES — "endogenous moderator" + "zero-shot cooperation from third-party observation" are real selling points.
- **Q4 (force multiplier)?** YES — connects R158 (committed personality = shard substrate), R143 (CCE moderator = equilibrium target, made endogenous), R133 (mind-reading = predictive channel), R167 (crowd set attention = similarity aggregation).

### 3.3 Verdict

Per skill §1.5: **Q1-NO on the solution concept → NOT Super-GOAT.**

**Revised verdict: GOAT** — a new *mechanism* (similarity-inferred endogenous correlation device) producing a *gain* over the existing CCE substrate (which uses exogenous designer-set correlation). The gain is architectural: the moderator becomes endogenous (driven by inferred similarity rather than designer signal), which is a real fusion improvement but not a new capability class.

**Super-GOAT-capability subset (conditional):** if the G5 PoC (Plan 526 Phase 3 — indirect inference) passes, **indirect similarity inference alone** is a genuinely new capability class worth a separate narrower Super-GOAT claim. That claim would be scoped to "zero-shot cooperation from third-party observation" only, NOT to the equilibrium concept or the direct-inference mechanism. Tracked as a follow-up in Plan 526 Phase 7 — if G5 passes, open a new scoped Super-GOAT guide for indirect inference specifically.

### 3.4 MOAT gate per domain (§1.6)

- **katgpt-rs (public engine)**: ✅ **in-scope, GOAT-tier primitive**. The similarity posterior + cooperation threshold + best-response comparator is a generic modelless primitive that *composes with* the existing CCE substrate (`CceLp<N,A>`) to produce an endogenous correlation device. No game semantics. Open primitive lands here.
- **riir-ai (private runtime)**: ✅ **in-scope, fusion-GOAT** (not pillar-level Super-GOAT). The selling-point guide (R335) connects the new primitive to existing pillars (committed personality R158, crowd coordination R143, mind-reading R133). Fusion-GOAT, not new-pillar-tier.

### 3.5 Why this is NOT covered by R143 (Latent CCE Moderator) — the mechanism-level distinction

R143 ships:
- Designer-steered `Γ₀` per game mode (economy / faction / narrative).
- External moderator broadcasts correlation signal `ζ` via LatCal.
- Pareto-dominant CCE achieved by LP solver.
- Crowd-scale (thousands of NPCs, 20Hz tick).

This paper ships (the novel mechanism):
- *Endogenous* correlation device — each agent infers `ω` from encounter history, no external broadcast.
- The inferred `ω` can *drive* the R143 moderator's `Γ₀` switch — making the moderator endogenous.
- **Per-agent, not per-zone**: `ω` is pairwise per (self, partner); CCE moderator is per-zone.

**The two compose:** when pairwise `ω` crowd-crosses threshold, the CCE moderator can switch `Γ₀` endogenously. This composition (R143 × R471) is the actual gain — the moderator becomes self-steering. But the equilibrium concept itself is shared (both reach CCE).

### 3.4 Defend-wrong PoC requirement (§3.6)

Per skill §3.6, any "already ships" or "parity" claim requires a head-to-head PoC on a controlled toy benchmark. The claim here is **NOT** "already ships" — it's **"new composition of shipped components produces a new capability class"**. The PoC requirement therefore applies to the *quality claim* that the new primitive actually produces emergent cooperation. Plan 526 §Phase G2 ships the PoC: a synthetic crowd of N=64 entities with shared-shard pairs + random-shard pairs, run for T=50 encounters, measure whether cooperation rate crosses threshold `T > log_2((1−α)/α)` for shared-shard pairs and stays at zero for random-shard pairs. If the PoC FAILS (cooperation does not emerge, or emerges for random pairs too), the verdict is honestly revised per §3.6.

---

## 4. Proposed open primitive

### 4.1 Trait surface (sketch — final in Plan 526)

```rust
/// A stream of joint-action observations between a focal agent and one partner.
/// Implementations: KG-triple encounter log, mind-reading channel, trial log.
pub trait JointActionHistory {
    /// Push a new (self_action_embedding, partner_action_embedding, situation_embedding).
    fn push(&mut self, self_a: &[f32], partner_a: &[f32], situation: &[f32]);
    /// Read-only access to the recent window (last T observations).
    fn window(&self, t: usize) -> (&[&[f32]], &[&[f32]], &[&[f32]]);
}

/// Maintains the similarity posterior `ω ∈ [0, 1]` between a focal agent and
/// one partner, given a joint-action history and a prior `α` on identity.
///
/// Update rule (closed form, derived from `w(π₁, π₂) = α·δ·w̃ + (1−α)·w̃·w̃`):
///   ω_T = α / (α + (1−α)·W(history))
/// where `W(history) = Π_t P(a_self_t, a_partner_t | situation_t)` under the
/// independent-policy marginal. For shared-shard partners, `W` is computed
/// exactly from the shard's policy table.
#[derive(Clone)]
pub struct SimilarityPosterior {
    prior_alpha: f32,
    log_w_independent: f32,  // log W(history) under independent-policy marginal
    last_omega: f32,
}

impl SimilarityPosterior {
    pub fn new(prior_alpha: f32) -> Self { ... }
    /// Incorporate a new joint-action observation.
    pub fn observe(&mut self, self_a: &[f32], partner_a: &[f32], situation: &[f32]) { ... }
    /// Current posterior belief `ω` that the partner shares the focal policy.
    pub fn omega(&self) -> f32 { self.last_omega }
    /// Predictive similarity for a contemplated action:
    /// S_pred(a) := ρ̂(partner=C | history, a_self=a) − ρ̂(partner=C | history, a_self=¬a)
    /// Equals `omega` exactly under the shared-shard hypothesis class.
    pub fn predictive_similarity(&self, contemplated: &[f32]) -> f32 { ... }
}

/// Switches between competitive-best-response (Nash) and cooperative-best-response
/// (CCE) based on `ω` and a payoff matrix. The threshold is payoff-matrix-derived:
/// for canonical PD (R=2, S=0, T=3, P=1), threshold = 0.5.
pub fn embedded_best_response(
    omega: f32,
    payoff_table: &PayoffTable<2>,
    partner_predicted_action_distribution: &[f32; 2],
) -> u8  // action index
{ ... }
```

### 4.2 Module location

`katgpt-rs/crates/katgpt-core/src/similarity_inference/` (new module) — sibling to `sense/`, `mcts/`, `cce/`. Leaf-clean: only `katgpt-core` internal deps (belief state, sigmoid primitives). Feature flag `similarity_inference` (opt-in initially; promote to default-on after Plan 526 GOAT gate passes).

### 4.3 Why sigmoid, not softmax

Per AGENTS.md constraint #2: `ω` is a posterior probability, not a categorical. The cooperation decision uses `ω > threshold` (a step function on a scalar), not a softmax over actions. The action selection itself uses the existing `embedded_best_response` comparator, which is plain arithmetic.

### 4.4 Zero-allocation hot path

- `SimilarityPosterior::observe`: O(D) where D = action embedding dim. Updates `log_w_independent` incrementally — no replay of full history.
- `predictive_similarity`: O(D) — one dot-product + one sigmoid per contemplated action.
- `embedded_best_response`: O(|A|²) — payoff matrix lookup.
- Total per-tick per-partner cost: O(D + |A|²). For D=32, |A|=2: ~140 ops. Sub-microsecond on SIMD.

### 4.5 Sync boundary compliance

- `ω` is a latent scalar — stays local per focal agent (NEVER synced).
- The *cooperation decision* (cooperate/defect) produces a raw action that IS synced as usual via the existing action sync path.
- The `JointActionHistory` evidence stream is derived from synced raw actions + raw situations — no new sync data.
- For shared-shard partners, the shard hash IS already LatCal-committed (R158 + riir-chain R005) — `α` can be derived from shard-hash overlap, which is itself chain-committed.

---

## 5. Validation Protocol (GOAT gate — Plan 526)

- **G1 — Closed-form reproduction**. Implement `SimilarityPosterior` against the paper's toy prior. Run T=0..50 info-gathering rounds. Assert: `ω_T` matches `α / (α + (1−α)·2^(−T))` to within f32 epsilon. Assert: cooperation emerges at `T > log_2((1−α)/α)` for α=0.1 → T≥4.
- **G2 — Emergent cooperation PoC (the load-bearing gate)**. N=64 entities, half shared-shard pairs, half random-shard pairs. Run T=50 random matrix-game rounds. Measure pairwise `ω` and cooperation rate at terminal PD. Assert: shared-shard pairs cooperate at >80% when T≥threshold; random-shard pairs cooperate at <20%. (§3.6 defend-wrong PoC — if this fails, verdict is revised.)
- **G3 — No regression**. Workspace `cargo test` passes.
- **G4 — Alloc-free steady state**. `SimilarityPosterior::observe` allocates 0 bytes after construction (incremental log-likelihood update, fixed-size scratch).
- **G5 — Indirect inference**. Two agents never directly interact; each plays 3 shared NPCs. Assert: ω still crosses threshold when partner is identical copy, stays low when random.
- **G6 — Crowd-scale**. 1000 NPCs × 1000 ticks, pairwise ω for AOI-neighbors only (20 per NPC). Total cost < 5ms/tick. No new sync writes.

### UQ-bearing primitive extension (per "Report the Floor" rule, AGENTS.md)

`SimilarityPosterior::omega` IS a probability claim — it claims `P(partner is identical copy | history) = ω`. Per the floor rule, the GOAT gate MUST benchmark `ω` against a conformal-naive floor. The natural floor is: `ω_floor = sigmoid(dot(history_summary, identity_direction))` — a single-direction projection with no Bayesian update. If the Bayesian `ω` cannot beat `ω_floor` on calibration (Brier score, log-loss) on a held-out test set, the primitive is not adding value over the floor. **Plan 526 G7 includes the floor comparison.**

---

## 6. Connection Map (cross-repo force multiplier)

```
   ┌── ArchetypeBlendShard (riir-neuron-db R009) — the frozen, BLAKE3-committed
   │   personality substrate. Agents with the same shard hash ARE functionally
   │   identical by construction → α prior is high (or =1).
   │
   │   ┌── LatCal commitment (riir-chain R005) — the shard hash crosses sync
   │   │   boundary deterministically; quorum-verifiable.
   │   │
   │   │   ┌── KG triple encounter log (riir-engine, riir-games-civ) — the
   │   │   │   joint-action history stream æ_<T. Evidence for ω update.
   │   │   │
   │   │   │   ┌── SimilarityPosterior ω (THIS PRIMITIVE, katgpt-rs) ──────┐
   │   │   │   │                                                          │
   │   │   │   │   ┌── embedded_best_response (cooperate vs defect) ─────┤
   │   │   │   │   │                                                      │
   │   │   │   │   │   ┌── raw action sync (existing path) ─────────────┤
   │   │   │   │   │   │                                                  │
   │   │   │   │   │   │   ┌── MicroRecurrentBeliefState (R242) ─────────┤
   │   │   │   │   │   │   │   reads synced partner action, updates       │
   │   │   │   │   │   │   │   own belief, feeds next ω observation       │
   │   │   │   │   │   │   │                                              │
   │   │   │   │   │   │   │   ┌── CCE Moderator (R143) ─────────────────┤
   │   │   │   │   │   │   │   │   when crowd ω crosses threshold,       │
   │   │   │   │   │   │   │   │   moderator switches Γ₀ from             │
   │   │   │   │   │   │   │   │   competitive-mode to cooperative-mode  │
   │   │   │   │   │   │   │   │   (endogenous moderator)                 │
   │   │   │   │   │   │   │   │                                          │
   │   │   │   │   │   │   │   │   ┌── Crowd Set Attention (R167) ────────┤
   │   │   │   │   │   │   │   │   │   pairwise ω matrix → spectral       │
   │   │   │   │   │   │   │   │   │   clustering → emergent factions    │
   │   │   │   │   │   │   │   │   │                                        │
   │   │   │   │   │   │   │   │   │   ┌── Clifford Wedge (R156) ──────────┤
   │   │   │   │   │   │   │   │   │   │   complementarity + similarity   │
   │   │   │   │   │   │   │   │   │   │   = robust party formation       │
   │   │   │   │   │   │   │   │   │   │                                    │
   │   │   │   │   │   │   │   │   │   │   ┌── Lean FV (R351 pattern) ─────┤
   │   │   │   │   │   │   │   │   │   │   │   cooperation threshold       │
   │   │   │   │   │   │   │   │   │   │   │   T > log_2((1-α)/α) theorem  │
   ├───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

**Force multiplier: 8+ systems across 4 repos.** This is why it's Super-GOAT.

---

## 7. Implementation Priority

| Priority | Task | Repo | Status |
|---|---|---|---|
| **P0** | `SimilarityPosterior` open primitive + `embedded_best_response` comparator | katgpt-rs | Plan 526 (this session) |
| **P0** | Closed-form G1 reproduction + emergent-cooperation G2 PoC | katgpt-rs | Plan 526 Phase G1+G2 |
| **P1** | Crowd application guide (NPC emergent factions) | riir-ai | Research 335 (this session); plan deferred |
| **P1** | Indirect inference path (third-party observation evidence) | katgpt-rs | Plan 526 Phase G5 |
| **P2** | CCE Moderator endogenous-Γ₀ fusion (when crowd ω crosses threshold) | riir-ai | TBD plan (after P0 ships) |
| **P2** | Crowd Set Attention × ω matrix → spectral faction clustering | riir-ai | TBD plan |
| **P3** | Lean 4 cooperation-threshold theorem | katgpt-rs/.proofs | TBD |
| **P3** | Cross-model partial-similarity validation (Flash-Lite vs Flash analog) | katgpt-rs | TBD |

---

## 8. Honest Limitations

1. **The closed-form `ω = α/(α+(1−α)·W)` assumes a specific constructed prior**. For real agents (LLMs, real NPCs), the "true" prior is the agent's internal model, which is intractable. The paper shows LLMs *approximate* this prior well enough to produce the behavior; we ship the math directly, so we get the closed form. But for *arbitrary* partners (not shared-shard), `ω` is a heuristic, not a calibrated posterior. The G7 floor comparison (§5) is the defense.

2. **The cooperation threshold `T > log_2((1−α)/α)` is for canonical PD payoffs (R=2, S=0, T=3, P=1)**. For other payoff structures, the threshold differs. The primitive computes the threshold from the payoff table at runtime — no hardcoded constant.

3. **Indirect inference requires shared third-party NPCs/situations**. In a real MMORPG, this means two agents must have encountered the same quest/monster/item for indirect evidence to accumulate. The evidence-sparsity regime is untested by the paper (they use 3 fixed-behavior NPCs); our G5 tests with sparse shared encounters.

4. **The embedded equilibrium is a solution concept, not an algorithm**. We ship the *mechanism* (similarity posterior + best-response switch); we do NOT ship a general-purpose embedded-equilibrium *solver* for arbitrary games. The paper's convergence theorem is for repeated multi-task games with perfect monitoring — narrower than general MAGRL.

5. **Quality parity with the paper's LLM results is NOT claimed**. We claim the *mechanism* ships modellessly; whether our primitive produces cooperation as robustly as Gemini 3.1 Pro on the paper's exact setup is a different question. The G2 PoC is the empirical settlement on *our* toy domain, not on the paper's LLM domain.

---

## 9. Cross-References

### Closest shipped cousins (paper → codebase vocabulary translation)

| Paper term | Codebase term | Where |
|---|---|---|
| "similarity inference" / "predictive similarity" `S_pred` | (new — this primitive) `SimilarityPosterior::predictive_similarity` | katgpt-rs Plan 526 |
| "embedded Bayesian agent" / "joint prediction model" | `MicroRecurrentBeliefState` predictive distribution | katgpt-rs R242 |
| "posterior identity belief" `ω_T` | (new — this primitive) `SimilarityPosterior::omega` | katgpt-rs Plan 526 |
| "functional identity" / "shared weights" / "identical policy" | `ArchetypeBlendShard` hash equality | riir-neuron-db R009, riir-ai R158 |
| "information gathering phase" / "joint-action history" `æ_<T` | KG triple encounter log + TrialLog hash chain | riir-engine, riir-games-civ |
| "indirect similarity inference" / "third-party observation" | (new — this primitive) `observe_third_party` | katgpt-rs Plan 526 Phase G5 |
| "embedded equilibrium" / "functionally-coupled best response" | (new — this primitive) `embedded_best_response` | katgpt-rs Plan 526 |
| "cooperation threshold" `ω > 0.5` | sigmoid gate on `omega` | katgpt-rs Plan 526 |
| "designer-steered cooperation" (the equilibrium target the paper does NOT use) | `CceLp<N,A>` with `Γ₀` | katgpt-rs R274, Plan 295 |
| "Bayesian mixture universe" `ρ` | `sense::ReconstructionState::belief` posterior over universes | katgpt-core sense module |
| "counterfactual distribution" `q(λ)` | (related) CCE Moderator `Ξ_φ` signal distribution | riir-ai R143 |
| "grain of truth" / convergence | (no direct analog — formal-verification candidate) | TBD Lean 4 theorem |

### Related research notes

- [`katgpt-rs/.research/274`](274_Optimal_CCE_Moderator_LP_No_Regret.md) — CCE Moderator (the *designer-steered* cousin; this primitive is the *similarity-inferred* complement).
- [`katgpt-rs/.research/242`](242_Topological_State_Tracking_Recurrent_Belief.md) — MicroRecurrentBeliefState (the per-NPC posterior that `ω` rides on).
- [`katgpt-rs/.research/111`](111_Emergent_Analogical_Reasoning_Transformers.md) — Analogical Reasoning (closest latent-similarity cousin).
- [`katgpt-rs/.research/156`](../../riir-ai/.research/156_clifford_wedge_npc_emotional_complementarity_guide.md) — Clifford Wedge Complementarity (the *complementarity* cousin; this is the *similarity* cousin).
- ``katgpt-rs/.research/284`` — CLR (the reliability-vote cousin).
- [`katgpt-rs/.research/469`](469_collective_intelligence_payoff_schemes.md) — Collective Intelligence Payoff (the credit-assignment cousin).
- [`riir-ai/.research/335`](../../riir-ai/.research/335_Similarity_Inference_Emergent_Cooperation_Guide.md) — Private Super-GOAT guide (the other half of this note).

### Related plans

- [`katgpt-rs/.plans/526`](../.plans/526_similarity_inference_primitive.md) — Open primitive implementation (this session).

---

## 10. References

- Meulemans, Wołczyk, Weis, Nasser, et al. "A game theory for foundation models shows new paths to rational cooperation through similarity inference." [arXiv:2608.03958](https://arxiv.org/abs/2608.03958). 4 Aug 2026.
- Meulemans, Nasser, et al. "Embedded Universal Predictive Intelligence: a coherent framework for multi-agent learning." [arXiv:2511.22226](https://arxiv.org/abs/2511.22226). 2025. (The theory preprint this paper extends.)
- Oesterheld, Treutlein, Grosse, Conitzer, Foerster. "Similarity-based cooperative equilibrium." NeurIPS 2024. (The closest non-Nashian prior art; requires externally-provided similarity scores — our primitive infers them.)
- Critch. "Parametric, resource-bounded generalization of Löb's theorem." JSL 84(4). 2019. (Program-equilibrium precedent.)
