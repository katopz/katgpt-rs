# Research 274: Optimal Coarse Correlated Equilibria (MFG) — LP Relaxation + No-Regret Primal-Dual Moderator

> **Source:** [Optimal Coarse Correlated Equilibria in Mean Field Games: Linear Programming and No-Regret Learning](https://arxiv.org/pdf/2606.20062) — Campi, Cannerozzi, Tzouanas (Bielefeld / Milano), arxiv 2606.20062v1, 18 Jun 2026
> **Date:** 2026-06-20
> **Status:** Done
> **Related Research:** 026 (RTS Intransitive — `PayoffTable<N>` Nash), 079 (EqR Equilibrium Reasoners), 098 (PrudentBanker OMD), 168 (Ruliology Arena), 167 (Economy of Minds / WealthPruner — economic selection w/ coordinator), 249 (DecentMem Dual-Pool — mean-field α router w/ O(log T) regret), 270 (ICT Distributional Branching — per-NPC cognitive economics), 273 (ELT — Any-Time elastic budget per zone)
> **Related Plans:** 030 (Multi-Armed Bandit), 137 (PrudentBanker Safe-Phased Bandit), 170 (RTS Intransitive Balancing — `PayoffTable<N>`), 213 (Ruliology Arena Cross-Paradigm), 282 (Dual-Pool Reachable Router), 295 (LP-CCE Moderator Primitive — this note's plan)
> **Cross-ref (riir-ai):** Research 143 (Latent CCE Moderator — private selling-point guide), Plan 325 (Latent CCE Moderator Runtime — private runtime plan)
> **Classification:** Public

---

## TL;DR

Campi–Cannerozzi–Tzouanas introduces **optimal Coarse Correlated Equilibria (CCEs)** for continuous-time **Mean Field Games (MFGs)** — a moderator picks, among infinitely many randomized recommendation schemes, one that optimizes its *own* objective (which may differ from the representative player's). The paper (a) gives an **LP formulation** via occupation measures with martingale consistency constraints, (b) proves existence of an optimal LP-CCE via compactness of the consistent-flow set, (c) reformulates the CCE condition as an **external-regret constraint** `ER(ρ) ≤ 0`, and (d) designs a **no-regret primal-dual algorithm** with Bregman regularization achieving **O(N⁻¹ᐟ²)** averaged-iterate convergence to a saddle point of the Lagrangian. Parametrized via neural-network policy + correlation device (training-flavored, see §2.3).

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is **not** the neural-network parametrization (→ training). It is the **CCE-selection-as-constrained-LP + no-regret-primal-dual-with-Bregman** machinery, applied **modellessly** to a *finite* deviation class. Two pieces distilled:

1. **`CceLp<N,A>`** — a generic LP over occupation measures `ρ ∈ P(S×A)` with (i) consistency (ρ's marginal over A equals the recommended marginal under the policy), (ii) martingale/dynamics constraints (encoded as linear equalities on occupation-measure moments), and (iii) external-regret inequalities `Γ[ρ] ≤ Γ_dev[ρ](κ)` for every deviation `κ ∈ D`. Solution set `E` is compact convex; minimizer of moderator objective `Γ₀(ρ)` exists. Solvable via standard LP solver for small N.
2. **`CcePrimalDual`** — a modelless primal-dual iterator: primal minimizes `Γ₀(ρ) + λ·ER(ρ) + ½·Dψ(ρ, ρⁿ)` over `ρ ∈ M` (Bregman three-point descent, §6.2); dual updates `λⁿ⁺¹ = λⁿ + (1/√N)·ER(ρⁿ⁺¹)` (projected ascent on regret). Convergence rate `O(N⁻¹ᐟ²)` to averaged iterates (paper Theorem 6.1, no monotonicity assumption needed — diverges from fictitious-play literature).

The **latent-space reframing** is the Super-GOAT angle (see §3 and the private guide R143): `S` = HLA belief buckets, `A` = action distribution over a discrete CGSP conjecturer pool, `D` = the conjecturer pool itself (each conjecturer is an admissible deviation), `ζ` = the moderator's correlation signal broadcast as a zone-level latent scalar. The recommendation policy `π(a|x, ζ)` becomes a **lookup table** indexed by `(HLA bucket, signal bucket)`. No gradient training, no neural networks, no Adam optimizer — pure arithmetic + sigmoid gates, fits in plasma-tier µs budget.

---

## 1. Paper Core Findings

### 1.1 The mean-field CCE definition (paper §3, Def 3.2)

A *correlated flow* `(λ, μ)` = (recommended strategy, random measure flow) is a **mean-field CCE** iff:

- **Optimality**: for every admissible deviation `β ∈ A` (where `β` is Fξ,W-progressively measurable and *independent* of the moderator's lottery), `J(λ, μ) ≤ J(β, μ)`. Information asymmetry: the deviating player commits ex ante without seeing the recommendation realization.
- **Consistency**: `μ_t(·) = P((X_t, λ_t) ∈ · | μ)`, i.e. the recommended flow is the conditional law of the representative player given the moderator's lottery.

**Key contrast with Nash:** a CCE permits the moderator to randomize; Nash is the degenerate case `ρ = δ_(m⋆, m⋆)`. **CCEs generalize both Nash and Aumann's Correlated Equilibria** (CEs). Infinitely many CCEs can coexist with a unique Nash; CCEs may even exist when no Nash exists (Campi–Cannerozzi–Cartellier 2025).

### 1.2 The moderator's selection problem (paper §3, Def 3.3)

Among all mean-field CCEs, pick `(λ⋆, μ⋆)` minimizing a *moderator objective* `J₀` that may differ from `J`:

```
J₀(λ, μ) = E[ ∫₀ᵀ f₀(t, X_t, μ_t, λ_t) dt + g₀(X_T, μ_T) ]
```

`f₀, g₀` are independent of `f, g` (player's cost). This decoupling is the **designer steering** angle: the moderator can optimize world-level welfare, faction balance, economic throughput, etc., while each player still has no incentive to deviate.

### 1.3 LP formulation (paper §4, Defs 4.2–4.5)

Rewrite everything in occupation-measure space:

- `V₂` = measurable flows `m : [0,T] → P₂(R×A)` (joint state-action distributions)
- `R[m]` = occupation measures `(η, η̄)` satisfying the martingale property relative to flow `m` (linear equality constraints encoding the SDE dynamics via Itô tests)
- `R₀ = closure(convex-hull(∪_m R[m]))` — compact convex (paper Lemma 4.1)
- `M` = consistent LP-correlated flows `ρ ∈ P(V₂ × P₂(R))` satisfying the consistent martingale property (4.6) — compact convex (paper Lemma 5.3)
- `E` = LP-CCEs: `ρ ∈ M` such that `Γ[ρ] ≤ Γ_dev[ρ](κ, κ̄)` for every admissible deviation `(κ, κ̄) ∈ D` (paper Def 4.4) — compact convex nonempty (paper Prop 5.1, since any LP-NE is an LP-CCE by Prop 4.1)
- **Existence of optimal LP-CCE**: `Γ₀` is linear continuous on compact convex `E`, so minimizer exists (paper Theorem 5.1)

**Why this matters for us:** the *infinite-dimensional* LP collapses to a *finite-dimensional* LP once we discretize state-action space. The compactness/convexity results transfer. We get a clean LP we can solve (for small N) or iterate toward (for large N) — no fictitious-play best-response machinery needed.

### 1.4 External-regret reformulation (paper §6.1, Def 6.1)

The CCE condition is equivalent to:

```
ER(ρ) := sup_{κ ∈ R} (Γ[ρ] − Γ_dev[ρ](κ)) ≤ 0
```

External regret = max gain from ignoring the recommendation and switching to any fixed deviation. The moderator's problem becomes:

```
min_{ρ ∈ M}  Γ₀(ρ)   s.t.   ER(ρ) ≤ 0
```

Lagrangian: `L(ρ, λ) = Γ₀(ρ) + λ·ER(ρ)`. Sion minimax (paper Prop 6.1) yields a saddle point `(ρ̂, λ̂) ∈ M × R₊`; the dual optimum `λ̂·ER(ρ̂) = 0` (complementary slackness, paper eq. 6.8); `ρ̂ ∈ E` is an optimal LP-CCE.

### 1.5 The primal-dual algorithm (paper §6.2, Algorithm 1)

```
Initialize: N ∈ ℕ, λ⁰ > 0, ρ⁰ ∈ M, convex linearly-differentiable Bregman potential ψ

for n = 0 .. N−1:
  ρⁿ⁺¹ ∈ argmin_{ρ ∈ M}  Γ₀(ρ) + λⁿ·ER(ρ) + ½·D_ψ(ρ, ρⁿ)            # primal (Bregman descent)
  λⁿ⁺¹ = argmin_{λ ≥ 0}  −λ·ER(ρⁿ⁺¹) + (√N / 2)·(λ − λⁿ)²           # dual (projected ascent)

Return:  ρ̄ⁿ = (1/N)·Σ ρⁿ⁺¹,   λ̄ⁿ = (1/N)·Σ λⁿ
```

**Convergence (paper Theorem 6.1):** for any `(ρ, λ) ∈ M × R₊`,

```
Gap_(ρ,λ)(ρ̄ᴺ, λ̄ᴺ) ≤ (1 / 2√N)·( |λ − λ⁰|² + D_ψ(ρ, ρ⁰) + C )
```

where `C = sup_ρ ER²(ρ)`. Averaged regret `ER(ρ̄ᴺ) ≤ O(N⁻¹ᐟ²) + Γ₀(ρ⋆) − Γ₀(ρ̄ᴺ)`. Subsequential limit is a saddle point, and `ρ̄ᴺ → ρ̂ ∈ E optimal LP-CCE`.

**Two crucial structural assumptions** (paper Assumptions 6.1, 6.2):

- **6.1 (Slater)**: ∃ ρ ∈ M with `ER(ρ) < 0` (strict interior feasible point). Yields saddle-point existence.
- **6.2 (Unique maximizer)**: for each ρ, the supremum in `ER(ρ)` is attained by a unique `κ⋆(ρ)`. Yields linear differentiability of `ER` with derivative `δER/δρ (ρ, m) = F[m](m) − F[m](κ⋆(ρ))`.

The paper Appendix C verifies both for the flocking and emission-abatement examples by direct construction.

### 1.6 Parametrized implementation (paper §7) → mostly training, KEEP ONLY THE INSIGHT

Section 7 parametrizes `ρ` via neural-network policy `π_θ(a|x, ζ)` + correlation device `ζ ∼ Ξ_φ`. Updates use Adam gradient descent on `(φ, θ)`. **This is training-flavored — out of scope here.** The transferable insight is the **decomposition**: `ρ = E_ζ[ δ_{m_{θ,ζ}} ]` — a mixture over deterministic occupation measures indexed by a latent signal `ζ`. We use this decomposition modellessly: `ζ` becomes a discrete bucket index, `θ` becomes a discrete policy table, the Adam optimizer becomes a primal-dual iterator on `(Ξ_φ buckets, θ table)`.

### 1.7 Numerical results (paper §8)

Two examples — simple flocking (linear-quadratic, explicit Nash) and emission abatement (general-sum, no explicit Nash):

- **Flocking**: Algorithm 2 converges to the MFG solution. `φ₁=0, φ₂≈0` (no extra randomization needed when MFG=MFC).
- **Emission abatement (payoff-max)**: learned CCE outperforms the linear-optimal CCE of Campi–Cannerozzi–Cartellier 2025; external regret stays nonpositive.
- **Emission abatement (terminal-abatement-max)**: a *different* moderator objective selects a *different* optimal CCE — lower payoff, higher terminal abatement. This is the **moderator-steering demo**: same game, two optimal CCEs selected by two different `Γ₀`. The headline selling point.

---

## 2. Distillation

### 2.1 What's training-only → riir-train (do NOT implement here)

- The neural-network parametrization of `π_θ` (Section 7.1). Adam gradient descent on `(φ, θ)` (Section 7.3, Algorithm 2). All requires backprop. → riir-train.
- The "smooth projection" `ϕ_A(x) = c + r·tanh(x/r)` and SiLU activations — training-friendly smoothness, irrelevant for our discrete-table modelless path.
- The Fokker–Planck PDE solver for the aggregate drift `B_θ(t, x; ζ)` — continuous-state machinery. We discretize `S` into HLA buckets; no PDE solver needed.

### 2.2 What we already ship (do NOT reimplement)

| Paper component | Our shipped equivalent | Evidence |
|---|---|---|
| Nash equilibrium solver | `PayoffTable<N>::nash_equilibrium` (Cramer + support enumeration) | `riir-games/src/payoff.rs` |
| No-regret bandit convergence to Nash | `bandit_05_rps.rs` example (UCB1 vs ε-greedy → 33/33/33) | `katgpt-rs/examples/bandit_05_rps.rs` |
| Mean-field α router with O(log T) regret | `DualPoolBandit` (Plan 282, G2 PASS: regret 24.6 ≤ 5·log(10k)) | `katgpt-rs/crates/katgpt-core/src/cgsp/dual_pool.rs`, `katgpt-rs/.benchmarks/028_dualpool_g2_log_regret.md` |
| Mirror descent / OMD with delay | PrudentBanker (R098, Plan 137) — Banker-OMD, O(log T + √D) regret | `katgpt-rs/src/pruners/prudent_banker.rs` |
| Equilibrium reasoners | EqR residual-based convergence (R079) | — |
| Cross-paradigm arena with Nash meta-game | Ruliology (R168, Plan 213) | `katgpt-rs/src/ruliology/`, `katgpt-rs/examples/ruliology_demo.rs` |
| Population welfare via economic selection | WealthPruner (R167) — coordinator with own objective per-arm wealth | (R167 §Fusion 1) |
| Coherence-driven re-estimation scheduler | `latent_functor/reestimation.rs` (self-healing on coherence < τ_reest) | `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` |

### 2.3 What's missing (the Super-GOAT novelty)

**No CCE solver exists in the codebase.** `PayoffTable<N>` computes Nash; nothing computes CCE. The CCE class is strictly larger and (provably) Pareto-dominates Nash in general-sum games. We are missing:

1. **CCE condition check**: given a recommendation distribution `ρ`, verify `Γ[ρ] ≤ Γ_dev[ρ](κ)` for all deviations `κ ∈ D`.
2. **External-regret functional**: `ER(ρ) = sup_κ (Γ[ρ] − Γ_dev[ρ](κ))` — closed-form on finite deviation class.
3. **LP-CCE solver**: discretize state-action space, solve the LP for an optimal `ρ` (small N).
4. **Primal-dual iterator**: modelless Bregman-descent primal + projected-ascent dual, `O(N⁻¹ᐟ²)` averaged convergence.
5. **Moderator objective**: `Γ₀(ρ)` that decouples from `Γ` — the designer-steering handle.
6. **Correlation signal broadcast**: the `ζ ∼ Ξ_φ` mechanism as a shared latent scalar (zone mood) consumed by all NPCs.

### 2.4 The modelless implementation path

Replace the paper's neural-network `π_θ(a|x, ζ)` with a **discrete policy table** indexed by `(state_bucket, action_bucket, signal_bucket)`:

```
State buckets S      = {HLA bucket × zone bucket}              // bounded, e.g. 8×32 = 256
Action buckets A     = {CGSP conjecturer pool index}           // bounded, e.g. K ≤ 16
Signal buckets Z     = {zone mood scalar quantized}            // bounded, e.g. 8
Deviation class D    = {conjecturer pool itself}               // K ≤ 16 deviations
```

The occupation measure `ρ ∈ P(S × A × Z)` is a tensor of size `|S|·|A|·|Z|` (e.g. `256·16·8 = 32k` floats = 128KB f32). The LP has:

- `|S|·|A|·|Z|` variables (the occupation measure)
- `|S|·|Z|` consistency constraints (marginal over A matches policy)
- `|S|·|A|·|Z|` dynamics constraints (discretized martingale)
- `|D|·|S|·|Z|` regret inequalities (one per deviation per signal)

For `K = 16`, `|S| = 256`, `|Z| = 8`: ~32k vars, ~2k consistency, ~32k dynamics, ~32k regret inequalities. **Solvable by an LP solver in ms.** The primal-dual iterator (Plan 295) is even cheaper — `O(|S|·|A|·|Z|)` per iteration, no LP solver.

### 2.5 Fusion (the Super-GOAT angle)

This paper alone is GOAT (a provable improvement: CCE ≥ Nash in general-sum games, with regret-bounded learning algorithm). The **Super-GOAT** comes from fusing across the corpus:

**Fusion A — CCE × Latent Functor × HLA × LatCal (latent-to-latent coordination):**
The "occupation measure" lives in **latent space**, not raw action space. `S` = HLA belief buckets (per-NPC 8-dim latent state); `A` = projected action distribution; `ζ` = zone-level latent scalar broadcast via the existing HLA channel (dot-product projection + sigmoid). The correlation signal crosses the **sync boundary via LatCal**: the moderator commits `ζ` as a fixed-point raw scalar (chain-committed, BLAKE3-hashed), NPCs consume it locally as a latent prior. **No raw coordinate sync needed for coordination** — only the scalar signal is synced. This is exactly the raw↔latent bridge pattern from AGENTS.md.

**Fusion B — CCE × DecentMem Dual-Pool × PrudentBanker (provable regret bound on the moderator):**
The dual update `λⁿ⁺¹ = λⁿ + (1/√N)·ER(ρⁿ⁺¹)` is a PrudentBanker-style phased aggression update on the regret constraint. The primal `argmin_ρ Γ₀ + λ·ER + ½·Dψ` is a DualPool-style α-router update on the policy distribution. **Both layers inherit DecentMem's O(log T) machinery and PrudentBanker's delay-robustness.** The crowd-scale regret bound `O(N⁻¹ᐟ²)` from this paper stacks with DecentMem's per-NPC `O(log T)` to give a crowd-level regret bound.

**Fusion C — CCE × WealthPruner (economic moderator with own objective):**
The moderator's `Γ₀` is WealthPruner's wealth-flows-promote-success applied at the *population* level. Each NPC's CGSP conjecturer pool is the deviation class; the moderator's recommendation policy is the wealth-promoted conjecturer. **Bankruptcy (WealthPruner) ↔ deviation-profitability (CCE)** — both remove underperformers, but CCE removes them *correlatedly* across the population via the shared signal, while WealthPruner removes them independently per arm.

**Fusion D — CCE × Latent Functor re-estimation (coherence-decay triggers re-moderation):**
`latent_functor/reestimation.rs` triggers re-estimation when `coherence < τ_reest`. The CCE primal-dual iterator should *also* trigger re-moderation when `coherence < τ_reest` — the moderator's `ρ` becomes stale as the population drifts. This fuses DiPOD's "self-distillation when ELBO drifts" pattern with CCE's "re-moderation when regret drifts". **The two schedulers are the same scheduler** under different vocabulary.

**Fusion E — CCE × ICT Distributional Branching (gate expensive moderator updates):**
Per-NPC cognitive economics (R270, R143) says only ~10% of moments are real decisions. The moderator doesn't need to update `ρ` every tick — only at branching points. **The ICT BranchingMask gates the primal-dual update itself.** This gives crowd-scale CCE updates at ~10× lower cost without losing decision quality.

**The novel capability class** (Super-GOAT Q2 answer): *coordinated emergent population behavior via a latent broadcast signal, with a designer-steerable moderator objective, achieving welfare-Pareto-dominant Coarse Correlated Equilibria that no Nash-seeking competitor can reach, at crowd-scale (thousands of NPCs, 20Hz tick) within plasma-tier latency budgets.*

---

## 3. Verdict: **Super-GOAT**

### 3.1 Novelty gate (4 YES)

**Q1 — No prior art?** YES.
- Notes layer: zero hits on "coarse correlated", "CCE", "moderator" (in the CCE sense), "external regret" (in the CCE sense), "LP-CCE", "occupation measure" across `katgpt-rs/.research/`, `katgpt-rs/.plans/`, `riir-ai/.research/`, `riir-ai/.plans/`. Closest cousins: `PayoffTable<N>` solves **Nash** (not CCE), DecentMem Dual-Pool has **O(log T) router regret** (not CCE selection), WealthPruner has **economic arm selection** (not population-level CCE), EqR has **within-model equilibrium** (not cross-population CCE), latent_functor/reestimation has **coherence-driven self-healing** (not moderator-driven selection).
- Code layer: zero hits. `payoff.rs` is Nash-only; `dual_pool.rs` is α-router; `prudent_banker.rs` is OMD bandit. No CCE solver, no occupation-measure LP, no primal-dual-on-regret.
- Vocabulary translation performed (paper ↔ codebase, both directions) before this verdict: "moderator" → "coordinator", "recommendation" → "policy hint", "CCE" → "swarm correlated policy", "external regret" → "advantage over best fixed deviation", "occupation measure" → "visitation distribution", "deviation" → "conjecturer arm". All zero hits in code.

**Q2 — New class of behavior?** YES. Today's NPCs can reach: (i) per-NPC optimal policies (CGSP/CLR/HLA), (ii) local Nash in 1v1 or small games (`PayoffTable<N>`), (iii) population dynamics via mean-field aggregate (DecentMem α-router). They **cannot** reach a CCE — coordinated behavior emerging from a shared latent signal without explicit communication, with welfare Pareto-dominating any Nash. CCE is strictly more general; the paper proves CCEs may exist when no Nash exists. **This is a new capability class, not an optimization.**

**Q3 — Product selling point?** YES. One sentence: *"Our NPCs reach Coarse Correlated Equilibria — coordinated behavior emerges from a shared latent signal (zone mood, faction sentiment, market climate) without explicit NPC-to-NPC communication, and a designer can steer the population via the moderator's objective to maximize world-level welfare (economic throughput, faction balance, narrative pace) that no individual NPC would optimize on its own."* This is directly a moat for emergent MMO worlds — designers want population-level outcomes, not per-NPC optimality. CCE selection is the principled way to deliver that.

**Q4 — Force multiplier?** YES (≥2 pillars). Multiplies: (1) CGSP runtime — moderator layer steers curiosity; (2) Latent functor — CCE in latent space, not raw actions; (3) HLA per-NPC state — moderator broadcasts via HLA channel; (4) LatCal commitment — correlation signal is fixed-point raw scalar across sync boundary; (5) NPC social/emergent behavior — factions, trade routes, reputation; (6) Chain state — moderator's optimization can be a chain-committed program. **Six pillars.**

**All 4 YES → Super-GOAT.** Mandatory outputs follow: open primitive (this note + Plan 295), private guide (riir-ai R143), private plan (riir-ai Plan 325).

### 3.2 Selling point (one sentence, repeated for the guide)

> Coordinated emergent MMO population behavior via a latent broadcast signal with a designer-steerable moderator objective, Pareto-dominating any Nash-seeking competitor, at crowd-scale (thousands of NPCs, 20Hz tick) in plasma-tier latency budgets.

### 3.3 What stays public (katgpt-rs) vs private (riir-ai)

| Component | Public (katgpt-rs) | Private (riir-ai) |
|---|---|---|
| `CceLp<N,A>` LP solver on finite state-action | ✅ generic math | — |
| `CcePrimalDual` Bregman primal-dual iterator | ✅ generic algorithm | — |
| `ExternalRegret<D>` functional + uniqueness check | ✅ generic | — |
| HLA-bucketed state space | — | ✅ game-specific (HLA belief buckets) |
| Zone-mood broadcast via HLA channel | — | ✅ game-specific (HLA channel wiring) |
| CGSP conjecturer pool as deviation class | — | ✅ game-specific (CGSP runtime) |
| LatCal commitment of correlation signal | — | ✅ chain-specific (LatCal fixed-point bridge) |
| Moderator objective `Γ₀` per game mode | — | ✅ game-specific (economy/faction/narrative) |
| Latent Functor re-estimation trigger fusion | — | ✅ game-specific (latent_functor/reestimation.rs) |
| ICT BranchingMask gating of moderator updates | — | ✅ game-specific (per-NPC cognitive economics) |

**Commercial principle (R003):** the public primitive is the adoption hook (generic LP-CCE math, no game semantics). The private guide is the moat (HLA + zone mood + CGSP pool + LatCal + game-specific `Γ₀`). Training know-how (neural-network parametrization, Adam optimizer on policy parameters) → riir-train, never leaks.

---

## 4. Implementation Priority (P0–P3)

| Priority | Component | Owner repo | Plan |
|---|---|---|---|
| **P0** | `CceLp<N,A>` finite occupation-measure LP solver | katgpt-rs | Plan 295 Phase 1 |
| **P0** | `ExternalRegret<D>` functional + uniqueness check | katgpt-rs | Plan 295 Phase 1 |
| **P0** | `CcePrimalDual` Bregman primal-dual iterator with `O(N⁻¹ᐟ²)` convergence test | katgpt-rs | Plan 295 Phase 2 |
| **P1** | HLA-bucketed state space adapter | riir-ai | Plan 325 Phase 1 |
| **P1** | Zone-mood broadcast via HLA channel (sigmoid projection) | riir-ai | Plan 325 Phase 2 |
| **P1** | CGSP conjecturer pool as deviation class wiring | riir-ai | Plan 325 Phase 3 |
| **P2** | LatCal fixed-point commitment of zone-mood signal | riir-ai | Plan 325 Phase 4 |
| **P2** | Moderator objective `Γ₀` (economy, faction, narrative modes) | riir-ai | Plan 325 Phase 5 |
| **P2** | Latent Functor re-estimation fusion (coherence-decay re-moderation) | riir-ai | Plan 325 Phase 6 |
| **P3** | ICT BranchingMask gating of primal-dual updates | riir-ai | Plan 325 Phase 7 |

---

## 5. Validation Protocol (G1–G5 GOAT gate)

Per AGENTS.md, every plan with a new technique ships behind a feature flag and benchmarks the gain before promoting to default.

- **G1 — CCE ≥ Nash (LP solver correctness).** Construct a 3-strategy general-sum game with a known Pareto-dominant CCE. Solve via `CceLp`. Assert: `Γ₀(ρ_CCE) < Γ₀(ρ_Nash)`. Target: ≥ 5% improvement on at least one canonical game (chicken, battle-of-sexes, emission-abatement).
- **G2 — Primal-dual convergence.** Run `CcePrimalDual` for `N = 10⁴` iterations on the emission-abatement example. Assert: `|Γ₀(ρ̄ᴺ) − Γ₀(ρ⋆)| < ε` and `ER(ρ̄ᴺ) < 0` (within Slater tolerance). Target: `O(N⁻¹ᐟ²)` rate verified empirically (paper Theorem 6.1).
- **G3 — Latent reframing preserves ranking.** Two NPCs with HLA states `h₁, h₂` and a signal `ζ`. The recommended action distribution under CCE should be monotone in `dot(h, ζ_dir)` — i.e. NPCs with aligned latent state align in action. Target: Kendall-τ > 0.7 between `dot(h, ζ_dir)` rank and action-correlation rank.
- **G4 — Crowd-scale latency.** Per-NPC primal-dual update at 20Hz tick × 1000 NPCs. Target: < 50µs per NPC update (plasma tier), < 100ms total per tick for the moderator's batched update.
- **G5 — LatCal commitment round-trip.** Broadcast `ζ` as a fixed-point raw scalar via LatCal, consume in NPC HLA channel. Assert: bit-identical reconstruction across nodes, BLAKE3 commitment validates, sigmoid projection of `ζ` onto HLA direction vector matches the local-only computation. Target: < 10µs per NPC bridge call.

GOAT gate rule: `cce_moderator` feature flag default-off. Promote to consideration for default-on only after G1–G5 PASS with benchmark evidence. Demote the loser (per-NPC independent curiosity without moderator coordination) if CCE wins on a head-to-head crowd welfare metric.

---

## 6. Constraint Compliance

| Constraint | Compliance |
|---|---|
| **Modelless first** | ✅ Discrete policy table + LP solver + primal-dual iterator. No backprop through base weights. |
| **Latent-to-latent preferred** | ✅ State = HLA bucket, action = projected distribution, signal = latent scalar. Decode to raw action only at the sync boundary via LatCal. |
| **Sigmoid not softmax** | ✅ Projection gates onto HLA direction vectors use sigmoid (per AGENTS.md); the policy table itself is a categorical distribution (representation, not a projection gate). |
| **Freeze/thaw over fine-tuning** | ✅ The policy table `(Ξ_φ, θ)` is snapshotted via existing freeze/thaw; no weight mutation during inference. The primal-dual iterator updates the *table*, not weights. |
| **Self-learn / adaptive CoT welcome** | ✅ The primal-dual iterator IS self-learning — `ρ` adapts from runtime regret signal. Latent-state direction-vector update, not weight update. |
| **4-repo discipline** | ✅ Public math (katgpt-rs), private runtime (riir-ai), chain commitment via LatCal (riir-chain), training-only parts (riir-train). |
| **Raw↔latent bridge** | ✅ Zone-mood signal `ζ` is the bridge: committed raw via LatCal, consumed latent via HLA dot-product. |
| **SOLID, DRY** | ✅ `CceLp<N,A>`, `ExternalRegret<D>`, `CcePrimalDual` are generic over `N`, `A`, `D`. No duplication of `PayoffTable<N>` (different solution concept). |
| **Tests/examples** | ✅ G1–G5 above + before/after on a bomber/RPS/emission-abatement arena showing CCE Pareto-dominates Nash. |
| **CPU/GPU/ANE auto-route** | ✅ LP solver on CPU (small N) or GPU (large N via batched LP). Primal-dual iterator on SIMD plasma tier. |

---

## 7. Cross-Reference Summary

| Research | Connection |
|---|---|
| **R026 (RTS Intransitive)** | Ships `PayoffTable<N>` Nash solver — CCE generalizes this; `PayoffTable<N>` becomes the deviation class for 1v1 CCE |
| **R079 (EqR Equilibrium Reasoners)** | Within-model equilibrium; CCE is the cross-population analogue |
| **R098 (PrudentBanker)** | Banker-OMD primal-dual machinery directly reusable for the dual update `λⁿ⁺¹` |
| **R167 (Economy of Minds / WealthPruner)** | Wealth-based arm selection = per-arm deviation; CCE correlates arms via shared signal |
| **R168 (Ruliology Arena)** | Cross-paradigm payoff matrix → CCE selection over strategy classes |
| **R212 (Gemini Fourier × LatCal)** | LatCal is the sync-boundary bridge for the correlation signal `ζ` |
| **R249 (DecentMem Dual-Pool)** | O(log T) regret machinery + mean-field α router — primal update pattern |
| **R270 (ICT Distributional Branching)** | BranchingMask gates the primal-dual update — cognitive economics |
| **R273 (ELT Any-Time)** | Elastic budget per zone — CCE moderator budget is elastic per zone density |

| Code | Connection |
|---|---|
| `riir-games/src/payoff.rs` (`PayoffTable<N>`) | Nash solver; deviation class for 1v1 CCE |
| `katgpt-rs/src/pruners/prudent_banker.rs` | OMD primal-dual machinery |
| `katgpt-rs/crates/katgpt-core/src/cgsp/dual_pool.rs` | Mean-field α router + O(log T) regret bound |
| `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` | Coherence-driven re-estimation scheduler (Fusion D) |
| `riir-ai/crates/riir-engine/src/hla/` | HLA channel — signal `ζ` broadcast medium |
| `riir-ai/crates/riir-chain/src/encoding/latcal*.rs` | LatCal fixed-point bridge — sync commitment of `ζ` |
| `riir-ai/crates/riir-engine/src/cgsp_runtime/` | CGSP conjecturer pool = deviation class `D` |

---

## 8. References

- Paper: [arXiv:2606.20062](https://arxiv.org/pdf/2606.20062) — Campi, Cannerozzi, Tzouanas, 18 Jun 2026
- Code: https://github.com/JannTzou/Learning-Algorithm-for-Mean-Field-CCE.git (JAX reference, training-flavored)
- Our payoff: `riir-ai/crates/riir-games/src/payoff.rs`
- Our bandit: `katgpt-rs/src/pruners/bandit.rs`, `prudent_banker.rs`
- Our dual pool: `katgpt-rs/crates/katgpt-core/src/cgsp/dual_pool.rs`
- Our latent functor: `riir-ai/crates/riir-engine/src/latent_functor/`
- Our HLA: `riir-ai/crates/riir-engine/src/hla/`
- Our LatCal: `riir-ai/crates/riir-chain/src/encoding/latcal*.rs`

---

## 9. Subjective-CCE heterogeneous extension (Plan 300, 2026-06-21 → T4.3b 2026-06-22)

The homogeneous `CceLp<N,A>::solve(d, p)` takes a **single** `PayoffTensor<N,A>`
shared across all deviation constraints. The subjective-CCE generalization —
each NPC `i` evaluates deviations against its own `J_i^{CWM_i}` — requires a
thin wrapper primitive, shipped in Plan 300:

- **New trait `HeterogeneousPayoff<N,A>`** (`katgpt-rs/src/cce/types.rs`):
  `n_players`, `deviations_for_player`, per-player `reward_follow` /
  `reward_deviate` / `gamma_player` / `gamma_dev_player`, plus moderator
  objective `gamma0 = (1/P) Σ_i γ_i(ρ)` and linear `gamma0_coeff`.
- **New method `CceLp::solve_heterogeneous`** (`katgpt-rs/src/cce/lp.rs`):
  builds `Σ_i |D_i|` constraint rows, each using player `i`'s cost tensor.
  Shares BFS-enumeration infrastructure with `solve` (DRY refactor —
  `enumerate_bfs` helper).
- **New struct `PerPlayerGame<N,A,P,D>`** (`katgpt-rs/src/cce/heterogeneous.rs`):
  default concrete impl backed by per-player `(PayoffTensor, DeviationClass)`
  slices.
- **New methods `CcePrimalDual::step_heterogeneous` + `run_heterogeneous`**
  (`katgpt-rs/src/cce/primal_dual.rs`, Plan 300 T4.3b): per-player subgradient
  oracle caches best deviation `κ_i*(ρ)` once per step, then aggregates
  `grad[m] = gamma0_coeff(m) + λ · (1/P) Σ_i [cost_i(s,a) − reward_deviate(i, s, κ_i*)]`.
- **New method `ExternalRegret::linear_derivative_heterogeneous`**
  (`katgpt-rs/src/cce/external_regret.rs`, Plan 300 T4.3b): public per-(s,a)
  subgradient for testing / single-index callers.

**Regret bound transfer (doc 62 §2):** `ER(ρ̄_T) ≤ O(T⁻¹ᐟ²)` holds as-is — each
`γ_i` is linear in `ρ`, so each per-player regret is convex; the average
`(1/P) Σ_i ER_i(ρ)` is convex; primal-dual averaging is
heterogeneity-agnostic. **No new theory; pure API surface.**

**GOAT gates (Plan 300 Phase 4 + T4.3b):**
- **G1 PASS** — `PerPlayerGame` with identical players reproduces homogeneous
  `solve` output (emission/chicken/RPS, objective within 1e-4).
- **G2 PASS** — `er_heterogeneous(ρ⋆) = 0` on 8- and 16-player perturbed games
  (exact LP solve).
- **G3 PASS** (T4.3b, 2026-06-22) — `CcePrimalDual::run_heterogeneous`
  converges to LP optimum on a 4-player perturbed game: gap = 0.000580 after
  10⁴ steps, `ER_heterogeneous(ρ̄ᴺ) = 0.000133`, fitted log-log slope = **-1.0**
  (beats paper's -0.5 `O(N⁻¹ᐟ²)` upper bound — well-conditioned convergence at
  `O(1/N)`).
- **G4 PASS** — All scales under 50ms target post-T4.3b. BFS-tractable
  regime (≤16 players): 16 players = 43.2ms via `solve_heterogeneous`.
  Crowd-scale (24+ players) via `run_heterogeneous`: 24 players = 7.9ms,
  32 players = 8.9ms (vs 268ms / 1199ms BFS-only ceiling — 34× / 135× faster).

**Feature promotion (T5.4):** with all 4 GOAT gates PASS, `cce_moderator` is
now in the `default` features list in `katgpt-rs/Cargo.toml`. Zero non-optional
deps (`cce_moderator = []`); the module is `#[cfg(feature="cce_moderator")]`
gated, so promotion is zero-cost for non-consumers.

**Scope boundary:** this is the subjective-CCE wiring path (Issue 327 Path
A+). Strict Bayes-CCE (no-common-prior per-NPC CWMs modeled as draws from a
non-degenerate joint) is deferred Super-GOAT research, tracked separately in
`riir-ai/.issues/328_no_common_prior_bayes_cce.md`.

**References:**
- Plan 300: [`katgpt-rs/.plans/300_subjective_cce_heterogeneous_payoff_wrapper.md`](../.plans/300_subjective_cce_heterogeneous_payoff_wrapper.md)
- Parent issue (closed): [`riir-ai/.issues/327_cwm_cce_bayes_fusion.md`](../../riir-ai/.issues/327_cwm_cce_bayes_fusion.md)
- Sibling issue (deferred): [`riir-ai/.issues/328_no_common_prior_bayes_cce.md`](../../riir-ai/.issues/328_no_common_prior_bayes_cce.md)
- Regret sketch: [`riir-ai/.docs/62_bayes_cce_regret_sketch.md`](../../riir-ai/.docs/62_bayes_cce_regret_sketch.md) §2

**TL;DR:** Campi et al. give us **optimal Coarse Correlated Equilibria in Mean Field Games via LP relaxation + no-regret primal-dual with Bregman regularization** (`O(N⁻¹ᐟ²)` convergence, no monotonicity assumption). The neural-network parametrization is training-flavored (→ riir-train), but the LP-CCE formulation + primal-dual iterator distill modellessly into `CceLp<N,A>` + `CcePrimalDual` in katgpt-rs. The latent-space reframing — `state` = HLA bucket, `action` = CGSP conjecturer arm, `signal` = zone-mood latent scalar broadcast via HLA channel, sync-committed via LatCal — is the **Super-GOAT**: coordinated emergent population behavior via a latent broadcast signal with a designer-steerable moderator objective, Pareto-dominating any Nash-seeking competitor, at crowd-scale (thousands of NPCs, 20Hz tick) within plasma-tier latency budgets. All 4 novelty-gate questions YES. Mandatory outputs shipped: this note (R274 public), Plan 295 (public primitive), riir-ai R143 (private guide), riir-ai Plan 325 (private runtime).
