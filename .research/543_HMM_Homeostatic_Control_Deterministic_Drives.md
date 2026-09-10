# Research 543: HMM Homeostatic Control — Deterministic Drives and the psafe Boundary

> **Source:** "Homeostasis Revisited and Reformulated Through Hidden Markov Model Control" — Rubén Moreno-Bote, [arXiv:2609.07508](https://arxiv.org/abs/2609.07508), q-bio.NC, 2026-09-07 (12 pp).
> **Date:** 2026-09-10
> **Status:** Active
> **Related Research:** 478 (MOP — the same author's prior paper, already shipped), 240 (CGSP), 298 (Induced CWM kernel), 423 (FORE occupancy)
> **Related Plans:** 573 (mop value-iteration primitive, COMPLETE), 590 (this paper's primitive plan, this session)
> **Cross-ref (riir-ai):** Research 338 (per-NPC MOP runtime guide — the sibling this extends), Plan 538 (mop_runtime, COMPLETE)
> **Classification:** Public

---

## TL;DR

The paper proves that the standard variational/free-energy route to homeostatic control (maximize an ELBO lower bound on P(desired observations)) is **strictly suboptimal** for the actual problem, and that directly maximizing P(y = y_d for all t ≤ T) over an HMM has an **exact solution via multiplicative backward messages** whose optimal policy is **strictly deterministic** (the objective is linear in each π_t → optimum at the simplex boundary). It also reinterprets the Maximum Occupancy Principle homeostatically: define homeostatic states as "any state that does not immediately entail termination", make the safe range a **boundary condition** via a per-(s,a) continuation probability `psafe`, and let the agent maximize path entropy subject to that boundary.

**Distilled for katgpt-rs (modelless, inference-time):** two exact, closed-form, zero-training drive solvers that compose with the shipped `MopSolver` substrate:

1. **`HmmControlSolver` (new mode):** finite-horizon reachability-probability control. Backward messages `β_t(x) = P(y_{t:T} = y_d | x)` are *probabilities* — bounded [0,1] by construction, multiplicative composition, deterministic argmax policy. This is the **setpoint drive** half: "walk to food with maximum survival-until-fed probability".
2. **psafe-MOP (upgrade to shipped MopSolver):** weight the bootstrap term of the MOP fixed point by a per-(s,a) continuation probability — the paper's Eq. 17-18. This is the **range drive** half: "explore entropy-optimal subject to not dying", with survival probability replacing the arbitrary discount knob.

---

## 1. Paper Core Findings

### 1.1 The variational route is a bounded approximation — and provably suboptimal

The homeostatic objective is `π* = argmax_π p_π(y = y_d)` over a horizon T (Eq. 3) — the probability of the desired observation at **every** step (product form through hidden states, hence not additive, hence no Bellman equation in the standard reward-MDP sense). The classical route (control-as-inference → variational inference → negative free energy) maximizes the ELBO `log p(y=yd) − D_KL(q_π ‖ γ)` instead. Three consequences (paper §2.2, Eq. 9-12):

- The variational objective **is** a standard entropy-regularized MDP with reward `α·log p(y_t=yd|x_t,a_t)` — hence its optimal policy is **strictly stochastic** (softmax over actions, Eq. 10).
- The ELBO is a lower bound; the gap `D_KL(q_π ‖ γ) ≥ 0` is generically nonzero because matching the ideal posterior γ would require *controlling states*, which the agent cannot do.
- The control-as-inference posterior exhibits **"wishful thinking"**: conditioning on y = y_d infers state-action trajectories whose realizability ignores transition-probability mass — the paper's rich-agent example: inferring the path to being rich can route through a lottery.

### 1.2 HMM control: the exact solution is multiplicative messages + a deterministic policy

Because the objective (Eq. 13) is **linear in each π_t** and policies live on simplexes, the optimum is attained at a vertex — **the optimal policy is strictly deterministic** (paper's first main theorem). The exact recursion (Eqs. 14-15) is max-product backward message passing:

```
a*_t(x_t) = argmax_a  p(y_t=yd | x_t, a) · Σ_x' p(x'|x_t, a) · β_{t+1}(x')
β_t(x_t)  = p(y_t=yd | x_t, a*_t(x_t)) · Σ_x' p(x'|x_t, a*_t(x_t)) · β_{t+1}(x')
β_t(x_t)  = P(y_{t:T} = y_d | x_t)        ← a probability, bounded [0,1]
```

Load-bearing structural facts:

- **Multiplicative, not additive.** Bellman backups sum rewards; these messages *multiply probabilities*. β is not a value function over an additive semiring — it is a reachability probability (max-product on the (max, ×) semiring — structurally Viterbi).
- **No discount factor.** The product over t already penalizes any single step with low success probability ("one bad step kills the whole product"). Horizon T replaces γ.
- **Risk-sensitive where variational is not.** Paper §3's T=2 analysis: HMM control weighs `p(x*_2|x_1,a_1)` — a tiny-probability path to a perfect state contributes proportionally — while control-as-inference (argmax in the α→∞ limit, Eq. 28) follows the best successor *regardless of transition mass*, and variational inference (Eq. 30 vs 31: sum inside vs outside the log) refuses any branch that can hit `p(y=yd)=0` with −∞. All three coincide only under deterministic dynamics.
- **Survives partial observability.** Constraining π to condition on `z_t = g(x_t)` preserves linearity in π_t, hence determinism of the optimum (§2.3 end).

### 1.3 MOP reinterpreted homeostatically — the psafe boundary (§2.4)

Homeostasis over a *range* (not a setpoint) needs no desired value y_d at all: treat the range boundaries as **terminal observations** (death), give the agent no task inside the safe range except maximizing action+observation path entropy — exactly the Maximum Occupancy Principle (Ramírez-Ruiz et al. 2024, our Research 478) with a termination boundary:

```
π*_t(a|x) ∝ exp( β·H(Y|x,a) + psafe(x,a) · Σ_x' p(x'|x,a)·V*_{t+1}(x') )
V*_t(x)   = log Σ_a exp( β·H(Y|x,a) + psafe(x,a) · Σ_x' p(x'|x,a)·V*_{t+1}(x') )
psafe(x,a) = P(safe observation | x, a)      ← continuation probability
```

- `psafe` acts as the **natural discount**: continuation probability *is* the discount factor. A state-action pair with 90% death risk prices its own future at 0.1×.
- The policy stays **stochastic by construction** (MOP's signature, Research 478 §9.3) but is now shaped by proximity to death — "rich structure … precluding the policy from being purely random" (paper §2.4).
- β < 0 recovers an approximation to Empowerment (paper §2.4) — the same β knob our MopConfig already carries.

### 1.4 Empirical shape (Fig. 2)

Grid world, T=20: the HMM agent walks deterministically to the unique "best location" (p(y=yd)=1) and hovers there; the variational agent (α=1, stochastic) never leaves the lower room. Raising α→∞ makes the variational policy nearly deterministic — paper notes this convergence is coincidental, not general (§3: the α→∞ policies still differ structurally at T≥2).

### 1.5 What the paper does NOT do (scope guards)

- No continuous state/action experiments (formal extension stated as straightforward §2.1).
- No learned kernels — p(x'|x,a) and p(y|x,a) are *given*. (Same binding-input caveat as Research 478 §9.2: the kernel is the quality ceiling.)
- No claim that active inference as practiced is worthless — the critique is scoped to *closed-loop-policy variational control of homeostasis*, their own framing (§2.2 second distinction).

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase analog | Where it ships / would ship |
|---|---|---|
| desired observation `y_d` | drive setpoint (food-in-zone, mate-in-zone, home-zone) | emission table over zone-KG states |
| safe range `(y_min, y_max)` | non-terminal predicate (alive, free, solvent) | `RawAvailabilitySource` (Plan 538) lifted into the solve |
| HMM hidden state `x_t` | zone-KG abstract state | `build_zone_kg_kernel` tabular states |
| action `a_t` | zone transition / civ action enum | kernel's action axis (A ≤ 16) |
| backward message `β_t(x)` | **survival/reachability scalar per state** | new `hmm_control` module output |
| deterministic policy `a*_t(x)` | argmax over the kernel's action row | new policy extractor |
| `psafe(x,a)` continuation prob. | terminal-risk weighting inside the MOP fixed point | new `MopConfig` option |
| maximum occupancy / path entropy | `MopSolver` (shipped, `mop_path_entropy`) | `katgpt_core::mop` |
| variational free energy | (not shipped — the critique says: don't) | — |

### 2.2 The transferable primitives

**P1 — `HmmControlSolver<const N: usize, const A: usize>` (new, feature `hmm_homeostasis`).** Inputs: frozen kernel `p: [[[f32; N]; A]; N]`, emission `e: [[f32; A]; N]` (per-(s,a) probability of the desired observation — the *only* new caller input vs MopSolver), horizon `T: u16`. Outputs: `beta: [[f32; N]; T+1]` (messages), `policy: [[u8; N]; T]` (deterministic argmax per state per step). Backward pass only — `O(T·N·A·N̄)` with the one-hot fast path (`onehot` pattern already in `MopScratch`). Zero-alloc via `HmmScratch`. All outputs bounded [0,1] by construction — the same bounded-scalar discipline Plan 353 proved for HLA scalars (a Lean-able property: products and convex combinations of [0,1] stay in [0,1]).

**P2 — psafe-MOP (upgrade, rides `mop_path_entropy`).** Add `MopConfig.terminal_psafe: Option<[[f32; A]; N]>` (default `None` ⇒ bit-identical behavior — the no-regression gate). When set, the per-action bootstrap exponent in the log-space LSE iteration scales by `psafe[i][k]`: the z-iteration's per-action contribution `(β/α)·H̄ + γ·Σ p·ln z` becomes `(β/α)·H̄ + γ·psafe[i][k]·Σ p·ln z`, plus an observation-entropy reward term `β·H(Y|s,a)` if the caller supplies an observation model (optional axis; ship the continuation weighting first — it is the load-bearing half). Invariant: `psafe ≡ 1` reduces exactly to the shipped solver. `psafe` absorbs γ: survival probability *is* the discount, so a kernel with honest terminal risk can run γ=1 without divergence (every non-ending path eventually multiplies a psafe < 1 — geometric contraction through the risk axis). That is the cleanest formulation of "no arbitrary discount knob".

**P3 — bounded survival scalar → affect bridge.** β messages are per-state probabilities on the think-brain kernel. Bridge (zero-alloc, gated): `fear_reach = sigmoid(−λ·(β(x_npc) − β₀))` — low reachability ⇒ fear — projecting onto the existing `NpcEmotionScalars.fear` axis. This is a **second, independent fear derivation**: the shipped `AffectProjection` derives fear from *low future path entropy* (exploration-value collapse, Research 338 §2.2); fear_reach derives from *death probability* (reachability collapse). Different signals, same scalar slot — compose (max, or weighted blend with hand-pinned `W_` constants per the §3.6 granularity rule — pin the blend, bench it, promote the winner).

### 2.3 Latent-space reframe (§1 step 3)

The whole solve is a think-brain (latent, never-synced) computation on the same zone-KG kernel the mop_runtime already consumes (`build_zone_kg_kernel`, BLAKE3-pinned). What crosses the sync boundary: the deterministic action (raw, seeded-deterministic — existing discipline), and at most one bounded scalar through the fear bridge (one of the 5 synced affect scalars). No latent→raw reconstruction anywhere; the β scalar is itself a *model probability*, not a latent embedding, so syncing it would even be raw-safe — but the two-brain rule says drives stay local; only the affect projection crosses.

### 2.4 Game-context reframe (§1 step 4)

- **Two-mode drive arbitration.** CGSP's `BeliefDrive` pool + priority bandit decides *which drive is active*; the active drive's *execution policy* splits by drive type: setpoint drives (hunger → food, loneliness → mate/social zone) execute **deterministically** via P1's argmax — exact, reproducible, QA-able ("NPC #482 will path to the granary under scenario seed S"); range drives (safety, comfort, curiosity) sample from psafe-MOP's π* — entropy-optimal, death-aware, non-collapsing. The paper supplies the exact math for both halves; the stack already ships the arbitration seam (`MopPriorityBandit<A>` drops into `NpcCgspRuntime`).
- **Fear that means something.** fear_reach is *derived from the world model*: an NPC standing one zone from a lethal boundary with no safe action computes β≈0 and projects fear≈1 — an emergent, principled survival signal replacing the HERO_HP_FLOOR-class POC hacks (Research 478 §1.3 row 1 already called for this; P1 delivers its setpoint twin).
- **Determinism as a product feature.** Raw-deterministic seeded replay is a stack-wide invariant (anti-cheat, quorum). Setpoint drives being *provably* deterministic (not merely seeded-sampled) tightens replay guarantees for drive-driven motion.
- ** Selling point:** *"NPCs solve their homeostasis exactly — setpoint drives take the provably-optimal deterministic action (maximum probability of staying healthy until the horizon), range drives explore entropy-optimal subject to not dying, and their fear is computed from their actual death probability in the world model — all at plasma-tier cost on a frozen, BLAKE3-pinned kernel."*

### 2.5 Consumer-context reframe (the healer — priority #2; canonical-failure-#5 guard)

Does the mechanism manifest on a healer surface? One honest pass: (a) retrieval index — no; (b) fix-trajectory memory/selection — **partial, idea-tier**: `self_evolve`'s trajectory store records binary oracle outcomes (compile+clippy verify pass/fail) per candidate class — a `P(class → success)` the way β is `P(state → success)`. The paper's determinism theorem says: when a *hard binary gate* exists downstream, argmax on exact success probability dominates soft blend selection (the `W_EVO·evolve + W_RATE·reliability` hand-pinned blend in `select_best_candidate` is a variational-style surrogate for exactly that probability). But per-candidate-class outcome counts are sparse and non-stationary — a BetaPosterior (which ships) is the right estimator there, and no measured healer gap is documented that this fills. Verdict: **fusion idea → note only; no plan, no issue** (an issue without a documented gap would be noise). The ablation ("argmax-on-posterior vs soft blend when the oracle is binary") is recorded here for the score-bench backlog.

### 2.6 Fusion (paper × 478 × 338)

Paper (setpoint control + psafe boundary) × Research 478 (MOP fixed point, shipped solver) × Research 338 (per-NPC runtime, kernel source, affect bridges) = **a two-mode homeostat** none of the three has alone: 478/338 ship only the range/exploration half with absorbing-state survival; the paper adds the setpoint half with exact deterministic optimality, and replaces the implicit "entropy-collapse survival" with an explicit death-probability boundary inside the solve. The combination also resolves a 478 §9.3 tension: MOP's stochasticity is *desired* for exploration but *wrong* for setpoint satisfaction — the two-mode split assigns each objective the policy shape its theorem proves optimal.

---

## 3. Verdict

**Super-GOAT (scoped: fusion novelty, not math novelty).** Q1 no prior art *in the stack* (grep: `mop` ships entropy-only; no setpoint-reachability drive exists — §4.2), with the honesty that the reachability recursion itself is classical control math (in-paper citations: Todorov 2009, Damiani et al. 2024; lineage: Bertsekas SSP, probabilistic model checking). Q2 new capability class: exact setpoint-seeking deterministic drives + death-probability-shaped exploration — no incumbent ships the pair. Q3 selling point above. Q4 force multiplier: mop_runtime kernel + RawAvailabilitySource + CGSP bandit seam + NpcEmotionScalars bridge + npc_episodic/KARC kernel sources ≥ 2 pillars.

Per §1.5 mandatory outputs (this session): open primitive plan → `katgpt-rs/.plans/590_hmm_homeostasis_open_primitive.md`; architectural guide → `riir-ai/.research/370_per_npc_homeostatic_drive_control_guide.md`; riir-ai runtime plan opens when the primitive lands (the 478 → 338 → 573 → 538 precedent chain).

**MOAT gate (§1.6):** katgpt-rs — in scope (generic tabular control math, no game semantics in the solver; belief/motivation-adjacent like `sense/` and `mop/`); riir-ai — in scope (pillar-level extension of the mop_runtime pillar). NOT riir-chain (no commitment/quorum angle — the β scalar stays think-brain; do not stretch LatCal here), NOT riir-neuron-db (nothing new to store — kernels are already BLAKE3-pinned snapshots via the existing path).

**Per-stack ledger (katgpt-rs):** stack slot = **motivation/drive control** (sibling of `mop_path_entropy`); feature `hmm_homeostasis` opt-in; promote to default only on GOAT (G1-G4) + a consumer wiring; `mop_path_entropy` keeps its slot, psafe rides it as a config extension — the "demote the loser" question does not arise (different objectives, composed not competing).

---

## 4. Prior-art search (mandatory §4)

**Pinned claim before searching:** *"Exact finite-horizon setpoint-reachability drive control (multiplicative messages, deterministic argmax) plus psafe-terminating MOP exploration, as a two-mode per-NPC homeostatic drive solver over the existing zone-KG kernel substrate — distinguished from the shipped mop_runtime (reward-free entropy-only, absorbing-state survival only, no setpoint mode, no continuation weighting in the solve) by the setpoint mode, the survival-weighted bootstrap, and the reachability-derived fear bridge."*

### 4.1 Published prior art

- Headline + component searches (2026-09-10): "HMM control homeostasis free energy principle deterministic policy variational comparison" → only FEP background (Friston primer-level material; no paper doing this comparison in a game/runtime context). The paper itself is 3 days old — no follow-up literature exists.
- Reachability-probability control is **classical**: stochastic shortest-path reachability (Bertsekas), probabilistic reachability/invariance for discrete stochastic systems (Abate et al., 600+ cites), probabilistic model checking (PRISM) — all compute exactly the β recursion. The paper cites its own lineage honestly (Todorov 2009 linearly-solvable control, Damiani et al. NeurIPS 2024 multiplicative-noise control). **Accordingly: no novelty is claimed for the recursion; the claim is the two-mode fusion in a game-runtime drive stack.** Class-level prior art does NOT kill the mechanism-level fusion (TTPO discipline): nobody ships setpoint-reachability as NPC drives.
- MOP-homeostatic: the paper §2.4 *is* the primary source for the psafe boundary — its own novelty vs its 2024 MOP paper, which had no termination boundary.

### 4.2 Shipped prior art (in-stack grep)

- `katgpt_core::mop` — `MopSolver` (log-space LSE, pins absorbing/terminal to V=0 exactly, one-hot fast path, golden parity) — the entropy half only; no emission input, no psafe, no setpoint. `bench_mop_solver.rs` exercises absorbing/terminal mask rows.
- riir-engine `mop_runtime` (Plan 538, Bench 680/681 GOAT) — kernel source, `MopPriorityBandit`, `AffectProjection` (fear = low entropy), `RawAvailabilitySource` (per-tick terminal short-circuit — runtime masking, NOT solve-level continuation weighting).
- riir-engine `cgsp_runtime` — `BeliefDrive` pool + `PriorityTableBandit` (which-drive arbitration, no execution policy per drive).
- civ `motivation.rs` — 14-dim stat vec with a "homeostatic calm ≈ comfort" proxy comment; no reachability.
- Canvas `can_reach`/`reachability_horizon` — boolean DAG reachability, different domain. DecentMem dual-pool reachability — memory-pool escape, unrelated.
- No hits for setpoint/survival-probability/reachability-value drive control anywhere in `*.rs`.

### 4.3 Conclusion

In-stack: clear (nothing computes a setpoint reachability or weights a solve by survival probability). Published: recursion classical + cited; fusion unclaimed. Q1 holds under the scoped claim.

---

## 5. Defend-wrong PoC protocol (§3.6 — quality claims are unproven until measured)

Paper §3 supplies exact analytic fixtures; the PoC needs **no new theory**, only fidelity checks + head-to-heads:

1. **G1 analytic parity (P1).** Hand-computed T=2 example from paper §3 (risky a₁ vs safe a′₁, rare path to x*₂): HMM control must pick the safe action below the transition-mass threshold where control-as-inference picks the risky one; Eq. 25 vs 28 vs 30/31 divergence reproduced exactly in f32.
2. **G2 paper-Fig-2 reproduction (P1).** 4-room gridworld variant (best-location emission 1.0, elsewhere 0.95, stay-noise p_s): HMM agent reaches and holds the best location; a softmax/entropy-regularized comparator (α=1) stays low — quantify the p(y=yd) gap over 10⁴ episodes as the paper does.
3. **G3 psafe survival gain (P2).** `ring_world_noisy` + a terminal zone (psafe=0 off the ring): psafe-MOP survival/occupancy vs plain MOP (absorbing-pin only) vs psafe≡1 regression identity. Gate: survival strictly improves without occupancy collapse (H(π*) floor).
4. **G4 backward-compat (P2).** `psafe ≡ 1` ⟹ bit-identical to shipped MopSolver on all existing arenas (golden parity reuse).

Competitors: paper mechanism (distilled) vs plain MOP (frozen baseline) vs entropy-regularized variational-shaped comparator. Lives in `riir-poc`; `CARGO_TARGET_DIR=/tmp`.

---

## 6. Latent vs raw boundary

| Signal | Space | Crosses sync? |
|---|---|---|
| zone-KG kernel `p`, emission `e`, `psafe` | think-brain (BLAKE3-pinned snapshot) | no |
| β messages, deterministic policy | think-brain | no |
| chosen action | raw (seeded-deterministic; setpoint argmax is deterministic by theorem) | yes — as movement/interaction intent |
| `fear_reach` scalar | sigmoid projection of β | yes — inside the 5 affect scalars |
| drive mode (setpoint vs range) | think-brain arbitration state | no |

No latent→raw reconstruction; no feature flag on raw sync correctness (the fear bridge is the only gated surface).

## 7. What stays public vs private

- **Public (katgpt-core):** `hmm_control` solver + psafe config option + arenas/benches — generic tabular math, no game semantics (the `mop/` precedent).
- **Private (riir-ai):** drive-mode wiring, emission/psafe derivation from game facts, fear blend, all selling-point composition (guide 370).
- Boundary check: clean layering, identical to 478 §10's verdict.

## 8. Honest limitations + risks

- **Tabular only.** Same N²A memory bound as MopSolver — zone-KG abstraction required (N ≤ few hundred). Continuous drives are out of scope (paper itself stays discrete).
- **Kernel quality is the ceiling.** Garbage p(x'|x,a) ⇒ confidently-optimal garbage. Mitigation: kernels are BLAKE3-pinned snapshots of *measured* transitions (Plan 538 chain), not guesses; still the binding input (478 §9.2 restated).
- **Determinism cuts both ways.** Provably-deterministic setpoint behavior is exploitable/predictable by players. Mitigation is architectural: determinism at the drive level sits *under* stochastic arbitration (which drive is active) and psafe-MOP range behavior — the composed behavior is not a fixed beeline.
- **Horizon choice T** is a new knob the γ-discounted MOP didn't have. Mitigation: T as config with the same discipline as α/β/γ (`MopConfig`-style validation); long-horizon messages numerically underflow to 0 — log-space accumulation if it bites (not expected at game horizons).
- **Not UQ-bearing:** β is an exact model-computed probability, not a calibrated uncertainty estimate over data — the conformal floor rule (Research 322) does not apply. Stated to keep the gate surface honest.
- **Variational critique scope.** The paper attacks closed-loop-policy variational control of homeostasis (its own framing), not active inference broadly. Our takeaway is constructive: *for setpoint drives, solve exactly; for range drives, keep MOP* — we ship no variational controller to demote.
- **Quality unproven until the PoC** (§5) — architectural + prior-art axes are the verified ones; G2/G3 gains are claims.

## 9. Cross-references

- **Closest cousins:** Research 478 / Plan 573 / riir-ai Research 338 / riir-ai Plan 538 (the MOP pillar this extends — the paper's own §2.4 formalizes the boundary condition that plan lacked); Research 298 / Plan 296 (kernel source); Research 240 (CGSP — the arbitration seam); Research 423 (FORE — the descriptive dual: realized occupancy vs this paper's prescriptive reachability).
- **Substrate-first check:** consumed, not duplicated — both primitives extend `katgpt_core::mop`'s substrate (kernel tables, one-hot fast path, scratch pattern, golden-parity discipline) and riir-engine's `mop_runtime` kernel pipeline. No parallel kernel builder, no parallel affect path (P3 projects onto the existing `NpcEmotionScalars`).
- **Boundary check:** open primitive katgpt-core; per-NPC wiring + drive IP riir-engine; game facts (emission tables from zone content) consumer-side. Clean.

## 10. Paper metadata

- arXiv:2609.07508v1, 2026-09-07, q-bio.NC + physics.bio-ph. Author: Rubén Moreno-Bote (UPF Barcelona) — same author as the MOP paper (refs [6,7] = Ramírez-Ruiz et al. 2024) and the MOP-vs-Empowerment-vs-FEP comparison ([12]).
- Key equations distilled: Eq. 3 (objective), Eq. 10-11 (variational policy, stochastic), Eq. 14-15 (HMM control, deterministic), Eq. 16-18 (MOP + psafe), §3 T=2 divergence cases (Eq. 25/28/30/31).
- LLM disclosure in paper: proofreading + MATLAB script generation only.
