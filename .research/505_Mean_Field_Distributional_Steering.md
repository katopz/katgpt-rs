# Research 505: Mean-Field Distributional Steering — Exact-Target Population Control via Feynman-Kac Reweighting

> **Source:** [A Mean-Field Framework for Inference-Time Distributional Control of Diffusion Models](https://arxiv.org/abs/2608.08770) — Samuel Howard, Nikolas Nüsken (Oxford Stats), arXiv:2608.08770, 9 Aug 2026, SPIGM workshop @ ICML 2026
> **Date:** 2026-08-24
> **Status:** Active
> **Related Research:** 290 (Latent Field Steering — the pointwise/heuristic special case this generalizes), 248 (BoM single-pass diverse sampling — batch-level heuristic this grounds), 468 (Beckmann MFG — the other mean-field flavor: transport vs tilt), 236 (QGF — per-action reward-gradient steering), 499 (Jagged Judges / signed_coupling — opinion-distribution observability), 143 (Latent CCE Moderator — crowd-level heuristic control), 517 (CDM amortized twist — the opaque-reward + learned-head delta this note's closed-form Table-2 rewards leave open)
> **Related Plans:** 577 (this paper's plan — the open primitive), 581 (twist_smc — opaque-reward steering + modelless amortization, consumes this module)
> **Cross-ref (riir-ai):** Research 344 (Crowd Distribution Targeting Game Runtime Guide)
> **Classification:** Public

---

## TL;DR

The paper converts **population-level steering** from a heuristic ("push every particle with a batch-gradient; hope the distribution lands somewhere reasonable") into an **exactly-characterized target with a convergent training-free sampler**: steer toward the implicit tilted measure `μ*(dx) ∝ e^{Ψ(x,μ*)} p₁(x) dx`, where `Ψ = δR/δμ` is the **first variation** of a measure-defined reward `R(μ)` (MMD-to-target, moment, entropy — all closed-form in their Table 2), via weighted interacting particles (McKean-Vlasov dynamics + **Feynman-Kac reweighting** + a cheap damped-Picard solve for the mean-field term `Ψ̇`, 0.03–0.24% of runtime). Pointwise reward steering (our Latent Field Steering's math cousin) drops out as the degenerate case `R(μ) = ∫r dμ → Ψ = r`.

**Distilled for katgpt-rs (modelless, inference-time):** four composable pieces, none of which ships anywhere in the stack today — (1) a **closed-form first/second-variation table** for standard measure-rewards, (2) **FK log-weight accumulation** `A_i += [b_i·∇Ψ_i + Ψ̇_i]δt` with `w_i ∝ e^{A_i}` (log-sum-exp stable), (3) a **damped Picard solver** for the implicit weight equation (weights define the measure, the measure defines the weights — the `feedback_payoff_scores` POC pattern, productionized), (4) optional **SMC residual resampling** (sampling consumers only — NOT for persistent agents; the convergence theorem tracks the *weighted* measure, which is exactly the salience-weighted crowd). All substrate to compose already ships: interacting populations (`signed_coupling`, `mean_field`, BoM hypotheses), fixed-point solvers (MOP/FastICA/Newton-Schulz house pattern), MMD as a scalar metric (`rbf_mmd_sq`), vector steering (`latent_steering.rs`).

---

## 1. Paper Core Findings

### 1.1 The problem quadrant the paper fills

|  | Heuristic gradient | Exact target μ* |
|---|---|---|
| **Pointwise reward r(x)** | classifier guidance, DPS, universal guidance | FK reweighting (Singhal 2501.06848, Skreta 2503.02819, flow maps Sabour 2511.22688) — mature |
| **Distributional reward R(μ)** | Particle Guidance (Corso), Shielded Diffusion, MMD guidance (Sani 2601.08379), experiment-guided AF3 (Maddipatla), DiffBED — "effective but largely heuristic; unclear what distribution these target" | **this paper** (+ one concurrent diversity-only special case: Azangulov variance-tilted 2606.22239, Doob h-transform, same workshop) |

Distributional rewards are the natural class for: **calibration to population-level observations** (their protein use-case: DEER/crystallography observe *ensemble proportions*, not individual structures), **diversity** (entropy/kernel repulsion rewards), **moment/balance constraints** ("crowd mean valence should be X").

### 1.2 The target (Prop 3.1 — the part that transfers furthest)

`μ* = argmax_μ { R(μ) − KL(μ‖p₁) }` has the first-order characterization:

```
μ*(dx) = (1/Z) · e^{Ψ(x, μ*)} · p₁(x) dx,    Ψ(x,μ) := δR/δμ (x, μ)
```

**The tilt is implicit/self-referential**: the potential depends on the measure it tilts. This is a mean-field fixed point — the same self-consistency structure as our Curie-Weiss `solve_q_fp`, MOP Eq. 7, and cp-Hopfield `mean_field` fields, but over a *measure* instead of a scalar/vector.

Closed-form first variations (their Table 2 — pure calculus, modelless):
- Linear `R = ∫f dμ` → `Ψ = f(x)` (pointwise steering recovered)
- Moment `R = F(∫φ dμ)` → `Ψ = F'(∫φ dμ)·φ(x)`
- MMD² `R = −∬k(x,y)(μ−ν)(dx)(dy)` → `Ψ(x,μ) = 2∫k(x,y)(μ−ν)(dy)` — a kernel-weighted contrast against the target measure, computable in closed form over a weighted particle population
- Entropy → `Ψ = log ρ_μ(x) + 1` (needs a density estimate — the one hard row)

Key structural fact: for a **divergence reward** `D(μ,ν)`, λ→∞ drives `μ → ν` *as an entire distribution* — whereas pointwise rewards concentrate on a single argmax. Distributional control is not reducible to pointwise (matching ν pointwise would need `r = log(ν/p₁)`, requiring the base density).

### 1.3 The procedure (Alg 1 + Thm 3.2/3.4)

Per step, per particle i of N:
1. **Position/steering**: standard sampler drift + `ε_t ∇_x Ψ_t(X_i, μ̂)` where `μ̂ = Σ_j w_j δ_{X_j}` — this is what the heuristic batch-steering methods already do.
2. **FK log-weight**: `A_i += g_i δt`, `g_i = b_i·∇Ψ_i + Ψ̇_i` — **the correction nobody ships**.
3. **Ψ̇ solve** (the mean-field term): implicit equation (Prop 3.3) involving the mean-centered second variation `Φ̃`. Two solvers: an N×N linear system (Alg 3) or **damped Picard over finite differences of reward evaluations** (Alg 4): candidate next-weights → next-measure → `Ψ̇ ≈ [Ψ(x, μ̃_{t+δt}) − Ψ(x, μ_t)]/δt` → updated weights; K_FP = 3–5 suffices (K=1 slightly biased), kernel evals reused across iterations. Measured cost: 0.036–0.244% of per-step runtime (network evals dominate).
4. **Resampling**: optional SMC residual resampling against weight degeneracy.
5. **Convergence** (Thm 3.4 + Cor): `μ̂_t^N → μ_t*` as N→∞ for bounded-Lipschitz test functions, with resampling (bounded K) preserved.

### 1.4 Empirics (the falsifiable harness)

- 1D bimodal GMM (weights 1:3) + MMD reward toward reweighted GMM (3:1), λ* ∈ {5,10,15}, three noise schedules: **mean-field steering's optimality gap `J(μ)−J(μ*)` is minimized exactly at λ = λ***; gradient-only steering's optimum lands elsewhere (its sampled law depends on steering strength, uncharacterized). K_FP ∈ {3,5} ≈ identical; K_FP=1 slightly biased.
- Closed-form Gaussian target (mean reward, d ∈ {5,10,20}): Bures-Wasserstein distance to ground truth minimized at λ = λ*; improves monotonically with N.
- Proteins (Boltz-2 HIV-1 protease 4608-dim → DEER-observed 25/75 state proportions; AdK 2-D angle distributions from MD; Protenix 4OLE X-ray electron density — genuine population-level observations): objective minimized at correct λ; stronger + better-calibrated tilts than gradient-only. ~50% runtime overhead total (one extra no-grad forward pass at X_{t+δt}).
- **Honest limitation (their own §5)**: resampling duplicates particles — "somewhat ill-suited to the diversity setting" (copies of near-identical samples); differentiable reward required; gradient-only may suffice when exactness isn't needed.

### 1.5 What this is NOT

Not training (explicitly contrasts the fine-tuning paradigm — Santi Flow Density Control 2511.22640 and Smith Calibrating 2510.10020 are the *trained* versions of this objective, riir-train cross-ref only). Not mean-field *games* (no strategic agents — one controller shaping a population measure; cf. our Beckmann 468 which is optimal *transport* under congestion).

---

## 2. Distillation

### 2.1 The transferable primitive (four pieces, `distributional_steering`)

1. **`MeasureReward` table** — closed-form `first_variation_into(x, pop, out)` (+ `second_variation` for the Picard/linear solvers) for `Linear(f)` / `Moment(F, φ)` / `Mmd(kernel, target_particles)`. Generic math, zero game semantics.
2. **`fk_weights`** — log-weight accumulator `A_i` (fixed `[f32; N]` scratch), update `A_i += (b_i·∇Ψ_i + Ψ̇_i)·δt`, weights via log-sum-exp normalize. The **weighted empirical measure** `μ̂ = Σ w_i δ_{X_i}` is the object that converges.
3. **`psi_dot_picard`** — damped fixed-point (Alg 4 shape) over finite differences of reward evaluations; kernel matrices computed once per step, reused across iterations; K_FP ∈ {3,5}; damping for strong tilts. The `feedback_payoff_scores` (riir-poc) implicit-weight pattern, productionized.
4. **`residual_resample`** — optional, sampling consumers only. For persistent-agent consumers (NPC crowds): **weights-only mode** — no resampling; the theorem's tracked object is the weighted measure anyway.

Plus the **gradient steering term** `∇_x Ψ` which composes with the existing `LatentField` machinery (for MMD rewards it is exactly a kernel-weighted attraction-to-target/repulsion field — the principled version of Particle Guidance's heuristic kernel).

### 2.2 Where the pieces already live (grep-verified, 2026-08-24)

| Piece | Existing location | Reuse |
|---|---|---|
| Interacting population w/ order params | `katgpt-core/src/signed_coupling.rs` (opt-in), `mean_field/mod.rs` (**default-on** κ/κ_a/Q + RegimeClassifier) | The particle set + the *observability* of the crowd measure |
| K-hypothesis particle batch | `BoMSampler` (`katgpt-micro-belief/src/bom.rs`, feature `bom_sampling`) — **unweighted**, argmax `select_best` | Gets per-hypothesis FK weights — the theoretical foundation its diversity heuristic lacked |
| Vector steering + spatial support | `katgpt-core/src/latent_steering.rs` (`LatentField`, `apply_field_to_crowd`) — **no target, no feedback** (open-loop) | Becomes the ∇Ψ carrier; gains a characterized target + closed loop |
| MMD metric | `katgpt-core/src/mag/transfer.rs` (`rbf_mmd_sq` — scalar only, never differentiated) | Becomes a differentiable reward (row of the table) |
| Picard/fixed-point house pattern | `MopSolver` Eq.7, `FastICA`, `newton_schulz`, cp-Hopfield self-consistent field; **`riir-poc/feedback_payoff_scores`** (implicit weight FP, 5 iters) | The Ψ̇ solver is the productionized version of the POC pattern |
| exp-weight normalization precedents | `elasticity_gated_update` (Σ-normalized Gaussian kernel weights), NNLS `w=exp(β)` mass matching (katgpt-attn-match), MGPO `exp(−γ|2p−1|)` | Numerical conventions to follow (Σ-normalize; log-sum-exp stability) |
| Reward-gradient steering at inference | QGF `QGradientOracle` (per-action), flow steering | Pointwise quadrant cousins — the degenerate case |

**Greenfield (nothing ships):** FK log-weights with population normalization, SMC residual resampling, first/second functional variations, any specified target tilted measure `μ*`, any closed-loop crowd-distribution control (grep on `tick_swarm_emotions*` family: signal *propagation* only — threat geometry → per-NPC fear → flee; CLR-weighted aggregation and threat centroids are emergent summaries, never reference targets).

### 2.3 Closest cousins (3)

1. **Research 290 / `latent_steering.rs`** — the pointwise special case sans theory: direction + α + support, open-loop, no target measure, no correction weights. This paper is its distributional generalization *with* a convergence theorem.
2. **Research 248 / `BoMSampler`** — batch-level diversity, heuristic; the paper's Table 1 row "distributional reward + heuristic gradient" verbatim. FK weights give BoM hypotheses a principled weighting (caveat: resampling-based exactness is ill-suited to diversity — weights-only mode is the honest fit).
3. **`feedback_payoff_scores` (riir-poc) + Beckmann MFG (R468)** — the implicit-weight fixed point (Picard shape, POC-tier) and the other mean-field control flavor (transport under congestion vs KL-tilt of a distribution). Fusion: Beckmann moves *mass*, mean-field steering reshapes *proportions*.

### 2.4 Fusion

**F1 (PRIMARY — riir-ai, Guide 344): Crowd affect-distribution targeting.** Particles = per-NPC HLA states; target ν = a designer dial OR **calibrated from live player telemetry** (the paper's population-observation analog: DEER observes ensemble proportions ↔ player analytics observe population affect proportions); weights = per-NPC salience ("who carries the distribution" — feeds per-tick salience emission, R281). Steers the *effective* (salience-weighted) crowd distribution to an exact dial at 20 Hz. Today: impossible — fields push open-loop, crowds emerge.

**F2 (signed_coupling × social_pressure): persuasion to an opinion DISTRIBUTION, not a consensus.** `social_pressure` (mmorpg 745) pushes every agent toward one stance; the distributional reward targets the population *histogram* (30/50/20). Realistic societies are not consensus — this is a new game-design capability with the `signed_coupling` order params (κ, susceptibility) as the read-out.

**F3 (BoM × FK weights): principled hypothesis weighting.** K single-pass hypotheses get FK weights against an MMD/entropy reward instead of argmax selection — closes the "which hypothesis is best" gap with a characterized target rather than a scorer heuristic.

**F4 (zone_affective_manifold × moment rewards): the manifold axes become φ.** The PCA axes of the crowd affect manifold are natural `MomentReward` feature functions — "crowd projected on axis-2 mean should be X" is a one-line reward.

### 2.5 Path 0 decomposition (modelless validity — all rows extractable)

| Component | Ships? | Extractable without GD? |
|---|---|---|
| Tilted target `μ* ∝ e^Ψ p` | ✗ | ✅ closed form (Prop 3.1) |
| First variation Ψ (Table 2) | ✗ (MMD as scalar metric only) | ✅ closed-form calculus |
| Gradient steering ∇Ψ | partial (`LatentField` const-dir; QGF per-action) | ✅ autodiff/closed form of Ψ |
| FK weights `w ∝ e^A` | ✗ | ✅ scalar accumulation |
| Ψ̇ mean-field term | ✗ | ✅ Picard finite-diff (Alg 4) — reward evals only |
| Second variation Φ̃ | ✗ | ✅ closed form for kernel rewards |
| Residual resampling | ✗ (PPoT = argmax, unweighted) | ✅ discrete op |
| Convergence guarantee | ✗ | ✅ theorem (cite; no implementation needed) |

**MODELLESS-VALIDABLE — no riir-train deferral.** (Cross-ref only: amortized/trained steering networks = the fine-tuning quadrant — Santi FDC 2511.22640, Smith 2510.10020 — explicitly *not* this paradigm.)

---

## 3. Verdict

### Tier: **Super-GOAT**

| Question | Answer | Evidence |
|---|---|---|
| Q1 No prior art (codebase)? | **YES** | Vocabulary-translated grep across all 6 repos (2026-08-24): zero FK weights, zero SMC/particle filters, zero variation calculus, zero target measures; crowd systems are propagation-only; `latent_steering` is open-loop. `feynman` hits = Hellmann–Feynman (different object). |
| Q1 No prior art (published)? | **YES** | Web sweep: the general training-free exact-target quadrant contains this paper + one concurrent diversity-only special case (Azangulov 2606.22239). **Zero ports to agent/swarm/crowd control — the transfer surface is unclaimed.** Window real but narrowing (3 principled-ish diversity papers May–Jun 2026). |
| Q2 New behavior class? | **YES** | Closed-loop control of a *population measure* to a *specified target* — vs today's open-loop direction pushes (290) and emergent crowd summaries (CLR, threat centroids). Includes capability impossible pointwise: drive crowd to an entire target *distribution* (λ→∞ → μ→ν, not argmax). |
| Q3 Product selling point? | **YES** | "Designers dial a target crowd affect distribution (calibrated from live player telemetry); the crowd's salience-weighted distribution converges to exactly the dial — and the per-NPC weights say who carries it." Demoable, differentiated, no competitor ships anything in this quadrant. |
| Q4 Force multiplier? | **YES** | Connects ≥6: HLA affect states, `latent_steering` (290), `mean_field` order params (default-on), `signed_coupling` (499), zone affective manifold, CLR collective, BoM (248), per-tick salience (281), MOP/Picard solver pattern, MMD metric. |

**Selling point:** population-calibrated living world — crowd distributions as exact designer dials with per-NPC salience weights; plus the first *theoretical foundation* for the batch-level steering heuristics we already ship (BoM diversity, latent fields).

### MOAT gate

- **katgpt-rs**: fundamental primitive via fusion — the variation table + FK weights + Picard solver are generic sampling-layer math (no game semantics). Transformer/sampling stack slot: inference-time steering (sibling of QGF, `latent_steering`). ✅
- **riir-ai**: pillar-level (crowd cognition / living world) — Guide 344. ✅
- Not chain/shard/training material (the weights are transient runtime state; no commitment semantics needed at the open layer — targets may be BLAKE3-committed on the private side via the existing frozen-artifact pattern).

### Honest caveats (defend-wrong posture)

1. **Quality claims are UNPROVEN until Plan 577's G1.** Architectural composition is grep-proven; "our implementation targets μ* as accurately as the paper's" is a quality claim requiring the PoC — G1 *is* the PoC (the paper's own 1D MMD harness: optimality gap minimized at λ=λ*, vs gradient-only + no-steer arms, on our Rust implementation). Per §3.6: architectural + latency axes confirmed by grep/bench; quality axis gated, not claimed.
2. **Resampling is wrong for persistent NPCs** — the paper's own diversity caveat generalized: duplicating a persistent NPC is meaningless. Weights-only mode is the correct consumer: the theorem tracks the *weighted* measure `μ̂ = Σw_i δ_{X_i}` — for crowds, that IS the salience-weighted effective distribution. Frame this as the design insight, not a workaround.
3. **Discrete-time port**: the theory is continuous-time Itô; Algorithm 1 is already the Euler-discretized form we'd ship. The `b_i` term = the consumer's own per-tick drift (HLA update / sampler step). No score needed for the Picard path (finite differences of reward evals).
4. **exp-normalize is not a softmax violation**: `w_i ∝ e^{A_i}` is the *defining math of exponential tilting* (`μ* ∝ e^Ψ p`) — an importance weight, not a semantic projection. The sigmoid-not-softmax house rule continues to govern gates/kernels (`LatentField` kernel falloff, CLR reliability, salience gates). State this in the plan's constraints.
5. **UQ floor rule**: this primitive is a control mechanism, not UQ-bearing (no intervals/coverage claims). If a future BoM+weights gate claims calibrated uncertainty, the "Report the Floor" rule attaches at that gate (BoMSampler is already grandfathered-bound).
6. **Support caveat (paper App. B)**: reweighting alone redistributes mass only over populated support. Crowd affect state spaces are richly populated (that's what diversity is for); for target modes outside support, the ∇Ψ steering term (which we keep) moves particles there.

### Routing

- **katgpt-rs/.plans/577** — open primitive `distributional_steering` (feature-gated, GOAT-gated; promotes only on G1 targeting PASS).
- **riir-ai/.research/344** — private game-runtime guide (this Super-GOAT's selling-point doc).
- **riir-ai/.plans/** — deferred until the primitive passes G1–G2 (the 290→153→309 precedent).

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ No training anywhere in the loop; reward variations are closed-form calculus; Picard is reward evals + arithmetic. |
| Latent-to-latent preferred | ✅ Operates on latent states (HLA/hypotheses); never decodes. |
| Sigmoid not softmax | ✅ (with the stated nuance) exp-tilt weights are the target's defining math — importance weights, not semantic gates; sigmoid remains for kernels/support/reliability. Log-sum-exp stable normalization. |
| Freeze/thaw over fine-tuning | ✅ Target measures ν are frozen artifacts (BLAKE3-committable via the existing envelope pattern, private side); weights are transient runtime state. |
| Raw scalars at sync boundary | ✅ Crowd steering stays latent; only the 5 affect scalars cross sync, as today; weights are local salience, not synced. |
| Zero-alloc hot path | ✅ Fixed `[f32; N]` weight/scratch buffers; kernel matrices into caller-provided scratch; Picard reuses evaluations. |
| 7-repo discipline | ✅ Open math → katgpt-rs; crowd semantics → riir-ai guide/plan. |

---

## 5. Open questions / risks

1. **Does the crowd G8 gate actually converge at 20 Hz?** The theorem is N→∞ asymptotic; finite-N bias at N=1000 crowd, 50-step arcs — measure in G8 (Sinkhorn divergence to target per tick window). Mitigation: K_FP≥3 + damping; multi-batch variant (their Alg 2) if per-tick batches are small.
2. **Ψ̇ stability under strong tilts** — paper needed damping α=0.5 for large λ. Ship damping as a config knob; default conservative.
3. **Entropy reward needs density estimation** — defer that row of the table (MMD/moment/linear ship first); entropy-as-reward for crowd diversity can be approximated via kernel MMD to a uniform reference on the manifold instead.
4. **Telemetry-calibrated targets (F1's punchline)** need a pipeline from player analytics to ν — riir-ai plan scope, post-G1.
5. **Name collision hazard**: "mean field" is overloaded in-repo (`mean_field` order params ≠ measure-tilt control; Hellmann–Feynman ≠ Feynman-Kac). The module ships as `distributional_steering` to keep greps clean.

---

> **PASS-Redirects (synthesis):** Blondel et al. [arXiv:2611.16736 "On the Difficulty of Sequence-Level Supervised Learning for Autoregressive Language Models"] (2026-11-12) — theory annotation for direction-based control: the stop-grad moment-matching direction (match the model's self-rollout distribution to a Dirac/moments target) is PROVEN a valid descent direction that is NOT the gradient of any loss (Jacobian non-self-adjoint), with an exact-per-step-beats-trajectory-MC variance ordering that grows with horizon — this note's steering shape (direction-without-loss; cf. 347 LATENTSEEK), and a proof-grade reason the house prefers exact/frozen reads over trajectory-MC gates; function-space only (the Gram operator deforms parameter-space inner products), reinforcing report-only over latch wiring.
