# Research 674: Compositional Reasoning under RL Post-Training — the Decomposed-to-Composed Asymmetry and the Coverage-Factor Bound

> **Source:** "Compositional Reasoning in Language Models under Reinforcement Learning Post-Training" — Yu He, Yingxi Li, Yifei Wang, Ellen Vitercik (Stanford + Amazon AGI Labs), [arXiv:2609.19465](https://arxiv.org/pdf/2609.19465), 16 Sep 2026.
> **Date:** 2026-11-15
> **Status:** Active — **Gain** (two tracks, no cascade: training track filed as riir-train Issue 693; modelless track filed as katgpt-rs Issue 1222)
> **Related Research:** 264 (Compositional Open-Ended Intelligence — the dependency-graph cousin), 361 (NEO/L2T — the FTCT training-data-design precedent), 444 (IMIR), 325 (latent-reasoning survey)
> **Related Plans:** none (the actionables are issues by design — poc/recipe class)
> **Cross-ref:** riir-train [Issue 693](../riir-train/.issues/693_composed_training_data_asymmetry_cell.md) (training half: the composed-data-design law + the 4090 cells) · katgpt-rs `Issue 1222` (modelless half: the `compounding` bound substrate POC) · riir-train Issue 679 / Plan 492 (the live DSpine lane whose stop-branch record this paper's theory explains)
> **Classification:** Public

---

## TL;DR

The paper formalizes compositional reasoning as execution over **dependency DAGs of reusable skills** (three levels: single-skill chain → horizon generalization; multi-skill chain → + skill switching; branch-merge graph → + non-local dependencies) and shows a consistent **decomposed-to-composed asymmetry**: GRPO RL on atomic skills does not transfer to compound tasks, while RL on compound tasks transfers back to atomic. The theory is the extractable part: compound failure ≤ **C_T(q)·ε_atom** where C_T is a coverage factor between atomic-training context distribution q and rollout context distributions μ_t; **C_T ≥ T is unavoidable** (no horizon-free bound exists, Prop B.2); and Level-III multi-invocation contexts lie **outside the support of any atomic training distribution** — a structural gap where even perfect atomic accuracy yields no compound guarantee. For this stack the paper lands twice: (1) as a **training-data-design law** (composed > atomic supervision) with a live consumer — riir-train plan 492/issue 679's DSpine lane already *measured* the coverage-shift signature (front-loaded acceptance decay, per-token objectives vs a composed gate); (2) as a **modelless evaluation primitive** — the coverage-factor bound is pure histogram arithmetic over already-persisted outcome counters, computable for any composed surface we run (fix campaigns, ESC sessions, quest chains).

**Distilled for katgpt-rs (modelless, inference-time):** a `compounding` substrate (feature-gated `katgpt-core::compounding`, landed under an existing feature-count family or with a reasoned grant row — the ratchet is live at 290/300): `coverage_factor(mu, q) -> CoverageFactor { Finite | Vacuous | Undecidable }`, `compound_failure_bound(eps_atom, C_T)`, `direct_bound(mu, e)`, `first_step_bound(eps_comp, C_1)`, `horizon_floor(T)`, `error_budget(S, C_T)`, `support_gap(mu, q)`. The `Vacuous` variant is the Level-III structural-gap theorem **as a type** — a rollout context with q-mass 0 refuses loud with the offending keys, never folds into 0.0. Consumers (not this repo): refine campaign bounds, rethink coverage-drift monitors, quest-DAG reach profiles. Filed as Issue 1222 with a fully-specified G1 fixture + falsification arm (POC tier — no consumer is blocked today; the strongest pull is instrument-side).

---

## 1. Paper core

### 1.1 The dependency-graph framework

Task τ = DAG G_τ = (V_τ, E_τ); node = one skill invocation (skill type κ(v) + local info ℓ_v), edge = output-of-u feeds input-of-v; unique sink emits the answer. A step-by-step solution is a **topological ordering** of G_τ (branch-merge graphs admit several valid orderings — Figure 2). Three levels, each isolating one bottleneck:

| Level | Graph shape | Isolated challenge |
|---|---|---|
| I — single-skill chain | path, one skill repeated | horizon generalization (staying correct over many dependent steps) |
| II — multi-skill chain | path, mixed skills | + skill switching (interfaces between heterogeneous skills, intermediate-state propagation) |
| III — branch-merge graph | general DAG | + non-local dependencies (merge nodes consume outputs of earlier branches; multi-invocation choice) |

**Decision context** (the load-bearing abstraction): at rollout state s, R(s) = ready nodes; Q(s) = the set of (skill, input) pairs ready at s; correctness is membership in I*(s) = {(κ,x) → f_κ(x) : (κ,x) ∈ Q(s)} — identified by skill+input, never node name. **Context invariance**: stepwise error e(c) is a function of the decision context only. An atomic task has a single-node DAG → its decision context is a **singleton**; a branch-merge rollout can present **non-singleton** contexts (several ready skill-input pairs at once).

### 1.2 The empirical asymmetry

GRPO (binary verifiable reward, verl + vLLM) on Qwen3-4B-Instruct (+ OLMo-3-7B / Llama-3.1-8B on Level II), data-structure tasks (Array, BST, Bloom Filter, Hashmap, Heap; train lengths 5–10, eval to 30):

- **RL-Atomic** (single insert/delete ops): improves atomic accuracy, transfers **poorly** to compound chains. At Level III (segment-tree construction — SplitInterval/MakeLeaf/MergeSummaries/AssembleNode), RL-Atomic **collapses to ~0** compound accuracy; RL-Compound is substantially better.
- **RL-Compound** (operation chains): improves compound **and** preserves/improves atomic.
- **SFT vs RL**: programmatically template-filled CoT SFT performs in-domain but degrades sharply on longer out-of-domain lengths; RL length-generalizes better (consistent with Chu et al., "SFT memorizes, RL generalizes").
- Skill-switch census (Figure 5): RL-Atomic stays flat as the number of switches grows — atomic training does not teach transitions.
- **Beyond length**: composed training is robust under structural distribution shift (BST skew α, graph threshold) but transfers only **modestly** to unseen skills (KD-tree, geometric graphs, NL-embedded variants) — composition of known skills, not new-skill acquisition.
- **BFCL pilot** (tool calling): simple_python (atomic) vs parallel/parallel_multiple (compound) splits reproduce the asymmetry — the practical data-design claim.

### 1.3 The theory (the extractable math)

- **Lemma 5.1**: Pr[compound failure] ≤ **C_T(q)·ε_atom** with C_T(q) = sup_{c:q(c)>0} Σ_{t=1..T} μ_t(c)/q(c), μ_t = decision-context distribution at step t **conditioned on no earlier mistake**. Tight in the worst case.
- **Horizon floor**: C_T ≥ T whenever every rollout position is reachable mistake-free (equality iff Σ_t μ_t = T·q) — so even perfectly matched coverage pays T·ε_atom.
- **Prop B.2 (impossibility)**: no bound g(ε_atom) → 0 exists uniformly over horizons — a context-invariant policy can err on exactly 1/T of atomic contexts (ε = 1/T) yet fail with probability 1 on the T-chain. The T is structural, not a proof artifact.
- **Level III structural gap**: atomic training distributions are supported on singleton contexts ONLY; a multi-invocation (≥2 distinct ready pairs) rollout context has q(c) = 0 — the coverage assumption **fails structurally**, and Lemma 5.1 gives no constraint at those decisions. Even ε_atom = 0 permits Pr[failure] = 1 at Level III.
- **Reverse direction (§5.2 + Prop B.3)**: compound → atomic is easy — a first-step mistake kills the rollout, so ε⁽¹⁾_atom ≤ ε_comp with **no horizon factor**; transfer to covered atomic contexts pays only a first-step coverage factor C_1(q, ρ_T).
- Interpretation: error compounding (T context visits) + **coverage shift** (μ_t vs q mismatch at skill-switch interfaces and merge-input tuples) explain the asymmetry; Level III adds the support gap.

## 2. Three-track merge (adversarial panel, coordinator-verified)

Both advocates spawned in one round; findings below are merged. Per the spot-check rule, every code/lane fact cited into a filed artifact was re-verified this session.

**No-GD advocate** — the paper's apparatus is histogram arithmetic + DAG statics + two theorems; zero gradients needed to *consume* it. 14 quantities extracted; strongest:
1. `katgpt-core::compounding` substrate (coverage_factor / compound_failure_bound / direct_bound / first_step_bound / horizon_floor / error_budget / support_gap) with the typed `Vacuous` refusal (→ Issue 1222).
2. `context_support_probe` — support membership is EXACT set arithmetic: classify each campaign/session step's context by ready-set width (singleton / switch / merge-tuple / multi-choice); a multi-choice context voids the bound even at ε = 0.
3. `context_coverage_ratio` for decision-serving — per-bin μ-vs-q ratio over the calibration/serving context histograms; formalizes the **already-measured** rethink Bench-089 banking77 finding (armed rate axis transferring out-of-window under paraphrase shift) as an observable with Wilson-floor flags.
4. Doctrine laws, free: (i) no compound-reliability claim without a horizon-scaled error budget (C_T ≥ T); (ii) any proposal claiming atomic→composed transfer carries a coverage analysis or is refused (Prop B.2); (iii) fit/calibrate on composed slices first (Lemma 5.2's C_1 = 1 case) — the reverse transfers down for free.
5. Game-runtime instruments: quest-book `reach_profile` (ready-set width statics over the interaction DAG — a quest tree IS the paper's dependency graph), `multi_drive_context_census` (ticks where ≥2 sigmoid drives cross threshold — the paper predicts this is the failure-locus tick class beside the FSM), episode error-budget instrument (per-stage ε from runetrace → episode PROGRESS bound — the quantified form of the existing "assert PROGRESS, not one-tick choice" behavior law, which this paper independently derives from first principles).

**Model-based advocate** — STRONG applicability; **plan 492/issue 679's DSpine record is a prior, independent measurement of this paper's asymmetry on our own data**:
- Issue 679's T5 decision cell: the DSpine draft chain head never matched the target in ANY cycle (j-histogram [0,200,0,…], mean_j 1.000) while deeper rows agreed 0.34–0.66 — the rollout-context position least covered by the atomic (per-position teacher-forced) objective is precisely where acceptance dies.
- Plan 492 T3 arm-1 (mixed-frac on-policy CE, the best objective tried): mean_j 2.740 vs 2.660 zero-shot = +0.08 — "the objective works, the acceptance ceiling doesn't (trajectory front-loaded 3.636@c10 decaying monotonically)" — the coverage-shift decay signature Lemma 5.1 formalizes.
- The gate is COMPOSED (mean accepted tokens per cycle ≥ 3.2) while every objective tried optimizes PER-TOKEN marginals — the objective/gate misalignment the paper predicts.
- Recipe items (full detail in riir-train Issue 693): **R1** composed-reward GRPO for the DSpine head (reward = mean accepted tokens per draft-verify cycle on the Q4_K_M cudarc forward — a NEW OBJECTIVE, which satisfies riir-train Issue 493's explicit reopen condition "new head family (objective or architecture) or a target swap"; ~4.5–9 GPU-h, gated on the specdec-lane verdict that 679 leaves open); **R2** quest-grammar LoRA lanes — VERIFIED this session: `quest_training.rs`/`grammar_training.rs` emit ATOMIC samples (state→dialog mappings, single grammar patterns) — recompose to full state chains + GRPO under the crate's own deterministic checkers (`grammar_output` validity ∧ `svop_kg` cross-state fact-consistency; 2–8 GPU-h); **R4** asft (090)/edge_lora (353): add a longer-composition eval row (~0 GPU-h; the paper supplies the reason the RL continuation stage must carry the composed signal); **R5** distill corpora: session-sampling only where a composed consumer exists — single-shot decision serving untouched (honest negative).
- Pre-registered cells: **Cell A** (asymmetry validity on our quest schema, Plan-221 model class, equal-budget RL-Atomic vs RL-Compound arms, 1–3 GPU-h — publishable negative either way, standalone-claimable) and **Cell B** (the R1 DSpine composed-GRPO arm under plan 492's unchanged bands, 4.5–9 GPU-h, lane-gated).

**Coordinator corrections applied**: the advocate's "issue 493" citation verified against live issue 679 (the file itself is removed per the noise-reduction rule; the reopen condition is quoted verbatim in 679). Plan 492's live state read this session: T3 stop branch fired 2026-11-14, lease closed, owner-side lane verdict is the remaining step — 693 is an INPUT to that verdict, never an auto-start.

## 3. Path 0 inventory (per component)

| Paper component | Modelless analog | Status |
|---|---|---|
| Decision context Q(s) | exact set computation over a DAG/trace (ready set, skill-input pairs) | **extractable** (trivial; part of Issue 1222's consumer wiring) |
| ε_atom per context | empirical error frequency from persisted outcome counters (traj rows, ESC windows, harness fixtures) | **exists as data, not yet as a bound** (signal-diff vs traj_store below) |
| Coverage factor C_T | closed form over two context histograms | **extracted** → Issue 1222 (`coverage_factor` — named for the paper's term; `coverage_ratio` is already taken by `coverage_manifest.rs:409`'s covered/census fraction, a different signal) |
| Lemma 5.1 bound | closed form | **extracted** → Issue 1222 (`compound_failure_bound`) |
| C_T ≥ T + Prop B.2 | ordering law + impossibility — gate-design doctrine | **extracted** (doctrine; recorded here) |
| Level-III support gap | typed `Vacuous` refusal on μ>0 ∧ q=0 | **extracted** → Issue 1222 (`support_gap` / `Vacuous`) |
| §5.2 first-step reverse bound | closed form | **extracted** → Issue 1222 (`first_step_bound`) |
| Composed-task RL recipe | NOT modelless | **deferred to training track** → riir-train Issue 693 (Path 0.5; applicable, cheap, consumer exists) |
| SFT-vs-RL length generalization | training-side finding | **deferred** (folded into 693's R4 gate adds) |
| Three-level taxonomy | evaluation-design vocabulary | **extracted** (consumers classify their surfaces: chain / switch / branch-merge) |

No component is marked "covered" by name-match alone; the two adjacency signal-diffs:

- **`coverage_manifest` (katgpt-core, Issue 983)** — adjacent FAMILY (coverage-shaped), distinct SIGNAL: its three suspicion predicates read per-section count/digest censuses against pinned per-query-class expectations (memory-coverage deficits). The paper's C_T is a sup importance-ratio between rollout-context mass and atomic-context mass with mistake-free conditioning — no pinned expectations, no sup-ratio, no rollout conditioning. Consume both; no merge.
- **`esc_voi` (rethink, Issue 047)** — adjacent observer (escalation rate, precision, value-per-escalation with CI), but window-scoped per decision; the compound bound composes across the session horizon and adds the support-gap type. Not covered by it.
- **`traj_store` (refine, Plan 019)** — ships the composed trajectories as DATA (rows + outcome counters) but computes no coverage factor, no compound bound, no context-width census. The substrate for B1; not a cover.
- **R264 PTG** — the framework cousin: PTGs mine EXECUTED traces for motif promotion (capability growth); 2609.19465's dependency graph is an EVALUATION formalism with a transfer bound (data-design and measurement). Complementary: PTG = what to build next; C_T = will it hold when composed.

## 4. Published prior-art sweep (§4, run before the verdict)

- Internal-first: `grep 2609.19465` across every workspace `.research/` + `.plans/` → **zero hits**; title-fragment grep → four unrelated files (Engram, latent-reasoning survey, NEO/L2T, spectral rewiring).
- Closest published cousins (the paper cites Yuan + Composition-RL; the sweep adds one it does not cite):
  - **arXiv:2512.01970 "Atomic Skills are the Prerequisite: When Reinforcement Learning Synthesizes Compositional Reasoning, and When It Only Amplifies"** (Cheng, Yin, Zhou et al.; v1 Dec 2025, v3 May 2026 retitle; cited-by 5, HF checkpoint + code released) — the closest published study on this lane, and a boundary-condition COMPLEMENT, not a subsumption: on Complementary Reasoning (parametric + contextual atomic skills), SFT directly on the composite rote-memorizes (90% seen / 18% novel) while RL synthesizes composite strategies **only when the base model first mastered the independent atomic skills via SFT**. Combined with 2609.19465 the standing curriculum law is: **SFT-atomic bootstrap → RL on the composed task synthesizes; atomic-RL drills alone never compose; composed supervision without atomic mastery does not synthesize either.** (19465 contributes the asymmetry + the coverage-factor theory + the Level-III support gap; 2512.01970 contributes the atomic-prerequisite boundary.)
  - **Yuan et al., arXiv:2509.25123 "From f(x) and g(x) to f(g(x))"** — RL CAN teach nested composition (positive result; the paper's boundary condition on when).
  - **Composition-RL, arXiv:2602.12036** — compose prompts into harder verifiable RL training data (the method-level sibling of the paper's data-design claim).
  - **Chu et al., "SFT memorizes, RL generalizes"** (ICML 2025) — the SFT-vs-RL length-generalization leg.
  - **Dziri et al., "Faith and Fate"** — computation-graph degradation with compositional complexity (the descriptive precursor; no training-axis, no bound).
  - Compounding folklore (0.95^10 multi-agent posts, MAKER's million-step zero-error framing) — the naive series product is common knowledge; the coverage factor, the C_T ≥ T floor, and the structural support gap are the paper's delta.
- Verdict-relevant conclusion: no published OR shipped analog of the bound-as-instrument exists in the workspace (grep-verified: katgpt-core has no coverage_factor/failure_bound/error_budget/compounding module; refine score_bench has no campaign/session metric). The modelless claim is pinned below.

**Pinned novelty claim (§1.5 precondition):** *a coverage-factor + compound-failure bound instrument (C_T from context histograms; typed Vacuous for the Level-III support gap; C_T ≥ T horizon floor) for the self-evolve stack's composed surfaces (refine fix campaigns, rethink ESC sessions, quest chains), consuming already-persisted outcome counters — distinguished from naive 0.95^10 compounding by the μ-vs-q coverage factor, the impossibility floor, and the support-gap type.* Scored honestly: Q1 yes (nothing ships; the math is published but consuming it is not), Q2 instrument-class (a new measurable, not a new NPC behavior — the reason this is NOT scored Super-GOAT), Q3 finishable ("our rulewright bounds multi-edit campaign success before running it"), Q4 yes (≥3 pillars). → **Gain, not Super-GOAT**; POC filed at Issue 1222 with a runnable proof path, no GOAT plan.

## 5. Fusion

The novel combination this paper × the stack enables: **composed-success accounting as a standing instrument**. `traj_store` rows (refine) + ESC windows (rethink) + quest DAGs (seal-std/riir-ai) already persist every input the bound needs; none of them computes it. Fusing the coverage factor with the trajectory store yields a pre-run answer to "will this campaign hold" — and the reverse-direction law (fit composed first) yields a free cal/eval design rule for the ESC gate (the Bench-089 lesson, formalized). The training half fuses with plan 492's own stop-branch record: the paper's theory names the failure class our data already exhibits and prescribes the objective class (composed rollout reward) that Issue 493's reopen condition admits.

## 6. Verdict

**Gain — two tracks, no cascade.**

- **Track (c) trained weights: Gain.** Actionable (unblocks a deferred lane decision; a measured prior exists). Filed: riir-train Issue 693 (law + Cell A/B pre-registration + R1–R5). Training runs are NOT launched this session: Cell A needs its harness built first; R1 is gated on the specdec-lane verdict 679 leaves open (delegated-verdict class under the 2026-11-11 gating decree — 693 supplies its decisive input).
- **Track (a) modelless inference: Gain (POC tier) — POC LANDED 2026-11-16 (`3746d0f14`, issue file removed; the full landing record is in git history at that commit).** Shipped as `katgpt-core/src/compounding.rs` under the `estimators` umbrella (opt-in, PARENT-gated — no new feature key; the ratchet holds katgpt-core at the 290 ceiling): `coverage_factor` (Lemma 5.1's C_T, typed `Vacuous` for the Level-III support gap, `Undecidable` for u64-overflow/thin-support) + `compound_failure_bound` (clamped, refusal-typed) + `direct_bound`/`first_step_bound`/`horizon_floor`/`error_budget`/`support_gap` + the zero-alloc `coverage_factor_with` scratch fold (O(|C|) two-pointer sorted merge — 0.24× of the naive accumulate-then-sort arm; the determinism tax measured as a determinism DIVIDEND). G1 fixture to 1e-12 + 100-seed property test + monotonicity/horizon-floor/clamp/falsification arms (15 in-module tests). ⚠ ONE SPEC TIGHTENING recorded at landing: the bound ≤ C_T·ε_atom theorem holds only under MATCHED per-step mass (C_T ≥ T·M/Q; M < Q = the under-sampled regime) — the property test generates matched-mass rollouts only, and the module doc carries the honest derivation. Escalates to a GOAT plan when a consumer wires it (§5's surfaces); the TIGHTNESS readout law (T6) is the consumer-side audit.
- **Doctrine (free, recorded here):** (i) any compound-reliability claim carries a horizon-scaled error budget; (ii) atomic→composed transfer claims carry a coverage analysis or are refused; (iii) fit/calibrate composed-first — the reverse transfers down for free; (iv) the two-paper curriculum law: SFT-atomic bootstrap → RL-composed synthesis — composed-primary with an atomic-SFT bootstrap, never atomic-RL drills, never composed-only (unseen-skill transfer stays modest — drills keep their bootstrap role).

**Weakest point (named):** the bound's usefulness stands or falls on context binning — hashing code shapes / decision contexts into comparable keys is a modeling choice (contrastive_scope precedent applies), and survivorship conditioning of μ_t biases the estimate on truncated rollouts. Both are estimator-design items recorded in Issue 1222 before any consumer wiring; a mis-binned context set makes C_T either vacuous-everywhere (useless) or 1.0-everywhere (vacuously comforting). The G1 falsification arm pins the arithmetic, not the binning — the binning's audit is the consumer-side TIGHTNESS readout (Issue 1222 T6: bound/measured-failure ratio + non-vacuous fraction of campaigns); a dominance check ("bound ≥ measured") alone is not an audit — any loose bound passes it.

## PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** Cheng, Yin, Zhou et al. [arXiv:2512.01970 "Atomic Skills are the Prerequisite: When Reinforcement Learning Synthesizes Compositional Reasoning, and When It Only Amplifies" (v3 retitle; v1 "From Atomic to Composite")] — the closest published study on this lane and a boundary-condition COMPLEMENT: RL synthesizes composite strategies only after SFT mastery of the atomic skills; recorded here rather than its own note because the measurement axis this stack consumes is 2609.19465's (asymmetry + coverage-factor theory + support gap), with 2512.01970 supplying the atomic-prerequisite boundary; training-recipe residue (the combined curriculum law) routed to riir-train Issue 693.
> **PASS-Redirects (synthesis):** Yuan et al. [arXiv:2509.25123 "From f(x) and g(x) to f(g(x)): LLMs learn new skills in RL by composing old ones"] — the positive-composition boundary condition; same routing (693), no separate file.
