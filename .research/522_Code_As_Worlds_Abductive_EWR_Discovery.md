# Research 522: Code-as-World — Agentic Discovery of Executable World Representations [Track A Super-GOAT / Track B GOAT]

> **Source:** [arXiv:2608.27549 "Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning"](https://arxiv.org/abs/2608.27549) — MirroS / Tsinghua / PKU / NTU, 27 Aug 2026 (cs.CV)
> **Date:** 2026-08-31
> **Status:** RECORD — Track A (modelless/runtime) = **Super-GOAT** on the fused claim; Track B (training) = **GOAT** (plan filed, secondary). Outputs shipped this session: this note + `katgpt-rs/.plans/584_*` + `riir-ai/.research/358_*` + `riir-ai/.plans/560_*` + `riir-ai/.issues/835_*` + `riir-train/.plans/366_*`.
> **Open primitive:** katgpt-rs Plan 584 (`katgpt-core/src/abductive.rs` — hodge-blame routing + `AbductiveRepairKernel` trait seam)
> **Related:** katgpt-rs 275 (Code World Models — the DeepMind sibling paper, Super-GOAT), 506 (LDR closed-form kinematic rollout), riir-ai 345 (anticipatory think-brain, PoC REFUTED — the honesty precedent), riir-ai 145 (CWM Runtime NPC-induced rules), riir-ai 341 (EVPI active perception), riir-ai 134 (two-brain bridge), riir-ai 312 (6-channel interpretability), riir-neuron-db archetype blend

---

## TL;DR

Code-as-World represents a physical world as an **executable world representation (EWR)** `p = (C composition, E evolution, A appearance)` over a simulator interface, and recovers it from observations via an **abductive agentic loop**: propose → instantiate → execute → render → verify, with structured frame-level diagnosis Δ routing **local component-wise revision**; the loop terminates on Accept (sufficient ∧ parsimonious) or **REJECT** after budget K. Verified EWRs then generate exact world-space QA supervision; GRPO (`r = exp(−|ŷ−y|/(|y|+ε))` + unit + format rewards) trains Code-as-World-VL to SOTA on QuantiPhy, beating Gemini-3.1-Flash.

For our stack the VLM/QuantiPhy half is the paper's consumer, not its content. The transferable mechanisms are (1) the **diagnosis-structured, budget-bounded, reject-capable discovery loop** — structurally the riir-clippy heal loop with a world simulator as the oracle — and (2) **component-wise belief repair**: an agent that holds a structured world hypothesis, forward-simulates it, and when observed events contradict the prediction, diagnoses WHICH component broke and repairs exactly that component. Fused into the two-brain model this is a new runtime capability class: **per-NPC abductive belief repair** — the think brain stops being a frozen snapshot (or a 345-style extrapolator) and becomes a *repaired micro-theory* of each tracked entity. That fused claim passes the novelty gate 4/4 (§6) → Super-GOAT, with the falsifiable PoC (riir-ai Issue 835) as the blocking gate — the 345 REFUTED precedent applies in full: beliefs stay frozen until the PoC wins.

---

## 1. Paper core findings

1. **EWR = (C, E, A) over one simulator interface.** C = entities, geometry, metric dims, physical params (mass, friction, gravity); E = initial states, temporal changes, key events, sim duration; A = camera/background/materials — appearance does NOT alter the physical process. MuJoCo backend with **two interchangeable engines behind one interface** (kinematic animation engine: time-varying poses; physics engine: forces/contacts/constraints). Physical equivalence over pixel duplication is the stated priority.
2. **Agentic discovery loop (Alg. 1).** Per round: `ModifyEWR(A, η, p, Δ)` → `CompileEWR` → `RunSimulation` → `Render/Project` → `CompareAndDiagnose` per selected key frame → aggregate frame-level discrepancies into structured Δ → local revision of the blamed component. Terminate on `Accept(A, p, Δ)` (sufficient + parsimonious) or **REJECT** at K=5.
3. **Evidence adapters per modality.** Text → semantic constraints + priors + defaults; video → depth/masks/tracks (SAM3, VGGT-Omega, SAM3D). The data filter is regime classification (camera-motion, motion-sufficiency, single-phenomenon purity).
4. **Headline loop result.** Iterative verify-guided refinement beats Best-of-5 independent sampling at matched evaluation budget on most metrics (Visual Alignment, Object IoU, Traj-ADE, Accuracy@2%D), on both engines.
5. **Application (the training track).** Phase-1 SFT on 73,335 image-space QA pairs auto-generated from RefCOCO*/GOT-10K boxes+tracks via central differences (Eq. 2/14 — pure arithmetic). Phase-2 GRPO on ~2.5K world-space VQA pairs whose labels are **read back from verified EWR state records** (exact by construction). Reward `r_num = exp(−|ŷ−y|/(|y|+ε))` + λ_u·unit + λ_f·format. 4B/9B direct-answer on 8×H100; 27B reasoning variant outcome-only (no process labels, no KL, eval at step 60). SOTA on QuantiPhy (55.4 / 58.6 MRA vs Gemini-3.1-Flash 54.8).
6. **Self-named limitations.** (a) Processes outside the simulator's modeling scope → the loop "may converge to a locally plausible EWR without recovering a mechanistically accurate explanation"; (b) the trained model does NOT internalize the discovery loop — hypothesis construction, diagnosis, and revision stay external to the model.

---

## 2. Path 0 inventory (three-track decomposition)

| # | Paper component | Modelless analog / extraction | Track | Disposition |
|---|---|---|---|---|
| 1 | Discovery loop (propose→execute→verify→repair→reject) | riir-clippy heal loop IS this shape (draft→apply→cargo-check→revert, bounded fixpoint, first-class decline) — document the isomorphism, reuse its budget/revert discipline | a | Covered (structural); the *world-hypothesis* instantiation is new → Plan 584 |
| 2 | Central-difference kinematics (Eq. 2/14) | `katgpt-core::kinematics::finite_difference_state` (Research 506 / Plan 578) — same stencil | a | **Analog exists** — signal-diff: consumes the same ring; the paper adds only the question-template layer (a track-c consumer) |
| 3 | Scale calibration γ = ρ/ρ_pix, y = γ·y_pix | one divide+multiply; raw-domain committed calibration → LatCal fixed-point | a | Covered (riir-chain latcal EPSILON/SCALE theorem family) |
| 4 | Structured Δ → local component revision | **No analog.** Nearest: `zone_gating` + `reestimation` (freeze-others, re-fit blamed latent) — primitives exist, the blame-routing composition does not; `dec::hodge_decompose` over the residual field supplies the routing law | a | **OPEN → Plan 584 flagship** |
| 5 | Arithmetic verifier (active-set-renormalized frame score, Eq. 5–6) | IoU + median + mean + Σw·s/Σw over validity-gated active sets — fully closed-form; `SenseModule::project` shape | a | Analog exists (aggregate-on-valid-set); adopt sentinel conventions as pin tests |
| 6 | SelectFrames (key-frame verification budget) | EVPI gate (Plan 544): verify only frames where accept/reject could flip | a | Covered — fuse into the PoC harness |
| 7 | Accept = sufficient ∧ parsimonious; REJECT at budget | Pareto predicate + bounded-fixpoint exhaustion + first-class decline — healer semantics | a | Covered; reject→**re-classify** (archetype re-blend) is the new half |
| 8 | Dual engines behind one interface | trait seam; kinematic engine = closed-form rollout, physics = escalation per `regime_predicates` | a | Analog exists (`EvpiSource` pluggable-source pattern); design note in Plan 584 |
| 9 | Regime-based evidence filter | `regime_predicates` — computable predicates; refuse out-of-regime up front | a | Covered |
| 10 | QA-pair synthesis from verified state records | deterministic sampling + validity predicates over replay records — **exact labels for free from our deterministic authority sim** | c | **OPEN → riir-train Plan 366 T1** (modelless generator, trained consumer) |
| 11 | Dense numeric reward `exp(−rel_err)` + aux | closed-form reward kernel; **fixes the measured zero-advantage-group failure** of binary rewards in `loss_grpo.rs` (`ZeroAdvantageReport.skipped_fraction` is the standing metric) | c | **OPEN → Plan 366 T2** |
| 12 | Two-phase curriculum (SFT grounding → RL calibration) | sequenced trainers exist (`gemma2_cubecl_train` → `loss_grpo`) | c | Recipe transfer → Plan 366 T4 |
| 13 | Outcome-only reasoning supervision, no KL, eval-at-60 | `thinking_lora.rs` route + `ScoreDiag` parse taxonomy | c | Recipe transfer → Plan 366 |
| 14 | A-channel (appearance); sim-to-real video re-rendering (Wan2.2-VACE) | — | — | **DISCARD — no consumer.** Our appearance channel is the view layer (riir-unity/bevy), already decoupled by the G5 boundary gates; we do not generate video. The decoupling *principle* is already enforced (no appearance field in the state channel). |
| 15 | Video evidence adapters (SAM3/VGGT-Omega/SAM3D) | — | — | **DISCARD — no consumer.** No video surface ships in the runtime; the evidence-adapter role is played by the think-brain observation ring (fog-gated sightings). |
| 16 | 8×H100 VLM training; 27B reasoning variant | — | — | **DISCARD — out of scale, no trainee.** Shipped trainers cover Gemma-2-2B (Metal), Gemma-4 QLoRA, Kimi-K3 (CUDA); no vision encoder on any trainee. Our lane is the small direct-answer lane only. |
| 17 | MRA threshold ladder + answer parsing | — | — | **DISCARD as standalone** — parse-gate + scored thresholds already shipped in rlvr parsing (`ScoreDiag`); no delta. |
| 18 | JEDi MMD / TRAJAN Fréchet distribution gates | closed-form statistics | — | **Downgrade to rider** — optional distribution check inside the Issue 835 PoC harness; `consolidation.rs` population gates already carry the gate shape. Not a standalone primitive. |

**Funnel:** every mechanism-level row lands in exactly one of (a) Plan 584 (rows 4, 7-new-half, 8-design-note), (b) Plan 366 (rows 10–13), (c) an audited discard (rows 14–18, reasons inline). No row dropped silently.

---

## 3. Panel merge (adversarial, one spawn round)

- **No-GD advocate** returned 26 ranked mechanisms; merged into rows 1–9 + 18 and the fusion table. Strongest novel mapping adopted: **hodge decomposition of the residual field as blame routing** (exact→C/state, solenoidal→E/dynamics, harmonic→structural/reject) — the Plan 584 flagship. Its riir-clippy-loop isomorphism (world sim ≡ cargo check; Δ ≡ typed diagnostics; reject ≡ revert) is recorded as structural reuse, not a new primitive.
- **Model-based advocate** returned 10 recipe items with verified shipped-code citations; flagship adopted as Plan 366 (world-QA generator + dense reward + curriculum + GOAT vs the modelless kinematic trio on **intervention worlds**, where dead-reckon structurally fails). Honest non-transfer classes recorded in Plan 366 (rows 14–16 above).
- **Prior-art advocate** verdict: Claim A (per-NPC component-diagnosed belief repair with reject semantics) — **no direct published prior art found** within the search envelope; closest = Reiter-lineage model-based diagnosis (owns component-wise fault localization; wrong domain: static declarative models) composed with WorldCoder/WAV (own verify→refine loops; whole-program/holistic, no diagnosis taxonomy, no reject/re-classify). Claim B (replay-verified partial-obs-question/exact-answer pairs) — no direct prior art; sits at the intersection of privileged-simulator-state distillation (Learning-by-Cheating lineage, π-Distill) and document-derived synthetic QA. **Residual risks recorded:** the unread dos Santos preprint ("The World Is a Hypothesis" — epistemic control for executable world-model agents) must be read before any external novelty claim; 4/10 queries timed out; absence-of-evidence is bounded by query coverage.

## 4. Signal-diff vs the shipped cousins (mandatory before "covered"/"novel")

| Cousin | What it consumes / does | What Code-as-World × our fusion adds | Verdict |
|---|---|---|---|
| katgpt-rs 275 (cwM, REx) | REx = Thompson-tree refinement over **whole CWM hypotheses**; edges = LLM rewrites conditioned on a failing unit test; LLM cold-tier; unit tests = whole-transition assertions | Component-LOCAL repair: hodge-blame names the component; closed-form re-fit of a few typed scalars (no LLM in the repair path); REJECT→re-classify semantics; fog-gated residuals at runtime cadence | Complementary: 275/145 = macro loop (learn a whole game, cold-tier); 522 = micro loop (repair a per-entity theory, runtime). REx remains the escalation when local repair exhausts |
| riir-ai 145 (CWM Runtime) | "Transition acc < τ_acc → re-INDUCE (REx refinement)" — whole-kernel replacement; per-NPC CWM pool; cgsp curiosity-triggered induction | Same complementarity; guide 358 positions belief repair INSIDE 145's program as the between-inductions maintenance loop | Fuse, not duplicate |
| riir-ai 345 (anticipatory belief, PoC REFUTED) | `residual_event` emits surprise event bits (z-score/CUSUM/impulse-vs-force); trajectory extrapolation refuted on decision metrics | Upgrade path: residual bits → blame-routed REPAIR of a structured hypothesis (state/param/rule), with reject→archetype re-blend; decision-metric discipline + pre-registration lessons (spread-vs-margin trap) carried into Issue 835's bars | Extends; the PoC gate is inherited |
| riir-clippy heal loop | draft→apply→compile-verify→revert, bounded fixpoint, decline-as-None, EvolveRecorder trajectories | The isomorphism lets the loop scaffolding (budget/decline/trace) be reused verbatim — recorded, not rebuilt | Structural reuse |
| `GenericSpatialBelief` (shipped) | pos + confidence; decay + fog-gated re-observe | Adds `EntityWorldHypothesis` (state + params + rules) ABOVE the position belief; position stays the raw ground-truth channel | Strictly additive layer |

## 5. Fusion (what none of them alone produces)

1. **Hodge-blame-routed belief repair (flagship).** Per tracked entity, maintain `EntityWorldHypothesis { state, params (speed/aggro/schedule…), rules (sleep-window/flee-trigger…) }`; forward-simulate via the kinematic engine (closed-form, `kinematic_extrapolate`); residuals `r = predicted − observed` form a discrete field per entity; `hodge_decompose` routes: **gradient (drift)** → initial-condition/scale → revise state (C); **coexact (oscillation/phase)** → dynamics/timing → revise rule/param (E); **harmonic** → structural mismatch (wrong entity class) → reject. Repair = closed-form least-squares re-fit of ONLY the blamed component under a `zone_gating`-style locality gate (untouched components bit-identical). K-round budget → reject → **archetype re-blend** (riir-neuron-db: "sleeper" weight ↓, "nocturnal" weight ↑) + KG triple emission (`entity-belief-revised`, extending 345's `entity-acted-non-kinematically`). Modelless end-to-end: zero gradient steps, ns–µs cost class, think-brain only.
2. **Heal-loop isomorphism.** The world-discovery loop is the riir-clippy loop with a simulator as oracle — Plan 584 reuses its budget/decline/trajectory-recording discipline rather than inventing new loop machinery.
3. **Verified-replay world-QA × privileged distillation (track B).** Our authority sim is the verified world the paper paid 8×H100 to build: replay records carry exact full-state labels; questions posed over fog-of-war partial views = privileged→observable supervision (riir-train 419 LOPD adjacency). The dense numeric reward unlocks gradient signal where binary rewards produce zero-advantage groups.
4. **EVPI frame/tick selection.** Verify only decision-relevant ticks (Plan 544 gate) — cuts repair-loop verification cost at equal decision accuracy; paired into the PoC.
5. **Belief hypotheses as a 7th interpretability channel.** `EntityWorldHypothesis` is inspectable state — a GM panel can show *"NPC believes wolf speed 3.2 u/t, sleep window 600–1800, confidence 0.4, last repaired tick 1203 (rule slot)"* — extending riir-ai 312's 6-channel interpretability stack with a mechanistic-belief channel.

## 6. Verdicts (per track — one verdict per track, not one per paper)

### Track A (modelless / game runtime): **Super-GOAT** on the fused claim

| Q | Answer | Evidence |
|---|---|---|
| Q1 no prior art | ✅ (bounded) | Web: no direct published prior for component-diagnosis + reject/re-classify on runtime agent world-hypotheses (closest: MBD diagnosis lineage — wrong domain; WorldCoder/WAV — holistic repair, no diagnosis taxonomy). In-stack: no component-local repair anywhere — 145 re-induces whole kernels, 345 extrapolates trajectories, the heal loop repairs code not world beliefs. **Caveats:** dos Santos preprint unread (flagged in guide 358); envelope-bounded negative. |
| Q2 new behavior class | ✅ | "NPCs revise their THEORY of an entity, not just their map": mechanism-level self-repair with explicit re-classification. No incumbent ships it; nothing in-stack does. |
| Q3 selling point | ✅ | *"Our named NPCs run a private micro-simulation of what they believe about each monster — and when reality breaks the simulation, they diagnose WHICH belief broke (position? speed? sleep schedule?) and repair exactly that belief — or, after five contradictions, stop believing the monster sleeps at all and re-classify it."* |
| Q4 force multiplier | ✅ | ≥4 pillars: two-brain (`GenericSpatialBelief`), DEC (`hodge_decompose`), kinematics (Research 506 / Plan 578), archetype-blend shards + KG triples, EVPI, 145's CWM program, 312's interpretability stack. |

**Blocking gate:** the falsifiable PoC (**riir-ai Issue 835**, harness `crates/riir-poc/tests/abductive_repair_poc.rs`) must PASS before any consumer wiring — the 345 precedent (borderline REFUTED → beliefs stayed frozen) governs. Pre-registered bars avoid 345's spread-vs-margin trap (sign test ≥13/16 seeds + mean margin, not max−min spread).

### Track B (trained weights): **GOAT** — `riir-train/.plans/366_verified_world_qa_training.md`

Affordable (Kimi-K3-0.4B bring-up <4 GPU-h; flagship ≈ one 4090 weekend), concrete measured gap (zero-advantage groups under binary rewards), modelless-reusable data generator + eval harness as by-products. **SECONDARY track by serving-envelope fit**: the modelless repair loop runs in the 20 Hz hot path; a trained numeric predictor would serve think-cadence and must beat the modelless kinematic trio at its GOAT gate before earning any slot. Path-0.5 documentation (decomposition / paths 1–3 / what requires GD / affordability / dual-track contribution) lives in Plan 366.

## 7. Mandatory outputs (shipped this session)

1. ✅ **Open primitive** → `katgpt-rs/.plans/584_abductive_repair_kernel.md` (generic hodge-blame routing + repair-loop seam; opt-in `abductive_repair`; no game semantics)
2. ✅ **Private guide** → `riir-ai/.research/358_Abductive_Belief_Repair_Runtime_Guide.md`
3. ✅ **Private plan** → `riir-ai/.plans/560_per_npc_abductive_belief_repair.md` (Phase 0 = PoC gate)
4. ✅ **PoC issue** → `riir-ai/.issues/835_abductive_belief_repair_poc.md`
5. ✅ **Training plan** → `riir-train/.plans/366_verified_world_qa_training.md`
6. ✅ **PASS-Redirects** → one-line cross-refs appended to 275, 145, 345

## 8. Latent vs raw boundary (non-negotiable)

- `EntityWorldHypothesis` is **think-brain latent state**: never synced, never committed as world truth, never rendered as fact. The info brain's raw `MapPos` stays the only ground truth; **a repaired belief never validates movement** (anti-cheat consumes raw exact positions — unchanged).
- What MAY cross sync: derived scalars only (confidence, regime enum as u8, the 5-affect-scalars class) via the sigmoid/dot bridge — same discipline as 345.
- KG triples (`entity-belief-revised`) are the social-domain product of latent similarity, per the house KG-triple emission rule — never substitutes for raw TxDelta of physical events.
- Archetype re-blend updates a **shard** (riir-neuron-db commits latent as-is, BLAKE3-committed) — the sanctioned Cold-tier path, not a sync violation.
- The verified-replay QA generator (track B) reads full state authority-side; the trained consumer sees only the partial-observation rendering — privileged context never ships to the client.

## 9. What NOT to take

- The VLM, QuantiPhy, video ingestion, sim-to-real rendering, MuJoCo scene synthesis — no consumers; the paper's own limitation (b) ("loop external to the model") is an argument FOR our shape: keep the loop external and deterministic.
- LLM-in-the-loop repair: our repair path is closed-form arithmetic on typed scalars; REx/LLM synthesis stays the cold-tier escalation (275/145 territory) when local repair exhausts.
- Appearance (A) as a runtime component: view-layer concern, already boundary-gated.

## 10. Validation protocol

1. **PoC (blocking, Issue 835):** 3 arms × 3 fault classes + 1 structural class; falsifiable decline gate (a planted unexplainable world must reach REJECT, not a forced fit); G1 bit-identical double-run; pre-registered sign-test bars; verdict pinned by a recorded constant.
2. **GOAT (on PoC PASS):** G1 determinism; G2 ns–µs per repair round at 1000 tracked entities; G3 default-off `belief_repair` until pass + count pins; G4 alloc-free (caller-owned scratch). §3.6 defend-wrong framing is satisfied by the PoC itself.
3. **Track B GOAT:** Plan 366 T4 — trained predictor vs `project_lead_position`/decay-projection/linear-extrapolation trio on unseen intervention worlds; MRA-ladder metric; adapter hot-swap with zero per-tick regression.

**§10 outcome (2026-09-01) — Track A PoC VERDICT: REFUTED at registered bars** (riir-ai [Bench 835](../../riir-ai/.benchmarks/835_abductive_belief_repair_poc.md), Issue 835, harness `riir-poc/tests/abductive_repair_poc.rs`): hodge-blame-routed repair WON the decision sign-test (14/16 seeds, composite 0.606 vs frozen 0.505) and passed cost (1.65× ≤ 2×), vs-whole-refit (≥ decisions at 0.052× cost), decline-falsifiability (16/16 REJECT + re-classify, zero forced fits), locality, and determinism — but missed the margin bar (+0.101 < +0.150). Mechanism: evidence starvation under regime faults (window-fault entities freeze out of fog-of-war sight; the router never sees the rest/motion signature), NOT the blame routing. Beliefs stay frozen; Plan 560 closed-negative; Plan 584 → OPEN-DEFERRED with reopen triggers. Track B (Plan 366) unaffected.

## 11. Paper metadata

| Field | Value |
|---|---|
| Authors | Hanyang Wang, Yimo Cai, Weiliang Chen, … Jialong Wu (project lead) — MirroS, Tsinghua, PKU, NTU |
| arxiv | 2608.27549v1, 27 Aug 2026 (cs.CV) |
| Code / project | github.com/mirros-lab/code-as-world · mirros-lab.github.io/code-as-world |
| Simulator | MuJoCo; dual engines (animation kinematic / physics) behind one EWR interface; Wan2.2-VACE re-rendering |
| Loop budget | K=5 rounds; beats Best-of-5 at matched budget on both engines |
| Training data | 73,335 image-space SFT pairs (RefCOCO*/GOT-10K) + 1,585 text-driven + 988 video-driven world-space GRPO pairs |
| Headline | Code-as-World-VL-9B 55.4 / 27B-reasoning 58.6 MRA on QuantiPhy vs Gemini-3.1-Flash 54.8 |

---

## TL;DR

Code-as-World (MirroS/Tsinghua) is the abductive-discovery sibling of DeepMind's cwM (our Research 275): worlds as executable code, recovered by a propose-execute-render-verify loop with structured diagnosis Δ, local component revision, parsimony-gated Accept, and budget-bounded REJECT; verified worlds then supervise VLM training to QuantiPhy SOTA. We take the loop, not the VLM: fused with the two-brain model, the new capability class is **per-NPC abductive belief repair** — hodge-blame-routed component repair over per-entity world hypotheses with archetype re-classification on reject (Track A Super-GOAT; blocking PoC = riir-ai Issue 835; open primitive = Plan 584; guide = riir-ai 358). The training half (verified-replay world-QA + dense numeric reward) files as riir-train Plan 366 (Track B GOAT, secondary by serving-envelope fit). Published-prior-art check: no direct prior for the fused mechanism (closest: MBD diagnosis + WorldCoder; caveats recorded); in-stack: strictly additive over 275/145/345.
