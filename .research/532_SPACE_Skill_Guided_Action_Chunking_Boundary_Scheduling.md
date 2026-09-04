# Research 532: SPACE — Skill-Guided Adaptive Action Chunking (Boundary-Sourced Decision Scheduling)

**Paper:** "Act More, Decide Less: Skill-Guided Adaptive Action Chunking for Long-Horizon LLM Agents" (Yang et al., Rutgers/Toronto/PolyU/Amazon/Microsoft, arXiv:2609.02042, EMNLP 2026 Camera Ready)
**Date:** 2026-09-04
**Status:** RECORD
**Verdict:** **Modelless track: PASS** (the when-to-decide class ships in six+ implementations; the mined-boundary variant has no live consumer). **Training track: GAIN** — one `.issues/` filed at `riir-train/.issues/512` (chunk-aware credit-assignment deposit + conditional L4 trajectory-expansion re-attempt, owner-gated, kill-criterion stated).

---

## TL;DR

SPACE trains LLM agents to emit variable-length action chunks instead of one action per ReAct round. Naive multi-action RL fails two ways — single-action collapse or over-commitment — because sparse terminal reward carries **zero information about where chunk boundaries belong**. SPACE's fix: mine two-level programmatic skills from *successful* trajectories (composite skill = ordered subskill calls; each subskill = exactly one chunk), so subskill boundaries become direct chunk-boundary supervision, distilled via hybrid on-/off-policy optimization with anchor-state-grouped credit assignment. +7.0–31.3% success, up to 78.9% fewer LLM rounds, chunk-level best-of-N test-time search gains ~2× the SR of primitive-action search at structurally lower call cost.

**For this stack:** the paper is one training recipe wrapped around a modelless decision-scheduling insight. The insight class ("state-dependent compute budget with endogenous deliberation cost") **already ships** — Research 363's coverage table lists six independent implementations (R149/P304 gain-cost halting, R218 breakeven router, R350 density scheduling, P231 PathwayTracker, P263 freshness gate, P194/R283 CoT bandit). The two things that are genuinely NEW here — (a) boundary *source* = mined successful-trajectory task structure rather than self-telemetry, and (b) the failure-mode taxonomy (collapse/over-commit as boundary-absence symptoms) — both turn out to be **validations of shipped design** rather than gaps: the shipped `deliberation_cadence` upgrade independently implements both guards (heading-churn trigger = collapse guard; settled early-commit = over-commit guard), and tree-verify (Bench 697: tree acceptance 0.8785 vs chain 0.30) already exploits chunk-level candidates as search units. The training recipe has real but consumer-less applicability — filed as an issue, not a plan.

---

## 1. Paper core findings (full read)

1. **The boundary problem, not the chunking problem.** ReAct agents burn most rounds on routine action sequences. Allowing variable-length multi-action output under GRPO produces two failure modes (Fig. 1): Qwen3-4B collapses to ~1 action/round; Llama-3.1-8B over-commits to 5–6 actions/round with *lower* success. Both are boundary-learning failures: terminal reward gives no gradient at boundaries.
2. **Skill structure as free supervision.** Successful trajectories are segmented (LLM-prompted) into composite skills of ordered subskill calls; static filters (syntax/compilability/signature), AST-canonicalized dedup, ~20/category cap, prune at zero long-term success. Each subskill emits exactly one chunk → subskill boundaries ARE chunk boundaries.
3. **Hybrid rollouts + expansion.** ρ_prim ∈ {0.5, 0.75} primitive-chunk vs skill-augmented modes; skill calls are Expand&Relabel'd — unrolled along the subskill sequence into chunk-level supervision targets (off-policy self-imitation, Oh et al. 2018; λ_off 0.05–0.1; w(A)=clip(Â,0,w_max)).
4. **Chunk-aware credit.** Two-level advantages (GiGPO-style): trajectory-level group-normalized reward ⊕ per-round discounted return γ_c broadcast across the chunk's actions, then re-normalized grouped by **anchor observation state**. Ablation: removing it costs 96.1→90.6 SR *and* 5.0→5.6 rounds; removing skills costs 96.1→86.7.
5. **Deployment without the library.** The trained policy emits chunks directly; no skill access at test time. Learned regime: ~3–4 actions/round (non-degenerate interior).
6. **Chunk-level test-time search.** Best-of-N over chunk candidates: +8.3 SR vs +4.2 for primitive candidates, at 48.3 vs 74.9 LLM calls — ~3.1× gain per call (2× per-candidate quality × 0.64× candidate count). "Chunk policies and TTS are genuinely complementary."
7. **Efficiency.** Reaches multi-action GRPO's final performance by 26.6% of training steps; converges faster with higher entropy (broader exploration, no collapse).

---

## 2. Path 0 decomposition — two-track inventory

Per-track verdicts (TTPO rule: one verdict per track, never one per paper).

| # | Component | Track | Modelless-extractable? | Disposition |
|---|---|---|---|---|
| C1 | Two-level skill IR (composite → ordered subskills; subskill = one chunk) | Modelless | YES — pure data structure + expansion fn | Recorded; nearest shape = `riir-games` `DeliberationPlan` waypoints + `riir-agents` task DAG + `katgpt-pruners` `skill_lifecycle` memory |
| C2 | **Boundary-sourcing law**: open-loop stretches need exogenous boundaries mined from successful traces, never self-generated | Modelless | YES — design law | **Recorded; partially covered** — see §3 signal-diff |
| C3 | Failure taxonomy: len→1 collapse / len→∞ over-commit as boundary-absence symptoms | Modelless | YES — diagnostics | **Covered by shipped guards** — §3 |
| C4 | Skill-library lifecycle (induct → filter → dedup → UCB retrieval → prune-at-zero) | Modelless | YES — all programmatic; LLM only as offline annotator | **Covered** — `skill_lifecycle` ships (PrunerMemory + test gates, katgpt-pruners); Research 172 ITSE; Research 105 SkillOpt |
| C5 | Boundary-gated open-loop execution with adaptive chunk length | Modelless | YES | **No live consumer** — see §4 consumer analysis |
| C6 | Chunk-level TTS search (search-unit law) | Modelless | YES | **Covered + validated** — tree-verify |
| C7 | Hybrid on-/off-policy (PPO-clip + SIL) | Training | NO | → riir-train issue (512) |
| C8 | Chunk-aware two-level advantages (anchor-state grouping + within-chunk broadcast) | Training | NO | → riir-train issue (512); the strongest deposit — no multi-turn RL surface currently runs it |
| C9 | ρ_prim mixing schedule | Training | NO | → riir-train issue (512), deferrable |
| C10 | Expand&Relabel as data engine | Training | NO | → riir-train issue (512) — the conditional L4 re-attempt mechanism |

---

## 3. Coverage / signal-diff (§3.6 discipline)

### 3.1 The "when to decide" class ships — R363's table, re-verified

Research 363 (arXiv:2606.26463, real-time RL planning budgets) already documented six modelless implementations of state-dependent compute budgeting: **R149/P304 GainCostLoopHalter** (`halt when Gain(r) < Cost(r)·τ`, per-NPC), **R218 breakeven complexity router**, **R350 density-aware scheduling**, **P231 PathwayTracker** (85% thinking-budget savings, default-on), **P263 cumprodsum freshness gate** (default-on), **P194/R283 adaptive CoT bandit** (+P212 collapse-aware). SPACE's decision-scheduling extraction is a *seventh vocabulary for this class*, not a new class.

### 3.2 Signal-diff: what SPACE's boundary source is vs what ships

| Shipped mechanism | Signal it consumes | SPACE boundary signal | Diff verdict |
|---|---|---|---|
| `ConvergenceCadence` (cadence_gate, shipped 2026-09-04, opt-in) | **Self-telemetry**: windowed update-magnitude trajectory → Settled \| Churning | Mined task structure from successful traces | Different signal (reactive telemetry vs proactive structure) |
| `PlannerConfig::replan_ticks = 40` (riir-games hero planner) | Fixed interval | Mined boundaries | Different (fixed cadence, no structure) |
| `DeliberationPlan` + waypoints (SwarmDeliberationSystem) | Route structure from one-shot search; boundary = waypoint arrival / stuck trigger | Mined skill subskill boundaries | Same behavior class (open-loop between decisions), different boundary source, no library/mining/UCB/prune |
| Tri-gate (R281, per-tick speak/silent/delegate) | Latent salience (zone-attention + curiosity) | Mined task boundaries | Different granularity: per-tick emission gating vs chunk amortization |
| `skill_lifecycle` (katgpt-pruners, shipped) | Per-pruner arm experiences + test gates | Trajectory-mined skill *boundaries* | Skill memory ships; boundary-bearing two-level skills used for decision scheduling do not |

**The gap is real but consumer-less.** Nothing mines successful trajectories for decision-boundary structure. But §4 shows no surface pays for it.

### 3.3 The failure taxonomy is already implemented (independently)

The shipped `deliberation_cadence` upgrade (katgpt-rs Issue 720 T3 → katgpt-core `cadence_gate` → riir-games swarm → mmorpg consumer, gates G1–G4 + non-vacuity PASS at 1000 NPCs) carries **both** SPACE failure-mode guards, discovered independently:

| SPACE failure mode | Shipped guard |
|---|---|
| **Over-commit** (open-loop execution drifts past the point where re-planning was needed; failure-detection latency grows) | **Settled early-commit** — steady escapers skip further deliberation/search *and* the mechanism is cadence-gated so a changed world re-triggers via the heading-churn signal |
| **Collapse** (decision overhead per action dominates; churning) | **Windowed heading-churn trigger** — catches gapped oscillators/trembling that the raw consecutive-flee counter missed, firing deliberation *early* |

The paper's contribution here is the **causal story** (both modes are boundary-supervision absence) — useful doctrine, not a missing mitigation. Over-commitment's risk (stale boundaries under distribution shift) is exactly the two-brain fog-of-war divergence the stack already treats as first-class.

### 3.4 Chunk-level search is already exploited — and validated

- **Tree-verify (Issues 717/721, Benches 694/697):** tree-structured draft candidates (= multi-token chunks) measured tree acceptance **0.8785 vs chain 0.30** — chunk-level candidates are dramatically better search/verification units on our stack, same direction as the paper's Table 4 (+8.3 vs +4.2 SR per BoN).
- The Issue 721 structural negative was **compute-side** (verify cost linear in T under current kernels on both Metal and Vulkan), not quality-side — the paper's finding confirms the quality axis was never the problem, sharpening the recorded reopen conditions (weight-stationary T-amortized GEMM first).

---

## 4. Consumer analysis (mandatory step-4 reframe)

**Game runtime (priority #1):** The decisions SPACE amortizes are LLM rounds (~seconds each). Our NPC decisions are µs-scale modelless math — motivation argmax, goal FSM, planner — with measured 20 Hz headroom (G2 gates pass at ~2 ms/tick, 1000 NPCs). The expensive cognition that exists (stuck-NPC deliberation, 8×10 search) is **already cadence-gated** by the shipped upgrade. A mined-boundary trigger would replace a cheap trigger with a cheaper-in-principle one on a surface where the trigger cost is not the bottleneck. **No measured pain → no pull.** Selling-point comparison: R149's "10,000 NPCs each at individually-optimal reasoning depth from a single frozen artifact" already exceeds anything SPACE's runtime half offers.

**riir-clippy healer (priority #2):** The expensive decision (L4, ~191 s/fix) is already behind a three-condition reachability contract and decline-default; modelless coverage is ~98% on score-bench. Rule-shaped chunking already ships (per-file fixpoint batches with one compile-verify per batch). The residual 2% miss territory is precisely where mined routines do not apply. Skill-mined fix *routines* (multi-step deterministic recipes from Certified-tier trajectories) are a conceivable coverage extension, but the miss set is dominated by cases needing semantic reasoning — no evidence routines extend coverage there.

**riir-agents:** modelless by design (BLAKE3 directions + sigmoid). No LLM rounds to amortize.

**Token-level:** tree-verify already does chunk-level search (§3.4).

**Conclusion:** the modelless decision-scheduling extraction has **zero live consumer pull**. The honest classification for the modelless track is **PASS** — with the boundary-source distinction recorded here so a future consumer (e.g., an LLM-planned agent tier, or a warm-tier planner surface) finds the design law and the diff against telemetry triggers already written.

---

## 5. Novelty gate (modelless extraction as the candidate)

1. **No prior art?** In-stack: NO for the class (six implementations, §3.1); the boundary-source variant does not ship but §4 shows that is not actionable. External: **arXiv:2509.03581 "Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents"** (UCL, 2025, v3 2026-02) formalizes dynamic when-to-plan decisions and trains them (SFT+RL); AdaPlanner (NeurIPS 2023) does closed-loop adaptive replanning; robotics has an adaptive-replanning-trigger literature; the options framework (Sutton 1999) is the ancient base for decide-at-boundaries/act-open-loop. "Learned when-to-decide" is published prior art as a class; "programmatic boundary-mined scheduling without training" is an unpublished narrow variant.
2. **New behavior class?** NO — open-loop-between-decisions behaviors already exist (DeliberationPlan, cadence gates, fixpoint batches). A different signal source, same behavior.
3. **Product selling point?** NO — shipped selling points (R149 crowd-scale depth control) are stronger; the paper's selling point presumes expensive decisions, which the hot path deliberately does not have.
4. **Force multiplier?** Would connect to ≥2 pillars in shape, but with no consumer it is a force-multiplier-shaped non-consumer.

→ **Not Super-GOAT, not GOAT, not Gain on the modelless track. PASS** (with PASS-Redirects, below).

---

## 6. Training track — the surviving actionable half

The advocates' merged assessment (both briefs on record in session):

- **No unconditional production consumer** for any training item; game cognition and clippy runtime are modelless by mandate.
- **One conditional, owner-gated lane:** the L4 fixer re-attempt. The banked failures (Plan 336: 0/60 EM, tied with frozen-backbone control; re-gate 0/60 DEGENERATE) are diagnosable as the paper's *naive-collapse signature* — advantage-free whole-trajectory loss on heterogeneous multi-edit sequences. SPACE's genuinely new mechanism changes the **data**, not just the loss: Expand&Relabel successful Certified-tier fix trajectories into chunk-level targets + advantage-clipped SIL + on-policy rollouts on the 9-reachable-miss set with compile-gate reward. Mechanically unblocked since the closed-loop LoRA-aware GPU forward landed (riir-ai `6bf51b592`). Est. 40–80 GPU-h at bonsai-27B QLoRA (gemma-2-2b proxy first). **Kill-criterion: same bar fail → the L4 training lane closes for good; CF-llama external arm remains the only quality path.**
- **One cheap enabling deposit:** chunk-aware two-level advantages (anchor-state grouping + within-chunk broadcast) as a grouped mode in `loss_grpo.rs`, gated on `remax_ppo`'s minatar env (M3-runnable, <5 GPU-h) with the paper's degenerate case (single-action episodes must reduce bit-identically to baseline GRPO) as the negative control.

→ **GAIN, filed as `riir-train/.issues/512`** (deposit + conditional re-attempt with pre-registered bar and kill-criterion). Not a plan: no unconditional consumer, owner-gated lane, two failures on record — an issue preserves the reopen material with unblock conditions, per the defer discipline.

---

## 7. MOAT gate per domain

| Domain | Verdict |
|---|---|
| `katgpt-rs` | Neutral — no new primitive; the class ships (R363 table). The boundary-source law is recorded doctrine for future consumers. |
| `riir-ai` | Neutral — deliberation_cadence already implements both failure-mode guards; the decision-scheduling variant has no consumer on µs-scale decisions. |
| `riir-chain` | N/A — no sync-boundary angle (chunk scheduling is local cognition; nothing new crosses sync). |
| `riir-neuron-db` | N/A — skill-library storage would reuse ShardIndex/MerkleFrozenEnvelope as-is (R172 already maps this). |
| `riir-train` | **Issue filed (512)** — recipe deposit + conditional L4 re-attempt. The paper's actual transferable asset lands here. |

---

## 8. PASS-Redirects + prior art

- arXiv:2509.03581 "Learning When to Plan: Efficiently Allocating Test-Time Compute for LLM Agents" — learned (SFT+RL) dynamic when-to-plan; the training-side prior art for the class; our coverage is modelless and ships.
- AdaPlanner (arXiv:2305.16653) — closed-loop feedback-driven plan refinement; refinement ≠ chunk amortization.
- Options framework (Sutton, Precup, Singh 1999) / semi-MDP — the ancient base for decide-at-boundaries/act-open-loop; SPACE itself cites it.
- Q-chunking (NeurIPS 2025), SEAR (arXiv:2603.01891), ACT (Zhao 2023) — action-chunking RL lineage the paper builds on (continuous-control / robotics settings).

**PASS-Redirects written into:** R363 (closest shipped-cousin note — the when-to-decide class), R172 (skill lifecycle), R281 (per-tick tri-gate).

---

## 9. Fusion subsection (mandatory even when unplanned)

The one combination worth naming for the future: **mined-boundary countdown (primary) → anchor-precondition interrupt (immediate) → churn/cadence verdict (fallback)** as a three-tier trigger ladder. This is the No-GD advocate's strongest composite: it composes the paper's boundary source with the shipped cadence gate as the fallback detector that catches distribution shift (the over-commitment risk under stale boundaries). If a surface ever appears where decision cost dominates (LLM-planned agent tier, warm-tier planner with real LLM calls), this ladder — not the raw mined boundary — is the shape to build, with the failure-detection-latency gate registered before any A/B (the paper's own taxonomy supplies the tripwire).

---

## TL;DR

**Modelless track: PASS.** SPACE's decision-scheduling insight ships in six+ implementations (R363); its two genuinely new elements — boundary source from mined task structure, and the collapse/over-commit failure taxonomy — are recorded doctrine and independent validations of shipped design (`deliberation_cadence` carries both guards; tree-verify already exploits chunk-level search, tree acceptance 0.8785 vs chain 0.30). No live consumer: hot-path decisions are µs modelless, the expensive surfaces are already gated or decline-defaulted. **Training track: GAIN via `riir-train/.issues/512`** — chunk-aware credit-assignment deposit (minatar gate, <5 GPU-h) + conditional owner-gated L4 trajectory-expansion re-attempt (40–80 GPU-h, pre-registered 9-case bar, kill-criterion stated: same-bar fail closes the L4 training lane permanently).
