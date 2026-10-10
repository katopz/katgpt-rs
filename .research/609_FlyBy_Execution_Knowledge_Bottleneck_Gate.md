# Research 609: FlyBy — Execution vs Knowledge Bottleneck Gate

> **Status:** Active — GOAT-tier fusion filed as Plan 621; Super-GOAT label deferred to measured G1 (never asserted)
> **Source:** "Knowing When Thinking Is Not Enough: Teaching Small Reasoning Models to Reason Beyond Their Parametric Knowledge" — Lee, Kang, Park, Yeo, Baek, Hwang (KAIST + DeepAuto.ai), [arXiv:2609.34327](https://arxiv.org/abs/2609.34327), 2026-09-28. Code: `github.com/tally0818/FlyBy` @ `9c4c2054dfed6d0a5c0859e97e6e267d67149a7b` (MIT, verl-tool based; distill-verified from a `.raw/` clone, removed after verdict)
> **Date:** 2026-10-08
> **Related Research:** 496 (SPADE — the `hint_regret` paired-VoI lineage), 493 (contrastive scope — decline wiring)
> **Related Plans:** katgpt-rs 621 (`state_probe` + `escalation_probe_gate`) · riir-train 448 (escalation-policy reproduction, SECONDARY) · riir-train Issue 619 (4090 long task) · riir-refine Issue 156 (state-level rescue mining)
> **Classification:** Public

---

## TL;DR

FlyBy's counterfactual interventions at intermediate reasoning states show that **self-refinement consolidates probability mass onto already-reachable solutions rather than making new ones reachable**; failures split into *execution bottlenecks* (reflection can recover) and *knowledge bottlenecks* (external information required). We distill the paper's **decision layer — not its trained policy —** into a modelless escalation gate: probe a decision state with a perturbation/candidate-pool ensemble (`V(s)` pass-fraction + Miller-Madow answer entropy), classify the bottleneck with a Wilson/LCB bound inequality, and let the class drive the existing ESC seams (`modelless_cap` / `EscalateSpec`) across cost tiers.

**Distilled for katgpt-rs (modelless, inference-time):** probe kernel + bottleneck bound + cost-aware utility, zero training, every input a plain count/histogram so the kernel stays reflex-free (public substrate, consumers downstream).

---

## 1. Paper Core Findings

- **Consolidation finding:** paired counterfactual interventions (EV-present vs EV-banned decoding; epistemic vs random vs oracle-information cues) across 8 models / 272K continuations: reflection recovers execution-like states but barely moves knowledge-like ones (on the N-sweep reclassified states: info gain **+0.242** vs EV **+0.026**).
- **State probe:** `V(s)` = fraction of N sampled continuations from state `s` reaching the correct answer; `H(s)` = answer entropy with **Miller-Madow** bias correction `(K−1)/(2N)`. Productive reasoning moves toward `(V, H) = (1, 0)`.
- **Operational split:** `V(s) > 0` = execution-like, `V(s) = 0` = knowledge-like at N=8. Robustness: 76 zero-success states → 28 flip positive by N=64, **but the reclassified states retain the intervention gap** — the class boundary is finite-sample-fragile, the *gap* is stable. (Calibration context for our N-sweep test; never a fixture assertion.)
- **sRM diagnosis:** small models express uncertainty frequently (EVs: *wait/hmm/maybe/…*) but convert it to progress rarely; utilization of external information *increases* with scale — help-seeking is a learned capability, not a prompting trick.
- **The FlyBy system:** SFT bootstrap (475 examples incl. **95 rescue trajectories** mined by Algorithm 1: `b0 = 0 ∧ bd ≥ 3` — zero of 4 plain continuations succeed, ≥3 of 4 tool-assisted succeed) + cost-aware GRPO (`r = I[correct]·(1 − λ·Ĉ_group)`, λ=0.1; cost penalized **only on success**, so failing cheaply is never rewarded); multi-depth query tool (3 backends, 128/512/1536-token budgets, escalating price); query contract (backend sees only the sub-question, 8-gram overlap filter, gold-answer redaction).
- **Results:** FlyBy-4B 45.96% pass@8 > Qwen3-14B 41.64% at 2.7× lower cost. At the states where the policy queries: querying **+5.30pp** vs forced thinking **−0.41pp**. Backend substitution (DeepSeek → GPT-5.6 Luna / HY3) retains most performance (43.21/43.96) — the policy learned *when/how much*, not backend identity.
- **Failure mode on record:** at λ=0.2, cost-centering assigns negative advantage to correct-but-expensive trajectories (suppression).

## 2. Distillation (the modelless primitive)

### 2.1 The critical adaptation — deterministic scorers need ensembles, not temperature

FlyBy samples N continuations from a *stochastic* LLM. Our decision engines are *deterministic*: reflex's corpus-is-the-model scorer gives N identical answers under temperature sampling, so `V(s)` degenerates to `{0,1}`. The modelless translation: **`V(s)` over an ensemble of the decision under input perturbation** — reflex `src/mc_ensemble.rs` (Issue 055 / Plan 008: seeded Bernoulli bucket-dropout over embeddings, per-request seed `BLAKE3(state ‖ prompt ‖ salt)`, sample 0 = the unperturbed served bytes) — or **over the drafter's candidate pool** (healer: pool pass-rate = the `V(s)` analog, `pool_depth_rescue_guard` territory). Honest naming: the perturbation ensemble measures *decision stability*, not literal reachability; the classifier discloses this.

### 2.2 Probe kernel (open primitive → katgpt-core)

`V̂ = k/N`; `Ĥ = H_emp + (K−1)/(2N)` (Miller-Madow — **absent workspace-wide until this note**); Wilson/LCB on `V̂` — **CONSUMES `hint_regret::gate::wilson_score_ci`**, no third copy. (Pre-existing duplication on record: a byte-identical twin lives in `speculative::qmc::BootstrapEstimate::wilson_ci` — folding the two is out of scope here, recorded as an observation.)

### 2.3 Bottleneck classifier

knowledge-like ⟺ `V_LCB(N, δ) < ε` — one inequality over the consumed bound. **Adaptive-N stopping** (keep sampling while the interval straddles ε, cap at N_max) handles the paper's own 76→28 reclassification fragility by construction; every flip is disclosed with its interval width.

### 2.4 Cost-aware utility

`U = I[predicted success]·(1 − λ·Ĉ)` as a *selection* utility over `{stay-local, escalate(d)}` — `gain_cost_halt.rs`'s scissors law generalized to escalation decisions, with FlyBy's asymmetry preserved: a failed escalation can never out-score a cheaper success. λ=0 byte-identical to the incumbent cap gate (G3).

### 2.5 Depth ladder

`EscalateSpec` additive field family (one-definition home: `riir-instinct/src/arsenal.rs`; rethink byte-pin lockstep). Shipped ESC chain already has 2 live depths (local → seat_rerank → hosted_expert) selected by *static manifest row*; the delta is *dynamic per-state depth selection*.

### 2.6 Rescue predicate (offline, → riir-refine Issue 156)

`b0 = 0 ∧ bd ≥ 3` → a state-level rescue exemplar — the `traj_store` Certified-tier shape (no-fix fails, fix verifies), and a local pre-filter for reflex's corpus-synthesis acceptance. **AUGMENTS the network-oracle agreement VETO, never replaces it**: the veto's cross-model agreement remains the authority; the local predicate can only narrow the candidate set (same-model self-consistency is weaker evidence than independent agreement).

**LANDED 2026-10-08** — refine `cf34f6c8` (substrate `src/rescue_mine.rs` + arm-H tallies + `--misses` rescue band + the G1 truth-table tests) and reflex `86fe38a` (OPT-IN `--synth-rescue-prefilter`, default OFF); measured yield honest-EMPTY locally, the FlyBy 95/800 comparison awaits the first ARMED arm-H run (owner-gated). riir-refine Issue 156 closed 2026-10-09 (`308be448`) — the durable record is riir-refine HISTORY.md.

### 2.7 The fusion (what none of the parts gives alone)

**probe (measured local-continuation ineffectiveness) ⊗ corpus-coverage distance (`CorpusDistanceGate`) → bottleneck class → tier selection.** Corpus distance makes "knowledge bottleneck" *measurable at serve time* (coverage); the probe makes "local continuation is futile" *measured* rather than self-reported. No prior art consumes this pair (§6); no shipped module does (§5).

## 3. Path 0 Inventory

| # | Component | Analog ships? | Extraction | Disposition |
|---|---|---|---|---|
| 1 | V(s)/H(s) state probe | Partial (MCTS game-tree values; `normalized_entropy` as drift gate; `mc_ensemble`/`perturbation_ensemble` ensembles). **Miller-Madow absent.** | YES — counting + two closed forms | Plan 621 Phase 1 |
| 2 | Bottleneck classifier | Partial (`hint_regret::triage` — offline content-level) | YES — LCB inequality + adaptive-N | Plan 621 Phase 2 |
| 3 | Paired counterfactual ΔV protocol | Partial (`ab_timing`, `hint_regret` CRN pairs) | YES — harness lane, offline | Plan 621 Phase 5 (reflex `--probe-delta-ab`) — **LANDED 2026-10-11** (reflex `2680159`, feature `probe_delta_ab`: the sign test over the discordant pairs + the adaptive-N disclosure on the corpus-ab frozen read; real first read massive_intent_en 11W/1L → MOVES VALUE, settled @ n=128) |
| 4 | EV-lexicon analysis | No | YES as **design law**: never gate escalation on verbalized confidence; gate on measured V | Recorded here; no code |
| 5 | Cost-aware reward shape | Partial (`llm_spend` hard caps stop spend, don't shape selection; bench_711 recorded the cost-blindness) | YES — selection utility (gain_cost_halt generalization) | Plan 621 Phase 3 + riir-train 448 |
| 6 | Multi-depth cost tiers | Partial (ESC chain 2 live depths, static row; `ThermalTier` stub) | YES — EscalateSpec field family | Plan 621 Phase 3; training arm in 448 |
| 7 | Rescue mining `b0=0 ∧ bd≥3` | Partial (`--misses`/`--frontier` rule-level; arm-H audit not miner) | YES — 8-row truth table | riir-refine Issue 156 — **T5.2 LANDED 2026-10-11** (refine `893f4df8`, feature `rescue_probe`: the Phase-2 classifier over the ACCUMULATED b0 leg — the attempt-count honesty beside the pinned truth table, augment-only; the arm-H escalation-ordering consumption arms with the first owner-gated ARMED run) |
| 8 | Query contract (isolation, overlap filter, redaction) | Partial (`code_egress` static snippet scoping ENFORCED) | Partial — dynamic composition needs the trained policy | Training track only (448) |
| 9 | SFT+GRPO recipe | Stabilizers SHIP (`loss_grpo.rs` DAPO 0.20/0.28 clip pair, `unbiased_advantage`) | — | riir-train Plan 448 (Path 0.5) |

## 4. Verdict

**Track A (modelless): GAIN — GOAT-tier fusion filed (Plan 621).** One-line: every ingredient is individually known (MCTS values, pass@k, semantic entropy, Miller-Madow textbook) and `hint_regret::triage` ships the same three-regime partition offline; the pinned composition — *measured local-continuation ineffectiveness ⊗ corpus-coverage distance → cost tiers, for a corpus-is-the-model consumer* — has no prior art (§6) and no shipped equivalent (§5). **The Super-GOAT label is deferred to measured G1** (probe-gated must beat `modelless_cap`-alone on the healer lane); behavior-class delta is a claim until then. No architectural guide filed at GOAT tier.

**Track C (model-based): GAIN → riir-train Plan 448 + Issue 619 (4090 long task). SECONDARY** per serving-envelope fit: the modelless gate runs in the hot path; the trained policy is an arming-signal source for rethink's ESC manifest *if it wins* the matched-cost comparison. GOAT gate: pass@1 at matched total cost vs the modelless gate + suppression regression.

**Track B (self-adaptive): folded into A** — the probe feeds live gates; no base-weight mutation anywhere.

**MOAT gate:** katgpt-rs — fundamental/base primitive on the escalation/abstention axis ✓ (public, reflex-free). riir-refine — consumer-first moat ✓ (rescue mining + gate lane). riir-reflex/instinct/rethink — consumers of the public kernel; no rethink-class content in any public file. riir-train — training-method moat ✓.

## 5. Closest Cousins + Signal-Diffs

| Cousin | What it consumes | Signal-diff vs the probe gate |
|---|---|---|
| `hint_regret::gate::triage` (katgpt-core) | Paired rollout returns under CRN, hint arm **materialized**; offline, content/curriculum granularity | Same three-regime partition (Frontier/Mastered/Intractable ≈ knowledge/execution/intractable); the probe is **online per-state without materializing the external arm** |
| `EscalationDispatch::gate` + `RollingRateLatch` (riir-refine) | The model's **self-reported score vs a static `modelless_cap`** + fleet-rate governance | The probe adds the **measured continuation** half; composition, not replacement — the cap gate is the G3 fallback |
| bench_711 margin gate (katgpt-core benches) | top1−top2 margin, single tier, defend-wrong bench only | Delta: bottleneck-classified signal + coverage fusion + tier ladder + serving wiring |
| `CorpusDistanceGate` + fused abstain + `noul` (katgpt-core/reflex) | Confidence + corpus distance, **pre-answer** | **COVERED** for abstention (arguably ahead of the paper); the probe adds the *escalation* half, not abstention |
| `ThermalTier` ladder (riir-refine kernel_opt) | — (documented stub; `route()` returns Plasma unconditionally) | The escalating-cost *shape* exists; the *trigger* is exactly what's missing |
| arm-H `fixer_regate` (riir-refine) | Paired modelless-vs-escalated per case → cap edit | Delta: **miner, not audit** — Issue 156 |

## 6. Prior Art (§4 searches, ~24 queries + internal greps)

Class is **crowded**: Zeng et al. arXiv:2604.17827 (trained SLM→LLM help-seeking; confirmed in FlyBy's own references) · Wang et al. ICML 2026 position (invoke external tools iff epistemically necessary — the *principle*, 4 months early) · ToolOrchestra (Su et al. 2511.21689) · SKR arXiv:2310.05002 (knowledge-boundary-gated retrieval, 2023) · "Cost-Saving LLM Cascades with Early Abstention" (~Feb 2025; ID not surfaced — cite venue+date) · Da Silva AAAI 2020 (uncertainty-triggered advice-seeking in RL — pre-LLM root). **The unqualified "training-free escalation gate" claim is DEAD** on C1/C3.

**The pinned composition survives**: no found prior art consumes the (measured local-continuation ineffectiveness ⊗ corpus-coverage distance) pair mapped to escalating cost tiers for a corpus-is-the-model scorer — every found gate consumes LLM-internal uncertainty (token entropy, consistency, verbalized confidence) or RL value estimates. FlyBy's consolidation analysis is novel *analysis*; its gate is trained and therefore supports, not kills, the modelless variant.

## 7. Public/Private Tier Check

`state_probe` is an open primitive in katgpt-core: **plain counts in, estimates out — zero reflex dependency** (reflex is private and downstream; BOUNDARY.md). reflex/refine appear as consumers only. No rethink-class content (serving arms, vessels, ESC enforcement) in any public file. riir-train plan/issue are private-tier training records.

## 8. GOAT Gates (full text in Plan 621)

- **G1** truth tables + **loud-zero baseline**: a gate that never fires on the two-class eval corpus FAILS; every run reports incumbent (cap-alone) vs probe-gated vs never-escalate floor → escalation rate, rescued fixes, spend.
- **G2** ≤10% probe-cost budget per gated decision, **fallback to the incumbent on breach** (measured, never asserted); bound math ≤1 µs.
- **G3** probe-off / λ=0 → incumbent decision byte-identical.
- **G4** zero-alloc hot path (fixed arrays, `alloc_delta` pattern).
- **Reclassification N-sweep test**: flips monotone (0→positive only), every flip disclosed with interval width; the paper's 28/76 is calibration context, never a fixture assertion.
- **UQ note:** the probe emits a binomial CI; G1 coverage arms assert its empirical coverage on fixtures. If any consumer treats the interval as calibrated UQ output, the Report-the-Floor extension binds at that consumer's re-gate.

## 9. Limitations (named, not hidden)

1. **Sampling-cost asymmetry** — V(s) costs N ensemble evaluations to gate one decision; affordable where the gated decision is expensive (healer fix dispatch, ESC escalations, harness lanes), **not** on reflex's ≤1 ms per-question hot path at N>1. First consumers scoped accordingly.
2. **Success-oracle dependency** — healer (parse/clippy-verify) and reflex suites (gold labels) have one by construction; open-ended domains do not.
3. **Ensemble resolution** — perturbation dropout measures decision stability, not literal policy reachability; the classifier's output is honest about this.
4. **Two-class corpus requirement** — any eval lane must contain both planted classes or G1 is vacuous.

## 10. Verdict Gate Record

`request_verdict` (claude_code), session `799fddbf-07e5-4245-a29b-bf6c1548fa79`: **Round 1 REVISE** — Wilson DRY (accepted: consume `hint_regret::gate::wilson_score_ci`, no third copy) · mc_ensemble citation (rebutted on disk evidence — `riir-reflex/src/mc_ensemble.rs` exists, reviewer retracted) · re-grade to GOAT-with-deferred-Super-GOAT (accepted) · gate hardening (accepted) · 4090 issue pairing (accepted) · veto relationship made explicit (accepted). **Round 2 AGREE** with two writing notes (kernel reflex-free; pin the rule not the ratio). **Round 3 final AGREE** (`final_round`).
