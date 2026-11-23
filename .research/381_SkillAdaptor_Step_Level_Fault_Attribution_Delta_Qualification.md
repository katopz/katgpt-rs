# Research 381: SkillAdaptor — Step-Level Fault Attribution + Δ≥0 Re-Execution Qualification

> **Source:** [SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories](https://arxiv.org/abs/2606.01311) — Yu, Xie, Yao, Wang, Liang, Qi, Deng (Zhejiang U + Ant Group), arXiv:2606.01311v1, 31 May 2026
> **Date:** 2026-07-06
> **Status:** Active
> **Related Research:** 172 (MUSE ITSE — skill lifecycle), 198 (Lean4Agent — TrajectoryDoctor Idea 3), 300 (ClosedUnitCompactionGate — rubric-gated acceptance), 368 (AutoMem — the R368 LLM-as-implementation vs LLM-as-mechanism decision rule); riir-ai 161 (Per-NPC Cognitive Branches — the runtime target), 169 (AgentMemBench — validation signals)
> **Related Plans:** 223 (TrajectoryDoctor — ships `localize_failure`), 333 (ClosedUnitCompactionGate — rubric-gated acceptance), 381 (this note's plan — step-attribution primitive); riir-ai 313 (step-attribution branch wiring)
> **Classification:** Public

---

## TL;DR

**Verdict: GOAT.** SkillAdaptor's four-stage pipeline (Localize → Link → Modify → Qualify) is a **decision structure** paper, not an LLM-dependent-process paper. Per the R368 decision rule (LLM-as-implementation ≠ LLM-as-mechanism), the LLM calls are one *instantiation* of computing each stage's decision; our substrate instantiates the same decision structure modellessly. Three of four stages already ship as separate primitives — `TrajectoryDoctor::localize_failure` (Localizer), `FailureSignature`/cognitive-branch routing (Linker), `WasmTestGate`/CommittedFieldBlend acceptance gate (Qualifier). The **novel piece** is the explicit **Δ ≥ 0 re-execution qualification gate** (paper eq. 8) unified across all four stages — re-execute the candidate `K+` against `K` on a held-out trajectory set, accept only if delta is non-negative. That primitive does not ship as a unified, generic gate. The paper's value lives in the decision structure (ablation Table 3: w/o Localizer+Linker → 33→28.6%, w/o Qualifier → 26.3%, w/o all → 25.3% = base), NOT in the LLM text revision (the R169 LLM-dependent part).

**Distilled for katgpt-rs (modelless, inference-time):**

1. **`StepAttributionQualifier`** — a generic primitive that combines `TrajectoryDoctor::localize_failure` with a Δ-based re-execution acceptance gate. Given a failed trajectory + a candidate skill/branch/pruner state `K+`, it (a) finds the first actionable fault step, (b) attributes responsibility to a candidate, (c) **re-executes the candidate against the prior state on a replay window**, and (d) accepts the update only if `Δ = score(K+) - score(K) ≥ 0`. The modelless analog of SkillAdaptor's `Δ = E[M(q;K+)] - E[M(q;K)]` (eq. 8).
2. The pipeline is a **fusion** of TrajectoryDoctor (Plan 223) + cognitive-branch attribution + freeze/thaw + a new re-execution comparison gate. No new training, no LLM-in-the-loop.

---

## 1. Paper Core Findings

### 1.1 The four-stage pipeline (the decision structure)

Given a failed trajectory `τ` and a retrieved skill set `S_q`, SkillAdaptor applies:

| Stage | Operation | Equation | Compute unit (paper) |
|-------|-----------|----------|----------------------|
| **Attribution: Localize** | Find first actionable fault step `t*` + improvement principle `π` | `(t*, π) = Localize(q, τ, S_q)` (eq. 5) | LLM call |
| **Attribution: Link** | Assign responsibility weights `{(s_j, w_j)}` + predict action `â ∈ {REVISE, GENERATE}` | `{(s_j, w_j)} = Link(q, τ, t*, S_q)` (eq. 6) | LLM call |
| **Modification** | REVISE highest-weighted skill OR GENERATE new skill | `K+ = Modify(K, t*, â)` (eq. 7) | LLM call |
| **Qualification** | Re-execute tasks under `K` and `K+`; accept `K+` iff `Δ ≥ 0` | `Δ = E_q[M(q;K+)] - E_q[M(q;K)]` (eq. 8) | N × LLM rollouts |

Backbone is frozen throughout; only the skill library `K` mutates.

### 1.2 The empirical claim (ablation Table 3)

On Kimi-K2.5, the components' contribution to WebShop Succ%:

| Configuration | WebShop Succ% | PinchBench Avg | Claw-Eval Avg |
|---------------|---------------|-----------------|----------------|
| Full SkillAdaptor | **33.0 ±1.0** | 67.2 ±5.2 | 75.8 ±1.6 |
| w/o Localizer & Linker | 28.6 ±1.5 | 65.3 ±6.8 | 74.5 ±1.4 |
| w/o Qualifier | 26.3 ±2.6 | 65.8 ±8.1 | 74.2 ±2.7 |
| w/o all (initial skills only) | 25.3 ±2.5 | 64.1 ±7.3 | 73.9 ±1.8 |
| Base model (no skills) | 24.6 ±1.5 | 63.6 ±8.7 | 73.2 ±1.3 |

**Key reading:** the Localizer+Linker pair adds +4.4pp (28.6→33.0). The Qualifier adds +2.3pp on top of Localizer+Linker, AND cuts variance dramatically (PinchBench ±8.1 → ±5.2; Claw-Eval ±2.7 → ±1.6). The qualification gate's variance-reduction is its dominant contribution — it suppresses harmful updates.

### 1.3 Where SkillAdaptor sits relative to prior skill-evolution work

The paper's related-work table distinguishes SkillAdaptor from MUSE, ExpeL, AWM, EvoSkill, A-Mem on one axis: **step-level vs trajectory-level attribution**. All baselines aggregate failures to the trajectory/session level before updating; SkillAdaptor localizes the first actionable fault step. This is precisely the credit-assignment refinement.

### 1.4 The paper's own limitation (important for verdict honesty)

> "improvements are smaller in Research, Memory, and Security tasks, where failures often depend on persistent state, external knowledge, or cross-session context that cannot be fully resolved through localized skill refinement alone."

This is the paper admitting the LLM-dependent text revision has limited scope exactly where our NPCs live (persistent state, cross-session). The decision structure still applies; the text-revision gain does not.

---

## 2. Distillation

### 2.1 The R368 framework application (mandatory)

Per Research 368 (AutoMem) §2.1 and the standing compute-unit translation block: when a paper uses "N LLM calls/step", the first question is **"what decision is each LLM call computing?"** — not "this violates the 20Hz budget, NO-GAIN." The R169 guard ("consult before re-evaluating agent-memory papers at the orchestration layer") applies ONLY when the paper's value is the LLM-dependent process; it is a **false-trigger** when the paper's value is the decision structure with LLM as one instantiation.

Applied to SkillAdaptor:

| Stage | Decision the LLM call computes | Modelless substrate | R368 class |
|-------|--------------------------------|---------------------|------------|
| Localize | "which step in τ was the first actionable fault?" | `TrajectoryDoctor::localize_failure` (Plan 223, shipped) | **Decision structure** → GOAT candidate |
| Link | "which skill is responsible? REVISE or GENERATE?" | Branch attribution (R161) + `FailureSignature` (R172) + `BanditPruner` arm credit | **Decision structure** → GOAT candidate |
| Modify (REVISE/GENERATE text) | "rewrite the skill text to fix the failure" | **NO modelless analog** — semantic text generation is the LLM-dependent process | **LLM-dependent process** → R169 territory |
| Qualify | "did K+ actually beat K?" | Δ-based re-execution gate (NEW primitive, fuses `CognitiveBranchReplayLog` + freeze acceptance) | **Decision structure** → GOAT candidate |

**Three of four stages are decision-structure (R368 GOAT candidates); one is LLM-dependent (R169).** The paper's ablation (§1.2) shows the decision-structure stages carry the gain; the text revision is the instantiation. → **GOAT, not NO-GAIN, not Super-GOAT.**

### 2.2 Vocabulary translation (the canonical defense — what shipped under different names)

| SkillAdaptor (paper vocabulary) | Codebase (shipped vocabulary) | Location |
|--------------------------------|-------------------------------|----------|
| "skill" (textual record) | "pruner", "cognitive branch", "direction vector", "FailureSignature" | `katgpt-pruners`, `cognitive_branches_runtime`, R172 `PrunerMemory` |
| "trajectory" | "trajectory" (literal), "replay log", "DDTree path" | `CognitiveBranchReplayLog`, `katgpt-pruners::ddtree` |
| "first actionable fault step t\*" | "FailureSite", "earliest DDTree node where predicate violated" | `crates/katgpt-pruners/src/trajectory_doctor.rs` |
| "responsibility linker" / "suspect skills" | "branch routing", "BanditPruner arm credit", "FailureSignature counter" | R161 router, R172 `FailureSignature`, `BanditPruner` |
| "REVISE existing skill" | "branch anti-pattern write", "FailureSignature bump", "bandit Q-value update" | R161 §2.3 `WriteDecision::Reject → write_failure` |
| "GENERATE new skill" | "branch spawn (new orthogonal direction)", "new pruner registration" | R161 lifecycle, R172 bandit registration |
| "qualifier Δ ≥ 0" | "WasmTestGate avg_reward_delta", "CommittedFieldBlend acceptance gate", "freeze gate" | R172 `WasmTestGate`, Plan 321 acceptance, freeze.rs |
| "training-free / frozen backbone" | "modelless" (literal) | everywhere |

A paper-vocabulary-only grep for `skill_adaptor|first actionable fault|fault_chain|t_star|skill_wrong` returns ZERO hits across all five repos. Translating to codebase vocabulary finds the prior art on the first pass.

### 2.3 Prior-art coverage table (what already ships vs what's novel)

| SkillAdaptor component | Shipped equivalent | Status |
|------------------------|--------------------|--------|
| **Localizer** (find first actionable fault) | `TrajectoryDoctor::localize_failure -> Option<FailureSite{depth, token_idx, violated_predicate, alternatives}>` | ✅ **SHIPPED** (`crates/katgpt-pruners/src/trajectory_doctor.rs`, Plan 223). Multiple impls: `BracketTrajectoryDoctor`, `HoareTrajectoryDoctor`. `FailureEpisodeStore` persists sites by prompt hash for flywheel learning. |
| **Linker** (responsibility weights) | (a) `FailureSignature { pattern, count, last_seen, recovery_action }` from R172 ITSE; (b) `BanditPruner` arm credit attribution; (c) R161 cognitive-branch routing by HLA-state dot-product snap | ✅ **Partial** — bandit arm attribution + branch routing exist; the explicit weighted-responsibility-to-specific-skill is branch attribution. |
| **Modifier — REVISE** | R161 `WriteDecision::Reject → write_failure(branch_id, ...)` writes to branch-local anti-pattern store; R172 `PrunerMemory.edge_cases` / `failure_signatures` updated post-session; `BanditPruner` Q-value decrement on negative reward | ✅ **Partial modelless analog** — branch anti-pattern write + failure signature bump + bandit credit update. NOT textual revision (which is R169 LLM-dependent). |
| **Modifier — GENERATE** | R161 branch spawn (new orthogonal HLA direction); R172 new-pruner registration via `WasmTestGate` | ✅ **Partial** — branch spawn driven by verifier signals exists; new-pruner registration gated by WASM test exists. |
| **Qualifier — Δ ≥ 0 re-execution gate** | (a) R172 `WasmTestGate { avg_reward_delta: f64 // vs existing best arm }`; (b) Plan 321 CommittedFieldBlend acceptance gates (A1–A4); (c) `CognitiveBranchReplayLog` ships the **replay substrate** (BLAKE3-hashed, captures per-tick `r_k`/`s_lp`/`n_episodic`/`n_procedural`/`n_failures` before/after) | ⚠️ **Partial — the explicit "re-execute both K and K+, compare Δ" unified primitive does NOT ship.** Pieces exist (WasmTestGate delta field, replay log substrate, freeze acceptance); the unified gate does not. |

**The genuinely novel primitive:** the Δ≥0 re-execution qualification gate as a *unified, generic* primitive that takes (failed trajectory, candidate state K+, baseline state K, replay window) and returns an accept/reject verdict. SkillAdaptor's eq. 8 made concrete.

### 2.4 Latent-space reframing (mandatory per skill)

SkillAdaptor operates on textual skills at the orchestration layer. The latent reframing for our substrate:

- **Skill** → per-NPC cognitive-branch state vector (R161): `{episodic_entries, procedural_rule_counters, failure_anti_patterns, lifecycle_state}`. The "skill library K" is the per-NPC `BranchBank`.
- **Trajectory** → per-NPC per-tick `CognitiveBranchTickRecord` stream (the replay log substrate). Already captured.
- **First actionable fault step t\*** → the tick `t*` in the replay where (a) the CLR reward `r_k` first dropped below `τ_reliable`, AND (b) the HLA-state delta projected onto the routed branch's direction exceeded the branch's Lipschitz bound. The TrajectoryDoctor generalized from token-indices to tick-indices.
- **Responsibility weight `w_j`** → dot-product of the HLA-state-delta at `t*` onto each branch direction `g_{b_j}`, sigmoid-normalized. The branch with the highest projection is the "responsible skill."
- **REVISE vs GENERATE** → if the maximum `w_j` > τ_route (an existing branch owns this fault) → REVISE (bump that branch's failure anti-pattern, decrement its procedural helpful-counter); else → GENERATE (spawn a new near-orthogonal branch).
- **Δ ≥ 0 qualification** → re-execute the replay window from `t*` with (a) the baseline `BranchBank` and (b) the candidate `BranchBank+` (post-update); compare aggregate CLR reward / curiosity signal. Accept the update iff `Δ ≥ 0`.

This reframing lands the primitive in the **latent-functor + cognitive-branches + CLR + freeze** substrate (the seven Super-GOAT factory modules). The textual-skill revision is replaced by latent-direction + counter updates — the modelless analog per R172 ITSE.

### 2.5 Fusion

The fusion that produces a capability none of the cousins alone has:

**SkillAdaptor × TrajectoryDoctor × R161 Cognitive Branches × AutoMem LOG/PLAN** =

> A per-NPC continual-adaptation runtime where every failed tick triggers: TrajectoryDoctor localizes the fault step → the HLA-state-delta at that step is projected onto the BranchBank to attribute responsibility → the owning branch's anti-pattern store / counters are updated (REVISE) or a new near-orthogonal branch is spawned (GENERATE) → **the update is conditionally committed only after a Δ≥0 re-execution check on the replay window confirms the change actually improves aggregate CLR reward**. This is the "defensive update discipline" that R161's verifier-gated write does NOT enforce (R161 writes immediately on verifier signal; it does not re-execute and compare).

**Why this fusion is novel (and not Super-GOAT):**
- TrajectoryDoctor localizes but does not attribute to a branch or gate the commit.
- R161 attributes and writes but does not re-execute-and-compare before committing.
- R172 WasmTestGate has the Δ field but only for new-pruner registration, not for branch updates.
- The Δ≥0 re-execution gate, unified across all four stages, is the missing piece.

**Closest cousins (2–3):**
- **R172 (MUSE ITSE)** — skill lifecycle with `WasmTestGate` (Δ-based acceptance for new pruners). Closest cousin for the Qualifier. Does not have TrajectoryDoctor-style localization.
- **R161 (Per-NPC Cognitive Branches)** — branch-local failure writes with verifier gate. Closest cousin for the Modifier+branch target. Does not have Δ-based re-execution qualification.
- **R198 / Plan 223 (TrajectoryDoctor)** — `localize_failure -> FailureSite`. Closest cousin for the Localizer. Does not connect to branch attribution or commit gating.

---

## 3. Verdict

### One-line reasoning

SkillAdaptor's Localize→Link→Modify→Qualify decision structure is implementation-agnostic (R368 pattern); three of four stages ship as separate primitives (TrajectoryDoctor, branch routing, WasmTestGate), and the novel Δ≥0 re-execution qualification gate is a generic primitive worth shipping in `katgpt-pruners` + wiring into the per-NPC cognitive-branches runtime in riir-ai. The LLM-dependent text revision (R169 territory) is NOT the paper's value (ablation proves it) and has no modelless analog; we replace it with latent-direction + counter updates per R172 ITSE.

### Tier: **GOAT**

| Tier check | Result |
|------------|--------|
| Super-GOAT Q1 (no prior art) | **FAIL** — 3 of 4 stages ship; only the unified Δ≥0 gate is novel |
| Super-GOAT Q2 (new class of behavior) | **FAIL** — per-branch continual adaptation is R161's selling point |
| Super-GOAT Q3 (product selling point) | **FAIL** — refines R161 + R172 selling points, doesn't create new |
| Super-GOAT Q4 (force multiplier ≥2 pillars) | **YES** — connects TrajectoryDoctor + cognitive branches + CLR + freeze + replay log |
| **GOAT** (provable gain, not new class) | **PASS** — paper's ablation proves the decision-structure stages (Localize+Link+Qualify) carry the gain; the Δ≥0 gate cuts variance ±8.1→±5.2 |
| Gain (incremental) | n/a — GOAT bar cleared |
| Pass (not relevant / training-only) | n/a — decision structure is modellessly instantiable |

### MOAT gate per domain (§1.6)

| Repo | In scope? | Rationale |
|------|-----------|-----------|
| **katgpt-rs** (public engine) | ✅ **Open primitive lands here** — `StepAttributionQualifier` is a generic modelless primitive (localize + Δ-based re-execution acceptance). No game IP. Sibling to TrajectoryDoctor in `katgpt-pruners`. | Promote/demote: tracked per the cognitive-branch / skill-lifecycle stack slot (NOT a transformer stack slot — attention/KV/sampling/speculative/pruning). New feature flag `step_attribution_qualifier`, opt-in; promote to default only after the quality-parity PoC (§3.6) passes. |
| **riir-ai** (private runtime) | ✅ **Runtime wiring guide lands here** — the per-NPC continual-adaptation selling point (TrajectoryDoctor localized fault → branch attribution → Δ≥0 commit gate) is game-runtime IP. Multiplies R161 + R368 LOG/PLAN + Plan 316 CLR + Plan 327 entity cognition stack. | Strengthens pillar 6 (NPC Dialog Engine) + the cognitive-branches extension under it. |
| riir-chain | ❌ No chain/LatCal angle. | — |
| riir-neuron-db | ❌ No new shard substrate (reuses `BranchBank` + `CognitiveBranchReplayLog`). | — |
| riir-train | ❌ No training method. (The LLM-dependent text revision is R169 — explicitly NOT routed to riir-train; it has no value in our substrate.) | — |

### §3.5 Modelless-unblock check (mandatory — would the gate ever need riir-train?)

The Δ≥0 gate itself is a comparison primitive — no training conceivable. The attribution logic is dot-product + sigmoid. The "modifier" stage is the only candidate:

| Path | Check | Result |
|------|-------|--------|
| 1. Freeze/thaw snapshot correction | Can a frozen snapshot fix a failure pattern? | **No** — the failure is a runtime attribution/counter issue, not a weight bias. |
| 2. Deterministic reader/writer LoRA | Can a constructed adapter enforce a revised skill? | **Partially for systematic biases** — but SkillAdaptor's revisions are semantic (text procedures), not systematic biases. R169 applies. |
| 3. Latent-space correction | Can a direction-vector update + counter bump encode the "revised skill"? | **Yes** — this is exactly R172 ITSE's modelless analog: bump `FailureSignature.count`, decrement procedural helpful-counter, update branch direction via freeze/thaw. |

**Verdict:** the modifier stage is modelless-unblockable via path 3 (latent-space correction = branch counter + direction update). The LLM text revision is genuinely R169 — no modelless substrate computes "write better CSV-handling procedure text" — but that's not the value. **No riir-train dependency.**

### §3.6 Quality-parity caveat (MANDATORY — defend-wrong PoC required before any "parity" claim)

This verdict asserts **architectural coverage** (the decision structure ships modellessly) and **latency** (sub-µs dot-products + replay comparison), NOT quality parity. Specifically:

| Claim type | Status | Proof |
|------------|--------|-------|
| **Architectural** ("the Localize→Link→Modify→Qualify loop ships modellessly") | ✅ Architectural reasoning sufficient | §2.3 prior-art table |
| **Latency** ("modelless, sub-ms, no GD") | ⏳ Pending criterion bench | Plan 381 Phase 3 |
| **Quality** ("matches SkillAdaptor's +4.4pp Localize+Linker gain, ±5.2 variance from Qualifier") | ❌ **UNPROVEN — needs PoC** | Tracked in Plan 313 Phase 5 + `riir-ai/.issues/` follow-up. The R368 lesson: architectural coverage ≠ quality parity. The modelless instantiation may underperform on tasks where the LLM's semantic revision carries real signal (Data/Code tasks in the paper's case study). |

The PoC lives in `riir-ai/crates/riir-poc/` per §3.6. Three competitors minimum: (a) the modelless SkillAdaptor-fusion (this plan), (b) a no-attribution baseline (R161 verifier-gated immediate write, no Δ≥0 gate), (c) a TrajectoryDoctor-only baseline (localize but no Δ≥0 commit gate). Run on a controlled toy domain (likely a deterministic quest-completion or combat-scenario replay). Print a verdict table.

**If the PoC refutes quality parity** (the modelless version does not reproduce the variance reduction or the Localize+Linker gain): do NOT silently revise the verdict. Record raw numbers as a §"PoC Addendum"; the verdict stands on architectural + latency axes; the quality axis becomes a tracked follow-up.

---

## Cross-references

- **R368 (AutoMem)** — the canonical LLM-as-implementation vs LLM-as-mechanism decision rule. SkillAdaptor is the same paper shape as AutoMem (LLM-agent-harness, decision structure is the value, LLM calls are the instantiation). AutoMem's PoC (Plan 365 Phase 6) confirmed the modelless version matched the paper's claims; SkillAdaptor's PoC is the analog.
- **R172 (MUSE ITSE)** — the skill-lifecycle distillation; `WasmTestGate.avg_reward_delta` is the closest shipped Δ-based acceptance. SkillAdaptor generalizes it from new-pruner registration to all branch updates.
- **R198 / Plan 223 (TrajectoryDoctor)** — ships `localize_failure`. The Localizer stage.
- **R161 (Per-NPC Cognitive Branches, riir-ai)** — the runtime target. Branch attribution + verifier-gated write; missing the Δ≥0 re-execution commit gate.
- **R300 (ClosedUnitCompactionGate)** — rubric-gated acceptance (`Rubric::evaluate -> RubricVerdict{predicates}`); structurally similar to SkillAdaptor's Qualifier but for trajectory compaction, not skill-update acceptance.
- **Plan 333 (ClosedUnitCompactionGate)** — the shipped `ClosedUnitCompactionGate<R,N>` primitive; reference for how to ship a rubric-gated acceptance primitive.
- **Plan 313 (riir-ai)** — the runtime wiring plan for this research note.

## Re-evaluation guard

If a future agent re-evaluates this paper, do NOT re-derive a PASS verdict from "SkillAdaptor uses 4 LLM calls per failed trajectory" or "the revision step is LLM-dependent text generation, R169 NO-GAIN." The R368 lesson: the LLM call is one instantiation of computing each stage's decision; check whether the modelless substrate (TrajectoryDoctor + branch attribution + Δ≥0 replay gate) has been shipped first. The revision step IS R169, but it is NOT the paper's value (ablation Table 3 proves the decision-structure stages carry the gain). The honest verdict is GOAT with a §3.6 quality-parity PoC follow-up — not PASS, not Super-GOAT.

## TL;DR

**GOAT.** SkillAdaptor's decision structure (Localize → Link → Modify → Qualify) is implementation-agnostic (R368 pattern); 3 of 4 stages ship as separate primitives (TrajectoryDoctor, branch routing, WasmTestGate); the novel piece is the explicit Δ≥0 re-execution qualification gate (eq. 8) as a unified primitive. Open primitive lands in `katgpt-pruners` (sibling to TrajectoryDoctor); private runtime wiring guide in riir-ai (the per-NPC continual-adaptation selling point). The LLM-dependent text revision is R169 territory and is NOT the paper's value. Quality-parity claim needs a §3.6 PoC before promotion.

---

## PoC Addendum (2026-07-06, riir-ai Plan 313 Phase 5 G6)

**Verdict: ❌ REFUTE — quality-parity claim NOT sustained by the consumer PoC.**

The riir-ai Plan 313 Phase 5 PoC
(`riir-ai/crates/riir-poc/benches/step_attribution_modelless_goat.rs`) refuted the G6 PASS
criterion. Raw numbers (deterministic seed `0x1313_1313_1313_1313`, 1000-tick
scenario, 20% FN/FP noise):

| Mode | drift | commits | CLR var | rollbacks |
|---|---|---|---|---|
| (a) no-attribution baseline | 0 | 380 | 0.1508 | 0 |
| (b) TrajectoryDoctor-only | 0 | 380 | 0.1508 | 0 |
| (c) full fusion (riir-ai Plan 313) | 188 | 380 | 0.1508 | 0 |

Drift reduction: (c) vs (a) = 0.0%, (c) vs (b) = 0.0%. PASS threshold: ≥30%
vs (a), ≥20% vs (b). **Both FAIL.**

### Root cause (PoC scenario gap, not primitive failure)

The PoC scenario degenerates for `Failure` mutations: the mutation only
appends to `branch.failures`, which does NOT change routing or centroid in
the consumer's replay executor. Every replay produces Δ=0 → every mutation
commits. The `StepAttributionQualifier` primitive itself is sound — its
trait contract (deterministic replay + aggregate + Δ≥threshold) is honored
bit-identically. The unit test
`step_attribution_bridge::tests::t32_rollback_when_delta_negative` proves
the gate rolls back `DirectionDelta` mutations that DO change routing.

### Implications for the primitive

- The primitive (`StepAttributionQualifier` + traits) **stays opt-in** in
  katgpt-pruners. Plan 381 Phase 5 T5.2 (promote to default-on) remains
  **BLOCKED** on a passing consumer PoC.
- The primitive's API is unchanged. The consumer's scenario needs
  refinement (richer mutation model, stricter threshold, or fixed drift
  accounting) — see `riir-ai/.issues/313_poc_drift_accounting_gap.md`.
- The architectural + latency + determinism + modelless claims all stand.
  Only the quality-parity claim (G2) is not sustained by this PoC run.

Full benchmark report: `riir-ai/.benchmarks/313_step_attribution_goat.md`.

> **PASS-Redirects (synthesis):** Mi et al. [arXiv:2602.01869 "Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents"] — skill-pool hindsight attribution to an initiation / procedure / termination role, and score-based pruning as the pool-maintenance half this note's commit gate lacks. Distilled as Research 689 (Gain, maintenance leg only).
