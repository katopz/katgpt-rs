# Research 689: Skill-Pro — Non-Parametric PPO Skill Pools (maintenance leg only)

> **Source:** Mi, Ma, Yang, Li, Wang, Zhang, Wang (CASIA / Bristol / PKU / UCL), "Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents", [arXiv:2602.01869](https://arxiv.org/abs/2602.01869) v3, 28 May 2026 (v1 2 Feb 2026). ICML 2026 spotlight. Code `github.com/Miracle1207/Skill-Pro` @ `3be7a9be2d4c024d132efe394d537404eba7e4c8`, MIT (clone removed after the read).
> **Date:** 2026-11-23
> **Status:** RECORD — **Gain (narrow)**. The loop is shipped from three sibling papers; one leg (advantage-over-baseline + confidence-penalised RETENTION) is un-shipped and has no host pool with an outcome feed today. No plan filed.
> **Related Research:** 105 (SkillOpt — the `ValidationGate` + rejected-edit buffer), 172 (MUSE/ITSE — `skill_lifecycle`, test-gated registration), 381 (SkillAdaptor — `StepAttributionQualifier`, the Δ≥threshold commit gate), 146 (GEPA reflective evolution), 244 (Self-Evolver faithfulness), 249 (DecentMem), 368 (AutoMem LOG/PLAN); riir-ai 313 (per-NPC Δ≥0 commit gate), 347 (Recuris — "0 deprecated", the pruning gap), 059 (GSE validators); riir-reflex 019 (landscape mention only, no arXiv ID pinned)
> **Related Plans:** 144 (SkillOpt text-space optimization — COMPLETE), 192 (ITSE — 23/23), 450 (step-attribution qualifier — DEFAULT-ON); riir-ai 214 (GSE `skill_lifecycle` — GOAT 13/13, default in 4 crates), 569 (branch wiring)
> **Cross-ref (riir-ai / riir-rethink / riir-instinct):** riir-ai Research 347 · riir-rethink `src/esc_voi.rs` (production outcome-blind) · riir-instinct `src/serve_edge.rs` (LRU is residency-only)
> **Classification:** Public
> **Verdict path:** `request_verdict` (claude-sub-agent-verdict) — see §Verdict provenance.

> **PASS-Redirects (synthesis, this note → its cousins):** Mi et al. [arXiv:2602.01869 "Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents"] — recorded here once for 105 / 172 / 381: Skill-Pro's loop (skill pool ⟨initiation, procedure, termination⟩ → hindsight attribution → accept-if-better gate → score-based pruning) is the same decision structure those three notes already distilled; its one unpublished component, a clipped importance-ratio gate on re-scored logged actions, is the component the field did not adopt (16 citing follow-ups read, all re-execution gates).

---

## TL;DR

**Verdict: Gain (narrow), not GOAT.** Skill-Pro evolves a pool of natural-language skills without weight updates. Every stage of that loop already ships in this stack, default-on and GOAT-proven, from three sibling papers distilled months ago. The paper's only component with no shipped cousin is the **maintenance leg**: a per-item score that is *advantage over a running baseline* with a confidence penalty, used to decide **retention**. No pool here keeps that score, and the pool that measurably needs pruning has no outcome feed to compute it from.

**Distilled for katgpt-rs (modelless, inference-time):**
1. A retention law `lcb(ω) = avg_gain − β·√(ln(1+t)/n)` with `avg_gain = Σ(outcome − baseline)/max(1, n)`, three-tier pruning (non-positive → redundant → ascending) and a newborn protection floor. Closed-form, zero-alloc, sigmoid-free.
2. A pool-health readout: the **reuse rate** (fraction of stored units invoked at least once), per pool and per transfer axis.
3. An explicit negative: the logprob importance-ratio gate is dominated by re-execution wherever re-execution is cheap, which it is here.

---

## 1. Paper core findings (numbers from the full text)

| Component | What the paper does | Ablation (Mastermind-v0, Table 3) |
|---|---|---|
| Skill | ω = ⟨initiation I, procedure π, termination β⟩, natural language | w/o Skill: 0.606 → 0.388 |
| Selection µ | cosine on I with an abstain threshold, or LLM top-k then pick by LCB | ε-greedy, 0.3 → 0.05 |
| Semantic gradient | an LLM attributes a trajectory's outcome to I / π / β; batch-aggregated | w/o SG: 0.530, gate pass rate −30% |
| PPO Gate | re-score logged actions under candidate vs old skill (ρ), clipped surrogate, accept iff J > 0, best-of-3 | w/o Gate: 0.453 |
| Score maintenance | gain = mean advantage over active steps; score = Σgain / max(1, N); prune ≤ 0, redundant, ascending | **w/o Score (FIFO): 0.439, reuse 0.925 → 0.131** |

Headline: reuse rates 0.83–0.93 vs ≤ 0.47 for RAG / ExpeL / A-MEM / AWM / G-Memory; ~816 stored tokens vs 10⁵–10⁸.

**Honest reading of the performance table.** The reuse and compression numbers are large and clean. The task-performance gains are not: ALFWorld OOD 0.909 ± 0.287 vs AWM 0.900 ± 0.042; Mastermind-v0 0.606 ± 0.234 vs G-Memory 0.577 ± 0.052; Extreme 0.333 vs G-Memory 0.356 (G-Memory wins); cross-agent on Gemma-3-4B 0.444 vs the no-memory State agent 0.414. The paper's contribution is pool *quality and compactness*, not a capability lift.

**What the code does that the paper text does not say** (read at the pinned sha):
- The shipped maintenance score is LCB-based, not the paper's plain mean: `lcb = mu − 0.2·√(ln(1+t)/n)` (`Skills/skill_pool.py:495-498`). Pruning stages: invalid → same-name duplicate → semantic duplicate (cosine ≥ 0.95 on the initiation embedding) → outdated `_vN` version → negative LCB → capacity trim ordered veterans > mid > newcomers (`:457-601`). Newborns are protected for `maturity < 3`.
- The gate's advantage is **trajectory-level**, spread uniformly: `traj_adv = reward − task_baseline; step_adv = traj_adv / T` (`Skills/skill_evolution.py:470-471`). The baseline is a per-task EMA, α = 0.1 (`run.py:42,124-125`).
- Gate log-probs come from the **evolver** model (default Llama-3.3-70B), not the decision LLM (`skill_evolution.py:526-544`, split at `run.py:79-83`), and are a per-token **mean** (`Skills/loss.py:99-101`).
- Acceptance is `J > 0.001` over best-of-3, ε = 0.2 (`skill_evolution.py:374-377, 495`).

---

## 2. Distillation

### 2.1 The pinned claim (§1.5 precondition)

*A trust-region-gated, non-parametric evolution of a condition-keyed pool of procedures, scored by advantage over a running baseline with confidence-penalised pruning, for pools that grow from experience — consuming realized outcomes per invocation, distinguished from the shipped `StepAttributionQualifier` / `skill_lifecycle` / Elo-by-retention-signal rather than admission signal.*

### 2.2 Component → shipped cousin → signal-diff

| Skill-Pro component | Shipped cousin | Signal the cousin consumes | Delta that survives |
|---|---|---|---|
| Skill ⟨I, π, β⟩ as data | `strategyvocab::Strategy { id, need, description, preconditions, body, cost }` (`riir-ai/crates/strategyvocab/src/doc.rs:26-37`); rule = trigger + fix (refine) | preconditions as rule queries | **no termination field**; the wire is `deny_unknown_fields` + `FORMAT_VERSION` — adding β is a wire change, not a relabel |
| Skill pool + gated registration | GSE `skill_lifecycle` (riir-ai Plan 214, "GOAT 13/13 … promoted to default in all 4 crates") | arena test result | none |
| Hindsight attribution to a component | `TrajectoryDoctor::localize_failure`; `TickFaultSite { tick_idx, violated, responsibility }` (`katgpt-pruners/src/step_attribution_qualifier.rs:197-207`) | first violated predicate + per-direction weights | attribution is to **directions**, not to the I / π / β roles |
| Accept-if-better gate (the PPO Gate's job) | `skill_opt::ValidationGate` — `accepted = delta > 0.0` (`katgpt-core/src/skill_opt/gate.rs`); `StepAttributionQualifier::qualify` — commit iff Δ ≥ threshold over a fixed replay window (DEFAULT-ON since 2026-07-06, `55a5184f`; `.benchmarks/381_…goat.md` G1–G6 PASS, 119 ns @ W=64) | realized score of a deterministic replay | see §2.3 — the gate's evidence channel |
| Rejected-edit memory, bounded edits | `skill_opt/{buffer,schedule}.rs` (Plan 144) | — | none |
| Selection: top-k by similarity then value | `pick_domain` + `SelectionMode::BetaPosterior` (refine default); `rating` Elo in `scene_loop` (`riir-instinct/src/scene_loop.rs:14-21`) | — | none |
| **Score-based retention** | `ValidatorCatalog::deregister(id)` "prune via `BanditPruner` decay" (riir-ai Plan 214 Task 3); refine `ActiveSet::demote` is operator-only (`active_set.rs:236-240`); `tier_guard` is a symmetric LR band on two beliefs (`tier_guard.rs:8, 106`); the statistics/time-triggered removal family recorded in riir-train R436 row 4 — `EvidenceTier::Withdrawn` (failure-count triggered, absorbing), `drop_level` (repair invalidation), engram `decay_confidence` (time-triggered) — plus R436's audited-discard operators (LFU eviction, contradiction-triggered REMOVE); Yang 2609.02042's prune-at-zero recorded as "Covered" in katgpt-rs R532 C4, an overclaim reconciled in §2.2a | bandit reward decay; failure counts; time; keep/revert counts | **none of these subtracts a running pool baseline** — LFU counts frequency, Withdrawn counts failures, bandit decay discounts a reward, but no item is scored against what the pool currently delivers, and none carries an LCB. That is the un-shipped form |

### 2.2a The R532 "Covered" record, checked and corrected

katgpt-rs R532 (SPACE) C4 records skill-library "prune-at-zero long-term success" as **Covered — "`skill_lifecycle` ships (PrunerMemory + test gates, katgpt-pruners)"**. Read at source, that citation overclaims for the prune leg: `PrunerMemory` is an append-only ring buffer whose only eviction is capacity overwrite (`crates/katgpt-pruners/src/skill_memory.rs:1-16`, `:58-59`; `append` at `:110-121`); it never prunes by score. The score-based removals that do ship live elsewhere — bandit arm demotion (`AbsorbCompress` reads recent rewards to promote/demote arms, `skill_memory.rs:10-11`) and `ValidatorCatalog::deregister` (riir-ai Plan 214 Task 3). A correction line has been appended to R532 pointing here. This note's claim survives in the precise form: **no shipped pool prunes on advantage-over-a-running-baseline with a confidence penalty** — bandit-decay removal is score-based but discounts a reward, it never asks whether the item still beats the pool.

### 2.3 Why the logprob gate is not the takeaway

The PPO Gate's evidence is the frozen LLM's token likelihood of logged actions under two prompt variants. Three things make it the wrong instrument here:

1. **It was not adopted.** Of the Skill-Pro-citing follow-ups whose abstracts were read (16), none reuses the importance-ratio gate; all re-derive acceptance from execution — SkillRevise 2606.01139, SkillSpec 2610.00704, AutoVerifier 2608.25637 ("replay validation detects no direct regressions"), Skill-α 2608.01678, SLIM 2605.10923. Prior art sits one week earlier on each ingredient: JitRL 2601.18510 (trust region + non-parametric memory + advantage), Training-Free GRPO 2510.08191 (RL-named text evolution), EMPO² 2602.23008 (the same ratio shape, to train weights).
2. **It is statistically weaker than it looks.** A per-step ratio re-scores actions but not the state distribution the old skill induced (the sequential off-policy bias). Accepting best-of-3 on the same logged batch selects on noise. The paper ablates gate on/off only.
3. **We have re-execution, cheaply.** `cargo clippy` is ~130 ms per verify; the harness replays labeled cal/test. Re-execution measures the thing the gate approximates.

The arithmetic itself already ships in the training vault (`riir-train-gpu/src/loss_grpo.rs:1296-1306`), so a model-based use — forward-only prompt-variant pre-selection — needs only a logprob logger. That is the model-based advocate's (B) "thin" finding and is recorded in §4, not filed.

### 2.4 The residual: a retention law

**Formula.** Per item ω, over invocations with realized outcome `r`:
- `gain += (r − r̄)` per active step; `n += 1`; `avg_gain = gain / max(1, n)`.
- `r̄` is a running baseline **per pool** (an EMA of pool outcomes), so an item must beat what the pool already delivers — this is what creates evolutionary pressure as the pool improves.
- `lcb(ω) = avg_gain − β·√(ln(1+t)/n)`, t = the oldest item's age in rounds.
- Retention: drop `lcb ≤ 0` once `n ≥ n_min`; drop semantic duplicates keeping the best `lcb`; if over capacity, drop ascending `lcb`. Never drop an item younger than the newborn floor.

**What is new here versus `BetaPosterior` and `elo_k_eff`.** `BetaPosterior` is an ε-quantile of `Beta(1+S, 1+F)` — a confidence penalty on a *success rate*. Elo is a *relative* rating. Neither subtracts a moving baseline, so neither can retire an item that succeeds often but no better than the pool. That is exactly RECURIS's finding: evolved memory "only grows: 51 skills added, 0 deprecated, 17 near-dupes survive; removing any single skill ≈ 0 effect" (riir-ai R347 `:34`), and injecting the whole library scores worse than none (65.6 vs 82.0, `:14`).

**Where it could act — and why nothing is filed.** A retention law needs (a) a pool with membership control and (b) an outcome per invocation. Verified state of the three candidate hosts:

| Host | Pool with membership control | Outcome per invocation | Usable today |
|---|---|---|---|
| Decision family (instinct / rethink) | the arsenal manifest — hand-authored, review-gated | **none in production**: `esc_voi.rs:25-28` "the decstat plane records receipts … never outcomes, so the production posture is `VoiPosture::ReceiptsOnly` and the value legs are BLIND" | offline on harness gold only (`cascade.rs:62-63`) |
| Rulewright (refine) | the corpus — `ActiveSet::demote` is operator-only, gate is decline-first (`adoption_gate.rs:35-39`) | yes — `.refine/adoption_outcomes.jsonl` | yes, but outcomes are keep/revert, not a scalar to baseline |
| Game runtime (riir-ai) | `ValidatorCatalog` (deregister via bandit decay); RECURIS consumer (Issue 047 T5 deferred) | validator arena results; kill credit | partially |

Choosing a host, and choosing what `r` is for keep/revert outcomes, are design decisions with a blast radius (the corpus lease, the arsenal pin lockstep). They are not made in a research note. The honest artifact for this leg is the formula and the three pre-registered checks below; the host is named by whoever picks the pool up.

### 2.5 Instinct: the specific answer

The user's framing was that this boosts Instinct. It does not, as things stand, and the blocker is a missing signal rather than a missing mechanism:

- Instinct's only residency policy is LRU on *lazy lanes*: "the least-recently-used Ready LAZY lane is released … best-effort, never a refusal; eager rows are never evicted" (`riir-instinct/src/serve_edge.rs:33-38`), fed by `LaneSlot`'s last-use stamp (`arsenal_ops.rs:418-420`). That is cache residency. Pool **membership** is the manifest (law A5). So Skill-Pro's FIFO ablation does not transfer: nothing offered to the selector is governed by recency.
- The ESC gate mutates only a latch: "`[min_rate, max_rate]` LATCHES the demotion" (`escalating_backend.rs:365`), `latched: Option<&'static str>` (`:377`). There is no score to evolve.
- `scene_loop` does keep a per-(arm, context-class) Elo book (`scene_loop.rs:200-211`) — a real mutated state, but advisory and opt-in (`runetrace_loop`).

What would unblock it: an **outcome join** on the decstat plane. That is already the named precondition in rethink's own code comment, so it is not filed again here.

### 2.6 The corpus question ("semantically conditional")

Three different things in the stack are called "the corpus"; the answer differs by which one is meant:

- **The rule corpora (refine).** Already conditional: every rule is a trigger plus a fix, dispatched per span. No consumer for termination semantics on an atomic span edit was identified: a rule is a trigger plus a replacement for one span, executed once under the compile gate — there is no procedure whose stop point a β field would govern.
- **The per-suite corpora (reflex).** Labeled exemplars the drafter scores against — not procedures. Making them conditional would mean replacing exemplar scoring with procedure selection, which is a different engine, not a corpus rebuild.
- **The strategy catalogue (riir-ai `strategyvocab`).** This is the real one. It is already condition-keyed (`preconditions`, `need`) and it is where RECURIS's growth problem lives. Skill-Pro adds one field (β, termination) and one operator (retention). The field is a closed-wire change (`FORMAT_VERSION` bump + golden fixture); the operator needs the host decision in §2.4.

### 2.7 On the MBT link

MBT (arXiv:2611.19800) contributes no content to this note. Its distillation (Issue 1239, the conditioned-KARC correction bank) shares only the phrase "keyed by condition": MBT re-keys a numeric correction by an exogenous regime label; Skill-Pro keys a text procedure by its own initiation condition and evolves the pool. Different mechanism, different substrate, no fusion cell. Issue 1239 is claimed and in progress by another lane; this note does not touch it.

### 2.8 Fusion (unplanned)

Skill-Pro's maintenance leg × RECURIS's invocation-conditional finding is the one combination worth a cell: score each item by advantage **conditional on invocation** (only steps where the item was active), which is already how the gain is defined, and retire items whose invocation-conditional advantage is non-positive. That is a direct test of RECURIS's "removing any single skill ≈ 0 effect" — the falsifier is that retiring the bottom quartile by LCB changes nothing, which would mean the pool is harmless redundancy rather than drag.

---

## 3. Verdict

**Gain (narrow).** One line: the paper's loop is already shipped and GOAT-proven from three sibling distillations; its single un-shipped leg, advantage-over-baseline + confidence-penalised retention, maps to one documented gap (RECURIS "0 deprecated") but has no host pool with an outcome feed that a plan could name today.

**Tiers, per track:**

| Track | Tier | Artifact |
|---|---|---|
| Modelless | **Gain** — the retention formula + reuse-rate readout; logprob gate explicitly not adopted | this note (§2.4); no plan — no host |
| Model-based | **Thin** — non-parametric paper; zero `backward(` in the clone; the only angle is a forward-only prompt-variant pre-selector on an owner-gated, thrice-negative carrier | recorded here (§2.3); no issue |
| Novelty | **2/4** — Q1 no (class covered by 105/172/381), Q2 no (not a new behavior class), Q3 no, Q4 n/a | not Super-GOAT |

**MOAT gate.** katgpt-rs: in scope as a generic formula, out of scope as a primitive until a host exists. riir-ai: the strategy-catalogue host is pillar-adjacent but rides the closed `strategyvocab` wire. Decision family: nothing lands — production is outcome-blind.

**Weakest points, named:**
1. The retention leg's value is inferred from RECURIS's growth measurement and Skill-Pro's FIFO ablation; neither is evidence that *LCB-on-advantage* beats bandit-decay pruning (which `ValidatorCatalog` already does). That comparison has not been run.
2. For refine, outcomes are binary keep/revert; "advantage over baseline" degenerates to success-rate-minus-pool-rate, which `BetaPosterior` nearly already is — for binary outcomes, §5 check 2 is the test that decides whether the baseline term means anything.
3. The published prior-art sweep was abstracts only; a gated variant in an appendix is not excluded.

---

## 4. Panel record

- **No-GD inventory** — the advocate session was rate-limited mid-run and had not returned when this note was written, so the coordinator verified the five gap questions directly: no `running_baseline` fn in katgpt-core / refine / reflex / instinct; `harness_evolution.rs` has `two_tier_score` + `champion_index`, no LCB or survival term; no corpus-unit reuse-rate readout beyond the field-level `--orphan-report`; `TickFaultSite` attributes to directions; instinct LRU is residency-only. The advocate's report, if it lands later, is corroboration only — any finding that changes §2.4 re-opens the verdict.
- **Model-based advocate — (B) THIN**, adopted. Verified: `grpo_loss_asymmetric` ratio/clip lines; the RLVR driver keeps no old logprobs.
- **Prior art — (b) novel composition of known parts**, adopted.
- **Cousin hunts** — both delivered; every fact cited above was re-read by the coordinator at `path:line`.
- **Round-1 reviewer findings folded (none changed the tier):** two internal cousins added to the §2.2 retention row — riir-train R436's removal operators and their shipped partials (`EvidenceTier::Withdrawn`, `drop_level`, `decay_confidence`; LFU and contradiction-REMOVE as audited discards) and katgpt-rs R532 C4's prune-at-zero record; the R532 overclaim reconciled at source (§2.2a; `skill_memory.rs` read at `path:line`; correction appended to R532).

---

## 5. Pre-registered checks (for whoever picks up §2.4)

Not a plan — three falsifiable statements the retention law must survive before it is a primitive:

1. **Beats decay-based pruning.** On a pool with a logged outcome history, retire-to-capacity by `lcb` vs by bandit-decay order; measure held-out pool score at matched capacity. A tie demotes the law.
2. **The baseline term earns its place.** Ablate `r̄` (score = mean outcome). If the two orders agree on ≥ 95% of retirements, the baseline is ceremony.
3. **Newborn protection is load-bearing.** With the floor at 0, the law must retire a non-trivial share of items before their first `n_min` invocations; with the floor, none.

**Report-the-Floor:** the law claims no distribution, interval or coverage — not applicable.

---

## 6. Verdict provenance

`request_verdict` (claude-sub-agent-verdict), session `5de1b809-b613-4e34-8668-e93a990591d3`.

- **Round 1: REVISE** — six fixes, tier unchanged ("The core verdict holds: **Gain (narrow)**"). All six applied: (1) §2.2 retention row widened with the R436 + R532 cousins; (2) the R532 prune-at-zero overclaim reconciled (§2.2a + correction appended to R532); (3) §2.6 reworded from assertion to scoped claim; (4) weakest point 2 now cites §5 check 2; (5) this section added; (6) redirect-appends normalized to one blank line.
- **Round 2: AGREE** (2026-11-23, same session) — "All six round-1 findings are applied and I checked them against the files. One cosmetic leftover is a fix-before-commit item, not grounds for another round" (the R172 blank-line surplus, fixed before commit; all appends now content / one blank / redirect). Tier unchanged and re-justified: Gain (narrow), model-based Thin, novelty 2/4, no plan filed. Commit executed after this line was written — the committed bytes are the agreed bytes.
