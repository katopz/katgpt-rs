# Research 583: Test-Time Communication — Verified Progress Sharing (team@k vs best@k)

> **Source:** [Scaling Discovery through Test-Time Communication](https://arxiv.org/abs/2609.21032) — Park, Kontonis, Garg, Krishnamurthy, Papailiopoulos (Microsoft Research + UC Berkeley), Sep 2026
> **Date:** 2026-09-22
> **Status:** Done — closed (validation note + one fusion issue filed).
> **Classification:** Public
> **Related Research:** 094 (Parallel-Probe 2D — the communicating-team arm, shipped), 386 (UnMaskFork — Eq 1 IS the paper's Prop 1 inequality, published Feb 2026), 248 (DeltaTok diverse sampling), 260 (MaxProof population test-time scaling), 367 (QuasiMoT QMC test-time scaling); riir-ai 337 (DeLM shared context — the admission-gated verified-progress substrate), riir-ai 369 (QSG crowd consensus), riir-ai 143 (Latent CCE moderator)
> **Related Plans:** none new (both search arms already shipped: `parallel_probe` + `renoise_ce::best_of_n_*`)
> **Cross-ref (riir-ai):** Issue 997 (stigmergic verified forage sharing — the one mechanism gap)
> **Verdict: Gain (validation-class).** ~8 of the paper's 11 mechanism rows ship in full — several at higher fidelity (fog-of-war gating, BLAKE3 admission) — one ships with its response polarity inverted (plateau detector yes, divergent family-switch no), and **two composed-mechanism gaps** remain: **stigmergic attractive forage-discovery sharing** and **plateau-triggered divergent family switch with peer-occupancy awareness** → both folded into riir-ai Issue 997. The paper's additive value to us is NOT a new primitive: the core inequality ships in our own lineage (UnMaskFork Eq 1, seven months earlier), and both search arms (communicating + independent) are deployed per-domain. The value is (a) the **conditions law** (stage-count m × verifier accessibility × per-agent compute) as external validation + a candidate orchestration-law packaging, and (b) the empirical multiplier (team@k ≈ best@4k–6.6k, growing with k).
> **Delta (2026-09-24):** Research 585 (SAT, arXiv:2609.22682) operationalizes this note's verifier-accessibility axis as a **measured** demonstrability score (external-panel discriminability of correct-vs-wrong certificates; ρ=0.90 with team-over-best across 8 benchmarks) — the axis recorded here as "candidate packaging" now has a published measurement protocol; our consumer-side instantiation filed as riir-clippy Issue 134.

---

## TL;DR

A team of k identical LLM agents sharing a workspace (append-only log, score leaderboard, disconfirmation log, atomic approach slots — no roles, no orchestrator) matches the solve rate of **4k–6.6k independent agents** on ARC-AGI-3, and the multiplier **grows with k**. On polyomino packing it sets a new best-known score (0.945 vs 0.894); on MNIST compression it beats the best-known human solution (1,957 B vs 2,461 B at 99.4% accuracy). The mechanism is **verified progress sharing**: once a discovery is verifier-confirmed, every agent builds on it — communication turns a *minimum of sums* into a *sum of minima*. The gains are conditional: multi-stage task (m>1) × agent-accessible verifier × sufficient per-agent compute; Terminal-Bench (verifier runs post-hoc, m≈1) shows **no** team advantage, and at 0.2× per-agent budget the team loses to a single agent (coordination tax).

**Distilled for katgpt-rs (modelless, inference-time):** the inequality and both arms already ship. `mcts_state_action_cache.rs` documents `Σ_z min_a ε_a(z) ≤ min_a Σ_z ε_a(z)` verbatim (UnMaskFork Eq 1 — same shape, (action × state) instead of (agent × stage), published Feb 2026). `parallel_probe.rs` IS the communicating arm (answer-consensus early stop + deviation pruning = adoption + herding control). `renoise_ce::best_of_n_*` is the independent arm. What does not ship: the **decision law keyed on (m, verifier-accessibility, per-agent compute, herding g)** choosing between the arms — recorded here as a candidate, honestly scoped as packaging of classical math (see §3.1) — and the two composed mechanisms of §2's gap rows.

---

## 1. Paper Core Findings

### 1.1 Headline results

| Task | Result | Detail |
|---|---|---|
| ARC-AGI-3 (Sonnet 4.6) | team@3 = best@13 (4.3×), team@5 = best@33 (6.6×) | Multiplier grows with k; advantage widens with depth (1.2×→3.6× from shallow to full solve at k=5) |
| Unsolvable-by-solo games | LP85: 0/64 single-agent trials → **65%** team@5; FT09: 25.9% best@3 → 90% team@3 | The effect is not merely efficiency — teams solve what no single agent can |
| Polyomino packing (Frontier-CS) | 0.945 (team@3 Sonnet, 3h) vs 0.883 solo / 0.894 prior best; 0.922 (team@4 Opus, 72h) vs 0.893 solo | New best-known score; single agents plateau, teams relay |
| MNIST classifier compression (GPT-5.6 Sol, 96h) | **1,957 B** at 99.4% vs 3,160 B best@4 and 2,461 B best-known human | ~20% smaller than the human solution |
| Terminal-Bench 2.0 | team@2 (60.67%) does **not** beat pass@2 (62.36%) | Negative result: verifier runs post-hoc, feedback partial |
| Token efficiency | Matching team's solve rate costs independent pools 3.8–4.9× the tokens | But best@k leads below ~400K output tokens/agent (coordination tax) |
| Per-agent efficiency (RHAE) | avg team@5 agent (8.9%) ≈ best@5 (8.8%); best team agent 13.6% | Communication improves the average member, not just the pool |

### 1.2 The mechanism — verified progress sharing

Task requires m successive improvements, verifier reports stage j/m. Writing X_ij for agent i's search time at stage j:

```text
T_best = min_i Σ_j X_ij        (one agent must complete the whole chain)
T_team = Σ_j min_i≤k X_ij      (a different agent may supply each breakthrough)
T_team ≤ T_best  pointwise
```

**Proposition 1** (X_ij ~ iid Exp(λ), τ = αm/λ, 1/k < α < 1, I(a) = a−1−ln a):
`Pr(T_team ≤ τ) ≥ 1 − e^{−m·I(kα)}` while `Pr(T_best ≤ τ) ≤ k·e^{−m·I(α)}` — team success → 1 exponentially in m while best@k → 0. Under herding (k agents collapse to g ≤ k effective search groups), the team's discovery rate drops to gλ and the high-success regime requires gα > 1: **sharing helps only to the extent that independent search survives after sharing**.

### 1.3 The conditions (when communication does NOT help)

1. **m = 1** (single-stage task): T_team = min X_i1 = T_best — no advantage even with a perfect final verifier. What matters is *dense faithful verification at test-time*, not evaluation alone.
2. **Verifier inaccessible at test time** (Terminal-Bench: evaluators check subsets, run after execution): no team advantage; premature adoption can make communication strictly worse (k agents → g < k effective candidates covering less than pass@k).
3. **Insufficient per-agent compute** (coordination tax): team@5 at 0.2× native budget each loses to a single agent at the same total budget. A single agent given 5× budget does NOT match the team — longer horizon ≠ coordination.

### 1.4 Protocol mechanics (the part that generalizes)

- **Atomic slot ownership** — agents race `mkdir slot-N` to claim distinct approaches (diversity by construction, no orchestrator).
- **Append-only findings log + score leaderboard + disconfirmation log** — async broadcast; results must carry measured outcomes + reproduction instructions; negative results are first-class.
- **Adopt-only-after-measured-better** — and after adoption, **preserve one meaningful variation** (anti-herding).
- **Plateau rule** — 3 consecutive non-improving attempts → switch to a *structurally different* approach family (not a variant); pick a family no active peer is on.
- Explicit SMC/particle-filter analogy: resampling degeneracy ↔ herding; population size ↔ team size; informative intermediate feedback ↔ verifier.

---

## 2. Substrate map — what already ships (mechanism-by-mechanism)

Full-tree grep + file reads across the 12-repo workspace (2026-09-22). The paper is **uncited anywhere** prior to this note.

| Paper mechanism | Status | Shipped cousin (consumed signal) |
|---|---|---|
| Verified progress sharing (adopt after verification) | **SHIPS — strongest overlap** | riir-ai `crates/riir-agents/src/shared_context.rs` `verify_admission` 2-stage gate (verbatim grounding + BLAKE3 tamper-evidence) + `async_coordinator.rs` instant-publish; GOAT G1–G5, failure-reuse 1.000 vs 0.870. riir-dapps `kat/replay.rs` K=2 quorum = verified adoption of peer fixes (submitter-excluded, canary-mixed), live on devnet |
| Communicating team@k | **SHIPS** | `katgpt-speculative/src/parallel_probe.rs` — N branches, majority-vote consensus early-stop, deviation-based pruning (verified verbatim: "consensus-based early stopping + deviation-based branch pruning") |
| Independent best@k | **SHIPS** | `katgpt-core/src/renoise_ce.rs` `best_of_n_stability` + `best_of_n_freedom` (near-best drift gate + occupancy diversity — best-of-k *with* diversity preservation) |
| min-of-sums inequality | **SHIPS (earlier)** | `katgpt-core/src/mcts_state_action_cache.rs` L42: `Σ_z min_a ε_a(z) ≤ min_a Σ_z ε_a(z)` — UnMaskFork Eq 1 (arXiv:2602.04344, **Feb 2026**, seven months before this paper), with property test `tests/mcts_state_action_cache_eq1.rs`. Same shape, (action × state) axes |
| Plateau detection (the trigger) | **SHIPS** | `katgpt-speculative/src/progressive_mcgs/stagnation.rs` `StagnationGate` (per-branch τ + global τ_global non-improving counters) + `cgsp::EntropyCollapse` (h < τ_low=0.30) — the detector half maps exactly |
| Plateau → **divergent** family switch (peer-unoccupied) | **PARTIAL — response polarity inverted** (verdict-round finding) | All three `StagnationTrigger` responses are *convergent*: `IntraBranchEvolve` references same-branch ancestors (= a variant, which the paper's rule explicitly excludes), `CrossBranchReference` adopts peers' top-N (= the opposite of "pick a family no active peer is on"), `MultiBranchAggregation` synthesizes the union of all branches. The paper's plateau rule is the anti-herding half and is *divergent*. Divergent fragments ship elsewhere: `EntropyCollapse::inject_exploration` (collapse-triggered divergence, single-agent, no peer-occupancy signal), UCT exploration bonus + `best_of_n_freedom` occupancy gate (occupancy-aware, not plateau-triggered). The **composed rule — plateau-trigger + structurally-different family + peer-unoccupied — ships nowhere** → second gap, folded into Issue 997 |
| Herding / premature convergence | **SHIPS (as mitigation + analysis)** | `agreement_contagion` + `katgpt_core::incidence::contagion_strength` (single witness = exactly zero crowd strength — anti-stampede); `qsg_crowd.rs` consensus-lottery/collapse laws (Nc ~ α/(m·|h|)); SMC ESS guard + systematic/residual resampling in `distributional_steering.rs` |
| Diversity preservation / avoid-set | **SHIPS** | `avoid_set_for` (ExperienceGraph Failure → −1 traversal); `set_admission` (colinearity cap + Vendi certificate + θ-ladder recovery); `best_of_n_freedom` occupancy gate |
| Shared verified subtrees across search branches | **SHIPS** | progressive MCGS DAG `reference_edges` (transposition reuse); spec-decode verified-prefix acceptance (the 99-module katgpt-speculative surface); `gdn2/tree_verify_bridge` commit-only-along-accepted-path |
| Cross-run verified progress memory | **SHIPS** | riir-clippy `traj_store.rs` (Elo + outcome counters steering future runs) + `.heal/fixseq/` attempt ring — sharing verified progress *across runs* instead of within (see §3.3) |
| **Stigmergic forage-discovery sharing between foragers** | **DOES NOT SHIP (attractive channel)** | The *repulsive* channel ships: `crowd_share` = `1/(1+CROWD_WEIGHT·n)` IFD interference over the shared `apple_hash` spatial index = negative stigmergy (avoid crowded cells). Missing: the *attractive* peer channel — a forager publishing a verified discovery that changes peers' targets. Note `apple_hash` is omniscient environment state, so the discovery-sharing value proposition lives in the fog-of-war lane where per-NPC `SpatialBelief` is private **by design** (two-brain model) → **riir-ai Issue 997** |

**Signal-diff on the two closest coverage claims (§3.6 discipline):**
- `parallel_probe` consumes **answer-equality** (`A: Eq + Hash`) as its verifier — weaker than the paper's arbitrary scorer (level completion, packing score). Same class (verified progress), narrower verifier. Not a gap worth an issue: answer extraction is the right verifier for the decode domain it serves.
- `shared_context` admission consumes **provenance/integrity** (verbatim grounding + BLAKE3), not a measured-better score. The paper's adopt-only-after-*measured-better* threshold is score-conditional; ours is integrity-conditional (the curator's Draft→Candidate→Live tiers are the governed-adoption analog). Partial delta, noted — not actionable at runtime scale (our entries carry Failure/Finding kinds; the demo's `peer` metric already measures the benefit side).

---

## 3. Distillation

### 3.1 The one candidate primitive: the conditions law (honestly scoped)

**Pinned claim:** *a closed-form decision law — sum-of-minima order-statistics inequality with a herding correction g ≤ k, keyed on (task stage-count m, verifier accessibility, per-agent compute) — choosing between independent parallel sampling (best@k) and communicating parallel search (team@k).*

**Prior-art verdict (web, 2026-09-22):** SURVIVES **narrowly, as packaging only**.
- The raw inequality is classical: series-parallel system reliability (textbook) + Lorenz 2016 parallel-restart completion probabilities. UnMaskFork Eq 1 (our own cited lineage, Feb 2026) publishes the same inequality for (action × state).
- Closest published decision rules: Wu et al. [arXiv:2408.00724] compute-optimal inference (parallel best-of-N vs sequential revision, keyed on model size + difficulty — no teams, no m, no verifier axis); Choi et al. [arXiv:2508.17536] Debate-or-Vote martingale theory (single-stage beliefs, no search); Kim et al. (Nature MI 2026) capable-models-outgrow-collaboration (empirical capability condition); Anthropic's multi-agent engineering blog (qualitative fit conditions; token usage alone explains 80% of BrowseComp variance — the null hypothesis any law must beat).
- The g ≤ k herding term *inside* a closed-form orchestration law: not found published.

**Why this is recorded, not planned:** both arms are already deployed correctly per-domain in this stack (parallel_probe where answers are extractable; best_of_n where they aren't; shared_context in the task-agent lane; traj_store across healer runs). The law's value here is (a) it *explains* those deployments, (b) it prices the one place we'd consider flipping a mode. A `team_vs_best_policy(m, verifier, compute, g)` pure function would be documentation wearing a function signature until a consumer with a measured gate exists. Per §1.55 this is "validates our design" — not actionable → no plan. Reopen trigger: any surface where the two arms actually compete for the same slot (e.g., a future healer parallel-fix lane with expensive per-attempt LLM calls, where m>1 cascades × ms→minutes compute would flip the law's verdict).

**Plateau-rule claim (B):** as a general mechanism it is KILLED by prior art — selection hyper-heuristics (Cowling et al. 2001 choice function; Drake et al. 2020 survey), stagnation-triggered restarts (SAT: Glucose/Luby lineage; EA: IPOP-CMA-ES), Go-Explore archive weighting. Survives only as LLM-team-scoped instantiation — and our substrate ships only the detector half of it (§2 row on the divergent family switch: the peer-unoccupied response is the second gap, riding Issue 997). No claim.

**Conditions-applied-to-new-domains claim (C):** genuine published gap (no condition-based application found in game-AI or code-repair orchestration). Our stack is ahead of the literature here — the note records the mapping:

| Law condition | Healer surface (consumer reframe) | Game surface |
|---|---|---|
| m > 1 | fixseq error-cascade chains are multi-stage | multi-stage foraging trips, deliberation ladders |
| Verifier accessible | real clippy oracle (dense, faithful, at fix time) | measured patch density, level completion |
| Per-agent compute | **ms-scale attempts → tax dominates → independence + cross-run sharing wins** (validates traj_store design) | energy budget (MVT patch-leaving already models this) |
| Herding g | nonergodic revive-gate; contagion_strength | agreement_contagion single-witness gate |

### 3.2 Fusion — the two composed-mechanism gaps (→ riir-ai Issue 997)

**Gap 1 — stigmergic attractive sharing.** Paper × riir-ai 337 (admission-gated shared context) × swarm MVT foraging = **stigmergic verified forage sharing**: foragers publish verifier-confirmed patch discoveries (measured density), peers adopt-after-adoption-threshold while one preserves variation, `contagion_strength` gates the crowd response (herding mitigation ships), MVT supplies the compute-sufficiency condition. The *repulsive* half already ships (`crowd_share` IFD interference over shared `apple_hash` — negative stigmergy); the fusion adds the *attractive* peer channel, valuable precisely in the fog-of-war lane where `SpatialBelief` is private by two-brain design. Selling-point shape: *forager teams that solve exploration no single NPC can*. **Novelty TBD** — ACO/pheromone stigmergy is 30+ years of prior art (and covers repulsion and attraction alike); the delta must be carried by verification-gated adoption + herding gate + the when-to-share conditions, not by stigmergy itself. Filed as an issue, not a plan, until the ACO delta check passes.

**Gap 2 — plateau-triggered divergent family switch.** The paper's anti-herding plateau rule (3 non-improvements → structurally different family, unoccupied by peers) composes three elements that ship only separately here (detector: StagnationGate/EntropyCollapse; divergent response: inject_exploration, single-agent; occupancy awareness: UCT bonus / best_of_n_freedom, not plateau-triggered). Folded into Issue 997 as a required design element of the forage lane (it is exactly the rule that keeps shared adoption from collapsing the swarm onto one patch) — not filed separately: without Gap 1's adoption channel there is no peer-occupancy signal to be aware *of*.

Fusion non-goals: workspace multi-session coordination (staged_set_audit, Session: markers, worktree verdicts) already implements the paper's protocol lessons at the discipline level; the paper's slot-ownership and disconfirmation-log rules are validation, not changes.

### 3.3 The cross-run insight (free, worth recording)

Our healer shares verified progress **across runs** (traj_store Elo, fixseq ring) rather than within runs. The paper's law justifies this exactly: per-attempt compute is ms-scale (coordination tax dominates within-run parallelism), so independence + persistent verified memory is the sum-of-minima benefit obtained serially without the tax. The K=2 replay quorum adds verified *adoption* where verification is the expensive part. Current design is what the law prescribes — recorded as validation.

### 3.4 Latent vs raw boundary (mandatory check)

No new boundary-crossing behavior. The proposed forage-sharing stays in the think-brain/latent lane (belief about patch value, sigmoid-gated adoption); physical position/energy stays raw + synced. Issue 997 carries this constraint.

---

## 4. Verdict

**Tier: Gain (validation-class).** One mechanism gap filed (riir-ai Issue 997); three cousin notes updated with PASS-Redirects; the conditions law recorded as a candidate with an explicit no-plan scoping.

| Gate | Criterion | Honest answer |
|---|---|---|
| **Q1** No prior art? | **FAIL for the mechanisms** (ship — see §2 table; the inequality itself ships via UnMaskFork Eq 1, earlier). **Narrow PASS for the law packaging** (no published m × verifier × compute × g keying — Wu/Choi/Kim don't cover it). |
| **Q2** New behavior class? | **FAIL here** — both behavior classes (communicating team, independent pool) ship. The new-behavior candidates (stigmergic attractive forage sharing; plateau-triggered divergent family switch) are deferred to Issue 997 pending the ACO delta check. |
| **Q3** Product selling point? | Not from this note directly. The selling-point candidate ("NPC teams that solve exploration no single NPC can") rides Issue 997, unproven. |
| **Q4** Force multiplier? | **YES as validation** — the law connects parallel_probe, best_of_n, shared_context, traj_store, contagion gating, MVT foraging across 3 repos; all already connected by deployment, now connected by *explanation*. |

**MOAT gate:** katgpt-rs — principle-level validation note (this file), no new primitive claimed. riir-ai — the fusion gap routed to `.issues/997` (game-surface home per the priority ladder). No riir-chain / riir-neuron-db angle (no commitment/freeze/consolidation mechanism in the paper).

**Latent-space reframing check:** belief framing — verified progress sharing IS belief-state updating from peer observations with an admission gate (shared_context, shipped); functor framing — adoption = projection onto the peer's verified direction with one-variation preservation (the preserve-variation rule is a rank-1 correction, ships as latent_functor pattern); SMC framing — resampling with ESS guard (shipped, `distributional_steering`). No reframing yields a capability we lack except Issue 997's forage lane.

**Weakest point (pre-registered for the reviewer):** the note's Gain verdict rests on the grep-agent's substrate map; I spot-verified the four load-bearing claims (parallel_probe verbatim, Eq 1 verbatim, shared_context GOAT numbers vs AGENTS.md, replay quorum vs AGENTS.md) but did not line-read `progressive_mcgs/stagnation.rs` or `forage_expand.rs` myself. Second: the ACO delta check for Issue 997 is *owed*, not done — the issue is filed as novelty-TBD precisely because an unchecked Super-GOAT claim is the canonical failure mode.

### Verdict-round addendum (2026-09-22)

The Claude reviewer's line-reads closed weakest point #1 **and found two corrections, both incorporated above**: (1) the plateau row was inverted — `StagnationGate`'s responses are convergent (reference/adopt/synthesize) while the paper's rule is divergent and peer-occupancy-aware; the detector ships, the response does not (downgraded to PARTIAL, second gap recorded); (2) Issue 997's premise understated shipped substrate — `crowd_share` IFD interference over shared `apple_hash` IS repulsive stigmergy; the gap is the *attractive* peer channel (issue reframed accordingly). Both new files staged before the gate-clean re-assertion per the reviewer's third bullet.

---

## 5. Routing

- **riir-ai Issue 997** — stigmergic verified forage sharing (fusion idea, novelty TBD vs ACO; carries BOTH composed-mechanism gaps; GOAT gate sketch + substrate-first map included).
- **PASS-Redirects** appended to: `.research/094_Parallel_Probe_2D_Probing_Parallel_Thinking.md` (this note), `.research/386_UnMaskFork_Deterministic_Action_Branching_MCTS.md` (Eq 1 priority + agent×stage dual), riir-ai `.research/337_Decentralized_Verified_Shared_Context_DeLM.md` (external validation of the admission-gate design + the measured-better-adoption delta).
- **No plan, no new primitive** — the conditions law is recorded in §3.1 with its reopen trigger (a surface where the two arms compete for the same slot at LLM-scale per-attempt cost).
- **riir-train**: nothing (the paper's MNIST task involves training classifiers, but the mechanism under study is test-time coordination; no recipe value).
