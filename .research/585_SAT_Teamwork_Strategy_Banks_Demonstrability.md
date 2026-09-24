# Research 585: SAT — Self-Organizing Agent Teams, Teamwork-Strategy Banks, and Demonstrability

> **Source:** [Self-Organizing Agent Teams Learn to Reason Together](https://arxiv.org/abs/2609.22682) — Pappu, Suzgun, Kwon, Bianchi, El, Kochenderfer, Cao, Zou (Stanford / Together AI / Emory), Sep 2026
> **Date:** 2026-09-24
> **Status:** Done — closed (one issue filed: riir-clippy 134)
> **Related Research:** 583 (test-time communication — the verifier-accessibility conditions law, the closest cousin), 146 (GEPA — the learning loop's method family), 438 (Sheaf-ADMM multi-agent coordination), 469 (collective-intelligence payoff schemes), 380 (conversable-complexity agentic collective), 094 (Parallel-Probe — the communicating-team arm), 386 (UnMaskFork — the min-of-sums inequality)
> **Related Plans:** none new (both closest mechanisms ship: frontier miner = strategy-bank learning; riir-clippy Issue 125 taxonomy = coverage/selective split)
> **Cross-ref (riir-clippy):** Issue 134 (demonstrability selection-gap gate — the one filed delta); Issues 089/102/125 (the shipped cousins this note signal-diffs against)
> **Cross-ref (riir-ai):** Issue 997 — CLOSED 2026-09-23 at G2 parity (Bench 952); the `forage_share` lane ships opt-in and its re-open triggers are the game-surface absorber (via Research 583)
> **Classification:** Public

---

## TL;DR

Fixed 3-model LLM teams learn **teamwork strategies** (an ordered DSL of conversational phases: participation set, rounds, local-vs-summary information flow, persistent role prompts, synthesis rules) by offline evolutionary "teamwork reflection" from only 15 (AIME) / 25 (GPQA) seed problems — a designated member mutates roles/phases against an archive of transcripts+outcomes, validation probes (5 held-out training problems) measure transfer, a leakage audit blocks problem-specific content, and a coverage-greedy **frozen bank of 10** deploys unchanged across benchmarks. Math team: 66.7% avg vs 48.8% best member, 58.7% compute-matched linearization, and **59.0% routing-oracle coverage** (perfect per-problem selection over members' independent answers) — beating the oracle proves interaction *constructs* answers no member sampled. The paper's second finding is the transferable law: **demonstrability** (Laughlin–Ellis 1986: whether correct reasoning can be distinguished from incorrect), operationalized as the balanced rate at which an external 10-model panel picks a correct over an incorrect team certificate, tracks improvement-over-best-member across 8 benchmarks at **Spearman ρ=0.90 (p=0.005)** — collaboration pays exactly where correct output is *recognizable once it appears*.

**Distilled for the stack (modelless, inference-time):** ~5 of 7 mechanism rows already ship or are absorbed (signal-diffs in §2): the conditions law → Research 583; defect-audit selection → clippy oracle-lane guards + the riir-clippy Issue 100 (rustc_errors) LEX keep rule; strategy-bank reflective learning → the frontier miner (validation probes ≈ verify-pass, coverage-greedy bank ≈ active-set); generation-vs-selection separation → riir-clippy Issue 125's `ChoiceTaxonomy` (`wrong_rule` vs `wrong_span`, `selective_accuracy`). The unshipped delta: **demonstrability as a measured, per-domain constant consumed as an allocation gate** — spend candidate-pool depth where selection discriminates, spend rerank/discrimination work where it does not — plus the **homogeneous-team control row** for multi-domain fanout claims. Filed as riir-clippy Issue 134.

---

## 1. Paper Core Findings

| Result | Number | Why it matters |
|---|---|---|
| Math/physics suite avg (5 benchmarks) | **66.7%** vs best member 48.8%, self-consistency 55.4%, linearization 58.7%, MoA 57.3%, homogeneous team 56.0% | Beats every control incl. compute-matched single-agent replay of the same strategy structure |
| vs **routing oracle** (perfect selector over members' independent answers) | 59.0% coverage → SAT exceeds by **+7.7** avg; AIME26 +13.4 | Surpassing truth-wins coverage = interaction produced answers absent from all members' samples |
| Knowledge/logic suite | SAT 72.8% (best) but **below** oracle 79.6%; team coverage **87.9%** | The gap separates *creating* a correct solution from *recognizing* it — selection is the bottleneck |
| **Demonstrability law** | ρ=0.90, p=0.005, n=8; leave-one-out ρ=0.86–0.96 | Measured certificate-discriminability predicts where collaboration pays |
| Learning cost | 15 + 25 seed problems; 6 mutation rounds/problem; frozen bank of 10 | Organization is learnable from tiny data and transfers unchanged (AIME24→25/26, HMMT, TheoremQA) |
| Judge design | Audits certificates for **named local defects**; forbidden from re-solving; answer-frequency allowed ONLY as tie-break among defect-free candidates | Selection without an oracle; protects correct minorities from frequency suppression (AIME26 II-01: audit overturns a wrong 2-member majority) |

Mechanism details worth keeping: (a) each strategy execution ends in a **certificate** — a short self-contained reasoning trace intended to be step-checkable, not a bare answer; (b) the search varies *who deliberates, when, with what information, under what roles* — never problem-specific decomposition; (c) strategies encode **member-specific comparative advantages** learned from transcripts (e.g. divergence_reconciliation keeps DeepSeek as a diversity source but excludes it from the verification phase); (d) debate's measured failure mode — wrong majority becomes unanimous (GPQA 132 transcript) — is the exact anti-pattern the defect-audit judge + minority-preservation roles counter.

## 2. Distillation — mechanism × stack signal-diff

| SAT mechanism | Ships as | Signal-diff (what each consumes) |
|---|---|---|
| Conditions law: collaboration pays iff verifier-accessible (per-583: m × verifier × per-agent compute) | Research 583 §conditions-law; the riir-ai `forage_share` lane (riir-ai Bench 952 — Issue 997 closed 2026-09-23 at G2 parity; the lane ships opt-in) | 583 records the **axis** in prose ("candidate packaging"); SAT **measures** it as a score with cross-benchmark predictive validity (ρ=0.90). Neither ships a consumer. |
| Defect-audit selection (reject only by naming a local defect; frequency as tie-break) | riir-clippy oracle lane (`clippy_feed.rs` P23 escape-refusal, P34 length-proof decline, whole-file parse gate, leftmost-wins overlap) + riir-clippy Issue 100 keep rule (rustc_errors: strict LEX decrease of (masking, count)) | Same signal class: named-defect rejection at selection time. **Covered** — the LLM-judge variant (auditing free-form certificates oracle-free) needs certificate artifacts + a judge we do not have modellessly; the oracle-present halves ship. |
| Strategy-bank learning from experience (reflect → mutate → validate on probes → coverage-greedy freeze) | Frontier miner (riir-clippy Issue 089: HeldOut/Frontier, verify-pass = real clippy, graduated families, archive rows) + self_evolve + active-set lifecycle (riir-clippy Issue 102) | Validation probes ≈ the miner's verify-pass/holdout; frozen bank ≈ freeze tables + active-set caps. **Covered** at the code-domain level — SAT is the LLM-team instantiation of an architecture we already run. |
| Generation-vs-selection separation (coverage vs accuracy) | riir-clippy Issue 125 `ChoiceTaxonomy`: `wrong_rule` (gold surfaced, outranked) vs `wrong_span` (never surfaced); `selective_accuracy` = correct/covered | The split **ships as measurement**. Not consumed as an allocation signal. |
| Routing-oracle evaluation bar | The taxonomy's gold-in-window + hits@1 already encode it; literature-standard (LLM-Blender 2023; Large Language Monkeys 2024) | **Covered** for the suggest lane's eval; worth naming in future multi-domain claims. |
| **Demonstrability as measured, per-domain constant → gate** | Nothing | The filed delta: a per-domain discriminability score over existing labeled outcomes, consumed by pool-depth/K and rerank-investment allocation. |
| Homogeneous-team / linearization controls | Nothing | Eval-design adoption: any "domain fanout beats single domain" claim should carry the best-single-domain + homogeneous-control rows. |

**Latent-reframe check (§1 step 3):** no per-NPC latent-space reframe is stronger than the healer-consumer one — demonstrability is a property of the *selection channel*, not of a latent kernel; the game-surface instance (deliberate only when the plan is sim-checkable) is already how `SwarmDeliberationSystem` works (forward-simulated escape routes = maximal-demonstrability tasks; trigger = stuck/churn, absorbed by 583 + the riir-ai `forage_share` re-open triggers, Bench 952).

## 3. Verdict

**Per-track verdicts (TTPO rule — one per track, never pooled):**

- **Track (a) modelless/healer — Gain.** Actionable: measured demonstrability axis + allocation gate (riir-clippy Issue 134) + homogeneous-control eval row. Not GOAT today: the measurement half is instrumentation; the gate half must pass its own GOAT (the ρ-style prediction on our domains) before promotion.
- **Track (b) game/runtime — absorbed, no new file.** Discard reason (mechanism-level, per §3.5 scrutiny): the condition law for collaborative compute on the game surface (verifier-accessibility gating) is recorded in Research 583 and instantiated by the riir-ai `forage_share` lane — riir-ai Issue 997, **CLOSED 2026-09-23** at G2 product-parity (Bench 952; the lane ships opt-in, retirement declined), whose re-open triggers (sparse-swarm regimes, task-level channel reach, genuinely expensive verification) are the live surface a new issue would duplicate; SAT's *heterogeneous-teams-beat-homogeneous* datum (66.7 vs 56.0) is external validation for the healer's domain-diverse banks, not a new game mechanism.
- **Track (c) model-based/training — panel skipped, no plan.** The paper is inference-side (skill exemption: no optimizer/loss/schedule; its "learning" is GEPA-family reflective prompt evolution — our analog ships as the frontier miner). Its own future-work (distill team reasoning into members) is a direction, not a recipe; no GPU-hours estimate is constructible. Auditable discard.

**Super-GOAT scoring (honest):** Q1 prior art — partial (the axis ships via 583; the measured-score + gate does not; literature: ROC-n-reroll anticipates the law theoretically for single-model BoN, no published per-domain gate in a production system). Q2 new behavior class — no (allocation/instrumentation). Q3 selling point — no. Q4 force multiplier — weak (healer selection + eval). **Not Super-GOAT, not GOAT-as-measured → Gain.**

**MOAT gate:** riir-clippy consumer-first — the filed issue targets measured healer quality (selection-gap axis feeding pool-depth allocation), squarely in-scope; nothing routes to katgpt-rs open primitives (the gate is a consumer of already-shipped metrics, not new substrate math).

**Prior art (§4 searches, both agents, 2026-09-24):** routing-oracle bar = YES prior (LLM-Blender ACL 2023 oracle ranking; Large Language Monkeys 2024 coverage-vs-selection; "When to Overturn Majority Voting" Jun 2026 74.3-vs-84.3 oracle gap). Defect-audit selection = PARTIAL (AgentAuditor arXiv:2602.09341 Feb 2026 — audit-replaces-voting via divergence-tree path search, not defect-naming + frequency-tie-break; ChatEval ICLR 2024 debate referees). Demonstrability→LLM-teams = NO prior found (Laughlin–Ellis 1986 psych-only; nearest theoretical anticipation: ROC-n-reroll, Dorner et al. ICLR 2026 — verifier ROC governs BoN payoff; a concurrent Sep-6-2026 process-loss/assembly-bonus paper imports org-psych framing without the demonstrability construct). Learned teamwork-strategy banks = PARTIAL (AFlow/MaAS/GPTSwarm/OPTAGENT learn workflows/topologies; GEPA evolves prompts; none learns frozen team-conversational strategy banks). The paper's own novelty rests on the bank + the demonstrability operationalization — consistent with our extraction focus.

**Weakest point (named by us):** the filed gate may be *adjacent-redundant* — riir-clippy Issues 102/130 already consume per-rule evidence streams; if pool-depth allocation never actually changes under the constant, the issue's gate half dies and only the standing-column instrumentation survives. The issue's GOAT is designed to falsify exactly that. A second, worse mode surfaced in review: a raw **success-rate** constant would let T3's ρ pass for the wrong reason (easy domains carry both high success and high gain-from-depth — failing green); repaired in the issue by construction — both T1 halves are coverage-conditioned (condition on the correct candidate having been present) and T3 refuses below five qualifying domains.

**Fusion (§1 step 6, recorded even if unplanned):** demonstrability (this note) × Research 583's stage-count m × the frontier miner's K-minimality = a *budget law for repair search*: candidate depth pays multiplicatively with stage count × discriminability — testable on the rustc_errors interacting corpus (m>1 by construction) vs single-shot clippy spans (m≈1). A measured instance of the same separation already sits in our record: the riir-ai `forage_share` G2 failure (Bench 952) is the **coverage-saturated pole** — blind diffusion already solved discovery at 1000-NPC density within 100–250 ticks, so collaborative compute could not pay at any pool depth (a *generation-side* saturation failure, NOT SAT's low-discriminability *selection-side* failure — same separation, opposite sides; cite it as the mirror, not as a confirmation).

---

## PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** Pappu et al. [arXiv:2609.22682 "Self-Organizing Agent Teams Learn to Reason Together"] — no PASS issued (Gain verdict); redirects absorbed as Related Research: 583 (conditions law), 146 (GEPA loop), 094/386 (team-vs-best arms).
