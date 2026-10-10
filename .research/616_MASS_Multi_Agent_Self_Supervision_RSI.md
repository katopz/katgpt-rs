# Research 616: MASS — Recursive Self-Improvement through Multi-Agent Self-Supervision (PASS modelless / GAIN training-recipe)

> **Source:** [Recursive Self-Improvement through Multi-Agent Self-Supervision (MASS)](https://arxiv.org/pdf/2610.12176) — Hyunin Lee, Jinglue Xu, Jeffrey Seely, Donghyun Lee, Somayeh Sojoudi, Matei Zaharia, Yujin Tang (UC Berkeley + Sakana AI), arXiv:2610.12176v1, 8 Oct 2026. Code/blog linked from the paper landing page.
> **Date:** 2026-10-10
> **Status:** Done — closed.
> **Classification:** Public
> **Related Research:** 440 (AIDE² — the canonical code-level-RSI PASS precedent; MASS is in that family but the inner loop is a BOUNDED vocabulary search, not free harness-code rewriting), 289 (RecursiveMAS — bi-level-already-shipped precedent), 512 (Meta^n — fixed-Ω PASS), 320 (Red Queen Gödel Machine — co-evolved evaluators, training→riir-train), 146 (RLM-GEPA — whose PASS-Redirects addendum already covers the OTHER "Mass" framework, arXiv:2502.02533 — name collision, NOT this paper), 040 (OpenDeepThink — BT pairwise elite evolution, shipped as `bt_rank`), 585 (SAT — AFlow/MaAS/GPTSwarm landscape row)
> **Related Plans:** 456-riir-train (`456_mass_recipe_extraction_waves.md` — the GAIN-track output: 12-item recipe waves with GOAT gates)
> **Related (riir-train):** Research 450 (J-Zero / Multi-Agent Evolve judge co-evolution — MASS cites MAE as the shared proposer/solver/judge cousin)
> **Verdict: per-track, not pooled.**
> - **Modelless / decision-serving track: PASS.** The inner loop's proposer is LLM text generation (the R440 test: "what decision is each LLM call computing?" → "write a better workflow spec") — no probe/draft/pruner, no freeze/thaw snapshot, no latent projection computes it. The loop's modelless analog already ships decomposed across the stack: elite-trajectory replay = refine `LatentFixMemory`/`traj_store` (evidence-tiered fix-trajectory memory + retrieval), sealed self-generated-corpus gates = reflex synthesis lane (teacher pass + agreement veto + echo gates), BT elite selection = `katgpt-pruners/src/bt_rank.rs` (R40), paired judging = the instinct arena + `stats.rs`. The extraction inventory below records the genuinely new-to-stack instruments (info-concentration metrics, agreement-gated judge trust, capacity attribution) with their consumer verdicts — none survives to plan grade on consumer strength today.
> - **Training track: GAIN.** The SFT-on-self-generated-multi-agent-traces recipe is concrete, cheap at pilot scale, and attacks a documented measured FAIL: the L4 LoRA lane's 0/60 EM (refine Plan 336). The **workflow-stripped-prompt internalization** finding (delegation 1/110 → 87/110 bare-prompt after training WITHOUT the workflow in the prompt) is the direct mechanism-level counter to that FAIL's diagnosis. Filed as **riir-train Plan 456** (Path 0.5 — applicable training paper gets a Plan, actively pursued per the standing doctrine).

## TL;DR

MASS runs a homogeneous RSI loop on Qwen3.6-27B-FP8: one model is agent + workflow-optimizer + evaluator. **Inner loop (Algorithm 2, prompt-only):** a workflow is a textual spec `{h = orchestrator hop order; per-subagent role r, instruction u, output contract c}` appended to the task prompt; an elite-size-1 evolutionary search proposes w(i), executes the coding agent under it, pairwise self-evaluates vs the incumbent workspace, retains w* on a win, and feeds the (verdict, rationale) history D_t into the next proposal — with the optimization prompt explicitly prioritizing **communication structure (h, c) over content (r, u)**. **Outer loop (Algorithm 3, weight mutation):** execute the retained workflow 18×/task, Bradley–Terry tournament over workspaces, SFT on the top-15 trajectories with **the workflow stripped from the prompt** (2,435 windows: 282 orchestrator / 2,054 subagent / 99 auxiliary, 17.2M supervised tokens, LoRA r64/α128, 2 epochs, ~19 h on 6×H100). Two cycles: +1.2–1.6× score/output-token on four research benchmarks, win rate vs base 53.9% → 69.9%, evaluator agreement with external judges 0.73 → 0.93, and bare-prompt coordination behaviors internalize (delegation 1/110 → 87/110, code handoffs 1/110 → 83/110).

**Why it matters here (three findings our stack can use without the loop):**
1. **Trace-efficiency law (Claim 2):** a student trained on multi-agent (orchestrator + bounded-subagent) traces beats a single-agent student trained on **1.4× more tokens** — M+ 68.3% @ 28.82M processed tokens vs S++ 64.0% @ 42M. Supervision shape is a first-class lever; decomposition is not overhead, it is signal.
2. **Bottleneck law (ablation):** upgrading the workflow **optimizer** moves the loop far more than upgrading the **evaluator** — in a homogeneous loop, capacity-attribution says invest in the proposer/routing layer, not the judge. Our escalation/arena posture (selection is the arm, lanes are static registered artifacts) is consistent with it.
3. **Internalization mechanism:** removing the scaffold from the training prompt is what converts scaffold-following into scaffold-free behavior. This is the cheap, directly testable recipe behind the L4 retry (Plan 456 Wave 1).

**Why the modelless track is still PASS:** the loop's only genuinely novel producer — "propose a better multi-agent workflow spec" — is LLM text generation over natural language. That is the R440/R512 class exactly. What the loop *consumes and produces around* the LLM (elite retention, BT tournaments, information-concentration metrics, agreement-gated trust, freeze/lineage of retained artifacts) ships in pieces across the quintet, catalogued in §2 with signal-diffs. The one composite our stack does NOT have — the alternation itself — is a product loop for LLM-agent stacks, not a modelless primitive; our ladder's analog (corpus growth + seat re-benching, `self_evolve` + score-bench) already climbs it without the evolutionary producer.

## 1. Paper core (what actually ships in it)

| Component | Mechanism | Track |
|---|---|---|
| Workflow spec `w = {h, (r, u, c)}` | Pure text appended to task prompt; orchestrator + subagents share one model | Modelless (prompt) |
| Elite-1 evolutionary search | Propose → execute → pairwise self-eval vs incumbent → retain on win; D_t history conditions proposals; structure-first mutation emphasis | Modelless (search control) |
| BT tournament + top-K | 18 executions/task → pairwise workspace judgments → BT scores → top-15 SFT, rank-16 validation | Modelless (selection math) |
| SFT with workflow stripped | Assistant-only loss masks; 2:1 subagent:orchestrator sampling; LoRA r64/α128/dropout 0.05; AdamW β2=0.95, LR 3e-5, 5 warmup, cosine→10%, 2 epochs; val-loss checkpoint; merge + block-FP8 requant; **fresh LoRA per cycle, never stacked** | Training (riir-train) |
| Info-concentration analysis (Claim 1) | CCA MI-proxy + total correlation over component embeddings: contracts+hops gain task-specificity, roles/instructions genericize, redundancy falls | Offline instrument |
| Internalization audit | Bare-prompt behavior rates (delegation, code handoffs) with matched seeds | Offline instrument |
| Capacity attribution | Executor fixed; swap evaluator-only vs evaluator+optimizer → proposer capacity dominates | Experimental design |
| Role transfer | Task-solving-only SFT improves evaluator (0.73→0.93) AND optimizer (its workflows win 64% even executed by the base) with zero role-targeted data | Training finding |

## 2. Path 0 inventory — merged advocate table with signal-diffs

Both advocates ran (three-track panel, one spawn round, 2026-10-10). No-GD: 32 extractions; Model-based: 12 recipe items. Merged; every discard carries its mechanism-level reason. **No-GD items are judged for consumer strength, not just existence.**

### 2.1 Kept at note level (instruments with real signal, no plan-grade consumer today)

| # | Extraction | Host (as shipped) | Why no file now |
|---|---|---|---|
| I1 | **Contract/hop information-concentration metric** (CCA MI-proxy + total correlation over component embeddings) as an offline design audit for orchestration manifests | riir-engine `latent_functor/` re-estimation machinery could host it; instinct `arsenal.toml` / EscalateSpec rows are the auditable artifacts | Our manifests are small and hand-curated, and already concentrate task-specificity in routing rows **by design** (the escalation grammar is suite-keyed routing). The metric would confirm, not change, the shipped shape. Consumer strength: weak. Reopen trigger: any orchestration surface growing to ≥20 programmatic rows with per-suite variance. |
| I2 | **Agreement-gated judge trust** (track evaluator-vs-external-anchor agreement across cycles; below floor → abstain/escalate, above → autonomous) | reflex synthesis-lane agreement veto + echo gates (the judgment shape); refine `seat_rerank`/`escalation_dispatch` (abstain ⇒ modelless answer stands); instinct T2 certification | The below-floor contract already ships in `escalation_dispatch` (every non-answer is `StoodByModelless`); the T2 gate already certifies lanes against frozen reads. The delta is *longitudinal* (agreement drift across self-improvement cycles) — we run no multi-cycle self-improvement loop for it to gate. Consumer strength: none until a cycle loop exists (Plan 456 Wave 4 would create one — the gate is written into that plan's task row). |
| I3 | **Optimizer-vs-evaluator capacity attribution** as a standing ablation design | instinct manifest rows (per-role lane/capacity selection) + `escalation_dispatch` worthiness/cost ceiling | One-line design law adopted instead: **spend hosted/escalation budget on proposer-side capacity before judge-side** — the ablation's verdict, imported as posture, not as an instrument. Already how the ladder is arranged (seat rerank is a rerank, generation stays local). |
| I4 | **Modelless RSI substitution** (SFT → elite-trajectory sealed-corpus replay = "corpus is the model") | refine `LatentFixMemory`/`traj_store` replay + reflex sealed-corpus gates + `Lz4FlexDrafter` | Signal-diff performed, coverage HOLDS: the healer's Warm-tier trajectory store already persists outcome-linked fix trajectories and replays them at retrieval time — that IS elite-trajectory replay, with evidence tiers. Reflex's synthesis lane already builds sealed self-generated corpora with contamination defense. The composite's only missing piece is the evolutionary workflow producer, which is the LLM-dependent part (R440 class). No gap remains that a modelless primitive fills. |
| I5 | **Freeze/lineage of retained artifacts** (w* per cycle as BLAKE3-addressed frozen object, parent_commitment chain) | riir-neuron-db `MerkleFrozenEnvelope` v2 parent_commitment chain; instinct A2 monotonic hot-swap | Coverage holds — the freeze/thaw + lineage + monotonic-swap machinery is exactly this. Nothing new to build. |
| I6 | **Internalization audit grammar** (fixed probe set + deterministic counter + CI + paired gate) | reflex `semantic_defects` fixed-case shape; instinct frozen-fixture `serve_gates`; refine score-bench floors | The grammar (probe set → counter → CI → paired gate) is the standard instrument shape the workspace already uses everywhere. The paper's specific probe (delegation/handoff rates) has no consumer here. |

### 2.2 Kept at plan grade (training track → riir-train Plan 456)

The model-based advocate's 12 items, three waves, cheapest-first. Highlights with their GOAT gates:

- **W1 (recipe edits, ~2–8 GPU-h):** workflow-stripped prompts (#4 — the L4 0/60 direct attack; OWNER-GATED re-arm), AdamW β2=0.95 + 5-warmup + cosine-10% + val-loss checkpoint (#7), LoRA r64/α128 on quest_training vs shipped r32/α64 (#6).
- **W2 (data shaping, ~1–2 4090-days):** BT tournament top-K over the refine trajectories **already stored unmined on the Warm tier** (#5), window decomposition + masked-context rule (#2), 2:1 orchestrator oversampling vs natural frequency (#3).
- **W3 (line + transfer, ~1 4090-day):** fresh-adapter-on-merged never-stack rule (#8), token-matched multi-agent-vs-single-agent distill A/B reproducing the 1.4× law (#9), role-transfer measurement riding every wave's checkpoints for free (#10).
- **W4 (the loop, owner-gated):** the full alternation pilot (#1) + data-budget sweep (#11) + cadence rule (#12) — only after W1–W3 prove the recipes transfer to ternary/quest-scale lanes.

Honest transfer caveats carried into the plan: 49k-token windows do not fit a 24 GB 4090 (re-shape to 8–16k, masked-context rule preserved); Qwen3.6-27B-FP8 hyperparameters are hypotheses each gate tests, not facts to import; the paper's discovered workflows do not transfer (qwen-code ≠ our harnesses) — the recipes do.

### 2.3 Discarded (with auditable reasons)

| Advocate item | Discard reason (mechanism-level) |
|---|---|
| Hop-order bandit over candidate schedules; adaptive-M bandit tournament; stagnation-diversify drive | Generic search-control shapes; the workspace ships the bandit substrate (`katgpt-ruliology/bandit.rs`) and the drive pattern (CGSP no-progress), and has **no artifact-evolution loop** for them to control. A bandit with no consumer is not a filing. Reopen with Wave 4 if it fires. |
| Pareto rank-0 / Beta-LCB selection upgrades | Already shipped — instinct `stats.rs` IS Wilson bounds + paired non-inferiority + Pareto rank-0 + Beta-LCB (verbatim the advocate's proposed host). Coverage, not dismissal. |
| Structure-first mutation ordering; typed-contract handoffs | Design laws, adopted as one-line posture in §2.1 I3's spirit; no runtime surface here evolves r/u/c. |
| Token-efficiency accounting axis on score-bench | Consumed into Plan 456 W2/W3 gates (the A/B benches carry the token axis); no separate healer issue — the healer's fix lanes are not the paper's consumer and `--misses`/`--horizon-report` already do per-rule value accounting. |

## 3. Novelty gate (§1.5) — per track

**Modelless/serving track:** Q1 prior art — the loop class is dense (RHI = half (a), same first author; SEAL/SiriuS = half (b); ADAS/AFlow/GPTSwarm = workflow-graph search; the OTHER Mass 2502.02533 already PASS'd in R146) and the in-stack pieces ship (§2.1 I4/I5 coverage holds by signal-diff). Q2 new behavior class — no (the loop is a product loop for LLM-agent stacks). Q3 selling point — no. Q4 force multiplier — no. **Not Super-GOAT; PASS.**

**Training track:** the recipes are applicable and cheap (Path 0.5); the paper's own evidence (119 trajectories → +16 pt win rate; 1.4× token efficiency; internalization) is directly load-bearing for Plan 456's gates. **GAIN — plan filed.**

**Pinned one-sentence claim (§4 precondition, written before the searches ran):** *MASS's defensible novelty is the closed alternation — evolutionary multi-agent workflow optimization whose elite trajectories are BT-ranked and distilled with the workflow stripped from the prompt — plus two empirical laws (multi-agent trace token-efficiency; optimizer-capacity bottleneck); claim 1 (information concentration) is a component-resolved sharpening of RHI's already-published information-theoretic hypothesis, and the paper is the R440 code-level/LLM-dependent RSI class for our modelless ladder.*

## 4. Published prior-art search (§4 — agent-run, 12 searches + 4 abstract fetches, 2026-10-10)

| MASS claim | Closest published prior art | Residual delta |
|---|---|---|
| The alternation (workflow evolution + SFT on its own traces) | **RHI** arXiv:2607.15524 (same first author; harness/workflow revision ONLY, explicitly no weight training); **SIA** arXiv:2605.27276 (feedback-agent decides harness-vs-weights per iteration); **SEAL** (NeurIPS 2025; self-generated SFT, no workflow object) | The fixed alternation with BT-ranked elite traces + workflow-stripped prompts is **not published as a unit**; the halves individually are |
| Claim 1 (info concentration) | **RHI again** — publishes the information-theoretic hypothesis ("task-specific context management / inter-agent information flow") | MASS adds the 4-way component decomposition (hops/roles/instructions/contracts) + total-correlation measurement — a quantification upgrade, not a new direction |
| Claim 2 (trace token efficiency) | **SiriuS** arXiv:2502.04780 (multi-agent trace SFT, fixed workflow, no efficiency-vs-single-agent comparison); Long-Short Trajectory Learning (AAAI 2025) | The matched-token student-vs-student design appears **novel as found** |
| Claim 3 (optimizer-not-evaluator bottleneck; evaluator agreement 0.73→0.93) | Self-Rewarding LMs arXiv:2401.10020; CycleResearcher (ICLR 2025) | The capacity-attribution ablation + longitudinal judge-agreement curve appear **unpublished as found** |
| Name collision | "MASS" is taken by arXiv:2502.02533 (workflow-search-only; already PASS'd in R146's addendum) | Cite by arXiv ID, never by the acronym alone |

## 5. Verdict routing (files created)

- **This note** — `katgpt-rs/.research/616_MASS_Multi_Agent_Self_Supervision_RSI.md` (public; no rethink-class content).
- **riir-train Plan 456** — `riir-train/.plans/456_mass_recipe_extraction_waves.md` (private; the GAIN-track output; Wave 1's L4 re-arm is OWNER-GATED).
- **PASS-Redirects lines** appended to the 3 closest cousins: R440 (canonical RSI PASS), R146 (the other-Mass addendum holder), R289 (bi-level-already-shipped).

## 6. Adversarial panel record

Three-track panel run in one parallel spawn round (2026-10-10): No-GD advocate (32 extractions, all (i)/(ii)/(iii)-annotated), Model-based advocate (12 recipe items with GPU-h + gates; derived the recipe is a 2-epoch BT-top-15-of-144 design), web prior-art agent (§4). Coordinator merges above; discards audited in §2.3. Panel brief hygiene held: repos described by shipped file/type names, never by verdict conclusions.

## 7. PASS-Redirects

> **PASS-Redirects (synthesis):** Lee, Xu, Seely, Lee, Sojoudi, Zaharia, Tang [arXiv:2610.12176 "Recursive Self-Improvement through Multi-Agent Self-Supervision"] — homogeneous RSI alternating elite-1 evolutionary multi-agent workflow search with SFT on BT-ranked self-generated multi-agent traces (workflow stripped from the SFT prompt); inner-loop proposer is LLM text generation → THIS note's code-level/LLM-dependent RSI class (modelless track PASS); the SFT-recipe half is the actionable Gain → riir-train Plan 456; Claim 1 is a component-resolved sharpening of RHI arXiv:2607.15524.

## Re-evaluation guard

If you arrived here from a grep: the modelless/serving verdict is **PASS** (LLM-dependent proposer, R440 class; in-stack pieces covered by signal-diff in §2.1) and the training verdict is **GAIN, already filed** as riir-train Plan 456 — do not re-distill; extend the plan. Two reopen triggers: (a) a Wave-4-class loop actually boots in-stack (then I2/I3's longitudinal instruments gain consumers), (b) a descendant paper strips the LLM from the workflow PROPOSER (e.g. a latent/combinatorial search over structured workflow vocabularies judged by a frozen evaluator) — that would be a genuinely new modelless primitive and should be re-run through the full gate, not this note.
