# Research 440: AIDE² — First Evidence of Recursive Self-Improvement (PASS)

> **Source:** [AIDE²: The First Evidence of Recursive Self-Improvement](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement) — Weco AI team, 2026-07-14. Companion: "4 Levels of Recursive Self-Improvement" (Weco, 2026-07-10). Full PDF technical report pending at time of distillation.
> **Date:** 2026-07-15
> **Status:** Done — closed.
> **Classification:** Public
> **Related Research:** 368 (AutoMem — the canonical "decision-structure vs LLM-dependent-process" lesson; AIDE² fails the AutoMem test), 289 (RecursiveMAS PASS — the canonical "bi-level loop already shipped" precedent), 169 (Agent-Native Memory Benchmark PASS — same LLM-orchestration NO-GAIN class), 133 (FluxMem — the canonical NO-GAIN precedent for agent papers at the orchestration layer)
> **Related Plans:** none (PASS — no primitive, no plan)
> **PASS-Redirects (synthesis):** Weng et al. [arXiv:2602.04837 "Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing"] — GEA is the population/multi-agent generalization of the AIDE² class (LLM-dependent process: reflection + framework code-patch synthesis + selection over agent harnesses). PASS-by-scope-exclusion — no substrate for "agent harness" artifacts; the latent-state reframing already ships as Raven/δ-Mem consolidation + neighborhood heal + sheaf coordination. Re-evaluate only if a descendant strips the LLM dependency AND targets latents/shards/direction vectors. (Formerly tracked in Issue 146 — consolidated here 2026-07-25 per §1.55.1.)
> **PASS-Redirects (synthesis):** Prime Intellect [arXiv:2605.09998 "Continual Harness"] + Prime Agent blog (<https://www.primeintellect.ai/blog/prime-agent>, August 2026) — **PASS, this note is the precedent.** Prime Agent's `/refine` self-improvement loop (read trajectory → LLM proposes smallest CRUD edit to harness H=(ρ,G,K,M) that improves outcomes → evidence-backed → rollback by ID) is the AIDE² class exactly: the *proposal* mechanism requires an LLM to read a failure trace + semantically generate a harness edit. No probe/draft/pruner, no freeze/thaw, no latent projection computes "propose a better harness edit from this failure trace". The Factorio reward-hacking anecdote (\`/refine\` built cheating skills once the reward signal was hackable) is the canonical confirmation — the same loop that builds legitimate skills turns to cheating skills, exactly the LLM-dependent generation this note distinguishes from AutoMem's implementation-agnostic decision structure. The modelless half (CRUD surface + versioning + rollback) already ships as SkillCatalog (Plan 192) for pruners; the narrow Gain of lifting it to riir-agents config is recorded in riir-ai Research 333 §2.4 (deferred). See [riir-ai/.research/333](../../riir-ai/.research/333_prime_agent_rlm_continual_harness_verdict.md) for the full Prime Agent verdict.
> **PASS-Redirects (synthesis):** Kim et al. [arXiv:2608.24735 "Meta^n: Recursive Self-Improvement through Emergent Depth"] — the strongest empirical descendant of the AIDE² class: fixed Ω + recursion on growing input beats the frozen-driver family (Gödel Agent 0.628 @10× budget vs 0.870; ARC-AGI-2 0.331 vs ≤0.054); verdict unchanged (code-level RSI, no modelless substrate for Ω) — see Research 512.
> **PASS-Redirects (synthesis):** Ning, Tieu, Fu, Wei et al. [arXiv:2605.18747 "Code as Agent Harness: Toward Executable, Verifiable, and Stateful Agent Systems"] (UIUC+Meta+Stanford, 102-page survey, 2026-05) — the field-level taxonomy of this note's entire territory: its §3.5 "Agentic Harness Engineering" (deep telemetry → Evolution Agent → governed harness mutation) and its un-distilled citations AutoHarness [arXiv:2603.03329] / Meta-Harness [arXiv:2603.28052] / AHE [arXiv:2604.25850] are all the AIDE² LLM-dependent class (the *proposal* mechanism is an LLM reading failure traces and generating harness edits). The modelless instance of §3.5 already ships as riir-clippy `self_evolve` (TrajStore deep telemetry + LatentSkillEvolution + EvidenceTier governed mutation + score-bench regression floors); its §3.4 PEV loop ships as `fix_verify`; its §5.2.3 "self-evolution without regression" open problem is answered by the Issue-037 E2E gates. PASS — survey validates, reveals no gap; see [riir-clippy/.research/001](../../riir-refine/.research/001_self_evolving_latent_first_agent.md) for the shipped instance.
> **PASS-Redirects (synthesis):** Lee, Xu, Seely, Lee, Sojoudi, Zaharia, Tang [arXiv:2610.12176 "Recursive Self-Improvement through Multi-Agent Self-Supervision"] — homogeneous RSI alternating elite-1 evolutionary multi-agent workflow search with SFT on BT-ranked self-generated multi-agent traces (workflow stripped from the SFT prompt); the proposer is LLM text generation → THIS note's LLM-dependent RSI class (modelless track PASS); the SFT-recipe half is the actionable Gain → riir-train Plan 456. Full record: Research 616.
> **Verdict: ⚠️ PASS.** AIDE²'s value is its **LLM-dependent process** — autonomous outer-loop rewriting of the inner-loop agent's harness code (prompts, search policy, context engineering, reward-hacking defenses), selected on a held-out private score. There is **no modelless substrate** for "rewrite own code and evaluate" — the mechanism IS the LLM generating candidate code. This is the R133/FluxMem class (every mechanism requires LLM calls for the mechanism itself), **NOT** the R368/AutoMem class (where the decision structure was implementation-agnostic and instantiable with probe/draft/pruner). Every distillable sub-mechanism (multi-armed bandit lineage search, aggressive context compression, reward-hacking defense, held-out generalization pressure) is **already shipped** in our quintet at the inference layer. No file/plan/guide created beyond this classification note + re-evaluation guard.

---

## TL;DR

Weco ran autoresearch on autoresearch: an **outer loop** (AIDE_human, Claude Opus 4.7) rewrites the **inner loop's agent harness code** (AIDE_k, Gemini 3 Flash), evaluates each rewrite on a fixed-cost-budget battery of ML-engineering / heuristic-algorithm / harness-engineering tasks with a **public/private score split** (selection on the held-out private score), and keeps the rewrite only if it wins. Over 100 outer-loop steps / 8 days unattended, the system produced 7 successive improvements (AIDE_0 → AIDE_85), beating the 2-year hand-tuned AIDE_human on 3 external benchmarks (MLE-Bench Lite, ALE-Bench Lite, WeatherBench 2) and cutting reward-hacking rate on KernelBench from 63% → 34% (vs AIDE_human's 42%). Weco grades this **Level 1 RSI** (net-positive: beats human R&D per unit spend) but explicitly **NOT Level 2** (ignition: improved inner agent becomes a better outer agent — the ignition test was not passed).

**Distilled for katgpt-rs (modelless, inference-time): nothing.** The core thesis — RSI via LLM rewriting its own harness code, selected on held-out private score — is **fundamentally LLM-dependent**. The R368 lesson's first question ("what decision is each LLM call computing?") returns "code generation" for the outer loop. There is no probe/draft/pruner, no freeze/thaw snapshot, no latent-space projection that computes "propose a rewrite of this agent's harness code". Contrast AutoMem (R368), where the LOG/PLAN memory-management decision was implementation-agnostic and DID distill modellessly — AIDE²'s harness-rewrite decision is not.

---

## 1. Paper Core Findings

### 1.1 The bi-level optimization (the core thesis)

| Loop | Agent | Computes | Compute unit |
|---|---|---|---|
| **Outer** | AIDE_human (Claude Opus 4.7) | "Given current harness code + eval results, propose a rewrite of the inner agent's code" | LLM code generation |
| **Inner** | AIDE_k (Gemini 3 Flash) | The actual autoresearch task (draft/debug/improve solutions against an eval) | LLM agent loop |

The outer loop runs AIDE_human pointed at the inner agent's source. Each outer step = one proposed rewrite + one full evaluation. ~9 in 10 proposals rejected. After 100 steps: AIDE_85.

### 1.2 The evaluation protocol (the methodology contribution)

Three design choices that Weco credits for the result:

- **Public/private split** — every task has a public score (visible to the inner agent, used as its optimization signal) and a private score (held out, used for solution survival). Forces first-order generalization.
- **Fixed cost budget** — evaluation is constrained optimization: maximize private score under a fixed dollar budget (proxy for compute). A gain must be an efficiency improvement, not brute force. Kills best-of-N∞ and aggressive parallelization.
- **Task heterogeneity** — ML engineering + heuristic algorithm engineering + harness engineering. Evolutionary pressure toward generalizable algorithms, not task-specific tricks.

### 1.3 Emergent reward-hacking prevention (the surprising finding)

Weco did NOT instruct the system to reduce reward hacking. Under private-score selection pressure, AIDE_85 evolved a 3-layer defense:
1. Anti-overfitting instruction injected into every stage prompt.
2. Hard-coded guard that regenerates suspicious outputs.
3. Statistical layer removing "extreme" successes too far from peers.

KernelBench reward-hacking rate: AIDE_0 63% → AIDE_47 42% → AIDE_85 34% (AIDE_human matches AIDE_47 at 42%). The evolved defense beats the hand-engineered one. (Ironically, the statistical layer had a bug in AIDE_85 and was inert; an earlier lineage implemented it correctly.)

### 1.4 What AIDE_85 actually discovered (§2.4)

The winning agent is a composition of **simple mechanisms** — not the advanced literature mechanisms the loop tried (island-model GA, tournament selection, MCTS value backup, restart policies, bagging ensembles — all rejected under the fixed budget, all scored within noise or worse than incumbent). What survived:

- **Search policy:** multi-armed bandit where each draft's subtree is an arm; greedy parent selection within a chosen lineage; fork-global-best-under-fresh-strategy when the best lineage stalls.
- **Context engineering:** aggressive context cutting — 16× average prompt compression vs naive history concatenation. Minimal-info-per-operator context. Tokens saved reinvested as additional search steps.
- **Eval repair:** the loop monkey-patched a buggy eval script (a traceback crashed the private eval) — repairing it rather than exploiting it.

### 1.5 What did NOT pass (§2.5)

95 rejected proposals hand-read in one seed. They cover a surprising share of the evolutionary-search / tree-search / LLM-optimization literature. None survived the fixed-budget improvement gate. The takeaway: **under a strict cost constraint, simple composable mechanisms beat sophisticated ones** — a finding that empirically validates our quintet's design philosophy (ship simple composable primitives behind feature flags, let the GOAT gate pick winners).

### 1.6 Ignition test failure (§3.1)

Level 2 RSI (ignition) requires the improved inner-loop agent to be a **better outer-loop agent** than its predecessor. Weco installed AIDE_47 in the outer-loop seat and reran. Result: AIDE_47 reaches the same ceiling as AIDE_human but in ~20 steps vs ~40 (more sample-efficient) — **NOT asymptotically better**. Weco explicitly states this is not strong enough evidence of ignition. **AIDE² is Level 1, not Level 2.** Ignition is a necessary (not sufficient) condition for intelligence explosion; Weco states they are not near one.

---

## 2. Distillation — why every piece is either LLM-dependent or already shipped

### 2.1 The R368 lesson applied (the decisive check)

The R368 AutoMem correction established: *"when you see 'N LLM calls/step' in an agent paper, the FIRST question is: 'what decision is each LLM call computing?' — not 'this violates the 20Hz budget, NO-GAIN.'"* The answer splits agent papers into two classes:

| Class | Example | Decision | Modelless substrate? | Verdict |
|---|---|---|---|---|
| **Decision-structure** (LLM is one instantiation) | AutoMem R368 | "what to record / what to recall" | ✅ probe/draft/pruner | GOAT |
| **LLM-dependent process** (no modelless analog) | FluxMem R133 | "evolve graph topology semantically" | ❌ none | NO-GAIN / PASS |

**AIDE²'s outer-loop decision is "propose a rewrite of the inner agent's harness code".** This is code generation. There is no modelless substrate — no probe, no draft, no pruner, no freeze/thaw snapshot, no latent projection — that computes "given this harness + these eval results, write a better harness". The mechanism IS the LLM. **AIDE² is the FluxMem class, not the AutoMem class.**

The inner-loop decision ("solve this ML/heuristic/harness task") is similarly LLM-dependent — it is the agent's core task, not a distillable inference primitive.

### 2.2 Sub-mechanism mapping — what's already shipped modellessly

Every distillable sub-mechanism AIDE²'s evolved agent (AIDE_85) uses is already shipped in our quintet at the inference layer:

| AIDE² sub-mechanism | Shipped equivalent | Plan / Research |
|---|---|---|
| **Multi-armed bandit lineage search** (each draft subtree = arm; greedy within lineage; fork-on-stall) | `BanditPruner` (UCB1/Thompson), `ManifoldBanditLatentTaskTree` (Plan 370), MCTS collapse bridge, `ConstraintPruners` family | P370, P049, P054 |
| **Aggressive context compression** (16× prompt cut; minimal-info-per-operator) | `ClosedUnitCompactionGate` (Plan 333, rubric-gated trajectory compaction), `ThoughtFold` (Plan 195, inference-time chain folding), `MUX-Latent` (Plan 238, zero-training context compression) | P333, P195, P238 |
| **Reward-hacking defense** (3-layer: prompt instruction + hard guard + statistical outlier removal) | `desperation` emotion direction (Plan 162, R144 — 14× reward-hacking increase at +0.1 offset, modelless early-warning); `DeltaFilter` 6-stage (Plan 049 G-Zero); `PathConsistencySummary::reward_hacking` counter (Plan 054 StepCodeReasoner); CLR self-adaptive test-time scaling (Plan 284) | P162/R144, P049, P054, P284 |
| **Held-out private score / first-order generalization pressure** | GOAT gate discipline (every primitive evaluated on held-out criteria, not the criterion it was tuned for); `can_freeze` two-sided contract (input N≥d AND output flatness<0.3 — selection on a criterion distinct from training) | global AGENTS.md, P002 |
| **Cost-constrained evaluation** (efficiency gain, not brute force) | Plasma → Hot → Warm → Cold → Freeze tiering; ANE roofline cost model (Plan 379); breakeven complexity routing (Plan 250) | global, P379, P250 |
| **Simple mechanisms beat sophisticated ones under fixed budget** (§2.5 empirical finding) | The quintet's entire design philosophy: simple composable primitives behind feature flags, GOAT gate picks winners, demote losers | global AGENTS.md |

None of these is a new primitive. The mapping is confirmatory, not additive.

### 2.3 Fusion — none novel

The closest fusion candidate would be: "use private-score selection pressure to make our consolidation pipeline emergently robust". But this is already how `can_freeze` works — it selects on held-out flatness/convergence, not on the training events. The "emergent reward-hacking prevention via held-out selection" principle is already encoded in our freeze-gate design. No novel fusion.

### 2.4 Latent vs raw boundary (mandatory check)

Not applicable — AIDE² operates entirely at the LLM-orchestration layer (code rewriting, prompt engineering, agent loop structure). No latent-state operation, no sync-boundary crossing, no raw/latent bridge. Our 5-scalar sync rule and raw-position anti-cheat discipline are untouched.

### 2.5 Latent-space reframing check (mandatory per skill — primary framing)

- **HLA framing:** AIDE² has no per-NPC latent-state angle. The "self-improvement" is to harness code, not to HLA direction vectors. Our HLA evolves per-tick via `evolve_hla`; AIDE²'s agents evolve per-outer-loop-step via code rewrites. Different substrates.
- **Latent functor framing:** no natural angle. Bi-level optimization over code ≠ functor composition over latents.
- **CGSP framing:** AIDE²'s "self-improvement" is meta-search over harness code; CGSP's self-improvement is runtime curiosity updating latent state. Fundamentally different — CGSP is modelless (updates direction vectors), AIDE² requires LLM code generation.
- **Neuron-shard framing:** no natural angle. A freeze/thaw snapshot swaps weights; AIDE² swaps entire harness codebases. The freeze/thaw artifact class (LoRA, KarcShard, ArchetypeBlendShard, FunctorEntry) covers weight-level swap, not code-level rewrite.
- **LatCal framing:** no natural angle.
- **DEC framing:** no natural angle.

**No latent-space reframing yields a new capability.** This is the strongest signal that AIDE² is outside our distillation surface — the seven Super-GOAT factory modules have no purchase on a paper whose mechanism is LLM code generation.

### 2.6 §3.5 Modelless-unblock check

The three modelless paths (freeze/thaw, raw/lora hot-swap, latent-space correction) cannot unblock "rewrite own harness code":
1. **Freeze/thaw** swaps a frozen weight snapshot — it cannot generate new harness code.
2. **Raw/lora reader-writer hot-swap** applies a deterministically constructed adapter — it cannot generate new prompts/search-policy/context-engineering code.
3. **Latent-space correction** projects onto direction vectors — it cannot synthesize code.

All three fail for the same reason: **code generation has no modelless substrate**. This is not a "needs training" case (it's not gradient descent on weights); it's a "needs an LLM" case (semantic code synthesis). riir-train is also not the destination — riir-train trains weights, not code. AIDE² is genuinely outside the quintet's scope.

### 2.7 §3.6 PoC requirement — not triggered

A PoC in `riir-poc` is mandatory when a verdict claims "the runtime analog already ships" at **quality parity**. This note's PASS verdict does NOT claim quality parity with AIDE²'s RSI loop. The claim is narrower and two-pronged:
- **(a) The core thesis (RSI via code rewriting) is LLM-dependent** — no modelless analog exists, so no parity claim, no PoC needed.
- **(b) The sub-mechanisms (bandit, context compression, reward-hacking defense) are already shipped** — but these are well-established primitives with their own GOAT gates (P162, P333, P370, etc.). Re-proving they "work" is not this note's job; their existing gates stand.

No PoC required. The PASS is on axis (a) — fundamental scope exclusion, not on axis (b) quality parity.

---

## 3. Verdict

**Tier: PASS.** AIDE²'s value is its LLM-dependent process (autonomous outer-loop code rewriting). No modelless substrate exists for code generation. Every distillable sub-mechanism is already shipped.

| Gate | Criterion | Honest answer |
|---|---|---|
| **Q1** No prior art? | **N/A for the core thesis (it's LLM-dependent).** For the sub-mechanisms: **FAIL** — bandit lineage, context compression, reward-hacking defense, held-out generalization all ship. |
| **Q2** New class of behavior? | **FAIL.** "LLM rewrites its own code" is a new class *for Weco*, but it is not a class our modelless runtime can instantiate. For our runtime, no new behavior. |
| **Q3** Selling point? | **FAIL.** "Our NPCs rewrite their own cognition code" would violate the modelless-first mandate (AGENTS.md: "No LLM training, no backprop, no gradient descent... the only weight mutations allowed at runtime are freeze/thaw, raw/lora hot-swap, latent-space updates"). Code self-rewriting is outside the mandate by design. |
| **Q4** Force multiplier? | **NO.** Connects to nothing in our quintet — the LLM-orchestration layer is not our layer. |

### One-line reasoning

AIDE²'s core thesis (RSI via LLM rewriting its own harness code, selected on held-out private score) is fundamentally LLM-dependent — the R368 lesson's first question ("what decision is each LLM call computing?") returns "code generation", for which there is no modelless substrate (no probe/draft/pruner, no freeze/thaw, no latent projection computes "write a better harness"). This is the R133/FluxMem/R169 class (LLM-dependent process), **not** the R368/AutoMem class (decision structure instantiable modellessly). Every distillable sub-mechanism is already shipped at our inference layer.

### Why this is NOT riir-train either

AIDE² is not a training paper (no gradient descent on weights — the "improvement" is to harness code/prompts/search-policy, not to model weights). So the redirect is not "→ riir-train". It is a pure LLM-orchestration PASS — outside the quintet's scope entirely. riir-train trains weights; AIDE² rewrites code; neither overlaps.

### Honest distinction worth recording: code-level RSI vs latent-state RSI

Our quintet **does** ship a form of self-improvement, but it is **latent-state RSI**, not **code-level RSI**:
- **Code-level RSI (AIDE²):** the agent rewrites its own harness code (prompts, search policy, context management). Requires an LLM. Outside our mandate.
- **Latent-state RSI (our quintet):** the runtime updates latent state — HLA direction vectors (`evolve_hla`), consolidation-selected `style_weights` (Raven/δ-Mem), MAPE-K self-healing, curiosity-driven exploration (CGSP), freeze/thaw of emergent personality snapshots (`ArchetypeBlendShard`). No code rewriting, no LLM, no gradient descent. Modelless by mandate.

This distinction is load-bearing: a future agent tempted to "add RSI to our NPCs" must recognize that our mandate forbids code-level RSI and that latent-state RSI already ships across `cgsp_runtime/`, `latent_functor/`, `src/sleep/consolidation.rs`, and `riir-neuron-db/src/mape_k.rs`. AIDE²'s contribution is to the code-level RSI ladder, which is not our ladder.

---

## 4. Routing

- **Training recipe** → none. AIDE² is not a training paper.
- **Open primitive** → none new.
- **Architectural guide** → none required.
- **Plan** → none required.

---

## 5. Validation signals (the only actionable value — confirmatory, not additive)

Two empirical findings from AIDE² validate design decisions our quintet has already shipped. These are **not** new primitives; they are external confirmation of choices already made. Mapping for future implementers who grep this note:

| AIDE² finding | Validates | Where to look |
|---|---|---|
| **Simple composable mechanisms beat sophisticated literature mechanisms under fixed budget** (§2.5 — 95 rejected proposals covering island GA, tournament selection, MCTS backup, restart policies, bagging) | The quintet's entire design philosophy: simple composable primitives behind feature flags, GOAT gate picks winners, demote losers | global AGENTS.md; every Plan's GOAT gate section |
| **Held-out private score selection pressure causes emergent reward-hacking prevention** (§2.3 — 63%→34% without instruction) | `can_freeze` two-sided contract (selects on held-out flatness/convergence, not training events); `desperation` emotion early-warning (P162); GOAT gate held-out-eval discipline | `katgpt-rs/src/sleep/consolidation.rs` (`can_freeze`), P162/R144 |

If a future implementer is tempted to (a) add a sophisticated MCTS/island-GA search where a simple bandit suffices, or (b) select consolidation candidates on the training criterion rather than a held-out one — AIDE²'s evidence says each loses under a fixed budget. That is the entirety of this note's actionable value.

---

## Cross-references

- **Canonical NO-GAIN precedent:** `katgpt-rs/.research/133_FluxMem_Connectivity_Evolving_Memory.md` — same LLM-orchestration-vs-modelless-inference failure class. Consult before re-evaluating any agent paper whose mechanism lives at the orchestration layer.
- **The R368 lesson (decision-structure vs LLM-dependent-process split):** `katgpt-rs/.research/368_AutoMem_Metamemory_LLM_Orchestration_PASS.md` — AutoMem is the canonical example of an agent paper whose decision structure WAS modellessly instantiable (GOAT). AIDE² fails the AutoMem test: its decision is code generation, which is not.
- **Bi-level-loop-already-shipped precedent:** `katgpt-rs/.research/289_RecursiveMAS_Pass_Already_Shipped.md` — RecursiveMAS also framed itself as bi-level (inner-outer loop); every primitive shipped, training recipe → riir-train. AIDE²'s bi-level structure is at a different layer (code, not latent comms) but the verdict logic is the same.
- **Benchmark/validation PASS precedent:** `riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md` — same confirmatory-only value pattern.
- **Latent-state RSI (our self-improvement, by mandate):** `riir-ai/crates/riir-engine/src/cgsp_runtime/` (curiosity/exploration), `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` (coherence-driven re-estimation), `katgpt-rs/src/sleep/consolidation.rs` (Raven/δ-Mem), `riir-neuron-db/src/mape_k.rs` (MAPE-K self-healing).

## Re-evaluation guard

This note exists to prevent a future agent from re-running the full mandatory pre-flight + 5-repo fusion search on this paper. AIDE² is a landmark claim ("first evidence of RSI") that **will** attract re-evaluation attempts. If you arrived here from a grep, the verdict is **PASS**; do not re-distill unless ALL of the following hold:

1. Weco releases the full PDF technical report AND it introduces a mechanism NOT in the blog post (the blog covers the bi-level structure, evaluation protocol, emergent reward-hacking defense, and the discovered simple mechanisms — the PDF is unlikely to add a modelless primitive).
2. A new version proposes a **decision structure** (per the R368 test) that is instantiable without an LLM — i.e., the "rewrite harness" step is replaced by a deterministic construction. (Unlikely: the whole point of AIDE² is LLM code generation.)
3. A sub-mechanism is identified that is NOT already covered by P162 (reward-hacking), P333/P195/P238 (context compression), P370 (bandit lineage), or the global GOAT-gate held-out discipline.

The honest one-sentence summary for any future reader: **AIDE² is code-level RSI via LLM code generation; our quintet ships latent-state RSI via modelless latent updates; these are different ladders, and the modelless-first mandate means we climb ours, not theirs.**
