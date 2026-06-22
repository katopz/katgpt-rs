# Research 283: FastContext — Exploration Subagent for Coding Agents

> **Source:** [FastContext: Training Efficient Repository Explorer for Coding Agents](https://arxiv.org/abs/2606.14066) — Microsoft + SJTU, Jun 2026
> **Date:** 2026-06-22
> **Status:** Done — Gain, no plan
> **Related Research:** 191 (Prism SubstrateGate), 218 (Breakeven Router), **281 (SalienceTriGate — closest prior art, "Delegate" arm)**
> **Related Plans:** 216 (SubstrateGate, default-on), 250 (Breakeven routing, default-on), 303 (SalienceTriGate — async delegation contract), 330 (riir-ai NPC salience gate runtime)
> **Classification:** Public

---

## TL;DR

**Gain — dev workflow win, weak research angle.** FastContext is a runtime delegation contract: separate repository exploration (a 4B–30B specialized subagent with READ/GLOB/GREP + parallel tool calls) from the main solver, returning file-path + line-range citations as compact context. Up to 60% main-agent token reduction, up to 5.5% accuracy gain on SWE-bench Multilingual/Pro/SWE-QA. The research value for our codebase is thin: the "specialized exploration subagent" pattern is **already shipped three ways** (SubstrateGate R191, Breakeven Router R218, and most directly SalienceTriGate R281's `Delegate` arm). Worth a one-line note + a `microsoft/fastcontext` tool trial on this codebase; not worth a plan.

**Distilled for katgpt-rs (modelless, inference-time):**
Nothing new to distill. The transferable primitive — *asymmetric delegation: a side process burns its own turns on broad search, only the compact evidence folds back to the main trajectory* — is already the `SalienceTriGate::Delegate → foldback_target` contract in Plan 303 / riir-ai Plan 330. FastContext's only novel ingredient is the *trained explorer policy* (SFT on parallel-tool-call decomposition + GRPO with file/line F1 reward), which is a **training** contribution → belongs in `riir-train`, not here.

---

## 1. Paper Core Findings

- **Mechanism**: Main agent invokes `fastcontext -q "..." --format concise`; subagent runs READ/GLOB/GREP with parallel tool calls for up to N turns, returns a `<final_answer>` block of `path:start-end` citations. Subagent's internal trajectory is *not* appended to the main agent's context — only the final citation list is.
- **Training recipe** (the paper's actual contribution):
  - **SFT corpus** (2,954 examples, Sonnet 4.6 traces), 3 splits: `parallel_toolcalls` (990, broad first-turn), `multiturn_traj` (983, evidence gathering), `linerange` (981, citation emission).
  - **RL refinement** (GRPO, 400 prompts), reward = `file_F1 + line_F1 + r_parallel − r_format`. Bonus for 3–6 parallel calls per turn; penalty for >20 citations, broken lines, >6 parallel calls.
- **Headline numbers**: GPT-5.4 + FC-4B-RL on SWE-QA → 60.3% main-token reduction at +0.1 accuracy. On SWE-bench Pro → +5.5 accuracy at −14.1% tokens. 4B-RL beats 30B-SFT on several combos (RL > scale for compact explorers).
- **Cost audit**: 4B-RL subagent = 22.58M tokens / 300 tasks = $4.52 (Fireworks $0.20/1M). Main-agent saving $69.03 net. Explorer = 2.1% of augmented cost.

## 2. Distillation

### Why the research angle is thin for us

The "delegation contract" half is the modelless half, and we already ship it:

| FastContext concept | Our shipped prior art | Status |
|---|---|---|
| Asymmetric delegation (subagent burns own turns, folds back compact result) | **R281 SalienceTriGate `Delegate` arm** + `foldback_target` (Plan 303 katgpt-rs, Plan 330 riir-ai) | Super-GOAT, shipping |
| Capability-routed dispatch to specialized substrate | **R191 SubstrateGate** (Plan 216, default-on, GOAT 7/7) | Default-on |
| Cost-aware tier routing (when to escalate to expensive solver) | **R218 Breakeven Router** (Plan 250, default-on, GOAT 7/7) | Default-on |
| Parallel tool calls in one turn | `spawn_agent` + parallel tool-call batching (this very workflow ships it) | Always-on |

### What FastContext adds that we don't have

Only the **trained explorer policy** — and that is a training contribution:
- SFT corpus construction from reference-model traces (3 split types).
- GRPO with patch-derived file/line F1 reward.

Per skill rules: training recipes → `riir-train`. No distillation target in the modelless/runtime repos. A hypothetical riir-train note would frame this as "task-grounded RL for a coding-agent explorer subagent" — but that's a one-liner there too, because the recipe is benchmark-specific (SWE-bench patch labels) and not obviously transferable to our 5-repo commercial product (we don't ship a coding agent).

### Latent-space reframing (mandatory per skill)

The latent-space re-cast is degenerate: FastContext operates entirely in **token space** (issue text → file paths → line ranges). There is no latent subspace, no HLA state, no functor application, no LatCal commitment. The only "latent" angle is trivial: the explorer's hidden state could in principle be cached as a per-repo embedding — but that's just retrieval, already covered by RAVEN/AnyRAG/shard retrieval. **No Super-GOAT angle exists.** Defaulting to the latent reframing confirms the Gain verdict (not Super-GOAT), per the §1.5 novelty gate Q3 ("can you finish 'Our NPCs/systems do X that no competitor can'?"). Answer: no — this is a coding-agent tool, not an NPC/game/chain/shard system.

## 3. Verdict

**🥉 Gain** — dev-workflow win for *us as coding-agent users* (this very agent could in principle be backed by a FastContext-style explorer); weak research angle for the modelless/runtime/chain/shard repos. The user's classification is correct and slightly generous — I'd argue Pass on the research axis and Gain only on the dev-tooling axis, but the user's combined "Gain" tier is defensible.

**One-line reasoning**: Delegation pattern already shipped three ways (R191, R218, R281); only novel ingredient is a benchmark-specific RL training recipe → riir-train territory; no latent-space Super-GOAT angle exists.

### Tiers

| Tier | Criteria | Routing |
|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + ≥2-pillar force multiplier | n/a — latent reframing is degenerate |
| GOAT | Provable gain over existing approach | n/a — delegation already ships as R281 `Delegate` |
| **Gain (assigned)** | Useful but not headline-worthy | One-line note (this file). No plan. |
| Pass | Not relevant | n/a — dev-tooling value keeps it at Gain |

### Recommended follow-up (not in scope for this session)

1. **Try `microsoft/fastcontext`** on this 5-repo codebase as a dev-workflow experiment. Particularly interesting test: does FC-4B-RL beat this agent's built-in `spawn_agent` exploration on cross-repo fusion grep (the §Workflow fusion protocol step 1)? If yes, that's a *tooling* win for us, not a research win.
2. If we ever want to ship a coding-agent product on top of katgpt-rs, the trained-explorer recipe belongs in `riir-train/.research/NNN_fastcontext_explorer_rl.md`. Not relevant today.

---

## TL;DR

Gain. Exploration-subagent delegation pattern already ships three ways in our stack (R191 SubstrateGate, R218 Breakeven Router, R281 SalienceTriGate `Delegate` arm). FastContext's only novel ingredient is a benchmark-specific RL training recipe for the explorer policy → riir-train territory, not katgpt-rs/riir-ai/riir-chain/riir-neuron-db. Latent-space reframing is degenerate (paper operates entirely in token space). Worth trying `microsoft/fastcontext` as a dev tool on this codebase; not worth a plan.
