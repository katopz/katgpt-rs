# Research 183: Lodestar — Completion-Distance Pruning

**Date:** 2026-06-07
**Status:** Proposed (GOAT verdict: PROCEED — modelless, perf-positive)
**Domain:** Modelless core (`katgpt-rs`) — engine. Fuel side lands in `riir-ai` (Research 072).
**Depends on:** `ConstraintPruner` (traits.rs), `build_dd_tree_pruned` (dd_tree.rs), SynPruner.
**Sibling work it composes with:** 177 Domino (prefix correction), 204 Selectivity Router (adaptive CoT), 202 RV Gated Compute Routing, 206 EGCS.

---

## 0. One-paragraph thesis

Every hard `ConstraintPruner` is implicitly a finite automaton over the token vocabulary
(SynPruner *is* a partial-parse DFA; Sudoku *is* a constraint automaton). For any such
automaton, precompute **one integer per state**: `d(s)` = the shortest number of tokens
needed to reach an *accepting* (complete/valid) state — the **shortest-accepting-distance**,
found once by reverse-BFS from accepting states. That single integer simultaneously powers
three things the literature treats as separate problems:

1. **Budget-aware masking** — prune any token whose successor state cannot still complete
   within the remaining token budget. (TRUNCPROOF's truncation guarantee.)
2. **Jump-ahead admissibility** — emit a deterministic singular-path span in one prefill
   step, but only when it fits the budget. (SGLang compressed-FSM speed.)
3. **Termination / convergence** — `d(s)` is a monotone non-increasing potential along any
   valid completion, so best-first expansion ordered by `d` provably terminates and reaches
   a complete valid output. (Knaster–Tarski / Banach fixed-point from the XML-prompting work.)

**The novel claim, made by no single source:** *min-completion-length (TRUNCPROOF) ≡
remaining lattice height (XML fixed-point) ≡ an admissible A\* heuristic.* They are the same
invariant viewed from three angles, so **one cheap precomputation** delivers a correctness
guarantee, a speedup, and a termination proof at once. We call this integer the **lodestar**
(the star you steer by): it is the distance-to-valid that guides DDTree expansion.

---

## 1. Link consolidation (the source material, distilled)

The brief supplied ~50 links across five clusters. Subagents fetched and consolidated them.
Below is the synthesis, organized by what each cluster contributes to the fusion.

### 1a. Constrained / guided generation — *the spine of Lodestar*

| Source | Mechanism | What Lodestar takes |
|---|---|---|
| **Domino — "Guiding LLMs The Right Way"** (arXiv 2403.06988) | Per-scanner-state vocab-aligned subterminal prefix trees; **minimal-invasiveness** proof; solves "bridge tokens". | Per-state token admissibility is precomputable; masking need not force worse tokenization. |
| **SGLang Compressed FSM** (LMSYS 2024-02-05) | Detect **singular transition edges**, compress into **singular paths**, **jump-forward** (prefill the whole deterministic span); re-tokenize string on jump. ~2× latency, 2.5× throughput. | Jump-ahead is the speed lever — Lodestar gates it on budget fit. |
| **Outlines** (Canoe) | Offline index: FSM state → permitted vocab; O(1) per-step lookup, prune tokens never on a valid path. | The distance table is built in the same offline pass as the index. |
| **TRUNCPROOF** (OpenReview lrc2xSoh9b) | LL(1) **min-tokens-to-complete** estimate; mask any token that cannot be grammatically completed within the remaining **token budget** → guaranteed valid+complete under a hard max-token limit. | This *is* budget-aware masking — and min-tokens-to-complete *is* `d(s)`. |
| **XML Prompting as Grammar-Constrained Interaction** (arXiv 2509.08182, Alpay & Alpay) | XML well-formedness as a **complete lattice** under refinement; monotone operators have stable fixed points by **Knaster–Tarski**; iterative guidance has **Banach-style convergence**. | `d(s)` = remaining lattice height ⇒ monotone refinement ⇒ termination proof. |
| **dev.to "Taming LLMs"** | **FSM-state checkpoint/resume across truncation** — serialize valid-next-token state, resume valid. | Lodestar state (current automaton node + budget) is serializable → valid streaming across budget boundaries. |
| **Shape AI — constrained SQL** | Inject runtime symbols (schema columns, enum values) as **grammar terminals** at build time → semantic validity enforced by the same DFA as syntax. | Domain symbols become automaton terminals (RIIR target grammar, game action enums). |
| **CodeAct** (arXiv 2402.01030) | Executable action space; semantic errors caught by the interpreter, not masking. | A post-hoc verifier layer (EGCS / cargo-check) catches what masking can't — Lodestar is the cheap syntactic front. |
| llama.cpp `--grammar` #2364 | (page had no technical content) | — |

**Cluster verdict:** the frontier has converged on (a) amortize the O(vocab) cost offline,
(b) jump-ahead on determinism, (c) guarantee minimal-invasiveness under tokenization. The
*unexploited gap* is fusing the **budget** axis (TRUNCPROOF) with the **jump-ahead** axis
(SGLang) and the **fixed-point** axis (XML) — which the single-integer lodestar does.

### 1b. Neuro-symbolic architecture — *the framing*

System-1 proposes, System-2 disposes; the symbolic side never trains the net, it gates /
validates / re-weights at inference (VeriPrajna schema-FSM masking, the PDDL planner
refusing infeasible goals, Kautz Type-6 tool router, game-theoretic equilibrium re-weighting
of completions, Ultralytics/Uplatz taxonomy). This is exactly our stack
(`logits → ConstraintPruner → DDTree → verify`). Two grounding ideas lifted directly: a
**blackboard** of authoritative world-state for the pruner to consult, and **prompt-side
feasibility summaries** to pre-shrink the branch space. The most ambitious cross-source idea
— *score a DDTree branch by the equilibrium value over its symbolically-feasible sub-actions*
— is noted as future work (Research 184 candidate); Lodestar is the prerequisite admissible
heuristic that makes best-first branch ordering principled.

### 1c. SLM / real-time NPC latency — *the routing extension (constraint #7)*

Hard immersion ceiling ≈ 1s end-to-end; best-in-class voice agents target **<500ms**.
Field answer is *spatial specialization by stage*: STT on-device/CPU (Moonshine 107ms),
LLM offloaded to a throughput backend (Cerebras 2,100+ tok/s, TTFT ~100ms). Two orthogonal
LLM-stage levers: architectural (Jet-Nemotron PostNAS, up-to-53× at long context — cost is
attention/KV, not the decode loop) and decode-loop (speculative decoding 2–3×, works *only*
because decode is memory-bandwidth-bound). **No source describes a load-aware policy.**
Lodestar contributes the missing controller signal: `d(s)` vs `budget_remaining` is a
*symbolic* estimate of remaining work, so the engine can route a low-`d` (almost-complete,
cheap) branch to the CPU floor and reserve the GPU for high-`d` long-horizon branches — and
fall back to a constraint-bounded CPU draft when the deadline controller predicts the GPU
path will miss budget. This is the CPU/GPU auto-route lever.

### 1d. Game design + classic agent architectures — *the action-space mapping*

The **paradox of choice / limiting player agency** is the design twin of constraint pruning:
a smaller curated action set yields faster *and better* decisions. FPS level-design patterns
(Hullett) show constraints should be **typed templates carrying an intended-effect
signature** (expected-vs-observed metrics = a built-in verifier), not arbitrary masks.
Classic AI gives the two-layer stack: an FSM/HTN **hard legality mask** under a
utility/capability-score **router** — which is precisely `ConstraintPruner` (mask) +
`ScreeningPruner` (graded) + DDTree (choose). Automated-testing/churn work supplies a
**verifier-free reward** (a small gradient-boosted churn/disengagement head scored on
trajectories), with the non-negotiable lesson of **concept drift** → refresh on a moving
window. *Relevance to Lodestar:* in the action-space (game) instantiation, `d(s)` = shortest
number of actions to a goal/terminal — the same integer guides branching-factor reduction
("limit agency") and supplies an admissible heuristic for MCTS/DDTree over actions.

### 1e. Safety / multi-agent / bias — *composition + budget economics*

**Configurable safety** (Inworld) = two composed pruners: an immutable base floor
(hate/self-harm, never disabled) under a hot-swappable topic-policy pruner whose active set
is a per-session config. **Blackboard** (Terrarium, Han & Zhang) = the shared, append-only,
content-driven coordination substrate where pruners post constraints and a factor-graph
router selects the next pruner from board state. **MAS-HQ "Cost of Knowing"** gives the
control law: every verifier call is a *costed* action; optimize `confidence_gain −
resource_penalty`, stop verifying when marginal hallucination removed no longer beats its
compute cost. FAIRGAMER (D_lstd) and YNTP (persona FSM) are additional pluggable pruners.
*Relevance to Lodestar:* the budget in "budget-aware masking" is exactly the MAS-HQ resource
budget; `d(s)` lets the router spend lookahead/verification only while a valid completion is
still reachable cheaply.

---

## 2. The fusion in detail

### 2.1 The object: an admissible completion-distance automaton

A `LodestarAutomaton` is `(States, δ: State × Token → State, Accept ⊆ States)`. From it we
precompute, once (offline / at construction), a distance vector:

```
d(s) = 0                                  if s ∈ Accept
d(s) = 1 + min_{t} d(δ(s, t))             otherwise   (∞ if no path to Accept)
```

This is single-source shortest path on the **reverse** automaton from the accepting set —
one BFS, O(|States| · |alphabet|). `d` is **admissible** (never overestimates true remaining
length) and **consistent** (`d(s) ≤ 1 + d(δ(s,t))`), so it is a valid A\* heuristic.

Most real grammars do not need an explicit per-token DFA: SynPruner already tracks a
parse state (bracket depth, expected-token class). We expose `d` through a thin trait method
so existing pruners opt in by returning their own shortest-to-valid estimate (e.g. bracket
depth = a lower bound on tokens-to-close).

### 2.2 The trait extension (SOLID: open for extension, no change to existing impls)

```rust
/// Admissible "distance to a complete, valid output" for budget-aware pruning.
/// Default impl returns 0 (no horizon info) — every existing ConstraintPruner
/// keeps working unchanged; Lodestar features are pure opt-in.
pub trait CompletionHorizon: ConstraintPruner {
    /// Lower bound on the number of additional tokens needed, from the state
    /// reached by `parent_tokens`, to reach a complete & valid output.
    /// MUST be admissible (never overestimate) for the budget guarantee to hold.
    fn min_completion_distance(&self, depth: usize, parent_tokens: &[usize]) -> u32 { 0 }

    /// Optional: length of the deterministic singular-path span from the current
    /// state (0 if the next step is a real branch). Enables jump-ahead.
    fn singular_span_len(&self, _depth: usize, _parent_tokens: &[usize]) -> u32 { 0 }
}
```

### 2.3 The three uses (one integer, three wins)

**(A) Budget-aware mask — the guarantee.** In `build_dd_tree_lodestar`, given
`budget_remaining = max_len − depth`, after the base `is_valid` check, additionally prune
candidate `t` when `1 + d(δ(s,t)) > budget_remaining`. Theorem (TRUNCPROOF, restated): every
branch the tree retains can be completed to an accepting state within `max_len` tokens — no
dead-ends, no truncated-invalid output. *Naive masking lacks this and can paint itself into a
corner under a tight budget.*

**(B) Jump-ahead — the speed.** When `singular_span_len = L > 0` and `L ≤ budget_remaining`,
emit the whole span as one node (one prefill) instead of `L` per-token mask steps (SGLang).
`d` along the span is strictly decreasing, so admissibility is preserved. If `L >
budget_remaining`, the span is inadmissible — Lodestar deliberately steers toward a shorter
completable branch rather than committing to a span it cannot finish.

**(C) Best-first ordering + termination — the proof.** Use `score' = score − λ·d(s)` as the
heap key (A\*-style: prefer high log-prob *and* low distance-to-valid). Because `d` strictly
decreases along accepted paths and is bounded below by 0, expansion terminates and the first
accepting node popped is budget-feasible. This is the Knaster–Tarski monotone-refinement
termination, made operational.

### 2.4 Adaptive-CoT and CPU/GPU routing (constraints #4, #7)

- **Adaptive CoT (self-learning, no training):** allocate tree/CoT budget proportional to
  `d(s₀)` (distance-to-valid at the root). Large `d` ⇒ think more; small `d` ⇒ short-circuit.
  A per-domain EMA bandit (reusing `pruners::bandit`) learns the multiplier — self-learning,
  inference-only. Composes with Plan 204 Selectivity Router as a *symbolic* selectivity
  signal (vs the entropy signal it already uses).
- **CPU/GPU auto-route:** feed `(d, budget_remaining, measured-bandwidth-pressure)` to the
  existing `inference_router` / Plan 202 RV gate: low-`d`/tight → CPU floor; high-`d`/slack →
  GPU; predicted-miss → constraint-bounded CPU fallback that still emits a valid-in-budget
  partial (the guarantee from (A) is what makes the fallback safe).

---

## 3. Novelty check (avoid duplication)

Grepped `katgpt-core`, `src/pruners`, and 188 research docs. Adjacent but distinct:

- **177 Domino** — *prefix-conditioned correction* of validity. Lodestar is orthogonal:
  forward *distance-to-completion*, not backward correction. They compose (Domino corrects
  `is_valid`; Lodestar bounds the horizon).
- **007 Screening / ScreeningPruner** — graded *relevance* of one token. Lodestar grades the
  *remaining path*, not the token.
- **206 EGCS / vr_loop** — episode-guided *synthesis* of new constraints + verify-refine.
  Lodestar is the cheap admissible front; EGCS is the semantic back. Complementary.
- **204 Selectivity Router / 202 RV gate** — *entropy/uncertainty* compute routing. Lodestar
  adds a *symbolic* distance signal those routers can consume.
- No existing code has `min_completion_distance`, budget-aware masking, jump-ahead, or a
  shortest-accepting-distance precompute. **Confirmed novel here.**

The cross-source equivalence (min-completion ≡ lattice height ≡ A\* heuristic) appears in no
single source — TRUNCPROOF has the budget bound, the XML paper has the lattice proof, SGLang
has jump-ahead, but none unify them on one precomputed integer.

---

## 4. GOAT verdict

**Gain — yes, on three measurable axes (proof plan in §6):**
1. **Correctness:** valid-AND-complete-within-budget rate. Naive masking truncates/invalidates
   a nonzero fraction under tight budgets; Lodestar → 100% by construction. (The headline
   metric — it directly hardens the "1-click RIIR that *compiles*" pitch: budgeted output is
   guaranteed a complete valid program, never cut mid-syntax.)
2. **Speed:** decode steps / tree nodes expanded. Jump-ahead collapses singular spans;
   dead-end pruning shrinks the heap. Expect fewer nodes for equal/za better acceptance.
3. **Quality:** A\* ordering reaches a valid completion in fewer expansions than greedy
   best-first under the same budget.

**Perf hurt — none expected.** `d` is a precomputed table; per-step cost is one O(1) lookup +
one comparison (branch-free `bool as usize`), per optimization.md. Jump-ahead strictly
*reduces* steps. Default trait method returns 0 ⇒ zero overhead and identical behavior for
every existing pruner that does not opt in. Feature-gated behind `lodestar`; isolate the
benchmark binary to avoid binary-bloat confounds (optimization.md).

**Decision:** PROCEED. After §6 GOAT+gain proof with no perf regression, promote to default-on
for pruners that implement `CompletionHorizon` (the bound only ever *adds* a guarantee; the
default-0 path is a no-op for the rest).

---

## 5. Commercial-strategy alignment (per Research 003)

Engine/fuel split stays intact:

| Layer | Lodestar piece | License |
|---|---|---|
| **Engine (open, MIT)** | `CompletionHorizon` trait, distance precompute (reverse-BFS), `build_dd_tree_lodestar`, jump-ahead, A\* ordering, adaptive-CoT/route hooks. Pure inference-time machinery. | MIT (`katgpt-rs`) |
| **Fuel (private, SaaS)** | The *automaton/grammar definitions* with domain symbols injected as terminals — RIIR target-Rust grammar, per-game action grammars, schema-aware column/enum terminals — compiled to `validator.wasm`. The distance tables are derived from these. | Private (`riir-ai`) |
| **Fuel (private)** | `lora.bin` semantic draft still required for *correct* marginals; Lodestar only guarantees *valid-in-budget*, not *semantically right*. "Ferrari needs gas" unchanged. | Private |

Lodestar *strengthens* the moat: the open engine now guarantees **complete, valid, in-budget**
output — a hard differentiator over "wrap GPT-4 and pray" competitors who truncate — while
semantic correctness still requires the private fuel. Model-based extension (proof-conditioned
LoRA warm-start from distance tables) is logged as `riir-ai` Research 072.

---

## 6. Proof plan (feeds Plan 207)

1. Standalone example `examples/lodestar_demo.rs`: a small balanced-bracket / mini-JSON
   grammar automaton. Compare under a tight token budget:
   - **Baseline** (naive `is_valid`-only masking): report truncation/invalid rate + nodes.
   - **Lodestar** (budget-aware + jump-ahead + A\* order): report valid-in-budget rate (target
     100%) + nodes (target lower) + steps saved by jump-ahead.
   - Print a before/after table: "thinking" (Lodestar lookahead) vs "non-thinking" (greedy).
2. Unit tests: admissibility (`d` never overestimates on a known automaton), monotonicity
   (`d` non-increasing along any accepted path), budget guarantee (no retained branch exceeds
   budget), jump-ahead correctness (span emission == per-token emission).
3. Micro-bench (isolated binary, optimization.md template): per-step overhead of
   `min_completion_distance` lookup vs `NoPruner` — confirm < ~50ns and no regression on the
   default-0 path.

---

## 7. References (consolidated from the brief)

**Constrained generation:** Domino arXiv:2403.06988 · SGLang Compressed FSM (LMSYS 2024-02-05)
· Outlines (guided generation) · TRUNCPROOF (OpenReview lrc2xSoh9b) · XML Prompting fixed-point
arXiv:2509.08182 · "Taming LLMs" (dev.to) · constrained SQL (Shape AI) · CodeAct arXiv:2402.01030.
**Neuro-symbolic:** VeriPrajna game-AI NS whitepaper · Ultralytics NS intro · Uplatz NS
integration taxonomy (Kautz types) · LLM-Reasoner+PDDL-Planner NPC arXiv:2501.10106 ·
Game-Theoretic Solvers for LMs arXiv:2402.01704.
**SLM / real-time NPC:** Jet-Nemotron / PostNAS (53× thread) · Moonshine + Cerebras NPC
pipeline · Inworld Contextual Mesh & gaming · Phi-3-Mini on-device (testgrid).
**Game design / agents:** Paradox of Choice (wayline) · FPS level-design patterns (Hullett
dissertation) · churn-without-players & churn training-set selection (U. Malta) · FSM+Utility
AI · Blackboard/event-bus · HTN in Unity3D · LLM agents for game testing.
**Safety / multi-agent:** chat moderation (GetStream) · NSFW API · Inworld Configurable Safety
· Terrarium blackboard arXiv:2510.14312 · LLM-MAS on blackboard · MAS-HQ "Cost of Knowing" ·
FAIRGAMER (chatpaper 182944) · YNTP arXiv:2510.14398.
