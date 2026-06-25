# Research 310: RIZZ × Latent Substrate — Non-Interference Memory Branches for Continual Adaptation (Super-GOAT Fusion)

> **Source:** [RIZZ: Routing Interactions to Near Zero-Interference Zones for Continual Adaptation of Black-Box Agents](https://arxiv.org/pdf/2606.20638) — Goel, Vaidhyanathan, Schorling, Ares, Osborne (Oxford Engineering Science), 2 Jun 2026, arXiv:2606.20638 [cs.AI]
> **Date:** 2026-06-26
> **Status:** Active — Super-GOAT (fusion)
> **Classification:** Public — open synthesis (no game IP, no chain IP, no shard IP)
> **Related Research:** 209 (BAKE — continual KG embedding learning), 264 (Compositional Open-Ended Intelligence — closure expansion), 278 (Engram — hash-addressed pattern memory), 281 (Per-Tick Salience Tri-Gate), 284 (CLR — reward-gated memory write), 239 (MLEvolve — progressive MCGS branch spawning), 295 (AC-GPT arbitrary conditionals prefix), 296 (Stokes DEC vocabulary crosswalk), 309 (ARG × Latent Substrate — protocol synthesis)
> **Related Plans:** 236 (BAKE precision-gated embeddings), 290 (Closure-Expansion Instrument), 299 (Engram), 303 (Salience Tri-Gate), 307 (Claim Rubric Runtime), 316 (Per-NPC CLR Runtime), 327 (ARG protocol primitives)
> **Cross-ref (riir-ai):** Research 123 (Latent Functor runtime guide — non-interfering role transport via null-space projection), 146 (Entity Cognition Stack), 160 (ARG-over-Latent-State runtime guide). Plan 338 (private runtime wiring — to be created).

---

## TL;DR

RIZZ is a continual-adaptation framework for **frozen black-box LLM agents** that shifts learning from weight updates into **external routed memory**. Its core mechanism: each input is routed to a **memory branch** (a "zero-interference zone") that accumulates verifier-approved episodic examples, procedural rules, and anti-pattern failures; only interactions that clear a verifier gate may update memory; branches spawn when input is novel, merge when redundant, prune when stale. Across TRACE, StreamBench, LongMemEval-S, and τ-Bench, RIZZ improves a frozen Claude Haiku 4.5 by +3.3 to +28.2 pp over the no-memory baseline while staying cheaper than ACE/AMem and never collapsing the context window.

**Distilled for katgpt-rs (modelless, inference-time, open):**

Direct-mapping RIZZ onto this codebase is **mostly Pass** — RIZZ is LLM-text-centric (LLM judge proposes labels, prompts are natural language, verifiers are task-specific scorers) and the "frozen black-box LLM" framing does not fit a modelless engine with no LLM-in-the-loop. **But fusing RIZZ's branch-isolation discipline with our latent-state substrate is Super-GOAT.** The codebase already ships ~80% of the pieces under different vocabulary — BAKE (continual learning at embedding granularity), CLR (reward-gated memory write), MCGS (branch spawning for search), Engram (hash-addressed pattern memory), ARG (lifecycle + typed candidates), closure-instrument (motif mining). What is missing is the **unifying primitive**: a `BranchBank` that holds persistent, verifier-gated, non-interfering cognitive branches, each backed by an orthogonal latent subspace, with a router that selects/creates branches at inference time and a verifier gate that decides what may be written.

Five generic open primitives, no game/chain/shard semantics:

1. **`BranchBank<B>`** — a bounded bank of `CognitiveBranch { label, spawn_anchor, episodic, procedural, failures, scope_ctx, stats }`. Each branch is a persistent zero-interference zone. Spawn/merge/prune lifecycle (composes with ARG `LifecycleState`).
2. **`BranchRouter<E>`** — routes an input embedding to a branch by hierarchical label snapping (cosine ≥ τ_snap) + Jaccard token-overlap fallback (≥ τ_J). Spawns when max score = 0 and capacity remains. Zero-alloc hot path using pre-normalized embeddings.
3. **`VerifierGate<V>`** — gates memory writes on a verifier score `r ∈ [0,1]`. `should_write(r, branch_stats) := r > τ_write ∧ quarantine_guard(r, branch_centroid)`. Composes with CLR `should_write_memory(r_k, S_LP)` and Claim Rubric L1/L2/L3 evidence scores.
4. **`NonInterferenceProjection`** — assigns each branch an orthogonal projection direction in latent space; updates projected onto a branch's direction do not contaminate other branches by construction (dot-product = 0 across branches). The latent analog of RIZZ's textual branch isolation.
5. **`BudgetCompiler`** — assembles retrieved material (scoped ctx, episodic, procedural, failures, working memory) into a bounded context under a fixed token/byte budget using a fixed priority cascade. Degrades gracefully to "frozen" (empty context) when budget is tight.

All five are pure engine plumbing — no IP leak.

---

## 1. Paper Core Findings

### 1.1 The stability-plasticity problem at the memory level

RIZZ's target regime: a frozen LLM is deployed inside a compound agent (tools, retrievers, memory, routers) and observes a non-stationary stream `(x_t, ŷ_t, r_t)` where the reward `r_t` is revealed only after the agent responds. The agent must decide (a) which prior experience to condition on, (b) whether the outcome is worth remembering, and (c) where to write it so it doesn't corrupt behavior on unrelated tasks. A single global memory is brittle (lessons from one task family contaminate another); complete isolation is wasteful (related structure should transfer). RIZZ resolves this at the **memory level** rather than the weight level.

### 1.2 The five-mechanism stack

| Mechanism | What it does | Key parameters |
|-----------|--------------|----------------|
| **Hierarchical routing** | LLM judge proposes label `(function, application)`; snap to existing branch by `cos(ψ(a_t), ψ(a_b)) ≥ τ_snap=0.92`; fall back to Jaccard token overlap `J(a_t, a_b) ≥ τ_J=0.40`. Spawn new branch if no match and capacity remains. | τ_snap, τ_J, B_max |
| **Branch-local memory** | Each branch stores episodic examples `(x_i, ŷ_i, r_i, e_i, δ_i)`, procedural rules `(u_j, α_j, β_j, H_j, A_j)` with helpful/harmful counters, substantive failures (concrete anti-patterns), scoped context, summary stats. | per-branch capacity |
| **Verifier-gated evolution** | After the model acts, verifier `μ_t` returns `r_t ∈ [0,1]`. Only `Writable(m_t)` interactions may update memory. Successful → episodic + rule credit; failures → anti-pattern store; quarantine guard rejects off-centroid low-reward writes. | τ_write, quarantine threshold |
| **Bounded prompt compilation** | Fixed priority cascade: scoped ctx → memory-use preamble → instruction → external evidence → procedural rules → episodic examples → common mistakes → working memory → query. Lower-priority blocks dropped first when token budget tight. | B_ctx (64K cap) |
| **Branch lifecycle** | Periodic sweeps merge redundant branches (same function, high app-overlap) and prune stale low-utility ones. Read/write locality decoupled: routing selects write target, but positive examples/rules may retrieve cross-branch; failures stay local. | merge/prune cadence |

### 1.3 Empirical results (Claude Haiku 4.5 frozen base)

| Benchmark | Frozen | AMem | ACE* | **RIZZ** | Δ vs Frozen |
|-----------|--------|------|------|----------|-------------|
| StreamBench | 0.601 | 0.623 | 0.172 (ctx collapse) | **0.634** | +3.3 pp |
| TRACE | 0.637 | 0.630 | 0.047 (ctx collapse) | **0.706** | +6.9 pp |
| LongMemEval-S | 0.202 | 0.322 | 0.082 | **0.484** | +28.2 pp |
| τ-Bench (overall) | 0.352 | 0.303 | 0.370 | **0.400** | +4.8 pp |

RIZZ is the **only** framework that reliably improves over Frozen while staying operationally lightweight. ACE collapses (playbook exceeds 200K context window); AMem avoids collapse but costs 5-10× more tokens. RIZZ stabilizes at 10-15K playbook tokens and 31-195 branches depending on benchmark.

### 1.4 When RIZZ wins and when it doesn't

**Wins** (recurring local structure): C-STANCE (+16.4 pp, all 800 examples route to one specialist), NUMGLUE-DS (+13.5 pp), FOMC (+7.4 pp), LongMemEval multi-session (+31.6 pp absolute).

**Losses** (sparse, fragmented, procedurally heterogeneous): DS-1000 — queries fragment across ~40 low-volume code branches with <10 examples each, too little repeated experience for procedural/episodic memory to compound. RIZZ underperforms Frozen here.

**Ablation signal** (Table 3): disabling verified-memory-rendering costs -6.6 pp (the biggest single ablation hit); forcing single-branch costs -2.0 pp; disabling Jaccard sibling snapping costs -1.1 pp. **The verifier gate is the highest-value component.**

---

## 2. Distillation

### 2.1 What we keep (modelless, inference-time, Rust-tractable)

The entire RIZZ mechanism is **inference-time and weight-frozen** — the base model is never updated. Adaptation happens through routing, retrieval, curation, and prompt compilation. This is squarely inside our modelless mandate (constraint #1).

**Direct-translation table (RIZZ → codebase):**

| RIZZ concept | Codebase equivalent | Status |
|--------------|---------------------|--------|
| Frozen black-box LLM | (N/A — we have no LLM-in-the-loop; the "model" is the per-NPC HLA + functor + Engram cluster) | Reframe |
| Memory branch / zero-interference zone | **CognitiveBranch** — persistent per-NPC memory zone backed by an orthogonal latent subspace + Engram shard cluster + closure motif set | ❌ Missing (the gap) |
| Hierarchical routing (LLM judge + cosine snap + Jaccard) | **BranchRouter** — embedding dot-product snap (no LLM judge needed; the "label" is the branch's spawn-anchor direction in latent space) | ❌ Missing |
| Verifier-gated memory evolution | **VerifierGate** — composes with CLR `should_write_memory(r_k, S_LP)` (Plan 284, already shipped) + Claim Rubric L1/L2/L3 (Plan 307) | ◐ Partial (CLR has the gate; no branch-aware write target) |
| Procedural rules (IF-THEN with helpful/harmful counters) | **closure motifs** (Plan 290 PTG nodes) + **vibe KG triples** — already latent | ◐ Partial (motifs exist; no helpful/harmful counter machinery) |
| Episodic examples | **Engram patterns** (Plan 299) + **consolidation wake events** (riir-neuron-db) | ✅ Exists (single-store; needs per-branch partitioning) |
| Failure memory (anti-patterns) | (no direct equivalent — closest is `N_b` failure store) | ❌ Missing |
| Branch lifecycle (spawn/merge/prune) | **ARG `LifecycleState`** (Active/Shadow/Deprecated/Removed) + **freeze/thaw** (riir-neuron-db) | ◐ Partial (ARG has lifecycle types; no spawn/merge/prune loop) |
| Bounded prompt compilation | **salience tri-gate** (Plan 303) + **MUX-Latent** (Plan 238) priority cascade | ◐ Partial (priority cascade exists; no budget-aware branch compiler) |
| Continual learning / catastrophic forgetting avoidance | **BAKE** (Plan 236, embedding-level precision) | ◐ Partial (BAKE is embedding-level, not branch-level) |

### 2.2 The latent-space reframing (mandatory per skill §1.5)

RIZZ operates on **textual** states (LLM prompts, natural-language rules, token-based Jaccard). The codebase operates on **latent** states (HLA 8-D vectors, functor applications, NeuronShard style_weights). The reframing:

**Each CognitiveBranch is an independent latent subspace + memory cluster:**
- **Spawn anchor** = a unit direction vector `g_b ∈ R^d` in HLA/functor space (the "label" is implicit in the direction)
- **Episodic memory** = a cluster of Engram shards whose hash addresses project onto `g_b`
- **Procedural memory** = closure motifs (PTG nodes) whose primitive-transition signatures align with `g_b`
- **Non-interference** = branches occupy **orthogonal subspaces** by construction. Two branches `b_i, b_j` are non-interfering iff `dot(g_{b_i}, g_{b_j}) ≈ 0`. Updates projected onto `g_{b_i}` have zero component along `g_{b_j}`.

**The router becomes a pure dot-product snap** (no LLM judge):
- Input embedding `e_t` (from HLA state or Engram query embedding)
- For each branch `b`: score `s(b) = dot(e_t, g_b)` if `≥ τ_snap`, else Jaccard fallback on hash-token overlap
- Spawn if `max_b s(b) < τ_∅` and capacity remains; new branch's anchor = normalized `e_t`

**The verifier becomes the CLR reward gate** (already shipped, Plan 284):
- `should_write(r_k, S_LP) := r_k > τ_reliable ∧ S_LP > τ_curiosity` — CLR's existing two-sided gate
- RIZZ adds: the write TARGET is branch-selected (not global), and failures go to the branch's local `N_b` (not discarded)

**Why this reframing is Super-GOAT-tier, not GOAT-tier:**
- GOAT framing would be "add branch routing to CLR's memory write." That's an optimization.
- Super-GOAT framing is "each NPC maintains N orthogonal cognitive branches, each accumulating verifier-approved experience without cross-contamination, enabling continual adaptation across heterogeneous task families (combat, dialog, crafting, social) without any one task family corrupting another." That's a **new capability class** — the codebase cannot currently do per-NPC non-interfering continual adaptation.

### 2.3 Fusion genealogy

This note synthesizes seven prior shipped primitives into one continual-adaptation loop:

```
RIZZ (Oxford, 2026) — non-interference memory branches
   │
   ├──× R209 BAKE (Plan 236) ──────────── continual learning at embedding granularity → branch granularity
   ├──× R284 CLR (Plan 316) ───────────── reward-gated memory write → branch-aware write target
   ├──× R239 MLEvolve (progressive_mcgs) ─ transient search branches → persistent memory branches
   ├──× R278 Engram (Plan 299) ────────── hash-addressed pattern memory → per-branch partitioned memory
   ├──× R309 ARG (Plan 327) ───────────── lifecycle (Active/Shadow/Deprecated/Removed) → branch lifecycle
   ├──× R264 Closure (Plan 290) ────────── motif mining → procedural rules per branch
   └──× R281 Salience (Plan 303) ──────── priority cascade → bounded context compiler per branch
              │
              ▼
       Non-Interference Memory Branches over Latent Substrate
       (Super-GOAT fusion — this note)
```

**What none alone has:** BAKE updates embedding dimensions globally (no branch isolation). CLR gates writes but to a single store (no branch target). MCGS branches are transient per-query (not persistent). Engram is hash-addressed but single-store (no branching). ARG has lifecycle types but no spawn/merge/prune loop. Closure-instrument mines motifs but doesn't drive adaptation. Salience tri-gate prioritizes but doesn't compile per-branch.

**What the fusion produces:** a per-NPC continual-adaptation runtime where each NPC maintains N orthogonal cognitive branches, each accumulating verifier-approved experience without cross-contamination — enabling the NPC to learn combat tactics, dialog styles, crafting recipes, and social relationships **in parallel without any one corrupting another**.

---

## 3. Novelty Gate (Super-GOAT check)

### Q1: No prior art? — **YES (combination is novel; components exist separately)**

**Vocabulary translation (RIZZ → codebase), both layers grepped:**
- "memory branch" / "zero-interference zone" / "branch bank" → `CognitiveBranch`, `BranchBank`, memory-zone, cognitive-branch
- "verifier-gated" / "reward-gated memory" → `should_write_memory` (CLR, Plan 284 — EXISTS but single-store), `VerifierGate`
- "continual adaptation" / "catastrophic forgetting" → BAKE (Plan 236 — EXISTS at embedding level), `schema_centroid` (Plan 210)

**Grep results (both layers, all repos):**
- `progressive_mcgs/` has `BranchId`, `BranchStagnationState`, `ExpansionOperator::MultiBranchAggregation` — **transient search branches**, not persistent memory branches. Different lifecycle, different semantics.
- `clr/learning_potential.rs` has `should_write_memory(r_k, S_LP)` — **reward-gated write to global store**, not branch-selected write. Different target.
- `segment_checkpoint/memory_soup.rs` has `BranchState { segment_ids, gates, weighted_state }` — **KV-cache segment branches**, not memory-store branches. Different layer.
- `engram/kernel.rs` has `sigmoid_fuse_into` with write gate — **single hash-addressed store**, no branching.
- `BAKE` (Plan 236) — **continual learning at embedding dimension granularity**, not branch granularity.
- `ARG` (Plan 327) — **lifecycle types** (Active/Shadow/Deprecated/Removed), no spawn/merge/prune loop.
- "non-interference" appears in 2 places: Plan 279 (MPI row orthogonality) and riir-ai Research 123 (KG null-space projection) — **both different contexts** (orthogonality of expert grams / role transport, not memory branches).

**Verdict:** The COMBINATION (persistent memory branches + verifier-gated branch-targeted writes + non-interference by orthogonal latent subspaces + spawn/merge/prune lifecycle) has **no prior art**. The components exist in different contexts. This is a fusion, not greenfield — but the fusion produces a mechanism none of the components alone can do.

### Q2: New class of behavior? — **YES**

Test: "Can the codebase currently do per-NPC continual adaptation with non-interfering cognitive branches across heterogeneous task families?"

- BAKE does continual learning at **embedding dimension** granularity (global per-embedding).
- CLR does reward-gated memory write to a **single global store**.
- MCGS does branching for **transient search** (per-query, discarded after).
- Engram does hash-addressed memory in a **single store**.
- ARG has lifecycle types but **no spawn/merge/prune loop**.

**None of these can do branch-level continual adaptation with non-interference.** The fusion produces a new capability class: per-NPC cognitive branching for parallel continual learning without cross-contamination.

### Q3: Product selling point? — **YES**

> "Our NPCs maintain orthogonal cognitive branches — one for combat tactics, one for dialog style, one for crafting, one for social relationships — each accumulating verifier-approved experience without corrupting the others. An NPC that learned a combat shortcut does not forget how to craft, and a dialog failure does not pollute combat decision-making. This is continual adaptation at the cognitive-branch granularity, modelless, inference-time, with no weight updates."

No competitor does this. Existing game-AI continual-learning is either global (catastrophic forgetting) or isolated-per-task (no transfer). RIZZ-over-Latent-Substrate is the middle path: branch-local isolation with cross-branch positive transfer.

### Q4: Force multiplier? — **YES (7 pillars)**

Connects to: BAKE (continual learning), CLR (reward gate), MCGS (branch spawning), Engram (memory), ARG (lifecycle), closure-instrument (motifs), Salience (priority compiler), HLA (latent state), NeuronShard (freeze/thaw per branch). ≥2 pillars satisfied.

### All 4 YES → **Super-GOAT (fusion)**

---

## 4. Verdict

**Super-GOAT (fusion).** RIZZ × BAKE × CLR × MCGS × Engram × ARG × closure-instrument produces a new capability class — per-NPC non-interfering cognitive branches for continual adaptation — that no existing primitive alone can deliver. The latent reframing (orthogonal HLA subspaces as non-interference zones, CLR reward gate as verifier, Engram clusters as episodic memory, closure motifs as procedural rules) is clean and implementable modellessly.

**Mandatory outputs (this session):**
1. **Open primitive** → `katgpt-rs/crates/katgpt-core/src/branching/` — five generic primitives (`BranchBank`, `BranchRouter`, `VerifierGate`, `NonInterferenceProjection`, `BudgetCompiler`) behind feature flag `non_interference_branches`. Plan 329.
2. **Architectural guide** → `riir-ai/.research/161_Per_NPC_Cognitive_Branch_Continual_Adaptation_Guide.md` — the private selling-point doc.
3. **Plan(s)** → `katgpt-rs/.plans/329_non_interference_memory_branches.md` (open) + `riir-ai/.plans/338_cognitive_branch_runtime_wiring.md` (private runtime).

---

## 5. Risks (honest)

1. **Sparse-branch failure mode** (RIZZ §4, DS-1000 result) — when queries fragment across many low-volume branches with <10 examples each, procedural/episodic memory cannot compound. **Mitigation:** the branch merge sweep must be aggressive enough to collapse sparse sibling branches; add a `min_examples_per_branch` floor below which the branch is merged into its nearest sibling. This is the empirically-documented failure mode and the merge heuristic is the documented partial fix.

2. **Verifier quality ceiling** — RIZZ's gains depend on verifier quality. With a weak verifier (e.g., per-NPC CLR reward signal that's noisy), the gate may admit contaminated writes or reject useful ones. **Mitigation:** the CLR `S_LP` (learning potential) term is a secondary gate that filters low-curiosity writes even when the reward is high. Compose both.

3. **Orthogonal-subspace capacity limit** — in a d-dimensional HLA space (d=8), the number of fully-orthogonal branches is ≤ d. For more branches, use near-orthogonal (dot-product < ε) directions, accepting small cross-contamination. **Mitigation:** document the capacity-vs-interference tradeoff explicitly; the `NonInterferenceProjection` primitive should expose `max_orthogonal_branches(d)` and degrade gracefully to near-orthogonal.

4. **Vocabulary collision** — "branch" is overloaded in this codebase (MCGS branches, speculative branches, substrate branches, segment-checkpoint branches). **Mitigation:** namespace under `branching::` and use `CognitiveBranch` (not `Branch`) to distinguish from transient search/speculative branches.

5. **Premature unification** — synthesizing seven primitives risks over-constraining future work. **Mitigation:** the five open primitives are pure types + traits, not a runtime. riir-ai stays free to compose them however; the game runtime is not forced into RIZZ's exact loop.

---

## TL;DR (one-line)

RIZZ's verifier-gated non-interference memory branches, reframed over our latent substrate (orthogonal HLA subspaces + Engram clusters + closure motifs + CLR reward gate + ARG lifecycle), is a Super-GOAT fusion that produces a new capability class — per-NPC continual adaptation without catastrophic forgetting — currently absent from the codebase.
