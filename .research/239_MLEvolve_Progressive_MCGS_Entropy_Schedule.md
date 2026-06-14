# Research 239: MLEvolve → Progressive MCGS + Entropy-Gated Exploration/Exploitation

> **Source:** "MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery" — Du, Yan, Shi, Cao, Feng, Liang, Sun, Peng, Zhou, Li, Zhou, He, Zhang, Bai (Shanghai AI Lab + ECNU), arXiv:2606.06473, 2026-06-04. Code: https://github.com/InternScience/MLEvolve
> **Date:** 2026-06-14
> **Status:** Active
> **Related Research:** 134 (BES entropy shell), 172 (MUSE skill lifecycle), 190 (regime-transition MDL gate), 075 (Survive-or-Collapse), 093 (committee search), 088/170 (proof/DAG search), 216 (MRAgent memory graph), 052 (SimpleTES eval-driven scaling)
> **Related Plans:** katgpt-rs Plan 272 (TBD — public `progressive_mcgs` module) — will be created next
> **Cross-ref (riir-ai):** Research 122 (Crowd-Scale Progressive MCGS for NPC Emergent Behavior), Plan 298 (riir-ai game-runtime instantiation)
> **Classification:** Public

---

## TL;DR

MLEvolve is an LLM-agent coding framework that hits SOTA on MLE-Bench (65.3% medal rate, 12h budget) by upgrading plain MCTS into **Progressive Monte-Carlo Graph Search (MCGS)**: a directed graph where **reference edges** carry cross-branch information *without* participating in backprop, plus an **entropy-inspired soft switch** that gradually collapses the active-branch distribution from broad UCT exploration (≈4.8 branches) to Elite-Guided exploitation (≈2.8 branches) as search time runs out. Combined with a **Retrospective Memory** (BM25 ⊕ FAISS → RRF) and **stagnation-triggered expansion operators** (intra-branch evolution, cross-branch reference, multi-branch aggregation), it sustains improvement across long horizons where vanilla MCTS plateaus.

**Distilled for katgpt-rs (modelless, inference-time):** Drop the LLM-coding-agent wrapper. Keep three transferable primitives: (1) **reference-edge graph search** — a DAG where auxiliary edges compose information across branches without polluting credit assignment; (2) **entropy-gated exploration→exploitation schedule** — use Shannon entropy of the branch-selection distribution as the *signal* (not the objective) to soft-switch between UCT and Elite-Guided modes via a decaying weight `w(t)`; (3) **stagnation-triggered expansion operators** — branch-level (τ consecutive non-improvements) and global-level (τ_global steps without best refresh) gates that fire composition/fusion operators. All three are runtime, allocation-free hot paths, gateable by feature flag.

---

## 1. Paper Core Findings

### 1.1 The three observed failures of existing MLE agents (their motivation)

| Failure | Symptom | MLEvolve's fix |
|---------|---------|----------------|
| Branch isolation | Tree/MCTS confines info within a single branch; can't transfer a winning trick from branch A to branch B | Reference edges `E_ref` connect nodes across branches |
| Memoryless search | Only scalar reward propagates; each plan is made in isolation | Retrospective Memory: cold-start KB + dynamic global memory with hybrid retrieval |
| One-shot generation | Plan + code fused into one LLM call; full rewrite every iteration | Planner-Coder decoupling + adaptive Base/Stepwise/Diff modes |

### 1.2 Progressive MCGS — the math

Search space is a directed graph `G = (V, E)`, `E = E_T ∪ E_ref`:

- **Primary edges** `E_T`: parent→child generative. Used for selection + backprop.
- **Reference edges** `E_ref`: `(r, v)` means `v` additionally read `r` (cross-branch or non-adjacent). **Excluded from backprop.** When `E_ref = ∅`, reduces to vanilla MCTS.

Unified expansion: `v_new = g_o(v_t, R)`, with `(v_t, v_new) ∈ E_T` and `{(r, v_new) | r ∈ R} ⊆ E_ref`.

**Selection (the entropy trick):** Operates only on the `E_T` tree backbone via UCT, but with a *probabilistic soft switch*:

```
P(S_t = UCT)   = w(t)        # broad exploration
P(S_t = Elite) = 1 - w(t)    # exploit top-K globally best nodes
```

where `w(t)` decays from 1.0 → `w_min=0.2` over a window `[explore_switch_start=0.5, explore_switch_end=0.7]` of normalized search progress. Elite-Guided mode samples from top-K nodes weighted by `1/rank(v_i)`. The schedule is designed so the *empirical branch-selection entropy* `H(π_t)` decreases monotonically — empirically observed dropping from exp(H)≈4.8 → 2.8 active branches, vs vanilla MCTS staying flat at ≈4.3.

**Reward (3-level, clean credit assignment):**
- `R(v) = -1` execution fails / no metric
- `R(v) = +1` succeeds but doesn't beat branch best
- `R(v) = +2` succeeds AND refreshes branch best

**Backprop:** Only along `E_T`. `N_u ← N_u + 1`, `W_u ← W_u + R(v)`, `Q_u = W_u / (N_u + ε)`.

### 1.3 Stagnation-triggered expansion operators (the graph part)

Two thresholds, four operators:

| Trigger | Threshold | Operator | Reference set R |
|---------|-----------|----------|-----------------|
| Branch stagnation | τ_branch = 3 consecutive non-improving expansions | **Intra-branch evolution** | `R_hist(v_t, k)` — nearest k ancestors in same branch |
| Branch stagnation (late stage) | τ_branch + other branches have strong solutions | **Cross-branch reference** | `R_cross(N)` — top-N nodes across all branches |
| Global stagnation | τ_global = 6 steps without global best refresh | **Multi-branch aggregation** | `R_agg = ⋃_b T_top_b` — top trajectories from all branches, spawns new branch under root |
| (baseline) | always available | **Primary expansion** | `R = ∅` |

### 1.4 Retrospective Memory

- **Static KB (cold-start):** curated (model, task-type, usage-guideline) triples; retrieved by keyword match for initial draft.
- **Dynamic Global Memory:** appends structured record (plan, code, metric, analysis, feedback) after each valid node execution.
- **Hybrid retrieval:** `score(d) = α · 1/(k + r_lex(d)) + (1-α) · 1/(k + r_vec(d))` — BM25 rank `r_lex` fused with FAISS rank `r_vec` via Reciprocal Rank Fusion.
- **Stage-aware query:** Planning stage uses the draft plan as query; Debug stage uses the error message as query.

### 1.5 Hierarchical planning (less transferable)

Planner (module-level, "what/why") → Coder (code-level, "how"), with three coding modes selected by state: Base (full rewrite, cold start), Stepwise (module-by-module, complex pipelines), Diff (patch edit, refinement). This part is LLM-coding-agent specific — **distill only the *state→mode routing table*** concept, not the LLM orchestration.

### 1.6 Results that matter for us

| Result | Value | Why we care |
|--------|-------|-------------|
| MLE-Bench medal rate (12h, half standard budget) | **65.3%** (SOTA) | Validates the search primitive at scale |
| Active branch entropy drop | 4.8 → 2.8 | Empirical proof that entropy-gated schedule concentrates compute |
| Vanilla MCTS plateau | ~70% beat ratio, plateaus early | Baseline fails without entropy schedule |
| MLEvolve late-stage gain | 98.2% beat ratio, keeps improving | Entropy schedule + reference edges sustain long-horizon search |
| Ablation: −Progressive MCGS | −13.6pp medal (largest drop) | MCGS is the load-bearing component, not memory |
| Ablation: −intra-branch evolution | −33pp medal (9-task subset) | Self-reflection on recent trajectory is the single most critical operator |
| Cross-domain (AlphaEvolve math) | 11/15 best vs AlphaEvolve-v2 | Generalizes beyond MLE — pure search result |

---

## 2. Distillation

### 2.1 What we keep (modelless, inference-time, Rust-tractable)

#### Primitive A: `ProgressiveMcgs` — graph search with reference edges

Decouple **credit-assignment edges** (`E_T`, primary, tree backbone) from **information-flow edges** (`E_ref`, auxiliary, DAG). Backprop only walks `E_T`. Reference edges are write-only at expansion time and read-only at proposal-construction time. This is the key insight: **composition without credit pollution**.

Maps to existing katgpt-rs primitives:
- `E_T` ≈ `BanditPruner` arms + `DDTree` parent links (already tree-shaped, already backprop)
- `E_ref` ≈ a *new* `ReferenceEdgeSet` side-table keyed by child node id, storing `Vec<NodeId>` of referenced nodes. Zero impact on existing backprop hot path.

#### Primitive B: `EntropyGatedScheduler` — exploration→exploitation soft switch

The schedule is the transferable IP. It is **not** "minimize entropy" (that would collapse too fast). It is "let the empirical branch-selection distribution's Shannon entropy *naturally decay* under a softening UCT weight, then hard-switch to Elite sampling past a progress threshold."

```rust
// Sketch — allocation-free, gateable
struct EntropyGatedScheduler {
    w_min: f32,            // 0.2 — floor on UCT probability
    switch_start: f32,     // 0.5 — normalized progress where decay begins
    switch_end: f32,       // 0.7 — normalized progress where decay saturates
    elite_topk: usize,     // 3
}

impl EntropyGatedScheduler {
    #[inline]
    fn w(&self, t_norm: f32) -> f32 {
        // piecewise-linear decay 1.0 → w_min over [switch_start, switch_end]
        if t_norm < self.switch_start { 1.0 }
        else if t_norm > self.switch_end { self.w_min }
        else {
            let s = (t_norm - self.switch_start) / (self.switch_end - self.switch_start);
            1.0 + s * (self.w_min - 1.0)
        }
    }

    #[inline]
    fn pick_mode(&self, t_norm: f32, rng: &mut impl Rng) -> SelectMode {
        if rng.gen::<f32>() < self.w(t_norm) { SelectMode::Uct } else { SelectMode::Elite }
    }
}
```

Use **sigmoid** if we want smoothness instead of piecewise (per AGENTS.md: sigmoid not softmax). The empirical entropy `H(π_t)` is a *diagnostic metric*, not the objective — we log it, we don't gradient through it.

#### Primitive C: `StagnationGate` — triggered expansion operators

Two counters per scope:
- `BranchStagnation { since_last_improve: u32, threshold: u32 = 3 }`
- `GlobalStagnation { since_last_best: u32, threshold: u32 = 6 }`

When fired, queue an expansion operator:
- **IntraBranchEvolve**: pass last-k ancestors as reference set to the proposer (most important — ablation says −33pp without it)
- **CrossBranchReference**: pass top-N globally as reference set
- **MultiBranchAggregation**: spawn a new root child, reference set = union of top trajectories per branch

These compose with our existing `ConstraintPruner` proposer pipeline: the reference set just becomes extra context fed to the proposer, exactly like `HintDelta` in G-Zero (R021, Benchmark 005).

#### Primitive D: `RetrospectiveMemory` — hybrid retrieval

BM25 (lexical) ⊕ FAISS-style vector search (we already have approximate variants via LSH/CMS in R195), fused by RRF. The *transferable* part is the **stage-aware query routing**:
- Planning query = current draft plan / current goal embedding
- Debug query = error signature / failure trace embedding
- Both retrieve from the same store, different filter masks

This is exactly the `HlaCacheProxy` pattern from riir-ai — latent store, scalar projection at the boundary. **No raw code or raw plan leaves the local tier**; only embeddings and BLAKE3 commitments sync.

### 2.2 What we drop (LLM-coding-agent specific, not modelless)

- **The LLM agents themselves** (Draft, Improve, Debug, Evolution, Fusion, Aggregation, Code Review, Data Leakage, Result Parse). These are Python orchestration around Gemini-3.1-Pro. Not Rust inference primitives.
- **Planner-Coder decoupling as described.** We have no LLM planner. But the *concept* (separate "what to modify" from "how to modify") maps to our existing split: `ConstraintPruner` (what) vs `dispatch_*` executors (how).
- **Adaptive Code Generation modes (Base/Stepwise/Diff).** We don't generate code; we route between frozen adapters. The analog is *adapter-edit granularity* (full swap vs layer-diff vs sparse-mask), but that's a riir-ai concern, not katgpt-rs.
- **Data Leakage Agent.** Specific to Kaggle eval setup. Not transferable.

### 2.3 Mapping to existing katgpt-rs primitives

| MLEvolve concept | katgpt-rs existing primitive | Gap to close |
|------------------|------------------------------|--------------|
| `E_T` primary edges | `BanditPruner` arm hierarchy + `DDTree` parent links | ✅ none |
| `E_ref` reference edges | — (new) | **Primitive A** — add `ReferenceEdgeSet` side-table |
| UCT selection | `BanditPruner::uct_select` | ✅ none |
| Elite-Guided selection | `OpusBanditPruner` Boltzmann (R129) | Close — add `1/rank` weighting option |
| Entropy-gated schedule | `BreakevenComplexityRouter` (R218) is the closest — but it routes on task complexity, not search-progress entropy | **Primitive B** — new `EntropyGatedScheduler` |
| Branch stagnation | `CollapseDetector` (R075, R179) detects reasoning collapse, not "no improvement in k steps" | **Primitive C** — simpler, add `StagnationGate` |
| Intra-branch evolution | `G-Zero` `HintDelta` (R021) feeds teacher-forced δ as context; SR²AM (R076) self-reflection | Conceptually same — generalize to "reference set" abstraction |
| Cross-branch reference | — (new) | **Primitive A+C** combo |
| Multi-branch aggregation | `MUSE` skill fusion (R172) fuses skills, not search branches | Extend fusion to operate on graph branches |
| Retrospective Memory KB | `LoRAWeightVersion` metadata + blake3 commitments | Conceptual match — KB = committed metadata |
| Retrospective Memory global | `EpisodePruner` DB + `AbsorbCompress` (R075) | **Primitive D** — add hybrid retrieval (BM25 ⊕ vector) |
| Hybrid retrieval RRF | LSH/CMS approximate cache (R195) | Add BM25 rank fusion on top |
| Reward `{-1, 0, +1, +2}` | `DeltaBanditPruner` reward shaping (R021) | ✅ compatible — extend reward enum |

### 2.4 Mapping to riir-ai (private, game runtime)

This is where the *aggregate-behavior* payoff lands:

- **NPC crowd exploration → exploitation**: At server-start, NPCs explore zone/behavior space broadly (UCT mode, `w(t)≈1`). As session matures, scheduler decays `w(t)`, NPCs converge on high-value zones/behaviors (Elite mode). The transition is *emergent* from the schedule, not scripted. **Single scheduler instance per zone**, not per NPC — the entropy is computed over the zone's branch-selection distribution.
- **Cross-NPC reference edges (social learning)**: When NPC A discovers a good trade route or combat pattern, that becomes a reference node. NPC B (different branch = different personality snapshot) can reference A's solution *without* inheriting A's credit assignment. This is exactly the **KG-triple emission** pattern from the latent-vs-raw rules: latent similarity triggers reference edge, raw sync still carries the actual position/inventory delta separately.
- **Stagnation → faction events**: τ_branch stagnation across multiple NPCs in a zone → trigger intra-zone "evolution event" (NPCs share recent trajectory). τ_global stagnation → trigger cross-zone "aggregation event" (found new faction from top trajectories across zones). This is emergent social/economic behavior — not scripted events, but *search-graph topology events*.
- **Retrospective Memory for NPC episodic state**: Each NPC's `HlaCacheProxy` is its dynamic global memory; the cold-start KB is the role/personality template. Hybrid retrieval = recall relevant past encounters by both lexical (quest ID, item ID) and latent (emotion dot-product) signals.

### 2.5 What this unblocks

1. **A clean abstraction for "composition without credit pollution"** — currently we conflate "this pruner used that pruner's output" with "that pruner gets credit for this success." Reference edges separate the two.
2. **A principled exploration→exploitation transition** that isn't a hand-tuned epsilon decay. The entropy diagnostic gives us a knob we can plot and gate.
3. **Stagnation as a first-class signal** — currently `CollapseDetector` fires on entropy collapse of the *reasoning trace*. `StagnationGate` fires on *reward* plateau. These are complementary: collapse = "model is broken", stagnation = "search is stuck but model is fine."
4. **A routing taxonomy for the planner**: Base/Stepwise/Diff maps to Full-Swap/Layer-Diff/Sparse-Mask adapter edits. Gives riir-ai a vocabulary for adapter-edit granularity.

---

## 3. Verdict

**GAIN** — not GOAT.

**Why GAIN, not GOAT:** MLEvolve is a *coding-agent orchestration* paper. Its SOTA on MLE-Bench is largely an LLM-prompting + search-structuring achievement, not a new inference kernel. The distilled primitives (reference edges, entropy-gated schedule, stagnation gates) are useful and unblock clean abstractions we currently lack, but they are *compositions of existing ideas* (MCTS + graph merging + entropy regularization + RAG). They will improve our routing/exploration quality, but they don't redefine the inference stack.

**Why not PASS:** Three of the four primitives (A: reference edges, B: entropy-gated schedule, C: stagnation gates) are *not* present in our codebase in this form. The closest analogs (BreakevenComplexityRouter R218, CollapseDetector R075, HintDelta R021) cover adjacent but distinct concerns. The MCGS formalism gives us a single coherent frame for "graph search with decoupled credit vs info flow" — and that frame directly enables the riir-ai NPC crowd-exploration pattern we've been hand-waving.

**Conditions for promotion to default:** Must demonstrate (a) entropy `H(π_t)` monotonically decays under the schedule on at least one benchmark, (b) reference edges do not corrupt backprop (Q-values on `E_T` match vanilla MCTS on identical rollouts with `E_ref=∅`), (c) stagnation-gated operators fire at the documented thresholds and improve best-reward-find-rate vs vanilla UCT.

**Suggested plan slot:** 272 (next free in `.plans/` after 271).

**Commercial strategy fit:** Primitive A (reference-edge graph) and B (entropy schedule) are *generic* → ship in `katgpt-rs` (public, MIT) as a `progressive_mcgs` module. Primitive C operator instantiations for game AI (faction events, zone aggregation) → stay in `riir-ai` (private). No training know-how leaks.

---

## 4. Open questions / risks

- **Reference-edge explosion.** If every node references top-K globally, `E_ref` grows O(N·K). Need a blake3-committed cap or LRU eviction. Mitigation: references are stored as `(child_id, ref_id)` pairs in a side-table — eviction doesn't touch the primary tree.
- **Entropy schedule hyperparameters are paper-specific.** `switch_start=0.5, switch_end=0.7, w_min=0.2` were tuned for 500-step / 12h budgets. Our 20Hz tick budget is *much* shorter — the schedule likely needs rescaling to tick-count, not wall-time. Defer to Plan 272.
- **Elite-Guided `1/rank` weighting** is *not* sigmoid. AGENTS.md prefers sigmoid. Reconcile: either (a) use sigmoid over rank-derived logits, (b) document why `1/rank` is acceptable here (it's a sampling weight, not a projection gate). Lean (b) — the sigmoid rule is about *latent projections onto direction vectors*, not about discrete rank-based sampling.
- **Reward `+2` for "refreshes branch best"** is non-stationary (branch best evolves). Need to snapshot branch-best at expansion time, not at backprop time, to avoid feedback loops. Standard MCTS gotcha but worth flagging.
- **Does this overlap with R218 (BreakevenComplexityRouter)?** Partially — both route based on dynamic signals. R218 routes *across inference strategies* (plasma/hot/warm) by task complexity; Progressive MCGS routes *within a search tree* by progress entropy. They compose, they don't conflict.

---

## TL;DR

MLEvolve's value to us is **not** the LLM coding agent — it's the **Progressive MCGS** formalism: a directed graph where reference edges compose information across branches without polluting backprop, plus an entropy-gated soft switch from UCT exploration to Elite-Guided exploitation, plus stagnation-triggered expansion operators. These three primitives are modelless, inference-time, allocation-free, and directly unblock (a) clean composition-without-credit-pollution in katgpt-rs's bandit/DDTree stack and (b) emergent crowd-scale NPC exploration→exploitation transitions in riir-ai. **Verdict: GAIN.** Plan 272 candidate: implement `progressive_mcgs` module behind `--features progressive_mcgs`, benchmark entropy decay + backprop correctness + stagnation-gate improvement over vanilla UCT.
