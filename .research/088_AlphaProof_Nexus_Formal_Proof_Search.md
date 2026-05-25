# Research 88: AlphaProof Nexus — AI-Driven Formal Proof Search

> **Paper:** [Advancing Mathematics Research with AI-Driven Formal Proof Search](https://arxiv.org/abs/2605.22763) — Tsoukalas, Kovsharov, Shirobokov et al. (Google DeepMind), May 2026
> **Code:** https://github.com/google-deepmind/alphaproof-nexus-results
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 037 (REAP model-based/modelless duality), 076 (SR²AM configurator), 021 (G-Zero self-play), 058 (GRAM recursive reasoning), 050 (LDT lattice deduction)
> **Related Plans:** Plan 128 (proof sketch evolution — planned)
> **Feature Gate:** `proof_sketch_evolution` (opt-in, for GOAT proof)

## TL;DR

AlphaProof Nexus is a framework where LLM-based agents search for formal proofs in Lean, combining generation (LLM) with verification (Lean compiler). The key architectural insight: **a basic generate-verify loop (modelless) solves most problems, but an evolutionary population with Elo-rated sketches (model-based) wins on the hardest problems by 2-5× cost efficiency**. This directly maps to our model-based/modelless spectrum: the basic agent is our `NoScreeningPruner` path (cheap, parallel, stateless), while the evolutionary agent is our `BanditPruner` path (stateful, adaptive, higher compute).

**Verdict: HIGH VALUE — Three distillable concepts:**
1. **Elo-rated sketch population** → distills into our `BanditPruner` Q-value system for proof strategy selection
2. **Binary fitness → graduated fitness via LLM rater** → solves our GOAT proof gap: formal verification is pass/fail, but Elo gives a smooth gradient for bandit exploration
3. **Global goal caching** → distills into `ProofGoalCache` for avoiding redundant constraint verification across MCTS branches

**What's NOT worth distilling:**
- Full Lean compiler integration → out of scope for Rust inference runtime
- AlphaProof RL-trained prover → requires massive TPU training, not applicable
- Gemini 3.1 Pro dependency → we use local models
- Python async infrastructure → we're in Rust

---

## Core Architecture: Four Agent Tiers

The paper presents a clear model-based/modelless progression:

```
Agent (A): Basic          — N independent subagents, no shared state
                           ↓ add AlphaProof tool
Agent (B): Basic+AP       — subagents can query focused proof tool
                           ↓ add evolutionary population
Agent (C): Evo            — shared sketch DB, Elo ratings, P-UCB sampling
                           ↓ combine both
Agent (D): Full-featured  — Evo + AlphaProof + goal caching
```

### Model-Based/Modelless Mapping

| AlphaProof Nexus | Our System | Type |
|------------------|------------|------|
| Agent (A): N independent prover subagents | `NoScreeningPruner` — accept all, brute force parallel | **Modelless** |
| Agent (B): + AlphaProof tool calls | `ConstraintPruner` — static rules filter obviously invalid | **Modelless** |
| Agent (C): Elo-rated population, P-UCB sampling | `BanditPruner<P>` — Q-value guided exploration | **Model-based** |
| Agent (D): Full + goal caching + focused solver | `DeltaBanditPruner` — adaptive with delta compression | **Model-based** |
| Gemini 3.0 Flash raters | `ScreeningPruner::relevance()` — cheap scoring | **Modelless signal** |
| Gemini 3.1 Pro provers | LLM generation in riir-ai | **Model-based signal** |

### Key Finding: Basic Agent Surprises

> "The effectiveness of our basic agent in our post-hoc analysis was surprising... We attribute the basic agent's success to both this shift [in LLM capabilities] and the power of compiler feedback in grounding LLM reasoning."

This mirrors our finding: `NoScreeningPruner` often matches `BanditPruner` quality on easy domains, but BanditPruner wins on hard domains (Bomber, Go) by 2-5×. The pattern: **modelless is sufficient for 80% of problems, model-based wins on the 20% that matter most**.

---

## Distillable Concept 1: Elo-Rated Sketch Population

### What the Paper Does

Binary fitness (proof compiles or doesn't) is bridged to graduated fitness via:
1. **LLM raters** (cheap model) compare P=7 sketches pairwise
2. **Plackett-Luce model** with hierarchical Gamma prior infers latent strength λ_s
3. **Gibbs sampling** (1000 samples, 200 burn-in) produces posterior mean λ_s^mean
4. **Elo conversion:** `Elo_s = 1200 + 400 * log10(λ_s^mean)`

### How This Maps to Our System

Our `BanditPruner` already uses Q-values from UCB1. The Elo insight adds:
- **Relative ranking** instead of absolute reward — more robust to reward noise
- **Hierarchical prior** (Gamma(1, r_s) where r_s ~ Gamma(1, 1)) — heavy tails prevent premature convergence
- **Population database** — not just top-1, but top-64 filtered with UCB exploration

**Distillation target:** `ProofSketchDB` struct in `katgpt-core`

```text
// Conceptual mapping (not literal code)
struct ProofSketchDB {
    sketches: HashMap<SketchId, SketchEntry>,
    elo_ratings: HashMap<SketchId, f64>,     // Elo_s scores
    visit_counts: HashMap<SketchId, usize>,  // for P-UCB
}

struct SketchEntry {
    proof_state: ProofState,     // current Lean-like state
    pending_goals: Vec<Goal>,    // unresolved subgoals
    lessons: Vec<String>,        // episode summaries
    timestamp: Instant,
}

// P-UCB sampling (already in our bandit, just applied to sketches)
fn p_ucb_score(sketch: &SketchEntry, total_visits: usize, c: f64) -> f64 {
    let q = normalize(sketch.elo_rating);  // [0,1] from top-64
    q + c * (total_visits.sqrt() / (sketch.visits + 1))
}
```

### Why It Matters for GOAT Proof

Our GOAT proofs currently use:
- Arena benchmarks (1000 rounds, win rate)
- Cosine similarity checks
- Wall-clock timing

Adding Elo-rated sketch population lets us prove that **evolutionary search converges faster than independent search** on formal verification tasks — a measurable, repeatable GOAT proof.

---

## Distillable Concept 2: Binary → Graduated Fitness via LLM Rater

### The Core Problem

Formal verification is binary: a proof compiles or it doesn't. This creates a **flat fitness landscape** where evolutionary search has no gradient to follow. Most sketches fail, and among failures there's no signal about which is "closer."

### The Paper's Solution

Use a cheaper LLM to rate sketches on:
1. **Clarity** of proof strategy
2. **Plausibility** of remaining goals
3. **Mathematical novelty** of approach

These are aggregated via Plackett-Luce into Elo ratings, providing smooth gradient.

### Our Application: Constraint Verification as Binary Fitness

Our `ConstraintPruner::is_valid()` is also binary — a move either satisfies constraints or doesn't. In game domains (Bomber, Go, Monopoly), most moves are valid but most valid moves are terrible. The paper's insight applies:

**Distillation:** Add `ScreeningPruner::relevance()` as a "rater" that provides graduated scores for moves that pass `ConstraintPruner`. Currently `relevance()` returns a single f64. The upgrade: **compare moves pairwise** using our existing `MaxSim` late-interaction scorer, then aggregate via Elo.

This is already partially implemented:
- `MaxSim::score()` → pairwise late-interaction scoring
- `BanditPruner<P>` → UCB1 exploration
- What's missing: **Plackett-Luce aggregation** across multiple scoring rounds

---

## Distillable Concept 3: Global Goal Cache

### What the Paper Does

Independent proving agents generate the same subgoals repeatedly. Agent (D) implements a **global goal cache** keyed by deep hash of the exact formal context:

```text
goal_id = hash(lean_context + target_statement)
```

Before querying AlphaProof, check cache. If the subgoal was previously proved/disproved in ANY sketch, reuse the result. Novel subgoals are batched and dispatched concurrently via non-blocking RPCs.

### Our Application: MCTS Transposition Table + Constraint Cache

We already have:
- `GoState` transposition table (Zobrist hash)
- `ConstraintPruner` caches validation results
- `TurboQuant` KV cache compression

What's missing: **cross-branch goal deduplication** in DDTree speculative decoding. When multiple draft branches hit the same constraint verification (e.g., "does this token sequence satisfy the validator?"), we re-verify from scratch.

**Distillation target:** `ProofGoalCache` — a global cache for constraint verification results across DDTree branches.

```text
// Conceptual mapping
struct ProofGoalCache {
    cache: HashMap<GoalHash, GoalResult>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GoalHash(blake3::Hash);  // blake3 per our conventions

enum GoalResult {
    Proved(Proof),
    Disproved(Counterexample),
    Unknown,
}

impl ProofGoalCache {
    fn get_or_verify(&self, goal: &Goal, verifier: &dyn ConstraintPruner) -> GoalResult {
        let hash = GoalHash(blake3::hash(&goal.canonical_bytes()));
        match self.cache.get(&hash) {
            Some(result) => result.clone(),
            None => {
                let result = verifier.verify_goal(goal);
                self.cache.insert(hash, result.clone());
                result
            }
        }
    }
}
```

### Why It Matters

For DDTree with `draft_lookahead=5` and `tree_budget=64`, we generate up to 320 draft branches. Many share sub-sequences that need identical constraint checks. Global goal caching could reduce verification cost by 3-10× on structured domains (Bomber, Go).

---

## Paper Results Summary

| Task | Result | Cost |
|------|--------|------|
| Erdős problems (353 attempted) | 9 solved (2.5%) | ~$200-500/problem |
| OEIS conjectures (492 attempted) | 44 proved (8.9%) | Not reported |
| Hilbert function (algebraic geometry) | 1/4 open problems solved | Not reported |
| Anchored GDA convergence (optimization) | Novel parameter schedule discovered | Not reported |
| Graph reconstruction variant | Complete proof | Not reported |

### Agent Comparison (Erdős problems)

| Agent | Solved | Cost Efficiency | Notes |
|-------|--------|----------------|-------|
| (A) Basic | 9/9 (post-hoc) | Baseline | Surprisingly effective |
| (B) Basic+AP | 9/9 | Better on hard problems | AlphaProof saves cost on #12(ii), #125 |
| (C) Evo | Not tested alone | — | Middle ground |
| (D) Full | 9/9 | 2-5× better on #138, #125 | Best on hardest problems |

### Key Failure Modes (Important for Our System)

1. **Goal offloading:** Agent frequently pushes core difficulty into a single `sorry` lemma that restates the target. Our equivalent: DDTree branches that pass `ConstraintPruner` but fail `ScreeningPruner` — technically valid but useless.
2. **Hallucinated lemmas:** Agent claims sorry-marked lemmas are established results. Our equivalent: `ConstraintPruner` accepting hallucinated move sequences that look structurally valid but are semantically wrong.
3. **Misformalization detection:** Agent found errors in problem formalization. Our equivalent: validator catching edge cases in game rules (already handled by `GameState::is_legal()`).

---

## Architecture: Ralph Loop → Our Episode Loop

The paper's "Ralph loop" (Figure 4) maps directly to our `GZeroLoop`:

```
Paper's Ralph Loop:                     Our GZeroLoop:
─────────────────────                   ────────────────
1. Start with proof sketch              1. Start with game state
2. Multi-turn LLM session               2. Multi-step MCTS rollout
3. Search-replace edits                 3. Move proposals
4. Lean compiler feedback               4. GameState::step() feedback
5. If sorry remains → summarize         5. If game not over → credit assignment
6. Next episode from current sketch     6. Next rollout from current policy
```

The key shared pattern: **compiler/feedback-grounded iteration with lesson summarization**. Our `DeltaBanditPruner` already captures "lessons" as delta-encoded reward adjustments. The paper formalizes this as episode-end summaries injected into the next episode's prompt.

---

## What We Already Have (No Action Needed)

| Paper Concept | Our Existing Implementation |
|---------------|---------------------------|
| Independent parallel subagents | Rayon parallelism in DDTree |
| Binary fitness (pass/fail) | `ConstraintPruner::is_valid()` |
| Generate-verify loop | DDTree draft → verify cycle |
| Lesson summarization | `BanditPruner` Q-value updates |
| UCB exploration | `BanditPruner` UCB1 formula |
| Population database (partial) | `GoState` transposition table |
| Goal caching (partial) | `ConstraintPruner` validation cache |
| LLM rater (partial) | `ScreeningPruner::relevance()` |

---

## What's Worth Distilling (New)

### 1. Elo-Rated Proof Sketch DB — `proof_sketch_evolution` feature gate

**Priority:** Medium
**Scope:** `katgpt-core`
**GOAT proof target:** Prove evolutionary search converges 2× faster than independent search on formal verification tasks

Add a `SketchPopulation` that:
- Maintains top-64 sketches with Elo ratings
- Uses Plackett-Luce aggregation for pairwise comparisons
- Samples via P-UCB (already in BanditPruner)
- Caches goal verification results globally

### 2. Plackett-Luce Pairwise Aggregation

**Priority:** Low
**Scope:** `katgpt-core`
**Note:** Our `BradleyTerry` (Plan 080) already does pairwise ranking. Plackett-Luce is the multi-item generalization. Consider extending BT to PL.

### 3. Global Goal Cache for DDTree

**Priority:** Medium-High
**Scope:** `katgpt-core`
**GOAT proof target:** Prove 3× reduction in constraint verification calls

Add `ProofGoalCache` using `blake3` hashes of canonical goal representations, shared across DDTree branches within a single decode step.

---

## Additional Insights (Supplementary Distillation)

### Insight 4: EVOLVE-VALUE Parameter Co-Search

The paper's optimization theory result is remarkable: the agent didn't just *verify* a fixed algorithm — it **discovered a novel parameter schedule** for anchored Gradient Descent-Ascent. The `EVOLVE-VALUE` markers allow the agent to search over parameter values simultaneously with the proof. This is not just proof search; it's **program synthesis with correctness guarantees**.

**Our mapping:** Our `RandOpt` weight perturbation (Research 081, Plan 121) searches weight-space randomly. The EVOLVE-VALUE insight adds: **constrain the search to provably-correct regions**. For game arenas, this means searching over heuristic parameters (e.g., MCTS exploration constant, bandit ε) while simultaneously verifying that the resulting policy beats a threshold. The co-search pattern:

```text
EVOLVE-VALUE in our context:
  1. Mark tunable parameters as evolvable (tree_budget, draft_lookahead, bandit ε)
  2. Evolutionary search over parameter space
  3. Each candidate evaluated by: run arena → verify win rate ≥ threshold
  4. Parameter + arena result stored in SketchPopulation
  5. P-UCB selects promising parameter configurations
```

**Priority:** Low — our current per-domain TOML config (riir-ai Plan 026) handles static parameter selection. Co-search would be a future upgrade for auto-tuning.

### Insight 5: Misformalization Detection via Test Lemmas

For the OEIS evaluation, the agent was required to prove **test lemmas** verifying the first few terms of each sequence against its formal definition before attempting the target conjecture. This caught autoformalization errors before wasting compute on wrong problem statements.

**Our mapping:** Our validators (`ConstraintPruner`, `ScreeningPruner`) are trusted components. But they can have bugs. The test-lemma pattern applies: **before running a full arena benchmark, validate the validator against known game positions**. For example:

```text
Validator test lemmas for Bomber:
  1. Known win position → assert validator returns true
  2. Known loss position → assert validator returns false
  3. Known illegal move → assert ConstraintPruner rejects
  4. Edge case (simultaneous death) → assert correct handling
```

We partially do this in unit tests, but the paper's insight is to make test lemmas **part of the evolutionary loop** — if a sketch's test lemmas fail, it's discarded immediately without wasting rating compute.

**Priority:** Low — our existing unit tests + GOAT proofs cover this. But adding `validate_validator()` calls at the start of each arena round would catch runtime bugs earlier.

### Insight 6: Parallelism Requirement for Population Search

The paper's ablation (Figure 10) reveals a critical finding: **running the full evolutionary method with only 1 generator underperforms the basic setup**. The population database only helps when multiple agents contribute asynchronously. With a single agent, sampling from the database is worse than just using the previous session's output.

> "This suggests that sampling is not beneficial unless one has an asynchronous pipeline and uses the database as a way of coordinating agents."

**Our mapping:** This means our `SketchPopulation` should only be activated when `rayon` parallelism is enabled and multiple DDTree branches are being explored simultaneously. For single-threaded decode (e.g., `NoScreeningPruner`), the population adds overhead without benefit.

**Implication for Plan 128:** The feature gate should require both `proof_sketch_evolution` AND parallel execution. Single-threaded fallback should skip population lookup and use basic Q-value bandit.

```text
// Feature gate dependency
proof_sketch_evolution = ["bandit_pruner"]  // needs bandit
// Runtime guard:
if rayon::current_num_threads() > 1 {
    population.sample_p_ucb()  // use evolutionary
} else {
    bandit.select()            // fallback to basic UCB
}
```

### Insight 7: Stochastic Diversity Injection

The evolutionary controller injects diversity by **stochastically adding instructions** like:
- "Decompose unsolved goals"
- "Combine ideas from prior attempts"
- "Try a completely new approach"

This is cheap (no extra compute) and prevents the population from collapsing into a single lineage.

**Our mapping:** Our `BanditPruner` explore arm currently picks random actions. The diversity injection pattern adds **structured exploration strategies**:
- "Decompose" → split a complex move into sub-moves (applicable in Go)
- "Combine" → merge two partial strategies (applicable in Bomber team tactics)
- "New approach" → switch to a completely different opening/heuristic

This maps to our `AbsorbCompress` heuristic promotion: instead of random perturbation, inject structured strategy hints.

**Priority:** Low-Medium — our bandit already explores, but structured exploration hints could improve convergence on hard domains (Go).

---

## Verdict

**HIGH VALUE for architecture validation, MEDIUM VALUE for new code.**

The paper validates our existing model-based/modelless architecture:
- **Basic agent (modelless) = our `NoScreeningPruner` path** — sufficient for 80% of problems
- **Evolutionary agent (model-based) = our `BanditPruner` path** — 2-5× better on hard problems
- **The convergence finding is exactly our finding from Bomber/Go arenas**

The three original distillable concepts (Elo population, Plackett-Luce, goal cache) are genuine improvements but not urgent. The most actionable is the **Global Goal Cache** — it's a straightforward optimization that provides measurable speedup for DDTree.

**Supplementary insights** (4-7) are lower priority but worth noting:
- **EVOLVE-VALUE co-search** — future auto-tuning of arena parameters
- **Misformalization detection** — validator validation at arena start
- **Parallelism requirement** — population search needs ≥2 threads
- **Diversity injection** — structured exploration strategies for bandit

**Recommended plan:** Plan 128 — implement `ProofGoalCache` and `SketchPopulation` behind `proof_sketch_evolution` feature gate, with GOAT proof measuring convergence speedup. Add runtime guard for parallelism requirement.

---

## References

- AlphaProof Nexus: https://arxiv.org/abs/2605.22763
- AlphaProof (olympiad-level): Hubert et al. 2025
- AlphaEvolve (evolutionary): Novikov et al. 2025
- Ralph loop: Huntley et al. 2025
- Formal Conjectures repo: https://github.com/google-deepmind/formal-conjectures
- Erdős problems catalog: https://www.erdosproblems.com
- Terence Tao's AI-Erdős wiki: https://terrytao.wordpress.com