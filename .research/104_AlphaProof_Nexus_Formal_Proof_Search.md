# Research 104: AlphaProof Nexus — AI-Driven Formal Proof Search

**Paper:** [arXiv:2605.22763](https://arxiv.org/abs/2605.22763) — Advancing Mathematics Research with AI-Driven Formal Proof Search
**Authors:** George Tsoukalas, Anton Kovsharov, Sergey Shirobokov, et al. (Google DeepMind, 2026)
**Date:** 2026-05-25
**Verdict:** 🟡 **Conditional Adopt — methodology paper validating our modelless-first + bandit-driven search architecture. Key extractable primitives: Plackett-Luce Elo ranking for partial solutions, P-UCB evolutionary sampling, global goal caching. Proof sketch evolution → Percepta compiler stack (super-GOAT, feature-gated).**

---

## TL;DR

AlphaProof Nexus is a framework for LLM-aided formal proof generation in Lean. Four agent variants tested: (A) basic Ralph loop, (B) basic + AlphaProof tool, (C) basic + evolution, (D) full-featured (evolution + AlphaProof). Agent (D) solved 9/353 open Erdős problems including two open for 56 years, proved 44/492 OEIS conjectures, and resolved open problems in optimization, graph theory, algebraic geometry, and quantum optics. Surprisingly, the basic agent (A) also solved all 9 Erdős problems but at higher cost on harder problems.

**Key finding:** Simple agentic loops with compiler feedback are surprisingly effective as LLMs improve. The full-featured agent's advantage is concentrated on the hardest problems.

---

## Core Mechanisms

### 1. Agent Architecture Spectrum

| Agent | Components | Cost Profile | When Best |
|-------|-----------|-------------|-----------|
| (A) Basic | N independent Ralph loops + Lean compiler feedback | Low/parallel | Most problems |
| (B) + AlphaProof | (A) + RL theorem prover as tool | Medium | Problems decomposable into subgoals |
| (C) + Evolution | (A) + population database + Elo ranking | Medium | Problems needing diverse strategies |
| (D) Full | (B) + (C) combined | High | Hardest problems (Erdős #125, #138) |

**Distillation for us:** Our DDTree + Bandit infrastructure already implements Agent (C)-style search. Adding AlphaProof-style focused tools (Agent B) is the gap — our SpeculativeVerifier trait serves this role.

### 2. Plackett-Luce Elo Rating

The paper uses a Plackett-Luce model (not simple Elo) for ranking incomplete proof sketches:

- Each sketch `s` has latent strength `λ_s` with hierarchical prior `Gamma(1, r_s)`, `r_s ~ Gamma(1, 1)`
- Gibbs sampling (1000 samples, 200 burn-in) for posterior inference
- Elo score = `1200 + 400 * log10(λ_mean)`
- Thompson sampling for sketch selection in matches (P=7 per match)

**Key difference from our Bandit:** We use UCB1/EpsilonGreedy on scalar rewards. Plackett-Luce handles **relative rankings** of multiple candidates simultaneously, not just pairwise comparisons. This is more information-efficient when evaluating partial solutions (our DDTree sketches).

### 3. P-UCB Evolutionary Sampling

Selection uses Predictor + Upper Confidence Bound:

```
score = q + c * sqrt(ln(ΣV_i) / (v + 1))
```

Where:
- `q` = normalized Elo in [0,1] from top-64 filtered population
- `v` = visit count for this sketch
- `c` = exploration constant (0.2)
- Top-64 pre-filter prevents search collapse

**Distillation for us:** Our SR²AM (Plan 112) uses UCB1 but doesn't do population pre-filtering or Elo-normalized base scores. Adding top-K filtering + Elo normalization could improve our configurator bandit's convergence.

### 4. Global Goal Cache

Independent agents generate identical subgoals. The system:
1. Computes deep hash of Lean state + target (goal_id)
2. Checks cache before dispatching to AlphaProof
3. Caches proof/disproof results for future reuse
4. Batches novel goals via non-blocking RPCs

**Distillation for us:** Our Fourier MCTS transposition table (Plan 061) does state hashing for board positions. The "goal cache" generalizes this to arbitrary proof/search states. Could enhance our DDTree's memoization.

### 5. EVOLVE-BLOCK / EVOLVE-VALUE Markers

Constrained mutation system:
- `EVOLVE-BLOCK-START/END`: agent can modify definitions, lemmas, proof steps
- `EVOLVE-VALUE-START/END`: agent can change parameter values
- Everything else is immutable (theorem statement, imports)

**Distillation for us:** Maps to our Percepta compiler stack's proof sketch system. The constrained mutation pattern prevents hallucinated modifications to problem statements — our `ConstraintPruner` trait serves a similar validation role.

### 6. Rater Agent Prompt Design

The rater evaluates sketches on three criteria (descending priority):
1. **Strategic Robustness**: generalization over specialization, avoid overfitting
2. **Decomposition Quality**: "good gaps" (routine) vs "bad gaps" (core insight missing)
3. **Logical Correctness**: valid plan, coherent steps

**Key insight:** "A sketch with good gaps (even if AlphaProof fails) > A sketch with no gaps (but dead-end strategy)." This validates our DDTree's approach of keeping partial solutions with unresolved sub-trees.

---

## Failure Modes (Important for Our System)

1. **Offloading core difficulty**: Agent frequently moved the hard part into a single `sorry` that restated the target. Explicitly prompting against this failed to prevent it.
2. **Hallucinated lemmas**: Top sketches relied on `sorry`-marked lemmas claimed as "established results" that were hallucinations.
3. **Search variance**: Per-problem costs exhibit high variance due to stochastic nature.

**Mitigation for our DDTree:** Our `ConstraintPruner.is_valid()` + `ScreeningPruner.relevance()` already guard against (1) and (2). The high variance observation validates our Bandit's exploration-exploitation tradeoff.

---

## Cost Analysis

- Per-problem inference cost: ~$200-1500 USD (full agent)
- AlphaProof: ~$60 USD per problem (27.5 TPU hours on v6e)
- Basic agent (A): cheaper on easy problems, 2-5× more expensive on hard ones
- Wall-clock: 10-40 hours per successful proof

**For us:** At our scale (game AI, not math research), the basic agent pattern is sufficient. The evolutionary + Elo enhancement is worth it for hard game-state exploration (Go endgames, Bomber optimal play).

---

## Distillation Mapping

| Paper Mechanism | Our Equivalent | Enhancement Opportunity |
|----------------|---------------|------------------------|
| Plackett-Luce Elo | Bradley-Terry ranking (Plan 040) | ✅ We already have BT. Plackett-Luce is a natural generalization for multi-candidate ranking |
| P-UCB sampling | SR²AM UCB1 (Plan 112) | 🟡 Add top-K pre-filter + Elo normalization |
| Global goal cache | Fourier MCTS transposition (Plan 061) | 🟡 Generalize to DDTree memoization |
| EVOLVE-BLOCK markers | ConstraintPruner trait | ✅ Already aligned |
| Rater agent | LoRA-as-Judge (riir-ai) | ✅ Already aligned |
| Basic agent loop | G-Zero modelless (Plan 049) | ✅ Validates modelless-first approach |
| Population database | DDTree frontier | 🟡 Add Elo scores to DDTree nodes |

---

## What NOT to Distill

1. **Lean-specific infrastructure**: We're not building a theorem prover. Lean compiler feedback → our game validator feedback.
2. **Gemini 3.1 Pro specifics**: Their LLM choice doesn't affect our architecture.
3. **Math domain knowledge**: The Erdős/OEIS results are impressive but irrelevant to our game AI.
4. **AlphaProof RL training**: Their RL-based theorem prover is too expensive for our use case. Our SpeculativeVerifier serves a lighter version of this role.

---

## Verdict

**🟡 Conditional Adopt** — The paper validates our architectural choices (modelless first, bandit-driven search, compiler/validator feedback loops). Three specific enhancements are worth implementing:

1. **Plackett-Luce ranking** for DDTree nodes (generalizes our BT ranking to multi-candidate)
2. **P-UCB sampling** for SR²AM (top-K filter + Elo normalization)
3. **Global goal cache** for DDTree (deep hash memoization across independent search threads)

These are infrastructure improvements, not new features. They enhance what we already have.

**Super-GOAT potential:** The proof sketch evolution pattern (EVOLVE-BLOCK + population database + Elo) applied to our Percepta compiler stack could be a genuine differentiator. If we can evolve proof sketches for our transformer-VM programs the same way Nexus evolves Lean proofs, that's a novel capability. **Feature-gate this** — it's a selling point.

**No game-specific distillation needed.** This is pure infrastructure that lives in katgpt-rs. The game domains benefit indirectly through better search.

---

## References

- Paper: [arXiv:2605.22763](https://arxiv.org/abs/2605.22763)
- Results repo: [github.com/google-deepmind/alphaproof-nexus-results](https://github.com/google-deepmind/alphaproof-nexus-results)
- AlphaProof: Hubert et al., "Olympiad-level formal mathematical reasoning with RL," Nature 2025
- AlphaEvolve: Novikov et al., arXiv:2506.13131, 2025
- Related research: Research 088 (AlphaProof Nexus formal proof search — existing), Research 040 (Bradley-Terry ranking)
