# Research 171: FrontierCS + FrontierSmith → Open-Ended Problem Evolution Arena

**Papers:**
- FrontierCS (arXiv:2512.15699) — Open-ended CS benchmark, 156 problems, partial scoring, verifiable
- FrontierSmith (arXiv:2605.14445) — Closed→open-ended problem synthesis, idea divergence, GRPO training

**Date:** 2026-06-05
**Verdict:** ✅ **ADOPT — 3 novel fusion components. Modelless first, no LLM training. Default ON when GOAT-proven.**

---

## 1. Paper Summaries

### FrontierCS

156 open-ended CS problems across two tracks: **algorithmic** (107: optimization, constructive, interactive) and **research** (49: OS, HPC, AI, DB, PL, Security). No known optimal solutions. Deterministic evaluators assign continuous scores.

Key findings:
- Best frontier model (Gemini 3.0 Pro): **29.37** vs human experts **95.41** — massive gap
- More reasoning tokens ≠ better (GPT-5 medium > high: 15.34 vs 12.63)
- **Micro-optimization trap**: models optimize locally, miss algorithmic structure
- Partial scoring enables RL training with reward signals

### FrontierSmith

Automated pipeline synthesizing open-ended coding problems from closed-ended seeds:
1. **Mutate** — change goals, restrict outputs, generalize inputs
2. **Filter** — coarse LLM judge + **idea divergence** metric (P[different strategies])
3. **Build** — test-case agent + verifier agent with cross-validation

Results: Qwen3.5-9B +8.82 on FrontierCS, +306 Elo on ALE-bench. Competitive with human-curated data.

---

## 2. Creative Fusion — What Neither Paper Does

The papers treat problem synthesis and evaluation as **separate from inference**. Our fusion:

### Insight: Arena-as-Evolver

**Every arena match is a problem synthesis event.** When BanditPruner plays bomber, the game configuration IS a problem instance. When the bandit adapts its arm selection, it's solving an open-ended optimization problem. The arena scores are partial scores. The WASM validator is the deterministic evaluator. We already have 90% of FrontierCS/FrontierSmith — just never connected the pieces.

### Three Novel Components

#### Component 1: `ProblemMutator` Trait (Modelless)

Closed→open-ended mutation, but at the **game configuration level**, not the code level. Takes a game config (arena seed) and produces variant configs via constrained mutation. Modelless: uses deterministic mutation operators (parameter perturbation, constraint injection, objective reweighting).

```rust
/// Mutation operators for game/problem configurations.
/// Modelless: deterministic, no LLM required.
pub trait ProblemMutator: Send + Sync {
    /// Mutate a game configuration into an open-ended variant.
    /// Returns 0+ candidate configs with estimated difficulty delta.
    fn mutate(&self, seed: &GameConfig) -> Vec<MutantConfig>;
}

/// A mutated game configuration with difficulty estimate.
pub struct MutantConfig {
    pub config: GameConfig,
    /// Expected difficulty shift from base. Positive = harder.
    pub difficulty_delta: f32,
    /// Mutation type that produced this variant.
    pub mutation_kind: MutationKind,
}

#[derive(Debug, Clone, Copy)]
pub enum MutationKind {
    /// Change objective weights (e.g., minimize bombs instead of maximize kills)
    GoalReweight,
    /// Add constraints (e.g., max-steps, forbidden zones)
    ConstrainOutputs,
    /// Expand input space (e.g., larger grid, more opponents)
    GeneralizeInputs,
}
```

#### Component 2: `PartialScorer` Trait (Modelless)

Extends `ConstraintPruner` binary accept/reject into graduated scoring. WASM validators return `[0.0, 1.0]` scores instead of `bool`. Enables bandit learning from partial credit, not just win/loss.

```rust
/// Graduated scoring for open-ended problem evaluation.
/// Extends binary ConstraintPruner into continuous quality scoring.
pub trait PartialScorer: Send + Sync {
    /// Score a game trace / solution on [0.0, 1.0].
    /// 0.0 = trivial baseline, 1.0 = reference solution quality.
    fn partial_score(&self, trace: &GameTrace) -> f32;
    
    /// Decompose score into per-criterion breakdown.
    /// Returns criteria name → subscore pairs.
    fn score_breakdown(&self, trace: &GameTrace) -> Vec<(&str, f32)>;
}
```

This slots between `ConstraintPruner` (binary) and `ScreeningPruner` (graded relevance). The key difference: `ScreeningPruner` scores *per-token relevance* during decoding; `PartialScorer` scores *per-episode quality* for bandit reward.

#### Component 3: `IdeaDivergence` Metric (Modelless)

Measures whether different bandit arms (strategies) explore genuinely different algorithmic ideas vs. minor variants of the same approach. Prevents proposer collapse (all arms converge to one strategy).

Modelless implementation: compare score vectors across test cases. If two arms produce similar score patterns across diverse problem instances, they're likely using similar strategies.

```rust
/// Measures strategic diversity across bandit arms.
/// Modelless: uses execution score vectors, no LLM judge.
pub struct IdeaDivergence {
    /// Per-arm score vectors across recent problem instances.
    arm_scores: Vec<Vec<f32>>,
    /// Minimum divergence threshold for acceptance.
    threshold: f32,
}

impl IdeaDivergence {
    /// Compute divergence between two arms' score vectors.
    /// Uses normalized L2 distance (matches FrontierSmith §3.2 eq. 3).
    pub fn divergence(&self, arm_a: usize, arm_b: usize) -> f32;
    
    /// Check if a new arm is sufficiently different from existing arms.
    /// Returns true if min divergence to any existing arm > threshold.
    pub fn is_novel(&self, new_arm_scores: &[f32]) -> bool;
}
```

---

## 3. Mapping to Existing Architecture

### What We Already Have

| FrontierCS/FrontierSmith Concept | Our Equivalent | Status |
|---|---|---|
| Arena proofs (156 problems) | Bomber/FFT/Go arenas (Plan 033/047/075) | ✅ |
| Deterministic evaluators | WASM validators (`riir-validator-sdk`) | ✅ |
| Partial scoring | `ScreeningPruner::relevance()` [0.0, 1.0] | ✅ (per-token, not per-episode) |
| Problem mutation (closed→open) | G-Zero `TemplateProposer` (R021) | ⚠️ Template-only |
| Idea divergence filtering | SDE path diversity (R012), blind-spot analysis (R093) | ⚠️ Solution-space, not problem-space |
| Test-case generation | Arena random seeds | ✅ |
| GRPO training | GZeroLoop + DPO/GRPO (Plan 059) | ✅ (model-based) |
| Quality-diversity | Plackett-Luce + P-UCB (Plan 128/143) | ✅ |
| Adaptive compute | ThinkingBandit (Plan 194) | ✅ |
| Skill lifecycle (MUSE) | Plan 189/214 | ✅ |

### The Gap → The Fusion

```
┌─────────────────────────────────────────────────────────────┐
│                    Arena-as-Evolver                          │
│                                                             │
│  ProblemMutator ──→ Arena ──→ PartialScorer ──→ BanditPruner│
│       │                │            │              │        │
│  MutationKind    GameTrace    score_breakdown    arm update  │
│  difficulty_delta  episodes    per-criteria       strategy   │
│                                reward             selection  │
│                                                             │
│  IdeaDivergence ──→ filters arm convergence                  │
│                                                             │
│  ←── GZeroLoop (Plan 049) ──→                               │
│  ←── MUSE lifecycle (Plan 189) ──→                          │
│  ←── ThinkingBandit (Plan 194) ──→                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. GOAT Gate Analysis

| Component | Default Feature | GOAT Proof Required | Perf Impact |
|---|---|---|---|
| `PartialScorer` | `partial_scoring` | Arena score ≥ binary scoring on bomber + go | Negligible — just changes reward from bool→f32 |
| `ProblemMutator` | `problem_mutator` | Mutated configs produce ≥1.5× arm diversity vs fixed seeds | None — config generation is offline |
| `IdeaDivergence` | `idea_divergence` | Bandit convergence time ↓ with divergence filter | Minimal — score vector comparison is O(arms²) |

**If GOAT, default ON.** No perf hurt — all components are inference-time only, no LLM training.

---

## 5. Commercial Alignment (per Verdict 003)

- **Engine/Fuel split intact**: `ProblemMutator` + `PartialScorer` + `IdeaDivergence` are engine (MIT)
- **Fuel**: Domain-specific mutation operators + partial scoring rubrics for specific game/CS domains (SaaS)
- **Curator model**: Curators submit game configs → platform generates mutation operators + partial scoring rubrics as WASM validators

---

## 6. Why This Is More Than Direct Mapping

FrontierCS evaluates LLMs on open-ended CS problems. FrontierSmith synthesizes those problems. Neither connects problem synthesis to **inference-time learning** (bandits). Our fusion:

1. **Bandit learns WHICH problems to generate** — not just which solutions to pick. The bandit's arm space includes problem variants, not just strategies.
2. **Partial scoring enables continuous reward** — binary win/loss → graduated quality. This is exactly what bandits need: smooth gradient signal.
3. **Idea divergence prevents collapse** — the #1 failure mode in self-play (Plan 111, Research 075). This is a structural fix, not a heuristic.
4. **Arena-as-benchmark** — every arena match IS a FrontierCS-style evaluation. No separate benchmark needed.

The meta-insight: **FrontierCS is what our arena already is. FrontierSmith is what our ProblemMutator + IdeaDivergence will be.** The fusion closes the loop: mutate problems → evaluate in arena → bandit learns → generate better problems.

---

## TL;DR

FrontierCS + FrontierSmith validate our arena approach and add three missing pieces: `ProblemMutator` (closed→open config mutation), `PartialScorer` (graduated episode scoring), `IdeaDivergence` (strategic novelty metric). All modelless, no LLM training. Slots cleanly into existing bandit/arena/validator/MUSE infrastructure. GOAT-gated, default ON if proven.

Next research number for katgpt-rs: 171 (this file).
