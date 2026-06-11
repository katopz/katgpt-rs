# Research 215: ECHO — Environment Prediction as Inference-Time Dense Supervision

**Date:** 2026-06
**Source:** arXiv:2605.24517 — ECHO: Terminal Agents Learn World Models for Free (Shrivastava et al., 2026)
**Verdict:** GOAT — High-gain for modelless inference-time environment prediction scoring
**Target:** Modelless (katgpt-rs) primary

---

## Executive Summary

ECHO proves that **environment observations are a free, dense supervision signal** already present in every rollout. By adding a cross-entropy loss on environment tokens alongside the standard policy-gradient loss on action tokens, ECHO doubles GRPO pass@1 (Qwen3-8B: 2.70→5.17%, Qwen3-14B: 5.17→10.79%) — with **zero architecture changes, zero extra rollouts**, same forward pass.

**The modelless fusion opportunity is NOT replicating ECHO's training loss** (that's model-based, riir-ai territory). Our opportunity is distilling ECHO's core insight — **prediction quality correlates with policy quality** — into our existing DDTree + BanditPruner + ScreeningPruner pipeline at inference time.

---

## Paper Core

### 1. The Hybrid Objective

```
L_ECHO(θ) = L_GRPO(θ; A) + λ · L_Env(θ; O')
```

- **L_GRPO**: Standard policy gradient on action tokens A
- **L_Env**: Cross-entropy on environment observation tokens O'
- **λ = 0.05**: Environment loss weight (0.01–0.05 safe, 0.2 degenerate)
- Same forward pass — zero overhead on inference compute

### 2. Key Results

| Metric | GRPO Alone | + ECHO | Gain |
|--------|-----------|--------|------|
| Qwen3-8B pass@1 | 2.70% | 5.17% | **+91%** |
| Qwen3-14B pass@1 | 5.17% | 10.79% | **+109%** |
| Env CE (off-policy) | 0.29 nats | 0.07 nats | **-76%** |
| SFT gap recovery | — | ~50% | Without 15K demos |
| Verifier-free self-improvement | — | +10pp | PyTerm, env-loss only |

### 3. What Worked / Didn't

| What | Verdict |
|------|---------|
| λ ∈ [0.01, 0.05] | ✅ Safe range |
| Auto-annealing λ | ✅ Prevents late-training instability |
| Env-only targets (stdout, not warnings) | ✅ Best target selection |
| Verifier-free on PyTerm | ✅ +10pp from env-loss alone |
| λ = 0.2 | ❌ Degenerate rollouts |
| TBLite verifier-free | ❌ Weak env-action coupling |
| SFT-initialized models | ❌ Less marginal gain (already good) |

### 4. Concurrent/Related Work

| Paper | Key Idea | Difference from ECHO |
|-------|----------|---------------------|
| **PaW** (arXiv:2606.02388) | Co-train policy + world model on next-obs | Nearly identical, concurrent |
| **CWM** (Meta, 2025) | Separate 32B world model on execution traces | Separate model, offline |
| **RLTF** (2026) | Predict judge critiques | Needs judge, not raw env |

---

## Fusion: Novel Modelless Applications of ECHO Insight

The paper's training-time approach is **not directly applicable** to our modelless constraint. But ECHO proves a deeper principle that IS applicable at inference time:

### Insight 1: Prediction Quality ≈ Policy Quality

ECHO shows that policies that better predict environment dynamics also better navigate those dynamics. At inference time, we can **score actions by how predictable their outcomes are** — not by training a predictor, but by using the game's own forward model speculatively.

**Our novel fusion: `EnvPredictorPruner`** — a `ScreeningPruner` that:
1. For each candidate action, runs the game's deterministic forward model (already exists for game engines)
2. Scores the resulting state by how "expected" it is (entropy of state features vs historical average)
3. Boosts actions leading to predictable states, suppresses actions leading to chaotic/surprising states
4. Uses bandit to learn which environments benefit from this scoring

This is **not ECHO's training loss** — it's the inference-time dual: instead of training to predict the environment, use the environment to score predictions.

### Insight 2: Failed Rollouts Are Information

ECHO's key finding: failed rollouts contain rich evidence about environment dynamics. Standard GRPO discards this. At inference time, our DDTree already explores failed branches — but we don't currently **learn from them across sessions**.

**Our novel fusion: `PredictionVerifier` bandit arm** — track prediction-vs-reality across DDTree branches:
1. During DDTree exploration, log predicted outcomes per branch
2. After verification (LeviathanVerifier), compare actual vs predicted
3. Feed accuracy signal into BanditPruner reward
4. AbsorbCompress promotes prediction strategies with high verification accuracy

### Insight 3: Dense Intra-Trajectory Credit

ECHO's environment prediction creates dense per-token credit. At inference time, we can approximate this via **step-level scoring** in DDTree:

**Our novel fusion: `ShapedBanditPruner`** — from Research 025 (StepCodeReasoner):
1. Intra-trajectory advantage: `Â(i) = r_i × (1 + future_accuracy)`
2. Steps that "pave the way" (lead to verified good outcomes) get boosted
3. Steps that are locally plausible but lead nowhere get suppressed
4. Pure post-hoc computation on DDTree verification paths — modelless

### Insight 4: Verifier-Free Self-Improvement

ECHO's most striking result: **environment prediction loss alone** (+10pp) enables self-improvement without any reward signal. At inference time, this maps to:

**Our novel fusion: `PredictionConsistencyGate`** — if the model's marginal predictions are consistent (low entropy across multiple DDTree branches), the action is likely correct. If predictions are inconsistent (high inter-branch entropy), the action needs more exploration. This is a **modelless consistency check** that requires no training — just entropy measurement on the existing DDTree output.

---

## Distillation: What's Training-Time Only (riir-ai)

- The auxiliary cross-entropy loss `L_Env` on environment tokens
- Joint GRPO + env-prediction training
- λ scheduling and auto-annealing
- Any weight updates (LoRA or full)

## What's Inference-Time (katgpt-rs)

- Environment forward model scoring (game engines have deterministic forward models)
- Prediction-vs-reality verification (compare speculative branches against actual outcomes)
- Bandit-driven prediction strategy selection
- Consistency-based confidence scoring (entropy across DDTree branches)
- Shaped intra-trajectory credit (post-hoc computation)

---

## Existing Infrastructure (80% Built)

| Component | What | ECHO Role |
|-----------|------|-----------|
| `ScreeningPruner::relevance()` | Token quality scoring | Environment prediction as relevance signal |
| `BanditPruner<P>` | Adaptive arm selection | Track which prediction strategies work |
| `DDTree` | Speculative tree search | Multi-step environment rollouts |
| `WasmPruner` | Deterministic validation | Verify predictions against game rules |
| `AbsorbCompress` | Promote winning patterns | Lock in working prediction strategies |
| `TrialLog` | Episode history | Prediction vs reality log |
| `HotSwapPruner` | Runtime swap | Change prediction strategies dynamically |
| `ConstraintPruner::is_valid()` | Binary accept/reject | Reject actions with invalid predicted outcomes |
| `NextLat` (Plan 217) | Belief-state drafter | Frozen MLP as environment predictor |
| Freeze/thaw | Cross-session persistence | Prediction skills survive sessions |

## Missing Primitives (The 20% Gap)

1. **`EnvPredictorPruner`** — ScreeningPruner that scores actions by predicted outcome quality
2. **`PredictionVerifier`** — Compare predicted state vs actual state, feed into bandit reward
3. **`PredictionConsistencyGate`** — Entropy-based confidence from DDTree branch consistency

---

## GOAT Gate

Feature flag: `echo_env_predictor` (default-OFF until GOAT proof passes)

### GOAT Proofs Required

| # | Metric | Threshold | Measurement |
|---|--------|-----------|-------------|
| G1 | Bomber HL score with EnvPredictorPruner | ≥ baseline (no regression) | Arena benchmark |
| G2 | Prediction accuracy bandit convergence | ≥70% correct after 100 rounds | Unit test |
| G3 | DDTree branch consistency improvement | ≥15% entropy reduction on hard queries | Benchmark |
| G4 | No hot-path latency regression | ≤5% overhead per token | Micro-bench |

---

## Verdict by 003 Commercial Strategy

- **Modelless first** ✅ — inference-time only, no LLM training
- **Engine territory** ✅ — fits katgpt-rs engine, no fuel dependency
- **SOLID/DRY** ✅ — extends existing ScreeningPruner/BanditPruner traits
- **Tests/examples** ✅ — bomber arena before/after, prediction accuracy test
- **CPU/GPU auto-route** ✅ — forward model is CPU (game engine), no GPU needed
- **Tier aware** ✅ — prediction scoring in Hot tier, verification in Warm tier
- **Adaptive threshold** ✅ — bandit learns when env prediction helps vs hurts

**Decision: GAIN — implement as feature-gated plan, GOAT gate before promotion.**

---

## TL;DR

ECHO proves environment observations are dense supervision. We distill this to modelless: score DDTree actions by predicted-outcome quality using the game's own forward model, verify predictions against reality, and bandit-learn which prediction strategies work. The infrastructure is 80% built — need 3 new primitives (EnvPredictorPruner, PredictionVerifier, PredictionConsistencyGate) wired into existing BanditPruner + DDTree + AbsorbCompress pipeline.
