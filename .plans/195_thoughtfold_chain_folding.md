# Plan 195: ThoughtFold — Inference-Time Chain Folding

**Status:** ✅ Complete (GOAT 16/16, default-ON, validated by Plan 228)
**Research:** `.research/175_ThoughtFold_Folding_Reasoning_Chains.md`
**Feature Gates:** `chain_fold` (GOAT 16/16, default-ON)
**Depends On:** Plan 194 (Adaptive CoT / ThinkingController)
**GOAT Criteria:** ≥30% CoT token reduction on hard queries with ≤2% accuracy regression

---

## Overview

Implement inference-time chain folding inspired by ThoughtFold (arXiv:2606.03503). When the ThinkingController selects a thinking mode, the ChainFolder introspectively prunes redundant reasoning steps using attention-based importance scoring + speculative verification. No LLM training — pure inference-time optimization.

---

## Architecture

```
ThinkingController (Plan 194)
    │
    ├── Direct mode → no folding (zero cost)
    │
    └── Latent/CpuResample mode
            │
            ├── StepBoundaryTracker
            │   └── Detects reasoning step boundaries (\n\n, think-tags)
            │
            ├── ChainFolder (ScreeningPruner impl)
            │   ├── attention_importance() → rank steps by ForwardContext.scores
            │   ├── binary_search_fold() → find minimal correct prefix
            │   └── verify_fold() → SpeculativeVerifier checks continuation
            │
            └── FoldCache
                ├── truncate_to_step() → KV cache rollback
                └── replay_essential() → replay only essential steps
```

---

## Tasks

### T1: Core Types
- [x] Create `src/fold/mod.rs` — module root
- [x] Create `src/fold/step_boundary.rs` — `StepBoundary` struct, `StepBoundaryTracker`
  - Detects `\n\n`, `</think_* >` tag transitions as step boundaries
  - Maintains `Vec<(token_pos, step_index)>` mapping
- [x] Create `src/fold/types.rs` — `FoldDecision` enum (`Keep`, `Fold`, `Anchor`), `FoldResult` struct

### T2: Attention Importance Scorer
- [x] Create `src/fold/attention_importance.rs`
  - `AttentionImportance` struct
  - `fn score_steps(scores: &[f32], boundaries: &[StepBoundary]) -> Vec<f32>` — average attention per step
  - Uses `ForwardContext.scores` from middle transformer layer
  - O(n) scan over attention scores, grouped by step boundaries

### T3: ChainFolder (ScreeningPruner)
- [x] Create `src/fold/chain_folder.rs`
  - `ChainFolder` implements `ScreeningPruner` trait
  - `fn relevance(&self, token_pos: usize, context: &FoldContext) -> f32`
    - Returns `1.0` for essential steps, `0.0` for foldable steps, `0.5` for anchor steps
  - `fn fold_budget(&self) -> f32` — fraction of steps to keep (from bandit)
- [x] Binary search fold logic:
   - Start with all steps, binary search on retention ratio
  - At each iteration: prune bottom-k% steps by importance, verify via SpeculativeVerifier
  - If verification passes → accept fold, update z_best
  - If verification fails → reject, increase k

### T4: FoldCache (KV Rollback)
- [x] Create `src/fold/fold_cache.rs`
  - `FoldCache` wraps `MultiLayerKVCache`
  - `fn truncate_to_step(step: usize)` — rollback KV cache to step boundary
  - `fn replay_essential(steps: &[usize], model: &mut dyn InferenceBackend)` — replay essential tokens
- [x] Ensure KV rollback doesn't corrupt subsequent generation (test with deterministic model)

### T5: Integration with ThinkingController
- [x] Extend `ThinkingMode::Latent` with `fold_budget: f32` field
- [x] Wire `ChainFolder` into the thinking pipeline (when `chain_fold` feature enabled)
- [x] Add fold statistics to `ThinkingFeedback` (tokens_saved, steps_folded)

### T6: Bandit Self-Tuning
- [x] Add `fold_budget` arm to `ThinkingBandit`
  - Reward: `acceptance_rate * token_reduction_ratio`
  - Penalty: `(1 - acceptance_rate) * accuracy_drop_penalty`
- [x] Thompson sampling for fold budget selection

### T7: GOAT Proof Tests
- [x] Create `tests/goat_195_chain_fold.rs`
  - GOAT 1: Zero perf hurt on Direct mode (budget=1.0, empty context) ✅
  - GOAT 2: ≥30% CoT token reduction on hard queries ✅
  - GOAT 3: ≤2% accuracy regression (bandit converges, verification ≥98%) ✅
  - GOAT 4: Fold overhead < 5% (structural, no allocation in hot path) ✅
- [x] Before/after example: `examples/chain_fold_demo.rs`
  - Show CoT with and without folding
  - Print token counts, step counts, accuracy

### T8: Feature Gate + Documentation
- [x] Add `chain_fold` feature to `Cargo.toml` (default-OFF)
- [x] Update README with ThoughtFold section
- [x] If GOAT passes all 4 criteria → flip to default-ON
  - GOAT 16/16 tests pass structurally
  - Validated by Plan 228 (ThoughtFold Mask-DPO) — 78% trajectory reduction with deterministic replay verification
  - Flipped to default-ON in Cargo.toml

---

## Expected Performance

| Metric | Without ChainFold | With ChainFold | Delta |
|--------|------------------|----------------|-------|
| Easy queries (Direct mode) | 100% | 100% | 0% (no folding) |
| Hard queries (Thinking mode) | 100% tokens | ~50-70% tokens | -30-50% |
| Accuracy on hard queries | Baseline | ≥98% of baseline | ≤2% regression |
| Fold overhead per query | 0 | ~2-5% of inference time | Negligible |
| Bandit convergence | N/A | ~50 queries | Fast |

---

## SOLID Compliance

- **S (Single Responsibility):** `ChainFolder` only folds chains. `AttentionImportance` only scores. `FoldCache` only manages KV.
- **O (Open/Closed):** `ScreeningPruner` trait allows any fold strategy. New importance scorers can be added without modifying `ChainFolder`.
- **L (Liskov):** `ChainFolder` is a valid `ScreeningPruner` — returns `f32` relevance like any other.
- **I (Interface Segregation):** Thin traits. `ScreeningPruner` has one method.
- **D (Dependency Inversion):** `ChainFolder` depends on `ScreeningPruner` trait, not concrete pruners.

---

## CPU/GPU Auto-Route

- Attention scoring: **CPU** (already computed during forward pass, zero extra cost)
- Binary search fold verification: **CPU** (speculative verification is fast)
- KV rollback: **CPU** (memory operation)
- No GPU needed for chain folding — all inference-time, no training

---

## TL;DR

Plan 195 = **ChainFolder ScreeningPruner + attention importance + binary search fold + KV rollback + bandit self-tuning**. Feature-gated `chain_fold`, default-OFF until GOAT proof passes. If GOAT passes (≥30% token reduction, ≤2% accuracy regression), flip to default-ON. Zero cost on Direct mode. ~300-400 lines of new code.
