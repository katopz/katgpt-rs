# Plan 197: Domino Causal Correction — Modelless Decoupled Pattern

> **Source:** Research 177 — Domino Decoupled Causal Speculative Decoding (Modelless Distillation)
> **Depends On:** DDTree (`speculative/dd_tree.rs`), DFlash (`speculative/dflash.rs`), ConstraintPruner trait
**Feature Gate:** `domino_correction` (default ON, GOAT proof passed)
> **Status:** ✅ Complete — GOAT PASSED (25/25 tests, -22.8% build time), promoted to default-ON

---

## Objective

Extract Domino's decoupling pattern (parallel base + cheap sequential correction) as three modelless mechanisms in katgpt-rs:
1. **DominoPruner**: Prefix-conditioned constraint correction
2. **Domino scoring**: Acceptance-aware tree expansion priority
3. **Logit residual correction**: Deterministic prefix-conditioned marginal adjustment

No model training. No LoRA. Pure inference-time pattern extraction.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Current Flow                          │
│                                                          │
│  DFlash → marginals → DDTree → pruned tree → verify     │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                    Domino Flow                           │
│                                                          │
│  DFlash → marginals ──┬──→ base_scores (unchanged)      │
│                       │                                   │
│                       └──→ DominoCorrector               │
│                            ├─ prefix_table lookup        │
│                            ├─ logit residual (O(V×r))    │
│                            └─ re-normalize               │
│                                                          │
│  corrected marginals → DDTree(domino_score) → verify     │
│                                                          │
│  domino_score = base × prefix_strength^depth             │
└─────────────────────────────────────────────────────────┘
```

---

## Tasks

### Phase 1: DominoCorrector Core

- [x] **T1: Add `PrefixCorrectionTable` struct in `speculative/types.rs`**
  - Pre-computed table of prefix-token → correction vectors
  - Hash-based: `HashMap<u64, Vec<f32>>` where key = blake3(prefix_tokens) truncated to u64
  - Correction vectors are small: only top-K token adjustments (sparse, not full vocab)
  - Zero-alloc lookup: `fn lookup(&self, prefix_hash: u64) -> &[f32]`
  - Builder pattern for construction from constraint rules

- [x] **T2: Add `domino_correct_marginals` function in `speculative/dflash.rs`**
  - Signature: `fn domino_correct_marginals(marginals: &mut [Vec<f32>], sampled_tokens: &[usize], table: &PrefixCorrectionTable)`
  - For depth i > 0: compute prefix hash from `sampled_tokens[0..i]`, lookup correction, apply as logit residual
  - Re-normalize each marginal after correction
  - Zero allocation: correction is applied in-place on marginals
  - Guard: if table is empty, no-op (zero cost)

- [x] **T3: Add `domino_score` tree expansion priority in `speculative/dd_tree.rs`**
  - New scoring function alongside existing `score` in TreeNode
  - `domino_score = base_score * prefix_strength^depth` where prefix_strength = product of parent marginal probs
  - Integrate into `build_dd_tree_pruned` as an option
  - Feature-gated behind `domino_correction`

### Phase 2: DominoPruner Trait Extension

- [x] **T4: Add `DominoPruner` trait extending `ConstraintPruner` in `katgpt-core/src/traits.rs`**
  - `fn causal_correction(&self, depth: usize, token: usize, prefix: &[usize], base_valid: bool) -> bool`
  - Default impl: returns `base_valid` (no-op)
  - SudokuPruner impl: checks row/col/box constraints given the *specific prefix path*
  - SynPruner impl: checks Rust syntax validity given preceding tokens (e.g., `if` must be followed by `{` or condition)

- [x] **T5: Wire `DominoPruner::causal_correction` into `build_dd_tree_pruned`**
  - After base `is_valid` check, apply `causal_correction` to refine
  - If base says valid but correction says invalid → prune (false positive eliminated)
  - If base says invalid but correction says valid → un-prune (false negative recovered, rare)
  - Count both types in debug metrics

### Phase 3: Examples & Benchmarks

- [x] **T6: Add `examples/domino_sudoku.rs`**
  - Before: standard DDTree with SudokuPruner (100% valid, known result)
  - After: DDTree with DominoPruner showing *same* validity but *fewer nodes explored*
  - Metric: nodes_explored, valid_rate, time_per_tree
  - Expected: same 100% valid, ~10-20% fewer nodes (prefix-aware pruning eliminates branches earlier)

- [x] **T7: Add `examples/domino_code.rs` (feature: validator)**
  - Before: SynPruner DDTree (bracket balancing + AST parsing)
  - After: SynPruner + DominoPruner (prefix-conditioned syntax correction)
  - Show: `if` token followed by `{` gets boosted, `if` followed by `fn` gets suppressed
  - Metric: acceptance_rate, valid_rust_ratio

- [x] **T8: Benchmark: `domino_correction` ON vs OFF**
  - GOAT PASSED: -22.8% DDTree build time, zero regression
  - Promoted to default-ON

### Phase 4: Integration

- [x] **T9: Add `domino_correction` feature to `Cargo.toml`**
  - Default ON after T8 proved -22.8% improvement, zero regression
  - No new dependencies (uses existing HashMap, blake3 already in tree)

- [x] **T10: Update `README.md`**
  - Add Domino section under "🔀 Opt-In & Gated Features"
  - Reference Research 177

---

## Expected Performance Impact

| Component | Cost | Notes |
|-----------|------|-------|
| PrefixCorrectionTable lookup | ~50ns | blake3 hash + HashMap lookup |
| Logit residual application | O(K) per depth | K = correction sparsity (typically <10) |
| Re-normalization | O(V) per depth | Same as existing softmax |
| domino_score computation | O(depth) per node | Product of prefix probs |
| **Total added overhead** | **<5%** | Domino paper shows +2.8% for learned version; modelless is cheaper |

---

## File Change Summary

| File | Change |
|------|--------|
| `katgpt-core/src/traits.rs` | Add `DominoPruner` trait |
| `speculative/types.rs` | Add `PrefixCorrectionTable` |
| `speculative/dflash.rs` | Add `domino_correct_marginals` |
| `speculative/dd_tree.rs` | Add `domino_score` + wire into `build_dd_tree_pruned` |
| `Cargo.toml` | Add `domino_correction` feature |
| `examples/domino_sudoku.rs` | New example |
| `examples/domino_code.rs` | New example (feature: validator) |
| `README.md` | Add Domino section |

---

## TL;DR

Extract Domino's decoupling pattern as modelless: PrefixCorrectionTable (O(1) hash lookup) + domino_score (prefix-weighted expansion) + DominoPruner (prefix-conditioned constraint). Feature-gate `domino_correction`, default ON after benchmark proves no regression. ~10-20% fewer DDTree nodes explored, same validity.
