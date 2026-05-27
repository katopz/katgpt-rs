# Plan 157: Sigmoid Margin Loss + Retrieval Margin Diagnostic

**Research:** 123 (TopK Dimensionality Barrier for Retrieval)
**Feature gate:** `sigmoid_margin` (opt-in, requires `maxsim`)
**Depends on:** Plan 080 (MaxSim scoring)
**Date:** 2026-05-27

## Goal

Implement the sigmoid margin loss from arXiv 2605.23556 (Bangachev et al.) as a reusable embedding training primitive, plus a `compute_margin` diagnostic for measuring retrieval quality. The paper proves this loss achieves O(log n) dimension scaling vs InfoNCE's O(n^(1/3)).

## Background

The paper's core finding: SigLIP-style element-wise sigmoid loss `softplus(t · (score - b) · sign)` produces margin-optimal embeddings with dramatically fewer dimensions than InfoNCE/contrastive loss. For k=2:

| n | Sigmoid min d | InfoNCE min d |
|---|--------------|---------------|
| 20 | 6 | 10 |
| 100 | 8 | 23 |
| 200 | 9 | 30 |
| 220 | 10 | **FAIL** |

We already use sigmoid extensively (SDAR, EGA, SdpaOutputGate, GeGELU) but NOT for embedding training. This plan closes that gap.

## Task

- [x] T1: `sigmoid_margin_loss` function in `katgpt-core/src/simd.rs`
- [x] T2: `compute_retrieval_margin` diagnostic in `katgpt-core/src/simd.rs`
- [x] T3: `dim_sufficiency_bound` theoretical bound check
- [x] T4: GOAT proof (7/7)
- [x] T5: Feature gate `sigmoid_margin` in Cargo.toml
- [x] T6: After GOAT pass → default-on promotion

### T1: Sigmoid Margin Loss

Port the paper's SigLIP-style loss to Rust:

```rust
/// SigLIP-style sigmoid margin loss: softplus(t · (score - b) · sign).
///
/// For each (query, doc) pair:
///   - positive pairs (A[i,j]=1): sign = +1, loss pushes score above bias
///   - negative pairs (A[i,j]=0): sign = -1, loss pushes score below bias
///
/// Global minimizers coincide with max-margin embeddings (Prop 7, arXiv 2605.23556).
///
/// # Arguments
/// - `scores`: [N × n] dot-product score matrix (row-major)
/// - `adjacency`: [N × n] binary adjacency (positive pairs = 1)
/// - `temperature`: learnable temperature (init 1.0)
/// - `bias`: learnable bias (init 0.0)
/// - `n_rows`, `n_cols`, `stride`: matrix dimensions
pub fn sigmoid_margin_loss(
    scores: &[f32],
    adjacency: &[f32],  // 0.0 or 1.0
    temperature: f32,
    bias: f32,
    n_rows: usize,
    n_cols: usize,
) -> f32
```

Implementation: `softplus(x) = log(1 + exp(x))`, numerically stable for large |x|.

### T2: Retrieval Margin Diagnostic

Port `compute_margin` from the paper's `sigmoid_embed.py`:

```rust
/// Compute retrieval margin: 0.5 × (min_pos_score - max_neg_score).
///
/// For each query embedding u_i with positive set P_i:
///   pos_min = min_{j ∈ P_i} dot(u_i, v_j)
///   neg_max = max_{j ∉ P_i} dot(u_i, v_j)
///   margin_i = 0.5 * (pos_min - neg_max)
///
/// Returns (min_pos_score, max_neg_score, margin) across all queries.
///
/// # Feature gate
/// `sigmoid_margin`
pub fn compute_retrieval_margin(
    queries: &[f32],     // [N × dim]
    documents: &[f32],   // [n × dim]
    neighborhoods: &[usize], // [N × k] positive pair indices
    dim: usize,
    n_queries: usize,
    n_docs: usize,
    k: usize,
) -> (f32, f32, f32)   // (pos_min, neg_max, margin)
```

### T3: Dimension Sufficiency Bound

```rust
/// Theoretical O(k log n) dimension sufficiency bound from arXiv 2605.23556.
///
/// Returns the minimum embedding dimension theoretically sufficient
/// for near-optimal retrieval margin, given query sparsity k and corpus size n.
///
/// Theorem 1.4: d = O(k · log n) is sufficient.
/// Theorem 1.5: d = O(k · log(n/k)) is also necessary → tight bound.
pub fn dim_sufficiency_bound(k: usize, n: usize) -> usize
```

This is a pure function for architecture validation (e.g., checking RtTurbo low_dim=16).

### T4: GOAT Proof

```
GOAT 157: Sigmoid Margin Loss
Feature gate: sigmoid_margin
Requires: maxsim

Proof 1: sigmoid_margin_loss matches paper's Python implementation
  - Generate random bipartite graph (n=20, k=2, d=8)
  - Compute loss with fixed t=1.0, b=0.0
  - Must match Python result within 1e-4

Proof 2: compute_retrieval_margin correctly identifies positive margin
  - Embeddings with known margin (orthogonal + noise)
  - Margin must match theoretical value within 5%

Proof 3: dim_sufficiency_bound returns O(k log n)
  - For k=2, n=100: bound ≤ 20 (constant factor ~1.5)
  - For k=4, n=1000: bound ≤ 60

Proof 4: Sigmoid loss converges to positive margin on synthetic data
  - Train embeddings with sigmoid loss on small bipartite graph
  - After 100 steps: margin > 0

Proof 5: Margin diagnostic validates MaxSim scoring quality
  - MaxSim score should correlate with retrieval margin
  - Higher margin → more confident MaxSim ranking

Proof 6: No performance regression on existing maxsim tests
  - All existing maxsim tests still pass

Proof 7: Feature gate isolation
  - Without sigmoid_margin feature: functions not visible
  - With feature: all proofs pass
```

### T5: Feature Gate

In `katgpt-core/Cargo.toml`:
```toml
sigmoid_margin = ["maxsim"]  # Sigmoid margin loss + retrieval margin diagnostic (Research 123, Plan 157)
```

### T6: Default-On Promotion

After GOAT 7/7 passes:
- Move to `default` features in Cargo.toml
- Update README feature flags table
- Add to GOAT production stack

## Module Structure

```
katgpt-core/src/simd.rs
  ├── sigmoid_margin_loss()     # T1
  ├── compute_retrieval_margin() # T2
  └── dim_sufficiency_bound()   # T3
```

All in `simd.rs` because they compose existing `simd_dot_f32` and follow the same feature-gated pattern as `maxsim_score`.

## Relationship to riir-ai

The loss function and diagnostic are open (katgpt-rs). The game-specific training loop upgrade (GoStyleEncoder) is private (riir-ai Plan 157).

## Cross-Reference

- Research 123: katgpt-rs `.research/123_TopK_Dimensionality_Barrier_Retrieval.md`
- Research 015: riir-ai `.research/015_Sigmoid_Margin_Loss_Game_Embeddings.md`
- MaxSim: Plan 080
- SDAR sigmoid: Plan 072/073
- EGA sigmoid: Plan 139
- Dirichlet Energy: Plan 149
