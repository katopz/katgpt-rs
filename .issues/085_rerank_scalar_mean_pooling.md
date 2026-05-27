# Issue 085: cosine_rerank_score_into uses scalar loops for mean-pooling

## Severity: Medium (under maxsim feature — retrieval reranking hot path)

## Location
- `src/rerank.rs` — `cosine_rerank_score_into()` L180-228

## Problem

The mean-pooling loops in `cosine_rerank_score_into` are scalar:

```rust
// L195-199: scalar accumulation for mean-pool
for t in 0..lq {
    let offset = t * dim;
    for d in 0..dim {
        q_mean[d] += query[offset + d];
    }
}

// L202-204: scalar scale
let inv_lq = 1.0 / lq as f32;
for v in q_mean[..dim].iter_mut() {
    *v *= inv_lq;
}
```

Same scalar pattern for `d_mean` (L207-217).

This is called per-document in the reranking loop. For large `dim` (e.g., 768 or 1024), the scalar loops leave significant SIMD throughput on the table.

Additionally, `cosine_score()` (L136-166) recomputes `q_norm` inside the inner `j` loop even though `q_row` doesn't change:

```rust
for i in 0..lq {
    let q_row = &queries[i * dim..(i + 1) * dim];
    let q_norm = simd_dot_f32(q_row, q_row, dim).sqrt(); // computed once per i ✓
    for j in 0..ld {
        let d_row = &documents[j * dim..(j + 1) * dim];
        let d_norm = simd_dot_f32(d_row, d_row, dim).sqrt(); // recomputed for every (i,j) ✓
        let dot = simd_dot_f32(q_row, d_row, dim);
        total += dot / (q_norm * d_norm);
    }
}
```

The `q_norm` is correctly hoisted, but `d_norm` for the same `d_row` is recomputed for every query token `i` (N×M computations when only M unique values exist).

## Fix

1. **Mean-pooling**: use `simd_add_inplace` for accumulation and `simd_scale_inplace` for the inverse-scale
2. **cosine_score d_norm**: pre-compute all `d_norm` values once (M calls) and index into them, instead of recomputing per query token (N×M → M calls)

```rust
// Pre-compute d_norms once
let d_norms: Vec<f32> = (0..ld)
    .map(|j| simd_dot_f32(&documents[j*dim..(j+1)*dim], &documents[j*dim..(j+1)*dim], dim).sqrt())
    .collect();

for i in 0..lq {
    let q_row = &queries[i * dim..(i + 1) * dim];
    let q_norm = simd_dot_f32(q_row, q_row, dim).sqrt();
    for j in 0..ld {
        let d_row = &documents[j * dim..(j + 1) * dim];
        let dot = simd_dot_f32(q_row, d_row, dim);
        total += dot / (q_norm * d_norms[j]);
    }
}
```

## Expected Impact
- Mean-pooling: ~3-4× faster for dim ≥ 64
- cosine_score d_norm: reduces N×M sqrt+dot computations to M + N×M dots (saves N×M sqrt operations)

## Optimization Reference
- optimization.md → "SIMD / Auto-vectorization" — use SIMD kernels for bulk operations
- optimization.md → "Don't: Recompute unchanged values" — compute once per position
