# 115 — Tiled Online-Softmax Attention (CPU SIMD)

## Tasks

- [x] T1: Add `tiled_attention` feature gate to `microgpt-core/Cargo.toml`
- [x] T2: Create `microgpt-core/src/attention.rs` with tiled flash attention skeleton
- [x] T3: Implement `tiled_attention_forward()` — online softmax with SIMD tile iteration
- [x] T4: Implement `exp2` temperature scaling trick (avoid `exp()`, use `exp2()`)
- [x] T5: Add threshold heuristic — fall back to full materialization for small N
- [x] T6: Wire into `transformer.rs` forward pass behind feature gate — integrated into `forward_prefill` Phase B with GQA support, repacks Q/K/V into per-head contiguous layout, threshold=128, compiles w/wo feature, 79 transformer tests pass
- [x] T7: Add benchmark `tests/bench_tiled_attention.rs` — before/after throughput + memory
- [x] T8: Add GOAT proof `tests/test_tiled_attention_goat.rs` — cosine similarity > 0.999
- [x] T9: Add benchmark result to `.benchmarks/012_tiled_attention_simd.md` — cosine sim=1.00000 all configs, peak memory 4KB/head (vs N²×4B), debug throughput at parity
- [x] T10: Update `lib.rs` re-exports, update README, commit

## Overview

Distill ThunderKittens' (Research 077) online-softmax flash attention algorithm to our CPU SIMD pipeline. The key idea: instead of materializing the full `N×N` attention score matrix, process Q in SIMD-width row tiles and K/V in column tiles, maintaining running `max`, `norm`, and `O` accumulators across tiles.

**This is a pure algorithmic improvement** — no GPU hardware dependencies. The same tiling pattern ThunderKittens uses for H100 Tensor Cores applies to our NEON/AVX2 SIMD, just with smaller tile sizes (4–8 rows vs 64 rows).

**Branch:** `develop/feature/115_tiled_attention`
**Depends on:** Research 077 (ThunderKittens distillation)
**Target crate:** `microgpt-core`
**Feature gate:** `tiled_attention`
**Related:** Plan 106 (riir-ai CubeCL GPU rewrite — GPU tiled attention goes there, not here)

## Problem Statement

Current attention in `types.rs` (`softmax_scaled()` + `matmul()`):
1. Materializes full `N×N` score matrix per head — `O(N²)` memory
2. Applies softmax over entire score matrix — two full passes (max, then exp/sum)
3. Then multiplies score × V — another `O(N²)` matmul

For `N=2048, H=8, D=64`: score matrix = `8 × 2048 × 2048 × 4B = 128 MB` per forward pass. This dominates memory allocation at long contexts.

**TK's solution (and ours):** Process in tiles. Maintain running max/norm/output. Peak memory = `Br × Bc × 4B` per head. With `Br=8, Bc=128`: `8 × 128 × 4B = 4 KB` per head.

## Algorithm

From Research 077, adapted for CPU SIMD:

```text
Input: Q[B, H, N, D], K[B, H, N, D], V[B, H, N, D]
Output: O[B, H, N, D]
Tile sizes: Br = SIMD_WIDTH (4 NEON, 8 AVX2), Bc = 64..256 (tunable)

For each (batch, head):
  Initialize: O[N, D] = 0, max[N] = -∞, norm[N] = 0

  For q_tile in 0..ceil(N / Br):           // outer: query rows
    q_rows = Q[q_tile*Br .. (q_tile+1)*Br]  // Br × D

    For k_tile in 0..ceil(N / Bc):          // inner: key/value columns
      k_cols = K[k_tile*Bc .. (k_tile*Bc+Bc)] // Bc × D
      v_cols = V[k_tile*Bc .. (k_tile*Bc+Bc)] // Bc × D

      // 1. Score tile: Br × Bc
      S = q_rows @ k_cols.T                   // SIMD matmul

      // 2. Update running max
      max_new = max(max_old, rowmax(S))       // SIMD max reduction

      // 3. Exp with correction (exp2 trick)
      correction = exp2((max_old - max_new) * scale)
      P̃ = exp2(S * scale - max_new * scale)

      // 4. Update running norm and output
      norm = correction * norm + rowsum(P̃)
      O[q_rows] = correction * O[q_rows] + P̃ @ v_cols  // fused

    // After all k_tiles for this q_tile:
    O[q_rows] = O[q_rows] / norm[q_rows]     // final normalize
```

### exp2 Trick (From TK)

```rust
// TK precomputes: temperature_scale = rsqrt(D) * log2(e)
// This converts softmax exp(x/√D) → exp2(x * scale)
// exp2() is faster than exp() on most hardware (IEEE 754 bit manipulation)
const LOG2_E: f32 = 1.44269504089f32;
let temperature_scale = (1.0f32 / (head_dim as f32).sqrt()) * LOG2_E;
```

### Fallback Threshold

For small N, tiling overhead (loop setup, register save/restore) exceeds the benefit. Threshold heuristic:

```rust
/// Use tiled attention when score matrix exceeds L1 cache.
/// L1 ≈ 32 KB (typical). Score = N * N * 4B.
/// Threshold: N > sqrt(32K / 4) ≈ 90. Round up to 128 for alignment.
const TILED_ATTENTION_THRESHOLD: usize = 128;
```

Below `N=128`: use current `softmax_scaled()` + `matmul()` (already correct, fast for small N).

## Architecture

### File Structure

```text
microgpt-core/src/
├── attention.rs          # NEW: tiled flash attention (behind feature gate)
├── lib.rs                # MODIFIED: add mod attention + re-export
├── simd.rs               # UNCHANGED: existing SIMD kernels
└── types.rs              # UNCHANGED: softmax_scaled, matmul still used for fallback
```

### API

```rust
// microgpt-core/src/attention.rs

/// Tiled online-softmax flash attention for CPU SIMD.
///
/// Processes Q in SIMD-width row tiles, K/V in column tiles.
/// Avoids materializing full N×N score matrix.
/// Falls back to full materialization for small N.
///
/// # Arguments
/// * `q` - Query tensor [seq_len × head_dim]
/// * `k` - Key tensor [seq_len × head_dim]
/// * `v` - Value tensor [seq_len × head_dim]
/// * `output` - Output tensor [seq_len × head_dim] (pre-allocated)
/// * `head_dim` - Dimension per attention head
/// * `scale` - Softmax temperature (typically 1/√head_dim)
#[cfg(feature = "tiled_attention")]
pub fn tiled_attention_forward(
    q: &[f32], k: &[f32], v: &[f32],
    output: &mut [f32],
    seq_len: usize, head_dim: usize,
    scale: f32,
)

/// Tiled attention for multi-head batched input.
/// Calls tiled_attention_forward per (batch, head) with Rayon parallelism.
#[cfg(feature = "tiled_attention")]
pub fn tiled_attention_batched(
    q: &[f32], k: &[f32], v: &[f32],
    output: &mut [f32],
    batch: usize, heads: usize,
    seq_len: usize, head_dim: usize,
)
```

### Integration Point

```rust
// microgpt-rs/src/transformer.rs — in the attention forward section

#[cfg(feature = "tiled_attention")]
{
    if seq_len >= TILED_ATTENTION_THRESHOLD {
        microgpt_core::tiled_attention_batched(
            q_buf, k_buf, v_buf, attn_out_buf,
            batch, heads, seq_len, head_dim,
        );
    } else {
        // Fallback to current path for small N
        microgpt_core::softmax_scaled(&mut scores, head_dim);
        microgpt_core::matmul(&scores, v_transposed, &mut attn_out, ...);
    }
}
#[cfg(not(feature = "tiled_attention"))]
{
    // Current path unchanged
    microgpt_core::softmax_scaled(&mut scores, head_dim);
    microgpt_core::matmul(&scores, v_transposed, &mut attn_out, ...);
}
```

## Feature Gate

```toml
# microgpt-core/Cargo.toml

[features]
default = []
coda_fusion = []
tiled_attention = []  # Plan 115: Tiled online-softmax flash attention
```

No new dependencies. Uses existing `simd.rs` kernels and `rayon` for batch parallelism.

## Benchmarks

### Setup
- Config: `Config::micro()` (8 heads, 64 dim)
- Sequence lengths: 64, 128, 256, 512, 1024, 2048, 4096
- Metrics: throughput (tok/s), peak memory (bytes allocated)
- Seeds: 42, 43, 44 (3 seeds, report median)
- Build: `--release`

### What to Measure

| Metric | Current Path | Tiled Path | Expected |
|:---|:---|:---|:---|
| Attention time @ N=512 | baseline ms | tiled ms | ~same (overhead ≈ savings) |
| Attention time @ N=2048 | baseline ms | tiled ms | **15-30% faster** (less allocation) |
| Attention time @ N=4096 | baseline ms | tiled ms | **30-50% faster** (O(N²) → O(N)) |
| Peak memory @ N=2048 | N² × 4B × H | Br × Bc × 4B × H | **32× less** per head |
| Cosine similarity | 1.0 (reference) | vs reference | **> 0.999** |

### Output
`.benchmarks/012_tiled_attention_simd.md`

## GOAT Proof

**Goal:** Tiled attention output matches full-materialization output to > 0.999 cosine similarity.

```rust
// tests/test_tiled_attention_goat.rs

#[test]
fn tiled_attention_cosine_similarity_goat() {
    let configs = [
        (8, 64, 64),    // small: fallback path
        (8, 64, 128),   // at threshold
        (8, 64, 256),   // tiled
        (8, 64, 512),   // tiled
        (8, 64, 1024),  // tiled, larger
    ];

    for &(heads, dim, seq) in &configs {
        // Generate random Q, K, V with seed 42
        // Compute reference: full softmax_scaled + matmul
        // Compute tiled: tiled_attention_forward
        // Assert cosine_similarity(reference, tiled) > 0.999
    }
}
```

**Failure criteria:** If cosine similarity < 0.999 for any config, investigate numerics (likely exp2 precision or correction factor bug). Do NOT ship with < 0.999 similarity.

## Implementation Notes

### SIMD Tile Sizes

```rust
/// Row tile size: one SIMD vector worth of query rows.
/// NEON = 4 f32 per register, AVX2 = 8 f32 per register.
/// We use 8 as default (works for both; NEON processes in 2 sub-tiles).
const BR: usize = 8;

/// Column tile size: tuned for L1 cache.
/// Each K tile = Bc × head_dim × 4B. For head_dim=64, Bc=128: 32 KB (fits L1).
const BC: usize = 128;
```

### Why Not Smaller/Larger Bc?

- `Bc < 64`: Too much loop overhead per K/V tile. Not enough work per tile.
- `Bc > 256`: Exceeds L1 cache for typical head_dim. Causes cache thrashing.
- `Bc = 128`: Good balance for head_dim ∈ [32, 128]. K tile = 128 × 64 × 4B = 32 KB.

### Online Softmax Correction Factor

The critical numerical detail from TK: when `max_new > max_old`, ALL previous accumulations must be rescaled:

```rust
// correction = exp((max_old - max_new) * scale)
// If max_new > max_old: correction < 1 (previous contributions discounted)
// If max_new == max_old: correction = 1 (no change)
// This MUST be applied to BOTH norm and O before adding current tile
let correction = (max_old * scale - max_new * scale).exp2();
norm *= correction;
// O rows: multiply each element by correction (SIMD broadcast)
```

### Edge Cases

1. **Sequence length not divisible by tile size:** Right-fill scores with `-inf` (TK's `warp::right_fill`). Our SIMD matmul naturally produces full tiles; we mask out-of-bounds elements before softmax.

2. **Causal masking (future):** This plan implements non-causal (bidirectional) attention first. Causal masking adds an upper-triangle `-inf` mask to the score tile before softmax. This is a simple extension — add a `causal: bool` parameter, apply mask in the score tile.

3. **GQA (grouped query attention):** The batched function maps `head → kv_group = head * n_kv_head / n_head` and shares K/V across query heads in the same group. This matches our existing `kv_dim()` logic.

## What This Plan Does NOT Do

1. **GPU kernel** — That's Plan 106 (CubeCL rewrite). This plan is CPU SIMD only.
2. **Sparse attention** — That's Plan 104 (DashAttention). This is dense tiled attention.
3. **KV cache compression** — That's TurboQuant/SpectralQuant/OCTOPUS. This plan assumes uncompressed KV.
4. **Causal masking** — Deferred to a follow-up. Non-causal first for simplicity and correctness.
5. **FP16 attention** — We compute in f32 throughout (our SIMD kernels are f32-native).

## Risks

| Risk | Mitigation |
|:---|:---|
| Numerical drift from online softmax | GOAT test with cosine similarity > 0.999 |
| No speedup for small N (loop overhead) | Fallback threshold at N=128 |
| exp2 trick precision differs from exp() | Compare exp2 vs exp numerics in test |
| SIMD tile alignment issues | Pad Q/K/V rows to SIMD width, mask excess |

## Timeline

- T1–T4: Core implementation (~2-3 hours)
- T5: Fallback heuristic (~30 min)
- T6: Integration (~1 hour)
- T7–T8: Benchmark + GOAT proof (~1 hour)
- T9–T10: Documentation + commit (~30 min)

**Total estimate:** ~6 hours