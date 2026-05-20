# Plan 080: MaxSim Late-Interaction Scoring

**Branch:** `develop/feature/080_maxsim_scoring`
**Depends on:** Plan 044 (PFlash), Research 45 (MaxSim)
**Research:** `.research/45_MaxSim_Memory_Efficient_Late_Interaction_Scoring.md`
**Source:** [erikkaum/maxsim](https://github.com/erikkaum/maxsim) — ColBERT/PyLate late-interaction kernel
**Goal:** Port MaxSim's memory-efficient `Σ_i max_j dot(q_i, d_j)` scoring to our CPU SIMD stack. Three targets: standalone `maxsim_score` primitive, PFlash block scoring upgrade (mean-K → maxsim), and `ScoreReduction::MaxSim` mode for TurboQuant/SpectralQuant fused kernels. All feature-gated under `maxsim`.

**Key Insight:** MaxSim's speedup (3-4× over naive) comes from **cache locality** — streaming over doc tokens with a running max, never materializing `[Lq × Ld]`. We already have `simd_dot_f32` and `simd_max_f32`. The distillation is composing them into a fused pattern. This is provably equivalent to the naive version (same math, less memory).

**Why CPU first:** The CPU `maxsim_score` is the foundation for PFlash block scoring and REST reranking. GPU WGSL kernel is deferred until CPU proves useful — the Metal kernel's simdgroup_matrix 2x/4x variants are GPU-specific and don't apply to our CPU path.

**Overlap with SpectralQuant (Research 39):** SpectralQuant already implements fused dequantize + scoring (`waterfill_dequant.wgsl`, `spectralquant_attention.wgsl`). We do NOT build a parallel MaxSim-on-compressed-KV pipeline. Instead, we add a `ScoreReduction` enum to the existing fused kernels. This keeps calibration, selective QJL, water-fill allocation, and variable-bit packing intact.

**Honest Scope:** We do NOT port the Metal `.metal`/`.mm` code, CUDA WMMA path, Python packaging, or backward pass. We port one algorithmic pattern (running-max dot scoring) to three locations in our existing codebase.

---

## Tasks

### Phase 1: Core Primitive — `maxsim_score`

- [ ] **T1: Add `maxsim_score` to `src/simd.rs`**
  - Signature: `pub fn maxsim_score(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32`
  - Computes `Σ_i max_j dot(q_i, d_j)` without allocating `[Lq × Ld]`
  - Uses running max per query token, calls `simd_dot_f32` for inner loop
  - FP32 accumulation regardless of input (matches Metal kernel design)
  - ~50 LOC, sits alongside existing `simd_dot_f32` and `simd_max_f32`
  ```rust
  /// Memory-efficient MaxSim scoring: `Σ_i max_j dot(q_i, d_j)`.
  ///
  /// Late-interaction relevance score (ColBERT/PyLate style) computed
  /// without materializing the [Lq × Ld] similarity matrix. Each query
  /// token's max similarity across all doc tokens is found via running
  /// max, then summed into the final score.
  ///
  /// Source: erikkaum/maxsim (Research 45).
  ///
  /// - `queries`:   [Lq, dim] row-major
  /// - `documents`: [Ld, dim] row-major
  /// - Returns: scalar score (fp32 accumulated)
  pub fn maxsim_score(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
      assert!(queries.len() >= lq * dim, "queries buffer too small: need {lq}*{dim}={}", lq*dim);
      assert!(documents.len() >= ld * dim, "documents buffer too small: need {ld}*{dim}={}", ld*dim);
      let mut score = 0.0f32;
      for i in 0..lq {
          let q_row = &queries[i * dim..(i + 1) * dim];
          let mut my_max = f32::NEG_INFINITY;
          for j in 0..ld {
              let d_row = &documents[j * dim..(j + 1) * dim];
              let dot = simd_dot_f32(q_row, d_row, dim);
              my_max = my_max.max(dot);
          }
          score += my_max;
      }
      score
  }
  ```

- [ ] **T2: Add `maxsim_score` tests to `src/simd.rs` mod tests**
  - `maxsim_matches_naive` — small random matrices, compare with materialized `[Lq × Ld]` then reduce
  - `maxsim_single_query_token` — Lq=1, should equal max over all doc dots
  - `maxsim_single_doc_token` — Ld=1, should equal sum over all query dots
  - `maxsym_symmetry_breaking` — verify result differs from `Σ_i dot(q_i, d_i)` (not diagonal)
  - `maxsim_empty_doc` — Ld=0 returns 0.0 (no tokens to match against)
  - `maxsim_large_dim_aligned` — dim=128, verify no alignment issues
  - **GOAT gate:** All tests pass, matches naive within 1e-6

- [ ] **T3: Add `maxsim_score_packed` to `src/simd.rs`**
  - Packed/ragged form: score N (query, doc) pairs with offset arrays
  - Signature:
    ```rust
    pub fn maxsim_score_packed(
        queries: &[f32],
        query_offsets: &[usize],    // [num_queries + 1]
        documents: &[f32],
        doc_offsets: &[usize],      // [num_docs + 1]
        pair_q_ids: &[usize],
        pair_d_ids: &[usize],
        dim: usize,
    ) -> Vec<f32>
    ```
  - Matches Metal kernel's canonical API (maxsim README "Packed (ragged segments)")
  - ~80 LOC
  - Tests: packed matches sequential individual `maxsim_score` calls

- [ ] **T4: Benchmark `maxsim_score` vs naive materialized baseline**
  - Add to `src/benchmark.rs` behind `maxsim` feature flag
  - Configs: dim ∈ {64, 128}, Lq ∈ {8, 32, 64}, Ld ∈ {32, 128, 256, 1024}
  - Metrics: wall time, peak allocation (approximated)
  - **GOAT gate:** ≥2× faster for Lq≥32, Ld≥128, dim=128

### Phase 2: PFlash Block MaxSim Scoring

- [ ] **T5: Add `ScoreReduction` enum to `src/speculative/types.rs`**
  ```rust
  /// Reduction mode for block/pair scoring.
  ///
  /// `SoftmaxSum` — standard attention: softmax-weighted sum (existing behavior).
  /// `MaxSim` — late-interaction: max per query token, then sum (MaxSim, Research 45).
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum ScoreReduction {
      SoftmaxSum,
      MaxSim,
  }

  impl Default for ScoreReduction {
      fn default() -> Self { Self::SoftmaxSum }
  }
  ```

- [ ] **T6: Add `block_score_maxsim` to `src/speculative/prefill.rs`**
  - New function: score block pairs using `maxsim_score` instead of `mean-K dot`
  - `block_score_maxsim(q_block: &[f32], k_block: &[f32], block_len_q: usize, block_len_k: usize, dim: usize) -> f32`
  - Wraps `maxsim_score` with block-level slicing
  - Feature-gated behind `maxsim`

- [ ] **T7: Wire `ScoreReduction` into `block_select`**
  - `block_select` currently calls `dot(Q_mean, K_mean)` for each block pair
  - When `ScoreReduction::MaxSim`: call `block_score_maxsim` instead
  - Config addition: `FlashPrefillConfig.score_reduction: ScoreReduction`
  - **GOAT gate:** ≥5% more "needle" blocks selected in synthetic spiky attention patterns vs mean-K

- [ ] **T8: Benchmark PFlash maxsim block scoring**
  - Synthetic: 1024 tokens, 32-token blocks, spike attention (1 needle per 20 tokens)
  - Metrics: needle recall, false positive rate, latency
  - Compare: mean-K (baseline) vs maxsim (T7)
  - **GOAT gate:** maxsim ≤3× latency overhead vs mean-K, ≥5% needle recall improvement

### Phase 3: TurboQuant/SpectralQuant `ScoreReduction::MaxSim`

- [ ] **T9: Add `ScoreReduction::MaxSim` to `src/turboquant/forward.rs`**
  - Extend `attention_turboquant` with a `score_reduction: ScoreReduction` parameter
  - When `MaxSim`: inner loop tracks `max(dot(q, dequant_k))` per query token instead of softmax-weighted value accumulation
  - Returns the maxsim score instead of attention output (different API shape — scoring-only, no V accumulation)
  - Well-commented: explain this is MaxSim (Research 45) adapted for compressed KV
  - Feature-gated behind both `turboquant` and `maxsim`
  - **GOAT gate:** matches uncompressed `maxsim_score` within 1e-3

- [ ] **T10: Add `ScoreReduction::MaxSim` to `src/spectralquant/forward.rs`**
  - Same pattern as T9 but for SpectralQuant's selective dequant path
  - Only score the `d_eff` semantic dimensions (tail is noise, skip for maxsim)
  - Well-commented: explain SpectralQuant's d_eff truncation means maxsim only sees semantic subspace
  - Feature-gated behind both `spectralquant` and `maxsim`
  - **GOAT gate:** matches CPU reference within 1e-3

- [ ] **T11: Add `ScoreReduction` to GPU SpectralQuant dispatch (riir-gpu)**
  - Extend `riir-ai/crates/riir-gpu/src/spectralquant/attention.rs`
  - Add `ScoreReduction` field to `SpectralQuantAttnParams`
  - WGSL kernel conditional: when `score_reduction == MAXSIM`, use `max` instead of `exp(score) * value` accumulation
  - Feature-gated behind `spectral_quant_gpu` and `maxsim`
  - **GOAT gate:** matches CPU reference within 1e-3, ≤5% latency overhead vs softmax-sum mode

### Phase 4: REST Reranking Integration

- [ ] **T12: Add `maxsim_score` to REST retrieval reranking**
  - In Plan 009's `merge_retrieved_branches`, use `maxsim_score` to score (query_hidden_state_seq, retrieved_token_embedding_seq) pairs
  - Replace cosine similarity with MaxSim late-interaction score
  - Feature-gated behind `maxsim`
  - **GOAT gate:** ≥2% better retrieval NDCG vs cosine similarity baseline

### Phase 5: Documentation

- [ ] **T13: Update README.md**
  - Add MaxSim section under "Key Features" or "Architecture"
  - Reference Research 45, Plan 080
  - List feature flag: `maxsim`

- [ ] **T14: Update `.docs/` if relevant**
  - Add to architecture doc if scoring section exists

---

## Feature Flag

```toml
[features]
maxsim = []  # MaxSim late-interaction scoring (Research 45, Plan 080)
```

Interacts with: `turboquant`, `spectralquant`, `spectral_quant_gpu`, `pflash`

---

## GOAT Proof Summary

All gates must pass before marking tasks complete:

| Task | Gate | Metric |
|------|------|--------|
| T2 | Correctness | `maxsim_score` matches naive within 1e-6 |
| T4 | Performance | ≥2× faster than naive for Lq≥32, Ld≥128 |
| T7 | Quality | ≥5% more needle blocks vs mean-K |
| T8 | Performance | maxsim block scoring ≤3× latency vs mean-K |
| T9 | Correctness | TQ maxsim matches uncompressed within 1e-3 |
| T10 | Correctness | SQ maxsim matches CPU reference within 1e-3 |
| T11 | Correctness + Perf | GPU matches CPU within 1e-3, ≤5% latency overhead |
| T12 | Quality | ≥2% better retrieval NDCG vs cosine |

**Failure mode:** If PFlash block maxsim (T7-T8) shows no improvement over mean-K, that application is abandoned. The CPU `maxsim_score` primitive (T1) and compressed KV mode (T9-T11) remain independently useful.

---

## Priority Assessment

| Task | Impact | Effort | Dependencies |
|------|--------|--------|-------------|
| T1 (CPU maxsim) | Medium | Low (~50 LOC) | None |
| T2 (Tests) | High | Low (~60 LOC) | T1 |
| T3 (Packed) | Low | Low (~80 LOC) | T1 |
| T4 (Bench) | Medium | Low (~40 LOC) | T1 |
| T5 (ScoreReduction enum) | Medium | Low (~15 LOC) | None |
| T6 (PFlash maxsim) | High | Low (~30 LOC) | T1, T5 |
| T7 (Wire block_select) | High | Medium (~50 LOC) | T6 |
| T8 (PFlash bench) | High | Medium (~50 LOC) | T7 |
| T9 (TQ maxsim) | Medium | Low (~30 LOC) | T1, T5 |
| T10 (SQ maxsim) | Medium | Low (~30 LOC) | T1, T5 |
| T11 (GPU SQ maxsim) | Low | Medium (~60 LOC) | T10, `riir-gpu` |
| T12 (REST reranking) | Low | Low (~30 LOC) | T1, Plan 009 |

**Recommended order:** T1 → T2 → T4 → T5 → T6 → T9 → T10 → T7 → T8 → T11 → T12

---

## Files Modified

| File | Changes |
|------|---------|
| `Cargo.toml` | Add `maxsim` feature flag |
| `src/simd.rs` | Add `maxsim_score`, `maxsim_score_packed`, tests |
| `src/speculative/types.rs` | Add `ScoreReduction` enum |
| `src/speculative/prefill.rs` | Add `block_score_maxsim`, wire into `block_select` |
| `src/turboquant/forward.rs` | Add `ScoreReduction::MaxSim` mode to `attention_turboquant` |
| `src/spectralquant/forward.rs` | Add `ScoreReduction::MaxSim` mode to spectral attention |
| `riir-ai/crates/riir-gpu/src/spectralquant/attention.rs` | Add GPU `ScoreReduction` dispatch |
| `src/benchmark.rs` | Add `maxsim_score` and PFlash maxsim benchmarks |
| `README.md` | Document MaxSim feature |

---

## References

- `.research/45_MaxSim_Memory_Efficient_Late_Interaction_Scoring.md` — research verdict
- `.raw/maxsim/maxsim_metal/maxsim.metal` — Metal kernel source (reference only)
- `.raw/maxsim/maxsim_metal/maxsim.mm` — Metal host-side dispatch (reference only)
- `.research/39_SpectralQuant_Calibrated_Eigenbasis_KV_Compression.md` — primary overlap
- `.research/44_ELF_Embedded_Language_Flows.md` — plan format reference
- `.plans/079_elf_embedded_language_flows_modelless.md` — plan format reference