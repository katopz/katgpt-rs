# Research Verdict 45: MaxSim — Memory-Efficient Late-Interaction Scoring

**Source:** [erikkaum/maxsim](https://github.com/erikkaum/maxsim) — Exact MaxSim kernel for ColBERT/PyLate late-interaction retrieval
**Author:** Erik Kaum
**Date:** 2026-05-20
**Status:** 🟢 ADOPT — Core `maxsim_score` primitive proven (CPU SIMD 7.46×, GPU 41–74× at scale). PFlash block scoring proven (371% better). Compressed KV path proven (SQ exact match). GPU dispatch proven (Plan 085, `maxsim` feature in `riir-gpu`).

---

## 1. Core Technique

MaxSim computes **late-interaction relevance scores** without materializing the full similarity matrix:

```
score(q, d) = Σ_i max_j dot(q_i, d_j)
```

For each query token, find the **max** similarity across all document tokens, then **sum** those per-query-token maxima. The key optimization: **never allocate `[Lq × Ld]`**. Each thread keeps a running `max(dot(q_i, d_j))` over its slice of doc tokens, tree-reduces in shared memory, then atomic-adds into the per-pair score.

This is the standard scoring function for ColBERT/PyLate late-interaction retrieval and reranking.

## 2. Key Findings

### 2.1 The Algorithm is Three Primitives Composed (maxsim.metal)

The entire kernel decomposes into operations we already have in `src/simd.rs`:

| MaxSim Operation | Our Existing Primitive |
|---|---|
| Per-token dot product | `simd_dot_f32(a, b, len)` |
| Running max across doc tokens | `simd_max_f32(x)` (but we need inline max, not over slice) |
| Outer product Q×Kᵀ | `simd_outer_product_acc(...)` |
| Row-wise matmul | `simd_matmul_rows(...)` |

**Gap:** We have `dot` and `max` separately but no **fused max-dot** that computes `Σ_i max_j dot(q_i, d_j)` without materializing intermediates. This is a ~50 LOC composition, not a new subsystem.

### 2.2 Memory Efficiency is the Real Win (README Benchmarks)

The naive approach materializes `[Lq × Ld]` then reduces. MaxSim skips the allocation:

| Workload (M3 Pro, fp16, dim=128) | Kernel | Naive | Speedup |
|---|---|---|---|
| SmallRerank (B=32, C=10, Lq=32, Ld=180) | 0.45ms | 1.44ms | **3.18×** |
| HeavyRerank (B=32, C=100, Lq=32, Ld=256) | 4.34ms | 16.63ms | **3.83×** |
| LongDocStress (B=8, C=16, Lq=64, Ld=1024) | 1.69ms | 3.70ms | **2.19×** |

The speedup comes from **cache locality** (streaming over doc tokens) not algorithmic improvement. The same `O(Lq × Ld × dim)` work happens, just without the intermediate allocation.

### 2.3 Metal simdgroup_matrix Provides 2x/4x Variants (maxsim.metal L200-400)

The kernel has 2x and 4x MMA variants that share the A-fragment (query) across multiple B-slices (documents):

- **2x:** Compute `Q @ D0ᵀ` and `Q @ D1ᵀ` in lockstep, sharing A-fragment load. Halves A-loads and barrier count.
- **4x:** One A-fragment shared across four consecutive 8-row slices of D. Quarters A-loads.

These fire when `dim % 8 == 0` (Metal) or `dim % 16 == 0` (CUDA). All typical embedding sizes (64, 96, 128) qualify.

**Distillation:** The sharing pattern is the same principle as our `simd_sparse_dot_f32` gather-scatter — reuse loaded data across multiple accumulations.

### 2.4 PFlash Mean-K Scoring is Suboptimal for Block Importance

Current PFlash block scoring: `block_score[i][j] = dot(Q_block_mean[i], K_block_mean[j])`.

Mean-pooling dilutes strong signals. If one query token has high affinity with one key token in a block, the mean buries it. MaxSim preserves the strongest signal:

```
Current: block_score = dot(mean(Q_block), mean(K_block))
MaxSim:  block_score = Σ_i max_j dot(q_i, k_j)  // over Q-block × K-block
```

This is the most impactful application for our architecture.

### 2.5 Compressed MaxSim is Redundant with SpectralQuant (Research 39)

SpectralQuant already implements fused dequantize + scoring:

| Proposed (MaxSim + compressed KV) | Already Exists (SpectralQuant) |
|---|---|
| Fused dequantize + dot per position | `waterfill_dequant.wgsl` |
| Skip full dequantized matrix allocation | Selective dequant on d_eff subspace |
| Per-position lazy decode | `spectralquant_attention.wgsl` |

The **only** difference is the reduction: SpectralQuant uses `softmax(Q·K)·V` (sum), MaxSim uses `max_j` per query token. Adding a `ScoreReduction` enum (`SoftmaxSum` | `MaxSim`) to the existing kernel is a ~20 LOC change, not a new pipeline.

Similarly for TurboQuant: `attention_turboquant` in `src/turboquant/forward.rs` already fuses dequantize + Q·K scoring. Adding a max-reduction mode is a minor extension.

### 2.6 REST Retrieval Reranking is the Natural Use Case (Plan 009)

Plan 009's `RestClient::retrieve()` queries anyrag with hidden-state embeddings and returns candidate sequences with similarity scores. When reranking multiple retrieved sequences, MaxSim is the correct scoring function:

- Query = current hidden state sequence `[Lq, dim]`
- Document = retrieved token sequence embeddings `[Ld, dim]`
- Score = `Σ_i max_j dot(q_i, d_j)` — exactly MaxSim

This applies MaxSim in its **original design context** (retrieval reranking) without any architectural changes.

---

## 3. What We Can Distill (Honest Assessment)

### ✅ Distillable Without Architecture Changes

| Technique | Target Module | Path | Risk |
|---|---|---|---|
| Fused `maxsim_score()` | `src/simd.rs` | CPU SIMD | Low — composing existing primitives |
| PFlash maxsim block scoring | `src/speculative/prefill.rs` | CPU | Low — replace mean-K with maxsim |
| REST reranking with maxsim | Plan 009 bridge | CPU | Low — scoring function change |
| `ScoreReduction::MaxSim` mode | `src/turboquant/forward.rs` | CPU | Low — minor extension of existing kernel |
| `ScoreReduction::MaxSim` mode | `riir-gpu` SpectralQuant | GPU | Low — minor extension of existing WGSL |
| Size-gated MaxSim dispatch | `riir-gpu` `maxsim` feature | GPU | ✅ Done — Plan 085, `maxsim_score.wgsl` + `MaxSimScorer` (threshold=256) |
| Fused SQ + MaxSim kernel | `riir-gpu` dual feature gate | GPU | ✅ Done — Plan 085 T5, `spectralquant_maxsim.wgsl`, dequant + MaxSim in one pass |

### ⚠️ Distillable With Moderate Changes

| Technique | Target Module | Path | Risk |
|---|---|---|---|
| Packed/ragged batch maxsim | `src/simd.rs` | CPU | Low — new API surface |
| WGSL `maxsim_block_score` kernel | `riir-gpu` PFlash pipeline | GPU | Medium — new shader |

### ❌ Not Distillable (Architecture Incompatibility)

| Technique | Why Not |
|---|---|
| Full Metal `.metal` / `.mm` code | We target wgpu (WGSL), not platform-native Metal. Our GPU path is `riir-gpu`. |
| CUDA WMMA path | We don't have CUDA; wgpu compiles to Metal/Vulkan/DX12. |
| Python HuggingFace `kernels` packaging | We're Rust-native, no Python dependency. |
| Backward pass | MaxSim explicitly doesn't implement it. Training uses different kernels. |
| simdgroup_matrix 2x/4x sharing | GPU-specific optimization; our CPU path benefits from cache locality instead. |

---

## 4. Modelless Distillation Targets

### 4.1 Fused `maxsim_score` — CPU SIMD (Primary Target)

**Paper basis:** MaxSim kernel core algorithm (maxsim.metal L24-60).

**Hypothesis:** Composing `simd_dot_f32` + inline running max provides memory-efficient late-interaction scoring for PFlash block importance and REST reranking, without allocating `[Lq × Ld]` intermediates.

**GOAT proof required:**
1. Correctness: `maxsim_score` matches naive materialized result within 1e-6
2. Memory: peak allocation stays at `O(dim)` not `O(Lq × Ld)`
3. Performance: ≥2× faster than materialized baseline for Lq≥32, Ld≥128
4. Integration: PFlash block scoring with maxsim ≥2% perplexity improvement over mean-K

**Risk assessment:** This is pure composition of existing tested primitives. The algorithm is provably equivalent to the naive version (same mathematical result). Failure mode is only performance, not correctness.

### 4.2 PFlash Block MaxSim Scoring (Secondary Target)

**Paper basis:** MaxSim's advantage over mean-pooling for relevance scoring.

**Hypothesis:** Replacing `dot(Q_mean, K_mean)` with `Σ_i max_j dot(q_i, k_j)` in PFlash block scoring produces better block importance rankings, because max preserves the strongest per-token signal while mean dilutes it.

**GOAT proof required:**
1. Block selection: ≥5% more "needle" blocks selected in synthetic attention patterns
2. Prefill quality: compressed prompt with maxsim block scoring ≥2% better downstream task score vs mean-K
3. Latency: maxsim block scoring ≤3× slower than mean-K (acceptable since block scoring is not the hot path)

**Connection to existing work:** PFlash (Plan 044) already has `FlashPrefillConfig` with pluggable scoring. MaxSim is a drop-in replacement for `block_select`'s importance function.

### 4.3 Embedding ScreeningPruner (Exploratory)

**Paper basis:** MaxSim's token-level similarity scoring.

**Hypothesis:** A `ScreeningPruner` that uses token embedding dot products as relevance scores can provide graded domain-fit scoring, replacing binary constraint checks.

**Status:** Conceptual only. Requires defining where embeddings come from at prune-time (they're normally computed during forward pass, not available during DDTree expansion). May be incompatible with the DDTree's lazy evaluation model.

---

## 5. Model-Based Distillation Targets

### 5.1 TurboQuant/SpectralQuant ScoreReduction Mode (Primary Target)

**Paper basis:** MaxSim's max-reduction vs standard attention's softmax-sum reduction.

**Hypothesis:** Adding a `ScoreReduction::MaxSim` variant to the existing fused dequantize+scoring kernels (both CPU `attention_turboquant` and GPU `spectralquant_attention`) enables late-interaction scoring directly on compressed KV, without a new pipeline.

**GOAT proof required:**
1. CPU: `attention_turboquant` with maxsim mode matches uncompressed `maxsim_score` within 1e-3
2. GPU: `spectralquant_attention` with maxsim mode matches CPU reference within 1e-3
3. Latency: maxsim mode ≤5% slower than standard softmax-sum mode (same fused kernel, different reduction)

**Risk assessment:** Very low. The fused dequantize+scoring path already exists. Adding `max` reduction alongside `softmax-sum` is a branch in the inner loop, not a new algorithm.

### 5.2 WGSL MaxSim Block Score Kernel (Future Target)

**Paper basis:** MaxSim Metal kernel architecture.

**Hypothesis:** A dedicated WGSL compute shader for PFlash block scoring can accelerate the maxsim per-block computation on GPU, similar to the Metal kernel's 2-3× speedup over naive.

**Status:** ✅ Done — Plan 085. `maxsim_score.wgsl` + `MaxSimScorer` with size-gated CPU/GPU dispatch (threshold=256). GPU 41–74× faster for large batches. Fused SQ+MaxSim kernel (`spectralquant_maxsim.wgsl`) also complete.

---

## 6. Cross-Reference with Existing Research

| Existing Research | MaxSim Connection | Compatibility |
|---|---|---|
| Research 39 (SpectralQuant) | **Primary overlap**: fused dequant+scoring already implemented. MaxSim only adds `max` reduction mode. | ✅ Compatible (minor extension) |
| Research 20 (TurboQuant) | MaxSim on uncompressed TQ dequant; subsumed by SpectralQuant's selective path | ✅ Compatible (also extendable) |
| Research 22 (Lighthouse Attention) | Both optimize attention scoring; MaxSim is orthogonal (batch dimension) | ✅ Orthogonal |
| Research 44 (PFlash) | **Primary application**: replace mean-K block scoring with maxsim per-block | ✅ Direct upgrade |
| Research 42 (SP-KV) | MaxSim can score selectively — only retained KV positions | ✅ Compatible |
| Research 00-01 (RAG/REST) | MaxSim is the natural scoring function for retrieved sequences | ✅ Natural fit |
| Research 37 (REAP) | Model-based/modelless duality — MaxSim enables modelless relevance scoring | ✅ Consistent |

**Key insight:** MaxSim's core idea — **reduce-while-scoring without materializing the full matrix** — is exactly what our attention kernels already do in the **temporal** dimension (dot Q against all K positions, never store all pairwise scores). MaxSim applies the same principle in the **batch** dimension. The distillation is about **composing** existing SIMD primitives (`simd_dot_f32`, `simd_max_f32`) into the right fused pattern.

---

## 7. Verdict Summary

**🟢 ADOPT (proven in our stack or zero-risk):**
- `maxsim_score()` CPU SIMD — composition of existing tested primitives, provably correct
- `ScoreReduction::MaxSim` for TurboQuant/SpectralQuant — one-parameter extension of existing kernels
- PFlash block maxsim scoring — **371% better** needle separation vs mean-K (T7 GOAT passed)
- REST reranking with MaxSim — `src/rerank.rs` module, `RerankMethod` enum, NDCG@10 proven ≥2% better than cosine (T12 GOAT passed, Benchmark 014)
- GPU MaxSim dispatch — `maxsim_score.wgsl` + `MaxSimScorer` (Plan 085), GPU **41–74× faster** for large batches, threshold=256 (T11 GOAT passed)
- Fused SQ + MaxSim kernel — `spectralquant_maxsim.wgsl`, dequant + MaxSim in one GPU pass (Plan 085 T5)

**🟡 INVESTIGATE (distillable, needs demand/validation):**
- Packed/ragged batch maxsim — useful API for multi-pair scoring, needs demand
- MaxSim reranking integration with live anyrag `/search/vector` endpoint — module proven, needs deployment

**🔴 REJECT (incompatible with our architecture):**
- Full Metal `.metal`/`.mm` code — we use wgpu (WGSL), not platform-native Metal
- CUDA WMMA path — we don't have CUDA
- Python HuggingFace `kernels` packaging — we're Rust-native
- Backward pass — MaxSim explicitly doesn't implement it
- Argmax-position output — not critical for our scoring pipeline

---

## 8. Honest Caveats

1. **MaxSim is O(Lq × Ld × dim) either way.** The speedup comes from cache locality (streaming over doc tokens) not algorithmic improvement. On CPU with large caches, the benefit may be smaller than on GPU.

2. **The 3-4× speedup is GPU-measured.** Our CPU SIMD path may see less relative gain because scalar dot product is already cache-friendly. The real win is **memory avoidance** (no `[Lq × Ld]` allocation), not raw compute speed.

3. **PFlash maxsim vs mean-K is a hypothesis, not proven.** Mean-pooling is theoretically better for capturing aggregate block similarity. MaxSim is theoretically better for capturing the strongest per-token signal. Which wins depends on the actual attention pattern distribution in our models.

4. **ColBERT/PyLate context is retrieval, not inference.** MaxSim was designed for scoring (query, document) pairs in search pipelines. Applying it to PFlash block importance or attention scoring is an adaptation, not a direct port.

5. **The compressed KV overlap with SpectralQuant is nearly complete.** We should not build a parallel "MaxSim on compressed KV" pipeline. The right approach is a `ScoreReduction` enum on the existing SpectralQuant/TurboQuant fused kernels.

6. **dim alignment matters for SIMD paths.** The Metal kernel's fast path requires `dim % 8 == 0`. Our SIMD paths have no such constraint but benefit from aligned lengths. Typical embedding sizes (64, 128) are always aligned.

7. **MaxSim amplifies quantization error 12–14×** (benchmark 013, Section 7 4-way matrix). The `max` operation selects the highest dot product per query token; if quantization noise shifts which doc token "wins", the error compounds far beyond per-vector reconstruction error. TQ: 2.8% cosine error → 40.5% MaxSim error (14.2×). SQ: 1.6% cosine error → 18.9% MaxSim error (12.2×). SQ's lower base error means its amplified MaxSim error is still 2.1× better than TQ, but the amplification means **higher bit budgets are more important for MaxSim than for cosine-based scoring**.

---

## 9. GOAT Proof Checklist

### Modelless Proposals (tested in microgpt-rs)
- [x] `maxsim_score()`: matches naive materialized result within 1e-6, **7.46× faster** (48.3µs vs 360.0µs, Lq=32, Ld=256, dim=128, release build) — Plan 080 T2/T4
- [x] PFlash block maxsim: **371% more** needle blocks selected (4.71× better separation: 20× vs 4.25× for mean-K) — Plan 080 T7
- [x] REST maxsim reranking: ≥2% better retrieval NDCG vs cosine similarity — `src/rerank.rs` module (`RerankMethod` enum, `ndcg_at`, `rerank`), `bench_maxsim_rerank` test, Benchmark 014 — Plan 080 T12

### Model-Based Proposals (tested in microgpt-rs CPU)
- [x] TurboQuant `ScoreReduction::MaxSim`: matches uncompressed maxsim within 0.95% at 4-bit; **40.54% error at 3-bit** — Plan 080 T9
- [x] SpectralQuant `ScoreReduction::MaxSim`: streaming vs dequantized **exact match (0.00%)**; **18.90% error at 3-bit** (2.1× less than TQ) — Plan 080 T10

### GPU Dispatch (Plan 085 — `riir-gpu` `maxsim` feature)
- [x] `maxsim_score.wgsl` + `MaxSimScorer`: size-gated CPU/GPU dispatch, threshold=256 — GPU **41–74× faster** for work_size ≥ 50K, crossover at work_size ≈ 300–800, correctness within 1e-3 — Plan 085 T1-T3
- [x] Fused `spectralquant_maxsim.wgsl`: dequantize K from compressed bitstream + MaxSim scoring in one GPU pass — Plan 085 T5

### 3-Way Matrix: TQ/SQ/OCT × Cosine/MaxSim (3-bit, calibrated)

TQ/SQ: kv_dim=16, 3-bit, 16 doc positions, 4 query tokens. OCT: d=128, 3-bit, 512 keys, 4 query tokens.

```
┌──────────────────────────────────┬──────────────┬──────────────┬──────────────┐
│ Metric                            │ TurboQuant   │ SpectralQuant│ OCTOPUS      │
├ ─ ─ Scoring Quality ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│ Key cosine (reconstruction)       │ 0.9692       │ 0.9812       │ 0.9870       │
│ MaxSim error (vs uncompressed)    │  27.15%      │   2.26%      │   1.06%      │
│ MSE (reconstruction)              │  0.0886      │  0.0379      │  0.0263      │
│ Compression ratio                 │ 5.3×         │ 9.7×         │ 8.8×         │
│ Calibration                       │ 0 samples    │ 256 samples  │ 0 samples    │
├ ─ ─ Verdict ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│                                   │ legacy       │ default-on   │ default-on   │
└──────────────────────────────────┴──────────────┴──────────────┴──────────────┘
```

**Key finding: MaxSim amplifies quantization error 12–14×** (MaxSim error ÷ cosine error).
TQ: 40.5% MaxSim error from 2.8% cosine error = 14.2× amplification.
SQ: 18.9% MaxSim error from 1.6% cosine error = 12.2× amplification.
OCT: 1.06% MaxSim error from 1.3% cosine error = 0.8× amplification — **OCTOPUS breaks the amplification pattern** at scale.

→ **MaxSim + OCTOPUS is the optimal combination** for late-interaction scoring on compressed KV — lowest MaxSim error, zero calibration, Pareto-better compression at 3-bit. OCTOPUS added to GOAT Bench 022 Section 9.

Cross-validated by `bench_spectralquant_cosine_vs_turboquant` test: SQ cosine 0.9917 > TQ 0.9692, SQ compression 9.1× > TQ 5.3×, both at 3-bit.

**Failure mode:** PFlash block maxsim shows strong improvement (371% better), not dead. REST reranking blocked on Plan 009. CPU `maxsim_score` primitive proven useful regardless. Detailed results in `.benchmarks/013_turboquant_vs_spectralquant_maxsim.md`.

---

## References

- Source: <https://github.com/erikkaum/maxsim>
- Kernel repo: <https://huggingface.co/kernels/erikkaum/maxsim>
- ColBERT paper: "Late Interaction over BERT" (Khattab & Zaharia, 2020)
- PyLate: "PyLate: Flexible Retrieval with Late Interaction" (NUBES)
- `.raw/maxsim/maxsim_metal/maxsim.metal` — Metal kernel source
- `.raw/maxsim/maxsim_metal/maxsim.mm` — Metal host-side dispatch

## Cross-References

- **Plan 080** (MaxSim Late Interaction Scoring) — implementation plan for this research
- **Research 39** (SpectralQuant) — primary overlap on compressed KV scoring
- **Research 44** (PFlash) — primary application target for block maxsim
- **Research 37** (REAP) — model-based/modelless duality framework
- **Plan 009** (REST Speculative Decoding) — natural use case for maxsim reranking