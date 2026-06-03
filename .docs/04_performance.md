# katgpt-rs: Performance Engineering

## Benchmark Results (release build, Apple Silicon)

Run on Apple Silicon (single-threaded, `--release`, 2000 iterations + 50 warmup, **zero-alloc hot paths**). Latest run: **2026-05-29**, default features.

**Models:** Target (embd=16, heads=4, mlp=64) · Draft (embd=4, heads=2, mlp=16) · `cargo run --release`

```
Method                         Throughput         μs/step  Avg Accept Len
───────────────────────────────────────────────────────────────────────────────
Transformer AR                  1,711,230 tok/s       0.58            1.00
DFlash                            423,396 tok/s       2.36            8.00
DDTree Build                      370,225 trees/s      2.70            —
Speculative (Simulated)           854,281 tok/s       5.85            5.00
Speculative (AR Draft)          1,144,103 tok/s       6.12            7.00
Leviathan (Algorithm 1)         1,630,214 tok/s       0.61            1.00
Leviathan (w/ rollback)           173,560 tok/s       6.75            1.17
Spec (conditioned)                933,739 tok/s       7.23            6.75
Prefill (no compress)             217,825 ops/s       4.59           64.00
Prefill (compressed)              207,760 ops/s       4.81            7.00
DDTree (chain-seed)               386,835 trees/s      2.58           16.00
DDTree (screened R=1.0)           296,927 trees/s      3.37           16.00
forward_raven (16 slots)        2,019,460 ops/s        0.49            —
raven_recall (1000 noise)      22,944,726 ops/s        0.04           63.21
───────────────────────────────────────────────────────────────────────────────
```

> **Reading the gain honestly:** at this micro scale the target forward pass is so cheap that
> plain AR has the highest raw `tok/s`. Speculative decoding's real win is the **avg accept
> length** — 7–8 tokens verified per target pass (DFlash 8.0, AR-draft 7.0, conditioned 6.75) —
> which is what amortizes a large target model. The headline `Speedup` line printed by `main.rs`
> compares `raven_recall` ops/s against `forward(flat)` ops/s (mislabeled as "Speculative vs AR"),
> so treat the per-row accept length, not that single number, as the speculative-decoding metric.

Speedup: Speculative vs AR went from **0.72×** → **1.82×** after zero-alloc optimization (earlier run `047`, commit `4a6b592`).

### GRAM Width-vs-Depth (`.benchmarks/019_gram_width_depth.md`)

Infrastructure benchmark validating width >> depth on DDTree with SDE noise. GOAT PENDING (1/3) — infrastructure validated, real game arenas needed for full proof.

| Sweep | Result |
|-------|--------|
| Width K=1→20 | +0.15% quality (linear latency cost) |
| Depth T=1→16 | -32.4% quality (diminishing returns) |

## Hot Path Optimizations

### Inline + Unsafe
- `#[inline(always)]` on all hot kernels: `matmul`, `softmax`, `rmsnorm`, `forward`, `sample_token`
- `get_unchecked` / `get_unchecked_mut` in inner matmul loops — eliminates bounds checks
- `copy_nonoverlapping` for KV cache store — faster than `copy_from_slice` for known sizes
- Edition 2024: explicit `unsafe {}` blocks inside `unsafe fn`
- SIMD intrinsics (NEON/AVX2) in `crates/katgpt-core/src/simd.rs` (re-exported via `src/simd.rs`) — runtime detection, safe API wrapping `core::arch::{aarch64, x86_64}` (Plan 060)

### Fused Kernels
- **`matmul_relu`**: single-pass MLP hidden layer (avoids extra scan of hidden buffer) — SIMD-accelerated dot product + fused ReLU zero-clamp
- **`attention_head`**: fused score → softmax → weighted value (avoids separate softmax write-back) — SIMD-accelerated via `simd_dot_f32`
- **Optimized softmax**: one-pass exp+sum, `inv_sum = 1.0/sum` multiply instead of divide
- **Optimized rmsnorm**: two-pass with `inv_rms` multiply instead of divide

### Impact
| Method | Before | After | Change |
|--------|--------|-------|--------|
| Transformer AR | 831K tok/s | 900K tok/s | **+8.3%** |
| DFlash | 2,941K tok/s | 4,231K tok/s | **+43.8%** |

## SIMD Acceleration (Plan 060)

NEON (ARM) / AVX2 (x86_64) SIMD dispatch via `katgpt-core/src/simd.rs` (re-exported through `src/simd.rs`). All kernels use runtime `SimdLevel` detection and provide scalar fallbacks. Public API:

### Kernel-Level Throughput (NEON, Apple Silicon, release)

| Operation | Throughput | µs/op |
|-----------|-----------|-------|
| matmul [16×16] | 15.6M ops/s | 0.06µs |
| matmul [32×32] | 5.1M ops/s | 0.20µs |
| matmul_relu [32×32] | 4.4M ops/s | 0.23µs |
| hla_update hd=4 | 16.4M ops/s | 0.06µs |
| ahla_step hd=4 | 18.2M ops/s | 0.05µs |
| maxsim_score | — | SIMD-parallel |
| maxsim_score_packed | — | batched MaxSim |
| simd_fused_decay_write | — | fused dst=α·dst+β·src |
| simd_add_into | — | vectorized dst=a+b |

### Additional SIMD Kernels (Plan 060+)

| Kernel | Description | Feature gate |
|--------|-------------|-------------|
| `simd_dot_f32` | Dot product (NEON/AVX2/scalar) | — |
| `simd_outer_product_acc` | Outer product accumulation | — |
| `simd_matvec` | Matrix-vector multiply | — |
| `simd_matmul_rows` | Row-parallel matmul | — |
| `simd_matmul_rows_parallel` | Rayon-parallel matmul (threshold-gated) | — |
| `simd_matmul_relu_rows` | Row-parallel matmul + ReLU clamp | — |
| `simd_fma_row` | Fused multiply-accumulate row | — |
| `simd_dot_f16_f32` | f16/f32 mixed-precision dot product | — |
| `simd_matmul_f16_f32_rows` | f16 weight / f32 input matmul | — |
| `simd_matmul_f16_f32_rows_parallel` | Rayon-parallel f16/f32 matmul | — |
| `simd_sparse_dot_f32` | Sparse dot product (alive mask) | — |
| `simd_sparse_matmul_rows` | Sparse matmul with index tracking | — |
| `simd_scale_inplace` | In-place scalar multiply | — |
| `simd_add_scalar_inplace` | In-place scalar addition | — |
| `simd_fused_sub_scale_inplace` | Fused dst = (a - b) * s | — |
| `simd_sum_f32` | Horizontal sum | — |
| `simd_add_inplace` | In-place vector add | — |
| `simd_max_f32` | Horizontal max | — |
| `simd_scale_mul_inplace` | Fused element-wise scale-multiply | — |
| `simd_exp_inplace` | Cephes-based exp approximation | — |
| `simd_fused_sub_acc` | Fused dst += a - b | — |
| `simd_fused_scale_acc` | Fused dst += α·src | — |
| `simd_gram_f32` | Gram matrix A^T A (upper triangle) | — |
| `simd_sum_sq` | Sum of squares | — |
| `simd_sum_abs_f32` | Sum of absolute values | — |
| `simd_dist_sq` | Squared Euclidean distance | — |
| `simd_ternary_matvec` | Ternary weight matvec (PlasmaPath) | `plasma_path` |
| `simd_ternary_matmul_batch` | Batched ternary matmul (Rayon) | `plasma_path` |
| `sigmoid_margin_loss` | Retrieval margin loss (MaxSim scoring) | — |
| `compute_retrieval_margin` | Retrieval quality margin | — |
| `dim_sufficiency_bound` | Dimension sufficiency theoretical bound | — |

### End-to-End Forward Throughput (Config::micro, 8 positions)

| Variant | tok/s | µs/tok |
|---------|-------|--------|
| forward (SDPA) | 1.1M/s | 0.93µs |
| forward_hla | 939K/s | 1.06µs |
| forward_ahla | 1.2M/s | 0.84µs |

### 30K CCU @ 20Hz Feasibility

| Metric | Value |
|--------|-------|
| Required throughput | 600K tok/s |
| Single-core HLA | 939K tok/s |
| Cores needed | 1 |
| 8-core headroom | 9.8× |

Verdict: **Single ARM core handles 30K concurrent game AI users at 20Hz with 9.8× headroom.**

### HLA Training (Plan 059)

SDPA→HLA distillation experiment shows KL divergence does NOT converge (Path C decision):
- SDPA→AHLA: KL diverges 4.62→7.43 over 500 steps
- SDPA→HLA: KL oscillates 8.54→8.42, cosine similarity drops
- Root cause: LoRA on QKV adjusts *inputs*, not the *attention mechanism itself*
- HLA is inference-only — streaming attention without SDPA's quadratic cost
- DeltaMemoryState handles facts/retrieval separately

## Zero-Allocation Strategy

### Problem
Every `dflash_predict`, `build_dd_tree`, and speculative step was allocating `Vec`s per call:
- `ForwardContext::new()` per DFlash call
- `MultiLayerKVCache::new()` per step
- `Vec::with_capacity()` for marginals, tree nodes, heap
- `logits.to_vec()` per forward step

### Solution: Pre-allocated Contexts

**SpeculativeContext** (`speculative/types.rs`):
- Holds `ForwardContext`, `MultiLayerKVCache`, flat marginals buffer, probs buffer, sampled tokens, accepted tokens, path buffer, residual buffer, p_distributions buffer
- `new(config)` allocates once, `reset()` clears for reuse
- All `_with()` function variants accept `&mut SpeculativeContext`

**TreeBuilder** (`speculative/dd_tree.rs`):
- Holds pre-allocated `BinaryHeap<TreeNode>`, `Vec<TreeNode>`, chain buffers
- `build()` clears and reuses internal buffers
- Returns `&[TreeNode]` (borrowed slice)

**ForwardContext** (`transformer.rs`):
- Already pre-allocated since Plan 003
- All forward pass intermediates (x, q, k, v, attn_out, hidden, logits) reused in-place

### Impact
| Method | Before (μs) | After (μs) | Improvement |
|--------|-------------|-------------|-------------|
| DFlash | 2.60 | 1.89 | **38% faster** |
| DDTree Build | 3.19 | 2.32 | **27% faster** |
| Speculative (Simulated) | 5.92 | 4.37 | **26% faster** |
| Speculative (AR Draft) | 5.70 | 4.26 | **25% faster** |
| Prefill (no compress) | 23.78 | 3.29 | **623% faster** |
| Prefill (compressed) | 23.99 | 3.57 | **572% faster** |
| DDTree (chain-seed) | 3.16 | 2.24 | **29% faster** |

## Rayon Parallelism

### Current Usage
- `dflash_predict_parallel` — `into_par_iter` + `map_init` with per-worker `ForwardContext`/`MultiLayerKVCache`
- `generate_batch` — multi-sample generation via `par_iter` with per-worker contexts

### Parallel Threshold
- `config.parallel_threshold` (default 128) — skip `par_iter` when `n_embd ≤ threshold`
- At micro scale (n_embd=16), Rayon overhead (~1-5μs) dominates sequential cost (~0.3μs)
- Parallelism only beneficial when individual forward passes are > 5μs

### What NOT to Parallelize
- DDTree initial heap population — vocab=27, not worth overhead for small vocab
- Benchmark runner — reduces wall time only, not throughput; already fast enough

## PagedKVCache Design

### Problem
DDTree branches clone entire `MultiLayerKVCache` → most data is shared prefix:
- small_target config: 32 × 131 KB = 4.2 MB of near-identical copies

### Solution
```rust
pub struct PagedKVCache {
    pages: Vec<Vec<f32>>,                     // pool of [PAGE_SIZE * kv_dim * 2] pages (K|V)
    layer_page_tables: Vec<Vec<Vec<usize>>>,  // [layer][seq] → page indices
    free_pages: Vec<usize>,                   // free list for page reuse
    kv_dim: usize,                            // n_kv_head * head_dim
    total_pages: usize,                       // monotonically increasing
    deficits: Vec<usize>,                     // scratch: per-layer page deficits
    new_pages: Vec<Vec<usize>>,               // scratch: per-layer new page indices
    all_new_buf: Vec<usize>,                  // scratch: flat buffer for ensure_pages()
    page_ref_counts: Vec<u32>,                // O(1) rollback (replaces HashSet scan)
    rollback_removed: Vec<usize>,             // scratch: drained page indices
}
```
- PAGE_SIZE = 16 tokens (power of 2)
- `fork(seq_idx, fork_at_pos)` — copy-on-write: shares prefix pages, new pages only after fork
- `alloc_page()` — reuse from free list or grow pool
- Memory: O(tree_budget × pos_used) instead of O(tree_budget × block_size)

### Status
- Struct implemented and tested (Plan 011)
- O(1) rollback via `page_ref_counts` (Issue 053)
- Zero-alloc hot path via scratch buffers (`deficits`, `new_pages`, `all_new_buf`, `rollback_removed`)
- Currently DDTree uses flat `snapshot()/restore()` which works but copies more data

### ScreeningPruner Overhead (Plan 021)

The `build_screened()` path adds ~28% overhead vs `build()` due to `relevance()` trait call + `ln(R)` per candidate:

| Method | μs/step | Notes |
|--------|---------|-------|
| DDTree (no chain) | 2.31 | Original `ConstraintPruner` path — zero regression |
| DDTree (screened R=1.0) | 2.96 | `NoScreeningPruner` — `ln(1.0)=0.0` no-op penalty |
| DDTree (screened adapter) | 2.94 | `BinaryScreeningPruner(NoPruner)` — adapter overhead |

Overhead is expected: screening calls `relevance()` + computes `ln(R)` for every candidate token. This is **opt-in** — existing `build()` path is untouched. When screening actually eliminates garbage branches, fewer nodes are explored → effective throughput improves.

## What each benchmark measures

The benchmarks progress from individual components to full pipelines. Benchmark categories (`BenchCategory` enum in `src/benchmark/mod.rs`):

| Category | Tag | Module | Description |
|----------|-----|--------|-------------|
| Speculative | `speculative` | `speculative.rs` | Speculative decoding throughput |
| TreeBuild | `tree_build` | `speculative.rs` | DDTree build performance |
| Infrastructure | `infrastructure` | `infrastructure.rs` | KV cache, prefill, recall primitives |
| HeuristicLearning | `heuristic_learning` | `heuristic.rs` | G-Zero self-play components |
| SpecDecoding | `SD` | `speculative.rs` | Draft/verify, tree search, MTP |
| KvOptimization | `KV` | `infrastructure.rs` | Cache compression, pruning, quantization |
| Attention | `Attn` | `hla.rs` | Novel attention, linear attention |
| Noise | `Noise` | `noise.rs` | SDE injection, diffusion schedules |
| Distillation | `Distill` | `distillation.rs` | LoRA, quantization, knowledge transfer |
| TestTimeCompute | `TTC` | `ttc.rs` | Adaptive budget, bandit exploration |
| Routing | `Route` | `routing.rs` | Raven slot routing, delta routing |
| Diffusion | `Diff` | `diffusion.rs` | D2F block/pipeline denoising |
| Game | `Game` | `games.rs` | E2E game timing (Sudoku/Go/Monopoly/Bomber) |
| SimdPerf | `SIMD` | `simd.rs` | Dense/sparse matmul, ternary, lattice |
| E2EGame | `e2e_game` | `batch.rs` | E2E timing through plasma/hot/warm/cold cache states |

### Core Components

| Benchmark | What it does | Why it matters |
|-----------|-------------|----------------|
| **Transformer AR** | Baseline: 1 target model forward pass → 1 token. The "slow path" that speculative decoding tries to beat. | Ground truth. If speculative can't beat this, it's not worth the complexity. |
| **DFlash** | Block-parallel draft prediction: runs 8 independent forward passes on the tiny draft model. Produces 8 marginal distributions without autoregressive feedback. | The "fast but inaccurate" draft. 8 tokens predicted in parallel vs 1 from AR. |
| **DDTree Build** | Builds a Best-First Search tree (budget=16 nodes) from DFlash marginals. | Converts flat marginals into a tree of candidate paths. Maximizes Expected Acceptance Length. |

### Speculative Decoding Pipelines

| Benchmark | Pipeline | Why it matters |
|-----------|----------|----------------|
| **Speculative (Simulated)** | DFlash → DDTree → extract best path → simulated 75% acceptance cap → bonus token. No target model. | Full pipeline without expensive target verification. Avg 5 accepted tokens per step. |
| **Speculative (AR Draft)** | Same pipeline but uses autoregressive drafting: each step feeds back the sampled token. | AR drafting produces more coherent sequences → better acceptance (7 tokens vs 5). |
| **Spec (unconditioned)** | Identical to "Speculative (Simulated)" — baseline for comparison with conditioned variant. | Control group: same draft model, same pipeline, no target information. |
| **Spec (conditioned)** | Target-conditioned draft: runs target model forward → projects hidden state into draft KV cache → better-informed marginals → DDTree → simulated acceptance. | Draft model gains partial access to target's representation. Higher avg acceptance (6.74 vs 5.0). |

### Leviathan: Real Target Verification (Algorithm 1)

| Benchmark | Pipeline | Why it matters |
|-----------|----------|----------------|
| **Leviathan (Algorithm 1)** | Full Algorithm 1: AR draft → target model scores ALL drafted tokens → real p/q rejection sampling → residual distribution → bonus token. **Mathematically distribution-preserving**. | Proves the algorithm works end-to-end. Slow here because our target is only 4× bigger than draft. |
| **Leviathan (no rollback)** | Standard Leviathan that resets target KV cache each step. No branch recovery. | Baseline for rollback comparison. |
| **Leviathan (w/ rollback)** | Leviathan with KV cache snapshot/rollback. On rejection, rolls back and tries the next candidate branch. | 73% faster than no-rollback. Essential for multi-branch verification. |

### Prompt Compression (PFlash-inspired)

| Benchmark | Pipeline | Why it matters |
|-----------|----------|----------------|
| **Prefill (no compress)** | Uses draft model's self-attention weights as per-token importance proxy. Runs forward passes over 64-token prompt, extracts attention scores. All 64 tokens kept. | Measures raw importance scoring speed. 623% zero-alloc speedup from reusing `SpeculativeContext`. |
| **Prefill (compressed)** | Same scoring, then compresses to keep_ratio=0.1 — keeps only top 10% most important tokens (~7 of 64). | Actual use case: reduces target prefill work by ~9×. |

### Tree Building Variants

| Benchmark | What differs | Why it matters |
|-----------|-------------|----------------|
| **DDTree (no chain)** | Standard best-first: seeds heap with all root marginals, expands greedily. | Baseline tree construction. 2.31 μs. |
| **DDTree (chain-seed)** | Chain-seed optimization: builds greedy backbone first, then seeds heap with siblings. | Chain provides a "highway" that pure best-first might miss. 2.24 μs (3% faster). |
| **DDTree (screened R=1.0)** | `build_screened()` with `NoScreeningPruner` — calls `relevance()` (returns 1.0) + `ln(1.0)=0.0`. | Measures screening trait call + log computation overhead. 2.96 μs (+28%). |
| **DDTree (screened adapter)** | `build_screened()` with `BinaryScreeningPruner(NoPruner)` — adapter wrapping. | Adapter overhead is negligible. 2.94 μs. |

### Raven RSM (O(1) Routing Slot Memory)

| Benchmark | What it does | Why it matters |
|-----------|-------------|----------------|
| **forward_raven (16 slots)** | Forward pass through `RavenKVCache` — 16 fixed routing slots, O(1) attention. | Proves Raven is faster than flat forward (0.62 vs 0.83 μs). Slot memory wins. |
| **raven_recall (1000 noise)** | Recall accuracy test: inject 1000 noise tokens, verify target tokens are recalled from frozen slots. | 9.25M tok/s with 63.21/64 recall — proves slot memory retains signal through noise. |

## TurboQuant KV Cache Compression (Plan 043) — Legacy Baseline

> **Note:** TurboQuant is now a legacy baseline for benchmarking/education. **SpectralQuant** (Plan 078, feature `spectral_quant`, on by default) replaces it with calibrated eigenbasis rotation + water-fill bit allocation. See `src/spectralquant/`.

Files: `src/turboquant/{mod,codebook,forward,kv_cache,rotation,types}.rs`

Feature gate: `turboquant` (opt-in, NOT in default features)

### Memory Compression
| Bits | Bytes/token | Compression | Key cos_sim | Attention corr |
|:----:|:----------:|:-----------:|:-----------:|:--------------:|
| 2 | 16 | 8.0× | 0.9242 | 0.9450 |
| 3 | 24 | 5.3× | 0.9825 | 0.9907 |
| 4 | 24 | 5.3× | 0.9958 | 0.9978 |

### At Scale (32K ctx, 32 layers, hd=128)
- Flat f32: 1073.7 MB
- TQ 3-bit: 151.0 MB (7.1× compression)
- TQ 2-bit: 83.9 MB (12.8× compression)

### Trade-off
Store+dequantize has ~200× compute overhead vs flat f32 copy. Net win at long contexts where memory bandwidth is the bottleneck, not compute.

## SpectralQuant KV Cache Compression (Plan 078) — Default

Calibrated eigenbasis quantization that replaces TurboQuant as the default KV cache compression method.

Files: `src/spectralquant/{mod,forward,spectral,spectral_kv_cache,spectral_rotation,nonuniform_quant,types}.rs`

Feature gate: `spectral_quant` (**on by default** in `Cargo.toml`)

### Architecture
1. **Offline calibration**: covariance → eigendecomposition → eigenbasis (`spectral.rs`)
2. **Two-regime allocation**: semantic (top `d_eff` dims after rotation) + tail dims (`nonuniform_quant.rs`)
3. **Water-fill**: per-dim bit allocation proportional to eigenvalue (`spectral.rs::waterfill_bits`)
4. **Lloyd-Max**: optimal non-uniform scalar quantizer per regime (`spectral.rs::LloydMaxQuantizer`)
5. **Variable-bit packing**: compressed storage with per-dim bit widths (`spectral_kv_cache.rs`)

### Key Types

```rust
// Per-layer calibration state (spectralquant/types.rs)
pub struct SpectralQuantLayer {
    calibration: SpectralQuantCalibration,  // eigenvectors, eigenvalues, d_eff
    qjl_signs: Vec<f32>,                    // QJL projection
    tail_codebook: LloydMaxCodebook,
    semantic_codebook: Option<LloydMaxCodebook>,          // v1 uniform
    per_dim_semantic_codebooks: Option<Vec<LloydMaxCodebook>>, // v2 water-fill
    d_eff: usize, b_high: u8, b_low: u8,
}

// Zero-alloc compressed KV cache (spectralquant/spectral_kv_cache.rs)
pub struct SpectralQuantKVCache {
    layers: Vec<SpectralQuantLayer>,
    key_indices: Vec<Vec<Vec<u8>>>,  // variable-bit packed
    key_norms: Vec<Vec<f32>>,
    val_indices: Vec<Vec<Vec<u8>>>,  // variable-bit packed
    val_norms: Vec<Vec<f32>>,
    // Scratch buffers (zero-alloc hot path)
    scratch_normalized, scratch_rotated, scratch_unrotated,
    scratch_semantic_indices, scratch_tail_indices,
    scratch_all_indices, scratch_all_bits,
    pos, n_layers, kv_dim, max_seq_len,
}
```

### Key Functions
- `calibrate_eigenbasis()` — offline calibration from sample covariance
- `calibrate_eigenbasis_dual_gram()` — dual Gram PCA calibration (feature `dual_gram_pca`)
- `waterfill_bits()` — per-dim bit allocation
- `participation_ratio()` — effective dimensionality `d_eff = (Σλ)² / Σ(λ²)`
- `spectral_gap()` — `λ_{d_eff} / λ_{d_eff+1}`
- `attention_spectralquant()` — self-contained attention scoring with dequantization
- `par_dequantize_spectral_keys_flat()` — Rayon-parallel dequantization (feature `maxsim`)
- `par_maxsim_score_spectralquant()` — parallel MaxSim scoring (features `spectral_quant` + `maxsim`)

### Rotation
- `SpectralRotation` (`spectral_rotation.rs`) — data-driven orthogonal rotation using calibrated eigenvectors
- `RandomRotation` — TurboQuant-compatible fallback (feature `turboquant`)

## PFlash Block-Sparse Prefill (Plan 044)

### Sequence Reduction
| Context | Alpha | Before | After | Reduction |
|:-------:|:-----:|:------:|:-----:|:---------:|
| 512 | 0.15 | 512 | 192 | 2.7× |
| 1024 | 0.15 | 1024 | 192 | 5.3× |
| 2048 | 0.15 | 2048 | 192 | 10.7× |
| 4096 | 0.15 | 4096 | 192 | 21.3× |

### NIAH Retrieval
20/20 = 100% across all context sizes (256-4096) and alpha values (0.05-0.85).

### block_select Throughput
| Scale | Blocks | blocks/s | µs/call |
|:-----:|:------:|:--------:|:-------:|
| 2K | 64 | 30M | 2.1µ |
| 32K | 1024 | 28M | 36µ |
| 128K | 4096 | 29M | 140µ |

### Combined with TurboQuant
TQ 3-bit + PF α=0.15 = 14.9% resources (6.7× total reduction).

## Feature-Gate Throughput Impact (bench 063→064 A/B, Plan 054)

Default features changed in Plan 051 from `["bandit", "g_zero"]` to `["sparse_mlp", "domain_latent", "ppot", "bandit"]`. Measured on same cool CPU, back-to-back:

### A/B Results

| Method | `bandit,g_zero` | `sparse_mlp,domain_latent,ppot,bandit` | Delta |
|--------|-----------------|----------------------------------------|-------|
| forward (flat) | 1,164,412 | 926,060 | **-20.5%** |
| forward_paged | 1,035,403 | 793,110 | **-23.4%** |
| Transformer AR | 1,170,941 | 924,803 | **-21.0%** |
| Leviathan (Alg 1) | 112,677 | 90,934 | **-19.3%** |
| DDTree Build | 362,635 | 363,978 | +0.4% |
| DDTree (chain-seed) | 378,874 | 384,435 | +1.5% |
| forward_raven | 1,649,131 | 1,594,088 | -3.3% |
| TQ-3bit (alloc) | 1,858,844 | 1,826,570 | -1.7% |

### Root Cause

1. **`sparse_mlp`** — `sparse_matmul` adds index-tracking overhead (`active_indices`, `active_values` buffers + alive-count branch) vs plain `matmul`. At micro scale (mlp=64), the extra branching costs more than skipping zero elements saves.
2. **`domain_latent`** — adds an extra `Option<&DomainLatent>` parameter to `forward_base()` + mid-layer `if layer_idx == n_layer / 2` branch. Changes function signature → different inlining/register allocation.
3. **DDTree, Raven, TQ, PFlash unaffected** — they use different code paths or the overhead is amortized.

### Verdict

Not a regression from Plan 054 stepcode (feature is off-by-default, not compiled). This is a **Plan 051 default-features decision**: trade ~20% raw forward throughput for sparsity + domain conditioning capability.

Run bench with `--features g_zero` to include heuristic learning (Plan 049: gate stays until T5 proven). `g_zero` does NOT touch `forward()` hot path — zero hits in `transformer.rs`.

### Bench 065 Stability Confirmation (064→065, same HEAD `1a00e32`)

| Method | 064 | 065 | Delta |
|--------|-----:|-----:|------:|
| DDTree (chain-seed) | 389,978 | 389,172 | -0.2% |
| DDTree Build | 372,598 | 366,950 | -1.5% |
| DFlash | 450,259 | 472,035 | +4.8% |
| PFlash block_select | 1,184,015 | 1,187,070 | +0.3% |
| Speculative (AR Draft) | 1,417,878 | 1,399,307 | -1.3% |
| TQ-3bit (zero-alloc) | 2,401,050 | 2,417,375 | +0.7% |
| Transformer AR | 924,098 | 945,989 | +2.4% |
| forward (flat) | 831,619 | 917,519 | +10.3% |
| forward_paged | 780,361 | 844,141 | +8.2% |

Core model benchmarks ±2% stable. Infrastructure (`forward (flat)`, `forward_paged`) shows higher variance due to thermal sensitivity — 064 ran on warmer CPU. Cool CPU + 3s cooldowns + infrastructure-first run order (commit `05d0a51`) gives reproducible results.

### Regression Visibility

- **`features` column** in `bench/*_results.csv` and `bench/timeseries.csv` — active feature flags (e.g. `sparse_mlp+domain_latent+ppot+bandit` vs `bandit+g_zero`) make feature-gate throughput diffs traceable across runs.
- **Timeseries chart titles** include the latest run's features (e.g. `Infrastructure Primitives — Time Series [sparse_mlp+domain_latent+ppot+bandit]`).
- **Run order:** Infrastructure benches run first (cool CPU) → speculative → tree → heuristic. 3s inter-group cooldowns reduce thermal throttling noise. The `forward (flat)` regression is clearly visible as a step-down in `bench/timeseries_infrastructure.png` when features change from `bandit+g_zero` to `sparse_mlp+domain_latent+ppot+bandit`.

## What We Don't Do (and why)

| Technique | Reason |
|-----------|--------|
| Rayon parallel matmul | n_embd=16, mlp=64 — thread pool overhead dominates |
| `std::simd` / `portable_simd` | Nightly-only; we use `core::arch` intrinsics directly (Plan 060) |
| Cache tiling for attention | block_size=16 already fits L1 |
| f16/bf16 weights | Would halve memory bandwidth but requires `half` crate; `simd_dot_f16_f32` kernels exist for mixed-precision matmul |
| GPU compute in inference | CPU-only for inference; GPU training is out of scope |