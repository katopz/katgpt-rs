# Research 77: ThunderKittens — Tile DSL for High-Performance AI Kernels

> **Source:** [Dissecting ThunderKittens: Anatomy of a Compact DSL for High-Performance AI Kernels](https://hamzaelshafie.bearblog.dev/dissecting-thunderkittens-anatomy-of-a-compact-dsl-for-high-performance-ai-kernels/) — Hamza Elshafie, May 2026
> **Paper:** [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399) — Hazy Research, Stanford
> **Code:** [github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) (Hopper/Blackwell, CUDA C++20)
> **Benchmark Repo:** `.raw/tk_attention/` (LCF attention kernel vs FA2/FA3)
> **Date:** 2026-05-21 (article), distilled 2026-05-23
> **Related:** Research 66 (TileRT), Research 67 (CODA), Research 29 (Rust GPU), Research 71 (DashAttention), Research 22 (Lighthouse Attention), Plan 106 (riir-ai CubeCL rewrite)
> **Verdict: CONCEPTUAL ADOPTION — TK's hardware-specific primitives (WGMMA, TMA, TMEM, SMEM swizzling) are NOT portable to our wgpu/Metal/CubeCL stack. However, three architectural patterns ARE portable: (1) tile-structured attention with online softmax (CPU SIMD path in katgpt-core), (2) LCF producer-consumer pipeline template (guides CubeCL kernel structure in Plan 106), (3) persistent tile scheduling with super-grouping for cache locality. Feature-gate: `tiled_attention` in katgpt-core for CPU flash attention; guide Plan 106 T2.5+ for GPU tiled path.**

---

## Executive Summary

ThunderKittens (TK) is a CUDA-embedded DSL from Stanford's Hazy Research that abstracts GPU kernel development around **16-based tile primitives** (`st`, `rt`, `gl`, `sv`, `rv`). It targets Hopper (H100) and Blackwell (B200) GPUs, achieving 98% cuBLAS on GEMM and competitive performance with FlashAttention-3 on attention kernels — with dramatically less code than raw CUDA/CUTLASS.

**The punchline for our stack:** We do NOT have NVIDIA Tensor Cores, TMA, or TMEM. Our GPU path is wgpu → Metal/Vulkan via CubeCL. Our CPU path is NEON/AVX2 SIMD. But TK's *architectural patterns* — not its hardware-specific instructions — are what we should distill:

1. **CPU (katgpt-core):** TK's online-softmax flash attention algorithm maps directly to our SIMD matmul pipeline. We already have `softmax_scaled()` and `matmul()` in `katgpt-core/src/types.rs`. What we lack is the *tiled iteration* pattern that avoids materializing the full `N×N` attention score matrix. This is a pure algorithmic improvement, independent of GPU hardware.

2. **GPU (riir-ai):** CubeCL already provides a tiled matmul pipeline (Plan 106 T2.4 complete). TK's LCF (load-compute-finish) template shows how to structure a *kernel template* with producer/consumer roles, pipelined SMEM staging, and persistent task scheduling. This guides our CubeCL kernel architecture.

**What TK does NOT change:** Our model-based/modelless duality, speculative decoding stack, pruners, validators, or game AI. TK is a kernel-level execution optimization, not a reasoning strategy.

---

## 1. TK's Core Abstractions (And What Maps to Us)

### 1.1 Tile Primitives

| TK Abstraction | Purpose | Our Analog |
|:---|:---|:---|
| `gl<T, b, d, r, c>` | Global memory layout descriptor (GMEM) | `wgpu::Buffer` + stride/shape metadata in our dispatch structs |
| `st<T, rows, cols>` | Shared memory tile (SMEM), owns storage | CubeCL `SharedMemory<T, N>` (already used in Plan 106) |
| `rt<T, rows, cols>` | Register tile, warp-distributed | SIMD register file — our `simd_dot_f32` accumulates into `f32x4`/`f32x8` |
| `sv<T, length>` | Shared vector (simple array in SMEM) | CubeCL `SharedMemory<T, N>` for reductions |
| `rv<T, length, layout>` | Register vector, warp-distributed | SIMD reduction result (already in our `softmax_scaled` via horizontal sum) |

**Key insight:** TK's 16-based tile granularity (`BASE_TILE_DIM = 16`) comes from tensor core fragment layouts. On CPU, our natural tile is the SIMD width: 4 (NEON) or 8 (AVX2). On Metal, CubeCL's `simdgroup_matrix` is 8×8×8. The *number* differs, but the *pattern* — building larger operations from fixed-size primitives — is the same.

### 1.2 Compute Primitives

| TK Operation | What It Does | Our Analog |
|:---|:---|:---|
| `warpgroup::mm<A, B>` | Warpgroup-level MMA (4 warps, 128 threads) | CubeCL `cubecl::matmul::tiled` (Plan 106 T2.4) |
| `warp::sum<axis::COL>(tile)` | Column-wise reduction via shuffle tree | `simd_horizontal_sum()` in `katgpt-core` |
| `warp::max<axis::COL>(tile)` | Column-wise max reduction | Our `softmax_scaled` already does this per-row |
| `warp::exp2(...)` | Elementwise exp2 on register tile | `f32::exp()` in our SIMD loops |
| `warp::add(a, b, dst)` | Elementwise add on shared/register tiles | `simd_add_f32()` in our NEON/AVX2 backends |

**The gap:** TK's tensor core operations achieve ~500 TFLOP/s on H100. Our Metal SIMD achieves ~2-5 TFLOP/s. The *algorithmic structure* is the same; the *throughput* differs by 100×. This means our CPU/GPU attention kernels should use the same tiling pattern, but with smaller tile sizes.

---

## 2. LCF Pipeline Template — The Key Architectural Pattern

TK's `lcf` (load-compute-finish) template is the most relevant abstraction for us. It provides:

### 2.1 Structure

```text
┌─────────────────────────────────────────────────────────┐
│ LCF Pipeline Template                                    │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │ common_setup │    │   Main Loop  │    │  finish    │ │
│  │ (task decode)│───▶│              │───▶│  (output)  │ │
│  └──────────────┘    │  ┌────────┐  │    └────────────┘ │
│                      │  │producer│──┼──load(Kj,Vj)──▶  │
│                      │  │ .load  │  │                   │
│                      │  └────────┘  │                   │
│                      │  ┌────────┐  │                   │
│                      │  │consumer│──┼──compute(Sij,PV)─▶│
│                      │  │.compute│  │                   │
│                      │  └────────┘  │                   │
│                      └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

**Roles:**
- **Producer warpgroup** (4 warps): issues TMA loads, manages SMEM staging ring buffer
- **Consumer warpgroups** (8+ warps): runs MMA compute, maintains online softmax state
- **Finish phase**: normalizes output, writes back via TMA store

### 2.2 Kernel Schema (What the User Provides)

```rust
// Pseudocode mapping to our types
struct AttnLayout {
    // Global tensors
    globals: { Q: Gl<Bf16>, O: Gl<Bf16>, K: Gl<Bf16>, V: Gl<Bf16> },
    // What gets streamed per iteration (Kj, Vj)
    input_block: { K: StTile<Bf16, Br, D>, V: StTile<Bf16, Bc, D> },
    // What persists across iterations (Qi — fixed for the task)
    scratch_block: { Q: [StTile<Bf16, Br, D>; NUM_CONSUMER_WGS] },
    // Per-task identity
    common_state: { batch: u32, head: u32, base_q_tile: u32 },
    // Per-warp running state for online softmax
    consumer_state: {
        max_vec: ColVec<f32>,      // running row-wise max
        norm_vec: ColVec<f32>,     // running row-wise sum
        o_reg: RegTile<f32>,       // running output accumulator
    },
}
```

### 2.3 Online Softmax Algorithm (The Math That Maps to Us)

This is the core algorithmic contribution that IS hardware-independent:

```text
For each KV tile j = 0..ceil(N/Bc):
  1. Sij = Qi @ Kj.T           // matmul (tile-sized)
  2. mi_new = max(mi_old, rowmax(Sij))
  3. P̃ij = exp(Sij * scale - mi_new * scale)  // exp2 trick
  4. correction = exp(mi_old * scale - mi_new * scale)
  5. ℓi = correction * ℓi + rowsum(P̃ij)
  6. Oi = correction * Oi + P̃ij @ Vj   // fused scale + matmul
  7. mi = mi_new
After all tiles:
  Oi = Oi / ℓi                 // final normalize
```

**Why this matters for us:** Our current CPU attention in `types.rs` materializes the full `N×N` score matrix for small configs. For `N > 1024`, this becomes a bottleneck. The tiled online softmax avoids the full materialization, working in `Br × Bc` tiles. On CPU with SIMD, our tile sizes would be `Br = SIMD_WIDTH` (4 or 8 rows at a time).

---

## 3. Benchmark Results (From `.raw/tk_attention/`)

### 3.1 Setup

- GPU: NVIDIA H100 PCIe 80GB
- TK kernel: `attn_lcf.cu` with `B_r=64, B_c=128, D=128`
- Baselines: FlashAttention-2 (PyTorch SDPA), FlashAttention-3 (Hopper)
- Inputs: seeded random BF16 (NOT all-zeros, avoiding power throttling artifacts)
- Shape: `B=16, H=16, D=128`, sequence lengths 768–12288

### 3.2 Results Summary

| Seq | TK (Ours) TFLOP/s | FA2 TFLOP/s | FA3 TFLOP/s | TK vs FA2 | TK vs FA3 |
|:----|:-----------------|:------------|:------------|:----------|:----------|
| 768 | ~520 | ~330 | ~500 | **1.58× faster** | ~1.04× faster |
| 1536 | ~560 | ~380 | ~580 | **1.47× faster** | 0.97× |
| 3072 | ~580 | ~380 | ~610 | **1.53× faster** | 0.95× |
| 6144 | ~590 | ~380 | ~640 | **1.55× faster** | 0.92× |
| 12288 | ~590 | ~370 | ~660 | **1.59× faster** | 0.89× |

**Key observations:**
1. TK beats FA2 by 44–59% across all sequence lengths (compact implementation!)
2. TK trails FA3 by 5–15% at longer sequences (FA3 has hardware-specific optimizations)
3. The LCF template achieves competitive performance with very little code vs raw CUDA
4. Random inputs are critical — predictable data inflates throughput by 10-20% (power throttling)

**Our context:** We're on Apple Metal via CubeCL, not NVIDIA Hopper. Peak throughput is ~100× lower. But the *relative* gains from tiling should be similar: avoiding full-score materialization and improving cache locality.

---

## 4. Distillations for Our Stack

### 4.1 CPU SIMD Path (katgpt-core)

**What we can adopt:**

1. **Tiled online softmax attention** — Process Q in rows of SIMD_WIDTH, K/V in tiles of `SIMD_WIDTH × head_dim`. Avoid materializing `N×N` score matrix. This is a pure algorithmic improvement.

2. **Register-blocking for output accumulation** — Our `simd_dot_f32` already accumulates into SIMD registers. The online softmax pattern keeps `O`, `max`, `norm` in registers across tiles, avoiding intermediate stores.

3. **Temperature scaling with exp2 trick** — TK precomputes `temperature_scale = rsqrt(D) * 1.44269504089f` (log2(e)). This lets them use `exp2()` instead of `exp()`, which is faster on most hardware. Our `softmax_scaled()` can adopt this.

**What we should NOT adopt:**
- SMEM swizzling patterns (no shared memory banks on CPU)
- TMA-based async loads (no DMA engine on CPU)
- Warp-specialized producer/consumer split (CPU is single-threaded per core; we use Rayon for parallelism, not warp cooperation)

**Estimated impact:** For `N > 1024` attention with our SIMD kernels, tiled attention should reduce peak memory from `O(N²)` to `O(Br × Bc)` per head. With `Br = 8, Bc = 128`, that's `8 × 128 × 4B = 4KB` per tile vs `N × N × 4B` for full materialization.

### 4.2 GPU CubeCL Path (riir-ai, Plan 106)

**What we can adopt:**

1. **LCF pipeline structure** — Our CubeCL attention kernel (`attention_cubecl.rs`) already does online softmax with tiled KV scan. The TK structure shows how to add SMEM double-buffering for K/V tiles (currently we do single-buffer).

2. **Super-grouping for L2 locality** — TK's CTA traversal order (group nearby output tiles to share operand loads) applies to any GPU. For Metal, this means grouping workgroups that share K/V tiles to improve L2 cache hit rate.

3. **Persistent kernel pattern** — TK uses persistent blocks that pull multiple tasks. Plan 106 could adopt this for batched decode: one workgroup processes multiple (query, head) pairs without re-dispatch.

**What we should NOT adopt:**
- WGMMA descriptors (NVIDIA-specific)
- TMEM allocation/deallocation (Blackwell-specific)
- TMA multicast (NVIDIA-specific)
- CLC scheduling (Blackwell-specific)
- 2×SM MMA (Blackwell-specific)

**CubeCL vs TK positioning:** CubeCL provides a similar abstraction level to TK — tiled matmul pipeline, shared memory, subgroup operations. The difference is CubeCL targets Metal/CUDA/ROCm portably, while TK targets NVIDIA Hopper/Blackwell specifically. For our cross-platform needs, CubeCL is the right choice (already selected in Plan 106).

### 4.3 Tile Scheduling Insights

TK's three scheduling strategies provide a useful mental model:

| Strategy | Description | Our Analog |
|:---|:---|:---|
| Single-tile | One CTA per output tile | Current: one workgroup per (batch, head) pair |
| Static persistent | Fixed workers pull multiple tiles | Plan 106 potential: batch multiple queries per dispatch |
| CLC (dynamic) | Hardware work-stealing | Not available on Metal; use CPU-side work scheduling instead |

The **super-grouping** traversal (group `SUPER_M` rows, traverse columns within group) is the most applicable insight: it improves L2 reuse of K/V tiles by having nearby workgroups process queries that share the same K/V data. On Metal, this means ordering our workgroup dispatch to maximize L2 cache hits.

---

## 5. Feature Gate Proposal

### 5.1 katgpt-core: `tiled_attention`

```toml
[features]
tiled_attention = []  # Tiled online-softmax flash attention for CPU SIMD
```

**Scope:**
- Tiled attention function in `katgpt-core/src/attention.rs` (new file)
- Processes Q in SIMD-width row tiles, K/V in column tiles
- Online softmax with exp2 trick
- Falls back to current `softmax_scaled()` for small N where tiling overhead dominates

**GOAT proof:** Compare attention output cosine similarity vs current full-materialization path for random weights. Target: >0.999 cosine similarity, with memory reduction from O(N²) to O(Br × Bc) per head.

### 5.2 riir-ai: Already covered by `cubecl_runtime` feature gate

The GPU tiled attention work goes into Plan 106 Track 2 (T2.5+ attention kernel). No new feature gate needed — it's part of the CubeCL migration.

---

## 6. What NOT to Distill

Being honest about the limitations:

1. **WGMMA/TMA/TMEM are NVIDIA-specific.** These are the hardware mechanisms that make TK fast on Hopper/Blackwell. They have no equivalent on Metal, Vulkan, or CPU. Attempting to "port" them would be cargo-cult engineering.

2. **16-based tile granularity is hardware-derived.** On CPU, our natural tile is SIMD width (4 or 8). On Metal, it's `simdgroup_matrix` size (8×8×8). Blindly copying TK's 16-based tiling would be suboptimal.

3. **SMEM swizzling is bank-conflict-specific.** CPU doesn't have banked shared memory. Metal's shared memory has different access patterns. This is a detail to handle at the CubeCL/kernel level, not in our shared algorithm code.

4. **TK's attention kernel is NOT the paper's kernel.** The benchmark repo uses a simpler LCF implementation. The paper reports matching FA3; the benchmark repo trails FA3 by 5-15%. The LCF template is a general-purpose pipeline, not the hand-tuned kernel from the paper.

---

## 7. Relationship to Existing Research

| Research | Relationship |
|:---|:---|
| R66 (TileRT) | TileRT = persistent pipeline at inference engine level. TK = tile-level kernel primitives. Complementary: TK shows *how* to tile, TileRT shows *when* to pipeline. |
| R67 (CODA) | CODA = GEMM-epilogue fusion (delay RMSNorm past matmul). TK = tile-structured kernel framework. CODA's epilogue visitor pattern fits naturally into TK's `finish` hook. |
| R29 (Rust GPU) | Assessed rust-gpu (Rust→SPIR-V). Conclusion: not ready. CubeCL won (Plan 106). TK validates CubeCL's abstraction level — both are "tile DSLs above raw GPU code." |
| R71 (DashAttention) | DashAttention = adaptive sparse block selection. TK = dense tile computation. DashAttention decides *which* blocks to attend to; TK shows *how* to attend to each block efficiently. |
| R22 (Lighthouse) | Lighthouse = hierarchical pyramid selection. Same relationship as DashAttention — selects blocks, then runs dense attention. TK optimizes the dense attention part. |

---

## 8. Concrete Next Steps

### For katgpt-rs (CPU SIMD):
1. Add `tiled_attention` feature gate to `katgpt-core/Cargo.toml`
2. Implement tiled online-softmax attention in `katgpt-core/src/attention.rs`
3. Benchmark: compare throughput and memory usage vs current full-materialization path
4. GOAT proof: cosine similarity > 0.999 vs reference SDPA

### For riir-ai (GPU CubeCL):
1. Guide Plan 106 T2.5+: use LCF pipeline structure for CubeCL attention kernel
2. Add SMEM double-buffering for K/V tiles (currently single-buffer)
3. Evaluate super-grouping for L2 locality in batched attention dispatch
4. No new feature gate — covered by existing `cubecl_runtime`

### NOT doing:
- Porting WGMMA/TMA/TMEM abstractions (NVIDIA-specific)
- Porting SMEM swizzling (hardware-specific)
- Porting CLC scheduling (Blackwell-specific)
- Adding CUDA dependency to our stack

---

## 9. References

- [ThunderKittens GitHub](https://github.com/HazyResearch/ThunderKittens)
- [ThunderKittens Paper](https://arxiv.org/abs/2410.20399)
- [TK Attention Benchmark Repo](https://github.com/HamzaElshafie/tk_attention) (`.raw/tk_attention/`)
- [Dissecting ThunderKittens Blog Post](https://hamzaelshafie.bearblog.dev/dissecting-thunderkittens-anatomy-of-a-compact-dsl-for-high-performance-ai-kernels/)
- [GPUs Go Brrr (Hazy Research Blog)](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- [ThunderKittens on Blackwell](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell)
- [FlashAttention-3 Paper](https://arxiv.org/pdf/2407.08608)
- [SemiAnalysis: Dissecting Nvidia Blackwell Tensor](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor)
- [Modular: Matrix Multiplication on Blackwell](https://www.modular.com/matrix-multiplication-on-blackwell)
- [Predictable Data Power Throttling](https://www.thonking.ai/p/strangely-matrix-multiplications)