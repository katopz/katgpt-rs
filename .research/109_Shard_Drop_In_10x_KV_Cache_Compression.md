# Research 109: Shard — Drop-In 10× KV Cache Compression via Asymmetric K/V Methods

**Source:** [Shard](https://krishgarg.com/shard) by Krish Garg, Kirrithan Sathananthan (May 2026)
**Code:** `.raw/shard/` (Python, PyTorch/Triton)
**Related:** Research 20 (TurboQuant), Research 39 (SpectralQuant), Research 63 (OCTOPUS), Research 65 (RotorQuant), Research 81 (Asymmetric K/V), Plan 123 (Asymmetric KV benchmarks)
**Date:** 2026-05-26

> **Verdict: HIGH VALUE — Asymmetric codec (not just asymmetric bit widths) is the missing piece. PCA-K + VQ-V achieves 10× compression, doubling our best result. Fused compressed attention is a novel capability we don't have.**

## TL;DR

Shard achieves **10× compression** at 8K context and **11.2× at 32K** on Llama-3.1-8B-Instruct with NIAH 1.000 and WikiText-2 PPL +0.26%. The key insight over our existing stack: **K and V need different compression *methods*, not just different bit widths.** Keys get PCA on no-RoPE basis (exploiting rank-192 out of 1024 structure), Values get Hadamard + k-means VQ (exploiting rotation-induced Gaussianity). Decode streaming uses data-oblivious Lloyd-Max (8-bit, bit-exact lossless). Fused attention computes Q·K directly on int4 PCA coefficients via a per-pair relative-Δ RoPE identity — no FP16 K reconstruction ever happens.

## Core Innovation: Asymmetric Codec Design

### Why Shard Beats Our Stack

| Aspect | Our Current Stack | Shard | Gap |
|--------|------------------|-------|-----|
| K/V treatment | Same codec, different bit widths | **Different codecs** entirely | Fundamental |
| K compression | OCT triplet or SQ eigenbasis (on raw K) | PCA on **no-RoPE** K basis | RoPE removal reveals rank-192 structure |
| V compression | Same as K (OCT triplet or SQ) | Hadamard + k-means VQ | Exploits rotation-induced Gaussianity |
| Decode streaming | Lloyd-Max (same as TQ) | Lloyd-Max (same) | No difference |
| Fused attention | Dequant K→FP16→matmul | Q·K on int4 PCA coefficients directly | **Novel — we don't have this** |
| Compression | 5.3× (TQ), ~9.7× (SQ) | **10.0–11.2×** | 2× over TQ, matches SQ but without calibration |
| Calibration | SQ needs 256 samples | **Zero calibration** (data-dependent PCA on prefill only) | Advantage for cold start |
| Decode throughput | ~1× FP16 | 0.4–0.5× FP16 | **We're faster** — important trade-off |

### The RoPE-Removal Insight

Shard's single most important finding (from Section 3 of the blog):

> Llama's K matrix is effectively rank-192 out of 1024 once you undo the rotation. RoPE was hiding the structure.

This means: `W_K` has limited effective rank. K activations live on a low-dimensional subspace. But RoPE rotates each token's K by a different angle, spreading energy across all dimensions. Undo RoPE → subspace reemerges → PCA becomes extremely effective.

Our SpectralQuant (Research 39) computes eigenbasis on the *raw* K (with RoPE applied). It works because variance concentration still exists under RoPE, but it's fighting the rotation. Shard removes the rotation first, revealing dramatically more structure.

**KVTC** (Staniszewski & Łańcucki, ICLR 2026) independently discovered the same insight: "Remove rotary position embeddings from keys before PCA."

### K Path: PCA on No-RoPE K

1. **Undo RoPE** per token: `k_no_rope = k ⊙ cos(θp) - rotate_half(k) ⊙ sin(θp)`
2. **Per-layer SVD**: `torch.svd_lowrank(centered, q=192, niter=24)` — niter=24 critical for stability
3. **DP bit allocation**: Groups of 64 components, bits ∈ {0,2,4,6,8}, with **4× drop penalty** on zero-bit option
4. **Quantize basis in int8**, coefficients in **symmetric int4** [-7, 7]

Per-token K storage at rank=192, 8K tokens: **~0.75 bits per element**.

### V Path: Hadamard + K-Means VQ

1. **Hadamard rotation**: Decorrelates channels → each looks independently Gaussian
2. **K-means VQ** on groups of 4 channels, 256-entry codebook
3. VQ captures joint structure that scalar quant misses

Per-token V storage: **2 bits per element** (256 bytes per layer for whole V).

### The 4× Drop Penalty

From "Quantization Dominates Rank Reduction" (arXiv:2604.11501): under softmax attention's Fisher metric, **dropping a direction is quadratically worse** than scalar-quantizing it. A deleted direction that determined routing causes a categorical, non-recoverable error. Scalar quant only adds noise.

```
err = gv * 4.0 if bits == 0 else gv / (3.0 * (1 << (2 * bits)))
```

One constant. NIAH went from 0.92 → 1.000.

### Attention Sinks + Recency Window

- **4 FP16 sink tokens**: First tokens receive disproportionate attention. Lossy compression = garbage.
- **64-token FP16 window**: Recent tokens critical for next-token prediction.
- Overhead: 68 fp16 tokens / 8192 total < 1%.

Layout per layer: `[4 sink fp16][middle: PCA(K) + VQ(V)][64 window fp16]`

### Fused Compressed Attention (Novel)

The key derivation: for rotate-half RoPE, the attention score between Q at position `p_q` and K at position `p_t` depends only on `Δ = p_t - p_q` and the no-RoPE vectors. This means:

1. Store K as int4 PCA coefficients in no-RoPE basis
2. Precompute `Q·B` (query projected to PCA basis) once per decode step
3. Compute attention scores as rank-192 inner products against int4 coefficients
4. Mix with per-pair `cos(θΔ)`, `sin(θΔ)` — precomputed from relative positions
5. **No FP16 K ever materializes**

Verified: max abs diff 0.0023, mean 0.0004 vs FP16 reference. Within fp16 tolerance.

### Fused V: Hadamard Past the Sum

Since Hadamard is linear: `Σ w_t · H(V_rot_t) = H(Σ w_t · V_rot_t)`. Apply one Hadamard at the end, not per token. Codebook lookup + broadcast multiply in rotated space is cheap.

## Experimental Results

### Compression by Context Length

| Context | Compression | Stored Cache |
|---------|-------------|-------------|
| 4K | 8.8× | 58 MB |
| 8K | **10.0×** | 102 MB |
| 16K | 10.8× | 190 MB |
| 32K | **11.2×** | 366 MB |

### Quality

| Metric | FP16 | Shard | Δ |
|--------|------|-------|---|
| NIAH recall (4K–32K, 20 needles) | 1.000 | **1.000** | 0 |
| LongBench-E avg | 16.24 | 16.19 | −0.05 |
| WikiText-2 PPL | 6.45 | 6.47 | **+0.26%** |
| 8-bit streaming match | — | **750/750** | Bit-exact lossless |
| Decode throughput | 29.8 tok/s | 11.6 tok/s | **0.39×** (memory win, not speed win) |

### Vs TurboQuant (Same Model)

| Metric | TurboQuant | Shard |
|--------|-----------|-------|
| Max compression | 4–6× | **10–11×** |
| NIAH | 0.997 | **1.000** |
| Data-oblivious | Fully | Prefill PCA, decode TQ |

## What Maps to Our System

### Directly Applicable

| Shard Concept | Our Equivalent | Gap |
|---------------|---------------|-----|
| **RoPE undo before compression** | Not implemented | **Critical missing piece** |
| **Asymmetric codec (PCA-K, VQ-V)** | Same codec for K and V | **Different methods, not just bit widths** |
| **DP bit allocation with drop penalty** | Water-fill in SpectralQuant | Similar concept, different implementation |
| **Fused int4 attention** | Dequant→FP16→matmul | **Novel capability** |
| **Hadamard past weighted sum** | Not implemented | **Linear algebra optimization** |
| **Attention sink protection** | Not implemented | **Critical for quality** |
| **Recency window** | Not implemented | Standard game technique |
| **Lloyd-Max decode streaming** | Already in TurboQuant | **No gap** |
| **Lazy per-layer eviction** | Not implemented | Implementation detail |

### Already in Our Stack (Reuse)

| Component | Where | Status |
|-----------|-------|--------|
| Lloyd-Max codebook | `turboquant/codebook.rs` | ✅ Direct reuse |
| Random rotation | `turboquant/rotation.rs` | ✅ Direct reuse |
| KV cache trait | `QuantizedKVCache` trait | ✅ Extend for asymmetric codec |
| Bit packing | `turboquant/kv_cache.rs` | ✅ Reuse for int4 coefficients |
| SVD/eigenbasis | `spectralquant/spectral.rs` | ✅ Adapt for no-RoPE input |

### What Does NOT Map

| Shard Concept | Why Not |
|---------------|---------|
| **HuggingFace Cache API** | We have our own `MultiLayerKVCache` trait |
| **Triton int4 matmul kernel** | We use wgpu (WGSL), not CUDA Triton |
| **Monkey-patching Llama attention** | Our model has different architecture |
| **torch.svd_lowrank** | Need Rust eigenvalue decomposition (we have this in `spectralquant/`) |
| **Python k-means** | Need Rust k-means or alternative VQ |
| **B200 GPU benchmarks** | Our target is CPU SIMD + Metal/wgpu |

## Comparison: Shard vs Our KV Compression Stack

| Aspect | TurboQuant | SpectralQuant | OCTOPUS/Hybrid | PlanarQuant | **Shard (Proposed)** |
|--------|-----------|---------------|----------------|-------------|---------------------|
| Compression | 5.3× | 9.7× | ~7× | ~5× | **10–11×** |
| Calibration | None | 256 samples | None | None | None (prefill PCA) |
| K/V symmetry | Same codec | Same codec | Same codec | Same codec | **Asymmetric codecs** |
| RoPE-aware | No | No | No | No | **Yes (undo before PCA)** |
| Fused attention | No | No | No | No | **Yes (int4 Q·K)** |
| Decode speed | ~1× FP16 | ~1× FP16 | ~1× FP16 | ~1× FP16 | **0.4× FP16** |
| Sink protection | No | No | No | No | **Yes (4 sink + 64 window)** |
| Data-oblivious | Fully | Calibration | Fully | Fully | Prefill PCA + TQ decode |

## Key Insights for Our Architecture

### 1. RoPE-Removal Is the Missing Piece (Critical)

Our SpectralQuant computes eigenbasis on raw K (with RoPE). Shard shows that undoing RoPE first reveals dramatically more structure (rank-192 out of 1024). This is a **drop-in enhancement** to our existing SpectralQuant pipeline — add RoPE undo before eigendecomposition.

### 2. Asymmetric Codec > Asymmetric Bit Widths (Deepens Research 81)

Research 81 proved V compression is "free" while K precision is critical. Shard goes deeper: it's not just that K needs *more bits*, K needs a *different method* (PCA) than V (VQ). Our current stack applies the same codec to both. The 2× compression gain over our best result comes from this architectural asymmetry.

### 3. Fused Compressed Attention Is a Novel Capability

Our `forward_quantized` dequantizes K to FP16 before attention. Shard's per-pair relative-Δ RoPE identity allows attention directly on int4 coefficients. For GPU paths (riir-gpu), this could be a major memory-bandwidth win at long contexts.

### 4. Attention Sinks Are Non-Negotiable

Shard went from NIAH 0.3 → 1.000 by adding 4 sink tokens + 64 recency window. Any aggressive KV compression that doesn't protect sinks will fail catastrophically.

### 5. Decode Throughput Is the Trade-Off

Shard is 0.4× FP16 decode speed. Our stack is ~1×. The 10× memory win comes at a 2.5× speed cost. For our architecture (CPU SIMD + batch inference), this trade-off profile may differ. Worth benchmarking.

### 6. K-Means VQ for Values

Our OCTOPUS uses octahedral triplet encoding for V. Shard uses k-means VQ on Hadamard-rotated groups of 4 channels. Both are data-dependent (during prefill). K-means VQ may be simpler to implement in Rust than octahedral maps, but OCTOPUS's joint 3×3 rounding is a quality advantage.

## Strategic Assessment

### Domain Placement

Per the MMO GOAT Pillars Decision Matrix (`.docs/27`):

- **KV cache compression is NOT a GOAT pillar** — it's infrastructure that supports all pillars
- Shard's techniques are **LoRA-independent** (pure algorithmic) → good for secondary moat
- **Not game-specific** → stays in katgpt-rs domain
- **Fused compressed attention on GPU** → could enhance riir-ai's riir-gpu but is NOT game-specific, so it stays in katgpt-rs

### Feature Gate Assessment

Shard introduces one new concept (asymmetric codec) that should be feature-gated:

- `shard_kv` — Shard asymmetric KV compression: PCA on no-RoPE K + Hadamard+VQ on V + fused int4 attention
- This is a **model-based inference** feature, not game-related
- If fused attention on GPU proves to be a competitive advantage for real-time game AI inference, the GPU kernel stays in riir-ai with `shard_fused` feature gate (secret, not public)

### What Should Stay Secret

Per the decision matrix, game-specific tuning is private. Shard itself is public research. The **secret sauce** would be:
- Domain-specific PCA rank tuning (which rank per game context length)
- Per-game attention sink token selection (which tokens are "sink" in game traces)
- GPU fused attention kernel implementation details (if we build wgpu version)

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RoPE undo not applicable to our model | Low | High | Our model uses same rotate-half RoPE |
| PCA overhead during prefill | Medium | Medium | Only runs once per prompt, amortized |
| K-means VQ training instability | Medium | Low | Fall back to OCTOPUS encoding for V |
| Decode throughput regression | High | Medium | Feature-gate; only use for memory-bound workloads |
| niter=24 SVD convergence on CPU | Medium | Medium | Our `spectralquant/` already handles this |
| Drop penalty constant tuning | Low | Low | Paper validated on Llama; may need per-model tuning |

## References

- Shard blog: https://krishgarg.com/shard
- Shard code: https://github.com/krish1905/shard
- TurboQuant: Zandieh et al., ICLR 2026 (our Research 20)
- KVTC: Staniszewski & Łańcucki, ICLR 2026, arXiv:2511.01815
- "Quantization Dominates Rank Reduction": arXiv:2604.11501
- StreamingLLM / Attention Sinks: Xiao et al., ICLR 2024, arXiv:2309.17453
- AsymKV: Tao et al., arXiv:2410.13212
- "Homogeneous Keys, Heterogeneous Values": NeurIPS 2025, arXiv:2506.05410
- QJL: Zandieh et al., AAAI 2025, arXiv:2406.03482
- RoFormer / RoPE: Su et al., arXiv:2104.09864
- Lloyd-Max: Lloyd 1982, Max 1960 (our existing `codebook.rs`)
