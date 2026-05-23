# Research 67: CODA — GEMM-Epilogue Programming for Transformer Fusion

> **Source:** [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/pdf/2605.19269) — Guo, Zhang, Menon, Guessous, Thakkar, Kim, Dao (MIT, Princeton, Together AI, Meta)
> **Code:** [github.com/HanGuo97/coda-kernels](https://github.com/HanGuo97/coda-kernels) (CUTLASS CuTeDSL, NVIDIA Hopper H100)
> **Local:** `.raw/coda-kernels/` (upstream Python/CuTeDSL)
> **Date:** 2026-05-20, distilled 2026-05
> **Related:** Research 29 (Rust GPU), Research 55 (Tri-Mode), Research 59 (MoE+SD), Research 66 (TileRT), Research 39 (SpectralQuant)
> **Verdict: ALGEBRAIC REPARAMETERIZATION — Three distillations: (1) delay RMSNorm past next GEMM (algebraic identity, zero-cost), (2) fused matmul+residual+rmsnorm+activation CPU SIMD kernels for microgpt-rs, (3) epilogue visitor pattern for riir-ai CubeCL rewrite. The GPU epilogue fusion (CODA's primary contribution) is hardware-specific to NVIDIA Hopper TMA/WGMMA and NOT directly portable to wgpu/Metal. The algebraic reparameterization IS portable to both CPU SIMD and GPU. Feature gate: `coda_fusion` in microgpt-rs for fused CPU kernels; guide Plan 106 (CubeCL) for GPU epilogue patterns.**

---

## Executive Summary

CODA reparameterizes Transformer computation as GEMM-plus-epilogue programs. The key insight: many separate memory-bound operations (normalization, activations, residual adds, reductions) can be algebraically rearranged to execute *while the GEMM output tile is still on-chip*, avoiding global memory round-trips.

**The punchline for us:** We don't have Hopper Tensor Cores or TMA. But the *algebraic reparameterization* is hardware-independent. Specifically:

1. **CPU (microgpt-rs):** Our `forward_base()` does `matmul → write output → rmsnorm → write output → matmul → write output → residual add → write output`. Each `write output` is a full-buffer store. CODA shows we can fuse matmul+residual+rmsnorm into a single SIMD pass, eliminating 2-3 intermediate buffer writes per layer.

2. **GPU (riir-ai):** Plan 106 (CubeCL rewrite) gets CODA's epilogue visitor pattern — composable primitives that plug into the tiled matmul pipeline. This is the *architectural* contribution, not just the math.

**What CODA does NOT change:** Our model-based/modelless duality, speculative decoding, pruners, or any of the reasoning layer. CODA is a pure execution optimization.

---

## 1. Key Ideas from CODA

### 1.1 GEMM-Residual-RMSNorm-GEMM Pattern

The core reparameterization. In a standard Transformer, the sequence is:

```
h0 = x @ W0          # GEMM (attention out-proj)
h1 = h0 + residual   # separate kernel / buffer write
h2 = RMSNorm(h1)     # separate kernel / buffer write
y  = h2 @ W1         # GEMM (MLP gate+up proj)
```

CODA observes that `RMSNorm(h1) = r * h1 * gamma` where `r` is a row-wise scalar. Since `r` is shared across the row, it commutes with the next GEMM:

```
y = (r * h1 * gamma) @ W1 = r * (h1 * gamma @ W1)
```

This means `r` doesn't need to be applied *before* W1. The computation splits into:

```
GEMM 1:   h0 = x @ W0
Epilogue 1: D = h0 + residual, S = partialRMS(D), O = D * gamma
Reduce:   r = 1 / sqrt(sum(S) + eps)     # lightweight reduction over tile partials
GEMM 2:   h3 = O @ W1
Epilogue 2: y = r * h3
```

**Why this matters:** The standalone RMSNorm kernel is eliminated. Instead, residual add + partial reduction + norm-weight scaling happen *inside the matmul output loop*. The only extra work is a tiny reduction over partial statistics.

### 1.2 Pairwise Activations (SwiGLU, RoPE)

SwiGLU splits the GEMM output into gate/up streams and applies `silu(gate) * up`. CODA arranges paired features to be adjacent in the accumulator layout, so the activation happens at register level without materializing the expanded intermediate.

Similarly, RoPE rotates feature pairs — register-level computation on adjacent values.

### 1.3 Epilogue Visitor Tree

CODA's architectural contribution: a composable epilogue interface:

```
consumer_begin()          # load per-tile inputs (gmem → smem)
consumer_begin_loop()     # per sub-tile: smem → registers
consumer_visit(rD)        # MUTATE accumulator tile — the core op
consumer_smem_store()     # extra smem writes (partial reductions)
consumer_tma_store()      # post-store callback
consumer_end()            # finalization
```

Each "visitor" is a primitive: residual add, row scaling, reduction, pairwise activation, matrix load. Visitors compose via `EVTList`.

### 1.4 Backward Pass Structure

Theorem: tile-local epilogues in the forward pass induce tile-local epilogues in the backward pass. The direction flips (forward epilogue attaches to the GEMM producing its input; backward epilogue attaches to the GEMM producing its output's gradient), but the structure is identical.

RMSNorm backward is the one exception — it needs a row-wise statistic. CODA moves this statistic to a GEMM boundary where both activation and gradient are already available.

---

## 2. Applicability to Our Stack

### 2.1 CPU (microgpt-rs) — SIMD Fusion

Our current `forward_base()` per layer:

```rust
// Pre-attention
rmsnorm(&mut ctx.x);                          // pass 1: sum_sq, pass 2: scale
ctx.xr[..n].copy_from_slice(&ctx.x[..n]);     // buffer write
rmsnorm(&mut ctx.x);                          // pass 1: sum_sq, pass 2: scale

// QKV + attention + out-proj
matmul(&mut ctx.q, &wq, &ctx.x, n, n);       // write ctx.q
matmul(&mut ctx.k, &wk, &ctx.x, kvd, n);     // write ctx.k
matmul(&mut ctx.v, &wv, &ctx.x, kvd, n);     // write ctx.v
// ... attention ...
matmul(&mut ctx.x, &wo, &attn_out, n, n);    // write ctx.x
simd_add_inplace(&mut ctx.x, &ctx.xr);        // read-modify-write ctx.x

// MLP
ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);    // buffer write
rmsnorm(&mut ctx.x);                           // pass 1: sum_sq, pass 2: scale
matmul_relu(&mut ctx.hidden, &w1, &ctx.x, mlp_hidden, n);  // write ctx.hidden
matmul(&mut ctx.x, &w2, &ctx.hidden, n, mlp_hidden);       // write ctx.x
simd_add_inplace(&mut ctx.x, &ctx.xr2);        // read-modify-write ctx.x
```

**Buffer writes per layer:** ~8-10 full-dimension writes (ctx.x, ctx.q, ctx.k, ctx.v, ctx.hidden, ctx.xr, ctx.xr2, attn_out).

**CODA-inspired fusion opportunities:**

| Fused Kernel | Eliminates | Savings |
|:---|:---|:---|
| `simd_matmul_residual_rmsnorm` | out_proj write → residual add → rmsnorm | 2 buffer passes |
| `simd_matmul_rmsnorm_swiglu` | gate+up matmul → rms scale → SwiGLU split | 1 buffer pass + expanded intermediate |
| `simd_matmul_residual` | down_proj → residual add | 1 buffer pass |

The algebraic trick: for the out_proj + residual + rmsnorm case:
```
D = attn_out @ Wo + xr       (fused: matmul each row, add residual, accumulate partial rms)
r = 1/sqrt(reduce(partial_rms) + eps)  (tiny reduction over block_size partials)
O = D * gamma                (scale by norm weight)
```

Then for MLP:
```
GEMV: h = O @ W_gate_up * r   (apply rms scale IN the matmul inner loop)
SwiGLU: gate, up = split(h); output = silu(gate) * up  (in-place on h)
GEMV: x = output @ W_down + xr2  (fused residual add)
```

**Key constraint:** CPU SIMD doesn't have registers-on-chip like GPU. But for BS=1 decode, the vectors are small (n_embd=64-512 for micro configs). The entire vector fits in L1 cache. The win is eliminating *function call overhead and buffer writes*, not global memory traffic.

### 2.2 GPU (riir-ai) — CubeCL Epilogue Patterns

Plan 106 (CubeCL rewrite) can directly adopt CODA's epilogue visitor pattern:

```
CubeCL Matmul Pipeline
  ├─ Tile Matmul (simdgroup_matrix 8×8×8)
  └─ Epilogue (CODA-style visitors)
      ├─ ResidualAdd: D = acc + C
      ├─ PartialRMSReduction: S[m, nb] = mean(D[m, nb*bs:(nb+1)*bs]^2)
      ├─ NormWeightScale: O = D * gamma
      ├─ RowScale: D = D * r  (delayed RMSNorm)
      ├─ SwiGLU: gate, up = split(D); O = silu(gate) * up
      └─ RoPE: rotate adjacent pairs with cos/sin tables
```

**Important caveat:** CODA's TMA (Tensor Memory Accelerator) and WGMMA (Warp Group Matrix Multiply Accumulate) are Hopper-specific. On Metal (Apple Silicon), the analog is `simdgroup_matrix` 8×8 and shared memory double-buffering. The *pattern* translates; the *primitives* need Metal-specific implementation.

### 2.3 What Does NOT Apply

| CODA Feature | Why Not Applicable |
|:---|:---|
| TMA async loads | NVIDIA Hopper only; Metal uses shared memory barriers |
| WGMMA ping-pong | NVIDIA Hopper only; Metal uses simdgroup_matrix |
| FP8 accumulator | Our CPU path is f32; GPU path targets f16 |
| LLM-authored kernels | Our kernel surface is small; manual is fine |
| CuTeDSL layout system | CubeCL has its own layout abstractions |
| Cross-entropy fusion | We do greedy decode, not training loss computation |

---

## 3. Distillation Targets

### D1: Algebraic Reparameterization (CPU, microgpt-rs)

The key identity: `RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'`

This lets us delay the row-wise scale `r` past the next GEMM. On CPU SIMD for BS=1 decode:
- Fuse `matmul + residual_add + partial_rms + norm_weight_scale` into one loop over weight rows
- Fuse `matmul + row_scale + SwiGLU` into one loop over weight rows
- Fuse `matmul + residual_add` into one loop over weight rows

**Feature gate:** `coda_fusion` in microgpt-rs

### D2: Epilogue Visitor Pattern (GPU, riir-ai)

Composable epilogue primitives for CubeCL kernels. Not a direct port — adapt the *pattern* (consumer_begin → consumer_visit → consumer_smem_store) to CubeCL's kernel compilation model.

**No separate feature gate** — this feeds into Plan 106's CubeCL task structure.

### D3: Backward Pass Reparameterization (Future)

CODA shows the same GEMM-epilogue structure applies in backward. This matters for riir-ai's GPU LoRA training path. **Defer** until GPU training is production-critical.

---

## 4. Numerical Considerations

CODA's reparameterization changes when RMSNorm scaling is applied. The paper shows this can actually *improve* numerical accuracy (Figure 6):

- Standard path: `FP32_rmsnorm(BF16_matmul_output)` → truncation before norm
- CODA path: `FP32_accumulator → epilogue_fusion → BF16_store` → higher precision maintained longer

For our CPU f32 path, this is moot — everything is f32 throughout. But for the GPU f16 path (Plan 106 T1.3), the delayed normalization could improve accuracy.

---

## 5. Related Work Context

| Paper/System | Relation to CODA |
|:---|:---|
| FlashAttention | Fuses attention score+softmax+weighted_value; CODA fuses everything *except* attention |
| Liger Kernels | Triton-based fused kernels; CODA uses CuTeDSL epilogue visitors |
| TileRT (Research 66) | Persistent GPU pipeline; CODA is per-kernel fusion. Complementary. |
| QuACK | CODA's GEMM template is built on QuACK's high-accuracy mainloop |
| ThunderKittens | Tile-level GPU programming; similar abstraction level to CODA's epilogue |

---

## 6. Verdict Summary

| Dimension | Assessment |
|:---|:---|
| Algebraic reparameterization | ✅ **Directly applicable** — delay RMSNorm past GEMM is a free algebraic identity |
| CPU SIMD fusion | ✅ **High value** — eliminates buffer writes at BS=1 where overhead dominates |
| GPU epilogue pattern | ⚠️ **Conceptual** — guides CubeCL design but needs Metal-specific primitives |
| GPU TMA/WGMMA fusion | ❌ **Not portable** — Hopper-specific, no Metal analog |
| Backward pass | ⏳ **Defer** — matters for GPU training, not current priority |
| LLM kernel authoring | ❌ **Not needed** — our kernel surface is tiny vs. CODA's full Transformer |

**Bottom line:** CODA's algebraic trick is the prize. The GPU epilogue architecture is a bonus for Plan 106. Everything else (TMA, CuTeDSL, LLM authoring) is Hopper-specific and not relevant to our Metal/CPU stack.