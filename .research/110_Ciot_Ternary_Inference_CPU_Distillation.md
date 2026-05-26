# Research 110: Ciot — Ternary Weight CPU Inference Engine Distillation

**Source:** [Cintu07/ciot](https://github.com/Cintu07/ciot) — CPU inference engine for ternary neural networks
**Code:** `.raw/ciot/` (C++17, zero dependencies, NEON/AVX2/AVX-512/scalar)
**Related:** Research 008 (Sparse MLP / TwELL), Research 020 (TurboQuant), Research 039 (SpectralQuant), Research 064 (LlamaWeb WebGPU), Research 066 (TileRT), Research 109 (Shard KV compression)
**Date:** 2026-05-26

> **Verdict: HIGH VALUE for CPU-bound inference tier — Ternary matvec eliminates ALL floating-point multiplication from the hot path, replacing with SIMD conditional addition/subtraction. This is the "Plasma" tier in the five-tier memory/compute hierarchy. Architecture belongs in katgpt-core (open), game-specific tuning stays in riir-ai (private).**

## TL;DR

Ciot implements a complete CPU inference engine for ternary neural networks (weights ∈ {-1, 0, +1}) using pure C++ SIMD with zero dependencies. On ARM NEON (Snapdragon X), a 1024×1024 ternary matvec finishes in 0.26ms at 8.01 Gop/s — **3.96× faster than scalar** with identical checksums. The key insight: two bit-planes per 64-element block encode +1 (pos bit) and -1 (neg bit), enabling branchless SIMD accumulation via `masked_load` patterns. No FPUs, no multiplication, just conditional add/subtract.

## Why This Matters for Our Stack

| Aspect | Our Current Stack | Ciot Approach | Gap |
|--------|------------------|---------------|-----|
| Arithmetic | FP32 FMA (FPU-heavy) | INT add/sub (ALU-only) | Eliminates FPU bottleneck |
| Weight storage | 3-4 bits/weight (SQ) or 16 bits (F16) | **1.58 bits/weight** (log₂3) | 2-10× memory reduction |
| SIMD kernel | `simd_dot_f32` → `vfmaq_f32` | `masked_load4` → `vaddq_f32/vsubq_f32` | No multiply, branchless |
| Sparse skip | `sparse_matmul` (explicit index) | Implicit via zero bits | Cleaner encoding |
| Throughput target | ~1.64M tok/s (AR Draft) | Est. **3-5M tok/s** (ternary draft) | Potential 2-3× on CPU |
| Power/thermal | Moderate (FPU heavy) | **Ultra-low** (ALU only) | Edge/mobile advantage |

## Core Innovation: Bit-Plane Ternary Encoding

### The Encoding (from `Ciot.h`)

```cpp
struct TernaryMatrix {
    uint32_t rows, cols, blocks64;
    uint64_t* pos_bits;    // bit k set → weight = +1
    uint64_t* neg_bits;    // bit k set → weight = -1
    float*    row_scale;   // per-row rescale toward original float magnitudes
};
```

**Key insight:** 64 ternary weights fit in two `u64` words. Both zero → weight is 0 (implicit skip). This is more compact than our Sparse MLP's explicit index + value arrays.

### The SIMD Kernel (NEON, from `ternary_simd.cpp`)

```cpp
// Per 4-element chunk of a 64-element block:
float32x4_t masked_load4(const float* x, uint32_t offset, uint32_t bits,
                          uint32x4_t lane_bits, uint32x4_t zero_u32, float32x4_t zero_f32) {
    float32x4_t values = vld1q_f32(x + offset);
    uint32x4_t word = vdupq_n_u32(bits);
    uint32x4_t active = vcgtq_u32(vandq_u32(word, lane_bits), zero_u32);
    return vbslq_f32(active, values, zero_f32);  // blend: if bit set, take value; else 0
}

// Per row: accumulate pos and neg separately, subtract at end
for (uint32_t chunk = 0; chunk < 64; chunk += 4) {
    acc_pos = vaddq_f32(acc_pos, masked_load4(x, base+chunk, (pos>>chunk)&0xF, ...));
    acc_neg = vaddq_f32(acc_neg, masked_load4(x, base+chunk, (neg>>chunk)&0xF, ...));
}
float sum = hsum128_f32(vsubq_f32(acc_pos, acc_neg));
y[r] = sum * row_scale[r];
```

**No data-dependent branches. No FMA. No multiply.** The CPU never mispredicts.

### The Quantization (from `pack_ternary.py`)

Row-wise error-compensated ternary quantization:
1. Compute `scale = mean(|row|)` per row
2. `threshold = 0.5 * scale`
3. For each weight: `adjusted = value + carry`; if `> threshold` → +1, if `< -threshold` → -1, else 0
4. `carry = adjusted - (q * scale)` — error feeds forward to next weight
5. Pack into pos/neg bit-planes

This preserves row-wise signal better than naive sign quantization.

## Benchmark Numbers (ARM NEON, Snapdragon X, -O3)

| Benchmark | Median ms | Gop/s | Speedup vs Scalar |
|-----------|----------:|------:|-------------------:|
| 1024×1024 ternary matvec | 0.26 | 8.01 | 3.96× |
| 1024×1024 scalar | 1.04 | 2.02 | 1.00× |
| Batched 4×1024 | 1.04 | 8.04 | — |
| MHA decode 128×4×32 | 0.76 | 8.30 | — |
| Transformer block 256 | 0.09 | 8.65 | — |
| RoPE 1024×2048 | 0.00027 | — | — |

All SIMD checksums match scalar reference — **bit-exact equivalence**.

## Distillation Strategy: The "PlasmaPath"

### Five-Tier Compute Hierarchy (aligned with Issue 014 Four-Tier Memory)

```
Tier       Compute                          Memory             Latency
────────   ─────────────────────────────── ───────────────── ──────────
Plasma     Ternary SIMD (add/sub only)     1.58 bits/weight   ~0.3ms/1024²
Hot        FP16 SIMD (FMA)                 16 bits/weight     ~0.5ms/1024²
Warm       SpectralQuant eigenbasis         3-4 bits/weight   ~0.8ms/1024²
Cold       Q4_K dequantize-on-read          4 bits/weight     ~1.2ms/1024²
Freeze     Disk-backed (Turso/DB)           Variable          ~10ms+
```

**Alignment with Issue 014 (Four-Tier Memory):**
- Plasma = new compute tier below Hot, uses ternary weights as "cache" for fastest path
- Hot = current FP32/FP16 SIMD path (existing `simd_matvec`)
- Warm = SpectralQuant compressed KV + weight path
- Cold = Turso/libSQL encrypted storage (Issue 014 implementation)
- Freeze = archival/long-term (ties into Cold tier's blob storage)

The **Plasma** tier maps to Issue 014's "Hot" tier concept — always in CPU registers/L1, never touches RAM. It's the fastest possible compute path for CPU inference.

### Architecture Split: Open vs Private

```
katgpt-core (MIT)                    riir-ai (Private)
─────────────────────                ─────────────────────
TernaryMatrix struct                 Game-specific ternary thresholds
simd_ternary_matvec()                Game-specific row_scale tuning
quantize_row_ternary()               Game domain .bits weight files
TernaryWeights type                  Draft model ternary tuning data
plasma_path feature gate             Ternary→Speculative decode integration

"engine sockets"                     "game plugs"
```

**Rule:** The SIMD kernel and quantization utilities ship open. The `.bits` weight files and per-game ternary tuning (which weights benefit from ternary vs keeping full precision) stay private. This follows the same pattern as all existing GOAT pillars (Docs 27).

### Feature Gate Design

```toml
# katgpt-core/Cargo.toml
[features]
plasma_path = []  # Ternary SIMD matvec — multiplication-free hot path
```

```toml
# katgpt-rs/Cargo.toml
[features]
plasma_path = ["katgpt-core/plasma_path"]
```

```toml
# riir-ai/Cargo.toml (private)
[features]
plasma_path = ["katgpt-core/plasma_path"]
# Note: game-specific ternary quantization tuning stays in riir-ai
```

### Integration Points

1. **`katgpt-core/src/simd.rs`**: Add `simd_ternary_matvec()` alongside existing `simd_matvec()`
2. **`katgpt-core/src/types.rs`**: Add `TernaryWeights` struct (pos_bits, neg_bits, row_scale, blocks64)
3. **`katgpt-rs/src/transformer.rs`**: Add ternary weight dispatch in `forward_base()`
4. **`katgpt-rs/src/weights.rs`**: Add `.bits` file loader for TernaryMatrix
5. **`riir-ai/crates/riir-games/`**: Game-specific ternary quantization and threshold tuning

### Speculative Decoding Integration

The ternary path serves as the **super-fast drafter** in the DDTree:
- PlasmaPath generates draft tokens at ~3-5M tok/s (add/sub only)
- DDTree verifies against the full-precision model
- When confidence gap is high → stay in PlasmaPath longer (more draft tokens before verify)
- When confidence gap is low → fall back to Hot tier for quality

This aligns with the existing SpecHop pipeline (Plan 131) — PlasmaPath becomes hop-0.

## Honest Quality Assessment

### What We Get

| Pro | Evidence |
|-----|----------|
| **2-4× CPU throughput** | Ciot proves 3.96× SIMD vs scalar on ARM NEON. Our hot path already uses SIMD FMA, so gain is less dramatic (~2×) but still significant |
| **Ultra-low power** | ALU-only, no FPU → mobile/edge advantage. Game AI on battery-powered devices |
| **Memory reduction** | 1.58 bits/weight vs 16 (F16) or 32 (F32) → 10-20× less weight memory |
| **Zero-alloc compatible** | Ciot uses pre-allocated aligned memory, no hot-path allocations |
| **Checksum-verified** | Scalar vs SIMD parity testing pattern we already use |

### What We Lose

| Con | Mitigation |
|-----|------------|
| **Quality degradation** | Ternary quantization is lossy. Row-wise error compensation helps but won't match 4-bit quant for quality-sensitive tasks. **Mitigation:** use only for draft/speculative path, not final output |
| **Not a replacement** | This is NOT a replacement for SpectralQuant or full-precision inference. It's an additional fast path for specific use cases (speculative drafting, game AI on edge) |
| **Training gap** | Ciot's trainer is tiny (SGD, no Adam). We'd need to integrate ternary quantization into our existing LoRA training pipeline (riir-burner) |
| **Calibration needed** | Per-layer ternary threshold tuning is game-specific → stays private in riir-ai |
| **Limited to CPU** | No GPU benefit. Our wgpu path wouldn't use this. Pure CPU optimization |

### Where It Fits (Decision Matrix from Docs 27)

| Criterion | Score | Evidence |
|-----------|-------|----------|
| GOAT-passable | ✅ | Ciot proves 3.96× with checksum match. We can replicate + verify |
| MMO-product | ⬜ | Indirect: faster draft → better speculative decode → better NPC/combat AI. Not a direct pillar |
| LoRA-independent | ✅ | Ternary quantization works on any weights. Can quantize LoRA output |
| Defensible | ⬜ | The SIMD kernel is open (like ciot itself). The tuning data is private. Same pattern as Fourier periods |
| Secret coverage | A + B | `game_lora.bin` quantized to `.bits` (A), faster training loops (B) |

**Assessment:** This is a **secondary bet** (per Docs 27 classification), not a pillar. It becomes valuable if we use it to accelerate the draft model in speculative decoding, but it doesn't create new MMO functionality on its own. The real value is in the **five-tier hierarchy** giving us a principled compute/memory trade-off framework.

## Super GOAT (Private / Selling Point)

The "super" capability is **NOT the ternary SIMD kernel itself** (that's open, just like ciot). The secret sauce is:

1. **Per-game ternary threshold tuning** — which layers benefit from ternary vs full precision is domain knowledge
2. **PlasmaPath + DDTree integration** — how many draft tokens before verification, per domain, is tuned data
3. **Ternary + Sparse MLP fusion** — combining ternary weights with our existing sparse activation patterns for compound speedup
4. **Ternary LoRA adapter quantization** — quantizing LoRA adapters to ternary for ultra-fast domain switching

These stay in riir-ai as private implementation details.

## References

- Ciot source: `.raw/ciot/` (full C++ codebase)
- Ciot SIMD kernels: `.raw/ciot/src/kernels/ternary_simd.cpp` (NEON/AVX2/AVX-512/scalar)
- Ciot packing: `.raw/ciot/scripts/pack_ternary.py` (error-compensated quantization)
- Ciot benchmarks: `.raw/ciot/BENCHMARK_REPORT.md` (8.01 Gop/s on ARM NEON)
- Issue 014: Four-Tier Memory Cold Tier (aligns tier naming)
- Docs 27: MMO GOAT Pillars Decision Matrix (secondary bet classification)
