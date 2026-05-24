# Research 64: LlamaWeb — WebGPU LLM Inference Distillation

**Paper:** [arXiv:2605.20706](https://arxiv.org/abs/2605.20706) (May 2026)
**Authors:** (LlamaWeb team)
**Venue:** Preprint

## TL;DR

LlamaWeb is a WebGPU backend for llama.cpp enabling browser-based LLM inference. It achieves 29-33% less memory and 45-69% faster decode vs WebLLM/Transformers.js through templated WGSL kernels, subgroup matrix operations, performance-portable kernel tuning, quantized KV cache, and FlashDecoding. Key gaps vs our riir-ai stack: subgroups, FlashDecoding, quantized KV, and auto-tuned tiles.

## Context

We maintain **60+ WGSL kernels** in `riir-gpu` and a complete GPU inference pipeline in `riir-ai`. Our stack already implements many optimizations LlamaWeb describes (fused passes, quantized GEMV, static buffers). This distillation identifies what we're missing and what's worth adopting.

## What We Already Do (riir-ai)

| Component | Our Implementation | LlamaWeb Equivalent | Status |
|-----------|-------------------|---------------------|--------|
| Static buffer allocation | Plan 077: `GpuScratchBuffers` | Single arena allocation | ✅ Done |
| Fused compute passes | Plan 098: 3 passes/layer instead of 16 | Pass fusion | ✅ Done |
| Quantized GEMV | Plan 100: Q4_K, +52% decode speedup | Multi-format GEMV | ✅ Done |
| Dual GEMV fusion | Plan 101: gate+up in single dispatch | FFN fusion | ✅ Done |
| Batched prefill kernels | Plan 102: `matmul_transb`, `rope_batch`, `attention_prefill` | Prefill kernels | ⚠️ Partial |
| Pipeline caching | All pipelines created once in `new()` | Pipeline caching | ✅ Done |
| WGSL kernel library | 60+ kernels in `riir-gpu` | Templated WGSL library | ✅ Done |

## What LlamaWeb Does That We Don't (Gaps)

### 1. Templated WGSL Preprocessor (pre-wgsl)

LlamaWeb has a C++ preprocessor for conditional compilation of WGSL kernels. It selects code paths based on:
- Quantization format (Q4_0, Q4_K, Q5_0, Q5_K, Q8_0, etc.)
- Subgroup support availability
- Workgroup size targets
- Platform-specific workarounds

**Our approach:** `include_str!` + separate kernel files per specialization. No templating — we duplicate kernels when variants are needed.

**Gap impact:** Medium. Our kernel count grows linearly with format count. A preprocessor would reduce maintenance but adds build complexity.

### 2. Subgroup Matrix Operations

LlamaWeb uses WebGPU's experimental subgroup matrix feature for hardware-accelerated matmul (tensor cores). This provides:
- Direct access to SIMD/matrix multiply instructions
- Apple M-series: SIMD matrix instructions via Metal
- Intel Arc: XMX matrix engines
- NVIDIA: Tensor cores

**Our approach:** No subgroup usage at all. Deferred in Plan 058. Standard workgroup shared-memory tiling only.

**Gap impact:** HIGH. Subgroup matrix ops are the single biggest untapped performance win. Potential 2-3× matmul speedup on Apple M-series hardware we primarily target.

### 3. Performance-Portable Kernel Tuning

LlamaWeb sweeps thousands of tiling configurations across 4 GPU vendors (Apple, Intel, NVIDIA, AMD) to find optimal tile sizes per kernel per device.

**Our approach:** Hardcoded 16×16 tiles and 256-thread workgroups.

**Gap impact:** Low-Medium. We primarily target Apple Metal where 16×16 is reasonable. Would matter more if we expanded hardware support.

### 4. Parameter Buffer Arena

LlamaWeb allocates a single GPU buffer with rotating slots for kernel parameters (uniform buffers). This reduces allocation overhead and memory fragmentation.

**Our approach:** Separate uniform buffers per operation. Works correctly but uses more total memory and creates more bind groups.

**Gap impact:** Low. Our static buffer plan (077) already addresses the main allocation concerns. Arena would be a marginal improvement.

### 5. Quantized KV Cache

LlamaWeb supports Q4_0/Q8_0 KV cache with in-kernel dequantization. Keys and values are stored compressed and dequantized during attention computation.

**Our approach:** F32 only KV cache. No compression.

**Gap impact:** Medium-High for long contexts. Q8_0 KV cache would reduce memory by 4× with minimal quality loss. Critical for scaling context length.

### 6. Kernel Fusion for Prefill

LlamaWeb notes WebLLM beats them in prefill due to TVM kernel fusion. They identify this as a weakness.

**Our approach:** Plan 102 (batched prefill) is partially done. Prefill is slower than decode in our stack too.

**Gap impact:** Medium. Prefill performance matters for first-token latency and batch processing.

### 7. FlashDecoding

LlamaWeb implements FlashDecoding for decode-phase attention. Multiple workgroups cooperate per query vector with intermediate partial reductions.

**Our approach:** Single-workgroup attention for decode. Each query vector is processed by one workgroup.

**Gap impact:** High for long contexts. FlashDecoding parallelizes the KV scan across workgroups, critical when KV sequence length exceeds what a single workgroup can efficiently process.

### 8. Cross-Device Benchmarking

LlamaWeb tests 16 devices from 8 GPU vendors (Apple, Intel, NVIDIA, AMD, Qualcomm, etc.).

**Our approach:** Apple Metal only.

**Gap impact:** Low for our current scope. Our target is Apple ecosystem.

## What's NOT Applicable to Us

| Feature | Why Not Applicable |
|---------|-------------------|
| Browser-specific (OPFS, Web Workers, WASM heap) | We run native, not in browser |
| Model loading streaming | We load weights upfront, no progressive loading |
| Safety check bypass | Native wgpu doesn't add bounds checks like browser WebGPU |
| 23 quantization formats | We only need Q4_K for now |
| WASM heap management | Native memory model, no 4GB WASM limit |
| Web Worker thread coordination | Native threads via Rayon |

## Key Numbers (LlamaWeb Benchmarks)

### Memory Efficiency
| Metric | vs WebLLM | vs Transformers.js |
|--------|-----------|-------------------|
| Memory reduction | 29% | 33% |

### Decode Speed
| Metric | vs WebLLM | vs Transformers.js |
|--------|-----------|-------------------|
| Speedup | 45% | 69% |

### Competitive Positioning
| Backend | Intel Arc | Apple (Vulkan) | Apple (Metal) |
|---------|-----------|----------------|---------------|
| LlamaWeb | Competitive | Beats Vulkan | N/A (WebGPU→Metal) |
| Native llama.cpp | Baseline | Baseline | Baseline |

### Limitations Noted
| Issue | Impact |
|-------|--------|
| 4.5 bpw Q4_K memory savings vs performance | Saves memory, not perf (dequant overhead) |
| Prefill vs WebLLM | 21-51% slower (no kernel fusion) |
| Browser safety checks | 14-42% prefill slowdown |

## Architecture Gap Analysis

```
riir-ai/crates/riir-gpu/src/
├── kernels/
│   ├── matmul.wgsl              # 16×16 tiled, NO subgroups ❌
│   ├── matmul_q4k.wgsl          # Quantized GEMV ✅
│   ├── attention_decode.wgsl    # Single workgroup ❌ (no FlashDecoding)
│   ├── attention_prefill.wgsl   # Batched, partial ⚠️
│   └── ...60+ kernels
├── gpu_context.rs               # Separate uniform buffers ❌ (no arena)
└── pipeline_cache.rs            # All cached at init ✅

MISSING:
├── subgroups/
│   ├── mod.rs                   # Feature gate: gpu_subgroup
│   ├── subgroup_matmul.wgsl     # SIMD matrix multiply
│   └── subgroup_detect.rs       # Capability detection
├── flash_decode/
│   ├── mod.rs                   # Feature gate: gpu_flash_decode
│   ├── flash_decode.wgsl        # Multi-workgroup attention
│   └── partial_reduce.wgsl      # Intermediate reduction
└── quantized_kv/
    ├── mod.rs                   # Feature gate: gpu_q8_kv_cache
    ├── kv_q8_decode.wgsl        # In-kernel dequant + attention
    └── kv_q8_encode.rs          # CPU-side Q8 encoding
```

## Verdict

### Priority Rankings for riir-ai (wgpu GPU training + inference)

| Priority | Feature | Expected Impact | Effort | Feature Gate |
|----------|---------|----------------|--------|-------------|
| **1** | Subgroup matrix operations | 2-3× matmul speedup | High | `gpu_subgroup` |
| **2** | FlashDecoding for decode attention | Significant at long contexts | Medium | `gpu_flash_decode` |
| **3** | Quantized KV cache (Q8_0) | 4× KV memory reduction | Medium | `gpu_q8_kv_cache` |
| **4** | Auto-tuned tile sizes | Marginal on Apple Metal | Low | `gpu_autotune` |

### Priority 1 — Subgroup Matrix Operations

Apple M-series has SIMD matrix instructions accessible via wgpu subgroups. This is the biggest untapped win in our GPU stack. Potential 2-3× matmul speedup on the hardware we primarily target.

**Implementation notes:**
- Feature gate: `gpu_subgroup`
- Requires wgpu device feature `SUBGROUP` + `SUBGROUP_MATRIX`
- Fallback to current tiling when unavailable
- Apple Metal: SIMD matrix multiply (8×8 input → 8×8 accumulate)
- Must detect at runtime and select kernel variant

### Priority 2 — FlashDecoding for Decode Attention

Our decode attention is naive — single workgroup per query vector. FlashDecoding splits the KV scan across multiple workgroups with intermediate partial reductions.

**Implementation notes:**
- Feature gate: `gpu_flash_decode`
- Split KV range into chunks, one workgroup per chunk
- Partial softmax reduction (online softmax over chunks)
- Final reduction pass combines partial results
- Most impactful when `seq_len >> workgroup_size`

### Priority 3 — Quantized KV Cache (Q8_0)

Memory savings for longer contexts. Our KV cache is always f32. Q8_0 KV cache would reduce memory by 4× with minimal quality loss.

**Implementation notes:**
- Feature gate: `gpu_q8_kv_cache`
- Store K/V as Q8_0 blocks (f32 → int8 with block scale)
- In-kernel dequantization during attention score computation
- Fused dequant+dot-product avoids materializing full f32 K/V
- Composable with OCTOPUS (Research 63) for extreme compression

### Priority 4 — Auto-Tuned Tile Sizes

Our 16×16 tiles are suboptimal on some hardware. A small benchmark at startup could select better tile sizes.

**Implementation notes:**
- Feature gate: `gpu_autotune`
- Run micro-benchmarks at startup for each kernel variant
- Cache results per GPU device ID
- Low priority since we primarily target Apple Metal (16×16 is decent)

### Implications for katgpt-rs (model-based/modelless)

**No direct wgpu code** — katgpt-rs is pure CPU (SIMD, Rayon). The paper's lessons apply architecturally:

| Principle | CPU Analog in katgpt-rs |
|-----------|--------------------------|
| Dispatch overhead reduction | Batch operations, avoid per-token overhead |
| Kernel fusion | Fused SIMD kernels (e.g., fused norm+activation) |
| Memory planning | Pre-allocated scratch buffers, arena allocation |
| Quantized inference | Q4_K dequant in `katgpt-rs/src/quantize/` |
| Static pipeline state | Pre-warmed Rayon thread pool, pinned cores |

**Indirect benefit:** riir-ai GPU improvements flow back to katgpt-rs through shared `riir-engine` types and quantization code.

### Recommended Feature Gates

```toml
[features]
default = []
gpu_subgroup = []       # Priority 1: Subgroup matrix operations
gpu_flash_decode = []   # Priority 2: FlashDecoding for decode
gpu_q8_kv_cache = []    # Priority 3: Quantized KV cache
gpu_autotune = []       # Priority 4: Auto-tuned tile sizes
```

### Risks & Limitations

1. **Subgroup availability**: Not all wgpu backends expose subgroups. Must have reliable fallback path.
2. **WebGPU vs native wgpu**: LlamaWeb targets browser WebGPU (subset of wgpu). Some features may not translate directly.
3. **Apple Metal specifics**: Our primary target has good subgroup support via Metal, but wgpu's subgroup API is still evolving.
4. **Browser paper, native stack**: ~30% of the paper is browser-specific optimization that doesn't apply to us.
5. **Quantization format scope**: LlamaWeb supports 23 formats. We only need Q4_K. Don't over-engineer format support.

## What NOT to Adopt

| Feature | Why Skip |
|---------|----------|
| pre-wgsl templating | Our `include_str!` approach is simpler, Rust-native |
| OPFS/model streaming | Native loads are fast enough |
| 23 quantization formats | YAGNI — Q4_K is sufficient |
| Web Worker coordination | Native threads via Rayon |
| Safety check workarounds | Not needed in native wgpu |

## References

- LlamaWeb paper: arXiv:2605.20706 (May 2026)
- Our Plan 058: Subgroup operations (deferred)
- Our Plan 077: `GpuScratchBuffers` (static allocation)
- Our Plan 098: Fused compute passes (3 passes/layer)
- Our Plan 100: Quantized GEMV (Q4_K)
- Our Plan 101: Dual GEMV fusion (gate+up)
- Our Plan 102: Batched prefill kernels (partial)
- Research 29: rust-gpu feasibility (WGSL migration)
- Research 63: OCTOPUS KV cache compression (complementary)