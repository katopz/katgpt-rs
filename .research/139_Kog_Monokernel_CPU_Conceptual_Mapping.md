# Research 139: Kog AI Monokernel — CPU Inference Conceptual Mapping

> **Source:** [Kog AI Blog Series](https://blog.kog.ai/) — Monokernel + DTP + KCCL, 2026-05
> **Date:** 2026-05-30
> **Related Research:** 066 (TileRT), 067 (CODA), 112 (mKernel), 012 (Lucebox megakernel)
> **Related Plans:** 012 (Lucebox distill — megakernel aspirational), 171 (riir-ai — GPU decode fusion, primary implementation)
> **Domain:** katgpt-rs (CPU SIMD inference — conceptual patterns only)

---

## TL;DR

Kog AI's monokernel approach (3,000 tok/s on 8× MI300X) provides three conceptual patterns for our CPU SIMD inference stack: (1) **dispatch overhead amortization** — the CPU analog of kernel launch overhead is function call + cache cold-start, and our existing fused SIMD kernels (e.g., `simd_dot_f32`, `coda` module) already do this correctly; (2) **memory bandwidth utilization** — our SIMD kernels are also memory-bound at small model sizes, and weight quantization (Q4_K in riir-ai) is the primary tool; (3) **offline preprocessing** — RMSNorm folding and weight repacking apply equally to CPU weight loading.

**Verdict: LOW DIRECT GAIN for katgpt-rs CPU inference — our CPU SIMD kernels are already well-optimized with no kernel-launch overhead analog (we're already in-process, no GPU↔CPU boundary). The two actionable insights are: (1) RMSNorm gamma folding into weight matrices at load time (saves one pass over the weight per norm in `forward_gemma2`), and (2) the MBU metric as a diagnostic for identifying when SIMD kernels are memory-bound vs compute-bound. Both are minor optimizations — our CPU bottleneck is FLOPs, not bandwidth, at the micro config scale. The real implementation lives in riir-ai Plan 171 (GPU decode fusion). No feature gate needed in katgpt-rs. No plan in katgpt-rs — the CPU-side RMSNorm folding can be a standalone PR if benchmarked as beneficial.**

---

## Conceptual Mapping: GPU → CPU

| Kog GPU Concept | CPU SIMD Analog | Already Done? |
|----------------|-----------------|---------------|
| Kernel launch overhead (4.5 µs) | Function call overhead (~1 ns) | N/A — negligible |
| Cache write-back at kernel boundary | L1/L2 cache pressure between ops | Partially — our fused kernels avoid this |
| Intermediate tensor materialization | `Vec` allocation in hot loop | ✅ Avoided — pre-allocated scratch buffers |
| Memory bandwidth bottleneck | Memory bandwidth bottleneck | ✅ Recognized — SIMD chunked loops |
| RMSNorm folding | Same — fold gamma into weights | ❌ Not done — potential micro-optimization |
| QKV interleaving | Weight row reorder for cache locality | ❌ Not done — potential for large models |
| GEMV not tensor cores at BS=1 | SIMD dot product (not matrix ops) | ✅ Already our approach |
| Persistent kernel (monokernel) | Single function call for whole forward | Partially — `forward_gemma2` is already one call |

---

## What Transfers to CPU

### 1. RMSNorm Gamma Folding (Minor)

In `forward_gemma2`, each layer has 4 RMSNorm calls. Each computes:
```rust
let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
for i in 0..dim { data[i] = data[i] * inv_rms * gamma[i]; }
```

If we fold gamma into the *following* projection weight at load time:
```rust
// At load time:
for row in 0..out_dim {
    for col in 0..n_embd {
        weight[row * n_embd + col] *= gamma[col];
    }
}
// At runtime: only compute inv_rms, multiply by weight (gamma already folded)
```

Saves: 4 weight loads per layer (reading gamma vectors) × 26 layers = 104 fewer memory accesses.
At our micro config (n_embd=16), this saves ~6.6 KB of reads per token — negligible.
At Gemma 2 scale (n_embd=2304), this saves ~960 KB — more meaningful.

**Verdict: Not worth it for micro configs. Worth it for Gemma 2 CPU path if we add one.**

### 2. MBU as Diagnostic

Kog's MBU metric = (actual bytes streamed) / (theoretical peak bandwidth × time).
Our CPU analog: bytes processed per second vs theoretical memory bandwidth.

For Apple M3 Max (400 GB/s unified):
- Gemma 2 2B Q4_K: 0.71 GB/token → 563 tok/s theoretical
- Our GPU: 12 tok/s = 2.1% MBU (from riir-ai benchmark)
- CPU: even lower (no GPU parallelism), but we don't benchmark CPU Gemma 2

**Verdict: Useful diagnostic concept. Apply to CPU SIMD benchmarking if we scale to larger CPU inference.**

---

## What Does NOT Transfer

| Concept | Why Not |
|---------|---------|
| Monokernel (persistent kernel) | CPU functions don't have launch overhead; we're already in-process |
| NaN-sentinel synchronization | No parallel compute units to synchronize on CPU (SIMD is inline) |
| Tensor parallelism / DTP | Single CPU, no device parallelism |
| Weight prefetch during compute | CPU has hardware prefetcher; explicit prefetch (`_mm_prefetch`) rarely helps in practice |
| Non-temporal load hints | `_mm_stream_load` exists but our working set fits in L2 cache at micro config |

---

## References

1. Full analysis: riir-ai `.research/027_Kog_Monokernel_DTP_Latency_Optimized_Inference.md`
2. Implementation plan: riir-ai `.plans/171_kog_distilled_gpu_decode_fusion.md`
3. Kog AI Blog: [Real-time LLM Inference](https://blog.kog.ai/real-time-llm-inference-on-standard-gpus-3-000-tokens-s-per-request/)
