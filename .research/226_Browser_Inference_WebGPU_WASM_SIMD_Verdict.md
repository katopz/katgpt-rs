# Research 226: Browser Inference — WebGPU + WASM SIMD (katgpt-rs Cross-Reference)

**Date:** 2026-06-12
**Source:** riir-ai Research 113 (Browser Inference Verdict)
**Related:** riir-ai Research 113, riir-ai Plan 286, Research 110 (CiOT Ternary Inference), Research 064 (LlamaWeb)
**Status:** GOAT Verdict — **GAIN** (cross-repo)

---

## Summary

**katgpt-rs modelless path gains WASM SIMD ternary bit-plane extraction.** No wgpu code lands in katgpt-rs (modelless stays CPU). But the SIMD kernels for ternary projection and small matmul should be authored here and shared to riir-ai.

---

## What Lands in katgpt-rs

### WASM SIMD Ternary Bit-Plane Extraction

`SenseModule::project()` uses ternary bit-plane extraction. WASM `v128` SIMD provides 128-bit bitwise ops:
- 16 ternary values (0/1/-1) extracted per SIMD instruction
- `v128_bitmask`, `v128_and`, `v128_or`, `v128_xor` for bit manipulation
- Maps perfectly to our ternary representation

This is a **new SIMD dispatch tier** in `katgpt-core/src/simd.rs`:

```
Current:  AVX2 → NEON → scalar
New:      AVX2 → NEON → WASM simd128 → scalar
```

### No GPU Code

katgpt-rs remains pure CPU. The WebGPU path lives in riir-ai/riir-gpu. The shared interface is:
- `katgpt-core` provides SIMD kernels (native + WASM)
- `riir-gpu` provides GPU kernels (native Metal + browser WebGPU)
- `riir-engine` auto-routes based on target

---

## GOAT Verdict per 003

| Question | Answer |
|----------|--------|
| Engine or fuel? | Engine — SIMD dispatch is plumbing |
| Ferrari, no gas? | Yes — ternary weights are fuel |
| Modelless? | ✅ Pure inference-time bit ops |
| Tests? | Benchmark: scalar vs WASM SIMD for ternary projection |

**VERDICT: GAIN — implement WASM SIMD dispatch in katgpt-core, shared to riir-ai via dependency**

---

## References

- riir-ai Research 113 — full browser inference verdict
- riir-ai Plan 286 — implementation plan
- Research 110 (CiOT) — ternary inference CPU distillation
- `katgpt-core/src/simd.rs` — existing SIMD dispatch

---

TL;DR: **WASM SIMD ternary bit-plane extraction lands in katgpt-core. New dispatch tier: AVX2 → NEON → WASM simd128 → scalar. No GPU code in katgpt-rs.**
