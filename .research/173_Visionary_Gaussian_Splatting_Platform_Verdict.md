# Verdict: Visionary Gaussian Splatting Platform — What's Actually Useful for Us

**Date:** 2026-06
**Source:** [Visionary-Laboratory/visionary](https://github.com/Visionary-Laboratory/visionary)
**Paper:** arXiv:2512.08478
**Status:** Verdict — High Signal on Architecture Patterns, Low Direct Application

---

## What Visionary Actually Is

Visionary is a **web-native 3D/4D Gaussian Splatting renderer** built on WebGPU + ONNX Runtime. It renders millions of Gaussian particles in the browser using:

1. **ONNX Generator Contract** — Any 3DGS variant exports as an ONNX model that outputs attribute tensors (positions, scales, rotations, opacities, SH colors) per frame
2. **WebGPU Compute Shaders** — Radix sort + splat rendering via custom WGSL kernels
3. **GPU Buffer Pipeline** — ONNX inference outputs → GPU buffers → compute shaders → canvas, zero CPU round-trip

**Architecture flow:**
```
Condition Token (time/pose) → ONNX Forward Pass → Attribute Tensor → GPU Radix Sort → Splat Rasterization
```

## What Gemini Got Wrong (Partially)

Gemini's "Holo-Attractor" proposal was **over-scoped** but not entirely wrong. The full volumetric rendering pipeline with ONNX is scope creep. BUT the user correctly identified that:

1. **We need cool viz** — Goodfire proves neural geometry visualization IS research presentation
2. **We have wgpu infrastructure** — 60+ WGSL kernels, just need rendering kernels instead of training kernels
3. **Our data is richer than Goodfire's** — NeuronShard + HLA + oscillatory state
4. **katgpt-rs IS a playground** — Percepta started slow, got 6000× better

**Updated verdict:** See Research 174 for the corrected creative fusion that IS our domain — neuro-geometric viz + physics playground + oscillatory state-space visualization.

## What's Actually Useful — Three Honest Distillations

### D1: Generator Contract Pattern → InferenceBackend Trait Refinement ⭐ GOAT

**The fundamental idea:** Visionary's genius is the **Gaussian Generator Contract** — any algorithm that implements `fn generate(condition) -> AttributeTensor` plugs into the same renderer. The renderer doesn't know or care what happens inside the ONNX model.

**Our mapping:** We already have `InferenceBackend` trait in katgpt-rs. But it's coupled to transformer forward passes. The pattern we should steal is:

```rust
// CURRENT: tightly coupled
trait InferenceBackend {
    fn forward(&mut self, tokens: &[Token], kv: &mut KVCache) -> Vec<f32>;
}

// DISTILLED: contract-based, renderer-agnostic
trait SpeculativeGenerator {
    type Condition;
    type Output;
    
    fn generate(&mut self, condition: &Self::Condition) -> Self::Output;
    fn validate(output: &Self::Output) -> bool;
}

// DDTree implements ConstraintPruner on Output
// InferenceRouter dispatches Condition to appropriate backend
// WASM sandbox validates Output
```

**Why this matters:** This is DRY + SOLID. Right now, DDTree, SpeculativeVerifier, and InferenceRouter all know about token logits. If we decouple the *generation contract* from the *generation mechanism*, the same DDTree + WASM + routing infrastructure works for:
- Text token generation (current)
- Game action generation (riir-ai)
- Any future generation domain

**Modelless:** ✅ Pure trait refactor, no training
**Gain:** DRY — removes duplication between katgpt-rs speculative and riir-ai game speculative
**Risk:** Refactoring cost, but trait-based so zero perf impact

**Verdict: GAIN. DRY refactor that makes our existing architecture more generic without changing behavior.**

### D2: GPU Buffer Zero-Copy Pipeline → riir-gpu Scratch Buffer Unification ⚠️ MARGINAL

**The fundamental idea:** Visionary achieves zero CPU round-trips by having ONNX inference write directly to GPU buffers that compute shaders read from. No `CPU → serialize → GPU upload → GPU compute`. It's all GPU-side.

**Our mapping:** Our riir-gpu crate already uses wgpu buffers for LoRA training and inference. But each kernel creates its own staging buffers. The pattern we could steal:

1. Pre-allocate a **GPU buffer arena** at startup
2. Each kernel dispatch writes into offset-regions of the arena
3. No per-dispatch allocation, no staging buffer churn

**However:** This is already partially implemented. Our WGSL matmul kernels use pre-allocated buffers. The gap is in the **staging buffer allocation pattern** — some paths allocate temporary buffers per forward pass.

**Modelless:** ✅ Pure optimization, no training
**Gain:** Marginal. Our hot path is already pre-allocated. The staging churn is in cold paths (model loading, weight transfer).
**Risk:** Adds complexity for marginal gain.

**Verdict: MARGINAL GAIN. Not worth a separate plan. Fold into existing GPU optimization work if buffer profiling shows staging churn as a bottleneck.**

### D3: Adaptive Precision Detection → Already Have This ⛔ SKIP

**The fundamental idea:** Visionary auto-detects model precision (FP32/FP16) from ONNX metadata and converts between formats using a compute shader.

**Our mapping:** We already have Q4_K, F16, F32 paths in riir-gpu with automatic selection. Our `precision_detector.ts` equivalent is our quantized weight loading with automatic dequantization paths.

**Verdict: NO GAIN. Already implemented.**

## What's NOT Useful (Directly) BUT Has Derivative Value

| Visionary Feature | Direct Use? | Derivative Value |
|---|---|---|
| Gaussian Splatting rendering | ❌ Wrong domain | ✅ WGSL splat shader pattern → our point cloud renderer |
| ONNX model loading | ❌ Wrong format | ❌ No derivative value |
| Radix sort WGSL | ❌ Different data | ✅ Sort algorithm → depth-sorting our point clouds |
| WebGPU canvas rendering | ❌ Browser-only | ✅ Same wgpu compute → render pipeline architecture |
| Three.js integration | ❌ Browser-only | ❌ No derivative value |
| Camera/projective transforms | ❌ No 3D data yet | ✅ Arcball camera for our 3D viz (Research 174) |

## Honest Gap Assessment

| Component | Visionary Has | We Have | Gap? |
|---|---|---|---|
| ONNX graph inference | ✅ | ❌ | **None needed** — we have our own weight format |
| GPU compute shaders | ✅ (WGSL for splatting) | ✅ (WGSL for matmul/attention) | **None** — different shaders for different domains |
| Adaptive compute tier routing | ❌ | ✅ (TriggerGate) | **We're ahead** |
| Speculative decode tree | ❌ | ✅ (DDTree + ConstraintPruner) | **We're ahead** |
| Deterministic validation | ❌ | ✅ (WASM sandbox) | **We're ahead** |
| Bandit-based adaptation | ❌ | ✅ (MultiArmedBandit, ConfiguratorBandit) | **We're ahead** |
| 3D rendering pipeline | ✅ | ❌ | **Irrelevant to our domain** |

## Final Verdict

**GOAT: D1 only — Generator Contract Pattern trait refactor.**

Visionary is impressive engineering for a domain that's orthogonal to ours. The one architectural pattern worth stealing is the **contract-based decoupling** of generation from rendering, which maps to our `InferenceBackend` → `SpeculativeGenerator` trait generalization. Everything else is either already implemented in our stack or irrelevant to text/game AI.

**No new GPU work needed. No ONNX integration needed. No visual rendering needed.**

The Gemini "Holo-Attractor" proposal would have us build an entire 3D rendering platform from scratch. That's months of work for zero benefit to our core product (text inference + game AI). The honest gain is a DRY trait refactor that makes our existing speculative decoding infrastructure more reusable across domains.

---

## TL;DR

Visionary = great 3D rendering platform. Direct domain mapping is wrong, but the **WGSL rendering architecture** (point cloud + radix sort + GPU buffer pipeline) IS useful for our neuro-geometric viz playground (Research 174). The contract-based generator trait (D1 from original verdict) maps to our `InferenceBackend` generalization (Plan 193). Two plans spawned: Plan 193 (trait refactor) and Plan 219 (viz + physics playground).
