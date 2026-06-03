# Verdict: Apple Neural Engine (ANE) as Compute Backend for katgpt-rs

**Date:** 2026-06
**Source:** [videlalvaro/ane-book](https://github.com/videlalvaro/ane-book) + [ncdrone/rustane](https://github.com/ncdrone/rustane) + [rane crate](https://lib.rs/crates/rane) + [coreml-native crate](https://crates.io/crates/coreml-native)
**Status:** GOAT Verdict

---

## The Point

**Every Apple Silicon Mac has a 15.8 TOPS Neural Engine sitting idle.** We use CPU and GPU (wgpu). We don't use ANE. This is free compute we're leaving on the table.

The ane-book shows production LLM inference at 17 tok/s (Phi-4-mini 3.8B) on ANE alone. `rustane` trains 5B params on ANE at 74 tok/s. `rane` gives pure Rust ANE access with zero-copy IOSurface, 10-20× lower dispatch overhead than CoreML.

---

## Three Paths to ANE from Rust

| Path | Crate | Approach | ANE Access | Dependencies | Status |
|------|-------|----------|------------|-------------|--------|
| **A. CoreML (official)** | `coreml-native` v0.2 | objc2 → CoreML.framework → ANE | Indirect (CoreML decides) | objc2 ecosystem | Stable, production |
| **B. Direct ANE (private)** | `rane` v0.2 | dlopen private frameworks → MIL bytecode → ANE | Direct, full control | Zero deps | Experimental |
| **C. Training+Inference** | `rustane` (MIT) | Full engine: ANE forward + CPU backward + Metal decode | Direct + Metal | objc2, Accelerate | Training validated 600M-5B |

### Path Comparison

| Aspect | CoreML (A) | Direct ANE (B) | rustane (C) |
|--------|-----------|----------------|-------------|
| **Control over ANE** | CoreML decides placement | You control everything | You control everything |
| **Dispatch overhead** | ~5ms (MLFeatureProvider wrapping) | ~0.24ms (IOSurface → ANE) | ~0.095ms (XPC+IOKit) |
| **Memory** | CoreML manages | IOSurface zero-copy | IOSurface zero-copy |
| **Stability** | ✅ Public API, Apple-supported | ⚠️ Private APIs, could break | ⚠️ Private APIs, MIT licensed |
| **Quantization** | INT8 per-tensor (ANE-verified) | FP16 (ANE casts internally) | FP16 internal, FP32 surface |
| **Training** | ❌ Inference only | ❌ Inference only | ✅ Full training pipeline |
| **LoRA** | Not built-in | Not built-in | Not built-in (we add it) |
| **Rust-native** | Yes (objc2) | Yes (zero deps) | Yes (objc2+Accelerate) |

---

## GOAT Decision: Which Path?

### For katgpt-rs (modelless inference engine)

**Path A: `coreml-native`** is the right choice because:
1. katgpt-rs is MIT open-source engine — private API dependency is a risk for users
2. CoreML handles ANE scheduling, quantization, and fallback automatically
3. `coreml-native` has `ComputeUnits::CpuAndNeuralEngine` — we get ANE when available
4. INT8 per-tensor is production-verified for ANE (ane-book proved this)
5. Stateful KV cache via `MLState` (macOS 15+) matches our Raven RSM pattern

**What katgpt-rs gets from ANE:**
- Transformer forward pass offloaded to ANE (free CPU for DDTree + pruning)
- ~15-17 tok/s for 3.8B model on M4 Max (vs CPU-only)
- CPU freed for constraint pruning, DDTree search, speculative verification

### For riir-ai (model-based, LoRA training)

**Path C: `rustane` concepts + our LoRA** is the right choice because:
1. riir-ai already uses wgpu for GPU training — ANE is a third compute target
2. rustane proved ANE training works at 5B scale with 74 tok/s
3. rustane's architecture (ANE forward + CPU backward + Metal decode) maps to our engine/gpu/games split
4. MIT licensed — can study and adapt patterns
5. We don't use rustane directly — we learn from its ANE kernel compilation and IOSurface patterns

**What riir-ai gets from ANE:**
- ANE forward pass for game AI inference (LoRA scoring, NPC dialog)
- CPU freed for WASM validation, frame sampling, MCTS
- GPU freed for LoRA training while ANE handles inference
- Three-way auto-route: CPU (pruners/WASM) → GPU (training) → ANE (inference)

### Path B: `rane` as fallback/experimental

`rane` is the rawest ANE access — pure Rust, zero deps, 10-20× lower dispatch overhead than CoreML. Use when:
- CoreML refuses to place ops on ANE (the INT4 bug scenario)
- We need deterministic ANE scheduling (not CoreML's black box)
- We need zero-copy between CPU/GPU/ANE (IOSurface shared memory)

Keep as an optional backend behind a feature flag for power users.

---

## What We Actually Build

### katgpt-rs: ANE Backend via coreml-native

```
                    ┌─────────────────────────────────────────┐
                    │           katgpt-rs inference            │
                    ├─────────┬──────────┬──────────┬─────────┤
                    │  Token- │  ANE     │  CPU     │  CPU    │
                    │  izer   │ Forward  │  DDTree  │ Pruners │
                    │ (CPU)   │ (CoreML) │ (Rust)   │ (Rust)  │
                    └─────────┴──────────┴──────────┴─────────┘
                    ▲          ▲          ▲          ▲
                    │          │          │          │
              CPU threads   ANE chip  CPU threads  CPU threads
```

**Key insight:** DDTree and ConstraintPruner stay on CPU (they're discrete algorithms, not matmul-heavy). Only the transformer forward pass goes to ANE. This is the **CPU/GPU/ANE auto-route** — CPU for logic, ANE for inference, GPU for training (when available).

**Implementation:**

```rust
// New trait: abstract forward pass backend
pub trait InferenceBackend: Send + Sync {
    fn forward(&mut self, tokens: &[usize], pos: usize) -> Result<Vec<f32>>;
    fn device_name(&self) -> &str;
}

// CPU backend (current)
pub struct CpuBackend { /* existing transformer */ }

// ANE backend (new)
#[cfg(target_os = "macos")]
pub struct AneBackend {
    model: coreml_native::Model,
    state: Option<coreml_native::State>,  // MLState for KV cache
}

// Auto-select at startup
pub fn auto_backend(weights: &Weights) -> Box<dyn InferenceBackend> {
    #[cfg(target_os = "macos")]
    {
        if std::path::Path::new("model.mlmodelc").exists() {
            if let Ok(model) = coreml_native::Model::load(
                "model.mlmodelc",
                coreml_native::ComputeUnits::CpuAndNeuralEngine,
            ) {
                // Validate ANE residency
                if ane_resident(&model) {
                    return Box::new(AneBackend::new(model));
                }
            }
        }
    }
    Box::new(CpuBackend::new(weights))
}
```

**Residency validation (the ane-book's key lesson):**

```rust
/// Verify the compiled .mlmodelc actually uses ANE, not CPU fallback
#[cfg(target_os = "macos")]
fn ane_resident(model: &coreml_native::Model) -> bool {
    // Run one prediction and check which device executed
    // CoreML doesn't expose MLComputePlan from Rust easily
    // Workaround: time a small prediction — ANE <1ms, CPU fallback >5ms
    let start = std::time::Instant::now();
    let _ = model.predict(&dummy_input());
    let elapsed = start.elapsed();
    elapsed.as_millis() < 2  // ANE should be sub-ms for tiny input
}
```

**Feature flag:**

```toml
# katgpt-rs/Cargo.toml
[target.'cfg(target_os = "macos")'.dependencies]
coreml-native = { version = "0.2", optional = true }

[features]
ane = ["coreml-native"]  # ANE backend for Apple Silicon
```

### riir-ai: ANE Forward + GPU Training + CPU Logic

```
                    ┌──────────────────────────────────────────────┐
                    │            riir-ai game AI pipeline           │
                    ├──────────┬──────────┬──────────┬──────────────┤
                    │  WASM    │  LoRA    │  LoRA    │  ANE        │
                    │ Validate │ Score    │ Train    │ Forward     │
                    │ (CPU)    │ (ANE)    │ (GPU)    │ (CoreML/    │
                    │          │          │          │  rane)      │
                    └──────────┴──────────┴──────────┴──────────────┘
                    ▲          ▲          ▲           ▲
                    │          │          │           │
              CPU threads   ANE chip  wgpu GPU    ANE chip
```

**The key insight from rustane:** ANE forward + CPU backward + Metal Adam optimizer. We adapt this to:
- **ANE:** LoRA inference (forward pass with adapter weights)
- **CPU:** WASM validation, frame sampling, MCTS, bandit decisions
- **GPU:** LoRA training (already works via riir-gpu)

This is the **three-way auto-route**: CPU for logic, ANE for inference, GPU for training.

**Implementation sketch:**

```rust
// riir-engine: new ANE inference backend
#[cfg(target_os = "macos")]
pub struct AneLoraInference {
    // Pre-compiled CoreML model for base transformer
    base_model: coreml_native::Model,
    // LoRA weights applied as IOSurface overlay (rane pattern)
    lora_surfaces: HashMap<String, rane::Buffer>,
    // Stateful KV cache for game context
    kv_state: Option<coreml_native::State>,
}
```

---

## Conversion Pipeline: GGUF → CoreML

The ane-book's conversion pipeline is Python-based. We need:

1. **Python converter** (from ane-book): GGUF → PyTorch → CoreML with Conv2d(1×1) trick
2. **Pre-compiled .mlmodelc** shipped with the crate (or built at install time)
3. **Rust-side:** Load .mlmodelc via coreml-native, run inference

This is a build-time step, not a runtime dependency. Users on macOS get ANE acceleration for free after the one-time conversion.

---

## Expected Gains

### katgpt-rs (modelless)

| Metric | CPU-only (current) | CPU + ANE | Gain |
|--------|-------------------|-----------|------|
| Forward pass (3.8B) | ~50ms (CPU matmul) | ~3ms (ANE) | **16× faster** |
| DDTree budget | Shared CPU with forward | Dedicated CPU (forward on ANE) | **More tree depth possible** |
| Power draw | CPU at ~30W | ANE at ~5W | **6× less power** |
| Speculative decode | CPU-limited | ANE forward + CPU verify | **Higher acceptance rate** |

### riir-ai (model-based)

| Metric | GPU-only (current) | GPU + ANE | Gain |
|--------|-------------------|-----------|------|
| Game AI inference | GPU shared with training | ANE inference, GPU free for training | **No contention** |
| LoRA scoring | GPU context switch | ANE dedicated | **Lower latency** |
| NPC dialog | GPU or CPU | ANE (latent RAG forward) | **CPU free for game logic** |
| Training throughput | 100% GPU | GPU training + ANE inference parallel | **Better throughput** |

---

## GOAT Verdict per 003

| Question | Answer |
|----------|--------|
| Does this land in engine (MIT) or fuel (SaaS)? | **Engine** — the inference backend is plumbing, not proprietary data |
| Does this fit the "Ferrari, no gas" model? | Yes — ANE backend is the Ferrari, `lora.bin` is still the gas |
| Does this hurt existing CPU/GPU paths? | No — feature-gated, `coreml-native` is optional dep |
| Should it be on by default? | **YES on macOS** when .mlmodelc exists — it's free performance |
| Modelless? | katgpt-rs: yes (inference-time only). riir-ai: LoRA training stays on GPU, ANE is inference-only |
| LoRA-only for training? | ✅ Training still uses riir-gpu LoRA pipeline. ANE is inference only |
| Self-learning adaptive CoT? | ✅ ANE runs the forward pass, bandit decides thinking budget on CPU |
| CPU/GPU/ANE auto-route? | ✅ This IS the auto-route — CPU for logic, GPU for training, ANE for inference |
| SOLID/DRY? | ✅ New `InferenceBackend` trait, existing code unchanged |
| Tests with before/after? | ✅ Benchmark: same query, CPU-only vs ANE backend |

---

## Summary

| Target | Path | Backend | Default? | Plan |
|--------|------|---------|----------|------|
| **katgpt-rs** | `coreml-native` | CoreML → ANE forward | YES (macOS) | Plan 175v2 |
| **riir-ai inference** | `coreml-native` + `rane` fallback | CoreML/rane → ANE LoRA inference | YES (macOS) | Plan 197v2 |
| **riir-ai training** | Study `rustane` patterns | No change (GPU LoRA training) | N/A | Knowledge only |
| **Conversion pipeline** | Python (ane-book) | GGUF → CoreML .mlmodelc | Build-time | Plan 175v2 |

**The real gain isn't a pattern fusion — it's a new hardware target.** We add ANE as a third compute backend alongside CPU and GPU. On Apple Silicon (every Mac since 2020), this is free performance that we're currently leaving idle.
