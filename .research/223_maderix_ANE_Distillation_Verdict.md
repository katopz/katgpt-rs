# Research 223: maderix/ANE Distillation — Modelless Fusion Verdict

**Date:** 2026-06-12
**Source:** [maderix/ANE](https://github.com/maderix/ANE) — ANE training of Stories110M (109M params)
**Related:** Research 155 (ANE Backend), Research 157 (CoreML Programmatic), Plan 175 (ANE-Inspired DDTree), Plan 176 (GPU/ANE Offload)
**Status:** GOAT Verdict

---

## What maderix/ANE Does

Trains a 109M-parameter Llama2 transformer (Stories110M) **directly on Apple Neural Engine** using private ANE APIs via Objective-C:

- **6 kernel types per layer**: fwdAttn, fwdFFN, ffnBwd, sdpaBwd1, sdpaBwd2, qkvBwd
- **72 ANE kernels** per compile (60 weight-bearing, 12 weight-free)
- **ANE forward + CPU backward hybrid**: ANE does forward pass + gradient propagation, CPU does weight gradient accumulation via async cblas_sgemm
- **Performance**: 107ms/step (9.6ms ANE eval, 4.1ms IO, 9.1ms classifier, 14.4ms cross-entropy, 0.1ms RMSNorm)
- **Compile budget**: ~119 compiles per process, uses `exec()` restart with checkpoint/resume

### Key Engineering Techniques

| Technique | What | Why It Matters |
|-----------|------|---------------|
| **1×1 conv = matmul** | Linear layers expressed as Conv2d(1×1) | ANE hardware is optimized for convolution |
| **FP16 weight blobs** | 128-byte header + FP16 data | Zero-copy IOSurface transfer to ANE |
| **Concat-tap pattern** | Forward kernels output result + intermediates (Q,K,V,xnorm) in single dispatch | Eliminates redundant ANE dispatches for backward |
| **NEON fp16↔fp32** | ARM NEON intrinsics for IOSurface conversion | Vectorized 8-wide conversion |
| **vDSP cross-entropy** | `vDSP_mtrans` + `vvexpf` + `vDSP_sve` | 8× faster than scalar |
| **Async weight gradients** | cblas_sgemm on GCD background queue | Overlapped with ANE eval |
| **exec() restart** | Process restart to clear compile budget | Checkpoint/resume transparent |
| **SDPA causal mask workaround** | ANE ignores attn_mask → decompose into Q@K^T (ANE) + mask+softmax (CPU) + scores@V (ANE) | Critical for attention on ANE |
| **Channel-first layout** | [DIM, SEQ] instead of [SEQ, DIM] | ANE spatial dimension alignment |
| **MIL program generation** | String-building of MIL text for each kernel | Runtime kernel compilation from weights |

### ANE Gotchas (from maderix)

- ANE compiler fails on `rsqrt`/`sqrt` after `reduce` ops — use `pow(-0.5)` workaround
- `dim` must be divisible by 128 for ANE efficiency
- Per-ANE-dispatch overhead: ~0.095ms (XPC + IOKit)
- Compile budget ~119 per process (requires restart)
- ANE ignores `attn_mask` in SDPA — must decompose attention manually

---

## Distillation: What's New vs Our Existing ANE Work

| Aspect | What We Have (Plans 155/176/197) | What maderix Adds | Novel? |
|--------|----------------------------------|-------------------|--------|
| ANE inference | ✅ CoreML + rane paths | ✅ Same (MIL approach) | No |
| ANE training | ❌ Inference only | ✅ Full training at 109M | **YES** |
| MIL kernel generation | Research 157 (protobuf spec) | String-building MIL text directly | **YES — simpler** |
| Concat-tap pattern | Not used | Forward saves all intermediates | **YES** |
| Async gradient accumulation | Not used | GCD queue overlapped with ANE | **YES** |
| Checkpoint/resume | Not ANE-specific | exec() restart pattern | Minor |
| NEON fp16 conversion | Not in our ANE path | Vectorized IOSurface transfer | Minor |

### The Three New Insights

1. **ANE training is real and works at scale** — 109M params, 107ms/step. Our game LoRA (rank 8 on microGPT = ~256 params) would be trivially fast.

2. **MIL string generation is simpler than protobuf spec** — maderix builds MIL text via `NSMutableString` appendFormat, compiles via `_ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:`. This is the same `_ANEInMemoryModelDescriptor` we explored in Research 157, but maderix proves it works at scale.

3. **The concat-tap pattern eliminates redundant dispatches** — Instead of separate ANE calls for forward + saving intermediates, one call outputs everything. This reduces ANE dispatch count by 2-3× per layer.

---

## Fusion Ideas — Modelless First

### Fusion 1: ANE-Latent NPC Brain Compute ⭐ GOAT

**What**: Move NPC "think brain" ops (sense reconstruction, emotion projection, zone attention) from CPU SIMD to ANE batch compute.

**Why creative**: maderix uses ANE for transformer matmuls, but ANE is good at ANY regular matmul pattern. Our NPC brain does:
- `SenseOctree` reconstruction: `[6×8] × [8]` matvec (SIMD, ~45ns/tick)
- `EmotionProjection`: dot-product + sigmoid (SIMD)
- `ZoneAttention`: dot-product + sigmoid (SIMD)
- All three are fixed-size, batch-friendly, matmul-heavy → **ANE sweet spot**

**For 1000 NPCs × 20Hz = 20K evaluations/sec:**
- CPU SIMD: 20K × ~200ns = 4ms/sec (small but contested with DDTree+WASM)
- ANE batch: batch 1000 NPCs into one dispatch → ~0.1ms, CPU free

**Per 003 verdict:**
| Question | Answer |
|----------|--------|
| Engine or fuel? | **Engine** — compute routing, no proprietary data |
| Ferrari, no gas? | Yes — works with any NPC brain weights |
| Hurts existing paths? | No — feature-gated, SIMD fallback |
| On by default? | After GOAT proof |
| Modelless? | ✅ No LLM, no training, inference-time only |
| CPU/GPU/ANE auto-route? | ✅ Extends TriggerGate to NPC compute |
| SOLID/DRY? | ✅ New `NpcBrainBackend` trait impl |
| Tests with before/after? | ✅ Benchmark: NPC tick CPU vs ANE batch |

**GOAT VERDICT: GAIN — Promote to Plan 254**

### Fusion 2: MIL-as-Runtime-Compute-Pipeline

**What**: Generate MIL text at runtime from `TransformerWeights` structs, compile via `_ANEInMemoryModelDescriptor`, no .mlmodelc file needed.

**Why**: Truly modelless — weights live in-memory, compute pipeline generated on-the-fly. Eliminates file dependency for ANE path.

**Per 003 verdict:**
| Question | Answer |
|----------|--------|
| Engine or fuel? | Engine |
| Ferrari, no gas? | Yes |
| Hurts existing paths? | No — optional feature |
| On by default? | **NO** — uses private APIs, risky for MIT |
| Modelless? | ✅ |
| SOLID/DRY? | ✅ |
| Tests? | Needs benchmarks |

**VERDICT: GAIN — **UNBLOCKED** by coremltools public API (Research 224). Use `mb.program` instead of private `_ANEInMemoryModelDescriptor`. Updated in Issue 004.

### Fusion 3: Concat-Tap for Existing ANE Inference

**What**: Modify our existing ANE inference backend to use concat-tap pattern — save Q,K,V intermediates during forward pass for potential reuse (KV cache warm, speculative verification).

**Per 003 verdict:**
| Question | Answer |
|----------|--------|
| Engine or fuel? | Engine |
| Gain vs effort? | LOW — our microGPT is 1-layer, concat-tap saves ~1 dispatch |
| Modelless? | ✅ |
| Tests? | Easy |

**VERDICT: MARGINAL for microGPT scale. GAIN when models scale to 12+ layers. Create issue for future.**

### Fusion 4: Async Pattern for Bandit + DDTree

**What**: Apply maderix's async cblas pattern — overlap DDTree node expansion with bandit arm evaluation on background threads.

**Per 003 verdict:**
| Question | Answer |
|----------|--------|
| Already done? | Partially — Plan 175 soft-route bandit, Plan 176 trigger gate |
| New gain? | MARGINAL — our async patterns are adequate |

**VERDICT: DEMOTE — already captured in existing plans**

### Fusion 5: NEON FP16 Conversion for ANE Backend

**What**: Adopt maderix's NEON fp16↔fp32 conversion for our ANE IOSurface data transfer path.

**Per 003 verdict:**
| Question | Answer |
|----------|--------|
| Already done? | Our CoreML path handles fp16 internally |
| New gain? | Only for `rane` direct path |

**VERDICT: MARGINAL — fold into existing `ane_direct` path when implemented**

---

## Summary Verdict

| Fusion | Target | Verdict | Action |
|--------|--------|---------|--------|
| **1: ANE-Latent NPC** | katgpt-rs | ⭐ GOAT | **Plan 254** |
| **2: MIL Runtime** | katgpt-rs | GAIN (experimental) | Issue for future |
| **2: Concat-Tap** | katgpt-rs/riir-ai | MARGINAL now | Issue for larger models |
| **4: Async Bandit** | katgpt-rs | DEMOTE | Already done |
| **5: NEON fp16** | katgpt-rs | MARGINAL | Fold into ane_direct |

**One plan created: Plan 255 (ANE-Latent NPC Brain Compute) for katgpt-rs.**
**Two issues created: MIL Runtime + Concat-Tap for future.**

---

## References

- [maderix/ANE](https://github.com/maderix/ANE) — ANE training at 109M params
- [maderix/ANE training/README.md](https://github.com/maderix/ANE/blob/master/training/README.md) — Architecture + performance details
- `stories_mil.h` — MIL program generators for all 6 ANE kernel types
- `stories_io.h` — IOSurface helpers, NEON conversion, kernel compile/eval
- `stories_cpu_ops.h` — vDSP RMSNorm, cross-entropy, Adam, embedding ops
- `train_large.m` — Main 12-layer training loop with checkpoint/resume
- Research 155 — Our ANE Compute Backend Verdict
- Research 157 — CoreML Programmatic Model Building
- Plan 175 — ANE-Inspired DDTree Improvements (complete)
- Plan 176 — Runtime GPU/ANE Offload with Trigger Gate (complete)
