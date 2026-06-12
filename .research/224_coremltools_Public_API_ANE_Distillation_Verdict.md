# Research 224: coremltools Public API ANE Distillation — Blocker Resolution Verdict

**Date:** 2026-06-12
**Source:** [apple/coremltools](https://github.com/apple/coremltools) — Apple's official CoreML conversion & optimization toolkit
**Related:** Research 223 (maderix/ANE), Research 155 (ANE Backend), Research 157 (CoreML Programmatic), Issue 004 (ANE MIL Runtime), Plan 255 (ANE-Latent NPC Brain)
**Status:** GOAT Verdict — **Blocker Removed**

---

## Executive Summary

**coremltools provides public, stable APIs that eliminate the "private API stability" blocker on Issue 004.**

The previous maderix/ANE distillation (Research 223) identified MIL Runtime Compute Pipeline as GAIN but deferred it due to private API risk (`_ANEInMemoryModelDescriptor`). coremltools is Apple's **official open-source** toolkit (Apache 2.0) that exposes the full MIL IR, programmatic model building, quantization, and ANE placement introspection — all via public APIs.

**Impact on existing verdicts:**

| Item | Previous Status | New Status |
|------|----------------|------------|
| Issue 004 (ANE MIL Runtime) | Deferred — private API blocker | **UNBLOCKED** — use `mb.program` public API |
| Plan 255 Part 2 | Use maderix MIL string building | **SIMPLIFIED** — use `mb.program` + coremltools |
| Conv2d(1×1) trick | Private MIL kernel generation | `mb.conv` with (1,1) kernel — public API |
| INT8 quantization | Manual weight quantization | `coremltools.optimize.coreml` — public API |
| ANE placement verification | Not available | `MLComputePlan` introspection — public API |
| Stateful NPC models | Not available | `read_state`/`coreml_update_state` (iOS 18+) |

---

## What coremltools Provides

### 1. Programmatic Model Building (`mb.program`)

Build CoreML models from scratch — **no PyTorch/TF dependency**:

```python
from coremltools.converters.mil import Builder as mb

@mb.program(
    input_specs=[mb.TensorSpec(shape=(1, 64), dtype=np.float32)],
    opset_version=ct.target.iOS18,
)
def npc_brain(sensor_input):
    w = mb.const(val=weights)
    hidden = mb.linear(x=sensor_input, weight=w, bias=b)
    hidden = mb.sigmoid(x=hidden)  # sigmoid, not softmax!
    return hidden

model = ct.convert(npc_brain, convert_to="mlprogram",
                   minimum_deployment_target=ct.target.iOS18)
```

**This replaces maderix's private MIL string building with a public API.**

### 2. MIL IR — Full Access via Public API

coremltools exposes the **same MIL IR** that maderix builds via private string generation:

```
Python mb.program → MIL Program (SSA) → Graph Passes → MLProgram → .mlpackage → .mlmodelc
```

- **Core ops**: `mb.conv`, `mb.linear`, `mb.matmul`, `mb.sigmoid`, `mb.tanh`, `mb.reduce_sum`, `mb.select`
- **Op versioning**: iOS 15/16/17/18 progressive — builder picks right version
- **25+ graph passes**: constant folding, dead code elimination, layer norm fusion, gelu fusion, fp16 cast

### 3. ANE Placement Introspection

```python
from coremltools.models.compute_plan import MLComputePlan

plan = MLComputePlan.load_from_path(model.get_compiled_model_path(),
                                     compute_units=ct.ComputeUnit.ALL)
for op in program["main"].block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    cost = plan.get_estimated_cost_for_mlprogram_operation(op)
    # usage.preferred_compute_device → CPU/GPU/NE
```

**We can verify ANE placement without private APIs.**

### 4. ComputeUnit Control

| ComputeUnit | Hardware | Min Platform |
|---|---|---|
| `ALL` (default) | CPU + GPU + NE | All |
| `CPU_AND_GPU` | CPU + GPU | All |
| `CPU_ONLY` | CPU | All |
| `CPU_AND_NE` | CPU + Neural Engine | macOS ≥ 13.0 |

`CPU_AND_NE` is exactly what we want for NPC brain — power-efficient, excludes GPU.

### 5. Quantization (Public API)

| Mode | Dtypes | Granularity |
|---|---|---|
| Linear symmetric | int8 [-127, 127], uint8 [0, 254] | Per-tensor or per-channel |
| Linear affine | int8 [-128, 127], uint8 [0, 255] | Per-tensor or per-channel |
| Blockwise (iOS 18+) | int4, int8, uint4, uint8 | Per-block |

### 6. Stateful Models (iOS 18+)

```python
@mb.program(input_specs=[
    mb.TensorSpec((1,), dtype=types.fp16),
    mb.StateTensorSpec((1,), dtype=types.fp16),
])
def prog(x, accumulator_state):
    current = mb.read_state(accumulator_state)
    updated = mb.add(x=x, y=current)
    result = mb.coreml_update_state(accumulator_state, updated)
    return updated
```

**This is relevant for NPC emotion accumulators and memory.**

### 7. Multifunction Models (iOS 18+)

Multiple functions sharing weights in one `.mlpackage`:
- `perception()` — sense reconstruction
- `emotion()` — emotion dot-product + sigmoid
- `zone()` — zone attention

---

## ANE Gotchas (coremltools)

- **FP16 hang on ANE**: Known Core ML framework bug with `compute_units=ALL` — always test with `CPU_AND_NE` fallback
- **32-byte alignment**: State tensor dimensions must be 32-byte aligned for ANE
- **No direct ANE placement control**: ANE placement is automatic by Core ML compiler; `MLComputePlan` only observes, not controls
- **Dim divisible by 128**: Same as maderix finding — ANE prefers dimensions divisible by 128

---

## Updated GOAT Decisions

### Issue 004 (ANE MIL Runtime) — BLOCKER REMOVED ⭐

| Previous | New |
|----------|-----|
| Blocked on private API `_ANEInMemoryModelDescriptor` | **Unblocked** — use `mb.program` public API |
| Risk: API could break across macOS versions | **Stable** — Apple-maintained, Apache 2.0 |
| Needs: private API stability survey | **Not needed** — public API with semver |

**Recommendation**: Update Issue 004 from "Deferred" to "Unblocked". The `mb.program` approach is the public-API equivalent of maderix's private MIL string building.

### Plan 255 Part 2 — SIMPLIFIED

| Previous | New |
|----------|-----|
| Manual MIL string building (maderix pattern) | `mb.program` + `mb.linear` + `mb.sigmoid` |
| Conv2d(1×1) via custom MIL kernel | `mb.conv` with (1,1) kernel — public API |
| Manual INT8 weight quantization | `coremltools.optimize.coreml` |
| No ANE placement verification | `MLComputePlan` introspection |
| Single function model | Multifunction model (perception + emotion + zone) |

### Key Constraint: Ternary Bit-Plane Projection

The ternary-to-float conversion concern from Plan 255 Part 1 finding remains:
- `SenseModule::project()` uses ternary bit-plane extraction (not float matmul)
- coremltools doesn't have ternary bit ops in its standard op set
- **Solution**: Convert ternary weights to float at model generation time in `generate_npc_brain_model.py`
- This is lossless for the boolean extraction (0/1/-1 → float) but changes the compute path

---

## New Fusions from coremltools

### Fusion 6: Stateful NPC Emotion Accumulator — GAIN (Future)

**What**: Use `StateTensorSpec` + `read_state`/`coreml_update_state` for persistent NPC emotion accumulators on ANE.

**Why**: Currently emotion accumulators are `f32` values in Rust. Stateful CoreML models could maintain this state on ANE, eliminating round-trip encoding/decoding.

**Per 003 verdict:**
| Question | Answer |
|----------|--------|
| Engine or fuel? | Engine |
| On by default? | After Plan 255 completes |
| Modelless? | ✅ State is NPC-specific, weights are engine |
| Tests? | Easy — compare stateful vs stateless output |

**VERDICT: GAIN — fold into Plan 255 Part 3 as stretch goal**

### Fusion 7: Multifunction NPC Brain — GAIN (Future)

**What**: Use iOS 18+ multifunction models to share weights between perception/emotion/zone functions.

**Why**: Currently Plan 255 assumes 3 separate ANE dispatches. Multifunction models share weights → single dispatch, lower overhead.

**VERDICT: GAIN — fold into Plan 255 Part 2 as stretch goal, requires iOS 18+ / macOS 15+**

### Fusion 8: Palettized Ternary Weights — GAIN (Future)

**What**: Use `constexpr_lut_to_dense` with 1-bit palettization to represent ternary weights (3 values: -1, 0, +1) in compressed form.

**Why**: Our ternary weights have only 3 values — 1-bit palettization is near-perfect compression.

**VERDICT: GAIN — explore after basic ANE path works**

---

## Revised Action Items

### Immediate
- [x] Create Research 224 — coremltools distillation verdict
- [x] Update Issue 004 — remove "private API" blocker, update approach
- [x] Update Plan 255 Part 2 — use `mb.program` instead of MIL string building

### Plan 255 Updates (Part 2 Revised)
- [x] `scripts/generate_npc_brain_model.py` — use `mb.program` with `mb.matmul`/`mb.sigmoid`
- [x] Ternary-to-float weight conversion in Python script
- [x] INT8 quantization via `coremltools.optimize.coreml`
- [x] ANE placement verification via `MLComputePlan`
- [ ] Output `npc_brain.mlpackage` — blocked on Python 3.12 for BlobWriter serialization

---

## References

- [apple/coremltools](https://github.com/apple/coremltools) — Official CoreML conversion toolkit
- coremltools MIL IR: `coremltools/converters/mil/mil/`
- coremltools Model Builder: `coremltools/converters/mil/mil/builder.py`
- coremltools Optimization: `coremltools/optimize/coreml/`
- coremltools Compute Plan: `coremltools/models/compute_plan.py`
- Research 155 — ANE Compute Backend Verdict
- Research 157 — CoreML Programmatic Model Building
- Research 223 — maderix/ANE Distillation
- Plan 255 — ANE-Latent NPC Brain Compute
- Issue 004 — ANE MIL Runtime Compute Pipeline

---

TL;DR: **coremltools public API removes the private API blocker from Issue 004.** Use `mb.program` for model building, `mb.linear` + `mb.sigmoid` for NPC brain ops, `coremltools.optimize.coreml` for INT8 quantization, and `MLComputePlan` for ANE placement verification. Three additional fusions identified (stateful accumulators, multifunction, palettized ternary) — all fold into Plan 255 as stretch goals.
