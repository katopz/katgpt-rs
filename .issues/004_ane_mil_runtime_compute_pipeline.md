# Issue 004: ANE CoreML Model Generation Pipeline

**Source:** Research 223 (maderix/ANE Distillation) + Research 224 (coremltools Public API)
**Status:** Unblocked
**Priority:** Medium (Plan 254 Part 2 dependency)
**Previously Blocked On:** ~~Private API stability testing~~ → **RESOLVED** by coremltools public API

## What

Generate CoreML NPC brain models using `coremltools` `mb.program` public API — replacing the original maderix private MIL string-building approach.

## Why

- Issue was previously blocked on private API (`_ANEInMemoryModelDescriptor`) stability risk
- **coremltools** (Research 224) provides the equivalent functionality via public, stable, Apache 2.0 API
- `mb.program` builds MIL IR → graph passes → `.mlpackage` → `.mlmodelc` — the standard CoreML pipeline
- No private API dependency, no macOS version breakage risk

## Approach (Updated)

### Public API Path (Recommended)

```python
# scripts/generate_npc_brain_model.py
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(BATCH, 8), dtype=np.float32),  # hla_state
        mb.TensorSpec(shape=(BATCH, 6, 8), dtype=np.float32),  # module inputs (ternary→float)
        mb.TensorSpec(shape=(BATCH, 8), dtype=np.float32),  # emotion direction
        mb.TensorSpec(shape=(BATCH, 8), dtype=np.float32),  # zone direction
    ],
    opset_version=ct.target.iOS18,
)
def npc_brain(hla_state, module_inputs, emotion_dir, zone_dir):
    # Op 1: Sense projection — [B, 6, 8] × [B, 8, 1] → sigmoid → [B, 6]
    sense = mb.matmul(x=module_inputs, y=hla_state_expand)  # needs reshape
    sense = mb.sigmoid(x=sense)
    
    # Op 2: Emotion dot-product + sigmoid → [B, 1]
    emotion = mb.reduce_sum(x=mb.mul(x=hla_state, y=emotion_dir), axes=[1])
    emotion = mb.sigmoid(x=emotion)
    
    # Op 3: Zone dot-product + sigmoid → [B, 1]
    zone = mb.reduce_sum(x=mb.mul(x=hla_state, y=zone_dir), axes=[1])
    zone = mb.sigmoid(x=zone)
    
    return sense, emotion, zone

model = ct.convert(npc_brain, convert_to="mlprogram",
                   minimum_deployment_target=ct.target.iOS18)
model.save("npc_brain.mlpackage")
```

### ANE Verification

```python
# Verify ANE placement after compilation
from coremltools.models.compute_plan import MLComputePlan
plan = MLComputePlan.load_from_path(model.get_compiled_model_path(),
                                     compute_units=ct.ComputeUnit.CPU_AND_NE)
# Check each op lands on NE
```

### Quantization

```python
from coremltools.optimize.coreml import LinearQuantizer, OptimizationConfig
config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric",
                                                                    weight_threshold=0))
quantizer = LinearQuantizer(config)
quantized_model = quantizer.compile(model)
```

## Tasks

- [x] Create `scripts/generate_npc_brain_model.py` using `mb.program`
- [x] Implement ternary-to-float weight conversion (lossless: -1/0/+1 → f32)
- [x] 3 fused ops: sense matmul + sigmoid, emotion dot + sigmoid, zone dot + sigmoid
- [x] `TernaryDir` / `SenseModule` Python mirrors matching Rust `project()` semantics
- [x] Weight binary export (`npc_brain_weights.bin`) for Rust-side verification
- [x] INT8 per-tensor quantization via `coremltools.optimize.coreml.LinearQuantizer`
- [x] ANE placement verification via `MLComputePlan`
- [x] Graceful error handling for Python 3.13+ missing native extensions
- [-] **Blocked**: Generate `npc_brain.mlpackage` — requires Python 3.12 for BlobWriter
- [-] Verify ANE placement (all ops on NE) — requires .mlpackage (blocked by above)
- [-] Verify FP16 I/O compatibility (ANE runs FP16 natively) (blocked by above)
- [-] Write test: generated model output matches `CpuTernaryBackend` (cosine ≥ 0.99) (blocked by above)

## Stretch Goals (from Research 224)

- [-] Multifunction model: share weights between perception/emotion/zone (iOS 18+) — stretch
- [-] Stateful model: persistent NPC emotion accumulators via `read_state`/`coreml_update_state` — stretch
- [-] Palettized ternary weights: 1-bit `constexpr_lut_to_dense` compression — stretch

## Previous Approach (Superseded)

~~Generate MIL text at runtime via `_ANEInMemoryModelDescriptor` (private API)~~
~~Blocked on private API stability across macOS versions~~

This approach is superseded by the public `coremltools` `mb.program` API which provides the same MIL IR access without private API risk.
