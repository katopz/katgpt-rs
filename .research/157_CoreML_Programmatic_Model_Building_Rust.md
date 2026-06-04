# Research: CoreML Programmatic Model Building in Rust for ANE Inference

**Date:** 2026-06-04
**Context:** katgpt-rs is "modelless" — weights live in-memory as `TransformerWeights` structs. No `.mlmodelc` file exists. We need to compile these weights into a CoreML model at runtime for ANE execution.
**Status:** Research Complete — Actionable Verdict

---

## Executive Summary

**The viable approach is: Build a CoreML protobuf spec in Rust → write to temp file → compile → load → delete temp file.**

There is also a newer in-memory path (`MLModelAsset` with `modelAssetWithSpecificationData:blobMapping:error:`) that avoids temp files entirely, available since macOS 14+.

| Approach | Feasibility | Complexity | ANE Access | Recommended |
|----------|-------------|------------|------------|-------------|
| **1. Protobuf spec → temp file → compile** | ✅ Proven | Medium | ✅ Full | **Primary** |
| **2. Protobuf spec → `MLModelAsset` in-memory** | ✅ Confirmed API exists | Medium-High | ✅ Full | **Best (macOS 14+)** |
| **3. Direct objc CoreML C API model building** | ❌ No programmatic API | Low | N/A | Dead end |
| **4. `candle-core` / `candle-coreml`** | ⚠️ Loading only | Low | ✅ (loading) | Not applicable |
| **5. Construct `.mlmodelc` bundle by hand** | ❌ Undocumented format | Extreme | Unknown | Dead end |
| **6. `gomlx/go-coreml` approach (port to Rust)** | ✅ Proven pattern | High | ✅ Full | Reference implementation |

---

## Approach 1: Direct CoreML Objective-C API via `objc` crate

### What we investigated

Can we call CoreML Objective-C methods directly to construct an `MLModel` from an `MLModelDescription` + weights without a file?

### Finding: **Dead End — No programmatic model building API exists**

The CoreML public API provides:
- `MLModel.modelWithContentsOfURL:configuration:error:` — **loads** compiled models from `.mlmodelc` path
- `MLModel.compileModelAtURL:error:` — **compiles** `.mlmodel` → `.mlmodelc` (requires file on disk)
- `MLModelConfiguration` — sets compute units only (`.All`, `.CPUOnly`, `.CPUAndGPU`, `.CPUAndNeuralEngine`)
- `MLDictionaryFeatureProvider` — provides input data for inference
- `MLMultiArray` — tensor data containers

There is **no public API** to:
- Add layers to a model programmatically
- Set weights on a model after loading
- Construct an `MLModel` from scratch in memory
- Build a neural network graph via ObjC method calls

The `coreml-native` crate (already in katgpt-rs's `Cargo.toml` as an optional dep) wraps exactly these APIs — loading compiled models, prediction, compilation. It cannot build models.

### Verdict: ❌ Not viable. CoreML's public API is load-and-predict only.

---

## Approach 2: CoreML Protobuf Spec (THE WINNER)

### How it works

Every `.mlmodel` file is a serialized protobuf message defined by Apple's public specification. The format is fully documented at [apple.github.io/coremltools/mlmodel](https://apple.github.io/coremltools/mlmodel/index.html).

**This is exactly how Python `coremltools` builds models.** We can replicate the same process in Rust.

### The protobuf format

From [Model.proto](https://github.com/apple/coremltools/blob/master/mlmodel/format/Model.proto):

```protobuf
message Model {
    int32 specificationVersion = 1;
    ModelDescription description = 2;
    oneof Type {
        NeuralNetwork neuralNetwork = 500;
        MILSpec.Program mlProgram = 502;
        // ... other types
    }
}
```

Two model types are relevant:

| Type | Field | Spec Version | ANE Support | Use For |
|------|-------|-------------|-------------|---------|
| `NeuralNetwork` | `neuralNetwork = 500` | 1-5 | Good | Legacy NN layers (conv, linear, activation) |
| `MILSpec.Program` | `mlProgram = 502` | 6+ | Better | Modern ops (matmul, einsum, flexible shapes) |

### NeuralNetwork model type (simpler, spec version 4-5)

From `NeuralNetwork.proto`, the key message is `NeuralNetworkLayer`:

```protobuf
message NeuralNetworkLayer {
    string name = 1;
    string input[...] = 2;
    string output[...] = 3;
    oneof layer {
        ConvolutionParams convolution = 10;
        InnerProductParams innerProduct = 20;  // ← This is linear/dense
        PoolingParams pooling = 30;
        ActivationParams activation = 40;
        // ... softmax, batchnorm, add, multiply, etc.
    }
}
```

For a transformer, we need:
- `InnerProductParams` (linear layers: wq, wk, wv, wo, w1, w2, lm_head)
- `ActivationParams` (ReLU/GELU for MLP)
- `SoftmaxParams` (attention softmax)
- `AddParams` (residual connections)
- `MultiplyParams` (scaling)
- `BatchnormParams` or `LayerNormParams`

**ANE optimization**: Use `ConvolutionParams` with 1×1 kernels instead of `InnerProductParams` — ANE hardware is optimized for convolution and maps 1×1 conv directly to matmul units.

### MILSpec.Program model type (more powerful, spec version 6+)

From `MIL.proto`, the modern approach uses an SSA (Static Single Assignment) program:

```protobuf
message Program {
    map<string, Function> functions = 1;
}

message Function {
    Block block = 1;
}

message Block {
    repeated Operation operations = 1;
    repeated string outputs = 2;
}

message Operation {
    string name = 1;
    string op_type = 2;
    map<string, Value> attributes = 3;
    repeated NamedValueType outputs = 4;
}
```

MIL supports: `matmul`, `softmax`, `gelu`, `layer_norm`, `reshape`, `transpose`, `concat`, `slice_by_index`, `const`, `identity`, and many more.

### Weight embedding

Weights are embedded in the protobuf as `floatValue` repeated fields or as external blob references:

```protobuf
message WeightParams {
    repeated float floatValue = 1;
    repeated int32 intValue = 2;
    bytes rawValue = 3;
    // For large weights, use external file references
}
```

For the `mlProgram` type, weights are stored as `const` operations with tensor values, or in external `weight.bin` files referenced by blob paths.

### Step-by-step: Building the protobuf spec in Rust

1. **Generate Rust protobuf types** from Apple's `.proto` files
   - Clone `apple/coremltools` repo's `mlmodel/format/` directory (30+ `.proto` files)
   - Use `protoc` with `prost` or `protobuf` crate to generate Rust structs
   - OR: Use the `prost-build` crate in a `build.rs` script

2. **Construct the `Model` message**:
   ```rust
   // Pseudocode for what the Rust builder would look like
   let mut model = Model::default();
   model.specification_version = 7;  // iOS 16+, macOS 13+
   
   // Set model description (inputs/outputs)
   model.description = Some(ModelDescription {
       input: vec![
           FeatureDescription {
               name: "input_ids".into(),
               r#type: Some(FeatureType {
                   multi_array_type: Some(ArrayFeatureType {
                       shape: vec![1, block_size as i64],
                       data_type: ArrayFeatureType::DataType::Int32 as i32,
                   }),
               }),
           },
       ],
       output: vec![
           FeatureDescription {
               name: "logits".into(),
               r#type: Some(FeatureType {
                   multi_array_type: Some(ArrayFeatureType {
                       shape: vec![1, vocab_size as i64],
                       data_type: ArrayFeatureType::DataType::Float32 as i32,
                   }),
               }),
           },
       ],
       ..default()
   });
   
   // Build neural network layers from TransformerWeights
   let mut nn = NeuralNetwork::default();
   
   // For each layer in transformer:
   for (i, layer_weights) in weights.layers.iter().enumerate() {
       // Embedding lookup → reshape
       // QKV projection (1×1 conv or innerProduct)
       // Attention scoring
       // Output projection
       // MLP (w1 → activation → w2)
       // Residual adds
   }
   
   model.neural_network = Some(nn);
   ```

3. **Serialize to bytes**:
   ```rust
   let spec_bytes = model.write_to_bytes().unwrap();
   ```

4. **Two loading paths** (see below)

### Loading Path A: Temp file → compile → load (works on all macOS versions)

```rust
// 1. Write spec to temp file
let tmp_dir = tempfile::tempdir()?;
let mlmodel_path = tmp_dir.path().join("model.mlmodel");
std::fs::write(&mlmodel_path, &spec_bytes)?;

// 2. Compile .mlmodel → .mlmodelc
let compiled_path = coreml_native::compile_model(&mlmodel_path)?;

// 3. Load compiled model
let model = coreml_native::Model::load(&compiled_path, ComputeUnits::All)?;

// 4. Temp file cleanup (compiled model is in CoreML's cache)
// tmp_dir goes out of scope, temp files deleted
```

### Loading Path B: `MLModelAsset` in-memory (macOS 14+)

Apple added `MLModelAsset` with in-memory loading. The Objective-C selector is:

```
modelAssetWithSpecificationData:(NSData *)specificationData
                    blobMapping:(NSDictionary<NSURL *, NSData *> *)blobMapping
                           error:(NSError **)error
```

This is confirmed in:
- [Apple docs: `MLModelAsset`](https://developer.apple.com/documentation/coreml/mlmodelasset)
- [Microsoft .NET bindings](https://learn.microsoft.com/en-us/dotnet/api/coreml.mlmodelasset.create) — shows all three overloads
- [coremltools `MLModelAsset.from_memory`](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html) — Python wrapper
- [dotnet/macios bindings](https://github.com/dotnet/macios/wiki/CoreML-macOS-xcode16.0-b2) — confirms `blobMapping` variant

In Rust via `coreml-native` / `objc2`:

```rust
// Conceptual — would need to be added to coreml-native or called via objc2
let spec_data = NSData::dataWithBytes_length(spec_bytes.as_ptr(), spec_bytes.len());
let asset = MLModelAsset::modelAssetWithSpecificationData_blobMapping_error(
    &spec_data,
    &blob_mapping,  // NSDictionary<NSURL*, NSData*>
);
// Then load model from asset
let model = MLModel::modelWithAsset_configuration_error(&asset, &config);
```

The `coreml-native` crate's README already mentions `load_from_bytes`:
```rust
// Load from in-memory bytes (macOS 14.4+)
let spec_bytes = std::fs::read("model.mlmodel")?;
let model = Model::load_from_bytes(&spec_bytes, ComputeUnits::All)?
    .block_on()?;
```

**This is already available in the crate we depend on!**

---

## Approach 3: `coremltools` Python Approach (Reference)

### How coremltools builds models

`coremltools` in Python does exactly what we want to do in Rust:

1. **Build a protobuf spec** using Python objects:
   ```python
   import coremltools as ct
   
   # Build using MIL builder (modern approach)
   with ct.ModelType(scope=None, opset_version=...) as model:
       x = ct.mlprogram.ops.coreml.placeholder("x", shape=[1, seq_len])
       w = ct.mlprogram.ops.coreml.const(weight_data)
       y = ct.mlprogram.ops.coreml.matmul(x=x, y=w)
   ```

2. **Serialize and compile**:
   ```python
   model = ct.models.MLModel(spec)  # Compiles spec → MLModel in-memory
   model.save("model.mlpackage")    # Optional: save to disk
   ```

3. **The in-memory path** (`MLModelAsset.from_memory`):
   ```python
   model = MLModel("my_model.mlpackage")
   spec_data = model.get_spec().SerializeToString()
   
   asset = ct.models.model.MLModelAsset.from_memory(
       spec_data=spec_data,
       blob_mapping={"weights/weight.bin": weights_data}
   )
   compiled_model = ct.models.CompiledMLModel.from_asset(asset=asset)
   result = compiled_model.predict({"x": np.array([1.0])})
   ```

### What we can learn from `gomlx/go-coreml`

The [gomlx/go-coreml](https://github.com/gomlx/go-coreml) project is a **complete Go implementation** of what we want to build in Rust:

- **Generated Go protobuf types** from Apple's `.proto` files
- **MIL program builder** with operations: `add`, `mul`, `matmul`, `conv2d`, `relu`, `sigmoid`, `softmax`, `reshape`, `transpose`, `concat`, `gather`, `slice`, `reduce_sum`, `reduce_mean`, `argmax`, etc.
- **Model serialization** to `.mlpackage` format with blob storage for large weights
- **Runtime** for compiling and executing MIL programs
- **Weight blob support** for large models

This proves the approach works end-to-end. We can port this to Rust.

---

## Approach 4: Temporary File Approach

### Assessment: ✅ Fully viable, but suboptimal

The temp file approach is the most battle-tested path:

1. Build protobuf spec in Rust
2. Write `.mlmodel` to a temp file
3. Call `coreml_native::compile_model()` → produces `.mlmodelc`
4. Load with `coreml_native::Model::load()` with `ComputeUnits::All`
5. Delete temp files

**Pros:**
- Works on all macOS versions (10.13+)
- No private API usage
- `coreml_native::compile_model()` is already implemented

**Cons:**
- Disk I/O overhead on every weight change (recompilation)
- Temp file management complexity
- Compilation takes ~100ms-1s depending on model size
- Not suitable for frequent weight updates

### For katgpt-rs specifically

Since weights are generated at runtime and change infrequently (only on `recompile_hint()`), the temp file approach is actually fine. The compilation cost is amortized over many inference calls.

---

## Approach 5: `candle-core` and Other ML Crates

### `candle-coreml`

[`candle-coreml`](https://crates.io/crates/candle-coreml) provides CoreML integration for Candle tensors. However:

- It **loads** existing `.mlmodelc` models
- It does **not** build models programmatically
- It depends on Candle, which katgpt-rs doesn't use
- Not useful for our use case

### `ort` (ONNX Runtime) with CoreML EP

ONNX Runtime has a CoreML execution provider that can use the ANE. However:
- Requires ONNX model format
- Would need ONNX → CoreML conversion at runtime
- Adds a heavy dependency
- Indirect ANE access

### No Rust crate exists that builds CoreML models

Neither `candle-core`, `burn`, `tract`, nor any other Rust ML framework provides CoreML model building. They all focus on:
- Loading pre-compiled `.mlmodelc` files
- Or providing their own compute backends

---

## Approach 6: Construct `.mlmodelc` Bundle by Hand

### What `.mlmodelc` contains

From [machinethink.net/blog/peek-inside-coreml](https://machinethink.net/blog/peek-inside-coreml/), a compiled `.mlmodelc` bundle contains:

```
model.mlmodelc/
├── coremldata.bin              # Model metadata, labels
├── model.espresso.net          # JSON: layer structure, connections
├── model.espresso.shape        # JSON: output sizes per layer
├── model.espresso.weights      # Binary: learned parameters (big file)
└── model/
    └── coremldata.bin          # Additional metadata
```

The `.espresso.*` format is Apple's proprietary internal representation. The JSON in `model.espresso.net` describes layers like:

```json
{
  "storage": "model.espresso.weights",
  "layers": [
    {
      "type": "convolution",
      "name": "conv1",
      "top": "conv1_output",
      "K": 3,
      "stride_x": 2,
      "blob_weights": 3,
      "blob_biases": 1,
      ...
    }
  ]
}
```

### Assessment: ❌ Not viable

- The Espresso format is undocumented and changes between macOS versions
- We would need to reverse-engineer the compilation process
- Any macOS update could break it
- From the securing.pl article, compiled models can even be encrypted
- Apple explicitly compiles `.mlmodel` → `.mlmodelc` using `coremlc`, which is a black box

---

## Recommended Implementation Plan

### Phase 1: Minimal Viable ANE Backend (NeuralNetwork spec, temp file)

```
1. Add prost/prost-build to Cargo.toml
2. Copy .proto files from apple/coremltools/mlmodel/format/
3. Build Model.proto + NeuralNetwork.proto + DataStructures.proto + FeatureTypes.proto
4. Implement a transformer → NeuralNetwork spec builder:
   - For each transformer layer, emit:
     - InnerProduct or Conv2D(1×1) for wq/wk/wv/wo/w1/w2
     - Activation (GELU/ReLU) for MLP
     - Softmax for attention
     - Add for residual connections
   - Embed weights directly into the protobuf WeightParams
5. Serialize to bytes → write temp .mlmodel → compile → load
6. Wire up to AneBackend::compile()
```

### Phase 2: MIL Program spec (more ANE-friendly)

```
1. Add MIL.proto to the protobuf build
2. Build transformer as MIL program:
   - Use matmul op directly (more ANE-friendly than conv hack)
   - Use gelu, layer_norm, softmax ops
   - External weight blobs for large models
3. Use MLModelAsset in-memory loading (macOS 14+)
4. Fall back to temp file on older macOS
```

### Phase 3: Optimization

```
1. ANE-friendly layer restructuring:
   - Fuse RMSNorm into adjacent layers
   - Use 1×1 Conv instead of InnerProduct (ANE hardware preference)
   - Batch multiple token predictions
2. FP16 weight storage in protobuf
3. KV cache as MLState (macOS 15+)
4. Weight update without full recompilation (if possible)
```

### Protobuf generation strategy

**Option A: `prost-build` in `build.rs`** (recommended)
```rust
// build.rs
fn main() {
    prost_build::Config::new()
        .compile_protos(
            &["proto/Model.proto"],
            &["proto/"],
        )
        .unwrap();
}
```

**Option B: Pre-generated Rust types**
- Run `protoc --rust_out=src/proto/ proto/*.proto` once
- Check in the generated code
- Simpler CI, no protobuf dependency at build time

### Key proto files needed

Minimum set for NeuralNetwork approach:
1. `Model.proto` — top-level Model message
2. `NeuralNetwork.proto` — layers, weights
3. `DataStructures.proto` — WeightParams, etc.
4. `FeatureTypes.proto` — input/output type descriptions

For MIL Program approach, add:
5. `MIL.proto` — modern ML Program operations
6. `Parameters.proto` — parameter descriptions

---

## Concrete Code Sketch: Transformer → CoreML Spec

```rust
use crate::transformer::{TransformerWeights, LayerWeights};
use crate::types::Config;

/// Build a CoreML NeuralNetwork protobuf spec from katgpt-rs weights.
pub fn build_nn_spec(
    weights: &TransformerWeights,
    config: &Config,
) -> Vec<u8> {
    let mut model = coreml_spec::Model::default();
    model.specification_version = 7;
    
    // Describe inputs: token embeddings + position
    model.description = build_model_description(config);
    
    // Build the neural network
    let mut nn = coreml_spec::NeuralNetwork::default();
    
    // Token embedding lookup → "input_emb"
    nn.layers.push(embedding_lookup_layer(&weights.wte, config));
    
    // Position embedding add → "pos_emb"
    nn.layers.push(position_embedding_layer(&weights.wpe, config));
    
    // For each transformer layer
    for (i, lw) in weights.layers.iter().enumerate() {
        let prefix = format!("layer{}", i);
        
        // QKV projections (use 1x1 conv for ANE optimization)
        nn.layers.push(conv1x1_layer(
            &format!("{prefix}_wq"), lw.attn_wq.as_slice(), config));
        nn.layers.push(conv1x1_layer(
            &format!("{prefix}_wk"), lw.attn_wk.as_slice(), config));
        nn.layers.push(conv1x1_layer(
            &format!("{prefix}_wv"), lw.attn_wv.as_slice(), config));
        
        // Attention scoring (Q·K^T / √d, softmax, ·V)
        nn.layers.push(attention_layer(&format!("{prefix}_attn"), config));
        
        // Output projection
        nn.layers.push(conv1x1_layer(
            &format!("{prefix}_wo"), lw.attn_wo.as_slice(), config));
        
        // Residual add
        nn.layers.push(add_layer(&format!("{prefix}_res1")));
        
        // MLP
        nn.layers.push(conv1x1_layer(
            &format!("{prefix}_w1"), lw.mlp_w1.as_slice(), config));
        nn.layers.push(gelu_layer(&format!("{prefix}_gelu")));
        nn.layers.push(conv1x1_layer(
            &format!("{prefix}_w2"), lw.mlp_w2.as_slice(), config));
        
        // Residual add
        nn.layers.push(add_layer(&format!("{prefix}_res2")));
    }
    
    // LM head
    nn.layers.push(conv1x1_layer("lm_head", weights.lm_head.as_slice(), config));
    
    model.r#type = Some(coreml_spec::model::Type::NeuralNetwork(nn));
    
    model.write_to_bytes().unwrap()
}
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ANE doesn't support our layer combo | Medium | High — falls back to CPU | Use ANE-friendly ops (1×1 conv), test residency |
| Protobuf format changes between versions | Low | Medium — model won't compile | Pin spec version, test on target macOS |
| Compilation time > 1s | Medium | Low — only on recompile | Cache compiled models, async compilation |
| Temp file approach has I/O overhead | Low | Low — amortized | Use in-memory `MLModelAsset` on macOS 14+ |
| Weight embedding makes protobuf huge | Medium | Medium — memory pressure | Use external blob files for large weights |
| Protobuf codegen complexity | Medium | Low — one-time setup | Use `prost-build`, copy Go project's approach |

---

## References

- [CoreML Model Format Specification](https://apple.github.io/coremltools/mlmodel/index.html)
- [Model.proto source](https://github.com/apple/coremltools/blob/master/mlmodel/format/Model.proto)
- [NeuralNetwork.proto source](https://github.com/apple/coremltools/blob/master/mlmodel/format/NeuralNetwork.proto)
- [MIL.proto source](https://github.com/apple/coremltools/blob/main/mlmodel/format/MIL.proto)
- [MLModelAsset Apple docs](https://developer.apple.com/documentation/coreml/mlmodelasset)
- [MLModelAsset.Create (.NET bindings — shows selectors)](https://learn.microsoft.com/en-us/dotnet/api/coreml.mlmodelasset.create)
- [coremltools MLModelAsset.from_memory](https://apple.github.io/coremltools/docs-guides/source/mlmodel-utilities.html)
- [gomlx/go-coreml — Full Go implementation of programmatic model building](https://github.com/gomlx/go-coreml)
- [coreml-native Rust crate](https://crates.io/crates/coreml-native) — already in katgpt-rs Cargo.toml
- [A peek inside CoreML — machinethink.net](https://machinethink.net/blog/peek-inside-coreml/)
- [From .mlmodel to encrypted .mlmodelc — securing.pl](https://www.securing.pl/en/from-mlmodel-to-mlmodelc-how-apple-encrypts-and-delivers-ml-models/)
- [ANE Compute Backend Verdict (katgpt-rs research #155)](157_CoreML_Programmatic_Model_Building_Rust.md)
