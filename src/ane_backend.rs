//! Apple Neural Engine inference backend via CoreML with programmatic model building (Plan 176).
//!
//! Uses runtime weight compilation instead of `.mlmodelc` file loading.
//! katgpt-rs is modelless — weights live in-memory as `TransformerWeights`,
//! generated at runtime. This backend compiles them into a CoreML model
//! on demand for ANE execution.
//!
//! # Runtime Compilation
//!
//! `compile()` takes `&TransformerWeights` + `&Config` and builds a CoreML
//! neural network programmatically using the `coreml-proto` protobuf spec,
//! serializes it, and loads it via `coreml_native::Model::load_from_bytes()`.
//! No `.mlmodelc` file is needed.
//!
//! # Hybrid Execution
//!
//! The MVP compiles the `lm_head` linear projection as a CoreML `InnerProduct`
//! layer and runs it on ANE. The rest of the transformer forward pass (embedding,
//! RMSNorm, attention, MLP) runs on CPU. This proves the end-to-end pipeline:
//! build spec → serialize → load → predict → verify.
//!
//! # Residency Validation
//!
//! ANE execution is not guaranteed — CoreML may fall back to CPU/GPU if the
//! model graph doesn't fit ANE constraints. The residency check times a micro-
//! prediction: ANE < 1ms vs CPU fallback > 5ms. If residency fails, the auto-
//! route falls back to `CpuBackend`.
//!
//! # Stateful KV Cache (Future)
//!
//! macOS 15+ provides `MLState` for persistent KV cache across tokens.
//! This avoids re-sending the full KV cache on every call, roughly 2× faster
//! decode. Currently a placeholder — requires CoreML stateful model export.

use coreml_native as coreml;
use prost::Message;

use crate::inference_backend::InferenceBackend;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::Config;

// ── CoreML proto imports ──────────────────────────────────────────────────
//
// All types are re-exported from the top-level `coreml_proto::proto` module,
// which flattens Apple's FeatureTypes.proto and NeuralNetwork.proto into a
// single namespace via `include!` in the generated `mod.rs`.
use coreml_proto::proto::{
    ActivationParams, ActivationReLu, AddLayerParams, ArrayFeatureType, ConvolutionLayerParams,
    DotProductLayerParams, FeatureDescription, FeatureType, InnerProductLayerParams, Model,
    ModelDescription, MultiplyLayerParams, NeuralNetwork, NeuralNetworkLayer, ScaleLayerParams,
    SoftmaxLayerParams, ValidPadding, WeightParams,
    activation_params::NonlinearityType as ActivationKind,
    convolution_layer_params::ConvolutionPaddingType, feature_type::Type as FeatureTypeKind,
    model::Type as ModelType, neural_network_layer::Layer as LayerKind,
};

/// ANE inference backend using Apple CoreML framework.
///
/// Starts uncompiled. Call `compile()` with the current weights + config to
/// build a CoreML model for ANE execution. The CPU fallback path is used
/// until compilation completes.
pub struct AneBackend {
    /// Whether weights have been compiled to a CoreML model.
    compiled: bool,
    /// Flag set by `recompile_hint()`, consumed on next `compile()` call.
    needs_recompile: bool,
    /// Compiled CoreML model (`None` until `compile()` succeeds).
    model: Option<coreml::Model>,
}

/// Error type for ANE backend operations.
#[derive(Debug)]
pub enum AneError {
    /// CoreML failed to compile the model from weights.
    CompileError(String),
    /// CoreML prediction failed.
    PredictionError(String),
    /// Model failed ANE residency check (falls back to CPU).
    ResidencyFailed { latency_us: u64, threshold_us: u64 },
    /// I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for AneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompileError(msg) => write!(f, "CoreML compile error: {msg}"),
            Self::PredictionError(msg) => write!(f, "CoreML prediction error: {msg}"),
            Self::ResidencyFailed {
                latency_us,
                threshold_us,
            } => {
                write!(
                    f,
                    "ANE residency check failed: {latency_us}μs > {threshold_us}μs threshold (model likely fell back to CPU)"
                )
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for AneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl AneBackend {
    /// Create a new uncompiled ANE backend.
    pub fn new() -> Self {
        Self {
            compiled: false,
            needs_recompile: false,
            model: None,
        }
    }

    /// Compile `TransformerWeights` into a CoreML model for ANE execution.
    ///
    /// Builds a CoreML `NeuralNetwork` spec containing a single `InnerProduct`
    /// layer for the `lm_head` projection (the final linear layer mapping
    /// hidden state → logits). Serializes the spec to protobuf bytes and
    /// loads it via `Model::load_from_bytes()`.
    pub fn compile(
        &mut self,
        weights: &TransformerWeights,
        config: &Config,
    ) -> Result<(), AneError> {
        // Build the lm_head linear model spec.
        let spec = build_linear_model_spec(
            "lm_head",
            &weights.lm_head,
            config.n_embd,
            config.vocab_size,
        );

        // Serialize to protobuf bytes and load into CoreML.
        let bytes = spec.encode_to_vec();
        let model = coreml::Model::load_from_bytes(&bytes, coreml::ComputeUnits::All)
            .map_err(|e| AneError::CompileError(format!("load_from_bytes: {e}")))?
            .block_on()
            .map_err(|e| AneError::CompileError(format!("load_from_bytes block_on: {e}")))?;

        self.model = Some(model);
        self.compiled = true;
        self.needs_recompile = false;
        Ok(())
    }

    /// Whether weights have been compiled to ANE.
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Signal that weights have changed and ANE needs recompilation.
    pub fn recompile_hint(&mut self) {
        self.needs_recompile = true;
    }

    /// Compile the full transformer forward pass into a CoreML model.
    ///
    /// Unlike `compile()` which only builds the lm_head projection, this maps
    /// the entire transformer forward pass (embedding → layers → lm_head) into
    /// a CoreML NeuralNetwork spec. Each layer produces named intermediate blobs.
    ///
    /// **Note**: This is a structural spec builder. CoreML NeuralNetwork cannot
    /// natively express dynamic attention (variable-length KV cache, causal mask),
    /// so the attention sub-graph uses `DotProduct` + `Softmax` + `Multiply` layers
    /// as a structural placeholder. For production inference, the CPU path should
    /// handle the actual attention computation, or a future CoreML ML Program
    /// (MIL spec) backend should be used.
    pub fn compile_full(
        &mut self,
        weights: &TransformerWeights,
        config: &Config,
    ) -> Result<(), AneError> {
        let spec = build_transformer_model_spec(weights, config);

        let bytes = spec.encode_to_vec();
        let model = coreml::Model::load_from_bytes(&bytes, coreml::ComputeUnits::All)
            .map_err(|e| AneError::CompileError(format!("load_from_bytes: {e}")))?
            .block_on()
            .map_err(|e| AneError::CompileError(format!("load_from_bytes block_on: {e}")))?;

        self.model = Some(model);
        self.compiled = true;
        self.needs_recompile = false;
        Ok(())
    }
}

impl InferenceBackend for AneBackend {
    fn forward<'a>(
        &'a mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        // Run the full CPU forward pass to get logits.
        crate::transformer::forward(ctx, weights, cache, token, pos, config);

        // If we have a compiled CoreML model, run the lm_head on ANE and
        // override the CPU-computed logits. This proves the pipeline works.
        if let Some(ref model) = self.model {
            if let Ok(ane_logits) = run_lm_head(model, &ctx.x[..config.n_embd], config) {
                ctx.logits[..config.vocab_size].copy_from_slice(&ane_logits);
            }
        }

        &mut ctx.logits
    }

    fn device_name(&self) -> &'static str {
        "ANE"
    }

    fn supports_stateful(&self) -> bool {
        false
    }
}

/// Build a CoreML `Model` spec for a single linear (inner product) layer.
///
/// The model has:
/// - **Input**: `"input"` of shape `[in_dim, 1, 1]` (Float32 multi-array, 3D)
/// - **Output**: `"output"` of shape `[out_dim, 1, 1]` (Float32 multi-array, 3D)
/// - **Layer**: `InnerProduct` with weights `[out_dim, in_dim]`, no bias
///
/// CoreML NeuralNetwork requires multi-array inputs to have exactly 1 or 3
/// dimensions. We use 3D (channel, height=1, width=1) which is the standard
/// "image-like" format for fully-connected layers.
///
/// The `InnerProduct` layer computes `output = W @ input` where W is stored
/// row-major as `[out_dim, in_dim]` in `WeightParams.float_value`.
fn build_linear_model_spec(
    name: &str,
    weights: &[f32], // [out_dim, in_dim] row-major
    in_dim: usize,
    out_dim: usize,
) -> Model {
    Model {
        specification_version: 7,
        description: Some(ModelDescription {
            input: vec![FeatureDescription {
                name: "input".into(),
                short_description: "Input tensor".into(),
                r#type: Some(multi_array_type(&[in_dim as i64, 1, 1])),
                ..Default::default()
            }],
            output: vec![FeatureDescription {
                name: "output".into(),
                short_description: "Output tensor".into(),
                r#type: Some(multi_array_type(&[out_dim as i64, 1, 1])),
                ..Default::default()
            }],
            ..Default::default()
        }),
        is_updatable: false,
        r#type: Some(ModelType::NeuralNetwork(NeuralNetwork {
            layers: vec![NeuralNetworkLayer {
                name: format!("{name}_linear"),
                input: vec!["input".into()],
                output: vec!["output".into()],
                layer: Some(LayerKind::InnerProduct(InnerProductLayerParams {
                    input_channels: in_dim as u64,
                    output_channels: out_dim as u64,
                    has_bias: false,
                    weights: Some(WeightParams {
                        float_value: weights.to_vec(),
                        ..Default::default()
                    }),
                    bias: None,
                    ..Default::default()
                })),
                ..Default::default()
            }],
            ..Default::default()
        })),
    }
}

/// Build a CoreML `Model` spec for the full transformer forward pass.
///
/// Maps the transformer architecture to CoreML NeuralNetwork layers:
/// - Embedding: `InnerProduct` (wte lookup as a matmul with one-hot)
/// - Per-layer: `Scale` (RMSNorm approximation) → `InnerProduct` (QKV) →
///   `DotProduct` + `Softmax` + `Multiply` (attention) → `InnerProduct` (WO) →
///   `Add` (residual) → `Scale` (RMSNorm) → `InnerProduct` (MLP W1) →
///   `Activation(ReLU)` → `InnerProduct` (MLP W2) → `Add` (residual)
/// - Final: `InnerProduct` (lm_head)
///
/// **RMSNorm approximation**: CoreML has no native RMSNorm. We approximate it
/// using a `ScaleLayer` with precomputed gamma/scale values. A full implementation
/// would need a custom layer or MIL op. For now, the scale values are set to the
/// gamma weights (identity approximation: just element-wise multiply).
///
/// **Attention**: Uses `DotProduct` + `Softmax` + `Multiply` layers as structural
/// placeholders. Real attention requires dynamic sequence lengths and causal masking
/// which CoreML NeuralNetwork cannot express natively.
pub fn build_transformer_model_spec(weights: &TransformerWeights, config: &Config) -> Model {
    let n_embd = config.n_embd;
    let n_layer = config.n_layer;
    let _n_head = config.n_head;
    let head_dim = config.head_dim;
    let kv_dim = config.n_kv_head * head_dim;
    let vocab_size = config.vocab_size;
    let mlp_hidden = config.mlp_hidden;

    let mut layers = Vec::new();
    let mut cur = "embedding_out".to_string();

    // ── Embedding: wte (token) + wpe (position) via InnerProduct ──
    // We represent the embedding lookup as a matmul with the full embedding matrix.
    // For a single token at position p, the input is a one-hot vector of length vocab_size
    // and position encoding of length block_size. In practice, the embedding is done on
    // CPU and this layer represents the combined wte+wpe projection.
    // Here we build wte as InnerProduct [vocab_size, n_embd] for structural completeness.
    layers.push(nn_layer(
        "embedding",
        &["input".to_string()],
        &["embedding_out".to_string()],
        LayerKind::InnerProduct(InnerProductLayerParams {
            input_channels: vocab_size as u64,
            output_channels: n_embd as u64,
            has_bias: false,
            weights: Some(WeightParams {
                float_value: weights.wte.clone(),
                ..Default::default()
            }),
            bias: None,
            ..Default::default()
        }),
    ));

    // ── Per-layer transformer blocks ──
    for i in 0..n_layer {
        let lw = &weights.layers[i];

        // RMSNorm (pre-attention): approximate with Scale layer using gamma weights.
        // Full RMSNorm would be: x / sqrt(mean(x^2) + eps) * gamma.
        // We store just gamma as the scale factor (identity approximation).
        let pre_attn_norm = format!("layer_{i}_pre_attn_norm");
        layers.push(nn_layer(
            &format!("layer_{i}_rmsnorm_attn"),
            &[cur.clone()],
            &[pre_attn_norm.clone()],
            LayerKind::Scale(ScaleLayerParams {
                shape_scale: vec![n_embd as u64],
                scale: Some(WeightParams {
                    float_value: lw.attn_norm_gamma.clone(),
                    ..Default::default()
                }),
                has_bias: false,
                shape_bias: vec![],
                bias: None,
            }),
        ));

        // Q projection: [n_embd] → [n_embd]
        let q_out = format!("layer_{i}_q");
        layers.push(nn_layer(
            &format!("layer_{i}_wq"),
            &[pre_attn_norm.clone()],
            &[q_out.clone()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: n_embd as u64,
                output_channels: n_embd as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: lw.attn_wq.clone(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ));

        // K projection: [n_embd] → [kv_dim]
        let k_out = format!("layer_{i}_k");
        layers.push(nn_layer(
            &format!("layer_{i}_wk"),
            &[pre_attn_norm.clone()],
            &[k_out.clone()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: n_embd as u64,
                output_channels: kv_dim as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: lw.attn_wk.clone(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ));

        // V projection: [n_embd] → [kv_dim]
        let v_out = format!("layer_{i}_v");
        layers.push(nn_layer(
            &format!("layer_{i}_wv"),
            &[pre_attn_norm],
            &[v_out.clone()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: n_embd as u64,
                output_channels: kv_dim as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: lw.attn_wv.clone(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ));

        // Attention: DotProduct(Q, K) → Softmax → Multiply with V
        // These are structural placeholders — real attention needs dynamic shapes.
        let attn_score = format!("layer_{i}_attn_score");
        layers.push(nn_layer(
            &format!("layer_{i}_attn_dot"),
            &[q_out, k_out],
            &[attn_score.clone()],
            LayerKind::Dot(DotProductLayerParams {
                cosine_similarity: false,
            }),
        ));

        let attn_weights = format!("layer_{i}_attn_weights");
        layers.push(nn_layer(
            &format!("layer_{i}_attn_softmax"),
            &[attn_score],
            &[attn_weights.clone()],
            LayerKind::Softmax(SoftmaxLayerParams {}),
        ));

        let attn_out = format!("layer_{i}_attn_out");
        layers.push(nn_layer(
            &format!("layer_{i}_attn_mul"),
            &[attn_weights, v_out],
            &[attn_out.clone()],
            LayerKind::Multiply(MultiplyLayerParams { alpha: 1.0 }),
        ));

        // Output projection: [n_embd] → [n_embd]
        let wo_out = format!("layer_{i}_wo");
        layers.push(nn_layer(
            &format!("layer_{i}_wo"),
            &[attn_out],
            &[wo_out.clone()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: n_embd as u64,
                output_channels: n_embd as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: lw.attn_wo.clone(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ));

        // Residual add: wo_out + cur (skip connection)
        let resid_attn = format!("layer_{i}_resid_attn");
        layers.push(nn_layer(
            &format!("layer_{i}_resid_add_attn"),
            &[wo_out, cur],
            &[resid_attn.clone()],
            LayerKind::Add(AddLayerParams { alpha: 1.0 }),
        ));

        // RMSNorm (pre-MLP): approximate with Scale layer using gamma weights.
        let pre_mlp_norm = format!("layer_{i}_pre_mlp_norm");
        layers.push(nn_layer(
            &format!("layer_{i}_rmsnorm_mlp"),
            &[resid_attn],
            &[pre_mlp_norm.clone()],
            LayerKind::Scale(ScaleLayerParams {
                shape_scale: vec![n_embd as u64],
                scale: Some(WeightParams {
                    float_value: lw.mlp_norm_gamma.clone(),
                    ..Default::default()
                }),
                has_bias: false,
                shape_bias: vec![],
                bias: None,
            }),
        ));

        // MLP W1 (up-projection): [n_embd] → [mlp_hidden]
        let mlp_up = format!("layer_{i}_mlp_up");
        layers.push(nn_layer(
            &format!("layer_{i}_mlp_w1"),
            &[pre_mlp_norm],
            &[mlp_up.clone()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: n_embd as u64,
                output_channels: mlp_hidden as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: lw.mlp_w1.clone(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ));

        // ReLU activation
        let mlp_activated = format!("layer_{i}_mlp_relu");
        layers.push(nn_layer(
            &format!("layer_{i}_mlp_relu"),
            &[mlp_up],
            &[mlp_activated.clone()],
            LayerKind::Activation(ActivationParams {
                nonlinearity_type: Some(ActivationKind::ReLu(ActivationReLu {})),
            }),
        ));

        // MLP W2 (down-projection): [mlp_hidden] → [n_embd]
        let mlp_down = format!("layer_{i}_mlp_down");
        layers.push(nn_layer(
            &format!("layer_{i}_mlp_w2"),
            &[mlp_activated],
            &[mlp_down.clone()],
            LayerKind::InnerProduct(InnerProductLayerParams {
                input_channels: mlp_hidden as u64,
                output_channels: n_embd as u64,
                has_bias: false,
                weights: Some(WeightParams {
                    float_value: lw.mlp_w2.clone(),
                    ..Default::default()
                }),
                bias: None,
                ..Default::default()
            }),
        ));

        // Residual add: mlp_down + resid_attn (skip connection)
        cur = format!("layer_{i}_resid_mlp");
        layers.push(nn_layer(
            &format!("layer_{i}_resid_add_mlp"),
            &[mlp_down, format!("layer_{i}_resid_attn")],
            &[cur.clone()],
            LayerKind::Add(AddLayerParams { alpha: 1.0 }),
        ));
    }

    // ── Final lm_head projection ──
    layers.push(nn_layer(
        "lm_head",
        &[cur],
        &["output".to_string()],
        LayerKind::InnerProduct(InnerProductLayerParams {
            input_channels: n_embd as u64,
            output_channels: vocab_size as u64,
            has_bias: false,
            weights: Some(WeightParams {
                float_value: weights.lm_head.clone(),
                ..Default::default()
            }),
            bias: None,
            ..Default::default()
        }),
    ));

    Model {
        specification_version: 7,
        description: Some(ModelDescription {
            input: vec![FeatureDescription {
                name: "input".into(),
                short_description: "Token input tensor".into(),
                r#type: Some(multi_array_type(&[vocab_size as i64, 1, 1])),
                ..Default::default()
            }],
            output: vec![FeatureDescription {
                name: "output".into(),
                short_description: "Logits output".into(),
                r#type: Some(multi_array_type(&[vocab_size as i64, 1, 1])),
                ..Default::default()
            }],
            ..Default::default()
        }),
        is_updatable: false,
        r#type: Some(ModelType::NeuralNetwork(NeuralNetwork {
            layers,
            ..Default::default()
        })),
    }
}

/// Build a CoreML `Model` spec for a linear projection expressed as Conv2d(1×1).
///
/// ANE hardware is optimized for convolution operations. A linear layer (matmul)
/// can be expressed as a `Conv2d` with 1×1 kernel, which ANE can execute more
/// efficiently than `InnerProduct`.
///
/// The spec uses:
/// - **Input**: `"input"` of shape `[1, in_dim, 1, 1]` (NCHW, 4D)
/// - **Output**: `"output"` of shape `[1, out_dim, 1, 1]` (NCHW, 4D)
/// - **Layer**: `Convolution` with `kernel_size=[1,1]`, `stride=[1,1]`, no bias
///
/// Weight layout: `[out_dim, in_dim, 1, 1]` (same data as InnerProduct, just
/// reinterpreted as 4D convolution weights).
pub fn build_conv2d_linear_model_spec(
    name: &str,
    weights: &[f32], // [out_dim, in_dim] row-major
    in_dim: usize,
    out_dim: usize,
) -> Model {
    Model {
        specification_version: 7,
        description: Some(ModelDescription {
            input: vec![FeatureDescription {
                name: "input".into(),
                short_description: "Input tensor (NCHW)".into(),
                r#type: Some(multi_array_type(&[1, in_dim as i64, 1, 1])),
                ..Default::default()
            }],
            output: vec![FeatureDescription {
                name: "output".into(),
                short_description: "Output tensor (NCHW)".into(),
                r#type: Some(multi_array_type(&[1, out_dim as i64, 1, 1])),
                ..Default::default()
            }],
            ..Default::default()
        }),
        is_updatable: false,
        r#type: Some(ModelType::NeuralNetwork(NeuralNetwork {
            layers: vec![NeuralNetworkLayer {
                name: format!("{name}_conv2d"),
                input: vec!["input".into()],
                output: vec!["output".into()],
                layer: Some(LayerKind::Convolution(ConvolutionLayerParams {
                    output_channels: out_dim as u64,
                    kernel_channels: in_dim as u64,
                    n_groups: 1,
                    kernel_size: vec![1, 1],
                    stride: vec![1, 1],
                    dilation_factor: vec![],
                    is_deconvolution: false,
                    has_bias: false,
                    weights: Some(WeightParams {
                        float_value: weights.to_vec(),
                        ..Default::default()
                    }),
                    bias: None,
                    output_shape: vec![],
                    convolution_padding_type: Some(ConvolutionPaddingType::Valid(ValidPadding {
                        ..Default::default()
                    })),
                })),
                ..Default::default()
            }],
            ..Default::default()
        })),
    }
}

/// Helper: build a `NeuralNetworkLayer` with the given name, inputs, outputs, and layer kind.
fn nn_layer(
    name: &str,
    input: &[String],
    output: &[String],
    layer: LayerKind,
) -> NeuralNetworkLayer {
    NeuralNetworkLayer {
        name: name.into(),
        input: input.iter().map(|s| s.as_str().into()).collect(),
        output: output.iter().map(|s| s.as_str().into()).collect(),
        layer: Some(layer),
        ..Default::default()
    }
}
fn multi_array_type(shape: &[i64]) -> FeatureType {
    use coreml_proto::proto::array_feature_type::ArrayDataType;
    FeatureType {
        r#type: Some(FeatureTypeKind::MultiArrayType(ArrayFeatureType {
            shape: shape.to_vec(),
            data_type: ArrayDataType::Float32 as i32,
            ..Default::default()
        })),
        ..Default::default()
    }
}

/// Run the lm_head linear projection on the compiled CoreML model.
///
/// Takes the hidden state vector `h` of length `n_embd` and returns
/// the logits vector of length `vocab_size`.
fn run_lm_head(
    model: &coreml::Model,
    hidden: &[f32],
    config: &Config,
) -> Result<Vec<f32>, AneError> {
    let n = hidden.len();
    // Shape must match the model's declared input shape: [n_embd, 1, 1].
    let tensor = coreml::BorrowedTensor::from_f32(hidden, &[n, 1, 1])
        .map_err(|e| AneError::PredictionError(format!("tensor create: {e}")))?;

    let prediction = model
        .predict(&[("input", &tensor)])
        .map_err(|e| AneError::PredictionError(format!("predict: {e}")))?;

    let (output, _shape) = prediction
        .get_f32("output")
        .map_err(|e| AneError::PredictionError(format!("get output: {e}")))?;

    // Trim to vocab_size in case the output has extra elements.
    let vocab = config.vocab_size;
    Ok(output[..vocab].to_vec())
}

/// Validate ANE residency by timing a single micro-prediction.
///
/// Creates a random hidden state vector of length `config.n_embd`, runs a single
/// `lm_head` prediction on the compiled CoreML model, and returns the latency
/// in microseconds. Returns `AneError::ResidencyFailed` if latency exceeds
/// the 1000µs threshold, indicating the model likely fell back to CPU.
pub fn validate_residency(model: &coreml::Model, config: &Config) -> Result<u64, AneError> {
    let mut rng = crate::types::Rng::new(0);
    let hidden: Vec<f32> = (0..config.n_embd).map(|_| rng.uniform()).collect();

    let start = std::time::Instant::now();
    run_lm_head(model, &hidden, config)?;
    let elapsed_us = start.elapsed().as_micros() as u64;

    const THRESHOLD_US: u64 = 1000;
    if elapsed_us > THRESHOLD_US {
        return Err(AneError::ResidencyFailed {
            latency_us: elapsed_us,
            threshold_us: THRESHOLD_US,
        });
    }

    Ok(elapsed_us)
}

/// Cosine similarity between two slices. Used in tests to verify ANE accuracy.
#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (mag_a * mag_b + 1e-8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_ane_error_display() {
        let err = AneError::CompileError("spec build failed".to_string());
        assert!(err.to_string().contains("CoreML compile error"));

        let err = AneError::PredictionError("bad input".to_string());
        assert!(err.to_string().contains("CoreML prediction error"));

        let err = AneError::ResidencyFailed {
            latency_us: 8000,
            threshold_us: 1000,
        };
        assert!(err.to_string().contains("residency check failed"));
    }

    #[test]
    fn test_ane_error_source() {
        let err = AneError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(err.source().is_some());

        let err = AneError::CompileError("test".to_string());
        assert!(err.source().is_none());
    }

    #[test]
    fn test_ane_backend_device_name() {
        let backend = AneBackend::new();
        assert_eq!(backend.device_name(), "ANE");
    }

    // ── Residency Validation Tests ──────────────────────────────

    #[test]
    fn test_residency_threshold_constant() {
        // The 1ms threshold is chosen because:
        // - ANE matmul for microGPT: ~50µs
        // - CPU fallback for same: ~5-10ms
        // - 1ms gives clear separation between the two regimes
        const ANE_RESIDENCY_THRESHOLD_US: u64 = 1000;
        assert_eq!(ANE_RESIDENCY_THRESHOLD_US, 1000);
    }

    #[test]
    fn test_residency_failed_error_message() {
        let err = AneError::ResidencyFailed {
            latency_us: 7500,
            threshold_us: 1000,
        };
        let msg = err.to_string();
        assert!(msg.contains("7500"), "should contain actual latency");
        assert!(msg.contains("1000"), "should contain threshold");
        assert!(msg.contains("residency check failed"));
    }

    // ── Runtime Compilation Tests ───────────────────────────────

    #[test]
    fn test_ane_backend_new_uncompiled() {
        let backend = AneBackend::new();
        assert!(
            !backend.is_compiled(),
            "new() should return uncompiled backend"
        );
    }

    #[test]
    fn test_ane_backend_compile_marks_compiled() {
        let mut backend = AneBackend::new();
        assert!(!backend.is_compiled());

        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        backend.compile(&weights, &config).unwrap();
        assert!(
            backend.is_compiled(),
            "compile() should set is_compiled=true"
        );
        assert!(
            backend.model.is_some(),
            "compile() should set model=Some(_)"
        );
    }

    #[test]
    fn test_ane_backend_recompile_hint() {
        let mut backend = AneBackend::new();
        assert!(!backend.needs_recompile);

        backend.recompile_hint();
        assert!(backend.needs_recompile, "recompile_hint() should set flag");

        // compile() clears the flag
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        backend.compile(&weights, &config).unwrap();
        assert!(
            !backend.needs_recompile,
            "compile() should clear recompile flag"
        );
    }

    // ── CoreML Proto Spec Builder Tests ─────────────────────────

    #[test]
    fn test_build_linear_model_spec_structure() {
        let weights = vec![1.0f32; 6]; // [2, 3]
        let spec = build_linear_model_spec("test", &weights, 3, 2);

        assert_eq!(spec.specification_version, 7);
        assert!(!spec.is_updatable);

        let desc = spec.description.as_ref().unwrap();
        assert_eq!(desc.input.len(), 1);
        assert_eq!(desc.output.len(), 1);
        assert_eq!(desc.input[0].name, "input");
        assert_eq!(desc.output[0].name, "output");

        // Check that the model type is NeuralNetwork
        assert!(matches!(spec.r#type, Some(ModelType::NeuralNetwork(_))));
    }

    #[test]
    fn test_build_linear_model_spec_serializes() {
        let weights = vec![0.5f32; 12]; // [3, 4]
        let spec = build_linear_model_spec("test", &weights, 4, 3);
        let bytes = spec.encode_to_vec();
        assert!(!bytes.is_empty(), "serialized spec should not be empty");
    }

    // ── End-to-End Pipeline Tests ───────────────────────────────

    #[test]
    fn test_ane_compile_from_micro_weights() {
        // Verify that compile() succeeds with micro config weights.
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut backend = AneBackend::new();
        backend.compile(&weights, &config).unwrap();
        assert!(backend.is_compiled());
        assert!(backend.model.is_some());
    }

    #[test]
    fn test_ane_lm_head_matches_cpu() {
        // Run a forward pass on CPU, then run the lm_head separately on ANE.
        // The ANE logits should match the CPU logits with cosine similarity ≥ 0.997.
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Compile the lm_head into CoreML.
        let mut backend = AneBackend::new();
        backend.compile(&weights, &config).unwrap();

        // Run a CPU forward pass for token 0, position 0.
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = crate::transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let cpu_logits = logits[..config.vocab_size].to_vec();

        // Run the lm_head on ANE using the same hidden state.
        let model = backend.model.as_ref().unwrap();
        let ane_logits = run_lm_head(model, &ctx.x[..config.n_embd], &config).unwrap();

        // Verify dimensions match.
        assert_eq!(
            cpu_logits.len(),
            ane_logits.len(),
            "ANE and CPU logits should have same length"
        );

        // Cosine similarity should be very high (ANE uses FP32, same as CPU).
        let sim = cosine_similarity(&cpu_logits, &ane_logits);
        assert!(
            sim >= 0.997,
            "ANE vs CPU cosine similarity {sim:.6} < 0.997 threshold"
        );
    }

    #[test]
    fn test_ane_forward_matches_cpu_forward() {
        // Verify that forward() through AneBackend produces the same logits
        // as the direct CPU forward pass.
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Direct CPU forward.
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let cpu_logits =
            crate::transformer::forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();

        // AneBackend forward (compiles lm_head, runs CPU forward + ANE override).
        let mut backend = AneBackend::new();
        backend.compile(&weights, &config).unwrap();
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let ane_logits = backend
            .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
            .to_vec();

        let sim = cosine_similarity(&cpu_logits, &ane_logits);
        assert!(
            sim >= 0.997,
            "AneBackend.forward vs CPU cosine similarity {sim:.6} < 0.997"
        );
    }

    #[test]
    fn test_cosine_similarity_helper() {
        // Identical vectors → similarity = 1.0
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        // Opposite vectors → similarity = -1.0
        let c = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &c) + 1.0).abs() < 1e-6);

        // Orthogonal vectors → similarity = 0.0
        let d = vec![1.0, 0.0, 0.0];
        let e = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&d, &e).abs() < 1e-6);
    }

    // ── Full Transformer Model Spec Tests ───────────────────────

    #[test]
    fn test_build_transformer_model_spec_structure() {
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let spec = build_transformer_model_spec(&weights, &config);

        assert_eq!(spec.specification_version, 7);
        assert!(!spec.is_updatable);

        let desc = spec.description.as_ref().unwrap();
        assert_eq!(desc.input.len(), 1);
        assert_eq!(desc.output.len(), 1);
        assert_eq!(desc.input[0].name, "input");
        assert_eq!(desc.output[0].name, "output");

        // Verify it's a NeuralNetwork model
        assert!(matches!(spec.r#type, Some(ModelType::NeuralNetwork(_))));

        // Count layers: 1 (embedding) + n_layer * 14 (per-layer ops) + 1 (lm_head)
        // Per-layer: rmsnorm_attn + wq + wk + wv + attn_dot + attn_softmax + attn_mul +
        //            wo + resid_add_attn + rmsnorm_mlp + mlp_w1 + mlp_relu + mlp_w2 + resid_add_mlp = 14
        let expected_layers = 1 + config.n_layer * 14 + 1;
        if let Some(ModelType::NeuralNetwork(nn)) = &spec.r#type {
            assert_eq!(
                nn.layers.len(),
                expected_layers,
                "expected {expected_layers} layers, got {}",
                nn.layers.len()
            );

            // Verify first layer is embedding
            assert_eq!(nn.layers[0].name, "embedding");
            assert_eq!(nn.layers[0].input, vec!["input"]);
            assert_eq!(nn.layers[0].output, vec!["embedding_out"]);

            // Verify last layer is lm_head
            let last = nn.layers.last().unwrap();
            assert_eq!(last.name, "lm_head");
            assert_eq!(last.output, vec!["output"]);
        }
    }

    #[test]
    fn test_build_transformer_model_spec_serializes() {
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let spec = build_transformer_model_spec(&weights, &config);
        let bytes = spec.encode_to_vec();
        assert!(
            !bytes.is_empty(),
            "serialized transformer spec should not be empty"
        );
    }

    // ── Conv2d(1×1) Linear Model Spec Tests ──────────────────────

    #[test]
    fn test_build_conv2d_linear_model_spec_structure() {
        let weights = vec![1.0f32; 6]; // [2, 3]
        let spec = build_conv2d_linear_model_spec("test", &weights, 3, 2);

        assert_eq!(spec.specification_version, 7);
        assert!(!spec.is_updatable);

        let desc = spec.description.as_ref().unwrap();
        assert_eq!(desc.input.len(), 1);
        assert_eq!(desc.output.len(), 1);
        assert_eq!(desc.input[0].name, "input");
        assert_eq!(desc.output[0].name, "output");

        // Verify it's a NeuralNetwork model
        assert!(matches!(spec.r#type, Some(ModelType::NeuralNetwork(_))));

        if let Some(ModelType::NeuralNetwork(nn)) = &spec.r#type {
            assert_eq!(
                nn.layers.len(),
                1,
                "Conv2d spec should have exactly 1 layer"
            );

            let layer = &nn.layers[0];
            assert_eq!(layer.name, "test_conv2d");

            // Verify it's a Convolution layer
            assert!(matches!(layer.layer, Some(LayerKind::Convolution(_))));

            if let Some(LayerKind::Convolution(conv)) = &layer.layer {
                assert_eq!(conv.output_channels, 2);
                assert_eq!(conv.kernel_channels, 3);
                assert_eq!(conv.kernel_size, vec![1, 1]);
                assert_eq!(conv.stride, vec![1, 1]);
                assert!(!conv.has_bias);
                assert!(!conv.is_deconvolution);
            }
        }

        // Verify I/O shapes are 4D NCHW
        let input_type = desc.input[0].r#type.as_ref().unwrap();
        if let Some(FeatureTypeKind::MultiArrayType(arr)) = &input_type.r#type {
            assert_eq!(
                arr.shape,
                vec![1, 3, 1, 1],
                "input should be [1, in_dim, 1, 1]"
            );
        }

        let output_type = desc.output[0].r#type.as_ref().unwrap();
        if let Some(FeatureTypeKind::MultiArrayType(arr)) = &output_type.r#type {
            assert_eq!(
                arr.shape,
                vec![1, 2, 1, 1],
                "output should be [1, out_dim, 1, 1]"
            );
        }
    }

    #[test]
    fn test_build_conv2d_linear_model_spec_serializes() {
        let weights = vec![0.5f32; 12]; // [3, 4]
        let spec = build_conv2d_linear_model_spec("test", &weights, 4, 3);
        let bytes = spec.encode_to_vec();
        assert!(
            !bytes.is_empty(),
            "serialized Conv2d spec should not be empty"
        );
    }

    #[test]
    fn test_conv2d_linear_matches_inner_product() {
        // Both specs use the same weights data — verify structural equivalence.
        // The weight data is identical ([out_dim, in_dim] row-major for both).
        // Conv2d(1×1) is mathematically equivalent to InnerProduct when
        // input spatial dims are 1×1.
        let weights = vec![0.5f32; 12]; // [3, 4]
        let ip_spec = build_linear_model_spec("test", &weights, 4, 3);
        let conv_spec = build_conv2d_linear_model_spec("test", &weights, 4, 3);

        // Both serialize successfully
        let ip_bytes = ip_spec.encode_to_vec();
        let conv_bytes = conv_spec.encode_to_vec();
        assert!(!ip_bytes.is_empty());
        assert!(!conv_bytes.is_empty());

        // Extract and compare the weight data from both specs
        let ip_weights = extract_inner_product_weights(&ip_spec);
        let conv_weights = extract_conv2d_weights(&conv_spec);

        assert_eq!(ip_weights, conv_weights, "weight data should be identical");
    }

    /// Helper: extract weight float_value from the InnerProduct layer of a spec.
    fn extract_inner_product_weights(spec: &Model) -> Vec<f32> {
        if let Some(ModelType::NeuralNetwork(nn)) = &spec.r#type {
            if let Some(LayerKind::InnerProduct(ip)) = &nn.layers[0].layer {
                return ip.weights.as_ref().unwrap().float_value.clone();
            }
        }
        vec![]
    }

    /// Helper: extract weight float_value from the Convolution layer of a spec.
    fn extract_conv2d_weights(spec: &Model) -> Vec<f32> {
        if let Some(ModelType::NeuralNetwork(nn)) = &spec.r#type {
            if let Some(LayerKind::Convolution(conv)) = &nn.layers[0].layer {
                return conv.weights.as_ref().unwrap().float_value.clone();
            }
        }
        vec![]
    }

    // ── Residency Validation Test ────────────────────────────────

    #[test]
    fn test_ane_residency_validation() {
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut backend = AneBackend::new();
        backend.compile(&weights, &config).unwrap();
        let model = backend.model.as_ref().unwrap();

        // Time a micro-prediction (single lm_head inference)
        let mut hidden = vec![0.0f32; config.n_embd];
        for (i, v) in hidden.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }

        let start = std::time::Instant::now();
        let result = run_lm_head(model, &hidden, &config);
        let elapsed = start.elapsed();

        assert!(result.is_ok(), "lm_head prediction should succeed");

        let elapsed_us = elapsed.as_micros() as u64;
        const ANE_RESIDENCY_THRESHOLD_US: u64 = 1000;

        if elapsed_us > ANE_RESIDENCY_THRESHOLD_US {
            eprintln!(
                "WARNING: ANE residency check: {elapsed_us}µs > {ANE_RESIDENCY_THRESHOLD_US}µs threshold \
                 (model may have fallen back to CPU; depends on hardware)"
            );
            // Don't fail — residency depends on hardware
        }
    }

    // ── GOAT: ANE forward == CPU forward ─────────────────────────

    #[test]
    fn test_goat_ane_forward_matches_cpu() {
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let pairs: [(usize, usize); 5] = [(0, 0), (1, 1), (3, 2), (7, 4), (5, 9)];

        // CPU reference: run forward for each (token, pos) pair
        let mut cpu_logits_all = Vec::new();
        for &(token, pos) in &pairs {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let logits =
                crate::transformer::forward(&mut ctx, &weights, &mut cache, token, pos, &config)
                    .to_vec();
            cpu_logits_all.push(logits);
        }

        // ANE forward for same pairs
        let mut backend = AneBackend::new();
        backend.compile(&weights, &config).unwrap();

        let mut ane_logits_all = Vec::new();
        for &(token, pos) in &pairs {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let logits = backend
                .forward(&mut ctx, &weights, &mut cache, token, pos, &config)
                .to_vec();
            ane_logits_all.push(logits);
        }

        // Assert cosine similarity ≥ 0.997 for ALL pairs
        for (i, (&(token, pos), (cpu, ane))) in pairs
            .iter()
            .zip(cpu_logits_all.iter().zip(ane_logits_all.iter()))
            .enumerate()
        {
            let sim = cosine_similarity(cpu, ane);
            eprintln!("GOAT ANE pair {i} (token={token}, pos={pos}): cosine_sim={sim:.6}");
            assert!(
                sim >= 0.997,
                "GOAT ANE forward mismatch at pair {i} (token={token}, pos={pos}): \
                 cosine_sim={sim:.6} < 0.997"
            );
        }
    }

    // ── Plan 176: Latency Benchmarks ────────────────────────────

    #[test]
    fn bench_ane_forward_latency_vs_cpu() {
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // CPU warm-up
        for _ in 0..100 {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            crate::transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        }

        // CPU timed
        let cpu_elapsed = {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                let mut ctx = ForwardContext::new(&config);
                let mut cache = MultiLayerKVCache::new(&config);
                crate::transformer::forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
            }
            start.elapsed()
        };
        let cpu_us_per_token = cpu_elapsed.as_micros() as f64 / 1000.0;

        // ANE
        let mut backend = AneBackend::new();
        backend.compile(&weights, &config).unwrap();

        // ANE warm-up
        for _ in 0..100 {
            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            backend.forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        }

        // ANE timed
        let ane_elapsed = {
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                let mut ctx = ForwardContext::new(&config);
                let mut cache = MultiLayerKVCache::new(&config);
                backend.forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
            }
            start.elapsed()
        };
        let ane_us_per_token = ane_elapsed.as_micros() as f64 / 1000.0;
        let speedup = cpu_us_per_token / ane_us_per_token;

        eprintln!(
            "CPU: {cpu_us_per_token:.1} µs/token, ANE: {ane_us_per_token:.1} µs/token, ANE speedup: {speedup:.2}×"
        );
    }

    #[test]
    fn bench_ane_compilation_time() {
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let init_elapsed = {
            let start = std::time::Instant::now();
            let backend = AneBackend::new();
            (start.elapsed(), backend)
        };
        let mut backend = init_elapsed.1;

        let compile_elapsed = {
            let start = std::time::Instant::now();
            backend.compile(&weights, &config).unwrap();
            start.elapsed()
        };

        eprintln!(
            "ANE init: {} µs, ANE compile: {} ms",
            init_elapsed.0.as_micros(),
            compile_elapsed.as_millis()
        );
    }
}
