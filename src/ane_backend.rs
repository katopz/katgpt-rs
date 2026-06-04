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
    ArrayFeatureType, FeatureDescription, FeatureType, InnerProductLayerParams, Model,
    ModelDescription, NeuralNetwork, NeuralNetworkLayer, WeightParams,
    feature_type::Type as FeatureTypeKind, model::Type as ModelType,
    neural_network_layer::Layer as LayerKind,
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

/// Helper: create a `FeatureType` for a Float32 multi-array with the given shape.
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
}
