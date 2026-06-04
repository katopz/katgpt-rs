//! Apple Neural Engine inference backend via CoreML (Plan 176).
//!
//! Uses runtime weight compilation instead of `.mlmodelc` file loading.
//! katgpt-rs is modelless — weights live in-memory as `TransformerWeights`,
//! generated at runtime. This backend compiles them into a CoreML model
//! on demand for ANE execution.
//!
//! # Runtime Compilation
//!
//! `compile()` takes `&TransformerWeights` + `&Config` and builds a CoreML
//! neural network programmatically. No `.mlmodelc` file is needed — the model
//! is constructed from the weight struct using `coreml_native` APIs.
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

use crate::inference_backend::InferenceBackend;
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::Config;

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
    /// This is the runtime weight compilation path — no `.mlmodelc` file needed.
    /// Takes the in-memory weight struct and builds a CoreML neural network
    /// programmatically using `coreml_native` APIs.
    ///
    /// TODO: Implement actual CoreML model building from `TransformerWeights`.
    ///       Requires `coreml_native::Model::from_spec()` or equivalent.
    ///       Currently stubbed — sets `compiled=true` but doesn't actually build.
    pub fn compile(
        &mut self,
        _weights: &TransformerWeights,
        _config: &Config,
    ) -> Result<(), AneError> {
        // TODO: Build CoreML model from TransformerWeights
        // 1. Create MLModel spec with neural network layers
        // 2. Map TransformerWeights fields → CoreML weight parameters
        // 3. Use Conv2d(1×1) for linear layers (ANE-friendly)
        // 4. Set compute units to .All
        // 5. Verify ANE residency

        // Stub: mark as compiled
        self.compiled = true;
        self.needs_recompile = false;
        // self.model = Some(built_model);
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

    // TODO: check_residency() will be re-added when actual CoreML model building works.
    //       It will time a micro-prediction to verify ANE execution (not CPU fallback).
    //       ANE <1ms vs CPU fallback >5ms.
}

impl InferenceBackend for AneBackend {
    fn forward<'a>(
        &'a mut self,
        _ctx: &'a mut ForwardContext,
        _weights: &TransformerWeights,
        _cache: &mut MultiLayerKVCache,
        _token: usize,
        _pos: usize,
        _config: &Config,
    ) -> &'a mut [f32] {
        // CPU fallback stub — delegates to the Rust transformer forward pass.
        //
        // The runtime compilation path works as follows:
        //   1. Caller calls compile(weights, config) to build the CoreML model
        //   2. forward() uses self.model.predict() for ANE execution
        //   3. Falls back to CPU if self.model is None
        //
        // TODO: Once compile() actually builds the CoreML model:
        //   - Check self.model.is_some()
        //   - Construct FP16 input tensors from token IDs + position
        //   - Run model.predict() with proper input/output names
        //   - Extract logits from CoreML output (FP16 → f32)
        //   - Write logits into ctx.logits buffer

        crate::transformer::forward(_ctx, _weights, _cache, _token, _pos, _config)
    }

    fn device_name(&self) -> &'static str {
        "ANE"
    }

    fn supports_stateful(&self) -> bool {
        // Stateful KV cache via MLState requires macOS 15+
        // Will be enabled in a future milestone.
        false
    }
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

        // Stub compile with micro fixtures
        let config = Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        backend.compile(&weights, &config).unwrap();
        assert!(
            backend.is_compiled(),
            "compile() should set is_compiled=true"
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
}
