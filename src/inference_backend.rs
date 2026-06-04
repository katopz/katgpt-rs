//! Inference backend abstraction for the transformer forward pass.
//!
//! Defines the [`InferenceBackend`] trait that decouples the high-level generate
//! loop from the concrete compute backend (CPU, Apple Neural Engine, etc.).
//!
//! The default [`CpuBackend`] delegates to [`crate::transformer::forward`].

use std::fmt;

use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::Config;

/// Error type for backend weight compilation.
#[derive(Debug)]
pub enum CompileError {
    /// The backend does not support runtime weight compilation.
    UnsupportedBackend(String),
    /// Compilation failed on the target device.
    DeviceError(String),
    /// Invalid weight dimensions for the target device.
    InvalidWeights(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::UnsupportedBackend(msg) => {
                write!(f, "unsupported backend: {msg}")
            }
            CompileError::DeviceError(msg) => {
                write!(f, "device error: {msg}")
            }
            CompileError::InvalidWeights(msg) => {
                write!(f, "invalid weights: {msg}")
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Backend for transformer forward pass inference.
///
/// Implementations: [`CpuBackend`] (default), AneBackend (Apple Neural Engine, feature-gated).
pub trait InferenceBackend {
    /// Run one forward pass: token + position → logits.
    ///
    /// Returns a logits slice of length `config.vocab_size`.
    /// The returned slice borrows from `ctx`.
    fn forward<'a>(
        &'a mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32];

    /// Human-readable device name for logging.
    fn device_name(&self) -> &'static str;

    /// Whether this backend supports stateful KV caching (e.g. CoreML MLState).
    fn supports_stateful(&self) -> bool {
        false
    }

    /// Reset any backend-specific state for a new sequence.
    fn reset(&mut self) {}

    /// Compile weights into a device-specific pipeline.
    ///
    /// Default: no-op (CPU doesn't need compilation).
    fn compile(
        &mut self,
        _weights: &TransformerWeights,
        _config: &Config,
    ) -> Result<(), CompileError> {
        Ok(())
    }

    /// Check if backend has valid compiled weights.
    ///
    /// Default: `true` (CPU is always "compiled").
    fn is_compiled(&self) -> bool {
        true
    }

    /// Signal that LoRA weights changed; backend should recompile on next forward.
    ///
    /// Default: no-op (CPU doesn't need recompilation).
    fn recompile_hint(&mut self) {}
}

// ---------------------------------------------------------------------------
// CpuBackend
// ---------------------------------------------------------------------------

/// CPU backend using the standard Rust transformer forward pass.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl InferenceBackend for CpuBackend {
    fn forward<'a>(
        &'a mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        crate::transformer::forward(ctx, weights, cache, token, pos, config)
    }

    fn device_name(&self) -> &'static str {
        "CPU"
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Auto-route selection
// ---------------------------------------------------------------------------

/// Backend selection for CLI flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BackendKind {
    /// Automatically select best available backend.
    #[default]
    Auto,
    /// Force CPU backend.
    Cpu,
    /// Force ANE backend (error if unavailable).
    Ane,
    /// Use TriggerGate + InferenceRouter for dynamic tier selection.
    Gate,
}

/// Select the best available inference backend.
///
/// Logic:
/// 1. If `kind` is `Cpu`, always return `CpuBackend`.
/// 2. If `kind` is `Ane`, try to create an uncompiled `AneBackend`.
/// 3. If `kind` is `Auto`:
///    - On macOS with `ane` feature: try ANE, fall back to CPU on failure.
///    - Otherwise: use CPU.
///
/// Logs which backend was selected via `log::info!`.
///
/// TODO: Remove `model_path` parameter — no longer needed with runtime compilation.
pub fn auto_backend(
    kind: BackendKind,
    model_path: Option<&std::path::Path>,
) -> Box<dyn InferenceBackend> {
    match kind {
        BackendKind::Cpu => {
            log::info!("Backend: CPU (forced)");
            Box::new(CpuBackend::new())
        }
        BackendKind::Ane => {
            // Will be caught below if ANE is not available
            try_ane_backend(model_path).expect("ANE backend requested but unavailable")
        }
        BackendKind::Auto => {
            let backend = try_ane_backend(model_path);
            match backend {
                Ok(b) => b,
                Err(reason) => {
                    log::info!("Backend: CPU (ANE unavailable: {reason})");
                    Box::new(CpuBackend::new())
                }
            }
        }
        BackendKind::Gate => {
            // Gate mode uses InferenceRouter, not a bare backend.
            // Return CpuBackend as placeholder — caller should use InferenceRouter instead.
            log::info!("Backend: CPU (gate mode — use InferenceRouter for dynamic routing)");
            Box::new(CpuBackend::new())
        }
    }
}

/// Attempt to create an ANE backend. Returns error message on failure.
///
/// Creates an uncompiled `AneBackend` — the caller must call `compile()`
/// before the first `forward()` pass.
fn try_ane_backend(
    _model_path: Option<&std::path::Path>,
) -> Result<Box<dyn InferenceBackend>, String> {
    #[cfg(all(target_os = "macos", feature = "ane"))]
    {
        use crate::ane_backend::AneBackend;

        let backend = AneBackend::new();
        log::info!("Backend: ANE (available, awaiting compile())");
        Ok(Box::new(backend))
    }

    #[cfg(not(all(target_os = "macos", feature = "ane")))]
    {
        Err("ANE not available (requires macOS + 'ane' feature)".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer;
    use crate::types::Rng;

    fn micro_fixtures() -> (
        Config,
        TransformerWeights,
        ForwardContext,
        MultiLayerKVCache,
    ) {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let ctx = ForwardContext::new(&config);
        let cache = MultiLayerKVCache::new(&config);
        (config, weights, ctx, cache)
    }

    #[test]
    fn test_cpu_backend_matches_direct_forward() {
        let (config, weights, _, _) = micro_fixtures();

        // Direct call
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let direct = transformer::forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();

        // Through CpuBackend
        let mut backend = CpuBackend::new();
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let via_backend = backend
            .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
            .to_vec();

        assert_eq!(
            direct, via_backend,
            "CpuBackend should produce identical logits to direct forward"
        );
    }

    #[test]
    fn test_cpu_backend_device_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.device_name(), "CPU");
    }

    #[test]
    fn test_cpu_backend_supports_stateful() {
        let backend = CpuBackend::new();
        assert!(!backend.supports_stateful());
    }

    #[test]
    fn test_cpu_backend_deterministic() {
        let (config, weights, _, _) = micro_fixtures();

        let mut backend = CpuBackend::new();

        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let run1 = backend
            .forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config)
            .to_vec();

        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let run2 = backend
            .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
            .to_vec();

        assert_eq!(run1, run2, "same input must produce same logits");
    }

    #[test]
    fn test_auto_backend_cpu_forced() {
        let backend = auto_backend(BackendKind::Cpu, None);
        assert_eq!(backend.device_name(), "CPU");
    }

    #[test]
    fn test_auto_backend_auto_falls_back_to_cpu() {
        // No model file available, so auto should fall back to CPU
        let backend = auto_backend(BackendKind::Auto, None);
        assert_eq!(backend.device_name(), "CPU");
    }

    #[test]
    fn test_backend_kind_default() {
        assert_eq!(BackendKind::default(), BackendKind::Auto);
    }

    #[test]
    fn test_compile_error_display() {
        let e = CompileError::UnsupportedBackend("ane".into());
        assert_eq!(e.to_string(), "unsupported backend: ane");

        let e = CompileError::DeviceError("timeout".into());
        assert_eq!(e.to_string(), "device error: timeout");

        let e = CompileError::InvalidWeights("dim mismatch".into());
        assert_eq!(e.to_string(), "invalid weights: dim mismatch");
    }

    #[test]
    fn test_cpu_backend_is_compiled() {
        let backend = CpuBackend::new();
        assert!(backend.is_compiled());
    }

    #[test]
    fn test_cpu_backend_compile_noop() {
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = CpuBackend::new();
        assert!(backend.compile(&weights, &config).is_ok());
    }

    #[test]
    fn test_cpu_backend_recompile_hint_noop() {
        let mut backend = CpuBackend::new();
        // Should not panic
        backend.recompile_hint();
    }
}
