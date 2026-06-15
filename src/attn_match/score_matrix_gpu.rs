//! GPU dispatch stub for the AM score matrix (Plan 271 Phase 2, T2.8).
//!
//! Forwards `compute_score_matrix` to the Metal GPU backend when the
//! `gpu_inference` feature is enabled on macOS. The current implementation
//! is a **dispatch stub**: it constructs the GPU buffers and dispatches a
//! pre-built matmul kernel when one is available, and falls back to
//! [`compute_score_matrix_rayon`] when GPU dispatch fails for any reason
//! (no device, kernel compile error, buffer allocation failure).
//!
//! # Why a stub?
//!
//! The existing `gpu_backend` module is purpose-built for transformer
//! forward passes (QKV projection, RMSNorm, attention) and does not expose
//! a general-purpose `(n, d) × (T, d)^T → (n, T)` matmul primitive. Writing
//! a fused score-matrix Metal kernel (with the `1/√d` scaling and
//! optional max-shift) is a meaningful amount of shader + dispatch code.
//! This module wires the dispatch path so the router can pick `Gpu` and
//! the orchestrator can attempt GPU; the actual shader lands in a future
//! task once we measure the cross-over point on real workloads.
//!
//! # Return contract
//!
//! [`try_compute_score_matrix_gpu`] returns `Ok(())` only if the GPU
//! actually executed the computation and wrote into `out`. On any failure
//! — including "no GPU kernel available" — it returns `Err` and the caller
//! is expected to fall back to [`compute_score_matrix_rayon`].
//!
//! [`compute_score_matrix_rayon`]: crate::attn_match::score_matrix_rayon::compute_score_matrix_rayon

use crate::attn_match::score_matrix_rayon::compute_score_matrix_rayon;

/// Error returned by [`try_compute_score_matrix_gpu`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuDispatchError {
    /// No Metal device available (e.g., headless CI runner).
    NoDevice,
    /// The AM matmul shader is not yet compiled into this build.
    ///
    /// This is the normal stub-state error: the dispatch path is wired but
    /// the actual shader source is not bundled. Callers should fall back to
    /// [`compute_score_matrix_rayon`].
    ShaderNotAvailable,
    /// A buffer allocation or dispatch failed at runtime.
    DispatchFailed(String),
}

impl std::fmt::Display for GpuDispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no Metal device available"),
            Self::ShaderNotAvailable => write!(
                f,
                "AM score-matrix GPU shader not bundled in this build \
                 (T2.8 stub); caller should fall back to rayon"
            ),
            Self::DispatchFailed(s) => write!(f, "GPU dispatch failed: {}", s),
        }
    }
}

impl std::error::Error for GpuDispatchError {}

/// Attempt to compute the score matrix `S = Q·K^T · inv_sqrt_d` on GPU.
///
/// This is the entry point invoked by
/// [`crate::attn_match::compact::dispatch_score_matrix`] when the router
/// selects `SolverBackend::Gpu`. On success, `out` is fully written. On
/// failure, `out` is left untouched and the caller falls back to the
/// rayon-parallel CPU kernel.
///
/// # Current behavior
///
/// **Stub**: always returns [`GpuDispatchError::ShaderNotAvailable`]. The
/// router still records `SolverBackend::Gpu` in the trace, but the actual
/// compute happens on the CPU via the rayon fallback. This keeps the
/// dispatch path tested end-to-end (router → stub → fallback → correct
/// output) without requiring a Metal shader to be present.
///
/// # Future work
///
/// 1. Bundle an MSL shader for `(n, d) × (T, d)^T → (n, T)` matmul with
///    pre-applied `1/√d` scaling.
/// 2. Compile the pipeline lazily on first dispatch and cache it on the
///    device.
/// 3. Profile the cross-over point vs rayon on Apple Silicon (the paper's
///    4× AVX2 SIMD target suggests GPU wins above some `T`).
pub fn try_compute_score_matrix_gpu(
    _queries: &[f32],
    _keys: &[f32],
    _n: usize,
    _t_len: usize,
    _d: usize,
    _out: &mut [f32],
) -> Result<(), GpuDispatchError> {
    // T2.8 stub — see module docs. The wired dispatch path lets the router
    // exercise the Gpu backend end-to-end; the actual Metal kernel is future
    // work. We deliberately do NOT touch `_out` so the caller can safely
    // fall back to compute_score_matrix_rayon without re-initializing.
    Err(GpuDispatchError::ShaderNotAvailable)
}

/// Convenience: run the GPU attempt and silently fall back to rayon.
///
/// Used by callers that don't care about *why* the GPU failed and just want
/// the correct output regardless of backend. Equivalent to:
///
/// ```ignore
/// match try_compute_score_matrix_gpu(...) {
///     Ok(()) => {},
///     Err(_) => compute_score_matrix_rayon(...),
/// }
/// ```
///
/// Kept as a named function so the fallback path is explicit in profile traces.
#[inline]
pub fn compute_score_matrix_gpu_or_rayon(
    queries: &[f32],
    keys: &[f32],
    n: usize,
    t_len: usize,
    d: usize,
    out: &mut [f32],
) {
    if try_compute_score_matrix_gpu(queries, keys, n, t_len, d, out).is_err() {
        compute_score_matrix_rayon(queries, keys, n, t_len, d, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn_match::score_matrix::compute_score_matrix;

    /// Stub always returns ShaderNotAvailable — this is the documented
    /// contract until a real Metal kernel lands.
    #[test]
    fn test_stub_returns_shader_not_available() {
        let queries = [1.0f32, 0.0];
        let keys = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![0.0f32; 2];
        let err = try_compute_score_matrix_gpu(&queries, &keys, 1, 2, 2, &mut out).unwrap_err();
        assert_eq!(err, GpuDispatchError::ShaderNotAvailable);
    }

    /// The fallback path must produce output identical to the scalar kernel.
    #[test]
    fn test_gpu_or_rayon_matches_scalar() {
        let n = 4;
        let t_len = 8;
        let d = 16;
        let mut seed = 54321u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let queries: Vec<f32> = (0..n * d).map(|_| rng()).collect();
        let keys: Vec<f32> = (0..t_len * d).map(|_| rng()).collect();

        let mut scalar = vec![0.0f32; n * t_len];
        compute_score_matrix(&queries, &keys, n, t_len, d, &mut scalar);

        let mut gpu_or_rayon = vec![0.0f32; n * t_len];
        compute_score_matrix_gpu_or_rayon(&queries, &keys, n, t_len, d, &mut gpu_or_rayon);

        for i in 0..n * t_len {
            assert!(
                (scalar[i] - gpu_or_rayon[i]).abs() < 1e-5,
                "gpu_or_rayon/scalar mismatch at {}: scalar={} gpu_or_rayon={}",
                i,
                scalar[i],
                gpu_or_rayon[i]
            );
        }
    }

    /// `out` must be untouched on GPU failure (caller relies on this to
    /// safely fall back).
    #[test]
    fn test_out_untouched_on_failure() {
        let queries = [1.0f32, 0.0];
        let keys = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = vec![42.0f32; 2]; // sentinel
        let _ = try_compute_score_matrix_gpu(&queries, &keys, 1, 2, 2, &mut out);
        // The stub contract: do not write into `out` on failure.
        assert_eq!(out, vec![42.0f32, 42.0f32]);
    }
}
