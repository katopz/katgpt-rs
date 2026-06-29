//! katgpt-spectral — Spectral quantization substrate.
//!
//! Calibrated eigenbasis KV cache compression (Plan 078):
//! - Offline calibration: covariance → eigendecomposition → eigenbasis
//! - Two-regime allocation: semantic (high-energy) + tail dimensions
//! - Water-fill: per-dim bit allocation proportional to eigenvalue
//! - Lloyd-Max: optimal non-uniform scalar quantizer per regime
//!
//! Compresses KV cache from f32 to ~3 bits/coordinate with minimal MSE.
//!
//! # Provenance
//!
//! Spun out of `katgpt-rs/src/spectralquant/` (Issue 015 Phase 2) because
//! the substrate has both KV consumers (`shard_kv`, `kvarn`) and non-KV
//! consumers (`funcattn_compose`, `chiaroscuro`, `benchmark/infrastructure`).
//! Folding it into `katgpt-kv` would have forced non-KV modules to depend
//! on a `-kv`-named crate — wrong direction. This is a standalone
//! foundational quantization crate that `katgpt-kv` depends on.
//!
//! The root `katgpt-rs` crate re-exports this as `katgpt_rs::spectralquant`
//! for back-compat (Issue 015 Phase 5).

pub mod forward;
pub mod nonuniform_quant;
pub mod spectral;
pub mod spectral_kv_cache;
pub mod spectral_rotation;
pub mod types;

#[cfg(feature = "outlier_guard")]
pub mod outlier_guard;

#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
pub use forward::par_maxsim_score_spectralquant;
pub use forward::{
    attention_spectralquant, dequantize_spectral_keys_flat, dequantize_spectral_values_flat,
    par_dequantize_spectral_keys_flat, par_dequantize_spectral_values_flat,
};
pub use nonuniform_quant::{CompressedVector, NonUniformQuantizer};
#[cfg(all(feature = "outlier_guard", feature = "stiff_anomaly"))]
pub use outlier_guard::StiffSoftCrossCheck;
#[cfg(feature = "outlier_guard")]
pub use outlier_guard::{
    ConfidenceLevel, LayerReport, OutlierAction, OutlierGuard, OutlierGuardConfig, OutlierGuardReport,
};
#[cfg(feature = "dual_gram_pca")]
pub use spectral::calibrate_eigenbasis_dual_gram;
pub use spectral::{
    BitAllocator, CalibrationResult, LloydMaxQuantizer, calibrate_eigenbasis,
    cumulative_variance_thresholds, generate_selective_qjl_signs, marginal_gain,
    participation_ratio, spectral_gap, waterfill_bits,
};
pub use spectral_kv_cache::{DequantizeScratch, SpectralQuantKVCache};
#[cfg(feature = "turboquant")]
pub use spectral_rotation::RandomRotation;
pub use spectral_rotation::SpectralRotation;
pub use types::{
    LloydMaxCodebook, SpectralQuantCalibration, SpectralQuantKVCacheConfig, SpectralQuantLayer,
    WaterfillAllocation,
};
