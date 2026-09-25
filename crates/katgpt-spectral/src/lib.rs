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

// ── Phase 4 (Proposal 003) absorptions: spectral diagnostics, alignment,
// and decomposition primitives. Moved out of `katgpt-rs/src/` because they
// are pure linear-algebra on eigenbases / factor pairs and belong with the
// spectral substrate, not the root crate.

/// River-valley diagnostic metrics — always-on (Plan 152 GOAT 25/25).
pub mod river_valley;
/// Spectral concentration adaptive rank — always-on (Plan 264 Phase 3).
pub mod spectral_concentration;
/// Shared power-iteration + L2 retraction helpers (always-on — consumed by
/// `gauge_invariant` and `manifold_power_iter_router` which are both
// default-ON at root).
pub mod spectral_retract;

#[cfg(feature = "gauge_invariant")]
pub mod gauge_invariant;
#[cfg(feature = "manifold_power_iter_router")]
pub mod manifold_power_iter_router;
#[cfg(feature = "off_principal_retrieval")]
pub mod off_principal;
#[cfg(feature = "peira_distill")]
pub mod peira;
#[cfg(feature = "orthogonal_procrustes")]
pub mod procrustes;
#[cfg(feature = "quantile_balance_router")]
pub mod quantile_balance_router;
#[cfg(feature = "spectral_budget")]
pub mod spectral_budget;
#[cfg(feature = "stiff_anomaly")]
pub mod stiff_anomaly;

#[cfg(feature = "outlier_guard")]
pub mod outlier_guard;

// ── Phase 12 absorption (Proposal 003, 2026-07-04): module moved from katgpt-rs/src/.
// HLA Windowed Eigenbasis Recovery — per-NPC eigenbasis recovery from windowed
// HLA activations (power iteration on D×D Gram, modelless, Issue 001).
// Gated by `hla_eigenbasis_recovery`; root re-exports as `katgpt_rs::hla_eigenbasis`.
#[cfg(feature = "hla_eigenbasis_recovery")]
pub mod hla_eigenbasis;

// ── Plan 475 (2026-08-11): ICA Lens — FastICA non-Gaussian direction mining
// + ERF diagnostic (arXiv:2606.11722). Sibling to `hla_eigenbasis` — ICA
// consumes the EigenbasisTracker Gram as its whitening step, then runs the
// FastICA fixed-point rotation. Gated by `ica_lens`; the
// `hla_eigenbasis_recovery` dep is required for the whitening substrate.
#[cfg(feature = "ica_lens")]
pub mod ica_lens;

// Spectral Rewiring — weight delta purification via base SVD projection
// (Plan 423, Research 406). The on-principal complement of `off_principal`
// (Plan 264). STAYS OPT-IN — Plan 423 Phase 3 GOAT (Bench 423, 2026-07-10):
// all mechanism gates PASS (G1a/G2/G3/G4/G5/G6), but the spectral concentration
// assumption (G1b) does NOT hold at NPC scale (≤64×64), verified via
// riir-train Issue 374 (trained LoRA deltas produce on_manifold_fraction in
// [0.27, 0.58], far below the SAR threshold > 0.8). Stays cold-tier only.
#[cfg(feature = "spectral_rewire")]
pub mod spectral_rewire;

// Issue 882 P4 rider (b) (Research 586, Bench 897): rank-1 spectral deflation
// `A′ = A − λ(A v₁)v₁ᵀ` of a rerank-stage M×M candidate affinity, gated on
// stable rank `‖A‖_F²/σ₁² < θ` (deflate only a COLLAPSED matrix — the
// anti-help trap). Consumes `spectral_retract::power_iter_step`. Opt-in.
#[cfg(feature = "affinity_deflation")]
pub mod affinity_deflation;

#[cfg(all(feature = "spectral_quant", feature = "maxsim"))]
pub use forward::par_maxsim_score_spectralquant;
pub use forward::{
    attention_spectralquant, dequantize_spectral_keys_flat, dequantize_spectral_values_flat,
    par_dequantize_spectral_keys_flat, par_dequantize_spectral_keys_flat_into,
    par_dequantize_spectral_values_flat, par_dequantize_spectral_values_flat_into,
};
pub use nonuniform_quant::{CompressedVector, NonUniformQuantizer};
#[cfg(all(feature = "outlier_guard", feature = "stiff_anomaly"))]
pub use outlier_guard::StiffSoftCrossCheck;
#[cfg(feature = "outlier_guard")]
pub use outlier_guard::{
    ConfidenceLevel, LayerReport, OutlierAction, OutlierGuard, OutlierGuardConfig,
    OutlierGuardReport,
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
