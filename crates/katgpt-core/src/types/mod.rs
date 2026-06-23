//! Shared configuration, RNG, math utilities, and inference types for the
//! katgpt-rs / riir-engine superset.
//!
//! Originally a single 5,148-line `types.rs` (2.5× the 2048-line ceiling),
//! split into topic-specific submodules. The full public surface is
//! re-exported here so `crate::types::*` paths remain unchanged.
//!
//! # Module layout
//!
//! - [`enums`] — small config enums (DepthTier, HlaMode, AttentionMode, …)
//!   plus WallConfig / ThinkingBudget
//! - [`config`] — the `Config` struct (~1.5k lines), `InferenceOverrides`,
//!   and `kv_dim`
//! - [`rng`] — XorShift64 PRNG
//! - [`math`] — SIMD-accelerated softmax / rmsnorm / matmul / sample_token
//!   (legacy home; candidates for relocation to `simd::`)
//! - [`lora`] — CPU-side LoRA adapter
//! - [`gpart`] — GPart Isometric Partition adapter (Research 227)
//! - [`domain`] — DomainLatent embedding (Plan 038)
//! - [`inference`] — InferenceResult, TaskType, ProposerTask, DataGate
//! - [`looping`] — Training-Free Loop types (Plan 136)
//! - [`ternary`] — Bit-plane ternary weights (`plasma_path`)
//! - [`hydra`] — Hydra Adaptive Layer Budget types
//! - [`sense`] — ShardEmbedding + sense composition types
//!
//! Test modules live alongside their topic (e.g. `rng::tests_rng`) or in
//! `tests_types.rs` for cross-cutting tests.

mod config;
mod domain;
mod enums;
mod gpart;
mod hydra;
mod inference;
mod looping;
mod lora;
pub mod math;
mod rng;
mod sense;
mod ternary;

#[cfg(test)]
mod tests_types;

// Re-export the entire public surface so `crate::types::*` paths are
// unchanged after the file → folder split. Feature gates mirror the
// gates on the underlying items.
pub use config::{Config, InferenceOverrides, kv_dim};
#[cfg(feature = "domain_latent")]
pub use domain::DomainLatent;
pub use enums::{
    AttentionMode, AttentionProjection, CacheLayout, ConvergenceSelector, DashAttnConfig,
    DepthTier, HlaMode, HybridPattern, LoopMode, ModelArchitecture, ResidualGate,
    RetrievalHeadRole, RtTurboConfig, SdpaOutputGate, WeightDtype,
};
#[cfg(feature = "deltanet_inference")]
pub use enums::DeltaNetLayerType;
#[cfg(feature = "collapse_aware_thinking")]
pub use enums::ThinkingBudget;
#[cfg(feature = "wall_attention")]
pub use enums::WallConfig;
#[cfg(feature = "sr2am_configurator")]
pub use enums::{ConfiguratorContext, PlanningDecision};
#[cfg(feature = "gpart_adapter")]
pub use gpart::{GPART_MAGIC, GPART_VERSION, GpartAdapter, GpartPair, GpartPrepared};
#[cfg(feature = "hydra_budget")]
pub use hydra::{HydraBudgetConfig, HydraLayerProfile};
pub use inference::InferenceResult;
#[cfg(feature = "data_gate")]
pub use inference::{DataGate, GateDecision, ProposerTask, TaskType};
pub use looping::{CacheStrategy, IterationMode, SubStepStrategy, TrainingFreeLoopConfig};
pub use lora::{LoraAdapter, LoraPair, lora_apply};
pub use math::{
    gegelu, gegelu_tanh, matmul, matmul_f16, matmul_f16_parallel, matmul_parallel, matmul_relu,
    rmsnorm, rmsnorm_with_gamma, rmsnorm_with_gamma_eps, sample_token, sample_token_into, silu,
    softmax, softmax_scaled, swiglu,
};
#[cfg(feature = "sparse_mlp")]
pub use math::sparse_matmul;
pub use rng::Rng;
pub use sense::{DilationConfig, SenseKind, SenseModule, ShardEmbedding, TernaryDir};
#[cfg(feature = "plasma_path")]
pub use ternary::TernaryWeights;

// Internal helpers (read_u32_le / read_f32_le / read_u16_le) live in
// `domain.rs` and are crate-private — not re-exported here. If other modules
// need them, import via `crate::types::domain::read_u32_le`.
