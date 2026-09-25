//! katgpt-types — Shared configuration, RNG, math utilities, SIMD kernels,
//! and inference types for the katgpt-rs / riir-engine superset.
//!
//! Pure substrate leaf crate. No katgpt-* dependencies — only `fastrand`,
//! `blake3`, `serde`, `half`. This is the foundational layer (types + SIMD
//! kernels) that every other katgpt-* crate (and riir-engine) builds on.
//! The `types` and `simd` modules are co-located because `types::math`
//! calls `simd` kernels (softmax / rmsnorm / matmul) and `simd::ternary`
//! uses `types::TernaryWeights` — they form a tight bidirectional leaf that
//! cannot be split further without breaking the cycle.
//!
//! Originally a single 5,148-line `types.rs` inside katgpt-core (2.5× the
//! 2048-line ceiling), split into topic-specific submodules. The full public
//! surface is re-exported here so consumers can use `katgpt_types::*` paths
//! directly.
//!
//! Spun out of `katgpt-core::types` (Issue 007 Phase E Tier 1 #2) as a
//! standalone publishable crate mirroring the `katgpt-dec` / `katgpt-transformer`
//! template.
//!
//! # Module layout
//!
//! - `enums` — small config enums (DepthTier, HlaMode, AttentionMode, …)
//!   plus WallConfig / ThinkingBudget
//! - `config` — the `Config` struct (~1.5k lines), `InferenceOverrides`,
//!   and `kv_dim`
//! - `rng` — XorShift64 PRNG
//! - `math` — SIMD-accelerated softmax / rmsnorm / matmul / sample_token
//!   (legacy home; candidates for relocation to `katgpt-simd::`)
//! - `lora` — CPU-side LoRA adapter
//! - `gpart` — GPart Isometric Partition adapter (Research 227)
//! - `domain` — DomainLatent embedding (Plan 038)
//! - `inference` — InferenceResult, TaskType, ProposerTask, DataGate
//! - `looping` — Training-Free Loop types (Plan 136)
//! - `ternary` — Bit-plane ternary weights (`plasma_path`)
//! - `hydra` — Hydra Adaptive Layer Budget types
//! - `sense` — ShardEmbedding (DEPRECATED, Issue 139) + sense composition types
//!
//! Test modules live alongside their topic (e.g. `rng::tests_rng`) or in
//! `tests_types.rs` for cross-cutting tests.

mod config;
/// Depth-Invariance Diagnostic & Magnitude-Regularized Residual (Plan 306).
/// Pure math, depends only on simd. Co-located here (the leaf) so both
/// katgpt-core (depth_invariance feature) and katgpt-micro-belief
/// (audit_depth_invariance methods) can consume it without a cycle.
#[cfg(feature = "depth_invariance")]
pub mod depth_invariance;
mod domain;
#[cfg(feature = "depth_invariance")]
pub use depth_invariance::{
    DepthInvarianceConfig, DepthInvarianceDiagnostic, DepthInvarianceKind, MagnitudeRegularization,
    Scratch, apply_magnitude_regularization, classify_chain, classify_chain_batched,
};
/// Binary `{-1,+1}` bit-plane packed weights (Issue 145, `binary_plasma` feature).
#[cfg(feature = "binary_plasma")]
pub mod binary;
mod enums;
mod gpart;
mod hydra;
mod inference;
/// QuantizedKVCache trait — shared extension point for all KV backends.
/// Promoted from `katgpt-rs/src/types.rs` (Issue 015 Phase 1) so backend
/// crates can depend on the trait without pulling the whole root crate.
pub mod kv_cache;
/// Shared leaky-integrator / delta-rule step primitive (Plan 276 Phase 2 T2.1).
/// Pure inline math, zero deps. Consumed by both katgpt-micro-belief
/// (`LeakyIntegrator::step`) and katgpt-core's sense reconstruction
/// (`ReconstructionState::evolve_belief`). Co-located here (the leaf) so both
/// consumers can share the single source of truth without a cycle.
pub mod leaky_core;
mod looping;
mod lora;
pub mod math;
pub mod merkle;
mod rng;
mod sense;
/// SIMD-accelerated linear algebra kernels (NEON / AVX2 / WASM-SIMD128 /
/// scalar fallback). Co-located with `types` because `types::math` calls
/// these kernels and `simd::ternary` uses `types::TernaryWeights`.
pub mod simd;
pub mod slod;
pub mod temporal;
mod ternary;
/// BITCOS — distribution-adaptive ternary layout: presence bitmap +
/// compacted signs, rate `2−z` bits/weight (Issue 864, `bitcos` feature).
#[cfg(feature = "bitcos")]
pub mod bitcos;
/// Ternary `{-1,0,+1}` bit-planes with per-128 f16 group scale — the
/// `Q2_0_g128` container (Issue 578, `ternary_group_scale` feature).
#[cfg(feature = "ternary_group_scale")]
pub mod ternary_group;
/// Activation-aware (imatrix-spelling) group-scale fit for the `Q2_0_g128`
/// authoring path — the diagonal is a plain `&[f32]` (Issue 886 P1,
/// `act_aware_fit` feature).
#[cfg(feature = "act_aware_fit")]
pub mod ternary_group_act_aware;
/// Ternary `{-1,0,+1}` packed 5-per-byte in base 3 — the 1.75 bits/weight
/// footprint tier (Issue 582, `ternary_trit_pack` feature).
#[cfg(feature = "ternary_trit_pack")]
pub mod ternary_trit;

#[cfg(test)]
mod tests_types;

// Re-export the entire public surface so `katgpt_types::*` paths are
// available at the crate root. Feature gates mirror the gates on the
// underlying items.
#[cfg(feature = "binary_plasma")]
pub use binary::{BinaryWeights, GROUP_SIZE};
#[cfg(feature = "bitcos")]
pub use bitcos::{
    TRIT_CROSSOVER_Z, BitcosWeights, ZeroDensityReport, bitcos_bits_per_weight,
    bitcos_payload_bytes_per_weight, should_use_bitcos, zero_density_report,
};
#[cfg(feature = "ternary_group_scale")]
pub use ternary_group::{
    TernaryBlockAoS, TernaryBlockContiguousWeights, TernaryFfnHook, TernaryGroupWeights,
    TernaryInputProjHook, TernaryMatvecHook,
};
#[cfg(feature = "act_aware_fit")]
pub use ternary_group_act_aware::{ACT_AWARE_SEARCH_GRID, ActAwareScaleFit};
#[cfg(feature = "ternary_trit_pack")]
pub use ternary_trit::{
    TRIT_CODE_LIMIT, TRIT_LUT, TRIT_POW3, TRITS_PER_BYTE, TernaryTritWeights,
};
pub use config::{Config, InferenceOverrides, kv_dim};
#[cfg(feature = "domain_latent")]
pub use domain::DomainLatent;
#[cfg(feature = "deltanet_inference")]
pub use enums::DeltaNetLayerType;
#[cfg(feature = "collapse_aware_thinking")]
pub use enums::ThinkingBudget;
#[cfg(feature = "wall_attention")]
pub use enums::WallConfig;
#[cfg(feature = "gemma4_inference")]
pub use enums::Gemma4LayerType;
pub use enums::{
    AttentionMode, AttentionProjection, CacheLayout, CalibrationMode, ConvergenceSelector,
    CopyLateShape, DashAttnConfig, DepthTier, HlaMode, HybridPattern, LoopMode, LoopStabilityMode,
    ModelArchitecture, ResidualGate, RetrievalHeadRole, RtTurboConfig, SdpaOutputGate, WeightDtype,
};
#[cfg(feature = "sr2am_configurator")]
pub use enums::{ConfiguratorContext, PlanningDecision};
#[cfg(feature = "gpart_adapter")]
pub use gpart::{GPART_MAGIC, GPART_VERSION, GpartAdapter, GpartPair, GpartPrepared};
#[cfg(feature = "hydra_budget")]
pub use hydra::{HydraBudgetConfig, HydraLayerProfile};
pub use inference::InferenceResult;
#[cfg(feature = "data_gate")]
pub use inference::{DataGate, GateDecision, ProposerTask, TaskType};
pub use kv_cache::QuantizedKVCache;
pub use leaky_core::leaky_step;
pub use looping::{CacheStrategy, IterationMode, SubStepStrategy, TrainingFreeLoopConfig};
pub use lora::{LoraAdapter, LoraPair, WeightEpoch, lora_apply};
#[allow(deprecated)]
pub use math::sample_token;
#[cfg(feature = "sparse_mlp")]
pub use math::sparse_matmul;
pub use math::{
    gegelu, gegelu_tanh, matmul, matmul_f16, matmul_f16_f16, matmul_f16_f16_parallel,
    matmul_f16_parallel, matmul_parallel, matmul_relu, rmsnorm, rmsnorm_with_gamma,
    rmsnorm_with_gamma_eps, sample_token_into, silu, situ, softmax, softmax_scaled, swiglu,
    swiglu_inplace,
};
pub use merkle::{
    HASH_SIZE, MERKLE_OCTREE_BRANCHING, MERKLE_OCTREE_DEPTH, MERKLE_OCTREE_INTERNAL,
    MERKLE_OCTREE_LEAVES, MERKLE_OCTREE_NODES, MerkleOctree, MerkleProof,
};
pub use rng::Rng;
#[allow(deprecated)]
pub use sense::ShardEmbedding;
pub use sense::{DilationConfig, SenseKind, SenseModule, TernaryDir};
pub use slod::ScaleBoundary;
pub use temporal::{TemporalDerivativeKernel, sigmoid_surprise_gate};
#[cfg(feature = "plasma_path")]
pub use ternary::{TernaryPackError, TernaryWeights};

// Internal helpers (read_u32_le / read_f32_le / read_u16_le) live in
// `domain.rs` and are crate-private — not re-exported here. If other modules
// need them, import via `crate::domain::read_u32_le`.
