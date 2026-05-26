//! katgpt-core: Shared types and SIMD kernels for katgpt-rs and riir-engine.
//!
//! This crate contains the common core shared between the two projects:
//! - **types**: Config, Rng, math utilities, LoRA, DomainLatent
//! - **simd**: NEON/AVX2 accelerated linear algebra kernels
//! - **traits**: Shared traits for game AI and speculative decoding
//!
//! No feature flags on types — both projects get the full superset.

#[cfg(feature = "tiled_attention")]
pub mod attention;
#[cfg(feature = "coda_fusion")]
pub mod coda;
pub mod simd;
pub mod traits;
pub mod types;

// Re-export consolidated traits (Plan 107 Phase 0)
pub use traits::{
    ActionSpaceLog, BinaryScreeningPruner, ConstraintPruner, GameState, NoPruner,
    NoScreeningPruner, RandomRolloutPolicy, RolloutPolicy, ScreeningPruner, StateHeuristic,
};

// Re-export key types at crate root for convenience
pub use types::{
    AttentionMode, Config, ConvergenceSelector, DashAttnConfig, HlaMode, HybridPattern,
    InferenceOverrides, InferenceResult, LoopMode, ModelArchitecture, ResidualGate,
    RetrievalHeadRole, Rng, RtTurboConfig, SdpaOutputGate, WeightDtype, kv_dim, matmul, matmul_f16,
    matmul_f16_parallel, matmul_parallel, matmul_relu, rmsnorm, sample_token, softmax,
    softmax_scaled,
};

#[cfg(feature = "domain_latent")]
pub use types::DomainLatent;

#[cfg(feature = "sr2am_configurator")]
pub use types::{ConfiguratorContext, PlanningDecision};

#[cfg(feature = "data_gate")]
pub use types::{DataGate, GateDecision, ProposerTask, TaskType};

#[cfg(feature = "sparse_mlp")]
pub use types::sparse_matmul;

#[cfg(feature = "coda_fusion")]
pub use coda::{
    GateActivation, compute_rstd, simd_matmul_residual, simd_matmul_residual_partial_rms,
    simd_matmul_rmsnorm_activation, simd_matmul_rmsnorm_rope, simd_matmul_rmsnorm_swiglu,
};

#[cfg(feature = "tiled_attention")]
pub use attention::{tiled_attention_batched, tiled_attention_forward};

pub use simd::SimdLevel;

#[cfg(feature = "questbench")]
pub mod questbench;
#[cfg(feature = "questbench")]
pub use questbench::{
    CspDomain, MemoryTier, QuestBenchDecision, SyntheticCsp, UnderspecConfig, find_sufficient_set,
    generate_synthetic_csps, tier_from_score, underspecification_score,
};

#[cfg(feature = "tf_loop")]
pub use types::{CacheStrategy, IterationMode, SubStepStrategy, TrainingFreeLoopConfig};

#[cfg(feature = "plasma_path")]
pub use types::TernaryWeights;
