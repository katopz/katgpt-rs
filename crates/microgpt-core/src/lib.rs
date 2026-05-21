//! microgpt-core: Shared types and SIMD kernels for microgpt-rs and riir-engine.
//!
//! This crate contains the common core shared between the two projects:
//! - **types**: Config, Rng, math utilities, LoRA, DomainLatent
//! - **simd**: NEON/AVX2 accelerated linear algebra kernels
//!
//! No feature flags on types — both projects get the full superset.

pub mod simd;
pub mod types;

// Re-export key types at crate root for convenience
pub use types::{
    AttentionMode, Config, HlaMode, InferenceOverrides, InferenceResult, ModelArchitecture, Rng,
    WeightDtype, kv_dim, matmul, matmul_relu, rmsnorm, sample_token, softmax, softmax_scaled,
};

#[cfg(feature = "domain_latent")]
pub use types::DomainLatent;

#[cfg(feature = "sparse_mlp")]
pub use types::sparse_matmul;

pub use simd::SimdLevel;
