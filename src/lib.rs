pub mod benchmark;
#[cfg(feature = "dllm")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub mod dllm;
#[cfg(feature = "feedback")]
pub mod feedback;
#[cfg(feature = "hla_attention")]
pub mod hla;
pub mod percepta;
pub mod plot;
pub mod pruners;
pub mod simd;
#[cfg(feature = "maxsim")]
pub mod rerank;
#[cfg(feature = "sp_kv")]
pub mod sp_kv;
#[cfg(feature = "spectral_quant")]
pub mod spectralquant;
pub mod speculative;
pub mod tokenizer;
pub mod transformer;
#[cfg(feature = "turboquant")]
pub mod turboquant;
pub mod types;

#[cfg(debug_assertions)]
pub mod alloc;

/// Debug-only global allocator that tracks allocation count and bytes.
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;

#[cfg(feature = "validator")]
pub mod validator;
