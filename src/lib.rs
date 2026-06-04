pub mod benchmark;
#[cfg(feature = "cache_prune")]
pub mod cache_prune;
#[cfg(feature = "dash_attn")]
pub mod dash_attn;
#[cfg(feature = "data_probe")]
pub mod data_probe;
#[cfg(any(feature = "peira_distill", feature = "ilc_distill"))]
pub mod distill;
#[cfg(feature = "dllm")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::needless_range_loop)]
pub mod dllm;
#[cfg(feature = "ega_attn")]
pub mod ega_attn;
#[cfg(feature = "feedback")]
pub mod feedback;
#[cfg(feature = "gdn2_attention")]
pub mod gdn2;
#[cfg(feature = "hla_attention")]
pub mod hla;
#[cfg(feature = "hybrid_oct_pq")]
pub mod hybrid_oct_pq;
pub mod inference_backend;
#[cfg(all(target_os = "macos", feature = "ane"))]
pub mod ane_backend;
#[cfg(feature = "iso_quant")]
pub mod iso_quant;
#[cfg(feature = "kog_cpu_fusion")]
pub mod mbu;
#[cfg(feature = "newton_schulz")]
pub mod newton_schulz;
#[cfg(feature = "octopus")]
pub mod octopus;
pub mod percepta;
#[cfg(feature = "planar_quant")]
pub mod planar_quant;
pub mod plot;
#[cfg(feature = "proof_cert")]
pub mod proof_cert;
pub mod pruners;
#[cfg(feature = "maxsim")]
pub mod rerank;
#[cfg(feature = "river_valley")]
pub mod river_valley;
#[cfg(feature = "rt_turbo")]
pub mod rt_turbo;
#[cfg(feature = "shard_kv")]
pub mod shard_kv;
pub mod simd;
#[cfg(feature = "skill_opt")]
pub mod skill_opt;
#[cfg(feature = "sleep_consolidation")]
pub mod sleep;
#[cfg(feature = "sp_kv")]
pub mod sp_kv;
#[cfg(feature = "spechop")]
pub mod spechop;
#[cfg(feature = "spectral_quant")]
pub mod spectralquant;
pub mod speculative;
#[cfg(feature = "stiff_anomaly")]
pub mod stiff_anomaly;
pub mod tokenizer;
pub mod transformer;
pub mod trigger_gate;
#[cfg(feature = "turboquant")]
pub mod turboquant;
pub mod types;
#[cfg(feature = "unit_distance")]
pub mod unit_distance;
pub mod weights;

#[cfg(debug_assertions)]
pub mod alloc;

/// Debug-only global allocator that tracks allocation count and bytes.
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: alloc::TrackingAllocator = alloc::TrackingAllocator;

#[cfg(feature = "validator")]
pub mod validator;

#[cfg(feature = "tf_loop")]
pub mod tf_loop;
