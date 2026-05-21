// microgpt-rs SIMD: re-exports from microgpt-core.
//
// All SIMD kernels (NEON, AVX2, scalar fallbacks) are defined in microgpt-core
// and re-exported here for backward compatibility.

pub use microgpt_core::simd::*;
