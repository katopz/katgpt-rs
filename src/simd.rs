// katgpt-rs SIMD: re-exports from katgpt-core.
//
// All SIMD kernels (NEON, AVX2, scalar fallbacks) are defined in katgpt-core
// and re-exported here for backward compatibility.

pub use katgpt_core::simd::*;
