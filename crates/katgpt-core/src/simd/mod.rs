//! SIMD-accelerated linear algebra kernels for inference.
//!
//! Provides NEON (aarch64), AVX2 (x86_64), and scalar backends for the hot-path
//! operations used throughout the crate:
//! - Dot products, outer-product accumulator, matvec, and matmul variants
//! - Sparse dot / sparse matmul (gather-based, for active-tokens-only matmul)
//! - Elementwise ops (scale / add / sum / max / fused-decay / scale-mul)
//! - Activations (exp / sigmoid / tanh-clamp / reciprocal / fast_sigmoid)
//!   — backed by a 6th-order Cephes polynomial for `exp` (~1 ULP, |x| < 88)
//! - Argmax (single-pass `(usize, f32)`)
//! - MaxSim late-interaction scoring
//! - Ternary bit-plane matvec (multiplication-free, `plasma_path` feature)
//! - Research primitives (sigmoid margin, retrieval margin, Gram, entropy,
//!   coincidence, sum_sq / sum_abs / dist_sq / fused_sub_acc / fused_scale_acc)
//!
//! # Dispatch
//!
//! Runtime detection picks the best backend: NEON is mandatory on `aarch64`;
//! AVX2+FMA is detected via `cpuid` on `x86_64` (cached in an `AtomicBool`);
//! everything else falls back to the 4-accumulator scalar form.
//!
//! # Stability
//!
//! Uses `core::arch` intrinsics directly — stable on both `aarch64` and
//! `x86_64`. No nightly features, no external SIMD crates.
//!
//! # Module layout
//!
//! This file is the dispatcher surface. Backends live in sibling files:
//! - [`dot`] — dot products, outer-product, matvec, matmul
//! - [`sparse`] — sparse dot / sparse matmul
//! - [`elementwise`] — scale / add / sum / max / fused ops
//! - [`activations`] — exp / sigmoid / tanh-clamp / reciprocal / fast_sigmoid
//! - [`argmax`] — single-pass argmax
//! - [`maxsim`] — ColBERT-style late-interaction scoring
//! - [`ternary`] — bit-plane ternary matvec (`plasma_path`)
//! - [`research`] — sigmoid margin, retrieval margin, Gram, entropy, norms
//! - [`horizontal`] — shared AVX2 horizontal reducers (`pub(super)`)

// Submodule backends. Each file owns its NEON/AVX2/scalar impls verbatim;
// only `is_avx2_fma_available` (below) and `horizontal::*` are shared.
mod activations;
mod argmax;
mod dot;
mod elementwise;
mod horizontal;
mod maxsim;
mod research;
mod sparse;
mod ternary;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_sense;

// Test-only imports: bring the scalar reference implementations (each
// `pub(super)` in its backend submodule) into `simd::` scope so the test
// modules can call them via bare names through `use super::*`. Only the
// scalars actually referenced by `tests.rs` are imported.
//
// NOTE: this block exists because `simd.rs` was recently split into a
// directory (`simd/`); the test file still references the scalar helpers
// that used to be in the same module. This is a minimal fix to unblock test
// compilation; a fuller refactor would move the scalar helpers into a
// dedicated `simd/scalar_ref.rs` submodule.
#[cfg(test)]
use dot::{scalar_dot_f32, scalar_outer_product_acc};
#[cfg(test)]
use elementwise::{scalar_add_inplace, scalar_add_into, scalar_add_scalar_inplace, scalar_fused_decay_write, scalar_max_f32, scalar_scale_inplace, scalar_sum_f32};
#[cfg(test)]
use research::scalar_sum_sq_quartic;
#[cfg(test)]
use sparse::scalar_sparse_dot_f32;

// Re-export the entire public surface so `crate::simd::*` paths are unchanged
// after the file → folder split. Existing call sites (e.g. `simd::simd_dot_f32`,
// `simd::SimdLevel`) continue to resolve without modification.
pub use activations::{fast_sigmoid, simd_exp_inplace, simd_exp_sum_inplace, simd_reciprocal_inplace, simd_sigmoid_inplace, simd_sigmoid_tanh_clamp_inplace};
pub use argmax::simd_argmax_f32;
pub use dot::{simd_dot_f16_f32, simd_dot_f32, simd_fma_row, simd_matmul_f16_f32_rows, simd_matmul_f16_f32_rows_parallel, simd_matmul_relu_rows, simd_matmul_rows, simd_matmul_rows_parallel, simd_matvec, simd_outer_product_acc, simd_outer_product_acc_scaled};
pub use elementwise::{simd_add_inplace, simd_add_into, simd_add_scalar_inplace, simd_fused_decay_write, simd_fused_sub_scale_inplace, simd_max_f32, simd_scale_inplace, simd_scale_mul_inplace, simd_sum_f32};
// Feature-gated re-exports — mirror the `#[cfg(feature = "...")]` gates on
// the underlying items so `cargo check --no-default-features` stays green.
#[cfg(feature = "maxsim")]
pub use maxsim::{maxsim_score, maxsim_score_packed};
#[cfg(feature = "sigmoid_margin")]
pub use research::{compute_retrieval_margin, dim_sufficiency_bound, sigmoid_margin_loss};
pub use research::{coincidence_score, entropy_f32, simd_dist_sq, simd_fused_scale_acc, simd_fused_sub_acc, simd_gram_f32, simd_sum_abs_f32, simd_sum_sq, simd_sum_sq_quartic};
pub use sparse::{simd_sparse_dot_f32, simd_sparse_matmul_rows};
pub use ternary::simd_ternary_dot_f32;
#[cfg(feature = "plasma_path")]
pub use ternary::{simd_ternary_matmul_batch, simd_ternary_matvec, ternary_matvec_scalar};

/// SIMD capability level detected at runtime.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD — scalar fallback.
    Scalar,
    /// ARM NEON (4× f32 per operation).
    Neon,
    /// x86 AVX2+FMA (8× f32 per operation).
    Avx2,
    /// WASM SIMD128 (4× f32 per operation) — compile-time gated by `target_feature = "simd128"`.
    WasmSimd128,
}

/// Detect the best available SIMD level for the current CPU.
///
/// On `aarch64`: always returns [`SimdLevel::Neon`] (mandatory on ARMv8+).
/// On `x86_64`: returns [`SimdLevel::Avx2`] if CPU supports AVX2+FMA, else [`SimdLevel::Scalar`].
/// On `wasm32` with `+simd128`: returns [`SimdLevel::WasmSimd128`] (compile-time feature gate).
/// On other architectures: returns [`SimdLevel::Scalar`].
#[inline]
pub fn simd_level() -> SimdLevel {
    #[cfg(target_arch = "aarch64")]
    {
        SimdLevel::Neon
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            SimdLevel::Avx2
        } else {
            SimdLevel::Scalar
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        SimdLevel::WasmSimd128
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        SimdLevel::Scalar
    }
}

// ── x86_64 Runtime Detection ─────────────────────────────────

/// Detect AVX2+FMA support on x86_64. Cached after first call.
///
/// `pub(super)` — every dispatcher in `dot`/`elementwise`/`activations`/etc.
/// calls this to pick between the AVX2 and scalar paths.
#[cfg(target_arch = "x86_64")]
pub(super) fn is_avx2_fma_available() -> bool {
    #[cfg(target_feature = "avx2")]
    {
        true
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static CACHED: AtomicBool = AtomicBool::new(false);
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let cpuid1 = unsafe { core::arch::x86_64::__cpuid(1) };
            let has_avx = (cpuid1.ecx & (1 << 28)) != 0;
            let has_fma = (cpuid1.ecx & (1 << 12)) != 0;
            let cpuid7 = unsafe { core::arch::x86_64::__cpuid(7) };
            let has_avx2 = (cpuid7.ebx & (1 << 5)) != 0;
            CACHED.store(has_avx && has_fma && has_avx2, Ordering::Relaxed);
        });
        CACHED.load(Ordering::Relaxed)
    }
}
