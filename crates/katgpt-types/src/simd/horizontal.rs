//! Shared AVX2 horizontal reducers — `_mm256`/`__m128` → f32 reduction helpers.
//!
//! Used by every AVX2 kernel that ends with an `__m256` accumulator: dot,
//! sum, max, sum_sq, sum_abs, dist_sq, exp_sum, sparse_dot, ternary_matvec.
//!
//! `pub(super)` — callers live in sibling `simd::*` submodules only.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) fn horizontal_max_256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::{
        _mm_cvtss_f32, _mm_max_ps, _mm_shuffle_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
    };
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m = _mm_max_ps(lo, hi);
    // Reduce 4 lanes via shuffle+max
    let shuf = _mm_shuffle_ps(m, m, 0xB1);
    let m2 = _mm_max_ps(m, shuf);
    let shuf2 = _mm_shuffle_ps(m2, m2, 0x4E);
    let m3 = _mm_max_ps(m2, shuf2);
    _mm_cvtss_f32(m3)
}

// ── x86_64 Horizontal Sum Helpers ─────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) fn horizontal_sum_256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::{_mm_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps};
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    horizontal_sum_128(sum128)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) fn horizontal_sum_128(v: core::arch::x86_64::__m128) -> f32 {
    // Safety: pure SIMD intrinsic computation, no memory access.
    use core::arch::x86_64::{_mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_shuffle_ps};
    unsafe {
        let shuf = _mm_shuffle_ps(v, v, 0xB1);
        let sums = _mm_add_ps(v, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x2A);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
}
