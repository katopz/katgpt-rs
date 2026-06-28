//! Minimal standalone SIMD kernels for katgpt-dec.
//!
//! katgpt-dec is a zero-dependency pure-math substrate (no `katgpt-core` dep)
//! so that `katgpt-core` can re-export it as `katgpt_core::dec` without
//! creating a cyclic package dependency. The two kernels here —
//! `simd_dot_f32` and `simd_sigmoid_inplace` — are the only SIMD surfaces
//! katgpt-dec needs; they are scalar implementations written to auto-vectorize
//! well on modern targets (NEON / AVX2 / wasm32 simd128 via LLVM).
//!
//! The full platform-specific NEON/AVX2/wasm32 intrinsics dispatch lives in
//! `katgpt-core::simd`; consumers that need the hand-tuned intrinsics paths
//! should call through `katgpt_core::simd::*` directly. The implementations
//! here are bit-compatible with the scalar fallbacks in katgpt-core (same
//! `mul_add` FMA semantics, same `fast_sigmoid` libm-exp contract).

/// Bounded sigmoid: σ(x) = 1/(1 + e^{-x}), output in (0, 1).
///
/// Uses `f32::exp()` via the platform's libm (hardware-accelerated on aarch64).
/// Early-exit for |x| > 40 where σ saturates to 0 or 1 in f32 precision.
#[inline(always)]
pub fn fast_sigmoid(x: f32) -> f32 {
    // sigmoid(40) = 1/(1 + e^{-40}) ≈ 1 - 4.2e-18, rounds to 1.0 in f32.
    // sigmoid(-40) ≈ 4.2e-18, rounds to 0.0 in f32.
    if x > 40.0 {
        return 1.0;
    }
    if x < -40.0 {
        return 0.0;
    }
    1.0 / (1.0 + (-x).exp())
}

/// In-place sigmoid: `x[i] = σ(x[i]) = 1/(1 + e^{-x[i]})`.
///
/// Scalar loop over [`fast_sigmoid`]; LLVM auto-vectorizes this on targets
/// with hardware SIMD (NEON / AVX2 / wasm32 simd128).
#[inline]
pub fn simd_sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = fast_sigmoid(*v);
    }
}

/// Dot product: `Σ a[i] * b[i]` for `len` elements.
///
/// 4-accumulator FMA form — keeps the FMA pipeline full and lets LLVM emit
/// 4-wide unrolled FMA on targets without hardware f32 SIMD. `mul_add`
/// preserves single-rounding FMA semantics on hardware that has it.
#[inline(always)]
pub fn simd_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] = (*a.get_unchecked(i)).mul_add(*b.get_unchecked(i), acc[0]);
            acc[1] = (*a.get_unchecked(i + 1)).mul_add(*b.get_unchecked(i + 1), acc[1]);
            acc[2] = (*a.get_unchecked(i + 2)).mul_add(*b.get_unchecked(i + 2), acc[2]);
            acc[3] = (*a.get_unchecked(i + 3)).mul_add(*b.get_unchecked(i + 3), acc[3]);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < len {
        unsafe {
            sum = (*a.get_unchecked(i)).mul_add(*b.get_unchecked(i), sum);
        }
        i += 1;
    }
    sum
}
