//! SIMD elementwise kernels — scale, add, sum, max, fused ops.
//!
//! Dispatchers + NEON/AVX2/scalar impls for:
//! - `simd_scale_inplace`, `simd_add_scalar_inplace`, `simd_fused_sub_scale_inplace`
//! - `simd_sum_f32`, `simd_add_inplace`, `simd_add_into`, `simd_max_f32`
//! - `simd_fused_decay_write`, `simd_scale_mul_inplace`
//!
//! AVX2 paths share horizontal reducers from `super::horizontal`.

// ── Scale Inplace ─────────────────────────────────────────────

/// SIMD-accelerated in-place scale: `x[i] *= scale` for all `i`.
///
/// General utility for softmax normalize, rmsnorm scale, HLA decay,
/// TurboQuant normalize, and any bulk `*= scale` pattern.
///
/// NEON: 4× f32 per op. AVX2: 8× f32 per op. Scalar fallback for remainder.

#[inline(always)]
pub fn simd_scale_inplace(x: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_scale_inplace(x, scale) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_scale_inplace(x, scale) }
        } else {
            scalar_scale_inplace(x, scale)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_scale_inplace(x, scale)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_scale_inplace(x: &mut [f32], scale: f32) {
    for val in x.iter_mut() {
        *val *= scale;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_scale_inplace(x: &mut [f32], scale: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};

    unsafe {
        let vs = vdupq_n_f32(scale);
        let mut i = 0;
        let chunks = x.len() / 4;

        for _ in 0..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let result = vmulq_f32(vx, vs);
            vst1q_f32(x.as_mut_ptr().add(i), result);
            i += 4;
        }

        while i < x.len() {
            *x.get_unchecked_mut(i) *= scale;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_scale_inplace(x: &mut [f32], scale: f32) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};

    unsafe {
        let vs = _mm256_set1_ps(scale);
        let mut i = 0;
        let chunks = x.len() / 8;

        for _ in 0..chunks {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let result = _mm256_mul_ps(vx, vs);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
            i += 8;
        }

        while i < x.len() {
            *x.get_unchecked_mut(i) *= scale;
            i += 1;
        }
    }
}

/// SIMD-accelerated in-place broadcast add: `x[i] += val` for all `i`.
///
/// Used for softmax max-subtraction: `scores[i] -= max_score`.
/// NEON: 4× f32 per `vaddq_f32`. AVX2: 8× f32 per `_mm256_add_ps`.
#[inline(always)]
pub fn simd_add_scalar_inplace(x: &mut [f32], val: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_add_scalar_inplace(x, val) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_add_scalar_inplace(x, val) }
        } else {
            scalar_add_scalar_inplace(x, val)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_add_scalar_inplace(x, val)
    }
}

/// SIMD-accelerated fused subtract and scale: `x[i] = (x[i] - sub) * scale` for all `i`.
///
/// Fuses two operations into one pass, saving one full SIMD traversal vs separate
/// `simd_add_scalar_inplace` + `simd_scale_inplace`.
#[inline(always)]
pub fn simd_fused_sub_scale_inplace(x: &mut [f32], sub: f32, scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_sub_scale_inplace(x, sub, scale) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_fused_sub_scale_inplace(x, sub, scale) }
        } else {
            scalar_fused_sub_scale_inplace(x, sub, scale)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_fused_sub_scale_inplace(x, sub, scale)
    }
}

/// SIMD-accelerated horizontal sum: returns `x[0] + x[1] + ... + x[n-1]`.
///
/// Used for softmax denominator computation (sum of exp-shifted scores).
/// NEON: uses `vaddvq_f32` for 4-lane horizontal sum. AVX2: uses 8-lane reduce.
#[inline(always)]
pub fn simd_sum_f32(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sum_f32(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sum_f32(x) }
        } else {
            scalar_sum_f32(x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_sum_f32(x)
    }
}

/// SIMD-accelerated in-place add: `dst[i] += src[i]` for all `i`.
///
/// Used for residual connections in transformer forward pass (attn + MLP).
/// NEON: 4× f32 per `vaddq_f32`. AVX2: 8× f32 per `_mm256_add_ps`.
#[inline(always)]
pub fn simd_add_inplace(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(
        dst.len(),
        src.len(),
        "simd_add_inplace: slices must have equal length"
    );
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_add_inplace(dst, src) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_add_inplace(dst, src) }
        } else {
            scalar_add_inplace(dst, src)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_add_inplace(dst, src)
    }
}

/// SIMD-accelerated zip add: `dst[i] = a[i] + b[i]` for all `i`.
///
/// Used for embedding addition (wte + wpe) in transformer forward pass.
/// NEON: 4× f32 per `vaddq_f32`. AVX2: 8× f32 per `_mm256_add_ps`.
#[inline(always)]
pub fn simd_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(
        dst.len(),
        a.len(),
        "simd_add_into: dst and a must have equal length"
    );
    debug_assert_eq!(
        dst.len(),
        b.len(),
        "simd_add_into: dst and b must have equal length"
    );
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_add_into(dst, a, b) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_add_into(dst, a, b) }
        } else {
            scalar_add_into(dst, a, b)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_add_into(dst, a, b)
    }
}

/// SIMD-accelerated max reduction: returns the maximum value in `x`.
///
/// Used for softmax numerical stability (pass 1 max-finding).
/// NEON: 4× f32 per `vmaxq_f32`. AVX2: 8× f32 per `_mm256_max_ps`.
#[inline(always)]
pub fn simd_max_f32(x: &[f32]) -> f32 {
    if x.is_empty() {
        return f32::NEG_INFINITY;
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_max_f32(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_max_f32(x) }
        } else {
            scalar_max_f32(x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_max_f32(x)
    }
}

/// SIMD-accelerated fused decay-write: `dst[i] = decay * dst[i] + write * src[i]`.
///
/// Used for Raven KV cache update (exponential moving average with gating).
/// NEON: fused via `vfmaq_f32`. AVX2: fused via `_mm256_fmadd_ps`.
#[inline(always)]
pub fn simd_fused_decay_write(dst: &mut [f32], decay: f32, src: &[f32], write: f32) {
    debug_assert_eq!(
        dst.len(),
        src.len(),
        "simd_fused_decay_write: slices must have equal length"
    );
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_decay_write(dst, decay, src, write) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_fused_decay_write(dst, decay, src, write) }
        } else {
            scalar_fused_decay_write(dst, decay, src, write)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_fused_decay_write(dst, decay, src, write)
    }
}

/// SIMD-accelerated fused scale+multiply: `x[i] = gamma[i] * x[i] * scale`.
///
/// Used for `rmsnorm_with_gamma`: fuses the inv_rms scale and learnable gamma
/// multiply into a single pass, saving one full buffer scan vs separate
/// `simd_scale_inplace` + elementwise multiply.
///
/// NEON: fused via `vmulq_f32` (2 multiplies per 4 elements).
/// AVX2: fused via `_mm256_mul_ps` (2 multiplies per 8 elements).
#[inline(always)]
pub fn simd_scale_mul_inplace(x: &mut [f32], gamma: &[f32], scale: f32) {
    debug_assert_eq!(
        x.len(),
        gamma.len(),
        "simd_scale_mul_inplace: slices must have equal length"
    );
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_scale_mul_inplace(x, gamma, scale) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_scale_mul_inplace(x, gamma, scale) }
        } else {
            scalar_scale_mul_inplace(x, gamma, scale)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_scale_mul_inplace(x, gamma, scale)
    }
}


// ── Scalar Fallbacks (new primitives) ─────────────────────────

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_add_inplace(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_add_scalar_inplace(x: &mut [f32], val: f32) {
    for v in x.iter_mut() {
        *v += val;
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_fused_sub_scale_inplace(x: &mut [f32], sub: f32, scale: f32) {
    // Rewrite (v - sub) * scale as v * scale + (-sub * scale) to enable FMA.
    // The additive term is loop-invariant — hoist outside the per-element work.
    let bias = -(sub * scale);
    for v in x.iter_mut() {
        *v = v.mul_add(scale, bias);
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_sum_f32(x: &[f32]) -> f32 {
    // 4 independent accumulators — addition is latency-bound on a single
    // accumulator (~3 cycles/iter on most FPUs). Unrolling 4-wide keeps the
    // adder pipeline full and helps LLVM auto-vectorize on targets without
    // dedicated f32 SIMD.
    let mut acc = [0.0f32; 4];
    let chunks = x.len() / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] += *x.get_unchecked(i);
            acc[1] += *x.get_unchecked(i + 1);
            acc[2] += *x.get_unchecked(i + 2);
            acc[3] += *x.get_unchecked(i + 3);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < x.len() {
        unsafe {
            sum += *x.get_unchecked(i);
        }
        i += 1;
    }
    sum
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_max_f32(x: &[f32]) -> f32 {
    let mut max = x[0];
    for i in 1..x.len() {
        // Branch-free: matches the SIMD path's vmaxq_f32 / _mm256_max_ps NaN semantics
        // (propagates NaN), unlike `if v > max` which silently drops NaN.
        max = max.max(unsafe { *x.get_unchecked(i) });
    }
    max
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_fused_decay_write(dst: &mut [f32], decay: f32, src: &[f32], write: f32) {
    for i in 0..dst.len() {
        unsafe {
            let d = *dst.get_unchecked(i);
            let s = *src.get_unchecked(i);
            // FMA: write * src + decay * dst (one FMA + one mul).
            // Picked `decay * d` as the FMA product since `decay` is typically the
            // recurrence weight (close to 1.0) — keeping it inside FMA preserves
            // the most precision in the dominant term.
            *dst.get_unchecked_mut(i) = decay.mul_add(d, write * s);
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_scale_mul_inplace(x: &mut [f32], gamma: &[f32], scale: f32) {
    // No FMA opportunity: x[i] = gamma[i] * x[i] * scale is a pure product of
    // three factors. The two-multiply form is already optimal.
    for i in 0..x.len() {
        unsafe {
            *x.get_unchecked_mut(i) = *gamma.get_unchecked(i) * *x.get_unchecked(i) * scale;
        }
    }
}

// ── NEON Backend (new primitives) ─────────────────────────────

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_add_inplace(dst: &mut [f32], src: &[f32]) {
    use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
    unsafe {
        let mut i = 0;
        let chunks = dst.len() / 4;
        for _ in 0..chunks {
            let vd = vld1q_f32(dst.as_ptr().add(i));
            let vs = vld1q_f32(src.as_ptr().add(i));
            let result = vaddq_f32(vd, vs);
            vst1q_f32(dst.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < dst.len() {
            *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_add_scalar_inplace(x: &mut [f32], val: f32) {
    use core::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vst1q_f32};
    unsafe {
        let vv = vdupq_n_f32(val);
        let mut i = 0;
        let chunks = x.len() / 4;
        for _ in 0..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let result = vaddq_f32(vx, vv);
            vst1q_f32(x.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < x.len() {
            *x.get_unchecked_mut(i) += val;
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_fused_sub_scale_inplace(x: &mut [f32], sub: f32, scale: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};
    unsafe {
        let sub_vec = vdupq_n_f32(sub);
        let scale_vec = vdupq_n_f32(scale);
        let mut i = 0;
        let chunks = x.len() / 4;
        for _ in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i));
            let result = vmulq_f32(vsubq_f32(v, sub_vec), scale_vec);
            vst1q_f32(x.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < x.len() {
            *x.get_unchecked_mut(i) = (*x.get_unchecked(i) - sub) * scale;
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sum_f32(x: &[f32]) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32};
    unsafe {
        // 4 independent accumulators to hide FADD pipeline latency.
        // The serial single-accumulator reduction is latency-bound (each add
        // depends on the previous); 4 lanes let the pipeline stay full.
        // Same associativity-reorder already used by `neon_dot_f32`.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let len = x.len();
        let chunks4 = len / 16;

        for _ in 0..chunks4 {
            acc0 = vaddq_f32(acc0, vld1q_f32(x.as_ptr().add(i)));
            acc1 = vaddq_f32(acc1, vld1q_f32(x.as_ptr().add(i + 4)));
            acc2 = vaddq_f32(acc2, vld1q_f32(x.as_ptr().add(i + 8)));
            acc3 = vaddq_f32(acc3, vld1q_f32(x.as_ptr().add(i + 12)));
            i += 16;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        let mut acc_rem = vdupq_n_f32(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            acc_rem = vaddq_f32(acc_rem, vld1q_f32(x.as_ptr().add(i)));
            i += 4;
        }
        sum += vaddvq_f32(acc_rem);

        while i < len {
            sum += *x.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
    unsafe {
        let mut i = 0;
        let chunks = dst.len() / 4;
        for _ in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let result = vaddq_f32(va, vb);
            vst1q_f32(dst.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < dst.len() {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_max_f32(x: &[f32]) -> f32 {
    use core::arch::aarch64::{vld1q_f32, vmaxq_f32, vmaxvq_f32};
    unsafe {
        let len = x.len();
        let chunks = len / 4;
        if chunks == 0 {
            let mut max = x[0];
            for j in 1..len {
                let v = *x.get_unchecked(j);
                if v > max {
                    max = v;
                }
            }
            return max;
        }

        // 4 independent vector-max accumulators to hide the max-op latency.
        // max is associative + commutative, so this is bit-identical to serial.
        let mut vmax0 = vld1q_f32(x.as_ptr());
        let mut vmax1 = vmax0;
        let mut vmax2 = vmax0;
        let mut vmax3 = vmax0;
        let mut i = 0;
        let chunks4 = len / 16;
        for _ in 0..chunks4 {
            vmax0 = vmaxq_f32(vmax0, vld1q_f32(x.as_ptr().add(i)));
            vmax1 = vmaxq_f32(vmax1, vld1q_f32(x.as_ptr().add(i + 4)));
            vmax2 = vmaxq_f32(vmax2, vld1q_f32(x.as_ptr().add(i + 8)));
            vmax3 = vmaxq_f32(vmax3, vld1q_f32(x.as_ptr().add(i + 12)));
            i += 16;
        }
        let mut vmax = vmaxq_f32(vmaxq_f32(vmax0, vmax1), vmaxq_f32(vmax2, vmax3));
        // Remaining 4-element chunks
        while i + 4 <= len {
            vmax = vmaxq_f32(vmax, vld1q_f32(x.as_ptr().add(i)));
            i += 4;
        }

        // Horizontal max of 4 lanes — single `vmaxvq_f32` instruction replaces
        // the transmute + scalar loop. Mandatory on ARMv8+.
        let mut max = vmaxvq_f32(vmax);

        // Handle tail
        while i < len {
            let v = *x.get_unchecked(i);
            if v > max {
                max = v;
            }
            i += 1;
        }
        max
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_fused_decay_write(dst: &mut [f32], decay: f32, src: &[f32], write: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    unsafe {
        let vd_decay = vdupq_n_f32(decay);
        let vd_write = vdupq_n_f32(write);
        let mut i = 0;
        let chunks = dst.len() / 4;
        for _ in 0..chunks {
            let vdst = vld1q_f32(dst.as_ptr().add(i));
            let vsrc = vld1q_f32(src.as_ptr().add(i));
            // FMA: write * vsrc + decay * vdst
            let result = vfmaq_f32(vmulq_f32(vdst, vd_decay), vsrc, vd_write);
            vst1q_f32(dst.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < dst.len() {
            *dst.get_unchecked_mut(i) =
                decay * *dst.get_unchecked(i) + write * *src.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_scale_mul_inplace(x: &mut [f32], gamma: &[f32], scale: f32) {
    use core::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    unsafe {
        let vs = vdupq_n_f32(scale);
        let mut i = 0;
        let chunks = x.len() / 4;
        for _ in 0..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let vg = vld1q_f32(gamma.as_ptr().add(i));
            let result = vmulq_f32(vg, vmulq_f32(vx, vs));
            vst1q_f32(x.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < x.len() {
            *x.get_unchecked_mut(i) = *gamma.get_unchecked(i) * *x.get_unchecked(i) * scale;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_add_inplace(dst: &mut [f32], src: &[f32]) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    unsafe {
        let mut i = 0;
        let chunks = dst.len() / 8;
        for _ in 0..chunks {
            let vd = _mm256_loadu_ps(dst.as_ptr().add(i));
            let vs = _mm256_loadu_ps(src.as_ptr().add(i));
            let result = _mm256_add_ps(vd, vs);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), result);
            i += 8;
        }
        while i < dst.len() {
            *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_add_scalar_inplace(x: &mut [f32], val: f32) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    unsafe {
        let vv = _mm256_set1_ps(val);
        let mut i = 0;
        let chunks = x.len() / 8;
        for _ in 0..chunks {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let result = _mm256_add_ps(vx, vv);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
            i += 8;
        }
        while i < x.len() {
            *x.get_unchecked_mut(i) += val;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn avx2_fused_sub_scale_inplace(x: &mut [f32], sub: f32, scale: f32) {
    use core::arch::x86_64::{
        _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps, _mm_sub_ps, _mm256_loadu_ps,
        _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };
    unsafe {
        let sub_vec = _mm256_set1_ps(sub);
        let scale_vec = _mm256_set1_ps(scale);
        let mut i = 0;
        let chunks = x.len() / 8;
        for _ in 0..chunks {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            let subbed = _mm256_sub_ps(v, sub_vec);
            let result = _mm256_mul_ps(subbed, scale_vec);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
            i += 8;
        }
        // Handle remaining 4-element chunk with SSE
        if i + 4 <= x.len() {
            let sub_128 = _mm_set1_ps(sub);
            let scale_128 = _mm_set1_ps(scale);
            let v = _mm_loadu_ps(x.as_ptr().add(i));
            let subbed = _mm_sub_ps(v, sub_128);
            let result = _mm_mul_ps(subbed, scale_128);
            _mm_storeu_ps(x.as_mut_ptr().add(i), result);
            i += 4;
        }
        while i < x.len() {
            *x.get_unchecked_mut(i) = (*x.get_unchecked(i) - sub) * scale;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_sum_f32(x: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_setzero_ps};
    unsafe {
        // 4 independent accumulators to hide FADD pipeline latency.
        // Same associativity-reorder already used by `avx2_dot_f32`.
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        let len = x.len();
        let chunks4 = len / 32;

        for _ in 0..chunks4 {
            acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(x.as_ptr().add(i)));
            acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(x.as_ptr().add(i + 8)));
            acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(x.as_ptr().add(i + 16)));
            acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(x.as_ptr().add(i + 24)));
            i += 32;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = horizontal_sum_256(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ));

        let mut acc = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(x.as_ptr().add(i)));
            i += 8;
        }
        sum += horizontal_sum_256(acc);

        while i < len {
            sum += *x.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    unsafe {
        let mut i = 0;
        let chunks = dst.len() / 8;
        for _ in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let result = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), result);
            i += 8;
        }
        while i < dst.len() {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_max_f32(x: &[f32]) -> f32 {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_max_ps};
    unsafe {
        let len = x.len();
        let chunks = len / 8;
        if chunks == 0 {
            let mut max = x[0];
            for j in 1..len {
                let v = *x.get_unchecked(j);
                if v > max {
                    max = v;
                }
            }
            return max;
        }

        // 4 independent vector-max accumulators to hide the max-op latency.
        // max is associative + commutative, so this is bit-identical to serial.
        let mut vmax0 = _mm256_loadu_ps(x.as_ptr());
        let mut vmax1 = vmax0;
        let mut vmax2 = vmax0;
        let mut vmax3 = vmax0;
        let mut i = 0;
        let chunks4 = len / 32;
        for _ in 0..chunks4 {
            vmax0 = _mm256_max_ps(vmax0, _mm256_loadu_ps(x.as_ptr().add(i)));
            vmax1 = _mm256_max_ps(vmax1, _mm256_loadu_ps(x.as_ptr().add(i + 8)));
            vmax2 = _mm256_max_ps(vmax2, _mm256_loadu_ps(x.as_ptr().add(i + 16)));
            vmax3 = _mm256_max_ps(vmax3, _mm256_loadu_ps(x.as_ptr().add(i + 24)));
            i += 32;
        }
        let mut vmax = _mm256_max_ps(_mm256_max_ps(vmax0, vmax1), _mm256_max_ps(vmax2, vmax3));
        // Remaining 8-element chunks
        while i + 8 <= len {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x.as_ptr().add(i)));
            i += 8;
        }

        // Horizontal max reduction
        let mut max = horizontal_max_256(vmax);

        // Handle tail
        while i < len {
            let v = *x.get_unchecked(i);
            if v > max {
                max = v;
            }
            i += 1;
        }
        max
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_fused_decay_write(dst: &mut [f32], decay: f32, src: &[f32], write: f32) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };

    unsafe {
        let vd = _mm256_set1_ps(decay);
        let vw = _mm256_set1_ps(write);
        let mut i = 0;
        let chunks = dst.len() / 8;

        for _ in 0..chunks {
            let vdst = _mm256_loadu_ps(dst.as_ptr().add(i));
            let vsrc = _mm256_loadu_ps(src.as_ptr().add(i));
            let result = _mm256_add_ps(_mm256_mul_ps(vd, vdst), _mm256_mul_ps(vw, vsrc));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), result);
            i += 8;
        }

        while i < dst.len() {
            *dst.get_unchecked_mut(i) =
                decay * *dst.get_unchecked(i) + write * *src.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_scale_mul_inplace(x: &mut [f32], gamma: &[f32], scale: f32) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps};

    unsafe {
        let vs = _mm256_set1_ps(scale);
        let mut i = 0;
        let chunks = x.len() / 8;

        for _ in 0..chunks {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            let vg = _mm256_loadu_ps(gamma.as_ptr().add(i));
            let result = _mm256_mul_ps(vg, _mm256_mul_ps(vx, vs));
            _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
            i += 8;
        }

        while i < x.len() {
            *x.get_unchecked_mut(i) = *gamma.get_unchecked(i) * *x.get_unchecked(i) * scale;
            i += 1;
        }
    }
}
