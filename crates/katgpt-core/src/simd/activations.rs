//! SIMD activation kernels — exp, sigmoid, tanh-clamp, reciprocal, fast_sigmoid.
//!
//! Backed by the 6th-order Cephes polynomial for `exp` (accurate to ~1 ULP for
//! `|x| < 88`). Mixed NEON/AVX2/scalar dispatch.
//!
//! AVX2 paths share horizontal reducers from `super::horizontal`.

// Cephes polynomial constants — range reduction for exp().
// Used by both the scalar tail and the AVX2/NEON polynomial kernels.

const CEPHES_LN2_HI: f32 = 6.931_457_5e-1;
const CEPHES_LN2_LO: f32 = 1.428_606_8e-6;
const CEPHES_INV_LN2: f32 = std::f32::consts::LOG2_E;

/// SIMD-accelerated in-place exp: `x[i] = exp(x[i])` for all `i`.
///
/// Uses a 6th-order Cephes polynomial approximation with range reduction,
/// accurate to ~1 ULP for inputs in [-88, 88]. Sufficient for softmax
/// where inputs are shifted by max (range [0, ~30]).
///
/// NEON: 4× f32 per iteration. AVX2: 8× f32 per iteration.
#[inline(always)]
pub fn simd_exp_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_exp_inplace(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_exp_inplace(x) }
        } else {
            scalar_exp_inplace(x)
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_exp_inplace(x) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_exp_inplace(x)
    }
}

/// Fused in-place exp + horizontal sum: `x[i] = exp(x[i])` and returns `Σ x[i]`.
///
/// Combines [`simd_exp_inplace`] + [`simd_sum_f32`] into one buffer traversal,
/// saving one full read+write pass. Used by softmax/softmax_scaled to fuse the
/// exp and denominator-computation passes — for vocab=256k this eliminates
/// ~1MB of memory traffic per token decode.
///
/// NEON: 4× f32 per iter, 4 independent accumulators for ILP.
/// AVX2: 8× f32 per iter, 4 independent accumulators for ILP.
#[inline(always)]
pub fn simd_exp_sum_inplace(x: &mut [f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_exp_sum_inplace(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_exp_sum_inplace(x) }
        } else {
            scalar_exp_sum_inplace(x)
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_exp_sum_inplace(x) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_exp_sum_inplace(x)
    }
}

/// SIMD-accelerated in-place reciprocal: `x[i] = 1.0 / x[i]`.
///
/// Used by sigmoid computation in activation functions (SiLU, SwiGLU, GeGLU)
/// to replace scalar reciprocal loops with vectorized division.
#[inline(always)]
pub fn simd_reciprocal_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_reciprocal_inplace(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_reciprocal_inplace(x) }
        } else {
            scalar_reciprocal_inplace(x)
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_reciprocal_inplace(x) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_reciprocal_inplace(x)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_reciprocal_inplace(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = 1.0 / *val;
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_exp_sum_inplace(x: &mut [f32]) -> f32 {
    let mut sum = 0.0f32;
    for val in x.iter_mut() {
        let e = cephes_exp_scalar(*val);
        *val = e;
        sum += e;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_reciprocal_inplace(x: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        let len = x.len();
        let chunks = len / 4;
        let ones = vdupq_n_f32(1.0);
        for i in 0..chunks {
            let v = vld1q_f32(x.as_ptr().add(i * 4));
            let r = vdivq_f32(ones, v);
            vst1q_f32(x.as_mut_ptr().add(i * 4), r);
        }
        for i in (chunks * 4)..len {
            *x.get_unchecked_mut(i) = 1.0 / *x.get_unchecked(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_reciprocal_inplace(x: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let len = x.len();
        let chunks = len / 8;
        let ones = _mm256_set1_ps(1.0);
        for i in 0..chunks {
            let v = _mm256_loadu_ps(x.as_ptr().add(i * 8));
            let r = _mm256_div_ps(ones, v);
            _mm256_storeu_ps(x.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..len {
            *x.get_unchecked_mut(i) = 1.0 / *x.get_unchecked(i);
        }
    }
}

/// Scalar Cephes exp approximation: accurate to ~1 ULP for |x| < 88.
/// Uses range reduction: exp(x) = exp(g) * 2^n where g = x - n*ln2, n = round(x/ln2).
/// The reduced argument g is in [-0.5*ln2, 0.5*ln2] for minimal polynomial error.
#[inline(always)]
fn cephes_exp_scalar(x: f32) -> f32 {
    // Range reduction: n = round(x / ln2)
    let n = (x * CEPHES_INV_LN2).round() as i32;

    // 2^n via bit manipulation: (n + 127) << 23.
    // Branches hoisted BEFORE the polynomial — saves ~6 FMAs in extreme cases.
    // This is the scalar tail of every SIMD exp kernel, so it runs on real inputs
    // whenever `len % 4 != 0` (NEON) or `len % 8 != 0` (AVX2).
    if n < -126 {
        return 0.0;
    }
    if n > 127 {
        return f32::INFINITY;
    }

    let g = x - n as f32 * CEPHES_LN2_HI - n as f32 * CEPHES_LN2_LO;

    // 6th-order Cephes polynomial for exp(g) in [-0.5*ln2, 0.5*ln2]
    // Q(g) = 1 + g*(1 + g/2*(1 + g/3*(1 + g/4*(1 + g/5*(1 + g/6)))))
    let q = 1.0
        + g * (1.0
            + g * 0.5
                * (1.0
                    + g * (1.0 / 3.0)
                        * (1.0 + g * 0.25 * (1.0 + g * 0.2 * (1.0 + g * (1.0 / 6.0))))));

    let bits = ((n + 127) as u32) << 23;
    let scale = f32::from_bits(bits);
    scale * q
}

/// Bounded sigmoid: σ(x) = 1/(1 + e^{-x}), output in (0, 1).
///
/// Uses `f32::exp()` via the platform's libm (hardware-accelerated on aarch64).
/// Early-exit for |x| > 40 where σ saturates to 0 or 1 in f32 precision.
///
/// **Correctness note**: the previous `0.5 + x/(2 + √(4+x²))` rational used in
/// several modules overshoots (0,1) for |x| > 2.67 — error reaches 8.3% at x=3,
/// 34.7% at x=12, and output exceeds 1.0 entirely. This implementation is exact
/// (to libm precision) and never leaves (0, 1).
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

/// Fused SIMD sigmoid → tanh-like state transform, in-place.
///
/// Computes `out[i] = (2·σ(a[i] + q[i]) − 1).clamp(-clamp, clamp)`
/// in a single vectorized pass, where σ(x) = 1/(1+e^{-x}).
///
/// This is the AttractorKernel state-writeback chain: it fuses three scalar
/// operations (sigmoid, scale-and-shift to tanh range, clamp) into one NEON/AVX2
/// traversal with no intermediate buffer.
///
/// ## Numerical contract
///
/// - σ computed via the same Cephes 6th-order polynomial used by `simd_exp_inplace`
///   (via `exp(-x)` then reciprocal). Bounded to (0, 1) for finite inputs.
/// - Output strictly in `(-clamp, clamp)`. `clamp > 0` is a debug_assert contract.
/// - `a` and `q` must have the same length; `out` must be at least that length.
/// - `a` and `out` may alias (writes happen after the read for each element),
///   but `q` must not alias `out`. Prefer separate buffers.
///
/// ## Equivalence to scalar
///
/// Matches `fast_sigmoid` to ~1 ULP for `|x| < 80`; diverges only in the libm
/// vs Cephes tail bits (max abs diff < 3e-7 in f32). The Plan 281 G1.3 σ=0
/// degeneracy test passes because `step()` and `sample_k_states` use the same
/// helper under the same feature flag (bit-identical outputs).
#[inline(always)]
pub fn simd_sigmoid_tanh_clamp_inplace(out: &mut [f32], a: &[f32], q: &[f32], clamp: f32) {
    debug_assert!(clamp > 0.0, "clamp must be positive: got {clamp}");
    debug_assert_eq!(a.len(), q.len(), "a/q length mismatch");
    debug_assert!(out.len() >= a.len(), "out too short");
    let len = a.len().min(out.len()).min(q.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sigmoid_tanh_clamp(&mut out[..len], &a[..len], &q[..len], clamp) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sigmoid_tanh_clamp(&mut out[..len], &a[..len], &q[..len], clamp) }
        } else {
            scalar_sigmoid_tanh_clamp(&mut out[..len], &a[..len], &q[..len], clamp)
        }
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            wasm32_sigmoid_tanh_clamp(&mut out[..len], &a[..len], &q[..len], clamp)
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_sigmoid_tanh_clamp(&mut out[..len], &a[..len], &q[..len], clamp)
    }
}

/// SIMD-vectorized in-place sigmoid: `x[i] = σ(x[i]) = 1/(1 + e^{-x[i]})`.
///
/// Pure sigmoid (no tanh/clamp post-processing). Backends share the same
/// Cephes 6th-order polynomial used by [`simd_sigmoid_tanh_clamp_inplace`] and
/// [`simd_exp_inplace`].
///
/// ## When to use this instead of [`fast_sigmoid`] in a loop
///
/// **Win threshold: ≥ 8 elements.** Below that, the scalar `fast_sigmoid` loop
/// wins — libm `expf` on modern hardware (Apple Silicon NEON, x86 with FMA)
/// is fast enough (~5 ns/call) that the SIMD polynomial setup overhead
/// (10+ `vdupq`/`_mm256_set1_ps` constants) exceeds the per-element savings.
/// The 6-element sense/action expand path was benchmarked and the scalar loop
/// is the GOAT there — see `sense::reconstruction::ReconstructionState::expand_with_weights`.
///
/// This helper wins when sigmoid is applied to 8+ contiguous elements, e.g.:
/// - Attractor kernel state-writeback chains (`dim` ≥ 8) — though those use
///   the fused `simd_sigmoid_tanh_clamp_inplace` variant.
/// - Batched projection of N entities × dots (when N×6 ≥ 8).
/// - Future larger HLA dimensions.
///
/// ## Numerical contract
///
/// - σ computed via the same Cephes 6th-order polynomial as the tanh-clamp
///   variant. Output strictly in `(0, 1)` for finite inputs.
/// - Matches [`fast_sigmoid`] to ~1 ULP for `|x| < 80`; diverges only in the
///   libm vs Cephes tail bits (max abs diff < 5e-6 in f32 across all input
///   ranges, typically < 1e-6).
/// - Reconstruction equivalence tests pin the cumulative divergence at <1e-4
///   across a full 3-step reconstruction cycle (see
///   `matvec_expand_matches_scalar`), well above the per-call <5e-6 floor.
#[inline]
pub fn simd_sigmoid_inplace(x: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sigmoid_inplace(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sigmoid_inplace(x) }
        } else {
            scalar_sigmoid_inplace(x)
        }
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_sigmoid_inplace(x) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_sigmoid_inplace(x)
    }
}

/// Scalar fallback for `simd_sigmoid_inplace` and the tail of the SIMD paths.
///
/// Uses `fast_sigmoid` (libm `exp`) so the scalar path is bit-exact with the
/// pre-SIMD code. This is also the scalar tail for NEON (len % 4) and AVX2
/// (len % 8) — keeping the tail bit-identical preserves determinism on
/// odd-length buffers.
#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_sigmoid_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = fast_sigmoid(*v);
    }
}

/// Scalar fallback for `simd_sigmoid_tanh_clamp_inplace`.
///
/// Uses `fast_sigmoid` (libm `exp`) so the scalar path is bit-exact with the
/// pre-SIMD code. This is also the scalar tail for NEON (len % 4) and AVX2
/// (len % 8) — keeping the tail bit-identical preserves determinism on
/// odd-length buffers.
#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_sigmoid_tanh_clamp(out: &mut [f32], a: &[f32], q: &[f32], clamp: f32) {
    for i in 0..out.len() {
        // a + q: f32 addition with +0.0 is exact, so q=0 preserves `a` bit-for-bit
        // (G1.3 degeneracy contract).
        let s = fast_sigmoid(a[i] + q[i]);
        let v = 2.0 * s - 1.0;
        out[i] = v.clamp(-clamp, clamp);
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_exp_inplace(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = cephes_exp_scalar(*val);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_exp_inplace(x: &mut [f32]) {
    use core::arch::x86_64::{
        _mm256_add_epi32, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtps_epi32, _mm256_loadu_ps,
        _mm256_max_epi32, _mm256_min_epi32, _mm256_mul_ps, _mm256_round_ps, _mm256_set1_epi32,
        _mm256_set1_ps, _mm256_slli_epi32, _mm256_storeu_ps, _mm256_sub_ps,
    };
    unsafe {
        const ROUND_NEAREST: i32 = 0x00;

        let v_inv_ln2 = _mm256_set1_ps(CEPHES_INV_LN2);
        let v_ln2_hi = _mm256_set1_ps(CEPHES_LN2_HI);
        let v_ln2_lo = _mm256_set1_ps(CEPHES_LN2_LO);
        let v_one = _mm256_set1_ps(1.0);
        let v_half = _mm256_set1_ps(0.5);
        let v_third = _mm256_set1_ps(1.0 / 3.0);
        let v_quarter = _mm256_set1_ps(0.25);
        let v_fifth = _mm256_set1_ps(0.2);
        let v_sixth = _mm256_set1_ps(1.0 / 6.0);

        let mut i = 0;
        let chunks = x.len() / 8;

        for _ in 0..chunks {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));

            // Range reduction: n = round(x * inv_ln2)
            let vn_f = _mm256_round_ps(_mm256_mul_ps(vx, v_inv_ln2), ROUND_NEAREST);
            let vn_i = _mm256_cvtps_epi32(vn_f);

            // g = x - n * ln2_hi - n * ln2_lo
            let vg = _mm256_sub_ps(
                _mm256_sub_ps(vx, _mm256_mul_ps(vn_f, v_ln2_hi)),
                _mm256_mul_ps(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial — CORRECT Horner-chain form matching
            // `cephes_exp_scalar`: Q = 1 + g·(1 + g/2·(1 + g/3·(1 + g/4·(1 + g/5·(1 + g/6))))).
            // (Issue 027: previous add-nested form g·(0.5 + g·(1/3 + ...)) gave 1/k
            // coefficients instead of 1/k!, up to 5% error on exp(2).)
            let p6 = _mm256_add_ps(v_one, _mm256_mul_ps(vg, v_sixth));
            let p5 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_fifth), p6));
            let p4 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_quarter), p5));
            let p3 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_third), p4));
            let p2 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_half), p3));
            let q = _mm256_add_ps(v_one, _mm256_mul_ps(vg, p2));

            // 2^n via AVX2 bit manipulation: shift = (n + 127) << 23
            // Clamp n to [-126, 127] before adding 127 — matches scalar `cephes_exp_scalar`
            // and NEON `neon_exp_inplace`. Without this, x > ~88 produces n+127 > 255,
            // overflowing the exponent bits and silently producing NaN/garbage.
            let vn_clamped = _mm256_max_epi32(
                _mm256_min_epi32(vn_i, _mm256_set1_epi32(127)),
                _mm256_set1_epi32(-126),
            );
            let vn_shifted_i = _mm256_add_epi32(vn_clamped, _mm256_set1_epi32(127));
            let v_scale_bits = _mm256_slli_epi32::<23>(vn_shifted_i);
            let v_scale = _mm256_castsi256_ps(v_scale_bits);

            let result = _mm256_mul_ps(v_scale, q);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), result);
            i += 8;
        }

        // Scalar tail
        while i < x.len() {
            *x.get_unchecked_mut(i) = cephes_exp_scalar(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// AVX2 backend for `simd_sigmoid_tanh_clamp_inplace`.
///
/// Computes `out[i] = (2·σ(a[i]+q[i]) − 1).clamp(-clamp, clamp)` in 8-wide
/// chunks. Mirrors `neon_sigmoid_tanh_clamp`: Cephes exp(-y) + reciprocal +
/// scale-shift + clamp. Uses `_mm256_div_ps` for the reciprocal (~1 ULP on x86).
///
/// Scalar tail uses `fast_sigmoid` to stay bit-exact with the pre-SIMD code
/// on odd-length buffers (AVX2 tail = len % 8).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_sigmoid_tanh_clamp(out: &mut [f32], a: &[f32], q: &[f32], clamp: f32) {
    use core::arch::x86_64::{
        _mm256_add_epi32, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtps_epi32, _mm256_div_ps,
        _mm256_loadu_ps, _mm256_max_epi32, _mm256_max_ps, _mm256_min_epi32, _mm256_min_ps,
        _mm256_mul_ps, _mm256_round_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_slli_epi32,
        _mm256_storeu_ps, _mm256_sub_ps, _mm256_xor_ps,
    };
    unsafe {
        const ROUND_NEAREST: i32 = 0x00;

        let v_inv_ln2 = _mm256_set1_ps(CEPHES_INV_LN2);
        let v_ln2_hi = _mm256_set1_ps(CEPHES_LN2_HI);
        let v_ln2_lo = _mm256_set1_ps(CEPHES_LN2_LO);
        let v_one = _mm256_set1_ps(1.0);
        let v_half = _mm256_set1_ps(0.5);
        let v_third = _mm256_set1_ps(1.0 / 3.0);
        let v_quarter = _mm256_set1_ps(0.25);
        let v_fifth = _mm256_set1_ps(0.2);
        let v_sixth = _mm256_set1_ps(1.0 / 6.0);
        let v_two = _mm256_set1_ps(2.0);
        let v_sign_flip = _mm256_set1_ps(f32::from_bits(0x8000_0000));
        let v_clamp = _mm256_set1_ps(clamp);
        let v_neg_clamp = _mm256_xor_ps(v_clamp, v_sign_flip);

        let mut i = 0;
        let chunks = out.len() / 8;

        for _ in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vq = _mm256_loadu_ps(q.as_ptr().add(i));
            let vy = _mm256_add_ps(va, vq);

            // σ(y) = 1/(1 + exp(-y)) → compute exp(-y) via Cephes.
            let vx = _mm256_xor_ps(vy, v_sign_flip);

            let vn_f = _mm256_round_ps(_mm256_mul_ps(vx, v_inv_ln2), ROUND_NEAREST);
            let vn_i = _mm256_cvtps_epi32(vn_f);

            let vg = _mm256_sub_ps(
                _mm256_sub_ps(vx, _mm256_mul_ps(vn_f, v_ln2_hi)),
                _mm256_mul_ps(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial for exp(g) — CORRECT Horner form matching
            // `cephes_exp_scalar`: Q = 1 + g*(1 + g/2*(1 + g/3*(1 + g/4*(1 + g/5*(1 + g/6))))).
            let gc_sixth = _mm256_mul_ps(vg, v_sixth);
            let p6 = _mm256_add_ps(v_one, gc_sixth);
            let gc_fifth = _mm256_mul_ps(vg, v_fifth);
            let p5 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_fifth, p6));
            let gc_quarter = _mm256_mul_ps(vg, v_quarter);
            let p4 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_quarter, p5));
            let gc_third = _mm256_mul_ps(vg, v_third);
            let p3 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_third, p4));
            let gc_half = _mm256_mul_ps(vg, v_half);
            let p2 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_half, p3));
            let qpoly = _mm256_add_ps(v_one, _mm256_mul_ps(vg, p2));

            let vn_clamped = _mm256_max_epi32(
                _mm256_min_epi32(vn_i, _mm256_set1_epi32(127)),
                _mm256_set1_epi32(-126),
            );
            let vn_shifted_i = _mm256_add_epi32(vn_clamped, _mm256_set1_epi32(127));
            let v_scale_bits = _mm256_slli_epi32::<23>(vn_shifted_i);
            let v_scale = _mm256_castsi256_ps(v_scale_bits);
            let exp_neg_y = _mm256_mul_ps(v_scale, qpoly);

            let denom = _mm256_add_ps(v_one, exp_neg_y);
            let sigma = _mm256_div_ps(v_one, denom);

            let tanh_like = _mm256_sub_ps(_mm256_mul_ps(v_two, sigma), v_one);
            let clamped = _mm256_max_ps(_mm256_min_ps(tanh_like, v_clamp), v_neg_clamp);

            _mm256_storeu_ps(out.as_mut_ptr().add(i), clamped);
            i += 8;
        }

        while i < out.len() {
            let s = fast_sigmoid(*a.get_unchecked(i) + *q.get_unchecked(i));
            let v = 2.0 * s - 1.0;
            *out.get_unchecked_mut(i) = v.clamp(-clamp, clamp);
            i += 1;
        }
    }
}

/// AVX2 backend for `simd_sigmoid_inplace`.
///
/// Computes `x[i] = σ(x[i]) = 1/(1 + exp(-x[i]))` in 8-wide chunks. Mirrors
/// `avx2_sigmoid_tanh_clamp`, minus the scale-shift and clamp. Uses
/// `_mm256_div_ps` for the reciprocal (~1 ULP on x86).
///
/// Scalar tail uses `fast_sigmoid` to stay bit-exact with the pre-SIMD code
/// on odd-length buffers (AVX2 tail = len % 8).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_sigmoid_inplace(x: &mut [f32]) {
    use core::arch::x86_64::{
        _mm256_add_epi32, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtps_epi32, _mm256_div_ps,
        _mm256_loadu_ps, _mm256_max_epi32, _mm256_min_epi32, _mm256_mul_ps, _mm256_round_ps,
        _mm256_set1_epi32, _mm256_set1_ps, _mm256_slli_epi32, _mm256_storeu_ps, _mm256_sub_ps,
        _mm256_xor_ps,
    };
    unsafe {
        const ROUND_NEAREST: i32 = 0x00;

        let v_inv_ln2 = _mm256_set1_ps(CEPHES_INV_LN2);
        let v_ln2_hi = _mm256_set1_ps(CEPHES_LN2_HI);
        let v_ln2_lo = _mm256_set1_ps(CEPHES_LN2_LO);
        let v_one = _mm256_set1_ps(1.0);
        let v_half = _mm256_set1_ps(0.5);
        let v_third = _mm256_set1_ps(1.0 / 3.0);
        let v_quarter = _mm256_set1_ps(0.25);
        let v_fifth = _mm256_set1_ps(0.2);
        let v_sixth = _mm256_set1_ps(1.0 / 6.0);
        let v_sign_flip = _mm256_set1_ps(f32::from_bits(0x8000_0000));

        let mut i = 0;
        let chunks = x.len() / 8;

        for _ in 0..chunks {
            let vx = _mm256_xor_ps(_mm256_loadu_ps(x.as_ptr().add(i)), v_sign_flip);

            let vn_f = _mm256_round_ps(_mm256_mul_ps(vx, v_inv_ln2), ROUND_NEAREST);
            let vn_i = _mm256_cvtps_epi32(vn_f);

            let vg = _mm256_sub_ps(
                _mm256_sub_ps(vx, _mm256_mul_ps(vn_f, v_ln2_hi)),
                _mm256_mul_ps(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial — Horner form matching `cephes_exp_scalar`.
            let gc_sixth = _mm256_mul_ps(vg, v_sixth);
            let p6 = _mm256_add_ps(v_one, gc_sixth);
            let gc_fifth = _mm256_mul_ps(vg, v_fifth);
            let p5 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_fifth, p6));
            let gc_quarter = _mm256_mul_ps(vg, v_quarter);
            let p4 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_quarter, p5));
            let gc_third = _mm256_mul_ps(vg, v_third);
            let p3 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_third, p4));
            let gc_half = _mm256_mul_ps(vg, v_half);
            let p2 = _mm256_add_ps(v_one, _mm256_mul_ps(gc_half, p3));
            let qpoly = _mm256_add_ps(v_one, _mm256_mul_ps(vg, p2));

            let vn_clamped = _mm256_max_epi32(
                _mm256_min_epi32(vn_i, _mm256_set1_epi32(127)),
                _mm256_set1_epi32(-126),
            );
            let vn_shifted_i = _mm256_add_epi32(vn_clamped, _mm256_set1_epi32(127));
            let v_scale_bits = _mm256_slli_epi32::<23>(vn_shifted_i);
            let v_scale = _mm256_castsi256_ps(v_scale_bits);
            let exp_neg_x = _mm256_mul_ps(v_scale, qpoly);

            let denom = _mm256_add_ps(v_one, exp_neg_x);
            let sigma = _mm256_div_ps(v_one, denom);

            _mm256_storeu_ps(x.as_mut_ptr().add(i), sigma);
            i += 8;
        }

        while i < x.len() {
            *x.get_unchecked_mut(i) = fast_sigmoid(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// Fused AVX2 exp + sum: `x[i] = exp(x[i])` and returns `Σ x[i]` in one pass.
///
/// 4 independent accumulators (32 elements per outer iteration) for ILP.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_exp_sum_inplace(x: &mut [f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_epi32, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtps_epi32, _mm256_loadu_ps,
        _mm256_mul_ps, _mm256_round_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_slli_epi32,
        _mm256_storeu_ps, _mm256_sub_ps,
    };
    unsafe {
        const ROUND_NEAREST: i32 = 0x00;

        let v_inv_ln2 = _mm256_set1_ps(CEPHES_INV_LN2);
        let v_ln2_hi = _mm256_set1_ps(CEPHES_LN2_HI);
        let v_ln2_lo = _mm256_set1_ps(CEPHES_LN2_LO);
        let v_one = _mm256_set1_ps(1.0);
        let v_half = _mm256_set1_ps(0.5);
        let v_third = _mm256_set1_ps(1.0 / 3.0);
        let v_quarter = _mm256_set1_ps(0.25);
        let v_fifth = _mm256_set1_ps(0.2);
        let v_sixth = _mm256_set1_ps(1.0 / 6.0);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        let len = x.len();
        let chunks4 = len / 32;

        macro_rules! step {
            ($acc:expr, $off:expr) => {{
                let vx = _mm256_loadu_ps(x.as_ptr().add(i + $off));
                let vn_f = _mm256_round_ps(_mm256_mul_ps(vx, v_inv_ln2), ROUND_NEAREST);
                let vn_i = _mm256_cvtps_epi32(vn_f);
                let vg = _mm256_sub_ps(
                    _mm256_sub_ps(vx, _mm256_mul_ps(vn_f, v_ln2_hi)),
                    _mm256_mul_ps(vn_f, v_ln2_lo),
                );
                // Cephes 6th-order polynomial — CORRECT Horner-chain form (Issue 027).
                let p6 = _mm256_add_ps(v_one, _mm256_mul_ps(vg, v_sixth));
                let p5 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_fifth), p6));
                let p4 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_quarter), p5));
                let p3 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_third), p4));
                let p2 = _mm256_add_ps(v_one, _mm256_mul_ps(_mm256_mul_ps(vg, v_half), p3));
                let q = _mm256_add_ps(v_one, _mm256_mul_ps(vg, p2));
                let vn_shifted_i = _mm256_add_epi32(vn_i, _mm256_set1_epi32(127));
                let v_scale_bits = _mm256_slli_epi32::<23>(vn_shifted_i);
                let v_scale = _mm256_castsi256_ps(v_scale_bits);
                let r = _mm256_mul_ps(v_scale, q);
                _mm256_storeu_ps(x.as_mut_ptr().add(i + $off), r);
                $acc = _mm256_add_ps($acc, r);
            }};
        }

        // Main loop: 32 elements per iteration (4 accumulators × 8 lanes)
        for _ in 0..chunks4 {
            step!(acc0, 0);
            step!(acc1, 8);
            step!(acc2, 16);
            step!(acc3, 24);
            i += 32;
        }

        let mut sum = horizontal_sum_256(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ));

        // Remaining 8-element chunks
        let mut acc_rem = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            step!(acc_rem, 0);
            i += 8;
        }
        sum += horizontal_sum_256(acc_rem);

        // Scalar tail (0-7 elements)
        while i < len {
            let e = cephes_exp_scalar(*x.get_unchecked(i));
            *x.get_unchecked_mut(i) = e;
            sum += e;
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_exp_inplace(x: &mut [f32]) {
    use core::arch::aarch64::{
        vaddq_f32, vaddq_s32, vcvtq_s32_f32, vdupq_n_f32, vdupq_n_s32, vld1q_f32, vmaxq_s32,
        vminq_s32, vmulq_f32, vreinterpretq_f32_s32, vrndq_f32, vshlq_n_s32, vst1q_f32, vsubq_f32,
    };
    unsafe {
        let v_inv_ln2 = vdupq_n_f32(CEPHES_INV_LN2);
        let v_ln2_hi = vdupq_n_f32(CEPHES_LN2_HI);
        let v_ln2_lo = vdupq_n_f32(CEPHES_LN2_LO);
        let v_one = vdupq_n_f32(1.0);
        let v_half = vdupq_n_f32(0.5);
        let v_third = vdupq_n_f32(1.0 / 3.0);
        let v_quarter = vdupq_n_f32(0.25);
        let v_fifth = vdupq_n_f32(0.2);
        let v_sixth = vdupq_n_f32(1.0 / 6.0);

        let mut i = 0;
        let chunks = x.len() / 4;

        for _ in 0..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i));

            // Range reduction: n = round(x * inv_ln2)
            let vn_f = vrndq_f32(vmulq_f32(vx, v_inv_ln2));
            let vn_i = vcvtq_s32_f32(vn_f);

            // g = x - n * ln2_hi - n * ln2_lo
            let vg = vsubq_f32(
                vsubq_f32(vx, vmulq_f32(vn_f, v_ln2_hi)),
                vmulq_f32(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial — CORRECT Horner-chain form matching
            // `cephes_exp_scalar`: Q = 1 + g·(1 + g/2·(1 + g/3·(1 + g/4·(1 + g/5·(1 + g/6))))).
            // (Issue 027: the previous add-nested form g·(0.5 + g·(1/3 + ...)) produced
            // coefficients 1/k instead of 1/k!, giving up to 5% error on exp(2). This
            // form matches the scalar fallback bit-for-bit and is algebraically exact.)
            let p6 = vaddq_f32(v_one, vmulq_f32(vg, v_sixth));        // 1 + g/6
            let p5 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_fifth), p6));   // 1 + g/5·p6
            let p4 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_quarter), p5)); // 1 + g/4·p5
            let p3 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_third), p4));   // 1 + g/3·p4
            let p2 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_half), p3));    // 1 + g/2·p3
            let q = vaddq_f32(v_one, vmulq_f32(vg, p2));              // 1 + g·p2

            // 2^n via branchless NEON bit manipulation
            // Clamp n to [-126, 127] to avoid IEEE overflow/underflow
            let v127 = vdupq_n_s32(127);
            let vneg126 = vdupq_n_s32(-126);
            let vn_clamped = vmaxq_s32(vminq_s32(vn_i, v127), vneg126);
            // Build 2^n: (n + 127) << 23 reinterpreted as f32
            let v_bias = vdupq_n_s32(127);
            let v_shifted = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vn_clamped, v_bias)));
            let vresult = vmulq_f32(v_shifted, q);

            vst1q_f32(x.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Scalar tail
        while i < x.len() {
            *x.get_unchecked_mut(i) = cephes_exp_scalar(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// NEON backend for `simd_sigmoid_tanh_clamp_inplace`.
///
/// Computes `out[i] = (2·σ(a[i]+q[i]) − 1).clamp(-clamp, clamp)` in 4-wide
/// chunks. σ(x) = 1/(1+exp(-x)) via the same Cephes polynomial as
/// `neon_exp_inplace`. The |x| > 40 early-exit that `fast_sigmoid` does is
/// folded into the n-clamp: n clamped to [-126, 127] drives exp(-y) to
/// 0 (y large positive → σ → 1) or +inf (y large negative → 1/(1+inf) → 0),
/// so σ saturates correctly without a branch.
///
/// Uses `vdivq_f32` for the reciprocal (same as `neon_reciprocal_inplace`) —
/// on Apple Silicon M-series, `fdiv` throughput is high enough that this matches
/// `vrecpeq+vrecpsq` while giving full ~1 ULP precision.
///
/// Scalar tail uses `fast_sigmoid` to stay bit-exact with the pre-SIMD code
/// on odd-length buffers (NEON tail = len % 4).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sigmoid_tanh_clamp(out: &mut [f32], a: &[f32], q: &[f32], clamp: f32) {
    use core::arch::aarch64::{
        vaddq_f32, vaddq_s32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32, vld1q_f32,
        vmaxq_f32, vmaxq_s32, vminq_f32, vminq_s32, vmulq_f32, vnegq_f32, vreinterpretq_f32_s32,
        vrndq_f32, vshlq_n_s32, vst1q_f32, vsubq_f32,
    };
    unsafe {
        let v_inv_ln2 = vdupq_n_f32(CEPHES_INV_LN2);
        let v_ln2_hi = vdupq_n_f32(CEPHES_LN2_HI);
        let v_ln2_lo = vdupq_n_f32(CEPHES_LN2_LO);
        let v_one = vdupq_n_f32(1.0);
        let v_half = vdupq_n_f32(0.5);
        let v_third = vdupq_n_f32(1.0 / 3.0);
        let v_quarter = vdupq_n_f32(0.25);
        let v_fifth = vdupq_n_f32(0.2);
        let v_sixth = vdupq_n_f32(1.0 / 6.0);
        let v_two = vdupq_n_f32(2.0);
        let v_clamp = vdupq_n_f32(clamp);
        let v_neg_clamp = vnegq_f32(v_clamp);

        let mut i = 0;
        let chunks = out.len() / 4;

        for _ in 0..chunks {
            // y = a[i] + q[i]  (the pre-sigmoid activation).
            let va = vld1q_f32(a.as_ptr().add(i));
            let vq = vld1q_f32(q.as_ptr().add(i));
            let vy = vaddq_f32(va, vq);

            // σ(y) = 1/(1 + exp(-y)). Compute exp(-y) via the Cephes polynomial:
            // negate y first, then the standard range-reduction + polynomial +
            // 2^n path identical to neon_exp_inplace.
            let vx = vnegq_f32(vy);

            let vn_f = vrndq_f32(vmulq_f32(vx, v_inv_ln2));
            let vn_i = vcvtq_s32_f32(vn_f);

            let vg = vsubq_f32(
                vsubq_f32(vx, vmulq_f32(vn_f, v_ln2_hi)),
                vmulq_f32(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial for exp(g) — CORRECT Horner form matching
            // `cephes_exp_scalar`: Q = 1 + g*(1 + g/2*(1 + g/3*(1 + g/4*(1 + g/5*(1 + g/6))))).
            // (This is NOT the same nesting as `neon_exp_inplace` — that helper uses
            // an add-nested form which overestimates for |g| > 0.1. The sigmoid
            // path sees wider g ranges, so we use the mathematically exact form.)
            let gc_sixth = vmulq_f32(vg, v_sixth);
            let p6 = vaddq_f32(v_one, gc_sixth);              // 1 + g/6
            let gc_fifth = vmulq_f32(vg, v_fifth);
            let p5 = vaddq_f32(v_one, vmulq_f32(gc_fifth, p6));   // 1 + g/5*p6
            let gc_quarter = vmulq_f32(vg, v_quarter);
            let p4 = vaddq_f32(v_one, vmulq_f32(gc_quarter, p5)); // 1 + g/4*p5
            let gc_third = vmulq_f32(vg, v_third);
            let p3 = vaddq_f32(v_one, vmulq_f32(gc_third, p4));   // 1 + g/3*p4
            let gc_half = vmulq_f32(vg, v_half);
            let p2 = vaddq_f32(v_one, vmulq_f32(gc_half, p3));    // 1 + g/2*p3
            let qpoly = vaddq_f32(v_one, vmulq_f32(vg, p2));      // 1 + g*p2

            // 2^n via branchless NEON bit manipulation. Clamp n to [-126, 127]
            // — also folds the |x| > 40 early-exit: exp(-y) for large positive
            // y underflows to 0 (σ → 1), for large negative y overflows to inf
            // (σ → 0). Both are the correct sigmoid saturation.
            let v127 = vdupq_n_s32(127);
            let vneg126 = vdupq_n_s32(-126);
            let vn_clamped = vmaxq_s32(vminq_s32(vn_i, v127), vneg126);
            let v_bias = vdupq_n_s32(127);
            let v_shifted =
                vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vn_clamped, v_bias)));
            let exp_neg_y = vmulq_f32(v_shifted, qpoly);

            // σ(y) = 1 / (1 + exp(-y)). vdivq gives ~1 ULP.
            let denom = vaddq_f32(v_one, exp_neg_y);
            let sigma = vdivq_f32(v_one, denom);

            // 2·σ − 1.
            let tanh_like = vsubq_f32(vmulq_f32(v_two, sigma), v_one);

            // clamp(-clamp, clamp) via min/max.
            let clamped = vmaxq_f32(vminq_f32(tanh_like, v_clamp), v_neg_clamp);

            vst1q_f32(out.as_mut_ptr().add(i), clamped);
            i += 4;
        }

        // Scalar tail — bit-exact with the pre-SIMD code via fast_sigmoid.
        while i < out.len() {
            let s = fast_sigmoid(*a.get_unchecked(i) + *q.get_unchecked(i));
            let v = 2.0 * s - 1.0;
            *out.get_unchecked_mut(i) = v.clamp(-clamp, clamp);
            i += 1;
        }
    }
}

/// NEON backend for `simd_sigmoid_inplace`.
///
/// Computes `x[i] = σ(x[i]) = 1/(1 + exp(-x[i]))` in 4-wide chunks. Same
/// Cephes 6th-order polynomial as `neon_sigmoid_tanh_clamp`, minus the
/// scale-shift (`2σ−1`) and clamp.
///
/// Scalar tail uses `fast_sigmoid` to stay bit-exact with the pre-SIMD code
/// on odd-length buffers (NEON tail = len % 4).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sigmoid_inplace(x: &mut [f32]) {
    use core::arch::aarch64::{
        vaddq_f32, vaddq_s32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32, vld1q_f32,
        vmaxq_s32, vminq_s32, vmulq_f32, vnegq_f32, vreinterpretq_f32_s32, vrndq_f32,
        vshlq_n_s32, vst1q_f32, vsubq_f32,
    };
    unsafe {
        let v_inv_ln2 = vdupq_n_f32(CEPHES_INV_LN2);
        let v_ln2_hi = vdupq_n_f32(CEPHES_LN2_HI);
        let v_ln2_lo = vdupq_n_f32(CEPHES_LN2_LO);
        let v_one = vdupq_n_f32(1.0);
        let v_half = vdupq_n_f32(0.5);
        let v_third = vdupq_n_f32(1.0 / 3.0);
        let v_quarter = vdupq_n_f32(0.25);
        let v_fifth = vdupq_n_f32(0.2);
        let v_sixth = vdupq_n_f32(1.0 / 6.0);
        let v127 = vdupq_n_s32(127);
        let vneg126 = vdupq_n_s32(-126);
        let v_bias = vdupq_n_s32(127);

        let mut i = 0;
        let chunks = x.len() / 4;

        for _ in 0..chunks {
            // σ(x) = 1/(1 + exp(-x)). Compute exp(-x) via the Cephes polynomial.
            let vx = vnegq_f32(vld1q_f32(x.as_ptr().add(i)));

            let vn_f = vrndq_f32(vmulq_f32(vx, v_inv_ln2));
            let vn_i = vcvtq_s32_f32(vn_f);

            let vg = vsubq_f32(
                vsubq_f32(vx, vmulq_f32(vn_f, v_ln2_hi)),
                vmulq_f32(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial — Horner form matching `cephes_exp_scalar`.
            let gc_sixth = vmulq_f32(vg, v_sixth);
            let p6 = vaddq_f32(v_one, gc_sixth);
            let gc_fifth = vmulq_f32(vg, v_fifth);
            let p5 = vaddq_f32(v_one, vmulq_f32(gc_fifth, p6));
            let gc_quarter = vmulq_f32(vg, v_quarter);
            let p4 = vaddq_f32(v_one, vmulq_f32(gc_quarter, p5));
            let gc_third = vmulq_f32(vg, v_third);
            let p3 = vaddq_f32(v_one, vmulq_f32(gc_third, p4));
            let gc_half = vmulq_f32(vg, v_half);
            let p2 = vaddq_f32(v_one, vmulq_f32(gc_half, p3));
            let qpoly = vaddq_f32(v_one, vmulq_f32(vg, p2));

            // 2^n via branchless NEON bit manipulation. Clamp n to [-126, 127]
            // — folds the |x| > 40 early-exit: large positive x → exp(-x) → 0
            // (σ → 1), large negative x → exp(-x) → inf (σ → 0). Both correct.
            let vn_clamped = vmaxq_s32(vminq_s32(vn_i, v127), vneg126);
            let v_shifted =
                vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vn_clamped, v_bias)));
            let exp_neg_x = vmulq_f32(v_shifted, qpoly);

            // σ = 1 / (1 + exp(-x)). vdivq gives ~1 ULP.
            let denom = vaddq_f32(v_one, exp_neg_x);
            let sigma = vdivq_f32(v_one, denom);

            vst1q_f32(x.as_mut_ptr().add(i), sigma);
            i += 4;
        }

        // Scalar tail — bit-exact with the pre-SIMD code via fast_sigmoid.
        while i < x.len() {
            *x.get_unchecked_mut(i) = fast_sigmoid(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// Fused NEON exp + sum: `x[i] = exp(x[i])` and returns `Σ x[i]` in one pass.
///
/// 4 independent accumulators for ILP — hides the FADD latency.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_exp_sum_inplace(x: &mut [f32]) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddq_s32, vaddvq_f32, vcvtq_s32_f32, vdupq_n_f32, vdupq_n_s32, vld1q_f32,
        vmaxq_s32, vminq_s32, vmulq_f32, vreinterpretq_f32_s32, vrndq_f32, vshlq_n_s32, vst1q_f32,
        vsubq_f32,
    };
    unsafe {
        let v_inv_ln2 = vdupq_n_f32(CEPHES_INV_LN2);
        let v_ln2_hi = vdupq_n_f32(CEPHES_LN2_HI);
        let v_ln2_lo = vdupq_n_f32(CEPHES_LN2_LO);
        let v_one = vdupq_n_f32(1.0);
        let v_half = vdupq_n_f32(0.5);
        let v_third = vdupq_n_f32(1.0 / 3.0);
        let v_quarter = vdupq_n_f32(0.25);
        let v_fifth = vdupq_n_f32(0.2);
        let v_sixth = vdupq_n_f32(1.0 / 6.0);
        let v127 = vdupq_n_s32(127);
        let vneg126 = vdupq_n_s32(-126);
        let v_bias = vdupq_n_s32(127);

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let len = x.len();
        let chunks4 = len / 16;

        // Main loop: 16 elements per iteration (4 accumulators × 4 lanes)
        for _ in 0..chunks4 {
            macro_rules! step {
                ($acc:expr, $off:expr) => {{
                    let vx = vld1q_f32(x.as_ptr().add(i + $off));
                    let vn_f = vrndq_f32(vmulq_f32(vx, v_inv_ln2));
                    let vn_i = vcvtq_s32_f32(vn_f);
                    let vg = vsubq_f32(
                        vsubq_f32(vx, vmulq_f32(vn_f, v_ln2_hi)),
                        vmulq_f32(vn_f, v_ln2_lo),
                    );
                    // Cephes 6th-order polynomial — CORRECT Horner-chain form (Issue 027).
                    let p6 = vaddq_f32(v_one, vmulq_f32(vg, v_sixth));
                    let p5 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_fifth), p6));
                    let p4 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_quarter), p5));
                    let p3 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_third), p4));
                    let p2 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_half), p3));
                    let q = vaddq_f32(v_one, vmulq_f32(vg, p2));
                    let vn_clamped = vmaxq_s32(vminq_s32(vn_i, v127), vneg126);
                    let v_shifted =
                        vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vn_clamped, v_bias)));
                    let r = vmulq_f32(v_shifted, q);
                    vst1q_f32(x.as_mut_ptr().add(i + $off), r);
                    $acc = vaddq_f32($acc, r);
                }};
            }
            step!(acc0, 0);
            step!(acc1, 4);
            step!(acc2, 8);
            step!(acc3, 12);
            i += 16;
        }

        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        // Remaining 4-element chunks into single accumulator
        let mut acc_rem = vdupq_n_f32(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let vn_f = vrndq_f32(vmulq_f32(vx, v_inv_ln2));
            let vn_i = vcvtq_s32_f32(vn_f);
            let vg = vsubq_f32(
                vsubq_f32(vx, vmulq_f32(vn_f, v_ln2_hi)),
                vmulq_f32(vn_f, v_ln2_lo),
            );
            // Cephes 6th-order polynomial — CORRECT Horner-chain form (Issue 027).
            let p6 = vaddq_f32(v_one, vmulq_f32(vg, v_sixth));
            let p5 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_fifth), p6));
            let p4 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_quarter), p5));
            let p3 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_third), p4));
            let p2 = vaddq_f32(v_one, vmulq_f32(vmulq_f32(vg, v_half), p3));
            let q = vaddq_f32(v_one, vmulq_f32(vg, p2));
            let vn_clamped = vmaxq_s32(vminq_s32(vn_i, v127), vneg126);
            let v_shifted = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vn_clamped, v_bias)));
            let r = vmulq_f32(v_shifted, q);
            vst1q_f32(x.as_mut_ptr().add(i), r);
            acc_rem = vaddq_f32(acc_rem, r);
            i += 4;
        }
        sum += vaddvq_f32(acc_rem);

        // Scalar tail (0-3 elements)
        while i < len {
            let e = cephes_exp_scalar(*x.get_unchecked(i));
            *x.get_unchecked_mut(i) = e;
            sum += e;
            i += 1;
        }

        sum
    }
}

// ── WASM SIMD128 backends (Issue 007) ──────────────────────────
//
// 4-wide f32, same algorithmic structure as the NEON kernels (also 4-wide).
// Gate: `#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]`.
//
// Mapping notes (NEON → WASM SIMD128):
//   vld1q_f32       → v128_load (with .cast())
//   vst1q_f32       → v128_store (with .cast())
//   vrndq_f32       → f32x4_nearest          (round to nearest-even, ties to even)
//   vcvtq_s32_f32   → i32x4_trunc_sat_f32x4  (saturating; identical for in-range n)
//   vmulq/vaddq/vsubq_f32 → f32x4_mul/add/sub
//   vdivq_f32       → f32x4_div
//   vminq/vmaxq_f32 → f32x4_min/max
//   vnegq_f32       → f32x4_neg
//   vminq/vmaxq_s32 → i32x4_min/max_s
//   vaddq_s32       → i32x4_add
//   vshlq_n_s32::<23> → i32x4_shl(_, 23)
//   vdupq_n_f32/s32 → f32x4_splat / i32x4_splat
//   vreinterpretq_f32_s32 → no-op. WASM SIMD128 has a single `v128` register
//     type; an i32x4 result is reinterpreted as f32 implicitly by passing it
//     straight into an f32x4 op. (This is the intended design.)
//
// No FMA intrinsic in the WASM SIMD128 base proposal — the Cephes polynomial
// uses separate `f32x4_mul` + `f32x4_add`. NEON's `cephes_exp_scalar` reference
// also uses separate mul+add (no FMA), so the WASM path matches the scalar
// reference modulo the rounding of the intermediate `n*ln2` products, which can
// differ by ≤ 1 ULP on some inputs (acceptable, documented here per the
// FMA-contraction rule for this issue).

/// WASM SIMD128 reciprocal: `x[i] = 1.0 / x[i]`. Mirrors `neon_reciprocal_inplace`
/// (4-wide `f32x4_div`). Uses plain division (~1 ULP), NOT a Newton-Raphson
/// reciprocal estimate — matches the NEON/AVX2 kernels which use `vdivq_f32` /
/// `_mm256_div_ps`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_reciprocal_inplace(x: &mut [f32]) {
    use core::arch::wasm32::{f32x4_div, f32x4_splat, v128_load, v128_store};

    unsafe {
        let len = x.len();
        let chunks = len / 4;
        let ones = f32x4_splat(1.0);
        for i in 0..chunks {
            let v = v128_load(x.as_ptr().add(i * 4).cast());
            let r = f32x4_div(ones, v);
            v128_store(x.as_mut_ptr().add(i * 4).cast(), r);
        }
        for i in (chunks * 4)..len {
            *x.get_unchecked_mut(i) = 1.0 / *x.get_unchecked(i);
        }
    }
}

/// WASM SIMD128 in-place exp via the 6th-order Cephes polynomial. Mirrors
/// `neon_exp_inplace` (4-wide). See the backend mapping notes at the top of
/// this section for the NEON→WASM intrinsic translation.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_exp_inplace(x: &mut [f32]) {
    use core::arch::wasm32::{
        f32x4_add, f32x4_mul, f32x4_nearest, f32x4_splat, f32x4_sub, i32x4_add, i32x4_max,
        i32x4_min, i32x4_shl, i32x4_splat, i32x4_trunc_sat_f32x4, v128_load, v128_store,
    };

    unsafe {
        let v_inv_ln2 = f32x4_splat(CEPHES_INV_LN2);
        let v_ln2_hi = f32x4_splat(CEPHES_LN2_HI);
        let v_ln2_lo = f32x4_splat(CEPHES_LN2_LO);
        let v_one = f32x4_splat(1.0);
        let v_half = f32x4_splat(0.5);
        let v_third = f32x4_splat(1.0 / 3.0);
        let v_quarter = f32x4_splat(0.25);
        let v_fifth = f32x4_splat(0.2);
        let v_sixth = f32x4_splat(1.0 / 6.0);
        let v127 = i32x4_splat(127);
        let vneg126 = i32x4_splat(-126);
        let v_bias = i32x4_splat(127);

        let mut i = 0;
        let chunks = x.len() / 4;

        for _ in 0..chunks {
            let vx = v128_load(x.as_ptr().add(i).cast());

            // Range reduction: n = round(x * inv_ln2)
            let vn_f = f32x4_nearest(f32x4_mul(vx, v_inv_ln2));
            let vn_i = i32x4_trunc_sat_f32x4(vn_f);

            // g = x - n * ln2_hi - n * ln2_lo
            let vg = f32x4_sub(
                f32x4_sub(vx, f32x4_mul(vn_f, v_ln2_hi)),
                f32x4_mul(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial — Horner-chain form matching
            // `cephes_exp_scalar`: Q = 1 + g·(1 + g/2·(1 + g/3·(1 + g/4·(1 + g/5·(1 + g/6))))).
            let p6 = f32x4_add(v_one, f32x4_mul(vg, v_sixth)); // 1 + g/6
            let p5 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_fifth), p6)); // 1 + g/5·p6
            let p4 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_quarter), p5)); // 1 + g/4·p5
            let p3 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_third), p4)); // 1 + g/3·p4
            let p2 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_half), p3)); // 1 + g/2·p3
            let q = f32x4_add(v_one, f32x4_mul(vg, p2)); // 1 + g·p2

            // 2^n via branchless bit manipulation: clamp n to [-126, 127], then
            // (n + 127) << 23 reinterpreted as f32. The i32x4 result is passed
            // straight into f32x4_mul — WASM reinterprets the bits implicitly.
            let vn_clamped = i32x4_max(i32x4_min(vn_i, v127), vneg126);
            let v_shifted = i32x4_shl(i32x4_add(vn_clamped, v_bias), 23);
            let vresult = f32x4_mul(v_shifted, q);

            v128_store(x.as_mut_ptr().add(i).cast(), vresult);
            i += 4;
        }

        // Scalar tail
        while i < x.len() {
            *x.get_unchecked_mut(i) = cephes_exp_scalar(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// WASM SIMD128 fused exp + horizontal sum. Mirrors `neon_exp_sum_inplace`
/// (4 accumulators × 4 lanes = 16 elements per outer iter for ILP). Horizontal
/// reduce via 4× `f32x4_extract_lane` (WASM has no `vaddvq_f32` equivalent).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_exp_sum_inplace(x: &mut [f32]) -> f32 {
    use core::arch::wasm32::{
        f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_nearest, f32x4_splat, f32x4_sub, i32x4_add,
        i32x4_max, i32x4_min, i32x4_shl, i32x4_splat, i32x4_trunc_sat_f32x4, v128_load,
        v128_store,
    };

    unsafe {
        let v_inv_ln2 = f32x4_splat(CEPHES_INV_LN2);
        let v_ln2_hi = f32x4_splat(CEPHES_LN2_HI);
        let v_ln2_lo = f32x4_splat(CEPHES_LN2_LO);
        let v_one = f32x4_splat(1.0);
        let v_half = f32x4_splat(0.5);
        let v_third = f32x4_splat(1.0 / 3.0);
        let v_quarter = f32x4_splat(0.25);
        let v_fifth = f32x4_splat(0.2);
        let v_sixth = f32x4_splat(1.0 / 6.0);
        let v127 = i32x4_splat(127);
        let vneg126 = i32x4_splat(-126);
        let v_bias = i32x4_splat(127);

        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut acc2 = f32x4_splat(0.0);
        let mut acc3 = f32x4_splat(0.0);
        let mut i = 0;
        let len = x.len();
        let chunks4 = len / 16;

        // Main loop: 16 elements per iteration (4 accumulators × 4 lanes).
        for _ in 0..chunks4 {
            macro_rules! step {
                ($acc:expr, $off:expr) => {{
                    let vx = v128_load(x.as_ptr().add(i + $off).cast());
                    let vn_f = f32x4_nearest(f32x4_mul(vx, v_inv_ln2));
                    let vn_i = i32x4_trunc_sat_f32x4(vn_f);
                    let vg = f32x4_sub(
                        f32x4_sub(vx, f32x4_mul(vn_f, v_ln2_hi)),
                        f32x4_mul(vn_f, v_ln2_lo),
                    );
                    // Cephes 6th-order polynomial — Horner-chain form (Issue 027).
                    let p6 = f32x4_add(v_one, f32x4_mul(vg, v_sixth));
                    let p5 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_fifth), p6));
                    let p4 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_quarter), p5));
                    let p3 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_third), p4));
                    let p2 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_half), p3));
                    let q = f32x4_add(v_one, f32x4_mul(vg, p2));
                    let vn_clamped = i32x4_max(i32x4_min(vn_i, v127), vneg126);
                    let v_shifted = i32x4_shl(i32x4_add(vn_clamped, v_bias), 23);
                    let r = f32x4_mul(v_shifted, q);
                    v128_store(x.as_mut_ptr().add(i + $off).cast(), r);
                    $acc = f32x4_add($acc, r);
                }};
            }
            step!(acc0, 0);
            step!(acc1, 4);
            step!(acc2, 8);
            step!(acc3, 12);
            i += 16;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3 → 4 lanes → scalar.
        let s01 = f32x4_add(acc0, acc1);
        let s23 = f32x4_add(acc2, acc3);
        let s = f32x4_add(s01, s23);
        let mut sum = f32x4_extract_lane::<0>(s)
            + f32x4_extract_lane::<1>(s)
            + f32x4_extract_lane::<2>(s)
            + f32x4_extract_lane::<3>(s);

        // Remaining 4-element chunks into a single accumulator.
        let mut acc_rem = f32x4_splat(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            let vx = v128_load(x.as_ptr().add(i).cast());
            let vn_f = f32x4_nearest(f32x4_mul(vx, v_inv_ln2));
            let vn_i = i32x4_trunc_sat_f32x4(vn_f);
            let vg = f32x4_sub(
                f32x4_sub(vx, f32x4_mul(vn_f, v_ln2_hi)),
                f32x4_mul(vn_f, v_ln2_lo),
            );
            // Cephes 6th-order polynomial — Horner-chain form (Issue 027).
            let p6 = f32x4_add(v_one, f32x4_mul(vg, v_sixth));
            let p5 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_fifth), p6));
            let p4 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_quarter), p5));
            let p3 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_third), p4));
            let p2 = f32x4_add(v_one, f32x4_mul(f32x4_mul(vg, v_half), p3));
            let q = f32x4_add(v_one, f32x4_mul(vg, p2));
            let vn_clamped = i32x4_max(i32x4_min(vn_i, v127), vneg126);
            let v_shifted = i32x4_shl(i32x4_add(vn_clamped, v_bias), 23);
            let r = f32x4_mul(v_shifted, q);
            v128_store(x.as_mut_ptr().add(i).cast(), r);
            acc_rem = f32x4_add(acc_rem, r);
            i += 4;
        }
        sum += f32x4_extract_lane::<0>(acc_rem)
            + f32x4_extract_lane::<1>(acc_rem)
            + f32x4_extract_lane::<2>(acc_rem)
            + f32x4_extract_lane::<3>(acc_rem);

        // Scalar tail (0-3 elements)
        while i < len {
            let e = cephes_exp_scalar(*x.get_unchecked(i));
            *x.get_unchecked_mut(i) = e;
            sum += e;
            i += 1;
        }

        sum
    }
}

/// WASM SIMD128 in-place sigmoid: `x[i] = 1/(1 + e^{-x[i]})`. Mirrors
/// `neon_sigmoid_inplace` (4-wide). exp(-x) via the same Cephes polynomial, then
/// `f32x4_div` for the reciprocal. Scalar tail uses `fast_sigmoid` to stay
/// bit-exact with the pre-SIMD code on odd-length buffers (tail = len % 4).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_sigmoid_inplace(x: &mut [f32]) {
    use core::arch::wasm32::{
        f32x4_add, f32x4_div, f32x4_mul, f32x4_nearest, f32x4_neg, f32x4_splat, f32x4_sub,
        i32x4_add, i32x4_max, i32x4_min, i32x4_shl, i32x4_splat, i32x4_trunc_sat_f32x4,
        v128_load, v128_store,
    };

    unsafe {
        let v_inv_ln2 = f32x4_splat(CEPHES_INV_LN2);
        let v_ln2_hi = f32x4_splat(CEPHES_LN2_HI);
        let v_ln2_lo = f32x4_splat(CEPHES_LN2_LO);
        let v_one = f32x4_splat(1.0);
        let v_half = f32x4_splat(0.5);
        let v_third = f32x4_splat(1.0 / 3.0);
        let v_quarter = f32x4_splat(0.25);
        let v_fifth = f32x4_splat(0.2);
        let v_sixth = f32x4_splat(1.0 / 6.0);
        let v127 = i32x4_splat(127);
        let vneg126 = i32x4_splat(-126);
        let v_bias = i32x4_splat(127);

        let mut i = 0;
        let chunks = x.len() / 4;

        for _ in 0..chunks {
            // σ(x) = 1/(1 + exp(-x)). Compute exp(-x) via the Cephes polynomial.
            let vx = f32x4_neg(v128_load(x.as_ptr().add(i).cast()));

            let vn_f = f32x4_nearest(f32x4_mul(vx, v_inv_ln2));
            let vn_i = i32x4_trunc_sat_f32x4(vn_f);

            let vg = f32x4_sub(
                f32x4_sub(vx, f32x4_mul(vn_f, v_ln2_hi)),
                f32x4_mul(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial — Horner form matching `cephes_exp_scalar`.
            let gc_sixth = f32x4_mul(vg, v_sixth);
            let p6 = f32x4_add(v_one, gc_sixth);
            let gc_fifth = f32x4_mul(vg, v_fifth);
            let p5 = f32x4_add(v_one, f32x4_mul(gc_fifth, p6));
            let gc_quarter = f32x4_mul(vg, v_quarter);
            let p4 = f32x4_add(v_one, f32x4_mul(gc_quarter, p5));
            let gc_third = f32x4_mul(vg, v_third);
            let p3 = f32x4_add(v_one, f32x4_mul(gc_third, p4));
            let gc_half = f32x4_mul(vg, v_half);
            let p2 = f32x4_add(v_one, f32x4_mul(gc_half, p3));
            let qpoly = f32x4_add(v_one, f32x4_mul(vg, p2));

            // 2^n via branchless bit manipulation. Clamp n to [-126, 127] — folds
            // the |x| > 40 early-exit: large positive x → exp(-x) → 0 (σ → 1),
            // large negative x → exp(-x) → inf (σ → 0). Both correct saturation.
            let vn_clamped = i32x4_max(i32x4_min(vn_i, v127), vneg126);
            let v_shifted = i32x4_shl(i32x4_add(vn_clamped, v_bias), 23);
            let exp_neg_x = f32x4_mul(v_shifted, qpoly);

            // σ = 1 / (1 + exp(-x)). f32x4_div gives ~1 ULP.
            let denom = f32x4_add(v_one, exp_neg_x);
            let sigma = f32x4_div(v_one, denom);

            v128_store(x.as_mut_ptr().add(i).cast(), sigma);
            i += 4;
        }

        // Scalar tail — bit-exact with the pre-SIMD code via fast_sigmoid.
        while i < x.len() {
            *x.get_unchecked_mut(i) = fast_sigmoid(*x.get_unchecked(i));
            i += 1;
        }
    }
}

/// WASM SIMD128 fused sigmoid → tanh-like clamp. Mirrors `neon_sigmoid_tanh_clamp`
/// (4-wide). Computes `out[i] = (2·σ(a[i]+q[i]) − 1).clamp(-clamp, clamp)` via
/// the same Cephes polynomial as `wasm32_exp_inplace`, then `f32x4_div` for the
/// reciprocal and `f32x4_min`/`f32x4_max` for the clamp. Scalar tail uses
/// `fast_sigmoid` to stay bit-exact on odd-length buffers (tail = len % 4).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_sigmoid_tanh_clamp(out: &mut [f32], a: &[f32], q: &[f32], clamp: f32) {
    use core::arch::wasm32::{
        f32x4_add, f32x4_div, f32x4_max, f32x4_min, f32x4_mul, f32x4_nearest, f32x4_neg,
        f32x4_splat, f32x4_sub, i32x4_add, i32x4_max, i32x4_min, i32x4_shl, i32x4_splat,
        i32x4_trunc_sat_f32x4, v128_load, v128_store,
    };

    unsafe {
        let v_inv_ln2 = f32x4_splat(CEPHES_INV_LN2);
        let v_ln2_hi = f32x4_splat(CEPHES_LN2_HI);
        let v_ln2_lo = f32x4_splat(CEPHES_LN2_LO);
        let v_one = f32x4_splat(1.0);
        let v_half = f32x4_splat(0.5);
        let v_third = f32x4_splat(1.0 / 3.0);
        let v_quarter = f32x4_splat(0.25);
        let v_fifth = f32x4_splat(0.2);
        let v_sixth = f32x4_splat(1.0 / 6.0);
        let v_two = f32x4_splat(2.0);
        let v_clamp = f32x4_splat(clamp);
        let v_neg_clamp = f32x4_neg(v_clamp);
        let v127 = i32x4_splat(127);
        let vneg126 = i32x4_splat(-126);
        let v_bias = i32x4_splat(127);

        let mut i = 0;
        let chunks = out.len() / 4;

        for _ in 0..chunks {
            // y = a[i] + q[i]  (the pre-sigmoid activation).
            let va = v128_load(a.as_ptr().add(i).cast());
            let vq = v128_load(q.as_ptr().add(i).cast());
            let vy = f32x4_add(va, vq);

            // σ(y) = 1/(1 + exp(-y)). Negate y, then the standard range-reduction
            // + polynomial + 2^n path identical to `wasm32_exp_inplace`.
            let vx = f32x4_neg(vy);

            let vn_f = f32x4_nearest(f32x4_mul(vx, v_inv_ln2));
            let vn_i = i32x4_trunc_sat_f32x4(vn_f);

            let vg = f32x4_sub(
                f32x4_sub(vx, f32x4_mul(vn_f, v_ln2_hi)),
                f32x4_mul(vn_f, v_ln2_lo),
            );

            // Cephes 6th-order polynomial — Horner form matching `cephes_exp_scalar`.
            let gc_sixth = f32x4_mul(vg, v_sixth);
            let p6 = f32x4_add(v_one, gc_sixth);
            let gc_fifth = f32x4_mul(vg, v_fifth);
            let p5 = f32x4_add(v_one, f32x4_mul(gc_fifth, p6));
            let gc_quarter = f32x4_mul(vg, v_quarter);
            let p4 = f32x4_add(v_one, f32x4_mul(gc_quarter, p5));
            let gc_third = f32x4_mul(vg, v_third);
            let p3 = f32x4_add(v_one, f32x4_mul(gc_third, p4));
            let gc_half = f32x4_mul(vg, v_half);
            let p2 = f32x4_add(v_one, f32x4_mul(gc_half, p3));
            let qpoly = f32x4_add(v_one, f32x4_mul(vg, p2));

            // 2^n via branchless bit manipulation. Clamp n to [-126, 127] — also
            // folds the |x| > 40 early-exit: exp(-y) for large positive y
            // underflows to 0 (σ → 1), for large negative y overflows to inf
            // (σ → 0). Both are the correct sigmoid saturation.
            let vn_clamped = i32x4_max(i32x4_min(vn_i, v127), vneg126);
            let v_shifted = i32x4_shl(i32x4_add(vn_clamped, v_bias), 23);
            let exp_neg_y = f32x4_mul(v_shifted, qpoly);

            // σ(y) = 1 / (1 + exp(-y)). f32x4_div gives ~1 ULP.
            let denom = f32x4_add(v_one, exp_neg_y);
            let sigma = f32x4_div(v_one, denom);

            // 2·σ − 1, then clamp(-clamp, clamp) via min/max.
            let tanh_like = f32x4_sub(f32x4_mul(v_two, sigma), v_one);
            let clamped = f32x4_max(f32x4_min(tanh_like, v_clamp), v_neg_clamp);

            v128_store(out.as_mut_ptr().add(i).cast(), clamped);
            i += 4;
        }

        // Scalar tail — bit-exact with the pre-SIMD code via fast_sigmoid.
        while i < out.len() {
            let s = fast_sigmoid(*a.get_unchecked(i) + *q.get_unchecked(i));
            let v = 2.0 * s - 1.0;
            *out.get_unchecked_mut(i) = v.clamp(-clamp, clamp);
            i += 1;
        }
    }
}
