//! SIMD dot products, outer-product accumulator, and matrix-vector kernels.
//!
//! - `simd_dot_f32` — matmul inner loop workhorse (NEON/AVX2/scalar dispatch)
//! - `simd_dot_f16_f32` — mixed-precision f16 weight × f32 input
//! - `simd_outer_product_acc` — HLA state update
//! - `simd_matvec`, `simd_matmul_rows*`, `simd_matmul_relu_rows`
//! - `simd_matmul_f16_f32_rows*`
//!
//! Backends share `is_avx2_fma_available` (from `super`) and AVX2 horizontal
//! reducers (from `super::horizontal`).

// x86_64 dispatch helpers from the parent `simd` module. Gated so other
// architectures don't see an unused-import warning.
#[cfg(target_arch = "x86_64")]
use super::horizontal::horizontal_sum_256;
#[cfg(target_arch = "x86_64")]
use super::is_avx2_fma_available;

// ── Dot Product ───────────────────────────────────────────────

/// SIMD-accelerated dot product: `Σ a[i] * b[i]` for `len` elements.
///
/// Dispatches to NEON, AVX2, or scalar based on compile-time target and
/// runtime CPU feature detection.
#[inline(always)]
pub fn simd_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    // Soundness gate (Issue 700): this is a SAFE fn whose backends do unchecked
    // pointer loads/stores up to `len`. Reslicing here turns a caller's bad `len`
    // into a clean panic instead of an OOB read/write, and hands LLVM the length.
    let (a, b) = (&a[..len], &b[..len]);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dot_f32(a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_dot_f32(a, b, len) }
        } else {
            scalar_dot_f32(a, b, len)
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_simd128_dot_f32(a, b, len) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_dot_f32(a, b, len)
    }
}

/// Single-row FMA: `Σ weight_row[i] * input[i]` — alias for [`simd_dot_f32`].
/// Named for clarity in matmul context.
#[inline(always)]
pub fn simd_fma_row(weight_row: &[f32], input: &[f32], len: usize) -> f32 {
    simd_dot_f32(weight_row, input, len)
}

/// Order-stable sequential dot product: `Σ a[i] * b[i]`, accumulated in
/// ascending index order with plain `mul`/`add` — no SIMD lane
/// reassociation, no FMA contraction (Rust does not reassociate float adds
/// without fast-math, so the fold is deterministic by construction).
///
/// [`simd_dot_f32`] computes the same sum through SIMD lanes (or the
/// 4-accumulator + `mul_add` scalar fallback), which reassociates the
/// additions — on the same input the two generally differ in the last ULP.
/// `ordered_dot_differs_from_simd_dot` in this module's tests pins a
/// crafted input where they diverge, so the distinction stays load-bearing
/// and a future "dedup" that collapses one into the other reds by
/// construction.
///
/// **When to use**: paths whose outputs are committed or compared bit-wise
/// across nodes (consensus scoring, Merkle-committed embeddings, replay
/// determinism). For inference kernels, prefer [`simd_dot_f32`].
///
/// Panics if lengths differ.
#[inline(always)]
pub fn dot_f32_ordered(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_f32_ordered: length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    // 4 independent accumulators (4 elements per outer iter) — same pattern as
    // `scalar_dot_f64` in peira.rs. Single-accumulator dot is FMA-latency-bound
    // (~4 cycles/iter on most FPUs); 4 parallel accumulators keep the FMA
    // pipeline full and let LLVM emit 4-wide unrolled FMA on targets without
    // hardware f32 SIMD (WASM, RISC-V, debug builds, or x86_64 without AVX2).
    //
    // `mul_add` (not `+= a * b`) preserves single-rounding FMA semantics on
    // hardware that has it, matching the SIMD path's `_mm256_fmadd_ps` /
    // `vfmaq_f32` numerically. On non-FMA targets `mul_add` falls back to
    // separate mul+add, which is bit-identical to the previous single-acc form.
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

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        // 4 independent accumulators to hide FMA pipeline latency
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks4 = len / 16;

        for _ in 0..chunks4 {
            acc0 = vfmaq_f32(
                acc0,
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vld1q_f32(a.as_ptr().add(i + 4)),
                vld1q_f32(b.as_ptr().add(i + 4)),
            );
            acc2 = vfmaq_f32(
                acc2,
                vld1q_f32(a.as_ptr().add(i + 8)),
                vld1q_f32(b.as_ptr().add(i + 8)),
            );
            acc3 = vfmaq_f32(
                acc3,
                vld1q_f32(a.as_ptr().add(i + 12)),
                vld1q_f32(b.as_ptr().add(i + 12)),
            );
            i += 16;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        let mut acc_rem = vdupq_n_f32(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            acc_rem = vfmaq_f32(
                acc_rem,
                vld1q_f32(a.as_ptr().add(i)),
                vld1q_f32(b.as_ptr().add(i)),
            );
            i += 4;
        }
        sum += vaddvq_f32(acc_rem);

        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps};

    unsafe {
        // 4 independent accumulators to hide FMA pipeline latency
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        let chunks4 = len / 32;

        for _ in 0..chunks4 {
            acc0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                acc0,
            );
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                acc3,
            );
            i += 32;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = horizontal_sum_256(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ));

        // Handle remaining elements with single accumulator
        let mut acc = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        sum += horizontal_sum_256(acc);

        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

/// WASM SIMD128 dot product — 4-wide f32, 4 independent accumulators.
///
/// Issue 007: ports the NEON structure to `core::arch::wasm32`. WASM
/// SIMD128 base proposal has no FMA intrinsic, so this uses separate
/// `f32x4_mul` + `f32x4_add` (the engine / wasmtime JIT fuses them when
/// profitable). Bit-identical accumulation order to the NEON kernel modulo
/// FMA contraction (NEON uses `vfmaq_f32`, WASM uses mul→add).
///
/// Compile-time gated by `target_feature = "simd128"`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128_load};

    unsafe {
        // 4 independent accumulators (4 lanes each = 16 elements per outer
        // iter) to hide the mul→add latency chain. Mirrors the NEON kernel's
        // unroll factor.
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut acc2 = f32x4_splat(0.0);
        let mut acc3 = f32x4_splat(0.0);
        let mut i = 0usize;
        let chunks4 = len / 16;

        for _ in 0..chunks4 {
            acc0 = f32x4_add(
                f32x4_mul(
                    v128_load(a.as_ptr().add(i).cast()),
                    v128_load(b.as_ptr().add(i).cast()),
                ),
                acc0,
            );
            acc1 = f32x4_add(
                f32x4_mul(
                    v128_load(a.as_ptr().add(i + 4).cast()),
                    v128_load(b.as_ptr().add(i + 4).cast()),
                ),
                acc1,
            );
            acc2 = f32x4_add(
                f32x4_mul(
                    v128_load(a.as_ptr().add(i + 8).cast()),
                    v128_load(b.as_ptr().add(i + 8).cast()),
                ),
                acc2,
            );
            acc3 = f32x4_add(
                f32x4_mul(
                    v128_load(a.as_ptr().add(i + 12).cast()),
                    v128_load(b.as_ptr().add(i + 12).cast()),
                ),
                acc3,
            );
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

        // Tail: process remaining 4-element chunks with a single accumulator.
        let mut acc = f32x4_splat(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            acc = f32x4_add(
                f32x4_mul(
                    v128_load(a.as_ptr().add(i).cast()),
                    v128_load(b.as_ptr().add(i).cast()),
                ),
                acc,
            );
            i += 4;
        }
        sum += f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        // Scalar tail for the last 0..3 elements.
        while i < len {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

// ── Outer Product Accumulation ────────────────────────

/// SIMD-accelerated outer product accumulation: `acc[i*n + j] += a[i] * b[j]`.
///
/// Used for HLA rank-1 updates (SK += kkᵀ, CQV += qvᵀ, PKV += kvᵀ).
/// `acc` is `[m × n]` row-major, `a` is `[m]`, `b` is `[n]`.
#[inline(always)]
pub fn simd_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_outer_product_acc(acc, a, b, m, n) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_outer_product_acc(acc, a, b, m, n) }
        } else {
            scalar_outer_product_acc(acc, a, b, m, n);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_simd128_outer_product_acc(acc, a, b, m, n) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_outer_product_acc(acc, a, b, m, n);
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    for i in 0..m {
        let ai = unsafe { *a.get_unchecked(i) };
        let row = &mut acc[i * n..i * n + n];
        for j in 0..n {
            unsafe {
                // FMA: acc[j] = ai * b[j] + acc[j] (single rounding, matches SIMD path).
                let bj = *b.get_unchecked(j);
                *row.get_unchecked_mut(j) = ai.mul_add(bj, *row.get_unchecked(j));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    use core::arch::aarch64::{vfmaq_f32, vld1q_dup_f32, vld1q_f32, vst1q_f32};

    unsafe {
        let n_chunks = n / 4;

        for i in 0..m {
            let ai = *a.get_unchecked(i);
            let va = vld1q_dup_f32(&ai);
            let row = &mut acc[i * n..i * n + n];

            let mut j = 0;
            for _ in 0..n_chunks {
                let vacc = vld1q_f32(row.as_ptr().add(j));
                let vb = vld1q_f32(b.as_ptr().add(j));
                let vresult = vfmaq_f32(vacc, va, vb);
                vst1q_f32(row.as_mut_ptr().add(j), vresult);
                j += 4;
            }

            while j < n {
                *row.get_unchecked_mut(j) += ai * *b.get_unchecked(j);
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    use core::arch::x86_64::{
        _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps,
    };

    unsafe {
        let n_chunks8 = n / 8;

        for i in 0..m {
            let ai = *a.get_unchecked(i);
            let va = _mm256_broadcast_ss(&ai);
            let row = &mut acc[i * n..i * n + n];

            let mut j = 0;
            for _ in 0..n_chunks8 {
                let vacc = _mm256_loadu_ps(row.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                let vresult = _mm256_fmadd_ps(va, vb, vacc);
                _mm256_storeu_ps(row.as_mut_ptr().add(j), vresult);
                j += 8;
            }

            while j < n {
                *row.get_unchecked_mut(j) += ai * *b.get_unchecked(j);
                j += 1;
            }
        }
    }
}

// ── WASM SIMD128 outer-product accumulation (Plan 008 Step 7b — port from riir-engine) ──

/// WASM SIMD128 outer-product accumulate: `acc[i,j] += a[i] * b[j]`.
///
/// Inner loop (j-dim) is vectorized as 4-wide FMA. The outer loop is scalar
/// (single broadcast of `a[i]` per row). Ported verbatim from the proven
/// riir-engine `simd::wasm32::simd_outer_product_acc` (Plan 286 T6/T7) so
/// that `katgpt_core::simd::simd_outer_product_acc` is no longer scalar-bound
/// on WASM targets.
///
/// Compile-time gated by `target_feature = "simd128"`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_outer_product_acc(
    acc: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, v128_load, v128_store};

    unsafe {
        let simd_n = n / 4 * 4;
        for i in 0..m {
            let ai = *a.get_unchecked(i);
            let vai = f32x4_splat(ai);
            let row = &mut acc[i * n..i * n + n];

            let mut j = 0;
            while j < simd_n {
                let vacc = v128_load(row.as_ptr().add(j) as *const _);
                let vb = v128_load(b.as_ptr().add(j) as *const _);
                let r = f32x4_add(vacc, f32x4_mul(vai, vb));
                v128_store(row.as_mut_ptr().add(j) as *mut _, r);
                j += 4;
            }
            for jj in simd_n..n {
                *row.get_unchecked_mut(jj) += ai * *b.get_unchecked(jj);
            }
        }
    }
}

// ── Scaled Outer Product Accumulation ─────────────────────────

/// SIMD-accelerated scaled outer product accumulation:
/// `acc[i*n + j] += scale * a[i] * b[j]`.
///
/// Used by `tiled_attention_parallax_forward` Phase 2, where it folds the
/// per-column attention weight `c_j` into the broadcast multiplier. This
/// removes the `pv_buf` materialization (`d` writes + `d` reads per `j`)
/// that the unscaled [`simd_outer_product_acc`] required, and replaces it
/// with a single FMA inside the inner SIMD kernel.
#[inline(always)]
pub fn simd_outer_product_acc_scaled(
    acc: &mut [f32],
    scale: f32,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_outer_product_acc_scaled(acc, scale, a, b, m, n) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_outer_product_acc_scaled(acc, scale, a, b, m, n) }
        } else {
            scalar_outer_product_acc_scaled(acc, scale, a, b, m, n);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_acc_scaled(acc, scale, a, b, m, n);
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_outer_product_acc_scaled(
    acc: &mut [f32],
    scale: f32,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) {
    for i in 0..m {
        // FMA: scale * a[i] + 0.0 — keeps single-rounding parity with the
        // SIMD paths' broadcast+mul, then per-j FMA matches `simd_outer_product_acc`.
        let ai = scale.mul_add(unsafe { *a.get_unchecked(i) }, 0.0);
        let row = &mut acc[i * n..i * n + n];
        for j in 0..n {
            unsafe {
                let bj = *b.get_unchecked(j);
                *row.get_unchecked_mut(j) = ai.mul_add(bj, *row.get_unchecked(j));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_outer_product_acc_scaled(
    acc: &mut [f32],
    scale: f32,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) {
    use core::arch::aarch64::{vfmaq_f32, vld1q_dup_f32, vld1q_f32, vst1q_f32};

    unsafe {
        let n_chunks = n / 4;

        for i in 0..m {
            // Fold `scale` into the broadcast multiplier — single FMA per row,
            // no extra SIMD op in the inner loop.
            let ai_scaled = scale * *a.get_unchecked(i);
            let va = vld1q_dup_f32(&ai_scaled);
            let row = &mut acc[i * n..i * n + n];

            let mut j = 0;
            for _ in 0..n_chunks {
                let vacc = vld1q_f32(row.as_ptr().add(j));
                let vb = vld1q_f32(b.as_ptr().add(j));
                let vresult = vfmaq_f32(vacc, va, vb);
                vst1q_f32(row.as_mut_ptr().add(j), vresult);
                j += 4;
            }

            while j < n {
                *row.get_unchecked_mut(j) += ai_scaled * *b.get_unchecked(j);
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_outer_product_acc_scaled(
    acc: &mut [f32],
    scale: f32,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) {
    use core::arch::x86_64::{
        _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps,
    };

    unsafe {
        let n_chunks8 = n / 8;

        for i in 0..m {
            // Fold `scale` into the broadcast multiplier.
            let ai_scaled = scale * *a.get_unchecked(i);
            let va = _mm256_broadcast_ss(&ai_scaled);
            let row = &mut acc[i * n..i * n + n];

            let mut j = 0;
            for _ in 0..n_chunks8 {
                let vacc = _mm256_loadu_ps(row.as_ptr().add(j));
                let vb = _mm256_loadu_ps(b.as_ptr().add(j));
                let vresult = _mm256_fmadd_ps(va, vb, vacc);
                _mm256_storeu_ps(row.as_mut_ptr().add(j), vresult);
                j += 8;
            }

            while j < n {
                *row.get_unchecked_mut(j) += ai_scaled * *b.get_unchecked(j);
                j += 1;
            }
        }
    }
}

// ── Matrix-Vector Multiply ────────────────────────────────────

/// SIMD-accelerated matvec: `acc[i] = Σ mat[i*cols + j] * vec[j]` for each row.
///
/// I.e. computes `acc = mat · vec` (standard row-major `M·v`).
/// `mat` is `[rows × cols]` row-major, `vec` is `[cols]`, `acc` is `[rows]`.
///
/// WARNING: this is NOT the HLA `qᵀ · M` readout kernel. For HLA readouts
/// (`qᵀ·SK`, `qᵀ·PKV`, `qᵀ·E`) use `katgpt_hla::transpose_matvec_into` /
/// the per-crate mirror in riir-engine — those compute the *transpose*
/// product `out[j] = Σ_i vec[i]·mat[i,j]`, which is structurally different
/// from this `M·v` for the non-symmetric matrices (PKV, E) used in AHLA.
/// Issue 009 (2026-06-30): confusing this docstring's old claim "Used for
/// HLA readout (qᵀ·SK, qᵀ·PKV)" with the function's actual `M·v` semantics
/// caused a 5-site silent regression in riir-engine (commit 0d3f9c19).
#[inline(always)]
pub fn simd_matvec(acc: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *acc.get_unchecked_mut(r) = simd_dot_f32(&mat[row_off..row_off + cols], vec, cols);
        }
    }
}

// ── Transpose Matrix-Vector Multiply (Wᵀ · v) ─────────────────

/// SIMD-accelerated transpose matvec: `out[j] = Σ_i mat[i*cols + j] * v[i]`.
///
/// I.e. computes `out = matᵀ · v` for row-major `[rows × cols]` mat — the
/// access pattern needed by `dL/dh = Wᵀ · dL/dy` in analytic backward passes
/// (MLA/MoE/KDA). This is NOT [`simd_matvec`] (`M·v`, independent per-row dot
/// products) — the transpose product accumulates across ALL rows into every
/// output element, so the loop order is inverted: outer loop over rows
/// (broadcast `v[i]`), inner loop over cols (contiguous FMA into `out`). A
/// naive column-outer loop (`for j { for i { acc += mat[i*cols+j] * v[i] } }`)
/// has stride-`cols` reads and defeats auto-vectorization; this loop keeps
/// both the `mat` row read and the `out` accumulation contiguous (mirrors
/// the loop-interchange already used by `katgpt_hla::kernel::transpose_matvec_into`).
///
/// Zeroes `out` first. Use [`simd_transpose_matvec_acc`] to accumulate into
/// an existing buffer.
#[inline(always)]
pub fn simd_transpose_matvec_into(out: &mut [f32], mat: &[f32], v: &[f32], rows: usize, cols: usize) {
    out[..cols].fill(0.0);
    simd_transpose_matvec_acc(out, mat, v, rows, cols);
}

/// Accumulating variant of [`simd_transpose_matvec_into`]:
/// `out[j] += Σ_i mat[i*cols + j] * v[i]`.
#[inline(always)]
pub fn simd_transpose_matvec_acc(out: &mut [f32], mat: &[f32], v: &[f32], rows: usize, cols: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_transpose_matvec_acc(out, mat, v, rows, cols) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_transpose_matvec_acc(out, mat, v, rows, cols) }
        } else {
            scalar_transpose_matvec_acc(out, mat, v, rows, cols);
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_simd128_transpose_matvec_acc(out, mat, v, rows, cols) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_transpose_matvec_acc(out, mat, v, rows, cols);
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_transpose_matvec_acc(
    out: &mut [f32],
    mat: &[f32],
    v: &[f32],
    rows: usize,
    cols: usize,
) {
    for i in 0..rows {
        let vi = unsafe { *v.get_unchecked(i) };
        let row = &mat[i * cols..i * cols + cols];
        for j in 0..cols {
            unsafe {
                let mj = *row.get_unchecked(j);
                // FMA: single-rounding, matches the SIMD paths' broadcast+FMA.
                *out.get_unchecked_mut(j) = vi.mul_add(mj, *out.get_unchecked(j));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_transpose_matvec_acc(out: &mut [f32], mat: &[f32], v: &[f32], rows: usize, cols: usize) {
    use core::arch::aarch64::{vfmaq_f32, vld1q_dup_f32, vld1q_f32, vst1q_f32};

    unsafe {
        let n_chunks = cols / 4;

        for i in 0..rows {
            let vi = *v.get_unchecked(i);
            let va = vld1q_dup_f32(&vi);
            let row = &mat[i * cols..i * cols + cols];

            let mut j = 0;
            for _ in 0..n_chunks {
                let vacc = vld1q_f32(out.as_ptr().add(j));
                let vm = vld1q_f32(row.as_ptr().add(j));
                let vresult = vfmaq_f32(vacc, va, vm);
                vst1q_f32(out.as_mut_ptr().add(j), vresult);
                j += 4;
            }

            while j < cols {
                *out.get_unchecked_mut(j) += vi * *row.get_unchecked(j);
                j += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_transpose_matvec_acc(out: &mut [f32], mat: &[f32], v: &[f32], rows: usize, cols: usize) {
    use core::arch::x86_64::{
        _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps,
    };

    unsafe {
        let n_chunks8 = cols / 8;

        for i in 0..rows {
            let vi = *v.get_unchecked(i);
            let va = _mm256_broadcast_ss(&vi);
            let row = &mat[i * cols..i * cols + cols];

            let mut j = 0;
            for _ in 0..n_chunks8 {
                let vacc = _mm256_loadu_ps(out.as_ptr().add(j));
                let vm = _mm256_loadu_ps(row.as_ptr().add(j));
                let vresult = _mm256_fmadd_ps(va, vm, vacc);
                _mm256_storeu_ps(out.as_mut_ptr().add(j), vresult);
                j += 8;
            }

            while j < cols {
                *out.get_unchecked_mut(j) += vi * *row.get_unchecked(j);
                j += 1;
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_transpose_matvec_acc(
    out: &mut [f32],
    mat: &[f32],
    v: &[f32],
    rows: usize,
    cols: usize,
) {
    use core::arch::wasm32::{f32x4_add, f32x4_mul, f32x4_splat, v128_load, v128_store};

    unsafe {
        let simd_cols = cols / 4 * 4;
        for i in 0..rows {
            let vi = *v.get_unchecked(i);
            let va = f32x4_splat(vi);
            let row = &mat[i * cols..i * cols + cols];

            let mut j = 0;
            while j < simd_cols {
                let vacc = v128_load(out.as_ptr().add(j) as *const _);
                let vm = v128_load(row.as_ptr().add(j) as *const _);
                let r = f32x4_add(vacc, f32x4_mul(va, vm));
                v128_store(out.as_mut_ptr().add(j) as *mut _, r);
                j += 4;
            }
            for jj in simd_cols..cols {
                *out.get_unchecked_mut(jj) += vi * *row.get_unchecked(jj);
            }
        }
    }
}

// ── Matmul Row Dispatch ───────────────────────────────────────

/// SIMD-accelerated matmul dispatch: `output[r] = dot(weight_row_r, input)`.
///
/// Replaces the inner loop of `matmul()` in `types.rs`.
#[inline(always)]
pub fn simd_matmul_rows(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *output.get_unchecked_mut(r) =
                simd_dot_f32(&weight[row_off..row_off + cols], input, cols);
        }
    }
}

/// Row-parallel matmul: splits output rows across rayon threads (Plan 096).
///
/// Each thread gets an exclusive `&mut [f32]` chunk of the output and reads
/// its corresponding weight rows (read-only). The input vector is shared (read-only).
///
/// Use this for large matmuls where row count >> core count:
/// - `down_proj`: 2304×9216 (21.1% of decode time)
/// - `lm_head`: 256000×2304 (22.6% of decode time)
///
/// Falls back to sequential `simd_matmul_rows` for small matmuls (rows < threshold).
#[inline]
pub fn simd_matmul_rows_parallel(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    /// Minimum rows before parallelizing. Below this, sequential is faster
    /// due to rayon thread pool scheduling overhead (~1-5µs per task).
    /// At 9216 rows, parallel gives ~3-4× on 8+ cores.
    const PARALLEL_ROWS_MIN: usize = 512;

    if rows < PARALLEL_ROWS_MIN {
        // Sequential: overhead would exceed savings
        simd_matmul_rows(output, weight, input, rows, cols);
    } else {
        // Parallel: split output into row chunks, each thread processes its chunk.
        // chunk_rows=256 balances parallelism (36 chunks for 9216 rows) with
        // low scheduling overhead (~1µs per task on Apple M3 Max).
        use rayon::prelude::*;
        const PARALLEL_CHUNK_ROWS: usize = 256;
        output
            .par_chunks_mut(PARALLEL_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * PARALLEL_CHUNK_ROWS;
                for (local_r, out) in out_chunk.iter_mut().enumerate() {
                    let r = start_row + local_r;
                    let row_off = r * cols;
                    *out = simd_dot_f32(&weight[row_off..row_off + cols], input, cols);
                }
            });
    }
}

/// Batched dense matmul with weight-reuse loop ordering (Issue 597).
///
/// Computes `y[p, r] = dot(weight[r, :], x[p, :])` for `p in 0..batch`, `r in 0..rows`.
/// Input `x` is `[batch, cols]` row-major; output `y` is `[batch, rows]` row-major;
/// weight is `[rows, cols]` row-major (same layout as [`simd_matmul_rows`]).
///
/// **Why the loop nest matters:** the naive approach (`batch` calls to
/// [`simd_matmul_rows`]) loads each weight row `batch` times — once per
/// position — because the weight matrix typically exceeds L2 cache. This
/// kernel transposes the loop nest so the outer loop is over weight rows
/// and the inner loop is over batch positions: each weight row is loaded
/// once and reused across all `batch` inputs, cutting weight memory traffic
/// by a factor of `batch`.
///
/// The result is **bit-identical** to `batch` separate [`simd_matmul_rows`]
/// calls — each output element is the same `simd_dot_f32(weight_row, x_row)`
/// sum, just computed in a different loop order. No floating-point
/// reassociation occurs inside the dot product.
///
/// For `batch == 1` this degenerates to exactly [`simd_matmul_rows`].
#[inline]
pub fn simd_matmul_rows_batched(
    y: &mut [f32],
    weight: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
) {
    if batch == 0 || rows == 0 {
        return;
    }
    debug_assert_eq!(weight.len(), rows * cols, "weight shape mismatch");
    debug_assert_eq!(x.len(), batch * cols, "x shape mismatch");
    debug_assert_eq!(y.len(), batch * rows, "y shape mismatch");

    // Weight-reuse loop nest: outer = weight rows, inner = batch positions.
    // Each weight row (cols f32s) is loaded once and dotted with all batch
    // inputs before moving to the next row.
    for r in 0..rows {
        let w_row = &weight[r * cols..(r + 1) * cols];
        for p in 0..batch {
            let x_row = &x[p * cols..(p + 1) * cols];
            unsafe {
                *y.get_unchecked_mut(p * rows + r) = simd_dot_f32(w_row, x_row, cols);
            }
        }
    }
}

/// SIMD-accelerated matmul + ReLU: `output[r] = max(0, dot(weight_row_r, input))`.
///
/// Replaces the inner loop of `matmul_relu()` in `types.rs`.
#[inline(always)]
pub fn simd_matmul_relu_rows(
    output: &mut [f32],
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        let sum = simd_dot_f32(&weight[row_off..row_off + cols], input, cols);
        unsafe {
            *output.get_unchecked_mut(r) = sum.max(0.0);
        }
    }
}

// ── f16×f32 Mixed-Precision Kernels ──────────────────────────

/// x86_64 AVX2+F16C dot kernel: `Σ f16_weight[i] * f32_input[i]` with the
/// f16→f32 widening done IN-REGISTER by `vcvtph2ps` (8 lanes/instr).
///
/// The x86 fallback this replaces measured **1 tok/s** on the 13700K
/// gemma-2-2b f16 lane (the 883 P0 calibration box) against the NEON
/// path's M3 numbers — `scalar_dot_f16_f32` was the only non-aarch64
/// backend. Structure mirrors `avx2_dot_f32` (4 independent accumulators,
/// 32 elements/iter, 8-wide leftover block, scalar tail); the f16 input
/// is loaded as `__m128i` (8×u16) and widened once per 8 lanes — no
/// per-element `to_f32()` call on the critical path.
///
/// Numerics: same FMA-lane semantics as the f32 AVX2 kernel (single
/// rounding per mul-add, lane-parallel then horizontal reduce) — a
/// DIFFERENT accumulation order than the scalar fallback's 4-chain
/// mul_add, in the same way `avx2_dot_f32` already differs from
/// `scalar_dot_f32`. Dispatch prefers this only under the runtime F16C
/// probe; F16C-less x86 keeps the scalar backend bit-identically.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn avx2_dot_f16_f32(w_f16: &[half::f16], x_f32: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps,
        _mm_loadu_si128,
    };
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut i = 0;
        let chunks4 = len / 32;

        for _ in 0..chunks4 {
            // 8 f16 weights per 128-bit load → widen → FMADD against f32.
            let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_f16.as_ptr().add(i) as *const __m128i));
            acc0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x_f32.as_ptr().add(i)), acc0);
            let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_f16.as_ptr().add(i + 8) as *const __m128i));
            acc1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x_f32.as_ptr().add(i + 8)), acc1);
            let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_f16.as_ptr().add(i + 16) as *const __m128i));
            acc2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x_f32.as_ptr().add(i + 16)), acc2);
            let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_f16.as_ptr().add(i + 24) as *const __m128i));
            acc3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x_f32.as_ptr().add(i + 24)), acc3);
            i += 32;
        }

        let mut sum = horizontal_sum_256(_mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        ));

        let mut acc = _mm256_setzero_ps();
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let w = _mm256_cvtph_ps(_mm_loadu_si128(w_f16.as_ptr().add(i) as *const __m128i));
            acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x_f32.as_ptr().add(i)), acc);
            i += 8;
        }
        sum += horizontal_sum_256(acc);

        while i < len {
            sum += (*w_f16.get_unchecked(i)).to_f32() * *x_f32.get_unchecked(i);
            i += 1;
        }
        sum
    }
}

/// SIMD dot product: `Σ f16_weight[i] * f32_input[i]`.
///
/// Converts f16 weights to f32 on-the-fly during accumulation.
/// This is the hot-path for f16 weight inference — halves memory bandwidth
/// for weight reads while maintaining f32 precision for accumulation.
#[inline]
pub fn simd_dot_f16_f32(w_f16: &[half::f16], x_f32: &[f32], len: usize) -> f32 {
    // Soundness gate (Issue 700): this is a SAFE fn whose backends do unchecked
    // pointer loads/stores up to `len`. Reslicing here turns a caller's bad `len`
    // into a clean panic instead of an OOB read/write, and hands LLVM the length.
    let (w_f16, x_f32) = (&w_f16[..len], &x_f32[..len]);
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dot_f16_f32(w_f16, x_f32, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if super::is_f16c_available() {
            unsafe { avx2_dot_f16_f32(w_f16, x_f32, len) }
        } else {
            scalar_dot_f16_f32(w_f16, x_f32, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_dot_f16_f32(w_f16, x_f32, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_dot_f16_f32(w: &[half::f16], x: &[f32], len: usize) -> f32 {
    // 4 independent accumulators — mirrors `scalar_dot_f32` (4-wide unroll)
    // and the NEON `neon_dot_f16_f32` path (also 4 accs). Single-accumulator
    // dot is FMA-latency-bound; the f16→f32 widening is cheap (1 cycle) and
    // not the bottleneck. mul_add preserves single-rounding FMA parity with
    // the NEON path's vfmaq_f32 lane semantics.
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] = (*w.get_unchecked(i))
                .to_f32()
                .mul_add(*x.get_unchecked(i), acc[0]);
            acc[1] = (*w.get_unchecked(i + 1))
                .to_f32()
                .mul_add(*x.get_unchecked(i + 1), acc[1]);
            acc[2] = (*w.get_unchecked(i + 2))
                .to_f32()
                .mul_add(*x.get_unchecked(i + 2), acc[2]);
            acc[3] = (*w.get_unchecked(i + 3))
                .to_f32()
                .mul_add(*x.get_unchecked(i + 3), acc[3]);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < len {
        unsafe {
            sum = (*w.get_unchecked(i))
                .to_f32()
                .mul_add(*x.get_unchecked(i), sum);
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
/// Convert 4 f16 values (8 bytes at `w_ptr`) to 4 f32 values using the
/// hardware `fcvtl` instruction via inline asm.
///
/// This avoids the unstable `float16x4_t` / `vcvt_f32_f16` intrinsics
/// (stdarch_neon_f16, issue #136306) while using a single hardware instruction
/// instead of ~10 integer bit-manipulation ops or 4 scalar `to_f32()` calls.
///
/// Only clobbers v0 and v1 — does NOT use `clobber_abi("C")`, so the
/// compiler can keep FMA accumulators in other NEON registers live
/// across calls. This is critical for the dot-product inner loop.
///
/// SAFETY: requires NEON target feature. `w_ptr` must point to at least 4
/// valid u16 values (8 bytes). `out` must point to at least 4 f32 values
/// (16 bytes).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fcvtl_4x(w_ptr: *const u16, out: *mut f32) {
    let mut _v0: f32;
    let mut _v1: f32;
    unsafe {
        core::arch::asm!(
            "ldr d0, [{addr}]",
            "fcvtl v1.4s, v0.4h",
            "str q1, [{out}]",
            addr = in(reg) w_ptr,
            out = in(reg) out,
            lateout("v0") _v0,
            lateout("v1") _v1,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot_f16_f32(w: &[half::f16], x: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        // 4 independent accumulators to hide FMA pipeline latency
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut i = 0;

        // Cast half::f16 slice to u16 pointer for fcvtl load.
        // half::f16 is #[repr(transparent)] over u16, so this is sound.
        let w_ptr = w.as_ptr() as *const u16;

        // Stack buffer for f16→f32 conversion. Reused across iterations.
        let mut buf = [0.0f32; 4];

        // Process 16 elements per iteration: 4× (load 4 f16 → fcvtl → FMA).
        let chunks16 = len / 16;
        for _ in 0..chunks16 {
            fcvtl_4x(w_ptr.add(i), buf.as_mut_ptr());
            let w0 = vld1q_f32(buf.as_ptr());
            fcvtl_4x(w_ptr.add(i + 4), buf.as_mut_ptr());
            let w1 = vld1q_f32(buf.as_ptr());
            fcvtl_4x(w_ptr.add(i + 8), buf.as_mut_ptr());
            let w2 = vld1q_f32(buf.as_ptr());
            fcvtl_4x(w_ptr.add(i + 12), buf.as_mut_ptr());
            let w3 = vld1q_f32(buf.as_ptr());

            acc0 = vfmaq_f32(acc0, w0, vld1q_f32(x.as_ptr().add(i)));
            acc1 = vfmaq_f32(acc1, w1, vld1q_f32(x.as_ptr().add(i + 4)));
            acc2 = vfmaq_f32(acc2, w2, vld1q_f32(x.as_ptr().add(i + 8)));
            acc3 = vfmaq_f32(acc3, w3, vld1q_f32(x.as_ptr().add(i + 12)));
            i += 16;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        // Handle remaining 4-element chunks
        let chunks4 = (len - i) / 4;
        let mut acc_rem = vdupq_n_f32(0.0);
        for _ in 0..chunks4 {
            fcvtl_4x(w_ptr.add(i), buf.as_mut_ptr());
            let w0 = vld1q_f32(buf.as_ptr());
            acc_rem = vfmaq_f32(acc_rem, w0, vld1q_f32(x.as_ptr().add(i)));
            i += 4;
        }
        sum += vaddvq_f32(acc_rem);

        // Scalar tail (0-3 elements)
        while i < len {
            sum += (*w.get_unchecked(i)).to_f32() * *x.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

/// SIMD f16×f32 matmul: `output[r] = dot(f16_weight_row_r, f32_input)`.
///
/// Replaces `simd_matmul_rows()` when weights are stored as f16.
/// Each row's f16 weights are converted to f32 during the dot product,
/// halving the memory bandwidth for weight reads.
#[inline(always)]
pub fn simd_matmul_f16_f32_rows(
    output: &mut [f32],
    weight_f16: &[half::f16],
    input_f32: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *output.get_unchecked_mut(r) =
                simd_dot_f16_f32(&weight_f16[row_off..row_off + cols], input_f32, cols);
        }
    }
}

/// Row-parallel f16×f32 matmul: splits output rows across rayon threads (Plan 096).
///
/// Same as [`simd_matmul_f16_f32_rows`] but uses `par_chunks_mut` for large matmuls.
/// Falls back to sequential for rows < 512 (thread overhead exceeds savings).
#[inline]
pub fn simd_matmul_f16_f32_rows_parallel(
    output: &mut [f32],
    weight_f16: &[half::f16],
    input_f32: &[f32],
    rows: usize,
    cols: usize,
) {
    const PARALLEL_ROWS_MIN: usize = 512;

    if rows < PARALLEL_ROWS_MIN {
        simd_matmul_f16_f32_rows(output, weight_f16, input_f32, rows, cols);
    } else {
        use rayon::prelude::*;
        const PARALLEL_CHUNK_ROWS: usize = 256;
        output
            .par_chunks_mut(PARALLEL_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * PARALLEL_CHUNK_ROWS;
                for (local_r, out) in out_chunk.iter_mut().enumerate() {
                    let r = start_row + local_r;
                    let row_off = r * cols;
                    *out = simd_dot_f16_f32(&weight_f16[row_off..row_off + cols], input_f32, cols);
                }
            });
    }
}

// ── f16 × f16 → f32 widening-FMA dot (Issue 201) ────────────────────────
//
// The Issue 200 weight-only path (f16 weights × f32 activations) was net-
// negative on Apple Silicon: f32 activations limit the bandwidth reduction
// to 25%, and the FCVT latency on the critical path more than eats the
// savings. This f16×f16 path uses the ARMv8.2-A FP16 widening FMA
// (`fmlal`/`fmlal2`) which does f16×f16→f32 in a single instruction — the
// f16→f32 widening happens inside the FMA, so there is NO explicit FCVT on
// the critical path. Combined with genuine 50% bandwidth reduction (2+2=4
// bytes/element vs 4+4=8 for f32×f32), this is the path that could actually
// break the f32 GEMV bandwidth ceiling.
//
// Requires `target_feature = "fp16"` + `target_feature = "fhm"` (FP16 FML
// instructions). Available on Apple Silicon M1+.

/// Compute `acc += (a_low × b_low)` where the low 4 f16 elements of each
/// `uint16x8_t` are multiplied and widened to f32, then accumulated into
/// the f32 `acc` vector. Uses the `fmlal` instruction (FEAT_FHM) — the
/// narrow `.4h` operand view selects the bottom half of the 128-bit source
/// registers.
///
/// This is the inline-asm wrapper around the unstable `vfmlalq_low_f16`
/// intrinsic (stdarch_neon_f16, issue #136306). Uses explicit NEON register
/// allocation (v0=acc, v1=a, v2=b) + `lateout` clobber discipline so the
/// compiler can keep other f32 accumulators live across calls.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16,fhm")]
#[inline]
unsafe fn fmlal_lo_inline(
    acc: core::arch::aarch64::float32x4_t,
    a: core::arch::aarch64::uint16x8_t,
    b: core::arch::aarch64::uint16x8_t,
) -> core::arch::aarch64::float32x4_t {
    let mut acc = acc;
    unsafe {
        core::arch::asm!
            (
                "fmlal v0.4s, v1.4h, v2.4h",
                inout("v0") acc,
                in("v1") a,
                in("v2") b,
                lateout("v1") _,
                lateout("v2") _,
                options(pure, nomem, nostack, preserves_flags),
            );
        acc
    }
}

/// Compute `acc += (a_high × b_high)` where the high 4 f16 elements of each
/// `uint16x8_t` are multiplied and widened to f32. Uses `fmlal2` (FEAT_FHM) —
/// same `.4h`-suffixed operand syntax as `fmlal`; the opcode bit (not the
/// operand width) is what selects the top half of the physical 128-bit
/// source registers. Confirmed empirically: the assembler rejects `.8h`
/// operands on `fmlal2` (invalid operand for instruction).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16,fhm")]
#[inline]
unsafe fn fmlal_hi_inline(
    acc: core::arch::aarch64::float32x4_t,
    a: core::arch::aarch64::uint16x8_t,
    b: core::arch::aarch64::uint16x8_t,
) -> core::arch::aarch64::float32x4_t {
    let mut acc = acc;
    unsafe {
        core::arch::asm!
            (
                "fmlal2 v0.4s, v1.4h, v2.4h",
                inout("v0") acc,
                in("v1") a,
                in("v2") b,
                lateout("v1") _,
                lateout("v2") _,
                options(pure, nomem, nostack, preserves_flags),
            );
        acc
    }
}

/// SIMD dot product: f16 weights × f16 activations → f32 scalar.
///
/// Uses the ARMv8.2-A FP16 widening FMA (`fmlal`/`fmlal2`) which performs
/// f16×f16→f32 in a single instruction with no explicit FCVT on the critical
/// path. Halves bandwidth for BOTH weight and activation reads vs f32×f32.
///
/// Falls back to scalar `to_f32().mul_add()` on non-aarch64 targets.
///
/// Requires `fp16` + `fhm` target features at runtime.
#[inline(always)]
pub fn simd_dot_f16_f16(w: &[half::f16], x: &[half::f16], len: usize) -> f32 {
    // Soundness gate (Issue 700): this is a SAFE fn whose backends do unchecked
    // pointer loads/stores up to `len`. Reslicing here turns a caller's bad `len`
    // into a clean panic instead of an OOB read/write, and hands LLVM the length.
    let (w, x) = (&w[..len], &x[..len]);
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("fp16")
            && std::arch::is_aarch64_feature_detected!("fhm")
        {
            unsafe { neon_dot_f16_f16(w, x, len) }
        } else {
            scalar_dot_f16_f16(w, x, len)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dot_f16_f16(w, x, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_dot_f16_f16(w: &[half::f16], x: &[half::f16], len: usize) -> f32 {
    // 4 independent accumulators — mirrors scalar_dot_f32 / scalar_dot_f16_f32.
    let mut acc = [0.0f32; 4];
    let chunks = len / 4;
    let mut i = 0;
    for _ in 0..chunks {
        unsafe {
            acc[0] = (*w.get_unchecked(i))
                .to_f32()
                .mul_add((*x.get_unchecked(i)).to_f32(), acc[0]);
            acc[1] = (*w.get_unchecked(i + 1))
                .to_f32()
                .mul_add((*x.get_unchecked(i + 1)).to_f32(), acc[1]);
            acc[2] = (*w.get_unchecked(i + 2))
                .to_f32()
                .mul_add((*x.get_unchecked(i + 2)).to_f32(), acc[2]);
            acc[3] = (*w.get_unchecked(i + 3))
                .to_f32()
                .mul_add((*x.get_unchecked(i + 3)).to_f32(), acc[3]);
        }
        i += 4;
    }
    let mut sum = acc.iter().sum::<f32>();
    while i < len {
        unsafe {
            sum = (*w.get_unchecked(i))
                .to_f32()
                .mul_add((*x.get_unchecked(i)).to_f32(), sum);
        }
        i += 1;
    }
    sum
}

/// NEON f16×f16→f32 dot product using widening FMA (`fmlal`/`fmlal2`).
///
/// Processes 16 f16 elements per iteration: 2 loads of 8 f16 each, 4 widening
/// FMAs into 4 independent f32 accumulators. The widening FMA does the
/// f16→f32 conversion implicitly — no explicit FCVT on the critical path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16,fhm")]
#[inline]
unsafe fn neon_dot_f16_f16(w: &[half::f16], x: &[half::f16], len: usize) -> f32 {
    use core::arch::aarch64::{
        float32x4_t, uint16x8_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_u16,
    };

    unsafe {
        // Cast half::f16 slices to u16 pointers. half::f16 is
        // #[repr(transparent)] over u16, so this is sound.
        let w_ptr = w.as_ptr() as *const u16;
        let x_ptr = x.as_ptr() as *const u16;

        // 4 independent f32 accumulators to hide FMA pipeline latency.
        let zero = vdupq_n_f32(0.0);
        let mut acc0: float32x4_t = zero;
        let mut acc1: float32x4_t = zero;
        let mut acc2: float32x4_t = zero;
        let mut acc3: float32x4_t = zero;
        let mut i = 0;

        // Process 16 f16 elements per iteration: 2× (load 8 f16 → fmlal + fmlal2).
        let chunks16 = len / 16;
        for _ in 0..chunks16 {
            let w0: uint16x8_t = vld1q_u16(w_ptr.add(i));
            let x0: uint16x8_t = vld1q_u16(x_ptr.add(i));
            acc0 = fmlal_lo_inline(acc0, w0, x0); // low 4 elements
            acc1 = fmlal_hi_inline(acc1, w0, x0); // high 4 elements

            let w1: uint16x8_t = vld1q_u16(w_ptr.add(i + 8));
            let x1: uint16x8_t = vld1q_u16(x_ptr.add(i + 8));
            acc2 = fmlal_lo_inline(acc2, w1, x1);
            acc3 = fmlal_hi_inline(acc3, w1, x1);

            i += 16;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        // Handle remaining 8-element chunks
        let chunks8 = (len - i) / 8;
        let mut acc_lo = zero;
        let mut acc_hi = zero;
        for _ in 0..chunks8 {
            let w0: uint16x8_t = vld1q_u16(w_ptr.add(i));
            let x0: uint16x8_t = vld1q_u16(x_ptr.add(i));
            acc_lo = fmlal_lo_inline(acc_lo, w0, x0);
            acc_hi = fmlal_hi_inline(acc_hi, w0, x0);
            i += 8;
        }
        sum += vaddvq_f32(vaddq_f32(acc_lo, acc_hi));

        // Scalar tail (0-7 elements)
        while i < len {
            sum += (*w.get_unchecked(i)).to_f32() * (*x.get_unchecked(i)).to_f32();
            i += 1;
        }

        sum
    }
}

/// SIMD f16×f16 matmul: `output[r] = dot(f16_weight_row_r, f16_input)`.
///
/// Replaces `simd_matmul_f16_f32_rows()` when BOTH weights and activations
/// are stored as f16. Uses widening FMA (`fmlal`/`fmlal2`) — 50% bandwidth
/// reduction vs f32×f32 with no explicit FCVT on the critical path.
#[inline(always)]
pub fn simd_matmul_f16_f16_rows(
    output: &mut [f32],
    weight_f16: &[half::f16],
    input_f16: &[half::f16],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *output.get_unchecked_mut(r) =
                simd_dot_f16_f16(&weight_f16[row_off..row_off + cols], input_f16, cols);
        }
    }
}

/// Row-parallel f16×f16 matmul: splits output rows across rayon threads.
/// Falls back to sequential for rows < 512 (thread overhead exceeds savings).
#[inline]
pub fn simd_matmul_f16_f16_rows_parallel(
    output: &mut [f32],
    weight_f16: &[half::f16],
    input_f16: &[half::f16],
    rows: usize,
    cols: usize,
) {
    const PARALLEL_ROWS_MIN: usize = 512;

    if rows < PARALLEL_ROWS_MIN {
        simd_matmul_f16_f16_rows(output, weight_f16, input_f16, rows, cols);
    } else {
        use rayon::prelude::*;
        const PARALLEL_CHUNK_ROWS: usize = 256;
        output
            .par_chunks_mut(PARALLEL_CHUNK_ROWS)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * PARALLEL_CHUNK_ROWS;
                for (local_r, out) in out_chunk.iter_mut().enumerate() {
                    let r = start_row + local_r;
                    let row_off = r * cols;
                    *out = simd_dot_f16_f16(&weight_f16[row_off..row_off + cols], input_f16, cols);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in [-1, 1). No rand dep, reproducible.
    fn pseudo(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// The batched kernel must produce bit-identical results to `batch`
    /// separate `simd_matmul_rows` calls — same dot product, different loop
    /// order. This is the Issue 597 G1 invariant.
    #[test]
    fn batched_matches_sequential_bit_identical() {
        let rows = 32;
        let cols = 17;
        let batch = 8;

        let mut seed = 42u64;
        let weight: Vec<f32> = (0..rows * cols).map(|_| pseudo(&mut seed)).collect();
        let x: Vec<f32> = (0..batch * cols).map(|_| pseudo(&mut seed)).collect();

        // Sequential: batch separate GEMVs
        let mut y_seq = vec![0.0f32; batch * rows];
        for p in 0..batch {
            let x_row = &x[p * cols..(p + 1) * cols];
            let y_row = &mut y_seq[p * rows..(p + 1) * rows];
            simd_matmul_rows(y_row, &weight, x_row, rows, cols);
        }

        // Batched: one call
        let mut y_batch = vec![0.0f32; batch * rows];
        simd_matmul_rows_batched(&mut y_batch, &weight, &x, rows, cols, batch);

        // Bit-identical
        for p in 0..batch {
            for r in 0..rows {
                let i = p * rows + r;
                assert_eq!(
                    y_seq[i], y_batch[i],
                    "mismatch at batch={p}, row={r}: seq={}, batched={}",
                    y_seq[i], y_batch[i]
                );
            }
        }
    }

    /// `batch == 1` must degenerate to exactly `simd_matmul_rows`.
    #[test]
    fn batched_batch_one_matches_gemv() {
        let rows = 16;
        let cols = 33;
        let mut seed = 7u64;
        let weight: Vec<f32> = (0..rows * cols).map(|_| pseudo(&mut seed)).collect();
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut seed)).collect();

        let mut y_gemv = vec![0.0f32; rows];
        simd_matmul_rows(&mut y_gemv, &weight, &x, rows, cols);

        let mut y_batch = vec![0.0f32; rows];
        simd_matmul_rows_batched(&mut y_batch, &weight, &x, rows, cols, 1);

        assert_eq!(y_gemv, y_batch);
    }

    /// Zero-batch and zero-rows are no-ops (don't panic).
    #[test]
    fn batched_zero_batch_is_noop() {
        let mut y = vec![0.0f32; 0];
        simd_matmul_rows_batched(&mut y, &[], &[], 4, 4, 0);
    }
}

#[cfg(test)]
mod ordered_dot_tests {
    use super::{dot_f32_ordered, simd_dot_f32};

    /// The frozen sequential-fold value on a cancellation-heavy input.
    /// Hand-computed: ((1e8 + 1) loses the +1 to f32 spacing at 1e8 (8), so
    /// 1e8; + (-1e8) = 0; + 1 = 1.0). Any reassociating backend (SIMD lanes
    /// or the 4-accumulator scalar fallback) pairs {1e8, -1e8} = 0 and
    /// {1, 1} = 2 → 2.0. The frozen value is the CONTRACT; if it ever reds,
    /// the fold stopped being sequential.
    #[test]
    fn ordered_dot_is_the_sequential_fold() {
        let a = [1e8f32, 1.0, -1e8, 1.0];
        let b = [1.0f32; 4];
        assert_eq!(dot_f32_ordered(&a, &b), 1.0);
    }

    /// The anti-dedup pin: on this input the SIMD kernel's reassociated sum
    /// differs from the ordered fold. Crafted at len = 16 — the smallest length
    /// that engages a grouped path on EVERY backend (NEON 16-wide unroll,
    /// AVX2 8-wide remainder loop, wasm-simd128 16-wide unroll, and the scalar
    /// fallback's 4-accumulator chunks). Hand-computed, b = all ones:
    /// ordered = 1e8 +1 +1 +1 (each +1 lost, ulp 8 at 1e8) −1e8 → 0, +1+1+1
    /// = 3.0; every reassociating backend pairs the cancellation inside one
    /// lane/accumulator group and keeps the three +1s → 6.0.
    ///
    /// ⚠ Length is load-bearing: below one full vector the SIMD kernels
    /// legitimately converge with the ordered fold BY CONSTRUCTION (the AVX2
    /// tail at len < 8 IS a sequential `sum +=` loop; NEON/wasm at len < 4
    /// same; the scalar fallback at len ≤ 4 puts one element per accumulator
    /// and folds them in lane order). len = 4 was the original craft and red
    /// on x86_64 from the day it landed (caught by the x86_64 execution
    /// matrix, 2026-09-21) while passing on aarch64, where 4 IS the full
    /// vector width. If a future backend makes this red at len = 16, the two
    /// kernels have converged and this module's reason to exist must be
    /// re-adjudicated.
    #[test]
    fn ordered_dot_differs_from_simd_dot() {
        let mut a = [0.0f32; 16];
        a[0] = 1e8;
        a[1] = 1.0;
        a[2] = 1.0;
        a[3] = 1.0;
        a[4] = -1e8;
        a[5] = 1.0;
        a[6] = 1.0;
        a[7] = 1.0;
        let b = [1.0f32; 16];
        assert_eq!(dot_f32_ordered(&a, &b), 3.0, "pin the reference fold");
        let simd = simd_dot_f32(&a, &b, 16);
        assert_ne!(dot_f32_ordered(&a, &b), simd, "ordered == simd on the crafted pin input");
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn ordered_dot_panics_on_length_mismatch() {
        let _ = dot_f32_ordered(&[1.0, 2.0], &[1.0]);
    }

    #[test]
    fn ordered_dot_agrees_with_simd_on_generic_inputs_within_tolerance() {
        // The kernels answer the same mathematical question — they differ
        // only in last-ULP rounding. On pseudo-random well-scaled inputs the
        // relative gap stays tiny (last-ULP class), which is what lets the
        // SIMD kernel serve inference while the ordered fold serves
        // committed-value paths.
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut pseudo = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1 << 24) as f32 - 8.0
        };
        let a: Vec<f32> = (0..1024).map(|_| pseudo()).collect();
        let b: Vec<f32> = (0..1024).map(|_| pseudo()).collect();
        let ordered = dot_f32_ordered(&a, &b);
        let simd = simd_dot_f32(&a, &b, 1024);
        let rel = ((ordered - simd) / ordered).abs();
        assert!(rel < 1e-5, "relative gap {rel} exceeds last-ULP class");
    }
}
