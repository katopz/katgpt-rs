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

// ── Dot Product ───────────────────────────────────────────────

/// SIMD-accelerated dot product: `Σ a[i] * b[i]` for `len` elements.
///
/// Dispatches to NEON, AVX2, or scalar based on compile-time target and
/// runtime CPU feature detection.
#[inline(always)]
pub fn simd_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
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
                f32x4_mul(v128_load(a.as_ptr().add(i).cast()), v128_load(b.as_ptr().add(i).cast())),
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
                f32x4_mul(v128_load(a.as_ptr().add(i).cast()), v128_load(b.as_ptr().add(i).cast())),
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
            scalar_outer_product_acc(acc, a, b, m, n)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_acc(acc, a, b, m, n)
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
            scalar_outer_product_acc_scaled(acc, scale, a, b, m, n)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_outer_product_acc_scaled(acc, scale, a, b, m, n)
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
/// Used for HLA readout (qᵀ·SK, qᵀ·PKV, etc.).
/// `mat` is `[rows × cols]` row-major, `vec` is `[cols]`, `acc` is `[rows]`.
#[inline(always)]
pub fn simd_matvec(acc: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *acc.get_unchecked_mut(r) = simd_dot_f32(&mat[row_off..row_off + cols], vec, cols);
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

/// SIMD dot product: `Σ f16_weight[i] * f32_input[i]`.
///
/// Converts f16 weights to f32 on-the-fly during accumulation.
/// This is the hot-path for f16 weight inference — halves memory bandwidth
/// for weight reads while maintaining f32 precision for accumulation.
#[inline]
pub fn simd_dot_f16_f32(w_f16: &[half::f16], x_f32: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dot_f16_f32(w_f16, x_f32, len) }
    }
    #[cfg(not(target_arch = "aarch64"))]
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
        let chunks4 = len / 16;

        for _ in 0..chunks4 {
            // Convert 4×4 f16 → f32 via scalar conversion (compiles to hardware fcvt on Apple Silicon)
            // then vectorize the FMA. Each group of 4 f16 values becomes a NEON f32 vector.
            let w0 = [
                (*w.get_unchecked(i)).to_f32(),
                (*w.get_unchecked(i + 1)).to_f32(),
                (*w.get_unchecked(i + 2)).to_f32(),
                (*w.get_unchecked(i + 3)).to_f32(),
            ];
            let w1 = [
                (*w.get_unchecked(i + 4)).to_f32(),
                (*w.get_unchecked(i + 5)).to_f32(),
                (*w.get_unchecked(i + 6)).to_f32(),
                (*w.get_unchecked(i + 7)).to_f32(),
            ];
            let w2 = [
                (*w.get_unchecked(i + 8)).to_f32(),
                (*w.get_unchecked(i + 9)).to_f32(),
                (*w.get_unchecked(i + 10)).to_f32(),
                (*w.get_unchecked(i + 11)).to_f32(),
            ];
            let w3 = [
                (*w.get_unchecked(i + 12)).to_f32(),
                (*w.get_unchecked(i + 13)).to_f32(),
                (*w.get_unchecked(i + 14)).to_f32(),
                (*w.get_unchecked(i + 15)).to_f32(),
            ];
            let vw0 = vld1q_f32(w0.as_ptr());
            let vw1 = vld1q_f32(w1.as_ptr());
            let vw2 = vld1q_f32(w2.as_ptr());
            let vw3 = vld1q_f32(w3.as_ptr());

            acc0 = vfmaq_f32(acc0, vw0, vld1q_f32(x.as_ptr().add(i)));
            acc1 = vfmaq_f32(acc1, vw1, vld1q_f32(x.as_ptr().add(i + 4)));
            acc2 = vfmaq_f32(acc2, vw2, vld1q_f32(x.as_ptr().add(i + 8)));
            acc3 = vfmaq_f32(acc3, vw3, vld1q_f32(x.as_ptr().add(i + 12)));
            i += 16;
        }

        // Horizontal reduce: acc0+acc1+acc2+acc3
        let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        // Handle remaining elements with single accumulator
        let mut acc_rem = vdupq_n_f32(0.0);
        let chunks = (len - i) / 4;
        for _ in 0..chunks {
            let w32 = [
                (*w.get_unchecked(i)).to_f32(),
                (*w.get_unchecked(i + 1)).to_f32(),
                (*w.get_unchecked(i + 2)).to_f32(),
                (*w.get_unchecked(i + 3)).to_f32(),
            ];
            let vw = vld1q_f32(w32.as_ptr());
            let vx = vld1q_f32(x.as_ptr().add(i));
            acc_rem = vfmaq_f32(acc_rem, vw, vx);
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
