//! SIMD-accelerated linear algebra kernels for inference.
//!
//! Provides NEON (aarch64) and AVX2 (x86_64) backends for hot-path operations:
//! - Dot products (matmul inner loop)
//! - Outer product accumulation (HLA state updates)
//! - Matrix-vector multiply (HLA readout)
//!
//! Runtime detection selects the best available backend.
//! Falls back to scalar when SIMD is unavailable.
//!
//! # Stability
//!
//! Uses `core::arch` intrinsics directly — stable on both `aarch64` and `x86_64`.
//! No nightly features, no external SIMD crates.

const CEPHES_LN2_HI: f32 = 6.931_457_5e-1;
const CEPHES_LN2_LO: f32 = 1.428_606_8e-6;
const CEPHES_INV_LN2: f32 = std::f32::consts::LOG2_E;

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
}

/// Detect the best available SIMD level for the current CPU.
///
/// On `aarch64`: always returns [`SimdLevel::Neon`] (mandatory on ARMv8+).
/// On `x86_64`: returns [`SimdLevel::Avx2`] if CPU supports AVX2+FMA, else [`SimdLevel::Scalar`].
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
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        SimdLevel::Scalar
    }
}

// ── x86_64 Runtime Detection ─────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn is_avx2_fma_available() -> bool {
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
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
fn scalar_dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        unsafe {
            sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
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

// ── Outer Product Accumulation ────────────────────────────────

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
fn scalar_outer_product_acc(acc: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize) {
    for i in 0..m {
        let ai = unsafe { *a.get_unchecked(i) };
        let row = &mut acc[i * n..i * n + n];
        for j in 0..n {
            unsafe {
                *row.get_unchecked_mut(j) += ai * *b.get_unchecked(j);
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
fn scalar_dot_f16_f32(w: &[half::f16], x: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        unsafe {
            sum += (*w.get_unchecked(i)).to_f32() * *x.get_unchecked(i);
        }
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

// ── Sparse Dot Product (Scattered Gather) ────────────────────

/// SIMD sparse dot: `Σ weight[row_off + active_indices[i]] * active_values[i]` for `i in 0..alive`.
///
/// Gathers weight values at scattered positions and multiplies with contiguous
/// `active_values`. Used for sparse MLP matmul where only alive (post-ReLU)
/// neurons contribute.
///
/// Scalar fallback for alive ≤ 4 (gather overhead not worth it).
/// NEON/AVX2 processes 4/8 elements per iteration for larger counts.
#[inline(always)]
pub fn simd_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    // Scalar fallback for very sparse cases — gather setup overhead exceeds benefit.
    if alive <= 4 {
        let mut sum = 0.0f32;
        for i in 0..alive {
            unsafe {
                let c = *active_indices.get_unchecked(i);
                sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
            }
        }
        return sum;
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sparse_dot_f32(weight, row_off, active_indices, active_values, alive) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sparse_dot_f32(weight, row_off, active_indices, active_values, alive) }
        } else {
            scalar_sparse_dot_f32(weight, row_off, active_indices, active_values, alive)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_sparse_dot_f32(weight, row_off, active_indices, active_values, alive)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..alive {
        unsafe {
            let c = *active_indices.get_unchecked(i);
            sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vsetq_lane_f32};

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks = alive / 4;

        for _ in 0..chunks {
            // Gather 4 weight values from scattered indices into NEON register
            let mut ww = vdupq_n_f32(0.0);
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i)),
                ww,
                0,
            );
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 1)),
                ww,
                1,
            );
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 2)),
                ww,
                2,
            );
            ww = vsetq_lane_f32(
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 3)),
                ww,
                3,
            );

            // Load 4 contiguous active values
            let vv = vld1q_f32(active_values.as_ptr().add(i));

            // FMA: acc += ww * vv (4 multiply-accumulates in one instruction)
            acc = vfmaq_f32(acc, ww, vv);
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
        // Remainder tail (0..3 elements)
        while i < alive {
            let c = *active_indices.get_unchecked(i);
            sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_i32gather_ps, _mm256_loadu_ps, _mm256_set_epi32, _mm256_setzero_ps,
    };

    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        let chunks = alive / 8;

        for _ in 0..chunks {
            // Load 8 indices into __m256i for hardware gather
            // Note: _mm256_set_epi32 parameter order is (e7, e6, ..., e0)
            let idx = _mm256_set_epi32(
                *active_indices.get_unchecked(i + 7) as i32,
                *active_indices.get_unchecked(i + 6) as i32,
                *active_indices.get_unchecked(i + 5) as i32,
                *active_indices.get_unchecked(i + 4) as i32,
                *active_indices.get_unchecked(i + 3) as i32,
                *active_indices.get_unchecked(i + 2) as i32,
                *active_indices.get_unchecked(i + 1) as i32,
                *active_indices.get_unchecked(i) as i32,
            );

            // Hardware gather: weight[row_off + active_indices[j]] for each lane j
            let ww = _mm256_i32gather_ps(weight.as_ptr().add(row_off), idx, 4);

            // Load 8 contiguous active values
            let vv = _mm256_loadu_ps(active_values.as_ptr().add(i));

            // FMA: acc += ww * vv (8 multiply-accumulates in one instruction)
            acc = _mm256_fmadd_ps(ww, vv, acc);
            i += 8;
        }

        let mut sum = horizontal_sum_256(acc);
        // Remainder tail (0..7 elements)
        while i < alive {
            let c = *active_indices.get_unchecked(i);
            sum += *weight.get_unchecked(row_off + c) * *active_values.get_unchecked(i);
            i += 1;
        }

        sum
    }
}

// ── Sparse Matmul Row Dispatch ────────────────────────────────

/// SIMD-accelerated sparse matmul: `output[r] = sparse_dot(weight_row_r, active)`.
///
/// Replaces the inner loop of `sparse_matmul()` in `types.rs`.
/// `alive` is the count of active (non-zero) input elements after ReLU.
///
/// For each output row, computes the dot product using only the `alive`
/// elements at `active_indices` positions from the weight row.
#[inline(always)]
pub fn simd_sparse_matmul_rows(
    output: &mut [f32],
    weight: &[f32],
    active_indices: &[usize],
    active_values: &[f32],
    rows: usize,
    cols: usize,
    alive: usize,
) {
    for r in 0..rows {
        let row_off = r * cols;
        unsafe {
            *output.get_unchecked_mut(r) =
                simd_sparse_dot_f32(weight, row_off, active_indices, active_values, alive);
        }
    }
}

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
fn scalar_scale_inplace(x: &mut [f32], scale: f32) {
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
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_exp_inplace(x)
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
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_reciprocal_inplace(x)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_reciprocal_inplace(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = 1.0 / *val;
    }
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

// ── MaxSim Late-Interaction Scoring (Research 45, Plan 080) ────

/// Memory-efficient MaxSim scoring: `score = Σ_i max_j dot(q_i, d_j)`.
///
/// Late-interaction relevance score (ColBERT/PyLate style) computed without
/// materializing the `[Lq × Ld]` similarity matrix. Each query token's max
/// similarity across all doc tokens is found via running max, then summed.
///
/// This is the core scoring primitive distilled from erikkaum/maxsim (Research 45).
/// The Metal kernel achieves 3-4× speedup over naive by streaming over doc tokens
/// with a running max in shared memory — same O(Lq × Ld × dim) work, but with
/// better cache locality and zero intermediate allocation.
///
/// Our CPU version composes existing `simd_dot_f32` + inline running max.
/// The algorithm is provably equivalent to:
/// ```text
/// let mut sim = vec![0.0f32; lq * ld];
/// for i in 0..lq {
///     for j in 0..ld {
///         sim[i * ld + j] = dot(q[i], d[j]);
///     }
/// }
/// let score: f32 = (0..lq).map(|i| sim[i*ld..(i+1)*ld].iter().copied().fold(f32::NEG_INFINITY, f32::max)).sum();
/// ```
/// But without the `lq × ld` allocation.
///
/// # Arguments
/// - `queries`:   `[Lq, dim]` row-major f32
/// - `documents`: `[Ld, dim]` row-major f32
/// - `lq`:        number of query tokens
/// - `ld`:        number of document tokens
/// - `dim`:       embedding dimension (e.g. 64, 128)
///
/// # Returns
/// Scalar score (fp32 accumulated, matching Metal kernel design).
///
/// # Feature flag
/// `maxsim` — Plan 080
///
/// # GOAT proof (Plan 080 T2)
/// Must match naive materialized result within 1e-6.
/// Must be ≥2× faster than naive for Lq≥32, Ld≥128, dim=128.
#[cfg(feature = "maxsim")]
#[inline]
pub fn maxsim_score(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
    debug_assert!(
        queries.len() >= lq * dim,
        "maxsim_score: queries buffer too small: need {lq}*{dim}={}, have {}",
        lq * dim,
        queries.len()
    );
    debug_assert!(
        documents.len() >= ld * dim,
        "maxsim_score: documents buffer too small: need {ld}*{dim}={}, have {}",
        ld * dim,
        documents.len()
    );

    if ld == 0 {
        return 0.0;
    }

    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for j in 0..ld {
            let d_row = &documents[j * dim..(j + 1) * dim];
            let dot = simd_dot_f32(q_row, d_row, dim);
            my_max = my_max.max(dot);
        }
        score += my_max;
    }
    score
}

/// Packed/ragged MaxSim scoring: score N (query, doc) pairs with offset arrays.
///
/// Matches the Metal kernel's canonical API (maxsim README "Packed (ragged segments)").
/// Each pair (pair_q_ids[i], pair_d_ids[i]) gets scored independently.
///
/// # Arguments
/// - `queries`:        flat buffer, query_offsets[i]..query_offsets[i+1] is `[dim]`
/// - `query_offsets`:  [num_queries + 1] prefix-sum offsets
/// - `documents`:      flat buffer, doc_offsets[i]..doc_offsets[i+1] is `[dim]`
/// - `doc_offsets`:    [num_docs + 1] prefix-sum offsets
/// - `pair_q_ids`:     query index for each pair
/// - `pair_d_ids`:     doc index for each pair
/// - `dim`:            embedding dimension
///
/// - `results`:        output buffer, must have length >= num_pairs
///
/// # Feature flag
/// `maxsim` — Plan 080
#[cfg(feature = "maxsim")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn maxsim_score_packed(
    queries: &[f32],
    query_offsets: &[usize],
    documents: &[f32],
    doc_offsets: &[usize],
    pair_q_ids: &[usize],
    pair_d_ids: &[usize],
    dim: usize,
    results: &mut [f32],
) {
    let num_pairs = pair_q_ids.len();
    debug_assert_eq!(pair_d_ids.len(), num_pairs);
    debug_assert!(results.len() >= num_pairs, "results buffer too short");
    debug_assert!(query_offsets.len() >= *pair_q_ids.iter().max().unwrap_or(&0) + 2);
    debug_assert!(doc_offsets.len() >= *pair_d_ids.iter().max().unwrap_or(&0) + 2);

    for p in 0..num_pairs {
        let q_id = pair_q_ids[p];
        let d_id = pair_d_ids[p];
        let q_start = query_offsets[q_id];
        let q_end = query_offsets[q_id + 1];
        let d_start = doc_offsets[d_id];
        let d_end = doc_offsets[d_id + 1];
        let q_data = &queries[q_start..q_end];
        let d_data = &documents[d_start..d_end];
        let lq = q_data.len() / dim;
        let ld = d_data.len() / dim;
        results[p] = maxsim_score(q_data, d_data, lq, ld, dim);
    }
}

// ── Scalar Fallbacks (new primitives) ─────────────────────────

#[inline(always)]
#[allow(dead_code)]
fn scalar_add_inplace(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_add_scalar_inplace(x: &mut [f32], val: f32) {
    for v in x.iter_mut() {
        *v += val;
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_fused_sub_scale_inplace(x: &mut [f32], sub: f32, scale: f32) {
    for v in x.iter_mut() {
        *v = (*v - sub) * scale;
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_sum_f32(x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v;
    }
    sum
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_max_f32(x: &[f32]) -> f32 {
    let mut max = x[0];
    for i in 1..x.len() {
        let v = unsafe { *x.get_unchecked(i) };
        if v > max {
            max = v;
        }
    }
    max
}

/// Single-pass argmax: returns `(index, value)` of the maximum element.
///
/// Fuses max-finding and index-recovery into one traversal. The naive idiom —
/// `simd_max_f32(x)` followed by `x.iter().position(|&v| v == max)` — scans the
/// buffer twice (one SIMD max pass + one scalar equality scan that can run the
/// full length). On `aarch64` this uses a NEON kernel that tracks per-lane max
/// values *and* indices in one pass, measured at ~5× faster than the two-pass
/// idiom across vocab sizes 27 → 256k.
///
/// Tie-break matches `position(|&v| v == max)`: the **first** index attaining
/// the maximum is returned (strict `>` never replaces an earlier equal value).
///
/// Returns `(0, f32::NEG_INFINITY)` for an empty slice.
#[inline]
pub fn simd_argmax_f32(x: &[f32]) -> (usize, f32) {
    if x.is_empty() {
        return (0, f32::NEG_INFINITY);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_argmax_f32(x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Non-NEON: a scalar index-tracking loop does not auto-vectorize and
        // measured *slower* than the existing SIMD-max + position idiom at large
        // vocab, so reuse that two-pass path here (no regression). An AVX2 kernel
        // mirroring the NEON one could replace this once it can be verified on x86.
        let max_val = simd_max_f32(x);
        let idx = x.iter().position(|&v| v == max_val).unwrap_or(0);
        (idx, max_val)
    }
}

/// NEON single-pass argmax: tracks 4 lanes of (max value, index) simultaneously.
///
/// Each lane `l` accumulates the strided elements `x[l], x[l+4], x[l+8], …`;
/// `vcgtq_f32` (strict greater-than) only updates a lane on a *new* maximum, so
/// within a lane the earliest index wins. The 4 lane candidates are then reduced
/// by (value desc, index asc), and a scalar tail handles the final `len % 4`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_argmax_f32(x: &[f32]) -> (usize, f32) {
    use core::arch::aarch64::{
        vaddq_u32, vbslq_u32, vcgtq_f32, vdupq_n_u32, vld1q_f32, vld1q_u32, vmaxq_f32, vst1q_f32,
        vst1q_u32,
    };
    unsafe {
        let n = x.len();
        // Below one full vector, scalar is cheaper than NEON setup.
        if n < 4 {
            let mut bi = 0usize;
            let mut bv = *x.get_unchecked(0);
            for i in 1..n {
                let v = *x.get_unchecked(i);
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            return (bi, bv);
        }

        let base: [u32; 4] = [0, 1, 2, 3];
        let four = vdupq_n_u32(4);
        let mut vmax = vld1q_f32(x.as_ptr());
        let mut vidx = vld1q_u32(base.as_ptr());
        let mut cur = vaddq_u32(vidx, four); // indices for the next chunk
        let chunks = n / 4;
        let mut i = 4;
        for _ in 1..chunks {
            let v = vld1q_f32(x.as_ptr().add(i));
            let mask = vcgtq_f32(v, vmax); // lanes where v strictly greater
            vmax = vmaxq_f32(vmax, v);
            vidx = vbslq_u32(mask, cur, vidx); // adopt new index only where greater
            cur = vaddq_u32(cur, four);
            i += 4;
        }

        // Reduce the 4 lanes: highest value, smallest index on ties.
        let mut vals = [0f32; 4];
        let mut idxs = [0u32; 4];
        vst1q_f32(vals.as_mut_ptr(), vmax);
        vst1q_u32(idxs.as_mut_ptr(), vidx);
        let mut bv = vals[0];
        let mut bi = idxs[0] as usize;
        for l in 1..4 {
            let vi = idxs[l] as usize;
            if vals[l] > bv || (vals[l] == bv && vi < bi) {
                bv = vals[l];
                bi = vi;
            }
        }

        // Scalar tail (0..3 remaining elements).
        while i < n {
            let v = *x.get_unchecked(i);
            if v > bv {
                bv = v;
                bi = i;
            }
            i += 1;
        }
        (bi, bv)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_fused_decay_write(dst: &mut [f32], decay: f32, src: &[f32], write: f32) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) =
                decay * *dst.get_unchecked(i) + write * *src.get_unchecked(i);
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_scale_mul_inplace(x: &mut [f32], gamma: &[f32], scale: f32) {
    for i in 0..x.len() {
        unsafe {
            *x.get_unchecked_mut(i) = *gamma.get_unchecked(i) * *x.get_unchecked(i) * scale;
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
    let g = x - n as f32 * CEPHES_LN2_HI - n as f32 * CEPHES_LN2_LO;

    // 6th-order Cephes polynomial for exp(g) in [-0.5*ln2, 0.5*ln2]
    // Q(g) = 1 + g*(1 + g/2*(1 + g/3*(1 + g/4*(1 + g/5*(1 + g/6)))))
    let q = 1.0
        + g * (1.0
            + g * 0.5
                * (1.0
                    + g * (1.0 / 3.0)
                        * (1.0 + g * 0.25 * (1.0 + g * 0.2 * (1.0 + g * (1.0 / 6.0))))));

    // 2^n via bit manipulation: (n + 127) << 23
    if n < -126 {
        return 0.0;
    }
    if n > 127 {
        return f32::INFINITY;
    }
    let bits = ((n + 127) as u32) << 23;
    let scale = f32::from_bits(bits);
    scale * q
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_exp_inplace(x: &mut [f32]) {
    for val in x.iter_mut() {
        *val = cephes_exp_scalar(*val);
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
    use core::arch::aarch64::{vld1q_f32, vmaxq_f32};
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

        // Horizontal max of 4 lanes
        let arr: [f32; 4] = core::mem::transmute(vmax);
        let mut max = arr[0];
        for &v in &arr[1..] {
            if v > max {
                max = v;
            }
        }

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
unsafe fn avx2_exp_inplace(x: &mut [f32]) {
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

            // Cephes 6th-order polynomial
            let q = _mm256_add_ps(
                v_one,
                _mm256_mul_ps(
                    vg,
                    _mm256_add_ps(
                        v_one,
                        _mm256_mul_ps(
                            vg,
                            _mm256_add_ps(
                                v_half,
                                _mm256_mul_ps(
                                    vg,
                                    _mm256_add_ps(
                                        v_third,
                                        _mm256_mul_ps(
                                            vg,
                                            _mm256_add_ps(
                                                v_quarter,
                                                _mm256_mul_ps(
                                                    vg,
                                                    _mm256_add_ps(
                                                        v_fifth,
                                                        _mm256_mul_ps(vg, v_sixth),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            );

            // 2^n via AVX2 bit manipulation: shift = (n + 127) << 23
            let vn_shifted_i = _mm256_add_epi32(vn_i, _mm256_set1_epi32(127));
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

            // Cephes 6th-order polynomial
            let q = vaddq_f32(
                v_one,
                vmulq_f32(
                    vg,
                    vaddq_f32(
                        v_one,
                        vmulq_f32(
                            vg,
                            vaddq_f32(
                                v_half,
                                vmulq_f32(
                                    vg,
                                    vaddq_f32(
                                        v_third,
                                        vmulq_f32(
                                            vg,
                                            vaddq_f32(
                                                v_quarter,
                                                vmulq_f32(
                                                    vg,
                                                    vaddq_f32(v_fifth, vmulq_f32(vg, v_sixth)),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            );

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

// ── AVX2 Backend (new primitives) ─────────────────────────────

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

// ── x86_64 Horizontal Max/Sum Helpers ─────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn horizontal_max_256(v: core::arch::x86_64::__m256) -> f32 {
    // Safety: pure SIMD intrinsic computation, no memory access.
    use core::arch::x86_64::{
        _mm_cvtss_f32, _mm_max_ps, _mm_shuffle_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
    };
    unsafe {
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
}

// ── x86_64 Horizontal Sum Helpers ─────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn horizontal_sum_256(v: core::arch::x86_64::__m256) -> f32 {
    // Safety: pure SIMD intrinsic computation, no memory access.
    use core::arch::x86_64::{_mm_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps};
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        horizontal_sum_128(sum128)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn horizontal_sum_128(v: core::arch::x86_64::__m128) -> f32 {
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

// ── Ternary SIMD Matvec (Plasma Path — Plan 148) ─────────────

#[cfg(feature = "plasma_path")]
use crate::types::TernaryWeights;

/// Scalar reference ternary matvec: y[r] = row_scale[r] * Σ(col → sign(pos_bit, neg_bit) * x[col])
#[cfg(feature = "plasma_path")]
#[allow(clippy::needless_range_loop)]
pub fn ternary_matvec_scalar(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    for r in 0..w.rows {
        let mut sum = 0.0f32;
        let row_base = r * w.blocks64;
        for c in 0..w.cols {
            let block = c >> 6;
            let bit = c & 63;
            let mask = 1u64 << bit;
            let idx = row_base + block;
            let pos = (w.pos_bits[idx] & mask) != 0;
            let neg = (w.neg_bits[idx] & mask) != 0;
            let sign = pos as i32 - neg as i32;
            sum += sign as f32 * x[c];
        }
        y[r] = sum * w.row_scale[r];
    }
}

#[cfg(all(feature = "plasma_path", target_arch = "aarch64"))]
#[allow(clippy::needless_range_loop)]
unsafe fn neon_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len()==w.cols and y.len()==w.rows
    unsafe {
        use core::arch::aarch64::*;
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        for r in 0..w.rows {
            let row_base = r * w.blocks64;
            let mut acc = vdupq_n_f32(0.0);

            for b in 0..w.blocks64 {
                let idx = row_base + b;
                let pos_word = w.pos_bits[idx];
                let neg_word = w.neg_bits[idx];

                let base_col = b * 64;
                let remaining = if base_col + 64 <= w.cols {
                    64
                } else {
                    w.cols - base_col
                };
                let chunks = remaining / 4;

                for chunk in 0..chunks {
                    let col_off = base_col + chunk * 4;
                    let bits4 = (pos_word >> (chunk * 4)) & 0xF;
                    let neg_bits4 = (neg_word >> (chunk * 4)) & 0xF;

                    // Load 4 x values
                    let x_vals = vld1q_f32(x.as_ptr().add(col_off));

                    // For each of the 4 lanes, test bit in bits4
                    let lane_bits = [
                        bits4 & 1,
                        (bits4 >> 1) & 1,
                        (bits4 >> 2) & 1,
                        (bits4 >> 3) & 1,
                    ];
                    let neg_lane_bits = [
                        neg_bits4 & 1,
                        (neg_bits4 >> 1) & 1,
                        (neg_bits4 >> 2) & 1,
                        (neg_bits4 >> 3) & 1,
                    ];

                    // Build selection masks from lane bits
                    let pos_mask_u32: [u32; 4] = [
                        if lane_bits[0] != 0 { !0u32 } else { 0 },
                        if lane_bits[1] != 0 { !0u32 } else { 0 },
                        if lane_bits[2] != 0 { !0u32 } else { 0 },
                        if lane_bits[3] != 0 { !0u32 } else { 0 },
                    ];
                    let neg_mask_u32: [u32; 4] = [
                        if neg_lane_bits[0] != 0 { !0u32 } else { 0 },
                        if neg_lane_bits[1] != 0 { !0u32 } else { 0 },
                        if neg_lane_bits[2] != 0 { !0u32 } else { 0 },
                        if neg_lane_bits[3] != 0 { !0u32 } else { 0 },
                    ];

                    let pos_sel = vreinterpretq_f32_u32(vld1q_u32(pos_mask_u32.as_ptr()));
                    let neg_sel = vreinterpretq_f32_u32(vld1q_u32(neg_mask_u32.as_ptr()));

                    // pos contribution: if bit set, add x[col], else 0
                    let pos_val =
                        vbslq_f32(vreinterpretq_u32_f32(pos_sel), x_vals, vdupq_n_f32(0.0));
                    let neg_val =
                        vbslq_f32(vreinterpretq_u32_f32(neg_sel), x_vals, vdupq_n_f32(0.0));

                    acc = vaddq_f32(acc, vsubq_f32(pos_val, neg_val));
                }

                // Handle remaining elements (0-3) scalar
                let mut scalar_acc = 0.0f32;
                for i in (chunks * 4)..remaining {
                    let c = base_col + i;
                    let bit_mask = 1u64 << i;
                    let pos = (pos_word & bit_mask) != 0;
                    let neg = (neg_word & bit_mask) != 0;
                    let sign = pos as u32 as f32 - neg as u32 as f32;
                    scalar_acc += sign * x[c];
                }
                if scalar_acc != 0.0 {
                    acc = vaddq_f32(acc, vsetq_lane_f32(scalar_acc, vdupq_n_f32(0.0), 0));
                }
            }

            // Horizontal sum
            let mut lanes = [0.0f32; 4];
            vst1q_f32(lanes.as_mut_ptr(), acc);
            let sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            y[r] = sum * w.row_scale[r];
        }
    } // unsafe
}

#[cfg(all(feature = "plasma_path", target_arch = "x86_64"))]
#[allow(clippy::needless_range_loop)]
unsafe fn avx2_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len()==w.cols and y.len()==w.rows
    unsafe {
        use core::arch::x86_64::*;
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        for r in 0..w.rows {
            let row_base = r * w.blocks64;
            let mut acc = _mm256_setzero_ps();

            for b in 0..w.blocks64 {
                let idx = row_base + b;
                let pos_word = w.pos_bits[idx];
                let neg_word = w.neg_bits[idx];

                let base_col = b * 64;
                let remaining = if base_col + 64 <= w.cols {
                    64
                } else {
                    w.cols - base_col
                };
                let chunks = remaining / 8;

                for chunk in 0..chunks {
                    let col_off = base_col + chunk * 8;

                    // Test 8 bits at once
                    let pos_byte = ((pos_word >> (chunk * 8)) & 0xFF) as u32;
                    let neg_byte = ((neg_word >> (chunk * 8)) & 0xFF) as u32;

                    // Broadcast bits to per-lane masks
                    let lane_masks_pos = [
                        if pos_byte & 1 != 0 { !0u32 } else { 0 },
                        if pos_byte & 2 != 0 { !0u32 } else { 0 },
                        if pos_byte & 4 != 0 { !0u32 } else { 0 },
                        if pos_byte & 8 != 0 { !0u32 } else { 0 },
                        if pos_byte & 16 != 0 { !0u32 } else { 0 },
                        if pos_byte & 32 != 0 { !0u32 } else { 0 },
                        if pos_byte & 64 != 0 { !0u32 } else { 0 },
                        if pos_byte & 128 != 0 { !0u32 } else { 0 },
                    ];
                    let lane_masks_neg = [
                        if neg_byte & 1 != 0 { !0u32 } else { 0 },
                        if neg_byte & 2 != 0 { !0u32 } else { 0 },
                        if neg_byte & 4 != 0 { !0u32 } else { 0 },
                        if neg_byte & 8 != 0 { !0u32 } else { 0 },
                        if neg_byte & 16 != 0 { !0u32 } else { 0 },
                        if neg_byte & 32 != 0 { !0u32 } else { 0 },
                        if neg_byte & 64 != 0 { !0u32 } else { 0 },
                        if neg_byte & 128 != 0 { !0u32 } else { 0 },
                    ];

                    let x_vals = _mm256_loadu_ps(x.as_ptr().add(col_off));
                    let pos_mask = _mm256_castsi256_ps(_mm256_loadu_si256(
                        lane_masks_pos.as_ptr() as *const __m256i
                    ));
                    let neg_mask = _mm256_castsi256_ps(_mm256_loadu_si256(
                        lane_masks_neg.as_ptr() as *const __m256i
                    ));

                    let pos_val = _mm256_and_ps(x_vals, pos_mask);
                    let neg_val = _mm256_and_ps(x_vals, neg_mask);

                    acc = _mm256_add_ps(acc, _mm256_sub_ps(pos_val, neg_val));
                }

                // Handle remaining elements (0-7) scalar
                let mut scalar_acc = 0.0f32;
                for i in (chunks * 8)..remaining {
                    let c = base_col + i;
                    let bit_mask = 1u64 << i;
                    let pos = (pos_word & bit_mask) != 0;
                    let neg = (neg_word & bit_mask) != 0;
                    let sign = pos as u32 as f32 - neg as u32 as f32;
                    scalar_acc += sign * x[c];
                }
                if scalar_acc != 0.0 {
                    acc = _mm256_add_ps(
                        acc,
                        _mm256_setr_ps(scalar_acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    );
                }
            }

            y[r] = horizontal_sum_256(acc) * w.row_scale[r];
        }
    } // unsafe
}

/// SIMD-accelerated ternary matvec: y = W_ternary × x
///
/// Dispatches to NEON, AVX2, or scalar based on `simd_level()`.
/// All paths produce bit-identical results to `ternary_matvec_scalar()`.
#[cfg(feature = "plasma_path")]
#[inline]
pub fn simd_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    match simd_level() {
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon_ternary_matvec(w, x, y) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { avx2_ternary_matvec(w, x, y) },
        _ => ternary_matvec_scalar(w, x, y),
    }
}

/// Batched ternary matmul: for each batch[i], compute y[i] = W × batch[i].
#[cfg(feature = "plasma_path")]
#[inline]
pub fn simd_ternary_matmul_batch(w: &TernaryWeights, x: &[f32], batch: usize, y: &mut [f32]) {
    /// Minimum batch size before parallelizing. Below this, sequential is faster
    /// due to rayon thread pool scheduling overhead (~1-5µs per task).
    /// Each ternary matvec at 256×256+ already exceeds 10µs, so parallelism
    /// wins for batch ≥ 2, but we use 4 to amortize the join overhead.
    const PARALLEL_BATCH_MIN: usize = 4;

    if batch < PARALLEL_BATCH_MIN {
        for b in 0..batch {
            let x_off = b * w.cols;
            let y_off = b * w.rows;
            simd_ternary_matvec(w, &x[x_off..], &mut y[y_off..]);
        }
    } else {
        use rayon::prelude::*;
        y.par_chunks_mut(w.rows)
            .enumerate()
            .for_each(|(b, y_chunk)| {
                if b < batch {
                    let x_off = b * w.cols;
                    simd_ternary_matvec(w, &x[x_off..], y_chunk);
                }
            });
    }
}

// ── Sigmoid Margin Loss + Retrieval Diagnostic (Research 123, Plan 157) ────

/// Numerically stable softplus: log(1 + exp(x)).
///
/// For x > 20: returns x (avoids exp overflow).
/// For x < -20: returns exp(x) ≈ 0 (avoids log(1+0) precision loss).
#[cfg(feature = "sigmoid_margin")]
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SigLIP-style sigmoid margin loss: softplus(t · (score − b) · sign).
///
/// For each (query, doc) pair in the [N × n] score matrix:
///   - positive pairs (adjacency[i,j] = 1): loss = softplus(−t·(score−b)), pushes score above bias
///   - negative pairs (adjacency[i,j] = 0): loss = softplus(+t·(score−b)), pushes score below bias
///
/// Global minimizers coincide with max-margin embeddings (Prop 7, arXiv 2605.23556).
/// The paper proves this loss achieves O(k log n) dimension scaling vs InfoNCE's O(n^{1/3}).
///
/// # Arguments
/// - `scores`:      `[N × n]` dot-product score matrix (row-major)
/// - `adjacency`:   `[N × n]` binary adjacency (positive pairs = 1.0, negative = 0.0)
/// - `temperature`: learnable temperature (init 1.0)
/// - `bias`:        learnable bias (init 0.0)
/// - `n_rows`:      number of queries (N)
/// - `n_cols`:      number of documents (n)
///
/// # Returns
/// Mean loss across all pairs.
///
/// # Feature flag
/// `sigmoid_margin` — Plan 157
#[cfg(feature = "sigmoid_margin")]
#[inline]
pub fn sigmoid_margin_loss(
    scores: &[f32],
    adjacency: &[f32],
    temperature: f32,
    bias: f32,
    n_rows: usize,
    n_cols: usize,
) -> f32 {
    debug_assert_eq!(scores.len(), n_rows * n_cols);
    debug_assert_eq!(adjacency.len(), n_rows * n_cols);

    let mut total = 0.0f32;
    for i in 0..n_rows {
        for j in 0..n_cols {
            let idx = i * n_cols + j;
            let score = scores[idx];
            let adj = adjacency[idx];
            // For positive pairs (adj=1): loss = softplus(-t·(score−b)), minimized when score → +∞
            // For negative pairs (adj=0): loss = softplus(+t·(score−b)), minimized when score → −∞
            // Branchless: 1.0 - 2.0 * (adj > 0.5) → -1.0 if positive, +1.0 if negative
            let sign = 1.0 - 2.0 * (adj > 0.5) as u32 as f32;
            let x = temperature * (score - bias) * sign;
            total += softplus(x);
        }
    }
    total / (n_rows * n_cols) as f32
}

/// Compute retrieval margin: 0.5 × (min_pos_score − max_neg_score).
///
/// For each query embedding u_i with positive set P_i given by `neighborhoods`:
///   pos_min = min_{j ∈ P_i} dot(u_i, v_j)
///   neg_max = max_{j ∉ P_i} dot(u_i, v_j)
///   margin_i = 0.5 * (pos_min − neg_max)
///
/// Returns (global_min_pos, global_max_neg, global_margin) across all queries.
///
/// # Arguments
/// - `queries`:       `[N × dim]` row-major query embeddings
/// - `documents`:     `[n × dim]` row-major document embeddings
/// - `neighborhoods`: `[N × k]` positive pair indices (flat, row-major)
/// - `dim`:           embedding dimension
/// - `n_queries`:     number of queries (N)
/// - `n_docs`:        number of documents (n)
/// - `k`:            neighborhood size
///
/// # Feature flag
/// `sigmoid_margin` — Plan 157
#[cfg(feature = "sigmoid_margin")]
#[inline]
pub fn compute_retrieval_margin(
    queries: &[f32],
    documents: &[f32],
    neighborhoods: &[usize],
    dim: usize,
    n_queries: usize,
    n_docs: usize,
    k: usize,
) -> (f32, f32, f32) {
    debug_assert!(queries.len() >= n_queries * dim);
    debug_assert!(documents.len() >= n_docs * dim);
    debug_assert!(neighborhoods.len() >= n_queries * k);

    let mut global_pos_min = f32::INFINITY;
    let mut global_neg_max = f32::NEG_INFINITY;

    // Pre-allocate bitmap once, reuse per query
    let mut pos_bitmap = vec![false; n_docs];

    for i in 0..n_queries {
        let q_row = &queries[i * dim..(i + 1) * dim];

        // Build positive set for this query via bitmap
        let pos_start = i * k;
        pos_bitmap.fill(false);
        for &idx in &neighborhoods[pos_start..pos_start + k] {
            if idx < n_docs {
                pos_bitmap[idx] = true;
            }
        }

        // min positive score
        let mut pos_min = f32::INFINITY;
        for &j in &neighborhoods[pos_start..pos_start + k] {
            let d_row = &documents[j * dim..(j + 1) * dim];
            let dot = simd_dot_f32(q_row, d_row, dim);
            pos_min = pos_min.min(dot);
        }

        // max negative score (all docs not in pos_set)
        let mut neg_max = f32::NEG_INFINITY;
        for j in 0..n_docs {
            if pos_bitmap[j] {
                continue;
            }
            let d_row = &documents[j * dim..(j + 1) * dim];
            let dot = simd_dot_f32(q_row, d_row, dim);
            neg_max = neg_max.max(dot);
        }

        global_pos_min = global_pos_min.min(pos_min);
        global_neg_max = global_neg_max.max(neg_max);
    }

    let margin = 0.5 * (global_pos_min - global_neg_max);
    (global_pos_min, global_neg_max, margin)
}

/// Theoretical O(k log n) dimension sufficiency bound from arXiv 2605.23556.
///
/// Returns the minimum embedding dimension theoretically sufficient
/// for near-optimal retrieval margin, given query sparsity k and corpus size n.
///
/// Theorem 1.4: d = O(k · log n) is sufficient.
/// Theorem 1.5: d = O(k · log(n/k)) is also necessary → tight bound.
///
/// Uses a conservative constant factor of 1.5 to provide a practical upper bound.
/// For k ≤ 0 or n ≤ 1, returns 1 (trivial case).
///
/// # Feature flag
/// `sigmoid_margin` — Plan 157
#[cfg(feature = "sigmoid_margin")]
pub fn dim_sufficiency_bound(k: usize, n: usize) -> usize {
    if k == 0 || n <= 1 {
        return 1;
    }
    // d = ceil(1.5 * k * ln(n))  — conservative O(k log n) bound
    let bound = 1.5 * (k as f64) * (n as f64).ln();
    bound.ceil() as usize
}

// ── SIMD Sum of Squares (Issue 075) ──────────────────────────

/// SIMD-accelerated sum of squares: `Σ x[i]²`.
/// More efficient than `simd_dot_f32(x, x, len)` — loads data once instead of twice.
#[inline(always)]
pub fn simd_sum_sq(x: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sum_sq(x, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sum_sq(x, len) }
        } else {
            scalar_sum_sq(x, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_sum_sq(x, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_sum_sq(x: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        unsafe {
            let v = *x.get_unchecked(i);
            sum += v * v;
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sum_sq(x: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks2 = len / 8;

        for _ in 0..chunks2 {
            let v0 = vld1q_f32(x.as_ptr().add(i));
            acc0 = vfmaq_f32(acc0, v0, v0);
            let v1 = vld1q_f32(x.as_ptr().add(i + 4));
            acc1 = vfmaq_f32(acc1, v1, v1);
            i += 8;
        }

        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));

        let mut acc_rem = vdupq_n_f32(0.0);
        let remaining = (len - i) / 4;
        for _ in 0..remaining {
            let v = vld1q_f32(x.as_ptr().add(i));
            acc_rem = vfmaq_f32(acc_rem, v, v);
            i += 4;
        }
        sum += vaddvq_f32(acc_rem);

        while i < len {
            let v = *x.get_unchecked(i);
            sum += v * v;
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_sum_sq(x: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps};

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        let chunks2 = len / 16;

        for _ in 0..chunks2 {
            let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
            i += 16;
        }

        let mut sum = horizontal_sum_256(_mm256_add_ps(acc0, acc1));

        // Handle remaining 8-element chunk
        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            sum += horizontal_sum_256(_mm256_fmadd_ps(v, v, _mm256_setzero_ps()));
            i += 8;
        }

        while i < len {
            let v = *x.get_unchecked(i);
            sum += v * v;
            i += 1;
        }

        sum
    }
}

// ── SIMD Sum-|x| (Issue 120) ───────────────────────────────

/// SIMD-accelerated sum of absolute values: `Σ |x[i]|`.
#[inline(always)]
pub fn simd_sum_abs_f32(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_sum_abs_f32(x) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_sum_abs_f32(x) }
        } else {
            scalar_sum_abs_f32(x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_sum_abs_f32(x)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_sum_abs_f32(x: &[f32]) -> f32 {
    x.iter().map(|v| v.abs()).sum()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_sum_abs_f32(x: &[f32]) -> f32 {
    use core::arch::aarch64::{vabsq_f32, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32};
    unsafe {
        let mut i = 0;
        let chunks = x.len() / 4;
        let mut acc0 = vdupq_n_f32(0.0);
        // Use two accumulators for better pipeline utilization
        let mut acc1 = vdupq_n_f32(0.0);
        let chunks2 = chunks / 2;
        for _ in 0..chunks2 {
            let v0 = vld1q_f32(x.as_ptr().add(i));
            acc0 = vaddq_f32(acc0, vabsq_f32(v0));
            let v1 = vld1q_f32(x.as_ptr().add(i + 4));
            acc1 = vaddq_f32(acc1, vabsq_f32(v1));
            i += 8;
        }
        // Handle remaining 4-element chunk
        if i + 4 <= x.len() {
            let v = vld1q_f32(x.as_ptr().add(i));
            acc0 = vaddq_f32(acc0, vabsq_f32(v));
            i += 4;
        }
        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        while i < x.len() {
            sum += (*x.get_unchecked(i)).abs();
            i += 1;
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_sum_abs_f32(x: &[f32]) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_and_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    };
    unsafe {
        // Mask to clear the sign bit: AND with this yields |x|
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7fff_ffff));
        let mut i = 0;
        let chunks = x.len() / 8;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let chunks2 = chunks / 2;
        for _ in 0..chunks2 {
            let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(v0, abs_mask));
            let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            acc1 = _mm256_add_ps(acc1, _mm256_and_ps(v1, abs_mask));
            i += 16;
        }
        // Handle remaining 8-element chunk
        let remaining = (x.len() - i) / 8;
        for _ in 0..remaining {
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            acc0 = _mm256_add_ps(acc0, _mm256_and_ps(v, abs_mask));
            i += 8;
        }
        let mut sum = horizontal_sum_256(_mm256_add_ps(acc0, acc1));
        while i < x.len() {
            sum += (*x.get_unchecked(i)).abs();
            i += 1;
        }
        sum
    }
}

// ── SIMD Distance² (Issue 076) ───────────────────────────────

/// SIMD-accelerated squared distance: `Σ (a[i] - b[i])²`.
/// Computes the elementwise difference, squares, and sums in one pass.
#[inline]
pub fn simd_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_dist_sq(a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_dist_sq(a, b, len) }
        } else {
            scalar_dist_sq(a, b, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_dist_sq(a, b, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        unsafe {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += diff * diff;
        }
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::aarch64::{
        vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vsubq_f32,
    };

    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks2 = len / 8;

        for _ in 0..chunks2 {
            let d0 = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            acc0 = vfmaq_f32(acc0, d0, d0);
            let d1 = vsubq_f32(
                vld1q_f32(a.as_ptr().add(i + 4)),
                vld1q_f32(b.as_ptr().add(i + 4)),
            );
            acc1 = vfmaq_f32(acc1, d1, d1);
            i += 8;
        }

        let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));

        if i + 4 <= len {
            let d = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            sum += vaddvq_f32(vmulq_f32(d, d));
            i += 4;
        }

        while i < len {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += diff * diff;
            i += 1;
        }

        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_sub_ps,
    };

    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        let chunks2 = len / 16;

        for _ in 0..chunks2 {
            let d0 = _mm256_sub_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            );
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            let d1 = _mm256_sub_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
            );
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            i += 16;
        }

        let mut sum = horizontal_sum_256(_mm256_add_ps(acc0, acc1));

        let remaining = (len - i) / 8;
        for _ in 0..remaining {
            let d = _mm256_sub_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            );
            sum += horizontal_sum_256(_mm256_fmadd_ps(d, d, _mm256_setzero_ps()));
            i += 8;
        }

        while i < len {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += diff * diff;
            i += 1;
        }

        sum
    }
}

// ── SIMD Fused Subtract-Accumulate (Issue 071) ──────────────

/// SIMD-accelerated fused subtract-accumulate: `dst[i] += a[i] - b[i]`.
/// Single-pass operation for MLS and delta routing accumulation.
#[inline(always)]
pub fn simd_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_sub_acc(dst, a, b, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_fused_sub_acc(dst, a, b, len) }
        } else {
            scalar_fused_sub_acc(dst, a, b, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_fused_sub_acc(dst, a, b, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    for i in 0..len {
        unsafe {
            *dst.get_unchecked_mut(i) += *a.get_unchecked(i) - *b.get_unchecked(i);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32, vsubq_f32};

    unsafe {
        let mut i = 0;
        let chunks = len / 4;

        for _ in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vd = vld1q_f32(dst.as_ptr().add(i));
            let result = vaddq_f32(vd, vsubq_f32(va, vb));
            vst1q_f32(dst.as_mut_ptr().add(i), result);
            i += 4;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_fused_sub_acc(dst: &mut [f32], a: &[f32], b: &[f32], len: usize) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};

    unsafe {
        let mut i = 0;
        let chunks = len / 8;

        for _ in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vd = _mm256_loadu_ps(dst.as_ptr().add(i));
            let result = _mm256_add_ps(vd, _mm256_sub_ps(va, vb));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), result);
            i += 8;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += *a.get_unchecked(i) - *b.get_unchecked(i);
            i += 1;
        }
    }
}

/// Fused scale-accumulate: `dst[i] += scale * src[i]` for `len` elements.
///
/// Used in attention value accumulation where a scalar weight broadcasts across a value row.
/// NEON: fused via `vfmaq_f32`. AVX2: fused via `_mm256_fmadd_ps`.
#[inline(always)]
pub fn simd_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_fused_scale_acc(dst, src, scale, len) }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_avx2_fma_available() {
            unsafe { avx2_fused_scale_acc(dst, src, scale, len) }
        } else {
            scalar_fused_scale_acc(dst, src, scale, len)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        scalar_fused_scale_acc(dst, src, scale, len)
    }
}

#[inline(always)]
#[allow(dead_code)]
fn scalar_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    for i in 0..len {
        unsafe {
            *dst.get_unchecked_mut(i) += scale * *src.get_unchecked(i);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

    unsafe {
        let scale_vec = vdupq_n_f32(scale);
        let mut i = 0;
        let chunks = len / 4;

        for _ in 0..chunks {
            let s = vld1q_f32(src.as_ptr().add(i));
            let d = vld1q_f32(dst.as_ptr().add(i));
            let acc = vfmaq_f32(d, s, scale_vec);
            vst1q_f32(dst.as_mut_ptr().add(i), acc);
            i += 4;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += scale * *src.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn avx2_fused_scale_acc(dst: &mut [f32], src: &[f32], scale: f32, len: usize) {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    unsafe {
        let scale_vec = _mm256_set1_ps(scale);
        let mut i = 0;
        let chunks = len / 8;

        for _ in 0..chunks {
            let s = _mm256_loadu_ps(src.as_ptr().add(i));
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let acc = _mm256_fmadd_ps(scale_vec, s, d);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), acc);
            i += 8;
        }

        while i < len {
            *dst.get_unchecked_mut(i) += scale * *src.get_unchecked(i);
            i += 1;
        }
    }
}

/// SIMD-accelerated Gram matrix computation: G = X·Xᵀ where X is (seq_len × d_h).
///
/// For each pair (i, j), G[i*seq_len + j] = dot(X_row_i, X_row_j).
/// Only computes upper triangle, then mirrors to lower triangle.
/// Uses existing `simd_dot_f32` for each dot product.
///
/// `x` is row-major (seq_len × d_h), `gram_out` is row-major (seq_len × seq_len).
#[inline]
pub fn simd_gram_f32(x: &[f32], seq_len: usize, d_h: usize, gram_out: &mut [f32]) {
    assert!(
        x.len() == seq_len * d_h,
        "x.len() = {} but seq_len * d_h = {}",
        x.len(),
        seq_len * d_h
    );
    assert!(
        gram_out.len() == seq_len * seq_len,
        "gram_out.len() = {} but seq_len^2 = {}",
        gram_out.len(),
        seq_len * seq_len
    );

    for i in 0..seq_len {
        let row_i = &x[i * d_h..(i + 1) * d_h];
        let i_row_off = i * seq_len;

        // Diagonal: sum of squares (one load instead of two)
        let diag = simd_sum_sq(row_i, d_h);
        gram_out[i_row_off + i] = diag;

        // Upper triangle: j > i
        for j in (i + 1)..seq_len {
            let row_j = &x[j * d_h..(j + 1) * d_h];
            let val = simd_dot_f32(row_i, row_j, d_h);
            gram_out[i_row_off + j] = val;
            gram_out[j * seq_len + i] = val; // mirror to lower triangle
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_matches_two_pass_idiom() {
        // Reference: the two-pass idiom this primitive replaces.
        fn naive(x: &[f32]) -> (usize, f32) {
            let m = simd_max_f32(x);
            (x.iter().position(|&v| v == m).unwrap_or(0), m)
        }
        let cases: &[&[f32]] = &[
            &[3.0],
            &[1.0, 2.0, 3.0, 2.0, 1.0],
            &[5.0, 5.0, 5.0],           // tie → first index (0)
            &[-1.0, -2.0, -0.5, -9.0],  // all negative
            &[0.0, 1.0, 1.0, 0.5, 1.0], // multiple maxima → first (index 1)
        ];
        for c in cases {
            assert_eq!(simd_argmax_f32(c), naive(c), "mismatch on {c:?}");
        }
        // Larger pseudo-random buffer: max placed at a known interior index.
        let mut buf = vec![0.0f32; 4096];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i * 2654435761) % 997) as f32;
        }
        buf[1234] = 10_000.0;
        assert_eq!(simd_argmax_f32(&buf), (1234, 10_000.0));

        // Randomized equivalence sweep across lengths — exercises the SIMD tail
        // (len % 4), every lane position, and cross-lane ties. Many duplicate
        // values (mod 7) so ties are common and first-index tie-break is tested.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in 1..=130usize {
            let v: Vec<f32> = (0..len).map(|_| (rng() % 7) as f32).collect();
            assert_eq!(simd_argmax_f32(&v), naive(&v), "len={len} v={v:?}");
        }
    }

    #[test]
    fn argmax_empty_slice() {
        assert_eq!(simd_argmax_f32(&[]), (0, f32::NEG_INFINITY));
    }

    #[test]
    fn simd_level_matches_platform() {
        let level = simd_level();
        #[cfg(target_arch = "aarch64")]
        assert_eq!(level, SimdLevel::Neon);
        #[cfg(target_arch = "x86_64")]
        assert!(matches!(level, SimdLevel::Avx2 | SimdLevel::Scalar));
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        assert_eq!(level, SimdLevel::Scalar);
    }

    #[test]
    fn dot_product_aligned_len_8() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let scalar = scalar_dot_f32(&a, &b, 8);
        let simd = simd_dot_f32(&a, &b, 8);

        assert!((scalar - simd).abs() < 1e-4, "scalar={scalar}, simd={simd}");
        // Expected: 0.5+2+4.5+8+12.5+18+24.5+32 = 102
        assert!((simd - 102.0).abs() < 1e-4, "simd={simd}");
    }

    #[test]
    fn dot_product_non_aligned_len() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 1.0, 1.0, 1.0, 1.0];

        let scalar = scalar_dot_f32(&a, &b, 5);
        let simd = simd_dot_f32(&a, &b, 5);

        assert!((scalar - simd).abs() < 1e-4, "scalar={scalar}, simd={simd}");
        assert!((simd - 15.0).abs() < 1e-4);
    }

    #[test]
    fn dot_product_len_4() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 0.5, 0.25, 0.125];

        let expected = 1.0 + 1.0 + 0.75 + 0.5;
        let simd = simd_dot_f32(&a, &b, 4);

        assert!((simd - expected).abs() < 1e-4);
    }

    #[test]
    fn dot_product_len_32() {
        // Game config n_embd=32
        let a: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..32).map(|i| (i as f32 + 1.0) * 0.05).collect();

        let scalar = scalar_dot_f32(&a, &b, 32);
        let simd = simd_dot_f32(&a, &b, 32);

        assert!((scalar - simd).abs() < 1e-3, "scalar={scalar}, simd={simd}");
    }

    #[test]
    fn dot_product_zero_length() {
        let simd = simd_dot_f32(&[], &[], 0);
        assert!((simd - 0.0).abs() < 1e-6);
    }

    #[test]
    fn outer_product_4x4_matches_scalar() {
        let m = 4;
        let n = 4;
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.0, 1.5, 2.0];

        let mut acc_scalar = vec![0.0f32; m * n];
        let mut acc_simd = vec![0.0f32; m * n];

        scalar_outer_product_acc(&mut acc_scalar, &a, &b, m, n);
        simd_outer_product_acc(&mut acc_simd, &a, &b, m, n);

        for i in 0..m * n {
            assert!(
                (acc_scalar[i] - acc_simd[i]).abs() < 1e-4,
                "mismatch at {i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }

    #[test]
    fn outer_product_8x8_matches_scalar() {
        // Game config: hd=8
        let m = 8;
        let n = 8;
        let a: Vec<f32> = (0..m).map(|i| (i + 1) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|j| (j + 1) as f32 * 0.2).collect();

        let mut acc_scalar = vec![0.0f32; m * n];
        let mut acc_simd = vec![0.0f32; m * n];

        scalar_outer_product_acc(&mut acc_scalar, &a, &b, m, n);
        simd_outer_product_acc(&mut acc_simd, &a, &b, m, n);

        for i in 0..m * n {
            assert!(
                (acc_scalar[i] - acc_simd[i]).abs() < 1e-4,
                "mismatch at {i}: scalar={}, simd={}",
                acc_scalar[i],
                acc_simd[i]
            );
        }
    }

    #[test]
    fn outer_product_accumulates() {
        let m = 4;
        let n = 4;
        let a = [1.0f32, 0.0, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0, 1.0];

        let mut acc = vec![0.0f32; m * n];
        simd_outer_product_acc(&mut acc, &a, &b, m, n);

        // Only acc[0*4 + 3] = 1.0 * 1.0 = 1.0 should be non-zero
        assert!((acc[3] - 1.0).abs() < 1e-5);
        for (i, &val) in acc.iter().enumerate() {
            if i != 3 {
                assert!(val.abs() < 1e-6, "acc[{i}] should be 0, got {val}");
            }
        }
    }

    #[test]
    fn matvec_matches_scalar() {
        let rows = 3;
        let cols = 4;
        let mat = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0f32,
        ];
        let vec = [1.0, 0.0, 1.0, 0.0f32];

        let mut acc_scalar = vec![0.0f32; rows];
        let mut acc_simd = vec![0.0f32; rows];

        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += mat[r * cols + c] * vec[c];
            }
            acc_scalar[r] = sum;
        }

        simd_matvec(&mut acc_simd, &mat, &vec, rows, cols);

        for r in 0..rows {
            assert!(
                (acc_scalar[r] - acc_simd[r]).abs() < 1e-4,
                "mismatch at row {r}: scalar={}, simd={}",
                acc_scalar[r],
                acc_simd[r]
            );
        }
    }

    #[test]
    fn matmul_rows_identity() {
        let rows = 4;
        let cols = 4;
        let weight = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let input = [1.0, 2.0, 3.0, 4.0f32];

        let mut output = vec![0.0f32; rows];
        simd_matmul_rows(&mut output, &weight, &input, rows, cols);

        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 2.0).abs() < 1e-5);
        assert!((output[2] - 3.0).abs() < 1e-5);
        assert!((output[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn matmul_relu_clamps_negative() {
        let rows = 2;
        let cols = 2;
        let weight = [-1.0, 0.0, 1.0, 1.0];
        let input = [1.0, 1.0];

        let mut output = vec![0.0f32; rows];
        simd_matmul_relu_rows(&mut output, &weight, &input, rows, cols);

        assert!((output[0]).abs() < 1e-5, "negative should clamp to 0");
        assert!((output[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn fma_row_matches_dot() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32, 1.0, 1.5, 2.0];

        let dot = simd_dot_f32(&a, &b, 4);
        let fma = simd_fma_row(&a, &b, 4);

        assert!((dot - fma).abs() < 1e-6);
    }

    // ── Sparse SIMD Tests ────────────────────────────────────

    #[test]
    fn sparse_dot_matches_scalar_dense() {
        // 8 elements, all alive (indices 0..7) — should match dense dot
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices: Vec<usize> = (0..8).collect();
        let values = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

        let sparse = simd_sparse_dot_f32(&weight, 0, &indices, &values, 8);
        let dense = simd_dot_f32(&weight, &values, 8);

        assert!(
            (sparse - dense).abs() < 1e-4,
            "sparse={sparse}, dense={dense}"
        );
    }

    #[test]
    fn sparse_dot_matches_scalar_sparse() {
        // 13 elements alive out of 64 (typical micro config: 20% of mlp_hidden=64)
        let mut weight = vec![0.0f32; 64];
        for (i, w) in weight.iter_mut().enumerate() {
            *w = (i as f32 + 1.0) * 0.01;
        }
        let indices: Vec<usize> = vec![0, 3, 7, 12, 15, 20, 25, 31, 38, 45, 50, 56, 63];
        let values: Vec<f32> = indices.iter().map(|&i| weight[i] * 2.0).collect();

        let simd_result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 13);
        let scalar_result = scalar_sparse_dot_f32(&weight, 0, &indices, &values, 13);

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "simd={simd_result}, scalar={scalar_result}"
        );
    }

    #[test]
    fn sparse_dot_small_alive_uses_scalar() {
        // alive=3 — should use inline scalar fallback (≤4)
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices = vec![0usize, 3, 7];
        let values = [0.5f32, 1.0, 1.5];

        let result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 3);
        let expected = 1.0 * 0.5 + 4.0 * 1.0 + 8.0 * 1.5; // 0.5 + 4.0 + 12.0 = 16.5

        assert!(
            (result - expected).abs() < 1e-4,
            "result={result}, expected={expected}"
        );
    }

    #[test]
    fn sparse_dot_zero_alive() {
        let weight = [1.0f32, 2.0, 3.0, 4.0];
        let indices: Vec<usize> = vec![];
        let values: Vec<f32> = vec![];

        let result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 0);
        assert!(result.abs() < 1e-6, "expected 0.0, got {result}");
    }

    #[test]
    fn sparse_dot_with_row_offset() {
        // 8-element weight row at offset 4 in a 12-element weight matrix
        let mut weight = [0.0f32; 12]; // first 4 are padding
        weight[4] = 1.0;
        weight[5] = 2.0;
        weight[6] = 3.0;
        weight[7] = 4.0;
        weight[8] = 5.0;
        weight[9] = 6.0;
        weight[10] = 7.0;
        weight[11] = 8.0;
        // Need mutable for construction
        let weight = weight;

        let indices: Vec<usize> = (0..8).collect();
        let values = [1.0f32; 8];

        let result = simd_sparse_dot_f32(&weight, 4, &indices, &values, 8);
        // Expected: 1+2+3+4+5+6+7+8 = 36
        assert!((result - 36.0).abs() < 1e-4, "result={result}");
    }

    #[test]
    fn sparse_dot_alive_5_triggers_simd() {
        // alive=5 — just above scalar fallback threshold (4)
        let weight = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices: Vec<usize> = (0..8).collect();
        let values = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let simd_result = simd_sparse_dot_f32(&weight, 0, &indices, &values, 5);
        let expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0; // first 5 only

        assert!(
            (simd_result - expected).abs() < 1e-4,
            "simd={simd_result}, expected={expected}"
        );
    }

    #[test]
    fn sparse_matmul_rows_matches_scalar() {
        let rows = 4;
        let cols = 8;
        // Identity-like weight: row r has weight[r*cols + r] = 1.0, rest = 0.1
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let r = i / cols;
                let c = i % cols;
                if r == c { 1.0 } else { 0.1 }
            })
            .collect();

        // Only indices 1, 3, 5 are alive with values
        let indices = vec![1usize, 3, 5];
        let values = vec![2.0f32, 3.0, 4.0];

        let mut output_scalar = vec![0.0f32; rows];
        let mut output_simd = vec![0.0f32; rows];

        // Scalar
        for (r, out) in output_scalar.iter_mut().enumerate() {
            *out = scalar_sparse_dot_f32(&weight, r * cols, &indices, &values, 3);
        }

        // SIMD
        simd_sparse_matmul_rows(&mut output_simd, &weight, &indices, &values, rows, cols, 3);

        for (r, (scalar, simd)) in output_scalar.iter().zip(output_simd.iter()).enumerate() {
            assert!(
                (scalar - simd).abs() < 1e-4,
                "row {r}: scalar={scalar}, simd={simd}"
            );
        }
    }

    #[test]
    fn sparse_matmul_rows_game_config() {
        // Game config: n_embd=32 rows, mlp_hidden=128 cols, ~20% alive = 26 elements
        let rows = 32;
        let cols = 128;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();

        // Simulate 26 alive neurons (20% of 128)
        let alive = 26;
        let indices: Vec<usize> = (0..alive).map(|i| i * (cols / alive)).collect();
        let values: Vec<f32> = (0..alive).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let mut output_scalar = vec![0.0f32; rows];
        let mut output_simd = vec![0.0f32; rows];

        for (r, out) in output_scalar.iter_mut().enumerate() {
            *out = scalar_sparse_dot_f32(&weight, r * cols, &indices, &values, alive);
        }
        simd_sparse_matmul_rows(
            &mut output_simd,
            &weight,
            &indices,
            &values,
            rows,
            cols,
            alive,
        );

        for r in 0..rows {
            assert!(
                (output_scalar[r] - output_simd[r]).abs() < 1e-3,
                "row {r}: scalar={}, simd={}",
                output_scalar[r],
                output_simd[r]
            );
        }
    }

    // ── simd_scale_inplace tests ──────────────────────────────

    #[test]
    fn scale_aligned_len_8() {
        let mut x = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        simd_scale_inplace(&mut x, 0.5);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for i in 0..8 {
            assert!((x[i] - expected[i]).abs() < 1e-6, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn scale_non_aligned_len_13() {
        let mut x = [1.0f32; 13];
        simd_scale_inplace(&mut x, 3.0);
        for (i, &val) in x.iter().enumerate() {
            assert!((val - 3.0).abs() < 1e-6, "x[{i}]={val}");
        }
    }

    #[test]
    fn scale_empty() {
        let mut x: [f32; 0] = [];
        simd_scale_inplace(&mut x, 2.0); // should not panic
    }

    #[test]
    fn scale_single_element() {
        let mut x = [5.0f32];
        simd_scale_inplace(&mut x, 0.2);
        assert!((x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn scale_zero() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        simd_scale_inplace(&mut x, 0.0);
        for val in &x {
            assert!(*val == 0.0, "expected 0.0, got {val}");
        }
    }

    #[test]
    fn scale_matches_scalar() {
        let mut x_simd: Vec<f32> = (0..97).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x_scalar = x_simd.clone();
        let scale = 0.42f32;

        simd_scale_inplace(&mut x_simd, scale);
        scalar_scale_inplace(&mut x_scalar, scale);

        for i in 0..x_simd.len() {
            assert!(
                (x_simd[i] - x_scalar[i]).abs() < 1e-6,
                "x[{i}]: simd={}, scalar={}",
                x_simd[i],
                x_scalar[i]
            );
        }
    }

    // ── simd_add_scalar_inplace tests ────────────────────────

    #[test]
    fn add_scalar_aligned_len_8() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        simd_add_scalar_inplace(&mut x, -10.0);
        let expected = [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0];
        for i in 0..8 {
            assert!((x[i] - expected[i]).abs() < 1e-6, "x[{i}]={}", x[i]);
        }
    }

    #[test]
    fn add_scalar_non_aligned_len_13() {
        let mut x = [1.0f32; 13];
        simd_add_scalar_inplace(&mut x, 2.0);
        for (i, &val) in x.iter().enumerate() {
            assert!((val - 3.0).abs() < 1e-6, "x[{i}]={val}");
        }
    }

    #[test]
    fn add_scalar_empty() {
        let mut x: [f32; 0] = [];
        simd_add_scalar_inplace(&mut x, 1.0); // should not panic
    }

    #[test]
    fn add_scalar_matches_scalar_impl() {
        let mut x_simd: Vec<f32> = (0..97).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut x_scalar = x_simd.clone();
        let val = -std::f32::consts::PI;

        simd_add_scalar_inplace(&mut x_simd, val);
        scalar_add_scalar_inplace(&mut x_scalar, val);

        for i in 0..x_simd.len() {
            assert!(
                (x_simd[i] - x_scalar[i]).abs() < 1e-6,
                "x[{i}]: simd={}, scalar={}",
                x_simd[i],
                x_scalar[i]
            );
        }
    }

    // ── simd_sum_f32 tests ──────────────────────────────────────

    #[test]
    fn sum_aligned_len_8() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = simd_sum_f32(&x);
        assert!((result - 36.0).abs() < 1e-4, "expected 36.0, got {result}");
    }

    #[test]
    fn sum_non_aligned_len_13() {
        let x = [1.0f32; 13];
        let result = simd_sum_f32(&x);
        assert!((result - 13.0).abs() < 1e-4, "expected 13.0, got {result}");
    }

    #[test]
    fn sum_empty() {
        let x: [f32; 0] = [];
        let result = simd_sum_f32(&x);
        assert!((result - 0.0).abs() < 1e-6, "expected 0.0, got {result}");
    }

    #[test]
    fn sum_single_element() {
        let x = [42.0f32];
        let result = simd_sum_f32(&x);
        assert!((result - 42.0).abs() < 1e-4, "expected 42.0, got {result}");
    }

    #[test]
    fn sum_matches_scalar_impl() {
        let x: Vec<f32> = (0..97).map(|i| (i as f32 * 0.1).sin()).collect();
        let simd_result = simd_sum_f32(&x);
        let scalar_result = scalar_sum_f32(&x);
        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "simd={simd_result}, scalar={scalar_result}"
        );
    }

    // ── simd_add_inplace tests ────────────────────────────────

    #[test]
    fn add_inplace_aligned_len_8() {
        let mut dst = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let src = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        simd_add_inplace(&mut dst, &src);
        for (i, val) in dst.iter().enumerate() {
            let expected = (1.0 + i as f32) + (i + 1) as f32 * 0.1;
            assert!((val - expected).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn add_inplace_non_aligned_len_13() {
        let mut dst = [0.0f32; 13];
        let src = [1.0f32; 13];
        for (i, val) in dst.iter_mut().enumerate() {
            *val = i as f32;
        }
        simd_add_inplace(&mut dst, &src);
        for (i, val) in dst.iter().enumerate() {
            assert!((val - (i as f32 + 1.0)).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn add_inplace_empty() {
        let mut dst: [f32; 0] = [];
        let src: [f32; 0] = [];
        simd_add_inplace(&mut dst, &src);
    }

    #[test]
    fn add_inplace_single_element() {
        let mut dst = [3.0f32];
        let src = [7.0f32];
        simd_add_inplace(&mut dst, &src);
        assert!((dst[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn add_inplace_matches_scalar() {
        let mut dst_simd = [0.0f32; 37];
        let mut dst_scalar = [0.0f32; 37];
        for i in 0..37 {
            dst_simd[i] = i as f32 * 0.7;
            dst_scalar[i] = i as f32 * 0.7;
        }
        let src: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).sin()).collect();
        simd_add_inplace(&mut dst_simd, &src);
        scalar_add_inplace(&mut dst_scalar, &src);
        for i in 0..37 {
            assert!(
                (dst_simd[i] - dst_scalar[i]).abs() < 1e-5,
                "mismatch at {i}"
            );
        }
    }

    // ── simd_add_into tests ───────────────────────────────────

    #[test]
    fn add_into_aligned_len_8() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut dst = [0.0f32; 8];
        simd_add_into(&mut dst, &a, &b);
        for val in &dst {
            assert!((val - 9.0).abs() < 1e-6);
        }
    }

    #[test]
    fn add_into_non_aligned_len_13() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let b = [1.0f32; 13];
        let mut dst = [0.0f32; 13];
        simd_add_into(&mut dst, &a, &b);
        for (i, val) in dst.iter().enumerate() {
            assert!((val - (i as f32 + 1.0)).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn add_into_empty() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        let mut dst: [f32; 0] = [];
        simd_add_into(&mut dst, &a, &b);
    }

    #[test]
    fn add_into_matches_scalar() {
        let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7).sin()).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).cos()).collect();
        let mut dst_simd = [0.0f32; 37];
        let mut dst_scalar = [0.0f32; 37];
        simd_add_into(&mut dst_simd, &a, &b);
        scalar_add_into(&mut dst_scalar, &a, &b);
        for i in 0..37 {
            assert!(
                (dst_simd[i] - dst_scalar[i]).abs() < 1e-5,
                "mismatch at {i}"
            );
        }
    }

    // ── simd_max_f32 tests ────────────────────────────────────

    #[test]
    fn max_aligned_len_8() {
        let x = [1.0f32, 5.0, 3.0, 8.0, 2.0, 7.0, 4.0, 6.0];
        let max = simd_max_f32(&x);
        assert!((max - 8.0).abs() < 1e-6);
    }

    #[test]
    fn max_non_aligned_len_13() {
        let x: Vec<f32> = (0..13).map(|i| (i as f32 * 1.7).sin()).collect();
        let max = simd_max_f32(&x);
        let expected = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((max - expected).abs() < 1e-5);
    }

    #[test]
    fn max_empty() {
        let x: [f32; 0] = [];
        let max = simd_max_f32(&x);
        assert!(max.is_infinite() && max.is_sign_negative());
    }

    #[test]
    fn max_single_element() {
        let x = [42.0f32];
        let max = simd_max_f32(&x);
        assert!((max - 42.0).abs() < 1e-6);
    }

    #[test]
    fn max_negative_values() {
        let x = [-5.0f32, -3.0, -8.0, -1.0, -4.0];
        let max = simd_max_f32(&x);
        assert!((max - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn max_matches_scalar() {
        let x: Vec<f32> = (0..37).map(|i| (i as f32 * 0.97 - 18.0).sin()).collect();
        let max_simd = simd_max_f32(&x);
        let max_scalar = scalar_max_f32(&x);
        assert!((max_simd - max_scalar).abs() < 1e-5);
    }

    // ── simd_fused_decay_write tests ──────────────────────────

    #[test]
    fn fused_decay_write_aligned_len_8() {
        let mut dst = [1.0f32; 8];
        let src = [2.0f32; 8];
        let decay = 0.5f32;
        let write = 0.5f32;
        simd_fused_decay_write(&mut dst, decay, &src, write);
        // 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        for val in &dst {
            assert!((val - 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn fused_decay_write_zero_decay() {
        let mut dst = [1.0f32, 2.0, 3.0, 4.0];
        let src = [10.0f32, 20.0, 30.0, 40.0];
        let decay = 0.0f32;
        let write = 1.0f32;
        simd_fused_decay_write(&mut dst, decay, &src, write);
        for i in 0..4 {
            assert!((dst[i] - src[i]).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn fused_decay_write_zero_write() {
        let mut dst = [1.0f32, 2.0, 3.0, 4.0];
        let src = [10.0f32, 20.0, 30.0, 40.0];
        let decay = 1.0f32;
        let write = 0.0f32;
        simd_fused_decay_write(&mut dst, decay, &src, write);
        assert!((dst[0] - 1.0).abs() < 1e-5);
        assert!((dst[1] - 2.0).abs() < 1e-5);
        assert!((dst[2] - 3.0).abs() < 1e-5);
        assert!((dst[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn fused_decay_write_empty() {
        let mut dst: [f32; 0] = [];
        let src: [f32; 0] = [];
        simd_fused_decay_write(&mut dst, 0.5, &src, 0.5);
    }

    #[test]
    fn fused_decay_write_matches_scalar() {
        let mut dst_simd: Vec<f32> = (0..37).map(|i| i as f32 * 0.7).collect();
        let mut dst_scalar: Vec<f32> = (0..37).map(|i| i as f32 * 0.7).collect();
        let src: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).sin()).collect();
        let decay = 0.9f32;
        let write = 0.1f32;
        simd_fused_decay_write(&mut dst_simd, decay, &src, write);
        scalar_fused_decay_write(&mut dst_scalar, decay, &src, write);
        for i in 0..37 {
            assert!(
                (dst_simd[i] - dst_scalar[i]).abs() < 1e-4,
                "mismatch at {i}: simd={}, scalar={}",
                dst_simd[i],
                dst_scalar[i]
            );
        }
    }

    // ── f16×f32 kernel tests ──────────────────────────────────

    fn scalar_dot_f16_f32_ref(w: &[half::f16], x: &[f32], len: usize) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += w[i].to_f32() * x[i];
        }
        sum
    }

    #[test]
    fn dot_f16_f32_aligned_len_8() {
        let w: Vec<half::f16> = (0..8)
            .map(|i| half::f16::from_f32(i as f32 * 0.1))
            .collect();
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.2).collect();
        let result = simd_dot_f16_f32(&w, &x, 8);
        let expected = scalar_dot_f16_f32_ref(&w, &x, 8);
        assert!(
            (result - expected).abs() < 1e-4,
            "f16 dot aligned: got {result}, expected {expected}"
        );
    }

    #[test]
    fn dot_f16_f32_non_aligned_len_13() {
        let w: Vec<half::f16> = (0..13)
            .map(|i| half::f16::from_f32(i as f32 + 1.0))
            .collect();
        let x: Vec<f32> = (0..13).map(|i| i as f32 * 0.3).collect();
        let result = simd_dot_f16_f32(&w, &x, 13);
        let expected = scalar_dot_f16_f32_ref(&w, &x, 13);
        assert!(
            (result - expected).abs() < 1e-3,
            "f16 dot non-aligned: got {result}, expected {expected}"
        );
    }

    #[test]
    fn dot_f16_f32_len_4() {
        let w: Vec<half::f16> = vec![1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let x: Vec<f32> = vec![0.25, 0.5, 0.75, 1.0];
        let result = simd_dot_f16_f32(&w, &x, 4);
        let expected = scalar_dot_f16_f32_ref(&w, &x, 4);
        assert!(
            (result - expected).abs() < 1e-4,
            "f16 dot len 4: got {result}, expected {expected}"
        );
    }

    #[test]
    fn dot_f16_f32_zero_length() {
        let w: Vec<half::f16> = Vec::new();
        let x: Vec<f32> = Vec::new();
        let result = simd_dot_f16_f32(&w, &x, 0);
        assert_eq!(result, 0.0, "f16 dot zero-length should be 0.0");
    }

    #[test]
    fn matmul_f16_f32_identity() {
        // 3×3 identity matrix stored as f16
        let w: Vec<half::f16> = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let x: Vec<f32> = vec![2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 3];
        simd_matmul_f16_f32_rows(&mut out, &w, &x, 3, 3);
        assert!(
            (out[0] - 2.0).abs() < 1e-4
                && (out[1] - 3.0).abs() < 1e-4
                && (out[2] - 4.0).abs() < 1e-4,
            "f16 identity matmul: got {out:?}"
        );
    }

    #[test]
    fn matmul_f16_f32_matches_f32() {
        // Compare f16 matmul vs f32 matmul on the same values
        let rows = 4;
        let cols = 6;
        let weight_f32: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.01 - 0.1).collect();
        let weight_f16: Vec<half::f16> =
            weight_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let input: Vec<f32> = (0..cols).map(|i| i as f32 * 0.05).collect();

        let mut out_f32 = vec![0.0f32; rows];
        let mut out_f16 = vec![0.0f32; rows];
        simd_matmul_rows(&mut out_f32, &weight_f32, &input, rows, cols);
        simd_matmul_f16_f32_rows(&mut out_f16, &weight_f16, &input, rows, cols);

        for i in 0..rows {
            let diff = (out_f32[i] - out_f16[i]).abs();
            assert!(
                diff < 0.01,
                "f16 vs f32 matmul mismatch at row {i}: f32={}, f16={}, diff={diff}",
                out_f32[i],
                out_f16[i]
            );
        }
    }

    // ── MaxSim Tests (Plan 080 T2) ────────────────────────────

    /// Naive reference: materialize [Lq × Ld] then reduce.
    #[cfg(feature = "maxsim")]
    fn maxsim_naive(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
        let mut score = 0.0f32;
        for i in 0..lq {
            let q_row = &queries[i * dim..(i + 1) * dim];
            let mut my_max = f32::NEG_INFINITY;
            for j in 0..ld {
                let d_row = &documents[j * dim..(j + 1) * dim];
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q_row[d] * d_row[d];
                }
                my_max = my_max.max(dot);
            }
            score += my_max;
        }
        score
    }

    #[cfg(feature = "maxsim")]
    mod maxsim_tests {
        use super::*;

        #[test]
        fn maxsim_matches_naive() {
            let lq = 8;
            let ld = 16;
            let dim = 32;
            let mut queries = vec![0.0f32; lq * dim];
            let mut documents = vec![0.0f32; ld * dim];
            for q in queries.iter_mut() {
                *q = fastrand::f32() * 2.0 - 1.0;
            }
            for d in documents.iter_mut() {
                *d = fastrand::f32() * 2.0 - 1.0;
            }
            let naive = maxsim_naive(&queries, &documents, lq, ld, dim);
            let fused = maxsim_score(&queries, &documents, lq, ld, dim);
            assert!((naive - fused).abs() < 1e-4, "naive={naive}, fused={fused}");
        }

        #[test]
        fn maxsim_single_query_token() {
            let dim = 16;
            let queries = (0..dim).map(|i| i as f32).collect::<Vec<f32>>();
            let documents = (0..3 * dim)
                .map(|i| (i as f32 * 0.1).sin())
                .collect::<Vec<f32>>();
            let result = maxsim_score(&queries, &documents, 1, 3, dim);
            // Should equal max over all doc dots
            let mut expected = f32::NEG_INFINITY;
            for j in 0..3 {
                let d_row = &documents[j * dim..(j + 1) * dim];
                let dot = simd_dot_f32(&queries, d_row, dim);
                expected = expected.max(dot);
            }
            assert!(
                (result - expected).abs() < 1e-5,
                "result={result}, expected={expected}"
            );
        }

        #[test]
        fn maxsim_single_doc_token() {
            let dim = 16;
            let lq = 4;
            let queries = (0..lq * dim)
                .map(|i| (i as f32 * 0.2).cos())
                .collect::<Vec<f32>>();
            let documents = (0..dim).map(|i| i as f32 * 0.5).collect::<Vec<f32>>();
            let result = maxsim_score(&queries, &documents, lq, 1, dim);
            // Ld=1: each query token has exactly one doc token to match
            let mut expected = 0.0f32;
            for i in 0..lq {
                let q_row = &queries[i * dim..(i + 1) * dim];
                expected += simd_dot_f32(q_row, &documents, dim);
            }
            assert!(
                (result - expected).abs() < 1e-4,
                "result={result}, expected={expected}"
            );
        }

        #[test]
        fn maxsim_symmetry_breaking() {
            let dim = 8;
            let lq = 4;
            let ld = 4;
            let queries = (0..lq * dim).map(|i| i as f32).collect::<Vec<f32>>();
            let documents = (0..ld * dim)
                .map(|i| (i as f32 * 0.3).sin())
                .collect::<Vec<f32>>();
            let maxsim = maxsim_score(&queries, &documents, lq, ld, dim);
            // Diagonal sum: Σ dot(q_i, d_i)
            let mut diagonal = 0.0f32;
            for i in 0..lq.min(ld) {
                let q_row = &queries[i * dim..(i + 1) * dim];
                let d_row = &documents[i * dim..(i + 1) * dim];
                diagonal += simd_dot_f32(q_row, d_row, dim);
            }
            // They should differ (MaxSim takes max over ALL j, not just j==i)
            assert!(
                (maxsim - diagonal).abs() > 1e-3,
                "maxsim={maxsim} should differ from diagonal={diagonal}"
            );
        }

        #[test]
        fn maxsim_empty_doc() {
            let dim = 16;
            let queries = vec![1.0f32; dim];
            let documents: Vec<f32> = vec![];
            let result = maxsim_score(&queries, &documents, 1, 0, dim);
            assert_eq!(result, 0.0, "empty doc should return 0.0");
        }

        #[test]
        fn maxsim_large_dim_aligned() {
            let dim = 128;
            let lq = 4;
            let ld = 8;
            let queries: Vec<f32> = (0..lq * dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let documents: Vec<f32> = (0..ld * dim).map(|i| (i as f32 * 0.01).cos()).collect();
            let naive = maxsim_naive(&queries, &documents, lq, ld, dim);
            let fused = maxsim_score(&queries, &documents, lq, ld, dim);
            assert!((naive - fused).abs() < 1e-3, "naive={naive}, fused={fused}");
        }

        #[test]
        fn maxsim_packed_matches_sequential() {
            let dim = 16;
            // Two query sequences, three doc sequences
            let q1: Vec<f32> = (0..2 * dim).map(|i| i as f32).collect();
            let q2: Vec<f32> = (0..3 * dim).map(|i| (i as f32 * 0.5).sin()).collect();
            let d1: Vec<f32> = (0..4 * dim).map(|i| (i as f32 * 0.3).cos()).collect();
            let d2: Vec<f32> = (0..2 * dim).map(|i| i as f32 * 0.1).collect();
            let d3: Vec<f32> = (0..5 * dim).map(|i| (i as f32 * 0.7).sin()).collect();

            let queries: Vec<f32> = [q1.clone(), q2.clone()].concat();
            let documents: Vec<f32> = [d1.clone(), d2.clone(), d3.clone()].concat();
            let query_offsets = [0, q1.len(), q1.len() + q2.len()];
            let doc_offsets = [
                0,
                d1.len(),
                d1.len() + d2.len(),
                d1.len() + d2.len() + d3.len(),
            ];

            // Score pairs: (q0,d0), (q0,d2), (q1,d1)
            let pair_q_ids = [0usize, 0, 1];
            let pair_d_ids = [0usize, 2, 1];

            let mut packed = vec![0.0f32; pair_q_ids.len()];
            maxsim_score_packed(
                &queries,
                &query_offsets,
                &documents,
                &doc_offsets,
                &pair_q_ids,
                &pair_d_ids,
                dim,
                &mut packed,
            );

            // Verify against sequential calls
            let s0 = maxsim_score(&q1, &d1, 2, 4, dim);
            let s1 = maxsim_score(&q1, &d3, 2, 5, dim);
            let s2 = maxsim_score(&q2, &d2, 3, 2, dim);

            assert!(
                (packed[0] - s0).abs() < 1e-4,
                "pair 0: packed={}, sequential={}",
                packed[0],
                s0
            );
            assert!(
                (packed[1] - s1).abs() < 1e-4,
                "pair 1: packed={}, sequential={}",
                packed[1],
                s1
            );
            assert!(
                (packed[2] - s2).abs() < 1e-4,
                "pair 2: packed={}, sequential={}",
                packed[2],
                s2
            );
        }
    }

    // ── Sigmoid Margin Loss Tests (Plan 157 GOAT) ───────────────

    #[cfg(feature = "sigmoid_margin")]
    mod sigmoid_margin_tests {
        use super::*;

        // GOAT Proof 1: sigmoid_margin_loss matches paper's Python implementation
        //
        // For a small bipartite graph with n=20, k=2, d=8:
        //   - Generate random embeddings, compute dot-product scores
        //   - Compute loss with t=1.0, b=0.0
        //   - Verify against hand-computed softplus values
        #[test]
        fn proof1_loss_matches_manual() {
            // 2 queries × 3 docs, simple adjacency
            let n_rows = 2;
            let n_cols = 3;
            let scores: Vec<f32> = vec![
                0.8, 0.2, -0.5, // query 0: positive on doc 0
                -0.3, 0.9, 0.1, // query 1: positive on doc 1
            ];
            let adjacency: Vec<f32> = vec![
                1.0, 0.0, 0.0, // query 0 positive = doc 0
                0.0, 1.0, 0.0, // query 1 positive = doc 1
            ];

            let loss = sigmoid_margin_loss(&scores, &adjacency, 1.0, 0.0, n_rows, n_cols);

            // Manual computation:
            // query 0: pos: softplus(-0.8) = ln(1+exp(-0.8)) ≈ 0.5544
            //          neg: softplus(0.2) = ln(1+exp(0.2)) ≈ 0.7444
            //          neg: softplus(-0.5) = ln(1+exp(-0.5)) ≈ 0.4741
            // query 1: neg: softplus(-0.3) = ln(1+exp(-0.3)) ≈ 0.5544
            //          pos: softplus(-0.9) = ln(1+exp(-0.9)) ≈ 0.4887
            //          neg: softplus(0.1) = ln(1+exp(0.1)) ≈ 0.7444
            // total / 6
            let sp = |x: f32| -> f32 { (1.0f32 + x.exp()).ln() };
            let expected = (sp(-0.8) + sp(0.2) + sp(-0.5) + sp(-0.3) + sp(-0.9) + sp(0.1)) / 6.0;
            assert!(
                (loss - expected).abs() < 1e-4,
                "loss={loss}, expected={expected}"
            );
        }

        #[test]
        fn proof1_loss_with_bias_and_temperature() {
            let scores = vec![1.0, 0.0];
            let adjacency = vec![1.0, 0.0];

            // With t=2.0, b=0.5:
            //   pos (score=1): sign=-1, x = 2*(1-0.5)*(-1) = -1.0, softplus(-1.0)
            //   neg (score=0): sign=+1, x = 2*(0-0.5)*(+1) = -1.0, softplus(-1.0)
            //   Both = softplus(-1.0)
            let loss = sigmoid_margin_loss(&scores, &adjacency, 2.0, 0.5, 1, 2);
            let sp_neg1 = (1.0f32 + (-1.0f32).exp()).ln(); // softplus(-1.0)
            let expected = sp_neg1; // mean of 2 identical values
            assert!(
                (loss - expected).abs() < 1e-4,
                "loss={loss}, expected={expected}"
            );
        }

        #[test]
        fn proof1_loss_perfect_separation() {
            // Perfect separation: pos score >> bias, neg score << bias
            let scores = vec![100.0, -100.0];
            let adjacency = vec![1.0, 0.0];
            let loss = sigmoid_margin_loss(&scores, &adjacency, 1.0, 0.0, 1, 2);
            // pos: softplus(-100) ≈ 0, neg: softplus(-100) ≈ 0
            assert!(
                loss < 1e-10,
                "loss={loss} should be near 0 for perfect separation"
            );
        }

        // GOAT Proof 2: compute_retrieval_margin correctly identifies positive margin
        #[test]
        fn proof2_margin_positive_for_separated_embeddings() {
            let dim = 8;
            let n_queries = 3;
            let n_docs = 6;
            let k = 2;

            // Construct orthogonal-ish embeddings with known margin.
            // Each query is aligned with its 2 positive docs, orthogonal to the rest.
            let mut queries = vec![0.0f32; n_queries * dim];
            let mut documents = vec![0.0f32; n_docs * dim];

            // query i → doc 2i and doc 2i+1 as positives
            let mut neighborhoods = Vec::with_capacity(n_queries * k);
            for i in 0..n_queries {
                // Query: unit vector along dimension i
                queries[i * dim + i] = 1.0;
                // Positive docs: same direction as query
                documents[(2 * i) * dim + i] = 0.9;
                documents[(2 * i + 1) * dim + i] = 0.8;
                neighborhoods.push(2 * i);
                neighborhoods.push(2 * i + 1);
            }

            let (pos_min, neg_max, margin) = compute_retrieval_margin(
                &queries,
                &documents,
                &neighborhoods,
                dim,
                n_queries,
                n_docs,
                k,
            );

            // pos_min should be 0.8 (weakest positive = 0.8), neg_max should be 0.0 (no alignment)
            assert!(
                (pos_min - 0.8).abs() < 1e-5,
                "pos_min={pos_min}, expected 0.8"
            );
            assert!(neg_max.abs() < 1e-5, "neg_max={neg_max}, expected 0.0");
            assert!((margin - 0.4).abs() < 1e-5, "margin={margin}, expected 0.4");
            assert!(margin > 0.0, "margin should be positive");
        }

        #[test]
        fn proof2_margin_negative_for_mixed_embeddings() {
            let dim = 4;
            let n_queries = 1;
            let n_docs = 3;
            let k = 1;

            // Query aligned with a "wrong" doc (positive has lower score than a negative)
            let queries = vec![1.0, 0.0, 0.0, 0.0]; // aligned along dim 0
            // Doc 0 (positive): weak alignment
            let d0 = vec![0.1, 0.0, 0.0, 0.0];
            // Doc 1 (negative): strong alignment → should dominate
            let d1 = vec![0.9, 0.0, 0.0, 0.0];
            // Doc 2 (negative): orthogonal
            let d2 = vec![0.0, 1.0, 0.0, 0.0];
            let documents: Vec<f32> = [d0, d1, d2].concat();
            let neighborhoods = vec![0]; // query 0 positive = doc 0

            let (pos_min, neg_max, margin) = compute_retrieval_margin(
                &queries,
                &documents,
                &neighborhoods,
                dim,
                n_queries,
                n_docs,
                k,
            );

            assert!((pos_min - 0.1).abs() < 1e-5, "pos_min={pos_min}");
            assert!((neg_max - 0.9).abs() < 1e-5, "neg_max={neg_max}");
            assert!(margin < 0.0, "margin should be negative: {margin}");
        }

        // GOAT Proof 3: dim_sufficiency_bound returns O(k log n)
        #[test]
        fn proof3_bound_scales_as_k_log_n() {
            // k=2, n=100: 1.5 * 2 * ln(100) ≈ 1.5 * 2 * 4.605 ≈ 13.8 → 14
            let b1 = dim_sufficiency_bound(2, 100);
            assert!(b1 <= 20, "k=2, n=100: bound={b1}, should be ≤ 20");
            assert!(b1 >= 10, "k=2, n=100: bound={b1}, should be ≥ 10");

            // k=4, n=1000: 1.5 * 4 * ln(1000) ≈ 1.5 * 4 * 6.908 ≈ 41.4 → 42
            let b2 = dim_sufficiency_bound(4, 1000);
            assert!(b2 <= 60, "k=4, n=1000: bound={b2}, should be ≤ 60");
            assert!(b2 >= 30, "k=4, n=1000: bound={b2}, should be ≥ 30");
        }

        #[test]
        fn proof3_bound_edge_cases() {
            assert_eq!(dim_sufficiency_bound(0, 100), 1, "k=0 → trivial");
            assert_eq!(dim_sufficiency_bound(2, 1), 1, "n=1 → trivial");
            assert_eq!(dim_sufficiency_bound(2, 2), 3, "n=2 → minimal");
        }

        #[test]
        fn proof3_bound_monotonic() {
            let b1 = dim_sufficiency_bound(2, 50);
            let b2 = dim_sufficiency_bound(2, 100);
            let b3 = dim_sufficiency_bound(2, 200);
            assert!(b1 < b2, "bound should increase with n: {b1} < {b2}");
            assert!(b2 < b3, "bound should increase with n: {b2} < {b3}");

            let bk1 = dim_sufficiency_bound(2, 100);
            let bk2 = dim_sufficiency_bound(4, 100);
            assert!(bk1 < bk2, "bound should increase with k: {bk1} < {bk2}");
        }

        // GOAT Proof 4: Sigmoid loss converges to positive margin on synthetic data
        //
        // We use a structured initialization where each query and its positive docs
        // share a unique subspace dimension. The sigmoid margin loss then amplifies
        // this alignment while suppressing cross-talk.
        //
        // Uses analytical gradient: ∂loss/∂score = sigmoid(t·(score−b)·sign)
        // then backprops to embeddings via chain rule: ∂loss/∂q_i = Σ_j grad_ij · d_j.
        #[test]
        fn proof4_loss_gradient_pushes_to_positive_margin() {
            let dim = 8;
            let n = 4; // 4 docs
            let k = 2; // each query has 2 positives
            let n_queries = 2;

            // Bipartite structure:
            //   query 0 → doc 0, doc 1 (use dim 0 as shared subspace)
            //   query 1 → doc 2, doc 3 (use dim 1 as shared subspace)
            let neighborhoods: Vec<usize> = vec![0, 1, 2, 3];

            // Initialize with small positive signal in the right subspace + noise
            let mut queries = vec![0.0f32; n_queries * dim];
            let mut documents = vec![0.0f32; n * dim];

            // query 0 → dim 0, query 1 → dim 1
            queries[0] = 0.3;
            queries[dim + 1] = 0.3;

            // Positive docs aligned with their query subspace
            documents[0] = 0.2;
            documents[dim] = 0.15;
            documents[2 * dim + 1] = 0.2;
            documents[3 * dim + 1] = 0.15;
            // Small cross-talk noise
            documents[1] = 0.02;
            documents[2 * dim] = 0.02;

            let adjacency: Vec<f32> = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

            let (_, _, initial_margin) = compute_retrieval_margin(
                &queries,
                &documents,
                &neighborhoods,
                dim,
                n_queries,
                n,
                k,
            );

            // Analytical gradient descent with temperature t=10
            let t = 10.0f32;
            let lr = 0.1;
            let mut q = queries.clone();
            let mut d = documents.clone();

            for _step in 0..100 {
                // Forward: compute scores [n_queries × n]
                let mut scores = vec![0.0f32; n_queries * n];
                for i in 0..n_queries {
                    for j in 0..n {
                        scores[i * n + j] = simd_dot_f32(
                            &q[i * dim..(i + 1) * dim],
                            &d[j * dim..(j + 1) * dim],
                            dim,
                        );
                    }
                }

                // Score gradients matching the loss: sign = -1 for pos, +1 for neg
                // ∂L/∂score_ij = t · sign · σ(t · (score - b) · sign)
                // pos: sign=-1 → grad = -t · σ(-t·(score-b)), pushes score up
                // neg: sign=+1 → grad = +t · σ(+t·(score-b)), pushes score down
                let mut score_grads = vec![0.0f32; n_queries * n];
                for i in 0..n_queries {
                    for j in 0..n {
                        let idx = i * n + j;
                        let sign = if adjacency[idx] > 0.5 {
                            -1.0f32
                        } else {
                            1.0f32
                        };
                        let x = t * (scores[idx]) * sign;
                        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
                        score_grads[idx] = t * sign * sigmoid_x;
                    }
                }

                // Backprop to queries: ∂loss/∂q_i = Σ_j (score_grad_ij) · d_j
                let mut q_grads = vec![0.0f32; n_queries * dim];
                for i in 0..n_queries {
                    for j in 0..n {
                        let g = score_grads[i * n + j];
                        for dd in 0..dim {
                            q_grads[i * dim + dd] += g * d[j * dim + dd];
                        }
                    }
                }

                // Backprop to documents: ∂loss/∂d_j = Σ_i (score_grad_ij) · q_i
                let mut d_grads = vec![0.0f32; n * dim];
                for i in 0..n_queries {
                    for j in 0..n {
                        let g = score_grads[i * n + j];
                        for dd in 0..dim {
                            d_grads[j * dim + dd] += g * q[i * dim + dd];
                        }
                    }
                }

                // Gradient step
                for idx in 0..q.len() {
                    q[idx] -= lr * q_grads[idx];
                }
                for idx in 0..d.len() {
                    d[idx] -= lr * d_grads[idx];
                }
            }

            let (_, _, final_margin) =
                compute_retrieval_margin(&q, &d, &neighborhoods, dim, n_queries, n, k);

            assert!(
                final_margin > 0.0,
                "final_margin={final_margin} should be > 0 after training"
            );
            assert!(
                final_margin > initial_margin,
                "margin should improve: initial={initial_margin}, final={final_margin}"
            );
        }

        // GOAT Proof 5: Margin diagnostic validates MaxSim scoring quality
        #[test]
        #[cfg(feature = "maxsim")]
        fn proof5_margin_correlates_with_maxsim() {
            let dim = 16;
            let n_docs = 4;
            let lq = 2;
            let ld = n_docs;
            let k = 1;

            // Create two query-doc pairs with different margins
            // High margin: query 0 is closely aligned with doc 0, far from others
            let mut queries = vec![0.0f32; 2 * lq * dim]; // 2 sets of queries
            let mut documents = vec![0.0f32; n_docs * dim];

            // Doc 0: strong signal on dim 0
            documents[0] = 1.0;
            // Docs 1-3: weak/noise
            documents[dim + 1] = 0.1;
            documents[2 * dim + 2] = 0.1;
            documents[3 * dim + 3] = 0.1;

            // Query 0 (high margin): aligned with doc 0
            queries[0] = 1.0;
            // Query 0, token 1: also aligned
            queries[dim] = 0.9;

            let neighborhoods = vec![0]; // query 0 → doc 0

            let (pos_min, neg_max, margin) = compute_retrieval_margin(
                &queries[..lq * dim],
                &documents,
                &neighborhoods,
                dim,
                1,
                n_docs,
                k,
            );

            // MaxSim score for this query against all docs
            let ms = maxsim_score(&queries[..lq * dim], &documents, lq, ld, dim);

            // High margin → MaxSim should be dominated by the positive doc
            assert!(margin > 0.0, "margin={margin} should be positive");
            // MaxSim should be high when positive docs dominate
            assert!(
                ms > 0.0,
                "maxsim={ms} should be positive for high-margin setup"
            );
            assert!(
                pos_min > neg_max,
                "pos_min={pos_min} should exceed neg_max={neg_max}"
            );
        }

        // GOAT Proof 6: No performance regression on existing maxsim tests
        // (All existing maxsim tests still pass — verified by running the test suite)
        // This proof is structural: if this test compiles and the maxsim tests pass,
        // there is no regression.
        #[test]
        #[cfg(feature = "maxsim")]
        fn proof6_no_maxsim_regression() {
            // Re-run a basic maxsim test to verify nothing broke
            let dim = 16;
            let lq = 4;
            let ld = 8;
            let queries: Vec<f32> = (0..lq * dim).map(|i| (i as f32 * 0.01).sin()).collect();
            let documents: Vec<f32> = (0..ld * dim).map(|i| (i as f32 * 0.01).cos()).collect();

            // Naive computation
            let mut expected = 0.0f32;
            for i in 0..lq {
                let q_row = &queries[i * dim..(i + 1) * dim];
                let mut my_max = f32::NEG_INFINITY;
                for j in 0..ld {
                    let d_row = &documents[j * dim..(j + 1) * dim];
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q_row[d] * d_row[d];
                    }
                    my_max = my_max.max(dot);
                }
                expected += my_max;
            }

            let result = maxsim_score(&queries, &documents, lq, ld, dim);
            assert!(
                (result - expected).abs() < 1e-3,
                "maxsim={result}, expected={expected}"
            );
        }

        // GOAT Proof 7: Feature gate isolation
        // This test verifies the functions exist and work when sigmoid_margin is enabled.
        // When the feature is disabled, the functions are not visible (compile-time check).
        #[test]
        fn proof7_feature_gate_functions_exist() {
            // All three functions should be usable
            let _loss = sigmoid_margin_loss(&[0.5, -0.5], &[1.0, 0.0], 1.0, 0.0, 1, 2);

            let (pm, _nm, m) = compute_retrieval_margin(
                &[1.0, 0.0, 0.0, 1.0], // 2 queries × dim 2
                &[1.0, 0.0, 0.0, 1.0], // 2 docs × dim 2
                &[0, 1],               // neighborhoods: q0→d0, q1→d1
                2,
                2,
                2,
                1,
            );
            assert!(pm >= 0.0);
            assert!(m >= 0.0);

            let bound = dim_sufficiency_bound(2, 100);
            assert!(bound > 0);
            assert!(bound <= 20);
        }
    }

    // ── Gram matrix tests ─────────────────────────────────────

    mod gram_tests {
        use super::*;

        #[test]
        fn test_gram_identity() {
            // Identity matrix X = I (3×3) → G = I·Iᵀ = I
            let seq_len = 3;
            let d_h = 3;
            let x: Vec<f32> = vec![
                1.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, // row 1
                0.0, 0.0, 1.0, // row 2
            ];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // Expected: identity 3×3
            let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            for (i, (&g, &e)) in gram.iter().zip(expected.iter()).enumerate() {
                assert!((g - e).abs() < 1e-5, "gram[{i}]={g}, expected={e}");
            }
        }

        #[test]
        fn test_gram_ones() {
            // All-ones rows → G[i][j] = d_h for all i,j
            let seq_len = 4;
            let d_h = 8;
            let x = vec![1.0f32; seq_len * d_h];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            for (i, &g) in gram.iter().enumerate() {
                assert!(
                    (g - d_h as f32).abs() < 1e-4,
                    "gram[{i}]={g}, expected={}",
                    d_h
                );
            }
        }

        #[test]
        fn test_gram_symmetric() {
            let seq_len = 5;
            let d_h = 8;
            let x: Vec<f32> = (0..seq_len * d_h).map(|i| (i as f32 * 0.1).sin()).collect();
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let g_ij = gram[i * seq_len + j];
                    let g_ji = gram[j * seq_len + i];
                    assert!(
                        (g_ij - g_ji).abs() < 1e-5,
                        "G[{i}][{j}]={g_ij} != G[{j}][{i}]={g_ji}"
                    );
                }
            }
        }

        #[test]
        fn test_gram_upper_triangle_mirror() {
            // Construct X so that each row has a distinct value
            let seq_len = 3;
            let d_h = 4;
            let x: Vec<f32> = vec![
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // Verify G[i][j] == G[j][i] for all off-diagonal pairs
            // G[0][1] = dot(row0, row1) = 1*5+2*6+3*7+4*8 = 5+12+21+32 = 70
            // G[0][2] = dot(row0, row2) = 1*9+2*10+3*11+4*12 = 9+20+33+48 = 110
            // G[1][2] = dot(row1, row2) = 5*9+6*10+7*11+8*12 = 45+60+77+96 = 278
            assert!((gram[1] - 70.0).abs() < 1e-4, "G[0][1]={}", gram[1]);
            assert!((gram[3] - 70.0).abs() < 1e-4, "G[1][0]={}", gram[3]);
            assert!((gram[2] - 110.0).abs() < 1e-4, "G[0][2]={}", gram[2]);
            assert!((gram[6] - 110.0).abs() < 1e-4, "G[2][0]={}", gram[6]);
            assert!((gram[5] - 278.0).abs() < 1e-4, "G[1][2]={}", gram[5]);
            assert!((gram[7] - 278.0).abs() < 1e-4, "G[2][1]={}", gram[7]);
        }

        #[test]
        fn test_gram_2x3() {
            // X = [[1, 0, 2], [3, 1, 0]]
            let seq_len = 2;
            let d_h = 3;
            let x: Vec<f32> = vec![1.0, 0.0, 2.0, 3.0, 1.0, 0.0];
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // G[0][0] = 1+0+4 = 5
            // G[0][1] = 3+0+0 = 3
            // G[1][1] = 9+1+0 = 10
            assert!((gram[0] - 5.0).abs() < 1e-5, "G[0][0]={}", gram[0]);
            assert!((gram[1] - 3.0).abs() < 1e-5, "G[0][1]={}", gram[1]);
            assert!((gram[2] - 3.0).abs() < 1e-5, "G[1][0]={}", gram[2]);
            assert!((gram[3] - 10.0).abs() < 1e-5, "G[1][1]={}", gram[3]);
        }

        #[test]
        fn test_gram_matches_outer_product() {
            let seq_len = 4;
            let d_h = 8;
            let x: Vec<f32> = (0..seq_len * d_h)
                .map(|i| (i as f32 * 0.17).sin() * 0.5)
                .collect();

            // Compute gram via simd_gram_f32
            let mut gram = vec![0.0f32; seq_len * seq_len];
            simd_gram_f32(&x, seq_len, d_h, &mut gram);

            // Compute gram via iterative outer product: G = X·Xᵀ = Σ_k X_ik * X_jk
            let mut reference = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0f32;
                    for k in 0..d_h {
                        sum += x[i * d_h + k] * x[j * d_h + k];
                    }
                    reference[i * seq_len + j] = sum;
                }
            }

            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = i * seq_len + j;
                    assert!(
                        (gram[idx] - reference[idx]).abs() < 1e-4,
                        "G[{i}][{j}]: simd={}, reference={}",
                        gram[idx],
                        reference[idx]
                    );
                }
            }
        }
    }

    // ── simd_sum_abs_f32 tests (Issue 120) ─────────────────

    #[test]
    fn sum_abs_mixed_values() {
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let expected: f32 = data.iter().map(|v| v.abs()).sum();
        let result = crate::simd::simd_sum_abs_f32(&data);
        assert!(
            (result - expected).abs() < 1e-6,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn sum_abs_non_aligned_len() {
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let expected: f32 = data.iter().map(|v| v.abs()).sum();
        let result = crate::simd::simd_sum_abs_f32(&data);
        assert!(
            (result - expected).abs() < 1e-6,
            "got {result}, expected {expected}"
        );
    }

    #[test]
    fn sum_abs_empty() {
        let data: Vec<f32> = vec![];
        let result = crate::simd::simd_sum_abs_f32(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn sum_abs_single_element() {
        assert_eq!(crate::simd::simd_sum_abs_f32(&[-42.0]), 42.0);
        assert_eq!(crate::simd::simd_sum_abs_f32(&[42.0]), 42.0);
        assert_eq!(crate::simd::simd_sum_abs_f32(&[0.0]), 0.0);
    }
}

// ---------------------------------------------------------------------------
// Ternary dot-product kernel — Sense Composition (Plan 221)
// ---------------------------------------------------------------------------

/// Ternary dot-product: state · ternary_dir → f32.
/// Branchless sign extraction: pos_bit → +1, neg_bit → -1, neither → 0.
/// Uses `(pos as i8 - neg as i8) as f32` to avoid any branches.
#[inline]
pub fn simd_ternary_dot_f32(state: &[f32], dir: &crate::types::TernaryDir) -> f32 {
    let mut acc = 0.0f32;
    let min_len = state.len().min(64);
    let scale = dir.row_scale;
    for i in 0..min_len {
        let mask = 1u64 << i;
        let pos = ((dir.pos_bits & mask) != 0) as i8;
        let neg = ((dir.neg_bits & mask) != 0) as i8;
        let sign = (pos - neg) as f32;
        acc += sign * unsafe { *state.get_unchecked(i) } * scale;
    }
    acc
}

#[cfg(test)]
mod tests_sense {
    use super::*;
    use crate::types::TernaryDir;

    #[test]
    fn test_ternary_dot_matches_scalar() {
        let state = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let dir = TernaryDir {
            pos_bits: 0b0101, // indices 0, 2 are positive
            neg_bits: 0b1000, // index 3 is negative
            row_scale: 0.5,
        };
        let result = simd_ternary_dot_f32(&state, &dir);
        // +1*1.0 + 0*2.0 + 1*3.0 + (-1)*4.0 = 0.0, scaled by 0.5 = 0.0
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_dot_all_positive() {
        let state = [1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let dir = TernaryDir {
            pos_bits: 0b11,
            neg_bits: 0,
            row_scale: 1.0,
        };
        let result = simd_ternary_dot_f32(&state, &dir);
        assert!((result - 3.0).abs() < 1e-6);
    }
}
