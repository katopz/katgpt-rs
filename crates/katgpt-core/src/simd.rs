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

/// SIMD capability level detected at runtime.
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
        // Runtime detection via cpuid
        let cpuid1 = unsafe { core::arch::x86_64::__cpuid(1) };
        let has_avx = (cpuid1.ecx & (1 << 28)) != 0;
        let has_fma = (cpuid1.ecx & (1 << 12)) != 0;

        let cpuid7 = unsafe { core::arch::x86_64::__cpuid(7) };
        let has_avx2 = (cpuid7.ebx & (1 << 5)) != 0;

        has_avx && has_fma && has_avx2
    }
}

// ── Dot Product ───────────────────────────────────────────────

/// SIMD-accelerated dot product: `Σ a[i] * b[i]` for `len` elements.
///
/// Dispatches to NEON, AVX2, or scalar based on compile-time target and
/// runtime CPU feature detection.
#[inline]
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

#[inline]
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
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks = len / 4;

        for _ in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
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
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps};

    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        let chunks = len / 8;

        for _ in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let prod = _mm256_mul_ps(va, vb);
            acc = _mm256_add_ps(acc, prod);
            i += 8;
        }

        let mut sum = horizontal_sum_256(acc);

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
#[inline]
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

#[inline]
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
        _mm256_add_ps, _mm256_broadcast_ss, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
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
                let prod = _mm256_mul_ps(va, vb);
                let vresult = _mm256_add_ps(vacc, prod);
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
#[inline]
pub fn simd_matvec(acc: &mut [f32], mat: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mat[r * cols..r * cols + cols];
        unsafe {
            *acc.get_unchecked_mut(r) = simd_dot_f32(row, vec, cols);
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

#[inline]
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
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        let chunks = len / 4;

        for _ in 0..chunks {
            // Convert 4 f16 → f32 scalarly, then vectorize the FMA.
            // Note: NEON vcvt_f32_f16 / fcvtl requires nightly or inline asm.
            // The scalar conversion via half::f16::to_f32() compiles to hardware
            // fcvt on Apple Silicon, so the per-element cost is ~1 cycle.
            // The FMA accumulation is still vectorized (4-wide).
            let w32 = [
                (*w.get_unchecked(i)).to_f32(),
                (*w.get_unchecked(i + 1)).to_f32(),
                (*w.get_unchecked(i + 2)).to_f32(),
                (*w.get_unchecked(i + 3)).to_f32(),
            ];
            let vw = vld1q_f32(w32.as_ptr());
            let vx = vld1q_f32(x.as_ptr().add(i));
            acc = vfmaq_f32(acc, vw, vx);
            i += 4;
        }

        let mut sum = vaddvq_f32(acc);
        // Handle remainder
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
#[inline]
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
#[inline]
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

#[inline]
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

/// SIMD-accelerated in-place add: `dst[i] += src[i]` for all `i`.
///
/// Used for residual connections in transformer forward pass (attn + MLP).
/// NEON: 4× f32 per `vaddq_f32`. AVX2: 8× f32 per `_mm256_add_ps`.
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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
/// # Returns
/// Vec of scores, one per pair.
///
/// # Feature flag
/// `maxsim` — Plan 080
#[cfg(feature = "maxsim")]
pub fn maxsim_score_packed(
    queries: &[f32],
    query_offsets: &[usize],
    documents: &[f32],
    doc_offsets: &[usize],
    pair_q_ids: &[usize],
    pair_d_ids: &[usize],
    dim: usize,
) -> Vec<f32> {
    let num_pairs = pair_q_ids.len();
    debug_assert_eq!(pair_d_ids.len(), num_pairs);
    debug_assert!(query_offsets.len() >= *pair_q_ids.iter().max().unwrap_or(&0) + 2);
    debug_assert!(doc_offsets.len() >= *pair_d_ids.iter().max().unwrap_or(&0) + 2);

    let mut results = Vec::with_capacity(num_pairs);
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
        results.push(maxsim_score(q_data, d_data, lq, ld, dim));
    }
    results
}

// ── Scalar Fallbacks (new primitives) ─────────────────────────

#[inline]
#[allow(dead_code)]
fn scalar_add_inplace(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
    }
}

#[inline]
#[allow(dead_code)]
fn scalar_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) = *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

#[inline]
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

#[inline]
#[allow(dead_code)]
fn scalar_fused_decay_write(dst: &mut [f32], decay: f32, src: &[f32], write: f32) {
    for i in 0..dst.len() {
        unsafe {
            *dst.get_unchecked_mut(i) =
                decay * *dst.get_unchecked(i) + write * *src.get_unchecked(i);
        }
    }
}

#[inline]
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
#[inline]
fn cephes_exp_scalar(x: f32) -> f32 {
    const LN2_HI: f32 = 6.931_457_5e-1; // 0x3f317200
    const LN2_LO: f32 = 1.428_606_8e-6; // 0x35bfbe8e
    const INV_LN2: f32 = std::f32::consts::LOG2_E; // 1.4426950408889634

    // Range reduction: n = round(x / ln2)
    let n = (x * INV_LN2).round() as i32;
    let g = x - n as f32 * LN2_HI - n as f32 * LN2_LO;

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

#[inline]
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

        let mut vmax = vld1q_f32(x.as_ptr());
        let mut i = 4;
        for _ in 1..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i));
            vmax = vmaxq_f32(vmax, vx);
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
        const LN2_HI: f32 = 6.9314575195e-01;
        const LN2_LO: f32 = 1.4286067773e-06;
        const INV_LN2: f32 = std::f32::consts::LOG2_E;
        const ROUND_NEAREST: i32 = 0x00;

        let v_inv_ln2 = _mm256_set1_ps(INV_LN2);
        let v_ln2_hi = _mm256_set1_ps(LN2_HI);
        let v_ln2_lo = _mm256_set1_ps(LN2_LO);
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
        vaddq_f32, vcvtq_s32_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vrndq_f32, vst1q_f32,
        vsubq_f32,
    };
    unsafe {
        const LN2_HI: f32 = 6.931_457_5e-1;
        const LN2_LO: f32 = 1.428_606_8e-6;
        const INV_LN2: f32 = std::f32::consts::LOG2_E;

        let v_inv_ln2 = vdupq_n_f32(INV_LN2);
        let v_ln2_hi = vdupq_n_f32(LN2_HI);
        let v_ln2_lo = vdupq_n_f32(LN2_LO);
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

            // 2^n via scalar bit manipulation (NEON lacks direct float-to-bits cast)
            let q_arr: [f32; 4] = core::mem::transmute(q);
            let n_arr: [i32; 4] = core::mem::transmute(vn_i);
            let mut result = [0.0f32; 4];
            for j in 0..4 {
                let n = n_arr[j];
                if n < -126 {
                    result[j] = 0.0;
                } else if n > 127 {
                    result[j] = f32::INFINITY;
                } else {
                    let bits = ((n + 127) as u32) << 23;
                    let scale = f32::from_bits(bits);
                    result[j] = scale * q_arr[j];
                }
            }
            let vresult = vld1q_f32(result.as_ptr());

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

        let mut vmax = _mm256_loadu_ps(x.as_ptr());
        let mut i = 8;
        for _ in 1..chunks {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            vmax = _mm256_max_ps(vmax, vx);
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
#[inline]
fn horizontal_max_256(v: core::arch::x86_64::__m256) -> f32 {
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
#[inline]
fn horizontal_sum_256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::{_mm_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps};
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        horizontal_sum_128(sum128)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn horizontal_sum_128(v: core::arch::x86_64::__m128) -> f32 {
    use core::arch::x86_64::{_mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_shuffle_ps};
    unsafe {
        let shuf = _mm_shuffle_ps(v, v, 0xB1);
        let sums = _mm_add_ps(v, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x2A);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
            for i in 0..queries.len() {
                queries[i] = fastrand::f32() * 2.0 - 1.0;
            }
            for i in 0..documents.len() {
                documents[i] = fastrand::f32() * 2.0 - 1.0;
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

            let packed = maxsim_score_packed(
                &queries,
                &query_offsets,
                &documents,
                &doc_offsets,
                &pair_q_ids,
                &pair_d_ids,
                dim,
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
}
