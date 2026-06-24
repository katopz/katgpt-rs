//! Sparse dot product and sparse matmul — gather-based kernels for the
//! "active-tokens-only" matmul path (typically <32 of D dim alive).

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
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe {
            wasm32_simd128_sparse_dot_f32(
                weight,
                row_off,
                active_indices,
                active_values,
                alive,
            )
        }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        scalar_sparse_dot_f32(weight, row_off, active_indices, active_values, alive)
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn scalar_sparse_dot_f32(
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
            // FMA: sum = weight[c] * val + sum (single rounding, matches SIMD path).
            sum =
                (*weight.get_unchecked(row_off + c)).mul_add(*active_values.get_unchecked(i), sum);
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

/// WASM SIMD128 sparse dot — 4-wide f32, gather + mul-add.
///
/// Issue 007: ports the NEON structure to `core::arch::wasm32`. WASM SIMD128
/// has no hardware gather intrinsic and no FMA intrinsic. The gather is
/// emulated via 4 scalar `f32x4_replace_lane` lane-fills (one per lane, indexing
/// into `weight` at the scattered positions); the FMA is emulated via
/// `f32x4_mul` + `f32x4_add` (the engine/wasmtime JIT fuses them when
/// profitable). Bit-identical accumulation order to the NEON kernel modulo FMA
/// contraction: the scalar reference uses `mul_add` (single rounding), WASM uses
/// separate mul+add (double rounding) — same divergence as
/// `wasm32_simd128_dot_f32` in `dot.rs`.
///
/// Compile-time gated by `target_feature = "simd128"`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_sparse_dot_f32(
    weight: &[f32],
    row_off: usize,
    active_indices: &[usize],
    active_values: &[f32],
    alive: usize,
) -> f32 {
    use core::arch::wasm32::{
        f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_replace_lane, f32x4_splat, v128_load,
    };

    unsafe {
        // Single accumulator (mirrors the NEON kernel's structure).
        let mut acc = f32x4_splat(0.0);
        let mut i = 0;
        let chunks = alive / 4;

        for _ in 0..chunks {
            // Gather 4 weight values from scattered indices into a v128.
            // WASM SIMD128 has no gather intrinsic, so fill lane-by-lane.
            let mut ww = f32x4_splat(0.0);
            ww = f32x4_replace_lane::<0>(
                ww,
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i)),
            );
            ww = f32x4_replace_lane::<1>(
                ww,
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 1)),
            );
            ww = f32x4_replace_lane::<2>(
                ww,
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 2)),
            );
            ww = f32x4_replace_lane::<3>(
                ww,
                *weight.get_unchecked(row_off + *active_indices.get_unchecked(i + 3)),
            );

            // Load 4 contiguous active values.
            let vv = v128_load(active_values.as_ptr().add(i).cast());

            // acc += ww * vv (no FMA intrinsic — separate mul+add).
            acc = f32x4_add(f32x4_mul(ww, vv), acc);
            i += 4;
        }

        // Horizontal reduce: 4 lanes → scalar.
        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        // Remainder tail (0..3 elements).
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
