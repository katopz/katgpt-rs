//! Ternary bit-plane matvec — multiplication-free CPU inference
//! (`plasma_path` feature, Plan 148). Includes scalar/NEON/AVX2 paths.

// ── Ternary SIMD Matvec (Plasma Path — Plan 148) ─────────

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(feature = "plasma_path")]
use super::{SimdLevel, simd_level};

// `horizontal_sum_256` is only used by the AVX2 ternary kernel, which itself
// requires both `plasma_path` and `x86_64`. Gate the import identically so
// other targets don't see an unused-import warning.
#[cfg(all(feature = "plasma_path", target_arch = "x86_64"))]
use super::horizontal::horizontal_sum_256;

#[cfg(feature = "plasma_path")]
use crate::TernaryWeights;

/// Scalar reference ternary matvec: y[r] = row_scale[r] * Σ(col → sign(pos_bit, neg_bit) * x[col])
#[cfg(feature = "plasma_path")]
#[allow(clippy::needless_range_loop)]
pub fn ternary_matvec_scalar(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    for r in 0..w.rows {
        let mut sum = 0.0f32;
        let row_base = r * w.blocks64;
        let mut col = 0usize;
        for b in 0..w.blocks64 {
            let pos_word = w.pos_bits[row_base + b];
            let neg_word = w.neg_bits[row_base + b];
            let remaining = (w.cols - col).min(64);
            for bit in 0..remaining {
                let mask = 1u64 << bit;
                let pos = (pos_word & mask) != 0;
                let neg = (neg_word & mask) != 0;
                let sign = pos as i32 - neg as i32;
                sum += sign as f32 * unsafe { *x.get_unchecked(col) };
                col += 1;
            }
        }
        y[r] = sum * w.row_scale[r];
    }
}

#[cfg(all(feature = "plasma_path", target_arch = "aarch64"))]
unsafe fn neon_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len()==w.cols and y.len()==w.rows
    unsafe {
        use core::arch::aarch64::{float32x4_t, uint32x4_t, *};
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        // SWAR bit-position masks: AND a splatted byte with these to isolate each of
        // the 4 bits in a nibble into its own lane (Issue 298, ported from validated POC).
        let mask_lo_arr: [u32; 4] = [1, 2, 4, 8];
        let mask_hi_arr: [u32; 4] = [16, 32, 64, 128];
        let mask_lo: uint32x4_t = vld1q_u32(mask_lo_arr.as_ptr());
        let mask_hi: uint32x4_t = vld1q_u32(mask_hi_arr.as_ptr());
        let one_u: uint32x4_t = vdupq_n_u32(1);

        for r in 0..w.rows {
            let row_base = r * w.blocks64;
            // 4 independent accumulators break the per-row serial dependency chain
            // and let the out-of-order engine overlap chunks.
            let mut acc0: float32x4_t = vdupq_n_f32(0.0);
            let mut acc1: float32x4_t = vdupq_n_f32(0.0);
            let mut acc2: float32x4_t = vdupq_n_f32(0.0);
            let mut acc3: float32x4_t = vdupq_n_f32(0.0);

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

                // 32-element unroll: 4 × 8-element chunks, each into its own accumulator.
                let chunks32 = remaining / 32 * 32;
                let mut col = 0usize;
                while col < chunks32 {
                    fmla_nibble8(
                        &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, one_u,
                    );
                    col += 8;
                    fmla_nibble8(
                        &mut acc1, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, one_u,
                    );
                    col += 8;
                    fmla_nibble8(
                        &mut acc2, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, one_u,
                    );
                    col += 8;
                    fmla_nibble8(
                        &mut acc3, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, one_u,
                    );
                    col += 8;
                }

                // Remaining 8-element chunks go into acc0
                while col + 8 <= remaining {
                    fmla_nibble8(
                        &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, one_u,
                    );
                    col += 8;
                }

                // Remaining 4-element chunk into acc0
                if col + 4 <= remaining {
                    let byte_off = col / 8;
                    let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as u32;
                    let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as u32;
                    let pos_splat = vdupq_n_u32(pos_byte);
                    let neg_splat = vdupq_n_u32(neg_byte);
                    // vcgeq_u32(., 1) → all-ones (-1 as i32) where bit set, 0 elsewhere.
                    let pos_nz =
                        vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
                    let neg_nz =
                        vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
                    // sign = neg - pos → +1 where pos, -1 where neg, 0 else.
                    let sign_f = vcvtq_f32_s32(vsubq_s32(neg_nz, pos_nz));
                    let x_v = vld1q_f32(x.as_ptr().add(base_col + col));
                    acc0 = vfmaq_f32(acc0, sign_f, x_v);
                    col += 4;
                }

                // Scalar tail (0-3 elements)
                let mut scalar_acc = 0.0f32;
                while col < remaining {
                    let bit_mask = 1u64 << col;
                    let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                    let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                    scalar_acc += (pos - neg) * *x.get_unchecked(base_col + col);
                    col += 1;
                }
                if scalar_acc != 0.0 {
                    acc0 = vaddq_f32(acc0, vsetq_lane_f32(scalar_acc, vdupq_n_f32(0.0), 0));
                }
            }

            // Merge 4 accumulators → horizontal sum.
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            y[r] = vaddvq_f32(acc0) * w.row_scale[r];
        }
    } // unsafe
}

/// NEON inner-loop helper for Issue 298: process 8 elements (one byte each of
/// pos/neg bit-planes, both nibbles) with SWAR mask construction + sign-FMLA.
///
/// `vcgeq_u32(and, 1)` returns all-ones (=-1 as i32) where the bit is set,
/// zero otherwise. Therefore `sign_i32 = neg_nz - pos_nz` gives exactly the
/// ternary sign `{-1, 0, +1}` per lane, and `vfmaq_f32(acc, sign_f, x)` fuses
/// the multiply-add into a single instruction — replacing the previous
/// `vbslq + vsub + vadd` (3 ops) chain.
#[cfg(all(feature = "plasma_path", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn fmla_nibble8(
    acc: &mut core::arch::aarch64::float32x4_t,
    pos_word: u64,
    neg_word: u64,
    col: usize,
    base_col: usize,
    x: &[f32],
    mask_lo: core::arch::aarch64::uint32x4_t,
    mask_hi: core::arch::aarch64::uint32x4_t,
    one_u: core::arch::aarch64::uint32x4_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let byte_off = col / 8;
        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as u32;
        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as u32;

        let pos_splat = vdupq_n_u32(pos_byte);
        let neg_splat = vdupq_n_u32(neg_byte);

        // Low nibble (cols [col..col+4]) and high nibble (cols [col+4..col+8]).
        let pos_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
        let neg_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
        let pos_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_hi), one_u));
        let neg_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_hi), one_u));

        // sign = neg - pos → +1 where pos, -1 where neg, 0 else.
        let sign_lo_f = vcvtq_f32_s32(vsubq_s32(neg_lo, pos_lo));
        let sign_hi_f = vcvtq_f32_s32(vsubq_s32(neg_hi, pos_hi));

        let x_lo = vld1q_f32(x.as_ptr().add(base_col + col));
        let x_hi = vld1q_f32(x.as_ptr().add(base_col + col + 4));

        // Single fused multiply-add per chunk (replaces vbslq + vsub + vadd).
        *acc = vfmaq_f32(*acc, sign_lo_f, x_lo);
        *acc = vfmaq_f32(*acc, sign_hi_f, x_hi);
    }
}

#[cfg(all(feature = "plasma_path", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len()==w.cols and y.len()==w.rows
    unsafe {
        use core::arch::x86_64::*;
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        // SWAR bit-position mask: AND a splatted byte with this isolates each of
        // the 8 bits in one byte into its own i32 lane (Issue 298, AVX2 port).
        let mask_byte: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_i: __m256i = _mm256_setzero_si256();

        for r in 0..w.rows {
            let row_base = r * w.blocks64;
            // 4 independent accumulators (break serial dep chain for OoO ILP).
            let mut acc0: __m256 = _mm256_setzero_ps();
            let mut acc1: __m256 = _mm256_setzero_ps();
            let mut acc2: __m256 = _mm256_setzero_ps();
            let mut acc3: __m256 = _mm256_setzero_ps();

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

                // 32-element unroll: 4 × 8-element chunks (one byte each), each
                // chunk into its own accumulator. Same pattern as the NEON port.
                let chunks32 = remaining / 32 * 32;
                let mut col = 0usize;
                while col < chunks32 {
                    fma_byte8_avx2(
                        &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                    );
                    col += 8;
                    fma_byte8_avx2(
                        &mut acc1, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                    );
                    col += 8;
                    fma_byte8_avx2(
                        &mut acc2, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                    );
                    col += 8;
                    fma_byte8_avx2(
                        &mut acc3, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                    );
                    col += 8;
                }

                // Remaining 8-element chunks → acc0
                while col + 8 <= remaining {
                    fma_byte8_avx2(
                        &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                    );
                    col += 8;
                }

                // Scalar tail (0-7 elements)
                let mut scalar_acc = 0.0f32;
                while col < remaining {
                    let bit_mask = 1u64 << col;
                    let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                    let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                    scalar_acc += (pos - neg) * *x.get_unchecked(base_col + col);
                    col += 1;
                }
                if scalar_acc != 0.0 {
                    acc0 = _mm256_add_ps(
                        acc0,
                        _mm256_setr_ps(scalar_acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    );
                }
            }

            // Merge 4 accumulators → horizontal sum.
            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            y[r] = horizontal_sum_256(acc0) * w.row_scale[r];
        }
    } // unsafe
}

/// AVX2 inner-loop helper for Issue 298: process 8 elements (one byte each of
/// pos/neg bit-planes) with SWAR mask construction + sign-FMA.
///
/// `_mm256_cmpgt_epi32(and, 0)` returns all-ones (=-1 as i32) where the bit is
/// set, zero otherwise. Therefore `sign_i = neg_nz - pos_nz` gives exactly the
/// ternary sign `{-1, 0, +1}` per lane, and `_mm256_fmadd_ps(sign_f, x, acc)`
/// fuses the multiply-add — replacing the previous `and + sub + add` chain.
#[cfg(all(feature = "plasma_path", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fma_byte8_avx2(
    acc: &mut core::arch::x86_64::__m256,
    pos_word: u64,
    neg_word: u64,
    col: usize,
    base_col: usize,
    x: &[f32],
    mask_byte: core::arch::x86_64::__m256i,
    zero_i: core::arch::x86_64::__m256i,
) {
    use core::arch::x86_64::*;
    unsafe {
        let byte_off = col / 8;
        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as i32;
        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as i32;

        let pos_splat = _mm256_set1_epi32(pos_byte);
        let neg_splat = _mm256_set1_epi32(neg_byte);

        // cmpgt_epi32(and, 0) → all-ones (-1) where bit set, zero elsewhere.
        let pos_nz = _mm256_cmpgt_epi32(_mm256_and_si256(pos_splat, mask_byte), zero_i);
        let neg_nz = _mm256_cmpgt_epi32(_mm256_and_si256(neg_splat, mask_byte), zero_i);

        // sign = neg - pos → +1 where pos, -1 where neg, 0 else.
        let sign_i = _mm256_sub_epi32(neg_nz, pos_nz);
        let sign_f = _mm256_cvtepi32_ps(sign_i);

        let x_v = _mm256_loadu_ps(x.as_ptr().add(base_col + col));

        // acc = sign_f * x_v + acc  (fused multiply-add).
        *acc = _mm256_fmadd_ps(sign_f, x_v, *acc);
    }
}

/// WASM SIMD128 ternary matvec — processes 4 f32 lanes per iteration using `v128`.
///
/// SWAR-optimized for Issue 298: ports the proven riir-engine implementation
/// (commit `23a2a8ff`, 4.34× speedup on wasmtime) plus 4 independent accumulators
/// for ILP. Bit-identical to `ternary_matvec_scalar()`.
///
/// Algorithm: broadcast the bit-plane byte to all 4 lanes of an `i32x4`, AND with
/// a bit-position mask `[1, 2, 4, 8]` (low nibble) or `[16, 32, 64, 128]` (high
/// nibble) to isolate each bit into its own lane, then `i32x4_ne(_, 0)` produces
/// a per-lane all-ones mask for `v128_bitselect`.
///
/// Compile-time gated by `target_feature = "simd128"` — requires `-C target-feature=+simd128`.
#[cfg(all(
    feature = "plasma_path",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
unsafe fn wasm32_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    // Safety: caller guarantees x.len()==w.cols and y.len()==w.rows
    unsafe {
        use core::arch::wasm32::*;
        assert_eq!(x.len(), w.cols);
        assert_eq!(y.len(), w.rows);

        // SWAR bit-position masks (ported from riir-engine proven impl).
        let mask_lo: v128 = i32x4(1, 2, 4, 8); // bits 0-3
        let mask_hi: v128 = i32x4(16, 32, 64, 128); // bits 4-7
        let zero_i: v128 = i32x4_splat(0);
        let zeros_f: v128 = f32x4_splat(0.0);

        for r in 0..w.rows {
            let row_base = r * w.blocks64;
            // 4 independent accumulators (break serial dep chain for ILP).
            let mut acc0: v128 = f32x4_splat(0.0);
            let mut acc1: v128 = f32x4_splat(0.0);
            let mut acc2: v128 = f32x4_splat(0.0);
            let mut acc3: v128 = f32x4_splat(0.0);

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

                // 32-element unroll: 4 × 8-element chunks, each into its own accumulator.
                let chunks32 = remaining / 32 * 32;
                let mut col = 0usize;
                while col < chunks32 {
                    bitselect_nibble8_wasm(
                        &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, zero_i,
                        zeros_f,
                    );
                    col += 8;
                    bitselect_nibble8_wasm(
                        &mut acc1, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, zero_i,
                        zeros_f,
                    );
                    col += 8;
                    bitselect_nibble8_wasm(
                        &mut acc2, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, zero_i,
                        zeros_f,
                    );
                    col += 8;
                    bitselect_nibble8_wasm(
                        &mut acc3, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, zero_i,
                        zeros_f,
                    );
                    col += 8;
                }

                // Remaining 8-element chunks → acc0
                while col + 8 <= remaining {
                    bitselect_nibble8_wasm(
                        &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi, zero_i,
                        zeros_f,
                    );
                    col += 8;
                }

                // Remaining 4-element chunk → acc0
                if col + 4 <= remaining {
                    let byte_off = col / 8;
                    let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as i32;
                    let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as i32;
                    let pos_splat = i32x4_splat(pos_byte);
                    let neg_splat = i32x4_splat(neg_byte);
                    let pos_m = i32x4_ne(v128_and(pos_splat, mask_lo), zero_i);
                    let neg_m = i32x4_ne(v128_and(neg_splat, mask_lo), zero_i);
                    let x_v = v128_load(x.as_ptr().add(base_col + col) as *const v128);
                    let pos_val = v128_bitselect(x_v, zeros_f, pos_m);
                    let neg_val = v128_bitselect(x_v, zeros_f, neg_m);
                    acc0 = f32x4_add(acc0, f32x4_sub(pos_val, neg_val));
                    col += 4;
                }

                // Scalar tail (0-3 elements)
                let mut scalar_acc = 0.0f32;
                while col < remaining {
                    let bit_mask = 1u64 << col;
                    let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                    let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                    scalar_acc += (pos - neg) * *x.get_unchecked(base_col + col);
                    col += 1;
                }
                if scalar_acc != 0.0 {
                    let scalar_arr: [f32; 4] = [scalar_acc, 0.0, 0.0, 0.0];
                    acc0 = f32x4_add(acc0, core::mem::transmute(scalar_arr));
                }
            }

            // Merge 4 accumulators → horizontal sum.
            acc0 = f32x4_add(f32x4_add(acc0, acc1), f32x4_add(acc2, acc3));
            let lanes: [f32; 4] = core::mem::transmute(acc0);
            y[r] = (lanes[0] + lanes[1] + lanes[2] + lanes[3]) * w.row_scale[r];
        }
    } // unsafe
}

/// WASM inner-loop helper for Issue 298: process 8 elements (one byte each of
/// pos/neg bit-planes, both nibbles) with SWAR mask construction + bitselect.
/// Direct port of the proven riir-engine `project_ternary_simd` inner loop.
#[cfg(all(
    feature = "plasma_path",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
#[inline(always)]
unsafe fn bitselect_nibble8_wasm(
    acc: &mut core::arch::wasm32::v128,
    pos_word: u64,
    neg_word: u64,
    col: usize,
    base_col: usize,
    x: &[f32],
    mask_lo: core::arch::wasm32::v128,
    mask_hi: core::arch::wasm32::v128,
    zero_i: core::arch::wasm32::v128,
    zeros_f: core::arch::wasm32::v128,
) {
    use core::arch::wasm32::*;
    unsafe {
        let byte_off = col / 8;
        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as i32;
        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as i32;

        let pos_splat = i32x4_splat(pos_byte);
        let neg_splat = i32x4_splat(neg_byte);

        // Low nibble (cols [col..col+4]) and high nibble (cols [col+4..col+8]).
        let pos_lo_m = i32x4_ne(v128_and(pos_splat, mask_lo), zero_i);
        let neg_lo_m = i32x4_ne(v128_and(neg_splat, mask_lo), zero_i);
        let pos_hi_m = i32x4_ne(v128_and(pos_splat, mask_hi), zero_i);
        let neg_hi_m = i32x4_ne(v128_and(neg_splat, mask_hi), zero_i);

        let x_lo = v128_load(x.as_ptr().add(base_col + col) as *const v128);
        let x_hi = v128_load(x.as_ptr().add(base_col + col + 4) as *const v128);

        let pos_lo_val = v128_bitselect(x_lo, zeros_f, pos_lo_m);
        let neg_lo_val = v128_bitselect(x_lo, zeros_f, neg_lo_m);
        let pos_hi_val = v128_bitselect(x_hi, zeros_f, pos_hi_m);
        let neg_hi_val = v128_bitselect(x_hi, zeros_f, neg_hi_m);

        *acc = f32x4_add(*acc, f32x4_sub(pos_lo_val, neg_lo_val));
        *acc = f32x4_add(*acc, f32x4_sub(pos_hi_val, neg_hi_val));
    }
}

/// SIMD-accelerated ternary matvec: y = W_ternary × x
///
/// Dispatches to NEON, AVX2, WASM-SIMD128, or scalar based on `simd_level()`.
/// All paths produce bit-identical results to `ternary_matvec_scalar()`.
#[cfg(feature = "plasma_path")]
#[inline]
pub fn simd_ternary_matvec(w: &TernaryWeights, x: &[f32], y: &mut [f32]) {
    match simd_level() {
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon_ternary_matvec(w, x, y) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { avx2_ternary_matvec(w, x, y) },
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        SimdLevel::WasmSimd128 => unsafe { wasm32_ternary_matvec(w, x, y) },
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

// ---------------------------------------------------------------------------
// Ternary dot-product kernel — Sense Composition (Plan 221)
// ---------------------------------------------------------------------------

/// Ternary dot-product: state · ternary_dir → f32.
/// Branchless sign extraction: pos_bit → +1, neg_bit → -1, neither → 0.
/// Uses `(pos as i8 - neg as i8) as f32` to avoid any branches.
#[inline]
pub fn simd_ternary_dot_f32(state: &[f32], dir: &crate::TernaryDir) -> f32 {
    let mut acc = 0.0f32;
    let min_len = state.len().min(64);
    for i in 0..min_len {
        let mask = 1u64 << i;
        let pos = ((dir.pos_bits & mask) != 0) as i8;
        let neg = ((dir.neg_bits & mask) != 0) as i8;
        let sign = (pos - neg) as f32;
        // FMA: acc = sign * state[i] + acc (single rounding).
        acc = sign.mul_add(unsafe { *state.get_unchecked(i) }, acc);
    }
    acc * dir.row_scale
}

// ---------------------------------------------------------------------------
// Single-vector ternary bit-plane projection (Plan 008 Step 7c — port from riir-engine)
// ---------------------------------------------------------------------------
//
// Projects an input vector against a single ternary {-1, 0, +1} row stored as
// two bit-planes (positive_bits, negative_bits). Returns the dot product
// `Σ_i sign(pos_bit_i, neg_bit_i) * input[i]`. This is the single-row analog
// of `simd_ternary_matvec` (which loops over rows of a `TernaryWeights`
// matrix); callers that only need one row avoid constructing the matrix.
//
// Ported verbatim from the proven riir-engine `simd::wasm32::project_ternary_simd`
// (Plan 286 T6/T7) so riir-engine can delete its local copy and consume
// `katgpt_core::simd::project_ternary_simd` directly (Plan 008 Step 7e).

/// Scalar reference: single-vector ternary bit-plane projection.
///
/// Computes `Σ_i sign(pos_bit_i, neg_bit_i) * input[i]` where sign is
/// `+1` if positive bit set, `-1` if negative bit set, `0` otherwise.
///
/// # Safety
/// Caller must ensure `input.len() >= n_elements` and bit-plane slices have
/// enough bytes to cover `n_elements` bits (`ceil(n_elements / 8)` bytes).
#[cfg(feature = "plasma_path")]
#[inline]
pub unsafe fn project_ternary_simd_scalar(
    input: &[f32],
    positive_bits: &[u8],
    negative_bits: &[u8],
    n_elements: usize,
) -> f32 {
    debug_assert!(input.len() >= n_elements);
    let mut sum = 0.0f32;
    for i in 0..n_elements {
        // SAFETY: caller guarantees input.len() >= n_elements > i, and
        // bit-plane slices cover ceil(n_elements/8) > i/8 bytes.
        unsafe {
            let byte_off = i / 8;
            let bit_off = i % 8;
            let is_pos = (*positive_bits.get_unchecked(byte_off) >> bit_off) & 1;
            let is_neg = (*negative_bits.get_unchecked(byte_off) >> bit_off) & 1;
            let sign: f32 = match (is_pos, is_neg) {
                (1, 0) => 1.0,
                (0, 1) => -1.0,
                _ => 0.0,
            };
            sum += sign * *input.get_unchecked(i);
        }
    }
    sum
}

/// Single-vector ternary bit-plane projection using WASM SIMD128 SWAR.
///
/// Projects an input vector against ternary {-1, 0, +1} weights stored as
/// bit-planes. Two bit-planes encode the sign: positive_bits and negative_bits.
/// Each bit position maps to one weight: (1,0)=+1, (0,1)=-1, (0,0)=0.
///
/// # Algorithm (SWAR bit-decoding)
///
/// Vectorizes the bit-decoding itself (vs scalar per-bit match):
/// 1. Broadcast the bit-plane byte to all 4 lanes of an `i32x4` via `i32x4_splat`.
/// 2. AND with a bit-position mask `[1, 2, 4, 8]` (or `[16, 32, 64, 128]` for the
///    high nibble) to isolate each of the 4 bits into its own lane.
/// 3. `i32x4_ne(_, 0)` produces a per-lane all-ones (-1) / all-zeros (0) mask.
/// 4. `v128_bitselect(input, zeros, mask)` picks the input value where the bit is
///    set, zero elsewhere.
/// 5. `acc += pos_val - neg_val` accumulates the signed contribution.
///
/// Processing 8 elements per iteration (both nibbles of one byte) doubles
/// throughput vs 4-element chunks and amortizes the byte broadcast.
///
/// # Safety
/// Caller must ensure `input.len() >= n_elements` and bit-plane slices have
/// enough bytes to cover `n_elements` bits (`ceil(n_elements / 8)` bytes).
#[cfg(all(
    feature = "plasma_path",
    target_arch = "wasm32",
    target_feature = "simd128"
))]
#[inline]
pub unsafe fn project_ternary_simd_wasm32(
    input: &[f32],
    positive_bits: &[u8],
    negative_bits: &[u8],
    n_elements: usize,
) -> f32 {
    use core::arch::wasm32::{
        f32x4_add, f32x4_extract_lane, f32x4_splat, f32x4_sub, i32x4, i32x4_ne, i32x4_splat,
        v128, v128_and, v128_bitselect, v128_load,
    };

    debug_assert!(input.len() >= n_elements);

    // SWAR bit-position masks: AND-ing a splatted byte with these isolates each
    // of the 4 bits in a nibble into its own i32 lane.
    let mask_lo: v128 = i32x4(1, 2, 4, 8); // bits 0-3
    let mask_hi: v128 = i32x4(16, 32, 64, 128); // bits 4-7
    let zero_i: v128 = i32x4_splat(0);
    let zeros_f: v128 = f32x4_splat(0.0);

    let mut acc = f32x4_splat(0.0);

    // Process 8 elements per iteration: one byte each of pos/neg bit-planes,
    // both nibbles. This is byte-aligned by construction (i is a multiple of 8).
    let chunks8 = n_elements / 8 * 8;
    let mut i = 0;
    while i < chunks8 {
        // SAFETY: caller guarantees input.len() >= n_elements > i+8, and
        // positive/negative_bits cover ceil(n_elements/8) > i/8 bytes.
        unsafe {
            let byte_off = i / 8;
            let pos_byte = *positive_bits.get_unchecked(byte_off) as i32;
            let neg_byte = *negative_bits.get_unchecked(byte_off) as i32;

            let pos_splat = i32x4_splat(pos_byte);
            let neg_splat = i32x4_splat(neg_byte);

            // Low nibble (bits 0-3) → 4 per-lane masks
            let pos_lo_m = i32x4_ne(v128_and(pos_splat, mask_lo), zero_i);
            let neg_lo_m = i32x4_ne(v128_and(neg_splat, mask_lo), zero_i);

            // High nibble (bits 4-7) → 4 per-lane masks
            let pos_hi_m = i32x4_ne(v128_and(pos_splat, mask_hi), zero_i);
            let neg_hi_m = i32x4_ne(v128_and(neg_splat, mask_hi), zero_i);

            // Load 8 input f32 values as two v128s
            let in_lo = v128_load(input.as_ptr().add(i) as *const v128);
            let in_hi = v128_load(input.as_ptr().add(i + 4) as *const v128);

            // Masked select: keep input value where bit is set, zero elsewhere.
            let pos_lo_val = v128_bitselect(in_lo, zeros_f, pos_lo_m);
            let neg_lo_val = v128_bitselect(in_lo, zeros_f, neg_lo_m);
            let pos_hi_val = v128_bitselect(in_hi, zeros_f, pos_hi_m);
            let neg_hi_val = v128_bitselect(in_hi, zeros_f, neg_hi_m);

            // acc += pos_val - neg_val  (sign ∈ {-1, 0, +1} × input)
            acc = f32x4_add(acc, f32x4_sub(pos_lo_val, neg_lo_val));
            acc = f32x4_add(acc, f32x4_sub(pos_hi_val, neg_hi_val));
        }

        i += 8;
    }

    // Handle a remaining 4-element chunk (n_elements % 8 >= 4).
    // i is a multiple of 8, so bit_off is always 0 here → use mask_lo directly.
    if i + 4 <= n_elements {
        // SAFETY: see above — bounds verified by caller precondition.
        unsafe {
            let byte_off = i / 8;
            let pos_byte = *positive_bits.get_unchecked(byte_off) as i32;
            let neg_byte = *negative_bits.get_unchecked(byte_off) as i32;

            let pos_splat = i32x4_splat(pos_byte);
            let neg_splat = i32x4_splat(neg_byte);

            let pos_m = i32x4_ne(v128_and(pos_splat, mask_lo), zero_i);
            let neg_m = i32x4_ne(v128_and(neg_splat, mask_lo), zero_i);

            let in_v = v128_load(input.as_ptr().add(i) as *const v128);
            let pos_val = v128_bitselect(in_v, zeros_f, pos_m);
            let neg_val = v128_bitselect(in_v, zeros_f, neg_m);
            acc = f32x4_add(acc, f32x4_sub(pos_val, neg_val));
        }

        i += 4;
    }

    // Horizontal reduce of the 4-lane f32x4 accumulator.
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);

    // Scalar tail (0-3 remaining elements).
    while i < n_elements {
        // SAFETY: caller precondition — bits/input cover this index.
        unsafe {
            let byte_off = i / 8;
            let bit_off = i % 8;
            let is_pos = (*positive_bits.get_unchecked(byte_off) >> bit_off) & 1;
            let is_neg = (*negative_bits.get_unchecked(byte_off) >> bit_off) & 1;
            let sign: f32 = match (is_pos, is_neg) {
                (1, 0) => 1.0,
                (0, 1) => -1.0,
                _ => 0.0,
            };
            sum += sign * *input.get_unchecked(i);
        }
        i += 1;
    }

    sum
}

/// Single-vector ternary bit-plane projection: `input · ternary_row → f32`.
///
/// Dispatches to the WASM SIMD128 SWAR kernel on `wasm32 +simd128`, else the
/// scalar reference. (NEON/AVX2 paths are not yet implemented; the scalar
/// path is the bit-exact reference and the WASM path was proven equivalent
/// in Plan 286 T6.) This is the single-row analog of [`simd_ternary_matvec`]
/// — callers that only need one row avoid constructing a `TernaryWeights`.
///
/// # Safety
/// Caller must ensure `input.len() >= n_elements` and that `positive_bits` /
/// `negative_bits` each have at least `ceil(n_elements / 8)` bytes.
#[cfg(feature = "plasma_path")]
#[inline]
pub unsafe fn project_ternary_simd(
    input: &[f32],
    positive_bits: &[u8],
    negative_bits: &[u8],
    n_elements: usize,
) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // SAFETY: forwarded to caller precondition (see function doc).
        unsafe {
            project_ternary_simd_wasm32(input, positive_bits, negative_bits, n_elements)
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        // SAFETY: forwarded to caller precondition (see function doc).
        unsafe {
            project_ternary_simd_scalar(input, positive_bits, negative_bits, n_elements)
        }
    }
}
