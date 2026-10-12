//! Ternary bit-plane matvec with group-wise f16 scale — the `Q2_0_g128`
//! kernel (`ternary_group_scale` feature, Issue 578). Scalar / NEON / AVX2 paths.
//!
//! Combines the two shipped tiers' halves: the two bit-planes of
//! [`crate::TernaryWeights`] (so the zero state survives) with the per-128
//! f16 `group_scale` of [`crate::BinaryWeights`].
//!
//! # Kernel design
//!
//! [`GROUP_SIZE`] is 128 and blocks are 64 bits, so **a group is exactly two
//! whole `u64` blocks** — a group boundary never splits a word. The NEON path
//! walks groups, and within each group its two blocks, switching only the
//! scale splat.
//!
//! The group scale is folded **into the sign vector** (`±scale` instead of
//! `±1`), the same trick the binary kernel uses (Issue 145 Gate A): the 4
//! accums then span the entire row with no per-group reset and no per-group
//! horizontal sum, keeping the out-of-order pipeline fed. Cost is one extra
//! multiply per chunk versus the row-scale ternary kernel, which applies its
//! single scale once at the very end.
//!
//! # Scalar vs SIMD agreement is close, not bit-identical
//!
//! Unlike [`super::simd_ternary_matvec`] — where both paths apply one row scale
//! at the end and therefore agree bit-for-bit — the paths here associate the
//! scale differently:
//!
//! - scalar: `Σ_g scale_g · (Σ_{col∈g} sign·x)` — scale applied once per group
//! - NEON/AVX2: `Σ_g Σ_{col∈g} (scale_g·sign)·x` — scale folded per element
//!
//! These are equal in exact arithmetic but not in f32. Agreement is ~1e-6
//! relative (asserted in tests). The scalar form is the reference because it
//! matches how `Q2_0_g128` is defined and how llama.cpp dequantizes it.
//!
//! # AVX2 port (Issue 578 follow-up, 2026-08-11)
//!
//! Mirrors [`super::binary`]'s AVX2 kernel shape: 4 `__m256` accums (8 lanes
//! each) cover 32 elements per outer unroll, with an 8-byte SWAR mask
//! `_mm256_setr_epi32(1,2,4,8,16,32,64,128)` extracting each bit of a splatted
//! pos/neg byte into its own lane. Unlike the binary kernel (which can use the
//! `fmadd(neg_2scale, bs_f, neg_scale)` two-state identity because every
//! weight is ±scale), ternary has **three** states so the sign must be computed
//! explicitly: `sign_f = cvt(neg_set − pos_set)` → +1/0/−1, then scaled by
//! `vmul(sign_f, scale_v)` and accumulated with `fmadd`.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

#[cfg(all(
    feature = "ternary_group_scale",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
use super::SimdLevel;
#[cfg(feature = "ternary_group_scale")]
use super::simd_level;
#[cfg(feature = "ternary_group_scale")]
use crate::{GROUP_SIZE, TernaryGroupWeights};

/// Blocks per group. `GROUP_SIZE / 64` — exactly 2 at the shipped group size.
/// Consumed only by the arch kernels (neon on aarch64, avx2 on x86_64) — the
/// scalar/wasm32 paths walk `>> 6` directly, so on other triples this is dead
/// and must not compile.
#[cfg(all(
    feature = "ternary_group_scale",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
const BLOCKS_PER_GROUP: usize = GROUP_SIZE / 64;

/// Scalar reference: `y[r] = Σ_g group_scale[r,g] · Σ_{col∈g} sign(col) · x[col]`
///
/// `sign = +1` where the pos plane is set, `-1` where the neg plane is set,
/// `0` where neither is (the state `BinaryWeights` cannot represent).
#[cfg(feature = "ternary_group_scale")]
pub fn ternary_group_matvec_scalar(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    scalar_row_range(w, x, y, 0);
}

/// Scalar kernel over the row range `[row_offset, row_offset + y.len())`.
///
/// Split out of [`ternary_group_matvec_scalar`] so
/// [`simd_ternary_group_matvec_parallel`] can hand each rayon worker a
/// disjoint slice of `y`. Rows are fully independent — row `r` reads only
/// `pos_bits`/`neg_bits`/`group_scale` at its own offsets and writes only
/// `y[r]` — so this is a pure partition, bit-identical to the serial loop.
#[cfg(feature = "ternary_group_scale")]
fn scalar_row_range(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32], row_offset: usize) {
    for (i, y_slot) in y.iter_mut().enumerate() {
        let r = row_offset + i;
        let row_base = r * w.blocks64;
        let group_base = r * w.groups_per_row;
        let mut row_sum = 0.0f32;
        for g in 0..w.groups_per_row {
            let g_start = g * GROUP_SIZE;
            let g_end = (g_start + GROUP_SIZE).min(w.cols);
            let mut group_acc = 0.0f32;
            for col in g_start..g_end {
                let idx = row_base + (col >> 6);
                let mask = 1u64 << (col & 63);
                let pos = (w.pos_bits[idx] & mask) != 0;
                let neg = (w.neg_bits[idx] & mask) != 0;
                let sign = pos as i32 - neg as i32;
                group_acc += sign as f32 * unsafe { *x.get_unchecked(col) };
            }
            row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
        }
        *y_slot = row_sum;
    }
}

/// NEON entry point. Dispatches to the **hoisted-scale** kernel since Issue 583
/// (1.11–1.12× over the folded one, measured on M3 Max across 4 runs — see
/// [Bench 583](../../../../.benchmarks/583_scale_hoist_goat.md)). The folded
/// kernel is retained as [`neon_row_range`] so the A/B stays reproducible.
#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
unsafe fn neon_ternary_group_matvec(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(y.len(), w.rows);
    unsafe { neon_row_range_hoisted(w, x, y, 0) }
}

/// NEON kernel over the row range `[row_offset, row_offset + y.len())`, with the
/// group scale **folded into the sign vector**.
///
/// **Superseded as the production path by [`neon_row_range_hoisted`] (Issue
/// 583).** Retained because it is the baseline half of that A/B — Bench 583
/// measures the two against each other on the same container, so deleting it
/// would make the 1.11× claim unreproducible. Reachable via
/// [`simd_ternary_group_matvec_folded`].
///
/// See [`scalar_row_range`] for why partitioning by row is safe.
///
/// # Safety
/// Caller guarantees `x.len() == w.cols` and that
/// `row_offset + y.len() <= w.rows`.
#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
unsafe fn neon_row_range(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32], row_offset: usize) {
    unsafe {
        use core::arch::aarch64::{float32x4_t, uint32x4_t, *};
        assert_eq!(x.len(), w.cols);
        debug_assert!(row_offset + y.len() <= w.rows);

        // SWAR bit-position masks: AND a splatted byte with these to isolate
        // each bit of a nibble into its own lane (same construction as the
        // row-scale ternary kernel).
        let mask_lo_arr: [u32; 4] = [1, 2, 4, 8];
        let mask_hi_arr: [u32; 4] = [16, 32, 64, 128];
        let mask_lo: uint32x4_t = vld1q_u32(mask_lo_arr.as_ptr());
        let mask_hi: uint32x4_t = vld1q_u32(mask_hi_arr.as_ptr());
        let one_u: uint32x4_t = vdupq_n_u32(1);

        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let row_base = r * w.blocks64;
            let group_base = r * w.groups_per_row;

            // 4 accumulators span the ENTIRE row (no per-group reset) — the
            // scale rides inside the sign vector instead.
            let mut acc0: float32x4_t = vdupq_n_f32(0.0);
            let mut acc1: float32x4_t = vdupq_n_f32(0.0);
            let mut acc2: float32x4_t = vdupq_n_f32(0.0);
            let mut acc3: float32x4_t = vdupq_n_f32(0.0);

            for g in 0..w.groups_per_row {
                let scale = w.group_scale[group_base + g].to_f32();
                let scale_v: float32x4_t = vdupq_n_f32(scale);

                // A group is exactly BLOCKS_PER_GROUP whole blocks; the final
                // group of a ragged row may be short.
                let b_start = g * BLOCKS_PER_GROUP;
                let b_end = (b_start + BLOCKS_PER_GROUP).min(w.blocks64);

                for b in b_start..b_end {
                    let idx = row_base + b;
                    let pos_word = w.pos_bits[idx];
                    let neg_word = w.neg_bits[idx];

                    let base_col = b * 64;
                    let remaining = if base_col + 64 <= w.cols {
                        64
                    } else {
                        w.cols - base_col
                    };

                    // 32-element unroll: 4 × 8-element chunks, one per accumulator.
                    let mut col = 0usize;
                    while col + 32 <= remaining {
                        fmla_scaled_nibble8(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                        fmla_scaled_nibble8(
                            &mut acc1, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                        fmla_scaled_nibble8(
                            &mut acc2, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                        fmla_scaled_nibble8(
                            &mut acc3, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                    }

                    // Remaining 8-element chunks.
                    while col + 8 <= remaining {
                        fmla_scaled_nibble8(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u, scale_v,
                        );
                        col += 8;
                    }

                    // Remaining 4-element chunk.
                    if col + 4 <= remaining {
                        let byte_off = col / 8;
                        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as u32;
                        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as u32;
                        let pos_splat = vdupq_n_u32(pos_byte);
                        let neg_splat = vdupq_n_u32(neg_byte);
                        let pos_nz =
                            vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
                        let neg_nz =
                            vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
                        // vcgeq gives all-ones (-1 as i32) where set, so
                        // neg - pos yields +1 for pos, -1 for neg, 0 for neither.
                        let sign_f = vcvtq_f32_s32(vsubq_s32(neg_nz, pos_nz));
                        let scaled = vmulq_f32(sign_f, scale_v);
                        let x_v = vld1q_f32(x.as_ptr().add(base_col + col));
                        acc0 = vfmaq_f32(acc0, scaled, x_v);
                        col += 4;
                    }

                    // Scalar tail (0-3 elements).
                    let mut scalar_acc = 0.0f32;
                    while col < remaining {
                        let bit_mask = 1u64 << col;
                        let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                        let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                        scalar_acc += (pos - neg) * scale * *x.get_unchecked(base_col + col);
                        col += 1;
                    }
                    if scalar_acc != 0.0 {
                        acc0 = vaddq_f32(acc0, vsetq_lane_f32(scalar_acc, vdupq_n_f32(0.0), 0));
                    }
                }
            }

            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            *y_slot = vaddvq_f32(acc0);
        }
    }
}

/// NEON inner-loop helper: 8 elements (one byte of each bit-plane, both
/// nibbles) with SWAR mask construction, group scale folded into the sign.
///
/// Identical to the row-scale kernel's `fmla_nibble8` except for the
/// `vmulq_f32(sign_f, scale_v)` that turns `±1` into `±scale`.
#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn fmla_scaled_nibble8(
    acc: &mut core::arch::aarch64::float32x4_t,
    pos_word: u64,
    neg_word: u64,
    col: usize,
    base_col: usize,
    x: &[f32],
    mask_lo: core::arch::aarch64::uint32x4_t,
    mask_hi: core::arch::aarch64::uint32x4_t,
    one_u: core::arch::aarch64::uint32x4_t,
    scale_v: core::arch::aarch64::float32x4_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let byte_off = col / 8;
        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as u32;
        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as u32;

        let pos_splat = vdupq_n_u32(pos_byte);
        let neg_splat = vdupq_n_u32(neg_byte);

        let pos_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
        let neg_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
        let pos_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_hi), one_u));
        let neg_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_hi), one_u));

        // sign = neg - pos → +1 where pos, -1 where neg, 0 where neither.
        let sign_lo_f = vcvtq_f32_s32(vsubq_s32(neg_lo, pos_lo));
        let sign_hi_f = vcvtq_f32_s32(vsubq_s32(neg_hi, pos_hi));

        // Fold the group scale in: ±1 → ±scale (0 stays 0).
        let scaled_lo = vmulq_f32(sign_lo_f, scale_v);
        let scaled_hi = vmulq_f32(sign_hi_f, scale_v);

        let x_lo = vld1q_f32(x.as_ptr().add(base_col + col));
        let x_hi = vld1q_f32(x.as_ptr().add(base_col + col + 4));

        *acc = vfmaq_f32(*acc, scaled_lo, x_lo);
        *acc = vfmaq_f32(*acc, scaled_hi, x_hi);
    }
}

/// NEON inner-loop helper, **unscaled** variant (Issue 583).
///
/// Identical to [`fmla_scaled_nibble8`] minus the two `vmulq_f32` that fold the
/// group scale into the sign vector. The caller applies the scale once per group
/// instead — see [`neon_row_range_hoisted`].
#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn fmla_nibble8_unscaled(
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

        let pos_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
        let neg_lo = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
        let pos_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_hi), one_u));
        let neg_hi = vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_hi), one_u));

        // sign = neg - pos → +1 where pos, -1 where neg, 0 where neither.
        let sign_lo_f = vcvtq_f32_s32(vsubq_s32(neg_lo, pos_lo));
        let sign_hi_f = vcvtq_f32_s32(vsubq_s32(neg_hi, pos_hi));

        let x_lo = vld1q_f32(x.as_ptr().add(base_col + col));
        let x_hi = vld1q_f32(x.as_ptr().add(base_col + col + 4));

        *acc = vfmaq_f32(*acc, sign_lo_f, x_lo);
        *acc = vfmaq_f32(*acc, sign_hi_f, x_hi);
    }
}

/// NEON kernel with the group scale **hoisted** out of the inner loop (Issue 583).
///
/// The shipped [`neon_row_range`] folds the scale into every sign vector — one
/// `vmulq` per 4 lanes, 32 per 128-weight group — so its 4 accumulators can span
/// the whole row. This variant resets the accumulators per group, horizontally
/// sums once, and multiplies by the scale once: **1 `vmulq` + 1 `vaddvq` per
/// group instead of 32 `vmulq`**.
///
/// Issue 578 chose the fold to avoid the per-group reset and hsum, on the
/// reasoning that they serialize the pipeline. Bench 582 measured the opposite
/// on the trit tier, which is why this variant exists — see Issue 583 for the
/// A/B and the promotion decision.
///
/// Bonus: this association (one scale per group) is exactly what
/// [`ternary_group_matvec_scalar`] does, and how `Q2_0_g128` is defined, so the
/// NEON path lands closer to its own reference.
///
/// # Safety
/// Caller guarantees `x.len() == w.cols` and `row_offset + y.len() <= w.rows`.
#[cfg(all(feature = "ternary_group_scale", target_arch = "aarch64"))]
unsafe fn neon_row_range_hoisted(
    w: &TernaryGroupWeights,
    x: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    unsafe {
        use core::arch::aarch64::{float32x4_t, uint32x4_t, *};
        debug_assert_eq!(x.len(), w.cols);
        debug_assert!(row_offset + y.len() <= w.rows);

        let mask_lo_arr: [u32; 4] = [1, 2, 4, 8];
        let mask_hi_arr: [u32; 4] = [16, 32, 64, 128];
        let mask_lo: uint32x4_t = vld1q_u32(mask_lo_arr.as_ptr());
        let mask_hi: uint32x4_t = vld1q_u32(mask_hi_arr.as_ptr());
        let one_u: uint32x4_t = vdupq_n_u32(1);

        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let row_base = r * w.blocks64;
            let group_base = r * w.groups_per_row;
            let mut row_sum = 0.0f32;

            for g in 0..w.groups_per_row {
                // Accumulators are per-GROUP here, not per-row.
                let mut acc0: float32x4_t = vdupq_n_f32(0.0);
                let mut acc1: float32x4_t = vdupq_n_f32(0.0);
                let mut acc2: float32x4_t = vdupq_n_f32(0.0);
                let mut acc3: float32x4_t = vdupq_n_f32(0.0);
                let mut scalar_acc = 0.0f32;

                let b_start = g * BLOCKS_PER_GROUP;
                let b_end = (b_start + BLOCKS_PER_GROUP).min(w.blocks64);

                for b in b_start..b_end {
                    let idx = row_base + b;
                    let pos_word = w.pos_bits[idx];
                    let neg_word = w.neg_bits[idx];

                    let base_col = b * 64;
                    let remaining = if base_col + 64 <= w.cols {
                        64
                    } else {
                        w.cols - base_col
                    };

                    let mut col = 0usize;
                    while col + 32 <= remaining {
                        fmla_nibble8_unscaled(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u,
                        );
                        col += 8;
                        fmla_nibble8_unscaled(
                            &mut acc1, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u,
                        );
                        col += 8;
                        fmla_nibble8_unscaled(
                            &mut acc2, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u,
                        );
                        col += 8;
                        fmla_nibble8_unscaled(
                            &mut acc3, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u,
                        );
                        col += 8;
                    }

                    while col + 8 <= remaining {
                        fmla_nibble8_unscaled(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_lo, mask_hi,
                            one_u,
                        );
                        col += 8;
                    }

                    if col + 4 <= remaining {
                        let byte_off = col / 8;
                        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as u32;
                        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as u32;
                        let pos_splat = vdupq_n_u32(pos_byte);
                        let neg_splat = vdupq_n_u32(neg_byte);
                        let pos_nz =
                            vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(pos_splat, mask_lo), one_u));
                        let neg_nz =
                            vreinterpretq_s32_u32(vcgeq_u32(vandq_u32(neg_splat, mask_lo), one_u));
                        let sign_f = vcvtq_f32_s32(vsubq_s32(neg_nz, pos_nz));
                        let x_v = vld1q_f32(x.as_ptr().add(base_col + col));
                        acc0 = vfmaq_f32(acc0, sign_f, x_v);
                        col += 4;
                    }

                    // Scalar tail (0-3 elements) — unscaled, folded in below.
                    while col < remaining {
                        let bit_mask = 1u64 << col;
                        let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                        let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                        scalar_acc += (pos - neg) * *x.get_unchecked(base_col + col);
                        col += 1;
                    }
                }

                let group_acc = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
                    + scalar_acc;
                row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
            }

            *y_slot = row_sum;
        }
    }
}

/// The hoisted-scale NEON kernel, reachable directly.
///
/// Since Issue 583 this is what [`simd_ternary_group_matvec`] dispatches to on
/// aarch64, so the two are equivalent there; it stays public so Bench 583 can
/// name both halves of the A/B explicitly rather than relying on which one
/// happens to be wired up.
#[cfg(feature = "ternary_group_scale")]
pub fn simd_ternary_group_matvec_hoisted(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");

    #[cfg(target_arch = "aarch64")]
    {
        if matches!(simd_level(), SimdLevel::Neon) {
            unsafe { neon_row_range_hoisted(w, x, y, 0) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if super::is_avx2_fma_available() {
            unsafe { avx2_row_range_hoisted(w, x, y, 0) };
            return;
        }
    }
    scalar_row_range(w, x, y, 0);
}

/// The **folded**-scale NEON/AVX2 kernel — the pre-Issue-583 production path.
///
/// Kept reachable so the A/B in Bench 583 measures a live code path rather than
/// a deleted one. On x86_64 it is still the *production* path — the AVX2 leg of
/// the hoist question (Issue 583 T4) is unmeasured, so the dispatch there was
/// deliberately left alone.
#[cfg(feature = "ternary_group_scale")]
pub fn simd_ternary_group_matvec_folded(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");

    #[cfg(target_arch = "aarch64")]
    {
        if matches!(simd_level(), SimdLevel::Neon) {
            unsafe { neon_row_range(w, x, y, 0) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if super::is_avx2_fma_available() {
            unsafe { avx2_row_range(w, x, y, 0) };
            return;
        }
    }
    scalar_row_range(w, x, y, 0);
}

// ── AVX2 (x86_64) ────────────────────────────────────────────────────
//
// Mirrors `avx2_binary_matvec` in [`super::binary`] but with two bit-planes
// (pos / neg) and the three-state sign (`+1`/`0`/`−1`). Validated on Haswell+
// via `is_avx2_fma_available` runtime detection (Issue 578 follow-up,
// 2026-08-11).

#[cfg(all(feature = "ternary_group_scale", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_ternary_group_matvec(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(y.len(), w.rows);
    unsafe { avx2_row_range(w, x, y, 0) }
}

/// AVX2 kernel over the row range `[row_offset, row_offset + y.len())`.
///
/// See [`scalar_row_range`] for why partitioning by row is safe.
///
/// # Safety
/// Caller guarantees `x.len() == w.cols` and that
/// `row_offset + y.len() <= w.rows`.
#[cfg(all(feature = "ternary_group_scale", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_row_range(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32], row_offset: usize) {
    use super::horizontal::horizontal_sum_256;
    use core::arch::x86_64::*;
    unsafe {
        assert_eq!(x.len(), w.cols);
        debug_assert!(row_offset + y.len() <= w.rows);

        // 8-bit SWAR mask: lane i holds bit i (1,2,4,8,16,32,64,128). ANDing
        // a splatted byte with this isolates each bit into its own lane.
        let mask_byte: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_i: __m256i = _mm256_setzero_si256();

        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let row_base = r * w.blocks64;
            let group_base = r * w.groups_per_row;

            // 4 accumulators × 8 lanes each span the ENTIRE row (no per-group
            // reset) — the scale rides inside the sign vector instead.
            let mut acc0: __m256 = _mm256_setzero_ps();
            let mut acc1: __m256 = _mm256_setzero_ps();
            let mut acc2: __m256 = _mm256_setzero_ps();
            let mut acc3: __m256 = _mm256_setzero_ps();

            for g in 0..w.groups_per_row {
                let scale = w.group_scale[group_base + g].to_f32();
                let scale_v: __m256 = _mm256_set1_ps(scale);

                // A group is exactly BLOCKS_PER_GROUP whole blocks; the final
                // group of a ragged row may be short.
                let b_start = g * BLOCKS_PER_GROUP;
                let b_end = (b_start + BLOCKS_PER_GROUP).min(w.blocks64);

                for b in b_start..b_end {
                    let idx = row_base + b;
                    let pos_word = w.pos_bits[idx];
                    let neg_word = w.neg_bits[idx];

                    let base_col = b * 64;
                    let remaining = if base_col + 64 <= w.cols {
                        64
                    } else {
                        w.cols - base_col
                    };

                    // 32-element unroll: 4 × 8-element chunks, one per accumulator.
                    let mut col = 0usize;
                    while col + 32 <= remaining {
                        fma_scaled_nibble8_avx2(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                        fma_scaled_nibble8_avx2(
                            &mut acc1, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                        fma_scaled_nibble8_avx2(
                            &mut acc2, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                        fma_scaled_nibble8_avx2(
                            &mut acc3, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                    }

                    // Remaining 8-element chunks.
                    while col + 8 <= remaining {
                        fma_scaled_nibble8_avx2(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                            scale_v,
                        );
                        col += 8;
                    }

                    // Scalar tail (0-7 elements).
                    let mut scalar_acc = 0.0f32;
                    while col < remaining {
                        let bit_mask = 1u64 << col;
                        let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                        let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                        scalar_acc += (pos - neg) * scale * *x.get_unchecked(base_col + col);
                        col += 1;
                    }
                    if scalar_acc != 0.0 {
                        acc0 = _mm256_add_ps(
                            acc0,
                            _mm256_setr_ps(scalar_acc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        );
                    }
                }
            }

            acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            *y_slot = horizontal_sum_256(acc0);
        }
    }
}

/// AVX2 inner-loop helper: 8 elements (one byte of each bit-plane) with SWAR
/// mask construction, group scale folded into the sign.
///
/// Sign computation differs from [`super::binary`]'s `fma_scaled_byte8_avx2`:
/// the binary kernel uses a two-state FMA identity
/// (`fmadd(neg_2scale, bs_f, neg_scale)` → ±scale), but ternary has three
/// states so the sign must be computed explicitly:
/// `sign_f = cvt(neg_set − pos_set)` → +1/0,−1, then `scaled = sign_f · scale`.
///
/// `pub(super)`: also consumed by [`super::bitcos`]'s pdep arm — the decode
/// (presence+signs → planes) differs, the dot from reconstructed planes is
/// byte-for-byte the same arithmetic, and a second copy would be a second
/// thing to get subtly wrong.
///
/// No `#[target_feature]` here — it would conflict with `#[inline(always)]`
/// (rust-lang/rust#145574). The `avx2,fma` feature is in force on the caller,
/// which is the only caller, so the body still compiles against the AVX2
/// intrinsics. Same shape as [`super::binary`]'s helper.
#[cfg(all(feature = "ternary_group_scale", target_arch = "x86_64"))]
#[inline(always)]
pub(super) unsafe fn fma_scaled_nibble8_avx2(
    acc: &mut core::arch::x86_64::__m256,
    pos_word: u64,
    neg_word: u64,
    col: usize,
    base_col: usize,
    x: &[f32],
    mask_byte: core::arch::x86_64::__m256i,
    zero_i: core::arch::x86_64::__m256i,
    scale_v: core::arch::x86_64::__m256,
) {
    use core::arch::x86_64::*;
    unsafe {
        let byte_off = col / 8;
        let pos_byte = ((pos_word >> (byte_off * 8)) & 0xFF) as i32;
        let neg_byte = ((neg_word >> (byte_off * 8)) & 0xFF) as i32;

        let pos_splat = _mm256_set1_epi32(pos_byte);
        let neg_splat = _mm256_set1_epi32(neg_byte);

        // cmpgt(and, 0) → all-ones (-1 as i32) where the bit is set, 0 where clear.
        let pos_set_i = _mm256_cmpgt_epi32(_mm256_and_si256(pos_splat, mask_byte), zero_i);
        let neg_set_i = _mm256_cmpgt_epi32(_mm256_and_si256(neg_splat, mask_byte), zero_i);

        // sign = neg_set − pos_set → +1 where pos, −1 where neg, 0 where neither.
        // (−1 − (−1) = 0; 0 − (−1) = +1; −1 − 0 = −1.)
        let sign_f = _mm256_cvtepi32_ps(_mm256_sub_epi32(neg_set_i, pos_set_i));

        // Fold the group scale in: ±1 → ±scale (0 stays 0).
        let scaled = _mm256_mul_ps(sign_f, scale_v);

        let x_v = _mm256_loadu_ps(x.as_ptr().add(base_col + col));
        *acc = _mm256_fmadd_ps(scaled, x_v, *acc);
    }
}

/// AVX2 inner-loop helper, **unscaled** variant (Issue 583 T4).
///
/// [`fma_scaled_nibble8_avx2`] minus the `_mm256_mul_ps` that folds the group
/// scale into the sign vector; the caller applies the scale once per group.
#[cfg(all(feature = "ternary_group_scale", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn fma_nibble8_avx2_unscaled(
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

        let pos_set_i = _mm256_cmpgt_epi32(_mm256_and_si256(pos_splat, mask_byte), zero_i);
        let neg_set_i = _mm256_cmpgt_epi32(_mm256_and_si256(neg_splat, mask_byte), zero_i);

        // sign = neg_set − pos_set → +1 where pos, −1 where neg, 0 where neither.
        let sign_f = _mm256_cvtepi32_ps(_mm256_sub_epi32(neg_set_i, pos_set_i));

        let x_v = _mm256_loadu_ps(x.as_ptr().add(base_col + col));
        *acc = _mm256_fmadd_ps(sign_f, x_v, *acc);
    }
}

/// AVX2 kernel with the group scale **hoisted** out of the inner loop
/// (Issue 583 T4) — the x86_64 mirror of [`neon_row_range_hoisted`].
///
/// 1 `_mm256_mul_ps` + 1 horizontal sum per 128-weight group, instead of 16
/// `_mm256_mul_ps` (one per 8 lanes). The NEON A/B measured 1.11–1.12× for this
/// restructuring on M3 Max; **whether it also wins on AVX2 is a separate
/// measurement** — this host cannot run it, so the kernel ships beside the
/// folded one and the dispatch is NOT switched until a 4090 run says so.
///
/// # Safety
/// Caller guarantees `x.len() == w.cols` and `row_offset + y.len() <= w.rows`.
#[cfg(all(feature = "ternary_group_scale", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_row_range_hoisted(
    w: &TernaryGroupWeights,
    x: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    use super::horizontal::horizontal_sum_256;
    use core::arch::x86_64::*;
    unsafe {
        assert_eq!(x.len(), w.cols);
        debug_assert!(row_offset + y.len() <= w.rows);

        let mask_byte: __m256i = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let zero_i: __m256i = _mm256_setzero_si256();

        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let row_base = r * w.blocks64;
            let group_base = r * w.groups_per_row;
            let mut row_sum = 0.0f32;

            for g in 0..w.groups_per_row {
                // Per-GROUP accumulators, unscaled.
                let mut acc0: __m256 = _mm256_setzero_ps();
                let mut acc1: __m256 = _mm256_setzero_ps();
                let mut acc2: __m256 = _mm256_setzero_ps();
                let mut acc3: __m256 = _mm256_setzero_ps();
                let mut scalar_acc = 0.0f32;

                let b_start = g * BLOCKS_PER_GROUP;
                let b_end = (b_start + BLOCKS_PER_GROUP).min(w.blocks64);

                for b in b_start..b_end {
                    let idx = row_base + b;
                    let pos_word = w.pos_bits[idx];
                    let neg_word = w.neg_bits[idx];

                    let base_col = b * 64;
                    let remaining = if base_col + 64 <= w.cols {
                        64
                    } else {
                        w.cols - base_col
                    };

                    let mut col = 0usize;
                    while col + 32 <= remaining {
                        fma_nibble8_avx2_unscaled(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                        );
                        col += 8;
                        fma_nibble8_avx2_unscaled(
                            &mut acc1, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                        );
                        col += 8;
                        fma_nibble8_avx2_unscaled(
                            &mut acc2, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                        );
                        col += 8;
                        fma_nibble8_avx2_unscaled(
                            &mut acc3, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                        );
                        col += 8;
                    }

                    while col + 8 <= remaining {
                        fma_nibble8_avx2_unscaled(
                            &mut acc0, pos_word, neg_word, col, base_col, x, mask_byte, zero_i,
                        );
                        col += 8;
                    }

                    // Scalar tail (0-7 elements) — unscaled, folded in below.
                    while col < remaining {
                        let bit_mask = 1u64 << col;
                        let pos = ((pos_word & bit_mask) != 0) as i32 as f32;
                        let neg = ((neg_word & bit_mask) != 0) as i32 as f32;
                        scalar_acc += (pos - neg) * *x.get_unchecked(base_col + col);
                        col += 1;
                    }
                }

                acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let group_acc = horizontal_sum_256(acc0) + scalar_acc;
                row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
            }

            *y_slot = row_sum;
        }
    }
}

/// Group-scale ternary matvec: `y = W × x`.
///
/// Dispatches to NEON / AVX2 where available, scalar otherwise.
///
/// The AVX2 port (Issue 578 follow-up, 2026-08-11) is gated on
/// [`super::is_avx2_fma_available`] at runtime so the binary stays portable to
/// pre-Haswell x86_64. The kernel mirrors [`super::binary`]'s AVX2 shape but
/// computes the ternary sign explicitly (`cvt(neg_set − pos_set)` → ±1/0)
/// rather than via the binary kernel's two-state FMA identity.
#[cfg(feature = "ternary_group_scale")]
#[inline]
pub fn simd_ternary_group_matvec(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    match simd_level() {
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => unsafe { neon_ternary_group_matvec(w, x, y) },
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => unsafe { avx2_ternary_group_matvec(w, x, y) },
        _ => ternary_group_matvec_scalar(w, x, y),
    }
}

/// Row-parallel group-scale ternary matvec: `y = W × x`, rows split across
/// rayon workers.
///
/// ## Why this exists (Issue 594 pre-flight, 2026-08-10)
///
/// [`simd_ternary_group_matvec`] is single-threaded, and the only rayon in this
/// tier was [`simd_ternary_group_matmul_batch`], which parallelizes over
/// **batch**. At `batch = 1` — i.e. every autoregressive decode step — that
/// gave *no* parallelism at all, so a 27 B ternary model decoded at **0.25
/// tok/s** on an M3 Max (measured: riir-ai
/// `bench_594_ternary_bonsai_throughput`) against llama.cpp Metal's 20-30.
///
/// The dense tier has had [`crate::math::matmul_parallel`] for this all along;
/// the ternary tier simply never grew the equivalent.
///
/// Tasks are capped at `PAR_MATVEC_CHUNK_ROWS` rows (Issue 935) so
/// work-stealing balances P- and E-cores.
///
/// ## Correctness
///
/// Rows are fully independent: row `r` reads only its own slices of
/// `pos_bits` / `neg_bits` / `group_scale`, and writes only `y[r]`. Splitting
/// `y` with `par_chunks_mut` is therefore a pure partition — the result is
/// **bit-identical** to the serial kernel, not merely close. Asserted by
/// `parallel_matches_serial_bit_identical`.
///
/// Below `PARALLEL_ROW_MIN` rows this delegates to the serial kernel: rayon's
/// per-task overhead (~1-5 µs) would otherwise dominate. That threshold matters
/// for real shapes — Bonsai's `ssm_alpha`/`ssm_beta` are only 48 rows.
#[cfg(feature = "ternary_group_scale")]
pub fn simd_ternary_group_matvec_parallel(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    /// Below this row count, rayon's task overhead outweighs the work.
    const PARALLEL_ROW_MIN: usize = 256;

    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");

    if w.rows < PARALLEL_ROW_MIN {
        simd_ternary_group_matvec(w, x, y);
        return;
    }

    simd_ternary_group_matvec_parallel_chunked(w, x, y, super::par_matvec_chunk(w.rows));
}

/// [`simd_ternary_group_matvec_parallel`] with an explicit rows-per-task
/// `chunk` — bit-identical for every `chunk >= 1` (a pure row partition).
#[cfg(feature = "ternary_group_scale")]
pub fn simd_ternary_group_matvec_parallel_chunked(
    w: &TernaryGroupWeights,
    x: &[f32],
    y: &mut [f32],
    chunk: usize,
) {
    use rayon::prelude::*;

    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    let chunk = chunk.max(1);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, y_chunk)| {
            let row_offset = ci * chunk;
            match simd_level() {
                #[cfg(target_arch = "aarch64")]
                // Hoisted since Issue 583 — must match the serial dispatch
                // above, or the parallel-vs-serial bit-identity gate breaks.
                SimdLevel::Neon => unsafe { neon_row_range_hoisted(w, x, y_chunk, row_offset) },
                #[cfg(target_arch = "x86_64")]
                SimdLevel::Avx2 => unsafe { avx2_row_range(w, x, y_chunk, row_offset) },
                _ => scalar_row_range(w, x, y_chunk, row_offset),
            }
        });
}

/// Batched group-scale ternary matmul: for each `batch[i]`, `y[i] = W × x[i]`.
///
/// Mirrors `simd_ternary_matmul_batch` / `simd_binary_matmul_batch`.
#[cfg(feature = "ternary_group_scale")]
pub fn simd_ternary_group_matmul_batch(
    w: &TernaryGroupWeights,
    x: &[f32],
    batch: usize,
    y: &mut [f32],
) {
    /// Below this, rayon's per-task overhead (~1-5µs) outweighs the work.
    const PARALLEL_BATCH_MIN: usize = 4;

    use rayon::prelude::*;

    if batch < PARALLEL_BATCH_MIN {
        for b in 0..batch {
            let x_off = b * w.cols;
            let y_off = b * w.rows;
            // Slice EXACTLY — the matvec asserts x.len() == w.cols, so an
            // open-ended `&x[x_off..]` panics for 2 <= batch < 4.
            simd_ternary_group_matvec(w, &x[x_off..x_off + w.cols], &mut y[y_off..y_off + w.rows]);
        }
        return;
    }
    y.par_chunks_mut(w.rows)
        .zip(x.par_chunks(w.cols))
        .enumerate()
        .for_each(|(b, (y_chunk, x_chunk))| {
            if b < batch {
                simd_ternary_group_matvec(w, x_chunk, y_chunk);
            }
        });
}

#[cfg(all(test, feature = "ternary_group_scale"))]
mod tests {
    use super::*;
    use crate::TernaryGroupWeights;

    /// Deterministic pseudo-random f32 in [-1, 1). No rand dep, reproducible.
    fn pseudo(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn filled(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        let mut s = seed;
        let mut w = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let v = pseudo(&mut s);
                let q = match v {
                    v if v > 0.33 => 1i8,
                    v if v < -0.33 => -1i8,
                    _ => 0i8,
                };
                w.set(r, c, q);
            }
            for g in 0..w.groups_per_row {
                // Vary scale per group so a per-row-scale bug cannot pass.
                w.set_scale(r, g, 0.5 + 0.25 * (g % 4) as f32);
            }
        }
        w
    }

    #[test]
    fn set_get_roundtrips_all_three_states() {
        let mut w = TernaryGroupWeights::new(3, 200);
        for c in 0..200 {
            w.set(0, c, 1);
            w.set(1, c, -1);
            w.set(2, c, 0);
        }
        for c in 0..200 {
            assert_eq!(w.get(0, c), 1);
            assert_eq!(w.get(1, c), -1);
            assert_eq!(w.get(2, c), 0);
        }
        // Overwrites must clear the other plane, not OR into it.
        w.set(0, 5, -1);
        assert_eq!(w.get(0, 5), -1);
        w.set(0, 5, 0);
        assert_eq!(w.get(0, 5), 0);
        assert!(w.invariant_holds());
    }

    #[test]
    fn group_geometry_matches_group_size() {
        let w = TernaryGroupWeights::new(2, 300);
        assert_eq!(GROUP_SIZE, 128);
        // BLOCKS_PER_GROUP is gated to the arch kernels' triples (aarch64/x86_64);
        // assert its VALUE everywhere via the definition instead of the const.
        assert_eq!(GROUP_SIZE / 64, 2);
        assert_eq!(w.blocks64, 300usize.div_ceil(64));
        assert_eq!(w.groups_per_row, 300usize.div_ceil(128));
        // 2.125 bits/weight: 2 planes * 8B/64w + 2B/128w.
        let w2 = TernaryGroupWeights::new(1, 128);
        assert_eq!(w2.encoded_bytes(), 8 * 2 + 8 * 2 + 2);
    }

    #[test]
    fn neon_matches_scalar_reference() {
        // Cover exact multiples of GROUP_SIZE and ragged tails (partial group,
        // partial block, and a sub-4 scalar tail).
        for &(rows, cols) in &[(4, 128), (3, 256), (2, 300), (5, 65), (1, 7), (2, 129)] {
            let w = filled(rows, cols, 0x51ED_u64.wrapping_add(cols as u64));
            let mut s = 0xABCD_u64;
            let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();

            let mut y_scalar = vec![0.0f32; rows];
            let mut y_simd = vec![0.0f32; rows];
            ternary_group_matvec_scalar(&w, &x, &mut y_scalar);
            simd_ternary_group_matvec(&w, &x, &mut y_simd);

            for r in 0..rows {
                let denom = y_scalar[r].abs().max(1.0);
                let rel = (y_scalar[r] - y_simd[r]).abs() / denom;
                assert!(
                    rel < 1e-6,
                    "rows={rows} cols={cols} r={r}: scalar={} simd={} rel={rel:e}",
                    y_scalar[r],
                    y_simd[r]
                );
            }
        }
    }

    #[test]
    fn zero_state_is_actually_skipped() {
        // All-zero weights must produce exactly 0 regardless of scale — the
        // state BinaryWeights cannot represent.
        let mut w = TernaryGroupWeights::new(2, 256);
        for r in 0..2 {
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 3.0);
            }
        }
        let x = vec![1.0f32; 256];
        let mut y = vec![9.9f32; 2];
        simd_ternary_group_matvec(&w, &x, &mut y);
        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn per_group_scale_is_applied_not_per_row() {
        // Row of all +1 with distinct per-group scales against x = 1:
        // y = Σ_g scale_g * group_len. A row-scale implementation cannot match.
        let cols = 256; // exactly 2 groups
        let mut w = TernaryGroupWeights::new(1, cols);
        for c in 0..cols {
            w.set(0, c, 1);
        }
        w.set_scale(0, 0, 1.0);
        w.set_scale(0, 1, 4.0);
        let x = vec![1.0f32; cols];
        let mut y = vec![0.0f32; 1];
        simd_ternary_group_matvec(&w, &x, &mut y);
        assert!((y[0] - (128.0 + 512.0)).abs() < 1e-3, "got {}", y[0]);
    }

    #[test]
    fn quantize_from_f32_holds_invariant_and_tracks_signs() {
        let (rows, cols) = (3, 300);
        let mut s = 0x1234_u64;
        let src: Vec<f32> = (0..rows * cols).map(|_| pseudo(&mut s) * 2.0).collect();
        let w = TernaryGroupWeights::quantize_from_f32(&src, rows, cols);
        assert!(w.invariant_holds());
        assert_eq!(w.groups_per_row, cols.div_ceil(GROUP_SIZE));
        // Large-magnitude inputs must not quantize to zero.
        let mut big = vec![0.0f32; cols];
        big[0] = 10.0;
        big[1] = -10.0;
        let wb = TernaryGroupWeights::quantize_from_f32(&big, 1, cols);
        assert_eq!(wb.get(0, 0), 1);
        assert_eq!(wb.get(0, 1), -1);
    }

    /// The reason `Q2_0_g128` exists: a scale per 128 weights tracks a signal
    /// whose magnitude varies along the row better than one scale per row.
    ///
    /// Asserted as a *comparison*, not an absolute bound. In an error-feedback
    /// quantizer the sum-reconstruction error equals the final carry
    /// (`Σ q·scale = Σ w - carry_final`), and the carry grows wherever `|w|`
    /// runs above the local scale — so the absolute error is signal-dependent
    /// and not worth pinning. What must hold is that finer scale granularity
    /// does not do *worse*.
    #[cfg(feature = "plasma_path")]
    #[test]
    fn group_scale_tracks_varying_magnitude_better_than_row_scale() {
        // Magnitude ramps 10x across the row: group 0 is small, group 1 large.
        // A single row scale must compromise between them; per-group cannot.
        let cols = 256;
        let src: Vec<f32> = (0..cols)
            .map(|i| {
                let amp = if i < 128 { 0.1 } else { 1.0 };
                amp * (i as f32 / 8.0).sin()
            })
            .collect();

        let gw = TernaryGroupWeights::quantize_from_f32(&src, 1, cols);
        let rw = crate::TernaryWeights::quantize_from_f32(&src, 1, cols);
        assert!(gw.invariant_holds());

        // Reconstruct each weight and compare against the source elementwise.
        let err_group: f32 = (0..cols)
            .map(|c| {
                let scale = gw.scale_at(0, c / GROUP_SIZE);
                (gw.get(0, c) as f32 * scale - src[c]).abs()
            })
            .sum();
        let err_row: f32 = (0..cols)
            .map(|c| (rw.get(0, c) as f32 * rw.row_scale[0] - src[c]).abs())
            .sum();

        assert!(
            err_group < err_row,
            "group-scale L1 error {err_group} should beat row-scale {err_row}"
        );
    }

    #[test]
    fn batch_matches_per_row_calls() {
        let (rows, cols, batch) = (4, 256, 6);
        let w = filled(rows, cols, 0x77);
        let mut s = 0x99_u64;
        let x: Vec<f32> = (0..cols * batch).map(|_| pseudo(&mut s)).collect();

        let mut y_batch = vec![0.0f32; rows * batch];
        simd_ternary_group_matmul_batch(&w, &x, batch, &mut y_batch);

        for b in 0..batch {
            let mut y_one = vec![0.0f32; rows];
            simd_ternary_group_matvec(&w, &x[b * cols..(b + 1) * cols], &mut y_one);
            for r in 0..rows {
                assert_eq!(y_batch[b * rows + r], y_one[r], "batch {b} row {r}");
            }
        }
    }

    #[cfg(feature = "plasma_path")]
    #[test]
    fn widening_from_row_scale_ternary_preserves_result() {
        let (rows, cols) = (3, 256);
        let mut s = 0xFEED_u64;
        let src: Vec<f32> = (0..rows * cols).map(|_| pseudo(&mut s)).collect();
        let tw = crate::TernaryWeights::quantize_from_f32(&src, rows, cols);
        let gw = TernaryGroupWeights::from_ternary(&tw);

        // Bit-planes copied verbatim.
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(tw.get(r, c), gw.get(r, c), "r={r} c={c}");
            }
        }
        assert!(gw.invariant_holds());

        // Same matvec up to the f32→f16 scale rounding.
        let x: Vec<f32> = (0..cols).map(|_| pseudo(&mut s)).collect();
        let mut y_row = vec![0.0f32; rows];
        let mut y_grp = vec![0.0f32; rows];
        super::super::simd_ternary_matvec(&tw, &x, &mut y_row);
        simd_ternary_group_matvec(&gw, &x, &mut y_grp);
        for r in 0..rows {
            let denom = y_row[r].abs().max(1.0);
            assert!(
                (y_row[r] - y_grp[r]).abs() / denom < 1e-3,
                "r={r}: row={} group={}",
                y_row[r],
                y_grp[r]
            );
        }
    }
}

#[cfg(all(test, feature = "ternary_group_scale"))]
mod parallel_tests {
    use super::*;
    use crate::TernaryGroupWeights;
    use half::f16;

    /// Build weights with all three ternary states and varied group scales.
    fn fixture(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(rows, cols);
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        };
        for r in 0..rows {
            for c in 0..cols {
                let v = match next() % 3 {
                    0 => -1i8,
                    1 => 0,
                    _ => 1,
                };
                if v != 0 {
                    w.set(r, c, v);
                }
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 0.01 + (r % 7) as f32 * 0.03);
            }
        }
        w
    }

    /// The correctness claim `simd_ternary_group_matvec_parallel` rests on:
    /// partitioning by row changes nothing, so the result must be **bit**-equal
    /// to the serial kernel — not approximately equal.
    ///
    /// Each row's accumulation order is untouched by the split; only which
    /// thread runs it changes. Any `!=` here means the partition is unsound
    /// (wrong `row_offset`, overlapping chunks, or a kernel that reads outside
    /// its own row).
    #[test]
    fn parallel_matches_serial_bit_identical() {
        // Shapes chosen to straddle PARALLEL_ROW_MIN (256) and to include a
        // row count that does NOT divide evenly by the worker count, which is
        // where an off-by-one in `row_offset` would surface.
        for &(rows, cols) in &[
            (256, 128),
            (257, 128), // ragged: rows % threads != 0
            (1024, 256),
            (1023, 512), // ragged again, larger
            (2048, 128),
        ] {
            let w = fixture(rows, cols, 0x5940 + rows as u64);
            let x: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.07 - 0.4).collect();

            let mut y_serial = vec![0.0f32; rows];
            let mut y_par = vec![0.0f32; rows];
            simd_ternary_group_matvec(&w, &x, &mut y_serial);
            simd_ternary_group_matvec_parallel(&w, &x, &mut y_par);

            for r in 0..rows {
                assert_eq!(
                    y_serial[r].to_bits(),
                    y_par[r].to_bits(),
                    "{rows}x{cols} row {r}: serial {} vs parallel {} — not bit-identical",
                    y_serial[r],
                    y_par[r]
                );
            }
        }
    }

    /// Issue 935: every rows-per-task split is a pure partition — bit-equal to
    /// the serial kernel for chunks below, at and above the row count.
    #[test]
    fn chunked_matches_serial_bit_identical_for_any_chunk() {
        let (rows, cols) = (1023, 384);
        let w = fixture(rows, cols, 0x935);
        let x: Vec<f32> = (0..cols).map(|i| (i % 11) as f32 * 0.09 - 0.45).collect();
        let mut y_serial = vec![0.0f32; rows];
        simd_ternary_group_matvec(&w, &x, &mut y_serial);
        for chunk in [0, 1, 7, 32, 64, 1000, rows, rows + 5] {
            let mut y = vec![0.0f32; rows];
            simd_ternary_group_matvec_parallel_chunked(&w, &x, &mut y, chunk);
            assert!(
                y.iter().zip(&y_serial).all(|(a, b)| a.to_bits() == b.to_bits()),
                "chunk {chunk}: not bit-identical to serial"
            );
        }
    }

    /// Below `PARALLEL_ROW_MIN` the parallel entry delegates to the serial
    /// kernel. Bonsai's `ssm_alpha`/`ssm_beta` are 48 rows, so this path is
    /// exercised by the real model, not just by tests.
    #[test]
    fn parallel_delegates_below_threshold_and_stays_correct() {
        for &rows in &[1usize, 2, 48, 255] {
            let cols = 256;
            let w = fixture(rows, cols, 0x5941 + rows as u64);
            let x: Vec<f32> = (0..cols).map(|i| (i % 5) as f32 * 0.11).collect();

            let mut y_serial = vec![0.0f32; rows];
            let mut y_par = vec![0.0f32; rows];
            simd_ternary_group_matvec(&w, &x, &mut y_serial);
            simd_ternary_group_matvec_parallel(&w, &x, &mut y_par);

            assert_eq!(y_serial, y_par, "{rows} rows: delegation path diverged");
        }
    }

    /// A zero row must stay exactly zero through the parallel path — catches a
    /// chunk that accumulates into a neighbour's slot.
    #[test]
    fn parallel_preserves_zero_rows() {
        let (rows, cols) = (512, 256);
        let mut w = fixture(rows, cols, 7);
        // Wipe row 300 entirely.
        for i in 0..w.blocks64 {
            w.pos_bits[300 * w.blocks64 + i] = 0;
            w.neg_bits[300 * w.blocks64 + i] = 0;
        }
        for g in 0..w.groups_per_row {
            w.group_scale[300 * w.groups_per_row + g] = f16::from_f32(0.5);
        }

        let x: Vec<f32> = (0..cols).map(|i| 1.0 + i as f32).collect();
        let mut y = vec![f32::NAN; rows];
        simd_ternary_group_matvec_parallel(&w, &x, &mut y);

        assert_eq!(y[300], 0.0, "all-zero row must produce exactly 0.0");
        assert!(
            y.iter().all(|v| v.is_finite()),
            "NaN survived — a slot was never written"
        );
    }
}
