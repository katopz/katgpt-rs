//! Trit-packed ternary matvec — the base-3 footprint kernel
//! (`ternary_trit_pack` feature, Issue 582).
//!
//! Scalar / NEON / AVX2 paths. Same arithmetic as [`super::ternary_group`],
//! different unpack. The weights
//! arrive 5-per-byte in base 3 ([`crate::TernaryTritWeights`]) instead of as two
//! 1-bit planes, which is 1.725 bits/weight instead of 2.125 — **18.8% fewer
//! bytes** to pull through memory.
//!
//! # The trade this kernel makes
//!
//! The bit-plane kernel extracts a weight with a shift and a mask. This one
//! cannot: base-3 digits are not bit-aligned, so extraction would be a divide.
//! Instead it **decodes a whole group into a stack scratch through a 2 KB LUT**
//! ([`crate::TRIT_LUT`], byte → 5 signed values padded to 8 lanes for an
//! aligned 8-byte store), then accumulates over the scratch with the ordinary
//! sign×x inner loop.
//!
//! So: one extra pass over ~26 bytes of L1 per group, in exchange for 18.8%
//! less traffic from RAM. Which side wins depends on whether the weights are
//! resident — for a 27B model streaming 5.82 GB instead of 7.17 GB it should,
//! for a 512×512 matrix that fits in cache it should not. **The benchmark
//! decides, not this comment** (Issue 582 G2b).
//!
//! # Group boundaries split bytes
//!
//! `GROUP_SIZE % TRITS_PER_BYTE == 128 % 5 == 3`, so a group's first weight can
//! sit at any trit position in its first byte, and its last byte is shared with
//! the next group. The decode always unpacks **whole bytes**, so the group's
//! first weight lands `off = w_start % 5` lanes into the scratch and the kernel
//! reads from `scratch[off]` — the leading lanes belong to the previous group.
//! Shared bytes are decoded twice (once per group) rather than carried between
//! iterations: 1 redundant byte per 26, and it keeps each group independent,
//! which is what lets rows and groups be reordered freely.
//!
//! # Scalar vs SIMD agreement
//!
//! Both paths apply the group scale **once per group**, matching
//! [`super::ternary_group::ternary_group_matvec_scalar`]'s association exactly.
//! The scalar path here is therefore bit-identical to the bit-plane scalar
//! reference (same operations in the same order — asserted in tests), which
//! makes cross-tier equality a real gate rather than a tolerance check. The
//! NEON path differs only in summation order (4 lanes + horizontal add), so it
//! agrees to ~1e-6 relative.

#![allow(clippy::needless_range_loop)]

#[cfg(all(feature = "ternary_trit_pack", target_arch = "aarch64"))]
use super::SimdLevel;
#[cfg(feature = "ternary_trit_pack")]
use super::simd_level;
#[cfg(feature = "ternary_trit_pack")]
use crate::{GROUP_SIZE, TRIT_LUT, TRITS_PER_BYTE, TernaryTritWeights};

/// Decode scratch length.
///
/// A group spans at most `ceil((GROUP_SIZE + TRITS_PER_BYTE - 1) / TRITS_PER_BYTE)`
/// = 27 bytes, each written as an 8-byte store at stride 5, so the last store
/// starts at `5*26 = 130` and ends at 138. The deepest *read* is
/// `off + live - 1 <= 4 + 127 = 131`. Rounded up to 160.
#[cfg(feature = "ternary_trit_pack")]
const SCRATCH_LEN: usize = 160;

/// Decode group `g` of row `r` into `scratch`, returning `(base, live)`.
///
/// On return, `scratch[base + j]` is weight `w_start + j` for `j < live`, where
/// `base` is the group's starting trit offset within its first byte.
///
/// **The offset is forward, not backward.** The group's first byte begins at
/// weight `5 * b_start <= w_start`, so decoding that whole byte places weight
/// `w_start` at `off = w_start % 5` lanes *into* the scratch — the first `off`
/// lanes hold the tail of the *previous* group and must be skipped. Getting this
/// sign wrong is silent: single-group shapes still pass, and only shapes with a
/// second group diverge (which is what
/// `scalar_is_bit_identical_to_the_bit_plane_scalar_reference` at 3x256 caught).
///
/// Zero allocations — `scratch` is a caller-owned stack array.
#[cfg(feature = "ternary_trit_pack")]
#[inline]
fn decode_group(
    w: &TernaryTritWeights,
    row: usize,
    group: usize,
    scratch: &mut [i8; SCRATCH_LEN],
) -> (usize, usize) {
    let (b_start, b_end, off) = w.group_byte_span(group);
    let row_base = row * w.bytes_per_row;
    let w_start = group * GROUP_SIZE;
    let live = (w_start + GROUP_SIZE).min(w.cols) - w_start;

    for b in b_start..b_end {
        let entry = &TRIT_LUT[w.trits[row_base + b] as usize];
        let dst = (b - b_start) * TRITS_PER_BYTE;
        // 8-byte store; the 3 pad lanes are overwritten by the next byte's
        // entry (stride 5), and the final overhang lands inside the scratch.
        scratch[dst..dst + 8].copy_from_slice(entry);
    }

    (off, live)
}

/// Scalar reference: `y[r] = Σ_g group_scale[r,g] · Σ_{col∈g} weight(col) · x[col]`
///
/// Bit-identical to [`super::ternary_group::ternary_group_matvec_scalar`] on the
/// same logical weights — same summation order, same single scale multiply per
/// group. That equality is asserted in the tests and is the tier's G1.
#[cfg(feature = "ternary_trit_pack")]
pub fn ternary_trit_matvec_scalar(w: &TernaryTritWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    scalar_row_range(w, x, y, 0);
}

/// Scalar kernel over the row range `[row_offset, row_offset + y.len())`.
///
/// Rows are independent — row `r` reads only `trits`/`group_scale` at its own
/// offsets and writes only `y[r]` — so a caller may partition `y` across
/// threads and get a bit-identical result.
#[cfg(feature = "ternary_trit_pack")]
fn scalar_row_range(w: &TernaryTritWeights, x: &[f32], y: &mut [f32], row_offset: usize) {
    let mut scratch = [0i8; SCRATCH_LEN];
    for (i, y_slot) in y.iter_mut().enumerate() {
        let r = row_offset + i;
        let group_base = r * w.groups_per_row;
        let mut row_sum = 0.0f32;
        for g in 0..w.groups_per_row {
            let (base, live) = decode_group(w, r, g, &mut scratch);
            let w_start = g * GROUP_SIZE;
            let mut group_acc = 0.0f32;
            for j in 0..live {
                let sign = scratch[base + j] as f32;
                group_acc += sign * unsafe { *x.get_unchecked(w_start + j) };
            }
            row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
        }
        *y_slot = row_sum;
    }
}

/// NEON kernel over the row range `[row_offset, row_offset + y.len())`.
///
/// 16 decoded trits per iteration: one `vld1q_s8`, widened i8→i16→i32→f32 into
/// four `float32x4_t`, each multiplied into an accumulator against the matching
/// `x` lanes. The group scale is applied once, after the horizontal sum — one
/// hsum per 128 weights, which is why this kernel does not need the bit-plane
/// kernel's fold-scale-into-the-sign trick.
///
/// # Safety
/// Caller guarantees `x.len() == w.cols` and `row_offset + y.len() <= w.rows`.
#[cfg(all(feature = "ternary_trit_pack", target_arch = "aarch64"))]
unsafe fn neon_row_range(w: &TernaryTritWeights, x: &[f32], y: &mut [f32], row_offset: usize) {
    unsafe {
        use core::arch::aarch64::*;
        debug_assert!(row_offset + y.len() <= w.rows);

        let mut scratch = [0i8; SCRATCH_LEN];
        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let group_base = r * w.groups_per_row;
            let mut row_sum = 0.0f32;

            for g in 0..w.groups_per_row {
                let (base, live) = decode_group(w, r, g, &mut scratch);
                let w_start = g * GROUP_SIZE;

                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                let chunks = live / 16;
                for c in 0..chunks {
                    let s = vld1q_s8(scratch.as_ptr().add(base + c * 16));
                    // i8 -> i16 -> i32 -> f32, low and high halves.
                    let lo16 = vmovl_s8(vget_low_s8(s));
                    let hi16 = vmovl_s8(vget_high_s8(s));
                    let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                    let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                    let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                    let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));

                    let xp = x.as_ptr().add(w_start + c * 16);
                    acc0 = vfmaq_f32(acc0, f0, vld1q_f32(xp));
                    acc1 = vfmaq_f32(acc1, f1, vld1q_f32(xp.add(4)));
                    acc2 = vfmaq_f32(acc2, f2, vld1q_f32(xp.add(8)));
                    acc3 = vfmaq_f32(acc3, f3, vld1q_f32(xp.add(12)));
                }

                let mut group_acc =
                    vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
                // Tail: fewer than 16 weights left in a ragged final group.
                for j in chunks * 16..live {
                    group_acc +=
                        *scratch.get_unchecked(base + j) as f32 * *x.get_unchecked(w_start + j);
                }

                row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
            }
            *y_slot = row_sum;
        }
    }
}

/// AVX2 kernel over the row range `[row_offset, row_offset + y.len())`
/// (Issue 582 follow-up, 2026-08-12).
///
/// The base-3 decode itself is architecture-independent — [`decode_group`] is
/// plain Rust byte moves through [`crate::TRIT_LUT`] — so only the accumulate
/// loop differs from NEON. AVX2 unpacks the decoded `i8` lanes with a **single**
/// `_mm256_cvtepi8_epi32` (VPMOVSXBD: 8 signed bytes → 8 `i32`), which is
/// cheaper than the bit-plane kernel's SWAR path (splat a byte, AND against a
/// bit-position mask, compare) — the same reason the NEON version wins.
///
/// 32 decoded trits per outer unroll across 4 `__m256` accumulators, group scale
/// applied once after the horizontal sum.
///
/// # Safety
/// Caller guarantees `x.len() == w.cols` and `row_offset + y.len() <= w.rows`.
#[cfg(all(feature = "ternary_trit_pack", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_trit_row_range(w: &TernaryTritWeights, x: &[f32], y: &mut [f32], row_offset: usize) {
    use super::horizontal::horizontal_sum_256;
    use core::arch::x86_64::*;
    unsafe {
        debug_assert_eq!(x.len(), w.cols);
        debug_assert!(row_offset + y.len() <= w.rows);

        let mut scratch = [0i8; SCRATCH_LEN];
        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let group_base = r * w.groups_per_row;
            let mut row_sum = 0.0f32;

            for g in 0..w.groups_per_row {
                let (base, live) = decode_group(w, r, g, &mut scratch);
                let w_start = g * GROUP_SIZE;

                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

                // 8 trits per accumulator, 4 accumulators = 32 per iteration.
                let chunks = live / 32;
                for c in 0..chunks {
                    let sp = scratch.as_ptr().add(base + c * 32);
                    let xp = x.as_ptr().add(w_start + c * 32);
                    // _mm_loadl_epi64 reads 8 bytes; cvtepi8_epi32 sign-extends
                    // them into 8 i32 lanes in one instruction.
                    let s0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(sp as *const __m128i));
                    let s1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(sp.add(8) as *const __m128i));
                    let s2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(sp.add(16) as *const __m128i));
                    let s3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(sp.add(24) as *const __m128i));
                    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s0), _mm256_loadu_ps(xp), acc0);
                    acc1 =
                        _mm256_fmadd_ps(_mm256_cvtepi32_ps(s1), _mm256_loadu_ps(xp.add(8)), acc1);
                    acc2 =
                        _mm256_fmadd_ps(_mm256_cvtepi32_ps(s2), _mm256_loadu_ps(xp.add(16)), acc2);
                    acc3 =
                        _mm256_fmadd_ps(_mm256_cvtepi32_ps(s3), _mm256_loadu_ps(xp.add(24)), acc3);
                }

                // Remaining 8-element chunks.
                let mut j = chunks * 32;
                while j + 8 <= live {
                    let s = _mm256_cvtepi8_epi32(_mm_loadl_epi64(
                        scratch.as_ptr().add(base + j) as *const __m128i
                    ));
                    acc0 = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(s),
                        _mm256_loadu_ps(x.as_ptr().add(w_start + j)),
                        acc0,
                    );
                    j += 8;
                }

                acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let mut group_acc = horizontal_sum_256(acc0);
                // Scalar tail (0-7 trits in a ragged final group).
                while j < live {
                    group_acc +=
                        *scratch.get_unchecked(base + j) as f32 * *x.get_unchecked(w_start + j);
                    j += 1;
                }

                row_sum += w.group_scale[group_base + g].to_f32() * group_acc;
            }
            *y_slot = row_sum;
        }
    }
}

/// `y = w @ x` — dispatches to NEON / AVX2 where available, else the scalar
/// reference.
///
/// Writes into a caller-owned `y`, allocating nothing (Issue 582 G4).
#[cfg(feature = "ternary_trit_pack")]
pub fn simd_ternary_trit_matvec(w: &TernaryTritWeights, x: &[f32], y: &mut [f32]) {
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
            unsafe { avx2_trit_row_range(w, x, y, 0) };
            return;
        }
    }
    let _ = simd_level();
    scalar_row_range(w, x, y, 0);
}

/// Row-parallel `y = w @ x` — the trit tier's decode-step kernel.
///
/// Mirrors [`super::ternary_group::simd_ternary_group_matvec_parallel`], which
/// exists because the only rayon entry point in that tier parallelizes over
/// *batch* and therefore did nothing at `batch = 1` — i.e. at every
/// autoregressive decode step. Without this, a consumer switching from the
/// bit-plane tier to this one to save 18.8% of its footprint would silently give
/// up that tier's measured **7.21×** row-parallel gain on real Bonsai geometry,
/// which is a far bigger loss than the footprint win.
///
/// ## Correctness
///
/// Rows are fully independent: row `r` reads only its own span of `trits` and
/// `group_scale`, and writes only `y[r]`. Every group re-decodes its own bytes
/// (shared boundary bytes included), so no state crosses a row or group
/// boundary and `par_chunks_mut` is a pure partition — **bit-identical** to
/// serial, not merely close. Asserted by `parallel_matches_serial_bit_identical`.
///
/// Below `PARALLEL_ROW_MIN` rows it delegates to serial, matching the bit-plane
/// tier's threshold so the two tiers switch over at the same shape (Bonsai's
/// 48-row `ssm_alpha`/`ssm_beta` stay serial in both).
#[cfg(feature = "ternary_trit_pack")]
pub fn simd_ternary_trit_matvec_parallel(w: &TernaryTritWeights, x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;

    /// Same value as the bit-plane tier's threshold — see the doc comment.
    const PARALLEL_ROW_MIN: usize = 256;

    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");

    if w.rows < PARALLEL_ROW_MIN {
        simd_ternary_trit_matvec(w, x, y);
        return;
    }

    // Capped rows per task so work-stealing balances P- and E-cores (Issue 935).
    let chunk = super::par_matvec_chunk(w.rows);
    y.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(ci, y_chunk)| {
            let row_offset = ci * chunk;
            #[cfg(target_arch = "aarch64")]
            {
                if matches!(simd_level(), SimdLevel::Neon) {
                    unsafe { neon_row_range(w, x, y_chunk, row_offset) };
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if super::is_avx2_fma_available() {
                    unsafe { avx2_trit_row_range(w, x, y_chunk, row_offset) };
                    return;
                }
            }
            scalar_row_range(w, x, y_chunk, row_offset);
        });
}

#[cfg(all(test, feature = "ternary_trit_pack"))]
mod tests {
    use super::*;

    /// Deterministic ternary pattern with all three states and an uneven mix.
    fn fill_pattern(rows: usize, cols: usize) -> Vec<i8> {
        (0..rows * cols)
            .map(|i| match (i * 7 + i / 13) % 5 {
                0 | 1 => 1i8,
                2 => -1,
                3 => 0,
                _ => -1,
            })
            .collect()
    }

    fn make(rows: usize, cols: usize) -> TernaryTritWeights {
        let pat = fill_pattern(rows, cols);
        let mut w = TernaryTritWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                w.set(r, c, pat[r * cols + c]);
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 0.25 + 0.5 * ((r + g) % 4) as f32);
            }
        }
        w
    }

    fn xs(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect()
    }

    #[test]
    fn set_get_roundtrips_all_three_states_across_byte_boundaries() {
        // 13 cols = 2 full bytes + 3 trits, so the last byte is ragged.
        let mut w = TernaryTritWeights::new(3, 13);
        for r in 0..3 {
            for c in 0..13 {
                let v = match (r + c) % 3 {
                    0 => -1i8,
                    1 => 0,
                    _ => 1,
                };
                w.set(r, c, v);
            }
        }
        for r in 0..3 {
            for c in 0..13 {
                let want = match (r + c) % 3 {
                    0 => -1i8,
                    1 => 0,
                    _ => 1,
                };
                assert_eq!(w.get(r, c), want, "({r},{c})");
            }
        }
        // Overwriting one trit must not disturb its four byte-mates.
        w.set(0, 1, 1);
        assert_eq!(w.get(0, 0), -1);
        assert_eq!(w.get(0, 1), 1);
        assert_eq!(w.get(0, 2), 1);
        assert!(w.is_canonical(), "every byte must stay < 243");
    }

    #[test]
    fn footprint_is_1_725_bits_per_weight() {
        // 1280 cols = 10 whole groups, no ragged tail — the clean case.
        let w = TernaryTritWeights::new(4, 1280);
        // 1280/5 = 256 B trits + 10 * 2 B scale = 276 B per row.
        assert_eq!(w.encoded_bytes(), 4 * (256 + 20));
        // 8/5 bits per weight + 16 bits per 128-weight group = 1.6 + 0.125.
        let bits_per_weight = (w.encoded_bytes() * 8) as f64 / (4 * 1280) as f64;
        assert!(
            (bits_per_weight - 1.725).abs() < 1e-9,
            "expected 1.725 bits/weight, got {bits_per_weight}"
        );
    }

    #[test]
    fn footprint_beats_the_bit_plane_tier_by_18_percent() {
        // The G2 gate: the -18.8% must be realized, not eaten by padding.
        let trit = TernaryTritWeights::new(64, 1280);
        let plane = crate::TernaryGroupWeights::new(64, 1280);
        let ratio = trit.encoded_bytes() as f64 / plane.encoded_bytes() as f64;
        assert!(
            ratio < 0.83,
            "trit tier must be under 83% of the bit-plane tier, got {ratio}"
        );
        assert!((ratio - 1.725 / 2.125).abs() < 1e-3, "ratio {ratio}");
    }

    #[test]
    fn group_byte_span_reports_the_shared_boundary_byte() {
        let w = TernaryTritWeights::new(1, 512);
        // Group 0: weights 0..128 -> bytes 0..26 (byte 25 holds weights 125-129,
        // so it is shared with group 1), starting trit offset 0.
        assert_eq!(w.group_byte_span(0), (0, 26, 0));
        // Group 1: weights 128..256 -> starts 3 trits into byte 25.
        let (b_start, b_end, off) = w.group_byte_span(1);
        assert_eq!((b_start, off), (25, 3));
        assert_eq!(b_end, 52); // 256/5 = 51.2 -> 52
        // The shared byte proves the hazard is real, not hypothetical.
        assert_eq!(w.group_byte_span(0).1 - 1, b_start);
    }

    #[test]
    fn neon_matches_scalar_reference() {
        // Shapes chosen to hit: clean groups, ragged group, cols not divisible
        // by 5 (ragged byte), cols < one group, cols < 16 (pure NEON tail).
        for &(rows, cols) in &[
            (4usize, 128usize),
            (3, 256),
            (5, 300),
            (2, 133),
            (7, 64),
            (3, 13),
            (2, 1280),
        ] {
            let w = make(rows, cols);
            let x = xs(cols);
            let mut got = vec![0.0f32; rows];
            let mut want = vec![0.0f32; rows];
            simd_ternary_trit_matvec(&w, &x, &mut got);
            ternary_trit_matvec_scalar(&w, &x, &mut want);
            for r in 0..rows {
                let denom = want[r].abs().max(1.0);
                assert!(
                    (got[r] - want[r]).abs() / denom < 1e-6,
                    "{rows}x{cols} row {r}: neon {} vs scalar {}",
                    got[r],
                    want[r]
                );
            }
        }
    }

    #[test]
    fn zero_weights_yield_zero_at_nonunit_scale() {
        let mut w = TernaryTritWeights::new(3, 300);
        for r in 0..3 {
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 3.0);
            }
        }
        // All-zero weights encode as byte 121, not byte 0 — the test would
        // pass vacuously if `new` zeroed the buffer, so assert the encoding.
        assert!(w.trits.iter().all(|&b| b == 121), "zero weight => trit 1");
        let x = xs(300);
        let mut y = vec![9.0f32; 3];
        simd_ternary_trit_matvec(&w, &x, &mut y);
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn non_canonical_bytes_are_detected() {
        let mut w = TernaryTritWeights::new(2, 20);
        assert!(w.is_canonical());
        w.trits[3] = 250; // corruption: >= 3^5
        assert!(!w.is_canonical(), "byte 250 must be rejected");
        // And it decodes to zeros rather than garbage.
        assert_eq!(TRIT_LUT[250], [0i8; 8]);
    }

    #[test]
    fn parallel_matches_serial_bit_identical() {
        // Straddle PARALLEL_ROW_MIN (256) and include a ragged row count, a
        // ragged byte count (cols % 5 != 0) and a zeroed row.
        for &(rows, cols) in &[(255usize, 300usize), (256, 300), (513, 133), (1024, 512)] {
            let mut w = make(rows, cols);
            // Zero one row entirely: all-zero weights encode as byte 121, so
            // this also checks the parallel path does not assume a zeroed buffer.
            for c in 0..cols {
                w.set(rows / 2, c, 0);
            }
            let x = xs(cols);
            let mut y_par = vec![0.0f32; rows];
            let mut y_ser = vec![0.0f32; rows];
            simd_ternary_trit_matvec_parallel(&w, &x, &mut y_par);
            simd_ternary_trit_matvec(&w, &x, &mut y_ser);
            assert_eq!(y_par, y_ser, "{rows}x{cols} must be bit-identical");
            assert_eq!(y_par[rows / 2], 0.0, "{rows}x{cols} zeroed row");
        }
    }

    #[test]
    fn parallel_delegates_below_threshold() {
        // 48 rows is Bonsai's ssm_alpha/ssm_beta shape — under the threshold,
        // so this exercises the serial delegation with a real geometry.
        let w = make(48, 5120);
        let x = xs(5120);
        let mut y_par = vec![0.0f32; 48];
        let mut y_ser = vec![0.0f32; 48];
        simd_ternary_trit_matvec_parallel(&w, &x, &mut y_par);
        simd_ternary_trit_matvec(&w, &x, &mut y_ser);
        assert_eq!(y_par, y_ser);
    }

    #[test]
    fn quantize_is_bit_identical_to_the_bit_plane_tier() {
        // The two tiers run the same mean-abs scale, same 0.5*scale threshold,
        // same error carry — so they must agree on every weight AND every
        // scale. This is a stronger claim than "signs look right": it pins the
        // trit tier to the shipped reference quantizer rather than to a
        // hand-written expectation. (Error-feedback quantization legitimately
        // flips an individual large weight when the carry runs against it, so
        // per-weight sign assertions are the wrong gate here.)
        let vals: Vec<f32> = (0..3 * 300)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.3)
            .collect();
        let trit = TernaryTritWeights::quantize_from_f32(&vals, 3, 300);
        let plane = crate::TernaryGroupWeights::quantize_from_f32(&vals, 3, 300);
        assert!(trit.is_canonical());
        for r in 0..3 {
            for c in 0..300 {
                assert_eq!(trit.get(r, c), plane.get(r, c), "weight ({r},{c})");
            }
            for g in 0..trit.groups_per_row {
                assert_eq!(trit.scale_at(r, g), plane.scale_at(r, g), "scale ({r},{g})");
            }
        }
        // Sanity: the fixture must actually exercise all three states.
        let mut counts = [0usize; 3];
        for r in 0..3 {
            for c in 0..300 {
                counts[(trit.get(r, c) + 1) as usize] += 1;
            }
        }
        assert!(counts.iter().all(|&n| n > 50), "states: {counts:?}");
    }

    #[test]
    fn repack_round_trips_through_the_bit_plane_tier() {
        // G1: from_group -> to_group is lossless in weights and scales, over
        // shapes with ragged bytes (cols % 5 != 0) and ragged groups.
        for &(rows, cols) in &[(4usize, 128usize), (3, 300), (2, 133), (5, 13)] {
            let plane = {
                let pat = fill_pattern(rows, cols);
                let mut p = crate::TernaryGroupWeights::new(rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        p.set(r, c, pat[r * cols + c]);
                    }
                    for g in 0..p.groups_per_row {
                        p.set_scale(r, g, 0.25 + 0.5 * ((r + g) % 4) as f32);
                    }
                }
                p
            };
            let trit = TernaryTritWeights::from_group(&plane);
            assert!(trit.is_canonical(), "{rows}x{cols}");
            let back = trit.to_group();
            assert_eq!(back.pos_bits, plane.pos_bits, "{rows}x{cols} pos");
            assert_eq!(back.neg_bits, plane.neg_bits, "{rows}x{cols} neg");
            assert_eq!(back.group_scale, plane.group_scale, "{rows}x{cols} scale");
            assert_eq!(trit.checksum(), plane.checksum(), "{rows}x{cols} checksum");
        }
    }

    #[test]
    fn scalar_is_bit_identical_to_the_bit_plane_scalar_reference() {
        // Both apply one scale per group over an in-order sum, so this is
        // exact equality, not a tolerance — the tier's headline G1.
        for &(rows, cols) in &[(4usize, 128usize), (3, 256), (5, 300), (2, 133), (3, 13)] {
            let trit = make(rows, cols);
            let plane = trit.to_group();
            let x = xs(cols);
            let mut got = vec![0.0f32; rows];
            let mut want = vec![0.0f32; rows];
            ternary_trit_matvec_scalar(&trit, &x, &mut got);
            super::super::ternary_group::ternary_group_matvec_scalar(&plane, &x, &mut want);
            assert_eq!(got, want, "{rows}x{cols}");
        }
    }
}
