//! Ternary group-scale matvec with **int8 activations** — the SDOT kernel
//! (`ternary_group_q8_act`, Issue 935).
//!
//! The f32 kernel ([`super::ternary_group`]) expands every weight bit into an
//! f32 sign and FMAs it: ~18 NEON ops per 8 weights, compute-bound at ~25× the
//! 2-bit memory-traffic floor on Bonsai-2-27B decode. This kernel quantizes the
//! activation vector ONCE per call to int8 per 128-wide group (the weight group,
//! so one f32 rescale per group covers both scales), then per 16 weights:
//! `tbl` + `tst` per bit-plane, one `sub` for the {-1,0,+1} i8 sign, one
//! `sdot` — ~7 ops per 16 weights.
//!
//! # Semantics (lossy — the activation rounding)
//!
//! `y[r] = Σ_g (wscale[r,g] · xscale[g]) · Σ_{c∈g} sign(r,c) · xq[c]`, with
//! `xscale[g] = max|x_g| / 127` and `xq = round(x / xscale)`. The inner sum is
//! an exact integer, so the NEON path and [`ternary_group_q8_matvec_scalar`]
//! are **bit-identical**; the loss vs the f32 kernel is the activation rounding
//! only (the regime llama.cpp's CPU `Q2_0` path runs: q8 activations).
//!
//! Fast path requires `cols % 128 == 0` (every Bonsai shape); anything else
//! falls through to the f32 kernel. x86_64 runs the AVX2 port
//! (`psignb` + `pmaddubsw`, Issue 935 T4), bit-identical to the same scalar
//! reference.

#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;

use super::ternary_group::simd_ternary_group_matvec_parallel_chunked;
use crate::{GROUP_SIZE, TernaryGroupWeights};

/// Quantize `x` to int8 per [`GROUP_SIZE`] group: `xq[c] = round(x[c] / s_g)`,
/// `s_g = max|x_g| / 127` (an all-zero group gets `s_g = 0`, `xq = 0`).
///
/// Caller guarantees `x.len() % GROUP_SIZE == 0`, `xq.len() == x.len()`,
/// `xs.len() == x.len() / GROUP_SIZE`.
pub fn quantize_activations_q8(x: &[f32], xq: &mut [i8], xs: &mut [f32]) {
    debug_assert_eq!(x.len() % GROUP_SIZE, 0);
    debug_assert_eq!(xq.len(), x.len());
    debug_assert_eq!(xs.len(), x.len() / GROUP_SIZE);
    for ((xg, qg), s) in x
        .as_chunks::<GROUP_SIZE>()
        .0
        .iter()
        .zip(xq.as_chunks_mut::<GROUP_SIZE>().0.iter_mut())
        .zip(xs.iter_mut())
    {
        let amax = xg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            *s = 0.0;
            qg.fill(0);
            continue;
        }
        *s = amax / 127.0;
        let inv = 127.0 / amax;
        for (q, &v) in qg.iter_mut().zip(xg) {
            *q = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

/// Scalar reference over pre-quantized activations — rows `row_offset ..
/// row_offset + y.len()`. The NEON kernel is asserted bit-identical to it.
pub fn ternary_group_q8_rows_scalar(
    w: &TernaryGroupWeights,
    xq: &[i8],
    xs: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    for (i, y_slot) in y.iter_mut().enumerate() {
        let r = row_offset + i;
        let mut row_sum = 0.0f32;
        for g in 0..w.groups_per_row {
            let mut isum = 0i32;
            for c in g * GROUP_SIZE..(g + 1) * GROUP_SIZE {
                let idx = r * w.blocks64 + (c >> 6);
                let bit = 1u64 << (c & 63);
                let s = ((w.pos_bits[idx] & bit) != 0) as i32 - ((w.neg_bits[idx] & bit) != 0) as i32;
                isum += s * xq[c] as i32;
            }
            let scale = w.group_scale[r * w.groups_per_row + g].to_f32() * xs[g];
            row_sum += scale * isum as f32;
        }
        *y_slot = row_sum;
    }
}

/// NEON + SDOT row range. Same contract as [`ternary_group_q8_rows_scalar`].
///
/// # Safety
/// CPU supports `dotprod`; `w.cols % GROUP_SIZE == 0`; `xq.len() == w.cols`;
/// `row_offset + y.len() <= w.rows`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn neon_q8_row_range(
    w: &TernaryGroupWeights,
    xq: &[i8],
    xs: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    use core::arch::aarch64::*;
    const BIT_MASK: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    // Chunk k of a 64-bit block covers cols 16k..16k+16: lanes 0-7 read byte
    // 2k, lanes 8-15 byte 2k+1 (little-endian), then `tst` picks the bit.
    const TBL_IDX: [[u8; 16]; 4] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
        [6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ];
    debug_assert_eq!(w.cols % GROUP_SIZE, 0);
    debug_assert_eq!(xq.len(), w.cols);
    debug_assert!(row_offset + y.len() <= w.rows);
    unsafe {
        let bit_mask = vld1q_u8(BIT_MASK.as_ptr());
        let idx = [
            vld1q_u8(TBL_IDX[0].as_ptr()),
            vld1q_u8(TBL_IDX[1].as_ptr()),
            vld1q_u8(TBL_IDX[2].as_ptr()),
            vld1q_u8(TBL_IDX[3].as_ptr()),
        ];
        let xq_ptr = xq.as_ptr();
        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let row_base = r * w.blocks64;
            let pos = w.pos_bits.as_ptr().add(row_base);
            let neg = w.neg_bits.as_ptr().add(row_base);
            let scales = w.group_scale.as_ptr().add(r * w.groups_per_row);
            let mut row_sum = 0.0f32;
            for g in 0..w.groups_per_row {
                let mut acc = [vdupq_n_s32(0); 4];
                for bb in 0..GROUP_SIZE / 64 {
                    let b = g * (GROUP_SIZE / 64) + bb;
                    let pw = vreinterpretq_u8_u64(vdupq_n_u64(*pos.add(b)));
                    let nw = vreinterpretq_u8_u64(vdupq_n_u64(*neg.add(b)));
                    let xb = xq_ptr.add(b * 64);
                    for k in 0..4 {
                        let pm = vtstq_u8(vqtbl1q_u8(pw, idx[k]), bit_mask);
                        let nm = vtstq_u8(vqtbl1q_u8(nw, idx[k]), bit_mask);
                        // mask lanes are -1 where set: neg − pos = +1 pos, −1 neg.
                        let sign = vsubq_s8(vreinterpretq_s8_u8(nm), vreinterpretq_s8_u8(pm));
                        acc[k] = vdotq_s32(acc[k], sign, vld1q_s8(xb.add(16 * k)));
                    }
                }
                let isum = vaddvq_s32(vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3])));
                let scale = (*scales.add(g)).to_f32() * xs[g];
                row_sum += scale * isum as f32;
            }
            *y_slot = row_sum;
        }
    }
}

/// AVX2 row range (Issue 935 T4). Same contract as
/// [`ternary_group_q8_rows_scalar`] and bit-identical to it.
///
/// Per 32 weights: broadcast one u32 of each bit-plane, `pshufb` byte k to
/// lanes 8k..8k+8, `and`+`cmpeq` against the per-lane bit → 0xFF masks,
/// `sub` → {-1,0,+1}, `psignb` applies it to `xq`, `pmaddubsw` against ones
/// pair-sums to i16. The four i16 partials of a 128-group (each ≤ 254 in
/// magnitude, sum ≤ 1016) stay in i16; one `pmaddwd` widens per group.
///
/// # Safety
/// CPU supports `avx2`; `w.cols % GROUP_SIZE == 0`; `xq.len() == w.cols`;
/// `row_offset + y.len() <= w.rows`; `|xq| <= 127` (no `psignb` overflow).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_q8_row_range(
    w: &TernaryGroupWeights,
    xq: &[i8],
    xs: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(w.cols % GROUP_SIZE, 0);
    debug_assert_eq!(xq.len(), w.cols);
    debug_assert!(row_offset + y.len() <= w.rows);
    unsafe {
        // `pshufb` is in-lane: the low 128-bit lane takes bytes 0,1 of the
        // broadcast u32, the high lane bytes 2,3 (indices are lane-relative,
        // and every lane holds the full broadcast word).
        let shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, //
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        );
        let bit = _mm256_set1_epi64x(0x8040_2010_0804_0201u64 as i64);
        let ones_u8 = _mm256_set1_epi8(1);
        let ones_i16 = _mm256_set1_epi16(1);
        let xq_ptr = xq.as_ptr();
        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let row_base = r * w.blocks64;
            let pos = w.pos_bits.as_ptr().add(row_base) as *const u32;
            let neg = w.neg_bits.as_ptr().add(row_base) as *const u32;
            let scales = w.group_scale.as_ptr().add(r * w.groups_per_row);
            let mut row_sum = 0.0f32;
            for g in 0..w.groups_per_row {
                let mut acc16 = _mm256_setzero_si256();
                for k in 0..GROUP_SIZE / 32 {
                    let wi = g * (GROUP_SIZE / 32) + k;
                    // Little-endian: u32 word wi of the plane = cols 32wi..32wi+32.
                    let pw = _mm256_shuffle_epi8(_mm256_set1_epi32(pos.add(wi).read_unaligned() as i32), shuf);
                    let nw = _mm256_shuffle_epi8(_mm256_set1_epi32(neg.add(wi).read_unaligned() as i32), shuf);
                    let pm = _mm256_cmpeq_epi8(_mm256_and_si256(pw, bit), bit);
                    let nm = _mm256_cmpeq_epi8(_mm256_and_si256(nw, bit), bit);
                    // mask lanes are -1 where set: neg − pos = +1 pos, −1 neg.
                    let sign = _mm256_sub_epi8(nm, pm);
                    let xv = _mm256_loadu_si256(xq_ptr.add(wi * 32) as *const __m256i);
                    let sx = _mm256_sign_epi8(xv, sign);
                    acc16 = _mm256_add_epi16(acc16, _mm256_maddubs_epi16(ones_u8, sx));
                }
                let acc32 = _mm256_madd_epi16(acc16, ones_i16);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(acc32), _mm256_extracti128_si256::<1>(acc32));
                let s64 = _mm_add_epi32(s128, _mm_unpackhi_epi64(s128, s128));
                let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32::<0b01>(s64));
                let isum = _mm_cvtsi128_si32(s32);
                let scale = (*scales.add(g)).to_f32() * xs[g];
                row_sum += scale * isum as f32;
            }
            *y_slot = row_sum;
        }
    }
}

/// Whether the integer fast path runs on this CPU (aarch64 + `dotprod`, or
/// x86_64 + `avx2` — runtime-probed, never a compile-time `target_feature`).
#[inline]
fn q8_fast_path_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(target_arch = "x86_64")]
    {
        super::is_avx2_fma_available()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// One row range over pre-quantized activations — the SDOT / AVX2 kernel.
///
/// Only reached when [`q8_fast_path_available`] is true.
#[inline]
fn q8_row_range(w: &TernaryGroupWeights, xq: &[i8], xs: &[f32], y: &mut [f32], row_offset: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: the caller checked dotprod and the shape contract.
    unsafe {
        neon_q8_row_range(w, xq, xs, y, row_offset)
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the caller checked avx2 and the shape contract; quantization
    // clamps xq to ±127.
    unsafe {
        avx2_q8_row_range(w, xq, xs, y, row_offset)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    ternary_group_q8_rows_scalar(w, xq, xs, y, row_offset);
}

thread_local! {
    /// Per-calling-thread activation scratch, reused across calls (zero
    /// steady-state allocation).
    static Q8_SCRATCH: RefCell<(Vec<i8>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// Row-parallel ternary group matvec with int8 activations: `y ≈ W × x`.
///
/// Quantizes `x` once (per-128-group int8), then runs the SDOT kernel over
/// rayon row chunks (the same capped split as the f32 kernel). Where the fast
/// path cannot run — `cols % 128 != 0`, or neither aarch64 `dotprod` nor x86_64
/// `avx2` (the scalar integer arm would be far slower than the f32 kernel) —
/// this IS the f32 kernel, exactly.
pub fn simd_ternary_group_matvec_q8_parallel(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;

    /// Below this row count, rayon's task overhead outweighs the work.
    const PARALLEL_ROW_MIN: usize = 256;

    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    if !w.cols.is_multiple_of(GROUP_SIZE) || !q8_fast_path_available() {
        let chunk = super::par_matvec_chunk(w.rows);
        simd_ternary_group_matvec_parallel_chunked(w, x, y, chunk);
        return;
    }
    Q8_SCRATCH.with_borrow_mut(|(xq, xs)| {
        xq.resize(w.cols, 0);
        xs.resize(w.cols / GROUP_SIZE, 0.0);
        quantize_activations_q8(x, xq, xs);
        let (xq, xs) = (&xq[..], &xs[..]);
        if w.rows < PARALLEL_ROW_MIN {
            q8_row_range(w, xq, xs, y, 0);
            return;
        }
        #[cfg(feature = "decode_spin_pool")]
        if super::spin_pool::spin_row_chunks(y, super::PAR_MATVEC_CHUNK_ROWS, &|c, row0| q8_row_range(w, xq, xs, c, row0)) {
            return;
        }
        let chunk = super::par_matvec_chunk(w.rows);
        y.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, y_chunk)| q8_row_range(w, xq, xs, y_chunk, ci * chunk));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::simd_ternary_group_matvec;

    fn fixture(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(rows, cols);
        let mut s = seed;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s >> 33
        };
        for r in 0..rows {
            for c in 0..cols {
                match next() % 3 {
                    0 => w.set(r, c, -1),
                    1 => {}
                    _ => w.set(r, c, 1),
                }
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 0.01 + (r % 7) as f32 * 0.03);
            }
        }
        w
    }

    fn activations(cols: usize) -> Vec<f32> {
        // Mixed magnitudes across groups, one all-zero group.
        (0..cols)
            .map(|i| match i / GROUP_SIZE {
                1 => 0.0,
                g => (((i * 37) % 101) as f32 * 0.01 - 0.5) * (1.0 + g as f32),
            })
            .collect()
    }


    /// Issue 936: the spin-pool arm is the rayon arm, bit-for-bit.
    #[cfg(feature = "decode_spin_pool")]
    #[test]
    fn spin_pool_arm_matches_rayon_arm() {
        let pool = crate::simd::spin_pool::SpinPool::new(3);
        for &(rows, cols) in &[(300, 256), (1025, 384)] {
            let w = fixture(rows, cols, 0x936 ^ rows as u64);
            let x: Vec<f32> = (0..cols).map(|i| ((i * 37) % 101) as f32 * 0.01 - 0.5).collect();
            let mut want = vec![0.0f32; rows];
            simd_ternary_group_matvec_q8_parallel(&w, &x, &mut want);
            let mut got = vec![f32::NAN; rows];
            crate::simd::spin_pool::with_spin_pool(&pool, || simd_ternary_group_matvec_q8_parallel(&w, &x, &mut got));
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{rows}x{cols}");
        }
    }
    #[test]
    fn quantize_round_trips_within_half_a_step() {
        let x = activations(512);
        let mut xq = vec![0i8; 512];
        let mut xs = vec![0.0f32; 4];
        quantize_activations_q8(&x, &mut xq, &mut xs);
        assert_eq!(xs[1], 0.0, "all-zero group must get scale 0");
        for c in 0..512 {
            let g = c / GROUP_SIZE;
            let err = (xq[c] as f32 * xs[g] - x[c]).abs();
            assert!(err <= 0.5 * xs[g] + 1e-6, "col {c}: err {err} > half step {}", xs[g]);
        }
        // The group max hits ±127 exactly.
        assert!(xq[..GROUP_SIZE].iter().any(|&q| q.unsigned_abs() == 127));
    }

    /// The integer inner sum makes NEON / AVX2 and scalar bit-identical.
    /// Elsewhere the entry point is the f32 kernel (no q8 port to compare).
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn simd_matches_scalar_q8_bit_identical() {
        if !q8_fast_path_available() {
            // aarch64: every Apple-silicon / ARMv8.4 CPU has dotprod — a miss
            // there is a defect. x86_64 without AVX2 legitimately has no port.
            #[cfg(target_arch = "aarch64")]
            panic!("aarch64 CPU without dotprod");
            #[cfg(target_arch = "x86_64")]
            {
                eprintln!("SKIP: no avx2 — the q8 entry point is the f32 kernel here");
                return;
            }
        }
        for &(rows, cols) in &[(1, 128), (37, 256), (300, 640), (1023, 384)] {
            let w = fixture(rows, cols, 0x935 ^ rows as u64);
            let x = activations(cols);
            let mut xq = vec![0i8; cols];
            let mut xs = vec![0.0f32; cols / GROUP_SIZE];
            quantize_activations_q8(&x, &mut xq, &mut xs);
            let mut y_ref = vec![0.0f32; rows];
            ternary_group_q8_rows_scalar(&w, &xq, &xs, &mut y_ref, 0);
            let mut y = vec![0.0f32; rows];
            simd_ternary_group_matvec_q8_parallel(&w, &x, &mut y);
            for r in 0..rows {
                assert_eq!(y[r].to_bits(), y_ref[r].to_bits(), "{rows}x{cols} row {r}");
            }
        }
    }

    /// Worst-case magnitudes: every weight ±1 and every |xq| = 127 — the i16
    /// partials (AVX2 `pmaddubsw` chain) and i32 sums must not saturate.
    #[test]
    fn saturated_extremes_match_exact_integer_sum() {
        let (rows, cols) = (64, 512);
        let mut w = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                w.set(r, c, if r % 2 == 0 { 1 } else { -1 });
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 1.0);
            }
        }
        // x = ±1 alternating by row parity of the sign → every xq = ±127.
        let x: Vec<f32> = (0..cols).map(|c| if (c / 7) % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let mut xq = vec![0i8; cols];
        let mut xs = vec![0.0f32; cols / GROUP_SIZE];
        quantize_activations_q8(&x, &mut xq, &mut xs);
        assert!(xq.iter().all(|&q| q.unsigned_abs() == 127));
        let mut y_ref = vec![0.0f32; rows];
        ternary_group_q8_rows_scalar(&w, &xq, &xs, &mut y_ref, 0);
        let mut y = vec![0.0f32; rows];
        simd_ternary_group_matvec_q8_parallel(&w, &x, &mut y);
        let mut y_f32 = vec![0.0f32; rows];
        simd_ternary_group_matvec(&w, &x, &mut y_f32);
        for r in 0..rows {
            // ±1 activations quantize exactly, so every arm agrees with f32 too.
            assert!((y[r] - y_f32[r]).abs() <= 1e-3, "row {r}: {} vs f32 {}", y[r], y_f32[r]);
            if q8_fast_path_available() {
                assert_eq!(y[r].to_bits(), y_ref[r].to_bits(), "row {r}");
            }
        }
    }

    /// The loss vs the f32 kernel is activation rounding only: bounded by half
    /// a quantization step per element times the row's |weight| mass.
    #[test]
    fn q8_tracks_f32_kernel_within_rounding_bound() {
        let (rows, cols) = (512, 1024);
        let w = fixture(rows, cols, 0x9350);
        let x = activations(cols);
        let mut xq = vec![0i8; cols];
        let mut xs = vec![0.0f32; cols / GROUP_SIZE];
        quantize_activations_q8(&x, &mut xq, &mut xs);
        let mut y_f32 = vec![0.0f32; rows];
        simd_ternary_group_matvec(&w, &x, &mut y_f32);
        let mut y_q8 = vec![0.0f32; rows];
        simd_ternary_group_matvec_q8_parallel(&w, &x, &mut y_q8);
        for r in 0..rows {
            let mut bound = 1e-4f32;
            for g in 0..w.groups_per_row {
                let ws = w.group_scale[r * w.groups_per_row + g].to_f32();
                bound += ws * 0.5 * xs[g] * GROUP_SIZE as f32;
            }
            let err = (y_q8[r] - y_f32[r]).abs();
            assert!(err <= bound, "row {r}: |q8 − f32| = {err} > bound {bound}");
        }
    }

    /// A ragged shape (cols % 128 != 0) takes the f32 kernel — bit-identical to it.
    #[test]
    fn ragged_cols_fall_back_to_f32_kernel() {
        let (rows, cols) = (300, 200);
        let w = fixture(rows, cols, 0x9351);
        let x: Vec<f32> = (0..cols).map(|i| (i % 13) as f32 * 0.07 - 0.4).collect();
        let mut y_f32 = vec![0.0f32; rows];
        simd_ternary_group_matvec(&w, &x, &mut y_f32);
        let mut y = vec![0.0f32; rows];
        simd_ternary_group_matvec_q8_parallel(&w, &x, &mut y);
        for r in 0..rows {
            assert_eq!(y[r].to_bits(), y_f32[r].to_bits(), "row {r}");
        }
    }
}
