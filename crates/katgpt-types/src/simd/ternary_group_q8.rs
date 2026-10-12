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
//! falls through to the f32 kernel.

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

/// Dispatch one row range over pre-quantized activations.
#[inline]
fn q8_row_range(w: &TernaryGroupWeights, xq: &[i8], xs: &[f32], y: &mut [f32], row_offset: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: dotprod detected; the caller checked the shape contract.
            unsafe { neon_q8_row_range(w, xq, xs, y, row_offset) };
            return;
        }
    }
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
/// rayon row chunks (the same capped split as the f32 kernel). Shapes the fast
/// path cannot take (`cols % 128 != 0`) fall back to the f32 kernel.
pub fn simd_ternary_group_matvec_q8_parallel(w: &TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;

    /// Below this row count, rayon's task overhead outweighs the work.
    const PARALLEL_ROW_MIN: usize = 256;

    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    if !w.cols.is_multiple_of(GROUP_SIZE) {
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

    /// The integer inner sum makes NEON and scalar bit-identical.
    #[test]
    fn neon_matches_scalar_q8_bit_identical() {
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
