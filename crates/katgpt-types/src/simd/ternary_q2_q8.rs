//! Ternary matvec over **raw Q2_0 blocks** with int8 activations
//! (`ternary_q2_q8`, Issue 935 T5).
//!
//! The bit-plane q8 kernel ([`super::ternary_group_q8`]) pays ~7 ops per 16
//! weights just to rebuild a {-1,0,+1} byte from two planes (`tbl`+`tst` per
//! plane, then `sub`). The GGUF `Q2_0` block already stores each weight as a
//! 2-bit code `q = w + 1` (4 per byte, LSB-first), so a byte of codes is one
//! `shr`+`and` away — no plane walk, no weight repack. The only change is on
//! the ACTIVATION side, paid once per call: `xq` is permuted so the codes at
//! shift `2s` of 16 consecutive bytes meet 16 consecutive activations.
//!
//! # Layout
//!
//! [`Q2Block`] is ggml's `block_q2_0` byte-for-byte: `d` (f16 bits) then
//! `qs[32]`, weight `j` at `(qs[j/4] >> 2(j%4)) & 3`. Within a 128-group, half
//! `h ∈ {0,1}` = `qs[16h..16h+16]`; shift `s` of byte `16h + i` is column
//! `64h + 4i + s`. The permuted activations are
//! `xperm[32s + 16h + i] = xq[64h + 4i + s]` — contiguous 16 for NEON
//! (`sdot` per (s,h)), contiguous 32 for AVX2 (`pmaddubsw` per s).
//!
//! # Semantics
//!
//! `Σ_c (q_c − 1)·xq_c = Σ_c q_c·xq_c − Σ_c xq_c`: the per-group activation
//! sum is computed once per call, so the inner sum is the same exact integer
//! the bit-plane kernel computes. Output is **bit-identical** to
//! [`super::ternary_group_q8::ternary_group_q8_rows_scalar`] over the same
//! weights (asserted below) — this tier changes speed, not numbers.
//!
//! Code `3` (`+2`, ggml's fourth state) evaluates as `+2` exactly like
//! ggml's dequant; [`TernaryQ2Weights::from_group`] never produces it.

#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;

use half::f16;

use super::ternary_group_q8::quantize_activations_q8;
use crate::{GROUP_SIZE, TernaryGroupWeights};

/// Bytes of 2-bit codes per 128-weight group.
const QS_BYTES: usize = GROUP_SIZE / 4;

/// One 128-weight group in ggml `block_q2_0` layout (34 bytes, align 2).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q2Block {
    /// Group scale, f16 bits.
    pub d: u16,
    /// 2-bit codes `q = w + 1`, 4 per byte, LSB-first.
    pub qs: [u8; QS_BYTES],
}

const _: () = assert!(core::mem::size_of::<Q2Block>() == 34);

/// Ternary weights as row-major Q2_0 blocks: `blocks[r * groups_per_row + g]`.
///
/// Requires `cols % GROUP_SIZE == 0` (Q2_0 has no ragged tail).
#[derive(Clone, Debug)]
pub struct TernaryQ2Weights {
    pub rows: usize,
    pub cols: usize,
    pub groups_per_row: usize,
    pub blocks: Vec<Q2Block>,
}

impl TernaryQ2Weights {
    /// Wrap pre-built blocks (e.g. copied from a GGUF `Q2_0` tensor).
    ///
    /// # Panics
    /// `cols % GROUP_SIZE != 0` or `blocks.len() != rows * cols / GROUP_SIZE`.
    pub fn from_blocks(rows: usize, cols: usize, blocks: Vec<Q2Block>) -> Self {
        assert!(cols.is_multiple_of(GROUP_SIZE), "Q2_0 needs cols % {GROUP_SIZE} == 0, got {cols}");
        let groups_per_row = cols / GROUP_SIZE;
        assert_eq!(blocks.len(), rows * groups_per_row, "block count must be rows * cols / {GROUP_SIZE}");
        Self { rows, cols, groups_per_row, blocks }
    }

    /// Repack bit-plane weights (lossless: the scale is copied verbatim).
    ///
    /// # Panics
    /// `w.cols % GROUP_SIZE != 0`, or a column set in both planes (not ternary).
    pub fn from_group(w: &TernaryGroupWeights) -> Self {
        assert!(w.cols.is_multiple_of(GROUP_SIZE), "Q2_0 needs cols % {GROUP_SIZE} == 0, got {}", w.cols);
        let gpr = w.cols / GROUP_SIZE;
        let mut blocks = Vec::with_capacity(w.rows * gpr);
        for r in 0..w.rows {
            for g in 0..gpr {
                let mut qs = [0u8; QS_BYTES];
                for j in 0..GROUP_SIZE {
                    let c = g * GROUP_SIZE + j;
                    let idx = r * w.blocks64 + (c >> 6);
                    let bit = 1u64 << (c & 63);
                    let (p, n) = (w.pos_bits[idx] & bit != 0, w.neg_bits[idx] & bit != 0);
                    assert!(!(p && n), "row {r} col {c}: set in both planes");
                    let q = match (p, n) {
                        (true, _) => 2u8,
                        (_, true) => 0,
                        _ => 1,
                    };
                    qs[j / 4] |= q << (2 * (j % 4));
                }
                blocks.push(Q2Block { d: w.group_scale[r * w.groups_per_row + g].to_bits(), qs });
            }
        }
        Self { rows: w.rows, cols: w.cols, groups_per_row: gpr, blocks }
    }

    /// Inverse of [`Self::from_group`] (lossless). Code `3` (`+2`) has no
    /// bit-plane encoding.
    ///
    /// # Panics
    /// A block carries code `3`.
    pub fn to_group(&self) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(self.rows, self.cols);
        for r in 0..self.rows {
            for g in 0..self.groups_per_row {
                let b = &self.blocks[r * self.groups_per_row + g];
                w.group_scale[r * w.groups_per_row + g] = f16::from_bits(b.d);
                for j in 0..GROUP_SIZE {
                    match (b.qs[j / 4] >> (2 * (j % 4))) & 3 {
                        0 => w.set(r, g * GROUP_SIZE + j, -1),
                        1 => {}
                        2 => w.set(r, g * GROUP_SIZE + j, 1),
                        _ => panic!("row {r} group {g} weight {j}: code 3 (+2) is not ternary"),
                    }
                }
            }
        }
        w
    }

    /// Whether every code is ternary (no code `3`) — the contract the
    /// bit-plane repack enforces at load. Branch-free per byte: a pair is
    /// `0b11` iff both of its bits are set.
    pub fn is_ternary(&self) -> bool {
        self.blocks.iter().all(|b| b.qs.iter().all(|&q| q & (q >> 1) & 0x55 == 0))
    }

    /// Storage bytes (34 per 128 weights = 2.125 bits/weight).
    pub fn storage_bytes(&self) -> usize {
        self.blocks.len() * core::mem::size_of::<Q2Block>()
    }
}

/// Permute per-group int8 activations into the code-shift order (module docs)
/// and sum each group. `xperm.len() == xq.len()`, `xsum.len() == xq.len() / 128`.
pub fn permute_activations_q2(xq: &[i8], xperm: &mut [i8], xsum: &mut [i32]) {
    debug_assert_eq!(xq.len() % GROUP_SIZE, 0);
    debug_assert_eq!(xperm.len(), xq.len());
    debug_assert_eq!(xsum.len(), xq.len() / GROUP_SIZE);
    for ((src, dst), sum) in xq
        .as_chunks::<GROUP_SIZE>()
        .0
        .iter()
        .zip(xperm.as_chunks_mut::<GROUP_SIZE>().0.iter_mut())
        .zip(xsum.iter_mut())
    {
        for s in 0..4 {
            for h in 0..2 {
                for i in 0..16 {
                    dst[32 * s + 16 * h + i] = src[64 * h + 4 * i + s];
                }
            }
        }
        *sum = src.iter().map(|&v| v as i32).sum();
    }
}

/// Scalar reference over the UNPERMUTED activations — rows `row_offset ..
/// row_offset + y.len()`. Same accumulation order as the bit-plane q8 scalar.
pub fn ternary_q2_q8_rows_scalar(w: &TernaryQ2Weights, xq: &[i8], xs: &[f32], y: &mut [f32], row_offset: usize) {
    for (i, y_slot) in y.iter_mut().enumerate() {
        let r = row_offset + i;
        let mut row_sum = 0.0f32;
        for g in 0..w.groups_per_row {
            let b = &w.blocks[r * w.groups_per_row + g];
            let mut isum = 0i32;
            for j in 0..GROUP_SIZE {
                let q = ((b.qs[j / 4] >> (2 * (j % 4))) & 3) as i32;
                isum += (q - 1) * xq[g * GROUP_SIZE + j] as i32;
            }
            let scale = f16::from_bits(b.d).to_f32() * xs[g];
            row_sum += scale * isum as f32;
        }
        *y_slot = row_sum;
    }
}

/// NEON + SDOT row range over permuted activations.
///
/// # Safety
/// CPU supports `dotprod`; `xperm.len() == w.cols`; `xsum`/`xs` have
/// `w.groups_per_row` entries; `row_offset + y.len() <= w.rows`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn neon_q2_row_range(
    w: &TernaryQ2Weights,
    xperm: &[i8],
    xsum: &[i32],
    xs: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    use core::arch::aarch64::*;
    debug_assert_eq!(xperm.len(), w.cols);
    debug_assert!(row_offset + y.len() <= w.rows);
    unsafe {
        let m3 = vdupq_n_u8(3);
        let xp = xperm.as_ptr();
        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let blocks = w.blocks.as_ptr().add(r * w.groups_per_row);
            let mut row_sum = 0.0f32;
            for g in 0..w.groups_per_row {
                let b = &*blocks.add(g);
                let lo = vld1q_u8(b.qs.as_ptr());
                let hi = vld1q_u8(b.qs.as_ptr().add(16));
                let x = xp.add(g * GROUP_SIZE);
                let c = |v: uint8x16_t| vreinterpretq_s8_u8(v);
                // Two accumulators: the (h=0, h=1) sdot chains run in parallel.
                let mut a0 = vdotq_s32(vdupq_n_s32(0), c(vandq_u8(lo, m3)), vld1q_s8(x));
                let mut a1 = vdotq_s32(vdupq_n_s32(0), c(vandq_u8(hi, m3)), vld1q_s8(x.add(16)));
                a0 = vdotq_s32(a0, c(vandq_u8(vshrq_n_u8::<2>(lo), m3)), vld1q_s8(x.add(32)));
                a1 = vdotq_s32(a1, c(vandq_u8(vshrq_n_u8::<2>(hi), m3)), vld1q_s8(x.add(48)));
                a0 = vdotq_s32(a0, c(vandq_u8(vshrq_n_u8::<4>(lo), m3)), vld1q_s8(x.add(64)));
                a1 = vdotq_s32(a1, c(vandq_u8(vshrq_n_u8::<4>(hi), m3)), vld1q_s8(x.add(80)));
                a0 = vdotq_s32(a0, c(vshrq_n_u8::<6>(lo)), vld1q_s8(x.add(96)));
                a1 = vdotq_s32(a1, c(vshrq_n_u8::<6>(hi)), vld1q_s8(x.add(112)));
                let isum = vaddvq_s32(vaddq_s32(a0, a1)) - xsum[g];
                let scale = f16::from_bits(b.d).to_f32() * xs[g];
                row_sum += scale * isum as f32;
            }
            *y_slot = row_sum;
        }
    }
}

/// AVX2 row range over permuted activations. Per shift: `psrlw`+`pand` →
/// 32 u8 codes, `pmaddubsw` (u8 code × i8 x) → i16 pairs (|·| ≤ 762); four
/// shifts sum to ≤ 3048 in i16, widened once per group by `pmaddwd`.
///
/// # Safety
/// CPU supports `avx2`; same shape contract as [`neon_q2_row_range`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_q2_row_range(
    w: &TernaryQ2Weights,
    xperm: &[i8],
    xsum: &[i32],
    xs: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(xperm.len(), w.cols);
    debug_assert!(row_offset + y.len() <= w.rows);
    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let ones_i16 = _mm256_set1_epi16(1);
        let xp = xperm.as_ptr();
        for (i, y_slot) in y.iter_mut().enumerate() {
            let r = row_offset + i;
            let blocks = w.blocks.as_ptr().add(r * w.groups_per_row);
            let mut row_sum = 0.0f32;
            for g in 0..w.groups_per_row {
                let b = &*blocks.add(g);
                let v = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);
                let x = xp.add(g * GROUP_SIZE) as *const __m256i;
                let c0 = _mm256_and_si256(v, m3);
                let c1 = _mm256_and_si256(_mm256_srli_epi16::<2>(v), m3);
                let c2 = _mm256_and_si256(_mm256_srli_epi16::<4>(v), m3);
                let c3 = _mm256_and_si256(_mm256_srli_epi16::<6>(v), m3);
                let p01 = _mm256_add_epi16(
                    _mm256_maddubs_epi16(c0, _mm256_loadu_si256(x)),
                    _mm256_maddubs_epi16(c1, _mm256_loadu_si256(x.add(1))),
                );
                let p23 = _mm256_add_epi16(
                    _mm256_maddubs_epi16(c2, _mm256_loadu_si256(x.add(2))),
                    _mm256_maddubs_epi16(c3, _mm256_loadu_si256(x.add(3))),
                );
                let acc32 = _mm256_madd_epi16(_mm256_add_epi16(p01, p23), ones_i16);
                let s128 = _mm_add_epi32(_mm256_castsi256_si128(acc32), _mm256_extracti128_si256::<1>(acc32));
                let s64 = _mm_add_epi32(s128, _mm_unpackhi_epi64(s128, s128));
                let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32::<0b01>(s64));
                let isum = _mm_cvtsi128_si32(s32) - xsum[g];
                let scale = f16::from_bits(b.d).to_f32() * xs[g];
                row_sum += scale * isum as f32;
            }
            *y_slot = row_sum;
        }
    }
}

/// Whether a SIMD arm runs on this CPU (aarch64 `dotprod` / x86_64 `avx2`,
/// runtime-probed). Public so a loader can decline the raw-Q2 layout where
/// only the scalar reference would run: there the bit-plane q8 entry falls
/// back to the f32 kernel instead, a different (and faster) arm.
#[inline]
pub fn q2_fast_path_available() -> bool {
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

/// One row range: the SIMD arm when available, else the scalar reference.
#[inline]
fn q2_row_range(
    w: &TernaryQ2Weights,
    xq: &[i8],
    xperm: &[i8],
    xsum: &[i32],
    xs: &[f32],
    y: &mut [f32],
    row_offset: usize,
) {
    if !q2_fast_path_available() {
        return ternary_q2_q8_rows_scalar(w, xq, xs, y, row_offset);
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: dotprod checked above; shapes built by the entry point.
    unsafe {
        neon_q2_row_range(w, xperm, xsum, xs, y, row_offset)
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: avx2 checked above; shapes built by the entry point.
    unsafe {
        avx2_q2_row_range(w, xperm, xsum, xs, y, row_offset)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = (xperm, xsum);
}

/// Per-call activation scratch: `xq`, `xperm`, `xs`, `xsum`.
type Q2Scratch = (Vec<i8>, Vec<i8>, Vec<f32>, Vec<i32>);

thread_local! {
    /// Per-calling-thread activation scratch, reused across calls (zero
    /// steady-state allocation).
    static Q2_SCRATCH: RefCell<Q2Scratch> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())) };
}

/// Below this row count, rayon's task overhead outweighs the work.
const PARALLEL_ROW_MIN: usize = 256;

/// The prepared int8 activation operands `(xq, xperm, xsum, xs)` of one `x`.
type Q2Acts<'a> = (&'a [i8], &'a [i8], &'a [i32], &'a [f32]);

/// Quantize + permute `x` once into the per-thread scratch and hand the
/// operands to `f` — shared by the single and the pair entry points.
fn with_q2_activations<R>(x: &[f32], f: impl FnOnce(Q2Acts<'_>) -> R) -> R {
    Q2_SCRATCH.with_borrow_mut(|(xq, xperm, xs, xsum)| {
        let groups = x.len() / GROUP_SIZE;
        xq.resize(x.len(), 0);
        xperm.resize(x.len(), 0);
        xs.resize(groups, 0.0);
        xsum.resize(groups, 0);
        quantize_activations_q8(x, xq, xs);
        permute_activations_q2(xq, xperm, xsum);
        f((&xq[..], &xperm[..], &xsum[..], &xs[..]))
    })
}

/// Row-parallel ternary matvec over Q2_0 blocks with int8 activations:
/// `y ≈ W × x`, bit-identical to the bit-plane q8 kernel on the same weights.
pub fn simd_ternary_q2_matvec_q8_parallel(w: &TernaryQ2Weights, x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;

    assert_eq!(x.len(), w.cols, "x vector length must match weight cols");
    assert_eq!(y.len(), w.rows, "y vector length must match weight rows");
    with_q2_activations(x, |(xq, xperm, xsum, xs)| {
        if w.rows < PARALLEL_ROW_MIN {
            q2_row_range(w, xq, xperm, xsum, xs, y, 0);
            return;
        }
        #[cfg(feature = "decode_spin_pool")]
        if q2_rows_on_spin_pool([w], &mut [&mut *y], (xq, xperm, xsum, xs)) {
            return;
        }
        let chunk = super::par_matvec_chunk(w.rows);
        y.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, y_chunk)| q2_row_range(w, xq, xperm, xsum, xs, y_chunk, ci * chunk));
    });
}

/// The row-parallel body on this thread's installed [`super::spin_pool`]
/// (Issue 936): fixed [`super::PAR_MATVEC_CHUNK_ROWS`]-row tasks over all `N`
/// matrices, one flat task index each. `false` (nothing run) when no pool is
/// installed. Same per-row kernel as the rayon arm ⇒ bit-identical.
#[cfg(feature = "decode_spin_pool")]
fn q2_rows_on_spin_pool<const N: usize>(ws: [&TernaryQ2Weights; N], ys: &mut [&mut [f32]; N], acts: Q2Acts<'_>) -> bool {
    use super::spin_pool::SyncMutPtr;
    const CHUNK: usize = super::PAR_MATVEC_CHUNK_ROWS;
    let (xq, xperm, xsum, xs) = acts;
    let outs: [(SyncMutPtr<f32>, usize); N] = std::array::from_fn(|i| (SyncMutPtr(ys[i].as_mut_ptr()), ys[i].len()));
    let n_tasks: usize = ws.iter().map(|w| w.rows.div_ceil(CHUNK)).sum();
    super::spin_pool::spin_pool_run(n_tasks, &|mut t| {
        for (w, &(ptr, len)) in ws.iter().zip(&outs) {
            let chunks = w.rows.div_ceil(CHUNK);
            if t >= chunks {
                t -= chunks;
                continue;
            }
            let row0 = t * CHUNK;
            let rows = CHUNK.min(len - row0);
            // SAFETY: task t of matrix w owns rows [row0, row0 + rows) of its
            // output alone; `ys` outlives the run (spin_pool_run returns only
            // after every task finished).
            let y = unsafe { std::slice::from_raw_parts_mut(ptr.0.add(row0), rows) };
            q2_row_range(w, xq, xperm, xsum, xs, y, row0);
            return;
        }
    })
}

/// `N` Q2_0 matvecs over ONE shared `x` (an FFN's gate + up, a `DeltaNet`
/// layer's qkv + z, an attention layer's q/k/v): `ys[i] ≈ ws[i] × x` with one
/// activation quantize+permute and one fork-join instead of `N`. Rows are
/// independent, so this is bit-identical to `N`
/// [`simd_ternary_q2_matvec_q8_parallel`] calls. Zero-allocation: the operand
/// sets are fixed-size arrays.
///
/// # Panics
/// Any `ws[i].cols != x.len()`, or `ys[i].len() != ws[i].rows`.
pub fn simd_ternary_q2_matvec_q8_parallel_n<const N: usize>(
    ws: [&TernaryQ2Weights; N],
    x: &[f32],
    mut ys: [&mut [f32]; N],
) {
    use rayon::prelude::*;

    for (w, y) in ws.iter().zip(&ys) {
        assert_eq!(x.len(), w.cols, "x vector length must match every weight's cols");
        assert_eq!(y.len(), w.rows, "ys[i] length must match ws[i] rows");
    }
    with_q2_activations(x, |(xq, xperm, xsum, xs)| {
        if ws.iter().map(|w| w.rows).sum::<usize>() < PARALLEL_ROW_MIN {
            for (w, y) in ws.iter().zip(ys.iter_mut()) {
                q2_row_range(w, xq, xperm, xsum, xs, y, 0);
            }
            return;
        }
        #[cfg(feature = "decode_spin_pool")]
        if q2_rows_on_spin_pool(ws, &mut ys, (xq, xperm, xsum, xs)) {
            return;
        }
        ys.par_iter_mut()
            .zip(ws.par_iter())
            .flat_map(|(y, &w)| {
                let chunk = super::par_matvec_chunk(w.rows);
                y.par_chunks_mut(chunk)
                    .enumerate()
                    .map(move |(ci, c)| (w, c, ci * chunk))
            })
            .for_each(|(w, y_chunk, row0)| q2_row_range(w, xq, xperm, xsum, xs, y_chunk, row0));
    });
}

/// Two Q2_0 matvecs over ONE shared `x` (an FFN's gate and up projections) —
/// [`simd_ternary_q2_matvec_q8_parallel_n`] at `N = 2`, bit-identical to two
/// [`simd_ternary_q2_matvec_q8_parallel`] calls.
///
/// # Panics
/// `wa.cols != wb.cols`, or a length mismatch against `x` / `ya` / `yb`.
pub fn simd_ternary_q2_matvec_q8_parallel_pair(
    wa: &TernaryQ2Weights,
    wb: &TernaryQ2Weights,
    x: &[f32],
    ya: &mut [f32],
    yb: &mut [f32],
) {
    simd_ternary_q2_matvec_q8_parallel_n([wa, wb], x, [ya, yb]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::ternary_group_q8::ternary_group_q8_rows_scalar;

    fn fixture(rows: usize, cols: usize, seed: u64) -> TernaryGroupWeights {
        let mut w = TernaryGroupWeights::new(rows, cols);
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        (0..cols)
            .map(|i| match i / GROUP_SIZE {
                1 => 0.0,
                g => (((i * 37) % 101) as f32 * 0.01 - 0.5) * (1.0 + g as f32),
            })
            .collect()
    }

    fn quantized(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let mut xq = vec![0i8; x.len()];
        let mut xs = vec![0.0f32; x.len() / GROUP_SIZE];
        quantize_activations_q8(x, &mut xq, &mut xs);
        (xq, xs)
    }

    #[test]
    fn permutation_is_a_bijection_and_sums_match() {
        let xq: Vec<i8> = (0..256).map(|i| (i % 251) as i8).collect();
        let mut xp = vec![0i8; 256];
        let mut sum = vec![0i32; 2];
        permute_activations_q2(&xq, &mut xp, &mut sum);
        for g in 0..2 {
            let mut a: Vec<i8> = xq[g * 128..(g + 1) * 128].to_vec();
            let mut b: Vec<i8> = xp[g * 128..(g + 1) * 128].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "group {g} must be a permutation");
            assert_eq!(sum[g], xq[g * 128..(g + 1) * 128].iter().map(|&v| v as i32).sum::<i32>());
        }
        // Spot-check the documented index map: xperm[32s + 16h + i] = xq[64h + 4i + s].
        assert_eq!(xp[32 * 3 + 16 + 5], xq[64 + 4 * 5 + 3]);
    }

    #[test]
    fn from_group_round_trips_the_ggml_code_mapping() {
        let w = fixture(3, 256, 7);
        let q = TernaryQ2Weights::from_group(&w);
        assert_eq!(q.storage_bytes(), 3 * 2 * 34);
        assert!(q.is_ternary());
        let back = q.to_group();
        assert_eq!((back.pos_bits.clone(), back.neg_bits.clone()), (w.pos_bits.clone(), w.neg_bits.clone()));
        assert_eq!(back.group_scale, w.group_scale);
        let mut bad = q.clone();
        bad.blocks[1].qs[7] |= 0b11 << 4;
        assert!(!bad.is_ternary(), "a 0b11 pair must be refused");
        for r in 0..3 {
            for c in 0..256 {
                let b = &q.blocks[r * 2 + c / 128];
                let j = c % 128;
                let code = (b.qs[j / 4] >> (2 * (j % 4))) & 3;
                assert_eq!(code as i8 - 1, w.get(r, c), "row {r} col {c}");
            }
        }
    }

    /// The pair entry is two single calls, bit-for-bit — unequal row counts,
    /// the serial (< 256 rows total) and the parallel posture.
    #[test]
    fn pair_matches_two_single_calls() {
        for &(ra, rb, cols) in &[(3, 5, 128), (100, 155, 256), (300, 700, 640), (1024, 1024, 384)] {
            let a = TernaryQ2Weights::from_group(&fixture(ra, cols, 0xA5 ^ ra as u64));
            let b = TernaryQ2Weights::from_group(&fixture(rb, cols, 0x5A ^ rb as u64));
            let x = activations(cols);
            let (mut ya, mut yb) = (vec![0.0f32; ra], vec![0.0f32; rb]);
            simd_ternary_q2_matvec_q8_parallel(&a, &x, &mut ya);
            simd_ternary_q2_matvec_q8_parallel(&b, &x, &mut yb);
            let (mut pa, mut pb) = (vec![f32::NAN; ra], vec![f32::NAN; rb]);
            simd_ternary_q2_matvec_q8_parallel_pair(&a, &b, &x, &mut pa, &mut pb);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&pa), bits(&ya), "{ra}x{cols} a");
            assert_eq!(bits(&pb), bits(&yb), "{rb}x{cols} b");
        }
    }

    /// The N-way entry at N = 3 (attention q/k/v: one wide, two narrow) is
    /// three single calls, bit-for-bit — serial and parallel postures.
    #[test]
    fn n_way_matches_single_calls() {
        for &(rq, rkv, cols) in &[(4, 2, 128), (60, 30, 256), (1024, 256, 384)] {
            let shapes = [rq, rkv, rkv];
            let ws: Vec<_> = shapes
                .iter()
                .enumerate()
                .map(|(i, &r)| TernaryQ2Weights::from_group(&fixture(r, cols, 0x3A ^ (i * 977 + r) as u64)))
                .collect();
            let x = activations(cols);
            let singles: Vec<Vec<f32>> = ws
                .iter()
                .map(|w| {
                    let mut y = vec![0.0f32; w.rows];
                    simd_ternary_q2_matvec_q8_parallel(w, &x, &mut y);
                    y
                })
                .collect();
            let (mut q, mut k, mut v) = (vec![f32::NAN; rq], vec![f32::NAN; rkv], vec![f32::NAN; rkv]);
            simd_ternary_q2_matvec_q8_parallel_n([&ws[0], &ws[1], &ws[2]], &x, [&mut q, &mut k, &mut v]);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            for (i, got) in [&q, &k, &v].into_iter().enumerate() {
                assert_eq!(bits(got), bits(&singles[i]), "{rq}/{rkv}x{cols} arm {i}");
            }
        }
    }

    /// Issue 936: the spin-pool arm is the rayon arm, bit-for-bit — single and
    /// N = 3 entries, row counts that do and do not divide the 32-row task.
    #[cfg(feature = "decode_spin_pool")]
    #[test]
    fn spin_pool_arm_matches_rayon_arm() {
        let pool = super::super::spin_pool::SpinPool::new(3);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        for &(r0, r1, cols) in &[(300, 77, 256), (1024, 256, 384), (513, 513, 128)] {
            let ws: Vec<_> = [r0, r1, r1]
                .iter()
                .enumerate()
                .map(|(i, &r)| TernaryQ2Weights::from_group(&fixture(r, cols, 0x936 ^ (i * 31 + r) as u64)))
                .collect();
            let x = activations(cols);
            let mut want = vec![0.0f32; r0];
            simd_ternary_q2_matvec_q8_parallel(&ws[0], &x, &mut want);
            let mut got = vec![f32::NAN; r0];
            super::super::spin_pool::with_spin_pool(&pool, || simd_ternary_q2_matvec_q8_parallel(&ws[0], &x, &mut got));
            assert_eq!(bits(&got), bits(&want), "single {r0}x{cols}");

            let (mut a, mut b, mut c) = (vec![0.0f32; r0], vec![0.0f32; r1], vec![0.0f32; r1]);
            simd_ternary_q2_matvec_q8_parallel_n([&ws[0], &ws[1], &ws[2]], &x, [&mut a, &mut b, &mut c]);
            let (mut sa, mut sb, mut sc) = (vec![f32::NAN; r0], vec![f32::NAN; r1], vec![f32::NAN; r1]);
            super::super::spin_pool::with_spin_pool(&pool, || {
                simd_ternary_q2_matvec_q8_parallel_n([&ws[0], &ws[1], &ws[2]], &x, [&mut sa, &mut sb, &mut sc])
            });
            assert_eq!((bits(&sa), bits(&sb), bits(&sc)), (bits(&a), bits(&b), bits(&c)), "n-way {r0}/{r1}x{cols}");
        }
    }

    /// The tier's whole claim: bit-identical to the bit-plane q8 scalar on the
    /// same weights — scalar reference AND the dispatched SIMD arm.
    #[test]
    fn matches_bitplane_q8_bit_identical() {
        for &(rows, cols) in &[(1, 128), (37, 256), (300, 640), (1023, 384)] {
            let w = fixture(rows, cols, 0x9355 ^ rows as u64);
            let q = TernaryQ2Weights::from_group(&w);
            let x = activations(cols);
            let (xq, xs) = quantized(&x);
            let mut y_ref = vec![0.0f32; rows];
            ternary_group_q8_rows_scalar(&w, &xq, &xs, &mut y_ref, 0);
            let mut y_scalar = vec![0.0f32; rows];
            ternary_q2_q8_rows_scalar(&q, &xq, &xs, &mut y_scalar, 0);
            let mut y = vec![0.0f32; rows];
            simd_ternary_q2_matvec_q8_parallel(&q, &x, &mut y);
            for r in 0..rows {
                assert_eq!(y_scalar[r].to_bits(), y_ref[r].to_bits(), "scalar {rows}x{cols} row {r}");
                assert_eq!(y[r].to_bits(), y_ref[r].to_bits(), "simd {rows}x{cols} row {r}");
            }
        }
    }

    /// Worst-case magnitudes: every |xq| = 127, every weight ±1 — no i16
    /// (AVX2) or i32 saturation, and the `− xsum` correction holds.
    #[test]
    fn saturated_extremes_bit_identical() {
        let (rows, cols) = (64, 512);
        let mut w = TernaryGroupWeights::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                w.set(r, c, if (r + c / 3) % 2 == 0 { 1 } else { -1 });
            }
            for g in 0..w.groups_per_row {
                w.set_scale(r, g, 1.0);
            }
        }
        let q = TernaryQ2Weights::from_group(&w);
        let x: Vec<f32> = (0..cols).map(|c| if (c / 7) % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let (xq, xs) = quantized(&x);
        assert!(xq.iter().all(|&v| v.unsigned_abs() == 127));
        let mut y_ref = vec![0.0f32; rows];
        ternary_group_q8_rows_scalar(&w, &xq, &xs, &mut y_ref, 0);
        let mut y = vec![0.0f32; rows];
        simd_ternary_q2_matvec_q8_parallel(&q, &x, &mut y);
        for r in 0..rows {
            assert_eq!(y[r].to_bits(), y_ref[r].to_bits(), "row {r}");
        }
    }

    #[test]
    #[should_panic(expected = "Q2_0 needs cols")]
    fn ragged_cols_are_refused() {
        let _ = TernaryQ2Weights::from_group(&TernaryGroupWeights::new(4, 200));
    }
}
