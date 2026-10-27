//! SIMD argmax — single-pass `(usize, f32)` reducer used by token sampling.

use crate::simd::simd_max_f32;

/// Single-pass argmax: returns `(index, value)` of the maximum element.
///
/// Fuses max-finding and index-recovery into one traversal. The naive idiom —
/// `simd_max_f32(x)` followed by `x.iter().position(|&v| v == max)` — scans the
/// buffer twice (one SIMD max pass + one scalar equality scan that can run the
/// full length). On `aarch64` this uses a NEON kernel that tracks per-lane max
/// values *and* indices in one pass, measured at ~5× faster than the two-pass
/// idiom across vocab sizes 27 → 256k.
///
/// Default-on `argmax_blocked` (Issue 967) replaces the aarch64 lane-wise
/// kernel at n ≥ 512 with a blocked two-level kernel (8 independent max
/// accumulators per 256-float block, then a one-block first-index rescan):
/// 2.3–5.3× faster than the lane-wise pass at every max position, same
/// result ([Bench 958](../../.benchmarks/958_issue967_argmax_blocked_goat.md)).
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
    #[cfg(all(target_arch = "aarch64", feature = "argmax_blocked"))]
    {
        unsafe { neon_argmax_blocked_f32(x) }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "argmax_blocked")))]
    {
        unsafe { neon_argmax_f32(x) }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { wasm32_simd128_argmax_f32(x) }
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        two_pass_argmax_f32(x)
    }
}

/// The pre-Issue-967 aarch64 kernel (NEON lane-wise single pass), kept
/// callable so the A/B instrument (`tests/bench_967_argmax_blocked_ab.rs`)
/// measures it against the blocked kernel whatever the dispatch says. On every
/// other target this is `simd_argmax_f32` itself.
#[doc(hidden)]
#[inline]
pub fn simd_argmax_lanewise_f32(x: &[f32]) -> (usize, f32) {
    if x.is_empty() {
        return (0, f32::NEG_INFINITY);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_argmax_f32(x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        simd_argmax_f32(x)
    }
}

/// Issue 967 blocked two-level argmax (aarch64 NEON; other targets forward to
/// `simd_argmax_f32`). Same contract as `simd_argmax_f32`: first index whose
/// value `==` the maximum, `(0, -inf)` on empty.
#[doc(hidden)]
#[inline]
pub fn simd_argmax_blocked_f32(x: &[f32]) -> (usize, f32) {
    if x.is_empty() {
        return (0, f32::NEG_INFINITY);
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon_argmax_blocked_f32(x) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        simd_argmax_f32(x)
    }
}

/// Block width of the Issue-967 blocked argmax: 8 independent 4-lane
/// accumulators × 8 steps. Below two blocks the lane-wise kernel runs.
#[cfg(target_arch = "aarch64")]
const ARGMAX_BLOCK: usize = BLOCK_MAX_WIDTH;

/// Index of the first element `>= t` in `x`, or `None` (also for a NaN `t`
/// and for NaN elements, which compare false). aarch64: 16-wide `vcgeq_f32`
/// probes, then a scalar pass inside the hit chunk; elsewhere a scalar scan.
/// Sparse-hit scans (an argmax rescan, a top-K floor filter) cost ~one probe
/// per 16 floats.
#[inline]
pub fn simd_first_ge_f32(x: &[f32], t: f32) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    let i = neon_skip_below(x, t);
    #[cfg(not(target_arch = "aarch64"))]
    let i = 0usize;
    x[i..].iter().position(|&v| v >= t).map(|j| i + j)
}

/// Start of the first 16-float chunk of `x` holding an element `>= t` (or of
/// the `< 16` tail): every element before it is `< t` or NaN.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_skip_below(x: &[f32], t: f32) -> usize {
    use core::arch::aarch64::{vcgeq_f32, vdupq_n_f32, vld1q_f32, vmaxvq_u32, vorrq_u32};
    let n = x.len();
    let mut i = 0usize;
    // SAFETY: every load reads [i, i + 16) with i + 16 <= n.
    unsafe {
        let tv = vdupq_n_f32(t);
        while i + 16 <= n {
            let p = x.as_ptr().add(i);
            let ge = |o: usize| vcgeq_f32(vld1q_f32(p.add(o)), tv);
            let hit = vorrq_u32(vorrq_u32(ge(0), ge(4)), vorrq_u32(ge(8), ge(12)));
            if vmaxvq_u32(hit) != 0 {
                break;
            }
            i += 16;
        }
    }
    i
}

/// Max of the 256 floats at `p`: 8 independent 4-lane `vmaxq_f32`
/// accumulators (throughput-bound, not a dependent chain), one `vmaxvq_f32`.
/// NaN-propagating (`FMAX`/`FMAXV`). Caller guarantees 256 readable floats.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_block_max_256(p: *const f32) -> f32 {
    use core::arch::aarch64::{float32x4_t, vld1q_f32, vmaxq_f32, vmaxvq_f32};
    unsafe {
        let mut a: [float32x4_t; 8] = [vld1q_f32(p); 8];
        for (j, acc) in a.iter_mut().enumerate() {
            *acc = vld1q_f32(p.add(4 * j));
        }
        let mut k = 32;
        while k < ARGMAX_BLOCK {
            for (j, acc) in a.iter_mut().enumerate() {
                *acc = vmaxq_f32(*acc, vld1q_f32(p.add(k + 4 * j)));
            }
            k += 32;
        }
        let m01 = vmaxq_f32(a[0], a[1]);
        let m23 = vmaxq_f32(a[2], a[3]);
        let m45 = vmaxq_f32(a[4], a[5]);
        let m67 = vmaxq_f32(a[6], a[7]);
        vmaxvq_f32(vmaxq_f32(vmaxq_f32(m01, m23), vmaxq_f32(m45, m67)))
    }
}

/// Width of the blocks [`simd_block_max_256_into`] reduces.
pub const BLOCK_MAX_WIDTH: usize = 256;

/// Width of the sub-blocks [`simd_block_max_256_sub64_into`] also reports.
pub const BLOCK_MAX_SUB_WIDTH: usize = 64;

/// [`neon_block_max_256`] plus the four contiguous 64-float sub-block maxima:
/// accumulator pair `(2q, 2q+1)` owns sub-block `q`, so the eight chains stay
/// independent and the extra cost is three more `vmaxvq_f32`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_block_max_256_sub64(p: *const f32) -> (f32, [f32; 4]) {
    use core::arch::aarch64::{float32x4_t, vld1q_f32, vmaxq_f32, vmaxvq_f32};
    unsafe {
        let mut a: [float32x4_t; 8] = [vld1q_f32(p); 8];
        for (j, acc) in a.iter_mut().enumerate() {
            *acc = vld1q_f32(p.add(64 * (j / 2) + 4 * (j % 2)));
        }
        let mut k = 8;
        while k < BLOCK_MAX_SUB_WIDTH {
            for (j, acc) in a.iter_mut().enumerate() {
                *acc = vmaxq_f32(*acc, vld1q_f32(p.add(64 * (j / 2) + k + 4 * (j % 2))));
            }
            k += 8;
        }
        let q0 = vmaxq_f32(a[0], a[1]);
        let q1 = vmaxq_f32(a[2], a[3]);
        let q2 = vmaxq_f32(a[4], a[5]);
        let q3 = vmaxq_f32(a[6], a[7]);
        let sub = [
            vmaxvq_f32(q0),
            vmaxvq_f32(q1),
            vmaxvq_f32(q2),
            vmaxvq_f32(q3),
        ];
        (
            vmaxvq_f32(vmaxq_f32(vmaxq_f32(q0, q1), vmaxq_f32(q2, q3))),
            sub,
        )
    }
}

/// Per-block maxima over 256-float blocks: `out[b] = max(x[256b .. 256b+256])`
/// (the last block may be short). Writes `x.len().div_ceil(256)` entries,
/// clamped to `out.len()`, and returns that count. NaN-propagating on aarch64
/// (a block holding a NaN reports NaN); other targets use `simd_max_f32` per
/// block, whose NaN behaviour is the target's.
#[inline]
pub fn simd_block_max_256_into(x: &[f32], out: &mut [f32]) -> usize {
    let nblk = x.len().div_ceil(BLOCK_MAX_WIDTH).min(out.len());
    #[cfg(target_arch = "aarch64")]
    {
        let full = (x.len() / BLOCK_MAX_WIDTH).min(nblk);
        for (b, o) in out[..full].iter_mut().enumerate() {
            // SAFETY: b < x.len() / 256, so 256 floats from 256b are in bounds.
            *o = unsafe { neon_block_max_256(x.as_ptr().add(b * BLOCK_MAX_WIDTH)) };
        }
        if full < nblk {
            out[full] = simd_max_f32(&x[full * BLOCK_MAX_WIDTH..]);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (o, block) in out[..nblk].iter_mut().zip(x.chunks(BLOCK_MAX_WIDTH)) {
            *o = simd_max_f32(block);
        }
    }
    nblk
}

/// One full 256-float block: `(max, [the four 64-float sub-block maxima])`.
/// NaN-propagating on aarch64 (the 8-accumulator NEON kernel); elsewhere
/// `simd_max_f32` per sub-block. For consumers that fuse per-block work
/// (ranking, early exits) into the reduction loop.
#[inline(always)]
pub fn simd_block_max_256_sub64(block: &[f32; BLOCK_MAX_WIDTH]) -> (f32, [f32; 4]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `block` is exactly 256 readable floats.
        unsafe { neon_block_max_256_sub64(block.as_ptr()) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sub = [0f32; 4];
        for (o, c) in sub.iter_mut().zip(block.chunks(BLOCK_MAX_SUB_WIDTH)) {
            *o = simd_max_f32(c);
        }
        (simd_max_f32(&sub), sub)
    }
}

/// [`simd_block_max_256_into`] that also writes the 64-float sub-block maxima:
/// `sub[4b + q] = max(x[256b + 64q ..][..64])` (an empty sub-block of a short
/// last block reports `-inf`). Returns the block count; nothing is written
/// unless `sub.len() >= 4 * blocks`, which the caller sizes. A block-skip
/// consumer (Issue 966 `topk_pools_into`) ranks the block maxima to bound a
/// top-K exactly, then rescans only the sub-blocks that clear the bound.
#[inline]
pub fn simd_block_max_256_sub64_into(x: &[f32], out: &mut [f32], sub: &mut [f32]) -> usize {
    const SUBS: usize = BLOCK_MAX_WIDTH / BLOCK_MAX_SUB_WIDTH;
    let nblk = x
        .len()
        .div_ceil(BLOCK_MAX_WIDTH)
        .min(out.len())
        .min(sub.len() / SUBS);
    let full = (x.len() / BLOCK_MAX_WIDTH).min(nblk);
    #[cfg(target_arch = "aarch64")]
    for (b, o) in out[..full].iter_mut().enumerate() {
        // SAFETY: b < x.len() / 256, so 256 floats from 256b are in bounds.
        let (m, s4) = unsafe { neon_block_max_256_sub64(x.as_ptr().add(b * BLOCK_MAX_WIDTH)) };
        *o = m;
        sub[SUBS * b..SUBS * b + SUBS].copy_from_slice(&s4);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (b, o) in out[..full].iter_mut().enumerate() {
        let block = &x[b * BLOCK_MAX_WIDTH..(b + 1) * BLOCK_MAX_WIDTH];
        for (q, c) in block.chunks(BLOCK_MAX_SUB_WIDTH).enumerate() {
            sub[SUBS * b + q] = simd_max_f32(c);
        }
        *o = simd_max_f32(block);
    }
    if full < nblk {
        let block = &x[full * BLOCK_MAX_WIDTH..];
        out[full] = simd_max_f32(block);
        let mut chunks = block.chunks(BLOCK_MAX_SUB_WIDTH);
        for q in 0..SUBS {
            sub[SUBS * full + q] = chunks.next().map_or(f32::NEG_INFINITY, simd_max_f32);
        }
    }
    nblk
}

/// Blocked two-level argmax (Issue 967).
///
/// The lane-wise kernel is one dependent `max → compare → select` chain per
/// 4 floats — latency-bound. This splits the job: pass 1 reduces each
/// 256-float block with 8 INDEPENDENT `vmaxq_f32` accumulators (throughput-
/// bound) and keeps the first block whose maximum is strictly greater than
/// every earlier block's; pass 2 finds the first index `== max` inside that
/// one block. The global first-index-of-max lives in the first block attaining
/// the global max, and strict `>` never moves past it, so the tie rule is
/// identical to the lane-wise kernel's.
///
/// NaN: a block max is NaN iff the block holds a NaN (`FMAX`/`FMAXV`
/// propagate), and the lane-wise kernel's NaN behaviour is order-dependent, so
/// any NaN block hands the WHOLE slice to the lane-wise kernel — bit-identical
/// there by construction. Returned value is the maximum (`==` the element at
/// the index; signed zeros may differ in sign, as they already could between
/// the lane-wise and two-pass paths).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_argmax_blocked_f32(x: &[f32]) -> (usize, f32) {
    let n = x.len();
    if n < 2 * ARGMAX_BLOCK {
        return unsafe { neon_argmax_f32(x) };
    }
    unsafe {
        let nblk = n / ARGMAX_BLOCK;
        let base = x.as_ptr();
        let mut best = neon_block_max_256(base);
        if best.is_nan() {
            return neon_argmax_f32(x);
        }
        let mut best_blk = 0usize;
        for b in 1..nblk {
            let m = neon_block_max_256(base.add(b * ARGMAX_BLOCK));
            if m > best {
                best = m;
                best_blk = b;
            } else if m.is_nan() {
                return neon_argmax_f32(x);
            }
        }
        let tail = nblk * ARGMAX_BLOCK;
        if tail < n {
            let m = simd_max_f32(x.get_unchecked(tail..));
            if m > best {
                best = m;
                best_blk = nblk;
            } else if m.is_nan() {
                return neon_argmax_f32(x);
            }
        }
        let lo = best_blk * ARGMAX_BLOCK;
        let hi = (lo + ARGMAX_BLOCK).min(n);
        // First `>= best` is first `== best`: `best` is the block's max.
        if let Some(j) = simd_first_ge_f32(x.get_unchecked(lo..hi), best) {
            return (lo + j, best);
        }
        // Unreachable: `best` is the FMAX of [lo, hi), a non-NaN element of it.
        neon_argmax_f32(x)
    }
}

/// The x86_64 (and generic non-SIMD) path: `simd_max_f32` reduce +
/// `position(== max)`. Provenance: a scalar index-tracking loop does not
/// auto-vectorize and measured SLOWER than this two-pass idiom at large
/// vocab, so the two-pass is the incumbent — and it survived a second
/// challenger on measured evidence. Issue 817 ported the NEON single-pass
/// (8 lanes tracking (max, index), blend-on-strict-gt) and the dispatch A/B
/// measured it a net LOSS on this box: the two-pass wins `iid` 2.8–3.5× and
/// `early` 3–4.7× at n ≥ 256 (ILP'd `max_ps` reduce + a vectorized
/// early-exiting `position` beat a latency-bound cmp/blend chain), while the
/// kernel only wins when the maximum sits in the last few percent of the
/// scan (1.5–1.9×) and at n=64 (fixed-overhead, ~20 ns). Both profile arms,
/// tight round bands:
/// [Bench 811](../../.benchmarks/812_argmax_avx2_single_pass_goat.md).
/// Demote-on-loss applied; the dispatch was reverted the same day. Reopen
/// trigger: re-run `tests/bench_817_argmax_dispatch_ab.rs` after a major
/// toolchain/CPU change — if the two-pass absolute columns regress or the
/// position-scan advantage disappears, re-try the port.
///
/// Cost note (measured, Bench 810): this path's cost depends on WHERE the
/// maximum sits — the second pass rescans until it finds `max`, so a late
/// maximum pays two full scans (`late_peak` k=1 lost ~0.9× in the argtopk
/// k=1 rows). That loss is ACCEPTED: fixing it costs more everywhere else
/// (Bench 811).
///
/// Gated to exactly the platforms the dispatch arm above serves (x86_64 and
/// the generic non-SIMD triples) — on aarch64 / wasm32+simd128 the NEON and
/// wasm kernels own the dispatch, and an ungated copy here is dead code there.
#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline]
fn two_pass_argmax_f32(x: &[f32]) -> (usize, f32) {
    let max_val = simd_max_f32(x);
    let idx = x.iter().position(|&v| v == max_val).unwrap_or(0);
    (idx, max_val)
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

/// WASM SIMD128 single-pass argmax — 4-wide, tracks (max value, index) per lane.
///
/// Issue 007: ports the NEON kernel structure to `core::arch::wasm32`. Uses
/// `f32x4_max` for lane-wise max and `v128_bitselect` to pick the candidate
/// index only where it strictly exceeds the running lane max (mirrors NEON's
/// `vbslq_u32` semantics: `bitselect(cand_idx, best_idx, gt_mask)` =
/// `mask ? cand_idx : best_idx`). Tie-break is first-index-wins via strict `>`
/// (`f32x4_gt`), matching `position(|&v| v == max)` on the scalar path.
///
/// Compile-time gated by `target_feature = "simd128"`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn wasm32_simd128_argmax_f32(x: &[f32]) -> (usize, f32) {
    use core::arch::wasm32::{
        f32x4_extract_lane, f32x4_gt, f32x4_max, i32x4_add, i32x4_extract_lane, i32x4_splat,
        v128_bitselect, v128_load,
    };
    unsafe {
        let n = x.len();
        // Below one full vector, scalar is cheaper than SIMD128 setup.
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

        // Seed: vmax = x[0..4], vidx = [0,1,2,3], cur = [4,5,6,7] (next chunk's indices).
        let base: [i32; 4] = [0, 1, 2, 3];
        let four = i32x4_splat(4);
        let mut vmax = v128_load(x.as_ptr().cast());
        let mut vidx = v128_load(base.as_ptr().cast());
        let mut cur = i32x4_add(vidx, four);
        let chunks = n / 4;
        let mut i = 4;
        for _ in 1..chunks {
            let v = v128_load(x.as_ptr().add(i).cast());
            let mask = f32x4_gt(v, vmax); // lanes where v strictly greater
            vmax = f32x4_max(vmax, v);
            // Adopt new index only where strictly greater; else keep old.
            vidx = v128_bitselect(cur, vidx, mask);
            cur = i32x4_add(cur, four);
            i += 4;
        }

        // Reduce the 4 lanes: highest value, smallest index on ties.
        let v0 = f32x4_extract_lane::<0>(vmax);
        let v1 = f32x4_extract_lane::<1>(vmax);
        let v2 = f32x4_extract_lane::<2>(vmax);
        let v3 = f32x4_extract_lane::<3>(vmax);
        let i0 = i32x4_extract_lane::<0>(vidx) as usize;
        let i1 = i32x4_extract_lane::<1>(vidx) as usize;
        let i2 = i32x4_extract_lane::<2>(vidx) as usize;
        let i3 = i32x4_extract_lane::<3>(vidx) as usize;
        let mut bv = v0;
        let mut bi = i0;
        for (v, idx) in [(v1, i1), (v2, i2), (v3, i3)] {
            if v > bv || (v == bv && idx < bi) {
                bv = v;
                bi = idx;
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
