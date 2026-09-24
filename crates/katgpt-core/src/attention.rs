//! Tiled online-softmax flash attention for CPU SIMD.
//!
//! Processes Q in SIMD-width row tiles, K/V in column tiles.
//! Avoids materializing full N×N score matrix.
//! Falls back to full materialization for small N.
//!
//! Reference: ThunderKittens (Research 077) online-softmax algorithm
//! adapted for CPU NEON/AVX2 SIMD.

#[cfg(feature = "tiled_attention")]
use rayon::prelude::*;
#[cfg(feature = "tiled_attention")]
use std::cell::UnsafeCell;

/// Threshold: use tiled attention when N > 128 (score matrix > L1 cache).
/// L1 ≈ 32 KB. Score = N × N × 4B. sqrt(32K / 4) ≈ 90, round up to 128.
const TILED_ATTENTION_THRESHOLD: usize = 128;

/// Row tile size: SIMD-width query rows.
/// NEON = 4 f32/register, AVX2 = 8 f32/register. Use 8 (NEON processes 2 sub-tiles).
const BR: usize = 8;

/// Column tile size: tuned for L1 cache.
/// K tile = BC × head_dim × 4B. For head_dim=64, BC=128: 32 KB (fits L1).
const BC: usize = 128;

/// Tiled online-softmax flash attention for CPU SIMD.
///
/// Processes Q in SIMD-width row tiles, K/V in column tiles.
/// Avoids materializing full N×N score matrix.
/// Falls back to full materialization for small N.
///
/// # Arguments
/// * `q` - Query tensor [seq_len × head_dim], row-major
/// * `k` - Key tensor [seq_len × head_dim], row-major
/// * `v` - Value tensor [seq_len × head_dim], row-major
/// * `output` - Output tensor [seq_len × head_dim], row-major (pre-allocated)
/// * `seq_len` - Sequence length N
/// * `head_dim` - Dimension per attention head D
/// * `scale` - Softmax temperature (typically 1/√head_dim)
///
/// # Panics
/// Debug-asserts that slice lengths match expected dimensions.
#[cfg(feature = "tiled_attention")]
pub fn tiled_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) {
    tiled_attention_forward_impl(q, k, v, output, seq_len, head_dim, scale, None, None);
}

/// SSMax-augmented tiled attention forward (Plan 411 T2.4).
///
/// Identical to [`tiled_attention_forward`] but applies the SSMax
/// length-aware log-N attention temperature by folding `s_L · log(N)` into
/// the softmax scale. This is mathematically equivalent to rescaling each
/// pre-softmax logit by `s_L · log(N)`:
///
/// `softmax(scale · q·k) = softmax((scale · s_L · log N) · q·k)`
///
/// because softmax is scale-equivariant in its input. For the online-softmax
/// (flash-attention) kernel that doesn't materialize the full score matrix,
/// this fold is the zero-overhead way to apply SSMax — one multiply on the
/// scale parameter, no extra pass over the scores.
///
/// # Arguments
/// Same as [`tiled_attention_forward`], plus:
/// * `ssmax` - SSMax mode (the per-layer source of `s_L`).
///
/// # When `seq_len ≤ 1`
///
/// `log(N) = 0`, so the scale is unchanged — SSMax is a no-op. This preserves
/// the small-N no-regression guarantee (G5).
#[cfg(all(feature = "tiled_attention", feature = "ssmax_temperature"))]
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_forward_ssmax(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    ssmax: &crate::ssmax::SsmaxMode,
) {
    let log_n = if seq_len > 1 {
        (seq_len as f32).ln()
    } else {
        0.0
    };
    let ssmax_scale = scale * ssmax.multiplier(log_n);
    tiled_attention_forward_impl(q, k, v, output, seq_len, head_dim, ssmax_scale, None, None);
}

/// Implementation that accepts an optional pre-allocated scores scratch buffer.
///
/// When `scores_buf` is `Some`, it is used as scratch space for the fallback
/// path (seq_len < TILED_ATTENTION_THRESHOLD), avoiding a per-call heap allocation.
/// The buffer must be at least `seq_len * seq_len` elements.
/// When `None`, the buffer is allocated on demand.
#[cfg(feature = "tiled_attention")]
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_forward_with_scores(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    scores_buf: Option<&mut [f32]>,
) {
    tiled_attention_forward_impl(q, k, v, output, seq_len, head_dim, scale, scores_buf, None);
}

/// Inner implementation: accepts optional pre-allocated `scores_buf` and `o_tile`
/// scratch buffers to avoid per-call heap allocation. `tiled_attention_forward`
/// and `tiled_attention_forward_with_scores` both delegate here.
#[cfg(feature = "tiled_attention")]
#[allow(clippy::too_many_arguments)]
fn tiled_attention_forward_impl(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    scores_buf: Option<&mut [f32]>,
    o_tile: Option<&mut [f32]>,
) {
    let expected = seq_len * head_dim;
    debug_assert_eq!(q.len(), expected, "Q slice length mismatch");
    debug_assert_eq!(k.len(), expected, "K slice length mismatch");
    debug_assert_eq!(v.len(), expected, "V slice length mismatch");
    debug_assert_eq!(output.len(), expected, "output slice length mismatch");

    match seq_len {
        0 => return,
        n if n < TILED_ATTENTION_THRESHOLD => {
            let needed = seq_len * seq_len;
            let buf = scores_buf;
            attention_fallback(q, k, v, output, seq_len, head_dim, scale, buf, needed);
            return;
        }
        _ => {}
    }

    // Allocate o_tile only if caller didn't provide one.
    // Buffer must be at least BR * head_dim elements.
    let tile_elems = BR * head_dim;
    let mut local_o_tile;
    let o_tile: &mut [f32] = if let Some(buf) = o_tile {
        debug_assert!(buf.len() >= tile_elems, "o_tile buffer too small");
        buf
    } else {
        local_o_tile = vec![0.0f32; tile_elems];
        &mut local_o_tile
    };

    tiled_attention_inner(
        q,
        k,
        v,
        output,
        seq_len,
        head_dim,
        scale,
        o_tile,
        &mut NoStats,
    );
}

/// Per-row statistics hooks for [`tiled_attention_inner`].
///
/// Monomorphized per sink: [`NoStats`] has `ACTIVE = false` and empty
/// `#[inline(always)]` bodies, so the plain forward carries no extra work —
/// not a branch, not a copy (the `y_row` copy is behind `S::ACTIVE`, a
/// compile-time constant).
#[cfg(feature = "tiled_attention")]
trait TileStats {
    /// Whether the kernel must keep a pre-exp copy of each score row.
    const ACTIVE: bool;
    /// A new query tile begins.
    fn reset_tile(&mut self);
    /// Row `i`'s running max moved: `delta = scale·(m_old − m_new)` in logit
    /// units, `correction = e^delta`, `l_old` the normalizer before rescale.
    fn rebase(&mut self, i: usize, delta: f32, correction: f32, l_old: f32);
    /// Row `i` folded one K tile: `y` = shifted logits, `p` = `e^y`.
    fn fold(&mut self, i: usize, y: &[f32], p: &[f32]);
    /// Row `i` (global query `row`) is final with normalizer `l` over `n` keys.
    fn finish(&mut self, row: usize, i: usize, l: f32, n: usize);
}

/// The inert sink — the plain forward.
#[cfg(feature = "tiled_attention")]
struct NoStats;

#[cfg(feature = "tiled_attention")]
impl TileStats for NoStats {
    const ACTIVE: bool = false;
    #[inline(always)]
    fn reset_tile(&mut self) {}
    #[inline(always)]
    fn rebase(&mut self, _: usize, _: f32, _: f32, _: f32) {}
    #[inline(always)]
    fn fold(&mut self, _: usize, _: &[f32], _: &[f32]) {}
    #[inline(always)]
    fn finish(&mut self, _: usize, _: usize, _: f32, _: usize) {}
}

/// The attention-SNR sink (Issue 882 P1): the two extra online-softmax
/// registers `T = Σe^y·y` and `R₂ = Σe^{2y}` per tile row, finalized into
/// exact entropy + participation ratio. See [`crate::attention_snr`].
#[cfg(all(feature = "tiled_attention", feature = "attention_snr"))]
struct SnrSink<'a> {
    t: [f32; BR],
    r2: [f32; BR],
    out: &'a mut [crate::attention_snr::SnrRowStats],
}

#[cfg(all(feature = "tiled_attention", feature = "attention_snr"))]
impl TileStats for SnrSink<'_> {
    const ACTIVE: bool = true;
    #[inline(always)]
    fn reset_tile(&mut self) {
        self.t = [0.0; BR];
        self.r2 = [0.0; BR];
    }
    #[inline(always)]
    fn rebase(&mut self, i: usize, delta: f32, correction: f32, l_old: f32) {
        // First K tile: l_old = 0 and delta = −∞; `−∞·0` would be NaN.
        if l_old > 0.0 {
            self.t[i] = correction * (self.t[i] + delta * l_old);
            self.r2[i] *= correction * correction;
        }
    }
    #[inline(always)]
    fn fold(&mut self, i: usize, y: &[f32], p: &[f32]) {
        self.t[i] += crate::simd::simd_dot_f32(p, y, p.len());
        self.r2[i] += crate::simd::simd_dot_f32(p, p, p.len());
    }
    #[inline(always)]
    fn finish(&mut self, row: usize, i: usize, l: f32, n: usize) {
        let (t, r2) = (self.t[i], self.r2[i]);
        self.out[row] = crate::attention_snr::SnrRowStats {
            entropy: if l > 0.0 {
                (l.ln() - t / l).max(0.0)
            } else {
                0.0
            },
            participation_ratio: if r2 > 0.0 { l * l / r2 } else { 0.0 },
            n: n as u32,
        };
    }
}

/// Scratch length `tiled_attention_forward_snr` needs for its `o_tile`.
#[cfg(all(feature = "tiled_attention", feature = "attention_snr"))]
#[inline]
pub const fn tiled_snr_scratch_len(head_dim: usize) -> usize {
    BR * head_dim
}

/// Tiled flash attention that also writes exact per-query-row softmax
/// entropy + participation ratio (Issue 882 P1, the streaming attention-SNR
/// accumulators).
///
/// Output is **bit-identical** to [`tiled_attention_forward`] for
/// `seq_len ≥ 128` (the same kernel; the sink never touches the output
/// accumulators). Below that threshold the plain forward takes its
/// materialized fallback while this always runs the tiled kernel — equal to
/// f32 rounding, not bitwise.
///
/// Allocation-free: `o_tile` is caller scratch of at least
/// [`tiled_snr_scratch_len`]`(head_dim)` elements; `stats` has `seq_len` rows.
#[cfg(all(feature = "tiled_attention", feature = "attention_snr"))]
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_forward_snr(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    o_tile: &mut [f32],
    stats: &mut [crate::attention_snr::SnrRowStats],
) {
    let expected = seq_len * head_dim;
    debug_assert_eq!(q.len(), expected, "Q slice length mismatch");
    debug_assert_eq!(k.len(), expected, "K slice length mismatch");
    debug_assert_eq!(v.len(), expected, "V slice length mismatch");
    debug_assert_eq!(output.len(), expected, "output slice length mismatch");
    assert!(stats.len() >= seq_len, "stats needs one row per query");
    assert!(o_tile.len() >= BR * head_dim, "o_tile scratch too small");
    if seq_len == 0 {
        return;
    }
    let mut sink = SnrSink {
        t: [0.0; BR],
        r2: [0.0; BR],
        out: stats,
    };
    tiled_attention_inner(q, k, v, output, seq_len, head_dim, scale, o_tile, &mut sink);
}

/// Inner tiled attention implementation with online-softmax.
///
/// Algorithm (per query tile):
/// 1. Initialize: o_tile = 0, max_tile = -inf, norm_tile = 0
/// 2. For each K/V tile:
///    a. Score tile: S = q_tile @ k_tile.T
///    b. Update running max: max_new = max(max_old, rowmax(S))
///    c. Correction: exp2((max_old - max_new) * log2e_scale)
///    d. Exp with correction: P̃ = exp2((S - max_new) * log2e_scale)
///    e. Update: norm = correction * norm + rowsum(P̃)
///    f. Update: o_tile = correction * o_tile + P̃ @ v_tile
/// 3. Final normalize: o_tile / norm_tile
#[cfg(feature = "tiled_attention")]
#[allow(clippy::too_many_arguments)]
fn tiled_attention_inner<S: TileStats>(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    // Scratch buffer for output tile accumulation. Must be at least `BR * head_dim` elements.
    // Zeroed at the start of each query tile.
    o_tile: &mut [f32],
    // Per-row statistics sink. `NoStats` compiles every hook to nothing, so the
    // plain forward is the same machine code it was before the sink existed.
    stats: &mut S,
) {
    let log2e_scale = scale * std::f32::consts::LOG2_E;
    let q_tiles = seq_len.div_ceil(BR);
    let k_tiles = seq_len.div_ceil(BC);

    let tile_elems = BR * head_dim;

    // s_tile is hoisted out of the query-tile loop. It is BR*BC = 1024 f32
    // (4 KiB), so re-declaring it per query tile meant a 4 KiB memset per tile
    // — 512 KiB of pure stores per head at seq_len=1024. That init is dead: the
    // only two reads of s_tile are `s_tile[i*BC .. i*BC + actual_bc]` at the row
    // max and at the P̃ computation, both for `i in 0..actual_br`, and the score
    // pass just above writes exactly that same `[0..actual_br) × [0..actual_bc)`
    // rectangle on every k-tile iteration. So no slot is ever read before being
    // written in the same pass, and stale values from a previous query tile can
    // never be observed. The one-time -inf fill is retained only as a
    // debugging-friendly poison value.
    let mut s_tile = [f32::NEG_INFINITY; BR * BC];
    // Pre-exp copy of one score row, read only by an active stats sink (the
    // entropy numerator needs `y·e^y`, and `exp` overwrites `y` in place).
    let mut y_row = [0.0f32; BC];

    for q_tile_idx in 0..q_tiles {
        let q_start = q_tile_idx * BR;
        let q_end = (q_start + BR).min(seq_len);
        let actual_br = q_end - q_start;

        // Reuse pre-allocated tile buffer
        o_tile[..tile_elems].fill(0.0);
        let mut max_tile = [f32::NEG_INFINITY; BR];
        let mut norm_tile = [0.0f32; BR];
        stats.reset_tile();

        for k_tile_idx in 0..k_tiles {
            let k_start = k_tile_idx * BC;
            let k_end = (k_start + BC).min(seq_len);
            let actual_bc = k_end - k_start;

            // 1. Score tile: S = q_tile @ k_tile.T (BR × BC)
            // `q_row` is invariant in `j` but was being re-sliced (range check +
            // panic path) on every one of `actual_bc` iterations; the score row
            // is pre-sliced too, which drops the `i * BC + j` recompute and its
            // bounds check. `s_row.len() == actual_bc`, so `enumerate()` walks
            // exactly the same `j` range as before.
            for i in 0..actual_br {
                let q_off = (q_start + i) * head_dim;
                let q_row = &q[q_off..q_off + head_dim];
                let s_row = &mut s_tile[i * BC..i * BC + actual_bc];
                for (j, s) in s_row.iter_mut().enumerate() {
                    let k_off = (k_start + j) * head_dim;
                    *s = crate::simd::simd_dot_f32(q_row, &k[k_off..k_off + head_dim], head_dim);
                }
                // j >= actual_bc: never read (see the s_tile note above)
            }
            // i >= actual_br: stays -inf (boundary query rows)

            // 2+3. Row max + correction + P̃ + accumulate (fused per row)
            for i in 0..actual_br {
                let rm = crate::simd::simd_max_f32(&s_tile[i * BC..i * BC + actual_bc]);
                let m_old = max_tile[i];
                let m_new = m_old.max(rm);
                max_tile[i] = m_new;

                // Fast path: when m_new == m_old (typical after the first K-tile,
                // since softmax max saturates quickly), correction is 1.0 and the
                // `simd_scale_inplace` + `norm_tile *=` work is a no-op. Skip it.
                if m_new > m_old {
                    // Correction factor: exp2((m_old - m_new) * log2e_scale)
                    let correction = ((m_old - m_new) * log2e_scale).exp2();

                    // Apply correction to existing accumulators FIRST (SIMD-accelerated)
                    crate::simd::simd_scale_inplace(
                        &mut o_tile[i * head_dim..i * head_dim + head_dim],
                        correction,
                    );
                    stats.rebase(i, (m_old - m_new) * scale, correction, norm_tile[i]);
                    norm_tile[i] *= correction;
                }

                // Compute P̃ in-place on s_tile row: exp((s - m_new) * scale)
                // Mathematically equivalent to exp2((s - m_new) * log2e_scale)
                // since exp(x) = exp2(x * LOG2_E).
                let p_row = &mut s_tile[i * BC..i * BC + actual_bc];
                crate::simd::simd_fused_sub_scale_inplace(p_row, m_new, scale);
                if S::ACTIVE {
                    y_row[..actual_bc].copy_from_slice(p_row);
                }
                crate::simd::simd_exp_inplace(p_row);
                stats.fold(i, &y_row[..actual_bc], p_row);

                // Rowsum via SIMD (single reduction vs scalar accumulator)
                let rowsum = crate::simd::simd_sum_f32(p_row);

                // Accumulate P̃[i][j] × V[j] into o_tile[i] (SIMD-accelerated)
                // `o_tile`'s row slice is invariant in `j` yet was re-borrowed
                // (range check) on every iteration of the kernel's innermost
                // accumulate. Hoist it. `p_row.len() == actual_bc` already, so
                // the old `.take(actual_bc)` was a no-op and the `j` sequence —
                // hence the accumulation order into `o_row` — is unchanged.
                let o_row = &mut o_tile[i * head_dim..i * head_dim + head_dim];
                for (j, &p) in p_row.iter().enumerate() {
                    let v_off = (k_start + j) * head_dim;
                    crate::simd::simd_fused_scale_acc(
                        o_row,
                        &v[v_off..v_off + head_dim],
                        p,
                        head_dim,
                    );
                }

                norm_tile[i] += rowsum;
            }
        }

        // 4. Final normalize: o_tile / norm_tile (fused copy+scale in single SIMD pass)
        for (i, norm_tile_i) in norm_tile.iter().enumerate().take(actual_br) {
            let inv_norm = 1.0 / *norm_tile_i;
            let o_off = i * head_dim;
            let out_off = (q_start + i) * head_dim;
            crate::simd::simd_fused_decay_write(
                &mut output[out_off..out_off + head_dim],
                0.0,
                &o_tile[o_off..o_off + head_dim],
                inv_norm,
            );
            stats.finish(q_start + i, i, *norm_tile_i, seq_len);
        }
    }
}

/// Fallback attention using full score matrix materialization.
/// Uses existing `softmax_scaled` for numerically stable softmax.
/// Called when seq_len < TILED_ATTENTION_THRESHOLD.
#[cfg(feature = "tiled_attention")]
#[allow(clippy::too_many_arguments)]
fn attention_fallback(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    scores_buf: Option<&mut [f32]>,
    needed: usize,
) {
    if seq_len == 0 {
        return;
    }

    // 1. Compute scores = Q @ K.T (seq_len × seq_len)
    let mut scores_local;
    let scores: &mut [f32] = match scores_buf {
        // No pre-zeroing: `needed == seq_len * seq_len` (the caller's only
        // definition, line 146) and the score pass below *assigns* every
        // `scores[i * seq_len + j]` for `i, j in 0..seq_len`, i.e. exactly
        // `[0, needed)`. The softmax pass and the `scores @ V` pass then read
        // only that same range. So every slot read was written first in this
        // call — the O(seq_len²) memset was dead, on what is the common
        // short-sequence decode path.
        Some(buf) if buf.len() >= needed => buf,
        _ => {
            scores_local = vec![0.0f32; needed];
            &mut scores_local
        }
    };
    for i in 0..seq_len {
        let q_off = i * head_dim;
        // `q_row` is invariant in `j`; pre-slicing it and the score row hoists
        // their range checks out of the inner loop. `s_row.len() == seq_len`, so
        // `enumerate()` covers the identical `j` range.
        let q_row = &q[q_off..q_off + head_dim];
        let s_row = &mut scores[i * seq_len..(i + 1) * seq_len];
        for (j, s) in s_row.iter_mut().enumerate() {
            let k_off = j * head_dim;
            *s = crate::simd::simd_dot_f32(q_row, &k[k_off..k_off + head_dim], head_dim);
        }
    }

    // 2. Apply scaled softmax row by row
    for i in 0..seq_len {
        let row = &mut scores[i * seq_len..(i + 1) * seq_len];
        crate::types::softmax_scaled(row, scale);
    }

    // 3. Compute output = scores @ V (seq_len × head_dim)
    //    Loop order (i, j, d) for contiguous V row access and cache-friendly output accumulation
    for i in 0..seq_len {
        let out_off = i * head_dim;
        // Both rows are invariant in `j`: the output row was re-borrowed (range
        // check) and `scores[scores_off + j]` re-indexed on every iteration of
        // this innermost accumulate. `s_row.len() == seq_len`, so the `j`
        // sequence — and therefore the accumulation order into `out_row` — is
        // unchanged.
        let s_row = &scores[i * seq_len..(i + 1) * seq_len];
        let out_row = &mut output[out_off..out_off + head_dim];
        out_row.fill(0.0);
        for (j, &s) in s_row.iter().enumerate() {
            let v_off = j * head_dim;
            crate::simd::simd_fused_scale_acc(out_row, &v[v_off..v_off + head_dim], s, head_dim);
        }
    }
}

/// Tiled attention for multi-head batched input.
///
/// Calls `tiled_attention_forward` per (batch, head) pair with rayon parallelism.
/// Q, K, V layout: [batch × heads × seq_len × head_dim], row-major.
///
/// # Arguments
/// * `q` - Query tensor [batch × heads × seq_len × head_dim]
/// * `k` - Key tensor [batch × heads × seq_len × head_dim]
/// * `v` - Value tensor [batch × heads × seq_len × head_dim]
/// * `output` - Output tensor [batch × heads × seq_len × head_dim] (pre-allocated)
/// * `batch` - Batch size B
/// * `heads` - Number of attention heads H
/// * `seq_len` - Sequence length N
/// * `head_dim` - Dimension per attention head D
#[cfg(feature = "tiled_attention")]
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_batched(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let head_size = seq_len * head_dim;
    let total = batch * heads;

    if total == 0 {
        return;
    }

    let scores_buf_size = seq_len * seq_len;
    let o_tile_size = BR * head_dim;

    // Reuse grow-only scratch buffers per OS thread via thread_local.
    // Both sequential and parallel paths share these — the sequential path
    // also benefits from cross-call persistence (e.g. an L-layer transformer
    // calling this L times converts 2L allocations to ≤2 grow-only).
    // Perf: UnsafeCell avoids RefCell's runtime borrow-check overhead.
    // Safety: thread_local guarantees exclusive per-thread access — both
    // for the single-threaded sequential path (trivially exclusive) and for
    // Rayon's work-stealing parallel path (each worker thread gets its own slot).
    thread_local! {
        static SCORES_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
        static O_TILE_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
    }

    let run_one = |idx: usize, out_chunk: &mut [f32]| {
        let offset = idx * head_size;
        SCORES_BUF.with(|scores| {
            O_TILE_BUF.with(|o_tile| {
                // Safety: thread_local guarantees exclusive per-thread access.
                let scores = unsafe { &mut *scores.get() };
                let o_tile = unsafe { &mut *o_tile.get() };
                // `resize` zero-fills the new tail; the existing prefix is
                // preserved but `tiled_attention_forward_impl` zeros the
                // working range itself, so no extra fill is needed on the
                // grow path either.
                if scores.len() < scores_buf_size {
                    scores.resize(scores_buf_size, 0.0);
                }
                if o_tile.len() < o_tile_size {
                    o_tile.resize(o_tile_size, 0.0);
                }
                tiled_attention_forward_impl(
                    &q[offset..offset + head_size],
                    &k[offset..offset + head_size],
                    &v[offset..offset + head_size],
                    out_chunk,
                    seq_len,
                    head_dim,
                    scale,
                    Some(&mut scores[..scores_buf_size]),
                    Some(&mut o_tile[..o_tile_size]),
                );
            });
        });
    };

    if total <= 2 || seq_len * head_dim < 1024 {
        // Sequential fallback for tiny workloads — avoids Rayon scheduling overhead.
        // Buffers come from the shared thread_local above (grow-only across calls).
        for idx in 0..total {
            run_one(idx, &mut output[idx * head_size..(idx + 1) * head_size]);
        }
    } else {
        // Parallel for larger workloads — Rayon overhead amortized.
        output
            .par_chunks_mut(head_size)
            .enumerate()
            .for_each(|(idx, out_chunk)| run_one(idx, out_chunk));
    }
}

// ── Unit Tests ────────────────────────────────────────────────

#[cfg(all(test, feature = "tiled_attention"))]
mod tests {
    use super::*;

    /// Empty sequence → no-op, no crash.
    #[test]
    fn test_empty_sequence() {
        let q: [f32; 0] = [];
        let k: [f32; 0] = [];
        let v: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        tiled_attention_forward(&q, &k, &v, &mut output, 0, 64, 0.125);
    }

    /// Single token: softmax(1 elem) = 1.0, output = V.
    #[test]
    fn test_single_token() {
        let head_dim = 8;
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let k = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v = [0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let mut output = [0.0f32; 8];
        let scale = 1.0 / (head_dim as f32).sqrt();

        tiled_attention_forward(&q, &k, &v, &mut output, 1, head_dim, scale);

        // Single token: attention score = 1.0, softmax = 1.0, output = V
        for (d, &out_d) in output[..head_dim].iter().enumerate() {
            let diff = (out_d - 0.5).abs();
            assert!(diff < 1e-5, "output[{d}] = {out_d}, expected 0.5");
        }
    }

    /// Two identical tokens: output should be average of V rows (uniform attention).
    #[test]
    fn test_two_identical_tokens() {
        let head_dim = 4;
        let seq_len = 2;
        // Q = K = identity-like, so scores are all equal → uniform softmax
        let q = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let k = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let v = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut output = [0.0f32; 8];
        let scale = 1.0 / (head_dim as f32).sqrt();

        tiled_attention_forward(&q, &k, &v, &mut output, seq_len, head_dim, scale);

        // Both rows of Q are [1,0,0,0], both rows of K are [1,0,0,0]
        // Score matrix: [[1,1],[1,1]] → softmax → [[0.5,0.5],[0.5,0.5]]
        // Output row 0: 0.5*[1,0,0,0] + 0.5*[0,1,0,0] = [0.5, 0.5, 0, 0]
        // Output row 1: same
        let expected = [0.5f32, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        for i in 0..seq_len * head_dim {
            let out_i = output[i];
            let exp_i = expected[i];
            let diff = (out_i - exp_i).abs();
            assert!(diff < 1e-5, "output[{i}] = {out_i}, expected {exp_i}");
        }
    }

    /// Batched attention with 2 batches × 2 heads.
    #[test]
    fn test_batched_basic() {
        let batch = 2;
        let heads = 2;
        let seq_len = 4;
        let head_dim = 4;
        let total = batch * heads * seq_len * head_dim;

        let mut q = vec![0.0f32; total];
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        let mut output = vec![0.0f32; total];

        // Fill with simple pattern: each (batch, head) has independent data
        let mut rng = fastrand::Rng::with_seed(42);
        for idx in 0..total {
            q[idx] = rng.f32();
            k[idx] = rng.f32();
            v[idx] = rng.f32();
        }

        tiled_attention_batched(&q, &k, &v, &mut output, batch, heads, seq_len, head_dim);

        // Verify no NaN/Inf in output
        for (i, &val) in output.iter().enumerate() {
            assert!(val.is_finite(), "output[{i}] = {val}, expected finite");
        }
    }
}

// ── SSMax SDPA wrapper tests (Plan 411 T2.4) ──────────────────────

#[cfg(all(test, feature = "tiled_attention", feature = "ssmax_temperature"))]
mod ssmax_tests {
    use super::*;
    use crate::ssmax::SsmaxMode;

    /// `tiled_attention_forward_ssmax` with `SsmaxMode::Fixed { s_l: 1.0 }` must
    /// produce the same output as calling `tiled_attention_forward` with
    /// `scale * log(N)`. This verifies the scale-folding equivalence —
    /// the wrapper adds zero overhead beyond the single `ln(N)` computation.
    #[test]
    fn ssmax_wrapper_matches_scale_folding() {
        let seq_len = 16;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32) * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();

        let mode = SsmaxMode::Fixed { s_l: 1.0 };
        let log_n = (seq_len as f32).ln();
        let folded_scale = scale * mode.multiplier(log_n);

        let mut out_wrapper = vec![0.0f32; seq_len * head_dim];
        let mut out_folded = vec![0.0f32; seq_len * head_dim];
        tiled_attention_forward_ssmax(
            &q,
            &k,
            &v,
            &mut out_wrapper,
            seq_len,
            head_dim,
            scale,
            &mode,
        );
        tiled_attention_forward(&q, &k, &v, &mut out_folded, seq_len, head_dim, folded_scale);

        for i in 0..(seq_len * head_dim) {
            assert_eq!(
                out_wrapper[i], out_folded[i],
                "SSMax wrapper must match scale-folded at [{i}]"
            );
        }
    }

    /// SSMax at n=1 is a no-op: log(1)=0, multiplier=0. But the wrapper guards
    /// `seq_len <= 1` by setting `log_n = 0`, giving `mult = 0` and `scale * 0 = 0`.
    /// At n=1, softmax of a single zero score is [1.0], so output = V regardless.
    /// Verify the wrapper doesn't panic and produces V.
    #[test]
    fn ssmax_wrapper_n1_is_v() {
        let head_dim = 4;
        let q = [1.0f32, 0.0, 0.0, 0.0];
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [0.5f32, 0.5, 0.5, 0.5];
        let mut output = [0.0f32; 4];
        let mode = SsmaxMode::Fixed { s_l: 1.0 };
        tiled_attention_forward_ssmax(&q, &k, &v, &mut output, 1, head_dim, 0.25, &mode);
        // Single-token attention: output = V regardless of scale (softmax of 1 elem = 1.0).
        for (i, &out_i) in output.iter().enumerate() {
            assert!(
                (out_i - 0.5).abs() < 1e-5,
                "n=1 output[{i}] = {out_i}, expected 0.5"
            );
        }
    }
}
