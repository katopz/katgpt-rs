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
    let o_tile: &mut [f32] = match o_tile {
        Some(buf) => {
            debug_assert!(buf.len() >= tile_elems, "o_tile buffer too small");
            buf
        }
        None => {
            local_o_tile = vec![0.0f32; tile_elems];
            &mut local_o_tile
        }
    };

    tiled_attention_inner(q, k, v, output, seq_len, head_dim, scale, o_tile);
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
fn tiled_attention_inner(
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
) {
    let log2e_scale = scale * std::f32::consts::LOG2_E;
    let q_tiles = seq_len.div_ceil(BR);
    let k_tiles = seq_len.div_ceil(BC);

    let tile_elems = BR * head_dim;

    for q_tile_idx in 0..q_tiles {
        let q_start = q_tile_idx * BR;
        let q_end = (q_start + BR).min(seq_len);
        let actual_br = q_end - q_start;

        // Reuse pre-allocated tile buffer
        o_tile[..tile_elems].fill(0.0);
        let mut max_tile = [f32::NEG_INFINITY; BR];
        let mut norm_tile = [0.0f32; BR];

        // s_tile initialized to -inf so padding columns (j >= actual_bc) are masked.
        // The used region [0..actual_br, 0..actual_bc] is fully overwritten each k_tile
        // by the score computation below — no need to re-clear.
        let mut s_tile = [f32::NEG_INFINITY; BR * BC];
        for k_tile_idx in 0..k_tiles {
            let k_start = k_tile_idx * BC;
            let k_end = (k_start + BC).min(seq_len);
            let actual_bc = k_end - k_start;

            // 1. Score tile: S = q_tile @ k_tile.T (BR × BC)
            for i in 0..actual_br {
                let q_off = (q_start + i) * head_dim;
                for j in 0..actual_bc {
                    let k_off = (k_start + j) * head_dim;
                    s_tile[i * BC + j] = crate::simd::simd_dot_f32(
                        &q[q_off..q_off + head_dim],
                        &k[k_off..k_off + head_dim],
                        head_dim,
                    );
                }
                // j >= actual_bc: stays -inf (masked from softmax)
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
                    norm_tile[i] *= correction;
                }

                // Compute P̃ in-place on s_tile row: exp((s - m_new) * scale)
                // Mathematically equivalent to exp2((s - m_new) * log2e_scale)
                // since exp(x) = exp2(x * LOG2_E).
                let p_row = &mut s_tile[i * BC..i * BC + actual_bc];
                crate::simd::simd_fused_sub_scale_inplace(p_row, m_new, scale);
                crate::simd::simd_exp_inplace(p_row);

                // Rowsum via SIMD (single reduction vs scalar accumulator)
                let rowsum = crate::simd::simd_sum_f32(p_row);

                // Accumulate P̃[i][j] × V[j] into o_tile[i] (SIMD-accelerated)
                for (j, p_row_j) in p_row.iter().enumerate().take(actual_bc) {
                    let p = *p_row_j;
                    let v_off = (k_start + j) * head_dim;
                    crate::simd::simd_fused_scale_acc(
                        &mut o_tile[i * head_dim..],
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
        Some(buf) if buf.len() >= needed => {
            buf[..needed].fill(0.0);
            buf
        }
        _ => {
            scores_local = vec![0.0f32; needed];
            &mut scores_local
        }
    };
    for i in 0..seq_len {
        let q_off = i * head_dim;
        for j in 0..seq_len {
            let k_off = j * head_dim;
            scores[i * seq_len + j] = crate::simd::simd_dot_f32(
                &q[q_off..q_off + head_dim],
                &k[k_off..k_off + head_dim],
                head_dim,
            );
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
        let scores_off = i * seq_len;
        let out_off = i * head_dim;
        output[out_off..out_off + head_dim].fill(0.0);
        for j in 0..seq_len {
            let s = scores[scores_off + j];
            let v_off = j * head_dim;
            crate::simd::simd_fused_scale_acc(
                &mut output[out_off..out_off + head_dim],
                &v[v_off..v_off + head_dim],
                s,
                head_dim,
            );
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

    if total <= 2 || seq_len * head_dim < 1024 {
        // Sequential fallback for tiny workloads — avoids Rayon scheduling overhead
        let mut scores_buf = vec![0.0f32; scores_buf_size];
        let mut o_tile_buf = vec![0.0f32; o_tile_size];
        for idx in 0..total {
            let offset = idx * head_size;
            tiled_attention_forward_impl(
                &q[offset..offset + head_size],
                &k[offset..offset + head_size],
                &v[offset..offset + head_size],
                &mut output[offset..offset + head_size],
                seq_len,
                head_dim,
                scale,
                Some(&mut scores_buf[..scores_buf_size]),
                Some(&mut o_tile_buf[..o_tile_size]),
            );
        }
    } else {
        // Parallel for larger workloads — Rayon overhead amortized
        // Reuse grow-only scratch buffers per OS thread via thread_local.
        // Perf: UnsafeCell avoids RefCell's runtime borrow-check overhead.
        // Safety: Each Rayon worker thread gets its own thread_local slot,
        // so there is no actual concurrent access to the same cell.
        thread_local! {
            static SCORES_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
            static O_TILE_BUF: UnsafeCell<Vec<f32>> = const { UnsafeCell::new(Vec::new()) };
        }

        output
            .par_chunks_mut(head_size)
            .enumerate()
            .for_each(|(idx, out_chunk)| {
                let offset = idx * head_size;
                SCORES_BUF.with(|scores| {
                    O_TILE_BUF.with(|o_tile| {
                        // Safety: thread_local guarantees exclusive per-thread access.
                        // Rayon's work-stealing ensures each closure runs on one thread.
                        let scores = unsafe { &mut *scores.get() };
                        let o_tile = unsafe { &mut *o_tile.get() };
                        if scores.len() < scores_buf_size {
                            // `resize` zero-fills the new tail; the existing
                            // prefix is preserved but `tiled_attention_forward_impl`
                            // will zero the working range itself (L279), so no
                            // extra fill is needed here on the grow path either.
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
            });
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
        for d in 0..head_dim {
            let out_d = output[d];
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
