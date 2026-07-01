//! Rayon-parallel blocked score matrix kernel (Plan 271 Phase 2, T2.3).
//!
//! Implements `S = Q·K^T · inv_sqrt_d` parallelized across query rows in
//! L2-resident blocks. Each rayon task owns a contiguous slice of `out` so
//! no atomics or locks are needed in the hot loop — the score matrix is
//! row-independent, so the only "merge" is the trivial disjoint write into
//! `out`.
//!
//! # When to use
//!
//! The router dispatches to this backend when `t >= simd_max_t` and no GPU
//! is available (see `router::SolverBackend::CpuRayon`). For typical
//! 8-core Apple Silicon, this wins starting around `T ≈ 2048` and `n ≥ 8`.
//!
//! # Block layout
//!
//! `block_rows = 4096 / d` rows per block — keeps `(block_rows × d)` query
//! slice + `(T × d)` key slice together in L2. For `d=64` this is 64 query
//! rows per block.
//!
//! Per AGENTS.md hot-loop rules:
//! - Caller pre-allocates `out`; we write in-place.
//! - No allocation inside rayon tasks — the inner kernel reuses the SIMD
//!   `dot_8wide` from `score_matrix_simd`.
//! - Per-block work far exceeds rayon dispatch overhead (~5 μs) by design.

use crate::score_matrix_simd::dot_8wide;
use rayon::prelude::*;

/// Default L2-resident block size in bytes. With `d` f32 elements per row,
/// this yields `block_rows = DEFAULT_BLOCK_BYTES / (d * 4)` rows per block.
///
/// Conservative: 32 KB L1, 256 KB–1 MB L2 on Apple Silicon / Zen 4. We pick
/// 4 KB so two blocks (Q + K-row-window) plus the output row fit in L1, and
/// the rest of the keys stream from L2.
pub const DEFAULT_BLOCK_BYTES: usize = 4096;

/// Compute the score matrix `S = Q·K^T · inv_sqrt_d` in parallel using rayon.
///
/// This is a drop-in replacement for [`crate::score_matrix::compute_score_matrix`]
/// that parallelizes across query rows in L2-resident blocks. The output is
/// **not** stabilized — callers must apply max-shift before `exp()`.
///
/// # Arguments
/// * `queries` - `(n, d)` row-major query vectors.
/// * `keys` - `(T, d)` row-major key vectors.
/// * `n` - Number of queries.
/// * `t_len` - Number of keys (the paper's `T`).
/// * `d` - Head dimension.
/// * `out` - Caller-allocated `(n, T)` row-major output buffer.
///
/// # Panics
/// Panics on dimension mismatch. No-ops if `n == 0` or `t_len == 0`.
#[inline]
pub fn compute_score_matrix_rayon(
    queries: &[f32],
    keys: &[f32],
    n: usize,
    t_len: usize,
    d: usize,
    out: &mut [f32],
) {
    assert_eq!(queries.len(), n * d, "queries buffer size mismatch");
    assert_eq!(keys.len(), t_len * d, "keys buffer size mismatch");
    assert_eq!(out.len(), n * t_len, "output buffer size mismatch");
    if n == 0 || t_len == 0 {
        return;
    }

    let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
    // Block size: rows per rayon task. Default 4 KB / (d*4) = 1024/d rows.
    // For d=64 → 16 rows/block. For d=128 → 8 rows/block. This keeps the
    // query slice (16 * 64 * 4 = 4 KB) and one key-row window (64 * 4 = 256 B)
    // both in L1.
    let rows_per_block = compute_block_rows(d);

    // Parallelize across query-row blocks. Each block owns a disjoint slice
    // of `out` so no atomic accumulate or synchronization is needed. Block
    // size is `rows_per_block * t_len` cells per slice.
    out.par_chunks_mut(rows_per_block * t_len)
        .enumerate()
        .for_each(|(block_idx, out_chunk)| {
            let q_start = block_idx * rows_per_block;
            let q_end = (q_start + rows_per_block).min(n);
            let rows_in_block = q_end - q_start;
            // Inner loop: per-row dot product, reuses the SIMD `dot_8wide`.
            for i in 0..rows_in_block {
                let q_row = &queries[(q_start + i) * d..(q_start + i + 1) * d];
                let out_row = &mut out_chunk[i * t_len..(i + 1) * t_len];
                for j in 0..t_len {
                    let k_row = &keys[j * d..(j + 1) * d];
                    out_row[j] = dot_8wide(q_row, k_row, d) * inv_sqrt_d;
                }
            }
        });
}

/// Compute the score matrix in parallel with an explicit per-row reduction
/// over reference-query attention scores. Used by the HighestAttn key
/// selector when `T` is large enough that parallelism beats the linear scan.
///
/// Produces `out_rms[j] = sqrt(mean_i(score_ij^2))` — the RMS attention mass
/// for each of the `T` keys, computed in parallel across query rows.
///
/// Because multiple rayon tasks may contribute to the same `out_rms[j]`,
/// we use **per-task local accumulators** and merge at the end (no atomics
/// in the hot loop).
///
/// # Panics
/// Panics on dimension mismatch.
#[inline]
pub fn compute_rms_attention_rayon(
    queries: &[f32],
    keys: &[f32],
    n: usize,
    t_len: usize,
    d: usize,
    out_rms: &mut [f32],
) {
    assert_eq!(queries.len(), n * d, "queries buffer size mismatch");
    assert_eq!(keys.len(), t_len * d, "keys buffer size mismatch");
    assert_eq!(out_rms.len(), t_len, "rms output size mismatch");
    if n == 0 || t_len == 0 {
        return;
    }

    let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
    let rows_per_block = compute_block_rows(d);

    // Parallelize across query-row blocks using `fold` + `reduce` so each rayon
    // worker owns a single reusable `(t_len,)` accumulator (allocated once per
    // worker, reused across all chunks the worker processes) instead of
    // allocating a fresh `Vec` per chunk and collecting into a `Vec<Vec<f64>>`
    // for a sequential merge. This eliminates the outer `Vec` allocation and
    // the post-parallel merge loop. Per AGENTS.md hot-loop rules: no allocation
    // inside the parallel closure body.
    //
    // Reduction is over rows (not cells), so no per-cell atomic is needed —
    // the only merge is the disjoint thread-local accumulator sum in `reduce`.
    let n_f64 = n as f64;
    let total_sum_sq: Vec<f64> = queries
        .par_chunks(rows_per_block * d)
        .fold(
            || vec![0.0f64; t_len],
            |mut acc, q_block| {
                let rows_in_block = q_block.len() / d;
                for i in 0..rows_in_block {
                    let q_row = &q_block[i * d..(i + 1) * d];
                    for j in 0..t_len {
                        let k_row = &keys[j * d..(j + 1) * d];
                        let s = (dot_8wide(q_row, k_row, d) * inv_sqrt_d) as f64;
                        acc[j] += s * s;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; t_len],
            |mut a, b| {
                for j in 0..t_len {
                    a[j] += b[j];
                }
                a
            },
        );
    for j in 0..t_len {
        out_rms[j] = (total_sum_sq[j] / n_f64).sqrt() as f32;
    }
}

/// Pick a block-row count that keeps the query slice L2-resident.
#[inline]
fn compute_block_rows(d: usize) -> usize {
    if d == 0 {
        return 1;
    }
    let bytes_per_row = d * core::mem::size_of::<f32>();
    let rows = DEFAULT_BLOCK_BYTES / bytes_per_row.max(1);
    rows.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score_matrix::compute_score_matrix;

    /// Parity vs the scalar reference kernel.
    #[test]
    fn test_rayon_matches_scalar() {
        let n = 8;
        let t_len = 16;
        let d = 32;
        let mut seed = 42u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let queries: Vec<f32> = (0..n * d).map(|_| rng()).collect();
        let keys: Vec<f32> = (0..t_len * d).map(|_| rng()).collect();

        let mut scalar = vec![0.0f32; n * t_len];
        compute_score_matrix(&queries, &keys, n, t_len, d, &mut scalar);

        let mut rayon_out = vec![0.0f32; n * t_len];
        compute_score_matrix_rayon(&queries, &keys, n, t_len, d, &mut rayon_out);

        for i in 0..n * t_len {
            assert!(
                (scalar[i] - rayon_out[i]).abs() < 1e-5,
                "rayon/scalar mismatch at {}: scalar={} rayon={}",
                i,
                scalar[i],
                rayon_out[i]
            );
        }
    }

    /// RMS attention: block-mean of squared scores matches the closed form.
    #[test]
    fn test_rms_attention_known() {
        // 2 queries, 4 keys, d=2. All-ones Q and K → every score = 2 * inv_sqrt_d.
        let n = 2;
        let t_len = 4;
        let d = 2;
        let queries = vec![1.0f32; n * d];
        let keys = vec![1.0f32; t_len * d];

        let mut rms = vec![0.0f32; t_len];
        compute_rms_attention_rayon(&queries, &keys, n, t_len, d, &mut rms);

        let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
        let expected_score = (2.0f32 * inv_sqrt_d) as f64;
        let expected_rms = expected_score; // both queries contribute the same score
        for &r in &rms {
            assert!(
                (r as f64 - expected_rms).abs() < 1e-5,
                "rms {} should be {}",
                r,
                expected_rms
            );
        }
    }

    /// Empty input is a no-op (no panic, no writes).
    #[test]
    fn test_rayon_empty() {
        let queries: Vec<f32> = vec![];
        let keys: Vec<f32> = vec![];
        let mut out: Vec<f32> = vec![];
        compute_score_matrix_rayon(&queries, &keys, 0, 0, 8, &mut out);
        assert!(out.is_empty());
    }

    /// Block size picks a sane value for common head dims.
    #[test]
    fn test_block_rows_sane() {
        assert_eq!(compute_block_rows(64), 16); // 4096 / 256
        assert_eq!(compute_block_rows(128), 8); // 4096 / 512
        assert_eq!(compute_block_rows(0), 1); // defensive
    }

    /// Larger workload parity check (exercises multi-block rayon dispatch).
    #[test]
    fn test_rayon_matches_scalar_large() {
        let n = 64;
        let t_len = 256;
        let d = 64;
        let mut seed = 7u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let queries: Vec<f32> = (0..n * d).map(|_| rng()).collect();
        let keys: Vec<f32> = (0..t_len * d).map(|_| rng()).collect();

        let mut scalar = vec![0.0f32; n * t_len];
        compute_score_matrix(&queries, &keys, n, t_len, d, &mut scalar);

        let mut rayon_out = vec![0.0f32; n * t_len];
        compute_score_matrix_rayon(&queries, &keys, n, t_len, d, &mut rayon_out);

        let mut max_diff = 0.0f32;
        for i in 0..n * t_len {
            max_diff = max_diff.max((scalar[i] - rayon_out[i]).abs());
        }
        assert!(max_diff < 1e-4, "max rayon/scalar diff: {}", max_diff);
    }
}
