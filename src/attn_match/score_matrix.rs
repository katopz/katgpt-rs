//! Score matrix computation: `Q·K^T / √d` with max-shift stabilization.
//!
//! This is the foundational kernel for AM. All subsequent steps (HighestAttn
//! aggregation, OMP mass features, β and Cv fitting) build on this output.
//!
//! Per AGENTS.md hot-loop rules:
//! - Caller pre-allocates the output buffer; we write in-place.
//! - 8-wide chunked inner loop enables SIMD auto-vectorization on AVX2/NEON.
//! - No allocation inside the hot loop.

#![allow(clippy::needless_range_loop)]

use crate::attn_match::STABILITY_EPS;

/// Compute the score matrix `S = Q·K^T · inv_sqrt_d` for `(n, d)` queries and
/// `(T, d)` keys. Output is row-major `(n, T)` written into the caller-provided
/// `out` buffer (length `n * T`).
///
/// The output is NOT stabilized — callers must apply max-shift before `exp()`.
/// Use [`compute_softmax_attention`] for the stabilized softmax version.
///
/// # Panics
/// Panics if `out.len() != n * T` or if dimension mismatches.
#[inline]
pub fn compute_score_matrix(
    queries: &[f32], // (n, d) row-major
    keys: &[f32],    // (T, d) row-major
    n: usize,
    t_len: usize,
    d: usize,
    out: &mut [f32], // (n, T) row-major
) {
    assert_eq!(queries.len(), n * d, "queries buffer size mismatch");
    assert_eq!(keys.len(), t_len * d, "keys buffer size mismatch");
    assert_eq!(out.len(), n * t_len, "output buffer size mismatch");

    let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();

    // Reuse the shared `dot_8wide` kernel (8-wide chunked FMA, auto-vectorizes
    // on AVX2/NEON) instead of a duplicated manual unroll. Keeps the scalar and
    // SIMD paths in `score_matrix_simd` bit-identical by construction (DRY).
    use crate::attn_match::score_matrix_simd::dot_8wide;
    for i in 0..n {
        let q_row = &queries[i * d..(i + 1) * d];
        let out_row = &mut out[i * t_len..(i + 1) * t_len];
        for j in 0..t_len {
            let k_row = &keys[j * d..(j + 1) * d];
            out_row[j] = dot_8wide(q_row, k_row, d) * inv_sqrt_d;
        }
    }
}

/// Compute per-query row-max of an `(n, T)` row-major matrix.
///
/// Used for the max-shift stabilization before softmax.
#[inline]
pub fn row_max(matrix: &[f32], n: usize, t_len: usize, out: &mut [f32]) {
    assert_eq!(matrix.len(), n * t_len);
    assert_eq!(out.len(), n);
    for i in 0..n {
        let row = &matrix[i * t_len..(i + 1) * t_len];
        let mut m = row[0];
        for &v in &row[1..] {
            m = m.max(v);
        }
        out[i] = m;
    }
}

/// Compute the softmax attention matrix `A = softmax(S)` where `S = Q·K^T / √d`.
///
/// Returns the `(n, T)` row-major matrix in `attn_out` and the unnormalized
/// mass per query (sum of `exp(s_ij)` BEFORE normalization) in `mass_out`.
/// The mass vector is needed by the β NNLS fitter.
///
/// Numerical stability: applies the standard per-row max-shift before `exp()`.
///
/// # Panics
/// Panics on dimension mismatch.
#[inline]
pub fn compute_softmax_attention(
    scores: &[f32], // (n, T) row-major, already scaled by inv_sqrt_d
    n: usize,
    t_len: usize,
    attn_out: &mut [f32], // (n, T) row-major
    mass_out: &mut [f32], // (n,) — Σ_j exp(s_ij) BEFORE normalization
) {
    assert_eq!(scores.len(), n * t_len);
    assert_eq!(attn_out.len(), n * t_len);
    assert_eq!(mass_out.len(), n);

    // Per-row fused softmax: max-shift + exp + normalize, with no scratch
    // allocation. The row is hot in L1 between the max scan and the exp pass
    // for typical `t_len`, so fusing beats the prior two-buffer approach
    // (`row_maxes` Vec + separate `row_max` call). Replaces `t_len` divisions
    // per row with 1 reciprocal + `t_len` multiplies.
    for i in 0..n {
        let row = &scores[i * t_len..(i + 1) * t_len];
        let out_row = &mut attn_out[i * t_len..(i + 1) * t_len];

        // Pass 1: per-row max (scalar accumulator, no buffer).
        let mut max_s = row[0];
        for &v in &row[1..] {
            max_s = max_s.max(v);
        }

        // Pass 2: shifted exp + accumulate sum_exp.
        let mut sum_exp = 0.0f32;
        for j in 0..t_len {
            let e = (row[j] - max_s).exp();
            out_row[j] = e;
            sum_exp += e;
        }

        // Pass 3: normalize via reciprocal-multiply (1 division + N multiplies).
        let denom = sum_exp.max(STABILITY_EPS);
        let inv_denom = 1.0 / denom;
        for out in out_row.iter_mut() {
            *out *= inv_denom;
        }
        mass_out[i] = sum_exp;
    }
}

/// Fused softmax + attention-output kernel.
///
/// Computes the unnormalized mass `m_i = Σ_j exp(s_ij − max_i)`, the
/// normalized attention-output `Y_i = (Σ_j exp(s_ij − max_i) · V_j) / m_i`,
/// and optionally the normalized attention weights `A_ij = exp(...)/m_i`.
///
/// # Why fuse
///
/// Separately, `compute_softmax_attention` writes the full `(n, t_len)`
/// attention matrix into `attn_out`, and `compute_attention_output` then
/// reads it back to compute `Y = A · V`. The fused kernel folds the `exp`
/// directly into the `V` reduction, eliminating the `(n, t_len)` materialized
/// attention matrix when the caller does not need it (i.e. when
/// `attn_out_opt` is `None`).
///
/// # Float-op reordering (disclosed)
///
/// The unfused path computes `A_ij = exp(...)/m_i` first, then
/// `Y_ik = Σ_j A_ij · V_jk`. The fused path accumulates
/// `num_ik = Σ_j exp(...) · V_jk` and divides by `m_i` once at the end:
/// `Y_ik = num_ik / m_i`. Distributing the division changes last-bit rounding
/// (fewer divisions → arguably more accurate). This is a deliberate fusion
/// tradeoff, not a bit-identical rewrite.
///
/// # Panics
/// Panics if `scores.len() != n*t_len`, `values.len() != t_len*d`,
/// `mass_out.len() != n`, `y_out.len() != n*d`, or if `attn_out_opt` is
/// `Some` with a slice whose length `!= n*t_len`.
#[inline]
#[allow(clippy::too_many_arguments)] // hot-path kernel: raw-slice args avoid struct indirection
pub fn compute_softmax_attention_and_output(
    scores: &[f32], // (n, t_len) row-major, already scaled by inv_sqrt_d
    values: &[f32], // (t_len, d) row-major
    n: usize,
    t_len: usize,
    d: usize,
    mass_out: &mut [f32], // (n,) — Σ_j exp(s_ij) BEFORE normalization
    y_out: &mut [f32],    // (n, d) — must be pre-zeroed
    mut attn_out_opt: Option<&mut [f32]>, // (n, t_len) if the normalized weights are needed downstream
) {
    assert_eq!(scores.len(), n * t_len);
    assert_eq!(values.len(), t_len * d);
    assert_eq!(mass_out.len(), n);
    assert_eq!(y_out.len(), n * d);
    for i in 0..n {
        let row = &scores[i * t_len..(i + 1) * t_len];
        let y_row = &mut y_out[i * d..(i + 1) * d];

        // Pass 1: per-row max (max-shift stabilization).
        let mut max_s = row[0];
        for &v in &row[1..] {
            max_s = max_s.max(v);
        }

        // Pass 2: shifted exp → accumulate mass + Y numerator (+ optional attn).
        let mut sum_exp = 0.0f32;
        for j in 0..t_len {
            let e = (row[j] - max_s).exp();
            sum_exp += e;
            // Y numerator: num_ik += e * V_jk. Sequential reads of V_j.
            let v_row = &values[j * d..(j + 1) * d];
            for k in 0..d {
                y_row[k] += e * v_row[k];
            }
        }

        // Pass 3: divide numerators by the mass → normalized Y and attn.
        let denom = sum_exp.max(STABILITY_EPS);
        let inv_denom = 1.0 / denom;
        for k in 0..d {
            y_row[k] *= inv_denom;
        }
        mass_out[i] = sum_exp;
        if let Some(attn_out) = &mut attn_out_opt {
            // Reconstruct normalized A_ij = exp(...)/denom. One extra exp pass,
            // only when the caller actually needs the attention matrix (report).
            let attn_row = &mut attn_out[i * t_len..(i + 1) * t_len];
            for j in 0..t_len {
                attn_row[j] = (row[j] - max_s).exp() * inv_denom;
            }
        }
    }
}

/// Compute the attention output `Y = A · V` where `A` is `(n, t_len)` attention
/// weights and `V` is `(t_len, d)` values. Output is `(n, d)` row-major.
///
/// This is the standard attention output computation `Y_i = Σ_j A_ij V_j`,
/// shared between [`crate::attn_match::compact::compact`] and
/// [`crate::attn_match::compact_with_fixed_beta`] for building the Cv-fit
/// target. Extracted as a single source of truth so both fast paths benefit
/// from any future kernel improvement.
///
/// # Loop order
///
/// `i` (query) × `j` (source token) × `k` (head dim), with `k` innermost.
/// Both `values[j*d..]` and `out[i*d..]` are read/written sequentially within
/// a fixed `i,j` — the cache-friendly ijk GEMM variant. The prior duplicated
/// code had a k-outer / j-inner form that did strided `values[j*d + k]` reads.
///
/// # Panics
/// Panics on dimension mismatch.
#[inline]
pub fn compute_attention_output(
    attn: &[f32],   // (n, t_len) row-major — normalized attention weights
    values: &[f32], // (t_len, d) row-major
    n: usize,
    t_len: usize,
    d: usize,
    out: &mut [f32], // (n, d) row-major — must be pre-zeroed
) {
    assert_eq!(attn.len(), n * t_len, "attn buffer size mismatch");
    assert_eq!(values.len(), t_len * d, "values buffer size mismatch");
    assert_eq!(out.len(), n * d, "output buffer size mismatch");
    for i in 0..n {
        let attn_row = &attn[i * t_len..(i + 1) * t_len];
        let y_row = &mut out[i * d..(i + 1) * d];
        for j in 0..t_len {
            let a = attn_row[j];
            let v_row = &values[j * d..(j + 1) * d];
            for k in 0..d {
                y_row[k] += a * v_row[k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_matrix_basic() {
        // 2 queries, 3 keys, dim 2
        let queries = [1.0, 0.0, 0.0, 1.0]; // (2,2)
        let keys = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // (3,2)
        let mut out = vec![0.0f32; 6];
        compute_score_matrix(&queries, &keys, 2, 3, 2, &mut out);
        let inv_sqrt_d = 1.0f32 / 2.0f32.sqrt();
        // q0 = (1,0): scores = [1, 0, 1] * inv_sqrt_d
        // q1 = (0,1): scores = [0, 1, 1] * inv_sqrt_d
        assert!((out[0] - 1.0 * inv_sqrt_d).abs() < 1e-6);
        assert!((out[1] - 0.0).abs() < 1e-6);
        assert!((out[2] - 1.0 * inv_sqrt_d).abs() < 1e-6);
        assert!((out[3] - 0.0).abs() < 1e-6);
        assert!((out[4] - 1.0 * inv_sqrt_d).abs() < 1e-6);
        assert!((out[5] - 1.0 * inv_sqrt_d).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_attention_sums_to_one() {
        // 1 query, 4 keys — all equal scores → uniform 0.25
        let scores = [1.0f32, 1.0, 1.0, 1.0];
        let mut attn = vec![0.0; 4];
        let mut mass = vec![0.0; 1];
        compute_softmax_attention(&scores, 1, 4, &mut attn, &mut mass);
        for &a in &attn {
            assert!((a - 0.25).abs() < 1e-6, "softmax row should be uniform");
        }
        // Mass = 4 (since all shifted to exp(0)=1)
        assert!((mass[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_attention_peaks_at_max() {
        let scores = [-1.0f32, 5.0, -2.0, 0.5];
        let mut attn = vec![0.0; 4];
        let mut mass = vec![0.0; 1];
        compute_softmax_attention(&scores, 1, 4, &mut attn, &mut mass);
        // Peak should be at index 1
        let peak_idx = attn
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(peak_idx, 1);
        // Sum to one
        let sum: f32 = attn.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_row_max() {
        let m = [1.0f32, 3.0, 2.0, 5.0, -1.0, 0.0]; // (2,3)
        let mut out = vec![0.0; 2];
        row_max(&m, 2, 3, &mut out);
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 5.0);
    }

    #[test]
    fn test_score_matrix_with_chunked_d() {
        // d=10 to exercise both chunked and remainder paths
        let n = 2;
        let t = 3;
        let d = 10;
        let queries: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.1).collect();
        let keys: Vec<f32> = (0..t * d).map(|i| (i as f32) * 0.05).collect();
        let mut out = vec![0.0f32; n * t];
        compute_score_matrix(&queries, &keys, n, t, d, &mut out);
        // Manual check for entry (0, 0)
        let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
        let mut expected = 0.0f32;
        for k in 0..d {
            expected += queries[k] * keys[k];
        }
        expected *= inv_sqrt_d;
        assert!((out[0] - expected).abs() < 1e-5);
    }
}
