//! Score matrix computation: `Q·K^T / √d` with max-shift stabilization.
//!
//! This is the foundational kernel for AM. All subsequent steps (HighestAttn
//! aggregation, OMP mass features, β and Cv fitting) build on this output.
//!
//! Per AGENTS.md hot-loop rules:
//! - Caller pre-allocates the output buffer; we write in-place.
//! - 8-wide chunked inner loop enables SIMD auto-vectorization on AVX2/NEON.
//! - No allocation inside the hot loop.

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

    // 8-wide chunked inner loop — auto-vectorizes on AVX2 (8x f32) / NEON (4x f32).
    // Per AGENTS.md: chunked loops help LLVM auto-vectorize, branch-free inner.
    for i in 0..n {
        let q_row = &queries[i * d..(i + 1) * d];
        let out_row = &mut out[i * t_len..(i + 1) * t_len];
        for j in 0..t_len {
            let k_row = &keys[j * d..(j + 1) * d];
            // Chunked dot product in 8-element groups.
            let mut dot = 0.0f32;
            let mut k = 0usize;
            let chunk = 8usize;
            while k + chunk <= d {
                let q_chunk = &q_row[k..k + chunk];
                let k_chunk = &k_row[k..k + chunk];
                // Manually unrolled — compiler will SIMD this.
                dot += q_chunk[0] * k_chunk[0];
                dot += q_chunk[1] * k_chunk[1];
                dot += q_chunk[2] * k_chunk[2];
                dot += q_chunk[3] * k_chunk[3];
                dot += q_chunk[4] * k_chunk[4];
                dot += q_chunk[5] * k_chunk[5];
                dot += q_chunk[6] * k_chunk[6];
                dot += q_chunk[7] * k_chunk[7];
                k += chunk;
            }
            while k < d {
                dot += q_row[k] * k_row[k];
                k += 1;
            }
            out_row[j] = dot * inv_sqrt_d;
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
    scores: &[f32],   // (n, T) row-major, already scaled by inv_sqrt_d
    n: usize,
    t_len: usize,
    attn_out: &mut [f32], // (n, T) row-major
    mass_out: &mut [f32], // (n,) — Σ_j exp(s_ij) BEFORE normalization
) {
    assert_eq!(scores.len(), n * t_len);
    assert_eq!(attn_out.len(), n * t_len);
    assert_eq!(mass_out.len(), n);

    // Scratch buffer for row maxima — caller should reuse across calls but this
    // is small (n elements) and not in the tightest inner loop.
    let mut row_maxes = vec![f32::NEG_INFINITY; n];
    row_max(scores, n, t_len, &mut row_maxes);

    for i in 0..n {
        let max_s = row_maxes[i];
        let row = &scores[i * t_len..(i + 1) * t_len];
        let out_row = &mut attn_out[i * t_len..(i + 1) * t_len];
        let mut sum_exp = 0.0f32;
        for j in 0..t_len {
            let e = (row[j] - max_s).exp();
            out_row[j] = e;
            sum_exp += e;
        }
        let denom = if sum_exp < STABILITY_EPS {
            STABILITY_EPS
        } else {
            sum_exp
        };
        for j in 0..t_len {
            out_row[j] /= denom;
        }
        mass_out[i] = sum_exp;
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
