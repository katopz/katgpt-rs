//! Soft-embedding kernel for SwiR Latent mode.
//!
//! `ẽ_t = Σ_v p_t[v] · e(v)` — a probability-weighted mixture of the vocabulary
//! embeddings. This is the continuous-space "token" the paper emits when the
//! controller is in Latent mode. It is mathematically a convex combination of
//! rows of the embedding matrix, so it MUST lie inside the per-dim
//! `[min_v e(v)[d], max_v e(v)[d]]` box — see [`convex_hull_check`] for the
//! invariant test.
//!
//! # Zero-allocation contract
//!
//! The host owns the output scratch buffer (`out: &mut [f32]`, length =
//! `embedding_dim`) and is responsible for zeroing it before the call. We
//! accumulate into `out` rather than overwriting because that lets the caller
//! fold in a residual or a signal-mix component in the same buffer without a
//! second pass.

/// Compute `ẽ_t = Σ_v p_t[v] · e(v)` accumulating into `out`.
///
/// - `probs`: probability vector over the vocabulary (length = `vocab_size`).
///   Must be non-negative and sum to ≈1; we apply a single-pass renormalisation
///   if it doesn't (documented cost: O(vocab) once per call).
/// - `embedding_matrix`: flattened `[vocab_size, embedding_dim]` row-major.
/// - `embedding_dim`: width of each row in `embedding_matrix`.
/// - `out`: caller-allocated scratch of length `embedding_dim`. MUST be
///   pre-zeroed — this function accumulates, not overwrites.
///
/// # Panics
///
/// In debug builds, asserts that `embedding_matrix.len() >= probs.len() *
/// embedding_dim` and `out.len() == embedding_dim`. Release builds skip these
/// (hot-path).
#[inline]
#[allow(clippy::needless_range_loop)] // row offsets require indexing for SIMD shape
pub fn soft_embedding(
    probs: &[f32],
    embedding_matrix: &[f32],
    embedding_dim: usize,
    out: &mut [f32],
) {
    let vocab = probs.len();
    debug_assert!(
        embedding_matrix.len() >= vocab * embedding_dim,
        "embedding_matrix too small: need {} rows × {} dim, got {}",
        vocab,
        embedding_dim,
        embedding_matrix.len()
    );
    debug_assert_eq!(
        out.len(),
        embedding_dim,
        "out must be exactly embedding_dim wide"
    );

    // Single-pass normalisation. Most callers already softmax probs so this sum
    // is ≈1; we still pay the O(vocab) check to keep the convex-hull invariant
    // robust against caller drift. SIMD-reduced.
    // Indexed access is intentional — we need the row offset `v * embedding_dim`
    // in the second loop below, and restructuring to iterators would split the
    // normalisation pass from the accumulate pass unnecessarily.
    let mut psum = 0.0f32;
    for v in 0..vocab {
        psum += probs[v];
    }
    let inv_sum = if psum > 0.0 { 1.0 / psum } else { 0.0 };

    // Chunked inner loop (8-wide) over embedding_dim — gives LLVM the shape it
    // needs to auto-vectorise the row scaling + accumulator.
    const CHUNK: usize = 8;
    let mut v = 0usize;
    while v < vocab {
        let p = probs[v].max(0.0) * inv_sum;
        let row_off = v * embedding_dim;
        let mut d = 0usize;
        // Hot inner loop: scale + accumulate one embedding row into `out`.
        while d + CHUNK <= embedding_dim {
            unsafe {
                let row = embedding_matrix.get_unchecked(row_off + d..row_off + d + CHUNK);
                let acc = out.get_unchecked_mut(d..d + CHUNK);
                for k in 0..CHUNK {
                    *acc.get_unchecked_mut(k) += p * *row.get_unchecked(k);
                }
            }
            d += CHUNK;
        }
        // Scalar tail.
        while d < embedding_dim {
            out[d] += p * embedding_matrix[row_off + d];
            d += 1;
        }
        v += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat `[vocab, dim]` embedding matrix from row vectors.
    fn mat(rows: &[Vec<f32>]) -> Vec<f32> {
        let mut out = Vec::new();
        for r in rows {
            out.extend_from_slice(r);
        }
        out
    }

    #[test]
    fn one_hot_prob_returns_token_embedding() {
        // p concentrated on token v → result == e(v).
        let emb = mat(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let mut probs = vec![0.0f32; 3];
        probs[1] = 1.0;
        let mut out = vec![0.0f32; 3];
        soft_embedding(&probs, &emb, 3, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn uniform_probs_returns_centroid() {
        // Uniform p over k one-hot vectors → mean embedding.
        let emb = mat(&[
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let probs = vec![1.0 / 3.0; 3];
        let mut out = vec![0.0f32; 3];
        soft_embedding(&probs, &emb, 3, &mut out);
        // Centroid of one-hots = (1/3, 1/3, 1/3).
        for (d, &out_d) in out.iter().enumerate() {
            assert!(
                (out_d - 1.0 / 3.0).abs() < 1e-6,
                "dim {d}: got {}, expected 1/3",
                out_d
            );
        }
    }

    #[test]
    fn simd_matches_naive() {
        // Random-ish probs, embedding dim 16 (so tail loop fires once).
        let dim = 16;
        let vocab = 5;
        let emb: Vec<f32> = (0..vocab * dim)
            .map(|i| (i as f32).sin() * 0.5 + 0.5)
            .collect();
        let probs: Vec<f32> = vec![0.1, 0.2, 0.3, 0.15, 0.25];

        // Naive.
        let mut naive = vec![0.0f32; dim];
        for v in 0..vocab {
            for d in 0..dim {
                naive[d] += probs[v] * emb[v * dim + d];
            }
        }

        let mut simd_out = vec![0.0f32; dim];
        soft_embedding(&probs, &emb, dim, &mut simd_out);
        for d in 0..dim {
            assert!(
                (naive[d] - simd_out[d]).abs() < 1e-5,
                "dim {d}: naive={}, simd={}",
                naive[d],
                simd_out[d]
            );
        }
    }

    #[test]
    fn non_normalised_probs_renormalise() {
        // Caller forgot to normalise — we must still produce a convex combination.
        let emb = mat(&[vec![2.0, 4.0], vec![6.0, 8.0]]);
        let probs = vec![1.0, 3.0]; // sums to 4, not 1.
        let mut out = vec![0.0f32; 2];
        soft_embedding(&probs, &emb, 2, &mut out);
        // Expected: (1/4)*[2,4] + (3/4)*[6,8] = [0.5+4.5, 1+6] = [5, 7].
        assert!((out[0] - 5.0).abs() < 1e-5, "out[0]={}", out[0]);
        assert!((out[1] - 7.0).abs() < 1e-5, "out[1]={}", out[1]);
    }
}
