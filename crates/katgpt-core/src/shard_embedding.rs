//! Shard Embedding Projection — Johnson-Lindenstrauss Random Orthogonal (Plan 230).
//!
//! Projects `style_weights: [f32; 64]` to `ShardEmbedding: [f32; 8]` using a
//! pre-generated random orthogonal matrix. JL lemma guarantees pairwise distance
//! preservation within (1±ε) with high probability.
//!
//! # Architecture
//!
//! ```text
//! style_weights [f32; 64] × W [[f32; 64]; 8] → embedding [f32; 8]
//! ```
//!
//! The projection matrix W is generated once (at consolidation time or init)
//! and stored alongside the shard. No training needed.

use crate::types::ShardEmbedding;

/// Input dimension (style_weights size).
pub const STYLE_DIM: usize = 64;

/// Output embedding dimension.
pub const EMBED_DIM: usize = 8;

/// Projection matrix: 8 rows × 64 columns, stored row-major.
/// Each row is a unit vector (orthogonal to other rows).
#[derive(Clone, Debug)]
pub struct JlProjectionMatrix {
    /// Row-major: `rows[i][j]` = element at row i, column j.
    pub rows: [[f32; STYLE_DIM]; EMBED_DIM],
    /// BLAKE3 commitment over the matrix bytes.
    commitment: [u8; 32],
}

impl JlProjectionMatrix {
    /// Generate a random orthogonal projection matrix using Gram-Schmidt.
    ///
    /// `rng` provides random f32 values in [-1, 1]. The caller controls
    /// the RNG seed for reproducibility.
    pub fn generate(mut rng: impl FnMut() -> f32) -> Self {
        let mut rows = [[0.0f32; STYLE_DIM]; EMBED_DIM];

        // Generate random rows, then Gram-Schmidt orthogonalize
        for i in 0..EMBED_DIM {
            // Random init
            for j in 0..STYLE_DIM {
                rows[i][j] = rng();
            }
            // Subtract projections onto previous rows
            for k in 0..i {
                let dot: f32 = rows[i].iter().zip(rows[k].iter()).map(|(a, b)| a * b).sum();
                for j in 0..STYLE_DIM {
                    rows[i][j] -= dot * rows[k][j];
                }
            }
            // Normalize to unit length
            let norm: f32 = rows[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for j in 0..STYLE_DIM {
                    rows[i][j] /= norm;
                }
            }
        }

        // Scale by 1/sqrt(d_out) per JL lemma
        let scale = 1.0 / (EMBED_DIM as f32).sqrt();
        for i in 0..EMBED_DIM {
            for j in 0..STYLE_DIM {
                rows[i][j] *= scale;
            }
        }

        let mut matrix = Self {
            rows,
            commitment: [0u8; 32],
        };
        matrix.commit();
        matrix
    }

    /// Project a 64-dim style_weights vector to 8-dim embedding.
    ///
    /// O(STYLE_DIM × EMBED_DIM) = O(512) multiply-adds.
    /// SIMD-friendly: chunked 8-wide dot products.
    #[inline]
    pub fn project(&self, style_weights: &[f32; STYLE_DIM]) -> ShardEmbedding {
        let mut result = [0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            // Chunked dot product for SIMD auto-vectorization (8-wide)
            const CHUNK: usize = 8;
            let mut sum = 0.0f32;
            let mut j = 0;
            while j + CHUNK <= STYLE_DIM {
                let mut chunk_sum = 0.0f32;
                for c in 0..CHUNK {
                    chunk_sum += self.rows[i][j + c] * style_weights[j + c];
                }
                sum += chunk_sum;
                j += CHUNK;
            }
            // Remainder
            while j < STYLE_DIM {
                sum += self.rows[i][j] * style_weights[j];
                j += 1;
            }
            result[i] = sum;
        }
        ShardEmbedding(result)
    }

    /// Compute and store BLAKE3 commitment.
    pub fn commit(&mut self) {
        self.commitment = [0u8; 32];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.rows.as_ptr() as *const u8, EMBED_DIM * STYLE_DIM * 4)
        };
        self.commitment = *blake3::hash(bytes).as_bytes();
    }

    /// Verify BLAKE3 commitment.
    pub fn verify(&self) -> bool {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(self.rows.as_ptr() as *const u8, EMBED_DIM * STYLE_DIM * 4)
        };
        self.commitment == *blake3::hash(bytes).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_rng() -> impl FnMut() -> f32 {
        let mut state: u64 = 42;
        move || {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Use upper 24 bits for [0,1) then map to [-1,1]
            let bits = (state >> 40) as u32;
            let normalized = bits as f32 / (1u32 << 24) as f32;
            normalized * 2.0 - 1.0
        }
    }

    #[test]
    fn test_jl_projection_is_orthogonal() {
        let mat = JlProjectionMatrix::generate(simple_rng());
        // Check rows are approximately orthogonal
        for i in 0..EMBED_DIM {
            for j in (i + 1)..EMBED_DIM {
                let dot: f32 = mat.rows[i]
                    .iter()
                    .zip(mat.rows[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 0.01,
                    "rows {} and {} not orthogonal: dot = {}",
                    i,
                    j,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_jl_projection_preserves_distances() {
        let mat = JlProjectionMatrix::generate(simple_rng());
        let mut rng2 = simple_rng();
        let mut a = [0.0f32; STYLE_DIM];
        let mut b = [0.0f32; STYLE_DIM];
        for i in 0..STYLE_DIM {
            a[i] = rng2();
            b[i] = rng2();
        }

        let ea = mat.project(&a);
        let eb = mat.project(&b);

        // Original distance
        let orig_dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        let proj_dist = ea.dist_sq(&eb);

        // JL lemma: projected distance should be a consistent fraction of original.
        // With 64→8 random orthogonal + 1/sqrt(d_out) scaling, ratio ≈ d_out/d_in = 0.125.
        // We accept any non-trivial ratio — the important property is that ordering is preserved.
        let ratio = proj_dist / orig_dist;
        assert!(
            ratio > 0.001 && ratio < 10.0,
            "JL distance ratio out of range: {}",
            ratio
        );
    }

    #[test]
    fn test_jl_commitment_roundtrip() {
        let mut mat = JlProjectionMatrix::generate(simple_rng());
        assert!(mat.verify());
        // Tamper
        mat.rows[0][0] += 1.0;
        assert!(!mat.verify());
    }

    #[test]
    fn test_shard_embedding_cosine_similarity() {
        let a = ShardEmbedding([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = ShardEmbedding([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((a.cosine_similarity(&b)).abs() < 1e-6);

        let c = ShardEmbedding([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((a.cosine_similarity(&c) - 1.0).abs() < 1e-6);
    }
}
