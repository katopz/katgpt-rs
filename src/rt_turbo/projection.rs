//! Low-dimensional pre-RoPE projection for retrieval head scoring.
//!
//! Implements the 16-dimensional projection from the RTPurbo paper (arXiv 2605.16928).
//! Projects pre-RoPE query/key vectors into a low-dimensional space for efficient
//! relevance scoring during decode.
//!
//! # Key Insight
//!
//! High-frequency RoPE components are noise for long-range retrieval. Projecting
//! pre-RoPE vectors to 16 dimensions captures the low-frequency information that
//! retrieval heads actually use, achieving >90% recall at 97% sparsity.
//!
//! # Projection Math
//!
//! For retrieval head h with projection matrices W_Q, W_K ∈ ℝ^{head_dim × low_dim}:
//!
//! ```text
//! q_proj = q_pre^T @ W_Q  → [low_dim]   (project down)
//! k_proj = k_pre^T @ W_K  → [low_dim]   (project down)
//! s(m,n) = q_proj · k_proj  → f32        (relevance score)
//! ```
//!
//! # Storage Layout
//!
//! Weights are stored in row-major order per retrieval head:
//! - `w_q[h * head_dim * low_dim + i * low_dim + j]` = W_Q for head h, row i, col j
//! - Same layout for `w_k`

use serde::{Deserialize, Serialize};

/// Low-dimensional pre-RoPE projection weights for all retrieval heads.
///
/// Each retrieval head has its own pair of projection matrices W_Q, W_K
/// of shape `[head_dim, low_dim]`. These are trained via the RTPurbo
/// two-stage distillation pipeline and loaded at inference time.
///
/// # Dimensions
///
/// For a typical model with head_dim=128, low_dim=16, and ~5 retrieval heads:
/// - Per head: 2 × 128 × 16 = 4,096 floats (16 KB)
/// - Total: ~80 KB — fits comfortably in L2 cache
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalProjection {
    /// Query projection weights: `[n_retrieval_heads * head_dim * low_dim]`.
    /// Layout: `w_q[h * stride + i * low_dim + j]` where stride = head_dim * low_dim.
    w_q: Vec<f32>,
    /// Key projection weights: same layout as `w_q`.
    w_k: Vec<f32>,
    /// Number of retrieval heads.
    n_retrieval_heads: usize,
    /// Dimension of each attention head (e.g., 128).
    head_dim: usize,
    /// Low-dimensional projection size (e.g., 16).
    low_dim: usize,
}

impl RetrievalProjection {
    /// Stride per head in the weight arrays: `head_dim * low_dim`.
    #[inline]
    fn head_stride(&self) -> usize {
        self.head_dim * self.low_dim
    }

    /// Total weight elements per array: `n_retrieval_heads * head_dim * low_dim`.
    #[inline]
    fn total_weights(&self) -> usize {
        self.n_retrieval_heads * self.head_stride()
    }

    /// Create zero-initialized projection weights.
    ///
    /// Used as initialization before training. All projected scores will be zero.
    pub fn zeros(n_retrieval_heads: usize, head_dim: usize, low_dim: usize) -> Self {
        let total = n_retrieval_heads * head_dim * low_dim;
        Self {
            w_q: vec![0.0; total],
            w_k: vec![0.0; total],
            n_retrieval_heads,
            head_dim,
            low_dim,
        }
    }

    /// Create identity-like projection weights.
    ///
    /// W_Q and W_K have 1.0 in the first `low_dim` diagonal positions,
    /// zero elsewhere. This makes the projection equivalent to selecting the
    /// first `low_dim` dimensions of each vector — useful for testing and
    /// as a sanity check baseline.
    pub fn identity(n_retrieval_heads: usize, head_dim: usize, low_dim: usize) -> Self {
        let stride = head_dim * low_dim;
        let total = n_retrieval_heads * stride;
        let mut w_q = vec![0.0f32; total];
        let mut w_k = vec![0.0f32; total];

        for h in 0..n_retrieval_heads {
            let offset = h * stride;
            // Set diagonal: W[i][i] = 1.0 for i in 0..low_dim
            for i in 0..low_dim {
                w_q[offset + i * low_dim + i] = 1.0;
                w_k[offset + i * low_dim + i] = 1.0;
            }
        }

        Self {
            w_q,
            w_k,
            n_retrieval_heads,
            head_dim,
            low_dim,
        }
    }

    /// Create projection with random weights (Xavier/Glorot uniform initialization).
    ///
    /// Scale = sqrt(6.0 / (head_dim + low_dim)), matching Glorot uniform.
    /// Uses a deterministic xorshift64 PRNG seeded from a fixed constant.
    pub fn xavier(n_retrieval_heads: usize, head_dim: usize, low_dim: usize) -> Self {
        let scale = (6.0f32 / (head_dim + low_dim) as f32).sqrt();
        let stride = head_dim * low_dim;
        let total = n_retrieval_heads * stride;

        // Deterministic xorshift64 PRNG for reproducible init
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next_random = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Take top 23 bits to form mantissa → float in [1.0, 2.0)
            let bits = ((state >> 41) as u32) | 0x3F80_0000;
            let f = f32::from_bits(bits) - 1.0; // [0.0, 1.0)
            (f * 2.0 - 1.0) * scale // [-scale, +scale)
        };

        let w_q: Vec<f32> = (0..total).map(|_| next_random()).collect();
        let w_k: Vec<f32> = (0..total).map(|_| next_random()).collect();

        Self {
            w_q,
            w_k,
            n_retrieval_heads,
            head_dim,
            low_dim,
        }
    }

    /// Create projection from raw weight arrays.
    ///
    /// # Panics
    ///
    /// Panics if weight arrays have incorrect size
    /// (`n_retrieval_heads * head_dim * low_dim`).
    pub fn from_weights(
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        n_retrieval_heads: usize,
        head_dim: usize,
        low_dim: usize,
    ) -> Self {
        let expected = n_retrieval_heads * head_dim * low_dim;
        assert_eq!(w_q.len(), expected, "w_q length mismatch");
        assert_eq!(w_k.len(), expected, "w_k length mismatch");

        Self {
            w_q,
            w_k,
            n_retrieval_heads,
            head_dim,
            low_dim,
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of retrieval heads.
    pub fn n_retrieval_heads(&self) -> usize {
        self.n_retrieval_heads
    }

    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Low-dimensional projection size.
    pub fn low_dim(&self) -> usize {
        self.low_dim
    }

    /// Get W_Q slice for a specific retrieval head.
    ///
    /// Returns slice of shape `[head_dim * low_dim]` (row-major).
    /// `head_local_idx` is the index within retrieval heads (0-based), not
    /// the global head index.
    ///
    /// # Panics
    ///
    /// Panics if `head_local_idx >= n_retrieval_heads`.
    pub fn w_q_for_head(&self, head_local_idx: usize) -> &[f32] {
        assert!(
            head_local_idx < self.n_retrieval_heads,
            "head_local_idx out of range"
        );
        let start = head_local_idx * self.head_stride();
        &self.w_q[start..start + self.head_stride()]
    }

    /// Get W_K slice for a specific retrieval head.
    ///
    /// # Panics
    ///
    /// Panics if `head_local_idx >= n_retrieval_heads`.
    pub fn w_k_for_head(&self, head_local_idx: usize) -> &[f32] {
        assert!(
            head_local_idx < self.n_retrieval_heads,
            "head_local_idx out of range"
        );
        let start = head_local_idx * self.head_stride();
        &self.w_k[start..start + self.head_stride()]
    }

    // -----------------------------------------------------------------------
    // Projection
    // -----------------------------------------------------------------------

    /// Project a pre-RoPE query vector down to `low_dim` dimensions.
    ///
    /// Computes: `q_proj = q_pre^T @ W_Q` → `[low_dim]`
    ///
    /// # Arguments
    ///
    /// * `head_local_idx` — Index within retrieval heads (0-based).
    /// * `q_pre` — Pre-RoPE query vector `[head_dim]`.
    ///
    /// # Returns
    ///
    /// Projected query vector `[low_dim]`.
    pub fn project_query(&self, head_local_idx: usize, q_pre: &[f32]) -> Vec<f32> {
        assert_eq!(q_pre.len(), self.head_dim, "q_pre dimension mismatch");
        let w = self.w_q_for_head(head_local_idx);
        // q_proj[j] = sum_i q_pre[i] * W[i][j] = sum_i q_pre[i] * w[i * low_dim + j]
        let mut q_proj = vec![0.0f32; self.low_dim];
        #[allow(clippy::needless_range_loop)] // multi-dim indexing: w[i * low_dim + j]
        for i in 0..self.head_dim {
            let qi = q_pre[i];
            let row = i * self.low_dim;
            for j in 0..self.low_dim {
                q_proj[j] += qi * w[row + j];
            }
        }
        q_proj
    }

    /// Project a pre-RoPE query vector into a pre-allocated buffer (zero-alloc).
    ///
    /// Same as [`project_query`](Self::project_query) but writes into `out`
    /// instead of allocating. Use in hot loops like `batch_project_scores`.
    pub fn project_query_into(&self, head_local_idx: usize, q_pre: &[f32], out: &mut [f32]) {
        assert_eq!(q_pre.len(), self.head_dim, "q_pre dimension mismatch");
        assert_eq!(out.len(), self.low_dim, "out dimension mismatch");
        out.fill(0.0);
        let w = self.w_q_for_head(head_local_idx);
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.head_dim {
            let qi = q_pre[i];
            let row = i * self.low_dim;
            for j in 0..self.low_dim {
                out[j] += qi * w[row + j];
            }
        }
    }

    /// Project a pre-RoPE key vector down to `low_dim` dimensions.
    ///
    /// Computes: `k_proj = k_pre^T @ W_K` → `[low_dim]`
    ///
    /// # Arguments
    ///
    /// * `head_local_idx` — Index within retrieval heads (0-based).
    /// * `k_pre` — Pre-RoPE key vector `[head_dim]`.
    ///
    /// # Returns
    ///
    /// Projected key vector `[low_dim]`.
    pub fn project_key(&self, head_local_idx: usize, k_pre: &[f32]) -> Vec<f32> {
        assert_eq!(k_pre.len(), self.head_dim, "k_pre dimension mismatch");
        let w = self.w_k_for_head(head_local_idx);
        let mut k_proj = vec![0.0f32; self.low_dim];
        #[allow(clippy::needless_range_loop)] // multi-dim indexing: w[i * low_dim + j]
        for i in 0..self.head_dim {
            let ki = k_pre[i];
            let row = i * self.low_dim;
            for j in 0..self.low_dim {
                k_proj[j] += ki * w[row + j];
            }
        }
        k_proj
    }

    /// Compute low-dim relevance score for a single query-key pair.
    ///
    /// s(m,n) = (W_Q · q_pre)^T · (W_K · k_pre)
    ///
    /// This is the core scoring function for retrieval head token selection.
    /// Pre-RoPE vectors are projected to `low_dim` dimensions, then their
    /// dot product gives the relevance score.
    ///
    /// Fused to avoid allocating intermediate vectors.
    ///
    /// # Arguments
    ///
    /// * `head_local_idx` — Index within retrieval heads (0-based).
    /// * `q_pre` — Pre-RoPE query vector `[head_dim]`.
    /// * `k_pre` — Pre-RoPE key vector `[head_dim]`.
    ///
    /// # Returns
    ///
    /// Relevance score (scalar).
    pub fn project_score(&self, head_local_idx: usize, q_pre: &[f32], k_pre: &[f32]) -> f32 {
        assert_eq!(q_pre.len(), self.head_dim, "q_pre dimension mismatch");
        assert_eq!(k_pre.len(), self.head_dim, "k_pre dimension mismatch");

        let w_q = self.w_q_for_head(head_local_idx);
        let w_k = self.w_k_for_head(head_local_idx);

        // Fused: sum_j (sum_i q[i]*W_Q[i,j]) * (sum_i k[i]*W_K[i,j])
        let mut score = 0.0f32;
        for j in 0..self.low_dim {
            let mut qj = 0.0f32;
            let mut kj = 0.0f32;
            for i in 0..self.head_dim {
                qj += q_pre[i] * w_q[i * self.low_dim + j];
                kj += k_pre[i] * w_k[i * self.low_dim + j];
            }
            score += qj * kj;
        }
        score
    }

    /// Batch scoring over full KV cache for a single retrieval head.
    ///
    /// Computes relevance scores for all cached keys against a single query:
    ///
    /// ```text
    /// for n in 0..seq_len:
    ///     scores[n] = (W_Q · q_pre)^T · (W_K · k_cache[n])
    /// ```
    ///
    /// Optimized to project the query once, then compute dot products
    /// against projected keys. The inner loops are structured for
    /// auto-vectorization by the compiler (sequential access pattern).
    ///
    /// # Arguments
    ///
    /// * `head_local_idx` — Index within retrieval heads (0-based).
    /// * `q_pre` — Pre-RoPE query vector `[head_dim]`.
    /// * `k_cache` — Pre-RoPE key cache `[seq_len * head_dim]`, row-major
    ///   (one key vector per position).
    ///
    /// # Returns
    ///
    /// Relevance scores `[seq_len]`, one per cached position.
    pub fn batch_project_scores(
        &self,
        head_local_idx: usize,
        q_pre: &[f32],
        k_cache: &[f32],
    ) -> Vec<f32> {
        assert_eq!(q_pre.len(), self.head_dim, "q_pre dimension mismatch");
        let seq_len = k_cache.len() / self.head_dim;
        assert_eq!(
            k_cache.len(),
            seq_len * self.head_dim,
            "k_cache length not divisible by head_dim"
        );

        // Step 1: Project query once into stack buffer (zero-alloc)
        let mut q_proj = [0.0f32; 64]; // low_dim <= 64 for any practical config
        let q_proj = &mut q_proj[..self.low_dim];
        self.project_query_into(head_local_idx, q_pre, q_proj);

        // Step 2: For each key, project into stack buffer and dot with q_proj.
        // Loops are structured so the inner j-loop accesses w_k[row+j] sequentially
        // (row-major), improving cache locality vs the column-major access pattern
        // of the naive transpose.
        let w_k = self.w_k_for_head(head_local_idx);
        let mut scores = vec![0.0f32; seq_len];

        #[allow(clippy::needless_range_loop)] // multi-dim indexing: k_cache[n * head_dim + i]
        for n in 0..seq_len {
            let k_off = n * self.head_dim;
            // Project key into stack buffer
            let mut k_proj = [0.0f32; 64];
            let k_proj = &mut k_proj[..self.low_dim];
            for i in 0..self.head_dim {
                let ki = k_cache[k_off + i];
                let row = i * self.low_dim;
                for j in 0..self.low_dim {
                    k_proj[j] += ki * w_k[row + j]; // sequential access — cache-friendly
                }
            }
            // Dot product with projected query
            let mut score = 0.0f32;
            for j in 0..self.low_dim {
                score += q_proj[j] * k_proj[j];
            }
            scores[n] = score;
        }

        scores
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serialize to JSON string.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    /// Serialize to JSON bytes.
    pub fn to_json_bytes(&self) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    /// Deserialize from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }

    /// Save projection weights to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self
            .to_json_bytes()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load projection weights from a JSON file.
    pub fn load(path: &std::path::Path) -> serde_json::Result<Self> {
        let bytes = std::fs::read(path).map_err(serde_json::Error::io)?;
        Self::from_json_bytes(&bytes)
    }

    /// Validate internal consistency of the projection weights.
    ///
    /// Checks array sizes, dimension constraints, and weight finiteness.
    pub fn validate(&self) -> Result<(), String> {
        let expected = self.total_weights();
        if self.w_q.len() != expected {
            let len = self.w_q.len();
            return Err(format!("w_q length {len}: expected {expected}"));
        }
        if self.w_k.len() != expected {
            let len = self.w_k.len();
            return Err(format!("w_k length {len}: expected {expected}"));
        }
        if self.n_retrieval_heads == 0 {
            return Err("n_retrieval_heads must be > 0".to_string());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".to_string());
        }
        if self.low_dim == 0 {
            return Err("low_dim must be > 0".to_string());
        }
        if self.low_dim > self.head_dim {
            return Err(format!(
                "low_dim ({}) must be <= head_dim ({})",
                self.low_dim, self.head_dim
            ));
        }
        // Check for NaN/Inf in weights
        for (i, &v) in self.w_q.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("w_q[{i}] is not finite: {v}"));
            }
        }
        for (i, &v) in self.w_k.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("w_k[{i}] is not finite: {v}"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_dir() -> PathBuf {
        std::env::temp_dir().join("rt_turbo_projection_test")
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
    }

    // --- Construction Tests ---

    #[test]
    fn test_zeros_all_weights_are_zero() {
        let proj = RetrievalProjection::zeros(3, 64, 16);
        assert_eq!(proj.w_q.len(), 3 * 64 * 16);
        assert_eq!(proj.w_k.len(), 3 * 64 * 16);
        assert!(proj.w_q.iter().all(|&v| v == 0.0));
        assert!(proj.w_k.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_identity_diagonal_ones() {
        let proj = RetrievalProjection::identity(2, 64, 16);
        for h in 0..2 {
            let w_q = proj.w_q_for_head(h);
            let w_k = proj.w_k_for_head(h);
            for i in 0..16 {
                assert_eq!(w_q[i * 16 + i], 1.0, "w_q diagonal at head {h}, row {i}");
                assert_eq!(w_k[i * 16 + i], 1.0, "w_k diagonal at head {h}, row {i}");
            }
            // Count non-zeros: should be exactly low_dim per matrix
            let nonzero_q = w_q.iter().filter(|&&v| v != 0.0).count();
            let nonzero_k = w_k.iter().filter(|&&v| v != 0.0).count();
            assert_eq!(nonzero_q, 16);
            assert_eq!(nonzero_k, 16);
        }
    }

    #[test]
    fn test_xavier_nonzero_and_finite() {
        let proj = RetrievalProjection::xavier(4, 128, 16);
        assert!(proj.w_q.iter().all(|v| v.is_finite()));
        assert!(proj.w_k.iter().all(|v| v.is_finite()));
        assert!(proj.w_q.iter().any(|&v| v != 0.0));
        assert!(proj.w_k.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_from_weights_dimension_check() {
        let w = vec![1.0f32; 128 * 16];
        // Correct size
        let _ = RetrievalProjection::from_weights(w.clone(), w.clone(), 1, 128, 16);
        // Wrong size → panic
        let result = std::panic::catch_unwind(|| {
            RetrievalProjection::from_weights(vec![1.0; 10], vec![1.0; 10], 1, 128, 16);
        });
        assert!(result.is_err());
    }

    // --- Projection Tests ---

    #[test]
    fn test_zeros_projection_uniform_zero_score() {
        let proj = RetrievalProjection::zeros(2, 64, 16);
        let q = vec![1.0f32; 64];
        let k = vec![2.0f32; 64];

        // All projected vectors should be zero
        let q_proj = proj.project_query(0, &q);
        let k_proj = proj.project_key(0, &k);
        assert!(q_proj.iter().all(|&v| v == 0.0));
        assert!(k_proj.iter().all(|&v| v == 0.0));

        // Score should be zero
        let score = proj.project_score(0, &q, &k);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_identity_projection_matches_first_dims() {
        let proj = RetrievalProjection::identity(1, 64, 16);
        let mut q = vec![0.0f32; 64];
        let mut k = vec![0.0f32; 64];

        // Set specific values in first 16 dims
        for i in 0..16 {
            q[i] = (i as f32) * 0.1;
            k[i] = (i as f32) * 0.2;
        }
        // Values in dims 16..64 should be ignored by identity projection
        for i in 16..64 {
            q[i] = 99.0;
            k[i] = 99.0;
        }

        // With identity projection, q_proj should match q[0..16]
        let q_proj = proj.project_query(0, &q);
        let k_proj = proj.project_key(0, &k);

        for i in 0..16 {
            assert!((q_proj[i] - q[i]).abs() < 1e-6, "q_proj[{i}] mismatch");
            assert!((k_proj[i] - k[i]).abs() < 1e-6, "k_proj[{i}] mismatch");
        }

        // Score should be dot product of first 16 dims only
        let expected: f32 = (0..16).map(|i| q[i] * k[i]).sum();
        let score = proj.project_score(0, &q, &k);
        assert!(
            (score - expected).abs() < 1e-4,
            "score mismatch: {score} vs {expected}"
        );
    }

    #[test]
    fn test_projection_symmetric_self_score() {
        // If w_q == w_k and q == k, score should be non-negative (||proj||^2)
        let mut proj = RetrievalProjection::identity(1, 32, 8);
        proj.w_k = proj.w_q.clone();

        let v = vec![1.0f32; 32];
        let score = proj.project_score(0, &v, &v);
        assert!(
            score >= 0.0,
            "self-score should be non-negative, got {score}"
        );
        assert!(
            score > 0.0,
            "self-score should be positive for non-zero input"
        );
    }

    #[test]
    fn test_multi_head_isolation() {
        // Different heads should produce different projections
        let mut proj = RetrievalProjection::zeros(3, 32, 8);
        // Set head 0 to identity, head 1 to zeros, head 2 to 2x identity
        let stride = 32 * 8;
        for i in 0..8 {
            proj.w_q[0 * stride + i * 8 + i] = 1.0;
            proj.w_k[0 * stride + i * 8 + i] = 1.0;

            proj.w_q[2 * stride + i * 8 + i] = 2.0;
            proj.w_k[2 * stride + i * 8 + i] = 2.0;
        }

        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let score_0 = proj.project_score(0, &v, &v);
        let score_1 = proj.project_score(1, &v, &v);
        let score_2 = proj.project_score(2, &v, &v);

        assert!(
            score_0 > 0.0,
            "head 0 (identity) should have positive score"
        );
        assert_eq!(score_1, 0.0, "head 1 (zeros) should have zero score");
        // head 2 has 2x identity: score = (2v)^T (2v) = 4 * (v^T v) = 4 * score_0
        assert!(
            (score_2 - 4.0 * score_0).abs() < 1e-4,
            "head 2 (2x identity) should be 4x head 0: {score_2} vs {}",
            4.0 * score_0
        );
    }

    // --- Batch Scoring Tests ---

    #[test]
    fn test_batch_zeros_all_zero() {
        let proj = RetrievalProjection::zeros(1, 32, 8);
        let q = vec![1.0f32; 32];
        let k_cache = vec![1.0f32; 128 * 32]; // 128 positions
        let scores = proj.batch_project_scores(0, &q, &k_cache);
        assert_eq!(scores.len(), 128);
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_batch_identity_matches_individual() {
        let proj = RetrievalProjection::identity(1, 32, 8);
        let q: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let k_cache: Vec<f32> = (0..64 * 32).map(|idx| ((idx % 32) as f32) * 0.05).collect();

        let batch_scores = proj.batch_project_scores(0, &q, &k_cache);

        // Verify against individual calls
        assert_eq!(batch_scores.len(), 64);
        for n in 0..64 {
            let k = &k_cache[n * 32..(n + 1) * 32];
            let individual = proj.project_score(0, &q, k);
            assert!(
                (batch_scores[n] - individual).abs() < 1e-4,
                "batch[{n}] = {} vs individual = {}",
                batch_scores[n],
                individual
            );
        }
    }

    #[test]
    fn test_batch_dimensionality() {
        let proj = RetrievalProjection::identity(1, 32, 8);
        let q = vec![1.0f32; 32];
        let k_cache = vec![1.0f32; 100 * 32]; // 100 positions
        let scores = proj.batch_project_scores(0, &q, &k_cache);
        assert_eq!(scores.len(), 100);
    }

    #[test]
    fn test_batch_single_key() {
        let proj = RetrievalProjection::identity(1, 16, 4);
        let q = vec![
            1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let k = vec![
            0.5, 1.0, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let scores = proj.batch_project_scores(0, &q, &k);
        assert_eq!(scores.len(), 1);
        // dot product of first 4 dims: 1*0.5 + 2*1 + 3*1.5 + 4*2 = 0.5 + 2 + 4.5 + 8 = 15
        assert!(
            (scores[0] - 15.0).abs() < 1e-4,
            "expected 15.0, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_batch_consistency_with_single_calls() {
        // Comprehensive: xavier weights, verify batch == individual for all positions
        let proj = RetrievalProjection::xavier(1, 16, 4);
        let q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let k_cache: Vec<f32> = (0..32 * 16).map(|i| (i as f32) * 0.07 - 1.0).collect();

        let batch = proj.batch_project_scores(0, &q, &k_cache);
        for n in 0..32 {
            let k = &k_cache[n * 16..(n + 1) * 16];
            let single = proj.project_score(0, &q, k);
            assert!(
                (batch[n] - single).abs() < 1e-3,
                "position {n}: batch={} single={}",
                batch[n],
                single
            );
        }
    }

    // --- Serialization Tests ---

    #[test]
    fn test_json_roundtrip() {
        let proj = RetrievalProjection::xavier(3, 64, 16);
        let json = proj.to_json().unwrap();
        let loaded = RetrievalProjection::from_json(&json).unwrap();

        assert_eq!(loaded.n_retrieval_heads, 3);
        assert_eq!(loaded.head_dim, 64);
        assert_eq!(loaded.low_dim, 16);
        assert_eq!(loaded.w_q, proj.w_q);
        assert_eq!(loaded.w_k, proj.w_k);
    }

    #[test]
    fn test_file_roundtrip() {
        let dir = test_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_projection.json");

        let proj = RetrievalProjection::identity(2, 32, 8);
        proj.save(&path).unwrap();
        let loaded = RetrievalProjection::load(&path).unwrap();

        assert_eq!(loaded.w_q, proj.w_q);
        assert_eq!(loaded.w_k, proj.w_k);
        assert_eq!(loaded.n_retrieval_heads, 2);
        assert_eq!(loaded.head_dim, 32);
        assert_eq!(loaded.low_dim, 8);

        cleanup(&path);
    }

    // --- Validation Tests ---

    #[test]
    fn test_validate_valid() {
        let proj = RetrievalProjection::identity(1, 64, 16);
        assert!(proj.validate().is_ok());
    }

    #[test]
    fn test_validate_catches_nan() {
        let mut proj = RetrievalProjection::identity(1, 64, 16);
        proj.w_q[0] = f32::NAN;
        let err = proj.validate().unwrap_err();
        assert!(err.contains("not finite"));
    }

    #[test]
    fn test_validate_catches_inf() {
        let mut proj = RetrievalProjection::identity(1, 64, 16);
        proj.w_k[42] = f32::INFINITY;
        let err = proj.validate().unwrap_err();
        assert!(err.contains("not finite"));
    }

    #[test]
    fn test_validate_low_dim_exceeds_head_dim() {
        let proj = RetrievalProjection::zeros(1, 8, 16); // low_dim > head_dim
        let err = proj.validate().unwrap_err();
        assert!(err.contains("low_dim"));
    }

    #[test]
    fn test_validate_zero_heads() {
        let proj = RetrievalProjection::zeros(0, 32, 8);
        let err = proj.validate().unwrap_err();
        assert!(err.contains("n_retrieval_heads"));
    }

    // --- Accessor Tests ---

    #[test]
    fn test_accessors() {
        let proj = RetrievalProjection::identity(5, 128, 16);
        assert_eq!(proj.n_retrieval_heads(), 5);
        assert_eq!(proj.head_dim(), 128);
        assert_eq!(proj.low_dim(), 16);
    }

    #[test]
    fn test_w_for_head_out_of_range_panics() {
        let proj = RetrievalProjection::identity(2, 32, 8);
        let result = std::panic::catch_unwind(|| proj.w_q_for_head(5));
        assert!(result.is_err());
    }

    // --- Edge Case Tests ---

    #[test]
    fn test_single_dim_projection() {
        // Edge case: low_dim = 1
        let proj = RetrievalProjection::identity(1, 32, 1);
        let q = vec![3.0f32; 32];
        let k = vec![2.0f32; 32];
        let score = proj.project_score(0, &q, &k);
        // Only first dim passes through: 3.0 * 2.0 = 6.0
        assert!((score - 6.0).abs() < 1e-4, "expected 6.0, got {score}");
    }

    #[test]
    fn test_low_dim_equals_head_dim() {
        // Edge case: low_dim == head_dim → full-rank identity
        let proj = RetrievalProjection::identity(1, 8, 8);
        let q: Vec<f32> = (0..8).map(|i| (i + 1) as f32).collect();
        let k: Vec<f32> = (0..8).map(|i| (i + 1) as f32 * 0.5).collect();
        let score = proj.project_score(0, &q, &k);
        // Full dot product: sum((i+1) * (i+1) * 0.5) = 0.5 * sum((i+1)^2)
        // = 0.5 * (1+4+9+16+25+36+49+64) = 0.5 * 204 = 102
        assert!((score - 102.0).abs() < 1e-2, "expected 102.0, got {score}");
    }

    #[test]
    fn test_large_head_count() {
        // Simulate 32-head model with 5 retrieval heads (15% of 32 ≈ 5)
        let proj = RetrievalProjection::xavier(5, 128, 16);
        assert!(proj.validate().is_ok());
        assert_eq!(proj.w_q.len(), 5 * 128 * 16);
        assert_eq!(proj.w_k.len(), 5 * 128 * 16);

        let q = vec![0.5f32; 128];
        let k = vec![0.3f32; 128];
        for h in 0..5 {
            let score = proj.project_score(h, &q, &k);
            assert!(score.is_finite(), "head {h} score not finite: {score}");
        }
    }

    #[test]
    fn test_batch_large_seq_len() {
        // Simulate 8K context with identity projection
        let proj = RetrievalProjection::identity(1, 64, 16);
        let q = vec![1.0f32; 64];
        let k_cache = vec![0.5f32; 8192 * 64];
        let scores = proj.batch_project_scores(0, &q, &k_cache);
        assert_eq!(scores.len(), 8192);
        // With identity and constant inputs, all scores should be identical
        let first = scores[0];
        assert!(first > 0.0);
        assert!(scores.iter().all(|&s| (s - first).abs() < 1e-4));
    }
}
