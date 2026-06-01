//! Learned chunk summaries via head_cls vectors.
//!
//! Each KV-head has a learnable `head_cls` query vector used to summarize a
//! chunk of keys via local SDPA: k̄_c = softmax(q̄ · K_chunk / √d) · K_chunk.
//! When `head_cls` is zero-initialized (default), this degenerates to mean
//! pooling for backward compatibility.

// ---------------------------------------------------------------------------
// ChunkSummaryQuery
// ---------------------------------------------------------------------------

/// Per-KV-head learned query for chunk summarization.
///
/// `head_cls` layout: `[n_kv_head * head_dim]`.
/// Zero-initialized by default → mean pooling (backward-compatible).
/// After training, these vectors learn to attend to the most informative
/// positions within each chunk.
#[derive(Clone)]
pub struct ChunkSummaryQuery {
    /// Learned class token embeddings: flat `[n_kv_head * head_dim]`.
    pub head_cls: Vec<f32>,
    pub n_kv_head: usize,
    pub head_dim: usize,
    /// Cached result of scanning `head_cls` for all-zeros.
    /// Updated on construction; call [`recompute_zero_init_cache`] after any
    /// direct mutation of `head_cls`.
    zero_init_cache: bool,
}

impl ChunkSummaryQuery {
    /// Create with zero-initialized `head_cls` (mean-pooling mode).
    pub fn new(n_kv_head: usize, head_dim: usize) -> Self {
        Self {
            head_cls: vec![0.0; n_kv_head * head_dim],
            n_kv_head,
            head_dim,
            zero_init_cache: true,
        }
    }

    /// Create with random initialization for training.
    pub fn new_random(n_kv_head: usize, head_dim: usize, rng: &mut crate::types::Rng) -> Self {
        let scale = (2.0 / head_dim as f32).sqrt();
        let mut head_cls = Vec::with_capacity(n_kv_head * head_dim);
        for _ in 0..n_kv_head * head_dim {
            head_cls.push(rng.normal() * scale);
        }
        Self {
            head_cls,
            n_kv_head,
            head_dim,
            zero_init_cache: false,
        }
    }

    /// Get query slice for a specific head.
    #[inline]
    pub fn head_query(&self, head_idx: usize) -> &[f32] {
        let start = head_idx * self.head_dim;
        &self.head_cls[start..start + self.head_dim]
    }

    /// Check if head_cls is effectively zero (mean-pooling mode).
    ///
    /// Returns a cached bool — O(1) instead of scanning the entire vector.
    /// Call [`recompute_zero_init_cache`] after any direct mutation of `head_cls`.
    #[inline]
    pub fn is_zero_init(&self) -> bool {
        self.zero_init_cache
    }

    /// Recompute the cached `is_zero_init` result after mutating `head_cls`.
    pub fn recompute_zero_init_cache(&mut self) {
        self.zero_init_cache = self.head_cls.iter().all(|&v| v == 0.0);
    }
}

// ---------------------------------------------------------------------------
// ChunkSummaryCache
// ---------------------------------------------------------------------------

/// Cache for completed chunk summaries: `[n_chunks][n_kv_head][head_dim]`.
///
/// Populated during prefill; reused during decode for routing.
#[derive(Clone)]
pub struct ChunkSummaryCache {
    /// Summaries indexed by chunk: `[n_chunks][n_kv_head][head_dim]`.
    pub summaries: Vec<Vec<Vec<f32>>>,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

impl ChunkSummaryCache {
    /// Create an empty cache.
    pub fn new(n_kv_head: usize, head_dim: usize) -> Self {
        Self {
            summaries: Vec::new(),
            n_kv_head,
            head_dim,
        }
    }

    /// Pre-allocate for a known number of chunks.
    pub fn allocate(&mut self, n_chunks: usize) {
        // Reuse existing allocation if already the right size
        if self.summaries.len() == n_chunks {
            // Clear in-place without deallocating
            for chunk in &mut self.summaries {
                for head in chunk.iter_mut() {
                    head.fill(0.0);
                }
            }
        } else {
            self.summaries = (0..n_chunks)
                .map(|_| {
                    (0..self.n_kv_head)
                        .map(|_| vec![0.0; self.head_dim])
                        .collect()
                })
                .collect();
        }
    }

    /// Append a single chunk summary (one entry per KV head).
    pub fn append(&mut self, summary: Vec<Vec<f32>>) {
        debug_assert_eq!(summary.len(), self.n_kv_head);
        for head_summary in &summary {
            debug_assert_eq!(head_summary.len(), self.head_dim);
        }
        self.summaries.push(summary);
    }

    /// View summaries for a specific chunk (all heads).
    pub fn view(&self, chunk_idx: usize) -> &[Vec<f32>] {
        &self.summaries[chunk_idx]
    }

    /// Number of cached chunks.
    pub fn n_chunks(&self) -> usize {
        self.summaries.len()
    }

    /// Clear for a new sequence.
    pub fn reset(&mut self) {
        self.summaries.clear();
    }
}

// ---------------------------------------------------------------------------
// Chunk summarization kernel
// ---------------------------------------------------------------------------

/// Summarize a chunk via local SDPA: k̄_c = softmax(q̄ · K_chunk / √d) · K_chunk.
///
/// At zero-init `head_cls`: returns mean pooling of all keys in the chunk.
///
/// # Arguments
/// * `query` - The chunk summary query holding learned `head_cls` vectors.
/// * `chunk_keys` - Flat key buffer `[chunk_size * head_dim]` for one KV head.
/// * `chunk_size` - Number of tokens in this chunk.
/// * `head_idx` - Which KV head to summarize.
/// * `head_dim` - Dimension per head (must match query and key layout).
///
/// Prefer [`summarize_chunk_into`] on hot paths to avoid per-call allocation.
#[inline]
pub fn summarize_chunk(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; head_dim];
    let mut scores_buf = vec![0.0f32; chunk_size.max(1)];
    summarize_chunk_into(
        query,
        chunk_keys,
        chunk_size,
        head_idx,
        head_dim,
        &mut out,
        &mut scores_buf,
    );
    out
}

/// Zero-alloc variant of [`summarize_chunk`].
///
/// Writes the summary into `out[..head_dim]` and uses `scores_buf` as scratch.
pub fn summarize_chunk_into(
    query: &ChunkSummaryQuery,
    chunk_keys: &[f32],
    chunk_size: usize,
    head_idx: usize,
    head_dim: usize,
    out: &mut [f32],
    scores_buf: &mut [f32],
) {
    let hd = head_dim;
    let q = query.head_query(head_idx);

    // Check if query is zero → mean pooling fallback
    // Use cached result instead of scanning
    if query.is_zero_init() {
        mean_pool_keys_into(chunk_keys, chunk_size, hd, out);
        return;
    }

    // Compute attention scores: q · k_t / sqrt(d)
    let scale = 1.0 / (hd as f32).sqrt();
    debug_assert!(scores_buf.len() >= chunk_size);
    for (t, key_chunk) in chunk_keys.chunks_exact(hd).enumerate() {
        let dot: f32 = q.iter().zip(key_chunk.iter()).map(|(a, b)| a * b).sum();
        scores_buf[t] = dot * scale;
    }
    // Handle remainder if hd doesn't evenly divide
    let remainder_start = chunk_keys.chunks_exact(hd).len() * hd;
    if remainder_start < chunk_keys.len() {
        let t = chunk_keys.chunks_exact(hd).len();
        let mut dot = 0.0f32;
        for d in 0..hd {
            let k_val = if remainder_start + d < chunk_keys.len() {
                chunk_keys[remainder_start + d]
            } else {
                0.0
            };
            dot += q[d] * k_val;
        }
        if t < scores_buf.len() {
            scores_buf[t] = dot * scale;
        }
    }

    // Softmax (numerically stable)
    softmax_inplace(&mut scores_buf[..chunk_size]);

    // Weighted sum of keys → summary
    out[..hd].fill(0.0);
    for (score, key_chunk) in scores_buf[..chunk_size]
        .iter()
        .zip(chunk_keys.chunks_exact(hd))
    {
        for d in 0..hd {
            out[d] += score * key_chunk[d];
        }
    }
}

/// Mean pooling over chunk keys (zero-init fallback).
#[allow(dead_code)]
fn mean_pool_keys(chunk_keys: &[f32], chunk_size: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; head_dim];
    mean_pool_keys_into(chunk_keys, chunk_size, head_dim, &mut out);
    out
}

/// Zero-alloc mean pooling into pre-allocated buffer.
fn mean_pool_keys_into(chunk_keys: &[f32], chunk_size: usize, head_dim: usize, out: &mut [f32]) {
    out[..head_dim].fill(0.0);
    if chunk_size == 0 {
        return;
    }
    // Accumulate all tokens
    for t in 0..chunk_size {
        let k_start = t * head_dim;
        for d in 0..head_dim {
            out[d] += chunk_keys[k_start + d];
        }
    }
    // Scale once at the end
    let inv = 1.0 / chunk_size as f32;
    for d in out[..head_dim].iter_mut() {
        *d *= inv;
    }
}

/// In-place softmax with max subtraction for numerical stability.
fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum_exp += *s;
    }
    if sum_exp > 0.0 {
        for s in scores.iter_mut() {
            *s /= sum_exp;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const N_KV_HEAD: usize = 2;
    const HEAD_DIM: usize = 4;

    #[test]
    fn test_chunk_summary_query_new_zero() {
        let query = ChunkSummaryQuery::new(N_KV_HEAD, HEAD_DIM);
        assert!(query.is_zero_init());
        assert_eq!(query.head_cls.len(), N_KV_HEAD * HEAD_DIM);
        assert_eq!(query.n_kv_head, N_KV_HEAD);
        assert_eq!(query.head_dim, HEAD_DIM);
    }

    #[test]
    fn test_chunk_summary_query_head_slices() {
        let mut query = ChunkSummaryQuery::new(N_KV_HEAD, HEAD_DIM);
        // Write different values per head
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        query.head_cls[HEAD_DIM..2 * HEAD_DIM].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

        let h0 = query.head_query(0);
        assert_eq!(h0, &[1.0, 2.0, 3.0, 4.0]);
        let h1 = query.head_query(1);
        assert_eq!(h1, &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_chunk_summary_cache_allocate() {
        let mut cache = ChunkSummaryCache::new(N_KV_HEAD, HEAD_DIM);
        cache.allocate(3);
        assert_eq!(cache.n_chunks(), 3);
        // Each chunk has n_kv_head entries, each of length head_dim
        for chunk in &cache.summaries {
            assert_eq!(chunk.len(), N_KV_HEAD);
            for head_summary in chunk {
                assert_eq!(head_summary.len(), HEAD_DIM);
                assert!(head_summary.iter().all(|&x| x == 0.0));
            }
        }
    }

    #[test]
    fn test_chunk_summary_cache_append() {
        let mut cache = ChunkSummaryCache::new(N_KV_HEAD, HEAD_DIM);
        let summary = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        cache.append(summary.clone());
        assert_eq!(cache.n_chunks(), 1);
        assert_eq!(cache.view(0)[0], &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_chunk_summary_cache_reset() {
        let mut cache = ChunkSummaryCache::new(N_KV_HEAD, HEAD_DIM);
        cache.append(vec![vec![0.0; HEAD_DIM]; N_KV_HEAD]);
        cache.append(vec![vec![0.0; HEAD_DIM]; N_KV_HEAD]);
        assert_eq!(cache.n_chunks(), 2);
        cache.reset();
        assert_eq!(cache.n_chunks(), 0);
    }

    #[test]
    fn test_summarize_chunk_mean_pool_fallback() {
        let query = ChunkSummaryQuery::new(1, HEAD_DIM);
        // 3 tokens, each with known keys
        let chunk_keys: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 2.0, 0.0, 0.0, // token 1
            0.0, 0.0, 3.0, 0.0, // token 2
        ];
        let summary = summarize_chunk(&query, &chunk_keys, 3, 0, HEAD_DIM);
        // Mean of the 3 vectors: [1/3, 2/3, 1.0, 0.0]
        let expected = [1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0];
        for (i, (&got, &exp)) in summary.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "mean pool mismatch at dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_summarize_chunk_with_learned_query() {
        let mut query = ChunkSummaryQuery::new(1, HEAD_DIM);
        // Use large magnitude so softmax concentrates sharply on matching token.
        // [0, 100, 0, 0] · token 1 [0, 2, 0, 0] = 200 (dominant).
        query.head_cls[0..HEAD_DIM].copy_from_slice(&[0.0, 100.0, 0.0, 0.0]);
        query.recompute_zero_init_cache();

        let chunk_keys: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // token 0
            0.0, 2.0, 0.0, 0.0, // token 1
            0.0, 0.0, 3.0, 0.0, // token 2
        ];
        let summary = summarize_chunk(&query, &chunk_keys, 3, 0, HEAD_DIM);

        // Attention should heavily concentrate on token 1 → summary ≈ [0, 2, 0, 0]
        let expected = [0.0, 2.0, 0.0, 0.0];
        for (i, (&got, &exp)) in summary.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-2,
                "learned query mismatch at dim {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_summarize_chunk_single_token() {
        let query = ChunkSummaryQuery::new(1, HEAD_DIM);
        let chunk_keys: Vec<f32> = vec![4.0, 5.0, 6.0, 7.0];
        let summary = summarize_chunk(&query, &chunk_keys, 1, 0, HEAD_DIM);
        // Single token mean pool = the token itself
        assert_eq!(summary, &[4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_mean_pool_keys_empty_chunk() {
        let result = mean_pool_keys(&[], 0, 4);
        assert_eq!(result, &[0.0; 4]);
    }

    #[test]
    fn test_softmax_inplace_uniform() {
        let mut scores = vec![1.0, 1.0, 1.0];
        softmax_inplace(&mut scores);
        // All equal → uniform 1/3
        for &s in &scores {
            assert!((s - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {s}");
        }
    }

    #[test]
    fn test_softmax_inplace_peaked() {
        let mut scores = vec![10.0, 0.0, 0.0];
        softmax_inplace(&mut scores);
        let dominant = scores[0];
        assert!(
            dominant > 0.99,
            "expected near-1.0 for dominant, got {dominant}"
        );
        assert!(scores[1] < 0.01);
        assert!(scores[2] < 0.01);
        let sum: f32 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax must sum to 1.0, got {sum}"
        );
    }
}
