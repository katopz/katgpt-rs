//! Reranking module — MaxSim vs Cosine similarity for retrieval reranking.
//!
//! Provides [`rerank`] with pluggable scoring methods and [`ndcg_at`] for
//! retrieval quality evaluation (NDCG@k). Feature-gated behind `maxsim` (Plan 080).

use crate::simd::{maxsim_score, simd_dot_f32};

// ── Types ─────────────────────────────────────────────────────

/// Reranking method for scoring query–document pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankMethod {
    /// Cosine similarity on mean-pooled token embeddings.
    Cosine,
    /// MaxSim late-interaction: `Σ_i max_j dot(q_i, d_j)`.
    MaxSim,
}

/// A document with its reranking score and original index.
#[derive(Debug, Clone)]
pub struct RerankedDoc {
    /// Index into the original `docs` slice.
    pub doc_index: usize,
    /// Computed relevance score (higher = more relevant).
    pub score: f32,
}

// ── Core Functions ────────────────────────────────────────────

/// Rerank documents against a query using the specified scoring method.
///
/// # Arguments
/// - `query` — flat buffer of query tokens `[Lq × dim]`
/// - `docs` — slice of per-document flat buffers, each `[Ld_i × dim]`
/// - `doc_lengths` — number of tokens per document
/// - `dim` — embedding dimension
/// - `method` — [`RerankMethod::Cosine`] or [`RerankMethod::MaxSim`]
///
/// # Returns
/// Documents sorted by score descending.
pub fn rerank(
    query: &[f32],
    docs: &[Vec<f32>],
    doc_lengths: &[usize],
    dim: usize,
    method: RerankMethod,
) -> Vec<RerankedDoc> {
    if dim == 0 || docs.is_empty() {
        return Vec::new();
    }

    let lq = query.len() / dim;

    let mut results: Vec<RerankedDoc> = docs
        .iter()
        .enumerate()
        .map(|(i, doc_data)| {
            let ld = doc_lengths[i];
            let score = match method {
                RerankMethod::Cosine => cosine_rerank_score(query, lq, doc_data, ld, dim),
                RerankMethod::MaxSim => maxsim_score(query, doc_data, lq, ld, dim),
            };
            RerankedDoc {
                doc_index: i,
                score,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain at position k).
///
/// NDCG = DCG@k / IDCG@k, where:
/// - DCG@k = Σ_{i=0}^{k-1} (2^rel_i − 1) / log₂(i + 2)
/// - IDCG@k = DCG@k under ideal (oracle) ranking
///
/// # Arguments
/// - `ranking` — reranked documents (sorted by score, descending)
/// - `ground_truth` — relevance score per document index, i.e. `ground_truth[doc_index]`
/// - `k` — cutoff rank
pub fn ndcg_at(ranking: &[RerankedDoc], ground_truth: &[f32], k: usize) -> f32 {
    let k = k.min(ranking.len());
    if k == 0 {
        return 0.0;
    }

    // DCG@k
    let dcg: f64 = (0..k)
        .map(|i| {
            let rel = ground_truth
                .get(ranking[i].doc_index)
                .copied()
                .unwrap_or(0.0);
            (2.0f64.powf(rel as f64) - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    // IDCG@k: ideal ranking from ground truth, sorted descending.
    let mut ideal_rels: Vec<f64> = ground_truth.iter().map(|&r| r as f64).collect();
    ideal_rels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f64 = (0..k.min(ideal_rels.len()))
        .map(|i| (2.0f64.powf(ideal_rels[i]) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    match idcg > 0.0 {
        true => (dcg / idcg) as f32,
        false => 0.0,
    }
}

// ── Public Similarity Functions ───────────────────────────────

/// Compute mean cosine similarity across all query–document token pairs.
///
/// For each `(q_i, d_j)` pair, computes `cosine = dot(q_i, d_j) / (|q_i| * |d_j|)`,
/// then returns the average over all `lq * ld` pairs.
pub fn cosine_score(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
    if lq == 0 || ld == 0 || dim == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    let mut count = 0usize;

    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let q_norm = simd_dot_f32(q_row, q_row, dim).sqrt();
        if q_norm < 1e-12 {
            continue;
        }
        for j in 0..ld {
            let d_row = &documents[j * dim..(j + 1) * dim];
            let d_norm = simd_dot_f32(d_row, d_row, dim).sqrt();
            if d_norm < 1e-12 {
                continue;
            }
            let dot = simd_dot_f32(q_row, d_row, dim);
            total += dot / (q_norm * d_norm);
            count += 1;
        }
    }

    match count {
        0 => 0.0,
        _ => total / count as f32,
    }
}

/// Compute mean cosine similarity between two multi-vector embeddings.
///
/// Generic version operating on two flat buffers with `la` / `lb` token counts
/// and embedding dimension `dim`. Delegates to [`cosine_score`].
pub fn mean_cosine_similarity(a: &[f32], b: &[f32], la: usize, lb: usize, dim: usize) -> f32 {
    cosine_score(a, b, la, lb, dim)
}

// ── Internal Helpers ──────────────────────────────────────────

/// Cosine similarity between mean-pooled query and mean-pooled document.
fn cosine_rerank_score(query: &[f32], lq: usize, doc: &[f32], ld: usize, dim: usize) -> f32 {
    if ld == 0 || lq == 0 {
        return 0.0;
    }

    // Mean-pool query tokens into `q_mean`.
    let mut q_mean = vec![0.0f32; dim];
    for t in 0..lq {
        let offset = t * dim;
        for d in 0..dim {
            q_mean[d] += query[offset + d];
        }
    }
    let inv_lq = 1.0 / lq as f32;
    for v in q_mean.iter_mut() {
        *v *= inv_lq;
    }

    // Mean-pool document tokens into `d_mean`.
    let mut d_mean = vec![0.0f32; dim];
    for t in 0..ld {
        let offset = t * dim;
        for d in 0..dim {
            d_mean[d] += doc[offset + d];
        }
    }
    let inv_ld = 1.0 / ld as f32;
    for v in d_mean.iter_mut() {
        *v *= inv_ld;
    }

    // Cosine similarity = dot(a, b) / (|a| × |b|)
    let dot = simd_dot_f32(&q_mean, &d_mean, dim);
    let q_norm = simd_dot_f32(&q_mean, &q_mean, dim).sqrt();
    let d_norm = simd_dot_f32(&d_mean, &d_mean, dim).sqrt();

    match q_norm < 1e-12 || d_norm < 1e-12 {
        true => 0.0,
        false => dot / (q_norm * d_norm),
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn rerank_cosine_orders_by_similarity() {
        let dim = 4;
        let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 tokens

        // Doc 0: aligned with both query tokens
        let doc0: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        // Doc 1: orthogonal to both query tokens
        let doc1: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let docs = vec![doc0, doc1];
        let doc_lengths = vec![2, 2];

        let ranked = rerank(&query, &docs, &doc_lengths, dim, RerankMethod::Cosine);
        assert_eq!(
            ranked[0].doc_index, 0,
            "doc0 should rank first (more similar)"
        );
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn rerank_maxsim_orders_by_max_dot() {
        let dim = 4;
        let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 2 tokens

        // Doc 0: strong match with query token 0
        let doc0: Vec<f32> = vec![0.9, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0];
        // Doc 1: weak match with both query tokens
        let doc1: Vec<f32> = vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0];

        let docs = vec![doc0, doc1];
        let doc_lengths = vec![2, 2];

        let ranked = rerank(&query, &docs, &doc_lengths, dim, RerankMethod::MaxSim);
        assert_eq!(ranked[0].doc_index, 0, "doc0 should rank first");
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn rerank_empty_docs_returns_empty() {
        let query = vec![1.0f32, 0.0];
        let ranked = rerank(&query, &[], &[], 2, RerankMethod::MaxSim);
        assert!(ranked.is_empty());
    }

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        let ranking = vec![
            RerankedDoc {
                doc_index: 0,
                score: 3.0,
            },
            RerankedDoc {
                doc_index: 1,
                score: 2.0,
            },
            RerankedDoc {
                doc_index: 2,
                score: 1.0,
            },
        ];
        let ground_truth = vec![3.0, 2.0, 1.0];

        let ndcg = ndcg_at(&ranking, &ground_truth, 3);
        assert!(
            approx_eq(ndcg, 1.0, 1e-6),
            "Perfect ranking should have NDCG=1.0, got {ndcg}"
        );
    }

    #[test]
    fn ndcg_worst_ranking_is_low() {
        let ranking = vec![
            RerankedDoc {
                doc_index: 2,
                score: 0.1,
            },
            RerankedDoc {
                doc_index: 1,
                score: 0.2,
            },
            RerankedDoc {
                doc_index: 0,
                score: 0.3,
            },
        ];
        let ground_truth = vec![3.0, 2.0, 1.0];

        let ndcg = ndcg_at(&ranking, &ground_truth, 3);
        assert!(
            ndcg < 1.0,
            "Worst ranking should have NDCG < 1.0, got {ndcg}"
        );
    }

    #[test]
    fn ndcg_empty_ranking_is_zero() {
        let ranking: Vec<RerankedDoc> = vec![];
        let ground_truth = vec![1.0, 2.0];

        let ndcg = ndcg_at(&ranking, &ground_truth, 5);
        assert!(
            approx_eq(ndcg, 0.0, 1e-6),
            "Empty ranking should have NDCG=0.0, got {ndcg}"
        );
    }

    #[test]
    fn ndcg_k_larger_than_ranking_clamps() {
        let ranking = vec![RerankedDoc {
            doc_index: 0,
            score: 1.0,
        }];
        let ground_truth = vec![1.0];

        let ndcg = ndcg_at(&ranking, &ground_truth, 10);
        assert!(
            approx_eq(ndcg, 1.0, 1e-6),
            "Single perfect doc at k=10 should still be NDCG=1.0, got {ndcg}"
        );
    }

    #[test]
    fn cosine_score_identical_vectors() {
        let dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let d = vec![1.0, 0.0, 0.0, 0.0];
        let score = cosine_score(&q, &d, 1, 1, dim);
        assert!(approx_eq(score, 1.0, 1e-4), "expected 1.0, got {score}");
    }

    #[test]
    fn cosine_score_orthogonal() {
        let dim = 4;
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0, 0.0];
        let score = cosine_score(&q, &d, 1, 1, dim);
        assert!(approx_eq(score, 0.0, 1e-4), "expected 0.0, got {score}");
    }

    #[test]
    fn cosine_score_multi_token_averages() {
        let dim = 2;
        // 2 query tokens, 2 doc tokens
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let d = vec![1.0, 0.0, 0.0, 1.0];
        // (q0,d0)=1.0, (q0,d1)=0.0, (q1,d0)=0.0, (q1,d1)=1.0 → mean = 0.5
        let score = cosine_score(&q, &d, 2, 2, dim);
        assert!(approx_eq(score, 0.5, 1e-4), "expected 0.5, got {score}");
    }

    #[test]
    fn cosine_score_empty_returns_zero() {
        assert!(approx_eq(cosine_score(&[], &[], 0, 0, 4), 0.0, 1e-5));
    }

    #[test]
    fn mean_cosine_matches_cosine_score() {
        let dim = 4;
        let a = vec![1.0, 1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 1.0];
        let cs = cosine_score(&a, &b, 1, 1, dim);
        let mcs = mean_cosine_similarity(&a, &b, 1, 1, dim);
        assert!(
            approx_eq(cs, mcs, 1e-6),
            "mean_cosine_similarity should match cosine_score: {cs} vs {mcs}"
        );
    }
}
