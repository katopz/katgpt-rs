//! Reranking module — MaxSim vs Cosine similarity for retrieval reranking.
//!
//! Provides [`rerank`] with pluggable scoring methods and [`ndcg_at`] / [`ndcg_at_into`]
//! for retrieval quality evaluation (NDCG@k). Feature-gated behind `maxsim` (Plan 080).
//!
//! ## Deep Manifold: Symmetric Boundary Pair (Plan 085)
//!
//! [`SymmetricBoundaryPair`] provides symmetric boundary conditions for BT ranking,
//! based on Deep Manifold Part 2 (arXiv:2512.06563, §2.6.2).
//! Feature-gated behind `bt_rank`.

use crate::simd::{maxsim_score, simd_add_inplace, simd_dot_f32, simd_scale_inplace};

// ── Types ─────────────────────────────────────────────────────

/// Reranking method for scoring query–document pairs.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankMethod {
    /// Cosine similarity on mean-pooled token embeddings.
    Cosine,
    /// MaxSim late-interaction: `Σ_i max_j dot(q_i, d_j)`.
    MaxSim,
}

/// A document with its reranking score and original index.
///
/// Field order: usize (8-byte aligned) before f32 (4-byte aligned)
/// eliminates 4 bytes of padding on 64-bit targets.
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

    // Pre-allocate scratch buffers for cosine rerank (avoids per-doc allocation).
    // MaxSim path doesn't need these — skip 2 heap allocations.
    let (mut q_mean, mut d_mean) = match method {
        RerankMethod::Cosine => (vec![0.0f32; dim], vec![0.0f32; dim]),
        RerankMethod::MaxSim => (Vec::new(), Vec::new()),
    };

    // Cosine path: mean-pool the query ONCE and reuse across all docs.
    //
    // Before this hoist, `cosine_rerank_score_into` recomputed `q_mean` (O(lq*dim)
    // adds + O(dim) scalar muls) and `q_norm` (O(dim) dot + sqrt) for every doc,
    // despite both being query-invariant. For N docs that was N× redundant work.
    //
    // We pre-compute `q_mean` and `q_norm` here and pass them down via a small
    // helper closure so the per-doc inner loop only touches the document.
    let q_norm_precalc: f32 = if matches!(method, RerankMethod::Cosine) && lq > 0 {
        q_mean[..dim].fill(0.0);
        for t in 0..lq {
            let offset = t * dim;
            simd_add_inplace(&mut q_mean[..dim], &query[offset..offset + dim]);
        }
        let inv_lq = 1.0 / lq as f32;
        simd_scale_inplace(&mut q_mean[..dim], inv_lq);
        simd_dot_f32(&q_mean[..dim], &q_mean[..dim], dim).sqrt()
    } else {
        0.0
    };

    let mut results: Vec<RerankedDoc> = docs
        .iter()
        .enumerate()
        .map(|(i, doc_data)| {
            let ld = doc_lengths[i];
            let score = match method {
                RerankMethod::Cosine => cosine_rerank_score_doc_only(
                    &q_mean[..dim],
                    q_norm_precalc,
                    doc_data,
                    ld,
                    dim,
                    &mut d_mean,
                ),
                RerankMethod::MaxSim => maxsim_score(query, doc_data, lq, ld, dim),
            };
            RerankedDoc {
                doc_index: i,
                score,
            }
        })
        .collect();

    // sort_unstable_by is faster than sort_by for ranking — doesn't preserve
    // equal-element order (fine for ranking) and avoids O(n) worst-case merges.
    results.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results
}

/// Zero-alloc variant of [`ndcg_at`].
///
/// `ideal_rels_buf` must have length `>= ground_truth.len()`.
/// The buffer is used as scratch space for sorting ideal relevance scores.
pub fn ndcg_at_into(
    ranking: &[RerankedDoc],
    ground_truth: &[f32],
    k: usize,
    ideal_rels_buf: &mut [f64],
) -> f32 {
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
    let n = ground_truth.len().min(ideal_rels_buf.len());
    for i in 0..n {
        ideal_rels_buf[i] = ground_truth[i] as f64;
    }
    ideal_rels_buf[..n]
        .sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f64 = (0..k.min(n))
        .map(|i| (2.0f64.powf(ideal_rels_buf[i]) - 1.0) / (i as f64 + 2.0).log2())
        .sum();

    match idcg > 0.0 {
        true => (dcg / idcg) as f32,
        false => 0.0,
    }
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain at position k).
///
/// Allocating wrapper — prefer [`ndcg_at_into`] in hot paths.
pub fn ndcg_at(ranking: &[RerankedDoc], ground_truth: &[f32], k: usize) -> f32 {
    let mut ideal_rels = vec![0.0f64; ground_truth.len()];
    ndcg_at_into(ranking, ground_truth, k, &mut ideal_rels)
}

// ── Public Similarity Functions ───────────────────────────────

/// Compute mean cosine similarity across all query–document token pairs.
///
/// For each `(q_i, d_j)` pair, computes `cosine = dot(q_i, d_j) / (|q_i| * |d_j|)`,
/// then returns the average over all `lq * ld` pairs.
///
/// Pre-computes document norms into `d_norms` to avoid O(lq*ld) redundant computation.
/// `d_norms` must have length `>= ld`.
pub fn cosine_score_into(
    queries: &[f32],
    documents: &[f32],
    lq: usize,
    ld: usize,
    dim: usize,
    d_norms: &mut [f32],
) -> f32 {
    if lq == 0 || ld == 0 || dim == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    let mut count = 0usize;

    // Pre-compute document norms once into caller-provided buffer
    for j in 0..ld {
        let d_row = &documents[j * dim..(j + 1) * dim];
        d_norms[j] = simd_dot_f32(d_row, d_row, dim).sqrt();
    }

    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let q_norm = simd_dot_f32(q_row, q_row, dim).sqrt();
        if q_norm < 1e-12 {
            continue;
        }
        // Hoist the query-side reciprocal out of the document loop — one
        // division per query instead of one per (query, document) pair.
        let inv_q_norm = 1.0 / q_norm;
        for j in 0..ld {
            let d_row = &documents[j * dim..(j + 1) * dim];
            let d_norm = d_norms[j];
            if d_norm < 1e-12 {
                continue;
            }
            let dot = simd_dot_f32(q_row, d_row, dim);
            let inv_d_norm = 1.0 / d_norm;
            total += dot * (inv_q_norm * inv_d_norm);
            count += 1;
        }
    }

    match count {
        0 => 0.0,
        _ => total / count as f32,
    }
}

/// Allocating wrapper — prefer `cosine_score_into` in hot paths.
pub fn cosine_score(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
    let mut d_norms = vec![0.0f32; ld];
    cosine_score_into(queries, documents, lq, ld, dim, &mut d_norms)
}

/// Compute mean cosine similarity between two multi-vector embeddings.
///
/// Generic version operating on two flat buffers with `la` / `lb` token counts
/// and embedding dimension `dim`. Delegates to [`cosine_score`].
pub fn mean_cosine_similarity(a: &[f32], b: &[f32], la: usize, lb: usize, dim: usize) -> f32 {
    cosine_score(a, b, la, lb, dim)
}

// ── Internal Helpers ──────────────────────────────────────────

/// Per-document cosine score for reranking.
///
/// The query mean (`q_mean`) and its L2 norm (`q_norm`) are query-invariant
/// and are computed ONCE per query by the caller (see [`rerank`]). This
/// function only mean-pools the document and returns `dot(q_mean, d_mean) /
/// (q_norm × d_norm)`.
///
/// `d_mean` is caller-owned scratch (length `>= dim`). It's filled each call.
#[inline]
fn cosine_rerank_score_doc_only(
    q_mean: &[f32],
    q_norm: f32,
    doc: &[f32],
    ld: usize,
    dim: usize,
    d_mean: &mut [f32],
) -> f32 {
    if ld == 0 || q_norm < 1e-12 {
        return 0.0;
    }

    // Mean-pool document tokens into `d_mean`.
    d_mean[..dim].fill(0.0);
    for t in 0..ld {
        let offset = t * dim;
        simd_add_inplace(&mut d_mean[..dim], &doc[offset..offset + dim]);
    }
    let inv_ld = 1.0 / ld as f32;
    simd_scale_inplace(&mut d_mean[..dim], inv_ld);

    let dot = simd_dot_f32(q_mean, &d_mean[..dim], dim);
    let d_norm = simd_dot_f32(&d_mean[..dim], &d_mean[..dim], dim).sqrt();

    if d_norm < 1e-12 {
        0.0
    } else {
        dot / (q_norm * d_norm)
    }
}

// ── Deep Manifold: Symmetric Boundary Pair (Plan 085) ─────────

/// Deep Manifold §2.6.2: Symmetric boundary pair for BT ranking.
///
/// When fixed-point location is unknown, symmetric boundaries
/// (positive attraction + negative repulsion) produce the narrowest
/// convergence corridor. BT pairwise ranking IS symmetric boundary
/// condition application.
///
/// # Feature Gate
///
/// Behind `bt_rank` — same gate as the BT ranking infrastructure.
#[cfg(feature = "bt_rank")]
#[derive(Debug, Clone, Copy)]
pub struct SymmetricBoundaryPair {
    /// Positive (chosen) boundary strength.
    pub attraction: f32,
    /// Negative (rejected) boundary strength.
    pub repulsion: f32,
}

#[cfg(feature = "bt_rank")]
impl SymmetricBoundaryPair {
    /// Create a new symmetric boundary pair.
    pub fn new(attraction: f32, repulsion: f32) -> Self {
        Self {
            attraction,
            repulsion,
        }
    }

    /// Paper Eq. 73: symmetric contrastive boundary strength.
    ///
    /// Higher = more symmetric = better convergence corridor.
    /// Returns 0.0 if both boundaries are zero (no signal).
    pub fn symmetry(&self) -> f32 {
        let sum = self.attraction + self.repulsion;
        if sum < 1e-8 {
            return 0.0;
        }
        1.0 - (self.attraction - self.repulsion).abs() / sum
    }

    /// Adaptive β based on boundary quality.
    ///
    /// More symmetric pairs → higher β → stronger boundary enforcement.
    /// Asymmetric pairs → lower β → weaker enforcement (trust the signal less).
    pub fn adaptive_beta(&self, base_beta: f32) -> f32 {
        base_beta * (0.5 + 0.5 * self.symmetry())
    }

    /// Check if the boundary pair is well-formed (both positive).
    pub fn is_valid(&self) -> bool {
        self.attraction >= 0.0 && self.repulsion >= 0.0
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
