//! MaxSim scoring — ColBERT-style max-similarity for late-interaction
//! retrieval (Plan 157 sigmoid_margin feature).

// ── MaxSim Late-Interaction Scoring (Research 45, Plan 080) ────

/// Memory-efficient MaxSim scoring: `score = Σ_i max_j dot(q_i, d_j)`.
///
/// Late-interaction relevance score (ColBERT/PyLate style) computed without
/// materializing the `[Lq × Ld]` similarity matrix. Each query token's max
/// similarity across all doc tokens is found via running max, then summed.
///
/// This is the core scoring primitive distilled from erikkaum/maxsim (Research 45).
/// The Metal kernel achieves 3-4× speedup over naive by streaming over doc tokens
/// with a running max in shared memory — same O(Lq × Ld × dim) work, but with
/// better cache locality and zero intermediate allocation.
///
/// Our CPU version composes existing `simd_dot_f32` + inline running max.
/// The algorithm is provably equivalent to:
/// ```text
/// let mut sim = vec![0.0f32; lq * ld];
/// for i in 0..lq {
///     for j in 0..ld {
///         sim[i * ld + j] = dot(q[i], d[j]);
///     }
/// }
/// let score: f32 = (0..lq).map(|i| sim[i*ld..(i+1)*ld].iter().copied().fold(f32::NEG_INFINITY, f32::max)).sum();
/// ```
/// But without the `lq × ld` allocation.
///
/// # Arguments
/// - `queries`:   `[Lq, dim]` row-major f32
/// - `documents`: `[Ld, dim]` row-major f32
/// - `lq`:        number of query tokens
/// - `ld`:        number of document tokens
/// - `dim`:       embedding dimension (e.g. 64, 128)
///
/// # Returns
/// Scalar score (fp32 accumulated, matching Metal kernel design).
///
/// # Feature flag
/// `maxsim` — Plan 080
///
/// # GOAT proof (Plan 080 T2)
/// Must match naive materialized result within 1e-6.
/// Must be ≥2× faster than naive for Lq≥32, Ld≥128, dim=128.
#[cfg(feature = "maxsim")]
use super::simd_dot_f32;

#[cfg(feature = "maxsim")]
#[inline]
pub fn maxsim_score(queries: &[f32], documents: &[f32], lq: usize, ld: usize, dim: usize) -> f32 {
    debug_assert!(
        queries.len() >= lq * dim,
        "maxsim_score: queries buffer too small: need {lq}*{dim}={}, have {}",
        lq * dim,
        queries.len()
    );
    debug_assert!(
        documents.len() >= ld * dim,
        "maxsim_score: documents buffer too small: need {ld}*{dim}={}, have {}",
        ld * dim,
        documents.len()
    );

    if ld == 0 {
        return 0.0;
    }

    let mut score = 0.0f32;
    for i in 0..lq {
        let q_row = &queries[i * dim..(i + 1) * dim];
        let mut my_max = f32::NEG_INFINITY;
        for j in 0..ld {
            let d_row = &documents[j * dim..(j + 1) * dim];
            let dot = simd_dot_f32(q_row, d_row, dim);
            my_max = my_max.max(dot);
        }
        score += my_max;
    }
    score
}

/// Packed/ragged MaxSim scoring: score N (query, doc) pairs with offset arrays.
///
/// Matches the Metal kernel's canonical API (maxsim README "Packed (ragged segments)").
/// Each pair (pair_q_ids[i], pair_d_ids[i]) gets scored independently.
///
/// # Arguments
/// - `queries`:        flat buffer, query_offsets[i]..query_offsets[i+1] is `[dim]`
/// - `query_offsets`:  [num_queries + 1] prefix-sum offsets
/// - `documents`:      flat buffer, doc_offsets[i]..doc_offsets[i+1] is `[dim]`
/// - `doc_offsets`:    [num_docs + 1] prefix-sum offsets
/// - `pair_q_ids`:     query index for each pair
/// - `pair_d_ids`:     doc index for each pair
/// - `dim`:            embedding dimension
///
/// - `results`:        output buffer, must have length >= num_pairs
///
/// # Feature flag
/// `maxsim` — Plan 080
#[cfg(feature = "maxsim")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn maxsim_score_packed(
    queries: &[f32],
    query_offsets: &[usize],
    documents: &[f32],
    doc_offsets: &[usize],
    pair_q_ids: &[usize],
    pair_d_ids: &[usize],
    dim: usize,
    results: &mut [f32],
) {
    let num_pairs = pair_q_ids.len();
    debug_assert_eq!(pair_d_ids.len(), num_pairs);
    debug_assert!(results.len() >= num_pairs, "results buffer too short");
    // Compute max indices in one pass instead of two .iter().max() scans
    let mut max_q_id = 0usize;
    let mut max_d_id = 0usize;
    for p in 0..num_pairs {
        max_q_id = max_q_id.max(pair_q_ids[p]);
        max_d_id = max_d_id.max(pair_d_ids[p]);
    }
    debug_assert!(query_offsets.len() >= max_q_id.saturating_add(2));
    debug_assert!(doc_offsets.len() >= max_d_id.saturating_add(2));

    for p in 0..num_pairs {
        let q_id = pair_q_ids[p];
        let d_id = pair_d_ids[p];
        let q_start = query_offsets[q_id];
        let q_end = query_offsets[q_id + 1];
        let d_start = doc_offsets[d_id];
        let d_end = doc_offsets[d_id + 1];
        let q_data = &queries[q_start..q_end];
        let d_data = &documents[d_start..d_end];
        let lq = q_data.len() / dim;
        let ld = d_data.len() / dim;
        results[p] = maxsim_score(q_data, d_data, lq, ld, dim);
    }
}
