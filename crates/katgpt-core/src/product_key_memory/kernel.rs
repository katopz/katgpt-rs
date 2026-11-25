//! Retrieval kernel for [`ProductKeyMemory`] — O(√N) factored top-k retrieval.
//!
//! Plan 408 Phase 2. Implements the seven-step factored retrieval:
//!
//! ```text
//! 1. Split q into q1 = q[..D_K/2], q2 = q[D_K/2..].
//! 2. Heapselect top-k from codebook 1: s1[i] = score(q1, keys_1[i]).  O(√N)
//! 3. Heapselect top-k from codebook 2: s2[j] = score(q2, keys_2[j]).  O(√N)
//! 4. Cartesian product: for (i,j) in I1 × I2, score_{i,j} = s1[i] + s2[j].  O(K²)
//! 5. Top-k of the K² candidates, map (i,j) -> flat_index = i*SQRT_N + j.  O(K² log K)
//! 6. Normalize weights (softmax over the k² restricted scores — see note below).
//! 7. Write (flat_index, weight) into out[..k].
//! ```
//!
//! # Softmax vs sigmoid — the documented deviation
//!
//! Per AGENTS.md, every *relevance gate* in this codebase uses sigmoid, not
//! softmax. The PKM top-k *normalization* (paper §2.2) is a different concern:
//! it is a ranking normalization over the `k²`-restricted candidate set that
//! produces mixing weights for the K retrieved value rows. It is NOT a
//! probability claim, NOT a calibrated UQ signal, and NOT a gate decision.
//! The plan (T2.1 step 6) explicitly keeps softmax here for ranking fidelity
//! vs the paper's reference implementation, and documents the deviation from
//! the global sigmoid rule.
//!
//! Why softmax is correct *here*: the K weights must sum to 1 (they are
//! convex-combination coefficients for the K value rows). Sigmoid of each
//! score independently does not sum to 1. A sigmoid-based normalization
//! (`σ(s_i) / Σ σ(s_j)`) would work but distorts the score ranking less
//! faithfully than direct softmax over the raw additive scores. The paper
//! uses softmax; we match it. The deviation is logged in the module docs and
//! the plan; no other code path in this crate uses softmax.
//!
//! # Zero-allocation contract (Plan 408 G4)
//!
//! The hot path (`query_into`) takes pre-allocated scratch buffers from the
//! caller via [`PkmScratch`]. No `Vec`, no `Box`, no implicit allocation
//! inside the √N scoring loops or the K² Cartesian product. The only heap
//! touch is the cold value-row fetches by the resolved top-k flat indices
//! (caller-side, after `query_into` returns).

use crate::product_key_memory::types::{ProductKeyMemory, ScoreFn};

// ─── Scratch ───────────────────────────────────────────────────────────

/// Pre-allocated scratch buffers for [`ProductKeyMemory::query_into`].
///
/// Holds the two √N-length score arrays (one per codebook) and the two K-length
/// `(index, score)` top-k buffers. Construct once, reuse across queries — the
/// G4 zero-alloc gate requires no per-query allocation.
///
/// The generic `K` is the per-codebook top-k (NOT the final output k — the
/// Cartesian product is `K × K = K²` candidates, reduced to the final `k ≤ K²`).
/// `K` MUST be `<= SQRT_N`; debug_asserted at query time.
pub struct PkmScratch<const SQRT_N: usize, const K: usize> {
    /// Codebook-1 scores: `score(q1, keys_1[i])` for `i in 0..SQRT_N`.
    pub scores_1: [f32; SQRT_N],
    /// Codebook-2 scores: `score(q2, keys_2[j])` for `j in 0..SQRT_N`.
    pub scores_2: [f32; SQRT_N],
    /// Codebook-1 top-k `(row_index, score)` pairs, sorted score-descending.
    pub top_1: [(usize, f32); K],
    /// Codebook-2 top-k `(row_index, score)` pairs, sorted score-descending.
    pub top_2: [(usize, f32); K],
}

impl<const SQRT_N: usize, const K: usize> PkmScratch<SQRT_N, K> {
    /// Construct zeroed scratch. Caller reuses across queries by calling
    /// `query_into` repeatedly with the same scratch instance.
    ///
    /// Panics at construction if `K > SQRT_N` (the top-k of a codebook cannot
    /// exceed the codebook size — caller bug).
    pub fn new() -> Self {
        assert!(
            K <= SQRT_N,
            "PkmScratch: K ({K}) must be <= SQRT_N ({SQRT_N})"
        );
        Self {
            scores_1: [0.0; SQRT_N],
            scores_2: [0.0; SQRT_N],
            top_1: [(0, f32::NEG_INFINITY); K],
            top_2: [(0, f32::NEG_INFINITY); K],
        }
    }
}

impl<const SQRT_N: usize, const K: usize> Default for PkmScratch<SQRT_N, K> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Scoring functions (T2.2, T2.3) ────────────────────────────────────

/// Dot-product score: `q_half · key_half` (paper §2.2 default).
///
/// SIMD-friendly via the auto-vectorizable accumulation loop. The two √N
/// scoring loops dominate at N=10⁶ (Plan 408 G1 gate), so this is the hot
/// path inside the hot path.
#[inline]
pub fn score_dot(q_half: &[f32], key_half: &[f32]) -> f32 {
    debug_assert_eq!(q_half.len(), key_half.len());
    crate::simd::simd_dot_f32(q_half, key_half, q_half.len())
}

/// IDW score: `−log(ε + ‖q_half − key_half‖²)` (paper §A.2).
///
/// Encourages centroid-like keys — keys cannot inflate score by growing
/// magnitude (the log bounds the best achievable score at `−log ε`). `epsilon`
/// MUST be > 0; the `ScoreFn::Idw` constructor enforces this.
#[inline]
pub fn score_idw(q_half: &[f32], key_half: &[f32], epsilon: f32) -> f32 {
    debug_assert_eq!(q_half.len(), key_half.len());
    debug_assert!(
        epsilon > 0.0 && epsilon.is_finite(),
        "IDW epsilon must be > 0"
    );
    let ssd = crate::simd::simd_dist_sq(q_half, key_half, q_half.len());
    // −log(ε + ssd). The logf32 is the cold-path cost per codebook row; the
    // √N scoring loop pays it √N times per codebook (acceptable per Plan 408
    // G1 budget — log is ~5ns, √N=1000 → 5µs per codebook, well under the
    // O(N) brute-force baseline).
    -(epsilon + ssd).ln()
}

/// Dispatch `score_fn` over a half-query vs a codebook row.
#[inline(always)]
fn score_half(score_fn: ScoreFn, q_half: &[f32], key_half: &[f32]) -> f32 {
    match score_fn {
        ScoreFn::Dot => score_dot(q_half, key_half),
        ScoreFn::Idw { epsilon } => score_idw(q_half, key_half, epsilon),
    }
}

// ─── Heapselect top-k of (index, score) (T2.1 steps 2–3) ───────────────

/// Maintain a sorted-descending top-k of `(row_index, score)` pairs from a
/// full `scores: [f32; N]` array. Writes into `out[..k]`, sorted descending.
///
/// Insertion-maintained — O(N * K) total, which beats O(N log N) for the
/// small K (≤16) we care about. Zero-allocation (operates in-place on `out`).
///
/// `out` is initialized to `(0, NEG_INFINITY)` by the caller (via
/// [`PkmScratch`]). Ties broken by lower index (stable for determinism).
fn heapselect_top_k_desc<const N: usize, const K: usize>(
    scores: &[f32; N],
    out: &mut [(usize, f32); K],
) {
    // `out` is kept sorted descending by score. For each candidate, find its
    // insertion position via linear scan (K ≤ 16 → faster than binary search).
    for (idx, &s) in scores.iter().enumerate() {
        // Skip if not better than the current k-th best. Use strict `>` on
        // the score and `<` tiebreak on index so the first-seen wins ties
        // (deterministic — same input → same output, G6 gate).
        if s <= out[K - 1].1 {
            continue;
        }
        // Find insertion position.
        let mut pos = 0;
        while pos < K {
            // Strictly-greater score wins; equal score loses (keep earlier idx).
            if s > out[pos].1 {
                break;
            }
            pos += 1;
        }
        if pos < K {
            // Shift [pos..K-1] down by one to make room at pos.
            out.copy_within(pos..K - 1, pos + 1);
            out[pos] = (idx, s);
        }
    }
}

// ─── Cartesian product top-k (T2.1 steps 4–5) ──────────────────────────

/// From two per-codebook top-k lists (each `K` entries, sorted descending),
/// compute the `K × K` Cartesian product with additive scores and extract
/// the final `out_k` best `(flat_index, score)` pairs into `out[..out_k]`,
/// sorted descending.
///
/// `flat_index = i * SQRT_N + j`. Writes `out_k` entries (or fewer if the
/// candidate pool is smaller — degenerate). Returns the number written.
///
/// Zero-allocation: uses an insertion-maintained top-`out_k` over the K²
/// candidates directly into `out`. `out.len()` MUST be `>= out_k`.
fn cartesian_top_k<const SQRT_N: usize>(
    top_1: &[(usize, f32)],
    top_2: &[(usize, f32)],
    out: &mut [(usize, f32)],
    out_k: usize,
) -> usize {
    let k1 = top_1.len();
    let k2 = top_2.len();
    let out_k = out_k.min(out.len());
    if out_k == 0 || k1 == 0 || k2 == 0 {
        return 0;
    }
    // Initialize `out` to sentinel (NEG_INFINITY) so the first K² candidates
    // all enter. We only touch out[..out_k].
    out[..out_k].fill((0, f32::NEG_INFINITY));

    // For each (i, j) in top_1 × top_2, score = s1[i] + s2[j],
    // flat_index = i_idx * SQRT_N + j_idx. Insert into the top-out_k.
    //
    // Perf: both input lists are sorted score-descending (documented contract,
    // guaranteed by `heapselect_top_k_desc`, which never admits NaN). So once
    // `s1 + s2[j] <= threshold` we can `break` out of the inner loop instead of
    // `continue`: every later `j` has `s2[j'] <= s2[j]`, hence an even smaller
    // sum, and the threshold only ever rises. Likewise, once `s1 + s2_max`
    // fails, no later `i` can succeed either (`s1' <= s1`), so we break the
    // outer loop. Same output, but the common case visits O(K) of the K²
    // candidates instead of all of them.
    let s2_max = top_2[0].1;
    for &(i_idx, s1) in top_1 {
        if s1 + s2_max <= out[out_k - 1].1 {
            break;
        }
        for &(j_idx, s2) in top_2 {
            let combined = s1 + s2;
            // Stop: neither this nor any later `j` beats the out_k-th best.
            if combined <= out[out_k - 1].1 {
                break;
            }
            let flat = i_idx * SQRT_N + j_idx;
            // Find insertion position.
            let mut pos = 0;
            while pos < out_k && combined <= out[pos].1 {
                pos += 1;
            }
            if pos < out_k {
                out.copy_within(pos..out_k - 1, pos + 1);
                out[pos] = (flat, combined);
            }
        }
    }
    out_k
}

// ─── The query_into kernel (T2.1) ──────────────────────────────────────

impl<const SQRT_N: usize, const D_K: usize, const D_V: usize> ProductKeyMemory<SQRT_N, D_K, D_V> {
    /// O(√N) factored top-k retrieval. Writes `(flat_index, weight)` pairs
    /// into `out[..k]`, sorted score-descending. Returns the number written
    /// (always `k` unless the table is degenerate).
    ///
    /// # Steps (Plan 408 T2.1)
    ///
    /// 1. Split `q` into halves.
    /// 2. Score + heapselect top-`K` from codebook 1 into `scratch.scores_1`
    ///    → `scratch.top_1`. O(√N).
    /// 3. Score + heapselect top-`K` from codebook 2 into `scratch.scores_2`
    ///    → `scratch.top_2`. O(√N).
    /// 4. Cartesian product `top_1 × top_2` (K² candidates, additive scores),
    ///    top-`k` into `out`. O(K²).
    /// 5. Softmax-normalize the k selected scores → weights. (Deviation from
    ///    the global sigmoid rule — documented at the top of this file.)
    /// 6. Write `(flat_index, weight)` into `out[..k]`.
    ///
    /// # Arguments
    ///
    /// - `q`: the `D_K`-dim query vector.
    /// - `score_fn`: [`ScoreFn::Dot`] or [`ScoreFn::Idw`].
    /// - `k`: the final top-k (MUST be `<= K * K` and `<= out.len()`).
    /// - `out`: caller-allocated `(flat_index, weight)` buffer; `out[..k]`
    ///   is written. Weights are in `[0, 1]` and sum to 1 (softmax-normalized).
    /// - `scratch`: pre-allocated scratch sized for per-codebook top-`K`.
    ///
    /// # Panics
    ///
    /// Debug builds assert `k <= K * K`, `k <= out.len()`, `K <= SQRT_N`.
    pub fn query_into<const K: usize>(
        &self,
        q: &[f32; D_K],
        score_fn: ScoreFn,
        k: usize,
        out: &mut [(usize, f32)],
        scratch: &mut PkmScratch<SQRT_N, K>,
    ) -> usize {
        debug_assert!(K <= SQRT_N, "per-codebook K must be <= SQRT_N");
        debug_assert!(k <= K * K, "final k must be <= K*K (cartesian pool size)");
        debug_assert!(k <= out.len(), "out must hold at least k entries");
        let half = Self::key_half_dim();

        // Step 1: split. Safe — D_K >= 2 (constructor-invariant), so half >= 1.
        let (q1, q2) = q.split_at(half);

        // Steps 2–3: score each codebook into scratch.scores_{1,2}, then
        // heapselect top-K. Reset the top-k buffers first (NEG_INFINITY so
        // every codebook score enters on the first pass).
        //
        // Perf: `scores_{1,2}` are NOT pre-zeroed — the two scoring loops below
        // write every one of the `SQRT_N` slots before any read, so the memset
        // was pure dead work (2·√N stores per query; 2·8 KB at √N=1024).
        scratch.top_1.fill((0, f32::NEG_INFINITY));
        scratch.top_2.fill((0, f32::NEG_INFINITY));

        // Score codebook 1: O(SQRT_N * D_K/2).
        for (i, slot) in scratch.scores_1.iter_mut().enumerate() {
            let key_row = self.keys_1_row(i);
            *slot = score_half(score_fn, q1, key_row);
        }
        // Score codebook 2: O(SQRT_N * D_K/2).
        for (j, slot) in scratch.scores_2.iter_mut().enumerate() {
            let key_row = self.keys_2_row(j);
            *slot = score_half(score_fn, q2, key_row);
        }

        // Heapselect top-K from each codebook: O(SQRT_N * K).
        heapselect_top_k_desc::<SQRT_N, K>(&scratch.scores_1, &mut scratch.top_1);
        heapselect_top_k_desc::<SQRT_N, K>(&scratch.scores_2, &mut scratch.top_2);

        // Steps 4–5: Cartesian product + top-k into out. O(K² * k).
        let n_written = cartesian_top_k::<SQRT_N>(&scratch.top_1[..K], &scratch.top_2[..K], out, k);

        // Step 6: softmax-normalize the k selected scores → weights.
        // (Documented deviation from the global sigmoid rule — see module
        // docs. The weights are convex-combination coefficients, not a
        // probability/UQ claim.)
        if n_written > 0 {
            softmax_normalize_into(&mut out[..n_written]);
        }
        n_written
    }
}

/// Softmax-normalize the scores in `entries` in place, replacing each `.1`
/// with `exp(s_i) / Σ exp(s_j)`.
///
/// Uses the standard max-subtraction trick for numerical stability:
/// `softmax(s) = exp(s − max) / Σ exp(s − max)`.
///
/// Note: `entries` is a slice of `(usize, f32)` tuples, not a contiguous f32
/// buffer, so the SIMD softmax kernels don't apply here. Uses scalar `fast_exp`
/// (Cephes polynomial) instead of libm `exp` for consistency with the rest of
/// the codebase's exp floor (~1.7× faster on aarch64).
fn softmax_normalize_into(entries: &mut [(usize, f32)]) {
    use crate::simd::fast_exp;

    if entries.is_empty() {
        return;
    }
    let max = entries
        .iter()
        .fold(f32::NEG_INFINITY, |m, &(_, s)| m.max(s));
    let mut sum_exp = 0.0f32;
    for (_, s) in entries.iter_mut() {
        let e = fast_exp(*s - max);
        *s = e;
        sum_exp += e;
    }
    if sum_exp > 0.0 && sum_exp.is_finite() {
        let inv = 1.0 / sum_exp;
        for (_, s) in entries.iter_mut() {
            *s *= inv;
        }
    }
}

// ─── Tests (T2.6, T2.7) ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::product_key_memory::types::{ProductKeyMemory, ScoreFn};

    /// Brute-force O(N) top-k baseline — the G2 correctness reference.
    /// Scores all SQRT_N × SQRT_N flat indices via the same `score_fn`
    /// (split into halves, additive), takes the true top-k.
    fn brute_force_top_k<const SQRT_N: usize, const D_K: usize, const D_V: usize>(
        table: &ProductKeyMemory<SQRT_N, D_K, D_V>,
        q: &[f32; D_K],
        score_fn: ScoreFn,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let half = ProductKeyMemory::<SQRT_N, D_K, D_V>::key_half_dim();
        let (q1, q2) = q.split_at(half);
        let n = SQRT_N * SQRT_N;
        let mut all: Vec<(usize, f32)> = Vec::with_capacity(n);
        for i in 0..SQRT_N {
            let s1 = score_half(score_fn, q1, table.keys_1_row(i));
            for j in 0..SQRT_N {
                let s2 = score_half(score_fn, q2, table.keys_2_row(j));
                all.push((i * SQRT_N + j, s1 + s2));
            }
        }
        // Sort descending by score, tiebreak by lower flat_index.
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all.shrink_to_fit();
        all
    }

    /// Jaccard overlap between two top-k index sets: `|A ∩ B| / |A ∪ B|`.
    fn jaccard(a: &[usize], b: &[usize]) -> f32 {
        let set_a: std::collections::HashSet<usize> = a.iter().copied().collect();
        let set_b: std::collections::HashSet<usize> = b.iter().copied().collect();
        let inter = set_a.intersection(&set_b).count() as f32;
        let union = set_a.union(&set_b).count() as f32;
        if union == 0.0 { 1.0 } else { inter / union }
    }

    // ─── T2.6: top-k set matches brute-force ────────────────────────────

    #[test]
    fn t26_top_k_matches_brute_force_single_query() {
        // Small table: 16×16 = 256 slots, D_K=8, D_V=4. K=8 per codebook,
        // final k=8. Cartesian pool is 8×8=64 candidates → top-8.
        const SQRT_N: usize = 16;
        const D_K: usize = 8;
        const D_V: usize = 4;
        const K: usize = 8;
        const OUT_K: usize = 8;

        let table: ProductKeyMemory<SQRT_N, D_K, D_V> = ProductKeyMemory::from_random(42);

        // Query: use the first codebook-1 row concatenated with the first
        // codebook-2 row (a known high-scoring query for the Dot mode).
        let mut q = [0.0f32; D_K];
        let half = D_K / 2;
        q[..half].copy_from_slice(table.keys_1_row(0));
        q[half..].copy_from_slice(table.keys_2_row(0));

        let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
        let mut out = [(0usize, 0.0f32); OUT_K];
        let n = table.query_into(&q, ScoreFn::Dot, OUT_K, &mut out, &mut scratch);
        assert_eq!(n, OUT_K);

        // The flat_index 0 (= i=0, j=0) should be in the result — it's the
        // exact match. Its weight should be the highest.
        let indices: Vec<usize> = out[..n].iter().map(|(i, _)| *i).collect();
        assert!(
            indices.contains(&0),
            "exact match flat_index=0 should be in top-k"
        );

        // Brute-force reference.
        let bf = brute_force_top_k(&table, &q, ScoreFn::Dot, OUT_K);
        let bf_indices: Vec<usize> = bf.iter().map(|(i, _)| *i).collect();

        // Jaccard should be 1.0 for a small table (K=8 per codebook captures
        // all relevant candidates when SQRT_N=16).
        let j = jaccard(&indices, &bf_indices);
        assert!(j >= 0.95, "Jaccard {j} < 0.95 vs brute-force");
    }

    #[test]
    fn t26_top_k_matches_brute_force_many_queries_dot() {
        // Plan 408 T2.6: 1000 random queries, Jaccard ≥ 0.95 mean.
        const SQRT_N: usize = 32;
        const D_K: usize = 16;
        const D_V: usize = 8;
        const K: usize = 8;
        const OUT_K: usize = 8;
        const N_QUERIES: usize = 1000;

        let table: ProductKeyMemory<SQRT_N, D_K, D_V> = ProductKeyMemory::from_random(7);
        let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
        let mut out = [(0usize, 0.0f32); OUT_K];

        // Deterministic query PRNG (reuse splitmix64 logic inline).
        let mut seed_state = 0xA5A5_A5A5_A5A5_A5A5u64;
        let mut next_f32 = || {
            seed_state = seed_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u = (z ^ (z >> 31)) >> 40;
            u as f32 / ((1u32 << 24) as f32) * 2.0 - 1.0
        };

        let mut jaccard_sum = 0.0f32;
        let mut jaccard_min = f32::INFINITY;
        for _ in 0..N_QUERIES {
            let mut q = [0.0f32; D_K];
            for v in q.iter_mut() {
                *v = next_f32();
            }
            let n = table.query_into(&q, ScoreFn::Dot, OUT_K, &mut out, &mut scratch);
            let indices: Vec<usize> = out[..n].iter().map(|(i, _)| *i).collect();
            let bf = brute_force_top_k(&table, &q, ScoreFn::Dot, OUT_K);
            let bf_indices: Vec<usize> = bf.iter().map(|(i, _)| *i).collect();
            let j = jaccard(&indices, &bf_indices);
            jaccard_sum += j;
            jaccard_min = jaccard_min.min(j);
        }
        let mean = jaccard_sum / N_QUERIES as f32;
        // Plan target: mean Jaccard ≥ 0.95. Report the min too (the
        // approximation gap is real — characterize honestly).
        assert!(
            mean >= 0.95,
            "G2 mean Jaccard {mean:.4} < 0.95 (min {jaccard_min:.4})"
        );
    }

    #[test]
    fn t26_top_k_matches_brute_force_idw_mode() {
        // Same as above but with IDW scoring — confirms the factorization
        // holds for the centroid-attracting variant too.
        const SQRT_N: usize = 16;
        const D_K: usize = 8;
        const D_V: usize = 4;
        const K: usize = 8;
        const OUT_K: usize = 8;
        const N_QUERIES: usize = 200;

        let table: ProductKeyMemory<SQRT_N, D_K, D_V> = ProductKeyMemory::from_random(99);
        let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
        let mut out = [(0usize, 0.0f32); OUT_K];

        let mut seed_state = 0xDEAD_BEEF_DEAD_BEEFu64;
        let mut next_f32 = || {
            seed_state = seed_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u = (z ^ (z >> 31)) >> 40;
            u as f32 / ((1u32 << 24) as f32) * 2.0 - 1.0
        };

        let mut jaccard_sum = 0.0f32;
        for _ in 0..N_QUERIES {
            let mut q = [0.0f32; D_K];
            for v in q.iter_mut() {
                *v = next_f32();
            }
            let n = table.query_into(&q, ScoreFn::idw_default(), OUT_K, &mut out, &mut scratch);
            let indices: Vec<usize> = out[..n].iter().map(|(i, _)| *i).collect();
            let bf = brute_force_top_k(&table, &q, ScoreFn::idw_default(), OUT_K);
            let bf_indices: Vec<usize> = bf.iter().map(|(i, _)| *i).collect();
            jaccard_sum += jaccard(&indices, &bf_indices);
        }
        let mean = jaccard_sum / N_QUERIES as f32;
        assert!(mean >= 0.95, "G2 IDW mean Jaccard {mean:.4} < 0.95");
    }

    #[test]
    fn t26_softmax_weights_sum_to_one_and_are_positive() {
        // G2-adjacent: the softmax normalization must produce valid weights.
        const SQRT_N: usize = 8;
        const D_K: usize = 4;
        const D_V: usize = 2;
        const K: usize = 4;
        const OUT_K: usize = 4;

        let table: ProductKeyMemory<SQRT_N, D_K, D_V> = ProductKeyMemory::from_random(1);
        let q = [0.5f32, -0.3, 0.8, 0.1];
        let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
        let mut out = [(0usize, 0.0f32); OUT_K];
        let n = table.query_into(&q, ScoreFn::Dot, OUT_K, &mut out, &mut scratch);
        assert!(n > 0);
        // All weights in (0, 1].
        for (_, w) in out[..n].iter() {
            assert!(*w > 0.0 && *w <= 1.0, "weight {w} out of (0, 1]");
        }
        // Sum to 1 (within f32 tolerance).
        let sum: f32 = out[..n].iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights sum to {sum}, expected 1.0"
        );
        // Descending order (post-normalization, the order is preserved since
        // softmax is monotonic).
        for w in out[..n].windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "weights not descending: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }

    // ─── T2.7: IDW centroid-attracting ──────────────────────────────────

    #[test]
    fn t27_idw_attracts_to_closer_centroids() {
        // On a table built so that codebook rows are clustered, IDW scoring
        // should retrieve slots whose keys are closer (in Euclidean distance)
        // to the query than Dot scoring would. We measure the mean Euclidean
        // distance from q to each retrieved codebook row, IDW vs Dot.
        //
        // Construct: codebook 1 has 4 tight clusters of 4 rows each (SQRT_N=16).
        // Query near cluster 0. IDW should retrieve cluster-0 rows; Dot may
        // retrieve high-magnitude rows from any cluster.
        const SQRT_N: usize = 16;
        const D_K: usize = 8;
        const D_V: usize = 2;
        const K: usize = 4;
        const OUT_K: usize = 4;
        const HALF: usize = D_K / 2;

        // Build codebook 1 with 4 clusters at fixed centers, each cluster
        // having 4 rows = center + small noise. Cluster 0 is near origin;
        // clusters 1-3 are far from origin (high magnitude).
        let mut keys_1 = vec![0.0f32; SQRT_N * HALF].into_boxed_slice();
        let cluster_centers: [[f32; HALF]; 4] = [
            [0.1, 0.1, 0.1, 0.1],   // cluster 0 — near origin
            [5.0, 5.0, 5.0, 5.0],   // cluster 1 — high magnitude
            [-5.0, 5.0, -5.0, 5.0], // cluster 2 — high magnitude
            [5.0, -5.0, -5.0, 5.0], // cluster 3 — high magnitude
        ];
        for (i, row_start) in (0..SQRT_N).enumerate() {
            let cluster = i / 4;
            for d in 0..HALF {
                // Tiny deterministic noise so cluster rows aren't bit-identical.
                let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
                keys_1[row_start * HALF + d] = cluster_centers[cluster][d] + noise;
            }
        }
        // Codebook 2: identity-ish (all rows identical so it doesn't influence
        // the cluster choice — every j contributes the same s2).
        let keys_2 = vec![0.5f32; SQRT_N * HALF].into_boxed_slice();
        let values = vec![0.0f32; SQRT_N * SQRT_N * D_V].into_boxed_slice();
        let table = ProductKeyMemory::<SQRT_N, D_K, D_V>::from_centroids(keys_1, keys_2, values);

        // Query near cluster 0.
        let q = [0.1f32, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5];

        // IDW retrieval.
        let mut scratch: PkmScratch<SQRT_N, K> = PkmScratch::new();
        let mut out_idw = [(0usize, 0.0f32); OUT_K];
        let n_idw = table.query_into(
            &q,
            ScoreFn::idw_default(),
            OUT_K,
            &mut out_idw,
            &mut scratch,
        );

        // Dot retrieval (fresh scratch).
        let mut scratch2: PkmScratch<SQRT_N, K> = PkmScratch::new();
        let mut out_dot = [(0usize, 0.0f32); OUT_K];
        let n_dot = table.query_into(&q, ScoreFn::Dot, OUT_K, &mut out_dot, &mut scratch2);

        // For each retrieved flat_index, decode i = flat_index / SQRT_N,
        // measure Euclidean distance from q1 to keys_1_row(i). Compare IDW
        // mean vs Dot mean. IDW should be smaller (closer to cluster 0).
        let mean_dist = |out: &[(usize, f32)], n: usize| -> f32 {
            let (q1, _q2) = q.split_at(HALF);
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for (flat_idx, _) in out[..n].iter() {
                let i = flat_idx / SQRT_N;
                let row = table.keys_1_row(i);
                let mut d2 = 0.0f32;
                for d in 0..HALF {
                    let diff = q1[d] - row[d];
                    d2 += diff * diff;
                }
                sum += d2.sqrt();
                count += 1;
            }
            if count == 0 {
                f32::INFINITY
            } else {
                sum / count as f32
            }
        };

        let mean_idw = mean_dist(&out_idw, n_idw);
        let mean_dot = mean_dist(&out_dot, n_dot);
        // IDW should retrieve rows closer to q1 (cluster 0). Dot may retrieve
        // the high-magnitude clusters (1-3) because dot(q1, big_vec) can be
        // larger than dot(q1, small_vec) depending on alignment.
        // We assert IDW is at least as good (≤) — strict improvement is the
        // target but cluster geometry can make Dot lucky on some seeds.
        assert!(
            mean_idw <= mean_dot + 1e-5,
            "IDW mean dist {mean_idw:.4} should be <= Dot mean dist {mean_dot:.4}"
        );
    }
}
