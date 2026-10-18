//! Smooth-minimum soft similarity for variable-length multi-token retrieval.
//!
//! Distilled from SoftMatcha 2 (Yoneda et al., ICML 2026, arXiv:2602.10908)
//! by Research 385. The smooth-min aggregation interpolates between plain-min
//! (β→∞, the strictest aggregation) and plain-sum (β≈1, the most lenient),
//! providing better discrimination than plain-cosine (mean) on multi-token
//! queries with partial mismatches.
//!
//! ## GOAT gate (Issue 041 PoC, 2026-07-12; Issue 041 removed, see git history)
//!
//! The PoC (`examples/issue_041_smooth_min_poc.rs`) uses synthetic per-token
//! embeddings — 200 items, 200 queries with all 4 tokens mismatched (same
//! cluster, different word). Results:
//!
//! | Gate | Metric | Result |
//! |------|--------|--------|
//! | G1 (quality) | recall@5 gain | **+12.0pp** (0.815 vs 0.695) |
//! | G2 (latency) | overhead vs mean | **~0 ns** (LLVM vectorized) |
//! | G3 (β sensitivity) | all β ∈ [10¹, 10⁶] beat plain cosine | ✅ |
//!
//! The gain is **modelless** (pure arithmetic on pre-computed cosines, no
//! training, no weights). The PoC resolves the contradiction between
//! Research 385 §4 (PoC uses synthetic data) and Issue 041 (PoC blocked on
//! consumer prerequisites) — the PoC does NOT need a real consumer.
//!
//! ## When to use
//!
//! Use `smooth_min_similarity` when you have per-position cosine similarities
//! (one per token pair) and want to aggregate them into a single score that
//! penalizes low-cosine positions more than plain averaging would. This is
//! the scenario where a distractor item has 1-2 exact-match positions (high
//! cosine) but several unrelated positions (low cosine) — plain mean is fooled
//! by the high-cosine positions, smooth-min correctly penalizes the low ones.
//!
//! ## Feature gate
//!
//! DEFAULT-ON (2026-07-12, Issue 041 T6). Promoted after the first consumer
//! GOAT gate passed: `RerankMethod::SmoothMinAligned` in katgpt-attn-match
//! achieved recall@5 = 1.000 vs Cosine 0.495 (+50.5pp) on position-aligned
//! multi-token retrieval.
//!
//! Smooth-min soft pattern matching — modelless latent-space utility for
//! fuzzy multi-token retrieval (Research 385, SoftMatcha 2, Issue 041; Issue 041 removed, see git history).
//! GOAT PoC PASS: +12pp recall@5 over plain cosine, ~0ns overhead.
//! Consumer GOAT PASS (T6 SmoothMinAligned): recall@5 = 1.000 vs Cosine 0.495
//! (+50.5pp) on position-aligned multi-token retrieval.
//! DEFAULT-ON (2026-07-12): first consumer GOAT gate passes with modelless gain.
//!
//! recos — Rearrangement-Inequality Cosine Similarity (Plan 437, Research 421,
//! arXiv:2602.05266). Saturates under ordinal concordance — wider capture range
//! than cosine. Sits inside the `similarity` module (which is gated on
//! smooth_min_similarity; `recos` implies smooth_min_similarity so the module
//! compiles under --no-default-features). Opt-in until the Phase 2 GOAT gate.

/// Smooth-minimum similarity for variable-length soft pattern matching.
///
/// Aggregates per-position cosine similarities into a single score in [0, 1]
/// using the smooth-min (softmin / LogSumExp family) function:
///
/// ```text
/// sim = 1 - log_β( Σ(β^(1-c_i) - 1) + 1 )
/// ```
///
/// where `c_i` is the i-th cosine similarity and `β` controls sharpness:
/// - `β → ∞`: approaches `min(c_i)` — strict, one bad match kills the score.
/// - `β ≈ 1`: approaches `Σ(c_i) / m` — lenient, like plain mean.
/// - `β = 10⁴`: the paper's empirically-best operating point.
///
/// # Arguments
///
/// * `cosines` — per-position cosine similarities, each in `[-1, 1]`.
/// * `beta` — sharpness parameter. Must be `> 1.0` (use plain mean for β=1).
///
/// # Returns
///
/// Similarity score. Equals `1.0` when all cosines are `1.0` (perfect
/// match). Decreases as cosines decrease. Can go below `0` for poor matches
/// (low cosines or many tokens) — this is correct behavior, as the score is
/// used for ranking, not as a probability. The `+1` and `-1` terms in the
/// formula ensure `sim = 1.0` at the all-perfect-match boundary.
///
/// # Panics
///
/// Panics if `beta <= 1.0` (ln(β) ≤ 0 makes the logarithm undefined) or if
/// `cosines` is empty.
///
/// # Example
///
/// ```
/// # use katgpt_core::smooth_min_similarity;
/// // All positions match moderately → moderate score
/// let sim = smooth_min_similarity(&[0.7, 0.7, 0.7, 0.7], 1e4);
/// assert!(sim > 0.4 && sim < 0.7, "moderate match: {sim}");
///
/// // One position is a perfect match, rest are bad → low score
/// // (plain mean would give 0.40, smooth-min correctly penalizes)
/// assert!(smooth_min_similarity(&[1.0, 0.2, 0.2, 0.2], 1e4) < 0.1);
/// ```
#[inline]
pub fn smooth_min_similarity(cosines: &[f32], beta: f32) -> f32 {
    use crate::simd::fast_exp;

    assert!(beta > 1.0, "beta must be > 1.0, got {beta}");
    assert!(!cosines.is_empty(), "cosines must not be empty");

    let log_beta = beta.ln();
    // Σ(β^(1-c_i) - 1) + 1
    // mul_add for FMA fusion: (1-c) * ln(β) computed in one instruction.
    let sum = cosines
        .iter()
        .map(|&c| fast_exp((1.0f32 - c).mul_add(log_beta, 0.0)) - 1.0)
        .sum::<f32>()
        + 1.0;

    // Guard against sum ≤ 0 (can happen if cosines are very negative)
    if sum <= 0.0 {
        return 0.0;
    }

    1.0 - sum.ln() / log_beta
}

/// Insertion/deletion penalty using Zipfian-whitened norm.
///
/// Computes `exp(-norm_sq / gamma)` — a decay penalty for edit operations
/// (insertions/deletions) in variable-length pattern matching. Tokens with
/// high Zipfian-whitened norm (high information content) are penalized more
/// for editing; low-information tokens (stopwords like "the", "of") are cheap
/// to edit.
///
/// # Arguments
///
/// * `norm_sq` — squared norm of the edited token's embedding (post-Zipfian
///   whitening). Should be `>= 0`.
/// * `gamma` — penalty scale. The paper sets `γ = m · γ'` where `γ'` is tuned
///   so the penalty equals `1/e` at the 50th-lowest norm for query length `m`.
///
/// # Returns
///
/// Penalty multiplier in `(0, 1]`. `norm_sq = 0` → 1.0 (no penalty).
/// `norm_sq = gamma` → `1/e ≈ 0.368`.
///
/// # Example
///
/// ```
/// # use katgpt_core::edit_penalty;
/// assert!((edit_penalty(0.0, 1.0) - 1.0).abs() < 1e-6);
/// assert!((edit_penalty(1.0, 1.0) - std::f32::consts::E.powi(-1)).abs() < 1e-6);
/// ```
#[inline]
pub fn edit_penalty(norm_sq: f32, gamma: f32) -> f32 {
    crate::simd::fast_exp(-norm_sq / gamma)
}

// ── recos — Rearrangement-Inequality Cosine Similarity ───────────────────
// Distilled from Ai (2026), arXiv:2602.05266 "Beyond Cosine Similarity"
// (Research 421, Plan 437). Gated on `recos` (which implies
// `smooth_min_similarity` so this module compiles under --no-default-features).
//
// recos saturates at 1.0 under ORDINAL CONCORDANCE (any monotonic
// relationship) — a strictly wider capture range than cosine, which requires
// LINEAR dependence. Always |recos| >= |cos| in absolute value (Corollary 2);
// unlike decos, recos does NOT collapse to cosine on unit-norm vectors
// (Corollary 3) — important because the pipeline unit-normalizes embeddings.

/// Dot product on fixed 8-dim vectors (HLA dimension).
///
/// Local to this module because `riir-neuron-db`'s `dot_8` is private and
/// lives in a different crate. Kept simple (no FMA) so recos stays
/// bit-deterministic across platforms — the Phase 2 GOAT G1 gate depends on it.
#[cfg(feature = "recos")]
#[inline]
fn dot_8(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Rearrangement-inequality-based cosine similarity (recos).
///
/// Distilled from Ai (2026), arXiv:2602.05266 (Research 421). Saturates at 1.0
/// under ordinal concordance (any monotonic relationship) — a strictly wider
/// capture range than cosine (which requires linear dependence). Always
/// `|recos| >= |cos|` in absolute value (Corollary 2).
///
/// Cost: O(d log d) — one sort per vector. For d=8 this is ~24 comparisons + 8
/// FMA. The sort is the dominant cost vs cosine's 8 FMA + pre-computed-norm
/// shortcut (which recos structurally cannot reuse — the rearrangement bound
/// is a function of sorted order, not norm; see Plan 437 §"Critical subtlety").
///
/// Use when embeddings are known to have nonlinear-but-consistent relationships
/// (consolidated style_weights, trained direction vectors, schema-centroid item
/// embeddings). Use cosine when embeddings are already linearly aligned with
/// the query (raw text embeddings from a sentence transformer).
///
/// # Zero-vector guard
///
/// If either vector is all zeros, the rearrangement bound is zero; returns 0.0
/// (no NaN, no panic). NaN inputs will panic inside `sort_by` — that's a caller
/// bug.
#[cfg(feature = "recos")]
#[inline]
pub fn recos_sim(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let dot = dot_8(a, b);
    // Rearrangement bound: sort both, dot the sorted.
    // For dot >= 0: u↑·v↑ (both ascending) = max permutation dot.
    // For dot <  0: u↑·v↓ (b descending) = min permutation dot (negative),
    //               and dot/bound is negative/negative = positive in [0,1].
    let mut a_sorted = *a;
    let mut b_sorted = *b;
    a_sorted.sort_by(|x, y| x.total_cmp(y));
    if dot >= 0.0 {
        b_sorted.sort_by(|x, y| x.total_cmp(y));
    } else {
        b_sorted.sort_by(|x, y| y.total_cmp(x));
    }
    let bound = dot_8(&a_sorted, &b_sorted);
    if bound.abs() < 1e-12 {
        0.0
    } else {
        dot / bound
    }
}

/// recos ranking score — preserves ordering, returns `(dot/bound)²` copysigned
/// by `dot` (so negative-recos ranks below positive). Use for top-k selection
/// where only the ORDER matters. Mirrors `cosine_sim_ranking`'s squared
/// convention (squaring widens the gap between similar/dissimilar without
/// changing the order on each side of zero).
///
/// NOTE: unlike `cosine_sim_ranking_scaled`, this does NOT take a pre-computed
/// `norm_a_sq` — the rearrangement bound is not a function of norm alone. The
/// `ShardIndex::query` consumer (Plan 437 Phase 4) must call this 3× without the
/// norm fold; this is why the G2 latency gate measures the full 3-candidate
/// rerank, not single-pair recos-vs-cosine.
#[cfg(feature = "recos")]
#[inline]
pub fn recos_sim_ranking(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let dot = dot_8(a, b);
    let mut a_sorted = *a;
    let mut b_sorted = *b;
    a_sorted.sort_by(|x, y| x.total_cmp(y));
    if dot >= 0.0 {
        b_sorted.sort_by(|x, y| x.total_cmp(y));
    } else {
        b_sorted.sort_by(|x, y| y.total_cmp(x));
    }
    let bound = dot_8(&a_sorted, &b_sorted);
    if bound.abs() < 1e-12 {
        0.0
    } else {
        (dot / bound).powi(2).copysign(dot)
    }
}

/// recos on arbitrary-length slices (generic dim). Used by MAG transfer scoring
/// (d=64 style_weights) and any variable-dimension consumer. Same algorithm as
/// [`recos_sim`] but heap-backed sorts via `sort_unstable_by` (cheaper than the
/// stable sort; ties don't carry meaning here since we only read the dot of the
/// sorted arrays).
///
/// `to_vec()` allocates — acceptable for the cold MAG path. The d=8 variants
/// sort stack arrays and are alloc-free. For the zero-alloc hot path, use
/// [`recos_sim_slice_into`].
#[cfg(feature = "recos")]
#[inline]
pub fn recos_sim_slice(a: &[f32], b: &[f32]) -> f32 {
    let mut a_owned = a.to_vec();
    let mut b_owned = b.to_vec();
    recos_sim_slice_into(&mut a_owned, &mut b_owned)
}

/// Zero-alloc variant of [`recos_sim_slice`] — sorts `a` and `b` **in place**.
///
/// The core recos algorithm on mutable slices. Both buffers are sorted in
/// place (ascending for `a`; ascending-or-descending for `b` depending on the
/// sign of the dot product). The caller must not rely on buffer order after
/// this call.
///
/// Used by `transfer_score_into` (MAG zero-alloc hot path, Plan 437 Phase 3)
/// where the caller owns the scratch buffers. Single source of truth for the
/// generic-dim recos algorithm — [`recos_sim_slice`] delegates here.
#[cfg(feature = "recos")]
#[inline]
pub fn recos_sim_slice_into(a: &mut [f32], b: &mut [f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    a.sort_unstable_by(|x, y| x.total_cmp(y));
    if dot >= 0.0 {
        b.sort_unstable_by(|x, y| x.total_cmp(y));
    } else {
        b.sort_unstable_by(|x, y| y.total_cmp(x));
    }
    let bound: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    if bound.abs() < 1e-12 {
        0.0
    } else {
        dot / bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    // ── smooth_min_similarity: basic properties ──

    #[test]
    fn all_ones_gives_one() {
        // All cosines = 1.0 → perfect match → similarity = 1.0
        let sim = smooth_min_similarity(&[1.0, 1.0, 1.0, 1.0], 1e4);
        assert!(
            (sim - 1.0).abs() < TOL,
            "all-ones should give 1.0, got {sim}"
        );
    }

    #[test]
    fn single_element_returns_it() {
        // m=1: smooth_min([c]) = 1 - log_β(β^(1-c) - 1 + 1) = 1 - (1-c) = c
        let sim = smooth_min_similarity(&[0.7], 1e4);
        assert!(
            (sim - 0.7).abs() < TOL,
            "single element should return c, got {sim}"
        );
    }

    #[test]
    fn all_equal_cosines_decreases_with_token_count() {
        // For all c_i = c and m > 1, smooth_min = 1 - log_β(m·(β^(1-c)−1)+1)
        // which is LESS than c (the log_β(m) offset). More tokens → lower
        // score for the same cosine level. This is by design: a 4-token match
        // at cosine 0.5 is less confident than a 1-token match at 0.5.
        let sim_m1 = smooth_min_similarity(&[0.5], 1e4);
        let sim_m2 = smooth_min_similarity(&[0.5, 0.5], 1e4);
        let sim_m4 = smooth_min_similarity(&[0.5, 0.5, 0.5, 0.5], 1e4);
        assert!(
            (sim_m1 - 0.5).abs() < TOL,
            "m=1 should return c, got {sim_m1}"
        );
        assert!(sim_m2 < sim_m1, "m=2 should be < m=1: {sim_m2} < {sim_m1}");
        assert!(sim_m4 < sim_m2, "m=4 should be < m=2: {sim_m4} < {sim_m2}");
    }

    #[test]
    fn beta_large_approaches_min_minus_offset() {
        // β → ∞: smooth_min → min(c_i) − log_β(m)
        // For m=4, β=10^10: log_10^10(4) ≈ 0.0602, so sim ≈ min − 0.06
        let cosines = [0.3, 0.8, 0.5, 0.9];
        let min_val = 0.3f32;
        let sim_large = smooth_min_similarity(&cosines, 1e10);
        // Should be close to min, slightly below due to the m-offset
        assert!(
            sim_large > min_val - 0.1 && sim_large <= min_val,
            "β=10^10 should be near min ({min_val}), got {sim_large}"
        );
    }

    #[test]
    fn beta_affects_score() {
        // Different β values give different scores (the function is
        // responsive to β). The direction depends on the input distribution:
        // for all-moderate cosines, lower β (sum-like) can give higher scores;
        // for mixed high+low cosines, higher β (min-like) can give higher scores.
        // The PoC showed all β ∈ [10¹, 10⁶] beat plain cosine on recall@5.
        let cosines = [0.3, 0.8, 0.5, 0.9];
        let sim_low = smooth_min_similarity(&cosines, 1e2);
        let sim_mid = smooth_min_similarity(&cosines, 1e4);
        let sim_high = smooth_min_similarity(&cosines, 1e6);
        // Just verify they're all different and finite
        assert!(sim_low.is_finite() && sim_mid.is_finite() && sim_high.is_finite());
        assert!(
            sim_low != sim_mid || sim_mid != sim_high,
            "β should affect the score"
        );
    }

    #[test]
    fn penalizes_low_cosine_more_than_mean() {
        // [0.9, 0.9, 0.1, 0.1] vs [0.5, 0.5, 0.5, 0.5]
        // mean: 0.5 vs 0.5 (tie!) but smooth_min should prefer all-moderate
        let mixed = [0.9f32, 0.9, 0.1, 0.1];
        let uniform = [0.5f32, 0.5, 0.5, 0.5];
        let sim_mixed = smooth_min_similarity(&mixed, 1e4);
        let sim_uniform = smooth_min_similarity(&uniform, 1e4);
        assert!(
            sim_uniform > sim_mixed,
            "uniform {sim_uniform} should beat mixed {sim_mixed}"
        );
    }

    #[test]
    fn handles_negative_cosines() {
        // Anti-correlated tokens should give low (possibly negative) scores
        let sim = smooth_min_similarity(&[0.5, -0.5, 0.5, -0.5], 1e4);
        assert!(
            sim < 0.0,
            "anti-correlated should give negative score, got {sim}"
        );
    }

    #[test]
    fn empty_panics() {
        let result = std::panic::catch_unwind(|| {
            smooth_min_similarity(&[], 1e4);
        });
        assert!(result.is_err(), "empty cosines should panic");
    }

    #[test]
    fn beta_leq_one_panics() {
        let result = std::panic::catch_unwind(|| {
            smooth_min_similarity(&[0.5], 1.0);
        });
        assert!(result.is_err(), "beta=1.0 should panic");
    }

    #[test]
    fn deterministic() {
        // Same input → same output
        let cosines = [0.3, 0.7, 0.5, 0.9];
        let a = smooth_min_similarity(&cosines, 1e4);
        let b = smooth_min_similarity(&cosines, 1e4);
        assert!((a - b).abs() < f32::EPSILON, "should be deterministic");
    }

    // ── edit_penalty: basic properties ──

    #[test]
    fn edit_penalty_zero_norm_is_one() {
        assert!((edit_penalty(0.0, 1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn edit_penalty_at_gamma_is_inv_e() {
        let expected = std::f32::consts::E.powi(-1);
        assert!((edit_penalty(1.0, 1.0) - expected).abs() < TOL);
    }

    #[test]
    fn edit_penalty_decreasing_in_norm() {
        // Higher norm → lower penalty (more information lost)
        let p1 = edit_penalty(0.5, 1.0);
        let p2 = edit_penalty(1.0, 1.0);
        let p3 = edit_penalty(2.0, 1.0);
        assert!(
            p1 > p2 && p2 > p3,
            "should be decreasing: {p1} > {p2} > {p3}"
        );
    }

    #[test]
    fn edit_penalty_increasing_in_gamma() {
        // Higher gamma → less penalty (more tolerant)
        let p1 = edit_penalty(1.0, 0.5);
        let p2 = edit_penalty(1.0, 1.0);
        let p3 = edit_penalty(1.0, 2.0);
        assert!(
            p1 < p2 && p2 < p3,
            "should be increasing: {p1} < {p2} < {p3}"
        );
    }

    #[test]
    fn edit_penalty_always_in_unit_interval() {
        // For non-negative norm_sq, penalty is in [0, 1]. The open-interval (0,...)
        // bound is relaxed to [0,...] because Cephes `fast_exp` (the codebase-wide
        // exp floor) returns exactly 0.0 for arguments < -87.3 (where libm `exp`
        // would return a subnormal ~3.7e-44). For norm_sq=100, gamma=1.0:
        // exp(-100) is mathematically ~3.7e-44, which rounds to 0.0 in f32 for
        // any correct implementation that doesn't preserve subnormals.
        for &norm_sq in &[0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let p = edit_penalty(norm_sq, 1.0);
            assert!(
                (0.0..=1.0).contains(&p),
                "penalty {p} not in [0,1] for norm_sq={norm_sq}"
            );
        }
    }

    // ── Numerical properties from the GOAT PoC ──

    #[test]
    fn poc_correct_item_beats_distractor() {
        // From the PoC: correct item has all-moderate cosines, distractor has
        // 2 exact matches + 2 unrelated. Plain cosine is fooled; smooth-min
        // correctly ranks the correct item higher.
        let correct = [0.53f32, 0.55, 0.29, 0.33];
        let distractor = [0.29f32, 1.0, -0.12, 1.0];

        let correct_plain = correct.iter().sum::<f32>() / 4.0;
        let distractor_plain = distractor.iter().sum::<f32>() / 4.0;

        let correct_smooth = smooth_min_similarity(&correct, 1e4);
        let distractor_smooth = smooth_min_similarity(&distractor, 1e4);

        // Plain cosine: distractor wins (wrong!)
        assert!(
            distractor_plain > correct_plain,
            "plain cosine should rank distractor higher: {distractor_plain} > {correct_plain}"
        );

        // Smooth-min: correct wins (right!)
        assert!(
            correct_smooth > distractor_smooth,
            "smooth-min should rank correct higher: {correct_smooth} > {distractor_smooth}"
        );
    }

    #[test]
    fn two_token_case() {
        // m=2: smooth_min of [0.8, 0.6] should be below min(0.6) due to the
        // log_β(m) offset, but above the distractor scenario.
        let sim = smooth_min_similarity(&[0.8, 0.6], 1e4);
        // For m=2, β=10⁴: sim ≈ 0.586 (below min 0.6, above 0)
        assert!(
            sim > 0.0 && sim < 0.6,
            "two-token case: {sim} should be in (0, 0.6)"
        );
    }

    #[test]
    fn large_token_count() {
        // m=64: should not panic or overflow. With many tokens, the
        // log_β(m) offset is larger, so the score can be lower.
        let cosines: Vec<f32> = (0..64).map(|i| 0.3 + 0.01 * i as f32).collect();
        let sim = smooth_min_similarity(&cosines, 1e4);
        assert!(
            sim.is_finite(),
            "large token count should give finite result"
        );
        // With 64 tokens at cosine 0.3-0.93, sim is dominated by the low end
        // and the large m offset. Just check it's finite and ordered correctly.
    }

    // ── recos: Rearrangement-Inequality Cosine Similarity (Plan 437) ──
    //
    // Covers Corollaries 2 & 3, the zero-vector guard, and slice/d=8
    // consistency. Gated on `recos` (implied feature on during these tests).

    #[cfg(feature = "recos")]
    fn cosine_sim(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let dot = dot_8(a, b);
        let na = dot_8(a, a).sqrt();
        let nb = dot_8(b, b).sqrt();
        if na < 1e-12 || nb < 1e-12 {
            0.0
        } else {
            dot / (na * nb)
        }
    }

    #[cfg(feature = "recos")]
    fn normalize(a: [f32; 8]) -> [f32; 8] {
        let n = dot_8(&a, &a).sqrt();
        if n < 1e-12 {
            return a;
        }
        let mut out = a;
        for x in &mut out {
            *x /= n;
        }
        out
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_ordinal_concordant_is_one() {
        // Both strictly increasing → perfect ordinal concordance → recos = 1.0
        // exactly. We use a monotonic-but-NONLINEAR pair (b = a²) so that
        // cosine stays strictly < 1.0 while recos saturates — this is the
        // wider-capture-range property (Corollary 2) demonstrated without
        // the trivial linear case b = k·a where cosine also hits 1.0.
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]; // b = a²
        let r = recos_sim(&a, &b);
        assert!(
            (r - 1.0).abs() < 1e-5,
            "ordinal concordant should be 1.0, got {r}"
        );
        let c = cosine_sim(&a, &b);
        // recos == 1.0, cosine < 1.0 — demonstrates the wider capture range.
        assert!(
            c < 1.0 - 1e-4,
            "cosine {c} should be < 1.0 here (nonlinear pair)"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_discordant_below_one() {
        // Shuffled b (non-monotonic w.r.t. a) → recos < 1.0.
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // b is NOT monotonic in the same order as a: high where a is low.
        let b = [80.0f32, 10.0, 70.0, 20.0, 60.0, 30.0, 50.0, 40.0];
        let r = recos_sim(&a, &b);
        assert!(r < 1.0 - 1e-4, "discordant recos should be < 1.0, got {r}");
        assert!(r.is_finite(), "recos should be finite, got {r}");
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_gte_cos_abs() {
        // Corollary 2: |recos| >= |cos| - eps for all pairs. Fuzz 1000 pairs.
        // Use a fixed seed for determinism (the GOAT G1 gate depends on this).
        // Simple LCG — no dep on a PRNG crate.
        const EPS: f32 = 1e-4;

        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to f32 in [-5, 5].
            ((state >> 40) as f32) / ((1u64 << 24) as f32) * 10.0 - 5.0
        };
        for _ in 0..1000 {
            let a: [f32; 8] = std::array::from_fn(|_| next());
            let b: [f32; 8] = std::array::from_fn(|_| next());
            let r = recos_sim(&a, &b);
            let c = cosine_sim(&a, &b);
            assert!(
                r.abs() >= c.abs() - EPS,
                "Corollary 2 violated: |recos| {} < |cos| {} - {}",
                r.abs(),
                c.abs(),
                EPS
            );
        }
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_distinct_from_cos_unit_norm() {
        // Corollary 3: on unit-norm vectors decos collapses to cos, but recos
        // stays distinct. The critical property for our pipeline, which
        // unit-normalizes embeddings. Construct a monotonic-but-nonlinear pair
        // (b = a^2): recos = 1.0 (ordinal concordance), cos < 1.0 (nonlinear).
        let a_raw = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b_raw = [1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
        let a = normalize(a_raw);
        let b = normalize(b_raw);
        let r = recos_sim(&a, &b);
        let c = cosine_sim(&a, &b);
        assert!(
            (r - 1.0).abs() < 1e-4,
            "unit-norm recos should saturate at 1.0, got {r}"
        );
        assert!(
            c < 1.0 - 1e-3,
            "unit-norm cosine should stay < 1.0 (nonlinear), got {c}"
        );
        assert!(
            r > c,
            "recos {r} should beat cosine {c} on monotonic-nonlinear"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_zero_vector_guard() {
        // Zero vector → bound is zero → return 0.0 (no NaN, no panic).
        let zero = [0.0f32; 8];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let r = recos_sim(&zero, &b);
        assert!(
            r == 0.0 && r.is_finite(),
            "zero-vector recos should be 0.0, got {r}"
        );
        let r2 = recos_sim(&b, &zero);
        assert!(
            r2 == 0.0 && r2.is_finite(),
            "zero-vector recos (b,zero) should be 0.0, got {r2}"
        );
        // Ranking variant too.
        let r3 = recos_sim_ranking(&zero, &b);
        assert!(
            r3 == 0.0 && r3.is_finite(),
            "zero-vector ranking should be 0.0, got {r3}"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_negative_dot_returns_positive_in_unit() {
        // When dot < 0: bound (min permutation dot) is also negative, so
        // dot/bound is positive in [0,1]. Sign-flipped concordant vectors
        // (b = -k·a, k>0) have strictly negative dot but remain ordinally
        // concordant after the sign flip — recos saturates at +1.0.
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [-10.0f32, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0];
        let dot = dot_8(&a, &b);
        assert!(dot < 0.0, "precondition: dot should be negative, got {dot}");
        let r = recos_sim(&a, &b);
        assert!(
            (r - 1.0).abs() < 1e-4,
            "sign-flipped concordant recos should be +1.0, got {r}"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_sim_slice_matches_d8() {
        // recos_sim_slice on an 8-len slice must equal recos_sim on [f32;8].
        let a: [f32; 8] = [0.3, -1.2, 4.5, 2.2, -0.7, 3.1, 1.9, -2.4];
        let b: [f32; 8] = [1.1, 0.4, -2.8, 3.3, 1.7, -0.9, 2.5, 0.6];
        let fixed = recos_sim(&a, &b);
        let slice = recos_sim_slice(&a, &b);
        assert!(
            (fixed - slice).abs() < 1e-5,
            "slice {slice} should match d8 {fixed}"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn ranking_preserves_order() {
        // For non-negative-recos cases, recos_sim_ranking orders the same as
        // recos_sim (squaring is monotonic on [0,1]).
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Three candidates with increasing ordinal concordance.
        let b_perfect = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let b_mid = [10.0f32, 30.0, 20.0, 50.0, 40.0, 70.0, 60.0, 80.0];
        let b_low = [80.0f32, 10.0, 70.0, 20.0, 60.0, 30.0, 50.0, 40.0];

        let r_perf = recos_sim(&a, &b_perfect);
        let r_mid = recos_sim(&a, &b_mid);
        let r_low = recos_sim(&a, &b_low);
        let k_perf = recos_sim_ranking(&a, &b_perfect);
        let k_mid = recos_sim_ranking(&a, &b_mid);
        let k_low = recos_sim_ranking(&a, &b_low);

        // Same ordering under both scorers.
        assert!(
            r_perf >= r_mid && r_mid >= r_low,
            "recos order: {r_perf} >= {r_mid} >= {r_low}"
        );
        assert!(
            k_perf >= k_mid && k_mid >= k_low,
            "ranking order: {k_perf} >= {k_mid} >= {k_low}"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_deterministic() {
        let a = [0.3f32, -1.2, 4.5, 2.2, -0.7, 3.1, 1.9, -2.4];
        let b = [1.1f32, 0.4, -2.8, 3.3, 1.7, -0.9, 2.5, 0.6];
        let x = recos_sim(&a, &b);
        let y = recos_sim(&a, &b);
        assert!(
            (x - y).abs() < f32::EPSILON,
            "recos should be deterministic"
        );
    }
}
