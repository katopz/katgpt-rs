//! Sigmoid-gated top-K slice application.
//!
//! Reuses the SP-KV `soft_gate_bias` convention (Plan 070, see
//! `sp_kv/utility_predictor.rs:146`): the gate is written as an *additive
//! log-bias* in score space, `bias = log(s + ε)`. This is sigmoid-compatible
//! (adding it to logits then taking sigmoid preserves a valid probability) and
//! **never softmax** — softmax would require subtracting a partition-function
//! normalizer, which is exactly the non-additive, global-coupling operation we
//! must avoid for a local KV-slice gate.
//!
//! # Apply contract
//!
//! Two entry points mirroring the SP-KV `predict` / `predict_into` pattern
//! (`src/sp_kv/utility_predictor.rs`):
//!
//! - **[`GatedKvSlice::apply`]** — zero-allocation hot path. Caller provides
//!   `out_bias: &mut [f32]` *and* `idx_scratch: &mut [usize]`, both of length
//!   `n_groups`. Nothing is allocated; safe to call in a decode loop or per
//!   tick. This is the path the GOAT G-zero-alloc gate (T3.5) exercises.
//! - **[`GatedKvSlice::apply_allocating`]** — convenience wrapper that owns its
//!   scratch. One `Vec` allocation per call; use for one-shot probe outputs.
//!
//! Top-K groups (K = `budget.k_for(ca)`) → `log(score_normalized + ε)`, the
//! rest → `f32::NEG_INFINITY` (fully pruned in score space).
//! `score_normalized = score / max_score`, so the strongest retained group gets
//! bias ≈ 0 (no effect under sigmoid) and weaker retained groups get mildly
//! negative bias.

use super::types::{DensityBudget, KvGroupRanking};

/// Gate applier: turns a `KvGroupRanking` + `DensityBudget` + `ca` into a
/// top-K additive log-bias vector over KV groups.
///
/// Stateless — kept as a unit struct for API symmetry with `CsKvProbe` and to
/// give future per-channel scaling a place to hang config.
pub struct GatedKvSlice;

/// ε for the `log(s + ε)` soft-gate. Matches SP-KV `soft_gate_bias` exactly
/// (`(utility + 1e-8f32).ln()`). Sigmoid-compatible, never softmax.
const GATE_EPS: f32 = 1e-8_f32;

impl GatedKvSlice {
    /// Apply ranking + budget → gate-bias vector. **Zero-allocation hot path.**
    ///
    /// - `ranking`: per-group importance scores.
    /// - `budget`: the `K(ca)` interpolator config.
    /// - `ca`: context-awareness scalar ∈ [0, 1].
    /// - `_kv`: flattened KV cache slice. **Reserved** for future per-channel
    ///   scaling; currently unused (the bias is group-level, derived from the
    ///   ranking only). Kept in the signature so callers don't need to change
    ///   when that lands.
    /// - `idx_scratch`: caller-owned `&mut [usize]` of length `n_groups`, used
    ///   as scratch for the top-K index sort. Overwritten; contents on return
    ///   are unspecified. Pass a reused buffer in hot loops.
    /// - `out_bias`: caller-owned `&mut [f32]` of length `n_groups`. Overwritten
    ///   in place with the additive log-bias vector.
    ///
    /// Returns nothing. Panics if buffer lengths disagree with `ranking.n_groups`.
    ///
    /// This is the path the GOAT G-zero-alloc gate (T3.5) exercises — calling it
    /// 10K times in a loop performs zero heap allocations after warmup.
    pub fn apply(
        ranking: &KvGroupRanking,
        budget: &DensityBudget,
        ca: f32,
        _kv: &[f32],
        idx_scratch: &mut [usize],
        out_bias: &mut [f32],
    ) {
        let n_groups = ranking.n_groups;
        assert_eq!(
            ranking.scores.len(),
            n_groups,
            "ragged ranking: scores.len() != n_groups"
        );
        assert_eq!(
            out_bias.len(),
            n_groups,
            "out_bias length must equal n_groups"
        );
        assert_eq!(
            idx_scratch.len(),
            n_groups,
            "idx_scratch length must equal n_groups"
        );

        // Default everything to pruned; we only "un-prune" the top-K.
        for b in out_bias.iter_mut() {
            *b = f32::NEG_INFINITY;
        }
        // Initialize scratch to identity permutation in-place.
        for (i, slot) in idx_scratch.iter_mut().enumerate() {
            *slot = i;
        }

        if n_groups == 0 {
            return;
        }

        let k = budget.k_for(ca).min(n_groups);
        if k == 0 {
            return;
        }

        // Max score is the normalization divisor. Floor at GATE_EPS so a
        // uniformly-zero ranking still yields finite (≈ log(ε)) biases rather
        // than NaN.
        let mut max_score = 0.0_f32;
        for &s in ranking.scores.iter() {
            if s > max_score {
                max_score = s;
            }
        }
        let max_score = max_score.max(GATE_EPS);

        // Top-K by score, descending. Full sort of the small index buffer
        // (n_groups is typically ≤ n_heads, e.g. 64); for these sizes this beats
        // the bookkeeping of a partial selection network. Operates in-place on
        // caller-provided scratch — zero allocation.
        idx_scratch.sort_by(|&a, &b| {
            ranking.scores[b]
                .partial_cmp(&ranking.scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &g in idx_scratch.iter().take(k) {
            // Normalize to [0, 1] w.r.t. the max, then soft-gate-bias.
            let s = (ranking.scores[g] / max_score).max(0.0);
            out_bias[g] = (s + GATE_EPS).ln();
        }
    }

    /// Convenience wrapper around [`Self::apply`] that owns its scratch buffer.
    ///
    /// Allocates one `Vec<usize>` of length `n_groups` per call. Prefer
    /// [`Self::apply`] in hot loops — this is for one-shot probe outputs where
    /// the allocation overhead is negligible relative to the probe itself.
    pub fn apply_allocating(
        ranking: &KvGroupRanking,
        budget: &DensityBudget,
        ca: f32,
        kv: &[f32],
        out_bias: &mut [f32],
    ) {
        let n_groups = ranking.n_groups;
        let mut idx = vec![0_usize; n_groups];
        Self::apply(ranking, budget, ca, kv, &mut idx, out_bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranking(scores: &[f32]) -> KvGroupRanking {
        KvGroupRanking::from_scores(scores.to_vec())
    }

    fn run(r: &KvGroupRanking, budget: &DensityBudget, ca: f32, out: &mut [f32]) {
        let n = r.n_groups;
        let mut idx = vec![0_usize; n];
        GatedKvSlice::apply(r, budget, ca, &[], &mut idx, out);
    }

    #[test]
    fn test_apply_writes_top_k_only() {
        // 8 groups, descending-interleaved scores. ca=0.5 on D=8 budget.
        let r = ranking(&[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]);
        let budget = DensityBudget::for_dim(8);
        let k = budget.k_for(0.5);
        let mut out = vec![0.0_f32; 8];
        run(&r, &budget, 0.5, &mut out);

        let finite = out.iter().filter(|&&b| b.is_finite()).count();
        assert_eq!(
            finite, k,
            "expected exactly {k} finite entries (K(ca=0.5) of D=8), got {finite}"
        );
        let neg_inf = out.iter().filter(|&&b| b.is_infinite() && b < 0.0).count();
        assert_eq!(neg_inf, 8 - k);

        // The finite entries must be exactly the top-k scoring groups:
        // {1(0.9), 3(0.8), 5(0.7), 7(0.6), ...} in descending order.
        let mut expected: Vec<(usize, f32)> = r.scores.iter().copied().enumerate().collect();
        expected.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k_groups: std::collections::HashSet<usize> =
            expected.iter().take(k).map(|(g, _)| *g).collect();
        for (g, &b) in out.iter().enumerate() {
            match b.is_finite() {
                true => assert!(top_k_groups.contains(&g), "group {g} finite but not in top-{k}"),
                false => assert!(!top_k_groups.contains(&g), "group {g} pruned but in top-{k}"),
            }
        }
    }

    #[test]
    fn test_allocating_wrapper_matches_zero_alloc() {
        let r = ranking(&[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]);
        let budget = DensityBudget::for_dim(8);
        let mut out_a = vec![0.0_f32; 8];
        let mut out_b = vec![0.0_f32; 8];
        run(&r, &budget, 0.5, &mut out_a);
        GatedKvSlice::apply_allocating(&r, &budget, 0.5, &[], &mut out_b);
        // Both -∞ is a valid match (pruned entry); only finite entries need
        // approximate numeric equality. NaN-from-(-inf - -inf) is avoided.
        for (a, b) in out_a.iter().zip(out_b.iter()) {
            if a.is_finite() && b.is_finite() {
                assert!((a - b).abs() < 1e-6, "wrapper divergence: {a} vs {b}");
            } else {
                assert_eq!(a.is_finite(), b.is_finite(), "prune/retain mismatch: {a} vs {b}");
            }
        }
    }

    #[test]
    fn test_top_group_bias_is_near_zero() {
        // The strongest retained group has normalized score 1.0, so its bias
        // ≈ log(1 + ε) ≈ ε ≈ 0. Sigmoid(· + 0) is the identity step — exactly
        // the "no effect" semantics from SP-KV soft_gate_bias(u→1)→0.
        let r = ranking(&[0.0, 0.0, 1.0, 0.0]);
        let budget = DensityBudget::for_dim(4);
        let mut out = vec![0.0_f32; 4];
        run(&r, &budget, 1.0, &mut out); // ca=1 → retain all
        assert!(out[2].abs() < 1e-5, "top group bias should be ~0, got {}", out[2]);
    }

    #[test]
    fn test_ca_zero_retains_only_sparse_floor() {
        let r = ranking(&[0.5, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]);
        let budget = DensityBudget::for_dim(8);
        let mut out = vec![0.0_f32; 8];
        run(&r, &budget, 0.0, &mut out);
        let finite = out.iter().filter(|b| b.is_finite()).count();
        assert_eq!(finite, budget.k_sparse);
        // And the single retained group is the global max (group 1, score 0.9).
        assert!(out[1].abs() < 1e-5, "sparse floor should retain the top group");
    }

    #[test]
    fn test_all_zero_ranking_stays_finite_for_retained() {
        // Uniformly-zero ranking: max_score floored to ε, retained groups get
        // log(0/ε + ε) = log(ε) ≈ −18 (finite, not NaN). The pruned groups are
        // -∞ by design — verify retained entries are finite and non-NaN.
        let r = ranking(&[0.0, 0.0, 0.0, 0.0]);
        let budget = DensityBudget::for_dim(4);
        let mut out = vec![0.0_f32; 4];
        run(&r, &budget, 1.0, &mut out);
        let finite_count = out.iter().filter(|b| b.is_finite()).count();
        assert!(finite_count > 0, "at least one group must be retained at ca=1.0");
        for &b in out.iter() {
            // No NaN permitted anywhere. Pruned entries are -∞ (finite==false
            // but is_nan()==false too); retained entries are finite.
            assert!(!b.is_nan(), "NaN bias leaked for all-zero ranking: {b}");
        }
    }

    #[test]
    fn test_zero_alloc_signature() {
        // Compile-time check that apply takes &mut [f32] and returns nothing —
        // no Vec on the hot path. This is the T3.5 precondition.
        let r = ranking(&[0.1, 0.9, 0.2, 0.8]);
        let budget = DensityBudget::for_dim(4);
        let mut idx = vec![0_usize; 4];
        let mut out = vec![0.0_f32; 4];
        let _: () = GatedKvSlice::apply(&r, &budget, 0.5, &[], &mut idx, &mut out);
        // Touch out so it isn't optimized away.
        assert!(out.iter().any(|b| b.is_finite()));
    }
}
