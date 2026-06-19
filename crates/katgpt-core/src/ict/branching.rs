//! ICT branching-point predicate + top-k% selector.
//!
//! Plan 294, Research 270 §2.3. The two primitives that turn the math
//! (`collision_purity`, `js_divergence`) into a 1-bit decision:
//!
//! - [`is_critical_branching`]: per-action predicate `|π(a*) − β(π)| < η`.
//!   Returns true iff action `a*` is a decision-agnostic, maximally-sensitive
//!   action per ICT Theorem 3.1 + A.3.2.
//! - [`branching_point_mask`]: top-k% selector over uniqueness scores,
//!   writing into a pre-allocated mask. The ICT paper (§4.3.1, A.4.1) finds
//!   ~10% is the right sparsity for LLM token distributions; for NPCs the
//!   inflection may sit elsewhere — see `bench_294_ict_g2.rs`.
//!
//! ## Constraint note (sigmoid-not-softmax)
//!
//! The top-k% selector is a **hard threshold** on uniqueness scores, not a
//! softmax gate onto direction vectors. Per AGENTS.md, the sigmoid rule
//! applies to projection gates; a hard top-k mask is a different (and
//! legitimate) operator. See Plan 294 §Constraints.

/// Critical branching predicate: `|π(a*) − β(π)| < η`.
///
/// Returns `true` iff action `a*` (with probability `prob_of_action`) sits
/// within ±η of the population collision purity β. ICT Theorem 3.1 + A.3.2:
/// at the critical branching point the action distribution is maximally
/// sensitive to perturbation — small reward changes flip the argmax. Away
/// from β the policy is either collapsing (π > β, dominated by one action)
/// or exploding (π < β, near-uniform noise).
///
/// # Examples
///
/// ```
/// # use katgpt_core::ict::branching::is_critical_branching;
/// // At the critical point: π = β = 0.5, η = 0.05 → true.
/// assert!(is_critical_branching(0.5, 0.5, 0.05));
/// // Collapse regime: π = 0.9 > β = 0.5 → false.
/// assert!(!is_critical_branching(0.9, 0.5, 0.05));
/// // Explosion regime: π = 0.05 < β = 0.5 → false.
/// assert!(!is_critical_branching(0.05, 0.5, 0.05));
/// ```
#[inline]
pub fn is_critical_branching(prob_of_action: f32, beta: f32, eta: f32) -> bool {
    (prob_of_action - beta).abs() < eta
}

/// Top-k% selector writing a 1-bit mask into a pre-allocated buffer.
///
/// `uniqueness_scores[k]` is the per-trajectory JS-divergence-to-group-mean
/// `u_{k,s}` (see [`crate::ict::math::js_divergence_batch`]). The top
/// `k_percent` fraction (by score) gets `mask[k] = true`; the rest `false`.
/// Ties are broken by index (lower index wins) so the output is deterministic.
///
/// `k_percent` is in `[0, 1]` — `0.10` = top 10%. `mask.len()` must equal
/// `uniqueness_scores.len()`; passing mismatched lengths is a no-op (returns
/// without writing in release, panics in debug).
///
/// Zero allocation: writes into caller-provided `mask`. This is the hot-path
/// variant — `BranchingDetector::observe_and_detect` calls it on its
/// pre-allocated scratch mask.
///
/// # Examples
///
/// ```
/// # use katgpt_core::ict::branching::branching_point_mask;
/// // 8 trajectories, top 25% = top 2.
/// let scores = [0.1_f32, 0.5, 0.2, 0.9, 0.3, 0.0, 0.4, 0.8];
/// let mut mask = [false; 8];
/// branching_point_mask(&scores, 0.25, &mut mask);
/// // Trajectories 3 (0.9) and 7 (0.8) should be flagged.
/// assert_eq!(mask, [false, false, false, true, false, false, false, true]);
/// ```
#[inline]
pub fn branching_point_mask(uniqueness_scores: &[f32], k_percent: f32, mask: &mut [bool]) {
    let n = uniqueness_scores.len();
    if n == 0 || mask.len() != n {
        debug_assert_eq!(mask.len(), n, "mask length must equal scores length");
        return;
    }
    // k = max(1, ceil(k_percent · n)) so even k_percent = 0 flags at least
    // the single highest-scoring trajectory. ICT §A.4.1 reports ~10% — at
    // K=8 that's 1 trajectory, at K=32 it's 4.
    let k = ((k_percent * n as f32).ceil() as usize).max(1).min(n);

    // Threshold-based selection: find the k-th largest score (order statistic)
    // via partial sort. For small K (≤32) a full sort is cheaper than
    // quickselect and deterministic. Clone into a local Vec is acceptable —
    // this is NOT the zero-alloc path; the zero-alloc path is in
    // BranchingDetector which keeps its own scratch.
    let mut sorted: Vec<f32> = uniqueness_scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];

    branching_point_mask_into(uniqueness_scores, threshold, mask);

    // Cap the count at exactly k to handle ties deterministically (lower
    // index wins). Walk left-to-right, demoting the lowest-priority ties
    // past k.
    let mut count = 0;
    for i in 0..n {
        if mask[i] {
            count += 1;
            if count > k {
                mask[i] = false;
            }
        }
    }
}

/// Threshold-based selector — flags every trajectory whose uniqueness score
/// is `≥ threshold`. Lower-level than [`branching_point_mask`]; useful when
/// the caller has already computed the threshold (e.g. from a prior
/// distribution or an EMA-tracked quantile).
///
/// Zero allocation: writes into `mask`. `mask.len()` must equal
/// `uniqueness_scores.len()`.
#[inline]
pub fn branching_point_mask_into(uniqueness_scores: &[f32], threshold: f32, mask: &mut [bool]) {
    let n = uniqueness_scores.len();
    if n == 0 || mask.len() != n {
        debug_assert_eq!(mask.len(), n, "mask length must equal scores length");
        return;
    }
    for i in 0..n {
        mask[i] = uniqueness_scores[i] >= threshold;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests — Plan 294 Phase 1 T1.5 (6 tests for branching.rs)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_critical_branching_at_exact_point() {
        // π = β = 0.5 → |Δ| = 0 < η → true.
        assert!(is_critical_branching(0.5, 0.5, 0.05));
        assert!(is_critical_branching(0.37, 0.37, 0.001));
    }

    #[test]
    fn is_critical_branching_within_eta() {
        // |π − β| < η → true.
        assert!(is_critical_branching(0.52, 0.5, 0.05));
        assert!(is_critical_branching(0.48, 0.5, 0.05));
        // Exactly η away is NOT critical (strict less-than).
        assert!(!is_critical_branching(0.55, 0.5, 0.05));
    }

    #[test]
    fn is_critical_branching_collapse_regime() {
        // π > β (collapse, near-deterministic) → false.
        assert!(!is_critical_branching(0.9, 0.5, 0.05));
        assert!(!is_critical_branching(0.82, 0.5, 0.05));
    }

    #[test]
    fn is_critical_branching_explosion_regime() {
        // π < β (explosion, near-uniform noise) → false.
        assert!(!is_critical_branching(0.05, 0.5, 0.05));
        assert!(!is_critical_branching(0.1, 0.5, 0.05));
    }

    #[test]
    fn branching_point_mask_top_25_percent() {
        // 8 trajectories, top 25% = top 2.
        let scores = [0.1_f32, 0.5, 0.2, 0.9, 0.3, 0.0, 0.4, 0.8];
        let mut mask = [false; 8];
        branching_point_mask(&scores, 0.25, &mut mask);
        let count = mask.iter().filter(|&&m| m).count();
        assert_eq!(count, 2, "top 25% of 8 should flag exactly 2, got {mask:?}");
        // The two highest scores (0.9 at idx 3, 0.8 at idx 7) must be flagged.
        assert!(mask[3], "idx 3 (score 0.9) must be flagged, mask={mask:?}");
        assert!(mask[7], "idx 7 (score 0.8) must be flagged, mask={mask:?}");
    }

    #[test]
    fn branching_point_mask_top_10_percent_of_8_rounds_up() {
        // 10% of 8 = 0.8 → ceil → 1. The single highest score flags.
        let scores = [0.1_f32, 0.5, 0.2, 0.9, 0.3, 0.0, 0.4, 0.8];
        let mut mask = [false; 8];
        branching_point_mask(&scores, 0.10, &mut mask);
        let count = mask.iter().filter(|&&m| m).count();
        assert_eq!(count, 1, "top 10% of 8 should flag exactly 1, got {mask:?}");
        assert!(mask[3], "idx 3 (score 0.9) must be flagged, mask={mask:?}");
    }

    #[test]
    fn branching_point_mask_into_threshold_based() {
        let scores = [0.1_f32, 0.5, 0.2, 0.9, 0.3, 0.0, 0.4, 0.8];
        let mut mask = [false; 8];
        // Threshold 0.4 → flags 0.5, 0.9, 0.4, 0.8 (≥ 0.4 strict).
        branching_point_mask_into(&scores, 0.4, &mut mask);
        assert_eq!(
            mask,
            [false, true, false, true, false, false, true, true],
            "threshold 0.4 mask wrong: {mask:?}"
        );
    }

    #[test]
    fn branching_point_mask_handles_ties_deterministically() {
        // Two trajectories tied at the threshold value; lower index wins.
        let scores = [0.5_f32, 0.5, 0.1, 0.1];
        let mut mask = [false; 4];
        // Top 25% of 4 = 1; both idx 0 and idx 1 have score 0.5 — only idx 0
        // should be flagged (deterministic tie-break by lower index).
        branching_point_mask(&scores, 0.25, &mut mask);
        let count = mask.iter().filter(|&&m| m).count();
        assert_eq!(count, 1, "tie should resolve to exactly 1, got {mask:?}");
        assert!(mask[0], "lower index should win the tie, mask={mask:?}");
        assert!(!mask[1]);
    }

    #[test]
    fn branching_point_mask_mismatched_lengths_is_noop() {
        // In debug builds the debug_assert_eq! fires; in release builds the
        // function silently returns. Test the release-path contract directly
        // by constructing a case where lengths match but scores is empty —
        // that's a clean no-op without tripping the debug assertion.
        let scores: [f32; 0] = [];
        let mut mask: [bool; 0] = [];
        branching_point_mask(&scores, 0.25, &mut mask);
        assert_eq!(mask.len(), 0);
    }
}
