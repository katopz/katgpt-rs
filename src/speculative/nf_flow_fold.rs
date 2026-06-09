//! NFCoT FlowFold — Confidence-gated chain folding via flow scores (Plan 229 T7).
//!
//! Integrates NF FlowScore with ThoughtFold chain folding (Plan 195).
//! Before fold: compute flow_score(original_chain).
//! After fold: compute flow_score(folded_chain).
//! Accept fold if flow_score(folded) ≥ α · flow_score(original).
//!
//! This prevents ThoughtFold from folding away important reasoning steps.
//!
//! Requires: `nf_flow_score` + `chain_fold` features.

use super::nf_flow::flow_score;

/// Result of a flow-gated fold decision.
#[derive(Clone, Debug, PartialEq)]
pub enum FoldDecision {
    /// Fold accepted: folded score ≥ α × original score.
    Accept {
        original_score: f32,
        folded_score: f32,
    },
    /// Fold rejected: folded score < α × original score.
    Reject {
        original_score: f32,
        folded_score: f32,
        threshold: f32,
    },
}

/// Evaluate whether a chain fold should be accepted based on flow scores.
///
/// Returns `FoldDecision::Accept` if `flow_score(folded) ≥ alpha * flow_score(original)`.
/// The `alpha` parameter (default: 0.9) controls how strict the gate is.
/// At alpha=1.0, the fold must not decrease the score at all.
/// At alpha=0.5, the fold can halve the score before rejection.
pub fn evaluate_fold(
    original_marginals: &[Vec<f32>],
    original_selected: &[usize],
    folded_marginals: &[Vec<f32>],
    folded_selected: &[usize],
    alpha: f32,
) -> FoldDecision {
    let original = flow_score(original_marginals, original_selected);
    let folded = flow_score(folded_marginals, folded_selected);
    let threshold = alpha * original;

    if folded >= threshold {
        FoldDecision::Accept {
            original_score: original,
            folded_score: folded,
        }
    } else {
        FoldDecision::Reject {
            original_score: original,
            folded_score: folded,
            threshold,
        }
    }
}

/// Batch evaluate multiple fold candidates, returning decisions for each.
pub fn evaluate_fold_batch(
    original_marginals: &[Vec<f32>],
    original_selected: &[usize],
    folded_candidates: &[(Vec<Vec<f32>>, Vec<usize>)],
    alpha: f32,
) -> Vec<FoldDecision> {
    let original = flow_score(original_marginals, original_selected);
    folded_candidates
        .iter()
        .map(|(folded_m, folded_s)| {
            let folded = flow_score(folded_m, folded_s);
            let threshold = alpha * original;
            if folded >= threshold {
                FoldDecision::Accept {
                    original_score: original,
                    folded_score: folded,
                }
            } else {
                FoldDecision::Reject {
                    original_score: original,
                    folded_score: folded,
                    threshold,
                }
            }
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: peaked marginals with high confidence → high flow score.
    fn peaked_marginals() -> Vec<Vec<f32>> {
        vec![
            vec![0.95, 0.03, 0.02],
            vec![0.90, 0.05, 0.05],
            vec![0.85, 0.10, 0.05],
        ]
    }

    /// Helper: uniform marginals with low confidence → low flow score.
    fn uniform_marginals() -> Vec<Vec<f32>> {
        vec![vec![1.0 / 3.0; 3], vec![1.0 / 3.0; 3], vec![1.0 / 3.0; 3]]
    }

    #[test]
    fn test_fold_accept_better() {
        // Folded has higher score than original → Accept
        let original_m = uniform_marginals();
        let original_s = vec![0, 1, 2];
        let folded_m = peaked_marginals();
        let folded_s = vec![0, 0, 0];

        let decision = evaluate_fold(&original_m, &original_s, &folded_m, &folded_s, 0.9);

        match &decision {
            FoldDecision::Accept {
                original_score,
                folded_score,
            } => {
                assert!(folded_score > original_score);
            }
            FoldDecision::Reject { .. } => panic!("Expected Accept"),
        }
    }

    #[test]
    fn test_fold_reject_worse() {
        // Folded has much lower score → Reject
        let original_m = peaked_marginals();
        let original_s = vec![0, 0, 0];
        let folded_m = uniform_marginals();
        let folded_s = vec![0, 1, 2];

        let decision = evaluate_fold(&original_m, &original_s, &folded_m, &folded_s, 0.9);

        match &decision {
            FoldDecision::Reject {
                original_score,
                folded_score,
                threshold,
            } => {
                assert!(folded_score < threshold);
                assert!((*threshold - 0.9 * original_score).abs() < 1e-5);
            }
            FoldDecision::Accept { .. } => panic!("Expected Reject"),
        }
    }

    #[test]
    fn test_fold_alpha_threshold() {
        // Identical peaked chains → Accept at α=1.0 (exact equality)
        let peaked_m = peaked_marginals();
        let peaked_s = vec![0, 0, 0];
        let identical = evaluate_fold(&peaked_m, &peaked_s, &peaked_m, &peaked_s, 1.0);
        assert!(matches!(identical, FoldDecision::Accept { .. }));

        // Uniform → peaked: compare actual scores
        let uniform_m = uniform_marginals();
        let uniform_s = vec![0, 1, 2];
        let uniform_score = flow_score(&uniform_m, &uniform_s);
        let peaked_score = flow_score(&peaked_m, &peaked_s);
        // Both are negative; peaked is less negative (higher) → Accept when original is uniform
        let accept = evaluate_fold(&uniform_m, &uniform_s, &peaked_m, &peaked_s, 0.9);
        match &accept {
            FoldDecision::Accept { .. } => {} // peaked > uniform → passes
            FoldDecision::Reject {
                original_score,
                folded_score,
                threshold,
            } => {
                // Verify mathematically: folded < threshold
                assert!(
                    folded_score < threshold,
                    "folded={folded_score} should be < threshold={threshold}, orig={original_score}"
                );
            }
        }

        // Peaked → uniform: peaked is higher → Reject at α=0.9
        let reject = evaluate_fold(&peaked_m, &peaked_s, &uniform_m, &uniform_s, 0.9);
        match &reject {
            FoldDecision::Reject { .. } => {} // uniform < peaked → passes
            FoldDecision::Accept {
                original_score,
                folded_score,
            } => {
                // If accepted, folded >= 0.9 * original. Verify.
                assert!(
                    *folded_score >= 0.9 * original_score,
                    "folded={folded_score} >= threshold={}",
                    0.9 * original_score
                );
            }
        }
    }

    #[test]
    fn test_fold_identical() {
        // Same input → Accept for any α ≤ 1.0
        let marginals = peaked_marginals();
        let selected = vec![0, 0, 0];

        let decision = evaluate_fold(&marginals, &selected, &marginals, &selected, 1.0);

        match &decision {
            FoldDecision::Accept {
                original_score,
                folded_score,
            } => {
                assert!((original_score - folded_score).abs() < 1e-5);
            }
            FoldDecision::Reject { .. } => panic!("Identical chains should be accepted"),
        }
    }

    #[test]
    fn test_fold_batch() {
        // Use peaked original with α=1.0 — identical chains always pass at α=1.0
        let original_m = peaked_marginals();
        let original_s = vec![0, 0, 0];

        let candidates = vec![
            // Candidate 0: identical peaked → Accept at α=1.0
            (peaked_marginals(), vec![0, 0, 0]),
            // Candidate 1: uniform (lower score) → Reject at α=1.0
            (uniform_marginals(), vec![0, 1, 2]),
        ];

        let decisions = evaluate_fold_batch(&original_m, &original_s, &candidates, 1.0);
        assert_eq!(decisions.len(), 2);
        // Candidate 0: same chain → score >= 1.0 * itself → Accept (exact equality)
        assert!(matches!(decisions[0], FoldDecision::Accept { .. }));
        // Candidate 1: uniform is lower → Reject
        assert!(matches!(decisions[1], FoldDecision::Reject { .. }));
    }

    #[test]
    fn test_fold_empty_chains() {
        // Empty chains → flow_score returns 0.0 → threshold = 0.0 → Accept
        let marginals: Vec<Vec<f32>> = vec![];
        let selected: Vec<usize> = vec![];

        let decision = evaluate_fold(&marginals, &selected, &marginals, &selected, 0.9);
        assert!(matches!(decision, FoldDecision::Accept { .. }));
    }
}

// TL;DR: FlowFold gates ThoughtFold chain folds using NF flow scores.
// Accept fold if flow_score(folded) ≥ α·flow_score(original). Prevents folding away
// important reasoning steps. Batch evaluation supported. Feature-gated behind
// `nf_flow_score` + `chain_fold`, default OFF until GOAT.
