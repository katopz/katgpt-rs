//! SubstrateGate DDTree integration — capability-aware branch scoring (Plan 216 T6-T7).
//!
//! Extends DDTree branch scoring with substrate recovery:
//! `score = logprob × sigmoid(recovery) × constraint_validity`
//!
//! Each branch can specify a capability name and use a different SubstrateMask
//! for the forward pass, enabling capability-routed speculative decoding.

use super::substrate_types::SubstrateMask;

// ── sigmoid helper ──────────────────────────────────────────────

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── SubstrateBranch ────────────────────────────────────────────

/// A DDTree branch associated with a capability substrate.
///
/// Each branch carries a SubstrateMask that will be applied during
/// the forward pass, routing computation through the capability's
/// sparse MLP channels.
#[derive(Clone, Debug)]
pub struct SubstrateBranch {
    /// Human-readable capability name.
    pub capability_name: String,
    /// The substrate mask for this branch's forward pass.
    pub mask: SubstrateMask,
    /// Log probability of the branch (from draft model).
    pub logprob: f32,
    /// Constraint validity score [0, 1] — how well the branch satisfies constraints.
    pub constraint_validity: f32,
}

impl SubstrateBranch {
    /// Create a new substrate branch.
    pub fn new(
        capability_name: String,
        mask: SubstrateMask,
        logprob: f32,
        constraint_validity: f32,
    ) -> Self {
        Self {
            capability_name,
            mask,
            logprob,
            constraint_validity,
        }
    }

    /// Compute the substrate-aware branch score.
    ///
    /// `score = logprob × sigmoid(recovery) × constraint_validity`
    ///
    /// Uses sigmoid (never softmax) per project conventions.
    pub fn score(&self) -> f32 {
        substrate_branch_score(
            self.logprob,
            self.mask.recovery_score(),
            self.constraint_validity,
        )
    }
}

// ── substrate_branch_score ─────────────────────────────────────

/// Compute substrate-aware branch score.
///
/// `score = logprob × sigmoid(recovery) × constraint_validity`
///
/// - `logprob`: draft model log probability (typically negative)
/// - `recovery`: substrate mask recovery score [0, 1]
/// - `constraint_validity`: constraint satisfaction score [0, 1]
///
/// Uses sigmoid (not softmax) to bound recovery contribution.
pub fn substrate_branch_score(logprob: f32, recovery: f32, constraint_validity: f32) -> f32 {
    logprob * sigmoid(recovery * 10.0 - 5.0) * constraint_validity
}

// ── Branch expansion ───────────────────────────────────────────

/// Result of substrate-aware branch expansion.
#[derive(Debug)]
pub struct ExpansionResult {
    /// Branches sorted by score (descending).
    pub branches: Vec<SubstrateBranch>,
    /// Number of branches that had sufficient recovery.
    pub viable_count: usize,
    /// Best capability name (highest score).
    pub best_capability: Option<String>,
}

/// Expand candidate branches with substrate masks.
///
/// Given a set of capability masks and draft log probabilities,
/// scores each branch and returns them sorted by score.
///
/// # Arguments
/// * `masks` — available substrate masks with capability names
/// * `logprobs` — draft log probabilities for each candidate
/// * `constraint_validity` — constraint scores (default 1.0 if unconstrained)
/// * `min_recovery` — minimum recovery score for a branch to be viable
pub fn expand_substrate_branches(
    masks: &[(String, SubstrateMask)],
    logprobs: &[f32],
    constraint_validity: &[f32],
    min_recovery: f32,
) -> ExpansionResult {
    let mut branches: Vec<SubstrateBranch> = Vec::with_capacity(masks.len());
    let mut viable_count = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_capability = None;

    for (cap_name, mask) in masks {
        let recovery = mask.recovery_score();
        let cv = constraint_validity.first().copied().unwrap_or(1.0);
        let logprob = logprobs.first().copied().unwrap_or(0.0);

        let branch = SubstrateBranch::new(cap_name.clone(), mask.clone(), logprob, cv);

        let score = branch.score();
        if recovery >= min_recovery {
            viable_count += 1;
        }

        if score > best_score {
            best_score = score;
            best_capability = Some(cap_name.clone());
        }

        branches.push(branch);
    }

    // Sort by score descending
    branches.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    ExpansionResult {
        branches,
        viable_count,
        best_capability,
    }
}

/// Select the best branch from an expansion result.
///
/// Returns None if no viable branches exist.
pub fn select_best_branch(result: &ExpansionResult, min_recovery: f32) -> Option<&SubstrateBranch> {
    result
        .branches
        .iter()
        .find(|b| b.mask.recovery_score() >= min_recovery)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mask(name: &str, recovery: f32) -> SubstrateMask {
        let mut mask = SubstrateMask::new(1, 64, name.to_string(), "model".to_string());
        mask.set(0, 10);
        mask.set_recovery_score(recovery);
        mask
    }

    #[test]
    fn test_substrate_branch_score_sigmoid() {
        // sigmoid(0) = 0.5 → score = 1.0 * 0.5 * 1.0 = 0.5
        let score = substrate_branch_score(1.0, 0.5, 1.0);
        assert!(
            (score - 0.5).abs() < 0.01,
            "sigmoid(0.5*10-5=0) = 0.5, expected ~0.5, got {}",
            score,
        );
    }

    #[test]
    fn test_branch_score_high_recovery() {
        // High recovery → sigmoid → ~1.0
        let score = substrate_branch_score(1.0, 0.9, 1.0);
        assert!(
            score > 0.9,
            "high recovery should give score close to logprob, got {}",
            score,
        );
    }

    #[test]
    fn test_branch_score_zero_recovery() {
        // Very low recovery → sigmoid → ~0
        let score = substrate_branch_score(1.0, 0.0, 1.0);
        assert!(
            score < 0.01,
            "zero recovery should give near-zero score, got {}",
            score,
        );
    }

    #[test]
    fn test_branch_score_constraint_validity() {
        let score_full = substrate_branch_score(1.0, 0.5, 1.0);
        let score_half = substrate_branch_score(1.0, 0.5, 0.5);
        assert!(
            (score_full - score_half * 2.0).abs() < 0.01,
            "halving constraint_validity should halve score",
        );
    }

    #[test]
    fn test_branch_score_negative_logprob() {
        let score = substrate_branch_score(-2.0, 0.5, 1.0);
        assert!(score < 0.0, "negative logprob should give negative score");
    }

    #[test]
    fn test_substrate_branch_new() {
        let mask = make_mask("test", 0.8);
        let branch = SubstrateBranch::new("test".to_string(), mask, -1.5, 0.9);
        assert_eq!(branch.capability_name, "test");
        assert!((branch.logprob - (-1.5)).abs() < 0.001);
        assert!((branch.constraint_validity - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_expand_branches_sorted() {
        let masks = vec![
            ("low".to_string(), make_mask("low", 0.3)),
            ("mid".to_string(), make_mask("mid", 0.6)),
            ("high".to_string(), make_mask("high", 0.9)),
        ];
        let logprobs = vec![1.0];
        let cv = vec![1.0];

        let result = expand_substrate_branches(&masks, &logprobs, &cv, 0.0);

        assert_eq!(result.branches.len(), 3);
        // Should be sorted by score (high recovery → highest score with positive logprob)
        assert_eq!(result.branches[0].capability_name, "high");
        assert_eq!(result.branches[1].capability_name, "mid");
        assert_eq!(result.branches[2].capability_name, "low");
    }

    #[test]
    fn test_expand_branches_viable_count() {
        let masks = vec![
            ("low".to_string(), make_mask("low", 0.2)),
            ("good".to_string(), make_mask("good", 0.7)),
            ("best".to_string(), make_mask("best", 0.9)),
        ];

        let result = expand_substrate_branches(&masks, &[-1.0], &[1.0], 0.5);
        assert_eq!(result.viable_count, 2); // good and best above 0.5
    }

    #[test]
    fn test_expand_branches_best_capability() {
        let masks = vec![
            ("a".to_string(), make_mask("a", 0.3)),
            ("b".to_string(), make_mask("b", 0.9)),
        ];

        let result = expand_substrate_branches(&masks, &[1.0], &[1.0], 0.0);
        assert_eq!(result.best_capability.as_deref(), Some("b"));
    }

    #[test]
    fn test_select_best_branch() {
        let masks = vec![
            ("low".to_string(), make_mask("low", 0.2)),
            ("high".to_string(), make_mask("high", 0.9)),
        ];

        let result = expand_substrate_branches(&masks, &[-1.0], &[1.0], 0.0);
        let best = select_best_branch(&result, 0.5);

        assert!(best.is_some());
        assert_eq!(best.unwrap().capability_name, "high");
    }

    #[test]
    fn test_select_best_branch_none_viable() {
        let masks = vec![("low".to_string(), make_mask("low", 0.1))];

        let result = expand_substrate_branches(&masks, &[-1.0], &[1.0], 0.5);
        let best = select_best_branch(&result, 0.5);
        assert!(best.is_none());
    }

    #[test]
    fn test_expand_empty_masks() {
        let result = expand_substrate_branches(&[], &[], &[], 0.0);
        assert!(result.branches.is_empty());
        assert_eq!(result.viable_count, 0);
        assert!(result.best_capability.is_none());
    }
}
