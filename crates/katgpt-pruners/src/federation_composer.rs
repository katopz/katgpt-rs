//! Deep Manifold §7.5 — Federation Triangle Composer (Research 205, Plan 231)
//!
//! Paper Eq. 158-159: Agentic behavior = composite fixed-point iteration:
//!   x_{t+1} = Φ_tool ∘ Φ_agent ∘ Φ_model(x_t)
//!
//! Model  = ConstraintPruner (what's valid)
//! Agent  = ScreeningPruner (what's relevant)
//! Tool   = BanditPruner    (what works)
//!
//! With RESIDUAL CHECKING between each step for early termination.

use katgpt_core::traits::{ConstraintPruner, ScreeningPruner};

/// Residual check result after each federation step.
#[derive(Debug, Clone)]
pub struct ResidualCheck {
    pub candidates_before: usize,
    pub candidates_after: usize,
    /// 1 - (after/before). High = step removed many. Low = step barely changed.
    pub residual: f32,
}

impl ResidualCheck {
    pub fn new(before: usize, after: usize) -> Self {
        let residual = if before > 0 {
            1.0 - (after as f32 / before as f32)
        } else {
            0.0
        };
        Self {
            candidates_before: before,
            candidates_after: after,
            residual,
        }
    }
    /// Low residual = step didn't help much → consider stopping.
    pub fn should_terminate(&self, threshold: f32) -> bool {
        self.residual < threshold
    }
}

/// Federation composer: explicit Model→Agent→Tool pipeline with residual checking.
pub struct FederationComposer<'a, C, S> {
    pub constraint: &'a C,
    pub screening: &'a S,
    pub residual_threshold: f32,
}

impl<'a, C: ConstraintPruner, S: ScreeningPruner> FederationComposer<'a, C, S> {
    pub fn new(constraint: &'a C, screening: &'a S) -> Self {
        Self {
            constraint,
            screening,
            residual_threshold: 0.01,
        }
    }

    /// Run full federation pipeline with residual checking.
    pub fn compose_and_prune(
        &self,
        candidates: &[usize],
        depth: usize,
        parent_token: &[usize],
    ) -> (Vec<usize>, Vec<ResidualCheck>) {
        let mut checks = Vec::with_capacity(2);
        let n = candidates.len();

        // Step 1: Model → ConstraintPruner
        let valid: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&t| self.constraint.is_valid(depth, t, parent_token))
            .collect();
        let c1 = ResidualCheck::new(n, valid.len());
        checks.push(c1.clone());
        if c1.should_terminate(self.residual_threshold) && valid.len() == n {
            return (valid, checks);
        }

        // Step 2: Agent → ScreeningPruner
        let relevant: Vec<usize> = valid
            .iter()
            .copied()
            .filter(|&t| self.screening.relevance(depth, t, parent_token) > 0.5)
            .collect();
        let c2 = ResidualCheck::new(valid.len(), relevant.len());
        checks.push(c2.clone());
        if c2.should_terminate(self.residual_threshold) && relevant.len() == valid.len() {
            return (relevant, checks);
        }

        (relevant, checks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::traits::NoPruner;

    struct HalfConstraintPruner;
    impl ConstraintPruner for HalfConstraintPruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx < 5
        }
    }

    struct HighRelevanceScreener;
    impl ScreeningPruner for HighRelevanceScreener {
        fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            if token_idx < 3 { 0.9 } else { 0.3 }
        }
    }

    struct AllValidScreener;
    impl ScreeningPruner for AllValidScreener {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            1.0
        }
    }

    #[test]
    fn federation_prunes_by_constraint_then_screening() {
        let cp = HalfConstraintPruner;
        let sp = HighRelevanceScreener;
        let composer = FederationComposer::new(&cp, &sp);

        let candidates: Vec<usize> = (0..10).collect();
        let (result, checks) = composer.compose_and_prune(&candidates, 0, &[]);
        assert_eq!(result, vec![0, 1, 2]);
        assert_eq!(checks.len(), 2);
        // Step 1: 10 → 5 (constraint)
        assert_eq!(checks[0].candidates_before, 10);
        assert_eq!(checks[0].candidates_after, 5);
        // Step 2: 5 → 3 (screening)
        assert_eq!(checks[1].candidates_before, 5);
        assert_eq!(checks[1].candidates_after, 3);
    }

    #[test]
    fn residual_check_terminate_low_residual() {
        let rc = ResidualCheck::new(100, 100);
        assert!(rc.should_terminate(0.01));
        assert!((rc.residual - 0.0).abs() < 1e-6);
    }

    #[test]
    fn residual_check_no_terminate_high_residual() {
        let rc = ResidualCheck::new(100, 10);
        assert!(!rc.should_terminate(0.01));
        assert!((rc.residual - 0.9).abs() < 1e-6);
    }

    #[test]
    fn early_termination_when_no_pruning() {
        let cp = NoPruner;
        let sp = AllValidScreener;
        let composer = FederationComposer::new(&cp, &sp);

        let candidates: Vec<usize> = (0..5).collect();
        let (result, checks) = composer.compose_and_prune(&candidates, 0, &[]);
        // Step 1: no pruning (all valid) → low residual → early termination
        assert_eq!(result.len(), 5);
        // Should terminate after step 1
        assert_eq!(checks.len(), 1);
    }

    #[test]
    fn empty_candidates() {
        let cp = NoPruner;
        let sp = AllValidScreener;
        let composer = FederationComposer::new(&cp, &sp);

        let (result, checks) = composer.compose_and_prune(&[], 0, &[]);
        assert!(result.is_empty());
        assert_eq!(checks.len(), 1);
    }
}
