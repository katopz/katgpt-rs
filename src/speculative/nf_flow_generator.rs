//! FlowScoredGenerator — SpeculativeGenerator wrapper that re-ranks by NF flow score (Plan 229 T2).
//!
//! Decorator pattern: wraps any `SpeculativeGenerator<Condition=TokenCondition, Output=TokenOutput>`
//! and provides `generate_scored()` which returns candidates sorted by descending flow score.
//!
//! The base `generate()` / `generate_batch()` methods delegate to the inner generator unchanged,
//! preserving trait compatibility. Flow scoring is opt-in.

use katgpt_core::SpeculativeGenerator;

use super::nf_flow::NfFlowScore;
use super::spec_generator::{TokenCondition, TokenGenError, TokenOutput};

// ── Types ──────────────────────────────────────────────────────

/// Token output augmented with flow score.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredToken {
    /// Original token output from inner generator.
    pub token: TokenOutput,
    /// NF flow score for this candidate trajectory.
    pub flow_score: f32,
}

/// Errors from `FlowScoredGenerator`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowScoredError {
    /// Inner generator error.
    InnerError(TokenGenError),
    /// No marginals history provided for scoring.
    NoMarginalsHistory,
}

// ── FlowScoredGenerator ────────────────────────────────────────

/// SpeculativeGenerator wrapper that re-ranks candidates by NF flow score (Plan 229 T2).
///
/// Wraps any `SpeculativeGenerator<Condition=TokenCondition, Output=TokenOutput>`
/// and re-ranks its output candidates by flow_score instead of base log-probability.
///
/// This is a decorator pattern: the inner generator produces candidates as before,
/// but `generate_scored()` returns them sorted by descending flow score.
pub struct FlowScoredGenerator<G> {
    inner: G,
    scorer: NfFlowScore,
}

impl<G> FlowScoredGenerator<G> {
    /// Wrap an existing generator with flow-score re-ranking.
    #[inline]
    pub fn new(inner: G) -> Self {
        Self {
            inner,
            scorer: NfFlowScore::new(),
        }
    }

    /// Access inner generator.
    #[inline]
    pub fn inner(&self) -> &G {
        &self.inner
    }

    /// Mutable access to inner generator.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut G {
        &mut self.inner
    }
}

impl<G> FlowScoredGenerator<G>
where
    G: SpeculativeGenerator<
            Condition = TokenCondition,
            Output = TokenOutput,
            Error = TokenGenError,
        >,
{
    /// Generate candidates from inner generator, score each by flow score,
    /// and return them sorted by descending flow score.
    ///
    /// `marginals_history` provides the per-position marginal distributions
    /// used to compute flow scores for each candidate trajectory.
    pub fn generate_scored(
        &mut self,
        condition: &TokenCondition,
        marginals_history: &[Vec<f32>],
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<ScoredToken>, FlowScoredError> {
        if marginals_history.is_empty() {
            return Err(FlowScoredError::NoMarginalsHistory);
        }

        let candidates = self
            .inner
            .generate(condition, rng)
            .map_err(FlowScoredError::InnerError)?;

        // Build selected trajectory for each candidate: parent tokens + this candidate.
        let mut scored: Vec<ScoredToken> = candidates
            .into_iter()
            .map(|token| {
                let mut selected: Vec<usize> = condition.parent_tokens.clone();
                selected.push(token.token_idx);
                let fs = self.scorer.score(marginals_history, &selected);
                ScoredToken {
                    token,
                    flow_score: fs,
                }
            })
            .collect();

        // Sort by descending flow score.
        scored.sort_unstable_by(|a, b| {
            b.flow_score
                .partial_cmp(&a.flow_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored)
    }
}

// ── SpeculativeGenerator delegation ────────────────────────────

impl<G> SpeculativeGenerator for FlowScoredGenerator<G>
where
    G: SpeculativeGenerator<
            Condition = TokenCondition,
            Output = TokenOutput,
            Error = TokenGenError,
        >,
{
    type Condition = TokenCondition;
    type Output = TokenOutput;
    type Error = TokenGenError;

    #[inline]
    fn generate(
        &mut self,
        condition: &Self::Condition,
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Self::Output>, Self::Error> {
        self.inner.generate(condition, rng)
    }

    #[inline]
    fn generate_batch(
        &mut self,
        conditions: &[Self::Condition],
        rng: &mut fastrand::Rng,
    ) -> Result<Vec<Vec<Self::Output>>, Self::Error> {
        self.inner.generate_batch(conditions, rng)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock inner generator that returns deterministic candidates.
    struct MockGenerator;

    impl SpeculativeGenerator for MockGenerator {
        type Condition = TokenCondition;
        type Output = TokenOutput;
        type Error = TokenGenError;

        fn generate(
            &mut self,
            _condition: &Self::Condition,
            _rng: &mut fastrand::Rng,
        ) -> Result<Vec<Self::Output>, Self::Error> {
            // Returns candidates with different log-probs: 0.5, 0.9, 0.3
            Ok(vec![
                TokenOutput {
                    token_idx: 0,
                    log_prob: 0.5,
                },
                TokenOutput {
                    token_idx: 1,
                    log_prob: 0.9,
                },
                TokenOutput {
                    token_idx: 2,
                    log_prob: 0.3,
                },
            ])
        }
    }

    fn make_condition() -> TokenCondition {
        TokenCondition {
            parent_tokens: vec![],
            depth: 0,
            marginals: vec![0.3, 0.5, 0.2],
        }
    }

    #[test]
    fn test_flow_scored_generator_reorders() {
        // Marginals with different entropies so flow scores differ from base log-prob.
        let marginals_history = vec![
            vec![0.9, 0.05, 0.05], // peaked at token 0
            vec![0.3, 0.5, 0.2],   // current position
        ];
        let mut wrapper = FlowScoredGenerator::new(MockGenerator);
        let mut rng = fastrand::Rng::new();
        let scored = wrapper
            .generate_scored(&make_condition(), &marginals_history, &mut rng)
            .unwrap();

        // Verify reordered: not in original [0.5, 0.9, 0.3] base order.
        // Token 0 gets highest flow score because marginals[0] is peaked at 0.
        assert!(scored.len() > 1);
        // At least one adjacent pair should differ from base log-prob ordering.
        let has_reorder = scored.windows(2).any(|w| {
            w[0].token.log_prob < w[1].token.log_prob && w[0].flow_score > w[1].flow_score
        });
        assert!(
            has_reorder,
            "flow scoring should reorder candidates relative to base log-prob"
        );
    }

    #[test]
    fn test_flow_scored_generator_preserves_all_candidates() {
        let marginals_history = vec![vec![0.3, 0.5, 0.2]];
        let mut wrapper = FlowScoredGenerator::new(MockGenerator);
        let mut rng = fastrand::Rng::new();
        let scored = wrapper
            .generate_scored(&make_condition(), &marginals_history, &mut rng)
            .unwrap();

        assert_eq!(scored.len(), 3);
    }

    #[test]
    fn test_flow_scored_generator_delegates() {
        let mut wrapper = FlowScoredGenerator::new(MockGenerator);
        let mut rng = fastrand::Rng::new();
        let result = wrapper.generate(&make_condition(), &mut rng).unwrap();

        // Should return inner generator's output without scoring.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].token_idx, 0);
        assert_eq!(result[1].token_idx, 1);
        assert_eq!(result[2].token_idx, 2);
    }

    #[test]
    fn test_scored_token_fields() {
        let marginals_history = vec![vec![0.9, 0.05, 0.05], vec![0.3, 0.5, 0.2]];
        let mut wrapper = FlowScoredGenerator::new(MockGenerator);
        let mut rng = fastrand::Rng::new();
        let scored = wrapper
            .generate_scored(&make_condition(), &marginals_history, &mut rng)
            .unwrap();

        // Every scored token should have a valid flow score (finite).
        for st in &scored {
            assert!(st.flow_score.is_finite(), "flow_score should be finite");
            assert_eq!(st.token.token_idx, st.token.token_idx); // field exists
        }
    }

    #[test]
    fn test_flow_scored_generator_no_marginals() {
        let mut wrapper = FlowScoredGenerator::new(MockGenerator);
        let mut rng = fastrand::Rng::new();
        let result = wrapper.generate_scored(&make_condition(), &[], &mut rng);

        assert_eq!(result, Err(FlowScoredError::NoMarginalsHistory));
    }
}

// TL;DR: FlowScoredGenerator wraps a SpeculativeGenerator and re-ranks candidates by NF flow
// score via generate_scored(). Trait methods delegate unchanged. 5 tests: reorder, preserve
// count, delegation, ScoredToken fields, and no-marginals error.
