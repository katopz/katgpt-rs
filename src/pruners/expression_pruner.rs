//! ExpressionPruner — ScreeningPruner wrapper using a fitted SymbolicExpression.
//!
//! Extracts features from (depth, token, parents, scores), evaluates a compact
//! symbolic expression, and blends the result with an inner pruner's relevance.
//!
//! **Feature gate:** `symbolic_distill`

use crate::speculative::types::ScreeningPruner;

use super::symbolic_expression::SymbolicExpression;

// ── Feature Extractor ──────────────────────────────────────────

/// Trait for extracting feature vectors from screening context.
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from the current screening context.
    fn extract(
        &self,
        depth: usize,
        token: usize,
        parents: &[usize],
        inner_scores: &[f32],
    ) -> Vec<f32>;

    /// Human-readable names for each feature dimension.
    fn feature_names(&self) -> Vec<&str>;
}

// ── Default Feature Extractor ──────────────────────────────────

/// Default feature extractor producing 5 basic features:
/// 1. `depth` (f32)
/// 2. `token_idx` (f32)
/// 3. `parent_count` (f32)
/// 4. `mean_score` (0.0 if empty)
/// 5. `max_score` (0.0 if empty)
pub struct DefaultFeatureExtractor;

impl FeatureExtractor for DefaultFeatureExtractor {
    fn extract(
        &self,
        depth: usize,
        token: usize,
        parents: &[usize],
        inner_scores: &[f32],
    ) -> Vec<f32> {
        let mean_score = match inner_scores.is_empty() {
            true => 0.0,
            false => inner_scores.iter().sum::<f32>() / inner_scores.len() as f32,
        };

        let max_score = match inner_scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            Some(v) => v,
            None => 0.0,
        };

        vec![
            depth as f32,
            token as f32,
            parents.len() as f32,
            mean_score,
            max_score,
        ]
    }

    fn feature_names(&self) -> Vec<&str> {
        vec!["depth", "token", "parent_count", "mean_score", "max_score"]
    }
}

// ── Expression Pruner ──────────────────────────────────────────

/// ScreeningPruner that blends an inner pruner with a fitted symbolic expression.
///
/// Relevance is computed as: `0.5 * inner_relevance + 0.5 * expr_result`.
/// The expression result is sigmoid-bounded to [0, 1].
pub struct ExpressionPruner<P: ScreeningPruner> {
    inner: P,
    expression: SymbolicExpression,
    feature_extractor: Box<dyn FeatureExtractor>,
}

impl<P: ScreeningPruner> ExpressionPruner<P> {
    /// Create with default feature extractor.
    pub fn new(inner: P, expression: SymbolicExpression) -> Self {
        Self {
            inner,
            expression,
            feature_extractor: Box::new(DefaultFeatureExtractor),
        }
    }

    /// Create with a custom feature extractor.
    pub fn with_extractor(
        inner: P,
        expression: SymbolicExpression,
        extractor: Box<dyn FeatureExtractor>,
    ) -> Self {
        Self {
            inner,
            expression,
            feature_extractor: extractor,
        }
    }
}

impl<P: ScreeningPruner> ScreeningPruner for ExpressionPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Extract features from context — we don't have inner_scores here,
        // so we pass the inner pruner's own relevance as a single-element score.
        let inner_rel = self.inner.relevance(depth, token_idx, parent_tokens);
        let inner_scores = [inner_rel];

        let features =
            self.feature_extractor
                .extract(depth, token_idx, parent_tokens, &inner_scores);
        let expr_result = self.expression.evaluate(&features);

        // Blend: 50/50 inner + expression
        0.5 * inner_rel + 0.5 * expr_result
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::symbolic_expression::{BasisFn, Term};
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    /// A pruner that returns a fixed relevance for testing.
    struct FixedPruner(f32);

    impl ScreeningPruner for FixedPruner {
        fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
            self.0
        }
    }

    #[test]
    fn test_screening_pruner_delegates_correctly() {
        // Expression: 1.0 × identity(depth) + bias 0.0
        // With depth=0, features[0]=0.0 → raw=0.0 → sigmoid(0.0)=0.5
        // inner relevance = 1.0
        // blend = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        let expr = SymbolicExpression {
            terms: vec![Term {
                basis: BasisFn::Identity,
                coefficient: 1.0,
                feature_idx: 0,
            }],
            bias: 0.0,
        };

        let pruner = ExpressionPruner::new(FixedPruner(1.0), expr);
        let result = pruner.relevance(0, 5, &[]);
        let expected = 0.5 * 1.0 + 0.5 * sigmoid(0.0_f32);
        assert!(
            (result - expected).abs() < 1e-5,
            "result={} expected={}",
            result,
            expected
        );
    }

    #[test]
    fn test_feature_extraction_dimensions() {
        let extractor = DefaultFeatureExtractor;
        let features = extractor.extract(3, 7, &[1, 2, 3], &[0.2, 0.5, 0.8]);

        assert_eq!(
            features.len(),
            5,
            "DefaultFeatureExtractor should produce 5 features"
        );
        assert!((features[0] - 3.0).abs() < 1e-6, "depth");
        assert!((features[1] - 7.0).abs() < 1e-6, "token_idx");
        assert!((features[2] - 3.0).abs() < 1e-6, "parent_count");
        // mean of [0.2, 0.5, 0.8] = 0.5
        assert!((features[3] - 0.5).abs() < 1e-6, "mean_score");
        // max of [0.2, 0.5, 0.8] = 0.8
        assert!((features[4] - 0.8).abs() < 1e-6, "max_score");

        let names = extractor.feature_names();
        assert_eq!(names.len(), 5);
    }

    #[test]
    fn test_feature_extraction_empty_scores() {
        let extractor = DefaultFeatureExtractor;
        let features = extractor.extract(0, 0, &[], &[]);

        assert!(
            (features[3] - 0.0).abs() < 1e-6,
            "mean_score should be 0.0 for empty"
        );
        assert!(
            (features[4] - 0.0).abs() < 1e-6,
            "max_score should be 0.0 for empty"
        );
    }

    #[test]
    fn test_expression_pruner_scores_bounded() {
        // Use large coefficients to push expression to extremes
        let expr = SymbolicExpression {
            terms: vec![Term {
                basis: BasisFn::Identity,
                coefficient: 100.0,
                feature_idx: 0, // depth
            }],
            bias: 0.0,
        };

        let pruner = ExpressionPruner::new(FixedPruner(0.5), expr);

        // Test with various depths
        for depth in 0..20 {
            let result = pruner.relevance(depth, 0, &[]);
            assert!(
                (0.0..=1.0).contains(&result),
                "relevance out of [0,1]: depth={} result={}",
                depth,
                result
            );
        }
    }

    #[test]
    fn test_expression_pruner_with_no_screening() {
        // NoScreeningPruner always returns 1.0
        let expr = SymbolicExpression {
            terms: Vec::new(),
            bias: 0.0,
        };

        let pruner = ExpressionPruner::new(NoScreeningPruner, expr);
        // sigmoid(0.0) = 0.5, blend = 0.5 * 1.0 + 0.5 * 0.5 = 0.75
        let result = pruner.relevance(0, 0, &[]);
        assert!((result - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_feature_names_match_extraction() {
        let extractor = DefaultFeatureExtractor;
        let features = extractor.extract(1, 2, &[3], &[0.5]);
        let names = extractor.feature_names();

        assert_eq!(
            features.len(),
            names.len(),
            "feature count must match name count"
        );
    }

    #[test]
    fn test_custom_extractor() {
        struct SingleFeatureExtractor;

        impl FeatureExtractor for SingleFeatureExtractor {
            fn extract(
                &self,
                _depth: usize,
                _token: usize,
                _parents: &[usize],
                _inner_scores: &[f32],
            ) -> Vec<f32> {
                vec![1.0]
            }
            fn feature_names(&self) -> Vec<&str> {
                vec!["constant"]
            }
        }

        let expr = SymbolicExpression {
            terms: vec![Term {
                basis: BasisFn::Identity,
                coefficient: 2.0,
                feature_idx: 0,
            }],
            bias: 1.0,
        };

        let pruner = ExpressionPruner::with_extractor(
            FixedPruner(0.0),
            expr,
            Box::new(SingleFeatureExtractor),
        );

        // features = [1.0], raw = 2.0 * 1.0 + 1.0 = 3.0, sigmoid(3.0) ≈ 0.9526
        // inner = 0.0, blend = 0.5 * 0.0 + 0.5 * sigmoid(3.0)
        let result = pruner.relevance(0, 0, &[]);
        let expected = 0.5 * sigmoid(3.0_f32);
        assert!((result - expected).abs() < 1e-5, "result={}", result);
    }

    // ── Helper ─────────────────────────────────────────────────

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}
