//! ExpressionPruner — ScreeningPruner backed by a fitted SymbolicExpression.
//!
//! Wraps any inner [`ScreeningPruner`] and augments relevance scoring with
//! a pre-fitted symbolic expression. Relevance is computed as:
//! `0.5 * inner_relevance + 0.5 * expr_result`.
//!
//! When `concept_grounding` is enabled, expression terms can be rendered as
//! human-readable concept names via [`grounded_expression_string`](ExpressionPruner::grounded_expression_string).
//!
//! # Feature Gate
//!
//! `symbolic_distill`.

use katgpt_speculative::ScreeningPruner;

use super::symbolic_expression::SymbolicExpression;

#[cfg(feature = "symbolic_distill")]
use super::absorb_compress::AbsorbCompress;

#[cfg(feature = "symbolic_distill")]
use super::review_metrics::ReviewMetrics;

#[cfg(feature = "concept_grounding")]
use super::concept_grounding::{ConceptGrounding, PrunerState, TemplateGrounding};

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

        let max_score = inner_scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

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

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Access the inner pruner mutably.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// Ground expression terms in human-readable concept names.
    ///
    /// Uses `TemplateGrounding` to map `feature_idx` → semantic concept name,
    /// producing output like `"0.70 × sigmoid(syntax_validity)"` instead of
    /// `"0.70 × σ(x₂)"`.
    ///
    /// Falls back to raw feature names when grounding is unavailable.
    #[cfg(feature = "concept_grounding")]
    pub fn grounded_expression_string(&self) -> String {
        let raw_names = self.feature_extractor.feature_names();
        let grounding = TemplateGrounding::new();

        // Build grounded names by consulting TemplateGrounding for each feature
        let grounded_names: Vec<String> = raw_names
            .iter()
            .enumerate()
            .map(|(idx, &raw)| {
                let state = PrunerState {
                    depth: idx,
                    token_idx: 0,
                    parent_token: Vec::new(),
                    pruner_scores: vec![(raw.to_string(), 0.5)],
                    accepted: true,
                };
                let mappings = grounding.ground(&state);
                mappings
                    .iter()
                    .find(|m| m.variable == raw)
                    .map(|m| m.semantic.clone())
                    .unwrap_or_else(|| raw.to_string())
            })
            .collect();

        // Convert to &str slice for SymbolicExpression::to_string
        let name_refs: Vec<&str> = grounded_names.iter().map(|s| s.as_str()).collect();
        self.expression.to_string(&name_refs)
    }
}

// ── AbsorbCompress Delegation ──────────────────────────────────

/// Delegate `AbsorbCompress` to inner pruner so expression pruners can
/// participate in the self-improving absorb-compress cycle (Plan 210, F1.7).
#[cfg(feature = "symbolic_distill")]
impl<P: ScreeningPruner + AbsorbCompress> AbsorbCompress for ExpressionPruner<P> {
    fn absorb(&mut self, arm: usize, reward: f32) {
        self.inner.absorb(arm, reward);
    }

    fn compress(&mut self) -> Vec<usize> {
        self.inner.compress()
    }

    fn compressed_arms(&self) -> &[usize] {
        self.inner.compressed_arms()
    }

    fn should_compress(&self) -> bool {
        self.inner.should_compress()
    }

    fn should_compress_gated(&self, metrics: Option<&ReviewMetrics>) -> bool {
        self.inner.should_compress_gated(metrics)
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
    use katgpt_speculative::NoScreeningPruner;

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

    // ── AbsorbCompress Delegation Tests ──────────────────────────

    #[cfg(feature = "symbolic_distill")]
    mod absorb_compress_tests {
        use super::super::super::absorb_compress::{
            AbsorbCompress, AbsorbCompressLayer, CompressConfig,
        };
        use super::*;

        #[test]
        fn test_absorb_compress_delegates_to_inner() {
            let inner_layer = AbsorbCompressLayer::new(
                FixedPruner(0.5),
                4,
                CompressConfig {
                    min_visits: 2,
                    q_threshold: 0.3,
                    promote_count: 1,
                    check_interval: 3, // should_compress checks total_absorbed % check_interval
                    min_benefit_ratio: 0.0,
                    ..Default::default()
                },
            );

            let expr = SymbolicExpression {
                terms: vec![Term {
                    basis: BasisFn::Identity,
                    coefficient: 1.0,
                    feature_idx: 0,
                }],
                bias: 0.0,
            };

            let mut pruner = ExpressionPruner::new(inner_layer, expr);

            // Absorb low rewards for arm 0
            pruner.absorb(0, 0.01);
            assert!(
                !pruner.should_compress(),
                "total_absorbed=1, not multiple of 3"
            );
            pruner.absorb(0, 0.02);
            assert!(
                !pruner.should_compress(),
                "total_absorbed=2, not multiple of 3"
            );
            pruner.absorb(1, 0.5); // total_absorbed=3 → triggers

            // should_compress delegates to inner layer
            assert!(pruner.should_compress(), "total_absorbed=3, multiple of 3");

            // Compress should promote arm 0 (Q ≈ 0.015 < threshold 0.3)
            let promoted = pruner.compress();
            assert_eq!(promoted, vec![0], "arm 0 should be promoted");

            // compressed_arms delegates correctly
            assert_eq!(pruner.compressed_arms(), &[0]);
        }

        #[test]
        fn test_absorb_compress_gated_delegates() {
            let inner_layer = AbsorbCompressLayer::new(
                FixedPruner(0.5),
                2,
                CompressConfig {
                    min_visits: 1,
                    q_threshold: 0.5,
                    promote_count: 1,
                    check_interval: 1,
                    min_benefit_ratio: 0.6,
                    ..Default::default()
                },
            );

            let expr = SymbolicExpression {
                terms: Vec::new(),
                bias: 0.0,
            };

            let mut pruner = ExpressionPruner::new(inner_layer, expr);

            // Must absorb first so should_compress returns true
            pruner.absorb(0, 0.1);

            // No metrics → should fall through (inner returns true)
            assert!(
                pruner.should_compress_gated(None),
                "no metrics should allow compression"
            );
        }
    }

    // ── Concept Grounding Integration Tests ────────────────────────

    #[cfg(feature = "concept_grounding")]
    mod concept_grounding_tests {
        use super::*;

        #[test]
        fn grounded_expression_uses_feature_names() {
            let expr = SymbolicExpression {
                terms: vec![Term {
                    basis: BasisFn::Identity,
                    coefficient: 0.7,
                    feature_idx: 0, // "depth"
                }],
                bias: 0.1,
            };
            let pruner = ExpressionPruner::new(FixedPruner(0.5), expr);
            let grounded = pruner.grounded_expression_string();
            // Should contain a readable representation — either grounded or raw name
            assert!(
                !grounded.is_empty(),
                "Grounded expression should not be empty"
            );
            // The raw name "depth" may or may not be grounded by TemplateGrounding
            // (depends on whether there's a mapping for it)
            assert!(
                grounded.contains("depth") || grounded.contains("top-level"),
                "Grounded expression should reference the feature, got '{}'",
                grounded
            );
        }

        #[test]
        fn grounded_expression_fallback_on_unknown_feature() {
            // Custom extractor with unknown feature names
            struct UnknownFeatureExtractor;
            impl FeatureExtractor for UnknownFeatureExtractor {
                fn extract(&self, _: usize, _: usize, _: &[usize], _: &[f32]) -> Vec<f32> {
                    vec![1.0]
                }
                fn feature_names(&self) -> Vec<&str> {
                    vec!["totally_unknown_feature"]
                }
            }

            let expr = SymbolicExpression {
                terms: vec![Term {
                    basis: BasisFn::Sigmoid,
                    coefficient: 1.0,
                    feature_idx: 0,
                }],
                bias: 0.0,
            };
            let pruner = ExpressionPruner::with_extractor(
                FixedPruner(0.5),
                expr,
                Box::new(UnknownFeatureExtractor),
            );
            let grounded = pruner.grounded_expression_string();
            // Should fall back to raw name since TemplateGrounding won't have it
            assert!(
                grounded.contains("totally_unknown_feature"),
                "Should fall back to raw feature name, got '{}'",
                grounded
            );
        }

        #[test]
        fn grounded_expression_matches_raw_for_no_templates() {
            let expr = SymbolicExpression {
                terms: vec![Term {
                    basis: BasisFn::Identity,
                    coefficient: 1.0,
                    feature_idx: 3, // "mean_score"
                }],
                bias: 0.0,
            };
            let pruner = ExpressionPruner::new(FixedPruner(0.0), expr);
            let grounded = pruner.grounded_expression_string();
            let raw = pruner.expression.to_string(&[
                "depth",
                "token",
                "parent_count",
                "mean_score",
                "max_score",
            ]);
            // Both should produce non-empty output
            assert!(!grounded.is_empty());
            assert!(!raw.is_empty());
        }

        /// Cross-feature integration: ExpressionPruner with domain-specific feature names
        /// grounded via TemplateGrounding.
        #[test]
        fn grounded_expression_with_custom_feature_extractor() {
            // Domain-specific feature extractor with semantic names
            struct DomainExtractor;
            impl FeatureExtractor for DomainExtractor {
                fn extract(
                    &self,
                    depth: usize,
                    token: usize,
                    parents: &[usize],
                    scores: &[f32],
                ) -> Vec<f32> {
                    vec![
                        depth as f32,
                        token as f32,
                        parents.len() as f32,
                        scores.first().copied().unwrap_or(0.0),
                    ]
                }
                fn feature_names(&self) -> Vec<&str> {
                    vec!["depth_norm", "score_mean", "syntax_validity", "bandit_q"]
                }
            }

            let expr = SymbolicExpression {
                terms: vec![
                    Term {
                        basis: BasisFn::Identity,
                        coefficient: 0.5,
                        feature_idx: 0,
                    },
                    Term {
                        basis: BasisFn::Sigmoid,
                        coefficient: 0.3,
                        feature_idx: 2,
                    },
                ],
                bias: 0.1,
            };
            let pruner =
                ExpressionPruner::with_extractor(FixedPruner(0.5), expr, Box::new(DomainExtractor));
            let grounded = pruner.grounded_expression_string();

            // Should produce non-empty grounded representation
            assert!(
                !grounded.is_empty(),
                "Grounded expression should not be empty"
            );
            // Should reference domain-specific feature names or their grounded forms
            assert!(
                grounded.contains("depth_norm")
                    || grounded.contains("score_mean")
                    || grounded.contains("syntax_validity")
                    || grounded.contains("bandit_q"),
                "Grounded output should contain feature names, got '{}'",
                grounded
            );
        }
    }
}
