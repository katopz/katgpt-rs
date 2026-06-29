//! Rubric scoring trait and pattern-based scorer.
//!
//! [`RubricScorer`] defines the interface for scoring responses against rubric
//! templates. Implementations use WASM validators, pattern matching, or
//! game-state queries — no LLM needed.
//!
//! # Implementations
//!
//! - [`PatternScorer`] — regex-free pattern-based criterion checks (cheap, always available)
//!
//! # Multi-Reference Scoring (Critical from Ablation)
//!
//! ROPD ablation (Table 6) shows m=4→m=1 costs **−17.94 pts** — the single biggest impact.
//! Single reference over-anchors rubric to one solution trajectory.
//! Use [`score_with_references`] with M ≥ 2 references to prevent collapse.

use std::collections::HashMap;

use super::template::RubricCriterion;
use super::template::RubricTemplate;
use super::types::RubricVector;

// ── RubricScorer Trait ──────────────────────────────────────────

/// Scores a response against a rubric template — no LLM.
///
/// Implementations use WASM validators, pattern matching, or game-state queries.
/// Each criterion gets a score ∈ [0.0, 1.0] representing pass/fail or graded quality.
///
/// # Usage
///
/// ```rust,ignore
/// let scorer = PatternScorer::new();
/// let rubric = scorer.score("response text", &RubricTemplate::generic());
/// println!("Weighted score: {}", rubric.weighted_score());
/// ```
pub trait RubricScorer: Send + Sync {
    /// Score a response against all criteria in the template.
    ///
    /// Returns [`RubricVector`] with per-criterion scores and weights from the template.
    fn score(&self, response: &str, template: &RubricTemplate) -> RubricVector;
}

// ── ScoreResult ─────────────────────────────────────────────────

/// Result of multi-reference scoring.
///
/// Contains the student's rubric and all reference rubrics for gap analysis.
pub struct ScoreResult {
    /// Student's rubric vector.
    pub student: RubricVector,
    /// Reference rubric vectors (M ≥ 2 recommended).
    pub references: Vec<RubricVector>,
}

impl ScoreResult {
    /// Compute gaps between student and all references.
    ///
    /// Uses [`RubricVector::gap_vs_references`] for multi-reference gap analysis.
    pub fn gaps(&self) -> Vec<(usize, f32)> {
        self.student.gap_vs_references(&self.references)
    }

    /// Compute scalar delta between student and best reference.
    ///
    /// Returns the gap vs the reference with the highest weighted score.
    pub fn scalar_delta(&self) -> f32 {
        let best_ref = self.references.iter().max_by(|a, b| {
            a.weighted_score()
                .partial_cmp(&b.weighted_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match best_ref {
            Some(reference) => self.student.to_scalar_delta(reference),
            None => 0.0,
        }
    }
}

// ── PatternScorer ───────────────────────────────────────────────

/// Pattern-based rubric scorer — cheap, always available, no regex.
///
/// Maps each [`RubricCriterion`] to a set of string patterns. A criterion's score
/// is the fraction of its patterns that match the response (pass/fail per pattern).
///
/// # Default Patterns
///
/// The default patterns are simple heuristics for generic text scoring:
///
/// | Criterion | Default Check |
/// |-----------|--------------|
/// | TaskFulfillment | Response is non-trivial (len > 10) |
/// | OutputStructure | Contains structural markers (brackets, braces, colons) |
/// | ConstraintSatisfaction | Contains constraint keywords (must, should, limit, bound) |
/// | Completeness | Response is substantial (len > 50) |
/// | Correctness | Contains factual markers (is, equals, result) |
///
/// Override with [`PatternScorer::with_patterns`] for domain-specific checks.
pub struct PatternScorer {
    /// Per-criterion pattern rules: criterion → list of patterns to check.
    rules: HashMap<RubricCriterion, Vec<PatternRule>>,
}

/// A single pattern rule for scoring.
///
/// Checks if a response satisfies some condition, producing a score ∈ [0.0, 1.0].
#[derive(Clone)]
pub struct PatternRule {
    /// Description of what this rule checks.
    pub description: &'static str,
    /// The check function. Returns score ∈ [0.0, 1.0].
    pub check: fn(&str) -> f32,
}

impl PatternRule {
    /// Create a new pattern rule.
    pub fn new(description: &'static str, check: fn(&str) -> f32) -> Self {
        Self { description, check }
    }
}

impl PatternScorer {
    /// Create a scorer with default pattern rules.
    pub fn new() -> Self {
        Self {
            rules: Self::default_rules(),
        }
    }

    /// Create a scorer with no rules (empty).
    pub fn empty() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// Add a pattern rule for a specific criterion.
    pub fn with_rule(mut self, criterion: RubricCriterion, rule: PatternRule) -> Self {
        self.rules.entry(criterion).or_default().push(rule);
        self
    }

    /// Replace all patterns for a criterion.
    pub fn with_patterns(mut self, criterion: RubricCriterion, patterns: Vec<PatternRule>) -> Self {
        self.rules.insert(criterion, patterns);
        self
    }

    /// Default pattern rules for generic text scoring.
    fn default_rules() -> HashMap<RubricCriterion, Vec<PatternRule>> {
        let mut rules = HashMap::new();

        rules.insert(
            RubricCriterion::TaskFulfillment,
            vec![
                PatternRule::new(
                    "non_trivial_length",
                    |s| {
                        if s.len() > 10 { 1.0 } else { 0.0 }
                    },
                ),
                PatternRule::new("contains_letters", |s| {
                    if s.chars().any(|c| c.is_alphabetic()) {
                        1.0
                    } else {
                        0.0
                    }
                }),
            ],
        );

        rules.insert(
            RubricCriterion::OutputStructure,
            vec![PatternRule::new("has_structural_markers", |s| {
                let markers = ['{', '[', '(', ':', '-', '#'];
                let count = markers.iter().filter(|&&m| s.contains(m)).count();
                (count as f32 / markers.len() as f32).min(1.0)
            })],
        );

        rules.insert(
            RubricCriterion::ConstraintSatisfaction,
            vec![PatternRule::new("constraint_awareness", |s| {
                let keywords = ["must", "should", "limit", "bound", "max", "min", "within"];
                let matches = keywords
                    .iter()
                    .filter(|k| s.to_lowercase().contains(*k))
                    .count();
                (matches as f32 / keywords.len() as f32).min(1.0)
            })],
        );

        rules.insert(
            RubricCriterion::Completeness,
            vec![
                PatternRule::new("substantial_length", |s| {
                    if s.len() > 50 {
                        1.0
                    } else if s.len() > 20 {
                        0.5
                    } else {
                        0.0
                    }
                }),
                PatternRule::new("multi_sentence", |s| {
                    let sentences = s.split('.').filter(|p| !p.trim().is_empty()).count();
                    if sentences >= 3 {
                        1.0
                    } else if sentences >= 2 {
                        0.5
                    } else {
                        0.0
                    }
                }),
            ],
        );

        rules.insert(
            RubricCriterion::Correctness,
            vec![PatternRule::new("factual_markers", |s| {
                let markers = [
                    "is", "equals", "result", "answer", "correct", "true", "false",
                ];
                let matches = markers
                    .iter()
                    .filter(|m| s.to_lowercase().contains(*m))
                    .count();
                (matches as f32 / markers.len() as f32).min(1.0)
            })],
        );

        rules
    }
}

impl Default for PatternScorer {
    fn default() -> Self {
        Self::new()
    }
}

impl RubricScorer for PatternScorer {
    fn score(&self, response: &str, template: &RubricTemplate) -> RubricVector {
        let mut scores = Vec::with_capacity(template.len());
        let mut weights = Vec::with_capacity(template.len());

        for (criterion, weight) in &template.criteria {
            let score = match self.rules.get(criterion) {
                Some(rules) if !rules.is_empty() => {
                    let total: f32 = rules.iter().map(|r| (r.check)(response)).sum();
                    (total / rules.len() as f32).clamp(0.0, 1.0)
                }
                _ => 0.0, // No rules → assume fail
            };
            scores.push(score);
            weights.push(*weight);
        }

        RubricVector::new(scores, weights, 0)
    }
}

// ── Multi-Reference Scoring ─────────────────────────────────────

/// Score student response against M references.
///
/// Single reference over-anchors rubric to one trajectory (−17.9 pts from ROPD ablation).
/// Multiple references prevent collapse to path-matching.
///
/// # Arguments
///
/// * `scorer` — The rubric scorer implementation
/// * `student_response` — The response to evaluate
/// * `references` — M ≥ 2 reference responses (golden/hint-assisted baselines)
/// * `template` — The rubric template defining criteria and weights
///
/// # Returns
///
/// [`ScoreResult`] containing student and reference rubric vectors.
pub fn score_with_references(
    scorer: &dyn RubricScorer,
    student_response: &str,
    references: &[&str],
    template: &RubricTemplate,
) -> ScoreResult {
    let student = scorer.score(student_response, template);
    let ref_rubrics: Vec<RubricVector> = references
        .iter()
        .map(|r| scorer.score(r, template))
        .collect();

    ScoreResult {
        student,
        references: ref_rubrics,
    }
}

/// Score with a pre-assigned template_id.
///
/// Same as [`score_with_references`] but sets `template_id` on all vectors.
pub fn score_with_references_id(
    scorer: &dyn RubricScorer,
    student_response: &str,
    references: &[&str],
    template: &RubricTemplate,
    template_id: usize,
) -> ScoreResult {
    let mut student = scorer.score(student_response, template);
    student.template_id = template_id;

    let ref_rubrics: Vec<RubricVector> = references
        .iter()
        .map(|r| {
            let mut rv = scorer.score(r, template);
            rv.template_id = template_id;
            rv
        })
        .collect();

    ScoreResult {
        student,
        references: ref_rubrics,
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn generic_template() -> RubricTemplate {
        RubricTemplate::generic()
    }

    #[test]
    fn test_pattern_scorer_default() {
        let scorer = PatternScorer::new();
        let response = "The result is correct. The answer equals 42. This must be within bounds.";
        let rv = scorer.score(response, &generic_template());

        // Response should score above 0 on most criteria
        assert!(!rv.is_empty());
        assert!(rv.weighted_score() > 0.0, "Should have positive score");
    }

    #[test]
    fn test_pattern_scorer_empty_response() {
        let scorer = PatternScorer::new();
        let rv = scorer.score("", &generic_template());

        // Empty response should score low
        assert!(
            (rv.weighted_score()).abs() < 0.5,
            "Empty response should score low"
        );
    }

    #[test]
    fn test_pattern_scorer_perfect_response() {
        let scorer = PatternScorer::new();
        let response = "The correct answer equals result. This must be within limit bounds. \
                        The output should have structure { key: value }. \
                        This is the complete factual result.";
        let rv = scorer.score(response, &generic_template());

        assert!(
            rv.weighted_score() > 0.3,
            "Rich response should score well, got {}",
            rv.weighted_score()
        );
    }

    #[test]
    fn test_pattern_scorer_custom_rule() {
        let scorer = PatternScorer::empty().with_rule(
            RubricCriterion::TaskFulfillment,
            PatternRule::new("contains_hello", |s| {
                if s.contains("hello") { 1.0 } else { 0.0 }
            }),
        );

        let template =
            RubricTemplate::new("test").with_criterion(RubricCriterion::TaskFulfillment, 1.0);

        let rv_good = scorer.score("hello world", &template);
        assert!((rv_good.weighted_score() - 1.0).abs() < 1e-6);

        let rv_bad = scorer.score("goodbye world", &template);
        assert!((rv_bad.weighted_score()).abs() < 1e-6);
    }

    #[test]
    fn test_pattern_scorer_no_rules_scores_zero() {
        let scorer = PatternScorer::empty();
        let rv = scorer.score("anything", &generic_template());

        // No rules → all criteria score 0.0
        assert!((rv.weighted_score()).abs() < 1e-6);
    }

    #[test]
    fn test_pattern_scorer_multiple_rules_averaged() {
        let scorer = PatternScorer::empty()
            .with_rule(
                RubricCriterion::Correctness,
                PatternRule::new("rule1", |_| 1.0),
            )
            .with_rule(
                RubricCriterion::Correctness,
                PatternRule::new("rule2", |_| 0.0),
            );

        let template =
            RubricTemplate::new("test").with_criterion(RubricCriterion::Correctness, 1.0);

        let rv = scorer.score("test", &template);
        // Average of [1.0, 0.0] = 0.5
        assert!((rv.score(0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_score_with_references_basic() {
        let scorer = PatternScorer::new();
        let template = generic_template();

        let result = score_with_references(
            &scorer,
            "short",
            &[
                "The result is correct. Must be within bounds.",
                "Another reference.",
            ],
            &template,
        );

        assert!(!result.student.is_empty());
        assert_eq!(result.references.len(), 2);
        assert_eq!(result.gaps().len(), result.student.len());
    }

    #[test]
    fn test_score_with_references_empty_refs() {
        let scorer = PatternScorer::new();
        let template = generic_template();

        let result = score_with_references(&scorer, "response", &[], &template);

        assert!(!result.student.is_empty());
        assert!(result.references.is_empty());
        assert!(result.gaps().is_empty());
    }

    #[test]
    fn test_score_result_scalar_delta() {
        let student = RubricVector::new(vec![0.5], vec![1.0], 0);
        let reference = RubricVector::new(vec![1.0], vec![1.0], 0);

        let result = ScoreResult {
            student,
            references: vec![reference],
        };

        let delta = result.scalar_delta();
        assert!((delta - 0.5).abs() < 1e-6, "Expected 0.5, got {delta}");
    }

    #[test]
    fn test_score_result_scalar_delta_best_reference() {
        let student = RubricVector::new(vec![0.3], vec![1.0], 0);
        let ref_low = RubricVector::new(vec![0.5], vec![1.0], 0);
        let ref_high = RubricVector::new(vec![0.9], vec![1.0], 0);

        let result = ScoreResult {
            student,
            references: vec![ref_low, ref_high],
        };

        // Should use ref_high (best reference)
        let delta = result.scalar_delta();
        assert!((delta - 0.6).abs() < 1e-6, "Expected 0.6, got {delta}");
    }

    #[test]
    fn test_score_with_references_id_sets_template_id() {
        let scorer = PatternScorer::new();
        let template = generic_template();

        let result = score_with_references_id(&scorer, "response", &["ref"], &template, 42);

        assert_eq!(result.student.template_id, 42);
        assert_eq!(result.references[0].template_id, 42);
    }

    #[test]
    fn test_score_result_gaps_multi_reference_uses_max() {
        let student = RubricVector::new(vec![0.3, 0.5], vec![1.0, 1.0], 0);
        let ref1 = RubricVector::new(vec![0.8, 0.4], vec![1.0, 1.0], 0);
        let ref2 = RubricVector::new(vec![0.6, 0.9], vec![1.0, 1.0], 0);

        let result = ScoreResult {
            student,
            references: vec![ref1, ref2],
        };

        let gaps = result.gaps();
        // Criterion 0: max(0.8, 0.6) = 0.8, gap = 0.5
        // Criterion 1: max(0.4, 0.9) = 0.9, gap = 0.4
        assert_eq!(gaps.len(), 2);
        // Check that both gaps are present
        let gap_0 = gaps
            .iter()
            .find(|(i, _)| *i == 0)
            .map(|(_, g)| *g)
            .unwrap_or(0.0);
        let gap_1 = gaps
            .iter()
            .find(|(i, _)| *i == 1)
            .map(|(_, g)| *g)
            .unwrap_or(0.0);
        assert!((gap_0 - 0.5).abs() < 1e-6, "Expected 0.5, got {gap_0}");
        assert!((gap_1 - 0.4).abs() < 1e-6, "Expected 0.4, got {gap_1}");
    }

    #[test]
    fn test_pattern_rule_new() {
        let rule = PatternRule::new("test_rule", |s| if s.len() > 5 { 1.0 } else { 0.0 });
        assert_eq!(rule.description, "test_rule");
        assert!((rule.check)("longer") == 1.0);
        assert!((rule.check)("short") == 0.0);
    }
}
