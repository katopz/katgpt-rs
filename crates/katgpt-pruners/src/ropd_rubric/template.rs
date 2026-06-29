//! Domain-specific rubric templates — fixed criteria + weights, no LLM needed.
//!
//! Each [`RubricTemplate`] maps to a deterministic set of criteria that can be
//! checked by WASM validators, pattern matching, or game-state queries.
//!
//! # From ROPD
//!
//! `ropd.rubric.v1` schema: `{criterion_id, category, criterion, points}`
//! Weight semantics: 5=decisive, 4=strong, 2=supporting, 1=routine.
//!
//! # Templates
//!
//! - [`RubricTemplate::bomber()`] — survival + safety + efficiency (3 criteria, single-axis)
//! - [`RubricTemplate::fft_tactics()`] — role + coordination + survival (3 criteria, multi-axis)
//! - [`RubricTemplate::generic()`] — task + structure + constraints (3 criteria, baseline)

use serde::{Deserialize, Serialize};

// ── RubricCriterion ─────────────────────────────────────────────

/// Fixed rubric criteria per domain — no LLM generation needed.
///
/// Each variant maps to a deterministic WASM-checkable criterion.
/// Criteria are domain-agnostic; templates select which apply and with what weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RubricCriterion {
    /// Did it answer the question / complete the task?
    TaskFulfillment,
    /// Is the output in valid format / type?
    OutputStructure,
    /// Does it satisfy constraints (budget, bounds, limits)?
    ConstraintSatisfaction,
    /// Are all required components present?
    Completeness,
    /// Is the answer correct (verifiable domains only)?
    Correctness,
}

impl RubricCriterion {
    /// Human-readable name for this criterion.
    pub fn name(&self) -> &'static str {
        match self {
            Self::TaskFulfillment => "task_fulfillment",
            Self::OutputStructure => "output_structure",
            Self::ConstraintSatisfaction => "constraint_satisfaction",
            Self::Completeness => "completeness",
            Self::Correctness => "correctness",
        }
    }

    /// All known criteria in canonical order.
    pub fn all() -> &'static [RubricCriterion] {
        &[
            Self::TaskFulfillment,
            Self::OutputStructure,
            Self::ConstraintSatisfaction,
            Self::Completeness,
            Self::Correctness,
        ]
    }
}

impl std::fmt::Display for RubricCriterion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── RubricTemplate ──────────────────────────────────────────────

/// Domain-specific rubric template: which criteria apply + their weights.
///
/// Mirrors ROPD's weight semantics: 5=decisive, 4=strong, 2=supporting, 1=routine.
/// Weights determine how much each criterion contributes to the overall score.
///
/// # Construction
///
/// Use the built-in domain constructors ([`bomber()`], [`fft_tactics()`], [`generic()`])
/// or build custom templates with [`new()`] and [`with_criterion()`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RubricTemplate {
    /// (criterion, weight) pairs — ordered by convention, weights > 0.
    pub criteria: Vec<(RubricCriterion, f32)>,
    /// Template identifier for logging/debugging.
    pub id: String,
}

impl RubricTemplate {
    /// Create an empty template with a given id.
    pub fn new(id: &str) -> Self {
        Self {
            criteria: Vec::new(),
            id: id.to_string(),
        }
    }

    /// Add a criterion with weight.
    pub fn with_criterion(mut self, criterion: RubricCriterion, weight: f32) -> Self {
        self.criteria.push((criterion, weight));
        self
    }

    /// Number of criteria in this template.
    pub fn len(&self) -> usize {
        self.criteria.len()
    }

    /// Whether this template has no criteria.
    pub fn is_empty(&self) -> bool {
        self.criteria.is_empty()
    }

    /// Total weight across all criteria.
    pub fn total_weight(&self) -> f32 {
        self.criteria.iter().map(|(_, w)| *w).sum()
    }

    /// Find the weight for a specific criterion, or 0.0 if not present.
    pub fn weight_of(&self, criterion: RubricCriterion) -> f32 {
        self.criteria
            .iter()
            .find(|(c, _)| *c == criterion)
            .map(|(_, w)| *w)
            .unwrap_or(0.0)
    }

    /// Whether this template includes the given criterion.
    pub fn has_criterion(&self, criterion: RubricCriterion) -> bool {
        self.criteria.iter().any(|(c, _)| *c == criterion)
    }

    // ── Domain Templates ────────────────────────────────────────

    /// Bomber arena template — survival + safety + efficiency.
    ///
    /// Single quality axis (survival dominates). Rubrics expected to show
    /// minimal gain over scalar δ in single-axis domains (Plan 071 hypothesis).
    ///
    /// | Criterion | Weight | Rationale |
    /// |-----------|--------|-----------|
    /// | TaskFulfillment | 4.0 | Survival = primary objective |
    /// | ConstraintSatisfaction | 2.0 | Safety (avoid blast zone) |
    /// | Completeness | 1.0 | Efficiency (used bombs well) |
    pub fn bomber() -> Self {
        Self {
            criteria: vec![
                (RubricCriterion::TaskFulfillment, 4.0),
                (RubricCriterion::ConstraintSatisfaction, 2.0),
                (RubricCriterion::Completeness, 1.0),
            ],
            id: "bomber".to_string(),
        }
    }

    /// FFT Tactics arena template — role fulfillment + team coordination + survival.
    ///
    /// Multi-axis domain where rubrics should help most (Plan 071 hypothesis).
    /// Independent quality axes prevent inter-dimensional interference.
    ///
    /// | Criterion | Weight | Rationale |
    /// |-----------|--------|-----------|
    /// | TaskFulfillment | 4.0 | Role fulfillment (did its job) |
    /// | Completeness | 3.0 | Team coordination (helped allies) |
    /// | ConstraintSatisfaction | 2.0 | Survival (stayed alive) |
    pub fn fft_tactics() -> Self {
        Self {
            criteria: vec![
                (RubricCriterion::TaskFulfillment, 4.0),
                (RubricCriterion::Completeness, 3.0),
                (RubricCriterion::ConstraintSatisfaction, 2.0),
            ],
            id: "fft_tactics".to_string(),
        }
    }

    /// Generic template — task + structure + constraints.
    ///
    /// Baseline for domains without a specific template.
    ///
    /// | Criterion | Weight | Rationale |
    /// |-----------|--------|-----------|
    /// | TaskFulfillment | 4.0 | Core task completion |
    /// | OutputStructure | 2.0 | Valid output format |
    /// | ConstraintSatisfaction | 2.0 | Constraint compliance |
    pub fn generic() -> Self {
        Self {
            criteria: vec![
                (RubricCriterion::TaskFulfillment, 4.0),
                (RubricCriterion::OutputStructure, 2.0),
                (RubricCriterion::ConstraintSatisfaction, 2.0),
            ],
            id: "generic".to_string(),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criterion_name() {
        assert_eq!(RubricCriterion::TaskFulfillment.name(), "task_fulfillment");
        assert_eq!(RubricCriterion::Correctness.name(), "correctness");
    }

    #[test]
    fn test_criterion_all_canonical_order() {
        let all = RubricCriterion::all();
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], RubricCriterion::TaskFulfillment);
        assert_eq!(all[4], RubricCriterion::Correctness);
    }

    #[test]
    fn test_criterion_display() {
        assert_eq!(
            format!("{}", RubricCriterion::OutputStructure),
            "output_structure"
        );
    }

    #[test]
    fn test_template_bomber() {
        let t = RubricTemplate::bomber();
        assert_eq!(t.id, "bomber");
        assert_eq!(t.len(), 3);
        assert_eq!(t.total_weight(), 7.0);
        assert!(t.has_criterion(RubricCriterion::TaskFulfillment));
        assert!(!t.has_criterion(RubricCriterion::Correctness));
    }

    #[test]
    fn test_template_fft_tactics() {
        let t = RubricTemplate::fft_tactics();
        assert_eq!(t.id, "fft_tactics");
        assert_eq!(t.len(), 3);
        assert_eq!(t.total_weight(), 9.0);
    }

    #[test]
    fn test_template_generic() {
        let t = RubricTemplate::generic();
        assert_eq!(t.id, "generic");
        assert_eq!(t.len(), 3);
        assert_eq!(t.total_weight(), 8.0);
    }

    #[test]
    fn test_template_custom_builder() {
        let t = RubricTemplate::new("custom")
            .with_criterion(RubricCriterion::Correctness, 5.0)
            .with_criterion(RubricCriterion::Completeness, 2.0);

        assert_eq!(t.id, "custom");
        assert_eq!(t.len(), 2);
        assert!((t.total_weight() - 7.0).abs() < 1e-6);
        assert!((t.weight_of(RubricCriterion::Correctness) - 5.0).abs() < 1e-6);
        assert!((t.weight_of(RubricCriterion::TaskFulfillment)).abs() < 1e-6);
    }

    #[test]
    fn test_template_empty() {
        let t = RubricTemplate::new("empty");
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert!((t.total_weight()).abs() < 1e-6);
    }

    #[test]
    fn test_template_weight_of_missing() {
        let t = RubricTemplate::bomber();
        assert!((t.weight_of(RubricCriterion::Correctness)).abs() < 1e-6);
    }

    #[test]
    fn test_template_serialization_roundtrip() {
        let t = RubricTemplate::bomber();
        let json = serde_json::to_string(&t).unwrap();
        let deserialized: RubricTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(t.id, deserialized.id);
        assert_eq!(t.criteria.len(), deserialized.criteria.len());
        for (i, (c, w)) in t.criteria.iter().enumerate() {
            assert_eq!(*c, deserialized.criteria[i].0);
            assert!((w - deserialized.criteria[i].1).abs() < 1e-6);
        }
    }
}
