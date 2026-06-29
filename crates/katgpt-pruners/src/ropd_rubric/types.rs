//! Structured multi-criteria score — ROPD's reward without LLM.
//!
//! [`RubricVector`] replaces scalar [`HintDelta`](super::super::g_zero::types::HintDelta)
//! with a per-criterion pass/fail vector. Each criterion gets its own score ∈ [0, 1],
//! enabling targeted absorb and inter-dimensional gap tracking.
//!
//! # ROPD Formula
//!
//! Weighted score: `Σ(w_k * v_k) / Σ(w_k)`
//!
//! This matches ROPD's `s_i = Σ(w_k * v_{i,k}) / (Σ w_k + ε)` where `v_{i,k} ∈ {0,1}`.
//! We generalize to continuous scores ∈ [0.0, 1.0] for graded criteria.
//!
//! # Multi-Reference Gap (Critical from Ablation)
//!
//! ROPD ablation (Table 6) shows m=4→m=1 costs −17.94 pts — the single biggest impact.
//! Single reference over-anchors rubric to one solution trajectory.
//! [`gap_vs_references`] uses `max(reference_scores)` per criterion to prevent collapse.

use serde::{Deserialize, Serialize};

// ── RubricVector ────────────────────────────────────────────────

/// Structured multi-criteria score — ROPD's reward without LLM.
///
/// Replaces scalar `HintDelta.value` with per-criterion pass/fail vector.
/// Each score ∈ [0.0, 1.0] represents how well the response satisfies that criterion.
///
/// # Construction
///
/// ```rust,ignore
/// let rv = RubricVector::new(
///     vec![0.8, 0.6, 1.0],  // scores
///     vec![4.0, 2.0, 1.0],  // weights
///     0,                     // template_id
/// );
/// ```
///
/// # Design Principle (SRP)
///
/// `RubricVector` is pure data — no domain logic. Domain-specific scoring
/// stays in the template/validator.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RubricVector {
    /// Per-criterion scores [0.0, 1.0].
    pub scores: Vec<f32>,
    /// Per-criterion weights (importance).
    pub weights: Vec<f32>,
    /// Template that generated this rubric.
    pub template_id: usize,
}

impl RubricVector {
    /// Create a new rubric vector from scores, weights, and template id.
    ///
    /// # Panics
    ///
    /// Panics if `scores` and `weights` have different lengths.
    pub fn new(scores: Vec<f32>, weights: Vec<f32>, template_id: usize) -> Self {
        assert_eq!(
            scores.len(),
            weights.len(),
            "scores and weights must have same length"
        );
        Self {
            scores,
            weights,
            template_id,
        }
    }

    /// Create a zero rubric vector with given weights.
    ///
    /// All scores are 0.0. Useful as a baseline for gap computation.
    pub fn zero(weights: Vec<f32>, template_id: usize) -> Self {
        let n = weights.len();
        Self {
            scores: vec![0.0; n],
            weights,
            template_id,
        }
    }

    /// Create a perfect rubric vector with given weights.
    ///
    /// All scores are 1.0. Useful as a reference ceiling.
    pub fn perfect(weights: Vec<f32>, template_id: usize) -> Self {
        let n = weights.len();
        Self {
            scores: vec![1.0; n],
            weights,
            template_id,
        }
    }

    /// Number of criteria in this vector.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether this vector has no criteria.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// ROPD weighted score: `Σ(w_k * v_k) / Σ(w_k)`.
    ///
    /// Returns 0.0 if weights sum to zero (degenerate template).
    pub fn weighted_score(&self) -> f32 {
        let total_weight: f32 = self.weights.iter().copied().sum();
        if total_weight.abs() < f32::EPSILON {
            return 0.0;
        }
        let weighted_sum: f32 = self
            .scores
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| s * w)
            .sum();
        weighted_sum / total_weight
    }

    /// Aggregate gap across M references (critical — ablation shows m=1 costs 17.9 pts).
    ///
    /// For each criterion, gap = `max(reference_scores) - student_score`.
    /// Returns criteria sorted by `weight × gap` magnitude (descending).
    ///
    /// # Arguments
    ///
    /// * `references` — M ≥ 2 reference rubrics (golden/hint-assisted baselines).
    ///   Single reference still works but degrades quality (ROPD ablation: −17.9 pts).
    ///
    /// # Returns
    ///
    /// `Vec<(criterion_index, gap_magnitude)>` sorted by `weight × gap` descending.
    /// Empty if references are empty.
    pub fn gap_vs_references(&self, references: &[RubricVector]) -> Vec<(usize, f32)> {
        if references.is_empty() || self.scores.is_empty() {
            return Vec::new();
        }

        let n = self.scores.len();
        let mut gaps: Vec<(usize, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            // Max reference score for this criterion across all references
            let max_ref_score = references
                .iter()
                .filter_map(|r| r.scores.get(i).copied())
                .fold(0.0_f32, f32::max);

            let gap = (max_ref_score - self.scores[i]).max(0.0);
            gaps.push((i, gap));
        }

        // Sort by weight × gap descending
        gaps.sort_by(|a, b| {
            let wa = self.weights.get(a.0).copied().unwrap_or(0.0) * a.1;
            let wb = self.weights.get(b.0).copied().unwrap_or(0.0) * b.1;
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
        });

        gaps
    }

    /// Which criteria have gaps vs a single reference — for targeted absorb.
    ///
    /// Returns `(criterion_index, gap_magnitude)` sorted by `weight × gap` descending.
    /// Only criteria where `reference_score > student_score` are included.
    pub fn gap_criteria(&self, reference: &RubricVector) -> Vec<(usize, f32)> {
        if self.scores.is_empty() {
            return Vec::new();
        }

        let n = self.scores.len().min(reference.scores.len());
        let mut gaps: Vec<(usize, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            let ref_score = reference.scores[i];
            let student_score = self.scores[i];
            if ref_score > student_score {
                gaps.push((i, ref_score - student_score));
            }
        }

        // Sort by weight × gap descending
        gaps.sort_by(|a, b| {
            let wa = self.weights.get(a.0).copied().unwrap_or(0.0) * a.1;
            let wb = self.weights.get(b.0).copied().unwrap_or(0.0) * b.1;
            wb.partial_cmp(&wa).unwrap_or(std::cmp::Ordering::Equal)
        });

        gaps
    }

    /// Compress to scalar for compatibility with existing bandit.
    ///
    /// Equivalent to `HintDelta.value` for drop-in replacement.
    /// Computes `reference.weighted_score() - self.weighted_score()`.
    ///
    /// Returns 0.0 if both scores are equal, positive if reference is better.
    pub fn to_scalar_delta(&self, reference: &RubricVector) -> f32 {
        reference.weighted_score() - self.weighted_score()
    }

    /// Whether any criterion has a gap vs the given reference.
    pub fn has_any_gap(&self, reference: &RubricVector, threshold: f32) -> bool {
        let n = self.scores.len().min(reference.scores.len());
        for i in 0..n {
            if reference.scores[i] - self.scores[i] > threshold {
                return true;
            }
        }
        false
    }

    /// Whether a specific criterion has a gap exceeding the threshold.
    pub fn has_criterion_gap(
        &self,
        criterion_idx: usize,
        reference: &RubricVector,
        threshold: f32,
    ) -> bool {
        let student = self.scores.get(criterion_idx).copied().unwrap_or(0.0);
        let ref_score = reference.scores.get(criterion_idx).copied().unwrap_or(0.0);
        ref_score - student > threshold
    }

    /// Get the score for a specific criterion index.
    pub fn score(&self, idx: usize) -> f32 {
        self.scores.get(idx).copied().unwrap_or(0.0)
    }

    /// Get the weight for a specific criterion index.
    pub fn weight(&self, idx: usize) -> f32 {
        self.weights.get(idx).copied().unwrap_or(0.0)
    }

    /// Per-criterion gap vector — all gaps, not filtered.
    ///
    /// Returns `Vec<f32>` where `gap[i] = max(reference.scores[i] - self.scores[i], 0.0)`.
    /// Unlike [`gap_criteria()`] which filters and sorts, this preserves the full
    /// criterion index mapping for reward computation.
    ///
    /// Use this when you need the raw gap vector for per-criterion reward signals.
    /// Use [`gap_criteria()`] when you need sorted/filtered gaps for targeting.
    pub fn per_criterion_gaps(&self, reference: &RubricVector) -> Vec<f32> {
        let n = self.scores.len().min(reference.scores.len());
        (0..n)
            .map(|i| (reference.scores[i] - self.scores[i]).max(0.0))
            .collect()
    }

    /// Quadratic weighted gap reward — fixes scalar collapse (Issue 061).
    ///
    /// Unlike [`to_scalar_delta()`] which collapses via `Σ(w_i × gap_i) / Σ(w_i)`,
    /// this uses `Σ(w_i × gap_i²) / Σ(w_i)` — a **quadratic** weighted sum.
    /// The quadratic form breaks permutation symmetry: concentrated gaps in one
    /// criterion score higher than spread gaps across multiple criteria, even when
    /// the linear weighted sum is identical.
    ///
    /// # Why Quadratic Fixes the Collapse
    ///
    /// Linear: `Σ(w_i × gap_i)` is symmetric — swapping gaps between criteria
    /// leaves the sum unchanged. Two profiles with same `weighted_score` produce
    /// identical reward regardless of gap distribution.
    ///
    /// Quadratic: `Σ(w_i × gap_i²)` penalizes large gaps in single criteria more
    /// than small gaps across many. Swapping gaps between criteria with different
    /// weights changes the result because `w_i × gap_i² ≠ w_j × gap_i²` when `w_i ≠ w_j`.
    ///
    /// # Formula
    ///
    /// `reward = Σ(w_i × gap_i²) / Σ(w_i)`
    ///
    /// # Example (Bomber: weights=[4.0, 2.0, 1.0], reference=perfect)
    ///
    /// | Profile | Scores | Gaps | `weighted_score` | `quadratic_reward` |
    /// |---------|--------|------|------------------|--------------------|
    /// | A | (1.0, 0.0, 0.0) | (0.0, 1.0, 1.0) | 0.571 | 0.429 |
    /// | C | (0.5, 0.5, 1.0) | (0.5, 0.5, 0.0) | 0.571 | 0.214 |
    ///
    /// Same `weighted_score` (0.571) → **different** `quadratic_reward` (0.429 ≠ 0.214).
    /// Profile A has concentrated failures (actionable) → higher reward → bandit explores more.
    /// Profile C has spread failures (less actionable) → lower reward → bandit explores less.
    pub fn quadratic_weighted_reward(&self, reference: &RubricVector) -> f32 {
        let n = self.scores.len().min(reference.scores.len());
        let total_weight: f32 = self.weights.iter().take(n).copied().sum();
        if total_weight.abs() < f32::EPSILON || n == 0 {
            return 0.0;
        }

        let mut reward = 0.0;
        for i in 0..n {
            let gap = (reference.scores[i] - self.scores[i]).max(0.0);
            reward += self.weights[i] * gap * gap;
        }
        reward / total_weight
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(scores: Vec<f32>, weights: Vec<f32>) -> RubricVector {
        RubricVector::new(scores, weights, 0)
    }

    #[test]
    fn test_weighted_score_basic() {
        // scores: [1.0, 0.5, 0.0], weights: [4.0, 2.0, 1.0]
        // weighted_score = (4*1 + 2*0.5 + 1*0) / (4+2+1) = 5/7 ≈ 0.714
        let rv = make_vector(vec![1.0, 0.5, 0.0], vec![4.0, 2.0, 1.0]);
        let score = rv.weighted_score();
        assert!(
            (score - 5.0 / 7.0).abs() < 1e-6,
            "Expected 5/7, got {score}"
        );
    }

    #[test]
    fn test_weighted_score_all_perfect() {
        let rv = make_vector(vec![1.0, 1.0, 1.0], vec![4.0, 2.0, 1.0]);
        assert!((rv.weighted_score() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_score_all_zero() {
        let rv = make_vector(vec![0.0, 0.0, 0.0], vec![4.0, 2.0, 1.0]);
        assert!((rv.weighted_score()).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_score_zero_weights() {
        let rv = make_vector(vec![1.0, 1.0], vec![0.0, 0.0]);
        assert!(
            (rv.weighted_score()).abs() < 1e-6,
            "Zero weights → score is 0"
        );
    }

    #[test]
    fn test_gap_vs_references_single_ref() {
        let student = make_vector(vec![0.5, 0.3, 0.8], vec![4.0, 2.0, 1.0]);
        let reference = make_vector(vec![1.0, 0.9, 1.0], vec![4.0, 2.0, 1.0]);

        let gaps = student.gap_vs_references(&[reference]);

        // Gap[0] = 1.0 - 0.5 = 0.5, w*gap = 4*0.5 = 2.0
        // Gap[1] = 0.9 - 0.3 = 0.6, w*gap = 2*0.6 = 1.2
        // Gap[2] = 1.0 - 0.8 = 0.2, w*gap = 1*0.2 = 0.2
        // Sorted: [0 (2.0), 1 (1.2), 2 (0.2)]
        assert_eq!(gaps.len(), 3);
        assert_eq!(gaps[0].0, 0);
        assert!((gaps[0].1 - 0.5).abs() < 1e-6);
        assert_eq!(gaps[1].0, 1);
        assert!((gaps[1].1 - 0.6).abs() < 1e-6);
        assert_eq!(gaps[2].0, 2);
        assert!((gaps[2].1 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_gap_vs_references_multiple_uses_max() {
        let student = make_vector(vec![0.5, 0.3], vec![4.0, 2.0]);
        let ref1 = make_vector(vec![0.8, 0.9], vec![4.0, 2.0]);
        let ref2 = make_vector(vec![1.0, 0.6], vec![4.0, 2.0]);

        let gaps = student.gap_vs_references(&[ref1, ref2]);

        // Criterion 0: max(0.8, 1.0) = 1.0, gap = 0.5, w*gap = 2.0
        // Criterion 1: max(0.9, 0.6) = 0.9, gap = 0.6, w*gap = 1.2
        assert_eq!(gaps[0].0, 0);
        assert!((gaps[0].1 - 0.5).abs() < 1e-6);
        assert_eq!(gaps[1].0, 1);
        assert!((gaps[1].1 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_gap_vs_references_empty() {
        let student = make_vector(vec![0.5], vec![1.0]);
        assert!(student.gap_vs_references(&[]).is_empty());
    }

    #[test]
    fn test_gap_vs_references_no_gap_clamped() {
        // Student scores >= reference scores → gap clamped to 0
        let student = make_vector(vec![1.0, 0.9], vec![4.0, 2.0]);
        let reference = make_vector(vec![0.5, 0.8], vec![4.0, 2.0]);

        let gaps = student.gap_vs_references(&[reference]);
        // All gaps are 0.0 (student >= reference)
        for (_, gap) in &gaps {
            assert!((*gap).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gap_criteria_only_positive_gaps() {
        let student = make_vector(vec![0.5, 0.9, 0.3], vec![4.0, 2.0, 1.0]);
        let reference = make_vector(vec![1.0, 0.8, 0.9], vec![4.0, 2.0, 1.0]);

        let gaps = student.gap_criteria(&reference);

        // Criterion 0: ref=1.0 > stud=0.5 → gap=0.5, w*gap=2.0
        // Criterion 1: ref=0.8 < stud=0.9 → no gap (not included)
        // Criterion 2: ref=0.9 > stud=0.3 → gap=0.6, w*gap=0.6
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0].0, 0, "Highest w*gap first");
        assert!((gaps[0].1 - 0.5).abs() < 1e-6);
        assert_eq!(gaps[1].0, 2);
        assert!((gaps[1].1 - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_gap_criteria_empty_scores() {
        let student = RubricVector::zero(vec![], 0);
        let reference = make_vector(vec![1.0], vec![1.0]);
        assert!(student.gap_criteria(&reference).is_empty());
    }

    #[test]
    fn test_to_scalar_delta() {
        let student = make_vector(vec![0.5, 0.5], vec![4.0, 2.0]);
        // student weighted = (4*0.5 + 2*0.5) / 6 = 3/6 = 0.5
        let reference = make_vector(vec![1.0, 1.0], vec![4.0, 2.0]);
        // ref weighted = (4*1 + 2*1) / 6 = 6/6 = 1.0

        let delta = student.to_scalar_delta(&reference);
        assert!((delta - 0.5).abs() < 1e-6, "delta = ref - student = 0.5");
    }

    #[test]
    fn test_to_scalar_delta_equal() {
        let rv = make_vector(vec![0.5, 0.5], vec![4.0, 2.0]);
        assert!((rv.to_scalar_delta(&rv)).abs() < 1e-6);
    }

    #[test]
    fn test_to_scalar_delta_student_better() {
        let student = make_vector(vec![1.0, 1.0], vec![4.0, 2.0]);
        let reference = make_vector(vec![0.5, 0.5], vec![4.0, 2.0]);
        let delta = student.to_scalar_delta(&reference);
        assert!(delta < 0.0, "Student better → negative delta");
    }

    #[test]
    fn test_zero_constructor() {
        let rv = RubricVector::zero(vec![4.0, 2.0, 1.0], 42);
        assert_eq!(rv.template_id, 42);
        assert_eq!(rv.len(), 3);
        assert!((rv.weighted_score()).abs() < 1e-6);
        assert!(rv.scores.iter().all(|&s| (s).abs() < 1e-6));
    }

    #[test]
    fn test_perfect_constructor() {
        let rv = RubricVector::perfect(vec![4.0, 2.0, 1.0], 0);
        assert!((rv.weighted_score() - 1.0).abs() < 1e-6);
        assert!(rv.scores.iter().all(|&s| (s - 1.0).abs() < 1e-6));
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths_panics() {
        RubricVector::new(vec![0.5], vec![1.0, 2.0], 0);
    }

    #[test]
    fn test_has_any_gap() {
        let student = make_vector(vec![0.5, 0.9], vec![4.0, 2.0]);
        let reference = make_vector(vec![0.8, 0.8], vec![4.0, 2.0]);

        assert!(student.has_any_gap(&reference, 0.1));
        assert!(!student.has_any_gap(&reference, 0.5));
    }

    #[test]
    fn test_has_criterion_gap() {
        let student = make_vector(vec![0.5, 0.9], vec![4.0, 2.0]);
        let reference = make_vector(vec![0.8, 0.8], vec![4.0, 2.0]);

        assert!(student.has_criterion_gap(0, &reference, 0.1));
        assert!(!student.has_criterion_gap(1, &reference, 0.1));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let rv = make_vector(vec![0.5, 0.8, 1.0], vec![4.0, 2.0, 1.0]);
        let json = serde_json::to_string(&rv).unwrap();
        let deserialized: RubricVector = serde_json::from_str(&json).unwrap();
        assert_eq!(rv, deserialized);
    }

    #[test]
    fn test_len_and_empty() {
        let rv = make_vector(vec![0.5], vec![1.0]);
        assert_eq!(rv.len(), 1);
        assert!(!rv.is_empty());

        let empty = RubricVector::zero(vec![], 0);
        assert!(empty.is_empty());
    }
}
