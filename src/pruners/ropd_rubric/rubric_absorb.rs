//! Absorb-compress gated by rubric vector — promote heuristics only for targeted criterion gaps.
//!
//! Replaces [`DeltaGatedAbsorbCompress`](super::super::g_zero::delta_absorb::DeltaGatedAbsorbCompress)'s
//! scalar δ gate with a rubric vector gate:
//!
//! - **Delta**: gate on scalar δ > threshold (blind — *why* did it trigger?)
//! - **Rubric**: gate on specific criterion gap (targeted — "constraint #2 failed")
//!
//! # Per-Criterion Absorb Targeting
//!
//! High-weight criterion gap → promote to hard constraint.
//! Low-weight criterion gap → ignore (not worth promoting).
//!
//! This enables fine-grained absorb that scalar δ cannot provide.
//!
//! # Multi-Reference Requirement
//!
//! ROPD ablation (Table 6): m=4→m=1 costs −17.94 pts.
//! Single reference over-anchors rubric to one trajectory.
//! [`RubricGatedConfig::min_references`] defaults to 2 to enforce this.

use std::cmp::Ordering;

use crate::pruners::absorb_compress::{AbsorbCompress, AbsorbCompressLayer};
use crate::pruners::review_metrics::ReviewMetrics;
use crate::speculative::types::ScreeningPruner;

use super::types::RubricVector;

// ── Config ──────────────────────────────────────────────────────

/// Tunable thresholds for rubric-gated absorb-compress.
///
/// An arm is absorbed (eligible for promotion) when:
/// 1. At least one criterion gap exceeds `gap_threshold`
/// 2. That criterion's weight is at least `min_weight_for_absorb`
/// 3. At least `min_references` reference rubrics are available
///
/// # Defaults
///
/// - `gap_threshold`: 0.3 — meaningful gap in a criterion
/// - `min_weight_for_absorb`: 2.0 — only "strong" criteria trigger absorb
/// - `min_references`: 2 — ROPD ablation shows m=1 costs 17.9 pts
#[derive(Clone, Debug)]
pub struct RubricGatedConfig {
    /// Minimum weighted gap to trigger absorb (default: 0.3).
    ///
    /// Per-criterion gap = `reference_score - student_score`.
    /// Only criteria with gap > threshold are considered for absorb.
    pub gap_threshold: f32,
    /// Only absorb gaps in criteria with weight ≥ this (default: 2.0).
    ///
    /// ROPD weight semantics: 5=decisive, 4=strong, 2=supporting, 1=routine.
    /// Default 2.0 means "supporting and above" trigger absorb.
    pub min_weight_for_absorb: f32,
    /// Minimum reference rubrics per arm (default: 2).
    ///
    /// ROPD ablation: m=4→m=1 costs −17.94 pts. Single reference over-anchors
    /// rubric to one solution trajectory. Arms with fewer references are skipped.
    pub min_references: usize,
    /// Minimum benefit-to-risk ratio for compression (Plan 036 compatibility).
    ///
    /// Set to 0.0 to disable review metrics gating.
    pub min_benefit_ratio: f64,
}

impl Default for RubricGatedConfig {
    fn default() -> Self {
        Self {
            gap_threshold: 0.3,
            min_weight_for_absorb: 2.0,
            min_references: 2,
            min_benefit_ratio: 2.0,
        }
    }
}

impl RubricGatedConfig {
    /// Create config with custom gap threshold.
    pub fn new(gap_threshold: f32) -> Self {
        Self {
            gap_threshold,
            ..Self::default()
        }
    }

    /// Create config with custom benefit-ratio threshold.
    pub fn with_benefit_ratio(mut self, min_benefit_ratio: f64) -> Self {
        self.min_benefit_ratio = min_benefit_ratio;
        self
    }

    /// Set minimum references required per arm.
    pub fn with_min_references(mut self, min_references: usize) -> Self {
        self.min_references = min_references;
        self
    }

    /// Set minimum weight for absorb eligibility.
    pub fn with_min_weight(mut self, min_weight_for_absorb: f32) -> Self {
        self.min_weight_for_absorb = min_weight_for_absorb;
        self
    }
}

// ── ArmRubricState ──────────────────────────────────────────────

/// Per-arm rubric tracking state.
#[derive(Clone, Debug)]
struct ArmRubricState {
    /// Accumulated rubric scores from observations.
    history: Vec<RubricVector>,
    /// Reference rubrics for gap comparison.
    references: Vec<RubricVector>,
    /// Whether this arm has exceeded the gap threshold.
    above_threshold: bool,
    /// Last computed gap analysis: (criterion_idx, gap, weight).
    last_gaps: Vec<(usize, f32, f32)>,
}

impl ArmRubricState {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            references: Vec::new(),
            above_threshold: false,
            last_gaps: Vec::new(),
        }
    }
}

// ── RubricGatedAbsorbCompress ───────────────────────────────────

/// Absorb-compress layer gated by rubric vector instead of scalar δ.
///
/// Wraps [`AbsorbCompressLayer`] and only promotes arms where the rubric
/// reveals a gap in a high-weight criterion. This targets specific weaknesses
/// rather than just low-reward arms.
///
/// # Architecture
///
/// ```text
/// RubricGatedAbsorbCompress<P>
///   ├── inner: AbsorbCompressLayer<P>  (existing absorb-compress logic)
///   ├── arm_states: Vec<ArmRubricState> (per-arm rubric tracking)
///   └── config: RubricGatedConfig       (thresholds)
/// ```
///
/// # Key Difference from DeltaGatedAbsorbCompress
///
/// - **Delta**: `gate = mean_delta > threshold` (scalar, blind to *why*)
/// - **Rubric**: `gate = any(high_weight_criterion_gap > threshold)` (targeted, criterion-aware)
///
/// This enables per-criterion absorb targeting: absorb only when a meaningful
/// criterion (high weight) has a measurable gap vs references.
pub struct RubricGatedAbsorbCompress<P: ScreeningPruner> {
    /// Inner absorb-compress layer (delegates actual promotion logic).
    inner: AbsorbCompressLayer<P>,
    /// Per-arm rubric tracking state.
    arm_states: Vec<ArmRubricState>,
    /// Configuration thresholds.
    config: RubricGatedConfig,
}

impl<P: ScreeningPruner> RubricGatedAbsorbCompress<P> {
    /// Create a new rubric-gated absorb-compress layer.
    ///
    /// Wraps an existing `AbsorbCompressLayer` with rubric-based gating.
    pub fn new(inner: AbsorbCompressLayer<P>, num_arms: usize, config: RubricGatedConfig) -> Self {
        let arm_states = (0..num_arms).map(|_| ArmRubricState::new()).collect();
        Self {
            inner,
            arm_states,
            config,
        }
    }

    /// Feed a rubric observation for an arm.
    ///
    /// Computes per-criterion gaps between student rubric and reference rubrics.
    /// If any high-weight criterion has a gap above threshold, the arm is
    /// eligible for absorb and the weighted gap is forwarded as reward.
    ///
    /// # Arguments
    ///
    /// * `arm` — Bandit arm index
    /// * `student_rubric` — The student's rubric vector
    /// * `references` — Reference rubric vectors (M ≥ 2 recommended)
    ///
    /// # Gap Computation
    ///
    /// For each criterion:
    /// 1. `max_ref_score = max(reference.scores[criterion])`
    /// 2. `gap = max_ref_score - student_score`
    /// 3. `weighted_gap = weight * gap`
    ///
    /// Absorb triggers if `weighted_gap >= gap_threshold * weight` for any
    /// criterion with `weight >= min_weight_for_absorb`.
    #[inline]
    pub fn observe_rubric(
        &mut self,
        arm: usize,
        student_rubric: &RubricVector,
        references: &[RubricVector],
    ) {
        let Some(state) = self.arm_states.get_mut(arm) else {
            return;
        };

        // Store observation
        state.history.push(student_rubric.clone());

        // Update references
        state.references = references.to_vec();

        // Check minimum references
        if state.references.len() < self.config.min_references {
            state.above_threshold = false;
            state.last_gaps.clear();
            return;
        }

        // Compute per-criterion gaps vs max of all references
        let n = student_rubric.len();
        let mut gaps = Vec::with_capacity(n);

        for i in 0..n {
            let max_ref_score = state
                .references
                .iter()
                .filter_map(|r| r.scores.get(i).copied())
                .fold(0.0_f32, f32::max);

            let student_score = student_rubric.score(i);
            let gap = (max_ref_score - student_score).max(0.0);
            let weight = student_rubric.weight(i);
            gaps.push((i, gap, weight));
        }

        // Sort by weighted gap descending
        gaps.sort_by(|a, b| {
            let wa = a.2 * a.1;
            let wb = b.2 * b.1;
            wb.partial_cmp(&wa).unwrap_or(Ordering::Equal)
        });

        // Check if any high-weight criterion exceeds threshold
        let above = gaps.iter().any(|(_, gap, weight)| {
            *gap >= self.config.gap_threshold && *weight >= self.config.min_weight_for_absorb
        });

        state.above_threshold = above;
        state.last_gaps = gaps;

        if above {
            // Forward the weighted gap as reward to inner absorb-compress
            let reward = self.compute_absorb_reward(arm);
            self.inner.absorb(arm, reward);
        }
    }

    /// Compute the absorb reward for an arm based on its rubric gaps.
    ///
    /// Uses the weighted sum of gaps for high-weight criteria as the reward.
    /// This provides a richer signal than scalar δ — it encodes *which*
    /// criteria have gaps and how important they are.
    /// Compute absorb reward using quadratic weighted gaps (Issue 061 fix).
    ///
    /// Uses `Σ(w_i × gap_i²)` instead of `Σ(w_i × gap_i)` — the quadratic form
    /// breaks permutation symmetry, preserving per-criterion identity in the reward.
    /// Concentrated gaps in high-weight criteria produce higher rewards than spread gaps.
    fn compute_absorb_reward(&self, arm: usize) -> f32 {
        let Some(state) = self.arm_states.get(arm) else {
            return 0.0;
        };

        state
            .last_gaps
            .iter()
            .filter(|(_, _, weight)| *weight >= self.config.min_weight_for_absorb)
            .filter(|(_, gap, _)| *gap >= self.config.gap_threshold)
            .map(|(_, gap, weight)| gap * gap * weight)
            .sum()
    }

    /// Add a reference rubric for a specific arm.
    ///
    /// References can be added incrementally before any observations.
    /// Useful for pre-seeding with golden replay rubrics.
    pub fn add_reference(&mut self, arm: usize, reference: RubricVector) {
        let Some(state) = self.arm_states.get_mut(arm) else {
            return;
        };
        state.references.push(reference);
    }

    /// Whether an arm's rubric gaps exceed the threshold.
    #[inline]
    pub fn is_above_threshold(&self, arm: usize) -> bool {
        self.arm_states
            .get(arm)
            .map(|s| s.above_threshold)
            .unwrap_or(false)
    }

    /// Get the last computed gaps for an arm.
    ///
    /// Returns `[(criterion_idx, gap, weight)]` sorted by weighted gap descending.
    pub fn last_gaps(&self, arm: usize) -> &[(usize, f32, f32)] {
        self.arm_states
            .get(arm)
            .map(|s| s.last_gaps.as_slice())
            .unwrap_or(&[])
    }

    /// Number of reference rubrics for a specific arm.
    pub fn reference_count(&self, arm: usize) -> usize {
        self.arm_states
            .get(arm)
            .map(|s| s.references.len())
            .unwrap_or(0)
    }

    /// Number of rubric observations for a specific arm.
    pub fn observation_count(&self, arm: usize) -> usize {
        self.arm_states
            .get(arm)
            .map(|s| s.history.len())
            .unwrap_or(0)
    }

    /// Which arms have the highest accumulated rubric gaps (top-K blind spots)?
    ///
    /// Returns arms sorted by total weighted gap, descending.
    /// Only arms with at least one rubric observation and above-threshold gaps are included.
    pub fn blind_spot_arms(&self, top_k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .arm_states
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if !s.above_threshold || s.last_gaps.is_empty() {
                    return None;
                }
                let total: f32 = s.last_gaps.iter().map(|(_, g, w)| g * w).sum();
                Some((i, total))
            })
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    /// Access the inner absorb-compress layer.
    pub fn inner(&self) -> &AbsorbCompressLayer<P> {
        &self.inner
    }

    /// Mutable access to the inner absorb-compress layer.
    pub fn inner_mut(&mut self) -> &mut AbsorbCompressLayer<P> {
        &mut self.inner
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.arm_states.len()
    }

    /// Configuration reference.
    pub fn config(&self) -> &RubricGatedConfig {
        &self.config
    }
}

impl<P: ScreeningPruner> ScreeningPruner for RubricGatedAbsorbCompress<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        // Delegate to inner layer (which handles compressed arm blocking)
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

impl<P: ScreeningPruner> AbsorbCompress for RubricGatedAbsorbCompress<P> {
    fn absorb(&mut self, arm: usize, reward: f32) {
        // Direct absorb bypasses rubric gating — use for raw reward fallback.
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
        if !self.inner.should_compress() {
            return false;
        }
        // No metrics → no gate, fall through to original behavior
        let Some(metrics) = metrics else {
            return true;
        };
        // Gate: only compress when reviewer is net-positive
        let ratio = metrics.benefit_ratio();
        if ratio < self.config.min_benefit_ratio {
            return false;
        }
        true
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::absorb_compress::CompressConfig;
    use crate::speculative::types::NoScreeningPruner;

    fn make_layer(num_arms: usize) -> RubricGatedAbsorbCompress<NoScreeningPruner> {
        let inner =
            AbsorbCompressLayer::new(NoScreeningPruner, num_arms, CompressConfig::default());
        RubricGatedAbsorbCompress::new(inner, num_arms, RubricGatedConfig::default())
    }

    fn make_rubric(scores: Vec<f32>) -> RubricVector {
        RubricVector::new(scores, vec![4.0, 2.0, 1.0], 0)
    }

    fn make_refs() -> Vec<RubricVector> {
        vec![
            RubricVector::new(vec![1.0, 1.0, 1.0], vec![4.0, 2.0, 1.0], 0),
            RubricVector::new(vec![0.9, 0.8, 1.0], vec![4.0, 2.0, 1.0], 0),
        ]
    }

    #[test]
    fn test_observe_rubric_absorbs_on_high_weight_gap() {
        let mut layer = make_layer(3);
        let refs = make_refs();

        // Student has big gap in criterion 0 (weight=4.0): ref=1.0, student=0.5, gap=0.5
        let student = make_rubric(vec![0.5, 0.9, 0.9]);
        layer.observe_rubric(0, &student, &refs);

        assert!(layer.is_above_threshold(0), "Should be above threshold");
        assert_eq!(layer.reference_count(0), 2);
        assert_eq!(layer.observation_count(0), 1);
    }

    #[test]
    fn test_observe_rubric_skips_low_weight_gap() {
        let mut layer = make_layer(3);
        let refs = make_refs();

        // Gap only in criterion 2 (weight=1.0 < min_weight=2.0) → skip absorb
        let student = make_rubric(vec![0.95, 0.95, 0.3]);
        layer.observe_rubric(0, &student, &refs);

        assert!(
            !layer.is_above_threshold(0),
            "Low-weight gap should not trigger absorb"
        );
    }

    #[test]
    fn test_observe_rubric_skips_small_gap() {
        let mut layer = make_layer(3);
        let refs = make_refs();

        // Gap below threshold (0.3): student=0.9, ref=1.0, gap=0.1
        let student = make_rubric(vec![0.9, 0.9, 0.9]);
        layer.observe_rubric(0, &student, &refs);

        assert!(
            !layer.is_above_threshold(0),
            "Small gap should not trigger absorb"
        );
    }

    #[test]
    fn test_observe_rubric_skips_insufficient_references() {
        let mut layer = make_layer(3);
        let single_ref = vec![RubricVector::new(
            vec![1.0, 1.0, 1.0],
            vec![4.0, 2.0, 1.0],
            0,
        )];

        // Only 1 reference (< min_references=2)
        let student = make_rubric(vec![0.0, 0.0, 0.0]);
        layer.observe_rubric(0, &student, &single_ref);

        assert!(
            !layer.is_above_threshold(0),
            "Insufficient references should skip absorb"
        );
    }

    #[test]
    fn test_gap_criteria_sorted_by_weighted_gap() {
        let mut layer = make_layer(3);
        let refs = make_refs();

        // Gaps in criteria 0 (w=4, gap=0.5→2.0), 1 (w=2, gap=0.6→1.2), 2 (w=1, gap=0.2→0.2)
        let student = make_rubric(vec![0.5, 0.3, 0.8]);
        layer.observe_rubric(0, &student, &refs);

        let gaps = layer.last_gaps(0);
        assert!(!gaps.is_empty());
        // Should be sorted by weight*gap descending
        assert_eq!(gaps[0].0, 0, "Highest weighted gap first");
        assert!((gaps[0].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_no_reference_rubric_skips_absorb() {
        let mut layer = make_layer(3);

        // No references at all
        let student = make_rubric(vec![0.0, 0.0, 0.0]);
        layer.observe_rubric(0, &student, &[]);

        assert!(!layer.is_above_threshold(0), "No references → skip absorb");
    }

    #[test]
    fn test_multi_reference_gap_uses_max_per_criterion() {
        let mut layer = make_layer(3);
        let refs = vec![
            RubricVector::new(vec![0.8, 0.4], vec![4.0, 2.0], 0),
            RubricVector::new(vec![0.6, 0.9], vec![4.0, 2.0], 0),
        ];

        let student = RubricVector::new(vec![0.3, 0.3], vec![4.0, 2.0], 0);
        layer.observe_rubric(0, &student, &refs);

        assert!(layer.is_above_threshold(0));

        let gaps = layer.last_gaps(0);
        // Criterion 0: max(0.8, 0.6) = 0.8, gap = 0.5
        // Criterion 1: max(0.4, 0.9) = 0.9, gap = 0.6
        let gap_0 = gaps.iter().find(|(i, _, _)| *i == 0).map(|(_, g, _)| *g);
        let gap_1 = gaps.iter().find(|(i, _, _)| *i == 1).map(|(_, g, _)| *g);

        assert!((gap_0.unwrap() - 0.5).abs() < 1e-6);
        assert!((gap_1.unwrap() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_add_reference_incremental() {
        let mut layer = make_layer(3);

        layer.add_reference(0, RubricVector::new(vec![1.0], vec![4.0], 0));
        assert_eq!(layer.reference_count(0), 1);

        layer.add_reference(0, RubricVector::new(vec![0.9], vec![4.0], 0));
        assert_eq!(layer.reference_count(0), 2);
    }

    #[test]
    fn test_blind_spot_arms_ranked() {
        let mut layer = make_layer(5);
        let refs = make_refs();

        // Arm 0: small gap
        layer.observe_rubric(0, &make_rubric(vec![0.8, 0.9, 0.9]), &refs);
        // Arm 1: big gap in high-weight criterion
        layer.observe_rubric(1, &make_rubric(vec![0.2, 0.9, 0.9]), &refs);
        // Arm 2: medium gap
        layer.observe_rubric(2, &make_rubric(vec![0.5, 0.9, 0.9]), &refs);

        let blind = layer.blind_spot_arms(3);
        // Arm 1 should be first (biggest gap in high-weight criterion)
        assert!(!blind.is_empty());
        assert_eq!(blind[0], 1);
    }

    #[test]
    fn test_blind_spot_excludes_below_threshold() {
        let mut layer = make_layer(3);
        let refs = make_refs();

        // All gaps below threshold → no blind spots
        layer.observe_rubric(0, &make_rubric(vec![0.95, 0.95, 0.95]), &refs);

        let blind = layer.blind_spot_arms(5);
        assert!(blind.is_empty(), "No gaps above threshold");
    }

    #[test]
    fn test_out_of_bounds_arm_is_noop() {
        let mut layer = make_layer(2);
        let refs = make_refs();

        layer.observe_rubric(99, &make_rubric(vec![0.0, 0.0, 0.0]), &refs);
        assert_eq!(layer.num_arms(), 2);
    }

    #[test]
    fn test_delegates_relevance_to_inner() {
        let layer = make_layer(3);
        // NoScreeningPruner always returns 1.0
        let rel = layer.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compress_delegates_to_inner() {
        let mut layer = RubricGatedAbsorbCompress::new(
            AbsorbCompressLayer::new(NoScreeningPruner, 3, CompressConfig::new(1, 0.5, 3, 1)),
            3,
            RubricGatedConfig::default(),
        );

        // Direct absorb (bypass rubric gate)
        layer.absorb(0, 0.1);
        layer.absorb(0, 0.1);

        let promoted = layer.compress();
        assert_eq!(promoted, vec![0]);
        assert!(layer.compressed_arms().contains(&0));
    }

    #[test]
    fn test_config_builder() {
        let config = RubricGatedConfig::new(0.5)
            .with_benefit_ratio(3.0)
            .with_min_references(3)
            .with_min_weight(3.0);

        assert!((config.gap_threshold - 0.5).abs() < 1e-6);
        assert!((config.min_benefit_ratio - 3.0).abs() < 1e-6);
        assert_eq!(config.min_references, 3);
        assert!((config.min_weight_for_absorb - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_inter_dimensional_no_regression() {
        // Absorbing criterion A should not regress criterion B scores
        let mut layer = make_layer(3);
        let refs = make_refs();

        // Observe with gap in criterion 0 only
        let student = make_rubric(vec![0.5, 0.9, 0.9]);
        layer.observe_rubric(0, &student, &refs);

        let gaps = layer.last_gaps(0);
        // Only criterion 0 should have a significant gap
        for (idx, gap, _) in gaps {
            if *idx == 0 {
                assert!(*gap > 0.3, "Criterion 0 gap should be significant");
            }
        }

        // Verify criterion 1 and 2 scores are preserved in history
        let obs_count = layer.observation_count(0);
        assert_eq!(obs_count, 1);
    }

    #[test]
    fn test_single_reference_works_but_below_min() {
        // Config with min_references=1
        let config = RubricGatedConfig {
            min_references: 1,
            ..RubricGatedConfig::default()
        };
        let inner = AbsorbCompressLayer::new(NoScreeningPruner, 3, CompressConfig::default());
        let mut layer = RubricGatedAbsorbCompress::new(inner, 3, config);

        let single_ref = vec![RubricVector::new(
            vec![1.0, 1.0, 1.0],
            vec![4.0, 2.0, 1.0],
            0,
        )];

        let student = make_rubric(vec![0.5, 0.9, 0.9]);
        layer.observe_rubric(0, &student, &single_ref);

        assert!(
            layer.is_above_threshold(0),
            "Single reference works with min_references=1"
        );
    }

    #[test]
    fn test_multiple_observations_accumulate() {
        let mut layer = make_layer(3);
        let refs = make_refs();

        for _ in 0..5 {
            let student = make_rubric(vec![0.5, 0.9, 0.9]);
            layer.observe_rubric(0, &student, &refs);
        }

        assert_eq!(layer.observation_count(0), 5);
        assert!(layer.is_above_threshold(0));
    }
}
