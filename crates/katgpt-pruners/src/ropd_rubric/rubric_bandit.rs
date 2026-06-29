//! Bandit pruner using rubric-weighted scores as reward signal.
//!
//! Replaces [`DeltaBanditPruner`](super::super::g_zero::delta_bandit::DeltaBanditPruner)'s
//! scalar δ reward with rubric-weighted scores:
//!
//! - **Delta**: reward = `δ.value` (scalar, intrinsic, log-prob based)
//! - **Rubric**: reward = `weighted_gap` (vector → scalar, criterion-aware)
//!
//! # Why rubric-reward is more aligned than δ-reward
//!
//! ROPD shows teacher logit AUC = 0.35 (near random) vs rubric AUC = 0.90.
//! Our `HintDelta` is also log-prob-based. If δ shares logit's misalignment,
//! rubric vectors provide a more correctness-aligned signal.
//!
//! # Reward Computation
//!
//! ```text
//! reward = (reference.weighted_score() - student.weighted_score()) / max_score
//! ```
//!
//! Where `max_score` normalizes to [0, 1] range. Positive reward = student
//! outperforms reference, negative = student underperforms.
//!
//! # Usage
//!
//! ```rust,ignore
//! let pruner = RubricBanditPruner::new(
//!     BanditPruner::new(domain_screener, BanditStrategy::Ucb1, 10),
//!     10,
//! );
//!
//! // Feed rubric observation
//! pruner.observe_rubric(3, &student_rubric, &reference_rubric);
//!
//! // Find blind spots
//! let blind = pruner.blind_spot_arms(3);
//! ```

use std::cmp::Ordering;

use crate::bandit::BanditPruner;
use katgpt_speculative::ScreeningPruner;

use super::types::RubricVector;

// ── Config ──────────────────────────────────────────────────────

/// Configuration for [`RubricBanditPruner`].
#[derive(Clone, Debug)]
pub struct RubricBanditConfig {
    /// Normalize rewards by max possible score (default: true).
    ///
    /// When true, reward ∈ [−1.0, 1.0]. When false, raw weighted gap.
    pub normalize_reward: bool,
    /// Use per-criterion sub-bandits instead of scalar reward (default: false).
    ///
    /// Per-criterion bandits allow fine-grained arm selection per quality axis.
    /// Disabled by default — start simple, enable if scalar reward doesn't converge.
    pub per_criterion_bandits: bool,
    /// Reward floor — minimum reward to feed bandit (default: 0.0).
    ///
    /// Negative rewards (student outperforms reference) are set to floor.
    /// Set to negative values to allow negative reward signals.
    pub reward_floor: f32,
}

impl Default for RubricBanditConfig {
    fn default() -> Self {
        Self {
            normalize_reward: true,
            per_criterion_bandits: false,
            reward_floor: 0.0,
        }
    }
}

impl RubricBanditConfig {
    /// Create config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable per-criterion sub-bandits.
    pub fn with_per_criterion(mut self) -> Self {
        self.per_criterion_bandits = true;
        self
    }

    /// Set custom reward floor.
    pub fn with_reward_floor(mut self, floor: f32) -> Self {
        self.reward_floor = floor;
        self
    }

    /// Disable reward normalization.
    pub fn without_normalization(mut self) -> Self {
        self.normalize_reward = false;
        self
    }
}

// ── RubricBanditPruner ──────────────────────────────────────────

/// Bandit pruner using rubric-weighted scores as reward signal.
///
/// Wraps [`BanditPruner`] and feeds rubric-based rewards instead of δ.
/// The reward is computed as the gap between student and reference weighted scores,
/// providing a criterion-aware quality signal.
///
/// # Architecture
///
/// ```text
/// RubricBanditPruner<P>
///   ├── inner: BanditPruner<P>       (existing bandit logic)
///   ├── rubric_history: Vec<RubricVector>  (per-arm rubric observations)
///   ├── reference_rubrics: Vec<RubricVector> (per-arm best references)
///   ├── config: RubricBanditConfig
///   └── num_criteria: usize           (criteria count from template)
/// ```
///
/// # Reward Signal (Issue 061 fix)
///
/// `reward = student.quadratic_weighted_reward(reference)` — uses quadratic weighted gaps
/// instead of scalar `weighted_score()` difference. This preserves per-criterion identity:
/// two `RubricVector`s with the same `weighted_score()` but different gap profiles produce
/// **different** rewards, enabling the bandit to learn criterion-aware strategies.
///
/// Quadratic: `Σ(w_i × gap_i²) / Σ(w_i)` penalizes concentrated failures more than spread.
/// This fixes the scalar collapse where `RubricPlayer ≡ GZeroPlayer`.
pub struct RubricBanditPruner<P: ScreeningPruner> {
    /// Inner bandit pruner (delegates arm selection logic).
    inner: BanditPruner<P>,
    /// Per-arm accumulated rubric scores.
    rubric_history: Vec<Vec<RubricVector>>,
    /// Per-arm best reference rubric (highest weighted score).
    reference_rubrics: Vec<Option<RubricVector>>,
    /// Per-arm cumulative reward from rubric observations.
    cumulative_rewards: Vec<f32>,
    /// Per-arm observation count.
    observation_counts: Vec<usize>,
    /// Number of criteria per rubric.
    num_criteria: usize,
    /// Configuration.
    config: RubricBanditConfig,
}

impl<P: ScreeningPruner> RubricBanditPruner<P> {
    /// Create a new rubric-driven bandit pruner.
    ///
    /// Wraps an existing `BanditPruner` with rubric-based reward feeding.
    ///
    /// # Arguments
    ///
    /// * `inner` — Existing bandit pruner to wrap
    /// * `num_arms` — Number of bandit arms
    /// * `num_criteria` — Number of rubric criteria per observation
    pub fn new(inner: BanditPruner<P>, num_arms: usize, num_criteria: usize) -> Self {
        Self::with_config(inner, num_arms, num_criteria, RubricBanditConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(
        inner: BanditPruner<P>,
        num_arms: usize,
        num_criteria: usize,
        config: RubricBanditConfig,
    ) -> Self {
        Self {
            inner,
            rubric_history: (0..num_arms).map(|_| Vec::new()).collect(),
            reference_rubrics: vec![None; num_arms],
            cumulative_rewards: vec![0.0; num_arms],
            observation_counts: vec![0; num_arms],
            num_criteria,
            config,
        }
    }

    /// Feed a rubric observation as reward to the bandit.
    ///
    /// Uses [`RubricVector::quadratic_weighted_reward()`] to compute reward that preserves
    /// per-criterion gap identity (fixes Issue 061 scalar collapse). Two rubrics with the
    /// same `weighted_score()` but different gap profiles produce different rewards.
    ///
    /// The bandit's Q-value for this arm is updated incrementally.
    ///
    /// # Arguments
    ///
    /// * `arm` — Bandit arm index
    /// * `student_rubric` — The student's rubric vector
    /// * `reference_rubric` — The reference rubric to compare against
    #[inline]
    pub fn observe_rubric(
        &mut self,
        arm: usize,
        student_rubric: &RubricVector,
        reference_rubric: &RubricVector,
    ) {
        let Some(history) = self.rubric_history.get_mut(arm) else {
            return;
        };

        // Store observation
        history.push(student_rubric.clone());

        // Update best reference (keep highest weighted score)
        let best_ref = self.reference_rubrics.get_mut(arm);
        if let Some(Some(current_best)) = best_ref {
            if reference_rubric.weighted_score() > current_best.weighted_score() {
                *current_best = reference_rubric.clone();
            }
        } else if let Some(slot) = best_ref {
            *slot = Some(reference_rubric.clone());
        }

        // Compute reward: gap between reference and student
        let reward = self.compute_reward(student_rubric, reference_rubric);

        // Update cumulative stats
        // SAFETY: arm bounds checked above via get_mut on rubric_history
        unsafe {
            *self.cumulative_rewards.get_unchecked_mut(arm) += reward;
            *self.observation_counts.get_unchecked_mut(arm) += 1;
        }

        // Feed as reward to inner bandit
        self.inner.update(arm, reward);
    }

    /// Feed rubric with multiple references — uses best reference for reward.
    ///
    /// Selects the reference with the highest weighted score as the comparison baseline.
    /// This is the multi-reference analog of single-reference [`observe_rubric`].
    ///
    /// # Arguments
    ///
    /// * `arm` — Bandit arm index
    /// * `student_rubric` — The student's rubric vector
    /// * `references` — Multiple reference rubrics (M ≥ 2 recommended)
    #[inline]
    pub fn observe_rubric_multi(
        &mut self,
        arm: usize,
        student_rubric: &RubricVector,
        references: &[RubricVector],
    ) {
        let best_ref = references.iter().max_by(|a, b| {
            a.weighted_score()
                .partial_cmp(&b.weighted_score())
                .unwrap_or(Ordering::Equal)
        });

        match best_ref {
            Some(reference) => self.observe_rubric(arm, student_rubric, reference),
            None => {
                // No references — store observation but don't update bandit
                if let Some(history) = self.rubric_history.get_mut(arm) {
                    history.push(student_rubric.clone());
                }
            }
        }
    }

    /// Compute reward from student and reference rubrics using quadratic weighted gaps.
    ///
    /// Fixes Issue 061: uses [`RubricVector::quadratic_weighted_reward()`] which computes
    /// `Σ(w_i × gap_i²) / Σ(w_i)` instead of scalar `weighted_score()` difference.
    /// This breaks permutation symmetry — concentrated gaps in high-weight criteria
    /// produce higher rewards than spread gaps, even when linear weighted sum is identical.
    fn compute_reward(&self, student: &RubricVector, reference: &RubricVector) -> f32 {
        let reward = student.quadratic_weighted_reward(reference);

        let reward = if self.config.normalize_reward {
            // Quadratic reward is already normalized by total weight.
            // Cap at [0.0, 1.0] range for bandit stability.
            reward.clamp(0.0, 1.0)
        } else {
            reward
        };

        reward.max(self.config.reward_floor)
    }

    /// Mean reward for a specific arm.
    ///
    /// Returns 0.0 for unobserved arms.
    #[inline]
    pub fn mean_reward(&self, arm: usize) -> f32 {
        let Some(&count) = self.observation_counts.get(arm) else {
            return 0.0;
        };
        if count == 0 {
            return 0.0;
        }
        let total = self.cumulative_rewards.get(arm).copied().unwrap_or(0.0);
        total / count as f32
    }

    /// Cumulative reward for a specific arm.
    #[inline]
    pub fn total_reward(&self, arm: usize) -> f32 {
        self.cumulative_rewards.get(arm).copied().unwrap_or(0.0)
    }

    /// Number of rubric observations for a specific arm.
    #[inline]
    pub fn observation_count(&self, arm: usize) -> usize {
        self.observation_counts.get(arm).copied().unwrap_or(0)
    }

    /// Which arms have the highest accumulated rubric gaps (top-K blind spots)?
    ///
    /// Returns arms sorted by total cumulative reward, descending.
    /// Arms with high cumulative reward have the most room for improvement.
    /// Only arms with at least one observation are included.
    pub fn blind_spot_arms(&self, top_k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .cumulative_rewards
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, r)| *r > 0.0 && self.observation_counts.get(*i).copied().unwrap_or(0) > 0)
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
    }

    /// Best reference rubric for a specific arm (if any).
    pub fn best_reference(&self, arm: usize) -> Option<&RubricVector> {
        self.reference_rubrics.get(arm).and_then(|opt| opt.as_ref())
    }

    /// Last observed rubric for a specific arm (if any).
    pub fn last_rubric(&self, arm: usize) -> Option<&RubricVector> {
        self.rubric_history.get(arm).and_then(|h| h.last())
    }

    /// Access the inner bandit pruner.
    pub fn inner(&self) -> &BanditPruner<P> {
        &self.inner
    }

    /// Mutable access to the inner bandit pruner.
    pub fn inner_mut(&mut self) -> &mut BanditPruner<P> {
        &mut self.inner
    }

    /// Number of arms tracked.
    pub fn num_arms(&self) -> usize {
        self.cumulative_rewards.len()
    }

    /// Number of criteria per rubric.
    pub fn num_criteria(&self) -> usize {
        self.num_criteria
    }

    /// Configuration reference.
    pub fn config(&self) -> &RubricBanditConfig {
        &self.config
    }
}

// Delegate ScreeningPruner to inner BanditPruner
impl<P: ScreeningPruner> ScreeningPruner for RubricBanditPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        self.inner.relevance(depth, token_idx, parent_tokens)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BanditStrategy;
    use katgpt_speculative::NoScreeningPruner;

    fn make_pruner(num_arms: usize, num_criteria: usize) -> RubricBanditPruner<NoScreeningPruner> {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
        RubricBanditPruner::new(inner, num_arms, num_criteria)
    }

    fn make_pruner_with_config(
        num_arms: usize,
        num_criteria: usize,
        config: RubricBanditConfig,
    ) -> RubricBanditPruner<NoScreeningPruner> {
        let inner = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
        RubricBanditPruner::with_config(inner, num_arms, num_criteria, config)
    }

    fn make_rubric(scores: Vec<f32>) -> RubricVector {
        RubricVector::new(scores, vec![4.0, 2.0, 1.0], 0)
    }

    fn make_reference() -> RubricVector {
        RubricVector::new(vec![1.0, 1.0, 1.0], vec![4.0, 2.0, 1.0], 0)
    }

    #[test]
    fn test_observe_rubric_feeds_reward() {
        let mut pruner = make_pruner(3, 3);
        let reference = make_reference();

        // Student scores 0.5 on all criteria → gap = 1.0 - weighted_score(student)
        let student = make_rubric(vec![0.5, 0.5, 0.5]);
        pruner.observe_rubric(2, &student, &reference);

        // Should have accumulated reward
        assert!(pruner.total_reward(2) > 0.0, "Reward should be positive");
        assert_eq!(pruner.observation_count(2), 1);
    }

    #[test]
    fn test_reward_positive_when_student_underperforms() {
        let mut pruner = make_pruner(3, 3);
        let reference = make_reference();

        // Student: scores=[0.3, 0.3, 0.3], gaps=[0.7, 0.7, 0.7]
        // Quadratic: (4*0.7² + 2*0.7² + 1*0.7²) / 7 = (1.96+0.98+0.49)/7 = 3.43/7 ≈ 0.49
        let student = make_rubric(vec![0.3, 0.3, 0.3]);
        pruner.observe_rubric(0, &student, &reference);

        let reward = pruner.total_reward(0);
        assert!(
            reward > 0.0,
            "Positive reward when student underperforms, got {reward}"
        );
        assert!(
            (reward - 0.49).abs() < 0.01,
            "Expected ~0.49 (quadratic), got {reward}"
        );
    }

    #[test]
    fn test_reward_zero_when_student_outperforms() {
        let mut pruner = make_pruner(3, 3);
        let reference = make_rubric(vec![0.3, 0.3, 0.3]);

        // Student: weighted_score ≈ 1.0 > reference weighted_score ≈ 0.3
        let student = make_reference();
        pruner.observe_rubric(0, &student, &reference);

        let reward = pruner.total_reward(0);
        // Clamped to floor (0.0)
        assert!(
            (reward).abs() < 1e-6,
            "Reward clamped to floor when student outperforms, got {reward}"
        );
    }

    #[test]
    fn test_reward_custom_floor() {
        // Quadratic reward is always >= 0 (gaps clamped to max(0, ...)).
        // Test positive floor: small gap student gets reward raised to floor.
        let config = RubricBanditConfig::new().with_reward_floor(0.5);
        let mut pruner = make_pruner_with_config(3, 3, config);
        let reference = make_reference();

        // Small gap: gaps=[0.2, 0.2, 0.2]
        // quad = (4*0.04 + 2*0.04 + 1*0.04)/7 = 0.28/7 = 0.04
        // With floor=0.5: reward = max(0.04, 0.5) = 0.5
        let student = make_rubric(vec![0.8, 0.8, 0.8]);
        pruner.observe_rubric(0, &student, &reference);

        let reward = pruner.total_reward(0);
        assert!(
            (reward - 0.5).abs() < 0.01,
            "Reward clamped to floor=0.5, got {reward}"
        );
    }

    #[test]
    fn test_reward_not_normalized() {
        let config = RubricBanditConfig::new().without_normalization();
        let mut pruner = make_pruner_with_config(3, 3, config);
        let reference = make_reference();

        let student = make_rubric(vec![0.5, 0.5, 0.5]);
        pruner.observe_rubric(0, &student, &reference);

        let reward = pruner.total_reward(0);
        // Without normalization, gap = raw weighted score difference
        assert!(reward > 0.0, "Should be positive");
    }

    #[test]
    fn test_reward_zero_when_scores_match() {
        let mut pruner = make_pruner(3, 3);
        let student = make_rubric(vec![0.5, 0.5, 0.5]);
        let reference = make_rubric(vec![0.5, 0.5, 0.5]);

        pruner.observe_rubric(0, &student, &reference);

        let reward = pruner.total_reward(0);
        assert!(
            (reward).abs() < 1e-6,
            "Zero reward when scores match, got {reward}"
        );
    }

    #[test]
    fn test_blind_spot_arms_ranked() {
        let mut pruner = make_pruner(5, 3);
        let reference = make_reference();

        // Arm 0: small gap
        pruner.observe_rubric(0, &make_rubric(vec![0.8, 0.9, 0.9]), &reference);
        // Arm 1: big gap
        pruner.observe_rubric(1, &make_rubric(vec![0.2, 0.3, 0.3]), &reference);
        // Arm 2: medium gap
        pruner.observe_rubric(2, &make_rubric(vec![0.5, 0.5, 0.5]), &reference);

        let blind = pruner.blind_spot_arms(3);
        assert_eq!(blind[0], 1, "Arm 1 has biggest gap");
        assert!(blind.contains(&2), "Arm 2 is second");
    }

    #[test]
    fn test_blind_spot_excludes_unobserved() {
        let pruner = make_pruner(3, 3);
        let blind = pruner.blind_spot_arms(5);
        assert!(blind.is_empty(), "No observations = no blind spots");
    }

    #[test]
    fn test_blind_spot_top_k_limits() {
        let mut pruner = make_pruner(5, 3);
        let reference = make_reference();

        for i in 0..5 {
            let student = make_rubric(vec![1.0 - i as f32 * 0.1; 3]);
            pruner.observe_rubric(i, &student, &reference);
        }

        let blind = pruner.blind_spot_arms(2);
        assert_eq!(blind.len(), 2);
    }

    #[test]
    fn test_mean_reward() {
        let mut pruner = make_pruner(3, 3);
        let reference = make_reference();

        pruner.observe_rubric(0, &make_rubric(vec![0.5, 0.5, 0.5]), &reference);
        pruner.observe_rubric(0, &make_rubric(vec![0.6, 0.6, 0.6]), &reference);

        let mean = pruner.mean_reward(0);
        assert!(mean > 0.0, "Mean should be positive");
        assert_eq!(pruner.observation_count(0), 2);
    }

    #[test]
    fn test_mean_reward_unobserved() {
        let pruner = make_pruner(3, 3);
        assert!((pruner.mean_reward(2)).abs() < 1e-6);
    }

    #[test]
    fn test_observe_rubric_multi_uses_best_reference() {
        let mut pruner = make_pruner(3, 3);
        let student = make_rubric(vec![0.3, 0.3, 0.3]);

        let ref_low = make_rubric(vec![0.5, 0.5, 0.5]);
        let ref_high = make_rubric(vec![1.0, 1.0, 1.0]);

        pruner.observe_rubric_multi(0, &student, &[ref_low, ref_high]);

        let reward = pruner.total_reward(0);
        // Should use ref_high: gaps=[0.7, 0.7, 0.7]
        // quad = (4*0.49 + 2*0.49 + 1*0.49)/7 = 3.43/7 ≈ 0.49
        assert!(
            (reward - 0.49).abs() < 0.01,
            "Should use best reference (quadratic ≈0.49), got {reward}"
        );
    }

    #[test]
    fn test_observe_rubric_multi_empty_references() {
        let mut pruner = make_pruner(3, 3);
        let student = make_rubric(vec![0.5, 0.5, 0.5]);

        pruner.observe_rubric_multi(0, &student, &[]);

        // No references → no reward update, no formal observation counted
        assert_eq!(pruner.observation_count(0), 0);
        assert!((pruner.total_reward(0)).abs() < 1e-6);
    }

    #[test]
    fn test_best_reference_updated() {
        let mut pruner = make_pruner(3, 3);
        let student = make_rubric(vec![0.5, 0.5, 0.5]);

        let ref_low = make_rubric(vec![0.6, 0.6, 0.6]);
        pruner.observe_rubric(0, &student, &ref_low);

        assert!(pruner.best_reference(0).is_some());
        assert!(
            (pruner.best_reference(0).unwrap().weighted_score() - ref_low.weighted_score()).abs()
                < 1e-6
        );

        // Better reference replaces old one
        let ref_high = make_reference();
        pruner.observe_rubric(0, &student, &ref_high);

        assert!(
            (pruner.best_reference(0).unwrap().weighted_score() - ref_high.weighted_score()).abs()
                < 1e-6
        );
    }

    #[test]
    fn test_last_rubric() {
        let mut pruner = make_pruner(3, 3);
        let reference = make_reference();

        let student1 = make_rubric(vec![0.5, 0.5, 0.5]);
        let student2 = make_rubric(vec![0.6, 0.6, 0.6]);

        pruner.observe_rubric(0, &student1, &reference);
        pruner.observe_rubric(0, &student2, &reference);

        let last = pruner.last_rubric(0).unwrap();
        assert!((last.score(0) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_out_of_bounds_arm_is_noop() {
        let mut pruner = make_pruner(2, 3);
        let reference = make_reference();
        let student = make_rubric(vec![0.5, 0.5, 0.5]);

        pruner.observe_rubric(99, &student, &reference);

        assert_eq!(pruner.num_arms(), 2);
        assert!((pruner.total_reward(0)).abs() < 1e-6);
    }

    #[test]
    fn test_delegates_relevance_to_inner() {
        let pruner = make_pruner(3, 3);

        // NoScreeningPruner always returns 1.0
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_num_criteria() {
        let pruner = make_pruner(3, 5);
        assert_eq!(pruner.num_criteria(), 5);
    }

    #[test]
    fn test_config_builder() {
        let config = RubricBanditConfig::new()
            .with_per_criterion()
            .with_reward_floor(-0.5)
            .without_normalization();

        assert!(config.per_criterion_bandits);
        assert!((config.reward_floor - (-0.5)).abs() < 1e-6);
        assert!(!config.normalize_reward);
    }

    #[test]
    fn test_multiple_observations_accumulate() {
        let mut pruner = make_pruner(2, 3);
        let reference = make_reference();

        for _ in 0..10 {
            let student = make_rubric(vec![0.5, 0.5, 0.5]);
            pruner.observe_rubric(0, &student, &reference);
        }

        assert_eq!(pruner.observation_count(0), 10);
        let mean = pruner.mean_reward(0);
        // Each observation contributes ~0.5 gap
        assert!(mean > 0.0, "Mean reward should be positive");
    }

    #[test]
    fn test_bandit_converges_toward_better_arms() {
        let mut pruner = make_pruner(3, 3);

        // Arm 0: student always underperforms (big gap) → high reward signal
        // Arm 1: student always matches (no gap) → zero reward
        // Arm 2: student always outperforms (negative gap clamped) → zero reward
        for _ in 0..100 {
            let reference = make_reference();
            pruner.observe_rubric(0, &make_rubric(vec![0.2, 0.2, 0.2]), &reference);
            pruner.observe_rubric(1, &make_rubric(vec![1.0, 1.0, 1.0]), &reference);
            pruner.observe_rubric(2, &make_rubric(vec![1.0, 1.0, 1.0]), &reference);
        }

        // Arm 0 should have highest cumulative reward (biggest blind spot)
        assert!(
            pruner.total_reward(0) > pruner.total_reward(1),
            "Arm 0 should have higher reward than arm 1"
        );
        assert!(
            pruner.total_reward(0) > pruner.total_reward(2),
            "Arm 0 should have higher reward than arm 2"
        );
    }
}
