//! Strategic novelty metric to prevent bandit arm convergence (Plan 191).
//!
//! Implements the FrontierSmith divergence filter: before a new arm is
//! promoted, its score vector must be sufficiently distant from all
//! existing arms. This prevents the bandit from collapsing to a single
//! strategy.

use katgpt_core::PartialScorer;

// ── IdeaDivergence ───────────────────────────────────────────────

/// Strategic novelty metric to prevent bandit arm convergence.
///
/// Maintains score vectors for all active arms. A new arm is "novel"
/// if its minimum L2 distance to existing arms exceeds `novelty_threshold`.
pub struct IdeaDivergence {
    /// Score vectors for all active arms.
    arm_scores: Vec<Vec<f32>>,
    /// Minimum L2 distance for novelty.
    novelty_threshold: f32,
}

impl IdeaDivergence {
    /// Create a new divergence filter with the given novelty threshold.
    ///
    /// A threshold of 0.0 accepts all arms; a higher threshold requires
    /// more strategic diversity.
    #[inline]
    pub fn new(novelty_threshold: f32) -> Self {
        Self {
            arm_scores: Vec::new(),
            novelty_threshold: novelty_threshold.max(0.0),
        }
    }

    /// Check if a new arm's score vector is novel (min distance > threshold).
    ///
    /// Returns `true` if no existing arm is within `novelty_threshold`
    /// L2 distance, or if no arms have been registered yet.
    #[inline]
    pub fn is_novel(&self, scores: &[f32]) -> bool {
        if self.arm_scores.is_empty() {
            return true;
        }
        let mut min_dist = f32::MAX;
        for existing in &self.arm_scores {
            let d = Self::divergence(scores, existing);
            if d < min_dist {
                min_dist = d;
            }
        }
        min_dist > self.novelty_threshold
    }

    /// Compute normalized L2 distance between two score vectors.
    ///
    /// If vectors have different lengths, only the overlapping prefix
    /// is compared. Returns 0.0 for identical vectors.
    #[inline]
    pub fn divergence(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        if len == 0 {
            return 0.0;
        }
        let sum_sq: f32 = (0..len)
            .map(|i| {
                let diff = a[i] - b[i];
                diff * diff
            })
            .sum();
        sum_sq.sqrt()
    }

    /// Add a new arm's score vector.
    ///
    /// Call this after confirming novelty via [`is_novel()`].
    pub fn add_arm(&mut self, scores: Vec<f32>) {
        self.arm_scores.push(scores);
    }

    /// Number of registered arms.
    #[inline]
    pub fn arm_count(&self) -> usize {
        self.arm_scores.len()
    }

    /// Current novelty threshold.
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.novelty_threshold
    }

    /// Reset all registered arms.
    pub fn clear(&mut self) {
        self.arm_scores.clear();
    }

    /// Compute the minimum divergence from `scores` to any registered arm.
    ///
    /// Returns `f32::MAX` if no arms are registered.
    pub fn min_divergence(&self, scores: &[f32]) -> f32 {
        if self.arm_scores.is_empty() {
            return f32::MAX;
        }
        self.arm_scores
            .iter()
            .map(|existing| Self::divergence(scores, existing))
            .fold(f32::MAX, f32::min)
    }
}

// ── Free functions ───────────────────────────────────────────────

/// Convenience: check novelty of a scorer's breakdown against registered arms.
///
/// Uses [`PartialScorer::score_breakdown()`] to extract the score vector
/// then checks against the divergence filter.
pub fn is_scorer_novel<S: PartialScorer>(
    divergence: &IdeaDivergence,
    scorer: &S,
    trace: &katgpt_core::GameTrace,
) -> bool {
    let breakdown = scorer.score_breakdown(trace);
    let scores: Vec<f32> = breakdown.iter().map(|(_, v)| *v).collect();
    divergence.is_novel(&scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_always_novel() {
        let div = IdeaDivergence::new(0.5);
        assert!(div.is_novel(&[1.0, 0.0, 0.5]));
    }

    #[test]
    fn identical_scores_not_novel() {
        let mut div = IdeaDivergence::new(0.1);
        div.add_arm(vec![1.0, 0.5, 0.3]);
        assert!(!div.is_novel(&[1.0, 0.5, 0.3]));
    }

    #[test]
    fn distant_scores_are_novel() {
        let mut div = IdeaDivergence::new(0.5);
        div.add_arm(vec![1.0, 0.0, 0.0]);
        assert!(div.is_novel(&[0.0, 1.0, 0.0]));
    }

    #[test]
    fn divergence_identical_is_zero() {
        let d = IdeaDivergence::divergence(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((d - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn divergence_orthogonal_is_sqrt_n() {
        let d = IdeaDivergence::divergence(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn divergence_symmetric() {
        let a = [0.3, 0.7, 0.1];
        let b = [0.5, 0.2, 0.9];
        let d_ab = IdeaDivergence::divergence(&a, &b);
        let d_ba = IdeaDivergence::divergence(&b, &a);
        assert!((d_ab - d_ba).abs() < f32::EPSILON);
    }

    #[test]
    fn divergence_different_lengths() {
        // Only overlapping prefix compared
        let d = IdeaDivergence::divergence(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0]);
        assert!((d - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn divergence_empty() {
        let d = IdeaDivergence::divergence(&[], &[]);
        assert!((d - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn add_arm_increments_count() {
        let mut div = IdeaDivergence::new(0.5);
        assert_eq!(div.arm_count(), 0);
        div.add_arm(vec![1.0]);
        assert_eq!(div.arm_count(), 1);
        div.add_arm(vec![0.5]);
        assert_eq!(div.arm_count(), 2);
    }

    #[test]
    fn clear_resets() {
        let mut div = IdeaDivergence::new(0.5);
        div.add_arm(vec![1.0]);
        div.add_arm(vec![0.5]);
        div.clear();
        assert_eq!(div.arm_count(), 0);
        assert!(div.is_novel(&[1.0]));
    }

    #[test]
    fn threshold_negative_clamped_to_zero() {
        let div = IdeaDivergence::new(-1.0);
        assert!((div.threshold() - 0.0).abs() < f32::EPSILON);
        // With threshold 0.0: distance > 0.0 is false for identical, true for any delta
        let mut div2 = IdeaDivergence::new(-1.0);
        div2.add_arm(vec![1.0]);
        assert!(!div2.is_novel(&[1.0])); // distance=0, 0 > 0 = false
        assert!(div2.is_novel(&[1.01])); // distance=0.01, 0.01 > 0 = true
    }

    #[test]
    fn min_divergence_empty_is_max() {
        let div = IdeaDivergence::new(0.5);
        assert_eq!(div.min_divergence(&[1.0]), f32::MAX);
    }

    #[test]
    fn min_divergence_single_arm() {
        let mut div = IdeaDivergence::new(0.5);
        div.add_arm(vec![1.0, 0.0]);
        let d = div.min_divergence(&[0.0, 1.0]);
        assert!((d - 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn novelty_prevents_convergence() {
        let mut div = IdeaDivergence::new(0.3);
        // Arm 1: aggressive (high kills, low survival)
        div.add_arm(vec![0.2, 0.8, 0.1]);
        // Arm 2: nearly identical to arm 1 — should NOT be novel
        assert!(!div.is_novel(&[0.21, 0.79, 0.11]));
        // Arm 3: defensive (low kills, high survival) — should be novel
        assert!(div.is_novel(&[0.8, 0.1, 0.5]));
    }

    #[test]
    fn multiple_arms_min_distance() {
        let mut div = IdeaDivergence::new(0.5);
        div.add_arm(vec![1.0, 0.0]);
        div.add_arm(vec![0.0, 1.0]);
        div.add_arm(vec![0.5, 0.5]);
        // Query is equidistant from all three
        let md = div.min_divergence(&[0.5, 0.5]);
        assert!((md - 0.0).abs() < f32::EPSILON); // exact match with arm 3
    }
}

// TL;DR: IdeaDivergence tracks arm score vectors and rejects new arms that are too similar (L2 distance < threshold) to existing ones.
