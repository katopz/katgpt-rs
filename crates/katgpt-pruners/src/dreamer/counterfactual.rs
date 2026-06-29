//! Counterfactual dropout utility estimation for Auto-Dreamer.
//!
//! Estimates the utility of each merged arm group by measuring how much
//! the full set's utility drops when that group is removed. Uses Monte
//! Carlo dropout for robust estimation.

use katgpt_types::Rng;

use super::types::ReplacementSet;

/// Counterfactual dropout utility estimator.
///
/// Computes `utility(e) = U(S) − E[U(S \ {e})]` for each merged group `e`
/// in the replacement set, where random dropout of additional arms provides
/// regularization.
pub struct CounterfactualEstimator {
    /// Fraction of arms to randomly drop (paper: ρ=0.25-0.5).
    pub dropout_fraction: f32,
    /// Number of Monte Carlo samples (paper: M≥1).
    pub mc_samples: usize,
}

impl CounterfactualEstimator {
    pub fn new(dropout_fraction: f32, mc_samples: usize) -> Self {
        Self {
            dropout_fraction,
            mc_samples,
        }
    }

    /// Estimate utility of each merged arm group in the replacement set.
    ///
    /// `utility(e) = U(S) − E[U(S\\{e})]` for random dropout.
    /// Uses the provided evaluator function as utility signal.
    pub fn estimate_utility(
        &self,
        replacement: &ReplacementSet,
        evaluator: &dyn Fn(&[usize]) -> f32,
        rng: &mut Rng,
    ) -> Vec<f32> {
        if replacement.merged.is_empty() {
            return Vec::new();
        }

        // Full set utility
        let all_indices: Vec<usize> = replacement
            .merged
            .iter()
            .flat_map(|(indices, _)| indices.iter().copied())
            .collect();
        let u_full = evaluator(&all_indices);

        // Per-group utility via counterfactual dropout
        let mut utilities = Vec::with_capacity(replacement.merged.len());

        for (group_indices, _) in &replacement.merged {
            let mut utility_sum = 0.0f32;

            for _ in 0..self.mc_samples {
                // Create set without this group
                let without: Vec<usize> = all_indices
                    .iter()
                    .filter(|&&i| !group_indices.contains(&i))
                    .copied()
                    .collect();

                // Randomly drop additional fraction of other arms
                let without_dropped: Vec<usize> = without
                    .iter()
                    .filter(|_| rng.uniform() > self.dropout_fraction)
                    .copied()
                    .collect();

                let u_without = if without_dropped.is_empty() {
                    0.0
                } else {
                    evaluator(&without_dropped)
                };

                utility_sum += u_full - u_without;
            }

            utilities.push(utility_sum / self.mc_samples.max(1) as f32);
        }

        utilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_replacement(merged: Vec<(Vec<usize>, f32)>, forgotten: Vec<usize>) -> ReplacementSet {
        let utility = vec![0.0; merged.len()];
        ReplacementSet {
            merged,
            forgotten,
            utility,
        }
    }

    #[test]
    fn test_empty_replacement_returns_empty() {
        let estimator = CounterfactualEstimator::new(0.25, 1);
        let replacement = make_replacement(vec![], vec![]);
        let mut rng = Rng::new(42);
        let evaluator = |_: &[usize]| 1.0f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        assert!(utilities.is_empty());
    }

    #[test]
    fn test_single_group_zero_counterfactual_utility() {
        let estimator = CounterfactualEstimator::new(0.0, 1);
        let replacement = make_replacement(vec![(vec![0, 1], 0.5)], vec![]);
        let mut rng = Rng::new(42);
        // Evaluator returns sum of indices as utility
        let evaluator = |indices: &[usize]| indices.len() as f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        // Single group removed → empty set → u_without=0, utility = u_full - 0 = u_full
        assert_eq!(utilities.len(), 1);
        assert!(utilities[0] > 0.0);
    }

    #[test]
    fn test_two_groups_each_contributing() {
        let estimator = CounterfactualEstimator::new(0.0, 1);
        let replacement = make_replacement(vec![(vec![0, 1], 0.5), (vec![2, 3], 0.8)], vec![]);
        let mut rng = Rng::new(42);
        // Evaluator: count of arms
        let evaluator = |indices: &[usize]| indices.len() as f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        assert_eq!(utilities.len(), 2);
        // Removing group [0,1] leaves [2,3] → utility = 4 - 2 = 2
        assert!((utilities[0] - 2.0).abs() < f32::EPSILON);
        // Removing group [2,3] leaves [0,1] → utility = 4 - 2 = 2
        assert!((utilities[1] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_utility_non_negative_for_positive_evaluator() {
        let estimator = CounterfactualEstimator::new(0.25, 5);
        let replacement =
            make_replacement(vec![(vec![0], 0.5), (vec![1], 0.3), (vec![2], 0.7)], vec![]);
        let mut rng = Rng::new(123);
        let evaluator = |indices: &[usize]| indices.iter().sum::<usize>() as f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        assert_eq!(utilities.len(), 3);
        // All utilities should be >= 0 for monotonically increasing evaluator
        for u in &utilities {
            assert!(*u >= 0.0);
        }
    }

    #[test]
    fn test_mc_samples_averaging() {
        // With 1 sample, result varies; with many, it should stabilize
        let estimator = CounterfactualEstimator::new(0.5, 100);
        let replacement = make_replacement(vec![(vec![0, 1], 0.5)], vec![]);
        let mut rng = Rng::new(999);
        let evaluator = |indices: &[usize]| indices.len() as f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        // With dropout=0.5, roughly half of remaining arms survive
        // Average utility should be somewhere between 0 and 2 (full set size)
        assert!(utilities[0] > 0.0);
    }

    #[test]
    fn test_zero_dropout_no_randomness() {
        let estimator = CounterfactualEstimator::new(0.0, 1);
        let replacement = make_replacement(vec![(vec![0], 0.5), (vec![1], 0.5)], vec![]);
        let mut rng = Rng::new(42);
        let evaluator = |indices: &[usize]| indices.len() as f32;

        let u1 = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        let u2 = estimator.estimate_utility(&replacement, &evaluator, &mut rng);

        // With zero dropout and 1 sample, results are deterministic
        assert!((u1[0] - u2[0]).abs() < f32::EPSILON);
        assert!((u1[1] - u2[1]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_full_dropout_kills_all_remaining() {
        let estimator = CounterfactualEstimator::new(1.0, 1);
        let replacement = make_replacement(vec![(vec![0], 0.5), (vec![1], 0.5)], vec![]);
        let mut rng = Rng::new(42);
        let evaluator = |indices: &[usize]| indices.len() as f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        // With dropout=1.0, all remaining arms dropped → u_without=0 → utility = u_full
        assert!((utilities[0] - 2.0).abs() < f32::EPSILON);
        assert!((utilities[1] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_overlapping_groups() {
        let estimator = CounterfactualEstimator::new(0.0, 1);
        let replacement = make_replacement(vec![(vec![0, 1], 0.5), (vec![1, 2], 0.8)], vec![]);
        let mut rng = Rng::new(42);
        let evaluator = |indices: &[usize]| indices.iter().sum::<usize>() as f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        // all_indices via flat_map: [0, 1, 1, 2] → sum = 4
        // Remove group [0,1]: filter removes ALL 0s and 1s → [2] → sum = 2 → utility = 4 - 2 = 2
        assert!((utilities[0] - 2.0).abs() < f32::EPSILON);
        // Remove group [1,2]: filter removes ALL 1s and 2s → [0] → sum = 0 → utility = 4 - 0 = 4
        assert!((utilities[1] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_estimator_new_params() {
        let estimator = CounterfactualEstimator::new(0.3, 5);
        assert!((estimator.dropout_fraction - 0.3).abs() < f32::EPSILON);
        assert_eq!(estimator.mc_samples, 5);
    }

    #[test]
    fn test_large_replacement_set() {
        let merged: Vec<(Vec<usize>, f32)> =
            (0..100).map(|i| (vec![i], i as f32 / 100.0)).collect();
        let estimator = CounterfactualEstimator::new(0.25, 3);
        let replacement = make_replacement(merged, vec![]);
        let mut rng = Rng::new(77);
        let evaluator = |indices: &[usize]| indices.len() as f32;
        let utilities = estimator.estimate_utility(&replacement, &evaluator, &mut rng);
        assert_eq!(utilities.len(), 100);
        // Each group has 1 arm out of 100 → utility ≈ 100 - 99*dropout_survival
        for u in &utilities {
            assert!(*u >= 0.0);
        }
    }
}
