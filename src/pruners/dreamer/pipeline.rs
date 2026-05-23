//! High-level orchestration API for the Auto-Dreamer consolidation pipeline.
//!
//! Wires together scheduler → consolidator → decay → counterfactual into a
//! single `on_episode_complete()` call. Callers provide `ArmInfo` slices
//! (no direct dependency on `BanditPruner`).

use std::collections::HashSet;

use crate::types::Rng;

use super::consolidator::DreamerConsolidator;
use super::counterfactual::CounterfactualEstimator;
use super::decay::MemoryDecay;
use super::scheduler::{ArmInfo, DreamerScheduler};
use super::types::{DecayPolicy, DreamerConfig};

// ---------------------------------------------------------------------------
// ConsolidationResult
// ---------------------------------------------------------------------------

/// Result of a single consolidation cycle.
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Merged arm groups: (original indices in group, new averaged Q-value).
    pub merged: Vec<(Vec<usize>, f32)>,
    /// Arms marked for removal.
    pub forgotten: Vec<usize>,
    /// Arms outside the working region with decayed Q-values: (arm_index, decayed_q).
    pub decayed: Vec<(usize, f32)>,
    /// Counterfactual utility scores per merged group.
    pub utility: Vec<f32>,
    /// Episode at which this consolidation happened.
    pub episode: usize,
    /// Total arm count before consolidation.
    pub arms_before: usize,
    /// Total arm count after consolidation.
    pub arms_after: usize,
}

impl ConsolidationResult {
    /// Number of arms that were actually removed.
    pub fn arms_removed(&self) -> usize {
        let mut to_remove: HashSet<usize> = self.forgotten.iter().copied().collect();
        for (indices, _) in &self.merged {
            // Keep first arm in each group, remove the rest
            for &idx in indices.iter().skip(1) {
                to_remove.insert(idx);
            }
        }
        to_remove.len()
    }
}

// ---------------------------------------------------------------------------
// DreamerPipeline
// ---------------------------------------------------------------------------

/// High-level orchestration of the Dreamer consolidation cycle.
///
/// Composes scheduler, consolidator, decay, and counterfactual estimator
/// into a single `on_episode_complete()` entry point.
pub struct DreamerPipeline {
    config: DreamerConfig,
    scheduler: DreamerScheduler,
    consolidator: DreamerConsolidator,
    decay: MemoryDecay,
    counterfactual: CounterfactualEstimator,
    current_episode: usize,
    consolidation_count: usize,
}

impl DreamerPipeline {
    /// Create a new pipeline from config.
    pub fn new(config: DreamerConfig) -> Self {
        let decay_policy = DecayPolicy::Exponential {
            factor: config.decay_factor,
        };
        Self {
            scheduler: DreamerScheduler::new(config.clone()),
            consolidator: DreamerConsolidator::new(config.clone()),
            decay: MemoryDecay::new(decay_policy),
            counterfactual: CounterfactualEstimator::new(
                config.dropout_fraction,
                config.mc_samples,
            ),
            config,
            current_episode: 0,
            consolidation_count: 0,
        }
    }

    /// Current episode counter.
    pub fn episode(&self) -> usize {
        self.current_episode
    }

    /// Number of consolidations performed so far.
    pub fn consolidation_count(&self) -> usize {
        self.consolidation_count
    }

    /// Access the pipeline configuration.
    pub fn config(&self) -> &DreamerConfig {
        &self.config
    }

    /// Call after each episode completes.
    ///
    /// Increments the episode counter and, if consolidation should trigger,
    /// runs the full pipeline: select region → consolidate → decay → counterfactual.
    pub fn on_episode_complete(
        &mut self,
        arms: &[ArmInfo],
        rng: &mut Rng,
    ) -> Option<ConsolidationResult> {
        self.current_episode += 1;

        if !self.scheduler.should_consolidate(self.current_episode) {
            return None;
        }

        // Nothing to consolidate
        if arms.is_empty() {
            return None;
        }

        let arms_before = arms.len();

        // 1. Select working region
        let region = self.scheduler.select_region(arms, self.current_episode);

        if region.arm_indices.is_empty() {
            return None;
        }

        // 2. Consolidate region into replacement set
        let replacement = self.consolidator.consolidate(&region);

        // 3. Apply decay to arms NOT in the working region
        //    Build last_access map from arms
        let last_access: Vec<usize> = {
            let max_idx = arms.iter().map(|a| a.index).max().unwrap_or(0);
            let mut la = vec![0usize; max_idx + 1];
            for a in arms {
                la[a.index] = a.last_write_episode.max(a.last_retrieve_episode);
            }
            la
        };

        // Build full q_values vector indexed by arm index
        let q_values: Vec<f32> = {
            let max_idx = arms.iter().map(|a| a.index).max().unwrap_or(0);
            let mut qv = vec![0.0f32; max_idx + 1];
            for a in arms {
                qv[a.index] = a.q_value;
            }
            qv
        };

        let decayed = self
            .decay
            .apply(&q_values, &last_access, &region, self.current_episode);

        // 4. Counterfactual utility estimation
        let utility = self.counterfactual.estimate_utility(
            &replacement,
            &|indices: &[usize]| -> f32 {
                // Simple evaluator: sum of Q-values for given indices
                indices
                    .iter()
                    .map(|&i| q_values.get(i).copied().unwrap_or(0.0))
                    .sum()
            },
            rng,
        );

        // 5. Compute arms_after
        let mut to_remove: HashSet<usize> = replacement.forgotten.iter().copied().collect();
        for (indices, _) in &replacement.merged {
            // Keep first arm in each merged group, remove the rest
            for &idx in indices.iter().skip(1) {
                to_remove.insert(idx);
            }
        }
        let arms_after = arms_before.saturating_sub(to_remove.len());

        self.consolidation_count += 1;

        Some(ConsolidationResult {
            merged: replacement.merged,
            forgotten: replacement.forgotten,
            decayed,
            utility,
            episode: self.current_episode,
            arms_before,
            arms_after,
        })
    }

    /// Convert raw bandit data into `ArmInfo` for the scheduler.
    ///
    /// Uses a conservative mapping: treats any access as both read and write.
    pub fn extract_arm_info(
        q_values: &[f32],
        visits: &[u32],
        last_access: &[usize],
        _current_episode: usize,
    ) -> Vec<ArmInfo> {
        let len = q_values.len().min(visits.len()).min(last_access.len());
        (0..len)
            .map(|i| ArmInfo {
                index: i,
                q_value: q_values[i],
                visits: visits[i] as usize,
                last_write_episode: last_access[i],
                last_retrieve_episode: last_access[i],
            })
            .collect()
    }

    /// Apply consolidation results to bandit state in place.
    ///
    /// - Updates merged arm Q-values to averaged values
    /// - Applies decayed Q-values
    /// - Removes forgotten arms
    pub fn apply_consolidation(
        &self,
        q_values: &mut Vec<f32>,
        visits: &mut Vec<u32>,
        result: &ConsolidationResult,
    ) {
        if q_values.is_empty() {
            return;
        }

        // 1. Apply decayed Q-values (arms outside region)
        for &(idx, decayed_q) in &result.decayed {
            if idx < q_values.len() {
                q_values[idx] = decayed_q;
            }
        }

        // 2. Update merged arm Q-values
        //    First arm in each group gets the merged value; rest are marked for removal
        let mut indices_to_remove: HashSet<usize> = result.forgotten.iter().copied().collect();

        for (indices, merged_q) in &result.merged {
            if let Some(&first_idx) = indices.first() {
                if first_idx < q_values.len() {
                    q_values[first_idx] = *merged_q;
                    // Sum visits for merged group
                    let total_visits: u32 = indices
                        .iter()
                        .filter(|&&i| i < visits.len())
                        .map(|&i| visits[i])
                        .sum();
                    visits[first_idx] = total_visits;
                }
                // Non-first arms in group are removed
                for &idx in indices.iter().skip(1) {
                    indices_to_remove.insert(idx);
                }
            }
        }

        // 3. Remove forgotten + non-first merged arms
        //    Remove from highest index to lowest to preserve lower indices
        let mut sorted_removals: Vec<usize> = indices_to_remove.into_iter().collect();
        sorted_removals.sort_by(|a, b| b.cmp(a));

        for idx in sorted_removals {
            if idx < q_values.len() && idx < visits.len() {
                q_values.remove(idx);
                visits.remove(idx);
            }
        }
    }

    /// Reset pipeline state (e.g., for a new session).
    pub fn reset(&mut self) {
        self.current_episode = 0;
        self.consolidation_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pipeline(config: DreamerConfig) -> DreamerPipeline {
        DreamerPipeline::new(config)
    }

    fn make_arms(count: usize) -> Vec<ArmInfo> {
        (0..count)
            .map(|i| ArmInfo {
                index: i,
                q_value: (i as f32) / (count as f32),
                visits: i * 2 + 5,
                last_write_episode: i,
                last_retrieve_episode: i,
            })
            .collect()
    }

    // ---- ConsolidationResult tests ----

    #[test]
    fn test_consolidation_result_arms_removed_counts_forgotten_and_merged_extras() {
        let result = ConsolidationResult {
            merged: vec![(vec![0, 1, 2], 0.5), (vec![3], 0.8)],
            forgotten: vec![4, 5],
            decayed: vec![],
            utility: vec![0.3, 0.7],
            episode: 10,
            arms_before: 6,
            arms_after: 3,
        };
        // Merged group [0,1,2]: remove indices 1,2 (keep 0)
        // Merged group [3]: keep 3
        // Forgotten: 4, 5
        // Total removed: 1, 2, 4, 5 = 4
        assert_eq!(result.arms_removed(), 4);
    }

    #[test]
    fn test_consolidation_result_arms_removed_nothing() {
        let result = ConsolidationResult {
            merged: vec![(vec![0], 0.5)],
            forgotten: vec![],
            decayed: vec![],
            utility: vec![1.0],
            episode: 10,
            arms_before: 1,
            arms_after: 1,
        };
        assert_eq!(result.arms_removed(), 0);
    }

    // ---- DreamerPipeline::new ----

    #[test]
    fn test_new_initializes_episode_and_count() {
        let pipeline = make_pipeline(DreamerConfig::default());
        assert_eq!(pipeline.episode(), 0);
        assert_eq!(pipeline.consolidation_count(), 0);
    }

    // ---- on_episode_complete ----

    #[test]
    fn test_no_consolidation_before_cadence() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 10,
            ..DreamerConfig::default()
        });
        let arms = make_arms(10);
        let mut rng = Rng::new(42);

        for episode in 1..10 {
            let result = pipeline.on_episode_complete(&arms, &mut rng);
            assert!(
                result.is_none(),
                "Should not consolidate at episode {episode}"
            );
        }
        assert_eq!(pipeline.episode(), 9);
    }

    #[test]
    fn test_consolidation_at_cadence() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 10,
            min_visits: 1,
            region_fraction: 0.5,
            merge_threshold: 0.5,
            ..DreamerConfig::default()
        });
        let arms = make_arms(10);
        let mut rng = Rng::new(42);

        // Advance to episode 9 (next call increments to 10)
        for _ in 0..9 {
            let _ = pipeline.on_episode_complete(&arms, &mut rng);
        }

        let result = pipeline.on_episode_complete(&arms, &mut rng);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.episode, 10);
        assert_eq!(r.arms_before, 10);
        assert_eq!(pipeline.consolidation_count(), 1);
    }

    #[test]
    fn test_no_consolidation_with_empty_arms() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 1,
            ..DreamerConfig::default()
        });
        let mut rng = Rng::new(42);

        let result = pipeline.on_episode_complete(&[], &mut rng);
        assert!(result.is_none());
    }

    #[test]
    fn test_consolidation_multiple_cycles() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 5,
            min_visits: 1,
            region_fraction: 0.8,
            merge_threshold: 0.9,
            ..DreamerConfig::default()
        });
        // Arms with very recent activity so region selection works at episodes 5, 10, 15
        let arms: Vec<ArmInfo> = (0..10)
            .map(|i| ArmInfo {
                index: i,
                q_value: i as f32 / 10.0,
                visits: i * 2 + 5,
                last_write_episode: 100,
                last_retrieve_episode: 100,
            })
            .collect();
        let mut rng = Rng::new(42);

        let mut consolidation_episodes = Vec::new();
        for _ in 0..15 {
            if let Some(r) = pipeline.on_episode_complete(&arms, &mut rng) {
                consolidation_episodes.push(r.episode);
            }
        }

        assert_eq!(consolidation_episodes, vec![5, 10, 15]);
        assert_eq!(pipeline.consolidation_count(), 3);
    }

    #[test]
    fn test_result_contains_decayed_arms() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 1,
            min_visits: 1,
            region_fraction: 0.3,
            merge_threshold: 0.5,
            decay_factor: 0.9,
            ..DreamerConfig::default()
        });
        let arms = make_arms(10);
        let mut rng = Rng::new(42);

        let result = pipeline.on_episode_complete(&arms, &mut rng);
        assert!(result.is_some());
        let r = result.unwrap();
        // Some arms should be outside region and decayed
        assert!(!r.decayed.is_empty());
        // Decayed values should be less than or equal to original
        for &(idx, decayed_q) in &r.decayed {
            let original_q = arms.get(idx).map(|a| a.q_value).unwrap_or(0.0);
            assert!(decayed_q <= original_q + f32::EPSILON);
        }
    }

    #[test]
    fn test_result_contains_utility_scores() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 1,
            min_visits: 1,
            region_fraction: 0.5,
            merge_threshold: 0.9,
            ..DreamerConfig::default()
        });
        let arms = make_arms(10);
        let mut rng = Rng::new(42);

        let result = pipeline.on_episode_complete(&arms, &mut rng);
        assert!(result.is_some());
        let r = result.unwrap();
        // Should have utility scores for each merged group
        assert_eq!(r.utility.len(), r.merged.len());
    }

    // ---- extract_arm_info ----

    #[test]
    fn test_extract_arm_info_basic() {
        let q_values = vec![0.1, 0.5, 0.9];
        let visits = vec![5u32, 10, 15];
        let last_access = vec![3usize, 7, 9];

        let arms = DreamerPipeline::extract_arm_info(&q_values, &visits, &last_access, 10);

        assert_eq!(arms.len(), 3);
        assert_eq!(arms[0].index, 0);
        assert!((arms[0].q_value - 0.1).abs() < f32::EPSILON);
        assert_eq!(arms[0].visits, 5);
        assert_eq!(arms[0].last_write_episode, 3);
        assert_eq!(arms[0].last_retrieve_episode, 3);

        assert_eq!(arms[2].index, 2);
        assert!((arms[2].q_value - 0.9).abs() < f32::EPSILON);
        assert_eq!(arms[2].visits, 15);
    }

    #[test]
    fn test_extract_arm_info_mismatched_lengths() {
        let q_values = vec![0.1, 0.5, 0.9];
        let visits = vec![5u32, 10]; // shorter
        let last_access = vec![1usize, 2, 3, 4]; // longer

        let arms = DreamerPipeline::extract_arm_info(&q_values, &visits, &last_access, 10);
        // Should use minimum length
        assert_eq!(arms.len(), 2);
    }

    #[test]
    fn test_extract_arm_info_empty() {
        let arms = DreamerPipeline::extract_arm_info(&[], &[], &[], 0);
        assert!(arms.is_empty());
    }

    #[test]
    fn test_extract_arm_info_conservative_access_mapping() {
        let q_values = vec![0.5];
        let visits = vec![10u32];
        let last_access = vec![42usize];

        let arms = DreamerPipeline::extract_arm_info(&q_values, &visits, &last_access, 100);

        // Conservative: last_access used for both read and write
        assert_eq!(arms[0].last_write_episode, 42);
        assert_eq!(arms[0].last_retrieve_episode, 42);
    }

    // ---- apply_consolidation ----

    #[test]
    fn test_apply_consolidation_removes_forgotten() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let mut q_values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut visits = vec![5u32, 10, 15, 20, 25];

        let result = ConsolidationResult {
            merged: vec![],
            forgotten: vec![1, 3],
            decayed: vec![],
            utility: vec![],
            episode: 10,
            arms_before: 5,
            arms_after: 3,
        };

        pipeline.apply_consolidation(&mut q_values, &mut visits, &result);

        assert_eq!(q_values.len(), 3);
        assert_eq!(visits.len(), 3);
        // After removing indices 3 then 1: [0.1, 0.3, 0.5] → visits [5, 15, 25]
        assert!((q_values[0] - 0.1).abs() < f32::EPSILON);
        assert!((q_values[1] - 0.3).abs() < f32::EPSILON);
        assert!((q_values[2] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_consolidation_merges_groups() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let mut q_values = vec![0.1, 0.15, 0.5, 0.9];
        let mut visits = vec![5u32, 10, 15, 20];

        let result = ConsolidationResult {
            merged: vec![
                (vec![0, 1], 0.125), // merge arms 0 and 1
            ],
            forgotten: vec![],
            decayed: vec![],
            utility: vec![0.5],
            episode: 10,
            arms_before: 4,
            arms_after: 3,
        };

        pipeline.apply_consolidation(&mut q_values, &mut visits, &result);

        assert_eq!(q_values.len(), 3);
        // Arm 0 gets merged Q-value
        assert!((q_values[0] - 0.125).abs() < f32::EPSILON);
        // Arm 0 gets summed visits
        assert_eq!(visits[0], 15);
        // Remaining arms shifted
        assert!((q_values[1] - 0.5).abs() < f32::EPSILON);
        assert!((q_values[2] - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_consolidation_applies_decay() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let mut q_values = vec![1.0, 1.0, 1.0];
        let mut visits = vec![5u32, 5, 5];

        let result = ConsolidationResult {
            merged: vec![],
            forgotten: vec![],
            decayed: vec![(0, 0.9), (2, 0.8)],
            utility: vec![],
            episode: 10,
            arms_before: 3,
            arms_after: 3,
        };

        pipeline.apply_consolidation(&mut q_values, &mut visits, &result);

        assert!((q_values[0] - 0.9).abs() < f32::EPSILON);
        assert!((q_values[1] - 1.0).abs() < f32::EPSILON); // unchanged
        assert!((q_values[2] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_consolidation_full_workflow() {
        let pipeline = make_pipeline(DreamerConfig::default());
        // 5 arms: merge [0,1] into one, forget arm 4, decay arm 2
        let mut q_values = vec![0.1, 0.12, 0.8, 0.5, 0.01];
        let mut visits = vec![5u32, 10, 20, 15, 1];

        let result = ConsolidationResult {
            merged: vec![(vec![0, 1], 0.11)],
            forgotten: vec![4],
            decayed: vec![(2, 0.72)],
            utility: vec![0.6],
            episode: 10,
            arms_before: 5,
            arms_after: 3,
        };

        pipeline.apply_consolidation(&mut q_values, &mut visits, &result);

        // After: arm 0 → merged 0.11, arm 1 removed (merged), arm 2 → decayed 0.72, arm 3 → 0.5, arm 4 → removed
        assert_eq!(q_values.len(), 3);
        assert_eq!(visits.len(), 3);
        assert!((q_values[0] - 0.11).abs() < f32::EPSILON);
        assert!((q_values[1] - 0.72).abs() < f32::EPSILON);
        assert!((q_values[2] - 0.5).abs() < f32::EPSILON);
        assert_eq!(visits[0], 15); // 5 + 10 merged
    }

    #[test]
    fn test_apply_consolidation_empty_q_values() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let mut q_values: Vec<f32> = vec![];
        let mut visits: Vec<u32> = vec![];

        let result = ConsolidationResult {
            merged: vec![(vec![0], 0.5)],
            forgotten: vec![1],
            decayed: vec![],
            utility: vec![],
            episode: 10,
            arms_before: 0,
            arms_after: 0,
        };

        pipeline.apply_consolidation(&mut q_values, &mut visits, &result);
        assert!(q_values.is_empty());
        assert!(visits.is_empty());
    }

    #[test]
    fn test_apply_consolidation_out_of_bounds_indices_ignored() {
        let pipeline = make_pipeline(DreamerConfig::default());
        let mut q_values = vec![0.5];
        let mut visits = vec![10u32];

        let result = ConsolidationResult {
            merged: vec![],
            forgotten: vec![5],       // out of bounds
            decayed: vec![(10, 0.1)], // out of bounds
            utility: vec![],
            episode: 10,
            arms_before: 1,
            arms_after: 1,
        };

        pipeline.apply_consolidation(&mut q_values, &mut visits, &result);
        assert_eq!(q_values.len(), 1);
        assert!((q_values[0] - 0.5).abs() < f32::EPSILON);
    }

    // ---- reset ----

    #[test]
    fn test_reset_clears_counters() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 1,
            min_visits: 1,
            ..DreamerConfig::default()
        });
        let arms = make_arms(5);
        let mut rng = Rng::new(42);

        let _ = pipeline.on_episode_complete(&arms, &mut rng);
        let _ = pipeline.on_episode_complete(&arms, &mut rng);

        assert_eq!(pipeline.episode(), 2);
        assert_eq!(pipeline.consolidation_count(), 2);

        pipeline.reset();

        assert_eq!(pipeline.episode(), 0);
        assert_eq!(pipeline.consolidation_count(), 0);
    }

    // ---- end-to-end ----

    #[test]
    fn test_e2e_extract_then_consolidate_then_apply() {
        let config = DreamerConfig {
            cadence: 5,
            min_visits: 1,
            region_fraction: 0.6,
            merge_threshold: 0.3,
            decay_factor: 0.9,
            dropout_fraction: 0.0,
            mc_samples: 1,
        };
        let mut pipeline = DreamerPipeline::new(config);

        let q_values = vec![0.1, 0.12, 0.5, 0.55, 0.9];
        let visits = vec![5u32, 8, 12, 15, 20];
        let last_access = vec![3usize, 4, 3, 4, 5];
        let mut rng = Rng::new(42);

        let arms = DreamerPipeline::extract_arm_info(&q_values, &visits, &last_access, 5);

        // Advance to episode 5
        let mut result_opt = None;
        for _ in 0..5 {
            result_opt = pipeline.on_episode_complete(&arms, &mut rng);
        }

        assert!(result_opt.is_some());
        let result = result_opt.unwrap();
        assert_eq!(result.episode, 5);
        assert_eq!(result.arms_before, 5);

        // Apply consolidation
        let mut q_mut = q_values.clone();
        let mut v_mut = visits.clone();
        pipeline.apply_consolidation(&mut q_mut, &mut v_mut, &result);

        // Should have fewer arms after consolidation
        assert!(q_mut.len() <= 5);
        assert_eq!(q_mut.len(), v_mut.len());
    }

    #[test]
    fn test_e2e_conservative_config() {
        let mut pipeline = DreamerPipeline::new(DreamerConfig::conservative());
        let arms = make_arms(20);
        let mut rng = Rng::new(42);

        // Conservative cadence=20, so episode 20 should trigger
        let mut result_opt = None;
        for _ in 0..20 {
            result_opt = pipeline.on_episode_complete(&arms, &mut rng);
        }

        assert!(result_opt.is_some());
        assert_eq!(pipeline.consolidation_count(), 1);
    }

    #[test]
    fn test_e2e_aggressive_config() {
        let mut pipeline = DreamerPipeline::new(DreamerConfig::aggressive());
        let arms = make_arms(20);
        let mut rng = Rng::new(42);

        // Aggressive cadence=5, so episodes 5, 10, 15, 20 trigger
        let mut count = 0;
        for _ in 0..20 {
            if pipeline.on_episode_complete(&arms, &mut rng).is_some() {
                count += 1;
            }
        }

        assert_eq!(count, 4);
        assert_eq!(pipeline.consolidation_count(), 4);
    }

    #[test]
    fn test_arms_after_is_correctly_computed() {
        let mut pipeline = make_pipeline(DreamerConfig {
            cadence: 1,
            min_visits: 1,
            region_fraction: 0.9,
            merge_threshold: 0.8,
            ..DreamerConfig::default()
        });

        // Arms with similar Q-values that will merge
        let arms: Vec<ArmInfo> = (0..6)
            .map(|i| ArmInfo {
                index: i,
                q_value: if i < 4 { 0.5 } else { 0.9 },
                visits: 10,
                last_write_episode: 1,
                last_retrieve_episode: 1,
            })
            .collect();

        let mut rng = Rng::new(42);
        let result = pipeline.on_episode_complete(&arms, &mut rng);

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.arms_after <= r.arms_before);
        assert_eq!(r.arms_after, r.arms_before - r.arms_removed());
    }
}
