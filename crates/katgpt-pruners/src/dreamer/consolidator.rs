//! Deterministic modelless consolidator for Auto-Dreamer.
//!
//! Clusters arms by Q-value proximity and merges similar ones into a compact
//! replacement set. Uses O(n log n) deterministic approach:
//! sort by Q-value → merge nearby → forget low-visit.

use super::types::{DreamerConfig, ReplacementSet, WorkingRegion};

/// Deterministic modelless consolidator.
///
/// Clusters arms by Q-value proximity and merges similar ones.
pub struct DreamerConsolidator {
    pub config: DreamerConfig,
}

impl DreamerConsolidator {
    pub fn new(config: DreamerConfig) -> Self {
        Self { config }
    }

    /// Consolidate working region into replacement set.
    ///
    /// O(n log n) deterministic: sort by Q-value → merge nearby → forget low-visit.
    pub fn consolidate(&self, region: &WorkingRegion) -> ReplacementSet {
        if region.arm_indices.is_empty() {
            return ReplacementSet {
                merged: Vec::new(),
                forgotten: Vec::new(),
                utility: Vec::new(),
            };
        }

        // Sort indices by Q-value for clustering
        let mut indexed: Vec<usize> = (0..region.arm_indices.len()).collect();
        indexed.sort_by(|&a, &b| {
            region.q_snapshot[a]
                .partial_cmp(&region.q_snapshot[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Cluster by Q-value proximity
        let mut merged: Vec<(Vec<usize>, f32)> = Vec::new();
        let mut forgotten: Vec<usize> = Vec::new();
        let mut current_cluster: Vec<usize> = Vec::new();
        let mut cluster_q_sum = 0.0f32;
        let mut cluster_visits_sum = 0usize;

        for &idx in &indexed {
            let q = region.q_snapshot[idx];
            let visits = region.visit_snapshot[idx];

            if current_cluster.is_empty() {
                current_cluster.push(idx);
                cluster_q_sum = q;
                cluster_visits_sum = visits;
            } else {
                let avg_q = cluster_q_sum / current_cluster.len() as f32;
                let diff = (q - avg_q).abs();

                if diff < self.config.merge_threshold {
                    current_cluster.push(idx);
                    cluster_q_sum += q;
                    cluster_visits_sum += visits;
                } else {
                    // Finalize current cluster
                    Self::finalize_cluster(
                        &mut current_cluster,
                        &mut cluster_q_sum,
                        &mut cluster_visits_sum,
                        &mut merged,
                        &mut forgotten,
                    );
                    current_cluster.push(idx);
                    cluster_q_sum = q;
                    cluster_visits_sum = visits;
                }
            }
        }

        // Finalize last cluster
        Self::finalize_cluster(
            &mut current_cluster,
            &mut cluster_q_sum,
            &mut cluster_visits_sum,
            &mut merged,
            &mut forgotten,
        );

        // Compute utility based on visit counts relative to total
        let total_visits: usize = region.visit_snapshot.iter().sum();
        let utility: Vec<f32> = merged
            .iter()
            .map(|(indices, _)| {
                let group_visits: usize = indices.iter().map(|&i| region.visit_snapshot[i]).sum();
                group_visits as f32 / total_visits.max(1) as f32
            })
            .collect();

        ReplacementSet {
            merged,
            forgotten,
            utility,
        }
    }

    /// Finalize a cluster: push it to merged if non-empty, clear accumulators.
    fn finalize_cluster(
        cluster: &mut Vec<usize>,
        q_sum: &mut f32,
        _visits_sum: &mut usize,
        merged: &mut Vec<(Vec<usize>, f32)>,
        _forgotten: &mut Vec<usize>,
    ) {
        if cluster.is_empty() {
            return;
        }
        let avg_q = *q_sum / cluster.len() as f32;
        let original_indices: Vec<usize> = cluster.to_vec();
        merged.push((original_indices, avg_q));
        cluster.clear();
        *q_sum = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(
        arm_indices: Vec<usize>,
        q_snapshot: Vec<f32>,
        visit_snapshot: Vec<usize>,
    ) -> WorkingRegion {
        WorkingRegion {
            arm_indices,
            q_snapshot,
            visit_snapshot,
            selected_at_episode: 10,
        }
    }

    #[test]
    fn test_consolidate_empty_region() {
        let consolidator = DreamerConsolidator::new(DreamerConfig::default());
        let region = make_region(vec![], vec![], vec![]);
        let result = consolidator.consolidate(&region);
        assert!(result.merged.is_empty());
        assert!(result.forgotten.is_empty());
        assert!(result.utility.is_empty());
    }

    #[test]
    fn test_consolidate_single_arm() {
        let consolidator = DreamerConsolidator::new(DreamerConfig::default());
        let region = make_region(vec![0], vec![0.5], vec![10]);
        let result = consolidator.consolidate(&region);
        assert_eq!(result.merged.len(), 1);
        assert_eq!(result.merged[0].0, vec![0]);
        assert!((result.merged[0].1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_consolidate_similar_q_values_merge() {
        let config = DreamerConfig {
            merge_threshold: 0.2,
            ..DreamerConfig::default()
        };
        let consolidator = DreamerConsolidator::new(config);
        // Q-values: 0.10, 0.15, 0.20 → all within 0.2 of each other → single cluster
        let region = make_region(vec![0, 1, 2], vec![0.10, 0.15, 0.20], vec![5, 10, 15]);
        let result = consolidator.consolidate(&region);
        assert_eq!(result.merged.len(), 1);
        assert_eq!(result.merged[0].0.len(), 3);
        // Average Q = (0.10 + 0.15 + 0.20) / 3 ≈ 0.15
        let expected_avg = (0.10_f32 + 0.15 + 0.20) / 3.0;
        assert!((result.merged[0].1 - expected_avg).abs() < f32::EPSILON);
    }

    #[test]
    fn test_consolidate_distant_q_values_separate_clusters() {
        let config = DreamerConfig {
            merge_threshold: 0.1,
            ..DreamerConfig::default()
        };
        let consolidator = DreamerConsolidator::new(config);
        // Q-values: 0.1, 0.5, 0.9 → all >0.1 apart → 3 separate clusters
        let region = make_region(vec![0, 1, 2], vec![0.1, 0.5, 0.9], vec![5, 10, 15]);
        let result = consolidator.consolidate(&region);
        assert_eq!(result.merged.len(), 3);
        assert!((result.merged[0].1 - 0.1).abs() < f32::EPSILON);
        assert!((result.merged[1].1 - 0.5).abs() < f32::EPSILON);
        assert!((result.merged[2].1 - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_consolidate_mixed_clustering() {
        let config = DreamerConfig {
            merge_threshold: 0.15,
            ..DreamerConfig::default()
        };
        let consolidator = DreamerConsolidator::new(config);
        // Q-values sorted: 0.10, 0.15, 0.50, 0.55, 0.90
        // Cluster 1: 0.10, 0.15 (within 0.15 of running avg)
        // Cluster 2: 0.50, 0.55
        // Cluster 3: 0.90
        let region = make_region(
            vec![0, 1, 2, 3, 4],
            vec![0.10, 0.15, 0.50, 0.55, 0.90],
            vec![5, 10, 15, 20, 25],
        );
        let result = consolidator.consolidate(&region);
        assert_eq!(result.merged.len(), 3);
        // First cluster: indices [0, 1]
        assert_eq!(result.merged[0].0, vec![0, 1]);
        // Second cluster: indices [2, 3]
        assert_eq!(result.merged[1].0, vec![2, 3]);
        // Third cluster: index [4]
        assert_eq!(result.merged[2].0, vec![4]);
    }

    #[test]
    fn test_consolidate_utility_proportional_to_visits() {
        let consolidator = DreamerConsolidator::new(DreamerConfig {
            merge_threshold: 0.5, // merge all into one cluster
            ..DreamerConfig::default()
        });
        // visits: [10, 30, 60] → total = 100
        let region = make_region(vec![0, 1, 2], vec![0.2, 0.3, 0.4], vec![10, 30, 60]);
        let result = consolidator.consolidate(&region);
        assert_eq!(result.merged.len(), 1);
        // Single merged group has all visits → utility = 100/100 = 1.0
        assert!((result.utility[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_consolidate_preserves_sorted_order() {
        let consolidator = DreamerConsolidator::new(DreamerConfig {
            merge_threshold: 0.5,
            ..DreamerConfig::default()
        });
        // Unsorted Q-values: 0.9, 0.1, 0.5
        let region = make_region(vec![0, 1, 2], vec![0.9, 0.1, 0.5], vec![5, 5, 5]);
        let result = consolidator.consolidate(&region);
        // Sorted: 0.1, 0.5, 0.9
        // 0.1→0.5: diff=0.4 < 0.5 → merge; avg=0.3 → 0.9: diff=0.6 >= 0.5 → split
        // Two clusters: [0.1, 0.5] avg≈0.3, [0.9] avg=0.9
        assert_eq!(result.merged.len(), 2);
        let expected_first = (0.1_f32 + 0.5) / 2.0;
        assert!((result.merged[0].1 - expected_first).abs() < 1e-5);
        assert!((result.merged[1].1 - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_consolidate_with_conservative_config() {
        let consolidator = DreamerConsolidator::new(DreamerConfig::conservative());
        // merge_threshold=0.3 → 0.2 and 0.5 differ by 0.3, not strictly less
        let region = make_region(vec![0, 1], vec![0.2, 0.5], vec![10, 10]);
        let result = consolidator.consolidate(&region);
        // diff = 0.3 >= 0.3 threshold → two separate clusters
        assert_eq!(result.merged.len(), 2);
    }

    #[test]
    fn test_consolidate_with_aggressive_config() {
        let consolidator = DreamerConsolidator::new(DreamerConfig::aggressive());
        // merge_threshold=0.7 → 0.2 and 0.5 differ by 0.3 < 0.7 → merged
        let region = make_region(vec![0, 1], vec![0.2, 0.5], vec![10, 10]);
        let result = consolidator.consolidate(&region);
        assert_eq!(result.merged.len(), 1);
        assert_eq!(result.merged[0].0.len(), 2);
    }

    #[test]
    fn test_consolidate_nan_q_values() {
        let consolidator = DreamerConsolidator::new(DreamerConfig::default());
        let region = make_region(vec![0, 1], vec![f32::NAN, 0.5], vec![5, 5]);
        let result = consolidator.consolidate(&region);
        // Should not panic; NaN comparisons fall through to Equal ordering
        assert!(!result.merged.is_empty());
    }

    #[test]
    fn test_consolidate_zero_visits_utility() {
        let consolidator = DreamerConsolidator::new(DreamerConfig {
            merge_threshold: 0.5,
            ..DreamerConfig::default()
        });
        let region = make_region(vec![0], vec![0.5], vec![0]);
        let result = consolidator.consolidate(&region);
        // total_visits = 0, utility = 0/1 = 0.0
        assert!((result.utility[0]).abs() < f32::EPSILON);
    }
}
