//! Consolidation scheduler — decides when and what to consolidate.
//!
//! The scheduler triggers consolidation at a fixed cadence and selects
//! a working region from the bandit arms based on recent activity.

use super::types::{DreamerConfig, WorkingRegion};

/// Bandit arm metadata needed for region selection.
///
/// Callers provide this data; Dreamer doesn't depend on concrete `BanditPruner` type.
#[derive(Debug, Clone)]
pub struct ArmInfo {
    pub index: usize,
    pub q_value: f32,
    pub visits: usize,
    pub last_write_episode: usize,
    pub last_retrieve_episode: usize,
}

/// Consolidation scheduler — decides when and what to consolidate.
pub struct DreamerScheduler {
    pub config: DreamerConfig,
}

impl DreamerScheduler {
    pub fn new(config: DreamerConfig) -> Self {
        Self { config }
    }

    /// Check if consolidation should trigger at this episode.
    pub fn should_consolidate(&self, episode: usize) -> bool {
        episode > 0 && episode.is_multiple_of(self.config.cadence)
    }

    /// Select working region from bandit arms.
    ///
    /// Region = recently written arms ∪ recently retrieved arms, capped at `region_fraction`.
    pub fn select_region(&self, arms: &[ArmInfo], episode: usize) -> WorkingRegion {
        let cutoff = episode.saturating_sub(self.config.cadence);

        // Collect arms with recent activity and sufficient visits
        let mut active: Vec<&ArmInfo> = arms
            .iter()
            .filter(|a| a.last_write_episode >= cutoff || a.last_retrieve_episode >= cutoff)
            .filter(|a| a.visits >= self.config.min_visits)
            .collect();

        // Sort by recency (most recent first)
        active.sort_by(|a, b| {
            let a_recency = a.last_write_episode.max(a.last_retrieve_episode);
            let b_recency = b.last_write_episode.max(b.last_retrieve_episode);
            b_recency.cmp(&a_recency)
        });

        // Cap at region_fraction of total arms
        let max_arms = ((arms.len() as f32) * self.config.region_fraction).ceil() as usize;
        active.truncate(max_arms.max(1));

        WorkingRegion {
            arm_indices: active.iter().map(|a| a.index).collect(),
            q_snapshot: active.iter().map(|a| a.q_value).collect(),
            visit_snapshot: active.iter().map(|a| a.visits).collect(),
            selected_at_episode: episode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_arms() -> Vec<ArmInfo> {
        vec![
            ArmInfo {
                index: 0,
                q_value: 0.9,
                visits: 20,
                last_write_episode: 8,
                last_retrieve_episode: 9,
            },
            ArmInfo {
                index: 1,
                q_value: 0.1,
                visits: 1,
                last_write_episode: 1,
                last_retrieve_episode: 1,
            },
            ArmInfo {
                index: 2,
                q_value: 0.5,
                visits: 10,
                last_write_episode: 5,
                last_retrieve_episode: 6,
            },
            ArmInfo {
                index: 3,
                q_value: 0.7,
                visits: 15,
                last_write_episode: 3,
                last_retrieve_episode: 4,
            },
            ArmInfo {
                index: 4,
                q_value: 0.3,
                visits: 0,
                last_write_episode: 0,
                last_retrieve_episode: 0,
            },
        ]
    }

    #[test]
    fn test_should_consolidate_at_cadence() {
        let scheduler = DreamerScheduler::new(DreamerConfig::default());
        assert!(scheduler.should_consolidate(10));
        assert!(scheduler.should_consolidate(20));
        assert!(scheduler.should_consolidate(100));
    }

    #[test]
    fn test_should_not_consolidate_between_cadence() {
        let scheduler = DreamerScheduler::new(DreamerConfig::default());
        assert!(!scheduler.should_consolidate(0));
        assert!(!scheduler.should_consolidate(1));
        assert!(!scheduler.should_consolidate(5));
        assert!(!scheduler.should_consolidate(9));
        assert!(!scheduler.should_consolidate(11));
    }

    #[test]
    fn test_should_not_consolidate_at_zero() {
        let scheduler = DreamerScheduler::new(DreamerConfig::default());
        assert!(!scheduler.should_consolidate(0));
    }

    #[test]
    fn test_select_region_filters_by_min_visits() {
        let scheduler = DreamerScheduler::new(DreamerConfig::default());
        let arms = make_arms();
        // min_visits=3 in default config → arms 1 (visits=1) and 4 (visits=0) excluded
        let region = scheduler.select_region(&arms, 10);
        for &idx in &region.arm_indices {
            assert!(arms[idx].visits >= 3);
        }
    }

    #[test]
    fn test_select_region_caps_at_region_fraction() {
        let config = DreamerConfig {
            region_fraction: 0.4,
            ..DreamerConfig::default()
        };
        let scheduler = DreamerScheduler::new(config);
        let arms = make_arms();
        // 5 arms * 0.4 = 2.0 → ceil = 2
        let region = scheduler.select_region(&arms, 10);
        assert!(region.arm_indices.len() <= 2);
    }

    #[test]
    fn test_select_region_preserves_recency_order() {
        let scheduler = DreamerScheduler::new(DreamerConfig::default());
        let arms = make_arms();
        let region = scheduler.select_region(&arms, 10);
        // Most recent arm (index 0, recency=9) should be first
        if !region.arm_indices.is_empty() {
            assert_eq!(region.arm_indices[0], 0);
        }
    }

    #[test]
    fn test_select_region_snapshots_match() {
        let scheduler = DreamerScheduler::new(DreamerConfig::default());
        let arms = make_arms();
        let region = scheduler.select_region(&arms, 10);
        assert_eq!(region.arm_indices.len(), region.q_snapshot.len());
        assert_eq!(region.arm_indices.len(), region.visit_snapshot.len());
        assert_eq!(region.selected_at_episode, 10);
    }

    #[test]
    fn test_select_region_empty_arms() {
        let scheduler = DreamerScheduler::new(DreamerConfig::default());
        let region = scheduler.select_region(&[], 10);
        assert!(region.arm_indices.is_empty());
        assert!(region.q_snapshot.is_empty());
        assert!(region.visit_snapshot.is_empty());
    }

    #[test]
    fn test_select_region_no_recent_activity() {
        let scheduler = DreamerScheduler::new(DreamerConfig {
            cadence: 10,
            min_visits: 1,
            ..DreamerConfig::default()
        });
        let arms = vec![ArmInfo {
            index: 0,
            q_value: 0.5,
            visits: 5,
            last_write_episode: 0,
            last_retrieve_episode: 0,
        }];
        // Episode 10, cutoff=0 → arm with episode 0 is at boundary (>=0)
        let region = scheduler.select_region(&arms, 10);
        assert_eq!(region.arm_indices.len(), 1);
    }

    #[test]
    fn test_select_region_respects_cadence_window() {
        let config = DreamerConfig {
            cadence: 5,
            min_visits: 1,
            ..DreamerConfig::default()
        };
        let scheduler = DreamerScheduler::new(config);
        let arms = vec![
            ArmInfo {
                index: 0,
                q_value: 0.5,
                visits: 5,
                last_write_episode: 8,
                last_retrieve_episode: 0,
            },
            ArmInfo {
                index: 1,
                q_value: 0.5,
                visits: 5,
                last_write_episode: 2,
                last_retrieve_episode: 0,
            },
        ];
        // Episode 10, cadence=5, cutoff=5
        // Arm 0 (write=8 >= 5) included, arm 1 (write=2 < 5) excluded
        let region = scheduler.select_region(&arms, 10);
        assert!(region.arm_indices.contains(&0));
        assert!(!region.arm_indices.contains(&1));
    }

    #[test]
    fn test_custom_cadence() {
        let config = DreamerConfig {
            cadence: 3,
            ..DreamerConfig::default()
        };
        let scheduler = DreamerScheduler::new(config);
        assert!(scheduler.should_consolidate(3));
        assert!(scheduler.should_consolidate(6));
        assert!(scheduler.should_consolidate(9));
        assert!(!scheduler.should_consolidate(4));
    }

    #[test]
    fn test_select_region_at_least_one_arm() {
        let config = DreamerConfig {
            region_fraction: 0.01,
            min_visits: 1,
            cadence: 100,
            ..DreamerConfig::default()
        };
        let scheduler = DreamerScheduler::new(config);
        let arms = vec![ArmInfo {
            index: 0,
            q_value: 0.5,
            visits: 5,
            last_write_episode: 50,
            last_retrieve_episode: 50,
        }];
        let region = scheduler.select_region(&arms, 100);
        // Even with tiny fraction, we get at least 1 arm
        assert_eq!(region.arm_indices.len(), 1);
    }
}
