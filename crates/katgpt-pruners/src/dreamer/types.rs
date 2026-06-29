//! Core types for the Auto-Dreamer offline memory consolidation module.

/// Configuration for the Dreamer consolidation scheduler.
#[derive(Debug, Clone)]
pub struct DreamerConfig {
    /// Episodes between consolidation events (paper: k=5-10).
    pub cadence: usize,
    /// Fraction of bank to include in working region.
    pub region_fraction: f32,
    /// Similarity threshold for merging arms during consolidation.
    pub merge_threshold: f32,
    /// Decay factor per consolidation event (0.0 = full decay, 1.0 = no decay).
    pub decay_factor: f32,
    /// Counterfactual dropout fraction (paper: ρ=0.25-0.5).
    pub dropout_fraction: f32,
    /// Number of Monte Carlo samples for counterfactual estimation.
    pub mc_samples: usize,
    /// Minimum visits before an arm is eligible for consolidation.
    pub min_visits: usize,
}

impl Default for DreamerConfig {
    fn default() -> Self {
        Self {
            cadence: 10,
            region_fraction: 0.3,
            merge_threshold: 0.5,
            decay_factor: 0.9,
            dropout_fraction: 0.25,
            mc_samples: 1,
            min_visits: 3,
        }
    }
}

impl DreamerConfig {
    /// Conservative consolidation: less aggressive merging.
    pub fn conservative() -> Self {
        Self {
            cadence: 20,
            region_fraction: 0.2,
            merge_threshold: 0.3,
            decay_factor: 0.95,
            dropout_fraction: 0.1,
            mc_samples: 1,
            min_visits: 5,
        }
    }

    /// Aggressive consolidation: more merging, more pruning.
    pub fn aggressive() -> Self {
        Self {
            cadence: 5,
            region_fraction: 0.5,
            merge_threshold: 0.7,
            decay_factor: 0.8,
            dropout_fraction: 0.4,
            mc_samples: 3,
            min_visits: 2,
        }
    }
}

/// A working region selected from the memory bank for consolidation.
#[derive(Debug, Clone)]
pub struct WorkingRegion {
    /// Indices of arms in the working region.
    pub arm_indices: Vec<usize>,
    /// Q-values at time of selection (read-only snapshot).
    pub q_snapshot: Vec<f32>,
    /// Visit counts at time of selection.
    pub visit_snapshot: Vec<usize>,
    /// Timestamp of selection.
    pub selected_at_episode: usize,
}

/// A compact replacement set synthesized from a working region.
#[derive(Debug, Clone)]
pub struct ReplacementSet {
    /// Merged arms: (original_indices, new Q-value).
    pub merged: Vec<(Vec<usize>, f32)>,
    /// Arms to forget (omitted from replacement).
    pub forgotten: Vec<usize>,
    /// Counterfactual utility scores per merged group.
    pub utility: Vec<f32>,
}

/// Policy for memory decay during consolidation.
#[derive(Debug, Clone, Copy)]
pub enum DecayPolicy {
    /// No decay (baseline).
    None,
    /// Exponential: q *= decay_factor each consolidation.
    Exponential { factor: f32 },
    /// Access-based: decay proportional to episodes since last access.
    AccessBased { half_life: usize },
}

impl Default for DecayPolicy {
    fn default() -> Self {
        Self::Exponential { factor: 0.9 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let config = DreamerConfig::default();
        assert_eq!(config.cadence, 10);
        assert!((config.region_fraction - 0.3).abs() < f32::EPSILON);
        assert!((config.merge_threshold - 0.5).abs() < f32::EPSILON);
        assert!((config.decay_factor - 0.9).abs() < f32::EPSILON);
        assert!((config.dropout_fraction - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.mc_samples, 1);
        assert_eq!(config.min_visits, 3);
    }

    #[test]
    fn test_conservative_config() {
        let config = DreamerConfig::conservative();
        assert_eq!(config.cadence, 20);
        assert!((config.region_fraction - 0.2).abs() < f32::EPSILON);
        assert!((config.merge_threshold - 0.3).abs() < f32::EPSILON);
        assert!((config.decay_factor - 0.95).abs() < f32::EPSILON);
        assert!((config.dropout_fraction - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.mc_samples, 1);
        assert_eq!(config.min_visits, 5);
    }

    #[test]
    fn test_aggressive_config() {
        let config = DreamerConfig::aggressive();
        assert_eq!(config.cadence, 5);
        assert!((config.region_fraction - 0.5).abs() < f32::EPSILON);
        assert!((config.merge_threshold - 0.7).abs() < f32::EPSILON);
        assert!((config.decay_factor - 0.8).abs() < f32::EPSILON);
        assert!((config.dropout_fraction - 0.4).abs() < f32::EPSILON);
        assert_eq!(config.mc_samples, 3);
        assert_eq!(config.min_visits, 2);
    }

    #[test]
    fn test_working_region_fields() {
        let region = WorkingRegion {
            arm_indices: vec![0, 2, 4],
            q_snapshot: vec![0.1, 0.5, 0.9],
            visit_snapshot: vec![10, 20, 5],
            selected_at_episode: 42,
        };
        assert_eq!(region.arm_indices.len(), 3);
        assert_eq!(region.q_snapshot.len(), 3);
        assert_eq!(region.visit_snapshot.len(), 3);
        assert_eq!(region.selected_at_episode, 42);
    }

    #[test]
    fn test_replacement_set_empty() {
        let replacement = ReplacementSet {
            merged: Vec::new(),
            forgotten: Vec::new(),
            utility: Vec::new(),
        };
        assert!(replacement.merged.is_empty());
        assert!(replacement.forgotten.is_empty());
        assert!(replacement.utility.is_empty());
    }

    #[test]
    fn test_replacement_set_with_merges() {
        let replacement = ReplacementSet {
            merged: vec![(vec![0, 1], 0.3), (vec![2], 0.8)],
            forgotten: vec![3],
            utility: vec![0.6, 0.4],
        };
        assert_eq!(replacement.merged.len(), 2);
        assert_eq!(replacement.forgotten, vec![3]);
        assert_eq!(replacement.utility.len(), 2);
    }

    #[test]
    fn test_decay_policy_default() {
        let policy = DecayPolicy::default();
        match policy {
            DecayPolicy::Exponential { factor } => assert!((factor - 0.9).abs() < f32::EPSILON),
            _ => panic!("Expected Exponential decay policy"),
        }
    }

    #[test]
    fn test_decay_policy_variants() {
        let none = DecayPolicy::None;
        let exp = DecayPolicy::Exponential { factor: 0.85 };
        let access = DecayPolicy::AccessBased { half_life: 100 };

        match none {
            DecayPolicy::None => {}
            _ => panic!("Expected None"),
        }
        match exp {
            DecayPolicy::Exponential { factor } => assert!((factor - 0.85).abs() < f32::EPSILON),
            _ => panic!("Expected Exponential"),
        }
        match access {
            DecayPolicy::AccessBased { half_life } => assert_eq!(half_life, 100),
            _ => panic!("Expected AccessBased"),
        }
    }

    #[test]
    fn test_config_clones_independently() {
        let a = DreamerConfig::default();
        let mut b = a.clone();
        b.cadence = 99;
        assert_eq!(a.cadence, 10);
        assert_eq!(b.cadence, 99);
    }
}
