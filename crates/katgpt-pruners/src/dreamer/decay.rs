//! Memory decay — applies forgetting to arms NOT in the working region.
//!
//! During consolidation, arms outside the working region undergo decay.
//! This implements the "forgetting" side of the dreamer cycle: knowledge
//! not recently accessed or consolidated gradually fades.

use std::collections::HashSet;

use super::types::{DecayPolicy, WorkingRegion};

/// Memory decay — applies forgetting to arms NOT in the working region.
pub struct MemoryDecay {
    pub policy: DecayPolicy,
}

impl MemoryDecay {
    pub fn new(policy: DecayPolicy) -> Self {
        Self { policy }
    }

    /// Apply decay to Q-values. Arms in the working region are exempt.
    ///
    /// Returns vec of `(index, decayed_q)` for arms not in region.
    pub fn apply(
        &self,
        q_values: &[f32],
        last_access: &[usize],
        region: &WorkingRegion,
        current_episode: usize,
    ) -> Vec<(usize, f32)> {
        let region_set: HashSet<usize> = region.arm_indices.iter().copied().collect();

        match self.policy {
            DecayPolicy::None => q_values
                .iter()
                .enumerate()
                .filter(|(i, _)| !region_set.contains(i))
                .map(|(i, &q)| (i, q))
                .collect(),
            DecayPolicy::Exponential { factor } => q_values
                .iter()
                .enumerate()
                .filter(|(i, _)| !region_set.contains(i))
                .map(|(i, &q)| (i, q * factor))
                .collect(),
            DecayPolicy::AccessBased { half_life } => q_values
                .iter()
                .enumerate()
                .filter(|(i, _)| !region_set.contains(i))
                .map(|(i, &q)| {
                    let age =
                        current_episode.saturating_sub(last_access.get(i).copied().unwrap_or(0));
                    let decay = 0.5f32.powi(age as i32 / half_life.max(1) as i32);
                    (i, q * decay)
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(arm_indices: Vec<usize>) -> WorkingRegion {
        WorkingRegion {
            arm_indices,
            q_snapshot: vec![0.5; 10],
            visit_snapshot: vec![5; 10],
            selected_at_episode: 10,
        }
    }

    #[test]
    fn test_no_decay_preserves_values() {
        let decay = MemoryDecay::new(DecayPolicy::None);
        let q_values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let last_access = vec![5, 5, 5, 5, 5];
        let region = make_region(vec![2]); // exempt index 2
        let result = decay.apply(&q_values, &last_access, &region, 10);
        assert_eq!(result.len(), 4);
        for (i, q) in &result {
            assert!((q - q_values[*i]).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_exponential_decay_reduces_values() {
        let decay = MemoryDecay::new(DecayPolicy::Exponential { factor: 0.9 });
        let q_values = vec![1.0, 1.0, 1.0];
        let last_access = vec![0, 0, 0];
        let region = make_region(vec![]); // no exempt arms
        let result = decay.apply(&q_values, &last_access, &region, 10);
        assert_eq!(result.len(), 3);
        for (_, q) in &result {
            assert!((*q - 0.9).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_exponential_decay_factor_zero() {
        let decay = MemoryDecay::new(DecayPolicy::Exponential { factor: 0.0 });
        let q_values = vec![0.9, 0.5, 0.1];
        let last_access = vec![0; 3];
        let region = make_region(vec![]);
        let result = decay.apply(&q_values, &last_access, &region, 10);
        for (_, q) in &result {
            assert!(q.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_exponential_decay_factor_one() {
        let decay = MemoryDecay::new(DecayPolicy::Exponential { factor: 1.0 });
        let q_values = vec![0.42, 0.77];
        let last_access = vec![0; 2];
        let region = make_region(vec![]);
        let result = decay.apply(&q_values, &last_access, &region, 10);
        assert!((result[0].1 - 0.42).abs() < f32::EPSILON);
        assert!((result[1].1 - 0.77).abs() < f32::EPSILON);
    }

    #[test]
    fn test_access_based_decay_recent_arms_preserved() {
        let decay = MemoryDecay::new(DecayPolicy::AccessBased { half_life: 10 });
        let q_values = vec![1.0, 1.0];
        let last_access = vec![10, 0]; // arm 0 accessed this episode, arm 1 long ago
        let region = make_region(vec![]);
        let result = decay.apply(&q_values, &last_access, &region, 10);
        // arm 0: age=0 → decay=0.5^0=1.0 → q=1.0
        assert!((result[0].1 - 1.0).abs() < f32::EPSILON);
        // arm 1: age=10, half_life=10 → decay=0.5^(10/10)=0.5 → q=0.5
        assert!((result[1].1 - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_access_based_decay_old_arms_fade() {
        let decay = MemoryDecay::new(DecayPolicy::AccessBased { half_life: 5 });
        let q_values = vec![1.0];
        let last_access = vec![0]; // 20 episodes ago
        let region = make_region(vec![]);
        let result = decay.apply(&q_values, &last_access, &region, 20);
        // age=20, half_life=5 → 0.5^(20/5) = 0.5^4 = 0.0625
        assert!((result[0].1 - 0.0625).abs() < f32::EPSILON);
    }

    #[test]
    fn test_region_arms_exempt_from_decay() {
        let decay = MemoryDecay::new(DecayPolicy::Exponential { factor: 0.0 });
        let q_values = vec![1.0, 1.0, 1.0];
        let last_access = vec![0; 3];
        let region = make_region(vec![1]); // exempt arm 1
        let result = decay.apply(&q_values, &last_access, &region, 10);
        // Only arms 0 and 2 decayed (to 0.0); arm 1 exempt
        assert_eq!(result.len(), 2);
        let decayed_indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        assert!(!decayed_indices.contains(&1));
        assert!(decayed_indices.contains(&0));
        assert!(decayed_indices.contains(&2));
    }

    #[test]
    fn test_empty_q_values() {
        let decay = MemoryDecay::new(DecayPolicy::Exponential { factor: 0.5 });
        let region = make_region(vec![]);
        let result = decay.apply(&[], &[], &region, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_all_arms_in_region() {
        let decay = MemoryDecay::new(DecayPolicy::Exponential { factor: 0.0 });
        let q_values = vec![1.0, 1.0, 1.0];
        let last_access = vec![0; 3];
        let region = make_region(vec![0, 1, 2]); // all exempt
        let result = decay.apply(&q_values, &last_access, &region, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_access_based_decay_zero_half_life_safe() {
        let decay = MemoryDecay::new(DecayPolicy::AccessBased { half_life: 0 });
        let q_values = vec![1.0];
        let last_access = vec![0];
        let region = make_region(vec![]);
        // half_life=0 → .max(1) = 1, age=10 → 0.5^10
        let result = decay.apply(&q_values, &last_access, &region, 10);
        assert_eq!(result.len(), 1);
        // Should not panic; half_life clamped to 1
        assert!(result[0].1 > 0.0 && result[0].1 < 1.0);
    }

    #[test]
    fn test_access_based_missing_last_access_entry() {
        let decay = MemoryDecay::new(DecayPolicy::AccessBased { half_life: 10 });
        let q_values = vec![1.0, 1.0];
        let last_access = vec![5]; // only 1 entry for 2 arms
        let region = make_region(vec![]);
        let result = decay.apply(&q_values, &last_access, &region, 15);
        assert_eq!(result.len(), 2);
        // arm 0: age=15-5=10, decay=0.5^(10/10)=0.5^1=0.5
        assert!((result[0].1 - 0.5).abs() < f32::EPSILON);
        // arm 1: last_access missing → unwrap_or(0) → age=15, decay=0.5^(15/10)=0.5^1=0.5
        assert!((result[1].1 - 0.5).abs() < f32::EPSILON);
    }
}
