//! MuxTarget Freeze/Thaw — Pre-computed multiplexed patterns (Research 158, MUX).
//!
//! Pre-compute MUX-style superposition targets from observed CoT traces and store them
//! as frozen patterns. At inference time, the bandit selects a difficulty tier → maps to
//! a pre-computed target pattern → model's logits are compared to the target.
//!
//! This is modelless self-learning: the system learns which pre-computed targets produce
//! the best results, without any model training.

/// Difficulty tier for query classification.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifficultyTier {
    Easy,
    Medium,
    Hard,
}

/// A pre-computed MUX superposition target pattern.
#[derive(Clone, Debug)]
pub struct MuxTarget {
    /// Span width K (number of superposed tokens).
    pub span_k: usize,
    /// Geometric decay ratio.
    pub decay: f32,
    /// Difficulty tier this pattern is optimized for.
    pub tier: DifficultyTier,
    /// KL-divergence reward from last evaluation.
    pub kl_reward: f32,
}

impl MuxTarget {
    /// Returns whether two targets share the same structural parameters
    /// (span_k, decay, tier), ignoring kl_reward.
    fn matches_structure(&self, other: &MuxTarget) -> bool {
        self.span_k == other.span_k && self.decay == other.decay && self.tier == other.tier
    }
}

/// Persistent store for pre-computed MUX superposition patterns.
///
/// Patterns are keyed by (query_type, difficulty_tier). Multiple patterns
/// per key are ranked by kl_reward — thaw returns the best one.
pub struct MuxPatternStore {
    patterns: std::collections::HashMap<String, Vec<MuxTarget>>,
}

impl MuxPatternStore {
    pub fn new() -> Self {
        Self {
            patterns: std::collections::HashMap::new(),
        }
    }

    /// Freeze a winning superposition pattern into the store.
    ///
    /// Associates the pattern with a query type. If a pattern with the same
    /// span_k, decay, and tier already exists, updates its kl_reward.
    pub fn freeze(&mut self, query_type: &str, pattern: &MuxTarget) {
        let entries = self.patterns.entry(query_type.to_string()).or_default();
        if let Some(existing) = entries.iter_mut().find(|e| e.matches_structure(pattern)) {
            existing.kl_reward = pattern.kl_reward;
        } else {
            entries.push(pattern.clone());
        }
    }

    /// Thaw the best pre-computed pattern for a query type and tier.
    ///
    /// Returns the pattern with the highest kl_reward matching the given tier.
    /// Returns None if no patterns exist for this query type / tier combination.
    pub fn thaw(&self, query_type: &str, tier: DifficultyTier) -> Option<&MuxTarget> {
        self.patterns
            .get(query_type)?
            .iter()
            .filter(|p| p.tier == tier)
            .max_by(|a, b| {
                a.kl_reward
                    .partial_cmp(&b.kl_reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Thaw the best pattern regardless of tier.
    pub fn thaw_best(&self, query_type: &str) -> Option<&MuxTarget> {
        self.patterns.get(query_type)?.iter().max_by(|a, b| {
            a.kl_reward
                .partial_cmp(&b.kl_reward)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Number of stored patterns.
    pub fn len(&self) -> usize {
        self.patterns.values().map(|v| v.len()).sum()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.values().all(|v| v.is_empty())
    }

    /// Clear all patterns.
    pub fn clear(&mut self) {
        self.patterns.clear();
    }

    /// List all query types in the store.
    pub fn query_types(&self) -> Vec<&String> {
        self.patterns.keys().collect()
    }
}

impl Default for MuxPatternStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn easy_target(span_k: usize, decay: f32, reward: f32) -> MuxTarget {
        MuxTarget {
            span_k,
            decay,
            tier: DifficultyTier::Easy,
            kl_reward: reward,
        }
    }

    fn medium_target(span_k: usize, decay: f32, reward: f32) -> MuxTarget {
        MuxTarget {
            span_k,
            decay,
            tier: DifficultyTier::Medium,
            kl_reward: reward,
        }
    }

    fn hard_target(span_k: usize, decay: f32, reward: f32) -> MuxTarget {
        MuxTarget {
            span_k,
            decay,
            tier: DifficultyTier::Hard,
            kl_reward: reward,
        }
    }

    #[test]
    fn test_freeze_thaw_roundtrip() {
        let mut store = MuxPatternStore::new();
        let target = easy_target(4, 0.85, 1.2);
        store.freeze("math", &target);

        let thawed = store.thaw("math", DifficultyTier::Easy).unwrap();
        assert_eq!(thawed.span_k, 4);
        assert!((thawed.decay - 0.85).abs() < f32::EPSILON);
        assert!((thawed.kl_reward - 1.2).abs() < f32::EPSILON);
        assert_eq!(thawed.tier, DifficultyTier::Easy);
    }

    #[test]
    fn test_freeze_thaw_best_by_tier() {
        let mut store = MuxPatternStore::new();
        store.freeze("math", &easy_target(2, 0.9, 0.5));
        store.freeze("math", &easy_target(4, 0.8, 1.5));
        store.freeze("math", &easy_target(8, 0.7, 0.8));
        store.freeze("math", &hard_target(16, 0.6, 2.0));

        let best_easy = store.thaw("math", DifficultyTier::Easy).unwrap();
        assert_eq!(best_easy.span_k, 4);
        assert!((best_easy.kl_reward - 1.5).abs() < f32::EPSILON);

        let best_hard = store.thaw("math", DifficultyTier::Hard).unwrap();
        assert_eq!(best_hard.span_k, 16);
    }

    #[test]
    fn test_freeze_updates_reward() {
        let mut store = MuxPatternStore::new();
        store.freeze("math", &easy_target(4, 0.85, 1.0));
        store.freeze("math", &easy_target(4, 0.85, 2.5));

        let thawed = store.thaw("math", DifficultyTier::Easy).unwrap();
        assert!((thawed.kl_reward - 2.5).abs() < f32::EPSILON);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_thaw_nonexistent_returns_none() {
        let store = MuxPatternStore::new();
        assert!(store.thaw("unknown", DifficultyTier::Easy).is_none());
        assert!(store.thaw_best("unknown").is_none());

        let mut store = MuxPatternStore::new();
        store.freeze("math", &easy_target(4, 0.85, 1.0));
        assert!(store.thaw("math", DifficultyTier::Hard).is_none());
    }

    #[test]
    fn test_thaw_best_ignores_tier() {
        let mut store = MuxPatternStore::new();
        store.freeze("math", &easy_target(4, 0.85, 1.0));
        store.freeze("math", &medium_target(8, 0.7, 3.0));
        store.freeze("math", &hard_target(16, 0.6, 2.0));

        let best = store.thaw_best("math").unwrap();
        assert_eq!(best.span_k, 8);
        assert_eq!(best.tier, DifficultyTier::Medium);
        assert!((best.kl_reward - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_store_empty_and_len() {
        let store = MuxPatternStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        let mut store = MuxPatternStore::new();
        store.freeze("a", &easy_target(4, 0.85, 1.0));
        store.freeze("a", &hard_target(16, 0.6, 2.0));
        store.freeze("b", &medium_target(8, 0.7, 1.5));
        assert_eq!(store.len(), 3);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_clear_removes_all() {
        let mut store = MuxPatternStore::new();
        store.freeze("math", &easy_target(4, 0.85, 1.0));
        store.freeze("code", &hard_target(16, 0.6, 2.0));
        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.query_types().is_empty());
    }

    #[test]
    fn test_query_types() {
        let mut store = MuxPatternStore::new();
        store.freeze("math", &easy_target(4, 0.85, 1.0));
        store.freeze("code", &hard_target(16, 0.6, 2.0));
        store.freeze("reason", &medium_target(8, 0.7, 1.5));

        let mut types = store.query_types();
        types.sort();
        assert_eq!(types, vec!["code", "math", "reason"]);
    }
}
