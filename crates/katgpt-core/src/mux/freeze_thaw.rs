//! `MuxTarget` + `MuxPatternStore` — persistent superposition patterns.
//!
//! Freeze stores successful superposition patterns for later thaw (reuse)
//! when similar logit distributions are encountered.

use std::collections::HashMap;

/// A frozen superposition pattern: token IDs and their weights.
#[derive(Debug, Clone)]
pub struct MuxTarget {
    /// Depth at which this pattern was recorded.
    pub depth: usize,
    /// Token IDs in the superposition.
    pub tokens: Vec<u32>,
    /// Weights for each token.
    pub weights: Vec<f32>,
}

impl MuxTarget {
    pub fn new(tokens: Vec<u32>, weights: Vec<f32>, depth: usize) -> Self {
        Self {
            depth,
            tokens,
            weights,
        }
    }
}

/// Store for frozen superposition patterns, keyed by a hash of the
/// logit distribution shape.
#[derive(Debug, Clone, Default)]
pub struct MuxPatternStore {
    /// Cached total pattern count — incremented on freeze, avoids O(n) scan on read.
    total_patterns: usize,
    patterns: HashMap<u64, Vec<MuxTarget>>,
}

impl MuxPatternStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a store with pre-allocated capacity for `n` keys.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            total_patterns: 0,
            patterns: HashMap::with_capacity(n),
        }
    }

    /// Freeze a pattern: store it under the given key.
    #[inline]
    pub fn freeze(&mut self, key: u64, target: MuxTarget) {
        self.patterns.entry(key).or_default().push(target);
        self.total_patterns += 1;
    }

    /// Thaw patterns: retrieve all patterns for a given key.
    #[inline]
    pub fn thaw(&self, key: u64) -> &[MuxTarget] {
        self.patterns.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Number of distinct keys stored.
    pub fn key_count(&self) -> usize {
        self.patterns.len()
    }

    /// Total number of patterns across all keys.
    /// O(1) — maintained incrementally on freeze.
    pub fn pattern_count(&self) -> usize {
        self.total_patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freeze_and_thaw() {
        let mut store = MuxPatternStore::new();
        let target = MuxTarget::new(vec![1, 2, 3], vec![0.5, 0.3, 0.2], 0);
        store.freeze(42, target);

        let thawed = store.thaw(42);
        assert_eq!(thawed.len(), 1);
        assert_eq!(thawed[0].tokens, vec![1, 2, 3]);
    }

    #[test]
    fn thaw_missing_key_returns_empty() {
        let store = MuxPatternStore::new();
        assert!(store.thaw(999).is_empty());
    }

    #[test]
    fn multiple_patterns_per_key() {
        let mut store = MuxPatternStore::new();
        store.freeze(1, MuxTarget::new(vec![1], vec![1.0], 0));
        store.freeze(1, MuxTarget::new(vec![2], vec![0.5], 1));
        store.freeze(2, MuxTarget::new(vec![3], vec![0.8], 0));

        assert_eq!(store.key_count(), 2);
        assert_eq!(store.pattern_count(), 3);
        assert_eq!(store.thaw(1).len(), 2);
    }
}
