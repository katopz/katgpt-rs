//! Standard O(n) softmax attention KV cache.
//!
//! Reference implementation for verifying hard attention results.
//! Uses exact softmax over all stored keys, unlike the CHT which
//! uses hard (argmax) attention.

use super::types::Vec2;

/// Single entry in the standard KV cache.
#[derive(Clone, Debug)]
struct Entry {
    key: Vec2,
    val: [f64; 2],
}

/// O(n) softmax attention KV cache for a single attention head.
///
/// Computes standard softmax attention over all stored keys,
/// producing a weighted average of values.
pub struct StandardCache {
    entries: Vec<Entry>,
}

impl Default for StandardCache {
    fn default() -> Self {
        Self::new()
    }
}

impl StandardCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create an empty cache with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: Vec2, val: [f64; 2]) {
        self.entries.push(Entry { key, val });
    }

    /// Query with softmax attention.
    ///
    /// Returns the softmax-weighted average of values, or `[0.0, 0.0]` if empty.
    ///
    /// Uses online softmax: single pass with running max correction for
    /// numerical stability. Equivalent to the two-pass approach but avoids
    /// iterating twice over the entries.
    pub fn query(&self, query: &Vec2) -> [f64; 2] {
        if self.entries.is_empty() {
            return [0.0, 0.0];
        }

        // Online softmax: maintain running max and correction factor
        let mut max_score = f64::NEG_INFINITY;
        let mut sum_exp = 0.0f64;
        let mut out = [0.0f64; 2];

        for e in &self.entries {
            let s = query.dot(&e.key);
            let new_max = max_score.max(s);
            // Rescale accumulated values when max changes
            let correction = (max_score - new_max).exp();
            sum_exp *= correction;
            out[0] *= correction;
            out[1] *= correction;

            let exp_val = (s - new_max).exp();
            sum_exp += exp_val;
            out[0] += exp_val * e.val[0];
            out[1] += exp_val * e.val[1];
            max_score = new_max;
        }

        if sum_exp == 0.0 {
            return [0.0, 0.0];
        }

        [out[0] / sum_exp, out[1] / sum_exp]
    }

    /// Query with scaled softmax attention (temperature parameter).
    ///
    /// `temperature` > 1.0 makes the distribution sharper (closer to hard attention),
    /// `temperature` < 1.0 makes it smoother.
    ///
    /// Uses online softmax: single pass with running max correction.
    pub fn query_scaled(&self, query: &Vec2, temperature: f64) -> [f64; 2] {
        if self.entries.is_empty() {
            return [0.0, 0.0];
        }

        let inv_temp = 1.0 / temperature;

        // Online softmax: maintain running max and correction factor
        let mut max_score = f64::NEG_INFINITY;
        let mut sum_exp = 0.0f64;
        let mut out = [0.0f64; 2];

        for e in &self.entries {
            let s = query.dot(&e.key) * inv_temp;
            let new_max = max_score.max(s);
            // Rescale accumulated values when max changes
            let correction = (max_score - new_max).exp();
            sum_exp *= correction;
            out[0] *= correction;
            out[1] *= correction;

            let exp_val = (s - new_max).exp();
            sum_exp += exp_val;
            out[0] += exp_val * e.val[0];
            out[1] += exp_val * e.val[1];
            max_score = new_max;
        }

        if sum_exp == 0.0 {
            return [0.0, 0.0];
        }

        [out[0] / sum_exp, out[1] / sum_exp]
    }

    /// Query with hard attention (argmax).
    ///
    /// Returns the value with the highest dot product score, or `None` if empty.
    pub fn query_hard(&self, query: &Vec2) -> Option<[f64; 2]> {
        let best = self.entries.iter().max_by(|a, b| {
            let sa = query.dot(&a.key);
            let sb = query.dot(&b.key);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        Some(best.val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compare two [f64; 2] within tolerance.
    fn approx_eq(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol
    }

    #[test]
    fn empty_cache_returns_zeros_for_softmax() {
        let cache = StandardCache::new();
        let q = Vec2::new(1.0, 0.0);
        let result = cache.query(&q);
        assert_eq!(result, [0.0, 0.0]);
    }

    #[test]
    fn empty_cache_returns_none_for_hard() {
        let cache = StandardCache::new();
        let q = Vec2::new(1.0, 0.0);
        assert!(cache.query_hard(&q).is_none());
    }

    #[test]
    fn single_element_softmax_returns_value() {
        let mut cache = StandardCache::new();
        cache.insert(Vec2::new(1.0, 0.0), [3.0, 4.0]);

        let q = Vec2::new(1.0, 0.0);
        let result = cache.query(&q);
        assert!(approx_eq(result, [3.0, 4.0], 1e-10));
    }

    #[test]
    fn single_element_hard_returns_value() {
        let mut cache = StandardCache::new();
        cache.insert(Vec2::new(1.0, 0.0), [3.0, 4.0]);

        let q = Vec2::new(1.0, 0.0);
        let result = cache.query_hard(&q);
        assert_eq!(result, Some([3.0, 4.0]));
    }

    #[test]
    fn two_elements_softmax_averages() {
        let mut cache = StandardCache::new();

        // Both keys are identical, so softmax weights are equal (0.5 each)
        cache.insert(Vec2::new(1.0, 0.0), [2.0, 0.0]);
        cache.insert(Vec2::new(1.0, 0.0), [4.0, 0.0]);

        let q = Vec2::new(1.0, 0.0);
        let result = cache.query(&q);

        // Both scores equal → exp equal → weights 0.5 each → average
        assert!(approx_eq(result, [3.0, 0.0], 1e-10));
    }

    #[test]
    fn two_elements_hard_picks_max_score() {
        let mut cache = StandardCache::new();

        // key1 · query = 1.0, key2 · query = 2.0
        cache.insert(Vec2::new(1.0, 0.0), [10.0, 0.0]);
        cache.insert(Vec2::new(2.0, 0.0), [20.0, 0.0]);

        let q = Vec2::new(1.0, 0.0);
        let result = cache.query_hard(&q);

        // key2 scores 2.0 > key1 scores 1.0
        assert_eq!(result, Some([20.0, 0.0]));
    }

    #[test]
    fn temperature_scaling_approaches_hard_when_dominant() {
        let mut cache = StandardCache::new();

        // key1 · query = 0.01, key2 · query = 100.0
        cache.insert(Vec2::new(0.01, 0.0), [1.0, 0.0]);
        cache.insert(Vec2::new(100.0, 0.0), [99.0, 0.0]);

        let q = Vec2::new(1.0, 0.0);

        // With very low temperature (high sharpness), softmax → hard
        let result = cache.query_scaled(&q, 0.001);
        assert!(
            approx_eq(result, [99.0, 0.0], 1e-6),
            "Low temperature should approximate hard attention, got {result:?}"
        );

        // With very high temperature (smooth), softmax → uniform average
        // T=1e6 makes scores ~1e-8 and ~1e-4, both negligible → ~equal weights
        let result = cache.query_scaled(&q, 1_000_000.0);
        assert!(
            approx_eq(result, [50.0, 0.0], 0.1),
            "High temperature should approximate uniform average, got {result:?}"
        );
    }

    #[test]
    fn clear_removes_all_entries() {
        let mut cache = StandardCache::new();
        cache.insert(Vec2::new(1.0, 0.0), [1.0, 2.0]);
        cache.insert(Vec2::new(0.0, 1.0), [3.0, 4.0]);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);

        let q = Vec2::new(1.0, 0.0);
        assert_eq!(cache.query(&q), [0.0, 0.0]);
        assert!(cache.query_hard(&q).is_none());
    }

    #[test]
    fn with_capacity_preallocates() {
        let cache = StandardCache::with_capacity(100);
        assert!(cache.is_empty());
        assert!(cache.entries.capacity() >= 100);
    }

    #[test]
    fn orthogonal_keys_isolate_attention() {
        let mut cache = StandardCache::new();

        // key1 is along x-axis, key2 is along y-axis
        cache.insert(Vec2::new(1.0, 0.0), [10.0, 0.0]);
        cache.insert(Vec2::new(0.0, 1.0), [0.0, 20.0]);

        // Query along x-axis: score(key1)=1.0, score(key2)=0.0
        let q = Vec2::new(1.0, 0.0);
        let result = cache.query(&q);

        // exp(1.0) / (exp(1.0) + exp(0.0)) ≈ 0.731 / (0.731 + 1.0)
        // So result[0] ≈ 0.731 * 10.0, result[1] ≈ 0.0 * weight + 20.0 * (1-weight)
        let w1 = 1.0_f64.exp() / (1.0_f64.exp() + 1.0);
        let expected = [w1 * 10.0, (1.0 - w1) * 20.0];
        assert!(
            approx_eq(result, expected, 1e-10),
            "Expected {expected:?}, got {result:?}"
        );
    }
}
