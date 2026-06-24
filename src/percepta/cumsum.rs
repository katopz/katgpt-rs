//! Cumulative sum via uniform attention (fetch_sum equivalent).
//!
//! Uses `HardAttentionHead` with AVERAGE tie-breaking and uniform keys
//! to compute exact running sums of inserted values.
//!
//! # How it works
//!
//! 1. Each inserted value is stored with a **uniform key** (constant kx, ky=0),
//!    so all keys produce the same dot product score.
//! 2. Querying with `q = [0.0, 0.0]` routes to the global metadata aggregator.
//! 3. With AVERAGE tie-breaking, the result is the mean of all values.
//! 4. Multiplying by count gives the exact cumulative sum.

use super::hull::HardAttentionHead;
use super::types::TieBreak;

/// Cumulative sum tracker using uniform attention.
///
/// Each inserted value is stored with a constant (uniform) key,
/// so all keys produce the same dot product score. With AVERAGE
/// tie-breaking, the attention head returns the mean of all values.
/// Multiplying by count gives the exact cumulative sum.
pub struct CumSum {
    head: HardAttentionHead,
    count: usize,
}

impl Default for CumSum {
    fn default() -> Self {
        Self::new()
    }
}

impl CumSum {
    /// Create a new cumulative sum tracker.
    pub fn new() -> Self {
        Self {
            head: HardAttentionHead::new(),
            count: 0,
        }
    }

    /// Insert a value with a uniform key.
    ///
    /// Uses `kx = 1.0, ky = 0.0` so all entries share the same key,
    /// making them collinear (identical score). The `val` stores
    /// the actual value at index 0 and position at index 1.
    pub fn insert(&mut self, value: f64, position: f64, seq: i32) {
        // Uniform key: all keys are the same, so all tie
        let key = [1.0, 0.0];
        let val = [value, position];
        self.head.insert(key, val, seq);
        self.count += 1;
    }

    /// Query the cumulative sum of all inserted values.
    ///
    /// With uniform keys and AVERAGE tie-breaking, the global
    /// meta returns the mean. Multiplying by count yields the sum.
    pub fn query(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        // qx=0, qy=0 routes to global meta; Average tie-break gives mean
        let query = [0.0, 0.0];
        match self.head.query(query, TieBreak::Average) {
            Some(val) => val[0] * self.count as f64,
            None => 0.0,
        }
    }

    /// Number of inserted values.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether no values have been inserted.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reset the cumulative sum tracker.
    pub fn clear(&mut self) {
        self.head.clear();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::CumSum;

    #[test]
    fn test_empty_query_returns_zero() {
        let cs = CumSum::new();
        assert!(cs.is_empty());
        assert_eq!(cs.len(), 0);
        assert_eq!(cs.query(), 0.0);
    }

    #[test]
    fn test_single_element() {
        let mut cs = CumSum::new();
        cs.insert(42.0, 1.0, 1);
        assert!(!cs.is_empty());
        assert_eq!(cs.len(), 1);
        let sum = cs.query();
        assert!((sum - 42.0).abs() < 1e-9, "expected 42.0, got {sum}");
    }

    #[test]
    fn test_counter_cumsum() {
        // Insert 1, 2, 3, 4, 5 → cumulative sum at each step = n*(n+1)/2
        let mut cs = CumSum::new();
        let mut expected = 0.0_f64;

        for i in 1..=5 {
            let val = i as f64;
            cs.insert(val, i as f64, i as i32);
            expected += val;
            let got = cs.query();
            assert!(
                (got - expected).abs() < 1e-9,
                "step {i}: expected sum {expected}, got {got}"
            );
        }

        // Final sum = 1+2+3+4+5 = 15
        assert!((cs.query() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_fibonacci_cumsum() {
        // Fibonacci: 1, 1, 2, 3, 5, 8, 13
        let fibs: [f64; 7] = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let mut cs = CumSum::new();
        let mut expected = 0.0_f64;

        for (i, &val) in fibs.iter().enumerate() {
            cs.insert(val, i as f64, i as i32);
            expected += val;
            let got = cs.query();
            assert!(
                (got - expected).abs() < 1e-9,
                "fib step {i}: expected sum {expected}, got {got}"
            );
        }

        // 1+1+2+3+5+8+13 = 33
        assert!((cs.query() - 33.0).abs() < 1e-9);
    }

    #[test]
    fn test_clear_resets() {
        let mut cs = CumSum::new();
        cs.insert(10.0, 1.0, 1);
        cs.insert(20.0, 2.0, 2);
        assert_eq!(cs.len(), 2);
        assert!((cs.query() - 30.0).abs() < 1e-9);

        cs.clear();
        assert!(cs.is_empty());
        assert_eq!(cs.query(), 0.0);
    }
}
