use std::collections::HashMap;

/// Tracks state visitation distribution during bandit rollouts.
///
/// Reports entropy-based coverage metrics. Zero-cost when disabled
/// (caller simply doesn't call [`observe`](StateVisitationTracker::observe)).
pub struct StateVisitationTracker {
    /// Hash-based state counter (blake3 prefix hash → visit count).
    visits: HashMap<u64, u32>,
    /// Total visits for entropy computation.
    total: u64,
    /// Coverage threshold for exploration boost signal.
    coverage_threshold: f32,
}

impl StateVisitationTracker {
    pub fn new(coverage_threshold: f32) -> Self {
        Self {
            visits: HashMap::new(),
            total: 0,
            coverage_threshold,
        }
    }

    /// Record a visited state (prefix hash). O(1) amortized via HashMap.
    pub fn observe(&mut self, prefix_hash: u64) {
        *self.visits.entry(prefix_hash).or_insert(0) += 1;
        self.total += 1;
    }

    /// Compute visitation entropy H = -Σ p(s) log₂ p(s).
    ///
    /// Higher entropy → more diverse exploration.
    /// Returns 0.0 when no visits or only one unique state.
    pub fn entropy(&self) -> f32 {
        if self.total == 0 || self.visits.len() <= 1 {
            return 0.0;
        }

        let mut h = 0.0f32;
        for &count in self.visits.values() {
            let p = count as f32 / self.total as f32;
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Is coverage above threshold?
    ///
    /// Checks normalized entropy (entropy / log₂(unique_states)) against
    /// `coverage_threshold`. When false → suggest exploration boost.
    pub fn coverage_ok(&self) -> bool {
        let n = self.visits.len();
        if n <= 1 || self.total == 0 {
            return false;
        }
        let max_entropy = (n as f32).log2();
        if max_entropy == 0.0 {
            return false;
        }
        self.entropy() / max_entropy >= self.coverage_threshold
    }

    /// Number of unique states visited.
    pub fn unique_states(&self) -> usize {
        self.visits.len()
    }

    /// Reset tracker for new episode.
    pub fn reset(&mut self) {
        self.visits.clear();
        self.total = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_zero_entropy() {
        let t = StateVisitationTracker::new(0.8);
        assert_eq!(t.entropy(), 0.0);
        assert_eq!(t.unique_states(), 0);
        assert!(!t.coverage_ok());
    }

    #[test]
    fn single_state_zero_entropy() {
        let mut t = StateVisitationTracker::new(0.8);
        t.observe(1);
        t.observe(1);
        assert_eq!(t.entropy(), 0.0);
    }

    #[test]
    fn uniform_high_entropy() {
        let mut t = StateVisitationTracker::new(0.5);
        for i in 0..4u64 {
            t.observe(i);
        }
        // Uniform over 4 states: entropy = log2(4) = 2.0
        let e = t.entropy();
        assert!((e - 2.0f32).abs() < 0.01, "expected ~2.0, got {e}");
        assert!(t.coverage_ok()); // normalized entropy = 1.0 >= 0.5
    }

    #[test]
    fn reset_clears() {
        let mut t = StateVisitationTracker::new(0.8);
        t.observe(1);
        t.observe(2);
        t.reset();
        assert_eq!(t.unique_states(), 0);
        assert_eq!(t.entropy(), 0.0);
    }
}
