//! Factorization Scoring for Game Traces
//!
//! Based on the epiplexity paper (arXiv:2601.03220): the order in which data
//! is presented to a learner affects the structural information extractable.
//!
//! - **Forward** order: actions → state (easy to compute, e.g. moves → board)
//! - **Reverse** order: state → actions (requires inference, e.g. board → moves)
//!
//! The paper shows reverse order has higher epiplexity (S_T) AND better OOD
//! accuracy for structured data like chess games.

use super::EpiplexityEstimator;

/// Which factorization order to use for scoring game traces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FactorizationOrder {
    /// Actions → State order (e.g., moves → board position).
    /// Easier to compute, lower epiplexity for structured data.
    Forward,
    /// State → Actions order (e.g., board position → moves).
    /// Requires inference, higher epiplexity per the paper.
    Reverse,
    /// Choose per-trace based on which has higher epiplexity.
    #[default]
    Adaptive,
}

impl std::fmt::Display for FactorizationOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forward => write!(f, "Forward"),
            Self::Reverse => write!(f, "Reverse"),
            Self::Adaptive => write!(f, "Adaptive"),
        }
    }
}

/// Scores game traces by their factorization (ordering) quality using epiplexity.
///
/// A "trace" is a sequence of f32 values representing loss or complexity
/// at each step of a game trajectory. The scorer compares forward vs reverse
/// ordering to determine which carries more structural information.
///
/// # Paper Finding
///
/// For chess game traces, the paper (arXiv:2601.03220) shows:
/// - Reverse order (board → moves) has **higher** S_T than forward (moves → board)
/// - Higher S_T correlates with better OOD generalization
/// - This validates that "explaining" moves from positions is harder but more informative
#[derive(Clone, Debug)]
pub struct FactorizationScorer {
    /// Capacity for the internal epiplexity estimators.
    capacity: usize,
}

impl FactorizationScorer {
    /// Create a new scorer with the given ring buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
        }
    }

    /// Score a trace in forward (as-is) order.
    ///
    /// Uses the **last** value as the "final loss" — this models the training
    /// process where the final converged loss is the endpoint. For a decreasing
    /// loss curve (structured data), S_T will be large. For increasing or flat,
    /// S_T will be small.
    pub fn score_forward(&self, trace: &[f32]) -> f32 {
        if trace.is_empty() {
            return 0.0;
        }
        let final_loss = *trace.last().unwrap_or(&0.0);
        let mut est = EpiplexityEstimator::new(self.capacity);
        for &loss in trace {
            est.record_step(loss);
        }
        est.compute_epiplexity(final_loss)
    }

    /// Score a trace in reverse order.
    ///
    /// Reverses the trace and computes epiplexity using the last element
    /// (i.e., the **first** element of the original trace) as final_loss.
    /// For a monotonically decreasing trace, reverse has low S_T because
    /// the "final loss" (original first, which is high) dominates.
    /// This simulates "state → actions" ordering per the epiplexity paper.
    pub fn score_reverse(&self, trace: &[f32]) -> f32 {
        if trace.is_empty() {
            return 0.0;
        }
        let reversed: Vec<f32> = trace.iter().copied().rev().collect();
        self.score_forward(&reversed)
    }

    /// Determine the preferred factorization order for a given trace.
    ///
    /// Compares forward vs reverse epiplexity and returns the order
    /// with higher S_T.
    pub fn preferred_order(&self, trace: &[f32]) -> FactorizationOrder {
        let forward = self.score_forward(trace);
        let reverse = self.score_reverse(trace);
        match reverse > forward {
            true => FactorizationOrder::Reverse,
            false => FactorizationOrder::Forward,
        }
    }

    /// Score a trace according to the given factorization order.
    pub fn score(&self, trace: &[f32], order: FactorizationOrder) -> f32 {
        match order {
            FactorizationOrder::Forward => self.score_forward(trace),
            FactorizationOrder::Reverse => self.score_reverse(trace),
            FactorizationOrder::Adaptive => {
                let fwd = self.score_forward(trace);
                let rev = self.score_reverse(trace);
                fwd.max(rev)
            }
        }
    }

    /// Compute the epiplexity gap: S_reverse - S_forward.
    ///
    /// Per the paper, structured data (e.g., self-play chess traces)
    /// should have a positive gap — reverse order carries more information.
    /// Random data should have gap ≈ 0.
    pub fn epiplexity_gap(&self, trace: &[f32]) -> f32 {
        self.score_reverse(trace) - self.score_forward(trace)
    }

    /// Score multiple traces and rank them by epiplexity in the given order.
    ///
    /// Returns indices sorted by descending epiplexity score.
    pub fn rank_traces(&self, traces: &[&[f32]], order: FactorizationOrder) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = traces
            .iter()
            .enumerate()
            .map(|(i, trace)| (i, self.score(trace, order)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    /// Compare forward vs reverse for a batch of traces.
    ///
    /// Returns the count of traces where each order is preferred.
    pub fn order_preference_counts(traces: &[&[f32]], capacity: usize) -> (usize, usize) {
        let scorer = Self::new(capacity);
        let mut forward_wins = 0usize;
        let mut reverse_wins = 0usize;
        for trace in traces {
            match scorer.preferred_order(trace) {
                FactorizationOrder::Forward => forward_wins += 1,
                FactorizationOrder::Reverse => reverse_wins += 1,
                FactorizationOrder::Adaptive => {} // won't happen from preferred_order
            }
        }
        (forward_wins, reverse_wins)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_scoring_empty() {
        let scorer = FactorizationScorer::new(100);
        assert_eq!(scorer.score_forward(&[]), 0.0);
    }

    #[test]
    fn test_forward_scoring_decreasing() {
        let scorer = FactorizationScorer::new(100);
        // Decreasing: last=1.4, excess = Σ(5.0-i*0.4 - 1.4) > 0
        let trace: Vec<f32> = (0..10).map(|i| 5.0 - (i as f32) * 0.4).collect();
        let s = scorer.score_forward(&trace);
        assert!(s > 0.0, "decreasing trace should have S>0, got {s}");
    }

    #[test]
    fn test_forward_scoring_constant() {
        let scorer = FactorizationScorer::new(100);
        let trace = vec![3.0; 10];
        let s = scorer.score_forward(&trace);
        assert!(s < 0.01, "constant trace should have S≈0, got {s}");
    }

    #[test]
    fn test_forward_scoring_increasing_near_zero() {
        let scorer = FactorizationScorer::new(100);
        // Increasing: last=4.6, all prior values < last → S≈0
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.4).collect();
        let s = scorer.score_forward(&trace);
        assert!(
            s < 0.01,
            "increasing trace should have S≈0 (last is max), got {s}"
        );
    }

    #[test]
    fn test_reverse_reverses_trace() {
        let scorer = FactorizationScorer::new(100);
        // Increasing trace: forward S≈0 (last=max), reversed S>0 (last=min)
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.4).collect();
        let fwd = scorer.score_forward(&trace);
        let rev = scorer.score_reverse(&trace);
        assert!(
            rev > fwd,
            "reversed increasing trace should have higher S: rev={rev}, fwd={fwd}"
        );
    }

    #[test]
    fn test_preferred_order_decreasing() {
        let scorer = FactorizationScorer::new(100);
        // Decreasing: forward has high S (last=min), reverse has S≈0 (last=max)
        let trace: Vec<f32> = (0..10).map(|i| 5.0 - (i as f32) * 0.4).collect();
        let order = scorer.preferred_order(&trace);
        assert_eq!(order, FactorizationOrder::Forward);
    }

    #[test]
    fn test_preferred_order_increasing() {
        let scorer = FactorizationScorer::new(100);
        // Increasing: forward S≈0 (last=max), reverse S>0 (last=min after reversal)
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.4).collect();
        let order = scorer.preferred_order(&trace);
        assert_eq!(order, FactorizationOrder::Reverse);
    }

    #[test]
    fn test_epiplexity_gap_structured() {
        let scorer = FactorizationScorer::new(100);
        // Structured: increasing trace → reverse has higher S → positive gap
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.4).collect();
        let gap = scorer.epiplexity_gap(&trace);
        assert!(
            gap > 0.0,
            "structured increasing trace → positive gap, got {gap}"
        );
    }

    #[test]
    fn test_epiplexity_gap_constant() {
        let scorer = FactorizationScorer::new(100);
        let trace = vec![3.0; 10];
        let gap = scorer.epiplexity_gap(&trace);
        assert!(gap.abs() < 0.01, "constant trace → gap≈0, got {gap}");
    }

    #[test]
    fn test_score_adaptive_takes_max() {
        let scorer = FactorizationScorer::new(100);
        // Increasing: reverse > forward
        let trace: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.4).collect();
        let adaptive = scorer.score(&trace, FactorizationOrder::Adaptive);
        let reverse = scorer.score_reverse(&trace);
        assert!(
            (adaptive - reverse).abs() < 1e-6,
            "adaptive should pick reverse (higher), got adaptive={adaptive}, reverse={reverse}"
        );
    }

    #[test]
    fn test_rank_traces() {
        let scorer = FactorizationScorer::new(100);
        let decreasing: Vec<f32> = (0..10).map(|i| 5.0 - (i as f32) * 0.4).collect();
        let constant = vec![3.0; 10];
        let increasing: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.4).collect();

        let traces: &[&[f32]] = &[&constant, &increasing, &decreasing];
        let ranked = scorer.rank_traces(traces, FactorizationOrder::Forward);

        // Forward order: decreasing (high S) > constant ≈ increasing (both ≈0)
        assert_eq!(ranked[0].0, 2, "decreasing should rank first (forward)");
    }

    #[test]
    fn test_order_preference_counts() {
        let decreasing: Vec<f32> = (0..10).map(|i| 5.0 - (i as f32) * 0.4).collect();
        let increasing: Vec<f32> = (0..10).map(|i| 1.0 + (i as f32) * 0.4).collect();

        let traces: &[&[f32]] = &[&decreasing, &increasing];
        let (fwd, rev) = FactorizationScorer::order_preference_counts(traces, 100);
        assert_eq!(fwd, 1, "decreasing prefers forward");
        assert_eq!(rev, 1, "increasing prefers reverse");
    }

    #[test]
    fn test_default_factorization_order() {
        assert_eq!(FactorizationOrder::default(), FactorizationOrder::Adaptive);
    }

    #[test]
    fn test_factorization_order_display() {
        assert_eq!(format!("{}", FactorizationOrder::Forward), "Forward");
        assert_eq!(format!("{}", FactorizationOrder::Reverse), "Reverse");
        assert_eq!(format!("{}", FactorizationOrder::Adaptive), "Adaptive");
    }
}
