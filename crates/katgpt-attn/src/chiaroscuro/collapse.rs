//! CollapseDiscoveryHarness — automated operator promotion (Plan 269, Fusion C).
//!
//! Implements paper's Remark 1: routing collapse as a discovery mechanism.
//!
//! When a multi-operator router's utilization entropy U → 0, the collapsed
//! subset is the **sufficient operator set** for the workload. Removing
//! redundant operators (paper removes RBF) improves quality.
//!
//! This harness:
//! 1. Runs the operator router under a calibration stream
//! 2. Detects when U drops below threshold (collapse)
//! 3. Identifies the survivor subset
//! 4. Optionally validates by re-running with the collapsed subset
//! 5. Emits `OpPromotion { keep, demote }` recommendations

use crate::chiaroscuro::op_trait::ChiaroscuroRouter;

/// Default collapse threshold. Below this normalized entropy, collapse is detected.
///
/// Paper's RBF-only-collapse had U ≈ 0.02 within 1000 training steps.
/// We use 0.10 to allow some diversity while still catching effective collapse.
pub const DEFAULT_COLLAPSE_THRESHOLD: f32 = 0.10;

/// Default sliding window size for utilization tracking.
pub const DEFAULT_WINDOW_SIZE: usize = 1024;

/// Recommendation from collapse discovery.
///
/// Tells the caller which operators to keep (high-utilization survivors) and
/// which to demote (zero-utilization candidates for removal).
#[derive(Clone, Debug, PartialEq)]
pub struct OpPromotion {
    /// Indices of operators with non-zero utilization (the survivor subset).
    pub keep: Vec<usize>,
    /// Indices of operators with zero utilization (candidates for demotion).
    pub demote: Vec<usize>,
    /// Observed utilization entropy U ∈ [0, 1].
    pub utilization_entropy: f32,
    /// Total observations in the window.
    pub total_observations: u64,
}

impl OpPromotion {
    /// Whether collapse was detected.
    pub fn collapsed(&self) -> bool {
        self.utilization_entropy < DEFAULT_COLLAPSE_THRESHOLD && self.total_observations > 32
    }

    /// Number of operators recommended for demotion.
    #[inline]
    pub fn num_demoted(&self) -> usize {
        self.demote.len()
    }
}

/// Collapse discovery harness.
///
/// Wraps a [`ChiaroscuroRouter`] and observes its utilization over a sliding
/// window. When utilization entropy U drops below threshold, emits an
/// [`OpPromotion`] identifying the survivor subset.
///
/// # Usage
///
/// ```ignore
/// let mut harness = CollapseDiscoveryHarness::new(router, 1024, 0.10);
/// for token in calibration_stream {
///     harness.observe(token);
/// }
/// if let Some(promotion) = harness.check_collapse() {
///     println!("Collapse detected! Keep: {:?}, Demote: {:?}",
///              promotion.keep, promotion.demote);
/// }
/// ```
pub struct CollapseDiscoveryHarness {
    pub router: ChiaroscuroRouter,
    window_size: usize,
    collapse_threshold: f32,
    /// Whether collapse has been detected and reported.
    collapsed_reported: bool,
}

impl CollapseDiscoveryHarness {
    /// Create a new harness wrapping the given router.
    pub fn new(router: ChiaroscuroRouter, window_size: usize, collapse_threshold: f32) -> Self {
        Self {
            router,
            window_size,
            collapse_threshold,
            collapsed_reported: false,
        }
    }

    /// Observe a single token's H(x) (pre-computed).
    ///
    /// Routes via the wrapped router and tracks utilization.
    #[inline]
    pub fn observe_h(&mut self, h_x: f32) {
        self.router.route_from_h(h_x);
    }

    /// Observe a raw embedding — computes H(x) and routes.
    #[inline]
    pub fn observe(&mut self, x: &[f32]) {
        self.router.route(x);
    }

    /// Number of observations so far.
    #[inline]
    pub fn total_observations(&self) -> u64 {
        self.router.total_observations()
    }

    /// Check if routing has collapsed.
    ///
    /// Returns `Some(OpPromotion)` if U < threshold and not already reported,
    /// else `None`. Once collapse is detected, subsequent calls return `None`
    /// unless [`reset`] is called.
    pub fn check_collapse(&mut self) -> Option<OpPromotion> {
        if self.collapsed_reported {
            return None;
        }
        let total = self.router.total_observations();
        if (total as usize) < self.window_size {
            return None;
        }
        let u = self.router.utilization_entropy();
        if u < self.collapse_threshold {
            let keep = self.router.survivor_ops();
            let demote = self.router.zero_utilization_ops();
            let promotion = OpPromotion {
                keep,
                demote,
                utilization_entropy: u,
                total_observations: total,
            };
            self.collapsed_reported = true;
            Some(promotion)
        } else {
            None
        }
    }

    /// Get the current promotion snapshot (regardless of collapse state).
    ///
    /// Useful for diagnostic logging — always returns the current survivor/demote
    /// split, even before collapse is detected.
    pub fn current_snapshot(&self) -> OpPromotion {
        OpPromotion {
            keep: self.router.survivor_ops(),
            demote: self.router.zero_utilization_ops(),
            utilization_entropy: self.router.utilization_entropy(),
            total_observations: self.router.total_observations(),
        }
    }

    /// Reset the harness for a new calibration run.
    pub fn reset(&mut self) {
        self.router.reset_utilization();
        self.collapsed_reported = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chiaroscuro::op_trait::{ChiaroscuroOp, DctMixOp, FullAttnOp};

    fn make_router() -> ChiaroscuroRouter {
        let ops: Vec<Box<dyn ChiaroscuroOp>> =
            vec![Box::new(DctMixOp::default()), Box::new(FullAttnOp::default())];
        ChiaroscuroRouter::new(ops)
    }

    #[test]
    fn test_no_collapse_before_window() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 100, 0.10);
        // Only 10 observations — not enough to evaluate.
        for _ in 0..10 {
            harness.observe_h(0.5); // → DctMix
        }
        assert!(harness.check_collapse().is_none());
    }

    #[test]
    fn test_detects_collapse_to_dct() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 100, 0.10);
        // All low-entropy → all go to DctMix → FullAttn gets 0 utilization.
        for _ in 0..200 {
            harness.observe_h(0.5);
        }
        let promotion = harness.check_collapse().expect("should detect collapse");
        assert!(promotion.collapsed(), "promotion should signal collapse");
        assert_eq!(promotion.keep, vec![0], "DctMix should be the survivor");
        assert_eq!(promotion.demote, vec![1], "FullAttn should be demoted");
        assert!(promotion.utilization_entropy < 0.10);
    }

    #[test]
    fn test_detects_collapse_to_full_attn() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 100, 0.10);
        // All high-entropy → all go to FullAttn → DctMix gets 0 utilization.
        for _ in 0..200 {
            harness.observe_h(0.95);
        }
        let promotion = harness.check_collapse().expect("should detect collapse");
        assert_eq!(promotion.keep, vec![1], "FullAttn should be the survivor");
        assert_eq!(promotion.demote, vec![0], "DctMix should be demoted");
    }

    #[test]
    fn test_no_collapse_when_uniform() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 100, 0.10);
        for _ in 0..100 {
            harness.observe_h(0.5); // → DctMix
            harness.observe_h(0.95); // → FullAttn
        }
        // Total = 200, but balanced → U ≈ 1.0 → no collapse.
        let promotion = harness.check_collapse();
        assert!(promotion.is_none(), "no collapse when utilization is uniform");
    }

    #[test]
    fn test_reported_once() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 50, 0.10);
        for _ in 0..100 {
            harness.observe_h(0.5);
        }
        assert!(harness.check_collapse().is_some(), "first detection");
        assert!(harness.check_collapse().is_none(), "not re-detected");
    }

    #[test]
    fn test_reset_clears_state() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 50, 0.10);
        for _ in 0..100 {
            harness.observe_h(0.5);
        }
        assert!(harness.check_collapse().is_some());
        harness.reset();
        assert_eq!(harness.total_observations(), 0);
        // Can detect again after reset.
        for _ in 0..100 {
            harness.observe_h(0.95);
        }
        let promotion = harness.check_collapse().expect("should re-detect after reset");
        assert_eq!(promotion.keep, vec![1]); // now FullAttn survives
    }

    #[test]
    fn test_snapshot_works_pre_collapse() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 100, 0.10);
        harness.observe_h(0.5);
        harness.observe_h(0.95);
        let snap = harness.current_snapshot();
        assert_eq!(snap.total_observations, 2);
        assert_eq!(snap.keep.len(), 2, "both ops should have non-zero utilization");
        assert_eq!(snap.demote.len(), 0);
    }

    #[test]
    fn test_observe_with_embedding_runs() {
        let router = make_router();
        let mut harness = CollapseDiscoveryHarness::new(router, 10, 0.10);
        // Constant embedding → low entropy → DctMix.
        harness.observe(&[1.0f32; 64]);
        assert_eq!(harness.total_observations(), 1);
    }
}
