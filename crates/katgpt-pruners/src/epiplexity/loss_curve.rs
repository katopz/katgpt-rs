//! Loss Curve Tracker — hooks into training loop for epiplexity estimation.
//!
//! Two granularities:
//! - [`LossCurveTracker`] — batch/epoch level, prequential estimate
//! - [`PerPositionLossTracker`] — per-token-position fine-grained scoring

use std::collections::VecDeque;

use super::EpiplexityEstimator;

// ── LossCurveTracker ────────────────────────────────────────────

/// Tracks loss curves at batch/epoch granularity for prequential epiplexity estimation.
///
/// The prequential estimate uses the running loss history to estimate S_T
/// without requiring a separate held-out evaluation pass.
///
/// ```text
/// batch losses ──→ ring buffer ──→ prequential S_T estimate
/// epoch losses  ──→ epoch buffer ──→ epoch-level S_T
/// ```
#[derive(Clone, Debug)]
pub struct LossCurveTracker {
    /// Per-batch average losses (ring buffer).
    batch_losses: VecDeque<f32>,
    /// Per-epoch validation losses (ring buffer).
    epoch_losses: VecDeque<f32>,
    /// Maximum batches to retain.
    batch_capacity: usize,
    /// Maximum epochs to retain.
    epoch_capacity: usize,
    /// Running minimum loss (prequential final loss estimate).
    running_min: f32,
    /// Epiplexity estimator for batch-level scoring.
    estimator: EpiplexityEstimator,
}

impl LossCurveTracker {
    /// Create a new tracker with bounded history.
    pub fn new(batch_capacity: usize, epoch_capacity: usize) -> Self {
        Self {
            batch_losses: VecDeque::with_capacity(batch_capacity),
            epoch_losses: VecDeque::with_capacity(epoch_capacity),
            batch_capacity: batch_capacity.max(1),
            epoch_capacity: epoch_capacity.max(1),
            running_min: f32::MAX,
            estimator: EpiplexityEstimator::new(batch_capacity),
        }
    }

    /// Record the end of a training batch.
    pub fn on_batch_end(&mut self, _batch_idx: usize, avg_loss: f32) {
        if self.batch_losses.len() >= self.batch_capacity {
            self.batch_losses.pop_front();
        }
        self.batch_losses.push_back(avg_loss);
        self.estimator.record_step(avg_loss);

        if avg_loss < self.running_min {
            self.running_min = avg_loss;
        }
    }

    /// Record the end of a training epoch.
    pub fn on_epoch_end(&mut self, _epoch: usize, val_loss: f32) {
        if self.epoch_losses.len() >= self.epoch_capacity {
            self.epoch_losses.pop_front();
        }
        self.epoch_losses.push_back(val_loss);
    }

    /// Prequential epiplexity estimate: S_T using running minimum as final loss.
    ///
    /// Uses the running minimum loss as the "final loss" estimate,
    /// computing area above this baseline across all recorded batch losses.
    pub fn epiplexity_estimate(&self) -> f32 {
        let final_loss = self.running_min();
        self.estimator.compute_epiplexity(final_loss)
    }

    /// Epoch-level epiplexity: S_T using last epoch loss as final.
    pub fn epoch_epiplexity(&self) -> f32 {
        let final_loss = match self.epoch_losses.back() {
            Some(&l) => l,
            None => return 0.0,
        };
        self.epoch_losses
            .iter()
            .map(|&loss| (loss - final_loss).max(0.0))
            .sum()
    }

    /// Running minimum loss (prequential final loss estimate).
    pub fn running_min(&self) -> f32 {
        if self.running_min == f32::MAX {
            0.0
        } else {
            self.running_min
        }
    }

    /// Number of recorded batches.
    pub fn batch_count(&self) -> usize {
        self.batch_losses.len()
    }

    /// Number of recorded epochs.
    pub fn epoch_count(&self) -> usize {
        self.epoch_losses.len()
    }

    /// Access the underlying estimator.
    pub fn estimator(&self) -> &EpiplexityEstimator {
        &self.estimator
    }

    /// Get the latest batch loss (if any).
    pub fn latest_batch_loss(&self) -> Option<f32> {
        self.batch_losses.back().copied()
    }

    /// Get the latest epoch loss (if any).
    pub fn latest_epoch_loss(&self) -> Option<f32> {
        self.epoch_losses.back().copied()
    }

    /// Compute loss drop from first to last batch.
    pub fn total_loss_drop(&self) -> f32 {
        match (self.batch_losses.front(), self.batch_losses.back()) {
            (Some(&first), Some(&last)) => (first - last).max(0.0),
            _ => 0.0,
        }
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.batch_losses.clear();
        self.epoch_losses.clear();
        self.running_min = f32::MAX;
        self.estimator.clear();
    }
}

// ── PerPositionLossTracker ──────────────────────────────────────

/// Fine-grained per-token-position loss tracking for epiplexity scoring.
///
/// Tracks the loss contribution at each token position across training
/// steps, enabling per-position epiplexity analysis.
///
/// Use cases:
/// - Identify which positions carry the most structural information
/// - Rank training samples by per-position epiplexity contribution
/// - Validate that structured positions have higher S_T than random ones
#[derive(Clone, Debug)]
pub struct PerPositionLossTracker {
    /// Per-position loss histories (ring buffers of bounded capacity).
    position_losses: Vec<VecDeque<f32>>,
    /// Maximum history length per position.
    capacity: usize,
}

impl PerPositionLossTracker {
    /// Create a new per-position tracker.
    ///
    /// `n_positions` — number of token positions to track.
    /// `capacity` — max history length per position.
    pub fn new(n_positions: usize, capacity: usize) -> Self {
        let position_losses = (0..n_positions)
            .map(|_| VecDeque::with_capacity(capacity))
            .collect();
        Self {
            position_losses,
            capacity: capacity.max(1),
        }
    }

    /// Record per-position losses for a single training step.
    ///
    /// `losses[i]` = loss at position `i`. Length must match `n_positions`.
    pub fn record_step(&mut self, losses: &[f32]) {
        for (pos, loss) in losses.iter().enumerate() {
            if pos >= self.position_losses.len() {
                break;
            }
            let buf = &mut self.position_losses[pos];
            if buf.len() >= self.capacity {
                buf.pop_front();
            }
            buf.push_back(*loss);
        }
    }

    /// Compute per-position epiplexity contribution.
    ///
    /// Uses the minimum loss at each position as the "final loss" estimate.
    /// Returns a vector where element `i` is the epiplexity at position `i`.
    pub fn per_position_epiplexity(&self) -> Vec<f32> {
        self.position_losses
            .iter()
            .map(|buf| {
                if buf.is_empty() {
                    return 0.0;
                }
                let final_loss = buf.iter().copied().fold(f32::MAX, f32::min);
                buf.iter().map(|&loss| (loss - final_loss).max(0.0)).sum()
            })
            .collect()
    }

    /// Compute per-position epiplexity using provided final losses.
    pub fn per_position_epiplexity_with_final(&self, final_losses: &[f32]) -> Vec<f32> {
        self.position_losses
            .iter()
            .enumerate()
            .map(|(pos, buf)| {
                let final_loss = final_losses.get(pos).copied().unwrap_or(0.0);
                buf.iter().map(|&loss| (loss - final_loss).max(0.0)).sum()
            })
            .collect()
    }

    /// Total epiplexity across all positions.
    pub fn total_epiplexity(&self) -> f32 {
        self.per_position_epiplexity().iter().sum()
    }

    /// Number of positions being tracked.
    pub fn n_positions(&self) -> usize {
        self.position_losses.len()
    }

    /// Number of recorded steps at a given position.
    pub fn step_count(&self, position: usize) -> usize {
        self.position_losses
            .get(position)
            .map(|buf| buf.len())
            .unwrap_or(0)
    }

    /// Get the loss history for a specific position.
    pub fn position_losses(&self, position: usize) -> Option<&VecDeque<f32>> {
        self.position_losses.get(position)
    }

    /// Identify positions with highest structural information (top-k).
    pub fn top_k_structural(&self, k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .per_position_epiplexity()
            .into_iter()
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Clear all position histories.
    pub fn clear(&mut self) {
        for buf in &mut self.position_losses {
            buf.clear();
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_tracking() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        tracker.on_batch_end(1, 4.0);
        tracker.on_batch_end(2, 3.0);
        assert_eq!(tracker.batch_count(), 3);
        assert!((tracker.latest_batch_loss().unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_epoch_tracking() {
        let mut tracker = LossCurveTracker::new(100, 5);
        tracker.on_epoch_end(0, 4.5);
        tracker.on_epoch_end(1, 3.5);
        assert_eq!(tracker.epoch_count(), 2);
        assert!((tracker.latest_epoch_loss().unwrap() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_prequential_estimate() {
        let mut tracker = LossCurveTracker::new(100, 10);
        // Structured: losses decrease
        for i in 0..10 {
            let loss = 5.0 - (i as f32) * 0.4;
            tracker.on_batch_end(i, loss);
        }
        let s = tracker.epiplexity_estimate();
        assert!(s > 0.0, "structured data should have S>0, got {s}");
    }

    #[test]
    fn test_running_min_updates() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        assert!((tracker.running_min() - 5.0).abs() < 1e-6);
        tracker.on_batch_end(1, 3.0);
        assert!((tracker.running_min() - 3.0).abs() < 1e-6);
        tracker.on_batch_end(2, 4.0); // goes up, min unchanged
        assert!((tracker.running_min() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_epoch_epiplexity() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_epoch_end(0, 5.0);
        tracker.on_epoch_end(1, 3.0);
        tracker.on_epoch_end(2, 2.0);
        let s = tracker.epoch_epiplexity();
        // S = (5-2) + (3-2) + (2-2) = 3 + 1 + 0 = 4.0
        assert!((s - 4.0).abs() < 1e-6, "expected 4.0, got {s}");
    }

    #[test]
    fn test_epoch_epiplexity_empty() {
        let tracker = LossCurveTracker::new(100, 10);
        assert_eq!(tracker.epoch_epiplexity(), 0.0);
    }

    #[test]
    fn test_total_loss_drop() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        tracker.on_batch_end(1, 3.0);
        tracker.on_batch_end(2, 2.0);
        assert!((tracker.total_loss_drop() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_ring_buffer_overflow() {
        let mut tracker = LossCurveTracker::new(3, 10);
        for i in 0..5 {
            tracker.on_batch_end(i, i as f32);
        }
        assert_eq!(tracker.batch_count(), 3);
    }

    #[test]
    fn test_clear_resets() {
        let mut tracker = LossCurveTracker::new(100, 10);
        tracker.on_batch_end(0, 5.0);
        tracker.on_epoch_end(0, 4.0);
        tracker.clear();
        assert_eq!(tracker.batch_count(), 0);
        assert_eq!(tracker.epoch_count(), 0);
        assert_eq!(tracker.running_min(), 0.0);
    }

    #[test]
    fn test_per_position_basic() {
        let mut tracker = PerPositionLossTracker::new(3, 10);
        tracker.record_step(&[5.0, 4.0, 3.0]);
        tracker.record_step(&[4.0, 3.0, 2.0]);
        tracker.record_step(&[3.0, 2.0, 1.0]);

        let epi = tracker.per_position_epiplexity();
        assert_eq!(epi.len(), 3);
        // All positions have decreasing structure → S > 0
        for (i, &s) in epi.iter().enumerate() {
            assert!(s > 0.0, "position {i} should have S>0, got {s}");
        }
    }

    #[test]
    fn test_per_position_with_final() {
        let mut tracker = PerPositionLossTracker::new(2, 10);
        tracker.record_step(&[3.0, 2.0]);
        tracker.record_step(&[2.0, 1.0]);

        let final_losses = vec![1.0, 0.5];
        let epi = tracker.per_position_epiplexity_with_final(&final_losses);
        // pos 0: (3-1) + (2-1) = 3.0
        assert!((epi[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", epi[0]);
        // pos 1: (2-0.5) + (1-0.5) = 2.0
        assert!((epi[1] - 2.0).abs() < 1e-5, "expected 2.0, got {}", epi[1]);
    }

    #[test]
    fn test_per_position_empty() {
        let tracker = PerPositionLossTracker::new(3, 10);
        let epi = tracker.per_position_epiplexity();
        assert_eq!(epi, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_total_epiplexity() {
        let mut tracker = PerPositionLossTracker::new(2, 10);
        tracker.record_step(&[4.0, 3.0]);
        tracker.record_step(&[2.0, 1.0]);
        let total = tracker.total_epiplexity();
        assert!(total > 0.0, "total should be > 0, got {total}");
    }

    #[test]
    fn test_top_k_structural() {
        let mut tracker = PerPositionLossTracker::new(4, 10);
        // Position 0: high structure (large drop)
        // Position 1: medium structure
        // Position 2: low structure (constant)
        // Position 3: no data
        tracker.record_step(&[8.0, 5.0, 3.0, 0.0]);
        tracker.record_step(&[6.0, 4.0, 3.0, 0.0]);
        tracker.record_step(&[2.0, 3.0, 3.0, 0.0]);

        let top2 = tracker.top_k_structural(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0, "position 0 should be most structural");
    }

    #[test]
    fn test_position_step_count() {
        let mut tracker = PerPositionLossTracker::new(3, 10);
        tracker.record_step(&[1.0, 2.0, 3.0]);
        tracker.record_step(&[1.0, 2.0, 3.0]);
        assert_eq!(tracker.step_count(0), 2);
        assert_eq!(tracker.step_count(1), 2);
        assert_eq!(tracker.step_count(99), 0); // out of bounds
    }

    #[test]
    fn test_position_ring_buffer_overflow() {
        let mut tracker = PerPositionLossTracker::new(1, 3);
        for i in 0..5 {
            tracker.record_step(&[i as f32]);
        }
        assert_eq!(tracker.step_count(0), 3);
    }
}
