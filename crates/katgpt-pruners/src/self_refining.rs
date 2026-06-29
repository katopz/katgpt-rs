//! Self-Refining Pruner — bandit-driven threshold and topology adjustment (Plan 214 P2).
//!
//! Tracks per-slot accuracy (TP/TN/FP/FN) and uses sigmoid-based threshold adjustment
//! to reduce false positives/negatives. Topology mode prunes low-value branches and
//! expands high-value ones based on acceptance rate.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "coexplain_pruner")]`.

use super::ted_lite::PrunerDivergence;

// ── Sigmoid ─────────────────────────────────────────────────────────

/// Numerically stable sigmoid: σ(x) = 1 / (1 + exp(-x)).
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ── PrunerAccuracy ──────────────────────────────────────────────────

/// Tracks per-slot accuracy for a self-refining pruner.
///
/// Each slot stores `[true_positive, true_negative, false_positive, false_negative]`.
#[derive(Debug, Clone)]
pub struct PrunerAccuracy {
    /// Per-slot: [TP, TN, FP, FN]
    pub slots: Vec<[u32; 4]>,
}

impl PrunerAccuracy {
    /// Create a new accuracy tracker with `slot_count` slots, all zeroed.
    pub fn new(slot_count: usize) -> Self {
        Self {
            slots: vec![[0u32; 4]; slot_count],
        }
    }

    /// Record a prediction outcome for a given slot.
    ///
    /// Updates TP/TN/FP/FN counters based on predicted vs actual accept/reject.
    pub fn record(&mut self, slot: usize, predicted_accept: bool, actual_accept: bool) {
        if slot >= self.slots.len() {
            return;
        }
        match (predicted_accept, actual_accept) {
            (true, true) => self.slots[slot][0] += 1,   // TP
            (false, false) => self.slots[slot][1] += 1, // TN
            (true, false) => self.slots[slot][2] += 1,  // FP
            (false, true) => self.slots[slot][3] += 1,  // FN
        }
    }

    /// Precision for a slot: TP / (TP + FP).
    ///
    /// Returns 0.0 if TP + FP = 0.
    pub fn precision(&self, slot: usize) -> f32 {
        if slot >= self.slots.len() {
            return 0.0;
        }
        let [tp, _, fp, _] = self.slots[slot];
        let denom = tp + fp;
        match denom {
            0 => 0.0,
            _ => tp as f32 / denom as f32,
        }
    }

    /// Recall for a slot: TP / (TP + FN).
    ///
    /// Returns 0.0 if TP + FN = 0.
    pub fn recall(&self, slot: usize) -> f32 {
        if slot >= self.slots.len() {
            return 0.0;
        }
        let [tp, _, _, fn_] = self.slots[slot];
        let denom = tp + fn_;
        match denom {
            0 => 0.0,
            _ => tp as f32 / denom as f32,
        }
    }

    /// F1 score for a slot: harmonic mean of precision and recall.
    ///
    /// Returns 0.0 if precision + recall = 0.
    pub fn f1(&self, slot: usize) -> f32 {
        let p = self.precision(slot);
        let r = self.recall(slot);
        match p + r {
            0.0 => 0.0,
            s => 2.0 * p * r / s,
        }
    }

    /// Acceptance rate for a slot: (TP + TN) / total.
    ///
    /// Returns 0.5 (neutral) if total = 0.
    pub fn acceptance_rate(&self, slot: usize) -> f32 {
        if slot >= self.slots.len() {
            return 0.5;
        }
        let [tp, tn, fp, fn_] = self.slots[slot];
        let total = tp + tn + fp + fn_;
        match total {
            0 => 0.5,
            _ => (tp + tn) as f32 / total as f32,
        }
    }
}

// ── TopologyAction ──────────────────────────────────────────────────

/// Action to take on a DDTree branch based on acceptance rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TopologyAction {
    /// Acceptance rate < ε → prune this branch.
    Prune,
    /// Acceptance rate > 1-ε → expand this branch.
    Expand,
    /// In between → keep as-is.
    Keep,
}

// ── Threshold Adjustment ────────────────────────────────────────────

/// Compute threshold adjustment based on FP/FN ratio.
///
/// Uses sigmoid: `adjustment = sigmoid(α * (FP_rate - FN_rate))`.
/// The result is clamped via TED-Lite `lambda_t`.
///
/// # Arguments
///
/// * `accuracy`      — per-slot accuracy tracker
/// * `slot`          — slot index to adjust
/// * `learning_rate` — α (default: 0.1)
/// * `lambda_t`      — divergence clamp from TED-Lite
///
/// # Returns
///
/// Threshold adjustment in range `[0, 1]` (sigmoid output), clamped by lambda_t.
pub fn compute_threshold_adjustment(
    accuracy: &PrunerAccuracy,
    slot: usize,
    learning_rate: f32,
    lambda_t: f32,
) -> f32 {
    if slot >= accuracy.slots.len() {
        return 0.0;
    }

    let [tp, _, fp, fn_] = accuracy.slots[slot];

    // FP_rate = FP / (TP + FP)
    let fp_denom = tp + fp;
    let fp_rate = match fp_denom {
        0 => 0.0,
        _ => fp as f32 / fp_denom as f32,
    };

    // FN_rate = FN / (TP + FN)
    let fn_denom = tp + fn_;
    let fn_rate = match fn_denom {
        0 => 0.0,
        _ => fn_ as f32 / fn_denom as f32,
    };

    // Sigmoid adjustment — shifted to be centered at 0
    // Raw sigmoid output is in (0, 1), subtract 0.5 to center around 0
    let raw = sigmoid(learning_rate * (fp_rate - fn_rate)) - 0.5;

    // Clamp via TED-Lite lambda_t
    let div = PrunerDivergence {
        threshold_divergence: 0.0,
        topology_divergence: 0.0,
        lambda_t,
    };
    match div.clamp_adjustment(raw) {
        Some(clamped) => clamped,
        None => raw,
    }
}

// ── Topology Adjustment ─────────────────────────────────────────────

/// Determine topology actions for each branch based on acceptance rates.
///
/// - Acceptance rate < `branch_threshold_low` → [`TopologyAction::Prune`]
/// - Acceptance rate > `branch_threshold_high` → [`TopologyAction::Expand`]
/// - Otherwise → [`TopologyAction::Keep`]
///
/// Returns one action per slot in the accuracy tracker.
pub fn adjust_topology(
    accuracy: &PrunerAccuracy,
    branch_threshold_low: f32,
    branch_threshold_high: f32,
) -> Vec<TopologyAction> {
    accuracy
        .slots
        .iter()
        .enumerate()
        .map(|(slot, _)| {
            let rate = accuracy.acceptance_rate(slot);
            if rate < branch_threshold_low {
                TopologyAction::Prune
            } else if rate > branch_threshold_high {
                TopologyAction::Expand
            } else {
                TopologyAction::Keep
            }
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruner_accuracy_record() {
        let mut acc = PrunerAccuracy::new(2);

        // Slot 0: TP=2, TN=1, FP=1, FN=1
        acc.record(0, true, true); // TP
        acc.record(0, true, true); // TP
        acc.record(0, false, false); // TN
        acc.record(0, true, false); // FP
        acc.record(0, false, true); // FN

        assert_eq!(acc.slots[0], [2, 1, 1, 1]);

        // Slot 1: untouched
        assert_eq!(acc.slots[1], [0, 0, 0, 0]);

        // Out of bounds → no panic
        acc.record(99, true, true);
    }

    #[test]
    fn test_precision_recall_f1() {
        let mut acc = PrunerAccuracy::new(1);
        // TP=3, TN=2, FP=1, FN=1 → total=7
        acc.record(0, true, true); // TP
        acc.record(0, true, true); // TP
        acc.record(0, true, true); // TP
        acc.record(0, false, false); // TN
        acc.record(0, false, false); // TN
        acc.record(0, true, false); // FP
        acc.record(0, false, true); // FN

        // Precision = 3 / (3+1) = 0.75
        assert!((acc.precision(0) - 0.75).abs() < 1e-6);
        // Recall = 3 / (3+1) = 0.75
        assert!((acc.recall(0) - 0.75).abs() < 1e-6);
        // F1 = 2 * 0.75 * 0.75 / 1.5 = 0.75
        assert!((acc.f1(0) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_precision_recall_zero_denominator() {
        let acc = PrunerAccuracy::new(1);
        // All zeros → precision=0, recall=0, f1=0
        assert_eq!(acc.precision(0), 0.0);
        assert_eq!(acc.recall(0), 0.0);
        assert_eq!(acc.f1(0), 0.0);
    }

    #[test]
    fn test_threshold_adjustment_sigmoid() {
        let mut acc = PrunerAccuracy::new(1);

        // More FP than FN → adjustment should be positive (raise threshold)
        acc.record(0, true, true); // TP=1
        acc.record(0, true, false); // FP=1
        acc.record(0, true, false); // FP=2
        acc.record(0, false, true); // FN=1

        let adj = compute_threshold_adjustment(&acc, 0, 0.1, 1.0);
        // FP_rate = 2/(1+2) = 0.667, FN_rate = 1/(1+1) = 0.5
        // sigmoid(0.1 * 0.167) - 0.5 ≈ small positive
        assert!(adj > 0.0, "adjustment should be positive when FP > FN");

        // Verify symmetry: more FN than FP → negative adjustment
        let mut acc2 = PrunerAccuracy::new(1);
        acc2.record(0, true, true); // TP=1
        acc2.record(0, false, true); // FN=1
        acc2.record(0, false, true); // FN=2
        acc2.record(0, true, false); // FP=1

        let adj2 = compute_threshold_adjustment(&acc2, 0, 0.1, 1.0);
        assert!(adj2 < 0.0, "adjustment should be negative when FN > FP");
    }

    #[test]
    fn test_threshold_adjustment_clamped() {
        let mut acc = PrunerAccuracy::new(1);
        // Extreme FP/FN imbalance with high learning rate → should clamp
        acc.record(0, true, true); // TP=1
        acc.record(0, true, false); // FP=1

        let adj = compute_threshold_adjustment(&acc, 0, 100.0, 0.01);
        // With huge learning_rate, sigmoid output → ~0.5 or ~-0.5, clamped to ±0.01
        assert!(adj.abs() <= 0.01 + 1e-6, "should be clamped to lambda_t");
    }

    #[test]
    fn test_threshold_adjustment_empty_slot() {
        let acc = PrunerAccuracy::new(1);
        // No data → adjustment = 0
        let adj = compute_threshold_adjustment(&acc, 0, 0.1, 0.1);
        assert!((adj - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_topology_action_prune_low() {
        let mut acc = PrunerAccuracy::new(1);
        // All predictions wrong → acceptance_rate = 0 → Prune
        acc.record(0, true, false); // FP
        acc.record(0, false, true); // FN

        let actions = adjust_topology(&acc, 0.1, 0.9);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], TopologyAction::Prune);
    }

    #[test]
    fn test_topology_action_expand_high() {
        let mut acc = PrunerAccuracy::new(1);
        // All predictions correct → acceptance_rate = 1.0 → Expand
        acc.record(0, true, true); // TP
        acc.record(0, false, false); // TN

        let actions = adjust_topology(&acc, 0.1, 0.9);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0], TopologyAction::Expand);
    }

    #[test]
    fn test_topology_action_keep_middle() {
        let mut acc = PrunerAccuracy::new(1);
        // Mixed → acceptance_rate around 0.5 → Keep
        acc.record(0, true, true); // TP
        acc.record(0, true, false); // FP

        let actions = adjust_topology(&acc, 0.1, 0.9);
        assert_eq!(actions[0], TopologyAction::Keep);
    }

    #[test]
    fn test_topology_multiple_slots() {
        let mut acc = PrunerAccuracy::new(3);

        // Slot 0: Prune (all wrong)
        acc.record(0, true, false);
        // Slot 1: Keep (mixed)
        acc.record(1, true, true);
        acc.record(1, true, false);
        // Slot 2: Expand (all correct)
        acc.record(2, true, true);

        let actions = adjust_topology(&acc, 0.1, 0.9);
        assert_eq!(actions.len(), 3);
        assert_eq!(actions[0], TopologyAction::Prune);
        assert_eq!(actions[1], TopologyAction::Keep);
        assert_eq!(actions[2], TopologyAction::Expand);
    }
}
