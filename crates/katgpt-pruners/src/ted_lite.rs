//! TED-Lite Divergence Metric — measures pruner drift from golden reference (Plan 214 P1).
//!
//! Computes threshold and topology divergence between current and original pruner state.
//! Rejects bandit updates that exceed configurable divergence clamp (lambda_t).
//!
//! # Metric Definitions
//!
//! - **Threshold divergence**: Σ |τ_current - τ_original| / N — L1 distance on threshold vectors
//! - **Topology divergence**: Hamming distance / N — fraction of branches that changed existence
//! - **lambda_t**: developer-configurable clamp (default 0.1) that bounds how far a pruner can drift
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "ted_lite")]`.

/// Divergence metrics for a single pruner.
#[derive(Debug, Clone)]
pub struct PrunerDivergence {
    /// Σ |τ_current - τ_original| / N — threshold divergence.
    pub threshold_divergence: f32,
    /// Hamming distance on branch existence vectors / N — topology divergence.
    pub topology_divergence: f32,
    /// Developer-configurable divergence clamp (default: 0.1).
    pub lambda_t: f32,
}

impl PrunerDivergence {
    /// Compute divergence between current and original pruner thresholds.
    ///
    /// O(k) per pruner where k = number of threshold slots.
    ///
    /// # Arguments
    ///
    /// * `current_thresholds`  — active threshold values
    /// * `original_thresholds` — golden-reference threshold values
    /// * `current_branches`    — active branch existence vector
    /// * `original_branches`   — golden-reference branch existence vector
    /// * `lambda_t`            — divergence clamp (clamped to ≥ 0)
    pub fn compute(
        current_thresholds: &[f32],
        original_thresholds: &[f32],
        current_branches: &[bool],
        original_branches: &[bool],
        lambda_t: f32,
    ) -> Self {
        // Threshold divergence: L1 / N
        let n_thresh = current_thresholds.len().max(original_thresholds.len());
        let threshold_divergence = match n_thresh {
            0 => 0.0,
            _ => {
                let sum: f32 = current_thresholds
                    .iter()
                    .zip(original_thresholds.iter())
                    .map(|(c, o)| (c - o).abs())
                    .sum();
                // Handle unequal lengths: count missing elements as full divergence
                let extra =
                    (current_thresholds.len() as f32 - original_thresholds.len() as f32).abs();
                (sum + extra) / n_thresh as f32
            }
        };

        // Topology divergence: Hamming distance / N
        let n_branch = current_branches.len().max(original_branches.len());
        let topology_divergence = match n_branch {
            0 => 0.0,
            _ => {
                let matching = current_branches
                    .iter()
                    .zip(original_branches.iter())
                    .filter(|(c, o)| c == o)
                    .count();
                let hamming = n_branch - matching;
                hamming as f32 / n_branch as f32
            }
        };

        let lambda_t = lambda_t.max(0.0);

        Self {
            threshold_divergence,
            topology_divergence,
            lambda_t,
        }
    }

    /// Reject bandit updates that exceed lambda_t.
    ///
    /// Returns `None` if the adjustment is within bounds,
    /// `Some(clamped)` if the proposed delta exceeds the divergence clamp.
    pub fn clamp_adjustment(&self, proposed_delta: f32) -> Option<f32> {
        if proposed_delta.abs() <= self.lambda_t {
            return None;
        }
        // Clamp to lambda_t, preserving sign
        Some(self.lambda_t * proposed_delta.signum())
    }

    /// Emit divergence metrics per N tokens (behind `log` crate).
    ///
    /// Only logs when `token_count` is a multiple of `interval` to avoid log spam.
    pub fn emit_diagnostic(&self, pruner_name: &str, token_count: usize, interval: usize) {
        if interval == 0 || !token_count.is_multiple_of(interval) {
            return;
        }
        log::info!(
            "[TED-Lite] pruner={} tokens={} threshold_div={:.4} topology_div={:.4} lambda_t={:.4}",
            pruner_name,
            token_count,
            self.threshold_divergence,
            self.topology_divergence,
            self.lambda_t,
        );
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_threshold_divergence_identical() {
        let thresholds = [0.5, 0.3, 0.8];
        let branches = [true, false, true];
        let div = PrunerDivergence::compute(&thresholds, &thresholds, &branches, &branches, 0.1);
        assert_eq!(div.threshold_divergence, 0.0);
        assert_eq!(div.topology_divergence, 0.0);
    }

    #[test]
    fn test_compute_threshold_divergence_different() {
        let current = [0.5, 0.3, 0.8];
        let original = [0.4, 0.3, 0.9];
        let branches = [true, false, true];
        let div = PrunerDivergence::compute(&current, &original, &branches, &branches, 0.1);
        // (0.1 + 0.0 + 0.1) / 3 ≈ 0.0667
        assert!(div.threshold_divergence > 0.0);
        assert!((div.threshold_divergence - 0.0667).abs() < 0.001);
        // Branches identical → topology divergence = 0
        assert_eq!(div.topology_divergence, 0.0);
    }

    #[test]
    fn test_compute_topology_divergence_hamming() {
        let thresholds = [0.5];
        let current_branches = [true, false, true, true];
        let original_branches = [true, true, false, true];
        let div = PrunerDivergence::compute(
            &thresholds,
            &thresholds,
            &current_branches,
            &original_branches,
            0.1,
        );
        // Hamming: positions 1 and 2 differ → 2/4 = 0.5
        assert!((div.topology_divergence - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_clamp_accepts_small_delta() {
        let div = PrunerDivergence {
            threshold_divergence: 0.05,
            topology_divergence: 0.0,
            lambda_t: 0.1,
        };
        // |0.05| <= 0.1 → None (accepted)
        assert!(div.clamp_adjustment(0.05).is_none());
        assert!(div.clamp_adjustment(-0.05).is_none());
    }

    #[test]
    fn test_clamp_rejects_large_delta() {
        let div = PrunerDivergence {
            threshold_divergence: 0.2,
            topology_divergence: 0.0,
            lambda_t: 0.1,
        };
        // |0.5| > 0.1 → Some(0.1)
        let clamped = div.clamp_adjustment(0.5);
        assert!(clamped.is_some());
        assert!((clamped.unwrap() - 0.1).abs() < 1e-6);

        // Negative direction
        let clamped_neg = div.clamp_adjustment(-0.5);
        assert!(clamped_neg.is_some());
        assert!((clamped_neg.unwrap() - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_zero_lambda_rejects_all() {
        let div = PrunerDivergence {
            threshold_divergence: 0.0,
            topology_divergence: 0.0,
            lambda_t: 0.0,
        };
        // Any nonzero delta exceeds lambda_t=0
        assert!(div.clamp_adjustment(0.001).is_some());
        assert!(div.clamp_adjustment(0.0).is_none());
    }

    #[test]
    fn test_emit_diagnostic_interval() {
        let div = PrunerDivergence {
            threshold_divergence: 0.05,
            topology_divergence: 0.02,
            lambda_t: 0.1,
        };
        // Should not panic — just verifies the function runs
        div.emit_diagnostic("test_pruner", 100, 100);
        div.emit_diagnostic("test_pruner", 50, 100);
        div.emit_diagnostic("test_pruner", 100, 0);
    }
}
