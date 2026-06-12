//! `MuxSpanPruner` — checks logit distribution shape for valid superposition.
//!
//! A valid superposition exhibits geometric decay among the top-K peaks:
//! each successive peak must be no larger than `decay_rate` times the previous.

use crate::mux::top_k::{MAX_TOP_K, extract_top_k_into};

/// Minimum ratio between consecutive peaks for geometric decay.
const DEFAULT_DECAY_RATE: f32 = 0.5;

/// Minimum number of peaks required for a valid superposition span.
const MIN_PEAKS: usize = 2;

/// Pruner that validates whether a logit distribution supports
/// a valid superposition span at a given tree depth.
#[derive(Debug, Clone)]
pub struct MuxSpanPruner {
    /// Number of top-K peaks to inspect.
    pub k: usize,
    /// Geometric decay threshold: peak[i+1] / peak[i] >= decay_rate.
    pub decay_rate: f32,
}

impl MuxSpanPruner {
    pub fn new(k: usize, decay_rate: f32) -> Self {
        Self { k, decay_rate }
    }

    /// Returns `true` if the logit distribution exhibits geometric decay
    /// among its top-K peaks, indicating a valid superposition.
    /// Zero-alloc: uses stack buffer for top-K extraction.
    pub fn is_valid(&self, logits: &[f32], _depth: usize) -> bool {
        let mut buf = [0.0f32; MAX_TOP_K];
        let peaks = extract_top_k_into(logits, self.k, &mut buf);
        if peaks.len() < MIN_PEAKS {
            return false;
        }
        // Manual indexed loop — avoids windows(2) iterator overhead for tiny k (≤16).
        for i in 0..peaks.len() - 1 {
            if peaks[i + 1] < peaks[i] * self.decay_rate {
                return false;
            }
        }
        true
    }
}

impl Default for MuxSpanPruner {
    fn default() -> Self {
        Self::new(4, DEFAULT_DECAY_RATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_decay_is_valid() {
        let pruner = MuxSpanPruner::new(4, 0.5);
        // Peaks: 1.0, 0.7, 0.5, 0.3 — each >= 0.5 * previous
        let logits = vec![0.1, 1.0, 0.2, 0.7, 0.05, 0.5, 0.0, 0.3];
        assert!(pruner.is_valid(&logits, 0));
    }

    #[test]
    fn single_peak_is_invalid() {
        let pruner = MuxSpanPruner::new(4, 0.5);
        let logits = vec![0.0, 5.0, 0.0, 0.0];
        assert!(!pruner.is_valid(&logits, 0));
    }

    #[test]
    fn sharp_drop_is_invalid() {
        let pruner = MuxSpanPruner::new(4, 0.5);
        // Peaks: 1.0, 0.1 — second is < 0.5 * first
        let logits = vec![1.0, 0.1, 0.0, 0.0];
        assert!(!pruner.is_valid(&logits, 0));
    }
}
