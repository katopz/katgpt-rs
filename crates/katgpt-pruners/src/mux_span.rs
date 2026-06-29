//! MuxSpanPruner — ConstraintPruner operating in vocabulary simplex space (Research 158, MUX).
//!
//! Detects "valid multiplexed spans" in logit distributions by checking whether the top-k
//! mass forms a geometric decay pattern indicative of latent superposition. Prunes branches
//! where the distribution indicates latent collapse: either diffuse noise (no clear peaks)
//! or single-token collapse (one dominant peak with no geometric tail).
//!
//! # Architecture
//!
//! The `ConstraintPruner` trait operates on `(depth, token_idx, parent_tokens)` — indices only,
//! no logit information. `MuxSpanPruner` therefore provides:
//! - `is_valid` — trivial pass-through (indices alone cannot indicate superposition)
//! - `is_valid_logits` — the real decision boundary using raw logit vectors
//!
//! # Geometric Decay Detection
//!
//! A valid multiplexed span has top-k logits that decay approximately geometrically:
//! `logit[i] ≈ logit[0] * decay^i`. The pruner checks:
//! 1. **Separation**: top-k peaks are well-separated from the remaining mass
//! 2. **Decay ordering**: consecutive peaks follow the decay ratio within tolerance
//! 3. **Collapse rejection**: rejects diffuse (flat) or collapsed (single-peak) distributions

use katgpt_speculative::ConstraintPruner;

// ── MuxSpanPruner ──────────────────────────────────────────────────

/// Pruner that validates multiplexed token spans from logit distributions.
///
/// Detects geometric decay patterns among top-k logit peaks. Branches where
/// the distribution indicates latent collapse (noise or single-token) are pruned.
///
/// (Research 158, MUX)
pub struct MuxSpanPruner {
    /// Geometric decay ratio expected between consecutive top-k peaks.
    /// Default: 0.9 — each successive peak is ~90% of the previous.
    pub decay: f32,
    /// Number of top peaks to consider as the "multiplexed span".
    /// Default: 5.
    pub span_k: usize,
    /// Minimum separation between the k-th peak and the (k+1)-th token's logit.
    /// Ensures the top-k span is well-separated from the background.
    /// Default: 0.3.
    pub separation_threshold: f32,
}

impl MuxSpanPruner {
    /// Create a new `MuxSpanPruner` with default parameters.
    pub fn new() -> Self {
        Self {
            decay: 0.9,
            span_k: 5,
            separation_threshold: 0.3,
        }
    }

    /// Create a builder with custom parameters.
    pub fn with_params(decay: f32, span_k: usize, separation_threshold: f32) -> Self {
        Self {
            decay,
            span_k,
            separation_threshold,
        }
    }

    /// Validate whether a logit distribution represents a valid multiplexed span.
    ///
    /// Returns `true` if:
    /// - The top-k peaks are well-separated from the background (`separation_threshold`)
    /// - The peaks follow approximate geometric decay within tolerance
    /// - The distribution is neither collapsed (single dominant peak) nor diffuse (flat)
    pub fn is_valid_logits(&self, logits: &[f32]) -> bool {
        if logits.is_empty() || self.span_k == 0 {
            return false;
        }

        let k = self.span_k.min(logits.len());
        let peaks = Self::extract_top_k_peaks(logits, k);

        // Need at least 2 peaks for a meaningful superposition
        if peaks.len() < 2 {
            return false;
        }

        let top_val = peaks[0].1;

        // Check separation: k-th peak must be well above the next token's logit
        if k < logits.len() {
            // Find the (k+1)-th largest logit
            let kth_peak_val = peaks.last().map(|&(_, v)| v).unwrap_or(f32::NEG_INFINITY);
            // The background level is estimated as the mean of logits outside the top-k
            let bg_level = mean_of_remaining(logits, &peaks);
            if kth_peak_val - bg_level < self.separation_threshold {
                return false;
            }
        }

        // Check geometric decay ordering
        for (i, &(_, actual)) in peaks.iter().enumerate().skip(1) {
            let expected = top_val * self.decay.powi(i as i32);
            // Allow 50% tolerance on the decay ratio
            let tolerance = expected.abs() * 0.5;
            if (actual - expected).abs() > tolerance {
                return false;
            }
        }

        // Reject collapsed distribution: dominant peak > 90% of mass concentration
        // Measured by ratio of top-1 to top-2
        if peaks.len() >= 2 {
            let ratio = if peaks[1].1.abs() > 1e-8 {
                peaks[0].1.abs() / peaks[1].1.abs()
            } else {
                f32::INFINITY
            };
            // If top peak is > 20x the second, it's a collapse, not superposition
            if ratio > 20.0 {
                return false;
            }
        }

        true
    }

    /// Extract the top-k peaks (token_id, logit_value) from a logit vector,
    /// sorted by descending logit value.
    ///
    /// Uses partial selection sort — O(k * n) which is optimal for small k.
    #[inline]
    pub fn extract_top_k_peaks(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
        let k = k.min(logits.len());
        if k == 0 {
            return Vec::new();
        }

        // Partial selection: maintain top-k via insertion
        let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);

        for (idx, &val) in logits.iter().enumerate() {
            if top.len() < k {
                insert_sorted(&mut top, idx, val);
            } else if val > top.last().unwrap().1 {
                // Replace the smallest entry
                let last = top.len() - 1;
                top[last] = (idx, val);
                // Re-sort from the replaced position
                bubble_up(&mut top, last);
            }
        }

        top
    }
}

impl Default for MuxSpanPruner {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintPruner for MuxSpanPruner {
    /// Pass-through: token indices alone don't carry logit information.
    /// Use [`MuxSpanPruner::is_valid_logits`] for the actual superposition check.
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        true
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Insert (idx, val) into a sorted-by-descending-value vec.
#[inline]
fn insert_sorted(buf: &mut Vec<(usize, f32)>, idx: usize, val: f32) {
    let pos = buf.partition_point(|&(_, v)| v >= val);
    buf.insert(pos, (idx, val));
}

/// Bubble the entry at `pos` up to its correct position in descending order.
#[inline]
fn bubble_up(buf: &mut [(usize, f32)], mut pos: usize) {
    while pos > 0 && buf[pos].1 > buf[pos - 1].1 {
        buf.swap(pos, pos - 1);
        pos -= 1;
    }
}

/// Compute the mean logit value of tokens NOT in the top-k peaks.
fn mean_of_remaining(logits: &[f32], peaks: &[(usize, f32)]) -> f32 {
    let full_len = logits.len();
    if peaks.len() >= full_len {
        return f32::NEG_INFINITY;
    }

    let peak_set: Vec<bool> = {
        let mut s = vec![false; full_len];
        for &(idx, _) in peaks {
            if idx < full_len {
                s[idx] = true;
            }
        }
        s
    };

    let (sum, count) = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| !peak_set[*i])
        .fold((0.0f32, 0usize), |(s, c), (_, &v)| (s + v, c + 1));

    if count == 0 {
        f32::NEG_INFINITY
    } else {
        sum / count as f32
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn geometric_logits(vocab_size: usize, k: usize, decay: f32) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab_size];
        // Place top-k with geometric decay at specific token positions
        for i in 0..k {
            logits[10 + i * 7] = 10.0 * decay.powi(i as i32);
        }
        logits
    }

    #[test]
    fn test_mux_span_pruner_accepts_valid_superposition() {
        let pruner = MuxSpanPruner::with_params(0.9, 5, 0.3);
        let logits = geometric_logits(100, 5, 0.9);
        assert!(
            pruner.is_valid_logits(&logits),
            "valid geometric superposition should be accepted"
        );
    }

    #[test]
    fn test_mux_span_pruner_rejects_collapsed() {
        let pruner = MuxSpanPruner::with_params(0.9, 5, 0.3);
        // Single-token collapse: one huge peak, everything else near zero
        let mut logits = vec![0.01f32; 100];
        logits[42] = 100.0; // dominant peak
        logits[1] = 0.5; // second peak is negligible
        assert!(
            !pruner.is_valid_logits(&logits),
            "collapsed single-peak distribution should be rejected"
        );
    }

    #[test]
    fn test_mux_span_pruner_rejects_diffuse_noise() {
        let pruner = MuxSpanPruner::with_params(0.9, 5, 0.3);
        // Flat/uniform distribution — no clear peaks
        let logits = vec![1.0f32; 100];
        assert!(
            !pruner.is_valid_logits(&logits),
            "diffuse uniform distribution should be rejected"
        );
    }

    #[test]
    fn test_extract_top_k_peaks_ordering() {
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.7];
        let peaks = MuxSpanPruner::extract_top_k_peaks(&logits, 3);
        assert_eq!(peaks.len(), 3);
        assert_eq!(peaks[0], (3, 0.9));
        assert_eq!(peaks[1], (5, 0.7));
        assert_eq!(peaks[2], (1, 0.5));
    }

    #[test]
    fn test_extract_top_k_peaks_empty() {
        let peaks = MuxSpanPruner::extract_top_k_peaks(&[], 5);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_extract_top_k_peaks_k_exceeds_len() {
        let logits = vec![1.0, 2.0];
        let peaks = MuxSpanPruner::extract_top_k_peaks(&logits, 10);
        assert_eq!(peaks.len(), 2);
    }

    #[test]
    fn test_mux_span_pruner_rejects_too_few_peaks() {
        let pruner = MuxSpanPruner::with_params(0.9, 5, 0.3);
        // Only one non-zero entry
        let mut logits = vec![0.0f32; 100];
        logits[0] = 5.0;
        assert!(
            !pruner.is_valid_logits(&logits),
            "single peak should be rejected (need >= 2 for superposition)"
        );
    }

    #[test]
    fn test_mux_span_constraint_pruner_passthrough() {
        let pruner = MuxSpanPruner::new();
        // ConstraintPruner::is_valid always returns true (pass-through)
        assert!(pruner.is_valid(0, 42, &[]));
        assert!(pruner.is_valid(5, 0, &[1, 2, 3]));
    }

    #[test]
    fn test_mux_span_pruner_default() {
        let pruner = MuxSpanPruner::default();
        assert!((pruner.decay - 0.9).abs() < 1e-6);
        assert_eq!(pruner.span_k, 5);
        assert!((pruner.separation_threshold - 0.3).abs() < 1e-6);
    }
}
