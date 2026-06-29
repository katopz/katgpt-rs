//! MuxBfs — Superposition-guided dynamic width for DDTree BFS (Research 158, MUX).
//!
//! At each DDTree depth, reads the logit distribution shape. If it forms a valid
//! superposition (multiple peaks with geometric decay), ALL peaks become parallel BFS
//! branches. If peaked (one dominant token), falls back to narrow expansion.
//! This produces dynamic-width tree search where tree width adapts per-depth — no training needed.

use super::mux_span::MuxSpanPruner;

/// Dynamic-width BFS expansion strategy guided by logit superposition shape.
///
/// Peaked distributions → narrow (width=1).
/// Multi-peak distributions → wide (expand all peaks up to max_width).
pub struct MuxBfs {
    /// Maximum allowed width at any depth.
    pub max_width: usize,
    /// Geometric decay ratio for peak detection.
    pub decay: f32,
    /// Separation threshold for peak validity.
    pub separation: f32,
}

impl MuxBfs {
    pub fn new(max_width: usize, decay: f32, separation: f32) -> Self {
        Self {
            max_width,
            decay,
            separation,
        }
    }

    /// Detect the number of valid branches to expand from a logit vector.
    ///
    /// Returns the number of valid superposition peaks (clamped to max_width).
    /// If the distribution is peaked (single dominant token), returns 1.
    /// If it's a valid multi-peak superposition, returns the peak count.
    /// If it's noise (no valid superposition), returns 0.
    pub fn detect_width(&self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 0;
        }

        let k = self.max_width.min(logits.len());
        let peaks = MuxSpanPruner::extract_top_k_peaks(logits, k);

        if peaks.is_empty() {
            return 0;
        }

        // Single peak → peaked distribution, narrow expansion
        if peaks.len() == 1 {
            return 1;
        }

        let top_val = peaks[0].1;

        // Check for collapsed distribution: top peak dominates (ratio > 20x)
        if peaks.len() >= 2 {
            let ratio = if peaks[1].1.abs() > 1e-8 {
                peaks[0].1.abs() / peaks[1].1.abs()
            } else {
                f32::INFINITY
            };
            if ratio > 20.0 {
                // Peaked — single dominant token
                return 1;
            }
        }

        // Find the number of peaks that follow geometric decay.
        // Truncate at the first peak that deviates from the expected decay pattern.
        let mut valid_count = peaks.len();
        for (i, &(_, actual)) in peaks.iter().enumerate().skip(1) {
            let expected = top_val * self.decay.powi(i as i32);
            let tolerance = expected.abs() * 0.5;
            if (actual - expected).abs() > tolerance {
                // This peak (and all after) don't follow geometric decay
                valid_count = i;
                break;
            }
        }

        // valid_count = 0 means no valid peaks at all → noise
        if valid_count == 0 {
            return 0;
        }

        // If only 1 peak survives decay check, it's peaked
        if valid_count == 1 {
            return 1;
        }

        // Check separation using only the valid peaks
        if valid_count < logits.len() {
            let valid_peaks: Vec<(usize, f32)> =
                peaks.iter().take(valid_count).copied().collect();
            let bg_level = mean_of_remaining(logits, &valid_peaks);
            let last_val = valid_peaks.last().map(|&(_, v)| v).unwrap_or(f32::NEG_INFINITY);
            if last_val - bg_level < self.separation {
                // No clear separation from background → noise
                return 0;
            }
        }

        valid_count.min(self.max_width)
    }

    /// Extract branch expansions from a logit vector.
    ///
    /// Returns a list of (token_id, logit_value) pairs representing the
    /// branches to expand in parallel. Empty if the distribution is invalid.
    pub fn extract_branches(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let width = self.detect_width(logits);
        if width == 0 {
            return Vec::new();
        }

        let peaks = MuxSpanPruner::extract_top_k_peaks(logits, width);
        peaks.into_iter().take(width).collect()
    }
}

impl Default for MuxBfs {
    fn default() -> Self {
        Self::new(8, 0.9, 0.3)
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

    /// Generate logit vector with geometric decay at specific positions.
    fn geometric_logits(vocab_size: usize, k: usize, decay: f32) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab_size];
        for i in 0..k {
            logits[10 + i * 7] = 10.0 * decay.powi(i as i32);
        }
        logits
    }

    #[test]
    fn test_mux_bfs_peaked_narrow() {
        // Single dominant token — should return width=1
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        let mut logits = vec![0.01f32; 100];
        logits[42] = 100.0; // dominant peak
        logits[1] = 0.5; // second peak negligible

        let width = bfs.detect_width(&logits);
        assert_eq!(width, 1, "peaked distribution should produce width=1");
    }

    #[test]
    fn test_mux_bfs_multi_peak_wide() {
        // Multiple valid peaks with geometric decay → wide expansion
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        let logits = geometric_logits(100, 5, 0.9);

        let width = bfs.detect_width(&logits);
        assert_eq!(
            width, 5,
            "valid multi-peak superposition should expand all 5 peaks"
        );
    }

    #[test]
    fn test_mux_bfs_noise_zero() {
        // Uniform noise — no valid superposition → width=0
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        let logits = vec![1.0f32; 100];

        let width = bfs.detect_width(&logits);
        assert_eq!(width, 0, "uniform noise should produce width=0");
    }

    #[test]
    fn test_mux_bfs_clamped_max_width() {
        // Peaks exceed max_width → should be clamped
        let bfs = MuxBfs::new(3, 0.9, 0.3);
        let logits = geometric_logits(100, 5, 0.9);

        let width = bfs.detect_width(&logits);
        assert_eq!(
            width, 3,
            "width should be clamped to max_width when peaks exceed it"
        );
    }

    #[test]
    fn test_mux_bfs_extract_branches() {
        // Verify that extract_branches returns the correct (token_id, logit) pairs
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        let logits = geometric_logits(100, 3, 0.9);

        let branches = bfs.extract_branches(&logits);
        assert_eq!(branches.len(), 3, "should extract 3 branches");

        // First branch should be the highest peak
        assert_eq!(
            branches[0].0, 10,
            "first branch token_id should be at position 10"
        );
        assert!(
            (branches[0].1 - 10.0).abs() < 1e-4,
            "first branch logit should be ~10.0"
        );

        // Second branch: position 17, value ~9.0
        assert_eq!(
            branches[1].0, 17,
            "second branch token_id should be at position 17"
        );
        assert!(
            (branches[1].1 - 9.0).abs() < 1e-4,
            "second branch logit should be ~9.0"
        );

        // Third branch: position 24, value ~8.1
        assert_eq!(
            branches[2].0, 24,
            "third branch token_id should be at position 24"
        );
        assert!(
            (branches[2].1 - 8.1).abs() < 0.5,
            "third branch logit should be ~8.1"
        );
    }

    #[test]
    fn test_mux_bfs_extract_branches_noise_empty() {
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        let logits = vec![1.0f32; 100];

        let branches = bfs.extract_branches(&logits);
        assert!(
            branches.is_empty(),
            "noise distribution should produce no branches"
        );
    }

    #[test]
    fn test_mux_bfs_empty_logits() {
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        assert_eq!(bfs.detect_width(&[]), 0, "empty logits → width=0");
        assert!(
            bfs.extract_branches(&[]).is_empty(),
            "empty logits → no branches"
        );
    }

    #[test]
    fn test_mux_bfs_single_nonzero_peak() {
        // Only one non-zero logit entry — should be treated as peaked
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        let mut logits = vec![0.0f32; 100];
        logits[42] = 5.0;

        let width = bfs.detect_width(&logits);
        assert_eq!(width, 1, "single non-zero entry should produce width=1");
    }

    #[test]
    fn test_mux_bfs_default() {
        let bfs = MuxBfs::default();
        assert_eq!(bfs.max_width, 8);
        assert!((bfs.decay - 0.9).abs() < 1e-6);
        assert!((bfs.separation - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_mux_bfs_peaked_with_two_peaks_high_ratio() {
        // Two peaks but ratio > 20x → still peaked (width=1)
        let bfs = MuxBfs::new(8, 0.9, 0.3);
        let mut logits = vec![0.01f32; 100];
        logits[0] = 50.0;
        logits[1] = 1.0; // ratio = 50x > 20x

        let width = bfs.detect_width(&logits);
        assert_eq!(width, 1, "high ratio (>20x) should be peaked");
    }
}
