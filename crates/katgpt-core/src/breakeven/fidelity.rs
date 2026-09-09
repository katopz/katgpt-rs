//! FidelityMatcher — Error-Matched KV Compression Level Selection
//!
//! Applies the paper's "error-matched classical solver" concept to KV cache
//! compression: for each sequence position, find the compression level that
//! produces the same perplexity as full attention (the "classical solver").
//!
//! This is Phase 3 of Plan 250. Stub implementation for now.

/// Compression level for KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CompressionLevel {
    /// No compression — full-precision KV cache.
    None = 0,
    /// 8-bit quantization.
    Bit8 = 1,
    /// 4-bit quantization.
    Bit4 = 2,
    /// 3-bit quantization.
    Bit3 = 3,
    /// 2-bit quantization.
    Bit2 = 4,
}

impl CompressionLevel {
    /// All levels from highest to lowest fidelity.
    pub const ALL: [Self; 5] = [Self::None, Self::Bit8, Self::Bit4, Self::Bit3, Self::Bit2];

    /// Bits per element.
    pub const fn bits(&self) -> u8 {
        match self {
            Self::None => 32,
            Self::Bit8 => 8,
            Self::Bit4 => 4,
            Self::Bit3 => 3,
            Self::Bit2 => 2,
        }
    }

    /// Compression ratio vs full precision.
    pub const fn compression_ratio(&self) -> f64 {
        32.0 / self.bits() as f64
    }
}

/// Per-position perplexity measurement.
/// Stored as cross-entropy (negative log-prob) — lower is better.
/// Perplexity = exp(cross_entropy).
#[derive(Clone, Debug, Default)]
pub struct CalibrationProfile {
    /// Per-position cross-entropy from full-attention forward pass.
    /// Position i → cross_entropy[i]. Empty means not yet calibrated.
    baseline_ce: Vec<f32>,
}

/// Per-compression-level perplexity delta profile.
#[derive(Clone, Debug, Default)]
pub struct CompressionSweep {
    /// deltas[level_index][position] = CE_delta vs baseline.
    /// level_index = CompressionLevel as u8 (0=None, 1=Bit8, etc.)
    /// Empty means not yet swept.
    deltas: Vec<Vec<f32>>,
}

/// Error-matched compression level selector.
///
/// Given a target perplexity delta (how much quality loss is acceptable),
/// returns the highest compression level that stays within the delta
/// at each sequence position.
///
/// Requires calibration (T13) and compression sweep (T14) data for
/// accurate level selection. Without calibration, returns safe defaults.
pub struct FidelityMatcher {
    /// Target perplexity delta (0.0 = lossless).
    target_delta: f64,
    /// T13: per-position baseline cross-entropy from full-attention forward pass.
    calibration: CalibrationProfile,
    /// T14: per-compression-level CE deltas vs baseline.
    sweep: CompressionSweep,
}

impl FidelityMatcher {
    /// Create with target perplexity delta.
    ///
    /// `target_delta` = acceptable increase in perplexity over full attention.
    /// 0.0 = lossless, 0.1 = ~10% perplexity increase acceptable.
    pub fn new(target_delta: f64) -> Self {
        Self {
            target_delta,
            calibration: CalibrationProfile::default(),
            sweep: CompressionSweep::default(),
        }
    }

    /// Whether calibration data is available.
    pub fn is_calibrated(&self) -> bool {
        !self.calibration.baseline_ce.is_empty()
    }

    /// T13: Record baseline cross-entropy for a single position.
    ///
    /// Call this after running forward() with full attention (CompressionLevel::None).
    /// `logits` = output logits from forward pass, `target_token` = ground-truth next token.
    pub fn record_baseline(&mut self, pos: usize, logits: &[f32], target_token: usize) {
        let ce = cross_entropy(logits, target_token);
        let buf = &mut self.calibration.baseline_ce;
        if pos >= buf.len() {
            buf.resize(pos + 1, 0.0f32);
        }
        buf[pos] = ce;
    }

    /// T13: Finalize baseline calibration (trim to actual length, validate).
    ///
    /// Returns number of positions calibrated.
    pub fn finalize_baseline(&mut self) -> usize {
        let buf = &mut self.calibration.baseline_ce;
        // Trim trailing zeros that were allocated but never recorded.
        while buf.last() == Some(&0.0f32) && !buf.is_empty() {
            buf.pop();
        }
        buf.len()
    }

    /// T14: Record compressed cross-entropy delta for a single position + level.
    ///
    /// `level` = compression level used, `logits` = output logits, `target_token` = ground-truth.
    /// Computes delta = compressed_CE - baseline_CE for this position.
    /// Does nothing if baseline is not available at this position.
    pub fn record_compressed(
        &mut self,
        pos: usize,
        level: CompressionLevel,
        logits: &[f32],
        target_token: usize,
    ) {
        let baseline_ce = match self.calibration.baseline_ce.get(pos) {
            Some(&ce) if ce != 0.0f32 => ce,
            _ => return, // No baseline at this position
        };

        let compressed_ce = cross_entropy(logits, target_token);
        let delta = compressed_ce - baseline_ce;

        let level_idx = level as usize;
        let deltas = &mut self.sweep.deltas;
        if deltas.is_empty() {
            deltas.resize_with(CompressionLevel::ALL.len(), Vec::new);
        }
        let level_buf = &mut deltas[level_idx];
        if pos >= level_buf.len() {
            level_buf.resize(pos + 1, 0.0f32);
        }
        level_buf[pos] = delta;
    }

    /// T14: Finalize compression sweep. Returns number of levels swept.
    pub fn finalize_sweep(&mut self) -> usize {
        let deltas = &mut self.sweep.deltas;
        // Trim trailing zeros from each level, then remove empty levels.
        for level_buf in deltas.iter_mut() {
            while level_buf.last() == Some(&0.0f32) && !level_buf.is_empty() {
                level_buf.pop();
            }
        }
        deltas.retain(|buf| !buf.is_empty());
        deltas.len()
    }

    /// Get the error-matched compression level for a given position.
    ///
    /// Decision logic:
    /// 1. If not calibrated → return `Bit4` (stub behavior)
    /// 2. If calibrated but no sweep data → return `Bit8` (safe default)
    /// 3. If calibrated + swept → find highest compression where `delta[pos] <= target_delta`.
    ///    Positions beyond calibration length extrapolate using the last known value.
    pub fn error_matched_level(&self, pos: usize) -> CompressionLevel {
        if !self.is_calibrated() {
            return CompressionLevel::Bit4;
        }

        let deltas = &self.sweep.deltas;
        if deltas.is_empty() {
            return CompressionLevel::Bit8;
        }

        // Clamp position to calibration length for extrapolation.
        let cal_len = self.calibration.baseline_ce.len();
        let effective_pos = if pos >= cal_len {
            cal_len.saturating_sub(1)
        } else {
            pos
        };

        // Find highest compression (lowest fidelity) where delta <= target_delta.
        // Iterate from highest compression (Bit2) down to None.
        for &level in CompressionLevel::ALL.iter().rev() {
            let level_idx = level as usize;
            let Some(level_deltas) = deltas.get(level_idx) else {
                continue;
            };
            let Some(&delta) = level_deltas.get(effective_pos) else {
                continue;
            };
            if delta as f64 <= self.target_delta {
                return level;
            }
        }

        // No compression level meets the delta target — return None (full precision).
        CompressionLevel::None
    }

    /// Target perplexity delta.
    #[inline]
    pub fn target_delta(&self) -> f64 {
        self.target_delta
    }

    /// Reference to calibration profile (for testing/inspection).
    pub fn calibration(&self) -> &CalibrationProfile {
        &self.calibration
    }

    /// Reference to compression sweep (for testing/inspection).
    pub fn sweep(&self) -> &CompressionSweep {
        &self.sweep
    }
}

impl CalibrationProfile {
    /// Baseline cross-entropy values.
    pub fn baseline_ce(&self) -> &[f32] {
        &self.baseline_ce
    }
}

impl CompressionSweep {
    /// Deltas for a specific compression level. Returns empty slice if not available.
    pub fn deltas_for_level(&self, level: CompressionLevel) -> &[f32] {
        self.deltas
            .get(level as usize)
            .map_or(&[], |d| d.as_slice())
    }
}

/// Numerically-stable cross-entropy computation.
/// logits = raw logit scores, target = index of ground-truth token.
/// Delegates the max-shift + logsumexp pass to the shared kernel
/// `simd::logsumexp_parts` (Issue 740 T1 factoring — one kernel shape shared
/// with the regime-probe conditional entropy). Arithmetic order is identical
/// to the former inline pass, so outputs are bit-identical (the kernel's third
/// return value, the softmax mean shift, is unused here).
fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let (max_val, ln_sum_exp, _) = crate::simd::logsumexp_parts(logits);
    -(logits[target] - max_val) + ln_sum_exp
}

impl Default for FidelityMatcher {
    fn default() -> Self {
        Self::new(0.05) // 5% perplexity increase acceptable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_level_bits() {
        assert_eq!(CompressionLevel::None.bits(), 32);
        assert_eq!(CompressionLevel::Bit8.bits(), 8);
        assert_eq!(CompressionLevel::Bit4.bits(), 4);
        assert_eq!(CompressionLevel::Bit3.bits(), 3);
        assert_eq!(CompressionLevel::Bit2.bits(), 2);
    }

    #[test]
    fn test_compression_ratio() {
        assert_eq!(CompressionLevel::None.compression_ratio(), 1.0);
        assert_eq!(CompressionLevel::Bit8.compression_ratio(), 4.0);
        assert_eq!(CompressionLevel::Bit4.compression_ratio(), 8.0);
        assert_eq!(CompressionLevel::Bit3.compression_ratio(), 32.0 / 3.0);
        assert_eq!(CompressionLevel::Bit2.compression_ratio(), 16.0);
    }

    #[test]
    fn test_fidelity_matcher_default() {
        let fm = FidelityMatcher::default();
        assert_eq!(fm.target_delta(), 0.05);
    }

    #[test]
    fn test_fidelity_matcher_returns_level() {
        let fm = FidelityMatcher::new(0.1);
        let level = fm.error_matched_level(0);
        assert_eq!(level, CompressionLevel::Bit4);
    }

    // --- T13: Calibration tests ---

    #[test]
    fn test_calibration_record_baseline() {
        let mut fm = FidelityMatcher::new(0.1);

        // Simulate logits for 3 positions (vocab size = 4 for simplicity).
        // Position 0: token 1 has highest logit → low CE
        let logits_0: Vec<f32> = vec![0.1, 2.0, 0.3, 0.1];
        // Position 1: token 2 has highest logit
        let logits_1: Vec<f32> = vec![0.5, 0.2, 3.0, 0.4];
        // Position 2: token 0 has highest logit
        let logits_2: Vec<f32> = vec![4.0, 0.1, 0.2, 0.3];

        fm.record_baseline(0, &logits_0, 1);
        fm.record_baseline(1, &logits_1, 2);
        fm.record_baseline(2, &logits_2, 0);

        assert!(fm.is_calibrated()); // Has baseline data after recording
        let ce = fm.calibration().baseline_ce();
        assert_eq!(ce.len(), 3);

        // CE should be low when target has highest logit.
        assert!(ce[0] > 0.0f32, "CE should be positive");
        assert!(ce[0] < 1.0f32, "CE should be low for confident prediction");
        assert!(ce[1] > 0.0f32 && ce[1] < 1.0f32);
        assert!(ce[2] > 0.0f32 && ce[2] < 1.0f32);
    }

    #[test]
    fn test_calibration_finalize_trims() {
        let mut fm = FidelityMatcher::new(0.1);

        // Record at position 0 and 2, leaving position 1 as gap (zero-filled).
        let logits_0: Vec<f32> = vec![0.1, 2.0, 0.3, 0.1];
        let logits_2: Vec<f32> = vec![4.0, 0.1, 0.2, 0.3];

        fm.record_baseline(0, &logits_0, 1);
        fm.record_baseline(2, &logits_2, 0);

        // Position 1 was zero-filled during resize.
        assert_eq!(fm.calibration().baseline_ce().len(), 3);

        // Finalize should trim trailing zeros.
        let count = fm.finalize_baseline();
        // After trimming, position 2 is the last non-zero, so length stays 3.
        // But position 1 is zero — finalize only trims trailing zeros.
        // Since pos 2 is non-zero, the trailing trim stops at pos 2.
        assert_eq!(count, 3);
        assert_eq!(fm.calibration().baseline_ce().len(), 3);
    }

    // --- T14: Compression sweep tests ---

    #[test]
    fn test_compression_sweep_record() {
        let mut fm = FidelityMatcher::new(0.5);

        // First, calibrate baseline for 3 positions.
        let logits_baseline: Vec<f32> = vec![0.1, 2.0, 0.3, 0.1];
        fm.record_baseline(0, &logits_baseline, 1);
        fm.record_baseline(1, &logits_baseline, 1);
        fm.record_baseline(2, &logits_baseline, 1);

        // Record compressed logits for Bit8 — slightly worse (lower target logit).
        let logits_bit8: Vec<f32> = vec![0.1, 1.5, 0.3, 0.1]; // target logit reduced
        fm.record_compressed(0, CompressionLevel::Bit8, &logits_bit8, 1);
        fm.record_compressed(1, CompressionLevel::Bit8, &logits_bit8, 1);
        fm.record_compressed(2, CompressionLevel::Bit8, &logits_bit8, 1);

        // Record compressed logits for Bit4 — even worse.
        let logits_bit4: Vec<f32> = vec![0.1, 1.0, 0.3, 0.1];
        fm.record_compressed(0, CompressionLevel::Bit4, &logits_bit4, 1);
        fm.record_compressed(1, CompressionLevel::Bit4, &logits_bit4, 1);
        fm.record_compressed(2, CompressionLevel::Bit4, &logits_bit4, 1);

        let sweep = fm.sweep();
        let bit8_deltas = sweep.deltas_for_level(CompressionLevel::Bit8);
        let bit4_deltas = sweep.deltas_for_level(CompressionLevel::Bit4);

        assert_eq!(bit8_deltas.len(), 3);
        assert_eq!(bit4_deltas.len(), 3);

        // Deltas should be positive (compressed CE > baseline CE).
        for i in 0..3 {
            assert!(
                bit8_deltas[i] > 0.0f32,
                "Bit8 delta at pos {i} should be positive"
            );
            assert!(
                bit4_deltas[i] > 0.0f32,
                "Bit4 delta at pos {i} should be positive"
            );
            // More compression → larger delta.
            assert!(
                bit4_deltas[i] > bit8_deltas[i],
                "Bit4 delta at pos {i} should exceed Bit8 delta"
            );
        }
    }

    // --- error_matched_level tests ---

    #[test]
    fn test_error_matched_level_uncalibrated() {
        let fm = FidelityMatcher::new(0.1);
        assert_eq!(fm.error_matched_level(0), CompressionLevel::Bit4);
        assert_eq!(fm.error_matched_level(100), CompressionLevel::Bit4);
    }

    #[test]
    fn test_error_matched_level_calibrated_no_sweep() {
        let mut fm = FidelityMatcher::new(0.1);
        let logits: Vec<f32> = vec![0.1, 2.0, 0.3, 0.1];
        fm.record_baseline(0, &logits, 1);
        fm.record_baseline(1, &logits, 1);
        fm.finalize_baseline();

        // Calibrated but no sweep → Bit8 safe default.
        assert_eq!(fm.error_matched_level(0), CompressionLevel::Bit8);
        assert_eq!(fm.error_matched_level(1), CompressionLevel::Bit8);
    }

    #[test]
    fn test_error_matched_level_uses_sweep_data() {
        let mut fm = FidelityMatcher::new(0.3); // 0.3 target delta

        // Calibrate baseline for 3 positions — confident predictions.
        let logits_baseline: Vec<f32> = vec![0.1, 2.0, 0.3, 0.1];
        for pos in 0..3usize {
            fm.record_baseline(pos, &logits_baseline, 1);
        }

        // Bit8: small delta (within 0.3)
        let logits_bit8: Vec<f32> = vec![0.1, 1.5, 0.3, 0.1];
        for pos in 0..3usize {
            fm.record_compressed(pos, CompressionLevel::Bit8, &logits_bit8, 1);
        }

        // Bit4: larger delta — check if it exceeds 0.3 or not.
        let logits_bit4: Vec<f32> = vec![0.1, 1.0, 0.3, 0.1];
        for pos in 0..3usize {
            fm.record_compressed(pos, CompressionLevel::Bit4, &logits_bit4, 1);
        }

        // Bit2: much larger delta (definitely exceeds 0.3)
        let logits_bit2: Vec<f32> = vec![0.1, 0.2, 0.3, 0.1];
        for pos in 0..3usize {
            fm.record_compressed(pos, CompressionLevel::Bit2, &logits_bit2, 1);
        }

        // Bit3: moderate delta
        let logits_bit3: Vec<f32> = vec![0.1, 0.7, 0.3, 0.1];
        for pos in 0..3usize {
            fm.record_compressed(pos, CompressionLevel::Bit3, &logits_bit3, 1);
        }

        let level = fm.error_matched_level(0);
        // Should return the highest compression level where delta <= 0.3.
        // The exact level depends on the cross-entropy deltas computed.
        // At minimum, Bit8 should always be selected since its delta is smallest.
        assert!(
            level >= CompressionLevel::Bit8,
            "Expected at least Bit8, got {level:?}"
        );
    }

    #[test]
    fn test_error_matched_level_extrapolates_beyond_calibrated() {
        let mut fm = FidelityMatcher::new(0.3);

        // Calibrate only position 0.
        let logits_baseline: Vec<f32> = vec![0.1, 2.0, 0.3, 0.1];
        fm.record_baseline(0, &logits_baseline, 1);

        // Bit8 with small delta.
        let logits_bit8: Vec<f32> = vec![0.1, 1.5, 0.3, 0.1];
        fm.record_compressed(0, CompressionLevel::Bit8, &logits_bit8, 1);

        fm.finalize_baseline();

        // Position 100 is beyond calibration length (1).
        // Should extrapolate using position 0's data.
        let level = fm.error_matched_level(100);
        assert_eq!(
            level,
            fm.error_matched_level(0),
            "Should extrapolate using last calibrated position"
        );
    }
}
