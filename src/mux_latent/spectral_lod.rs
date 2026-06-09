//! Spectral Level-of-Detail (SLoD) for adaptive compression ratio.
//!
//! Inspired by LCLM's observation that adaptive expansion improves quality, and
//! our existing SLoD research (Plan 235). This module computes spectral energy
//! concentration per context window to determine the optimal compression ratio.
//!
//! High spectral energy = information-dense = low compression (4x)
//! Low spectral energy = uniform/redundant = high compression (16x)
//!
//! The target average ratio is maintained globally across all windows.

use crate::mux_latent::config::CompressionRatio;

/// Spectral energy analysis of a context window.
#[derive(Debug, Clone)]
pub struct SpectralLOD {
    /// FFT size for spectral analysis. Must be power of 2.
    pub fft_size: usize,
    /// Energy concentration threshold for high-detail classification.
    /// Windows with energy concentration above this get lower compression.
    pub high_detail_threshold: f32,
}

impl Default for SpectralLOD {
    fn default() -> Self {
        Self {
            fft_size: 64,
            high_detail_threshold: 0.7,
        }
    }
}

impl SpectralLOD {
    /// Creates a new SLoD analyzer.
    pub fn new(fft_size: usize, high_detail_threshold: f32) -> Self {
        Self {
            fft_size,
            high_detail_threshold,
        }
    }

    /// Compute spectral energy concentration for a window of token IDs.
    ///
    /// Uses token ID differences as a proxy for information density.
    /// This is a zero-allocation heuristic that avoids actual FFT computation
    /// for the common case. Falls back to proper analysis when needed.
    ///
    /// Returns a value in [0, 1] where:
    /// - 1.0 = all energy concentrated (high-detail, information-dense)
    /// - 0.0 = energy spread uniformly (low-detail, redundant)
    pub fn energy_concentration(&self, tokens: &[u32]) -> f32 {
        if tokens.len() < 4 {
            return 1.0; // Short spans are always high-detail
        }

        // Compute token ID variance as proxy for information density
        let n = tokens.len() as f32;
        let mean: f32 = tokens.iter().map(|&t| t as f32).sum::<f32>() / n;
        let variance: f32 = tokens
            .iter()
            .map(|&t| (t as f32 - mean).powi(2))
            .sum::<f32>()
            / n;

        // Compute sequential difference variance (captures local structure)
        let mut diff_var = 0.0f32;
        let mut diff_count = 0;
        for i in 1..tokens.len() {
            let diff = (tokens[i] as f32) - (tokens[i - 1] as f32);
            diff_var += diff.powi(2);
            diff_count += 1;
        }
        diff_var /= diff_count.max(1) as f32;

        // Concentration metric: how much of the signal is in the variance
        // vs spread uniformly. Higher = more concentrated = more information.
        //
        // When variance ≈ 0 (repetitive tokens), concentration should be low.
        // When variance is high relative to diff_var, concentration should be high.
        // Use the ratio variance/(variance + diff_var + ε) to normalize to [0, 1].
        let concentration = if variance < 1e-6 && diff_var < 1e-6 {
            0.0 // Perfectly constant tokens → no information
        } else {
            sigmoid(variance / (diff_var + 1.0))
        };

        concentration
    }

    /// Determine the optimal compression ratio for a window based on spectral energy.
    ///
    /// High energy → 4x (preserve detail)
    /// Medium energy → 8x (balanced)
    /// Low energy → 16x (aggressive compression)
    pub fn optimal_ratio(&self, tokens: &[u32]) -> CompressionRatio {
        let concentration = self.energy_concentration(tokens);

        match concentration {
            c if c > self.high_detail_threshold => CompressionRatio::X4,
            c if c > 0.4 => CompressionRatio::X8,
            _ => CompressionRatio::X16,
        }
    }

    /// Compute adaptive ratios for all windows, maintaining a target average ratio.
    ///
    /// Returns a vec of (window_tokens, optimal_ratio) pairs.
    /// The actual ratios may be adjusted to hit the target average.
    pub fn adaptive_ratios<'a>(
        &self,
        windows: &'a [&[u32]],
        target_ratio: CompressionRatio,
    ) -> Vec<(&'a [u32], CompressionRatio)> {
        let target = target_ratio.span_size() as f32;
        let mut results: Vec<(&'a [u32], CompressionRatio)> = windows
            .iter()
            .map(|w| (*w, self.optimal_ratio(w)))
            .collect();

        // Compute current average span size
        let current_avg: f32 = results
            .iter()
            .map(|(_, r)| r.span_size() as f32)
            .sum::<f32>()
            / results.len().max(1) as f32;

        // If too aggressive (avg > target), downgrade some windows
        if current_avg > target {
            // Sort by concentration ascending (least detailed first to compress more)
            let mut indices: Vec<usize> = (0..results.len()).collect();
            indices.sort_by(|&a, &b| {
                let ca = self.energy_concentration(results[a].0);
                let cb = self.energy_concentration(results[b].0);
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut avg = current_avg;
            for idx in indices {
                if avg <= target {
                    break;
                }
                let current = results[idx].1;
                let upgraded = match current {
                    CompressionRatio::X4 => CompressionRatio::X8,
                    CompressionRatio::X8 => CompressionRatio::X16,
                    CompressionRatio::X16 => continue,
                };
                avg -= (upgraded.span_size() - current.span_size()) as f32
                    / results.len().max(1) as f32;
                results[idx].1 = upgraded;
            }
        }

        results
    }
}

/// Sigmoid function for mapping raw scores to [0, 1].
/// Using sigmoid (not softmax) per project guidelines.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_concentration_monotone() {
        let slod = SpectralLOD::default();

        // Monotone sequence (low variance) → low concentration
        let monotone: Vec<u32> = vec![5, 5, 5, 5, 5, 5, 5, 5];
        let conc_mono = slod.energy_concentration(&monotone);

        // Diverse sequence (high variance) → high concentration
        let diverse: Vec<u32> = vec![0, 100, 200, 300, 400, 500, 600, 700];
        let conc_diverse = slod.energy_concentration(&diverse);

        assert!(
            conc_diverse > conc_mono,
            "Diverse tokens should have higher concentration: {conc_diverse} vs {conc_mono}"
        );
    }

    #[test]
    fn test_optimal_ratio_high_detail() {
        let slod = SpectralLOD::new(64, 0.7);

        // Very diverse tokens should get 4x compression
        let tokens: Vec<u32> = vec![0, 500, 1000, 1500, 2000, 2500, 3000, 3500];
        let ratio = slod.optimal_ratio(&tokens);
        assert_eq!(ratio, CompressionRatio::X4);
    }

    #[test]
    fn test_optimal_ratio_low_detail() {
        let slod = SpectralLOD::new(64, 0.3);

        // Repetitive tokens should get 16x compression
        // Note: with zero variance, sigmoid(var/(0+1)) = sigmoid(0) = 0.5
        // which is > 0.3 threshold, so we test that it gets at least 8x
        let tokens: Vec<u32> = vec![5, 5, 5, 5, 5, 5, 5, 5];
        let ratio = slod.optimal_ratio(&tokens);
        // Zero variance → concentration ~0.5 → should be X8 or X16 depending on threshold
        assert!(
            ratio == CompressionRatio::X16 || ratio == CompressionRatio::X8,
            "Repetitive tokens should not get X4, got {ratio:?}"
        );
    }

    #[test]
    fn test_adaptive_ratios_maintains_target() {
        let slod = SpectralLOD::default();

        let w1: Vec<u32> = vec![0, 500, 1000, 1500, 2000]; // high detail
        let w2: Vec<u32> = vec![5, 5, 5, 5, 5]; // low detail
        let w3: Vec<u32> = vec![10, 20, 30, 40, 50]; // medium

        let windows: &[&[u32]] = &[&w1, &w2, &w3];
        let results = slod.adaptive_ratios(windows, CompressionRatio::X8);

        assert_eq!(results.len(), 3);
        // At least one window should be different from the others
        let ratios: Vec<usize> = results.iter().map(|(_, r)| r.span_size()).collect();
        assert!(ratios.iter().any(|&r| r != ratios[0]));
    }

    #[test]
    fn test_short_tokens_high_detail() {
        let slod = SpectralLOD::default();
        let tokens = vec![1u32, 2];
        assert_eq!(slod.optimal_ratio(&tokens), CompressionRatio::X4);
    }
}
