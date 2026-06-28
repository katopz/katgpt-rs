//! Computational Irreducibility Gate — Kolmogorov complexity proxy via compression ratio.
//!
//! Wolfram's key finding: winning strategies are simple but can only be found by
//! running the game. This gate detects when a game IS predictable (low irreducibility)
//! and allows skipping expensive simulation.
//!
//! Uses run-length encoding (RLE) as a cheap compression proxy for Kolmogorov complexity.
//! If the win matrix compresses well (low ratio), the game has predictable structure
//! and analytical shortcuts may exist. If the ratio is high, full simulation is required.
//!
//! Plan 188 Phase 4.

use crate::ruliology::types::WinMatrix;

// ── IrreducibilityResult ──────────────────────────────────────

/// Result of irreducibility analysis.
#[derive(Debug, Clone)]
pub struct IrreducibilityResult {
    /// Compression ratio of the win matrix (0.0 = fully compressible, 1.0 = fully random).
    /// Ratio = compressed_size / raw_size.
    pub compression_ratio: f32,
    /// Whether the game is considered irreducible (ratio above threshold).
    pub is_irreducible: bool,
    /// Mean absolute payoff in the matrix (indicator of game dynamics).
    pub mean_abs_payoff: f64,
    /// Payoff variance (high variance = complex dynamics).
    pub payoff_variance: f64,
}

// ── IrreducibilityGate ────────────────────────────────────────

/// Gate that determines if a game/strategy space is computationally irreducible.
///
/// Uses a simple run-length encoding (RLE) compression as a Kolmogorov complexity
/// proxy. If the win matrix compresses well (low ratio), the game has predictable
/// structure and analytical shortcuts may exist.
///
/// Threshold: compression_ratio > 0.7 → irreducible (must simulate)
///            compression_ratio ≤ 0.7 → reducible (shortcuts possible)
pub struct IrreducibilityGate {
    /// Compression ratio threshold above which we consider the game irreducible.
    pub threshold: f32,
}

impl Default for IrreducibilityGate {
    /// Default gate with 0.7 threshold.
    #[inline]
    fn default() -> Self {
        Self::new(0.7)
    }
}

impl IrreducibilityGate {
    /// Create a new gate with the given compression ratio threshold.
    #[inline]
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Analyze a win matrix for irreducibility.
    ///
    /// Returns compression ratio, irreducibility verdict, and payoff statistics.
    ///
    /// Uses Shannon entropy of the quantized byte distribution as the primary
    /// Kolmogorov complexity proxy. Low entropy = low complexity = reducible.
    /// For high-entropy matrices, falls back to RLE compression ratio.
    pub fn analyze(&self, matrix: &WinMatrix) -> IrreducibilityResult {
        let raw = self.quantize_matrix(matrix);

        // Compute byte frequency distribution.
        let mut freq = [0u32; 256];
        let total = raw.len() as u32;
        for &b in &raw {
            freq[b as usize] += 1;
        }

        // Shannon entropy of byte distribution (bits).
        let entropy = if total == 0 {
            0.0f32
        } else {
            let mut h = 0.0f32;
            for &count in &freq {
                if count > 0 {
                    let p = count as f32 / total as f32;
                    h -= p * p.log2();
                }
            }
            h
        };

        // Normalized entropy: 0.0 = all same byte, 1.0 = uniform distribution.
        // Max entropy for byte data = 8 bits.
        let normalized_entropy = entropy / 8.0;

        // RLE compression ratio as secondary signal.
        let compressed = rle_compress(&raw);
        let rle_ratio = if raw.is_empty() {
            0.0
        } else {
            compressed.len() as f32 / raw.len() as f32
        };

        // Effective compression ratio: use entropy when it's low (structured data),
        // RLE when entropy is high (potentially compressible despite high entropy).
        let compression_ratio = if entropy < 4.0 {
            // Low entropy → highly structured, use normalized entropy as ratio.
            normalized_entropy
        } else if rle_ratio < normalized_entropy {
            // High entropy but RLE compresses → some structure exists.
            rle_ratio
        } else {
            // High entropy, RLE doesn't help → likely irreducible.
            normalized_entropy
        };

        let (mean, variance) = self.payoff_stats(matrix);

        IrreducibilityResult {
            compression_ratio,
            is_irreducible: compression_ratio > self.threshold,
            mean_abs_payoff: mean,
            payoff_variance: variance,
        }
    }

    /// Quick check: is the game irreducible?
    pub fn is_irreducible(&self, matrix: &WinMatrix) -> bool {
        self.analyze(matrix).is_irreducible
    }

    /// Quantize payoff matrix to bytes for compression.
    /// Maps [-1, 1] → [0, 255].
    fn quantize_matrix(&self, matrix: &WinMatrix) -> Vec<u8> {
        let n = matrix.payoffs.len();
        let mut result = Vec::with_capacity(n * n);
        for row in &matrix.payoffs {
            for &val in row {
                // Map [-1, 1] → [0, 255]
                let normalized = ((val + 1.0) * 127.5).clamp(0.0, 255.0);
                result.push(normalized as u8);
            }
        }
        result
    }

    /// Compute mean absolute payoff and variance.
    fn payoff_stats(&self, matrix: &WinMatrix) -> (f64, f64) {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;

        for row in &matrix.payoffs {
            for &val in row {
                let abs_val = val.abs();
                sum += abs_val;
                sum_sq += abs_val * abs_val;
                count += 1;
            }
        }

        if count == 0 {
            return (0.0, 0.0);
        }

        let mean = sum / count as f64;
        let variance = sum_sq / count as f64 - mean * mean;
        (mean, variance.max(0.0)) // numerical guard
    }
}

// ── RLE Compression ───────────────────────────────────────────

/// Simple run-length encoding compression.
/// Returns (value, count) pairs as a flat byte sequence.
fn rle_compress(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(data.len());
    let mut current = data[0];
    let mut count: u8 = 1;

    for &byte in &data[1..] {
        if byte == current && count < 255 {
            count += 1;
        } else {
            result.push(current);
            result.push(count);
            current = byte;
            count = 1;
        }
    }
    result.push(current);
    result.push(count);

    result
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliology::{FsmEnumerator, matching_pennies};
    use std::time::Instant;

    /// Matching pennies with 2-state FSMs should have lots of structure → reducible.
    #[test]
    fn test_irreducibility_simple_game_reducible() {
        let strategies = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
        let gate = IrreducibilityGate::default();
        let result = gate.analyze(&matrix);

        // Matching pennies with 2-state FSMs produces payoffs clustered around
        // a few distinct values, so value diversity is low → reducible.
        assert!(
            !result.is_irreducible,
            "matching pennies should be reducible, ratio={}",
            result.compression_ratio
        );
    }

    /// A random matrix should have a high compression ratio (near 1.0).
    #[test]
    fn test_irreducibility_random_matrix_irreducible() {
        // Build a matrix with pseudo-random payoffs that don't compress well.
        let n = 22;
        let mut payoffs = Vec::with_capacity(n);
        // Use a simple LCG for reproducibility.
        let mut state: u64 = 42;
        for _ in 0..n {
            let mut row = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((state >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
                row.push(val);
            }
            payoffs.push(row);
        }

        let ids: Vec<u64> = (0..n as u64).collect();
        let matrix = WinMatrix::new(payoffs, ids);
        let gate = IrreducibilityGate::default();
        let result = gate.analyze(&matrix);

        assert!(
            result.is_irreducible,
            "random matrix should be irreducible, ratio={}",
            result.compression_ratio
        );
        assert!(
            result.compression_ratio > 0.7,
            "random matrix should have high compression ratio, got {}",
            result.compression_ratio
        );
    }

    /// A uniform matrix should compress very well (all same values → 2 bytes).
    #[test]
    fn test_irreducibility_uniform_matrix_reducible() {
        let n = 10;
        let payoffs = vec![vec![0.5; n]; n];
        let ids: Vec<u64> = (0..n as u64).collect();
        let matrix = WinMatrix::new(payoffs, ids);

        let gate = IrreducibilityGate::default();
        let result = gate.analyze(&matrix);

        assert!(
            !result.is_irreducible,
            "uniform matrix should be reducible, ratio={}",
            result.compression_ratio
        );
        // All same value quantizes to same byte → RLE produces 2 bytes for n*n elements.
        assert!(
            result.compression_ratio < 0.1,
            "uniform matrix should compress very well, got {}",
            result.compression_ratio
        );
    }

    /// Verify RLE on known data.
    #[test]
    fn test_rle_compress_basic() {
        // [1, 1, 2, 2, 3] → [1, 2, 2, 2, 3, 1]
        let data = [1u8, 1, 2, 2, 3];
        let compressed = rle_compress(&data);
        assert_eq!(compressed, vec![1, 2, 2, 2, 3, 1]);
    }

    /// All same values should compress to exactly 2 bytes.
    #[test]
    fn test_rle_compress_all_same() {
        let data = [42u8; 100];
        let compressed = rle_compress(&data);
        assert_eq!(compressed.len(), 2, "all-same should compress to 2 bytes");
        assert_eq!(compressed[0], 42);
        assert_eq!(compressed[1], 100);
    }

    /// Benchmark: analyze() should be sub-millisecond for a 22x22 matrix.
    #[test]
    fn test_gate_overhead() {
        // Build a 22x22 matrix (typical FSM(2) tournament size).
        let strategies = FsmEnumerator::enumerate(2);
        let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
        let gate = IrreducibilityGate::default();

        // Warm up.
        let _ = gate.analyze(&matrix);

        // Measure.
        let iterations = 1000u64;
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(gate.analyze(std::hint::black_box(&matrix)));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / iterations as u32;

        assert!(
            per_call.as_micros() < 1000,
            "gate overhead should be <1ms per call, got {:?}",
            per_call
        );
    }

    /// Verify all IrreducibilityResult fields are populated correctly.
    #[test]
    fn test_irreducibility_result_fields() {
        let payoffs = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let ids = vec![1u64, 2];
        let matrix = WinMatrix::new(payoffs, ids);

        let gate = IrreducibilityGate::new(0.5);
        let result = gate.analyze(&matrix);

        // All fields should be populated (no zeros from bugs).
        assert!(result.compression_ratio >= 0.0 && result.compression_ratio <= 2.0);
        assert!(
            result.mean_abs_payoff > 0.0,
            "mean_abs_payoff should be positive"
        );
        assert!(
            result.payoff_variance >= 0.0,
            "variance should be non-negative"
        );

        // Matrix has only 2 distinct quantized values (0 and 255).
        // log2(2)/8 = 0.125, which is below the 0.5 threshold → not irreducible.
        assert!(
            !result.is_irreducible,
            "binary-valued matrix should be reducible, ratio={}",
            result.compression_ratio
        );
        assert!(
            (result.compression_ratio - 0.125).abs() < 0.01,
            "2 distinct values should give ratio ~0.125, got {}",
            result.compression_ratio
        );
    }
}

// TL;DR: IrreducibilityGate — RLE compression ratio as Kolmogorov complexity proxy. Low ratio = game is predictable (skip simulation), high ratio = irreducible (must simulate). Default threshold 0.7. Sub-millisecond overhead for 22x22 matrices.
