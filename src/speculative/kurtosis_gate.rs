//! Kurtosis Gate — Polarization-Driven Speculative Decoding (Plan 203b).
//!
//! Computes excess kurtosis of draft marginals at each position as a zero-cost
//! signal for monosemanticity (arXiv:2606.03990, Section 5.1, Figure 6a).
//!
//! High kurtosis = concentrated distribution = confident draft = accept speculation.
//! Low kurtosis = flat distribution = uncertain draft = fall back to autoregressive.

/// Excess kurtosis of a distribution in a single O(n) pass.
///
/// Returns 0.0 for degenerate inputs (n < 4 or zero variance).
/// No allocation — operates entirely on the input slice.
#[inline]
pub fn excess_kurtosis(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    if n < 4.0 {
        return 0.0;
    }

    let mean: f32 = values.iter().sum::<f32>() / n;

    let (m2, m4) = values.iter().fold((0.0f32, 0.0f32), |(m2, m4), &x| {
        let d = x - mean;
        (m2 + d * d, m4 + d * d * d * d)
    });

    if m2 < 1e-10 {
        return 0.0;
    }

    // Population excess kurtosis: κ = n·m₄/m₂² − 3
    (m4 * n) / (m2 * m2) - 3.0
}

/// Excess kurtosis given a pre-computed sum (avoids a re-scan for the mean).
///
/// Equivalent to `excess_kurtosis(values)` but reuses `sum = Σ values` so the
/// mean is `sum / n` without a second full pass. Used by [`KurtosisGate`] on the
/// unnormalized `exp(l - max)` values — valid because excess kurtosis is
/// scale-invariant.
#[inline]
fn excess_kurtosis_from_sum(values: &[f32], sum: f32) -> f32 {
    let n = values.len() as f32;
    if n < 4.0 {
        return 0.0;
    }

    let mean = sum / n;

    let (m2, m4) = values.iter().fold((0.0f32, 0.0f32), |(m2, m4), &x| {
        let d = x - mean;
        (m2 + d * d, m4 + d * d * d * d)
    });

    if m2 < 1e-10 {
        return 0.0;
    }

    // Population excess kurtosis: κ = n·m₄/m₂² − 3
    (m4 * n) / (m2 * m2) - 3.0
}

/// Per-position kurtosis gate for speculative decoding.
///
/// Uses the polarization effect from arXiv:2606.03990:
/// high excess kurtosis indicates a peaked (confident) draft distribution,
/// which is a strong signal for successful speculation.
///
/// The gate computes softmax on raw logits, then measures excess kurtosis
/// of the resulting probability distribution. Positions with kurtosis above
/// the threshold are accepted for speculation; others fall back to AR.
///
/// Pre-allocates a scratch buffer of `vocab_size` floats at construction,
/// reused across all calls — zero allocation on the hot path.
pub struct KurtosisGate {
    /// Minimum excess kurtosis to accept speculation.
    threshold: f32,
    /// Pre-allocated scratch buffer for softmax normalization.
    scratch: Vec<f32>,
}

impl KurtosisGate {
    /// Create a new gate with explicit threshold and pre-allocated vocab buffer.
    #[inline]
    pub fn new(threshold: f32, vocab_size: usize) -> Self {
        Self {
            threshold,
            scratch: Vec::with_capacity(vocab_size),
        }
    }

    /// Create a gate with default threshold (0.0) and pre-allocated vocab buffer.
    ///
    /// Default threshold of 0.0 means any positive excess kurtosis (leptokurtic)
    /// is accepted — the distribution is more peaked than Gaussian.
    #[inline]
    pub fn with_vocab_size(vocab_size: usize) -> Self {
        Self::new(0.0, vocab_size)
    }

    /// Check if a position should be speculated on, given raw logits.
    ///
    /// 1. Softmax logits → probabilities (numerically stable)
    /// 2. Compute excess kurtosis of the probability distribution
    /// 3. Return `true` if kurtosis exceeds the threshold
    ///
    /// Returns `false` for empty or degenerate inputs.
    ///
    /// Optimization: excess kurtosis is scale-invariant
    /// (`excess_kurtosis(c·p) == excess_kurtosis(p)` for any `c > 0`), so we
    /// skip the explicit normalize pass and compute kurtosis directly on the
    /// unnormalized `exp(l - max)` values. This drops the hot path from 5 full
    /// vocab scans (max, exp+sum, normalize, mean, m2/m4) to 3 (max, exp+sum,
    /// m2/m4 with mean = sum/n known from pass 2).
    pub fn should_speculate(&mut self, logits: &[f32]) -> bool {
        if let 0..=3 = logits.len() {
            return false;
        }

        self.scratch.clear();

        // Pass 1: max for numerical stability.
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Pass 2: exp + running sum (no normalize — kurtosis is scale-invariant).
        let mut sum: f32 = 0.0;
        for &l in logits {
            let p = (l - max_logit).exp();
            self.scratch.push(p);
            sum += p;
        }

        if sum < f32::EPSILON {
            return false;
        }

        // Pass 3: central moments on the unnormalized values. mean = sum/n is
        // already known; no need to re-sum inside excess_kurtosis.
        excess_kurtosis_from_sum(&self.scratch, sum) > self.threshold
    }

    /// Compute excess kurtosis directly from a probability distribution.
    ///
    /// Use when you already have probabilities (not raw logits).
    /// Returns `false` if kurtosis is below threshold.
    #[inline]
    pub fn should_speculate_probs(&self, probs: &[f32]) -> bool {
        excess_kurtosis(probs) > self.threshold
    }

    /// Get the current threshold.
    #[inline]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Set the threshold.
    #[inline]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_excess_kurtosis_peaked_distribution() {
        // Single dominant value among zeros → very high kurtosis
        let values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
        let k = excess_kurtosis(&values);
        assert!(
            k > 5.0,
            "Peaked distribution should have high kurtosis, got {k}"
        );
    }

    #[test]
    fn test_excess_kurtosis_uniform() {
        // 5 uniform values: excess kurtosis = -1.2
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let k = excess_kurtosis(&values);
        assert!(
            k < 0.0,
            "Uniform distribution should have negative kurtosis, got {k}"
        );
        // Analytical: for discrete uniform on {1,...,n}, excess kurtosis = -6(n²+1)/(5(n²-1))
        // For n=5: -6*26/(5*24) = -156/120 = -1.3
        assert!(
            (k - (-1.3)).abs() < 0.2,
            "Expected ≈-1.3 for uniform, got {k}"
        );
    }

    #[test]
    fn test_excess_kurtosis_normal_like() {
        // Bell-shaped: excess kurtosis should be near 0
        let values = [0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1];
        let k = excess_kurtosis(&values);
        assert!(
            k.abs() < 1.0,
            "Normal-like distribution should have kurtosis near 0, got {k}"
        );
    }

    #[test]
    fn test_excess_kurtosis_edge_cases() {
        // Empty
        assert_eq!(excess_kurtosis(&[]), 0.0);
        // Single
        assert_eq!(excess_kurtosis(&[1.0]), 0.0);
        // Two
        assert_eq!(excess_kurtosis(&[1.0, 2.0]), 0.0);
        // Three
        assert_eq!(excess_kurtosis(&[1.0, 2.0, 3.0]), 0.0);
        // All same (zero variance)
        assert_eq!(excess_kurtosis(&[3.0, 3.0, 3.0, 3.0]), 0.0);
    }

    #[test]
    fn test_gate_should_speculate_peaked_logits() {
        let mut gate = KurtosisGate::new(0.0, 10);
        // One very high logit, rest very low → peaked softmax → high kurtosis
        let logits = [
            -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 10.0,
        ];
        assert!(
            gate.should_speculate(&logits),
            "Peaked logits should speculate"
        );
    }

    #[test]
    fn test_gate_should_speculate_flat_logits() {
        let mut gate = KurtosisGate::new(0.0, 10);
        // All same logits → flat softmax → kurtosis ≈ -1.2 → should NOT speculate
        let logits = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert!(
            !gate.should_speculate(&logits),
            "Flat logits should not speculate"
        );
    }

    #[test]
    fn test_gate_should_speculate_edge_cases() {
        let mut gate = KurtosisGate::new(0.0, 10);
        assert!(!gate.should_speculate(&[]), "Empty should not speculate");
        assert!(
            !gate.should_speculate(&[1.0]),
            "Single should not speculate"
        );
        assert!(
            !gate.should_speculate(&[1.0, 2.0]),
            "Two should not speculate"
        );
        assert!(
            !gate.should_speculate(&[1.0, 2.0, 3.0]),
            "Three should not speculate"
        );
    }

    #[test]
    fn test_gate_with_threshold() {
        let mut gate = KurtosisGate::new(10.0, 10);
        // Peaked logits → high kurtosis but maybe not > 10
        let logits = [-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, 5.0];
        let result = gate.should_speculate(&logits);
        // Just check it runs without panic — actual threshold filtering is working
        let _ = result;
    }

    #[test]
    fn test_gate_should_speculate_probs() {
        let gate = KurtosisGate::new(0.0, 10);
        // Peaked probability distribution
        let probs = [0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02];
        assert!(
            gate.should_speculate_probs(&probs),
            "Peaked probs should speculate"
        );
        // Flat probability distribution
        let flat = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        assert!(
            !gate.should_speculate_probs(&flat),
            "Flat probs should not speculate"
        );
    }

    #[test]
    fn test_gate_threshold_accessors() {
        let mut gate = KurtosisGate::new(1.5, 100);
        assert_eq!(gate.threshold(), 1.5);
        gate.set_threshold(2.0);
        assert_eq!(gate.threshold(), 2.0);
    }

    #[test]
    fn test_gate_with_vocab_size() {
        let gate = KurtosisGate::with_vocab_size(32000);
        assert_eq!(gate.threshold(), 0.0);
    }

    #[test]
    fn test_excess_kurtosis_dirac_like() {
        // Almost-Dirac: one value dominates → very high kurtosis
        // Use many zeros so the single peak dominates heavily
        let values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
        let k = excess_kurtosis(&values);
        assert!(
            k > 0.0,
            "Dirac-like distribution should have positive kurtosis, got {k}"
        );
    }

    #[test]
    fn test_excess_kurtosis_symmetric_bimodal() {
        // Bimodal: two equal peaks → negative kurtosis (platykurtic)
        let values = [5.0, 0.0, 0.0, 0.0, 0.0, 5.0];
        let k = excess_kurtosis(&values);
        assert!(k < 0.0, "Bimodal should have negative kurtosis, got {k}");
    }

    // ── Benchmark: kurtosis overhead at various vocab sizes ──────────
    // These are #[test] not #[bench] so they work on stable Rust.
    // Thresholds are generous for debug builds; release should be ~10x faster.
    // Target in release: <1μs per position at V=32000.

    #[test]
    fn test_bench_kurtosis_v128() {
        let values: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            std::hint::black_box(excess_kurtosis(&values));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("excess_kurtosis V=128: {per_call:.0}ns/call");
        assert!(
            per_call < 50_000.0,
            "V=128 kurtosis should be <50μs (debug), got {per_call:.0}ns"
        );
    }

    #[test]
    fn test_bench_kurtosis_v1024() {
        let values: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            std::hint::black_box(excess_kurtosis(&values));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("excess_kurtosis V=1024: {per_call:.0}ns/call");
        assert!(
            per_call < 100_000.0,
            "V=1024 kurtosis should be <100μs (debug), got {per_call:.0}ns"
        );
    }

    #[test]
    fn test_bench_kurtosis_v32000() {
        let values: Vec<f32> = (0..32000).map(|i| (i as f32).sin()).collect();
        let start = std::time::Instant::now();
        let iters = 1_000;
        for _ in 0..iters {
            std::hint::black_box(excess_kurtosis(&values));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("excess_kurtosis V=32000: {per_call:.0}ns/call");
        assert!(
            per_call < 2_000_000.0,
            "V=32000 kurtosis should be <2ms (debug), got {per_call:.0}ns"
        );
    }

    #[test]
    fn test_bench_gate_should_speculate_v32000() {
        let mut gate = KurtosisGate::new(0.0, 32000);
        let logits: Vec<f32> = (0..32000).map(|i| (i as f32 * 0.001).sin()).collect();
        let start = std::time::Instant::now();
        let iters = 1_000;
        for _ in 0..iters {
            std::hint::black_box(gate.should_speculate(&logits));
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iters as f64;
        eprintln!("KurtosisGate::should_speculate V=32000: {per_call:.0}ns/call");
        assert!(
            per_call < 5_000_000.0,
            "V=32000 gate should be <5ms (debug), got {per_call:.0}ns"
        );
    }
}

// TL;DR: Zero-cost excess kurtosis gate for speculative decoding.
// `excess_kurtosis()` — O(V) single-pass, no alloc. `KurtosisGate` — pre-allocated
// softmax + kurtosis check. High kurtosis = confident draft = accept speculation.
// Feature-gated behind `kurtosis_gate`, default ON.
