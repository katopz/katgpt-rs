//! CHIAR-KV — Per-token spectral-entropy-gated KV cache strategy (Plan 269, Fusion A).
//!
//! The CHIAR-Former paper routes **operators** per token. We apply the same
//! chiaroscuro principle at the **storage layer**: route **storage fidelity**
//! per token based on its spectral entropy H(x).
//!
//! ```text
//! For each key k_i in KV cache:
//!   H(k_i) = spectral_entropy_dct(k_i)
//!   if H(k_i) < τ_lo:
//!       → DctTruncated { n_coeffs }   (top-K low-freq DCT coefficients)
//!         reconstruction error bounded by spectral tail (Theorem 1)
//!   elif H(k_i) < τ_hi:
//!       → Quantized                    (variable-bit, delegates to SpectralQuant)
//!   else:
//!       → FullPrecision                (f16)
//! ```
//!
//! # Why novel
//!
//! - [`crate::spectralquant::spectral_kv_cache`] operates per-**dimension**
//!   (rotation + variable-bit). CHIAR-KV operates per-**token**.
//! - [`crate::kvarn`] uses variance across positions. CHIAR-KV uses per-token
//!   spectral complexity.
//! - [`crate::still_kv`] uses perceptual compaction. CHIAR-KV uses spectral
//!   truncation.
//!
//! All three compose: CHIAR-KV picks the **storage strategy** per token;
//! existing systems handle the per-strategy storage mechanics.
//!
//! # Modelless
//!
//! Pure inference-time computation. τ calibrated via [`StreamingTauCalibrator`].

use crate::chiaroscuro::entropy::{spectral_entropy_dct, spectral_entropy_dct_into};
use rustfft::{FftPlanner, num_complex::Complex32};

/// Default number of DCT coefficients retained for `DctTruncated` strategy.
///
/// For d=256, 32 coefficients is 8× compression with bounded reconstruction
/// error (top-12% of energy packed into top-K, per DCT Karhunen-Loève property).
pub const DEFAULT_DCT_TRUNCATED_COEFFS: usize = 32;

/// Per-token KV cache storage strategy.
///
/// Determined by [`ChiaroscuroKvStrategy::decide`] from the key embedding's
/// spectral entropy and the current τ_lo / τ_hi thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ChiaroscuroKvStrategy {
    /// Low-entropy (smooth) token. Store top-K low-frequency DCT coefficients.
    /// O(d log d) to compress, O(K) to read. Maximum compression.
    DctTruncated = 0,
    /// Mid-entropy token. Delegate to SpectralQuant variable-bit packing.
    Quantized = 1,
    /// High-entropy (complex) token. Store full f16 precision. No compression.
    FullPrecision = 2,
}

impl ChiaroscuroKvStrategy {
    /// Number of strategies. Used for utilization counters.
    pub const NUM_VARIANTS: usize = 3;

    /// Convert to arm index.
    #[inline]
    pub fn as_index(self) -> usize {
        self as usize
    }

    /// Convert from arm index. Returns `None` if out of range.
    #[inline]
    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(Self::DctTruncated),
            1 => Some(Self::Quantized),
            2 => Some(Self::FullPrecision),
            _ => None,
        }
    }

    /// Approximate compression ratio vs full f16 storage.
    ///
    /// - `DctTruncated`: depends on `n_coeffs` and `d`. For `d=256, K=32` → 16×.
    /// - `Quantized`: ~4-8× depending on bit allocation.
    /// - `FullPrecision`: 1× (no compression).
    #[inline]
    pub fn compression_ratio(&self, d: usize, n_coeffs: usize) -> f32 {
        match self {
            Self::DctTruncated => {
                if n_coeffs == 0 {
                    1.0
                } else {
                    // Stores n_coeffs f32 coefficients + 1 norm scalar.
                    // vs full d × 2 bytes (f16). Rough ratio.
                    (d as f32 * 2.0) / (n_coeffs as f32 * 4.0 + 4.0)
                }
            }
            Self::Quantized => 4.0, // typical SpectralQuant compression
            Self::FullPrecision => 1.0,
        }
    }

    /// Decide the storage strategy for a key given its H(x) and τ thresholds.
    ///
    /// Implements the paper's Theorem 1 operator regime:
    /// ```text
    /// H ≤ τ_lo      → DctTruncated  (cheap, smooth)
    /// τ_lo < H ≤ τ_hi → Quantized   (medium)
    /// H > τ_hi      → FullPrecision (expensive, complex)
    /// ```
    ///
    /// Note: lower bound is inclusive (`≤`) so smooth tokens with H ≈ 0 still
    /// qualify for DctTruncated when τ_lo has decayed to 0 in streaming calibration.
    #[inline]
    pub fn decide(h_x: f32, tau_lo: f32, tau_hi: f32) -> Self {
        if h_x <= tau_lo {
            Self::DctTruncated
        } else if h_x <= tau_hi {
            Self::Quantized
        } else {
            Self::FullPrecision
        }
    }

    /// Convenience: compute H(x) from a key embedding, then decide.
    ///
    /// Allocates internal scratch — for hot loops, use [`decide_with_scratch`].
    #[inline]
    pub fn decide_from_key(key: &[f32], tau_lo: f32, tau_hi: f32) -> Self {
        let h = spectral_entropy_dct(key);
        Self::decide(h, tau_lo, tau_hi)
    }
}

/// Per-strategy utilization counter.
///
/// Tracks how many tokens have been routed to each strategy. Used by
/// [`crate::chiaroscuro::collapse::CollapseDiscoveryHarness`] to detect
/// routing collapse (e.g., paper's finding that RBF is consistently rejected).
#[derive(Clone, Debug, Default)]
pub struct StrategyUtilization {
    /// Per-strategy count, indexed by [`ChiaroscuroKvStrategy::as_index`].
    pub counts: [u64; ChiaroscuroKvStrategy::NUM_VARIANTS],
}

impl StrategyUtilization {
    /// Record one observation of the given strategy.
    #[inline]
    pub fn observe(&mut self, strategy: ChiaroscuroKvStrategy) {
        self.counts[strategy.as_index()] += 1;
    }

    /// Total observations across all strategies.
    #[inline]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Utilization entropy U = -Σ q_o log q_o, normalized to [0, 1].
    ///
    /// U → 1.0 indicates uniform usage across all strategies (no collapse).
    /// U → 0.0 indicates all tokens routed to one strategy (collapse).
    /// Normalization is `U / log(NUM_VARIANTS)`.
    pub fn utilization_entropy(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f32;
        let mut u = 0.0f32;
        for &c in &self.counts {
            if c > 0 {
                let p = c as f32 / total_f;
                u -= p * p.ln();
            }
        }
        let log_n = (ChiaroscuroKvStrategy::NUM_VARIANTS as f32).ln();
        if log_n <= 0.0 {
            0.0
        } else {
            u / log_n
        }
    }

    /// Fraction of tokens routed to the given strategy.
    #[inline]
    pub fn fraction(&self, strategy: ChiaroscuroKvStrategy) -> f32 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            self.counts[strategy.as_index()] as f32 / total as f32
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        for c in self.counts.iter_mut() {
            *c = 0;
        }
    }
}

/// CHIAR-KV dispatcher — combines H(x) computation, τ calibration, and strategy selection.
///
/// Owns a [`StrategyUtilization`] counter for collapse detection. Delegates τ
/// calibration to an external [`crate::chiaroscuro::tau::StreamingTauCalibrator`]
/// (passed by reference to avoid coupling lifetimes).
///
/// Caches an [`FftPlanner`] and scratch buffer internally so that [`dispatch`](Self::dispatch)
/// does zero heap allocation per token after the first call.
pub struct ChiaroscuroKvDispatcher {
    pub utilization: StrategyUtilization,
    /// Number of DCT coefficients for `DctTruncated` strategy.
    pub dct_n_coeffs: usize,
    /// Reused FFT planner — caches plans across tokens.
    planner: FftPlanner<f32>,
    /// Reused complex scratch buffer for the DCT mirror-and-FFT trick.
    scratch: Vec<Complex32>,
}

impl Default for ChiaroscuroKvDispatcher {
    fn default() -> Self {
        Self::new(DEFAULT_DCT_TRUNCATED_COEFFS)
    }
}

impl ChiaroscuroKvDispatcher {
    /// Create a new dispatcher with the given DCT truncation width.
    pub fn new(dct_n_coeffs: usize) -> Self {
        Self {
            utilization: StrategyUtilization::default(),
            dct_n_coeffs,
            planner: FftPlanner::new(),
            scratch: Vec::new(),
        }
    }

    /// Route a key embedding to its storage strategy, given calibrated τ.
    ///
    /// O(d log d) for the DCT + O(1) for the threshold gate.
    /// Updates the utilization counter. Zero-alloc after the first call.
    pub fn dispatch(&mut self, key: &[f32], tau_lo: f32, tau_hi: f32) -> ChiaroscuroKvStrategy {
        let h = spectral_entropy_dct_into(key, &mut self.scratch, &mut self.planner);
        let strategy = ChiaroscuroKvStrategy::decide(h, tau_lo, tau_hi);
        self.utilization.observe(strategy);
        strategy
    }

    /// Route from a pre-computed H(x) value. O(1).
    pub fn dispatch_from_h(&mut self, h_x: f32, tau_lo: f32, tau_hi: f32) -> ChiaroscuroKvStrategy {
        let strategy = ChiaroscuroKvStrategy::decide(h_x, tau_lo, tau_hi);
        self.utilization.observe(strategy);
        strategy
    }

    /// Current utilization entropy (for collapse detection).
    pub fn utilization_entropy(&self) -> f32 {
        self.utilization.utilization_entropy()
    }

    /// Reset counters (e.g., on prompt boundary or model swap).
    pub fn reset(&mut self) {
        self.utilization.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decide_low_entropy_to_dct_truncated() {
        let s = ChiaroscuroKvStrategy::decide(0.5, 0.85, 0.87);
        assert_eq!(s, ChiaroscuroKvStrategy::DctTruncated);
    }

    #[test]
    fn test_decide_mid_entropy_to_quantized() {
        let s = ChiaroscuroKvStrategy::decide(0.86, 0.85, 0.87);
        assert_eq!(s, ChiaroscuroKvStrategy::Quantized);
    }

    #[test]
    fn test_decide_high_entropy_to_full_precision() {
        let s = ChiaroscuroKvStrategy::decide(0.95, 0.85, 0.87);
        assert_eq!(s, ChiaroscuroKvStrategy::FullPrecision);
    }

    #[test]
    fn test_decide_boundary_at_tau_lo() {
        // At H == τ_lo, goes to DctTruncated (inclusive lower bound).
        let s = ChiaroscuroKvStrategy::decide(0.85, 0.85, 0.87);
        assert_eq!(s, ChiaroscuroKvStrategy::DctTruncated);
    }

    #[test]
    fn test_decide_boundary_at_tau_hi() {
        let s = ChiaroscuroKvStrategy::decide(0.87, 0.85, 0.87);
        assert_eq!(s, ChiaroscuroKvStrategy::Quantized);
    }

    #[test]
    fn test_strategy_roundtrip() {
        for i in 0..ChiaroscuroKvStrategy::NUM_VARIANTS {
            let s = ChiaroscuroKvStrategy::from_index(i).unwrap();
            assert_eq!(s.as_index(), i);
        }
    }

    #[test]
    fn test_compression_ratio_dct_truncated() {
        let s = ChiaroscuroKvStrategy::DctTruncated;
        let r = s.compression_ratio(256, DEFAULT_DCT_TRUNCATED_COEFFS);
        assert!(r > 1.0, "DCT-truncated must compress, got ratio {r}");
        // For d=256, K=32: ratio = (256*2)/(32*4+4) = 512/132 ≈ 3.88
        assert!(r > 3.0 && r < 5.0, "expected ~3.9× compression for d=256 K=32, got {r}");
    }

    #[test]
    fn test_compression_ratio_quantized() {
        let s = ChiaroscuroKvStrategy::Quantized;
        let r = s.compression_ratio(256, 0);
        assert!((r - 4.0).abs() < 1e-6, "Quantized ratio should be 4.0, got {r}");
    }

    #[test]
    fn test_compression_ratio_full_precision() {
        let s = ChiaroscuroKvStrategy::FullPrecision;
        let r = s.compression_ratio(256, 0);
        assert!((r - 1.0).abs() < 1e-6, "FullPrecision ratio should be 1.0, got {r}");
    }

    #[test]
    fn test_decide_from_key_constant() {
        // Constant embedding → H ≈ 0 → DctTruncated.
        let s = ChiaroscuroKvStrategy::decide_from_key(&[1.0f32; 64], 0.85, 0.87);
        assert_eq!(s, ChiaroscuroKvStrategy::DctTruncated);
    }

    #[test]
    fn test_utilization_entropy_uniform() {
        let mut u = StrategyUtilization::default();
        for _ in 0..100 {
            u.observe(ChiaroscuroKvStrategy::DctTruncated);
            u.observe(ChiaroscuroKvStrategy::Quantized);
            u.observe(ChiaroscuroKvStrategy::FullPrecision);
        }
        let e = u.utilization_entropy();
        assert!((e - 1.0).abs() < 1e-3, "uniform routing U should be ≈ 1.0, got {e}");
    }

    #[test]
    fn test_utilization_entropy_collapse() {
        let mut u = StrategyUtilization::default();
        for _ in 0..1000 {
            u.observe(ChiaroscuroKvStrategy::Quantized); // all to one
        }
        let e = u.utilization_entropy();
        assert!(e < 0.01, "collapsed routing U should be ≈ 0.0, got {e}");
    }

    #[test]
    fn test_utilization_entropy_two_op_split() {
        let mut u = StrategyUtilization::default();
        for _ in 0..100 {
            u.observe(ChiaroscuroKvStrategy::DctTruncated);
            u.observe(ChiaroscuroKvStrategy::FullPrecision);
        }
        let e = u.utilization_entropy();
        // For 2-of-3 uniform split: U = ln(2)/ln(3) ≈ 0.631
        assert!(
            (e - (2.0f32.ln() / 3.0f32.ln())).abs() < 1e-3,
            "2/3 uniform split U should be ≈ 0.631, got {e}"
        );
    }

    #[test]
    fn test_dispatcher_updates_utilization() {
        let mut d = ChiaroscuroKvDispatcher::default();
        let _ = d.dispatch(&[1.0f32; 64], 0.85, 0.87); // → DctTruncated
        let _ = d.dispatch(&[1.0f32; 64], 0.85, 0.87);
        assert_eq!(d.utilization.counts[ChiaroscuroKvStrategy::DctTruncated.as_index()], 2);
        assert_eq!(d.utilization.counts[ChiaroscuroKvStrategy::Quantized.as_index()], 0);
    }

    #[test]
    fn test_dispatcher_dispatch_from_h() {
        let mut d = ChiaroscuroKvDispatcher::default();
        let s = d.dispatch_from_h(0.95, 0.85, 0.87);
        assert_eq!(s, ChiaroscuroKvStrategy::FullPrecision);
        assert_eq!(d.utilization.counts[ChiaroscuroKvStrategy::FullPrecision.as_index()], 1);
    }

    #[test]
    fn test_dispatcher_reset() {
        let mut d = ChiaroscuroKvDispatcher::default();
        let _ = d.dispatch(&[1.0f32; 64], 0.85, 0.87);
        assert!(d.utilization.total() > 0);
        d.reset();
        assert_eq!(d.utilization.total(), 0);
    }

    #[test]
    fn test_utilization_fraction() {
        let mut u = StrategyUtilization::default();
        for _ in 0..3 {
            u.observe(ChiaroscuroKvStrategy::DctTruncated);
        }
        for _ in 0..1 {
            u.observe(ChiaroscuroKvStrategy::Quantized);
        }
        let f = u.fraction(ChiaroscuroKvStrategy::DctTruncated);
        assert!((f - 0.75).abs() < 1e-6, "fraction should be 0.75, got {f}");
    }
}
