//! ChiarRegimeGate — naturalistic vs synthetic prompt gate (Plan 269, Fusion D).
//!
//! Paper's operating regime characterization: CHIAR spectral preprocessing
//! benefits large naturalistic text, but hurts small datasets or synthetic
//! pattern-matching (ListOps).
//!
//! At inference time, we approximate the regime signal using:
//! - Prompt length (proxy for "large enough to benefit")
//! - H(x) variance across the prompt (proxy for "naturalistic diversity")
//!
//! Low H(x) variance → synthetic / repetitive → skip CHIAR.
//! High H(x) variance + long prompt → naturalistic → apply CHIAR.

use crate::chiaroscuro::entropy::spectral_entropy_dct;

/// Default minimum prompt length to consider CHIAR worth applying.
///
/// Paper's breakeven was around 4096 tokens for WikiText-2 → WikiText-103.
/// We use the same threshold.
pub const DEFAULT_MIN_PROMPT_TOKENS: usize = 4096;

/// Default minimum H(x) variance for "naturalistic" classification.
///
/// Synthetic tasks (ListOps) have very low H(x) variance — tokens are
/// structurally similar. Naturalistic text has high variance.
pub const DEFAULT_NATURALISTIC_VARIANCE: f32 = 0.0005;

/// Online Welford variance tracker for H(x) observations.
///
/// Reuses the pattern from `pruners::acceptance_variance` and
/// `pruners::reward_calibrator`. Kept local to CHIAR module to avoid coupling.
#[derive(Clone, Debug, Default)]
pub struct WelfordVariance {
    count: u64,
    mean: f64,
    m2: f64,
}

impl WelfordVariance {
    /// Observe one sample. O(1), 3 flops.
    #[inline]
    pub fn observe(&mut self, x: f32) {
        self.count += 1;
        let x = x as f64;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Number of observations.
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sample mean. Returns 0.0 if no observations.
    #[inline]
    pub fn mean(&self) -> f32 {
        self.mean as f32
    }

    /// Sample variance (n-1 denominator). Returns 0.0 for < 2 observations.
    pub fn variance(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        (self.m2 / (self.count as f64 - 1.0)) as f32
    }

    /// Sample standard deviation.
    #[inline]
    pub fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Reset to empty state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Operating regime gate — decides whether to apply CHIAR spectral preprocessing.
///
/// Combines prompt length and H(x) variance into a single boolean decision.
/// Integrates with [`crate::breakeven`] and [`crate::trigger_gate`] — the gate's
/// output feeds into the cost-aware tier matrix.
pub struct ChiarRegimeGate {
    /// Tokens seen in the current prompt.
    prompt_tokens: usize,
    /// Running H(x) variance.
    h_variance: WelfordVariance,
    /// Minimum prompt length to consider CHIAR.
    min_prompt_tokens: usize,
    /// Minimum H(x) variance for "naturalistic" classification.
    naturalistic_variance: f32,
}

impl Default for ChiarRegimeGate {
    fn default() -> Self {
        Self::new(DEFAULT_MIN_PROMPT_TOKENS, DEFAULT_NATURALISTIC_VARIANCE)
    }
}

impl ChiarRegimeGate {
    /// Create a new regime gate with the given thresholds.
    pub fn new(min_prompt_tokens: usize, naturalistic_variance: f32) -> Self {
        Self {
            prompt_tokens: 0,
            h_variance: WelfordVariance::default(),
            min_prompt_tokens,
            naturalistic_variance,
        }
    }

    /// Observe one token's H(x). Updates variance tracker and token count.
    #[inline]
    pub fn observe_h(&mut self, h_x: f32) {
        self.prompt_tokens += 1;
        self.h_variance.observe(h_x);
    }

    /// Observe a raw key embedding — computes H(x) and updates state.
    #[inline]
    pub fn observe_key(&mut self, key: &[f32]) {
        let h = spectral_entropy_dct(key);
        self.observe_h(h);
    }

    /// Whether CHIAR spectral preprocessing should be applied.
    ///
    /// Returns true iff prompt is large enough AND H(x) variance is high enough.
    /// Uses **sigmoid smoothing** so the decision isn't a sharp cliff:
    ///
    /// ```text
    /// p_length = σ(α · (prompt_tokens - min_prompt_tokens))
    /// p_variance = σ(β · (variance - naturalistic_variance))
    /// apply = (p_length > 0.5) AND (p_variance > 0.5)
    /// ```
    ///
    /// Sigmoid (not softmax) per project constraint.
    pub fn should_apply_chiar(&self) -> bool {
        if self.prompt_tokens == 0 {
            return false;
        }
        // Sigmoid smoothness constants — steepness of transition.
        let alpha = 0.01; // gentle — 100-token window for transition
        let beta = 1000.0; // sharp — variance is small in absolute terms

        let length_signal = sigmoid(alpha * (self.prompt_tokens as f32 - self.min_prompt_tokens as f32));
        let variance_signal = sigmoid(beta * (self.h_variance.variance() - self.naturalistic_variance));

        // AND gate: both signals must individually cross 0.5 (their respective thresholds).
        // This avoids the case where variance = 0 still passes because length signal ≈ 1.
        length_signal > 0.5 && variance_signal > 0.5
    }

    /// Sigmoid-smoothed probability of applying CHIAR. ∈ (0, 1).
    ///
    /// Useful for cost-aware routing where partial activation makes sense.
    pub fn apply_probability(&self) -> f32 {
        if self.prompt_tokens == 0 {
            return 0.0;
        }
        let alpha = 0.01;
        let beta = 1000.0;
        let length_signal = sigmoid(alpha * (self.prompt_tokens as f32 - self.min_prompt_tokens as f32));
        let variance_signal = sigmoid(beta * (self.h_variance.variance() - self.naturalistic_variance));
        length_signal * variance_signal
    }

    /// Number of tokens observed so far.
    #[inline]
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Current H(x) variance estimate.
    #[inline]
    pub fn h_variance(&self) -> f32 {
        self.h_variance.variance()
    }

    /// Current H(x) mean.
    #[inline]
    pub fn h_mean(&self) -> f32 {
        self.h_variance.mean()
    }

    /// Reset for a new prompt.
    pub fn reset(&mut self) {
        self.prompt_tokens = 0;
        self.h_variance.reset();
    }
}

/// Standard sigmoid (re-exported from entropy for convenience).
#[inline]
fn sigmoid(x: f32) -> f32 {
    crate::chiaroscuro::entropy::sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_variance_basic() {
        let mut w = WelfordVariance::default();
        w.observe(1.0);
        w.observe(2.0);
        w.observe(3.0);
        assert!((w.mean() - 2.0).abs() < 1e-6);
        assert!(w.variance() > 0.0, "variance of {{1,2,3}} should be positive");
        // Sample variance of {1,2,3} = ((1-2)² + (2-2)² + (3-2)²) / (3-1) = 2/2 = 1.0
        assert!((w.variance() - 1.0).abs() < 1e-6, "variance should be 1.0");
    }

    #[test]
    fn test_welford_single_sample_zero_variance() {
        let mut w = WelfordVariance::default();
        w.observe(0.5);
        assert_eq!(w.variance(), 0.0);
    }

    #[test]
    fn test_welford_reset() {
        let mut w = WelfordVariance::default();
        w.observe(1.0);
        w.observe(2.0);
        w.reset();
        assert_eq!(w.count(), 0);
        assert_eq!(w.variance(), 0.0);
    }

    #[test]
    fn test_gate_default_does_not_apply_initially() {
        let g = ChiarRegimeGate::default();
        assert!(!g.should_apply_chiar(), "empty gate should not apply");
    }

    #[test]
    fn test_gate_short_prompt_does_not_apply() {
        let mut g = ChiarRegimeGate::default();
        // 100 tokens of mixed entropy.
        for i in 0..100 {
            g.observe_h(0.85 + ((i % 10) as f32) * 0.001);
        }
        // Short prompt → no.
        assert!(!g.should_apply_chiar(), "short prompt should not trigger CHIAR");
    }

    #[test]
    fn test_gate_long_low_variance_prompt_does_not_apply() {
        let mut g = ChiarRegimeGate::default();
        // 8192 tokens of constant entropy → low variance → synthetic-like.
        for _ in 0..8192 {
            g.observe_h(0.85);
        }
        assert!(
            !g.should_apply_chiar(),
            "long but low-variance prompt should not trigger CHIAR (synthetic regime)"
        );
    }

    #[test]
    fn test_gate_long_high_variance_prompt_applies() {
        let mut g = ChiarRegimeGate::default();
        // 8192 tokens of varied entropy → naturalistic.
        let mut state: u32 = 42;
        for _ in 0..8192 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let h = 0.80 + ((state >> 8) as f32 / 16777216.0) * 0.15; // [0.80, 0.95]
            g.observe_h(h);
        }
        assert!(
            g.should_apply_chiar(),
            "long high-variance prompt should trigger CHIAR (naturalistic regime)"
        );
    }

    #[test]
    fn test_gate_apply_probability_bounded() {
        let mut g = ChiarRegimeGate::default();
        for _ in 0..10000 {
            g.observe_h(0.9);
        }
        let p = g.apply_probability();
        assert!((0.0..=1.0).contains(&p), "probability must be in [0, 1], got {p}");
    }

    #[test]
    fn test_gate_reset() {
        let mut g = ChiarRegimeGate::default();
        for _ in 0..100 {
            g.observe_h(0.85);
        }
        assert_eq!(g.prompt_tokens(), 100);
        g.reset();
        assert_eq!(g.prompt_tokens(), 0);
        assert_eq!(g.h_variance(), 0.0);
    }

    #[test]
    fn test_observe_key_runs() {
        let mut g = ChiarRegimeGate::default();
        g.observe_key(&[1.0f32; 64]);
        g.observe_key(&[0.5f32; 64]);
        assert_eq!(g.prompt_tokens(), 2);
    }
}
