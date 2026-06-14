//! `VarianceAdaptiveGuidance` (Plan 268 F4) — sigmoid-gated per-query
//! guidance weight `1/β`.
//!
//! # QGF Paper Insight
//!
//! The paper (Fig 20) shows the guidance weight `1/β` has a sweet spot:
//! too low → no improvement, too high → off-manifold exploitation.
//! The paper uses a *fixed* `1/β` tuned per-domain.
//!
//! # Our Extension
//!
//! We make `1/β` **adaptive per-query** using the critic's own variance
//! as the confidence signal. This is a novel extension the paper doesn't
//! explore, but is natural given our existing variance-probe infrastructure
//! (Thicket, Plan 267).
//!
//! ```text
//! confidence = 1.0 - normalized_variance(Q(s, ·))
//! guidance_weight = sigmoid(k · (confidence - threshold))
//! ```
//!
//! - **Low confidence** (noisy critic, novel state) → weight ≈ 0 →
//!   pure BC reference policy (safe fallback)
//! - **High confidence** (stable critic, familiar state) → weight ≈ 1 →
//!   strong Q-guidance (aggressive, high-quality)
//!
//! # Why Sigmoid, Not Softmax
//!
//! Per project rules (see `.contexts/optimization.md`):
//! - Sigmoid is per-query (independent of other queries)
//! - Softmax would couple queries (unnecessary, expensive)
//! - Sigmoid is one SIMD op; softmax requires a normalization pass
//!
//! # Stability
//!
//! Confidence is **EMA-smoothed** to avoid oscillation. Use the existing
//! PrudentBanker pattern (`alpha = 0.01`) — confidence changes slowly
//! because the critic's variance changes slowly.

/// Compute the adaptive guidance weight from a confidence value.
///
/// `guidance_weight = sigmoid(k · (confidence − threshold))`
///
/// Returns a value in `(0.0, 1.0)` suitable for use as `1/β` in QGF.
///
/// # Arguments
///
/// - `confidence`: critic confidence at the current state, in `[0.0, 1.0]`.
///   From `QGradientOracle::confidence()` or Thicket variance probe.
/// - `threshold`: the confidence level at which guidance is half-maximal.
///   Typical: `0.5` (guidance activates when critic is above average confidence).
/// - `steepness`: controls the transition sharpness. Typical: `4.0` to `8.0`.
///   Higher = sharper on/off; lower = smoother ramp.
///
/// # Properties
///
/// - At `confidence = threshold`, returns exactly `0.5`
/// - At `confidence << threshold`, returns ≈ `0.0` (pure BC reference)
/// - At `confidence >> threshold`, returns ≈ `1.0` (strong Q-guidance)
/// - Monotonically increasing in `confidence`
/// - Continuous (no discontinuity at the threshold)
///
/// # Example
///
/// ```
/// # use katgpt_core::qgf::adaptive_guidance_weight;
/// // Low confidence: ~5% guidance (safe)
/// let w_low = adaptive_guidance_weight(0.1, 0.5, 6.0);
/// assert!(w_low < 0.1);
///
/// // High confidence: ~95% guidance (aggressive)
/// let w_high = adaptive_guidance_weight(0.9, 0.5, 6.0);
/// assert!(w_high > 0.9);
///
/// // At threshold: exactly 50%
/// let w_mid = adaptive_guidance_weight(0.5, 0.5, 6.0);
/// assert!((w_mid - 0.5).abs() < 1e-6);
/// ```
#[inline]
pub fn adaptive_guidance_weight(confidence: f32, threshold: f32, steepness: f32) -> f32 {
    // Standard logistic sigmoid: 1 / (1 + exp(-x))
    // We use the numerically-stable branch: for x >= 0, use 1/(1+exp(-x));
    // for x < 0, use exp(x)/(1+exp(x)).
    let x = steepness * (confidence - threshold);
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Default configuration for adaptive guidance.
///
/// Tuned to be conservative — guidance activates only when the critic is
/// clearly confident. This matches QGF's finding that overly-aggressive
/// guidance pushes actions off-manifold.
pub const DEFAULT_THRESHOLD: f32 = 0.5;
pub const DEFAULT_STEEPNESS: f32 = 6.0;

/// Convenience: adaptive guidance weight with default config.
#[inline]
pub fn adaptive_guidance_weight_default(confidence: f32) -> f32 {
    adaptive_guidance_weight(confidence, DEFAULT_THRESHOLD, DEFAULT_STEEPNESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_confidence_near_zero() {
        let w = adaptive_guidance_weight(0.1, 0.5, 6.0);
        assert!(w < 0.1, "low confidence should give < 0.1 weight, got {w}");
    }

    #[test]
    fn test_high_confidence_near_one() {
        let w = adaptive_guidance_weight(0.9, 0.5, 6.0);
        assert!(w > 0.9, "high confidence should give > 0.9 weight, got {w}");
    }

    #[test]
    fn test_at_threshold_is_half() {
        let w = adaptive_guidance_weight(0.5, 0.5, 6.0);
        assert!(
            (w - 0.5).abs() < 1e-6,
            "at threshold should be 0.5, got {w}"
        );
    }

    #[test]
    fn test_monotonic_in_confidence() {
        // Guidance weight must be monotonically increasing in confidence.
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let conf = i as f32 / 100.0;
            let w = adaptive_guidance_weight(conf, 0.5, 6.0);
            assert!(
                w >= prev - 1e-6,
                "not monotonic at conf={conf}: prev={prev}, w={w}"
            );
            prev = w;
        }
    }

    #[test]
    fn test_continuous_at_threshold() {
        // No discontinuity — values approach 0.5 smoothly from both sides.
        let below = adaptive_guidance_weight(0.49, 0.5, 6.0);
        let at = adaptive_guidance_weight(0.5, 0.5, 6.0);
        let above = adaptive_guidance_weight(0.51, 0.5, 6.0);
        assert!(below < at);
        assert!(at < above);
        assert!((at - below).abs() < 0.05);
        assert!((above - at).abs() < 0.05);
    }

    #[test]
    fn test_steeper_is_sharper() {
        // Higher steepness → sharper transition.
        let gentle = adaptive_guidance_weight(0.55, 0.5, 2.0);
        let steep = adaptive_guidance_weight(0.55, 0.5, 20.0);
        assert!(
            steep > gentle,
            "steeper should activate faster: gentle={gentle}, steep={steep}"
        );
    }

    #[test]
    fn test_output_range() {
        // Output must always be in [0, 1] for any finite input.
        // Note: sigmoid asymptotically approaches (0, 1) but saturates at the
        // float limit for extreme inputs (e.g. confidence=3.28 with steepness=6
        // gives x=16.68, where exp(-16.68) underflows to 0 → sigmoid = 1.0 exactly).
        for i in -1000..=1000 {
            let conf = i as f32 / 100.0;
            let w = adaptive_guidance_weight(conf, 0.5, 6.0);
            assert!(
                w >= 0.0 && w <= 1.0,
                "weight out of [0,1] for conf={conf}: w={w}"
            );
        }
    }
}
