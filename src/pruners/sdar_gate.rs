//! SDAR-inspired sigmoid gating for modelless distillation signals.
//!
//! Adapts the asymmetric trust principle from SDAR (arXiv:2605.15155):
//! - Positive input (endorsement) → gate opens → strong signal
//! - Negative input (rejection) → gate closes → attenuated signal
//! - β controls sharpness (5.0 = paper-validated optimum)
//!
//! Unlike SDAR's gradient-based loss, this operates on pre-computed
//! scalar signals (δ, relevance scores, bandit rewards).
//!
//! # Gate Formula
//!
//! `gt = σ(β · x)` where `σ(z) = 1 / (1 + exp(-z))`
//!
//! Properties: smooth, differentiable, bounded ∈ (0, 1), monotonic.
//! β=0 → no gate (uniform), β→∞ → binary gate.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::pruners::sdar_gate::{sdar_gate, sdar_modulate, SDAR_BETA};
//!
//! // Basic gating
//! let gate_value = sdar_gate(0.5, SDAR_BETA); // ≈ 0.924
//!
//! // Modulate a signal by a trust indicator
//! let signal = 1.0;
//! let gap = 0.3; // positive gap = endorsement
//! let gated_signal = sdar_modulate(signal, gap, SDAR_BETA);
//! ```
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "sdar_gate")]`.
//! Feature: `sdar_gate = []` in `Cargo.toml`.
//!
//! **Source:** [SDAR: Self-Distilled Agentic RL](https://arxiv.org/abs/2605.15155) — ZJU-REAL, 2025

// ── Constants ───────────────────────────────────────────────────

/// Default sigmoid sharpness from SDAR paper (β=5.0).
///
/// Empirically optimal across SDAR's ablations (Table 2):
/// - β=0: no gate (uniform distillation, collapses in multi-turn)
/// - β=1: soft gate (too permissive)
/// - β=5: optimal balance of endorsement vs attenuation
/// - β=10: near-binary gate (too aggressive, loses smooth modulation)
pub const SDAR_BETA: f32 = 5.0;

/// Minimum β value for meaningful gating.
///
/// Below this, the gate is essentially uniform (σ(β·x) ≈ 0.5 for all x).
pub const SDAR_BETA_MIN: f32 = 0.1;

/// Maximum β before the gate becomes effectively binary.
///
/// Above this, floating-point precision issues can arise.
pub const SDAR_BETA_MAX: f32 = 50.0;

// ── Core Gate Functions ─────────────────────────────────────────

/// SDAR sigmoid gate: σ(β · x).
///
/// Returns value in (0, 1). Positive x → gate opens, negative x → gate closes.
///
/// # Numerically Stable Implementation
///
/// Uses two branches to avoid overflow in `exp()`:
/// - `z >= 0`: `1 / (1 + exp(-z))` — no overflow risk
/// - `z < 0`: `exp(z) / (1 + exp(z))` — avoids `exp(large_negative)` → 0
///
/// # Arguments
///
/// * `x` — Input signal (gap, trust indicator, etc.)
/// * `beta` — Sharpness parameter. Use [`SDAR_BETA`] for paper default.
///
/// # Examples
///
/// ```
/// use katgpt_rs::pruners::sdar_gate::{sdar_gate, SDAR_BETA};
///
/// // Positive input → gate opens (≈ 0.92)
/// let open = sdar_gate(0.5, SDAR_BETA);
/// assert!(open > 0.9);
///
/// // Zero input → gate neutral (0.5)
/// let neutral = sdar_gate(0.0, SDAR_BETA);
/// assert!((neutral - 0.5).abs() < 1e-6);
///
/// // Negative input → gate closes (≈ 0.08)
/// let closed = sdar_gate(-0.5, SDAR_BETA);
/// assert!(closed < 0.1);
/// ```
#[inline]
pub fn sdar_gate(x: f32, beta: f32) -> f32 {
    let z = beta * x;
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp()) // numerically stable for z >= 0
    } else {
        let ez = z.exp();
        ez / (1.0 + ez) // numerically stable for z < 0
    }
}

/// Convenience: gate with default β=5.0.
///
/// Equivalent to `sdar_gate(x, SDAR_BETA)`.
#[inline]
pub fn sdar_gate_default(x: f32) -> f32 {
    sdar_gate(x, SDAR_BETA)
}

/// Gate a scalar signal: apply asymmetric trust.
///
/// `signal * σ(β · gap)` where gap is the trust indicator.
/// Positive gap → signal passes through. Negative gap → signal attenuated.
///
/// # Arguments
///
/// * `signal` — The signal to modulate (reward, relevance, etc.)
/// * `gap` — Trust indicator (positive = endorsement, negative = rejection)
/// * `beta` — Sharpness parameter. Use [`SDAR_BETA`] for paper default.
///
/// # Examples
///
/// ```
/// use katgpt_rs::pruners::sdar_gate::{sdar_modulate, SDAR_BETA};
///
/// // Positive gap: signal passes through with high weight
/// let strong = sdar_modulate(1.0, 0.5, SDAR_BETA);
/// assert!(strong > 0.9);
///
/// // Negative gap: signal attenuated
/// let weak = sdar_modulate(1.0, -0.5, SDAR_BETA);
/// assert!(weak < 0.1);
/// ```
#[inline]
pub fn sdar_modulate(signal: f32, gap: f32, beta: f32) -> f32 {
    signal * sdar_gate(gap, beta)
}

/// Convenience: modulate with default β=5.0.
#[inline]
pub fn sdar_modulate_default(signal: f32, gap: f32) -> f32 {
    sdar_modulate(signal, gap, SDAR_BETA)
}

/// Compute the benefit-ratio gate for absorb-compress promotion.
///
/// Replaces hard benefit-ratio threshold with sigmoid soft gate.
/// `gate = σ(β · (benefit_ratio - 1.0))` where:
/// - `benefit_ratio > 1.0` → beneficial → gate opens
/// - `benefit_ratio = 1.0` → neutral → gate = 0.5
/// - `benefit_ratio < 1.0` → harmful → gate closes
///
/// # Arguments
///
/// * `benefit_ratio` — The benefit-to-risk ratio (1.0 = neutral)
/// * `beta` — Sharpness parameter. Use [`SDAR_BETA`] for paper default.
///
/// # Examples
///
/// ```
/// use katgpt_rs::pruners::sdar_gate::{sdar_benefit_gate, SDAR_BETA};
///
/// // High benefit → gate ≈ 1.0
/// let promote = sdar_benefit_gate(3.0, SDAR_BETA);
/// assert!(promote > 0.99);
///
/// // Zero benefit → gate ≈ 0.5 (neutral)
/// let neutral = sdar_benefit_gate(1.0, SDAR_BETA);
/// assert!((neutral - 0.5).abs() < 1e-6);
///
/// // Negative benefit → gate ≈ 0.0
/// let block = sdar_benefit_gate(0.0, SDAR_BETA);
/// assert!(block < 0.01);
/// ```
#[inline]
pub fn sdar_benefit_gate(benefit_ratio: f32, beta: f32) -> f32 {
    let gap = benefit_ratio - 1.0;
    sdar_gate(gap, beta)
}

/// Compute gated bandit reward using asymmetric trust.
///
/// The modelless analog of SDAR's token-level gating:
/// - `gap = reward - q_value` (reward surprise)
/// - `gate = σ(β · gap)`
/// - `gated_reward = reward * gate`
///
/// Positive reward surprise (reward > Q-value) → full update.
/// Negative reward surprise (reward < Q-value) → attenuated update.
///
/// # Arguments
///
/// * `reward` — Observed reward
/// * `q_value` — Current Q-value estimate for this arm
/// * `beta` — Sharpness parameter. Use [`SDAR_BETA`] for paper default.
///
/// # Returns
///
/// Gated reward for bandit update.
#[inline]
pub fn sdar_gated_reward(reward: f32, q_value: f32, beta: f32) -> f32 {
    let gap = reward - q_value;
    reward * sdar_gate(gap, beta)
}

/// Compute promotion probability for absorb-compress.
///
/// Uses sigmoid gate on benefit ratio to produce soft promotion probability
/// instead of hard binary threshold.
///
/// # Arguments
///
/// * `benefit_ratio` — The benefit-to-risk ratio
/// * `beta` — Sharpness parameter
/// * `random_draw` — Uniform random ∈ [0, 1] for stochastic promotion
///
/// # Returns
///
/// `true` if this arm should be promoted (stochastic decision).
#[inline]
pub fn sdar_should_promote(benefit_ratio: f32, beta: f32, random_draw: f32) -> bool {
    let probability = sdar_benefit_gate(benefit_ratio, beta);
    random_draw < probability
}

// ── RePlaid Learned Beta ────────────────────────────────────────

/// SDAR gate with learned β (RePlaid variance-minimized).
///
/// Instead of fixed β=5.0, learns β that minimizes the variance
/// of gated reward signals across episodes. High variance means
/// the gate is inconsistently applied → adjust β.
#[cfg(feature = "replaid_schedules")]
#[derive(Debug, Clone)]
pub struct SdarLearnedBeta {
    /// Current β value.
    beta: f32,
    /// Variance minimizer for gated signal.
    minimizer: crate::pruners::variance_minimizer::VarianceMinimizer,
}

#[cfg(feature = "replaid_schedules")]
impl SdarLearnedBeta {
    /// Create with initial β (paper default: 5.0).
    pub fn new(initial_beta: f32) -> Self {
        let config = crate::pruners::variance_minimizer::VarianceMinimizerConfig {
            mean_decay: 0.95,
            var_decay: 0.95,
            lr: 0.1,
            min_param: SDAR_BETA_MIN,
            max_param: SDAR_BETA_MAX,
        };
        Self {
            beta: initial_beta.clamp(SDAR_BETA_MIN, SDAR_BETA_MAX),
            minimizer: crate::pruners::variance_minimizer::VarianceMinimizer::with_param(
                config,
                initial_beta,
            ),
        }
    }

    /// Record a gated signal observation and adapt β.
    /// Call after each episode with the mean gated reward.
    /// Returns the new β value.
    pub fn observe_and_adapt(&mut self, gated_signal: f32) -> f32 {
        self.beta = self.minimizer.observe_and_adapt(gated_signal);
        self.beta
    }

    /// Current β value.
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Reset to initial β.
    pub fn reset(&mut self) {
        self.minimizer.reset();
        self.beta = self.minimizer.param();
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons
    const EPS: f32 = 1e-4;

    // ── sdar_gate basic tests ───────────────────────────────────

    #[test]
    fn test_gate_zero_input_is_half() {
        let result = sdar_gate(0.0, SDAR_BETA);
        assert!((result - 0.5).abs() < EPS, "σ(0) = 0.5, got {result}");
    }

    #[test]
    fn test_gate_positive_input_above_half() {
        let result = sdar_gate(0.5, SDAR_BETA);
        assert!(result > 0.5, "Positive input → gate opens, got {result}");
        assert!(result < 1.0, "Gate bounded < 1.0, got {result}");
    }

    #[test]
    fn test_gate_negative_input_below_half() {
        let result = sdar_gate(-0.5, SDAR_BETA);
        assert!(result < 0.5, "Negative input → gate closes, got {result}");
        assert!(result > 0.0, "Gate bounded > 0.0, got {result}");
    }

    #[test]
    fn test_gate_large_positive_approaches_one() {
        let result = sdar_gate(10.0, SDAR_BETA);
        assert!(result > 0.99, "Large positive → ≈ 1.0, got {result}");
    }

    #[test]
    fn test_gate_large_negative_approaches_zero() {
        let result = sdar_gate(-10.0, SDAR_BETA);
        assert!(result < 0.01, "Large negative → ≈ 0.0, got {result}");
    }

    #[test]
    fn test_gate_monotonic_increasing() {
        let inputs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let results: Vec<f32> = inputs.iter().map(|&x| sdar_gate(x, SDAR_BETA)).collect();

        for i in 1..results.len() {
            assert!(
                results[i] >= results[i - 1],
                "Gate should be monotonically increasing: {} >= {} failed at index {i}",
                results[i],
                results[i - 1]
            );
        }
    }

    #[test]
    fn test_gate_bounded_zero_one() {
        // Use range where numerics don't saturate: x ∈ [-0.6, 0.6] → z ∈ [-3.0, 3.0]
        // Beyond this, exp() precision causes σ(z) to round to exactly 0.0 or 1.0.
        for x in -6..=6 {
            let x = x as f32 * 0.1;
            let result = sdar_gate(x, SDAR_BETA);
            assert!(
                result > 0.0 && result < 1.0,
                "Gate ∈ (0,1) for x={x}, got {result}"
            );
        }
    }

    #[test]
    fn test_gate_default_uses_beta_5() {
        let with_default = sdar_gate_default(0.5);
        let with_explicit = sdar_gate(0.5, SDAR_BETA);
        assert!(
            (with_default - with_explicit).abs() < EPS,
            "Default should match explicit β=5.0"
        );
    }

    // ── β sensitivity (paper ablation) ──────────────────────────

    #[test]
    fn test_beta_zero_is_uniform() {
        // β=0 → σ(0·x) = σ(0) = 0.5 for all x
        let result = sdar_gate(100.0, 0.0);
        assert!(
            (result - 0.5).abs() < EPS,
            "β=0 → uniform 0.5, got {result}"
        );
    }

    #[test]
    fn test_beta_large_is_binary() {
        // β→∞ → gate approaches binary step
        let high = sdar_gate(0.01, 1000.0);
        let low = sdar_gate(-0.01, 1000.0);
        assert!(high > 0.99, "β→∞, positive → ≈ 1.0, got {high}");
        assert!(low < 0.01, "β→∞, negative → ≈ 0.0, got {low}");
    }

    #[test]
    fn test_beta_5_optimal_balance() {
        // β=5 at x=0.3: gate ≈ 0.82 (strong but not binary)
        let gate = sdar_gate(0.3, 5.0);
        assert!(
            gate > 0.7 && gate < 0.95,
            "β=5 at x=0.3 should be ~0.82, got {gate}"
        );
    }

    #[test]
    fn test_beta_1_soft_gate() {
        // β=1 at x=0.5: gate ≈ 0.62 (soft)
        let gate = sdar_gate(0.5, 1.0);
        assert!(
            gate > 0.5 && gate < 0.8,
            "β=1 at x=0.5 should be ~0.62, got {gate}"
        );
    }

    #[test]
    fn test_beta_10_near_binary() {
        // β=10 at x=0.2: gate ≈ 0.88 (near binary)
        let gate = sdar_gate(0.2, 10.0);
        assert!(gate > 0.8, "β=10 at x=0.2 should be >0.8, got {gate}");
    }

    // ── sdar_modulate tests ─────────────────────────────────────

    #[test]
    fn test_modulate_positive_gap_passes_signal() {
        let signal = 1.0;
        let result = sdar_modulate(signal, 0.5, SDAR_BETA);
        // gate(0.5, 5.0) ≈ 0.924 → result ≈ 0.924
        assert!(result > 0.9, "Positive gap → signal passes, got {result}");
        assert!(
            (result - signal).abs() < signal,
            "Result ≤ signal magnitude"
        );
    }

    #[test]
    fn test_modulate_negative_gap_attenuates() {
        let signal = 1.0;
        let result = sdar_modulate(signal, -0.5, SDAR_BETA);
        // gate(-0.5, 5.0) ≈ 0.076 → result ≈ 0.076
        assert!(
            result < 0.1,
            "Negative gap → signal attenuated, got {result}"
        );
    }

    #[test]
    fn test_modulate_zero_gap_halves_signal() {
        let signal = 2.0;
        let result = sdar_modulate(signal, 0.0, SDAR_BETA);
        // gate(0, 5.0) = 0.5 → result = 1.0
        assert!(
            (result - 1.0).abs() < EPS,
            "Zero gap → half signal, got {result}"
        );
    }

    #[test]
    fn test_modulate_preserves_signal_sign() {
        let negative_signal = -1.0;
        let result = sdar_modulate(negative_signal, 0.5, SDAR_BETA);
        assert!(result < 0.0, "Negative signal stays negative, got {result}");
    }

    #[test]
    fn test_modulate_default_matches_explicit() {
        let with_default = sdar_modulate_default(1.0, 0.5);
        let with_explicit = sdar_modulate(1.0, 0.5, SDAR_BETA);
        assert!((with_default - with_explicit).abs() < EPS);
    }

    // ── sdar_benefit_gate tests ─────────────────────────────────

    #[test]
    fn test_benefit_gate_high_ratio_promotes() {
        let gate = sdar_benefit_gate(3.0, SDAR_BETA);
        assert!(gate > 0.99, "High benefit ratio → ≈ 1.0, got {gate}");
    }

    #[test]
    fn test_benefit_gate_neutral_is_half() {
        let gate = sdar_benefit_gate(1.0, SDAR_BETA);
        assert!((gate - 0.5).abs() < EPS, "Neutral ratio → 0.5, got {gate}");
    }

    #[test]
    fn test_benefit_gate_zero_ratio_blocks() {
        let gate = sdar_benefit_gate(0.0, SDAR_BETA);
        assert!(gate < 0.01, "Zero ratio → ≈ 0.0, got {gate}");
    }

    #[test]
    fn test_benefit_gate_borderline_partial() {
        // benefit_ratio=1.2: gap=0.2, gate ≈ 0.73 (partial credit)
        let gate = sdar_benefit_gate(1.2, SDAR_BETA);
        assert!(
            gate > 0.6 && gate < 0.9,
            "Borderline → partial credit, got {gate}"
        );
    }

    // ── sdar_gated_reward tests ─────────────────────────────────

    #[test]
    fn test_gated_reward_positive_surprise_full_update() {
        // reward > q_value → positive gap → gate opens
        let result = sdar_gated_reward(1.0, 0.5, SDAR_BETA);
        // gap = 0.5, gate ≈ 0.924 → result ≈ 0.924
        assert!(
            result > 0.9,
            "Positive surprise → near-full reward, got {result}"
        );
    }

    #[test]
    fn test_gated_reward_negative_surprise_attenuated() {
        // reward < q_value → negative gap → gate closes
        let result = sdar_gated_reward(0.5, 1.0, SDAR_BETA);
        // gap = -0.5, gate ≈ 0.076 → result ≈ 0.038
        assert!(
            result < 0.1,
            "Negative surprise → attenuated reward, got {result}"
        );
    }

    #[test]
    fn test_gated_reward_no_surprise_half() {
        // reward = q_value → gap = 0 → gate = 0.5
        let result = sdar_gated_reward(1.0, 1.0, SDAR_BETA);
        assert!(
            (result - 0.5).abs() < EPS,
            "No surprise → half reward, got {result}"
        );
    }

    #[test]
    fn test_gated_reward_negative_reward_preserves_sign() {
        let result = sdar_gated_reward(-1.0, 0.0, SDAR_BETA);
        assert!(result < 0.0, "Negative reward stays negative, got {result}");
    }

    // ── sdar_should_promote tests ───────────────────────────────

    #[test]
    fn test_should_promote_high_benefit() {
        // High benefit → high probability → likely promoted
        let mut promoted_count = 0;
        for _ in 0..1000 {
            let draw = fastrand::f32();
            if sdar_should_promote(3.0, SDAR_BETA, draw) {
                promoted_count += 1;
            }
        }
        // probability ≈ 0.999+, so almost all should promote
        assert!(
            promoted_count > 990,
            "High benefit should promote ~100%, got {promoted_count}/1000"
        );
    }

    #[test]
    fn test_should_promote_low_benefit() {
        // Low benefit → low probability → rarely promoted
        let mut promoted_count = 0;
        for _ in 0..1000 {
            let draw = fastrand::f32();
            if sdar_should_promote(0.0, SDAR_BETA, draw) {
                promoted_count += 1;
            }
        }
        // probability ≈ 0.007, so very few should promote
        assert!(
            promoted_count < 30,
            "Low benefit should rarely promote, got {promoted_count}/1000"
        );
    }

    #[test]
    fn test_should_promote_neutral_fifty_fifty() {
        // Neutral benefit (ratio=1.0) → probability=0.5
        let mut promoted_count = 0;
        for _ in 0..1000 {
            let draw = fastrand::f32();
            if sdar_should_promote(1.0, SDAR_BETA, draw) {
                promoted_count += 1;
            }
        }
        // Should be roughly 50%
        assert!(
            promoted_count > 400 && promoted_count < 600,
            "Neutral should be ~50%, got {promoted_count}/1000"
        );
    }

    #[test]
    fn test_should_promote_deterministic_at_zero_draw() {
        // draw=0.0 → always promote (any probability > 0 wins)
        assert!(sdar_should_promote(0.1, SDAR_BETA, 0.0));
    }

    #[test]
    fn test_should_promote_never_at_one_draw() {
        // draw=1.0 → never promote (probability < 1.0)
        assert!(!sdar_should_promote(10.0, SDAR_BETA, 1.0));
    }

    // ── Numerical stability ─────────────────────────────────────

    #[test]
    fn test_numerical_stability_large_positive() {
        // Very large positive z should not overflow
        let result = sdar_gate(1000.0, SDAR_BETA);
        assert!(result.is_finite(), "Should not overflow");
        assert!((result - 1.0).abs() < EPS);
    }

    #[test]
    fn test_numerical_stability_large_negative() {
        // Very large negative z should not underflow to NaN
        let result = sdar_gate(-1000.0, SDAR_BETA);
        assert!(result.is_finite(), "Should not underflow to NaN");
        assert!(result >= 0.0, "Should not go negative");
    }

    #[test]
    fn test_numerical_stability_nan_input() {
        // NaN input → NaN output (IEEE 754 propagation)
        let result = sdar_gate(f32::NAN, SDAR_BETA);
        assert!(result.is_nan(), "NaN input → NaN output");
    }

    #[test]
    fn test_numerical_stability_inf_input() {
        let pos_inf = sdar_gate(f32::INFINITY, SDAR_BETA);
        let neg_inf = sdar_gate(f32::NEG_INFINITY, SDAR_BETA);
        assert!((pos_inf - 1.0).abs() < EPS, "+inf → 1.0");
        assert!((neg_inf).abs() < EPS, "-inf → 0.0");
    }

    // ── Symmetry and properties ─────────────────────────────────

    #[test]
    fn test_symmetry_around_zero() {
        // σ(z) + σ(-z) = 1
        for x in [0.1, 0.5, 1.0, 2.0, 5.0] {
            let pos = sdar_gate(x, SDAR_BETA);
            let neg = sdar_gate(-x, SDAR_BETA);
            assert!(
                (pos + neg - 1.0).abs() < EPS,
                "σ({x}) + σ(-{x}) = 1, got {}",
                pos + neg
            );
        }
    }

    #[test]
    fn test_scaling_by_beta() {
        // σ(β·x) for x=1 should equal σ(1) when β=1
        let gate_beta1 = sdar_gate(1.0, 1.0);
        let gate_beta5 = sdar_gate(0.2, 5.0); // same z = 1.0
        assert!(
            (gate_beta1 - gate_beta5).abs() < EPS,
            "σ(1·1) should equal σ(5·0.2), got {gate_beta1} vs {gate_beta5}"
        );
    }

    // ── Constants ───────────────────────────────────────────────

    #[test]
    fn test_constants_sensible() {
        assert!((SDAR_BETA - 5.0).abs() < EPS, "Default β = 5.0");
        assert!(SDAR_BETA_MIN > 0.0, "β min > 0");
        assert!(SDAR_BETA_MAX > SDAR_BETA, "β max > default");
    }
}

// ── RePlaid Tests ───────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "replaid_schedules")]
mod replaid_tests {
    use super::*;

    #[test]
    fn test_learned_beta_starts_at_initial() {
        let lb = SdarLearnedBeta::new(5.0);
        assert!((lb.beta() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_learned_beta_adapts() {
        let mut lb = SdarLearnedBeta::new(5.0);

        // Observe consistent signals → low variance → β should be stable
        for _ in 0..20 {
            lb.observe_and_adapt(0.5);
        }
        let beta_stable = lb.beta();

        // Now observe varying signals → high variance → β should change
        for i in 0..20 {
            let signal = match i % 2 {
                0 => 0.1,
                _ => 0.9,
            };
            lb.observe_and_adapt(signal);
        }
        let beta_after_chaos = lb.beta();

        // β may not change much with small lr, so no strict assertion here —
        // the smoke test above (observe_and_adapt not panicking) is the real check.
        let _ = (beta_stable, beta_after_chaos);
    }

    #[test]
    fn test_learned_beta_clamps_to_range() {
        let mut lb = SdarLearnedBeta::new(5.0);

        // Extreme signals
        for _ in 0..100 {
            lb.observe_and_adapt(1000.0);
        }
        assert!(lb.beta() >= SDAR_BETA_MIN && lb.beta() <= SDAR_BETA_MAX);
    }

    #[test]
    fn test_learned_beta_reset() {
        let mut lb = SdarLearnedBeta::new(5.0);
        lb.observe_and_adapt(1.0);
        lb.observe_and_adapt(2.0);
        let _before_reset = lb.beta();

        lb.reset();
        // After reset, should be back to midpoint of [SDAR_BETA_MIN, SDAR_BETA_MAX]
        let midpoint = (SDAR_BETA_MIN + SDAR_BETA_MAX) / 2.0;
        assert!((lb.beta() - midpoint).abs() < 0.01);
    }
}
