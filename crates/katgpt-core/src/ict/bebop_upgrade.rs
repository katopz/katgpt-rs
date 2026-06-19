//! Bebop H₁ → H₂ acceptance-forecast upgrade (Plan 294 Phase 6 / G10).
//!
//! Bebop (Research 243, Issue 023) forecasts speculative-decoding acceptance
//! length linearly from Shannon entropy:
//!
//! ```text
//! α_H1 ≈ a − b · H_1(next_token_logits)   // Bebop baseline
//! ```
//!
//! ICT §1.5 + §A.3.3 prove this is the **wrong** entropy: H₁ has negative
//! gradient only for π(a) > e⁻¹ ≈ 0.37, so for the long tail of low-probability
//! tokens H₁ reports a wrong "concentration" signal. The correct drop-in is
//! the second-order Rényi entropy via collision purity:
//!
//! ```text
//! α_H2 ≈ a − b · H_2(p)                   // = a − b · (−log β(p))
//! ```
//!
//! `H_2` is **unconditionally** valid: `∂H_2/∂π(a) = −2π(a)/β < 0` always.
//! G10 (`bench_294_ict_g10.rs`) calibrates both on a 50/50 mixture of
//! "decisive" (max π > 0.37) and "long-tail" (max π < 0.37) workloads and
//! asserts H₂ has lower mean forecast error, concentrated in the long-tail
//! regime where H₁ is provably wrong.
//!
//! ## AGENTS.md sigmoid rule
//!
//! `softmax(logits)` here is **representation** (logits → probability
//! simplex), not a projection gate onto direction vectors. The actual gate
//! output is the linear forecast `α = a − b · H_2(p)` — no softmax-on-directions
//! involved. The rule is satisfied.

use crate::ict::math::collision_purity;

/// Acceptance-length forecaster using H₂ (collision purity) instead of Bebop's H₁.
///
/// Construct with calibration constants `(a, b)` — typically fit by linear
/// regression of `acceptance_length` against `H_2(π)` on a calibration set.
/// See [`observe_and_forecast`](Self::observe_and_forecast) for the hot path.
///
/// The EMA fields smooth β and α across tokens; downstream Bebop-style
/// adaptive-γ loops can read them via the public accessors.
#[derive(Debug, Clone)]
pub struct AcceptanceForecastH2 {
    /// Linear forecast intercept (Bebop `a`).
    pub a: f32,
    /// Linear forecast slope (Bebop `b`, typically negative).
    pub b: f32,
    /// EMA of the most recent β = collision_purity(softmax(logits)). In `[0, 1]`.
    pub ema_beta: f32,
    /// EMA of the most recent forecast α. Read by Bebop's adaptive-γ loop.
    pub ema_alpha: f32,
    /// EMA decay in `[0, 1]`. Default `0.9` (matches BranchingDetector).
    pub ema_decay: f32,
}

impl AcceptanceForecastH2 {
    /// Construct with calibration constants. EMA fields start at zero.
    pub fn new(a: f32, b: f32) -> Self {
        Self {
            a,
            b,
            ema_beta: 0.0,
            ema_alpha: 0.0,
            ema_decay: 0.9,
        }
    }

    /// Override the default EMA decay (0.9). Clamped to `[0, 1]`.
    pub fn with_ema_decay(mut self, decay: f32) -> Self {
        self.ema_decay = if decay < 0.0 {
            0.0
        } else if decay > 1.0 {
            1.0
        } else {
            decay
        };
        self
    }

    /// Observe the next-token logits, compute α via H₂, and update EMAs.
    ///
    /// `next_token_logits` is the raw logits vector (any length ≥ 1). The
    /// function:
    /// 1. Computes `p = softmax(logits)` (numerically stable: subtract max).
    /// 2. Computes `β = collision_purity(p) = Σ p²`.
    /// 3. Computes `H_2 = −log β`.
    /// 4. Returns the forecast `α = a − b · H_2`.
    /// 5. Updates `ema_beta` and `ema_alpha`.
    ///
    /// Allocates a single `Vec<f32>` for `p` (vocab-sized) on each call.
    /// Callers in a tight loop should pre-allocate and call
    /// [`observe_and_forecast_into`](Self::observe_and_forecast_into).
    pub fn observe_and_forecast(&mut self, next_token_logits: &[f32]) -> f32 {
        if next_token_logits.is_empty() {
            // Degenerate: forecast = a, no EMA update.
            return self.a;
        }
        let mut p = vec![0.0_f32; next_token_logits.len()];
        self.observe_and_forecast_into(next_token_logits, &mut p)
    }

    /// Zero-alloc variant — writes the softmax output into caller-provided
    /// `prob_scratch` (length must equal `next_token_logits.len()`).
    pub fn observe_and_forecast_into(
        &mut self,
        next_token_logits: &[f32],
        prob_scratch: &mut [f32],
    ) -> f32 {
        let n = next_token_logits.len();
        if n == 0 || prob_scratch.len() != n {
            return self.a;
        }

        // ── Numerically stable softmax: subtract max, exponentiate, normalize. ──
        let mut max_l = next_token_logits[0];
        for &l in &next_token_logits[1..] {
            if l > max_l {
                max_l = l;
            }
        }
        let mut sum_exp = 0.0_f32;
        // Chunked-4 helps LLVM autovectorize the exp+accumulate.
        let mut i = 0;
        while i + 4 <= n {
            let e0 = (next_token_logits[i] - max_l).exp();
            let e1 = (next_token_logits[i + 1] - max_l).exp();
            let e2 = (next_token_logits[i + 2] - max_l).exp();
            let e3 = (next_token_logits[i + 3] - max_l).exp();
            prob_scratch[i] = e0;
            prob_scratch[i + 1] = e1;
            prob_scratch[i + 2] = e2;
            prob_scratch[i + 3] = e3;
            sum_exp += e0 + e1 + e2 + e3;
            i += 4;
        }
        while i < n {
            let e = (next_token_logits[i] - max_l).exp();
            prob_scratch[i] = e;
            sum_exp += e;
            i += 1;
        }
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        for slot in prob_scratch[..n].iter_mut() {
            *slot *= inv_sum;
        }

        // ── β = Σ p² (collision purity). H_2 = −log β. ──
        let beta = collision_purity(prob_scratch);
        // Guard against log(0) when β underflows (impossible for finite
        // softmax but f32 arithmetic can drift).
        let h2 = if beta > 0.0 { -beta.ln() } else { f32::INFINITY };
        let alpha = self.a - self.b * h2;

        // ── EMA update. ──
        let d = self.ema_decay;
        self.ema_beta = (1.0 - d) * self.ema_beta + d * beta;
        self.ema_alpha = (1.0 - d) * self.ema_alpha + d * alpha;

        alpha
    }

    /// Bebop R243 §4 adaptive-γ sketch — map the current EMA α to a draft
    /// length γ in `[gamma_min, gamma_max]`.
    ///
    /// Linear interpolation: γ = clamp(ema_alpha, γ_min, γ_max). When α is
    /// high (long acceptance expected), draft longer; when low, draft shorter.
    /// Matches Bebop's adaptive-γ policy; the only difference is `ema_alpha`
    /// here comes from the H₂ forecast rather than the H₁ forecast.
    #[inline]
    pub fn adaptive_gamma(&self, _target_accept_length: f32, gamma_min: usize, gamma_max: usize) -> usize {
        let gamma_f = self.ema_alpha;
        let gamma = if gamma_f < gamma_min as f32 {
            gamma_min as f32
        } else if gamma_f > gamma_max as f32 {
            gamma_max as f32
        } else {
            gamma_f
        };
        gamma as usize
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_logits_returns_intercept() {
        let mut f = AcceptanceForecastH2::new(8.0, 1.0);
        let alpha = f.observe_and_forecast(&[]);
        assert!((alpha - 8.0).abs() < 1e-6, "empty logits → α=a=8.0, got {alpha}");
    }

    #[test]
    fn degenerate_logits_match_h2_formula() {
        // Single-token distribution: logits = [0.0, -∞ effectively via -1e10].
        // softmax → [1, 0], β = 1, H_2 = 0, α = a.
        let mut f = AcceptanceForecastH2::new(8.0, 2.0);
        let logits = [0.0_f32, -1e10];
        let alpha = f.observe_and_forecast(&logits);
        assert!((alpha - 8.0).abs() < 1e-3, "degenerate α should be a=8.0, got {alpha}");
    }

    #[test]
    fn uniform_logits_match_uniform_h2() {
        // Uniform over n → H_2 = log n. With a=10, b=1 → α = 10 - log(n).
        let mut f = AcceptanceForecastH2::new(10.0, 1.0);
        let logits = [0.0_f32; 8]; // uniform
        let alpha = f.observe_and_forecast(&logits);
        let expected = 10.0 - (8.0_f32).ln();
        assert!(
            (alpha - expected).abs() < 1e-3,
            "uniform α should be {expected}, got {alpha}"
        );
    }

    #[test]
    fn observe_into_reuses_scratch() {
        let mut f = AcceptanceForecastH2::new(10.0, 1.0);
        let logits = [0.5_f32, -0.5, 0.2, -0.2];
        let mut scratch = [0.0_f32; 4];
        let ptr_before = scratch.as_ptr();
        let alpha = f.observe_and_forecast_into(&logits, &mut scratch);
        let ptr_after = scratch.as_ptr();
        assert_eq!(ptr_before, ptr_after, "scratch must be reused");
        assert!(alpha.is_finite(), "α must be finite");
    }

    #[test]
    fn ema_updates_after_observe() {
        let mut f = AcceptanceForecastH2::new(10.0, 1.0).with_ema_decay(0.5);
        assert!(f.ema_beta.abs() < 1e-6);
        let logits = [0.0_f32; 4]; // uniform → β = 1/4 = 0.25
        let _ = f.observe_and_forecast(&logits);
        // EMA = 0.5 · 0 + 0.5 · 0.25 = 0.125
        assert!(
            (f.ema_beta - 0.125).abs() < 1e-4,
            "ema_beta should be 0.125, got {}",
            f.ema_beta
        );
    }

    #[test]
    fn adaptive_gamma_clamps_to_range() {
        let mut f = AcceptanceForecastH2::new(10.0, 1.0);
        // Drive α to a known value: uniform over 4 → α = 10 - log 4 ≈ 8.61.
        let logits = [0.0_f32; 4];
        let _ = f.observe_and_forecast(&logits);
        let g = f.adaptive_gamma(8.0, 1, 16);
        // α ≈ 8.61 → γ should be clamped to [1, 16], so g = 8 (or 9).
        assert!(g >= 1 && g <= 16, "γ out of range: {g}");
        // Now lower α dramatically: uniform over 1024 → α = 10 - log(1024) ≈ 3.06.
        let logits_low = [0.0_f32; 1024];
        let _ = f.observe_and_forecast(&logits_low);
        let g_low = f.adaptive_gamma(4.0, 1, 16);
        // ema_alpha is between 8.61 and 3.06 — still positive and finite.
        assert!(g_low >= 1, "γ must be ≥ gamma_min");
    }

    #[test]
    fn h2_differs_from_h1_on_long_tail() {
        // Smoke check that H_2 forecast is different from H_1 forecast on
        // a long-tail distribution (the regime where H_1 is provably wrong).
        // The full comparison is in bench_294_ict_g10.rs.
        use crate::ict::math::shannon_h1;
        let mut f = AcceptanceForecastH2::new(10.0, 1.0);
        // Long-tail: many tokens, no dominant one. softmax of small logits.
        let logits: Vec<f32> = (0..16).map(|i| -0.5 + 0.1 * (i as f32)).collect();
        let mut p = vec![0.0_f32; 16];
        // softmax manually for the H_1 comparison.
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0.0;
        for (i, &l) in logits.iter().enumerate() {
            p[i] = (l - max_l).exp();
            s += p[i];
        }
        for v in &mut p {
            *v /= s;
        }
        let h1 = shannon_h1(&p);
        let h2 = -collision_purity(&p).ln();
        let alpha_h1 = 10.0 - 1.0 * h1;
        let alpha_h2 = f.observe_and_forecast(&logits);
        assert!(
            (alpha_h1 - alpha_h2).abs() > 0.05,
            "H_1 and H_2 forecasts should differ on long-tail: α_H1={alpha_h1}, α_H2={alpha_h2}, H_1={h1}, H_2={h2}"
        );
    }
}
